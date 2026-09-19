"""玩家定位：模板匹配（每个玩家一套 WZ 合成外观模板）。

架构定位（重要）：
    怪物走 YOLO 训练（地图数据通用），玩家**不走训练** ——
    玩家外观因人而异，训进模型就把模型绑死在某个账号上。

    正确做法是模板匹配：每个玩家导出一套外观模板，放在
        datasets/sprites/player/{角色id}/*.png
    换玩家 = 换模板目录，零训练。这满足「一套地图数据支持不同玩家」。

    模板来自 WZ 合成（用户导出），PNG 带 alpha 透明通道，匹配时 alpha 做掩码。

模板目录结构（用户约定）：
    player/{角色id}/{角色id}-{动作}-{帧号}.png
    例：player/珠缨/珠缨-stand1-0.png
"""
from pathlib import Path

import cv2
import numpy as np

from core.imgio import imread

DEFAULT_ROOT = Path("datasets/sprites/player")


class PlayerLocator:
    """用一套玩家外观模板，在画面里定位玩家自己。

    scale：画面里角色尺寸 / WZ 素材尺寸。随采集/推流分辨率变 ——
        1920x1080 训练帧实测 ≈ 1.25；1366x768 实时流要重标（约 0.89）。
    threshold：TM_CCORR_NORMED 匹配分阈值（实测正确命中 0.81+）。
    """

    def __init__(self, player_id, root=None, scale=1.25, threshold=0.78,
                 filter_prefix=None):
        self.player_id = player_id
        self.scale = float(scale)
        self.threshold = float(threshold)
        self.filter_prefix = filter_prefix   # 只加载文件名含此子串的模板（如 "stand"）
        self._templates = []       # [(name, bgr, mask), ...]
        self._load(root or DEFAULT_ROOT)

    # ---------------- 模板加载 ----------------

    def _load(self, root):
        d = Path(root) / self.player_id
        if not d.is_dir():
            raise FileNotFoundError("找不到角色模板目录: %s\n"
                                    "请把 WZ 合成的角色动作帧放到这里，"
                                    "结构为 {角色id}/{角色id}-{动作}-{帧号}.png" % d)
        for f in sorted(d.glob("*.png")):
            if self.filter_prefix:
                # 支持逗号分隔的多个动作前缀，如 "stand,walk"（匹配含 stand 或 walk 的）
                prefixes = [p.strip() for p in self.filter_prefix.split(",") if p.strip()]
                if prefixes and not any(p in f.stem for p in prefixes):
                    continue
            t = self._load_tpl(f)
            if t:
                b, m = t
                self._templates.append((f.stem, b, m))
                # 镜像出朝左的模板，运行时左右都能匹配
                self._templates.append((f.stem + "@L", cv2.flip(b, 1), cv2.flip(m, 1)))

        if not self._templates:
            raise FileNotFoundError("角色模板目录里没有可用 PNG: %s" % d)

    @staticmethod
    def _load_tpl(path):
        """载入单张模板：裁掉透明边框，返回 (bgr, alpha掩码)。"""
        im = imread(path, cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            return None
        b, al = im[:, :, :3].copy(), im[:, :, 3]
        ys, xs = np.nonzero(al > 128)
        if len(xs) < 100:
            return None
        b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return b, ((al > 128) * 255).astype(np.uint8)

    # ---------------- 定位 ----------------

    def locate(self, bgr, search_rect=None):
        """在一帧 BGR 图里定位玩家。

        search_rect: (x, y, w, h) 限定搜索区域（画面坐标），None = 全图。
            横版 ARPG 玩家在画面中央附近，限定区域能把耗时降一个量级。
        返回 (cx, cy, w, h, score, name) —— cx/cy 是框中心（**画面坐标**，
        即使限定区域也会加回偏移），w/h 是框尺寸；找不到返回 None。
        """
        h, w = bgr.shape[:2]
        if search_rect is not None:
            sx, sy, sw, sh = search_rect
            sx = max(0, min(w - 1, int(sx)))
            sy = max(0, min(h - 1, int(sy)))
            sw = max(1, min(w - sx, int(sw)))
            sh = max(1, min(h - sy, int(sh)))
            canvas = bgr[sy:sy + sh, sx:sx + sw]
        else:
            sx = sy = 0
            sw, sh = w, h
            canvas = bgr

        best = None
        for name, tb, tm in self._templates:
            tw = max(8, int(round(tb.shape[1] * self.scale)))
            th = max(8, int(round(tb.shape[0] * self.scale)))
            if tw >= sw or th >= sh:
                continue
            tbb = cv2.resize(tb, (tw, th), interpolation=cv2.INTER_AREA)
            tmm = cv2.resize(tm, (tw, th), interpolation=cv2.INTER_NEAREST)
            try:
                r = cv2.matchTemplate(canvas, tbb, cv2.TM_CCORR_NORMED, mask=tmm)
            except Exception:
                continue
            r = np.nan_to_num(r)
            _, mx, _, ml = cv2.minMaxLoc(r)
            if best is None or mx > best[0]:
                best = (float(mx), ml[0], ml[1], tw, th, name)

        if best is None or best[0] < self.threshold:
            return None
        score, x, y, tw, th, name = best
        return x + sx + tw / 2.0, y + sy + th / 2.0, tw, th, score, name

    @property
    def template_count(self):
        return len(self._templates)
