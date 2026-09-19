"""玩家状态读取：位置 + 朝向（模板匹配）+ HP/MP（UI 条颜色占比）。

玩家定位走模板匹配（不走 YOLO 训练），复用 perception/player_locator.py。
HP/MP 读屏幕 UI 条：在条区域里数「条颜色」像素占比，不用 OCR。

定位策略（按用户「初始化 → 追踪」的思路）：
    1. 未定位：冷却重试 —— 每 retry_every 帧才真跑一次匹配（中央/全图），
       其余帧直接跳过，避免「每帧都慢匹配」把帧率拖死。
    2. 已定位：动态 ROI 追踪 —— 每 interval 帧在上一帧位置附近小范围匹配，
       快且跟随移动。
    3. 丢失：回到冷却重试，不硬跑全图。

    另有 set_position(x, y) 供 GUI 手动指定初始位置（用户手动调整后调用）。

UI 条配置（ui 参数）结构：
    ui = {
        "hp": {"x":.., "y":.., "w":.., "h":.., "low":[B,G,R], "high":[B,G,R]},
        "mp": {...},
    }
坐标是「流画面坐标」。low/high 是条颜色的 BGR 上下界。
没配（None）就跳过，hp/mp 保持默认 1.0。
"""
import cv2
import numpy as np

from perception.player_locator import PlayerLocator
from perception.world_state import Player


class PlayerStateReader:
    def __init__(self, player_id, scale=1.0, threshold=0.78, ui=None,
                 filter_prefix="stand", search_frac=0.4, interval=5,
                 roi_margin=80, retry_every=15):
        self.locator = PlayerLocator(player_id, scale=scale, threshold=threshold,
                                    filter_prefix=filter_prefix)
        self.ui = ui or {}
        self.search_frac = search_frac          # 中央兜底区域占画面比例
        self.interval = max(1, int(interval))   # 追踪中每 N 帧真正匹配一次
        self.roi_margin = int(roi_margin)       # 动态 ROI 的边距（px）
        self.retry_every = max(1, int(retry_every))  # 丢失后每 N 帧重试一次
        self._frame = 0
        self._last = None                       # (cx, cy, facing) 或 None
        self._full_done = False                 # 全图兜底只做一次
        self._miss_frames = 0

    def set_position(self, x, y, facing=1):
        """手动指定玩家位置（GUI 用户调整后调用），之后从这里开始追踪。"""
        self._last = (float(x), float(y), int(facing))

    def read(self, bgr):
        """从一帧 BGR 读取玩家状态，返回 Player。"""
        player = Player()
        self._frame += 1
        h, w = bgr.shape[:2]

        # 决定这一帧要不要真正跑匹配
        if self._last is not None:
            # 追踪中：每 interval 帧匹配一次，其余帧复用
            do_locate = (self._frame % self.interval == 0)
        else:
            # 未定位：冷却重试，每 retry_every 帧才跑一次，避免每帧慢匹配拖死帧率
            self._miss_frames += 1
            do_locate = (self._miss_frames >= self.retry_every)
            if do_locate:
                self._miss_frames = 0

        if do_locate:
            r = self._locate(bgr, h, w)
            if r:
                cx, cy, tw, th, score, name = r
                # 朝向：模板名带 @L 后缀 = 镜像模板 = 朝左
                facing = -1 if name.endswith("@L") else 1
                self._last = (cx, cy, facing)
            else:
                self._last = None

        if self._last is not None:
            player.x, player.y, player.facing = self._last
            player.found = True

        hp = self._read_bar(bgr, self.ui.get("hp"))
        mp = self._read_bar(bgr, self.ui.get("mp"))
        if hp is not None:
            player.hp = hp
        if mp is not None:
            player.mp = mp
        return player

    def _locate(self, bgr, h, w):
        """三级搜索：动态 ROI → 中央兜底 → 全图（只一次）。"""
        # 1) 动态 ROI：上一帧位置附近
        if self._last is not None:
            cx, cy, _f = self._last
            m = self.roi_margin
            sx = max(0, int(cx - m))
            sy = max(0, int(cy - m))
            sw = min(w - sx, m * 2)
            sh = min(h - sy, m * 2)
            r = self.locator.locate(bgr, search_rect=(sx, sy, sw, sh))
            if r is not None:
                return r

        # 2) 中央区域兜底
        sw, sh = int(w * self.search_frac), int(h * self.search_frac)
        sx, sy = (w - sw) // 2, (h - sh) // 2
        r = self.locator.locate(bgr, search_rect=(sx, sy, sw, sh))
        if r is not None:
            return r

        # 3) 全图兜底只做一次（彻底定位不到时）；之后 miss 就返回 None
        if not self._full_done:
            self._full_done = True
            return self.locator.locate(bgr)
        return None

    @staticmethod
    def _read_bar(bgr, cfg):
        """按配置读一条 UI 条，返回 0~1 填充比例；无配置返回 None。"""
        if not cfg:
            return None
        x, y, w, h = (int(cfg[k]) for k in ("x", "y", "w", "h"))
        region = bgr[y:y + h, x:x + w]
        if region.size == 0:
            return None
        low = np.array(cfg.get("low", [0, 0, 0]), dtype=np.uint8)
        high = np.array(cfg.get("high", [255, 255, 255]), dtype=np.uint8)
        mask = cv2.inRange(region, low, high)
        return float(np.count_nonzero(mask)) / mask.size
