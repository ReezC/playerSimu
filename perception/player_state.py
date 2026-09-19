"""玩家状态读取：位置 + 朝向（模板匹配）+ HP/MP（UI 条颜色占比）。

玩家定位走模板匹配（不走 YOLO 训练），复用 perception/player_locator.py。
HP/MP 读屏幕 UI 条：在条区域里数「条颜色」像素占比，不用 OCR。

性能要点（都是实测踩出来的）：
    - 全图 64 模板匹配一次要 8.5s，没法实时。限定中央区域（search_frac）+ 只
      加载站姿模板（filter_prefix="stand"）后降到几十 ms。
    - 玩家移动慢，每 interval 帧才真正匹配一次，其余帧复用上次结果。

UI 条配置（ui 参数）结构：
    ui = {
        "hp": {"x":.., "y":.., "w":.., "h":.., "low":[B,G,R], "high":[B,G,R]},
        "mp": {...},
    }
坐标是「流画面坐标」（1366x768）。low/high 是条颜色的 BGR 上下界。
没配（None）就跳过，hp/mp 保持默认 1.0 —— 标定前先跑通，标定后补上。
"""
import cv2
import numpy as np

from perception.player_locator import PlayerLocator
from perception.world_state import Player


class PlayerStateReader:
    def __init__(self, player_id, scale=1.0, threshold=0.78, ui=None,
                 filter_prefix="stand", search_frac=0.4, interval=1):
        self.locator = PlayerLocator(player_id, scale=scale, threshold=threshold,
                                    filter_prefix=filter_prefix)
        self.ui = ui or {}
        self.search_frac = search_frac      # 搜索区域占画面比例（中央）
        self.interval = max(1, int(interval))  # 每 N 帧真正匹配一次
        self._frame = 0
        self._last = None                   # (cx, cy, facing) 或 None

    def read(self, bgr):
        """从一帧 BGR 读取玩家状态，返回 Player。"""
        player = Player()
        self._frame += 1

        # 低频匹配：每 interval 帧真正跑一次模板匹配，其余帧复用上次结果。
        if self._last is None or self._frame % self.interval == 0:
            h, w = bgr.shape[:2]
            sw, sh = int(w * self.search_frac), int(h * self.search_frac)
            sx, sy = (w - sw) // 2, (h - sh) // 2
            r = self.locator.locate(bgr, search_rect=(sx, sy, sw, sh))
            if r:
                cx, cy, tw, th, score, name = r
                # 朝向：模板名带 @L 后缀 = 镜像模板 = 朝左（见 PlayerLocator._load）
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

    @staticmethod
    def _read_bar(bgr, cfg):
        """按配置读一条 UI 条，返回 0~1 填充比例；无配置返回 None。

        cfg: {x, y, w, h, low=[B,G,R], high=[B,G,R]} —— 条区域 + 条颜色范围。
        """
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
