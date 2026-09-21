"""平台感知：从 BGR 帧提取可站立顶边，并维护跨帧 ID 与玩家运动状态。

这里刻意只输出 ``x1/x2/y``。树叶等绿色物体即使偶尔进入候选，也必须是长而
近水平的连通带才会留下；后续决策只把这些线当作碰撞面，不依赖平台贴图高度。
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from perception.world_state import JumpPrediction, Platform


class PlatformDetector:
    """面向固定横版游戏画面的轻量 HSV + 水平形态学检测器。"""

    def __init__(self, min_width=70, roi_top=0.04, roi_bottom=0.88):
        self.min_width = int(min_width)
        self.roi_top = float(roi_top)
        self.roi_bottom = float(roi_bottom)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        top, bottom = int(h * self.roi_top), int(h * self.roi_bottom)
        if bottom <= top:
            return []
        roi = bgr[top:bottom]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 草坪色域；饱和度下限会排掉灰色背景和白色 UI。
        mask = cv2.inRange(hsv, (25, 45, 35), (100, 255, 255))
        # 不能直接用绿色轮廓的矩形：树冠会和草坪一样绿，还可能横向很长。
        # 改为逐行找「草的顶边」，再要求其正下方有连续棕/暗色土层。只有这两个
        # 条件同时成立才是可站立平台，树叶、UI 和怪物均不会通过土层验证。
        out = []
        # 土层在此游戏画面中既偏黄棕，也明显比天空/树冠暗；V 上限不能放到
        # 亮绿树冠，否则它下面的阴影会重新把整片背景放行。
        soil = cv2.inRange(hsv, (0, 40, 20), (45, 255, 150)) > 0
        join = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
        for y in range(0, max(0, mask.shape[0] - 30)):
            # 草边有锯齿/小缺口，因此在相邻三行取并集，再只横向补小缝。
            row = np.any(mask[y:y + 3] > 0, axis=0).astype(np.uint8) * 255
            row = cv2.morphologyEx(row.reshape(1, -1), cv2.MORPH_CLOSE, join)[0] > 0
            edges = np.diff(np.r_[False, row, False].astype(np.int8))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            for x1, x2 in zip(starts, ends):
                if x2 - x1 < self.min_width:
                    continue
                grass = mask[y:y + 4, x1:x2] > 0
                dirt = soil[y + 5:y + 30, x1:x2]
                grass_density = float(grass.mean()) if grass.size else 0.0
                dirt_density = float(dirt.mean()) if dirt.size else 0.0
                # 平台草皮在横向上是连续的；背景草丛虽可能同色，但在一行中
                # 会留下大量空隙。较高的覆盖率可排掉这种“拼起来的长线”。
                if grass_density >= 0.55 and dirt_density >= 0.35:
                    out.append((float(x1), float(x2), float(y + top),
                                min(1.0, 0.5 * grass_density + 0.5 * dirt_density)))
        # 同一草皮会在相邻数行各产出一次，也会被角色/道具切成数段；先在
        # 同一高度合并相邻段，再去掉被更长候选覆盖的短段。
        out.sort(key=lambda p: (p[2], p[0]))
        merged = []
        for x1, x2, y, conf in out:
            joined = False
            for i, (ox1, ox2, oy, oc) in enumerate(merged):
                if abs(y - oy) <= 5 and x1 <= ox2 + 30 and x2 >= ox1 - 30:
                    merged[i] = (min(x1, ox1), max(x2, ox2), min(y, oy), max(conf, oc))
                    joined = True
                    break
            if not joined:
                merged.append((x1, x2, y, conf))
        kept = []
        for cand in sorted(merged, key=lambda p: p[1] - p[0], reverse=True):
            x1, x2, y, _conf = cand
            covered = False
            for ox1, ox2, oy, _oc in kept:
                overlap = max(0.0, min(x2, ox2) - max(x1, ox1))
                if abs(y - oy) <= 28 and overlap / max(1.0, x2 - x1) >= 0.70:
                    covered = True
                    break
            if not covered:
                kept.append(cand)
        return sorted(kept, key=lambda p: (p[2], p[0]))


class PlatformTracker:
    """用位置/横向重叠关联平台；短暂失检时保留上一帧，减少 ID 抖动。"""

    def __init__(self, detector=None, max_age_s=0.5):
        self.detector = detector or PlatformDetector()
        self.max_age_s = float(max_age_s)
        self._tracks = []
        self._next_id = 1

    def update(self, bgr, now=None):
        now = time.monotonic() if now is None else float(now)
        detections = self.detector.detect(bgr)
        used = set()
        fresh = []
        for track in self._tracks:
            best, best_cost = None, float("inf")
            for i, d in enumerate(detections):
                if i in used:
                    continue
                x1, x2, y, conf = d
                overlap = max(0.0, min(track["x2"], x2) - max(track["x1"], x1))
                if overlap <= 0 and abs((x1 + x2) / 2 - (track["x1"] + track["x2"]) / 2) > 50:
                    continue
                cost = abs(y - track["y"]) + 0.08 * abs(x1 - track["x1"])
                if cost < best_cost and cost < 55:
                    best, best_cost = i, cost
            if best is not None:
                used.add(best)
                x1, x2, y, conf = detections[best]
                # 指数平滑既不延迟太多，也抑制草边缘 1~2px 的闪烁。
                a = 0.72
                track.update(x1=a*x1+(1-a)*track["x1"], x2=a*x2+(1-a)*track["x2"],
                             y=a*y+(1-a)*track["y"], conf=conf, age=track["age"] + 1,
                             seen=now)
            if now - track["seen"] <= self.max_age_s:
                fresh.append(track)
        for i, (x1, x2, y, conf) in enumerate(detections):
            if i not in used:
                fresh.append({"id": self._next_id, "x1": x1, "x2": x2, "y": y,
                              "conf": conf, "age": 1, "seen": now})
                self._next_id += 1
        self._tracks = fresh
        return [Platform(t["id"], t["x1"], t["x2"], t["y"], t["conf"], t["age"])
                for t in self._tracks]


class PlayerMotionTracker:
    """从玩家框的连续观测补齐速度、腾空状态、当前平台和本次下落落点。"""

    def __init__(self, ground_tolerance=12):
        self.ground_tolerance = float(ground_tolerance)
        self._last = None

    def update(self, player, platforms, ts=None):
        ts = time.monotonic() if ts is None else float(ts)
        if not player.found:
            self._last = None
            return None
        if self._last is not None:
            old_x, old_bottom, old_ts, old_vx, old_vy = self._last
            dt = ts - old_ts
            if 0.001 <= dt <= 0.5:
                # 轻微平滑，避免单帧 YOLO 框抖动被误判成起跳。
                player.vx = 0.65 * ((player.x - old_x) / dt) + 0.35 * old_vx
                player.vy = 0.65 * ((player.bottom - old_bottom) / dt) + 0.35 * old_vy
        current = None
        for p in platforms:
            if p.x1 - player.w / 2 <= player.x <= p.x2 + player.w / 2 and abs(player.bottom - p.y) <= self.ground_tolerance:
                if current is None or abs(player.bottom - p.y) < abs(player.bottom - current.y):
                    current = p
        player.current_platform_id = current.id if current else None
        player.grounded = current is not None and abs(player.vy) < 90
        player.jumping = not player.grounded and player.vy < -25
        player.falling = not player.grounded and player.vy > 25
        self._last = (player.x, player.bottom, ts, player.vx, player.vy)
        if not player.falling:
            return None
        # 在已知下落速度下，预测与下一条位于脚下的平台顶边相交的时刻。
        best = None
        for p in platforms:
            dy = p.y - player.bottom
            if dy < -self.ground_tolerance:
                continue
            landing_t = dy / max(player.vy, 1.0)
            if landing_t > 1.5:
                continue
            landing_x = player.x + player.vx * landing_t
            if p.x1 <= landing_x <= p.x2:
                cand = (landing_t, p, landing_x)
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is None:
            return None
        landing_t, p, landing_x = best
        return JumpPrediction(p.id, landing_x, p.y, landing_t, True)
