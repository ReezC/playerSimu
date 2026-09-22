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
        # 横向 opening 删除树干/角色等短小绿色块，closing 补草坪上的小间隙。
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < self.min_width or cw < ch * 3:
                continue
            # 顶部附近必须有足够多的绿色像素，避免把整片灌木的外框误作平台。
            band = mask[y:min(y + max(2, min(5, ch)), mask.shape[0]), x:x + cw]
            density = float(np.count_nonzero(band)) / max(1, band.size)
            if density < 0.30:
                continue
            out.append((float(x), float(x + cw), float(y + top), min(1.0, density)))
        # 同一平台可能被细小纹理切成相邻轮廓；先合并再输出一条顶边。
        out.sort(key=lambda p: (p[2], p[0]))
        merged = []
        for x1, x2, y, conf in out:
            if merged and abs(y - merged[-1][2]) <= 5 and x1 <= merged[-1][1] + 18:
                ox1, ox2, oy, oc = merged[-1]
                merged[-1] = (min(ox1, x1), max(ox2, x2), min(oy, y), max(oc, conf))
            else:
                merged.append((x1, x2, y, conf))
        return merged


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


def relate_terrain(mobs, player, platforms):
    """给每只怪打上所在平台 id（fh），并标记是否与玩家当前平台连接。

    这是「地形关系识别」，属于感知层：Agent 只消费 ``mob.reachable``，
    不再自己做「怪是否和玩家同平台」的几何判断。

    当前阶段「连接」= 同一平台顶边；将来做跳跃标定后可扩展为「相邻可
    走通平台」，Agent 侧无需改动。
    """
    pid = player.current_platform_id
    if pid is None:
        # 玩家当前平台未知 → 退化旧行为：全部视为可达
        for m in mobs:
            m.platform_id = None
            m.reachable = True
        return

    current = next((p for p in platforms if p.id == pid), None)
    if current is None:
        for m in mobs:
            m.platform_id = None
            m.reachable = True
        return

    for m in mobs:
        mob_bottom = m.y + m.h / 2.0
        tolerance = max(36.0, m.h * 1.5)

        # 站在玩家当前平台顶边 → 直接可达
        if current.x1 <= m.x <= current.x2 and abs(mob_bottom - current.y) <= tolerance:
            m.platform_id = pid
            m.reachable = True
            continue

        # 找怪实际所在平台（fh）
        m.platform_id = None
        for p in platforms:
            if p.x1 <= m.x <= p.x2 and abs(mob_bottom - p.y) <= tolerance:
                m.platform_id = p.id
                break

        # 连接 = 与玩家同一平台（当前阶段）
        m.reachable = (m.platform_id == pid)
