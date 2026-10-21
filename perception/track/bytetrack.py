# -*- coding: utf-8 -*-
"""**ByteTrack 跟踪内核**（2026-10-17 用户定稿 ✓ 替代原来那一套"防抖置信度 + 防抖时间"✗）。

用户原话（2026-10-17 ✓）：
  「去掉现在的怪物检测框防抖置信度和防抖时间的配置和逻辑，现在完全不对。
   接 ByteTrack，新参数『**失联帧长**』（track_buffer）代替以前的『防抖时间』，默认 1000ms。
   稳定 ID 后，再用卡尔曼/EMA 平滑输出框坐标做轻量平滑，对 `(cx, cy, w, h)` 建 **8 维状态**，
   **匀速模型**够用：观测 = YOLO 框；预测 = 上一帧速度外推；更新 = 观测值融合。
   按 `track_id` 维护每个目标的：**最近 N 帧位置、平均置信、存活时间**；
   新出现的**低分框先挂 buffer，连续 2~3 帧稳定再认**。」

⚠ 为什么原来的那套"完全不对"（说清楚，别再走回去 ✗）：
  · 旧实现把两件事混在一起 —— ①「这一拍**显不显示**」（确认闸 ✓）
    ②「消失后**还留多久**」（`debounce_conf` / `debounce_ms` ✓）；
  · 于是调 `debounce_ms` 对"框闪"几乎没用 ✓（闪是"显示"那一侧的事 ✓）；
  · 而"留多久"按**时间**算、又要求"曾经到过 `debounce_conf`" ⇒ 置信度刚好在阈值上下
    跳的目标**两头不靠** ✓（用户现场：框一闪一闪、甚至整拍消失 ✓）。
⇒ ByteTrack 的口径是**一套**：两张关联表（高分档 / 低分档）+ 一条**失联缓冲**
  （`track_buffer`，按**帧**算、参数按 **ms** 给 ✓）+ 卡尔曼匀速外推 ✓。

本模块**只做跟踪**（不看业务、不 import 检测器 ✓）：
  · `ByteTracker.update(high, low)` —— `high`/`low` 都是 `[(x1,y1,x2,y2,conf), ...]`；
  · 每条 `Track` 带：`tid` / `box`（Kalman 平滑后的 `cx,cy,w,h`）/ `conf` / `avg_conf` /
    `history`（最近 `HISTORY_LEN` 帧的位置）/ `age`（帧）/ `alive_ms`（毫秒）/ `missed`；
  · `ByteTracker.shift(dx, dy)` —— 相机 / 角色位移补偿（把轨迹整体挪一帧 ✓ 见 `MobTracker` ✓）。

⚠ 只有**高分档**才能**新建**轨迹 ✓（低分档只给已存在的轨迹"续命" ✓）——
  这就是 BYTE 的第二阶段，也是"低分框先挂 buffer、连续 2~3 帧稳定再认"那条 ✓。
"""
import time
from collections import deque

import numpy as np

from perception.assign import BIG, hungarian

# ---- 轨迹状态 ----
NEW = 0
TRACKED = 1
LOST = 2
REMOVED = 3

#: 新目标要**连续被看见**几帧才"认"（用户：连续 **2~3** 帧稳定再认 ✓ ⇒ 取 3 ✓ 宁缺勿错 ✓）。
CONFIRM_FRAMES = 3
#: **失联帧长**的默认值（毫秒 ✓ 用户 2026-10-17 定稿：默认 **1000ms** ✓）。
TRACK_BUFFER_MS = 1000.0
#: 每轨最多留多少帧历史位置（"最近 N 帧位置" ✓ 也给界面/统计用 ✓）。
HISTORY_LEN = 30

#: 两阶段关联的 IoU 下限（低于它 = 配不上 ✓ 代价表里直接 `BIG` ✗ 不进图 ✓）。
MATCH_IOU_HIGH = 0.2      # 第一阶段：已跟踪轨迹 × 高分框
MATCH_IOU_LOW = 0.5       # 第二阶段：还没配上的轨迹 × **低分框**（擦着阈值的那种 ✓）
#: 第三阶段：**未确认**轨迹 × 剩下的高分框。
#: ⚠ 一开始写成 0.7 ✗ ⇒ **快速移动的目标永远确认不了** ✓（40px 的框 10px/帧 ⇒ IoU 只有
#:   0.6 < 0.7 ⇒ 每帧都"配不上、删掉、再新建" ✓ 实测 id 每帧都在换 ✓）
#:   —— 而头两帧速度还是 0（Kalman 没学到 ✓）⇒ 预测也帮不上 ✓ ⇒ 阈值必须和第一阶段同档 ✓。
MATCH_IOU_UNCONF = 0.3
#: 取消置位 / 移除时，"闪一下"记几条（与 `MobTracker.FLICK_KEEP` 同口径 ✓）。
FLICK_KEEP = 32
#: 「刚丢的确认轨迹」现场留多少条（同 `FLICK_KEEP` 的上限思路 ✓ 防无限涨 ✗）
LOST_KEEP = 300


def iou_xyxy(a, b):
    """两个 `xyxy` 框的 IoU（`perception.tracker._iou` 同一口径 ✓ 这里自留一份不互相 import ✓）。"""
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _xyxy_to_cxcywh(det):
    x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0,
            max(x2 - x1, 1.0), max(y2 - y1, 1.0))


def _cxcywh_to_xyxy(box):
    cx, cy, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class _Kalman:
    """8 维状态 `(cx, cy, w, h, vx, vy, vw, vh)` 的**匀速模型**卡尔曼滤波（用户 2026-10-17 ✓）。

    · **观测** = YOLO 框的 `(cx, cy, w, h)` ✓；
    · **预测** = 上一帧速度**外推**（`F` 里那条 `x += v·dt` ✓）；
    · **更新** = 观测值与预测值按协方差**融合** ✓（`K` 自动决定信谁多一点 ✓）。
    ⚠ 输出用 `mean[:4]`（滤波后的**平滑位置** ✓）—— 而不是原始检测框 ✓。
    ⚠ 匀速模型对"突然转弯 / 被击退"会慢一点 ✓ —— 那是**有意**的（要的就是轻量平滑 ✓），
      真跳变由关联那一侧处理：IoU 配不上 ⇒ 老轨迹失联、新框另起一条 ✓。
    """

    #: 观测噪声：框宽高的 1/20（原版 ByteTrack 的口径 ✓ 别拍脑袋改 ✗）
    _STD_POS = 1.0 / 20.0
    #: **尺寸**那两维的观测噪声（**1/2** ✓ 见 `update` 里那段说明：框宽高抖得厉害、
    #: 也没必要跟快 ✓）。实测（框 40×50、宽度在 18~45 之间跳）：1/6 时逐帧最大跳 **15px**，
    #: 1/2 时 **7.8px** ✓ —— 而位置那两维（瞄准要用 ✓）仍保持 1/20 的紧口径 ✓。
    _STD_SIZE = 1.0 / 2.0
    #: 过程噪声：框宽高的 1/160（同上 ✓）
    _STD_VEL = 1.0 / 160.0

    __slots__ = ("mean", "cov")

    def __init__(self, box):
        cx, cy, w, h = box
        w, h = max(float(w), 1.0), max(float(h), 1.0)
        self.mean = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=float)
        std = np.array([
            2 * self._STD_POS * w, 2 * self._STD_POS * h,
            2 * self._STD_POS * w, 2 * self._STD_POS * h,
            10 * self._STD_VEL * w, 10 * self._STD_VEL * h,
            10 * self._STD_VEL * w, 10 * self._STD_VEL * h,
        ])
        self.cov = np.diag(std ** 2)

    @property
    def box(self):
        cx, cy, w, h = (float(v) for v in self.mean[:4])
        return (cx, cy, max(w, 1.0), max(h, 1.0))

    def predict(self, dt=1.0):
        """匀速外推：位置 += 速度 × dt（`dt=1` = 一帧 ✓）。"""
        f = np.eye(8)
        for i in range(4):
            f[i, i + 4] = float(dt)
        self.mean = f @ self.mean
        # 宽高不许跑到 0 以下（否则 IoU 全 0，这条轨迹就再也配不上了 ✗）
        self.mean[2] = max(float(self.mean[2]), 1.0)
        self.mean[3] = max(float(self.mean[3]), 1.0)
        w, h = self.mean[2], self.mean[3]
        q = np.diag(np.array([
            self._STD_VEL * w, self._STD_VEL * h,
            self._STD_VEL * w, self._STD_VEL * h,
            self._STD_VEL * w, self._STD_VEL * h,
            self._STD_VEL * w, self._STD_VEL * h,
        ]) ** 2)
        self.cov = f @ self.cov @ f.T + q

    def update(self, box):
        """观测融合：`mean += K·(z − H·mean)` ✓。"""
        cx, cy, w, h = box
        w, h = max(float(w), 1.0), max(float(h), 1.0)
        z = np.array([cx, cy, w, h], dtype=float)
        h_mat = np.zeros((4, 8))
        for i in range(4):
            h_mat[i, i] = 1.0
        # ⚠ **尺寸那两维的观测噪声故意大得多** ✗（1/6 而不是 1/20）：YOLO 框宽高天天在
        #   ±50% 地跳（只框到半个身子 ✓），而"框多大"本来就不需要跟得那么快 ✓ ——
        #   位置（cx/cy）保持紧（1/20 ✓ 瞄准要用 ✓），尺寸让它平一点 ✓（用户要的"轻量平滑"✓）。
        r = np.diag(np.array([
            self._STD_POS * w, self._STD_POS * h,
            self._STD_SIZE * w, self._STD_SIZE * h,
        ]) ** 2)
        s = h_mat @ self.cov @ h_mat.T + r
        k = self.cov @ h_mat.T @ np.linalg.inv(s)
        self.mean = self.mean + k @ (z - h_mat @ self.mean)
        self.cov = (np.eye(8) - k @ h_mat) @ self.cov

    def shift(self, dx, dy):
        """相机/角色位移补偿：位置平移，**速度不动** ✓（速度是目标自己的 ✓）。"""
        if not dx and not dy:
            return
        self.mean[0] += float(dx)
        self.mean[1] += float(dy)


class Track:
    """一条轨迹（`tid` 稳定 ✓ —— 决策层"盯住同一只怪"靠它 ✓）。

    按用户 2026-10-17 的要求**逐条记账**：最近 N 帧位置 / 平均置信 / 存活时间 ✓。
    """

    __slots__ = ("tid", "state", "kf", "conf", "peak_conf", "_conf_sum", "_conf_n",
                 "hits", "hits_streak", "age", "missed", "born", "last_seen",
                 "confirm_frames", "history", "_ever")

    def __init__(self, det, tid, now, confirm_frames=CONFIRM_FRAMES):
        self.tid = int(tid)
        self.state = NEW
        self.kf = _Kalman(_xyxy_to_cxcywh(det))
        self.conf = float(det[4])
        self.peak_conf = self.conf
        self._conf_sum = self.conf
        self._conf_n = 1
        self.hits = 1                 # 累计被观测到几次
        self.hits_streak = 1          # **连续**被观测到几次（确认判据 ✓ 漏一拍就归零 ✓）
        self.age = 1                  # 出生到现在过了几帧
        self.missed = 0               # 连续漏检帧数（>0 = 靠失联缓冲留着的"幽灵框" ✓）
        self.born = float(now)
        self.last_seen = float(now)
        self.confirm_frames = int(confirm_frames)
        #: 「**确认过**」这件事**只涨不落** ✓ —— 一旦攒够连续帧数就永远算"认过的"✓。
        #: ⚠ 不能拿 `hits_streak` 现算 ✗：它漏一拍就归零 ⇒ 失联缓冲期里的幽灵框会
        #:   因为"没确认"被当场删掉 ⇒ **失联帧长形同虚设** ✓（第一版实测就是这样 ✗）。
        self._ever = False
        self.history = deque(maxlen=HISTORY_LEN)
        self.history.append(self.kf.box)

    # ---- 读数 ----
    @property
    def box(self):
        """Kalman 平滑后的框 `(cx, cy, w, h)` —— **输出给外界的就它** ✓。"""
        return self.kf.box

    @property
    def confirmed(self):
        """够不够"我看见它了"：**曾经连续 `confirm_frames` 帧** ✓（只涨不落 ✓ 见 `_ever`）。"""
        return bool(self._ever) and self.state != REMOVED

    @property
    def avg_conf(self):
        return self._conf_sum / self._conf_n if self._conf_n else 0.0

    @property
    def alive_ms(self):
        return (self.last_seen - self.born) * 1000.0

    def velocity(self):
        return (float(self.kf.mean[4]), float(self.kf.mean[5]))

    # ---- 生命周期 ----
    def predict(self):
        self.age += 1
        self.kf.predict()

    def update(self, det, now, low=False):
        self.kf.update(_xyxy_to_cxcywh(det))
        self.conf = float(det[4])
        #: ⚠ **低分档不抬 `peak_conf`** ✗（2026-10-11 的规矩 ✓）：`peak_conf` 是"这条轨迹
        #:   到底被看清过没有"的读数 ✓ —— 擦着阈值的低分框去抬它，等于把"看清过"这件事
        #:   悄悄放宽 ✓（老用例 `t_low_conf_maintains_identity` ③ 就是钉这个 ✓）。
        if not low:
            self.peak_conf = max(self.peak_conf, self.conf)
        self._conf_sum += self.conf
        self._conf_n += 1
        self.hits += 1
        self.hits_streak += 1
        self.missed = 0
        self.state = TRACKED
        # 「连续 2~3 帧稳定」攒够 ⇒ **从此算认过**（只涨不落 ✓ 见 `_ever` 说明 ✓）
        if self.hits_streak >= self.confirm_frames:
            self._ever = True
        self.last_seen = float(now)
        self.history.append(self.kf.box)

    def mark_lost(self):
        self.missed += 1
        self.hits_streak = 0        # 「连续」断了 ⇒ 回来要重新攒 ✓（用户要的"稳定"口径 ✓）
        if self.state != NEW:
            self.state = LOST


class ByteTracker:
    """ByteTrack：两阶段关联 + 失联缓冲 + 确认闸 + 逐轨记账。

    `update(high, low)` 返回 `{"tracks": [Track...], "diag": {...}}`：
      · `tracks` 只含**确认过**的（`confirmed` ✓），其中 `missed > 0` 的是**失联缓冲期内的幽灵框** ✓
        （老链路就靠 `Mob.missed > 0` 认这个 ✓ 口径不变 ✓）；
      · `diag` 是这一帧的读数（`matched / low / created / lost / removed / flick` ✓ 给日志用 ✓）。
    """

    def __init__(self, track_buffer_ms=TRACK_BUFFER_MS, confirm_frames=CONFIRM_FRAMES,
                 match_iou_high=MATCH_IOU_HIGH, match_iou_low=MATCH_IOU_LOW,
                 match_iou_unconf=MATCH_IOU_UNCONF):
        self.tracks = []
        self._next_id = 1
        self.frame_id = 0
        #: **失联帧长**（毫秒 ✓ 用户定的新参数 ✓ 由设置灌 ✓）
        self.track_buffer_ms = float(track_buffer_ms)
        self.confirm_frames = int(confirm_frames)
        self.match_iou_high = float(match_iou_high)
        self.match_iou_low = float(match_iou_low)
        self.match_iou_unconf = float(match_iou_unconf)
        #: 帧率估计（**失联帧长按 ms 给、缓冲按帧算** ⇒ 必须知道 fps ✓ 用实测帧间隔 ✓
        #:  估不出来（第一帧 / 掉帧）就退回 30 ✓ 宁缺勿错 ✓）。
        self.fps = 30.0
        self._last_ts = None
        self.diag = {}
        self.flick = []
        #: 「**认过的轨迹刚丢失**」的现场（`missed` 0→1 ✓）—— 给 `mob_lost` 打点用 ✓
        self.lost_ev = []

    # ---------------- 关联 ----------------
    @staticmethod
    def _associate(tracks, dets, iou_thresh):
        """IoU 关联（**全局最优** ✓ 走 `perception.assign.hungarian` ✓ 不新造轮子 ✓）。

        返回 `[(tracks 下标, dets 下标), ...]`；低于阈值的一律**不进代价图**（`BIG` ✓）。
        """
        if not tracks or not dets:
            return []
        cost = []
        for t in tracks:
            tb = _cxcywh_to_xyxy(t.box)
            row = []
            for d in dets:
                # ① IoU（静止 / 慢速走这条 ✓）
                ov = iou_xyxy(tb, d)
                # ② **离得够近**也算认得上 —— 快速移动时 IoU 会掉得很凶：
                #    40px 的框 30px/帧 ⇒ IoU 只有 0.14 ✗（40px 高/宽，位移 30 ⇒ (40-30)/(40+30)）
                #    ⇒ 光靠 IoU ⇒ **快速移动的目标每帧换 id** ✓（老用例 `t_fast_moving_one_box` 就是钉它 ✓）。
                #    口径沿用旧内核那条"认领半径"：**0.9 × 框的长边** ✓（与 `_gate` 同源 ✓）。
                _rad = 0.9 * max(float(t.box[2]), float(t.box[3]), 1.0)
                _dx = float(t.box[0]) - (float(d[0]) + float(d[2])) / 2.0
                _dy = float(t.box[1]) - (float(d[1]) + float(d[3])) / 2.0
                _dist = (_dx * _dx + _dy * _dy) ** 0.5
                prox = max(0.0, 1.0 - _dist / _rad)          # 1 = 正中，0 = 半径外
                score = max(ov, prox)
                row.append(BIG if score < iou_thresh else (1.0 - score))
            cost.append(row)
        return hungarian(cost)

    def _buffer_frames(self):
        return max(1, int(round(self.track_buffer_ms / 1000.0 * float(self.fps))))

    def _note_lost(self, t):
        """记一条「**认过的轨迹刚丢失**」（`missed` 0→1 的那一帧 ✓ 只记一次 ✓）。"""
        try:
            self.lost_ev.append({
                "id": int(t.tid),
                "cx": round(float(t.box[0]), 1), "cy": round(float(t.box[1]), 1),
                "w": round(float(t.box[2]), 1), "h": round(float(t.box[3]), 1),
                "conf": round(float(t.conf), 3),
                "peak": round(float(t.peak_conf), 3),
                "age": int(t.age),
            })
            if len(self.lost_ev) > LOST_KEEP:
                del self.lost_ev[:-LOST_KEEP]
        except Exception:                      # noqa: BLE001 —— 记账不许拖垮跟踪 ✗
            pass

    def drain_lost(self):
        """取走「刚丢的确认轨迹」现场（取完就清 ✓ 同 `drain_flick` ✓）。"""
        out, self.lost_ev = self.lost_ev, []
        return out

    def _note_flick(self, t):
        try:
            self.flick.append({
                "id": int(t.tid),
                "age": int(t.age),
                "conf": round(float(t.conf), 3),
                "peak": round(float(t.peak_conf), 3),
                "missed": int(t.missed),
                "ms": round(float(t.alive_ms), 1),
            })
            if len(self.flick) > FLICK_KEEP:
                del self.flick[:-FLICK_KEEP]
        except Exception:                      # noqa: BLE001 —— 记账不许拖垮跟踪 ✗
            pass

    # ---------------- 主循环 ----------------
    def update(self, high, low=()):
        now = time.monotonic()
        # 0) 帧率估计（EMA ✓ 只认"看得过去"的间隔 ✓）
        if self._last_ts is not None:
            dt = now - self._last_ts
            if 1e-3 < dt < 0.5:
                # ⚠ 头几帧**直接取值**（不做 EMA ✓）—— 从 30 用 EMA 慢慢爬下来要二十来帧，
                #   那期间「失联帧长 ⇒ 帧数」会算大一截 ✓（实测 700ms 算出 17 帧而不是 11 ✓）
                if self.frame_id <= 2:
                    self.fps = 1.0 / dt
                else:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self._last_ts = now
        self.frame_id += 1
        high = [list(map(float, d[:5])) for d in (high or ())]
        low = [list(map(float, d[:5])) for d in (low or ())]

        diag = {"matched": 0, "low": 0, "created": 0, "lost": 0,
                "removed": 0, "flick": 0, "buffer": self._buffer_frames(),
                "tracks": 0}
        # ⚠ `flick` **不在这里清** ✗ —— 清了的话"这一拍记的闪一下"当场就没了 ✓
        #   （外部是"每帧 update 完取一次"✓ 见 `drain_flick` ✓；没人取也有 `FLICK_KEEP` 上限 ✓）

        # 1) 全体外推（匀速模型 ✓ "预测 = 上一帧速度外推" ✓）
        for t in self.tracks:
            t.predict()

        # 2) 第一阶段：**已跟踪 + 失联** 的轨迹 × **高分框**
        pool = [t for t in self.tracks if t.state in (TRACKED, LOST)]
        used_t, used_d = set(), set()
        for ti, di in self._associate(pool, high, self.match_iou_high):
            t, d = pool[ti], high[di]
            t.update(d, now)
            used_t.add(id(t))
            used_d.add(di)
            diag["matched"] += 1

        # 3) 第二阶段：还没配上的轨迹 × **低分框**（BYTE ✓ 低分只续命、**永不新建** ✓）
        rest = [t for t in pool if id(t) not in used_t]
        if rest and low:
            for ti, di in self._associate(rest, low, self.match_iou_low):
                t = rest[ti]
                t.update(low[di], now, low=True)      # 低分不抬 peak_conf ✓ 见 `Track.update`
                used_t.add(id(t))
                diag["low"] += 1

        # 4) 第三阶段：**未确认**的新轨迹 × 剩下的高分框（它们抖得厉害 ⇒ 阈值放宽一点 ✓
        #    —— 这就是"连续 2~3 帧稳定再认"那条的另一半：攒够帧数才认 ✓）
        unconf = [t for t in self.tracks
                  if t.state == NEW and id(t) not in used_t and not t.confirmed]
        free_high = [i for i in range(len(high)) if i not in used_d]
        if unconf and free_high:
            for ti, dj in self._associate(unconf, [high[i] for i in free_high],
                                          self.match_iou_unconf):
                t = unconf[ti]
                di = free_high[dj]
                t.update(high[di], now)
                used_t.add(id(t))
                used_d.add(di)

        # 5) 还没认领的高分框 ⇒ 新轨迹（**只有高分档才能新建** ✓）
        for di, d in enumerate(high):
            if di in used_d:
                continue
            t = Track(d, self._next_id, now, self.confirm_frames)
            self._next_id += 1
            self.tracks.append(t)
            # ⚠ 必须记成"这一帧认领过"✗ —— 否则下面第 6 步会把它当成"没配上"
            #   ⇒ **当拍就删** ⇒ 永远攒不到确认帧 ⇒ 一个框都出不来 ✓（第一版实测踩到 ✓）
            used_t.add(id(t))
            diag["created"] += 1

        # 6) 失联 / 移除（**失联帧长**说了算 ✓ 但只对"确认过"的保留 ✓
        #    —— 没确认过的本来就不该有身份 ✗ 直接丢、并记一条 `flick` ✓）
        keep, buf = [], self._buffer_frames()
        for t in self.tracks:
            if id(t) in used_t:
                keep.append(t)
                continue
            t.mark_lost()
            # ⚠ 先记下"确认过没有" ✗ —— 下面把 `state` 置成 `REMOVED` 之后
            #   `confirmed` 就是 False 了 ⇒ 会把**确认过的老轨迹**也记成"闪一下" ✓
            #   （实测现场：`mob_flick` 天天误报 ✓）
            _was_confirmed = bool(t.confirmed)
            # ⭐ **"确认过的轨迹刚丢"记一条现场**（用户 2026-10-17 ✓ 原话：
            #   「**离玩家远的时候怪物检出框较稳定 离得近时开始频繁丢失**」✗）：
            #   光看"丢了"分不出病因（**被玩家/特效挡住** ✗ vs **近身目标太大超出训练分布** ✗
            #   vs **血条/名字压在身上** ✗）⇒ 得把**丢的时候那些框在哪、多大、多少分**留下 ✓
            #   （`live_thread` 会再补"和玩家框的重叠 / 距离"✓ 见那边的 `mob_lost` ✓）。
            if _was_confirmed and t.missed == 1:
                self._note_lost(t)
            if _was_confirmed and t.missed <= buf:
                keep.append(t)
                diag["lost"] += 1
            else:
                t.state = REMOVED
                diag["removed"] += 1
                if not _was_confirmed:
                    diag["flick"] += 1
                    self._note_flick(t)
        self.tracks = keep

        # 7) 输出：确认过的（含失联缓冲内的幽灵 ✓ `missed > 0` 即幽灵 ✓）
        out = [t for t in self.tracks if t.confirmed]
        diag["tracks"] = len(out)
        self.diag = diag
        return {"tracks": out, "diag": diag}

    # ---------------- 外部操作 ----------------
    def shift(self, dx, dy):
        """整体平移（相机 / 角色位移补偿 ✓ 见 `MobTracker._shift_anchor` ✓）。"""
        for t in self.tracks:
            t.kf.shift(dx, dy)

    def remove(self, tid):
        """立即抹掉一条轨迹（幽灵框被攻击时调 ✓ 决策层 `kill` ✓）。"""
        self.tracks = [t for t in self.tracks if t.tid != int(tid)]

    def drain_flick(self):
        out, self.flick = self.flick, []
        return out
