"""怪物跨帧追踪：给检测框稳定的 id。

决策层要「盯住同一只怪」——单帧检测框没有 id，某帧漏检就会换目标。
追踪器用 IoU 匹配把当前帧的框关联到上一帧的轨迹，漏检时用旧轨迹续命。

实现选贪心 IoU 匹配，不引入 ByteTrack 那套卡尔曼/匈牙利：
    横版 ARPG 的怪移动慢（一帧几个像素），IoU 匹配简单够用，还好调试。
    以后怪跑得快、漏检多、id 乱跳再升级。
"""
import time

from perception.world_state import Mob


def _iou(a, b):
    """两个 xyxy 框的 IoU。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


class MobTracker:
    """贪心 IoU 追踪器。

    update(detections) 返回本帧匹配到的 Mob 列表；
    detections 形如 [(x1, y1, x2, y2, conf), ...]（YOLO 输出的 xyxy + 置信度）。
    """

    def __init__(self, min_iou=0.3, max_age=15):
        self.min_iou = min_iou
        self.max_age = max_age
        self.debounce_conf = 0.0    # 防抖置信度：高于它的框消失后保留位置
        self.debounce_ms = 0.0      # 防抖时间（毫秒）
        self._tracks = []       # 每条是 dict，字段见 _new_track
        self._next_id = 1

    @staticmethod
    def _new_track(tid, x1, y1, x2, y2, conf):
        return {
            "id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf, "missed": 0, "age": 1, "vx": 0.0, "vy": 0.0,
            "last_seen": time.monotonic(),
        }

    def update(self, detections):
        used = [False] * len(detections)
        out = []
        now = time.monotonic()

        # 1) 贪心匹配：每个轨迹找当前帧里 IoU 最大的框
        for t in self._tracks:
            t["missed"] += 1
            best_i, best_iou = -1, self.min_iou
            for i, d in enumerate(detections):
                if used[i]:
                    continue
                v = _iou((t["x1"], t["y1"], t["x2"], t["y2"]), d[:4])
                if v > best_iou:
                    best_iou, best_i = v, i

            if best_i >= 0:
                used[best_i] = True
                x1, y1, x2, y2, cf = detections[best_i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                pcx = (t["x1"] + t["x2"]) / 2.0
                pcy = (t["y1"] + t["y2"]) / 2.0
                t["vx"] = cx - pcx
                t["vy"] = cy - pcy
                t["x1"], t["y1"], t["x2"], t["y2"], t["conf"] = \
                    x1, y1, x2, y2, cf
                t["missed"] = 0
                t["age"] += 1
                t["last_seen"] = now
                out.append(self._to_mob(t))
            elif (t["conf"] >= self.debounce_conf
                  and (now - t["last_seen"]) * 1000.0 <= self.debounce_ms):
                # 漏检但高置信度 + 防抖时间内：保留消失前的位置（幽灵目标），
                # 让决策层不因短暂漏检丢目标；下一帧若在该框范围扫出 IoU
                # 匹配的新框，会正常更新位置。
                out.append(self._to_mob(t))

        # 2) 未匹配的检测框 → 新轨迹
        for i, d in enumerate(detections):
            if not used[i]:
                x1, y1, x2, y2, cf = d
                self._tracks.append(self._new_track(self._next_id,
                                                    x1, y1, x2, y2, cf))
                self._next_id += 1
                out.append(self._to_mob(self._tracks[-1]))

        # 3) 清理长期漏检的轨迹。
        # 用时间戳（和防抖同单位）：漏检时间超过「防抖时间 + 500ms 缓冲」才清理。
        # 不能用 max_age 帧数 —— 帧数 × fps 随帧率变，高帧率时防抖还没到期
        # 轨迹就被清掉了（防抖 1000ms ≈ 27 帧，但 max_age=15 帧 ≈ 0.5s 就清理）。
        keep_ms = self.debounce_ms + 500.0
        self._tracks = [t for t in self._tracks
                        if (now - t["last_seen"]) * 1000.0 <= keep_ms]

        return out

    def _to_mob(self, t):
        return Mob(
            id=t["id"],
            x=(t["x1"] + t["x2"]) / 2.0,
            y=(t["y1"] + t["y2"]) / 2.0,
            w=t["x2"] - t["x1"],
            h=t["y2"] - t["y1"],
            conf=t["conf"],
            vx=t["vx"],
            vy=t["vy"],
            age=t["age"],
            missed=t["missed"],
        )

    def kill(self, mob_id):
        """立即消除某个轨迹（防抖幽灵框被攻击时调用，避免持续空放技能）。"""
        self._tracks = [t for t in self._tracks if t["id"] != mob_id]


class PlayerTracker:
    """跟住「我」：在每帧 YOLO 的玩家框里挑出操作者那个。

    不依赖外观（不需要「其他玩家」类）。YOLO 只给出「所有玩家框」，谁是自己
    得靠**位置连续性**判断：

      · 未锁定（开局 / 跟丢后）→ 取置信度最高的框锁定
      · 已锁定 → 每帧选「离预测位置最近、且不超过 max_jump」的候选
        （预测位置 = 上一帧位置 + 速度外推，速度做指数平滑）
      · 漏检 → 用速度外推补一个框，连续漏检超过 max_missed 帧就判定跟丢，
        解锁退回未锁定（下一帧重新按置信度锁）

    这样别的玩家从旁边经过时不会被「抢锁」（只要他离预测位置更远），
    比「每帧取置信度最高」稳得多。

    cands 形如 [(x1, y1, x2, y2, conf), ...]（画面坐标 xyxy）。
    返回 player_box = (cx, cy, bottom, conf, bw, bh)，和 live_thread 原格式一致；
    无框返回 None。
    """

    def __init__(self, max_jump=150.0, max_missed=10, smooth=0.6):
        self.max_jump = float(max_jump)     # 位置突变阈值（像素）：超过就认为不是「我」
        self.max_missed = int(max_missed)   # 连续漏检多少帧放弃锁定
        self.smooth = float(smooth)         # 速度平滑系数（0~1，越大越平滑）
        self._box = None                    # 当前锁定框 (cx, cy, bottom, conf, bw, bh)
        self._vx = 0.0
        self._vy = 0.0
        self._missed = 0

    @property
    def locked(self):
        return self._box is not None

    @staticmethod
    def _cand_to_box(c):
        x1, y1, x2, y2, conf = c
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, y2, conf, x2 - x1, y2 - y1)

    def update(self, cands):
        """喂本帧的玩家候选框，返回跟住的 player_box（或 None）。"""
        if self._box is None:
            # 未锁定：取置信度最高的（开局 / 跟丢后重新锁）
            if not cands:
                return None
            self._box = self._cand_to_box(max(cands, key=lambda c: c[4]))
            self._vx = self._vy = 0.0
            self._missed = 0
            return self._box

        px = self._box[0] + self._vx   # 预测位置（位置 + 速度外推）
        py = self._box[1] + self._vy

        best, best_d = None, None
        for c in cands:
            cx = (c[0] + c[2]) / 2.0
            cy = (c[1] + c[3]) / 2.0
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if d > self.max_jump:
                continue               # 太远：不是「我」（可能是别人经过 / 已传送）
            if best_d is None or d < best_d:
                best_d, best = d, c

        if best is not None:
            nb = self._cand_to_box(best)
            a = self.smooth
            self._vx = a * self._vx + (1 - a) * (nb[0] - self._box[0])
            self._vy = a * self._vy + (1 - a) * (nb[1] - self._box[1])
            self._box = nb
            self._missed = 0
            return nb

        # 没匹配到 / 本帧没候选：外推一个框，累计漏检
        self._missed += 1
        if self._missed > self.max_missed:
            self._box = None           # 跟丢：解锁，下一帧重新按置信度锁
            self._vx = self._vy = 0.0
            return None
        cx, cy, bottom, conf, bw, bh = self._box
        self._box = (px, py, bottom + (py - cy), conf, bw, bh)
        return self._box
