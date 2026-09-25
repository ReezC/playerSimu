"""怪物跨帧追踪：给检测框稳定的 id。

决策层要「盯住同一只怪」——单帧检测框没有 id，某帧漏检就会换目标。

**实现选贪心匹配 + 运动补偿，不引入 ByteTrack 那套卡尔曼/匈牙利**：
    横版 ARPG 的怪移动慢，用不上那么重的东西；这里的三条规则都是冲着
    **实测踩过的现象**去的，出了问题好定位：

    1) **运动补偿匹配**（只比 IoU 不够）
       相邻两帧 IoU 会随移动掉得很快：同尺寸的框横移 d（框宽 w）时
       IoU ≈ (w-d)/(w+d) —— 移动超过约 0.54w 就跌破 0.3。怪在走、框又小的时候
       这就是常态 → 同一个怪**每帧新起一条轨迹**，屏幕上是"移动轨迹上排着一串框"。
       所以先按速度**预测**它这一帧该在哪，再用「IoU 达标」或「离预测位置够近
       （和框尺寸挂钩）」两种方式认领。YOLO 把框判大判小（只框到半个身子）
       中心却没怎么动 —— 这类也靠距离那条兜住。

    2) **同体合并**（幽灵框 + 新轨迹是同一只怪）
       上一条漏了的时候：防抖留下的幽灵框停在旧位置，新起的轨迹在新位置，
       两者**同时**输出 → 看起来又是一串框。合并时保留**活得久的那个 id**
       （决策层的目标锁定 CD 靠 id 稳定），位置取更新鲜的那条。

    3) **位置/尺寸平滑 + 幽灵外推**（"框在抖"与"框粘在原地"）
       检测本身有噪声，直接输出框会一直晃；防抖期间冻结在旧位置，移动目标
       又会和框越差越远（决策层朝空位置走 / 空放技能）。所以：位置按"移动快
       就不平滑、几乎不动就重平滑"的自适应系数走，尺寸恒定平滑；漏检期间按
       衰减的速度继续走一小段（有总位移上限）。

对外接口不变：`update(detections) -> [Mob]`、`kill(id)`；`Mob.missed > 0`
仍然表示「幽灵框（这一帧没检测到，靠防抖留着）」—— 决策层就靠它防空放技能。
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
    """贪心匹配的怪追踪器（带运动补偿 / 同体合并 / 平滑，见模块文档）。

    update(detections) 返回本帧的 Mob 列表；
    detections 形如 [(x1, y1, x2, y2, conf), ...]（YOLO 输出的 xyxy + 置信度）。

    下面这些参数都是"手感"参数，**先固定在内部**：调歪了比不调更糟，
    真要开给界面也得先把每一条的现象描述清楚（就像 debounce_* 那两条）。
    """

    def __init__(self, min_iou=0.3, max_age=15, gate_ratio=0.9, merge_iou=0.5,
                 smooth=0.55, size_smooth=0.5, vel_smooth=0.6,
                 ghost_decay=0.75, ghost_max_shift=0.6):
        self.min_iou = min_iou      # IoU 匹配阈值（达到就直接认领）
        self.max_age = max_age      # 保留兼容：清理现在按**时间**算（见 update 第 4 步）
        # 认领半径 = gate_ratio × 框的长边（兜住"移动快 / 框判小"导致的 IoU 崩掉）
        self.gate_ratio = gate_ratio
        self.merge_iou = merge_iou  # 两条轨迹 IoU 超过它 = 同一只怪（合并）
        self.smooth = smooth        # 位置平滑上限（移动快时自动降下来，见 _gain）
        self.size_smooth = size_smooth  # 尺寸平滑（恒定，怪的大小不会突变）
        self.vel_smooth = vel_smooth    # 速度平滑（预测用，抖动的速度预测不准）
        self.ghost_decay = ghost_decay  # 幽灵框每帧速度衰减
        self.ghost_max_shift = ghost_max_shift  # 幽灵总位移上限（× 框长边）

        self.debounce_conf = 0.0    # 防抖置信度：高于它的框消失后保留位置
        self.debounce_ms = 0.0      # 防抖时间（毫秒）
        self._tracks = []       # 每条是 dict，字段见 _new_track
        self._next_id = 1

    # ---------------- 轨迹 ----------------

    @staticmethod
    def _new_track(tid, x1, y1, x2, y2, conf, now):
        return {
            "id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf, "missed": 0, "age": 1, "vx": 0.0, "vy": 0.0,
            "shifted": 0.0,          # 幽灵外推累计走了多远（到上限就不走了）
            "last_seen": now,
        }

    @staticmethod
    def _center(t):
        return ((t["x1"] + t["x2"]) / 2.0, (t["y1"] + t["y2"]) / 2.0)

    @staticmethod
    def _size(t):
        return (t["x2"] - t["x1"], t["y2"] - t["y1"])

    def _gate(self, t):
        """认领半径：和框的尺寸挂钩。

        为什么不用固定像素：小框（远处的怪）动几像素 IoU 就崩，大框能跑远点 ——
        阈值跟着框走，各种距离下才是同一套手感。
        """
        w, h = self._size(t)
        return max(10.0, self.gate_ratio * max(w, h))

    def _score(self, t, d):
        """这个检测框能不能认成这条轨迹：能认返回 >0，不能返回 -1。

        分数只用来**排序**，不参与别处计算：IoU 达标的（score = IoU）永远排在
        「只满足距离」的前面 —— 有真正重叠的框时，距离近不该抢走它。
        """
        iou = _iou((t["x1"], t["y1"], t["x2"], t["y2"]), d[:4])
        if iou >= self.min_iou:
            return iou

        cx, cy = self._center(t)
        pcx, pcy = cx + t["vx"], cy + t["vy"]       # 预测位置（位置 + 速度外推）
        dx = (d[0] + d[2]) / 2.0 - pcx
        dy = (d[1] + d[3]) / 2.0 - pcy
        dist = (dx * dx + dy * dy) ** 0.5
        gate = self._gate(t)
        if dist > gate:
            return -1.0
        # 距离越近分数越高，但乘 0.99 压在 min_iou 之下（排序上永远次于 IoU 达标）
        return self.min_iou * 0.99 * (1.0 - dist / gate)

    # ---------------- 每帧 ----------------

    def update(self, detections):
        now = time.monotonic()
        used = [False] * len(detections)

        # 1) 匹配：每条轨迹认领一个检测框。
        #    按 _tracks 的顺序来 —— 老的轨迹（含幽灵）排在前面，有优先认领权，
        #    这样"原来那条轨迹"更容易续上自己的 id，而不是被新轨迹顶掉。
        for t in self._tracks:
            t["missed"] += 1
            best_i, best_s = -1, 0.0
            for i, d in enumerate(detections):
                if used[i]:
                    continue
                s = self._score(t, d)
                if s > best_s:
                    best_s, best_i = s, i
            if best_i >= 0:
                used[best_i] = True
                self._absorb(t, detections[best_i], now)
            else:
                self._drift(t)

        # 2) 没被认领的检测框 → 新轨迹
        for i, d in enumerate(detections):
            if not used[i]:
                x1, y1, x2, y2, cf = d
                self._tracks.append(self._new_track(self._next_id, x1, y1,
                                                    x2, y2, cf, now))
                self._next_id += 1

        # 3) 同体合并：把"其实是同一只怪"的多条并成一条（一串框的元凶）
        self._dedupe()

        # 4) 清理长期漏检的轨迹。
        # 用时间戳（和防抖同单位）：漏检时间超过「防抖时间 + 500ms 缓冲」才清理。
        # 不能用 max_age 帧数 —— 帧数 × fps 随帧率变，高帧率时防抖还没到期
        # 轨迹就被清掉了（防抖 1000ms ≈ 27 帧，但 max_age=15 帧 ≈ 0.5s 就清理）。
        keep_ms = self.debounce_ms + 500.0
        self._tracks = [t for t in self._tracks
                        if (now - t["last_seen"]) * 1000.0 <= keep_ms]

        # 5) 输出：匹配上的 + 新轨迹 + 防抖期内的幽灵。
        #    合并之后每条轨迹最多出一个框 —— id 也不会重复。
        out = []
        for t in self._tracks:
            if t["missed"] == 0:
                out.append(self._to_mob(t))
            elif (t["conf"] >= self.debounce_conf
                  and (now - t["last_seen"]) * 1000.0 <= self.debounce_ms):
                out.append(self._to_mob(t))
        return out

    def _gain(self, t, cx, cy):
        """位置平滑系数：几乎不动的目标重平滑（去抖），走得快的轻平滑（不拖沓）。

        不加这条的话只能二选一：要么框一直抖（噪声直接输出），要么移动时框
        永远慢半拍（固定重平滑）。判据用**这一帧的实测位移**和框尺寸的比值。
        """
        pcx, pcy = self._center(t)
        moved = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
        w, h = self._size(t)
        ref = max(8.0, 0.5 * max(w, h))
        return self.smooth * max(0.0, 1.0 - moved / ref)

    def _absorb(self, t, d, now):
        """把这一帧的检测并进轨迹：平滑中心/尺寸、更新速度。

        用「中心 + 尺寸」而不是四条边分别平滑：这样**位置**和**大小**能各用一个
        系数 —— 移动的目标位置要紧跟（不然框拖在后面），而框大小不该跟着 YOLO
        一帧大一帧小（那正是"框在抖"的来源）。
        """
        x1, y1, x2, y2, cf = d
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ocx, ocy = self._center(t)
        ow, oh = self._size(t)

        a = self._gain(t, cx, cy)
        b = self.size_smooth
        mcx = a * ocx + (1 - a) * cx
        mcy = a * ocy + (1 - a) * cy
        mw = b * ow + (1 - b) * (x2 - x1)
        mh = b * oh + (1 - b) * (y2 - y1)
        t["x1"], t["y1"] = mcx - mw / 2.0, mcy - mh / 2.0
        t["x2"], t["y2"] = mcx + mw / 2.0, mcy + mh / 2.0

        v = self.vel_smooth
        t["vx"] = v * t["vx"] + (1 - v) * (cx - ocx)
        t["vy"] = v * t["vy"] + (1 - v) * (cy - ocy)

        t["conf"] = cf
        t["missed"] = 0
        t["age"] += 1
        t["shifted"] = 0.0
        t["last_seen"] = now

    def _drift(self, t):
        """漏检一帧：幽灵框按（衰减的）速度继续走一小段。

        为什么不让它冻在原地：怪在走的时候，冻住的框和实际位置越差越远
        （决策层会朝空位置走 / 空放技能）；为什么不让它一直走：防抖结束前
        速度必须收住，否则框会飘到地图另一头去。所以衰减 + 总位移上限。
        """
        w, h = self._size(t)
        cap = self.ghost_max_shift * max(w, h)
        if t["shifted"] >= cap:
            return
        dx, dy = t["vx"], t["vy"]
        t["x1"] += dx
        t["x2"] += dx
        t["y1"] += dy
        t["y2"] += dy
        t["shifted"] += (dx * dx + dy * dy) ** 0.5
        t["vx"] *= self.ghost_decay
        t["vy"] *= self.ghost_decay

    # ---------------- 同体合并 ----------------

    def _same_body(self, a, b):
        """两条轨迹是不是同一只怪。

        **保守**：只在两种情况下判"是同一只"——
          · 框重叠得厉害（IoU 达标）→ 一帧里 YOLO 对同一个怪吐了两个框；
          · 一条是幽灵框、另一条是本帧新检测到的，且新框离幽灵的预测位置
            在认领半径内 → 就是"上一条没认领成功"的那个漏网之鱼。
        两条**都是本帧检测到**的、只是离得近，不合并 —— 那可能是两只真挨着
        站着的怪，吞掉一只比多一个框更糟。
        """
        if _iou((a["x1"], a["y1"], a["x2"], a["y2"]),
                (b["x1"], b["y1"], b["x2"], b["y2"])) >= self.merge_iou:
            return True
        fresh, ghost = (a, b) if a["missed"] == 0 else (b, a)
        if fresh["missed"] != 0 or ghost["missed"] == 0:
            return False                    # 两条要么都活、要么都幽灵 → 不合并
        cx, cy = self._center(ghost)
        gx, gy = cx + ghost["vx"], cy + ghost["vy"]
        fx = (fresh["x1"] + fresh["x2"]) / 2.0
        fy = (fresh["y1"] + fresh["y2"]) / 2.0
        d = ((fx - gx) ** 2 + (fy - gy) ** 2) ** 0.5
        return d <= self._gate(ghost)

    def _dedupe(self):
        """把同体的多条轨迹合成一条：**活得久的当主体**（id 更"熟"）。"""
        keep = []
        for t in sorted(self._tracks, key=lambda x: -x["age"]):
            host = next((k for k in keep if self._same_body(k, t)), None)
            if host is None:
                keep.append(t)
            else:
                self._merge_into(host, t)
        self._tracks = keep

    @staticmethod
    def _merge_into(host, t):
        """把 t 并进 host：id/age 用 host 的，位置/状态用**更新鲜**的那条。"""
        if t["last_seen"] < host["last_seen"]:
            return                          # host 本来就更新，没什么可并的
        for k in ("x1", "y1", "x2", "y2", "vx", "vy", "missed", "shifted",
                  "last_seen"):
            host[k] = t[k]
        host["conf"] = max(host["conf"], t["conf"])

    # ---------------- 输出 ----------------

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
