"""怪物跨帧追踪：给检测框稳定的 id。

决策层要「盯住同一只怪」——单帧检测框没有 id，某帧漏检就会换目标。

**实现：运动补偿 + 同体合并 + 平滑/幽灵 + 确认闸 + 全局最优分配**：
    前几条是自己踩出来的（见下）；2026-10-11 又借了 `visual_tracking_sdk_20260920` 的
    **OC-SORT 纪律**（确认 / 低分维持身份 / 全局分配 ✓ 见 `docs/开发日志-4.md` 那一天 ✓）。
    ⚠ **不用卡尔曼** ✗（那是"视频里目标会匀速滑行"的假设 ✓；这里是游戏，怪会急停/瞬移 ✓
    「运动补偿 + 认领半径」更贴合 ✓），但**认领那一步改成全局最优** ✓（原因见第 4 条 ✓）。
    每条规则都冲着**实测踩过的现象**去的，出了问题好定位：

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

    4) **确认才输出 / 低分只维持身份 / 全局最优认领**（2026-10-11 ✓ 用户原话："**防抖置信度+
       防抖时间的配置 基本无效，同个怪物的检出框疯狂闪烁**"✓ ⇒ 照 `visual_tracking_sdk_20260920`
       的 OC-SORT 纪律补的三条 ✓ 每条都有它治的现象，见 `update` 的说明 ✓）：
       · 新轨迹**要确认**（连续 2 拍 **且** 活满 60ms ✓）才输出 ⇒ 单帧假框不再"闪一下" ✓；
       · `conf_mob` 以下的框**只能续已有轨迹、不能新建身份**（借 BYTE ✓）⇒ 置信度在阈值
         上下跳的那几拍不再"整拍查无此怪" ✓（原来幽灵规则也救不了它 ✓）；
       · 认领改成**全局最优**（`perception/assign.py` ✓）⇒ 两条轨迹不再"先到先挑"✓
         （贪心偶尔会把两条轨迹的对像互换 ⇒ id 跳一下 ⇒ 看着就是闪 ✓）。
    ⚠⚠ 那两个**界面旋钮**（`debounce_conf` / `debounce_ms`）只管"**消失后还留多久**"✗
      —— 它们**不决定"这一拍显不显示"** ✓（显示由上面的**确认闸**管 ✓ 别再把两者混起来 ✗）。

对外接口：`update(detections, low_detections=()) -> [Mob]`、`kill(id)`；
`Mob.missed > 0` 仍然表示「幽灵框（这一帧没检测到，靠防抖留着）」—— 决策层就靠它防空放技能。
`Mob.peak_conf`（2026-10-11 新增 ✓）是这条轨迹**历史最高**置信度，给界面角标看"它到底多确信"✓。
"""
import time

from perception.assign import BIG as _ASSIGN_BIG, hungarian as _assign_pairs
from perception.world_state import Mob

#: ⭐⭐ 确认闸（2026-10-11 ✓ 治"框疯狂闪烁"✓ 借 OC-SORT 的 `lock_confirmations` 思路 ✓）：
#:  · `CONFIRM_HITS` —— 一条新轨迹**连续被看到几拍**才有资格显示；
#:  · `CONFIRM_MS`   —— **并且**它得**活满这么久**（时间地板 ✓ **不是**帧数 ✗）。
#:    ⚠⚠ 时间地板不是摆设：60fps 下两帧只隔 16ms ⇒ 连着的两次假检出往往**同源**
#:      （同一个噪声块 ✓ 或同一次标注错误 ✓）⇒ **光数帧数拦不住** ✓ —— 这正是 OC-SORT
#:      各处都写成 `CONFIRMATIONS + CONFIRMATION_S` 两个数的原因 ✓（`branch_tracker.py:34-51` ✓）。
CONFIRM_HITS = 2
CONFIRM_MS = 60.0

#: ⭐⭐ **相机位移的"离谱线"**（像素 ✓ 用户 2026-10-13 ✓ 见 `MobTracker._shift_cam` ✓）：
#: 两帧之间相机挪超过这么多 ⇒ 当成**传送 / 换图**（不是走路 ✓）⇒ 这一帧**不挪**、只重锚 ✓。
#: 依据：正常走路时画面每帧滚十几像素（卡帧顶多两三倍 ✓），几百像素只可能是传送 ✓。
CAM_MAX_STEP = 200.0

#: ⭐ **低分档的比例**（BYTE ✓）：`conf_mob × 它` 以下不进来 ✓、以上到 `conf_mob` 之间
#: 只用来**续住已有轨迹**（`split_dets` 用它分档 ✓ 只此一处 ✓）。
MOB_LOW_CONF_RATIO = 0.5

#: 速度方向代价（借 OC-SORT 的 ORU ✓ `association.py:251-284` ✓）：候选在**速度方向的
#: 反方向**上 ⇒ 分数打折（最多打掉这么多 ✓）。权重**故意小**：它只该在"两个候选差不多"时
#: 起作用 ✓ 不该推翻明确的 IoU/距离判据 ✗。
MOTION_DIR_WEIGHT = 0.15

#: 活得 ≤ 这么多拍、且**从没确认过**就没了 ⇒ 记一条"闪一下"（打点用 ✓ 见 `_note_flick` ✓）。
FLICK_MAX_AGE = 2

#: "闪一下"的现场**最多留几条**（给日志/界面看 ✓ 有上限 ⇒ 不会被刷爆 ✓）。
FLICK_KEEP = 8

#: 全局分配时**每条轨迹只考虑自己分数最高的几个框**（规模闸 ✓ 见 `_assign` ✓）：
#: 30 只怪的全矩阵实测 7.4 ms ✗（每帧预算里不能忍 ✓）⇒ 留 6 个之后是 0.5 ms 量级 ✓。
ASSIGN_TOP_K = 6

#: ⭐⭐⭐⭐ **GMC（全局运动补偿）兜底**（用户 2026-10-13 ✓ 原话："**多目标追踪算法能否缓解这个
#: 问题？**"＋"**全部做完**"✓）—— 这就是 MOT 里对"**移动相机**"的标准做法（BoT-SORT 的 `gmc.py`
#: 就是干这个 ✓），借 vendored SDK 的**多数派运动**（`visual_tracking_sdk_20260920/liescd/
#: ocsort_tracker.py:601-613` 的 `_majority_motion` ＋ `edge_identity.py:133-154` 的 consensus ✓）：
#:   · 样本 = **同一只怪**这一帧与**上一帧**的**画面位移**（**原始检测** ✓ 不含任何补偿 ✓）
#:     —— 大多数怪自己没怎么动 ⇒ 中位数 ≈ 画面整体平移（= 相机滚动 ✓）；
#:   · 至少要 `GMC_MIN_TRACKS` 个样本（SDK 也这么卡 ✓）：样本太少时中位数会被"那一只在动的怪"
#:     整个带走 ✗（3 只里 2 只在跑 ⇒ 中位数就是跑的那只 ✓）；
#:   · 再用 `GMC_INLIER_PX` 剔一次离群（在跑的 / 被打飞的那只 ✓），剔完还要够数 ✓；
#:   · ⚠ 只用**高分档**的配对（低分档是"擦着阈值"的框 ✓ 位置噪声更大 ✗）。
#: 用途：外部锚（小地图黄点世界坐标）拿不到的那几拍，**用它顶上** ✓ —— 见 `_shift_anchor` ✓。
GMC_MIN_TRACKS = 3
GMC_INLIER_PX = 12.0
#: ⭐ **样本不够时怎么办**：**沿用上一次的估计并衰减**（`× 它`），衰减到 < 0.5px 就丢掉 ✓。
#: 为什么不能直接清成 `None` ✗：**漏帧那几拍本来就没有配对**（幽灵期 ✓）—— 而"幽灵框得跟着
#: 画面一起走"恰恰是最需要补偿的时候 ✗（不跟 ⇒ 屏幕上那个幽灵框停在原地 ✗ ⇒ Agent 朝错位置
#: 走 ✓ 而这正是用户报的"丢失目标"✓）。衰减是为了"动完了要停"：一直照旧挪会自己走飞 ✗。
GMC_HOLD_RATIO = 0.7

#: ⭐⭐⭐ **跨漏帧重关联（"补桥"）**（用户 2026-10-13 ✓ 同上）—— 借 SDK `kalmanfilter.py:425-475`
#: （unfreeze 的虚拟轨迹）与 OC-SORT 的 OCR 再匹配思路 ✓：
#: 这一拍还没配上的**已确认**轨迹，拿剩下的正常框**再配一次**（放宽 `MATCH_RELAX` 倍 ✓）。
#: 目的只有一个：**同一个 id 回来** ✓ —— 不这么做时，"漏两拍又出现"必然换新 id ✗，
#: 而用户看到的"疯狂闪烁"里就有它 ✓（现场：`age=1 / missed≈80` ⇒ 一次都没配上 ✓）。
#: ⚠ 放宽有上限：太松会把**旁边另一只怪**认成它 ✗（那比换 id 更糟：目标直接跳错 ✓）。
MATCH_RELAX = 1.8

#: ⭐⭐ 怪链路的「**同 ID 跳变隔离**」常量（用户 2026-10-13 ✓ 同上）—— 借 SDK
#: `branch_tracker.py:106-116` 的 `ASSOCIATION_QUARANTINE_FRAMES = 3` 思路 ✓，
#: ⚠⚠ 但**阈值必须相对"认领半径"** ✗ 不能照抄 SDK 的绝对像素（`JUMP_MIN/MAX = 24/96`）：
#:   SDK 的关联是 IoU + 卡尔曼预测 ✓ 没有固定像素闸；我们的是 `max(10, 0.9 × 框长边)`
#:   （40px 的怪 ≈ 36px ✓）⇒ **96px 的绝对跳变根本走不到"配对上"** ✗
#:   ⇒ 那段判据会变成**死代码**（本轮实测就是：`diag.quarantine` 恒 0 ✓——记下来别再踩 ✗）。
#:   ⇒ 口径 = "**这一拍挪得比认领半径的 `0.6` 倍还多**" ⇒ 先别认，要**连着 3 拍都指向同一处**才认 ✓。
#: 治的是"**id 没变、但框跳到了另一边**"（两只贴着的怪整体换身 ✓）——
#: ⚠ 真·快速移动（掉平台 / 被击退 ✓）3 拍后照样认下 ✓ —— 不拦"真的在动" ✗；
#: ⚠ 万一那 3 拍里检测自己新建了一条 ✓ 还有 `_dedupe` 兜（同体合并保留老 id ✓）。
MOB_QUARANTINE_JUMP_RATIO = 0.6   # 可疑位移 = 认领半径 × 它（相对尺 ✓ 无量纲 ✓）
MOB_QUARANTINE_HOLD_PX = 24.0     # "还是同一处"的容差（绝对像素够用：比认领半径小 ✓）
MOB_QUARANTINE_FRAMES = 3         # 连着这么多拍都指向同一处才认 ✓


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

        #: 防抖置信度：这只怪**曾经达到过**它 ⇒ 消失后按「防抖时间」残留（幽灵框 ✓）。
        #: ⭐ 2026-10-01 起判据是**历史最高置信度**（`peak_conf`），不是最后一次 ✓。
        #: ⚠⚠ 2026-10-13：「**锁定怪物丢框防抖时间**」那套（`locked_id` / `lock_ghost_ms` /
        #:   `Mob.lock_ghost`）**整套删掉了** ✓（用户原话："移除「锁定怪物丢框防抖时间」的配置与
        #:   功能（**这是重复功能**）"✓—— 它本来就是"只对锁定那只、把防抖时间换个数"✓
        #:   与这一格重复 ✓）⇒ 现在**一条链**：见下 `debounce_*` ✓。
        #: ⭐⭐⭐ **相机锚 + 这一帧的相机位移**（用户 2026-10-13 ✓ 见 `_shift_cam` ✓）：
        #:   `cam` = 上一次吃的可信相机（`世界 = 画面 + 相机` ✓ 由调用方灌 ✓ 见 `update` ✓）；
        #:   `cam_dx/dy` = 这一帧挪了多少（打点 / 排查用 ✓）。
        self.cam = None
        self.cam_dx = self.cam_dy = 0.0
        self.debounce_conf = 0.0
        self.debounce_ms = 0.0      # 防抖时间（毫秒）
        self._tracks = []       # 每条是 dict，字段见 _new_track
        self._next_id = 1
        #: ⭐⭐⭐⭐ GMC 兜底（见 `GMC_MIN_TRACKS` ✓ 用户 2026-10-13 ✓）：
        #: `gmc` = 上一次算出的"**画面整体平移**"（`None` = 样本不够 / 还没算过 ⇒ 那几拍照旧
        #: 不挪 ✓ 宁缺勿错 ✓）；`_gmc_samples` = **这一帧**收集到的原始位移样本（`_absorb` 写、
        #: `update` 末尾结算并清 ✓ 一个事实一个写者 ✓）；`_frame_no` 只在 `update` 里 +1 ✓
        #: —— 用来判"这只怪**上一帧也见过**"（只有连续两帧都见到，它的位移才是"这一帧"的 ✓）。
        self.gmc = None
        self._gmc_samples = []
        self._frame_no = 0
        #: ⭐⭐ **这一拍的账**（2026-10-11 ✓ 给 `behavior.log` 用 ✓ 见 `drain_diag`）：
        #:   新建 / 认领 / 低分续命 / 幽灵 / **没确认就没了（= 闪一下 ✓）** 各几条。
        #:   ⚠ 它是**读数**：只在 `update` 里重置并累加 ✓ 别处不许写 ✗（R1 一个事实一个写者 ✓）。
        self.diag = {"created": 0, "matched": 0, "ghost": 0, "low": 0,
                     "dropped_young": 0, "unconfirmed": 0}
        #: 「闪一下」的现场（最多留 `FLICK_KEEP` 条 ✓）—— 由 `drain_flick()` 取走打点 ✓。
        #: ⚠ 为什么要有它：用户报的"**同个怪物的检出框疯狂闪烁**"有两种完全不同的成因
        #:   （① 同一个 id 闪 = 检测不稳 ✓；② **每帧换 id** = 认领/新建有问题 ✓）✗
        #:   ⇒ 不把 id 与存活拍数打出来，**只能靠猜** ✓（沿用 SKILL 315 那条：
        #:     只在日志里 = 等于没有 ✓ 但至少要有 ✓）。
        self.flick = []

    # ---------------- 轨迹 ----------------

    @staticmethod
    def _new_track(tid, x1, y1, x2, y2, conf, now):
        return {
            "id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf,
            #: ⭐ **历史最高置信度**（用户 2026-10-01 ✓）：**界面角标**用"它到底多确信"
            #:   （只认**高分档** ✓ 见 `_absorb` ✓）。
            "peak_conf": conf,
            #: ⭐⭐⭐ **两个档位都算的最高置信度** → **防抖判据读它**（用户 2026-10-13 ✓
            #:   原话："只要曾经怪物到过该置信度阈值…"✓）；与 `peak_conf` 分开的原因见 `_absorb` ✓。
            "any_conf": conf,
            "missed": 0, "age": 1, "vx": 0.0, "vy": 0.0,
            "shifted": 0.0,          # 幽灵外推累计走了多远（到上限就不走了）
            "last_seen": now,
            #: ⭐ **确认闸的两个依据**（2026-10-11 ✓ 见 `CONFIRM_HITS` / `CONFIRM_MS` ✓）：
            #: 出生时刻（算"活满 60ms 没有"✓）与"确认过了吗"（**只涨不落** ✓
            #: —— 一条轨迹一旦确认过就不再回退 ✓ 否则它会因为偶尔漏拍而重新"闪" ✓）。
            "born": now, "confirmed": False,
            #: ⭐ 2026-10-13 三样（用户"全部做完"✓ 各自见常量）：
            #:  · `det` / `seen_frame` —— 上一次**原始检测中心**与那一拍的帧号 ⇒ GMC 采样用
            #:    （只有"上一帧也见过"的配对才算 ✓ 见 `_absorb` ✓）；
            #:  · `_q_box` / `_q_n` —— 同 ID 跳变隔离的计数（见 `update` 第 1 步 ✓）。
            "det": None, "seen_frame": -1,
            "_q_box": None, "_q_n": 0,
        }

    @staticmethod
    def _center(t):
        return ((t["x1"] + t["x2"]) / 2.0, (t["y1"] + t["y2"]) / 2.0)

    @staticmethod
    def _size(t):
        return (t["x2"] - t["x1"], t["y2"] - t["y1"])

    def _shift_anchor(self, player_world):
        """**把已有轨迹整体挪进"角色的相对系"**（用户 2026-10-13 ✓ 原话："**不应该**去算相机的
        位移，而是应该**根据角色的世界坐标**去算**相对位移、相对速度**"✓）。

        ⚠⚠ **为什么是"角色的世界坐标"而不是"相机的位移"**（用户 2026-10-13 ✓ 追问）：
          「同一只怪」这件事的本质是"**怪相对角色的位移/速度**" ✓ —— 角色自己的世界坐标一变
          （走过去 / 被推 / 相机跟不动 ✓），怪在**画面**上的位置也会跟着变 ✗；而相机那份估算
          会被冻结 / 拒收（`camera()` 的 `cam_resid` 闸 ✓）⇒ 拿它当锚不稳 ✗。
          ⇒ 锚 = **角色的世界坐标** ✓（由调用方给：`画面 + 相机` ✓ 与全仓的
          `world = screen + cam` 同一口径 ✓ 见 `PlayerTracker._cam_of` ✓）；
          这一帧把它与上一帧比 ⇒ 得到**角色的世界位移** ✓ ⇒ 把老轨迹**反着挪**回去 ✗
          ⇒ 之后 `_score` 比的才是"**怪自己**相对角色动了多少" ✓（`vx/vy` 也顺带成了
          **相对速度** ✓ —— 它由连续两帧的**相对位置**算出 ✓）。
        ⚠ 下面这个函数名/字段名保留了 `cam_*` 的历史叫法（`cam_dx/cam_dy` = 这一帧挪了多少 ✓）
          —— 它算的就是"画面该反向挪多少" ✓ 与角色世界位移**逐字相等** ✓（世界 = 画面 + 相机 ✓）。

        ⚠⚠ **为什么非有不可**：怪框是**画面坐标** ✓，而"世界 = 画面 + 相机"（见
          `PlayerTracker.camera()` ✓）⇒ 角色一走 / 相机一滚，**同一只怪在画面上就平移** ✓
          （世界没动、画面动了 ✓）。而 `_score` 那道闸只有 `max(10, 0.9 × 框长边)`——
          一只 40px 的怪 ≈ 36px ✗ ⇒ 相机每帧滚十几像素、两拍就顶破 ✗ ⇒ **认领失败 ⇒ 同一个怪
          换一个新 id** ✓✓（正是用户看到的那两条现象：**框立刻消失 + 再出现是新 id** ✓）。
          ⇒ 认领之前先把轨迹按**相机位移**反着挪一次（`画面 = 世界 − 相机` ✓ 见 `_cam_of` ✓），
          这样两边都在"这一帧的画面系"里，`_score` 比的才是**怪自己**的运动 ✓。

        ⚠ 相机拿不到（`None` ✓ 没定位 / 刚起 / 小地图没读出黄点 ✓）⇒ **这一帧不挪**（= 老行为 ✓
          宁缺勿错 ✓）；⚠ 位移大得离谱（传送 / 换图 ✓ 超过 `CAM_MAX_STEP`）⇒ **不挪、只重锚** ✓
          （硬挪会把一堆轨迹甩到错地方 ✗）。
        ⚠ 挪的是**位置**；`vx/vy`（画面速度 ✓）**不动** ✗ —— 它本来就含相机速度，用来自我外推
          下一帧的画面位置**正是对的** ✓（`_score` 的预测项直接用 ✓）。
        """
        self.cam_dx = self.cam_dy = 0.0
        try:
            ok = player_world is not None and len(player_world) >= 2
        except TypeError:
            ok = False
        if not ok:
            self.cam = None                  # 拿不到 ⇒ 忘掉旧锚（别拿发霉的值硬挪 ✗）
            # ⭐⭐⭐⭐ **GMC 兜底**（用户 2026-10-13 ✓ 见 `GMC_MIN_TRACKS` ✓）：
            #   外部锚（小地图黄点世界坐标）拿不到的那几拍，用**上一次算出的"画面整体平移"
            #   （多数派运动 ✓）**顶上 ✓ —— 原来这里**直接 `return`**（这一帧**完全不挪**）✗
            #   ⇒ 认领闸（`0.9 × 框长边` ≈ 36px）被相机滚动顶破 ⇒ **框消失 + 换新 id** ✗✓
            #   （现场那批 `age=1 / missed≈80` 的"闪一下"就是它 ✓）。
            #   ⚠ 样本不够 / 还没算过 ⇒ 还是**不挪** ✓（宁缺勿错 ✓ 老行为 ✓）。
            _g = getattr(self, "gmc", None)
            if _g is None:
                return
            dcx, dcy = float(_g[0]), float(_g[1])
            if (abs(dcx) > CAM_MAX_STEP or abs(dcy) > CAM_MAX_STEP
                    or (not dcx and not dcy)):
                return                   # 离谱（传送 / 换图 ✓）⇒ 不挪、只重锚 ✓
            for t in self._tracks:
                t["x1"] -= dcx
                t["x2"] -= dcx
                t["y1"] -= dcy
                t["y2"] -= dcy
            self.cam_dx, self.cam_dy = dcx, dcy
            # ⚠ 先记成一次性标记 ✓（这一拍 `diag` 还没重置 ✓ 见 `update` 里那段说明 ✓）
            self._gmc_used = True
            return
        cx, cy = float(player_world[0]), float(player_world[1])
        if self.cam is None:
            self.cam = (cx, cy)              # 第一帧：只记锚 ✓（没有"上一帧"可比 ✓）
            return
        dcx, dcy = cx - self.cam[0], cy - self.cam[1]
        self.cam = (cx, cy)
        if abs(dcx) > CAM_MAX_STEP or abs(dcy) > CAM_MAX_STEP:
            return                           # 传送 / 换图 ⇒ 重锚、这一帧不挪 ✓
        if not dcx and not dcy:
            return
        for t in self._tracks:
            t["x1"] -= dcx
            t["x2"] -= dcx
            t["y1"] -= dcy
            t["y2"] -= dcy
        self.cam_dx, self.cam_dy = dcx, dcy

    def _gmc_estimate(self):
        """这一帧的「**画面整体平移**」估计 —— `None` = 样本不够（宁缺勿错 ✓）。

        样本是"同一只怪这一帧与上一帧的**原始画面位移**"（`_absorb` 里收集 ✓ 见那边 ✓），
        这里只做两件事（口径见 `GMC_MIN_TRACKS` ✓ 借 SDK 的 `_majority_motion` ✓）：
          · **中位数**（多数派 ✓）：大多数怪自己没怎么动 ⇒ 中位数 ≈ 相机滚动 ✓；
          · **再剔一次离群**（在跑的 / 被打飞的那只 ✓）后重新取中位数，剔完还得够数 ✓。
        ⚠ 只用**高分档**的配对（低分档是"擦着阈值"的框 ✓ 位置噪声更大 ✗ 见 `_absorb` ✓）。
        """
        smp = self._gmc_samples
        self._gmc_samples = []
        if len(smp) < GMC_MIN_TRACKS:
            # 这一拍没得算（配对太少 / 全在漏帧 ✓）⇒ **沿用上一次并衰减** ✓（见 `GMC_HOLD_RATIO` ✓）
            g = getattr(self, "gmc", None)
            if g is None:
                return None
            nx = float(g[0]) * GMC_HOLD_RATIO
            ny = float(g[1]) * GMC_HOLD_RATIO
            if abs(nx) < 0.5 and abs(ny) < 0.5:
                return None
            return (nx, ny)

        def _med(vals):
            s = sorted(vals)
            return s[len(s) // 2]

        mx, my = _med([v[0] for v in smp]), _med([v[1] for v in smp])
        inl = [v for v in smp
               if abs(v[0] - mx) <= GMC_INLIER_PX and abs(v[1] - my) <= GMC_INLIER_PX]
        if len(inl) < GMC_MIN_TRACKS:
            return None
        return (_med([v[0] for v in inl]), _med([v[1] for v in inl]))

    def _gate(self, t, relax=1.0):
        """认领半径：和框的尺寸挂钩（`relax` = 给"补桥"那一趟放宽的倍数 ✓ 见 `MATCH_RELAX` ✓）。

        为什么不用固定像素：小框（远处的怪）动几像素 IoU 就崩，大框能跑远点 ——
        阈值跟着框走，各种距离下才是同一套手感。
        ⚠ `relax` **只给补桥那一趟**用（`update` 1c ✓）：正常那一趟永远 1.0 ✓
          —— 别在这儿写死放大 ✗（那等于把"两只贴着的怪"直接放到一起比 ✓ 更容易换错 ✓）。
        """
        w, h = self._size(t)
        return max(10.0, self.gate_ratio * max(w, h)) * float(relax)

    def _score(self, t, d, relax=1.0):
        """这个检测框能不能认成这条轨迹：能认返回 >0，不能返回 -1。

        分数只用来**排序**，不参与别处计算：IoU 达标的（score = IoU）永远排在
        「只满足距离」的前面 —— 有真正重叠的框时，距离近不该抢走它。

        ⚠ `relax`（默认 `1.0` = 老口径 ✓）只放宽**认领半径**那一条 ✗，**不放宽 IoU 档** ✓
          —— 补桥那一趟（`MATCH_RELAX` ✓）要的是"漏了几拍之后还认得回来" ✓，
          不是"允许随便配" ✓。
        """
        iou = _iou((t["x1"], t["y1"], t["x2"], t["y2"]), d[:4])
        if iou >= self.min_iou:
            base = iou
        else:
            cx, cy = self._center(t)
            pcx, pcy = cx + t["vx"], cy + t["vy"]   # 预测位置（位置 + 速度外推）
            dx = (d[0] + d[2]) / 2.0 - pcx
            dy = (d[1] + d[3]) / 2.0 - pcy
            dist = (dx * dx + dy * dy) ** 0.5
            gate = self._gate(t, relax)
            if dist > gate:
                return -1.0
            # 距离越近分数越高，但乘 0.99 压在 min_iou 之下（排序上永远次于 IoU 达标）
            base = self.min_iou * 0.99 * (1.0 - dist / gate)
        # ⚠ 年龄偏好**必须小到不会翻过 `min_iou`**（`1e-5 × 最多 50 拍` = 5e-4 ✓
        #   而"只满足距离"那一档最高 0.297 ✓ ⇒ 加完仍 < 0.3 ✓ 那条不变量还在 ✓）
        return base * self._dir_factor(t, d) + 1e-5 * min(int(t.get("age", 1) or 1), 50)

    def _dir_factor(self, t, d):
        """候选相对**上一拍中心**的位移方向，与这条轨迹速度方向的一致度：`[1-W, 1]` ✓。

        借 OC-SORT 的 ORU（`_vendor/ocsort/association.py:251-284` ✓ 把"方向差"加进代价 ✓）：
        怪在往右走时，右边那个候选比左边那个更"顺" ✓ —— 只在两个候选本来就接近时起作用 ✓
        （权重 `MOTION_DIR_WEIGHT = 0.15` 很小 ✓ 不该推翻明确的 IoU / 距离判据 ✗）。
        ⚠ 没有速度（刚认领 / 静止 ✓）⇒ 恒 `1.0`（完全不参与 ✓ 老行为 ✓ 不许抛 ✗）。
        """
        sx, sy = float(t.get("vx", 0.0) or 0.0), float(t.get("vy", 0.0) or 0.0)
        speed = (sx * sx + sy * sy) ** 0.5
        if speed <= 1e-6:
            return 1.0
        try:
            cx, cy = self._center(t)
            ux = (d[0] + d[2]) / 2.0 - cx
            uy = (d[1] + d[3]) / 2.0 - cy
        except (TypeError, IndexError, ValueError):
            return 1.0
        nu = (ux * ux + uy * uy) ** 0.5
        if nu <= 1e-6:
            return 1.0
        cos = (ux * sx + uy * sy) / (nu * speed)
        cos = max(-1.0, min(1.0, cos))
        return 1.0 - MOTION_DIR_WEIGHT * (1.0 - cos) / 2.0

    # ---------------- 每帧 ----------------

    def update(self, detections, low_detections=(), player_world=None):
        """这一帧：认领 → 低分续命 → 漂移 → 新建 → 合并 → 清理 → **只输出确认过的** ✓。

        ⭐⭐ 2026-10-11 三条（用户原话："**防抖置信度+防抖时间的配置 基本无效，同个怪物的
        检出框疯狂闪烁**"✓ ⇒ 照 `visual_tracking_sdk_20260920` 的 OC-SORT 纪律补 ✓）：
          ① **全局最优认领**（`_assign` ✓）：原来是"按轨迹顺序贪心、老轨迹先挑"✗ ——
             贪心会"先到先挑"⇒ 两条轨迹偶尔**互换对像** ⇒ id 跳一下 ⇒ 看着就是闪 ✓；
          ② **新轨迹要确认才输出**（`_confirmed` ✓）：原来新轨迹**当拍就输出**✗ ⇒
             单帧假框 = 闪一下 ✓（时间地板见 `CONFIRM_MS` 的说明 ✓）；
          ③ **低分框只续命、不新建身份**（`low_detections` ✓ BYTE ✓）：怪框 conf 在
             `conf_mob` 上下跳的那几拍，原来**整拍查无此怪**✗ ⇒ 幽灵规则也救不了
             （它要求"曾经达到过 防抖置信度"✓）⇒ 现在由低分框把轨迹续住 ✓。

        ⚠⚠ **两个旋钮的边界**（回应用户那句"配置基本无效"✓）：`debounce_conf` /
          `debounce_ms` 只管"**消失后还留多久**"✗，**不决定"这一拍显不显示"** ✓
          —— 显不显示由**确认闸**说了算 ✓。想"更不容易闪"该调的是"看见"那一侧，
          不是"消失"那一侧 ✓（这两件事以前被混在一起了 ✓）。
        """
        now = time.monotonic()
        self._frame_no = int(getattr(self, "_frame_no", 0) or 0) + 1
        low_detections = list(low_detections or ())
        # ⭐⭐⭐ **先按"角色的世界位移"把轨迹挪进这一帧的相对系**（用户 2026-10-13 ✓
        #   见 `_shift_anchor` ✓："**不应该**去算相机的位移，而是应该**根据角色的世界坐标**
        #   去算**相对位移、相对速度**"✓）
        #   ⚠ 必须在**认领之前** ✗ —— 挪晚了 `_score` 比的还是"两个不同系"的位置 ✓。
        self._shift_anchor(player_world)
        used = [False] * len(detections)
        self.diag = {"created": 0, "matched": 0, "ghost": 0, "low": 0,
                     "dropped_young": 0, "unconfirmed": 0,
                     # ⭐ 2026-10-13 三条（用户"全部做完"✓）：GMC 兜底用上了 / 补桥救回一条 id /
                     #   跳变隔离拦下一拍（见各自常量 ✓）—— 都是**读数**，给 behavior.log 看 ✓。
                     "gmc": 0, "bridge": 0, "quarantine": 0}
        # ⚠⚠ 「这一拍用了 GMC 兜底吗」必须在**上面那次移位里就记下来** ✓ —— 移位发生在
        #   `used`/`diag` **之前** ✗ ⇒ 记进 `diag` 会被下面这行**当场抹掉** ✓（本轮用例
        #   实测：`diag.gmc` 恒 0 ✓ 而其实挪了 ✓）⇒ 用这个一次性标记，`_shift_anchor` 里置上、
        #   这里搬到 `diag` ✓（一处写一处搬 ✓ 不新造第二份状态 ✓）。
        if getattr(self, "_gmc_used", False):
            self.diag["gmc"] += 1
            self._gmc_used = False

        # 0) 先记账：这一拍谁都没认领到（认领成功的会被 `_absorb` 归零 ✓）
        for t in self._tracks:
            t["missed"] += 1

        # 1) 认领（**全局最优** ✓）：代价 = `1 - 分数`，配不上的行留空 ✓
        #    ⭐⭐⭐ 2026-10-13 起：这里多了**同 ID 跳变隔离**（用户"全部做完"✓ 见
        #      `MOB_QUARANTINE_*` ✓ 借 SDK `branch_tracker.py:106-116` ✓）——
        #      治"**id 没变、但框跳到了另一个更连续的位置**"（两只贴着的怪整体换身 ✓）：
        #      这一拍**先别认**（让它按 coast 走 ✓），要**连着 3 拍都指向同一处**才认 ✓。
        #      ⚠ 真·快速移动（掉平台 / 被击退 ✓）3 拍后照样认下 ✓ —— 不拦"真的在动" ✗；
        #      ⚠ 万一那 3 拍里检测自己新建了一条 ✓ 还有 `_dedupe` 兜（同体合并保留老 id ✓）。
        for ti, di in self._assign(self._tracks, detections):
            t = self._tracks[ti]
            d = detections[di]
            _ocx = (t["x1"] + t["x2"]) / 2.0
            _ocy = (t["y1"] + t["y2"]) / 2.0
            _ncx = (d[0] + d[2]) / 2.0
            _ncy = (d[1] + d[3]) / 2.0
            _mv = ((_ncx - _ocx) ** 2 + (_ncy - _ocy) ** 2) ** 0.5
            # ⚠ 判据是**相对**的（认领半径 × `MOB_QUARANTINE_JUMP_RATIO` ✓ 见常量说明：
            #   绝对像素在这儿是死代码 ✗ —— 认领闸本身就只有 0.9×框长边 ✓）
            _jump_lim = max(10.0, MOB_QUARANTINE_JUMP_RATIO * self._gate(t))
            if _mv > _jump_lim:
                _qb = t.get("_q_box")
                if (_qb is not None
                        and ((_ncx - _qb[0]) ** 2
                             + (_ncy - _qb[1]) ** 2) ** 0.5 <= MOB_QUARANTINE_HOLD_PX):
                    t["_q_n"] = int(t.get("_q_n", 0) or 0) + 1
                else:
                    t["_q_n"] = 1
                t["_q_box"] = (_ncx, _ncy)
                if int(t["_q_n"]) < MOB_QUARANTINE_FRAMES:
                    self.diag["quarantine"] += 1
                    continue                  # ⛔ 这一拍不认 ⇒ 它按 coast 走 ✓（下一拍再判 ✓）
            else:
                t["_q_n"], t["_q_box"] = 0, None
            used[di] = True
            self._absorb(t, d, now)
            self.diag["matched"] += 1

        # 1b) **低分框**只喂给"这一拍还没被正常框认领"的轨迹 ✓（⛔ 不能新建身份 ✗ = BYTE ✓）
        if low_detections:
            left = [i for i, t in enumerate(self._tracks) if t["missed"] > 0]
            if left:
                for k, dj in self._assign([self._tracks[i] for i in left],
                                          low_detections):
                    self._absorb(self._tracks[left[k]], low_detections[dj],
                                 now, low=True)
                    self.diag["low"] += 1

        # 1c) ⭐⭐⭐ **跨漏帧重关联（"补桥"）**（用户 2026-10-13 ✓ 见 `MATCH_RELAX` ✓）：
        #   这一拍还没配上的**已确认**轨迹，拿**剩下的正常框**再配一次（放宽 `MATCH_RELAX` 倍 ✓）
        #   —— 目的只有一个：**同一个 id 回来** ✓。不补这一刀时，"漏两拍又出现"必然换新 id ✗
        #   （现场 `age=1 / missed≈80` 那批：一次都没配上 ⇒ 只能新起一条 ✓ 用户看到的就是"闪"✓）。
        #   ⚠ 只给**确认过**的轨迹（`confirmed` ✓）：没确认过的本来就不该有身份 ✗（防噪声复活 ✓）；
        #   ⚠ 必须排在 1b（低分续命）之后 ✗ —— 低分框是"续命"用的 ✓ 不该抢这一趟的正经框 ✓。
        _free = [i for i, _d in enumerate(detections) if not used[i]]
        #   ⚠⚠ **正在跳变隔离里的不计入** ✗（本轮实测踩到 ✓ 记下来）：隔离那一拍会把配对
        #     `continue` 掉 ✗，而这一趟补桥用的是**放宽后的闸**（`MATCH_RELAX` ✓）⇒ 它立刻
        #     又把同一对配了回来 ⇒ **隔离形同虚设** ✓（用例当场红 ✓）⇒ 必须在这里排掉 ✓。
        _lost = [t for t in self._tracks
                 if t["missed"] > 0 and t["confirmed"] and int(t.get("_q_n", 0) or 0) == 0]
        if _free and _lost:
            for k, dj in self._assign(_lost, [detections[i] for i in _free],
                                      relax=MATCH_RELAX):
                t = _lost[k]
                di = _free[dj]
                used[di] = True
                self._absorb(t, detections[di], now)
                self.diag["bridge"] += 1

        # 2) 还没认领到的 ⇒ 幽灵外推（同老行为 ✓ 只是挪到"低分续命"之后 ✓
        #    —— 低分框已经认下它的，就不该再被推走 ✓）
        for t in self._tracks:
            if t["missed"] > 0:
                self._drift(t)

        # 3) 没被认领的**正常**框 → 新轨迹（⚠ **低分框永远不新建** ✗ = 第 ③ 条 ✓）
        for i, d in enumerate(detections):
            if not used[i]:
                x1, y1, x2, y2, cf = d
                _nt = self._new_track(self._next_id, x1, y1, x2, y2, cf, now)
                # ⭐⭐⭐ **到过「防抖置信度」⇒ 出生那一刻就算数**（用户 2026-10-13 ✓ 原话：
                #   "只要曾经怪物到过该置信度阈值…"✓）—— 见 `_confirmed` 里那条短路 ✓。
                #   ⚠ 必须在**这儿**也判一次 ✗：不判的话这条新轨迹要等下一拍 `_absorb` 才确认
                #     ⇒ "只被看见一帧"的怪**一个框都不出** ✗（真机日志里 429 条 `mob_flick`
                #     全是这一档 ✓）⇒ 用户看到的"框一闪一闪"里那一半就是它 ✓。
                if self._confirmed(_nt, now):
                    _nt["confirmed"] = True
                self._tracks.append(_nt)
                self._next_id += 1
                self.diag["created"] += 1

        # 4) 同体合并：把"其实是同一只怪"的多条并成一条（一串框的元凶）
        self._dedupe()

        # 5) 清理长期漏检的轨迹。
        # 用时间戳（和防抖同单位）：漏检时间超过「防抖时间 + 500ms 缓冲」才清理。
        # 不能用 max_age 帧数 —— 帧数 × fps 随帧率变，高帧率时防抖还没到期
        # 轨迹就被清掉了（防抖 1000ms ≈ 27 帧，但 max_age=15 帧 ≈ 0.5s 就清理）。
        # ⭐ 顺手记"**闪一下**"：活得短、又从没确认过 ⇒ 记一条现场（给日志/界面 ✓ 见 `flick`）
        # ⚠ 2026-10-13：「锁定怪物丢框防抖」删了 ⇒ 保留窗口就**只有一个**（防抖时间 ✓）
        #   （原来要取 `max(debounce_ms, lock_ghost_ms)` ✗ 见版本历史 ✓）。
        keep_ms = self.debounce_ms + 500.0
        keep = []
        for t in self._tracks:
            if (now - t["last_seen"]) * 1000.0 <= keep_ms:
                keep.append(t)
            elif not t["confirmed"]:
                self.diag["dropped_young"] += 1
                self._note_flick(t, now)
        self._tracks = keep

        # 6) 输出：**确认过的** + 防抖期内的幽灵。
        #    合并之后每条轨迹最多出一个框 —— id 也不会重复。
        out = []
        for t in self._tracks:
            if not t["confirmed"]:
                # 还没攒够"我看见它了" ⇒ **一个字都不输出** ✓（这就是治闪的那道闸 ✓）
                self.diag["unconfirmed"] += 1
                continue
            if t["missed"] == 0:
                out.append(self._to_mob(t))
                continue
            _gap_ms = (now - t["last_seen"]) * 1000.0
            # ⚠ 2026-10-13：「锁定怪物丢框防抖」那条分支（`locked_id` / `lock_ghost_ms` /
            #   `lock_ghost` ✓）**删了** ✓ —— 与下面这条重复 ✓（用户："这是重复功能"✓）。
            # ⭐ 防抖判据 = 「**曾经达到过**」+「**窗口内**」（用户 2026-10-01 ✓ 2026-10-13 修订 ✓）：
            #   · 「曾经达到过」看 `any_conf` ✓（**两个档位都算** ✓ 见 `_absorb` ✓ ——
            #     2026-10-13 前是 `peak_conf`（只算高分档 ✗）⇒ 配得比检出阈值低的
            #     「防抖置信度」**永远不生效** ⇒ 框一闪一闪 ✓ 现场就是它 ✓）；
            #   · 「窗口内」看 `missed`/`last_seen` ✓ ⇒ 检出框**再次出现**时 `_absorb` 会把
            #     `missed` 归零、`last_seen` 刷新 ⇒ **重新计时** ✓（用户要的"再出现就重置"✓）；
            #   · **只在输出行为把它清掉时例外** ✓（`_reap_ghost_mobs` → `tracker.kill()` ✓
            #     那是"主动清框"✓ 不是防抖该管的 ✓ 用户也点名保留这个例外 ✓）。
            if (t["any_conf"] >= self.debounce_conf
                    and _gap_ms <= self.debounce_ms):
                self.diag["ghost"] += 1
                out.append(self._to_mob(t))
        # ⭐⭐⭐⭐ **结算这一帧的 GMC**（见 `_gmc_estimate` ✓ 放到最后是因为样本由 `_absorb` 收集 ✓
        #   —— 结算出来的这一份**下一帧**才用得上 ✓ 与 SDK 的用法一致 ✓）。
        self.gmc = self._gmc_estimate()
        return out

    # ---------------- 认领 / 确认 / 记账 ----------------

    def _assign(self, tracks, dets, relax=1.0):
        """`tracks` × `dets` 的**全局最优**配对 → `[(轨迹下标, 检测框下标), ...]` ✓。

        代价 = `1 - _score`（分数越高代价越小 ✓）；`_score <= 0` = 配不上 ⇒ 禁止 ✓。
        ⚠ 空表 / 全禁止 ⇒ 返回空（这一拍谁都没配上 ✓ 不许抛 ✗）。
        ⚠⚠ **规模闸**（`ASSIGN_TOP_K` ✓）：每条轨迹只留自己分数最高的 K 个框 ✓
          —— 全矩阵是 O(n²) 的（30 只怪时实测 **7.4 ms** ✗ 那是每帧的预算里不能忍的 ✓）
          ⇒ 留 K 个之后 30×6 是 **0.5 ms** 量级 ✓。这一步在"真配对必在前 K 名"时**不损失最优性** ✓
          （分数是排序用的 ✓ 只满足距离那一档本来就被压在下面 ✓），极端情况才与全矩阵不同 ✓。
        """
        if not tracks or not dets:
            return []
        # 每条轨迹 → [(分数, 检测框下标), ...]（按分数降序 ✓ 只留前 K 个 ✓）
        picks = []
        for t in tracks:
            row = []
            for i, d in enumerate(dets):
                s = self._score(t, d, relax)
                if s > 0.0:
                    row.append((s, i))
            row.sort(key=lambda x: -x[0])
            picks.append(row[:ASSIGN_TOP_K])
        cols = sorted({i for row in picks for _s, i in row})
        if not cols:
            return []
        col_of = {i: k for k, i in enumerate(cols)}
        cost = []
        for row in picks:
            full = [0.0] * len(cols)
            for k in range(len(cols)):
                full[k] = _ASSIGN_BIG                    # 没进前 K 的 = 禁止 ✓
            for s, i in row:
                full[col_of[i]] = max(0.0, 1.0 - s)
            cost.append(full)
        return [(ti, cols[di]) for ti, di in _assign_pairs(cost)]

    def _confirmed(self, t, now):
        """这条轨迹够不够"我看见它了"：**连续 N 拍 且 活满 M 毫秒**（两个都要 ✓）。

        两个数缺一个都能被绕过（见 `CONFIRM_MS` 那段 ✓）；`confirmed` 一旦置上就**不再回退** ✓
        —— 否则一条老轨迹偶尔漏拍会被打回"没确认"，画面又闪起来 ✓。

        ⭐⭐⭐ **2026-10-13 加一条短路**（用户原话："**只要曾经怪物到过该置信度阈值**，那么其框
        消失会被保留「防抖时间」（除非是被输出行为的清框逻辑清掉的），且检出框重新出现时重新
        计时重置"✓）：**历史最高置信度（两个档位都算 ✓ `any_conf`）≥「防抖置信度」⇒ 直接算数** ✓。

        ⚠⚠ **为什么必须加**（真机 `behavior.log` 2026-10-13 01:39 一大片 ✓，共 429 条
          `mob_flick`，全是 `age=1/2`、`conf`/`peak` 落在 **0.30~0.46** ✓）：
          那些"置信度刚好在检出阈值附近晃"的怪**永远过不了 2 拍 + 60ms** ✗ ⇒ 被当成"闪一下"
          删掉 ✗ ⇒ **它的框一闪就没** ✓，而且**连"丢框保留防抖时间"那一步都走不到** ✗
          （输出那支先看 `confirmed` ✗ 见 `update` ✓）⇒ 用户看到的"**防抖不生效 + 框一闪一闪**"
          正是这一条 ✓✓。
        ⚠ **低于**这个阈值的（真·噪声 ✓）**照旧走确认闸**（2 拍 + 60ms ✓）⇒ "单帧假框一个都不出"
          那条没丢 ✓ —— 于是两个旋钮的语义终于一致了：
          「防抖置信度」= **这个档位以上我就认它**（一帧也算 ✓）；
          更低的那批 = 要**连续两拍且活满 60ms** 才认 ✓。
        ⚠ 配 **0 / 负** = **这条短路不启用** ✓（照旧走确认闸 ✓ —— `MobTracker()` 的默认就是 0.0 ✓，
          用例/离线工具都靠它保持老行为 ✓；真实项目里配的是 0.3~0.6 ✓ 短路生效 ✓）。
        ⚠ 现在得是**实例方法**（要读 `self.debounce_conf` ✓）—— 调用点本来就是 `self._confirmed(...)`
          ✓，没有别的静态调用 ✗。
        """
        _dc = float(self.debounce_conf or 0.0)
        if _dc > 0.0 and float(t.get("any_conf") or 0.0) >= _dc:
            return True
        if int(t.get("age", 1) or 1) < CONFIRM_HITS:
            return False
        return (float(now) - float(t.get("born", now) or now)) * 1000.0 >= CONFIRM_MS

    def _note_flick(self, t, now):
        """记一条"闪一下"的现场（**有上限** ✓ 别把内存吃光 ✗）。"""
        try:
            self.flick.append({
                "id": int(t.get("id") or 0),
                "age": int(t.get("age") or 0),
                "conf": round(float(t.get("conf") or 0.0), 3),
                "peak": round(float(t.get("peak_conf") or 0.0), 3),
                "missed": int(t.get("missed") or 0),
                "ms": round((float(now) - float(t.get("born", now) or now)) * 1000.0, 1),
            })
            if len(self.flick) > FLICK_KEEP:
                del self.flick[:-FLICK_KEEP]
        except Exception:                       # noqa: BLE001 —— 记账不许拖垮跟踪 ✗
            pass

    def drain_flick(self):
        """取走累计的"闪一下"现场（并清空 ✓）—— 给 `behavior.log` 用 ✓。"""
        out, self.flick = self.flick, []
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

    def _absorb(self, t, d, now, low=False):
        """把这一帧的检测并进轨迹：平滑中心/尺寸、更新速度。

        用「中心 + 尺寸」而不是四条边分别平滑：这样**位置**和**大小**能各用一个
        系数 —— 移动的目标位置要紧跟（不然框拖在后面），而框大小不该跟着 YOLO
        一帧大一帧小（那正是"框在抖"的来源）。

        `low=True` = 这是**低分档**的框（借 BYTE ✓ 只用来续命 ✓）：位置/速度照常更新 ✓，
        但它**不抬 `peak_conf`** ✗（见下面那段 ✓）。
        """
        x1, y1, x2, y2, cf = d
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # ⭐⭐⭐ **GMC 采样**（见 `GMC_MIN_TRACKS` ✓ 用户 2026-10-13 ✓）：只有"**上一帧也见过
        #   它**"的这一对才有意义 ✓ —— 隔了几帧的差里混着那段走路的累积 ✗（与玩家那条链的
        #   `TELEPORT_MAX_GAP_MS` 同一个道理 ✓）。⚠ 只用**高分档**（`low=False` ✓）：低分档是
        #   "擦着阈值"的框 ✓ 位置噪声更大 ✗。
        _f = int(getattr(self, "_frame_no", 0) or 0)
        _pd = t.get("det")
        if (not low and _pd is not None
                and int(t.get("seen_frame", -1)) == _f - 1):
            self._gmc_samples.append((cx - _pd[0], cy - _pd[1]))
        t["det"] = (cx, cy)
        t["seen_frame"] = _f
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
        # ⭐ 历史最高置信度只涨不落（**界面角标**用：它到底多确信 ✓ 见 `peak_conf`）
        # ⚠⚠ **低分档不参与**（`low=True` ✗ 2026-10-11）：它按定义就是"低于 `conf_mob`"
        #   的框 ✓ ⇒ 让它抬高 `peak_conf` 会把"其实没真正看清过"记成看清过 ✓
        if not low and cf > t["peak_conf"]:
            t["peak_conf"] = cf
        # ⭐⭐⭐ **`any_conf`：两个档位都算** —— **防抖判据读的是它** ✓（用户 2026-10-13 ✓
        #   原话："**只要曾经怪物到过该置信度阈值**，那么其框消失会被保留「防抖时间」
        #   （除非是被输出行为的清框逻辑清掉的），且检出框再次出现时重新计时重置"✓）。
        #   ⚠⚠ **为什么非跟 `peak_conf` 分开不可**（2026-10-13 现场："框一闪一闪"✓）：
        #     「防抖置信度」常被配得**低于**检出阈值（本项目 0.3 < `conf_mob` ✓）——
        #     那种怪**只会走低分档**（`low=True` ✓，用来"续命"✓）⇒ 老判据（只看 `peak_conf`）
        #     **永远不成立** ✗ ⇒ 它一丢框就立刻消失 ⇒ **就是闪** ✗✓（这条是病根 ✓）。
        #   ⚠ `peak_conf` 保持原义（只认高分档 ✓）⇒ 界面角标/曲线观感不变 ✓。
        if cf > t["any_conf"]:
            t["any_conf"] = cf
        t["missed"] = 0
        t["age"] += 1
        # ⭐ **确认闸**（见 `_confirmed`）：攒够就**永久**置上 ✓（只涨不落 ✓）
        #   ⚠ 必须在 `age += 1` **之后**判（这一拍也算"看见了一次" ✓）
        if not t["confirmed"] and self._confirmed(t, now):
            t["confirmed"] = True
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
                  "last_seen", "det", "seen_frame", "_q_n", "_q_box"):
            host[k] = t.get(k, host.get(k))
        host["conf"] = max(host["conf"], t["conf"])
        host["peak_conf"] = max(host["peak_conf"], t["peak_conf"])
        # ⭐ `any_conf`（防抖判据读它 ✓ 见 `_absorb`）合并时也要取最大 ✓，否则
        #   "两条并一条"之后那条的"曾经到过阈值"可能被丢掉 ⇒ 又闪 ✓（2026-10-13 ✓）。
        host["any_conf"] = max(float(host.get("any_conf") or 0.0),
                              float(t.get("any_conf") or 0.0))

    # ---------------- 输出 ----------------

    def _to_mob(self, t):
        return Mob(
            id=t["id"],
            x=(t["x1"] + t["x2"]) / 2.0,
            y=(t["y1"] + t["y2"]) / 2.0,
            w=t["x2"] - t["x1"],
            h=t["y2"] - t["y1"],
            conf=t["conf"],
            # ⭐ 历史最高置信度（2026-10-11 ✓ 界面角标 `#id conf/peak` 用它 ✓
            #   也是"这条轨迹到底有没有被看清过"的读数 ✓）
            peak_conf=t["peak_conf"],
            vx=t["vx"],
            vy=t["vy"],
            age=t["age"],
            missed=t["missed"],
        )

    def kill(self, mob_id):
        """立即消除某个轨迹（防抖幽灵框被攻击时调用，避免持续空放技能）。"""
        self._tracks = [t for t in self._tracks if t["id"] != mob_id]


#: ⭐⭐⭐ 步 2 的四个常量（2026-10-11 ✓ 用户原话："**角色使用传送点后被其他玩家抢检出框**"✓）
#: —— 全借 `visual_tracking_sdk_20260920` 的 `ASSOCIATION_*` 那一组
#: （`liescd/branch_tracker.py:106-116` ✓），逐条对应见 `PlayerTracker.update` 的说明 ✓。
#:  · `COMPETITOR_MARGIN_PX` —— **竞争者余量**（同 SDK 的 `ASSOCIATION_COMPETITOR_DISTANCE_MARGIN_PX=16`
#:    ✓）：要换框时，最优与次优**差不多近** ⇒ 谁都不换 ✓；
#:  · `SWITCH_SUSPECT_RATIO` —— 离旧框超过 `max_jump × 它` 才算"要换人"（下面的判据只对换人生效 ✓
#:    ⇒ 站在别人旁边**不动**照旧 ✓ 那种情况下候选还是"同一个人" ✓）；
#:  · `QUARANTINE_FRAMES` —— **跳变隔离**（同 SDK 的 `ASSOCIATION_QUARANTINE_FRAMES=3` ✓）：
#:    同一个新候选要**连着赢几拍**才认账 ✓（不会因为别人从旁边经过那一下就把锁交出去 ✓）；
#:  · `TELEPORT_MIN_WORLD_PX` —— **传送武装**（SDK 没有这条 ✓ 是我们独有的：
#:    它只有"隔离几拍再看"✓ 我们多一个**独立传感器** —— 小地图黄点 ✓）。
COMPETITOR_MARGIN_PX = 16.0
SWITCH_SUSPECT_RATIO = 0.5
QUARANTINE_FRAMES = 3
#: 黄点**世界坐标**一帧内跳这么多 ⇒ 只能是**传送 / 换图**（不是走路 ✓ 也不是别人走近 ✓）。
#: ⚠ 这个数怎么来的（**别拍脑袋 ✗**）：现场口径是"1 个实时小地图像素 ≈ 8.5 世界像素"✓
#:   ⇒ 400 世界像素 ≈ **47 个面板像素** ✓ —— 而走路一帧（30fps 下 ≈ 4 px ✓ 见 `walk_v_max`
#:   那段）连它的零头都不到 ✓ ⇒ 门槛**很宽松**、宁可不触发也不误触发 ✓
#:   （⚠ 画面 px 与世界 px **不是同一把尺** ✗ ⇒ 这里只敢用"世界 px 的绝对值"这一个量，
#:     不去和 `max_jump`（画面 px ✓）互相换算 ✗ —— 那条换算本仓库没有 ✓ 不新造 ✓）。
TELEPORT_MIN_WORLD_PX = 400.0
#: ⚠⚠ **"大"还不等于"传送"** ✗（2026-10-11 用户现场："**又触发了，隔这么远直接飞过去锁别人**"✗）：
#: 只有当两次世界读数**确实相邻**时，那个差才是"**一帧**的位移" ✓ —— 中间要是隔了好几帧
#: （黄点漏检被"沿用上一帧" ✓ / 框面积闸把这一拍的消费关掉 ✓ / 定位回路给的是旧值 ✓），
#: 差值里就混着**那段走路的累积** ✓ ⇒ 轻轻松松越过 400px ⇒ 被误当传送 ⇒ 预测点被推到
#: 荒谬的位置 ⇒ **谁离那个荒谬点近就锁谁** ✗（"飞过去锁别人"就是这么来的 ✓）。
#: ⚠ 取 150ms ≈ 2.5 帧（capture 15fps ≈ 67ms/帧 ✓）：容得下偶发的一拍抖动 ✓、
#:   拦得住"隔了好几帧"的累积 ✓。**别调大** ✗（越大越容易把走路累积算成传送 ✓）。
TELEPORT_MAX_GAP_MS = 150.0
#: ⭐⭐ **武装那一拍的容差比例**（比普通帧**紧**得多 ✓ 见 `update` 里 `_lim` 那段）：
#: 传送时"旧框 + ΔW"按平移口径**就是**新位置 ✓（只差几个像素 ✓）⇒ 拿普通帧的
#: `max_jump`（150px ✗）去接是**太宽**的 ✓ —— 而宽出来的那一圈，正好装着"别人"✗。
#: 双保险：万一 ΔW 本身是假的（黄点锁错 / 标定漂 ✓）⇒ 附近根本没候选 ⇒ **谁也不认** ✓
#: ⇒ 不会"飞过去锁别人" ✓✓。
TELEPORT_TOL_RATIO = 0.35
#: 武装之后连着这么多拍都"谁也不认" ⇒ 认账失败 ⇒ **解锁重锁** ✓（不许因为一个坏读数永久瞎 ✗）。
TELEPORT_HOLD_MAX = 3
#: ① **黄点权威关**的相对余量（倍数 ✓ **无量纲** ✓ —— 见 `_authority_filter` 那段警告：
#: 画面 px 与世界 px 不是同一把尺 ✗ ⇒ 只用"离黄点比最优那个远出几倍"来判 ✓）。
AUTHORITY_FACTOR = 4.0
#: 现场记录最多留几条（给日志看 ✓ 有上限 ⇒ 不会被刷爆 ✓）。
EVENT_KEEP = 8
#: ⭐⭐ **玩家那条链的确认闸**（2026-10-11 ✓ 同怪那条的口径 ✓ 只是这边只有一条轨迹 ✓）：
#: 原因与怪那边不同 ✗ —— 玩家框**认错人的后果更重** ✓：它会污染**可信相机** ✓
#: ⇒ 所有"画面 → 世界"的换算整体平移 ✗，攻击框还会跳到错的一侧 ✓
#: ⇒ 所以"将信将疑的那一拍"**宁可不出框** ✓（下游全都按 `player_box is None` 处理 ✓
#: 与"这一拍没检测到"完全一样 ✓）。
PLAYER_CONFIRM_HITS = 2
PLAYER_CONFIRM_MS = 60.0
#: ⭐ **玩家低分档的比例**（BYTE ✓ 2026-10-11）：`conf_player × 它` 到 `conf_player` 之间
#: 的框**只允许续命** ✓（见 `PlayerTracker.update` 的"两个前提" ✓）—— 而且玩家这边
#: 比怪**多一条**前提（黄点必须对得上 ✓）：玩家低分框很可能是**别人** ✗。
PLAYER_LOW_CONF_RATIO = 0.5


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

    ⭐ **另一个传感器：小地图上那颗黄点**（用户 2026-09-27/29）—— 黄点的**世界坐标**
    是**权威**的（来自 `perception/minimap.py`，不经过 YOLO ✓）。除了上面那套"挑哪个框是
    我"，它还让本类能做一件事：**把一个"可信的相机"记下来**（`camera()` ✓）。
    为什么需要它（用户 2026-09-29 任务 2："玩家坐标改用 / 用小地图的权威世界坐标核对"）：
    所有"**画面 → 世界**"的换算都要减掉相机，而老口径的相机是**每拍现算**的
    `玩家世界坐标 − 玩家画面框` ⇒ **玩家框抖 Δ，每只怪的世界坐标就同量平移 Δ** ✗
    （`docs/交接.md` §2 那条）。这里改成：**只在"框与黄点对得上"的那一拍**更新相机
    （判据见 `_cam_step`），并把它交给下游用（`world_state.Player.cam_x/cam_y` ✓）。

    cands 形如 [(x1, y1, x2, y2, conf), ...]（画面坐标 xyxy）。
    返回 player_box = (cx, cy, bottom, conf, bw, bh)，和 live_thread 原格式一致；
    无框返回 None。
    """

    def __init__(self, max_jump=150.0, max_missed=10, smooth=0.6,
                 walk_v_max=128.0, walk_accel=1500.0, walk_brake=900.0):
        self.max_jump = float(max_jump)     # 位置突变阈值（像素）：超过就认为不是「我」
        self.max_missed = int(max_missed)   # 连续漏检多少帧放弃锁定
        self.smooth = float(smooth)         # 速度平滑系数（0~1，越大越平滑）
        #: ⭐ **输入感知运动模型**（2026-10-01 ✓，借 `Maple_xfeat` 的
        #:   `InputAwareHorizontalKalman` 那套思路）：方向键不是"观测"，而是
        #:   **已知控制输入** —— 按住就在加速、松手就在地面摩擦下减速。
        #:   ⚠ 单位是**画面像素**（本类所有量都是画面坐标 ✓），不是世界像素；
        #:     默认值照搬那个项目的世界像素常数（冒险岛走路 ≈125 px/s @100% 速度）。
        #:     **画面若做过缩放**（推流分辨率 ≠ 游戏分辨率）就要按倍率改 ✗。
        self.walk_v_max = float(walk_v_max)     # 走路最高速（像素/秒）
        self.walk_accel = float(walk_accel)     # 按住方向键的加速度（像素/秒²）
        self.walk_brake = float(walk_brake)     # 松手后的减速度（像素/秒²）
        #: 当前**输入意图**：-1 左 / 0 没按 / +1 右（`set_input` 喂进来）
        self._in_x = 0
        #: 同上，竖直（爬梯用 ↑/↓）—— 只在真喂了值时才参与 ✓
        self._in_y = 0
        #: 上一次 `update` 的时刻（monotonic）—— 只用来算 `dt`，`None` = 从头开始 ✓
        self._t_prev = None
        #: ⭐ **有没有人喂过输入**（`set_input` 调过一次就为真）。
        #:   **没喂过 ⇒ 一切照旧**（速度保持上一帧那个，一字不变 ✓）—— 这条闸是必须的：
        #:   没有它的话，`direction=0`（没喂 / 没按）会被当成"松手" ⇒ 速度按摩擦衰减到 0
        #:   ⇒ 连**不用这个功能的人**的老行为也被改了 ✗（那就不是"可选增强"了）。
        #:   喂过（哪怕喂的是 0）⇒ 说明上层**知道**玩家此刻按没按 ⇒ 按输入算才对 ✓。
        self._input_aware = False
        self._box = None                    # 当前锁定框 (cx, cy, bottom, conf, bw, bh)
        self._vx = 0.0
        self._vy = 0.0
        self._missed = 0
        #: 上一次**喂进来的黄点世界坐标** `(world_x, world_y)`（None = 那几拍没有）
        self._last_world = None
        #: ⭐ 上一次黄点读数的**时刻**（2026-10-11 ✓）：`_world_step` 的差只有在"两次读数相邻"
        #: 时才是"**一帧**的位移" ✓（隔了几帧的差里混着走路累积 ⇒ 那是假传送 ✓ 见常量 ✓）。
        self._last_world_at = 0.0
        #: 武装状态下"连着几拍谁也不认"（够 `TELEPORT_HOLD_MAX` ⇒ 解锁重锁 ✓ 见 `update` ✓）。
        self._tele_miss = 0
        #: 上一次**拿到的框**（解锁也不清 —— 它只用来算"上一帧的相机" ✓，
        #: 见 `_world_dist`；相机对**所有候选是同一次平移** ⇒ 旧一点也只影响刻度、不影响排序 ✓）
        self._last_box = None
        #: ⭐ **可信相机** `(cam_x, cam_y)`：`世界 = 画面 + 相机`（见 `camera()`）；
        #: `None` = 还没建立（此时下游一律退回老口径"世界坐标 − 画面坐标"现算 ✓）
        self.cam = None
        #: 连续几拍"框与黄点对不上"（对得上就归零）—— 只用来决定"还冻着还是该认账"
        self.cam_missed = 0
        #: 这一拍**是不是**因为"对不上"而没更新相机（打点/界面用 ✓）
        self.cam_rejected = False
        #: ⭐ 这一拍两个传感器差了多少（**世界像素**；判不出来 = `None`）—— 它是个可读的数：
        #: "玩家框那一路到底稳不稳"。"世界坐标整体偏"那类排查要的就是它 ✓
        #: （1 个实时小地图像素 ≈ 8.5 世界像素 ⇒ 差几个像素 = 差零点几个面板格子 ✓）。
        self.cam_resid = None
        #: ⭐⭐ 步 2 的四关给的三样状态（2026-10-11 ✓ 一个事实一个写者：**只有本类写** ✓）：
        #:  · 可疑换人的**隔离计数**（`(框中心, 连续赢了几拍)` ✓ 见 `_quar_ok`）；
        #:  · 这一拍的**账**（认领 / 被权威关剔掉 / 含糊 / 隔离 / 传送各几条 ✓ 给 `behavior.log` ✓）；
        #:  · 值得留痕的**现场**（换锁 / 隔离 / 传送 ✓ 由 `drain_events()` 取走打点 ✓）。
        self._quar_box = None
        self._quar_n = 0
        self.diag = {}
        self.events = []
        #: ⭐ **确认闸**（2026-10-11 ✓ 见 `PLAYER_CONFIRM_HITS` 的说明 ✓）：
        #: `_born`/`_hits` 从"锁定那一刻"重新计 ✓（跟丢重锁要**重新确认** ✓）；
        #: `_confirmed` **只涨不落** ✓（否则偶尔漏一拍就把老框打回"没确认"⇒ 画面又闪 ✓）。
        #: ⚠ 只影响**输出**（`update` 的返回值 ✓）—— 内部该锁还是锁、相机该建还是建 ✓
        #:   （下游对 `player_box is None` 的处理与"这一拍没检测到"完全一样 ✓）。
        self._born = None
        self._hits = 0
        self._confirmed = False
        #: ⭐ 给**界面角标**看的两个只读镜像（2026-10-11 ✓ 见 `last_missed`/`last_low` ✓）：
        #: 这一拍那个框是"真检出 / 外推"、以及"是不是低分档续命来的" ✓。
        #: ⚠ 只由 `update` 写 ✓ 只读用途 ✓（一个事实一个写者 ✓）。
        self.last_missed = 0
        self.last_low = False

    def camera(self):
        """**可信相机** `(cam_x, cam_y)` 或 `None` —— 把画面坐标换算成世界坐标要减掉的那个量。

        口径与 `perception.minimap.screen_to_world` **完全一致**（x 用框中心、y 用**框底** +
        `world_state.FOOT_OFFSET_PX` ✓），下游别自己再减一遍（约定 10：一条口径一处实现 ✓）。
        ⚠ `None` = **还没建立** ⇒ 调用方必须退回老口径（现算 `世界 − 画面` ✗ 不许当成 0 ✗）。
        """
        return self.cam

    @staticmethod
    def _cam_of(cx, bottom, world):
        """(框, 本帧黄点) → 隐含的相机；缺数据 ⇒ `None`（**不猜** ✗）。"""
        if world is None:
            return None
        try:
            from perception import world_state as _ws
            return (float(world[0]) - float(cx),
                    float(world[1]) - (float(bottom) + float(_ws.FOOT_OFFSET_PX)))
        except (TypeError, ValueError):
            return None

    def _cam_step(self, cx, bottom, world):
        """按"**框与黄点对不对得上**"决定要不要更新可信相机 → `True` = 这一拍对不上。

        **为什么用"和上一次采纳的相机差多少"当判据**：两个传感器（YOLO 认画面 / 小地图读
        黄点）各自给"我在哪"，它们**不一致**的唯一表现就是"隐含的相机跳了"——
        阈值直接复用 `max_jump`（本类"位置突变"的容忍量 ✓），**不新造常数** ✓。
        ⚠ 它管的**不是**"框自己抖"：框抖超过 `max_jump` 的那一下**根本走不到这里**
          （`update` 里第一道闸就把它扔了 ✓）⇒ 这里剩下的是"**两个传感器各说各的**"
          （典型：黄点自己被认错、或者框锁到别人身上了）✓ —— 那是纯连续性判据看不见的
          一类错误 ✓（`cam_resid` 就是它的读数 ✓）。

        **对不上时怎么办**：**这一拍不更新相机**（继续用上次那个）—— 宁可旧一点，
        也不要把一个抖出来的 Δ 灌进所有"画面 → 世界"的换算 ✓（怪的世界坐标、
        "脚下属于哪块集合"全都会跟着抖 ✗）。
        ⚠ **不会永远冻住**：连续 `max_missed` 拍都对不上 ⇒ **认账**（真的传送 / 换图 /
        框锁到别人身上去了）⇒ 重新采纳 ✓；另外"跟丢 ⇒ 解锁"那条路也会把相机清掉重来 ✓。
        """
        imp = self._cam_of(cx, bottom, world)
        if imp is None:
            self.cam_resid = None
            return False                    # 这一拍判不出来 ⇒ 不动（也不算"对不上" ✓）
        if self.cam is None:
            self.cam, self.cam_missed, self.cam_resid = imp, 0, 0.0
            self._cam_cand, self._cam_cand_n = None, 0
            return False
        dx = imp[0] - self.cam[0]
        dy = imp[1] - self.cam[1]
        self.cam_resid = (dx * dx + dy * dy) ** 0.5
        if self.cam_resid <= self.max_jump:
            self.cam, self.cam_missed = imp, 0
            self._cam_cand, self._cam_cand_n = None, 0     # 对上了 ⇒ 候选清空 ✓
            return False
        # ⭐⭐⭐ **"认账"必须要求"同一个新位置连续出现"** ✗（用户 2026-10-17 ✓ 现场：
        #   「**为什么每次 L4 爬到一半就停了？？**」✗）。
        # ⛔ 老写法只数**次数**（`cam_missed + 1 >= max_missed` ✓）⇒ 两个候选**交替**出现时
        #   **两个都会被采纳** ✗ ⇒ 相机以 `max_missed` 拍为周期**来回横跳** ✓✓
        #   （现场 `behavior.log` 04:18:55：`cam_x` 在 `242.9 ⇄ −1949.9` 之间跳 ✓、
        #    `pos` 跟着在「轨道 x=1080」⇄「左下鳄鱼 x=−1279」之间跳 ✓ ⇒ 刚吸上 **L4**
        #    （`x=661` ✓）下一拍就被判成"**上了 L18**"（`x=−1587` ✗）⇒
        #    「位置状态非目标」⇒ **任务当场撤掉** ⇒ 爬一半就停 ✓）。
        # ⭐ 新写法：把"正在被拒的那个候选"记下来（`_cam_cand` ✓）——**只有它连续出现**
        #   `max_missed` 拍才认账 ✓ ⇒ 交替时 `n` 永远回到 1 ⇒ **一次都不采纳** ✓✓；
        #   真的传送 / 换图（新位置**稳定**出现 ✓）行为**一字不变** ✓（既有用例
        #   `t_player_tracker_trusted_camera` ③ 钉着"不许永远冻着" ✓）。
        # ⚠ 字段用 `getattr` 兜底 ✓：老对象 / 别处新建的 tracker 也不炸 ✓。
        _cand = getattr(self, "_cam_cand", None)
        if (_cand is not None
                and abs(imp[0] - _cand[0]) <= self.max_jump
                and abs(imp[1] - _cand[1]) <= self.max_jump):
            self._cam_cand_n = int(getattr(self, "_cam_cand_n", 0)) + 1
        else:
            self._cam_cand, self._cam_cand_n = imp, 1
        self.cam_missed += 1
        if self._cam_cand_n >= max(1, self.max_missed):
            self.cam, self.cam_missed = imp, 0             # 认账（新的稳定位置 ✓）
            self._cam_cand, self._cam_cand_n = None, 0
            return False
        return True

    @property
    def locked(self):
        return self._box is not None

    @staticmethod
    def _cand_to_box(c):
        x1, y1, x2, y2, conf = c
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, y2, conf, x2 - x1, y2 - y1)

    def _world_dist(self, cx, bottom, world):
        """候选框 → **世界坐标**、再量到「本帧黄点」的距离；缺数据 ⇒ `None` ✓。

        相机（**上一帧**的）= `上一次的 world − 上一次的框`：
        x 用框中心、y 用**框底**（`_last_box[2]`）—— 和 `perception.minimap.screen_to_world`
        的口径完全一致（那边玩家也是 `x` / `bottom` ✓ 约定 10）。
        ⚠ 拿不到任何一边（没有 world / 还没有框）⇒ `None`（调用方退回像素判据 ✓ 不猜 ✗）。
        """
        if world is None or self._last_world is None or self._last_box is None:
            return None
        try:
            cam_x = float(self._last_world[0]) - float(self._last_box[0])
            # ⭐ 脚底偏移（用户 2026-09-28 ✓）—— 与 `screen_to_world` **同一份口径** ✓
            #   （读 `world_state` 的镜像 ✓ 那边不做反向 import ✓）。
            from perception import world_state as _ws
            cam_y = (float(self._last_world[1])
                     - (float(self._last_box[2]) + float(_ws.FOOT_OFFSET_PX)))
            dx = (float(cx) + cam_x) - float(world[0])
            dy = (float(bottom) + cam_y) - float(world[1])
        except (TypeError, ValueError):
            return None
        return (dx * dx + dy * dy) ** 0.5

    # ---------------- 输入感知的运动预测（2026-10-01 ✓） ----------------

    def set_input(self, direction=0, vertical=0):
        """喂「**此刻按着哪个方向**」（-1 / 0 / +1）⇒ 预测位置照它算 ✓。

        **为什么这比"拿上一帧速度线性外推"准**：角色的移动是**被我们的按键驱动的**
        —— 这是个**已知控制输入**。漏检那几帧里，我们**明知**自己按着右键 ⇒ 它该往右
        加速；松了手 ⇒ 它在摩擦下滑行减速。而"上一帧位移"外推完全不知道这件事：玩家
        中途变向 / 停住 / 起跳时，它照旧朝原方向跑 ⇒ 预测越漂越远 ⇒ ① 挑"哪个框是我"
        挑错、② 漏检补出来的框也是错的 ✗（这两个坑共用同一个预测位置，见 `update`）。
        这正是 `Maple_xfeat` 那个 `InputAwareHorizontalKalman` 的做法（方向键当控制
        输入，按 push/drag 加速度推演 ✓）。

        ⚠ **这里是"意图"，不是"事实"**：键发出去了角色可能还没动（卡墙、端到端延迟）。
        拿它**预测**是对的（本来就该超前一点 ✓）；但**别拿它当"现在在干嘛"的判据** ——
          那正是 2026-09-28 爬绳那件事踩过的坑（"本机按键状态会跟执行器意图飘开" ✗，
          所以 `ladder_id` 改吃"执行器通知"了）。用途不同，别混 ✗。
          兜底还在：`max_missed` 那道闸该解锁还得解锁 ⇒ 意图骗人时不会永远漂下去 ✓。

        ⚠ **不喂**（默认 `0`）⇒ 完全退回老口径（上一帧速度外推）⇒ 现有行为一字不变 ✓。
        """
        try:
            self._in_x = 1 if float(direction) > 0 else (-1 if float(direction) < 0 else 0)
        except (TypeError, ValueError):
            self._in_x = 0
        try:
            self._in_y = 1 if float(vertical) > 0 else (-1 if float(vertical) < 0 else 0)
        except (TypeError, ValueError):
            self._in_y = 0
        self._input_aware = True    # 喂过就按输入算（见 `_input_aware` 那段说明 ✓）

    def _advance_input(self, now):
        """按"已知控制输入"把 `_vx` / `_vy` 推进一步（单位：**像素/帧**，与老口径一致 ✓）。

        ⚠ 三件事必须说清，不然这里最容易改错：
          · **`_vx` 的语义是"每帧位移"**（老代码是 `px = box[0] + _vx` 这么用的）⇒
            速度上限要换算成"这一帧该走多少"：`v_max × dt` ✓；
          · `walk_accel` / `walk_brake` 是"像素/秒²" ⇒ 每帧的速度增量是
            `a × dt × dt`（先乘 dt 得"像素/秒"，再乘 dt 得"像素/帧" ✓）；
          · **`dt` 必须夹住**：主循环卡一下（推理卡了半秒）时，不夹会一口气把预测推出去
            几百像素 ⇒ 比"干脆不预测"还坏 ✗。夹到 1/4 秒 ✓。
        """
        if not self._input_aware:
            return              # 没人喂输入 ⇒ **一个字都不动**（老口径：速度保持 ✓）
        if self._t_prev is None:
            self._t_prev = now
            return
        dt = float(now) - float(self._t_prev)
        self._t_prev = now
        if dt <= 0.0:
            return
        dt = min(0.25, dt)
        self._vx = self._step_axis(self._vx, self._in_x, dt)
        self._vy = self._step_axis(self._vy, self._in_y, dt)

    def _step_axis(self, v, direction, dt):
        """单个轴推进一步 → 新的"像素/帧"。按着 ⇒ 加速到上限；松手 ⇒ 摩擦减速到 0。"""
        if direction:
            goal = direction * self.walk_v_max * dt
            if (goal - v) * direction > 0:        # 还没到目标速度 ⇒ 加速
                v += direction * self.walk_accel * dt * dt
                if (v - goal) * direction > 0:    # 这一下超了 ⇒ 夹住（别冲过上限 ✗）
                    v = goal
            else:                                 # 已经比目标快（被打飞 / 下坡）⇒ 按摩擦拉回
                v -= direction * self.walk_brake * dt * dt
                if (v - goal) * direction < 0:
                    v = goal
            return v
        dv = self.walk_brake * dt * dt            # 松手：地面摩擦，减到 0 就停在 0 ✓
        if abs(v) <= dv:
            return 0.0
        return v - (1.0 if v > 0 else -1.0) * dv

    def _note(self, what, **fields):
        """留一条现场（给 `behavior.log` ✓ 有上限 ✓ 别把内存吃光 ✗）。"""
        try:
            rec = {"what": str(what)}
            rec.update(fields)
            self.events.append(rec)
            if len(self.events) > EVENT_KEEP:
                del self.events[:-EVENT_KEEP]
        except Exception:                       # noqa: BLE001 —— 记账不许拖垮跟踪 ✗
            pass

    def drain_events(self):
        """取走累计的现场（并清空 ✓）—— 给 `behavior.log` 用 ✓。"""
        out, self.events = self.events, []
        return out

    def _world_step(self, world):
        """**两次黄点读数之差**（世界坐标 ✓）；缺一边 ⇒ `None`（**不猜** ✗）。

        ⭐⭐ 它是第 ④ 关"传送武装"的**唯一出处** ✓（`update` 拿它的模长当"大不大" ✓
        拿它的向量当"我该在画面哪"的预测 ✓）——别再在别处算一遍 ✗（那就是第二份口径 ✓）。
        ⚠⚠ 它只回答"**差多少**" ✗，**不回答"这算不算一帧的位移"** ✗ —— 后者要看**两次读数
          隔了多久** ✓（`_last_world_at` ✓ 判据在 `update` 里 ✓ 见 `TELEPORT_MAX_GAP_MS` ✓）。
          这条分开很要紧：**"大"≠"传送"** ✓（隔几帧的走路累积也能很大 ✓ 而那不该武装 ✗）。
        ⚠ 它**不写** `_last_world`（那是 `update` 统一做的 ✓ 一处写 ✓）。
        """
        if world is None or self._last_world is None:
            return None
        try:
            return (float(world[0]) - float(self._last_world[0]),
                    float(world[1]) - float(self._last_world[1]))
        except (TypeError, ValueError, IndexError):
            return None

    def _authority_filter(self, scored, teleport):
        """① **黄点权威关**：离本帧黄点**远出一个量级**的候选直接出局 ✓。

        为什么这条能治"被别的玩家抢框"（用户 2026-10-11 ✓）：别的玩家**自己也会被 YOLO
        认成玩家框** ✓，但他站的位置**离本帧黄点很远** ✓ —— 这一关直接把他剔掉 ✓。
        老判据只问"**离我的预测位置近不近**"✗ ⇒ 一旦传送走，他恰好就在旧位置附近 ✓ 于是赢 ✓。

        ⚠⚠ **只用相对判据**（"是最优那个的几倍"✓ **无量纲** ✓）**绝不新造"画面 px ↔ 世界 px"
          的换算** ✗：两者不是同一把尺 ✓（推流分辨率可能 ≠ 游戏分辨率 ✓ 见 `set_input` 那段 ✓）
          ⇒ 拿 `max_jump`（画面 px ✓）当世界 px 的阈值是错的 ✗。
        ⚠ 相机没建立 / 这拍没黄点 / 只有一个候选 ⇒ **一个字都不动** ✓（老行为 ✓ 绝不因此丢锁 ✗）；
          而且**永远不许把候选筛空** ✓（筛空就等于"世界数据一坏就丢锁"✗）。
        ⚠ 传送那几拍**跳过这一关** ✗：相机还是旧地点的 ✓ 拿它筛必然把真框筛掉 ✓（见 `update` ✓）。
        """
        if teleport or len(scored) < 2:
            return list(scored)
        _ws = [it[2] for it in scored if it[2] is not None]
        if len(_ws) < 2:
            return list(scored)
        _best = max(1e-6, min(_ws))
        out = []
        for it in scored:
            _wd = it[2]
            if _wd is None or _wd <= _best * AUTHORITY_FACTOR:
                out.append(it)
            else:
                self.diag["reject_authority"] = self.diag.get("reject_authority", 0) + 1
        return out or list(scored)

    def _quar_ok(self, cx, cy):
        """③ **跳变隔离**：同一个新候选要**连着赢** `QUARANTINE_FRAMES` 拍才算数 ✓。

        返回 `True` = **认账**（可以换 ✓）；`False` = 再等等（这一拍继续用预测位置 ✓）。
        ⚠ 判据是"和上一拍那个候选**是不是同一团**"（`max_jump` 之内 ✓ 同一把尺 ✓）
          —— 换了对象 ⇒ 计数重来 ✓（正是 SDK `ASSOCIATION_QUARANTINE_FRAMES` 的口径 ✓）。
        """
        if (self._quar_box is not None
                and ((cx - self._quar_box[0]) ** 2
                     + (cy - self._quar_box[1]) ** 2) ** 0.5 <= self.max_jump):
            self._quar_n += 1
        else:
            self._quar_box, self._quar_n = (cx, cy), 1
        return self._quar_n >= QUARANTINE_FRAMES

    def _seen(self, now):
        """这一拍**确实看见它**了（有框被认领 ✓）：攒确认闸；够了就**一直置着** ✓。

        三个数分开写清（照怪那条 `_confirmed` 的口径 ✓ 见 `PLAYER_CONFIRM_HITS` 的说明 ✓）：
          · `_hits` 数"看见几次" ✓；`_born` 是这一段的起点 ✓（重锁会重置 ✓）；
          · **两个都要满足**（连着看见 N 次 **且** 活满 M 毫秒 ✓）—— 60fps 下两次"看见"
            只隔 16ms，往往是**同源**的同一团假框 ✓ 光数次数拦不住 ✓。
        """
        self._hits += 1
        if not self._confirmed:
            if (self._hits >= PLAYER_CONFIRM_HITS
                    and (float(now) - float(self._born if self._born is not None
                                            else now)) * 1000.0 >= PLAYER_CONFIRM_MS):
                self._confirmed = True
        return self._confirmed

    def _gate_out(self, box, now=None):
        """输出闸：**没确认过 ⇒ 一个字都不输出** ✓（返回 `None` ✓ 同"这一拍没检测到"✓）。

        ⚠⚠ **确认判据每一拍都要复评** ✗（2026-10-11 实测踩到）：只在"看见的那一拍"判的话，
          会出现"**次数够了、但那一下时间还没到**"⇒ 之后再没有任何一拍去复评
          ⇒ `_confirmed` **永远不翻** ✗ ⇒ 玩家框永久消失 ✓（用例当场报 `None` ✓）。
          ⇒ 时间地板与次数两个条件都在这儿复评 ✓（谁哪天先满足都行 ✓）。
        """
        if not self._confirmed:
            if (self._hits >= PLAYER_CONFIRM_HITS
                    and (float(now if now is not None else time.monotonic())
                         - float(self._born if self._born is not None else 0.0))
                    * 1000.0 >= PLAYER_CONFIRM_MS):
                self._confirmed = True
            else:
                self.diag["unconfirmed"] = self.diag.get("unconfirmed", 0) + 1
                return None
        return box

    def update(self, cands, world=None, low=()):
        """喂本帧的玩家候选框，返回跟住的 player_box（或 None）。

        `world`（可选；用户 2026-09-27 提的"用小地图的玩家世界坐标辅助玩家框防抖"）=
        **本帧黄点的世界坐标** `(world_x, world_y)`。黄点是**独立于 YOLO 的第二个传感器**
        （一个认画面、一个读小地图 ✓）⇒ 除了"位置连续性"，还能问一句"**它现在到底在哪**"：
        每个候选按上一帧的相机换算成世界坐标，**离本帧黄点最近的那个才是「我」** ✓
        （别人贴着一起走、或者自己被打飞时，这一把尺比外推的像素位置靠谱 ✓）。

        ⚠ **一定要"本帧黄点 + 上一帧的框"**：同一帧的 world 和 box 相减会退化成
          "离上一帧框多远"（就是老判据），等于没加 ✗ —— 所以调用方得**先算黄点定位**再挑框
          （见 `gui/live_thread.py` 挑框那一段 ✓）。
        ⚠ `world` 没给 / 上一次也没给 / 两个候选不等价 ⇒ **退回老判据**（像素连续性 ✓），
          绝不因为"世界数据缺"就把锁丢了 ✗。

        ⭐ 顺带维护**可信相机**（`camera()` ✓）：挑中的那一只按 `_cam_step` 与黄点对一次
        账 —— 对得上才更新相机；对不上就**冻着用旧的**（`cam_rejected=True`，
        打点 `cam_reject` ✓，理由见 `_cam_step`）。

        ⭐⭐ 2026-10-11 起挑框多了**四关**（用户原话："**角色使用传送点后被其他玩家抢检出框**"✓
        ⇒ 全借 `visual_tracking_sdk_20260920` 的 `ASSOCIATION_*` 那一组思路 ✓ 常量在类上面 ✓）：
          ① **黄点权威关**（`_authority_filter`）：离本帧黄点远一个量级的候选直接出局 ✓
             —— 旁边那个玩家的框就死在这一关 ✓；
          ② **竞争者余量**（`COMPETITOR_MARGIN_PX=16` ✓）：最优与次优差不多近 ⇒ 谁都不换 ✓；
          ③ **跳变隔离**（`QUARANTINE_FRAMES=3` ✓）：同一个新候选要**连着赢**才认账 ✓；
          ④ **传送武装**（`_teleport_jumped` ✓）：黄点世界坐标一次大跳 = 传送 ⇒ **立刻认账**
             并重建相机 ✓（SDK 没有这条 ✓ 我们有独立传感器 ⇒ 能把"真传送"与"别人走近"分开 ✓）。
        ⚠⚠ ②③ **只对"要换框"生效** ✗（`SWITCH_SUSPECT_RATIO` ✓）：同一个人接着走、或者站在
          别人旁边**不动**（候选还是同一个人 ✓）都不受影响 ✓ —— 这条保证了改动**不会**把
          "跟前一个人"这件事弄钝 ✓。
        """
        self.cam_rejected = False
        now = time.monotonic()
        low = list(low or ())           # ⭐ 低分档（只续命 ✓ 见类上面 `PLAYER_LOW_CONF_RATIO` ✓）
        #: 这一拍的账（四关各自拦了几次 ✓ 见 `drain_events` / live_thread 打点 ✓）
        self.diag = {"picked": 0, "reject_authority": 0, "ambiguous": 0,
                     "quarantine": 0, "teleport": 0, "relock": 0, "missed": 0,
                     "low": 0, "unconfirmed": 0}
        if self._box is None:
            # 未锁定（开局 / 跟丢后）—— ⭐⭐ 2026-10-11 起**也过黄点关**（① ✓）：
            #   原来只有"谁最近 / 谁置信度最高"✗ ⇒ **开局那一下认错人没人拦** ✗
            #   （用户报的"传送后被别的玩家抢框"里就有一半是这种 ✓）。
            #   判据与 `_authority_filter` **同一份**（**相对倍数** ✓ 无量纲 ✓）：
            #     离黄点**远一个量级**的候选直接出局 ✓（旁边站着的别人先出局 ✓）。
            #   ⚠ 没有黄点读数（`world` 没给 / 上一次没框 ⇒ 量不出世界距离）⇒
            #     **一个字都不动** ✓（退回老口径：取置信度最高 ✓ 绝不因此不锁 ✗）。
            #   ⛔ **低分档不许参与锁定** ✗（`low` 在这里**故意不看** ✓
            #     —— "低分只续命"那条规矩的第一步 ✓）。
            if not cands:
                return None
            _scored = []
            for c in cands:
                _cx, _cy = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
                _scored.append((c, 0.0, self._world_dist(_cx, c[3], world), _cx, _cy))
            _scored = self._authority_filter(_scored, teleport=False)
            _wd_ok = [(it[2], it[0]) for it in _scored if it[2] is not None]
            if _wd_ok:
                pick = min(_wd_ok, key=lambda t: t[0])[1]
            else:
                pick = max([it[0] for it in _scored], key=lambda c: c[4])
            self._box = self._cand_to_box(pick)
            self._vx = self._vy = 0.0
            self._t_prev = None         # 重新锁定 ⇒ 计时重来（别拿断开那段时间算 dt ✗）
            self._missed = 0
            self._last_box = self._box
            if world is not None:
                self._last_world = (float(world[0]), float(world[1]))
                self._last_world_at = now       # ⭐ 时刻一起记（"相邻两帧"判据要用 ✓ 一处写 ✓）
            self._tele_miss = 0                 # 刚锁定 ⇒ 不在"武装认账失败"状态 ✓
            # ⭐ **重新锁定 = 重新建立相机**（`cam=None` ⇒ `_cam_step` 直接采纳这一拍 ✓）：
            #   跟丢/传送/换图之后相机本来就该重来（旧那个是上一段的，沿用只会把它带偏 ✗）。
            self.cam = None
            self.cam_missed = 0
            self._cam_step(self._box[0], self._box[2], world)
            # ⭐⭐ **确认闸从"锁定那一刻"重新开始**（② ✓ 锁定 ≠ 敢用 ✓ 见 `_seen`）：
            #   这一段里 `_hits` 数"又看见了几次" ✓、`_born` 是这一段起点 ✓。
            self._born, self._hits, self._confirmed = now, 0, False
            self._seen(now)
            return self._gate_out(self._box, now)

        # ⭐ **先按"此刻按着什么"把速度推进一拍，再算预测位置**（2026-10-01 ✓）：
        #   这个预测位置是**两件事共用**的 —— ① 下面挑"哪个框是我"；② 没匹配到时外推
        #   那个框（见函数末尾）。放在这里，两处一起受益 ✓（模型见 `set_input`）。
        self._advance_input(now)
        #: ① 按键/速度外推那条预测（**老口径** ✓ 步子永远只有"走路那么大" ✓）
        px0 = self._box[0] + self._vx
        py0 = self._box[1] + self._vy
        #: ② 真正拿去**挑候选**的预测位置（大位移时会被换成"旧框 + ΔW" ✓ 见下）
        px, py = px0, py0

        # ⭐⭐ **黄点给的第二件事：玩家"该在画面里往哪走"**（2026-10-11 ✓ 治"传送后被抢框"✓）
        #   位移 `ΔW` = 这一拍黄点世界坐标 − 上一拍黄点世界坐标 ✓。
        #   · 它**小** ⇒ 正常走路 ✓（与"按键/速度外推"那条预测互相印证 ✓）；
        #   · 它**大**（`TELEPORT_MIN_WORLD_PX` ✓）⇒ 预测改成"**旧框 + ΔW**" ✓
        #     —— 这时按键外推**完全失效** ✗（人一瞬间到了地图另一头 ✓）必须换尺子 ✓。
        #   ⚠⚠ **"大"≠"传送"** ✗（2026-10-11 用户现场："**又触发了，隔这么远直接飞过去
        #     锁别人**"✗ 根因就在这条 ✓）：只有**两次读数确实相邻**（`TELEPORT_MAX_GAP_MS` ✓）
        #     的大跳才算"**我看见它跳了**"这条**证据** ✓；隔了好几帧的差值里混着走路累积 ✓
        #     ⇒ 那只是"我已经不在原来的地方了" ✓ ⇒ **预测照改 ✓、但不武装** ✗
        #     （权威关与跳变隔离照旧生效 ✓ —— 那两道关正是防"顺手锁到旁边那个人"的 ✓）。
        #   ⚠ 为什么敢把"世界像素"当"画面像素"用：本仓库**本来就是平移口径** ✓
        #     （`_cam_of`：`cam = world − box` **没有比例项** ✓ ⇒ 两套坐标在用法上 1:1 ✓）。
        dw = self._world_step(world)
        _gap_ms = (now - float(self._last_world_at or 0.0)) * 1000.0
        _adjacent = _gap_ms <= TELEPORT_MAX_GAP_MS
        _big = (dw is not None
                and (dw[0] * dw[0] + dw[1] * dw[1]) ** 0.5 >= TELEPORT_MIN_WORLD_PX)
        teleport = bool(_big and _adjacent)
        if _big and not _adjacent:
            # 看得见（调 `TELEPORT_MAX_GAP_MS` 时先看这一条 ✓）：这段"走路累积"没被当传送 ✓
            self.diag["teleport_gap"] = self.diag.get("teleport_gap", 0) + 1
            self._note("teleport_gap", gap_ms=round(_gap_ms, 1), dw=dw)
        if _big:
            # 位移够大 ⇒ **挑候选**换尺子（不管隔了几帧：黄点说我在哪，我就该在那儿附近 ✓）
            # ⚠⚠ 但**幽灵外推那一步不吃它** ✗（2026-10-11 实测踩到 ✓）：不然一个**没确认的**
            #   大位移会让报出去的框**自己飞过去** ✓ —— 用户原话就是"**隔这么远直接飞过去**"✗。
            #   ⇒ 精确说：`px/py` 只给"**挑候选**"用 ✓；幽灵那份仍用 `px0/py0` ✓（步长有界 ✓），
            #     除非这一拍是**武装过的真传送** ✓（那时预测有证据 ✓ 见函数末尾那段 ✓）。
            px = self._box[0] + dw[0]
            py = self._box[1] + dw[1]
        if teleport:
            # 传送那一拍（预测上面已经换成"旧框 + ΔW" ✓）
            #   —— 这就是"传送武装"：另一名玩家站在**我原来的位置**附近时，
            #   他离"旧框 + ΔW"远得离谱 ✓ 直接出局 ✓（老判据"离旧位置近就赢"必错 ✓）。
            self.cam = None                 # 相机是旧地点的 ⇒ 作废、下一拍重建 ✓（老行为 ✓）
            self.cam_missed = 0
            self._quar_box, self._quar_n = None, 0
            self.diag["teleport"] = self.diag.get("teleport", 0) + 1
            self._note("teleport", world=world, old=self._box, dw=dw)
        else:
            self._tele_miss = 0             # 不在武装状态 ⇒ "认账失败"的计数归零 ✓

        # 候选打分：`d` = 到**预测位置**的距离（大位移/传送时那个预测已经换成"旧框 + ΔW" ✓）
        # ⭐⭐ 武装那一拍用**紧容差**（`TELEPORT_TOL_RATIO` ✓ 见常量说明）：传送时
        #   "旧框 + ΔW"按平移口径**就是**新位置 ✓（只差几个像素 ✓）⇒ 该比普通帧紧得多 ✓。
        #   ⚠ 这一条是"**万一 ΔW 本身是假的**"（黄点锁错 / 标定漂 ✓）的双保险：
        #     假预测点附近**根本没有候选** ⇒ 这一拍谁也不认 ✓ ⇒ 不会"飞过去锁别人" ✓✓
        #     （用户 2026-10-11 报的那一幕 ✓）；而它**不影响真传送** ✓（真点就在预测点上 ✓）。
        _lim = self.max_jump * (TELEPORT_TOL_RATIO if teleport else 1.0)
        scored = []
        for c in cands:
            cx = (c[0] + c[2]) / 2.0
            cy = (c[1] + c[3]) / 2.0
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if d > _lim:
                continue               # 太远：不是「我」（可能是别人经过 / 已传送）
            scored.append((c, d, self._world_dist(cx, c[3], world), cx, cy))
        # ① **黄点权威关**（见 `_authority_filter`）：离黄点远一个量级的直接出局 ✓
        #   ⚠ 传送那几拍**跳过**它 ✗（相机刚作废、`_world_dist` 拿的是旧相机 ⇒ 判不了 ✓）
        scored = self._authority_filter(scored, teleport)

        # 排序：平时 **世界距离优先**（小地图是独立观测 ✓）、缺了退回像素连续性 ✓；
        #   传送那几拍只能用像素（`d` ✓ 而它现在的依据是 ΔW ✓）
        def _rank(it):
            _wd = it[2]
            if teleport or _wd is None:
                return (1, it[1])
            return (0, _wd)
        scored.sort(key=_rank)

        accept = None
        best = scored[0] if scored else None
        second = scored[1] if len(scored) > 1 else None
        if best is not None:
            _c, _d, _wd, _cx, _cy = best
            _old = (self._box[0], self._box[1])
            _move = ((_cx - _old[0]) ** 2 + (_cy - _old[1]) ** 2) ** 0.5
            # ⚠ 下面两道关**只对"要换框"生效**（`SWITCH_SUSPECT_RATIO` 的用处 ✓）：
            #   "同一个人接着往下走"（走多远都算 ✓）不该被拦 ✓ —— 而站在别人旁边
            #   **不动**时，最优候选一直是"同一个人" ✓ ⇒ 也不受影响 ✓（这条很要紧 ✓）。
            if _move <= self.max_jump * SWITCH_SUSPECT_RATIO:
                accept = best
                self._quar_box, self._quar_n = None, 0
            else:
                _amb = (second is not None
                        and abs(second[1] - _d) < COMPETITOR_MARGIN_PX)
                if _amb:
                    # ② **竞争者余量**（借 SDK 的 16px ✓）：最优与次优差不多近 ⇒ 谁都不换 ✓
                    #   ⚠ 传送那几拍**也留着它** ✗：新地点要是也站着别人，同样该"先别换"✓
                    self.diag["ambiguous"] = self.diag.get("ambiguous", 0) + 1
                    self._note("ambiguous", old=_old, new=(_cx, _cy),
                               best_d=round(_d, 1), second_d=round(second[1], 1))
                elif (not teleport) and not self._quar_ok(_cx, _cy):
                    # ③ **跳变隔离**（借 SDK 的 3 拍 ✓）：同一个新候选**连着赢**才认账 ✓
                    #   ⚠ 传送那几拍**不隔离** ✗ —— ΔW 本身就是"我看到它跳了"的**证据** ✓
                    #     （SDK 缺的正是这个证据 ✓ 所以它只能一律隔离 ✓ 我们能分辨 ✓）。
                    self.diag["quarantine"] = self.diag.get("quarantine", 0) + 1
                    self._note("quarantine", old=_old, new=(_cx, _cy),
                               n=self._quar_n, move=round(_move, 1))
                else:
                    accept = best
                    self.diag["relock"] = self.diag.get("relock", 0) + 1
                    self._note("relock", old=_old, new=(_cx, _cy),
                               move=round(_move, 1), cam_resid=self.cam_resid)
        # ⭐⭐ ③ **低分档只续命**（BYTE ✓ 2026-10-11 ✓ **两个前提缺一不可** ✗）：
        #   前提 ①：**已锁定**（走到这儿必然满足 ✓ 这一段在 `if self._box is None` 之后 ✓）；
        #   前提 ②：**黄点对得上** —— 这一拍有黄点 ✓、这个低分框量出来的世界距离
        #           跟普通档最优**一个量级**（`AUTHORITY_FACTOR` ✓ 同一把**相对**尺 ✓）。
        #   ⚠⚠ 为什么玩家比怪**多一条**前提：玩家的低分框很可能是**别人** ✗（怪的类别只有怪 ✓）；
        #     而认错人的代价是**污染可信相机** ✓ ⇒ 宁可继续外推也不认 ✓。
        #   ⚠ 它只**兜底**：普通档已有可用候选（`accept` 非空）⇒ 低分档**不参与** ✗（= BYTE ✓）。
        _low_accept = None
        if accept is None and low:
            _best_wd = None
            for _it in scored:
                if _it[2] is not None and (_best_wd is None or _it[2] < _best_wd):
                    _best_wd = _it[2]
            _low_ok = []
            for _c in low:
                _lx, _ly = (_c[0] + _c[2]) / 2.0, (_c[1] + _c[3]) / 2.0
                _ld = ((_lx - px) ** 2 + (_ly - py) ** 2) ** 0.5
                if _ld > _lim:
                    continue                       # 离预测位置太远 ⇒ 不是「我」✓
                    #  ⚠ 同样吃**武装那拍的紧容差** ✓（低分框要是也放宽，等于给"锁别人"留口子 ✗）
                _wd = self._world_dist(_lx, _c[3], world)
                if _wd is None or self.cam is None or world is None:
                    continue                       # 前提②判不出来 ⇒ **别用** ✗（不猜 ✓）
                if _best_wd is not None and _wd > _best_wd * AUTHORITY_FACTOR:
                    continue                       # 离黄点太远 ⇒ 多半是别人 ✗
                _low_ok.append((_c, _ld, _wd, _lx, _ly, True))
            if _low_ok:
                _low_ok.sort(key=lambda it: it[2])
                _low_accept = _low_ok[0]
        # ⭐⭐ **「低分维护」也要有出口**（用户 2026-10-16 ✓ 现场原话："**开启只检测视野框后，
        #   玩家检出框很容易永久定住不动**"✓）。
        #   · `_low_accept`（见上面那段）= 正常检出失败时，从"**靠近预测位置**的低分框"里挑一个
        #     **接受** ✓ —— 它的前提是"预测位置还靠谱"✓；而那些低分框要是**静止的背景片** ✓，
        #     预测就不动 ⇒ 框**定住** ✓；
        #   · ⚠⚠ 而这条路上**原本一个出口都没有** ✗（只有"大位移 / 传送"那条有 ✓ 见下面
        #     `teleport_giveup` ✓ 它的原话是"**宁可慢几拍，也不许因为一个坏读数永久瞎**"✓）
        #     ⇒ 定住就是**永久** ✓ = 用户报的现象 ✓；
        #   · ⇒ 连着 `TELEPORT_HOLD_MAX` 拍**都只能靠低分维护** ⇒ 认账失败 ⇒ **解锁重锁** ✓
        #     （下一拍走正常锁定路"在'离黄点最近'里挑"✓）—— ⚠ **复用现成的那个拍数** ✓
        #     与上面那条出口**同一个口径** ✓ 不新造判据、不新造数 ✓；
        #   · ⚠ 计数**一旦有正常接受就清** ✓（`accept` 不是低分那份 ⇒ 跟踪是好的 ⇒ 别乱解锁 ✗）。
        _only_low = bool(_low_accept is not None and accept is None)   # 这一拍**只**能靠低分维护
        if _only_low:
            self._low_hold = getattr(self, "_low_hold", 0) + 1
            if self._low_hold >= TELEPORT_HOLD_MAX:
                self._box = None
                self._vx = self._vy = 0.0
                self._t_prev = None
                self._born, self._hits, self._confirmed = None, 0, False
                self._low_hold = 0
                self.diag["low_giveup"] = self.diag.get("low_giveup", 0) + 1
                self._note("low_giveup", n=TELEPORT_HOLD_MAX)
                return None
        else:
            self._low_hold = 0                    # 有正常接受 ⇒ 跟踪是好的 ⇒ 别乱解锁 ✓
        if _low_accept is not None:
            accept = _low_accept
            self.diag["low"] = self.diag.get("low", 0) + 1
            self._note("low_maintain", new=(round(accept[3], 1), round(accept[4], 1)),
                       wd=round(float(accept[2]), 1))

        if world is not None:
            self._last_world = (float(world[0]), float(world[1]))
            self._last_world_at = now           # ⭐ 时刻一起记（"相邻两帧"判据要用 ✓ 一处写 ✓）

        if teleport or _big:
            # ⚠⚠ **大位移要有出口**（2026-10-11 ✓ 用户现场："又触发了，隔这么远直接飞过去锁别人"✗）：
            #   连着 `TELEPORT_HOLD_MAX` 拍"**谁也不认**" ⇒ 这次大位移多半是**读数错了**
            #   （黄点锁错 / 标定漂 / 换图 ✓）⇒ 认账失败 ⇒ **解锁重锁** ✓ ——
            #   下一拍走正常锁定路（"离黄点最近"里挑 ✓ 那才是"我"✓）。
            #   ⚠ 宁可慢几拍，也**不许因为一个坏读数永久瞎** ✗（不出框 = 决策层没有位置 ✓）。
            #   ⚠ 条件里带上 `_big` ✗：**没武装的大位移**同样会让人盯着一个够不着的预测点
            #     干等 ✓（`max_missed` 是 10 拍 ⇒ 一寸一寸挪回来太慢 ✓）。
            if accept is None:
                self._tele_miss += 1
                if self._tele_miss >= TELEPORT_HOLD_MAX:
                    self._box = None
                    self._vx = self._vy = 0.0
                    self._t_prev = None
                    self._born, self._hits, self._confirmed = None, 0, False
                    self._tele_miss = 0
                    self.diag["teleport_giveup"] = (
                        self.diag.get("teleport_giveup", 0) + 1)
                    self._note("teleport_giveup", n=TELEPORT_HOLD_MAX)
                    return None
            else:
                self._tele_miss = 0

        if accept is not None:
            nb = self._cand_to_box(accept[0])
            # ⚠ 这一拍是**低分档续命**来的吗？（`scored` 的元组是 5 项 ✓ 低分档是 6 项 ✓）
            _is_low = len(accept) > 5 and bool(accept[5])
            a = self.smooth
            self._vx = a * self._vx + (1 - a) * (nb[0] - self._box[0])
            self._vy = a * self._vy + (1 - a) * (nb[1] - self._box[1])
            self._box = nb
            self._last_box = nb         # 相机那一份也更新（下一帧要用它 ✓）
            self._missed = 0
            self.last_missed, self.last_low = 0, _is_low   # 角标用（这拍是真检出 ✓）
            self.diag["picked"] = self.diag.get("picked", 0) + 1
            # ⭐ **确认闸记账**（② ✓ 见 `_seen`）：看见就记 ✓ 没攒够 ⇒ 这一拍**不出框** ✓
            #   （内部该锁还是锁 ✓ 相机也照旧维护 ✓ —— 只拦"给外面用"那一下 ✓）
            self._seen(now)
            # ⭐ 可信相机：**这一拍跟黄点对得上才更新**（对不上就冻着用旧的 ✓ —— 理由、
            #   以及"不会永远冻着"那两条出口都在 `_cam_step` 里 ✓）。
            #   ⚠⚠ **低分档续命的那几拍不校相机** ✗：它本来就不够格校准 ✓
            #     既不该"采纳"、也不该"报对不上" ⇒ **保持相机原样** ✓
            #     （否则一条低分框就能把相机带偏 ✓ 而相机是所有"画面→世界"的尺子 ✗）。
            if not _is_low:
                self.cam_rejected = self._cam_step(nb[0], nb[2], world)
            return self._gate_out(nb, now)

        # 没匹配到 / 本帧没候选 / 被②③拦下：外推一个框，累计漏检
        self.diag["missed"] = self.diag.get("missed", 0) + 1
        self._missed += 1
        if self._missed > self.max_missed:
            self._box = None           # 跟丢：解锁，下一帧重新按置信度锁
            self._vx = self._vy = 0.0
            self._t_prev = None         # 计时也重来（同上面那条：断开的时长不该算 dt ✗）
            # ⭐ 解锁 ⇒ **确认也重来**（重锁之后要**重新确认** ✓ 见 `_seen`）
            self._born, self._hits, self._confirmed = None, 0, False
            return None
        cx, cy, bottom, conf, bw, bh = self._box
        # ⚠⚠ **幽灵外推只吃"有界的"那条预测**（`px0/py0` ✓），**除非**这一拍是**武装过的
        #   真传送**（`teleport` ✓ 那是有证据的大位移 ✓）：
        #   2026-10-11 实测踩到 —— 大位移一改预测，幽灵也跟着跳 ⇒ **报出去的框自己飞过去** ✓
        #   （用户原话："**隔这么远直接飞过去锁别人**"✗）。⇒ 挑候选用 `px/py` ✓、
        #   外推用 `px0/py0` ✓（步长永远只有"走路那么大"✓ 绝不飞 ✓）。
        _gx, _gy = (px, py) if teleport else (px0, py0)
        self._box = (_gx, _gy, bottom + (_gy - cy), conf, bw, bh)
        self._last_box = self._box    # 外推的这一份也是"我最好的估计" ⇒ 相机照它算 ✓
        # ⭐ 角标用：这一拍是**外推**来的（位置不是实测 ✓ 见 live_thread 那个 `*` ✓）
        self.last_missed, self.last_low = self._missed, False
        # ⭐ 外推出来的那一拍同样要过**确认闸**（② ✓）：没确认过 ⇒ 不出框 ✓
        return self._gate_out(self._box, now)
