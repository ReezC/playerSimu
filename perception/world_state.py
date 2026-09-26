"""WorldState —— 决策层的唯一输入。

汇总「感知层」的全部输出：怪物追踪结果 + 玩家状态。
决策模块只读 WorldState，不直接碰检测框、原始图像或追踪器内部状态。

数据流（README 已定骨架）：
    画面 → YOLO 检测 → Tracker(跨帧 id) → Mob
          → 模板匹配 + UI 条读取 → Player
                              ↓
                        WorldState → decision
"""
from dataclasses import dataclass, field


@dataclass
class Mob:
    id: int            # 追踪 ID（跨帧稳定，决策用它「盯住同一只怪」）
    x: float           # 框中心 x（画面坐标）
    y: float           # 框中心 y
    w: float           # 框宽
    h: float           # 框高
    conf: float        # 检测置信度
    vx: float = 0.0    # 屏幕速度（px/帧，追踪得到）
    vy: float = 0.0
    age: int = 0       # 连续被追踪到的帧数
    missed: int = 0                  # 连续漏检帧数（>0 = 防抖保留的幽灵框）


@dataclass
class Player:
    x: float = 0.0
    y: float = 0.0
    facing: int = 1         # -1 朝左 / +1 朝右
    hp: float = 1.0         # 0~1，读不到时保持 1.0
    mp: float = 1.0         # 0~1
    state: str = "idle"     # idle / move / attack / hit（先留 idle，后续补）
    found: bool = False     # 本帧是否成功定位到玩家
    bottom: float = 0.0     # 玩家框底部 y（画面坐标），平地巡逻高度过滤用
    w: float = 0.0          # 玩家框宽（画面坐标）
    h: float = 0.0          # 玩家框高
    vx: float = 0.0         # 屏幕速度（px/s，由连续帧估计）
    vy: float = 0.0

    # ---------------- 世界坐标（小地图定位算好后由感知层填）----------------
    #:
    #: 这套是**游戏内部坐标**（WZ 地形里的 foothold / 传送点 / 怪刷新点都用它）。
    #: 由 `perception/minimap.PlayerLocator` 从「小地图面板上的黄点」换算得来。
    #: **算不出来时是 None，不是 0** —— 0 是个合法的世界坐标（地图西北角），
    #: 拿它当"没定位"会让寻路朝地图角落走。
    world_x: float | None = None
    world_y: float | None = None
    #: 站在**地形**的哪条段（`core/mapdata.Terrain.segment_of` 给的 `Segment.index`）；
    #: None = 没落在任何平台上（半空/墙里/没定位）。**寻路要的是这一个。**
    segment_id: int | None = None
    #: True = 这一拍没认出黄点，上面的坐标/段号是**沿用上一帧**的（防抖窗口内，
    #: 口径同 `perception/tracker.py` 的幽灵框）。可以用，但别当新鲜观测。
    world_held: bool = False
    # ---- 寻路执行器要的两样（由实时回路按世界坐标算好，2026-09-26 补）----
    #: 脚下这条 foothold 属于**哪些命名集合**（`core.zones.set_of` 的结果）。
    #: ⚠ 以前**没有任何地方写它** ⇒ `ClimbJob._arrived` 的"脚下已是目标集合"判据
    #: 永不触发，只能靠几何兜底（寻路到了也判不出来）。
    here_sets: list = field(default_factory=list)
    #: 现在**贴在哪根绳上**（`core.zones.ladder_ids` 的那套编号，如 "L1"）；不在绳上 = None。
    #: ⚠ 同样以前没人写 ⇒ "从绳上掉下来"每 2 秒误判一次 ⇒ 任务不停失败重试。
    ladder_id: str | None = None
    #: 没算出来 / 没落平台时的原因（一句话，给界面与日志；正常时是空串）
    world_note: str = ""


@dataclass
class WorldState:
    frame_id: int = 0
    ts: float = 0.0
    width: int = 0         # 画面宽（「基于画面中心」的视野计算用）
    height: int = 0        # 画面高
    mobs: list = field(default_factory=list)
    player: Player = field(default_factory=Player)
    #: 当前**端到端延迟**（毫秒，实时回路填；拿不到时 0）。
    #: 用途：上绳对齐的"保持窗口" = 设置里的保持时间 + 它（见 `agent._climb_tick`）——
    #: 定位读数本来就是"过去某一刻"的位置，延迟越大越不能拿单帧当真。
    e2e_ms: float = 0.0
