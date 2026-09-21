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
    grounded: bool = False
    jumping: bool = False
    falling: bool = False
    current_platform_id: int | None = None


@dataclass
class Platform:
    """可站立平台的碰撞顶边，而不是平台整块的视觉矩形。"""
    id: int
    x1: float
    x2: float
    y: float
    conf: float = 0.0
    age: int = 0


@dataclass
class JumpPrediction:
    """玩家处于下落阶段时，对本次落点的纯视觉预测。"""
    target_platform_id: int
    landing_x: float
    landing_y: float
    landing_time: float
    reachable: bool


@dataclass
class WorldState:
    frame_id: int = 0
    ts: float = 0.0
    width: int = 0         # 画面宽（「基于画面中心」的视野计算用）
    height: int = 0        # 画面高
    mobs: list = field(default_factory=list)
    player: Player = field(default_factory=Player)
    platforms: list = field(default_factory=list)
    jump_prediction: JumpPrediction | None = None
