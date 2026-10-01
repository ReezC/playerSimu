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
    #: ⭐ **到没到某根绳的上端**（绳号 `"L2"`；`None` = 没到 / 判不出来）—— 上爬的**到达判据** ✓。
    #: 用户 2026-09-27 定的新架构："所有的位置状态更新由**位置状态机**自治"⇒ 这个字段由
    #: `perception/pos_state.py::PositionStateMachine` 算好（判据原文在那里 ✓），
    #: **执行器只读、不许自己比坐标** ✗（`ClimbJob._arrived` 已经改成读它 ✓）。
    at_ladder_top: str | None = None
    #: ⭐ **到没到某根绳的下端**（绳号 `"L2"`；`None` = 判不了）—— 下爬的**到达判据** ✓
    #: （与 `at_ladder_top` 对称：y ≥ 绳下端 − 「坐标对齐误差范围」且 y 在「移动操作尝试
    #: 间隔」内**不再变大** ✓）。判据原文在 `perception/pos_state.py` ✓。
    at_ladder_bottom: str | None = None
    #: ⭐ **只看位置**编出来的绳号（**不管按没按 ↑/↓** ✓；`None` = 位置不在任何绳段里）——
    #: 治的是那条**循环依赖**：`ladder_id` 带按键许可，而**执行器自己会松键** ⇒ 那几十
    #: 毫秒里它变空 ⇒ 执行器"自己把自己判成没上绳" ✗（用户 2026-09-27 报的现象 ✓）。
    #: 用途：`ClimbJob`（"位置还在绳段里"那条记忆 ✓）与 `agent.tick`（攀爬中不打架 ✓）。
    on_rope_pos: str | None = None
    #: ⭐ **脚下那块面的 y**（玩家 x 处的面 y ✓；判不出来 = `None`）—— 下爬"落到目标平台的
    #: 面"的判据 ✓。
    ground_y: float | None = None
    #: ⭐ 脚下这些集合横着占的 **x 范围** `(左, 右)`（非墙 foothold 的并集 ✓；判不出来 = `None`）
    #: —— "我这块平台有多宽" ✓，`agent._reachable_without_path`（"够得着就不用下寻路任务"）
    #: 读它，**不再自己去问 `set_span_of`** ✗。
    here_span: tuple | None = None
    #: ⭐ 攀爬的**「攀爬失败」广播**（2a）：位置贴着绳（`LADDER_DX`=24 那把**宽**尺 ✓）、x 偏出
    #: 「坐标对齐误差范围」，**且僵着不动**（y 在「移动操作尝试间隔」内没变 ✓）⇒ 抓空 / 磨绳 ✓。
    #: ⚠ 「僵着」这一条是**必需**的（2026-09-28 加 ✓）：**斜跳**飞过去时必然穿过"x 差
    #:   10~24px"那一段，而那时人**已经吸上绳、y 正在上升** ⇒ 只看 x 会把每一跳都杀掉 ✗
    #:   （用户报"**斜向跳上绳全部失败**"✗）；原地跳跳之前已对齐到 `|dx| ≤ tol` ⇒ 不中招 ✓。
    #: 执行器只读它判失败、**不再自己比 x** ✗（用户 2026-09-28 核心思路 ✓）。
    climb_failed: bool = False
    #: ⭐ 攀爬的**「补发按住↑」广播**（2c）：y 在「移动操作尝试间隔」内变化没超过容差 ⇒ 卡住 ⇒
    #: 请执行器补发按住 ↑ ✓（执行器**不再自己跟踪 y** ✗）。
    climb_stalled: bool = False
    #: ⭐ 上面这些字段就是**位置状态广播的全部内容**（机器每拍写 ✓）；要"整份快照"用
    #: `perception/pos_state.py::PosSnapshot.of(player)` ✓ —— **拼装只那一处** ✓
    #: （把以前散在 agent / 执行器里的 `getattr` 拆字段收成一处 ✓，用户 2026-09-27 要求
    #: "提高内聚、降低耦合" ✓）。⚠ 不再另存一份整快照（两份存储会不一致 ✗）。
    #: 没算出来 / 没落平台时的原因（一句话，给界面与日志；正常时是空串）
    world_note: str = ""
    #: ⭐⭐ **可信相机** `(cam_x, cam_y)`：把**画面坐标**换算成**世界坐标**要加的那个量
    #: （`世界 = 画面 + 相机` ✓；口径同 `perception.minimap.screen_to_world`：x 用框中心、
    #: y 用**框底** + `FOOT_OFFSET_PX` ✓）。由 `perception/tracker.PlayerTracker` 给出 ——
    #: 它拿「小地图黄点的世界坐标（**权威**）」核对玩家框（用户 2026-09-29 任务 2 ✓）：
    #: **只在对得上的那一拍更新** ⇒ 玩家框抖 Δ 时不会把 Δ 原样灌进所有"画面 → 世界"
    #: 的换算里（怪的世界坐标、脚下的集合都会跟着抖 ✗ 见 `docs/交接.md` §2）。
    #: ⚠ `None` = 还没建立 ⇒ 调用方**必须退回老口径**（现算 `世界 − 画面`），
    #:   **不许当成 0** ✗（0 是个合法的相机值）。
    cam_x: float | None = None
    cam_y: float | None = None


@dataclass
class WorldState:
    frame_id: int = 0
    ts: float = 0.0
    width: int = 0         # 画面宽（「基于画面中心」的视野计算用）
    height: int = 0        # 画面高
    mobs: list = field(default_factory=list)
    player: Player = field(default_factory=Player)
    #: **界面状态**（感知层在原生帧上做弹窗模板匹配的结论 ✓，M1「测谎报警」）：
    #: "combat" / "lie_warn"（测谎预警弹窗）/ "lie_game"（测谎小游戏）/ "lie_success"。
    #: 决策层 `tick` 见它带 `lie` 前缀 ⇒ **停发一切按键**（弹窗期间角色被锁操作 ✓）。
    screen: str = "combat"
    #: 当前**端到端延迟**（毫秒，实时回路填；拿不到时 0）。
    #: 用途：上绳对齐的"保持窗口" = 设置里的保持时间 + 它（见 `agent._climb_tick`）——
    #: 定位读数本来就是"过去某一刻"的位置，延迟越大越不能拿单帧当真。
    e2e_ms: float = 0.0


# ---------------------------------------------------------------- 运行期镜像

#: ⭐ **脚底偏移（像素）的运行期镜像**（用户 2026-09-28 ✓）—— `gui/live_thread` **每帧**把
#:   设置里的 `player_foot_offset_px` 写进来，本模块与 `perception` 里那些"画幅 → 世界"
#:   的换算直接读它 ✓。
#: ⚠ 为什么不各自 `from decision import agent` 现读：`decision/agent.py` **已经 import 了
#:   perception** ⇒ 反过来再 import 会绕成**循环依赖** ✗。所以用这块"谁都能读的小白板" ✓。
FOOT_OFFSET_PX = 0.0


# ---------------------------------------------------------------- 颜色小工具

def norm_hex_color(text, fallback):
    """`#RRGGBB` 归一成 `#rrggbb`（小写 ✓）；写得不对 ⇒ 返回 `fallback`（原值 ✓）。

    为什么放在这儿：用户在「玩家位置」组里**手填**箭头颜色（用户 2026-09-28 ✓）——
    填错了要**保留原值**（悄悄换成默认色反而让人看不出自己填错了 ✗）；只认 6 位十六进制 ✓。
    `gui/player_panel` 与 `gui/live_thread` 两端都 import 它 ⇒ **一套规则** ✓。
    """
    s = str(text or "").strip()
    # ⚠ **7 位（`#RRGGBB`）和 9 位（`#AARRGGBB`，带透明度）都认** ✓ —— 取色弹窗
    #   （`QColorDialog` + `ShowAlphaChannel`）给的就是 9 位 ✓（用户 2026-09-28 起箭头
    #   颜色也走那个弹窗 ✓）；画的时候由 `hex_to_bgr` **丢掉 alpha** ✓（cv2 不用它 ✗）。
    if len(s) in (7, 9) and s.startswith("#"):
        try:
            int(s[1:], 16)
            return "#" + s[1:].lower()
        except ValueError:
            pass
    return str(fallback or "")


def hex_to_bgr(text, fallback=(0, 229, 255)):
    """`#RRGGBB` ⇒ cv2 要的 `(B, G, R)`；认不出来 ⇒ `fallback`（BGR ✓）。"""
    s = norm_hex_color(text, "")
    if not s:
        return tuple(int(v) for v in fallback)
    body = s[1:][-6:]                 # 9 位（#AARRGGBB）⇒ 取后 6 位，**alpha 丢掉** ✓
    return (int(body[4:6], 16), int(body[2:4], 16), int(body[0:2], 16))
