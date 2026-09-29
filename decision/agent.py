"""决策状态机：在「一个平台」上找怪打。

输入 WorldState（玩家位置 + 怪物列表），输出按键动作。

状态（最简 HFSM）：
    idle   —— 没怪 / 没开自动，松开所有键
    chase  —— 有怪但超出攻击距离，朝怪水平移动
    attack —— 怪进入攻击距离，停下攻击

「一个平台」的含义：横版 ARPG 先只做水平方向找怪打，不跨平台跳跃。
垂直（怪在头顶/脚下）留到下一版，先让「找到 → 靠近 → 打」这条闭环跑通。

**线程模型**：agent 跑在实时推理线程里，settings 由 GUI 主线程写。
CPython 下对 float/bool/引用 的简单赋值是原子的（GIL 保证不崩），
读到旧值顶多延迟一帧生效，可接受 —— 不为此上锁。
"""

import random
import threading
import time

from core import behavior, perf
from perception.pos_state import PosSnapshot     # 位置状态广播快照（唯一写者是状态机 ✓）
from decision.input import DEFAULT_KEYMAP, KeyState, key_down, key_up, tap

# 决策参数的落点：**只按项目存** —— projects/<项目>/project.yaml 的 decision 段，
# 由界面打开项目时注册钩子写回（见下面的 set_save_hook）。
#
# config/decision.json 是**旧版**的全局存法，已退役：代码不再读、也不再写它
# （文件留着当档，想搬参数就用参数面板的「加载模板」指到它）。它当年的问题是
# 「谁改参数就被谁覆盖」，于是打开一个还没存过参数的项目会拿到**最后一次改过参数
# 的那个项目**那套 —— 看着就像「一开就是另一个项目的参数」。
# 现在没打开项目时，用的是**最近打开的那个项目**那份（见 gui/project.py 的
# last_opened / gui/main_window.py 的 _bind_decision_params）。

#: 保存钩子：界面打开项目时注册，save() 会顺便把这份参数写回那个项目。
#: **为什么用钩子而不是让 decision/ 认识「项目」**：decision/ 不该依赖 gui/，
#: 而项目对象是界面侧的东西；这里只要能拿到一个 dict 就够了。
_save_hook = None


def set_save_hook(fn):
    """注册/取消「保存时写回项目」的钩子（fn(dict) 或 None）。"""
    global _save_hook
    _save_hook = fn

# 回身输出（反向跳回身）的默认行为序列。
# 每个元素是 dict：
#   {"type": "down",  "key": "..."}  键按下
#   {"type": "up",    "key": "..."}  键松开
#   {"type": "delay", "ms": 200}     额外延迟（毫秒）
# key 可以是键盘映射里的键名（left/right/up/down/attack/jump/...），
# 或特殊键 "back"（反方向）/ "forward"（目标方向），执行时按当前朝向解析。
DEFAULT_BACK_JUMP_SEQ = [
    {"type": "down", "key": "back"},
    {"type": "down", "key": "jump"},
    {"type": "delay", "ms": 50},
    {"type": "up", "key": "jump"},
    {"type": "up", "key": "back"},
    {"type": "down", "key": "forward"},
    {"type": "down", "key": "attack"},
    {"type": "delay", "ms": 30},
    {"type": "up", "key": "attack"},
    {"type": "up", "key": "forward"},
]

# 输出行为默认序列：一个输出键（按下 + 松开）。
DEFAULT_OUTPUT_SEQ = [
    {"type": "down", "key": "attack"},
    {"type": "up", "key": "attack"},
]


# 防掉线行为类型：(id, 显示名)。加新类型时**必须**同时在决策层实现它的流程
# （见 `_run_rest` 与 `docs/开发计划.md` P2），别只加个名字就完事 —— 那样用户选了
# 却什么都不发生 ✗。
ANTI_AFK_TYPES = [("hidden_rest", "隐身休息"),
                  ("spot_rest", "定点休息")]

#: 休息状态机的**唯一契约表**（2026-09-26 收敛，用户批准）——
#: 以前一个状态要在 **7 处**各登记一次（状态清单 / 是否 spot / 界面文字 / 是否有倒计时 /
#: 是否可被打断 / 主 tick 分发 / trace 发布），**漏一处就静默出错** ✗，而且已经实锤两次：
#:   · 漏「界面文字」⇒ 玩家面板写着「未休息」✗（用户当场就问"是没读到时长还是显示错"）；
#:   · 漏 route_panel 那张表 ⇒ 实时页一行都不显示 ✗。
#: 现在只在这里定义，其余全部**派生**（`REST_STATES` / `SPOT_STATES` / `REST_STATES_TIMED` /
#: `REST_STATES_INTERRUPT` / `rest_state_text`）⇒ 加新类型只需改这一处 ✓。
#:
#: 字段含义：
#:   text      —— 给界面的一句话（两个面板共用，别各写一份 ✗）
#:   spot      —— 属于「定点休息」那一型（手动结束 / 被打断的处理和隐身那型不一样）
#:   timed     —— 这个阶段 `rest_until_monotonic` 是有效值（界面能显示剩余）
#:   interrupt —— 这一阶段「补了血」算被打断（勾了「被打断重试」才生效）
REST_STATE_SPEC = {
    "afk_enter":     {"text": "进入隐身…",   "spot": False, "timed": True,
                      "interrupt": True},
    "afk_rest":      {"text": "休息中",      "spot": False, "timed": True,
                      "interrupt": True},
    "afk_exit":      {"text": "退出隐身…",   "spot": False, "timed": False,
                      "interrupt": False},
    "afk_spot_walk": {"text": "前往休息点…", "spot": True,  "timed": False,
                      "interrupt": True},
    "afk_spot_act":  {"text": "到达后行为…", "spot": True,  "timed": False,
                      "interrupt": True},
    "afk_spot_rest": {"text": "定点休息中",  "spot": True,  "timed": True,
                      "interrupt": True},
    "afk_spot_back": {"text": "结束后前往…", "spot": True,  "timed": False,
                      "interrupt": False},
}

#: 属于「休息」的状态（主 tick 靠它决定"这一拍走休息分支"）。
REST_STATES = tuple(REST_STATE_SPEC)
#: 其中属于「定点休息」的那几个（见 `_interrupt_rest` 与主 tick 里 `rest_abort` 那段）。
SPOT_STATES = tuple(s for s, v in REST_STATE_SPEC.items() if v["spot"])
#: 带倒计时的阶段（发布 `rest_until_monotonic` 时按它过滤）。
REST_STATES_TIMED = tuple(s for s, v in REST_STATE_SPEC.items() if v["timed"])
#: 补血算「被打断」的阶段（隐身那型的"走到一半"、定点那型的"正在歇"）。
REST_STATES_INTERRUPT = tuple(s for s, v in REST_STATE_SPEC.items() if v["interrupt"])


def rest_state_text(state):
    """休息阶段 → 短句；空 / 认不出来 ⇒ 空串（**不许猜**成"未休息"那种肯定句 ✗）。"""
    return (REST_STATE_SPEC.get(str(state or "")) or {}).get("text", "")

# 时序拍的最小间隔（秒）：主循环按 `next_deadline()` 精确唤醒，但**至少**隔这么久
# 才醒一次（避免到点时刻已过时忙等）。序列计时走本机绝对时钟，10 毫秒的粒度对
# 输出CD / delay / 跳间隔足够（抓帧节拍：窗口模式默认 15fps ≈ 67ms、推流按流自己的
# fps（实测 ≈30fps ≈ 33ms）—— 两种都比 10ms 大一个数量级 ✓）。
TIMING_TICK = 0.01

#: 「站桩输出」时换向所需的方向键按住时长（秒）。
#:
#: 为什么单列一个数：`min_turn_hold_ms`（换向后方向键至少按住多久）是给**走着的
#: 时候**用的 —— 那时按着方向键本来就在走，多按一会儿没坏处。但**站桩输出**时
#: 它会把"停下来打"重新变成"一边走一边打"：实测（扫平台 + 怪换到另一边）方向键
#: 被按住整整 **1.000 秒**，角色从怪身上走过去，出范围后又回头 → 来回抖。
#: 站桩时按方向键**只为了把角色转过来**（转向后输出还有 `turn_output_delay_ms`
#: 兜着），所以按到"转过来"就够，不该按到"走起来"。
TURN_TAP_S = 0.15

#: 「站桩输出」时，**每几轮攻击**补一个"朝目标"的方向键（用户 2026-09-28 要求 2，原话：
#: "attack 增加一个逻辑：**每 3 次 attack 补一个朝向目标的方向键**，持续『最小切换朝向时间』，
#: 并在这个参数里补上 tips" ✓）。
#: 与 `TURN_TAP_S` 的分工：**不站桩**时换向一律按满「最小切换朝向时间」✓；**站桩时**默认只
#: 点一下（`TURN_TAP_S` ✓ 那条的说明：按满就是从怪身上走过去 ✗），**每 `STATION_TURN_EVERY`
#: 轮**才按满一次 ✓（"每 3 次 attack"就是它 ✓）。
#: ⚠ 站桩期间的轮次计数**退出站桩即清零**（用户 2026-09-28 明确："只要退出站桩计数清零"✓）。
#: ⚠ 「3」是**用户指定的数**（不是拍的 ✓）—— 要改就连这句注释一起改 ✓。
STATION_TURN_EVERY = 3


class DecisionSettings:
    """决策参数（UI 写，决策线程读）。

    **只按项目存**：界面打开项目时注册钩子，save() 就把这份参数整份写进该项目的
    project.yaml（`decision:` 段），换项目就换一套。没打开项目时参数取自
    **最近打开的那个项目**（只读不写）；没有那个项目时用这里的默认值。
    细节见文件顶部那段说明与 `set_save_hook`。
    """

    def __init__(self):
        # 自动打怪开关：**不持久化** —— 每次启动都从「关」开始，
        # 免得「一开机就自己打怪」。由按钮/F11/断线重连流程改。
        self.enabled = False
        # ---- **攻击范围 = 矩形**（用户 2026-09-27 定的）----
        # 以前只看水平（x）；现在由**四个距离**围成的矩形决定（判据见 `_in_box`）：
        #   水平 = 「最小攻击距离」~「最大攻击距离」（前者 >0 时它以内是**盲区**）；
        #   竖直 = 角色中心往上「向上攻击距离」、往下「向下攻击距离」。
        # ⚠ **0 = 该方向不限**（= 以前完全不看 y 的那个行为 ✓）⇒ 没配过的项目行为
        #   一点不变 ✓（老项目文件里也没有这两个键 ⇒ `from_dict` 兜底 0 ✓）。
        self.attack_dist = 80.0     # 最大攻击距离（像素，从角色中心量到怪框最近的边）
        self.min_attack_dist = 0    # 最小攻击距离（同口径）；>0 时它以内 = **盲区**（不算可攻击 + 触发规避）
        # ⚠ **0 就是 0，负数才算"不限"**（用户 2026-09-27 纠正过一次，别又当成"0=无限" ✗）：
        #   · `> 0` ⇒ 那一方向的**具体距离** ✓；
        #   · `0`   ⇒ 那一方向**没有可攻击区域**（按 0 算 —— 上/下配 0 就真的打不到那边 ✓）；
        #   · `< 0` ⇒ 那一方向**不限**（= 老行为"根本不看 y" ✓）⇒ 这也是**默认值**（-1 ✓）
        #     ⇒ 老项目文件里没有这两个键 ⇒ 兜底 -1 ⇒ 行为一点不变 ✓。
        # ⚠ 整个框**面积为 0**（最大攻击距离 = 0、或上下**都**恰好配 0）⇒ 该框**判定直接
        #   跳过**（打不到任何怪）**并且不绘制** ✓（`attack_box_rect` 会返回 None ✓）。
        self.attack_up_dist = -1    # 向上攻击距离（像素；-1/负数 = 不限 = 老行为）
        self.attack_down_dist = -1  # 向下攻击距离（像素；-1/负数 = 不限 = 老行为）
        #: ⚠ 「**跳跃攻击范围**」（用户 2026-09-27 记的一笔，**本次不实现**）：等跳跃物理
        #:   做好后，它表示"怪框落在这个范围内 ⇒ 按跳就能把它带进攻击范围框 ⇒ 触发 attack"。
        #:   设置里已经有它的**颜色 + 开关**（`jump_attack_color` / `jump_attack_on` ✓），
        #:   逻辑等「跳跃标定」（见 `docs/寻路设计.md`）之后再补 ✓。
        self.chase_jump_enabled = False  # 追击起跳开关
        self.chase_jump_min = 0     # 追击起跳区间下限（相对最大攻击距离的偏移，像素）
        self.chase_jump_max = 50    # 追击起跳区间上限（同样是相对最大攻击距离的偏移）
        # **追击起跳需要的冲刺时间（毫秒）**（用户 2026-09-26 要求）：
        # `chase` 状态必须**连续维持**这么久，才有起跳的资格；中途进别的状态就归零。
        # 为什么：刚进追击（或刚打完、刚转身）就跳，常常是"为了跳而跳"—— 人还没冲起来，
        # 跳出去够不着还把节奏打断。0 = 不额外要求（老行为）。
        self.chase_jump_dash_ms = 0
        self.mouse_speed = 1.0      # 触控板灵敏度：本地鼠标位移 → 远程鼠标位移的比例（1.0 = 1:1）
        self.evade_type = "jump"    # 规避类型："jump" 跳 / "back" 后退
        self.jump_interval = 200    # 跳间隔（毫秒）：跳键和输出键之间的间隔
        self.jump_random_prob = 0.1 # 乱跳几率（0~1）：正常攻击时随机跳+输出的概率
        self.back_jump_seq = [dict(e) for e in DEFAULT_BACK_JUMP_SEQ]  # 回身输出行为序列
        self.output_seq = [dict(e) for e in DEFAULT_OUTPUT_SEQ]        # 输出行为序列（attack 状态执行）
        self.keymap = dict(DEFAULT_KEYMAP)
        self.target_cd = [500, 1000]  # 目标切换 CD [min, max]（毫秒）
        self.input_device = "local"   # 输入设备："local" 本机 / "remote" Pro Micro 远程 / "serial" Pro Micro 本地
        # 输出行为 CD（毫秒）：两次输出之间的**最小间隔**，从上次输出时刻起算
        # （实际间隔 = 这个值本身；输出序列自身更长时以序列为准）
        #
        # ⚠ **「随机输入延迟」2026-09-27 按用户要求整个删掉**（原来这里有一对
        #   `input_delay = [70, 130]` 毫秒，另外叠在 输出CD / 序列元素间隔 / 追击跳间隔 /
        #   规避间隔 上做"人工化抖动"）。删它的理由：抖掉的是**可复现性** —— 同一份配置
        #   两次跑出来的节奏不一样，"这一下为什么慢/快"就永远对不上 ✗。
        #   删掉之后那些间隔就是配置里那个数本身；序列元素之间退化成"下一帧"
        #   （见 `_run_seq`）。**这个参数不许回来**（`tools/selftest_decision.py` 反向钉着）。
        self.attack_cd = 0
        self.attack_lock_debounce_ms = 300  # 输出后锁定防抖（毫秒）：这段时间内 attack 目标保持锁定，不切换到攻击范围内其他框
        # 最小切换朝向时间（毫秒）：换向后方向键至少按住这么久。按下去立刻松开的话，
        # 角色的转身动作可能还没做完 —— 这时候输出会打向**错误的方向**（用户实测）。
        # 这段按住时间只能被「又换一次朝向」打断（那是新的一次换向，重新计时）。
        # 0 = 不约束（保持老行为）。
        self.min_turn_hold_ms = 0
        # 转向后输出延迟（毫秒）：换向后推迟这么久才开始输出。和上面那条配合用 ——
        # 前者保证方向键按住够久，后者保证输出等转身做完。0 = 不延迟。
        self.turn_output_delay_ms = 0
        self.player_track_jump = 150        # 玩家追踪阈值（像素）：新玩家框离预测位置超此距离就不认（防跟错）
        # ==================== ⭐⭐ 「玩家位置」参数组（用户 2026-09-28 要求 ✓）====================
        #: ① **脚底偏移（像素）**：算"脚底 / 相机 y"时用 `框底 + 这个值`（正 = 往下挪 ✓）。
        #:   为什么需要：人物框的底边**不一定**正好压在脚底（鞋底阴影 / 披风 / 特效会让框多
        #:   出一截 ✗）⇒ 差多少由这儿补正（在实时预览上对着箭头调 ✓）。
        #:   ⚠ 用它的地方（四处，口径要一致 ✓）：`live_thread._resolve` 的相机 y、
        #:     `perception/minimap.screen_to_world`、`perception/tracker._world_dist`、
        #:     实时预览上那对箭头 ✓。
        self.player_foot_offset_px = 0
        #: ② **框面积最小占比（%）**：框面积 < 画面面积 × 这个比例 ⇒ 这一拍**不算有效检测**
        #:   （人太小 / 框抖掉了 ⇒ 位置不可信 ⇒ 宁可"这一拍不定位"✓）。**0 = 不启用** ✓。
        self.player_box_min_area_pct = 0.0
        #: ③ **面积基线窗口**（拍）：拿最近这么多拍的框面积求**滚动均值**当"正常大小" ✓。
        #: ④ **面积容差（%）**：当前框面积 ≤ 基线 × (1 − 容差%) ⇒ 这一拍**推迟 / 不做位置查询** ✓
        #:   ⚠ 用户明确："**拦在查询之前**，而不是靠放宽挑面"✗ —— 见 `live_thread` 里
        #:     `_locate_mmap` 之前的那道闸 ✓。**容差 0 = 这闸关掉**（老行为 ✓）。
        #: ⭐ **单位是「秒」**（用户 2026-09-28 ✓："『拍』是什么？是多久？需要可量化的
        #:   描述"✓）—— 原来是「30 拍」：拍数**依赖帧率**（30fps 的 30 拍 = 1 秒、
        #:   15fps 就成 2 秒）⇒ 说不清到底多久 ✗。默认 **3 秒** ✓。
        self.player_box_area_base_s = 3.0
        self.player_box_area_tol_pct = 0.0
        #: ⑤ **实时预览上的箭头**（用户 2026-09-28 ✓："以玩家位置为原点画「向前箭头 + 向上
        #:   箭头」，颜色、线段长度可配"）：以**玩家位置**为原点画两根 ——
        #:   水平那根朝**世界 x 正方向**、竖直那根朝**世界 y 正方向**（画面上就是"向上"✓）。
        #:   它只干一件事：**让人肉眼看出"定位 / 脚底偏移到底对不对"** ✓（正好用来调 ① ✓）。
        #:   颜色用 `#RRGGBB`；长度 = 线段的像素长（0 = 不画 ✓）。
        # ⚠ 玩家坐标箭头（颜色 / 粗细 / 长度）**不在这里** —— 它属于"本机外观"，
        #   归 **设置 → 界面 → 辅助线与标记（实时预览）**，存在 `config/ui.yaml` 的 `vis:` 段
        #   （见 `gui/theme.VIS_DEFAULTS` ✓）。⚠ 判据见 `docs/UI规范.md` §4：
        #   **"只是画给人看的"进那一组，"会影响怎么打 / 怎么走"才进这里** ✓。
        #   ⛔ 别再加回这一组（2026-09-28 犯过一次 ✗）。
        self.hp_bar = None          # HP 条区域 (x, y, w, h) 或 None
        self.mp_bar = None          # MP 条区域 (x, y, w, h) 或 None
        # 探针框选标定的结果（人工在实时画面上框出来的）。**按项目存** ——
        # 和 HP/MP 条同一套思路：不同项目可能分辨率/客户端布局不同；没打开项目时
        # 随「最近打开的项目」走（见 main_window._bind_decision_params）。
        # 形状（存**画面比例**，换分辨率不用重标；read_bits 支持浮点 cell）：
        #   {"x_ratio","y_ratio","cell_ratio","gap_ratio","bits","frame","solved_at"}
        # {} = 没标定过 → 回退到 config/probe_calib.json（旧版全局标定）→
        #      再回退到 config/link.yaml 的 probe.x/y/cell/gap 像素值。
        self.probe_calib = {}
        self.hp_color = None        # HP 填充色范围 [[B,G,R],[B,G,R]] 或 None（默认红）
        self.mp_color = None        # MP 填充色范围 [[B,G,R],[B,G,R]] 或 None（默认蓝）
        self.hp_threshold = 30      # HP 阈值（百分比 0~100）
        self.mp_threshold = 20      # MP 阈值（百分比 0~100）
        self.pot_cd = 1000          # 喝药冷却（毫秒），喝完后这段时间不再喝
        #: ⭐ 「**怪物判定缓存时长(s)**」（用户 2026-09-28 要求 ✓ 原话："直接做"）：
        #:   同一只怪 + **局面指纹没变**（怪没动、人没换层 ✓）⇒ 这段时间内用缓存 ✓
        #:   （见 `_locked_mob_info` ✓）；`0` = 每次都查（= 关掉缓存 ✓）。
        #:   ⚠ 原来写死常量 `MOB_SETS_TTL_S = 3.0` ✗；也与「前往重下间隔」**拆开**了
        #:     （后者从此专管「**目标切换 CD**」✓）。
        self.mob_sets_ttl_s = 3.0
        self.auto_hp_pot = False    # 自动补血开关
        self.auto_mp_pot = False    # 自动补蓝开关
        self.strategy = "patrol"    # 战斗策略类型："patrol" 平地巡逻
        # 路线测试：「路线识别」里选的**目标平台**（集合名，空 = 没选）。
        # 放在这里而不是 panel 里，是为了**跟着项目存**（换项目/重开还是这个）。
        # 面板那句「命令前往」按它算一遍路；等 P4 的执行器做出来再由它真去走。
        self.route_goto_set = ""
        self.vision_top = 200       # 向上视野（像素），<0 = 不限制
        self.vision_bottom = 200    # 向下视野
        self.vision_left = 200      # 向左视野
        self.vision_right = 200     # 向右视野
        self.vision_center = False  # 基于画面中心：勾选则以画面屏幕中心为基准，否则以角色为中心
        self.vision_off_x = 0       # 视野框 x 偏移（像素）
        self.vision_off_y = 0       # 视野框 y 偏移（像素）
        self.sweep_turn_cd = 1000   # 扫平台：当前朝向没怪持续此时间（毫秒）才换朝向
        self.back_range = 100       # 扫平台：允许锁定背后多远距离内的怪（像素）

        # 判定参数（设置 → 「判定参数」页签）：**对齐类**功能（寻路走到某个 x、上绳前对齐
        # 绳的 x、判定"在不在绳上"）靠这两个数说话 —— "多近才算对齐"、"对齐要保持多久算成功"。
        # 为什么"保持时间"不能省：小地图定位与按键下发**不是同一时刻**的（有延迟、还会抖），
        # 单帧落进误差范围不代表真的站住了；等它稳定一小会儿再算成功，才不会被抖动来回骗。
        # 实际等待 = 延迟时间 + 这个值（见 align_hold_ms 的注释与设置里的说明）。
        # ⚠ 默认 **10**（2026-09-27 从 6 改，用户要求把"名不副实"的容差一起改掉）：
        #   这张图 **1 个实时小地图像素 ≈ 8.5 世界像素**，而读数抖 ±0.5 面板像素
        #   （±4.3 世界像素）、双点标定残差 ≈6 ⇒ 要"对齐到 6 像素"等于要求亚像素精度 ✗
        #   （现场观感：对齐时来回抖、按很久）。10 ≈ 1.2 面板像素，且**现场已经在用**这个值 ✓。
        #   口径与 `route.ALIGN_TOL_PX` 必须一致（用例 `t_align_tol_matches_route` 钉着 ✓）。
        self.align_tol_px = 10      # 坐标对齐误差范围（世界像素）
        self.align_hold_ms = 250    # 坐标对齐误差时间（毫秒）
        # **寻路超时时间（秒）**（用户 2026-09-26 要求）：一个寻路任务（走 / 爬 / 下跳）
        # 从**下达那一刻**算起持续这么久还没结束 ⇒ 切断（如实说明，不再重试）。
        # 为什么不能只靠任务自己的 `timeout_s`：失败重来会重置任务内部的计时
        #（`ClimbJob.retry` 清 `_t0`、`WalkJob.retry` 同样）⇒ 累计可能远超预期。
        # 0 = 不限时（老行为）。见 `agent._climb_tick` 里那道总闸。
        self.goto_timeout_s = 30
        #: ⭐⭐ **「追怪寻路.duration(s)」**（用户 2026-09-28 要求 ✓ 原话："表示本次任务如果是
        #:   追怪下达的，那么如果任务超过了这个时间就结束任务"）—— 单位**秒** ✓。
        #:   ⚠ **只对"追怪下达的"任务生效**（`_climb_origin` 的 `kind == "chase"` ✓）：
        #:     「命令前往」/「定点休息」/「回战斗区域」等都**不受它管** ✗。
        #:   ⚠ **0（默认）= 不启用** ✓ ⇒ **老项目一字不变** ✓。
        #:   ⚠ 与「寻路超时时间」是**两道独立的闸**（同一把钟 `_climb_started` ✓ 但各判各的 ✓）：
        #:     这一道更"专"（只掐追击 ✓），通常设得**比寻路超时更短** ✓。
        self.chase_goto_max_s = 0.0
        #: ⭐ **「寻路超时后按键」**（用户 2026-09-28 要求）：寻路**超时切断**那一下，额外按一下
        #: 这个键（键名字符串 ✓ —— 与 `keymap` / 自定义按键同一套名义 ✓）；`None` = **不按** ✓
        #: （默认 ✓）。在「设置 → 判定参数」页里选：下拉 = **固定键 + 自定义按键** ✓
        #: （与行为编辑器可选键同一口径，见 `gui/seq_editor.SEQ_KEYS` ✓）。
        #: ⚠ 为什么是"**按一下**"（点按）而不是"按住"：它接在「寻路超时时间」那道闸后面
        #: （见 `_climb_tick` 里的 `goto_timeout` ✓），用途是"卡住时给游戏一个信号"；
        #: 按住会把它变成一个**持续输入**、糊住后面每一拍的判断 ✗ ⇒ 点按最干净 ✓。
        #: ⚠ 默认 `None` ⇒ **老项目行为一点不变** ✓（一个键都不多发 ✓）。
        self.goto_timeout_key = None
        #: ⛔ **禁用杀怪寻路**（用户 2026-09-28 要求 ✓，界面在「设置 → 判定参数」**顶层** ✓）：
        #:   **开启**（`True`）后 ——
        #:     · **不再查询**"怪物框所属的 foothold 集合"（画面上那行 `查#怪号 集合名` 也不再出现 ✓）；
        #:     · **不再因追怪而下达寻路任务**（含"朝方向逐层逼近"那条降级 ✓）；
        #:     · 只是**单纯地朝锁定怪物走** = **无寻路时的老逻辑**（`tick` 里 `_steer` ✓）。
        #:   ⚠ 默认 `False` ⇒ **老项目行为一点不变** ✓（和 `goto_timeout_key` 同一个纪律 ✓）。
        #:   ⚠ **它管的是"追怪寻路"这一条链** ✗；**别的**也在查集合的地方**不受影响**：
        #:     · `_zone_only`（「编辑战斗区域」勾了「禁止战斗」时的**区域筛** ✓）——
        #:       那是"别追出禁战区"，是**另一个功能** ⇒ 停了会让配好的禁战失效 ✗；
        #:     · 追击起跳（由 `chase_jump_enabled` 单独管 ✓）。
        self.disable_chase_pathfinding = False
        # ⚠ 这里原来有一格「**卡住判定时长(s)**」（`stall_s`）—— **2026-09-27 用户要求移除** ✓：
        # 原话："移除卡住判定时长(s)，**统一采用移动操作尝试间隔(ms)**" ✓。
        # 那些"多久没进展就算卡住"的判据（上爬爬不动 / 下爬等落地 / 跳跃没落到）现在**统一**
        # 读 `move_retry_ms`（换算成秒 ✓，见 `start_climb` 里灌 job 的那一段 ✓）；
        # 走（`WalkJob`）照旧用 `route.STALL_S` 常量 + 「寻路超时时间」兜底 ✓（用户早前明确过）
        # —— 它吃的不是这一件参数 ✓。老项目文件里残留的 `stall_s` 键**读进来直接忽略** ✓
        # （见 `from_dict` ✓，不报错、也不再用 ✗）。
        # 「**起跳距离(px)**」（用户 2026-09-27 要求；界面上在「路线识别 → 物理参数」组 ✓）：
        # **两个地方**吃它（2026-09-27 起）：
        #   ① 「跳(jump)」：角色 x 到**最近那条目标 foothold 的 x 范围**（最近端点）的距离
        #      ≤ 它 ⇒ 按跳 ✓；**0 = 只有 x 真进了目标 foothold 范围才跳**（默认 ✓ 不拍脑袋 ✓）；
        #   ② **上绳的「斜跳」**（用户 2026-09-27："利用「起跳距离范围」设计「斜跳上绳」逻辑，
        #      要求使用「开始对齐绳梯x的距离」参数兼容旧的「对齐x原地起跳上绳」逻辑"）：
        #      「开始对齐绳梯x的距离」(`climb_align_near_px`) < 离绳 x 距离 ≤ 它 **且人站在绳
        #      入绳端那块 foothold 上** ⇒ 按住朝绳方向 + 起跳（`ClimbJob.jump_start_px` ✓）；
        #      **0 = 斜跳整个关掉** ⇒ 与老版本完全一致 ✓（兼容底线）。
        # ⚠ 用户原话："先配一个起跳距离测试用，后续物理相关逻辑实现后，移除起跳距离参数，
        #   增加角色移动速度、跳跃力的配置后台换算" ⇒ 它**早晚会被删掉** ✓，
        #   别再往它身上挂**第三件**事 ✗。
        # ⚠ 与「追击起跳」（`chase_jump_*`，战斗中追怪用的那个跳）**不是一回事** ✓。
        self.jump_start_px = 0
        # 「**移动操作尝试间隔(ms)**」（2026-09-27 用户要求：把「爬不动时先补按 ↑ 观察(s)」
        # 改名成它，并从「攀爬参数」子组搬到外面的「寻路配置」组；单位从**秒**改成**毫秒** ✓）。
        # 它现在管**两件**同一语义的事 —— "一段移动操作发出去之后，隔多久没看到预期变化
        # 就补发/重试一遍" ✓：
        #   · **爬绳**（`ClimbJob.reassert_s`，秒）：y 不再变好 ⇒ 先**补按一次 ↑**，
        #     观察这么久还不好才判失败 ✓（用户口径"在绳上、不偏离、↑ 没松就一定能上"
        #     ⇒ "不动"首先是"键没按上" ✗）；
        #   · **下跳**（`DropJob.retry_ms`，毫秒）：按住 ↓ + 点按跳之后这么久 Y 还没动
        #     ⇒ **补发一个"松开 ↓"**，再重新按住 ↓ + 点按跳 ✓。
        # 0 = 不重试（爬绳直接判失败 / 下跳一直按着等）—— **老行为**✓。
        # ⚠ 毫秒一律整数（`docs/UI规范.md` §9；`tools/check_ui.py` 的 `check_ms_is_integer`
        #   会当场报错 ✗）。
        self.move_retry_ms = 3000
        # ⚠ 「**前往重下间隔(s)**」这一格 **2026-09-28 用户要求移除** ✓：
        #   它搬进了**每一个战斗区域项**（`battle_zones[i]["cd_s"]` ✓）、改名
        #   「**区域查询CD(s)**」✓（原话："「前往重下间隔」参数移除，逻辑移动到战斗区域限制
        #   配置的每一项上（每项会配不一样），并改名「区域查询CD(s)」"✓）。
        #   ⇒ 运行期一律走 `agent._zone_cd_s(zone)` / `_zone_cd_of_sets(names)` ✓，
        #     拿不到就兜底模块常量 `ZONE_GOTO_RETRY_S` ✓。**这个字段不要再加回来** ✗。
        # 「**对齐绳梯移动延迟(ms)**」（用户 2026-09-27 要求，「攀爬参数」组里的新参数）：
        # 对齐绳梯 x 时**两次按下方向键之间**至少要隔这么久（`ClimbJob.align_gap_ms`）。
        # 默认 180 **就是** `route.TAP_PERIOD_S`（原来写死的那个点按周期）—— 那边是常量、
        # 这里是它的设置快照，两处必须一致（自检 `t_climb_align_gap` 钉着这条 ✓）。
        # 远距离（>20px）是一口气按住走（只有"一次按下" ⇒ 天然满足 ✓，点按会太慢/超时 ✗）；
        # 真正受它约束的是**快到位时的一下一下点按** ✓。0 = 不限制。
        self.climb_align_gap_ms = 180
        # 「**开始对齐绳梯x的距离(px)**」（用户 2026-09-27 要求，「攀爬参数」组里的新参数）：
        # 上绳对齐时，差的比它远 ⇒ **一口气按住走**；**进到它以内 ⇒ 改成点按**（微调，
        # 受上面那个「移动延迟」约束的就是这一段）。原来是**写死**在 `route.NEAR_PX` 的
        # 20px，用户实测"太近了"（精细那一段来得太晚）⇒ 抽成一个数。
        # 默认 20 = `route.NEAR_PX` —— **老行为一点不变** ✓（自检钉着"两处必须一致"）。
        self.climb_align_near_px = 20
        #: ⭐ **「路线脚本 → 战斗区域」**（2026-09-28 用户重构：原「限制战斗区域」按钮 +
        #: 「前往重下间隔(s)」那个全局参数，**合并成每一项自己的配置** ✓）。
        #: 一项 = **一个 foothold 集合**的完整配置 ✓（list of dict）：
        #:   · `set`          —— foothold 集合名（**唯一** ✓；空 ⇒ 该项无效 ✓）
        #:   · `cd_s`         —— 「**区域查询CD(s)**」（= 旧的「前往重下间隔(s)」搬进每一项 ✓）
        #:                       三处共用：① 不在区域 ⇒ 重下「回区域」的节流（用**目标**那一项 ✓）；
        #:                       ② 追击下「前往」的节流 + ③ 区域筛缓存时效 ⇒ 都用「**怪那块
        #:                       foothold 命中的那个区域项**」✓（归属不到 ⇒ 兜底
        #:                       `ZONE_GOTO_RETRY_S` ✓ 用户 2026-09-28 定的 ✓）。
        #:   · `idle_foothold` —— 「**idle 回归 foothold**」（**foothold id**；空 = 无 ✓）：
        #:                       位于本集合时 idle（没怪）⇒ **水平走**向它的中心、到中心就停 ✓
        #:                       （**不跨层** ✓ 用户 2026-09-28 明确）。
        #:   · `fight_max_s`   —— 「**最大战斗时长(s)**」（0 = 不限 ✓）：在本集合里**有战斗**
        #:                       （attack / chase）连续累计超过它 ⇒ 前往 `fight_dst` ✓
        #:                       （没怪 / 离开 / 换区域 ⇒ **清零** ✓）。
        #:   · `fight_dst`     —— 到点要去的**目标地 foothold 集**（集合名 ✓；`fight_max_s>0` 才有意义 ✓）
        #:   · ⭐ `can_fight`  —— 「**可以战斗**」（用户 2026-09-28 要求 ✓，勾选 ✓）：
        #:                       **勾上 = 这块平台允许打架** ✓；人在**别的**平台上时这一拍
        #:                       不打架、先下「前往」去一块**能打**的地方 ✓；而且只打**能打区里**的怪 ✓。
        #:                       ⚠⚠ **一个都没勾 = 哪都能打**（不限制 ✓ 用户 2026-09-28 明确 ✓
        #:                       "都没勾 = 哪都能打（= 现在的「不限制」老行为）"）。
        #:                       ⚠⚠ **改名不改值**（2026-09-28 改 ✗）：这个键原来叫 `no_fight`
        #:                       （界面写着「禁止战斗」✗）—— 但**代码一直把它当白名单消费**
        #:                       （`_in_battle_zone` = `here & zones` ✓、`_zone_only` 只保留名单里的 ✓）
        #:                       ⇒ 它**本来就是**"这块能打"的意思 ✓ ⇒ 若按字面取反，用户**已配好的
        #:                       那块能打区会全部变成不能打** ✗✗（角色会一直想逃出去 ✓）。
        #:                       ⇒ 所以只**改名 + 改文案**、**值原样保留** ✓（老键 `no_fight`
        #:                       读进来也**照抄原值** ✓ 见 `_load_battle_zones` ✓）。
        #:                       实现：派生副本 `battle_zone_sets` **只收 `can_fight=True` 的项** ✓
        #:                       ⇒ `_in_battle_zone` / `_zone_only` / `_leave_battle_zone_tick`
        #:                       这些老消费方**一个字都不用改** ✓（见 `sync_battle_zone_sets`）。
        #: 判据仍用感知给的 `here_sets`（和寻路同一份 ✓）。
        self.battle_zones = []
        #: ⚠ **派生副本**：`battle_zones` 里**勾了「可以战斗」**那些项的**集合名**列表
        #: （= "可战斗区白名单" ✓）。消费方（`_in_battle_zone` / `_zone_only` /
        #: `_leave_battle_zone_tick` …）全都"拿集合名求交集" ✓ ⇒ 名字不用动 ✓。
        #: **由 `from_dict` / 区域编辑弹窗写入**（真源是 `battle_zones` ✓，改配置请改那个 ✗）。
        #: ⛔ 为什么不做成 `@property`（2026-09-28 踩过 ✗）：这个类有 `__getattr__` 代理
        #:   （给"挂在全局 settings 上的参数"兜底 ✓），而 property 在 `__init__` 赋值**之前**
        #:   被读到时会走 `__getattr__`、代理又回来读 ⇒ **无限递归 ⇒ 进程栈溢出直接崩**
        #:   （`0xC0000409`，连异常都不是 ✗ 极难查）⇒ **普通属性最安全** ✓。
        self.battle_zone_sets = []
        # ⚠ 这里**曾经**有 `walk_mode`（走的方向：center / left / right，还摆进了设置里）。
        # 2026-09-26 用户明确删掉：**走只有一种走法 —— 朝目标集合的 x 中点**，
        # 设置里不许有这个参数（设置是全局参数，用户没提过的一律不加）。
        # 将来若某一步真需要"只按 ←/→"，那是**逐边**配置（foothold 编辑器 →「可到达」窗口）。
        # 上绳梯/下跳**失败后延迟激活时间**（秒，2026-09-26 用户要求）：
        # 以前失败了是**立即**重新激活 —— 失败那一下人往往还在原地、朝向也没变，
        # 立刻重来容易在同一处再歪一次。等一会儿再重新对齐，成功率更高。
        # 0 = 立即重来（老行为）。参数在「路线识别」页（`gui/route_panel.py`）。
        self.climb_retry_delay_s = 1.0
        # 「**延迟增量**(s)」（用户 2026-09-26 要求）：每失败一次，下次等待就**多等**这么多
        # ⇒ 第 1 次等 `climb_retry_delay_s`，第 2 次等「基础 + 增量×1」，第 3 次「基础+增量×2」✓。
        # 为什么：上绳失败常常是"这次就是歪了"，越往后越该多稳一会儿再重来 ✓。0 = 不累加（老行为）。
        # 算法**只有一处**（`retry_delay_s` ✓，便于自检 ✓）。
        self.climb_retry_delay_inc_s = 1.0
        self.debounce_conf = 0.5    # 防抖置信度：高于它的怪框消失后保留位置
        self.debounce_ms = 300      # 防抖时间（毫秒）：保留消失前位置的时长
        # ⚠ 「自动喂宠」整块**已移除**（用户 2026-09-26：他会用**自定义定时行为**自己实现 ✓）。
        #    按键那一层（`keymap["feed_pet"]` / 行为编辑器里的「喂宠」）**必须保留** ✓ ——
        #    那正是他实现它要用的东西，别顺手一起删掉 ✗。
        self.custom_timers = []          # 自定义定时行为：[{name, seq, interval:[min,max]}]
        self.custom_timer_next = {}      # {name: next_monotonic}，运行时状态，不持久化
        #: 「手动触发」的**请求通道**（2026-09-26 用户要求）：界面往里塞名字 → 实时线程里
        #: 的 agent 取走，**立刻演一次并把计时从头排**。**运行时状态，不持久化**（同
        #: `custom_timer_next`：它不是参数，重启后不该还留着一条"待触发" ✗）。
        self.custom_timer_fire = []
        self.facing_timeout_min = 10     # 朝向无变化超时（分钟）：超过就停止自动，0=禁用
        self.player_lost_timeout_min = 3 # 找不到玩家超时（分钟）：超过就停止自动，0=禁用
        self.resetall_interval = 60      # 定时 RELEASEALL（秒）：清空固件侧按键防卡键，0=禁用
        self.anti_afk_enabled = False   # 防掉线开关
        self.anti_afk_type = "hidden_rest"  # 防掉线行为类型，见 ANTI_AFK_TYPES
        self.anti_afk_min = 5           # 防掉线触发时间下限（分钟）
        self.anti_afk_max = 10          # 防掉线触发时间上限（分钟）
        # 「隐身休息」类型的参数：进入/退出隐身各一套行为序列，中间休息随机时长。
        self.anti_afk_enter_seq = []    # 进入隐身行为序列（down/up/delay）
        self.anti_afk_exit_seq = []     # 退出隐身行为序列
        self.anti_afk_rest_min = 10     # 休息时长下限（分钟）
        self.anti_afk_rest_max = 20     # 休息时长上限（分钟）
        # 「被打断重试」：休息期间**触发了自动补血**就算被打断 —— 补血说明隐身没兜住
        # （隐身到期 / 被范围技能扫到 / 有东西在打我们），这时候继续歇着等于等着挨打。
        # 勾选后：立刻收工回战斗，并把下次休息提前到 retry_sec 之后（不再是随机 N~M 分钟）。
        # ⚠ **哪些阶段算"被打断的窗口"按类型不同**（用户 2026-09-26 明确）：
        #    隐身那型 = 进入隐身 → 退出隐身之间；
        #    定点那型 = **从"开始前往指定地点"就开算**（到达后行为 / 正歇着也算；
        #    「结束后前往」那段不算 —— 那时休息已经完成了）。
        #    真源是 `REST_STATE_SPEC` 的 `interrupt` 列，别在这里再抄一份 ✗。
        # ---- 「定点休息」专用（用户 2026-09-26 要求；流程见 docs/开发计划.md P2）----
        self.anti_afk_spot_set = ""     # 指定地点：已注册的 foothold 集合名
        self.anti_afk_spot_seq = []     # 到达后行为（行为序列）
        self.anti_afk_spot_after = ""   # 结束后前往：另一个集合名（空 = 休息完直接回战斗）
        # ---- ⭐ 「休息过程中循环行为」（用户 2026-09-28 要求 ✓）----
        # 原话："定点休息类型，到达后行为按钮下面加个配置……勾选框「休息过程中循环行为」；
        #       勾选后，下方缩进出现参数：「循环行为编辑」→行为编辑器按钮 / 循环时间(s) A ~ B"。
        # 语义：**只在「休息中」（`afk_spot_rest`）这一阶段**按 A~B 秒的间隔**反复**跑这段序列
        #（"到达后行为"是**一次性**的，这段是**循环**的 ✓ —— 两者互不影响 ✓）。
        # ⚠ 它**不挂** `custom_timers`（那套会被 `_pause_timers` 在休息期间冻住 ✗）——
        #   自持 `_loop_ctx` / `_loop_next_ts`，休息期间照跑 ✓（同 `afk_spot_rest` 的做法 ✓）。
        self.anti_afk_spot_loop = False        # 开关（默认关 = 老行为一字不变 ✓）
        #: ⭐ **循环行为列表**（用户 2026-09-28 升级 ✓）：`[{"name": 名字, "seq": 一段行为}, …]`
        #:   —— **一次"循环" = 依次跑完列表里的每一项**，然后等 A~B 秒再来一遍 ✓。
        #:   ⚠ 原来是**一段**序列（`anti_afk_spot_loop_seq`）✗ ⇒ 老配置在 `from_dict` 里
        #:   会**自动变成一项**（名字「循环行为」✓）⇒ 不丢用户已经配好的东西 ✓。
        #:   ⚠ 界面上每项可以**重命名 / 删除 / 双击打开行为编辑器**（用户点名 ✓）。
        self.anti_afk_spot_loop_items = []
        self.anti_afk_spot_loop_min = 30.0     # 循环时间下限（**秒** ✓ 与"休息时长(min)"不同单位 ✓）
        self.anti_afk_spot_loop_max = 60.0     # 循环时间上限（秒）
        #: ⭐ **「休息结束推迟到循环执行完」**（用户 2026-09-28 要求 ✓ 原话："勾上休息过程中
        #:   循环行为时，再加一个开关子参数『休息结束推迟到循环执行完』"）。
        #:   · **不勾**（默认）= 老行为：休息**到点** ⇒ 循环**当场收摊**（`_stop_spot_loop`
        #:     掐断 + 松键 ✓）、立刻转「结束后前往」/ 回战斗 ✓；
        #:   · **勾上** = 休息到点、而**这一轮循环还在演** ⇒ **先不结束休息**，等它把这一轮
        #:     演完（`_loop_ctx` 变 `None` ✓）再结束 ✓。
        #:   ⚠ 要解决的是**同一件事的两套动作抢键**：到点时那一轮可能才演到一半，直接转
        #:     「结束后前往」就是"一边按着循环的键、一边下发寻路的键" ⇒ 两边打架 ✗。
        #:   ⚠ 只管"**休息时间到点**"这一条（`_rest_until` ✓）—— **手动结束休息**另走
        #:     "收尾中"那条（只等**手上这一项** ✓ 见 `tick` 里 `rest_abort` 那段 ✓）。
        #:   ⚠ 实现见 `_hold_rest_for_loop`（含"不许再开新一轮"的理由 ✓）。
        self.anti_afk_spot_loop_hold = False
        self.anti_afk_retry_on_interrupt = False
        self.anti_afk_retry_sec = 60    # 被打断后多久重试下一次休息（秒）
        # ---- 断线自动重连（decision/reconnect.py）----
        self.reconnect_enabled = False        # 断线自动重连开关
        # 玩家框丢多久之后开始探界面（0 = 立刻）。默认 0：断线提示框只显示
        # 两三秒，等 5 秒再探就错过它了，客户端会一直卡在提示框上等人按确定。
        # 探一次只要 ~4ms，不需要为省这点开销推迟。
        self.reconnect_probe_after_lost_sec = 0.0
        self.reconnect_step_timeout_ms = 3000      # 每步等「界面变化」的超时
        self.reconnect_queue_timeout_ms = 180000   # 排队弹窗专用超时（长）
        self.reconnect_max_retry = 3               # 同一步最多重试几次
        self.reconnect_resume_auto = True          # 回到游戏后自动恢复自动打怪
        # 以下是运行时状态，不持久化（to_dict 不导出）：
        self.rest_abort = False         # UI 置 True 请求「手动结束休息」，agent 消费后清掉
        self.rest_request = False       # UI 置 True 请求「手动进入休息」，agent 消费后清掉
        self.rest_state = ""            # 当前休息阶段（"" / afk_enter / afk_rest / afk_exit），UI 只读
        self.rest_until_monotonic = 0.0 # 本次休息的结束时刻（time.monotonic），UI 读它算倒计时
        self.next_afk_monotonic = 0.0   # 下次防掉线触发时刻（0 = 未排期 / 防掉线关着）
        self.rest_pending = False       # 已到触发时间，等攻击范围内的怪清空（「待休息」）
        self.custom_keys = {}           # 自定义按键：{键名: 物理键名}，键名自动命名（custom1...）
        self.reconnect_note = ""        # 重连状态文字（reconnect.py 写，UI 只读）
        self.input_link_ok = True       # 指令通道是否健康（agent 的周期体检写，UI 只读）
        self.input_link_err = ""        # 通道不健康的原因（上面那个为 False 时才有值）
        self.input_resync = False       # UI/手动输入置 True 请求「重同步按键状态」，agent 消费后清掉

    def set_key(self, name, key):
        self.keymap[name] = key

    def to_dict(self):
        """导出所有决策参数（只导**参数**；运行时状态如 enabled / rest_* / custom_timer_next
        不在里面 ✓）。"""
        return {"attack_dist": self.attack_dist,
                "min_attack_dist": self.min_attack_dist,
                "attack_up_dist": self.attack_up_dist,
                "attack_down_dist": self.attack_down_dist,
                "chase_jump_enabled": self.chase_jump_enabled,
                "chase_jump_min": self.chase_jump_min,
                "chase_jump_max": self.chase_jump_max,
                "chase_jump_dash_ms": self.chase_jump_dash_ms,
                "mouse_speed": self.mouse_speed,
                "evade_type": self.evade_type,
                "jump_interval": self.jump_interval,
                "jump_random_prob": self.jump_random_prob,
                "back_jump_seq": self.back_jump_seq,
                "output_seq": self.output_seq,
                "keymap": self.keymap,
                "target_cd": self.target_cd,
                "input_device": self.input_device,
                "attack_cd": self.attack_cd,
                "attack_lock_debounce_ms": self.attack_lock_debounce_ms,
                "min_turn_hold_ms": self.min_turn_hold_ms,
                "turn_output_delay_ms": self.turn_output_delay_ms,
                "player_track_jump": self.player_track_jump,
                # ⭐ 「玩家位置」参数组（用户 2026-09-28 ✓）—— 跟项目一起存 ✓
                "player_foot_offset_px": self.player_foot_offset_px,
                "player_box_min_area_pct": self.player_box_min_area_pct,
                "player_box_area_base_s": self.player_box_area_base_s,
                "player_box_area_tol_pct": self.player_box_area_tol_pct,
                "hp_bar": self.hp_bar, "mp_bar": self.mp_bar,
                "probe_calib": self.probe_calib,
                "hp_color": self.hp_color, "mp_color": self.mp_color,
                "hp_threshold": self.hp_threshold,
                "mp_threshold": self.mp_threshold,
                "pot_cd": self.pot_cd,
                "mob_sets_ttl_s": self.mob_sets_ttl_s,
                "auto_hp_pot": self.auto_hp_pot,
                "auto_mp_pot": self.auto_mp_pot,
                "strategy": self.strategy,
                "vision_top": self.vision_top,
                "vision_bottom": self.vision_bottom,
                "vision_left": self.vision_left,
                "vision_right": self.vision_right,
                "vision_center": self.vision_center,
                "vision_off_x": self.vision_off_x,
                "vision_off_y": self.vision_off_y,
                "sweep_turn_cd": self.sweep_turn_cd,
                "back_range": self.back_range,
                "route_goto_set": self.route_goto_set,
                "align_tol_px": self.align_tol_px,
                "align_hold_ms": self.align_hold_ms,
                "climb_align_gap_ms": self.climb_align_gap_ms,
                "climb_align_near_px": self.climb_align_near_px,
                "climb_retry_delay_s": self.climb_retry_delay_s,
                "climb_retry_delay_inc_s": self.climb_retry_delay_inc_s,
                "goto_timeout_s": self.goto_timeout_s,
                "chase_goto_max_s": self.chase_goto_max_s,
                # ⭐ 「寻路超时后按键」（用户 2026-09-28 加 ✓）：None = 不按 ✓
                "goto_timeout_key": self.goto_timeout_key,
                # ⛔ 「**禁用杀怪寻路**」（用户 2026-09-28 加 ✓）：跟着项目存 ✓
                "disable_chase_pathfinding": bool(self.disable_chase_pathfinding),
                # ⚠ 「**前往重下间隔(s)**」**已移除**（2026-09-28 用户要求：它搬进每一个战斗
                #   区域项、改名「**区域查询CD(s)**」✓）⇒ 不再往这里写 ✓（老项目文件里可能
                #   还留着它 ⇒ `from_dict` 读它**只用于迁移**，见那儿 ✓）。
                "jump_start_px": self.jump_start_px,
                "move_retry_ms": self.move_retry_ms,
                # ⭐ 「路线脚本 → 战斗区域」：**整项**写出去（每项一份配置 ✓ 见 `__init__` ✓）
                "battle_zones": [dict(z) for z in self.battle_zones],
                # ⚠ `battle_zone_sets` 是**派生副本**（真源是上面那个 ✓）—— 还是写出去一份：
                #   ① 老版本 / 老工具读得懂 ✓；② `check_ui` 有"设置字段必须能存下去"的检查 ✓
                #   （派生字段也得写，否则它一直提示 ✗）。读的时候**以 `battle_zones` 为准** ✓。
                "battle_zone_sets": list(self.battle_zone_sets),
                "debounce_conf": self.debounce_conf,
                "debounce_ms": self.debounce_ms,
                # ⚠ 「自动喂宠」的三个键已移除（2026-09-26）—— 老项目文件里可能还留着，
                #    `from_dict` **直接忽略**它们 ✓（已经不是参数了，别再写回去 ✗）。
                "custom_timers": self.custom_timers,
                "facing_timeout_min": self.facing_timeout_min,
                "player_lost_timeout_min": self.player_lost_timeout_min,
                "resetall_interval": self.resetall_interval,
                "anti_afk_enabled": self.anti_afk_enabled,
                "anti_afk_type": self.anti_afk_type,
                "anti_afk_min": self.anti_afk_min,
                "anti_afk_max": self.anti_afk_max,
                "anti_afk_enter_seq": self.anti_afk_enter_seq,
                "anti_afk_exit_seq": self.anti_afk_exit_seq,
                "anti_afk_rest_min": self.anti_afk_rest_min,
                "anti_afk_rest_max": self.anti_afk_rest_max,
                "anti_afk_spot_set": self.anti_afk_spot_set,
                "anti_afk_spot_seq": self.anti_afk_spot_seq,
                "anti_afk_spot_after": self.anti_afk_spot_after,
                "anti_afk_spot_loop": self.anti_afk_spot_loop,
                "anti_afk_spot_loop_items": self.anti_afk_spot_loop_items,
                "anti_afk_spot_loop_min": self.anti_afk_spot_loop_min,
                "anti_afk_spot_loop_max": self.anti_afk_spot_loop_max,
                "anti_afk_spot_loop_hold": self.anti_afk_spot_loop_hold,
                "anti_afk_retry_on_interrupt": self.anti_afk_retry_on_interrupt,
                "anti_afk_retry_sec": self.anti_afk_retry_sec,
                "reconnect_enabled": self.reconnect_enabled,
                "reconnect_probe_after_lost_sec": self.reconnect_probe_after_lost_sec,
                "reconnect_step_timeout_ms": self.reconnect_step_timeout_ms,
                "reconnect_queue_timeout_ms": self.reconnect_queue_timeout_ms,
                "reconnect_max_retry": self.reconnect_max_retry,
                "reconnect_resume_auto": self.reconnect_resume_auto,
                "custom_keys": self.custom_keys}

    def save(self):
        """落盘：整份写回**当前项目**（钩子由界面注册，见 set_save_hook）。

        **没注册钩子（没打开项目）时什么都不写**：那种情况下界面上的改动只活在
        内存里（实时预览照样立刻生效），一打开项目就被项目的值覆盖。
        参数必须有个明确的归属，不然又回到「不知道这份是谁的」。

        enabled 不存 —— 自动开关是运行时状态，重启后总是关闭。
        写失败不抛：参数存不下去是坏事，但不能连界面一起炸。
        """
        data = self.to_dict()
        if _save_hook is not None:
            try:
                _save_hook(data)
            except Exception:
                pass

    def from_dict(self, data):
        """从 dict 恢复所有决策参数（参数模板加载也走这里）。"""
        data = data or {}
        self.attack_dist = float(data.get("attack_dist", 80.0))
        self.min_attack_dist = int(data.get("min_attack_dist", 0))
        # 「攻击范围 = 矩形」的那两条竖边（2026-09-27 新增）：**负数 = 该方向不限** ✓
        # ⇒ 老项目文件里没有这两个键 ⇒ 兜底 **-1** ⇒ 行为一点不变 ✓
        # ⚠ 别退回 0：0 是"那一方向没有可攻击区域"（按 0 算 ✗ 不是不限 ✗）。
        self.attack_up_dist = int(data.get("attack_up_dist", -1))
        self.attack_down_dist = int(data.get("attack_down_dist", -1))
        self.chase_jump_enabled = bool(data.get("chase_jump_enabled", False))
        self.chase_jump_min = int(data.get("chase_jump_min", 0))
        self.chase_jump_max = int(data.get("chase_jump_max", 50))
        self.chase_jump_dash_ms = int(data.get("chase_jump_dash_ms", 0))
        self.mouse_speed = float(data.get("mouse_speed", 1.0))
        self.evade_type = data.get("evade_type", "jump")
        self.jump_interval = int(data.get("jump_interval", 200))
        self.jump_random_prob = float(data.get("jump_random_prob", 0.1))
        self.back_jump_seq = self._load_seq(data.get("back_jump_seq"))
        self.output_seq = self._load_seq(data.get("output_seq"), DEFAULT_OUTPUT_SEQ)
        km = data.get("keymap") or {}
        for k, v in km.items():
            if k in self.keymap and (isinstance(v, str) or v is None):
                self.keymap[k] = v
        cd = data.get("target_cd")
        if isinstance(cd, (list, tuple)) and len(cd) == 2:
            self.target_cd = [int(cd[0]), int(cd[1])]
        dev = data.get("input_device")
        if dev in ("local", "remote", "serial"):
            self.input_device = dev
        # 老配置/老模板里的 `input_delay` **不再读**（2026-09-27 删掉的功能）：留着不报错，
        # 下次保存时这个键自然就没了（参数模板同理 ✓）。
        self.attack_cd = int(data.get("attack_cd", 0))
        self.attack_lock_debounce_ms = int(data.get("attack_lock_debounce_ms", 300))
        self.min_turn_hold_ms = int(data.get("min_turn_hold_ms", 0))
        self.turn_output_delay_ms = int(data.get("turn_output_delay_ms", 0))
        # 旧键 player_debounce_dist 兼容读一次（改名前的配置 / 模板里是这个）
        self.player_track_jump = int(data.get(
            "player_track_jump", data.get("player_debounce_dist", 150)))
        # ⭐ 「玩家位置」参数组（用户 2026-09-28 ✓）—— ⚠ 老项目文件里**没有**这些格
        #   ⇒ 全部兜底成"这个功能关着"（= 老行为一字不变 ✓）：偏移 0 / 面积闸 0（关）/
        #   箭头长度给个能看见的默认值（它只是画给人看 ✓ 不参与决策 ✓）。
        self.player_foot_offset_px = int(data.get("player_foot_offset_px", 0) or 0)
        self.player_box_min_area_pct = float(
            data.get("player_box_min_area_pct", 0.0) or 0.0)
        # ⚠ **旧键 `player_box_area_base_n`（拍）仍读一次**（别让老项目把值丢了 ✗）：
        #   按 30fps 折算成秒 ✓（30 拍 ⇒ 1 秒 ✓）。
        _bs = data.get("player_box_area_base_s")
        if _bs is None:
            _bn = data.get("player_box_area_base_n")
            # ⚠ 旧键在 ⇒ 按 30fps 折算（30 拍 ⇒ 1 秒 ✓）；**两个都没有 ⇒ 3.0 秒**（默认值 ✓
            #   别掉到 1.0 ✗ —— `data.get(k, default)` 的 default 是**立即求值**的，写成一行的
            #   话"没有旧键"也会算出 1.0 ✗ 这个坑当场踩过 ✓）。
            _bs = (float(_bn) / 30.0) if _bn else 3.0
        self.player_box_area_base_s = max(0.5, float(_bs or 3.0))
        self.player_box_area_tol_pct = float(
            data.get("player_box_area_tol_pct", 0.0) or 0.0)
        # ⚠ 箭头那几个键（`player_arrow_*`）**不再读**（2026-09-28 搬到 `config/ui.yaml`
        #   的 `vis:` 段了 ✓）：老项目文件里若还留着这几个键，**留着不报错**，
        #   下次保存时自然就没了 ✓（同 `input_delay` 那条的处理 ✓）。
        self.hp_bar = load_rect(data.get("hp_bar"))
        self.mp_bar = load_rect(data.get("mp_bar"))
        self.probe_calib = dict(data.get("probe_calib") or {})
        self.hp_color = self._load_color(data.get("hp_color"))
        self.mp_color = self._load_color(data.get("mp_color"))
        self.hp_threshold = int(data.get("hp_threshold", 30))
        self.mp_threshold = int(data.get("mp_threshold", 20))
        self.pot_cd = int(data.get("pot_cd", 1000))
        # ⭐ 「怪物判定缓存时长(s)」（老配置没有 ⇒ 3.0 = 老行为一字不变 ✓；0 = 不缓存 ✓）
        self.mob_sets_ttl_s = max(0.0, float(data.get("mob_sets_ttl_s", 3.0)))
        self.auto_hp_pot = bool(data.get("auto_hp_pot", False))
        self.auto_mp_pot = bool(data.get("auto_mp_pot", False))
        self.strategy = data.get("strategy", "patrol")
        self.vision_top = int(data.get("vision_top", 200))
        self.vision_bottom = int(data.get("vision_bottom", 200))
        self.vision_left = int(data.get("vision_left", 200))
        self.vision_right = int(data.get("vision_right", 200))
        self.vision_center = bool(data.get("vision_center", False))
        self.vision_off_x = int(data.get("vision_off_x", 0))
        self.vision_off_y = int(data.get("vision_off_y", 0))
        self.sweep_turn_cd = int(data.get("sweep_turn_cd", 1000))
        self.back_range = int(data.get("back_range", 100))
        self.route_goto_set = str(data.get("route_goto_set") or "")
        # 老项目文件里没有这个键时用 **10**（= 上面那个默认值 ✓，别再退回 6 —— 那是在
        # 噪声里，见那儿的说明 ✓）
        self.align_tol_px = int(data.get("align_tol_px", 10))
        self.align_hold_ms = int(data.get("align_hold_ms", 250))
        # 「对齐绳梯移动延迟(ms)」：老项目文件里没有这个键 ⇒ 用默认 180（= 老行为 ✓）
        try:
            self.climb_align_gap_ms = max(0, int(data.get("climb_align_gap_ms", 180)))
        except (TypeError, ValueError):
            self.climb_align_gap_ms = 180
        # 「开始对齐绳梯x的距离(px)」：老项目文件里没有这个键 ⇒ 用默认 20（= `route.NEAR_PX`
        # = 老行为 ✓）。比 0 小没意义（那等于"永远按住"）⇒ 至少 1。
        try:
            self.climb_align_near_px = max(1, int(data.get("climb_align_near_px", 20)))
        except (TypeError, ValueError):
            self.climb_align_near_px = 20
        self.climb_retry_delay_s = float(data.get("climb_retry_delay_s", 1.0))
        self.climb_retry_delay_inc_s = float(data.get("climb_retry_delay_inc_s", 1.0))
        self.goto_timeout_s = float(data.get("goto_timeout_s", 30.0))
        # ⭐ 「追怪寻路.duration(s)」（用户 2026-09-28 ✓）—— ⚠ 老项目文件里**没有**这一格
        #   ⇒ 兜底成 **0 = 不启用** ✓（老行为一字不变 ✓）。
        self.chase_goto_max_s = max(
            0.0, float(data.get("chase_goto_max_s", 0.0) or 0.0))
        # ⭐ 「**寻路超时后按键**」（用户 2026-09-28 加 ✓）：老项目文件里没这个键 ⇒ `None`
        #   （= 不按 ✓ **行为一点不变** ✓）。空串 / 非字符串也当 `None` ✓
        #   （手改坏的项目文件别被带进运行期 ✗ —— 那种值发出去就是"按一个不存在的键" ✗）。
        _gk = data.get("goto_timeout_key")
        self.goto_timeout_key = (str(_gk).strip()
                                 if isinstance(_gk, str) and str(_gk).strip() else None)
        # ⛔ 「**禁用杀怪寻路**」（用户 2026-09-28 加 ✓）：老项目文件里**没这个键** ⇒ `False`
        #   （= 老行为 ✓ 一点不变 ✓）。手改坏的 yaml 给个非布尔值也照 `bool()` 收 ✓。
        self.disable_chase_pathfinding = bool(data.get("disable_chase_pathfinding", False))
        # ⚠ 「**前往重下间隔(s)**」**2026-09-28 已移除**（用户要求：搬进**每一个战斗区域项**、
        #   改名「**区域查询CD(s)**」✓）⇒ 这里**只把老值读出来放在局部变量里**备用 ✓（不挂到
        #   设置上 ✗）—— 老项目文件是"一串集合名 + 一个全局 CD"，下面 `battle_zones` 的迁移
        #   会用它把每一项**原样**补齐 ✓。下限 0.5：再小就是"每拍重下前往"= 刷屏 ✗。
        try:
            _legacy_zone_cd = max(0.5, float(data.get("goto_retry_s", ZONE_GOTO_RETRY_S)))
        except (TypeError, ValueError):
            _legacy_zone_cd = ZONE_GOTO_RETRY_S
        # ⚠ 「**卡住判定时长(s)**」这一格 **2026-09-27 用户要求移除** ✓（"统一采用移动操作
        # 尝试间隔(ms)"）：老项目文件里可能还留着 `stall_s` 键 ⇒ **读进来直接忽略** ✓
        # （不再往设置上挂、也不报错 —— 手改过的配置不该因为这个打不开 ✗）。
        # 那些"多久没进展算卡住"的判据统一读 `move_retry_ms`（见 `move_retry_ms` 那段说明 ✓）。
        # 「起跳距离(px)」（2026-09-27 新增，「跳」用的**临时参数** ✓）：
        # 老项目文件里没有这个键 ⇒ 兜底 **0**（= 进了目标 foothold 范围才跳 ✓，不拍数 ✓）。
        try:
            self.jump_start_px = max(0, int(data.get("jump_start_px", 0) or 0))
        except (TypeError, ValueError):
            self.jump_start_px = 0
        # 「移动操作尝试间隔(ms)」（2026-09-27 从 `climb_reassert_s` **改名 + 改单位**）：
        # ⚠ 老项目文件里存的是**秒**（float，默认 3.0）—— 不加这一步折算，"3.0" 会被当成
        #   **3 毫秒** ⇒ 爬绳刚"不动"就判失败、下跳每 3ms 重试一次（键盘风暴 ✗）。
        #   所以：先读新键；没有新键但有老键 ⇒ ×1000 折算；两个都没有 ⇒ 3000（老默认 ✓）。
        if "move_retry_ms" in data:
            try:
                self.move_retry_ms = max(0, int(data.get("move_retry_ms") or 0))
            except (TypeError, ValueError):
                self.move_retry_ms = 3000
        elif "climb_reassert_s" in data:
            try:
                self.move_retry_ms = max(0, int(round(
                    float(data.get("climb_reassert_s") or 0.0) * 1000.0)))
            except (TypeError, ValueError):
                self.move_retry_ms = 3000
        else:
            self.move_retry_ms = 3000
        # ⭐ 「**路线脚本 → 战斗区域**」（2026-09-28 重构 ✓）：新键 `battle_zones`（每项一份
        #   配置 ✓）优先；老项目文件里只有 `battle_zone_sets`（一串集合名 ✓）⇒ **迁移**：
        #   CD 取老的「前往重下间隔(s)」全局值（`_legacy_zone_cd` ✓）、其余取默认
        #   ⇒ **老配置的行为一点不变** ✓。
        self.battle_zones = self._load_battle_zones(data, _legacy_zone_cd)
        # ⚠ **必须同步派生副本**（`battle_zone_sets` ✓ 老消费方读的是它 ✓ —— `__init__` 里
        #   已经同步过一次空表 ⇒ 这里读完真配置再同步一次 ✓；改配置的地方也要调 ✓）。
        self.sync_battle_zone_sets()
        # ⚠ 老配置里可能还留着 `walk_mode` 键 —— **直接忽略**（用户 2026-09-26 删掉了它：
        # 走只有"朝集合中点"一种走法）⇒ 不读、也不再写回（见 `to_dict`）。
        self.debounce_conf = float(data.get("debounce_conf", 0.5))
        self.debounce_ms = int(data.get("debounce_ms", 300))
        # ⚠ 老项目文件里可能还留着 `auto_feed_pet` / `feed_interval_min|max`
        #    （「自动喂宠」2026-09-26 已整块移除）—— **直接忽略** ✓，不读、也不再写回 ✓。
        #    想喂宠就用「自定义定时行为」配一条（按键那层的「喂宠」键还在 ✓）。
        self.custom_timers = self._load_timers(data.get("custom_timers"))
        self.facing_timeout_min = float(data.get("facing_timeout_min", 10))
        self.player_lost_timeout_min = float(data.get("player_lost_timeout_min", 3))
        self.resetall_interval = int(data.get("resetall_interval", 60))
        self.anti_afk_enabled = bool(data.get("anti_afk_enabled", False))
        atype = str(data.get("anti_afk_type") or "")
        self.anti_afk_type = (atype if any(atype == t for t, _ in ANTI_AFK_TYPES)
                              else ANTI_AFK_TYPES[0][0])
        self.anti_afk_min = float(data.get("anti_afk_min", 5))
        self.anti_afk_max = float(data.get("anti_afk_max", 10))
        if self.anti_afk_max < self.anti_afk_min:
            self.anti_afk_max = self.anti_afk_min
        self.anti_afk_enter_seq = self._load_seq(data.get("anti_afk_enter_seq"), [])
        self.anti_afk_exit_seq = self._load_seq(data.get("anti_afk_exit_seq"), [])
        self.anti_afk_rest_min = float(data.get("anti_afk_rest_min", 10))
        self.anti_afk_rest_max = float(data.get("anti_afk_rest_max", 20))
        if self.anti_afk_rest_max < self.anti_afk_rest_min:
            self.anti_afk_rest_max = self.anti_afk_rest_min
        self.anti_afk_spot_set = str(data.get("anti_afk_spot_set") or "")
        self.anti_afk_spot_seq = self._load_seq(data.get("anti_afk_spot_seq"), [])
        self.anti_afk_spot_after = str(data.get("anti_afk_spot_after") or "")
        # ⭐ 「休息过程中循环行为」：老配置里没有这几格 ⇒ 默认**关**（老行为一字不变 ✓）
        self.anti_afk_spot_loop = bool(data.get("anti_afk_spot_loop", False))
        # ⭐ **循环行为列表**（用户 2026-09-28 ✓）：每项 = `{"name": 名字, "seq": 一段行为}`
        self.anti_afk_spot_loop_items = self._load_seq_items(
            data.get("anti_afk_spot_loop_items"))
        # ⚠⚠ **老配置迁移**：这一格原来是**一段**序列（`anti_afk_spot_loop_seq` ✓）——
        #  读到它就把它包成**一项**（名字「循环行为」✓）⇒ 用户已经配好的东西**不丢** ✓
        #  （不清个"迁移标记"这种账 ✗：老键还在就再包一次是无害的 —— 因为新键一旦写出，
        #    `to_dict` 就不会再写老键了 ✓）。
        if not self.anti_afk_spot_loop_items:
            _old = self._load_seq(data.get("anti_afk_spot_loop_seq"), [])
            if _old:
                self.anti_afk_spot_loop_items = [{"name": "循环行为", "seq": _old}]
        self.anti_afk_spot_loop_min = max(0.1, float(
            data.get("anti_afk_spot_loop_min", 30.0)))
        self.anti_afk_spot_loop_max = max(
            self.anti_afk_spot_loop_min,
            float(data.get("anti_afk_spot_loop_max", 60.0)))
        # ⭐ 「休息结束推迟到循环执行完」（用户 2026-09-28 ✓）：老配置里没有这格
        #   ⇒ 默认**关**（老行为一字不变 ✓ 口径见 `__init__` 那段注释 ✓）
        self.anti_afk_spot_loop_hold = bool(data.get("anti_afk_spot_loop_hold", False))
        self.anti_afk_retry_on_interrupt = bool(
            data.get("anti_afk_retry_on_interrupt", False))
        # 至少 1 秒：0/负值会变成"退出隐身的同时立刻又要休息"，来回抖
        self.anti_afk_retry_sec = max(1.0, float(data.get("anti_afk_retry_sec", 60)))
        # 断线自动重连
        self.reconnect_enabled = bool(data.get("reconnect_enabled", False))
        self.reconnect_probe_after_lost_sec = float(
            data.get("reconnect_probe_after_lost_sec", 0.0))
        self.reconnect_step_timeout_ms = int(data.get("reconnect_step_timeout_ms", 3000))
        self.reconnect_queue_timeout_ms = int(
            data.get("reconnect_queue_timeout_ms", 180000))
        self.reconnect_max_retry = int(data.get("reconnect_max_retry", 3))
        self.reconnect_resume_auto = bool(data.get("reconnect_resume_auto", True))
        ck = data.get("custom_keys") or {}
        if isinstance(ck, dict):
            self.custom_keys = {str(k): (v if isinstance(v, str) or v is None else None)
                                for k, v in ck.items()}

    @staticmethod
    def _load_color(v):
        """读回颜色范围 [[B,G,R],[B,G,R]]，非法返回 None。"""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                low = [int(x) for x in v[0]]
                high = [int(x) for x in v[1]]
                if len(low) == 3 and len(high) == 3:
                    return [low, high]
            except Exception:
                pass
        return None

    def sync_battle_zone_sets(self):
        """把 `battle_zones`（真源 ✓）里**勾了「禁止战斗」**那些项的集合名同步到
        `battle_zone_sets`（派生副本 ✓）。

        ⭐ **2026-09-28 用户重构**：原来的「限制战斗区域」（人在区域外 ⇒ 这一拍不打架、
        先下「前往」回去 ✓ 只打区域里的怪 ✓）**搬进每一项自己的「禁止战斗」勾选** ✓
        ⇒ 派生副本**只收 `no_fight=True` 的项** ✓✓✓
        ⇒ 老的消费方（`_in_battle_zone` / `_zone_only` / `_leave_battle_zone_tick` /
        tick 里那道门 …）**一个字都不用改**，语义自动变成"只对**禁战区**生效" ✓
        （这正是当初把它做成派生副本的价值 ✓）。

        谁调：`from_dict`（读完配置 ✓）、以及编辑弹窗保存时 ✓ ——
        **一处实现**（别各写一遍列表推导 ✗：漏一处就会出现"弹窗改完了、筛怪还按老名单"✗）。

        ⚠ **必须自己在所有"改 `battle_zones`"的地方调它** ✓（它做不成自动同步 —— 见
        `battle_zone_sets` 那段说明：这个类有 `__getattr__` 代理，`@property` 会**无限递归**
        把进程打崩 ✗）。
        """
        self.battle_zone_sets = [str(z.get("set") or "").strip()
                                 for z in (self.battle_zones or [])
                                 # ⚠⚠ **`can_fight`（不是取反的 `no_fight`）**（2026-09-28 改名 ✗）：
                                 #   这个键原来叫 `no_fight`，但消费方一直是**白名单**语义
                                 #   （`here & zones` ✓）⇒ 它**本来就是"这块能打"** ✓
                                 #   ⇒ 改名时**值原样保留** ✓（取反 = 把已配好的能打区全废掉 ✗）。
                                 if z.get("can_fight")
                                 and str(z.get("set") or "").strip()]

    @staticmethod
    def _load_battle_zones(data, legacy_cd=None):
        """读「**战斗区域**」项（`battle_zones` ✓）；没有新键就**迁移**老的 `battle_zone_sets` ✓。

        迁移规则（2026-09-28：用户把「前往重下间隔(s)」搬进每一项 ✓）：
          · 老的**每一个集合名** ⇒ 一项 ✓；
          · `cd_s` 取老那个**全局**值（`legacy_cd` ✓，没有就用 `ZONE_GOTO_RETRY_S` ✓）
            ⇒ **老配置行为一点不变** ✓；
          · `idle_foothold` / `fight_dst` 留空、`fight_max_s = 0`（= 不限 ✓）。
        ⚠ **一律清洗**（手改坏的 yaml 不该被带进运行期 ✗）：集合名去空格；**同一集合只留第一项**
          （它同时是"这个集合归哪一项管"的判据 ⇒ 重复会让取值变成掷骰子 ✗）；数值钳到合法范围 ✓。
        """
        raw = data.get("battle_zones")
        migrated = not (isinstance(raw, list) and raw)
        if migrated:
            raw = [{"set": str(x)} for x in (data.get("battle_zone_sets") or [])]
        # ⭐ **「禁止战斗」的老配置迁移**（2026-09-28 ✓）—— 这条**必须准**，否则升级后
        #   "限制战斗区域"会**悄悄失效** ✗（那是用户最在意的老行为 ✓）：
        #   · `raw` 的**每一项都还没有 `no_fight` 键** ⇒ 这是"重构之前"存下来的配置 ✓
        #     那时 `battle_zone_sets`（派生副本）装的是**全部**区域项的名字，而它当时的
        #     **语义就是**「限制战斗区域」（人在区域外 ⇒ 先回区域 ✓）
        #     ⇒ **副本名单里那些名字对应的项，标 `no_fight=True`** ✓ ⇒ 老项目行为一点不变 ✓；
        #   · 反之（至少一项已有该键）⇒ 新格式 ⇒ 副本只是"禁战名单"的副本
        #     ⇒ **只信每项自己的 `no_fight`** ✓（拿副本当名单会把"只配了参数、没禁战"的项
        #       也算进去 ✗）。
        _raw_items = [it for it in raw if isinstance(it, dict)]
        # ⚠ 认**两个键名**：`can_fight`（2026-09-28 起的新名 ✓）与 `no_fight`（旧名 ✓，
        #   **语义相同**——那时也被当"能打"消费 ✓ 见 `sync_battle_zone_sets` 的说明 ✓）
        #   ⇒ 任一存在就说明是"新格式（逐项带勾选）"✓，不再是"一串集合名"的老格式 ✓。
        legacy_flag = migrated or not any(
            ("can_fight" in it) or ("no_fight" in it) for it in _raw_items)
        _legacy_sets = set()
        if legacy_flag and not migrated:
            _legacy_sets = set(str(x).strip()
                               for x in (data.get("battle_zone_sets") or []))
        out, seen = [], set()
        for it in raw:
            if not isinstance(it, dict):
                continue
            name = str(it.get("set") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            try:
                cd = max(0.5, float(it.get("cd_s", legacy_cd if migrated
                                               else ZONE_GOTO_RETRY_S)))
            except (TypeError, ValueError):
                cd = ZONE_GOTO_RETRY_S
            try:
                fmax = max(0.0, float(it.get("fight_max_s", 0.0) or 0.0))
            except (TypeError, ValueError):
                fmax = 0.0
            if migrated:
                # 老迁移：这些项本来就是从「限制战斗区域」那份名单造的 ⇒ **都能打** ✓
                # （那份名单当年就是"允许打架的集合" ✓ 消费方是白名单 ✓）
                cf = True
            elif legacy_flag:
                cf = name in _legacy_sets          # 老格式：副本名单里有的才算"能打" ✓
            else:
                # 新格式：先认 `can_fight`（新名 ✓）；⚠ 老键 `no_fight` **照抄原值、不取反** ✗✗
                #   —— 它当年也是"能打"的意思（白名单 ✓），取反会让你**丢掉所有能打区** ✗
                cf = bool(it.get("can_fight", it.get("no_fight")))
            out.append({"set": name,
                        "cd_s": cd,
                        "idle_foothold": str(it.get("idle_foothold") or "").strip(),
                        "fight_max_s": fmax,
                        "fight_dst": str(it.get("fight_dst") or "").strip(),
                        "can_fight": cf})
        return out

    def _load_seq_items(self, v):
        """读回**循环行为列表**：`[{"name": 名字, "seq": 一段行为}, …]`（用户 2026-09-28 ✓）。

        ⚠ 容错口径与 `_load_seq` 一致：**坏数据当空**（别让一个坏项把整份配置带崩 ✗）；
          名字空 ⇒ 给个占位名（界面上要显示一行字 ✓ 空名字会变成一个看不见的项 ✗）。
        ⚠ 是**实例方法**（不是 `@staticmethod`）：里面要复用 `self._load_seq` ✓。
        """
        out = []
        for it in (v or []):
            if not isinstance(it, dict):
                continue
            seq = self._load_seq(it.get("seq"), [])
            out.append({"name": str(it.get("name") or "（未命名）"), "seq": seq})
        return out

    @staticmethod
    def _load_seq(v, default=None):
        """读回行为序列（list of dict），非法则返回 default（默认回身输出序列）。

        元素可带 prob（执行几率 0~100，默认 100），非 100 时才存字段。
        """
        if default is None:
            default = [dict(e) for e in DEFAULT_BACK_JUMP_SEQ]
        if not isinstance(v, list):
            return [dict(e) for e in default]
        out = []
        for e in v:
            if not isinstance(e, dict):
                continue
            t = e.get("type")
            if t == "delay":
                elem = {"type": "delay", "ms": int(e.get("ms", 0))}
            elif t in ("down", "up"):
                key = e.get("key")
                if not (isinstance(key, str) and key):
                    continue
                elem = {"type": t, "key": key}
            else:
                continue
            prob = e.get("prob", 100)
            try:
                prob = max(0, min(100, int(prob)))
            except Exception:
                prob = 100
            if prob != 100:
                elem["prob"] = prob
            # 递归加载「触发后执行」的子序列
            then = e.get("then")
            if isinstance(then, list) and then:
                sub = DecisionSettings._load_seq(then, [])
                if sub:
                    elem["then"] = sub
            out.append(elem)
        return out or [dict(e) for e in default]

    @staticmethod
    def _load_timers(v):
        """读回自定义定时行为列表：[{name, seq, interval:[min,max], paused?, paused_left?}]。

        `paused` / `paused_left` 只在暂停过时才存在（UI 上的暂停按钮写的）——
        跟着项目一起存，重开项目仍是暂停状态、倒计时还冻在原来那个数上。
        """
        if not isinstance(v, list):
            return []
        out = []
        for e in v:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "").strip()
            if not name:
                continue
            seq = DecisionSettings._load_seq(e.get("seq"), [])
            iv = e.get("interval")
            if not isinstance(iv, (list, tuple)) or len(iv) != 2:
                continue
            try:
                lo = max(0.0, float(iv[0]))
                hi = max(lo, float(iv[1]))
            except Exception:
                continue
            item = {"name": name, "seq": seq, "interval": [lo, hi]}
            # 「休息时暂停计时」（每条行为一份，缺省 True = 老行为）：**只在 False 时落一格** ✓。
            # ⚠ 这一格必须**读回来** —— 原来这里只重建 name/seq/interval ⇒ 用户勾掉的那个
            #   （False）过了存档就没了、重开项目又按默认 True 算 ✗（2026-09-26 用户问的
            #   "关了 GUI 再打开还读得到吗"就是它 ✗）。而且更糟：下一次 `save()` 用内存里
            #   这份重建过的 list 整条写回 ⇒ 文件里那格会被**抹掉** ✗（设置永久丢）。
            #   True 是默认值，不占版面（和下面 `paused` 同一个套路：非默认态才写 ✓）。
            if "pause_on_rest" in e and not bool(e.get("pause_on_rest")):
                item["pause_on_rest"] = False
            if e.get("paused"):
                try:
                    left = max(0.0, float(e.get("paused_left") or 0.0))
                except Exception:
                    left = 0.0
                item["paused"] = True
                item["paused_left"] = left
            out.append(item)
        return out


#: 当前**正在跑的** CombatAgent（「实时」线程启动时登记、退出时注销）。
#: 为什么要有它：「路线识别」页的「命令前往」要**主动下命令**（`start_climb`），而 agent
#: 原本只是 `gui/live_thread.py` 里的**局部变量** ⇒ 面板拿不到，只能"算一遍给你看"。
#: 跨线程只做**引用赋值**（GUI 线程写、推理线程读）—— 和 settings 一个路子，不加锁：
#: 读到 None 就按"实时没在跑"处理（见 `route_panel._command_first_step`），不会误发按键。
CURRENT = None

#: 寻路任务**结束后**，那句"为什么结束"还在界面上挂多久（秒）。
#: 为什么要有：用户问的常是"为什么在 -170 就松开了 ↑"，而任务一结束那行就消失了
#: ⇒ 看见的时候已经无从查证（2026-09-26 连着两次都是这么绕远的）。做法同
#: `decision/reconnect.py` 的 NOTE_KEEP。
GOTO_NOTE_KEEP_S = 10.0

#: **卡键看门狗**的判据（秒）：输出序列的键已经按了这么久，而角色**早就不在输出状态**
#: ⇒ 认定漏放：记一笔（perf 的 `key_stuck` + 一句说明）、**松开**、清掉上下文。
#:
def attack_edge(v):
    """攻击范围那一个方向的数 → `None`（= **不限**）或 `float`（**含 0** ✓）。

    ⚠ 用户 2026-09-27 定的口径（我一开始把 0 当"不限"，是错的 ✗）：
      · **`< 0` ⇒ 不限**（这才是"无限"的写法 ✓）；
      · **`0` ⇒ 就是 0**（该方向没有可攻击区域 —— 上/下配 0 就真的打不到那一边 ✓）；
      · `> 0` ⇒ 具体距离 ✓。
    判据（`DecisionAgent._in_box`）与绘制（`attack_box_rect`）**共用这一处** ✓。
    """
    f = float(v or 0.0)
    return None if f < 0 else f


def attack_box_rect(cx, cy, ad, up=0.0, down=0.0, facing=1,
                    frame_w=None, frame_h=None, from_d=0.0):
    """一个**矩形框**的四条边（画面坐标 → `(left, top, right, bottom)`）。

    「攻击盲区框 / 攻击范围框 / 追击起跳框」**都走它一处** —— 判据那边走
    `DecisionAgent._in_box`，两边**同源**：水平从角色中心朝**朝向**量、竖直向上 `up`、
    向下 `down` ✓（各画一套的话，画出来的框和打得到的范围迟早对不上 ✗）。

    水平范围 = **从 `from_d` 到 `ad`**（都按"离角色中心的距离"算 ✓）。三个框各自的传法：
      · 「攻击**范围**框」= 可攻击区 ⇒ `from_d=min_ad`（最小攻击距离）、`ad=max_ad` ✓
        —— ⚠ **不含盲区那一块**（用户 2026-09-27 的图上就是这么并排的：盲区框在外框
        **左边**、两者不重叠 ✓，而且这也正是判据那边的口径 `[min, max]` ✓）；
      · 「攻击**盲区**框」= `from_d=0`、`ad=min_ad` ✓；
      · 「**追击起跳**框」= `from_d=max_ad + chase_jump_min`、`ad=max_ad + chase_jump_max` ✓
        （区间本来就定义成"相对最大攻击距离的偏移" ✓）。

    ⚠ 返回 **`None` = 这个框是空集**（用户 2026-09-27："面积是 0 则直接跳过对应的攻击
      判定，也不应该绘制"）⇒ 调用方**不判、不画** ✓。空集两种情形：
        · 水平跨度 ≤ 0（`ad - from_d <= 0` —— 比如「最大攻击距离」配 0，或者盲区配得
          跟它一样大 ✓）；
        · 竖直跨度 = 0：向上、向下**都恰好配 0** ✓。
      ⚠ 「不限」（**负数** ⇒ `attack_edge` 给 None）**不算空** ✓，那是无限大 ✓。
    ⚠ 传了 `frame_w/frame_h` 时，某一方向"不限"就画到画面边（把"不限"如实画出来 ✓）；
      没传就退回 `cy`（退化成一条水平线 ✓）。
    ⚠ 「跳跃攻击范围框」**不用它** —— 那个的逻辑还没做（见 `DecisionSettings` 里那条
      "记一笔"），连"框该多大"都还没定义 ⇒ 现在**不画**（画了就是拍脑袋 ✗）✓。
    """
    ex = float(cx)
    dirn = 1 if facing >= 0 else -1
    ad_v, lo_v = float(ad), float(from_d)
    up_v, down_v = attack_edge(up), attack_edge(down)
    # **空集**：水平跨度 0、或竖直跨度 0（上下都恰好是 0）⇒ None（不判、不画 ✓）
    if (ad_v - lo_v) <= 0:
        return None
    if up_v is not None and down_v is not None and (up_v + down_v) <= 0:
        return None
    near, far = ex + dirn * lo_v, ex + dirn * ad_v
    left, right = (near, far) if far >= near else (far, near)
    top = (cy - up_v) if up_v is not None else (0.0 if frame_h else float(cy))
    bottom = (cy + down_v) if down_v is not None else (
        (float(frame_h) - 1.0) if frame_h else float(cy))
    return left, top, right, bottom


def retry_delay_s(base_s, inc_s, attempt):
    """第 `attempt` 次失败后要等多久再重新激活（秒）—— **规则只写这一处**（便于自检 ✓）。

    第 1 次失败等 `base_s`；之后**每失败一次多等 `inc_s`**（用户 2026-09-26：
    "每次失败后延迟的时间累加这个时间"✓）。两个值都按 0 兜底（负值当 0 ✓）。
    """
    return (max(0.0, float(base_s))
            + max(0.0, float(inc_s)) * max(0, int(attempt) - 1))


#: 「不在战斗区域 ⇒ 先回去」那条规则**最短多久重下一次「前往」命令**（秒）。
#: 为什么要它：规则是在攻击分支**之前**早退的 ⇒ 万一路径解析不出来（或解析器说
#: "你已经在目标集合里了"、可脚下集合又对不上 ✗），每拍重下就是刷屏 + 反复造任务 ✗。
#: 3 秒足够让它自己跑起来，也不至于卡住不动 ✓。
#: ⚠ 2026-09-28 起它是「**区域查询CD(s)**」的**兜底值** —— 真正的口径搬到了**每一个战斗
#:   区域项**里（`DecisionSettings.battle_zones[i]["cd_s"]` ✓，用户在「路线脚本 → 战斗区域」
#:   弹窗里逐项配 ✓）。运行期一律走 `agent._zone_cd_s(zone)`（某项）/ `_zone_cd_of_sets(names)`
#:   （"这批集合命中哪个区域项"）✓；**只有**"问不到任何区域项"时才用它兜底 ✓
#:   （老项目文件 / 老配置迁移 / 怪不落在任何区域项里 ✓）。
#: ⚠ 三处用途没变（只是"每处传哪个区域项"不同）：区域归位重下 / 追击下前往 / 区域筛缓存时效 ✓。
ZONE_GOTO_RETRY_S = 3.0

#: ⭐ 「**锁定的目标**」那份 foothold 集合的缓存时效（秒）—— 用户 2026-09-28 要求：
#: 原话"**锁定一个目标时每秒更新其位于的 foothold 集合**"✓。
#: 现场症状 = "**锁着目标来回走**"：`mob_sets_of` **每拍现算**（画面→世界 + 落在哪块面 ✓），
#: 而怪框每拍都在抖（几像素）、它那条 foothold 在**边界上**时就会在「一楼 / 二楼」之间来回跳
#: ⇒ 追击的目的地跟着跳 ⇒ 人走到一半目的地又被改写 ⇒ 来回走 ✗
#: （日志形状：同一只怪、同一个目的地，连着十几次「追击：怪不在我这块平台上（它在「一楼」），先过去」✓）。
#: ⇒ **一秒一次**：滤掉帧级抖动，同时"人真换了层"（爬绳 / 下跳要好几百毫秒 ✓）绝不会被漏掉 ✓。
#: ⚠ 别调成 0（= 老行为，每拍现算 ✗）。
#: ⭐ 2026-09-28 改口径：**主驱动不再是时间** —— 用户原话"**在恰当的时机重新查询**
#:   （之前我说 1s 但是可能不太好用）" ✓ ⇒ 现在是"**局面变了就重查**"（`agent._mob_sig`：
#:   怪在画面上的位置档 + 玩家脚下集合 ✓）。这个值退成**最长保鲜（兜底）**：
#:   怪一直不动、玩家也没换层时，仍然每隔这么久重问一次 —— 万一地形/集合文件被编辑过 ✓
#:   （不该抱着旧答案到永远 ✗）。
MOB_SETS_TTL_S = 3.0

#: 判"这只怪还是不是上次那个局面"时，位置按它**量化**（画面像素 ✓）——
#: 检测框每拍抖几像素，**逐像素比 = 每拍重查** ✗（那就回到老毛病了）。
#: 取 24：帧间抖动（几像素）远小于它 ✓；而"走一格 / 跳一下 / 换一层"都远大于它 ✓。
MOB_RESIG_PX = 24.0


#: 为什么只看**输出序列**的键：移动键（走 / 追怪）按住几十秒是正常的 ✗；而输出/技能键
#: 在"已经不在打"的情况下还按着，一定是漏放。给 1.5 秒宽限：正常那一拍 up 就回来了，
#: 而且状态切换本来就有几帧的缓冲。
OUT_KEY_STUCK_S = 1.5


class CombatAgent:
    def __init__(self, settings):
        self.settings = settings
        self.keys = KeyState()
        self.state = "idle"
        #: 当前挂着的**上绳/下跳任务**（`decision/route.py` 的 ClimbJob / DropJob）——
        #: None = 没有。由 `start_climb()` 挂上；跑完/失败/取消都会清掉（见 `_climb_tick`）。
        self._climb = None
        #: 「这一拍**有没有人推过任务**」—— `_climb_tick` 是**唯一**会推任务的地方，它置 True；
        #: 决策拍开头清 False，出口由 `_task_settle` 统一结算（2026-09-26 结构性收口）。
        #: 为什么不手工在每条早退里各调一次：tick 里的早退有五六条，漏一条就变成
        #: "任务被挡住的那段时间算成它自己的错"（实测：未定位玩家 / 关自动各一条 ✗）。
        self._task_ran = False
        #: 任务**最后一次真正跑过的那一拍**的时刻（`time.monotonic()` 秒；None = 还没跑过第一拍）。
        #: 报信时用它当**暂停起点**、不是"现在"：结算发生在**下一拍**（见 `_task_settle`），
        #: 拿"现在"当起点会把被挡住的**第一拍**漏掉、少扣一次（约 1 拍 ≈ 16~20ms）。
        self._task_ran_at = None
        #: 失败后**等哪一刻再重新激活**（`time.monotonic()` 秒；None = 没在等）。
        #: 由 `_climb_tick` 设/清，延迟长短读 `settings.climb_retry_delay_s`。
        self._climb_retry_at = None
        #: ⭐⭐ **「刚挨过打」的时刻**（用户 2026-09-28 ✓ 原话："在 `_climb_interrupted` 里打一个
        #:   『刚挨过打』的时间戳"）—— 只有 `_climb_interrupted`（= 挨打/休息打架导致**任务没跑**
        #:   的**唯一通知口** ✓）会写它 ✓。
        self._climb_hit_at = 0.0
        #: ⭐⭐ **这一轮"按过跳"的时刻**（用户 2026-09-28 ✓ 原话："失败那段改成：**按跳后**挨过打
        #:   ⇒ 用延迟"）—— 由 `_climb_tick` 看 `out["jump"]` 记 ✓（`route` 侧不用改 ✗）。
        #:   ⚠ 判据是"**挨打发生在按跳之后**"（`_climb_hit_at >= _climb_jump_at`）✓：
        #:     还没按跳就被打（比如对齐阶段）⇒ **不算**，该**立刻重来** ✗。
        self._climb_jump_at = 0.0
        #: 这个寻路任务是**哪一刻**下的命令（给"寻路超时时间"那道总闸用）。
        self._climb_started = 0.0
        #: 「这一步」的签名 `(执行器类名, 起点集合, 目标集合)` —— 判"重下的是不是同一步"，
        #: 同一步重下**不重置**上面那把钟（见 `start_climb` 里那段 ✓）。
        self._climb_sig = None
        #: 「**寻路超时后按键**」上一次真发出去时的 `(键名, 时刻)`（没发过 = `None` ✓）——
        #: 这是它在 **agent** 侧的可观察痕迹：① 界面 / 排查能一眼看到"刚才超时发了哪个键" ✓；
        #: ② 用例**只能读它** ✓ —— 别指望 patch `ag.tap`：`Harness._patched()` 会把输入层
        #:    换成替身、把外面 patch 的 `tap` 覆盖掉 ✗（2026-09-28 在这上面绕了一圈 ✓）。
        #: 见 `_fire_timeout_key`。
        self.last_timeout_key = None
        #: ⭐ **定频快照**（用户 2026-09-28 要求 ✓）：`pos`（位置 + 血量）/ `act`（动作）**每秒各一条**
        #:   —— 用来回答"人在哪 / 脚下是哪块 / 血多少 / 有没有在动 / 有没有在按方向键"。
        #:   ⚠ 为什么必须加：2026-09-28 那次"卡了 2 小时 22 分"（去「右下休息平台」走不到），
        #:     日志里**只有** `climb_dx`/`climb_y` 两条探针 ⇒ 人是"死了动不了"还是"系统没发键"
        #:     **分不出来** ✗，只能靠反推（`dx=393.8` ⇒ 人在离绳 400px 处）✓。
        #:   见 `_snap_beat`。
        self._snap_at = 0.0
        #: 任务/走位**这一拍给的方向**（-1/0/+1）—— 由 `_steer` 写、`_snap_beat` 读完**清零**
        #:（不清的话"没动的那几拍"会一直残留上一次的值 ✗）。
        #: ⚠ `_snap_beat` 挂在 tick **靠前**的位置（休息分支之前 ✓）⇒ 打到日志里的其实是
        #:   **上一拍**的动作，差一拍（≈16ms），1 秒粒度下无所谓 ✓。
        self._last_move = 0
        #: 「**锁定 / 正被追的那只怪**」的集合缓存：`(目标 id, 查的时刻, 解析器那份 info)` ✓
        #: —— **一秒一次**（`MOB_SETS_TTL_S` ✓ 用户 2026-09-28："锁定一个目标时每秒更新其位于的
        #: foothold 集合"）。见 `_locked_mob_info` ✓。`None` = 还没有 / 上次没查出来 ✓。
        self._mob_info_cache = None
        #: 「**追击为什么没下任务**」上一次记下的原因 —— 每条早退都会 `_note_chase_skip`
        #: 一次，但**只在原因变化时**才真打点（早退是每拍都会命中的 ⇒ 全记会把 log 淹掉 ✗）。
        #: 见 `_note_chase_skip`（用户 2026-09-28："**依旧没有走『取向量 → … → 下达寻路任务』**"✓
        #: —— 这类"静默早退"已经让同一个 bug 查了三轮 ✓）。
        self._last_chase_skip = None
        #: ⭐⭐ **这条路线「因为什么」下的**（用户 2026-09-28 的「**来源签名**」✓）——
        #: `why` 是一句**人话**（给日志 / 界面看 ✓），这个是给**逻辑**看的结构化那份 ✓。
        #:   `{"kind": "chase", "mob_id": 123}`（见 `plan_and_start_route` 的 `origin` ✓）
        #: ⚠ 用途只有一个：在**两个节点**复核"有没有更便宜的怪"（`_climb_recheck` ✓）。
        #:   **只有 `kind == "chase"` 参与**（用户 2026-09-28 定：**只掐「追击」类** ✓）——
        #:   其余来源（命令前往 / 任务队列 / 定点休息 / 回战斗区域 / 到点去哪）一律**豁免** ✓：
        #:   那些是**人手安排 / 策略定的**，被自动掐掉会让人以为按钮没生效 ✗。
        #:   ⚠ 不传 `origin` 的调用点**天然豁免** ✓（不用在复核里逐个列白名单 ✓）。
        self._route_origin = None
        #: 「**这一步**」的来源签名（从 `_route_origin` 抄一份 ✓）—— 随 `_climb` 一起生灭 ✓
        #: （`start_climb` 写入 / `stop_climb` 与 `_task_finished` 清 ✓）。
        self._climb_origin = None
        #: ⭐ 两个复核节点**各自"这一步查过了没有"**（用户 2026-09-28 要求 ✓）：
        #:   ① `_armed_done` = "对齐完成、起跳前"（靠 `job._armed_at` 认那一刻 ✓）；
        #:   ② `_half_done` = "寻路超时计时过半"。
        #: ⚠ **每步只查一次**（`start_climb` 里复位 ✓）—— 否则每拍都算一轮候选怪的代价，
        #:   白烧 CPU，而且"严格更便宜就掐"（用户选的口径 ✓）在代价接近时会来回掐 ✗。
        self._recheck_armed_done = False
        self._recheck_half_done = False
        #: 「**最大战斗时长**」这一轮"连着打"的计时起点（`None` = 现在没在累计 ✓）。
        #: 口径见 `battle_zones` 的 `fight_max_s` ✓；每拍由 `_fight_beat` 维护 ✓。
        #: ⚠⚠ **2026-09-28 口径改过**（用户报"**当前任务一直是战斗，但是时间走几秒就没了**"✗）
        #:   —— 见 `_fight_beat` 顶上的那段说明，这里只留结论：
        #:   **"在战斗" = 没有寻路任务**（= 界面「当前任务」显示「**战斗**」那行 ✓ **同一个源** ✓），
        #:   而**不是** `self.state`（那是另一套判据，会在攻击/换目标之间**闪** ⇒ 一闪就把
        #:   时间清掉 ⇒ 用户看到"走几秒就没了" ✗）。
        self._fight_since = None
        #: ⭐ **实际"在打"的累计秒数**（用户 2026-09-28 定的口径 ✓："**只要一直战斗就不应该
        #:   有任何理由停时间**"）。⚠ 到点判据用它、**不用** `now - _fight_since` ✗：
        #:   后者一旦"暂停"就废（寻路打断一次就得重头 ✗）。`_fight_since` 保留只为
        #:   界面/用例当"这一轮的起点"看 ✓。
        self._fight_acc = 0.0
        #: 上一拍的 `now`（用来算"这一拍打了多久" ⇒ 累进 `_fight_acc` ✓；`None` = 刚起算 ✓）
        self._fight_last = None
        #: ⭐ **正在计时的那个区域项**（与 `_fight_since` 配对：计时的是**哪一项** ✓）——
        #: 界面上「编辑战斗区域」拿它显示"这一项还剩多久"（用户 2026-09-28 要求 ✓：
        #: "编辑战斗区域里，要用**灰字**显示**最大战斗时长的倒计时**"✓）。
        #: ⚠ 与 `_fight_since` **同设同清**（口径一处 ✓ 见 `_fight_beat`）—— 只记不判 ✓。
        self._fight_zone = None
        #: ⭐ **给界面看的"战斗时长倒计时"三件套**（用户 2026-09-28：编辑战斗区域里要用**灰字**
        #: 显示倒计时 ✓）—— 由 `_publish_fight_clock` 在**开始计时 / 清零**两个边沿写 ✓，
        #: 界面**只读**、自己算剩余 ✓。⚠ **运行时状态、不进 `to_dict`** ✓（同 `enabled` /
        #: 休息那几项 ✓ ⇒ 不落盘、不跨会话复活 ✓）。这里显式初始化只为**声明清楚**
        #: （别靠首次赋值凭空造属性 ✗ —— 这份清单本身也是给别人看的文档 ✓）。
        self.fight_zone_name = None
        self.fight_elapsed_s = 0.0
        self.fight_cap_s = 0.0
        #: ⭐ **循环行为的运行态**（用户 2026-09-28："休息过程中希望能看到**每一个循环项目**的
        #:   当前状态信息"✓）—— 同 `fight_zone_name` 那套：**运行时状态、不进 `to_dict`** ✓、
        #:   界面**只读** ✓（剩余时间由界面自己按下面那个绝对时刻算 ✓ 免得每拍写一遍字 ✓）。
        #:   · `spot_loop_total` = 列表里**有效项**的个数（空行为的不算 ✓）；
        #:   · `spot_loop_idx`   = 正在演第几项（**1 基** ✓ `0` = 没在演/没在跑 ✓）；
        #:   · `spot_loop_name`  = 正在演那一项的**名字**（界面上直接显示 ✓）；
        #:   · `spot_loop_next_at` = 下一轮**起跑**的绝对时刻（`time.monotonic()` ✓ 与界面同钟 ✓；
        #:     `0` = 没在等 ⇒ 界面别显示倒计时 ✓）。
        self.spot_loop_total = 0
        self.spot_loop_idx = 0
        self.spot_loop_name = ""
        self.spot_loop_next_at = 0.0
        #: ⭐ **注入位**：`foothold id -> 那条线的世界中心 x`（`fid` 传字符串 ✓，取不到回 `None` ✓）
        #: —— 给「**idle 回归**」算"该往哪边走"用（`battle_zones` 的 `idle_foothold` ✓）。
        #: ⚠ agent **不持有地形** ⇒ 和 `mob_sets_of` / `route_plan` / `set_span_of` 一样
        #:   **由 `gui/live_thread.py` 装进来** ✓（见那边的 `_make_foothold_x_resolver` ✓）；
        #:   没装（老环境 / 用例没给）⇒ 什么都不做（照旧站住 ✓）。
        self.foothold_x = None
        #: **多步路径**：还没跑的任务队列 + 进度（见 `start_route`）。
        self._route = []
        self._route_why = ""
        self._route_step = 0
        self._route_total = 0
        #: 「**任务队列**」（用户 2026-09-27：「命令」组里「命令前往」右边那个
        #: 「添加任务队列」按钮）—— 一串**目的地集合名**：当前这条路线**整条走完**之后
        #: 自动接着下一条 ✓（`_task_finished` → `_goto_queue_next`）。
        #: 界面上排在小地图下面「当前任务」的下方，**一行一个**（`route_panel._osd_lines`
        #: 读 `goto_queue()` ✓）。
        #: ⚠ **运行时状态，不进 `to_dict`**（和 `enabled` / 休息那几项一样 ✓）：
        #:   它是"这一次实时里排的活"，不该存进项目文件、也不该跨会话复活 ✓。
        #: ⚠ 队列里存的是**命令**（目的地），不是路线 —— 每一条**出发时才**按"我现在站哪"
        #:   重新解析（和 `plan_and_start_route` 同一个道理：起点每一刻都可能变 ✓）。
        self._goto_queue = []
        #: 上一次看到的「**位置状态**」—— 用户 2026-09-27："当玩家的当前位置状态（位于
        #:   foothold 集或绳梯等）改变时，要重新评判一次最优路径"。
        #: **定义（口径只此一处）**：`(here_sets, ladder_id)` = 玩家在**地形语义**上"站在哪"
        #:   的**粗粒度**快照：① 脚下那个 foothold 所属的**全部命名集合**（0 个 = 空集，
        #:   空集也是一种状态 ✓）；② 现在贴在哪根绳上（不在绳上 = ""）。
        #: **粒度边界**（用户 2026-09-27 澄清，最容易想歪的一条）：
        #:   · **算**变化：换到另一个集合 / 掉出所有集合 / 上或下了某根绳 ✓；
        #:   · **不算**变化：在**同一个集合里**走来走去、**被怪撞来撞去**、x/y 变、朝向变 ——
        #:     路线的**节点**没变、计划仍成立 ⇒ 不该重算 ✓（每帧都变的量当"状态" = 每拍重算 ✗）。
        #: ⚠ **别和「任务相位」混**（`job.phase`，执行器自己写、一步里变几次 ✓）—— 相位只当
        #:   "动作进行中先别重算"的**守卫**用（`route.INFLIGHT_PHASES` ✓）。
        #: ⚠ 每挂上一步就清成 `None`（`start_climb` ✓）⇒ 比的是"**这一步执行期间**的变化" ✓
        #:   （正常到站那一下不算变化 —— 那是收工那套的工作，拿它重算会来回抖 ✗）。
        self._loc_state = None
        #: 「**添加任务队列**」刚排进来、还没起跑的那面旗子（用户 2026-09-27：
        #:   "我期望的是：战斗是最低优先级的任务，寻路任务队列要依次执行，现在我排队列
        #:   都没反应"）。为什么要旗子、而不是在 `queue_goto` 里直接起跑：
        #:   · `queue_goto` 是**界面线程**调的（`gui/route_panel._on_goto_queue` ✓），而
        #:     起跑要 `plan_and_start_route` —— 读地形 / 写 `behavior` / 动 `_climb`，
        #:     都该在**决策线程**里做 ✓（同 `stop_route` "GUI 只拍意图"那条规矩 ✓）；
        #:   · 下一拍 tick 看到旗子就去起跑（≈20ms，用户感觉上就是"立刻"✓）。
        #: 起跑条件（tick 里判）：**没东西在跑**（`_climb is None and not _route`）；
        #: 有任务在跑 ⇒ 老实排队（收工时由 `_task_finished` → `_goto_queue_next` 接下一棒 ✓）。
        self._queue_kick = False
        #: 由**实时线程**注入的**路径解析器**：`dst_set → {"path","jobs","why","here"}`
        #:（见 `gui/live_thread._make_route_resolver` 与 `decision.route.plan_jobs`）。
        #: 为什么是注入而不是自己算：地形与集合数据在实时线程手里
        #:（`_fill_route_ctx` 同一处），agent 不持有它们 —— 不发明第二套数据源。
        #: ⚠ 它是**函数**，不是预存的表：路径的起点是"我现在站哪个集合"，每一刻都可能变
        #:（`docs/开发计划.md` 原先写的是 `set_name → [job,…]` —— 那样只能在某一刻算一次 ✗）。
        self.route_plan = None
        #: 「**这只怪要走多远**」的解析器（`(player, mob) → {"cost", "sets", "why"}`）——
        #: 由实时线程注入（`gui/live_thread._make_mob_cost_resolver` ✓，和 `mob_sets_of`
        #: 共用同一次"画面→世界"换算 ✓）。`_nearest` 用它把"挑哪只怪"从**画面绝对距离**
        #: 换成**寻路距离**（用户 2026-09-27 要求）；没注入 ⇒ 自动退回老口径 ✓。
        self.mob_cost_of = None
        #: 「**某个集合在世界上占的 x 范围**」的解析器（`name → (左, 右) | None`）——
        #: 由实时线程注入（`gui/live_thread._make_set_span_resolver` ✓，底层是
        #: `route.set_span` ✓）。用户 2026-09-27 要求 2 用它判"怪框底边和我站的这块平台的
        #: x 范围有没有交集"（有 ⇒ 够得着 ⇒ 不必下寻路任务 ✓）；没注入 ⇒ 那条豁免不生效 ✓。
        #: ⚠ 这里原来挂"集合名 → x 范围"的解析器（`set_span_of`）给 `_reachable_without_path`
        #: 自己算平台宽度 —— 2026-09-27 用户要求**收编**（几何只在 `perception/pos_state.py`
        #: 算一次 ✓）⇒ 那个属性**已删** ✗；同样那份查询现在由 `live_thread` 喂给
        #: **位置状态机**（`span_of=` ✓），执行器读广播的 `here_span` ✓。
        #: 「定点休息」当前那个"要走出去"的阶段**出发了没有**（见 `_run_rest_spot`）。
        #: 为什么要有它：路径一结束时 `_climb` 会被清空 —— 可"还没出发"和"走完了"都长这样，
        #: 必须分得清（不然刚进休息那一拍就会被当成"走完了"或"没走成" ✗）。
        self._spot_started = False
        #: **临时战斗**（用户 2026-09-27 要求）：**休息收工/被打断之后**、以及**寻路失败
        #: 之后**那一段里，"有怪就先打" —— **不受「限制战斗区域」的约束**。
        #: 为什么：那两种情形本身就意味着"出事了"（被打断 = 有东西正在打我们；寻路失败 =
        #: 半路停下），这时一路不还手地往回走 = 白挨打 ✗。
        #: 点亮：`_finish_rest`（休息收工/被打断）与 `_task_finished(failed=True)`（寻路失败）；
        #: 熄灭：**这一波打完了**（视野里再没有可打的：`in_range` 与 `candidates` 都空）——
        #: 见 `tick` 里那一支。⚠ **不设时长**：结束条件就是"没得打了"本身 ✓（别加常数 ✗）。
        self._temp_fight = False
        #: 输出序列的键"在非输出状态下还按着"是从哪一刻开始的（卡键看门狗用）。
        self._out_held_since = None

        #: 任务里最后那句 note + 它的时刻（任务结束之后还要能答"为什么结束"，
        #: 见 `current_goto_note` 与 `GOTO_NOTE_KEEP_S`）。
        self._last_goto_note = ("", 0.0)
        self.facing = 1             # 朝向：+1 右（默认）/ -1 左，由最后按的方向键决定
        self._patrol_dir = 1        # 扫平台倾向朝向（巡逻主方向）：打背后怪不改变它
        self._last_facing_change = time.monotonic()  # 朝向最后一次变化的时刻（超时监控用）
        self._turn_at = 0.0         # 最近一次换向的时刻（最小按住时间 / 转向后输出延迟用）
        self._player_lost_since = None  # 找不到玩家的起始时刻（monotonic），定位到就重置
        #: 「画面里连续看不到角色」的起始时刻（用户 2026-09-27 要求）。
        #: ⚠ **与"自动开着没"无关** —— 它只做**留痕**（perf 注记 + 计数），
        #: 见 `_watch_player_visible`。None = 现在看得到。
        self._player_gone_since = None
        self._player_gone_counted = False
        self._was_enabled = False   # 上一次 enabled 状态（检测开启自动的上升沿）
        self._deadzone = 6.0        # |dx| 小于它就停，避免左右抖
        self._last_output = 0.0     # 最近一次**输出行为**的开始时刻（monotonic），
                                    # 输出CD 与「输出后锁定防抖」都以它为锚点
        self._attack_duration = 0.03  # 每次点按攻击键的按住时长（秒）
        self._target_id = None      # 当前锁定的目标 id
        self._target_until = 0.0    # 锁定到期时间（monotonic）
        self._next_hp_pot = 0.0     # 下次补血的时刻
        self._next_mp_pot = 0.0     # 下次补蓝的时刻
        # ⚠ 这里原来还有 `_next_feed` / `_was_feed_enabled`（自动喂宠的排期）——
        #    功能整块移除（用户 2026-09-26）⇒ 一并删掉 ✓（要喂宠请用「自定义定时行为」✓）。
        self._timer_states = {}      # 自定义定时行为执行中序列状态：{name: [phase, next_ts, held]}
        self._kill_mobs = set()     # 要立即消除的防抖幽灵框 id（攻击幽灵框时记录，避免空放技能）
        self._next_evade = 0.0      # 下次规避动作（跳）的时刻
        self._next_chase_jump = 0.0 # 下次追击起跳的时刻（全局节流，只在沿抖动时兜底）
        #: `chase` 状态**连续**保持的起始时刻（None = 现在不在 chase）。
        #: 给「追击起跳需要的冲刺时间」用：每拍按**上一拍**的状态续/清（见 tick 里那段）。
        self._chase_since = None
        self._band_had = False      # 上一拍起跳范围内有没有怪（用来判「从无到有」的沿）
        self._next_resetall = 0.0   # 下次定时 RELEASEALL 的时刻
        self._pending_attack = None # 待发的输出键时刻（跳规避：跳键后 interval 发输出）
        self._no_target_since = None  # 朝向没目标的起始时刻（换朝向防抖用）
        self._back_ctx = None       # 回身输出序列上下文 [seq, phase, next, held, sub_stack]
        self._output_ctx = None     # 输出行为序列上下文
        self._stand_attack = False  # 站桩输出：攻击范围内有目标 → 本帧不给方向键。
                                    # **两个策略通用**（原来只有扫平台用它）——
                                    # 用来看"这一拍该不该按方向键"（见 _attack_state
                                    # 与 TURN_TAP_S 那一段）。
        #: **站桩输出的第几轮**（用户 2026-09-28 要求 2：站桩时**每 3 轮**补一个"朝目标"的
        #: 方向键、按满「最小切换朝向时间」；**退出站桩就清零** ✓）。
        #: 计数点在"新开一轮输出序列"那一处（`_output_actions` 的 attack 支 ✓）——
        #: 一轮 = 一次 attack ✓；非 attack 状态（`_output_actions` 开头）统一清零 ✓。
        self._atk_round = 0
        #: 这一拍**广播说人在"本任务那根绳"上**（`PosSnapshot.on_rope` ✓ 带按键许可）——
        #: 两个地方要用：① 绳梯豁免的收窄（见 `tick` 里 `_on_rope` ✓）；
        #: ② **绳上绝不按左右**（按了在游戏里就是松手 ⇒ 会掉 ✗ 用户 2026-09-28 定 ✓）。
        self._in_rope_now = False
        #: 「该补朝向键的那一轮」的**开始时刻**（`time.monotonic()` 秒；None = 现在不欠）。
        #: ⚠ 为什么不能复用 `_hold_turn` 的时间窗（`_turn_at`）：站桩时**每一轮** `_attack_state`
        #:   都会 `set_facing`（朝目标 ✓），而 `set_facing` 会把 `_turn_at` 刷新 ⇒ 那个窗
        #:   **永远重新开始** ⇒ 方向键会**一直按着不放** ✗（实测：4 秒的用例里方向键按了
        #:   3.94 秒 ✗）。⇒ 这里记一个**属于那一轮自己的**起点，窗口一到就停 ✓。
        self._turn_due_at = None
        #: 上一次「每 3 轮补朝向键」的**结束时刻**（`None` = 还没补过）—— 用来**防止窗口
        #: 首尾相接**（否则方向键会一直按着 ✗ 见 `_run_output_ctx` 里那段 ✓）。
        self._turn_until = None
        self._link_checked = 0.0    # 上次指令通道体检时刻
        self._link_retry = 0.0      # 上次尝试重连通道的时刻（避免狂重连）
        self._next_afk = 0.0        # 下次防掉线触发时刻
        self._was_afk_enabled = False  # 防掉线开关上一次状态（上升沿检测）
        self._afk_ctx = None        # 当前防掉线阶段的序列上下文（进入/退出隐身）
        self._rest_pending = False  # 已到防掉线触发时间，等攻击范围内的怪清空
        self._rest_last_tick = 0.0  # 上次推进「暂停计时器」的时刻
        self._rest_until = 0.0      # 本次休息的结束时刻
        self._rest_retry_after_interrupt = False  # 这次休息被补血打断 → 下次按 retry_sec 排
        #: ⭐ 「休息过程中循环行为」的运行态（用户 2026-09-28 ✓）：
        #:   `_loop_ctx` = 正在演的序列上下文（格式同 `_afk_ctx` ✓；`None` = 没在演 ✓）；
        #:   `_loop_next_ts` = 下一次**起跑**的时刻（`<=0` = 还没排 ✓）。
        #: ⚠ **独立字段**（不能借用 `_afk_ctx` ✗）：那个在 `afk_spot_rest` 阶段被好几处
        #:   无条件置 `None`（见 `_run_rest_spot` 各出口 ✓）⇒ 借用会被顺手清掉 ✗。
        self._loop_ctx = None
        self._loop_next_ts = 0.0
        #: 这一轮**跑到列表第几项**了（0 基 ✓）—— "**每个循环里按项目依次执行**"就靠它 ✓
        #:   （用户 2026-09-28：行为列表升级成"列表配置"之后 ✓）。
        self._loop_idx = 0
        #: ⭐ 「演完手上这一段就收」（用户 2026-09-28 更正）：结束休息**不打断**正在演的那一项
        #:   （不松键），只是**不再开始新的**，它一演完就收摊。
        #:   ⚠ 与"立刻杀"分工不同，别混：结束休息 ⇒ 本标记；停自动/关防掉线/切项目
        #:     （`_release_combat_keys`）⇒ 立刻 `_stop_spot_loop`（安全网）。
        self._loop_drain = False
        #: ⭐ 「**休息结束推迟到循环执行完**」的运行态（用户 2026-09-28 ✓）：置上 = "休息时间
        #:   到了，但已经说好**等这一轮演完**"（见 `_hold_rest_for_loop` ✓）。
        #:   ⚠ 它的作用是**拦住新一轮起跑** —— 少了它，循环间隔比决策拍还短时会在
        #:     "演完 ⇒ 又起跑"之间**无限推迟**休息结束 ✗（口径见 `_spot_loop_beat` 开头 ✓）。
        #:   由 `_stop_spot_loop` 清（收摊 ⇒ 这笔账也结清 ✓）。
        self._loop_hold = False
        #: ⭐ "**手动结束休息**已经受理、正在等手上那一段循环演完"（用户 2026-09-28 ✓）：
        #:   置上 = **留在休息状态里收尾**（口径见 `tick` 里 `rest_abort` 那段 ✓ ——
        #:   为什么不能"立刻收工、让它在后台演完"那儿写了 ✗ 的教训 ✓）。
        #:   由 `_finish_rest` 清 ✓。
        self._rest_stop_pending = False

    @staticmethod
    def _center_dist(m, player):
        """**角色中心点**到怪框最近边缘的水平距离（像素）；0 = 怪框已盖住角色中心。

        起算点用角色中心，不用玩家框的朝向边缘：够不够得着是看角色，而玩家框是
        贴图框（宽度随动作变），拿它的边缘起算，同一只怪的距离会跟着框宽漂。
        终点仍取怪框最近的边 —— 大框目标（龙）不必再靠近半个框宽才算够得着。
        只算水平：横版先只做水平方向找怪打（见模块开头）。

        这个距离是**共用口径**：攻击判定 / 最近怪 / 最小距离规避 / 追击起跳 /
        背后锁定 全走它，实时预览画的攻击范围框也按它起算（gui/live_thread.py）。

        ⚠ 它只管**水平**那一维（用户 2026-09-27 之后，"够不够得着"还要看竖直，
          口径在 `_in_box` / `_in_attack_box` 那**一处** ✓ —— 别在这里塞 y ✗：
          `_nearest` / 追击 这些"找最近"的用途仍然只该按水平比 ✓）。
        """
        m_left = m.x - m.w / 2.0
        m_right = m.x + m.w / 2.0
        if m_right < player.x:
            return player.x - m_right   # 怪完全在中心左侧
        if m_left > player.x:
            return m_left - player.x    # 怪完全在中心右侧
        return 0.0                      # 怪框跨过中心 → 贴身

    @staticmethod
    def _vdist(m, player):
        """怪框在**竖直方向**离角色中心的最近距离（画面像素；怪框跨过角色那条水平线时为 0）。

        口径和 `_center_dist` 一致：起算点 = **角色中心**、终点 = 怪框最近的边 ✓
        （大框怪不必再靠近半个框高才算够得着 ✓）。
        """
        top = m.y - m.h / 2.0
        bot = m.y + m.h / 2.0
        if bot < player.y:
            return player.y - bot          # 怪整个在角色**上方**
        if top > player.y:
            return top - player.y          # 怪整个在角色**下方**
        return 0.0                         # 怪框跨过角色那条水平线 ⇒ 竖直方向没有距离

    def _box_empty(self, lo, hi):
        """这个框是不是**空集**（面积 0）⇒ 它的判定**整条跳过**（用户 2026-09-27 定的 ✓）。

        空集两种情形（和 `attack_box_rect` 返回 None 的规则**同一套** ✓）：
          · 水平跨度 ≤ 0（"可攻击区" `[min_d, max_d]` 是空 —— 比如最大攻击距离配 0、
            或者最小攻击距离配得跟它一样大 ⇒ 全是盲区 ✓）；
          · 竖直跨度 = 0：向上、向下**都恰好配 0** ✓（`< 0` 是"不限"，**不算空** ✓）。
        ⚠ 空集时**判定跳过 = 一律不成立**（「攻击框空 ⇒ 打不到任何怪」、
          「盲区框空 ⇒ 不算贴脸/不规避」✓）—— 这也正是"0 按 0 算"的直接后果 ✓。
        """
        if float(hi) - float(lo) <= 0:
            return True
        up = attack_edge(getattr(self.settings, "attack_up_dist", -1))
        down = attack_edge(getattr(self.settings, "attack_down_dist", -1))
        return up is not None and down is not None and (up + down) <= 0

    def _in_box(self, m, player, max_d, min_d=0.0, facing=None):
        """怪框在**「最大/最小距离 + 上下距离」围成的矩形**里吗 —— 判定口径**只有这一处** ✓。

        ⭐ **`facing`（单侧判定，用户 2026-09-28 ✓）**：原话"攻击范围判定改成『**单侧**』
          （跟画面一致）"✓ —— 不传 ⇒ 用 `self.facing`（朝向那侧 ✓，与绘制 `attack_box_rect`
          的 `dirn` **同一口径** ✓）；传 `0` ⇒ **左右都算**（老行为 ✓）；传 `±1` ⇒ 显式要那一侧 ✓。
        ⚠ 原来这里**不看朝向**（只比 `_center_dist` ✗），而绘制是**单侧** ✗ ⇒ 两边不一致 ✗
          （用户 2026-09-28 当场发现："判定和画面不是一个口径"✓）。

        用户 2026-09-27 定的矩形：`attack_dist / min_attack_dist / attack_up_dist /
        attack_down_dist` 四条边（界面上是「战斗参数 → 攻击」那个子组 ✓）。

        · 水平：`min_d ≤ 水平距离 ≤ max_d`（`min_d > 0` 时它以内是**盲区** ⇒ 不算在里面）；
        · 竖直：怪框离角色中心的**竖直最近距离** ≤ 该方向的「向上/向下攻击距离」——
          怪在**上方**看「向上攻击距离」、在**下方**看「向下攻击距离」（`attack_edge`：
          **负数 = 不限**、**0 = 就是 0** ✓）；
        · **框是空集**（面积 0）⇒ 直接不成立（见 `_box_empty` ✓）。

        ⚠ "不限"（负数）走的是 `lim is None` 那条：**直接算在里面** ✓（= 老行为不看 y ✓）；
          而 `lim = 0` 只放过"怪框跨过角色那条水平线"（`vd = 0` ✓）—— 这两件事
          **不是一回事**，别写混 ✗（用户 2026-09-27 纠正的就是这里）。
        ⚠ 读数缺失（`m.y/m.h` 或 `player.y` 还没有）⇒ 竖直那一维**按 0 距离算**（= 不构成
          排除理由 ✓）：观测问题不该被当成"不在范围"✗。
        ⚠ 边界（`d == min_d`）算**盲区**（保守：刚好压在那条线上时宁可当贴脸 ✓）。
        """
        if self._box_empty(min_d, max_d):
            return False
        # ⭐⭐ **单侧**（用户 2026-09-28 ✓ 原话："攻击范围判定改成『单侧』（跟画面一致）"）：
        #   怪必须在**朝向那一侧** ✓（`dx × dirn ≥ 0`）—— `facing=None` ⇒ 用 `self.facing`
        #   ✓（与绘制 `attack_box_rect` 的 `dirn` **同一口径** ✓）；`0` ⇒ 两边都算（老行为 ✓）。
        #   ⚠ 边界 `dx == 0`（正对上）⇒ 算**在范围内** ✓（合理 ✓）。
        #   ⚠ 这一道和"距离"是**两把尺**：先按方向砍掉一半，再比距离/竖直 ✓。
        if facing is None:
            facing = self.facing
        _f = int(facing or 0)
        if _f != 0:
            _dx = (float(getattr(m, "x", 0.0) or 0.0)
                   - float(getattr(player, "x", 0.0) or 0.0))
            if _dx * (1 if _f > 0 else -1) < 0:
                return False              # 在**背后** ⇒ 不算在"这一侧的攻击范围"里 ✓
        d = self._center_dist(m, player)
        if d > float(max_d):
            return False
        min_d = max(0.0, float(min_d or 0.0))
        if min_d > 0 and d < min_d:
            return False
        vd = self._vdist(m, player)
        if float(getattr(m, "y", 0.0) or 0.0) < float(getattr(player, "y", 0.0) or 0.0):
            lim = attack_edge(getattr(self.settings, "attack_up_dist", -1))
        else:
            lim = attack_edge(getattr(self.settings, "attack_down_dist", -1))
        return True if lim is None else vd <= lim

    def _in_attack_box(self, m, player, facing=None):
        """怪框在**攻击范围框**里吗（够得着）—— 「攻击判定」的唯一入口 ✓。

        ⭐ `facing`（用户 2026-09-28 ✓）：`None` ⇒ **按 `self.facing` 单侧判** ✓（跟画面一致 ✓，
        也是"正面有怪才打、背后的怪转身"那条口径的基础 ✓）；传 `0` ⇒ **不分前后** ✓
        （`_in_range_all` 要的就是它 ✓ —— 别把它也筛成单侧 ✗）。

        `_in_range` / `_in_range_all` / 「站桩优先」那些判据**全走它**：老代码各自写一遍
        `_center_dist <= attack_dist`，加一维竖直之后必然漏掉某一处 ✗（用户 2026-09-27
        要求"改配置 + 逻辑"时的第一件事就是把口径收到一处 ✓）。
        """
        s = self.settings
        return self._in_box(m, player, float(s.attack_dist),
                            float(s.min_attack_dist), facing)

    def _in_blind_box(self, m, player):
        """怪框在**盲区框**里吗（0 ~ 最小攻击距离，竖直径向同宽）—— 贴脸要规避的那一块 ✓。

        「最小攻击距离」= 0（默认）⇒ 盲区是**空集** ⇒ 永远 False（= 老行为"不规避" ✓）。
        """
        s = self.settings
        return self._in_box(m, player, float(s.min_attack_dist), 0.0)

    def _nearest(self, mobs, player, ws=None):
        """挑要打的那只怪，返回 `(target, 画面距离)`；没怪返回 `(None, None)`。

        用户 2026-09-27 的要求（原话）："优化：锁定目标优先级要按「**寻路距离**」最近，
        而不是旧的应该是**绝对距离**" ⇒ 排序键**先比寻路距离、再比画面绝对距离**：

          · **寻路距离**（主键）= 从我站的集合走到怪站的集合要多少距离（`route.path_cost` ✓，
            世界像素；口径 = 最少段数 + 段内按 `hop_cost` 取最近 ✓，同择路那一套 ✓）；
          · **画面绝对距离**（副键）= 老的 `_center_dist`（角色中心 → 怪框边缘的水平像素 ✓）。

        为什么：`_center_dist` 是**画面上的直线** ✗ —— 隔着平台 / 要绕绳的怪，直线可能很近、
        实际要走很远 ⇒ 会"盯着够不着的那只打" ✗。

        ⚠ **只在"能问"的时候才问寻路距离**，否则**整条退回老口径**（绝对距离 ✓）：
          · 没注入 `mob_cost_of`（没开自动 / 老环境）；
          · 这一拍拿不到 `ws`；
          · 脚下的 `here_sets` 是空的（不知道我在哪个集合 ⇒ 没法算路 ✓）。
          算出来的那些里，**算得出代价的排前面**；代价是 `None` 的按画面距离排 ✓（不猜 ✓）。
        ⚠ **绝不改 `_center_dist`**：它同时是**攻击判定**的水平距离来源（`_in_box` ✓）——
          改它 = 顺手把攻击范围也改了 ✗（用户只要改"挑哪只" ✓）。
        ⚠ 返回的那个 `dist` **仍是画面距离**（不是寻路距离！）：下游「追击起跳」的区间判定、
          打点、`behavior.sample("chase_dist")` 用的都是它 ⇒ 换掉会连带改掉那些行为 ✗。
        ⚠ 开销：寻路距离只在**要挑目标**那一刻算（`_locked_target` 里锁定过期才调这一处 ✓，
          `target_cd` 是 500~1000ms ✓）⇒ 不是每拍、也不是每只怪都算 ✓
          （`route.path_cost` 一次 = 一次 BFS + 每段一次 `pick_edge` ✓，见它的说明）。
        """
        dists = [(self._center_dist(m, player), m) for m in mobs]
        if not dists:
            return None, None
        dists.sort(key=lambda t: t[0])                 # 老口径（也是副键 ✓）
        cost_fn = getattr(self, "mob_cost_of", None)
        here = getattr(player, "here_sets", None)
        # ⛔ 「**禁用杀怪寻路**」开着 ⇒ **连"挑目标"这一步也不查**（用户 2026-09-28 ✓ 原话：
        #   "**不再查询怪物框所属的 foothold 集合**"✓）。⚠ 这里必须显式停：`mob_cost_of`
        #   内部会去问"怪在哪块平台"（它**闭包捕获**了 `mob_sets_of` ⇒ 光把
        #   `agent.mob_sets_of` 置 `None` 是**停不掉**这条查询的 ✗ —— 一次性挑目标要对**每只**
        #   候选各算一轮 BFS，是这页查询里最大的一笔开销 ✓）。
        #   ⇒ 当成"拿不到寻路距离" ⇒ 走下面那条**老口径**：按**画面绝对距离**挑 ✓
        #     （也正合"单纯走向锁定怪物"那条要求 ✓）。
        if bool(getattr(self.settings, "disable_chase_pathfinding", False)):
            cost_fn = None
        if cost_fn is None or ws is None or not here:
            d0, m0 = dists[0]
            return m0, d0                              # 拿不到寻路距离 ⇒ 老口径 ✓
        scored = []
        sets_by_id = {}                                # 顺手记下每只怪的集合 → 给界面那行用 ✓
        for d, m in dists:
            try:
                info = cost_fn(ws.player, m)
            except Exception:                          # noqa: BLE001 —— 解析器坏了别停摆 ✗
                info = None
            if info:
                # 这份集合是**已经算出来**的（`mob_cost_of` 内部就调了 `mob_sets_of` ✓）
                # ⇒ 留下来给「锁定框下方那行集合名」用 ✓，绘制层就不用再扫一遍 foothold ✓。
                sets_by_id[m.id] = [str(s) for s in (info.get("sets") or []) if str(s)]
            c = None if not info else info.get("cost")
            # 排序键：(算不出代价的排后面, 代价, 画面距离) ✓
            scored.append((0 if c is not None else 1,
                           float(c) if c is not None else 0.0, d, m))
            behavior.event("lock_scored")      # 战斗/锁定：给一只候选怪算了代价（行为打点 ✓）
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        _cost, d0, m0 = scored[0][1], scored[0][2], scored[0][3]
        # ⚠ 这里原来把挑中那只怪的集合**顺手缓存**起来（`_target_sets`，给"锁定框下方那行
        #   集合名"用 ✗）—— 用户 2026-09-27 明确"锁定框表地点就不要了" ⇒ 缓存**已删** ✗。
        return m0, d0

    def _in_chase_jump_range(self, dist):
        """追击起跳判定：距离 dist（角色中心 → 怪框边缘）是否落在起跳区间内。

        区间 = [最大攻击距离 + min, 最大攻击距离 + max]：
          min/max 都是相对「最大攻击距离」的偏移；
          都为正 → 纯外侧；都为负 → 纯内侧；一负一正 → 跨攻击距离两侧。
        """
        s = self.settings
        lo, hi = float(s.chase_jump_min), float(s.chase_jump_max)
        if lo > hi:
            lo, hi = hi, lo   # 容错：填反了也能用
        ad = float(s.attack_dist)
        return ad + lo <= dist <= ad + hi

    def _maybe_chase_jump(self, dist, now, edge=False):
        """追击起跳：**只在「起跳范围内从无怪变成有怪」那一拍**才按跳。

        两个条件，缺一不可：
          · `edge`（由 tick 每拍算好，见那边的说明）—— 起跳范围刚由「无怪」变成
            「有怪」的那一拍。所以同一只怪一直挂在区间里**不会反复跳**：它只在刚
            进来那一下有跳的资格；
          · **攻击范围内没有别的怪** —— 有得打就先打，跳会打断输出、还可能把自己
            跳出攻击距离。目标**自己**在范围内不算「别的怪」：区间落在攻击距离
            内侧时（-15~0 这种）本来就该如此，那正是「已进范围但偏远，跳一下够着」。
            这一条由三个调用点各自保证：
              · 「攻击范围为空」那两处（chase / sweep）—— 前提天然成立；
              · tick 里 `if in_range:` 分支那一处 —— 显式要求 `len(in_range) == 1`
                （范围内只有当前目标）。**别把它删掉**：删了内侧区间就一次都不跳。
            回归用例见 tools/selftest_decision.py 的 t_chase_jump_inside_band。

        `_next_chase_jump` 这个全局节流保留：万一起跳区间边缘抖动（怪正好压在边界
        上来回），沿会反复出现，靠它兜住别连跳。
        """
        s = self.settings
        if not s.chase_jump_enabled or not edge:
            return
        # **冲刺时间闸**（用户 2026-09-26 要求）：`chase` 状态得**连续**维持够久才准起跳，
        # 否则就算区间、沿都对也不跳。计时只要进别的状态就归零（`_set_state` 之外的那处
        # 每拍续/清，见 tick 里 `_chase_since` 那段）—— 所以"刚打完 / 刚转身 / 刚进追击"
        # 这些时刻都还得先冲一段再跳。
        min_ms = max(0, int(getattr(s, "chase_jump_dash_ms", 0) or 0))
        if min_ms > 0 and (self._chase_since is None
                           or (now - self._chase_since) * 1000.0 < min_ms):
            return
        if now < self._next_chase_jump:
            return
        if not self._in_chase_jump_range(dist):
            return
        jump_key = s.keymap.get("jump")
        if not jump_key:
            return
        tap(jump_key, self._attack_duration)
        behavior.event("jump", why="追击跳")    # 战斗：追击时朝怪跳一下（行为打点 ✓）
        attack_cd_s = max(0.0, float(s.attack_cd)) / 1000.0
        # （原来这里还叠一个随机延迟；2026-09-27 删掉 ⇒ 追击跳的间隔就是 CD 本身 ✓）
        self._next_chase_jump = now + max(0.3, attack_cd_s)

    def _watch_output_held(self, now):
        """输出序列的键按着不放、而角色**早就不在输出状态** ⇒ 记一笔 + 松开 + 说明。

        2026-09-26 用户要求"先优化查卡键"（他报过"画面上一个框都没有、角色却持续在按
        输出"）。已修的根因是"怪在输出序列 down 与 up 之间消失 ⇒ up 永不发出"；这里是
        **兜底网**：任何漏放路径都会被它兜住，而且卡键从此在 perf.log 里查得到
        （`key_stuck` 计数 + `perf.note` 那句"哪个键、按了多久、当时什么状态"）——
        以前这件事**完全看不出来** ✗，只能靠猜。

        判据只看**输出序列**的键（`_output_ctx` 的 held）：移动键按住几十秒是正常的 ✗，
        输出/技能键在非输出状态下还按着就一定是漏放 ✓。宽限 `OUT_KEY_STUCK_S` 秒。
        """
        ctx = self._output_ctx
        held = set(ctx[3]) if ctx else set()
        #: 处于这些状态时，输出键按着是**正常**的（正在打）
        if self.state in ("attack", "evade_jump", "evade_back_jump"):
            self._out_held_since = now if held else None
            return
        if not held:
            self._out_held_since = None
            return
        if self._out_held_since is None:
            self._out_held_since = now
            return
        if (now - self._out_held_since) < OUT_KEY_STUCK_S:
            return
        behavior.event("key_stuck", keys="/".join(sorted(held)),
                       secs=round(now - self._out_held_since, 1), state=self.state)
        # ↑ "输出键按住太久 ⇒ 松开"（角色状态：卡键了 ✓）
        self._release_ctx(ctx)          # 逐个 key_up（不是掐断序列）
        self._output_ctx = None
        self._out_held_since = None

    def set_facing(self, f):
        """设置朝向；朝向真的变了就重置「朝向无变化」超时计时 + 记下换向时刻。"""
        if f != self.facing:
            self.facing = f
            now = time.monotonic()
            self._last_facing_change = now   # 朝向无变化超时监控用
            self._turn_at = now              # 转向后输出延迟 / 最小按住时间用

    def _steer(self, dx, keys):
        """朝目标方向走：按对应方向键并更新朝向。dx=0 不动。

        ⭐ 顺手记下"这一拍给的方向"（用户 2026-09-28 ✓ 见 `_last_move`）—— 它是**唯一**
          设移动方向的地方 ⇒ 挂这儿一处就够 ✓（`act` 快照读它 ✓）。
        """
        self._last_move = 0 if not dx else (1 if dx > 0 else -1)
        if dx > 0:
            keys.add(self.settings.keymap["right"])
            self.set_facing(1)
        elif dx < 0:
            keys.add(self.settings.keymap["left"])
            self.set_facing(-1)

    def _hold_turn(self, keys, now, cap_s=None):
        """换向后，方向键至少按住「最小切换朝向时间」——返回要真正发出去的键。

        为什么：换方向的键按下去立刻就松，角色的转身动作可能还没做完，这时候
        输出会打向**错误的方向**。所以转向期间只做一件事：**不许松开方向键**。

        「该按住的时间只能被换朝向打断」—— 所以这里**不**阻止换向：
        又按了另一个方向时，`set_facing` 已经把计时重置成新的一轮（那一刻新方向的
        键也在 keys 里），这里只处理「这一帧决策里没有方向键、但按住时间还没到」
        的情况，把当前朝向对应的方向键补回去。

        0 = 关闭（不干预，保持老行为）。

        `cap_s`：这次按住时间的**上限**（秒）。站桩输出时传 `TURN_TAP_S` ——
        那会儿按方向键只为了把角色转过来，按到"走起来"就是从怪身上走过去
        （实测按住 1 秒 = 走 1 秒，出范围又回头 → 来回抖）。
        """
        s = self.settings
        hold = max(0.0, float(s.min_turn_hold_ms)) / 1000.0
        if cap_s is not None:
            hold = min(hold, max(0.0, float(cap_s)))
        if hold <= 0 or now - self._turn_at >= hold:
            return keys
        km = s.keymap
        dirs = {km.get("left"), km.get("right"), km.get("up"), km.get("down")}
        dirs.discard(None)
        if keys & dirs:
            return keys          # 本来就在按方向键（含刚换的那个方向）→ 不用管
        key = km.get("right") if self.facing > 0 else km.get("left")
        if not key:
            return keys
        return set(keys) | {key}

    # ---------------- 上绳任务（执行器，见 decision/route.py）----------------

    def start_route(self, jobs, why="", origin=None, keep_clock=False):
        """跑一条**多步路径**（「命令前往」/「定点休息」共用）→ True = 已起跑。

        `jobs` 是**已经解析好**的任务列表（每步一个 `route.job_for_edge` 的产物：
        WalkJob / ClimbJob / DropJob）—— 路径怎么解析由**实时线程**负责
        （它手里才有地形与集合，见 `route_plan` 的说明），这里只管按顺序跑。

        每一步跑完（到了）自动接下一步；**任何一步失败就整条停掉**并把"断在第几步、
        为什么"写进 note（如实说，别硬着头皮往下走 ✗）。

        `origin` = **来源签名**（用户 2026-09-28 ✓）：`{"kind": "chase", "mob_id": …}` ——
        只有 **`"chase"`** 会在"对齐完成 / 超时过半"那两个节点被复核（见 `_climb_recheck` ✓）。
        不传 ⇒ `None` ⇒ **天然豁免**（命令前往 / 任务队列 / 定点休息 / 回区域 / 到点去哪 ✓）。

        `keep_clock` = **"这是同一趟追击被攻击逻辑重新下发"** ⇒ 时间钟**接着走、不重打** ✓
        （用户 2026-09-29 第 2 条 ✓ 见 `start_climb` 里那段说明 ✓）。默认 `False` = 老口径 ✓。
        """
        jobs = list(jobs or [])
        if not jobs:
            return False
        self._route = jobs[1:]
        self._route_why = str(why or "")
        self._route_origin = dict(origin) if origin else None
        self._route_step = 1
        self._route_total = len(jobs)
        # 行为打点（`core/behavior.py`）：一条多步路线**起跑**了。
        # 为什么记在**这一处**：所有任务（命令前往 / 定点休息 / 区域归位）都从 `start_route`
        # 进来 ⇒ 一处就够，不会漏。
        # ⚠ 用 `task_open`（**暂存**，不马上落盘 ✓）：瞬时任务（起跑到收工 < 1 秒）会被
        #   `task_settle` **合并成一条** `task_quick` —— 实测这类占 `task_done` 的 ~45%、
        #   连上起跑那条一共吃掉一半多的日志量 ✗（用户 2026-09-28 要求降噪 ✓）。
        #   ⚠ **别和下面那个事件名 `task_begin` 混**：那个说的是"**挂上这一步的执行器**"
        #     （`start_climb` 里 ✓）；这个说的是"**先暂存这条待写的起跑行**" ✓。
        behavior.task_open("task_start", why=self._route_why, steps=len(jobs))
        self.start_climb(jobs[0], keep_clock=keep_clock)
        return True

    def stop_route(self, why="取消寻路"):
        """把整条路径停掉（含还没跑的那些步）。"""
        # 玩家行为打点：**确实有任务在跑**才记 —— 这个函数在"关自动"那条早退里是**每拍**
        # 都被调一次的（空停 ✓），不加这道前提就会把日志刷满 ✗。
        if self._climb is not None or self._route:
            # ⚠ `merge=False`：被叫停**不是"走完了"** ⇒ 不许伪装成 `task_quick` ✗
            #   （走 `task_settle` 只是为了让暂存的起跑那条**按原时间戳补上** ✓）
            behavior.task_settle("task_cancel", why=why, secs=self._step_secs(),
                                 merge=False,
                                 step=("%d/%d" % (self._route_step, self._route_total)
                                       if self._route_total else ""))
        self._route = []
        self._route_step = self._route_total = 0
        # 整条路线都没了 ⇒ 「路线的来源签名」也清掉（用户 2026-09-28 ✓）
        self._route_origin = None
        # 「结束当前寻路」= **停整条**（用户 2026-09-26 的口径 ✓）⇒ 任务队列也一起清掉 ✓
        # （不清的话，人刚说"别走了"，下一拍队列又给它派一条 ⇒ 看着像按钮没生效 ✗）。
        if self._goto_queue:
            behavior.event("goto_queue_clear", n=len(self._goto_queue), why=str(why))
            self._goto_queue = []
        # 「别走了」（结束寻路 / 关自动 / 失败收工）⇒ 那面"排进去就起跑"的旗子也一起清 ✓
        #（不然它会留到下一拍，把**之后**排进来的东西当成"刚排的"处理 —— 虽然结果一样，
        # 但状态要清干净 ✓）。
        self._queue_kick = False
        self.stop_climb(why)

    def _step_secs(self):
        """这一步**有效**花了多久（秒，1 位小数）—— 进战斗那几秒**不算**
        （`_climb_started` 会在被战斗打断时重打，见 `_climb_interrupted` ✓）。
        没有起点（任务还没真跑过）⇒ `None` ⇒ 打点里**不写**这个字段（别编一个数 ✗）。"""
        t0 = self._climb_started
        if not t0:
            return None
        return round(max(0.0, time.monotonic() - t0), 1)

    def _climb_interrupted(self, now):
        """**任务没跑的那段时间**⇒ 告诉任务一声（`now` = 这段暂停的**起点**）。

        ⚠ **谁调它**：只有 `_task_settle` 一处（2026-09-26 结构性收口）。以前是散在
        `if in_range:` / 休息赶路打架两处**手工**调，"未定位玩家"和"关自动"那两条早退
        就漏了 ⇒ 任务被冤判（见 `_task_settle` 里那两条实测）。

        为什么必须显式通知（不让任务自己猜"怎么好久没喂我"）：任务内部全是**绝对时刻**
        锚点（`_t0` 超时 / `_best_at` "不动了" / `_off_since` "掉下来了"），而 `now`
        一直在走 ⇒ 打完架回来，它会把"站着挨打的那几秒"算成**爬不动了 / 走不动了 / 超时**
        ✗（用户 2026-09-26 报的"爬到 y=-170 就不再升了（3.0s 没变好）"就是这个形状 ✗）。
        收到之后由任务自己决定"回来怎么继续"（见 `route.ClimbJob.interrupted` / `_resume`）。
        幂等：同一段战斗里每拍都会被叫一声，任务只记第一次的时刻 ✓。
        """
        job = self._climb
        # ⭐⭐ **打一个"刚挨过打"的时间戳**（用户 2026-09-28 ✓ 原话："在 `_climb_interrupted` 里
        #   打一个『刚挨过打』的时间戳"）。
        #   ⚠ 写在这里的理由：它是"**任务没跑的那段时间**"的**唯一通知口** ✓（注释里写着只有
        #     `_task_settle` 一处调它 ✓），挨打 / 休息赶路打架那几拍都会走到 ✓ —— 不用另找
        #     "喝药"信号 ✗；失败那段再和"按跳时刻"比一比就知道是不是"**按跳后挨的打**"✓。
        self._climb_hit_at = float(now)
        fn = getattr(job, "interrupted", None)
        if callable(fn):
            fn(now)
        # ⚠ **真正生效的那把超时钟在这里**（`_climb_tick` 用的是 `_climb_started` +
        # 设置里的「寻路超时时间」）—— 任务内部那把已经关掉（见 `start_climb`）。
        # 忘了重打它，就成了"任务以为重算了、agent 那边照样按老起点收掉" ✗
        #（2026-09-26 用户实测："刚跳上去没爬两下就停住不爬了" ✗）。
        if self._climb_started:
            self._climb_started = now

    def _task_settle(self, now):
        """**唯一**决定"这一拍任务没跑 ⇒ 要不要报信"的地方（2026-09-26 结构性收口）。

        用户的要求：把这条收成**一处**，以后再加早退分支也不会漏。

        怎么做到：`_climb_tick` 是**唯一**会推任务的地方，它盖一个"我跑过"的章
        （`self._task_ran = True` + 记下 `_task_ran_at`）；`tick` 的**最前面**（在任何
        早退之前）调这里结算一次 —— 于是**只要这一拍没人推过任务就报信**，和它是因为
        什么没跑（战斗 / 休息 / 未定位玩家 / 关自动 / 区域规则 / 以后新加的）**无关** ✓。

        为什么原来会漏（两条实测，2026-09-26）：
          · **未定位玩家** 5 秒：恢复后第一拍 `走不动了：卡在 x=500（…还差 350 px）`；
          · **关自动** 40 秒：同上 —— 明明是人自己把自动关了，任务却被判"走不动"。
          而同样注入口径的"进战斗 5 秒"因为报了信，恢复后 note 一字未变 ✓。

        ⚠ 两个"不报"的例外，都是有理由的：
          · `_task_ran` 为真 —— 这一拍推过，任务的钟本来就该走 ✓；
          · `_task_ran_at is None` —— 任务**还没跑过第一拍**（刚挂上 / 一直没轮到），
            没有"最后一次跑的拍"可以当起点 ⇒ 不报：此时任务内部 `_t0` 还没落定，
            报信反而会把一个**将来的**起点提前写成过去（`_resume` 会扣掉一大截 ✗）。
        ⚠ 起点用 `_task_ran_at`（最后一次真正跑的拍）而**不是** `now`：结算发生在**下一拍**，
          用 `now` 会漏掉被挡住的**第一拍**，少扣一次（≈1 拍）。
        ⚠ **时序拍不参与**：`tick_timing` 本来就不推任务，那不是"任务被跳过"；参与进来会
          把 `_climb_started`（寻路超时钟）每 `TIMING_TICK`=10ms 重打一次 ⇒ 超时永不触发 ✗。
        """
        if self._climb is None or self._task_ran or self._task_ran_at is None:
            return
        self._climb_interrupted(self._task_ran_at)

    def plan_and_start_route(self, dst_set, why="", origin=None, keep_clock=False):
        """让**实时线程**解析「从现在这儿去 `dst_set`」的多步路径并起跑 ⇒ `(ok, 一句人话)`。

        谁在用：防掉线「定点休息」（`_run_rest_spot`）——它要自己走去指定地点，
        而 agent 不持有地形 / 集合 ⇒ 解析这一层是注入进来的 `route_plan`（**函数**，
        见它的说明；起点"我现在站哪个集合"每一刻都可能变，所以不能预存成表）。

        三种回值：
          · `(True, "已出发…")` —— 起跑了（`_climb` 非空，后面每一拍由 `_climb_tick` 推）；
          · `(True, "已经在「X」上了")` —— **不用走**（`_climb` 空着，直接进下一步）；
          · `(False, "为什么")` —— 没注入 / 解析不出来 / 没有可走的路 ⇒ 调用方**如实说**
            （不许硬走，也不许静默 ✗）。
        """
        dst = str(dst_set or "").strip()
        if not dst:
            return self._plan_fail(dst, "没选地点（先去玩家面板里选一个集合）")
        fn = self.route_plan
        if not callable(fn):
            return self._plan_fail(dst, "实时线程还没把「路径解析器」交给我 —— "
                                        "先在「实时」页开始，命令才发得出去")
        try:
            res = fn(dst)
        except Exception as ex:                  # noqa: BLE001
            return self._plan_fail(dst, "解析路径时出错：%s" % ex)
        if not isinstance(res, dict):
            return self._plan_fail(dst, "路径解析器给的东西看不懂：%r" % (res,))
        jobs = list(res.get("jobs") or [])
        if not jobs:
            if res.get("here"):
                return True, str(res.get("why") or ("已经在「%s」上了" % dst))
            return self._plan_fail(dst, str(res.get("why") or "解析不出路线"))
        if not self.start_route(jobs, why=why or ("前往：%s" % dst),
                                origin=origin, keep_clock=keep_clock):
            return self._plan_fail(dst, "路径是空的，没起跑")
        return True, ("已出发：%s" % " → ".join(res.get("path") or [dst]))

    def _plan_fail(self, dst, why):
        """「压根没起跑」的统一出口 ⇒ `(False, 一句人话)` + 一条行为打点。

        为什么单开一个（而不是在每个 `return False` 前各写一行 event）：**一处**才不会漏，
        而且以后再加一条失败分支时也顺手带上 ✓（同 `_task_finished` 的收口思路 ✓）。
        """
        behavior.event("task_plan_fail", dst=dst, why=why)
        return False, why

    def _task_finished(self, failed=False, why=""):
        """当前这一**步**收工（到了 / 失败了）⇒ 推进多步路径 → True = 整条都完事了。

        为什么单独一个方法：任务收工的点有好几个（到了 / 到集合收工 / 超时 / 没坐标 /
        到次数上限放弃），**每个点都要决定"下一步"** —— 散着写必然漏一个 ——
        所以统一收口到这里。
        """
        job = self._climb
        self._climb = None
        self._climb_retry_at = None
        # ⛔⛔ **这一步收工 ⇒ 「这一步的签名」也要清掉**（2026-09-28 现场修 ✗ —— 用户报
        #   "**三楼前往二楼的 drop 时，左右晃了很长时间**"✓）。现场 log 一眼就能看出来：
        #     `05:48:49.523  task_begin  dst=二楼 kind=DropJob`
        #     `05:48:50.377  task_fail   寻路超时：这一段已经跑了 10 秒  secs=10.0`
        #   —— **只隔 0.85 秒** ✗ ⇒ 一个刚建的任务带着**别人 10 秒的旧账**开工 ✓。
        #   病根：这个签名原来**只在 `stop_climb` 里清**（那条注释一字不差地写着这个病 ✓：
        #   "留着它会让下一个任务被误判成重下同一步 ⇒ 新任务一上来就超时"✗），
        #   而**任务正常收工**（到了 / 到集合收工 / 超时失败 / 没坐标）走的是**这里** ✗
        #   ⇒ 签名残留 ⇒「追击」再下一次 `DropJob(三楼 → 二楼)`（**执行器类与起终点全一样** ✓）
        #   被当成"同一步重下" ⇒ 钟**接着上一段走** ⇒ 一开工就超时 ⇒ 失败收起 ⇒ 追击又下
        #   ⇒ **死循环** ✗。人看到的就是"**左右晃**"：每一轮都从 `ALIGN` 重新走回 foothold
        #   中心（就近平齐 ✓），走到一半又被超时掐掉 ✓。
        #   ⚠ **别把判据改成"看 `_climb` 空不空"** ✗：`start_climb` 覆盖一个**正在跑**的任务时
        #     `_climb` 也是非空 —— 那种才是真"同一步重下"（钟**该接着走** ✓ 2026-09-28 那天
        #     修的就是它，见 `start_climb` 里那段 ✓）。⇒ 分界点只能是"**上一次收工了没有**" ✓，
        #     而"收工"的唯一出口就是这里（`_climb` 被置 `None` 只有三处：`__init__` ✓、
        #     这里 ✓、`stop_climb`（本来就清 ✓））。
        self._climb_sig = None
        # ⭐ 「这一步的来源签名」同理要清（用户 2026-09-28 ✓）：它是**这一步**的东西
        #   ⇒ 收工就没了 ✓（⚠ 但要**接着走下一段**时 `start_climb` 会从 `_route_origin`
        #   重新抄一份 ⇒ 整条路线的来源不变 ✓ 见那边）。
        self._climb_origin = None
        self._release_combat_keys()
        note = str(why or getattr(job, "note", "") or "")
        if note:
            self._last_goto_note = (note, time.monotonic())
        if failed:
            # 寻路失败 ⇒ 这一段也算**临时战斗**（用户 2026-09-27 要求）：半路停下时，
            # 附近有怪就**先打**，别一路不还手地往回走 ✗（熄灭见 `tick` 里"这一波
            # 打完了"那一行 —— 它不做"多久之内"的判定，只用"还有没有得打"✓）。
            self._temp_fight = True
            # 玩家行为打点：**这是这份日志最想回答的那类问题** —— "它为什么停了"。
            # `reason` 用任务自己写的那句（和界面上那行同源 ✓）；`left` = 后面还有几步没走。
            # ⚠ `merge=False`：失败**不是"走完了"** ⇒ 不许并进 `task_quick` ✗
            #   （见 `core/behavior.py::task_settle` 的说明 ✓）
            behavior.task_settle("task_fail",
                                 step="%d/%d" % (self._route_step, self._route_total),
                                 secs=self._step_secs(), merge=False,
                                 left=len(self._route), reason=note)
            if self._route:
                self._last_goto_note = (
                    "路径中断（第 %d/%d 步）：%s；后面 %d 步没走"
                    % (self._route_step, self._route_total, note, len(self._route)),
                    time.monotonic())
                self._route = []
                self._route_step = self._route_total = 0
            # ⚠ **队列也要停掉**（用户 2026-09-27 的口径："失败就整条停" ✓）：人还卡在同一处
            #   ⇒ 接着跑队列只会一条条跟着失败 ✗。清掉并说清（免得队列"悄悄没了"）。
            if self._goto_queue:
                _n = len(self._goto_queue)
                self._last_goto_note = ("%s；**队列已清空**（剩 %d 个没走）"
                                        % (self._last_goto_note[0], _n),
                                        time.monotonic())
                behavior.event("goto_queue_clear", n=_n, why=note)
                self._goto_queue = []
            return True
        if self._route:
            _done = self._route_step        # 刚走完的是第几步（下一行 `_route_step` 会 +1 ✓）
            nxt = self._route.pop(0)
            self._route_step += 1
            behavior.event("task_step",
                           step="%d/%d" % (_done, self._route_total),
                           secs=self._step_secs(), note=note)
            self.start_climb(nxt)
            return False            # 这一拍不算完事：下一拍接着走下一步
        if self._route_total > 1:
            self._last_goto_note = ("整条路径走完（共 %d 步）" % self._route_total,
                                    time.monotonic())
        # 整条走完（单步路线也走这里 ✓）—— 打点收尾。
        # ⚠ 用 `task_settle`：起跑到这一步 < 1 秒（"下一条、完成一条"）⇒ **合并成一条**
        #   `task_quick`（用户 2026-09-28 要求 ✓）；跑得久的 ⇒ 起跑那条按原时间戳补上 ✓
        behavior.task_settle("task_done", steps=self._route_total,
                             secs=self._step_secs(), note=note)
        self._route = []
        self._route_step = self._route_total = 0
        # 整条路线走完 ⇒ 「路线的来源签名」也清掉（用户 2026-09-28 ✓）
        self._route_origin = None
        # 整条路线收工 ⇒ **看队列**：还有就接着下一条（用户 2026-09-27 的「任务队列」✓）
        return self._goto_queue_next()

    def _goto_queue_next(self):
        """整条路线收工之后：**队列里还有就接着下一条** ⇒ True = 真的都完事了 ✓。

        用户 2026-09-27 的「添加任务队列」：队列里存的是**目的地**（不是路线）——
        每一条**出发时才**重新解析"从我现在站的地方怎么过去"（起点会变 ✓，见
        `plan_and_start_route`）。
        · 某一条**解析不出来 / 没有可走的路** ⇒ **停整个队列**并说清（和"一段失败就
          整条停"同一个口径：留着只会一条条接着失败 ✗）；
        · 某一条**已经在那个集合上了**（`plan_and_start_route` 报 `ok=True` 但没起跑 ✓）
          ⇒ 那一条**不用走**，直接接着下一条 ✓（所以这里是 `while` 不是 `if` ✓）。
        """
        while self._goto_queue:
            dst = self._goto_queue.pop(0)
            left = len(self._goto_queue)
            ok, msg = self.plan_and_start_route(dst, why="任务队列：前往 %s" % dst)
            if not ok:
                self._goto_queue = []
                self._last_goto_note = ("队列里的「%s」下不去：%s；后面 %d 个已取消"
                                        % (dst, msg, left), time.monotonic())
                behavior.event("goto_queue_fail", dst=dst, why=msg, left=left)
                return True
            if self._climb is not None:
                # 真的起跑了 ⇒ 剩下的等它走完再接着来 ✓
                self._last_goto_note = ("任务队列：接下一条「%s」（队列里还剩 %d 个）"
                                        % (dst, left), time.monotonic())
                behavior.event("goto_queue_step", dst=dst, left=left)
                return False
            # `ok=True` 但没起跑 = "已经在「X」上了" ⇒ 这一条不用走，继续下一条 ✓
            self._last_goto_note = ("任务队列：「%s」已经到了，跳过（还剩 %d 个）"
                                    % (dst, left), time.monotonic())
        return True

    def start_climb(self, job, keep_clock=False):
        """挂上一个**上绳/下跳任务**（`route.ClimbJob` / `DropJob`）；下一次 tick 开始执行。

        谁来调：`gui/route_panel.py` 的「命令前往」（把路线的第一步交过来，2026-09-26）。
        这里**没有"要不要按跳"的开关**了 —— 命令挂上来就是授权，一路做到位。

        `keep_clock=True` = **"这是同一趟追击被攻击逻辑重新下发的"** ⇒ 已有的时间钟
        **接着走、不重打**（用户 2026-09-29 第 2 条 ✓ 见下面那段说明 ✓）。
        ⚠ 它只在"**真的有一趟追击正在跑**"时才算数 ✓（见下面 `_keep` 那个判据 ✓）——
          跑完 / 被超时切掉之后再下新的追击，那是**新的一趟**，预算该是新的 ✓
          （不然会带着上一趟的旧账一开工就超时 ⇒ 又回到"左右晃"那个死循环 ✗）。
        """
        # ⚠ 这两件必须在覆盖之前取（下面要用"**被替换掉的是不是一趟在跑的追击**"✓）：
        _had_climb = self._climb is not None
        _was_chase = self.chase_tagged()
        self._climb = job
        self._climb_retry_at = None      # 清掉上一个任务留下的"等我到点再重来"
        # ⭐ 两个"挨打 / 按跳"时间戳也归零（用户 2026-09-28 ✓）：**新任务**必须从"没挨过打、
        #   没跳过"开始 ✓ —— 不然上一个任务挨的那顿打会被当成本任务的 ⇒ 白等一段延迟 ✗。
        #   ⚠ 只在这里（新任务）清 ✓：**同一任务失败重来**时**不清** ✗
        #     （"按跳后挨过打"的记忆要靠它 ✓，清了就永远判成"没挨打"⇒ 该等的也不等了 ✗）。
        self._climb_hit_at = 0.0
        self._climb_jump_at = 0.0
        # 位置状态的观察窗**从这一步开始**（`_maybe_replan` ✓）：刚挂上的那一眼不算"变化"，
        # 只比"**这一步执行期间**的变化" ✓。
        self._loc_state = None
        # ⚠ 新任务**还没跑过第一拍** ⇒ 清掉上一个任务的"最后跑过的那拍"（`_task_settle`
        #   靠 None 判断"还没轮到它"，不清的话会拿**上一个任务**的时刻当暂停起点 ✗）。
        self._task_ran_at = None
        # ⛔ **「同一步被重下」不许重置「寻路超时时间」**（2026-09-28 用户报"**没有触发寻路
        #   超时**"✓ —— 这条是本次最要害的 ✓）。数据对得上：`behavior.log` 里
        #   `task_begin kind=DropJob` **318 次**，而 `goto_timeout` 只有 **57 次** ✗。
        #   病根就在这一行原来**每次都重打** `_climb_started`：而"追击改目的地 / 回区域"
        #   那条路（`_chase_goto_if_elsewhere` → `plan_and_start_route`）**每 1~3 秒就可能
        #   重下一次同一步**（`mob_goto_ok` ✓）⇒ 10 秒的钟被 3 秒一次的重下**反复归零**
        #   ⇒ 卡住的那一段**永远等不到超时** ✗（现场看到的就是"卡住不动、也没报超时"✗）。
        #   ⚠ 与设计意图也不符：下面那条注释白纸黑字写着"**每完成一段**就重新计时" ✓，
        #     而"重下同一步"**不是**完成一段 ✗。
        #   ⇒ 判据 = **同一步**（同类执行器 + 同一个起点集合 + 同一个目标集合）⇒ 钟**接着走** ✓；
        #     真换了下一步（`_task_finished` 接下一段 / 新命令 / 手工取消）⇒ 重新计时 ✓
        #     （`_climb_sig` 在 `stop_climb` 里清掉 ⇒ 不会把"走→爬→走"那种"集合名凑巧一样"
        #      的**新任务**误判成同一步 ✗）。
        _sig = (type(job).__name__, str(getattr(job, "src_set", "") or ""),
                str(getattr(job, "dst_set", "") or ""))
        # ⭐⭐ **攻击逻辑重下 ⇒ 不许刷新那把钟**（用户 2026-09-29 第 2 条 ✓ 原话："追击签注的
        #   寻路任务不应该被 attack 刷新，可以被单段寻路执行器成功刷新"）。
        #   · 上面那套"同一步重下不重置"只挡得住**同一步**（同类执行器 + 同源 + 同目标 ✓）——
        #     而追击最常见的重下恰恰是**怪换层了 ⇒ 目标集合变了 ⇒ 签名变了** ✗ ⇒ 钟被重打
        #     ⇒ 一趟永远追不上的追击可以**无限续命** ✗✗（30 秒的闸形同虚设 ✓）。
        #   · ⇒ 给攻击那两处下发带 `keep_clock=True`（`_chase_goto_if_elsewhere` /
        #     `_mob_goto_towards` ✓）：**已经有钟就接着走** ✓（这一趟追击的总预算不变 ✓），
        #     没有钟（第一次下发 / 上一趟已经收工 ✓）才起钟 ✓。
        #   · ⚠ **"单段寻路执行器成功"那条路不受影响**（用户明确要它能刷新 ✓）：它走的是
        #     `_task_finished` → `start_climb(nxt)`（**不带**这个参数 ✓）⇒ 照旧按签名重打 ✓
        #     ⇒ 走完一段、接下一段时预算是新的 ✓（"一段"= 一条边的一个执行器 job ✓）。
        # ⚠ `_keep` 还要"**被替换掉的那一步正是一趟在跑的追击**"✓：
        #   · `_had_climb` + `_was_chase` ⇒ 是"同一趟追击在跑、被杀怪逻辑重下" ⇒ 钟接着走 ✓；
        #   · 跑完 / 被超时切掉之后再下（`_had_climb=False`）⇒ 那是**新的一趟** ⇒ 起新钟 ✓
        #     （不然带着上一趟的旧账一开工就超时 ⇒ 退回"三楼→二楼 drop 左右晃"那个死循环 ✗）。
        _keep = bool(keep_clock and _had_climb and _was_chase and self._climb_started)
        if (not _keep) and (_sig != self._climb_sig or not self._climb_started):
            self._climb_started = time.monotonic()   # 「寻路超时时间」从这一刻算起
        self._climb_sig = _sig
        # ⭐ **「这一步」的来源签名**（用户 2026-09-28 ✓）—— 从**这条路线**那份抄一份
        #   （`_route_origin` ✓），随这一步生灭 ✓。接下一段时 `_task_finished` 会再调这里 ⇒
        #   自动拿到同一份来源 ✓（整条路线的来源不变 ✓）。
        self._climb_origin = dict(self._route_origin) if self._route_origin else None
        # ⭐ 两个复核节点的"这一步查过了没有"也**随任务复位** ✓（用户 2026-09-28 ✓）：
        #   ① 「对齐完成、起跳前」；② 「寻路超时计时过半」—— 每个都**每步只查一次** ✓
        #   （不复位 ⇒ 下一步永远不会复核 ✗；不限制次数 ⇒ 每拍都算一轮代价，白烧 CPU ✗）。
        self._recheck_armed_done = False
        self._recheck_half_done = False
        # ⚠ **超时只留一把钟**（2026-09-26 现场："明明在打架，超时还是把这一段收掉了"）：
        # 任务自己还带一个 `timeout_s`（默认 12 秒，**不受设置控制** ✗），于是"设置里的
        # 寻路超时时间"根本管不住它 —— 两个钟各掐各的，人在界面上改了一个还以为生效了 ✗。
        # 现在统一由 **agent** 管（`_climb_tick` 里 `_climb_started` + 设置值 ✓，
        # 而且「每次进战斗」会重打它 ✓），任务内部那把钟**关掉**（设成很大的数 ✓）。
        try:
            job.timeout_s = 1e9
        except Exception:                     # 老执行器没有这个属性也不该炸
            pass
        # 「**移动操作尝试间隔(ms)**」（2026-09-27 用户要求：改名 + 搬到「寻路配置」组）——
        # 一件参数管两处"按了没反应就再来一次"（见 `DecisionSettings.move_retry_ms` 的说明）：
        #   · 爬绳：`ClimbJob.reassert_s`（**秒**）—— y 不动 ⇒ 先补按 ↑ 观察这么久；
        #   · 下跳：`DropJob.retry_ms`（**毫秒**）—— ↓+点按跳之后这么久 Y 没动 ⇒ 补发
        #     "松开 ↓" 再来一次 ✓。
        # 两个任务各拿自己需要的单位（换算只在这一处做 ⇒ 不会出现两套口径 ✗）。
        # 0 = 不重试（老行为）✓；读不到设置也给 0（保守：宁可退回老行为，别乱重试 ✗）。
        try:
            _retry_ms = max(0, int(getattr(self.settings, "move_retry_ms", 0) or 0))
        except Exception:                     # noqa: BLE001
            _retry_ms = 0
        try:
            job.retry_ms = _retry_ms                   # DropJob 用毫秒 ✓
        except Exception:                     # noqa: BLE001
            pass
        # 「**连按跳的一轮窗口**」= 现在**统一**用「移动操作尝试间隔」换算成秒 ✓
        # （2026-09-27 用户要求："移除卡住判定时长(s)，统一采用移动操作尝试间隔(ms)" ✓）——
        # `ClimbJob` / `JumpJob` 那些"多久没进展算卡住"的判据也走同一件参数 ✓。
        try:
            from decision import route as route_mod
            job.stall_s = route_mod.stall_s_from_retry_ms(_retry_ms)   # 秒 ✓（一处实现 ✓）
            # ⭐ 「**中途跳下**」落地之后**再按住 ↑ 多久**才松（用户 2026-09-28 原话：
            #   "直到**返回成功或失败后**再经过「**移动操作尝试间隔**」松开"✓）——
            #   用的就是**同一个参数** `move_retry_ms` ✓（口径一处 ✓ 别新造常数 ✗），
            #   换算成**秒**灌进去（执行器里一律用秒比 ✓ 同 `stall_s` ✓）。
            job.mid_hold_s = max(0.0, _retry_ms / 1000.0)
        except Exception:                     # noqa: BLE001
            pass
        # 「Y 变化超过多少才算**真的开始下落**」= 用户口径里的「坐标对齐误差范围」
        # （`DropJob.y_tol_px`，世界像素 ✓）。**同一把尺**：直接取设置里那个数，
        # 不许在任务里另拍一个 ✗（默认值取 `route.ALIGN_TOL_PX` 也是这个意思 ✓）。
        try:
            from decision import route as _route     # 局部 import：避免模块级循环 ✓
            _tol0 = float(_route.ALIGN_TOL_PX)
        except Exception:                     # noqa: BLE001
            _tol0 = 10.0                      # 兜底只求别把这一拍弄坏 ✓
        try:
            job.y_tol_px = max(1.0, float(
                getattr(self.settings, "align_tol_px", _tol0) or _tol0))
        except Exception:                     # noqa: BLE001
            pass
        # 「**对齐绳梯移动延迟(ms)**」也灌进任务（用户 2026-09-27 加的参数）：
        # 对齐 x 时**两次按下方向键之间的最小间隔**。0 = 不限制；设置里没有这个键时
        # 给 180（= `route.TAP_PERIOD_S`，也就是老行为 ✓）。
        try:
            job.align_gap_ms = max(0, int(
                getattr(self.settings, "climb_align_gap_ms", 180) or 0))
        except Exception:                     # noqa: BLE001
            pass
        # 「**开始对齐绳梯x的距离(px)**」也灌进任务（用户 2026-09-27 加的参数，同属
        # "对齐那一步"）：差得比它远 ⇒ 一口气按住；≤ 它 ⇒ 点按微调。实在读不到就给 20
        #（= `route.NEAR_PX` = 老行为 ✓，不让一条读配置出错的路把上绳弄坏 ✗）。
        try:
            job.near_px = max(1.0, float(
                getattr(self.settings, "climb_align_near_px", 20) or 20))
        except Exception:                     # noqa: BLE001
            pass
        behavior.event("task_begin", kind=type(job).__name__,
                       dst=getattr(job, "dst_set", ""),
                       ladder=getattr(job, "ladder_id", "") or "")
        return job

    def stop_climb(self, why="取消上绳"):
        """撤掉上绳任务（换目标 / 关自动 / 出错都走这里）：状态收干净，别再按键。"""
        if self._climb is None:
            return
        self._climb.cancel(why)
        self._climb = None
        self._climb_retry_at = None
        # ⚠ 顺手把"这一步的签名"也清掉（见 `start_climb` 里那段 ✓）：留着它会让**下一个**
        #   任务被误判成"重下同一步"⇒ 那颗「寻路超时」的钟不重打 ⇒ 新任务**一上来就超时** ✗
        #   （"走→爬→走"这种集合名凑巧一样的连着来，最容易撞上 ✓）。
        self._climb_sig = None
        # 来源签名同理（这一步没了 ⇒ 它也没了 ✓）；整条路线的来源由 `stop_route` 清 ✓
        self._climb_origin = None
        self._release_combat_keys()
        behavior.event("task_abort", why=str(why))
        if self.state == "climb":
            self._set_state("idle")

    def queue_goto(self, dst_set):
        """把「前往 `dst_set`」**排进任务队列**（用户 2026-09-27 的「添加任务队列」）⇒ 队列长度。

        谁在用：`gui/route_panel._on_goto_queue`。
        ⚠ 空名字**不加**（下拉选的是空项时别塞一个"前往空"✗）；重复也**不去重**：
           "同一块平台去两趟"是合法需求（绕路 / 再走一遍 ✓）。
        """
        dst = str(dst_set or "").strip()
        if not dst:
            return len(self._goto_queue)
        self._goto_queue.append(dst)
        behavior.event("goto_queue_add", dst=dst, n=len(self._goto_queue))
        # **排进去就该跑起来**（用户 2026-09-27："寻路任务队列要依次执行，现在我排队列
        # 都没反应"）：手上没任务 ⇒ 拍旗子，下一拍 tick 就起跑第一条 ✓
        #（不用再按一次「命令前往」✓）；已经有任务在跑 ⇒ 它就排队等（`_task_finished`
        # 收工时接下一棒 ✓ —— 这就是"依次执行"✓）。
        # ⚠ 只拍旗子、**不在这里起跑**：`plan_and_start_route` 那套活儿属于决策线程 ✓。
        if self._climb is None and not self._route:
            self._queue_kick = True
        return len(self._goto_queue)

    def holding_vertical(self):
        """现在**按着 ↑ 或 ↓**吗 —— "在不在绳梯上"那条判定的**许可条件** ✓。

        用户 2026-09-27 要求（原话）："位置状态判定优化：**除非按住了 ↑ 或 ↓，不能主动判定为
        在绳梯上**，不然角色碰到绳子就卡住不走" ✓ —— 感知层（`gui/live_thread._fill_route_ctx`）
        在写 `player.ladder_id` 之前会问这一句 ✓。

        口径：直接看**当前按着的键**（`KeyState.pressed()` ✓，就是本机真按住的那几个 ✓），
        不另记账 —— 记账会跟实际按键飘开 ✗。键位从 `settings.keymap` 取（"up"/"down" ✓）。
        """
        km = getattr(self.settings, "keymap", None) or {}
        try:
            held = self.keys.pressed()
        except Exception:                      # noqa: BLE001 —— 拿不到就按"没按"处理 ✓
            return False
        return bool((km.get("up") and km.get("up") in held)
                    or (km.get("down") and km.get("down") in held))

    def climbing_vertical(self):
        """climb / drop 执行器上一拍**发了 ↑/↓**吗 —— ladder_id 许可的**新来源**（诉求 1 ✓）。

        用户 2026-09-28："位置状态允许判定在绳梯的入口为『climb 执行器按住 ↑ 后发起通知』，
        而不是『按住 ↑』" —— 判定"在绳上"的许可，从"本机按着 ↑ 键"（`holding_vertical` 看
        `KeyState.pressed`）改成"执行器上一拍确实发了 ↑/↓"（`ClimbJob._last_vert` ✓）。

        为什么：`KeyState.pressed` 是**本机按键状态**，而"执行器发键 → 远端角色真爬"之间
        有端到端延迟、还可能丢键 ⇒ 拿本机按键状态当许可，会跟执行器意图飘开 ✗；执行器
        自己记的 `_last_vert`（"上一拍我请他按的竖直键"）才是"发起通知"的准确事实 ✓。
        ⚠ 覆盖**两个**执行器（用户 2026-09-28 追加："drop 按 ↓ 时也发通知，直到 drop 收工
        再关上"）：`ClimbJob`（按 ↑/↓ 爬）与 `DropJob`（按 ↓ 下跳，`dir=-1` ✓）——
        两者都记 `_last_vert`；走/跳上一拍发的方向不算 ✓。
        """
        job = self._climb
        if job is None or type(job).__name__ not in ("ClimbJob", "DropJob"):
            return False
        return bool(getattr(job, "_last_vert", 0) != 0)

    # ⛔ `current_target_sets()`（"锁定目标在哪块集合" → 给锁定框下方那行用）**已删** ✗：
    #   用户 2026-09-27 明确"**以前的锁定框表地点就不要了**" ⇒ 界面不再要这份东西 ⇒
    #   连带它那份缓存（`_target_sets`）一起删掉（**死数据不留** ✗）。
    #   ⚠ 地点现在只在**「查过的怪框」**那一行上（`live_thread` 读 `queried_mob_boxes()` ✓，
    #     数据来自 `mob_sets_of` 那个**唯一漏斗**的记账 ✓ —— 绘制层依旧**不许**自己扫 foothold ✗）。

    def goto_queue(self):
        """任务队列里**还没跑**的那些目的地（一行一个显示用 ✓）。

        给界面用（`route_panel._osd_lines`）—— ⚠ **别让界面去读 `_goto_queue`**：
        那是私有的运行时状态（同 `current_goto_set` 那条规矩 ✓）。
        """
        return list(self._goto_queue)

    def current_goto_set(self):
        """当前**寻路任务**的目标集合名（没有任务时空串）。

        给界面用（「路线识别」页的「当前任务」那行、以及「结束当前寻路」按钮）：
        寻路任务就是挂着的上绳/下跳 job（`start_climb`），它带着 `dst_set`。
        **别再让界面去读 `_climb`** —— 那是私有的运行时状态，改一次名字就全线崩。
        """
        job = self._climb
        return str(getattr(job, "dst_set", "") or "") if job is not None else ""

    def current_goto_text(self):
        """当前**执行器**那一行（界面用）：`下跳(drop):{详细内容}` —— 名字 + 它现在在干什么 ✓。

        用户 2026-09-27 要求（原话）："你能把当前的执行器写一下吗？例如格式：
        `下跳(drop):{详细内容}`" —— 原来那行只有一句自由文本（`current_goto_note` ✓），
        卡住时**看不出现在跑的是哪个执行器**（走 / 爬 / 下跳 / 跳）✗。

        中文名**不在这里拼**：走 `route.job_label`（它再走 `zones.kind_label` ⇒ 一处口径 ✓）：
        `走(walk)` / `爬（绳梯）(climb)` / `跳(jump)` / `下跳(drop)` ✓。
        没有 job 时退回 `current_goto_note()`（任务刚结束还挂一会儿的那句 ✓ 老行为不变 ✓）。
        """
        from decision import route as route_mod

        return route_mod.job_label(self._climb) or self.current_goto_note()

    def current_goto_note(self):
        """寻路任务**当前那一步在干什么 / 为什么失败**（没任务又一无所有时空串）。

        给画面那行用（`route_panel._osd_lines`）。为什么非要露出来：2026-09-26 用户连着
        两次报"角色爬到某个 y 就不动了"，而任务里其实**一直写着原因**（对齐中差几像素 /
        偏离绳 / 爬不动了 / 拿不到世界坐标 / 到达）—— 只是没人看得到 ⇒ 只能靠猜。

        ⚠ 任务**结束之后**还要挂一段时间（`GOTO_NOTE_KEEP_S`）：用户问的是"为什么在 -170
        就松开了 ↑"，而任务一结束那行就没了 ⇒ 看见的时候已经无从查证 ✗。
        """
        job = self._climb
        if job is not None:
            return str(getattr(job, "note", "") or "")
        note, at = getattr(self, "_last_goto_note", ("", 0.0))
        if note and (time.monotonic() - at) <= GOTO_NOTE_KEEP_S:
            return "刚才：%s" % note
        return ""

    def _target_dir(self, ws):
        """锁定的目标在玩家的**哪一边**（`+1` 右 / `-1` 左 / `0` 没有目标或同列 ✓）。

        用途：⭐ 下跳"跳下绳子"那一步的**水平方向**（用户 2026-09-28 按流程图定的口径 ✓
        原话："跳下绳子：按任意方向+跳（**若有锁定目标则是锁定目标方向**）"）——
        注入给 `DropJob._target_dir_fn` ✓（见 `_climb_tick` 里那一行 ✓）。

        ⚠ **只比画面 x 的左右**（`ws.player.x` vs 怪框 `m.x` ✓）：两边**同一个坐标系**，
          只判左右 ⇒ **不需要世界坐标** ✓（脱离只要"往哪边按"✓）。
        ⚠ 拿不到（没锁定 / 目标不在这帧的 `ws.mobs` 里 / 玩家缺失）⇒ `0` ✓
          ⇒ 调用方退回"任意方向"（背离下跳点中心 ✓），**不许因此报错** ✗。
        """
        try:
            tid = getattr(self, "_target_id", None)
            if tid is None or ws is None:
                return 0
            p = getattr(ws, "player", None)
            if p is None:
                return 0
            for m in (getattr(ws, "mobs", None) or ()):
                if getattr(m, "id", None) == tid:
                    _dx = float(getattr(m, "x", 0.0) or 0.0) \
                        - float(getattr(p, "x", 0.0) or 0.0)
                    if abs(_dx) < 1.0:
                        return 0                  # 正好同列 ⇒ 别瞎挑一边 ✓
                    return 1 if _dx > 0 else -1
            return 0                              # 目标这一帧没框 ⇒ 不认 ✓
        except Exception:                         # noqa: BLE001
            return 0

    def current_goto_tag(self):
        """「当前任务」的**短签注**（信息栏那行括号里那个词 ✓ 用户 2026-09-28 举例：
        `前往：底层(追击  剩余 00:15)`）。

        ⚠ 口径：**只从现成的来源取短词，不自造词表** ✗（项目纪律"不许拍脑袋补数"✓）——
          来源 = `_climb_origin["kind"]`（结构化的"这条任务为什么下"，唯一写口在
          `_chase_origin` ✓，值如 `{"kind": "chase", "mob_id": 123}` ✓）。
        ⚠ **没有签注 ⇒ 返回空串** ✓（信息栏就只显示任务名 + 剩余时间，**不硬塞一个词** ✗）；
          以后新增 kind ⇒ 只要往下面那张小映射加一行即可 ✓。
        """
        o = getattr(self, "_climb_origin", None) or {}
        k = str(o.get("kind") or "")
        return {"chase": "追击"}.get(k, "")

    def chase_tagged(self):
        """当前这一步是不是**「追击」签注**的 —— 这是"有没有时间限制"的**唯一判据** ✓
        （用户 2026-09-29 第 1 条："只有追击签注的任务是有时间限制的（到点结束任务），
        其他任务不应该有时间限制（无限）"✓）。

        判据用 `_climb_origin` 的结构化 `kind`（唯一写口 `_chase_origin` ✓），**不是**
        一句话原因 / 执行器类名 ✗ —— 那些"凑巧提到追击"的任务不该被算进来 ✓
        （见 `current_goto_tag` 的映射表 ✓）。
        """
        return str((getattr(self, "_climb_origin", None) or {}).get("kind")
                   or "") == "chase"

    def goto_time_left(self):
        """「当前任务」还剩几秒（信息栏用 ✓ 用户 2026-09-28："签注需要有生存时间"）。

        ⚠ 口径**只有一处**：和 `_climb_tick` 判"寻路超时"用的是**同一把钟** ——
          `self._climb_started`（同一步重下**不重置** ✓、挨打暂停会重打 ✓）
          ＋ `settings.goto_timeout_s`（「寻路超时时间」✓）
          ⇒ 显示的"剩余"和真到点那一刻**必然一致** ✓（自己另起一把钟就会对不上 ✗）。
        ⭐⭐ **只有「追击」签注才有时间限制**（用户 2026-09-29 第 1 条 ✓）：其它任务
          （命令前往 / 定点休息 / 回战斗区域 / 战斗时长到点换地方…）**一律不限时** ⇒ 这里
          返回 `None` ✓（信息栏那行**不显示剩余** ✓ 也就不会让人以为"它马上要被切"✗）。
          ⚠ 与 `_climb_tick` 里那道总闸**同一个判据**（`chase_tagged()` ✓）—— 显示与
            真到点必须一致 ✓（一处口径 ✗ 别各判各的）。
        ⭐⭐ **取两道闸里"更早到期的那个"**（2026-09-28 用户报 ✓："我改了 5 秒，但是信息栏写的
          追击还是 **9 秒**开始（也可能是 10）"✗）：追怪任务（`_climb_origin` 的
          `kind == "chase"` ✓）且「追怪寻路.duration(s)」> 0 时，把它与「寻路超时时间」**取 min** ✓
          —— 两道闸**谁先到、任务就结束在谁那儿**，只显示大的那个会**骗人** ✗。
        ⚠ 返回 `None` 的情况（**都不显示** ✓ 别显示 `0:00` ✗）：
          · **不是追击签注**（用户 2026-09-29 ✓ "其他任务不限时"）；
          · **两道闸都没开**（`goto_timeout_s <= 0` **且**（不是追怪 或 追怪值也 <= 0）✓）；
          · 还没起钟（`_climb_started` 为 0 ✓）；
          · 当前没有在跑的寻路任务（`_climb` 为空 ✓）。
        """
        t0 = float(getattr(self, "_climb_started", 0.0) or 0.0)
        if t0 <= 0.0 or getattr(self, "_climb", None) is None:
            return None
        if not self.chase_tagged():
            return None                      # 非追击 ⇒ 不限时 ⇒ 不显示剩余 ✓
        cap = max(0.0, float(getattr(self.settings, "goto_timeout_s", 0.0) or 0.0))
        # ⭐⭐ **取两道闸里"更早到期的那个"**（用户 2026-09-28 报 ✓ 原话："我改了 5 秒，但是
        #   信息栏写的追击还是 **9 秒**开始（也可能是 10）"✗）。
        #   · 信息栏那行说的是"**离这趟任务结束还有多久**" ✓ ⇒ 而两道闸**谁先到、任务就结束在
        #     谁那儿** ✗ ⇒ 只显示大的那个会**骗人**（用户就是这样被骗的：设了 5 秒，屏上却从
        #     10 秒开始倒数 ✓）；
        #   · 追怪任务（`_climb_origin` 的 `kind == "chase"` ✓）且「追怪寻路.duration(s)」> 0
        #     ⇒ 把它一起算进来 ✓；否则只看「寻路超时时间」✓（老行为一字不变 ✓）。
        _caps = [cap] if cap > 0.0 else []
        _chase = max(0.0, float(
            getattr(self.settings, "chase_goto_max_s", 0.0) or 0.0))
        if (_chase > 0.0
                and str((getattr(self, "_climb_origin", None) or {}).get("kind")
                        or "") == "chase"):
            _caps.append(_chase)
        if not _caps:
            return None                      # 两道闸都没开 ⇒ 不显示 ✓（别显示 0:00 ✗）
        return max(0.0, min(_caps) - (time.monotonic() - t0))

    def resetall_left(self):
        """离下次**定时清键**（RELEASEALL）还有几秒；没排期时给 None。

        给界面用（「当前任务」下面那几行计时）：和 `_next_resetall` 同一把钟
        （`time.monotonic` 秒）。**排期发生在 tick 里**，所以刚开自动那一拍到排期
        之间就是 None —— 界面据此写"未排期"，别显示 0:00 骗人。
        """
        if self._next_resetall <= 0:
            return None
        return max(0.0, self._next_resetall - time.monotonic())

    def _watch_player_visible(self, now, s, ws):
        """「画面里连续看不到角色」⇒ **如实留痕**（用户 2026-09-27 要求）。

        **为什么单开一条**（它和下面 `player_lost_timeout_min` 那条的区别）：
          · 那条只做一件事 —— 把自动停掉；而且它排在 `if not s.enabled: return`
            **之后** ⇒ 自动一旦已经关着（或它刚自己把自动停掉），这件事就
            **完全不留痕** ✗；
          · 现场正是这样：角色凌晨被卡住打死，上午翻 perf.log 只看到"每 30 秒一条
            按键"（那是定时 RELEASEALL），**分不清是死了、是卡了、还是根本没开自动** ✗。
        所以这一条：
          · **和"自动开着没"无关**（决策拍每拍都看，且排在关自动早退之前 ✓）；
          · 记一条**持续更新**的注记 ⇒ 段头就能看到"画面里看不到角色多久了" ✓
            （注记不设阈值：看不见就是看不见，不拿我编的数去替用户设的那个 ✓）；
          · 到了设置里那个时长（「找不到玩家停止自动」，0 = 不看这件事）**再记一次**
            `player_gone` 计数 ⇒ 一段日志里出现它 = 那段时间确实丢过角色 ✓。
        恢复时记 `player_back` 并把注记清掉 ✓。
        """
        if ws is None:
            return
        if ws.player.found:
            if self._player_gone_since is not None:
                behavior.event("player_back")
                self._player_gone_since = None
                self._player_gone_counted = False
            return
        if self._player_gone_since is None:
            self._player_gone_since = now      # 刚看不见：先起算（漏一两拍很正常，别急着报）
            return
        gap = max(0.0, now - self._player_gone_since)
        lim = max(0.0, float(s.player_lost_timeout_min)) * 60.0
        if lim > 0 and gap >= lim:
            if not self._player_gone_counted:
                self._player_gone_counted = True
                behavior.event("player_gone", mins=round(gap / 60.0, 1),
                               why="画面里看不到角色（死亡界面/弹窗/切图？）")

    # ---- 「位置状态变了就重算剩下的路」（用户 2026-09-27）----

    def _route_final_dst(self, job):
        """这条路线**最终要去哪**（重算的目标）—— 多步路径 / 队列都一样：**最后一个节点** ✓。

        `_route` 空（单步任务）⇒ 就是当前这一步的 `dst_set` ✓。
        """
        for j in reversed(self._route or []):
            d = str(getattr(j, "dst_set", "") or "")
            if d:
                return d
        return str(getattr(job, "dst_set", "") or "")

    def _maybe_replan(self, p):
        """位置状态变了 ⇒ **按现在站的地方重新评判一次最优路径** ⇒ True = 重算过（已起跑）。

        用户 2026-09-27："当玩家的当前位置状态（位于 foothold 集或绳梯等）改变时，要重新
        评判一次最优路径"。

        「**位置状态**」= `(here_sets, ladder_id)`（定义与粒度边界见 `_loc_state` 那段 ✓）：
        只有"**换集合 / 掉出所有集合 / 上或下绳**"才算变化；在**同一个集合里**走动、被怪
        撞来撞去（x/y 变、朝向变）**都不算** ✓ —— 路线的节点没变、计划仍然成立，重算只会
        白打断它 ✗。

        为什么需要：`start_route` 拿到的是一条**已经算好**的任务列表（按"下命令那一刻人在
        哪"算的 ✓）。半路被人撞下去 / 走岔 / 落点偏了 ⇒ 剩下那几步**全都不对了** ✗
        （最坏是一路朝反方向走到「寻路超时」）。所以每拍对一次 `(脚下属于哪些集合, 贴在
        哪根绳上)`，一变就按**新的当前位置**重新解析到**同一个最终目的地** ✓。

        四条守卫（都是为了别把正在做的事打断、别来回抖 ✗）：
          · 还没看过 ⇒ 只记不算（刚挂上这一步那一眼不算"变化" ✓）；
          · **动作进行中**（`route.INFLIGHT_PHASES` 那几个相位，或人贴在绳梯上）⇒ **先不
            重算**，而且**不记状态** —— 等它落地 / 离绳，下一拍还认得出这次变化 ✓；
          · 脚下**就是这一步要去的那块** ⇒ 那是**正常到站**，交给收工那套（`_task_finished`
            → 接下一步 ✓）—— 拿它重算会来回抖 ✗；
          · 重新解析**失败**（没地形 / 脚下没圈进任何集合 —— 半空、缝隙里很常见）⇒
            **保持现状** + 记一笔 note（别把好端端的路线砍掉 ✗，后面还有「寻路超时」兜底 ✓）。

        重算成功 ⇒ 走 `start_route`（= 停掉当前这一步 + 按新计划起跑第一步 ✓），顺带把那把
        「**寻路超时**」的钟**重新计时** ✓（新计划从这一刻起算才合理 ✓）。
        """
        from decision import route as _route_mod

        here = set(getattr(p, "here_sets", None) or ())
        loc = (frozenset(here), str(getattr(p, "ladder_id", "") or ""))
        if loc == self._loc_state:
            return False
        if self._loc_state is None:              # 刚挂上这一步的第一眼 ⇒ 只记 ✓
            self._loc_state = loc
            return False
        job = self._climb
        if job is None:
            self._loc_state = loc
            return False
        # 动作进行中（人在绳上 / 空中）⇒ 等它告一段落；**状态不记**（下一拍还能认出来 ✓）
        if (str(getattr(p, "ladder_id", "") or "")
                or str(getattr(job, "phase", "") or "")
                in _route_mod.INFLIGHT_PHASES):
            return False
        # 正常到站（脚下就是这一步要去的那块）⇒ 不重算 ✓
        if str(getattr(job, "dst_set", "") or "") in here:
            self._loc_state = loc
            return False
        dst = self._route_final_dst(job)
        resolver = getattr(self, "route_plan", None)
        plan = None
        if dst and callable(resolver):
            try:
                plan = resolver(dst)
            except Exception as ex:              # noqa: BLE001
                plan = {"jobs": [], "why": "重新解析时出错：%s" % ex}
        self._loc_state = loc
        where = "、".join(sorted(here)) or "不在任何集合里"
        jobs = list((plan or {}).get("jobs") or [])
        if not jobs:
            why = str((plan or {}).get("why") or "解析不出路线（没有解析器？）")
            behavior.event("replan_fail", dst=dst, why=why)
            text = ("位置变了（现在在「%s」）⇒ 重新算去「%s」的路没算出来：%s"
                    " —— 先按原计划走，「寻路超时」兜底" % (where, dst, why))
            # ⚠ `job.note` 只活到这一拍的 `job.update(...)`（就在下面几行，会被盖掉 ✗）
            # ⇒ **真正留痕的是这两处**：`behavior.log`（上面那条 `replan_fail` ✓）、
            #   `_last_goto_note`（任务结束之后还能回答"刚才为什么没重算"，
            #   `GOTO_NOTE_KEEP_S` 之内 ✓）。
            job.note = text
            self._last_goto_note = (text, time.monotonic())
            return False
        behavior.event("replan", dst=dst, at=sorted(here), steps=len(jobs))
        # ⚠ **来源签名要接着传**（用户 2026-09-28 ✓）：重算的还是**同一条路线的同一件事**
        #   ⇒ 不传的话 `start_route` 会把它清成 `None` ⇒ "追击"这一步**当场失去来源**，
        #   那两个节点就再也复核不了 ✗（人走得越久越容易被重算 ⇒ 越容易丢 ✓ 很隐蔽 ✗）。
        #   ⚠ 传的是**当前那份**（`start_route` 里先读再写 ⇒ 同一个值原样回来 ✓ 不会自噬 ✓）。
        self.start_route(jobs, why="位置变了（现在在「%s」）⇒ 重算去「%s」的最优路径"
                                  % (where, dst), origin=self._route_origin)
        return True

    def _mark_route_change(self, job):
        """把**任务内部的变化**（相切换 / 斜跳 / 失败重来）打进 `behavior.log`（用户
        2026-09-28 要求："寻路的各类变化也做进行为打点，方便排查"）。

        和已有的 `task_start/step/done/fail`（**任务层级**：走到第几步、整条成没成）互补：
        那些说"**这一步**起跑 / 收工"，这里说"**这一步里面**发生了什么" —— 排查"卡在哪、
        为什么断"时，光看 `task_fail` 只能看到最后一句，中间"对齐→上绳→爬不动"的过程只有
        靠这里的 `route_phase` 才拼得起来 ✓。

        四件事（都**只在变化那一刻**记一条，不是每拍 ✓）：
          · `route_phase`：任何任务的 `phase` 变了（align→climb、climb→done、walk→done…）；
          · `route_diag`  ：爬绳的**斜跳**起跳 / 结束（排查"斜跳飞行会卡"就靠它 ✓）；
          · `route_retry` ：一次尝试**失败重来**（`attempt` 涨了 —— 知道它重试了几次 ✓）；
          · `walk_hop`    ：walk 的**单点跳**（走不动后跳一下跳过小台阶，`_hop_at` 置上 ✓）。

        ⚠ 打点专用状态挂在 `self` 上（`_bc_*`），**惰性初始化**：任务对象一换（`start_climb`
        挂了新任务）就重置基线，不把上一个任务的状态带过来 ✗。
        ⚠ 只调 `getattr` 兜底（走 / 跳任务没有 `_diag_flying` / `attempt` ⇒ 当 `None`/False），
        别因为字段名不同就把打点弄崩 ✗（打点不许搞挂主流程，同 `core/behavior` 的纪律 ✓）。
        """
        if getattr(self, "_bc_job", None) is not job:
            self._bc_job = job
            self._bc_phase = None
            self._bc_diag = False
            self._bc_attempt = None
            self._bc_hop_at = None
        _lad = str(getattr(job, "ladder_id", "") or "")
        _note = str(getattr(job, "note", "") or "")[:160]
        phase = getattr(job, "phase", None)
        # ② 斜跳（`_diag_flying` 从 False→True = 起跳；True→False = 结束 ✓）
        diag = bool(getattr(job, "_diag_flying", False))
        if diag != self._bc_diag:
            self._bc_diag = diag
            if diag:
                behavior.event("route_diag", ladder=_lad, what="起跳", note=_note)
            else:
                _climb_ph = getattr(job, "CLIMB", None)
                _sucked = (_climb_ph is not None and phase == _climb_ph)
                behavior.event("route_diag", ladder=_lad,
                               what=("结束·吸上绳" if _sucked else "结束·没吸上"), note=_note)
        # ① 相切换（第一拍不算"变化"：挂上来就是那个相 ✓）
        if phase is not None and phase != self._bc_phase:
            _prev = self._bc_phase
            self._bc_phase = phase
            if _prev is not None:
                behavior.event("route_phase", ladder=_lad, prev=_prev, to=phase,
                               note=_note)
        # ③ 失败重来（`attempt` 涨了 ✓）
        attempt = getattr(job, "attempt", None)
        if (attempt is not None and self._bc_attempt is not None
                and attempt > self._bc_attempt):
            behavior.event("route_retry", ladder=_lad, attempt=int(attempt), note=_note)
        self._bc_attempt = attempt
        # ④ walk 的「单点跳」（走不动后跳一下，`_hop_at` 从 None → 非 None ✓）
        _hop = getattr(job, "_hop_at", None)
        if _hop != getattr(self, "_bc_hop_at", None):
            self._bc_hop_at = _hop
            if _hop is not None:
                behavior.event("walk_hop", note=_note)

    def _fire_timeout_key(self):
        """「**寻路超时后按键**」（用户 2026-09-28）：寻路**超时切断**的那一下，额外点按它 ✓。

        · `None`（默认 ✓）⇒ **什么都不做** ✓（老项目行为一点不变 ✓）；
        · 键名来自设置（「设置 → 判定参数」那个下拉 ✓ —— 里面是**固定键 + 自定义按键** ✓）；
        · **点按**（`decision.input.tap` ✓ 按下 → 松开 ✓）—— 为什么不是按住：按住会变成
          一个**持续输入**、糊住后面每一拍的判断 ✗（超时只是个"给游戏一个信号"的动作 ✓）；
        · 顺手留一条 `behavior.event("goto_timeout_key")` ✓：下次翻日志能知道它**到底发没发**
          （没这条就只能猜 ✗ —— 和 `drop_taps` 同一个理由 ✓）。
        ⚠⚠ **键名必须先解析成真实物理键**（2026-09-28 现场修 ✗）：下拉里**自定义按键存的是
          「名字」**（`settings.custom_keys` 的 key ✓），而 `input.tap` 走 `resolve_vk`、
          只认**固定键名义 / 物理键名** ✗ ⇒ 解析不出时 `tap` 里是 `if vk is None: return`
          —— **静默返回、一个键都不发** ✗（现象 = "配了、日志也有痕迹、游戏里毫无反应" ✗）。
          ⇒ 用现成的 `_resolve_seq_key`（行为序列同一套口径 ✓）。
        """
        key = getattr(self.settings, "goto_timeout_key", None)
        if not key:
            return
        # ⭐⭐ **先解析成真实物理键**（2026-09-28 现场修 ✗ —— 用户报"**寻路超时没有触发我配的
        #   「寻路超时后按键」**"✓，而 log 里 `goto_timeout_key key=表情_无语` 明明有 **27 条** ✓
        #   ⇒ **不是没触发，是键没发出去** ✗）：
        #   ⚠ 下拉里**自定义按键存的是「名字」**（`settings.custom_keys` 的 key ✓）；而
        #     `input.tap(name)` 走 `resolve_vk(name)` —— 只认**固定键名义 / 物理键名** ✗
        #     ⇒ 解析不出时 `tap` 内部是 `if vk is None: return`，**静默返回、一个键都不发** ✗
        #     ⇒ 表现正是"界面上配了、日志也有痕迹、游戏里毫无反应" ✗。
        #   ⚠ **功能键同样得解析**（实测 `attack → keymap 里的 ctrl` ✓）⇒ 不解析的话，连
        #     "攻击 / 跳跃"这些固定项都发不出去 ✗（这个坑比自定义键更宽 ✓）。
        #   ⇒ 用项目里**现成的**那一层：`_resolve_seq_key`（行为序列走的也是它 ✓ ——
        #     先查 `keymap`、再查 `custom_keys`、都没有就原样返回 ✓）。**一处口径，别另写一套** ✗。
        phys = str(self._resolve_seq_key(str(key)))
        # ⚠ **留痕要写在 `tap()` 之前**（2026-09-28 踩过 ✗）：发键走输入层，而它在测试 / 离线
        #   环境里可能是**替身**（`Harness._patched()` 会换掉 `tap` ✗）⇒ 那一步会抛 ⇒ 把留痕
        #   写在它后面就会被下面那个 `except` **一起吞掉** ✗（用例 / 界面永远看不到）。
        #   语义上也该如此：**"决定要发这个键"** 就是那条痕 ✓（发失败是输入层的事 ✓）。
        # 记 **三元组** `(配置里的名字, 真正发出的物理键, 时刻)` —— 第 ② 项是排查关键 ✓：
        #   `('表情_无语', '表情_无语', …)` ⇒ **那个自定义键没绑物理键**（去「自定义按键」绑一个 ✓）；
        #   `('attack', 'ctrl', …)` ⇒ 解析成功 ✓。
        self.last_timeout_key = (str(key), phys, time.monotonic())
        try:
            tap(phys)
        except Exception:      # noqa: BLE001 —— 发个键失败不该把「超时处理」本身带崩 ✗
            return
        # 打点带上 `phys`：下次一眼看出"配的是什么名字、真正发的是哪个物理键" ✓
        behavior.event("goto_timeout_key", key=str(key), phys=phys)

    def _climb_recheck(self, ws, now, node):
        """⭐ **在指定节点主动复核一次：有没有更便宜的怪** ⇒ 有就**掐掉这一步**（用户 2026-09-28 ✓）。

        用户原话："我们的寻路任务应该需要**来源签名**，我想在这些节点**主动找一次**看是否有
        **代价更低的怪**（用**当前状态**跟**签名来源**对比），如果有那就**主动掐掉寻路任务**：
        ① **climb 对齐完成后起跳前**；② **寻路超时计时过半**" ✓。

        口径（三个决定都是用户 2026-09-28 现场定的 ✓）：
          · **只有「追击」来源参与**（`_climb_origin["kind"] == "chase"` ✓）—— 「命令前往」/
            「任务队列」/「定点休息」/「回战斗区域」/「到点去哪」一律**豁免** ✓
            （那些是**人手安排 / 策略定的**，被自动掐掉会让人以为按钮没生效 ✗）；
          · **严格更便宜就掐**（`新 < 来源`，不加余量 ✓）—— 代价口径见 `mob_cost_of`
            （= 从我站的集合到怪站的集合的**寻路距离** ✓）；
          · 掐完**顺手把锁定目标换成新怪** ✓ —— ⚠⚠ 否则下一拍的追击逻辑会把**同一个**任务
            又下回来（死循环 ✗）；换掉之后"人正走的那条"自然就不成立了 ✓。

        ⚠⚠ **两边代价都从"现在的位置"现算**（来源怪也**重新算一遍** ✓）——**不许**记"当时的代价" ✗：
          那个数是从**当时站的位置**算的，而人一直在走 ⇒ 跟现在不可比 ✓。
        ⚠ **来源怪不在了 / 算不出代价 ⇒ 什么都不做**（保守 ✓ 宁缺勿错）—— "它死了该换目标"
          是**原有追击链**的事 ✓（那条链本来就会重挑 ✓），不归这里管 ✗。
        ⚠ **每步只调一次**（`start_climb` 里复位 `_recheck_*_done` ✓）：否则每拍都算一轮代价
          （白烧 CPU），而且"严格更便宜"在两只怪代价接近时会**来回掐** ✗。

        返回 True = 真的掐了（调用方别再往下跑这一步 ✓）。
        """
        o = self._climb_origin
        if not o or o.get("kind") != "chase" or self._climb is None or ws is None:
            # ⛔ **只有「追击」来源参与**（用户 2026-09-28 定 ✓）—— 这一条是**用户决定**的落点，
            #   不是"眼下只有追峰会传 origin 所以可以省"✗：**将来**任何一个调用点开始传
            #   `origin`（比如"任务队列"要带它做别的判断 ✓），不能让那些来源**顺手**被这条复核掐掉 ✗。
            return False                      # 不是追击来源 / 手上没任务 ⇒ 豁免 ✓
        src_id = o.get("mob_id")
        if src_id is None:
            return False
        fn = getattr(self, "mob_cost_of", None)
        p = getattr(ws, "player", None)
        if not callable(fn) or p is None or not getattr(p, "here_sets", None):
            return False                      # 算不出代价 ⇒ 不动（宁缺勿错 ✓）
        # 候选走**同一条漏斗**（`_candidates` ✓ = 锁定/追击链的唯一入口 ✓）。
        # ⚠ 它自带缓存（`_zone_only` ✓）⇒ 这里重复调一次**不会**多打点、也不会多算 ✓。
        try:
            px = float(getattr(p, "x", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        lockable = self._candidates(list(getattr(ws, "mobs", None) or []), px,
                                    ws=ws, now=now)

        def _cost(m):
            """这只怪**现在**的代价（算不出给 `None` ✓ 不猜 ✗）。"""
            try:
                info = fn(p, m)
            except Exception:                 # noqa: BLE001 —— 解析器坏了当"算不出" ✓
                return None
            if not info:
                return None
            c = info.get("cost")
            try:
                return None if c is None else float(c)
            except (TypeError, ValueError):
                return None

        src = next((m for m in lockable if getattr(m, "id", None) == src_id), None)
        if src is None:
            return False                      # 来源怪不在了 ⇒ 不判（保守 ✓）
        c_src = _cost(src)
        if c_src is None:
            return False                      # 来源算不出 ⇒ 无从比（不猜 ✓）
        best, c_best = None, None
        for m in lockable:
            if getattr(m, "id", None) == src_id:
                continue
            c = _cost(m)
            if c is None:
                continue
            if c_best is None or c < c_best:
                best, c_best = m, c
        # ⭐ **严格更便宜**（用户 2026-09-28 定的口径 ✓ —— 不加余量）
        if best is None or not (c_best < c_src):
            return False
        # ---- 掐 + 换目标 ----
        # ⚠ 顺序：**先换锁定目标、再停**（用户要的第三件 ✓）—— 换完之后"人正走的方向"就不成立了，
        #   下一拍的追击链会为新目标下任务 ✓（它就靠"目标不在我这块平台"触发 ✓）。
        self._target_id = getattr(best, "id", None)
        self._target_until = now + self._random_target_cd()
        # ⚠ 缓存里存的是**上一只怪**的集合 ⇒ 换目标必须作废（见 `_locked_mob_info` 的说明 ✓）
        self._mob_info_cache = None
        behavior.event("climb_recheck_switch", node=str(node), src=src_id,
                       src_cost=round(c_src, 1), mob=self._target_id,
                       cost=round(c_best, 1),
                       dst=str(getattr(self._climb, "dst_set", "") or ""))
        self.stop_route("复核（%s）发现有**更便宜**的怪：来源 #%s 要走 %.0f，"
                        "而 #%s 只要 %.0f ⇒ 改追后者"
                        % (node, src_id, c_src, self._target_id, c_best))
        return True

    def _climb_tick(self, now, wx, keys, ws):
        """跑一拍上绳任务 → True = **这一帧就算完事**（到了 / 失败了）。

        为什么任务优先于追怪、巡逻：那是我**主动下的命令**，不是"看见什么就跟着跑"。
        而攻击范围内有怪时根本到不了这里（tick 里 `if in_range:` 在前面接管）
        ⇒ 「有怪先打、打完继续走」**天然成立**，不用在这儿再写一道仲裁。

        任务一旦挂上就**一路做到位**（对齐 → 按跳 → 判到达），中间不再停一截等谁点开关
        —— 这是 2026-09-26 用户定的：「命令前往」那次点击就是授权。

        ⚠ `wx` 必须是**世界坐标 x**（`ws.player.world_x`），**不是画面坐标**！
        2026-09-26 踩过：这里以前收的是 `ws.player.x`（画面检测框中心），而 `ClimbJob.x`
        是世界坐标 ⇒ `dx` 永远是个大数 ⇒ 角色朝一个方向一直走/来回抖，**无限循环**
        （用户报的"卡在 -300~-380"就是这个）。`py` 从头就在用 `world_y` —— 一个画面
        一个世界，本身就是自相矛盾的。

        另外两处也是同一天修的：
          · `hold_ms` = 设置里的对齐保持时间 **+ 当前端到端延迟**（`ws.e2e_ms`）——
            用户要求 3：定位读数是"过去某一刻"的位置，延迟越大越不能拿单帧当真；
          · `ladder_id` / `here_sets` 现在由感知层写进 `Player`（**以前没有任何地方写**，
            于是"到了"永远判不出来、"掉下绳"每 2 秒误判一次 ⇒ 任务不停失败重试）。
        """
        job = self._climb
        p = ws.player
        # ⭐ 用户 2026-09-27 报的卡死（截图：`当前任务 前往：顶层` + `fh：顶层`，人却卡在
        #   "绳底那块面在 y=-182~-182（现在 y=-366）⇒ 先走到绳下端那一层的平台上"上）：
        #   **这趟要去的地方，人已经站在里面了** ⇒ 整条路线**收工** ✓。
        # 为什么必须在**路线**这一层判、不能塞进 `ClimbJob`（试过，被现有用例当场按住 ✗）：
        #   爬绳 job 的"到达"是**双判据**（y 也得够到绳上端 ✓，`t_climb_flow_rules` 钉着）；
        #   而"这趟要不要去某个集合"是**路线**的事 —— `plan_jobs` / 集合图本来就是**集合级**
        #   的语义（`mapdata.segment_of` 的 ⚠ 原文："语义判定一律走集合" ✓）。
        # 现场数据（106010105，用户那张图）：路线目标「顶层」，而这条爬边是「三楼 → 顶层」
        #   —— 绳底那块面 #99 属「三楼」、人脚下那块 #74 属「顶层」⇒ 人已在目标里，
        #   再让他"先走到绳底那块面上"就是让他**白跑一趟（还是往下跑）** ✗。
        # ⚠ 只看**最终目的地**（`self._route[-1]`；没有剩余步时就是当前这一步 ✓）——
        #   别拿中间某一站的 `dst_set` 判，那会把整条路线提前掐掉 ✗。
        goal = str(getattr(self._route[-1] if self._route else job, "dst_set", "") or "")
        here = set(str(s) for s in (getattr(p, "here_sets", None) or ()))
        # ⛔ **人正挂在绳上爬的时候，不许拿"脚下集合"判"这趟已经到了"** ✗
        #   （用户 2026-09-27 报：屏上写着 `绳梯：L3　到顶：否`，攀爬却**擅自判成功** ✗ ——
        #    看截图那句"人已经站在「一楼」里了 ⇒ 收工"，就是**下面这条**判的 ✓）。
        #   为什么只有爬绳时会撞上：**绳段横跨好几层**，而"脚下属于哪块集合"取的是
        #   黄点正下方那条 foothold —— 挂在绳上时它可能落在**别层**的平台上 ✗
        #   （同一批数据就是当天那道"绳底那块面属三楼、人脚下属顶层" ✓）。
        #   ⇒ 判据：**人不在绳上**（`p.ladder_id` 为空 = 广播说没贴着绳 ✓）才允许用
        #     "脚下集合"收工 ✓；在绳上时**一律等广播说到达**
        #     （上爬 `at_ladder_top` ✓／下爬 `at_ladder_bottom` + `ground_y` ✓）——
        #     那条链一个字没动 ✓，爬不完就由它自己的"爬不动 / 寻路超时"如实报错 ✓。
        _on_rope_now = bool(PosSnapshot.of(p).ladder_id)     # 广播说"正贴着绳" ✓
        # ⭐ 诉求 2（用户 2026-09-28）：climb 执行器正处在「到顶后再按住 ↑ 那一下」阶段时，
        #   **不许**用"脚下已经是目的地"提前收工 ✗ —— 人刚迈上平台（`ladder_id` 空 + `here_sets`
        #   变目的地）就被这条截断，"再按住 ↑ `base_hold_ms`"那一下白按了（用户观察到的
        #   "位置状态切为目的地后立马停了"就是它 ✓）。让 `holding_up` 那几拍走完再说 ✓。
        _hold_up = bool(getattr(job, "holding_up", None) and job.holding_up())
        if goal and here and goal in here and not _on_rope_now and not _hold_up:
            behavior.event("route_goal_reached", goal=goal)
            self._task_finished(
                failed=False,
                why="人已经站在「%s」里了（这就是这趟要去的地方）⇒ 收工 ✓" % goal)
            return True
        # ---- **盖章：这一拍任务跑了**（`_task_settle` 就靠它判断要不要报信）----
        # 这里是**唯一**推任务的地方 ⇒ 也是唯一该盖章的地方 ✓（2026-09-26 结构性收口）。
        # `_task_ran_at` 同时是"暂停起点"用的锚（报信发生在下一拍，见 `_task_settle`）✓。
        self._task_ran = True
        self._task_ran_at = now
        # **寻路超时**（用户 2026-09-26 要求，当天又明确了口径）：按**每一段**算 ——
        # **每完成一段**（从一个集合走到另一个集合）就重新计时 ✓。
        # 代码上靠 `start_climb()` 重打 `_climb_started`，而 `_task_finished` 接下一段时
        # 正是调它 ✓（所以"多段路线共用一次超时"是**不会**发生的 —— 用例
        # `t_goto_timeout_per_hop` 钉着，别把它改成整条路线一个钟 ✗）。
        # ⇒ 它挡的是"**某一段**卡住 / 在一段里反复重试"，不是"整条路线太久"：
        #    5 段各自都正常的路线，总耗时可以远超这个值，那**不算超时** ✓。
        # ⭐⭐ **「追怪寻路.duration(s)」**（用户 2026-09-28 要求 ✓ 原话："表示本次任务如果是
        #   追怪下达的，那么如果任务超过了这个时间就结束任务"）。
        #   · **只对"追怪下达的"任务生效**：判据 = `_climb_origin` 的 `kind == "chase"` ✓
        #     （结构化的"这条任务为什么下"，唯一写口 `_chase_origin` ✓）——
        #     「命令前往」/「定点休息」/「回战斗区域」等**一律不参与** ✗（它们想跑多久跑多久 ✓）；
        #   · **与「寻路超时」共用同一把钟**（`_climb_started` ✓ 同一步重下不重置 ✓）⇒
        #     绝不会出现"一个从开始算、一个从别处算"那种两把尺 ✗；
        #   · `chase_goto_max_s <= 0`（**默认**）= **不启用** ✓（老行为一字不变 ✓）；
        #   · ⚠ 放在"寻路超时"**之前**：追怪上限通常**更短**（比如 15 秒 vs 30 秒 ✓）
        #     ⇒ 让"为了追一只怪而超时"单独报出来，而不是被笼统的「寻路超时」盖掉 ✓。
        _chase_cap = max(0.0, float(
            getattr(self.settings, "chase_goto_max_s", 0.0) or 0.0))
        if (_chase_cap > 0 and self._climb_started
                and (now - self._climb_started) > _chase_cap
                and self.chase_tagged()):
            job.cancel("追怪寻路超时：这趟追击已经跑了 %.0f 秒（上限 %.0f 秒）"
                       "—— 别为了追一只怪把时间都耗在这上面"
                       % (now - self._climb_started, _chase_cap))
            behavior.event("chase_goto_timeout",
                           secs=round(now - self._climb_started, 1),
                           cap=round(_chase_cap, 1))
            return self._task_finished(failed=True)
        # ⭐⭐ **「寻路超时」只对「追击」签注生效**（用户 2026-09-29 第 1 条 ✓ 原话："只有追击
        #   签注的任务是有时间限制的（到点结束任务），其他任务不应该有时间限制（无限）"）：
        #   · 判据 = `chase_tagged()`（结构化 `kind == "chase"` ✓）—— 和 `goto_time_left`
        #     显示的剩余**同一处口径** ✓（显示说"无限"、闸却还在掐 ⇒ 最坏的那种不一致 ✗）；
        #   · 为什么其它任务该无限：一条长路线（走→爬→走）跑两分钟很正常，30 秒的闸
        #     会把**正常**的路线当卡住切掉 ✗；而"追不上就别追了"是追击**独有**的语义 ✓。
        #   ⚠ 各执行器自己的"卡住重试/放弃"（`stall_s` / `reassert_s` / `retry_ms` ✓）
        #     **不受影响** ⇒ "真卡住"仍然各自处理 ✓（这一条只管**总时长**那把闸 ✓）。
        cap_s = (max(0.0, float(getattr(self.settings, "goto_timeout_s", 0.0) or 0.0))
                 if self.chase_tagged() else 0.0)
        # ⭐ **节点②「寻路超时计时过半」**（用户 2026-09-28 要求 ✓）—— 放在超时判定**之前** ✓。
        #   意义：这一段已经花掉一半预算 ⇒ 主动复核一次"还值不值得往下走"（有更便宜的怪就掐 ✓）；
        #   过了这个节点就只剩"被超时硬切"了 ⇒ 这是最后一次自己拿主意的机会 ✓。
        #   ⚠ **每步只查一次**（`_recheck_half_done` ✓）：过半之后每一拍都过半 ⇒
        #     不限制的话就是"每拍算一轮代价" ✗。
        #   ⚠ 与超时**共用同一把钟** `_climb_started` / `cap_s`（一处口径 ✓ 见 `start_climb` 那段 ✓）。
        if (cap_s > 0 and self._climb_started and not self._recheck_half_done
                and (now - self._climb_started) > cap_s * 0.5):
            self._recheck_half_done = True
            if self._climb_recheck(ws, now, "超时过半"):
                return True
        if cap_s > 0 and self._climb_started and (now - self._climb_started) > cap_s:
            job.cancel("寻路超时：这一段已经跑了 %.0f 秒（上限 %.0f 秒）"
                       "—— 先看它卡在哪一步" % (now - self._climb_started, cap_s))
            behavior.event("goto_timeout", secs=round(now - self._climb_started, 1),
                           cap=round(cap_s, 1))
            # ⭐ 「**寻路超时后按键**」（用户 2026-09-28）：超时切断的**这一下**额外点按它 ✓
            #   （`None` = 一个键都不发 ✓ 见 `_fire_timeout_key` ✓）。放在 `_task_finished`
            #   之前：切断动作先做完，再收拾任务状态 ✓。
            self._fire_timeout_key()
            return self._task_finished(failed=True)
        # 任务被清掉之后，界面还要能回答"刚才为什么松开了 ↑" ⇒ 每一拍把最新那句留一份
        #（`current_goto_note` 在任务已结束时也能给出它，见那里的说明）。
        self._last_goto_note = (str(getattr(job, "note", "") or ""), now)
        if wx is None:
            # 定不了位（小地图那条没跑 / 没认出黄点）⇒ **如实失败**，绝不拿画面坐标硬凑
            #（那正是上面那个死循环的成因）。这里直接放弃：坐标拿不到不是"再试一次"能好的。
            behavior.event("climb_giveup", why="拿不到世界坐标（小地图定位没输出）")
            job.cancel("拿不到世界坐标（小地图定位没有输出）—— 先确认「实时」页在跑、"
                       "小地图那块能认出黄点")
            return self._task_finished(failed=True)
        # ---- **位置状态变了 ⇒ 重算剩下的路**（用户 2026-09-27）----
        # 位置挑过：① 在"拿不到坐标"那道**之后**（连自己在哪都不知道就没法重算 ✓）；
        # ② 在下面 `job.update` **之前**（这一拍就按新计划走，不空转一拍 ✓）；
        # ③ 在「寻路超时」那道总闸**之后**（真卡住了该如实超时，不许用重算把它盖掉 ✗）。
        if self._maybe_replan(p):
            job = self._climb          # 重算会换掉当前这一步（`start_route` → `start_climb` ✓）
            if job is None:
                return False           # 重算之后手上没活儿（不该发生；真发生就等下一拍 ✓）
            self._last_goto_note = (str(getattr(job, "note", "") or ""), now)
        py = getattr(p, "world_y", None)
        # 打点：上绳期间把 y 曲线留进 **behavior.log**（用户 2026-09-28："角色状态"归这里 ✓）——
        # "y 不动"到底是"人真的没爬"还是"读数没动"，只有曲线能说话（2026-09-26 用户报
        # "每次都会触发补按"✗）。
        # ⚠ **连续量必须给 `min_gap`**（用户 2026-09-28："改成定频采样"✓）：`climb_y` /
        #   `climb_dx` **每拍都在变** ⇒ 光靠"值变了就记"等于不节流 ✗（实测把日志刷到
        #   760 KB）⇒ 按 **1 秒一个点**记，看趋势够用 ✓。
        #   而 `climb_y_held` / `climb_phase` / `climb_on_ladder` 是**离散状态**（0/1/2）
        #   ⇒ 保持默认（**值变了立刻记** ✓，跳变最要紧）✓。
        if py is not None:
            behavior.sample("climb_y", round(float(py), 1), min_gap=1.0)
            behavior.sample("climb_y_held", 1 if getattr(p, "world_held", False) else 0)
            # "y 冻住"只可能是三件事：人在绳上不动 / 根本没上绳 / 相位还没到上绳。
            # 这三个 sample 就是用来分开它们的（2026-09-26 用户报"每次都会触发补按"✓）：
            #   climb_phase    1 = 已经在"按跳+方向"那一相（0 = 还在对齐）；
            #                  **2 = 下爬的「跳下绳梯」那一相**（2026-09-26 加：松 ↓、
            #                  按跳离开绳、按住方向等落地）—— 它和 1 必须分开看，
            #                  否则"已经松手跳下去了"和"还挂在绳上"在日志里长得一样 ✗
            #   climb_on_ladder 1 = **这一拍任务认为自己在绳上**（`ladder_id` 对上了 ✓）
            #   climb_dx       离绳的 x 有多远（对齐对不对；跳不上去时它是关键 ✓）
            # ⚠ `CLIMB` / `JUMP_DOWN` 只有上绳任务有（走 / 下跳的相位名不一样）⇒ 必须
            #    `getattr` 兜底：直接写 `job.CLIMB` 会让走那一步当场 AttributeError ✗
            #    （写那条时踩过一次 ✓；`JUMP_DOWN` 同理 —— 下跳任务没有这个相）。
            _ph_climb = getattr(job, "CLIMB", None)
            _ph_jump_dn = getattr(job, "JUMP_DOWN", None)
            behavior.sample("climb_phase", (
                1.0 if (_ph_climb is not None and job.phase == _ph_climb)
                else (2.0 if (_ph_jump_dn is not None
                              and job.phase == _ph_jump_dn) else 0.0)))
            # ⚠ **走 / 跳 / 下跳没有 `ladder_id`** ⇒ 原来那个写法对它们**恒 0** ✗（2026-09-28
            #   现场：下跳卡住时 `climb_on_ladder` 一路 0，看着像"位置状态压根没判绳梯"，
            #   其实是**探针不适用** ✗ —— 用户当场问的就是这个 ✓）。⇒ 没有 `ladder_id` 的
            #   执行器改看**只看位置**那份 `on_rope_pos`："这一拍人的位置**落进某根绳的绳段**
            #   了吗" ✓（与按键许可无关 ✓ —— 正是下跳要判的那件事：**被绳吸住 = 位置进绳段 +
            #   脚下没面** ✓；它为空就说明"人根本不在绳段里"，别再怀疑许可 ✗）。
            _lid = getattr(job, "ladder_id", None)
            if _lid:
                _on = 1 if getattr(p, "ladder_id", None) == _lid else 0
            else:
                _on = 1 if getattr(p, "on_rope_pos", None) else 0
            behavior.sample("climb_on_ladder", _on)
            _jx = getattr(job, "x", None)
            if _jx is not None:
                behavior.sample("climb_dx", round(abs(float(wx) - float(_jx)), 1),
                                min_gap=1.0)     # 连续量 ⇒ 定频（1 秒一个点 ✓）
            # `drop_phase` / `drop_dx`：**下跳任务（`DropJob`）专属**（2026-09-28 加 ✓）——
            # 上面那三个探针只认上绳任务（`climb_phase` 对下跳任务**恒 0** ✗），而 2026-09-28
            # 现场卡住的恰恰是下跳：近 9 秒里日志只剩一排 `climb_phase=0.0`，
            # **看不出卡在"对齐 x / 按住↓ / 脱离绳"哪一相** ✗（用户当场只能报"实际爬上了
            # 向下爬的绳子"）。有了它，下一次一眼就能定位 ✓。
            #   drop_phase：1=对齐 2=按住↓ 3=按跳 4=等落地 5=脱离（**离散 ⇒ 值变就记** ✓）
            #   drop_dx   ：离"这一轮挑中的下跳点中心"还差多少 px（**连续 ⇒ 1 秒一个点** ✓）
            _pc = getattr(job, "phase_code", None)
            if callable(_pc):
                behavior.sample("drop_phase", float(_pc()))
                _at = getattr(job, "align_target_x", None)
                _ax = _at() if callable(_at) else None
                if _ax is not None:
                    behavior.sample("drop_dx", round(abs(float(wx) - _ax), 1),
                                    min_gap=1.0)
                # `drop_taps` / `drop_round`：**跳到底发了没有、轮次有没有在转**（2026-09-28 加 ✓）
                # —— 用户当天报"**趴在那不动了也不按跳**"，而 `drop_phase=3` 只说"在 DROP 相"，
                #   **分不清**"跳压根没发"和"跳发了、游戏不响应" ✗（那两件事的修法完全不同）。
                #   drop_taps ：点按跳的**上升沿**累计数（**连续 ⇒ 1 秒一个点** ✓，数字在涨就是在发 ✓）
                #   drop_round：第几轮"按住 ↓ + 连按跳"（**离散 ⇒ 值变就记** ✓，转不转一目了然 ✓）
                _tc = getattr(job, "tap_count", None)
                if callable(_tc):
                    behavior.sample("drop_taps", float(_tc()), min_gap=1.0)
                # ⭐ `drop_reassert`：**补按「按住 ↓」发了几次**（2026-09-28 加 ✓）——
                #   与上面 `drop_taps` **并排看**：跳的计数在涨、它**不涨** ⇒ 就是
                #   "**只有跳在补、↓ 没补**"那个形状 ✓（用户当天报的 bug 原来在日志里
                #   **看不出来**：`act keys=down` 只报"键集"，报不出"发了几次 PRESS" ✗）。
                #   ⚠ `getattr` 兜底：`ClimbJob` 没这个属性 ⇒ 跳过 ✓（别硬塞 ✗）。
                _rc = getattr(job, "reassert_count", None)
                if _rc is not None:
                    behavior.sample("drop_reassert", float(_rc()), min_gap=1.0)
                _rn = getattr(job, "round_n", None)
                if callable(_rn):
                    behavior.sample("drop_round", float(_rn()))
        # 保持窗口 = **任务自己的**保持时间 + 当前端到端延迟（用户要求 3）。
        # ⚠ 别拿 `settings.align_hold_ms` 覆盖它：任务是「命令前往」那一刻按设置建的
        #（`route_panel._command_first_step` 传的就是它），覆盖会把"任务自己说了算"
        # 的语义弄丢（自检里 `hold_ms=0` 的用例当场变成 250ms，一等就红）。
        # 延迟每拍都在变，所以在**基线**上叠，基线只记一次（否则会一层层累加）。
        # ⚠ 走（`WalkJob`）**没有"保持窗口"这回事**（走到就算到，不用再按住一会儿）
        # ⇒ 用 `getattr` 兜底：没有 `hold_ms` 的任务直接跳过这一段，别硬塞一个给它。
        if getattr(job, "hold_ms", None) is not None:
            if getattr(job, "base_hold_ms", None) is None:
                job.base_hold_ms = int(job.hold_ms)
            job.hold_ms = max(0, int(job.base_hold_ms)
                              + int(round(float(getattr(ws, "e2e_ms", 0.0) or 0.0))))
        # ⭐ **到达判据**全走**位置状态广播**（用户 2026-09-27 ✓）：
        #   · `at_ladder_top`（上爬：到没到绳**上端** ✓）；
        #   · `at_ladder_bottom`（下爬：到没到绳**下端** ✓，与上端对称 ✓）；
        #   · `ground_y`（下爬：**脚下那块面的 y** ⇒ 与本任务的目标平台面比 = "落到目标
        #     平台的面了" ✓）；
        #   · `on_rope_pos`（**只看位置**编出来的绳号，与按键许可无关 ✓ —— 执行器自己会松键，
        #     拿带许可的 `ladder_id` 会"自己把自己判成没上绳" ✗）。
        # 三态原样透传：`None` = 判不出来（**没有几何兜底了** ✗，用户 2026-09-27 明确
        #   "删掉过渡兜底" ✓）、`""` = 判过了没到（**不许自己比坐标** ✗）、`"L2"` = 已到
        #   那根绳的那一端 ✓。口径在 `perception/pos_state.py` ✓。
        # ⚠ 后三个**只有爬绳任务**（`ClimbJob`）的 `update` 认：走 / 下跳 / 跳那三个的签名里
        #   没有它们 ⇒ 一律硬传会当场 `TypeError` ✗（2026-09-27 实测：一批 `WalkJob` 用例
        #   全炸 ✓）。**别为了"统一"去给那三个也加上**（它们根本不用 ✗）。
        # ⭐ **节点①「对齐完成、起跳前」**（用户 2026-09-28 要求 ✓）—— 故意放在 `job.update`
        #   **之前**：那才是"起跳前"真的拦得住的位置 ✓。
        #   · `DropJob`：进 `ARMED` 相的那一刻记下 `_armed_at`，而 `ARM_HOLD_S` 之内**还没按跳**
        #     （按跳在 `DROP` 相 ✓）⇒ 下一次进到这里时人**还没跳** ✓ **真的赶在起跳前** ✓；
        #   · `ClimbJob`：它的"对齐完成"与"按跳"在**同一拍之内**（见 `route.py` 那段说明 ✓）
        #     ⇒ 这里最早只能在**下一拍**看到 ⇒ 那一下跳已经发了 ⚠（如实说：这里能拦的是
        #     "还没真爬上去"，不是"跳之前"✗ —— 真正能起跳前拦住的是 `DropJob` ✓）。
        #   ⚠ 判据只认**已对齐过**（`_armed_at` 非 `None` ✓）+ **这一步还没查过**
        #     （`_recheck_armed_done` ✓）：复查一次就够 —— 不是"每次回到对齐都查" ✗
        #     （那会变成每轮都算一遍代价，而且"严格更便宜"在代价接近时会来回掐 ✗）。
        if (getattr(job, "_armed_at", None) is not None
                and not self._recheck_armed_done):
            self._recheck_armed_done = True
            if self._climb_recheck(ws, now, "对齐完成"):
                return True
        # ⭐ **给下跳任务注入"锁定目标在哪边"**（用户 2026-09-28 按流程图 ✓ 原话："跳下绳子：
        #   按任意方向+跳（**若有锁定目标则是锁定目标方向**）"）。
        #   ⚠ 只挂给**下跳任务**（`_detach_dir` 是 `DropJob` 独有 ⇒ 用它当判据，**不必为此在
        #     热路径 import** ✗）；⚠ **每拍重挂**（锁定目标会换 ⇒ 必须**现读**，不许建任务时
        #     定死 ✗）；拿不到目标 ⇒ `_target_dir` 给 `0` ⇒ 调用方退回"任意方向" ✓。
        if hasattr(job, "_detach_dir"):
            job._target_dir_fn = lambda: self._target_dir(ws)
        out = job.update(now, float(wx), py=py,
                         **PosSnapshot.of(p).as_kwargs())   # 快照 → 参数（**唯一一处映射** ✓）
        # ⭐⭐ **这一轮"按过跳"记一笔**（用户 2026-09-28 ✓ 原话："失败那段改成：**按跳后**挨过打
        #   ⇒ 用延迟"）—— `route` 侧**不用改** ✗：`out["jump"]` 就是"这一拍按了跳"✓（点按时
        #   会连着几拍为真 ⇒ 记最后一次就够 ✓）。
        if out.get("jump"):
            self._climb_jump_at = float(now)
        # 任务内部的变化（相切换 / 斜跳 / 失败重来）打进 behavior.log（用户 2026-09-28 要求 ✓）
        self._mark_route_change(job)
        km = self.settings.keymap
        if out["move"]:
            # 方向交给 `_steer`（它会 `set_facing`，别名/转身都走同一套）
            self._steer(float(out["move"]), keys)
        # 方向键：**不按跳时也要按**（下跳要求"先按住 ↓，再按跳"，见 DropJob.ARMED）
        if out["dir"]:
            dk = km.get("up") if out["dir"] > 0 else km.get("down")
            if dk:
                keys.add(dk)
        if out["jump"]:
            jk = km.get("jump")
            if jk:
                keys.add(jk)              # **按住**：KeyState.set 会一直按着
        if out.get("reassert"):
            # **补按**（用户 2026-09-26 方案 B）：`KeyState` 只在键集**变化**时才发键 ⇒
            # 对面把键丢了（中继重连 / 串口抖动）本机**不知道** ✗ ⇒ `keys.set` 以为还按着
            # ⇒ 一路都不再发 ⇒ 人停在绳上不动、任务却自信地按着 ↑ ✗。
            # 这里直接对当前几个键**重发一次 PRESS**（固件 `addHeld` 会去重、本地后端幂等 ✓）。
            for _k in sorted(keys):
                try:
                    key_down(_k)
                except Exception:             # noqa: BLE001
                    pass
            # ⚠ 计数名**不能**叫 `key_reassert` —— 那是"清空指令通道后重按"那条路的
            #    名字（`_reassert_ctx` ✓）；撞在一起就分不清是谁触发的 ✗（实测踩过 ✓）。
            # 把**当时那句 note**一起记下来（里面有卡在哪个 y、离目标面还差多少 ✓）：
            # 事后一眼就能分辨是"人没爬"还是"读数没动"（见 `climb_y` 那条 sample ✓）。
            behavior.event("climb_reassert", y=py,
                           note=str(getattr(self._climb, "note", ""))[:160])
        if out["failed"]:
            # ① **失败之后，先问广播"是不是其实已经到了"**（用户 2026-09-27 第 1 条：
            #    "位置状态变化即广播，且**绝对权威**" ✓）。
            #    ⛔ 原来这条**只**看"脚下集合里有 `dst_set`"就改判**成功** ✗ ——
            #       而**爬绳时那个读数根本不可信**：绳段横跨好几层，黄点正下方那条
            #       foothold 可能是**别层**的（同一批数据就是当天那道"绳底那块面属三楼、
            #       人脚下属顶层" ✗）⇒ **位置状态还没广播到顶，执行器就擅自结束了** ✗
            #       （用户 2026-09-27 亲测：三楼 → 顶层途中又发生一次 ✗）。
            #    ⇒ 只有下面两种**权威**情形才算"到了"：
            #      · 广播说**已到本任务这根绳的上端**（`at_ladder_top` == 本任务的绳号 ✓）；或
            #      · **人已经不在绳上**（`ladder_id` 空 = 广播说没贴着绳 ✓）**且**脚下
            #        确实是目标集合 ✓ —— 这条才是老兜底的本意（"到了之后又被打下来 /
            #        定位先抖了一下"那种 ✓，那时人已经离开绳了 ✓）。
            #    ⚠ 在绳上时**绝不**用集合改判成功 ✗（那是这一轮报的那个 bug ✓）。
            _pos_f = PosSnapshot.of(p)                            # 这一拍的广播快照 ✓
            _lid_now = str(_pos_f.ladder_id or "")
            if _pos_f.at_rope_end(getattr(job, "ladder_id", ""), top=True):
                # 单独记一笔：和"几何/集合"那些分开，日志里一眼能看出是哪种结束 ✓
                behavior.event("climb_done_top",
                               ladder=getattr(job, "ladder_id", ""))
                return self._task_finished(failed=False)
            here = set(getattr(p, "here_sets", None) or ())
            if (not _lid_now) and job.dst_set and job.dst_set in here:
                # 单独记一笔：**它是"失败后别再重试"的兜底，不是到达判据** ——
                # 日志里要和几何到达（climb_done）分开，否则又看不出是哪种结束的 ✗。
                behavior.event("climb_done_set", ladder=getattr(job, "ladder_id", ""),
                               set=job.dst_set)
                # 人已经不在绳上、又在目标集合里 ⇒ 这一步算**到了**（多步路径继续往下走）
                return self._task_finished(failed=False)
            # ② **失败保护**：失败不是终止 —— 重新激活再来一次（上绳 / 下跳本来就容易歪
            #    一下：偏离绳 / 掉下来 / 点按跳没生效都算）。但要**延迟**激活（要求 1）：
            #    等待期间**不按键**（这一拍只返回 False ⇒ 状态还是 climb、keys 空
            #    ⇒ 角色站着），到点才重新对齐。
            # ⚠ **不设"最多试几次"**（用户 2026-09-27："我好像在寻路任务时看到一个 1/3 尝试的
            #   显示……应该需要去掉，我们现在都走「寻路超时时间」"）⇒ 这里**没有**"到上限就
            #   放弃"那条分支了：失败就一直原地重来，由上面那道「寻路超时时间」总闸收口
            #   （到点 `job.cancel("寻路超时：…")` + **整条路线**停掉 ✓）。
            #   重试之间的等待 = 基础 + 增量×(已试次数-1)（见 `retry_delay_s` ✓）——
            #   它只影响节奏，不影响"能不能放弃" ✓。
            _base = getattr(self.settings, "climb_retry_delay_s", 0.0) or 0.0
            _inc = getattr(self.settings, "climb_retry_delay_inc_s", 0.0) or 0.0
            # ⭐⭐ **延迟只给"按跳后挨过打"那一种失败**（用户 2026-09-28 ✓ 原话："失败那段改成：
            #   **按跳后**挨过打 ⇒ 用延迟"／"注意攀爬失败后延迟激活的时间**只对『因喝药而失败』
            #   生效**"✓）。
            #   · **按跳后挨过打**（`_climb_hit_at >= _climb_jump_at`、且两个都非 0 ✓）⇒ 用原来的
            #     延迟（基础 + 增量×已试次数 ✓ —— 挨打之后确实要缓一缓再上 ✓）；
            #   · **其余失败**（最典型 = **上绳梯落回去 = 没跳准** ✓）⇒ **`delay = 0` ⇒ 立刻重来** ✓。
            #   ⚠ 这正是用户报的"**落回去失败后发呆**"✗ 的成因：老代码（3094 那段注释"失败不是
            #     终止…**但要延迟激活**"✗）把**所有失败**都套了那段延迟，**不看原因** ✗，
            #     而且**越失败等越久**（1s / 2s / 3s…✗）⇒ 等待期间"这一拍不按键"⇒ 角色站着 = 发呆 ✓。
            _hit_after_jump = (self._climb_hit_at > 0.0 and self._climb_jump_at > 0.0
                               and self._climb_hit_at >= self._climb_jump_at)
            delay = retry_delay_s(_base, _inc, job.attempt) if _hit_after_jump else 0.0
            if delay <= 0.0:
                # ⭐ **打点 `climb_retry_fast`**（用户 2026-09-28 ✓）："没挨打（或被设为不等）⇒
                #   **立刻重来**"这一档 ✓ —— 日志里跟 `climb_retry_wait`（挨打后要等的 ✓）
                #   一眼分清 ✓；带上 `hit` 字段便于分辨是"真没挨打"还是"delay 被设成 0"✓。
                behavior.event("climb_retry_fast", ladder=getattr(job, "ladder_id", ""),
                               attempt=job.attempt, hit=bool(_hit_after_jump))
                job.retry()
                return False             # 这一帧不算完事，下一帧从"重新对齐"接着走
            if self._climb_retry_at is None:
                self._climb_retry_at = now + delay
                behavior.event("climb_retry_wait", ladder=getattr(job, "ladder_id", ""),
                               attempt=job.attempt, delay=round(delay, 1))
                # 把"基础 + 增量"也写出来：不然人只会看到等待变长、不知道为什么 ✓
                # ⚠ 次数**不写分母**（没有上限了 —— 那个「1/3」就是从这里来的 ✗）
                job.note = ("%s（等 %.1fs 后重新对齐：基础 %.1fs + 增量 %.1fs×%d，"
                            "已试 %d 次）" % (
                                job.note, delay, float(_base), float(_inc),
                                max(0, int(job.attempt) - 1), job.attempt))
                return False
            if now < self._climb_retry_at:
                return False         # 还在等：这一拍不按键
            self._climb_retry_at = None
            behavior.event("climb_retry", ladder=getattr(job, "ladder_id", ""),
                           attempt=job.attempt)
            job.retry()
            return False
        if out["done"]:
            behavior.event("climb_done", ladder=getattr(job, "ladder_id", ""),
                           note=str(getattr(job, "note", ""))[:160])
            # 到了 ⇒ 交给统一收口：多步路径要接着走下一步（见 _task_finished）
            return self._task_finished(failed=False)
        return False

    def _resolve_seq_key(self, name):
        """序列里的键名 → 实际物理键名。

        back/forward 按当前朝向解析；其他先查 keymap（功能键 → 物理键）、
        再查自定义按键，都查不到就原样返回（本身是物理键名，如 esc）。
        """
        if name == "back":
            return self.settings.keymap["left"] if self.facing > 0 else self.settings.keymap["right"]
        if name == "forward":
            return self.settings.keymap["right"] if self.facing > 0 else self.settings.keymap["left"]
        v = self.settings.keymap.get(name)
        if v:
            return v
        v = self.settings.custom_keys.get(name)
        return v if v else name

    def _run_seq(self, now, ctx, output=False):
        """执行序列上下文一步，支持「触发后执行」的嵌套子序列。

        ctx = [seq, phase, next_ts, held, sub_stack]；sub_stack 是 then
        子序列的栈（list of ctx），执行时优先栈顶。返回更新后的 ctx；
        序列走完返回 None（完成时释放本层 held 键）。

        output=True 表示这是**输出行为**（攻击输出 / 回身输出）：开跑时会给
        「上次输出时刻」打点，供输出CD（`_next_output_ts`）与输出后锁定防抖用。
        其它序列（进入/退出隐身、自定义定时行为）不算输出行为 —— 它们只有在
        碰巧按到攻击键时才算一次「外部输出」。
        """
        seq, phase, next_ts, held, sub_stack = ctx
        while True:
            # 子栈优先：先执行 then 子序列
            if sub_stack:
                sub = self._run_seq(now, sub_stack[-1])
                if sub is None:
                    sub_stack.pop()   # 子序列跑完：同一帧继续父序列，不白占一帧
                    continue
                sub_stack[-1] = sub
                return ctx
            if now < next_ts:
                return ctx
            if not seq or phase >= len(seq):
                self._release_held_set(held)
                return None
            elem = seq[phase]
            phase += 1
            # **输出行为从这一刻开始计时**：输出CD 记的是「输出行为」之间的间隔，
            # 与这一轮按了哪些键无关。层级是 攻击状态 > 输出行为 > 输出按键 ——
            # 原来只在按下攻击键时打点，序列里没有攻击键（只放技能键 / 只放跳）时
            # 输出CD就完全失效（实测退化成背靠背、只剩序列自身耗时）。
            # 概率没命中的元素也算开始：它确实占用了这一轮。
            if output and phase == 1:
                self._last_output = now
            # 执行几率：默认 100%；未命中则跳过本元素。
            # 跳过的元素不产生任何输入，也就不该占用一个 tick —— 同一帧里接着看
            # 下一个。否则「10% 几率的跳」在 90% 的情况下白吃一个 tick（15fps 下
            # 66.7ms），把输出间隔拉长，而它明明什么都没发。
            prob = elem.get("prob", 100)
            if prob < 100 and random.random() * 100 >= prob:
                continue
            if elem["type"] == "delay":
                next_ts = now + max(0, int(elem.get("ms", 0))) / 1000.0
            else:
                key = self._resolve_seq_key(elem.get("key"))
                if key:
                    if elem["type"] == "down":
                        key_down(key)
                        held.add(key)
                        if output:
                            # 性能口径：从「取到这一帧」到「输出键真的按下去」隔了多久
                            # —— 本机处理 + 排期等待的总和。推理尖刺、排期拖后都会
                            # 直接体现在这里（控制质量的关键量，见 core/perf.py）。
                            perf.since_frame("out_key_ms")
                        # 非输出序列（定时行为之类）按了攻击键，也算一次「外部输出」，
                        # 让战斗输出避开它；输出序列自己已经在开跑时打过点了。
                        if not output and key == self.settings.keymap.get("attack"):
                            self._last_output = now
                    else:
                        key_up(key)
                        held.discard(key)
                # 键动作之间**不再插随机延迟**（2026-09-27 删掉），改成项目里**已有的**
                # 「点按按住时长」（`_attack_duration` = 30ms，`tap()` 那几条路用的也是它
                # ✓）—— 不新造数：间隔就是这个"一下按键"的长度，同一份配置每次都一样 ✓。
                # 想要更长的间隔，就在序列里插一条「额外延迟」（编辑器里那个 ✓）。
                next_ts = now + self._attack_duration
            then = elem.get("then")
            if isinstance(then, list) and then:
                sub_stack.append([then, 0, 0.0, set(), []])
            return [seq, phase, next_ts, held, sub_stack]

    def _release_ctx(self, ctx):
        """释放上下文里所有按下的键（含 then 子序列栈）。"""
        if not ctx:
            return
        _, _, _, held, sub_stack = ctx
        self._release_held_set(held)
        for sub in sub_stack:
            self._release_ctx(sub)

    def _reassert_ctx(self, ctx):
        """清空指令通道之后：把上下文里**该按着的键**重按一遍（**进度和记录都保留**）。

        用户 2026-09-26 定的口径：寻路 / 行为编辑器编的宏都可能很长，中间被"清空指令
        通道"（定期 RELEASEALL、点「重置指令通道」）打断时，固件侧那些键被一次性松掉了
        —— 而本机这边**进度还在**（`next_ts` 还早着呢）⇒ 没人补按 ⇒ 表现就是"按着 ↑
        结果被停了、之后一路都不动" ✗。这里就做那一件小事：把 `held` 里每个键**重新
        PRESS** 一次。（固件 `addHeld` 会去重 ✓，本地后端下重复 PRESS 也是幂等的 ✓。）

        ⚠ 记录**故意保留**（原来这里叫 `_clear_ctx_held`：清掉就算完 ✗）：
          · 清掉 ⇒ 这一轮补按上了，可**下一次** RELEASEALL 就再也想不起来要补 ✗；
          · 留着 ⇒ 每一轮都能补，而且序列走到 `up` 元素时照常发 RELEASE ✓ 不会卡键 ✓。
        收尾照旧由序列自己负责（`_release_ctx` / `_release_held_set` ✓）。
        """
        if not ctx:
            return
        keys = sorted(k for k in ctx[3] if k)
        for k in keys:
            try:
                key_down(k)
            except Exception:                   # noqa: BLE001
                pass
        if keys:
            behavior.event("key_reassert", keys="/".join(sorted(keys)))
        for sub in ctx[4]:
            self._reassert_ctx(sub)

    @staticmethod
    def _release_held_set(held):
        """释放一个 held 集合里的键。"""
        for k in held:
            try:
                key_up(k)
            except Exception:
                pass
        held.clear()

    def _random_target_cd(self):
        """目标切换 CD（秒）：从 [min, max] 毫秒区间随机取。"""
        lo, hi = self.settings.target_cd
        lo = max(0.0, float(lo)) / 1000.0
        hi = max(lo, float(hi)) / 1000.0
        return random.uniform(lo, hi)

    def _vision_rect(self, player, ws):
        """计算视野矩形 (left, top, right, bottom)，None 表示该方向不限制。

        基准点：角色中心（player.x, player.y）或画面中心（「基于画面中心」勾选时），
        再叠加 x/y 偏移。向上下左右各扩展 vision_* 像素。
        """
        s = self.settings
        if s.vision_center:
            cx = ws.width / 2.0
            cy = ws.height / 2.0
        else:
            cx = player.x
            cy = player.y
        cx += s.vision_off_x
        cy += s.vision_off_y
        left = cx - s.vision_left if s.vision_left >= 0 else None
        right = cx + s.vision_right if s.vision_right >= 0 else None
        top = cy - s.vision_top if s.vision_top >= 0 else None
        bottom = cy + s.vision_bottom if s.vision_bottom >= 0 else None
        return left, top, right, bottom

    def _filter_mobs(self, mobs, player, ws):
        """按视野矩形过滤怪物：视野外的怪不参与决策。某方向 <0 则不限制该方向。"""
        left, top, right, bottom = self._vision_rect(player, ws)
        out = []
        for m in mobs:
            if left is not None and m.x < left:
                continue
            if right is not None and m.x > right:
                continue
            if top is not None and m.y < top:
                continue
            if bottom is not None and m.y > bottom:
                continue
            out.append(m)
        return out

    # ---------------- 决策辅助 ---------------- 

    def _candidates(self, mobs, px, ws=None, now=None):
        """锁定候选：平地巡逻锁全部；扫平台只锁背后 back_range 内的怪。
        （朝向方向的怪不锁定，靠 attack 优先级就近攻击。）
        「背后」相对倾向朝向（_patrol_dir），与打背后怪时的临时转身无关。

        ⚠ **「限制战斗区域」也在这里筛**（用户 2026-09-27 要求 1，原话："锁定目标不能在
        非限制战斗区域内"）：配了 `settings.battle_zone_sets`（非空 = 启用 ✓）时，**怪自己站的
        那块 foothold 集合**必须和区域有交集，否则**不给锁、也不追** ✓。

        为什么落在这一处：`_candidates` 是**锁定/追击链的唯一漏斗**
        （`_locked_target` ← `_nearest` ✓），而且**不碰攻击链**（`_in_range*` 吃的是原始
        `mobs` ✓）⇒ **够得着的怪照打** ✓，只是不"锁住它、追出区域" ✓ —— 正是"锁定目标"
        这四个字 ✓。
        ⚠ **拿不到就不拦**（没配区域 / 没注入 `mob_sets_of` / 解析器抛错 / 怪那条 foothold
          没圈进集合）✓ —— 绝不因为算不出来就把怪全筛掉 ✗。
        """
        s = self.settings
        if s.strategy == "sweep":
            back = max(0.0, float(s.back_range))
            out = [m for m in mobs if -back <= (m.x - px) * self._patrol_dir < 0]
        else:
            out = list(mobs)
        return self._zone_only(out, ws, now)

    def _zone_cd_s(self, zone=None):
        """「**区域查询CD(s)**」= **某个战斗区域项**的 CD（秒）✓ —— 旧「前往重下间隔(s)」的
        **逐项版本**（用户 2026-09-28："参数移除，逻辑移动到战斗区域限制配置的每一项上、
        改名「区域查询CD(s)」"✓）。

        `zone` = 问**哪个集合**那一项（空 / 找不到 ⇒ 退回 `ZONE_GOTO_RETRY_S` ✓）。
        三处调用点各传各的（用户 2026-09-28 定 ✓）：
          · **回区域重下**（`_leave_battle_zone_tick`）⇒ 传**目标**那个区域 ✓；
          · **追击下前往**（`_chase_goto_if_elsewhere`）+ **区域筛缓存时效**（`_zone_only`）
            ⇒ 传「**怪那块 foothold 命中的那个区域项**」✓（归属不到 ⇒ 不传 ⇒ 兜底 ✓）。
        ⚠ 下限 0.5：再小就等于"每拍重下 / 每拍重算" ✗（`_load_battle_zones` 里也钳，双保险 ✓）。
        """
        name = str(zone or "").strip()
        if name:
            for z in (getattr(self.settings, "battle_zones", None) or []):
                if str(z.get("set") or "") == name:
                    try:
                        return max(0.5, float(z.get("cd_s", ZONE_GOTO_RETRY_S)))
                    except (TypeError, ValueError):
                        break
        return max(0.5, float(ZONE_GOTO_RETRY_S))

    def _zone_cd_of_sets(self, names):
        """`names`（一堆集合名，比如"怪那块 foothold 属于哪几个集合"✓）里**第一个命中战斗
        区域项**的那个 ⇒ 它的「区域查询CD(s)」✓；一个都没命中 ⇒ 兜底 `ZONE_GOTO_RETRY_S` ✓。

        用户 2026-09-28 原话："②③ 也用**目标区域项**：追击时**怪那块 foothold 所属的区域项**
        （归属不到任何项 ⇒ 退回默认 3s）；区域筛缓存同理" ✓ —— 这一处把那条口径收成一处 ✓
        （追击下前往 / 区域筛缓存**共用**它 ✓，别在两处各写一遍"怎么找那一项" ✗）。
        """
        zs = [str(z).strip() for z in (names or []) if str(z).strip()]
        for n in zs:
            for z in (getattr(self.settings, "battle_zones", None) or []):
                if str(z.get("set") or "") == n:
                    return self._zone_cd_s(n)
        return self._zone_cd_s()

    def _zone_of_sets(self, names):
        """`names`（一堆集合名 —— 通常是**脚下** `here_sets` ✓）里**第一个命中战斗区域项**
        的那个 ⇒ **那一项 dict** ✓；一个都没命中 ⇒ `None` ✓。

        ⚠ 与 `_zone_cd_of_sets`（那只回 CD 那个 float ✓）**同一套"怎么找那一项"**的口径 ——
          两边都是照 `settings.battle_zones` 里的 `z["set"]` 匹配 ✓（别再各写一份 ✗）。
        用途（2026-09-28 接行为时加 ✓）：「最大战斗时长 / 到点去哪」和「idle 回归 foothold」
        都挂在**当前这一项**上 ⇒ 得先问出"我现在在哪一项里" ✓（`_in_battle_zone` 只回 bool ✗）。
        """
        for n in [str(x).strip() for x in (names or []) if str(x).strip()]:
            for z in (getattr(self.settings, "battle_zones", None) or []):
                if str(z.get("set") or "") == n:
                    return z
        return None

    def _fight_beat(self, now, ws):
        """**「最大战斗时长」到点就换地方**（`battle_zones` 的 `fight_max_s` / `fight_dst` ✓）。

        ⭐⭐ **口径（用户 2026-09-28 现场改过一次，别改回去 ✗）**：

        | 事项 | 结论 |
        |---|---|
        | **"在战斗"怎么判** | **没有寻路任务** ⇒ 就是界面「当前任务」显示「**战斗**」的那一行 ✓ —— 判据**必须**和它**同一个源**（`current_goto_set()` ✓）。**不许**用 `self.state` ✗ |
        | **什么时候停时间** | **只有寻路会打断** ✓ —— 别的一律**不许**停（用户原话："**只要一直战斗就不应该有任何理由停时间**"✓） |
        | 寻路打断时 | **暂停累计**（不清 ✓）—— 寻路结束接着算 |
        | 真换到另一个区域 | **归零**（`_fight_acc = 0` ✓） |
        | 区域**读不到**一瞬 | **什么都不做**（定位抖动 ⇒ 不清 ✓） |
        | 到点判据 | **`_fight_acc`（实际在打的总秒数）≥ 上限** —— ⚠ **不是** `now - _fight_since` ✗（那种算法一旦暂停就废） |

        ⚠⚠ **为什么改**（用户报的 bug ✗）：原来判据是 `self.state in ("attack", "chase")` +
        `z` 非空，**任一不满足就清零**。而 `self.state` 是**另一套**东西 —— 它在攻击 / 换目标 /
        追击之间会**闪**，`z`（`here_sets`）也会因**定位抖动**短暂读不到（`docs/寻路设计.md`
        里写过"定位抖动就这个量级"✓）⇒ 时间被反复清掉 ⇒ 用户看到"**当前任务一直是战斗，
        但是时间走几秒就没了**"✗。

        在哪调：**每拍公共量**那一区（和 `_chase_since` 同一个位置 ✓）—— 它**只记时间 / 到点
        下任务**，不直接改状态（状态由这一拍后头的攻击支 / 任务支决定 ✓）。
        ⚠ 下之前先看 `self._climb is not None`：手上已经有行程就不硬插（同别处的纪律 ✓）。
        """
        z = self._zone_of_sets(getattr(getattr(ws, "player", None), "here_sets", None))
        if z is not None:
            try:
                max_s = max(0.0, float(z.get("fight_max_s") or 0.0))
            except (TypeError, ValueError):
                max_s = 0.0
            if max_s <= 0.0:
                self._fight_reset(now)        # 这一项 = 不限 ⇒ 收摊 ✓
                return
            # 真换到**另一个区域** ⇒ 归零（⚠ 只在**两边都读得到**时才敢这么判 ——
            # 读不到一瞬不是"离开"，用户口径 ✓）
            if (self._fight_zone is not None
                    and str(z.get("set") or "") != str(self._fight_zone.get("set") or "")):
                self._fight_reset(now)
            if self._fight_zone is None:
                self._fight_since = now       # 这一轮从这一刻算起（给界面/用例看 ✓）
            self._fight_zone = z              # ⭐ 记下"计时的是**这一项**"（界面要用 ✓）
        if self._fight_zone is None:
            self._fight_last = now
            return
        if z is None:
            # ⭐ **区域读不到 ⇒ 整拍作废**（用户 2026-09-28："只要一直战斗就不应该有任何
            #   理由停时间"✓ ⇒ 定位抖一瞬**既不算时间、也不判到点**）。
            #   ⚠⚠ 这里**曾经漏了这一步**（踩过 ✗）：只跳过"读区域"那一段、却继续往下累加
            #   ⇒ 抖动期间照样涨秒数 ⇒ **一涨到上限就"到点清账"** ⇒ 界面那行跳回满 ⇒
            #   用户看到的就是"一直战斗、时间走几秒就没了" ✗（正是他报的那个 bug ✓）。
            self._fight_last = now
            return
        # ⭐ **"在战斗" = 没有寻路任务**（= 界面「当前任务」那行显示「战斗」✓ **同一个源** ✓）
        #   ⚠ **只有寻路会打断时间**（用户 2026-09-28 原话 ✓）—— 而且这里是**暂停累计**、
        #   不是清零 ✓：寻路一结束，接着往下算 ✓
        fighting = not bool(self.current_goto_set())
        if self._fight_last is not None and fighting:
            self._fight_acc += max(0.0, now - self._fight_last)   # 只把"在打"的这拍累进去 ✓
        self._fight_last = now
        cap = max(0.0, float(self._fight_zone.get("fight_max_s") or 0.0))
        self._publish_fight_clock(str(self._fight_zone.get("set") or ""),
                                  self._fight_acc, cap)
        if self._fight_acc < cap:
            return
        dst = str(self._fight_zone.get("fight_dst") or "")
        self._fight_reset(now)                # 到点 ⇒ 收摊（别每拍都来一遍 ✓）
        if not dst or self._climb is not None:
            return
        ok, msg = self.plan_and_start_route(
            dst, why="战斗时长到了（本区域累计已打 %.0f 秒）⇒ 换去「%s」"
                     % (cap, dst))
        self._last_goto_note = (str(msg), now)
        behavior.event("fight_timeout_goto" if ok else "fight_timeout_fail", dst=str(dst))

    def _fight_reset(self, now=None):
        """「最大战斗时长」这一轮**收摊**（`_fight_beat` 的三个出口共用 ✓）。

        ⚠ 只清**本区域这一轮**的账（`_fight_since` / `_fight_acc` / `_fight_zone` /
        `_fight_last` ✓）+ 把界面那三件套清掉 ✓ —— "进入另一个区域""这项改成不限""到点换地方"
        三件事共用同一份口径（一处 ✗ 别各写一遍）。
        """
        self._fight_since = None
        self._fight_acc = 0.0
        self._fight_last = now
        self._fight_zone = None
        self._publish_fight_clock(None, 0.0, 0.0)

    def _publish_fight_clock(self, zone, elapsed_s, cap_s):
        """把「**哪个区域在计时 / 从哪一刻 / 上限多少**」写到 `settings` 上，**只给界面读** ✓。

        用户 2026-09-28 要求："编辑战斗区域里，要用**灰字**显示**最大战斗时长的倒计时**"✓。
        ⚠ 为什么落 `settings` 而不是"让面板去拿 agent" ✗：那个弹窗（`gui/player_panel.py`
          的 `BattleZoneListDialog`）只 import 了 **`settings` 单例** ✓，拿不到 agent 实例
          ⇒ 这是**接线最省**的一条路（同 `enabled` / 休息那几项"运行时状态挂 settings"的
          老做法 ✓ 它们**都不进 `to_dict`** ⇒ **不落盘** ✓ 也不会跨会话复活 ✓）。
        ⚠ **只写不判** ✓ —— 面板是**只读者**，不许反过来改这几个（同 `pos_state()` 那条纪律 ✓）。
        ⚠⚠ **`elapsed_s` 是"已累计在打的秒数"**（用户 2026-09-28 改口径后 ✓）—— 面板算剩余
          只需 `cap_s - elapsed_s` ✓，**不用**再拿时钟减（时间已经由 agent 那边累计好了 ✓）。
          所以现在**每拍都要写**（累计值一直在变 ✓ 原来是只在边沿写一次 ✗ —— 那种写法配上
          "会暂停"的新口径就错了）。
        """
        try:
            self.settings.fight_zone_name = zone
            self.settings.fight_elapsed_s = float(elapsed_s or 0.0)
            self.settings.fight_cap_s = float(cap_s or 0.0)
        except Exception:                     # noqa: BLE001 —— 界面用的小账，坏了别影响决策 ✗
            pass

    def fight_remain(self):
        """⭐ 「**本区域这一轮连着打还剩多少秒**」→ `(集合名, 剩余秒, 上限秒)` / `None`（2026-09-28 ✓）。

        给界面用（用户原话："编辑战斗区域里，要用**灰字**显示**最大战斗时长的倒计时**"✓）。
        ⚠ **只读** ✓ —— 界面**不许**当第二个写者（同 `pos_state()` 那条纪律 ✓）。
        ⚠ 口径与 `_fight_beat` **完全同一份**（`_fight_zone` + `_fight_acc` ✓）。
        `None` = **这会儿没在计时**（不在区域 / 那一项 `fight_max_s = 0` = 不限 ✓）
          ⇒ 界面显示静态的「最多打 Ns」就行 ✓（别显示 `0:00` ✗）。
        ⚠⚠ 剩余用 **`_fight_acc`** 算（"实际在打的总秒数" ✓）—— **不是** `now - 起点` ✗：
          寻路打断期间是**暂停**的，拿墙钟减会把打断的那段也算进去 ✗（用户 2026-09-28 的口径）。
        """
        z = self._fight_zone
        if not z:
            return None
        try:
            max_s = max(0.0, float(z.get("fight_max_s") or 0.0))
        except (TypeError, ValueError):
            return None
        if max_s <= 0.0:
            return None
        return (str(z.get("set") or ""), max(0.0, max_s - self._fight_acc), max_s)

    def _idle_walk_beat(self, keys, ws):
        """**idle 回归**：没怪、也没任务时，**水平走**回本区域那一格的中心 x ✓。

        用户 2026-09-28 定的口径（见 `__init__` 里 `battle_zones` 的 `idle_foothold`
        注释原文 ✓）：
          · 只有在"**位于本集合**"时才做（脚下 `here_sets` 命中某个区域项 ✓）；
          · **水平走**（只按 ←/→ ✓），到中心就停（`_deadzone` 之内 ⇒ 一个键都不按 ✓）；
          · **不跨层** ✗ —— 不下寻路任务、不爬绳（"跨层换地方"是 `fight_dst` 那种事 ✓）。
        ⚠ 中心 x 由**注入的** `self.foothold_x`（`fid -> 世界中心 x` ✓，`gui/live_thread.py`
          装的 ✓）给 —— agent **不持有地形** ✓（同 `mob_sets_of` / `route_plan` 的纪律 ✓；
          拿不到就什么都不做 ⇒ 照旧站住 ✓ 绝不自作聪明 ✗）。
        """
        z = self._zone_of_sets(getattr(getattr(ws, "player", None), "here_sets", None))
        fid = str((z or {}).get("idle_foothold") or "")
        fx = getattr(self, "foothold_x", None)
        if not fid or fx is None:
            return
        try:
            cx = fx(str(fid))
        except Exception:                     # noqa: BLE001 —— 解析器坏了就照旧站住 ✓
            return
        wx = getattr(getattr(ws, "player", None), "world_x", None)
        if cx is None or wx is None:
            return
        dx = float(cx) - float(wx)
        if abs(dx) <= float(self._deadzone):
            return                            # 到中心了 ⇒ 站住（一个键都不按 ✓）
        self._steer(dx, keys)

    def _zone_only(self, mobs, ws, now):
        """把**不在「限制战斗区域」里**的怪筛掉（用户 2026-09-27 要求 1）；判不了就不筛 ✓。

        判据：怪站的集合 ∩ `settings.battle_zone_sets` ≠ ∅（**玩家**那条在
        `_in_battle_zone`，管的是"我在不在区域里" ✓；这条管的是"**怪在不在**区域里" ✓）。
        ⚠ 没有 `battle_zone_sets` 时**恒不筛**（老行为 ✓ —— 用例
          `t_temp_fight_out_of_zone` 明钉"没配区域不许改坏" ✓）。
        ⚠ 怪换块平台没多少帧那么快 ⇒ **缓存**结果，间隔借 `ZONE_GOTO_RETRY_S`
          （同一件"重下间隔"参数 ✓，不新造常数 ✓）；怪不见了就把缓存项丢掉 ✓。
        """
        zs = set(str(z) for z in (getattr(self.settings, "battle_zone_sets", None) or [])
                 if str(z))
        fn = getattr(self, "mob_sets_of", None)
        cache = getattr(self, "_zone_cache", None)
        if not zs or fn is None or ws is None:
            self._zone_cache = {}
            return list(mobs)                  # 没配区域 / 没解析器 ⇒ 老行为 ✓
        if now is None:
            now = time.monotonic()
        if cache is None:
            cache = self._zone_cache = {}
        keep, alive = [], set()
        for m in mobs:
            key = getattr(m, "id", None)
            hit = cache.get(key) if key is not None else None
            # ⭐ TTL = 「**上次这个怪命中的那个区域项**」的「区域查询CD(s)」✓（用户 2026-09-28
            #   口径："区域筛缓存同理" ⇒ 按**怪那块 foothold 所属的区域项**取 ✓）；还没算过 /
            #   上次没命中 ⇒ 兜底 `ZONE_GOTO_RETRY_S` ✓。
            _cd = (self._zone_cd_s(hit[2]) if (hit and len(hit) > 2)
                   else self._zone_cd_of_sets(None))
            if key is None or hit is None or now - hit[0] >= _cd:
                ok = None
                _zname = ""
                try:
                    info = fn(ws.player, m)
                except Exception:              # noqa: BLE001 —— 解析器坏了别把候选清空 ✗
                    info = None
                ms = set(str(x) for x in ((info or {}).get("sets") or []) if str(x))
                if ms:
                    ok = bool(ms & zs)
                    # 命中的区域项（可能不止一个 ⇒ 取**确定性的**第一个 ✓，别用 set 的序 ✗）
                    _hit = sorted(ms & zs)
                    _zname = _hit[0] if _hit else ""
                    behavior.event("mob_zone_in" if ok else "mob_zone_out",
                                   mob=getattr(m, "id", ""))
                if key is not None:
                    cache[key] = hit = (now, ok, _zname)
            if key is not None:
                alive.add(key)
            if hit is None or hit[1] is not False:
                keep.append(m)                 # `None` = 判不了 ⇒ **不拦** ✓
        for k in list(cache):
            if k not in alive:
                del cache[k]                   # 怪走了 ⇒ 缓存项丢掉 ✓
        return keep

    def _in_range(self, mobs, ws):
        """朝向前方、在**攻击范围矩形**里的框，按水平距离升序。

        攻击只朝前方：背后的框即使够得着也不算在攻击范围内，走 chase 转身。
        ⚠ 矩形口径（含上下攻击距离）只在 `_in_attack_box` **一处**（用户 2026-09-27）✓。
        """
        px = ws.player.x
        return sorted([m for m in mobs
                       if (m.x - px) * self.facing >= 0
                       and self._in_attack_box(m, ws.player)],
                      key=lambda m: self._center_dist(m, ws.player))

    def _in_range_all(self, mobs, ws):
        """**攻击范围矩形**里、**不分前后**的怪（同样按水平距离升序）。

        和 `_in_range` 只差一条：**不筛朝向**。谁用：**挂着「命令前往」任务的时候**
        （见 tick 里那句注释）。

        为什么非要它（用户 2026-09-26 报："命令前往的寻路途中，攻击范围内有怪也不 attack"）：
        `_in_range` 只认**正前方** ✓ —— 正常路径里身后的怪由「回身输出」那条分支接 ✓
        （`_locked_target` + back_ctx），可**任务在跑时 else 分支是任务自己**（`_climb_tick`）
        ⇒ 身后怪**没人接** ✗ ⇒ 一路白挨打（走路时怪多半从身后追上来，所以看起来就是
        "身边有怪却不打" ✗）。算进来之后，`_attack_state` 里"挂任务只许站桩"那一支会
        **原地转身**把它打掉 ✓，而**不会**走位 ✓。
        """
        return sorted([m for m in mobs
                       # ⚠ **必须显式 `facing=0`**（用户 2026-09-28 ✓）：本方法的存在意义就是
                       #   "**不筛朝向**" ✓（见上面那段注释 ✓）—— 而 `_in_attack_box` 现在
                       #   **默认按 `self.facing` 单侧判** ✗ ⇒ 不显式传 0 就会被它顺手筛掉一半 ✗
                       #   （那这条分支就废了：身后怪又没人接 ⇒ 退回"身边有怪却不打"✗）。
                       if self._in_attack_box(m, ws.player, 0)],
                      key=lambda m: self._center_dist(m, ws.player))

    def _in_battle_zone(self, ws):
        """这一拍**允许打架**吗 —— 「限制战斗区域」那条规则（空列表 = 不限制 ✓）。

        判据用感知给的 `here_sets`（脚下属于哪些命名集合 ✓）—— 和寻路那边**同一份**
        数据（不另发明一套"我在哪块平台"✗）。**拿不到 `here_sets` 也算"不在区域里"** ✓：
        宁可先回去，也别在不知道自己在哪的时候开打 ✗。
        """
        zones = [str(z) for z in
                 (getattr(self.settings, "battle_zone_sets", None) or []) if str(z)]
        if not zones:
            return True                     # 没配 = 不限制（老行为 ✓）
        here = set(getattr(getattr(ws, "player", None), "here_sets", None) or ())
        return bool(here & set(zones))

    def _mob_sig(self, target, ws):
        """这只怪**当前这个局面**的指纹：`(怪的画面 x/y 档, 玩家脚下集合)` ✓。

        用户 2026-09-28 提："**在恰当的时机重新查询（之前我说 1s 但是可能不太好用）**" ✓ ——
        比"按时间过期"（`MOB_SETS_TTL_S` 纯时间驱动 ✗）好在：**该重查才重查** ✓。
        · 怪**站着不动、玩家也没换层** ⇒ 指纹不变 ⇒ **不重查** ✓（判定稳定，不会来回走 ✓）；
        · 怪**走了 / 跳了 / 换层了** ⇒ 指纹变 ⇒ **立刻重查** ✓（不用等那一秒 ✓）；
        · **玩家换层** ⇒ 也算变 ✓（同一个 x 在新平台上算得出不同答案 ✓）。

        ⚠ 位置按 `MOB_RESIG_PX` **量化**（不是逐像素比 ✗）：检测框每拍都在抖几像素，
          逐像素比等于"每拍都重查" ✗ = 回到老毛病 ✓。
        """
        try:
            mx = round(float(getattr(target, "x", 0.0) or 0.0) / MOB_RESIG_PX)
            my = round(float(getattr(target, "y", 0.0) or 0.0) / MOB_RESIG_PX)
        except (TypeError, ValueError):
            mx = my = None
        hs = tuple(sorted(str(s) for s in
                          (getattr(getattr(ws, "player", None), "here_sets", None) or [])))
        return (mx, my, hs)

    def _locked_mob_info(self, ws, target, now):
        """**被锁定 / 正被追的那只怪**站在哪块 foothold 集合 —— **每秒只查一次** ✓。

        用户 2026-09-28 要求（原话）："**锁定一个目标时每秒更新其位于的 foothold 集合**"。
        （现场症状："锁着目标来回走" ✓ —— 见模块里 `MOB_SETS_TTL_S` 的说明。）

        回值 = 解析器那份 `{"world", "sets", …}`（原来 `_chase_goto_if_elsewhere` 直接调
        `mob_sets_of` 拿的东西 ✓ —— 调用方后面还要 `info["world"]` 去算"够不够得着" ✓
        ⇒ 这里**整份**缓存，别只缓存 `sets` ✗：两处口径必须来自同一次换算 ✓）。
        查不到（没注入解析器 / 抛错 / 换算不出）⇒ `None`（**不缓存"空"** ✓）。

        ⚠ **换目标必须重查**（缓存里存了目标 id ✓）：否则会拿**上一只怪**的集合去下前往 ✗。
        """
        key = None if getattr(target, "id", None) is None else str(target.id)
        sig = self._mob_sig(target, ws)         # ⭐ 局面指纹（见 `_mob_sig` ✓）
        ent = self._mob_info_cache
        # ⭐ **缓存时长走参数**（用户 2026-09-28："直接做" ✓）—— 模块常量只是**兜底** ✓
        #   （老调用方 / 替身对象没有 `settings` 时别炸 ✗）；`0` = 每次都重查 ✓。
        _ttl = float(getattr(self.settings, "mob_sets_ttl_s", MOB_SETS_TTL_S) or 0.0)
        if (key is not None and ent is not None and len(ent) > 3
                and ent[0] == key and ent[3] == sig
                and 0.0 <= (now - ent[1]) < _ttl):
            return ent[2]                       # 同一只 + **局面没变** ⇒ 用缓存 ✓
        fn = getattr(self, "mob_sets_of", None)
        if fn is None:
            self._mob_info_cache = None
            return None
        try:
            info = fn(ws.player, target)
        except Exception:                       # noqa: BLE001 —— 解析器坏了别把追怪弄停 ✗
            info = None
        if not info:
            # ⚠ **查不到就不缓存**：那说明这一刻给不出答案（地形刚载入 / 怪在边界外 ✓）
            #   ⇒ 下一拍该再试；缓存一个"空"会让它整整一秒都不再判 ✗。
            self._mob_info_cache = None
            return None
        if key is not None:
            self._mob_info_cache = (key, now, info, sig)
        return info

    def _mob_x_reaches_only(self, m, player):
        """**水平够得着、但竖直够不着**吗（用户 2026-09-28 那条 ③ 的判据 ✓）。

        用户原话："**判错了：程序认为怪与玩家处于相同 foothold 集合，但玩家的攻击范围框 x 长范围
        与怪框 x 长范围相交后 <攻击范围框 y 长与怪框 y 长范围没有交集>（即攻击范围框框不住怪碰撞盒，
        典型的矩形相交问题），也要朝「向量方向的最近集合」逐层逼近**" ✓。

        两件都成立才算（**只按"打不到"太宽** ✗ —— 怪在天边也会被判成该上下跑 ✗）：
          · **水平够**（`min_attack_dist ≤ 水平距离 ≤ attack_dist` ✓ 同一套 `_center_dist` 口径 ✓）；
          · **整体打不到**（`not _in_attack_box` ✓ —— 它就是"攻击框 ∩ 怪框 = ∅"的口径 ✓
            含竖直那两条边 ✓）。
        """
        s = self.settings
        try:
            d = float(self._center_dist(m, player))
            if not (float(s.min_attack_dist) <= d <= float(s.attack_dist)):
                return False
        except (TypeError, ValueError):
            return False
        return not self._in_attack_box(m, player)

    def _note_chase_skip(self, why):
        """⭐ **追击里每条早退都要留痕**（2026-09-28 加 ✓）。

        用户 2026-09-28 原话："**这次又卡住了，依旧没有走『取向量 → 在小地图上根据黄点 + 向量
        找首个 foothold 集合 → 下达寻路任务』**"✓ —— 而当时 log 里**一条线索都没有** ✗。

        病根：`_chase_goto_if_elsewhere` / `_mob_goto_towards` 的早退有**五六条**，其中
        **大多是静默的** ✗（没注入解析器 / 玩家 `here_sets` 空 / 解析器返回空 / 那层走不到…）
        ⇒ 现场只看到"贴在那儿不动、也不下任务"，**分不清卡在哪一条** ✗✗
        —— 这正是这个 bug 查了三轮还没收尾的原因 ✓。

        ⚠ **只记"原因变了"的那一次**（同 `mob_fh` 的手法 ✓）：早退是**每拍**都会命中的，
          每条都记会把 log 淹掉 ✗（那反而更看不清 ✓）。
        ⚠ 节流那条（`_mob_goto_at`）**故意不记** —— 它本来就该每拍命中 ✓ 纯噪音 ✗。
        """
        key = str(why)
        if key == getattr(self, "_last_chase_skip", None):
            return
        self._last_chase_skip = key
        try:
            behavior.event("chase_skip", why=key)
        except Exception:                      # noqa: BLE001 —— 打点坏了别影响追击 ✓
            pass

    @staticmethod
    def _chase_origin(target):
        """「**追击**」这个来源的签名（用户 2026-09-28 的「来源签名」✓）。

        它是**唯一**参与"那两个节点复核"的来源（用户 2026-09-28 定：**只掐「追击」类** ✓）
        —— 其余来源（命令前往 / 任务队列 / 定点休息 / 回战斗区域 / 到点去哪）**一律豁免** ✓。

        `mob_id` = **当初要追的那只怪** ✓（复核时按它在当前候选里找回来、**现算**一次它的代价）。
        ⚠⚠ **不记"当时算出来的代价"** ✗：那个数是从**当时站的位置**算的，而人一直在走
          ⇒ 跟"现在的代价"**不可比** ✗（复核必须两边同口径 —— 都从**现在**的位置算 ✓）。
        """
        return {"kind": "chase", "mob_id": getattr(target, "id", None)}

    def _mob_goto_towards(self, target, ws, now, why):
        """**降级路径**：朝「玩家 → 怪」方向的**最近集合**下一次「前往」✓。

        用户 2026-09-28 定稿（原话："**如果玩家的攻击范围框 x 长与怪框 x 长范围相交了却没有触发
        attack，就下达一个前往『玩家到怪物向量』指向的最近一个 foothold 集合的任务**" ✓）。
        为什么要它：①「怪所在集合」那条路**要先判准怪在哪层** ✗（这几轮翻车的都是它）；
        **方向**却是可靠的（画面 1:1 已验证 ✓）⇒ 用方向绕开层判定 ✓。

        **三条触发**（都是"怪那层不可信 / 走不到" ✓）：
          · ② 怪所在集合**判不出**（`sets` 空）或**下前往失败**；
          · ③ 判成"**同一集合**"、但**攻击框 ∩ 怪框 = ∅**（`_mob_x_reaches_only` ✓）。
        ⚠ 解析器是**注入**的（`agent.nearest_set_towards` ✓，几何在 `gui/live_thread.py` ✓）——
          没注入 / 解不出 ⇒ `False`（**不动** ✓ 宁缺勿错 ✓，绝不自作聪明 ✗）。
        ⚠ 节流共用 `_mob_goto_at`（同一条规则 ✓ 别另开一把钟 ✗）。
        """
        fn = getattr(self, "nearest_set_towards", None)
        if fn is None:
            # ⚠ **早退必须留痕**（2026-09-28 ✓）：原来这条是**静默**的 ⇒ 现场"不走降级、
            #   也没有任何线索" ✗（用户："**依旧没有走『取向量 → 在小地图上根据黄点 + 向量
            #   找首个 foothold 集合 → 下达寻路任务』**"✓）。
            self._note_chase_skip("没注入 nearest_set_towards（实时线程没把方向解析器交给我）")
            return False
        if now - getattr(self, "_mob_goto_at", 0.0) < self._zone_cd_s():
            return False        # ⚠ 节流：**故意不留痕**（每拍都会命中 ⇒ 记了纯噪音 ✗）
        try:
            dst = fn(ws.player, target)
        except Exception as ex:                 # noqa: BLE001 —— 解析器坏了别把追怪弄停 ✗
            # ⛔ **绝不能静默**（2026-09-28 血的教训 ✗）：这条降级（"怪判不出集合 ⇒ 朝
            #   『玩家 → 怪』方向最近的集合逐层逼近"✓）**从上线起每次都在抛** ——
            #   `live_thread._make_toward_set_resolver` 里把 `set_span` 的**二元组**按
            #   **四元组**解包（`sp[2]` ⇒ `IndexError` ✓）⇒ 这里一律 `return False` ⇒
            #   **一条事件都不打** ✗ ⇒ 现场只看到"原地卡住、也不下任务"，而 log 里干干净净 ✗
            #   （用户原话："**没找到也没根据向量下达寻路任务**"✓）—— 查了很久才定位 ✓。
            #   ⇒ 至少留一条痕（类型名就够定位 ✓）。
            behavior.event("mob_goto_toward_err", err=type(ex).__name__)
            return False
        dst = str(dst or "")
        if not dst:
            self._note_chase_skip("朝方向解析器判不出集合（那条路上没有可去的层）")
            return False
        self._mob_goto_at = now
        # ⭐ 来源签名 = 「追击」（用户 2026-09-28 ✓）⇒ 这一步会参与那两个节点的复核 ✓
        # ⚠ `keep_clock=True`（用户 2026-09-29 第 2 条 ✓）：**攻击重下同一趟追击 ⇒ 不刷新
        #   时间钟** ✓（不然怪一换层就重打，30 秒的闸永远等不到 ✓）。
        ok, msg = self.plan_and_start_route(dst, why=str(why) % dst,
                                           origin=self._chase_origin(target),
                                           keep_clock=True)
        self._last_goto_note = (str(msg), now)
        behavior.event("mob_goto_toward" if ok else "mob_goto_toward_fail", dst=str(dst))
        return bool(ok)

    def _chase_goto_if_elsewhere(self, target, now, ws):
        """追的怪**不在我站的这块平台上** ⇒ 先下「前往」任务；返回 True（这一帧别朝它走 ✓）。

        用户 2026-09-27 的要求（原话）："chase状态需要判定一下怪物位于的foothold集合，
        若不与玩家处于同一个，需要先下达前往任务"。

        为什么要（老毛病）：`chase` 只会**水平**按左右方向键（`_steer`）⇒ 怪在**上层 /
        下层 / 被墙隔开**的平台上时，人只会贴着平台边缘干蹭、或者朝墙走到底 ✗
        （`docs/寻路设计.md` 把"怪物追击与寻路的融合"原本列在"先不做"里 —— 这就是那件事 ✓）。

        怎么判：`self.mob_sets_of`（**实时线程注入**的解析器 ✓ agent 不持有地形/集合 ✓）
        给 (玩家, 怪) 换回 `{"world","sets"}`；玩家这头是 `ws.player.here_sets`
        （脚下那条 foothold 属于哪些命名集合 ✓）⇒ **两边有交集 = 同一块** ✓。

        ⚠ **保守四条**（拿不准就**照旧追**，绝不因此站着不动 ✗）：
          · 没注入解析器 / 解析器抛错 / 怪换算不出世界坐标 / 它的那条 foothold 没圈进集合
            ⇒ 返回 False（继续按老方式追 ✓）；
          · 玩家的 `here_sets` 空（脚下没圈进任何集合）⇒ 返回 False
            （**不知道自己在哪块 ⇒ 也没法说"怪不在我这块"** ✓ 和「限制战斗区域」正好相反：
              那条是"拿不到也算不在"（要回区域 ✓），这条是"拿不到就不动" ✓）；
          · 手上**已有任务**时**不再是无条件不打断**（2026-09-28 改 ✓）：只有当"**当前任务要去的
            那个集合**"仍在这次查出来的集合里（或者那任务说不清目标 —— 没有 `dst_set` ✓）
            才不打断 ✓；**不在** ⇒ 照旧往下走、节流放行时**重下**新目的地 ✓
            （用户原话："**查询了框如果不一样要下寻路任务呀**"✓）——
            `plan_and_start_route` 会**覆盖当前行程**，这时**正是要的** ✓（怪真换层了）；
          · 重下要有间隔（`ZONE_GOTO_RETRY_S`，与「限制战斗区域」共用同一件参数 ✓）：
            解析不出来时每拍重下会刷屏、还反复造任务 ✗。
        怪可能**同时在好几个集合**里 ⇒ 从前往后试，第一条能造出路线的就用它 ✓。
        """
        if getattr(self, "mob_sets_of", None) is None:
            self._note_chase_skip("没注入 mob_sets_of（实时线程没把「怪在哪块」交给我）")
            return False
        here = set(str(s) for s in (getattr(ws.player, "here_sets", None) or []))
        if not here:
            # ⚠ **玩家自己脚下没圈进任何集合** ⇒ 这条原来也是**静默**地不下任务 ✗（2026-09-28 ✓）：
            #   它和"怪判不出"是**完全不同的两件事**（后者会走降级 ✓），而现场**分不清** ✗。
            self._note_chase_skip("玩家的 here_sets 是空的（脚下那条 foothold 没圈进集合）")
            return False
        # ⭐ **锁定的怪：集合每秒只查一次**（用户 2026-09-28 ✓）—— "锁着目标来回走"就是这里
        #   原来**每拍现算**引起的：怪框抖几像素 ⇒ 它的集合在「一楼 / 二楼」之间跳 ⇒ 目的地
        #   跟着跳 ⇒ 人走到一半目的地又被改写 ✗。缓存口径见 `MOB_SETS_TTL_S` / `_locked_mob_info` ✓。
        info = self._locked_mob_info(ws, target, now)
        if not info:
            self._note_chase_skip("怪的世界坐标 / 集合解不出来（解析器给了空）")
            return False
        msets = [str(s) for s in (info.get("sets") or []) if str(s)]
        if not msets:
            # ⭐ **② 降级**（用户 2026-09-28 ✓）：怪那层**判不出** ⇒ 朝「玩家 → 怪」方向最近的
            #   集合逐层逼近 ✓（比"站着不动"强，比"乱猜一层"安全 ✓）。
            return self._mob_goto_towards(
                target, ws, now, "怪那层判不出 ⇒ 朝方向最近集合「%s」逐层逼近")
        if here & set(msets):
            # ⭐ **③**（用户 2026-09-28 定，原话见 `_mob_x_reaches_only` ✓）：判成"**同一集合**"、
            #   但**攻击框框不住怪**（x 相交、y 不相交 ✓）⇒ **照样要降级** ✓ ——
            #   这是现场"贴着干蹭"的真凶：原来这里**直接 `return False`** ⇒ **压根不下前往** ✗。
            #   ⚠ 两件都成立才降级（只按"打不到"太宽 ⇒ 怪在天边也会上下跑 ✗）；
            #     否则**照旧追** ✓（老行为一字不变 ✓）。
            if not self._mob_x_reaches_only(target, ws.player):
                return False
            return self._mob_goto_towards(
                target, ws, now,
                "判成同一集合、但攻击框框不住怪 ⇒ 朝方向最近集合「%s」逐层逼近")
        # ⭐ 用户 2026-09-27 要求 2：**够得着 + 同一条 x 线 ⇒ 不必下寻路任务，直接锁** ✓
        #   原话："如果攻击范围框延伸过去与目标框有交集，且其框底边与角色玩家当前 foothold
        #   集合的 x 范围有交集，那么就不需要下达寻路任务了可以直接锁"。
        #   放在"同一块平台"判完之后、「手上已有任务 / 节流 / 真下任务」之前 ✓ ——
        #   它是一条**豁免**：本来要下前往的，这两件成立就不用下了 ✓。
        if self._reachable_without_path(target, ws, info.get("world")):
            self._last_goto_note = ("够得着（延伸攻击框 ∩ 怪框，且怪框底边 ∩ 我这块集合的 x "
                                    "范围）⇒ 不下前往，直接追", now)
            behavior.event("mob_reach_skip")
            return False
        if self._climb is not None:
            # ⭐ **手上已有任务、但"查出来的框"跟它要去的**不是一回事 ⇒ **也要重下**
            #   （用户 2026-09-28 原话："**查询了框如果不一样要下寻路任务呀**"✓）。
            #   ⚠ 这里原来写的是**无条件**不打断 ✗（"正在走的路线不该被这条规则打断"）——
            #   实测过：手上正走「丁平台」、查出来怪在「甲平台」⇒ 目的地**一动不改**、
            #   人照旧走向丁平台 ✗，而怪**早就换层了** ✗（追人追丢就是这么来的）。
            #   ⇒ 判据改成"**当前任务要去的那个集合，在不在这次查出来的集合里**"：
            #     · 在（或说不清 —— 执行器没有 `dst_set` ✓）⇒ 不打断 ✓（还在对的路上）；
            #     · **不在** ⇒ 继续往下走 ⇒ 节流放行时**重下**（覆盖当前行程 ✓）——
            #       这正是"怪换层了就得改目的地" ✓。
            #   ⚠ 只动**这一条**：「回区域」（`_leave_battle_zone_tick`）那条"手上已有任务
            #     不打断"**保持原样** ✗（用户只说了追击这一处 ✓）。
            _dst = str(getattr(self._climb, "dst_set", "") or "")
            if (not _dst) or (_dst in msets):
                return False                   # 正走的就是它那块 ⇒ 不打断 ✓
        # 节流（同「限制战斗区域」✓）—— ⭐ CD 按「**怪那块 foothold 所属的那个区域项**」取 ✓
        # （用户 2026-09-28："②③ 也用目标区域项"✓；归属不到任何项 ⇒ 退回默认 ✓）。
        if now - getattr(self, "_mob_goto_at", 0.0) < self._zone_cd_of_sets(msets):
            return False
        self._mob_goto_at = now
        ok, msg = False, ""
        for name in msets:
            # ⚠ `keep_clock=True`（用户 2026-09-29 第 2 条 ✓）：这条正是"怪换层了 ⇒ 目标
            #   集合变了"的最常见来源 ⇒ 不带上它，签名一变就重打钟 = 被 attack 刷新 ✗。
            ok, msg = self.plan_and_start_route(
                name, why="追击：怪不在我这块平台上（它在「%s」），先过去" % name,
                origin=self._chase_origin(target), keep_clock=True)
            if ok:
                break
        if not ok:
            # ⭐ **② 降级**（用户 2026-09-28 ✓）：怪那层**走不到**（造不出路线 / 解析失败 ✓）
            #   ⇒ 朝「玩家 → 怪」方向的最近集合逐层逼近 ✓ —— 比"原地卡住"强 ✓。
            #   ⚠ 它自己有节流（共用 `_mob_goto_at` ✓）；成功就收工 ✓。
            if self._mob_goto_towards(
                    target, ws, now, "怪那层走不到 ⇒ 朝方向最近集合「%s」逐层逼近"):
                return True
        self._last_goto_note = (str(msg), now)
        behavior.event("mob_goto_ok" if ok else "mob_goto_fail", dst=str(msg)[:120])
        return bool(self._climb is not None)    # 下成功 ⇒ 这一帧别再朝它走 ✓

    def _reachable_without_path(self, m, ws, world):
        """「**够得着 ⇒ 不用下寻路任务，直接锁**」的几何豁免 —— 合格给 True。

        用户 2026-09-27 要求 2（原话）："如果**攻击范围框延伸过去与目标框有交集**，且**其框底边
        与角色玩家当前 foothold 集合的 x 范围有交集**，那么就不需要下达寻路任务了可以直接锁"。

        两件**都**成立才算：
          ① **延伸攻击框 ∩ 怪框** —— 复用 `_in_box`（攻击范围矩形的**唯一**判定口径 ✓），
             只是把"最大距离"换成 **`inf` = 无限向前延伸** ✓。
             ⚠ 用户 2026-09-27 明确："**不能用「追击起跳」的 `chase_jump_max`**，而是玩家的
               攻击范围框**无限向前延伸**"（第一版借了 `chase_jump_max` ⇒ 折算成"`attack_dist`
               再加几十像素"，那不是用户要的口径 ✗）⇒ 水平方向**不设上限** ✓；
             ⚠ "无限"**只管水平向前**：**上下**两条边（`attack_up_dist` / `attack_down_dist` ✓）
               与**盲区**（`min_attack_dist` ✓）照旧出力 —— 上下那两条才是"它在不跟我同一层 /
               同一高度"的界限 ✓（怪高出两百像素还豁免，就会又变成"贴着平台边缘干蹭" ✗）。
          ② **怪框底边 ∩ 我站的集合的 x 范围** —— 怪框是轴对齐矩形 ⇒ "底边"的 x 范围就是它的
             整条宽 `[wx ± w/2]`（和 `_center_dist` 用左右边**同一口径** ✓，没有第二套 ✓）；
             `wx` 取 `mob_sets_of` 给的**世界**脚点（`world[0]` ✓ = 画面 `m.x`、`m.y + h/2` 换算
             过去的 ✓ ⇒ 画面→世界是**纯平移**、宽高不变 ✓，所以宽度直接照搬 ✓）；
             再和 **位置状态广播给的"我这块平台有多宽"（`here_span`，非墙 foothold 的并集 ✓）
             的 x 范围「外扩「最大攻击距离」」**（两边各减/加 `attack_dist` ✓）比 x 区间相交 ✓
             —— ⚠ 2026-09-27 起**不再自己 `set_span_of` 算** ✗（几何只在 `pos_state.py` 算一次 ✓）。
             ⚠ 用户 2026-09-27 的**优化**（原话）："应该是「其框底边与角色玩家当前 foothold 集合的
               **(x 范围 + 最大攻击距离)** 有交集」" —— 第一版拿**光秃秃的 x 范围**比 ✗，那会漏掉
               "怪就在我平台边外一点点（站过去伸手就够到）"这类 ⇒ 白下一次前往 ✗。
               外扩量就是设置里的「最大攻击距离」`attack_dist` ✓（**不新造参数** ✓ —— 它本来就是
               "攻击框能伸出去多远" ✓；`min_attack_dist` 是盲区、不管这事 ✗）。

        为什么这条能省掉寻路任务：怪的**脚就横在我这块平台上**（x 范围有交集）+ **伸伸手就够到**
        ⇒ 它要么跟我同层、要么只差一点，`_steer` 左右走 + 起跳就够了 ✓，而
        `_chase_goto_if_elsewhere` 那套（换世界坐标 → 找集合 → 规划路线）在**集合图上确实"
        不是同一块"**（比如它站在我脚下平台旁边那条窄边上 ✗）⇒ 会白下一次「前往」✗。

        ⚠ **判不了就 False**（不豁免 ⇒ 按老路"先下前往" ✓）：没有世界脚点 /
          **位置状态广播没给出 `here_span`**（这一拍没读数、或脚下没圈进任何集合 ✓）⇒
          False ✓ —— 与 `_chase_goto_if_elsewhere` 其余各条同一条纪律：**拿不准就照旧** ✓。
        ⚠ ② 的"我这块平台有多宽"**不再自己算** ✗（2026-09-27 用户要求**收编**）：读广播的
          `here_span` ✓（`perception/pos_state.py` 算的 ✓，口径 = `route.set_span` =
          "集合里有哪些**非墙** foothold"的唯一一份 ✓）。
        """
        if not world:
            return False
        player = ws.player
        s = self.settings
        # 「延伸量」= **无限向前**（用户 2026-09-27 明确：不用 `chase_jump_max` ✓）：
        # 水平最大距离给 `inf`（不设限 ✓），盲区与上下两条边照旧（它们是"同一层 / 同一高度"
        # 的界限 ✓，少了就会"贴着平台边缘干蹭" ✗）。
        if not self._in_box(m, player, float("inf"),
                            float(getattr(s, "min_attack_dist", 0.0) or 0.0)):
            return False                       # ① 连"无限向前"的攻击框都罩不住 ⇒ 老路（下前往）✓
        lo = float(world[0]) - float(getattr(m, "w", 0.0) or 0.0) / 2.0
        hi = float(world[0]) + float(getattr(m, "w", 0.0) or 0.0) / 2.0
        # 「我这块平台」的 x 范围要**外扩「最大攻击距离」**再比（用户 2026-09-27 的优化 ✓）：
        # 怪站在平台**边外** attack_dist 以内 ⇒ 我走到边上伸手就够得到 ⇒ 同样不必寻路 ✓。
        pad = max(0.0, float(getattr(s, "attack_dist", 0.0) or 0.0))
        # ② 「我这块平台有多宽」⭐ 读**位置状态广播**（`here_span` ✓）—— 2026-09-27 用户要求
        #   **收编**：原来这里自己 `set_span_of(集合名)` 逐个问、再自己取并集 ✗（同一件事两处算，
        #   尺子一不一样都看不出来）；现在几何只在 `perception/pos_state.py` 算一次 ✓。
        #   ⚠ 判不出来（这一拍没读数 / 没圈进集合）⇒ `None` ⇒ 这一件**不成立** ✓
        #     （照旧"先下前往"✓，不硬认 ✓ —— 与 docstring 里那条纪律一致 ✓）。
        sp = getattr(player, "here_span", None)
        if not sp:
            return False
        return bool(lo <= float(sp[1]) + pad and float(sp[0]) - pad <= hi)

    def _pick_battle_zone(self, ws, zones):
        """在**能打**的几块里挑**代价最低**的那块（用户 2026-09-28 要求 ✓）。

        用户原话："如果当前 foothold 集合**不允许战斗**，则找**代价最低的可战斗区**"✓。
        代价口径 = **寻路距离**（用户 2026-09-28 定 ✓："从我现在站的集合走到那个可战斗区" ✓，
        与「追击」挑怪**同一套**：`route.path_cost` ✓，由实时线程注入成 `set_cost_of` ✓）。

        ⚠⚠ **一次只算一轮，而且调用方必须把它放在节流放行之后** ✗：一次代价 = 一次 BFS +
        每段 `pick_edge`（`route.path_cost` 的说明里写明"**别挂进每拍**"✗），而
        `_leave_battle_zone_tick` 是**每拍**都会进的（tick 那道门 ✓）⇒ 每拍算整张名单必炸 ✗。
        ⚠ **算不出就 `None`**（没注入 / 全都走不到 / 只配了一块）⇒ 调用方退回 `zones[0]`
        = **老行为** ✓（宁缺勿错 ✓ 绝不猜一块 ✗）。
        """
        names = [str(z) for z in (zones or []) if str(z)]
        if len(names) <= 1:
            return names[0] if names else None            # 就一块 ⇒ 没什么好挑 ✓
        fn = getattr(self, "set_cost_of", None)
        player = getattr(ws, "player", None)
        if not callable(fn) or not getattr(player, "here_sets", None):
            return None                                   # 拿不到代价 ⇒ 不挑（退回老行为 ✓）
        best, best_c = None, None
        for n in names:
            try:
                c = fn(player, [n])
            except Exception:                             # noqa: BLE001 —— 解析器坏了别把归位弄停 ✗
                c = None
            if c is None:
                continue
            try:
                c = float(c)
            except (TypeError, ValueError):
                continue
            if best_c is None or c < best_c:
                best, best_c = n, c
        if best is not None:
            behavior.event("zone_pick", dst=best, cost=round(best_c, 1), n=len(names))
        return best

    def _leave_battle_zone_tick(self, now, ws, keys):
        """「不在战斗区域」那一拍：**不打架**，先下 / 接着跑回区域的路线 ✓。

        用户 2026-09-26 的要求："要进战斗任务时，检查现在是不是在配置的区域里，
        不是的话就下前往命令"（**先默认前往第一个配置的集合** ✓，更多策略他后续补 ✓）。

        ⚠ 为什么要在这里把路线也推一拍：这条规则是在攻击分支**之前**早退的 ⇒
          正常路径末尾那次 `_climb_tick` 到不了 ⇒ 不自己推就永远站着不动 ✗
          （和防掉线休息分支是同一个坑 ✓）。
        """
        zones = [str(z) for z in
                 (getattr(self.settings, "battle_zone_sets", None) or []) if str(z)]
        # 「还没在**回去的路上** ⇒ 下一条「前往」」（默认去**第一个**配置的集合 ✓）。
        # ⚠ 判据**只认 `self._climb`**（现在有没有人在走这条路），**不看 `self._route`**
        #   （2026-09-26 用户现场定）：`_route` 是"行程单上还没走的几步"，而**只有 `_climb`
        #   会去取用**；执行的人不在了，那几步就是**死账** —— 拿死账当"我已经在回去的
        #   路上了"的证据，就会一条命令都不下。实测（3 步路线 + 点「结束当前寻路」+
        #   此刻不在战斗区域）：角色**站着不动**，每拍只报"不在战斗区域，先回去" ✗。
        #   现在账上就算还留着死账也照常下回去的命令 —— `start_route` 会把 `_route`
        #   整个覆盖掉 ⇒ 死账顺手清掉 ✓（那几步行不成"从死账继续走"的半截状态）。
        if self._climb is None:
            # 重下要有间隔：解析不出来时每拍重下会把 note 刷屏、还反复造任务 ✗。
            # ⭐ CD 按「**目标那个区域项**」取 ✓（用户 2026-09-28："① 用目标区域项"✓）——
            #   目标就是下面那行的 `zones[0]`（默认回**第一个**配置的集合 ✓）。
            target = zones[0] if zones else ""
            if now - getattr(self, "_zone_goto_at", 0.0) >= self._zone_cd_s(target):
                # ⭐ **节流放行 ⇒ 这一刻才挑"代价最低的可战斗区"**（用户 2026-09-28 ✓）。
                #   ⚠ 位置**必须在节流之内** ✗：挑一轮 = 对名单里每块各一次 `path_cost`
                #     （一次 BFS ✓），而这个函数**每拍**都会进 ⇒ 放外面就是"每拍算一整轮"✗✗。
                #   ⚠ 挑不出来（没注入 / 都走不到）⇒ `None` ⇒ **保留 `zones[0]`**（老行为 ✓）。
                target = self._pick_battle_zone(ws, zones) or target
                self._zone_goto_at = now
                ok, msg = self.plan_and_start_route(
                    target, why="限制战斗区域：先回「%s」" % target)
                self._last_goto_note = (str(msg), now)
                behavior.event("zone_goto_ok" if ok else "zone_goto_fail",
                               dst=target)
        if self._climb is not None and not self._climb_tick(
                now, getattr(ws.player, "world_x", None), keys, ws):
            self._set_state("climb")
        return bool(self._climb)

    def _attack_state(self, target, best, mobs, ws):
        """攻击范围内有框时的状态选择（attack / 规避贴脸）。返回 keys。"""
        s = self.settings
        px = ws.player.x
        min_dist = s.min_attack_dist
        keys = set()

        # **挂着寻路任务时，战斗只许"站桩 attack"、不许走位**（2026-09-26 用户要求 0）：
        # 规避（后退 / 跳规避）会按方向键、还会跳 —— 那会把"对齐到绳的 x"整个推翻，
        # 两套逻辑抢方向键时人看到的就是"来回左右走"。攻击照打，只是不再挪窝。
        if self._climb is not None:
            if target.x > px:
                self.set_facing(1)
            elif target.x < px:
                self.set_facing(-1)
            self._set_state("attack")
            return keys

        # 贴脸判定走**盲区框**（用户 2026-09-27：攻击范围是矩形 ⇒ 盲区也是矩形 ✓）：
        # 只有"水平进了最小距离**而且**竖直也在上下距离之内"才算贴脸 —— 站在正上方
        # 平台上的怪不该触发规避 ✗（`_in_blind_box` 一处实现 ✓）。
        # `min_dist = 0` ⇒ 盲区是空集 ⇒ 永远 False（= 老行为 ✓）。
        target_too_close = bool(min_dist > 0 and self._in_blind_box(target, ws.player))
        any_too_close = bool(min_dist > 0 and
                             any(self._in_blind_box(m, ws.player) for m in mobs))

        if s.evade_type == "jump":
            if target_too_close:
                state = "evade_back_jump"
            elif any_too_close:
                state = "evade_jump"
            else:
                state = ("evade_jump"
                         if min_dist > 0 and random.random() < s.jump_random_prob
                         else "attack")
        else:  # back 后退
            if any_too_close:
                state = "evade"
                keys.add(self._resolve_seq_key("back"))
            else:
                state = "attack"

        if state == "attack":
            # **攻击范围内有目标 ⇒ 站桩输出：一个方向键都不按。**
            #
            # 为什么不能"按住朝目标的方向键，确保面向它"（老写法，2026-09-26 用户报的
            # 就是它）：这类游戏**按住方向键 = 走**，于是两次输出之间角色一直在朝怪挪，
            # 挪到身上、出范围又回头 → 来回抖（用户原话：希望只要攻击范围内有目标就
            # 站桩输出）。
            #
            # 而且按方向键对"面向"**毫无帮助**：`_in_range` 的判据里已经带了
            # `(m.x - px) * facing >= 0` —— 能进这个分支的怪，本来就在当前朝向的
            # 正前方。真正需要转身的情况（怪在背后）走的是另一条路：`_locked_target`
            # + 回身输出序列，那里按 `min_turn_hold_ms` 兜着，才是"刚转向"的例外。
            #
            # 只同步**内部朝向**、不按键：游戏里的朝向跟着"最后按的方向键"走，把内部
            # 朝向贴住实际面向，免得下一帧 `_in_range` 拿错朝向把背后的框算进范围。
            if target.x > px:
                self.set_facing(1)
            elif target.x < px:
                self.set_facing(-1)

        self._set_state(state)
        return keys

    def _set_state(self, new_state):
        """切换状态；处理回身输出序列的进入/退出清理（残留键要松开）。"""
        if new_state == "evade_back_jump" and self.state != "evade_back_jump":
            self._back_ctx = None   # 进入时重置（执行时按 back_jump_seq 初始化）
        elif new_state != "evade_back_jump" and self.state == "evade_back_jump":
            self._release_ctx(self._back_ctx)
            self._back_ctx = None
        _prev_state = self.state
        self.state = new_state
        if new_state != _prev_state:            # 战斗状态切换（抖动/攻击↔追击一眼可见 ✓）
            behavior.event("battle_state", frm=_prev_state or "-", to=new_state)

    def _best_behind_target(self, mobs, ws):
        """**背后（反侧）攻击范围内**、离得最近的那只怪（没有 ⇒ `None` ✓）。

        ⭐⭐ 用户 2026-09-28 ✓ 原话："锁定怪在前面，**背后攻击范围内有怪**，chase 想朝前，
          但其实**转向更高效** ⇒ 是不是做成**背后怪抢锁**更好？" —— 就是给那个抢锁用的 ✓。
        为什么"抢锁"是对的：背后那只**现在就能打**（转身即可 ✓），而当前锁定那只可能还要
        **走过去** ⇒ 先转身打掉眼前这只，比一路朝前追更省时间 ✓。

        ⚠ 判据 = `_in_attack_box(m, player, **-self.facing**)` ✓（**反侧**的单侧判定 ✓）——
          这正是那个 `facing` 参数该用的地方 ✓（`_in_range` 要 `+self.facing`=正面 ✓，
          这里要**背面** ✓）。
        ⚠ 正对上（`dx == 0`）两边都算 ⇒ 会和"正面那组"重叠 ✓（无害：抢锁那里还比了 `id` ✓，
          同一只怪不会自己抢自己 ✓）。
        """
        _f = -1 if int(self.facing) >= 0 else 1
        cands = [m for m in mobs if self._in_attack_box(m, ws.player, _f)]
        if not cands:
            return None
        return min(cands, key=lambda m: self._center_dist(m, ws.player))

    def _locked_target(self, lockable, ws, now):
        """chase 用的锁定目标：target_cd 机制 ＋ ⭐ **背后怪抢锁**（2026-09-28 起 ✓）。返回 (target, best)。

        ⭐ **抢锁规则**（用户 2026-09-28 ✓）：锁定还在期内，只要**背后（反侧）攻击范围内有怪**
          ⇒ **改锁它** ✓（那只现在就能打 ⇒ 转身比继续朝前追更高效 ✓，见 `_best_behind_target` ✓）。
          ⚠ 只抢"**真的够得着**"的 ✓；抢完**照常续期** `_target_until` ✓（防来回跳 ✗）。

        ⚠ `best` 是**画面距离**（下游「追击起跳」的区间 / 打点都用它 ✓）——
          虽然挑谁按「寻路距离」排（`_nearest` ✓），但返回的距离口径没变 ✓。
        """
        target = None
        if self._target_id is not None:
            for m in lockable:
                if m.id == self._target_id:
                    target = m
                    break

        if target is not None and now < self._target_until:
            # ⭐⭐ **背后怪抢锁**（用户 2026-09-28 ✓ 原话："锁定怪在前面，背后攻击范围内有怪，
            #   chase 想朝前，但其实转向更高效 ⇒ 是不是做成背后怪抢锁更好？"✓）。
            #   条件：**背后（反侧）的攻击范围内有怪** ✓ ⇒ 那只**现在就能打**（转身即可 ✓），
            #   而当前锁定那只可能还要走过去 ⇒ **抢锁更高效** ✓。
            #   ⚠ **必须"真的够得着"才抢** ✓（`_best_behind_target` 用的就是 `_in_attack_box`
            #     反侧判定 ✓）—— 否则"随便背后有只远怪"也会被抢过来，反而更慢 ✗。
            #   ⚠ 抢锁**要按正常 target_cd 续期** ✓（`_target_until = now + cd` ✓ 别绕过它 ✗，
            #     否则会被反复抢 ⇒ 目标来回跳 ✗）；同一只怪**不重复抢** ✓（比 `id` ✓）。
            _behind = self._best_behind_target(lockable, ws)
            if _behind is not None and _behind.id != target.id:
                target = _behind
                best = self._center_dist(_behind, ws.player)
                self._target_id = target.id
                self._target_until = now + self._random_target_cd()
                self._no_target_since = None
                behavior.event("target_steal", mob=self._target_id,
                               why="背后攻击范围内有怪 ⇒ 抢锁转身打它")
            else:
                best = self._center_dist(target, ws.player)
                self._no_target_since = None
        else:
            target, best = self._nearest(lockable, ws.player, ws=ws)
            if target is not None:
                self._target_id = target.id
                self._target_until = now + self._random_target_cd()
                self._no_target_since = None
            else:
                self._target_id = None
                self._target_until = 0.0
                #: ⛔ 这里原来初始化「锁定目标所在的 foothold 集合」缓存（`_target_sets` ✗）——
                #: 它只服务"锁定框下方那行集合名"，用户 2026-09-27 否掉那行 ⇒ **缓存已删** ✗
                #: （`_nearest` / `_chase_goto_if_elsewhere` 里的两处写入也一起删了 ✓）。
                #: ⚠ 界面要看"怪在哪块集合"现在**只有**「查过的怪框」那一行：
                #: `live_thread` 读 `queried_mob_boxes()`（数据来自 `mob_sets_of` 的记账 ✓），
                #: 绘制层依旧**不许**自己扫 foothold ✗（纪律见 `_make_mob_sets_resolver` 的 ⚠）。
        return target, best

    def _reap_ghost_mobs(self, target, mobs, ws):
        """输出行为触发的防呆：把「攻击范围内的防抖幽灵框」标记为待消除。

        背景：MobTracker 会为短暂漏检的高置信度框保留位置（missed > 0 = 幽灵框），
        决策层不至于因一帧漏检丢目标。但对幽灵框继续输出就是空放技能（怪其实
        已经消失）。所以一旦进入输出，就把当前目标、以及攻击范围内（最小攻击
        距离 ~ 最大攻击距离）的所有幽灵框都记进 _kill_mobs；上层（live_thread）
        每帧取一个调 mob_tracker.kill() 立即消除该轨迹。

        为什么不止当前目标：只消当前那个的话，范围内的其他幽灵框下一帧会顶上
        来当目标，继续空放；一次性清掉更干脆。
        """
        if target is not None and getattr(target, "missed", 0) > 0:
            self._kill_mobs.add(target.id)
        s = self.settings
        lo = max(0.0, float(s.min_attack_dist))
        hi = float(s.attack_dist)
        for m in mobs:
            if (getattr(m, "missed", 0) > 0
                    and lo <= self._center_dist(m, ws.player) <= hi):
                self._kill_mobs.add(m.id)

    def _run_output_ctx(self, now, ctx, seq, cd, tag="out"):
        """推进一条输出序列（攻击输出 / 回身输出）；跑完按输出CD排下一轮。

        决策拍（`tick`）和时序拍（`tick` 的 timing_only）都调它 —— 两边的推进逻辑
        必须一模一样，否则「什么时候发下一步」会按下发路径分叉。
        """
        r = self._run_seq(now, ctx, output=True)
        if r is not None:
            return r
        due = self._next_output_ts(now, cd)
        # 打点：两轮输出之间实际隔了多久（上一轮开始 → 这一轮开始）。**这里**才是
        # 「重新排期」的唯一位置，「CD=0 为什么不立即输出」看这个分布就有答案。
        if self._last_output > 0.0:
            perf.sample(tag + "_gap_ms", (now - self._last_output) * 1000.0)
        perf.count(tag + "_round")
        # ⭐ **站桩输出的第几轮**（用户 2026-09-28 要求 2："站桩输出时**每 3 次 attack** 补一个
        #   朝向目标的方向键，持续「最小切换朝向时间」"✓）。
        #   为什么**必须**记在这儿：这里才是"跑完一轮 ⇒ 重新排期"的**唯一位置** ✓
        #   （别记在 `_output_actions` 里"`_output_ctx is None`"那一刻 ✗ —— 那个条件只在
        #    **第一次**成立，因为序列跑完返回的是**新 ctx** 而不是 None ⇒ 计数会恒为 1 ✗，
        #    实测踩过 ✓）。`tag == "out"` 才算"一次 attack"（`back` 是回身输出那条 ✓）。
        #   ⚠ 清零在 `_output_actions` 开头（`state != "attack"` ⇒ 退出站桩清零 ✓）。
        if tag == "out":
            self._atk_round += 1
            # ⚠ **不许和上一次补键的窗口重叠**：`min_turn_hold_ms`（用户配 1500ms）比重排期
            #   周期长时（`attack_cd` 小 / 序列短 ⇒ 每 3 轮可能只有几百毫秒 ✗），窗口会
            #   **首尾相接** ⇒ 表现出来就是"方向键一直按着不放"✗（实测：`attack_cd=0` 时
            #   2 秒里按了 1.67 秒 ✗）。⇒ 上一次补键没结束就不再起新的 ✓
            #   （实际频率 = max(每 3 轮, `min_turn_hold_ms`) ✓ —— 想让"每 3 轮"真的按得上，
            #    就把「最小切换朝向时间」配得比"3 轮"短一些 ✓）。
            _hold_s = max(0.0, float(self.settings.min_turn_hold_ms)) / 1000.0
            if (self._atk_round % STATION_TURN_EVERY == 0
                    and (self._turn_until is None or now >= self._turn_until)):
                # 该补的那一轮 ⇒ 记下起点（tick 末尾据此补"朝目标"的方向键、按满
                # 「最小切换朝向时间」✓ 见 `_turn_due_at` 的说明 ✓）
                self._turn_due_at = now
                self._turn_until = now + _hold_s
        nxt = [seq, 0, due, set(), []]
        if due <= now:
            # 序列本身比 CD + 随机延迟还长 → 同一帧直接起步，不白等一个 tick
            r = self._run_seq(now, nxt, output=True)
            if r is not None:
                return r
        return nxt

    def _advance_running_seqs(self, now):
        """只推进「已经在跑」的输出序列（该不该起新的、跑哪条，由决策拍按状态决定）。"""
        s = self.settings
        cd = max(0.0, float(s.attack_cd)) / 1000.0
        if self._output_ctx is not None and self.state == "attack":
            self._output_ctx = self._run_output_ctx(now, self._output_ctx,
                                                    s.output_seq, cd)
        if self._back_ctx is not None and self.state == "evade_back_jump":
            self._back_ctx = self._run_output_ctx(now, self._back_ctx,
                                                  s.back_jump_seq, cd, tag="back")

    def next_deadline(self):
        """下一个「到点该发」的时刻（monotonic）；没有任何序列在跑时返回 None。

        主循环用它决定这一轮等多久：到点就醒来跑一次时序拍（`tick_timing`），
        序列计时因此不再被抓帧节拍量化。

        只看序列上下文（输出 / 回身输出 / 进入退出隐身 / 自定义定时行为）——
        防掉线排期、喂宠这些是分钟级的，精度要求低，跟着画面帧走就够，
        不值得为它们把主循环叫醒。
        """
        times = []
        for ctx in (self._output_ctx, self._back_ctx, self._afk_ctx):
            if ctx is not None:
                times.append(ctx[2])
        for st in self._timer_states.values():
            if st is not None:
                times.append(st[2])
        return min(times) if times else None

    def tick_timing(self, ws=None):
        """时序拍：把「已经排好期、到点该发」的东西发出去（不依赖画面帧）。

        见 `tick` 的 timing_only 参数。ws 只是为了兜底（这一路不读世界状态）。
        """
        return self.tick(ws, timing_only=True)

    def _next_output_ts(self, now, cd):
        """下一次输出动作的最早时刻（monotonic）。

        以「上次**输出行为**的开始时刻」(_last_output) 为锚点，而不是「序列走完的
        时刻」、也不是「某个输出按键按下的时刻」—— 层级是 攻击状态 > 输出行为 >
        输出按键，输出CD 属于中间那一层（见 `_run_seq` 的 output 参数）。
        序列从输出键按下到整条走完还有一段时间（后续元素的按键间隔 + 主循环
        tick 粒度），原来用 `now + cd` 会把这部分**叠加**在 CD 之上 —— 实测
        3 元素序列、15fps、CD=200ms 时，两次输出隔了 400ms（多出 3 个 tick）。

        改成从 _last_output 起算后，attack_cd 就是「两次输出之间的最小间隔」，
        序列自身的耗时被 CD 吸收；序列比 CD 长时以序列为准（max 里的 now），
        绝不会让两轮序列重叠。
        ⚠ 末尾原来还叠一点随机延迟做"人工化抖动"，**2026-09-27 已删** ⇒ 现在就是
        `max(now, 上次输出 + CD)`，同一份配置的节奏可复现 ✓。
        """
        return max(now, self._last_output + cd)

    def _output_actions(self, now, target, mobs, ws):
        """按当前状态输出动作：attack 连点 / evade_jump 跳+输出 / evade_back_jump 序列。"""
        s = self.settings
        attack_cd = max(0.0, float(s.attack_cd)) / 1000.0
        attack_key = s.keymap["attack"]

        # ⭐ **退出站桩 ⇒ 轮次清零**（用户 2026-09-28 明确："只要退出站桩计数清零"✓）。
        # 收在**一处**：无论从哪条路退出站桩（怪没了 / 关自动 / 未定位 / 进规避 / 去追怪），
        # 都在这里统一清 ✓ —— 别在各分支里各清一遍（漏一处就是"上次的轮次带到下一次打架"✗）。
        if self.state != "attack":
            self._atk_round = 0

        # 输出行为触发的防呆：攻击范围内的防抖幽灵框标记待消除（各输出状态通用，
        # 规避时也在输出，同样会空放）
        self._reap_ghost_mobs(target, mobs, ws)

        # 转向后输出延迟：刚换完朝向时角色的转身动作还没做完，这时候打出去的方向
        # 是错的。所以**新开一轮输出**要往后推；已经在跑的序列让它跑完（半截掐断
        # 会留下按着的键）。
        turn_wait = ((now - self._turn_at)
                     < max(0.0, float(s.turn_output_delay_ms)) / 1000.0)

        if self.state == "attack":
            # 执行输出行为序列（output_seq），走完按 attack_cd 排下一轮
            if self._output_ctx is None and not turn_wait:
                # **进攻击状态也要走输出CD排期**：不能一进范围就立刻开一轮，否则
                # 状态在攻击/追击之间抖一下（目标丢一帧又回来）就会比 CD 密。
                # 和另外两处（序列走完、回身输出）用同一个排期函数 —— CD 已经过时
                # （比如刚从追击/待机过来）它会返回「现在 + 随机延迟」，一样是立刻打，
                # 不会变迟钝。
                self._output_ctx = [s.output_seq, 0,
                                    self._next_output_ts(now, attack_cd), set(), []]
                # ⚠ **轮次计数不在这里**（这里写的 `_output_ctx` 只是"第一次排期"✗）——
                #   真正"每完成一轮就重新排期"的位置是 `_run_output_ctx` 里那句
                #   `perf.count(tag + "_round")` ✓，计数点放在那儿 ✓（见那处注释 ✓）。
                # ⭐ **轮到"该补朝向键"的那一轮** ⇒ 记下它的开始时刻（见 `_turn_due_at` ✓）：
                #   接下来「最小切换朝向时间」这段时间里，tick 末尾会把"朝目标"的方向键补上 ✓
                #   （用户 2026-09-28 要求 2："站桩输出时**每 3 次 attack** 补一个朝向目标的
                #    方向键，持续「最小切换朝向时间」"✓）。
                if self._atk_round % STATION_TURN_EVERY == 0:
                    self._turn_due_at = now
            # _output_ctx 还是 None = 刚转向、这一帧先不输出（序列一旦开跑就让它跑完，
            # 半截掐断会留下按着的键）
            if self._output_ctx is not None:
                self._output_ctx = self._run_output_ctx(now, self._output_ctx,
                                                        s.output_seq, attack_cd)
        else:
            # 离开攻击状态：重置输出序列 + 松开残留键
            self._release_ctx(self._output_ctx)
            self._output_ctx = None

        if self.state == "evade_jump":
            jump_key = s.keymap.get("jump")
            interval = max(0.0, float(s.jump_interval)) / 1000.0
            if now >= self._next_evade:
                if jump_key:
                    tap(jump_key, self._attack_duration)
                    behavior.event("jump", why="闪避")
                self._pending_attack = now + interval
                # （随机延迟已删，2026-09-27 ⇒ 规避间隔 = interval + CD ✓）
                self._next_evade = now + interval + attack_cd
            if (self._pending_attack is not None and now >= self._pending_attack
                    and not turn_wait):
                # 输出这一下也受「转向后输出延迟」约束：跳可以先跳（是位移，
                # 不吃朝向），但攻击键要等转身做完，否则打向错误的方向。
                # 没到点就继续挂着 _pending_attack，下一帧再补。
                tap(attack_key, self._attack_duration)
                behavior.event("out_evade")
                self._last_output = now
                self._pending_attack = None

        if self.state == "evade_back_jump":
            if self._back_ctx is None and not turn_wait:
                self._back_ctx = [s.back_jump_seq, 0, 0.0, set(), []]
            if self._back_ctx is None:
                return                   # 刚转向：先不输出（回身输出同样吃朝向）
            # 序列走完由 _run_output_ctx 按输出CD排下一轮（回身输出是循环行为）
            self._back_ctx = self._run_output_ctx(now, self._back_ctx,
                                                  s.back_jump_seq, attack_cd,
                                                  tag="back")

    def _take_kill_mobs(self):
        """取出并清空待消除的幽灵框 id 列表（tick 各返回路径统一带上）。

        一帧全清：不只当前目标，攻击范围内的防抖幽灵框都在这批里。
        """
        if not self._kill_mobs:
            return []
        out = sorted(self._kill_mobs)
        self._kill_mobs.clear()
        return out

    def _periodic_reset(self, now, s):
        """定期体检指令通道 + 清一次固件侧按键（防卡键）。停自动之后也要跑。

        卡键有两个来源，要分开对付：

        1. **某条 RELEASE 在路上丢了** → 固件还按着，而我们本地以为已经松开、
           之后再也不会补发（`_release_held_keys` 只释放「它记得按下的键」）。
           这种靠定期 RELEASEALL 兜底 —— 固件一次性清空，不依赖我们记没记住。
        2. **通道整个死了** → 后面所有命令（包括我们自己补发的 RELEASEALL）
           都到不了，还会一直假装发成功。这种必须**重连**：断开这个动作会让
           relay 往串口直接写一条 RELEASEALL（见 remote_kbd/relay.py），
           那条路不依赖我们的 TCP 还通不通。上次「只能关 GUI 才好」就是因为
           GUI 关闭时连接断了、relay 替我们松的键。

        放在 tick 的最前面（`if not s.enabled` 早退之前）：人发现卡了第一反应
        就是关自动，如果关自动就不跑这个，自愈机制在真正需要它时正好是关着的。
        """
        from decision import input as dinput
        # ---- 0. 按键状态重同步请求（「重置指令通道」按钮 / 手动输入按了键）----
        # 只清本地按键状态、**不发 RELEASE**，两个原因：
        #   · 清掉之后下一帧 keys.set 会重新发 PRESS（固件 addHeld 会去重，不会重复按）
        #   · 手动输入每按一个键都会请求一次，发 RELEASE 会让自动正按着的移动键
        #     一格一格闪断
        # 序列上下文故意不在这里清：清掉而不释放，序列里按着的键就卡住了。
        # 序列的收尾交给定期 RELEASEALL（它先发 RELEASEALL 再清上下文）。
        if s.input_resync:
            s.input_resync = False
            self.keys.clear()
        # ---- 1. 通道体检：10 秒一次。很便宜，就是看一眼上次发送的结果。----
        if now - self._link_checked >= 10.0:
            self._link_checked = now
            h = dinput.link_health()
            ok = bool(h.get("ok", True))
            s.input_link_ok = ok
            s.input_link_err = "" if ok else ("%s %s" % (h.get("backend", ""),
                                                        h.get("err", ""))).strip()
            # 不健康就重连。重连会阻塞（建连有超时），所以丢到后台线程 ——
            # 决策循环不能被一次网络重试卡住几秒。
            if not ok and now - self._link_retry >= 10.0:
                self._link_retry = now
                threading.Thread(target=dinput.reconnect_remote, daemon=True).start()
        # ---- 2. 定期 RELEASEALL ----
        iv = max(0.0, float(s.resetall_interval))
        if iv <= 0 or now < self._next_resetall:
            return
        self._next_resetall = now + iv
        # 手动输入开着 → 跳过这次。手动输入现在允许和自动同时开，RELEASEALL 会把
        # 人正按着的手动键一起松掉。跳过是一整个周期（不是每帧重试）。
        try:
            from decision import manual_input
            if manual_input.active():
                return
        except Exception:
            pass
        # 怎么"松"取决于后端：
        #   · 远程 / 串口 —— 固件那边一次 RELEASEALL 把所有键清掉，最可靠
        #     （逐键 RELEASE 可能丢在网络上）。本地只需同步**记录**（clear 故意不发 up）。
        #   · 本地(仅测试) —— **根本没有固件**，release_all_remote() 在那种后端上是
        #     空操作（见 decision/input.py：`if _remote is not None`）。这时若也只
        #     clear 记录，那个键就永远按着了：记录已清，后面 keys.set 再也不会补发 up
        #     （实测：本地后端下只看到 down left，永远看不到 up left）。
        if dinput.link_health().get("backend") == "local":
            self.keys.release_all()
        else:
            dinput.release_all_remote()
            self.keys.clear()
        # **本机规划的行为序列一律保留进度**，并且把「该按着的键」**重按回去**：
        #     _output_ctx     输出行为序列（攻击连点 / 跳输出，自己按 CD 排下一轮）
        #     _back_ctx       回身输出序列（循环行为）
        #     _afk_ctx        进入 / 退出隐身（里面可能有很长的 delay）
        #     _timer_states   自定义定时行为
        # RELEASEALL 松的是固件侧按着的键；「走到第几步、下次什么时候发」是本机
        # 记的进度，和它无关。原来把这些上下文一起清掉，等于每 resetall_interval
        # 秒让所有序列从头重来：长 delay 永远走不完、前面的动作被反复重放、
        # 输出序列还会无视 CD 立刻重打一轮。
        # ⚠ 但**光"作废记录"还不够**（2026-09-26 用户口径）：这些宏 / 寻路可能很长，
        #   序列里按着的键往往还要再按一会儿（`down ↑` 后面跟 3 秒 delay ⇒ 这 3 秒里
        #   ↑ 必须一直按着 ✓）。RELEASEALL 把它松了，记录又清了 ⇒ **没人补按** ⇒
        #   角色中途"松手" ✗（用户的原话："按着 ↑ 结果被停了，需要重启被清掉的键"）。
        #   ⇒ 走 `_reassert_ctx`：**重按 + 保留记录**（下次再清还能再按一遍 ✓）。
        for ctx in (self._output_ctx, self._back_ctx, self._afk_ctx):
            self._reassert_ctx(ctx)
        for st in self._timer_states.values():
            self._reassert_ctx(st)

    def tick(self, ws, timing_only=False):
        """跑一帧决策；`timing_only=True` 时只跑一次「时序拍」。

        **两种拍子的分工**（同一条线程，主循环按需调用）
            决策拍 `tick(ws)`   每帧一次：选目标 / 移动 / 状态机 / 喝药 / 排防掉线 ——
                                这些依赖世界状态。它同时也会推进序列（不变）。
            时序拍 `tick_timing()` 只在「等下一帧超时且序列到点」时跑：把已经排好期
                                的动作发出去 —— 序列元素、休息状态机、喂宠、定时行为。
                                不做任何画面相关决策（选目标/移动/喝药都不在这里）。

        **为什么要分**：序列里存的是**绝对到点时刻**（`next_ts = now + delay`），
        但原来只有抓到一帧时才检查「到点没」→ 到点被推迟到下一帧，输出CD、delay、
        跳间隔全被量化成 1/capture_fps（15fps → 66.7ms）。计时本来就该走本机绝对
        时钟，所以由主循环按 `next_deadline()` 精确唤醒。

        ws: WorldState（时序拍不读它，但为了兜底仍要求传上一次的世界状态）。
        """
        s = self.settings

        # 开启自动的上升沿：重置朝向监控计时。否则关掉自动后隔很久再开，
        # 会沿用旧的「最后一次朝向变化」时刻，被误判超时、把自动立刻停掉。
        if s.enabled and not self._was_enabled:
            self._last_facing_change = time.monotonic()
            self._player_lost_since = None   # 重新开启自动：重置找不到玩家计时
        self._was_enabled = s.enabled
        now = time.monotonic()

        # ---- 「任务这一拍没跑 ⇒ 报信」的**唯一结算点**（结构性收口，2026-09-26）----
        # 放在 tick **最前面**（任何早退之前）⇒ 以后再加早退分支也不可能漏 ✓
        #（原来是手工在 `if in_range:` / 休息赶路打架两处各调一次，结果"未定位玩家"和
        #  "关自动"两条早退漏了 ⇒ 任务被冤判"走不动了"，见 `_task_settle` 的实测）。
        # `_task_ran` 每拍清一次：这一拍谁推了任务（`_climb_tick`）谁盖章 ✓。
        # ⚠ **时序拍跳过**：它本来就不推任务，不是"任务被跳过"；参与会把寻路超时钟
        #    每 10ms 重打一次 ⇒ 超时永不触发 ✗（同见 `_task_settle`）。
        if not timing_only:
            self._task_settle(now)
            self._task_ran = False

        # 通道自愈 + 定时 RELEASEALL。**必须放在下面 `if not s.enabled` 早退之前**：
        # 卡键最常见的时刻恰恰是「发现卡了 → 去关自动」之后。如果关自动就不发，
        # 这个自愈机制在真正需要它的时候正好是关着的 —— 之前就踩了这个坑：
        # 一关自动就再也救不回来，只能关 GUI（那一下是 relay 在连接断开时
        # 替我们发的 RELEASEALL 救的，见 decision/input.py 的 reconnect_remote）。
        self._periodic_reset(now, s)

        # ---- 「画面里看不到角色」的留痕：**必须放在下面关自动的早退之前** ----
        # 这条是 2026-09-27 现场补的（角色被卡住打死，之后几小时的日志里
        # 只剩"定时 RELEASEALL"一条按键，看不出是死了还是卡了）—— 见该函数说明。
        if not timing_only:
            self._watch_player_visible(now, s, ws)

        if not s.enabled:
            if timing_only:
                return None      # 停自动的收尾由决策拍做（这里做的事都是幂等的，不必重复跑）
            # **「关自动」顺带把挂着的寻路任务也撤掉**（用户 2026-09-26 决定：
            #   "关自动 = 停手"，留着一个走到一半的任务只会让人以为它还在执行 ——
            #   界面上那行「当前任务」也就该回到"战斗"）。要重新走就再点一次
            #   「命令前往」（或「结束当前寻路」——那个按钮保留，用来只停任务不停自动）。
            # ⚠ 必须在**早退之前**做：这一拍之后整个函数就返回了，放到后面永远不会被调
            #   （而且这条早退每个决策拍都走一遍，不清掉就是"任务永远冻在半路"✗）。
            # ⚠ 为什么不是"留着任务、只重置计时"：那需要给 `_task_settle` 再开一条
            #   "关自动期间不算数"的例外，等于把刚收口的东西又拆开 ✗。
            #   `stop_route` 在没有任务时是空操作（`stop_climb` 自己判 None ✓），
            #   所以每拍调一次也刷不出日志 ✓。
            self.stop_route("关自动")
            # 释放所有按着的键：不仅 KeyState，还有序列（回身输出/防掉线/定时）残留的，
            # 否则停自动时若正处于序列中间，序列按下的键会卡住继续生效。
            self._release_held_keys()
            self.state = "idle"
            self._rest_pending = False   # 停自动一并取消「待休息」，重新开启后重新计时
            s.rest_state = ""            # 停自动也清掉休息状态（UI 按钮跟着变灰）
            s.rest_abort = False
            s.rest_request = False       # 停自动也丢掉「手动休息」的请求
            s.rest_until_monotonic = 0.0
            s.rest_pending = False
            s.next_afk_monotonic = 0.0
            return {"state": "idle", "reason": "未开启",
                    "kill_mobs": self._take_kill_mobs()}

        # 防掉线-隐身休息：休息流程独占本帧 —— 不走 RELEASEALL / 喂宠 / 自定义定时
        # （计时器由 _run_rest → _pause_timers 冻住），也不做任何战斗动作。
        #
        # 唯一例外是**自动喝药**：隐身不等于安全（隐身到期、被范围技能扫到、
        # 血本来就没满），血量掉了照样要补。注意正常路径里喝药在「定位到玩家」
        # 之后才跑，而休息时角色是隐身的、很可能定位不到，所以这里必须显式补一次。
        # ⚠ **休息与寻路互相独立**（用户 2026-09-26 明确要求，我上一版擅自耦合过 ✗）：
        #   休息的分支**不判断寻路**（不跳过、不改休息状态、不存/不恢复）；
        #   寻路任务那一侧也**不判断是否在休息**（`_climb_tick` 里没有任何休息条件）。
        #   两边各按自己的时刻表走 —— 同时挂着的处理见下面（各按各的键）。
        # ⚠ 状态清单走 `REST_STATES`（模块级常量）：加新休息类型时**只改那一处**，
        #   漏了就是"用户选了却什么都不发生 ✗"（`spot_rest` 就是这么补上的）。
        # ⭐ **位置 / 血量 / 动作 的快照**（用户 2026-09-28 要求 ✓ 见 `_snap_beat`）——
        #   ⚠⚠ 必须挂在**休息分支之前**：上次卡了 2 小时 22 分那时，状态恰恰是「定点休息」
        #   的 `afk_spot_walk`（属休息分支 ✓）⇒ 挂在常规路径上**一条都打不出来** ✗。
        self._snap_beat(now, ws)
        if self.state in REST_STATES:
            if not s.anti_afk_enabled:
                # 休息途中关掉防掉线：立刻退出休息，别把角色留在隐身里
                self._release_ctx(self._afk_ctx)
                self._afk_ctx = None
                self._finish_rest(now)
            else:
                # 手动「结束休息」：
                #  · 隐身那一型**不能直接跳回打怪** —— 角色还在隐身里，半截的进入序列
                #    也可能按着键 ⇒ 松掉当前序列、转去执行退出隐身行为，走完再 _finish_rest；
                #  · 定点那一型身上没有隐身（只可能在走路 / 演到达行为）⇒ 停掉路线直接收工，
                #    并把这一拍结束掉（别让它接着往下走一步）。
                if s.rest_abort and self.state != "afk_exit":
                    s.rest_abort = False
                    self._release_ctx(self._afk_ctx)
                    self._afk_ctx = None
                    if self.state in SPOT_STATES:
                        # ⭐⭐ 手上那一段循环**还在演** ⇒ **先不收工**（用户 2026-09-28 的口径：
                        #   "结束休息**不应该打断**正在进行的循环" ✓）—— 转成"**收尾中**"：
                        #   留在休息分支里把它推完（`_loop_drain` ⇒ **不再起跑下一项** ✓），
                        #   演完那一拍才 `_finish_rest` ✓。
                        #   ⚠⚠ **为什么不能"立刻收工、让它在后台演完"**（原来那套 ✗）：
                        #     回战斗之后只要**没怪**，"平地巡逻无怪 idle"那条就会**每拍**调
                        #     `_release_combat_keys()` ⇒ `_stop_spot_loop()` ⇒ 手上那一段
                        #     **当场被掐断** ✗（而且不留痕，极难查 ✓ —— 这次就是它 ✗）。
                        #   ⇒ 结论：**要"演完"，就必须留在休息状态里等** ✓。
                        if self._loop_ctx is not None:
                            self._loop_drain = True     # 演完手上这段就收（不起跑下一项 ✓）
                            self._loop_next_ts = 0.0    # 别再排下一轮 ✓
                            self._rest_stop_pending = True
                            self._rest_until = 0.0      # 视为"时间已到" ⇒ 下一拍走收尾 ✓
                        else:
                            self.stop_route("手动结束休息")
                            self._finish_rest(now)
                            return {"state": self.state, "reason": "手动结束休息",
                                    "target": None, "dx": 0, "dist": 0, "keys": [],
                                    "facing": self.facing,
                                    "kill_mobs": self._take_kill_mobs()}
                    else:
                        self.state = "afk_exit"
                self._run_rest(now, ws)
                if timing_only:
                    return None      # 休息期间不做战斗动作，也不读世界状态
                # 休息期间照常补血。勾了「被打断重试」时，**这一拍真的补了血就算被打断**：
                # 立刻收工回去接着打（见 _interrupt_rest）。
                # ⚠ 哪些阶段算「能被打断」由**契约表**说了算（`REST_STATES_INTERRUPT`）——
                #   别再手写元组（`afk_spot_act` 那种漏一次就是"补了血却继续歇着"✗）。
                if self._drink_potions(ws, now) and s.anti_afk_retry_on_interrupt \
                        and self.state in REST_STATES_INTERRUPT:
                    self._interrupt_rest(now)
                # **寻路任务照跑 + 赶路时"有怪先打"**（见 `_rest_travel_beat`）：
                # 前者是用户 2026-09-26 的要求（"即使在休息中，寻路也需要能生效"）✓；
                # 后者是同一天报的坑（"点手动进入休息后…攻击范围内有怪时没有触发 attack"
                # ✗）—— 「命令前往」走的是正常决策路径、那里有"有怪先打"的仲裁 ✓，
                # 而休息分支是**早退**的（除了补血不做任何战斗动作）⇒ 定点休息一路挨打 ✗。
                # ⚠ 必须紧跟一次 `self.keys.set(...)` —— 休息分支是**早退**，正常路径末尾那次
                # 下发（`self.keys.set(keys)`）到不了这儿，不补下发的话按的键发不出去 ✗。
                _rk = self._rest_travel_beat(now, ws)
                self.keys.set(_rk)
                return {"state": self.state,
                        "reason": ("定点休息" if self.state in SPOT_STATES else "隐身休息"),
                        "target": None,
                        "dx": 0, "dist": 0, "keys": sorted(_rk),
                        "facing": self.facing,
                        "kill_mobs": self._take_kill_mobs()}
        elif self.state not in REST_STATES:
            # 不在休息中：丢掉过期的「结束休息」请求，否则下一次休息一开始就会被它结束掉。
            # ⚠ 判断用"状态"而不是上面那个 `if` 的反面：休息中挂着寻路任务时上面那个 `if`
            # 是不成立的（那一拍让给任务），但**人点的「结束休息」不能被顺手丢掉**。
            s.rest_abort = False

        # 不依赖玩家定位的定时行为：到点就执行（自定义定时行为）
        # （「自动喂宠」2026-09-26 已整块移除 ⇒ 这里只剩自定义定时行为 ✓）
        self._custom_timers(now)

        if timing_only:
            # 时序拍到这儿就够：剩下的全要世界状态（选目标 / 移动 / 喝药 / 排防掉线）。
            # 序列只推进「已经在跑」的那些 —— 该不该起新序列由决策拍按状态决定。
            self._advance_running_seqs(now)
            return None

        # ⭐⭐ 「找不到玩家」的判据 = **小地图找不到黄点**（用户 2026-09-28 要求 ✓ 原话：
        #   "设置里的保护与诊断：**找不到玩家停止自动的判断依据修改为小地图找不到黄点**"）。
        #   ⚠ 以前看的是 `ws.player.found` —— 那是"**主画面里认没认出人物框**" ✗，与"有没有
        #     可用的位置"**不是一回事**：框好好的、但小地图那块**黄点糊了 / 被挡住 / 没识别出来**
        #     ⇒ 照样**没有可用位置** ⇒ 该算"找不到玩家" ✓（反过来，框没认出但黄点还在 ⇒
        #     位置照样可用、不该停 ✗）。
        #   · **有可用黄点** ⇔ `world_x` 有值 **且** 这一拍**不是**"沿用上一帧"；
        #     `world_held` 就是小地图那边给的标记："**这一拍没认出黄点**、位置沿用上一帧"
        #     （见 `perception/minimap.apply_to_player` ✓）⇒ 它置上也算"没找到" ✓。
        #   · ⚠ 判据**只看黄点**（用户点名要的依据 ✓），不再看人物框认没认出来 ✓。
        _mmap_ok = (ws.player.world_x is not None
                    and not bool(getattr(ws.player, "world_held", False)))
        if not _mmap_ok:
            # 小地图找不到黄点：连续超时就停止自动
            lost_timeout = max(0.0, float(s.player_lost_timeout_min)) * 60.0
            if lost_timeout > 0:
                if self._player_lost_since is None:
                    self._player_lost_since = now
                elif now - self._player_lost_since >= lost_timeout:
                    s.enabled = False
                    self._release_combat_keys()
                    self.state = "idle"
                    # 它**自己**把自动停了 ⇒ 日志里必须留下（不然事后只能说"不知道
                    # 自动怎么关的"✗）。丢角色本身的时长由 `player_gone` 注记负责 ✓。
                    behavior.event("player_lost_stop")
                    return {"state": "idle", "reason": "长时间未定位到玩家，已停止自动",
                            "kill_mobs": self._take_kill_mobs()}
            # 没定位到玩家：释放打怪相关按键，但保留定时行为序列（它们不依赖玩家）
            self._release_combat_keys()
            self.state = "idle"
            return {"state": "idle", "reason": "未定位玩家",
                    "kill_mobs": self._take_kill_mobs()}

        # 定位到玩家：重置找不到玩家计时
        self._player_lost_since = None

        # 朝向监控：朝向变化在 set_facing 里刷新计时；超过 facing_timeout_min
        # 分钟仍没变，认为卡死/异常，停止自动。
        timeout = max(0.0, float(s.facing_timeout_min)) * 60.0
        if timeout > 0 and now - self._last_facing_change >= timeout:
            s.enabled = False
            self.keys.release_all()
            self.state = "idle"
            return {"state": "idle", "reason": "朝向长时间未变化，已停止自动",
                    "kill_mobs": self._take_kill_mobs()}

        # 防掉线：开关上升沿随机一个下次触发时间；到点只标记「待休息」，
        # 真正进入隐身要等攻击范围内的怪清空（见下面 in_range 之后）。
        if s.anti_afk_enabled:
            if not self._was_afk_enabled:
                self._next_afk = now + self._random_afk_interval()
            # 手动「进入休息」：同样只标记待休息，不走后门 —— 也要等攻击范围内的
            # 怪清空才真的进隐身，否则正打着怪突然站住挨打。
            if s.rest_request:
                s.rest_request = False
                self._rest_pending = True
            if now >= self._next_afk:
                self._rest_pending = True
        else:
            # 防掉线关着：手动休息没有意义（进入/退出隐身序列就是防掉线的配置，
            # 而且休息分支在防掉线关掉时会立刻退出休息），请求丢掉
            s.rest_request = False
            self._rest_pending = False
            self._next_afk = 0.0        # 关掉防掉线：清排期，UI 不显示倒计时
        self._was_afk_enabled = s.anti_afk_enabled
        # 回读给 UI：下次休息倒计时 + 是否在等清怪
        s.next_afk_monotonic = self._next_afk
        s.rest_pending = self._rest_pending

        px = ws.player.x

        # 按视野矩形过滤怪物（视野外的怪不参与决策）
        # ⚠ 原来这里还有一道「地形关系」过滤（`m.reachable`：怪是否与玩家同一平台），
        # 它依赖**平台识别**；那套感知 2026-09-26 已整块移除 ⇒ 这道过滤没有依据了，
        # 一并删掉。**行为变化**：视野里别的平台上、其实过不去的怪，现在也会被盯上。
        mobs = self._filter_mobs(ws.mobs, ws.player, ws)

        # 追击起跳的**上升沿**：起跳范围内「从无怪变成有怪」的那一拍才有跳的资格。
        # 一只怪一直挂在范围内 → 只有进来的那一拍是沿，之后不再触发 —— 这就是
        # 「同一只怪不会一直跳」。放在这里每拍算一次，不塞进某个分支：下面三处
        # 起跳（attack 分支的内侧 / chase / sweep）共用同一个沿，行为不会随分支而变。
        # 另有一条前提在调用点上：**攻击范围内没有别的怪**（有得打就先打）。注意
        # 「别的」二字 —— 目标自己落在范围内不算，内侧区间（-15~0）全靠这一点。
        _band = [m for m in mobs
                 if self._in_chase_jump_range(self._center_dist(m, ws.player))]
        jump_edge = bool(_band) and not self._band_had
        self._band_had = bool(_band)

        # 「追击起跳需要的冲刺时间」的计时（用户 2026-09-26 要求）：**连续** chase 才累加，
        # 一进别的状态立刻归零。放在这里（每拍、所有分支之前）算一次，三处起跳调用点共用
        # 同一份计时 —— 和 `jump_edge` 同一个道理：判据不随分支而变。
        # ⚠ 用**上一拍**的 `self.state`（本拍的状态在各分支末尾才写）⇒ 有一拍延迟，
        # 对"要冲几百毫秒"这件事无所谓；换来的是**只改这一处**，不用在每条分支里各清一次。
        if self.state == "chase":
            if self._chase_since is None:
                self._chase_since = now
        else:
            self._chase_since = None

        # ⭐ 「**最大战斗时长 ⇒ 换地方**」（用户 2026-09-28 接上行为 ✓，口径见 `_fight_beat` ✓）：
        #   放在这里和 `_chase_since` 同一个位置 —— **每拍、所有分支之前**算一次 ✓
        #   （判据不随分支而变，同上面那条的理由 ✓）。它**只记时间 / 到点下任务**，
        #   不直接改状态 ✓（状态由后头的攻击支 / 任务支决定 ✓）。
        self._fight_beat(now, ws)
        # ⭐ 「结束休息时手上正演着的那一段」也在这条**每拍都走**的路上推（用户 2026-09-28 ✓）：
        #   ⚠ 必须放在这里（**所有分支之前** ✓）—— 因为休息一旦"结束"，`state` 就不再是
        #     `afk_spot_rest` ⇒ `_spot_loop_beat` **再也不会被调到** ⇒ 手上那段（还按着键 ✓）
        #     会**永远挂着、键也松不掉** ✗。放在这条公共路上才能"让它自然演完" ✓。
        self._loop_drain_beat(now)

        # 卡键看门狗（每拍一次；用**上一拍**的状态判断，见方法说明）
        self._watch_output_held(now)

        # 自动喝药：依赖玩家定位（读血/蓝），定位到玩家后才执行
        self._drink_potions(ws, now)

        # 锁定候选（sweep 只背后 back_range 内的怪；patrol 全部）
        # ⚠ 还要过一道「限制战斗区域」：**区域外的怪不给锁**（用户 2026-09-27 要求 1 ✓）；
        #   攻击链不走这里（它吃原始 `mobs` ✓）⇒ 够得着照打 ✓，见 `_candidates` 的说明。
        candidates = self._candidates(mobs, px, ws=ws, now=now)

        # attack 最高优先级：只要**攻击范围内**有框，就进入攻击（或规避）。
        # ⚠ 挂着「命令前往」任务时用 `_in_range_all`（**不分前后**）—— 用户 2026-09-26 报：
        #   "命令前往的寻路途中，攻击范围内有怪也不 attack" ✗。原因：`_in_range` 只认正前方，
        #   而正常路径里身后的怪由「回身输出」那条分支接 ✓ —— 任务在跑时 else 分支是**任务
        #   自己**（`_climb_tick`）⇒ 身后怪没人接 ✗ ⇒ 走路时从身后追上来的怪一路白打我们。
        #   打法交给 `_attack_state` 里"挂任务只许站桩"那一支：**原地转身打、不走位** ✓。
        if self._climb is not None:
            in_range = self._in_range_all(mobs, ws)
        else:
            in_range = self._in_range(mobs, ws)

        # ⚠ **「攻击范围有怪也不进 attack」的唯一特例**（用户 2026-09-27 要求 ①）：
        #   **已经在绳梯上、而且正在爬**（状态 climb + 感知说脚下就在那根绳上）。
        #   为什么：爬到一半被怪拉去打架 = **松手掉下来**（刚爬的几十像素白费，落点也不
        #   可控 ✗）—— 这段路本来就该"先上去、上去再打" ✓。
        #   判据**只看两件**（都是**这一拍**的实时事实 ✓，一件都不许来自"上一拍"✗）：
        #     ① 感知说脚下**确实在绳上**（`ladder_id` 非空）—— 它还隐含"**正按着 ↑/↓**"
        #        （那是写 `ladder_id` 的许可 ✓，见 `live_thread._fill_route_ctx`）；
        #     ③ 那根绳**就是当前任务要爬的那根**（`_climb.ladder_id`）—— 少了它，走路任务
        #        路过某根绳的 x（`ladder_at` 会认成在绳上）就会一路不还手 ✗
        #        （走路/下跳任务没有 `ladder_id` ⇒ 这里天然为假 ✓，`t_climb_attack_exception_agent` ③ 钉着 ✓）。
        #   ⚠ 这里原来还有**第三件**：`self.state == "climb"`（取上一拍的状态 ✗）——
        #     用户 2026-09-27 报"**climb 执行时应该屏蔽跳进 attack 状态，现在好像失效了**"，
        #     病根就是它：那是个**自锁** ✗ —— 只要有一拍没豁免成（抓绳那几拍 x/y 抖出绳框、
        #     `ladder_id` 短暂变空 ⇒ 这一拍进了 attack ✗），而**进了 attack 就不会再去 tick
        #     攀爬任务** ⇒ 状态停在 attack ⇒ 这道闸**永远不再打开** ⇒ 人明明挂在绳上，却被怪
        #     牵着打（"先上去、上去再打"整条失效 ✗）。⇒ **删掉那一件** ✓（②→❌），
        #     判据只剩「在绳上 + 就是这根绳」⇒ 下一拍立刻恢复爬 ✓。
        #   ⭐⭐ **2026-09-28 用户要求：取消"最后一个 attack 的绳梯豁免"**（原话：
        #     "取消最后一个 attack 的『绳梯豁免』：**对齐 x 期间如果进了 attack，就从入口开始**"）。
        #     · 之前 `holds_player` 有**两条**：`_jumped_once`（按过跳）**或** `_in_span`
        #       （人正站在这根绳那一格里）—— 后者就是"**站在绳底对齐 x 时不打架**"那个豁免 ✗；
        #     · 现在**对齐阶段（`ClimbJob.ALIGN`）一律不豁免** ⇒ 对齐 x 期间攻击范围内有怪
        #       就**照常进 attack** ✓（人在平台上，打得过就该打 ✓）；
        #     · **打完怎么继续：从入口开始** ✓（不是"接着对齐" ✗）—— 由 `_task_settle`
        #       → `_climb_interrupted` → `ClimbJob._resume` 那段把 `phase` 打回 `ALIGN` 并
        #       **清掉全部对齐锚点**（`route.py` 里 `_paused >= PAUSE_MIN_S` 那一支 ✓）
        #       ⇒ 等于重走整条上绳流程 ✓。
        #     ⚠ 为什么敢放开（2026-09-27 那个坑的对称处理）：那时靠"豁免"躲开"**攻击判定在
        #       任务 tick 之前** ⇒ 有怪就永远轮不到按跳"✗；现在靠"**打完重来**"绕开 ——
        #       怪被清掉之后天然会走完对齐 ✓。而 **`_jumped_once` 那一条照旧豁免** ✓
        #       （按下跳之后松手就掉 ✗，那一段不许进 attack ✓）。
        # ⭐ 用户 2026-09-27 补充要求（原话）："**在执行攀爬时，从「按下跳」到「攀爬成功」期间
        #   都不应该进 attack**" —— 实测现场：三楼 → 顶层，正在"对齐 x + 起跳"那一刻还是进了
        #   attack ✗。⇒ 判据加一条**任务自己说了算**的：`ClimbJob.holds_player(px, py)`
        #   （位置就在这根绳的范围里 **或** 已经按过跳且还没结束 ✓）。
        # ⚠ 为什么不能只等"任务按过跳"（上一版就是这样 ✗）：**攻击判定在任务 tick 之前**
        #   （本段在上面、任务在下面 else 分支才跑 ✓）⇒ 只要攻击范围内有怪，任务就**永远轮不到
        #   按跳** ⇒ `_jumped_once` 永远是 False ⇒ **既打不进去也爬不上去**（用户现场 ✗）。
        #   ⇒ 这里必须用**这一拍就能读到**的东西：⭐ 位置状态广播的 `on_rope_pos`
        #     （"**只看位置**编出来的绳号" ✓，与按键许可无关 ✓）。
        #   ⚠ 这里原来把 `ws.player.world_x/world_y` 交给任务自己的 `_at_rope` 去比坐标 ——
        #     2026-09-27 用户要求**收编** ⇒ 那套**已删** ✗；几何只在 `perception/pos_state.py`
        #     算一次 ✓（口径 = 感知那把尺：x 半宽 `mapdata.LADDER_DX`(**24**) + y 上下
        #     `LADDER_PAD`(**12**) ✓）。
        _pos = PosSnapshot.of(ws.player)        # 这一拍的位置状态广播（**一次拿走** ✓）
        _lid = str(getattr(self._climb, "ladder_id", "") or "")
        _climb_holds = bool(
            self._climb is not None
            and getattr(self._climb, "holds_player", None)
            and self._climb.holds_player(_pos.on_rope_pos))
        # ⭐ **豁免收窄成"只在**还没吸上绳**那一段"**（用户 2026-09-28 要求 1，原话：
        #   "取消……绳梯豁免的**第 2 种情况（爬绳途中）**：① **爬绳途中正常触发 attack**，
        #    但不会松开 ↑ 键"✓）。
        #   · `_in_rope` = 广播说人**正贴着本任务那根绳**（`ladder_id` 非空且是这根 ⇒ 隐含
        #     "正按着 ↑/↓" ✓ 见 `live_thread._fill_route_ctx` 的许可）⇒ **这一段不再豁免** ✓；
        #   · 豁免只留给**还没吸上绳**的那一段（对齐 → 按跳 → 飞行中）—— 用户 2026-09-27 报的
        #     "**从按下跳到攀爬成功期间都不应该进 attack**"在那一段**仍然生效** ✓
        #     （`holds_player` 的 `_jumped_once` 只管到"吸上绳"为止 ✓ 由这里的 `not _in_rope`
        #      划界 ✓）。
        #   ⚠ 为什么不直接删 `_climb_holds`：那会把"对齐/按跳"那几拍也放进来 ⇒ 又回到
        #     2026-09-27 那个坑（攻击判定在任务 tick 之前 ⇒ 附近有怪就永远轮不到按跳 ✗）。
        _in_rope = bool(_pos.on_rope(_lid))
        self._in_rope_now = _in_rope
        # ⭐⭐ **对齐阶段不再豁免**（用户 2026-09-28 ✓ 见上面那段说明）：
        #   `ClimbJob.ALIGN` 期间 ⇒ `_on_rope` 强制为 False ⇒ **照常进 attack** ✓；
        #   打完由任务侧的"被打断 ⇒ 回 ALIGN + 清锚点"让它**从入口重来** ✓。
        #   ⚠ 只有"**按下跳之后**"（`_jumped_once` 那一支）和"**已经在绳上**"仍豁免 ✓。
        from decision import route as _rmod_p        # 局部 import：避免模块级循环 ✓
        _align_phase = (str(getattr(self._climb, "phase", ""))
                        == _rmod_p.ClimbJob.ALIGN)
        _on_rope = bool(_climb_holds and not _in_rope and not _align_phase)

        # 「**添加任务队列**」刚排进来（`queue_goto` 拍的那面旗子）⇒ **这一拍就把第一条
        # 跑起来**（用户 2026-09-27："寻路任务队列要依次执行，现在我排队列都没反应" ——
        # 原来是**只有** `_task_finished` 收工那一刻才会去接队列，于是"光排队列"永远不动 ✗）。
        # ⚠ 位置是挑过的：
        #   · 在**休息之后**（上面 `REST_STATES` 已经早退）—— 正在休息说明"它先来的"，
        #     让它做完（用户口径"同优先级、谁先谁优先"✓）；旗子留着，休息一结束就起跑 ✓；
        #   · 在**战斗之前**（下面 `in_range` 那支）—— 队列先跑起来；攻击范围内有怪的下一拍
        #     照旧进 attack + 刷新超时（用户 2026-09-27 认可"依旧走进 attack + 刷新超时判定
        #     那套"✓，刷新见 `_climb_interrupted` ✓）；
        #   · 在**定位之后**（上面 `ws.player.found` 那支）—— 站哪儿都不知道，没法解析路径 ✓。
        # 旗子**只吃一次**：起跑完就归 False（免得每拍都去戳队列 ✓）。
        if self._queue_kick:
            self._queue_kick = False
            if self._goto_queue and self._climb is None and not self._route:
                self._goto_queue_next()

        # 防掉线-隐身休息：已到触发时间，且攻击范围内的怪已清空 → 开始进入隐身
        if self._rest_pending and not in_range:
            self._rest_pending = False
            # 同步清掉 UI 的两个排期量：这一帧之后就进入休息了（休息门禁会提前
            # return），不会再有人写它们，留着就是过期值。
            s.rest_pending = False
            s.next_afk_monotonic = 0.0
            self._begin_rest(now)
            self._run_rest(now, ws)
            return {"state": self.state, "reason": "进入隐身休息", "target": None,
                    "dx": 0, "dist": 0, "keys": [], "facing": self.facing,
                    "kill_mobs": self._take_kill_mobs()}

        # 「**限制战斗区域**」（用户 2026-09-26）：脚下不属于配置的集合 ⇒ **这一拍不打架**，
        # 先下一条「前往」回区域 ✓。⚠ 必须排在 `if in_range:` **之前** —— 否则怪一进范围
        # 就开打，规则等于没有 ✗。触发条件是"**本来要打了**"（攻击范围内有怪 / 有可追的候选 ✓），
        # 单纯待机不动不触发 ✓。
        # ⚠ **挂着「前往」任务时不判它**（用户 2026-09-26 现场定的判据，措辞就是判据本身）：
        #    「『命令前往』之后当前任务是**前往 X**；而这条规则的触发条件是**要把当前任务
        #    设成战斗** —— 两者天生不该冲突」。原来漏了这道前提 ⇒ 实测（寺院通道2，
        #    `battle_zone_sets=[右下]`）：「右下」<命令前往>「左上平台」，**一离开右下就
        #    全程不还手**（还在「上下过渡平台」上就已经是"不在战斗区域，先回去"），到了
        #    左上攻击范围内有怪也不进 attack，而且命令走完那一刻立刻被拉回右下 ✗。
        #    判据和上面 `_in_range_all` 那处**同一个**（`self._climb is not None` = 有任务
        #    在跑 ✓），不另发明一套"当前任务是不是前往"✗。
        #    任务收工之后（`_climb` 空了）这一条**照旧生效** ⇒ 原语义（不配区域就别乱打）
        #    一点没减 ✓，见 `t_battle_zone_restriction` / `t_zone_rule_yields_to_goto`。
        # ⚠ **「临时战斗」不判这条**（用户 2026-09-27 要求）：**休息收工/被打断之后**、
        #   **寻路失败之后**那一段，有怪就**先打** —— 那也是战斗，只是"临时的" ✓。
        #   熄灭条件就是下面那一行：**这一波打完了**（视野里再没有可打的）⇒ 规则回到正常，
        #   下次不在区域里又有怪，照旧下「回区域」✓。**不是时长**，所以没有常数 ✓。
        if self._temp_fight and not (in_range or candidates):
            self._temp_fight = False
        if (self._climb is None and (in_range or candidates)
                and not self._in_battle_zone(ws)
                and not self._temp_fight):
            _zk = set()
            self._leave_battle_zone_tick(now, ws, _zk)
            self.keys.set(_zk)
            return {"state": self.state, "reason": "不在战斗区域，先回去",
                    "target": None, "dx": 0, "dist": 0, "keys": sorted(_zk),
                    "facing": self.facing,
                    "kill_mobs": self._take_kill_mobs()}
        if in_range and not _on_rope:
            # ⚠ 这一拍**任务跑不了**（被战斗占用了）⇒ 要告诉它一声：不报信的话，打架那几秒
            # 会被它算成"爬不动了 / 走不动了 / 超时" ✗（用户 2026-09-26 报的"爬到 y=-170
            # 就不再升了（3.0s 没变好）"就是它 ✗）。
            # ⛔ 唯一例外写在上面那个 `_on_rope` 里（**在绳上就不进来这一支** ⇒ 走 else
            #    分支让任务接着跑 ✓）—— 别在这儿再补别的例外（对齐阶段要能被打断 ✓）
            # ⚠ **这里不再手工调 `_climb_interrupted`**（2026-09-26 结构性收口）：这一拍没有
            # 推任务 ⇒ `_task_settle` 在**下一拍开头**统一报信，起点取 `_task_ran_at`
            # （= 任务最后一次真跑的拍）⇒ 扣时精确到拍，和这里手工调**等价** ✓。
            # 留着这条注释是给"以后想在这里补回来"的人看的：补回来就是**双重报信**
            # （幂等无害，但会把"唯一一处"又变成两处，收口白做 ✗）。
            # attack 最高优先级：攻击范围内有框就攻击。
            # 输出后锁定防抖：这段时间内 attack 目标保持锁定，不切换到范围内其他框。
            lock_db = max(0.0, float(s.attack_lock_debounce_ms)) / 1000.0
            if lock_db > 0 and now - self._last_output < lock_db:
                target = next((m for m in in_range if m.id == self._target_id), None)
                if target is None:
                    target = in_range[0]
            else:
                target = in_range[0]
            best = self._center_dist(target, ws.player)
            self._target_id = target.id
            self._target_until = now + self._random_target_cd()
            self._no_target_since = None
            # 追击起跳（区间落在攻击距离**内侧**的情形：`chase_jump_min~max`
            # 都为负，如 -15~0 → 区间 [65,80]、攻击距离 80）。
            #
            # 这一处**必须单独判**：进区间的怪自己就落在攻击范围内，所以下面那句
            # 「攻击范围内有怪就不跳」对它永远成立，跳的资格一次都轮不上 ——
            # 只有「攻击范围为空」那两处调用点的话，这道配置就完全不跳了。
            # 前提仍然是同一条：**范围内没有别的怪**（`len(in_range) == 1`，即
            # 除目标外为 0）—— 有得打就先打，跳会打断输出、还可能把自己跳出攻击
            # 距离；目标自己在范围内不算「别的怪」，那正是「已进范围但偏远，跳一下
            # 够着」。沿也用同一个 `jump_edge`：同一只怪一直挂在区间里不会反复跳。
            if len(in_range) == 1 and jump_edge and self._climb is None:
                # 寻路任务挂着 ⇒ **不跳**：跳跃会毁掉"对齐绳的 x"（同上）。
                self._maybe_chase_jump(best, now, edge=True)
            keys = self._attack_state(target, best, mobs, ws)
            # ⭐ **人已经在绳上爬着的时候，任务这一拍照推**（用户 2026-09-28 要求 1 的后半句：
            #   "爬绳途中正常触发 attack，但**不会松开 ↑ 键**"✓）。
            #   ⚠ 位置必须在 `keys` 建好**之后**（这一支的 `keys` 是 `_attack_state` 给的 ✓；
            #     放到前面会 `UnboundLocalError` ✗ 实测踩过 ✓）。
            #   为什么必须显式推：这一支是**早退**的（不调 `_climb_tick`）⇒ 不推的话 `keys`
            #   里就没有 ↑ ⇒ `KeyState.set` 会**把 ↑ 松开** ⇒ 人从绳上掉下来 ✗（正是要避免的
            #   那件事 ✓）。
            #   ⚠ 人在绳上时任务只会走"按住 ↑ 往上爬"那一相（不会去对齐 / 按跳 ✓），所以这一推
            #     **不会**跟攻击抢方向键 ✓。
            #   ⚠ **不要**顺手调 `_set_state("climb")`：这一拍的状态是 attack ✓
            #     （`_climb_tick` 自己不设状态，设状态是调用方的事 ✓）。
            if _in_rope and self._climb is not None:
                self._climb_tick(now, ws.player.world_x, keys, ws)
            # 站桩输出：本帧进了 attack 状态（范围内有怪、且没在规避）→ 下一帧起一个
            # 方向键都不按，一直站到攻击范围内清空（下面 else 分支解除）。CD 期间也
            # 保持停下：范围内还有怪就不该往前走。**两个策略都要**（老写法只给扫平台，
            # 平地巡逻就成了"一边打一边朝怪挪"，见 _attack_state 的说明）。
            self._stand_attack = (self.state == "attack")
        else:
            # 前方攻击范围内没框
            self._stand_attack = False    # 清空 → 恢复移动（巡逻/追怪）
            target, best = self._locked_target(candidates, ws, now)
            keys = set()
            # **上绳任务优先**（见 _climb_tick）：它没跑完时这一帧只听它的；
            # 跑完的那一帧返回 True ⇒ 落回下面的追怪/巡逻，行为照旧。
            # ⚠ 交给上绳任务的是**世界坐标 x**（不是上面那个画面坐标 `px`）——
            # 见 `_climb_tick` 的说明：喂错了就是"朝一个方向一直走"的死循环。
            if (self._climb is not None
                    and not self._climb_tick(now, ws.player.world_x, keys, ws)):
                self._set_state("climb")
                perf.count("climb_tick")
                best = 0.0      # 收尾要 round(best)（跟 sweep 分支一样兜个底，别传 None）
            elif target is not None:
                # ⚠ 先判「**怪和我是不是同一块平台**」（用户 2026-09-27 要求）：不是 ⇒
                #   **先下「前往」任务**、这一帧就不朝它走了（`_steer` 只会水平按键 ⇒
                #   怪在别的平台上时人只会贴着边缘干蹭 ✗，见 `_chase_goto_if_elsewhere` ✓）。
                # ⛔ 「**禁用杀怪寻路**」开着 ⇒ **整条跳过**（用户 2026-09-28 ✓）：
                #   **不问**"怪在哪块平台"、**不下**前往任务 ⇒ 直接落进 `else` 的
                #   `self._steer(target.x - px, keys)` = **无寻路时的老逻辑** ✓（只按 ←/→ 朝它走 ✓）。
                #   ⚠ 闸门放在**调用处**（而不是进函数再加早退 ✗）：① "这条链要不要跑"本来就是
                #     这个调用点的事 ✓；② 函数内加早退会让那些**直接调 `_chase_goto_if_elsewhere`**
                #     的用例看不见开关（它们就不该受界面开关影响 ✓）。
                if (not self.settings.disable_chase_pathfinding
                        and self._chase_goto_if_elsewhere(target, now, ws)):
                    best = 0.0
                else:
                    # 有锁定目标（sweep 背后怪 / patrol 最近怪）：朝它走
                    self._steer(target.x - px, keys)
                    # 追击起跳：本分支攻击范围内本来就是空的，不存在「其他怪」，前提天然满足
                    self._maybe_chase_jump(best, now, edge=jump_edge)
                    # 打点：追的目标离多远、以及它是不是「幽灵框」（missed>0，即已经
                    # 漏检、靠防抖保留着位置的框）。chase_ghost 占追击时间越多，说明
                    # debounce_ms 保留太久 —— 角色正朝一个已经消失的框走过去。
                    # ⚠ 连续量 ⇒ `min_gap`（**定频**：1 秒一个点 ✓）—— 追击时距离**每拍
                    #   都在变**，不给它就是每拍一条（20~25 ms ✗，实测占日志 6.9% ✓）
                    behavior.sample("chase_dist", round(best or 0.0, 1), min_gap=1.0)
                    if getattr(target, "missed", 0) > 0:
                        behavior.event("chase_ghost")
                self._set_state("chase")
            elif s.strategy == "sweep":
                # 扫平台巡逻：无背后怪，朝倾向朝向走（不锁定、无红框）。
                # 物理朝向先拉回倾向朝向（可能刚转身打过背后怪），再朝它走。
                best = 0.0
                self.set_facing(self._patrol_dir)
                keys.add(s.keymap["right"] if self._patrol_dir > 0 else s.keymap["left"])
                self._set_state("chase")
                # 追击起跳：向倾向方向移动时，起跳区间内有怪（即使未锁定）也可以跳
                if jump_edge:
                    hit = next((m for m in mobs
                                if (m.x - px) * self._patrol_dir >= 0
                                and self._in_chase_jump_range(self._center_dist(m, ws.player))),
                               None)
                    if hit is not None:
                        self._maybe_chase_jump(self._center_dist(hit, ws.player),
                                               now, edge=jump_edge)
                # 换向：倾向朝向方向没怪持续 sweep_turn_cd 才换向。
                # 主方向有怪（哪怕是远处、还没进攻击范围）就不换向，保持朝它走。
                front_has_mob = any((m.x - px) * self._patrol_dir >= 0 for m in mobs)
                if front_has_mob:
                    self._no_target_since = None   # 主方向有怪：重置换向计时
                else:
                    if self._no_target_since is None:
                        self._no_target_since = now
                    turn_cd = max(0.0, float(s.sweep_turn_cd)) / 1000.0
                    if now - self._no_target_since >= turn_cd:
                        self._patrol_dir = -self._patrol_dir
                        self.set_facing(self._patrol_dir)
                        self._no_target_since = None
            else:
                # 平地巡逻：无怪，idle
                self._set_state("idle")
                self.keys.release_all()
                # ⚠ **必须把输出序列按着的键也松开**（2026-09-26 用户报的现象：
                # "画面上一个框都没有，角色却持续在按输出"）。
                # `self.keys.release_all()` 只松 **KeyState 的移动键**；输出键是 `_run_seq`
                # 用 `key_down` **直接发的**（不经 KeyState）⇒ 只有 `_release_combat_keys()`
                # （内部 `_release_ctx(_output_ctx)`）才松它。
                # 漏了它的后果：怪在输出序列的 down 与 up 之间消失（默认两拍隔 70~130ms，
                # 比一格帧间隔 66ms 还长，很容易撞上）⇒ 序列停在下过半截、"up" 永不发出
                # ⇒ 攻击键一直按着（而状态已是 idle、画面上也没框了）；默认 60s 的
                # RELEASEALL 会兜一次，所以表现为"偶尔持续一阵子"。
                # 别的早退（未定位玩家 / 关自动 / 再查一次开关）本来就都调了它，这里对齐。
                self._release_combat_keys()
                # ⭐ **idle 回归**（用户 2026-09-28，字段口径见 `battle_zones` 的 `idle_foothold` ✓）：
                #   没怪、也没任务时，**水平走**回本区域那一格的中心 x、到中心就停 ✓
                #   （**不跨层** —— 只按 ←/→ ✓，不下寻路任务 ✗）。
                #   ⚠ 它**在这条早退里**，所以键要自己带上（`keys` 字段 ✓）；没配 / 没注入
                #     解析器 ⇒ `_idle_walk_beat` 一个键都不加 ⇒ 行为和以前**一模一样** ✓。
                _ikeys = set()                # ⚠ `_steer` 往里 `add` ⇒ 必须是 **set** ✗ list 会炸
                self._idle_walk_beat(_ikeys, ws)
                return {"state": "idle", "reason": "无怪", "keys": list(_ikeys),
                        "kill_mobs": self._take_kill_mobs()}

        # 发键前再检查一次自动开关：主线程可能在这一帧决策中途把自动关掉并
        # 兜底释放了按键。若这里照常发键，会和主线程的释放竞争，导致移动键
        # 刚被释放又被按下，表现为「关掉自动还在左右走」。
        if not s.enabled:
            self._release_combat_keys()
            self.state = "idle"
            return {"state": "idle", "reason": "未开启",
                    "kill_mobs": self._take_kill_mobs()}

        # 最小切换朝向时间：换向后方向键至少按住这么久（决策里没有方向键时补回来）。
        # 放在发键之前 —— 这里返回的才是真正要发出去的键，也一并回读给 UI。
        #
        # **站桩输出时只按一小下**（`_stand_attack` = 攻击范围内有怪 → 本帧决策故意
        # 不给方向键）：那时按方向键只为了把角色转过来，按满 `min_turn_hold_ms` 就是
        # 一边走一边打，角色会从怪身上走过去。见 TURN_TAP_S。
        # ⚠ 挂任务时（`_climb` 非空）原来一律 cap 0（= 完全不补键 ✓ 任务自己管方向键），
        #   但**站桩攻击**那一拍必须留口子（用户 2026-09-26 报的"寻路途中身后的怪不打"）：
        #   攻击只打得到**面前**，不给这一下就**不会真的转身** ⇒ 算进来了也是**空放** ✗。
        #   `TURN_TAP_S` 那一小下是**原地转**、不改变位置 ✓；走路时仍然 cap 0，绝不干预任务 ✗。
        # ⭐ **站桩时每 `STATION_TURN_EVERY` 轮按满一次**（用户 2026-09-28 要求 2 ✓）：
        #   · 平时站桩 ⇒ 仍只点一下（`TURN_TAP_S` ✓ 见上面那段说明：按满就是从怪身上走过 ✗）；
        #   · **轮到"该补"的那一轮**（第 3、6、9… 轮 ✓）⇒ **按满「最小切换朝向时间」** ✓
        #     —— 就是用户说的"持续『最小切换朝向时间』"✓。
        #   ⚠ **绳上绝不按左右**（`_in_rope_now` ✓）：在绳上按左右在游戏里就是**松手** ⇒ 会掉 ✗
        #     （用户 2026-09-28 明确："在绳上只用逻辑朝向（`set_facing`）+ 攻击键，绝不按左右"
        #      ✓）⇒ 传 `cap=0`（`_hold_turn` 见 0 就不补键 ✓，`set_facing` 照旧每拍设 ✓）。
        if self._in_rope_now:
            cap = 0.0
        elif self._stand_attack:
            cap = TURN_TAP_S
        else:
            cap = 0.0 if self._climb is not None else None
        keys = self._hold_turn(keys, now, cap_s=cap)
        # ⭐ **每 3 轮的"补朝向键"那一下：按满「最小切换朝向时间」**（用户 2026-09-28 要求 2 ✓）。
        # ⚠ **不走 `_hold_turn`**：它那个窗（`_turn_at`）会被站桩里**每轮的 `set_facing`** 反复
        #   刷新 ⇒ 方向键会一直按着不放 ✗（实测 4 秒的用例里按了 3.94 秒 ✗）⇒ 这里用
        #   `_turn_due_at`（**那一轮自己的起点** ✓），时间一到就自然停 ✓。
        # ⚠ **绳上绝不按**（`_in_rope_now` ✓，用户 2026-09-28 明确）：在绳上按左右 = 松手掉下来 ✗。
        # ⚠ 键已经按着方向键时不重复加（`_hold_turn` 同款判断 ✓）。
        if (self._stand_attack and not self._in_rope_now
                and self._turn_due_at is not None):
            _hold_s = max(0.0, float(s.min_turn_hold_ms)) / 1000.0
            if (now - self._turn_due_at) < _hold_s:
                _km = s.keymap
                _dirs = {_km.get("left"), _km.get("right"), _km.get("up"), _km.get("down")}
                _dirs.discard(None)
                if not (keys & _dirs):
                    _k = _km.get("right") if self.facing > 0 else _km.get("left")
                    if _k:
                        keys = set(keys) | {_k}
        self.keys.set(keys)

        # 输出行为：攻击 / 跳规避 / 回身输出（按 self.state）
        self._output_actions(now, target, mobs, ws)

        tid = target.id if target is not None else None
        dx = round((target.x - px) if target is not None else 0.0, 1)
        return {"state": self.state, "target": tid, "dx": dx,
                "dist": round(best, 1),
                "keys": sorted(keys), "facing": self.facing,
                "kill_mobs": self._take_kill_mobs()}

    def _random_afk_interval(self):
        """随机一个下次防掉线触发间隔（秒）。"""
        s = self.settings
        lo = max(0.0, float(s.anti_afk_min))
        hi = max(lo, float(s.anti_afk_max))
        return random.uniform(lo, hi) * 60.0

    def _random_rest_interval(self):
        """随机一个本次休息时长（秒）。"""
        s = self.settings
        lo = max(0.0, float(s.anti_afk_rest_min))
        hi = max(lo, float(s.anti_afk_rest_max))
        return random.uniform(lo, hi) * 60.0

    def _begin_rest(self, now):
        """进入隐身休息：先彻底停下战斗 —— 松掉所有按着的键（含定时行为序列
        执行到一半残留的），否则休息期间角色还在走/还在打。

        休息时长从这里（进入隐身行为的**开始**）起算，不是等进入隐身走完才算 ——
        否则进入隐身的耗时会被白送给休息时间（用户期望从开始算）。
        """
        self._release_held_keys()
        self.keys.release_all()
        self._afk_ctx = None
        self._spot_started = False
        if str(self.settings.anti_afk_type or "") == "spot_rest":
            # 「定点休息」：先**走到指定地点**（路线由实时线程解析 ✓），休息时长
            # **从"到达后行为演完"才起算** ⇒ 这里先不定 `_rest_until` ✓。
            # 手上可能还挂着「命令前往」的任务 ⇒ 停掉：两边都会按键，会打架 ✗。
            self.stop_route("定点休息要出发")
            self.state = "afk_spot_walk"
            self._rest_until = 0.0
            behavior.event("afk_start", kind="spot_rest",
                           spot=self.settings.anti_afk_spot_set or "(没选地点)")
        else:
            # 「隐身休息」：进入隐身序列 → 等 → 退出隐身序列
            behavior.event("afk_start", kind="hide")
            self.state = "afk_enter"
            self._rest_until = now + self._random_rest_interval()
        self._rest_last_tick = now
        # 上一次的"被打断"标记不许带到这一次（否则这次正常结束也会按秒数重排）
        self._rest_retry_after_interrupt = False

    def _run_rest(self, now, ws=None):
        """休息状态机：**隐身那一型**走「进入隐身 → 等 → 退出隐身」；
        **定点那一型**走 `_run_rest_spot`（走 → 到达后行为 → 等 → 结束后前往）。

        `ws` 只有定点那一型要用（判"到了没有"看它脚下的集合 ✓ —— 和 executor 同一套判据）。
        """
        s = self.settings
        # ⭐⭐ **循环行为只在「休息中」活着**（用户 2026-09-28："结束休息需要杀掉循环行为正在
        #   执行的行为，且要终止循环"✓）—— 这里是**兜底**：只要这一拍的状态**不是**
        #   `afk_spot_rest`（比如中途把行为类型从"定点休息"改成"隐身休息"、或手动结束、
        #   或休息到点转去"结束后前往"），就一律**收摊**（松掉它按着的键 + 终止循环 ✓）。
        #   ⚠ 为什么非要有兜底：会离开这个状态的路有**十几条**（`_afk_ctx = None` 全文件
        #     就有 16 处 ✓）⇒ 逐处去加一句"顺手清循环"必然漏 ✗（漏一处 = **卡着键**继续走 ✗）。
        #     挂在这里 = **不管谁让状态变了**，下一拍都会收摊 ✓。
        if self.state != "afk_spot_rest":
            self._drain_spot_loop()        # 同上：演完就收，别当场掐断
        self._pause_timers(now)
        if self.state in SPOT_STATES:
            self._run_rest_spot(now, ws)
        elif self.state == "afk_enter":
            # 休息时长到了，而进入隐身的序列还没走完 → **把剩下的进入行为丢掉**，
            # 直接去执行退出隐身。进入隐身可能很长（含 delay、含连招），等它演完
            # 休息时间早就超了；用户要的是「到点就退」，不是「把进入演完再退」。
            # 注意休息时长是从**进入动作开始那一刻**起算的（见 _begin_rest）。
            if now >= self._rest_until:
                # 松掉进入序列按到一半、还没松开的键，否则会卡着那个键去执行退出
                self._release_ctx(self._afk_ctx)
                self._afk_ctx = None
                self.state = "afk_exit"
            else:
                if self._afk_ctx is None:
                    self._afk_ctx = [s.anti_afk_enter_seq, 0, 0.0, set(), []]
                r = self._run_seq(now, self._afk_ctx)
                if r is None:
                    self._afk_ctx = None
                    self.state = "afk_rest"     # 剩余时长在 _begin_rest 就算好了
                else:
                    self._afk_ctx = r
        elif self.state == "afk_rest":
            if now >= self._rest_until:
                self.state = "afk_exit"
                self._afk_ctx = None
        elif self.state == "afk_exit":
            if self._afk_ctx is None:
                self._afk_ctx = [s.anti_afk_exit_seq, 0, 0.0, set(), []]
            r = self._run_seq(now, self._afk_ctx)
            if r is None:
                self._afk_ctx = None
                self._finish_rest(now)
            else:
                self._afk_ctx = r
        # 回读给 UI（「手动结束休息」按钮的可用性 + 休息倒计时）：写在状态迁移之后，
        # 否则会慢一帧。退出走完时 _finish_rest 已把 rest_state 置空，这里不能再写成 "idle"。
        s.rest_state = self.state if self.state.startswith("afk_") else ""
        # 进入隐身的阶段剩余时间也已经在走了（休息从那一刻起算），一起回读
        s.rest_until_monotonic = (self._rest_until
                                  if self.state in REST_STATES_TIMED else 0.0)

    def _run_rest_spot(self, now, ws):
        """「定点休息」的五个阶段（顺序是用户 2026-09-26 定的，见 `docs/开发计划.md` P2）：

        ① 走到指定地点 → ② 到达后行为 → ③ 歇**通用休息时长** → ④「结束后前往」非空就
        先走过去 → ⑤ 回战斗。

        判"到了没有"用**执行器同一套判据**（脚下集合 `here_sets` ✓）—— 不另发明一个 ✗。
        走不到 / 半路断了 ⇒ **如实说 + 回战斗**（不许硬走，也不许静默 ✗）。
        """
        s = self.settings
        here = set(getattr(getattr(ws, "player", None), "here_sets", None) or ())
        dst = str(s.anti_afk_spot_set or "").strip()

        if self.state == "afk_spot_walk":
            if not self._spot_started:
                if dst and dst in here:                 # 已经站上去了：不用走
                    self.state = "afk_spot_act"
                    self._afk_ctx = None
                    return
                ok, msg = self.plan_and_start_route(dst, why="定点休息：去「%s」" % dst)
                self._spot_started = True
                if not ok:
                    self._rest_give_up(now, msg)
                    return
                if self._climb is None:                 # 解析器说"已经在上面了"
                    self.state = "afk_spot_act"
                    self._afk_ctx = None
                return
            if self._climb is not None:
                return                  # 还在走（这一拍外面会推 `_climb_tick`）
            if dst and dst in here:
                self.state = "afk_spot_act"
                self._afk_ctx = None
            else:
                self._rest_give_up(now, "去「%s」没走成：%s"
                                   % (dst or "(没选地点)",
                                      self.current_goto_note() or "任务没走完就停了"))
        elif self.state == "afk_spot_act":
            # ② 到达后行为（行为编辑器编的那条序列）
            if self._afk_ctx is None:
                self._afk_ctx = [s.anti_afk_spot_seq, 0, 0.0, set(), []]
            r = self._run_seq(now, self._afk_ctx)
            if r is None:
                self._afk_ctx = None
                self.state = "afk_spot_rest"
                # ③ 通用休息时长**从这一刻**起算（到达后行为演完才开始歇 ✓）
                self._rest_until = now + self._random_rest_interval()
                # ⭐ 循环行为第一次**立刻**跑（"到达后行为"刚演完 ⇒ 紧接着来一遍最自然 ✓；
                #   而不是"先干等 A 秒"✗）
                self._loop_next_ts = now
            else:
                self._afk_ctx = r
        elif self.state == "afk_spot_rest":
            # ⭐ **休息过程中循环行为**（用户 2026-09-28 ✓）：每拍推一步（口径见 `_spot_loop_beat` ✓）
            self._spot_loop_beat(now, s)
            if self._rest_stop_pending:
                # ⭐ "**手动结束休息**"已经受理、正等手上那一段演完（见 `tick` 里 `rest_abort`
                #   那段 ✓ —— 为什么非要"留在休息状态里等"，那儿写了 ✗ 的教训 ✓）。
                #   ⚠ 这一条**与用户勾没勾「推迟到循环执行完」无关** ✓：点「结束」就是要停，
                #     只是**手上这一项演完再停**（不半截掐断 ✓）。
                if self._loop_ctx is not None:
                    return      # 还在演 ⇒ 接着演（`_loop_drain` ⇒ 不再起跑下一项 ✓）
                self.stop_route("手动结束休息")
                # ⚠ 收尾与"自然到点"**共用同一份**（`_end_spot_rest` ✓）—— 手动结束**也要**
                #   走「结束后前往」✓（用户 2026-09-28 报："我点手动结束休息后，**没有收到
                #   寻路任务**"✗ —— 原来这里是直接 `_finish_rest`，把那一格整个跳过了 ✗）。
                self._end_spot_rest(now, s)
                return
            if now < self._rest_until:
                return
            # ⭐ **「休息结束推迟到循环执行完」**（用户 2026-09-28 ✓ 口径见 `_hold_rest_for_loop`）：
            #   到点但这一轮还在演 ⇒ **先不结束**，等它演完（下一拍再看 ✓）。
            if self._hold_rest_for_loop(s):
                return
            self._end_spot_rest(now, s)     # ⭐ 收尾（含「结束后前往」✓ 与手动结束同一份 ✓）
        elif self.state == "afk_spot_back":
            if not self._spot_started:
                ok, msg = self.plan_and_start_route(
                    s.anti_afk_spot_after,
                    why="定点休息结束：去「%s」" % s.anti_afk_spot_after)
                self._spot_started = True
                if not ok:
                    self._rest_note("定点休息：结束后前往没走成 —— %s" % msg)
                    self._finish_rest(now)
                elif self._climb is None:               # 已经在那儿了
                    self._finish_rest(now)
                return
            if self._climb is None:
                self._finish_rest(now)                  # ⑤ 回战斗

    def _rest_travel_beat(self, now, ws):
        """休息分支里**赶路阶段**那一拍：**有怪先打**，没有就接着走 ⇒ 这一拍要发的键。

        为什么非要有它（用户 2026-09-26 报："点手动进入休息后，会进行寻路，但是攻击范围内
        有怪时没有触发 attack"）：休息分支是**早退**的（除了补血不做任何战斗动作），而
        「有怪先打」那套仲裁只在**正常决策路径**里 ⇒ 「命令前往」会打 ✓、定点休息却一路
        挨打也不还手 ✗（走不到休息点，还可能被打死 ✗）。

        做法**照抄正常路径**（同一套判据，别自己发明 ✓）：正常路径里 `if in_range:` 那一支
        正是 `_attack_state` + `_output_actions`，而且**不**调 `_climb_tick` ——
        「有得打就先打、这一拍不往前走」✓，这里一模一样 ✓。

        ⚠ 只在**赶路**那两个阶段做（`afk_spot_walk` / `afk_spot_back`）：演「到达后行为」
          或正在歇的时候掺战斗，会把宏序列和按键搅在一起 ✗；隐身的三个阶段更是**一点战斗
          都不该有**（那正是它存在的意义 ✓）。
        ⚠ 借 `self.state` 这个**名字**时**必须还原**：休息阶段名就存在 `self.state` 里 ✗，
          被战斗改掉的话休息机器下一拍就认不出自己（等于悄悄放弃这次休息 ✗✗）。
        """
        out = set()
        if self.state in ("afk_spot_walk", "afk_spot_back"):
            mobs = self._filter_mobs(ws.mobs, ws.player, ws)
            in_range = self._in_range(mobs, ws)
            if in_range:
                # 同上：这一拍任务没跑（去打架了）—— 报信由 `_task_settle` 在下一拍开头
                # 统一做（见那儿），这里**不再手工调**（2026-09-26 结构性收口 ✓）。
                tgt = in_range[0]                 # 最近的（`_in_range` 已按距离升序 ✓）
                best = self._center_dist(tgt, ws.player)
                keep = self.state
                self.state = "attack"             # 只借名字，让输出那套（按状态选动作）跑起来
                try:
                    out |= self._attack_state(tgt, best, mobs, ws)
                    self._output_actions(now, tgt, mobs, ws)
                finally:
                    self.state = keep             # ← 不还原就等于放弃这次休息 ✗
                behavior.event("spot_travel_fight")
                return out
        if self._climb is not None:
            self._climb_tick(now, getattr(ws.player, "world_x", None), out, ws)
        return out

    def _random_spot_loop_interval(self):
        """下一次循环的间隔（**秒** ✓）—— 在 `anti_afk_spot_loop_min/max` 之间随机 ✓。"""
        s = self.settings
        try:
            lo = max(0.1, float(s.anti_afk_spot_loop_min or 0.1))
            hi = max(lo, float(s.anti_afk_spot_loop_max or lo))
        except (TypeError, ValueError):
            lo = hi = 30.0
        return random.uniform(lo, hi)

    def _end_spot_rest(self, now, s):
        """「定点休息」的**收尾** —— "**自然到点**"与"**手动结束**"**共用这一份** ✓。

        就两件事（口径由用户 2026-09-28 定 ✓）：
          · 循环行为**收摊**（`_stop_spot_loop` ✓ 防卡键 ✓）；
          · 配了「**结束后前往**」（`anti_afk_spot_after`）⇒ 转 `afk_spot_back` 去那儿；
            没配 ⇒ `_finish_rest` 直接回战斗 ✓（⑤）。

        ⚠ **手动结束也走这里**（用户 2026-09-28 要求 ✓ 原话："我点手动结束休息后，**没有收到
          寻路任务**"）—— 原来手动结束那条是**直接 `_finish_rest`**，把「结束后前往」整格
          **跳过了** ✗。"休息结束"在用户心里就是**一件事**，不该因为"是点按钮停的"就少做
          一步 ✓。
        ⚠ 停路线**由调用方负责**（手动结束用 `stop_route("手动结束休息")` ✓；"自然到点"
          那边由 `_finish_rest` 里的兜底管 ✓）—— 别在这儿再停一次（两处口径会漂 ✗）。
        """
        self._rest_stop_pending = False
        self._stop_spot_loop()          # 休息结束 ⇒ 循环收摊（⚠ 防卡键 ✓）
        self._afk_ctx = None
        after = str(getattr(s, "anti_afk_spot_after", "") or "").strip()
        if not after:
            self._finish_rest(now)                  # ⑤ 回战斗
            return
        self.state = "afk_spot_back"                # ④ 结束后前往
        self._spot_started = False

    def _hold_rest_for_loop(self, s):
        """休息**时间到了**，但「休息结束推迟到循环执行完」勾着、且这一轮还在演 ⇒ **先别结束**。

        ⭐ 用户 2026-09-28 要求 ✓（原话："勾上休息过程中循环行为时，再加一个开关子参数
          『休息结束推迟到循环执行完』"）。要解决的是**同一件事的两套动作抢键**：休息到点时
          那一轮可能**才演到一半**，此时若直接转「结束后前往」/回战斗，就成了"一边按着循环
          的键、一边下发寻路的键" ⇒ 两边打架 ✗（老口径是当场掐断，手上的动作**半截就断** ✗）。

        口径（三条）：
          · 勾着 + **正在演** ⇒ **推迟**（置 `_loop_hold` 拦住新一轮 ✓，返回 True）；
          · 勾着 + **没在演**（间隔期 / 整轮刚演完）⇒ **照常结束** ✓（没东西可等）；
          · 没勾（默认）⇒ **照常结束** ✓（老行为一字不变 ✓）。

        ⚠ `_loop_hold` 是**必须**的：它让"整轮演完"那一步**不再排下一次**（见 `_spot_loop_beat`
          ✓）——否则循环间隔比决策拍还短时，会在"演完 ⇒ 又起跑"之间**无限推迟**休息 ✗。
        ⚠ 只管**自然到点**这一条；**手动结束休息**不走这里（用户点「结束休息」就是要现在停 ✓，
          那种情况仍由 `_drain_spot_loop` 照看："手上这一项演完就收" ✓）。
        """
        if not bool(getattr(s, "anti_afk_spot_loop", False)):
            return False
        if not bool(getattr(s, "anti_afk_spot_loop_hold", False)):
            return False
        if self._loop_ctx is None:
            return False                    # 没在演 ⇒ 没什么可等 ✓
        if not self._loop_hold:
            self._loop_hold = True
            behavior.event("afk_rest_wait_loop",
                           idx=int(self._loop_idx) + 1,
                           total=len(getattr(s, "anti_afk_spot_loop_items", None) or []))
        return True

    def _drain_spot_loop(self):
        """**结束休息**时的收法（用户 2026-09-28 更正）：**不打断正在演的那一项**。

        · 正在演 ⇒ 只置 `_loop_drain`（让它演完再收，键继续按着）；
        · 没在演 ⇒ 直接收摊（没东西可"演完"）。
        无论如何都把 `_loop_next_ts` 归零（休息都结束了，再排下一轮没意义）。
        """
        self._loop_next_ts = 0.0
        if self._loop_ctx is None:
            self._stop_spot_loop()
        else:
            self._loop_drain = True

    def _publish_loop_state(self, total=0, idx=0, name="", next_at=0.0):
        """把**循环行为的运行态**写到 `settings` 上（**只给界面读** ✓ 同 `fight_zone_name` ✓）。

        ⚠ 口径与 `fight_zone_name` 那套一样：面板拿不到 agent 实例 ⇒ 只能读 `settings` ✓；
          这里**只写不判** ✓（界面是只读者 ✗ 不许反过来改）。
        """
        try:
            self.settings.spot_loop_total = int(total or 0)
            self.settings.spot_loop_idx = int(idx or 0)
            self.settings.spot_loop_name = str(name or "")
            self.settings.spot_loop_next_at = float(next_at or 0.0)
        except Exception:                     # noqa: BLE001 —— 界面用的小账，坏了别影响决策 ✗
            pass

    def _stop_spot_loop(self):
        """循环行为**收摊**：松掉它按着的键 + 清两个运行态字段 ✓（**防卡键** ✓）。

        ⚠⚠ 这个必须在**所有离开休息的路径**上被调到（`_finish_rest` ✓ / 休息到点 ✓ /
          关掉勾选 ✓）—— 循环序列也是"按着的键"的宿主，漏一处角色就会**卡着键**继续走 ✗
          （那种 bug 在游戏里极难查 ✓，同 `_finish_rest` docstring 里那条纪律 ✓）。
        """
        if self._loop_ctx is not None:
            self._release_ctx(self._loop_ctx)
            self._loop_ctx = None
        self._loop_next_ts = 0.0
        self._loop_idx = 0            # 收摊 ⇒ 下一轮从头（别从半截接着跑 ✗）
        self._loop_drain = False
        self._loop_hold = False       # ⭐ 「等这一轮演完」这笔账也一起结清 ✓
        # ⚠ 运行态也要跟着灭（不然停休息之后界面还挂着"循环 2/3「起身」"✗ 很误导 ✓）
        self._publish_loop_state()

    def _loop_drain_beat(self, now):
        """**不在休息中**时，把"结束休息时手上正演着的那一段"推完（用户 2026-09-28）。

        ⚠ 没它的话：`_spot_loop_beat` 只在 `afk_spot_rest` 阶段被调 ⇒ 休息一"结束"，
          手上那段**就没人推了** ⇒ 键**永远松不掉**。
        """
        if self._loop_ctx is None:
            return
        r = self._run_seq(now, self._loop_ctx)
        if r is not None:
            self._loop_ctx = r
            return
        self._stop_spot_loop()

    def _spot_loop_beat(self, now, s):
        """「休息过程中循环行为」这一拍（用户 2026-09-28 ✓）。

        口径（照 `_custom_timers` 那段现成的"到点起跑 ⇒ 跑完排下次"范式 ✓）：
          · 没勾 / 序列为空 ⇒ **收摊**（松键 ✓ 老行为一字不变 ✓）；
          · 正在演 ⇒ 推一步（`_run_seq` 给 `None` = 演完了 ⇒ 清 ctx + 排下次 ✓）；
          · 没在演且到点 ⇒ **起跑**（新塞一个 ctx ✓）。
        ⚠ 它**不受** `_pause_timers` 影响（那是给「自定义定时行为」的 ✓）—— 本循环是
          "休息期间**照跑**"的独立实现 ✓（这正是用户要的：休息时该做的动作别停 ✓）。
        """
        if not bool(getattr(s, "anti_afk_spot_loop", False)):
            self._stop_spot_loop()          # 没勾 ⇒ 收摊（中途取消勾选也走这条 ✓）
            return
        # ⭐ **列表**（用户 2026-09-28 ✓）：一次"循环" = **依次跑完每一项**，然后等 A~B ✓。
        #   ⚠ 空序列的项直接**跳过**（界面上允许先建个空项 ✓ 跳过比"卡在那儿"好 ✓）。
        seqs, names = [], []
        for it in (getattr(s, "anti_afk_spot_loop_items", None) or []):
            if isinstance(it, dict) and (it.get("seq") or []):
                seqs.append(list(it.get("seq") or []))
                names.append(str(it.get("name") or "（未命名）"))   # ⚠ 名字要跟着走（界面显示 ✓）
        if not seqs:
            self._stop_spot_loop()          # 勾了但一项都没编 ⇒ 什么都不做 ✓（不报错、不卡键 ✓）
            return
        if self._loop_ctx is not None:
            r = self._run_seq(now, self._loop_ctx)
            if r is not None:
                self._loop_ctx = r
                return
            # ---- 这一项演完了 ----
            self._loop_ctx = None
            if self._loop_drain:
                self._stop_spot_loop()      # 结束时在等它演完 ⇒ 到这儿就收（不排下次）
                return
            if self._loop_idx + 1 < len(seqs):
                # ⭐ **同一轮里紧接着跑下一项**（不等间隔 ✓ —— "按项目依次执行"就是这个意思 ✓）
                self._loop_idx += 1
                self._loop_ctx = [seqs[self._loop_idx], 0, 0.0, set(), []]
                self._publish_loop_state(len(seqs), self._loop_idx + 1, names[self._loop_idx])
                return
            # ⭐ **整轮跑完** ⇒ 排下一次（间隔从**演完那一刻**起算 ✓ 不是"起跑那一刻" ✗：
            #   序列本身占用的时间不该白算进间隔里，否则"间隔 10s + 序列 8s"会变成每 18 秒一轮 ✗）。
            self._loop_idx = 0
            if self._loop_hold:
                # ⭐ 「休息结束推迟到循环执行完」：这一轮**演完了** ⇒ **不再排下一轮**
                #   （下一拍 `_hold_rest_for_loop` 就会放行 ⇒ 休息正常结束 ✓）。
                self._loop_next_ts = 0.0
                self._publish_loop_state(len(seqs), 0, "（一轮完成）", 0.0)
                return
            self._loop_next_ts = now + self._random_spot_loop_interval()
            self._publish_loop_state(len(seqs), 0, "（一轮完成）", self._loop_next_ts)
            return
        if self._loop_next_ts <= 0.0:
            # 没排过（理论上只有"刚进休息那处没跑到"会走到 ✓）⇒ 现在就起跑。
            # ⚠⚠ **这里也必须是"带间隔"的兜底**：原来写 `= now` 看着"立刻跑"很顺，
            #   可一旦"跑完排下次"那一步漏了（改坏了 ✓），它就会变成**背靠背不停跑**
            #   ⇒ 用户设的「循环时间 A~B」**直接失效** ✗（反向验证时正是这么发现的 ✓）。
            #   ⇒ 兜底定成"下一次 = 现在 + 一个随机间隔"，最差也只是**晚一轮**，不会爆 ✗。
            self._loop_next_ts = now
        if now >= self._loop_next_ts:
            self._loop_idx = 0             # 新的一轮**从第一项开始** ✓
            self._loop_ctx = [seqs[0], 0, 0.0, set(), []]
            self._loop_next_ts = 0.0       # 演完那一步会重排 ✓（放着不排 = 只跑一次 ✗）
            self._publish_loop_state(len(seqs), 1, names[0])

    def _rest_give_up(self, now, why):
        """「定点休息」走不到 ⇒ **如实说 + 收工回战斗**（不许硬走，也不许静默 ✗）。

        ⭐ **"走不到"也算被打断**（用户 2026-09-28 明确要求 ✓）：置上
        `_rest_retry_after_interrupt` ⇒ `_finish_rest` 会把下次休息排到
        **`anti_afk_retry_sec` 秒之后**（而不是随机的 N~M 分钟）✓。

        ⚠⚠ **改之前是没置的**（用户问："确认去休息的寻路失败后是否激活了打断重试"✗）——
          那时只有"**自动补血**打断"（`_interrupt_rest` ✓）才置这个标记 ⇒ 走不到休息点时
          就按**随机间隔**排下一次 ⇒ 想再试一次得等几分钟 ✗ 而这种情况**恰恰最该快点重试**
          （路线/坐标出问题，多半一会儿就好了 ✓）。

        ⚠ **只有"去"的那一段算**（`afk_spot_walk` / `afk_spot_act` ⇒ 本方法 ✓）；
          「**结束后前往**」没走成**不置**（`_run_rest_spot` 的 `afk_spot_back` 分支 ✓）——
          那时**休息已经完成了**，没走成只是"回程不顺"，不该因此提前再来一轮休息 ✓
          （这条口径与 `REST_STATE_SPEC` 里"哪些阶段算被打断窗口"一致 ✓）。
        """
        behavior.event("afk_give_up", why=str(why),
                       retry_s=round(max(1.0, float(
                           getattr(self.settings, "anti_afk_retry_sec", 60.0) or 60.0)), 1))
        self._rest_note("定点休息：%s" % why)
        # ⭐ 「走不到」= 这次休息没做成 ⇒ 按"被打断"处理：**秒级重试**（用户要求 ✓）
        self._rest_retry_after_interrupt = True
        self._finish_rest(now)

    def _rest_note(self, text):
        """把一句"为什么"挂到画面上那行（和任务 note 同一处、同一种读法 ✓）。"""
        self._last_goto_note = (str(text), time.monotonic())

    def _interrupt_rest(self, now):
        """休息被**自动补血**打断 → 立刻转去执行退出隐身（下次休息由 _finish_rest 提前）。

        做法与「休息时长到点、但进入隐身还没演完」**同一套**（见 `_run_rest` 的
        afk_enter 分支）：松掉半截序列按着的键 → 转 afk_exit → 演完由 `_finish_rest`
        收尾。刻意**不半路把进入序列掐断再补按一次隐身键** —— "隐身术"这类切换型技能
        被按两次就反了，状态从此对不上；走"退出序列"语义才自洽（退出序列本来负责关隐身）。

        下次休息的时刻不在这里算：`_finish_rest` 看到标记会改用 `anti_afk_retry_sec`
        （**秒**，比"下次随机 N~M 分钟"灵活得多）。
        """
        if self.state == "afk_exit":
            return                              # 已经在退出了，别重复
        # 打点：**这次功能到底有没有在工作**只能靠它回答 ——
        # 否则以后出问题只能猜"它到底触发过没有"（2026-09-26 加）。
        behavior.event("afk_interrupt")     # 打点（每分钟几次的量级，不在热路径 ✓）
        if self.state in SPOT_STATES:
            # 「定点休息」**没有隐身在身上**（只可能在走路 / 演到达行为）⇒ 不走 `afk_exit`
            # 那套（那会去演"退出隐身"序列，按出无关的键 ✗），直接收工回战斗，
            # 并把下次休息提前（`_rest_retry_after_interrupt` 由 `_finish_rest` 消费）。
            self._release_ctx(self._afk_ctx)
            self._afk_ctx = None
            self.stop_route("定点休息被打断（补血了）")
            self._rest_note("定点休息被打断（补血了）⇒ 回战斗，下次提前")
            self._rest_retry_after_interrupt = True
            self._finish_rest(now)
            return
        self._release_ctx(self._afk_ctx)
        self._afk_ctx = None
        self.state = "afk_exit"
        self._rest_retry_after_interrupt = True

    def _finish_rest(self, now):
        """收工：**不管哪一型、从哪条路**退出休息，都在这里干净地回到战斗，并排下一次。

        ⚠ **这里自己兜底**（2026-09-26 用户批准的结构性收敛）：调用点来自好几处
        （tick 里的停自动 / 关防掉线 / 手动结束、休息状态机走完、定点休息的失败路径）——
        只要有一个忘了先松键，角色就会**卡着键**继续走 / 继续打 ✗，而那种 bug 在游戏里
        极难查 ✓。所以"松键、清标记、收路线"全放这儿，谁调都安全 ✓。
        """
        # ① **按键自保**：还按着的键松掉、序列标记清干净
        self._release_ctx(self._afk_ctx)
        self._afk_ctx = None
        # ⚠⚠ 循环行为在这儿是"**演完就收**"、不是"立刻杀"（用户 2026-09-28 更正：
        #   "结束休息不应该打断正在进行的循环"）—— 别换成 `_stop_spot_loop`（那会当场掐断+松键）。
        #   "立刻杀"留给停自动/关防掉线/切项目那条安全网（`_release_combat_keys`）。
        self._drain_spot_loop()
        self._spot_started = False
        # ⓪ **回战斗的这一段算「临时战斗」**（用户 2026-09-27 要求）：休息点多半不在
        #    战斗区域里，而"被打断"本身就说明有东西正在打我们 ⇒ 有怪就先打，别一路
        #    不还手地往回走 ✗（熄灭见 `tick` 里"这一波打完了"那一行）。
        self._temp_fight = True
        # ② **只收掉属于休息的那条路线**（定点休息"走过去"的任务）—— 用户自己下的
        #    「命令前往」不许连坐 ✗（它是有意"挂上就是授权、一路做到位"的）。
        if str(self._route_why or "").startswith("定点休息"):
            self.stop_route("休息结束")
        # ③ 状态复位
        self.state = "idle"
        self._rest_pending = False
        self._rest_last_tick = 0.0
        self._rest_until = 0.0
        self._rest_stop_pending = False     # ⭐ "手动结束"的收尾账也一起结清 ✓
        self.settings.rest_state = ""
        self.settings.rest_abort = False
        self.settings.rest_until_monotonic = 0.0
        self.settings.rest_pending = False
        # 休息期间没有战斗，朝向 / 玩家丢失这两个超时监控必须重新起算 ——
        # 否则休息时长一旦超过 facing_timeout_min（默认 10 分钟，模板里才 2 分钟），
        # 恢复的第一帧就会被判「朝向长时间未变化」直接停自动。
        self._last_facing_change = now
        self._player_lost_since = None
        if self._rest_retry_after_interrupt:
            # 这次是被补血打断的 → 按用户设的**秒数**重试，而不是随机 N~M 分钟
            self._rest_retry_after_interrupt = False
            self._next_afk = now + max(1.0, float(self.settings.anti_afk_retry_sec))
        else:
            self._next_afk = now + self._random_afk_interval()
        self.settings.next_afk_monotonic = self._next_afk   # UI 立刻能显示下次倒计时
        # ④ 打点（2026-09-26 补）：以后"它到底休息过没有 / 下次多久"看 perf.log 就够
        behavior.event("afk_done", kind=self.settings.anti_afk_type,
                       next_s=round(max(0.0, self._next_afk - now), 1))

    def _pause_timers(self, now):
        """暂停喂宠 / 自定义定时行为的计时器：把「下次触发时刻」随流逝时间一起往后推。

        效果 = 计时器冻住（剩余时间不变），UI 上的倒计时也自然停住。
        用增量推进而不是「休息结束时一次性补回」：休息途中关掉防掉线、异常退出，
        都不会把暂停时长算错。
        """
        s = self.settings
        last = self._rest_last_tick
        self._rest_last_tick = now
        if last <= 0:
            return
        delta = now - last
        if delta <= 0:
            return
        for k, v in list(s.custom_timer_next.items()):
            # ⚠ 只有**勾了「休息时暂停计时」**的那些才冻住（用户 2026-09-26 加的**项目级**
            #    开关，默认 True = 老行为）。没勾的照常倒数 ⇒ 休息期间也会到点、也会演 ✓
            #    （`_custom_timers` 在休息分支里本来就会跑 ✓）—— 给"休息时也要按的键"用 ✓。
            if v > 0 and self._timer_pauses_on_rest(k):
                s.custom_timer_next[k] = v + delta

    def _timer_pauses_on_rest(self, name):
        """这个定时行为在休息期间要不要**冻住计时** ⇒ bool（认不出来按 True = 冻住）。

        判据来自项目里那一条的 `pause_on_rest`（缺省 True = 老行为：休息时全都冻住 ✓）。
        设成 False ⇒ 休息期间它照常倒数、到点照演 ✓（按键那一层不受影响 ✓）。
        """
        t = next((x for x in self.settings.custom_timers
                  if (x.get("name") or "") == str(name or "")), None)
        return bool((t or {}).get("pause_on_rest", True))

    # ⚠ `_feed_pet`（自动喂宠）**已整块移除**（用户 2026-09-26）—— 他改用「自定义定时行为」
    #    自己实现 ✓（那条路走 `_custom_timers`，按键那层的「喂宠」键仍是现成的 ✓）。

    def _custom_timers(self, now):
        """自定义定时行为：每个行为每隔随机 [min,max] 分钟执行一次其序列。

        行为序列分帧执行（复用 _run_seq），执行期间不影响其他定时行为。
        另外处理**手动触发**（用户 2026-09-26 要求）：立刻演一次 + **计时从头开始** ✓。
        """
        s = self.settings
        # ---- 手动触发：界面往 `custom_timer_fire` 里塞名字，这里取走并立刻开演 ----
        # 为什么要有这条通道：按键与计时都在实时线程里 ⇒ 界面直接演会跟主回路抢按键 ✗
        #（和 `rest_request` / `rest_abort` 同一套做法 ✓）。
        # "计时从头开始"= 用 `now` 重新随机一个间隔 ✓（不是接着原来的剩余时间走 ✗）。
        # ⚠ 正在演的那一段要**先松键**再重排：否则它按下的键会一直卡着（序列被换掉了，
        #   没人再去 release 它）✗。
        pend = getattr(s, "custom_timer_fire", None)
        while pend:
            nm = str(pend.pop(0) or "")
            t = next((x for x in s.custom_timers
                      if (x.get("name") or "") == nm), None)
            # 名字对不上 / 这一项已暂停 ⇒ 丢掉请求（暂停的项在下面本来就跳过 ✗）
            if t is None or t.get("paused"):
                continue
            old = self._timer_states.pop(nm, None)
            if old is not None:
                self._release_ctx(old)
            iv = t.get("interval")
            lo, hi = ((max(0.0, float(iv[0])), max(0.0, float(iv[1])))
                      if isinstance(iv, (list, tuple)) and len(iv) == 2
                      else (0.0, 0.0))
            self._timer_states[nm] = [list(t.get("seq") or []), 0, 0.0, set(), []]
            s.custom_timer_next[nm] = now + random.uniform(lo, hi) * 60.0

        # ---- 孤儿清理（2026-09-26 用户批准的 ④ 项）----
        # 用户在序列**演到一半**时把那条定时行为删掉 ⇒ 它的上下文还留着、键还按着，
        # 而下面的循环再也不会碰它（名单里没这个名字了）⇒ **永久卡键** ✗ —— 更糟的是
        # 定期 RELEASEALL 现在还会把它**重按回去** ✗✗。所以名单一变就顺手收掉它。
        _names = {str(x.get("name") or "") for x in s.custom_timers}
        for nm in [k for k in self._timer_states if k not in _names]:
            st = self._timer_states.pop(nm)
            self._release_ctx(st)
            behavior.event("timer_orphan", name=nm)

        for t in s.custom_timers:
            name = t.get("name") or ""
            if not name:
                continue
            # 暂停：不排期、也不演序列。**正在演的那一段要立刻停下并松键** ——
            # 否则"暂停"只是不再触发下一次，手上这段序列还在按着键往下走。
            if t.get("paused"):
                st = self._timer_states.pop(name, None)
                if st is not None:
                    self._release_ctx(st)
                s.custom_timer_next.pop(name, None)
                continue
            seq = t.get("seq") or []
            iv = t.get("interval")
            if not isinstance(iv, (list, tuple)) or len(iv) != 2:
                continue
            lo = max(0.0, float(iv[0]))
            hi = max(lo, float(iv[1]))

            # 正在执行的序列：推进一步
            st = self._timer_states.get(name)
            if st is not None:
                r = self._run_seq(now, st)
                if r is None:
                    del self._timer_states[name]
                else:
                    self._timer_states[name] = r
                continue

            # 首次：随机一个间隔
            next_ts = s.custom_timer_next.get(name, 0.0)
            if next_ts <= 0.0:
                s.custom_timer_next[name] = now + random.uniform(lo, hi) * 60.0
                continue
            # 到点：启动序列，并随机下一个间隔
            if now >= next_ts:
                self._timer_states[name] = [seq, 0, 0.0, set(), []]
                s.custom_timer_next[name] = now + random.uniform(lo, hi) * 60.0

    def _drink_potions(self, ws, now):
        """自动喝药：血/蓝低于阈值就点按对应键，喝药冷却 pot_cd 内不再喝。

        **返回"这一拍是否真的喝了血药"** —— 休息期间它同时是「被打断」的信号
        （见 `_interrupt_rest`）。只有**真的按下去**的那一拍才返回 True，
        不是"血量低就一直 True"（否则休息会被同一个低血量反复打断）。
        """
        s = self.settings
        hp_pot = s.keymap.get("hp_pot")
        mp_pot = s.keymap.get("mp_pot")
        cd = max(0, float(s.pot_cd)) / 1000.0   # 喝药冷却（秒）
        drank_hp = False

        if s.auto_hp_pot and hp_pot and ws.player.hp < s.hp_threshold / 100.0:
            if now >= self._next_hp_pot:
                tap(hp_pot, self._attack_duration)
                self._next_hp_pot = now + cd
                drank_hp = True
                # ⭐ **喝药时把"触发那一刻的血"记下来**（用户 2026-09-28 要求 ✓）——
                #   以后能回答"死前到底喝到药没有、血掉到多少才触发"（原来连喝没喝都查不到 ✗：
                #   `drank_hp` 只被用来算"算不算一次输出"，从不落盘 ✓）。
                behavior.event("drink_hp", hp=round(float(ws.player.hp), 3),
                               thr=round(float(s.hp_threshold) / 100.0, 3))
        else:
            self._next_hp_pot = 0.0     # 血够了/没开自动补血，重置（下次低于阈值立刻补）

        if s.auto_mp_pot and mp_pot and ws.player.mp < s.mp_threshold / 100.0:
            if now >= self._next_mp_pot:
                tap(mp_pot, self._attack_duration)
                self._next_mp_pot = now + cd
        else:
            self._next_mp_pot = 0.0
        return drank_hp

    def _snap_beat(self, now, ws):
        """⭐ **位置 / 血量 / 动作的定频快照**（用户 2026-09-28 要求 ✓ 每秒各一条）。

        两条事件（与用户点名的三项一一对应 ✓）：
          · `pos`：`x` / `y`（**世界坐标** ✓）/ `sets`（**当前所在集合** ✓）/ `hp`（0~1 ✓）/ `st`（状态 ✓）；
          · `act`：`move`（这一拍任务给的方向 -1/0/+1 ✓）/ `keys`（**这一拍按着的键** ✓）/
            `task`（当前任务的相位，如 `align` ✓）。

        ⚠ 为什么这两条能把上次那种事**一眼看清**（这次花了十几轮反推 ✗）：
          · `pos` ⇒ "人在哪、脚下是哪块"（不用再拿 `climb_dx` 反推 ✓）；
          · `act` ⇒ "**有没有在按方向键**" ⇒ 两条一对就能分开
            「**人动不了**」（一直按着 → 位置不变 = 死了 / 卡死）和
            「**系统没发键**」（`move=0`、`keys` 空 → 位置不变 = 决策卡住）✗
            —— 这正是 2026-09-28 那次**分不出来**的地方 ✓。

        ⚠ 为什么**并成两条**而不是三四条：日志体积。一秒一条约 0.7 MB/小时；拆成三四条就是
          两三倍 ✗（`behavior.log` 本来就是 6 MB 级 ✓）。
        ⚠ 挂哪儿：tick 里**休息分支之前**（见调用处 ✓）—— 上次卡住的恰恰是「定点休息」的
          `afk_spot_walk`（属休息分支 ✓），挂在常规路径上**一条都打不出来** ✗。
        ⚠ 打点坏了不许影响决策（同其它打点 ✓）。
        """
        try:
            if now - self._snap_at < 1.0:
                return                      # 定频：1 秒一条 ✓
            self._snap_at = now
            p = getattr(ws, "player", None)
            wx = getattr(p, "world_x", None)
            wy = getattr(p, "world_y", None)
            hp = getattr(p, "hp", None)
            _sets = getattr(p, "here_sets", None) or ()
            behavior.event(
                "pos",
                x=(round(float(wx), 1) if wx is not None else ""),
                y=(round(float(wy), 1) if wy is not None else ""),
                sets=("/".join(str(x) for x in _sets) or ""),
                hp=(round(float(hp), 3) if hp is not None else ""),
                st=str(self.state))
            _keys = sorted(str(k) for k in self.keys.pressed())
            behavior.event("act", move=int(self._last_move),
                           keys=(",".join(_keys) or ""),
                           task=str(getattr(self._climb, "phase", "") or ""))
            self._last_move = 0             # ⚠ 读完清零（不然"没动"的那几拍会残留上一次 ✗）
        except Exception:                   # noqa: BLE001 —— 打点坏了别影响决策 ✗
            pass

    def _release_combat_keys(self):
        """释放打怪相关按键：KeyState + 回身输出/输出/防掉线序列。
        定时行为（custom_timers）的序列不在这里释放——它们不依赖玩家定位。
        释放后把序列上下文置空，避免下一帧重复发 RELEASE、也避免下次开自动
        时从旧序列中间继续导致按键状态错乱。

        ⭐ **「定点休息的循环行为」也在这里收摊** —— ⚠ 这处是"**松所有相关键**"的
        **总入口**（停自动 / 关防掉线 / 切项目 … 都会汇到它 ✓）⇒ 挂在这儿才**收得干净** ✓；
        只挂在 `_finish_rest` 上会漏掉那些"不走休息收尾"的路径 ⇒ 角色**卡着循环按的键**
        继续走 ✗（难查得要命 ✓）。

        ⚠⚠ **这里只做"立刻收摊"**（`_stop_spot_loop` 掐断 + 松键 ✓）—— 它是**安全网**
          （停自动 / 关防掉线 / 切项目 / **无怪 idle** 都要"键全松" ✓），**不许**为某一条
          路改成"演完再收" ✗。

        ⭐ 那"**演完再收**"（用户 2026-09-28 的口径："结束休息**不应该打断**正在进行的
          循环" ✓）由**休息分支自己**保证：手动结束 / 到点推迟都在 `afk_spot_rest` 里
          **等它演完**才收工（见 `tick` 里 `rest_abort` 那段 + `_hold_rest_for_loop` ✓）。
        ⚠⚠ **原来想靠"离开休息后由 `_loop_drain_beat` 在后台把它推完"✗ —— 这条路走不通**：
          回战斗之后只要**没怪**，本函数就被**每拍**调到（平地巡逻 idle 那条 ✓）⇒ 手上
          那一段**当场就被掐断** ✗（用户报"手动结束休息没收到寻路任务"那天挖出来的 ✓，
          `t_spot_loop_hold_rest_end` ⑤ 走真路径钉住 ✓）。
          ⇒ 结论：**要"演完"，就必须留在休息状态里等** ✓。
        """
        self.keys.release_all()
        self._release_ctx(self._back_ctx)
        self._release_ctx(self._output_ctx)
        self._release_ctx(self._afk_ctx)
        # ⚠ 循环行为**不是** `_afk_ctx`（它是休息中另起的一个 ctx ✓）⇒ 必须**单独**收摊 ✓
        self._stop_spot_loop()
        self._back_ctx = None
        self._output_ctx = None
        self._afk_ctx = None
        self._stand_attack = False  # 中止战斗：清掉「站桩输出」标记，别带到下一轮

    def _release_held_keys(self):
        """释放所有还按着的键：KeyState 的 + 序列（回身输出/防掉线/定时行为）残留的。"""
        self._release_combat_keys()
        for st in self._timer_states.values():
            self._release_ctx(st)
        self._timer_states.clear()

    def shutdown(self):
        from decision import input as dinput
        # 远程模式：发一次 RELEASEALL，让固件一次性清空所有按键（比逐键 RELEASE 可靠，
        # 能救回「某条 RELEASE 丢失导致卡键」的情况）
        dinput.release_all_remote()
        self._release_held_keys()
        # 兜底：无条件释放所有映射键（功能键 + 自定义按键）。KeyState 的
        # release_all 只释放「它记得按下的键」，若某次 RELEASE 命令在网络里丢了、
        # 固件却仍按着，KeyState 会误以为已释放、之后不再补发 —— 所以这里对
        # 所有映射键再补一次 RELEASE，对未按下的键发 RELEASE 无害（固件/本地都忽略）。
        keys = set(self.settings.keymap.values()) | set(self.settings.custom_keys.values())
        for key in keys:
            if key:
                try:
                    key_up(key)
                except Exception:
                    pass
        self.state = "idle"
        self.settings.rest_state = ""
        self.settings.rest_abort = False
        self.settings.rest_until_monotonic = 0.0
        self.settings.rest_pending = False
        self.settings.next_afk_monotonic = 0.0


def load_rect(v):
    """HP/MP 条区域 [nx, ny, nw, nh] —— 存的是**相对画面的比例**（0~1）。

    必须保留小数：曾经用 int(v) 读回，0.125 被截成 0，重开 GUI 就全变成
    [0,0,0,0]（HP/MP 条凭空消失）。零尺寸也当成「没框」。
    """
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            r = [float(x) for x in v]
        except Exception:
            return None
        if r[2] > 0 and r[3] > 0:
            return r
    return None


# 全局决策设置（单例）：GUI 主线程写，实时推理线程读。
# 用单例让「决策参数」页签和「实时」页共享同一份配置，不用层层传引用。
# **启动时不 load**：已经没有「全局那份」可读了 —— 界面起来后由
# gui/main_window.py 的 _bind_decision_params 决定用哪个项目的参数
# （没打开项目时用最近打开的那个；再没有就是这里的默认值）。
settings = DecisionSettings()
