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

from core import perf
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
# 输出CD / delay / 跳间隔足够（抓帧节拍是 66.7 毫秒，两者差一个数量级）。
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
        self.attack_dist = 80.0     # 最大攻击距离（像素，从角色中心量到怪框最近的边）
        self.min_attack_dist = 0    # 最小攻击距离（同口径），>0 时启用规避
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
        self.input_delay = [70, 130]  # 随机输入延迟 [min, max]（毫秒），按键之间
        # 输出行为 CD（毫秒）：两次输出之间的**最小间隔**，从上次输出时刻起算
        # （实际间隔 = CD + 随机延迟；输出序列自身更长时以序列为准）
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
        self.align_tol_px = 6       # 坐标对齐误差范围（像素）
        self.align_hold_ms = 250    # 坐标对齐误差时间（毫秒）
        # **寻路超时时间（秒）**（用户 2026-09-26 要求）：一个寻路任务（走 / 爬 / 下跳）
        # 从**下达那一刻**算起持续这么久还没结束 ⇒ 切断（如实说明，不再重试）。
        # 为什么不能只靠任务自己的 `timeout_s`：失败重来会重置任务内部的计时
        #（`ClimbJob.retry` 清 `_t0`、`WalkJob.retry` 同样）⇒ 累计可能远超预期。
        # 0 = 不限时（老行为）。见 `agent._climb_tick` 里那道总闸。
        self.goto_timeout_s = 30
        # ⚠ 这里**曾经**有 `walk_mode`（走的方向：center / left / right，还摆进了设置里）。
        # 2026-09-26 用户明确删掉：**走只有一种走法 —— 朝目标集合的 x 中点**，
        # 设置里不许有这个参数（设置是全局参数，用户没提过的一律不加）。
        # 将来若某一步真需要"只按 ←/→"，那是**逐边**配置（foothold 编辑器 →「可到达」窗口）。
        # 上绳梯/下跳**失败后延迟激活时间**（秒，2026-09-26 用户要求）：
        # 以前失败了是**立即**重新激活 —— 失败那一下人往往还在原地、朝向也没变，
        # 立刻重来容易在同一处再歪一次。等一会儿再重新对齐，成功率更高。
        # 0 = 立即重来（老行为）。参数在「路线识别」页（`gui/route_panel.py`）。
        self.climb_retry_delay_s = 1.0
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
                "input_delay": self.input_delay,
                "attack_cd": self.attack_cd,
                "attack_lock_debounce_ms": self.attack_lock_debounce_ms,
                "min_turn_hold_ms": self.min_turn_hold_ms,
                "turn_output_delay_ms": self.turn_output_delay_ms,
                "player_track_jump": self.player_track_jump,
                "hp_bar": self.hp_bar, "mp_bar": self.mp_bar,
                "probe_calib": self.probe_calib,
                "hp_color": self.hp_color, "mp_color": self.mp_color,
                "hp_threshold": self.hp_threshold,
                "mp_threshold": self.mp_threshold,
                "pot_cd": self.pot_cd,
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
                "climb_retry_delay_s": self.climb_retry_delay_s,
                "goto_timeout_s": self.goto_timeout_s,
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
        dly = data.get("input_delay")
        if isinstance(dly, (list, tuple)) and len(dly) == 2:
            self.input_delay = [int(dly[0]), int(dly[1])]
        self.attack_cd = int(data.get("attack_cd", 0))
        self.attack_lock_debounce_ms = int(data.get("attack_lock_debounce_ms", 300))
        self.min_turn_hold_ms = int(data.get("min_turn_hold_ms", 0))
        self.turn_output_delay_ms = int(data.get("turn_output_delay_ms", 0))
        # 旧键 player_debounce_dist 兼容读一次（改名前的配置 / 模板里是这个）
        self.player_track_jump = int(data.get(
            "player_track_jump", data.get("player_debounce_dist", 150)))
        self.hp_bar = load_rect(data.get("hp_bar"))
        self.mp_bar = load_rect(data.get("mp_bar"))
        self.probe_calib = dict(data.get("probe_calib") or {})
        self.hp_color = self._load_color(data.get("hp_color"))
        self.mp_color = self._load_color(data.get("mp_color"))
        self.hp_threshold = int(data.get("hp_threshold", 30))
        self.mp_threshold = int(data.get("mp_threshold", 20))
        self.pot_cd = int(data.get("pot_cd", 1000))
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
        self.align_tol_px = int(data.get("align_tol_px", 6))
        self.align_hold_ms = int(data.get("align_hold_ms", 250))
        self.climb_retry_delay_s = float(data.get("climb_retry_delay_s", 1.0))
        self.goto_timeout_s = float(data.get("goto_timeout_s", 30.0))
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
        #: 失败后**等哪一刻再重新激活**（`time.monotonic()` 秒；None = 没在等）。
        #: 由 `_climb_tick` 设/清，延迟长短读 `settings.climb_retry_delay_s`。
        self._climb_retry_at = None
        #: 这个寻路任务是**哪一刻**下的命令（给"寻路超时时间"那道总闸用）。
        self._climb_started = 0.0
        #: **多步路径**：还没跑的任务队列 + 进度（见 `start_route`）。
        self._route = []
        self._route_why = ""
        self._route_step = 0
        self._route_total = 0
        #: 由**实时线程**注入的**路径解析器**：`dst_set → {"path","jobs","why","here"}`
        #:（见 `gui/live_thread._make_route_resolver` 与 `decision.route.plan_jobs`）。
        #: 为什么是注入而不是自己算：地形与集合数据在实时线程手里
        #:（`_fill_route_ctx` 同一处），agent 不持有它们 —— 不发明第二套数据源。
        #: ⚠ 它是**函数**，不是预存的表：路径的起点是"我现在站哪个集合"，每一刻都可能变
        #:（`docs/开发计划.md` 原先写的是 `set_name → [job,…]` —— 那样只能在某一刻算一次 ✗）。
        self.route_plan = None
        #: 「定点休息」当前那个"要走出去"的阶段**出发了没有**（见 `_run_rest_spot`）。
        #: 为什么要有它：路径一结束时 `_climb` 会被清空 —— 可"还没出发"和"走完了"都长这样，
        #: 必须分得清（不然刚进休息那一拍就会被当成"走完了"或"没走成" ✗）。
        self._spot_started = False
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
        self._link_checked = 0.0    # 上次指令通道体检时刻
        self._link_retry = 0.0      # 上次尝试重连通道的时刻（避免狂重连）
        self._next_afk = 0.0        # 下次防掉线触发时刻
        self._was_afk_enabled = False  # 防掉线开关上一次状态（上升沿检测）
        self._afk_ctx = None        # 当前防掉线阶段的序列上下文（进入/退出隐身）
        self._rest_pending = False  # 已到防掉线触发时间，等攻击范围内的怪清空
        self._rest_last_tick = 0.0  # 上次推进「暂停计时器」的时刻
        self._rest_until = 0.0      # 本次休息的结束时刻
        self._rest_retry_after_interrupt = False  # 这次休息被补血打断 → 下次按 retry_sec 排

    @staticmethod
    def _center_dist(m, player):
        """**角色中心点**到怪框最近边缘的水平距离（像素）；0 = 怪框已盖住角色中心。

        起算点用角色中心，不用玩家框的朝向边缘：够不够得着是看角色，而玩家框是
        贴图框（宽度随动作变），拿它的边缘起算，同一只怪的距离会跟着框宽漂。
        终点仍取怪框最近的边 —— 大框目标（龙）不必再靠近半个框宽才算够得着。
        只算水平：横版先只做水平方向找怪打（见模块开头）。

        这个距离是**共用口径**：攻击判定 / 最近怪 / 最小距离规避 / 追击起跳 /
        背后锁定 全走它，实时预览画的攻击距离线也按它起算（gui/live_thread.py）。
        """
        m_left = m.x - m.w / 2.0
        m_right = m.x + m.w / 2.0
        if m_right < player.x:
            return player.x - m_right   # 怪完全在中心左侧
        if m_left > player.x:
            return m_left - player.x    # 怪完全在中心右侧
        return 0.0                      # 怪框跨过中心 → 贴身

    def _nearest(self, mobs, player):
        """找最近的怪（角色中心 → 怪框边缘的水平距离），返回 (target, dist)。

        没怪返回 (None, None)。
        """
        target, best = None, None
        for m in mobs:
            d = self._center_dist(m, player)
            if best is None or d < best:
                best = d
                target = m
        return target, best

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
        perf.count("jump")
        attack_cd_s = max(0.0, float(s.attack_cd)) / 1000.0
        self._next_chase_jump = now + max(0.3, attack_cd_s + self._random_input_delay())

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
        perf.count("key_stuck")
        perf.note("key_stuck", "%s 按住 %.1fs（当时状态 %s）—— 已松开"
                  % ("/".join(sorted(held)), now - self._out_held_since, self.state))
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
        """朝目标方向走：按对应方向键并更新朝向。dx=0 不动。"""
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

    def start_route(self, jobs, why=""):
        """跑一条**多步路径**（「命令前往」/「定点休息」共用）→ True = 已起跑。

        `jobs` 是**已经解析好**的任务列表（每步一个 `route.job_for_edge` 的产物：
        WalkJob / ClimbJob / DropJob）—— 路径怎么解析由**实时线程**负责
        （它手里才有地形与集合，见 `route_plan` 的说明），这里只管按顺序跑。

        每一步跑完（到了）自动接下一步；**任何一步失败就整条停掉**并把"断在第几步、
        为什么"写进 note（如实说，别硬着头皮往下走 ✗）。
        """
        jobs = list(jobs or [])
        if not jobs:
            return False
        self._route = jobs[1:]
        self._route_why = str(why or "")
        self._route_step = 1
        self._route_total = len(jobs)
        perf.count("route_start")
        self.start_climb(jobs[0])
        return True

    def stop_route(self, why="取消寻路"):
        """把整条路径停掉（含还没跑的那些步）。"""
        self._route = []
        self._route_step = self._route_total = 0
        self.stop_climb(why)

    def plan_and_start_route(self, dst_set, why=""):
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
            return False, "没选地点（先去玩家面板里选一个集合）"
        fn = self.route_plan
        if not callable(fn):
            return False, ("实时线程还没把「路径解析器」交给我 —— 先在「实时」页开始，"
                           "命令才发得出去")
        try:
            res = fn(dst)
        except Exception as ex:                  # noqa: BLE001
            return False, "解析路径时出错：%s" % ex
        if not isinstance(res, dict):
            return False, "路径解析器给的东西看不懂：%r" % (res,)
        jobs = list(res.get("jobs") or [])
        if not jobs:
            if res.get("here"):
                return True, str(res.get("why") or ("已经在「%s」上了" % dst))
            return False, str(res.get("why") or "解析不出路线")
        if not self.start_route(jobs, why=why or ("前往：%s" % dst)):
            return False, "路径是空的，没起跑"
        return True, ("已出发：%s" % " → ".join(res.get("path") or [dst]))

    def _task_finished(self, failed=False, why=""):
        """当前这一**步**收工（到了 / 失败了）⇒ 推进多步路径 → True = 整条都完事了。

        为什么单独一个方法：任务收工的点有好几个（到了 / 到集合收工 / 超时 / 没坐标 /
        到次数上限放弃），**每个点都要决定"下一步"** —— 散着写必然漏一个 ——
        所以统一收口到这里。
        """
        job = self._climb
        self._climb = None
        self._climb_retry_at = None
        self._release_combat_keys()
        note = str(why or getattr(job, "note", "") or "")
        if note:
            self._last_goto_note = (note, time.monotonic())
        if failed:
            if self._route:
                perf.count("route_break")
                self._last_goto_note = (
                    "路径中断（第 %d/%d 步）：%s；后面 %d 步没走"
                    % (self._route_step, self._route_total, note, len(self._route)),
                    time.monotonic())
                self._route = []
                self._route_step = self._route_total = 0
            return True
        if self._route:
            nxt = self._route.pop(0)
            self._route_step += 1
            perf.count("route_step")
            self.start_climb(nxt)
            return False            # 这一拍不算完事：下一拍接着走下一步
        if self._route_total > 1:
            perf.count("route_done")
            self._last_goto_note = ("整条路径走完（共 %d 步）" % self._route_total,
                                    time.monotonic())
        self._route = []
        self._route_step = self._route_total = 0
        return True

    def start_climb(self, job):
        """挂上一个**上绳/下跳任务**（`route.ClimbJob` / `DropJob`）；下一次 tick 开始执行。

        谁来调：`gui/route_panel.py` 的「命令前往」（把路线的第一步交过来，2026-09-26）。
        这里**没有"要不要按跳"的开关**了 —— 命令挂上来就是授权，一路做到位。
        """
        self._climb = job
        self._climb_retry_at = None      # 清掉上一个任务留下的"等我到点再重来"
        self._climb_started = time.monotonic()   # 「寻路超时时间」从这一刻算起
        perf.count("climb_start")
        return job

    def stop_climb(self, why="取消上绳"):
        """撤掉上绳任务（换目标 / 关自动 / 出错都走这里）：状态收干净，别再按键。"""
        if self._climb is None:
            return
        self._climb.cancel(why)
        self._climb = None
        self._climb_retry_at = None
        self._release_combat_keys()
        perf.count("climb_cancel")
        if self.state == "climb":
            self._set_state("idle")

    def current_goto_set(self):
        """当前**寻路任务**的目标集合名（没有任务时空串）。

        给界面用（「路线识别」页的「当前任务」那行、以及「结束当前寻路」按钮）：
        寻路任务就是挂着的上绳/下跳 job（`start_climb`），它带着 `dst_set`。
        **别再让界面去读 `_climb`** —— 那是私有的运行时状态，改一次名字就全线崩。
        """
        job = self._climb
        return str(getattr(job, "dst_set", "") or "") if job is not None else ""

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

    def resetall_left(self):
        """离下次**定时清键**（RELEASEALL）还有几秒；没排期时给 None。

        给界面用（「当前任务」下面那几行计时）：和 `_next_resetall` 同一把钟
        （`time.monotonic` 秒）。**排期发生在 tick 里**，所以刚开自动那一拍到排期
        之间就是 None —— 界面据此写"未排期"，别显示 0:00 骗人。
        """
        if self._next_resetall <= 0:
            return None
        return max(0.0, self._next_resetall - time.monotonic())

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
        # **寻路超时**（用户 2026-09-26 要求，当天又明确了口径）：按**每一段**算 ——
        # **每完成一段**（从一个集合走到另一个集合）就重新计时 ✓。
        # 代码上靠 `start_climb()` 重打 `_climb_started`，而 `_task_finished` 接下一段时
        # 正是调它 ✓（所以"多段路线共用一次超时"是**不会**发生的 —— 用例
        # `t_goto_timeout_per_hop` 钉着，别把它改成整条路线一个钟 ✗）。
        # ⇒ 它挡的是"**某一段**卡住 / 在一段里反复重试"，不是"整条路线太久"：
        #    5 段各自都正常的路线，总耗时可以远超这个值，那**不算超时** ✓。
        cap_s = max(0.0, float(getattr(self.settings, "goto_timeout_s", 0.0) or 0.0))
        if cap_s > 0 and self._climb_started and (now - self._climb_started) > cap_s:
            job.cancel("寻路超时：这一段已经跑了 %.0f 秒（上限 %.0f 秒）"
                       "—— 先看它卡在哪一步" % (now - self._climb_started, cap_s))
            perf.count("goto_timeout")
            return self._task_finished(failed=True)
        # 任务被清掉之后，界面还要能回答"刚才为什么松开了 ↑" ⇒ 每一拍把最新那句留一份
        #（`current_goto_note` 在任务已结束时也能给出它，见那里的说明）。
        self._last_goto_note = (str(getattr(job, "note", "") or ""), now)
        if wx is None:
            # 定不了位（小地图那条没跑 / 没认出黄点）⇒ **如实失败**，绝不拿画面坐标硬凑
            #（那正是上面那个死循环的成因）。这里直接放弃：坐标拿不到不是"再试一次"能好的。
            perf.count("climb_giveup")
            job.cancel("拿不到世界坐标（小地图定位没有输出）—— 先确认「实时」页在跑、"
                       "小地图那块能认出黄点")
            return self._task_finished(failed=True)
        py = getattr(p, "world_y", None)
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
        out = job.update(now, float(wx), py=py,
                         ladder_id=getattr(p, "ladder_id", None),
                         here_sets=getattr(p, "here_sets", None))
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
        if out["failed"]:
            # ① **已经到了目标集合 ⇒ 直接收工**（2026-09-26 用户要求 2）。
            #    失败判定可能晚于实际到达（定位抖一下、或者到了之后又被打下来），
            #    这时再"重新激活"就是"人已经站上去了、还在原地爬" —— 比不做更糟。
            #    判据用**集合**（`dst_set`，人工圈的、最贴近"到了哪块平台"）。
            here = set(getattr(p, "here_sets", None) or ())
            if job.dst_set and job.dst_set in here:
                # 单独记一笔：**它是"失败后别再重试"的兜底，不是到达判据** ——
                # 日志里要和几何到达（climb_done）分开，否则又看不出是哪种结束的 ✗。
                perf.count("climb_done_set")
                # 人已经在目标集合里 ⇒ 这一步算**到了**（多步路径继续往下走）
                return self._task_finished(failed=False)
            if job.attempt < job.max_attempts:
                # ② **失败保护**：失败不是终止 —— 重新激活再来一次（上绳本来就容易歪
                #    一下：偏离绳 / 掉下来都算）。但要**延迟**激活（要求 1）：
                #    等待期间**不按键**（这一拍只返回 False ⇒ 状态还是 climb、keys 空
                #    ⇒ 角色站着），到点才重新对齐。
                delay = max(0.0, float(
                    getattr(self.settings, "climb_retry_delay_s", 0.0) or 0.0))
                if delay <= 0.0:
                    perf.count("climb_retry")
                    job.retry()
                    return False         # 这一帧不算完事，下一帧从"重新对齐"接着走
                if self._climb_retry_at is None:
                    self._climb_retry_at = now + delay
                    perf.count("climb_retry_wait")
                    job.note = "%s（等 %.1fs 后重新对齐，已试 %d/%d 次）" % (
                        job.note, delay, job.attempt, job.max_attempts)
                    return False
                if now < self._climb_retry_at:
                    return False         # 还在等：这一拍不按键
                self._climb_retry_at = None
                perf.count("climb_retry")
                job.retry()
                return False
            # ③ 到上限才真放弃（数据写错时不能无限重来 —— 那看着就像卡死）
            perf.count("climb_giveup")
            return self._task_finished(failed=True)
        if out["done"]:
            perf.count("climb_done")
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
                next_ts = now + self._random_input_delay()
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
            perf.count("key_reassert")
            perf.note("key_reassert",
                      "清空指令通道后重按 %s（序列进度保留）" % "/".join(keys))
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

    def _random_input_delay(self):
        """随机输入延迟（秒）：所有点按类按键（攻击/补血/补蓝）之间的间隔，
        从 [min, max] 毫秒区间随机取，模拟人手的不规律节奏。"""
        lo, hi = self.settings.input_delay
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

    def _candidates(self, mobs, px):
        """锁定候选：平地巡逻锁全部；扫平台只锁背后 back_range 内的怪。
        （朝向方向的怪不锁定，靠 attack 优先级就近攻击。）
        「背后」相对倾向朝向（_patrol_dir），与打背后怪时的临时转身无关。"""
        s = self.settings
        if s.strategy == "sweep":
            back = max(0.0, float(s.back_range))
            return [m for m in mobs if -back <= (m.x - px) * self._patrol_dir < 0]
        return mobs

    def _in_range(self, mobs, ws):
        """朝向前方、攻击范围内的框（角色中心 → 怪框边缘 <= attack_dist），按距离升序。

        攻击只朝前方：背后的框即使水平距离近也不算在攻击范围内，走 chase 转身。
        """
        ad = self.settings.attack_dist
        px = ws.player.x
        return sorted([m for m in mobs
                       if (m.x - px) * self.facing >= 0
                       and self._center_dist(m, ws.player) <= ad],
                      key=lambda m: self._center_dist(m, ws.player))

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

        target_too_close = (min_dist > 0 and best < min_dist)
        any_too_close = (min_dist > 0 and
                         any(self._center_dist(m, ws.player) < min_dist for m in mobs))

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
        self.state = new_state
        perf.count("st_" + str(new_state))      # 状态分布：抖动（攻击↔追击）一眼可见

    def _locked_target(self, lockable, ws, now):
        """chase 用的锁定目标：target_cd 机制，无抢锁。返回 (target, best)。"""
        target = None
        if self._target_id is not None:
            for m in lockable:
                if m.id == self._target_id:
                    target = m
                    break

        if target is not None and now < self._target_until:
            best = self._center_dist(target, ws.player)
            self._no_target_since = None
        else:
            target, best = self._nearest(lockable, ws.player)
            if target is not None:
                self._target_id = target.id
                self._target_until = now + self._random_target_cd()
                self._no_target_since = None
            else:
                self._target_id = None
                self._target_until = 0.0
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
        绝不会让两轮序列重叠。末尾再叠一点随机延迟做人工化抖动。
        """
        return max(now, self._last_output + cd) + self._random_input_delay()

    def _output_actions(self, now, target, mobs, ws):
        """按当前状态输出动作：attack 连点 / evade_jump 跳+输出 / evade_back_jump 序列。"""
        s = self.settings
        attack_cd = max(0.0, float(s.attack_cd)) / 1000.0
        attack_key = s.keymap["attack"]

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
                    perf.count("jump")
                self._pending_attack = now + interval
                self._next_evade = now + interval + attack_cd + self._random_input_delay()
            if (self._pending_attack is not None and now >= self._pending_attack
                    and not turn_wait):
                # 输出这一下也受「转向后输出延迟」约束：跳可以先跳（是位移，
                # 不吃朝向），但攻击键要等转身做完，否则打向错误的方向。
                # 没到点就继续挂着 _pending_attack，下一帧再补。
                tap(attack_key, self._attack_duration)
                perf.count("out_evade")
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

        # 通道自愈 + 定时 RELEASEALL。**必须放在下面 `if not s.enabled` 早退之前**：
        # 卡键最常见的时刻恰恰是「发现卡了 → 去关自动」之后。如果关自动就不发，
        # 这个自愈机制在真正需要它的时候正好是关着的 —— 之前就踩了这个坑：
        # 一关自动就再也救不回来，只能关 GUI（那一下是 relay 在连接断开时
        # 替我们发的 RELEASEALL 救的，见 decision/input.py 的 reconnect_remote）。
        self._periodic_reset(now, s)

        if not s.enabled:
            if timing_only:
                return None      # 停自动的收尾由决策拍做（这里做的事都是幂等的，不必重复跑）
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
                        self.stop_route("手动结束休息")
                        self._finish_rest(now)
                        return {"state": self.state, "reason": "手动结束休息",
                                "target": None, "dx": 0, "dist": 0, "keys": [],
                                "facing": self.facing,
                                "kill_mobs": self._take_kill_mobs()}
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

        if not ws.player.found:
            # 找不到玩家超时：连续超时就停止自动
            lost_timeout = max(0.0, float(s.player_lost_timeout_min)) * 60.0
            if lost_timeout > 0:
                if self._player_lost_since is None:
                    self._player_lost_since = now
                elif now - self._player_lost_since >= lost_timeout:
                    s.enabled = False
                    self._release_combat_keys()
                    self.state = "idle"
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

        # 卡键看门狗（每拍一次；用**上一拍**的状态判断，见方法说明）
        self._watch_output_held(now)

        # 自动喝药：依赖玩家定位（读血/蓝），定位到玩家后才执行
        self._drink_potions(ws, now)

        # 锁定候选（sweep 只背后 back_range 内的怪；patrol 全部）
        candidates = self._candidates(mobs, px)

        # attack 最高优先级：只要前方攻击范围内有框，就进入攻击（或规避）
        in_range = self._in_range(mobs, ws)

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

        if in_range:
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
                # 有锁定目标（sweep 背后怪 / patrol 最近怪）：朝它走
                self._steer(target.x - px, keys)
                # 追击起跳：本分支攻击范围内本来就是空的，不存在「其他怪」，前提天然满足
                self._maybe_chase_jump(best, now, edge=jump_edge)
                self._set_state("chase")
                # 打点：追的目标离多远、以及它是不是「幽灵框」（missed>0，即已经
                # 漏检、靠防抖保留着位置的框）。chase_ghost 占追击时间越多，说明
                # debounce_ms 保留太久 —— 角色正朝一个已经消失的框走过去。
                perf.sample("chase_dist", best or 0.0)
                if getattr(target, "missed", 0) > 0:
                    perf.count("chase_ghost")
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
                return {"state": "idle", "reason": "无怪",
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
        # 上绳任务自己管方向键（对齐时要求"站住别动"）⇒ 关掉补键（cap 0 = 不干预）
        cap = 0.0 if self._climb is not None else (
            TURN_TAP_S if self._stand_attack else None)
        keys = self._hold_turn(keys, now, cap_s=cap)
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
            perf.count("afk_start")
            perf.note("afk_start", "定点休息：去「%s」"
                      % (self.settings.anti_afk_spot_set or "(没选地点)"))
        else:
            # 「隐身休息」：进入隐身序列 → 等 → 退出隐身序列
            perf.count("afk_start")
            perf.note("afk_start", "隐身休息：进入隐身")
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
            else:
                self._afk_ctx = r
        elif self.state == "afk_spot_rest":
            if now < self._rest_until:
                return
            self._afk_ctx = None
            after = str(s.anti_afk_spot_after or "").strip()
            if not after:
                self._finish_rest(now)                  # ⑤ 回战斗
                return
            self.state = "afk_spot_back"                # ④ 结束后前往
            self._spot_started = False
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
                tgt = in_range[0]                 # 最近的（`_in_range` 已按距离升序 ✓）
                best = self._center_dist(tgt, ws.player)
                keep = self.state
                self.state = "attack"             # 只借名字，让输出那套（按状态选动作）跑起来
                try:
                    out |= self._attack_state(tgt, best, mobs, ws)
                    self._output_actions(now, tgt, mobs, ws)
                finally:
                    self.state = keep             # ← 不还原就等于放弃这次休息 ✗
                perf.count("spot_travel_fight")
                return out
        if self._climb is not None:
            self._climb_tick(now, getattr(ws.player, "world_x", None), out, ws)
        return out

    def _rest_give_up(self, now, why):
        """「定点休息」走不到 ⇒ **如实说 + 收工回战斗**（不许硬走，也不许静默 ✗）。"""
        perf.count("afk_give_up")                     # 打点：这类"没休息成"要能事后数出来
        perf.note("afk_give_up", "定点休息走不到：%s" % why)
        self._rest_note("定点休息：%s" % why)
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
        # 局部 import：这里是每分钟几次的量级，不在热路径上；放模块头会把 core.perf
        # 拉进 decision 层的依赖里，没必要。
        try:
            from core import perf
            perf.count("afk_interrupt")
        except Exception:
            pass
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
        self._spot_started = False
        # ② **只收掉属于休息的那条路线**（定点休息"走过去"的任务）—— 用户自己下的
        #    「命令前往」不许连坐 ✗（它是有意"挂上就是授权、一路做到位"的）。
        if str(self._route_why or "").startswith("定点休息"):
            self.stop_route("休息结束")
        # ③ 状态复位
        self.state = "idle"
        self._rest_pending = False
        self._rest_last_tick = 0.0
        self._rest_until = 0.0
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
        perf.count("afk_done")
        perf.note("afk_done", "休息收工（类型 %s）⇒ 回战斗，下次 %.0f 秒后"
                  % (self.settings.anti_afk_type, max(0.0, self._next_afk - now)))

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
            perf.count("timer_orphan")
            perf.note("timer_orphan", "定时行为「%s」已经不在列表里 ⇒ 松开它按着的键"
                      % nm)

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
        else:
            self._next_hp_pot = 0.0     # 血够了/没开自动补血，重置（下次低于阈值立刻补）

        if s.auto_mp_pot and mp_pot and ws.player.mp < s.mp_threshold / 100.0:
            if now >= self._next_mp_pot:
                tap(mp_pot, self._attack_duration)
                self._next_mp_pot = now + cd
        else:
            self._next_mp_pot = 0.0
        return drank_hp

    def _release_combat_keys(self):
        """释放打怪相关按键：KeyState + 回身输出/输出/防掉线序列。
        定时行为（custom_timers）的序列不在这里释放——它们不依赖玩家定位。
        释放后把序列上下文置空，避免下一帧重复发 RELEASE、也避免下次开自动
        时从旧序列中间继续导致按键状态错乱。"""
        self.keys.release_all()
        self._release_ctx(self._back_ctx)
        self._release_ctx(self._output_ctx)
        self._release_ctx(self._afk_ctx)
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
