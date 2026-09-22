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

import json
import random
import time
from pathlib import Path

from decision.input import DEFAULT_KEYMAP, KeyState, key_down, key_up, tap

_SETTINGS_FILE = Path(__file__).resolve().parent.parent / "config" / "decision.json"

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


class DecisionSettings:
    """决策参数（UI 写，决策线程读），改动后持久化到 config/decision.json。"""

    def __init__(self):
        self.enabled = False
        self.attack_dist = 80.0     # 最大攻击距离（画面像素），小于它就打
        self.min_attack_dist = 0    # 最小攻击距离（像素），>0 时启用规避
        self.chase_jump_dist = -1   # 追击起跳距离（像素）：chase 时目标在此范围外延就跳；<=0 无效
        self.evade_type = "jump"    # 规避类型："jump" 跳 / "back" 后退
        self.jump_interval = 200    # 跳间隔（毫秒）：跳键和输出键之间的间隔
        self.jump_random_prob = 0.1 # 乱跳几率（0~1）：正常攻击时随机跳+输出的概率
        self.back_jump_seq = [dict(e) for e in DEFAULT_BACK_JUMP_SEQ]  # 回身输出行为序列
        self.output_seq = [dict(e) for e in DEFAULT_OUTPUT_SEQ]        # 输出行为序列（attack 状态执行）
        self.keymap = dict(DEFAULT_KEYMAP)
        self.target_cd = [500, 1000]  # 目标切换 CD [min, max]（毫秒）
        self.input_device = "local"   # 输入设备："local" 本机 / "remote" Pro Micro
        self.input_delay = [70, 130]  # 随机输入延迟 [min, max]（毫秒），按键之间
        self.attack_cd = 0          # 输出行为 CD（毫秒）：攻击/跳输出/反向跳回身等行为的节奏 = CD + 随机延迟
        self.attack_lock_debounce_ms = 300  # 输出后锁定防抖（毫秒）：这段时间内 attack 目标保持锁定，不切换到攻击范围内其他框
        self.player_debounce_dist = 150     # 玩家识别防抖距离（像素）：玩家框中心跳变超过此距离则沿用上一帧位置
        self.hp_bar = None          # HP 条区域 (x, y, w, h) 或 None
        self.mp_bar = None          # MP 条区域 (x, y, w, h) 或 None
        self.hp_color = None        # HP 填充色范围 [[B,G,R],[B,G,R]] 或 None（默认红）
        self.mp_color = None        # MP 填充色范围 [[B,G,R],[B,G,R]] 或 None（默认蓝）
        self.hp_threshold = 30      # HP 阈值（百分比 0~100）
        self.mp_threshold = 20      # MP 阈值（百分比 0~100）
        self.pot_cd = 1000          # 喝药冷却（毫秒），喝完后这段时间不再喝
        self.auto_hp_pot = False    # 自动补血开关
        self.auto_mp_pot = False    # 自动补蓝开关
        self.strategy = "patrol"    # 战斗策略类型："patrol" 平地巡逻
        self.vision_top = 200       # 向上视野（像素），<0 = 不限制
        self.vision_bottom = 200    # 向下视野
        self.vision_left = 200      # 向左视野
        self.vision_right = 200     # 向右视野
        self.vision_center = False  # 基于画面中心：勾选则以画面屏幕中心为基准，否则以角色为中心
        self.vision_off_x = 0       # 视野框 x 偏移（像素）
        self.vision_off_y = 0       # 视野框 y 偏移（像素）
        self.sweep_turn_cd = 1000   # 扫平台：当前朝向没怪持续此时间（毫秒）才换朝向
        self.back_range = 100       # 扫平台：允许锁定背后多远距离内的怪（像素）
        self.route_enabled = False  # 路线识别：平台识别/跳跃预测/扫平台 总开关（默认关，关掉不跑不卡）
        self.debounce_conf = 0.5    # 防抖置信度：高于它的怪框消失后保留位置
        self.debounce_ms = 300      # 防抖时间（毫秒）：保留消失前位置的时长
        self.auto_feed_pet = False  # 自动喂宠
        self.feed_interval_min = 5  # 喂宠间隔下限（分钟）
        self.feed_interval_max = 10 # 喂宠间隔上限（分钟）
        self.feed_next_monotonic = 0.0   # 下次喂宠时刻（time.monotonic），UI 倒计时读；运行时状态，不持久化
        self.custom_timers = []          # 自定义定时行为：[{name, seq, interval:[min,max]}]
        self.custom_timer_next = {}      # {name: next_monotonic}，运行时状态，不持久化
        self.facing_timeout_min = 10     # 朝向无变化超时（分钟）：超过就停止自动，0=禁用
        self.player_lost_timeout_min = 3 # 找不到玩家超时（分钟）：超过就停止自动，0=禁用
        self.anti_afk_enabled = False   # 防掉线开关
        self.anti_afk_min = 5           # 防掉线触发时间下限（分钟）
        self.anti_afk_max = 10          # 防掉线触发时间上限（分钟）
        self.anti_afk_seq = []          # 防掉线行为序列（down/up/delay）
        self.custom_keys = {}           # 自定义按键：{键名: 物理键名}，键名自动命名（custom1...）

    def set_key(self, name, key):
        self.keymap[name] = key

    def to_dict(self):
        """导出所有决策参数（不含运行时状态 enabled / feed_next_monotonic）。"""
        return {"attack_dist": self.attack_dist,
                "min_attack_dist": self.min_attack_dist,
                "chase_jump_dist": self.chase_jump_dist,
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
                "player_debounce_dist": self.player_debounce_dist,
                "hp_bar": self.hp_bar, "mp_bar": self.mp_bar,
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
                "route_enabled": self.route_enabled,
                "debounce_conf": self.debounce_conf,
                "debounce_ms": self.debounce_ms,
                "auto_feed_pet": self.auto_feed_pet,
                "feed_interval_min": self.feed_interval_min,
                "feed_interval_max": self.feed_interval_max,
                "custom_timers": self.custom_timers,
                "facing_timeout_min": self.facing_timeout_min,
                "player_lost_timeout_min": self.player_lost_timeout_min,
                "anti_afk_enabled": self.anti_afk_enabled,
                "anti_afk_min": self.anti_afk_min,
                "anti_afk_max": self.anti_afk_max,
                "anti_afk_seq": self.anti_afk_seq,
                "custom_keys": self.custom_keys}

    def save(self):
        """落盘。enabled 不存 —— 自动开关是运行时状态，重启后总是关闭。"""
        try:
            _SETTINGS_FILE.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception:
            pass

    def from_dict(self, data):
        """从 dict 恢复所有决策参数（参数模板加载也走这里）。"""
        data = data or {}
        self.attack_dist = float(data.get("attack_dist", 80.0))
        self.min_attack_dist = int(data.get("min_attack_dist", 0))
        self.chase_jump_dist = int(data.get("chase_jump_dist", -1))
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
        if dev in ("local", "remote"):
            self.input_device = dev
        dly = data.get("input_delay")
        if isinstance(dly, (list, tuple)) and len(dly) == 2:
            self.input_delay = [int(dly[0]), int(dly[1])]
        self.attack_cd = int(data.get("attack_cd", 0))
        self.attack_lock_debounce_ms = int(data.get("attack_lock_debounce_ms", 300))
        self.player_debounce_dist = int(data.get("player_debounce_dist", 150))
        self.hp_bar = self._load_rect(data.get("hp_bar"))
        self.mp_bar = self._load_rect(data.get("mp_bar"))
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
        self.route_enabled = bool(data.get("route_enabled", False))
        self.debounce_conf = float(data.get("debounce_conf", 0.5))
        self.debounce_ms = int(data.get("debounce_ms", 300))
        self.auto_feed_pet = bool(data.get("auto_feed_pet", False))
        self.feed_interval_min = int(data.get("feed_interval_min", 5))
        self.feed_interval_max = int(data.get("feed_interval_max", 10))
        if self.feed_interval_max < self.feed_interval_min:
            self.feed_interval_max = self.feed_interval_min
        self.custom_timers = self._load_timers(data.get("custom_timers"))
        self.facing_timeout_min = int(data.get("facing_timeout_min", 10))
        self.player_lost_timeout_min = int(data.get("player_lost_timeout_min", 3))
        self.anti_afk_enabled = bool(data.get("anti_afk_enabled", False))
        self.anti_afk_min = int(data.get("anti_afk_min", 5))
        self.anti_afk_max = int(data.get("anti_afk_max", 10))
        if self.anti_afk_max < self.anti_afk_min:
            self.anti_afk_max = self.anti_afk_min
        self.anti_afk_seq = self._load_seq(data.get("anti_afk_seq"), [])
        ck = data.get("custom_keys") or {}
        if isinstance(ck, dict):
            self.custom_keys = {str(k): (v if isinstance(v, str) or v is None else None)
                                for k, v in ck.items()}

    def load(self):
        """从 config/decision.json 读回上次的决策参数。"""
        try:
            self.from_dict(json.loads(_SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass

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
    def _load_rect(v):
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return [int(v[0]), int(v[1]), int(v[2]), int(v[3])]
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
        """读回自定义定时行为列表：[{name, seq, interval:[min,max]}]。"""
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
                lo = max(0, int(iv[0]))
                hi = max(lo, int(iv[1]))
            except Exception:
                continue
            out.append({"name": name, "seq": seq, "interval": [lo, hi]})
        return out


class CombatAgent:
    def __init__(self, settings):
        self.settings = settings
        self.keys = KeyState()
        self.state = "idle"
        self.facing = 1             # 朝向：+1 右（默认）/ -1 左，由最后按的方向键决定
        self._patrol_dir = 1        # 扫平台倾向朝向（巡逻主方向）：打背后怪不改变它
        self._last_facing_change = time.monotonic()  # 朝向最后一次变化的时刻（超时监控用）
        self._player_lost_since = None  # 找不到玩家的起始时刻（monotonic），定位到就重置
        self._was_enabled = False   # 上一次 enabled 状态（检测开启自动的上升沿）
        self._deadzone = 6.0        # |dx| 小于它就停，避免左右抖
        self._last_output = 0.0     # 最近一次输出行为（攻击）的时刻（monotonic）
        self._attack_duration = 0.03  # 每次点按攻击键的按住时长（秒）
        self._target_id = None      # 当前锁定的目标 id
        self._target_until = 0.0    # 锁定到期时间（monotonic）
        self._next_hp_pot = 0.0     # 下次补血的时刻
        self._next_mp_pot = 0.0     # 下次补蓝的时刻
        self._next_feed = 0.0       # 下次喂宠的时刻
        self._was_feed_enabled = False  # 喂宠开关上一次状态（上升沿检测，打开时不立即喂）
        self._timer_states = {}      # 自定义定时行为执行中序列状态：{name: [phase, next_ts, held]}
        self._kill_mobs = set()     # 要立即消除的防抖幽灵框 id（攻击幽灵框时记录，避免空放技能）
        self._next_evade = 0.0      # 下次规避动作（跳）的时刻
        self._next_chase_jump = 0.0 # 下次追击起跳的时刻
        self._pending_attack = None # 待发的输出键时刻（跳规避：跳键后 interval 发输出）
        self._no_target_since = None  # 朝向没目标的起始时刻（换朝向防抖用）
        self._back_ctx = None       # 回身输出序列上下文 [seq, phase, next, held, sub_stack]
        self._output_ctx = None     # 输出行为序列上下文
        self._next_afk = 0.0        # 下次防掉线触发时刻
        self._was_afk_enabled = False  # 防掉线开关上一次状态（上升沿检测）
        self._afk_ctx = None        # 防掉线序列上下文

    @staticmethod
    def _edge_dist(m, player):
        """玩家框到怪框的水平边缘距离（像素）；0 表示两框水平已接触/重叠。

        攻击是水平方向的，用框边缘而非中心点：只要怪框边缘进入玩家
        框边缘 + attack_dist 的范围就算够得着，避免大框目标被中心点
        欧氏距离误判成「太远」。
        """
        p_left = player.x - player.w / 2.0
        p_right = player.x + player.w / 2.0
        m_left = m.x - m.w / 2.0
        m_right = m.x + m.w / 2.0
        if m_right < p_left:
            return p_left - m_right   # 怪完全在玩家左侧
        if m_left > p_right:
            return m_left - p_right   # 怪完全在玩家右侧
        return 0.0                    # 水平重叠 → 边缘已接触

    def _nearest(self, mobs, player):
        """找最近的怪（框边缘水平距离），返回 (target, dist)；没怪返回 (None, None)。"""
        target, best = None, None
        for m in mobs:
            d = self._edge_dist(m, player)
            if best is None or d < best:
                best = d
                target = m
        return target, best

    def set_facing(self, f):
        """设置朝向；朝向真的变了就重置「朝向无变化」超时计时。"""
        if f != self.facing:
            self.facing = f
            self._last_facing_change = time.monotonic()

    def _steer(self, dx, keys):
        """朝目标方向走：按对应方向键并更新朝向。dx=0 不动。"""
        if dx > 0:
            keys.add(self.settings.keymap["right"])
            self.set_facing(1)
        elif dx < 0:
            keys.add(self.settings.keymap["left"])
            self.set_facing(-1)

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

    def _run_seq(self, now, ctx):
        """执行序列上下文一步，支持「触发后执行」的嵌套子序列。

        ctx = [seq, phase, next_ts, held, sub_stack]；sub_stack 是 then
        子序列的栈（list of ctx），执行时优先栈顶。返回更新后的 ctx；
        序列走完返回 None（完成时释放本层 held 键）。
        """
        seq, phase, next_ts, held, sub_stack = ctx
        # 子栈优先：先执行 then 子序列
        if sub_stack:
            sub = self._run_seq(now, sub_stack[-1])
            if sub is None:
                sub_stack.pop()
            else:
                sub_stack[-1] = sub
            return ctx
        if now < next_ts:
            return ctx
        if not seq or phase >= len(seq):
            self._release_held_set(held)
            return None
        elem = seq[phase]
        phase += 1
        # 执行几率：默认 100%；未命中则跳过本元素（下一帧执行下一个）
        prob = elem.get("prob", 100)
        if prob < 100 and random.random() * 100 >= prob:
            return [seq, phase, next_ts, held, sub_stack]
        if elem["type"] == "delay":
            next_ts = now + max(0, int(elem.get("ms", 0))) / 1000.0
        else:
            key = self._resolve_seq_key(elem.get("key"))
            if key:
                if elem["type"] == "down":
                    key_down(key)
                    held.add(key)
                    if key == self.settings.keymap.get("attack"):
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
        """朝向前方、攻击范围内的框（边缘距离 <= attack_dist），按距离升序。

        攻击只朝前方：背后的框即使水平距离近也不算在攻击范围内，走 chase 转身。
        """
        ad = self.settings.attack_dist
        px = ws.player.x
        return sorted([m for m in mobs
                       if (m.x - px) * self.facing >= 0
                       and self._edge_dist(m, ws.player) <= ad],
                      key=lambda m: self._edge_dist(m, ws.player))

    def _attack_state(self, target, best, mobs, ws):
        """攻击范围内有框时的状态选择（attack / 规避贴脸）。返回 keys。"""
        s = self.settings
        px = ws.player.x
        min_dist = s.min_attack_dist
        keys = set()

        target_too_close = (min_dist > 0 and best < min_dist)
        any_too_close = (min_dist > 0 and
                         any(self._edge_dist(m, ws.player) < min_dist for m in mobs))

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
            # 攻击时按住朝向目标的方向键：角色转向慢，确保面向目标、攻击打得到
            self._steer(target.x - px, keys)

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

    def _locked_target(self, lockable, ws, now):
        """chase 用的锁定目标：target_cd 机制，无抢锁。返回 (target, best)。"""
        target = None
        if self._target_id is not None:
            for m in lockable:
                if m.id == self._target_id:
                    target = m
                    break

        if target is not None and now < self._target_until:
            best = self._edge_dist(target, ws.player)
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

    def _output_actions(self, now, target):
        """按当前状态输出动作：attack 连点 / evade_jump 跳+输出 / evade_back_jump 序列。"""
        s = self.settings
        attack_cd = max(0.0, float(s.attack_cd)) / 1000.0
        attack_key = s.keymap["attack"]

        if self.state == "attack":
            # 执行输出行为序列（output_seq），走完等 attack_cd 再循环
            if self._output_ctx is None:
                self._output_ctx = [s.output_seq, 0, 0.0, set(), []]
            r = self._run_seq(now, self._output_ctx)
            if r is None:
                self._output_ctx = [s.output_seq, 0,
                                    now + attack_cd + self._random_input_delay(),
                                    set(), []]
            else:
                self._output_ctx = r
            # 攻击的是防抖幽灵框（漏检保留的）：立即消除，避免持续空放技能
            if target is not None and getattr(target, "missed", 0) > 0:
                self._kill_mobs.add(target.id)
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
                self._pending_attack = now + interval
                self._next_evade = now + interval + attack_cd + self._random_input_delay()
            if self._pending_attack is not None and now >= self._pending_attack:
                tap(attack_key, self._attack_duration)
                self._last_output = now
                self._pending_attack = None

        if self.state == "evade_back_jump":
            if self._back_ctx is None:
                self._back_ctx = [s.back_jump_seq, 0, 0.0, set(), []]
            r = self._run_seq(now, self._back_ctx)
            if r is None:
                # 序列走完：循环重来（回身输出是循环行为）
                self._back_ctx = [s.back_jump_seq, 0,
                                  now + attack_cd + self._random_input_delay(),
                                  set(), []]
            else:
                self._back_ctx = r

    def tick(self, ws):
        """跑一帧决策。ws: WorldState。返回动作描述 dict（调试/展示）。"""
        s = self.settings

        # 开启自动的上升沿：重置朝向监控计时。否则关掉自动后隔很久再开，
        # 会沿用旧的「最后一次朝向变化」时刻，被误判超时、把自动立刻停掉。
        if s.enabled and not self._was_enabled:
            self._last_facing_change = time.monotonic()
            self._player_lost_since = None   # 重新开启自动：重置找不到玩家计时
        self._was_enabled = s.enabled

        if not s.enabled:
            # 释放所有按着的键：不仅 KeyState，还有序列（回身输出/防掉线/定时）残留的，
            # 否则停自动时若正处于序列中间，序列按下的键会卡住继续生效。
            self._release_held_keys()
            self.state = "idle"
            return {"state": "idle", "reason": "未开启"}

        now = time.monotonic()

        # 不依赖玩家定位的定时行为：到点就执行（喂宠 / 自定义定时）
        self._feed_pet(now)
        self._custom_timers(now)

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
                    return {"state": "idle", "reason": "长时间未定位到玩家，已停止自动"}
            # 没定位到玩家：释放打怪相关按键，但保留定时行为序列（它们不依赖玩家）
            self._release_combat_keys()
            self.state = "idle"
            return {"state": "idle", "reason": "未定位玩家"}

        # 定位到玩家：重置找不到玩家计时
        self._player_lost_since = None

        # 朝向监控：朝向变化在 set_facing 里刷新计时；超过 facing_timeout_min
        # 分钟仍没变，认为卡死/异常，停止自动。
        timeout = max(0.0, float(s.facing_timeout_min)) * 60.0
        if timeout > 0 and now - self._last_facing_change >= timeout:
            s.enabled = False
            self.keys.release_all()
            self.state = "idle"
            return {"state": "idle", "reason": "朝向长时间未变化，已停止自动"}

        # 防掉线：开关上升沿随机一个下次触发时间；到点进入防掉线状态执行行为序列。
        if s.anti_afk_enabled:
            if not self._was_afk_enabled:
                lo = max(0.0, float(s.anti_afk_min))
                hi = max(lo, float(s.anti_afk_max))
                self._next_afk = now + random.uniform(lo, hi) * 60.0
            if self.state != "afk" and now >= self._next_afk:
                self.state = "afk"
                self._afk_ctx = None
            if self.state == "afk":
                self._run_afk(now)
                return {"state": self.state, "target": None, "dx": 0, "dist": 0,
                        "keys": [], "facing": self.facing}
        elif self.state == "afk":
            # 关掉防掉线：退出状态，松开残留键
            self._release_ctx(self._afk_ctx)
            self._afk_ctx = None
            self.state = "idle"
        self._was_afk_enabled = s.anti_afk_enabled

        px = ws.player.x

        # 按视野矩形过滤怪物（视野外的怪不参与决策）
        mobs = self._filter_mobs(ws.mobs, ws.player, ws)
        # 地形关系（怪是否与玩家当前平台连接）已由感知层算好，这里只消费结果
        mobs = [m for m in mobs if m.reachable]

        # 自动喝药：依赖玩家定位（读血/蓝），定位到玩家后才执行
        self._drink_potions(ws, now)

        # 锁定候选（sweep 只背后 back_range 内的怪；patrol 全部）
        candidates = self._candidates(mobs, px)

        # attack 最高优先级：只要前方攻击范围内有框，就进入攻击（或规避）
        in_range = self._in_range(mobs, ws)

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
            best = self._edge_dist(target, ws.player)
            self._target_id = target.id
            self._target_until = now + self._random_target_cd()
            self._no_target_since = None
            keys = self._attack_state(target, best, mobs, ws)
        else:
            # 前方攻击范围内没框
            target, best = self._locked_target(candidates, ws, now)
            keys = set()
            if target is not None:
                # 有锁定目标（sweep 背后怪 / patrol 最近怪）：朝它走
                self._steer(target.x - px, keys)
                # 追击起跳：锁定目标在 [attack_dist, attack_dist + chase_jump_dist] 就跳
                cjd = max(0.0, float(s.chase_jump_dist))
                if (cjd > 0 and s.attack_dist <= best <= s.attack_dist + cjd
                        and now >= self._next_chase_jump):
                    jump_key = s.keymap.get("jump")
                    if jump_key:
                        tap(jump_key, self._attack_duration)
                        attack_cd_s = max(0.0, float(s.attack_cd)) / 1000.0
                        self._next_chase_jump = now + max(0.3, attack_cd_s + self._random_input_delay())
                self._set_state("chase")
            elif s.strategy == "sweep":
                # 扫平台巡逻：无背后怪，朝倾向朝向走（不锁定、无红框）。
                # 物理朝向先拉回倾向朝向（可能刚转身打过背后怪），再朝它走。
                best = 0.0
                self.set_facing(self._patrol_dir)
                keys.add(s.keymap["right"] if self._patrol_dir > 0 else s.keymap["left"])
                self._set_state("chase")
                # 追击起跳：向倾向方向移动时，追击起跳范围内有怪（即使未锁定）也按跳
                cjd = max(0.0, float(s.chase_jump_dist))
                if cjd > 0 and now >= self._next_chase_jump:
                    jump_key = s.keymap.get("jump")
                    if jump_key:
                        hit = next((m for m in mobs
                                    if (m.x - px) * self._patrol_dir >= 0
                                    and s.attack_dist <= self._edge_dist(m, ws.player) <= s.attack_dist + cjd),
                                   None)
                        if hit is not None:
                            tap(jump_key, self._attack_duration)
                            attack_cd_s = max(0.0, float(s.attack_cd)) / 1000.0
                            self._next_chase_jump = now + max(0.3, attack_cd_s + self._random_input_delay())
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
                return {"state": "idle", "reason": "无怪"}

        self.keys.set(keys)

        # 输出行为：攻击 / 跳规避 / 回身输出（按 self.state）
        self._output_actions(now, target)

        kill_mob = self._kill_mobs.pop() if self._kill_mobs else None
        tid = target.id if target is not None else None
        dx = round((target.x - px) if target is not None else 0.0, 1)
        return {"state": self.state, "target": tid, "dx": dx,
                "dist": round(best, 1),
                "keys": sorted(keys), "facing": self.facing,
                "kill_mob": kill_mob}

    def _run_afk(self, now):
        """防掉线状态：执行行为序列。走完退出，并随机下一次触发时间。"""
        s = self.settings
        if self._afk_ctx is None:
            self._afk_ctx = [s.anti_afk_seq, 0, 0.0, set(), []]
        r = self._run_seq(now, self._afk_ctx)
        if r is None:
            self._afk_ctx = None
            self.state = "idle"
            lo = max(0.0, float(s.anti_afk_min))
            hi = max(lo, float(s.anti_afk_max))
            self._next_afk = now + random.uniform(lo, hi) * 60.0
        else:
            self._afk_ctx = r

    def _feed_pet(self, now):
        """自动喂宠：勾选后每隔 feed_cd 按一次喂宠键。

        开关边沿：关掉清计时；打开时不立即喂，先等一个完整间隔。
        """
        s = self.settings
        if not s.auto_feed_pet:
            s.feed_next_monotonic = 0.0
            self._next_feed = 0.0     # 关掉清计时
            self._was_feed_enabled = False
            return
        feed_key = s.keymap.get("feed_pet")
        if not feed_key:
            s.feed_next_monotonic = 0.0
            self._next_feed = 0.0     # 没键也清，设好键后立即吃
            self._was_feed_enabled = False
            return
        # 上升沿：刚打开，重置为「等一个间隔后再喂」，不立即吃
        if not self._was_feed_enabled:
            lo = max(0.0, float(s.feed_interval_min))
            hi = max(lo, float(s.feed_interval_max))
            self._next_feed = now + random.uniform(lo, hi) * 60.0
        self._was_feed_enabled = True
        if now >= self._next_feed:
            tap(feed_key, self._attack_duration)
            lo = max(0.0, float(s.feed_interval_min))
            hi = max(lo, float(s.feed_interval_max))
            self._next_feed = now + random.uniform(lo, hi) * 60.0
        s.feed_next_monotonic = self._next_feed

    def _custom_timers(self, now):
        """自定义定时行为：每个行为每隔随机 [min,max] 分钟执行一次其序列。

        行为序列分帧执行（复用 _run_seq），执行期间不影响其他定时行为。
        """
        s = self.settings
        for t in s.custom_timers:
            name = t.get("name") or ""
            if not name:
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
        """自动喝药：血/蓝低于阈值就点按对应键，喝药冷却 pot_cd 内不再喝。"""
        s = self.settings
        hp_pot = s.keymap.get("hp_pot")
        mp_pot = s.keymap.get("mp_pot")
        cd = max(0, float(s.pot_cd)) / 1000.0   # 喝药冷却（秒）

        if s.auto_hp_pot and hp_pot and ws.player.hp < s.hp_threshold / 100.0:
            if now >= self._next_hp_pot:
                tap(hp_pot, self._attack_duration)
                self._next_hp_pot = now + cd
        else:
            self._next_hp_pot = 0.0     # 血够了/没开自动补血，重置（下次低于阈值立刻补）

        if s.auto_mp_pot and mp_pot and ws.player.mp < s.mp_threshold / 100.0:
            if now >= self._next_mp_pot:
                tap(mp_pot, self._attack_duration)
                self._next_mp_pot = now + cd
        else:
            self._next_mp_pot = 0.0

    def _release_combat_keys(self):
        """释放打怪相关按键：KeyState + 回身输出/输出/防掉线序列。
        定时行为（custom_timers）的序列不在这里释放——它们不依赖玩家定位。"""
        self.keys.release_all()
        self._release_ctx(self._back_ctx)
        self._release_ctx(self._output_ctx)
        self._release_ctx(self._afk_ctx)

    def _release_held_keys(self):
        """释放所有还按着的键：KeyState 的 + 序列（回身输出/防掉线/定时行为）残留的。"""
        self._release_combat_keys()
        for st in self._timer_states.values():
            self._release_ctx(st)
        self._timer_states.clear()

    def shutdown(self):
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


# 全局决策设置（单例）：GUI 主线程写，实时推理线程读。
# 用单例让「决策参数」页签和「实时」页共享同一份配置，不用层层传引用。
settings = DecisionSettings()
settings.load()     # 启动时读回上次保存的决策参数
