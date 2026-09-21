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


class DecisionSettings:
    """决策参数（UI 写，决策线程读），改动后持久化到 config/decision.json。"""

    def __init__(self):
        self.enabled = False
        self.attack_dist = 80.0     # 最大攻击距离（画面像素），小于它就打
        self.min_attack_dist = 0    # 最小攻击距离（像素），>0 时启用规避
        self.evade_type = "jump"    # 规避类型："jump" 跳 / "back" 后退
        self.jump_interval = 200    # 跳间隔（毫秒）：跳键和输出键之间的间隔
        self.jump_random_prob = 0.1 # 乱跳几率（0~1）：正常攻击时随机跳+输出的概率
        self.back_jump_seq = [dict(e) for e in DEFAULT_BACK_JUMP_SEQ]  # 回身输出行为序列
        self.keymap = dict(DEFAULT_KEYMAP)
        self.target_cd = [500, 1000]  # 目标切换 CD [min, max]（毫秒）
        self.input_device = "local"   # 输入设备："local" 本机 / "remote" Pro Micro
        self.input_delay = [70, 130]  # 随机输入延迟 [min, max]（毫秒），按键之间
        self.attack_cd = 0          # 输出行为 CD（毫秒）：攻击/跳输出/反向跳回身等行为的节奏 = CD + 随机延迟
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
        self.sweep_turn_cd = 1000   # 扫平台：换朝向 CD（毫秒）
        self.turn_debounce_ms = 0   # 扫平台：换朝向防抖（毫秒），朝向没目标持续这么久才换
        self.back_range = 100       # 扫平台：允许锁定背后多远距离内的怪（像素）
        self.debounce_conf = 0.5    # 防抖置信度：高于它的怪框消失后保留位置
        self.debounce_ms = 300      # 防抖时间（毫秒）：保留消失前位置的时长
        self.auto_feed_pet = False  # 自动喂宠
        self.feed_interval_min = 5  # 喂宠间隔下限（分钟）
        self.feed_interval_max = 10 # 喂宠间隔上限（分钟）
        self.feed_next_monotonic = 0.0   # 下次喂宠时刻（time.monotonic），UI 倒计时读；运行时状态，不持久化
        self.facing_timeout_min = 10     # 朝向无变化超时（分钟）：超过就停止自动，0=禁用
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
                "evade_type": self.evade_type,
                "jump_interval": self.jump_interval,
                "jump_random_prob": self.jump_random_prob,
                "back_jump_seq": self.back_jump_seq,
                "keymap": self.keymap,
                "target_cd": self.target_cd,
                "input_device": self.input_device,
                "input_delay": self.input_delay,
                "attack_cd": self.attack_cd,
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
                "turn_debounce_ms": self.turn_debounce_ms,
                "back_range": self.back_range,
                "debounce_conf": self.debounce_conf,
                "debounce_ms": self.debounce_ms,
                "auto_feed_pet": self.auto_feed_pet,
                "feed_interval_min": self.feed_interval_min,
                "feed_interval_max": self.feed_interval_max,
                "facing_timeout_min": self.facing_timeout_min,
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
        self.evade_type = data.get("evade_type", "jump")
        self.jump_interval = int(data.get("jump_interval", 200))
        self.jump_random_prob = float(data.get("jump_random_prob", 0.1))
        self.back_jump_seq = self._load_seq(data.get("back_jump_seq"))
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
        self.turn_debounce_ms = int(data.get("turn_debounce_ms", 0))
        self.back_range = int(data.get("back_range", 100))
        self.debounce_conf = float(data.get("debounce_conf", 0.5))
        self.debounce_ms = int(data.get("debounce_ms", 300))
        self.auto_feed_pet = bool(data.get("auto_feed_pet", False))
        self.feed_interval_min = int(data.get("feed_interval_min", 5))
        self.feed_interval_max = int(data.get("feed_interval_max", 10))
        if self.feed_interval_max < self.feed_interval_min:
            self.feed_interval_max = self.feed_interval_min
        self.facing_timeout_min = int(data.get("facing_timeout_min", 10))
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
        """读回行为序列（list of dict），非法则返回 default（默认回身输出序列）。"""
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
                out.append({"type": "delay", "ms": int(e.get("ms", 0))})
            elif t in ("down", "up"):
                key = e.get("key")
                if isinstance(key, str) and key:
                    out.append({"type": t, "key": key})
        return out or [dict(e) for e in default]


class CombatAgent:
    def __init__(self, settings):
        self.settings = settings
        self.keys = KeyState()
        self.state = "idle"
        self.facing = 1             # 朝向：+1 右（默认）/ -1 左，由最后按的方向键决定
        self._last_facing_change = time.monotonic()  # 朝向最后一次变化的时刻（超时监控用）
        self._was_enabled = False   # 上一次 enabled 状态（检测开启自动的上升沿）
        self._deadzone = 6.0        # |dx| 小于它就停，避免左右抖
        self._next_attack = 0.0     # 下次攻击的时刻（time.monotonic）
        self._attack_duration = 0.03  # 每次点按攻击键的按住时长（秒）
        self._target_id = None      # 当前锁定的目标 id
        self._target_until = 0.0    # 锁定到期时间（monotonic）
        self._next_hp_pot = 0.0     # 下次补血的时刻
        self._next_mp_pot = 0.0     # 下次补蓝的时刻
        self._next_feed = 0.0       # 下次喂宠的时刻
        self._next_turn = 0.0       # 下次换朝向的时刻（扫平台用）
        self._next_evade = 0.0      # 下次规避动作（跳）的时刻
        self._pending_attack = None # 待发的输出键时刻（跳规避：跳键后 interval 发输出）
        self._no_target_since = None  # 朝向没目标的起始时刻（换朝向防抖用）
        self._back_jump_phase = 0   # 反向跳回身输出序列的阶段 0~4
        self._back_jump_next = 0.0  # 下一阶段时刻
        self._back_jump_held = set()  # 序列中按下的键（离开状态时松开）
        self._next_afk = 0.0        # 下次防掉线触发时刻
        self._was_afk_enabled = False  # 防掉线开关上一次状态（上升沿检测）
        self._afk_phase = 0         # 防掉线行为序列阶段
        self._afk_next = 0.0        # 防掉线序列下一阶段时刻
        self._afk_held = set()      # 防掉线序列按下的键

    @staticmethod
    def _dist(m, px, py):
        return ((m.x - px) ** 2 + (m.y - py) ** 2) ** 0.5

    def _nearest(self, mobs, px, py):
        """找最近的怪（欧氏距离），返回 (target, dist)；没怪返回 (None, None)。"""
        target, best = None, None
        for m in mobs:
            d = self._dist(m, px, py)
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

    def _seq_step(self, now, seq, phase, next_ts, held):
        """执行行为序列的一步。held（set）原地修改；返回 (新phase, 新next_ts, done)。

        done=True 表示序列已走完（空序列或 phase 越界）。
        """
        if now < next_ts:
            return phase, next_ts, False
        if not seq or phase >= len(seq):
            return phase, next_ts, True
        elem = seq[phase]
        phase += 1
        if elem["type"] == "delay":
            next_ts = now + max(0, int(elem.get("ms", 0))) / 1000.0
        else:
            key = self._resolve_seq_key(elem.get("key"))
            if key:
                if elem["type"] == "down":
                    key_down(key)
                    held.add(key)
                else:
                    key_up(key)
                    held.discard(key)
            next_ts = now + self._random_input_delay()
        return phase, next_ts, False

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

    def tick(self, ws):
        """跑一帧决策。ws: WorldState。返回动作描述 dict（调试/展示）。"""
        s = self.settings

        # 开启自动的上升沿：重置朝向监控计时。否则关掉自动后隔很久再开，
        # 会沿用旧的「最后一次朝向变化」时刻，被误判超时、把自动立刻停掉。
        if s.enabled and not self._was_enabled:
            self._last_facing_change = time.monotonic()
        self._was_enabled = s.enabled

        if not s.enabled or not ws.player.found:
            # 释放所有按着的键：不仅 KeyState，还有序列（回身输出/防掉线）残留的，
            # 否则停自动时若正处于序列中间，序列按下的键会卡住继续生效。
            self._release_held_keys()
            self.state = "idle"
            return {"state": "idle", "reason": "未开启" if not s.enabled else "未定位玩家"}

        now = time.monotonic()

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
                self._afk_phase = 0
                self._afk_next = 0.0
            if self.state == "afk":
                self._run_afk(now)
                return {"state": self.state, "target": None, "dx": 0, "dist": 0,
                        "keys": [], "facing": self.facing}
        elif self.state == "afk":
            # 关掉防掉线：退出状态，松开残留键
            for k in self._afk_held:
                try:
                    key_up(k)
                except Exception:
                    pass
            self._afk_held = set()
            self.state = "idle"
        self._was_afk_enabled = s.anti_afk_enabled

        px, py = ws.player.x, ws.player.y
        attack_dist = s.attack_dist
        min_dist = s.min_attack_dist

        # 按视野矩形过滤怪物（视野外的怪不参与决策）
        mobs = self._filter_mobs(ws.mobs, ws.player, ws)

        # 自动喝药 / 自动喂宠：独立于打怪，即使没怪也执行
        self._drink_potions(ws, now)
        self._feed_pet(now)

        # 目标锁定 + 切换 CD：
        #   · 锁定目标还在、且未到期 → 保持锁定（即使别的怪更近也不切）
        #   · 锁定目标消失 → CD 清零，立即重新找最近
        #   · CD 到期 → 重新找最近（可能切换），随机新 CD
        target = None

        # 候选怪：扫平台保留朝向倾向 —— 朝向方向全部 + 背后 back_range 内的怪
        # （按就近优先）；平地巡逻锁全部
        if s.strategy == "sweep":
            back = max(0.0, float(s.back_range))
            candidates = [m for m in mobs if (m.x - px) * self.facing >= -back]
        else:
            candidates = mobs

        if self._target_id is not None:
            for m in candidates:
                if m.id == self._target_id:
                    target = m
                    break

        if target is not None and now < self._target_until:
            best = self._dist(target, px, py)      # CD 内保持锁定
            self._no_target_since = None
        else:
            target, best = self._nearest(candidates, px, py)
            if target is not None:
                self._target_id = target.id
                self._target_until = now + self._random_target_cd()
                self._no_target_since = None
            else:
                self._target_id = None
                self._target_until = 0.0
                # 扫平台：朝向方向没怪 → 换朝向。
                # 换朝向防抖：朝向没目标持续 turn_debounce_ms 才换；
                # 换朝向 CD：换向后至少 sweep_turn_cd 才能再换。
                if s.strategy == "sweep":
                    if self._no_target_since is None:
                        self._no_target_since = now
                    debounce = max(0.0, float(s.turn_debounce_ms)) / 1000.0
                    if (now - self._no_target_since >= debounce
                            and now >= self._next_turn):
                        self.set_facing(-self.facing)
                        self._next_turn = now + max(0.0, float(s.sweep_turn_cd)) / 1000.0
                        self._no_target_since = None   # 换向后重新计时

        # chase 抢锁：正在追击时，若最小攻击距离~最大攻击距离范围内出现了怪，
        # 立即改锁就近的那个（不等 target_cd），更快进入攻击。
        if self.state == "chase":
            near, near_d = None, None
            for m in candidates:
                d = self._dist(m, px, py)
                if min_dist <= d <= attack_dist:
                    if near_d is None or d < near_d:
                        near, near_d = m, d
            if near is not None:
                target = near
                best = near_d
                self._target_id = near.id
                self._target_until = now + self._random_target_cd()

        if target is None:
            self.keys.release_all()
            self.state = "idle"
            return {"state": "idle", "reason": "无怪"}

        dx = target.x - px
        keys = set()

        # 怪在朝向侧（前面）才算「能打到」：dx 和 facing 同号（dx=0 也算前面）。
        # 怪在背后时不攻击，而是转身去追 —— 否则会在背后狂按输出。
        in_front = dx * self.facing >= 0

        # 攻击距离范围：[最小攻击距离, 最大攻击距离]。
        #   · 在范围内 → 攻击
        #   · 小于最小攻击距离（太近，有怪贴脸）→ 规避（跳 / 后退）

        if best <= attack_dist and in_front:
            target_too_close = (min_dist > 0 and best < min_dist)
            any_too_close = (min_dist > 0 and
                             any(self._dist(m, px, py) < min_dist for m in mobs))

            new_state = None
            if s.evade_type == "jump":
                if target_too_close:
                    # 锁定目标贴脸：反向跳回身输出
                    new_state = "evade_back_jump"
                elif any_too_close:
                    # 其他怪贴脸：跳 + 输出
                    new_state = "evade_jump"
                else:
                    # 没怪贴脸：乱跳几率跳+输出，否则正常输出。
                    # 最小攻击距离=0（未启用规避）时，乱跳也不生效。
                    new_state = ("evade_jump"
                                 if min_dist > 0 and random.random() < s.jump_random_prob
                                 else "attack")
            else:  # back 后退
                if any_too_close:
                    # 太近：后退（反向走），facing 保持面向怪
                    new_state = "evade"
                    keys.add(self._resolve_seq_key("back"))
                else:
                    new_state = "attack"

            if new_state == "attack":
                # 攻击时也按着朝向目标的方向键：角色转向慢，facing 变了但角色
                # 可能还没转过来 —— 按住方向键确保真正面向目标、攻击打得到。
                self._steer(dx, keys)

            # 进入反向跳回身序列时重置阶段；离开时松开序列残留的键
            if new_state == "evade_back_jump" and self.state != "evade_back_jump":
                self._back_jump_phase = 0
                self._back_jump_next = 0.0
            elif new_state != "evade_back_jump" and self.state == "evade_back_jump":
                for k in self._back_jump_held:
                    try:
                        key_up(k)
                    except Exception:
                        pass
                self._back_jump_held = set()

            self.state = new_state
        elif in_front and abs(dx) <= self._deadzone:
            # 怪在前面但水平很近、又不在攻击距离内（多半在头顶/脚下平台）：
            # 水平移动没用，停住别左右晃。
            self.state = "chase"
        elif dx != 0:
            # 怪在前方追击，或背后转身：朝目标方向走
            self.state = "chase"
            self._steer(dx, keys)
        else:
            # dx == 0（怪在正头顶/脚下），不动
            self.state = "chase"

        self.keys.set(keys)

        # 输出行为（攻击）：连按输出键，节奏 = 输出行为 CD + 随机输入延迟。
        attack_cd = max(0.0, float(s.attack_cd)) / 1000.0
        attack_key = s.keymap["attack"]
        if self.state == "attack":
            if now >= self._next_attack:
                tap(attack_key, self._attack_duration)
                self._next_attack = now + attack_cd + self._random_input_delay()
        else:
            self._next_attack = 0.0     # 离开攻击状态，下次进入立刻打

        # 跳规避（跳 + 输出）：跳键 + 输出键，两键间隔 jump_interval。
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
                self._pending_attack = None

        # 回身输出：执行配置的行为序列（down/up/delay 元素）。
        if self.state == "evade_back_jump":
            self._back_jump_phase, self._back_jump_next, done = self._seq_step(
                now, s.back_jump_seq, self._back_jump_phase,
                self._back_jump_next, self._back_jump_held)
            if done:
                # 序列走完：循环重来（回身输出是循环行为）
                self._back_jump_phase = 0
                self._back_jump_next = now + attack_cd + self._random_input_delay()

        return {"state": self.state, "target": target.id,
                "dx": round(dx, 1), "dist": round(best, 1),
                "keys": sorted(keys), "facing": self.facing}

    def _run_afk(self, now):
        """防掉线状态：执行行为序列。走完退出，并随机下一次触发时间。"""
        s = self.settings
        self._afk_phase, self._afk_next, done = self._seq_step(
            now, s.anti_afk_seq, self._afk_phase, self._afk_next, self._afk_held)
        if done:
            for k in self._afk_held:
                try:
                    key_up(k)
                except Exception:
                    pass
            self._afk_held = set()
            self.state = "idle"
            lo = max(0.0, float(s.anti_afk_min))
            hi = max(lo, float(s.anti_afk_max))
            self._next_afk = now + random.uniform(lo, hi) * 60.0

    def _feed_pet(self, now):
        """自动喂宠：勾选后每隔 feed_cd 按一次喂宠键。

        开关边沿：关掉清计时（_next_feed 归零），下次打开立即吃一次。
        """
        s = self.settings
        if not s.auto_feed_pet:
            s.feed_next_monotonic = 0.0
            self._next_feed = 0.0     # 关掉清计时
            return
        feed_key = s.keymap.get("feed_pet")
        if not feed_key:
            s.feed_next_monotonic = 0.0
            self._next_feed = 0.0     # 没键也清，设好键后立即吃
            return
        if now >= self._next_feed:
            tap(feed_key, self._attack_duration)
            lo = max(0.0, float(s.feed_interval_min))
            hi = max(lo, float(s.feed_interval_max))
            self._next_feed = now + random.uniform(lo, hi) * 60.0
        s.feed_next_monotonic = self._next_feed

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

    def _release_held_keys(self):
        """释放所有还按着的键：KeyState 的 + 序列（回身输出/防掉线）残留的。"""
        self.keys.release_all()
        for held in (self._back_jump_held, self._afk_held):
            for k in held:
                try:
                    key_up(k)
                except Exception:
                    pass
            held.clear()

    def shutdown(self):
        self._release_held_keys()
        # 兜底：无条件释放所有映射键。KeyState 的 release_all 只释放「它记得
        # 按下的键」，若某次 RELEASE 命令在网络里丢了、固件却仍按着，KeyState
        # 会误以为已释放、之后不再补发 —— 所以这里对所有映射键再补一次 RELEASE，
        # 对未按下的键发 RELEASE 无害（固件/SendInput 都忽略）。
        for key in self.settings.keymap.values():
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
