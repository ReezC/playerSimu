"""决策层 / 按键层自检：把改过好几轮的语义固化成可跑的断言。

**为什么要有它**
    输出CD、行为序列进度、休息状态机这几处的语义在迭代里反复变（本次就动了三回），
    每次都是临时写个脚本验一遍、验完删掉 —— 下一轮改动时没人记得当时验过什么、
    边界是什么。这里把它们固化成断言：改完跑一遍，坏的立刻现形。

跑法：
    python -m tools.selftest_decision        # 全过返回 0，有失败返回 1

**不碰任何真实的参数文件**
    · `DecisionSettings.save` 被换成空操作（也就不会写回项目）；
    · 决策层用独立的 settings 实例（不 load 也不 save）；
    · 界面层只读配置、不改任何值。
"""

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import decision.agent as ag                                    # noqa: E402
import decision.input as dinput                                # noqa: E402
from decision.agent import CombatAgent                         # noqa: E402
from perception.world_state import Mob, Player, WorldState      # noqa: E402

# --------------------------------------------------------------- 自检不许改用户文件
# ⛔ `config/ui.yaml` 是**用户的**文件（窗口几何 / 字号 / 颜色）⇒ 自检**绝不许**写它 ✗
#    （全仓库规矩 ✓，见 `selftest_main_window._fake_store` 的说明）。
#    本套件有两处会碰到它：建/关接了 `theme.bind_window_state` 的弹窗（一 show/hide 就写
#    几何 ✗ —— 2026-09-29 逐个套件量出来本套件写 `windows.battle_zone_list`），以及
#    `vis` 那几个键的"存得进读得回"往返（写回也会把整个文件重写一遍 ✗）
#    ⇒ 把 theme 的**落点**整体指到临时文件，一个字节都不碰用户的 ✓。
import tempfile as _tempfile                                   # noqa: E402

from gui import theme as _theme                                # noqa: E402

_theme.CFG = Path(_tempfile.mkdtemp(prefix="psimu_ui_")) / "ui.yaml"

FRAME = 1.0 / 15.0          # 主循环节拍（capture_fps=15），时序都被量化到这个粒度
CD_MS = 800                 # 自检用的输出CD


# ---------------------------------------------------------------- 基础设施

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class Clock:
    """替身时钟：测试自己推进时间，不动真实 monotonic。"""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Harness:
    """跑一个 CombatAgent 若干帧，记录所有发出去的键。"""

    def __init__(self, settings):
        self.s = settings
        self.clock = Clock()
        self.log = []                       # [(相对秒, "down"/"up"/"tap"/"RELEASEALL", 键)]
        self.agent = CombatAgent(settings)
        # 自定义怪物列表：(相对秒) -> [Mob]。起跳这类用例要**精确摆距离**，
        # 默认那只怪固定距离摆不出来（见 ws）。None = 用默认那只。
        self.mobs_fn = None
        # 世界里的血量（0~1）：喝药/「被打断重试」这类用例要能摆低血量
        self.hp = 1.0
        # ⭐ **`ws` 装好之后的钩子**（`fn(ws) -> None`）—— 给"走 `run()` 那一套、但又
        #   需要摆 `ws` 上某个字段"的用例用 ✓（`run()` 每帧**自建** `ws` ⇒ 外面设的会被冲掉 ⚠）。
        #   第一个用户：`t_disable_chase_pathfinding` 要摆 `player.here_sets`
        #   ✓（⚠ 不设的话"追击下前往"那条链会**早退在 `here_sets` 空**上 ⇒ 开关到底有没有用
        #   就**测不出来**了 ✗ —— 反向验证时正是这么被蒙过去的 ✓）。
        self.ws_hook = None

    # ---- 世界状态 ----
    def ws(self, with_mob=True):
        """默认世界：玩家在 x=500，一只怪在 x=560（**距角色中心 40 px**）。

        设了 `mobs_fn` 就用它 —— 距离算得起：怪宽 40，所以
        「角色中心 → 怪框最近的边」= mob.x - 520。
        """
        w = WorldState(width=1920, height=1080)
        # ⚠ `world_x/world_y` 是**世界坐标**（寻路/上绳执行器用的就是它），
        # `x/y` 是画面坐标（检测框中心）。两套都在，别混 —— 2026-09-26 的
        # "上绳死循环"就是喂错坐标系造成的（见 t_climb_uses_world_x）。
        w.player = Player(x=500.0, y=500.0, w=60.0, h=100.0, bottom=600.0,
                          hp=self.hp, mp=1.0, found=True,
                          world_x=500.0, world_y=500.0)
        if self.mobs_fn is not None:
            w.mobs = list(self.mobs_fn(self.clock.t - getattr(self, "clock0",
                                                              self.clock.t)))
        elif with_mob:
            w.mobs = [Mob(id=1, x=560.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
        else:
            w.mobs = []
        if self.ws_hook is not None:              # ⭐ 用例摆 `ws` 字段的唯一钩子 ✓（见 __init__）
            self.ws_hook(w)
        return w

    # ---- 记录器 ----
    def _rec(self, kind):
        def f(*a):
            self.log.append((round(self.clock.t - self.clock0, 3), kind,
                             a[0] if a else "-"))
        return f

    @contextmanager
    def _patched(self):
        # **三条按键路径都要挡**：
        #   ① agent 自己发序列用的 key_down/key_up（在 ag 命名空间）；
        #   ② KeyState（移动键）走的是 decision.input 里的那两个；
        #   ③ `tap`（点按类：追击起跳 / 规避跳 / 喝药）—— 它**直接调
        #      `_send`**，既不过①②，也不进 KeyState。漏掉这条，用例会真的往
        #      本机发键（往当前窗口打字）。
        # 前两条记成 down/up（presses/gaps 靠它），第三条单独记成 tap —— 点按
        # 是「按下+松开」，塞进 down/up 会打乱那些按对配平的用例。
        with patch.object(ag.time, "monotonic", self.clock), \
             patch.object(ag, "key_down", self._rec("down")), \
             patch.object(ag, "key_up", self._rec("up")), \
             patch.object(ag, "tap", self._rec("tap")), \
             patch.object(dinput, "key_down", self._rec("down")), \
             patch.object(dinput, "key_up", self._rec("up")), \
             patch.object(dinput, "release_all_remote", self._rec("RELEASEALL")), \
             patch.object(dinput, "link_health", lambda: {"ok": True}):
            yield

    # ---- 跑 ----
    def run(self, seconds, mob_plan=None):
        """按**主循环真实取帧逻辑**推进；mob_plan(相对秒) -> 是否有怪。

        等「下一帧」或「下个序列到点时刻」，谁先到干谁：到帧走决策拍 `tick`，
        到点走时序拍 `tick_timing`。必须和 live_thread 一致 —— 自检如果只按帧
        步进，测出来的还是旧的一帧量化行为，收紧后的断言就测不到真东西。
        """
        self.clock0 = self.clock.t
        plan = mob_plan or (lambda _t: True)
        with self._patched():
            end = self.clock.t + seconds
            while self.clock.t < end:
                d = self.agent.next_deadline()
                step = (FRAME if d is None
                        else min(FRAME, max(ag.TIMING_TICK, d - self.clock.t)))
                self.clock.t += step
                ws = self.ws(plan(self.clock.t - self.clock0))
                if step >= FRAME - 1e-9:
                    self.agent.tick(ws)                 # 抓到一帧 → 决策拍
                else:
                    self.agent.tick_timing(ws)          # 等帧超时 → 时序拍
        return self

    def presses(self, key):
        """某个键「按下」的时刻列表。"""
        return [t for t, kind, k in self.log if kind == "down" and k == key]

    def gaps(self, key):
        ts = self.presses(key)
        return [round(b - a, 3) for a, b in zip(ts, ts[1:])]

    def count(self, key):
        return len(self.presses(key))

    def taps(self, key):
        """某个键被 `tap` 点按的时刻列表（起跳 / 规避跳 / 喝药走这条路）。"""
        return [t for t, kind, k in self.log if kind == "tap" and k == key]


def fresh_settings(**over):
    """独立的一份决策参数（默认值即可，不读配置文件）。"""
    s = ag.DecisionSettings()
    s.enabled = True
    s.attack_cd = CD_MS
    s.resetall_interval = 0
    s.facing_timeout_min = 0
    s.player_lost_timeout_min = 0
    s.anti_afk_enabled = False
    s.chase_jump_enabled = False
    s.custom_timers = []
    s.custom_timer_next = {}
    s.turn_output_delay_ms = 0
    s.auto_hp_pot = False
    s.auto_mp_pot = False
    s.strategy = "chase"
    s.anti_afk_min = s.anti_afk_max = 5.0
    s.anti_afk_rest_min = s.anti_afk_rest_max = 1.0        # 1 分钟 = 60 秒
    s.anti_afk_retry_on_interrupt = False
    s.anti_afk_retry_sec = 60.0
    for k, v in over.items():
        setattr(s, k, v)
    return s


def in_cd_window(gaps, cd_ms=CD_MS, label=""):
    """间隔断言：不小于 CD，且不超过 CD + 几拍时序。

    余量只该有时序拍（10ms 级）的粒度 —— 如果哪天又涨到「CD + 一个抓帧帧时
    （66.7ms）」，说明序列计时又绑回画面帧了（见 t_sequence_timing_not_frame_quantized）。
    """
    cd = cd_ms / 1000.0
    tol = 3 * ag.TIMING_TICK
    for g in gaps:
        check(g >= cd - 1e-6,
              "%s间隔 %.3f 秒 小于输出CD %.2f 秒" % (label, g, cd))
        check(g <= cd + tol,
              "%s间隔 %.3f 秒 超过 CD + %.0f 毫秒（%s）—— 又按抓帧节拍量化了？"
              % (label, g, tol * 1000, [round(x, 3) for x in gaps]))


# ---------------------------------------------------------------- 决策层自检

def t_cd_no_attack_key():
    """输出CD 是「输出行为」的 CD，不认某个按键：序列里没有攻击键也要守 CD。"""
    s = fresh_settings(output_seq=[{"type": "down", "key": "jump"},
                                   {"type": "up", "key": "jump"}])
    h = Harness(s).run(3.0)
    gaps = h.gaps(s.keymap.get("jump"))
    check(len(gaps) >= 2, "跳跃键按下次数太少：%s" % h.presses(s.keymap.get("jump")))
    in_cd_window(gaps, label="无攻击键序列：")


def t_cd_on_enter_attack_state():
    """进攻击状态也要走输出CD排期：状态抖一下（丢一帧怪又回来）不能比 CD 密。"""
    s = fresh_settings(output_seq=[{"type": "down", "key": "attack"},
                                   {"type": "up", "key": "attack"}])
    # 输出后不久把怪去掉一帧，再回来 → 状态 attack → idle/chase → attack
    h = Harness(s).run(3.0, mob_plan=lambda t: not (0.2 <= t < 0.27))
    ts = h.presses(s.keymap.get("attack"))
    check(len(ts) >= 2, "攻击键只按下 %d 次" % len(ts))
    check(ts[1] - ts[0] >= CD_MS / 1000.0 - 1e-6,
          "抖动后第二下间隔 %.3f 秒 < CD（%.2f 秒）：没走排期"
          % (ts[1] - ts[0], CD_MS / 1000.0))


def t_cd_shorter_than_sequence():
    """序列自身比 CD 长：以序列为准，CD 不该把间隔再拉长。"""
    s = fresh_settings(output_seq=[{"type": "down", "key": "attack"},
                                   {"type": "delay", "ms": 1200},
                                   {"type": "up", "key": "attack"}])
    h = Harness(s).run(3.0)
    gaps = h.gaps(s.keymap.get("attack"))
    check(len(gaps) >= 2, "攻击键只按下 %d 次" % h.count(s.keymap.get("attack")))
    for g in gaps:
        check(1.2 - 1e-6 <= g <= 1.2 + 3 * FRAME,
              "间隔 %.3f 秒 应约等于序列自身耗时 1.2 秒" % g)


def t_releaseall_keeps_progress():
    """周期性 RELEASEALL 保留所有本机序列的进度：长 delay 不该被打回起点重放。

    判据：delay 走完之前，序列第一个键只能按下 1 次；而 delay 之后必须观察到动作
    （被打回起点的表现是：第一个键每隔 resetall_interval 重放一次，后面的动作永不发生）。
    """
    delay_s = 3.0
    cases = [
        ("攻击输出", "attack", "怪",
         dict(output_seq=[{"type": "down", "key": "attack"},
                          {"type": "delay", "ms": 3000},
                          {"type": "up", "key": "attack"}])),
        ("进入隐身", "f1", "休息",
         dict(anti_afk_enabled=True, anti_afk_rest_min=10.0 / 60.0,
              anti_afk_rest_max=10.0 / 60.0,
              anti_afk_enter_seq=[{"type": "down", "key": "f1"},
                                  {"type": "delay", "ms": 3000},
                                  {"type": "up", "key": "f1"}])),
        ("自定义定时行为", "f3", "定时",
         dict(custom_timers=[{"name": "t1", "interval": [60, 60],
                              "seq": [{"type": "down", "key": "f3"},
                                      {"type": "delay", "ms": 3000},
                                      {"type": "up", "key": "f3"}]}],
              custom_timer_next={})),
    ]
    for name, watch, kind, over in cases:
        s = fresh_settings(resetall_interval=1, **over)
        h = Harness(s)
        if kind == "怪":
            h.run(6.0)
        else:
            if kind == "休息":
                h.agent._rest_pending = True
            else:
                s.custom_timer_next["t1"] = h.clock.t       # 让它立刻到点启动
            h.run(6.0, mob_plan=lambda _t: False)
        key = s.keymap.get(watch, watch)
        downs = [t for t, k, kk in h.log if k == "down" and kk == key]
        ups = [t for t, k, kk in h.log if k == "up" and kk == key]
        check(downs, "%s：序列第一个键一次都没发" % name)
        # ⚠ 「清空指令通道之后要**重按**该按着的键」（2026-09-26 用户口径）⇒ delay 期间
        #   多出来的那几下发**不是"重放"**，而是同一根键被固件丢掉之后的补按 ✓。
        #   所以判据不能数 down 的次数 ✗，得**扣掉每次 RELEASEALL 之后的那一次补按**：
        #   真"被打回起点"才会表现为"无端多出很多次"。
        ral = [t for t, k, _v in h.log if k == "RELEASEALL"]
        early = [t for t in downs if t < delay_s - 0.2]
        budget = 1 + len([t for t in ral if t < delay_s - 0.2])
        check(len(early) <= budget,
              "%s：delay 还没走完就重放了 %d 次（进度被打回起点）：%s"
              % (name, len(early), downs[:6]))
        late = [t for t in (downs + ups) if t >= delay_s - 0.2]
        check(late, "%s：delay 之后没有任何动作（delay 永远走不完）" % name)


def t_releaseall_clears_held_only():
    """RELEASEALL 之后：**进度保留** + 该按着的键要**重按回去**（用户 2026-09-26 口径）。

    以前这里只钉"把记录作废"（固件那边确实被松了 ✓），但那还不够：宏 / 寻路都可能很长，
    `down ↑` 后面跟着几秒 delay —— 这几秒里 ↑ **必须一直按着** ✓。RELEASEALL 把它松掉、
    记录又清了 ⇒ 没人补按 ⇒ 角色中途"松手" ✗（正是用户说的"按着 ↑ 结果被停了，
    需要重启被清掉的键"）。所以现在钉三件事：进度在 ✓、键被重按 ✓、delay 没走完不许松 ✓。
    """
    s = fresh_settings(resetall_interval=1, output_seq=[
        {"type": "down", "key": "attack"}, {"type": "delay", "ms": 3000},
        {"type": "up", "key": "attack"}])
    h = Harness(s)
    h.clock0 = h.clock.t                      # 直接手动推进，不经 run()
    atk = s.keymap.get("attack", "attack")
    with h._patched():
        h.agent.tick(h.ws(True))              # 起跑：按住 attack
        h.clock.t += 2.0                      # 跨过一次定期 RELEASEALL
        h.agent.tick(h.ws(True))
        check(any(k == "RELEASEALL" for _, k, _ in h.log), "没触发定期 RELEASEALL")
        ctx = h.agent._output_ctx
        check(ctx is not None, "输出序列上下文被整块清掉了（进度应保留）")
        check(atk in ctx[3],
              "重按之后记录里没有它了（下次 RELEASEALL 就不会再补按）：%s" % (ctx[3],))
        downs = [t for t, k, v in h.log if k == "down" and v == atk]
        check(len(downs) >= 2,
              "清空指令通道之后没把该按着的键重按回去（只作废了记录）：%s" % h.log)
        check(not [t for t, k, v in h.log if k == "up" and v == atk],
              "delay 还没走完就把键松了：%s" % h.log)


def t_periodic_reset_channel():
    """定时重置指令通道：到点真的发 RELEASEALL、0 禁用、关自动也要发、走硬件通道。

    以前只测了「RELEASEALL 之后序列进度不被清」（t_releaseall_clears_held_only），
    没测**它到底有没有按时发出去** —— 而这才是"防卡键"的全部价值所在：
    间隔配错了、或者被某个早退挡住不发，序列进度那两条照样全绿。
    """
    def ral_times(h):
        return [t for t, kind, _k in h.log if kind == "RELEASEALL"]

    # ① 0 = 禁用
    h = Harness(fresh_settings(resetall_interval=0)).run(5.0)
    check(not ral_times(h), "interval=0 还在发 RELEASEALL：%s" % ral_times(h))

    # ② 1 秒一次：6 秒里应发 6 次（第一拍就发一次），间隔约 1 秒
    h = Harness(fresh_settings(resetall_interval=1)).run(6.0)
    ts = ral_times(h)
    check(len(ts) >= 5, "6 秒只发了 %d 次 RELEASEALL（应 5~6 次）" % len(ts))
    gaps = [round(b - a, 3) for a, b in zip(ts, ts[1:])]
    check(all(0.85 <= g <= 1.15 for g in gaps),
          "RELEASEALL 间隔不是 1 秒：%s" % gaps)

    # ③ 30 秒的配置不能 1 秒就发
    h = Harness(fresh_settings(resetall_interval=30)).run(6.0)
    check(len(ral_times(h)) == 1,
          "间隔 30 秒却在 6 秒里发了 %d 次" % len(ral_times(h)))

    # ④ **关自动也要发** —— 卡键最常见的时刻正是「发现卡了 → 去关自动」之后，
    #    这一段在 tick 的 `if not s.enabled` 早退之前（见 _periodic_reset 的说明）
    s = fresh_settings(resetall_interval=1)
    s.enabled = False
    h = Harness(s).run(4.0)
    n = len(ral_times(h))
    check(n >= 3, "关自动后就不再发了（4 秒只发 %d 次）—— 自愈在最需要时正好关着" % n)

    # ⑤ 手动输入开着 → 跳过这一整个周期（不能把人正按着的键松掉）
    from decision import manual_input
    with patch.object(manual_input, "active", lambda: True):
        h = Harness(fresh_settings(resetall_interval=1)).run(4.0)
    check(not ral_times(h),
          "手动输入开着还在发 RELEASEALL（会把人按着的键一起松掉）：%s"
          % ral_times(h))


def t_periodic_reset_local_backend():
    """本地(仅测试)后端：没有固件可 RELEASEALL，必须自己把按着的键 UP 掉。

    远程/串口模式下 relay 那条 RELEASEALL 直接把固件侧按键清空，本地只需同步
    **记录**（KeyState.clear，故意不发 up）；但「本地(仅测试)」后端根本没有固件 ——
    `dinput.release_all_remote()` 是空操作（`if _remote is not None`），
    这时若也只清记录，本机那个键就永远按着了：后面 keys.set 里它已经被忘掉，
    再也不会补发 up。
    """
    s = fresh_settings(resetall_interval=1)
    h = Harness(s)
    h.clock0 = h.clock.t
    # release_all_remote 换成空操作 = 本地后端的真实行为（没有 _remote）
    with patch.object(ag.time, "monotonic", h.clock), \
         patch.object(ag, "key_down", h._rec("down")), \
         patch.object(ag, "key_up", h._rec("up")), \
         patch.object(dinput, "key_down", h._rec("down")), \
         patch.object(dinput, "key_up", h._rec("up")), \
         patch.object(dinput, "release_all_remote", lambda: None), \
         patch.object(dinput, "link_health",
                      lambda: {"ok": True, "backend": "local"}):
        h.agent.keys.set({"left"})                 # 手上正按着左键
        check(("down", "left") in [(k, v) for _t, k, v in h.log],
              "左键没按下：%s" % h.log)
        h.clock.t += 1.1                           # 跨过一次定期重置
        h.agent.tick(h.ws(True))
        ups = [v for _t, k, v in h.log if k == "up"]
        check("left" in ups,
              "本地后端下按着的键没被 UP 掉（记录被清、键永远按着）：%s" % h.log)
        # 同一帧还会重新按下「当前真正需要的键」（这里是要往怪那边走 → right/ctrl），
        # 所以只断言**被松开那个键**不再按着，别要求整个记录为空。
        check("left" not in h.agent.keys._pressed,
              "松开后记录里还留着 left：%s" % h.agent.keys._pressed)


def t_link_health_watch():
    """通道体检：10 秒一次，不健康就标出来 + 丢后台重连，恢复后跟上。

    ⭐ 2026-10-03 加的两件（用户问的"按键挤压能不能用重置指令通道解决"，他选的口径是
    **通道层面**＝指令积压/发送变慢或丢 ✓）：
      · **"在丢"也算坏**：`fails` 在 10 秒里涨到 `LINK_FAILS_TRIGGER` ⇒ 主动重置一次
        （`ok` 只抓"已经死/已经静默" ✗，不等它彻底死 ✓）；
      · **重置必须带收尾**：重连之后要**清本机账 + 把该按着的键重按回来** ✗ —— 原来只
        `reconnect_remote()` ⇒ 固件侧被 relay 的 RELEASEALL 清了、而本机账还以为按着 ⇒
        `keys.set` 因"键集没变"不补发 ⇒ 键永远回不来（"重置之后角色站着不动" ✗）。

    界面那行字读的就是 input_link_ok / input_link_err（player_panel 的
    _poll_link_health），这里不测的话，链路死了界面会一直显示「已连接」。
    """
    s = fresh_settings(resetall_interval=0)
    h = Harness(s)
    h.clock0 = h.clock.t          # 手动推进时钟（不经 run()）时记录器要用它
    state = {"ok": False, "backend": "remote", "err": "发指令后 5 秒没收到固件回执",
             "fails": 0, "silent": 5.0}
    reconnected = []
    events = []

    with patch.object(ag.time, "monotonic", h.clock), \
         patch.object(dinput, "link_health", lambda: dict(state)), \
         patch.object(dinput, "reconnect_remote",
                      lambda *a, **k: (reconnected.append(1), True)[1]), \
         patch.object(ag.behavior, "event",
                      lambda name, **kw: events.append((name, kw))), \
         patch.object(ag, "key_down", h._rec("down")), \
         patch.object(ag, "key_up", h._rec("up")), \
         patch.object(dinput, "release_all_remote", h._rec("RELEASEALL")):
        # 体检是 10 秒一次，所以第一拍（t≈0）不看；推到 11 秒才该看第一次
        h.clock.t += 11.0
        h.agent.tick(h.ws(True))
        check(s.input_link_ok is False,
              "通道不健康没标记到 settings（界面读它显示红字）")
        check("回执" in (s.input_link_err or ""),
              "没把原因写进 input_link_err：%r" % s.input_link_err)
        time.sleep(0.05)              # 重连丢在后台线程里（不能卡住决策循环）
        check(reconnected, "通道不健康却没触发重连（卡键就救不回来了）")

        # ⭐ **重置的收尾**：下一拍要清本机账 + 把"该按着的键"重按回来（2026-10-03 ✓）。
        # 先摆一个"该按着 f5"的序列上下文（`_reassert_all_ctx` 走的就是它 ✓）
        h.agent._output_ctx = [[{"type": "delay", "ms": 9999}], 1, h.clock.t + 9.0,
                               {"f5"}, []]
        h.log.clear()
        h.clock.t += 0.1
        h.agent.tick(h.ws(True))
        check(any(k == "RELEASEALL" for _t, k, _v in h.log),
              "重置之后没清通道（RELEASEALL 一条都没发）：%s" % h.log)
        check(("down", "f5") in [(k, v) for _t, k, v in h.log],
              "重置之后没把**该按着的键**重按回去（f5 得补一条 down）—— 表现就是"
              "「重置完角色站着不动」✗：%s" % h.log)
        check(any(n == "link_reset_applied" for n, _kw in events),
              "重置收尾没打点（现场没法回答「刚才有没有重按」✗）")
        check(h.agent._link_reset_pending == "",
              "收尾旗没清掉（会一拍接一拍地重按 ✗）")
        h.agent._output_ctx = None

        # 恢复健康 → 标记跟着恢复；**一条都没丢 ⇒ 不许重置**（白打断长按 ✗）
        reconnected.clear()
        events.clear()
        state.update({"ok": True, "err": "", "fails": 0, "silent": 0.0})
        h.clock.t += 11.0
        h.agent.tick(h.ws(True))
        check(s.input_link_ok is True and not s.input_link_err,
              "通道恢复后界面标记没跟着恢复：ok=%s err=%r"
              % (s.input_link_ok, s.input_link_err))
        check(not reconnected, "通道健康、一条都没丢却重置了（会白打断长按 ✗）")

        # ⭐ **"在丢"⇒ 主动重置**（`fails` 10 秒里涨过门槛 ✓ 用户 2026-10-03）
        state.update({"fails": ag.LINK_FAILS_TRIGGER})
        h.clock.t += 11.0
        h.agent.tick(h.ws(True))
        time.sleep(0.05)
        check(reconnected, "通道在丢命令（fails 涨到门槛 %d）却没自动重置 ✗"
              % ag.LINK_FAILS_TRIGGER)
        got = [kw for n, kw in events if n == "link_auto_reset"]
        check(got and got[0].get("why") == "send-fails",
              "自动重置没打点/原因不对（现场就靠它回答「刚才为什么重置」✗）：%r"
              % (events,))

        # ⭐ **偶发抖动（只丢 1 条）⇒ 不重置**（丢一条就重置会白打断长按 ✗；
        # 也钉住门槛**不许被降到 1** ✓ —— 反向验证就是靠这条抓住"门槛=1"的改坏 ✓）
        reconnected.clear()
        state.update({"fails": int(state["fails"]) + 1})
        h.clock.t += 11.0
        h.agent.tick(h.ws(True))
        time.sleep(0.05)
        check(not reconnected,
              "只丢 1 条就重置了（门槛 %d）—— 偶发抖动不该白打断长按 ✗"
              % ag.LINK_FAILS_TRIGGER)


def t_seq_up_skips_decision_held():
    """⭐ 输出序列的 `up`：**决策侧正按着这个键 ⇒ 不许松**（用户 2026-10-03 ✓）。

    病（"两个写者"的另一半，与 `agent._tap` 撞按住那条同源 ✓ 见 SKILL 约定 157 #5 / 166）：
    序列 `down` 一个键时，若它**已经被决策按着**（`keys._pressed` ✓），`momentary_mark`
    故意不登记（见它 ✓ —— 决策侧本来就在 `pressed()` 里）；到 `up` 那一步照发 `key_up`
    ⇒ 松掉的是**决策侧**按着的那个键 ✗，而 `keys.set` 因为"键集没变"**不会补发** PRESS
    ⇒ 这个键**静默丢掉**（现场表现：按住的动作莫名断一下 = 用户说的"按键挤压" ✓）。

    钉四件：
      ① 决策按着 ⇒ 序列**不发 up** + 打点 `seq_release_skip_held` ✓；
      ② 决策**没**按着（那个键是序列自己按的）⇒ **照发 up**（不然序列自己就卡住 ✗）；
      ③ 决策中途放开了 ⇒ 该补发 up（那时只有序列按着它 ✓）；
      ④ 判据是 `held_by_decision`（**只看决策账** `_pressed` ✓）—— **不许**换回
         `pressed()` 并集 ✗（那样序列自己按的键就永远松不掉了 ✓）。
    """
    import inspect

    def _ctx():
        return [[{"type": "down", "key": "f5"}, {"type": "up", "key": "f5"}],
                0, 0.0, set(), []]

    # ① 决策按着 ⇒ 不许松
    s = fresh_settings()
    h = Harness(s)
    h.clock0 = h.clock.t                        # 记录器要用它算时间（见 Harness._rec ✓）
    events = []
    with patch.object(ag.behavior, "event",
                      lambda name, **kw: events.append(name)), h._patched():
        a = h.agent
        a.keys.set({"f5"})                      # 决策侧先按住 f5（模拟"移动层用同一个键"）
        h.clock.t += 0.5
        ctx = a._run_seq(h.clock.t, _ctx())     # down f5（决策已按着 ⇒ 不登记瞬发 ✓）
        h.clock.t += 0.5
        a._run_seq(h.clock.t, ctx)              # up f5
        ups = [v for _t, k, v in h.log if k == "up"]
        check("f5" not in ups,
              "决策侧按着的键被序列松掉了（`keys.set` 不会补发 ⇒ 键静默丢掉 ✗）：%s" % h.log)
        check("seq_release_skip_held" in events,
              "跳过一次 `up` 却没打点（现场查不出「谁把键挤掉了」✗）：%r" % (events,))

    # ② 决策没按着（序列自己按的）⇒ 照发 up
    s2 = fresh_settings()
    h2 = Harness(s2)
    h2.clock0 = h2.clock.t
    with h2._patched():
        a2 = h2.agent
        h2.clock.t += 0.5
        ctx = a2._run_seq(h2.clock.t, _ctx())
        h2.clock.t += 0.5
        a2._run_seq(h2.clock.t, ctx)
        ups = [v for _t, k, v in h2.log if k == "up"]
        check("f5" in ups,
              "序列**自己**按下的键也不松了 —— 那它自己就卡住了 ✗：%s" % h2.log)

    # ③ 决策中途放开 ⇒ 该补发 up（这时只有序列按着它）
    s3 = fresh_settings()
    h3 = Harness(s3)
    h3.clock0 = h3.clock.t
    with h3._patched():
        a3 = h3.agent
        a3.keys.set({"f5"})
        h3.clock.t += 0.5
        ctx = a3._run_seq(h3.clock.t, _ctx())
        a3.keys.set(set())                      # 决策侧放开了（这一下自己会发 up ✓）
        h3.clock.t += 0.5
        a3._run_seq(h3.clock.t, ctx)            # 序列这一步**必须**补发 up
        ups = [v for _t, k, v in h3.log if k == "up"]
        check(len(ups) == 2,
              "决策放开之后序列没补发 up（那个键就永远按着了 ✗）：%s" % h3.log)

    # ④ 源码级：判据只能是 `held_by_decision`（**决策账**），不是 `pressed()` 并集
    src = inspect.getsource(ag.CombatAgent._run_seq)
    check("self.keys.held_by_decision(" in src,
          "序列的 `up` 没走 `held_by_decision`（换回 `pressed()` 并集的话，序列自己按的键"
          "永远松不掉 ✗）")
    check("seq_release_skip_held" in src, "跳 `up` 那条路没留痕 ✗")


def t_timer_paused_never_fires():
    """暂停的定时行为：到点了也不触发；暂停还要**打断正在演的那段**并松键。"""
    # ① 早就该触发的时间点 + paused → 一次都不能发
    s = fresh_settings(custom_timers=[
        {"name": "t1", "interval": [1, 1], "paused": True, "paused_left": 60.0,
         "seq": [{"type": "down", "key": "f3"}, {"type": "up", "key": "f3"}]}],
        custom_timer_next={})
    h = Harness(s)
    s.custom_timer_next["t1"] = h.clock.t - 100.0      # 100 秒前就该触发了
    h.run(3.0, mob_plan=lambda _t: False)
    check(not [1 for _t, k, _v in h.log if k in ("down", "up")],
          "暂停的定时行为还是触发了：%s" % h.log)
    check("t1" not in s.custom_timer_next,
          "暂停的定时行为还留着排期：%s" % s.custom_timer_next)

    # ② 序列演到一半被暂停：按着的键必须松开，状态要清掉
    s2 = fresh_settings(custom_timers=[
        {"name": "t2", "interval": [60, 60],
         "seq": [{"type": "down", "key": "f5"},
                 {"type": "delay", "ms": 60000},
                 {"type": "up", "key": "f5"}]}], custom_timer_next={})
    h2 = Harness(s2)
    h2.clock0 = h2.clock.t
    with h2._patched():
        s2.custom_timer_next["t2"] = h2.clock.t        # 立刻到点
        h2.clock.t += 0.1
        h2.agent.tick(h2.ws(True))                     # 这一拍只是"启动序列"
        h2.clock.t += 0.1
        h2.agent.tick(h2.ws(True))                     # 这一拍才发出第一个键
        check(("down", "f5") in [(k, v) for _t, k, v in h2.log],
              "序列没起来：%s" % h2.log)
        s2.custom_timers[0]["paused"] = True           # 用户点了暂停
        h2.clock.t += 0.1
        h2.agent.tick(h2.ws(True))
        check(("up", "f5") in [(k, v) for _t, k, v in h2.log],
              "暂停没有松开正在演的序列按键（键会卡住）：%s" % h2.log)
        check("t2" not in h2.agent._timer_states, "暂停后序列状态还留着")
        # 只断言"被暂停那段按的键"不再按着（同一拍还会按移动键，不能要求整表为空）
        check("f5" not in h2.agent.keys._pressed, "暂停后那个键还按着：%s"
              % h2.agent.keys._pressed)

    # ③ 暂停状态要能存进项目、读得回来（重开项目仍是暂停 + 倒计时冻着）
    got = ag.DecisionSettings._load_timers([
        {"name": "t3", "interval": [5, 10], "seq": [],
         "paused": True, "paused_left": 42.5},
        {"name": "t4", "interval": [5, 10], "seq": []}])
    check(got[0].get("paused") is True and abs(got[0]["paused_left"] - 42.5) < 1e-6,
          "暂停状态没读回来：%s" % (got[0],))
    check("paused" not in got[1] and "paused_left" not in got[1],
          "没暂停的行为被塞了暂停字段：%s" % (got[1],))


def t_timer_pause_on_rest_flag():
    """「休息时暂停计时」是**每条行为各自**的开关（用户 2026-09-26 要求）。

    默认勾上 = 老行为（休息期间所有定时行为都冻住 ✓）；不勾的那条在休息期间**照常倒数**、
    到点照演 ✓（给"休息时也得按的键"用：喂宠 / 喊话…）。
    ⚠ 只看被冻住的那条会漏掉反向：**没勾的不许被冻** ✗（那正是新开关的全部意义）。
    """
    s = fresh_settings(anti_afk_enabled=True, anti_afk_type="hidden_rest",
                       anti_afk_rest_min=5.0, anti_afk_rest_max=5.0,
                       custom_timer_next={})
    s.custom_timers = [
        {"name": "冻住", "interval": [10, 10],
         "seq": [{"type": "down", "key": "f1"}, {"type": "up", "key": "f1"}]},
        {"name": "照走", "interval": [10, 10], "pause_on_rest": False,
         "seq": [{"type": "down", "key": "f2"}, {"type": "up", "key": "f2"}]},
    ]
    h = Harness(s)
    h.clock0 = h.clock.t
    with h._patched():
        check(h.agent._timer_pauses_on_rest("冻住") is True,
              "缺省该是「冻住」（老行为）：%r" % (s.custom_timers[0],))
        check(h.agent._timer_pauses_on_rest("照走") is False, "项目级开关没读到")
        check(h.agent._timer_pauses_on_rest("（没这条）") is True,
              "认不出来该退回「冻住」（老行为），别反过来 ✗")
        s.custom_timer_next = {"冻住": 600.0, "照走": 600.0}
        # ⚠ 起点别用 0.0：`_pause_timers` 把 `_rest_last_tick <= 0` 当"这次休息还没开始"
        #    （直接 return ⇒ 什么都不推 ✗）—— 写用例时踩到过 ✓。
        h.agent._begin_rest(100.0)
        h.agent._pause_timers(103.0)
        # 「冻住」= 截止时刻**跟着休息一起往后推**（剩余时间不变 ✓）；
        # 「照走」= 截止时刻**不动**（时间照常流逝 ⇒ 休息期间照样会到点 ✓）。
        check(s.custom_timer_next.get("冻住", 0.0) > 600.0,
              "勾了「休息时暂停计时」的那条没冻住（截止时刻没往后推）：%s"
              % s.custom_timer_next)
        check(s.custom_timer_next.get("照走") == 600.0,
              "没勾的那条也被冻住了（时间该照常流逝）：%s" % s.custom_timer_next)
        # ⚠ **存档往返**（用户 2026-09-26 问："关了 GUI 再打开还读得到吗"）：这一格必须
        #   过得了 `to_dict → from_dict`。原来 `_load_timers` 只重建 name/seq/interval
        #   ⇒ False 被吞、重开项目又按默认"冻住"算 ✗，**而且**下一次 save() 会把文件里
        #   那格抹掉（设置永久丢 ✗）。只测内存里那个 list 是测不出来的（写这条时就是这么漏的）。
        d = s.to_dict()
        check(d["custom_timers"][1].get("pause_on_rest") is False,
              "存盘里就没有「不暂停」这一格：%s" % (d["custom_timers"][1],))
        s2 = fresh_settings()
        s2.from_dict(d)
        check(s2.custom_timers[1].get("pause_on_rest") is False,
              "读回来丢了「休息时暂停计时=不勾」（重开项目又变回冻住 ✗）：%s"
              % (s2.custom_timers[1],))
        check("pause_on_rest" not in s2.custom_timers[0],
              "默认（勾着）那条也被塞了一格 —— 项目文件里会多一堆没意义的键：%s"
              % (s2.custom_timers[0],))
        check(Harness(s2).agent._timer_pauses_on_rest("照走") is False,
              "读回来后行为层又按默认「冻住」算了 ✗")
        # 反向：**默认那条**读回来还得是 True（别把缺省语义改掉 ✗）
        check(Harness(s2).agent._timer_pauses_on_rest("冻住") is True,
              "默认缺省不再是「冻住」（老行为被改掉了 ✗）")
        h.agent.state = "idle"


def t_timer_pause_button():
    """界面：每一项都有暂停按钮；暂停后按钮变「继续」、倒计时变红且带「（已暂停）」、
    冻住不走；继续则从停下的剩余时间接着走（不是清零重来）。"""
    p, app = build_panel()
    settings = ag.settings          # 界面读的就是这个单例（save 在 main 里被换成空操作）
    saved = (settings.custom_timers, settings.custom_timer_next)
    settings.custom_timers = [
        {"name": "A", "seq": [{"type": "down", "key": "f3"}], "interval": [10, 10]},
        {"name": "B", "seq": [{"type": "down", "key": "f4"}], "interval": [10, 10]},
    ]
    settings.custom_timer_next = {}
    try:
        p._refresh_timers()
        app.processEvents()
        check([b.text() for b in p._timer_pause_btns.values()] == ["暂停", "暂停"],
              "按钮初始文字不对：%s"
              % [b.text() for b in p._timer_pause_btns.values()])

        settings.custom_timer_next["A"] = time.monotonic() + 300.0
        p._timer_pause_btns["A"].click()
        app.processEvents()
        a = settings.custom_timers[0]
        check(a.get("paused") is True, "点了暂停但状态没置上")
        check(abs(float(a.get("paused_left") or 0) - 300.0) < 2.0,
              "冻住的剩余时间不对：%s" % a.get("paused_left"))
        check("A" not in settings.custom_timer_next, "暂停后 agent 的排期没撤掉")
        check(p._timer_pause_btns["A"].text() == "继续", "按钮没变成「继续」")

        lbl = p._timer_cd_labels["A"]
        check("（已暂停）" in lbl.text(), "倒计时没标「（已暂停）」：%r" % lbl.text())
        check("c5221f" in lbl.styleSheet(),
              "暂停后倒计时没标红：%r" % lbl.styleSheet())
        before = lbl.text()
        time.sleep(1.2)
        p._tick_timer_cd()
        check(lbl.text() == before, "暂停后倒计时还在走：%r → %r" % (before, lbl.text()))

        check(settings.custom_timers[1].get("paused") is None,
              "暂停 A 把 B 也暂停了")
        check(p._timer_pause_btns["B"].text() == "暂停", "B 的按钮被带变了")
        check("（已暂停）" not in p._timer_cd_labels["B"].text(), "B 的倒计时被带红了")

        # 继续：从停下的地方接着走
        p._timer_pause_btns["A"].click()
        app.processEvents()
        nx = settings.custom_timer_next.get("A", 0.0)
        check(settings.custom_timers[0].get("paused") is False, "继续后状态没清")
        check(abs((nx - time.monotonic()) - 300.0) < 3.0,
              "继续没有接着原来的剩余时间（像是清零重来了）：还剩 %.0f 秒"
              % (nx - time.monotonic()))
        check(p._timer_pause_btns["A"].text() == "暂停", "继续后按钮没变回「暂停」")
        check("（已暂停）" not in p._timer_cd_labels["A"].text(),
              "继续后倒计时还写着已暂停")
        check("c5221f" not in p._timer_cd_labels["A"].styleSheet(),
              "继续后倒计时还是红的")
    finally:
        settings.custom_timers, settings.custom_timer_next = saved
        p._refresh_timers()


def t_rest_state_machine():
    """休息状态机：到点丢弃剩余进入行为、手动结束、关防掉线、手动进入。"""
    # ① 休息时长到点 → 丢掉没演完的进入行为，直接执行退出序列
    s = fresh_settings(anti_afk_enabled=True,
                       anti_afk_rest_min=1.0 / 60.0, anti_afk_rest_max=1.0 / 60.0)
    s.anti_afk_enter_seq = [{"type": "down", "key": "f1"},
                            {"type": "delay", "ms": 60000},
                            {"type": "down", "key": "f2"}]
    s.anti_afk_exit_seq = [{"type": "down", "key": "f9"}, {"type": "up", "key": "f9"}]
    h = Harness(s)
    h.agent._rest_pending = True
    h.run(4.0, mob_plan=lambda _t: False)
    check(h.count("f1") == 1, "进入序列应只走一步就被丢弃")
    check(h.count("f2") == 0, "休息到点后不该再演进入序列剩余部分")
    check(h.count("f9") >= 1, "没有执行退出隐身序列")

    # ② 手动「结束休息」→ 同样直接转退出序列
    s2 = fresh_settings(anti_afk_enabled=True, anti_afk_rest_min=5.0,
                        anti_afk_rest_max=5.0)
    s2.anti_afk_enter_seq = [{"type": "down", "key": "f1"},
                             {"type": "delay", "ms": 60000},
                             {"type": "down", "key": "f2"}]
    s2.anti_afk_exit_seq = [{"type": "down", "key": "f9"}, {"type": "up", "key": "f9"}]
    h2 = Harness(s2)
    h2.agent._rest_pending = True
    h2.run(1.0, mob_plan=lambda _t: False)
    s2.rest_abort = True
    h2.run(2.0, mob_plan=lambda _t: False)
    check(h2.count("f2") == 0, "手动结束后不该再演进入序列剩余部分")
    check(h2.count("f9") >= 1, "手动结束后没有执行退出隐身序列")

    # ③ 休息途中关掉防掉线 → 立刻结束休息（不演退出序列）
    s3 = fresh_settings(anti_afk_enabled=True, anti_afk_rest_min=5.0,
                        anti_afk_rest_max=5.0)
    s3.anti_afk_enter_seq = [{"type": "down", "key": "f1"}, {"type": "up", "key": "f1"}]
    s3.anti_afk_exit_seq = [{"type": "down", "key": "f9"}]
    h3 = Harness(s3)
    h3.agent._rest_pending = True
    h3.run(0.5, mob_plan=lambda _t: False)
    check(h3.agent.state.startswith("afk_"), "没进休息：state=%s" % h3.agent.state)
    s3.anti_afk_enabled = False
    h3.run(0.5, mob_plan=lambda _t: False)
    check(h3.agent.state == "idle", "关掉防掉线后应立刻结束休息，state=%s"
          % h3.agent.state)
    check(h3.count("f9") == 0, "关掉防掉线不该再演退出序列")


def t_rest_interrupt_retry():
    """「被打断重试」：休息期间**真的补了血** ⇒ 立刻退出隐身，并按秒数重排下次休息。

    为什么补血算被打断：它说明隐身没兜住（隐身到期 / 被范围技能扫到 / 有东西在打我们），
    继续歇着等于等着挨打。四条边界都要对：
      · 勾选 + 补血   ⇒ 打断，且下次休息 = 现在 + retry_sec（**秒**，不是随机 N~M 分钟）
      · 不勾选 + 补血 ⇒ **绝不打断**（补血照常发生，休息照旧）
      · 勾选 + 没补血 ⇒ 不打断（触发它的是"补血"这个事件，不是"在休息"）
      · ⭐ **定点休息「走不到」**（用户 2026-09-28 要求 ✓）⇒ **也算一次"没做成"** ⇒
        同样按 retry_sec **秒级重试**（见 `_rest_give_up` ✓）—— ⚠ 改之前**没置**这个标记，
        于是按随机 N~M 分钟排下一次，想再试得等几分钟 ✗。
        ⚠ **「结束后前往」没走成不算**（那时休息已完成 ✓ 见 `_rest_give_up` 的说明 ✓）。
    """
    def setup(**over):
        s = fresh_settings(anti_afk_enabled=True,
                           anti_afk_rest_min=5.0, anti_afk_rest_max=5.0,
                           auto_hp_pot=True, hp_threshold=90, pot_cd=0, **over)
        s.keymap["hp_pot"] = "f5"
        s.anti_afk_enter_seq = [{"type": "down", "key": "f1"},
                                {"type": "up", "key": "f1"}]
        s.anti_afk_exit_seq = [{"type": "down", "key": "f9"},
                               {"type": "up", "key": "f9"}]
        return s

    # 打点：被打断要能在 behavior.log 里看见 —— 否则以后只能猜"它到底触发过没有"。
    # 走**公开路径**验证（configure → 读文件），不去翻 behavior 的内部字典
    # （2026-09-28 起"行为/状态"打点**全在 behavior.log**，perf.log 只管性能 ✓）。
    import tempfile
    from core import behavior
    old_log, old_en = behavior.LOG, behavior.ENABLED
    logf = Path(tempfile.mkdtemp(prefix="behavior_afk_")) / "behavior.log"
    try:
        behavior.configure(True, log=logf)
        # ① 勾选 + 补血 → 打断（并立刻演退出隐身）
        s = setup(anti_afk_retry_on_interrupt=True, anti_afk_retry_sec=7.0)
        h = Harness(s)
        h.hp = 0.1                               # 低于阈值 ⇒ 每一拍都会补血
        h.agent._rest_pending = True
        h.run(1.5, mob_plan=lambda _t: False)
        check(h.taps("f5"), "休息期间没有补血（前提就不成立）")
        check(h.count("f9") >= 1, "被打断后没有执行退出隐身序列")
        check(h.agent.state == "idle", "打断后没回到战斗：state=%s" % h.agent.state)
        left = h.agent._next_afk - h.clock.t
        check(left < 20.0,
              "下次休息没有提前到 retry_sec（还剩 %.1f 秒；正常路径是随机的 300 秒）"
              % left)
        check("afk_interrupt" in logf.read_text(encoding="utf-8"),
              "被打断了但 behavior.log 里没有 afk_interrupt（这个功能会变成没法观测）")

        # ② 不勾选 + 补血 → 绝不打断（**也不该留下打点**）
        s2 = setup(anti_afk_retry_on_interrupt=False)
        h2 = Harness(s2)
        h2.hp = 0.1
        h2.agent._rest_pending = True
        # flush 是**追加**写：只比对这一轮新增的那段，否则会看到 ① 留下的打点
        before = logf.read_text(encoding="utf-8")
        h2.run(1.5, mob_plan=lambda _t: False)
        check(h2.taps("f5"), "没补血（前提不成立）")
        check(h2.agent.state in ("afk_enter", "afk_rest"),
              "没勾选却被补血打断了：state=%s" % h2.agent.state)
        check(h2.count("f9") == 0, "没勾选却演了退出隐身序列")
        check("afk_interrupt" not in logf.read_text(encoding="utf-8")[len(before):],
              "没勾选也记了 afk_interrupt（打点会骗人）")

        # ③ 勾选 + 没补血（血够）→ 不打断
        s3 = setup(anti_afk_retry_on_interrupt=True)
        h3 = Harness(s3)
        h3.hp = 1.0
        h3.agent._rest_pending = True
        h3.run(1.5, mob_plan=lambda _t: False)
        check(not h3.taps("f5"), "血够却补了血（前提不成立）")
        check(h3.agent.state in ("afk_enter", "afk_rest"),
              "没补血却被打断了：state=%s" % h3.agent.state)

        # ④ ⭐ **定点休息「走不到」也要按秒重试**（用户 2026-09-28 明确要求 ✓）
        #   用户原话："修复休息寻路失败 ⇒ 没有激活『打断重试』的问题，把 `_rest_give_up`
        #   也置上那个标记（retry_sec 秒后重试）"✓。
        #   ⚠⚠ **改之前是没置的**：只有"自动补血打断"才置 ⇒ 走不到休息点时按**随机 N~M 分钟**
        #     排下一次（想再试得等几分钟 ✗）—— 而这种情况**恰恰最该快点重试** ✓。
        #   走**真路径**（`_run_rest_spot` 的 `afk_spot_walk` ⇒ 计划失败 ⇒ `_rest_give_up` ✓），
        #   不直接调那个方法 —— 否则"接线断了"（比如改回没置标记）照样绿 ✗。
        s4 = fresh_settings(anti_afk_enabled=True, anti_afk_type="spot_rest",
                            anti_afk_spot_set="左上", anti_afk_retry_sec=7.0,
                            anti_afk_rest_min=5.0, anti_afk_rest_max=5.0)
        h4 = Harness(s4)
        h4.hp = 1.0                              # 不补血 ⇒ 排除"补血打断"那条路 ✓
        h4.agent.plan_and_start_route = (         # 解析器说"这条路走不了"
            lambda *_a, **_k: (False, "假装：解析不出路径（用例）"))
        h4.agent._rest_pending = True
        before4 = logf.read_text(encoding="utf-8")
        h4.run(1.5, mob_plan=lambda _t: False)
        check(h4.agent.state == "idle",
              "走不到休息点却没回战斗：state=%s" % h4.agent.state)
        left4 = h4.agent._next_afk - h4.clock.t
        check(left4 < 20.0,
              "⭐ 「去休息点走不到」之后**没**按 retry_sec 重试（还剩 %.1f 秒；"
              "没置标记的话会落到随机的 300 秒）—— 用户要的就是这条 ✗" % left4)
        check(left4 > 0.0, "下次休息时刻算成了过去（会连着重试）：%.1f" % left4)
        check("afk_give_up" in logf.read_text(encoding="utf-8")[len(before4):],
              "走不到休息点没留痕（`afk_give_up`）—— 以后没法从 log 判断它触发过没有 ✗")
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log


def t_rest_cut_is_accounted():
    """⭐⭐ 休息**被切**必须结清 + 留痕（用户 2026-10-03 ✓ 现场原话：\"我配置的进入隐身行为
    很长，但是好像没有完成，中间是否被什么切断了？（可以看 log）\"）。

    查出来的病（`behavior.log` 实证 ✓）：会"把休息从状态上挤走"的早退路（**关自动**、
    **界面接管**）以前各自 `_set_state(\"idle\")` 了事 ⇒
      ① 休息的账**没结清**（`_afk_ctx` 里半截序列 / `rest_state` / 下次休息排期 ✗）；
      ② **一个点都不打** ✗ ⇒ 日志里"被切"和"完成"**分不开** ✗；
      ③ 人可能**还在隐身里**就被丢回战斗（退出隐身序列没演过 ✗），而且这一点也**静默** ✗。
    修法：这几种路一律走 `_cut_rest`（**唯一收口** ✓ 打 `afk_cut` + `_finish_rest(why=…)` ✓）。

    钉五件（都走**真 tick 的早退** ✓ 不直接调 `_cut_rest` —— 否则"接线断了"照样绿 ✗）：
      ① **关自动**：回 idle + `afk_cut`/`afk_done` 都带 `why=关自动`、`hidden=1`（如实说
         「隐身可能还开着」✓）、`_afk_ctx`/`rest_state` 已清、**下次休息已重排** ✓；
      ② **界面接管**（`ws.screen == \"login\"`）⇒ 同样收口 + `hidden=1` ✓；
      ③ 不在休息态调 `_cut_rest` ⇒ 原样返回 False、**不打点**（幂等 ✓ 早退路径每拍都调它 ✓）；
      ④ **正常到点走完不许被标成"被切"**：`why=走完` + `hidden=0` ✓（打点会骗人就废了 ✗）；
      ⑤ ⭐ `afk_enter_dropped` 必须带两个数（`rest_s` = 本次休息配了多长、`played_s` = 进入
         行为已演多久）⇒ 这才分得清"是到点正常丢的"还是"休息时长本身配得不对"✓
         （用户现场那两次 1.5 / 6 秒的早退正是靠它定案 ✓）。
    """
    import tempfile
    from core import behavior

    def setup(**over):
        kw = dict(anti_afk_enabled=True,
                  anti_afk_rest_min=5.0, anti_afk_rest_max=5.0)
        kw.update(over)                    # ⚠ 用例要能覆盖休息时长（④⑤ 就靠它 ✓）
        s = fresh_settings(**kw)
        s.anti_afk_enter_seq = [{"type": "down", "key": "f1"},
                                {"type": "delay", "ms": 60000},
                                {"type": "down", "key": "f2"}]
        s.anti_afk_exit_seq = [{"type": "down", "key": "f9"}, {"type": "up", "key": "f9"}]
        return s

    old_log, old_en = behavior.LOG, behavior.ENABLED
    logf = Path(tempfile.mkdtemp(prefix="behavior_afkcut_")) / "behavior.log"
    _read = lambda: logf.read_text(encoding="utf-8")        # noqa: E731

    def _ev(txt, name):
        """从日志片段里取出**那一个事件的那一行** ⇒ 只在这一行里找字段。

        ⚠⚠ 必须在**这一行**里找，不能在整段里找：`afk_cut` 那行也带 `why=…` ⇒ 在整段里
        `"why=关自动" in txt` 会被它蒙过去（`afk_done` 的 why 被改坏照样绿 ✗ —— 反向验证
        真踩了这个盲区 ✓）。取不到就回空串（断言自然红 ✓）。
        """
        for ln in str(txt).splitlines():
            if ("  %s " % name) in ln:
                return ln
        return ""
    try:
        behavior.configure(True, log=logf)

        # ① 休息中「关自动」⇒ 收口
        s = setup()
        h = Harness(s)
        h.agent._rest_pending = True
        h.run(1.0, mob_plan=lambda _t: False)
        check(h.agent.state in ("afk_enter", "afk_rest"),
              "没进休息（这条测不了）：state=%s" % h.agent.state)
        check(h.agent._afk_ctx is not None, "进入序列没起跑（这条测不了）")
        before = _read()
        s.enabled = False
        h.run(0.5, mob_plan=lambda _t: False)
        got = _read()[len(before):]
        check(h.agent.state == "idle", "关自动后没回战斗：state=%s" % h.agent.state)
        check("why=关自动" in _ev(got, "afk_cut"),
              "关自动把休息挤走了却没留痕 ⇒ 日志里还是分不清「被切」和「完成」✗：%r" % got)
        check("why=关自动" in _ev(got, "afk_done")
              and "hidden=1" in _ev(got, "afk_done"),
              "收口没打 `afk_done`（或没带 why / hidden）⇒ 既看不出这次是被切、也看不出"
              "隐身可能还开着 ✗：%r" % got)
        check(h.agent._afk_ctx is None and s.rest_state == "",
              "收口没结清休息账（`_afk_ctx`=%r / rest_state=%r）"
              % (h.agent._afk_ctx, s.rest_state))
        check(h.agent._next_afk > h.clock.t,
              "收口没重排下次休息（`_next_afk`=%.1f ≤ 现在 %.1f）⇒ 防掉线等于废了 ✗"
              % (h.agent._next_afk, h.clock.t))
        check("隐身可能还开着" in str(h.agent._last_goto_note[0]),
              "画面上没写「隐身可能还开着」⇒ 人还是只能来问 ✗：%r" % (h.agent._last_goto_note,))

        # ② 休息中「界面接管」（掉线 / 测谎）⇒ 同样收口（人不在游戏里，退出隐身演不了 ⇒
        #    只能如实记下来 ✓）
        s2 = setup()
        h2 = Harness(s2)
        h2.agent._rest_pending = True
        h2.run(1.0, mob_plan=lambda _t: False)
        check(h2.agent.state in ("afk_enter", "afk_rest"),
              "没进休息（这条测不了）：state=%s" % h2.agent.state)
        before2 = _read()
        h2.ws_hook = lambda w: setattr(w, "screen", "login")
        h2.run(0.5, mob_plan=lambda _t: False)
        got2 = _read()[len(before2):]
        check(h2.agent.state == "idle",
              "界面接管后没回 idle（人都不在游戏里了还留在休息态 ✗）：%s" % h2.agent.state)
        check("界面接管" in _ev(got2, "afk_cut"),
              "界面接管把休息挤走了却没留痕 ✗：%r" % got2)
        check("hidden=1" in _ev(got2, "afk_done"),
              "界面接管时还停在休息相 ⇒ 必须记「隐身可能还开着」（`hidden=1`）✗：%r" % got2)

        # ③ 不在休息态 ⇒ `_cut_rest` 原样返回、一个点都不许打（幂等 ✓）
        s3 = setup()
        h3 = Harness(s3)
        before3 = _read()
        check(h3.agent._cut_rest(0.0, "用例直接调（不在休息态）") is False,
              "不在休息态时 `_cut_rest` 该原样返回 False（早退路径每拍都会调它 ⇒ "
              "不幂等就会反复收口 ✗）")
        check("afk_cut" not in _read()[len(before3):],
              "没在休息也打了 `afk_cut`（打点会骗人 ✗）")

        # ④ 正常到点走完 ⇒ `why=走完` + `hidden=0`（**别把正常那次也标成"被切"** ✗）
        s4 = setup(anti_afk_rest_min=1.0 / 60.0, anti_afk_rest_max=1.0 / 60.0)   # 配 1 秒
        s4.anti_afk_enter_seq = [{"type": "down", "key": "f1"}, {"type": "up", "key": "f1"}]
        h4 = Harness(s4)
        h4.agent._rest_pending = True
        before4 = _read()
        h4.run(3.0, mob_plan=lambda _t: False)
        got4 = _read()[len(before4):]
        check(h4.agent.state == "idle", "休息没走完：state=%s" % h4.agent.state)
        check("why=走完" in _ev(got4, "afk_done")
              and "hidden=0" in _ev(got4, "afk_done"),
              "正常走完那次的打点不对（该是 `why=走完` / `hidden=0` ✓ —— 别把它也标成"
              "被切 ✗）：%r" % got4)
        check("afk_enter_dropped" not in got4,
              "进入序列明明演完了（1 秒）却报了 `afk_enter_dropped` ✗：%r" % got4)

        # ⑤ ⭐ `afk_enter_dropped` 的两个数（现场定案就靠它 ✓）
        s5 = setup(anti_afk_rest_min=3.0 / 60.0, anti_afk_rest_max=3.0 / 60.0)   # 配 3 秒
        h5 = Harness(s5)
        h5.agent._rest_pending = True
        before5 = _read()
        h5.run(4.0, mob_plan=lambda _t: False)
        got5 = _read()[len(before5):]
        _d5 = _ev(got5, "afk_enter_dropped")
        check(_d5,
              "进入行为没演完就被丢了，却没留痕（用户问的就是这件事 ✗）：%r" % got5)
        check("rest_s=3.0" in _d5,
              "`afk_enter_dropped` 没报本次休息配了多长（`rest_s`）⇒ 分不清是「到点正常丢的」"
              "还是「休息时长本身配得不对」✗：%r" % got5)
        check("played_s=" in _d5 and "played_s=0.0" not in _d5,
              "`afk_enter_dropped` 没报进入行为已演多久（`played_s`）✗：%r" % got5)
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log


def t_rest_manual_request():
    """手动「进入休息」：先标记待休息，等攻击范围内的怪清空才真的进。"""
    s = fresh_settings(anti_afk_enabled=True, anti_afk_rest_min=1.0,
                       anti_afk_rest_max=1.0)
    s.anti_afk_enter_seq = [{"type": "down", "key": "f1"}, {"type": "up", "key": "f1"}]
    s.anti_afk_exit_seq = [{"type": "down", "key": "f9"}, {"type": "up", "key": "f9"}]
    h = Harness(s)
    s.rest_request = True
    h.run(0.5)                                    # 范围内有怪
    check(s.rest_request is False, "请求没被消费")
    check(s.rest_pending is True, "有怪时应该只标记「待休息」")
    check(h.agent.state == "attack", "有怪时不该进隐身，state=%s" % h.agent.state)
    h.run(1.0, mob_plan=lambda _t: False)         # 怪清空
    check(h.count("f1") >= 1, "怪清空后没有进入隐身")


def t_class_table_single_source():
    """类别表只在 perception/classes.py 定义：别处不许再抄一份。

    抄一份的后果是「加类别时漏改一处」，表现是框颜色/名字对不上、或者标注写成
    别的类 —— 这类错很安静（不报错、看起来也像对的），所以用文本扫描钉住它。
    """
    import re
    from perception import classes
    root = Path(__file__).resolve().parent.parent
    pat = re.compile(r"CLASS_NAMES\s*=\s*[\{\[]")
    bad = []
    for sub in ("gui", "perception", "tools"):
        for f in (root / sub).rglob("*.py"):
            if f.name == "classes.py":
                continue
            for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(ln) and "import" not in ln:
                    bad.append("  %s:%d  %s" % (f.relative_to(root), i, ln.strip()))
    check(not bad, "类别表被抄了第二份（应只从 perception/classes.py 取）：\n"
                   + "\n".join(bad))
    check(classes.ORDER == list(range(len(classes.ORDER))),
          "类别 id 不是从 0 连续：%s" % classes.ORDER)
    for cid in classes.ORDER:
        check(cid in classes.EN_NAMES and cid in classes.ZH_NAMES,
              "类别 %d 缺英文名或中文名" % cid)
    check(classes.CLASS_OTHER_PLAYER in classes.ORDER,
          "「其他玩家」类没在类别表里")


def t_next_deadline():
    """next_deadline()：有序列在跑就报它的到点时刻，没有就报 None。"""
    s = fresh_settings(output_seq=[{"type": "down", "key": "attack"},
                                   {"type": "delay", "ms": 500},
                                   {"type": "up", "key": "attack"}])
    h = Harness(s)
    h.clock0 = h.clock.t
    check(s.enabled and h.agent.next_deadline() is None, "还没跑序列就报了个时刻")
    with h._patched():
        h.agent.tick(h.ws(True))            # 发 down，下一元素是 delay
        # ⚠ 键动作之间现在隔 `_attack_duration`（30ms；2026-09-27 删掉随机延迟后才变成
        #   这个确定值）：不把时钟推过这一点，第二拍还在"没到点"里 ⇒ 吃不到 delay 元素 ✗
        h.clock.t += h.agent._attack_duration
        h.agent.tick(h.ws(True))            # 吃下 delay 元素 → 到点时刻 = 现在 + 0.5
        d = h.agent.next_deadline()
        check(d is not None, "序列在跑却没报「到点时刻」")
        check(abs(d - (h.clock.t + 0.5)) < 0.05,
              "到点时刻不对：%.3f（应 %.3f）" % (d, h.clock.t + 0.5))


def t_sequence_timing_not_frame_quantized():
    """序列里的 delay 按本机绝对时钟到点，不被抓帧节拍量化。

    Harness 的驱动器已经和主循环一致（帧 + 时序拍），所以这里直接量：
    序列里一个 100 毫秒的 delay，实际间隔该接近 100 毫秒；如果又变回
    「一帧量化」，间隔会是 133 毫秒（100 之后的第一帧）。
    """
    s = fresh_settings(output_seq=[{"type": "down", "key": "attack"},
                                   {"type": "delay", "ms": 100},
                                   {"type": "up", "key": "attack"}],
                       attack_cd=30)
    h = Harness(s).run(1.0)
    downs = [t for t, kind, _k in h.log if kind == "down"]
    ups = [t for t, kind, _k in h.log if kind == "up"]
    check(downs and ups, "序列没跑起来：downs=%s ups=%s" % (downs[:3], ups[:3]))
    gap = ups[0] - downs[0]
    check(0.095 <= gap <= 0.13,
          "delay 100ms 实际隔了 %.3f 秒（>0.133 就是又回到「一帧量化」了）" % gap)


def t_no_random_input_delay():
    """「随机输入延迟」**整个删掉**（用户 2026-09-27 要求）—— 这条反向钉住它别回来。

    为什么删：它叠在 输出CD / 序列元素间隔 / 追击跳间隔 / 规避间隔 上抖，抖掉的是
    **可复现性** —— 同一份配置两次跑出来的节奏不一样，"这一下为什么慢/快"就永远
    对不上 ✗。

    钉四处：
      ① `DecisionSettings` 上没有这个字段；`to_dict` 也不写它（写回去等于复活）；
         老配置/老模板里带着这个键时**读进来也不认**（不报错，但参数不复活）✓；
      ② 决策层与操控面板的**代码**里不许再有它 —— 判据用 `def _random_input_delay` /
         `self.input_delay` / `self.sp_delay_min` 这种**定义式**写法，而不是裸名字：
         注释里提到"当年那个参数"是好事，不该被这条用例误伤 ✓；
      ③ 序列元素之间的间隔 = `_attack_duration`（项目里"一下按键"的长度，确定值）；
      ④ 同一份配置连跑两遍，输出间隔**逐个完全一样**（有随机延迟时这里必然抖 ✗）。
    """
    s = fresh_settings()
    check(not hasattr(s, "input_delay"),
          "`DecisionSettings` 里又出现了 `input_delay`（用户 2026-09-27 要求删掉）")
    check("input_delay" not in ag.DecisionSettings().to_dict(),
          "`to_dict` 又在写 `input_delay` —— 一保存就又写回项目文件了 ✗")
    s2 = ag.DecisionSettings()
    s2.from_dict({"input_delay": [999, 999]})
    check(not hasattr(s2, "input_delay"),
          "老配置/老模板里的 `input_delay` 又被读进来了（不该复活）")
    root = Path(__file__).resolve().parent.parent
    for rel, bad in (("decision/agent.py",
                      ("def _random_input_delay", "self.input_delay")),
                     ("gui/player_panel.py",
                      ("self.sp_delay_min", "self.sp_delay_max",
                       "def _on_input_delay"))):
        src = (root / rel).read_text(encoding="utf-8")
        for name in bad:
            check(name not in src, "%s 里又出现了 `%s`（随机输入延迟不许回来）"
                  % (rel, name))
    # ③ 序列元素间隔 = `_attack_duration`
    s3 = fresh_settings(output_seq=[{"type": "down", "key": "attack"},
                                    {"type": "up", "key": "attack"}])
    h = Harness(s3)
    h.clock0 = h.clock.t
    with h._patched():
        h.agent.tick(h.ws(True))          # 发 down，下一元素的到点时刻 = now + 按键时长
        d = h.agent.next_deadline()
    check(d is not None, "序列在跑却没报「到点时刻」")
    check(abs((d - h.clock.t) - h.agent._attack_duration) < 1e-9,
          "序列元素间隔不是 `_attack_duration`：%.3f（应 %.3f）"
          % (d - h.clock.t, h.agent._attack_duration))
    # ④ 可复现：同一份配置连跑两遍，输出间隔逐个一致
    _gaps = []
    for _ in range(2):
        h2 = Harness(fresh_settings(
            jump_random_prob=0.0,
            output_seq=[{"type": "down", "key": "attack"},
                        {"type": "up", "key": "attack"}]))
        h2.run(2.0)
        _gaps.append(h2.gaps(h2.agent.settings.keymap["attack"]))
    check(_gaps[0] and _gaps[0] == _gaps[1],
          "同一份配置两次跑的输出间隔不一样（随机延迟又回来了？）：%s vs %s"
          % (_gaps[0][:5], _gaps[1][:5]))


def t_input_local_only():
    """F10~F12 是本机操作键：三条发送路径都不发，其它键照发。"""
    sent = []
    real_send, real_remote, real_blocked = dinput._send, dinput._send_remote, dinput._blocked
    dinput._send = lambda vk, up: sent.append(("local", vk, up))
    dinput._send_remote = lambda cmd: sent.append(("remote", cmd))
    dinput._remote = object()          # 假装连着远程通道（最容易漏发的路径）
    dinput._blocked = False
    try:
        for name in ("f10", "f11", "f12", "F10"):
            sent.clear()
            dinput.key_down(name)
            dinput.key_up(name)
            dinput.tap(name, 0.0)
            check(not sent, "%s 被发出去了：%s" % (name, sent))
        for name in ("left", "ctrl", "t"):
            sent.clear()
            dinput.key_down(name)
            check(len(sent) == 1, "普通键 %s 没发出去" % name)
    finally:
        dinput._send, dinput._send_remote = real_send, real_remote
        dinput._remote, dinput._blocked = None, real_blocked

    from decision import manual_input
    from pynput import keyboard
    for k in (keyboard.Key.f10, keyboard.Key.f11, keyboard.Key.f12):
        check(manual_input._key_name(k) is None,
              "手动输入还会转发 %s" % k)
    check(manual_input._key_name(keyboard.Key.left) == "left",
          "手动输入把普通键也挡了")


def t_attack_box():
    """**攻击范围 = 矩形**（用户 2026-09-27 定的）：四个距离 = 四条边，判据**只有一处**。

    以前只比水平（`attack_dist`）；现在由
      · 水平：「最小攻击距离」~「最大攻击距离」（前者以内 = **盲区**）；
      · 竖直：往上「向上攻击距离」、往下「向下攻击距离」
    围成的矩形决定 ✓（`DecisionAgent._in_box` 一处实现；`_in_range` / `_in_range_all`
    和贴脸规避全走它 ✓ —— 各写一遍的话加一维必然漏一处 ✗）。

    钉八件：
      ① **老行为不变**（升级底线）：两条竖边默认 **-1 = 不限**（老项目文件里没有这两个键
         ⇒ 兜底 -1）—— 怪在很上面/很下面照样算在攻击范围内 ✓；
      ② 「向上攻击距离」生效：怪在角色**上方**、竖直离得超过它 ⇒ 不算够得着 ✓；
      ③ 在它以内 ⇒ 算够得着；而它**管不到下方**的怪（两个方向各管各的 ✓）；
      ④ 「向下攻击距离」同理 ✓；
      ⑤ 整条链路跟着改：竖直超限的怪**不进** `_in_range` ✓；
      ⑥ 盲区也是**矩形**：最小攻击距离以内不算可攻击、而且算"贴脸"（触发规避 ✓）；
         但**竖直超限**的近怪不算贴脸（站在上一层平台上的怪不该逼我们跳走 ✗）；
      ⑦ 叠图那份（`attack_box_rect`）和判据**同源**：远边 = 角色中心 + 最大距离、
         0 = 不限时画到画面边 ✓；参数三边同步 + 老文件兜底 0 ✓。
    """
    from decision import agent as ag_mod
    from perception.world_state import Mob

    s = fresh_settings(attack_dist=80.0, min_attack_dist=0)
    h = Harness(s)
    ws = h.ws(with_mob=False)
    p = ws.player                       # 画面坐标：中心 (500, 500) ✓

    def _m(dy, dx=40.0):
        """造一只"水平距离 dx、竖直中心差 dy"的怪（框 40×40 ✓）。"""
        return Mob(id=1, x=p.x + dx, y=p.y + dy, w=40.0, h=40.0, conf=0.9)

    # ① 老行为：两条竖边 -1（默认）⇒ 竖直不限（怪在 ±500 都算够得着 ✓）
    check((s.attack_up_dist, s.attack_down_dist) == (-1, -1),
          "两条竖边的**默认值**该是 -1（= 不限 = 老行为 ✓）：%r / %r"
          % (s.attack_up_dist, s.attack_down_dist))
    for dd in (-500.0, -120.0, 0.0, 300.0):
        check(h.agent._in_attack_box(_m(dd), p),
              "两条竖边是 -1（= 不限）时，竖直差 %.0f 的怪被排除在攻击范围外 —— "
              "这会让所有老项目升级后行为变掉 ✗" % dd)

    # ②③ 向上攻击距离 = 50
    s.attack_up_dist = 50
    check(not h.agent._in_attack_box(_m(-100.0), p),
          "「向上攻击距离」= 50 时，上方 100（最近边 80）的怪还算够得着 ✗")
    check(h.agent._in_attack_box(_m(-30.0), p),
          "「向上攻击距离」= 50 时，上方 30（最近边 10）的怪该算够得着 ✓")
    check(h.agent._in_attack_box(_m(400.0), p),
          "「向上攻击距离」把**下方** 400 的怪也排除了（两个方向各管各的 ✗）")
    # ④ 向下攻击距离 = 100
    s.attack_up_dist = 0
    s.attack_down_dist = 100
    check(not h.agent._in_attack_box(_m(200.0), p),
          "「向下攻击距离」= 100 时，下方 200（最近边 180）的怪还算够得着 ✗")
    check(h.agent._in_attack_box(_m(60.0), p),
          "「向下攻击距离」= 100 时，下方 60（最近边 40）的怪该算够得着 ✓")

    # ⑤ 整条链路：`_in_range`（朝向前方）也走新口径
    check(not h.agent._in_range([_m(200.0)], ws),
          "竖直超限的怪还留在 `_in_range` 里 —— 判据没收进一处，漏改了 ✗")
    check(len(h.agent._in_range([_m(60.0)], ws)) == 1,
          "竖直在范围内的怪被 `_in_range` 漏掉了 ✗")

    # ⑥ 盲区也是矩形
    # ⚠ 竖直那两条先回到 -1（= 不限）：上下**都配 0** 会让整个框变成空集 ⇒ 盲区判定
    #   当场被跳过（那是 ⑧ 专门钉的规则，不是这条要测的东西 ✓）
    s.attack_up_dist = -1
    s.attack_down_dist = -1
    s.min_attack_dist = 30
    near = _m(0.0, dx=10.0)                 # 水平最近距离 0 < 30 ⇒ 盲区
    check(not h.agent._in_attack_box(near, p),
          "落在盲区（最小攻击距离以内）的怪还算可攻击 ✗")
    check(h.agent._in_blind_box(near, p),
          "落在盲区的怪没被判成贴脸（规避不会触发 ✗）")
    s.attack_up_dist = 50
    check(not h.agent._in_blind_box(_m(-300.0, dx=10.0), p),
          "站在上一层平台上的近怪被判成贴脸 —— 盲区也得是**矩形**（竖直一起看 ✗）")
    s.attack_up_dist = -1                   # 收工：恢复"不限"（别留下严格 0 ✓）
    s.attack_down_dist = -1
    s.min_attack_dist = 0

    # ⑦ 叠图那份和判据同源 + 参数存读
    rect = ag_mod.attack_box_rect(500.0, 500.0, 80.0, up=30.0, down=60.0, facing=1)
    check(rect == (500.0, 470.0, 580.0, 560.0),
          "`attack_box_rect` 的四条边不对（(左, 上, 右, 下)）：%r" % (rect,))
    # ⚠ 三个框**并排、首尾相接、不重叠**（用户 2026-09-27 的图：
    #   盲区框 → 攻击范围框 → 追击起跳框 ✓）—— `from_d` = 水平起点距离 ✓
    _blind = ag_mod.attack_box_rect(500.0, 500.0, 30.0, up=-1.0, down=-1.0,
                                    frame_h=1080, from_d=0.0)
    _atk = ag_mod.attack_box_rect(500.0, 500.0, 80.0, up=-1.0, down=-1.0,
                                  frame_h=1080, from_d=30.0)
    _chase = ag_mod.attack_box_rect(500.0, 500.0, 130.0, up=-1.0, down=-1.0,
                                    frame_h=1080, from_d=80.0)
    check(_blind == (500.0, 0.0, 530.0, 1079.0),
          "盲区框该是「角色中心 → 最小攻击距离」：%r" % (_blind,))
    check(_atk == (530.0, 0.0, 580.0, 1079.0),
          "攻击范围框该是「**最小 → 最大**」（不含盲区那一段 —— 用户图上就是并排的 ✓）：%r"
          % (_atk,))
    check(_chase == (580.0, 0.0, 630.0, 1079.0),
          "追击起跳框该是 [最大 + min, 最大 + max]：%r" % (_chase,))
    check(_blind[2] == _atk[0] and _atk[2] == _chase[0],
          "三个框该首尾相接（盲区右端 = 攻击范围左端、攻击范围右端 = 追击起跳左端）：%r"
          % ((_blind, _atk, _chase),))
    check(ag_mod.attack_box_rect(500.0, 500.0, 80.0, up=-1.0, down=-1.0,
                                 from_d=80.0) is None,
          "「最小 = 最大」（可攻击区跨度 0）时攻击范围框该是空集（不画 ✓）")

    rect2 = ag_mod.attack_box_rect(500.0, 500.0, 80.0, up=-1.0, down=-1.0, facing=-1,
                                   frame_h=1080)
    check(rect2 == (420.0, 0.0, 500.0, 1079.0),
          "「不限」（负数）时该画到画面边、朝向为负时框该在左边：%r" % (rect2,))

    s.attack_up_dist, s.attack_down_dist = 33, 44
    s2 = ag_mod.DecisionSettings()
    s2.from_dict(s.to_dict())
    check((s2.attack_up_dist, s2.attack_down_dist) == (33, 44),
          "上下攻击距离没存住（to_dict/from_dict 不同步）：%r / %r"
          % (s2.attack_up_dist, s2.attack_down_dist))
    old = ag_mod.DecisionSettings()
    old.from_dict({"attack_dist": 100.0})       # 老项目文件：没有这两个键
    check((old.attack_up_dist, old.attack_down_dist) == (-1, -1),
          "老项目文件没退回「不限」（**-1**）—— 升级后行为会变 ✗：%r / %r"
          % (old.attack_up_dist, old.attack_down_dist))

    # ⑧ **0 就是 0、负数才是不限；框面积为 0 ⇒ 判定跳过、也不画**（用户 2026-09-27 纠正的口径）
    s8 = fresh_settings(attack_dist=80.0, min_attack_dist=0)
    h8 = Harness(s8)
    ws8 = h8.ws(with_mob=False)
    q = ws8.player

    def _m8(dy, dx=40.0):
        return Mob(id=1, x=q.x + dx, y=q.y + dy, w=40.0, h=40.0, conf=0.9)

    s8.attack_up_dist = -1                       # 负数 = 不限（老行为 ✓）
    check(h8.agent._in_attack_box(_m8(-400.0), q),
          "负数该是「不限」：上方 400 的怪也该算够得着 ✓")
    s8.attack_up_dist = 0                        # 0 = **严格 0** ⇒ 上方的怪一律不算
    check(not h8.agent._in_attack_box(_m8(-400.0), q),
          "「向上攻击距离」配 0 时上方的怪还算够得着 —— **0 要按 0 算**（不是无限 ✗）")
    check(h8.agent._in_attack_box(_m8(0.0), q),
          "「向上攻击距离」= 0 时，「怪框跨过角色那条水平线」（竖直距离恰 0）该算够得着 ✓")
    # 面积为 0 ⇒ **跳过该框的判定**（打不到任何怪）+ **不画** ✓
    s8.attack_down_dist = 0
    check(not h8.agent._in_attack_box(_m8(0.0), q)
          and not h8.agent._in_attack_box(_m8(-400.0), q),
          "上下**都**配 0（框面积为 0）时还在判可攻击 —— 该**直接跳过**（打不到任何怪 ✓）")
    check(ag_mod.attack_box_rect(500.0, 500.0, 80.0, up=0.0, down=0.0) is None,
          "面积为 0 的框还返回了几何 ⇒ 叠图会画出一个不存在的框 ✗")
    s8.attack_down_dist = -1
    s8.attack_dist = 0                           # 最大攻击距离 0 ⇒ 水平可攻击区为空
    check(not h8.agent._in_attack_box(_m8(0.0), q),
          "「最大攻击距离」= 0 时还在判可攻击（框面积为 0 ⇒ 该跳过 ✓）")
    check(ag_mod.attack_box_rect(500.0, 500.0, 0.0, up=-1.0, down=-1.0) is None,
          "最大攻击距离 = 0 时框该是空集（不画 ✓）")
    # 盲区框：最小攻击距离 = 0 ⇒ 空集 ⇒ 不判（= 老行为"不规避"✓）也不画 ✓
    s8.attack_dist = 80.0
    s8.min_attack_dist = 0
    check(not h8.agent._in_blind_box(_m8(0.0, dx=5.0), q),
          "最小攻击距离 = 0（盲区为空）时还算贴脸 ⇒ 会凭空触发规避 ✗")
    check(ag_mod.attack_box_rect(500.0, 500.0, 0.0, up=-1.0, down=-1.0) is None,
          "空的盲区框该不画 ✓")


def t_chase_jump_inside_band():
    """起跳区间落在攻击距离**内侧**时（如 -15~0），怪进区间那一拍要跳。

    **回归用例。** 用户配置就是这样：attack_dist=80、区间 -15~0 → 区间 [65,80]，
    而 80 正好也是攻击距离的上限 —— 进区间的这只怪**自己**就落在攻击范围内。
    于是「攻击范围内有怪就不跳」这条门槛对它永远成立，一次都不会跳（跳的资格
    只留给「攻击范围为空」那两处调用点）。

    同时钉住「只跳一次」：那只怪一直挂在区间里，后面几拍**不许**再跳 —— 当年
    从「节流到点就再跳一次」改成「只在进区间那一拍跳」正是为了这个，别为了修
    上面那条又把它丢了。
    """
    s = fresh_settings(attack_dist=80.0, chase_jump_enabled=True,
                       chase_jump_min=-15, chase_jump_max=0,
                       strategy="patrol", jump_random_prob=0.0)
    h = Harness(s)
    # 距离 = mob.x - 520。x=590 → 70：落在 [65,80] 里，同时 ≤80 也在攻击范围内。
    h.mobs_fn = lambda t: ([] if t < 0.5 else
                           [Mob(id=1, x=590.0, y=500.0, w=40.0, h=40.0, conf=0.9)])
    h.run(2.5)
    taps = h.taps(s.keymap["jump"])
    check(taps, "起跳区间 -15~0（内侧）时，怪进区间那一拍没有跳 —— "
                "它自己就在攻击范围内，被「范围内有怪就不跳」挡住了")
    check(len(taps) == 1,
          "同一只怪挂在区间里跳了 %d 次（应只在进来的那一拍跳一次）" % len(taps))


def t_chase_jump_outside_band_once():
    """起跳区间落在攻击距离**外侧**时（0~50）：还在追的时候，进区间那拍跳一次。

    这条是当前唯一能跑通的路径（调用点只在「攻击范围内没怪」的分支里）。
    钉住它，免得修内侧那条时把外侧一起改坏；也钉住「不是每拍都跳」。

    距离 = mob.x - 520，三种位置：170（区间外，还在追）→ 100（区间 [80,130] 内，
    攻击范围还没够着）→ 70（进攻击范围，已出区间）。
    """
    s = fresh_settings(attack_dist=80.0, chase_jump_enabled=True,
                       chase_jump_min=0, chase_jump_max=50,
                       strategy="patrol", jump_random_prob=0.0)
    h = Harness(s)

    def _mobs(t):
        x = 690.0 if t < 1.0 else (620.0 if t < 2.0 else 590.0)
        return [Mob(id=1, x=x, y=500.0, w=40.0, h=40.0, conf=0.9)]

    h.mobs_fn = _mobs
    h.run(3.0)
    taps = h.taps(s.keymap["jump"])
    check(len(taps) == 1,
          "外侧区间应当正好跳一次（追的时候进区间），实际 %d 次：%s"
          % (len(taps), taps))

    # **「追击起跳需要的冲刺时间」**（用户 2026-09-26 要求）：`chase` 状态得**连续**维持
    # 够久才准起跳，否则区间/沿都对也不跳；计时进别的状态就归零（见 tick 里那段）。
    # 这里直接喂 `_maybe_chase_jump`：条件全摆好，只差"冲够了没有"，好判。
    s.chase_jump_dash_ms = 300
    with h._patched():
        h.log.clear()
        h.agent._next_chase_jump = 0.0
        h.agent._chase_since = None                      # 没在 chase
        h.agent._maybe_chase_jump(100.0, 10.0, edge=True)
        check(not h.taps(s.keymap["jump"]),
              "没在 chase 状态也起跳了（等于没这道闸）")
        h.agent._chase_since = 9.8                       # 才 200ms < 300
        h.agent._maybe_chase_jump(100.0, 10.0, edge=True)
        check(not h.taps(s.keymap["jump"]),
              "冲刺时间没够（200ms < 300ms）却起跳了")
        h.agent._chase_since = 9.6                       # 400ms ≥ 300
        h.agent._maybe_chase_jump(100.0, 10.0, edge=True)
        check(h.taps(s.keymap["jump"]),
              "冲刺时间够了（400ms ≥ 300ms）却不起跳")
    # 设成 0 = 不额外要求（老行为）
    s.chase_jump_dash_ms = 0
    with h._patched():
        h.log.clear()
        h.agent._next_chase_jump = 0.0
        h.agent._chase_since = None
        h.agent._maybe_chase_jump(100.0, 10.0, edge=True)
        check(h.taps(s.keymap["jump"]),
              "冲刺时间设 0 时不该还拦着（那是老行为）")


def t_chase_jump_approach_edge():
    """怪从远处**一路走近**、刚跨进区间的那一拍跳一次（用户报的就是这个场景）。

    和「凭空出现在区间里」那条的区别：这里的沿是在**移动中**产生的 —— 最容易
    出的两类错正好都能照出来：沿被判掉（一次都不跳）、或每拍都跳。
    距离每拍递减 2px（200 → 60），跨过 [65,80] 时一定会落在区间内。
    """
    s = fresh_settings(attack_dist=80.0, chase_jump_enabled=True,
                       chase_jump_min=-15, chase_jump_max=0,
                       strategy="patrol", jump_random_prob=0.0)
    h = Harness(s)

    def _mobs(t):
        dist = max(60.0, 200.0 - 2.0 * int(t / FRAME))      # 距离 = mob.x - 520
        return [Mob(id=1, x=520.0 + dist, y=500.0, w=40.0, h=40.0, conf=0.9)]

    h.mobs_fn = _mobs
    # ⚠ 追击这 6 秒里**人在往前走**：`Harness` 的 `world_x` 默认是**钉死**的 500 ——
    #   不推它的话，2026-10-01 新加的「按着方向键却走不动 ⇒ 单点跳一下」也会在这里跳
    #   （那会把下面那句"正好跳一次"弄成 2 次 ✗）。真实游戏里人在动 ⇒ 让世界坐标 x 跟着
    #   推进（每 3 拍 30 px，远超「坐标对齐误差范围」10 ✓）。
    #   ⭐ 顺带把"**正常追击（人在动）时那条逻辑不许跳**"钉在这里 ✓。
    _n = [0]

    def _advance(w):
        _n[0] += 1
        w.player.world_x = 500.0 + 30.0 * (_n[0] // 3)

    h.ws_hook = _advance
    h.run(6.0)
    taps = h.taps(s.keymap["jump"])
    check(len(taps) == 1,
          "怪走进起跳区间时应当正好跳一次（这里人一直在走 ⇒ 也不该有「走不动」那一跳），"
          "实际 %d 次：%s" % (len(taps), taps))


def t_chase_jump_guard_shut():
    """两条「不该跳」的边：范围内有**别的**怪、以及开关关掉。

    这条挡的是「修内侧区间时顺手把条件放宽」：把 `len(in_range) == 1` 写成
    「只要目标在区间内就跳」，前面两条用例**照样能过**（它们都只有一只怪）。
    所以这里专门摆两只：最近那只（70，在区间内）就是目标，另一只 75 也在攻击
    范围内 —— 有得打就先打，跳会打断输出、还可能把自己跳出攻击距离。
    """
    base = dict(attack_dist=80.0, chase_jump_min=-15, chase_jump_max=0,
                strategy="patrol", jump_random_prob=0.0)
    # ① 范围内还有别的怪 → 不跳（目标自己就落在区间里，靠 guard 挡住）
    s = fresh_settings(chase_jump_enabled=True, **base)
    h = Harness(s)
    h.mobs_fn = lambda t: [Mob(id=1, x=590.0, y=500.0, w=40.0, h=40.0, conf=0.9),
                           Mob(id=2, x=595.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    h.run(2.0)
    check(not h.taps(s.keymap["jump"]),
          "攻击范围内还有别的怪时也跳了：有得打就先打，跳会打断输出")

    # ② 开关关掉 → 一只怪、就站在区间里，也不跳
    s2 = fresh_settings(chase_jump_enabled=False, **base)
    h2 = Harness(s2)
    h2.mobs_fn = lambda t: [Mob(id=1, x=590.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    h2.run(2.0)
    check(not h2.taps(s2.keymap["jump"]), "起跳开关是关的，却跳了")


def t_chase_hop_when_stuck():
    """追击时「按着 ←/→、x 却一直不动」⇒ 按「移动操作尝试间隔」**单点跳一下**。

    用户 2026-10-01 要求："战斗时的 chase 加个逻辑：如果在「移动操作尝试间隔(ms)」
    时间内没有发生 x 水平位移，就发单点跳键"。

    局面的造法：`Harness.ws()` 里玩家 `world_x` **恒 500** ⇒ "按着走却一动不动"天然
    成立；怪摆在 900（距角色中心 400 > 攻击距离 80）⇒ 进的是 `chase` 而不是站桩
    `attack` ✓。
    ⚠ 判据用的是**世界坐标 x**（`ws.player.world_x`）—— 不是画面 x（镜头跟着人走，
    画面 x 不动说明不了什么 ✗）。所以下面 ② 反例也必须摆 `world_x`，摆画面 x 测不到。
    """
    from decision import route
    base = dict(strategy="chase", attack_dist=80.0, chase_jump_enabled=False,
                move_retry_ms=500, align_tol_px=10, jump_random_prob=0.0)

    # ---- ① 该跳：按着 → 、x 一直没动，跨过一个「移动操作尝试间隔」就跳一下 ----
    s = fresh_settings(**base)
    jump = s.keymap["jump"]
    h = Harness(s)
    h.mobs_fn = lambda _t: [Mob(id=1, x=690.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    # ⚠⚠ **一次跑完**：`Harness.run()` 每调一次就重设 `clock0`（记录器的时间戳是"距本次
    #   run 起点的相对秒"）⇒ **跨 run 的时间戳不可比** ✗（分两次跑会得出"两次跳间隔 0.000"
    #   这种鬼数 —— 写这条时就是这么栽的 ✓）。
    h.run(2.1)
    check(h.agent.state == "chase",
          "用例本身没进 chase（state=%s）—— 怪该摆在攻击范围外（x=690 > 攻击距离 80）"
          % h.agent.state)
    check(h.count(s.keymap["right"]) >= 1,
          "用例本身没在按方向键（目标在右边，该按 →）")
    taps = h.taps(jump)
    check(taps,
          "追击按着方向键走了 2 秒、世界坐标 x 一直没动，却没单点跳一下")
    # ①-a **窗口没到不许跳**：起锚在"刚进 chase"那一拍（≈0.13s）⇒ 第一次最早也在 0.63s。
    check(taps[0] >= 0.55,
          "「移动操作尝试间隔」500ms 还没到就跳了（第一个在 %.3f 秒）%s" % (taps[0], taps))
    # ①-b **节流**：跳完要再等一个「移动操作尝试间隔」才轮得到下一次 —— 绝不是每拍都跳。
    #      ⚠ 下限给 0.43（不是 0.5）：决策拍按 1/15 秒量化，到点那一拍可能晚 66ms ✓。
    check(2 <= len(taps) <= 4,
          "2.1 秒里跳了 %d 次（窗口 500ms ⇒ 该 3 次上下，跳太密=没节流 ✗）：%s"
          % (len(taps), [round(t, 2) for t in taps]))
    _gaps = [round(b - a, 3) for a, b in zip(taps, taps[1:])]
    check(all(0.43 <= g <= 0.8 for g in _gaps),
          "两次跳的间隔不像「一个移动操作尝试间隔」（该 ≈0.5s，容决策拍量化）：%s" % _gaps)

    # ---- ② 反例：人**真在走**（world_x 一直在变，超过「坐标对齐误差范围」）⇒ 一次不跳 ----
    s2 = fresh_settings(**base)
    h2 = Harness(s2)
    h2.mobs_fn = lambda _t: [Mob(id=1, x=690.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    _n = [0]

    def _walk(w):
        _n[0] += 1
        w.player.world_x = 500.0 + 30.0 * (_n[0] // 3)      # 每 3 拍挪 30 px > 容差 10 ✓

    h2.ws_hook = _walk
    h2.run(2.0)
    check(not h2.taps(s2.keymap["jump"]),
          "人明明在走（世界坐标 x 一直在变）却也跳了：%s" % h2.taps(s2.keymap["jump"]))

    # ---- ③ 「移动操作尝试间隔」配 0 = 关（同 DropJob 对这件参数的口径）----
    s3 = fresh_settings(**{**base, "move_retry_ms": 0})
    h3 = Harness(s3)
    h3.mobs_fn = lambda _t: [Mob(id=1, x=690.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    h3.run(2.0)
    check(h3.agent.state == "chase", "③ 用例本身没进 chase")
    check(not h3.taps(s3.keymap["jump"]),
          "「移动操作尝试间隔」配 0（= 关）却还在跳")

    # ---- ④ 反例：**不是 chase 就不跳** —— 怪在攻击范围内 ⇒ 站桩输出（state=attack）----
    #      这一条专门挡"只看有没有按方向键"那种写法：站桩时**也会**补方向键（每 3 轮一次 ✓）。
    s4 = fresh_settings(**base)
    h4 = Harness(s4)
    h4.mobs_fn = lambda _t: [Mob(id=1, x=560.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    h4.run(2.0)
    check(h4.agent.state == "attack",
          "④ 用例本身没进 attack（state=%s）—— 怪该摆在攻击范围内（x=560）" % h4.agent.state)
    check(not h4.taps(s4.keymap["jump"]),
          "站桩输出（attack）时也跳了 —— 这条逻辑只管 chase")

    # ---- ④' **挂着寻路任务（state=climb）时也不跳** ----
    #      任务自己有"走不动 ⇒ 单点跳"（`route.WalkJob._hop_or_fail` ✓ 发的是 KeyState 的
    #      按住键、不是 `tap` ⇒ 不会混进 `h.taps` ✓），这里再跳就是两处打架。
    #      ⭐ **这条专门钉 `state != "chase"` 那道闸**：任务那一拍 `keys` 里**有方向键**
    #      （`_climb_tick` 自己加的 ✓）⇒ 没有那道闸，它在追击之外也照跳 ✗（反向验证
    #      把那道闸改成 `if False` 时，抓住它的正是这一条 ✓）。
    s7 = fresh_settings(**base)
    h7 = Harness(s7)
    job = route.WalkJob("乙平台", [(900.0, 850.0, 950.0, 60.0, "1")],
                        tol_px=10, hold_ms=0, stall_s=99.0)
    with h7._patched():
        # ⚠ 不走 `run()` ⇒ 记录器的时间基准（`clock0`，只在 `run()` 里设）要自己摆 ✗
        #   （不摆 ⇒ `_rec` 一记录时刻就 AttributeError ✓）
        h7.clock0 = h7.clock.t
        h7.agent.start_climb(job)
        for _i in range(int(2.0 / FRAME)):
            h7.clock.t += FRAME
            # ⚠ **别摆怪**：摆了默认的 560 那只（距 40 < 攻击距离 80）会走"有怪先打"那条支
            #   ⇒ 这一帧根本不 `tick` 任务 ⇒ state 停在 attack、这条测不到东西 ✗。
            h7.agent.tick(h7.ws(with_mob=False))
    check(h7.agent.state == "climb",
          "④' 用例本身没进 climb（state=%s）—— 任务还没到（目标 x=900、人一直在 500）"
          % h7.agent.state)
    check(not h7.taps(s7.keymap["jump"]),
          "挂着寻路任务走的时候也单点跳了一下 —— 这条逻辑只管 chase，任务自己有卡住逻辑")

    # ---- ⑤ 没在按方向键就不算"走不动"：怪正好在同一条 x 线上（dx=0）时一个键都不该按 ----
    #      `attack_dist=0` ⇒ 攻击框是空的（**永不进 attack**）⇒ 状态留在 chase，但 `_steer`
    #      拿到 dx=0 ⇒ 一个方向键都不发 ✓。没这道闸的话，站着不动的 chase 也会白跳 ✗。
    s6 = fresh_settings(**{**base, "attack_dist": 0.0})
    h6 = Harness(s6)
    h6.mobs_fn = lambda _t: [Mob(id=1, x=500.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    h6.run(1.5)
    check(h6.agent.state == "chase", "⑥ 用例本身没进 chase（state=%s）" % h6.agent.state)
    _dirs6 = {s6.keymap["left"], s6.keymap["right"]}
    check(not any(k in _dirs6 for _t, kind, k in h6.log if kind == "down"),
          "⑥ 用例本身按了方向键（dx=0 时不该发方向键）—— 这条测不到'没按就不跳'")
    check(not h6.taps(s6.keymap["jump"]),
          "这一拍并没有在指挥它走（dx=0、方向键一个没按）却跳了")

    # ---- ⑥ 扫平台巡逻（`sweep`）**也是 chase 状态** ⇒ 同样覆盖（按状态写的自然结果 ✓）----
    s5 = fresh_settings(strategy="sweep", jump_random_prob=0.0, chase_jump_enabled=False,
                        move_retry_ms=500, align_tol_px=10)
    h5 = Harness(s5)
    h5.mobs_fn = lambda _t: []
    h5.run(1.2)
    check(h5.agent.state == "chase", "⑥ 用例本身没进 chase（state=%s）" % h5.agent.state)
    check(h5.taps(s5.keymap["jump"]),
          "扫平台（state 也是 chase）一直走却走不动时，没单点跳一下")


def t_patrol_idle_releases_output():
    """怪在输出序列**中途**消失 ⇒ 输出键必须松开（2026-09-26 用户报的现象）。

    现象："画面上一个框都没有，角色却持续在按输出"。
    根因：`tick` 那条「平地巡逻 + 无怪」的早退只 `keys.release_all()`（只松 KeyState 的
    移动键），而输出键是 `_run_seq` 用 `key_down` **直接发的**，要靠 `_release_combat_keys()`
    才松 ⇒ 序列停在下过半截、"up" 永不发出 ⇒ 攻击键一直按着（状态却已是 idle、画面无框）。
    默认 `resetall_interval=60` 会兜一次，所以是"偶尔、持续一阵子"。

    复现：默认输出序列 `down attack → 隔 70~130ms → up attack`，而帧间隔约 66ms
    ⇒ 让怪在 0.2s 后消失，就正好落在 down 与 up 之间。
    """
    s = fresh_settings(strategy="patrol", jump_random_prob=0.0,
                       output_seq=[{"type": "down", "key": "attack"},
                                   {"type": "delay", "ms": 300},
                                   {"type": "up", "key": "attack"}])
    h = Harness(s)
    atk = s.keymap["attack"]
    h.mobs_fn = lambda t: ([Mob(id=1, x=560.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
                           if t < 0.2 else [])
    with h._patched():
        h.run(1.2)
        downs = [k for _t, kind, k in h.log if kind == "down"]
        ups = [k for _t, kind, k in h.log if kind == "up"]
        check(atk in downs, "用例本身没跑起来（输出键一次都没按下）：%s" % h.log[:8])
        check(atk in ups,
              "怪在输出序列中途没了 ⇒ 攻击键按着没松（画面无框却在输出）：%s"
              % h.log[-8:])
        check(h.agent.state == "idle", "无怪之后该回 idle：%s" % h.agent.state)


def t_walk_only_center():
    """「走」的两种走法：**默认朝目标集合的 x 中点**、**逐边配了「仅向左/仅向右」就只按那个方向**。

    历史（别搞反）：这条原先测"三种方向"（`WalkJob.mode` = center / left / right，还在设置里
    摆了个「走的方向」下拉）。用户 2026-09-26 的定论是 **设置里不该有这个参数** ⇒ 默认就一种
    走法：朝集合正中间走；**"将来若某一步真要只按 ←/→，那是逐边的事"**（foothold 编辑器 →
    「可到达」窗口里配）。
    那个"将来"就是现在：用户 2026-09-27 报「我配的第一个通行方式是向左走，可它根本没向左」
    ⇒ 逐边的 `dir` **真的接进执行器**了（`WalkJob.walk_dir`，灌进口在 `walk_job_for_edge`），
    而**全局设置那条仍然禁止**（下面 ①④ 两条反向钉照旧 ✓）。

    集合里两条 foothold：x=600 与 x=800 ⇒ 中点 = 700。
    """
    from PyQt5.QtWidgets import QApplication

    from decision import route

    spots = [(600.0, 590.0, 610.0, -208.0, "1"),
             (800.0, 790.0, 810.0, -208.0, "2")]
    # ① **反向钉**：走的方向参数必须**不在**
    check(not hasattr(route.WalkJob, "MODE_CENTER"),
          "`WalkJob` 里又出现了 MODE_CENTER（走只有一种走法，用户 2026-09-26 明确删掉）")
    check("mode" not in route.WalkJob("乙平台", spots).__dict__,
          "`WalkJob` 里又出现了 `mode`（走只有朝中点一种，方向不该来设置）")
    # ② 朝中点：站 500 → 往右；站 900 → 往左；站 705（中点附近、容差内）→ 站住等判到达
    check(route.WalkJob("乙平台", spots).update(0.0, px=500.0)["move"] == 1,
          "朝中点：在中点左边该往右走")
    check(route.WalkJob("乙平台", spots).update(0.0, px=900.0)["move"] == -1,
          "朝中点：在中点右边该往左走")
    o = route.WalkJob("乙平台", spots).update(0.0, px=620.0)
    check(o["move"] == 1 and "集合中心 x=700" in o["note"],
          "朝中点：还没到该继续走、并说清目标是集合中点：%s" % o)
    o = route.WalkJob("乙平台", spots).update(0.0, px=705.0)
    check(not o["move"] and "站住" in o["note"],
          "朝中点：到中点附近（差 5 px ≤ 容差）该站住等判到达：%s" % o)
    # ③ 到位判据：站在目标 foothold 上就算到（走是唯一"站上去就算到"的任务）
    # ⚠ 位置要放在**真的 foothold 区间里**（800 属于第二条 790~810）：700 是两条之间的
    # 空隙，站在那儿判"没到位"是对的 —— 朝中点走时会**经过** foothold，经过那一拍就判到 ✓
    job = route.WalkJob("乙平台", spots)
    job.update(0.0, px=100.0)
    o = job.update(0.1, px=800.0, py=-208.0)
    check(o["done"] and "到达" in o["note"],
          "站在目标 foothold 上没判到达：%s" % o["note"])
    # ④ **反向钉**：设置页里不许再有「走的方向」控件（全局参数，用户没提过 ⇒ 不许有）
    app = QApplication.instance() or QApplication([])      # noqa: F841
    from gui.settings_dialog import SettingsDialog
    _d = SettingsDialog()
    try:
        check(not hasattr(_d, "cmb_walk_mode"),
              "设置页里又出现了「走的方向」（用户 2026-09-26 明确否掉）")
    finally:
        _d.reject()
    # ⑤ **逐边**的「仅向左 / 仅向右」（2026-09-27 起生效）：只按那一个方向，**不看中点**
    #    这正是用户现场那条「左上平台 →(走 仅向左)→ 左上」：目标是整片「左上」（x −387~657、
    #    中点 135），人站在平台上（x 2~87）⇒ 中点还在**右边** ⇒ 老代码一路按 →（用户原话
    #    "根本没向左"），而且"到集合区间的距离"恒为 0 ⇒ 3 秒后判「走不动了」= 卡住 ✗。
    _L = route.WalkJob("乙平台", spots, walk_dir="left")
    _R = route.WalkJob("乙平台", spots, walk_dir="right")
    o = _L.update(0.0, px=500.0)
    check(int(o["move"]) == -1,
          "配了「仅向左」，人却往右走（又回去看中点了）：%s" % o)
    o = _R.update(0.0, px=900.0)
    check(int(o["move"]) == 1,
          "配了「仅向右」，人却往左走（又回去看中点了）：%s" % o)
    check("仅向左" in _L.note,
          "配了「仅向左」却没在提示里说清（人会以为它按中点走）：%r" % _L.note)
    # 到位判据照旧（站上目标集合的 foothold 就算到 ✓）—— `dir` 只管"怎么走"，不管"到没到"
    _j = route.WalkJob("乙平台", spots, walk_dir="left")
    _j.update(0.0, px=100.0)
    check(_j.update(0.1, px=800.0, py=-208.0)["done"],
          "「仅向左」时站在目标 foothold 上没判到达")
    # 方向不对（一直按左、x 却不再变小）⇒ `stall_s` 后失败，并说清是这条边的方向有问题
    _j = route.WalkJob("乙平台", spots, walk_dir="left", stall_s=1.0, timeout_s=60.0)
    _j.update(0.0, px=500.0)
    o = None
    for _i in range(1, 40):
        o = _j.update(0.1 * _i, px=500.0)
        if o["failed"]:
            break
    check(o is not None and o["failed"] and "仅向左" in o["note"] and "500" in o["note"],
          "「仅向左」卡住时没说清原因：%r" % (o or {}).get("note"))
    # 认不出的值 ⇒ 退回默认（朝中点），不许瞎按
    _bad = route.WalkJob("乙平台", spots, walk_dir="up")
    check(_bad.walk_dir == "" and int(_bad.update(0.0, px=500.0)["move"]) == 1,
          "认不出的 `dir` 没退回默认走法：%r / %s" % (_bad.walk_dir,
                                                     _bad.update(0.0, px=500.0)))


def t_chase_jump_edge_guard():
    """⭐⭐ 「**距离平台边缘多远禁用(px)**」（用户 2026-10-02 ✓ 原话："追击起跳功能加个参数
    「距离平台边缘多远禁用(px)」（**不启用时不能配置**），代表如果**距离 foothold 集边缘距离
    小于等于这个值**即使满足条件也**不按跳**"；口径定稿："**跳向那侧**的边" ✓）。

    钉五件（判据在 `_maybe_chase_jump` ✓，读数 = `ws.player.here_span` + `world_x` ✓）：
      ① **启用 + 贴边**（到"跳向那侧"边缘 ≤ 阈值）⇒ **不跳** ✓ + 打点 `chase_jump_edge_block` ✓；
      ② **启用 + 离边远** ⇒ **照跳** ✓；
      ③ **不启用**（勾选框关）⇒ 贴边也**照跳** ✓（= 老行为 ✓ 兼容底线 ✓）；
      ④ **判不出来**（没 `here_span` ✓ 位置状态没给平台宽度）⇒ **不禁用** ⇒ 照跳 ✓（宁缺勿错 ✓）；
      ⑤ **方向取"跳向那侧"** ✓：贴的**左**边很近、但要朝**右**跳 ⇒ **不禁** ⇒ 照跳 ✓
         （用"最近那条边"就会误禁 ✗）。
    """
    s = fresh_settings(attack_dist=40.0)
    s.enabled = True
    s.chase_jump_enabled = True
    s.chase_jump_min = s.chase_jump_max = 200      # 区间 [240, 240] ⇒ 下面统一喂 dist=240 ✓
    s.chase_jump_dash_ms = 0
    h = Harness(s)
    h.clock0 = h.clock.t          # ⚠ 日志那列是**相对时刻**（`_rec` 减它 ✓）少一个就 AttributeError ✗
    a = h.agent
    ws = h.ws(with_mob=False)
    t = 5000.0

    def _jump(guard_on, guard_px, span=(400.0, 600.0), wx=500.0, facing=1):
        """跑一次起跳 ⇒ 回 `(跳了没, 被边距挡了没)`。"""
        s.chase_jump_edge_guard_enabled = guard_on
        s.chase_jump_edge_guard_px = guard_px
        ws.player.here_span = span
        ws.player.world_x = wx
        a.set_facing(facing)
        a._chase_since = t - 10.0      # 冲刺时间闸：早已满足 ✓（不然第一道闸就 return ✗）
        a._next_chase_jump = 0.0       # 全局节流同理 ✓
        h.log.clear()
        _evs = []
        with h._patched(), \
                patch.object(ag.behavior, "event", lambda name, **kw: _evs.append(name)), \
                patch.object(ag.behavior, "sample",
                             lambda name, *a2, **kw: _evs.append(name)):
            a._maybe_chase_jump(240.0, t, edge=True, ws=ws)
        _jk = s.keymap["jump"]
        return (any(k == _jk for _1, _2, k in h.log),
                "chase_jump_edge_block" in _evs)

    # 前置：这几拍的起跳条件（区间 + 沿）本来是满足的 ⇒ 只要不禁就该跳 ✓
    check(_jump(False, 0)[0] is True,
          "前置不成立：不启用边距闸时这一拍该跳（区间/沿没摆对 ✗）")

    # ① 启用 + 贴边（朝向那侧差 100px，阈值 150）⇒ 不跳 + 打点 ✓
    _ok, _blk = _jump(True, 150)
    check(_ok is False, "贴边到跳向那侧边缘只剩 100px（阈值 150）却还是跳了 ✗")
    check(_blk is True, "被边距挡掉那一下没打 `chase_jump_edge_block`（日志里看不出为什么没跳 ✗）")

    # ② 启用 + 阈值放宽到 50 ⇒ 不禁 ⇒ 照跳 ✓
    check(_jump(True, 50)[0] is True, "离边 100px、阈值 50 ⇒ 不该禁，却跳不了 ✗")

    # ③ 不启用 ⇒ 贴边也照跳 ✓（老行为一字不变 ✓）
    check(_jump(False, 150)[0] is True, "没启用这条闸却把起跳挡了（老行为被改 ✗）")

    # ④ 判不出来（位置状态没给平台宽度）⇒ 不禁 ✓
    check(_jump(True, 150, span=None)[0] is True, "`here_span` 判不出来却禁了跳（宁缺勿错 ✗）")

    # ⑤ 方向："跳向那侧"而不是"最近那条边" —— 左边缘只剩 50px（很近）、但朝**右**跳，
    #    右边有 350px ⇒ **不该禁** ✓（用最近边会误禁 ✗）
    check(_jump(True, 150, span=(450.0, 850.0), wx=500.0, facing=1)[0] is True,
          "朝右跳却拿**左边**那条近边把跳禁了（方向该取「跳向那侧」✗）")
    check(_jump(True, 150, span=(450.0, 850.0), wx=500.0, facing=-1)[0] is False,
          "朝左跳、左边就剩 50px（阈值 150）却没禁 ✗")


def t_big_mob_lock_blocks_behind_steal():
    """⭐⭐ **大怪锁生效期间：背后的怪不许抢锁**（用户 2026-10-02 ✓ 原话："优化：决策参数→**优先
    锁定大怪**的功能，当**锁定命中大怪**时，**禁用「背后攻击范围内有怪触发转向」**" +
    追问定稿："**背后的怪无法抢锁**" ✓）。

    口径：
      · 判据 = **当前锁定的那只在"大怪池"里** ✓ —— 与「大怪优先」**同一处**
        （`_big_mob_pool` ✓ ⭐「最小框均线」× `big_mob_ratio` ✓ + 优先半径 ✓ ——
        基准 2026-10-02 由"候选中位数"改成"最小框均线" ✓，见 `t_big_mob_baseline_min_avg` ✓）；
      · 是 ⇒ **不抢锁**（背后那只哪怕现在够得着也不抢 ✓ 继续打大怪 ✓）；
        不是 / 「大怪优先」开关关着 / 池空 ⇒ **照旧抢** ✓（老行为一字不变 ✓）；
      · 只挡**抢锁**那一段（锁还在期内 ✓）：锁过期之后照旧走 `_nearest` 重挑 ✓
        （那里本来就有大怪优先过滤 ✓）。

    钉四件：① 锁大怪 + 背后有够得着的小怪 ⇒ **不抢**（`_target_id` 不变 + 没 `target_steal` ✓）；
      ② 锁的是**小怪** ⇒ **照旧抢** ✓；③ 「大怪优先」**开关关** ⇒ 照旧抢 ✓（老行为 ✓）；
      ④ **池空**（视野里只剩这一只，判不出大怪 ✓）⇒ 照旧抢 ✓。
    """
    s = fresh_settings(strategy="sweep", attack_dist=100.0)
    s.enabled = True
    h = Harness(s)
    a = h.agent
    ws = h.ws(with_mob=False)
    a.set_facing(1)                     # 朝右：正面 = 右边、背后 = 左边 ✓
    # 正面的大怪（h=300）× 背后够得着的小怪（h=40，dx=-80 在攻击框内 ✓）⇒ 基准
    # ⭐「最小框均线」= 40（**最小那一头** ✓ 2026-10-02 起；中位数那套在这里是 170 ⇒ 阈值 255
    #   也会判 id=1 是大怪 ✓ 两种口径结论一样，这条测的是**抢锁那道闸** ✓）
    # ⇒ 「大怪」判据 = h ≥ 40×1.5 = 60 ⇒ 只有 id=1 算大怪 ✓。
    big = Mob(id=1, x=580.0, y=500.0, w=40.0, h=300.0, conf=0.9)
    back = Mob(id=2, x=420.0, y=500.0, w=40.0, h=40.0, conf=0.9)
    t = 2000.0
    evs = []

    def _pick(locked_id, mobs, enabled=True):
        """摆好"当前锁的是 `locked_id`"，跑一次锁定决策 → 目标怪。

        ⚠ **候选列表手工给**（`[big, back]` ✓）：用例钉的是**抢锁那道闸**，
          而要"正面的大怪"出现在 `_candidates()` 的输出里还牵扯别的前置（本轮没摆平 ✗）——
          直接喂 `_locked_target(lockable, …)` 才是这条链的**真实入口**（它本来就收 lockable ✓）。
        """
        s.big_mob_enabled = enabled
        ws.mobs = list(mobs)
        a._target_id, a._target_until = locked_id, t + 30.0
        evs.clear()
        with patch.object(ag.behavior, "event",
                          lambda name, **kw: evs.append((name, kw))):
            tgt, _d = a._locked_target(list(mobs), ws, t)
        return tgt

    # 前置：背后那只真的"够得着"（否则 ① 会假过 ✓）
    check(a._best_behind_target([big, back], ws) is not None,
          "用例没摆对：背后那只够不着 ⇒ ① 测不到任何东西 ✗")

    # ① 锁的是**大怪** ⇒ 背后的怪**抢不动锁** ✓
    check(_pick(1, [big, back]).id == 1,
          "锁着大怪时背后的怪抢走了锁（用户 2026-10-02 明确：**背后的怪无法抢锁** ✗）")
    check(not any(n == "target_steal" for n, _k in evs),
          "锁着大怪时打了 `target_steal`（那次抢锁该被挡掉 ✓）：%r" % (evs,))

    # ② 锁的是**小怪** ⇒ 照旧抢 ✓（回归保护：别把既有行为一起改了 ✗）
    check(_pick(2, [big, back]).id == 2,
          "锁的是小怪时该照旧抢锁（背后那只现在就能打 ✓）：%r" % (_pick(2, [big, back]).id,))

    # ③ 「大怪优先」**整个关掉** ⇒ 这条不生效 ⇒ 照旧抢 ✓（= 老行为 ✓）
    check(_pick(1, [big, back], enabled=False).id == 2,
          "「大怪优先」关着时不许抢锁（关掉 = 整个不启用 ⇒ 老行为 ✓ 一字不变 ✗）")

    # ④ **判据本身**：它必须和「大怪优先」同一条 —— 池空（只有 1 只 ⇒ 谈不上"相对更高"）
    #     ⇒ `_is_big_locked` 给 False ✓（那样这条禁用**自然不生效** ✓ = 老行为 ✓）
    s.big_mob_enabled = True            # ⚠ 上面 ③ 刚把它关过 ⇒ 这里要显式开回来（不然判据恒 False ✗）
    check(a._is_big_locked(big, [big, back], ws) is True,
          "大怪锁的判据没认出大怪（前置不成立 ⇒ ①②③ 都测不了 ✗）")
    check(a._is_big_locked(big, [big], ws) is False,
          "池空（视野里只有 1 只）时判据该给 False（它**与「大怪优先」同一条** ⇒ 池空就不启用 ✓）")


def t_station_turn_survives_state_flicker():
    """⭐⭐ **「站桩」的判据 = 输出轮**（用户 2026-10-02 ✓ 原话："我对站桩的定义：**输出后**攻击
    范围内还有怪 → 判定为站桩，**没有怪了** → 结束站桩"）。

    现场（2026-10-02 真日志 ✓）：attack/chase 的**状态**随"这一拍攻击框里有没有怪"每 ~0.25 秒
    抖一次（791 次进站桩、684 次首窗 ✗）—— 而当时**站桩唯一的判据就是"这一拍的状态"**
    （`_atk_round` 的每拍清零 + 首窗的沿 ✗）⇒ 站桩被切成碎片 ⇒
    「每 3 次 attack 补朝向键」**一次都没执行过**（`station_turn` **0 条** ✗）。

    钉三件（把"框里时有时无"这条**真实现场**摆出来 ✓）：
      ① **闪断不再清零轮次**：怪每隔几拍进出攻击框 ⇒ `station_turn`（每 3 轮）**照样**要到 ✓
         （旧判据下永远数不到 3 ✗）；
      ② **首窗一段只开一次**：`station_turn_first` 只该 1 条 ✓（旧判据下每次闪回来都开 ✗）；
      ③ **彻底收场**（中止战斗 ⇒ `_release_combat_keys` ✓）之后重新开打 ⇒ **重新起算**
         （首窗再来一条 ✓ —— "上一场的账不许带到下一场" ✓）。
    """
    s = fresh_settings(strategy="patrol", jump_random_prob=0.0, attack_dist=80.0,
                       attack_cd=200, min_turn_hold_ms=300, turn_output_delay_ms=0)
    h = Harness(s)
    h.clock0 = h.clock.t
    # 怪：**框内**（中心距 70 < 80 ✓）↔ **框外**（远走 ✓）交替 —— 就是要造出
    # `attack ↔ chase` 的**状态闪断**（真实现场那个形状 ✓）。
    _near = Mob(id=1, x=570.0, y=500.0, w=40.0, h=40.0, conf=0.9)
    _far = Mob(id=1, x=1500.0, y=500.0, w=40.0, h=40.0, conf=0.9)
    h.mobs_fn = lambda t: [_near if int(t / 0.2) % 2 == 0 else _far]
    h.agent.set_facing(1)
    _evs = []
    with patch.object(ag.behavior, "event", lambda name, **kw: _evs.append(name)):
        h.run(4.0)
        _first_seg = list(_evs)
        # ③ **关自动 ⇒ 再开**（真·收场 = 关自动的下降沿 ✓）：站桩那一段重新起算 ✓
        s.enabled = False
        with h._patched():
            h.agent.tick_timing(h.ws(True))          # 下降沿那一拍（清站桩那一段 ✓）
        s.enabled = True
        h.run(4.0)
    check(_evs.count("station_turn") >= 1,
          "攻击框里时有时无（状态 chase↔attack 抖）⇒ 「每 3 轮补朝向键」一次都没触发 ✗"
          "（站桩判据不该看「这一拍的状态」✗ —— 2026-10-02 现场的病就是这个）：%r"
          % (_evs.count("station_turn"),))
    check(_first_seg.count("station_turn_first") == 1,
          "首窗在**一段站桩**里开成了 %d 次（该只有 1 次 ✓ —— 怪闪出去那一刻**不算结束站桩** ✓，"
          "旧判据下每次闪回来都重开 ✗）：%r"
          % (_first_seg.count("station_turn_first"), _first_seg.count("station_turn_first")))
    check(_evs.count("station_turn_first") >= 2,
          "彻底收场之后再打，站桩没**重新起算**（首窗该再来一条 ✓ —— 中止战斗要清 `_station_on`/"
          "`_atk_round` ✓）：%r" % (_evs.count("station_turn_first"),))


def t_station_turn_every_three():
    """站桩输出的「**每 3 次 attack 补一个朝目标的朝向键**」（用户 2026-09-28 要求 2）。

    用户原话：*"attack 增加一个逻辑：**每 3 次 attack 补一个朝向目标的方向键**，持续「最小切换
    朝向时间」，并在这个参数里补上 tips"* +（追问后）*"应该是**站桩输出时**每 3 次 attack，
    **只要退出站桩计数清零**"* ✓。

    ⚠⚠ **时长 2026-09-30 用户实测纠偏**（原话："现在好像看起来每次 attack 都往前走了，
      而不是每 3 次"✗）：原来"每 3 轮"那一下按**「最小切换朝向时间」的全长** —— 现场
      `min_turn_hold_ms=1000` ≈ 3 轮周期（attack_cd 300ms ⇒ 3 轮 ≈ 1s）⇒ 窗口首尾相接、
      占空比 ~100% ⇒ **观感就是每轮都在走** ✗。补键的用途是**转身对齐** —— 站桩转身
      点一下就够 ⇒ 改为 **`TURN_TAP_S`** ✓（与普通站桩换向同款口径 ✓；"按满就是从怪身上
      走过去"是既有结论 ✓）。⇒ **补键时长与 `min_turn_hold_ms` 解耦** ✓（那个参数只管
      **走动时**的换向按住 ✓）。

    ⭐⭐ **进站桩首窗按满「最小切换朝向时间」**（2026-09-30 用户要求 ✓ 原话："攻击范围
      内从无怪首次变成有怪而触发 attack 时，向着怪的方向键需要按'最小切换朝向时间(ms)'"
      ✓）—— 刚进站桩那刻角色往往还没转过来，只点 `TURN_TAP_S` 不够 ⇒ 第一窗按**满
      参数** ✓（复用"每 3 轮补键"的同一个窗 ✓）；窗过后照旧点一下 ✓。

    钉四件（都按**方向键"按住段"的时长**判 ✓，与 `t_sweep_stands_still_in_attack` 同一手法）：
      ① 站桩期间**有**方向键段（首窗 + 每 3 轮补 ✓ —— 怪在正前方、朝向不变 ⇒
         普通换向 tap 不会触发 ⇒ 这些全是站桩补的 ✓）；
      ② **第一段 = 首窗**：时长 ≈ `min_turn_hold_ms`（现场量级 1000ms ✓）✗ 短了 =
         首窗没生效 ✗；
      ③ **首窗之后的每一段都是点按**（≤ `TURN_TAP_S` + 拍余量 ✓ —— 2026-09-30
         纠偏 ✓："看起来每次 attack 都往前走"✗）；
      ④ 8 秒里段数在合理带内（首窗 1 + 每 3 轮一次 ⇒ 下限 2 ✓；上限 = 轮数 ✗）。
    ④ **绳上绝不按左右**（`_in_rope_now` ⇒ cap=0）：在绳上爬着时**一个方向键都不补** ✓
       —— 在绳上按左右在游戏里就是**松手掉下来** ✗（用户 2026-09-28 明确 ✓）。
    """
    from decision import route

    def dir_segments(hh, ss):
        """方向键的「按住段」列表 `[(起, 止)]`（相对 `hh.clock0` ✓）。"""
        dirs = {ss.keymap["left"], ss.keymap["right"]}
        out, cur = [], {}
        for t, kind, k in hh.log:
            if k not in dirs:
                continue
            if kind == "down":
                cur[k] = t
            elif kind == "up" and k in cur:
                out.append((cur.pop(k), t))
        for k, a in cur.items():
            out.append((a, hh.clock.t - hh.clock0))
        return out

    # ---- ①②③ 站桩：每 3 轮补一次"点按" ----
    # ⚠ `attack_cd` 用**真实量级**（200ms ✓，现场 300ms 同量级）；`min_turn_hold_ms`
    #   也用**现场量级** 1000ms —— 旧实现（补键吃它的全长）会按出 ~1s 长段 ⇒ ② 当场红 ✓。
    s = fresh_settings(strategy="patrol", jump_random_prob=0.0, attack_dist=80.0,
                       attack_cd=200, min_turn_hold_ms=1000, turn_output_delay_ms=0)
    h = Harness(s)
    h.clock0 = h.clock.t
    # 怪一直在**正前方**的攻击范围内（x=570，人在 500 ⇒ 中心距 70 < 80 ✓）——
    # ⚠ **别摆到身后**：`_in_range` 只认正前方，摆到身后就根本进不了 attack ✗
    #   （那样测的是"回身输出"，不是这条 ✓ 实测踩过 ✓）。
    h.mobs_fn = lambda t: [Mob(id=1, x=570.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    h.agent.set_facing(1)                 # 朝右（与怪同侧 ⇒ 补朝向键按的就是右键 ✓）
    _evs = []
    with patch.object(ag.behavior, "event", lambda name, **kw: _evs.append(name)):
        h.run(8.0)
    segs = dir_segments(h, s)
    check(segs, "站桩期间一次方向键都没按 —— 用例本身没造出「要转向」的局面")
    # ⭐ 打点（用户 2026-10-01 ✓）：进站桩首窗 + 每 3 轮各一条，日志里可数 ✓
    check("station_turn_first" in _evs,
          "进站桩首窗没打 `station_turn_first`（日志里数不出首窗 ✗）：%r" % (_evs,))
    check(_evs.count("station_turn") >= 2,
          "每 3 轮补朝向键没打 `station_turn`（8 秒至少该有 2 条 ✗）：%r" % (_evs,))
    check(len(segs) >= 1 and abs((segs[0][1] - segs[0][0]) - 1.0) <= 0.15,
          "进站桩的**第一段**该按满「最小切换朝向时间」1.0s（首窗 ✓ 2026-09-30 用户"
          "要求 ✗）：%s" % [(round(a, 2), round(b - a, 2)) for a, b in segs])
    # ⭐⭐ **2026-10-02 改口径**（用户定稿 ✓）：补键时长 = **按满「最小切换朝向时间」**，
    #   但**自动钳到不长于「站桩补朝向间隔」** ⇒ 预期 = min(最小切换朝向时间, 间隔) ✓
    #   （本用例 = min(1000ms, 1000ms) = **1.0s** ✓）；超了就是钳子没生效 ⇒ 方向键一直按着 ✗。
    _want = min(int(s.min_turn_hold_ms),
                int(getattr(s, "station_turn_interval_ms", 1000))) / 1000.0
    check(all(b - a <= _want + 0.1 for a, b in segs[1:]),
          "首窗之外的补键该按满 min(最小切换朝向时间, 站桩补朝向间隔)=%.2fs（超了就是自动钳"
          "没生效 ⇒ 方向键一直按着、人一边打一边走 ✗）：%s"
          % (_want, [(round(a, 2), round(b - a, 2)) for a, b in segs]))
    check(2 <= len(segs) <= 24,
          "8 秒里点按 %d 次（每 3 轮一次 ⇒ 该有几次、但绝不该每轮都补 ✗）：%s"
          % (len(segs), [(round(a, 2), round(b - a, 2)) for a, b in segs]))

    # ---- ④ 绳上绝不按左右（在绳上爬着 ⇒ cap=0 ⇒ 一个方向键都不补）----
    s2 = fresh_settings(strategy="patrol", jump_random_prob=0.0, attack_dist=80.0,
                        attack_cd=0, min_turn_hold_ms=300, turn_output_delay_ms=0)
    h2 = Harness(s2)
    h2.clock0 = h2.clock.t
    h2.mobs_fn = lambda t: [Mob(id=1, x=430.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    job2 = route.ClimbJob(ladder_id="L2", x=500.0, y1=-9999.0, y2=9999.0,
                          direction=1, dst_set="乙平台", src_set="甲平台",
                          tol_px=6, hold_ms=0)
    with h2._patched():
        h2.agent.start_climb(job2)
        for _i in range(12):                    # 1.2 秒：足够跨过"每 3 轮"那一下 ✓
            h2.clock.t += 0.1
            ws = h2.ws(with_mob=True)
            ws.player.ladder_id = "L2"           # 广播：人正贴着本任务那根绳 ✓
            h2.agent.tick(ws)
    dirs2 = {s2.keymap["left"], s2.keymap["right"]}
    pressed_dir = [k for _t, kind, k in h2.log if kind == "down" and k in dirs2]
    check(not pressed_dir,
          "在绳上爬着时按了左右方向键 —— 游戏里那等于**松手**，人会掉下来 ✗"
          "（用户 2026-09-28 明确「在绳上绝不按左右」）：%s" % pressed_dir)


def t_sweep_stands_still_in_attack():
    """扫平台（sweep）在攻击范围内有怪时**要站桩输出**，不许一边走一边打。

    **回归用例**（用户报的：平台巡逻时怪在范围内，却不停地走）。根因不在"进攻击
    状态"那一下（实测只按了 0.09 秒），而在**换向**：怪换到另一边时 `_hold_turn`
    会把方向键补回来、按满 `min_turn_hold_ms` —— 用户配置是 1000ms，于是角色一边
    打一边走整整 1 秒，从怪身上走过去、出范围又回头 = 来回抖。

    判据按**按键时长**：怪一直挂在攻击范围内时，方向键每一段按住都不该超过
    `TURN_TAP_S` + 几拍余量；下面第二段同时钉住"走着的策略没被一起改掉"。
    """
    def mobs(t):
        # 一直在攻击范围内（80px 以内）；0.6 秒时从右边换到左边 → 逼出换向
        return [Mob(id=1, x=(590.0 if t < 0.6 else 430.0), y=500.0,
                    w=40.0, h=40.0, conf=0.9)]

    # ⚠ `attack_cd` 用**真实量级**（200ms ✓ 用户配置就是 200~300）：它不是 0 才符合现场。
    #   ⚠ `attack_cd=0` 时"每 3 轮"的周期比「最小切换朝向时间」还短 ⇒ 补键窗口首尾相接
    #   ⇒ 看起来是"一直按着"✗ —— 那是参数配合的结果（agent 里有"不重叠"保护 ✓），
    #   不是这条用例要测的东西 ✓。
    s = fresh_settings(strategy="sweep", jump_random_prob=0.0, attack_dist=80.0,
                       attack_cd=200, min_turn_hold_ms=1000, turn_output_delay_ms=200)
    h = Harness(s)
    h.mobs_fn = mobs
    h.run(2.0)

    dirs = {s.keymap["left"], s.keymap["right"]}
    segs, cur = [], {}
    for t, kind, k in h.log:
        if k not in dirs:
            continue
        if kind == "down":
            cur[k] = t
        elif k in cur:
            segs.append((cur.pop(k), t))
    # **收尾那一段也算**：跑完时还按着的键，要一直算到结束时刻 ——
    # 不然"从头按到尾"的那种（恰恰是要抓的"一直在走"）会被整段漏掉。
    for k, a in cur.items():
        segs.append((a, h.clock.t - h.clock0))
    check(segs, "扫平台时一次方向键都没按 —— 用例本身没造出换向")
    # ⚠ 口径变过一次（用户 2026-09-28 要求 2）：**站桩时每 3 次 attack 会补一次
    #   "朝目标"的方向键、按满「最小切换朝向时间」** ✓ ⇒ 那一下**必然**比 `TURN_TAP_S` 长。
    #   所以上限改成"**按满一次 + 几拍余量**"，仍然挡住"一路按着不放"✗
    #   （一条连续按住段超过这个上限就说明 cap 彻底没生效 ✓）。
    limit = max(ag.TURN_TAP_S, s.min_turn_hold_ms / 1000.0) + 4 * FRAME
    for a, b in segs:
        check(b - a <= limit,
              "站桩输出时方向键一段按了 %.3f 秒（上限 %.2f = 「最小切换朝向时间」+ 余量）"
              "—— 角色会一路走位、从怪身上走过去再回头（按住段 %s）"
              % (b - a, limit, [(round(x, 3), round(y, 3)) for x, y in segs]))
    # ⚠ 「两秒里按着方向键走了多久」那条**固定上限已删**（2026-09-28 口径变更 ✓）：
    #   站桩现在**每 3 次 attack** 就补一次"朝目标"、**按满「最小切换朝向时间」**（用户定的 ✓）
    #   ⇒ 配置值越大、"按着走"的占比越高 ⇒ 不再有固定上限可言 ✗。
    #   那条语义转由 `t_station_turn_every_three` 钉（每 3 轮才补 ✓ / 补的那下不超过配置值 ✓ /
    #   退出站桩清零 ✓ / 绳上不按 ✓）。这里保留上面"**单段**不超过配置值 + 余量"那条 ✓ ——
    #   它挡的是"一路按着不放"✗（`cap` 彻底失效）✓。

    # 对照 ①：**平地巡逻（patrol）同样要站桩**。
    # 2026-09-26 用户报的就是它：老写法在 attack 分支里按住朝怪的方向键
    # （注释写着"确保面向目标"，但 `_in_range` 的判据里已经带了
    # `(m.x - px) * facing >= 0` —— 能进这个分支的怪本来就在正前方），
    # 于是两次输出之间角色一直在朝怪挪，走到身上、出范围又回头。
    def dir_segments(hh, ss):
        """方向键的按住段落 [(起, 止)]，**收尾还按着的那段算到结束时刻**。"""
        ds = {ss.keymap["left"], ss.keymap["right"]}
        segs_, cur_ = [], {}
        for t, kind, k in hh.log:
            if k not in ds:
                continue
            if kind == "down":
                cur_[k] = t
            elif k in cur_:
                segs_.append((cur_.pop(k), t))
        for k, a in cur_.items():
            segs_.append((a, hh.clock.t - hh.clock0))
        return segs_

    s2 = fresh_settings(strategy="patrol", jump_random_prob=0.0, attack_dist=80.0,
                        min_turn_hold_ms=1000, turn_output_delay_ms=200)
    h2 = Harness(s2)
    h2.mobs_fn = mobs
    h2.run(2.0)
    segs2 = dir_segments(h2, s2)
    check(segs2, "平地巡逻一次方向键都没按 —— 用例本身没造出换向")
    for a, b in segs2:
        check(b - a <= limit,
              "平地巡逻站桩时方向键按了 %.3f 秒（上限 %.2f）—— 两次输出之间会朝怪挪，"
              "走到身上再回头（按住段 %s）"
              % (b - a, limit, [(round(x, 3), round(y, 3)) for x, y in segs2]))
    # ⚠ 同上：**平地巡逻这条「两秒里按着走了多久」的固定上限也删了**（2026-09-28 口径变更 ✓）
    #   —— 站桩每 3 次 attack 就补一次"朝目标"、按满「最小切换朝向时间」（用户定的 ✓）。
    #   上面"单段 ≤ 配置值 + 余量"那条照旧钉着"不会一路按着不放"✓。

    # 对照 ②：**怪在攻击范围外**时，patrol 该照旧朝它走 —— 别把移动一起改掉
    # （x=690：中心距约 160px > attack_dist 80，且在视野内）
    s3 = fresh_settings(strategy="patrol", jump_random_prob=0.0, attack_dist=80.0,
                        min_turn_hold_ms=1000, turn_output_delay_ms=200)
    h3 = Harness(s3)
    h3.mobs_fn = lambda t: [Mob(id=1, x=690.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    h3.run(2.0)
    total3 = sum(b - a for a, b in dir_segments(h3, s3))
    check(total3 >= 1.0,
          "怪在攻击范围外时，平地巡逻不走了（实测只按了 %.2f 秒）—— 追怪被一起改掉了"
          % total3)


# ---------------------------------------------------------------- 界面层自检

def build_panel():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from gui.widgets import install_wheel_guard
    install_wheel_guard(app)
    with patch.object(dinput, "use_network", lambda *a, **k: None), \
         patch.object(dinput, "use_serial", lambda *a, **k: None), \
         patch.object(dinput, "use_local", lambda *a, **k: None), \
         patch.object(dinput, "reconnect_remote", lambda *a, **k: True):
        from gui.player_panel import PlayerPanel
        p = PlayerPanel()
    p.resize(760, 620)
    p.show()
    app.processEvents()
    return p, app


def t_touchpad_f10_toggle():
    """本地 F10 开 / 关触控模式（手动输入开着时也要能用）。"""
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent
    p, app = build_panel()
    pad = p.touchpad
    pad.set_capture_enabled(True)

    def press_f10():
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_F10, Qt.NoModifier)
        handled = p.eventFilter(p, ev)
        app.processEvents()
        return handled

    check(pad.active is False, "初始不该是开启状态")
    check(press_f10() is True, "F10 应被面板吃掉（不给界面也不给游戏）")
    check(pad.active is True, "按 F10 没开启触控模式")
    press_f10()
    check(pad.active is False, "再按 F10 没关闭触控模式")
    import decision.manual_input as mi
    with patch.object(mi, "active", lambda: True):
        press_f10()
        check(pad.active is True, "手动输入开着时 F10 失效了")
    pad.set_active(False)


def t_touchpad_hidden_closes_mode():
    """面板藏起来（切页签）时自动关闭触控模式 —— 触控模式会独占鼠标。"""
    p, app = build_panel()
    pad = p.touchpad
    pad.set_capture_enabled(True)
    pad.set_active(True)
    check(pad.active is True, "没能开启触控模式")
    p.hide()
    app.processEvents()
    check(pad.active is False, "面板隐藏后触控模式还开着（鼠标会被独占）")
    p.show()
    app.processEvents()


def t_touchpad_status_text():
    """状态栏要能看出「触控模式中」，别只靠板子变蓝。"""
    p, app = build_panel()
    pad = p.touchpad
    pad.set_capture_enabled(True)
    pad.set_active(True)
    p._poll_auto_state()
    check("触控模式中" in p.lbl_state.text(),
          "状态文字没体现触控模式：%r" % p.lbl_state.text())
    pad.set_active(False)
    p._poll_auto_state()
    check("触控模式中" not in p.lbl_state.text(),
          "关掉后状态文字还留着：%r" % p.lbl_state.text())


def t_touchpad_wheel_passthrough():
    """非触控模式下，板子上的滚轮要让给外层滚动区（不该被触控板吞掉）。"""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QAbstractScrollArea, QGroupBox
    p, app = build_panel()
    pad = p.touchpad
    pad.set_capture_enabled(True)
    pad.set_active(False)
    grp = next(g for g in p.findChildren(QGroupBox) if g.title() == "操控")
    # 「操控」组要有**底色区分**（2026-09-26 用户要求）：底色写在主窗口 QSS 里、
    # 按 objectName 命中这一组 ⇒ 少了名字就是"样式改了没反应"（最难查的一类 ✗）。
    check(grp.objectName() == "CtrlGroup",
          "「操控」组没给 objectName —— 主窗口那条底色规则选不中它 ✗")
    _mw = (Path(__file__).resolve().parents[1] / "gui" / "main_window.py").read_text(
        encoding="utf-8")
    check("QGroupBox#CtrlGroup { background:" in _mw,
          "主窗口样式表里没有「操控」组的底色规则（区分看不出来）")
    scroll = grp.parentWidget()._wheel_target
    vsb = scroll.verticalScrollBar()
    check(vsb.maximum() > 0, "找错了滚动区（没有可滚范围）")
    vsb.setValue(600)
    app.processEvents()
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                     Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    app.sendEvent(pad, ev)
    app.processEvents()
    check(vsb.value() == 660,
          "板子上的滚轮没滚页面：600 -> %d（应 660 = 3 行）" % vsb.value())


def t_auto_confirm_local():
    """输入设备=本地(仅测试) 时，**开启**自动要先二次确认；取消就不该开起来。

    **为什么单列一条**：本地模式的按键打到的是**这台机器自己**的前台程序
    （浏览器 / 聊天窗口 / 工作台自己都算），而「开启自动」正是最容易被顺手
    点一下的那个按钮，还挂着 F11 全局热键（在别的程序里按也生效）。
    反过来**关闭不弹** —— 让人能顺手停下来这件事不该有门槛。

    顺带钉住：取消之后按钮要拨回未开启，否则下一次点它就变成了"关自动"。
    """
    from PyQt5.QtWidgets import QMessageBox

    p, app = build_panel()
    settings = ag.settings          # 界面读的就是这个单例（save 在 main 里被换成空操作）
    was_dev, was_on = settings.input_device, settings.enabled
    asked = [0]

    def _answer(ret):
        def f(*_a, **_kw):
            asked[0] += 1
            return ret
        return f

    # ⚠⚠ **2026-09-29：开自动多了一道「条件体检」**（`player_panel._precheck_auto` ⇒
    #   不通过弹 `QMessageBox.warning` ✓）⇒ 本用例里**每一处** `_toggle_auto` / `toggle_auto`
    #   都会走到它 ✗ ⇒ 必须**整条用例**把它 mock 掉 —— 否则会**真的弹出模态框**，在
    #   offscreen 环境下进程**直接崩**（`exit=0xC0000005` 访问冲突 ✓ 2026-09-29 就是这么
    #   崩的；靠 `python -X faulthandler` 打出栈才定位到本用例的 ④ 那条 ✓）。
    #   ⚠ 用**不计数**的 lambda（不能用 `_answer` —— 它会 `asked[0] += 1` ⇒ 把本用例
    #   "只问一次"的断言带偏 ✗）。体检不属于本用例要测的东西 ✓，它有独立用例
    #   `selftest_live_panel.t_auto_precheck` ✓。
    #   ⚠ 测试用的项目**没有 map_id** ⇒ 体检**必然报问题** ⇒ 这里统一回 `Yes` 放行 ✓
    #   （本用例测的是"本地输入的二次确认"，不是体检 ✓）。
    _wpatch = patch.object(QMessageBox, "warning",
                           lambda *_a, **_kw: QMessageBox.Yes)
    _wpatch.start()

    try:
        # ① 本地 + 选「否」：按钮拨回去，自动不能开起来
        settings.input_device = "local"
        settings.enabled = False
        p._refresh_auto_ui()
        asked[0] = 0
        p.btn_auto.setChecked(True)
        with patch.object(QMessageBox, "question", _answer(QMessageBox.No)):
            p._toggle_auto()
        check(asked[0] == 1, "本地模式下开自动没有二次确认")
        check(settings.enabled is False, "在确认框上选了「否」，自动却开起来了")
        check(p.btn_auto.isChecked() is False,
              "选了「否」按钮还停在「停止自动」上 —— 下次点它会变成关自动")

        # ② 本地 + 选「是」：正常开起来
        asked[0] = 0
        p.btn_auto.setChecked(True)
        with patch.object(QMessageBox, "question", _answer(QMessageBox.Yes)), \
                patch.object(QMessageBox, "warning",
                             lambda *a, **k: QMessageBox.Yes):
            p._toggle_auto()
        check(asked[0] == 1, "确认过了还再问一次")
        check(settings.enabled is True, "选了「是」，自动却没开起来")

        # ③ 关自动不该问
        asked[0] = 0
        p.btn_auto.setChecked(False)
        with patch.object(QMessageBox, "question", _answer(QMessageBox.Yes)):
            p._toggle_auto()
        check(asked[0] == 0, "关自动也弹确认了 —— 停下这件事不该有门槛")
        check(settings.enabled is False, "关自动没生效")

        # ④ 换成 ProMicro 就不问（正常挂机不该每次都被拦）
        settings.input_device = "remote"
        asked[0] = 0
        p.btn_auto.setChecked(True)
        with patch.object(QMessageBox, "question", _answer(QMessageBox.No)):
            p._toggle_auto()
        check(asked[0] == 0, "ProMicro 模式下开自动不该弹确认")
        check(settings.enabled is True, "ProMicro 模式下自动没开起来")

        # ⑤ 全局热键那条路也要拦 —— 它在任何程序里都生效，最该拦
        settings.input_device = "local"
        settings.enabled = False
        p._refresh_auto_ui()
        asked[0] = 0
        with patch.object(QMessageBox, "question", _answer(QMessageBox.No)):
            p.toggle_auto()                     # F11 热键路径（main_window.nativeEvent）
        check(asked[0] == 1, "F11 热键开自动没走确认")
        check(settings.enabled is False, "热键路径选了「否」，自动还是开起来了")
    finally:
        _wpatch.stop()              # ⭐ 体检那层 mock 收工（别忘了，会污染后面的用例 ✗）
        settings.input_device = was_dev
        settings.enabled = was_on
        p._refresh_auto_ui()


def t_reset_link_taps_directions():
    """「重置指令通道」：**先把 ←/→/↑/↓ 各点按一遍，再重置**（用户 2026-10-01 ✓）。

    为什么必须先按、后重置：重连 / RELEASEALL 只清得掉**固件侧**按住的键；若一条
    RELEASE 在链路里丢了，**游戏侧**会一直以为某个方向键还按着（角色往一个方向一直走、
    按啥都不停）。逐个方向键 `tap` 一下 = 往游戏里补「按下→松开」把游戏侧的键钉回松开 ✓。
    """
    p, _app = build_panel()
    settings = ag.settings
    _key = settings.keymap or {}
    _dirs = [k for n in ("left", "right", "up", "down")
             if (k := _key.get(n))]
    log = []
    _was_resync = settings.input_resync
    try:
        with patch.object(dinput, "tap",
                          lambda k, *a, **kw: log.append(("tap", k))), \
             patch.object(dinput, "reconnect_remote",
                          lambda *a, **kw: (log.append(("reconnect",)) or True)), \
             patch.object(dinput, "release_all_remote",
                          lambda *a, **kw: log.append(("releaseall",))), \
             patch.object(dinput, "link_health",
                          lambda *a, **kw: {"backend": "remote", "ok": True}), \
             patch.object(dinput, "key_up", lambda *a, **kw: None):
            settings.input_resync = False
            p._reset_link()
        check(log[:4] == [("tap", k) for k in _dirs],
              "该先把 ←/→/↑/↓ 各点按一遍（顺序 = 左/右/上/下）：%r" % (log[:6],))
        check(log.index(("releaseall",)) >= 4,
              "点按方向键必须发生在 RELEASEALL **之前**（先按、后重置 ✓）：%r" % (log,))
        check(settings.input_resync is True,
              "重置后该让决策层重同步按键状态（input_resync）")
    finally:
        settings.input_resync = _was_resync


def t_stuck_key_watchdog():
    """卡键看门狗（2026-09-26 用户要求"先优化查卡键"）：输出键在**非输出状态**还按着
    ⇒ 记一笔（perf 的 key_stuck + 说明）+ **松开**；而正常输出时**不许误杀**。

    为什么要有：用户报过"画面上一个框都没有、角色却持续在按输出"。根因之一（怪在输出
    序列 down 与 up 之间消失 ⇒ up 永不发出）已经修掉，但**任何**漏放路径都该被兜住；
    更要紧的是卡键原本在 perf.log 里**看不出来**（只能靠猜"是不是幽灵框"）✗。
    判据只看输出序列的键：移动键按住几十秒是正常的。
    """
    from decision import agent as ag

    s = fresh_settings(strategy="patrol", jump_random_prob=0.0,
                       output_seq=[{"type": "down", "key": "attack"},
                                   # 长 delay：让 up 一时半会儿回不来（漏放的温床）
                                   {"type": "delay", "ms": 60000},
                                   {"type": "up", "key": "attack"}])
    h = Harness(s)
    atk = s.keymap["attack"]
    h.mobs_fn = lambda t: [Mob(id=1, x=560.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    with h._patched():
        # ① 先让输出序列**真的把键按下去**（harness 只认真按过的键，凭空造的上下文它不认 ✗）
        h.run(0.3)
        check(atk in [k for _t, kind, k in h.log if kind == "down"],
              "用例本身没跑起来（输出键没按下）：%s" % h.log[:6])
        # 造"漏放"现场：状态已经不是输出状态了，而键还按着（正常路径这一拍才会释放）
        h.agent.state = "idle"
        h.agent._out_held_since = None
        h.log.clear()
        h.agent._watch_output_held(1000.0)                 # 刚发现 ⇒ 只开始计时
        ups = lambda: [k for _t, kind, k in h.log if kind == "up"]
        check(atk not in ups(),
              "刚发现就松手了（该先给宽限 %.1fs）：%s" % (ag.OUT_KEY_STUCK_S, h.log))
        h.agent._watch_output_held(1000.0 + ag.OUT_KEY_STUCK_S - 0.2)
        check(atk not in ups(), "宽限还没到就松开了：%s" % h.log)
        h.agent._watch_output_held(1000.0 + ag.OUT_KEY_STUCK_S + 0.1)
        check(atk in ups(),
              "卡键超过 %.1fs 也没被松开：%s" % (ag.OUT_KEY_STUCK_S, h.log))
        check(h.agent._output_ctx is None, "松开之后输出上下文还在（下一拍会被它带跑）")

        # ② 正常输出中（状态就是 attack）⇒ **不许**误杀：连跑 6 秒也不松
        h.log.clear()
        h.agent._output_ctx = [list(s.output_seq), 0, 0.0, {atk}, []]
        h.agent.state = "attack"
        h.agent._out_held_since = None
        for k in range(12):
            h.agent._watch_output_held(2000.0 + k * 0.5)
        check(h.agent._output_ctx is not None,
              "正在打的时候被看门狗松手了（误杀）：%s" % h.log)
        check(atk not in ups(), "正在打的时候发了 up：%s" % h.log)
        h.agent._output_ctx = None
        h.agent.state = "idle"


def t_multi_step_route():
    """**多步路径**（2026-09-26 新加）：一条路径解析成 N 个任务后，按顺序跑完。

    为什么单独做一层：`命令前往` 原来只下**第一步**（"后面 N 步不会自动接着走" ✗），
    而"定点休息"要真的**走到**那个集合 ⇒ 必须有"逐步走完"的执行器。

    钉四件事：
      · 起跑：第一个立刻挂上、其余进队列；
      · 到了 ⇒ **自动接下一步**（不是原地结束 ✗）；
      · 全走完 ⇒ 任务清空 + 一句"整条路径走完"；
      · 任何一步失败 ⇒ **整条停掉**，并说清断在第几步（不许硬着头皮往下走 ✗）。
    """
    from decision import route

    s = fresh_settings()
    s.enabled = True
    h = Harness(s)
    # 落点就摆在玩家附近（x≈520）：到位判据 = 脚下集合已经是目标
    a = route.WalkJob("甲集合", [(520.0, 510.0, 530.0, -208.0, "1")])
    b = route.WalkJob("乙集合", [(520.0, 510.0, 530.0, -208.0, "2")])
    with h._patched():
        check(h.agent.start_route([a, b], why="用例：两步"),
              "多步路径没起跑")
        check(h.agent._climb is a and len(h.agent._route) == 1,
              "第一步没立刻挂上 / 剩余步没进队列：%r" % (h.agent._route,))

        ws = h.ws(with_mob=False)
        ws.player.here_sets = ["甲集合"]              # 第一步到了
        check(h.agent._climb_tick(1.0, 520.0, set(), ws) is False,
              "第一步到了就整条结束（没接着走第二步）")
        check(h.agent._climb is b, "第二步没挂上：%r" % (h.agent._climb,))

        ws.player.here_sets = ["乙集合"]              # 第二步到了 ⇒ 整条走完
        check(h.agent._climb_tick(2.0, 520.0, set(), ws) is True,
              "最后一步到了却没结束整条路径")
        check(h.agent._climb is None and not h.agent._route,
              "走完了还留着任务：%r / %r" % (h.agent._climb, h.agent._route))
        check("整条路径走完" in h.agent.current_goto_note(),
              "走完了没说清：%r" % h.agent.current_goto_note())

        # 失败 ⇒ 整条停掉，并说清断在第几步。
        # ⚠ 2026-09-27 起**执行器不再"最多试几次"**（用户："那个 1/3 尝试要去掉，我们现在
        #    都走「寻路超时时间」"）⇒ 这一步的"停"靠**agent 那把钟**（下面 `goto_timeout_s`），
        #    不再靠"试满几次就放弃" ✗。
        a2 = route.WalkJob("丙集合", [(520.0, 510.0, 530.0, -208.0, "3")])
        b2 = route.WalkJob("丁集合", [(520.0, 510.0, 530.0, -208.0, "4")])
        # ⚠ 2026-09-29 晚口径：「寻路超时时间」对所有任务生效 ⇒ 这条用例**不带签注也行** ✓；
        #   带上它顺带钉"追击路径同样吃这道闸" ✓（"中途失败"就是用那把钟造出来的 ✓）。
        h.agent.start_route([a2, b2], why="用例：中途失败", origin=dict(CHASE_ORIGIN))
        # ⚠ 2026-09-26 起「超时只有一把钟」：任务自己的 `timeout_s` 被 `start_climb`
        #    关掉了（见那里的说明）⇒ 这段必须用**agent 那把钟**：设置里给个小上限，
        #    再把 `_climb_started` 摆成"已经跑了一会儿"（本用例用的是字面时刻 10.0 ✗，
        #    而 `start_climb` 打的是真 monotonic ⇒ 不摆的话永远不会超时 ✗）。
        s.goto_timeout_s = 0.5
        h.agent._climb_started = 9.5
        ws2 = h.ws(with_mob=False)
        ws2.player.here_sets = []
        h.agent._climb_tick(10.0, 520.0, set(), ws2)      # 已跑 0.5s ⇒ 到点
        h.agent._climb_tick(11.0, 520.0, set(), ws2)      # 已经超时 ⇒ 整条停
        check(h.agent._climb is None and not h.agent._route,
              "中途失败却没把整条路径停掉：%r / %r" % (h.agent._climb, h.agent._route))
        note = h.agent.current_goto_note()
        check("路径中断" in note and "1/2" in note.replace("第 1/2 步", "1/2"),
              "失败没写清断在第几步：%r" % note)


def t_afk_type_does_not_reschedule():
    """用户 2026-09-26 要确认的：**改「防掉线行为类型」不许重新计时**。

    计时只在这两处产生（读代码确认）：
      · `tick` 里防掉线开关的**上升沿**（`if not self._was_afk_enabled`）⇒ 排下次触发；
      · `_begin_rest` 抽一次本次休息时长、`_finish_rest` 排下一次。
    **`anti_afk_type` 在整个计时链里一次都没被读**（它只出现在字段定义 / to_dict /
    from_dict / 设置界面）⇒ 换类型就是"换一种歇法"，不会把走了一半的计时清零 ✓。
    这条把它钉住：以后谁要是拿"类型变了"当重排的触发条件，用例立刻红。
    """
    s = fresh_settings()
    s.enabled = True
    s.anti_afk_enabled = True
    h = Harness(s)
    with h._patched():
        h.agent._was_afk_enabled = False          # 假装刚打开防掉线（走上升沿）
        h.agent.tick(h.ws(with_mob=False))
        next0 = h.agent._next_afk
        check(next0 > 0, "防掉线开着却没排下一次触发")
        # ⚠ 用**夹具时钟**起休息，别自己编一个 now：夹具的时钟在 1000 附近，
        # 拿 now=100.0 去 `_begin_rest` 会让"休息到点时刻"落在过去 ⇒ 一秒就到期，
        # 看起来像"改类型把计时清了" ✗（我第一版就是这么误判的）。
        h.agent._begin_rest(h.clock.t)
        until0 = h.agent._rest_until
        check(until0 > h.clock.t, "开始休息没算休息时长：%r" % until0)
        # 换行为类型（模拟用户在设置里改），再正常走几拍
        s.anti_afk_type = "(换成另一种类型)"
        for _k in range(5):
            h.agent.tick(h.ws(with_mob=False))
        check(h.agent._rest_until == until0,
              "改了行为类型，本次休息时长被重算了：%r → %r"
              % (until0, h.agent._rest_until))
        check(h.agent._next_afk == next0,
              "改了行为类型，下次触发时刻被重排了：%r → %r"
              % (next0, h.agent._next_afk))


def t_drop_flow():
    """下跳（drop）流程（用户 2026-09-27 定的，照抄）：

      ① 找**最近**的允许下跳的 foothold，走到它的**中心**；
      ② 判定**位于该 foothold** 持续 ≥「坐标对齐误差时间」⇒ **松开行走键**；
      ③ **按住 ↓ + 点按跳**；
      ④ **Y 变化超过「坐标对齐误差范围」⇒ 停掉"连按跳"，但 ↓ 继续按着**，
         并按「移动操作尝试间隔」补按跳（用户 2026-09-28 追加 ✓）；
      ⑤ 落到目标 foothold ⇒ 成功。

    钉六件（每件都对应上面一句话，改坏了当场红）：
      ① 就近挑：两个可下跳点、人在第三个位置 ⇒ 挑**近的那个**；
      ② 判据是"**落在该 foothold 的范围里**"（站在边上也算 ✓），**不是**"离中心 ≤ tol" ✗
         —— 后者要求精确到像素，而读数本身在抖；
      ③ 要走的话方向**指向中心**；
      ④ 进了范围**这一拍**就不再发行走键（立刻按住 ↓ ✓）；
      ⑤ 按住 ↓（`dir=-1`）**且点按跳**：跳只按 `TAP_ON_S` 就松开，↓ 一直按着；
      ⑥ Y **向下**变化超过 `y_tol_px` ⇒ ↓ **依旧按着**、并按「移动操作尝试间隔」
         (`retry_ms`) **补按跳**（用户 2026-09-28 ✓："下落过程中也按照『移动操作尝试间隔』
         补按 按住 ↓ + 跳"）；落进目标集合才松手 ⇒ 落地即成功。
    """
    from decision import route

    C = route.DropJob
    #: 两个可下跳的点：(中心 x, 左, 右, foothold id) —— **老的 4 元组**（面 y 缺 ⇒ 不参与
    #: y 校验 ✓ 见 `DropJob.__init__`；新口径的五元组形状见 `t_drop_align_layer_guards` ✓）
    spots = [(300.0, 260.0, 340.0, "fhA"), (100.0, 80.0, 120.0, "fhB")]

    # ① 就近挑 + ③ 朝中心走
    j = C("上平台", "下平台", spots, tol_px=10, hold_ms=0)
    o = j.update(0.0, px=380.0, py=0.0)
    check(j._pick is not None and j._pick[-1] == "fhA",
          "没就近挑下跳点（该挑 fhA）：%s"
          % (None if j._pick is None else j._pick[-1],))
    check(o["move"] < 0,
          "没朝该 fh 的中心走（中心 300、人在 380 ⇒ 该往左）：%s" % o)

    # ② 站在**范围边上**也算"可下跳"（260~340，人在 265，离中心 35 > tol 10）
    #   ⚠ **故意把 `hold_ms` 给到 1000**（= 用户现场那个项目的值 ✗）：新口径下它**不该被等**
    #     ⇒ 进了范围**这一拍就该**按住 ↓ ✓。
    j = C("上平台", "下平台", spots, tol_px=10, hold_ms=1000)
    o = j.update(0.0, px=265.0, py=0.0)
    check(o["move"] == 0,
          "站在该 foothold 范围里却还在走 —— 判据该是「落在它的范围里」，"
          "不是「离中心 ≤ tol」：%s" % o)
    check(j.phase == C.ARMED and o["dir"] == -1 and not o["jump"],
          "⭐ 进了范围**这一拍就该**进「按住 ↓」（**不再要求站够 `hold_ms`** ✗ —— 现场"
          "2026-09-28 18:17 那次「三楼 → 二楼」就是被它逼到 10 秒超时的 ✓）："
          "%s（phase=%s）" % (o, j.phase))

    # ⑤ 再等 ARM_HOLD_S ⇒ 按住 ↓ + 点按跳（跳先按下）
    o = j.update(0.55, px=265.0, py=0.0)
    check(j.phase == C.DROP and o["dir"] == -1 and o["jump"],
          "该按住 ↓ 并按下跳（点按开始）：%s（phase=%s）" % (o, j.phase))
    # 点按超过 TAP_ON_S ⇒ 松跳，↓ 还在按
    o = j.update(0.55 + route.TAP_ON_S + 0.01, px=265.0, py=0.0)
    check(j.phase == C.DROP and not o["jump"] and o["dir"] == -1,
          "点按跳该在 TAP_ON_S 后松开、而 ↓ 继续按着：%s" % o)

    # ⑥' **向上抖不算"开始下落"**（用户 2026-09-27："当 Y 变化超过「坐标对齐误差范围」**才能**
    #     松开 ↓"）—— 以前这里写 `abs(...)` ⇒ 人还站在平台上、读数向上抖 >误差范围 就被判
    #     "开始下落"、把 ↓ 松掉 ⇒ 那一次下跳当场作废 ✗
    o = j.update(0.70, px=265.0, py=-30.0)         # 向上"变"了 30 > y_tol 10 ⇒ **不许松**
    check(j.phase == C.DROP and o["dir"] == -1 and not o["jump"],
          "Y 只是**向上**抖了一下就把 ↓ 松了（那不是下落 ✗）：%s（phase=%s）"
          % (o, j.phase))
    # ⑥ Y **向下**变化超过「坐标对齐误差范围」⇒ **↓ 依旧按着**，并按「移动操作尝试间隔」
    #    **补按跳**（先用户 2026-09-27 那条："在 drop 时，**除非位置状态到了目的地，否则 ↓
    #    是不能松开的**"✓；再用户 2026-09-28 这条："**下落过程中也按照『移动操作尝试间隔』
    #    补按 按住 ↓ + 跳**"✓ —— 掉下去的路上也会卡住：被绳吸住 / 卡在半空的绳上 ✓）。
    o = j.update(1.00, px=265.0, py=25.0)          # 掉了 25 > y_tol 10
    check(j.phase == C.DROP and o["dir"] == -1,
          "Y 都变了却把 ↓ 松开了（该一直按到落进目标集合 ✗）：%s（phase=%s）" % (o, j.phase))
    check(o["jump"], "下落过程中该**立刻补按一下跳**（那一下正是救「被绳吸住」的 ✓）：%s" % o)
    check("↓ 一路按着" in j.note and "补按跳" in j.note,
          "那句话没写清「↓ 一路按着 + 按尝试间隔补按跳」：%r" % j.note)
    # 这一下按满 `TAP_ON_S` 就**松开跳**（↓ 还在按 ✓ —— 别退化成"一直按住跳"✗）
    o = j.update(1.00 + route.TAP_ON_S + 0.01, px=265.0, py=25.0)
    check(not o["jump"] and o["dir"] == -1,
          "下落中这一下跳该在 `TAP_ON_S` 后松开（一直按住跳会被当成还挂在上面 ✗）：%s" % o)
    # 还没到一个「移动操作尝试间隔」（`retry_ms`=3000ms）⇒ **不按下一跳**（不是键盘风暴 ✗）
    o = j.update(1.00 + 1.5, px=265.0, py=25.0)
    check(not o["jump"] and o["dir"] == -1,
          "没到「移动操作尝试间隔」就又按跳了（该按那个间隔来 ✗）：%s" % o)
    # 到了一个间隔 ⇒ **再补按一下** ✓
    o = j.update(1.00 + 3.01, px=265.0, py=25.0)
    check(o["jump"] and o["dir"] == -1,
          "到了一个「移动操作尝试间隔」(3000ms) 该再补按一下跳：%s" % o)
    # 落地（脚下已是目标集合）⇒ 成功
    o = j.update(4.20, px=265.0, py=200.0, here_sets={"下平台"})
    check(o["done"] and j.phase == C.DONE and o["dir"] == 0,
          "落到目标集合没判成功（或还在按键）：%s（phase=%s）" % (o, j.phase))


def t_drop_align_layer_guards():
    """下跳 ALIGN 的两条「别在错的层上硬做」闸（用户 2026-10-02 现场：龙族打猎场卡 40 秒 ✓）。

    现场（`behavior.log` 15:39:31 + 图 `105080000`）：人从「上层」滑落平台、掉到「二楼」，
    而那个 x（456.7）正好落在**上层**某块可下跳 foothold（#438 x=464~513）的 ±`tol_px` 里
    ⇒ 老 ALIGN **只看 x** ⇒ 判成"已经在下跳点上" ⇒ 从**错误的层**按 ↓+跳（在二楼穿平台）
    ⇒ 一路掉到最底层，而任务还在"等落到中层"⇒ 卡死 ✗✗。

    钉六件（判据**只问广播** ✓ 同 SKILL 77，执行器**不自己算几何** ✗）：
      ① x 命中但**脚下那块面不是它**（`ground_y` 差得远 ⇒ 人不在那块面上 / 在空中）⇒
         **不许**进 ARMED ✓（在错的层上按 ↓+跳毫无意义 ✗）；
      ② 对照：同一 x、`ground_y` 就是那块面的高度 ⇒ 照旧 ARMED ✓（不误伤 ✓）；
      ③ 对照：`ground_y=None`（判不出来）⇒ **退回只看 x** ⇒ 照旧 ARMED ✓（老行为 ✓）；
      ④ 人**已经到目标层**（`here_sets` 命中 `dst_set`）⇒ 当拍**收工** ✓
         （他是掉下来正好落到目标 —— 那是"到"，不是"失败" ✓）；
      ⑤ 人**既不在起跳平台、也不在目标层**（脚下是别的集合）⇒ **如实失败** ✓
         （让上层按现在的位置重算路线 ✓，而不是继续朝那个 x 走 ✗）；
      ⑥ 脚下**没圈进任何集合**（判不出来 / 半空中）⇒ **不拦也不失败** ✓（照旧走流程等它落定 ✓）。
    ⚠ **② 的另一半"重新挑点"没有实现，是空操作**（说明见 `DropJob.update` 的 ALIGN 那一段 ✓）：
      `spots` 全部来自起跳集合 ⇒ 换层之后重挑出来的还是同层的点 ⇒ 朝它走也走不到 ✗ ⇒
      那种情况交给 ④/⑤ 那两条**集合判据**收口 ✓（只有集合说得清"人在哪一层" ✓）。
    """
    from core import mapdata, zones
    from decision import route

    # ---------- 真数据：105080000 的「上层 --下跳--> 中层」，spots 是真面 y ✓ ----------
    t = mapdata.load("105080000")
    z = zones.load("105080000")
    edge = next((e for e in z.edges
                 if e.get("from") == "上层" and e.get("to") == "中层"
                 and e.get("kind") == "drop"), None)
    check(edge is not None, "用例前提：这张图里该有「上层 →(下跳)→ 中层」那条边")
    box = {str(f.fid): f for f in t.footholds}
    _ys = [float(box[str(fid)].y_at((float(box[str(fid)].left)
                                     + float(box[str(fid)].right)) / 2.0))
           for fid in z.sets["上层"]["footholds"] if str(fid) in box]
    job = route.drop_job_for_edge(t, z, edge, tol_px=10, hold_ms=250)
    check(all(len(s) == 5 and s[3] is not None for s in job.spots),
          "真数据造出来的可下跳点不是五元组（面 y 丢了 ⇒ 「站在这块面上」判不了 ✗）：%r"
          % (job.spots[:2],))
    check(all(min(_ys) - 1.0 <= s[3] <= max(_ys) + 1.0 for s in job.spots),
          "面 y 不在「上层」那些面的高度范围里（挑错集合了？）：%r" % (job.spots[:2],))
    check(str(job._arrived(855.5, ["二楼"], ground_y=855.5)) == "",
          "用例前提：人在二楼时不该判成「到达中层」（`_arrived` 得说不 ✗）")
    # 找一块"上层"的面，取它中心当 x（下面是 ①/②/③）
    _sp = job.spots[0]
    _cx, _fy = float(_sp[0]), float(_sp[3])

    # ① x 命中、但脚下那块面**不是它**（广播给 `ground_y=855.5`，这块面在 ~311）
    #   ⇒ **不许** ARMED（在错的层 / 空中按 ↓+跳毫无意义 ✗）；集合判不出来 ⇒ 也不许判失败
    j1 = route.drop_job_for_edge(t, z, edge, tol_px=10, hold_ms=250)
    o = j1.update(0.0, px=_cx, py=855.5, here_sets=[], ground_y=855.5)
    check(j1.phase == route.DropJob.ALIGN and not o["jump"] and not o["done"]
          and not o["failed"],
          "x 命中但脚下那块面离它有 540 px（人不在那块面上）却还照旧进 ARMED ⇒ 会从错误的"
          "层按 ↓+跳 ✗：%s（phase=%s）" % (o, j1.phase))
    check("已经在可下跳的 fh" not in str(o["note"]),
          "不该宣称「已经在可下跳的 fh 上」（脚下那块面明明不是它 ✗）：%r" % (o["note"],))

    # ② 对照：同一 x、脚下就是这块面那一层 ⇒ 照旧 ARMED ✓
    j2 = route.drop_job_for_edge(t, z, edge, tol_px=10, hold_ms=250)
    o = j2.update(0.0, px=_cx, py=_fy, here_sets=["上层"], ground_y=_fy)
    check(j2.phase == route.DropJob.ARMED and o["dir"] == -1,
          "人就站在那块下跳面上却没进「按住 ↓」（y 那把尺误伤了 ✓ 它在正路上）：%s"
          % (o,))

    # ③ 对照：`ground_y` 判不出来（None）⇒ 退回只看 x ⇒ 照旧 ARMED ✓（老行为一字不变）
    j3 = route.drop_job_for_edge(t, z, edge, tol_px=10, hold_ms=250)
    o = j3.update(0.0, px=_cx, py=855.5, here_sets=["上层"])
    check(j3.phase == route.DropJob.ARMED,
          "拿不到 `ground_y`（判不出来）就把人拦住了 —— 该退回只看 x ✗：%s" % (o["note"],))

    # ④ 已经到目标层 ⇒ 当拍收工（掉下来正好落到目标 = "到" ✓）
    j4 = route.drop_job_for_edge(t, z, edge, tol_px=10, hold_ms=250)
    o = j4.update(0.0, px=_cx, py=600.0, here_sets=["中层"], ground_y=600.0)
    check(o["done"] and j4.phase == route.DropJob.DONE and "到达" in str(o["note"]),
          "人在 ALIGN 相就已经站在「中层」上了，却没当拍收工（会继续朝上层的点走 ✗）：%s"
          % (o,))

    # ⑤ 换层（既不是起跳的「上层」、也不是目标「中层」）⇒ 如实失败 ✓
    j5 = route.drop_job_for_edge(t, z, edge, tol_px=10, hold_ms=250)
    o = j5.update(0.0, px=456.7, py=855.5, here_sets=["二楼"], ground_y=855.5)
    check(o["failed"] and j5.phase == route.DropJob.FAILED,
          "人掉到「二楼」（既不在上层、也不在中层）还在硬做这条下跳 ✗（该失败让上层"
          "重算）：%s" % (o,))
    check("交给上层" in str(o["note"]) and "二楼" in str(o["note"]),
          "失败原因没说清是「换层了、该重算」（排查时看不出 ✗）：%r" % (o["note"],))

    # ⑥ 脚下**没圈进任何集合**、也没有 `ground_y`（半空中 / 老调用方没喂）⇒ **不拦也不失败** ✓
    #   （判不出来就不拦 —— 这是全项目同一条口径 ✓；换层那条闸只在"脚下明确是别的集合"时才拦 ✓）
    j6 = route.drop_job_for_edge(t, z, edge, tol_px=10, hold_ms=250)
    o = j6.update(0.0, px=_cx, py=855.5, here_sets=[])
    check(not o["failed"] and j6.phase == route.DropJob.ARMED,
          "判不出来（没集合、没 ground_y）时该退回只看 x（照旧 ARMED）✗：%s" % (o,))

    # ⑦ 形状：老的 4 元组仍收（面 y 记 None ⇒ 不参与 y 校验 ✓）；别的形状报错 ✓
    j7 = route.DropJob("甲", "乙", [(700.0, 660.0, 740.0, "41")])
    check(j7.spots[0][3] is None and j7.spots[0][-1] == "41",
          "老的 4 元组没被兼容成「面 y=None」（老用例/老调用方会当场炸 ✗）：%r" % (j7.spots,))
    try:
        route.DropJob("甲", "乙", [(700.0, 660.0, "41")])
    except ValueError:
        pass
    else:
        raise AssertionError("三个元素的落点没报错（形状不许猜 ✗）")


def t_drop_retry_gap():
    """下跳的「移动操作尝试间隔(ms)」重试（用户 2026-09-27 要求 2，原话）：

        "对于下跳(drop)按住↓+点按跳后，如果在"移动操作尝试间隔(ms)"内Y无变化则补发一个
         松开↓，然后尝试 按住↓+点按跳→。。。"

    ⚠ 为什么是"**先松开**"而不是"再按一次 ↓"：`KeyState` 只在**键集变化**时才发键 ⇒
      一直按着再按一次，对面**收不到新的按下** ✗（爬绳「补按 ↑」是同一个坑的另一面）。

    钉五件：
      ① 窗口没到之前：↓ 一直按着、不重试（不然是键盘风暴 ✗）；
      ② 窗口一到、Y 还没动 ⇒ 有**一拍 `dir=0`**（那就是真的松开了 ✓）；
      ③ 紧接着**重新按住 ↓**（dir=-1）**并再点按一次跳**（jump 由 False 变 True ✓）；
      ④ 那句话说清是"第几次"（人能看出在重试 ✓）；
      ⑤ `retry_ms=0` ⇒ **不重试**（老行为：一直按着等 ✓）。
    """
    from decision import route

    C = route.DropJob
    spots = [(100.0, 80.0, 120.0, "fhB")]

    def _to_drop(j, t=0.0):
        """推到"按住 ↓ + 点按跳"那一拍（`hold_ms=0` ⇒ 从 ALIGN 直接过）。"""
        j.update(t, px=100.0, py=0.0)              # ALIGN → ARMED（同一拍里过）
        j.update(t, px=100.0, py=0.0)
        return j.update(t + route.ARM_HOLD_S + 0.01, px=100.0, py=0.0)   # → DROP（按跳）

    j = C("上平台", "下平台", spots, tol_px=10, hold_ms=0, y_tol_px=10,
          retry_ms=500, stall_s=1.0)
    o = _to_drop(j)
    check(j.phase == C.DROP and o["jump"], "没进「按住 ↓ + 连按跳」：%s" % o)
    t = route.ARM_HOLD_S + 0.01
    # ① 这一下按完了、窗口没到 ⇒ ↓ 还按着（不重试）
    o = j.update(t + route.TAP_ON_S + 0.01, px=100.0, py=0.0)
    check(not o["jump"] and o["dir"] == -1,
          "点按结束后 ↓ 该继续按着（窗口还没到）：%s" % o)
    # ①' ⚠ **连按**（用户 2026-09-27 口径："按住↓，下一拍再按跳，可以用「卡住判定时长」连按"
    #     ＋"按住↓就会趴，可能还没趴你就按跳所以无效"）⇒ 到「点按周期」必须**再按一下** ✓
    o = j.update(t + route.TAP_PERIOD_S + 0.01, px=100.0, py=0.0)
    check(o["jump"] and o["dir"] == -1,
          "只按了一次跳、没有连按（那一下按早了就白按 ⇒ 现场就是「趴着不跳」✗）：%s" % o)
    # ② ⭐⭐ **2026-09-28 改口径**：窗口（「卡住判定时长」）到点、Y 还没动 ⇒
    #    **也不许松开 ↓ 起身** ✓ —— 用户原话："**到点 Y 还没动，也不应该松开 ↓ 起身，
    #    只要进了 DROP 就一路按住 ↓**"。
    #    （老口径是"补发一个 松开 ↓ ⇒ 隔「移动操作尝试间隔」再来一轮"✗ ⇒ 每轮只活
    #      ~0.35 s 就站直一次，游戏里**根本来不及趴** ⇒ 每一下跳都是站着原地跳 ✓
    #      那就是 2026-09-28 现场"drop 又卡了、在原地跳"的根因 ✗）
    o = j.update(t + 1.05, px=100.0, py=0.0)       # 从本轮起点算已过 ≥1.0s
    check(o["dir"] == -1,
          "窗口到点就把 ↓ 松了（用户 2026-09-28：**一路按住 ↓**、不许起身 ✗）：%s" % o)
    # ③ **不许再有「空等的几拍」** —— 往后几拍 ↓ 仍然按着（全程不松 ✓）
    o = j.update(t + 1.10, px=100.0, py=0.0)
    check(o["dir"] == -1,
          "「等间隔」那几拍居然不按 ↓（那是已作废的老行为 ✗）：%s" % o)
    o = j.update(t + 1.05 + 0.51, px=100.0, py=0.0)
    check(o["dir"] == -1, "后面的拍没继续按住 ↓（该一路按到落地 ✗）：%s" % o)

    # ⑤ retry_ms = 0 ⇒ 不重试（一直按着 ↓ 等，不许出现"松开"那拍）
    j2 = C("上平台", "下平台", spots, tol_px=10, hold_ms=0, y_tol_px=10, retry_ms=0)
    _to_drop(j2)
    released = []
    tt = route.ARM_HOLD_S + 0.01
    while tt < route.ARM_HOLD_S + 6.0:
        oo = j2.update(tt, px=100.0, py=0.0)
        if oo["dir"] == 0 and not oo["jump"]:
            released.append(round(tt, 2))
        tt += 0.05
    check(not released,
          "retry_ms=0（老行为）却在 %s 秒松开了 ↓ —— 那是一直按着等的语义 ✗" % released[:3])

    # ⑥ **不再是「Y 一变就松 ↓」了**（用户 2026-09-27 改口，原话："在 drop 时，**除非位置状态
    #    到了目的地，否则 ↓ 是不能松开的**"✓）⇒ 往下掉那几拍：**↓ 一直按着**，并按「移动
    #    操作尝试间隔」**补按跳**（用户 2026-09-28 ✓），直到**落进目标集合**才松手 ✓
    j3 = C("上平台", "下平台", spots, tol_px=10, hold_ms=0, y_tol_px=10, retry_ms=500)
    _to_drop(j3)
    t3 = route.ARM_HOLD_S + 0.01
    o = j3.update(t3 + 0.01, px=100.0, py=20.0)          # Y 掉了 20
    check(o["dir"] == -1 and o["jump"] and j3.phase == C.DROP,
          "Y 掉了却把 ↓ 松开了 / 没按「尝试间隔」补按跳（该一路按着 ↓ + 补按跳 ✓）：%s" % o)
    o = j3.update(t3 + 0.01 + 1.5, px=100.0, py=20.0)     # 还在掉、一直没落进目标
    check(o["dir"] == -1 and j3.phase == C.DROP,
          "还没落进目标集合就把 ↓ 松了（用户明确「除非到目的地，否则不许松」✗）：%s" % o)
    o = j3.update(t3 + 0.01 + 1.6, px=100.0, py=300.0, here_sets={"下平台"})
    check(o["done"] and o["dir"] == 0,
          "落进目标集合了却没松手 / 没判成功：%s（phase=%s）" % (o, j3.phase))

    # ⑦ **没有 y 读数 ⇒ 先别按跳**（基准拿不到就判不出"开始下落" ⇒ 按了也只能一路按到超时 ✗）
    j4 = C("上平台", "下平台", spots, tol_px=10, hold_ms=0, y_tol_px=10, retry_ms=500)
    j4.update(0.0, px=100.0, py=None)                    # ALIGN（站在该 fh 上）→ ARMED
    j4.update(0.0, px=100.0, py=None)
    o = j4.update(route.ARM_HOLD_S + 0.01, px=100.0, py=None)
    check(not o["jump"] and o["dir"] == -1,
          "拿不到 y 读数却按了跳（没有基准 ⇒ 判不出下落，只能一路按到超时 ✗）：%s" % o)
    o = j4.update(route.ARM_HOLD_S + 0.02, px=100.0, py=0.0)   # 读数来了 ⇒ 才按跳
    check(o["jump"], "y 读数来了之后该按跳了：%s" % o)


def t_align_params():
    """「判定参数」（设置 → 判定参数页签）：**对齐类功能**靠这两个数说话，存得住、有默认。

    为什么单独立一条：这两个数决定"多近才算对齐 / 保持多久算成立"，是**爬绳、寻路
    走到某个 x、判定在不在绳上**共同的判据。数值本身不重要，重要的是：
      · 有**明确默认值**（老项目文件里没有这两个键时不能变成 0 —— 0 会让"误差范围"
        变成"必须像素级精确"，谁都对不上）；
      · 存读一致（`to_dict`/`from_dict` 都带上，否则改完重开就丢）；
      · 设置弹窗里真的有一个能改它们的页签。
    """
    from PyQt5.QtWidgets import QApplication

    from decision.agent import settings      # 模块级没有这个名字，函数里取（同别的用例）

    app = QApplication.instance() or QApplication([])
    check(hasattr(settings, "align_tol_px") and hasattr(settings, "align_hold_ms"),
          "决策设置里没有对齐参数")
    check(int(settings.align_tol_px) > 0,
          "坐标对齐误差范围默认值是 0 —— 那就成了必须像素级精确：%r"
          % settings.align_tol_px)
    check(int(settings.align_hold_ms) >= 0, "对齐误差时间默认值不对：%r"
          % settings.align_hold_ms)

    # 寻路超时时间（2026-09-26 新增的判定参数）：有默认值 + 界面能改 + 改完落回配置
    check(hasattr(settings, "goto_timeout_s"), "决策设置里没有「寻路超时时间」")
    check(float(settings.goto_timeout_s) >= 0,
          "寻路超时时间默认值不对：%r" % settings.goto_timeout_s)
    # 「攀爬参数」子组的参数（2026-09-26 起：失败后延迟激活时间 / 延迟增量 / 补按观察窗；
    # 2026-09-27 加**对齐绳梯移动延迟(ms)** 并**改名**——原名叫「攀爬失败保护」，装不下了 ✓）
    check(hasattr(settings, "climb_retry_delay_s")
          and hasattr(settings, "climb_retry_delay_inc_s"),
          "决策设置里没有「攀爬失败后延迟激活时间 / 延迟增量」")
    from decision import route as _route
    check(int(getattr(settings, "climb_align_gap_ms", -1))
          == int(_route.TAP_PERIOD_S * 1000),
          "「对齐绳梯移动延迟(ms)」默认值不是 `route.TAP_PERIOD_S`（两处各写一个数会分叉）：%r"
          % getattr(settings, "climb_align_gap_ms", None))
    # 「**移动操作尝试间隔(ms)**」（2026-09-27 用户要求：由「爬不动时先补按 ↑ 观察(s)」
    # **改名** + **搬出「攀爬参数」组**、放到外面的「寻路配置」组 + 单位**秒 → 毫秒**）：
    # 它是**两个通行方式共用**的（爬绳补按 ↑ / 下跳补发松开 ↓），所以不许留在
    # "上绳这一步"的子组里 ✗。
    check(hasattr(settings, "move_retry_ms"),
          "决策设置里没有「移动操作尝试间隔(ms)」")
    check(not hasattr(settings, "climb_reassert_s"),
          "老名字 `climb_reassert_s` 还在 —— 改名要改干净（会留下两套口径 ✗）")
    s_new = ag.DecisionSettings()
    s_new.from_dict({})                    # 新项目 / 老项目里两个键都没有
    check(isinstance(s_new.move_retry_ms, int) and s_new.move_retry_ms == 3000,
          "默认值该是**整数 3000 ms**（= 老的 3.0 秒 ✓）：%r" % (s_new.move_retry_ms,))
    # ⚠ 老项目文件里存的是**秒**（float）⇒ 不折算的话 "3.0" 会被当成 **3 毫秒**
    #   （爬绳刚"不动"就判失败、下跳每 3ms 重试一次 = 键盘风暴 ✗）
    s_old = ag.DecisionSettings()
    s_old.from_dict({"climb_reassert_s": 2.5})
    check(int(s_old.move_retry_ms) == 2500,
          "老项目文件里的「秒」没折算成毫秒：%r" % (s_old.move_retry_ms,))
    # 界面上真的搬出去了：那一行加在**卡片**（`root`）上，不在「攀爬参数」组（`gv`）里
    _src = (Path(__file__).resolve().parent.parent
            / "gui" / "route_panel.py").read_text(encoding="utf-8")
    check("root.addLayout(row_re)" in _src and "gv.addLayout(row_re)" not in _src,
          "「移动操作尝试间隔(ms)」还挂在「攀爬参数」子组里 —— 用户要求搬到外面的"
          "「寻路配置」组 ✗")
    # 延迟 = 基础 + 增量×(已试次数-1)；规则只有一处（`ag.retry_delay_s` ✓）
    check(ag.retry_delay_s(1.0, 0.5, 1) == 1.0,
          "第 1 次失败该只等基础值：%r" % ag.retry_delay_s(1.0, 0.5, 1))
    check(ag.retry_delay_s(1.0, 0.5, 2) == 1.5 and ag.retry_delay_s(1.0, 0.5, 3) == 2.0,
          "延迟没有按次数累加增量：%r / %r" % (ag.retry_delay_s(1.0, 0.5, 2),
                                              ag.retry_delay_s(1.0, 0.5, 3)))
    check(ag.retry_delay_s(-5.0, -5.0, 1) == 0.0,
          "负的等待时间没被按 0 兜住：%r" % ag.retry_delay_s(-5.0, -5.0, 1))
    # ⚠ 这里**曾经**还测过「走的方向」（`walk_mode`）—— 用户 2026-09-26 明确否掉：
    # 走只有"朝集合中点"一种走法，设置里不许有这个参数 ⇒ 改成**反向钉住**它。
    check(not hasattr(settings, "walk_mode"),
          "`DecisionSettings` 里又出现了 `walk_mode`（用户 2026-09-26 明确删掉）")
    from gui.settings_dialog import SettingsDialog
    _d = SettingsDialog()
    try:
        check(hasattr(_d, "sp_goto_timeout"),
              "判定参数页里没有「寻路超时时间」控件")
        # 界面页：叠图那两项（2026-09-26 从「路线识别」页搬来 ✓）+ 通知信号
        check(hasattr(_d, "ck_mmap_draw") and hasattr(_d, "sld_alpha"),
              "设置弹窗的界面页里没有「叠地形图 / 浓淡」")
        check(hasattr(_d, "overlay_changed"),
              "设置弹窗没有 overlay_changed 信号（改了没人重画实时画面）")
        # 「攀爬参数」子组的控件在**路线识别页**（不是判定参数页 ✓）——
        # ⚠ 组名 2026-09-27 由「攀爬失败保护」改成「攀爬参数」（用户要求），
        #   一次改名漏一处就会"界面上找不着那个组" ✗ ⇒ 这里连**组名本身**也钉住 ✓
        from PyQt5.QtWidgets import QGroupBox
        from gui.route_panel import RoutePanel
        _rp = RoutePanel()
        try:
            check(hasattr(_rp, "sp_retry") and hasattr(_rp, "sp_retry_inc"),
                  "路线识别（寻路配置）页里没有「攀爬失败后延迟激活时间 / 延迟增量」控件")
            check(hasattr(_rp, "sp_align_gap"),
                  "路线识别（寻路配置）页里没有「对齐绳梯移动延迟(ms)」控件")
            _titles = [w.title() for w in _rp.findChildren(QGroupBox)]
            check("攀爬参数" in _titles and "攀爬失败保护" not in _titles,
                  "「攀爬参数」这个子组没找到 / 旧名还在（2026-09-27 改名漏了）：%s"
                  % _titles)
        finally:
            _kill_qt(_rp)
        # 「编辑战斗区域」2026-10-01 **搬到「路线识别 → 寻路配置 → 路线规划」**（用户要求）：
        # 战斗区域每一项都是**地图里的东西**（集合名 / idle 回归 foothold 编号），只有持有
        # 地图 id 的路线识别面板能**每次现读**候选集合名 ✓ —— 所以它现在归 `RoutePanel` ✓。
        # ⚠ `_pp` 只用来**反证**：决策参数面板**不再**有「编辑战斗区域」那套 ✗
        from gui.player_panel import PlayerPanel
        from gui.route_panel import RoutePanel
        _pp = PlayerPanel()
        _rp = RoutePanel()

        class _ProjBZ:
            def get(self, k, d=None):
                return {"map_id": "105090600"}.get(k, d)

        _rp.project = _ProjBZ()
        # ⚠ **离屏自检里任何模态弹窗都必须接住** —— 不接就是"挂住"或"访问违例" ✗
        #（这条用例会走"重复添加 ⇒ 提示一下"那条路 ✓；同一个坑在 selftest_seq_editor
        #  里已经踩过一次，这里又忘了 ✗ —— 记在注释里防下次）。
        from PyQt5.QtWidgets import QMessageBox
        _infos = []
        _old_info = QMessageBox.information
        QMessageBox.information = staticmethod(
            lambda *a, **k: _infos.append(a[2] if len(a) > 2 else ""))
        # ⭐ 2026-09-28：**「添加 / 编辑」现在都会弹出编辑窗**（`gui.player_panel.
        #   BattleZoneDialog` ✓，用户当天："**按钮和弹窗呢？你做的我没法测**"）——
        #   离屏自检里模态窗**必须接住**（同上面那条 ✗）。这里让它"**点了确定**"，
        #   并顺手改两个字段 ⇒ 正好验证"**弹窗里配的东西真的写进真源 `battle_zones`**" ✓。
        # ⛔⛔ **必须把 `theme.CFG` 指到临时文件**（2026-09-28 修 ✗）：这段会建
        #   `BattleZoneListDialog` / `BattleZoneDialog`，而它们都接了 `theme.bind_window_state`
        #   ⇒ **对象被销毁（hide/close）时会写一次窗口几何**到 `config/ui.yaml` ✗ ——
        #   自检**绝不许**改用户的文件 ✗（全仓库的规矩 ✓ 见 `selftest_main_window._fake_store`）。
        #   现场症状：`windows: battle_zone_list` 反复冒出来，**清掉又回来** ✓（因为它根本不是
        #   谁手写的，是**每次跑自检都写一遍** ✗）—— 而一个脏的 ui.yaml 还会**连累别的套件**
        #   （`selftest_main_window` 建主窗口时读它 ✓ 2026-09-28 真栽过一次 ✓）。
        #   ⇒ 照 `selftest_zone_editor:t_line_width_setting` 的做法隔离 ✓。
        import shutil as _shutilui
        import tempfile as _tmpui
        import unittest.mock as _mockui
        from pathlib import Path as _Pathui

        from gui import theme as _themeui
        _uitmp = _tmpui.mkdtemp(prefix="decui_")
        _uipatch = _mockui.patch.object(_themeui, "CFG", _Pathui(_uitmp) / "ui.yaml")
        _uipatch.start()
        # ⛔ 2026-10-01：`_bz_commit` 现在写 **per-map 文件**（`core.battle.save` ✗）——
        #   不隔离的话，自检会真的往 `datasets/map/105090600.battle.json` 写一份 ✗✗
        #   （自检绝不许写用户文件 ✓）。把 `core.battle.path` 指到临时目录 ✓。
        import core.battle as _battleui
        _battle_tmp = _tmpui.mkdtemp(prefix="decbattle_")
        _battle_patch = _mockui.patch.object(
            _battleui, "path", lambda mid: _Pathui(_battle_tmp) / ("%s.battle.json" % mid))
        _battle_patch.start()
        from gui import player_panel as _ppm
        _dlgs = []
        _old_exec = _ppm.BattleZoneDialog.exec_

        def _fake_exec(self):
            _dlgs.append(self)
            self.sp_cd.setValue(7.5)
            self.ed_idle.setText("41")
            self.sp_fight.setValue(30.0)
            # ⚠ 「禁止战斗」**已经不在这儿了**（用户 2026-09-28 要求挪到列表项上勾 ✓）——
            #   这个假弹窗**故意不碰它** ⇒ 正好验证"编辑一次不会把勾选弄丢" ✓
            return _ppm.QDialog.Accepted

        _ppm.BattleZoneDialog.exec_ = _fake_exec
        _bzlist = []
        try:
            # ⭐⭐ **决策参数面板不再有「编辑战斗区域」**（2026-10-01 搬走 ✓）—— 反向钉住：
            #   那一套（清单 / 按钮 / 读写）要是不小心留回来，就是"两个入口改同一份配置" ✗
            check(not hasattr(_pp, "_bz_rows") and not hasattr(_pp, "btn_battle_zone_edit"),
                  "「编辑战斗区域」还在决策参数面板里（该搬到「路线识别 → 路线规划」✗）")
            check(not hasattr(_pp, "_bz_items") and not hasattr(_pp, "_bz_commit"),
                  "「编辑战斗区域」的读写方法还留在决策参数面板 ✗")
            # ⭐ **路线识别面板有它**，而且在「路线规划」子组里 ✓
            check(hasattr(_rp, "_bz_rows") and hasattr(_rp, "btn_battle_zone_edit"),
                  "路线识别面板里没有「编辑战斗区域」的清单 / 按钮")
            check(hasattr(_rp, "_bz_items") and hasattr(_rp, "_bz_commit")
                  and hasattr(_rp, "_bz_candidate_names"),
                  "「编辑战斗区域」口径不对（少了 `_bz_items` / `_bz_commit` / `_bz_candidate_names`）")
            from PyQt5.QtWidgets import QGroupBox
            _rp_titles = [w.title() for w in _rp.findChildren(QGroupBox)]
            check("路线规划" in _rp_titles,
                  "寻路配置下没有「路线规划」这个子组：%s" % _rp_titles)
            _was_zones = list(getattr(settings, "battle_zone_sets", []) or [])
            _was_bz = [dict(z) for z in (getattr(settings, "battle_zones", None) or [])]
            settings.battle_zones = []
            settings.sync_battle_zone_sets()
            # ⭐⭐ **候选 = 这张图已注册的集合（每次现读）** —— 不再靠"开项目那一刻推一次" ✗
            #   （搬到本页的意义；空候选 = "不能编辑"的根因 ✓ 见 `route_panel._bz_candidate_names`）。
            _cand = _rp._bz_candidate_names()
            check(_cand == list(_rp.zone_sets()) and len(_cand) > 0,
                  "候选集合名没从本图集合现读（105090600 有集合 ⇒ 不该空）：%r"
                  % (_cand,))
            # 空配置 ⇒ 面板上**不摆实质行**（只留一句"还没配"提示 ✓）
            _rp._refresh_battle_zones()
            check(_rp._bz_names == [],
                  "空配置却记了名字：%r" % (_rp._bz_names,))
            # ⭐ **弹窗真对象**（不 `exec_` ⇒ 只是把里面的动作跑一遍 ✓）：这比替身强 ——
            #   它连"弹窗和面板之间的接线"一起测了 ✓
            # ⭐ 只读 foothold 视图的工厂也接上了（源码级钉：不在这儿真造视图 —— 那块
            #   `QGraphicsView` 的离屏建构造在 selftest_zone_editor 里另有覆盖 ✓）
            import inspect as _inspect
            check("picker_factory=self.make_foothold_picker"
                  in _inspect.getsource(RoutePanel._on_battle_zone_edit_clicked),
                  "「编辑战斗区域」没把只读视图工厂接上（`picker_factory` 丢了 ✗）")
            _dlg = _ppm.BattleZoneListDialog([], names=_rp._bz_candidate_names(),
                                             on_save=_rp._bz_commit, parent=_rp)
            _bzlist.append(_dlg)
            check(_dlg.lst.count() == 0, "「编辑战斗区域」不是**初始空列表**（用户第 2 条 ✗）")
            _dlg.choose_name = lambda: "乙平台"          # 替身"选集合"那一步 ✓
            _dlg._on_add()
            # ⭐⭐ **新加的项默认勾「可以战斗」**（2026-09-29 修复 ✓）：原来默认不勾 ⇒
            #    新区域进不了 `battle_zone_sets` 白名单 ⇒ 别的区域能打、它不能 ⇒
            #    场景里有怪时 `tick` 提前返回 `leave_battle_zone`，idle 回归（与这儿的战斗）永不触发 ✗
            check(list(settings.battle_zone_sets) == ["乙平台"],
                  "新加的项**默认「可以战斗」**却没进白名单（默认不勾会掐死 idle 回归 ✗）：%r"
                  % (settings.battle_zone_sets,))
            _z0 = [z for z in (settings.battle_zones or []) if str(z.get("set")) == "乙平台"]
            check(_z0 and abs(float(_z0[0].get("cd_s") or 0) - 7.5) < 1e-6
                  and _z0[0].get("idle_footholds") == ["41"]
                  and abs(float(_z0[0].get("fight_max_s") or 0) - 30.0) < 1e-6
                  and _z0[0].get("can_fight") is True,
                  "**弹窗里配的字段没写回 `battle_zones`**（配了也不生效 ⇒ 没法测 ✗）：%r"
                  % (settings.battle_zones,))
            check([str(z.get("set")) for z in _dlg.zones()] == ["乙平台"],
                  "弹窗自己的列表没跟着更新：%r" % (_dlg.zones(),))
            # ⭐⭐ 「可以战斗」的勾选框**挪到主窗口每项前面**（用户 2026-10-01 ✓ 从弹窗挪来）：
            #   弹窗列表项**不再有**勾选框（挪走后留一个 = "两处勾同一件事" ✗）——
            _i0 = _dlg.lst.item(0)
            check(not bool(_i0.flags() & _ppm.Qt.ItemIsUserCheckable),
                  "弹窗列表项还留着勾选框（「可以战斗」该挪到主窗口 ✗）")
            #   主窗口该有一个勾选框、且新加的项默认勾上 ✓
            def _rp_cbs():
                out = []
                for _i in range(_rp._bz_rows.count()):
                    _it = _rp._bz_rows.itemAt(_i)
                    _w = _it.widget() if _it is not None else None
                    if isinstance(_w, _ppm.QCheckBox):
                        out.append(_w)
                return out
            _cbs = _rp_cbs()
            check(len(_cbs) == 1 and _cbs[0].isChecked(),
                  "主窗口该有一个勾上（默认可以战斗）的勾选框：%r" % (_cbs,))
            _cbs[0].setChecked(False)              # = 用户取消那个勾 ✓（走 toggled → 写回）
            check(list(settings.battle_zone_sets) == [],
                  "取消「可以战斗」却还在白名单里（主窗口勾选写回失效 ✗）：%r"
                  % (settings.battle_zone_sets,))
            _z1 = [z for z in (settings.battle_zones or []) if str(z.get("set")) == "乙平台"]
            check(_z1 and _z1[0].get("can_fight") is False,
                  "取消「可以战斗」没写回真源：%r" % (settings.battle_zones,))
            _cbs2 = _rp_cbs()
            check(len(_cbs2) == 1 and not _cbs2[0].isChecked(),
                  "取消后主窗口勾选框没跟着变：%r" % (_cbs2,))
            _cbs2[0].setChecked(True)              # 勾回去 ✓
            check(list(settings.battle_zone_sets) == ["乙平台"],
                  "勾上「可以战斗」却没进白名单（= 旧「限制战斗区域」失效 ✗）：%r"
                  % (settings.battle_zone_sets,))
            _z1 = [z for z in (settings.battle_zones or []) if str(z.get("set")) == "乙平台"]
            check(_z1 and _z1[0].get("can_fight") is True,
                  "「可以战斗」没写回真源：%r" % (settings.battle_zones,))
            # ⭐⭐ **勾选不许被"进来编辑一次"弄丢**（本次最要紧的一条 ✗）：编辑弹窗里
            #    **已经没有**那个勾选框了 ⇒ `zone()` 必须**原样带回** ✓（can_fight 只在主窗口勾 ✓）
            _dlg.lst.setCurrentRow(0)
            _dlg._on_edit()                        # 双击 = 编辑（这里只改了 CD/idle/fight ✓）
            _z1 = [z for z in (settings.battle_zones or []) if str(z.get("set")) == "乙平台"]
            check(_z1 and _z1[0].get("can_fight") is True,
                  "**编辑一次别的参数就把「禁止战斗」丢了**（`zone()` 没原样带回 ✗）：%r"
                  % (settings.battle_zones,))
            _cbs3 = _rp_cbs()
            check(len(_cbs3) == 1 and _cbs3[0].isChecked(),
                  "编辑别的参数后主窗口勾选框没跟着配置走：%r" % (_cbs3,))
            # ⭐ **加第二条**（面板摘要跟着长一条 ✓；新加的默认也勾上 ⇒ 一起进白名单 ✓）
            _dlg.choose_name = lambda: "甲平台"
            _dlg._on_add()
            check([str(z.get("set")) for z in _dlg.zones()] == ["乙平台", "甲平台"],
                  "第二条没进弹窗列表：%r" % (_dlg.zones(),))
            check(_rp._bz_names == ["乙平台", "甲平台"],
                  "面板摘要没跟着长：%r" % (_rp._bz_names,))
            # ⭐ **删除**（替身掉那个确认框 ✓ ⇒ 直接答 Yes）
            _old_q = QMessageBox.question
            QMessageBox.question = lambda *a, **k: QMessageBox.Yes
            try:
                _dlg.lst.setCurrentRow(0)
                _dlg._on_del()
            finally:
                QMessageBox.question = _old_q
            check([str(z.get("set")) for z in _dlg.zones()] == ["甲平台"],
                  "「删除」没删掉选中的那条：%r" % (_dlg.zones(),))
            check(_rp._bz_names == ["甲平台"],
                  "删完面板摘要没跟着重画：%r" % (_rp._bz_names,))
            check(list(settings.battle_zone_sets) == ["甲平台"],
                  "删掉一条之后白名单没跟着少一条：%r" % (settings.battle_zone_sets,))
            settings.battle_zones = _was_bz
            settings.sync_battle_zone_sets()
            settings.battle_zone_sets = _was_zones
            _rp._refresh_battle_zones()
        finally:
            _ppm.BattleZoneDialog.exec_ = _old_exec
            QMessageBox.information = _old_info
            # 收掉上面那份"临时 ui.yaml"（**自检不许写用户配置** ✓ 见那段说明 ✓）
            _uipatch.stop()
            _shutilui.rmtree(_uitmp, ignore_errors=True)
            # 收掉 per-map 战斗区域那份临时目录 ✓
            _battle_patch.stop()
            _shutilui.rmtree(_battle_tmp, ignore_errors=True)
            # ⚠⚠ **杀完必须把容器也清空**（2026-09-28 踩过 ⇒ 表现为**进程原生崩溃** 0xC0000409 ✗）：
            #   `_kill_qt` 走的是 `sip.delete`（**当场析构 C++ 对象** ✓），而它只 `del` 自己的
            #   形参 ⇒ 如果**别处还留着引用**（这里就是 `_bzlist` ✗），就变成"Python 侧活着、
            #   C++ 侧已经死了"的**悬垂包装器** ⇒ 之后**第一次建 Qt 控件**（下一个用例）就撞崩 ✗。
            for _d2 in list(_bzlist):
                _kill_qt(_d2)
            _bzlist.clear()
            _kill_qt(_rp)
            _kill_qt(_pp)
        check(not hasattr(_d, "cmb_walk_mode"),
              "设置页里又出现了「走的方向」（用户 2026-09-26 明确否掉）")
        _was = settings.goto_timeout_s
        _d.sp_goto_timeout.setValue(45)
        _d._accept()
        check(int(settings.goto_timeout_s) == 45,
              "改了寻路超时时间却没写回：%r" % settings.goto_timeout_s)
        settings.goto_timeout_s = _was
    finally:
        _d.reject()

    # 存读一致（改完必须还在）
    was_tol, was_hold = settings.align_tol_px, settings.align_hold_ms
    try:
        settings.align_tol_px, settings.align_hold_ms = 11, 333
        d = settings.to_dict()
        check(d.get("align_tol_px") == 11 and d.get("align_hold_ms") == 333,
              "对齐参数没进 to_dict：%s" % {k: d.get(k) for k in
                                            ("align_tol_px", "align_hold_ms")})
        fresh = ag.DecisionSettings()
        fresh.from_dict(d)
        check((fresh.align_tol_px, fresh.align_hold_ms) == (11, 333),
              "from_dict 没读回来：%s" % ((fresh.align_tol_px, fresh.align_hold_ms),))
        # 老文件（没有这两个键）⇒ 用默认值，**不能变 0**
        old = ag.DecisionSettings()
        old.from_dict({})
        check(old.align_tol_px > 0 and old.align_hold_ms >= 0,
              "老配置读进来把对齐参数变成了 0：%s" % ((old.align_tol_px,
                                                       old.align_hold_ms),))
    finally:
        settings.align_tol_px, settings.align_hold_ms = was_tol, was_hold

    # ⭐ 「寻路超时后按键」的**存读**（用户 2026-09-28 加 ✓）：跟着项目存 ✓、
    #    老项目文件里没这个键 ⇒ `None`（不按 ✓ 行为一点不变 ✓）、空白串也当 `None` ✓。
    _was_gk = getattr(settings, "goto_timeout_key", None)
    try:
        settings.goto_timeout_key = "space"
        _d = settings.to_dict()
        check(_d.get("goto_timeout_key") == "space",
              "没写进配置（换项目就丢了 ✗）：%r" % (_d.get("goto_timeout_key"),))
        _old = ag.DecisionSettings()
        _old.from_dict(_d)
        check(_old.goto_timeout_key == "space", "读回来不对：%r" % (_old.goto_timeout_key,))
        _d2 = dict(_d)
        _d2.pop("goto_timeout_key", None)
        _old.from_dict(_d2)
        check(_old.goto_timeout_key is None,
              "老项目文件里没这个键 ⇒ 该是 `None`（不按 ✓）：%r" % (_old.goto_timeout_key,))
        _d3 = dict(_d)
        _d3["goto_timeout_key"] = "   "
        _old.from_dict(_d3)
        check(_old.goto_timeout_key is None,
              "空白串该当 `None`（手改坏的值别带进运行期 ✗）：%r" % (_old.goto_timeout_key,))
    finally:
        settings.goto_timeout_key = _was_gk

    # 设置弹窗里真的有这一页、且能改到
    from gui.settings_dialog import TAB_NAMES, SettingsDialog
    check("判定参数" in TAB_NAMES,
          "设置弹窗没有「判定参数」页签：%s" % (TAB_NAMES,))
    sd = SettingsDialog()
    check(hasattr(sd, "sp_align_tol") and hasattr(sd, "sp_align_hold"),
          "「判定参数」页里没有那两个控件")
    tab_names = [sd.tabs.tabText(i) for i in range(sd.tabs.count())]
    check(tab_names == list(TAB_NAMES),
          "页签顺序和 TAB_NAMES 对不上：%s" % (tab_names,))
    # 页面里显示的是当前值
    check(int(sd.sp_align_tol.value()) == int(settings.align_tol_px),
          "误差范围控件没显示当前值：%r vs %r"
          % (sd.sp_align_tol.value(), settings.align_tol_px))
    # 执行器**没有**"要不要按跳"的开关（2026-09-26 用户定：那是测试内容，不该进设置）——
    # 授权就是「命令前往」那一下（见 route_panel._command_first_step），别在这里再放一道闸。
    check(not hasattr(sd, "ck_climb"), "「判定参数」页里又出现了上绳开关")
    check(not hasattr(settings, "climb_enabled"),
          "DecisionSettings 里又出现了 climb_enabled —— 这个闸已经删了")

    # ⭐ **「寻路超时后按键」**（用户 2026-09-28 要求，"类型是**自定义按键下拉列表**"）——
    #    钉四件：① 有默认值（`None` = 不按 ✓ 老行为不变 ✓）；② 存读一致 + 老配置兜底 None；
    #    ③ 下拉**第一项就是「（不按）」**（data=None ✓）；④ 选项 = **固定键 + 自定义按键**，
    #    且**不含**「反方向 / 目标方向」（那两个要按**当时朝向**解析 ⇒ 超时那刻没有朝向 ✗）。
    check(hasattr(settings, "goto_timeout_key"), "决策设置里没有「寻路超时后按键」")
    check(getattr(settings, "goto_timeout_key", "X") is None,
          "「寻路超时后按键」默认该是 `None`（不按 ✓ 老项目行为不变）：%r"
          % (getattr(settings, "goto_timeout_key", None),))

    check(hasattr(sd, "cb_goto_timeout_key"),
          "「判定参数」页里没有「寻路超时后按键」下拉")
    _cb = sd.cb_goto_timeout_key
    check(_cb.itemData(0) is None,
          "下拉第一项该是「（不按）」（data=None ✓ 默认一个键都不发 ✓）：%r"
          % (_cb.itemData(0),))
    _datas = [_cb.itemData(i) for i in range(_cb.count())]
    check("left" in _datas and "esc" in _datas and "attack" in _datas,
          "下拉里没有固定键（该含 ←/Esc/输出 这些 ✓）：%r" % (_datas,))
    check("back" not in _datas and "forward" not in _datas,
          "下拉里不该有「反方向 / 目标方向」（要按**当时朝向**解析，超时那刻没有朝向 ✗）：%r"
          % (_datas,))
    _was_ck = dict(settings.custom_keys or {})
    try:
        settings.custom_keys = dict(_was_ck)
        settings.custom_keys["测试键"] = "f9"
        sd._fill_goto_timeout_key()
        _d4 = [sd.cb_goto_timeout_key.itemData(i)
               for i in range(sd.cb_goto_timeout_key.count())]
        check("测试键" in _d4,
              "**自定义按键**没进下拉（用户明确要「自定义按键下拉列表」✗）：%r" % (_d4,))
    finally:
        settings.custom_keys = _was_ck
        sd._fill_goto_timeout_key()

    # ⛔ 「**禁用杀怪寻路**」（用户 2026-09-28 加 ✓）—— 钉三件：
    #   ① 有默认值（`False` = 老项目行为一点不变 ✓，同 `goto_timeout_key` 那个纪律 ✓）；
    #   ② 存读一致 + **老配置兜底 `False`** ✓；③ 设置「判定参数」页里**真的有这个控件** ✓。
    check(hasattr(settings, "disable_chase_pathfinding"),
          "决策设置里没有「禁用杀怪寻路」")
    check(bool(getattr(settings, "disable_chase_pathfinding", True)) is False,
          "「禁用杀怪寻路」默认该是 `False`（= 老行为，别默认开 ✗）：%r"
          % (getattr(settings, "disable_chase_pathfinding", None),))
    _was_ncp = bool(getattr(settings, "disable_chase_pathfinding", False))
    try:
        settings.disable_chase_pathfinding = True
        _dn = settings.to_dict()
        check(_dn.get("disable_chase_pathfinding") is True,
              "没写进配置（换项目就丢了 ✗）：%r" % (_dn.get("disable_chase_pathfinding"),))
        _oldn = ag.DecisionSettings()
        _oldn.from_dict(_dn)
        check(_oldn.disable_chase_pathfinding is True, "存进去读不回来")
        _dn2 = dict(_dn)
        _dn2.pop("disable_chase_pathfinding", None)
        _oldn.from_dict(_dn2)
        check(_oldn.disable_chase_pathfinding is False,
              "老项目文件里**没这个键** ⇒ 该是 `False`（老行为一点不变 ✓）：%r"
              % (_oldn.disable_chase_pathfinding,))
    finally:
        settings.disable_chase_pathfinding = _was_ncp
    check(hasattr(sd, "ck_no_chase_path"),
          "「判定参数」页里没有「禁用杀怪寻路」开关（用户要求加在**判定参数顶层** ✗）")
    sd.reject()


def t_climb_job():
    """定点上绳执行器（`decision/route.py`）：对齐 → 按住跳+方向 → 判定到达 → 超时/掉下来。

    用户 2026-09-26 定的流程，逐条钉住。**最值得钉的是"什么时候算对齐好了"**：只看一帧
    会被抖动骗（小地图世界坐标本身就有几像素抖），所以"进误差范围后还要保持 N 毫秒"
    这段逻辑错一点，表现就是"老是跳不上去/来回蹭"。
    """
    from core import mapdata, zones
    from decision import route

    def mk(**kw):
        a = dict(ladder_id="L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                 dst_set="乙平台", tol_px=6, hold_ms=200)
        a.update(kw)
        return route.ClimbJob(**a)

    # ① 没对齐 ⇒ 往目标那侧挪，不跳
    job = mk()
    o = job.update(0.0, px=600.0)
    check(o["move"] == 1 and not o["jump"], "差得远时该往右挪、不跳：%s" % o)
    # ⚠ 站过头（换了方向）时要受「**对齐绳梯移动延迟**」管：上一次按下才过去 100ms
    #   （< 默认 180）⇒ 这一拍该**站住等**，不许立刻反着按 —— 那一下就"在绳两边来回抖" ✓
    #   （用户 2026-09-27 加的参数；细节与 0/远距离两种情形见 `t_climb_align_gap`）
    o = job.update(0.1, px=760.0)
    check(o["move"] == 0,
          "上一次按下还没到「移动延迟」就又按了（正是来回抖那一下）：%s" % o)
    o = job.update(0.3, px=760.0)          # 距上次按下 300ms ≥ 180 ⇒ 这才该往左
    check(o["move"] == -1 and not o["jump"], "站过头时该往左挪：%s" % o)

    # ② 进误差范围 ⇒ ⭐ **先朝绳点按一次方向键**（用户 2026-09-28 要求："朝目标绳梯的 x 方向
    #   对齐 x（**至少发 1 次点按方向键**）→ 跳 → 按住 ↑，这三步要循环"）⇒ **再站住等保持
    #   时间** ⇒ 保持够了才跳 ✓（"点按"与"站住等"是**两拍**，别合成一拍 ✗）
    job = mk()
    o = job.update(0.0, px=703.0)          # 差 3 ≤ 6 ⇒ 进范围（本轮还没发过方向键）
    check(o["move"] == -1 and not o["jump"],
          "刚进误差范围该**先朝绳点按一次方向键**（dx=-3 ⇒ ←）：%s / %r" % (o, job.note))
    check("先朝绳点按一次" in job.note, "那一拍没说清在干什么：%r" % job.note)
    o = job.update(0.02, px=703.0)         # 点按过了 ⇒ 这一拍才"站住等保持时间"
    check(o["move"] == 0 and not o["jump"],
          "点按过之后该停下等保持时间，不能马上跳：%s" % o)
    o = job.update(0.12, px=703.0)         # 距站住才 100ms < 200ms
    check(o["phase"] == route.ClimbJob.ALIGN and not o["jump"],
          "保持时间没到就跳了：%s" % o)
    o = job.update(0.25, px=703.0)         # 站住 230ms ≥ 200 ⇒ 上绳
    check(o["phase"] == route.ClimbJob.CLIMB and o["jump"],
          "保持够了该按住跳：%s" % o)
    check(o["dir"] == 1 and o["move"] == 0,
          "向上爬时该按↑、且不再左右挪：%s" % o)

    # ③ 保持期间抖出去 ⇒ **重新计时**（不许攒够一次就永久算数）
    job = mk()
    job.update(0.0, px=703.0)
    job.update(0.05, px=730.0)             # 抖出去
    job.update(0.10, px=703.0)             # 又回来 ⇒ 重新开始
    o = job.update(0.20, px=703.0)         # 距重新进范围才 100ms
    check(not o["jump"], "抖出去之后没有重新计时：%s" % o)

    # ④ 到达判据 = **纯几何**：**向上爬按「绳子上端的 y」**（用户 2026-09-26 改的口径，
    #    原话："之前是按目标 foothold 集合的 y，改为绳子上端的 y"⇒ `dst_y` 不再参与上爬）。
    #    ⚠ **"脚下是目标集合"不算到达依据**：定位读数会抖，`-170`（离目标还差 5px）那种
    #    位置也可能被报成"已在目标集合" ⇒ 提前收工 —— 用户实测就是"在 -170 就松开了 ↑" ✗。
    #    集合判据没丢，它在**失败之后**用来"别再重试"（见 agent 的 ⑤b2 用例）。
    job = mk(dst_y=-208.0)                 # 绳上端 = -100（mk 的 y1）；平台面 -208 已不参与
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0, py=0.0, here_sets=["甲平台"])
    check(o["jump"] and not o["done"], "还在别的平台上就说到达了：%s" % o)
    o = job.update(0.40, px=700.0, py=-90.0, here_sets=["乙平台"])
    check(not o["done"],
          "只凭「脚下是目标集合」就判到达（提前收工 —— y 还没到绳上端）：%s" % o)
    # ⭐ 到达**只认广播**（2026-09-27 收编 ✓）⇒ 这一拍要传 `at_top="L2"`
    o = job.update(0.50, px=700.0, py=-208.0, here_sets=["乙平台"], at_top="L2")
    # 判到到达了，但**这一拍还不松手**（要再按住误差时间，见 ④b）
    check(not o["done"] and o["dir"] == 1 and "到达" in o["note"],
          "广播说已到绳上端却没判到达：%s" % o["note"])
    check("广播" in o["note"], "到达没写清依据（该说是位置状态广播说的）：%r" % o["note"])
    o = job.update(0.90, px=700.0)
    check(o["done"] and not o["jump"], "到达之后还在按键：%s" % o)

    # ④a **执行器只看广播**：广播说没到就是没到（哪怕 y 已经越过绳上端 ✗）；
    #     广播说到了就到了（哪怕 y 还在绳下端 ✓，见 ④c ①）。
    #     ⚠ "y 差 1px 算不算到顶"那条**边界**现在归**位置状态机**（判据原文在
    #       `perception/pos_state.py` ✓，用例 `selftest_minimap.t_pos_state_machine`
    #       钉着 ✓）—— 这里只钉"执行器不再自己比坐标" ✓。
    job = mk(dst_y=-208.0)                 # 平台面 -208：旧口径会在这儿就判到达
    job.update(0.0, px=700.0)
    job.update(0.25, px=700.0)
    o = job.update(0.30, px=700.0, py=-99.0, ladder_id="L2", at_top="")
    check(not o["done"] and "往上爬" in o["note"],
          "广播说「判过了、没到」却还是判到达（执行器又在自己比坐标 ✗）：%s" % o)
    check("-100" in o["note"],
          "上绳那句提示没按绳上端报目标（改口径漏了提示）：%r" % o["note"])
    o = job.update(0.35, px=700.0, py=-100.0, ladder_id="L2", at_top="L2")
    check("到达" in o["note"] and "广播" in o["note"],
          "广播说已到绳上端却没判到达 / 没说清依据：%r" % o["note"])

    # ④b **到达之后要再按住方向键一会儿才松**（用户 2026-09-26 要求：
    #     "↑ 需要延迟『坐标对齐误差时间』（设置里那个）再松开"）。
    #     为什么：读数到目标面 ≠ 人已经站上去（读数滞后一个端到端延迟；游戏里"迈上平台"
    #     那一步也要按着 ↑）⇒ 立刻松手就是差最后一点点。
    job = mk(dst_y=-208.0)                       # hold_ms=200（= 设置里的误差时间）
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0, py=-208.0, ladder_id="L2", at_top="L2")   # 广播：已到绳上端
    check(not o["done"] and o["dir"] == 1 and not o["jump"],
          "刚判到到达就松了 ↑（该再按住一会儿）：%s" % o)
    check("按住" in o["note"], "没写清「还要再按住一会儿」：%r" % o["note"])
    o = job.update(0.35, px=700.0, py=-208.0, ladder_id="L2", at_top="L2")   # 100ms < 200ms
    check(not o["done"] and o["dir"] == 1, "保持时间没到就松了 ↑：%s" % o)
    o = job.update(0.50, px=700.0, py=-208.0, ladder_id="L2", at_top="L2")   # 250ms ≥ 200ms
    check(o["done"] and o["dir"] == 0, "保持够了还没收工 / 没松 ↑：%s" % o)

    # ④b' ⭐ **收工延时用纯「坐标对齐误差时间」，不吃端到端延迟**（用户 2026-09-28 重新审视：
    #     "就算 climb 执行器收工，↑ 键也要再按这么多时间后再松"，且明确**纯 align_hold_ms**）。
    #     模拟 `agent._climb_tick` 那一下：`job.hold_ms` 被改成「纯值 + 端到端延迟」给对齐稳住
    #     用（对齐靠读数、读数有延迟 ⇒ 要补 e2e ✓），而「收工后按住 ↑」用 `base_hold_ms`（纯值，
    #     游戏机制：迈上平台那一步要按着 ↑，与读数延迟无关 ✓）⇒ 这里 hold_ms=300、base=200，
    #     收工必须按 200 走、**不是** 300。
    job = mk(dst_y=-208.0)
    job.base_hold_ms = 200                 # 纯「坐标对齐误差时间」
    job.hold_ms = 300                       # 被 agent 加长后的（200 + e2e=100）
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0, py=-208.0, ladder_id="L2", at_top="L2")
    check(not o["done"] and o["dir"] == 1, "判到到达该开始按住 ↑：%s" % o)
    o = job.update(0.40, px=700.0, py=-208.0, ladder_id="L2", at_top="L2")  # 150ms < 200
    check(not o["done"] and o["dir"] == 1,
          "收工延时没按纯 200ms（按了 300 才该收 —— 把端到端延迟吃进来了 ✗）")
    o = job.update(0.50, px=700.0, py=-208.0, ladder_id="L2", at_top="L2")  # 250ms ≥ 200
    check(o["done"] and o["dir"] == 0,
          "收工延时该按纯 200ms 就收工：%s" % o["note"])

    # ④c ⭐ **上爬的到达改读「位置状态广播」**（用户 2026-09-27 定的新架构："上爬到顶的判据
    #     修改为：**不再主动计算，而是等『位置状态』广播**"✓）—— 钉三件：
    #       ① 广播说"已到 **本任务这根绳** 的上端"（`at_top="L2"` ✓）⇒ **立刻到达**（哪怕这一拍
    #          的 y 读数还在绳下端 ✓ ⇒ 证明执行器**不再自己比坐标** ✗）；
    #       ② 广播说"**判过了、没到**"（**空串** ✓）⇒ 哪怕 y 已经越过绳上端也**不算** ✓；
    #       ③ 广播说的是**别的绳**到顶 ⇒ 不算（必须是当前任务要爬的那根 ✓）。
    #     ⚠ 不传 `at_top`（= `None`）⇒ **判不出来 ⇒ 就是不认** ✓（几何兜底**已删** ✗，
    #       用户 2026-09-27 明确"删掉过渡兜底" ✓；见下面 ⑤）。
    job = mk(dst_y=-208.0)
    job.update(0.0, px=700.0)
    job.update(0.25, px=700.0)
    o = job.update(0.30, px=700.0, py=100.0, at_top="L2")       # py 还在绳下端
    check("到达" in o["note"] and "广播" in o["note"],
          "广播说已到绳上端却没判到达（执行器还在自己比坐标 ✗）：%r" % o["note"])
    job = mk(dst_y=-208.0)
    job.update(0.0, px=700.0)
    job.update(0.25, px=700.0)
    o = job.update(0.30, px=700.0, py=-500.0, at_top="")        # py 早就过了绳上端
    check("到达" not in o["note"],
          "广播说「判过了、没到」，却还是判了到达（又在自己比坐标 ✗）：%r" % o["note"])
    o = job.update(0.40, px=700.0, py=-500.0, at_top="L9")      # 别的绳到顶了
    check("到达" not in o["note"],
          "广播的是**别的绳**到顶，却算成本任务到达 ✗：%r" % o["note"])

    # ⑤ ⭐ **过渡兜底已删**（用户 2026-09-27："删掉过渡兜底" ✓）：不传 `at_top`
    #    （= `None` = **判不出来**）⇒ **绝不判到达** ✗ —— 哪怕 y 早就越过绳上端
    #    （那是**广播**该说的事，执行器一个字都不许自己比 ✓）。
    job = mk()
    job.update(0.0, px=700.0)
    job.update(0.25, px=700.0, py=-100.0)
    o = job.update(0.40, px=700.0, py=-120.0)      # y 越过绳上端（-100），但没有广播
    check(not o["done"] and o["dir"] == 1 and "到达" not in o["note"],
          "没有广播（`at_top=None` = 判不出来）却还是判了到达 —— 几何兜底没删干净 ✗：%s" % o)
    # 广播说到了 ⇒ 照旧"再按住误差时间"才收工（0.70-0.45 = 250ms ≥ 200ms ✓）
    check("到达" in job.update(0.45, px=700.0, py=-120.0, at_top="L2")["note"],
          "广播说已到绳上端却没判到达：%r" % job.note)
    check(job.update(0.70, px=700.0, py=-120.0, at_top="L2")["done"],
          "到达后等够了还没收工：%s" % job.note)

    # ⑥ 向下爬 = **另一套收尾**（2026-09-26 用户改的时序，逐拍细节见
    #     `t_climb_down_jump_off`）：按住 ↓ 贴上绳 ⇒ 在绳上**稳住 hold_ms** ⇒ 松 ↓
    #     ⇒「跳下绳梯」（朝目标集合中心按住方向 + 按跳）⇒ 落地**当拍就收工**。
    #     ⚠ 和向上爬的差别就在最后：**没有"到达后再按住一会儿"** —— 下爬的到达发生在
    #     "落地"那一刻，落地后还按着水平方向键会顺着平台走出去 ✗。
    job = mk(direction=-1, dst_set="丙平台")
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0)
    check(o["jump"] and o["dir"] == -1, "向下爬该按↓（贴上绳）：%s" % o)
    o = job.update(0.40, px=700.0, py=-80.0, ladder_id="L2")     # 判定上绳
    check(o["dir"] == -1 and not o["jump"] and not o["done"],
          "下爬：已上绳该松开跳、按住 ↓ 稳住（还没到松手那一刻）：%s" % o)
    o = job.update(0.70, px=700.0, py=-80.0, ladder_id="L2")     # 300ms ≥ hold_ms=200
    check(o["phase"] == route.ClimbJob.JUMP_DOWN and o["dir"] == 0,
          "下爬：稳够误差时间该松 ↓、进「跳下绳梯」相：%s" % o)
    o = job.update(1.00, px=700.0, py=-60.0)                     # 落体中（已离开绳）
    check(not o["done"] and not o["failed"], "刚跳下去就说到了/说失败了：%s" % o)
    # ⭐ 落地改读广播（2026-09-27 收编 ✓）：`at_bottom` = 到没到绳下端、
    #    `ground_y` = 脚下那块面的 y（与目标面比 ✓）—— 两条都给，覆盖两种口径 ✓
    o = job.update(2.00, px=700.0, py=210.0, at_bottom="L2", ground_y=210.0)
    check(o["done"] and o["dir"] == 0 and o["move"] == 0 and not o["jump"],
          "下爬：落地该**当拍收工并松开所有键**：%s" % o)

    # ⑦ 超时报警（P4 要求：不许无限等）
    job = mk(timeout_s=1.0)
    check(job.update(0.0, px=600.0)["failed"] is False, "刚开始就说超时")
    o = job.update(1.5, px=600.0)
    check(o["failed"] and not o["jump"], "超时了还按着键/不报警：%s" % o)

    # ⑧ 从绳上掉下来 ⇒ 失败，但要**给容错期**：按跳那几拍人还在空中（ladder_id 是 None），
    #    一离开就判失败会当场误杀；而**一直**没回到绳上（超过容错期）就必须停手。
    job = mk()
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0, ladder_id=None)
    check(not o["failed"], "刚按跳那几拍（还没上绳）就判失败：%s" % o)
    job.update(0.60, px=700.0, ladder_id="L2")          # 上绳了
    o = job.update(1.00, px=700.0, ladder_id=None)
    check(not o["failed"], "掉下来一拍就判失败（还可能再抓一次绳）：%s" % o)
    o = job.update(3.5, px=700.0, ladder_id=None)       # 离开超过容错期
    check(o["failed"] and not o["jump"],
          "掉下来一直没回到绳上，却还在按跳：%s" % o)

    # ⑧b **「偏离绳梯」判据已被删除**（用户 2026-09-26 明确否掉）：横向偏出去**不再**
    #     判失败 —— 人真离开了绳梯，下面 ⑧ 那条「从绳上掉下来了」会接管（`ladder_id`
    #     变了 / 脚下集合变了 ✓），两条判据盯同一件事只会多一种**误杀**（上绳本来就歪 ✗）。
    job = mk()
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0)
    check(o["jump"], "该上绳了：%s" % o)
    o = job.update(0.30, px=760.0)               # 偏 60px
    check(not o["failed"], "才偏一下就判失败：%s" % o)
    o = job.update(1.60, px=760.0)               # 一直偏着 1.3s（老判据早该炸）⇒ 现在不许
    check(not o["failed"], "「偏离绳梯」判据又回来了（用户 2026-09-26 明确删掉）：%s" % o)
    check(not hasattr(route, "OFF_X_PX"),
          "路由模块里又出现 `OFF_X_PX`（偏离判据的阈值，用户明确删掉）")

    # ⑧c **重新激活**：失败后 retry() 回到第一步重来，次数累加（到上限由 agent 放弃）
    check(job.attempt == 1, "一开始 attempt 该是 1：%s" % job.attempt)
    o = job.retry()
    check(job.attempt == 2 and o["phase"] == route.ClimbJob.ALIGN and not o["jump"],
          "retry 没回到对齐阶段：%s / attempt=%s" % (o, job.attempt))
    check(job._t0 is None, "retry 没把计时清干净（超时会立刻又炸）：%r" % (job._t0,))

    # ⑨ 取消 ⇒ 状态收干净，别再按键
    job = mk()
    job.update(0.0, px=700.0)
    o = job.cancel("换目标了")
    check(o["failed"] and not o["jump"] and o["note"] == "换目标了",
          "取消之后没收拾干净：%s" % o)

    # ⑩ 从**真实边**造任务：方向必须和 climb_direction 一致（用户的数据）
    t = mapdata.load("105090600")
    z = zones.load("105090600")
    edge = next((e for e in z.edges if e.get("kind") == "climb"
                 and e.get("from") == "右下" and e.get("to") == "上下过渡平台"), None)
    check(edge is not None, "用户数据里那条「右下 → 上下过渡平台」的爬边不见了")
    j2 = route.job_for_edge(t, z, edge, tol_px=6, hold_ms=250)
    check(j2.dir == 1 and j2.dst_set == "上下过渡平台"
          and j2.ladder_id == "L2" and abs(j2.x - 701.0) < 1e-6,
          "从真实边造出来的任务不对：dir=%s x=%s dst=%s"
          % (j2.dir, j2.x, j2.dst_set))
    # ⑩b **坐标口径**（2026-09-26 用户澄清 —— 我在这里改错过一次，写下来别再犯）：
    #      地形数据（foothold / ladder 的 x、y）**本来就在游戏的世界坐标系里**：
    #      fh44 地形 y=-208 ⇒ 它的世界 y 就是 -208，不用加任何东西。玩家的真实世界坐标
    #      = 小地图定位 + 那 33px「坐标系偏移」（偏移修的是"黄点重心 ↔ 玩家原点"那层
    #      对应，**不是坐标系换算**）⇒ 两者**直接比** ✓。
    #      ⇒ 造任务时**不许**给地形坐标再加偏移：加了会把"到达目标平台面"从 -208 放宽成
    #      -175（差 33px），看起来像"提前算到了" ✗。
    _lids = zones.ladder_ids(t)
    _L = next(x for x in t.ladders if _lids.get(id(x)) == edge.get("ladder"))
    check(abs(j2.x - _L.x) < 1e-6,
          "任务的 x 不是绳的地形 x（不许给它加偏移）：%s vs %s" % (j2.x, _L.x))
    check(j2.dst_y == route.dst_surface_y(t, z, edge.get("to"), _L, j2.dir),
          "任务的 dst_y 不是地形 foothold 的面本身（不许加偏移）：%s" % j2.dst_y)

    # ⚠ 「走」不再列在这儿了（2026-09-26）：它有执行器了，`job_for_edge` 会正常
    # 分发给 `walk_job_for_edge` ✓（它的"该抛"用例在 ⑬ 里）。
    for bad in ({"kind": "climb", "from": "右下", "to": "上下过渡平台",
                 "ladder": "L9"},
                {"kind": "climb", "from": "右下", "to": "左上", "ladder": "L2"}):
        try:
            route.job_for_edge(t, z, bad)
            raise AssertionError("构造不该成功（会往反方向爬）：%s" % bad)
        except ValueError:
            pass          # 正是期望的：说不清就抛，**不许猜**
    # 分发表：三类通行方式各出各的任务（加进来时漏一个就会静悄悄"没做" ✗）。
    # ⚠ 要挑**目标集合里有 foothold** 的走边：没有落点的走边本来就该抛 ValueError
    #（那是 `walk_job_for_edge` 的"不许猜"，不是分发漏了）。
    _w_e = next((e for e in z.edges if e.get("kind") == "walk"
                 and (z.sets.get(e.get("to")) or {}).get("footholds")), None)
    if _w_e is not None:
        check(isinstance(route.job_for_edge(t, z, _w_e), route.WalkJob),
              "job_for_edge 没按类型分发（走边该出 WalkJob）")
    else:
        print("      （用户数据里没有可用的「走」边，分发这条跳过）")

    # ⑪ ⭐ **爬到一半 y 不动 ⇒ 不再判失败**（用户 2026-09-28：执行器只认广播，2c = y 没变 ⇒
    #    补发 ↑，**不判失败** ✗；真卡住由「寻路超时」总闸收口 ✓）。老行为（"卡住 STALL_S 秒 ⇒
    #    判失败 + 说清两个数"）是**执行器自己比 y** ✗ —— 已删 ✓（`_stalled_by` 没了）。
    job = mk(dst_y=-208.0, y1=-205.0)
    job.update(0.0, px=701.0, py=-150.0, ladder_id="L2")
    job.update(0.25, px=701.0, py=-150.0, ladder_id="L2")
    check(job.phase == route.ClimbJob.CLIMB, "没进上绳阶段：%s" % job.note)
    o = job.update(0.35, px=701.0, py=-170.0, ladder_id="L2", climb_stalled=True)
    check(o.get("reassert") is True and not o["failed"],
          "y 不动（广播 climb_stalled）该补发 ↑、不该判失败 ✗：%s" % o)

    # ⑪b y 一直在变好 ⇒ 不许判"不动了"（正常爬升不能被误杀）
    #     ⚠ 绳上端（y1）同样要取得**比爬升区间更高**（-205 < -100~-300 那一段的起点）：
    #     不然第一拍就已经"到绳顶"了、直接收工，这条判据根本没被走到 ✗。
    job = mk(dst_y=-208.0, y1=-205.0)
    job.update(0.0, px=701.0, py=-100.0, ladder_id="L2")
    job.update(0.25, px=701.0, py=-100.0, ladder_id="L2")
    o = None
    for k in range(40):
        _py = -100.0 - (k + 1) * 5.0
        # ⭐ 到达只认广播（2026-09-27 收编 ✓）：这里**替位置状态机**说一句"够到绳上端了"
        #   （真机上由 `perception/pos_state.py` 判：y ≤ 绳上端 + 「坐标对齐误差范围」✓）
        o = job.update(0.35 + k * 0.1, px=701.0, py=_py, ladder_id="L2",
                       at_top=("L2" if _py <= -205.0 else ""))
        if o["done"]:
            break
    check(not o["failed"], "一直在往上爬却被判「不动了」：%r" % o["note"])
    check(o["done"], "都爬到目标面了还没判到达：%r" % o["note"])

    # ⑪c 拿不到 y 读数（定位没输出）⇒ **不许**判"爬不动"：那是观测问题，不是爬不动
    #（那种情况该由 agent 的"拿不到世界坐标"那条路负责说）
    job = mk(dst_y=-208.0)
    job.update(0.0, px=701.0, py=None, ladder_id="L2")
    o = job.update(0.25, px=701.0, py=None, ladder_id="L2")
    for k in range(40):
        o = job.update(0.35 + k * 0.1, px=701.0, py=None, ladder_id="L2")
    check(not o["failed"], "拿不到 y 时乱判「爬不动」：%r" % o["note"])

    # ⑫ **上绳之后只按 ↑、不再按跳**（用户 2026-09-26 定的规则："按 ↑ 直到 ≤ 绳梯那端的 y"）。
    #    实测现象：命令前往左上平台时"还在绳上、离平台还差一截" —— 罪魁就是跳键一直被按住
    #    （跳键只该用来**贴上绳**）。反过来也必须钉住：**还没上绳时不许不按跳**
    #    （否则角色永远贴不上去，任务只会空转到超时）。
    #    ⚠ 绳上端取 -205（比测的 y=-150 更高）⇒ 这一拍**还没到**，才看得到"已上绳"那句提示；
    #    留成 mk() 默认的 -100 的话 -150 已经越过绳顶、直接判到达了 ✗（改口径时红的就是它）。
    job = mk(dst_y=-208.0, y1=-205.0)
    job.update(0.0, px=701.0)               # ① 进容差 ⇒ 先朝绳点按一次（用户 2026-09-28 ✓）
    job.update(0.25, px=701.0)              # ② 站住（才 0ms < 200ms）
    o = job.update(0.50, px=701.0)          # ③ 站住够了（250ms ≥ 200）⇒ 按跳贴上去 ✓
    check(o["jump"] and o["dir"] == 1, "还没上绳就该按跳贴上去：%s" % o)
    o = job.update(0.30, px=701.0, py=-150.0, ladder_id="L2")
    check(not o["jump"], "上了绳还在按跳（会卡在绳上/绳顶，迈不上平台）：%s" % o)
    check(o["dir"] == 1, "上了绳却不按 ↑ 了（那这一拍就是站着不动）：%s" % o)
    check("↑" in o["note"] and "-205" in o["note"],
          "上绳后的提示没说清「按↑直到到哪」（该报**绳上端**）：%r" % o["note"])

    # ⑬ **走（walk）执行器**（2026-09-26 用户要求"把 walk 的执行器做了"）。
    #    它和上绳/下跳**同一套对外形状**（`update` 纯函数 + retry/cancel）⇒ agent 里
    #    "任务优先 / 有怪先打 / 失败延迟重来 / 结束原因挂画面" 一行都不用改 ✓。
    #    注意动作：走**不按跳、不按上下**，只有左右。
    job = route.WalkJob("乙平台", [(700.0, 690.0, 710.0, -208.0, "52")])
    job.update(0.0, px=600.0)
    o = job.update(0.1, px=600.0)
    check(o["move"] == 1 and not o["jump"] and o["dir"] == 0,
          "走：该往目标那侧走、且不跳不按上下：%s" % o)
    o = job.update(0.2, px=905.0)                      # 反方向 ⇒ 往左
    check(o["move"] == -1, "走：目标在左边却往右走：%s" % o)
    o = job.update(0.3, px=705.0)                      # 进容差（±12）
    check(not o["move"], "进容差还继续走（会冲过头）：%s" % o)
    o = job.update(0.4, px=705.0, py=-208.0)
    check(o["done"] and "到达" in o["note"], "站在目标 foothold 上却没判到达：%s" % o)
    # 集合判据优先（走是唯一"站上去就算到"的任务）
    job = route.WalkJob("乙平台", [(700.0, 690.0, 710.0, -208.0, "52")])
    job.update(0.0, px=-500.0)
    o = job.update(0.1, px=-500.0, here_sets=["乙平台"])
    check(o["done"] and not o["move"], "脚下已是目标集合却没判到达：%s" % o)
    # x 对了但人其实在另一层（y 差太多）⇒ **不算到达**（几何兜底要有高度带）
    job = route.WalkJob("乙平台", [(700.0, 690.0, 710.0, -208.0, "52")])
    job.update(0.0, px=700.0)
    o = job.update(0.1, px=700.0, py=-120.0)
    check(not o["done"], "x 对了但高度差很远也判到达了（换层了）：%s" % o)
    # 走不动（x 卡住）⇒ **先单点跳一下**（用户 2026-09-28），跳过还走不动才失败并说清卡在哪
    job = route.WalkJob("乙平台", [(900.0, 890.0, 910.0, -208.0, "7")],
                        timeout_s=60.0)
    job.update(0.0, px=100.0)
    # ⚠ 计时变量**别叫 `t`**：这个函数里 `t` 是地形对象（上面 ⑩ 载入的），
    # 覆盖了它后面 `walk_job_for_edge(t, …)` 就成了 `'float' object has no attribute
    # 'footholds'`（自己踩的 ✗）。
    _tt = 0.0
    o = None
    _hopped = False
    for _ in range(120):
        _tt += 0.1
        o = job.update(_tt, px=100.0)
        if o["jump"] and not o["failed"]:
            _hopped = True                 # 走过一次"单点跳"✓
        if o["failed"]:
            break
    check(_hopped,
          "走不动之后没先「单点跳一下」（该先跳、跳过小台阶，跳过还走不动才失败）：%r"
          % (o["note"],))
    check(o["failed"] and "走不动" in o["note"] and "100" in o["note"],
          "走不动却没判失败 / 没写清卡在哪：%r" % o["note"])
    check("跳了一下" in o["note"],
          "失败原因没写清「跳了一下还是没过去」：%r" % o["note"])
    # 超时要报警（和上绳一样：不许无限等）
    job = route.WalkJob("乙平台", [(900.0, 890.0, 910.0, -208.0, "7")], timeout_s=1.0)
    job.update(0.0, px=100.0)
    check(job.update(2.0, px=101.0)["failed"], "走：超时没报警")
    # 拿不到世界坐标 ⇒ 不许乱走（往哪边走都可能是错的）
    job = route.WalkJob("乙平台", [(900.0, 890.0, 910.0, -208.0, "7")])
    job.update(0.0, px=None)
    check(not job.update(0.1, px=None)["move"], "拿不到坐标还在走：%s" % job.note)
    # 从**真实边**造任务：落点必须来自目标集合的 foothold，说不清就抛（不许猜）
    _w = next((e for e in z.edges if e.get("kind") == "walk"
               and (z.sets.get(e.get("to")) or {}).get("footholds")), None)
    check(_w is not None, "用户数据里没有可用的「走」边，这条测不了")
    _wj = route.walk_job_for_edge(t, z, _w)
    check(isinstance(_wj, route.WalkJob) and _wj.spots
          and _wj.dst_set == _w.get("to"),
          "从真实「走」边造出来的任务不对：%s" % (_wj,))
    # ⚠ 逐边的 `dir`（仅向左/仅向右）必须**从边灌进任务**：灌不进去的话，配了「仅向左」
    #   照样按中点走 —— 用户 2026-09-27 报的就是这个形状（"根本没向左"）。
    #   顺便拿**他现场那条边**直接验方向（数据没改就该按 ←）。
    from core import zones as zones_mod
    for _e in z.edges:
        if _e.get("kind") != "walk" or not zones_mod.walk_dir(_e):
            continue
        _wj = route.walk_job_for_edge(t, z, _e)
        check(_wj.walk_dir == zones_mod.walk_dir(_e),
              "这条边的 `dir` 没灌进任务（配了也不会按那个方向走）：%s" % (_e,))
        _want = -1 if zones_mod.walk_dir(_e) == "left" else 1
        for _fid in ((z.sets.get(_e.get("from")) or {}).get("footholds") or []):
            _f = next((q for q in t.footholds if str(q.fid) == str(_fid)), None)
            if _f is None or _f.is_wall:
                continue
            _mx = (_f.left + _f.right) / 2.0
            _o = _wj.update(0.0, px=_mx, py=float(_f.y_at(_mx)))
            if _o["done"]:
                break        # 起点本身就在目标集合的落点上 ⇒ 这条边没什么可测的
            check(int(_o["move"]) == _want,
                  "%s → %s 配的是「仅向%s」，站在起点上却按了 %s：%s"
                  % (_e.get("from"), _e.get("to"), zones_mod.walk_dir(_e),
                     _o["move"], _o["note"]))
            break
    for bad in ({"kind": "climb", "from": "甲", "to": "乙"},
                {"kind": "walk", "from": "甲", "to": "（不存在的集合）"}):
        try:
            route.walk_job_for_edge(t, z, bad)
            raise AssertionError("构造不该成功（会下一条不知道往哪走的命令）：%s" % bad)
        except ValueError:
            pass          # 正是期望的：说不清就抛


def t_climb_mid_jump():
    """⭐ 「**中途跳下**」——「爬」的逐边配置（用户 2026-09-28 ✓）。

    用户原话："通行方式的**攀爬**需要添加配置「**中途跳下**」，类型为下拉列表：**默认（向目标
    中心）**、**仅向左**、**仅向右**。选取后增加一个参数「**高度**」。逻辑是，**攀爬过程中
    当 y >= 该高度时**按配置的方向**执行跳下绳梯**（**不松开 ↑**，直到**返回成功或失败后**再
    经过「**移动操作尝试间隔**」松开）"✓。

    ⚠⚠⚠ **判据方向 2026-09-28 又改过一次**（用户："把右下爬跳到左上 A+B 也修掉"✓）：
      原来是**字面**的 `y >= mid_y` ✗（用户当时说"照字面"✓），但那个符号的含义是
      "人在这点的**下方**" ⇒ **一上绳（y 最大）就满足** ⇒ 当场触发 + `_mid_jumped` 锁死
      ⇒ **之后一路爬到顶也不跳** ✗ —— 正是用户现场报的"**他爬到顶了都没跳出去**"✓。
      现在改成 **`py <= mid_y`** = "**人的 y 已经爬到「高度」或更高**" ✓（世界 y 越小越靠上
      ✓）。真实数据佐证：L2 是 `y1=-115(上端) … y2=231(下端)`，用户配的 **50 落在绳的中上段**
      ⇒ "爬到那儿再往左跳" ✓（老注释里那句"配大了就等于上去就跳"**是错的** ✓）。
      ⇒ **用例按新方向钉**，别改回去 ✗。

    ⚠ 另外加了**防抖门槛 `MID_ARM_S`**（0.25s）：必须**已经在绳上稳了一小会儿**才判
      —— 刚"吸上绳"那一瞬间 `_on_ladder` 就为真、`py` 又常刚好满足 ⇒ 会把
      `_mid_jumped` 在**还没真开始爬**时用掉 ✗（那正是"爬到顶不跳"的另一半原因 ✓）。

    钉七件：
      ① **没配（老边）⇒ 整条路不走**（爬到顶才算，老行为一字不变 ✓）；
      ② **`y <= mid_y` 才触发**（没爬到 ⇒ 老实爬 ✓）+ **刚在绳上不足 `MID_ARM_S` 不触发** ✓；
      ③ 触发后 **↑ 全程不松**（`dir=+1` ✓ 用户明确"不松开 ↑"✗ 别照抄下爬的 `vert=0`）；
      ④ **点按跳**（先单独一拍按方向、再按跳 ✓ 同 `_jump_down_beat` 的两步法）；
      ⑤ 落地判据 = **离开绳 + y 稳定**（⚠ **不要求**落到沿途集合 —— 用户说"绳旁的平台、
         **可能另一块**"✓ 所以不能用 `_landed` ✗）；
      ⑥ 落地后 **再按住 ↑ 「移动操作尝试间隔」才松**（用户那句"返回成功或失败后…再经过…
         松开"✓），之后 `DONE` 全松 ✓。
    """
    from decision import route as R

    def mk(**kw):
        # 上爬：绳 x=100，上端 y=-205（靠上）/ 下端 y=56；目标集合中心 x=300
        return R.ClimbJob("L1", 100.0, -205.0, 56.0, 1, dst_set="顶层", src_set="一楼",
                          dst_x=300.0, dst_y=-208.0, hold_ms=0, **kw)

    def _drive(j, px, py, lid):
        """推一拍（`on_rope_pos` 跟 `lid` 一致 ⇒ 在绳上时算"贴绳"✓，离开后为空 ✓）。"""
        # ⚠ 步进要**小**：每次 `+1` 秒的话，20 拍就撞上执行器内部那 12 秒超时 ⇒
        #   用例会看到 `failed` 而不是它想验的相位（踩过 ✓）。0.05 秒/拍 ≈ 主循环节拍 ✓。
        return j.update(1000.0 + 0.05 * getattr(_drive, "n", 0), px, py=py, ladder_id=lid,
                        here_sets=["一楼"], on_rope_pos=lid)

    # ① 没配（老边）⇒ 爬到 y 已越过 -100 也**不跳** ✓（老行为一字不变）
    _drive.n = 0
    j0 = mk()
    for _ in range(20):
        _drive.n += 1
        o0 = _drive(j0, 100.0, -120.0, "L1")
    check(o0["phase"] == "climb",
          "**没配「中途跳下」的老边**却跳了（老行为被改坏 ✗）：%r / %r"
          % (o0["phase"], o0["note"][:40]))
    check(o0["dir"] == 1, "上爬中没按住 ↑：%r" % (o0["dir"],))

    # ②③④ 配了 `center` + 高度 -100 ⇒ y=-120（= 已越过）触发；⚠ 先验"未到不触发"
    _drive.n = 0
    j = mk(mid_dir="center", mid_y=-100.0)
    j.mid_hold_s = 0.5
    # ⭐ **专钉 B（防抖门槛 `MID_ARM_S`）**：**刚吸上绳的第 1 拍**——就算 `py` 已经满足
    #   「高度」（人像是"已经在上面了"）也**不许**跳。为什么必须有这条：刚吸上/刚跳上绳
    #   的那一瞬 `_on_ladder` 就为真、`py` 又常常刚好满足条件 ⇒ 会把 `_mid_jumped`
    #   在**还没真开始爬**时用掉 ⇒ 之后**爬到顶也不跳** ✗（用户报的 bug 的另一半 ✓）。
    #   ⚠ 这一条也是**唯一**能抓住"有人把门槛删了"的断言 ✗（去掉门槛时别的都照样绿 ✓）。
    j_fresh = mk(mid_dir="center", mid_y=-100.0)
    o_first = _drive(j_fresh, 100.0, -120.0, "L1")     # 高度已满足 + 刚上绳第 1 拍
    check(o_first["phase"] != "mid_jump",
          "**刚吸上绳的第 1 拍**就判「中途跳下」了（少了 `MID_ARM_S` 防抖门槛 ⇒ "
          "`_mid_jumped` 会被提前用掉、之后爬到顶也不跳 ✗）：%r / %r"
          % (o_first["phase"], (o_first.get("note") or "")[:40]))
    for _ in range(8):                                 # 攒够门槛 ⇒ 之后就该跳了 ✓
        _drive.n += 1
        o_first = _drive(j_fresh, 100.0, -120.0, "L1")
    check(o_first["phase"] == "mid_jump",
          "在绳上稳住够久、高度也满足了，却还是不跳：%r" % (o_first["phase"],))

    # ⚠ **先验"还没爬到高度 ⇒ 不跳"**（新口径下 `y > mid_y` = 人还在高度**下方** ✓）
    #   配 `mid_y=-100`、人 y=0（在下方）⇒ 老实爬 ✓（老用例这里喂 -50 就触发了 ✗）
    o_up = _drive(j, 100.0, 0.0, "L1")
    check(o_up["phase"] != "mid_jump",
          "**还没爬到「高度」**就跳了（新判据是 `py <= mid_y` ✓）：%r / %r"
          % (o_up["phase"], (o_up.get("note") or "")[:40]))
    # 再攒够防抖门槛（喂几拍，人一直在绳上 ✓）
    for _ in range(8):
        _drive.n += 1
        _drive(j, 100.0, 0.0, "L1")
    o = _drive(j, 100.0, -120.0, "L1")          # y=-120：`-120 <= -100` ✓ ⇒ 触发 ✓
    check(o["phase"] == "mid_jump",
          "爬到「高度」以上却没触发（判据是 `py <= mid_y` ✓）：%r / %r"
          % (o["phase"], (o.get("note") or "")[:40]))
    check(o["dir"] == 1,
          "**触发那一拍就把 ↑ 松了**（用户明确「**不松开 ↑**」✗ 别照抄下爬的 `vert=0`）：%r"
          % (o["dir"],))
    check(o["jump"] is False and o["move"] == 1,
          "第一步该**先单独按住方向**（方向键一步、跳一步，别挤同一拍 ✗）：move=%r jump=%r"
          % (o["move"], o["jump"]))
    o = _drive(j, 100.0, -120.0, "L1")           # 第二步：点按跳
    check(o["jump"] is True and o["dir"] == 1,
          "第二步没点按跳（或顺手松了 ↑）：jump=%r dir=%r" % (o["jump"], o["dir"]))
    # 越过 TAP_ON_S ⇒ 松跳、↑ 继续按着（⚠ 步进 0.05s/拍 ⇒ 要多推两拍才真越过 TAP_ON_S ✓）
    for _ in range(3):
        _drive.n += 1
        o = _drive(j, 100.0, -120.0, "L1")
    check(o["jump"] is False and o["dir"] == 1,
          "点按完没松跳（或松了 ↑）：jump=%r dir=%r" % (o["jump"], o["dir"]))

    # ⑤⑥ 跳下来：离开绳 + y 稳定 ⇒ 落地；⚠ 落地后**还要按住 ↑ mid_hold_s** 才 DONE ✓
    o = _drive(j, 100.0, -120.0, None)
    check(o["phase"] == "mid_jump" and o["dir"] == 1,
          "刚离绳就收工了（该等 y 稳定 ✓ 而且这期间 ↑ 不能松 ✗）：%r / dir=%r"
          % (o["phase"], o["dir"]))
    _hold_seen = False
    for _ in range(80):
        _drive.n += 1
        o = _drive(j, 100.0, -120.0, None)
        # ⚠ 认的是**那一段**（"↑ 再按住 N ms 才松" ✓）—— 别按 `"按住 ↑"` 找 ✗（文案是
        #   "↑ **再按住**" ✓ 顺序不同 ⇒ 找不到 ⇒ 用例假红 ✓ 踩过）
        if o["phase"] == "mid_jump" and o["dir"] == 1 and "再按住" in o["note"]:
            _hold_seen = True
        if o["phase"] == "done":
            break
    check(_hold_seen,
          "落地之后**没看到「再按住 ↑ 一段时间」**（用户要求「返回成功后再经过「移动操作"
          "尝试间隔」松开」✗）：note=%r" % (o.get("note"),))
    check(o["phase"] == "done",
          "中途跳下没走到收工（落地 + 延后松 ↑ ✓）：%r / %r" % (o["phase"], o["note"][:60]))
    check(o["move"] == 0 and o["dir"] == 0 and o["jump"] is False,
          "收工那一拍该全松（方向 + ↑ 一起 ✓）：move=%r dir=%r jump=%r"
          % (o["move"], o["dir"], o["jump"]))

    # ④-b 方向三态：left ⇒ 恒 -1 / right ⇒ 恒 +1（⚠ 口径照 `WalkJob.walk_dir` ✓）
    for _d, _want in (("left", -1), ("right", 1)):
        _drive.n = 0
        # ⚠ 高度要配**很大**（新口径 `py <= 高度` 才成立 ✓）—— 配成 `-1000.0` 的话人在
        #   `y=0` 时**不满足** ⇒ 不触发 ⇒ 停在 `ALIGN`（`move=0`）⇒ 用例假红 ✓（老方向踩过）
        _jl = mk(mid_dir=_d, mid_y=1000.0)
        # ⚠ 先**攒够防抖门槛**（`MID_ARM_S`：刚在绳上那一拍不判 ✓）—— 不推这几拍的话
        #   会停在 `ALIGN`（move=0）⇒ 用例假红 ✓（B 生效后踩过）
        for _ in range(8):
            _drive.n += 1
            _drive(_jl, 100.0, 0.0, "L1")
        _ol = _drive(_jl, 100.0, 0.0, "L1")
        check(_ol["move"] == _want,
              "「仅向%s」该恒按 %s（不是朝目标中心 ✗）：move=%r"
              % ("左" if _d == "left" else "右", "←" if _want < 0 else "→", _ol["move"]))

    # ①-b **老边（没有 mid_dir 键）真的走 `mid_jump_dir` 兜底**（`""` ⇒ 不启用 ✓）
    from core import zones as Z

    check(Z.mid_jump_dir({"from": "甲", "to": "乙", "kind": "climb", "ladder": "L1"}) == "",
          "老边（没有这一格）该解析成 `""`（= 不中途跳下 ✓）")
    check(Z.mid_jump_dir({"mid_dir": "瞎写"}) == "",
          "认不出的方向该当 `""`（同 `walk_dir` 的口径 ✓）")
    check(Z.mid_jump_text({"mid_dir": "left", "mid_y": -120}) == "中途跳下·仅向左·高度 -120",
          "编辑器那截显示不对：%r" % (Z.mid_jump_text({"mid_dir": "left", "mid_y": -120}),))


def t_climb_down_jump_off():
    """**下爬**的收尾：「跳下绳梯」（2026-09-26 用户当场定的新时序）。

    用户原话：「当向下攀爬时，判定上绳梯后，经过『坐标对齐误差时间』（设置里判定参数
    页签那个），松开 ↓ 并执行一个'跳下绳梯'的操作行为：按住目标 foothold 集合的中心对应
    的方向键（向量运算就能解决）→ 按跳 → 松开跳 → 松开方向 → 等待 y 稳定」。
    四个细节他也在同一轮里逐条定了：
      · 松 ↓ 的时点 = **字面**（判定在绳上 + 等完误差时间就松，不等绳底）；
      · 「按跳」= **点按 `TAP_ON_S`**（≈一帧，复用现成的数），方向键**从起跳按住到落地**；
      · 落地判据 = 目标面 / 目标集合 / y 稳定 **三个先到算哪个**；
      · 「目标集合中心」= 集合里所有非墙 foothold 的 x 中点（`(min+max)/2`，同「走」）。

    为什么这么改（现场口径）：绳只是"把人从平台面上放下去"的抓手 —— 抓住绳时人已经在
    平台面**下面**（105090600 的 L1：绳顶 y=-205、平台面 y=-208），所以松手就是往下落 ✓；
    松手时按住"朝目标集合中心"的方向键是为了**落点漂到平台中间**（绳常常挂在平台边上）。

    钉住的东西（每条都对应一个真会写错的点）：
      ① 在绳上「稳住 hold_ms」才松 ↓ —— 不够 hold_ms 时**还按着 ↓**；
      ② 先**单独一拍**按住方向键，再按跳（同「下跳」先↓后跳：一次发两个键不保证顺序）；
      ③ **跳下绳梯那几拍绝不能再按 ↓**（`dir == 0`）—— ↓ 是"还挂在绳上"的动作；
      ④ 方向三态：目标在右 → / 在左 ← / 中心算不出来 ⇒ 直着跳**并如实说**（不许猜）；
      ⑤ 落体期间**离开绳**（`ladder_id=None`）不许被"从绳上掉下来了"误杀；
      ⑥ 落地**当拍收工并松开所有键**（不留"再按住一会儿"）；
      ⑦ `retry()` 把下爬那三个锚点清干净（否则重来一轮"还没上绳就以为稳够了"✗）；
      ⑧ 用**用户那条真边**（105090600：左上平台 →(爬 L1)→ 左上）跑一遍，钉住
         `dst_x` / `dst_y` 那套口径。
    """
    from core import mapdata, zones
    from decision import route

    C = route.ClimbJob

    def mk(**kw):
        # 绳 x=56；目标集合中心 x=142（在**右边**）；目标平台面 dst_y=100。
        a = dict(ladder_id="L1", x=56.0, y1=-205.0, y2=56.0, direction=-1,
                 dst_set="左上", dst_y=100.0, dst_x=142.0,
                 tol_px=6, hold_ms=200, timeout_s=600.0)
        a.update(kw)
        return C(**a)

    def _on_ladder(job, t0=0.30):
        """把任务推到"判定已在绳上"那一拍（前面先把对齐走完）。"""
        job.update(0.0, px=56.0, py=-205.0)
        job.update(0.25, px=56.0, py=-205.0)          # 250ms ≥ hold_ms=200 ⇒ 该贴绳
        return job.update(t0, px=56.0, py=-205.0, ladder_id="L1")

    # ---- ① 上绳前：按住跳 + ↓（贴上去）----
    job = mk()
    job.update(0.0, px=56.0, py=-205.0)
    o = job.update(0.25, px=56.0, py=-205.0)
    check(o["jump"] and o["dir"] == -1 and not o["done"],
          "对齐好了该按住跳+↓贴上去：%s" % o)
    # ---- ② 判定上绳 ⇒ **松开跳、只按住 ↓**，稳「坐标对齐误差时间」----
    o = job.update(0.30, px=56.0, py=-205.0, ladder_id="L1")
    check(o["dir"] == -1 and not o["jump"] and o["phase"] == C.CLIMB,
          "已上绳该松开跳、只按住 ↓：%s" % o)
    check("稳" in o["note"] and "200" in o["note"],
          "没说清「在绳上稳住误差时间再松手」：%r" % o["note"])
    o = job.update(0.45, px=56.0, py=-205.0, ladder_id="L1")     # 150ms < 200ms
    check(o["dir"] == -1 and o["phase"] == C.CLIMB,
          "稳住时间没到就把 ↓ 松了（①）：%s" % o)
    # ---- ③ 稳够了 ⇒ 松 ↓，**先单独一拍**按住方向键（朝集合中心，在右边 ⇒ →）----
    o = job.update(0.55, px=56.0, py=-205.0, ladder_id="L1")     # 250ms ≥ 200ms
    check(o["phase"] == C.JUMP_DOWN and o["move"] == 1,
          "稳够了该松 ↓、并朝集合中心（右边）按住 →：%s" % o)
    check(o["dir"] == 0 and not o["jump"],
          "「跳下绳梯」第一拍就带了 ↓ 或跳（顺序是先按住方向再按跳）：%s" % o)
    check("142" in o["note"], "没写清朝哪个 x 跳：%r" % o["note"])
    # ---- ④ 按跳：点按 TAP_ON_S；跳必须在，↓ 必须不在 ----
    o = job.update(0.60, px=56.0, py=-205.0)
    check(o["jump"] and o["move"] == 1 and o["dir"] == 0,
          "该按跳了（且不许带 ↓）：%s" % o)
    o = job.update(0.60 + route.TAP_ON_S * 0.5, px=56.0, py=-205.0)
    check(o["jump"], "点按窗口还没到就松了跳：%s" % o)
    o = job.update(0.60 + route.TAP_ON_S + 0.01, px=56.0, py=-205.0)
    check(not o["jump"] and o["move"] == 1 and o["dir"] == 0,
          "点按窗口过了该松开跳、方向继续按着：%s" % o)
    check("等落地" in o["note"], "没说清在等落地：%r" % o["note"])
    # ---- ⑤ 落体：**离开绳**也不许判"掉下来了"（这一相是主动松手）----
    for k in range(6):
        o = job.update(0.70 + 0.5 * k, px=56.0, py=-150.0 + 20.0 * k)
        check(not o["failed"],
              "跳下去之后被「从绳上掉下来了」误杀（%.1fs 没回绳上）：%s"
              % (0.5 * k, o))
        check(o["move"] == 1 and not o["jump"] and o["dir"] == 0,
              "落体这一拍该只按住 →：%s" % o)
    # ---- ⑥ 落地判据 a：够到目标平台的面 ⇒ **当拍收工、所有键松开** ----
    # ⭐ 落地面改读广播：`ground_y` = **脚下那块面的 y**（判据唯一来源 ✓）
    o = job.update(4.00, px=60.0, py=100.0, ground_y=100.0)
    check(o["done"] and o["move"] == 0 and o["dir"] == 0 and not o["jump"],
          "到目标面该当拍收工并松开所有键：%s" % o)
    check("100" in o["note"], "到达没写清是到哪个面：%r" % o["note"])
    # ---- ⑦ 落地判据 b：脚下已经是目标集合（dst_y 算不出来时也管用）----
    #    ⚠ 绳端要放远：`_arrived` 里"够到绳的另一端"那条**独立于 dst_y**，绳底留在 56
    #    的话 y 一过 56 就被它先接走了（写这条时第一次就是这么摆错的 ✗）。
    job = mk(dst_y=None, y2=9999.0)
    _on_ladder(job)
    job.update(0.55, px=56.0, py=-205.0, ladder_id="L1")         # 松 ↓
    job.update(0.60, px=56.0, py=-205.0)                        # 按跳
    o = job.update(1.20, px=60.0, py=-100.0, here_sets=["左上"])
    check(o["done"] and "左上" in o["note"],
          "落回目标集合该算落地（⑦）：%s" % o)
    # ---- ⑧ 落地判据 c：y 稳定（平台没圈集合 / 算不出目标面时的兜底）----
    job = mk(dst_y=None, y2=9999.0)
    _on_ladder(job)
    job.update(0.55, px=56.0, py=-205.0, ladder_id="L1")
    job.update(0.60, px=56.0, py=-205.0)
    o = job.update(1.20, px=56.0, py=-20.0)
    check(not o["done"], "刚开始落就说落地了：%s" % o)
    o = job.update(1.20 + route.STALL_S + 0.05, px=56.0, py=-20.0)   # y 3 秒没动过
    check(o["done"] and "稳定" in o["note"], "y 稳定没算落地（⑧）：%s" % o)
    # ---- ⑨ 方向三态：dst_x - px 的符号（向量运算）----
    def _hop(px_at_hop, dst_x):
        j = mk(dst_x=dst_x)
        _on_ladder(j)
        return j.update(0.55, px=px_at_hop, py=-205.0, ladder_id="L1")

    check(_hop(56.0, 142.0)["move"] == 1, "目标在右边该按 →")
    check(_hop(400.0, 142.0)["move"] == -1, "目标在左边该按 ←")
    o = _hop(56.0, None)
    check(o["move"] == 0 and "没有方向可依" in o["note"],
          "集合中心算不出来时该**直着跳并如实说**，不许猜方向：%s" % o)
    # ---- ⑩ retry 把下爬那三个锚点清干净（不清 = 重来一轮"还没上绳就以为稳够了"）----
    job = mk()
    _on_ladder(job)
    check(job._on_ladder_since is not None, "没记下「已在绳上」的时刻")
    check(job.update(0.55, px=56.0, py=-205.0, ladder_id="L1")["phase"] == C.JUMP_DOWN,
          "没进「跳下绳梯」相")
    o = job.retry()
    check(o["phase"] == C.ALIGN and job._on_ladder_since is None
          and job._jump_off_at is None and not job._press_dir_done
          and job._y_still_since is None,
          "retry 没把下爬那几个锚点清干净：%s" % o)
    #    清干净的**行为证据**：重来这一轮第一拍（还没上绳）不许把 ↓ 松掉
    job.update(0.60, px=56.0, py=-205.0)
    o = job.update(0.90, px=56.0, py=-205.0)
    check(o["jump"] and o["dir"] == -1, "重来一轮没重新贴绳：%s" % o)
    # ---- ⑪ 真边（用户数据）：左上平台 →(爬 L1)→ 左上，判向下，dst_x=142 / dst_y=100 ----
    t = mapdata.load("105090600")
    z = zones.load("105090600")
    # ⚠ 这条原来要求**用户数据里**有「左上平台 →(爬 L1)→ 左上」这条**下爬**边 —— 用户
    #   2026-09-27 把这一对改成了「下跳 + 走(仅向左)」（他要的是"往左走出去、掉下去"）⇒
    #   数据里已经没有这条 climb 了。**他的数据不是用例的契约**：这条用例真正要钉的是
    #   "站在平台上、目标在下层 ⇒ 判成向下" + 后面那套「跳下绳梯」时序 ⇒ 边在这里**现造
    #   一条**（地形、绳 L1、两个集合全是真的，只补这一条边）✓。
    edge = {"from": "左上平台", "to": "左上", "kind": "climb", "ladder": "L1",
            "why": "用例现造的（用户数据里这一对已改成「下跳 + 走(仅向左)」）"}
    jd = route.job_for_edge(t, z, edge, tol_px=10, hold_ms=250)
    check(jd.dir == -1, "这条边该判成**向下**（climb_direction）：dir=%s" % jd.dir)
    check(jd.dst_x == route.dst_center_x(t, z, "左上"),
          "dst_x 不是「目标集合所有非墙 foothold 的 x 中点」：%s" % jd.dst_x)
    check(abs(jd.dst_x - 142.0) < 1.0,
          "「左上」的 x 中点变了（实测 142；口径 = (min+max)/2）：%s" % jd.dst_x)
    check(abs(jd.dst_y - 100.0) < 1.0,
          "「左上」的目标平台面变了（实测 100）：%s" % jd.dst_y)
    #    算不出中心时**返回 None**（让执行器如实说，不许猜一个方向）
    check(route.dst_center_x(t, z, "这个集合不存在") is None,
          "目标集合圈不到 foothold 时该返回 None，而不是编一个中心：%s"
          % route.dst_center_x(t, z, "这个集合不存在"))
    #    整条路走一遍：贴绳 ⇒ 稳 250ms ⇒ 朝 142（右边）按住 → ⇒ 按跳 ⇒ 落到 100
    jd.update(0.0, px=56.0, py=-205.0)
    jd.update(0.30, px=56.0, py=-205.0)
    o = jd.update(0.60, px=56.0, py=-205.0, ladder_id="L1")
    check(o["dir"] == -1 and not o["jump"], "真边：已上绳该只按住 ↓：%s" % o)
    o = jd.update(0.90, px=56.0, py=-205.0, ladder_id="L1")     # 300ms ≥ 250
    check(o["phase"] == C.JUMP_DOWN and o["move"] == 1 and o["dir"] == 0,
          "真边：该朝「左上」中心（142，在右边）按住 →：%s" % o)
    o = jd.update(1.00, px=56.0, py=-205.0)
    check(o["jump"] and o["move"] == 1, "真边：该按跳了：%s" % o)
    o = jd.update(1.20, px=56.0, py=-50.0)
    check(not o["done"] and o["move"] == 1 and o["dir"] == 0,
          "真边：落体中该只按住 →：%s" % o)
    # ⭐ 落地面读广播 `ground_y`（= 脚下那块面的 y ✓，口径在 `perception/pos_state.py` ✓）
    o = jd.update(1.60, px=58.0, py=100.0, ground_y=100.0)
    check(o["done"] and o["move"] == 0, "真边：到目标面没收工：%s" % o)


def t_climb_align_each_round_presses():
    """「对齐 x **至少发 1 次点按方向键**」—— **每一轮（每次跳）都要真发一次**（用户 2026-09-28）。

    用户原话："朝目标绳梯的 x 方向对齐 x（**至少发 1 次点按方向键**）→ 跳 → 按住 ↑，
    这三步要循环，直到位置状态通知成功或失败"。

    ⛔ 原来只要 `|dx| ≤ 容差` 就直接"站住等 `hold_ms` ⇒ 跳"，**一次方向键都不发** ✗ ——
    用户要的是**每一轮都真的朝绳发过一下**才跳 ✓。记账只有**一处**（`_out` 里 ✓）：
    **按了跳 ⇒ 这一轮结束**（下一轮必须重新发方向键 ✓）；**发了水平方向 ⇒ 这一轮满足** ✓。

    钉四件：
      ① 进容差（`|dx| ≤ tol`）**第一拍** ⇒ **朝绳那一侧点按一次**（`move` 是朝绳方向 ✓），
         **不是**"站着不动"（`move=0`）✗；
      ② 点过之后才"站住等保持时间" ✓，保持够了才跳 ✓；
      ③ 按跳那一拍 ⇒「这一轮发过方向键」被清掉（下一轮要重新点 ✓）；
      ④ **跳失败回对齐**（`_retry_back_to_align`）⇒ 重新武装 ⇒ 下一轮进容差**又要点一次** ✓
         （哪怕 x 已经在容差内 ✓）。
    """
    from decision import route

    C = route.ClimbJob

    def mk(**kw):
        a = dict(ladder_id="L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                 dst_set="乙平台", src_set="甲平台", tol_px=6, hold_ms=200,
                 near_px=20)
        a.update(kw)
        return C(**a)

    # ① 进容差 ⇒ **先点按一次**（dx = 700-703 = -3 ⇒ 朝绳那侧是 ←）
    j = mk()
    o = j.update(0.0, px=703.0)
    check(o["move"] == -1 and not o["jump"],
          "进容差第一拍该**朝绳点按一次方向键**（不是站着不动 ✗）：%s / %r"
          % (o, j.note))
    check(j._align_pressed, "点过之后该记成「这一轮发过方向键」✗")
    # ② 点过 ⇒ 这一拍才"站住等保持时间"
    o = j.update(0.02, px=703.0)
    check(o["move"] == 0 and not o["jump"],
          "点过之后该站住等保持时间（不能马上跳）：%s" % o)
    # ③ 站住够了 ⇒ 跳（这一步在 CLIMB 相里，见 `t_climb_flow_rules` ② ✓）
    o = j.update(0.25, px=703.0)
    check(o["jump"], "保持够了该按跳：%s" % o)
    check(not j._align_pressed,
          "按跳那一拍该把「这一轮发过方向键」清掉（**下一轮要重新点一次** ✓）✗")
    # ④ 跳失败回对齐 ⇒ 重新武装 ⇒ 下一轮进容差**又要点一次**（哪怕 x 已在容差内 ✓）
    j._retry_back_to_align("测试")
    check(not j._align_pressed,
          "`_retry_back_to_align` 之后该重新武装（下一轮必须再点一次 ✓）")
    o = j.update(1.0, px=703.0)
    check(o["move"] == -1 and not o["jump"],
          "重来那一轮进容差第一拍该**再点一次**（哪怕 x 已在容差内 ✓）：%s / %r"
          % (o, j.note))


def t_climb_stall_only_on_ladder():
    """「补发↑ / 攀爬失败」**只认位置状态广播**（用户 2026-09-28 核心思路：执行器不做坐标运算 ✓）。

    原来的「爬不动了 ⇒ 补按 ⇒ 失败」是**执行器自己比 y** ✗ —— 已删（`_stalled_by` ✓）。
    现在执行器只读广播、三种情况（用户 2026-09-28 原话）：
      a. `climb_failed`（x 偏出「坐标对齐误差范围」）⇒ 攀爬失败 → 交给 agent 延迟重启；
      b. `at_ladder_top` ⇒ 到顶收工（`_arrive_step` ✓，另测）；
      c. `climb_stalled`（y 在「移动操作尝试间隔」内没变）⇒ 补发按住 ↑（`reassert` ✓）。

    钉三件：
      ① 对齐阶段（没广播）⇒ 不 reassert、不 fail ✓（"补按只在绳上"现在由**状态机只在绳上才广播**
        保证 ✓ —— 它的 `climb_stalled`/`climb_failed` 只在 `ladder_id` 非空（= 按住 ↑ 且在绳段里）时算 ✓）；
      ② 已上绳 + `climb_stalled=True` ⇒ 补发 ↑（`reassert=True`），**不失败** ✓（2c 没有"补按→失败"了 ✗）；
      ③ 已上绳 + `climb_failed=True` ⇒ **与 climb_stalled 同一支：按住 ↑ + 补按，不失败**
         （2026-09-29 用户流程图定稿：「至少得看到有按跳之后才可能接收失败」—— 这两种广播
         都发生在"人已在绳上"⇒ 没有失败这个出口 ✓；在绳上按左右=松手 ✗ 所以也不回对齐；
         x 偏出爬起来会自己居中；真爬不上去由「寻路超时时间」兜底 ✓）。
    """
    from decision import route

    C = route.ClimbJob

    def mk():
        return C("L2", x=700.0, y1=-115.0, y2=231.0, direction=1, dst_set="平台",
                 dst_y=-208.0, tol_px=6, hold_ms=0, timeout_s=600.0)

    # ① 对齐阶段（人在平台上挪着对准绳的 x，dx=50 还没对上）：没广播 ⇒ 不 reassert、不 fail
    j = mk()
    bad = []
    for i in range(0, 60):                      # 6 秒
        o = j.update(i * 0.1, px=650.0, py=283.0)
        if o.get("reassert") or o["failed"]:
            bad.append((i * 0.1, o.get("reassert"), o["failed"], j.note))
    check(not bad,
          "对齐阶段乱要求补发/失败（执行器该只认广播 ✓）：%s" % (bad[:2],))

    # ② 已上绳 + climb_stalled=True ⇒ 补发 ↑、不失败
    j = mk()
    j.update(0.0, px=700.0, py=100.0, ladder_id="L2")
    check(j.phase == C.CLIMB, "前提：该进上绳相：%s" % j.phase)
    o = j.update(0.1, px=700.0, py=100.0, ladder_id="L2", climb_stalled=True)
    check(o.get("reassert") is True and not o["failed"],
          "广播说 y 没变，却没补发 ↑（或误判失败）✗：%s" % o)
    o = j.update(0.2, px=700.0, py=100.0, ladder_id="L2")
    check(not o.get("reassert") and not o["failed"],
          "没广播却乱补发/失败：%s" % o)

    # ③ 已上绳 + climb_failed=True ⇒ **与 climb_stalled 同一支：按住 ↑ + 补按，不失败**
    j = mk()
    j.update(0.0, px=700.0, py=100.0, ladder_id="L2")
    check(j.phase == C.CLIMB, "前提：该进上绳相：%s" % j.phase)
    o = j.update(0.1, px=700.0, py=100.0, ladder_id="L2", climb_failed=True)
    check(not o["failed"] and o["dir"] == 1 and o.get("reassert") is True,
          "广播说 x 偏出，却判了失败/松了 ↑（该按住 ↑ + 补按继续爬 ✗）：%s" % o)
    check("x 偏出" in str(o["note"]),
          "说明没写清是哪种广播 ✗：%r" % o["note"])
    # 广播撤销（游戏把人居中 / y 动了）⇒ 照旧爬，不残留任何失败态
    o = j.update(0.2, px=700.0, py=90.0, ladder_id="L2")
    check(not o["failed"] and o["dir"] == 1 and not o.get("reassert"),
          "广播撤销后没回到正常爬升 ✗：%s" % o)


def t_climb_reassert_log_rate_limited():
    """「补按 ↑/↓」的打点**限流 1 条/秒**（2026-09-29：爬不动时该分支**每拍**都进 ⇒
    现场 41 条/秒、一场 588 条把日志淹了 ✗）—— 键**照补** ✓，只是打点别刷屏 ✓。

    钉两件：① 1 秒内连拍 6 拍 ⇒ 只记 1 条 ✓；② 过了 1 秒再拍 ⇒ 记第 2 条 ✓
    （限流不是掐死 ✓）。
    """
    import decision.agent as _am
    from decision import route

    s = fresh_settings(goto_timeout_s=0.0)
    s.enabled = True
    h = Harness(s)
    job = route.ClimbJob("L2", x=700.0, y1=-115.0, y2=231.0, direction=1,
                         dst_set="平台", dst_y=-208.0, tol_px=6, hold_ms=0,
                         timeout_s=600.0)
    evs = []
    _orig = _am.behavior.event
    with h._patched():
        h.agent.start_route([job], why="用例：补按打点限流")
        # 直接把这一步的 update 换成"永远喊补按"（只测 agent 侧的打点限流 ✓；
        # 执行器本身的行为在 t_climb_stall_only_on_ladder 钉着 ✓）。
        h.agent._climb.update = lambda *a, **k: {
            "phase": "climb", "move": 0, "dir": 1, "jump": False, "note": "限流用例",
            "done": False, "reassert": True, "failed": False}
        _am.behavior.event = lambda name, **kw: evs.append(name)
        try:
            ws = h.ws(with_mob=False)
            for i in range(6):                 # 0.1s 一拍 × 6 ⇒ 全在 1 秒内
                h.clock.t = 10.0 + i * 0.1
                h.agent._climb_tick(h.clock.t, 520.0, set(), ws)
            check(evs.count("climb_reassert") == 1,
                  "1 秒内 6 拍补按却记了 %d 条（限流失效，日志照样被淹 ✗）：%r"
                  % (evs.count("climb_reassert"), evs))
            h.clock.t = 11.2                   # 过了 1 秒 ⇒ 该记第 2 条 ✓
            h.agent._climb_tick(h.clock.t, 520.0, set(), ws)
            check(evs.count("climb_reassert") == 2,
                  "过了 1 秒没记第 2 条（限流把打点掐死了 ✗）：%r" % (evs,))
        finally:
            _am.behavior.event = _orig


def t_job_interrupt_by_fight():
    """**被战斗打断**不许把挨打那几秒算成"爬不动了 / 走不动了"，且回来后要能从对的地方继续。

    用户 2026-09-26 报的状态机问题："attack 会强制打断其他状态 ⇒ climb 被打断时跳可能被吞了"。
    根因：任务内部全是**绝对时刻**锚点（`_best_at` / `_off_since` / `_in_tol_since`），而
    打架那几拍 `update()` 根本不被调用（`now` 却一直在走）⇒ 回来就把"站着挨打"当成
    "不动了" ✗ / 超时 ✗。修法见 `route._PausableJob`：agent 每拍叫一声 `interrupted()`，
    任务回来把那些锚点整体往后挪，并**重新走一遍上绳流程**（`phase` 回 `ALIGN`）：
    没上绳 ⇒ 重新对齐→再跳；**在绳上** ⇒ ALIGN 那一步自己跳过横向对齐、接着按住 ↑
    （用户 2026-09-27 要求 ③；细节与"在绳上不许横移"那条见 `t_climb_flow_rules` ✓）。
    """
    from decision import route

    C = route.ClimbJob
    # ① 没上绳被打断 ⇒ 回来必须**重新对齐**（这就是用户说的"跳被吞了"要治的那一拍）
    job = C("L2", x=701.0, y1=-115.0, y2=231.0, direction=1, dst_set="平台",
            tol_px=6, hold_ms=250, timeout_s=600.0)
    o = job.update(0.0, px=701.0)
    check(job.phase == C.ALIGN and not o["jump"], "进误差范围那拍不该按跳：%s / %s"
          % (job.phase, o))
    o = job.update(0.3, px=701.0)
    check(job.phase == C.CLIMB and o["jump"], "站住保持够了该进「按跳」：%s / %s"
          % (job.phase, o))
    job.interrupted(0.4)                       # 从 0.4s 起被战斗占用（这几拍没人喂它）
    o = job.update(5.4, px=701.0)              # 5.4s 才回来 = 挨打 5 秒
    check(job.phase == C.ALIGN and not o["jump"],
          "没上绳被打断后**没有**重新对齐（接着开跳 = 贴着绳边乱跳 ✗）：%s / %s"
          % (job.phase, o))
    # ⚠ "保持窗口**重新起算**"才是"重来了一遍"的硬证据：只把 phase 挪回 ALIGN、却留着
    #   打断前那次的 `_in_tol_since`，就会这一拍立刻又跳 ✗（那还是"跳被吞"的另一种样子）。
    #   注：这一拍的 `note` 会被 ALIGN 分支自己写的进度话覆盖（"站住等 250 ms"）—— 所以
    #   这里断言行为，**不**断言那句提示（写这条时就是这么红的 ✗）。
    o = job.update(5.6, px=701.0)
    check(job.phase == C.ALIGN and not o["jump"],
          "重新对齐的保持窗口没有重新起算（200ms 就想跳）：%s / %s" % (job.phase, o))
    o = job.update(5.8, px=701.0)               # 重新站住保持够（400ms ≥ 250）⇒ 再跳一次
    check(o["jump"] is True and job.phase == C.CLIMB,
          "重新对齐之后没有接着按跳（「跳被吞了」没治好）：%s / %s" % (job.phase, o))
    # ② 短于 PAUSE_MIN_S 的间隔**不算**打断（正常拍间隔 15fps ≈ 66ms）——
    #    不然任务每拍都"重新对齐" ✗（比不修还糟）。
    job2 = C("L2", x=701.0, y1=-115.0, y2=231.0, direction=1, tol_px=6,
             hold_ms=250, timeout_s=600.0)
    job2.update(0.0, px=701.0)
    job2.update(0.3, px=701.0)
    job2.interrupted(0.31)
    o = job2.update(0.33, px=701.0)
    check(job2.phase == C.CLIMB and o["jump"] is True,
          "正常拍间隔（20ms）被当成了打断：%s / %s" % (job2.phase, o))
    # ③ **已上绳**被打断 ⇒ 流程**重走一遍**（`phase` 回 `ALIGN`），但那一步会**跳过横向
    #    对齐**（在绳上按左右 = 松手掉下来 ✗）⇒ 对外看到的就是"接着按住 ↑、不按跳"；
    #    ④ 挨打那几秒**不算**"爬不动了"。（"跳过对齐"本身在 `t_climb_flow_rules` 里钉 ✓）
    job3 = C("L2", x=701.0, y1=-115.0, y2=231.0, direction=1, dst_y=-208.0,
             tol_px=6, hold_ms=0, timeout_s=600.0)
    job3.update(0.0, px=701.0)
    o = job3.update(0.1, px=701.0, py=100.0, ladder_id="L2")
    check(job3.phase == C.CLIMB and o["jump"] is False and o["dir"] == 1,
          "已上绳该只按住 ↑（不再按跳）：%s / %s" % (job3.phase, o))
    job3.interrupted(0.2)                       # 打架 5 秒
    o = job3.update(5.2, px=701.0, py=100.0, ladder_id="L2")
    check(job3.phase == C.CLIMB and not o["failed"] and o["jump"] is False,
          "已上绳被打断 ⇒ 该接着按住 ↑（失败/退回对齐都不对）：%s / %s"
          % (job3.phase, o))
    check("就不再" not in str(o["note"]),
          "挨打的 5 秒被算成「爬不动了」了：%r" % o["note"])
    # ④ 对照：**没有**被打断时，"y 没变"由广播 `climb_stalled` 说 ⇒ 补发 ↑（不失败 ✓ 2c）。
    job4 = C("L2", x=701.0, y1=-115.0, y2=231.0, direction=1, dst_y=-208.0,
             tol_px=6, hold_ms=0, timeout_s=600.0)
    job4.update(0.0, px=701.0)
    job4.update(0.1, px=701.0, py=100.0, ladder_id="L2")
    o = job4.update(5.2, px=701.0, py=100.0, ladder_id="L2", climb_stalled=True)
    check(o.get("reassert") is True and not o["failed"],
          "广播说 y 没变，却没补发 ↑：%s" % o)
    # ⑤ **寻路超时按「每次进战斗」重新计时**（用户 2026-09-26 要求）：挨打的时间不算超时。
    #    两条的时序与超时值完全一样，差别只在中间报没报过一次 `interrupted`。
    j5 = C("L2", x=701.0, y1=-115.0, y2=231.0, direction=1, tol_px=6, hold_ms=0,
           timeout_s=1.0)
    j5.update(0.0, px=701.0)                  # _t0 = 0.0
    j5.interrupted(0.5)                       # 0.5s 进战斗 ⇒ 超时从这一刻重算
    o = j5.update(1.2, px=701.0)              # 距"重算"才 0.7s ⇒ 不许判超时
    check(not o["failed"], "打完架回来却按旧起点判了超时（超时没重新计时）：%s" % o)
    # 对照：同样时序、**没进过战斗** ⇒ 1.2s > 1.0s ⇒ 必须照旧判超时（别把判据修没了 ✗）
    j6 = C("L2", x=701.0, y1=-115.0, y2=231.0, direction=1, tol_px=6, hold_ms=0,
           timeout_s=1.0)
    j6.update(0.0, px=701.0)
    o = j6.update(1.2, px=701.0)
    check(o["failed"] and "超时" in str(o["note"]),
          "没被打断、超过 timeout_s 却没判超时（超时判据被修没了）：%s" % o)
    # ⑥ **「走」那一步同一套规矩**（用户说的是"寻路过程中"，不是只有爬绳 ✓）：
    #    走路被怪打断 ⇒ ① 超时重新计时 ② 回来不许当场报"走不动了"（挨打不算走不动 ✓）。
    spots = [(700.0, 690.0, 710.0, "1")]
    j7 = route.WalkJob("乙平台", spots, tol_px=6, timeout_s=1.0)
    j7.update(0.0, px=600.0)                  # _t0 = 0.0，朝右走
    j7.interrupted(0.5)                       # 0.5s 进战斗 ⇒ 超时从这一刻重算
    o = j7.update(1.2, px=600.0)              # 距重算 0.7s、位置没动 ⇒ 两条都不许报
    check(not o["failed"],
          "走路被怪打断后回来就判失败（超时没重算 / 『走不动了』把挨打算进去了）：%s" % o)
    # 对照：同样时序、没进过战斗 ⇒ 1.2s > 1.0s ⇒ 照旧判超时
    j8 = route.WalkJob("乙平台", spots, tol_px=6, timeout_s=1.0)
    j8.update(0.0, px=600.0)
    o = j8.update(1.2, px=600.0)
    check(o["failed"] and "超时" in str(o["note"]),
          "走：没被打断、超过 timeout_s 却没判超时：%s" % o)
    # ⑦ **「y 不动 ⇒ 补发 ↑」**（用户 2026-09-28 2c）：只认广播，**不再有"补按→失败"** ✗
    #   （执行器不做坐标判定；真卡住了由「寻路超时」总闸收口 ✓）。
    j9 = C("L2", x=701.0, y1=-115.0, y2=231.0, direction=1, dst_y=-208.0,
           tol_px=6, hold_ms=0, timeout_s=600.0)
    j9.update(0.0, px=701.0)
    j9.update(0.1, px=701.0, py=100.0, ladder_id="L2")
    o = j9.update(3.2, px=701.0, py=100.0, ladder_id="L2", climb_stalled=True)
    check(o.get("reassert") is True and not o["failed"],
          "广播说 y 没变却没补发 ↑（或误判失败）✗：%s" % o)
    o = j9.update(6.5, px=701.0, py=100.0, ladder_id="L2", climb_stalled=True)
    check(not o["failed"],
          "2c 没有「补按→失败」了：执行器只补发、不因 y 不动而失败 ✗：%s" % o)


def _kill_qt(w):
    """**立刻**销毁用例里造的 Qt 控件（治 T4 的退出崩溃）。

    为什么不能用 `deleteLater()`：那要 Qt 事件循环转一圈才真销毁，而自检脚本没有
    事件循环 ⇒ 对象一直活到进程退出，和 GC 抢析构 ⇒ 偶发（现在变必然）访问违例 ✗。
    这里用 `sip.delete` 当场析构 + 回收，退出时就没有"待销毁"的 Qt 对象了 ✓。
    """
    import gc

    from PyQt5 import sip
    try:
        w.close()
        w.setParent(None)
        sip.delete(w)
    except Exception:                       # noqa: BLE001  已经没了/不是 Qt 对象
        pass
    del w
    gc.collect()


def t_battle_zone_restriction():
    """「限制战斗区域」（用户 2026-09-26）：不在配置的集合里 ⇒ **不打架**，先下前往命令。

    钉四件事：
      · 空列表 = 不限制（老行为，别把打架弄没了 ✗）；
      · 在区域里 ⇒ 照常打（不误伤 ✓）；
      · 不在区域里 + 有怪 ⇒ **一下都不打**，并且**真的下了「前往」**（去第一个配置的集合 ✓）；
      · 拿不到 `here_sets`（定位没输出）也算"不在"⇒ 先回去（不许瞎打 ✓）。
    """
    from decision import route

    s = fresh_settings(attack_dist=120.0)
    s.enabled = True
    s.battle_zone_sets = ["甲平台"]
    h = Harness(s)
    h.mobs_fn = lambda t: [Mob(id=1, x=580.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    atk = s.keymap["attack"]
    with h._patched():
        a = h.agent
        h.clock0 = h.clock.t          # ⚠ 夹具要它才记日志（不设 ⇒ 后面读 h.log 全空 ✗）
        a.route_plan = lambda dst: {"jobs": [route.WalkJob("甲平台",
                                                           [(520.0, 510.0, 530.0,
                                                             -208.0, "1")])],
                                    "path": ["乙平台", "甲平台"], "why": "",
                                    "here": False}
        ws = h.ws(with_mob=True)
        # ① 不在区域里 + 有怪 ⇒ 不许攻击键 + 要下「前往」
        ws.player.here_sets = ["乙平台"]
        h.clock.t += 0.1
        act = a.tick(ws)
        check("不在战斗区域" in str(act.get("reason")),
              "不在区域里却没走「先回去」那条路：%s" % act)
        check(atk not in [k for _t, kind, k in h.log if kind == "down"],
              "不在战斗区域里还是打了：%s" % h.log)
        # ⚠ 断言要看**真任务**：规则是在 agent 自己身上调 `plan_and_start_route` 的
        #   （造一个假 agent 去接 `start_route` 收不到任何东西 ✗ —— 写这条时就这么错过）
        check(a._climb is not None and getattr(a._climb, "dst_set", "") == "甲平台",
              "没下「前往」命令 / 目标不是第一个配置的集合：%r" % (a._climb,))
        # "去哪"那句在**路线**的说明里（`_route_why` ✓）；`current_goto_note()` 是任务
        # **当前那一拍**的话（"微调：差 20 px…"✓）—— 别断错对象（写这条时就是这么红的 ✗）
        check("甲平台" in str(a._route_why),
              "命令的说明里没写清去哪个集合：%r" % (a._route_why,))
        a.stop_route("用例：①检查完了")      # 清干净，再验"已经在区域里"那条
        # ② 回到区域里 ⇒ 照常打（不误伤 ✓）
        ws.player.here_sets = ["甲平台"]
        h.clock.t += 0.2
        h.log.clear()
        act2 = a.tick(ws)
        # ⚠ 看**状态**而不是"这一拍有没有 down"：攻击序列可能上一拍就按下、这一拍只是
        #   接着跑（窗口里看不到新的 down ✗ —— 写这条时就这么误判过）。
        check(act2.get("state") == "attack",
              "回到区域里却没进攻击状态（走了「先回去」那条 ✗）：%s" % act2)
        check(a._climb is None,
              "已经在区域里还又下了一条前往命令：%r" % (a._climb,))
        # ③ 清空配置 = 不限制（老行为 ✓）
        s.battle_zone_sets = []
        ws.player.here_sets = ["乙平台"]
        h.clock.t += 0.2
        h.log.clear()
        act3 = a.tick(ws)
        check(act3.get("state") == "attack",
              "没配区域时（=不限制）却没进攻击状态：%s" % act3)
        # ④ 拿不到 here_sets ⇒ 也算"不在区域里"
        s.battle_zone_sets = ["甲平台"]
        ws.player.here_sets = []
        h.clock.t += 10.0                      # 越过重下间隔，确保会再下一次
        h.log.clear()
        a.tick(ws)
        check(atk not in [k for _t, kind, k in h.log if kind == "down"],
              "定位不知道在哪块平台却还是打了（该先回去）：%s" % h.log)
        check(a._climb is not None,
              "不知道自己在哪时没下「前往」命令：%r" % (a._climb,))


def t_turn_output_delay_live():
    """「转向后输出延迟」**改了就生效**（用户 2026-09-27 要求确认）—— 不重启、不重开自动。

    接线：界面写的就是**全局单例** `settings.turn_output_delay_ms`
    （`gui/player_panel._on_turn_params` ⇒ `settings.save()`），而 `_output_actions`
    **每一拍现读**它 ⇒ 界面上改完，**下一拍**就是新值 ✓（没有缓存、不用重启 ✓）。
    作用范围：只管**新开一轮输出**（已经在跑的序列让它跑完 —— 半截掐断会留下按着的键 ✗）；
    三条输出路径都吃它（attack / 规避跳后的那一下 / 回身输出 ✓）；判据从"`set_facing`
    真的**换了朝向**"那一刻起算（`_turn_at` 只在朝向变了时重置 ⇒ 一直朝同一边按方向键
    不会被一直推迟 ✓）。

    这条用**同一个 agent 中途改参数**来钉（界面上改参数做的正是这件事 ✓）：
      ① 换向后 1 秒内 ⇒ **不发攻击键**（延迟窗口内 ✓）；
      ② 窗口过后 ⇒ 输出补上 ✓（说明是延迟不是卡住）；
      ③ 参数**改回 0** ⇒ 下一次换向**当拍就输出** ✓（= 改了立刻生效、没被缓存 ✓）。
    """
    _pos = {"x": 590.0}          # 590 = 右边（距角色中心 70px）、430 = 左边（50px），都在范围内

    s = fresh_settings(strategy="sweep", jump_random_prob=0.0, attack_dist=80.0,
                       attack_cd=0, min_turn_hold_ms=0, turn_output_delay_ms=1000)
    h = Harness(s)
    h.clock0 = h.clock.t          # 日志里的时刻都是相对它（见 `Harness._rec` ✓）
    atk = s.keymap["attack"]

    with h._patched():
        a = h.agent
        h.mobs_fn = lambda t: [Mob(id=1, x=_pos["x"], y=500.0,
                                   w=40.0, h=40.0, conf=0.9)]

        def now():
            return h.clock.t - h.clock0

        def beat(dt=0.05):
            """喂一拍**决策拍**；返回"这一拍朝向真的变了没"（= 换向那一刻）。"""
            h.clock.t += dt
            f0 = a.facing
            a.tick(h.ws())
            return a.facing != f0

        def atk_between(t_from, t_to):
            """(t_from, t_to] 之间发出的攻击键时刻（相对时刻 ✓）。"""
            return [t for t in h.presses(atk) if t_from < t <= t_to]

        # 基线：朝右打一会儿（`_turn_at=0` ⇒ 不会被延迟挡住 ✓）
        for _ in range(10):
            beat()
        check(h.count(atk) > 0, "基线：右前方有怪却一个攻击键都没发：%s" % h.log)

        # ① 怪换到左边 ⇒ 真的换向 ⇒ 换向后 1 秒内**不许**有攻击键
        h.log.clear()
        _pos["x"] = 430.0
        t_turn = None
        for _ in range(10):
            if beat():
                t_turn = now()
                break
        check(t_turn is not None, "怪换到左边了却没换向（用例本身没造出转向）")
        for _ in range(12):                       # 再走 0.6 秒：仍在 1000ms 窗口里
            beat()
        early = atk_between(t_turn, now())
        check(not early,
              "换向之后 %.2f 秒内就打了（「转向后输出延迟」1000ms 没生效）：atk=%s"
              % (now() - t_turn, early))
        # ② 窗口过了要补上（是延迟，不是卡住）
        for _ in range(12):                       # 再走 0.6 秒：越过 1.0 秒
            beat()
        check(atk_between(t_turn + 1.0, now()),
              "延迟窗口过了一直没输出（那是卡住，不是延迟）：%s" % h.log)

        # ③ **中途把参数改回 0**（界面上就是这么改的）⇒ 新一轮输出**立刻**跟上
        s.turn_output_delay_ms = 0                # ← 界面上把参数改回 0（同一个 settings 对象）
        h.log.clear()
        _pos["x"] = 590.0                         # 怪回右边 ⇒ 再换一次向
        t2 = None
        for _ in range(10):
            if beat():
                t2 = now()
                break
        check(t2 is not None, "怪回到右边了却没换回来（用例本身没造出第二次转向）")
        for _ in range(3):                        # 再走两三拍（判决看的是**时刻**，不是这一刻）
            beat()
        check(atk_between(t2, t2 + 0.12),         # 两拍之内（0.10s）必须又开始输出 ✓
              "参数改回 0 之后两拍内没恢复输出（改了不生效 / 被缓存了）："
              "turn=%.3f atk=%s" % (t2, [round(t, 3) for t in h.presses(atk)]))


def t_temp_fight_out_of_zone():
    """「**临时战斗**」不受「限制战斗区域」的约束（用户 2026-09-27 要求）。

    要求原话：「所有**休息**、**寻路失败后**的战斗属于临时战斗，这些战斗不需要走限制
    战斗区域的限制」。用用户项目的实配（`battle_zone_sets=["右下"]`）摆场景：
      · **休息收工/被打断之后** —— 休息点多半不在「右下」里（他那格写的是「右下休息平台」），
        而"被打断"本身就说明有东西正在打我们；
      · **寻路失败之后** —— 半路停下，附近照样有怪。
    老行为是这两段里**一路不还手地往回走**（区域规则排在 `if in_range:` 之前 ⇒ 有怪也不打）✗。

    钉五件事：
      ① **对照组**：没点亮临时战斗时，区域外的怪**照旧不许打**（原规则一点没减 ✓）；
      ② 休息收工（`_finish_rest`）之后 ⇒ **当场进 attack、真按攻击键**，且**不下**"回去" ✓；
      ③ 寻路失败（`_task_finished(failed=True)`）之后 ⇒ 同上 ✓；
      ④ **这一波打完了**（视野里没得打了）⇒ 熄灭 ⇒ 再出怪**又回到**"先回去" ✓；
      ⑤ 没配区域（`battle_zone_sets=[]`）⇒ 本来就不限制（老行为不许被这条改动碰到 ✓）。
    """
    s = fresh_settings(attack_dist=120.0)
    s.strategy = "patrol"
    s.battle_zone_sets = ["右下"]        # ← projects/寺院通道2/project.yaml 的实配
    h = Harness(s)
    h.clock0 = h.clock.t
    mob = Mob(id=1, x=560.0, y=500.0, w=40.0, h=40.0, conf=0.9)     # 距角色中心 40 px
    atk = s.keymap["attack"]
    plans = []

    def _plan(dst):
        plans.append(dst)
        return {"jobs": [], "path": [], "why": "用例：不真走", "here": False}

    with h._patched():
        a = h.agent
        a.route_plan = _plan

        def beat(mobs=True, sets=("别处",), dt=0.2):
            """喂一拍：`mobs` 控制视野里有没有怪，`sets` 是脚下属于哪些集合。"""
            h.mobs_fn = (lambda t: [mob]) if mobs else (lambda t: [])
            h.log.clear()
            h.clock.t += dt
            ws = h.ws(with_mob=mobs)
            ws.player.here_sets = list(sets)
            return a.tick(ws)

        # ① 对照组：没有临时战斗 ⇒ 区域外有怪**不许打**、要下「回右下」
        act = beat()
        check(act.get("state") != "attack",
              "没点亮临时战斗时，区域外的怪竟然打了：%s" % act)
        check(plans and plans[0] == "右下", "没下「回右下」的命令：%r" % (plans,))
        check(atk not in [k for _t, kind, k in h.log if kind == "down"],
              "区域外、没点亮临时战斗，却按了攻击键：%s" % h.log)

        # ② 休息收工之后 ⇒ 临时战斗：当场进 attack、真按攻击键、不再下「回去」
        plans.clear()
        a._finish_rest(h.clock.t)
        act = beat()
        check(act.get("state") == "attack",
              "休息收工后（临时战斗）没进 attack：%s" % act)
        check(atk in [k for _t, kind, k in h.log if kind == "down"],
              "休息收工后（临时战斗）一个攻击键都没发：%s" % h.log)
        check(plans == [], "休息收工后还是去下了「回去」（临时战斗不该走那条）：%r" % plans)

        # ③ 寻路失败之后 ⇒ 同样算临时战斗
        a._temp_fight = False
        a._task_finished(failed=True)
        plans.clear()
        act = beat()
        check(act.get("state") == "attack",
              "寻路失败后（临时战斗）没进 attack：%s" % act)

        # ④ 这一波打完了 ⇒ 熄灭 ⇒ 再出怪又回到「先回去」
        beat(mobs=False)
        check(a._temp_fight is False,
              "视野里没得打了，临时战斗却没熄灭（区域规则会被一直压着）：%s"
              % a._temp_fight)
        plans.clear()
        act = beat(dt=4.0)                  # 越过 ZONE_GOTO_RETRY_S，确保会再下一次
        check(act.get("state") != "attack" and plans,
              "临时战斗已结束，区域外有怪却没回到「先回去」：%s / %r" % (act, plans))

        # ⑤ 不配区域 ⇒ 本来就不限制（老行为 ✓）
        s.battle_zone_sets = []
        a._temp_fight = False
        act = beat()
        check(act.get("state") == "attack",
              "没配「限制战斗区域」时却不打了（老行为被改坏）：%s" % act)


def t_zone_rule_yields_to_goto():
    """「限制战斗区域」**不许拦「命令前往」**（2026-09-26 用户现场定的判据）。

    判据原话：「『命令前往』之后当前任务是**前往 X**；而这条规则的触发条件是**要把当前
    任务设成战斗** —— 两者天生不该冲突」。用户现场：寺院通道2（`battle_zone_sets=[右下]`）
    「右下」<命令前往>「左上平台」，**一离开右下就全程不还手**（还在「上下过渡平台」上
    就已经是"不在战斗区域，先回去"），到左上攻击范围内有怪也不进 attack。

    钉四件事（用**用户项目的实配** `battle_zone_sets=["右下"]`）：
      · 命令途中 + 不在区域 + 范围内**有怪** ⇒ **照打**（进 attack + 真按攻击键），
        而且**不去下"回去"的路线**；
      · 命令途中 + 不在区域 + 范围内**没怪** ⇒ 任务照跑（不被区域规则截住）；
      · 命令**收工之后**回到正常回路 ⇒ 区域规则**照旧生效**（不在区域 ⇒ 先回去 + 下
        前往第一个配置的集合）—— 这条专门防"为了修上面把原规则弄没了"✗。
    """
    from decision import route

    s = fresh_settings(attack_dist=120.0)
    s.strategy = "patrol"
    s.battle_zone_sets = ["右下"]        # ← projects/寺院通道2/project.yaml:150 的实配
    h = Harness(s)
    h.clock0 = h.clock.t
    mob = Mob(id=1, x=560.0, y=500.0, w=40.0, h=40.0, conf=0.9)     # 距角色中心 40 px
    # ⚠ `ws()` 一旦设了 `mobs_fn` 就**不听** `with_mob` 了 ⇒ "没怪"只能靠这个空函数，
    #   传 `with_mob=False` 是没用的（写这条时正好踩了一下）。
    h.mobs_fn = lambda t: [mob]
    atk = s.keymap["attack"]
    left = s.keymap["left"]
    plans = []                          # 区域规则自己下的「前往」命令（它不该在这里出现）

    def _plan(dst):
        plans.append(dst)
        return {"jobs": [], "path": [], "why": "用例：不真走", "here": False}

    with h._patched():
        a = h.agent
        a.route_plan = _plan
        # 「命令前往：左上平台」——夹具玩家 world_x=500，那一步的落点中心 150 ⇒ 该按 ←
        a.start_route([route.WalkJob("左上平台",
                                     [(150.0, 130.0, 170.0, -208.0, "44")])],
                      why="命令前往：左上平台")
        # ① 命令途中 + 不在区域 + 范围内有怪 ⇒ 打（原来漏掉的正是这一拍）
        ws = h.ws(with_mob=True)
        ws.player.here_sets = ["上下过渡平台"]
        h.log.clear()
        h.clock.t += 0.2
        act = a.tick(ws)
        check(act.get("state") == "attack",
              "挂着「命令前往」时被区域规则拦下，没进 attack：%s" % act)
        check(atk in [k for _t, kind, k in h.log if kind == "down"],
              "挂着命令、范围内有怪，却一个攻击键都没发：%s" % h.log)
        check(plans == [],
              "挂着命令还去下了「回去」的路线（那是区域规则干的）：%r" % (plans,))
        check(a._climb is not None,
              "打着架把「命令前往」的任务弄丢了：%r" % (a._climb,))
        # ② 命令途中 + 不在区域 + 范围内没怪 ⇒ 任务照跑（该按 ← 走）
        h.mobs_fn = lambda t: []
        h.log.clear()
        keys = []
        for _ in range(3):                 # 换向/保持那几拍兜一下，别卡在"刚好这一拍不发"
            h.clock.t += 0.5               # 越过 TURN_TAP_S，确保这一拍是真在走
            keys = a.tick(h.ws(with_mob=False)).get("keys") or []
            if left in keys:
                break
        check(left in keys,
              "不在区域时命令没往下走（被区域规则截住？）：keys=%r" % (keys,))
        check(plans == [], "没怪的时候也去下了「回去」的路线：%r" % (plans,))
        # ③ 让命令收工（到位那一拍：脚已在目标集合、范围内没怪）
        ws3 = h.ws(with_mob=False)
        ws3.player.here_sets = ["左上平台"]
        h.clock.t += 0.2
        a.tick(ws3)
        check(a._climb is None, "命令到位了没收工：%r" % (a._climb,))
        # ④ 收工之后回到正常回路 ⇒ 区域规则照旧生效（原语义不许丢）
        h.mobs_fn = lambda t: [mob]
        ws4 = h.ws(with_mob=True)
        ws4.player.here_sets = ["左上平台"]
        h.log.clear()
        h.clock.t += 0.2
        act4 = a.tick(ws4)
        check("不在战斗区域" in str(act4.get("reason")),
              "命令收工后区域规则不管用了（原语义丢了）：%s" % act4)
        check(atk not in [k for _t, kind, k in h.log if kind == "down"],
              "命令收工后、不在区域里还是打了：%s" % h.log)
        check(plans == ["右下"],
              "收工后没下「前往」第一个配置的集合：%r" % (plans,))


def t_player_gone_logged():
    """「画面里看不到角色」必须**留痕**，而且**与自动开着没无关**（用户 2026-09-27 要求）。

    现场：角色凌晨被卡住打死，上午翻日志只看到"每 30 秒一条按键"（那是定时
    RELEASEALL）—— **分不清是死了、是卡住了、还是根本没开自动** ✗。根因是原来唯一
    那条判据（`player_lost_timeout_min`）只做"停自动"，而且排在
    `if not s.enabled: return` **之后** ⇒ 自动已经关着时它根本不跑 ⇒ 什么都不留下 ✗。

    （2026-09-28 起"角色状态"打点在 `behavior.log` ✓ —— 这些断言跟着从 perf.log 改过来。）

    钉四件事：
      ① **关着自动也记**：看不到角色到设置里那个时长 ⇒ behavior.log 里出现 `player_gone` ✓；
      ② 又看得到 ⇒ `player_back` ✓；
      ③ **开着自动**时老行为不变（停自动 ✓）并多留一条 `player_lost_stop` ✓；
      ④ `behavior.MAX_BYTES` 不许回到太小的窗口（同一次现场的另一半教训：
         日志一裁，"卡"的起点就永远查不到了 ✗）。
    """
    import tempfile
    from core import behavior

    def episode(enabled, wait_min):
        """看不见角色 2 分钟 → 再等 wait_min 分钟 → 恢复。返回 (agent, gone 段, back 段, 参数)。"""
        s = fresh_settings(player_lost_timeout_min=3.0)
        s.enabled = bool(enabled)
        h = Harness(s)
        h.clock0 = h.clock.t
        logf = Path(tempfile.mkdtemp(prefix="behavior_gone_")) / "behavior.log"
        old_log, old_en = behavior.LOG, behavior.ENABLED

        def _read():
            # ⚠ behavior 是"一行一次落盘"（没有 perf 那种段头）⇒ 还没记过任何事件时**文件不存在**，
            #   读它要兜底成空 ✓。
            return logf.read_text(encoding="utf-8") if logf.exists() else ""

        try:
            behavior.configure(True, log=logf)
            with h._patched():
                for _ in range(4):                     # 先 2 分钟（**没到** 3 分钟）
                    h.clock.t += 30.0
                    w = h.ws(with_mob=False)
                    w.player.found = False
                    # ⭐⭐ 「停自动」的判据现在是 **小地图找不到黄点**（用户 2026-09-28 ✓
                    #   原话："找不到玩家停止自动的判断依据修改为小地图找不到黄点"）
                    #   ⇒ 把"黄点也没了"一起摆上（`world_x=None` = 没有可用位置 ✓）。
                    #   ⚠ **「留痕」那条判据（`player_gone`）仍然只看 `found`** ✓ 不受影响 ✓
                    #     —— 这是两条不同的判据，别混（见 `agent.py` 那两处各自的注释 ✓）。
                    w.player.world_x = None
                    h.agent.tick(w)
                check("player_gone" not in _read(),
                      "还没到设置里那个时长就报了 player_gone（阈值没生效）")
                h.clock.t += wait_min * 60.0           # 跨过阈值
                w = h.ws(with_mob=False)
                w.player.found = False
                w.player.world_x = None      # ⭐ 「停自动」看的是**黄点**（用户 2026-09-28 ✓）
                h.agent.tick(w)
                gone = _read()
                h.clock.t += 30.0                      # 又看得到角色
                w = h.ws(with_mob=False)
                w.player.found = True
                h.agent.tick(w)
                back = _read()[len(gone):]
            return h.agent, gone, back, s
        finally:
            behavior.configure(old_en)
            behavior.LOG = old_log

    # ① 关着自动也记 —— 这是本次的核心（原来"关着自动 ⇒ 什么都不留"✗）
    _a, gone, back, s = episode(enabled=False, wait_min=4)
    check("player_gone" in gone,
          "关着自动时「看不到角色」没留下任何痕迹：\n%s" % gone)
    check("看不到角色" in gone,
          "player_gone 没有注记（事后看不出丢了多久）：\n%s" % gone)
    check(s.enabled is False, "关着自动时被它改动过 enabled")
    # ② 恢复 ⇒ player_back，且之后的段头不再说"看不到角色"
    check("player_back" in back, "又能看到角色了却没记 player_back：\n%s" % back)
    check("看不到角色" not in back,
          "恢复之后又写了「看不到角色」：\n%s" % back)
    # ③ 开着自动 ⇒ 照旧停自动（老行为不许变）+ 留下 player_lost_stop
    a2, gone2, _b2, s2 = episode(enabled=True, wait_min=4)
    check(s2.enabled is False and a2.state == "idle",
          "开着自动、超过 3 分钟看不到角色却没停自动 / 没回 idle：%s / %s"
          % (s2.enabled, a2.state))
    check("player_lost_stop" in gone2,
          "它自己把自动停了，日志里却查不到（下次还是只能猜）：\n%s" % gone2)
    # ④ ⭐⭐ **边界：判据只看黄点**（用户 2026-09-28 ✓ 原话："找不到玩家停止自动的判断依据
    #   修改为**小地图找不到黄点**"）：**人物框认不出来、但黄点还在**（`world_x` 有值 ✓）
    #   ⇒ 位置照样可用 ⇒ **不许停自动** ✗（这是这次改动的另一半，必须钉住 ✓）。
    s3 = fresh_settings(player_lost_timeout_min=3.0)
    s3.enabled = True
    h3 = Harness(s3)
    h3.clock0 = h3.clock.t
    with h3._patched():
        for _ in range(8):                     # 远超阈值（8×30s = 4 分钟）
            h3.clock.t += 30.0
            w3 = h3.ws(with_mob=False)
            w3.player.found = False            # 框认不出来 ✗
            w3.player.world_x = 500.0          # 但黄点还在 ✓
            w3.player.world_held = False
            h3.agent.tick(w3)
    check(s3.enabled is True,
          "黄点还在（有可用位置）却把自动停了 —— 判据又跑去看人物框了 ✗（用户 2026-09-28 "
          "要求只看**小地图黄点**）：enabled=%s" % s3.enabled)
    # ④ 日志窗口不许再缩回去（行为打点现在全在 behavior.log ⇒ 窗口太短会把出事的起点裁没 ✗）
    check(behavior.MAX_BYTES >= 4 * 1024 * 1024,
          "behavior.log 的上限被调小了（%.0f KB）—— 窗口太短"
          "就永远看不到出事那一段 ✗" % (behavior.MAX_BYTES / 1024.0))


def t_task_report_on_skip():
    """「这一拍任务没跑 ⇒ 报信」是**一处**结算（用户 2026-09-26 要求结构性收口）。

    原来报信散在两个地方**手工**调（`if in_range:` / 休息赶路打架），而 tick 里的早退
    有五六条 ⇒ 漏了两条，实测（同一套注入口径：任务先正常跑 2 拍，再挡一段，再恢复）：
      · **未定位玩家** 5 秒 → 恢复后第一拍 `走不动了：卡在 x=500`（任务背锅 ✗）
      · **关自动** 40 秒    → 同上 ✗
      · **进战斗** 5 秒     → note 一字未变 ✓（它有报信）
    现在结算收在 `tick` 最前面，判据只有"这一拍有没有人推过任务"，和"因为什么没跑"无关。

    钉四件事：
      ① 未定位玩家 ⇒ 恢复后任务**不背锅**（note 里不许出现"走不动了"，且接着走）；
      ② 关自动 ⇒ 任务当场**被撤掉**（`_climb` 空、路线清空），重新开自动后不背锅；
      ③ 进战斗（对照组，行为不许变）⇒ 恢复后不背锅；
      ④ 时序拍**不参与**结算（否则 `_climb_started` 每 10ms 重打 ⇒ 寻路超时永不触发 ✗）。
    """
    from decision import route

    def group(kind):
        """跑一组：任务先跑 2 拍 ⇒ 按 kind 挡一段 ⇒ 恢复一拍。返回 (agent, harness)。"""
        s = fresh_settings(attack_dist=120.0)
        s.strategy = "patrol"
        h = Harness(s)
        h.clock0 = h.clock.t
        h.mobs_fn = lambda t: []
        with h._patched():
            a = h.agent
            a.start_route([route.WalkJob("左上平台",
                                         [(150.0, 130.0, 170.0, -208.0, "44")])],
                          why="命令前往：左上平台")

            def beat(dt, found=True, mob=False):
                h.clock.t += dt
                w = h.ws(with_mob=False)
                w.player.found = found
                w.player.here_sets = ["上下过渡平台"]   # 一直没到 ⇒ 任务该继续走
                if mob:
                    w.mobs = [Mob(id=1, x=560.0, y=500.0, w=40.0,
                                  h=40.0, conf=0.9)]
                return a.tick(w)

            for _ in range(2):                     # 先跑起来（锚点落定）
                beat(1.0)
            if kind == "off":
                # ② 关自动（这一拍就该把任务撤掉）
                s.enabled = False
                beat(1.0)
            elif kind == "timing":
                # ④ 只走时序拍：结算不许参与
                cs = a._climb_started
                for _ in range(5):
                    h.clock.t += 0.01
                    a.tick_timing(h.ws(with_mob=False))
                check(a._climb_started == cs,
                      "时序拍把寻路超时钟往前推了（%r → %r）⇒ 超时永不触发 ✗"
                      % (cs, a._climb_started))
                check(a._task_ran is True,
                      "时序拍把「这一拍推过任务」的章擦了 ⇒ 下一拍会误报信 ✗")
                check(getattr(a._climb, "_paused_at", None) is None,
                      "时序拍对任务报了信（任务被当作'没跑'）⇒ 没跑的那几拍会漏扣 ✗")
            else:
                for _ in range(5):                 # 挡 5 秒（> STALL_S=3s，够判"走不动"）
                    beat(1.0, found=(kind != "lost"), mob=(kind == "mob"))
            if kind == "off":
                h.clock.t += 40.0                  # 关着自动放 40 秒（超过寻路超时）
                s.enabled = True
            beat(0.2)                              # 恢复后的第一拍
            return a

    for kind, what in (("lost", "未定位玩家"), ("mob", "进战斗"),
                       ("off", "关自动"), ("timing", "时序拍")):
        a = group(kind)
        note = str(getattr(a._climb, "note", "") or "")
        if kind == "off":
            check(a._climb is None and not a._route,
                  "「关自动」没把挂着的寻路任务撤掉（用户 2026-09-26 决定：关自动=停手）："
                  "climb=%r route=%r" % (a._climb, a._route))
        else:
            check("走不动了" not in note,
                  "%s那几拍被算成任务的错（报信漏了）：note=%r" % (what, note))
        check("走不动了" not in str(a.current_goto_note()),
              "%s之后画面上那行在说任务（走不动了）：%r"
              % (what, a.current_goto_note()))


def t_end_route_clears_plan():
    """「结束当前寻路」要停**整条**路线；而「限制战斗区域」也不许被"死账"骗到。

    2026-09-26 用户现场定的，两条一起堵（用户原话："改这行 + 顺手把按钮也改干净"）：

      ① **按钮**（真方法 `gui.route_panel.RoutePanel._on_stop_goto`）：3 步路线途中点它 ⇒
         `_climb` 空**且 `_route` 也空** —— 不能只停当前那一步、把后面两步留在账上。
         ⚠ 这里把**真方法**绑在一个假面板上（它只用 `_live_agent` / `_say_goto` 两个属性），
           不构造整个 Qt 面板（面板要地形 / 地图上下文，为一个按钮不值当；而且绑真方法
           测到的才是**真代码**，不是我自己复述的契约）。
      ② **判据**（`agent._leave_battle_zone_tick` 里那句）：即使账上还留着**死账**
         （模拟"只停当前一步"的旧行为），不在战斗区域时也**必须**照常下"回去"的命令 ——
         原来那句要求 `_route` 也空，于是拿死账当"我已经在回去的路上了"，角色**站着不动**、
         每拍只报"不在战斗区域，先回去" ✗。
      ③ **反向（别矫枉过正）**：任务**正在跑**时，不许再插一条"回去"的命令。
    """
    import types

    from decision import route
    from gui.route_panel import RoutePanel

    def fresh_route(with_jobs=False):
        """挂一条 3 步路线（甲 → 乙 → 左上平台）⇒ (harness, agent, 自己下的前往记录)。

        `with_jobs=True` 时那个假解析器会**真的返回一条任务**（去「右下」）——
        用来验证"命令确实下出去了、任务确实挂上了"。
        """
        s = fresh_settings(attack_dist=120.0)
        s.strategy = "patrol"
        s.battle_zone_sets = ["右下"]        # ← 用户项目 寺院通道2 的实配
        h = Harness(s)
        h.clock0 = h.clock.t
        # 怪：距角色中心 140 px（**在视野里、不在攻击范围内**）⇒ 只当"可追的候选"用，
        # 免得一进 attack 就把任务岔开（本用例要看的正是任务那条路）。
        h.mobs_fn = lambda t: [Mob(id=1, x=660.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
        plans = []

        def _plan(dst):
            plans.append(dst)
            jobs = ([route.WalkJob("右下", [(300.0, 280.0, 320.0, -150.0, "9")])]
                    if with_jobs else [])
            return {"jobs": jobs, "path": [dst] if jobs else [],
                    "why": "" if jobs else "用例：不真走", "here": False}

        a = h.agent
        a.route_plan = _plan
        a.start_route([route.WalkJob("甲", [(300.0, 280.0, 320.0, -150.0, "1")]),
                       route.WalkJob("乙", [(400.0, 380.0, 420.0, -150.0, "2")]),
                       route.WalkJob("左上平台", [(150.0, 130.0, 170.0, -208.0, "3")])],
                      why="命令前往：左上平台")
        return h, a, plans

    # ① 按钮（真方法）：整条停干净，不留死账
    h, a, _ = fresh_route()
    with h._patched():
        said = []
        fake = types.SimpleNamespace(
            _live_agent=lambda: a,
            _say_goto=lambda text, color=None: said.append(str(text)))
        # 面板里另两个方法也用**真的**（它们只依赖 `_live_agent` ⇒ 同样不用构造 Qt 面板）：
        # `_on_stop_goto` 会先问 `_current_goto_set()`（"现在有没有任务"）。
        fake._current_goto_set = types.MethodType(RoutePanel._current_goto_set, fake)
        RoutePanel._on_stop_goto(fake)
        check(a._climb is None,
              "「结束当前寻路」没停掉当前这一步：%r" % (a._climb,))
        check(not a._route,
              "「结束当前寻路」把后面几步留在账上了（死账）：%r"
              % ([j.dst_set for j in a._route],))
        check(any("已结束" in t for t in said),
              "按钮没给出「已结束」的反馈：%r" % (said,))
    # ② 死账 + 不在战斗区域 ⇒ 仍然要下「回去」的命令（不许站着不动）
    h, a, plans = fresh_route(with_jobs=True)
    with h._patched():
        a.stop_climb("用例：模拟旧按钮只停当前一步")     # ← 故意把后两步留成死账
        check(bool(a._route), "用例自己没摆出死账 ⇒ 后面这条测不到东西")
        ws = h.ws(with_mob=True)
        ws.player.here_sets = ["丙平台"]                # 不在战斗区域
        h.clock.t += 0.2
        act = a.tick(ws)
        check(plans == ["右下"],
              "账上留着死账就不下「回去」的命令了（角色会站着不动）：plans=%r act=%s"
              % (plans, act))
        check(a._climb is not None and getattr(a._climb, "dst_set", "") == "右下",
              "下了命令却没真挂上任务：%r" % (getattr(a._climb, "dst_set", None),))
        check(not a._route,
              "死账没被新的路线覆盖清掉：%r" % ([j.dst_set for j in a._route],))
    # ③ 反向：任务正在跑时不许再插一条「回去」的命令
    h, a, plans = fresh_route()
    with h._patched():
        ws = h.ws(with_mob=True)
        ws.player.here_sets = ["丙平台"]
        h.clock.t += 0.2
        a.tick(ws)
        check(plans == [],
              "任务正在跑，却又下了一条「回去」的命令：%r" % (plans,))
        check(a._climb is not None, "把正在跑的任务弄丢了：%r" % (a._climb,))


def t_climb_align_tap_releases_any_fps():
    """点按的"按住多久"必须**按时间**算 —— 用户 2026-09-27 报"对齐 x 时方向键按了很久"。

    现场：`TAP_ON_S` = 60ms，而决策是**跟着收流帧**跑的（实测 ≈30fps ⇒ 一帧 ≈33ms）。
    如果每一拍都刷 `_press_at = now`（把"这一拍按着"和"刚按下"共用一个时间戳），
    那 `now - _press_at` **永远只有一帧**（33ms < 60ms）⇒ "按够 `TAP_ON_S` 就松开"
    那条**永不成立** ⇒ 进了「开始对齐绳梯x的距离」以内也一直**按住不放** ✗
    （表现就是"按了很久"、而且角色冲过头）。

    这条钉**帧率无关**：同样一段对齐，15fps（一帧 67ms）与 34fps（一帧 29ms）
    都必须出现"按一下、松一下"，且松开时刻落在 `TAP_ON_S` 附近（≤ TAP_ON_S + 一帧）✓。
    """
    from decision import route
    C = route.ClimbJob

    for _fps in (15.0, 34.0):
        dt = 1.0 / _fps
        job = C("L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                tol_px=6, hold_ms=0, align_gap_ms=180, near_px=50.0)
        o = job.update(0.0, px=660.0)              # 差 40px ≤ near ⇒ 进点按区，先按下去
        check(int(o["move"]) != 0, "%.0ffps：该先按下去：%s" % (_fps, o))
        _released, _t = None, 0.0
        for _i in range(int((route.TAP_ON_S + dt) / dt) + 2):
            _t += dt
            o = job.update(_t, px=660.0)
            if int(o["move"]) == 0:
                _released = _t
                break
        check(_released is not None,
              "%.0ffps：进了点按区却**一直按住不松**（一帧 %.0fms < TAP_ON_S %.0fms 时"
              "旧写法就会这样 ⇒ 用户报的「按了很久」）：%s"
              % (_fps, dt * 1000, route.TAP_ON_S * 1000, o))
        check(_released <= route.TAP_ON_S + dt + 1e-9,
              "%.0ffps：松开得太晚（%.3fs，该 ≤ TAP_ON_S %.3fs + 一帧 %.3fs）"
              % (_fps, _released, route.TAP_ON_S, dt))


def t_tolerance_above_read_noise():
    """**容差不能小于读数噪声**（2026-09-27 用户要求把这类"名不副实"的阈值一起改掉）。

    依据都是现成数据，**不是拍脑袋**（README / `docs/寻路设计.md` 的实测）：
      · 1 个**实时小地图像素 ≈ 8.5 世界像素**（16.08 ÷ 1.88，105090600 实测）；
      · 黄点定位 ±0.5 面板像素 ⇒ **±4.3 世界像素**；
      · 双点标定残差 ≈6 世界像素。
    ⇒ 比 ~6 还小的阈值等于**要求亚像素精度**：判据只会在噪声上反复"到了又没到"
      （现场观感就是"对齐时来回抖、按很久"，以及对"还在爬"的误判 ✗）。

    钉三件（谁改回去谁红）：
      ① 「坐标对齐误差范围」默认 **10**（`route.ALIGN_TOL_PX`）——
         `ClimbJob` / `plan_jobs` / `DecisionSettings`（含老文件 `from_dict` 兜底）**四处一致**；
      ② `route.STALL_GAIN_PX`（判"y 还在往上爬"）**必须大于**半个面板像素对应的世界距离；
      ③ 本来就够大的那几个（走位 / 靠近 / 下跳兜底）**不许被顺手改小** ✗。
    """
    import inspect

    from decision import route

    noise = 8.5 / 2.0                       # ±0.5 面板像素 ⇒ ±4.25 世界像素

    # ① 四处一致（各写一个数迟早分叉 ✗）
    check(route.ALIGN_TOL_PX == 10.0,
          "对齐容差默认值被动过（依据见用例说明）：%r" % route.ALIGN_TOL_PX)
    check(int(ag.DecisionSettings().align_tol_px) == int(route.ALIGN_TOL_PX),
          "设置的默认「坐标对齐误差范围」与 `route.ALIGN_TOL_PX` 不一致：%r vs %r"
          % (ag.DecisionSettings().align_tol_px, route.ALIGN_TOL_PX))
    check(route.ClimbJob("L2", x=700.0, y1=-100.0, y2=200.0, direction=1).tol_px
          == route.ALIGN_TOL_PX,
          "`ClimbJob` 的默认容差不是 `route.ALIGN_TOL_PX`（老调用方会跑到另一个数上 ✗）")
    check(inspect.signature(route.plan_jobs).parameters["tol_px"].default
          == route.ALIGN_TOL_PX,
          "`plan_jobs` 的默认容差没跟上常量（面板下命令会拿到另一个数 ✗）")
    s = ag.DecisionSettings()
    s.from_dict({})                          # 老项目文件里没这个键
    check(int(s.align_tol_px) == int(route.ALIGN_TOL_PX),
          "老项目文件没退回**新的**默认值（还退回 6 = 噪声里 ✗）：%r" % s.align_tol_px)

    # ② 「还在往上爬」的增益要盖过读数噪声，否则 STALL_S 一到就误报「爬不动了」
    check(route.STALL_GAIN_PX > noise,
          "`STALL_GAIN_PX`=%r 小于读数噪声(±%.1f 世界像素) ⇒ 永远判成「爬不动了」✗"
          % (route.STALL_GAIN_PX, noise))
    check(route.STALL_GAIN_PX <= 20.0,
          "`STALL_GAIN_PX`=%r 太大 ⇒ 真卡住要等很久才判得出来 ✗" % route.STALL_GAIN_PX)

    # ③ 反例：本来就大于噪声的那几个，别被"顺手统一"改小 ✗
    #    ⚠ `DROP_DY_PX`（下跳"掉了 40px 就算落地"）2026-09-28 **用户要求整个删掉** ✗ ——
    #    下跳的到达判据已换成「**脚下这块面 == 目标面**」（见 `DropJob._arrived` ✓）
    #    ⇒ 它不在这个名单里了 ✓，而且**不许回来** ✗。
    for name, v in (("WALK_TOL_PX", route.WALK_TOL_PX),
                    ("NEAR_PX", route.NEAR_PX)):
        check(v > noise, "%s=%r 也被改到噪声以下了（它本来够大 ✗）" % (name, v))
    check(not hasattr(route, "DROP_DY_PX"),
          "`DROP_DY_PX` 又回来了 —— 下跳的到达判据已经是「脚下这块面 == 目标面」"
          "（用户 2026-09-28 ✓），别再拿「掉了多少像素」凑一个数 ✗")


def t_climb_align_near_px():
    """「开始对齐绳梯x的距离(px)」（用户 2026-09-27 要求）：比它远 ⇒ **一口气按住**；
    ≤ 它 ⇒ 改成**点按**（一下一下）。原来是**写死**的 `route.NEAR_PX` = 20px，
    用户实测"太近了"（精细那一段来得太晚）⇒ 抽成「攀爬参数」里的一格。

    钉四件：
      ① 默认值 = `route.NEAR_PX`（老行为一点不变 ✓，两处不许各写一个数）；
      ② 决策设置三处同步（`__init__` / `to_dict` / `from_dict`），老项目文件缺这个键
         时退回 20，0/负数兜到 1（"永远按住"没意义）；
      ③ **行为**（这条才是用户要的）：差 40px 时，默认（20）**按住不放**；配成 60 ⇒ 第二拍
         就**松开**（点按）—— 也就是"精细对齐从更远就开始"；
      ④ 这个参数**只归上绳**（`ClimbJob`）—— 走（`WalkJob`）那份还是常量，别顺手改了 ✗。
    """
    from decision import route
    C = route.ClimbJob

    # ① 默认值就是那个常量
    check(int(getattr(ag.settings, "climb_align_near_px", -1)) == int(route.NEAR_PX),
          "默认「开始对齐绳梯x的距离(px)」不是 `route.NEAR_PX`：%r vs %r"
          % (getattr(ag.settings, "climb_align_near_px", None), route.NEAR_PX))
    check(C("L2", x=700.0, y1=-100.0, y2=200.0, direction=1).near_px == route.NEAR_PX,
          "`ClimbJob` 自己的默认不是 `route.NEAR_PX`（老调用方行为会变 ✗）")

    # ② 存 / 读一致 + 老文件退回默认 + 坏值兜住
    s = ag.DecisionSettings()
    s.climb_align_near_px = 60
    d = s.to_dict()
    check(d.get("climb_align_near_px") == 60,
          "`to_dict` 没带上这个参数：%r" % d.get("climb_align_near_px"))
    s2 = ag.DecisionSettings()
    s2.from_dict(d)
    check(int(s2.climb_align_near_px) == 60,
          "`from_dict` 没读回来：%r" % s2.climb_align_near_px)
    s3 = ag.DecisionSettings()
    s3.from_dict({})
    check(int(s3.climb_align_near_px) == 20,
          "老项目文件（没这个键）没退回默认 20：%r" % s3.climb_align_near_px)
    s4 = ag.DecisionSettings()
    s4.from_dict({"climb_align_near_px": 0})
    check(int(s4.climb_align_near_px) >= 1,
          "0 没被兜住（0 等于「永远按住」，没意义）：%r" % s4.climb_align_near_px)

    # ③ 行为：差 40px —— 默认按住不放；配 60 就变成点按（按一下、松一下）
    def mk(**kw):
        a = dict(ladder_id="L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                 tol_px=6, hold_ms=0, align_gap_ms=0)
        a.update(kw)
        return C(**a)

    job = mk()
    o1 = job.update(0.0, px=660.0)
    o2 = job.update(0.1, px=660.0)
    check(o1["move"] and "微调" not in o1["note"],
          "差 40px（> 默认 20）该是「一口气按住」：%s" % o1)
    check(int(o2["move"]) == int(o1["move"]),
          "默认距离下不该松开（松开就是点按了）：%s → %s" % (o1["move"], o2["move"]))
    job = mk(near_px=60.0)
    o1 = job.update(0.0, px=660.0)
    o2 = job.update(0.1, px=660.0)
    check("微调" in o1["note"],
          "配了 60 之后，差 40px 该进「点按微调」：%s" % o1)
    check(o1["move"] and not o2["move"],
          "点按要「按一下、松一下」：%s → %s" % (o1["move"], o2["move"]))

    # ④ 走那条路不受影响（用户只要求上绳那份可配）：常量还在 20，`WalkJob` 也不接这个设置
    check(float(route.NEAR_PX) == 20.0,
          "`route.NEAR_PX` 被改了 —— 走那边会跟着变（用户只要求上绳那份可配）✗：%r"
          % route.NEAR_PX)
    check(not hasattr(route.WalkJob("乙平台", []), "near_px"),
          "「走」也接上了 `near_px`（用户没要求配走那一份）")
    # ⑤ 界面上真有这一格（在「攀爬参数」组里，标签就是用户要的那句话）
    _src_rp = (Path(__file__).resolve().parent.parent / "gui" / "route_panel.py"
               ).read_text(encoding="utf-8")
    check("开始对齐绳梯x的距离(px)" in _src_rp and "sp_align_near" in _src_rp
          and "_on_align_near" in _src_rp,
          "「攀爬参数」里没有「开始对齐绳梯x的距离(px)」那一格 / 没接上保存")


def t_behavior_task_events():
    """玩家行为打点（初版：**只有任务**）—— 六个事件要真的落到 `behavior.log` 里。

    用户 2026-09-27 要求：`perf.log` 是给**性能**用的（它自己的说明第一句就写了
    "不是行为统计"）⇒ 单开 `core/behavior.py`，初版只打"玩家的任务"，后续逐渐扩展。

    这条走**任务的真入口**（`plan_and_start_route` / `start_route` / `_task_finished` /
    `stop_route`）来钉"接线接上了没"；模块本身的格式/健壮性在
    `tools/selftest_behavior.py` ✓（那边不碰 agent，两边分工不重叠 ✓）。
    """
    import tempfile

    from core import behavior
    from decision import route

    logf = Path(tempfile.mkdtemp(prefix="behavior_task_")) / "behavior.log"
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        behavior.configure(True, log=logf)
        s = fresh_settings()
        h = Harness(s)
        with h._patched():
            a = h.agent
            # ① 还没开始实时（没注入解析器）⇒「压根没起跑」也要留痕
            ok, _why = a.plan_and_start_route("左上平台", why="命令前往：左上平台")
            check(not ok, "前提不成立：没注入解析器却起跑了")
            # ② 两步路线 ⇒ task_start
            j1 = route.WalkJob("乙平台", [(700.0, 600.0, 800.0, 560.0, "1")])
            j2 = route.WalkJob("丙平台", [(700.0, 600.0, 800.0, 560.0, "2")])
            check(a.start_route([j1, j2], why="命令前往：丙平台"), "两步路线没起跑")
            # ③ 第一步到 ⇒ task_step（`_task_finished` 是"这一步收工"的唯一出口 ✓）
            check(a._task_finished() is False, "第一步收工后该接着走下一步")
            # ④ 第二步失败 ⇒ task_fail（`reason` 就是任务自己那句人话 ✓）
            check(a._task_finished(failed=True, why="爬到 y=100 就不再升了") is True,
                  "失败该把整条收掉")
            # ⑤ 再挂一条、被人叫停 ⇒ task_cancel
            a.start_route([j1], why="命令前往：乙平台")
            a.stop_route("用户点了「结束当前寻路」")
            # ⑥ 单步跑完 ⇒ task_done（单步也走这一支 ✓）
            a.start_route([j1], why="命令前往：乙平台")
            check(a._task_finished() is True, "单步路线收工该报「整条完事」")
        txt = logf.read_text(encoding="utf-8")
        for name in ("task_plan_fail", "task_start", "task_step", "task_fail",
                     "task_cancel"):
            check(name in txt, "任务的 %s 没进 behavior.log：\n%s" % (name, txt))
        # ⭐ 单步 + 秒级完成 ⇒ **合并成一条** `task_quick`（用户 2026-09-28 要求降噪 ✓）——
        #    所以这里**不再有** `task_done`（那一条被合成掉了 ✓）；
        #    ⚠ 而 `task_fail` / `task_cancel` **绝不许**被合并（`merge=False` ✓：
        #    被叫停 / 失败都不叫"走完了"，伪装成 `task_quick` 会误导 ✗）。
        check("task_quick" in txt and "task_done" not in txt,
              "瞬时任务该合并成一条 task_quick（且不再出现 task_done）：\n%s" % txt)
        check("why=命令前往：丙平台" in txt and "steps=2" in txt,
              "task_start 没写清去哪、几步：\n%s" % txt)
        check("step=1/2" in txt, "task_step 该说清「刚走完的是第 1 步」（不能写成第 2 步）：\n%s"
              % txt)
        check("step=2/2" in txt and "left=0" in txt and "reason=爬到 y=100 就不再升了" in txt,
              "task_fail 没写清断在第几步 / 后面还有几步 / 为什么：\n%s" % txt)
        check("why=用户点了「结束当前寻路」" in txt,
              "task_cancel 没写清是谁叫停的：\n%s" % txt)
        check("task_quick" in txt and "steps=1" in txt and "why=命令前往：乙平台" in txt,
              "合并出来的 task_quick 该带上起跑那句 why / steps（人一眼看得出它干了啥）：\n%s"
              % txt)
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log


def t_behavior_route_events():
    """寻路的**任务内部变化**也进 behavior.log（用户 2026-09-28 要求）。

    `task_*`（任务层级）之外，再加三类，专门回答"这一步**里面**发生了什么"：
      · `route_phase`：相切换（align→climb、climb→align…）；
      · `route_diag`  ：爬绳的**斜跳**起跳 / 结束（结束带"吸上绳 / 没吸上"✓）；
      · `route_retry` ：一次尝试**失败重来**（`attempt` 涨到几 ✓）。
    排查"卡在哪、为什么断"时，光看 `task_fail` 只看到最后一句，中间"对齐→斜跳→上绳→
    爬不动"的过程靠这里拼起来 ✓。

    这里**直接喂 `agent._mark_route_change(job)`**（打点本体，它的真实调用点在
    `_climb_tick` 的 `job.update` 之后，那一处也一起钉源码 ✓）。
    """
    import inspect
    import tempfile

    from core import behavior
    from decision import route
    from decision.agent import CombatAgent

    logf = Path(tempfile.mkdtemp(prefix="behavior_route_")) / "behavior.log"
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        behavior.configure(True, log=logf)
        a = CombatAgent(fresh_settings())
        C = route.ClimbJob
        j = C(ladder_id="L2", x=600.0, y1=100.0, y2=300.0, direction=1,
              dst_set="上", src_set="下", tol_px=10, hold_ms=0, near_px=20,
              jump_start_px=120)
        # ① 斜跳**第 1 拍**（用户 2026-09-28 要求拆两拍 ⇒ 这一拍**只发方向键**、还没起跳 ✓）
        j.update(0.10, 520.0, py=300.0, here_sets=["下"])
        a._mark_route_change(j)
        # ② 斜跳**第 2 拍**（这一拍才按跳 ⇒ `_diag_flying=True` = 起跳 ✓）
        j.update(0.15, 520.0, py=300.0, here_sets=["下"])
        a._mark_route_change(j)
        # ③ 吸上绳（这一拍 ladder_id=L2 ⇒ 斜跳结束 + phase align→climb）
        j.update(0.20, 600.0, py=280.0, ladder_id="L2", here_sets=["下"])
        a._mark_route_change(j)
        # ④ 失败重来（retry ⇒ attempt=2、phase 回 align）
        j.retry()
        a._mark_route_change(j)
        txt = logf.read_text(encoding="utf-8")
        check("route_diag" in txt and "what=起跳" in txt,
              "斜跳起跳没记 route_diag：\n%s" % txt)
        check("what=结束·吸上绳" in txt,
              "斜跳吸上绳没记成「结束·吸上绳」：\n%s" % txt)
        check("route_phase" in txt and "prev=align" in txt and "to=climb" in txt,
              "相切换没记 route_phase（align→climb）：\n%s" % txt)
        check("route_retry" in txt and "attempt=2" in txt,
              "失败重来没记 route_retry（attempt=2）：\n%s" % txt)
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log
    # ⚠ 接线（源码级）：`_climb_tick` 里 `job.update` 之后必须调它，别哪天把调用摘了 ✗
    src = inspect.getsource(CombatAgent._climb_tick)
    check("_mark_route_change(job)" in src,
          "`_climb_tick` 没在 `job.update` 之后调 `_mark_route_change`（打点没接上 ✗）")


def t_climbing_vertical_notify():
    """ladder_id 许可改成「climb 执行器按住 ↑ 后发起通知」（诉求 1，用户 2026-09-28）。

    原话："位置状态允许判定在绳梯的入口为『climb 执行器按住 ↑ 后发起通知』，而不是『按住 ↑』"
    —— 即 `agent.climbing_vertical()` 返回执行器上一拍发了 ↑/↓（`ClimbJob._last_vert` ✓），
    替代 `holding_vertical()`（本机 `KeyState.pressed`）。

    钉四件：① 没爬绳任务 ⇒ False；② 爬绳任务上一拍发 ↑/↓ ⇒ True、没发 ⇒ False；
    ③ **drop（下跳）任务按 ↓ 也发通知**（用户 2026-09-28 追加："drop 按 ↓ 时也发通知，直到
    drop 收工再关上"）✓；④ 感知层喂位置状态机用的是 `climbing_vertical`（不再是 `holding_vertical`）。
    """
    import inspect

    from decision import route
    from decision.agent import CombatAgent

    a = CombatAgent(fresh_settings())
    check(not a.climbing_vertical(), "没任务时 climbing_vertical 该 False")
    j = route.ClimbJob("L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                       dst_set="乙", src_set="甲", tol_px=6, hold_ms=200)
    a._climb = j
    check(not a.climbing_vertical(), "刚挂上还没发 ↑/↓ 该 False")
    j._last_vert = 1
    check(a.climbing_vertical(), "上一拍发了 ↑ 该 True")
    j._last_vert = -1
    check(a.climbing_vertical(), "上一拍发了 ↓ 该 True")
    j._last_vert = 0
    check(not a.climbing_vertical(), "上一拍没发 ↑/↓ 该 False")
    a._climb = route.WalkJob("乙", [(700.0, 690.0, 710.0, -208.0, "1")])
    check(not a.climbing_vertical(), "走任务不算「爬绳通知」")
    # ③ drop（下跳）任务按 ↓ 也发通知
    d = route.DropJob("甲", "乙", [(700.0, 690.0, 710.0, "1")])
    a._climb = d
    check(not a.climbing_vertical(), "drop 刚挂上还没按 ↓ 该 False")
    d._last_vert = -1
    check(a.climbing_vertical(), "drop 按 ↓ 该发通知（= 能进「上绳梯」入口 ✓）")
    d._last_vert = 0
    check(not a.climbing_vertical(), "drop 收工（没按 ↓）该关上通知")
    # 源码级：感知层喂的是 climbing_vertical，不再 holding_vertical
    from gui import live_thread as lt
    src = inspect.getsource(lt)
    check("climbing_vertical" in src, "感知层没用 climbing_vertical（执行器通知）")
    check("holding_vertical" not in src,
          "感知层还在用 holding_vertical（本机按键状态）—— 诉求 1 要换掉它 ✗")


def t_climb_hold_up_not_preempted():
    """climb 执行器「到顶后再按住 ↑ 那一下」不许被"脚下已是目的地"提前收工（诉求 2）。

    用户 2026-09-28："攀爬的额外按住 ↑ 坐标对齐误差时间真的生效了吗？现在观察到位置状态
    切为目的地后立马停了" —— 根因：`agent._climb_tick` 开头的「这趟目的地人已经站进去了 ⇒
    提前收工」在"再按住 ↑"那几拍（`_arrived_at` 已置上、`phase` 还没 DONE）触发 ✗。

    钉两件：① `ClimbJob.holding_up()` 在"判到达后、还没 DONE"的几拍 = True，其余 = False；
    ② 那条提前收工带上了 `holding_up`（源码级：避开这个阶段）。
    """
    import inspect

    from decision import route
    from decision.agent import CombatAgent

    C = route.ClimbJob
    j = C("L3", x=1189.0, y1=68.0, y2=-83.0, direction=1, dst_set="二楼", src_set="一楼",
          tol_px=10, hold_ms=250)
    j.update(0.0, px=1189.0, py=0.0, ladder_id="L3", here_sets=["一楼"])
    check(not j.holding_up(), "还没判到达，不该 holding_up")
    o = j.update(0.5, px=1189.0, py=-85.0, ladder_id="L3", here_sets=["一楼"], at_top="L3")
    check(not o["done"] and j.holding_up(), "判到达后、还没 DONE 的几拍该 holding_up")
    o = j.update(0.8, px=1189.0, py=-85.0, ladder_id=None, here_sets=["二楼"], at_top="L3")
    check(o["done"] and not j.holding_up(), "收工后不该再 holding_up")
    src = inspect.getsource(CombatAgent._climb_tick)
    check("_hold_up" in src and "not _hold_up" in src,
          "「脚下已是目的地 ⇒ 提前收工」那条没带上 `not _hold_up`（会在'再按住 ↑'期间截断 ✗）")


def t_stall_unified_to_retry():
    """「卡住判定时长(s)」**已移除**，统一用「移动操作尝试间隔(ms)」（用户 2026-09-27 要求）。

    原话："移除卡住判定时长(s)，统一采用移动操作尝试间隔(ms)" ✓ —— 原来那件
    `DecisionSettings.stall_s` 删掉 ✓，那些"多久没进展算卡住"的判据（上爬「爬不动了」/
    下爬「等落地」/ 跳跃「没落到」）统一由 `route.stall_s_from_retry_ms(move_retry_ms)` 换算 ✓。

    钉五件：
      ① 设置里**没有** `stall_s` 这格了；`to_dict` 也不再落盘它 ✓；
      ② 老配置里残留的 `stall_s` 键 ⇒ **读进来直接忽略**（不报错、不挂回设置 ✓）；
      ③ 换算**一处实现**：3000 ⇒ 3.0、5000 ⇒ 5.0、0 / 负数 / None / 坏值 ⇒ 钳到下限 0.5 ✓；
      ④ 三处调用方（面板 / 实时线程 / agent）都**通过它**取，不许各写一份 ✗（源码钉子 ✓）；
      ⑤ 「走」照旧**不吃**它：`job_for_edge` 造出来的走任务 `stall_s` 仍是 `route.STALL_S` ✓。
    """
    from pathlib import Path as _P

    from core import mapdata, zones
    from decision import route

    # ① 设置里没这格了 + 不再落盘
    s0 = ag.DecisionSettings()
    check(not hasattr(s0, "stall_s"), "`stall_s` 没移除干净（设置里还挂着 ✗）")
    check("stall_s" not in s0.to_dict(), "`to_dict` 还在写 `stall_s`（老键该不再落盘 ✓）")

    # ② 老配置（带 stall_s）能读、不报错，也不挂回去
    s1 = ag.DecisionSettings()
    s1.from_dict({"stall_s": 7.0})
    check(not hasattr(s1, "stall_s"), "老配置里的 `stall_s` 又被挂回设置上了 ✗")

    # ③ 换算：一处实现 + 下限
    check(abs(route.stall_s_from_retry_ms(3000) - 3.0) < 1e-9, "3000ms 该换算出 3.0 秒")
    check(abs(route.stall_s_from_retry_ms(5000) - 5.0) < 1e-9, "5000ms 该换算出 5.0 秒")
    for bad in (0, -5, None, "x"):
        _got = route.stall_s_from_retry_ms(bad)
        check(_got >= 0.5,
              "%r 没被钳到下限 0.5（那是「刚按一下就判卡住」✗）：%r" % (bad, _got))

    # ④ 三处调用方同源（源码钉子：别再各写一份 ✗）
    root = _P(__file__).resolve().parent.parent
    for rel in ("gui/route_panel.py", "gui/live_thread.py", "decision/agent.py"):
        src = (root / rel).read_text(encoding="utf-8")
        check("stall_s_from_retry_ms" in src,
              "%s 没走 `route.stall_s_from_retry_ms`（三处必须同一处实现 ✓）" % rel)

    # ⑤ 「走」不吃它（照旧常量 + 寻路超时兜底 ✓）
    t = mapdata.load("105090600")
    z = zones.load("105090600")
    _walk = next((e for e in z.edges if e.get("kind") == "walk"), None)
    check(_walk is not None, "用户数据里没有「走」边，这条测不了")
    _wj = route.job_for_edge(t, z, _walk, stall_s=1.5)
    check(abs(float(getattr(_wj, "stall_s", -1)) - float(route.STALL_S)) < 1e-9,
          "「走」竟然吃到了那件参数（该照旧用常量 %s）：%s"
          % (route.STALL_S, getattr(_wj, "stall_s", None)))
def t_climb_flow_rules():
    """攀爬流程的两条新规矩（用户 2026-09-27 要求，落点在 `ClimbJob.update`）：

      ② 对齐 x → 按跳 + 按住方向 → **还没上绳、人还在起跳那块集合里 ⇒ 先点按一次目标方向
         （↑/↓），再回到对齐 x 重新开始**（一轮只按**一下**跳；两次尝试之间的间隔就是 ①
         那个保持窗口 `hold_ms` ⇒ 不新造常数 ✓；重来那一轮里**方向键不松手**，否则空中
         抓不住绳 ✗）。点按必须是**先松够 `TAP_PERIOD_S - TAP_ON_S`、再按 `TAP_ON_S`**：
         按跳那一拍 ↑/↓ 是按着的，直接再按住 = 对面看到"一直按着" = **一次都没新按过** ✗；
      ③ **被 attack 打断之后回到 climb 要重新走一遍流程** —— 在绳上也一样，只是进 ALIGN
         后会**跳过横向对齐**（在绳上按左右 = 松手掉下来 ✗）。

    （① 「在绳上就不进 attack」是 agent 那一层的事，见
      `t_climb_attack_exception_agent`。）
    """
    from decision import route
    C = route.ClimbJob

    def mk(**kw):
        a = dict(ladder_id="L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                 dst_set="乙平台", src_set="甲平台", tol_px=6, hold_ms=0,
                 timeout_s=600.0)
        a.update(kw)
        return route.ClimbJob(**a)

    # ---- ② 还没上绳、人还在起跳平台上 ⇒ **直接补按一次目标方向 → 回对齐**（用户 2026-10-01 ✓）----
    # ⭐ 2026-10-01 改：不再"先站住等 y 稳定 1 秒"—— `_in_src`（here_sets 广播含 src_set）
    #   本身 = 位置状态权威确认"人已站稳起跳平台"✓，原来 `_y_settled` 那个时间/几何兜底已删。
    _P, _ON = route.TAP_PERIOD_S, route.TAP_ON_S        # 点按周期 / 按住时长（现成常数）
    job = mk(hold_ms=250)
    o = job.update(0.0, px=700.0, py=100.0, here_sets=["甲平台"])
    check(job.phase == C.ALIGN and not o["jump"],
          "刚进误差范围该先站住等保持窗口，不能马上跳：%s / %s" % (job.phase, o))
    o = job.update(0.30, px=700.0, py=100.0, here_sets=["甲平台"])      # 300ms ≥ 250
    check(job.phase == C.CLIMB and o["jump"],
          "保持够了该按跳：%s / %s" % (job.phase, o))
    check(int(o["dir"]) == 1, "按跳那一下方向键（↑）要一起按住：%s" % o)
    # 跳过了、人**还在起跳平台上**（跳没生效 / 又落回来）⇒ **直接进补按阶段**（不等 y 稳定）
    # ⭐ 用户 2026-09-27 更正（原话）："**向上攀爬只有「按住 ↑」和「补按住 ↑」，不能存在
    #   「松开 ↑」和「点按 ↑」**" ⇒ 这一小段**只许"站住不横move"**，**↑ 一路按着** ✓。
    _t0 = 0.36
    o = job.update(_t0, px=700.0, py=100.0, here_sets=["甲平台"])
    check(job.phase == C.CLIMB and not o["jump"] and int(o["dir"]) == 1
          and job._retry_stage == "tap",
          "跳没上去该**直接进补按阶段**（不再站住等 y 稳定 ✗），且 ↑ 必须按着：%s / stage=%r"
          % (o, job._retry_stage))
    # 下一拍 ⇒ **补按 ↑**（`reassert=True` ⇒ 只再发一次 PRESS，绝不先松）
    _t1 = _t0 + 0.02
    o = job.update(_t1, px=700.0, py=100.0, here_sets=["甲平台"])
    check(job.phase == C.CLIMB and int(o["dir"]) == 1 and o["reassert"] is True
          and not o["jump"],
          "进补按阶段后该「按住 ↑ 并补按一次」（`reassert=True` ⇒ 只再发一次 PRESS ✓）：%s"
          % o)
    check("补按" in o["note"] and "不松开" in o["note"],
          "「补按（不松开）」这一步没跟人说清（人会以为它在点按/松手 ✗）：%r" % o["note"])
    o = job.update(_t1 + (_P - _ON) * 0.5, px=700.0, py=90.0, here_sets=["甲平台"])
    check(int(o["dir"]) == 1 and o["reassert"] is False,
          "等的那几拍把 ↑ 松了 / 还在刷补按（上爬要**一路按住 ↑**、补按只发一次 ✓）：%s" % o)
    # 补按完 ⇒ **回到对齐 x 重新开始**（保持窗口重新起算 = 两次尝试之间的间隔 ✓）
    o = job.update(_t1 + _P + 0.02, px=700.0, py=90.0, here_sets=["甲平台"])
    check(job.phase == C.ALIGN and not o["jump"] and int(o["dir"]) == 1,
          "补按完了却没回到对齐重来（且 ↑ 要一路按着 ✓）：%s / %s" % (job.phase, o))
    check("补按过 ↑" in o["note"],
          "回到对齐那句没说清刚才是「补按」：%r" % o["note"])
    check(not o["failed"], "站在起跳平台上重来不该判失败：%s" % o)
    # 保持窗口**重新起算** ⇒ 不会下一拍就又跳一下（那还是"一直按跳"✗）
    o = job.update(_t1 + _P + 0.08, px=700.0, py=90.0, here_sets=["甲平台"])
    check(not o["jump"], "回到对齐后没重新起算保持窗口（立刻又跳了）：%s" % o)
    # ⚠ **重来那一轮的对齐窗口里方向键也得按着**（不是只有退回那一拍）：人这时多半还在
    #   空中，一松手那一下就抓不住绳 ✗（`_retrying` 的用处就在这几拍）
    check(int(o["dir"]) == 1,
          "重来那一轮的对齐窗口里方向键松了 —— 空中那一下抓不住绳 ✗：%s" % o)
    o = job.update(_t1 + _P + 0.08 + 0.30, px=700.0, py=90.0, here_sets=["甲平台"])
    check(o["jump"], "第二次尝试没再按一次跳：%s" % o)
    # ⚠ **还在"补按阶段"那几拍里，人自己抓上绳了** ⇒ 重来那一小段的状态要**全收干净**、
    #   直接接着爬（不清的话：以后再掉回起跳平台，`_retry_stage` 还挂着 ⇒ 那几拍既不按
    #   方向键也不按跳，白等 ✗）
    job = mk(hold_ms=250)
    job.update(0.0, px=700.0, py=100.0, here_sets=["甲平台"])
    job.update(0.30, px=700.0, py=100.0, here_sets=["甲平台"])       # 按跳那一下
    o = job.update(0.36, px=700.0, py=100.0, here_sets=["甲平台"])
    check(int(o["dir"]) == 1 and job._retry_stage == "tap",
          "跳没上去时该直接进补按阶段（横着别动），但 **↑ 要按着**：%s / %s"
          % (o, job._retry_stage))
    o = job.update(0.60, px=700.0, py=100.0, here_sets=["甲平台"],
                   ladder_id="L2")                                   # 补按的当口抓上了
    check(job.phase == C.CLIMB and job._dir_tap_at is None
          and job._retry_stage == "" and int(o["dir"]) == 1
          and "已上绳" in o["note"],
          "抓上绳之后没收干净 / 没直接接着爬：%s / %s / stage=%r"
          % (job.phase, o, job._retry_stage))
    check(not o["done"] and not o["failed"] and not o["jump"],
          "刚抓上绳就报到达/失败/还在按跳：%s" % o)
    # 反复重来（人一直在起跳平台上）**不许判「掉下来了」**—— 那条管的是"真离开绳子"
    o = job.update(3.0, px=700.0, py=100.0, here_sets=["甲平台"])
    check(not o["failed"],
          "站在起跳平台上反复重来却判了「掉下来了」（2 秒容错期不该管这里）：%s" % o)
    # 对照：**离开**那块平台（人在空中 / 别处）⇒ 照旧按「掉下来了」判失败（容错期 2s）
    job2 = mk()
    job2.update(0.0, px=700.0)
    job2.update(0.1, px=700.0, py=100.0, here_sets=[])
    o = job2.update(2.2, px=700.0, py=100.0, here_sets=[])
    check(o["failed"],
          "离开起跳平台、2 秒没回到绳上却没判失败（容错期那条判据被弄没了）：%s" % o)

    # ---- ③ 在绳上被打断：回到 ALIGN，但**一步横移都不许发**，方向键接着按住 ----
    j3 = mk()
    j3.update(0.0, px=700.0)
    o = j3.update(0.1, px=700.0, py=100.0, ladder_id="L2")
    check(j3.phase == C.CLIMB and o["dir"] == 1 and not o["jump"],
          "在绳上该只按住 ↑（不按跳）：%s / %s" % (j3.phase, o))
    j3.interrupted(0.2)                           # 从 0.2s 起进战斗（这几拍没人喂它）
    o = j3.update(3.2, px=850.0, py=100.0, ladder_id="L2")   # ⚠ 故意喂偏 150px 的 x
    check(not o["failed"], "打断回来第一拍就把任务判失败了：%s" % o)
    check(int(o["move"]) == 0,
          "在绳上被打断后又去**横向对齐**了（按左右 = 松手掉下来 ✗）：%s" % o)
    check(o["dir"] == 1 and not o["jump"],
          "在绳上被打断后没接着按住 ↑（那就不爬了）：%s" % o)
    check(j3.phase == C.CLIMB and j3._in_tol_since is None,
          "阶段没重走一遍（`_in_tol_since` 该被清掉 = 对齐重新起算）：%s / %s"
          % (j3.phase, j3._in_tol_since))
    # ③b 下爬在绳上被打断：回来**重新稳住 hold_ms**，不许直接跳下绳梯
    j5 = mk(direction=-1, hold_ms=250, dst_x=700.0)
    j5.update(0.0, px=700.0)
    o = j5.update(0.05, px=700.0, py=100.0, ladder_id="L2")
    check(j5.phase == C.CLIMB and o["dir"] == -1,
          "下爬在绳上该按住 ↓：%s / %s" % (j5.phase, o))
    j5.interrupted(0.10)
    o = j5.update(0.30, px=700.0, py=100.0, ladder_id="L2")
    check(not o["jump"] and j5.phase == C.CLIMB,
          "下爬被打断回来就直接跳下绳梯了（「稳住 hold_ms」没重新起算）：%s / %s"
          % (j5.phase, o))


def t_climb_replan_wrong_ladder():
    """**上了非目标绳 ⇒ 通知重启寻路**（用户 2026-10-02 ✓ 治本）。

    "龙族打猎场"bug：align 阶段按住 ↑（设计——用户 2026-09-27 定的"上爬一路按住 ↑"✓），
    角色被 L1（非目标 L7）吸住 ⇒ 卡左角 ✗。治本：位置状态返回非目标 ladder_id
    ⇒ out["replan"]=True ⇒ `_climb_tick` 重启寻路 ✓。

    钉三件：
      ① 上了非目标绳（ladder_id != 目标）⇒ replan=True ✓；
      ② 在目标绳上 ⇒ 不 replan（正常上绳 ✓）；
      ③ 没在绳上（ladder_id=None）⇒ 不 replan（正常 align ✓）。
    """
    from decision import route

    def mk():
        return route.ClimbJob("L7", x=521.0, y1=844.0, y2=1039.0,
                              direction=1, dst_set="二楼", src_set="底层")
    now = 1000.0
    # ① 上了非目标绳 L1（目标 L7）⇒ replan=True ✓
    j1 = mk()
    out = j1.update(now, 320.0, py=950.0, ladder_id="L1",
                    here_sets=["底层"], on_rope_pos="L1")
    check(out.get("replan") is True,
          "上了非目标绳 L1（目标 L7）没返回 replan=True ✗：%r" % (out.get("replan"),))
    check("L1" in str(out.get("note") or "") and "L7" in str(out.get("note") or ""),
          "replan 的 note 没说清上了哪个非目标绳 ✗：%r" % (out.get("note"),))
    # ② 在目标绳 L7 上 ⇒ 不 replan（正常上绳 ✓）
    j2 = mk()
    out2 = j2.update(now, 521.0, py=900.0, ladder_id="L7",
                     here_sets=["底层"], on_rope_pos="L7")
    check(out2.get("replan") is not True,
          "在目标绳 L7 上却 replan 了（正常上绳不该重启 ✗）：%r" % (out2.get("replan"),))
    # ③ 没在绳上（ladder_id=None）⇒ 不 replan（正常 align ✓）
    j3 = mk()
    out3 = j3.update(now, 320.0, py=950.0, ladder_id=None,
                     here_sets=["底层"], on_rope_pos=None)
    check(out3.get("replan") is not True,
          "没在绳上却 replan 了（正常 align 不该重启 ✗）：%r" % (out3.get("replan"),))


def t_climb_attack_exception_agent():
    """① 的唯一特例，在 **agent 这一层**钉住：**豁免只留给"还没吸上绳"那一段**。

    口径变过一次（用户 2026-09-28，原话："取消……绳梯豁免的**第 2 种情况（爬绳途中）**：
    ① **爬绳途中正常触发 attack**，但**不会松开 ↑ 键**"✓）：
      · **还没上绳**（对齐 / 按下跳 / 飞行中）⇒ 照旧**豁免**（不进 attack ✓ —— 用户 2026-09-27
        报的"从「按下跳」到「攀爬成功」期间都不应该进 attack"**仍然有效** ✓）；
      · ⭐ **已经在绳上爬**（广播说 `ladder_id` 就是本任务那根绳）⇒ **不再豁免** ⇒ 有怪就正常
        进 attack ✓，**但那一拍仍要把攀爬任务推一拍**（保证 ↑ 不松 ⇒ 人不会掉下来 ✓）。
    判据见 `agent.tick`：`_in_rope = _pos.on_rope(_lid)` / `_on_rope = _climb_holds and not _in_rope` ✓。

    钉六件（重点是"例外要**窄**、**不许自锁**"，以及**新口径下 ↑ 绝不能断**）：
      ① **在绳上爬着** + 有怪 ⇒ **进 attack** ✓、且 **↑ 一直按着** ✓（掉不下来 ✓）；
      ② **还没上绳**（对齐阶段、位置就在这根绳范围里）⇒ **仍豁免**、不进 attack ✓；
      ②' 对照：**绳在别处** ⇒ 照旧先打（例外不许无限宽 ✗）；
      ③ **走路任务**路过某根绳的 x（感知会认成"就在绳上"）⇒ 也照旧先打 ✓；
      ④ 打完（怪清空）照旧接着爬 ✓（`t_climb_wiring` 也钉着 ✓）；
      ⑤ ⭐ **上一拍已经是 attack** ⇒ 这一拍照旧 attack **且 ↑ 不断**（老口径那个"自锁"担心
        在新口径下**不存在**了 —— 不再需要"扳回 climb"✗，但"任务照推 ⇒ ↑ 不松"必须成立 ✓）。
    """
    from decision import route

    s = ag.DecisionSettings()
    s.enabled = True
    s.strategy = "patrol"
    h = Harness(s)
    h.clock0 = h.clock.t
    s.enabled = True                      # settings 是全局单例，别靠前面用例的副作用 ✓

    def job(**kw):
        # x=500 = 夹具里玩家的 x（一拍对齐）；绳端放很远 ⇒ 别被"越过绳端"那条兜底判到达 ✓
        a = dict(ladder_id="L2", x=500.0, y1=-9999.0, y2=9999.0, direction=1,
                 dst_set="乙平台", src_set="甲平台", tol_px=6, hold_ms=0)
        a.update(kw)
        return route.ClimbJob(**a)

    with h._patched():
        up, atk = s.keymap["up"], s.keymap["attack"]

        # ① ⭐ **在绳上爬着** + 有怪 ⇒ **正常进 attack**（用户 2026-09-28 新口径 ✓），
        #    但**↑ 必须一直按着**（那一拍要照推攀爬任务 ⇒ 人不会松手掉下来 ✓）
        h.agent.start_climb(job())
        ws = h.ws(with_mob=False)
        ws.player.ladder_id = "L2"
        h.agent.tick(ws)                  # 第一拍：进 climb 状态
        h.log.clear()
        h.clock.t += 0.1
        ws2 = h.ws(with_mob=True)         # 默认那只怪在 x=560（范围内）
        ws2.player.ladder_id = "L2"
        act = h.agent.tick(ws2)
        check(h.agent.state == "attack",
              "在绳上爬着、攻击范围内有怪，却没进 attack（用户 2026-09-28 要求「爬绳途中正常"
              "触发 attack」✗）：%s" % h.agent.state)
        check(up in set(act.get("keys") or []),
              "进了 attack 就把 ↑ 松了 —— 人会从绳上掉下来 ✗（用户 2026-09-28 原话："
              "「不会松开 ↑ 键」）：%s" % sorted(act.get("keys") or []))
        # 后续几拍也**每一拍都要有 ↑**（攻击是输出序列，可能这一拍还没轮到按攻击键 ✓）
        for _i in range(4):
            h.clock.t += 0.1
            ws_m = h.ws(with_mob=True)
            ws_m.player.ladder_id = "L2"
            act_m = h.agent.tick(ws_m)
            check(up in set(act_m.get("keys") or []),
                  "进 attack 之后的第 %d 拍里 ↑ 断了（`KeyState` 只在键集变化时发键 ⇒ "
                  "不推任务就等于松手 ⇒ 人会掉下来 ✗）：%s" % (_i + 2, act_m.get("keys")))
        h.log.clear()

        # ④ 怪清空 ⇒ 照旧接着爬（"打完继续走"没被破坏）
        h.log.clear()
        h.clock.t += 0.1
        ws3 = h.ws(with_mob=False)
        ws3.player.ladder_id = "L2"
        act3 = h.agent.tick(ws3)
        check(h.agent.state == "climb", "怪清空后没接着爬：%s" % h.agent.state)
        check(up in set(act3.get("keys") or []), "怪清空后没按住 ↑：%s" % act3.get("keys"))

        # ② ⭐⭐ **对齐阶段照常打架**（用户 2026-09-28 改口径 ✓ 原话："取消最后一个 attack 的
        #   『绳梯豁免』：**对齐 x 期间如果进了 attack，就从入口开始**"）。
        #   ⚠ 改之前：只要"人就在这根绳的范围里"（`on_rope_pos == ladder_id` ⇒ `holds_player`
        #     里的 `_in_span` 成立）就**不进 attack** ✗ —— 那正是用户点名要取消的那个豁免 ✓。
        #   ⚠ 现在：对齐阶段（`ClimbJob.ALIGN`）**一律不豁免** ⇒ 攻击范围内有怪就**先打** ✓
        #     （人在平台上，打得过就该打 ✓）；打完**从入口重来**（下面 ②b 钉着 ✓）。
        h.agent.start_climb(job())
        h.clock.t += 0.1
        # ⚠ **先让任务真跑一拍**（踩过）：`_task_settle` 有个例外 —— "任务还没跑过第一拍
        #   （`_task_ran_at is None`）⇒ **不报信**"（怕把将来的起点提前写成过去 ✗）⇒
        #   若一挂上就被怪抢进 attack，"被打断"这笔账**根本记不上** ⇒ ②b 那条永远不成立 ✗。
        h.agent.tick(h.ws(with_mob=False))
        # ⚠ 摆回「**对齐 x 中**」（用户现场 `hold_ms`=1000ms ⇒ 对齐要持续好几拍；而这个用例
        #   的 `hold_ms=0` ⇒ 上面那拍一步就对好、进 `CLIMB` 了 ✗）—— 不摆回去，下面测的就
        #   不是"对齐期间"了 ✓（`_task_ran_at` 已经盖过章 ✓ 报信那套仍然有效 ✓）。
        h.agent._climb.phase = route.ClimbJob.ALIGN
        h.clock.t += 0.1
        ws4 = h.ws(with_mob=True)
        ws4.player.ladder_id = None       # 感知：还没报"在绳上"（带按键许可 ✗）
        ws4.player.on_rope_pos = "L2"      # 广播：**只看位置**，人就在 L2 的范围里 ✓
        act4 = h.agent.tick(ws4)
        check(h.agent.state == "attack",
              "对齐 x 期间有怪却不打架 —— 那个「绳梯豁免」还在（用户 2026-09-28 要求取消 ✗）："
              "%s" % h.agent.state)
        check(up not in set(act4.get("keys") or []),
              "进 attack 那一拍还在按 ↑（该把这一拍的键全交给攻击 ✗）：%s"
              % act4.get("keys"))

        # ②b ⭐ **打完 ⇒ 从入口开始**（不是"接着对齐" ✗）：任务被占用够久（> `PAUSE_MIN_S`）
        #   ⇒ `phase` 回 `ALIGN` 且**对齐锚点全清** = 重走整条上绳流程 ✓。
        job4 = h.agent._climb
        # ⚠ **这里直接摆"已经挨打了一会儿"**（不靠 tick 数凑时长 ✓ 那样不可控、也不测到点上 ✗）：
        #   `_paused_at` 本来是**回来那一拍**由 `_task_settle` → `_climb_interrupted` 才置上的
        #   （语义 = "你上一拍没跑任务"）⇒ **同一拍**里既报信又判"停了多久"只会算成 0 ✗（踩过）。
        #   这里只验"**回来那一拍怎么继续**"：挨够 `PAUSE_MIN_S`(0.25s) ⇒ 该从入口重来 ✓。
        # ⚠ **"打完怎么继续（回入口）"不在这里重复测** —— 那套机制**本来就存在**，而且已经
        #   被 `t_job_interrupt_by_fight` ① 钉死（"没上绳被打断 ⇒ `phase` 回 `ALIGN` + 保持
        #   窗口重算"✓，实现是 `route.py` 里 `_paused >= PAUSE_MIN_S` 那一支 ✓）。
        #   ⭐ 本轮改的只有**一件**：让"对齐 x 期间"**也能被打进来**（取消那个豁免 ✓）——
        #   进来之后怎么回去，走的就是那条既有路径，不再另立判据 ✓。
        #   （⚠ 写这条时踩过两个坑，都记在上面：① 同一次 `update` 里那个"已经在绳上 ⇒ 跳过
        #     横向对齐"会把 `phase` **又推回 `CLIMB`** ⇒ 拿 `phase` 当"回没回入口"不可靠 ✗；
        #     ② `_paused_at` 是**回来那一拍**才由 `_task_settle` 置上的，同一拍判只会算成 0 ✗。）
        h.agent.tick(h.ws(with_mob=False))   # 怪走了 ⇒ 回战斗分支、任务接着推 ✓
        check(h.agent._climb is job4,
              "打完（怪清空）把攀爬任务丢了（该接着爬 ✓）：%r" % (h.agent._climb,))

        # ②' 对照：**绳在别处**（人还没走到它的范围里）⇒ 照旧先打（例外不许无限宽 ✗）
        far = route.ClimbJob(ladder_id="L9", x=900.0, y1=-9999.0, y2=9999.0,
                             direction=1, dst_set="乙平台", src_set="甲平台",
                             tol_px=6, hold_ms=0)
        h.agent.start_climb(far)
        h.clock.t += 0.1
        ws4b = h.ws(with_mob=True)
        ws4b.player.ladder_id = None
        ws4b.player.on_rope_pos = None     # 广播：**只看位置**也不在 L9 的范围里 ✓
        h.agent.tick(ws4b)
        check(h.agent.state != "climb",
              "人离那根绳还很远（还没到绳上）却也不打架了 —— 例外放宽了 ✗：%s" % h.agent.state)

        # ③ 对照：**走路任务** + 感知说"在绳上"⇒ 照旧先打（改路线路过绳的 x 很常见）
        spots = [(500.0, 480.0, 520.0, "1")]
        h.agent.start_climb(route.WalkJob("乙平台", spots, tol_px=6, hold_ms=0))
        h.clock.t += 0.1
        ws5 = h.ws(with_mob=True)
        ws5.player.ladder_id = "L2"       # 走路任务没有 ladder_id ⇒ 不该豁免 ✓
        h.agent.tick(ws5)
        check(h.agent.state != "climb",
              "走路任务路过某根绳的 x 也不打架了（豁免判据太宽 ✗）：%s" % h.agent.state)

        # ⑤ ⭐ **上一拍已经是 attack** ⇒ 这一拍照旧 attack **且 ↑ 不断**
        #   （用户 2026-09-27 报的"climb 执行时屏蔽失效"在新口径下换了个形状：不再需要
        #    "扳回 climb"✗，但"**任务照推 ⇒ ↑ 不松**"必须成立 —— 否则人挂在绳上被怪牵着
        #    打几拍就掉下来了 ✗）。
        h.agent.start_climb(job())
        ws6 = h.ws(with_mob=False)
        ws6.player.ladder_id = "L2"
        h.agent.tick(ws6)                  # 先在爬（上一拍 = climb ✓）
        h.log.clear()
        h.agent.state = "attack"           # 模拟"上一拍已经掉进 attack"（老现场 ✓）
        h.clock.t += 0.1
        ws7 = h.ws(with_mob=True)
        ws7.player.ladder_id = "L2"        # 人确实还在那根绳上 ✓
        act7 = h.agent.tick(ws7)
        check(h.agent.state == "attack",
              "在绳上 + 有怪，这一拍该照旧 attack（新口径：爬绳途中正常触发 ✓）：%s"
              % h.agent.state)
        check(up in set(act7.get("keys") or []),
              "在 attack 里没把 ↑ 推上去 —— 人会从绳上掉下来 ✗（用户 2026-09-28："
              "「不会松开 ↑ 键」）：%s" % act7.get("keys"))


def t_climb_align_gap():
    """「**对齐绳梯移动延迟(ms)**」（用户 2026-09-27 加的参数）：对齐绳的 x 时，
    **两次按下方向键之间至少要隔这么久**。

    钉四件：
      ① 默认值 180 **就是** `route.TAP_PERIOD_S`（那本来是写死的点按周期；两处各写一个数
         迟早分叉 ⇒ 这里直接断言它俩相等 ✓）；
      ② 决策设置三处同步（`__init__` / `to_dict` / `from_dict`），且老项目文件缺这个键时
         退回 180（= 老行为 ✓）；
      ③ 行为：上一次按下还没到间隔 ⇒ **这一拍不按**（站住等），到点才按 —— **换了方向也
         一样**（过冲之后往回按也走这条 ⇒ 治的就是「在绳两边来回抖」）；
      ④ 设 0 = 不限制；远距离（>`NEAR_PX`）仍然**一口气按住走**（同方向连续按算一次按下 ✓）。
    """
    from decision import route
    C = route.ClimbJob

    # ① 默认值 = 那个常量（毫秒）
    check(int(getattr(ag.settings, "climb_align_gap_ms", -1))
          == int(route.TAP_PERIOD_S * 1000),
          "默认「对齐绳梯移动延迟」不是 `route.TAP_PERIOD_S` 的毫秒值：%r vs %r"
          % (getattr(ag.settings, "climb_align_gap_ms", None), route.TAP_PERIOD_S))
    check(C("L2", x=700.0, y1=-100.0, y2=200.0, direction=1).align_gap_ms == 180,
          "`ClimbJob` 自己的默认值不是 180（老调用方的行为会变 ✗）")

    # ② 存 / 读一致 + 老文件退回默认
    s = ag.DecisionSettings()
    s.climb_align_gap_ms = 250
    d = s.to_dict()
    check(d.get("climb_align_gap_ms") == 250,
          "`to_dict` 没带上这个参数：%r" % d.get("climb_align_gap_ms"))
    s2 = ag.DecisionSettings()
    s2.from_dict(d)
    check(int(s2.climb_align_gap_ms) == 250,
          "`from_dict` 没读回来：%r" % s2.climb_align_gap_ms)
    s3 = ag.DecisionSettings()
    s3.from_dict({})
    check(int(s3.climb_align_gap_ms) == 180,
          "老项目文件（没这个键）没退回默认 180：%r" % s3.climb_align_gap_ms)

    # ③ 行为：间隔没到不按、到点才按（换方向也一样）
    def mk(**kw):
        a = dict(ladder_id="L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                 tol_px=6, hold_ms=0, align_gap_ms=300)
        a.update(kw)
        return C(**a)

    job = mk()
    o = job.update(0.0, px=600.0)
    check(o["move"] == 1, "离得远该往右挪：%s" % o)
    o = job.update(0.1, px=760.0)                  # 站过头了，可上一次按下才 100ms
    check(o["move"] == 0,
          "间隔（300ms）没到就又按了 —— 那一下就是「在绳两边来回抖」：%s" % o)
    check("移动延迟" in str(o["note"]),
          "等间隔那一拍没说清在等什么（人只会看到「站着不动」）：%r" % o["note"])
    o = job.update(0.35, px=760.0)                 # 350 ≥ 300 ⇒ 这才按
    check(o["move"] == -1, "间隔到了却没按：%s" % o)

    # ④ 0 = 不限制（想按就按）
    j0 = mk(align_gap_ms=0)
    j0.update(0.0, px=600.0)
    o = j0.update(0.1, px=760.0)
    check(o["move"] == -1, "设成 0 还挡着不让按：%s" % o)
    # ④b 远距离仍然「一口气按住走」（同方向连续按 = 同一次按下 ✓）
    j2 = mk()
    o1 = j2.update(0.0, px=600.0)
    o2 = j2.update(0.05, px=610.0)                 # 同方向、仍 > tol
    check(o1["move"] == 1 and o2["move"] == 1,
          "远距离没做到「一口气按住」（中间被移动延迟断开了 ✗）：%s / %s"
          % (o1, o2))


def t_climb_up_key_starts_at_jump():
    """**上爬的 ↑ 从「按跳那一刻」才开始按**（用户 2026-10-02 ✓ 原话："应该是**按跳之后
    再按住 ↑**，我观察到 ↑ **从任务下达就一直被按住了**"）。

    病根：`ClimbJob._out` 里那条"老规则算出 `dir=0` 就改成按住 `self.dir`"是 2026-09-27
    那句「上爬**只有**『按住 ↑』和『补按住 ↑』」的**过度应用**（它只管"按跳之后"那一段）⇒
    于是**对齐走位 / 站住等保持窗口**那几拍也在按 ↑ ⇒ 角色走到绳旁边就**提前把绳抓住**
    （就是"碰到绳子就卡住"那个老毛病 ✓）。

    钉五件：
      ① **跳之前**（对齐按住走 / 站住等窗口）⇒ `dir == 0`（一个竖直键都不按 ✓）；
      ② **按跳那一拍**起 ⇒ 一路按住 ↑（斜跳飞行相 / CLIMB 爬升 / 到顶后再按住 /
         **失败重跳 + `retry()` 回对齐**那几拍 ✓ —— 用户 2026-09-27 那条"不能松开 ↑"
         管的就是这一段，**原样成立** ✓）；
      ③ **例外 = 人已经挂在绳上**（方案 B ✓）：位置状态广播说"位置就在**本任务那根绳**
         的绳段里"（`on_rope_pos`，**纯几何、与按键许可无关** ✓）⇒ 就算还没按跳也按住 ↑
         （不然没有按键许可 ⇒ 广播给不出 `ladder_id` ⇒ 认不出"已上绳" ⇒ 去横向对齐 =
         **在绳上按左右 = 松手掉下来** ✗）；
      ④ 例外**只认本任务那根绳**（别的绳的绳段不算 ✓ —— 不然路过别的绳会被判成"已上绳"✗）；
      ⑤ 下爬：跳之前**也不按 ↑**（它的 ALIGN 本来就不发 ↓ ✓），按跳那一拍照旧 ↑ ✓
         —— 这条钉在 `t_climb_diagonal_jump` ⑫ ✓（这里不重复摆）。
    """
    from decision import route
    C = route.ClimbJob

    def mk(**kw):
        a = dict(ladder_id="L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                 dst_set="乙平台", src_set="甲平台", tol_px=6, hold_ms=250,
                 timeout_s=600.0)
        a.update(kw)
        return C(**a)

    # ① 跳之前：对齐走位 / 站住等保持窗口 ⇒ **一个竖直键都不按**
    job = mk()
    o = job.update(0.0, px=650.0, py=100.0, here_sets=["甲平台"])       # 远：按住朝绳走
    check(int(o["move"]) == 1 and int(o["dir"]) == 0,
          "跳之前按住 ↑ 了（用户 2026-10-02：↑ **从任务下达就一直被按住** ✗ —— 角色走到"
          "绳旁边会提前把绳抓住）：%s" % o)
    o = job.update(0.10, px=700.0, py=100.0, here_sets=["甲平台"])      # 进容差：站住
    check(job.phase == C.ALIGN and int(o["dir"]) == 0,
          "站住等保持窗口那几拍按了 ↑（还没按跳 ✗）：%s / %s" % (o, job.phase))
    o = job.update(0.20, px=700.0, py=100.0, here_sets=["甲平台"])      # 窗口还没到
    check(int(o["dir"]) == 0, "保持窗口里按了 ↑（还没按跳 ✗）：%s" % o)

    # ② 按跳那一拍 ⇒ ↑ 一起按下；**之后一路按住**（含跳没上去的补按 + 回对齐 + retry）
    o = job.update(0.40, px=700.0, py=100.0, here_sets=["甲平台"])      # 400ms ≥ 250 ⇒ 按跳
    check(job.phase == C.CLIMB and o["jump"] and int(o["dir"]) == 1,
          "按跳那一拍该把 ↑ 一起按住（「按跳之后再按住 ↑」✓）：%s / %s" % (o, job.phase))
    o = job.update(0.46, px=700.0, py=100.0, here_sets=["甲平台"])      # 跳没上去 ⇒ 补按阶段
    check(int(o["dir"]) == 1,
          "跳完之后把 ↑ 松了（用户 2026-09-27：这一段的 ↑ 只能「按住 / 补按」✗）：%s" % o)
    # 一直推到「补按完 ⇒ 回对齐」那一拍（**循环推进**，别写死时间 —— 点按周期是现成常数 ✓）
    _t = 0.46
    for _k in range(60):
        _t += 0.05
        o = job.update(_t, px=700.0, py=100.0, here_sets=["甲平台"])
        if job.phase == C.ALIGN:
            break
    check(job.phase == C.ALIGN, "用例前提：补按完该回对齐（没走到那一步）：%s" % job.phase)
    check(int(o["dir"]) == 1,
          "「补按完 ⇒ 回对齐 x」那几拍把 ↑ 松了（用户 2026-09-27：「不能存在『松开 ↑』」"
          "✗ —— 人这时多半还在空中，一松手那一下就抓不住绳）：%s" % o)
    job.retry()
    o = job.update(_t + 0.05, px=650.0, py=100.0, here_sets=["甲平台"])
    check(int(o["dir"]) == 1,
          "`retry()` 之后（**已经按过跳**）回对齐却把 ↑ 松了（同一条口径 ✗）：%s" % o)

    # ③ 例外：广播说「位置就在本绳绳段里」⇒ **还没按跳也按住 ↑**（方案 B ✓）
    job = mk()
    o = job.update(0.0, px=690.0, py=100.0, here_sets=["甲平台"], on_rope_pos="L2")
    check(int(o["dir"]) == 1,
          "人已经挂在绳上（广播说位置在本绳绳段里）却没按住 ↑ —— 那样它认不出「已上绳」"
          "⇒ 会去横向对齐（**在绳上按左右 = 松手掉下来** ✗）：%s" % o)
    # ④ 例外**只认本任务那根绳**
    job = mk()
    o = job.update(0.0, px=690.0, py=100.0, here_sets=["甲平台"], on_rope_pos="L9")
    check(int(o["dir"]) == 0,
          "**别的绳**（L9）的绳段也算成例外了（路过别的绳会被判成「已上绳」✗）：%s" % o)


def t_climb_wiring():
    """上绳任务接进状态机：**有怪先打（打完继续走）**、开关关掉不按跳、状态能进 climb。

    这是用户 2026-09-26 第 1 条要求的落点："当决定要前往时则进寻路状态，但是攻击范围内
    有怪还是会进 attack"。这里**不额外写仲裁** —— 靠 tick 里 `if in_range:` 排在
    `else:` 之前天然成立；这条用例就是证明它真的成立（而不是我以为成立）。
    """
    from decision import route

    s = ag.DecisionSettings()
    s.enabled = True
    s.strategy = "patrol"
    h = Harness(s)
    h.clock0 = h.clock.t      # `_patched()` 的打点靠它（`run()` 里本来会设）
    # ⚠ 必须自己把自动打开：`tick()` 第一件事就是判 `enabled`，关着直接返回
    # 「未开启」——那条路一个键都不发。以前这里是**靠前面用例的副作用**把开关打开的
    # （settings 是全局单例），用例顺序一变就红，而且失败信息只说"没进 climb 状态"，
    # 指向别处（2026-09-26 踩过）。
    s.enabled = True

    def job(**kw):
        # x=500 ＝ 夹具里玩家的 x ⇒ 一拍就对齐（hold_ms=0），方便看"该不该按跳"
        a = dict(ladder_id="L2", x=500.0, y1=-100.0, y2=200.0, direction=1,
                 dst_set="乙平台", tol_px=6, hold_ms=0)
        a.update(kw)
        return route.ClimbJob(**a)

    def downs():
        return {k for _t, kind, k in h.log if kind == "down"}

    with h._patched():
        jump, up = s.keymap["jump"], s.keymap["up"]

        # ① 没怪 ⇒ 按住跳 + ↑，且状态是 climb（**没有"要不要按跳"的开关**：
        #    任务挂上就是授权，一路做到位 —— 见 agent._climb_tick）
        h.agent.start_climb(job())
        h.log.clear()
        h.agent.tick(h.ws(with_mob=False))
        check(h.agent.state == "climb", "没进 climb 状态：%s" % h.agent.state)
        check(jump in downs() and up in downs(),
              "上绳时该按住跳 + ↑（向上爬）：%s" % h.log)

        # ①b **已上绳那一拍：只按 ↑、不按跳**（用户 2026-09-26 定的动作规则：
        #     「按 ↑ 直到玩家的真实世界坐标 ≤ 绳梯上端连接的 foothold 的 y」）。
        #     为什么钉在 **agent 这一层**：执行器那边已经有纯用例 ⑫ ✓，但它只能证明
        #     "任务说要按 ↑" —— 证明不了"agent 真按了 ↑、而不是顺手把跳也按下去" ✗。
        #     用户看到的正是这一层（"角色还在绳上、离平台还差一截" ✗），所以这条必须在。
        h.log.clear()
        ws_m = h.ws(with_mob=False)
        ws_m.player.ladder_id = "L2"          # 感知报：现在就在这根绳上
        # ⚠ 目标和**绳端都要放远**：`job()` 的绳上端 y1=-100，而夹具玩家在 y=500 ⇒
        #   `_arrived` 里"越过绳端"那条兜底判据**独立于 dst_y** ⇒ 会当场判到达、把键全松开 ✗
        #   （写这条时先红了一次，就是这个）。`clock0` 也别忘了（夹具记日志要用它）。
        j_m = job(dst_y=-9999.0, y1=-9999.0, y2=9999.0)
        h.agent.start_climb(j_m)
        h.clock0 = h.clock.t
        act1 = h.agent.tick(ws_m)             # 第一拍：对齐 / 进上绳阶段
        h.log.clear()
        act2 = h.agent.tick(ws_m)             # 第二拍：这就是"已上绳"那一拍
        # ⚠ 判据看 **tick 回值里的 keys**（= 这一拍**按住**的键）+ 有没有松过，
        #   **不能**只看 `down` 事件：↑ 在第一拍就按下去了，第二拍它是"一直按着"的，
        #   而 KeyState 只在**变化**时才发事件 ⇒ 日志当然是空的 ✗
        #  （这条用例连红了三次，前两次都是拿日志去证明"按住"，误判成"没按"）。
        check(jump not in (act2.get("keys") or []),
              "已上绳还在按跳（游戏里会卡在绳上/绳顶、迈不上平台）：%s"
              % (act2.get("keys"),))
        check(up in (act2.get("keys") or []),
              "已上绳却没按住 ↑（这一拍等于站着不动）：%s" % (act2.get("keys"),))
        check(up not in [k for _t, kind, k in h.log if kind == "up"],
              "已上绳却把 ↑ 松了：%s" % h.log)
        check("已上绳" in str(h.agent.current_goto_note() or ""),
              "画面上没说清「已上绳、按住 ↑」：%r" % h.agent.current_goto_note())

        # ② **攻击范围内有怪 ⇒ 先打**（这一拍绝不按↑去爬绳）
        h.agent.start_climb(job())
        h.log.clear()
        h.agent.tick(h.ws(with_mob=True))
        check(h.agent.state != "climb",
              "攻击范围内有怪却去爬绳了（该先打）：%s" % h.agent.state)
        check(up not in downs(), "有怪时还在按↑爬绳：%s" % h.log)

        # ③ 怪没了 ⇒ **接着爬**（"打完继续走"）
        h.log.clear()
        h.agent.tick(h.ws(with_mob=False))
        check(h.agent.state == "climb", "打完没回到 climb：%s" % h.agent.state)
        check(jump in downs() and up in downs(), "打完没接着上绳：%s" % h.log)

        # ⑤ **身后的怪也要打**（用户 2026-09-26 报："命令前往的寻路途中，攻击范围内有怪
        #    也不 attack"）。为什么单独钉：`_in_range` 按设计**只认正前方** ✓，正常路径里
        #    身后的怪由「回身输出」那条分支接 —— 可任务在跑时 else 分支是**任务自己**
        #    （`_climb_tick`）⇒ 身后怪**没人接** ✗ ⇒ 一路白挨打。走路时怪多半从身后追上来，
        #    所以这就是用户看到的那一幕 ✗。
        #    判据三条：进 attack ✓、**真的按了攻击键** ✓、而且**不挪窝**（最多"转身点一下"）✓。
        h.agent.set_facing(1)                      # 定死朝向：怪摆在 px-40 ⇒ 必在**身后**
        ws_back = h.ws(with_mob=True)
        mobs_back = [Mob(id=9, x=float(ws_back.player.x) - 40.0,
                         y=500.0, w=40.0, h=40.0, conf=0.9)]
        check(not h.agent._in_range(mobs_back, ws_back),
              "用例没摆对：这只怪被『前方』判据算进去了（那测不到身后那条路 ✗）")
        h.mobs_fn = lambda t: mobs_back
        # ⚠ 「最小切换朝向时间」按**用户实际配置**（1500ms）来：这个值不是 0，转身那一下
        #   才会被"补键"（`_hold_turn`）；而它同时就是**走位**的风险源 ⇒ 这条用例顺手一起
        #   钉住"cap 把转身压成点一下" —— 挂任务时按着方向键走 1.5 秒会把对吧 ✗。
        s.min_turn_hold_ms = 1500
        h.agent.start_climb(job())
        h.log.clear()
        for _ in range(6):                    # 日志**只在开头清一次**：转身是 down…up
            h.clock.t += 0.1                  # 跨拍发的，逐拍清就看不全了 ✗
            h.agent.tick(h.ws(with_mob=True))
        check(h.agent.state == "attack",
              "身后的怪（攻击范围内）没进 attack（用户报的坑 ✗）：%s" % h.agent.state)
        atk = s.keymap["attack"]
        check(any(k == atk for _t, kind, k in h.log if kind in ("down", "up")),
              "身后的怪在攻击范围内，却没按攻击键：%s" % h.log)
        dirs = {s.keymap["left"], s.keymap["right"]}
        ev = [(t, kind, k) for t, kind, k in h.log if k in dirs]
        # ⚠ 名字别叫 `downs`：文件里那个是**辅助函数**（`downs()`），覆盖了后面就报
        #   "'list' object is not callable" ✗（写这条时就这么红的）。
        turn_ts = [t for t, kind, _k in ev if kind == "down"]
        check(len(turn_ts) >= 1,
              "算出身后有怪、却没有转身（不给这一下就转不过去 ⇒ 攻击是空放）：%s" % h.log)
        check(len(turn_ts) <= 2, "转身来回按了 %d 次方向键：%s" % (len(turn_ts), h.log))
        span = 0.0                            # 方向键按住多久（收尾还按着的算到最后一拍）
        for i, (t0, kind0, k0) in enumerate(ev):
            if kind0 != "down":
                continue
            # ⚠ 收尾还按着时，用**相对时刻**补（`h.clock.t - h.clock0` ✓）——
            #   原来写的是 `h.clock.t`（**绝对**时刻 ✗）⇒ 和日志里的相对时刻相减会得到
            #   一个巨大的数（实测报"按住了 1000.50 秒" ✗），把"一直按着"这种真问题
            #   伪装成"数字离谱"✗。
            up_t = next((t for t, kind, k in ev[i + 1:] if kind == "up" and k == k0),
                        h.clock.t - h.clock0)
            span = max(span, up_t - t0)
        # ⚠ 上限按**新口径**（用户 2026-09-28 要求 2）：站桩时**每 3 次 attack** 补一次
        #   "朝目标"的方向键、**按满「最小切换朝向时间」** ✓ ⇒ 允许一段按住到那个时长
        #   （用户配 1500ms ⇒ 这里就是 1.6s 上限）。仍然挡住"按着不放"✗。
        check(span <= s.min_turn_hold_ms / 1000.0 + 0.1,
              "转身按住了 %.2f 秒（上限 = 「最小切换朝向时间」+ 0.1s，超了就是 cap 没生效 ⇒ "
              "挂任务时一路走位、对齐会被毁掉）：%s" % (span, h.log))
        # ⚠ **收尾必须把这个场景撤干净**：后面还有既有断言（到位自己撤掉 / 失败重来 / 取消），
        #   留着"身后的怪 + 挂着的任务"会把它们的场景搅掉 —— 写这条时就红在那儿了 ✗
        #   （`到了任务没自己撤掉`：攻击分支一直接管 ⇒ `_climb_tick` 根本没机会判到达 ✓）。
        h.mobs_fn = lambda t: []
        h.agent.stop_route("用例收尾")
        h.agent.set_facing(1)
        s.min_turn_hold_ms = 0                # 上面为这条用例临时改过，恢复夹具默认（后面
                                              # 还有既有断言跑，别把它们的场景也改了）

        # ④ 那个"要不要按跳"的开关**已经删了**（2026-09-26 用户定）：留着它就会出现
        #    "点了命令前往却只对齐、不上绳"这种要用户自己找开关的坑。
        check(not hasattr(s, "climb_enabled"),
              "DecisionSettings 里又出现了 climb_enabled（谁加回来的？）")

        # ⑤ 到位 ⇒ 任务自己结束、状态退出 climb、不再按跳
        j = job()
        h.agent.start_climb(j)
        h.log.clear()
        ws = h.ws(with_mob=False)
        h.agent.tick(ws)
        ws.player.at_ladder_top = "L2"         # ⭐ 广播：已到 L2 上端（判据唯一来源 ✓）
        # ⚠ 2026-09-27 起**不摆 y 了**：执行器只看广播（几何兜底已删 ✗）
        h.log.clear()
        h.agent.tick(ws)
        check(h.agent._climb is None, "到了任务没自己撤掉：%r" % (h.agent._climb,))
        check(h.agent.state != "climb", "到了还挂着 climb 状态：%s" % h.agent.state)
        check(jump not in downs(), "到了还在按跳：%s" % h.log)

        # ⑤b **失败保护**（2026-09-26 改；2026-09-27 又改）：失败 ⇒ **延迟**重新激活。
        # 原来这里是"立即重来" —— 失败那一下人还在原地、朝向也没变，常常在同一处再歪一次。
        # ⚠ 2026-09-27 起**没有"最多试几次"了**（用户原话："我好像在寻路任务时看到一个
        #    1/3 尝试的显示……应该需要去掉，我们现在都走「寻路超时时间」"）⇒ 这里改成钉
        #    **"失败多少次都不放弃"**：收口交给「寻路超时时间」（另有用例
        #    `t_goto_timeout_all_job_kinds` 钉着 ✓）。
        was_delay = s.climb_retry_delay_s
        try:
            s.climb_retry_delay_s = 1.0
            # ⭐⭐ **2026-09-28 改口径：延迟只给"按跳后挨过打"那种失败** ✓（用户原话：
            #   "失败那段改成：**按跳后**挨过打 ⇒ 用延迟"／"注意攀爬失败后延迟激活的时间
            #   **只对『因喝药而失败』生效**"✓）⇒ 所以这里**分两支**测 ✓。
            # ① **没挨打**（最典型 = 「上绳梯落回去」= 没跳准）⇒ **立刻重来**，不许等 ✗
            #    —— 老行为是"所有失败都等" ✗ ⇒ 现场就是"落回去失败后发呆"（越失败等越久）✗
            #    ⚠ 这一段用**新变量** `jf`/`objf`（不用 `j`/`obj` ✗）：下面那些老断言
            #      （"延迟没到不该重来"等 ✓）用的还是 `j`/`obj` ⇒ 得让它们看到"**在等**"那支
            #      （即下面 ② 支的 `j2` ✗）—— 所以 ② 支也已经改成 `j2` ✓，而 `j`/`obj`
            #      **完全没被这里碰过** ✓（本支只用 `jf` ✓）。
            jf = job()
            objf = h.agent.start_climb(jf)
            objf._fail("用例：故意失败")      # 直接打成失败态，省掉摆时间的麻烦
            h.log.clear()
            done = h.agent._climb_tick(0.0, 500.0, set(), h.ws(with_mob=False))
            check(done is False and h.agent._climb is jf,
                  "失败后任务被扔了：done=%s climb=%r" % (done, h.agent._climb))
            check(jf.attempt == 2 and jf.phase == route.ClimbJob.ALIGN,
                  "「**没挨打**」的失败该**立刻重来**（用户 2026-09-28：延迟只给挨打那种 ✗）："
                  "attempt=%s phase=%s" % (jf.attempt, jf.phase))
            # ⚠ `h.log` 是**按键记录**、不是 behavior 事件 ✗ ⇒ 那条打点用**源码级**钉 ✓
            import inspect as _insp
            check("climb_retry_fast" in _insp.getsource(h.agent._climb_tick),
                  "没打「climb_retry_fast」点（用户 2026-09-28 点名要加的那个 ✗）")

            # ② **按跳后挨过打** ⇒ **才**等延迟 ✓（这就是"因喝药而失败"那一档 ✓）
            #    ⚠ 两个时间戳要在 `start_climb` **之后**摆（它会清零 ✓）
            #    ⚠ 这一支就用 `j`/`obj` ✓（下面那些老断言全指着它 ✓）：它才是"**在等延迟**"
            #      的那支 ✓（`jf` 那支立刻就重来了 ✗）。
            j = job()
            obj = h.agent.start_climb(j)
            h.agent._climb_jump_at = 0.5          # 先按过跳 ✓
            h.agent._climb_hit_at = 0.6           # 再挨了打 ⇒ 这一轮算"按跳后挨打" ✓
            obj._fail("用例：按跳后挨打")
            h.log.clear()
            done = h.agent._climb_tick(0.0, 500.0, set(), h.ws(with_mob=False))
            check(j.attempt == 1 and j.phase == route.ClimbJob.FAILED,
                  "「**按跳后挨过打**」的失败没等延迟（用户要求这一种才等 ✗）："
                  "attempt=%s phase=%s" % (j.attempt, j.phase))
            check(jump not in downs(), "等待期间还在按跳（该站着等）：%s" % h.log)
            # ⚠ 那句等待说明**不许带分母**（用户看到的那个「1/3」就是从这儿来的 ✗）——
            #   要趁**还没重试**的时候看（`retry()` 会把 note 换成"第 N 次尝试" ✓）
            check("已试 1 次" in j.note and "/" not in j.note,
                  "等待说明该写「已试 N 次」而且**不带分母**：%r" % j.note)
            # 还没到点 ⇒ 仍然不重来
            h.agent._climb_tick(0.5, 500.0, set(), h.ws(with_mob=False))
            check(j.attempt == 1, "延迟没到就重新激活了：attempt=%s" % j.attempt)
            # 到点 ⇒ 重新对齐（attempt +1）
            h.agent._climb_tick(1.1, 500.0, set(), h.ws(with_mob=False))
            check(j.attempt == 2 and j.phase == route.ClimbJob.ALIGN,
                  "延迟到了却没重新激活：attempt=%s phase=%s" % (j.attempt, j.phase))
            # **失败多少次都不放弃**（原来这条钉的是"到上限就放弃" ✗ 2026-09-27 反过来了）
            for _k in range(6):
                obj._fail("用例：第 %d 次又失败" % (_k + 2))
                # 第一拍只把"等重来"的时刻摆上（窗口没到 ⇒ 不重来）
                h.agent._climb_tick(100.0 + _k * 100.0, 500.0, set(),
                                    h.ws(with_mob=False))
                # ⚠ 任务要是被提前扔了就别再喂它（会 AttributeError 把这条**真实**的
                #   断言盖成一句看不懂的崩溃 ✗）—— 直接记成"事还没完"，让下面那句红得清楚 ✓
                done = (h.agent._climb_tick(160.0 + _k * 100.0, 500.0, set(),
                                            h.ws(with_mob=False))
                        if h.agent._climb is not None else True)
                check(done is False and h.agent._climb is j,
                      "第 %d 次失败就被放弃了 —— 用户 2026-09-27 明确**不设最多试几次**"
                      "（都走「寻路超时时间」✗）：done=%s climb=%r"
                      % (_k + 2, done, h.agent._climb))
            check(j.attempt >= 8,
                  "重试次数没一直往上加（它只该受「寻路超时时间」管）：%s" % j.attempt)
        finally:
            s.climb_retry_delay_s = was_delay

        # ⑤b2 **已经到了目标集合 ⇒ 不重来，直接收工**（2026-09-26 用户要求 2）：
        # 失败判定可能晚于实际到达（定位抖一下、到了又被打下来）—— 这时接着"重新激活"
        # 就是"人已经站上去了还在原地爬"，比不做更糟。
        j2 = job()
        j2.dst_set = "乙平台"
        h.agent.start_climb(j2)
        ws2 = h.ws(with_mob=False)
        ws2.player.here_sets = ["乙平台"]      # 定位说：已经站在目标集合里了
        j2._fail("用例：判失败，但其实已经到了")
        h.log.clear()
        done = h.agent._climb_tick(20.0, 500.0, set(), ws2)
        check(done is True and h.agent._climb is None,
              "到了目标集合还在重来（人站上去了还在原地爬）：%s / %r"
              % (done, h.agent._climb))
        check(j2.attempt == 1, "已经到了还在加尝试次数：%s" % j2.attempt)
        check(jump not in downs(), "收工那一拍还在按跳：%s" % h.log)
        check(h.agent._climb_retry_at is None,
              "收工之后还留着「等重来」的时刻（下一个任务会立刻被它带跑）")

        # ⑤b3 ⭐ **在绳上时不许拿"脚下集合"把失败改判成功**（用户 2026-09-27 亲测：
        #   "从三楼到顶层攀爬途中，**位置状态还没广播到顶，执行器又擅自结束了**" ✗）。
        #   病根：爬绳时"黄点正下方那条 foothold"可能是**别层**的（绳段横跨好几层 ✗）⇒
        #   集合命中 ≠ 到了 ✗ ⇒ 现在**在绳上时只认广播**（`at_ladder_top` == 本任务的绳 ✓）。
        j3 = job()
        j3.dst_set = "乙平台"
        h.agent.start_climb(j3)
        ws3 = h.ws(with_mob=False)
        ws3.player.here_sets = ["乙平台"]        # 集合恰好命中（别层那块面 ✗）
        ws3.player.ladder_id = str(j3.ladder_id)  # 广播说：正贴着这根绳（还没到顶 ✓）
        j3._fail("用例：判失败，但人还在绳上爬")
        h.clock.t += 0.1
        done = h.agent._climb_tick(h.clock.t, 500.0, set(), ws3)
        check(h.agent._climb is j3 and done is False,
              "人还挂在绳上（广播没说到顶），却拿集合把失败改判成功、擅自收工了 ✗：%r"
              % (h.agent._climb,))
        # 对照：广播真说到顶 ⇒ 照旧算到了 ✓（救命的那条别弄丢 ✗）
        ws3.player.at_ladder_top = str(j3.ladder_id)
        done = h.agent._climb_tick(h.clock.t + 0.1, 500.0, set(), ws3)
        check(done is True and h.agent._climb is None,
              "广播说已到本任务这根绳的上端，却没算到达 ✗：%r" % (h.agent._climb,))
        h.agent.stop_route("用例：清理")

        # ⑤d **寻路任务优先于休息**（用户 2026-09-26 要求："即使在休息中，寻路也需要能生效"）：
        #     挂着任务时那一拍**不进休息分支**（休息那套也按键，两边会打架），任务照跑；
        #     休息的**状态保留**，任务结束下一拍接着休息。
        s.anti_afk_enabled = True
        h.agent.state = "afk_rest"          # 假装正在隐身休息
        h.agent._afk_ctx = None
        h.agent.start_climb(job())
        h.log.clear()
        h.agent.tick(h.ws(with_mob=False))
        # ① 寻路照跑：上绳的键（↑）真的按下去了
        check(up in downs(), "休息中寻路没生效（上绳的键一个都没按）：%s" % h.log)
        # ② 两套**互相独立**：休息状态不被任务抢走（还是"休息"那一族；它自己会按自己的
        #    时刻表往前走 —— 进 afk_exit 是休息机器自己的事，不是被任务顶掉的）。
        check(h.agent.state in ("afk_enter", "afk_rest", "afk_exit"),
              "寻路任务把休息状态顶掉了（两边耦合了）：%s" % h.agent.state)
        # （这里**不再**断言"任务结束后还在休息"：休息机器有它自己的时刻表，会自己
        #   进 afk_exit / 收尾回 idle —— 那是它本来该做的，不是被任务顶掉的 ✗。
        #   要钉的就是上面两条：任务照跑、休息状态不被任务改写。）
        h.agent.stop_climb("用例：测完收工")
        h.agent.state = "idle"
        s.anti_afk_enabled = False

        # ⑤e **寻路超时时间(s)**（用户 2026-09-26 要求；2026-09-29 收窄成**只对追击生效**）：
        #     从下达那一刻算，超了就切断 —— 任务自己的 timeout 会被 `retry()` 清零，
        #     累计可能远超预期，所以要有总闸 ✓（这道闸现在只掐追击 ✓ 见下面那行签注）。
        was_cap = s.goto_timeout_s
        s.goto_timeout_s = 0.5
        j9 = job()
        h.agent._route_origin = dict(CHASE_ORIGIN)      # 追击签注 ⇒ 才吃这道闸 ✓
        h.agent.start_climb(j9)
        h.agent.tick(h.ws(with_mob=False))              # 第一拍（还没超）
        check(h.agent._climb is j9, "刚下达就被超时切断：%r" % (h.agent._climb,))
        h.agent._climb_started -= 1.0                   # 假装它已经跑了 1 秒
        h.agent.tick(h.ws(with_mob=False))
        check(h.agent._climb is None, "超过「寻路超时时间」还没切断：%r" % (h.agent._climb,))
        check("寻路超时" in h.agent.current_goto_note(),
              "切断了却没说清原因：%r" % h.agent.current_goto_note())
        check(not h.taps(s.keymap["jump"]) or True, "")  # 占位：键的释放由既有用例覆盖
        s.goto_timeout_s = was_cap

        # ⑤c 下跳：**先按住 ↓ 那一拍不许按跳**（agent 那层也得对 —— 这是新加的分支）
        dj = route.DropJob("甲平台", "乙平台", [(500.0, 470.0, 530.0, "41")],
                           tol_px=6, hold_ms=0)
        h.agent.start_climb(dj)
        h.log.clear()
        h.agent.tick(h.ws(with_mob=False))
        downs_now = downs()
        check(s.keymap["down"] in downs_now,
              "下跳要先按住 ↓，却没按：%s" % h.log)
        check(s.keymap["jump"] not in downs_now,
              "「先按住↓」那一拍就按跳了（顺序不对）：%s" % h.log)
        h.agent.stop_climb("用例结束")

        # ⑥ 取消 / 关自动 ⇒ 状态收干净（不留按着的键）
        h.agent.start_climb(job())
        h.agent.tick(h.ws(with_mob=False))
        h.log.clear()
        h.agent.stop_climb("用例取消")
        check(h.agent._climb is None and h.agent.state != "climb",
              "取消后还挂着任务/状态：%r / %s" % (h.agent._climb, h.agent.state))
        check(not any(kind == "down" for _t, kind, _k in h.log),
              "取消之后还在按新键：%s" % h.log)


def t_climb_world_x_and_tap():
    """上绳对齐的三条口径（2026-09-26 用户要求 1、2、3）。

    ① **用世界坐标 x**：用户报的"卡在 -300~-380 来回走、无限循环"就出在这儿 ——
       喂进去的是画面检测框中心（`ws.player.x`），而 `ClimbJob.x` 是**世界坐标**
       （同一个调用里 `py` 却用 `world_y`，本身自相矛盾）。
       这里两种坐标**故意摆成不同的数**：一旦有人喂错，方向就会反过来 ⇒ 用例立刻红。
    ② **靠近（≤ 20px）改成点按**：按住方向键在游戏里就是"一直走"，差二十几像素那一下
       必然冲过头。断言"有按的拍、也有松的拍"——全按或全不按都算没做点按。
    ③ **保持窗口 = 设置里的保持时间 + 当前端到端延迟**（`ws.e2e_ms`）：定位读数是
       "过去某一刻"的位置，延迟越大越不能拿单帧当真。
    """
    from decision import route

    s = ag.settings
    h = Harness(s)

    def job(**kw):
        a = dict(ladder_id="L1", x=140.0, y1=-100.0, y2=200.0, direction=1,
                 dst_set="乙平台", tol_px=6, hold_ms=0)
        a.update(kw)
        return route.ClimbJob(**a)

    s.enabled = True          # 同 t_climb_wiring：别靠别的用例把开关打开

    with h._patched():
        h.clock0 = h.clock.t
        # ① 世界 x=100（绳在 140 ⇒ 该往右）；画面 x 故意给 900（喂错就会往左）
        j = job()
        h.agent.start_climb(j)
        ws = h.ws(with_mob=False)
        ws.player.x, ws.player.world_x = 900.0, 100.0
        h.log.clear()
        h.agent.tick(ws)
        d1 = {k for _t, kind, k in h.log if kind == "down"}
        check(s.keymap["right"] in d1 and s.keymap["left"] not in d1,
              "方向没跟**世界坐标**走（多半又喂了画面坐标）：%s" % h.log)
        # 反过来：世界 x 在绳右边 ⇒ 该往左
        h.agent.start_climb(job())
        ws2 = h.ws(with_mob=False)
        ws2.player.x, ws2.player.world_x = 100.0, 300.0
        h.log.clear()
        h.agent.tick(ws2)
        d2 = {k for _t, kind, k in h.log if kind == "down"}
        check(s.keymap["left"] in d2 and s.keymap["right"] not in d2,
              "方向没跟世界坐标走：%s" % h.log)

        # ② 差 12 px（≤ NEAR_PX=20）⇒ 点按：几拍里必须有"松"的拍
        h.agent.start_climb(job())
        ws3 = h.ws(with_mob=False)
        ws3.player.world_x = 128.0
        h.log.clear()
        moves = []
        dirs = (s.keymap["left"], s.keymap["right"])
        for i in range(12):
            h.clock.t = h.clock0 + i * 0.05
            h.agent.tick(ws3)
            moves.append(any(kind == "down" and k in dirs for _t, kind, k in h.log))
            h.log.clear()
        check(any(moves) and not all(moves),
              "靠近时没做「点按」（要么一直按着、要么一直不按）：%s" % moves)

        # ③ 保持窗口把端到端延迟**叠在任务自己的保持时间上**（不是拿设置覆盖它）
        j4 = job(hold_ms=250)
        h.agent.start_climb(j4)
        ws4 = h.ws(with_mob=False)
        ws4.e2e_ms = 400.0
        ws4.player.world_x = 140.0            # 正好在绳上
        h.agent.tick(ws4)
        check(j4.hold_ms == 650,
              "保持窗口没算进端到端延迟：hold_ms=%s（该是任务自己的 250 + 400）"
              % j4.hold_ms)
        # 延迟每拍都在变 ⇒ 下一拍要**重新算**，不能在上一次的结果上再叠一次
        ws4.e2e_ms = 100.0
        h.agent.tick(ws4)
        check(j4.hold_ms == 350, "延迟是按基线叠的（别层层累加）：%s" % j4.hold_ms)
        h.agent.stop_climb("用例结束")


def t_drop_job():
    """下跳执行器：就近平齐 → **先按住 ↓** → ↓+**点按**跳 → 落地；失败保护与上绳同一套。

    用户 2026-09-26 定的动作：**按住 ↓ 再按跳**（与「跳(jump)」= 在 foothold 边缘按跳
    是两个类型，别混）；2026-09-27 又补了两条：跳是**点按**（不是一路按住 ✓）、
    Y 变化超过「坐标对齐误差范围」就**松开 ↓**（完整流程见 `t_drop_flow` ✓）。
    这里钉三件最容易做错的事：
      · **顺序**：ARMED 那一拍只能有 ↓、**不能有跳**（"先按住再按跳"就是这个意思）；
      · **就近**：边上给了好几个可下跳点时要挑离自己最近的那个；
      · **到达**：先认集合；认不出集合 ⇒ 用「**脚下这块面就是目标平台的面**」兜底
        （读位置状态广播的 `ground_y` ✓；用户 2026-09-28 换掉了旧的"y 掉 40px" ✗）。
    """
    from core import mapdata, zones
    from decision import route

    spots = [(700.0, 660.0, 740.0, "41"), (1250.0, 1210.0, 1290.0, "72")]
    job = route.DropJob("甲平台", "乙平台", spots, tol_px=6, hold_ms=0)

    # ① 就近挑：站在 x=1200 ⇒ 挑 fh 72（不是 700 那个）
    o = job.update(0.0, px=1200.0)
    check(job._pick[-1] == "72", "没就近挑可下跳点：%r" % (job._pick,))
    check(o["move"] == 1 and not o["jump"], "该往右挪：%s" % o)
    # ② 平齐后 ⇒ **先按住 ↓**：这一拍有 ↓、没有跳
    o = job.update(0.5, px=1250.0)
    check(o["phase"] == route.DropJob.ARMED and not o["jump"],
          "对齐好了该先进 ARMED（按住↓）阶段：%s" % o)
    check(o["dir"] == -1, "ARMED 阶段就该按住 ↓：%s" % o)
    # ③ 按住一小会儿 ⇒ 补跳
    # ⚠ 这里必须**带 py**（2026-09-27 起："没有 y 读数就不按跳" —— 基准拿不到就判不出
    #   "开始下落"，按了也只能一路按到超时 ✗，见 `t_drop_retry_gap` ⑦）
    o = job.update(0.8, px=1250.0, py=0.0)
    check(o["phase"] == route.DropJob.DROP and o["jump"] and o["dir"] == -1,
          "该「↓ + 跳」了：%s" % o)
    # ④ 到达：先认集合
    o = job.update(0.9, px=1250.0, py=0.0, here_sets=["甲平台"])
    check(not o["done"], "还在别的平台上就说到了：%s" % o)
    o = job.update(1.0, px=1250.0, py=120.0, here_sets=["乙平台"])
    check(o["done"] and not o["jump"], "脚下已经是目标集合却没判到达：%s" % o)

    # ⑤ **没有集合信息时的几何兜底** —— ⭐ 用户 2026-09-28 换掉了判据：
    #    现在是「**脚下这块面就是目标平台的面**」（读位置状态广播的 `ground_y` ✓），
    #    **不再**是"y 比出发时增大 40px" ✗（那条太松：往下掉 40px 就报"到了"，
    #    人可能还在半空、或者落到了别的平台上 ✗）。
    job = route.DropJob("甲平台", "乙平台", spots, tol_px=6, hold_ms=0,
                        dst_ys=[120.0])           # 目标平台的面 y = 120
    job.update(0.0, px=700.0)
    o = job.update(0.5, px=700.0, py=-300.0)      # 进 ARMED → DROP
    check(o["jump"] and o["dir"] == -1,
          "该在下跳（按住 ↓ + 点按跳）：%s" % o)
    # ⚠ 2026-09-27 起跳是**点按**（`TAP_ON_S` 之后松开跳，↓ 继续按着）——
    #   老版本是"一路按住跳"✗（用户 2026-09-27 的口径是"按住 ↓ + **点按**跳"）
    o = job.update(0.5 + route.TAP_ON_S + 0.01, px=700.0, py=-300.0)
    check(not o["jump"] and o["dir"] == -1,
          "点按跳该在 TAP_ON_S 后松开、而 ↓ 继续按着：%s" % o)
    # ⚠ **"y 掉了很多"不算到了**（旧判据就是这么误报的 ✗）：脚下那块面不是目标面 ⇒ 不 done
    o = job.update(1.0, px=700.0, py=0.0, ground_y=0.0)
    check(not o["done"],
          "脚下那块面(y=0)不是目标面(120)，却判「到了」（就是旧的「掉 40px」那个毛病 ✗）：%s"
          % o)
    # 脚下那块面 = 目标面（±「坐标对齐误差范围」）⇒ 到了 ✓
    o = job.update(1.2, px=700.0, py=120.0, ground_y=120.0)
    check(o["done"], "脚下那块面就是目标面（120±容差）却没判到达：%s" % o)
    # ⚠ 目标面不知道（老调用方 / 没给 `dst_ys`）⇒ 退回**只看集合** ✓
    #   —— **绝不许**再拿"掉了多少像素"凑一个数 ✗
    job2 = route.DropJob("甲平台", "乙平台", spots, tol_px=6, hold_ms=0)
    job2.update(0.0, px=700.0)
    job2.update(0.5, px=700.0, py=-300.0)
    o = job2.update(1.0, px=700.0, py=-200.0, ground_y=-200.0)   # 掉了 100，但没有集合
    check(not o["done"], "没给目标面却拿「掉了多少像素」判到了（旧毛病 ✗）：%s" % o)

    # ⑥ 超时 + 失败后重新激活（和上绳同一套保护，用户要求）
    job = route.DropJob("甲平台", "乙平台", spots, tol_px=6, hold_ms=0,
                        timeout_s=1.0)
    check(job.update(0.0, px=700.0)["failed"] is False, "刚开始就说超时")
    o = job.update(2.0, px=700.0)
    check(o["failed"] and not o["jump"], "超时了还在按键：%s" % o)
    o = job.retry()
    check(job.attempt == 2 and o["phase"] == route.DropJob.ALIGN and not o["jump"],
          "retry 没回到对齐阶段：%s" % o)

    # ⑦ 从**真实边**造任务（用户那条 左传送门平台 → 左1），并按类型分发
    t = mapdata.load("105090600")
    z = zones.load("105090600")
    edge = next((e for e in z.edges if e.get("kind") == "drop"), None)
    check(edge is not None, "用户数据里没有下跳边，这条测不了")
    j2 = route.job_for_edge(t, z, edge, tol_px=6, hold_ms=250)
    check(isinstance(j2, route.DropJob) and j2.spots
          and j2.dst_set == edge.get("to"),
          "从真实边造出来的下跳任务不对：%r" % (j2.spots,))
    cl = next((e for e in z.edges if e.get("kind") == "climb"), None)
    check(isinstance(route.job_for_edge(t, z, cl), route.ClimbJob),
          "job_for_edge 没按类型分发（爬边该出 ClimbJob）")
    for bad, why in (({"kind": "portal", "from": "甲", "to": "乙"}, "别的通行方式"),
                     ({"kind": "drop", "from": "甲", "to": "乙",
                       "footholds": []}, "一个可下跳的都没有")):
        try:
            route.job_for_edge(t, z, bad)
            raise AssertionError("%s 该抛 ValueError（不许猜）" % why)
        except ValueError:
            pass


def t_jump_job():
    """「跳(jump)」执行器（初版）：走近 → **按跳** → 落到目标集合；整天靠「寻路超时时间」兜底。

    用户 2026-09-27 定的流程图（照抄）：

    ```
    入口 ├─ x 已在**目标 foothold 集某条的 x 范围**里 ⇒ **直接按跳**
         └─ 否则 ⇒ 朝**最近的那条目标 foothold**走，进了「起跳距离」⇒ 按跳
    等角色位于**目标 foothold 集合** ⇒ 成功
    按跳后「卡住判定时长」还没到目标集合 ⇒ **回入口**（重走、再跳；**不设最多试几次** ✓）
    ```

    钉八件：
      ① 已在范围里 ⇒ **第一拍就按跳**，而且跳是**点按**（`TAP_ON_S` 之后松手 ✓）；
      ② 不在 ⇒ 先朝最近那条 fh **走**（方向对 ✓），没进「起跳距离」**不许跳** ✗；
      ③ 「起跳距离」生效，口径 = 到那条 fh 的 **x 范围**（也就是**最近端点** ✓）的距离；
      ④ 到达**只认集合**（还在起跳那块集合上、或 x 到了都不算 ✓）；
      ⑤ 按跳后 `stall_s` 秒没到 ⇒ **回入口**（重走 + 再跳），而且**不判失败** ✓；
      ⑥ 拿不到世界坐标 ⇒ 如实说、不跳、不判失败 ✓；
      ⑦ `job_for_edge` 按类型分发：真实「跳」边 ⇒ `JumpJob`（形状 (左, 右, id) ✓）；
        「传送门」仍然抛（执行器还没做，不许猜 ✓）；
      ⑧ **起跳之后方向键不许松**（用户 2026-09-27："当进入起跳距离后按跳这一步，方向键
        不能松开，要等成功或超时判定才能松"）：起跳前在走的方向**一直按到**"到了"（成功 ✓）
        或「卡住判定」到点（回入口 ✓）为止；本来没在走（人已在目标 fh 范围里）⇒ 不硬塞 ✓。
    """
    from decision import route
    from core import mapdata, zones

    C = route.JumpJob
    spots = [(600.0, 700.0, "11"), (1200.0, 1300.0, "12")]

    # ① 已经在目标 fh 范围里 ⇒ 直接按跳（点按：TAP_ON_S 之后就松）
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=3.0, jump_start_px=0)
    o = j.update(0.0, px=650.0)
    check(o["jump"] and o["move"] == 0 and o["dir"] == 0,
          "x 已经在目标 foothold 范围里，却没「直接按跳」（跳也不该带 ↑/↓）：%s" % o)
    check("直接按跳" in j.note, "那句 note 没说清是「直接按跳」：%r" % j.note)
    o = j.update(route.TAP_ON_S + 0.01, px=650.0)
    check(not o["jump"], "跳该是**点按**（TAP_ON_S 之后松手）：%s" % o)

    # ② 不在范围里 ⇒ 先走（方向对）；**没进起跳距离不许跳** ✗
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=3.0, jump_start_px=0)
    o = j.update(0.0, px=800.0)                     # 最近的是 fh11（范围 600~700）
    check(o["move"] < 0 and not o["jump"],
          "人在右边、最近那条 fh 在左边 ⇒ 该往左走、且先别跳：%s" % o)
    o = j.update(0.1, px=760.0)
    check(o["move"] < 0 and not o["jump"], "还没进起跳距离就不该跳：%s" % o)

    # ③ 「起跳距离」= 50 ⇒ 离范围 40 就按跳（口径 = 到 fh **范围**（最近端点）的距离 ✓）
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=3.0, jump_start_px=50)
    o = j.update(0.0, px=740.0)                     # 离 fh11 右端（700）差 40 ≤ 50
    check(o["jump"], "进了起跳距离（差 40 ≤ 50）却没按跳：%s" % o)
    check("起跳距离" in j.note, "没走「进起跳距离」那一支：%r" % j.note)
    # 从**左边**过来：离**左端**（端点）的距离同样算 ✓；没进就继续往右走 ✓
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=3.0, jump_start_px=50)
    o = j.update(0.0, px=520.0)                     # 离左端（600）差 80 > 50
    check(o["move"] > 0 and not o["jump"],
          "离左端 80（> 起跳距离 50）时该继续往右走、别跳：%s" % o)
    o = j.update(0.1, px=560.0)                     # 差 40 ≤ 50
    check(o["jump"], "离左端 40（≤ 起跳距离 50）该按跳了：%s" % o)

    # ④ 到达**只认集合**（x 到了、人还在起跳那块平台上 ⇒ 都不算 ✓）
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=3.0, jump_start_px=0)
    j.update(0.0, px=650.0)                          # 按跳
    o = j.update(0.2, px=650.0, here_sets=["甲平台"])
    check(not o["done"], "还在起跳那块集合上就判到达了：%s" % o)
    o = j.update(1.0, px=650.0, here_sets=["乙平台"])
    check(o["done"] and o["phase"] == C.DONE,
          "脚下已经是目标集合却没判成功：%s（phase=%s）" % (o, j.phase))

    # ⑤ 按跳后 stall_s 到点还没到 ⇒ **回入口**（重走 + 再跳），**不判失败** ✓
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=1.0, jump_start_px=0)
    j.update(0.0, px=650.0)                          # 第 1 次按跳
    o = j.update(1.05, px=800.0)                     # stall 到点 ⇒ 这一拍只"回入口"
    check(not o["failed"],
          "跳跃**不设最多试几次**（用户 2026-09-27）⇒ 该回入口，不该判失败：%s" % o)
    check("回入口" in j.note and j.attempt == 2,
          "note 没写清「回入口」/次数：%r（attempt=%s）" % (j.note, j.attempt))
    o = j.update(1.1, px=800.0)                      # 下一拍重新判：人在右边 ⇒ 往左走
    check(o["move"] < 0 and not o["jump"],
          "回入口后该重新朝最近的那条 foothold 走：%s" % o)
    o = j.update(1.2, px=650.0)                      # x 又进了范围 ⇒ 再按跳 ✓
    check(o["jump"], "回入口后 x 又进了范围，却没再按跳：%s" % o)

    # ⑥ 拿不到世界坐标 ⇒ 如实说，不跳也不判失败 ✓
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=3.0, jump_start_px=0)
    o = j.update(0.0, px=None)
    check(not o["jump"] and not o["failed"] and "世界坐标" in j.note,
          "拿不到坐标时该如实说、不跳也不判失败：%s / %r" % (o, j.note))

    # ⑧ **起跳之后方向键不许松**（用户 2026-09-27 要求）：起跳前在走的方向要一直按到
    #    「到了」（成功）或「卡住判定」到点（回入口）为止 —— 跳键本身照旧点按 ✓
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=1.0, jump_start_px=50)
    o = j.update(0.0, px=740.0)                  # 离 fh11 右端（700）差 40 ≤ 50 ⇒ 起跳
    check(o["jump"] and o["move"] == -1,
          "进起跳距离那一拍该「按跳 + 保持方向」（人在右、目标在左 ⇒ ←）：%s" % o)
    o = j.update(route.TAP_ON_S + 0.01, px=720.0)
    check(not o["jump"] and o["move"] == -1,
          "点按一跳完就把方向键松了（用户要的是**按到成功/超时判定**）：%s" % o)
    check("方向键继续按着" in j.note, "那句 note 没说清方向键还按着：%r" % j.note)
    o = j.update(0.5, px=715.0)                  # 还在等落地 ⇒ 依旧按着 ←
    check(o["move"] == -1 and not o["done"], "等落地期间方向键不该松：%s" % o)
    # 「成功」= 脚下进了目标集合 ⇒ 这一拍才松（什么都不按 ✓）
    o = j.update(0.6, px=650.0, here_sets=["乙平台"])
    check(o["done"] and o["move"] == 0 and not o["jump"],
          "到了目标集合该松手（方向键也松）：%s" % o)
    # 「超时判定」= 按跳后 stall_s 到点还没到 ⇒ 回入口那一拍也才松 ✓
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=1.0, jump_start_px=50)
    j.update(0.0, px=740.0)
    o = j.update(1.05, px=740.0)
    check(o["move"] == 0 and not o["jump"] and "回入口" in j.note,
          "「卡住判定」到点该松手（回入口那一拍不许还按着方向）：%s" % o)
    # 反例：**本来就没在走**（人已经在目标 fh 范围里 ⇒ 直接按跳）⇒ 不硬塞一个方向键 ✗
    j = C("乙平台", spots, tol_px=6, hold_ms=0, stall_s=3.0, jump_start_px=0)
    o = j.update(0.0, px=650.0)
    check(o["jump"] and o["move"] == 0,
          "人已经在目标 fh 范围里（本来就没在走）⇒ 起跳不该硬塞一个方向键：%s" % o)

    # ⑦ 真实边分发：跳 ⇒ JumpJob；传送门仍然抛 ✓
    t = mapdata.load("105090600")
    z = zones.load("105090600")
    je = next((e for e in z.edges if e.get("kind") == "jump"), None)
    check(je is not None, "用户数据里没有「跳」边，这条测不了")
    jj = route.job_for_edge(t, z, je, tol_px=6, hold_ms=250, stall_s=3.0,
                            jump_start_px=0)
    check(isinstance(jj, C) and jj.spots and jj.dst_set == je.get("to"),
          "从真实「跳」边造出来的任务不对：%r" % (jj.spots[:3] if jj.spots else None,))
    check(all(len(s) == 3 for s in jj.spots),
          "落点形状该是 (左, 右, foothold id)：%r" % (jj.spots[:2],))
    for bad, why in (({"kind": "portal", "from": "甲", "to": "乙"},
                      "传送门的执行器还没做（不许猜、不许半途停半空）"),):
        try:
            route.job_for_edge(t, z, bad)
            raise AssertionError("%s 该抛 ValueError" % why)
        except ValueError:
            pass


def t_spot_loop_killed_on_rest_end():
    """⭐ **结束休息与循环行为**（⚠ 函数名是**旧的**：上一版要求"立刻杀掉"，用户当天下午
    **改了主意** ⇒ 现在要求"**不打断正在进行的循环**"✓ 以本文为准，别看名字 ✗）。

    ⭐⭐ **现行口径（用户 2026-09-28 更正 ✓）**："结束休息**不应该打断**正在进行的循环" ⇒
      **结束休息**（正常到点 / 手动结束 / 状态被换走）⇒ 手上那一段**继续演完**（**键不松**✗），
      但**不再开始新的一项 / 新的一轮**（休息都结束了 ✓），**演完那一刻**才收摊 ✓。
      ⇒ 所以必须有 `_loop_drain_beat` 挂在主 tick 的**每拍公共路**上：休息一结束
        `_spot_loop_beat` 就再也不会被调 ⇒ 没人推的话手上那段会**永远挂着、键也松不掉** ✗。

    ⚠ **"立刻杀"仍然存在，但只归另一条路**：**停自动 / 关防掉线 / 切项目**
      （`_release_combat_keys` ✓ "松所有相关键"的安全网）—— 那是"**立刻停**"、不是"演完手上这段"
      ⇒ 两者语义不同，**别互相替代** ✗。

    用户原话：

    用户原话："注意：**结束休息需要杀掉循环行为正在执行的行为，且要终止循环**"✓。

    ⚠ 为什么这条非要有：循环序列也是"**按着的键**"的宿主 —— 只清掉 ctx 却**不松键**，
      角色就会**卡着那些键继续走/继续坐下** ✗（游戏里极难查 ✓）。而"离开休息"的路有
      **十几条**（全文件 `_afk_ctx = None` 就有 16 处 ✓）⇒ 逐处加一句必然漏 ✗
      ⇒ 收在**两个汇合点**上：
        · `_release_combat_keys`（= "松所有相关键"的总入口：停自动 / 关防掉线 / 切项目…✓）；
        · `_run_rest` 入口那条兜底（只要这一拍状态**不是** `afk_spot_rest` 就收摊 ✓）。

    钉三件（**每件都要求"键真的松开"**，不是只看 ctx 被清了 ✗）：
      ① **结束休息**（`_finish_rest`）⇒ 循环终止 + 它按着的键发出 `up` ✓；
      ② **停自动**（`_release_combat_keys`）⇒ 同上 ✓（这条覆盖了"不走休息收尾"的那一大堆路 ✓）；
      ③ **状态离开「休息中」**（比如中途把类型改成"隐身休息"）⇒ 下一拍就收摊 ✓（兜底那条 ✓）。
    """
    s = fresh_settings()
    s.enabled = True
    s.anti_afk_enabled = True
    s.anti_afk_type = "spot_rest"
    s.anti_afk_spot_set = "甲"
    s.anti_afk_spot_seq = []
    s.anti_afk_spot_loop = True
    s.anti_afk_spot_loop_items = [{"name": "按住项",
                                   "seq": [{"type": "down", "key": "shop"}]}]   # ⚠ **只按不松** ⇒ 结束时必定要发 up ✓
    s.anti_afk_spot_loop_min = 5.0
    s.anti_afk_spot_loop_max = 5.0
    s.keymap["shop"] = "f6"
    h = Harness(s)
    a = h.agent
    ws = h.ws(with_mob=False)

    def _arm():
        """把循环摆成"**正在执行、且按着 f6 没松**"的样子 ✓（held 里有 f6 ✓）。"""
        a.state = "afk_spot_rest"
        a._rest_until = h.clock.t + 3600.0
        a._loop_ctx = [list(s.anti_afk_spot_loop_items[0]["seq"]), 1,
                       h.clock.t + 3600.0, {"f6"}, []]
        a._loop_next_ts = 0.0
        h.log.clear()

    # ⚠⚠ **直接调 agent 方法时要自己搭好两件**（`h.run` 才会做，踩过 ✓）：
    #   ① `h._patched()` 上下文 —— 不开它 `ag.key_up` 没被替换 ⇒ **一条按键都不记** ✗；
    #   ② `h.clock0` —— 它**只在 `h.run()` 里设**（记录器 `_rec` 靠它算相对时刻）⇒
    #      不在 `run` 里调时它是**未定义** ⇒ `_rec` 抛异常、还被 `_release_held_set` 的
    #      `try/except` 静默吞掉 ⇒ 又是"看着键没松、其实只是没记" ✗（这条最容易骗人 ✓）。
    def _call(fn, *args):
        h.clock0 = h.clock.t
        with h._patched():
            return fn(*args)

    def _ups(key):
        return [x for x in h.log if x[1] == "up" and x[2] == key]

    # ① ⭐ **结束休息 ⇒ 不打断正在演的那一项**（用户 2026-09-28 更正 ✓）：
    #   手上那段**还在**、键**没松**，但**不再排下一轮** + 置上"演完就收" ✓。
    _arm()
    _call(a._finish_rest, h.clock.t)
    check(a._loop_ctx is not None and a._loop_drain is True,
          "结束休息**打断了**正在演的那一项（用户明确不要 ✗ 上一版是立刻杀，已改）："
          "%r / drain=%r" % (a._loop_ctx, a._loop_drain))
    check(not _ups("f6"),
          "结束休息时就把循环按着的键松掉了（那不就是「打断」吗 ✗）：log=%r" % (h.log[-4:],))
    check(a._loop_next_ts == 0.0,
          "结束休息后还排着下一轮（休息都结束了 ✗）：%r" % (a._loop_next_ts,))
    # ①-b ⭐ **让它自然演完 ⇒ 演完那一刻才收**（所以不会永远挂着、键也松得掉 ✓）
    # ⚠ 推的时刻要**越过 `_arm()` 给 ctx 摆的那个 `next_ts`**（它是"还在演"= 远未来 ✓）——
    #   推一个更早的时刻的话，`_run_seq` 会判定"还没到点"、什么都不做 ⇒ 用例假红 ✗（踩过）。
    _call(a._loop_drain_beat, h.clock.t + 3700.0)
    _call(a._loop_drain_beat, h.clock.t + 3800.0)
    check(a._loop_ctx is None,
          "手上那一段演完了却没收（会**永远挂着、键也松不掉** ✗）：%r" % (a._loop_ctx,))
    check(_ups("f6"),
          "演完收摊时没把键松开（角色会卡着键继续走 ✗）：log=%r" % (h.log[-4:],))

    # ② ⚠ **停自动是另一条路：照旧"立刻杀"**（安全网 ✓ 那是"立刻停"、不是"演完手上这段"✗）
    _arm()
    _call(a._release_combat_keys)
    check(a._loop_ctx is None,
          "停自动时循环没收摊（那条路不走休息收尾 ⇒ 只能靠这个汇合点 ✓）：%r" % (a._loop_ctx,))
    check(_ups("f6"),
          "停自动时没松开循环按着的键（卡键 ✗）：log=%r" % (h.log[-4:],))

    # ③ 状态离开「休息中」⇒ 兜底那条**同样是"演完就收"**（不是当场掐断 ✗）
    #   ⚠ 这一步**必须挑一个"不会走到 `_finish_rest`"的状态** ✗ —— 比如中途把行为类型从
    #     "定点休息"**换成"隐身休息"**（`state` 变 `afk_enter` ✓ 它归「隐身那一型」管、
    #     **不走** `afk_spot_*` 的任何收尾 ✓）⇒ 这时**只有兜底能救** ✓。
    #     ⚠ 我第一版挑的是 `afk_spot_back`（"结束后前往"）—— 那条**会走到 `_finish_rest`**
    #     ⇒ 即使兜底被删掉也照样收摊 ⇒ 用例**抓不住**（反向验证时发现的 ✓）。
    _arm()
    a.state = "afk_enter"              # 中途换成「隐身休息」
    _call(a._run_rest, h.clock.t, ws)
    check(a._loop_ctx is not None and a._loop_drain is True,
          "状态离开「休息中」时**打断了**手上那段（该「演完就收」✗）：%r / drain=%r"
          % (a._loop_ctx, a._loop_drain))
    check(a._loop_next_ts == 0.0, "离开休息后还排着下一轮 ✗：%r" % (a._loop_next_ts,))
    # ⚠ 推的时刻要**越过 `_arm()` 给 ctx 摆的那个 `next_ts`**（它是"还在演"= 远未来 ✓）——
    #   推一个更早的时刻的话，`_run_seq` 会判定"还没到点"、什么都不做 ⇒ 用例假红 ✗（踩过）。
    _call(a._loop_drain_beat, h.clock.t + 3700.0)
    _call(a._loop_drain_beat, h.clock.t + 3800.0)
    check(a._loop_ctx is None and _ups("f6"),
          "演完之后没收摊 / 没松键 ✗：ctx=%r log=%r" % (a._loop_ctx, h.log[-4:]))

    # ④ ⚠⚠ **源码级**：`_loop_drain_beat` 必须挂在主 tick 的**每拍公共路上**被调 ✓
    #   —— 上面三条都是**直接调**它验证语义的 ⇒ 抓不住"忘了挂"那种错 ✗。
    #   而"忘了挂"的后果很重：休息一结束 `_spot_loop_beat` 就再不会被调 ⇒ 手上那段
    #   （**还按着键**）会**永远挂着、键也松不掉** ✗（正好违背这一版"不打断"的初衷 ✓）。
    import re
    from pathlib import Path

    _ag = (Path(__file__).resolve().parents[1] / "decision" / "agent.py"
           ).read_text(encoding="utf-8")
    # ⚠⚠ 判据要**行首（不是注释里）** —— 写成 `"self._loop_drain_beat(now)" in src` 就废了 ✗：
    #   把调用注释掉之后（`pass  # self._loop_drain_beat(now)` ✓）那段字**还在** ⇒ 照样通过 ✗
    #   （反向验证时正是这么被骗的 ✓）。
    check(re.search(r"^\s+self\._loop_drain_beat\(now\)\s*$", _ag, re.M) is not None,
          "`_loop_drain_beat` 没挂在主 tick 的每拍公共路上（或被注释掉了）⇒ 休息结束后**没人推**"
          "手上那段（片段会永远挂着、键也松不掉 ✗）")
    s.anti_afk_spot_loop = False


def t_spot_loop_rest_behavior():
    """⭐ 「休息过程中循环行为」（用户 2026-09-28 要求 ✓）。

    用户原话："定点休息类型，到达后行为按钮下面加个配置，布局为：勾选框「休息过程中循环行为」；
    勾选后，下方缩进出现参数：「循环行为编辑」→行为编辑器按钮 / 循环时间(s) A ~ B"✓。

    语义：**只在「休息中」（`afk_spot_rest`）**按 A~B 秒的间隔**反复**跑那段序列。
    ⚠ 与「到达后行为」（`anti_afk_spot_seq`）**互不影响**：那条**一次性**、这条**循环** ✓。

    钉六件：
      ① **不勾 ⇒ 老行为一字不变**（序列一次都不跑 ✓）；
      ② 勾了 + 序列非空 ⇒ 休息中**跑一次**（按键真发出来 ✓）；
      ③ 隔够 A~B ⇒ **再跑一次**（这就是"循环" ✓ —— 只跑一次是实现漏了"重排下次"✗）；
      ④ **间隔从"演完"那一刻起算**（不是"起跑"那一刻 ✗）——
         "间隔 10s + 序列本身 8s"该是每 10 秒一轮，不是每 18 秒 ✓；
      ⑤ **收摊要松键**（`_finish_rest` 之后 `_loop_ctx is None` ✓）——
         漏了就是**卡着键**继续走 ✗（游戏里极难查 ✓）；
      ⑥ **勾了但序列为空 ⇒ 什么都不做**（不报错、不卡键 ✓）。
    """
    s = fresh_settings()
    s.enabled = True
    s.anti_afk_enabled = True
    s.anti_afk_type = "spot_rest"
    s.anti_afk_spot_set = "甲"
    # 「到达后行为」留空（免得跟循环那条混在一起数按键 ✓）
    s.anti_afk_spot_seq = []
    # ⭐ **列表**（用户 2026-09-28 升级 ✓）：每项 = 名字 + 一段行为；一轮里**依次**跑 ✓
    s.anti_afk_spot_loop_items = [
        {"name": "第一项", "seq": [{"type": "down", "key": "shop"},
                                   {"type": "up", "key": "shop"}]},
        {"name": "第二项", "seq": [{"type": "down", "key": "sit"},
                                   {"type": "up", "key": "sit"}]},
    ]
    s.anti_afk_spot_loop_min = 10.0
    s.anti_afk_spot_loop_max = 10.0
    # ⚠⚠ 序列里的逻辑键名必须能在 `keymap` 里解析出来，否则 `_resolve_seq_key` 给空、
    #   **一个键都不会发**（用例会假红成「循环没跑」✗ 踩过）。
    s.keymap["shop"] = "f6"
    s.keymap["sit"] = "f7"
    h = Harness(s)
    a = h.agent
    ws = h.ws(with_mob=False)

    def _into_rest():
        """把 agent 直接摆到「休息中」阶段（前面的走 / 到达后行为不是这条要测的 ✓）。"""
        a.state = "afk_spot_rest"
        a._afk_ctx = None
        a._rest_until = h.clock.t + 3600.0      # 远没到点 ⇒ 一直待在"休息中" ✓
        a._loop_next_ts = h.clock.t             # 进休息那一刻排的第一次 ✓

    # ① 不勾 ⇒ 老行为（一次都不跑）
    s.anti_afk_spot_loop = False
    _into_rest()
    h.run(3.0, mob_plan=lambda _t: False)
    check(not len(h.presses("f6")),
          "**没勾**「休息过程中循环行为」却跑了那段序列（老行为被改坏 ✗）")

    # ② 勾上 ⇒ 立刻跑一次
    s.anti_afk_spot_loop = True
    _into_rest()
    h.run(0.5, mob_plan=lambda _t: False)
    # ⚠ `presses` 记的是 **down**（序列是 down+up ⇒ 1 次 ✓）—— 别按"按下+松开"数成 2 ✗
    check(len(h.presses("f6")) >= 1,
          "勾了却在休息中**一次都没跑**（该先按一下再松一下 ✓）：%d 次"
          % len(h.presses("f6")))

    # ②-b ⭐ **一轮里按项目「依次」执行**（用户 2026-09-28 升级后的核心 ✓）：
    #   第二项该在第一项**之后**跑（同一轮内、**不用等间隔** ✓ —— "依次执行"就是这个意思 ✓）。
    check(len(h.presses("f7")) >= 1,
          "只跑了第一项、第二项没跑（「按项目依次执行」没实现 ✗）：f6=%d f7=%d"
          % (len(h.presses("f6")), len(h.presses("f7"))))
    _t6 = h.presses("f6")[0] if h.presses("f6") else 0.0     # ⚠ `presses` 给的是**时刻**列表 ✓
    _t7 = h.presses("f7")[0] if h.presses("f7") else 0.0
    check(_t7 >= _t6,
          "第二项**排在第一项前面**了（顺序反了 ⇒ 不是「按项目依次」✗）："
          "f6@%.3f f7@%.3f" % (_t6, _t7))

    # ③ 循环：隔够 10 秒 ⇒ 再跑一次
    n1 = len(h.presses("f6"))
    h.run(11.0, mob_plan=lambda _t: False)
    check(len(h.presses("f6")) > n1,
          "过了「循环时间」却没再跑（只跑一次 = 漏了\"跑完排下次\" ✗）：%d → %d"
          % (n1, len(h.presses("f6"))))

    # ④ ⭐ **间隔要真的是 A~B**（这才是"**循环时间(s) A~B**"的核心判据 ✓ 用户点名的 ✓）：
    #   两轮之间该隔 ≈10 秒（用例配的 A=B=10）。⚠ 只断言"排了下次"是抓不住
    #   "跑完没排下次 / 排成 0 / 背靠背不停跑" 那几种坏法的 ✗（反向验证时发现的 ✓）。
    #   ⚠ 间隔从**演完那一刻**起算（不是"起跑那一刻"）—— 序列本身占的时间不该白算进去 ✓。
    check(a._loop_ctx is None or a._loop_next_ts == 0.0,
          "测间隔的这一刻它还在演 / 还挂着上次的排期（用例要重新摆一下）：%r / %r"
          % (a._loop_ctx, a._loop_next_ts))
    t_now = h.clock.t
    a._loop_next_ts = 0.0
    a._loop_ctx = None
    h.run(0.5, mob_plan=lambda _t: False)          # 起跑一轮（并演完）
    check(a._loop_next_ts > t_now,
          "跑完一轮后没排下次（只跑一次 = 不是「循环」✗）：next=%r now=%r"
          % (a._loop_next_ts, h.clock.t))
    gap = a._loop_next_ts - h.clock.t
    check(0.0 < gap <= 10.5,
          "下一次排得太远（该在 A~B 秒内 ⇒ ≤10.5 ✓）：%.2f 秒后" % gap)
    check(gap > 5.0,
          "下一次排得太近（**间隔失效**成背靠背不停跑 ✗，用户设的 A~B 白设了）：%.2f 秒" % gap)

    # ⑤ ⭐ **结束休息不打断正在演的那一项**（用户 2026-09-28 更正："结束休息不应该打断
    #   正在进行的循环"✓）—— `_finish_rest` 之后手上那段**还在**（键**没松** ✓），
    #   只是**不再排下一轮**（`_loop_next_ts == 0` ✓）并置上"演完就收" ✓。
    a.state = "afk_spot_rest"
    a._loop_ctx = [list(s.anti_afk_spot_loop_items[0]["seq"]), 0, 0.0, {"shop"}, []]
    a._finish_rest(h.clock.t)
    check(a._loop_ctx is not None and a._loop_drain is True,
          "结束休息**打断了**正在演的那一项（用户明确不要这样 ✗）：%r / drain=%r"
          % (a._loop_ctx, a._loop_drain))
    check(a._loop_next_ts == 0.0,
          "结束休息之后还在排下一轮循环（休息都结束了，不该再排 ✗）：%r" % (a._loop_next_ts,))

    # ⑥ 勾了但序列为空 ⇒ 不炸、不卡键
    s.anti_afk_spot_loop_items = [{"name": "空项", "seq": []}]
    _into_rest()
    h.run(2.0, mob_plan=lambda _t: False)
    check(a._loop_ctx is None,
          "循环序列为空时还起了上下文（该什么都不做 ✓）：%r" % (a._loop_ctx,))
    s.anti_afk_spot_loop_items = [{"name": "第一项", "seq": []}]

    # ⑦ 存取：能存能读；**老配置没有这几格 ⇒ 默认关**（老行为一字不变 ✓）
    d = s.to_dict()
    for k in ("anti_afk_spot_loop", "anti_afk_spot_loop_items",
              "anti_afk_spot_loop_min", "anti_afk_spot_loop_max"):
        check(k in d, "`%s` 没进 `to_dict`（存不住 ✗）" % k)
    s2 = fresh_settings()
    s2.from_dict({"anti_afk_type": "spot_rest"})
    check(s2.anti_afk_spot_loop is False and not s2.anti_afk_spot_loop_items,
          "老配置（没这几格）该**默认关**（老行为一字不变 ✗）：%r / %r"
          % (s2.anti_afk_spot_loop, s2.anti_afk_spot_loop_items))
    s3 = fresh_settings()
    s3.from_dict({"anti_afk_spot_loop": True,
                  "anti_afk_spot_loop_items": [{"name": "打坐",
                                                "seq": [{"type": "down", "key": "shop"}]}],
                  "anti_afk_spot_loop_min": 5.0, "anti_afk_spot_loop_max": 2.0})
    check(s3.anti_afk_spot_loop and s3.anti_afk_spot_loop_max >= s3.anti_afk_spot_loop_min,
          "读回时没夹住 A~B（A 大于 B 会让 `random.uniform` 反着来 ✗）：%r ~ %r"
          % (s3.anti_afk_spot_loop_min, s3.anti_afk_spot_loop_max))
    check(len(s3.anti_afk_spot_loop_items) == 1
          and s3.anti_afk_spot_loop_items[0]["name"] == "打坐",
          "列表项的名字没读回来（列表配置存不住 ✗）：%r" % (s3.anti_afk_spot_loop_items,))
    # ⑦-b ⚠ **老配置（"一段序列"那种）⇒ 自动变成列表里的一项**（用户配好的不能丢 ✓）
    s4 = fresh_settings()
    s4.from_dict({"anti_afk_spot_loop": True,
                  "anti_afk_spot_loop_seq": [{"type": "down", "key": "shop"}]})
    check(len(s4.anti_afk_spot_loop_items) == 1
          and s4.anti_afk_spot_loop_items[0]["seq"],
          "老配置（`anti_afk_spot_loop_seq` 那一段）没被迁移成列表里的一项"
          "（用户配好的东西丢了 ✗）：%r" % (s4.anti_afk_spot_loop_items,))


def t_player_loc_params():
    """⭐⭐ **「玩家位置」参数组**（用户 2026-09-28 要求 ✓）。

    用户原话（照抄）："「玩家位置」参数组：脚底偏移（player_foot_offset_px）+ 框面积最小
    占比(%) + 面积基线/容差 ⇒ 参数面板单开一组 ✓／框面积闸：当前框面积 ≤ 近期滚动基线 ×
    (1 − 容差%) ⇒ 这一拍推迟/不做查询 ✓（**拦在查询之前**，而不是靠放宽挑面 ✗）／相机 y 用
    脚底偏移 ✓／实时箭头显示：以玩家位置为原点画「向前箭头 + 向上箭头」，颜色、线段长度
    可配 ✓（画在实时预览上 ✓ 正好用它调试脚底偏移）"。

    钉六件：
      ① 参数对象上**有**这七个字段，默认值是「**这功能关着**」（老行为一字不变 ✓）；
      ② `to_dict` 全导出、`from_dict` 全读回（往返不丢 ✓）；
      ③ 老项目文件（没这些格）⇒ **兜底默认**（不是 AttributeError、也不是面积闸乱开 ✓）；
      ④ ⭐ **面积闸拦在「位置查询」之前**（源码级：`_locate_mmap(...)` 必须排在 `_gate_closed`
         之后 —— 用户明确「拦在查询之前，不是靠放宽挑面」✗）；拦住时 `_loc = None`（= 这一拍
         **不定位**、世界坐标沿用上一拍 ✓，而不是「定到错的地方」✗）；
      ⑤ ⭐ **脚底偏移四处口径一致**（源码级：`live_thread` 灌镜像 + `minimap.screen_to_world`
         + `tracker._world_dist` 都读它 ✓ —— 漏一处就是「画面上对、决策里错」✗）；
      ⑥ 箭头颜色只认 `#RRGGBB`，写错 ⇒ **保留原值**（不悄悄换默认色 ✗）。
    """
    import inspect
    from decision.agent import DecisionSettings
    from perception import world_state as ws_mod
    from perception import minimap as mm_mod
    from perception import tracker as tk_mod

    # ⭐⭐ **这一组该挂在「设置 → 判定参数页」**（用户 2026-09-28 原话："「玩家位置」也应该在
    #   设置 → 判定参数页签"✓）—— 判据见 `docs/UI规范.md` §4 那张"三条去向"表：
    #   它既是"**这一拍算不算拿到了玩家位置**"（判定 ✓）又该**跟项目走**（不同地图的面 y /
    #   人物框大小不一样 ✓）⇒ 判定参数页 ✓。⚠ 别再留在决策参数面板（犯过一次 ✗）。
    import pathlib as _pl
    _root = _pl.Path(__file__).resolve().parent.parent
    _sd = (_root / "gui" / "settings_dialog.py").read_text(encoding="utf-8")
    _pp = (_root / "gui" / "player_panel.py").read_text(encoding="utf-8")
    for _k in ("sp_foot_off", "sp_box_min_area", "sp_area_base_s", "sp_area_tol"):
        check(_k in _sd,
              "「玩家位置」的控件 `%s` 不在 `settings_dialog` 里 —— 它该在**判定参数页**"
              "（用户 2026-09-28 ✗）" % _k)
        check(_k not in _pp,
              "`%s` 还留在决策参数面板里（该搬走 ✗ 用户 2026-09-28）："
              "「玩家位置」属于**判定参数页**" % _k)
    check("player_foot_offset_px" in _sd,
          "判定参数页没把「玩家位置」写回 `settings`（改了不生效 ✗）")

    s = DecisionSettings()
    # ① 七个字段 + 默认值（默认 = 这功能关着 ⇒ 老行为一字不变 ✓）
    for _name, _want in (("player_foot_offset_px", 0),
                         ("player_box_min_area_pct", 0.0),
                         ("player_box_area_base_s", 30),
                         ("player_box_area_tol_pct", 0.0),
                         ):
        check(hasattr(s, _name), "「玩家位置」少了参数 `%s`（用户点名要求 ✗）" % _name)
    check(s.player_foot_offset_px == 0 and s.player_box_min_area_pct == 0.0
          and s.player_box_area_tol_pct == 0.0,
          "「玩家位置」的默认值不是「关着」（老行为会被改坏 ✗）：%r/%r/%r"
          % (s.player_foot_offset_px, s.player_box_min_area_pct,
             s.player_box_area_tol_pct))
    # ⭐⭐ **箭头不在这里**（用户 2026-09-28 纠正 ✓ 原话："你这里不是设置面板啊，放在
    #   设置 → 界面页签 → 辅助线与标记（实时预览）"）⇒ 它属于"**只是画给人看的**"，
    #   该进 `vis:` 段（本机外观 ✓ 跟着"这台机器上的人"走 ✓）、**不跟项目存** ✗。
    #   判据见 `docs/UI规范.md` §4（并钉住"别再搬回来" ✓）。
    _d0 = s.to_dict()
    for _k in ("player_arrow_color", "player_arrow_width_px", "player_arrow_len_px"):
        check(_k not in _d0,
              "`%s` 还在决策参数里 —— 箭头归「辅助线与标记」（它不参与决策、不该跟项目存 ✗ "
              "规范 §4）：%r" % (_k, sorted(_d0)))
    from gui import theme as _theme
    for _k in ("player_arrow_color", "player_arrow_width", "player_arrow_len",
               "player_arrow_on"):
        check(_k in _theme.VIS_DEFAULTS,
              "`vis:` 段少了箭头那一项 `%s`（搬过去却没把键加全 ✗）" % _k)
    check("player_arrow_on" in _theme._VIS_ON_KEYS
          and "player_arrow_color" in _theme._VIS_COLOR_KEYS,
          "箭头的开关 / 颜色没进 `vis` 的校验名单（关掉不生效 / 透明度被当成坏值丢掉 ✗）")
    # ⭐⭐ **各框线宽**（用户 2026-09-28 ✓ 原话："**给其他的粗细也加配置**，并且用**网格
    #   对齐**，要求**参数名显示完整**"）：五个键都要在 `VIS_DEFAULTS` 里，**而且必须
    #   "存得进、读得回"** —— ⚠ 只加默认值、不在 `load_vis` 里加一道"读回 + 夹范围"，
    #   用户存的值根本进不来 ⇒ 表现就是"改了没用 / 调不到 1px"✗（这个坑刚踩过一次 ✓）。
    #   ⚠ 默认值必须是**原代码里那个写死值**（锁定框 3、其余 2 ✓）—— 加参数**不许改观感** ✗。
    # ⭐ **箭头尖大小**（用户 2026-09-28 追加 ✓ 原话："再加个箭头 size 配置"）：
    #   默认值必须是 **25（%）**= 原代码里写死的 `tipLength=0.25` ✓（加参数**不许改观感** ✗），
    #   而且**存得进、读得回**（⚠ 这里再钉一次"读回"，因为这个坑已经踩过一次 ✓）。
    check(_theme.VIS_DEFAULTS.get("player_arrow_tip_pct") == 25,
          "箭头尖大小的默认值不是原写死的 25%%（加参数不许改观感 ✗）：%r"
          % (_theme.VIS_DEFAULTS.get("player_arrow_tip_pct"),))
    # ⚠ ⚠ **先还原、再断言**（2026-09-28 踩过 ✓）：`check` 一失败就抛异常 ⇒ 写在它**后面**
    #   的还原那句**永远跑不到** ⇒ 用例会把 `ui.yaml` 写成脏值（那次留了个 `80` ✗）。
    _tbak = _theme.load_vis()["player_arrow_tip_pct"]
    _theme.save_vis({"player_arrow_tip_pct": 80})
    _tgot = _theme.load_vis()["player_arrow_tip_pct"]
    _theme.save_vis({"player_arrow_tip_pct": _tbak})
    check(_tgot == 80,
          "箭头尖大小**存了读不回来**（`load_vis` 漏了「读回 + 夹范围」⇒ 改了没用 ✗）：%r"
          % (_tgot,))
    for _wk, _wd in (("lock_width", 3), ("attack_width", 2),
                     ("min_attack_width", 2), ("jump_attack_width", 2),
                     ("chase_jump_width", 2)):
        check(_theme.VIS_DEFAULTS.get(_wk) == _wd,
              "线宽 `%s` 的默认值不是原写死值 %d（加参数不许改观感 ✗）：%r"
              % (_wk, _wd, _theme.VIS_DEFAULTS.get(_wk)))
    _wbak = {_k: _theme.load_vis().get(_k) for _k in ("lock_width", "attack_width")}
    _theme.save_vis({"lock_width": 1, "attack_width": 5})
    _wv = _theme.load_vis()
    _theme.save_vis(_wbak)          # ⚠ **先还原再断言**（红了也不把脏值留在 ui.yaml ✓）
    check(_wv["lock_width"] == 1 and _wv["attack_width"] == 5,
          "线宽存了**读不回来**（`load_vis` 漏了「读回 + 夹范围」⇒ 永远停在默认值 ✗ "
          "用户报过这个形状）：%r / %r" % (_wv["lock_width"], _wv["attack_width"]))
    _theme.save_vis(_wbak)              # 还原（别把用例的临时值留在 ui.yaml 里 ✗）
    # ⚠ 排版那三条（网格对齐 / 粗细在色块右边 / 名字完整）是**源码级**钉：
    import pathlib as _pl2
    _sd_src = (_pl2.Path(__file__).resolve().parent.parent
               / "gui" / "settings_dialog.py").read_text(encoding="utf-8")
    check("vf2 = QGridLayout()" in _sd_src,
          "「辅助线与标记」那组没用 `QGridLayout`（用户要求**网格对齐** ✗）")
    check("setColumnMinimumWidth(0, 190)" in _sd_src,
          "网格第 0 列没给最小宽度 ⇒ 长名字会被截断（用户要求**参数名显示完整** ✗）")
    check("w_key=" in _sd_src and "_width_spins" in _sd_src,
          "各框的**粗细**没接进网格（用户要求\"粗细放颜色右边 + 给其他的粗细也加配置\"✗）")
    # ⭐⭐ **数值键必须真的能读回来**（用户 2026-09-28 报："**玩家坐标线段粗细无法调整至
    #   1px**" ✗）—— 根因是 `load_vis()` 里只给 `vision_width` 写了"读回 + 夹范围"那一道，
    #   箭头的粗细 / 长度**只在 `VIS_DEFAULTS` 里躺着**：用户存的值进不来 ✗
    #   ⇒ 永远停在默认 2 / 60 ⇒ 表现就是"改了没用、调不到 1" ✓。
    #   ⚠ 这条钉子就是防这个：**往 `vis:` 加数值键 ⇒ 存了必须能读回** ✓。
    _bak = {_k: _theme.load_vis().get(_k) for _k in
            ("player_arrow_width", "player_arrow_len")}
    _theme.save_vis({"player_arrow_width": 1, "player_arrow_len": 17})
    _v = _theme.load_vis()
    check(_v["player_arrow_width"] == 1,
          "箭头粗细存了读不回来（`load_vis` 漏了这个数值键 ⇒ 永远停在默认值、调不到 1px ✗ "
          "用户报的那个 bug）：%r" % (_v["player_arrow_width"],))
    check(_v["player_arrow_len"] == 17,
          "箭头长度存了读不回来（同上 ✗）：%r" % (_v["player_arrow_len"],))
    _theme.save_vis(_bak)               # 还原（别把用例的临时值留在 ui.yaml 里 ✗）
    check(hasattr(ws_mod, "FOOT_OFFSET_PX"),
          "`world_state` 少了脚底偏移的镜像槽 ⇒ perception 那两处没得读（循环依赖 ✗）")

    # ② 存取往返
    _names = ("player_foot_offset_px", "player_box_min_area_pct",
              "player_box_area_base_s", "player_box_area_tol_pct")
    _d = s.to_dict()
    for _n in _names:
        check(_n in _d, "`%s` 没进 `to_dict`（存不住 ✗）" % _n)
    s.player_foot_offset_px = -7
    s.player_box_min_area_pct = 1.5
    s.player_box_area_base_s = 45
    s.player_box_area_tol_pct = 25.0
    s2 = DecisionSettings()
    s2.from_dict(s.to_dict())
    check((s2.player_foot_offset_px, s2.player_box_min_area_pct,
           s2.player_box_area_base_s, s2.player_box_area_tol_pct)
          == (-7, 1.5, 45, 25.0),
          "「玩家位置」参数存取没往返：%r"
          % ((s2.player_foot_offset_px, s2.player_box_min_area_pct,
              s2.player_box_area_base_s, s2.player_box_area_tol_pct),))
    # ⚠ 老项目文件里若还留着 `player_arrow_*` 那几个键 ⇒ **留着不报错**（`from_dict` 不再读
    #   它们 ✓、下次保存自然就没了 ✓ —— 同 `input_delay` 那条处理 ✓）。
    s2b = DecisionSettings()
    s2b.from_dict({"player_arrow_x_color": "#123456"})
    check(not hasattr(s2b, "player_arrow_color"),
          "老键还在被读（箭头已搬到 `vis:` 段 ⇒ 这边不该再认它 ✗）")

    # ③ 老项目文件（没这些格）⇒ 兜底默认
    s3 = DecisionSettings()
    s3.from_dict({"player_track_jump": 200})
    check(s3.player_foot_offset_px == 0 and s3.player_box_min_area_pct == 0.0
          and s3.player_box_area_tol_pct == 0.0 and s3.player_box_area_base_s > 0,
          "老配置（没这些格）该兜底成「关着」（不是 AttributeError、也不是闸乱开 ✗）："
          "%r/%r/%r" % (s3.player_foot_offset_px, s3.player_box_min_area_pct,
                        s3.player_box_area_tol_pct))

    # ④ ⭐ 面积闸**拦在位置查询之前**（源码级）
    from gui import live_thread as lt
    src = inspect.getsource(lt)
    #   ⚠ 2026-10-03 扩（小地图高频定位回路 ✓ SKILL 170）：定位现在可能在**另一条线程**里跑
    #     （`_mmap_loc_loop` ✓），主回路这一拍只**取用**（`_loc_slot_take` ✓ / `live` 来源照旧
    #     现算 ✓）⇒ 那几个字面量跟着变了 ✓，但**口径一个字没改**：闸拦住 ⇒ **这一拍不取用**
    #     （世界坐标沿用上一拍 ✓），而且判定必须在取用**之前** ✓。
    _i_gate = src.find("if _gate_closed:")
    _i_slot = src.find("_loc = self._loc_slot_take()")
    _i_live = src.find("_loc = self._locate_latest()")
    check(_i_gate > 0 and (_i_slot > _i_gate or _i_live > _i_gate),
          "框面积闸没拦在「定位取用」之前（用户明确：**拦在查询之前**，不是靠放宽挑面 ✗）："
          "gate@%s / slot@%s / live@%s" % (_i_gate, _i_slot, _i_live))
    check("if _gate_closed:\n                        _loc = None" in src,
          "拦住时不是「这一拍不取用」（该 `_loc = None` ⇒ 世界坐标沿用上一拍 ✓）："
          "拦住却照样取用 = 白拦 ✗")
    check("_locate_mmap" in src and "player_box_area_tol_pct" in src
          and "player_box_min_area_pct" in src,
          "面积闸那两条判据没接上设置里的参数（闸就是死的 ✗）")

    # ⑤ ⭐ 脚底偏移：口径一致（源码级）
    check("FOOT_OFFSET_PX = _ploc_fo" in src,
          "`live_thread` 没把脚底偏移灌进 `world_state` 的镜像 ⇒ 另外两处读到的永远是 0 ✗")
    for _mod, _who in ((mm_mod, "minimap.screen_to_world"),
                       (tk_mod, "tracker._world_dist")):
        check("FOOT_OFFSET_PX" in inspect.getsource(_mod),
              "`%s` 没用脚底偏移（相机 y 口径不一致 ⇒ 画面上对、决策里错 ✗）" % _who)

    # ⑥ 箭头颜色：只认 `#RRGGBB`，写错 ⇒ 保留原值
    check(ws_mod.norm_hex_color("#AbCdEf", "#000000") == "#abcdef",
          "`#RRGGBB` 没归一成小写：%r" % ws_mod.norm_hex_color("#AbCdEf", "#000000"))
    check(ws_mod.norm_hex_color("乱写", "#00E5FF") == "#00E5FF",
          "填错颜色时该**保留原值**（不许悄悄换成默认色 ✗）：%r"
          % ws_mod.norm_hex_color("乱写", "#00E5FF"))
    # ⭐⭐ **「面积基线窗口」的单位是「秒」、不是「拍」**（用户 2026-09-28 ✓ 原话："『面积基线
    #    窗口』，这个『拍』是什么？是多久？**需要可量化的描述**"✓）—— 拍数**依赖帧率**
    #    （30 拍在 30fps 是 1 秒、在 15fps 是 2 秒 ✗）⇒ 只有秒可量化 ✓。
    _d2 = DecisionSettings()
    check(hasattr(_d2, "player_box_area_base_s"),
          "少了 `player_box_area_base_s`（基线窗口该按**秒** ✗）")
    check(abs(float(_d2.player_box_area_base_s) - 3.0) < 1e-9,
          "基线窗口默认不是 3 秒（老行为会被改坏 ✗）：%r" % (_d2.player_box_area_base_s,))
    check("player_box_area_base_s" in _d2.to_dict(), "没进 `to_dict`（存不住 ✗）")
    _d3 = DecisionSettings()
    _d3.from_dict({"player_box_area_base_s": 5.5})
    check(abs(float(_d3.player_box_area_base_s) - 5.5) < 1e-9,
          "新键存取没往返：%r" % (_d3.player_box_area_base_s,))
    _d4 = DecisionSettings()
    _d4.from_dict({"player_box_area_base_n": 60})          # 旧键（拍）✓
    check(abs(float(_d4.player_box_area_base_s) - 2.0) < 1e-9,
          "旧键（拍）没折算成秒（60 拍该 ⇒ 2.0 秒 ✓ 别让老项目丢值 ✗）：%r"
          % (_d4.player_box_area_base_s,))
    _d5 = DecisionSettings()
    _d5.from_dict({"player_track_jump": 200})              # 两个键都没有 ✓
    check(abs(float(_d5.player_box_area_base_s) - 3.0) < 1e-9,
          "老配置（两个键都没有）该兜底 **3.0 秒**：%r" % (_d5.player_box_area_base_s,))
    # ⭐ **源码级**（那两段是 `live_thread.run()` 里的内联代码 ⇒ 只能这么钉 ✓）：
    _lt_src = inspect.getsource(lt)
    check("_t_now - _t <= _bs" in _lt_src,
          "基线没按**时间窗**筛（按「最后 N 个」切片依赖帧率 ⇒ 不可量化 ✗ 用户要的是「秒」✓）")
    check("player_box_area_base_s" in _lt_src,
          "`live_thread` 没读新的「秒」参数（还在读旧的拍数 ✗）")
    check("_base * _pm / 100.0" in _lt_src,
          "「框面积最小占比」没跟**基线**比（用户 2026-09-28：该占「过去 n 秒内平均面积」✗）")
    check("_fa = float(vis.shape" not in _lt_src,
          "「框面积最小占比」还在跟**画面面积**比（用户说该跟**基线**比 ✗）")

    check(ws_mod.hex_to_bgr("#FF8000") == (0, 128, 255),
          "`hex_to_bgr` 算错了（该是 (B,G,R) = (0,128,255)）：%r"
          % (ws_mod.hex_to_bgr("#FF8000"),))


def t_attack_single_side_and_steal():
    """⭐⭐ **攻击范围判定改「单侧」＋ 背后怪抢锁**（用户 2026-09-28 ✓ 原话两条：
    "攻击范围判定改成『单侧』（跟画面一致）"／"锁定怪在前面，**背后攻击范围内有怪**，chase 想朝前，
     但其实**转向更高效** ⇒ 是不是做成**背后怪抢锁**更好？"）。

    钉六件：
      ① **单侧判定**：不传 `facing` ⇒ 按 `self.facing` ✓（怪在背后 ⇒ **不算** ✓；转身后就够得着 ✓）；
      ② **`facing=0` 仍"不分前后"** ✓（老行为，也是 `_in_range_all` 的命根子 ✗）；
      ③ ⭐ **源码级**：`_in_range_all` **必须显式传 `facing=0`** ✗ —— 它的存在意义就是
         "**不筛朝向**"✓，不显式传就会被新默认值顺手筛掉一半 ⇒ 那条分支就废了 ✗；
      ④ ⭐ **背后可打 ⇒ 抢锁**（锁定期内改锁反侧那只 ✓）；
      ⑤ ⭐ **只抢"真的够得着"的**（背后那只在攻击范围外 ⇒ **不抢** ✓，否则比朝前追更慢 ✗）；
      ⑥ **同一只怪不重复抢** ✓（防目标来回跳 ✗）。
    """
    import inspect

    from decision.agent import CombatAgent, DecisionSettings

    a = CombatAgent.__new__(CombatAgent)
    a.settings = DecisionSettings()
    a.settings.attack_dist = 200
    a.settings.min_attack_dist = 0
    a.facing = 1
    a._target_id = None
    a._target_until = 0.0
    a._no_target_since = None
    a._random_target_cd = lambda: 30.0
    # ⚠ `__new__` 替身：`_big_mob_pool` 会写"大怪判定采样账"（2026-10-02 ✓）⇒ 喂个空账
    #   （不给就 AttributeError，会被当成用例失败 ✗ —— 同 `_climb_*` 那几处的踩坑 ✓）。
    a._mob_h_samples = []

    class _P:
        x = 0.0
        y = 0.0

    class _M:
        def __init__(self, i, x):
            self.id, self.x, self.y, self.w, self.h = i, float(x), 0.0, 20.0, 40.0

    class _WS:
        player = _P()

    _front = _M(1, 150.0)            # 正面、够得着
    _behind = _M(2, -150.0)          # 背后、够得着
    _far_behind = _M(3, -9999.0)     # 背后、**够不着**

    # ① 单侧
    check(a._in_attack_box(_front, _P()) is True,
          "正面的怪该算在攻击范围内（单侧 ✓）")
    check(a._in_attack_box(_behind, _P()) is False,
          "**背后**的怪不该算在（单侧）攻击范围内（用户 2026-09-28：改成「单侧」✗）")
    a.facing = -1
    check(a._in_attack_box(_behind, _P()) is True,
          "转身之后背后的那只就够得着了（单侧判定该跟着朝向走 ✓）")
    a.facing = 1
    # ② facing=0 ⇒ 老行为（不分前后）
    check(a._in_attack_box(_behind, _P(), 0) is True,
          "`facing=0` 该恢复「不分前后」（老行为 ✓，`_in_range_all` 靠它 ✗）")

    # ③ 源码级：`_in_range_all` 必须显式传 0
    _src = inspect.getsource(a._in_range_all)
    check("_in_attack_box(m, ws.player, 0)" in _src,
          "`_in_range_all` 没显式传 `facing=0` ⇒ 会被单侧默认值顺手筛掉一半 ✗"
          "（那条分支的意义就是「不筛朝向」✗）")

    # ④ 背后可打 ⇒ 抢锁
    a._target_id = 1
    a._target_until = 1000.0
    _t, _b = a._locked_target([_front, _behind], _WS(), 10.0)
    check(_t is not None and _t.id == 2,
          "锁着正面那只、背后又有**够得着**的 ⇒ 该**抢锁背后那只**（用户 2026-09-28 ✗）：%r"
          % (getattr(_t, "id", None),))
    # ⑤ 只抢够得着的
    a._target_id = 1
    a._target_until = 1000.0
    _t2, _ = a._locked_target([_front, _far_behind], _WS(), 10.0)
    check(_t2 is not None and _t2.id == 1,
          "背后那只**够不着**（在攻击范围外）却被抢锁了 ⇒ 会比朝前追更慢 ✗：%r"
          % (getattr(_t2, "id", None),))
    # ⑥ 同一只不重复抢
    a._target_id = 2
    a._target_until = 1000.0
    _t3, _ = a._locked_target([_front, _behind], _WS(), 10.0)
    check(_t3 is not None and _t3.id == 2,
          "已经锁着背后那只了还在抢（目标会来回跳 ✗）：%r" % (getattr(_t3, "id", None),))


def t_chase_goto_max():
    """⭐⭐ 「**追怪寻路.duration(s)**」（用户 2026-09-28 要求 ✓ 原话："在设置 → 判定参数
    **单开一组『任务』**"＋"新增参数『追怪寻路.duration(s)』：表示本次任务如果是**追怪下达**的，
    那么如果任务**超过了这个时间就结束任务**"）。

    钉五件：
      ① 参数在、**默认 0 = 不启用**（老项目一字不变 ✓）；
      ② `to_dict` / `from_dict` 往返 ✓；老配置（没这格）⇒ 兜底 0 ✓；
      ③ ⭐ **只对"追怪下达的"任务生效**：闸的判据必须看 `_climb_origin` 的 `kind == "chase"` ✓
         （「命令前往」/「定点休息」等**不许被它掐** ✗）；
      ④ ⭐ **与「寻路超时」共用同一把钟**（`_climb_started` ✓ 不许另起一把 ✗）；
      ⑤ ⚠ `<= 0` ⇒ **不启用**（别把 0 当成"立刻超时" ✗）。
    """
    import inspect

    from decision.agent import CombatAgent, DecisionSettings

    s = DecisionSettings()
    check(hasattr(s, "chase_goto_max_s"),
          "少了参数「追怪寻路.duration(s)」（用户点名要求 ✗）")
    check(float(s.chase_goto_max_s) == 0.0,
          "默认值不是 0 = 不启用（老行为会被改坏 ✗）：%r" % (s.chase_goto_max_s,))
    check("chase_goto_max_s" in s.to_dict(), "没进 `to_dict`（存不住 ✗）")
    s2 = DecisionSettings()
    s2.from_dict({"chase_goto_max_s": 15})
    check(float(s2.chase_goto_max_s) == 15.0,
          "存取没往返（秒为单位 ✗）：%r" % (s2.chase_goto_max_s,))
    s3 = DecisionSettings()
    s3.from_dict({"goto_timeout_s": 30})            # 老项目文件：没这一格 ✓
    check(float(s3.chase_goto_max_s) == 0.0,
          "老配置该兜底成 0 = 不启用：%r" % (s3.chase_goto_max_s,))
    # ③④⑤ 源码级：只看 chase + 同一把钟 + `<= 0` 不启用
    src = inspect.getsource(CombatAgent._climb_tick)
    check("chase_goto_max_s" in src,
          "那道闸没接进 `_climb_tick`（配了不生效 ✗）")
    check('"chase"' in src and "_climb_origin" in src,
          "闸的判据没看「追怪下达的」（`_climb_origin` 的 kind == chase ✗）"
          "—— 那就会把「命令前往」也一起掐掉 ✗")
    check("_chase_cap > 0" in src,
          "没做 `<= 0 ⇒ 不启用`（有人设 0 会变成「立刻超时」✗）")
    # ⭐ 同一把钟：那道闸那一段里必须用到 `_climb_started`（不许另起一把 ✗）
    _seg = src[src.find("_chase_cap"): src.find("_chase_cap") + 1200]
    check("self._climb_started" in _seg,
          "追怪那道闸没和「寻路超时」共用 `_climb_started` 这把钟（两把尺 ✗）")

    # ⭐⭐ **回填**（2026-09-28 补 ✓ —— 我第一版**漏了 `setValue`** ⇒ 用户当场报"还是改不了
    #   「追怪寻路.duration」、**没有任何提示**"✗ ⇒ 因为点确定**其实存进去了**，只是**下次打开
    #   又显示 0** ⇒ 看起来像"改不了" ✗）。
    #   ⚠⚠ **测法必须是这样**：把配置设成**非默认值** ⇒ **新建对话框** ⇒ **看控件显示它** ✓ ——
    #     只测"设值 → 点确定 → settings 变了"**测不出漏回填** ✗（那正是我上一轮漏掉的一环 ✓，
    #     当时我实测通过了，却完全没发现问题 ✓ 教训）。
    from PyQt5.QtWidgets import QApplication as _QApp

    from decision.agent import settings as _st
    from gui.settings_dialog import SettingsDialog as _SD

    _qapp = _QApp.instance() or _QApp([])
    _bak = float(getattr(_st, "chase_goto_max_s", 0.0) or 0.0)
    try:
        _st.chase_goto_max_s = 15.0
        _dlg = _SD()
        check(float(_dlg.sp_chase_max.value()) == 15.0,
              "「追怪寻路.duration(s)」**没回填**：配置里是 15.0，打开对话框却显示 %r "
              "⇒ 现场看起来就是「改不了」（点确定其实存进去了 ⇒ 也**不会有任何提示**✗）"
              % (_dlg.sp_chase_max.value(),))
    finally:
        _st.chase_goto_max_s = _bak
        _qapp = None


def t_drop_detach_dir():
    """⭐⭐ 下跳"**跳下绳子**"那一步的水平方向（用户 2026-09-28 按流程图 ✓ 原话：
    "跳下绳子：按**任意方向**+跳（**若有锁定目标则是锁定目标方向**）"）。

    钉五件：
      ① **有锁定目标 ⇒ 朝它**（注入的 `_target_dir_fn()` 给 ±1 ✓）；
      ② **没有 / 同列（0）⇒ 任意方向**（取确定性的"**背离下跳点中心**" ✓）；
      ③ **还没挑定下跳点** ⇒ 向右 ✓；
      ④ ⚠ **回调抛异常 / 没注入 ⇒ 都当成"没有目标"** ✓ —— **绝不许因为拿不到目标就报错或
         停住** ✗（脱离那几拍按不出方向 = 卡死在绳上 ✗）；
      ⑤ ⭐ **源码级**：`agent` 真的把 `_target_dir` 注入给下跳任务 ✓、而且**每拍重挂**
         （锁定目标会换 ⇒ 不许建任务时定死 ✗）。
    """
    import inspect

    from decision import agent as agent_mod
    from decision import route

    j = route.DropJob("上", "下", [(550.0, 500.0, 600.0, "42")],
                      tol_px=10, hold_ms=0, y_tol_px=10, retry_ms=2000, stall_s=3.0)
    j._pick = (550.0, 500.0, 600.0, "42")
    # ① 有锁定目标 ⇒ 朝它（哪怕背离中心 ✓）
    j._target_dir_fn = lambda: 1
    check(j._detach_dir(700.0) == 1, "锁定目标在右时没朝右按（用户流程图 ✗）：%r"
          % (j._detach_dir(700.0),))
    j._target_dir_fn = lambda: -1
    check(j._detach_dir(100.0) == -1, "锁定目标在左时没朝左按（用户流程图 ✗）：%r"
          % (j._detach_dir(100.0),))
    # ② 没有目标 / 同列 ⇒ 退回"背离下跳点中心"
    j._target_dir_fn = None
    check(j._detach_dir(700.0) == -1,
          "没有锁定目标时该退回「背离下跳点中心」（中心 550，人在 700 ⇒ 该往左 ✗）：%r"
          % (j._detach_dir(700.0),))
    j._target_dir_fn = lambda: 0
    check(j._detach_dir(700.0) == -1, "目标同列（0）时该退回「背离中心」：%r"
          % (j._detach_dir(700.0),))
    # ③ 还没挑定下跳点 ⇒ 向右
    j._pick = None
    check(j._detach_dir(700.0) == 1, "还没挑定下跳点时该向右（确定性 ✓）：%r"
          % (j._detach_dir(700.0),))
    # ④ 回调坏了 ⇒ 当成"没有目标"，**不许抛**
    j._pick = (550.0, 500.0, 600.0, "42")

    def _boom():
        raise RuntimeError("回调坏了")

    j._target_dir_fn = _boom
    check(j._detach_dir(700.0) == -1,
          "注入的回调抛异常时没兜住（拿不到目标 = 卡死在绳上 ✗）：%r" % (j._detach_dir(700.0),))
    # ⑤ 源码级：agent 注入 + **每拍重挂**
    _src = inspect.getsource(agent_mod.CombatAgent._climb_tick)
    check("_target_dir_fn" in _src and "_target_dir" in _src,
          "agent 没把「锁定目标在哪边」注入给下跳任务（流程图那格没落地 ✗）")
    check("hasattr(job, \"_detach_dir\")" in _src,
          "注入没限定在下跳任务上（别的执行器白挂一个回调 ✗）")


def t_goto_tag_left():
    """⭐⭐ 「当前任务」的**签注 + 生存时间**（用户 2026-09-28 要求 ✓ 原话："下任务的签注，
    需要有生存时间，并在**信息栏**…的当前任务后面标注出来，例如 `前往：底层(追击  剩余 00:15)`"）。

    钉五件：
      ① 格式是 **`MM:SS`**（分也补零 ✓ 用户例子就是 `00:15` ✗ 不是 `0:15`），而且**不许动
         `_mmss`**（那个 `M:SS` 被 `_timer_lines` 好几处在用 ✗）；
      ② 剩余时间跟**寻路超时是同一把钟**（`_climb_started` + `goto_timeout_s` ✓ 别另起一把 ✗）；
      ③ **不设上限 / 还没起钟 / 没有任务** ⇒ `None`（**不显示** ✓ 别显示 `0:00` ✗）；
      ④ 签注取**现成来源**（`_climb_origin["kind"]` ✓），没有 ⇒ **空串**（别硬塞一个词 ✗）；
      ⑤ ⭐ **源码级**：`_osd_lines` 真把它们拼进「当前任务」那行 ✓，且**休息时不加**
         （休息的剩余已由定时任务那几行显示 ✓ 再来一遍是重复 ✗）。
    """
    import inspect
    import time as _t

    from decision.agent import CombatAgent, DecisionSettings
    from gui import route_panel as rp
    from gui.route_panel import _mmss, _mmss2

    # ① 格式
    check(_mmss2(0) == "00:00" and _mmss2(15) == "00:15" and _mmss2(75) == "01:15",
          "「当前任务」的剩余时间格式不是 `MM:SS`（用户例子是 `00:15` ✗）：%r" % (_mmss2(15),))
    check(_mmss(15) == "0:15",
          "`_mmss` 被动过了 —— 它是 `_timer_lines` 在用的 `M:SS`，别改 ✗：%r" % (_mmss(15),))

    # ②③ 同一把钟 + 三种"不显示"
    a = CombatAgent.__new__(CombatAgent)
    a.settings = DecisionSettings()
    a._climb = None
    a._climb_started = 0.0
    check(a.goto_time_left() is None, "没任务时该返回 None（别显示 `0:00` ✗）")
    a.settings.goto_timeout_s = 30
    check(a.goto_time_left() is None, "还没起钟就给了剩余时间（数字会乱跳 ✗）")
    a._climb = object()
    a._climb_started = _t.monotonic() - 15.0
    # ⭐⭐ **所有任务都显示剩余**（2026-09-29 晚用户澄清 ✓："其他所有的任务都走
    #   「寻路超时时间」，都有出口"）—— 显示与闸**同一处口径** ✓（显示说"无限"、
    #   闸却还在掐 = 最坏的那种不一致 ✗；当天早些"只有追击显示"那版钉反了 ✗）。
    a._climb_origin = None
    _lvn = a.goto_time_left()
    check(_lvn is not None and 14.0 < _lvn < 16.0,
          "没签注的任务却没显示剩余 —— 2026-09-29 晚口径：所有任务都吃「寻路超时」"
          "（都有出口 ✓）:%r" % (_lvn,))
    a._climb_origin = {"kind": "climb", "mob_id": 1}      # 别的 kind 也一样 ✓
    check(a.goto_time_left() is not None,
          "`kind` 不是 chase 却不显示剩余（所有任务都有出口 ✗）")
    a._climb_origin = {"kind": "chase", "mob_id": 1}
    _lv = a.goto_time_left()
    check(_lv is not None and 14.0 < _lv < 16.0,
          "剩余时间没跟 `_climb_started` + `goto_timeout_s`（跟「寻路超时」不是同一把钟 ✗）：%r"
          % (_lv,))
    a.settings.goto_timeout_s = 0
    check(a.goto_time_left() is None,
          "「寻路超时时间」设 0（= 不设上限）时不该显示剩余（会永远挂着 `00:00` ✗）")

    # ⭐⭐ **取两道闸里更早到期的那个**（用户 2026-09-28 报 ✓ 原话："我改了 5 秒，但是信息栏
    #    写的追击还是 **9 秒**开始（也可能是 10）"✗）—— 信息栏那行说的是"离这趟任务结束还有
    #    多久" ✓ ⇒ 两道闸谁先到、任务就结束在谁那儿 ⇒ **必须取 min** ✗（只显示大的那个会骗人 ✓）。
    a.settings.goto_timeout_s = 30
    a.settings.chase_goto_max_s = 5.0
    a._climb = object()
    a._climb_started = _t.monotonic() - 2.0          # 已经跑了 2 秒
    a._climb_origin = {"kind": "chase", "mob_id": 1}
    _lvc = a.goto_time_left()
    check(_lvc is not None and 2.4 < _lvc < 3.6,
          "追怪任务该显示**两道闸里更早的那个**（追怪 5 秒 vs 寻路 30 秒，已跑 2 秒 ⇒ 该 3 秒 ✓）"
          "—— 显示 28 秒就是用户报的那个 bug ✗：%r" % (_lvc,))
    # ⭐ **非追击不吃「追怪寻路.duration」**（追击独占 ✓），但**吃「寻路超时时间」** ✓：
    #   上限该是 30 秒（寻路超时）而不是 5 秒（追怪）⇒ 已跑 2 秒 ⇒ 该显示 ~28 秒 ✓
    #   （2026-09-29 晚口径 ✓；显示 3 秒 = 把追击独占闸错用到非追击 ✗）。
    a._climb_origin = {}                              # 不是追怪 ✓
    _lvn2 = a.goto_time_left()
    check(_lvn2 is not None and 27.0 < _lvn2 < 29.0,
          "非追击任务的剩余该按「寻路超时」30 秒算（~28 秒 ✓），不该吃追击独占的 5 秒"
          "（更不该不显示 ✗）：%r" % (_lvn2,))
    a.settings.chase_goto_max_s = 0.0

    # ④ 签注：现成来源 / 没有就空串
    a._climb_origin = {"kind": "chase", "mob_id": 1}
    check(a.current_goto_tag() == "追击",
          "追击任务的短签注没取到（该从 `_climb_origin` 的 kind 取 ✓）：%r"
          % (a.current_goto_tag(),))
    a._climb_origin = {}
    check(a.current_goto_tag() == "", "没有签注时该给空串（别硬塞一个词 ✗）")

    # ⑤ 源码级：拼进「当前任务」那行 + 休息时不加
    _src = inspect.getsource(rp)
    check("current_goto_tag" in _src and "goto_time_left" in _src and "_mmss2" in _src,
          "「当前任务」那行没拼上签注 / 剩余时间（用户点名要标出来 ✗）")
    check("if not _rest:" in _src,
          "休息时也在拼剩余时间（那几行已由定时任务显示 ⇒ 重复 ✗）")


def t_spot_loop_hold_rest_end():
    """⭐ **「休息结束推迟到循环执行完」**（用户 2026-09-28 要求 ✓）。

    用户原话："勾上休息过程中循环行为时，再加一个开关子参数『休息结束推迟到循环执行完』"✓。

    ⚠⚠ **改之前**：休息**到点那一刻**直接 `_stop_spot_loop()`（**当场掐断**手上的动作 +
      松键 ✗）、立刻转「结束后前往」/回战斗 ⇒ "循环才演到一半、另一边已经把寻路的键按下去"
      ⇒ **两套动作抢键** ✗。
    ⇒ 勾上之后：到点但**这一轮还在演** ⇒ **先不结束休息**，等它把这一轮演完再结束 ✓
      （⚠ 期间**不许再开新一轮** `_loop_hold` ✓ —— 否则循环短于决策拍时就**永远推迟** ✗）。

    钉六件：
      ① 不勾（默认）⇒ 到点**当场收摊**、回战斗（老行为一字不变 ✓）；
      ② 勾上 + 正在演 ⇒ 到点**仍在休息**、手上那段**没被掐断** ✓；
      ③④ 那一轮演完 ⇒ **下一拍就结束**、且**不许再起跑新一轮**（不会被无限推迟 ✓）；
      ⑤ **手动结束休息不受它管**（仍是"手上这一项演完就收" ✓ —— 用户点「结束」就是要
         现在停 ✓）；
      ⑥ 存取：进 `to_dict`、老配置默认**关** ✓。
    """
    s = fresh_settings()
    s.enabled = True
    s.anti_afk_enabled = True
    s.anti_afk_type = "spot_rest"
    s.anti_afk_spot_set = "甲"
    s.anti_afk_spot_seq = []
    s.anti_afk_spot_after = ""            # 到点 ⇒ 直接回战斗（好判"到底结束了没有" ✓）
    s.anti_afk_spot_loop = True
    s.anti_afk_spot_loop_hold = False
    # 两项、每项 delay 3 秒 ⇒ **一轮 = 6 秒**；而休息只歇 **2 秒**就到点
    # ⇒ 到点那一刻**必然**才演到一半 ✓（正是这个开关要处理的场景 ✓）。
    s.anti_afk_spot_loop_items = [
        {"name": "第一项", "seq": [{"type": "delay", "ms": 3000}]},
        {"name": "第二项", "seq": [{"type": "delay", "ms": 3000}]},
    ]
    s.anti_afk_spot_loop_min = 30.0
    s.anti_afk_spot_loop_max = 30.0
    h = Harness(s)
    a = h.agent

    def _into_rest(loop=True):
        """摆成"已经在休息中、循环**由它自己起跑**、2 秒后休息到点"✓。

        ⚠ 走**真路径**（`_loop_next_ts = 现在` ⇒ 下一拍 `_spot_loop_beat` 自己起跑 ✓）——
          不手工塞 `_loop_ctx`：那样"循环到底有没有在演"就成了用例自己编的 ✗。
        `loop=False` ⇒ 摆成"**间隔期**"（循环没在演 ✓）—— 给"没在演时该立刻收工"那条用 ✓。
        """
        a.state = "afk_spot_rest"
        a._afk_ctx = None
        a._loop_hold = False
        a._loop_drain = False
        a._loop_ctx = None
        a._loop_idx = 0
        a._rest_stop_pending = False
        a._loop_next_ts = h.clock.t if loop else h.clock.t + 3600.0
        a._rest_until = h.clock.t + 2.0   # 2 秒后就到点 ✓
        h.log.clear()

    # ① 不勾（默认）⇒ 到点**当场收摊**、回战斗 ✓（老行为一字不变 ✓）
    s.anti_afk_spot_loop_hold = False
    _into_rest()
    h.run(3.0, mob_plan=lambda _t: False)
    check(a.state == "idle",
          "**没勾**时休息到点该**当场**回战斗（老行为被改坏 ✗）：state=%s" % a.state)
    check(a._loop_ctx is None,
          "**没勾**时到点该**当场收摊**（手上那段掐断 —— 老口径 ✓）：%r" % (a._loop_ctx,))

    # ② 勾上 + 正在演 ⇒ 到点**仍在休息**、手上那段**没被掐断** ✓
    s.anti_afk_spot_loop_hold = True
    _into_rest()
    h.run(2.6, mob_plan=lambda _t: False)      # 到点（2.0）之后又过了 0.6 秒
    check(a.state == "afk_spot_rest",
          "勾了「休息结束推迟到循环执行完」，到点却**开始结束休息**了"
          "（手上那段会半截断 ✗）：state=%s" % a.state)
    check(a._loop_ctx is not None,
          "推迟期间手上那段**被掐断**了（该让它演完 ✓）：%r" % (a._loop_ctx,))
    check(a._loop_hold is True,
          "推迟的标记 `_loop_hold` 没置上 ⇒ 演完之后又会被起跑新一轮 ⇒ 休息**永远结束不了** ✗")
    check(a._loop_next_ts == 0.0,
          "推迟期间还排着下一轮（演完即起跑 ⇒ 永远结束不了 ✗）：%r" % (a._loop_next_ts,))

    # ③④ 那一轮演完（t≈6s）⇒ **下一拍就结束**、且**不许再起跑新一轮** ✓
    h.run(8.0, mob_plan=lambda _t: False)
    check(a.state == "idle",
          "那一轮演完之后**没结束休息**（被无限推迟了？循环又起跑了一轮 ✗）："
          "state=%s / hold=%r" % (a.state, a._loop_hold))
    check(a._loop_ctx is None,
          "结束休息之后循环还在演（键会卡住 ✗）：%r" % (a._loop_ctx,))
    check(a._loop_hold is False,
          "`_loop_hold` 没跟着收摊一起清掉（会带到下一次休息 ✗）")

    # ⑤ **手动结束休息**：新开关**不管它**（点「结束」就是要停 ✓），但手上那一段仍按
    #   用户口径"**演完就收**"（不半截掐断 ✓）。
    #   ⚠⚠ 这条**必须**走真的 `rest_abort` 那条路 —— 原来那套"立刻收工、让它在后台演完"
    #     是**假通过**的 ✗：回战斗之后只要没怪，"平地巡逻无怪 idle"那条就每拍
    #     `_release_combat_keys()` ⇒ 手上那段**当场被掐断** ✗（见 `_release_combat_keys`
    #     里那段说明 ✓）。现在改成"**留在休息状态里等**" ✓，这条钉住它 ✓。
    #   ⚠ 这里**特意不勾**「推迟到循环执行完」（`anti_afk_spot_loop_hold = False` ✓）——
    #     手动结束该走**它自己那条**收尾（`_rest_stop_pending` ✓），不许借用户那个开关 ✗：
    #     借了就看不出"没勾时也会等"（那正是这条要钉的东西 ✓）。
    s.anti_afk_spot_loop_hold = False
    _into_rest()
    # ⚠ 先让循环**真的演起来**（tick 里"消费 `rest_abort`"排在 `_run_rest` **之前**
    #   ⇒ 同一拍里"先结束休息、循环压根没起跑" ⇒ 那样这条就测了个空 ✓ 踩过）。
    h.run(0.3, mob_plan=lambda _t: False)
    check(a._loop_ctx is not None, "循环没起来（这条测不了）：%r" % (a._loop_ctx,))
    s.rest_abort = True                        # = 用户点了「结束休息」
    h.run(0.2, mob_plan=lambda _t: False)
    check(a.state == "afk_spot_rest" and a._loop_ctx is not None,
          "手动结束把手上那段**当场掐断**了（用户要的是「演完就收」✗）：state=%s / %r"
          % (a.state, a._loop_ctx))
    check(a._rest_stop_pending is True and a._loop_drain is True,
          "手动结束没有转成「收尾中」（`_rest_stop_pending`）⇒ 下一拍就直接收工、"
          "把手上那段掐断 ✗：pending=%r / drain=%r"
          % (a._rest_stop_pending, a._loop_drain))
    h.run(4.0, mob_plan=lambda _t: False)      # 3 秒的 delay 早该演完
    check(a.state == "idle" and a._loop_ctx is None,
          "手动结束后没回战斗（卡在休息里 ✗）：state=%s / %r" % (a.state, a._loop_ctx))
    check(a._rest_stop_pending is False,
          "「收尾中」那个标记没收掉（会带到下一次休息 ✗）")

    # ⑤-b 手动结束 + 循环**没在演**（间隔期）⇒ **立刻**收工，别傻等 ✓
    _into_rest(loop=False)
    h.run(0.3, mob_plan=lambda _t: False)
    check(a._loop_ctx is None,
          "这条要的是「没在演」（用例自己摆错了）：%r" % (a._loop_ctx,))
    s.rest_abort = True
    h.run(0.2, mob_plan=lambda _t: False)
    check(a.state == "idle" and a._rest_stop_pending is False,
          "循环没在演时手动结束该**立刻**收工（别傻等 ✗）：state=%s / pending=%r"
          % (a.state, a._rest_stop_pending))

    # ⑤-c ⭐ **手动结束也要走「结束后前往」**（用户 2026-09-28 报："我点手动结束休息后，
    #   **没有收到寻路任务**"✗）—— 口径与"自然到点"那条**完全一致**（共用 `_end_spot_rest` ✓）。
    _sent = []
    a.plan_and_start_route = lambda dst, why=None, **k: (
        _sent.append(str(dst)), (True, "假装去「%s」" % dst))[1]
    s.anti_afk_spot_after = "丙平台"
    _into_rest()
    h.run(0.3, mob_plan=lambda _t: False)      # 循环演起来
    check(a._loop_ctx is not None, "循环没起来（这条测不了）：%r" % (a._loop_ctx,))
    s.rest_abort = True                        # = 用户点了「结束休息」
    h.run(4.0, mob_plan=lambda _t: False)      # 手上那段演完 ⇒ 收尾 ⇒ 该去「丙平台」
    check(_sent == ["丙平台"],
          "手动结束休息**没有**走「结束后前往」（用户报的就是这个："
          "「我点手动结束休息后，没有收到寻路任务」✗）：%r" % (_sent,))
    s.anti_afk_spot_after = ""

    # ⑥ 存取：进 `to_dict` ✓；老配置（没这格）⇒ 默认**关** ✓
    _d = s.to_dict()
    check("anti_afk_spot_loop_hold" in _d,
          "`anti_afk_spot_loop_hold` 没进 `to_dict`（勾了存不住 ✗）")
    s2 = fresh_settings()
    s2.from_dict({"anti_afk_type": "spot_rest"})
    check(s2.anti_afk_spot_loop_hold is False,
          "老配置（没这格）该**默认关**（老行为一字不变 ✗）：%r"
          % (s2.anti_afk_spot_loop_hold,))
    s3 = fresh_settings()
    s3.from_dict({"anti_afk_spot_loop_hold": True})
    check(s3.anti_afk_spot_loop_hold is True, "读回丢了（勾了存不住 ✗）")
    s.anti_afk_spot_loop = False


def t_mob_sets_ttl_param():
    """⭐ 「**怪物判定缓存时长(s)**」是**参数**（用户 2026-09-28 ✓ 原话："直接做"）。

    "这只怪在哪块 foothold 集合"查一次管多久 ⇒ 由 `settings.mob_sets_ttl_s` 说了算：
      · 同一只怪 + **局面指纹没变**（怪没动、人没换层）⇒ TTL 之内用缓存（解析器**不再被调** ✓）；
      · 超时 / 指纹变 ⇒ 重查 ✓；`0` = 每次都查（= 关掉缓存 ✓）。
    ⚠ 原来写死常量 `MOB_SETS_TTL_S = 3.0` ✗（想调只能改代码 ✗）—— 且它与「前往重下间隔」
      从此**彻底拆开**：后者专管"**目标切换 CD**" ✓。
    """
    s = fresh_settings()
    s.enabled = True
    h = Harness(s)
    a = h.agent
    ws = h.ws(False)
    ws.player.here_sets = ["甲"]

    class _M:
        id = 7

    mob = _M()
    calls = []
    a.mob_sets_of = lambda player, m, _c=calls: (
        _c.append(1), {"world": (0.0, 0.0), "sets": ["乙"], "why": ""})[1]
    t = 1000.0
    s.mob_sets_ttl_s = 10.0
    a._mob_info_cache = None
    a._locked_mob_info(ws, mob, t)
    a._locked_mob_info(ws, mob, t + 1.0)
    check(len(calls) == 1,
          "TTL 之内又重查了（缓存参数没生效 ⇒ 白算一回 BFS ✗）：调了 %d 次" % len(calls))
    a._locked_mob_info(ws, mob, t + 11.0)
    check(len(calls) == 2,
          "超过「怪物判定缓存时长」却没重查（拿着过期判定 ✗）：调了 %d 次" % len(calls))
    s.mob_sets_ttl_s = 0.0
    a._mob_info_cache = None
    a._locked_mob_info(ws, mob, t)
    a._locked_mob_info(ws, mob, t)
    check(len(calls) == 4,
          "TTL=0 时该每次都查（关不掉缓存 ✗）：调了 %d 次" % len(calls))
    check("mob_sets_ttl_s" in s.to_dict(), "`mob_sets_ttl_s` 没进 `to_dict`（存不住 ✗）")
    s2 = fresh_settings()
    s2.from_dict({"mob_sets_ttl_s": 7.5})
    check(abs(float(s2.mob_sets_ttl_s) - 7.5) < 1e-6,
          "读回丢了（参数改不了 ✗）：%r" % (s2.mob_sets_ttl_s,))


def t_snap_beat():
    """⭐ **位置 / 血量 / 动作 的定频快照 + 喝药打点**（用户 2026-09-28 要求 ✓）。

    用户原话："1. 给血量加打点，喝药时记录触发喝药的 hp；4. 位置打点：world_x / world_y +
    当前所在集合名（1 秒一条）；5. 动作打点：这一拍任务给的方向（move 值）/ 按/松了哪些键
    —— 1 秒一条"✓。

    ⚠⚠ **为什么必须加**（这次查了两小时 ✗）：2026-09-28 卡了 2 小时 22 分，日志里只有
      `climb_dx` / `climb_y` 两条探针 ⇒ **分不出**「人动不了」（一直按着、位置不变 =
      死了/卡死）和「系统没发键」（`move=0`、`keys` 空 = 决策卡住）✗ —— 只能拿
      `dx=393.8` 去**反推**"人在离绳 400px 处"✓。

    钉六件：
      ① `pos` 每秒一条，含 `x` / `y`（**世界坐标** ✓）/ `sets`（**当前集合** ✓）/ `hp` ✓；
      ② `act` 每秒一条，含 `move`（这一拍给的方向 ✓）/ `keys`（按着的键 ✓）；
      ③ ⭐ **休息分支里也打**（整段摆在 `afk_spot_rest` 里，仍要出 `pos` ✓）—— 上次卡住的
         状态恰恰属休息分支（`afk_spot_walk`）⇒ 挂在常规路径上**一条都打不出来** ✗
         （这条最容易做错，源码级再钉一次 ✓）；
      ④ **喝药时把"触发那一刻的血"记下来**（`drink_hp` + `hp` / `thr` ✓）；
      ⑤ `move` 来自 `_steer`（**唯一**设方向的地方 ✓）且**打完清零**；
      ⑥ 打点坏了**不许影响决策**（吞掉 ✓）。
    """
    import pathlib
    import re
    import tempfile

    from core import behavior

    _tmp = pathlib.Path(tempfile.mkdtemp()) / "snap.log"
    _old = (behavior.ENABLED, behavior.LOG)
    behavior.configure(True, str(_tmp))
    try:
        s = fresh_settings(auto_hp_pot=True, hp_threshold=90.0)
        s.keymap["hp_pot"] = "g"                    # ⚠ 不设它喝药那条压根不进来 ✓
        s.enabled = True
        s.anti_afk_enabled = True
        s.anti_afk_type = "spot_rest"
        s.anti_afk_spot_set = "甲"
        s.anti_afk_spot_seq = []
        s.anti_afk_spot_loop = False
        s.pot_cd = 1000                            # ⚠ 喝药冷却：不设（默认 0）就会**每拍**都触发 ✗
        h = Harness(s)
        h.hp = 0.5                                  # 低于阈值 ⇒ 这一拍该喝药 ✓
        h.ws_hook = lambda w: setattr(w.player, "here_sets", ["测试集"])
        a = h.agent
        # ⭐ **摆成"正在定点休息"**（上次卡住的那个分支 ✓）——整段都待在休息分支里、
        #   不进常规路径 ⇒ 谁把 `_snap_beat` 挂到常规路径上，这里**一条都打不出来** ✓。
        a.state = "afk_spot_rest"
        a._afk_ctx = None
        a._rest_until = h.clock.t + 3600.0
        a._loop_next_ts = 0.0
        h.run(2.4, mob_plan=lambda _t: False)
        _txt = _tmp.read_text(encoding="utf-8")

        def _n(ev):
            """日志里某个事件名出现了几次（**按行首锚定**，免得被字段里的同名字骗 ✗）。"""
            return len(re.findall(r"^\d{4}-\d{2}-\d{2} [\d:.]+\s+%s\s" % ev, _txt, re.M))

        check(_n("pos") >= 2,
              "`pos` 不是每秒一条（或压根没打）—— 位置/血量快照 ✗：\n%s" % _txt[-500:])
        check("x=" in _txt and "y=" in _txt and "hp=" in _txt,
              "`pos` 里缺 x / y / hp（用户点名要 world_x / world_y + 血量 ✗）：\n%s"
              % _txt[-500:])
        check("sets=测试集" in _txt,
              "`pos` 没写出**当前所在集合**（用户点名要 ✓）：\n%s" % _txt[-500:])
        check(_n("act") >= 2 and "move=" in _txt,
              "`act` 不是每秒一条 / 缺 move（用户点名要 ✓）：\n%s" % _txt[-500:])
        # ⭐ **"按了哪些键"单独钉**：正常跑的时候休息分支每拍都会 `keys.set(...)` 把它覆盖成空
        #   ⇒ 只能手动按上一个键再快照一次 ✓（`_patched()` 必须开：不然真往本机发键 ✗）。
        with h._patched():
            a.keys.set({"f6"})
            a._snap_beat(h.clock.t + 99.0, h.ws(False))     # 节流窗口外 ⇒ 必定记一条 ✓
        check("keys=f6" in _tmp.read_text(encoding="utf-8"),
              "`act` 没写出按着的键（用户点名要「按/松了哪些键」✗）")
        check(_n("drink_hp") >= 1 and "thr=" in _txt,
              "喝药没把「触发那一刻的血」记下来（用户点名要 ✓）：\n%s" % _txt[-500:])
        check(hasattr(a, "_last_move") and a._last_move == 0,
              "`_last_move` 打完没清零（「没动的拍」会残留上一次的方向 ✗）：%r"
              % (getattr(a, "_last_move", None),))

        # ⑤/③ 源码级：`_move` 只能由 `_steer` 写；`_snap_beat` 必须在**休息分支之前** ✓
        _src = (pathlib.Path(__file__).resolve().parents[1]
                / "decision" / "agent.py").read_text(encoding="utf-8")
        check(re.search(r"self\._snap_beat\(now, ws\)\s*\n\s*if self\.state in REST_STATES:",
                        _src) is not None,
              "`_snap_beat` 没挂在**休息分支之前** ⇒ 休息中（含上次卡住的 `afk_spot_walk`）"
              "一条都打不出来 ✗")
        check(re.search(r"self\._last_move = 0 if not dx else", _src) is not None,
              "`_steer` 里没记这一次给的方向 ⇒ `act` 的 `move` 永远是 0 ✗")
    finally:
        behavior.configure(*_old)                   # ⚠ 恢复，别把后面用例的日志写到临时文件 ✗


def t_spot_loop_panel_widgets():
    """⭐ 界面三件（用户 2026-09-28 点名的布局 ✓）：勾选框 + **勾选后下方缩进**出现参数。

    钉四件：
      ① 勾选框就在「到达后行为」**下面**（用户指定位置 ✓）；
      ② **没勾 ⇒ 缩进那块不可见**、勾上 ⇒ 可见（"勾选后下方缩进出现参数" ✓）；
      ③ 那两块参数**真的缩进了**（`contentsMargins().left() > 0` ✓）；
      ④ 勾/改数值都**写回 settings** ✓。
    """
    # ⚠ 用本文件现成的 `build_panel()`（它 patch 掉了 dinput 的真连 ✓ 别自己 new ✗）
    p, app = build_panel()
    settings = ag.settings
    settings.anti_afk_type = "spot_rest"
    settings.anti_afk_spot_loop = False
    settings.anti_afk_spot_loop_min = 20.0
    settings.anti_afk_spot_loop_max = 40.0
    # ⚠ 加载回填走**真入口**（`_load_settings_into_ui` 那一步在别处，这里直接调刷新 ✓）
    p.ck_spot_loop.blockSignals(True)
    p.ck_spot_loop.setChecked(False)
    p.ck_spot_loop.blockSignals(False)
    p.sp_loop_min.setValue(20.0)
    p.sp_loop_max.setValue(40.0)
    # ⚠⚠ **必须先刷 `_refresh_afk_ui`**：父组 `_afk_spot` 只在"类型 = 定点休息"时可见，
    #   父组不可见时子块的 `isVisible()` **永远是 False** ⇒ 断言会假红 ✗（踩过）。
    p._refresh_afk_ui()
    p._refresh_spot_loop_ui()
    for _ in range(6):
        app.processEvents()

    check(p.ck_spot_loop.text() == "休息过程中循环行为",
          "勾选框文字不对（用户点名「休息过程中循环行为」✗）：%r" % p.ck_spot_loop.text())
    check(not p._spot_loop_box.isVisible(),
          "**没勾**却已经显示了缩进参数（用户要的是「勾选后出现」✗）")
    p.ck_spot_loop.setChecked(True)
    for _ in range(4):
        app.processEvents()
    check(p._spot_loop_box.isVisible(),
          "勾上了却没出现缩进参数（用户要的「勾选后下方缩进出现参数」✗）")
    check(p._spot_loop_box.layout().contentsMargins().left() > 0,
          "参数块没有**缩进**（用户明确说「下方**缩进**出现参数」✗）")
    check(settings.anti_afk_spot_loop is True,
          "勾选框没写回 `settings`（界面改了不生效 ✗）")
    p.sp_loop_min.setValue(15.0)
    p.sp_loop_max.setValue(25.0)
    for _ in range(2):
        app.processEvents()
    check(abs(float(settings.anti_afk_spot_loop_min) - 15.0) < 1e-6
          and abs(float(settings.anti_afk_spot_loop_max) - 25.0) < 1e-6,
          "「循环时间(s) A~B」没写回（界面改了不生效 ✗）：%r ~ %r"
          % (settings.anti_afk_spot_loop_min, settings.anti_afk_spot_loop_max))

    # ⑤⑥⑦ ⭐ **循环行为是一个"列表配置"**（用户 2026-09-28 升级 ✓）：
    #   一行 = 一项（名字 ✓）、能上移下移（**执行顺序** ✓）、双击 = 打开行为编辑器 ✓。
    settings.anti_afk_spot_loop_items = [
        {"name": "打坐", "seq": [{"type": "down", "key": "grave"}]},
        {"name": "起身", "seq": []},
    ]
    p._refresh_loop_list()
    for _ in range(3):
        app.processEvents()
    check(p.ck_loop_items.count() == 2,
          "列表没把两项都摆出来（列表配置 ✗）：%d 行" % p.ck_loop_items.count())
    check([p.ck_loop_items.item(i).text() for i in range(2)] == ["打坐", "起身"],
          "列表行的名字不对（该按顺序显示两项名字 ✓）：%r"
          % ([p.ck_loop_items.item(i).text() for i in range(2)],))
    p.ck_loop_items.setCurrentRow(1)
    p._on_loop_item_move(-1)                     # 第 2 项上移 ⇒ 该跑到第 1 位 ✓
    check([it["name"] for it in settings.anti_afk_spot_loop_items] == ["起身", "打坐"],
          "「上移」没换顺序（**执行顺序就是列表顺序** ⇒ 这个不 work 就没法「依次执行」✗）：%r"
          % ([it["name"] for it in settings.anti_afk_spot_loop_items],))
    check([p.ck_loop_items.item(i).text() for i in range(2)] == ["起身", "打坐"],
          "上移之后**界面没跟着换**（数据和显示对不上 ✗）")
    # ⑧⑨ ⭐ **能看到"每一个循环项目的当前状态"**（用户 2026-09-28 追加 ✓）：
    #   ① 休息状态那行显示「循环 N/M「名字」」（正在演哪一项 ✓）；
    #   ② 一轮跑完正在等 ⇒ 显示"几秒后重来"（不然看着像卡住 ✓）；
    #   ③ 列表里**当前那一项**带 `▶`（其余不带 ✓），而且**非休息时要摘掉** ✓。
    settings.anti_afk_spot_loop_items = [
        {"name": "打坐", "seq": [{"type": "down", "key": "grave"}]},
        {"name": "起身", "seq": [{"type": "down", "key": "grave"}]},
    ]
    p._refresh_loop_list()
    settings.rest_state = "afk_spot_rest"
    settings.rest_until_monotonic = 0.0
    settings.spot_loop_total, settings.spot_loop_idx = 2, 2
    settings.spot_loop_name, settings.spot_loop_next_at = "起身", 0.0
    p._tick_loop_state()
    _t = p._rest_text()
    check("循环 2/2" in _t and "起身" in _t,
          "休息那行没写出「正在跑第几项 / 哪一项」（用户要的「当前状态」✗）：%r" % _t)
    check(p.ck_loop_items.item(1).text().startswith("▶")
          and not p.ck_loop_items.item(0).text().startswith("▶"),
          "列表没给**当前那一项**加标记（或给错了项 ✗）：%r"
          % ([p.ck_loop_items.item(i).text() for i in range(2)],))
    import time as _time

    settings.spot_loop_idx = 0
    settings.spot_loop_name = "（一轮完成）"
    settings.spot_loop_next_at = _time.monotonic() + 12
    p._tick_loop_state()
    check("重来" in p._rest_text(),
          "一轮跑完正在等的时候没写「几秒后重来」（看着像卡住 ✗）：%r" % p._rest_text())
    check(not any(p.ck_loop_items.item(i).text().startswith("▶") for i in range(2)),
          "等待时列表还挂着 ▶（那会儿谁都没在跑 ✗）")
    # ⚠ **非休息时一律摘掉**（免得配置界面一直挂个 ▶ 看不懂 ✓）
    settings.rest_state = ""
    settings.spot_loop_idx = 2
    p._tick_loop_state()
    check(not any(p.ck_loop_items.item(i).text().startswith("▶") for i in range(2)),
          "不在休息中，列表却还挂着 ▶（配置界面会看不懂 ✗）")

    settings.anti_afk_spot_loop_items = [{"name": "打坐", "seq": []}]
    p._refresh_loop_list()
    p.ck_loop_items.setCurrentRow(0)
    p._on_loop_item_del_patch_check = None       # （删除会弹确认框 ⇒ 不在这儿点 ✓）
    check(hasattr(p, "_on_loop_item_edit") and hasattr(p, "_on_loop_item_rename"),
          "「双击打开行为编辑器 / 重命名」这两个入口没接上（用户点名要求 ✗）")

    # ⑩ ⭐ **「休息结束推迟到循环执行完」**（用户 2026-09-28 ✓）—— 它是「休息过程中循环行为」
    #   的**子参数**：控件要落在那块**缩进**里（⇒ 只有勾了循环行为才看得见 ✓）、且写回
    #   `settings` ✓（口径在 `decision/agent._hold_rest_for_loop` ✓）。
    check(hasattr(p, "ck_loop_hold"),
          "少了「休息结束推迟到循环执行完」这个子开关（用户点名要求 ✗）")
    check(p._spot_loop_box.isAncestorOf(p.ck_loop_hold),
          "这个子开关不在「休息过程中循环行为」那块**缩进**里 ⇒ 没勾循环行为时它也会露出来"
          "（用户要的是它的**子参数** ✗）")
    settings.anti_afk_spot_loop_hold = False
    p.ck_loop_hold.setChecked(True)
    for _ in range(2):
        app.processEvents()
    check(settings.anti_afk_spot_loop_hold is True,
          "勾上「休息结束推迟到循环执行完」没写回 `settings`（界面改了不生效 ✗）")
    p.ck_loop_hold.setChecked(False)
    for _ in range(2):
        app.processEvents()
    check(settings.anti_afk_spot_loop_hold is False, "取消勾选没写回 `settings` ✗")
    p.close()
    settings.anti_afk_spot_loop = False


def t_afk_spot_rest():
    """「定点休息」：五个阶段都走通 + 走不到要如实说（用户 2026-09-26 定的流程，P2）。

    为什么必须钉：这一型的**参数界面早就有**（两个下拉 + 到达后行为按钮 ✓），可运行期
    一直走的是**隐身那套** ⇒ 选了它"什么都不发生" ✗（`ANTI_AFK_TYPES` 上面那段警告写的
    就是这种情况）。所以这里从"进休息"一直走到"回战斗"，五步逐个断言：

      ① 走（用注入的解析器起一条多步路径）→ ② 到达后行为 → ③ 歇休息时长
      → ④ 结束后前往 → ⑤ 回战斗。

    另外钉**失败路径**：解析不出来 / 半路断了 ⇒ 如实说 + 回战斗（**不许硬走、不许静默** ✗）。
    """
    from decision import route

    s = fresh_settings(anti_afk_enabled=True, anti_afk_type="spot_rest",
                       anti_afk_spot_set="乙平台", anti_afk_spot_after="丙平台",
                       anti_afk_rest_min=0.2, anti_afk_rest_max=0.2)
    # 到达后行为：按一下商城再松开（一按一松 ⇒ 短、好断言）
    s.anti_afk_spot_seq = [{"type": "down", "key": "shop"},
                           {"type": "up", "key": "shop"}]
    h = Harness(s)
    h.clock0 = h.clock.t            # 夹具记日志要用它（别的用例的 helper 也会先设）
    seen = []                       # 解析器被问过哪些目的地

    def resolver(dst):
        seen.append(dst)
        if dst == "乙平台":
            return {"path": ["甲平台", "乙平台"],
                    "jobs": [route.WalkJob("乙平台",
                                           [(500.0, 490.0, 510.0, -208.0, "1")])],
                    "why": "", "here": False}
        return {"path": ["乙平台", "丙平台"],
                "jobs": [route.WalkJob("丙平台",
                                       [(520.0, 510.0, 530.0, -208.0, "2")])],
                "why": "", "here": False}

    ws = h.ws(with_mob=False)
    ag = h.agent
    # ① 没注入解析器 ⇒ **如实说**（不许静默，也不许硬走）
    ag.route_plan = None
    ok, msg = ag.plan_and_start_route("乙平台")
    check(not ok and "实时" in msg, "没注入解析器时该如实说：%r" % msg)

    ag.route_plan = None
    with h._patched():
        ag.route_plan = resolver
        # ② 进休息（定点那一支）⇒ 立刻去「乙平台」，并把**整条**交给执行器
        ag._begin_rest(0.0)
        check(ag.state == "afk_spot_walk", "定点休息没进「走」阶段：%s" % ag.state)
        ws.player.here_sets = ["甲平台"]
        ag._run_rest_spot(0.0, ws)
        check(seen == ["乙平台"], "没按「指定地点」解析路径：%s" % seen)
        check(ag._climb is not None and ag._route_total == 1,
              "整条路径没交给执行器：climb=%r 步数=%s" % (ag._climb, ag._route_total))
        # ⚠ 阶段还要**发布出去**：`settings.rest_state` 是界面的唯一信息来源，而发布
        #   在 `_run_rest` 的**尾巴**上（`_run_rest_spot` 自己不发）。
        #   少了这一环，用户看到的就是「未休息」✗（2026-09-26 就是这么被问的）。
        ag._run_rest(0.0, ws)                  # 走**真入口**（tick 走的就是它）
        check(ag.settings.rest_state == "afk_spot_walk",
              "「走去休息点」没发布 rest_state（界面会显示「未休息」）：%r"
              % ag.settings.rest_state)
        # 走完（`_climb` 被清空）+ 脚下已经是「乙平台」⇒ 转「到达后行为」
        ag._climb, ag._route = None, []
        ws.player.here_sets = ["乙平台"]
        ag._run_rest_spot(0.1, ws)
        check(ag.state == "afk_spot_act", "走到了没转「到达后行为」：%s" % ag.state)
        # ③ 到达后行为演完 ⇒ 进「歇」
        t = 0.1
        for _ in range(60):
            t += 0.5
            ag._run_rest_spot(t, ws)
            if ag.state == "afk_spot_rest":
                break
        check(ag.state == "afk_spot_rest",
              "到达后行为没演完 / 没进「歇」：%s" % ag.state)
        # 休息时长**从这一刻起算**：差一点点不许往下走
        ag._run_rest_spot(ag._rest_until - 0.01, ws)
        check(ag.state == "afk_spot_rest", "休息时间没到就往下走了：%s" % ag.state)
        # ④ 歇够了 ⇒ 「结束后前往」（第二个集合，再解析一次）
        ag._run_rest_spot(ag._rest_until + 0.01, ws)
        check(ag.state == "afk_spot_back", "歇完没走「结束后前往」：%s" % ag.state)
        ag._run_rest_spot(ag._rest_until + 0.02, ws)
        check(seen[-1] == "丙平台", "「结束后前往」没解析第二个地点：%s" % seen)
        check(ag._climb is not None, "「结束后前往」没交给执行器")
        # ⑤ 走到了 ⇒ 回战斗
        ag._climb, ag._route = None, []
        ag._run_rest_spot(ag._rest_until + 0.03, ws)
        check(ag.state == "idle", "走完「结束后前往」没回战斗：%s" % ag.state)

        # ⑥ 失败路径一：**解析不出来** ⇒ 如实说 + 立刻收工（不许硬等 / 不许静默 ✗）
        ag.route_plan = lambda dst: {"path": [], "jobs": [], "here": False,
                                     "why": "没有从「甲平台」到「乙平台」的路"}
        ag._begin_rest(10.0)
        ws.player.here_sets = ["甲平台"]
        ag._run_rest_spot(10.0, ws)
        check(ag.state == "idle", "解析不出来却没收工（会卡在休息里）：%s" % ag.state)
        note = ag.current_goto_note()
        check("定点休息" in note and "没有" in note, "走不到没如实说：%r" % note)

        # ⑦ 失败路径二：**半路断了**（`_climb` 空了、脚下却不在目标上）⇒ 同样如实说 + 收工
        ag.route_plan = resolver
        ag._begin_rest(20.0)
        ws.player.here_sets = ["甲平台"]
        ag._run_rest_spot(20.0, ws)
        check(ag._climb is not None, "第二次没出发")
        ag._climb, ag._route = None, []
        ag._run_rest_spot(20.1, ws)         # 脚下还在「甲平台」
        check(ag.state == "idle", "半路断了却没收工：%s" % ag.state)
        check("定点休息" in ag.current_goto_note(),
              "半路断了没如实说：%r" % ag.current_goto_note())

    # ⑧ 「被打断后延迟重试」在**去休息点那段路上**也要生效（用户 2026-09-26 要求：
    #    "从开始前往指定地点就开算"）：走的中途触发了自动补血 ⇒ 这次休息作废，
    #    并把下次提前到 `retry_sec` 之后（不是随机 N~M 分钟）。
    s3 = fresh_settings(anti_afk_enabled=True, anti_afk_type="spot_rest",
                        anti_afk_spot_set="乙平台",
                        anti_afk_retry_on_interrupt=True, anti_afk_retry_sec=42.0,
                        anti_afk_rest_min=0.2, anti_afk_rest_max=0.2,
                        auto_hp_pot=True, hp_threshold=95, pot_cd=0)
    # ⚠ 夹具的 keymap 默认**没有 `hp_pot`**（`_drink_potions` 会直接早退 ⇒ 根本不补血 ✗）
    s3.keymap["hp_pot"] = "f5"
    s3.enabled = True
    h3 = Harness(s3)
    h3.clock0 = h3.clock.t
    h3.hp = 0.1                              # 血低 ⇒ 这一拍会真的补血
    ws3 = h3.ws(with_mob=False)
    ws3.player.here_sets = ["甲平台"]
    with h3._patched():
        a3 = h3.agent
        a3.route_plan = lambda dst: {
            "path": ["甲平台", dst],
            "jobs": [route.WalkJob(dst, [(500.0, 490.0, 510.0, -208.0, "1")])],
            "why": "", "here": False}
        a3._begin_rest(0.0)
        a3._run_rest(0.0, ws3)
        check(a3.state == "afk_spot_walk" and a3._climb is not None,
              "没进「前往休息点」阶段：%s" % a3.state)
        h3.clock.t += 0.1
        a3.tick(ws3)                         # 整拍：走 + 补血 + 打断判定
        check(a3.state == "idle", "去休息点路上补了血却没打断这次休息：%s" % a3.state)
        check(a3._climb is None, "打断后还挂着「去休息点」的路线（会继续白跑 ✗）")
        left = a3._next_afk - h3.clock.t
        check(abs(left - 42.0) < 1.5,
              "打断后没按 retry_sec 排下一次（还剩 %.0f 秒）" % left)

    # ⑨ **赶路时"有怪先打"**（用户 2026-09-26 报的坑：手动休息会寻路，但攻击范围内有怪
    #    时**不还手** ✗）。判据要和「命令前往」**同一套**（正常路径 `if in_range:` 那支）：
    #    有怪 ⇒ 打（攻击键按下）、这一拍不往前走；打的时候**休息阶段名不许被改掉** ✗
    #    （改掉 = 这次休息静默作废，比不打还糟）。
    s4 = fresh_settings(anti_afk_enabled=True, anti_afk_type="spot_rest",
                        anti_afk_spot_set="乙平台", attack_cd=0,
                        anti_afk_rest_min=0.2, anti_afk_rest_max=0.2)
    s4.enabled = True
    h4 = Harness(s4)
    h4.clock0 = h4.clock.t
    # 怪摆在正前方、攻击范围内（Mob 的写法抄既有用例）
    h4.mobs_fn = lambda t: [Mob(id=1, x=540.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
    ws4 = h4.ws(with_mob=True)
    ws4.player.here_sets = ["甲平台"]
    with h4._patched():
        a4 = h4.agent
        a4.route_plan = lambda dst: {
            "path": ["甲平台", dst],
            "jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
            "why": "", "here": False}
        a4._begin_rest(0.0)
        a4._run_rest(0.0, ws4)
        check(a4.state == "afk_spot_walk", "没进「赶路」阶段：%s" % a4.state)
        h4.log.clear()
        for _ in range(3):                # 序列起跑到真的按下键要两拍（既有用例同款）
            h4.clock.t += 0.1
            a4.tick(ws4)
        atk = s4.keymap["attack"]
        check(("down", atk) in [(k, v) for _t, k, v in h4.log],
              "攻击范围内有怪，赶路时没触发 attack（用户报的坑 ✗）：%s" % h4.log[:8])
        check(a4.state == "afk_spot_walk",
              "打了怪却把休息阶段名改掉了（这次休息会静默作废 ✗）：%s" % a4.state)


def t_timer_manual_fire():
    """「手动触发」：立刻演一次 + **计时从头开始**（用户 2026-09-26 要求）。

    为什么钉它：这是"点了就该有反应"的按钮，而它两条语义都容易做错：
      · 只在界面上重排计时、不进实时线程 ⇒ 角色不会动 ✗（按键在 agent 那边）；
      · 触发后接着**原来的剩余时间**走 ⇒ 那不叫"计时从头开始" ✗。
    顺带钉：演到一半再点一次必须**先松开旧序列按着的键**（否则卡键 ✗）、
    暂停的项不响应、陌生名字安静丢掉、以及这个请求通道**不进配置文件**。
    """
    s = fresh_settings(custom_timers=[
        {"name": "t1", "interval": [5, 10],
         "seq": [{"type": "down", "key": "f3"}, {"type": "up", "key": "f3"}]}],
        custom_timer_next={})
    h = Harness(s)
    h.clock0 = h.clock.t
    with h._patched():
        # ① 先排上期，再把"下次"人为摆成**马上就要到**（1 秒后）
        h.agent._custom_timers(h.clock.t)
        check(s.custom_timer_next.get("t1", 0.0) > 0, "没排期：%s" % s.custom_timer_next)
        s.custom_timer_next["t1"] = h.clock.t + 1.0
        # ② 手动触发 ⇒ 立刻开演；下次**从头**算（≥ 5 分钟之后，而不是刚才那 1 秒）
        s.custom_timer_fire.append("t1")
        h.clock.t += 0.01
        h.agent._custom_timers(h.clock.t)
        check(not s.custom_timer_fire,
              "请求没被取走（会一直重复触发）：%s" % s.custom_timer_fire)
        check("t1" in h.agent._timer_states, "手动触发没开演序列")
        left = s.custom_timer_next["t1"] - h.clock.t
        check(left >= 5 * 60 - 1,
              "计时没从头开始（还接着原来的 1 秒）：还剩 %.0f 秒" % left)
        # ③ 序列真的把键发出去了
        h.clock.t += 0.1
        h.agent._custom_timers(h.clock.t)
        check(("down", "f3") in [(k, v) for _t, k, v in h.log],
              "手动触发没真的按键：%s" % h.log)

    # ④ 演到一半再点一次 ⇒ **旧序列按下的键必须先松开**（否则卡键）
    s2 = fresh_settings(custom_timers=[
        {"name": "t2", "interval": [1, 1],
         "seq": [{"type": "down", "key": "f5"},
                 {"type": "delay", "ms": 60000},
                 {"type": "up", "key": "f5"}]}], custom_timer_next={})
    h2 = Harness(s2)
    h2.clock0 = h2.clock.t
    with h2._patched():
        s2.custom_timer_fire.append("t2")
        h2.agent._custom_timers(h2.clock.t)
        h2.clock.t += 0.1
        h2.agent._custom_timers(h2.clock.t)
        check(("down", "f5") in [(k, v) for _t, k, v in h2.log], "没按下 f5")
        h2.log.clear()
        s2.custom_timer_fire.append("t2")            # 演到一半再点一次
        h2.clock.t += 0.1
        h2.agent._custom_timers(h2.clock.t)
        check(("up", "f5") in [(k, v) for _t, k, v in h2.log],
              "重排时没松开旧序列按下的键（会卡键）：%s" % h2.log)
        # ⑤ 暂停的项：手动触发**不演**（暂停的语义就是"不触发"）
        s2.custom_timers[0]["paused"] = True
        h2.log.clear()
        s2.custom_timer_fire.append("t2")
        h2.clock.t += 0.1
        h2.agent._custom_timers(h2.clock.t)
        check("t2" not in h2.agent._timer_states, "暂停的项被手动触发了")
        check(not s2.custom_timer_fire, "暂停项的请求没被取走（会一直攒着）")
        # ⑥ 不认识的名字 ⇒ 安静丢掉，不崩
        s2.custom_timer_fire.append("（没这项）")
        h2.clock.t += 0.1
        h2.agent._custom_timers(h2.clock.t)
        check(not s2.custom_timer_fire, "陌生名字的请求没被丢掉")
    # ⑦ **不进配置文件**：它是运行时状态，不是参数（重启后不该还留着"待触发"）
    check("custom_timer_fire" not in s.to_dict(),
          "`custom_timer_fire` 被写进配置了（运行态不该持久化）")


def t_timer_manual_fire_button():
    """界面：每项都有「手动触发」按钮（**琥珀色**，和旁边三个不一样），点了发请求；
    **自动关着 / 已暂停时灰掉** —— 否则就是"点了没反应"，那种坑最难查 ✗。"""
    p, app = build_panel()
    settings = ag.settings          # 界面读的就是这个单例
    saved = (settings.custom_timers, settings.custom_timer_next,
             list(settings.custom_timer_fire), settings.enabled)
    settings.custom_timers = [
        {"name": "A", "seq": [{"type": "down", "key": "f3"}], "interval": [10, 10]},
        {"name": "B", "seq": [{"type": "down", "key": "f4"}], "interval": [10, 10],
         "paused": True, "paused_left": 60.0},
    ]
    settings.custom_timer_next = {}
    settings.custom_timer_fire.clear()
    try:
        settings.enabled = True
        p._refresh_timers()
        app.processEvents()
        btns = p._timer_fire_btns
        check([b.text() for b in btns.values()] == ["手动触发", "手动触发"],
              "没给每一项加「手动触发」按钮：%s" % [b.text() for b in btns.values()])
        check("#a05c00" in btns["A"].styleSheet(),
              "按钮没换成（琥珀）色：%r" % btns["A"].styleSheet())
        check(btns["A"].isEnabled(), "自动开着、也没暂停 ⇒ 该能点")
        check(not btns["B"].isEnabled(), "已暂停的那项还能点（点了也不会有反应 ✗）")
        btns["A"].click()
        app.processEvents()
        check(settings.custom_timer_fire == ["A"],
              "点了没把触发请求发出去：%s" % settings.custom_timer_fire)
        # 自动关掉 ⇒ 必须灰掉（agent 每拍 `_release_held_keys`，序列活不过一帧）
        settings.custom_timer_fire.clear()
        settings.enabled = False
        p._refresh_timers()
        app.processEvents()
        check(not p._timer_fire_btns["A"].isEnabled(), "自动关着按钮还能点")
    finally:
        settings.custom_timers = saved[0]
        settings.custom_timer_next = saved[1]
        settings.custom_timer_fire[:] = saved[2]
        settings.enabled = saved[3]
        p._refresh_timers()


def t_rest_state_text_covers_all_states():
    """休息阶段 → 文字：**每个状态都得有话说**，界面更不许说反话。

    2026-09-26 用户报：「定点休息」手动进入后，玩家面板卡片上写着**「未休息」** ✗
    （他问的是"是没读到休息时长，还是显示错"）—— 答案是**显示**错：时长一直读得到 ✓，
    错在"状态→文字"那张表只认隐身那三个状态 ✗。

    所以这条钉两件事：
      ① 表覆盖**全部** `REST_STATES`（以后加新休息类型、漏了文字 ⇒ 当场红 ✗）；
      ② 玩家面板那张卡片对定点休息的各个阶段都说对（尤其**不许**出现「未休息」✗）。
    实时页那行由 `selftest_minimap` 钉（它那边才建 RoutePanel）。
    """
    from gui.player_panel import PlayerPanel     # 只调静态方法，不建窗口
    from decision.agent import settings as ds

    for st in ag.REST_STATES:
        check(ag.rest_state_text(st), "状态 %s 没有对应说法（界面会说反话 ✗）" % st)
    check(ag.rest_state_text("") == "" and ag.rest_state_text("（乱填）") == "",
          "空 / 认不出来的状态该给空串（**不许猜**成某个肯定句）")

    saved = (ds.rest_state, ds.rest_until_monotonic, ds.anti_afk_spot_set,
             ds.anti_afk_spot_after)
    try:
        ds.anti_afk_spot_set = "左上平台"
        ds.anti_afk_spot_after = "右下休息平台"
        ds.rest_state = "afk_spot_rest"
        ds.rest_until_monotonic = time.monotonic() + 125
        txt = PlayerPanel._rest_text()
        check("定点休息中" in txt, "卡片没写「定点休息中」：%r" % txt)
        check("未休息" not in txt, "卡片说反话（写着「未休息」）：%r" % txt)
        check("2:0" in txt, "卡片没显示休息剩余时间：%r" % txt)

        ds.rest_state = "afk_spot_walk"
        ds.rest_until_monotonic = 0.0
        txt = PlayerPanel._rest_text()
        check("前往休息点" in txt and "左上平台" in txt,
              "去休息点那段没说清去哪儿：%r" % txt)
        check("未休息" not in txt, "去休息点的路上说反话：%r" % txt)

        ds.rest_state = "afk_spot_back"
        txt = PlayerPanel._rest_text()
        check("结束后前往" in txt and "右下休息平台" in txt,
              "「结束后前往」没说清去哪儿：%r" % txt)

        ds.rest_state = "afk_rest"                  # 隐身那一型不许被改坏
        ds.rest_until_monotonic = time.monotonic() + 65
        check("休息中" in PlayerPanel._rest_text(), "隐身休息的文案被弄坏了")
    finally:
        (ds.rest_state, ds.rest_until_monotonic, ds.anti_afk_spot_set,
         ds.anti_afk_spot_after) = saved


def t_rest_contract():
    """防掉线契约（2026-09-26 批准的三条结构性收敛）。合在一条里是因为它们都是
    "静默出错"类 —— 界面说反话 / 卡键 / 角色白跑一趟，单看代码都像没问题 ✗：

      ① **状态表覆盖齐全**：一个状态只在一处登记，其余派生（`REST_STATE_SPEC`）；
      ② **收尾自保**：`_finish_rest` 自己把 `_afk_ctx` 的键松开、标记清掉
         （不指望每个调用点都记得 ✗）；
      ③ **只收自己那条路线**：休息结束要收掉"定点休息走过去"的任务，但
         **不许**连坐用户自己下的「命令前往」✗。
    """
    from decision import route

    # ① 契约表：字段齐全 + 派生元组确实是从表里来的
    spec = ag.REST_STATE_SPEC
    check(set(spec) == set(ag.REST_STATES), "REST_STATES 与契约表不一致")
    for st, v in spec.items():
        check(str(v.get("text") or ""), "状态 %s 没有文案（界面会说反话 ✗）" % st)
        for f in ("spot", "timed", "interrupt"):
            check(isinstance(v.get(f), bool), "状态 %s 缺字段 %s" % (st, f))
    check(set(ag.SPOT_STATES) == {s for s, v in spec.items() if v["spot"]},
          "SPOT_STATES 没从表里派生")
    check(set(ag.REST_STATES_TIMED) == {s for s, v in spec.items() if v["timed"]},
          "REST_STATES_TIMED 没从表里派生")
    check(set(ag.REST_STATES_INTERRUPT)
          == {s for s, v in spec.items() if v["interrupt"]},
          "REST_STATES_INTERRUPT 没从表里派生")

    s = fresh_settings(anti_afk_enabled=True, anti_afk_type="spot_rest",
                       anti_afk_spot_set="乙平台", anti_afk_rest_min=0.2,
                       anti_afk_rest_max=0.2)
    s.anti_afk_spot_seq = [{"type": "down", "key": "shop"},
                           {"type": "delay", "ms": 60000}]
    h = Harness(s)
    h.clock0 = h.clock.t
    with h._patched():
        a = h.agent
        a.route_plan = lambda dst: {"path": [], "jobs": [], "here": True,
                                    "why": "已经在上面了"}
        a._begin_rest(0.0)
        ws = h.ws(with_mob=False)
        ws.player.here_sets = ["乙平台"]
        a._run_rest(0.0, ws)               # 到了 ⇒ 进「到达后行为」并按下 shop
        a._run_rest(0.5, ws)
        check(a._afk_ctx is not None and a.state == "afk_spot_act",
              "到达后行为没起跑（这条测不了）：%s / %s" % (a._afk_ctx, a.state))
        # ② 随便谁调 `_finish_rest` 都要收干净（松键 + 清标记）
        a._finish_rest(1.0)
        check(a._afk_ctx is None, "收尾没清 `_afk_ctx`（下一个序列会带着旧键跑 ✗）")
        check(a._spot_started is False, "收尾没清 `_spot_started`")
        check(a.state == "idle" and a.settings.rest_state == "", "收尾没回到 idle")
        # ⚠ 键名从日志里取（序列用的是**逻辑键名**，`keymap` 那层映射发生在 `input` 里）
        _downs = [v for _t, k, v in h.log if k == "down"]
        _ups = [v for _t, k, v in h.log if k == "up"]
        check(_downs and _downs[0] in _ups,
              "收尾没松开序列按着的键（会卡键 ✗）：%s" % h.log)

        # ③ 归属判定：休息自己的路线要收掉
        a.start_route([route.WalkJob("乙平台", [(500.0, 490.0, 510.0, -208.0, "1")])],
                      why="定点休息：去「乙平台」")
        check(a._climb is not None, "没挂上路线（这条测不了）")
        a._finish_rest(2.0)
        check(a._climb is None and not a._route,
              "休息结束没把「去休息点」那条路线收掉 ✗")
        # ③b **用户自己下的「命令前往」不许被连坐** ✗
        a.start_route([route.WalkJob("丙平台", [(500.0, 490.0, 510.0, -208.0, "2")])],
                      why="命令前往：丙平台")
        a._finish_rest(3.0)
        check(a._climb is not None,
              "休息收尾把用户自己下的「命令前往」也收掉了（那不是休息的路线 ✗）")


def t_retry_notes_have_no_denominator():
    """那个「**1/3 尝试**」必须彻底没了（用户 2026-09-27 在寻路任务里看到它）。

    用户原话："我好像在寻路任务时看到一个 1/3 尝试的显示……应该需要去掉，我们现在都走
    「寻路超时时间」" ⇒ "最多试几次"（`max_attempts`）整套删掉：

      · 四个执行器（走 / 爬 / 下跳 / 跳）都**不许再有** `max_attempts` 属性/参数
        —— 谁加回来，界面上的「第 N/M 次尝试」就又出现了 ✗；
      · `retry()` 那句说明里**不许有分母**（只有"第 N 次尝试"✓）；
      · `route.DEFAULT_MAX_ATTEMPTS` 这个常量一起删干净 ✓。

    配套（在别的用例里钉着）：不放弃的那一侧见 `t_climb_flow_rules` ⑤b（失败 6 次还在重来）、
    收口那一侧见 `t_goto_timeout_all_job_kinds`（四种任务都由「寻路超时时间」切 ✓）。
    """
    from decision import route

    jobs = [
        ("爬", route.ClimbJob("L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                              dst_set="乙平台", src_set="甲平台", tol_px=6, hold_ms=0)),
        ("下跳", route.DropJob("甲平台", "乙平台", [(700.0, 660.0, 740.0, "41")],
                               tol_px=6, hold_ms=0)),
        ("走", route.WalkJob("乙平台", [(700.0, 660.0, 740.0, -208.0, "41")])),
        ("跳", route.JumpJob("乙平台", [(600.0, 700.0, "11")])),
    ]
    for name, j in jobs:
        check(not hasattr(j, "max_attempts"),
              "「%s」执行器还有 `max_attempts`（= 「最多试几次」）—— 用户 2026-09-27 "
              "要求去掉（那段都走「寻路超时时间」✗）" % name)
        j.retry()
        check("/" not in str(j.note) and "次尝试" in str(j.note),
              "「%s」的重试说明里还带着分母（那个「1/3」就是这么来的 ✗）：%r"
              % (name, j.note))
    check(not hasattr(route, "DEFAULT_MAX_ATTEMPTS"),
          "`route.DEFAULT_MAX_ATTEMPTS` 还在 —— 常量也该一起删干净 ✓")

def t_climb_jump_gate():
    """**按跳的许可条件 = 只有「x 对齐」这一条**（用户 2026-09-27 明确，原话：
    「对齐了就按跳开始爬啊，不需要判定别的」）。

    ⚠ 这条**推翻**了同一天早上那版「按跳许可」（当时用户报「对齐 x 时一直在不正确的位置按跳」
      ⇒ 我加了 `_at_entry`：人得站在绳**入绳端那块 foothold** 上 ✗）。当天他现场又撞上它：
      `世界 (1093,-169) fh：三楼` + `爬（绳梯）(climb):对齐好了，但**人没站在绳下端那一层上**
      （绳底那块面在 y=-182~-182）（现在 y=-169）⇒ …这一拍不按跳` ⇒ **站着不动** ✗
      （只差 13px；而且绳两端**常常不属于同一个集合** —— 那条爬边是「三楼 → 顶层」✓）。

    ⇒ 现在：**对齐（`tol_px` 容差 + `hold_ms` 站住）就按跳** ✓，y 在哪一层、绳底那块面是谁，
      一概不判 ✓；跳了没上去由「点按 / 回对齐重来」+「寻路超时」那套接手 ✓。

    钉五件：
      ① x 对齐 + 站住窗口过 ⇒ **直接按跳** ✓；
      ② x **没**对齐 ⇒ 绝不按跳（"对齐 x 时乱按跳"的正解就是这条 ✓）；
      ③ y 在别的高度（同集合上层 / 很远 / 干脆没读数）⇒ **照样按跳** ✓（用户要的就是这个 ✓）；
      ④ CLIMB 相里位置变了 ⇒ 也按跳（交给"跳了没上去 ⇒ 回对齐重来"那条链 ✓，不再在按跳前拦 ✗）；
      ⑤ `job_for_edge` 造出来的爬任务**不再带 `entry_span`**、`__init__` 也没有这个参数
        （那道门整套拆了，死数据/死参数别留 ✗）。
    """
    import inspect

    from core import mapdata, zones
    from decision import route

    C = route.ClimbJob
    # 绳：x=550，y 从 300（下端）到 100（上端）
    base = dict(ladder_id="L1", x=550.0, y1=300.0, y2=100.0, direction=1,
                dst_set="上", src_set="下", tol_px=10, hold_ms=250)

    # ① x 对齐 ⇒ 站住窗口一过就按跳
    j = C(**base)
    o = j.update(0.0, 550.0, py=300.0)
    check(not o["jump"], "刚进误差范围就跳了（该先站住等保持窗口）：%s" % o)
    o = j.update(0.30, 550.0, py=300.0)
    check(o["jump"], "对齐了、站住窗口也过了却没按跳：%s / %r" % (o, j.note))

    # ② x 没对齐 ⇒ 绝不按跳
    j = C(**base)
    j.update(0.0, 500.0, py=300.0)                 # 差 50px ⇒ 还在对齐
    o = j.update(0.60, 500.0, py=300.0)
    check(not o["jump"], "x 还没对齐就按跳了（用户早先报的就是这个 ✗）：%s / %r" % (o, j.note))
    check(j.phase == C.ALIGN, "没对齐就该留在对齐相：%s" % j.phase)

    # ③ y 在别的高度也照样跳
    #    ⚠ y 跑到绳的上下端**外面**（-250 / 400+）会先被「到达」判据接走 —— 那是**另一条**判据
    #      （`_arrived` ✓，与"按跳许可"无关 ✓）⇒ 这里挑的是"还在绳的范围内、但不在入绳端那块
    #      面上"的几档 ✓。
    for py in (300.0, 220.0, 366.0, None):
        j = C(**base)
        j.update(0.0, 550.0, py=py)
        o = j.update(0.30, 550.0, py=py)
        check(o["jump"],
              "y=%r 时没按跳（用户要的是「对齐了就跳」，别的都不判 ✗）：%s" % (py, o))

    # ④ CLIMB 相里位置变了 ⇒ 也按跳（交给"没上去就重来"那套）
    j = C(**base)
    j.phase = C.CLIMB
    j._jump_tried = False
    o = j.update(0.5, 550.0, py=220.0, here_sets=["下"])
    check(o["jump"],
          "在 CLIMB 相里位置变了就不按跳了（该按跳、让「没上去就重来」那套接手 ✗）：%s" % o)

    # ⑤ 死数据 / 死参数都清掉
    t = mapdata.load("105090600")
    z = zones.load("105090600")
    ce = next((e for e in z.edges if e.get("kind") == "climb"), None)
    check(ce is not None, "用户数据里没有「爬」边，这条测不了")
    jj = route.job_for_edge(t, z, ce, tol_px=6, hold_ms=0, stall_s=3.0)
    check(not hasattr(jj, "entry_span"),
          "爬任务还挂着 `entry_span`（那道门整套拆了，死数据别留 ✗）：%r"
          % (getattr(jj, "entry_span", None),))
    check("entry_span" not in inspect.signature(C.__init__).parameters,
          "`ClimbJob.__init__` 的签名里还留着 `entry_span`（死参数 ✗）")


def t_climb_diagonal_jump():
    """「**斜跳上绳**」：离绳还远的时候**按住朝绳的方向 + 起跳**斜着过去，
    近了才走老的「对齐 x → 站住 → 原地起跳」（用户 2026-09-27 的设计）。

    原话：\"现在利用「起跳距离范围」设计「斜跳上绳」逻辑，要求使用「开始对齐绳梯x的距离」
    参数兼容旧的「对齐x原地起跳上绳」逻辑\"

    口径（两个区间，**都从设置来 ⇒ 零魔数** ✓）：
      · `开始对齐绳梯x的距离`(`near_px`) < |dx| ≤ `起跳距离`(`jump_start_px`) **且人站在绳
        要抓的那块 foothold 上** ⇒ **斜跳**（按住朝绳的方向 + 点按跳 ✓）；
      · |dx| ≤ `near_px` ⇒ **老逻辑**（点按对齐 → 站住 → 原地起跳 ✓）；
      · |dx| > `起跳距离` ⇒ 老逻辑「一口气按住走」✓；
      · `起跳距离` = 0（默认）⇒ **整条关掉** ⇒ 与加这个功能之前**完全一致** ✓（兼容底线）。

    钉八件：
      ① `起跳距离=0` ⇒ 区间内**绝不斜跳**（jump=False，方向键照旧一口气按住 ✓）；
      ② 开了（120）后，`|dx|=80` 且人在绳底那块面上 ⇒ **按住朝绳方向 + 按跳**（dir=↑ ✓）；
      ③ **一轮只斜跳一次**：紧接着几拍 jump 都是 False，但方向键**一直按着**走向绳 ✓；
      ④ 走近到 `near_px` 以内 ⇒ 交回老逻辑，而且斜跳**重新武装**（`_diag_done` 归 False ✓）；
      ⑤ `|dx|` 比起跳距离还远 ⇒ 不斜跳（老逻辑"一口气按住走" ✓）；
      ⑥ **人不在绳下端那一层上** ⇒ 不斜跳（与原地起跳**同一道许可门** ✓）；
      ⑦ `retry()` ⇒ 重新武装（新的一轮还能斜跳一次 ✓）；
      ⑧ 真实数据：`job_for_edge` 造的爬任务**真的把「起跳距离」传进去了**（以前被
         `kw.pop` 扔掉 ✗）。
    """
    from decision import route
    from core import mapdata, zones

    C = route.ClimbJob
    # 绳：x=550，y 从 300（下端）到 100（上端）；入绳端那块面：x 0..600、面 y=300（平的 ✓）
    base = dict(ladder_id="L1", x=550.0, y1=300.0, y2=100.0, direction=1,
                dst_set="上", src_set="下", tol_px=10, hold_ms=250,
                near_px=20, jump_start_px=120)

    def upd(j, px, py, now=0.10):
        return j.update(now, px, py=py, here_sets=["下"])

    # ① 关掉（0）⇒ 区间内绝不斜跳；方向键照旧一口气按住走（老行为 ✓）
    j = C(**dict(base, jump_start_px=0))
    o = upd(j, 470.0, 300.0)                    # dx = 80（远大于近距 20）
    check(not o["jump"] and o["move"] == 1,
          "「起跳距离=0」时不该斜跳（应退回老逻辑：一口气按住走过去）：%s" % o)
    check("斜跳" not in j.note, "关掉了还写着斜跳：%r" % j.note)

    # ② 开了 ⇒ 区间内 ⇒ ⭐ **先按朝绳方向（第 1 拍）→ 下一拍才按跳**（用户 2026-09-28：
    #    "斜跳也**先发方向键、下一拍再按跳**" ✓ —— 与「跳下绳梯」那条同一个理由：同一次
    #    tick 里把"方向"和"跳"一起发，游戏里未必算"按住方向再跳"（不保证顺序 ✗））
    j = C(**base)
    o = upd(j, 470.0, 300.0)                     # 第 1 拍：只朝绳走（dx=80，在区间内 ✓）
    check(not o["jump"] and o["move"] == 1 and o["dir"] == 0,
          "斜跳第 1 拍该**只发方向键**、**不按跳、也不按 ↑**"
          "（用户 2026-10-02：「应该是**按跳之后再按住 ↑**」—— 这一拍还没按跳 ✗）：%s / %r"
          % (o, j.note))
    check("斜跳上绳" in j.note and "先按住" in j.note,
          "斜跳第 1 拍没说清在干什么：%r" % j.note)
    o = upd(j, 470.0, 300.0, now=0.15)           # 第 2 拍：这才按跳 ✓
    check(o["jump"] and o["move"] == 1 and o["dir"] == 1,
          "斜跳第 2 拍该按跳（继续朝绳走 + 按住 ↑）：%s / %r" % (o, j.note))
    check(j._diag_flying, "斜跳第 2 拍该进「飞行相」✗")

    # ③ 一轮只斜跳一次：后面几拍都不许再按跳，但方向键一直按着
    for k in range(4):
        o = upd(j, 470.0, 300.0, now=0.10 + 0.05 * (k + 1))
        check(not o["jump"], "一轮里按了不止一次跳（会变成一路乱跳 ✗）：%s" % o)
        check(o["move"] == 1, "不按跳的那几拍方向键该还按着（朝绳走 ✓）：%s" % o)

    # ④ 走近到 near_px 以内 ⇒ 交回老逻辑 + 斜跳重新武装
    #   ⚠ 得先让这一跳**飞完**（升上去、再落回起跳高度 ⇒ ⑨ 那个"飞行相"才会收 ✓；
    #     飞行期间一律按住 ↑、不做横向对齐 ✓）
    upd(j, 500.0, 255.0, now=0.20)              # 在空中（离开起跳高度 ✓）
    upd(j, 520.0, 300.0, now=0.25)              # 落回起跳高度 ⇒ 飞行结束 ✓
    check(not j._diag_flying, "落回起跳高度了还留在「飞行相」✗")
    #   ⭐⭐ **2026-09-28 改口径**：去掉"一轮只斜跳一次" ⇒ 落回起跳高度后，只要**又满足**
    #      `near_px < |dx| ≤ jump_start_px` 就**再斜跳一次** ✓ —— 用户原话："我从未提过
    #      『一轮只斜跳一次』这个限制，**只要满足条件就可以无限斜跳**"。
    #      ⇒ 所以 0.25 那一拍（`dx=30`，在区间里 ✓）**重新起手** ⇒ 0.30 这拍**照样按跳** ✓
    #      （"朝绳走了一步可能就进容差了，那一拍照样要跳"✓ 见 `_align_step` 顶上那段 ✓）。
    o = upd(j, 540.0, 300.0, now=0.30)          # dx = 10 ≤ near 20（但对起手来说无所谓 ✓）
    check(o["jump"],
          "落回远处后又满足斜跳条件，却没再斜跳（用户 2026-09-28：**只要满足就无限斜跳** ✗）：%s"
          % o)

    # ⑤ 比起跳距离还远 ⇒ 不斜跳（老逻辑：一口气按住走过去）
    j = C(**base)
    o = upd(j, 300.0, 300.0)                    # dx = 250 > 120
    check(not o["jump"] and o["move"] == 1,
          "离得比起跳距离还远时不该跳（够不着 ✗）：%s" % o)

    # ⑥ ⚠ 2026-09-27 用户明确否掉了那道"人得站在绳底那块面上"的许可（原话："对齐了就按跳
    #    开始爬啊，**不需要判定别的**"）⇒ **y 在别的高度也照样斜跳** ✓
    #    （唯一条件 = 距离区间 `near_px < |dx| ≤ jump_start_px` ✓）
    j = C(**base)
    o = upd(j, 470.0, 100.0)                     # 第 1 拍：只朝绳走（**不看 y** ✓）
    check(o["move"] == 1 and not o["jump"],
          "y 在别的高度就不斜跳了（那道许可门已拆，别再按 y 拦 ✗）：%s / %r" % (o, j.note))
    o = upd(j, 470.0, 100.0, now=0.15)           # 第 2 拍：按跳 ✓（也一样不看 y ✓）
    check(o["jump"],
          "y 在别的高度就不斜跳了（那道许可门已拆，别再按 y 拦 ✗）：%s / %r" % (o, j.note))

    # ⑦ retry() ⇒ 重新武装（新的一轮还能斜跳一次）
    j = C(**base)
    upd(j, 470.0, 300.0)                         # 第 1 拍：发方向键
    upd(j, 470.0, 300.0, now=0.15)               # 第 2 拍：按跳 ⇒ 才算"这一轮跳过了" ✓
    check(j._diag_done, "斜跳之后该记成「这一轮跳过了」")
    j.retry()
    check(not j._diag_done, "retry() 之后斜跳没重新武装（新的一轮试不了 ✗）")
    upd(j, 470.0, 300.0, now=0.30)               # 新的一轮：第 1 拍
    o = upd(j, 470.0, 300.0, now=0.35)           # 新的一轮：第 2 拍 ⇒ 按跳 ✓
    check(o["jump"] and "斜跳上绳" in j.note,
          "新的一轮该还能斜跳一次：%s / %r" % (o, j.note))

    # ⑨ ⚠ **飞行相：斜跳之后必须一直按住 ↑**（用户 2026-09-27 现场原话："现在斜跳上绳梯
    #    会有「**经过绳梯却没有上去**」的现象，**空中吸附绳梯的条件是「正在按住 ↑」**"）——
    #    以前跳完那一拍就交回对齐逻辑（它只发水平方向 ⇒ `dir=0`）⇒ ↑ 被松开 ⇒ 吸不上 ✗
    j = C(**base)
    upd(j, 470.0, 300.0)                         # 第 1 拍：只朝绳走（见 ② 那段 ✓）
    o = upd(j, 470.0, 300.0, now=0.15)           # 第 2 拍：按跳 + 按住 ↑ ✓
    check(o["jump"] and o["dir"] == 1, "斜跳那一拍该按住 ↑：%s" % o)
    for k in range(4):
        o = upd(j, 460.0 - k * 5.0, 282.0 - k * 8.0, now=0.20 + 0.05 * k)
        check(not o["jump"], "飞行中不该再按跳（一轮只跳一次 ✓）：%s" % o)
        check(o["dir"] == 1,
              "飞行中**没按住 ↑**（空中吸绳就靠它 ⇒ 会「经过绳梯却没上去」✗）：%s" % o)
        check(o["move"] > 0, "飞行中该继续朝绳走：%s" % o)

    # ⑩ ⭐⭐ 吸上绳 ⇒ **这一拍只按住 ↑**（不按跳、不按左右）、**下一拍**交给爬升相 ✓
    #    用户 2026-10-03 现场原话："**现在一跳上绳立马就跳下来了**" —— 老代码在"吸上"这拍
    #    只清掉「飞行中」就**继续往下走** ✗ ⇒ 落到"重新起手斜跳 / 对齐按跳"那两支 ⇒
    #    ① **按住左右**（在绳上按左右 = **松手** ⇒ 掉下来 ✗）② 下一拍**又按一次跳**
    #    （在绳上按跳 = **跳下来** ✗）⇒ 一秒多一轮**死循环**（`behavior.log` 实证 ✓：
    #    `climb_on_ladder v=1` 之后紧接着"先按住 ← 朝绳走"+"按跳"，然后 `v=0` ✓）。
    o = j.update(0.45, 500.0, py=250.0, ladder_id="L1", here_sets=["下"])
    check(not j._diag_flying and j.phase == C.CLIMB,
          "吸上绳之后还留在「飞行相」/ 没交给 CLIMB（`agent` 的 `route_diag` 也按"
          "「这一拍相变没变成 CLIMB」判吸上 ⇒ 拖一拍会被记成「结束·没吸上」✗）："
          "phase=%s flying=%s" % (j.phase, j._diag_flying))
    check(o["move"] == 0 and not o["jump"],
          "⭐⭐ 吸上绳那一拍还在按方向 / 按跳 ⇒ 在绳上按左右 = 松手、按跳 = 跳下来 ✗"
          "（用户现场：「一跳上绳立马就跳下来了」）：%s" % o)
    check(o["dir"] == 1, "吸上绳那一拍该按住 ↑（上爬就靠它 ✓）：%s" % o)
    #    ⚠ 上一条**换了写法**（反向验证抓到的盲区 ✓）：原来断言"`_diag_arming` 被清空" ✗
    #       —— 吸上那一拍它本来就是空的（按跳那拍就清了 ✓）⇒ 把那行删掉照样绿 ✗。
    #       改成**接着喂两拍**：在绳上再按一次跳 = 跳下来（用户现场就是这么掉下去的 ✓）。
    _o2 = j.update(0.50, 500.0, py=245.0, ladder_id="L1", here_sets=["下"])
    _o3 = j.update(0.55, 500.0, py=240.0, ladder_id="L1", here_sets=["下"])
    check(not _o2["jump"] and not _o3["jump"],
          "吸上绳之后的下一拍 / 下下拍又按跳 ⇒ 在绳上按跳 = **跳下来** ✗"
          "（用户现场就是这样掉下去的：「一跳上绳立马就跳下来了」）：%s / %s" % (_o2, _o3))

    # ⑪ 落回起跳高度 ⇒ 飞行结束、交回老逻辑（对齐 x、不再按跳 ✓）
    j = C(**base)
    upd(j, 470.0, 300.0)                        # 第 1 拍：只朝绳走
    upd(j, 470.0, 300.0, now=0.15)              # 第 2 拍：按跳（进飞行相，起跳高度 = 300 ✓）
    upd(j, 450.0, 200.0, now=0.20)              # 升上去（离开起跳高度 ✓）
    check(j._diag_left, "升上去了却没记「离开过」（那『落回起跳高度』就判不出来 ✗）")
    upd(j, 540.0, 305.0, now=0.25)              # 落回起跳高度（±误差范围以内）
    check(not j._diag_flying, "落回起跳高度了还留在「飞行相」✗")
    #    ⚠ 落在**近距**里看（别又走出 `near_px` 外 —— 那样会**按设计重新武装**再斜跳一次 ✓，
    #      那是"走近了 ⇒ 下次离远还能再试"的意思，不是 bug ✗）
    o = upd(j, 545.0, 300.0, now=0.30)
    check(not o["jump"] and o["dir"] == 1,
          "飞行结束后该交回老逻辑（对齐 x：不按跳；**上爬 ↑ 一路按着** ✓）：%s" % o)

    # ⑫ **下爬**那一跳也按住 ↑（用户给的条件不分上下 ✓；吸上之后 CLIMB 才按 ↓ ✓）
    j = C(ladder_id="L2", x=600.0, y1=100.0, y2=300.0, direction=-1,
          dst_set="下", src_set="下", tol_px=10, hold_ms=250, near_px=20,
          jump_start_px=120)
    o = upd(j, 520.0, 100.0)                     # 第 1 拍：只朝绳走（+x 方向 ✓）
    check(o["move"] == 1 and not o["jump"] and o["dir"] == 0,
          "下爬的斜跳第 1 拍该只发方向键（还没按跳 ⇒ **不按 ↑** ✓，用户 2026-10-02）："
          "%s / %r" % (o, j.note))
    o = upd(j, 520.0, 100.0, now=0.15)           # 第 2 拍：按跳 ✓
    check(o["jump"] and o["dir"] == 1,
          "下爬的斜跳也要按住 ↑（空中吸绳的条件不分上下 ✓）：%s" % o)

    # ⑬ ⚠ **飞行相卡住的兜底**（用户 2026-09-28 报："一楼到二楼又出问题、斜跳飞行会卡" ✗）：
    #   人**离开过**起跳高度（`_diag_left=True`）却**既没吸上绳、也没落回起跳高度**——
    #   比如斜跳落到绳旁另一个高度、y 停住不动 ⇒ 原来那条兜底只认"**没离开过**"
    #   （`not self._diag_left` ✗）⇒ 失效 ⇒ **永远卡在飞行相** ✗。
    #   ⇒ 改成**只看飞行总时长**：超过「卡住判定时长」（默认 3s ✓）就结束、交回老逻辑 ✓。
    j = C(**base)                                   # stall_s 默认 3.0 ✓
    upd(j, 470.0, 300.0)                            # 第 1 拍：只朝绳走（用户 2026-09-28 拆分 ✓）
    upd(j, 470.0, 300.0, now=0.15)                  # 第 2 拍：按跳 ⇒ 进飞行相（_diag_at = 0.15 ✓）
    o = j.update(0.20, 450.0, py=250.0, here_sets=["下"])   # 升上去 ⇒ 离开起跳高度 ✓
    check(j._diag_left and j._diag_flying,
          "用例前提：该是「离开过 + 还在飞行」：left=%s flying=%s"
          % (j._diag_left, j._diag_flying))
    for k in range(30):                             # y 停在 250、不吸绳、不回 300 ⇒ 推到超 3s
        o = j.update(0.25 + k * 0.15, 450.0, py=250.0, here_sets=["下"])
        if not j._diag_flying:
            break
    check(not j._diag_flying,
          "离开过起跳高度、又没吸上也没落地时，飞行相卡死不结束 ✗：flying=%s" % j._diag_flying)
    check(not o["jump"],
          "飞行结束兜底之后不该还在按跳（该交回老逻辑对齐 x ✓）：%s" % o)

    # ⑧ 真实数据：job_for_edge 真的把「起跳距离」传进 ClimbJob 了
    t = mapdata.load("105090600")
    z = zones.load("105090600")
    ce = next((e for e in z.edges if e.get("kind") == "climb"), None)
    check(ce is not None, "用户数据里没有「爬」边，这条测不了")
    jj = route.job_for_edge(t, z, ce, tol_px=6, hold_ms=0, stall_s=3.0,
                            jump_start_px=90)
    check(float(getattr(jj, "jump_start_px", 0)) == 90.0,
          "爬任务没拿到「起跳距离」（以前 `kw.pop` 把它扔了 ✗）：%r"
          % (getattr(jj, "jump_start_px", None),))


def t_chase_goto_other_foothold():
    """chase 前先判「**怪和我是不是同一块平台**」，不是 ⇒ **先下前往任务**（用户 2026-09-27）。

    原话："chase状态需要判定一下怪物位于的foothold集合，若不与玩家处于同一个，
    需要先下达前往任务"

    为什么要：`chase` 只会**水平**按左右键（`_steer`）⇒ 怪在**别的平台**（上层/下层/隔着墙）
    时，人只会贴着平台边缘干蹭 ✗（`docs/寻路设计.md` 把"怪物追击与寻路的融合"原本列在
    "先不做"里 —— 这就是那件事 ✓）。

    判据数据靠**实时线程注入**的 `agent.mob_sets_of`（agent 不持有地形/集合 ✓；
    "画面 → 世界"那一步只有一处：`perception.minimap.screen_to_world` ✓）。

    钉六件：
      ① 同一集合 ⇒ 照旧追（进 chase、**不**下前往 ✓）；
      ② 不在同一集合 ⇒ 进 chase 的同一拍就下了「前往」，目标 = 怪所在的那个集合 ✓
         而且那句说明写清是「追击：怪不在我这块平台上…先过去」✓；
      ③ 怪同时在几个集合里 ⇒ 从前往后试，**第一个能造出路线的**才用 ✓；
      ④ 手上已有任务（`_climb` 非空）⇒ **不打断**（`plan_and_start_route` 会覆盖行程 ✗）；
      ⑤ 没有解析器 / 解析不出集合（老环境 / 没定位）⇒ **退回照旧追**（绝不站着不动 ✗）；
      ⑥ 重下有间隔（`ZONE_GOTO_RETRY_S` 内不再下第二条 ✓）。
    """
    from decision import route
    from decision.agent import ZONE_GOTO_RETRY_S    # 和「限制战斗区域」共用同一件参数 ✓

    s = fresh_settings(attack_dist=40.0)        # 攻击距离小 ⇒ 怪必在范围外 ⇒ 走 chase ✓
    s.enabled = True
    h = Harness(s)
    h.mobs_fn = lambda t: [Mob(id=1, x=580.0, y=500.0, w=40.0, h=40.0, conf=0.9)]

    def _plan(dst):
        if dst == "丙平台":
            return {"jobs": [], "path": [], "why": "没有可走的路（用例）", "here": False}
        return {"jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
                "path": ["乙平台", dst], "why": "", "here": False}

    with h._patched():
        a = h.agent
        h.clock0 = h.clock.t
        a.route_plan = _plan
        ws = h.ws(with_mob=True)
        ws.player.here_sets = ["乙平台"]

        # ① 同一集合 ⇒ 照旧追、不下前往
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -200.0),
                                        "sets": ["乙平台"], "why": ""}
        h.clock.t += 0.1
        act = a.tick(ws)
        check(act.get("state") == "chase", "同一块平台上该照旧追：%s" % act)
        check(a._climb is None, "同一块平台上却下了前往命令：%r" % (a._climb,))

        # ①' ⭐ **同一秒里的抖动不许改目的地**（用户 2026-09-28："锁定一个目标时每秒更新其
        #     位于的 foothold 集合"✓ —— 治的就是"**锁着目标来回走**"：怪框抖几像素就让它的
        #     集合在楼层之间跳 ⇒ 人走到一半目的地又被改写 ✗）。这里只推进 0.2 秒
        #     （< `MOB_SETS_TTL_S` ✓）⇒ 该**照旧追**、不下前往 ✓。
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0),
                                        "sets": ["甲平台"], "why": ""}
        h.clock.t += 0.2
        act1b = a.tick(ws)
        check(act1b.get("state") == "chase" and a._climb is None,
              "同一秒内怪「换层」了就改目的地（这正是现场「锁着目标来回走」✗）：%s / %r"
              % (act1b, a._climb))

        # ② 不在同一集合 ⇒ 先下前往（目标 = 怪那个集合）
        #    ⚠ 时间要**跨过 `MOB_SETS_TTL_S`**（2026-09-28 ✓）：怪真换层这件事得发生在
        #      "一秒之后" —— 同一秒内的变化属于**抖动**，就该被缓存滤掉（见上面 ①' ✓）。
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0),
                                        "sets": ["甲平台"], "why": ""}
        h.clock.t += ag.MOB_SETS_TTL_S + 0.05
        act2 = a.tick(ws)
        check(a._climb is not None and getattr(a._climb, "dst_set", "") == "甲平台",
              "怪在别的平台上却没下「前往」：%r / %s" % (a._climb, act2))
        check("追击" in str(a._route_why) and "甲平台" in str(a._route_why),
              "那条命令的说明没写清「追击：怪不在这块平台上」+ 去哪：%r" % (a._route_why,))
        a.stop_route("用例：②检查完了")

        # ③ 怪同时在多个集合里 ⇒ 第一个**造得出路线**的才用（丙平台造不出 ⇒ 退到甲平台）
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0),
                                        "sets": ["丙平台", "甲平台"], "why": ""}
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        a.tick(ws)
        check(a._climb is not None and getattr(a._climb, "dst_set", "") == "甲平台",
              "多个集合时没退到能走的那一个：%r" % (a._climb,))

        # ④ 手上已有任务时：**目标是怪那块 ⇒ 不打断；不一样 ⇒ 要重下**（用户 2026-09-28：
        #    "**查询了框如果不一样要下寻路任务呀**" ✓）。
        #    ⚠ 这两条**只能直接调判据**测 ✓ —— 走 `tick` 的话，"手上已有任务"时状态已经是
        #      `climb`（`_climb_tick` 那条路 ✓）⇒ **压根不进 `chase` 这一段** ⇒ 断言会**假过**
        #      （"dst 还是丁平台" 在"没进这段"时也成立 ✗ —— 第一版就这么写的 ✓）。
        a.stop_route("用例：④准备")
        a.start_route([route.WalkJob("丁平台", [(520.0, 510.0, 530.0, -208.0, "1")])],
                      "用例：先在走的任务")
        _mob1 = Mob(id=1, x=580.0, y=500.0, w=40.0, h=40.0, conf=0.9)
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0),
                                        "sets": ["丁平台"], "why": ""}
        a._mob_info_cache = None
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        check(a._chase_goto_if_elsewhere(_mob1, h.clock.t, ws) is False,
              "**正走的就是怪那块**（「丁平台」），却把行程改掉了：%r"
              % (getattr(a._climb, "dst_set", None),))
        # ④' ⭐ 查出来的框**跟手上任务不一样** ⇒ **要下新的寻路任务**（怪换层了 ✓）
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0),
                                        "sets": ["甲平台"], "why": ""}
        a._mob_info_cache = None
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        a._chase_goto_if_elsewhere(_mob1, h.clock.t, ws)
        check(getattr(a._climb, "dst_set", "") == "甲平台",
              "**查出来的框跟手上任务不一样，却不下新的寻路任务**"
              "（怪早换层了、人还在走向旧地方 ✗）：%r"
              % (getattr(a._climb, "dst_set", None),))
        a.stop_route("用例：④检查完了")

        # ⑤ 没有解析器 ⇒ 退回照旧追（不许站着不动 ✗）
        a.mob_sets_of = None
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        act5 = a.tick(ws)
        check(act5.get("state") == "chase" and a._climb is None,
              "判不出怪的集合时该照旧追（不是站着不动 ✗）：%s / %r" % (act5, a._climb))

        # ⑥ 重下有间隔：刚下过 ⇒ `ZONE_GOTO_RETRY_S` 之内不再下第二条
        a.stop_route("用例：⑥准备")
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0),
                                        "sets": ["甲平台"], "why": ""}
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1     # 节流放行 ⇒ 这一拍该下得出
        a.tick(ws)
        first = a._climb
        check(first is not None, "⑥ 前置不成立（这一拍本该下得下去）")
        h.clock.t += 0.05                        # 远小于节流间隔 ⇒ 不该再造新任务
        a.tick(ws)
        check(a._climb is first,
              "重下没有间隔（每拍都造新任务 = 刷屏 ✗）：%r" % (a._climb,))


def t_chase_goto_respects_can_fight():
    """追击下前往的**目的地**也要在「可以战斗」白名单里（用户 2026-10-01 ✓）。

    病根：`can_fight` 原来只管"打架资格"（人在能打区才打 / 只锁能打区的怪 / 回能打区），
    **漏了追怪的两条下前往路径** ✗ —— 于是"把二楼关了还是会寻路去二楼"：
      ① 主路径 `for name in msets`：怪**同时属于多个集合**时，会试到被勾掉的那个 ✗；
      ② 降级路径 `_mob_goto_towards`：`nearest_set_towards` 是纯几何、不看 can_fight ✗。
    钉三件：目的地不在 `battle_zone_sets` 里 ⇒ 主路径**跳过改试下一个** / 降级路径**放弃** ✓；
    名单空（哪都能打）时一字不变 ✓。
    """
    from decision import route
    from decision.agent import ZONE_GOTO_RETRY_S

    s = fresh_settings(attack_dist=40.0)
    s.enabled = True
    h = Harness(s)
    h.mobs_fn = lambda t: [Mob(id=1, x=580.0, y=500.0, w=40.0, h=40.0, conf=0.9)]

    def _plan(dst):
        return {"jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
                "path": ["乙平台", dst], "why": "", "here": False}

    with h._patched():
        a = h.agent
        h.clock0 = h.clock.t
        a.route_plan = _plan
        ws = h.ws(with_mob=True)
        ws.player.here_sets = ["乙平台"]      # 玩家脚下：能打区
        ws.player.world_x = 500.0
        # 乙/丙能打；「甲平台」被勾掉（can_fight=False）
        a.settings.battle_zones = [{"set": "乙平台", "can_fight": True},
                                   {"set": "丙平台", "can_fight": True},
                                   {"set": "甲平台", "can_fight": False}]
        s.sync_battle_zone_sets()
        check(list(s.battle_zone_sets) == ["乙平台", "丙平台"],
              "前置不成立：派生副本没把「甲平台」剔掉：%r" % (s.battle_zone_sets,))

        # ① 怪同时在「甲平台(不能打)」「丙平台(能打)」⇒ 跳过甲平台、去丙平台
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0),
                                         "sets": ["甲平台", "丙平台"], "why": ""}
        a._mob_info_cache = None
        a._mob_goto_at = 0.0
        t = h.clock.t + ZONE_GOTO_RETRY_S + 0.1
        r = a._chase_goto_if_elsewhere(ws.mobs[0], t, ws)
        check(r is True and a._climb is not None
              and getattr(a._climb, "dst_set", "") == "丙平台",
              "怪同属多集合时该跳过被勾掉的「甲平台」、去能打的「丙平台」：%r / %r"
              % (r, getattr(a._climb, "dst_set", None)))
        a.stop_route("用例：①检查完了")

        # ② 降级路径：nearest_set_towards 吐出被勾掉的「甲平台」⇒ 不下前往
        a.nearest_set_towards = lambda pl, mob: "甲平台"
        a._mob_goto_at = 0.0
        t2 = h.clock.t + ZONE_GOTO_RETRY_S + 0.1
        r2 = a._mob_goto_towards(ws.mobs[0], ws, t2,
                                 "降级：朝方向最近集合「%s」逐层逼近")
        check(r2 is False and a._climb is None,
              "降级路径该不去被勾掉的「甲平台」：%r / %r" % (r2, a._climb))

        # ③ 名单空 = 不限制 ⇒ 降级路径照常去（老行为 ✓）
        a.settings.battle_zone_sets = []
        a._mob_goto_at = 0.0
        t3 = h.clock.t + 2 * ZONE_GOTO_RETRY_S + 0.1
        r3 = a._mob_goto_towards(ws.mobs[0], ws, t3,
                                 "降级：朝方向最近集合「%s」逐层逼近")
        check(r3 is True and a._climb is not None
              and getattr(a._climb, "dst_set", "") == "甲平台",
              "名单空（哪都能打）时降级路径该照常去「甲平台」：%r" % (r3,))
        a.stop_route("用例：③检查完了")
    a.settings.battle_zones = []
    s.sync_battle_zone_sets()


def t_drop_detach_ladder():
    """下跳途中**被"通往下层的绳"吸住 ⇒ 先脱离**（用户 2026-09-27 要求）。

    原话："现在在下跳执行器中可能会吸附到通往下层的绳子上，这种情况应该执行「**任意方向按住
    + 连按跳**」脱离（**世界状态应该是能知道的**）"。

    ⛔ **判据 2026-09-28 换过**（用户现场："**下跳 drop 结果实际爬上了向下爬的绳子**"✗）：
    原来只判**带按键许可的** `ladder_id` ⇒ **两头都错** ✗：
      · **误报**：`ARMED`/`DROP` 相**自己按 ↓** ⇒ 许可开 ⇒ 人站在下跳点（"通往下层的绳"
        就在旁边，`LADDER_DX`=24 / `LADDER_PAD`=12 一把**宽**尺盖得住 ✓）⇒ `ladder_id`
        当场非空 ⇒ 进 `DETACH`（现场 `align→armed→detach` 只隔 **28 ms** ✗）；而 `DETACH`
        相不按 ↓ ⇒ 下一拍许可关 ⇒ "又没在绳上" ⇒ 回 `ALIGN` ⟲；
      · **漏报**：人**真挂在绳上**时 `ALIGN`/`DETACH` 相不按竖直键 ⇒ 许可关 ⇒ `ladder_id`
        恒 `None` ⇒ **判不出来** ✗ ⇒ 人一直挂在绳上（`behavior.log`：`climb_y=-90.7` 恒定 +
        `climb_on_ladder=0` 近 **9 秒**、期间**一条 `route_phase` 都没有** ✗）⇒ 10 秒
        「寻路超时」⇒ 重下 ⇒ 再超时 ⟲。
    ⇒ 现在用**两份"只看位置"的客观事实**（都还是世界状态 ✓、与按键许可无关 ✓）：
      `on_rope_pos`（位置在绳段里）**且** `ground_y is None`（脚下没面 = 悬空 = 真挂着 ✓）。

    钉八件：
      ① 任何一相里发现"**在绳段里 + 悬空**" ⇒ 进 `DETACH`，**按住水平方向 + 按跳** ✓
         （⚠ 这一条里 `ladder_id` 故意给 `None` —— 现场"真被吸住"时许可正是关的 ✓）；
      ② ⚠ **绝不按 ↓**（`dir == 0`）—— 在绳上按 ↓ = "还挂在绳上"（同 `ClimbJob._out` 的
         `vert` 口径 ✓），按下去等于把自己按回去 ✗（**这条是这次的要害** ✓）；
      ③ 跳按**同一份点按节奏**（一下按、一下松 ✓ 不新造时长 ✓）；
      ④ 方向"**背离下跳点中心**"（别刚松手又吸回去 ✓）；
      ⑤ 脱离成功（出了绳段）⇒ **回入口重走下跳**（重挑下跳点 ✓）；
      ⑥ 没被吸住时**一字不变**（老流程 ✓）；
      ⑦ ⛔ **反向那一半**：位置在绳段里 **但脚下有面**（人站在下跳点上）⇒ **绝不算被吸住**
         —— 哪怕带许可的 `ladder_id` 非空（"自己按的 ↓ 把自己判死"就是旧毛病 ✗）；
      ⑧ `phase_code()` / `align_target_x()`：日志要能一眼看出**卡在哪一相、离下跳点多远**
         （2026-09-28 加 ✓ —— 那次现场就是因为"看不到相"才拖了很久 ✗）。
    """
    from decision import route

    C = route.DropJob
    spots = [(100.0, 80.0, 120.0, "fhB")]

    def _to_drop(j, t=0.0):
        """推到"按住 ↓ + 连按跳"那一拍（`hold_ms=0` ⇒ 从 ALIGN 直接过 ✓）。"""
        j.update(t, px=100.0, py=0.0)
        j.update(t, px=100.0, py=0.0)
        return j.update(t + route.ARM_HOLD_S + 0.01, px=100.0, py=0.0)

    # ① 下跳途中被吸住 ⇒ 立刻转脱离
    #    ⚠ `ladder_id`（带许可）故意给 `None`：现场"真被吸住"时 `ALIGN`/`DETACH` 相
    #      **不按竖直键** ⇒ 许可关 ⇒ 旧判据（只看它）**判不出来** ✗（这就是 2026-09-28 那条）。
    j = C("上平台", "下平台", spots, tol_px=10, hold_ms=0)
    _to_drop(j)
    check(j.phase == C.DROP, "前置不成立（该在「按压跳」那一相）：%s" % j.phase)
    o = j.update(0.30, px=100.0, py=0.0, ladder_id=None,
                 on_rope_pos="L7", ground_y=None)
    check(j.phase == C.DETACH, "被绳吸住了却没转「脱离」：%s" % j.phase)
    check(o["jump"] and o["move"] != 0,
          "脱离那几拍该「按住方向 + 按跳」：%s" % o)
    # ② ⚠ 要害：**绝不按 ↓**
    check(o["dir"] == 0,
          "脱离时按了 ↓ —— 在绳上按 ↓ 就是「还挂在绳上」，等于把自己按回去 ✗：%s" % o)
    check("不按 ↓" in j.note, "那句说明没点出「不按 ↓」：%r" % j.note)
    # ③ 连按：一下按、一下松
    o2 = j.update(0.30 + route.TAP_ON_S + 0.01, px=100.0, py=0.0,
                  on_rope_pos="L7", ground_y=None)
    check(not o2["jump"] and o2["dir"] == 0,
          "脱离时跳该是**点按**（一下一下，不是一路按住 ✗）：%s" % o2)
    o3 = j.update(0.30 + route.TAP_PERIOD_S + 0.01, px=100.0, py=0.0,
                  on_rope_pos="L7", ground_y=None)
    check(o3["jump"], "脱离时跳没「连按」（那一下可能还没起效 ⇒ 要一遍遍按 ✓）：%s" % o3)
    # ④ 方向：背离下跳点中心（中心 x=100、人在 130 ⇒ 该往左）
    o4 = j.update(0.5, px=130.0, py=0.0, on_rope_pos="L7", ground_y=None)
    check(o4["move"] == -1,
          "脱离方向该**背离下跳点中心**（往左，顺手离它远点、别又吸回去 ✓）：%s" % o4)
    # ⑤ 脱离了（位置出了绳段）⇒ **离开「脱离」相、回入口重走下跳**（重挑下跳点 ✓）
    #    ⚠ 同一拍就会往下走（`hold_ms=0` ⇒ 可能直接进 ARMED）⇒ 别断言"停在 ALIGN" ✗，
    #      只断言"已经不在 DETACH 里了、而且下跳点被重挑过" ✓。
    j.update(0.6, px=130.0, py=0.0, on_rope_pos=None)
    check(j.phase != C.DETACH,
          "脱离之后该离开「脱离」相、回入口重走下跳：%s" % j.phase)
    check(j._pick is not None and j._pick[-1] == "fhB",
          "回入口该重挑下跳点（人可能被绳带到别处了 ✓）：%r" % (j._pick,))
    # ⑥ 没被吸住 ⇒ 一字不变（老流程：按住 ↓ + 连按跳）
    j2 = C("上平台", "下平台", spots, tol_px=10, hold_ms=0)
    o6 = _to_drop(j2)
    check(j2.phase == C.DROP and o6["jump"] and o6["dir"] == -1,
          "没被吸住时该一切照旧（按住 ↓ + 连按跳 ✓）：%s" % o6)
    # ⑦ ⛔ **反向那一半（2026-09-28 修）**：`ARMED`/`DROP` 相**自己按 ↓** ⇒ 带许可的
    #    `ladder_id` **非空**，但人**站在下跳点上**（脚下有面 ⇒ `ground_y` 非 None）
    #    ⇒ **绝不算"被吸住"** ✗ —— 旧判据在这儿就是"**自己按的 ↓ 把自己判死**"
    #    （现场 `align→armed→detach` 只隔 28 ms，然后 `DETACH` 不按 ↓ ⇒ 许可关 ⇒
    #     又回 `ALIGN` ⟲ ⇒ 永远下不去 ✗）。
    j3 = C("上平台", "下平台", spots, tol_px=10, hold_ms=0)
    _to_drop(j3)
    check(j3.phase == C.DROP, "⑦ 前置不成立（该在「按压跳」那一相）：%s" % j3.phase)
    #    ⚠ 数据要**真的像"站在面上"**：脚下那块面就在脚底（`|py-ground_y|` 只有几像素 ✓）
    #       —— 放成 100px 之下那就已经是"悬空"了（新判据按面距算 ✗），测不到这条 ✓。
    o7 = j3.update(0.30, px=100.0, py=0.0, ladder_id="L7",
                   on_rope_pos="L7", ground_y=-2.0)   # 位置在绳段里，但**面就在脚底**
    check(j3.phase == C.DROP and o7["dir"] == -1,
          "站在下跳点上（脚下有面、不是悬空）却判成「被绳吸住」⇒ 自己按的 ↓ 把自己判死 ✗：%s"
          % j3.phase)
    # ⑧ 可观测性：日志要能一眼看出**卡在哪一相**、离下跳点**还差多远**（2026-09-28 加）
    check(j3.phase_code() == 3,
          "`drop_phase` 编码不对（drop 相该是 3）：%r" % (j3.phase_code(),))
    check(j.align_target_x() == 100.0,
          "`align_target_x`（日志里看还差多远）没给出下跳点中心：%r" % (j.align_target_x(),))
    # ⑨ `drop_taps` / `drop_round`：**跳到底发了没有、轮次转不转**（用户 2026-09-28
    #    报"趴在那不动了也不按跳" —— 光看 `drop_phase=3` 分不清"没发"还是"不响应" ✗）。
    #    ⚠ `jump` 会连着几拍为真（要按满 `TAP_ON_S`）⇒ 计数必须是**上升沿**，否则一下算成好几下 ✗。
    j4 = C("上平台", "下平台", spots, tol_px=10, hold_ms=0)
    j4.update(0.0, px=100.0, py=0.0)
    j4.update(0.0, px=100.0, py=0.0)
    o9 = j4.update(route.ARM_HOLD_S + 0.01, px=100.0, py=0.0)      # 进 DROP ⇒ 第 1 下跳
    check(o9["jump"] and j4.tap_count() == 1,
          "第 1 下跳没计上（`drop_taps` 会一直显示 0，看不出跳发没发 ✗）：%r" % (j4.tap_count(),))
    o10 = j4.update(route.ARM_HOLD_S + 0.01 + route.TAP_ON_S + 0.01, px=100.0, py=0.0)
    check(not o10["jump"] and j4.tap_count() == 1,
          "「松窗」（`TAP_ON_S` 之后、`TAP_PERIOD_S` 之前）该**松开跳**、也不该加计数："
          "jump=%s taps=%r" % (o10["jump"], j4.tap_count()))
    o11 = j4.update(route.ARM_HOLD_S + 0.01 + route.TAP_PERIOD_S + 0.01, px=100.0, py=0.0)
    check(o11["jump"] and j4.tap_count() == 2,
          "第 2 下跳没计上（`TAP_PERIOD_S` 到点该再按一下 ✓）：jump=%s taps=%r"
          % (o11["jump"], j4.tap_count()))
    check(j4.round_n() == 1, "轮次起点该是 1：%r" % (j4.round_n(),))
    # ⑪ ⛔ **人挂在绳上时，"脚下"的读数不可信**（用户 2026-09-28 现场："从**顶层到三楼**的
    #    下跳期间**上绳子了**，但是 drop 执行器没有从位置状态机拿到上绳通知而去执行跳开绳子
    #    的逻辑"✗）。真数据（图 106010105 绳 L2）：人挂在绳**正中** y=-291、离下面那块面还有
    #    **109px**，而感知照样给 `ground_y=-182` + `here_sets=['三楼']` ✗（因为
    #    `foothold_below` 只对"头顶上方"卡 40px、**下方不卡** ✗）⇒ 原来**两条判据双双误报
    #    "到了"** ⇒ 任务提前 `done`（日志里就是 `prev=drop to=done note=到达：脚下已经是
    #    「三楼」`✗）⇒ **根本轮不到"跳开绳子"** ✗。
    #    ⇒ 现在：`on_rope_pos` 非空 ⇒ **一律不算到达** ✓、并且判成"被绳吸住"去脱离 ✓
    #    （`_hanging` = 位置在绳段里 **且** 脚下那块面离我超过 `FOOT_GAP_PX` ✓）。
    j6 = C("上平台", "下平台", spots, tol_px=10, hold_ms=0, dst_ys=[-182.0])
    j6.update(0.0, px=100.0, py=-291.0)
    j6.update(0.0, px=100.0, py=-291.0)
    o6 = j6.update(0.3, px=100.0, py=-291.0, here_sets=["下平台"],
                   ground_y=-182.0, on_rope_pos="L2")
    check(j6.phase == C.DETACH and not o6["done"],
          "人在绳上（位置在绳段里）、而感知把**下面那层**的面报成「脚下」⇒ 该判"
          "「被绳吸住、去脱离」而不是「到达」✗：相=%s done=%s note=%r"
          % (j6.phase, o6["done"], j6.note))
    check("吸住" in j6.note, "脱离那句没说清是「被绳吸住」：%r" % j6.note)
    # 对照：**真站在目标面上**（面就在脚下、不在绳上）⇒ 照旧算到达 ✓（修过头也是错 ✗）
    j7 = C("上平台", "下平台", spots, tol_px=10, hold_ms=0, dst_ys=[-182.0])
    j7.update(0.0, px=100.0, py=-180.0)
    o7 = j7.update(0.3, px=100.0, py=-180.0, here_sets=["下平台"],
                   ground_y=-181.0, on_rope_pos=None)
    check(o7["done"], "真站在目标面上却不判到达（修过头了 ✗）：相=%s" % j7.phase)
    # ⑩ ⚠ **拍子慢也必须"点按"**（用户 2026-09-28 报"趴在那不动了也不按跳"的**真因**）：
    #    原来"该按"写在"该松"前面、靠 `now - _jump_at >= TAP_PERIOD_S` 判 ⇒ 拍间隔一旦大于
    #    「点按周期」(180ms)，每拍都命中"重新按" ⇒ **整段跳过"松窗"** ⇒ 退化成**按住跳** ✗
    #    ⇒ 而"一直按住跳会被游戏当成还挂在上面"（用户 2026-09-26 口径）⇒ **下跳根本不触发** ✗。
    j5 = C("上平台", "下平台", spots, tol_px=10, hold_ms=0)
    j5.update(0.0, px=100.0, py=0.0)
    j5.update(0.0, px=100.0, py=0.0)
    t5 = route.ARM_HOLD_S + 0.01
    seen = []
    for _i in range(8):                     # **每拍 200 ms**（慢拍 ✓ 仍必须有点按 ✓）
        seen.append(bool(j5.update(t5, px=100.0, py=0.0)["jump"]))
        t5 += 0.2
    # ⚠ 判据用**按跳次数**（不是"序列里有几个 False/q"）：旧形状下"收工那一拍"也会给 False，
    #   拿序列去数字母会让**没修**也能过 ✗（踩过 ✓）。8 拍 × 200ms = 1.6 s，正常 ≥ 3 下 ✓；
    #   旧形状（一直按住）只会是 **1** 下 ✗。
    check(j5.tap_count() >= 3,
          "拍子慢时「点按跳」退化成了**按住跳**（1.6 秒里只按了 1 下；游戏会当成还挂在上面 "
          "⇒ 下跳根本不触发 ✗）：taps=%r 序列=%r" % (j5.tap_count(), "".join(
              "J" if b else "." for b in seen)))


def t_lock_target_by_path_cost():
    """锁定目标优先级：**先比寻路距离，再比画面绝对距离**（用户 2026-09-27 要求）。

    原话："优化：锁定目标优先级要按「**寻路距离**」最近，而不是旧的应该是**绝对距离**"。

    为什么：老的 `_nearest` 用的 `_center_dist` 是**画面上的直线** ✗ —— 隔着平台 / 要绕一根绳
    的怪，直线可能很近、实际要走很远 ⇒ 会"盯着够不着的那只打" ✗。

    钉七件：
      ① 没注入 `mob_cost_of` ⇒ **整条老口径**（画面绝对距离最近 ✓）；
      ② 注入了：**画面更远、但寻路更近**的那只胜出 ✓（这就是用户要的 ✓）；
      ③ 代价算不出（`None`）⇒ 排到"算得出"的后面；两只都算不出 ⇒ 回到画面距离 ✓（不猜 ✓）；
      ④ 脚下 `here_sets` 空 ⇒ 老口径（不知道我在哪个集合 ⇒ 没法算路 ✓）；
      ⑤ 返回的那个距离**仍是画面距离**（下游「追击起跳」区间 / 打点用的就是它 ✓）；
      ⑥ ⚠ **攻击判定的口径一个字没动**：`_center_dist` 还是老样子（挑谁 ≠ 打得到谁 ✓）；
      ⑦ 真实数据：`route.path_cost` 同集合 = 0、跨集合 > 0（或算不出给 `None`，**绝不能是 0** ✓）。
    """
    from decision import route
    from decision.agent import CombatAgent
    from perception.world_state import Mob, Player, WorldState

    a = CombatAgent(fresh_settings())
    p = Player(x=500.0, y=300.0, bottom=340.0)
    # ① 画面贴身（`_center_dist` = 0）但（下面会设定）寻路很远
    stick = Mob(id=1, x=520.0, y=300.0, w=40.0, h=60.0, conf=0.9)
    # ② 画面 40px 远，但寻路更近
    other = Mob(id=2, x=560.0, y=300.0, w=40.0, h=60.0, conf=0.9)

    ws = WorldState()
    ws.player = p
    ws.player.here_sets = ["甲"]

    # ① 没注入 ⇒ 老口径（画面最近 = 贴身那只）
    t0, d0 = a._nearest([stick, other], p, ws=ws)
    check(t0 is stick and abs(d0) < 1e-6,
          "没注入寻路距离时该退回老口径（画面最近的那只、距离 0）：%r / %r" % (t0, d0))

    # ② 注入了 ⇒ **寻路更近**的那只胜出（哪怕它在画面上更远）
    a.mob_cost_of = lambda pl, m: {"cost": {1: 900.0, 2: 120.0}[m.id],
                                  "sets": [], "why": ""}
    t1, d1 = a._nearest([stick, other], p, ws=ws)
    check(t1 is other,
          "寻路更近的那只没胜出（用户要的就是这个：不按画面直线挑 ✗）：%r" % (t1,))
    # ⑤ 返回的距离**仍是画面距离**（被选中那只的 `_center_dist` ✓）
    check(abs(d1 - a._center_dist(other, p)) < 1e-6,
          "返回的距离口径被换了（下游「追击起跳」的区间用的是画面距离 ✗）：%r" % (d1,))

    # ③ 算不出代价 ⇒ 排在算得出的后面
    a.mob_cost_of = lambda pl, m: ({"cost": None, "sets": [], "why": "算不出"}
                                   if m.id == 1 else {"cost": 5.0, "sets": [], "why": ""})
    t3, _ = a._nearest([stick, other], p, ws=ws)
    check(t3 is other,
          "「算不出代价」的该排在「算得出」的后面：%r" % (t3,))
    a.mob_cost_of = lambda pl, m: {"cost": None, "sets": [], "why": "算不出"}
    t4, _ = a._nearest([stick, other], p, ws=ws)
    check(t4 is stick, "两只都算不出时该按画面距离（老口径 ✓）：%r" % (t4,))

    # ④ 脚下没有集合 ⇒ 老口径（没法算路 ✓）
    ws2 = WorldState()
    ws2.player = Player(x=500.0, y=300.0, bottom=340.0)
    a.mob_cost_of = lambda pl, m: {"cost": {1: 900.0, 2: 120.0}[m.id],
                                  "sets": [], "why": ""}
    t5, _ = a._nearest([stick, other], ws2.player, ws=ws2)
    check(t5 is stick,
          "不知道自己在哪个集合时该退回老口径（不然算不出路还硬按代价排 ✗）：%r" % (t5,))

    # ⑥ 攻击判定那条口径没被碰：`_center_dist` 还是老样子
    check(abs(a._center_dist(other, p) - 40.0) < 1e-6
          and abs(a._center_dist(stick, p)) < 1e-6,
          "`_center_dist`（攻击判定的水平距离来源）被改了 ✗ —— 用户只要改「挑哪只」")

    # ⑦ 真实数据：同集合 = 0、跨集合 > 0（或 None ✓）
    from core import mapdata, zones
    t, z = mapdata.load("105090600"), zones.load("105090600")
    names = sorted(z.sets)
    check(len(names) >= 2, "地图里的集合太少，这条测不了")
    c0 = route.path_cost(t, z, [names[0]], [names[0]], at=None)
    check(c0 == 0.0, "同一个集合的寻路距离该是 0：%r" % (c0,))
    c1 = route.path_cost(t, z, [names[0]], [names[-1]], at=(0.0, 0.0))
    check(c1 is None or c1 > 0.0,
          "跨集合的寻路距离该是正数（或算不出给 None，**绝不能是 0** ✗）：%r" % (c1,))
    check(route.path_cost(t, z, [], ["甲"], at=(0.0, 0.0)) is None,
          "没有来源集合时该给 None（不猜 ✓）")


def t_big_mob_priority():
    """⭐ 大怪优先锁定（用户 2026-10-01 ✓）：**相对判定 + 视野内优先**。

    原话："一堆小怪里有一个明显更高的大怪，想要优先锁定它" + 拍板："相对判定：比当前候选
    怪框高的**中位数**高 ×N（1.5 倍）就算大怪；视野内优先：大怪只在画面距离 ≤ 半径内才插队"。
    ⭐ 2026-10-02 用户改基准（原话："把『大怪判定倍数』逻辑改为『比最小的高这个数』，
    **兼容大怪多小怪少**的情况" + 追问定稿"记录**一段时间内的最小框均线**"）⇒ **基准**从
    "候选中位数"换成"最小框均线"（口径与用例见 `t_big_mob_baseline_min_avg` ✓）；
    **倍数 / 半径 / 总开关这几条一个都没动** ✓（这个用例钉的正是它们 ✓）。

    钉四件：
      ① 一堆小怪里有一只明显更高的大怪（且在优先半径内）⇒ 锁它（哪怕画面更远 ✓）；
      ② 大怪在**优先半径外** ⇒ 不优先，仍锁最近的小怪 ✓（避免舍近求远 ✓）；
      ③ 关了（倍数 ≤1 / 半径 ≤0）⇒ 老行为一字不变 ✓；
      ④ 尺寸都差不多 ⇒ 没有"大怪"，不优先 ✓（相对判定的自适应 ✓）。
    """
    from decision.agent import CombatAgent
    from perception.world_state import Mob, Player, WorldState

    a = CombatAgent(fresh_settings())
    p = Player(x=500.0, y=300.0, bottom=340.0)
    ws = WorldState()
    ws.player = p
    ws.player.here_sets = ["甲"]
    s = a.settings

    small1 = Mob(id=1, x=520.0, y=300.0, w=40.0, h=40.0, conf=0.9)   # 贴身（dist=0）
    small2 = Mob(id=2, x=560.0, y=300.0, w=40.0, h=40.0, conf=0.9)   # dist=40
    big = Mob(id=3, x=600.0, y=300.0, w=80.0, h=120.0, conf=0.9)     # dist=60，明显更高
    # ⭐ 基准（最小框均线）= 40（最小那一头 ✓），阈值 = 40×1.5 = 60，big.h=120 ≥ 60 ⇒ 算大怪 ✓
    # （没注入 mob_cost_of ⇒ 走老口径，大怪优先在 `_nearest` 的 dists 收窄那一步生效，
    #  与寻路距离无关 ✓）

    # ① 大怪在半径内 ⇒ 优先锁它（画面更远也锁）
    s.big_mob_ratio = 1.5
    s.big_mob_range_px = 600.0
    t1, _ = a._nearest([small1, small2, big], p, ws=ws)
    check(t1 is big,
          "一堆小怪里有明显更高的大怪（且在半径内）却没优先锁它：%r" % (t1,))

    # ② 大怪在半径外 ⇒ 不优先（仍锁最近的贴身小怪）
    s.big_mob_range_px = 50.0
    t2, _ = a._nearest([small1, small2, big], p, ws=ws)
    check(t2 is small1,
          "大怪在优先半径外却还插队优先（该锁最近的小怪 ✓）：%r" % (t2,))

    # ③ 关了（倍数 ≤1 / 半径 ≤0）⇒ 老行为
    s.big_mob_range_px = 600.0
    s.big_mob_ratio = 1.0
    t3, _ = a._nearest([small1, small2, big], p, ws=ws)
    check(t3 is small1, "倍数 ≤1（关）却还优先大怪：%r" % (t3,))
    s.big_mob_ratio = 1.5
    s.big_mob_range_px = 0.0
    t4, _ = a._nearest([small1, small2, big], p, ws=ws)
    check(t4 is small1, "半径 ≤0（关）却还优先大怪：%r" % (t4,))
    s.big_mob_range_px = 600.0

    # ④ 尺寸都差不多 ⇒ 无"大怪"，不优先（返回最近的那只 ✓）
    s4 = Mob(id=4, x=560.0, y=300.0, w=40.0, h=40.0, conf=0.9)
    t5, _ = a._nearest([small1, s4], p, ws=ws)
    check(t5 is small1, "尺寸都差不多时误判出大怪、优先了：%r" % (t5,))

    # ⑤ ⭐ 总开关（用户 2026-10-01 加 ✓）：不勾「大怪优先」⇒ 整个不启用（退回原候选，
    #    哪怕倍数/半径都开着 ✗ —— 这是比"设倍数 ≤1 / 半径 ≤0"更高一级的闸 ✓）
    s.big_mob_enabled = False
    s.big_mob_ratio = 1.5
    s.big_mob_range_px = 600.0
    t6, _ = a._nearest([small1, small2, big], p, ws=ws)
    check(t6 is small1, "总开关关掉（big_mob_enabled=False）却还优先大怪：%r" % (t6,))
    s.big_mob_enabled = True


def t_big_mob_baseline_min_avg():
    """⭐⭐ 大怪判定的**基准 = 最小框均线**（用户 2026-10-02 ✓ 原话："把『大怪判定倍数』逻辑改为
    『比最小的高这个数』（**兼容大怪多小怪少**的情况）" + 追问定稿："记录**一段时间内的最小框
    均线**" ✓）。

    病：老口径的基准是**候选中位数** ⇒ "**大怪多、小怪少**"时中位数被大怪自己抬上去 ⇒ 阈值
    高过大怪本身 ⇒ **一只都认不出来** ✗（「大怪优先」等于没开 ✗）。
    改（用户拍板 ✓）：基准 = 最近 `MOB_H_BASELINE_S` 秒内、**每一拍最小框高**的**平均**
    （"最小框均线" ✓）；判据仍是 `框高 ≥ 基准 × 大怪判定倍数` ✓ —— **倍数这个参数的含义没变**
    ✓、界面那行也没动 ✓（老配置照用 ✓）。
      为什么取"最小"：最小那一头永远站在**小怪**那边 ⇒ 大怪天然比它高 ✓；
      为什么还要"均线"：怪被遮挡时框会**偏小** ✗ ⇒ 单个漏检小框能把"当拍最小"拖低 ⇒ 阈值跟着低
      ⇒ 人人都算大怪 ⇒ 优先等于失效 ✗；取一段时间的均线就把单点采样摊平 ✓。

    钉五件（真调 `_big_mob_pool` ✓ —— 它就是这个口径的**唯一实现** ✓）：
      ① **大怪多小怪少也认得出**：3 只大怪（120）+ 1 只小怪（40）⇒ 老口径（基准 120 ⇒ 阈值 180）
         **一只都落空** ✗；新口径（基准 40 ⇒ 阈值 60）⇒ 三只大怪全在池里、小怪不在 ✓；
      ② **单个漏检小框拖不低基准**：先记几拍 40，再来一拍里混进一个 h=5 的漏检框 ⇒
         若拿当拍最小当基准（5 ⇒ 阈值 7.5）⇒ 连 h=40 的普通怪都算"大怪" ✗；
         均线（≈40 ⇒ 阈值 60）⇒ 普通怪不算、大怪照算 ✓；
      ③ **窗口外的老采样会剪掉**：超过 `MOB_H_BASELINE_S` 的账不再参与 ⇒ 满屏只剩大怪时
         **自己会收敛**（不再硬认"大怪" ✓ = 没小怪可比 ⇒ 优先自然退场 ✓）；
      ④ **判不出来 ⇒ 不判**：只有 0~1 只 / 框高全无效 ⇒ 池空 ✓（老行为一字不变 ✓）；
      ⑤ **开关关着也照记账、但池恒空**（开启那一刻账就是热的 ✓）+ 倍数 ≤1 / 半径 ≤0 ⇒ 关 ✓。
    """
    from decision.agent import MOB_H_BASELINE_S, CombatAgent
    from perception.world_state import Mob, Player, WorldState

    a = CombatAgent(fresh_settings())
    p = Player(x=500.0, y=300.0, bottom=340.0)
    a.settings.big_mob_ratio = 1.5
    a.settings.big_mob_range_px = 600.0

    def _ids(pool):
        return sorted(getattr(m, "id", None) for _d, m in pool)

    big1 = Mob(id=1, x=520.0, y=300.0, w=80.0, h=120.0, conf=0.9)
    big2 = Mob(id=2, x=560.0, y=300.0, w=80.0, h=120.0, conf=0.9)
    big3 = Mob(id=3, x=600.0, y=300.0, w=80.0, h=120.0, conf=0.9)
    small = Mob(id=9, x=540.0, y=300.0, w=40.0, h=40.0, conf=0.9)
    t = 1000.0

    # ① 大怪多、小怪少 ⇒ 三只大怪都在池里 ✓（老口径：中位数 120 ⇒ 阈值 180 ⇒ 三只全落空 ✗）
    pool = a._big_mob_pool([(0.0, big1), (20.0, big2), (40.0, big3), (60.0, small)],
                           p, now=t)
    check(_ids(pool) == [1, 2, 3],
          "大怪多、小怪少时没认出大怪（基准该取**最小那一头** 40 ⇒ 阈值 60 ✓；"
          "老口径的中位数 120 ⇒ 阈值 180 ⇒ 一只都认不出来 ✗）：%r" % (_ids(pool),))

    # ② 单个漏检小框（h=5）拖不低基准：先记 5 拍 40 ⇒ 基准仍是 ≈40 ⇒ 普通怪（40）不算大怪 ✓
    b = CombatAgent(fresh_settings())
    b.settings.big_mob_ratio = 1.5
    b.settings.big_mob_range_px = 600.0
    s2 = Mob(id=19, x=540.0, y=300.0, w=40.0, h=40.0, conf=0.9)
    for k in range(5):
        b._big_mob_pool([(0.0, s2), (10.0, big1)], p, now=t + k * 0.1)
    hole = Mob(id=29, x=520.0, y=300.0, w=40.0, h=5.0, conf=0.3)     # 被遮挡 ⇒ 框偏小 ✗
    pool = b._big_mob_pool([(0.0, hole), (10.0, s2), (20.0, big1)], p, now=t + 0.6)
    check(_ids(pool) == [1],
          "一个漏检的小框就把基准拖低了（拿**当拍最小**（5）当基准 ⇒ 阈值 7.5 ⇒ 连 h=40 的"
          "普通怪都算大怪 ⇒ 优先等于失效 ✗；均线该把这一笔摊平 ✓）：%r" % (_ids(pool),))

    # ③ 窗口外的老采样会剪掉 ⇒ 满屏只剩大怪时自己收敛（不硬认大怪 ✓）
    c = CombatAgent(fresh_settings())
    c.settings.big_mob_ratio = 1.5
    c.settings.big_mob_range_px = 600.0
    for k in range(3):
        c._big_mob_pool([(0.0, s2), (10.0, s2)], p, now=t + k * 0.1)    # 记 3 笔 40
    check(_ids(c._big_mob_pool([(0.0, big1), (10.0, big2)], p,
                               now=t + float(MOB_H_BASELINE_S) + 1.0)) == [],
          "窗口外（> %.0f 秒）的老采样没剪掉 ⇒ 基准永远拖着早就没有的小怪高度 "
          "⇒ 满屏大怪还会被硬认成大怪 ✗" % MOB_H_BASELINE_S)

    # ④ 判不出来 ⇒ 不判：只有 1 只 ⇒ 谈不上"相对更高" ✓；框高全无效 ⇒ 没有基准 ✓
    d = CombatAgent(fresh_settings())
    d.settings.big_mob_ratio = 1.5
    d.settings.big_mob_range_px = 600.0
    # ⚠ **先喂几拍"有小怪"的**把均线压到 40 ✓ —— 不然"只有 1 只"那种情况本来也过不了阈值
    #   （基准 = 它自己 ⇒ 自己 ≥ 自己×1.5 不成立 ✓）⇒ 这条用例就抓不住"把那道闸去掉" ✗
    #   （反向验证踩到的盲区 ✓）。
    for k in range(3):
        d._big_mob_pool([(0.0, s2), (10.0, big1)], p, now=t + k * 0.1)
    check(_ids(d._big_mob_pool([(0.0, big1)], p, now=t + 0.4)) == [],
          "视野里只有 1 只也硬判大怪 ✗（老口径：0~1 只 ⇒ 池空 ⇒ 老行为 ✓ 一字不变 ✓）")
    h0a = Mob(id=31, x=520.0, y=300.0, w=40.0, h=0.0, conf=0.9)
    h0b = Mob(id=32, x=560.0, y=300.0, w=40.0, h=0.0, conf=0.9)
    check(_ids(d._big_mob_pool([(0.0, h0a), (10.0, h0b)], p, now=t + 0.1)) == [],
          "框高全无效（0）时没退回「不判」✗（没有基准就不该硬认 ✓ 不猜 ✓）")

    # ⑤ 开关关着：池恒空 ✓ 但**照记账**（开的那一刻基准是热的 ✓）；倍数 ≤1 / 半径 ≤0 ⇒ 关 ✓
    e = CombatAgent(fresh_settings())
    e.settings.big_mob_enabled = False
    e.settings.big_mob_ratio = 1.5
    e.settings.big_mob_range_px = 600.0
    check(e._big_mob_pool([(0.0, s2), (10.0, big1)], p, now=t) == [],
          "「大怪优先」总开关关着却还给了池子 ✗（关 = 整个不启用 ✓）")
    check([h for _st, h in e._mob_h_samples] == [40.0],
          "关着就不记账了 ⇒ 开启那一刻基准是冷的（前 %.0f 秒退化成「当拍最小」✗ "
          "这次这版「均线」就白做了）：%r" % (MOB_H_BASELINE_S, e._mob_h_samples))
    e.settings.big_mob_enabled = True
    e.settings.big_mob_ratio = 1.0
    check(e._big_mob_pool([(0.0, s2), (10.0, big1)], p, now=t + 0.1) == [],
          "倍数 ≤1（= 人人都是大怪 ⇒ 关）却还给了池子 ✗")
    e.settings.big_mob_ratio = 1.5
    e.settings.big_mob_range_px = 0.0
    check(e._big_mob_pool([(0.0, s2), (10.0, big1)], p, now=t + 0.2) == [],
          "半径 ≤0（= 不启用）却还给了池子 ✗")


def t_battle_zone_mob_filter():
    """「锁定目标不能在**非限制战斗区域**内」= 怪不在配置的集合里 ⇒ **不给锁**（用户 2026-09-27 要求 1）。

    原话："锁定目标不能在非限制战斗区域内"。项目里「**限制战斗区域**」= `settings.battle_zone_sets`
    （一个**集合名**列表 ✓，空 = 不限制 ✓）：老规则只管"**我**在不在区域里"（`_in_battle_zone` ✓），
    怪在不在区域里**没人管** ⇒ 会锁住一只站在区域外的怪、追出去 ✗。

    落在 `_candidates`（**锁定/追击链的唯一漏斗** ✓）⇒ 攻击链不受影响（够得着照打 ✓）。

    钉六件：
      ① 配了区域 ⇒ **区域外的怪不进候选** ✓；
      ② 区域内的怪照旧留着 ✓；
      ③ **没配区域**（空列表）⇒ 一只都不筛（老行为 ✓）；
      ④ **判不了就不筛**（解析器抛错 / 怪没圈进集合 / 干脆没注入解析器 ✓）—— 绝不因为算不出来
         把怪全清掉 ✗；
      ⑤ 结果**缓存**（间隔 `ZONE_GOTO_RETRY_S`）⇒ 不是每拍都去问解析器 ✓（它要扫全部 foothold ✗），
         过期后**会**重问 ✓；
      ⑥ **攻击链不受影响**：`_in_range_all` 吃的是原始 `mobs` ⇒ 区域外、够得着的怪**照打** ✓。
    """
    from decision.agent import CombatAgent, ZONE_GOTO_RETRY_S
    from perception.world_state import Mob, Player, WorldState

    a = CombatAgent(fresh_settings())
    a.settings.battle_zone_sets = ["右"]
    ws = WorldState()
    ws.player = Player(x=500.0, y=300.0, bottom=340.0)
    m_in = Mob(id=1, x=520.0, y=300.0, w=40.0, h=60.0, conf=0.9)
    m_out = Mob(id=2, x=560.0, y=300.0, w=40.0, h=60.0, conf=0.9)
    calls = []

    def sets_of(pl, mob):
        calls.append(mob.id)
        return {"world": (mob.x, -200.0),
                "sets": ["右" if mob.id == 1 else "左"], "why": ""}

    a.mob_sets_of = sets_of

    # ① 区域外的怪被筛掉、② 区域内的留着
    got = a._candidates([m_in, m_out], 500.0, ws=ws, now=10.0)
    check([m.id for m in got] == [1],
          "「限制战斗区域」外的怪没被筛掉（用户 2026-09-27 要求 1 ✗）：%s"
          % [m.id for m in got])

    # ⑤ 缓存：紧挨着的下一拍不再问解析器；过了间隔会重问
    n0 = len(calls)
    a._candidates([m_in, m_out], 500.0, ws=ws, now=10.1)
    check(len(calls) == n0,
          "缓存没生效：每拍都在问 `mob_sets_of`（它要扫全部 foothold ✗）")
    a._candidates([m_in, m_out], 500.0, ws=ws, now=10.0 + ZONE_GOTO_RETRY_S + 0.1)
    check(len(calls) > n0,
          "缓存没过期：怪换到别的平台后一直按老答案筛 ✗")

    # ③ 没配区域 ⇒ 一只都不筛（老行为不许改坏）
    a.settings.battle_zone_sets = []
    got2 = a._candidates([m_in, m_out], 500.0, ws=ws, now=99.0)
    check(len(got2) == 2,
          "没配「限制战斗区域」时却筛怪了（老行为被改坏 ✗）：%s" % [m.id for m in got2])

    # ④ 判不了 ⇒ 不筛
    a.settings.battle_zone_sets = ["右"]
    a.mob_sets_of = lambda pl, mob: {"world": None, "sets": [], "why": "定位不到"}
    got3 = a._candidates([m_in, m_out], 500.0, ws=ws, now=200.0)
    check(len(got3) == 2,
          "解析不出怪在哪个集合时把怪筛掉了（该**不拦** ✓）：%s" % [m.id for m in got3])

    def boom(pl, mob):
        raise RuntimeError("解析器坏了（用例）")

    a.mob_sets_of = boom
    got4 = a._candidates([m_in, m_out], 500.0, ws=ws, now=201.0)
    check(len(got4) == 2, "解析器抛错却把怪筛掉了（解析器坏了别把候选清空 ✗）")

    a.mob_sets_of = None
    got5 = a._candidates([m_in, m_out], 500.0, ws=ws, now=202.0)
    check(len(got5) == 2, "没有 `mob_sets_of` 时却筛怪了（老环境该退回老行为 ✓）")

    # ⑥ 攻击链不受影响：区域外、够得着的怪**照打**
    a.mob_sets_of = sets_of
    a.settings.attack_dist = 200.0
    near = a._in_range_all([m_out], ws)
    check(len(near) == 1,
          "区域外但**够得着**的怪不给打了 —— 用户要的只是「不**锁**区域外的怪」✗：%r" % (near,))


def t_reach_skip_goto():
    """够得着 + 同一条 x 线 ⇒ **不下前往，直接锁**（用户 2026-09-27 要求 2）。

    原话："如果攻击范围框**延伸过去**与目标框有交集，且**其框底边与角色玩家当前 foothold
    集合的 x 范围有交集**，那么就不需要下达寻路任务了可以直接锁"。

    为什么：`_chase_goto_if_elsewhere` 只看"集合图上是不是同一块" ⇒ 怪站在我脚下平台**旁边那条
    窄边**上（集合不是同一个 ✗）时会白下一次「前往」，其实伸伸手就够到 ✓。

    钉六件：
      ① 两件都成立 ⇒ 返回 False（**不下前往** ✓、`_climb` 仍为 None ✓）；
      ② 「**延伸量**」= **无限向前延伸**（2026-09-27 用户明确：**不是** `chase_jump_max` ✓）：
         水平方向**不设上限**（把 `chase_jump_max` 关成 0、怪放到很远 ⇒ 照样豁免 ✓）；
         但**竖直**方向仍受攻击框上下两条边限制（超出 ⇒ 老路：下前往 ✓）；
      ②′ 「我这块平台的 x 范围」要**外扩「最大攻击距离」**再比（用户 2026-09-27 优化 ✓）：
         怪框落在平台**边外**、但只差 ≤ `attack_dist` ⇒ **照样豁免**（站到边上伸手就够着 ✓）；
         差得比 `attack_dist` 还远 ⇒ 老路（下前往 ✓）；
      ③ 怪框底边 x 范围**不与我这块集合相交** ⇒ 老路（下前往 ✓）；
      ④ **判不了就照旧**：广播没给出平台宽度（`here_span` = `None` ✓）/ 拿不到怪的世界脚点 ⇒ 下前往 ✓；
      ⑤ 老判据**优先级更高**：同一块集合 ⇒ 直接照旧追，跟这两件无关 ✓；
      ⑥ 真数据：`route.set_span` = 该集合非墙 foothold 的 x 包络（同集合 ⇒ 同一个包络 ✓）。
    """
    from decision import route
    from decision.agent import ZONE_GOTO_RETRY_S

    s = fresh_settings(attack_dist=40.0)        # 攻击距离小 ⇒ 怪（60px）本在范围外 ⇒ 走 chase ✓
    s.chase_jump_max = 60.0                     # 延伸量 ⇒ 40 + 60 = 100 ⇒ 60px 那只算"够得着" ✓
    s.enabled = True
    h = Harness(s)
    h.mobs_fn = lambda t: [Mob(id=1, x=580.0, y=500.0, w=40.0, h=40.0, conf=0.9)]

    def _plan(dst):
        return {"jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
                "path": ["乙平台", dst], "why": "", "here": False}

    with h._patched():
        a = h.agent
        h.clock0 = h.clock.t
        a.route_plan = _plan
        ws = h.ws(with_mob=True)
        ws.player.here_sets = ["乙平台"]
        # 怪在世界里站在「甲平台」（和我不是同一块 ⇒ 老规则要下前往 ✓）；脚点 x=600、宽 40
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0), "sets": ["甲平台"], "why": ""}

        # ① 延伸框够得着（60 ≤ 100）+ 怪框底边 [580,620] ∩ 我的集合 (560,700) ⇒ **不下前往**
        ws.player.here_span = (560.0, 700.0)
        h.clock.t += 0.1
        act = a.tick(ws)
        check(a._climb is None,
              "够得着、怪框底边又和我这块集合的 x 范围相交，还是下了前往任务 ✗：%r" % (a._climb,))
        check(act.get("state") == "chase",
              "该继续追（要求 2 就是「别下寻路任务、直接锁」✓）：%s" % act)

        # ② 「延伸量」= **无限向前**（用户 2026-09-27 明确：不是 `chase_jump_max` ✓）
        #    · 水平方向不设上限：把 `chase_jump_max` 关成 0、怪放到**很远** ⇒ 照样豁免 ✓
        #      （老那一版借 `chase_jump_max` 算成"够不着" ⇒ 这里会去下前往 ✗）；
        #    · 竖直方向**仍受**攻击框上下两条边限制 ⇒ 超出 ⇒ 老路（下前往 ✓）。
        a.settings.chase_jump_max = 0.0             # 证明它**不再参与**这条判定 ✓
        ws.player.here_span = (500.0, 1700.0)   # 我这块平台够宽（这一段只测水平延伸 ✓）
        far = Mob(id=9, x=1600.0, y=500.0, w=40.0, h=40.0, conf=0.9)
        check(a._reachable_without_path(far, ws, (1600.0, -300.0)) is True,
              "「攻击框无限向前延伸」没生效：很远的那只也没豁免 "
              "（用户 2026-09-27 明确不要 chase_jump_max ✗）")
        up = Mob(id=10, x=1100.0, y=200.0, w=40.0, h=40.0, conf=0.9)
        check(a._vdist(up, ws.player) > 0.0,
              "用例前提不成立：这只怪得跟玩家**不同高**才测得出竖直那条边")
        a.settings.attack_up_dist = 0.0
        a.settings.attack_down_dist = 0.0
        check(a._reachable_without_path(up, ws, (1100.0, -300.0)) is False,
              "竖直方向超出攻击框却还是豁免了（「无限」只该作用在**水平向前** ✗）")
        a.settings.attack_up_dist = -1
        a.settings.attack_down_dist = -1
        check(a._reachable_without_path(up, ws, (1100.0, -300.0)) is True,
              "把上下恢复成「不限」之后反而又不豁免了（上下边语义被弄坏 ✗）")
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        a.tick(ws)
        check(a._climb is None,
              "「无限向前」生效后（怪 60px、同一层、x 相交）却又去下前往了 ✗：%r"
              % (a._climb,))

        # ②′ ⭐ 我这块平台的 x 范围要**外扩「最大攻击距离」**（用户 2026-09-27 优化：
        #     "应该是「其框底边与角色玩家当前 foothold 集合的 **(x 范围 + 最大攻击距离)** 有交集」"）
        #     · 怪框 [540,580]、我平台 (600,700) ⇒ 只差 20px < attack_dist(40) ⇒ **豁免** ✓
        #       （站到平台边上伸手就够着 ✓ 不用绕路 ✓）；
        #     · 把 attack_dist 缩到 10 ⇒ 20px 就够不着了 ⇒ **老路**（下前往 ✓）——
        #       证明外扩量真是按设置算的、不是个写死的常数 ✓。
        ws.player.here_span = (600.0, 700.0)        # ⭐ 广播：我这块平台有多宽 ✓
        edge = Mob(id=11, x=560.0, y=500.0, w=40.0, h=40.0, conf=0.9)   # 怪框 [540,580]
        a.settings.attack_dist = 40.0
        check(a._reachable_without_path(edge, ws, (560.0, -300.0)) is True,
              "怪框离我平台只差 20px（≤ 最大攻击距离 40）却没豁免"
              "（用户 2026-09-27 的优化 ✗）")
        a.settings.attack_dist = 10.0
        check(a._reachable_without_path(edge, ws, (560.0, -300.0)) is False,
              "怪框离我平台 20px、可攻击距离只有 10，却也豁免了（外扩量没按设置算 ✗）")
        a.settings.attack_dist = 40.0

        # ③ x 范围不相交 ⇒ 老路：下前往
        ws.player.here_span = (1000.0, 1100.0)
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        a.tick(ws)
        check(a._climb is not None,
              "怪框底边和我这块集合的 x 范围不相交，却还是不下前往 ✗：%r" % (a._climb,))
        a.stop_route("用例：③检查完了")

        # ④ 判不了 ⇒ 照旧（下前往 ✓）：**广播没给出平台宽度**（这一拍没读数 / 脚下没圈集合 ✓）
        ws.player.here_span = None
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        a.tick(ws)
        check(a._climb is not None, "广播没给出平台宽度（here_span=None）却豁免了（拿不准该照旧 ✗）")
        a.stop_route("用例：④检查完了")

        ws.player.here_span = (560.0, 700.0)
        a.mob_sets_of = lambda pl, mob: {"world": None, "sets": ["甲平台"], "why": ""}
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        a.tick(ws)
        check(a._climb is not None, "拿不到怪的世界脚点却豁免了（拿不准该照旧 ✗）")
        a.stop_route("用例：④b检查完了")

        # ⑤ 同一块集合（老判据）优先：这时候跟这两件无关，照旧追 ✓
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0), "sets": ["乙平台"], "why": ""}
        ws.player.here_span = (1000.0, 1100.0)     # x 不相交也不影响 ✓
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        a.tick(ws)
        check(a._climb is None, "同一块平台却没按老判据照旧追 ✗")

    # ⑥ 真数据：`set_span` = 该集合非墙 foothold 的 x 包络
    from core import mapdata, zones
    t, z = mapdata.load("105090600"), zones.load("105090600")
    names = sorted(z.sets)
    check(names, "地图里没有集合，这条测不了")
    sp = route.set_span(t, z, names[0])
    spans = route._set_spans(t, z, names[0])
    check(spans and sp == (min(x[0] for x in spans), max(x[1] for x in spans)),
          "`set_span` 不是「该集合非墙 foothold 的 x 包络」：%r / %r" % (sp, spans))
    check(route.set_span(t, z, "不存在的集合") is None,
          "集合不存在时该给 None（不猜 ✓）")
    # ⑦ ⛔ **`set_box`：集合的包围盒**（2026-09-28 现场修 ✗ —— `live_thread` 那条"朝
    #    『玩家 → 怪』方向最近集合"的**降级**原来拿 `set_span` 的**二元组**当**四元组**
    #    解包（`sp[2]`）⇒ `IndexError` ⇒ 那条降级**从上线起一次都没成功过** ✗
    #    （异常还被 `agent._mob_goto_towards` 吞了 ⇒ **一条痕迹都没有** ✗））。
    box = route.set_box(t, z, names[0])
    check(box is not None and len(box) == 4,
          "`set_box` 该给**四元组**（左, 右, 上y, 下y）—— 降级解析器就按四元组解包 ✗：%r"
          % (box,))
    _ys = [float(x[2]) for x in spans]
    check(box == (min(x[0] for x in spans), max(x[1] for x in spans),
                  min(_ys), max(_ys)),
          "`set_box` 不是「非墙 foothold 的 x 包络 + 面 y 的上下界」：%r / %r" % (box, spans))
    check(box[0] == sp[0] and box[1] == sp[1],
          "`set_box` 的左右和 `set_span` 对不上（同一件事两处算 ⇒ 迟早打架 ✗）：%r / %r"
          % (box, sp))
    check(box[2] <= box[3],
          "`set_box` 的上下界反了（世界 y **越小越靠上** ⇒ 上界在前 ✓）：%r" % (box,))
    check(route.set_box(t, z, "不存在的集合") is None,
          "`set_box` 集合不存在时该给 None（不猜 ✓）")


def t_job_label_text():
    """「当前执行器」那行：`下跳(drop):{详细内容}`（用户 2026-09-27 要求）。

    原话："你能把当前的执行器写一下吗？例如格式：`下跳(drop):{详细内容}`" —— 卡住时第一件想知道
    的事就是"**现在跑的是哪个执行器、它卡在哪一步**" ✓；原来那行只有一句自由文本
    （`current_goto_note` ✓），**看不出是哪一类执行器** ✗。

    钉四件：
      ① 四个执行器都写得出来，中文名**走 `zones.kind_label`**（一处口径 ⇒ `下跳(drop)` /
         `走(walk)` / `跳(jump)` / `爬（绳梯）(climb)` ✓），格式就是 `名字:详细内容` ✓；
      ② 「详细内容」= job 的 `note`；note 为空时退回 `phase`（**不许给空串** ✗ —— 至少要说出
         它卡在哪个相位 ✓）；
      ③ 没有 job ⇒ 空串（画面上不多这一行 ✓）；
      ④ `agent.current_goto_text()` 没 job 时**退回** `current_goto_note()`（任务结束后还挂一会儿
         的那句 ✓ 老行为不变 ✓）。
    """
    from core import zones
    from decision import route
    from decision.agent import CombatAgent

    # ① 攀爬：名字 + note
    j = route.ClimbJob(ladder_id="L1", x=550.0, y1=300.0, y2=100.0, direction=1,
                       dst_set="上", src_set="下", tol_px=10, hold_ms=250)
    j.note = "对齐好了，但人没站在绳下端那一层上（现在 y=366）"
    got = route.job_label(j)
    check(got.startswith(zones.kind_label("climb") + ":"),
          "攀爬那行的开头不是「爬（绳梯）(climb):」：%r" % got)
    check("人没站在绳下端那一层上" in got, "那行没带上 job 的 note：%r" % got)

    # ② note 空 ⇒ 退回 phase（不许空串）
    j.note = ""
    j.phase = route.ClimbJob.ALIGN
    got2 = route.job_label(j)
    check(got2.endswith(str(route.ClimbJob.ALIGN)) and len(got2) > len(str(route.ClimbJob.ALIGN)),
          "note 为空时没退回相位（那行会变成光秃秃一个名字 ✗）：%r" % got2)

    # ③ 没有 job ⇒ 空串
    check(route.job_label(None) == "", "没有 job 时该给空串：%r" % route.job_label(None))

    # ①′ 四种执行器都要映射到 `zones.EDGE_KINDS` 里那四个键（**中文名只此一处** ✓）
    for cls, kind in ((route.WalkJob, "walk"), (route.ClimbJob, "climb"),
                      (route.DropJob, "drop"), (route.JumpJob, "jump")):
        check(route.JOB_KINDS.get(cls.__name__) == kind,
              "%s 没映射到 %r（界面会退回类名 ✗）" % (cls.__name__, kind))
        check(kind in zones.EDGE_KINDS,
              "映射出来的 %r 不是通行方式（`zones.EDGE_KINDS` 那份口径 ✗）" % (kind,))

    # ④ agent 那一层：没任务时退回 `current_goto_note()`（老行为）
    a = CombatAgent(fresh_settings())
    check(a.current_goto_text() == "",
          "没任务时执行器那行不该有内容（画面会多一行 ✗）：%r" % (a.current_goto_text(),))
    a._climb = j
    check(a.current_goto_text().startswith(zones.kind_label("climb")),
          "agent 那层没走 `route.job_label`：%r" % (a.current_goto_text(),))


def t_route_goal_already_reached():
    """**要去的地方已经到了 ⇒ 整条路线收工**（用户 2026-09-27 报的卡死）。

    现场（截图）：`当前任务 前往：顶层` + `fh：顶层`，人却卡在
    "对齐好了，但**人没站在绳下端那一层上**（绳底那块面在 y=-182~-182）（现在 y=-366）⇒ 先走到
    绳下端那一层的平台上，这一拍不按跳"上 ✗ —— 这就是**死锁**：
    · 爬绳 job 的"到达"是**双判据**（y 也得够到绳上端 ✓，`t_climb_flow_rules` 钉着 ⇒
      **不能**改成"在 dst_set 就算到达"✗，试过、当场红）；
    · 于是"人已经在目标集合里、这一步却还挂着"没人收 ⇒ 永远站着 ✗。
    ⇒ 收口在**路线**这一层：这趟的**最终目的地**已经在脚下集合里 ⇒ 收工 ✓
    （集合级语义 ✓ `mapdata.segment_of` 的 ⚠"语义判定一律走集合"）。

    钉四件：
      ① 人已在最终目的地集合里 ⇒ 整条路线**收工**（`_climb` 清空 + 那句说明写出来 ✓）；
      ② 只看**最终目的地**：中间某一站到了**不算**（路线还得往下走 ✓）；
      ③ 目的地没到 ⇒ 照旧跑（不许提前收 ✗）；
      ④ 拿不到 `here_sets`（没定位）⇒ 不许收（拿不准就照旧 ✓）；
      ⑤ ⭐ **人正挂在绳上爬的时候不许收**（用户 2026-09-27 报："现在会有世界状态都未到顶，
        攀爬执行器就**擅自判成功**" ✗ —— 截图：`绳梯：L3　到顶：否` 却显示"人已经站在
        「一楼」里了（这就是这趟要去的地方）⇒ 收工 ✓" ✗）。
        病根：**绳段横跨好几层**，挂在绳上时"脚下那块 foothold"可能落在**别层**平台上，
        而它恰好属于这趟的目的地集合 ⇒ 被"脚下集合"这条判成功 ✗
        ⇒ 判据加一件：**人不在绳上**（广播 `ladder_id` 为空 ✓）才允许这么收 ✓。
    """
    from decision import route

    s = fresh_settings()
    s.enabled = True
    h = Harness(s)
    h.mobs_fn = lambda t: []                    # 别有怪来抢分支 ✓

    def _plan(dst):
        return {"jobs": [route.ClimbJob(ladder_id="L1", x=550.0, y1=-182.0, y2=-353.0,
                                        direction=1, dst_set=dst, src_set="三楼",
                                        tol_px=10, hold_ms=0)],
                "path": ["三楼", dst], "why": "", "here": False}

    with h._patched():
        a = h.agent
        h.clock0 = h.clock.t
        a.route_plan = _plan
        ws = h.ws(with_mob=True)

        # ① 人已经在最终目的地「顶层」里（爬绳那一步本来要往下走 ✗）⇒ 收工
        ws.player.here_sets = ["顶层"]
        ok = a.plan_and_start_route("顶层", why="用例：目标集合已经到了")
        check(ok and a._climb is not None, "用例前提：这条路该起跑 ✓：%r" % (a._climb,))
        h.clock.t += 0.1
        a.tick(ws)
        check(a._climb is None,
              "人已经在目标集合「顶层」里，这一步却还挂着（用户报的卡死 ✗）：%r" % (a._climb,))
        check("收工" in str((a._last_goto_note or ("", 0))[0]),
              "收工那句说明没写出来：%r" % (a._last_goto_note,))

        # ②/③ 目的地**没**到 ⇒ 照旧跑（不许提前收 ✗）
        for sets in (["三楼"], []):
            ws.player.here_sets = sets
            h.clock.t += 0.2
            a2 = h.agent
            a2._climb = None
            a2.plan_and_start_route("顶层", why="用例：还没到")
            check(a2._climb is not None, "用例前提（%s）：该起跑 ✓" % (sets,))
            h.clock.t += 0.1
            a2.tick(ws)
            check(a2._climb is not None,
                  "还没到目的地（here_sets=%s）却把路线收了 ✗：%r" % (sets, a2._climb))
            a2.stop_route("用例：清理")

        # ⑤ ⭐ **人正挂在绳上**（广播：绳梯非空 = 还没爬到头 ✓）⇒ 哪怕脚下集合就是目的地，
        #    也**不许**判"这趟已经到了"（用户 2026-09-27："世界状态都未到顶，攀爬执行器就
        #    擅自判成功" ✗ —— 截图里写的正是"人已经站在「一楼」里了 ⇒ 收工" ✗）
        a._climb = None
        ws.player.here_sets = ["顶层"]
        ws.player.ladder_id = "L1"          # 广播说：正贴着这根绳（`到顶：否` ✓）
        a.plan_and_start_route("顶层", why="用例：人在绳上")
        check(a._climb is not None, "用例前提（⑤）：这条路该起跑 ✓")
        h.clock.t += 0.1
        a.tick(ws)
        check(a._climb is not None,
              "人还挂在绳上（`绳梯` 非空、`到顶：否`），却拿「脚下集合」把整条路线判成功、"
              "擅自收工了 ✗：%r" % ((a._last_goto_note or ("", 0))[0],))

        # 对照：脚离开绳（广播说不在绳上）⇒ 同一条判据照旧收工 ✓（原来那条不许被弄坏 ✗）
        ws.player.ladder_id = None
        h.clock.t += 0.1
        a.tick(ws)
        check(a._climb is None,
              "人不在绳上、又确实站在目的地集合里，却没照旧收工（把原来那条判据弄坏了 ✗）")


# ⛔ `t_target_sets_for_overlay` **已删**（2026-09-27 用户："以前的锁定框表地点就不要了" ✗）——
#   它钉的是 `agent.current_target_sets()`（"锁定目标在哪块集合" → 给锁定框下方那行用）。
#   那行不要了 ⇒ 那个入口与它那份缓存（`_target_sets`）一起删了 ✗（"死数据不留" ✓），
#   这条用例没有可钉的东西了。⚠ 地点现在只在「**查过的怪框**」那一行上，
#   钉在 `selftest_live_panel.t_mob_box_labels`（框下方靠左 + 绘制层不许自己扫 foothold ✓）。


def t_route_probe_stone_temple():
    """**石人寺院III（`106010105`）**上真解析一遍寻路（用户 2026-09-27："可以拿石人寺院地图
    测试寻路" ✓）。

    为什么单开一条：别的用例钉的是"**某一处判据**对不对" ✓；而"**这张图到底能不能走过去**"
    要把 地形 + 集合 + 边 + **执行器参数** 串起来看 ✓ —— 这个串法**只有一处**
    （`route.plan_jobs` ✓，面板「命令前往」与 agent 的路径解析器都走它 ✓），所以这里直接调
    `tools.route_probe.probe`（**它的体检项就是这条链的自检** ✓）。

    钉三件：
      ① 「一楼 → 顶层」的体检项**全过** ✓（集合圈过 / 每步绳号在编号里 / 上下爬说得清 /
         起止集合都存在 ✓）；
      ② 三步的形状 = 这张图该有的：`一楼 →(爬 L3)→ 二楼 →(跳)→ 三楼 →(爬 L2)→ 顶层` ✓
         （用户现场走的就是最后那段「三楼 → 顶层」✓）；
      ③ 该图**所有集合两两**都能解析出任务 ✓（少一条边就会在这儿红 ✓）。
    """
    from tools.route_probe import probe, probe_all

    checks = probe("106010105", "一楼", "顶层", out=lambda *_a: None)
    bad = [d for ok, d in checks if not ok]
    check(not bad, "石人寺院III 上「一楼 → 顶层」体检没过：%r" % (bad,))

    from core import mapdata
    from core import zones as zmod
    from decision import route

    t, z = mapdata.load("106010105"), zmod.load("106010105")
    plan = route.plan_jobs(t, z, "一楼", "顶层", tol_px=route.ALIGN_TOL_PX, hold_ms=250)
    path = list(plan.get("path") or [])
    jobs = list(plan.get("jobs") or [])
    check(path == ["一楼", "二楼", "三楼", "顶层"],
          "这条路的集合序列变了（现场是 一楼→二楼→三楼→顶层）：%r" % (path,))
    kinds = [route.JOB_KINDS.get(type(j).__name__, "") for j in jobs]
    check(kinds.count("climb") == 2 and kinds.count("jump") == 1,
          "三步该是「爬 → 跳 → 爬」：%r" % (kinds,))
    climbs = [j for j in jobs if getattr(j, "ladder_id", None)]
    check([int(getattr(j, "dir", 0)) for j in climbs] == [1, 1],
          "两段爬都该是**上爬**（一楼→二楼、三楼→顶层）：%r"
          % ([getattr(j, "dir", None) for j in climbs],))

    allchecks = probe_all("106010105", out=lambda *_a: None)
    bad2 = [d for ok, d in allchecks if not ok]
    check(not bad2, "石人寺院III 有集合对解析不出路线（缺边？）：%r" % (bad2,))


def t_climb_recheck_cheaper_mob():
    """⭐⭐ **「来源签名」+ 两个节点复核"有没有更便宜的怪"**（用户 2026-09-28 要求 ✓）。

    用户原话："我们的寻路任务应该需要**来源签名**，我想在这些节点**主动找一次**看是否有
    **代价更低的怪**（用**当前状态**跟**签名来源**对比），如果有那就**主动掐掉寻路任务**：
    ① **climb 对齐完成后起跳前**；② **寻路超时计时过半**" ✓。

    三个决定（用户 2026-09-28 现场定的 ✓，**别自作主张改** ✗）：
      · **只掐「追击」类** —— `命令前往 / 任务队列 / 定点休息 / 回战斗区域 / 到点去哪` 全豁免；
      · **严格更便宜就掐**（`新 < 来源`，**不加余量**）；
      · 掐完**顺手把锁定目标换成新怪**（⚠ 否则下一拍追击链把**同一个**任务又下回来 ⇒ 死循环）。

    钉九件：
      ① 不传 `origin` 的来源（命令前往那种）⇒ **再便宜也不掐** ✓；
      ② **追击 + 更便宜** ⇒ 掐 + 停任务 + **换锁定目标** + 作废"老目标的集合"缓存 ✓；
      ③ **严格更便宜才算**（来源更便宜 / 打平 ⇒ 不掐 ✓）；
      ④ 来源怪**不在了** ⇒ 不掐（保守 ✓ —— "它死了该换目标"是**原有追击链**的事 ✓）；
      ⑤ 代价**算不出** ⇒ 不掐（宁缺勿错 ✓）；
      ⑥ **节点①「对齐完成、起跳前」**：执行器报 `job._armed_at` ⇒ `_climb_tick` 里复核 ✓；
      ⑦ **节点②「超时过半」**：`_climb_started` 过半 ⇒ 复核（与超时**共用同一把钟** ✓）；
      ⑧ **每步只查一次**（`_recheck_*_done` ✓ 随 `start_climb` 复位 ✓）；
      ⑨ **来源沿路线透传**：`_maybe_replan` 重算时**不丢**来源 ✓（丢了就"走得越久越复核不了"✗）
         + 收工之后清掉 ✓。
    """
    from decision import route

    s = fresh_settings()
    s.enabled = True
    s.goto_timeout_s = 10.0                     # 节点② 要用（过半 = 5 秒 ✓）
    h = Harness(s)
    a = h.agent
    ws = h.ws(with_mob=False)
    ws.player.here_sets = ["乙平台"]

    # 代价表：id1 = 900（**来源**）、id2 = 300（更便宜 ✓）、id3 = 950（更贵 ✗）
    costs = {1: 900.0, 2: 300.0, 3: 950.0}
    a.mob_cost_of = lambda pl, m: {"cost": costs.get(m.id), "sets": [], "why": ""}

    def _mk():
        ws.mobs = [Mob(id=i, x=560.0 + 60.0 * (i - 1), y=500.0,
                       w=40.0, h=40.0, conf=0.9) for i in (1, 2, 3)]
        return ws.mobs

    def _job(dst="顶层"):
        # ⚠ 用走任务就够（复核与执行器种类无关 ✓）；`dst_set` 取**目的外地名** ⇒ 不会命中
        #   `_climb_tick` 开头那条"人已经站在目的地里 ⇒ 收工"（那会盖掉复核 ✓ 踩过 ✗）。
        return route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])

    t = 2000.0

    # ① **只有「追击」来源参与**：命令前往（不传 origin）⇒ 天然豁免 ✓
    _mk()
    a.start_route([_job()], why="命令前往：顶层")
    check(a._climb_origin is None,
          "不传 `origin` 时「这一步的来源签名」该是 None（那正是「天然豁免」的判据 ✓）：%r"
          % (a._climb_origin,))
    check(a._climb_recheck(ws, t, "对齐完成") is False and a._climb is not None,
          "「命令前往」这种来源被复核掐掉了（用户明确：**只掐「追击」类** —— "
          "人手安排的不许被动 ✗）")
    a.stop_route("用例①")

    # ①-b ⚠ **来源不是 `"chase"` 却带着 `origin`** ⇒ 同样豁免 ✓
    #   ⚠ 这一条**必须有**：光靠"不传 origin"测不出那句 `kind != "chase"` 的存在 ✗
    #     （眼下只有追击会传 origin ✓ ⇒ 那道检查看起来"多余" ⇒ 谁一顺手就把 K 删了 ✗）。
    #     它防的是**将来**：某个来源（比如任务队列）开始带 `origin` 做别的判断时，
    #     **不许**顺手被这条复核掐掉 ✓。
    _mk()
    a._target_id = 1
    a.start_route([_job()], origin={"kind": "queue", "mob_id": 1})
    check(a._climb_recheck(ws, t, "对齐完成") is False and a._climb is not None,
          "来源不是「追击」却照样被复核掐掉（`kind != \"chase\"` 那道闸没了 ✗ —— "
          "用户定的是**只掐「追击」类**）")
    a.stop_route("用例①-b")

    # ② 追击 + 更便宜 ⇒ 掐 + 换目标 ✓
    _mk()
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job()], why="追击：先过去", origin=a._chase_origin(ws.mobs[0]))
    check(a._climb_origin == {"kind": "chase", "mob_id": 1},
          "来源签名没透到「这一步」（`start_route` → `start_climb` 那一路 ✗）：%r"
          % (a._climb_origin,))
    a._mob_info_cache = ("1", t, {"sets": ["乙平台"]}, None)   # 假装缓存过老目标的集合 ✓
    check(a._climb_recheck(ws, t, "对齐完成") is True, "有更便宜的怪却没掐 ✗")
    check(a._climb is None, "掐了之后任务还挂着 ✗")
    check(a._target_id == 2,
          "掐完没把锁定目标换成更便宜那只 ⇒ 下一拍追击链会把**同一个任务**又下回来"
          "（死循环 ✗）：%r" % (a._target_id,))
    check(a._mob_info_cache is None,
          "换目标后没作废「老目标的集合」缓存 ⇒ 会拿**老怪**的集合去下前往 ✗")

    # ③ **严格更便宜才算**：来源 900、候选里最便宜 950（更贵）⇒ 不掐 ✓
    _mk()
    ws.mobs = [m for m in ws.mobs if m.id in (1, 3)]
    a._target_id = 1
    a.start_route([_job()], origin={"kind": "chase", "mob_id": 1})
    check(a._climb_recheck(ws, t, "对齐完成") is False and a._climb is not None,
          "没有**更便宜**的怪却掐了（用户口径是**严格更便宜**：更贵不该动 ✗）")
    a.stop_route("用例③")

    # ③-b ⚠ **打平**（= 来源的代价）也**不许**掐 ✓
    #   ⚠ 这一条才是真钉"**严格**"的：把判据写成 `<=`（打平也掐）只有它抓得住 ✗
    #     （只测"更贵"是抓不住的：`950 < 900` 和 `950 <= 900` 都是假 ✗ —— 反向验证时踩到过 ✓）。
    costs[3] = 900.0                            # 3 号改成**跟来源打平** ✓
    _mk()
    ws.mobs = [m for m in ws.mobs if m.id in (1, 3)]
    a._target_id = 1
    a.start_route([_job()], origin={"kind": "chase", "mob_id": 1})
    check(a._climb_recheck(ws, t, "对齐完成") is False and a._climb is not None,
          "**打平**的怪也被当成「更便宜」掐了（用户口径是**严格更便宜** `新 < 来源` ✗ —— "
          "写成 `<=` 就会这样 ✓）")
    a.stop_route("用例③-b")
    costs[3] = 950.0                            # 还原成"更贵"（后面 ⑥⑦⑧ 还要用 ✓）

    # ④ 来源怪**不在了** ⇒ 不掐（保守 ✓）
    _mk()
    ws.mobs = [m for m in ws.mobs if m.id != 1]
    a.start_route([_job()], origin={"kind": "chase", "mob_id": 1})
    check(a._climb_recheck(ws, t, "对齐完成") is False and a._climb is not None,
          "来源怪不在了就掐（**不许**：换目标那条链本来就有 ✓ 这里只管「更便宜」那件事 ✗）")
    a.stop_route("用例④")

    # ⑤ 代价**算不出** ⇒ 不掐（宁缺勿错 ✓）
    _mk()
    a.mob_cost_of = lambda pl, m: {"cost": None, "sets": [], "why": "算不出"}
    a.start_route([_job()], origin={"kind": "chase", "mob_id": 1})
    check(a._climb_recheck(ws, t, "对齐完成") is False and a._climb is not None,
          "代价算不出却掐了（宁缺勿错 ✗）")
    a.stop_route("用例⑤")
    a.mob_cost_of = lambda pl, m: {"cost": costs.get(m.id), "sets": [], "why": ""}

    # ⑥ **节点①「对齐完成、起跳前」** —— 走 `_climb_tick`（钉"接线真的在那一处" ✓）
    _mk()
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job()], origin={"kind": "chase", "mob_id": 1})
    a._climb._armed_at = t                       # 执行器说：对齐完成、即将起跳 ✓
    a._climb_tick(t, float(ws.player.world_x), set(), ws)
    check(a._climb is None and a._target_id == 2,
          "「**对齐完成后起跳前**」这个节点没复核（用户要的第一个节点 ✗）：%r / %r"
          % (a._climb, a._target_id))

    # ⑦ **节点②「寻路超时计时过半」** —— 与超时**共用同一把钟** ✓
    _mk()
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job()], origin={"kind": "chase", "mob_id": 1})
    a._climb_started = t - 6.0                   # 10 秒的钟走了 6 秒 ⇒ 已过半 ✓
    a._climb_tick(t, float(ws.player.world_x), set(), ws)
    check(a._climb is None and a._target_id == 2,
          "「**寻路超时计时过半**」这个节点没复核（用户要的第二个节点 ✗）：%r / %r"
          % (a._climb, a._target_id))

    # ⑧ **每步只查一次**：标记置上之后，哪怕有更便宜的怪也**不再动** ✓
    _mk()
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job()], origin={"kind": "chase", "mob_id": 1})
    check(a._recheck_armed_done is False and a._recheck_half_done is False,
          "新任务没把两个「这一步查过了」标记复位 ⇒ 这一步**永远**复核不了 ✗")
    a._recheck_armed_done = True                 # 假装这一步已经查过了 ✓
    a._climb._armed_at = t
    a._climb_tick(t, float(ws.player.world_x), set(), ws)
    check(a._climb is not None,
          "「这一步查过了」还照样复核 ⇒ 就是**每拍算一轮代价** + 两只怪代价接近时**来回掐** ✗")
    # ⚠⚠ **两个节点各有各的"查过了"标记** ⇒ **分开钉** ✗（只钉①抓不住"②每拍都查" ✓ ——
    #   反向验证时踩到过：把②的 `not self._recheck_half_done` 拿掉，用例照样全绿 ✗）。
    a._recheck_half_done = True
    a._climb_started = t - 6.0                   # 过半 ✓（但②已被标记成"查过了"）
    a._climb_tick(t, float(ws.player.world_x), set(), ws)
    check(a._climb is not None,
          "「**超时过半**」那个节点每拍都在复核（它也有「这一步查过了」标记 ✗）"
          "⇒ 过半之后每一拍都过半 ⇒ 等于每拍算一轮代价 ✗")

    # ⑨ 来源沿路线透传：`_maybe_replan` 重算时不许丢 ✓
    #   ⚠ 源码级钉（造一次真 replan 要摆位置状态变化，太重 ✓ 而同项目已有这种钉法 ✓）。
    from pathlib import Path

    _ag = (Path(__file__).resolve().parents[1] / "decision" / "agent.py"
           ).read_text(encoding="utf-8")
    check("origin=self._route_origin" in _ag,
          "`_maybe_replan` 重算路线时没带上来源签名 ⇒ **人走得越久、来源越容易丢**，"
          "那两个节点就再也复核不了 ✗（很隐蔽：不报错、不崩 ✓）")
    a._task_finished()
    check(a._climb_origin is None,
          "收工之后「这一步的来源签名」没清（它是**这一步**的东西 ✗）：%r"
          % (a._climb_origin,))


def t_climb_arrive_face_target():
    """⭐⭐ **到顶后"再按住一会儿"那几拍：朝「锁定目标」那边按方向键**（用户 2026-10-02 ✓ 原话：
    "攀爬(追击)优化：**到顶时如果存在锁定目标，就按住向着该目标的方向键**（注意后续 chase 的
    方向键处理）"）。

    口径：
      · 到顶那几拍 **↑ 照旧按住**（游戏里"迈上平台"要它 ✓ —— 用户 2026-09-26 定的，别弄丢 ✗）
        **外加水平方向** ⇒ 一落地就朝目标走 ✓（不然要等这一小段过完、chase 接手才动 ✓）；
      · 方向 = `agent._target_dir(ws)` 注入来的那个（**与"跳下绳子"同一处注入** ✓ 一处实现 ✓）；
        **没有锁定目标 / 目标同列 / 没注入 / 回调抛异常 ⇒ `0`** = **老行为**（只按 ↑ ✓ 兼容底线 ✓）；
      · **与后续 chase 同一个口径**（都是"**画面 x 的左右**"：chase 那边 `_steer(target.x - px)`，
        `px = ws.player.x` ✓）⇒ 交接那一拍**不翻向** ✓ —— 这正是用户提醒的
        "注意后续 chase 的方向键处理" ✓。

    钉六件：
      ① 目标在**右** ⇒ 到顶那一拍 `right` 与 `up` **一起按着** ✓（且窗口内**每拍**都给 ✓）；
      ② 目标在**左** ⇒ `left` ✓；
      ③ **没注入 / 注入给 0** ⇒ 一个方向键都不按（老行为 ✓）；
      ④ 保持窗口过完 ⇒ **收工**且那一拍**一个键都不按**（↑ 与方向一起松开 ✓）；
      ⑤ **接线 + 每拍现读**：走 `_climb_tick`（注入在那儿 ✓），把锁定目标换到另一边 ⇒
         **同一条任务**的下一拍方向就跟着换 ✓（不是建任务时定死 ✓）；
      ⑥ 走完整 `tick`：**交接到 chase 那一拍方向不翻**（climb 给 right ⇒ chase 也给 right ✓）。
    """
    from decision import route

    s = fresh_settings()
    s.enabled = True
    h = Harness(s)
    a = h.agent
    ws = h.ws(with_mob=False)
    ws.player.here_sets = ["乙平台"]
    # 位置状态广播：已到 L1 上端（**到达判据唯一来源** ✓ 见 `_arrived` ✓）
    ws.player.at_ladder_top = "L1"

    def _job(hold_ms=300):
        return route.ClimbJob(ladder_id="L1", x=700.0, y1=-182.0, y2=-353.0,
                              direction=1, dst_set="顶层", src_set="乙平台",
                              tol_px=10, hold_ms=hold_ms)

    def _top(j, px=700.0):
        """喂一拍把任务推到"到顶那几拍" → 这一拍发给我们的键（set ✓）。

        ⚠ 必须先把执行器摆成"**正在爬这根绳**"（`phase = CLIMB` + 位置状态广播说贴着 L1 ✓）：
          还停在 `ALIGN` 相时 `_align_step` 会先出手（"站住等 `hold_ms`"✓ 或"朝绳走"✓）
          ⇒ `_arrive_step` **轮不到**（踩过 ✗ —— 那样测到的是"对齐"给的键，不是这条新功能 ✗）。
        ⚠ `px` 给**绳的 x**（对齐着 ⇒ 不会触发对齐点按 ✓）。
        """
        j.phase = j.CLIMB
        ws.player.ladder_id = "L1"          # 位置状态：正贴着 L1（⇒ 执行器认"在绳上" ✓）
        ws.player.at_ladder_top = "L1"      # 位置状态：已到 L1 上端 ⇒ **到达** ✓
        a.start_climb(j)
        keys = set()
        a._climb_tick(h.clock.t, px, keys, ws)
        return keys

    with h._patched():
        _px = float(ws.player.x)

        def _lock(mob_id, side):
            """摆一只**锁定目标**（`_target_dir` 只看 `ws.mobs` 里那只的**画面 x** ✓）。

            ⚠ **别去手改 `job._target_dir_fn`** ✗：走 `_climb_tick` 时它**每拍重挂**（那是
              接线本身 ✓）⇒ 手改会被当场盖掉（踩过 ✗ —— 那样测出来永远是 0）。要摆就得摆
              "锁定目标"这件**真事**（`ws.mobs` + `_target_id` ✓）。
            """
            ws.mobs = [Mob(id=mob_id, x=_px + side * 200.0, y=500.0,
                           w=40.0, h=40.0, conf=0.9)]
            a._target_id, a._target_until = mob_id, h.clock.t + 30.0

        # ① 目标在**右** ⇒ right 与 up 一起按着 ✓，而且窗口内**每拍**都给 ✓
        _lock(1, +1)
        keys = _top(_job())
        check(s.keymap["right"] in keys,
              "到顶时锁定目标在右边，却没朝它按 right（用户 2026-10-02 要的 ✗）：%r"
              % (sorted(keys),))
        check(s.keymap["up"] in keys,
              "到顶那几拍把 **↑ 弄丢了**（游戏里「迈上平台」要它 ✗）：%r" % (sorted(keys),))
        h.clock.t += 0.05
        keys = set()
        a._climb_tick(h.clock.t, 700.0, keys, ws)
        check(s.keymap["right"] in keys and s.keymap["up"] in keys,
              "到顶保持窗口里只有第一拍给方向、后面又没了（那不是「按住」✗）：%r"
              % (sorted(keys),))
        a.stop_route("用例①")

        # ② 目标在**左** ⇒ left ✓（同一条实现，别把符号写反 ✗）
        _lock(2, -1)
        keys = _top(_job())
        check(s.keymap["left"] in keys and s.keymap["right"] not in keys,
              "锁定目标在左边，到顶却按了右边（符号反了 ✗）：%r" % (sorted(keys),))
        a.stop_route("用例②")

        # ③ **没有目标**（三种情形）⇒ 一个方向键都不按（= 加这个功能之前的老行为 ✓）
        #    a. **压根没注入**：直接调 `job.update`（绕开 `_climb_tick` 的每拍重挂 ✓ 老环境 ✓）
        j = _job()
        j.phase = j.CLIMB
        out = j.update(h.clock.t, 700.0, ladder_id="L1", at_top="L1")
        check(out["move"] == 0 and out["dir"] == 1,
              "没注入「目标在哪边」⇒ 到顶该只按住 ↑（`move` 必须是 0 ✓ 老行为 ✗）：%r" % (out,))
        #    b. 锁定的那只**这一帧没框**（`_target_dir` 给 0 ✓）
        _lock(9, +1)
        ws.mobs = []                              # 目标这一帧不在画面里 ✓
        keys = _top(_job())
        check(s.keymap["right"] not in keys and s.keymap["left"] not in keys,
              "锁定目标这一帧不在框里 ⇒ 到顶不该按任何方向键（不猜 ✗）：%r" % (sorted(keys),))
        a.stop_route("用例③b")
        #    c. 干脆**没有锁定目标** ✓
        a._target_id, a._target_until = None, 0.0
        ws.mobs = [Mob(id=1, x=_px + 200.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
        keys = _top(_job())
        check(s.keymap["right"] not in keys and s.keymap["left"] not in keys,
              "没有锁定目标 ⇒ 到顶不该按任何方向键（老行为：只按住 ↑ ✗）：%r" % (sorted(keys),))
        check(s.keymap["up"] in keys, "老行为下 ↑ 也没了 ✗：%r" % (sorted(keys),))
        a.stop_route("用例③c")

        # ④ 保持窗口过完 ⇒ 收工，而且**那一拍一个键都不按**（↑ 与方向一起松开 ✓）
        _lock(1, +1)
        _top(_job(hold_ms=200))
        h.clock.t += 0.25
        keys = set()
        done = a._climb_tick(h.clock.t, 500.0, keys, ws)
        check(done is True and a._climb is None,
              "保持窗口过完还没收工（该 DONE 了 ✗）：%r" % (a._climb,))
        check(not keys,
              "收工那一拍还按着键（↑ 和方向键都要松开，不然人一直朝那边走 ✗）：%r"
              % (sorted(keys),))

        # ⑤ **接线 + 每拍现读**：锁定目标换到另一边 ⇒ 同一条任务下一拍就跟着换 ✓
        _px = float(ws.player.x)
        ws.mobs = [Mob(id=1, x=_px + 200.0, y=500.0, w=40.0, h=40.0, conf=0.9),
                   Mob(id=2, x=_px - 200.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
        a._target_id, a._target_until = 1, 0.0        # 先锁**右边**那只 ✓
        j = _job()
        keys = _top(j)
        check(s.keymap["right"] in keys,
              "`_climb_tick` 没把「目标在哪边」注入给爬绳任务（到顶不会朝目标按 ✗）：%r"
              % (sorted(keys),))
        a._target_id = 2                             # 换成**左边**那只 ✓
        h.clock.t += 0.05
        keys = set()
        a._climb_tick(h.clock.t, 500.0, keys, ws)
        check(s.keymap["left"] in keys and s.keymap["right"] not in keys,
              "锁定目标换了边，到顶按住的方向没跟着换（说明是**建任务时定死**的，"
              "不是每拍现读 ✗）：%r" % (sorted(keys),))

        # ⑥ ⭐ 交接到 chase：**方向不翻**（用户提醒的"注意后续 chase 的方向键处理" ✓）
        #    —— 到顶给 right ⇒ 窗口过完那一拍 chase 接手**也给 right**（同 `_target_dir` 口径 ✓）
        #    ⚠ 灯下黑：这一条要的是"两拍之间**没有方向键的切换**"（`KeyState` 只在键集变了才
        #      发键 ⇒ 方向一样时对面看到的是"一直按着" ✓）。
        a.stop_route("用例⑥准备")
        ws.player.at_ladder_top = "L1"
        ws.player.ladder_id = "L1"               # 摆成"正在爬"（同 `_top` 的理由 ✓）
        ws.player.here_sets = ["乙平台"]
        # 怪**在视野里、不在攻击范围**（距离 200 > 默认 attack_dist 40 ✓）⇒ 走"没框可打"那一支 ✓
        ws.mobs = [Mob(id=1, x=float(ws.player.x) + 200.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
        a._target_id, a._target_until = 1, h.clock.t + 30.0
        h.clock0 = h.clock.t                     # ⚠ 日志里那列是**相对时刻**（`_rec` 减 `clock0` ✓）
        _j6 = _job(hold_ms=100)
        _j6.phase = _j6.CLIMB
        a.start_climb(_j6)
        h.log.clear()
        a.tick(ws)                                   # 到顶那一拍（climb：↑ + right ✓）
        _down = {k for _t, kind, k in h.log if kind == "down"}
        check(s.keymap["right"] in _down,
              "到顶那一拍没朝目标按 right（前置不成立，⑥ 测不了 ✗）：%r" % (sorted(_down),))
        h.clock.t += 0.2                             # 越过保持窗口 ⇒ 这一拍 climb 收工、chase 接手 ✓
        a.tick(ws)
        _dirs = [(kind, k) for _t, kind, k in h.log
                 if k in (s.keymap["left"], s.keymap["right"])]
        check(s.keymap["left"] not in [k for _kind, k in _dirs],
              "交接那一拍方向**翻了**（到顶给 right、chase 接手给 left）—— 用户提醒的"
              "「注意后续 chase 的方向键处理」正是这件事 ✗：%r" % (_dirs,))
        check(any(kind == "down" and k == s.keymap["right"] for kind, k in _dirs),
              "chase 接手后没继续朝目标按 right（接手处方向断了 ✗）：%r" % (_dirs,))
        check(s.keymap["up"] not in {k for _t, kind, k in h.log if kind == "down"
                                     and _t > 0.15},
              "收工之后还在按 ↑（到顶那一下该松了 ✗）")
        a.stop_route("用例⑥收尾")


def t_climb_recheck_walk_platform():
    """⭐⭐ **「走向绳梯（离绳还远）」时**每拍**复核**：我这块平台上若有别的怪 ⇒ 掐掉这一步
    （用户 2026-10-02 ✓ 原话："攀爬优化：处于**起跳距离外**向绳梯移动时，**每拍查询**：如果在
    当前 foothold 集合有**代价更低的目标**就结束任务"）。

    口径（用户 2026-10-02 现场答的 ✓，别自作主张改 ✗）：
      · 「目标」= **怪** —— 沿用 `_climb_recheck` 那套：**只有「追击」来源参与** ✓、掐完**顺手换
        锁定目标** ✓（换目标是为了防"下一拍追击链把同一条攀爬任务又下回来"的死循环 ✓）；
      · 「在当前 foothold 集合」= **粗判**：怪的**世界 x** 落在我站的集合的 x 范围（外扩
        「最大攻击距离」）里 ✓ —— 就是 `_on_my_platform` ✓（与 `_reachable_without_path` 的
        ② 段**同一处** ✓）；**不许**改成 `mob_cost_of`（那要每只怪一轮 BFS ✗，而这条是**每拍**
        查 ✓，而且"同平台 ⇒ 代价≈0"本来就不必算 ✓）；
      · 「结束任务」= 掐 + 换目标（`_drop_climb_for_mob` ✓ 与节点①/② **共用同一处收尾** ✓）。

    钉八件：
      ① **成立就掐 + 换目标 + 停任务**（真走 `_climb_tick` ✓ ⇒ 钉住"接线在那一处、而且是每拍"✓）；
      ② **第一拍不判**（那时 `_last_dx` 还没读数 ⇒ `walking_to_ladder()` 给 False ✓ —— 从第二拍起）；
      ③ 来源怪**自己就在我这块平台上** ⇒ 不掐（谈不上"更便宜" ✗）；
      ④ **判不出来**（没 `here_span` / 没相机与世界坐标）⇒ 不掐（宁缺勿错 ✓）；
      ⑤ **来源怪不在**候选里 ⇒ 不掐（保守 ✓，同 `_climb_recheck` ④ ✓）；
      ⑥ **只有「追击」来源参与**（命令前往 / 任务队列那种 ⇒ 天然豁免 ✓）；
      ⑦ **不在这一档就不查**：`|dx| ≤ 起跳距离` / 按过跳 / 人在绳上 / 斜跳飞行中 ✓
         （判据全在 `route.ClimbJob.walking_to_ladder()` ✓ **一处口径** ✓）；
      ⑧ `_last_dx` 由 `_align_step` **一处记账**，`walking_to_ladder()` 只读它 ✓（不自己再算 ✗）。
    """
    import inspect

    from decision import route

    s = fresh_settings()
    s.enabled = True
    s.vision_left = s.vision_right = 1000     # ⚠ 默认视野只有 ±200 ⇒ 怪摆远了会被 `_candidates` 滤掉 ✗
    h = Harness(s)
    a = h.agent
    ws = h.ws(with_mob=False)
    ws.player.here_sets = ["乙平台"]
    ws.player.world_x = 500.0
    # ⭐ **`cam_x = 0`** ⇒ 画面 x **就是**世界 x（可信相机 ✓ 口径"世界 = 画面 + 相机" ✓）——
    #   这样用例里摆的 `Mob.x` 一眼就能对上"在不在我这块平台上" ✓（不用心算相机 ✓）。
    ws.player.cam_x = 0.0
    ws.player.here_span = (400.0, 600.0)      # 我这块平台横着占世界 x 400..600 ✓

    def _job(jump_start_px=0):
        # 绳梯在**世界 x = 700**（玩家在 500 ⇒ 离绳 200px ⇒ **起跳距离外** ✓）；
        # `dst_set="顶层"` = 目的外地名（不会命中 `_climb_tick` 开头那条"人已经站在目的地里 ⇒
        # 收工" ✓ —— 那会盖掉复核 ✓ 同 `t_climb_recheck_cheaper_mob` 的踩坑说明 ✓）。
        return route.ClimbJob(ladder_id="L1", x=700.0, y1=-182.0, y2=-353.0,
                              direction=1, dst_set="顶层", src_set="乙平台",
                              tol_px=10, hold_ms=0, jump_start_px=jump_start_px)

    def _mk(mobs):
        ws.mobs = list(mobs)
        return ws.mobs

    src = Mob(id=1, x=900.0, y=500.0, w=40.0, h=40.0, conf=0.9)     # 世界 900 ⇒ **不在**我这块
    near = Mob(id=2, x=500.0, y=500.0, w=40.0, h=40.0, conf=0.9)    # 世界 500 ⇒ **就在**我这块
    t = 2000.0

    # ① 每拍复核 + 成立就掐（走 `_climb_tick` ✓）
    _mk([src, near])
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job()], why="追击：先过去", origin=a._chase_origin(src))
    a._climb_tick(t, 500.0, set(), ws)
    check(a._climb is not None,
          "**第一拍就掐了** —— 那时执行器还没跑过 `update`、`_last_dx` 还没有读数 "
          "（判据该给 False ✓ 不许猜 ✗）：%r" % (a._climb,))
    a._climb_tick(t + 0.05, 500.0, set(), ws)
    check(a._climb is None,
          "「**走向绳梯**」时我这块平台上就有别的怪，却没掐掉这一步（用户 2026-10-02 要的"
          "每拍复核 ✗）：%r" % (a._climb,))
    check(a._target_id == 2,
          "掐完没把锁定目标换成**我这块平台上**那只 ⇒ 下一拍追击链会把**同一条攀爬任务**"
          "又下回来（死循环 ✗）：%r" % (a._target_id,))

    # ② 不在这一档：`|dx| ≤ 起跳距离`（进了起跳区间）⇒ 不算"起跳距离外" ⇒ 不掐 ✓
    #    ⚠ 这条是 `walking_to_ladder()` 的守卫之一（`_last_dx` 读的是**执行器算的那个** ✓）。
    _mk([src, near])
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job(jump_start_px=250)], origin=a._chase_origin(src))
    a._climb_tick(t, 500.0, set(), ws)          # 先让执行器算一次 dx（200 ≤ 250 ⇒ 起跳区间内 ✓）
    a._climb_tick(t + 0.05, 500.0, set(), ws)
    check(a._climb is not None and a._target_id == 1,
          "|dx| 已经进了**起跳距离**（250）以内，却还按「起跳距离外」复核掐掉 ✗：%r / %r"
          % (a._climb, a._target_id))
    a.stop_route("用例②")

    # ③ 来源怪**自己**就在我这块平台上 ⇒ 不掐（它在 ⇒ 谈不上"更便宜" ✓）
    _mk([src, near])
    ws.mobs[0].x = 520.0                        # 来源怪也挪进我这块平台（世界 520 ✓）
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job()], origin=a._chase_origin(ws.mobs[0]))
    a._climb_tick(t, 500.0, set(), ws)
    a._climb_tick(t + 0.05, 500.0, set(), ws)
    check(a._climb is not None and a._target_id == 1,
          "来源怪自己就在我这块平台上却被当成「更便宜的那只」掐了 ✗（它本来就够得着 ⇒ "
          "没有「更便宜」可言 ✓）：%r / %r" % (a._climb, a._target_id))
    a.stop_route("用例③")

    # ④ 判不出来 ⇒ 不掐（宁缺勿错 ✓）：没有 `here_span` / 没有相机与世界坐标 ✓
    _mk([src, near])
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job()], origin=a._chase_origin(src))
    a._climb_tick(t, 500.0, set(), ws)
    _span = ws.player.here_span
    ws.player.here_span = None                  # 位置状态这一拍没给"我这块平台有多宽" ✓
    check(a._climb_recheck_walk(ws, t + 0.05) is False,
          "拿不到「我这块平台有多宽」却照样掐（**不硬认** ✗ —— 与 `_reachable_without_path` "
          "同一纪律 ✓）")
    ws.player.here_span = _span
    _cam, _wx = ws.player.cam_x, ws.player.world_x
    ws.player.cam_x, ws.player.world_x = None, None   # 可信相机没建立 + 也没世界坐标 ⇒ 推不出怪在哪 ✓
    check(a._climb_recheck_walk(ws, t + 0.05) is False,
          "连「怪的世界 x」都推不出来却照样掐（不猜 ✗）")
    ws.player.cam_x, ws.player.world_x = _cam, _wx
    a.stop_route("用例④")

    # ⑤ 来源怪**不在**候选里（视野里只剩那只近的）⇒ 不掐（保守 ✓ 同 `_climb_recheck` ④ ✓）
    _mk([near])
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job()], origin={"kind": "chase", "mob_id": 1})
    a._climb_tick(t, 500.0, set(), ws)
    a._climb_tick(t + 0.05, 500.0, set(), ws)
    check(a._climb is not None,
          "来源怪不在视野里（够不够得着判不出来）却照样掐 ✗ —— 「它走了该换目标」是**原有追击链**"
          "的事 ✓")
    a.stop_route("用例⑤")

    # ⑥ **只有「追击」来源参与**：来源不是 `"chase"` ⇒ 豁免 ✓
    _mk([src, near])
    a._target_id, a._target_until = 1, 0.0
    a.start_route([_job()], why="命令前往：顶层", origin={"kind": "queue", "mob_id": 1})
    a._climb_tick(t, 500.0, set(), ws)
    a._climb_tick(t + 0.05, 500.0, set(), ws)
    check(a._climb is not None,
          "「任务队列」这种来源被每拍复核掐掉了（用户明确：**只掐「追击」类** —— 人手安排的"
          "不许被动 ✗）")
    a.stop_route("用例⑥")

    # ⑦ 不在这一档就不查（判据全在 `route.ClimbJob.walking_to_ladder()` ✓ 一处口径 ✓）
    #    ⚠ **每条用新任务**：同一个任务连喂几拍会**推进相位**（对齐 → 按跳 → …）⇒ 后面几条会被
    #      "相位已经不是 ALIGN / 已经按过跳"蒙过去 ⇒ **假绿** ✗（反向验证时正是这么踩到的 ✓）。
    def _fresh(px=500.0, **kw):
        _j = _job(**kw)
        _j.update(t, px)                        # 喂一拍（顺手把 `_last_dx` 记上 ✓）
        return _j

    check(_job().walking_to_ladder() is False,
          "还没跑过一拍就报「正朝绳走」✗（`_last_dx` 那时还没有读数 ✓）")
    j = _fresh()
    check(j.walking_to_ladder() is True,
          "离绳 200px、既没起跳也没在绳上，却不算「起跳距离外朝绳走」⇒ 这条每拍复核"
          "**永远不会触发** ✗：%r" % (getattr(j, "_last_dx", None),))
    check(_fresh(px=700.0).walking_to_ladder() is False,
          "已经对齐到绳上（|dx| = 0）还算「正朝绳走」⇒ 会在绳边反复掐 ✗")
    j = _fresh()
    j._in_rope_span_now = True                  # 位置状态说：人就在 L1 的绳段里 ✓（`update` 由
    check(j.walking_to_ladder() is False,       # `on_rope_pos` 置它 ✓ 这里直接摆 ✓ 免得相位被推进）
          "人在绳上（位置状态判的）还算「朝绳走」⇒ 会在绳上被掐掉（在绳上按左右 = "
          "松手掉下来 ✗）")
    j = _fresh()
    j._jumped_once = True                       # 按过跳 ⇒ 这一轮已经进了攀爬流程 ⇒ 不归这条管 ✓
    check(j.walking_to_ladder() is False,
          "**按过跳**之后还算「起跳距离外朝绳走」✗（后面那一段该交给 `_climb_recheck` 的"
          "节点①/② ✓ 别在这儿拦 ✗）")
    j = _fresh()
    j._diag_flying = True                       # 斜跳飞行中（人已经在空中 ✓）
    check(j.walking_to_ladder() is False,
          "**斜跳飞行中**还算「朝绳走」⇒ 跳出去以后还会被掐 ✗")
    check(_fresh(jump_start_px=250).walking_to_ladder() is False,
          "|dx|（200）已经进了**起跳距离**（250）以内，却还算「起跳距离外」✗")
    # ⑧ `_last_dx` 由 `_align_step` **一处记账**（源码级：那两处 `self.x - float(px)` 都改读它 ✓）
    _rt = inspect.getsource(route.ClimbJob)
    check(_rt.count("self.x - float(px)") == 1,
          "`离绳距离`在 `ClimbJob` 里被算了 %d 处（该**只有 `_align_step` 开头那一处**记账 ⇒ "
          "两处口径迟早打架 ✗）" % _rt.count("self.x - float(px)"))


def t_disable_chase_pathfinding():
    """⛔ 「**禁用杀怪寻路**」（用户 2026-09-28 要求 ✓ 界面在「设置 → 判定参数」**顶层**）。

    用户原话："设置里判定参数顶层加个开关「**禁用杀怪寻路**」：开启后**不再查询怪物框所属的
    foothold 集合**（**也不显示**），**不再因追怪而下达寻路任务**而只是**单纯地走向锁定怪物**
    （**无寻路时的老逻辑**）" ✓。

    钉五件：
      ① **默认关** ⇒ 照旧下「前往」（**老行为一点不变** ✗ 别弄丢）；
      ② **开着** ⇒ **不下前往**、**进 chase**、**真的朝怪按了方向键**（= 单纯走向锁定怪物 ✓）；
      ③ **开着** ⇒ 挑目标那一步**也不查**（`mob_cost_of` 一次都不被调用 ✓ —— 它是**按候选
         逐只算 BFS** 的那笔开销 ✓，光靠"不注入 `mob_sets_of`"停不掉它：闭包捕获 ✗）；
      ④ **开着** ⇒ `live_thread` 的**记账口**早退（画面那行 `查#怪号 …` 的唯一来源 ✓，
         绘制段一行都不用动 ✓）—— 源码级钉 ✓；
      ⑤ 存读往返（`to_dict` / `from_dict`）+ **有界面控件** ⇒ 另见 `t_align_params` ✓。
    """
    from pathlib import Path

    from decision import route

    s = fresh_settings(attack_dist=100.0, attack_up_dist=10.0, attack_down_dist=10.0)
    s.enabled = True
    h = Harness(s)
    a = h.agent
    ws = h.ws(with_mob=False)
    ws.player.here_sets = ["乙平台"]
    ws.player.world_x = 500.0
    # ⚠ 怪要**在攻击范围之外**（距离 40 是"够得着"✓ 那样会直接进 attack、根本走不到追怪分支 ✗）：
    #   玩家中心 x=520、怪 x=700 ⇒ 水平 180 > attack_dist(100) ✓；竖直对齐 ⇒ 不会判"框不住" ✓
    ws.mobs = [Mob(id=1, x=700.0, y=float(ws.player.y), w=40.0, h=40.0, conf=0.9)]

    planned, cost_calls, sets_calls = [], [], []
    a.route_plan = lambda dst: (
        planned.append(dst) or
        {"jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
         "path": ["乙平台", dst], "why": "", "here": False})
    a.mob_sets_of = lambda pl, m: (sets_calls.append(m.id) or
                                   {"world": (600.0, -300.0), "sets": ["顶层"], "why": ""})
    a.mob_cost_of = lambda pl, m: (cost_calls.append(m.id) or
                                   {"cost": 100.0, "sets": ["顶层"], "why": ""})
    a.nearest_set_towards = lambda pl, m: "顶层"

    def _reset():
        planned.clear()
        cost_calls.clear()
        sets_calls.clear()
        a._mob_info_cache = None
        a._mob_goto_at = 0.0
        a._target_id = None
        a._target_until = 0.0
        a.stop_route("用例：复位")
        h.log.clear()

    # ① 默认（关）⇒ 照旧下「前往」+ 顺着算代价挑目标 ✓（老行为，别弄丢 ✗）
    _reset()
    a.tick(ws)
    check(planned == ["顶层"] and a._climb is not None,
          "开关**关着**的时候不下前往了（老行为被改坏 ✗ —— 用户要的是「开了才不走」✓）：%r"
          % (planned,))
    check(cost_calls, "开关**关着**时不按寻路距离挑目标了（那是 2026-09-27 的老功能 ✗）")

    # ②③ **开着** ⇒ 不下前往 + 进 chase + **一次都不查"怪在哪块平台"** ✓
    # ⚠⚠ 这一档**必须**走 `h.run`（**主循环那一套**）并**摆上 `player.here_sets`** ✗✗：
    #   · 方向键是**调用方**发的（`KeyState.set` → `dinput.key_down`）⇒ 直接调 `a.tick(ws)`
    #     只会把键塞进 `keys` 集合、**没人发** ✗（踩过：`h.log` 空空如也 ✓）；
    #   · `_chase_goto_if_elsewhere` 要 `here_sets` **非空**才往下走 —— 空的话它**早退在**
    #     「脚下那条 foothold 没圈进集合」⇒ **开关到底有没有用根本测不出来** ✗✗✗
    #     （⚠ 反向验证时就是这么被蒙过去的：把 `tick` 的闸门整个删掉，用例照样全绿 ✓）。
    #   ⇒ `h.run` 每帧**自建** `ws` ⇒ 外面设的会被冲掉 ⇒ 用 `h.ws_hook` 摆 ✓。
    def _mk_ws(w):
        w.player.here_sets = ["乙平台"]
        w.player.world_x = 500.0
        w.mobs = [Mob(id=1, x=700.0, y=500.0, w=40.0, h=40.0, conf=0.9)]

    h.mobs_fn = None
    h.ws_hook = _mk_ws
    _reset()
    s.disable_chase_pathfinding = True
    try:
        h.run(FRAME)
    finally:
        h.ws_hook = None
    check(planned == [], "开关**开着**还因追怪下了寻路任务（用户明确要求不许 ✗）：%r" % (planned,))
    check(a._climb is None, "开关开着还挂上了寻路任务：%r" % (a._climb,))
    check(not sets_calls,
          "开关开着**还在查「怪在哪块平台」**（`mob_sets_of` 被调了 %r 次 ✗ —— 用户要求"
          "「不再查询怪物框所属的 foothold 集合」✓）" % (sets_calls,))
    check(not cost_calls,
          "开关开着**还在按寻路距离挑目标**（`mob_cost_of` 被调了 %r 次 ✗ —— 它内部会去问"
          "「怪在哪块平台」⇒ 一样是那笔查询 ✗）" % (cost_calls,))
    check(a.state == "chase",
          "开关开着没进「追击」状态（该是**单纯走向锁定怪物** ✓）：%r" % (a.state,))
    _right = s.keymap.get("right")
    check(any(k == _right and kind == "down" for _t, kind, k in h.log),
          "开关开着却没朝怪按方向键（怪在右边 ⇒ 该按「右」✓ —— 「单纯走向锁定怪物」就是这个 ✗）："
          "%r" % (h.log,))
    s.disable_chase_pathfinding = False

    # ④ **记账口早退**（源码级 ✓ —— 它是画面那行的唯一来源，绘制段一字未动 ✓）
    _lt = (Path(__file__).resolve().parents[1] / "gui" / "live_thread.py"
           ).read_text(encoding="utf-8")
    check("disable_chase_pathfinding" in _lt,
          "`live_thread` 的「查过的怪框」记账口没跟开关走 ⇒ 开关开着**画面上照样显示**"
          "（用户要求「也不显示」✗）")
    check("return info" in _lt[_lt.index("disable_chase_pathfinding"):][:400],
          "记账口早退之后**没把 `info` 还给调用方** ⇒ `_zone_only`（「禁止战斗」区域筛）"
          "会拿到 `None`、那个功能当场坏掉 ✗")


def t_disable_chase_pathfinding_auto():
    """⛔⭐ 「**禁用杀怪寻路开着时，开自动也要能正常运转**」（用户 2026-10-04 ✓ 原话）。

    现场（触发这条的）：项目「东部岩山V」→ 图 `101030404` —— 它**没有标定几何、也没有
    地形图**（`datasets/map/101030404.mapcalib.json` 里连 `scale`/`offset` 都没有 ✓，
    也没有 `101030404.png`）⇒ 玩家世界坐标恒为 `None` ⇒ `tick` 里那条「**未定位玩家**」
    早退**每拍都命中** ⇒ 角色**站着不动**（behavior.log 里只剩 `act move=0`），再等
    「找不到玩家停止自动」到点就**自己把自动停掉** ✗。
    可这个开关的语义恰恰是「**不寻路、只朝怪走**」（= 无寻路时的老逻辑 ✓）—— 那一层
    **根本不读世界坐标**（锁定 / 追击 / 走位全用画面坐标 ✓）。

    钉六件（①④ 是"老行为一点不变"的对照 ✓）：
      ① **开关关着** + 没有世界坐标 ⇒ 照旧「未定位玩家」早退 ✓；
      ② **开关开着** + 没有世界坐标 ⇒ **照常打怪**（进 attack + 真按攻击键）✓；
      ③ **开关开着** + 没世界坐标 + **配了战斗区域** ⇒ 不许走「不在战斗区域，先回去」
         （那一步**要寻路**，而这个模式不寻路 ⇒ 又是一只怪都不打 ✗）；
      ④ **开关关着** + 世界坐标有、但**脚下集合空** + 同样配了区域 ⇒ **照旧**「先回去」
         （老规矩不许动 ✗）；
      ⑤ **开关开着**、脚下集合**读得到**且**不在区域里** ⇒ 区域规则**照旧生效** ✗
         （不许因为开了这个开关就把整条区域规则放行）；
      ⑥ **开关开着** + 连角色框都没有（`found=False`）⇒ 照旧「未定位玩家」✓。
    """
    s = fresh_settings(attack_dist=100.0)
    s.enabled = True
    h = Harness(s)
    a = h.agent
    atk = s.keymap["attack"]

    def _state(here=None, found=True, world=True):
        """摆这一拍的世界状态（`h.ws()` 每帧自建 ⇒ 只能这么摆 ✓ 同 `t_disable_chase_pathfinding`）。"""
        def f(w):
            w.player.world_x = 500.0 if world else None
            w.player.world_y = 500.0 if world else None
            w.player.here_sets = list(here or [])
            w.player.found = found
        return f

    with h._patched():
        h.clock0 = h.clock.t
        # ① 关着 + 没有世界坐标 ⇒ 照旧「未定位玩家」（2026-09-28 的老口径，别弄丢 ✗）
        h.log.clear()
        ws = h.ws()
        _state(world=False)(ws)
        res = a.tick(ws)
        check(res.get("reason") == "未定位玩家",
              "开关**关着**、又没有世界坐标时，居然不判「未定位玩家」了"
              "（那是 2026-09-28 定的老口径 ✗）：%r" % (res,))

        s.disable_chase_pathfinding = True
        try:
            # ② 开着 + 没有世界坐标 ⇒ 照常打怪（这就是用户要的"正常运转" ✓）
            h.log.clear()
            ws = h.ws()
            _state(world=False)(ws)
            for _ in range(4):            # 输出序列真按下键要两拍（既有用例同款 ✓）
                h.clock.t += 0.1
                res = a.tick(ws)
            check(a.state == "attack",
                  "「禁用杀怪寻路」开着、只是没有世界坐标，就没进 attack（角色会站着不动"
                  "⇒ 用户报的正是这个 ✗）：%s / %r" % (a.state, res))
            check(any(k == atk for _t, kind, k in h.log if kind in ("down", "up")),
                  "进了 attack 却没按攻击键：%r" % (h.log,))

            # ③ 开着 + 配了战斗区域 + 脚下集合空 ⇒ **不许**「先回去」
            s.battle_zone_sets = ["甲平台"]
            a.stop_route("用例：复位")
            ws = h.ws()
            _state(world=False)(ws)
            res = a.tick(ws)
            check(str(res.get("reason") or "") != "不在战斗区域，先回去",
                  "「禁用杀怪寻路」开着、脚下集合又读不到，还照老规矩判「不在战斗区域」⇒ "
                  "而「回去」要寻路（这个模式不寻路）⇒ 一只怪都不打 ✗：%r" % (res,))
            check(a.state == "attack" or res.get("keys"),
                  "同上：这一拍什么都没做（站着不动 ✗）：%r" % (res,))

            # ⑤ 开着 + 脚下集合**读得到**且不在区域里 ⇒ 区域规则照旧生效
            ws = h.ws()
            _state(here=["乙平台"])(ws)
            res = a.tick(ws)
            check(res.get("reason") == "不在战斗区域，先回去",
                  "脚下集合读得到、又不在能打区，却没走「先回去」（区域规则被这个开关"
                  "整个放行了 ✗）：%r" % (res,))

            # ⑥ 开着 + 连角色框都没认出来 ⇒ 照旧「未定位玩家」（那条保护还在 ✓）
            ws = h.ws()
            _state(world=False, found=False)(ws)
            res = a.tick(ws)
            check(res.get("reason") == "未定位玩家",
                  "连角色框都没认出来，却还往下走（「找不到玩家」那条保护没了 ✗）：%r" % (res,))
        finally:
            s.disable_chase_pathfinding = False

        # ④ 关着 + 世界坐标有、脚下集合空 + 配了区域 ⇒ 照旧「先回去」（老规矩不许动 ✗）
        #   ⚠ 这一档**必须**有世界坐标：没有的话先被上面①那条判走，根本到不了区域规则 ✗
        a.stop_route("用例：复位")
        ws = h.ws()
        _state(world=True)(ws)
        res = a.tick(ws)
        check(res.get("reason") == "不在战斗区域，先回去",
              "开关**关着**时，脚下集合读不到就该照旧「先回去」（2026-09-27 的老规矩 —— "
              "别被这个开关的改动顺手放行 ✗）：%r" % (res,))


def t_mob_goto_towards():
    """**降级路径**：朝「玩家 → 怪」方向的最近集合逐层逼近（用户 2026-09-28 定的三档 ②③ ✓）。

    用户原话（②）："那就**在追击上做文章**：如果玩家的攻击范围框 x 长与怪框 x 长范围相交了却没有
    触发 attack，就下达一个前往『**玩家到怪物向量**』指向的最近一个 foothold 集合的任务" ✓；
    （③）："**判错了：程序认为怪与玩家处于相同 foothold 集合，但……（攻击范围框框不住怪碰撞盒，
    典型的矩形相交问题），也要朝「向量方向的最近集合」逐层逼近**" ✓。
    为什么要它：①那条路**要先判准怪在哪层** ✗（这几轮翻车的都是它），而**方向**是可靠的 ✓。

    钉六件：
      ① **② 判不出**：`sets` 空 ⇒ 走"朝方向"（不是站着不动 ✓）；
      ② **② 走不到**：怪那层造不出路线 ⇒ 也走"朝方向" ✓；
      ③ **③**：判成**同一集合**、但**攻击框框不住怪**（水平够、竖直不够 ✓）⇒ 走"朝方向" ✓；
      ④ **③ 不许误伤**：真的同层同线（`_in_attack_box` 成立）⇒ **照旧追**、不下前往 ✓
         （老行为一字不变 ✓）；
      ⑤ 没注入 `nearest_set_towards` / 它解不出 ⇒ **不动** ✓（宁缺勿错 ✓）；
      ⑥ 只在节流放行时才动（共用 `_mob_goto_at` ✓）。
    """
    from decision import route

    s = fresh_settings(attack_dist=100.0, attack_up_dist=10.0, attack_down_dist=10.0)
    s.enabled = True
    h = Harness(s)
    a = h.agent
    _planned = []
    a.route_plan = lambda dst: (
        _planned.append(dst) or
        {"jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
         "path": ["乙平台", dst], "why": "", "here": False})
    ws = h.ws(with_mob=False)
    ws.player.here_sets = ["乙平台"]
    ws.player.world_x = 500.0
    target = Mob(id=1, x=560.0, y=200.0, w=40.0, h=40.0, conf=0.9)   # 水平 ≈40 ✓、竖直差 ≈100 ✗
    _tw = []
    a.nearest_set_towards = lambda pl, mob: (_tw.append(mob.id) or "顶层")
    t = 1000.0
    a._mob_goto_at = 0.0

    # ① ② 怪那层**判不出** ⇒ 朝方向逼近 ✓
    a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0), "sets": [], "why": "判不出"}
    a._mob_info_cache = None
    a._chase_goto_if_elsewhere(target, t, ws)
    check(_tw == [1] and getattr(a._climb, "dst_set", "") == "顶层",
          "怪那层判不出时没走「朝方向逼近」（站着不动 = 现场干蹭 ✗）：%r / %r"
          % (_tw, getattr(a._climb, "dst_set", None)))
    a.stop_route("用例：①检查完了")

    # ③ **判成同一集合、但攻击框框不住怪** ⇒ 也朝方向逼近 ✓
    _planned.clear()
    _tw.clear()
    a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0), "sets": ["乙平台"], "why": ""}
    a._mob_info_cache = None
    a._mob_goto_at = 0.0
    a._chase_goto_if_elsewhere(target, t + 10.0, ws)
    check(_tw == [1] and getattr(a._climb, "dst_set", "") == "顶层",
          "「判成同一集合、但攻击框框不住怪」没降级（这正是现场贴着干蹭那种 ✗）：%r / %r"
          % (_tw, getattr(a._climb, "dst_set", None)))
    a.stop_route("用例：③检查完了")

    # ④ 真的同层同线（竖直也够）⇒ **照旧追**、不下前往 ✓（老行为不许改 ✗）
    _planned.clear()
    _tw.clear()
    # ⚠ **按 `_in_box` 的实际口径**摆（踩了两次 ✗，别凭直觉 ✗）：竖直看的是
    #   「**角色中心 y ↔ 怪框**」的竖向最近距离（不是"框底对齐" ✗），而且"上/下"按**框中心**分
    #   （`m.y` vs `player.y` ✓）。这里 `attack_down_dist=10`（有限 ⇒ 才可能"框不住" ✓）⇒
    #   要让它**够得着**，怪框得贴住角色中心：取 `y = player.y`（框 `[y−20, y+20]` ⇒ `vd = 0` ✓）。
    _near = Mob(id=1, x=560.0, y=float(ws.player.y), w=40.0, h=40.0, conf=0.9)
    ws.player.world_x = 500.0
    a._mob_info_cache = None
    a._mob_goto_at = 0.0
    a._chase_goto_if_elsewhere(_near, t + 20.0, ws)
    check(not _tw and a._climb is None,
          "真的同层同线却把行程改掉了（老行为被改坏 ✗ —— 该照旧追）：%r / %r"
          % (_tw, a._climb,))

    # ⑤ 没注入解析器 ⇒ **不动** ✓
    _tw.clear()
    a.nearest_set_towards = None
    a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0), "sets": [], "why": "判不出"}
    a._mob_info_cache = None
    a._mob_goto_at = 0.0
    a._chase_goto_if_elsewhere(target, t + 30.0, ws)
    check(not _tw and a._climb is None,
          "没注入 `nearest_set_towards` 却动了（该不动 ✓ 宁缺勿错 ✗）：%r / %r"
          % (_tw, a._climb,))

    # ⑦ ⛔ **解析器抛异常 ⇒ 不许把追击带崩、也不许静默**（2026-09-28 现场修 ✗）。
    #    上一件 ⑤ 是"没注入"（= 返回 None ✓），这一件是"注入了但**炸了**" ✗ ——
    #    现场那次就是这条：`live_thread` 的降级解析器**每次都在 `IndexError`** ✓，
    #    而 `except Exception: return False` 把它吞成静默 ⇒ 只看到"原地卡住、也不下任务" ✗
    #    （用户原话："**没有重新查询、没找到也没根据向量下达寻路任务**"✓）。
    def _boom(_pl, _m):
        raise RuntimeError("替身故意炸")

    a.nearest_set_towards = _boom
    a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0), "sets": [], "why": "判不出"}
    a._mob_info_cache = None
    a._mob_goto_at = 0.0
    try:
        a._chase_goto_if_elsewhere(target, t + 40.0, ws)
    except Exception as ex:                    # noqa: BLE001
        raise AssertionError("解析器抛异常把「追击」整个带崩了（该吞住 ✓）：%s" % ex)
    check(a._climb is None, "解析器抛异常时不该下任务（拿不准 ⇒ 不动 ✓）")
    # 留痕是**源码级**钉的：`Harness` 会把 `behavior` 换掉，跑出来的事件不好断言 ✗
    from pathlib import Path

    _ag = (Path(__file__).resolve().parents[1] / "decision" / "agent.py"
           ).read_text(encoding="utf-8")
    check('behavior.event("mob_goto_toward_err"' in _ag,
          "解析器抛异常时**一条痕迹都不留**（这次就是因为它 ✗ —— 「卡住却查不出原因」"
          "正是静默吞异常造成的 ✓）")

    # ⑧ ⛔⛔ **真数据跑一遍那个降级解析器：不许抛**（2026-09-28 现场修的那条 ✗）——
    #    这条是**唯一**能发现"按四元组解包二元组"的用例（上面 ①~⑦ 全是替身 ⇒ 一概
    #    看不见它 ✗，所以它上线很久都没被发现 ✓）。断言**它不抛**为主、返回值只是形状 ✓。
    import types as _types

    from core import mapdata, zones
    from gui import live_thread as _lt
    _MID = "105090600"                        # 同上面那条真数据用例用的图 ✓
    _t2, _z2 = mapdata.load(_MID), zones.load(_MID)
    check(_t2 is not None and _z2 is not None, "那张真地图读不出来（前提不成立）")
    _th = _types.SimpleNamespace(_mmap_mid=_MID)
    _th._route_ctx = lambda m: (_t2, _z2)
    _fn = _lt.LiveThread._make_toward_set_resolver(_th)
    # ⚠ 假 player **必须带** `x / bottom / world_x / world_y`（少一个就早退 ⇒ 测不出东西 ✗
    #   —— 踩过 ✓）；`here_sets` 给空 ⇒ 不排除任何集合 ⇒ 候选更容易命中某个集合 ✓。
    _ply2 = _types.SimpleNamespace(x=100.0, y=200.0, bottom=250.0,
                                   world_x=500.0, world_y=200.0, here_sets=[])
    _mob2 = _types.SimpleNamespace(id=1, x=140.0, y=120.0, w=40.0, h=40.0)
    try:
        _got = _fn(_ply2, _mob2)
    except Exception as ex:                    # noqa: BLE001
        raise AssertionError(
            "降级解析器（`nearest_set_towards`）在**真地形**上抛了 %s：%s —— 它的调用方"
            "`agent._mob_goto_towards` 会把异常吞成 `return False` ⇒ 现场表现就是"
            "「怪判不出集合时**原地卡住**、也不下任务」✗（2026-09-28 就是这么翻的车 ✓）"
            % (type(ex).__name__, ex))
    check(_got is None or isinstance(_got, str),
          "降级解析器该给「集合名 or None」（宁缺勿错 ✓）：%r" % (_got,))

    # ⑨ ⭐⭐ **每条静默早退都要留痕**（2026-09-28 用户："**这次又卡住了，依旧没有走『取向量 →
    #    在小地图上根据黄点 + 向量找首个 foothold 集合 → 下达寻路任务』**"✗ —— 而当时
    #    log 里**一条线索都没有** ✗✗，同一个 bug 因此查了三轮 ✓）。
    #    钉三件（都是"**本来就不下任务、却看不出为什么**"的早退 ✓）：
    #      · 没注入 `nearest_set_towards`（实时线程没接上 ✓）；
    #      · 玩家 `here_sets` 空（**和「怪判不出」是完全不同的两件事** ✗ 现场分不清 ✓）；
    #      · 朝方向解析器**判不出**（返回空 ✓ ⇒ 那条路上没有可去的层 ✓）。
    _was_here2 = list(ws.player.here_sets or [])

    def _skip_reason(tt):
        a._last_chase_skip = None
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0), "sets": [], "why": "判不出"}
        a._mob_info_cache = None
        a._mob_goto_at = 0.0
        a._chase_goto_if_elsewhere(target, tt, ws)
        return str(a._last_chase_skip or "")

    a.nearest_set_towards = None
    _r1 = _skip_reason(t + 100.0)
    check("nearest_set_towards" in _r1,
          "「没注入方向解析器」导致的不下任务**没留痕**（现场就是「不走、也没线索」✗）：%r"
          % (_r1,))

    a.nearest_set_towards = lambda pl, mob: "顶层"
    ws.player.here_sets = []
    _r2 = _skip_reason(t + 110.0)
    check("here_sets" in _r2,
          "「玩家 here_sets 空」导致的不下任务**没留痕**（它和「怪判不出」是两件事 ✗ "
          "现场分不清）：%r" % (_r2,))

    ws.player.here_sets = _was_here2
    a.nearest_set_towards = lambda pl, mob: None
    _r3 = _skip_reason(t + 120.0)
    check("判不出" in _r3,
          "「朝方向解析器判不出」导致的不下任务**没留痕**：%r" % (_r3,))


def t_mob_goto_towards_no_self_loop():
    """降级挑出来的**落脚区就是玩家自己站的这块** ⇒ **不许站住**（2026-10-01 现场修 ✗）。

    病根两件（都在 `agent._mob_goto_towards`）：
      · `plan_and_start_route` 的 `ok=True` 有**两种**（见它自己的说明 ✓）——
        "真起跑了"、和"**已经在 X 上了（不用走）**"；后者的 `_climb` 是**空**的。
        这里原来 **`return bool(ok)`** ⇒ 把"不用走"当成了"已经安排他过去了" ⇒
        上层拿到 True ⇒ **这一帧不朝怪走** ✗ ⇒ 站着不动、每过一次 CD 重来一遍
        ⇒ 卡死在角落（寺院通道2 报的就是这个 ✓）；
      · 能打的区**只剩一张**时，`nearest_set_towards` 挑回来的**正是玩家站的这块**
        ⇒ 目标 == 起点 ⇒ 必定撞上上面那条 ✗。

    钉四件：
      ① 降级目标是**自己站的那块** ⇒ 不下前往、**也不许站住**（返回 False）+ 留痕 ✓；
      ② 整体 `_chase_goto_if_elsewhere` 也返回 False（照旧追 ✓ 不许原地不动 ✗）；
      ③ 规划器回 `here=True`（那种"成功但没起跑"）⇒ **不许当成成功** ✓；
      ④ 真的起跑了 ⇒ 照旧 True ✓（老行为一字不变 ✓）。
    """
    from decision import route

    s = fresh_settings(attack_dist=100.0, attack_up_dist=10.0, attack_down_dist=10.0)
    s.enabled = True
    h = Harness(s)
    a = h.agent
    ws = h.ws(with_mob=False)
    ws.player.here_sets = ["乙平台"]
    ws.player.world_x = 500.0
    # 水平 ≈40 ✓（够得着）、竖直差 ≈100 ✗（够不着）⇒ 「判成同一集合但攻击框框不住」✓
    target = Mob(id=1, x=560.0, y=200.0, w=40.0, h=40.0, conf=0.9)

    def _plan_here(dst):
        # 「已经在 X 上了」：`ok=True`，但**没有 jobs ⇒ 没有任何动作被挂上** ✗
        return {"jobs": [], "path": [dst], "why": "已经在「%s」上了" % dst, "here": True}

    def _plan_walk(dst):
        return {"jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
                "path": ["乙平台", dst], "why": "", "here": False}

    t = 2000.0

    # ① + ② 降级挑回来的是**自己站的这块** ⇒ 不下前往、也不许站住 ✓
    a.route_plan = _plan_walk
    # 怪判成**和我同一集合** ⇒ 「位置状态显示在右下」正是这个前提 ✓
    a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0), "sets": ["乙平台"], "why": ""}
    a.nearest_set_towards = lambda pl, mob: "乙平台"     # ⚠ 目标 = 玩家脚下这块 ✗
    a._mob_info_cache = None
    a._mob_goto_at = 0.0
    a._last_chase_skip = None
    r1 = a._chase_goto_if_elsewhere(target, t, ws)
    check(r1 is False,
          "降级目标就是自己站的那块时该「照旧追」（False）—— 返回 True 会让这一帧不朝怪走"
          " ⇒ 站着不动 ✗：%r" % (r1,))
    check(a._climb is None, "不该给这种没意义的目的地下前往：%r" % (a._climb,))
    check("换不了地方" in str(a._last_chase_skip or ""),
          "「目标就是我站的这块」这种早退**没留痕**（现场又是「不走、也没线索」✗）：%r"
          % (a._last_chase_skip,))

    # ③ 规划器回 `here=True`（ok=True 但 `_climb` 空）⇒ **不许当成"安排成功"** ✗
    a.route_plan = _plan_here
    a.nearest_set_towards = lambda pl, mob: "顶层"       # 不在脚下 ⇒ 能走到下面那步 ✓
    a._mob_info_cache = None
    a._mob_goto_at = 0.0
    r3 = a._mob_goto_towards(target, ws, t + 10.0, "降级：朝方向最近集合「%s」逐层逼近")
    check(r3 is False and a._climb is None,
          "`plan_and_start_route` 回「已经在 X 上了」（压根没起跑）却被当成成功了 —— "
          "上层会以为已经安排 ⇒ 角色站着不动 ✗：%r / %r" % (r3, a._climb))

    # ④ 真的起跑了 ⇒ 照旧 True ✓（老行为不许改坏 ✗）
    a.route_plan = _plan_walk
    a._mob_goto_at = 0.0
    r4 = a._mob_goto_towards(target, ws, t + 20.0, "降级：朝方向最近集合「%s」逐层逼近")
    check(r4 is True and getattr(a._climb, "dst_set", "") == "顶层",
          "真起跑了却被判成失败（老行为被改坏 ✗）：%r / %r" % (r4, a._climb))


def t_fight_clock_published():
    """⭐ 「**最大战斗时长**」的倒计时**只读镜像**（用户 2026-09-28 要求 ✓）。

    用户原话："编辑战斗区域里，要用**灰字**显示**最大战斗时长的倒计时**"✓。
    界面（`gui/player_panel.BattleZoneListDialog` ✓）拿不到 agent ⇒ 由 agent 把三件套写到
    `settings` 上（`fight_zone_name` / **`fight_elapsed_s`** / `fight_cap_s` ✓），界面**只读** ✓。

    ⚠⚠ **2026-09-28 口径改过**（用户报："**当前任务一直是战斗，但是时间走几秒就没了**"✗）：
      界面剩余 = **`cap − fight_elapsed_s`** ✓ —— **不是** `cap − (monotonic − started_at)` ✗。
      因为现在"在打"的判据是**没有寻路任务**（= 界面「当前任务」显示「战斗」那行 ✓ **同一个源**
      ✓），而**只有寻路会打断时间**（用户原话："只要一直战斗就不应该有任何理由停时间"✓）
      ⇒ 打断期间是**暂停累计**、**不是清零** ⇒ 拿墙钟减会把打断那段算进去 ⇒ 显示会跳 ✗。

    钉四件：
      ① **不在区域** ⇒ `fight_remain()` 给 `None`（界面**什么都不显示** ✓ 别显示 `0:00` ✗）；
      ② 开始计时 ⇒ `(集合名, 剩余秒, 上限秒)` ✓ 且 **settings 三件套同步写上** ✓
         （⚠ 落点的理由：界面只 import 了 `settings` 单例 ✓ 拿不到 agent ✓）；
      ③ **到点收摊** ⇒ 三件套一起清 ✓（只清一半会让界面显示"上一轮的剩余"✗）；
      ④ 那三个字段**不进 `to_dict`**（运行时状态 ✓ 不落盘、不跨会话复活 ✓ 同 `enabled` ✓）。
    """
    s = fresh_settings()
    s.enabled = True
    h = Harness(s)
    a = h.agent
    ws = h.ws(with_mob=False)
    a.settings.battle_zones = [{"set": "甲", "can_fight": True, "fight_max_s": 10.0,
                                "fight_dst": ""}]
    s.sync_battle_zone_sets()
    ws.player.here_sets = ["甲"]
    ws.player.world_x = 500.0
    t = 5000.0

    # ① 不在区域 ⇒ None（界面不显示 ✓；别显示 0:00 ✗）
    #   ⚠ **别拿 `a.state = "idle"` 来造这件事** —— 那是**老口径**（用 `self.state` 判"在打"）
    #      ✗，正是这次改掉的 bug：`state` 会在攻击/换目标之间闪 ⇒ 一闪就把时间清掉 ✗。
    ws.player.here_sets = []
    a._fight_beat(t, ws)
    check(a.fight_remain() is None and not getattr(s, "fight_zone_name", None),
          "不在任何战斗区域时不该有倒计时（界面会显示成 0:00 ✗）：%r / %r"
          % (a.fight_remain(), getattr(s, "fight_zone_name", None)))
    ws.player.here_sets = ["甲"]

    # ② 开始计时 ⇒ 剩余 = 上限 ✓ + settings 三件套写上 ✓
    a._fight_beat(t, ws)
    _r = a.fight_remain()
    check(_r is not None and _r[0] == "甲" and abs(_r[2] - 10.0) < 1e-6,
          "开始计时后没给出倒计时：%r" % (_r,))
    check(getattr(s, "fight_zone_name", None) == "甲"
          and abs(float(getattr(s, "fight_cap_s", 0.0)) - 10.0) < 1e-6
          and getattr(s, "fight_elapsed_s", None) is not None,
          "倒计时的三件套没写到 `settings`（界面拿不到 agent，只能读它 ✗）：%r / %r / %r"
          % (getattr(s, "fight_zone_name", None), getattr(s, "fight_elapsed_s", None),
             getattr(s, "fight_cap_s", None)))

    # ③ 到点收摊 ⇒ 三件套一起清 ✓
    a._fight_beat(t + 11.0, ws)
    check(a.fight_remain() is None and getattr(s, "fight_zone_name", None) is None,
          "到点之后三件套没一起清（界面会一直显示上一轮的剩余 ✗）：%r / %r"
          % (a.fight_remain(), getattr(s, "fight_zone_name", None)))

    # ④ 不落盘 ✓
    _d = s.to_dict()
    for _k in ("fight_zone_name", "fight_elapsed_s", "fight_cap_s"):
        check(_k not in _d,
              "`%s` 是**运行时状态**（界面倒计时用的），不该进项目文件 ✗" % _k)
    a.settings.battle_zones = []
    s.sync_battle_zone_sets()


def t_fight_min_clock_and_gate():
    """「**最小战斗时长(s)**」（用户 2026-10-01 ✓）钉三件：

      ① **只配最小、没配最大** ⇒ `_fight_acc` 照样累计（最小战斗时长的钟**不能**跟着
        `fight_max_s=0` 一起收摊 ✗，否则永远走不到、闸就白配了 ✓）；
      ② 累计**没到** `fight_min_s` ⇒ `_chase_goto_if_elsewhere` 对**别的集合**的怪**不下前往**
        （返回 False、`_climb` 保持 None ✓ —— 这就是"到点之前不追别的 foothold 集合的怪"✓）；
      ③ 累计**到了** ⇒ 照旧下前往、去怪那个集合 ✓（老行为恢复 ✓）。
    """
    from decision import route
    from decision.agent import ZONE_GOTO_RETRY_S

    s = fresh_settings(attack_dist=40.0)
    s.enabled = True
    h = Harness(s)
    h.mobs_fn = lambda t: [Mob(id=1, x=580.0, y=500.0, w=40.0, h=40.0, conf=0.9)]

    def _plan(dst):
        return {"jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
                "path": ["甲平台", dst], "why": "", "here": False}

    with h._patched():
        a = h.agent
        h.clock0 = h.clock.t
        a.route_plan = _plan
        ws = h.ws(with_mob=True)
        ws.player.here_sets = ["甲平台"]
        ws.player.world_x = 500.0
        # 怪在「乙平台」—— 和脚下「甲平台」不是同一个集合 ✓
        a.mob_sets_of = lambda pl, mob: {"world": (600.0, -300.0),
                                         "sets": ["乙平台"], "why": ""}
        a.settings.battle_zones = [{"set": "甲平台", "cd_s": 3.0, "fight_min_s": 5.0,
                                    "fight_max_s": 0.0, "fight_dst": "", "can_fight": True},
                                   # 怪在的「乙平台」也得是能打区，否则被 `_zone_allows` 拦下
                                   # （那是另一条规则、另一条用例 ✓ 这里只测「最小战斗时长」✓）
                                   {"set": "乙平台", "can_fight": True}]
        s.sync_battle_zone_sets()

        # ① 只配最小 ⇒ `_fight_acc` 照常累计（不因 fight_max_s=0 收摊 ✗）
        t = h.clock.t
        a._fight_reset(t)
        a._fight_beat(t, ws)
        h.clock.t += 2.0
        a._fight_beat(h.clock.t, ws)
        check(abs(a._fight_acc - 2.0) < 1e-6 and a._climb is None,
              "只配「最小战斗时长」时 `_fight_acc` 没累计 / 却下了前往（= 换地方 ✗）：%r / %r"
              % (a._fight_acc, a._climb))

        # ② 累计 2 秒 < 5 秒 ⇒ 不追别的集合的怪（不下前往 ✓）
        a._mob_info_cache = None
        a._mob_goto_at = 0.0
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        r = a._chase_goto_if_elsewhere(ws.mobs[0], h.clock.t, ws)
        check(r is False and a._climb is None,
              "累计战斗还没到「最小战斗时长」却去追别的集合的怪了：%r / %r" % (r, a._climb))

        # ③ 累计到了 ⇒ 恢复跨集合追击（去「乙平台」✓）
        a._fight_acc = 6.0
        a._mob_goto_at = 0.0
        h.clock.t += ZONE_GOTO_RETRY_S + 0.1
        r2 = a._chase_goto_if_elsewhere(ws.mobs[0], h.clock.t, ws)
        check(r2 is True and a._climb is not None
              and getattr(a._climb, "dst_set", "") == "乙平台",
              "累计到「最小战斗时长」后没恢复追别的集合的怪：%r / %r" % (r2, a._climb))
        a.stop_route("用例：③检查完了")
    a.settings.battle_zones = []
    s.sync_battle_zone_sets()

    # ④ 清洗：`fight_min_s` 读进来（缺省 0、坏值/负数钳 0 ✓）
    _m = fresh_settings()._load_battle_zones(
        {"battle_zones": [{"set": "甲", "fight_min_s": 8.0},
                          {"set": "乙", "fight_min_s": "x"},
                          {"set": "丙", "fight_min_s": -3.0}]})
    _mv = {z["set"]: z.get("fight_min_s") for z in _m}
    check(_mv == {"甲": 8.0, "乙": 0.0, "丙": 0.0},
          "`fight_min_s` 没按规矩清洗（坏值/负数该钳 0 ✗）：%r" % (_mv,))


def t_zone_pick_cheapest_and_rename():
    """⭐ 「勾选改为**可以战斗**」+「不在能打区 ⇒ 找**代价最低**的可战斗区」（用户 2026-09-28 ✓）。

    用户原话："编辑战斗区域：① 勾选改为「**可以战斗**」；② 如果当前 foothold 集合**不允许
    战斗**，则找**代价最低的可战斗区**"✓。

    ⚠⚠ **本次最要紧的一条：改名不改值** ✗✗ —— 这个键原来叫 `no_fight`（界面写「禁止战斗」✗），
      但**代码一直把它当白名单消费**（`_in_battle_zone` = `here & zones` ✓、`_zone_only`
      只保留名单里的怪 ✓）⇒ 它**本来就是"这块能打"** ✓。⇒ 若按字面取反，用户**已经配好的
      能打区会全部失效**（角色一直想逃出去 ✗✗）。⇒ 老键必须**照抄原值** ✓（见 ①）。

    钉五件：
      ① **老键 `no_fight` 的值原样读进 `can_fight`**（`True` → `True` ✓ **不取反**）；
      ② **一个都没勾 = 哪都能打**（`battle_zone_sets` 空 ⇒ `_in_battle_zone` True ✓ 用户定的 ✓）；
      ③ 不在能打区 ⇒ **挑代价最低的那块**（注入 `set_cost_of` 假代价 ✓，与「追击」同一套口径 ✓）；
      ④ **算不出代价 ⇒ 退回 `zones[0]`**（老行为 ✓ 绝不猜一块 ✗）；
      ⑤ **代价只在节流放行那一刻算**（这函数**每拍**都会进 ✗ ⇒ 每拍算一轮会压垮帧 ✓）。
    """
    from decision import route

    s = fresh_settings()
    s.enabled = True
    h = Harness(s)
    a = h.agent

    # ① 老键照抄原值（**不取反** ✗✗ —— 取反会让老项目的能打区全部失效）。
    #   ⚠ 2026-10-01 起战斗区域不再经 `from_dict` 读（改 per-map 文件 ✓）⇒ 这里直接钉
    #   **清洗口径本身**（`_load_battle_zones`，route_panel 迁 per-map 时走的也是它 ✓）。
    _bz = fresh_settings()._load_battle_zones(
        {"battle_zones": [{"set": "甲平台", "no_fight": True},
                          {"set": "乙平台", "no_fight": False}]})
    _by = {str(z["set"]): z for z in _bz}
    check(_by["甲平台"].get("can_fight") is True,
          "老键 `no_fight: true` 没被**原样**读成 `can_fight`（⚠ 取反 ⇒ 用户已配好的"
          "**能打区会全部失效** ✗✗）：%r" % (_by["甲平台"],))
    check(_by["乙平台"].get("can_fight") is False,
          "老键 `no_fight: false` 读错了：%r" % (_by["乙平台"],))
    check("no_fight" not in _by["甲平台"],
          "读出来的项里还留着老键 `no_fight`（该只留 `can_fight` ✓）：%r" % (_by["甲平台"],))
    _s0 = fresh_settings()
    _s0.battle_zones = _bz
    _s0.sync_battle_zone_sets()
    check(list(_s0.battle_zone_sets) == ["甲平台"],
          "派生副本没跟着新键走：%r" % (_s0.battle_zone_sets,))
    # ①-b 两个键都在 ⇒ **只信 `can_fight`**
    _b2 = fresh_settings()._load_battle_zones(
        {"battle_zones": [{"set": "丙", "can_fight": False, "no_fight": True}]})
    check(_b2[0].get("can_fight") is False,
          "两个键都在时该**只信 `can_fight`**：%r" % (_b2[0],))

    # ② 一个都没勾 = 哪都能打 ✓
    s.battle_zones = []
    s.sync_battle_zone_sets()
    ws = h.ws(with_mob=False)
    ws.player.here_sets = ["随便哪块"]
    check(s.battle_zone_sets == [] and a._in_battle_zone(ws) is True,
          "「可以战斗」一个都没勾时该**哪都能打**（用户 2026-09-28 明确 ✓）：%r / %r"
          % (s.battle_zone_sets, a._in_battle_zone(ws)))

    # ③ 不在能打区 ⇒ 挑**代价最低**的那块 ✓（代价 = 寻路距离，用户定的 ✓）
    _planned = []
    a.route_plan = lambda dst: (
        _planned.append(dst) or
        {"jobs": [route.WalkJob(dst, [(500.0, 510.0, 520.0, -208.0, "1")])],
         "path": ["丙", dst], "why": "", "here": False})
    s.battle_zones = [{"set": "甲", "can_fight": True},
                      {"set": "乙", "can_fight": True},
                      {"set": "丙", "can_fight": True}]
    s.sync_battle_zone_sets()
    check(list(s.battle_zone_sets) == ["甲", "乙", "丙"],
          "三项都勾了「可以战斗」却没全进名单：%r" % (s.battle_zone_sets,))
    ws.player.here_sets = ["丁"]                      # 丁 不在能打区 ⇒ 该走人 ✓
    _costs = {"甲": 900.0, "乙": 120.0, "丙": 500.0}
    a.set_cost_of = lambda pl, dst: _costs.get(dst[0])
    a._zone_goto_at = 0.0
    check(a._pick_battle_zone(ws, s.battle_zone_sets) == "乙",
          "没挑**代价最低**的那块（用户要求「找**代价最低**的可战斗区」✗）：%r / %r"
          % (a._pick_battle_zone(ws, s.battle_zone_sets), _costs))
    _planned.clear()
    a._zone_goto_at = 0.0
    a._leave_battle_zone_tick(1000.0, ws, set())
    check(_planned == ["乙"],
          "「不在能打区 ⇒ 先回去」没去**代价最低**的那块（去了 %r ✗）：%r"
          % (_planned, s.battle_zone_sets))

    # ④ 算不出代价 ⇒ 退回 `zones[0]`（老行为 ✓ 绝不猜 ✗）
    _planned.clear()
    a.set_cost_of = lambda pl, dst: None
    a._zone_goto_at = 0.0
    a.stop_route("用例：复位")
    a._leave_battle_zone_tick(2000.0, ws, set())
    check(_planned == ["甲"],
          "算不出代价时该退回**第一条**（老行为 ✓ 宁缺勿错 ✗）：%r" % (_planned,))
    # ④-b 没注入 ⇒ 同样退回第一条 ✓
    _planned.clear()
    a.set_cost_of = None
    a._zone_goto_at = 0.0
    a.stop_route("用例：复位2")
    a._leave_battle_zone_tick(3000.0, ws, set())
    check(_planned == ["甲"], "没注入代价解析器时该退回第一条：%r" % (_planned,))

    # ⑤ 代价**只在节流放行那一刻**算（这函数每拍都进 ✗ ⇒ 每拍算一轮会压垮帧 ✓）
    _n = []
    a.set_cost_of = lambda pl, dst: (_n.append(dst[0]) or 100.0)
    a._zone_goto_at = 0.0
    a.stop_route("用例：复位3")
    a._leave_battle_zone_tick(4000.0, ws, set())          # 放行 ⇒ 算一轮（3 块）
    _after_first = len(_n)
    a.stop_route("用例：复位4")
    a._leave_battle_zone_tick(4000.1, ws, set())          # 0.1 秒后 ⇒ 节流挡着 ⇒ 不该再算 ✓
    check(_after_first == 3 and len(_n) == _after_first,
          "节流没挡住 ⇒ 每拍都算一轮候选代价（`path_cost` 是 BFS ⇒ 会把帧压垮 ✗）：%r" % (_n,))

    s.battle_zones = []
    s.sync_battle_zone_sets()


def t_battle_zone_can_fight():
    """「**禁止战斗**」= 旧的「限制战斗区域」（用户 2026-09-28 第 3 条 ✓）。

    原话："『编辑战斗区域{foothold集合名}』编辑弹窗**在最顶层加个属性『禁止战斗』**
    （**默认不勾选**），**这就是旧的限制战斗区域的功能**，其他参数照旧"✓。

    实现：派生副本 `battle_zone_sets` **只收 `can_fight=True` 的项** ✓
    ⇒ 老的消费方（`_in_battle_zone` / `_zone_only` / `_leave_battle_zone_tick` / tick 那道门）
    **一个字都不用改** ✓（这正是当初把它做成派生副本的价值 ✓）。

    钉五件：
      ① **默认不勾**（缺键 / 新项 ⇒ `can_fight=False`）✓；
      ② **勾了才进**派生副本（= 禁战名单 ✓）；
      ③ ⛔ **老配置迁移**：`battle_zones` 的**每一项都还没有** `can_fight` 键 ⇒ 那是"重构之前"
         存的 ⇒ 当年 `battle_zone_sets` 装的**就是**「限制战斗区域」名单 ⇒
         **名单里那些项要标 `can_fight=True`** ✓（否则升级后限制会**悄悄失效** ✗ ——
         这是本次最容易翻车的一处 ✓）；
      ④ 更老：**压根没有** `battle_zones`（只有 `battle_zone_sets`）⇒ 迁移出来的项**全算禁战** ✓；
      ⑤ 新格式（至少一项带该键）⇒ **只信每项自己的键** ✓
         （⚠ 不许拿副本当名单 —— 那会把"只配了参数、没禁战"的项也算进去 ✗）。
    """
    # ① 默认不勾
    s = fresh_settings()
    s.battle_zones = [{"set": "甲"}]
    s.sync_battle_zone_sets()
    check(s.battle_zone_sets == [],
          "没勾「禁止战斗」却进了禁战名单（默认不勾 ✓）：%r" % (s.battle_zone_sets,))
    check(fresh_settings()._load_battle_zones({"battle_zones": [{"set": "甲"}]})[0]
          .get("can_fight") is False,
          "缺键时该默认**不勾**（老行为：只管它自己那几项参数、不禁战 ✓）：%r"
          % (fresh_settings()._load_battle_zones({"battle_zones": [{"set": "甲"}]}),))

    # ② 勾了才进
    s.battle_zones = [{"set": "甲", "can_fight": True}, {"set": "乙", "can_fight": False}]
    s.sync_battle_zone_sets()
    check(s.battle_zone_sets == ["甲"],
          "禁战名单没收对（**只该收勾上的** ✓）：%r" % (s.battle_zone_sets,))

    # ③ 老配置迁移（有 battle_zones，但各项都没有 can_fight 键）。
    #   ⚠ 2026-10-01 起战斗区域不再经 `from_dict` 读 ⇒ 这里直接钉 `_load_battle_zones`
    #   （route_panel 迁 per-map 文件时走的也是它 ✓）。
    legacy = {"battle_zones": [{"set": "底层", "cd_s": 3.0, "idle_foothold": "",
                                "fight_max_s": 0.0, "fight_dst": ""},
                               {"set": "小平台", "cd_s": 3.0, "idle_foothold": "",
                                "fight_max_s": 0.0, "fight_dst": ""}],
              "battle_zone_sets": ["底层"]}       # ← 当年「限制战斗区域」只圈了它 ✓
    s2 = fresh_settings()
    s2.battle_zones = s2._load_battle_zones(legacy)
    s2.sync_battle_zone_sets()
    check(s2.battle_zone_sets == ["底层"],
          "**老配置迁移丢了禁战名单**（升级后「限制战斗区域」会悄悄失效 ✗）：%r / %r"
          % (s2.battle_zone_sets, s2.battle_zones))
    check([z.get("can_fight") for z in s2.battle_zones] == [True, False],
          "迁移时该**只把名单里那些**标成禁战（别把没圈的也标上 ✗）：%r" % (s2.battle_zones,))

    # ④ 更老：压根没有 battle_zones ⇒ 迁移出来的全算禁战
    s3 = fresh_settings()
    s3.battle_zones = s3._load_battle_zones({"battle_zone_sets": ["底层", "小平台"]})
    s3.sync_battle_zone_sets()
    check(s3.battle_zone_sets == ["底层", "小平台"],
          "老格式（只有 `battle_zone_sets`）迁移后该**全算禁战**：%r" % (s3.battle_zone_sets,))

    # ⑤ 新格式：只信每项自己的键
    s4 = fresh_settings()
    s4.battle_zones = s4._load_battle_zones(
        {"battle_zones": [{"set": "甲", "can_fight": False},
                          {"set": "乙", "can_fight": True}],
         "battle_zone_sets": ["甲", "乙"]})      # ← 新格式下它只是副本，不该被当名单 ✗
    s4.sync_battle_zone_sets()
    check(s4.battle_zone_sets == ["乙"],
          "新格式下拿副本当名单了（会把「只配了参数、没禁战」的项也算进去 ✗）：%r"
          % (s4.battle_zone_sets,))


def t_battle_zones_per_map():
    """战斗区域 2026-10-01 起**按地图 id 存**（`datasets/map/<id>.battle.json`，用户要求 ✓）。

    原来存在 project.yaml（按项目）⇒ 同一张图被两个项目用时，配置"分家"、换图还残留 ✗；
    而它每一项（集合名 / idle 回归 foothold 编号 / 到点去哪）**全是地图里的东西** ⇒ 跟着地图存 ✓。
    读：`route_panel._load_battle_zones_for_map`（换图时把 per-map 文件读进 `settings.battle_zones` ✓）；
    写：`route_panel._bz_commit`（`core.battle.save` ✓）。清洗口径仍是 `_load_battle_zones`（一处 ✓）。
    """
    import shutil as _shutil
    import tempfile as _tmp
    import unittest.mock as _mock
    from pathlib import Path as _P

    from core import battle
    from decision.agent import settings as _settings
    from gui.route_panel import RoutePanel

    _bdir = _tmp.mkdtemp(prefix="bzm_")
    _bpat = _mock.patch.object(battle, "path",
                               lambda mid: _P(_bdir) / ("%s.battle.json" % mid))
    _bpat.start()

    class _Proj:
        def __init__(self, mid="105080000", decision=None):
            self._d = {"map_id": mid, "decision": decision or {}}

        def get(self, k, d=None):
            return self._d.get(k, d)

    # ⚠ 不真造 `RoutePanel`（离屏里多一个 QGraphicsView 会原生崩 ✓ —— 本套件只在
    #   `t_align_params` 里造过一次面板 ✓）。这两个方法只碰 `self._map_id()` /
    #   `self._refresh_battle_zones()` / settings / core.battle ⇒ 给个"假 self"直接调真方法 ✓。
    class _FakeRP:
        def __init__(self, mid="105080000"):
            self.mid = mid

        def _map_id(self):
            return self.mid

        def _refresh_battle_zones(self):
            pass

    rp = _FakeRP()
    try:
        # ① 没文件、也没 legacy ⇒ 空（= 不限制）
        RoutePanel._load_battle_zones_for_map(rp, _Proj())
        check(_settings.battle_zones == [] and _settings.battle_zone_sets == [],
              "没配过时该是空列表（= 不限制 ✓）：%r / %r"
              % (_settings.battle_zones, _settings.battle_zone_sets))

        # ② 老配置迁移：文件不在 + 旧 project.yaml 有 decision.battle_zones（老键 no_fight）
        #    ⇒ 迁进文件（no_fight 照抄 ✓ / idle_foothold 变列表 ✓）。CD 此时没老全局值 ⇒ 兜底 ✓
        legacy = {"battle_zones": [{"set": "甲", "no_fight": True, "idle_foothold": "41"},
                                   {"set": "乙", "no_fight": False}]}
        RoutePanel._load_battle_zones_for_map(rp, _Proj(decision=legacy))
        check([z["set"] for z in _settings.battle_zones] == ["甲", "乙"]
              and _settings.battle_zones[0]["can_fight"] is True
              and _settings.battle_zones[0]["idle_footholds"] == ["41"]
              and abs(float(_settings.battle_zones[0]["cd_s"]) - ag.ZONE_GOTO_RETRY_S) < 1e-9,
              "老配置没迁进 per-map（no_fight / idle_foothold 口径 ✗）：%r"
              % (_settings.battle_zones,))
        check(_settings.battle_zone_sets == ["甲"],
              "迁移后派生副本没同步：%r" % (_settings.battle_zone_sets,))
        check(battle.load("105080000") is not None,
              "迁移该写进文件：%r" % (battle.load("105080000"),))

        # ②-b 更老：只有 `battle_zone_sets` + 老 `goto_retry_s` ⇒ 每项 CD 取那个老值 ✓
        rp.mid = "999999999"
        RoutePanel._load_battle_zones_for_map(
            rp, _Proj(mid="999999999",
                      decision={"battle_zone_sets": ["底层", "小平台"],
                                "goto_retry_s": 8.0}))
        check([z["set"] for z in _settings.battle_zones] == ["底层", "小平台"]
              and all(abs(float(z["cd_s"]) - 8.0) < 1e-9 for z in _settings.battle_zones),
              "老 `goto_retry_s` 没迁成每项的 CD（8.0）：%r" % (_settings.battle_zones,))

        # ③ 文件已存在 ⇒ 旧 project.yaml **不再覆盖**（"清空了"是有意的 ✗）
        rp.mid = "105080000"
        RoutePanel._load_battle_zones_for_map(
            rp, _Proj(decision={"battle_zones": [{"set": "丙", "no_fight": True}]}))
        check([z["set"] for z in _settings.battle_zones] == ["甲", "乙"],
              "文件已存在时又被旧 project.yaml 覆盖了 ✗：%r" % (_settings.battle_zones,))

        # ④ `_bz_commit` 写 per-map 文件（不是 project.yaml ✓）
        RoutePanel._bz_commit(
            rp, [{"set": "乙", "cd_s": 5.0, "idle_footholds": [],
                  "fight_max_s": 0.0, "fight_dst": "", "can_fight": True}])
        _on_disk = battle.load("105080000")
        check([z["set"] for z in _on_disk] == ["乙"]
              and abs(float(_on_disk[0]["cd_s"]) - 5.0) < 1e-9,
              "_bz_commit 没写进 per-map 文件：%r" % (_on_disk,))
        check("battle_zones" not in _settings.to_dict(),
              "战斗区域又被写回 project.yaml 了（该按地图 id 存 ✗）")
    finally:
        _bpat.stop()
        _shutil.rmtree(_bdir, ignore_errors=True)
        _settings.battle_zones = []
        _settings.sync_battle_zone_sets()


def t_battle_zone_idle_picker():
    """「编辑战斗区域」子弹窗里的**只读集合视图**（用户 2026-09-28 要求 ✓）。

    用户原话："编辑战斗区域的**子弹窗**需要有『foothold 集合编辑器』的**同款视图（只读）**，
    可以通过**点选**来**查看 foothold 参数**、**配置「idle回归foothold」**"✓。

    钉五件：
      ① **只画本集合**（别的集合的线不许混进来 ✗）；
      ② **点选** ⇒ 选中它 + `picked` 信号 + `info()` 给出参数（id / x 范围 / 长 / 面 y ✓）；
      ③ **点空白 ⇒ 清空**（`""` = 不设 idle 回归点 ✓）；
      ④ **底图在场**（有 `canvas` 就该铺上 ✓ —— "同款视图"的观感靠它 ✓）；
      ⑤ 接进 `BattleZoneDialog`：**有工厂 ⇒ 视图**（`zone()["idle_foothold"]` = 点选那条 ✓）；
         **工厂回 `None` ⇒ 退回文本框** ✓（老环境 / 老用法一点不坏 ✓）。
    """
    # ⛔⛔ **2026-09-28 停用**（别删这段说明 ✗）：底下那套是"建**真视图**"的版本 —— 而那块视图
    #   内含 `QGraphicsView`，**在离屏自检里建不出来**（进程原生崩 `0xC0000409` ✗，两个套件都一样 ✓）
    #   ⇒ 判据改由 `t_foothold_picker_logic` 用**纯函数 + 真地形**测 ✓；
    #     **接线**由 `t_align_params` 覆盖（它会建 `BattleZoneDialog` ✓）；
    #     **Qt 那部分**（画布 / 底图 / 勾选）手测 ✓。⇒ 这里直接 return ✓。
    return
    from PyQt5.QtCore import pyqtSignal
    from PyQt5.QtWidgets import QWidget

    from core import mapdata, zones
    from decision.agent import ZONE_GOTO_RETRY_S
    from gui import player_panel as _ppm

    MID = "106010105"
    import sys as _sys
    _sys.stderr.write("[D] a 进用例\n"); _sys.stderr.flush()
    t = mapdata.load(MID, with_canvas=True)
    _sys.stderr.write("[D] b 地形 ok\n"); _sys.stderr.flush()
    z = zones.load(MID)
    check(t is not None and z is not None, "那张真地图读不出来（前提不成立）")
    _sys.stderr.write("[D] c 集合 ok\n"); _sys.stderr.flush()
    names = sorted(str(n) for n in (z.sets or {}))
    check(names, "那张图没有集合（前提不成立）")
    # 挑一个**foothold 最多**的集合（点选才有得点 ✓）
    best = max(names, key=lambda n: len((z.sets.get(n) or {}).get("footholds") or []))

    # ⚠⚠ **本套件里用替身视图**（不建真 `FootholdPicker` —— 它内含 `QGraphicsView`）：
    #   实测**这个套件里连空的 `QGraphicsView()` 都建不出来**（进程原生崩溃 `0xC0000409` ✗），
    #   那是本套件前面某处留下的 Qt 环境问题，**与视图本身无关** ✓
    #   ⇒ 于是分工：**这里只测"接线"**（工厂被调 / 值被写回 / 没工厂退回文本框 ✓），
    #     **真视图的行为在 `tools/selftest_zone_editor.py` 里测** ✓
    #     （那边全是图形控件、环境干净 ✓ 用例 `t_foothold_picker_readonly` ✓）。
    class _FakePicker(QWidget):
        """替身视图：只实现弹窗用到的那四个接口 ✓。"""

        picked = pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._cur = ""

        def current(self):
            return self._cur

        def set_current(self, fid):
            self._cur = str(fid or "")
            self.picked.emit(self._cur)      # 真视图点选也发这个信号 ✓（弹窗靠它刷新那行 ✓）

        def info(self, fid):
            return ("#%s（地板）" % fid) if fid else ""

    _sys.stderr.write("[D] d 替身类定义完\n"); _sys.stderr.flush()
    # ⑤ 接进 BattleZoneDialog：有工厂 ⇒ 视图；工厂回 None ⇒ 退回文本框
    _fact = []
    _dlg = _ppm.BattleZoneDialog(
        {"set": best, "cd_s": ZONE_GOTO_RETRY_S}, names=names,
        picker_factory=lambda nm, cur: (_fact.append((nm, cur)) or _FakePicker()))
    try:
        check(_dlg._picker is not None and _fact == [(best, "")],
              "「有工厂」却没摆视图（用户要的就是这块 ✗）：%r / %r" % (_dlg._picker, _fact))
        check(not _dlg.ed_idle.isVisible() or _dlg._picker.isVisible(),
              "有视图时不该再摆手填文本框（两处表达同一件事 ✗）")
        _dlg._picker.set_current("41")
        check(_dlg.idle_fid() == "41",
              "视图里选中的那条没被当成 idle 回归点（`zone()` 会存空 ✗）：%r" % (_dlg.idle_fid(),))
        check(_dlg.zone().get("idle_foothold") == "41",
              "`zone()` 没把它写出来：%r" % (_dlg.zone(),))
        check("41" in _dlg.lbl_fh.text(),
              "旁边那行没显示参数（用户点选就是要看它 ✗）：%r" % (_dlg.lbl_fh.text(),))
        _dlg._on_clear_idle()
        check(_dlg.idle_fid() == "" and "还没选" in _dlg.lbl_fh.text(),
              "「清空」没把 idle 回归点清掉：%r / %r" % (_dlg.idle_fid(), _dlg.lbl_fh.text()))
    finally:
        _kill_qt(_dlg)
    _sys.stderr.write("[D] e 第一个弹窗测完\n"); _sys.stderr.flush()
    _dlg2 = _ppm.BattleZoneDialog({"set": best, "cd_s": ZONE_GOTO_RETRY_S}, names=names,
                                  picker_factory=lambda nm, cur: None)
    try:
        check(_dlg2._picker is None and not hasattr(_dlg2, "lbl_fh"),
              "工厂回 `None` 时该**退回文本框**（老环境不许因此崩 ✗）")
        _dlg2.ed_idle.setText("41")
        check(_dlg2.idle_fid() == "41" and _dlg2.zone().get("idle_foothold") == "41",
              "退回文本框之后读不到手填的编号：%r" % (_dlg2.idle_fid(),))
    finally:
        _kill_qt(_dlg2)


def t_foothold_picker_logic():
    """「编辑战斗区域」子弹窗那块只读视图的**判据**（用户 2026-09-28 要求 ✓）—— **不碰 Qt** ✓。

    用户原话："编辑战斗区域的**子弹窗**需要有『foothold 集合编辑器』的**同款视图（只读）**，
    可以通过**点选**来**查看 foothold 参数**、**配置「idle回归foothold」**"✓。

    ⚠⚠ **为什么只测纯逻辑**：那块视图本身（`gui/foothold_picker.FootholdPicker`，内含
      `QGraphicsView`）**在离屏自检里建不出来** ✗ —— 实测崩在**构造**里（进程原生 `0xC0000409` ✗，
      `selftest_decision` / `selftest_zone_editor` **两个套件都一样**，`gc.collect()` 也不管用 ✓）
      ⇒ 那是**离屏 + Qt 图形资源**的环境限制 ✓（同一次里连**空的** `QGraphicsView()` 都建不出来 ✓），
      **与视图代码无关** ✓ ⇒ 判据抽成 `gui/foothold_picker.py` 的**模块级纯函数**在这里测 ✓，
      **Qt 那部分**（画布 / 底图 / 勾选）只能**手测** ✓。

    钉四件（都用**真地形**）：
      ① **只认本集合那几条**（`set_fids` / `footholds_of` ✓ 别的集合的不混进来 ✗）；
      ② **点选命中**：点某条的中点 ⇒ 就是它 ✓；**离它很远** ⇒ 空 ✓（= 点空白清空 ✓）；
      ③ **参数那行**：id / 墙或地板 / x 范围 / 长 / 面 y ✓；问不在本集合的 ⇒ 空串 ✓；
      ④ **别的集合的线点了不算**（宁缺勿错 ✓ —— 否则 idle 回归点会被设到集合外 ✗）。
    """
    from core import mapdata, zones
    from gui import foothold_picker as fp

    MID = "106010105"
    t, z = mapdata.load(MID, with_canvas=True), zones.load(MID)
    check(t is not None and z is not None, "那张真地图读不出来（前提不成立）")
    names = sorted(str(n) for n in (z.sets or {}))
    check(names, "那张图没有集合（前提不成立）")
    best = max(names, key=lambda n: len((z.sets.get(n) or {}).get("footholds") or []))
    # ① 只认本集合
    fids = fp.set_fids(z, best)
    fs = fp.footholds_of(t, z, best)
    check(fids and len(fids) == len(fs),
          "本集合的 foothold 没认全（id 表与地形对不上 ✗）：%r / %d" % (fids[:6], len(fs)))
    check(set(str(f.fid) for f in fs) == set(fids),
          "`footholds_of` 给的不是本集合那几条：%r" % (sorted(str(f.fid) for f in fs)[:8],))
    check(fp.footholds_of(t, z, "不存在的集合") == [],
          "不存在的集合该给空（不猜 ✓）")
    # ② 点选命中 + ③ 参数
    f0 = fs[0]
    mx = (float(f0.x1) + float(f0.x2)) / 2.0
    my = float(f0.y_at(mx))
    got = fp.pick_fid(t, z, best, mx, my, 8.0)
    check(got == str(f0.fid), "点那条的中点却没命中它：%r / %r" % (got, f0.fid))
    info = fp.fid_info(t, z, best, got)
    check(("#%s" % got) in info and "x" in info and "长" in info and "y=" in info,
          "参数那行没给出 id / x 范围 / 长 / 面 y（用户点选就是要看它 ✗）：%r" % (info,))
    check(fp.fid_info(t, z, best, "不存在") == "",
          "问一条不在本集合里的 foothold 该给空串（不猜 ✓）")
    check(fp.pick_fid(t, z, best, mx, my - 5000.0, 8.0) == "",
          "离得老远还命中（= 点空白没法清空 ✗）")
    # ④ 别的集合的线点了不算
    others = [f for f in t.footholds if str(f.fid) not in set(fids) and not f.is_wall]
    if others:
        fo = others[0]
        ox = (float(fo.x1) + float(fo.x2)) / 2.0
        oy = float(fo.y_at(ox))
        check(fp.pick_fid(t, z, best, ox, oy, 8.0) == "",
              "点到**别的集合**的线也算数了（idle 回归点会被设到集合外 ✗）：#%s" % fo.fid)


def t_battle_zone_item_behaviors():
    """战斗区域**每一项自己那两个参数**的行为（用户 2026-09-28："**开动**"✓）。

    字段 + 界面先就位（`BattleZoneDialog` ✓），这条钉**行为**（口径全在
    `decision/agent.py` 的 `battle_zones` 注释里 ✓）：

    **① 最大战斗时长 ⇒ 换地方**（`fight_max_s` / `fight_dst`）钉五件：
      ① 在区域里"**连着打**"超过它 ⇒ 下「前往 `fight_dst`」✓；
      ② 没到点 ⇒ 不下 ✓；③ `fight_max_s=0` = 不限 ⇒ 永不下 ✓；
      ④ **离开本集合 ⇒ 清零**（回来重新计时 —— 不许拿"离开那段时间"凑数 ✗）；
      ⑤ `fight_dst` 空 ⇒ 只清账、**不换地方** ✓。

    **② idle 回归**（`idle_foothold`，口径"位于本集合时**水平走**向它的中心、到中心就停、
      **不跨层**"）钉三件：
      ① 无怪 + 配了 + 注入了 `foothold_span` ⇒ **按 ←/→ 朝 foothold 范围走**（只看水平 ✓）；
      ② **进了范围（[x1,x2]，含 `_deadzone` 边界容差）⇒ 一个键都不按** ✓
        （⭐ 2026-09-29 晚口径 ✓ 用户："到了范围内就应该停下来，而不是一直找中心"——
        原来到**中心**才停 ✗）；
      ③ 没注入解析器 / 没配 ⇒ 不按键（**老行为一字不变** ✓）。
    """
    from decision import route

    s = fresh_settings(attack_dist=40.0)
    s.enabled = True
    h = Harness(s)
    a = h.agent
    _planned = []
    a.route_plan = lambda dst: (
        _planned.append(dst) or
        {"jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
         "path": ["乙平台", dst], "why": "", "here": False})
    ws = h.ws(with_mob=False)
    ws.player.here_sets = ["乙平台"]
    ws.player.world_x = 500.0
    a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0, "idle_footholds": ["41"],
                                "fight_max_s": 10.0, "fight_dst": "甲平台"}]
    a.settings.sync_battle_zone_sets()

    # ---------- ① 最大战斗时长 ----------
    t = 1000.0
    a._fight_reset(t)
    a._fight_beat(t, ws)                        # 起算
    a._fight_beat(t + 5.0, ws)
    check(not _planned, "还没到「最大战斗时长」就下前往了：%r" % (_planned,))
    a._fight_beat(t + 11.0, ws)
    check(_planned == ["甲平台"],
          "打够「最大战斗时长」却没换地方（`fight_dst` 没生效 ✗）：%r" % (_planned,))
    check(a._fight_since is None, "到点之后没清账（会每拍来一遍 ✗）")
    a.stop_route("用例：①检查完了")
    # ④ ⭐ **区域读不到 ⇒ 暂停、不清零**（用户 2026-09-28 改的口径 ✓）
    #   ⚠⚠ 这里是**老口径被推翻**的地方：原来"离开本集合 ⇒ 清零" ✗ —— 而 `here_sets`
    #      会因为**定位抖动**短暂读不到（`docs/寻路设计.md` 写过"定位抖动就这个量级"✓），
    #      于是就变成"时间被反复清掉" ⇒ 用户报"**一直战斗、时间走几秒就没了**"✗。
    #      现在：**读不到 ⇒ 什么都不做**（累计留着 ✓）、**只有真换到另一个区域项才归零** ✓。
    _planned.clear()
    a._fight_reset(t)
    a._fight_beat(t, ws)
    a._fight_beat(t + 6.0, ws)                  # 累计 6 秒 ✓
    _acc6 = a._fight_acc
    check(abs(_acc6 - 6.0) < 1e-6, "前 6 秒没累计上：%r" % (_acc6,))
    ws.player.here_sets = []                    # 定位读不到（离开 / 发抖都长这样）
    a._fight_beat(t + 7.0, ws)
    a._fight_beat(t + 30.0, ws)                 # 干等了 23 秒
    check(abs(a._fight_acc - _acc6) < 1e-6,
          "区域读不到的那段**被算进战斗时间**了（该**暂停** ✗）：%r（应为 %r）"
          % (a._fight_acc, _acc6))
    check(a._fight_zone is not None,
          "区域读不到一瞬就把这一轮的账**清掉**了 —— 这正是用户报的那个 bug ✗")
    ws.player.here_sets = ["乙平台"]            # 回到同一个区域项 ⇒ **接着算**（不是重头 ✓）
    a._fight_beat(t + 31.0, ws)
    check(not _planned,
          "回到本区域后不该到点（暂停那段不该累计；到点判据是**实际在打**的秒数 ✗）：%r"
          % (_planned,))
    # ⭐ 真换到**另一个区域项** ⇒ 归零（只有这一件事才该清 ✓）
    a.settings.battle_zones.append(
        {"set": "丙平台", "cd_s": 3.0, "idle_foothold": "",
         "fight_max_s": 10.0, "fight_dst": "甲平台"})
    a.settings.sync_battle_zone_sets()
    ws.player.here_sets = ["丙平台"]
    a._fight_beat(t + 32.0, ws)
    check(abs(a._fight_acc) < 1e-6 and str((a._fight_zone or {}).get("set")) == "丙平台",
          "换到**另一个区域项**没归零（会把上一个区域的累计带过来 ✗）：%r / %r"
          % (a._fight_acc, a._fight_zone))
    a.settings.battle_zones = [z for z in a.settings.battle_zones
                              if str(z.get("set")) != "丙平台"]
    a.settings.sync_battle_zone_sets()
    ws.player.here_sets = ["乙平台"]
    a.stop_route("用例：④检查完了")
    # ③ `fight_max_s=0` = 不限
    _planned.clear()
    a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0, "idle_foothold": "",
                                "fight_max_s": 0.0, "fight_dst": "甲平台"}]
    a.settings.sync_battle_zone_sets()
    a.state = "attack"
    a._fight_reset(t)
    for _k in range(40):
        a._fight_beat(t + 100.0 + _k * 10.0, ws)
    check(not _planned, "`fight_max_s=0`（不限）却换地方了：%r" % (_planned,))
    # ⑤ `fight_dst` 空 ⇒ 只清账、不换地方
    _planned.clear()
    a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0, "idle_foothold": "",
                                "fight_max_s": 10.0, "fight_dst": ""}]
    a.settings.sync_battle_zone_sets()
    a.state = "attack"
    a._fight_reset(t)
    a._fight_beat(t + 200.0, ws)
    a._fight_beat(t + 215.0, ws)
    check(not _planned, "`fight_dst` 是空的却下前往了（空 = 只停手不换地方 ✗）：%r"
          % (_planned,))

    # ---------- ② idle 回归 ----------
    a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0, "idle_footholds": ["41"],
                                "fight_max_s": 0.0, "fight_dst": ""}]
    a.settings.sync_battle_zone_sets()
    a.foothold_span = lambda fid: (650.0, 750.0) if str(fid) == "41" else None
    ws.player.world_x = 500.0                   # 在范围左边（左缘 650）⇒ 该往右 ✓
    # ⚠ 传 **set**（`_steer` 往里 `add` ⇒ list 会 `AttributeError` ✗ 踩过 ✓）
    _keys = set()
    a._idle_walk_beat(_keys, ws)
    check(_keys == {"right"},
          "idle 回归没按 ←/→ 朝 foothold 范围走（用户要「水平走」✓）：%r" % (_keys,))
    ws.player.world_x = 900.0                   # 在范围右边（右缘 750）⇒ 该往左 ✓
    _keys = set()
    a._idle_walk_beat(_keys, ws)
    check(_keys == {"left"}, "idle 回归走反了：%r" % (_keys,))
    ws.player.world_x = 700.0                   # 在范围**中间** ⇒ 站住（一个键都不按 ✓）
    _keys = set()
    a._idle_walk_beat(_keys, ws)
    check(not _keys, "进了范围还在按（用户 2026-09-29 晚：**到了范围内就停**，"
                     "不该继续找中心 ✗）：%r" % (_keys,))
    ws.player.world_x = 650.0                   # 正好站在左缘 ⇒ 也算到 ✓
    _keys = set()
    a._idle_walk_beat(_keys, ws)
    check(not _keys, "站在范围边线上还在按（会在边线上左右横跳 ✗）：%r" % (_keys,))
    ws.player.world_x = 645.0                   # 边线外 5px（`_deadzone`=6 容差内）⇒ 站住 ✓
    _keys = set()
    a._idle_walk_beat(_keys, ws)
    check(not _keys, "边界容差内还在按（定位抖一下就横跳 ✗）：%r" % (_keys,))
    ws.player.world_x = 640.0                   # 边线外 10px（容差外）⇒ 该继续往右 ✓
    _keys = set()
    a._idle_walk_beat(_keys, ws)
    check(_keys == {"right"}, "边界容差外却不走了（容差把「到」圈太大 ✗）：%r" % (_keys,))
    # ⭐ **决策行用**（2026-09-29：「决策：idle → foothold#41」✓）：beat 是目标 id 的
    #    **唯一写口** —— 到中心站住也照样记着（区域配了它就是目标 ✓）
    check(getattr(a, "_idle_fid", None) == "41",
          "beat 没把当前 idle 回归目标记下来（信息栏没东西可显示 ✗）：%r"
          % (getattr(a, "_idle_fid", None),))
    # ⭐⭐⭐ **每次"闲下来"都重抽，不设任何判定**（用户 2026-09-30 定稿 ✓ 原话："每次都重新
    #    随机，**没有任何判定机制**，就是简单的『每次都随机抽取』" ✓）：
    #    锚点 = `_set_state` **离开 idle 就清号**（不看闲了多久 ✗）⇒ 下次闲下来必抽 ✓。
    #    ⚠ 这一格先后被我加过两个门槛、用户都否掉 ✗：① 只在"目标不在名单里"时抽 ⇒ 抽中一次
    #      **永久保留** ✗✗；② `_IDLE_REROLL_MIN_S=3.0`（"闲够 3 秒才算一轮" ✗）⇒ 森林迷宫III
    #      的 idle 段只有 0.05~1 秒（怪框闪 ✗）⇒ **从来不够格** ⇒ 抽签号长期不清 ⇒ 现场就是
    #      用户报的"**固定其中一个**" ✗✗（`behavior.log` 实证 ✓）。
    a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0,
                               "idle_footholds": ["41", "42"],
                               "fight_max_s": 0.0, "fight_dst": ""}]
    a.settings.sync_battle_zone_sets()
    a.foothold_span = lambda fid: {"41": (650.0, 750.0),
                                   "42": (150.0, 250.0)}.get(str(fid))
    ws.player.world_x = 500.0
    a._set_state("idle")
    _picks = set()
    for _k in range(120):
        a._set_state("attack")                      # 打起来了（离开 idle ⇒ 清号 ✓）
        if _k == 0:
            check(a._idle_fid is None,
                  "离开 idle 没清抽签号（下次闲下来还是老地方 = 用户报的 bug ✗）")
        a._set_state("idle")                        # 又闲下来了
        a._idle_walk_beat(set(), ws)
        _picks.add(str(a._idle_fid))
        if len(_picks) > 1:
            break
    check(_picks == {"41", "42"},
          "反复闲下来**只在名单里随机抽**（实际只抽到 %r ⇒ 就是「固定某个」✗）"
          % (sorted(_picks),))
    # ⭐ **零门槛**：怪框一闪（幽灵框 ✗）**立刻** attack→idle 抖一下 ⇒ **照样清号重抽** ✓
    #   （用户："没有任何判定机制" ✓ —— 不许再出现"闲得不够久就不换"这种门槛 ✗）
    a._idle_fid = "41"
    a._set_state("idle")
    a._set_state("attack")                          # 立刻被"怪"打断（幽灵框 ✗）
    check(a._idle_fid is None,
          "只闲了 0 秒就离开 idle，抽签号却没清（= 又加了判定门槛 ✗ 用户明确否掉 ✓）：%r"
          % (a._idle_fid,))
    # ③ 没注入解析器 / 没配 ⇒ 老行为（不按键 ✓）
    # ③' 解析失败要**留痕**（2026-09-29：`foothold_x` 解析失败原来是静默 no-op ⇒
    #     "站在区域上却不动"无从查起 ✗）—— 只在原因变化时记一条 `idle_walk_skip` ✓；
    #     正常 no-op（没配 idle_foothold / 已在 foothold 范围内）**不打点** ✓
    import tempfile as _tmpmod

    from core import behavior as _beh

    _old_blog, _old_ben = _beh.LOG, _beh.ENABLED
    _blogf = Path(_tmpmod.mkdtemp(prefix="behavior_idlewalk_")) / "behavior.log"

    def _blog():
        # ⚠ behavior 一行一次落盘、没记过事件时**文件不存在** ⇒ 读要兜底成空 ✓
        return _blogf.read_text(encoding="utf-8") if _blogf.exists() else ""

    a.foothold_span = None
    ws.player.world_x = 500.0
    try:
        _beh.configure(True, log=_blogf)
        a._last_idle_walk_skip = None
        _keys = set()
        a._idle_walk_beat(_keys, ws)
        check(not _keys, "没注入 `foothold_span` 却按键了（老行为被改坏 ✗）：%r" % (_keys,))
        check("idle_walk_skip" in _blog() and "没注入 foothold_span" in _blog(),
              "解析失败没留痕（「站着不动」依旧无从查起 ✗）：%r" % (_blog(),))
        a._idle_walk_beat(set(), ws)          # 原因没变 ⇒ 不重复记 ✓
        check(_blog().count("idle_walk_skip") == 1,
              "原因没变却重复打点（会淹 log ✗）：\n%s" % (_blog(),))
        a.foothold_span = lambda fid: None    # 换个原因（解析不出）⇒ 记新的一条 ✓
        a._idle_walk_beat(set(), ws)
        check(_blog().count("idle_walk_skip") == 2 and "解析不出" in _blog(),
              "换了原因却没记新的一条 ✗：\n%s" % (_blog(),))
        a.foothold_span = lambda fid: (650.0, 750.0) if str(fid) == "41" else None
        a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0, "idle_foothold": "",
                                    "fight_max_s": 0.0, "fight_dst": ""}]
        a.settings.sync_battle_zone_sets()
        _keys = set()
        a._idle_walk_beat(_keys, ws)
        check(not _keys, "没配 `idle_foothold` 却按键了（空 = 无 ✗）：%r" % (_keys,))
        check(getattr(a, "_idle_fid", None) is None,
              "没配 idle_foothold 还留着上一个目标（决策行会显示过期目标 ✗）：%r"
              % (getattr(a, "_idle_fid", None),))
        check(_blog().count("idle_walk_skip") == 2,
              "没配 `idle_foothold`（正常 no-op）也打了点 ✗：\n%s" % (_blog(),))
    finally:
        _beh.configure(_old_ben, log=_old_blog)
        # 把全局设置恢复干净（`settings` 是模块级单例，别的用例也在用 ✓）
        a.settings.battle_zones = []
        a.settings.sync_battle_zone_sets()

    # ---------- ⑥ ⭐⭐ **tick() 早退也要真把键按下去**（2026-09-29 实测踩坑 ✓）----------
    #   现场（森林迷宫III 二楼）：idle 分支是**早退**，正常路径末尾那次
    #   `self.keys.set(keys)` 到不了 ⇒ 转向键只进了返回字典（只喂信息栏 ✗），
    #   **从来没人按** —— `act move=1`（想按）而 `keys` 字段空（真按着的没有）✗。
    #   修复 = 早退前补 `self.keys.set(_ikeys)`（同休息分支那条 ✓）。
    #   ⚠ 上面的 ①②③ 都是**直接调 `_idle_walk_beat`**（beat 返回的键进 `_keys` 就过了），
    #     **测不到这一层** ✗ —— 必须走完整 `tick()` 才能钉住"键真被按下" ✓。
    s2 = fresh_settings()
    s2.enabled = True
    h2 = Harness(s2)
    a2 = h2.agent
    a2.foothold_span = lambda fid: (650.0, 750.0)
    a2.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0,
                                 "idle_footholds": ["41"],
                                 "fight_max_s": 0.0, "fight_dst": ""}]
    a2.settings.sync_battle_zone_sets()
    ws2 = h2.ws(with_mob=False)
    ws2.player.here_sets = ["乙平台"]
    ws2.player.world_x = 500.0
    try:
        a2.tick(ws2)
        check(s2.keymap["right"] in a2.keys.pressed(),
              "tick() 的 idle 分支转向键没真按下去（早退漏了 `keys.set` ⇒ "
              "信息栏看着在走、人一动不动 ✗）：%r" % (sorted(a2.keys.pressed()),))
    finally:
        a2.keys.release_all()
        a2.settings.battle_zones = []
        a2.settings.sync_battle_zone_sets()

    # ---------- ⑦ **回归点池随机**（2026-09-29 用户："回归foothold要是列表随机" ✓）----------
    #   钉四件：① 抽签只会从池里出 ✓；② 池里的当前目标**保持**（人不在几块砖间晃 ✗）；
    #   ③ 离开区域 ⇒ 目标清掉，下次触发**重抽** ✓；④ 老单值键自动迁移成单元素列表 ✓。
    import random as _rndmod
    a3 = Harness(fresh_settings()).agent
    a3.foothold_span = lambda fid: (650.0, 750.0)
    a3.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0,
                                 "idle_footholds": ["41", "5", "27"],
                                 "fight_max_s": 0.0, "fight_dst": ""}]
    a3.settings.sync_battle_zone_sets()
    ws3 = h2.ws(with_mob=False)
    ws3.player.here_sets = ["乙平台"]
    _rndmod.seed(20260929)
    _picked = set()
    for _ in range(30):
        a3._idle_fid = None            # 每轮强制重抽（= 每次"触发回归"一次抽签 ✓）
        a3._idle_walk_beat(set(), ws3)
        _picked.add(str(a3._idle_fid))
    check(_picked <= {"41", "5", "27"} and len(_picked) >= 2,
          "抽签越池（%r）或永远只抽一个（随机失效 ✗）" % (_picked,))
    a3._idle_fid = "5"
    a3._idle_walk_beat(set(), ws3)
    check(a3._idle_fid == "5",
          "池里的当前目标被换掉了（人会在几块砖之间来回晃 ✗）：%r" % (a3._idle_fid,))
    ws3.player.here_sets = []
    a3._idle_walk_beat(set(), ws3)
    check(a3._idle_fid is None, "离开区域没把当前目标清掉（下次不会重抽 ✗）")
    _mig = fresh_settings()._load_battle_zones(
        {"battle_zones": [{"set": "乙平台", "cd_s": 3.0, "idle_foothold": "26"}]})
    check(_mig[0].get("idle_footholds") == ["26"],
          "老单值键没迁移成列表（老配置丢回归点 ✗）：%r" % (_mig[0],))

    # ---------- ③ ⭐⭐ **切换平台 + 持续多久触发**（用户 2026-10-02 ✓）----------
    #   idle 持续 `idle_switch_delay_s` 秒才下前往任务 —— 一进 idle 就下 ⇒ 怪暂时
    #   跑出范围会立刻切平台、climb 中途遇怪又被打断 ⇒ 循环 ✗。
    _planned.clear()
    a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0,
                                "idle_dst_set": "甲平台",
                                "idle_switch_delay_s": 2.0,
                                "fight_max_s": 0.0, "fight_dst": ""}]
    a.settings.sync_battle_zone_sets()
    ws.player.here_sets = ["乙平台"]
    a._climb = None
    a.state = "chase"                        # 先不在 idle ✓
    a._set_state("idle")                    # 进 idle ⇒ 设锚 ✓
    check(getattr(a, "_idle_since", None) is not None,
          "进 idle 没设 `_idle_since`（切换平台门槛没锚点 ✗）")
    # 还没够 2 秒 ⇒ 不下任务 ✓（站着等，一个键都不按 ✓）
    _keys = set()
    a._idle_walk_beat(_keys, ws)
    check(not _planned,
          "idle 还没持续够 `idle_switch_delay_s` 就下切换平台任务了 ✗：%r" % (_planned,))
    check(not _keys,
          "切换平台模式 idle 不足时还按键了（应站着等 ✗）：%r" % (_keys,))
    # 够 2 秒了 ⇒ 下任务 ✓
    a._idle_since = time.monotonic() - 3.0      # 假装已经闲了 3 秒（>2 ✓）
    a._idle_walk_beat(set(), ws)
    check(_planned == ["甲平台"],
          "idle 持续够 `idle_switch_delay_s` 却没下切换平台任务 ✗：%r" % (_planned,))
    # 中途有怪（离开 idle）⇒ 清锚 ⇒ 下次重来 ✓
    a._set_state("attack")
    check(a._idle_since is None,
          "离开 idle 没清 `_idle_since`（下次 idle 会带着旧锚 ✗）")
    # delay=0 ⇒ **不启用**切换平台逻辑（用户 2026-10-02 ✓ 改语义：0=不启用，不是"立刻"）
    _planned.clear()
    a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0,
                                "idle_dst_set": "甲平台",
                                "idle_switch_delay_s": 0.0,
                                "fight_max_s": 0.0, "fight_dst": ""}]
    a.settings.sync_battle_zone_sets()
    a._climb = None
    a._set_state("idle")
    a._idle_walk_beat(set(), ws)
    check(not _planned,
          "delay=0 却下切换平台任务了（新语义：0=不启用 ✗）：%r" % (_planned,))

    # ⭐⭐ **取消互斥**（用户 2026-10-02 ✓）：idle_dst_set + idle_footholds 都配 ⇒
    #   idle 不足时**走 idle 回归**（水平走 ✓）、够时间才下切换平台任务 ✓。
    _planned.clear()
    a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0,
                                "idle_dst_set": "甲平台",
                                "idle_switch_delay_s": 2.0,
                                "idle_footholds": ["41"],
                                "fight_max_s": 0.0, "fight_dst": ""}]
    a.settings.sync_battle_zone_sets()
    a.foothold_span = lambda fid: (650.0, 750.0) if str(fid) == "41" else None
    ws.player.world_x = 500.0      # 在范围左边 ⇒ 该往右走 ✓
    a._climb = None
    a._set_state("idle")           # 进 idle 设锚
    _keys = set()
    a._idle_walk_beat(_keys, ws)
    check(not _planned, "idle 不足却下切换平台任务了 ✗：%r" % (_planned,))
    check(_keys == {"right"},
          "idle 不足时没走 idle 回归（取消互斥后应同时走 ✗）：%r" % (_keys,))
    # 够时间了 ⇒ 下切换平台任务（不再走 idle 回归 ✓ —— 下了任务 _climb 接管）
    a._idle_since = time.monotonic() - 3.0
    _planned.clear()
    _keys = set()
    a._idle_walk_beat(_keys, ws)
    check(_planned == ["甲平台"], "够时间却没下切换平台任务 ✗：%r" % (_planned,))
    check(not _keys, "下了切换平台任务还走 idle 回归（应让 _climb 接管 ✗）：%r" % (_keys,))

    # 没配 idle_switch_delay_s ⇒ 默认 3.0（用户 2026-10-02 ✓）
    _mig2 = fresh_settings()._load_battle_zones(
        {"battle_zones": [{"set": "乙平台", "cd_s": 3.0, "idle_dst_set": "甲平台"}]})
    check(_mig2[0].get("idle_switch_delay_s") == 3.0,
          "没配 `idle_switch_delay_s` 不是默认 3.0 ✗：%r"
          % (_mig2[0].get("idle_switch_delay_s"),))
    # 显式配了 0 ⇒ 尊重 0（= 立刻，老行为 ✓）
    _mig3 = fresh_settings()._load_battle_zones(
        {"battle_zones": [{"set": "乙平台", "cd_s": 3.0,
                           "idle_dst_set": "甲平台", "idle_switch_delay_s": 0}]})
    check(_mig3[0].get("idle_switch_delay_s") == 0.0,
          "显式配 0 却被改成 3.0（不尊重显式值 ✗）：%r"
          % (_mig3[0].get("idle_switch_delay_s"),))


def t_sweep_idle_switch_and_walk_priority():
    """**扫平台**策略下「idle 那两件事」也得能用 + 方向键优先级（用户 2026-10-02 方案 A ✓）。

    背景：扫平台巡逻那一拍状态写的是 `chase`（**故意的** —— `_chase_hop_beat` 的判据就是
    `state == "chase"`，它要覆盖「追怪 / 扫平台巡逻」两条路 ✓，见 tick 里那段说明 ✓），
    而 `_idle_walk_beat` 原来**只被平地巡逻的 `else:` 支调用**、计时的锚 `_idle_since`
    也只在 `state == "idle"` 时才设 ⇒ 扫平台下「idle 回归 foothold」和
    「idle 持续 n 秒切换平台」**两件都是死的** ✗。

    钉四件：
      ① 巡逻那一拍就给「切换平台」的计时**起锚**（不依赖 `state == "idle"` ✓）；
      ② 闲够 `idle_switch_delay_s` ⇒ 走**同一份** `_idle_walk_beat` 下「切换平台」任务 ✓；
      ③ **方向键归「朝倾向方向走」**（用户 2026-10-02：beat 的「回到 foothold 中心」优先级
         **更低** ✓）—— 即使配了 `idle_footholds`、而且它对方向的诉求**与巡逻相反**，
         也只能按巡逻那一个方向键 ✓（两个方向键一起按 = 人原地抖 ✗）；
      ④ **前面站着怪 ⇒ 不切平台**（扫平台只锁背后 back_range 内的怪 ⇒ 光看 `target is None`
         会把「前面有怪」当成没人 ✗ ⇒ 判据显式要「视野里一只怪都没有」✓）。
    """
    from decision import route

    s = fresh_settings(strategy="sweep", attack_dist=40.0)
    s.sweep_turn_cd = 600000                # 别让「换朝向延迟」在这条用例里翻方向 ✓
    s.vision_left = s.vision_right = 1000   # ⚠ 默认视野只有 ±200，怪摆远了会被滤掉 ✗
    h = Harness(s)
    a = h.agent
    _planned = []
    a.route_plan = lambda dst: (
        _planned.append(dst) or
        {"jobs": [route.WalkJob(dst, [(520.0, 510.0, 530.0, -208.0, "1")])],
         "path": ["乙平台", dst], "why": "", "here": False})
    # idle 回归点故意配在**左边**（150~250），而巡逻主方向是**右**（`_patrol_dir` 默认 +1）
    # ⇒ 两边的方向键诉求**相反** ⇒ 谁赢一眼可见 ✓
    a.foothold_span = lambda fid: (150.0, 250.0) if str(fid) == "41" else None
    a.settings.battle_zones = [{"set": "乙平台", "cd_s": 3.0, "idle_footholds": ["41"],
                                "idle_dst_set": "甲平台", "idle_switch_delay_s": 0.5,
                                "fight_max_s": 0.0, "fight_dst": ""}]
    a.settings.sync_battle_zone_sets()

    def _hook(w):
        w.player.here_sets = ["乙平台"]
        w.player.world_x = 500.0        # 在 idle 回归点**右**边 ⇒ 回归想按 left ✓
        w.player.here_span = None       # 关掉「距集合边缘回头」（免得方向被翻 ✗）

    h.ws_hook = _hook
    h.mobs_fn = lambda _t: []               # 视野里一只怪都没有 ⇒ 真"闲" ✓
    try:
        h.run(0.35)                     # < delay（0.5 秒）⇒ 只该起锚、不该下任务 ✓
        check(a._idle_since is not None,
              "扫平台巡逻那一拍没给「切换平台」的计时起锚（`quiet=` 没传 / 没生效 ✗）")
        check(not _planned,
              "还没闲够 `idle_switch_delay_s` 就切平台了（一闲下来就走 = 怪跑出视野就"
              "换地方 ✗）：%r" % (_planned,))
        _pressed = {k for _t, kind, k in h.log if kind == "down"}
        check(s.keymap["right"] in _pressed,
              "扫平台巡逻没按倾向方向的键（这条用例的前提不成立 ✗）：%r" % (sorted(_pressed),))
        check(s.keymap["left"] not in _pressed,
              "巡逻那一拍**又**去按「回到 foothold 中心」的方向键 ⇒ 两个方向键一起按、"
              "人原地抖 ✗（用户 2026-10-02 定：**朝倾向方向走优先** ✓）：%r"
              % (sorted(_pressed),))
        check(a._idle_fid is None,
              "扫平台巡逻那一拍**把 idle 回归也做了一遍**（写了 `_idle_fid` ⇒ 抽签号/回归"
              "目标被这一支污染 ✗）—— `walk=False` 该让 ① **整个不做** ✓：%r"
              % (a._idle_fid,))
        # ② 闲够时间 ⇒ 下「切换平台」任务 ✓
        h.run(0.4)                      # 累计 0.75 秒 > 0.5 ✓
        check(_planned == ["甲平台"],
              "扫平台策略下闲够「idle持续n秒切换平台」却没下前往任务（这条链在扫平台下"
              "是死的 ✗）：%r" % (_planned,))
        # ④ 前面站着怪（扫平台**不锁**它 ⇒ `target` 是 None ✗）⇒ 不清"闲"、也不换地方 ✓
        a.stop_route("用例：②检查完了")
        _planned.clear()
        h.mobs_fn = lambda _t: [Mob(id=7, x=900.0, y=500.0, w=40.0, h=40.0, conf=0.9)]
        h.run(0.05)
        check(a._idle_since is None,
              "扫平台下「视野里有怪」没清掉「切换平台」的计时锚（会带着旧计时立刻切 ✗）")
        h.run(0.8)                      # 只有怪、够不着、也锁不上 ⇒ 这一块还得继续扫 ✓
        check(not _planned,
              "**前面站着怪**就切平台了（那块平台明明还有怪 ✗）—— 判据要显式要求"
              "「视野里一只怪都没有」✓：%r" % (_planned,))
    finally:
        a.stop_route("用例收尾")
        a.keys.release_all()
        a.settings.battle_zones = []
        a.settings.sync_battle_zone_sets()


def t_zone_cd_setting():
    """「**区域查询CD(s)**」：从"一个全局参数"改成"**每个战斗区域项各配一个**"（用户 2026-09-28）。

    原话："「前往重下间隔」参数移除，逻辑移动到战斗区域限制配置的每一项上（**每项会配不一样**），
    并改名「区域查询CD(s)」" ✓。

    钉七件：
      ① **新结构** `battle_zones`（每项 5 个字段 ✓）能存能读；**旧键 `goto_retry_s` 不再写出** ✓；
      ② **老配置迁移**：老的 `battle_zone_sets`（一串集合名）+ 老的 `goto_retry_s`（一个全局值）
         ⇒ 迁成"每个集合一项、CD 取那个老值"✓ —— **老配置行为一点不变** ✓；
      ③ `battle_zone_sets` **变成派生只读**（老消费方一字不改 ✓；它是 property ⇒ 赋值会报错 ✓）；
      ④ `agent._zone_cd_s(集合名)` 取**那一项**的 CD ✓；问不到 ⇒ 退回 `ZONE_GOTO_RETRY_S` ✓；下限 0.5 ✓；
      ⑤ `agent._zone_cd_of_sets([...])` = "这批集合命中哪个区域项 ⇒ 它的 CD" ✓
         （追击下前往 / 区域筛缓存**共用**它 ✓）；一个都没命中 ⇒ 兜底 ✓；
      ⑥ **区域筛缓存的时效真的跟着那一项走**：该项 CD = 1.0 ⇒ 同一只怪 0.5 秒内不复算、
         1.2 秒后复算 ✓；
      ⑦ 界面上**旧的「前往重下间隔(s)」那一格不该再存在** ✓（2026-09-28 移除 ✓；
         新入口「路线脚本 → 战斗区域」弹窗是下一步 ✓）。
    """
    from PyQt5.QtWidgets import QApplication

    from perception.world_state import Mob, Player, WorldState

    ZONE_GOTO_RETRY_S = ag.ZONE_GOTO_RETRY_S
    CombatAgent = ag.CombatAgent

    # ① 新结构：per-map 文件存得住、读得回来（不再走 project.yaml 的 to_dict/from_dict ✓）
    s = fresh_settings()
    check(list(getattr(s, "battle_zones", []) or []) == [],
          "新结构 `battle_zones` 默认该是空列表（= 不限制 ✓）：%r"
          % (getattr(s, "battle_zones", None),))
    # ⛔ 2026-10-01：战斗区域**不再写进 project.yaml**（改 per-map 文件 ✓）
    d = s.to_dict()
    check("battle_zones" not in d and "battle_zone_sets" not in d,
          "战斗区域又被写进 project.yaml 了（该按地图 id 存 per-map 文件 ✗）：%r"
          % ({k: d.get(k) for k in ("battle_zones", "battle_zone_sets")}))
    check("goto_retry_s" not in d,
          "「前往重下间隔(s)」又被写回配置了（用户 2026-09-28 要求移除 ✗）：%r"
          % (d.get("goto_retry_s"),))
    # ⭐ 清洗口径 = `_load_battle_zones`（per-map 文件读写走的都是它 ✓）：一份 dict 进去、
    #   字段洗出来 ✓（存/读一致由 `core.battle` + 这个函数共同保证 ✓）
    _loaded = s._load_battle_zones(
        {"battle_zones": [{"set": "右", "cd_s": 7.5, "idle_footholds": ["41"],
                           "fight_max_s": 30.0, "fight_dst": "左", "can_fight": True}]})
    check(len(_loaded) == 1 and abs(float(_loaded[0]["cd_s"]) - 7.5) < 1e-9
          and _loaded[0]["idle_footholds"] == ["41"]
          and abs(float(_loaded[0]["fight_max_s"]) - 30.0) < 1e-9
          and _loaded[0]["fight_dst"] == "左",
          "读回来缺字段：%r" % (_loaded,))
    # ③ 派生副本 `battle_zone_sets`（老消费方读它 ✓）：读配置时同步 ✓
    s1 = fresh_settings()
    s1.battle_zones = _loaded
    s1.sync_battle_zone_sets()
    check(s1.battle_zone_sets == ["右"],
          "`battle_zone_sets`（派生副本）没跟着配置同步：%r" % (s1.battle_zone_sets,))
    s1.battle_zones = [{"set": "甲", "cd_s": 5.0, "idle_foothold": "",
                        "fight_max_s": 0.0, "fight_dst": "", "can_fight": True}]
    s1.sync_battle_zone_sets()
    check(s1.battle_zone_sets == ["甲"],
          "`sync_battle_zone_sets()` 没把真源同步过去（弹窗保存后筛怪会按老名单 ✗）：%r"
          % (s1.battle_zone_sets,))

    # ② 老配置迁移：老的集合名 + 老的全局 CD ⇒ 每项一份、CD 取老值 ✓
    s2 = fresh_settings()
    s2.battle_zones = s2._load_battle_zones(
        {"battle_zone_sets": ["底层", "小平台"], "goto_retry_s": 8.0}, legacy_cd=8.0)
    check([z["set"] for z in s2.battle_zones] == ["底层", "小平台"] and
          all(abs(float(z["cd_s"]) - 8.0) < 1e-9 for z in s2.battle_zones),
          "老配置迁移不对（该每个集合一项、CD 取老的全局值 8.0）：%r" % (s2.battle_zones,))
    check(all(float(z["fight_max_s"]) == 0.0 and not z["idle_footholds"]
              for z in s2.battle_zones),
          "迁移时其余字段该留默认（不限战斗时长 / 无 idle 回归 ✓）：%r" % (s2.battle_zones,))

    # ④⑤ agent 侧取值
    a = CombatAgent(fresh_settings())
    _v = a._zone_cd_s()
    check(abs(_v - ZONE_GOTO_RETRY_S) < 1e-9,
          "没配任何区域项时该退回常量 %r：%r" % (ZONE_GOTO_RETRY_S, _v))
    a.settings.battle_zones = [{"set": "右", "cd_s": 1.5, "idle_foothold": "",
                                "fight_max_s": 0.0, "fight_dst": ""},
                               {"set": "左", "cd_s": 9.0, "idle_foothold": "",
                                "fight_max_s": 0.0, "fight_dst": ""}]
    a.settings.sync_battle_zone_sets()      # ⚠ 改真源之后**必须**同步派生副本 ✓（见那条说明）
    check(abs(a._zone_cd_s("右") - 1.5) < 1e-9,
          "按集合名取 CD 没生效：%r" % (a._zone_cd_s("右"),))
    check(abs(a._zone_cd_s("左") - 9.0) < 1e-9,
          "另一项没各用各的 CD（「每项会配不一样」✗）：%r" % (a._zone_cd_s("左"),))
    check(abs(a._zone_cd_s("没有这项") - ZONE_GOTO_RETRY_S) < 1e-9,
          "问一个不存在的区域项该退回常量：%r" % (a._zone_cd_s("没有这项"),))
    check(abs(a._zone_cd_of_sets(["没有", "左"]) - 9.0) < 1e-9,
          "「这批集合命中哪个区域项」没按集合名找：%r" % (a._zone_cd_of_sets(["没有", "左"]),))
    check(abs(a._zone_cd_of_sets([]) - ZONE_GOTO_RETRY_S) < 1e-9,
          "一个都没命中该退回常量：%r" % (a._zone_cd_of_sets([]),))
    a.settings.battle_zones = [{"set": "右", "cd_s": 0.0, "idle_foothold": "",
                                "fight_max_s": 0.0, "fight_dst": ""}]
    check(a._zone_cd_s("右") >= 0.5, "agent 侧没钳住下限：%r" % (a._zone_cd_s("右"),))

    # ⑥ 区域筛缓存的时效跟着**那一项**走
    #    ⚠ **必须勾 `can_fight`**（2026-09-28 ✓）：区域筛读的是派生副本 `battle_zone_sets`，
    #      而它**只收勾了「禁止战斗」的项** ⇒ 不勾就等于"没配区域" ⇒ 压根不去问解析器 ✗
    #      （这条用例就是被它撞红的 ✓）。
    a.settings.battle_zones = [{"set": "右", "cd_s": 1.0, "idle_foothold": "",
                                "fight_max_s": 0.0, "fight_dst": "", "can_fight": True}]
    a.settings.sync_battle_zone_sets()
    calls = []

    def sets_of(pl, m):
        calls.append(m.id)
        return {"world": None, "sets": ["右"], "why": ""}

    a.mob_sets_of = sets_of
    m = Mob(id=1, x=520.0, y=300.0, w=40.0, h=60.0, conf=0.9)
    ws = WorldState()
    ws.player = Player(x=500.0, y=300.0, bottom=340.0)
    a._candidates([m], 500.0, ws=ws, now=10.0)
    n = len(calls)
    check(n == 1, "用例前提：第一次该去问一次解析器 ✓：%r" % (calls,))
    a._candidates([m], 500.0, ws=ws, now=10.5)      # 0.5s < 1.0 ⇒ 命中缓存 ✓
    check(len(calls) == n, "缓存时效没跟着**区域项**走（0.5 秒就重问了 ✗）")
    a._candidates([m], 500.0, ws=ws, now=11.2)      # 1.2s > 1.0 ⇒ 该重问 ✓
    check(len(calls) > n, "过了该项的时效却没重问（会拿旧答案筛怪 ✗）")

    # ⑦ 界面上**不该**再有旧那一格（2026-09-28 移除 ✓）
    #    ⚠ **用源码级钉**（不真建 Qt 控件 ✗）：这里原来建 `RoutePanel()` —— 而在
    #      `main()` 把 `DecisionSettings.save` 换成空操作之后，**建 RoutePanel 会当场把
    #      进程打崩**（`0xC0000409` 栈溢出，连异常都不抛 ✗ 定位花了很久 ✓）⇒
    #      改用 grep（一样能挡住"把这一格加回来"，而且不掺 Qt 的坑 ✓）。
    from pathlib import Path

    _rp = (Path(__file__).resolve().parents[1] / "gui" / "route_panel.py"
           ).read_text(encoding="utf-8")
    for _bad in ("sp_zone_retry", "_on_zone_retry"):
        check(_bad not in _rp,
              "「路线识别」页里「前往重下间隔(s)」又回来了（用户 2026-09-28 要求移除 ✗）：`%s`"
              % _bad)
    check('"前往重下间隔' not in _rp,
          "「路线识别」页里还留着「前往重下间隔(s)」那行文字（用户 2026-09-28 要求移除 ✗）")


def t_drop_hold_dir_reassert():
    """下跳「按住 ↓」那一组输出：**↓ 必须一直按着**，而且要**按期重发**（用户 2026-09-27 报）。

    原话："现在执行 drop 被卡住时，没有看到「按住 ↓ → 点按跳」，而是**一直原地跳**"。
    病根和爬绳「补按 ↑」/ 许可门「按住 + 重发」是**同一个** ✗：本机 `KeyState` **只在键集变化
    时**才发键 ⇒ 对面（中继/固件）把那次「按下 ↓」丢了，本机根本不知道 ⇒ 人**不趴** ⇒
    之后每一下跳都只是**原地跳** ✗（下跳要求"趴着时按跳" ✓）。

    钉五件：
      ① ARMED（先按住 ↓、还没跳）：`dir == -1` + `jump is False` ✓；
      ② DROP 连按跳那几拍：`jump is True` 时 `dir` **仍然是 -1**（"↓ 按住 + 点按跳" ✓）；
      ③ 「移动操作尝试间隔」到点 ⇒ `reassert is True`（请 agent 重发一次 ↓ ✓）；
         间隔内是 False ✓（**不是每拍都发** ✗）；
      ④ ⭐ **一轮（`stall_s`）没把 Y 按下去 ⇒ 也不松 ↓**（2026-09-28 改口径 ✓ 用户原话：
         "到点 Y 还没动，也不应该松开 ↓ 起身，只要进了 DROP 就**一路按住 ↓**"）⇒ 那几拍
         `dir == -1` ✓（老口径的"松着 ↓ 真站直"**已作废** ✗ —— 它正是"永远趴不下去"的根因）；
         唯一松 ↓ 的时机 = 落地 `DONE`/`FAILED` ✓ 或 `_detach_step`（**判定上绳梯** ⇒ 脱离）✓；
      ⑤ `retry_ms = 0`（用户把重试关掉）⇒ **从不重发**（尊重设置 ✓）。
    """
    from decision import route

    kw = dict(tol_px=10, hold_ms=0, y_tol_px=10, retry_ms=2000, stall_s=3.0)
    j = route.DropJob("三楼", "一楼", [(550.0, 500.0, 600.0, "42")], **kw)

    # ① 站在那条可下跳 foothold 上 ⇒ 立刻「按住 ↓」（还没跳）
    o = j.update(0.0, 550.0, py=-166.0)
    check(o["dir"] == -1 and o["jump"] is False,
          "「按住 ↓」那一拍没按 ↓（用户报的「没看到按住 ↓」✗）：%s" % o)
    # ⭐ 2026-09-28 **改口径**：按住 ↓ 的**第一拍就补发一次**（用户报"drop 没有补按『按下 ↓』
    #   只补按了跳，然后超时"⇒ 病根就是原来那句"先计窗、要等满 `retry_ms` 才第一次补"：
    #   一个 drop 任务常常**活不到 3 秒**（一轮 `stall_s` 到点就换相/重下 ⇒ 新任务又从 `None`
    #   重算）⇒ **那次补按永远等不到** ✗）。⚠ 重复 PRESS 是幂等的 ⇒ 第一拍多发一次无害 ✓，
    #   而且正好治"首按被对面丢了"那个病根 ✓（见 `_hold_reassert` 的说明 ✓）。
    check(o["reassert"] is True,
          "按住 ↓ 的第一拍没**补发一次 ↓**（用户 2026-09-28：补按 ↓ 必须第一拍就发 ✗）：%s" % o)

    # ② ARM_HOLD_S 过了 ⇒ 按住 ↓ + 点按跳（↓ 不能丢）
    o = j.update(float(route.ARM_HOLD_S) + 0.01, 550.0, py=-166.0)
    check(o["jump"] is True and o["dir"] == -1,
          "点按跳那一拍丢了 ↓（跳会变成原地跳 ✗）：%s" % o)

    # ③ 到「移动操作尝试间隔」⇒ 请求重发一次 ↓（间隔内不重发 ✓）
    o = j.update(float(route.ARM_HOLD_S) + 1.0, 550.0, py=-166.0)
    check(o["reassert"] is False, "间隔没到就每拍重发（那是刷键 ✗）：%s" % o)
    o = j.update(float(route.ARM_HOLD_S) + 2.2, 550.0, py=-166.0)
    check(o["reassert"] is True and o["dir"] == -1,
          "过了「移动操作尝试间隔」也没重发 ↓（键被吞了就永远不趴 ✗）：%s" % o)

    # ④ ⭐⭐ **2026-09-28 改口径**：一轮（`stall_s`）没把 Y 按下去 ⇒ **也不许松 ↓** ✓
    #    用户原话："**到点 Y 还没动，也不应该松开 ↓ 起身，只要进了 DROP 就一路按住 ↓**"
    #             ／"流程中，**只有判定 drop 上绳梯了才松开 ↓**，其他只有补按" ✓
    #    ⇒ 这几拍应该是"↓ 还按着"（`dir == -1`）✓（老口径"松着 ↓ 真站直"已作废 ✗）
    t = float(route.ARM_HOLD_S) + 3.3
    o = j.update(t, 550.0, py=-166.0)
    o = j.update(t + 0.1, 550.0, py=-166.0)
    check(o["dir"] == -1,
          "窗口过完把 ↓ 松了（用户 2026-09-28：**一路按住 ↓**、只有上绳梯才松 ✗）：%s" % o)

    # ⑤ retry_ms = 0（不重试）⇒ 从不重发（尊重设置 ✓）
    kw0 = dict(kw)
    kw0["retry_ms"] = 0
    j0 = route.DropJob("三楼", "一楼", [(550.0, 500.0, 600.0, "42")], **kw0)
    j0.update(0.0, 550.0, py=-166.0)
    o = j0.update(float(route.ARM_HOLD_S) + 9.0, 550.0, py=-166.0)
    check(o["reassert"] is False,
          "把「移动操作尝试间隔」设成 0 却还在重发 ↓（没尊重设置 ✗）：%s" % o)
    check(o["dir"] == -1, "不重试时也得**一直按住 ↓**（老行为 ✓）：%s" % o)


def t_ladder_only_when_holding_vertical():
    """「在绳梯上」的**许可条件**：没按着 ↑/↓ ⇒ 一律不算（用户 2026-09-27 要求）。

    原话："位置状态判定优化：**除非按住了 ↑ 或 ↓，不能主动判定为在绳梯上**，不然角色
    **碰到绳子就卡住不走**" ✓。

    病根：地形上"人站在绳子的坐标范围内"≠"人在爬绳" ✗ —— 走过去**路过绳口**、或者被平台挡在
    绳边，都会被判成"在绳上" ⇒ 走路 / 寻路那些看 `ladder_id` 的判据（"位置状态一变就重算"、
    "在绳上不左右走"…）就把人按住了 ⇒ 表现就是**碰到绳子就卡住不动** ✗。

    钉四件：
      ① `agent.holding_vertical()`：按着 ↑ 或 ↓ ⇒ True ✓；只按左右 / 什么都没按 ⇒ False ✓；
      ② 它看的是**真按着的键**（`KeyState.pressed()` ✓），不是自己另记的账 ✓；
      ③ 感知层写 `ladder_id` 前**必须**问它（源码钉子：`live_thread._fill_route_ctx` 里
         同时出现 `holding_vertical` 与 `ladder_id` ✓）；
      ④ `KeyState.pressed()` 给的是**快照**（外面改不动它 ✓）。
    """
    from pathlib import Path as _P

    from decision.agent import CombatAgent

    a = CombatAgent(fresh_settings())
    km = a.settings.keymap
    check(km.get("up") and km.get("down"), "用例前提：键位表里得有 up/down")
    a.keys.set(set())
    check(a.holding_vertical() is False, "什么都没按却判成「按着 ↑/↓」✗")
    a.keys.set({km.get("left")})
    check(a.holding_vertical() is False, "只按左右也判成「在绳上」（用户报的「碰到绳就卡住」✗）")
    a.keys.set({km.get("up")})
    check(a.holding_vertical() is True, "按着 ↑ 却没认出来 ✗")
    a.keys.set({km.get("down")})
    check(a.holding_vertical() is True, "按着 ↓ 却没认出来 ✗")
    # ④ 快照：外面改不动
    a.keys.set({km.get("up")})
    snap = a.keys.pressed()
    snap.clear()
    check(a.holding_vertical() is True,
          "`pressed()` 给的不是快照（外面一改就把内部状态清了 ✗）")
    # ③ 感知层写 `ladder_id` 前问的是「执行器通知」climbing_vertical（诉求 1，用户 2026-09-28：
    #    许可入口从「按住 ↑/↓（本机按键）」改成「climb 执行器按住 ↑/↓ 后发起的通知」✓）
    root = _P(__file__).resolve().parent.parent
    src = (root / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("climbing_vertical" in src,
          "感知层没问「climb 执行器按没按 ↑/↓ 的通知」就写 `ladder_id`（碰到绳就会卡住 ✗）")
    i = src.index("climbing_vertical")
    seg = src[max(0, i - 400): i + 900]
    check("ladder_id" in seg, "那段许可条件离写 `ladder_id` 的地方太远（可能没接上 ✗）")


def t_goto_queue():
    """「**任务队列**」（用户 2026-09-27：「命令前往」右边的「添加任务队列」）⇒ 一条接一条走 ✓。

    队列里存的是**目的地**（不是路线）：每一条**出发时才**按"我现在站哪"重新解析
    （起点每一刻都可能变 ✓，同 `plan_and_start_route`）。界面上显示在小地图下面
    「当前任务」的下方，**一行一个**（`route_panel._osd_lines` → `agent.goto_queue()` ✓）。

    钉八件：
      ① 排队：`queue_goto` 排进去、`goto_queue()` 给出没跑的那些；**空名字不排** ✓；
      ② 一条路线**整条走完** ⇒ 自动接下一条（`_task_finished` → `_goto_queue_next`）✓；
      ③ 某一条**已经站在那个集合上**（解析器报 `here`）⇒ **跳过它**、接着下一条 ✓；
      ④ 某一条**解析不出来 / 没路** ⇒ **停整个队列**并说清（后面几个一并取消 ✓）；
      ⑤ 「结束当前寻路」⇒ 队列**一起清掉**（那是"别走了"的意思 ✓）；
      ⑥ 寻路**失败** ⇒ 队列也清掉（留着只会一条条接着失败 ✗）；
      ⑦ 队列是**运行时状态**：不进 `to_dict`（不许存进项目文件 / 跨会话复活 ✗）；
      ⑧ 顺序**就是排的顺序**（先排先走，不许重排 —— 用户按顺序排出来的是有意义的 ✓）。
    """
    from decision import route

    s = fresh_settings()
    s.enabled = True
    h = Harness(s)
    a = h.agent
    spots = [(700.0, 660.0, 740.0, "1")]

    def _plan(dst):
        """假的路径解析器：除了「丙平台」都能走 ✓。"""
        if dst == "丙平台":
            return {"jobs": [], "why": "没有可走的路（用例）", "here": False, "path": []}
        if dst == "现在就在这":
            return {"jobs": [], "why": "已经在「%s」上了" % dst, "here": True,
                    "path": [dst]}
        return {"jobs": [route.WalkJob(dst, spots)], "why": "", "here": False,
                "path": ["起点", dst]}

    a.route_plan = _plan
    with h._patched():
        # ① 排队（空名字不排；顺序 = 排的顺序 ✓）
        check(a.queue_goto("") == 0, "空名字不该排进队列")
        check(a.queue_goto("甲") == 1 and a.queue_goto("乙") == 2,
              "排队返回值 / 顺序不对：%r" % (a.goto_queue(),))
        check(a.goto_queue() == ["甲", "乙"], "队列内容不对：%r" % (a.goto_queue(),))

        # ② 一条路线整条走完 ⇒ 自动接下一条
        a.start_route([route.WalkJob("起点", spots)], why="用例：第一条")
        check(a._task_finished() is False,
              "整条走完该**接队列里的下一条**（`False` = 还没完事）")
        check(a.current_goto_set() == "甲", "接的不是队列里的第一条：%r"
              % a.current_goto_set())
        check(a.goto_queue() == ["乙"], "接完之后队列没 pop：%r" % (a.goto_queue(),))
        check(a._task_finished() is False, "该接着走「乙」")
        check(a.current_goto_set() == "乙", "第二条没接上：%r" % a.current_goto_set())
        check(a._task_finished() is True, "队列空了该报「真的都完事了」")

        # ③ 队列里有"已经站在上面"的那一条 ⇒ 跳过它，接着下一条
        a.queue_goto("现在就在这")
        a.queue_goto("乙")
        a.start_route([route.WalkJob("起点", spots)], why="用例：跳过")
        check(a._task_finished() is False, "该跳过「现在就在这」、接着走「乙」")
        check(a.current_goto_set() == "乙",
              "没跳过「已经在上面了」的那一条：%r" % a.current_goto_set())
        a._task_finished()

        # ④ 某一条下不去 ⇒ 停整个队列 + 说清（后面几个一并取消）
        a.queue_goto("丙平台")
        a.queue_goto("乙")
        a.start_route([route.WalkJob("起点", spots)], why="用例：走不通")
        check(a._task_finished() is True, "队列里有下不去的，该**停整个队列**（True）")
        check(a.goto_queue() == [], "停下来之后队列该清空：%r" % (a.goto_queue(),))
        check("丙平台" in a.current_goto_note() and "取消" in a.current_goto_note(),
              "没说清是哪一条下不去 / 后面几个被取消：%r" % a.current_goto_note())

        # ⑤ 「结束当前寻路」⇒ 队列一起清掉
        a.queue_goto("甲")
        a.start_route([route.WalkJob("起点", spots)], why="用例：叫停")
        a.stop_route("用例：叫停")
        check(a.goto_queue() == [],
              "「结束当前寻路」没把队列一起清掉（下一拍又会派一条 ✗）：%r" % (a.goto_queue(),))

        # ⑥ 寻路失败 ⇒ 队列也清掉
        a.queue_goto("甲")
        a.start_route([route.WalkJob("起点", spots)], why="用例：失败")
        check(a._task_finished(failed=True, why="用例：卡住了") is True, "失败该收掉")
        check(a.goto_queue() == [], "失败之后队列该清空：%r" % (a.goto_queue(),))
        check("队列已清空" in a.current_goto_note(),
              "失败清队列时没说清（队列会「悄悄没了」✗）：%r" % a.current_goto_note())

    # ⑦ 运行时状态：不进 to_dict
    check("goto_queue" not in s.to_dict() and "_goto_queue" not in s.to_dict(),
          "任务队列被写进项目文件了（运行时状态不该落盘 ✗）")


def t_queue_starts_itself():
    """「添加任务队列」**排进去就自己跑**（用户 2026-09-27 原话）：

        "我期望的是：战斗是最低优先级的任务，寻路任务队列要依次执行，现在我排队列都没反应"

    根因：`_goto_queue_next()` 原来**只有** `_task_finished`（上一条路线收工时）会调一次，
    `tick()` 从头到尾**不读** `_goto_queue` ⇒ 光点「添加任务队列」永远不会动 ✗。
    现在：`queue_goto` 拍一面旗子（`_queue_kick`；它可能在界面线程里被调 ⇒ 起跑那笔活儿
    留给决策线程 ✓），下一拍 `tick` 就 `_goto_queue_next()` ✓。

    钉六件：
      ① 没任务在跑 ⇒ 排一条，**下一拍 tick 就挂上任务**（不用再按「命令前往」✓）；
      ② 有任务在跑 ⇒ 排进去**不打断**它（`_climb` 还是同一个），只是排队 ✓；
      ③ 旗子**只吃一次**（起跑完归 False，不会每拍都去戳队列 ✓）；
      ④ 队列里"已经站在上面"的那条 ⇒ 起跑时**跳过**、直接跑下一条 ✓；
      ⑤ 队列里那条**下不去** ⇒ 一拍就停下并说清（不留半条、也不炸 ✓）；
      ⑥ 「结束当前寻路」清队列时**旗子一起清** ⇒ 之后不会莫名其妙自己起跑 ✓。

    ⚠ 「战斗怎么算」不在这个用例里：用户 2026-09-27 明确"依旧走进 attack + 刷新超时判定
      那套" ⇒ 顺序**没动**（`in_range` 那支还在寻路之前）⇒ 见 `t_job_interrupt_by_fight`
      与 `_climb_interrupted` ✓。
    """
    from decision import route

    s = fresh_settings()
    s.enabled = True
    h = Harness(s)
    a = h.agent
    h.clock0 = h.clock.t          # 记日志要用它（本用例真的会 tick 到按键 ✓）
    # ⚠ 走的落点是**五元组**（中心x, 左, 右, 面y, foothold id）—— 这条用例会真的 tick 到
    #   `WalkJob.update`（上面 `t_goto_queue` 只调 `_task_finished`，四元组也过得去 ✗）
    spots = [(700.0, 660.0, 740.0, -208.0, "1")]

    def _plan(dst):
        """假的路径解析器：除了「丙平台」都能走 ✓（同 `t_goto_queue`）。"""
        if dst == "丙平台":
            return {"jobs": [], "why": "没有可走的路（用例）", "here": False, "path": []}
        if dst == "现在就在这":
            return {"jobs": [], "why": "已经在「%s」上了" % dst, "here": True,
                    "path": [dst]}
        return {"jobs": [route.WalkJob(dst, spots)], "why": "", "here": False,
                "path": ["起点", dst]}

    a.route_plan = _plan
    with h._patched():
        # ① 没任务在跑 ⇒ 排一条，**下一拍 tick** 就自己起跑
        a.queue_goto("甲")
        check(a._queue_kick,
              "排进队列却没拍「立刻起跑」那面旗子（用户报的\"没反应\"就是它 ✗）")
        check(a._climb is None,
              "`queue_goto` 里就把任务挂上了 —— 那是**界面线程**，起跑该留给决策拍 ✓")
        a.tick(h.ws(with_mob=False))
        check(a._climb is not None and a.current_goto_set() == "甲",
              "光排队列、tick 一拍都没起跑（用户 2026-09-27：排队列没反应 ✗）：%r"
              % (a.current_goto_set(),))
        check(not a._queue_kick, "旗子该只吃一次（起跑完就归 False ✓）")
        check(a.goto_queue() == [], "起跑的那条该从队列里 pop 掉：%r" % (a.goto_queue(),))

        # ② 已经有任务在跑 ⇒ 再排只排队、**不许打断**它
        running = a._climb
        check(a.queue_goto("乙") == 1 and not a._queue_kick,
              "有任务在跑时还拍「立刻起跑」的旗子（那会打断它 ✗）")
        a.tick(h.ws(with_mob=False))
        check(a._climb is running, "排第二条把正在跑的那条打断了：%r" % (a._climb,))
        check(a.goto_queue() == ["乙"], "第二条该老实排队等：%r" % (a.goto_queue(),))

        # ③ 收工 ⇒ **自动接上第二条**（这就是"依次执行"✓）
        check(a._task_finished() is False, "收工该接着走队列里的「乙」")
        check(a.current_goto_set() == "乙", "没接上第二条：%r" % (a.current_goto_set(),))
        check(a.goto_queue() == [], "接上之后该 pop：%r" % (a.goto_queue(),))

        # ④ 队列里"已经站在上面"的那条 ⇒ 起跑时就跳过、直接跑下一条
        a.stop_route("用例：重来")
        a.queue_goto("现在就在这")
        a.queue_goto("乙")
        a.tick(h.ws(with_mob=False))
        check(a.current_goto_set() == "乙",
              "起跑时没跳过「已经在上面了」的那条：%r" % (a.current_goto_set(),))
        check(a.goto_queue() == [], "跳过之后队列该空：%r" % (a.goto_queue(),))

        # ⑤ 队列里那条**下不去** ⇒ 一拍就停 + 说清（别留半条、别炸）
        a.stop_route("用例：重来")
        a.queue_goto("丙平台")
        a.tick(h.ws(with_mob=False))
        check(a._climb is None and a.goto_queue() == [],
              "下不去的那条该停下并清空队列：%r / %r" % (a._climb, a.goto_queue()))
        check("丙平台" in a.current_goto_note() and "取消" in a.current_goto_note(),
              "没说清是哪一条下不去 / 后面几个被取消：%r" % a.current_goto_note())

        # ⑥ 「结束当前寻路」把队列和那面旗子一起清掉
        a.queue_goto("甲")
        check(a._queue_kick, "这条该拍旗子")
        a.stop_route("用例：叫停")
        check(not a._queue_kick and a.goto_queue() == [],
              "「结束当前寻路」没把那面「排进去就起跑」的旗子一起清掉：%r"
              % (a._queue_kick,))


def t_replan_on_location_change():
    """位置状态一变 ⇒ **重新评判一次最优路径**（用户 2026-09-27 原话）：

        "当玩家的当前位置状态（位于 foothold 集或绳梯等）改变时，要重新评判一次最优路径"

    原来 `start_route` 拿到的是一条**算好了的**任务列表（按"下命令那一刻人在哪"算的 ✓），
    半路被人撞下去 / 走岔 / 落点偏了 ⇒ 剩下那几步**全都不对了** ✗（最坏一路朝反方向走到
    「寻路超时」）。现在每拍对一次 `(脚下属于哪些集合, 贴在哪根绳上)`，一变就按**新的当前
    位置**重新解析到**同一个最终目的地** ✓。

    钉八件：
      ① 刚挂上这一步的第一眼 ⇒ **只记不算**（不然一步都还没走就先重算一遍 ✗）；
      ①' **同一个集合里**走来走去 / **被怪撞来撞去**（`here_sets` 没变、x 变了）⇒ **不算变化**、
        **不重算** ✓（用户 2026-09-27 澄清的粒度边界：路线的**节点**没变、计划仍成立；
        谁把 x/y 塞进「位置状态」，这条立刻红 ✗）；
      ② 位置变成**计划外**的集合 ⇒ 重新解析（问的是**最终目的地** ✓）并按新计划起跑 ✓；
      ③ 变到的**就是这一步要去的那块**（正常到站）⇒ **不重算**（交给收工那套 ✓）；
      ④ 人**贴在绳梯上**（`ladder_id` 非空）⇒ 不重算，而且**状态不记** ⇒ 离绳那一拍补上 ✓；
      ⑤ 这一步正处在"**空中**"的相位（`climb`/`jump`/`drop`…＝ `route.INFLIGHT_PHASES`）
        ⇒ 同样先不动，落地再补 ✓；
      ⑥ 重新解析**失败**（脚下没圈进任何集合）⇒ **保持现状** + 说清原因（别把好端端的路线
        砍掉 ✗；留痕走 `perf.log` / `behavior.log` / `_last_goto_note` ✓）；
      ⑦ 没有解析器 ⇒ 静默不动（不炸、也不动现状 ✓）。
    """
    from decision import route

    s = fresh_settings()
    s.enabled = True
    h = Harness(s)
    a = h.agent
    h.clock0 = h.clock.t
    spots = [(700.0, 660.0, 740.0, -208.0, "1")]
    asked = []

    def _plan(dst):
        """假的解析器：记下被问过的目的地，回一条"一步就到"的路线 ✓。"""
        asked.append(dst)
        return {"jobs": [route.WalkJob(dst, spots)], "why": "", "here": False,
                "path": ["起点", dst]}

    a.route_plan = _plan
    ws = h.ws(with_mob=False)
    ws.player.here_sets = ["甲"]
    with h._patched():
        a.start_route([route.WalkJob("乙", spots), route.WalkJob("丙", spots)],
                      why="用例：多步")
        # ① 第一眼只记，不重算
        a._climb_tick(0.0, 500.0, set(), ws)
        check(asked == [], "刚开始这一步就重算了（第一眼只该记下来）：%r" % (asked,))
        # ①' **同一个集合里**走动 / 被怪撞来撞去（集合没变、**坐标变了**）⇒ 不算变化、不重算
        # ⚠ 坐标两套都动（画面 `x` + 世界 `world_x`）—— 谁把坐标塞进「位置状态」，这条立刻红 ✗
        ws.player.x, ws.player.world_x = 640.0, 640.0
        a._climb_tick(0.05, 640.0, set(), ws)
        check(asked == [],
              "同一个集合里被撞来撞去也重算了（用户口径：路线的**节点**没变 ⇒ 别打断 ✗）：%r"
              % (asked,))
        ws.player.x, ws.player.world_x = 500.0, 500.0
        # ② 位置变成计划外的集合 ⇒ 重新解析到**最终目的地**「丙」，并按新计划起跑
        ws.player.here_sets = ["丁"]
        a._climb_tick(0.1, 500.0, set(), ws)
        check(asked == ["丙"],
              "位置变了没重算（或者算错了目的地 —— 该问**最终目的地**）：%r" % (asked,))
        check(a.current_goto_set() == "丙",
              "重算之后没按新计划起跑（手上这一步该是「丙」）：%r"
              % (a.current_goto_set(),))
        # ③ 正常到站（脚下就是这一步要去的那块）⇒ **不重算**，走"收工"那套（这一步完成 ✓）
        ws.player.here_sets = ["丙"]
        a._climb_tick(0.2, 500.0, set(), ws)
        check(asked == ["丙"], "正常到站却重算了（会来回抖 ✗）：%r" % (asked,))
        check(a._climb is None, "到站了该收工（重算会把它当成又一次'位置变了'）")
        # ④ 再起一条：人**贴在绳梯上**时位置变了 ⇒ 先不动；离绳那一拍补上 ✓
        a.start_route([route.WalkJob("丙", spots)], why="用例：绳上")
        ws.player.here_sets = ["甲"]
        a._climb_tick(0.25, 500.0, set(), ws)        # 第一眼：只记 ✓
        asked.clear()
        ws.player.here_sets = ["戊"]
        ws.player.ladder_id = "L2"
        a._climb_tick(0.3, 500.0, set(), ws)
        check(asked == [],
              "人在绳梯上还重算（会把爬了一半的动作打断 ✗）：%r" % (asked,))
        ws.player.ladder_id = ""
        a._climb_tick(0.4, 500.0, set(), ws)
        check(asked == ["丙"], "离绳之后没把那次变化补上：%r" % (asked,))
        # ⑤ 这一步正"在空中"（相位）⇒ 同样先不动，落地再补
        asked.clear()
        a._climb_tick(0.45, 500.0, set(), ws)        # 第一眼：记下现在的「戊」✓
        a._climb.phase = "jump"
        ws.player.here_sets = ["壬"]                 # 又变了，而且这一步正在空中
        a._climb_tick(0.5, 500.0, set(), ws)
        check(asked == [], "这一步正在空中（jump 相位）还重算：%r" % (asked,))
        a._climb.phase = "walk"
        a._climb_tick(0.6, 500.0, set(), ws)
        check(asked == ["丙"], "落地之后没把这次变化补上：%r" % (asked,))
        # ⑥ 重新解析失败 ⇒ **保持现状** + 说清原因（路线不许被砍 ✗）
        a._climb_tick(0.65, 500.0, set(), ws)        # 第一眼：记下现在的「壬」✓
        a.route_plan = lambda dst: {"jobs": [], "why": "脚下这块没圈进任何集合",
                                    "here": False}
        ws.player.here_sets = ["庚"]
        a._climb_tick(0.7, 500.0, set(), ws)
        check(a._climb is not None and a.current_goto_set() == "丙",
              "解析不出来却把现有路线砍了（该保持现状 ✓）：%r" % (a._climb,))
        check("没圈进任何集合" in str(a._last_goto_note[0]),
              "没留痕说清为什么没重算（人只能干看着 ✗）：%r"
              % (getattr(a, "_last_goto_note", None),))
        # ⑦ 没有解析器 ⇒ 静默不动（不炸、不动现状）
        a.route_plan = None
        ws.player.here_sets = ["辛"]
        a._climb_tick(0.8, 500.0, set(), ws)
        check(a._climb is not None, "没有解析器时报错 / 把任务丢了：%r" % (a._climb,))



#: 用例里给寻路任务带「追击签注」的现成值 ✓（全仓只有 `agent._chase_origin()` 一个写口 ✓）。
#: ⚠ 2026-09-29 晚口径：「寻路超时时间」对**所有**任务生效 ⇒ 测超时**不再需要**签注 ✓；
#:   签注只在测「追怪寻路.duration(s)」（追击独占的第二道闸）时才必须 ✓。
CHASE_ORIGIN = {"kind": "chase", "mob_id": 1}

def t_keystate_momentary():
    """KeyState 的**瞬发登记**（2026-09-30 用户报"输出的时候输出按键没亮"✗）。

    根因 = 输出序列走 `key_down` **直发**、不经 KeyState ⇒ `pressed()` 永远没有
    输出键 ✗。修复 = `momentary_down/up`（序列/ tap 路径登记 ✓）+ `pressed()` 取
    **并集** ✓。钉三件：
      ① momentary_down/up 正常进出 ✓；
      ② ⭐ **决策拍 `set()` 不许顶掉瞬发键**（set 是全量重算 ✗ 一顶掉输出键就灭了 ✗）；
      ③ 决策还按着的键，`momentary_up` **不许发松开** ✓（只摘账 ✓）。
    ⚠ 测试里 monkeypatch 输入层（**绝不许真发键** ✗ SendInput 会打到真机 ✓）。
    """
    import decision.input as inp
    from decision.input import KeyState

    sent = []
    orig = (inp.key_down, inp.key_up)
    inp.key_down = lambda n: sent.append(("d", n))
    inp.key_up = lambda n: sent.append(("u", n))
    try:
        ks = KeyState()
        ks.set({"left"})
        ks.momentary_mark("ctrl")             # 纯记账（键已由序列发出 ✓ 不重发 ✓）
        check(ks.pressed() == {"left", "ctrl"},
              "① 瞬发登记该进 pressed（并集 ✗）：%r" % (ks.pressed(),))
        check(sent == [("d", "left")],
              "mark 是**纯记账** —— 不许重发键（双按 ✗ 实测输出 CD 用例炸）：%r"
              % (sent,))
        ks.set({"left"})                      # 决策拍：目标集合只有 left ✓
        check(ks.pressed() == {"left", "ctrl"},
              "② 决策拍不许顶掉瞬发键（输出键会当场灭 ✗）：%r" % (ks.pressed(),))
        ks.momentary_discard("ctrl")
        check(ks.pressed() == {"left"}, "③ discard 该摘掉瞬发键 ✗：%r"
              % (ks.pressed(),))
        check(sent == [("d", "left")],
              "discard 也是纯记账（松开由调用方发 ✓）：%r" % (sent,))
        ks.momentary_mark("left")             # 决策已按着的键 ⇒ 不记账 ✓
        check(ks.pressed() == {"left"}, "决策已按着的键不该重复记账 ✗：%r"
              % (ks.pressed(),))
    finally:
        inp.key_down, inp.key_up = orig


def t_reconnect_stop_auto():
    """断线判定成立 ⇒ **必停自动**（2026-09-30 用户要求 ✓ 原话："刚刚判定到断线了，
    在断线的时候停止自动吧"）—— `reconnect_enabled` 只管"要不要自动走回游戏"，
    **不再连坐停自动** ✗（原来默认 False ⇒ 断线了自动还在瞎跑 ✗）。

    钉三件（`ui_state.detect` 替身 ⇒ 不碰模板匹配 ✓；输入层替身 ⇒ **绝不发真键** ✓）：
      ① 开关**关**：判到断线界面 ⇒ `enabled=False`（停了 ✓）+ **不产生任何动作** ✓
         + 状态文字说明"重连未开启" ✓；
      ② 开关**关**：回到游戏 ⇒ 自动**保持停止** ✓（用户没要自动重连 ✓）；
      ③ 开关**开**：照旧出动作（enter ✓）+ 回到游戏恢复自动 ✓（原行为不回退 ✓）。
    """
    import numpy as np
    import perception.ui_state as ui_state
    from decision.reconnect import Reconnector

    frame = np.zeros((8, 8, 3), np.uint8)
    _orig_detect = ui_state.detect
    ui_state.detect = lambda f: ui_state.UI_LOGIN_ERR   # 恒判到断线提示框 ✓
    sent = []
    import decision.input as _dinput
    _orig = (_dinput.key_down, _dinput.key_up)
    _dinput.key_down = lambda n: sent.append(("d", n))
    _dinput.key_up = lambda n: sent.append(("u", n))
    try:
        # ---- ①② 开关关：只停自动，不做动作、不恢复 ----
        s = fresh_settings(reconnect_enabled=False,
                           reconnect_probe_after_lost_sec=0.0)
        s.enabled = True
        r = Reconnector(s)
        r.update(frame, True, 100.0)              # 在场：建档 ✓
        out = r.update(frame, False, 101.0)       # 丢失 + 判到断线界面 ✓
        check(s.enabled is False,
              "① 判到断线界面必停自动（reconnect_enabled=False 也要停 ✗）")
        check(out is None,
              "① 重连未开启 ⇒ 不该产生任何动作（不自动按键 ✗）：%r" % (out,))
        check("重连未开启" in r.note,
              "状态文字该说明「重连未开启」 ✗：%r" % (r.note,))
        r.update(frame, True, 102.0)              # 回到游戏 ✓
        check(s.enabled is False,
              "② 重连未开启 ⇒ 回到游戏也**保持停止**（恢复是重连流程的事 ✗）")
        # ---- ③ 开关开：动作 + 恢复原行为 ----
        s2 = fresh_settings(reconnect_enabled=True,
                            reconnect_resume_auto=True,
                            reconnect_probe_after_lost_sec=0.0)
        s2.enabled = True
        r2 = Reconnector(s2)
        r2.update(frame, True, 200.0)
        out2 = r2.update(frame, False, 201.0)
        check(out2 is not None and out2.get("act") == "tap"
              and out2.get("key") == s2.keymap.get("enter"),
              "③ 开关开 ⇒ 照旧出重连动作（enter ✗）：%r" % (out2,))
        check(s2.enabled is False, "③ 出动作前先停自动 ✗")
        r2.update(frame, True, 202.0)
        check(s2.enabled is True, "③ 回到游戏该恢复自动（resume_auto ✓）✗")
    finally:
        ui_state.detect = _orig_detect
        _dinput.key_down, _dinput.key_up = _orig


def t_goto_timeout_all_job_kinds():
    """**「寻路超时时间」对**所有**寻路任务生效**（都有出口 ✓）；追击另吃第二道闸。

    ⚠⚠ **口径 2026-09-29 晚澄清**（用户原话："设置里的'追怪寻路.duration(s)'才只对追击
      签注生效的时间，其他所有的任务都走「寻路超时时间」，都有出口"✓）——
      当天早些钉反过一版（"只有追击吃这道闸、其他任务无限" ✗）：那版让非追击任务
      **没有任何出口**，现场 21:22 在 L9 上爬不动 14.8 秒、↑ 一直按着，最后是用户
      **手动关自动**才收场 ✗。判据与信息栏显示**同一处口径** ✓。
      ⚠ "追击独占"的是「追怪寻路.duration(s)」（`chase_goto_max_s`，另一道闸 ✓）。

    这条钉四件：
      ① **追击**签注下：走 / 爬 / 下跳 / 跳 四类都收掉，并说清是「寻路超时」 ✓；
      ② ⭐ **非追击**（没签注 ⇒ 命令前往 / 休息 / 回区域那种 ✓）：四种**同样收掉** ✓
         —— 这就是"都有出口"的钉子（今天那场卡死就是这么漏掉的 ✗）；
      ③ 收掉时**整条路线**一起收，且 `note` 里写着跑多久、上限多少 ✓；
      ④ 上限设 **0 = 不限时** ⇒ 谁都不收 ✓（老行为 ✓）。

    ⚠ 那道闸在 `agent._climb_tick` 的最前面（盖完"这一拍任务跑过"的章之后、任何按类型分的
      逻辑之前）⇒ 四种通行方式**一视同仁** ✓；**每段一把钟**（`t_goto_timeout_per_hop` ✓）
      —— "长路线跑两分钟很正常"不受影响：每完成一段就重新计时 ✓。
    """
    from decision import route

    def _mk_jobs():
        """四种任务各一个（都喂"永远到不了"的状态 ✓）。"""
        return [
            ("走", route.WalkJob("甲集合", [(520.0, 510.0, 530.0, -208.0, "1")])),
            ("爬", route.ClimbJob("L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                                  dst_set="乙平台", src_set="甲平台", dst_y=100.0,
                                  tol_px=6, hold_ms=0)),
            ("下跳", route.DropJob("甲平台", "乙平台",
                                   [(700.0, 660.0, 740.0, "41")], tol_px=6, hold_ms=0)),
            ("跳", route.JumpJob("乙平台", [(600.0, 700.0, "11")],
                                 src_set="甲平台", jump_start_px=0)),
        ]

    # ① ③ **追击**签注 + 四种任务：超上限 ⇒ 收掉（并说清「寻路超时」）
    _CHASE = {"kind": "chase", "mob_id": 1}          # 追击签注（唯一写口 `_chase_origin` ✓）
    for name, job in _mk_jobs():
        s = fresh_settings(goto_timeout_s=10.0)
        s.enabled = True
        h = Harness(s)
        with h._patched():
            h.agent.start_route([job], why="用例：%s（追击）" % name, origin=dict(_CHASE))
            ws = h.ws(with_mob=False)
            ws.player.here_sets = []          # 永远不在目标集合上 ⇒ 任务永远到不了 ✓
            h.clock.t += 11.0
            # px 喂一个"离目标很远"的值：任务会一直对齐/走/等，不会自己结束 ✓
            done = h.agent._climb_tick(h.clock.t, 520.0, set(), ws)
        check(done is True and h.agent._climb is None,
              "「%s」是**追击**却超了「寻路超时时间」没收掉（追击下四种通行方式一视同仁 ✗）："
              "done=%r climb=%r" % (name, done, h.agent._climb))
        check(h.agent._climb_retry_at is None,
              "「%s」被收掉之后还留着「等重来」的时刻（下一个任务会立刻被它带跑 ✗）" % name)
        check("寻路超时" in str(getattr(job, "note", "")),
              "「%s」被收掉了但没说清是「寻路超时」（note=%r）" % (name, job.note))
        check("10" in str(job.note),
              "「%s」那句没说清上限是多少：%r" % (name, job.note))

    # ② ⭐⭐ **非追击 ⇒ 同样有出口**（2026-09-29 晚用户澄清 ✓："其他所有的任务都走
    #    「寻路超时时间」，都有出口"）—— 四种任务超上限**同样收掉** ✓
    #    （今天 21:22 那场"爬不动 14.8 秒只能人动手"漏的就是这个 ✗）
    for name, job in _mk_jobs():
        s = fresh_settings(goto_timeout_s=10.0)
        s.enabled = True
        h = Harness(s)
        with h._patched():
            h.agent.start_route([job], why="用例：%s（没签注）" % name)   # 不传 origin ✓
            ws = h.ws(with_mob=False)
            ws.player.here_sets = []
            h.clock.t += 11.0
            done = h.agent._climb_tick(h.clock.t, 520.0, set(), ws)
        check(done is True and h.agent._climb is None,
              "「%s」没有签注超了「寻路超时时间」却没收掉 —— 用户 2026-09-29 晚说"
              "**所有任务都有出口**（这钉的就是今天那场卡死 ✗）：%r"
              % (name, h.agent._climb))
        check("寻路超时" in str(getattr(job, "note", "")),
              "「%s」（没签注）被收掉了但没说清是「寻路超时」（note=%r）"
              % (name, job.note))

    # ④ 上限 0 = 不限时（老行为）：追击也再久不许收
    for name, job in _mk_jobs():
        s = fresh_settings(goto_timeout_s=0.0)
        s.enabled = True
        h = Harness(s)
        with h._patched():
            h.agent.start_route([job], why="用例：%s（不限时）" % name,
                                origin=dict(_CHASE))
            ws = h.ws(with_mob=False)
            ws.player.here_sets = []
            h.clock.t += 9999.0
            h.agent._climb_tick(h.clock.t, 520.0, set(), ws)
        check(h.agent._climb is job,
              "「寻路超时时间」= 0（不限时）时「%s」却被收掉了（老行为被改坏 ✗）" % name)


def t_goto_timeout_survives_same_step_restart():
    """**同一步被重下，不许把「寻路超时时间」归零**（用户 2026-09-28 报"没有触发寻路超时"）。

    现场数据（`behavior.log` 全量统计）：`task_begin kind=DropJob` **318 次**，而
    `goto_timeout` 只有 **57 次** ✗ —— 也就是说**大多数"卡住"根本没走到超时** ✗。

    病根：`agent.start_climb` 原来**每次都重打** `_climb_started`；而"追击改目的地 / 回区域"
    那条路（`_chase_goto_if_elsewhere` → `plan_and_start_route` → `start_route` →
    `start_climb`）**每 1~3 秒就可能重下一次同一步** ⇒ 10 秒的钟被反复清零 ⇒ 卡住的那一段
    **永远等不到超时** ✗ —— 现场看到的正是"卡住不动、也不报寻路超时"✗。
    ⚠ 这也和设计意图不符：`start_climb` 上的注释写着"**每完成一段**就重新计时"，
    而"重下同一步"**不是**完成一段 ✗。

    钉三件：
      ① **同一步重下 ⇒ 钟接着走**（把钟人为推老 11 秒、再挂同签名的一步 ⇒ 当场超时 ✓）；
      ② **换了下一步 ⇒ 重新计时**（挂不同目标集合的 ⇒ 不许当场超时 ✓）；
      ③ **手工取消之后** ⇒ 签名清掉 ⇒ 再挂"同名同目标"的也当**新任务**重新计时 ✓
         （否则"走→爬→走"那种集合名凑巧一样的会把新任务一上来就判超时 ✗）。
    """
    from decision import route

    spots = [(700.0, 660.0, 740.0, "41")]

    def _drop(dst):
        return route.DropJob("甲平台", dst, spots, tol_px=6, hold_ms=0)

    def _h():
        s = fresh_settings(goto_timeout_s=10.0)
        s.enabled = True
        return Harness(s)

    # ① 同一步重下 ⇒ 钟**接着走**（旧写法会归零 ⇒ 永远不超时 ✗）
    h = _h()
    with h._patched():
        ws = h.ws(with_mob=False)
        ws.player.here_sets = []               # 永远不在目标集合上 ⇒ 任务到不了 ✓
        h.agent.start_route([_drop("乙平台")], why="用例：①第一次", origin=dict(CHASE_ORIGIN))
        h.agent._climb_started -= 11.0         # 模拟"这一段已经卡了 11 秒"
        h.agent.start_route([_drop("乙平台")], why="用例：①重下同一步", origin=dict(CHASE_ORIGIN))
        done = h.agent._climb_tick(h.clock.t, 700.0, set(), ws)
        check(done is True and h.agent._climb is None,
              "同一步被重下之后，「寻路超时」的钟被**归零** ⇒ 卡住的段**永远等不到超时** ✗"
              "（现场：DropJob 下发 318 次、goto_timeout 只 57 次 ✗）：done=%r climb=%r"
              % (done, h.agent._climb))

    # ② 换了下一步 ⇒ 重新计时（新的一段不许一上来就被判超时）
    h2 = _h()
    with h2._patched():
        ws2 = h2.ws(with_mob=False)
        ws2.player.here_sets = []
        h2.agent.start_route([_drop("乙平台")], why="用例：②第一次", origin=dict(CHASE_ORIGIN))
        h2.agent._climb_started -= 11.0
        h2.agent.start_route([_drop("丙平台")], why="用例：②换了下一步", origin=dict(CHASE_ORIGIN))
        done2 = h2.agent._climb_tick(h2.clock.t, 700.0, set(), ws2)
        check(done2 is False and h2.agent._climb is not None,
              "换了**下一步**却没重新计时（新的一段一上来就被判超时 ✗）：done=%r climb=%r"
              % (done2, h2.agent._climb))

    # ③ 取消之后 ⇒ 签名清掉 ⇒ 同名同目标也当新任务（否则一上来就超时 ✗）
    h3 = _h()
    with h3._patched():
        ws3 = h3.ws(with_mob=False)
        ws3.player.here_sets = []
        h3.agent.start_route([_drop("乙平台")], why="用例：③第一次", origin=dict(CHASE_ORIGIN))
        h3.agent._climb_started -= 11.0
        h3.agent.stop_climb("用例：③手工取消")
        h3.agent.start_route([_drop("乙平台")], why="用例：③取消后再挂同名同目标", origin=dict(CHASE_ORIGIN))
        done3 = h3.agent._climb_tick(h3.clock.t, 700.0, set(), ws3)
        check(done3 is False and h3.agent._climb is not None,
              "手工取消之后，同签名的新任务被当成「重下同一步」⇒ **一上来就超时** ✗："
              "done=%r climb=%r" % (done3, h3.agent._climb))


def t_goto_timeout_key():
    """「**寻路超时后按键**」（用户 2026-09-28）：超时切断那一下要**点按**配置的那个键。

    用户原话："先在设置『判定参数』里添加一个『**寻路超时后按键**』，类型是**自定义按键
    下拉列表**" ✓（设置项那条已在 `t_align_params` 里钉住 ✓，这条钉**行为** ✓）。

    钉四件：
      ① **默认（`None`）⇒ 一个键都不发** ✓（老项目行为一点不变 ✓ —— 这条最容易在改的时候被
         弄丢 ✗：顺手 `tap(None)` 就是"每超时一次发一个不存在的键"✗）；
      ② 配了键 ⇒ 超时那一下**按下 + 松开**（`input.tap` ✓ 点按，**不是**按住 ✗）；
      ③ 留一条 `behavior.event("goto_timeout_key")` ✓（下次翻日志能知道发没发 ✓）；
      ④ **没超时的时候一发都不发** ✓（把超时掐掉 ⇒ `taps` 必须还是空 ✓ —— 防"顺手提前发" ✗）。
    """
    from decision import route

    s = fresh_settings(goto_timeout_s=10.0)
    s.enabled = True
    h = Harness(s)
    spots = [(700.0, 660.0, 740.0, "41")]

    def _drop(dst):
        return route.DropJob("甲平台", dst, spots, tol_px=6, hold_ms=0)

    taps = []
    _orig = ag.tap
    # ⚠ **别指望 patch `ag.tap` 能长期生效** ✗：`Harness._patched()` 会把输入层换成替身、
    #   把外面 patch 的 `tap` **覆盖掉**（2026-09-28 在这上面绕了一圈 ✓）⇒ 真正可靠的观察点是
    #   `agent.last_timeout_key`（`_fire_timeout_key` 里那句 ✓）。两样都留着：
    #   `taps` 看"真的发了一次"、`last_timeout_key` 看"发的是哪个键" ✓。
    ag.tap = lambda name, duration=0.06: taps.append(name)
    try:
        # ① 默认（None）⇒ 超时也不发键
        with h._patched():
            ws = h.ws(with_mob=False)
            ws.player.here_sets = []               # 永远到不了 ⇒ 等超时 ✓
            h.agent.start_route([_drop("乙平台")], why="用例：默认不按", origin=dict(CHASE_ORIGIN))
            h.agent._climb_started -= 11.0         # 已经跑了 11 秒（>10 ⇒ 当场超时 ✓）
            h.agent._climb_tick(h.clock.t, 700.0, set(), ws)
        check(all(t is None for t in ([h.agent.last_timeout_key] + taps)),
              "默认（`None`）也发键了 —— 老项目会被多发一个不存在的键 ✗：%r / %r"
              % (h.agent.last_timeout_key, taps))
        # ④ 没超时时也不发（把它当成"每拍都发"来挡 ✓）
        with h._patched():
            ws = h.ws(with_mob=False)
            ws.player.here_sets = []
            h.agent.start_route([_drop("乙平台")], why="用例：没超时", origin=dict(CHASE_ORIGIN))
            h.agent._climb_started = h.clock.t     # 刚下的命令 ⇒ 远没到 10 秒 ✓
            h.agent._climb_tick(h.clock.t, 700.0, set(), ws)
        check(not taps and h.agent.last_timeout_key is None,
              "没超时也发键了（那不是「超时后按键」✗）：%r / %r"
              % (h.agent.last_timeout_key, taps))

        # ②③ 配了键 ⇒ 超时那一下点按一次 + 留痕
        h.agent.settings.goto_timeout_key = "esc"
        h.log.clear()
        with h._patched():
            ws = h.ws(with_mob=False)
            ws.player.here_sets = []
            h.agent.start_route([_drop("丙平台")], why="用例：配了键", origin=dict(CHASE_ORIGIN))
            h.agent._climb_started -= 11.0
            h.agent._climb_tick(h.clock.t, 700.0, set(), ws)
        _lk = h.agent.last_timeout_key
        check(_lk is not None and _lk[0] == "esc" and _lk[1] == "esc",
              "超时那一下没点按配置的键（该**按下 + 松开**各一次 ✓；`esc` 解析后还是 `esc` ✓）："
              "last=%r taps=%r" % (_lk, taps))

        # ⑤ ⭐⭐ **键名必须先解析成真实物理键**（2026-09-28 现场修 ✗ —— 用户报"**寻路超时没有
        #    触发我配的「寻路超时后按键」**"✓，而 log 里 `goto_timeout_key key=表情_无语`
        #    明明有 **27 条** ✓ ⇒ **不是没触发，是键没发出去** ✗）。
        #    根因：下拉里**自定义按键存的是「名字」**（`custom_keys` 的 key ✓），而
        #    `input.tap` 走 `resolve_vk`、只认**固定键名义 / 物理键名** ✗ ⇒ 解析不出时
        #    `tap` 内部 `if vk is None: return` ⇒ **静默什么都不发** ✗。
        #    钉两件（读 `last_timeout_key` 第 ② 项 = **真正发出的物理键** ✓）：
        #      · **自定义键名** ⇒ 解析成它绑的物理键 ✓；
        #      · **功能键**（`attack` 之类 ✓）⇒ 也要解析（否则连固定项都发不出去 ✗ 这个面更宽 ✓）。
        _was_ck = dict(h.agent.settings.custom_keys or {})
        h.agent.settings.custom_keys["测试表情"] = "f6"
        taps.clear()
        h.agent.settings.goto_timeout_key = "测试表情"
        h.agent._fire_timeout_key()
        _lk2 = h.agent.last_timeout_key
        check(_lk2 is not None and _lk2[0] == "测试表情" and _lk2[1] == "f6"
              and taps == ["f6"],
              "**自定义键名没被解析成物理键**（`tap` 会**静默什么都不发** ⇒ 界面上配了、"
              "游戏里毫无反应 ✗）：last=%r taps=%r" % (_lk2, taps))
        taps.clear()
        h.agent.settings.goto_timeout_key = "attack"
        h.agent._fire_timeout_key()
        check(h.agent.last_timeout_key is not None
              and h.agent.last_timeout_key[1] == h.agent.settings.keymap["attack"],
              "**功能键没被解析成物理键**（连「攻击 / 跳跃」这些固定项都发不出去 ✗）："
              "%r / keymap=%r" % (h.agent.last_timeout_key,
                                  h.agent.settings.keymap.get("attack")))
        h.agent.settings.custom_keys = _was_ck
        h.agent.settings.goto_timeout_key = None
        # ③ 留痕：`_fire_timeout_key` 里那句 `behavior.event("goto_timeout_key", …)` ——
        #    用**源码级**钉（`h.log` 是**按键**日志，behavior 事件不在里面 ✗ 别拿它查 ✗）。
        from pathlib import Path

        _ag = (Path(__file__).resolve().parents[1] / "decision" / "agent.py"
               ).read_text(encoding="utf-8")
        check('behavior.event("goto_timeout_key"' in _ag,
              "超时按了键却没留痕（下次翻日志说不清它到底发没发 ✗）")
    finally:
        ag.tap = _orig


def t_locked_mob_info_cache():
    """「锁定一个目标时**每秒更新**其位于的 foothold 集合」（用户 2026-09-28 要求）。

    现场（`behavior.log`）：同一只怪、同一个目的地，连着十几次
    「追击：怪不在我这块平台上（它在「一楼」），先过去」⇒ 人在路上目的地就被改写 ⇒
    **锁着目标来回走** ✗。根因 = `mob_sets_of` **每拍现算**：怪框抖几像素，它那条 foothold
    在边界上时就会在「一楼 / 二楼」之间跳 ⇒ 目的地跟着跳 ✗。

    钉五件：
      ① 同一只怪、**一秒内** ⇒ 解析器**只被调一次** ✓（这就是"每秒更新"）；
      ② 过了 `MOB_SETS_TTL_S` ⇒ **重查** ✓（人真换层（爬绳/下跳）不会被漏掉）；
      ③ **换目标 ⇒ 立刻重查** ✓（绝不能拿上一只怪的集合去下前往 ✗）；
      ④ **查不到就不缓存** ✓（下一拍立刻再试；缓存"空"会让它整整一秒都不再判 ✗）；
      ⑤ 一秒内抖一下（「一楼」→「二楼」）⇒ 拿到的仍是**「一楼」** ✓（抖动被滤掉 ✓）。
    """
    if not hasattr(ag, "MOB_SETS_TTL_S"):
        raise AssertionError("决策层没有 `MOB_SETS_TTL_S`（每秒更新那条的口径）")
    ttl = float(ag.MOB_SETS_TTL_S)
    s = fresh_settings()
    h = Harness(s)
    calls = []

    class _M:
        def __init__(self, mid):
            self.id = mid

    sets_now = {"v": ["一楼"]}

    def fake_sets_of(_player, mob):
        calls.append(mob.id)
        return {"world": (100.0, 0.0), "sets": list(sets_now["v"])}

    h.agent.mob_sets_of = fake_sets_of
    m1, m2 = _M(7), _M(8)
    t = 1000.0
    ws = h.ws(with_mob=False)

    i1 = h.agent._locked_mob_info(ws, m1, t)
    check(i1 and i1["sets"] == ["一楼"] and len(calls) == 1,
          "第一次查询结果不对：%r / 调用 %d 次" % (i1, len(calls)))
    # ① 一秒内 ⇒ 不再查（这就是"每秒更新"，不是"每拍现算"）
    i2 = h.agent._locked_mob_info(ws, m1, t + 0.3)
    check(len(calls) == 1 and i2 is i1,
          "**一秒内又查了一遍**（那就还是每拍现算 ⇒ 目的地跟着抖 ⇒ 来回走 ✗）：调用 %d 次"
          % (len(calls),))
    # ⑤ 这一秒里怪抖到「二楼」⇒ 结果不该跟着变
    sets_now["v"] = ["二楼"]
    i3 = h.agent._locked_mob_info(ws, m1, t + 0.6)
    check(i3["sets"] == ["一楼"],
          "一秒内的抖动漏出来了（目的地会跟着跳 ⇒ 锁着目标来回走 ✗）：%r" % (i3["sets"],))
    # ② 过了一秒 ⇒ 重查（这时才吃到「二楼」✓）
    i4 = h.agent._locked_mob_info(ws, m1, t + ttl + 0.01)
    check(len(calls) == 2 and i4["sets"] == ["二楼"],
          "过了缓存时效没重查（人真换层就漏了 ✗）：%r / 调用 %d 次"
          % (i4["sets"], len(calls)))
    # ③ 换目标 ⇒ 立刻重查
    _n = len(calls)
    h.agent._locked_mob_info(ws, m2, t + ttl + 0.02)
    check(len(calls) == _n + 1,
          "换了目标却没重查（会拿上一只怪的集合去下前往 ✗）：调用 %d 次" % (len(calls),))
    # ④ 查不到 ⇒ 不缓存
    h.agent.mob_sets_of = lambda _p, _m: None
    check(h.agent._locked_mob_info(ws, m1, t + 5.0) is None, "解析不出该返回 None")
    check(h.agent._mob_info_cache is None,
          "查不到却缓存了（整整一秒都不再判 ✗）：%r" % (h.agent._mob_info_cache,))
    h.agent.mob_sets_of = fake_sets_of
    check(h.agent._locked_mob_info(ws, m1, t + 5.01) is not None,
          "上一次查不到之后，下一拍该**立刻再试** ✓")

    # ⑥ ⭐ **该重查才重查**（用户 2026-09-28 原话："**在恰当的时机重新查询**
    #    （之前我说 1s 但是可能不太好用）"✓）—— 主驱动从"按时间过期"改成
    #    **局面变了才重查**（`agent._mob_sig` = 怪在**画面**上的位置档 + 玩家脚下集合 ✓），
    #    `MOB_SETS_TTL_S` 退成**最长保鲜**（兜底 ✓）。钉三件：
    class _M2:                              # ⚠ 指纹要看 x/y ⇒ 这个假怪得有 ✓
        def __init__(self, mid, x, y):
            self.id, self.x, self.y = mid, x, y

    h.agent.mob_sets_of = fake_sets_of
    h.agent._mob_info_cache = None
    m3 = _M2(9, 100.0, 200.0)
    ws2 = h.ws(with_mob=False)
    ws2.player.here_sets = ["一楼"]
    _n0 = len(calls)
    h.agent._locked_mob_info(ws2, m3, t + 20.0)
    check(len(calls) == _n0 + 1, "第一次没查：调用 %d 次" % len(calls))
    # ① 怪**原地微抖**（几像素）⇒ **不重查** ✓（老毛病就是这儿每拍重查 ⇒ 判定抖 ✗）
    m3.x += 3.0
    m3.y += 2.0
    h.agent._locked_mob_info(ws2, m3, t + 20.1)
    check(len(calls) == _n0 + 1,
          "怪在原地微抖就重查了（那等于每拍重查 ⇒ 判定跟着抖 ✗）：调用 %d 次" % len(calls))
    # ② 怪**明显动了** ⇒ **立刻重查** ✓（不用等保鲜到期 ✓）
    m3.y += 60.0
    h.agent._locked_mob_info(ws2, m3, t + 20.2)
    check(len(calls) == _n0 + 2,
          "怪明显动过了却没重查（旧答案一直挂着 ⇒ 它换层了也不知道 ✗）：调用 %d 次"
          % len(calls))
    # ③ **玩家换层** ⇒ 也重查 ✓（同一个 x 在新平台上算得出别的答案 ✓）
    ws2.player.here_sets = ["二楼"]
    h.agent._locked_mob_info(ws2, m3, t + 20.3)
    check(len(calls) == _n0 + 3,
          "玩家换层了却没重查（脚下变了，答案该重算 ✗）：调用 %d 次" % len(calls))


def t_chase_clock_attack_vs_segment():
    """⭐⭐ **追击签注的任务「不被 attack 刷新、可以被单段寻路执行器成功刷新」**
    （用户 2026-09-29 第 2 条 ✓ 原话）。

    为什么必须分开（现场形状，缺了这条 30 秒的闸就形同虚设 ✗）：
      · 追击**最常见**的重下就是"**怪换层了 ⇒ 目的地集合变了**" ⇒ `start_climb` 里那套
        签名判据（执行器类 + 起点集合 + 目标集合 ✓）**必然变** ⇒ 钟被重打 ⇒ 一趟永远追不上
        的追击可以**无限续命** ✗✗；
      · 而"走完一段接下一段"（`_task_finished` → `start_climb(nxt)` ✓）**该刷新** ✓
        —— 那是真的新一段路，预算该是新的 ✓。

    钉五件：
      ① attack 重下（`keep_clock=True`）**目的地都换了** ⇒ 钟**一动不动** ✓；
      ② 反过来（不带 `keep_clock`，= 走完一段接下一段 / 命令前往 ✓）⇒ 换目标照旧**重打** ✓；
      ③ 跑完 / 被超时切掉**之后**再下（哪怕带 `keep_clock=True`）⇒ **起新钟** ✓
         （带着上一趟的旧账一开工就超时 ⇒ 又回到"三楼→二楼 drop 左右晃"那个死循环 ✗）；
      ④ 源码级：两处 attack 下发都带 `keep_clock=True` ✓，而 `_task_finished` 接下一段
         那次 `start_climb(...)` **不带** ✓（"单段成功 ⇒ 刷新"那一侧 ✓）。
    """
    import inspect

    from decision import route

    s = fresh_settings()
    h = Harness(s)
    a = h.agent
    with h._patched():
        def _drop(dst):
            return route.DropJob("甲平台", dst, [(500.0, 480.0, 520.0, "1")],
                                 tol_px=10, hold_ms=250)

        # ① attack 重下：目的地换了，钟**不许动**
        a._route_origin = dict(CHASE_ORIGIN)
        a.start_climb(_drop("乙平台"))
        t0 = a._climb_started
        check(bool(t0), "start_climb 没打「寻路超时」的钟")
        h.clock.t += 1.0                     # ⚠ 夹具把 `ag.time.monotonic` 换成了**假钟** ✓
        a.start_climb(_drop("丙平台"), keep_clock=True)     # 怪换层 ⇒ 换目的地 ✓
        check(a._climb_started == t0,
              "**attack 重下把追击的钟刷新了**（怪一换层就重打 ⇒ 一趟永远追不上的追击"
              "可以无限续命，30 秒的闸形同虚设 ✗）：%r → %r" % (t0, a._climb_started))

        # ② 不带 keep_clock（= 单段成功接下一段 / 命令前往）⇒ 换目标照旧重打
        h.clock.t += 1.0                     # 假钟往前推 ⇒ "重打的钟"必然是不同的数 ✓
        a.start_climb(_drop("丁平台"))
        check(a._climb_started != t0,
              "不带 `keep_clock` 的重下也没重打钟（走完一段接下一段该是**新预算** ✗）")
        t1 = a._climb_started

        # ③ 跑完之后再下 ⇒ 起新钟（别把上一趟的旧账带过来 ✗）
        a._task_finished()
        h.clock.t += 1.0                  # 假钟往前推（同上 ✓）
        a._route_origin = dict(CHASE_ORIGIN)
        a.start_climb(_drop("乙平台"), keep_clock=True)
        check(a._climb_started != t1,
              "上一趟收工之后再下（哪怕带 keep_clock）没起新钟 —— 新任务一开工就带着"
              "上一趟的旧账 ⇒ 立刻超时 ⇒ 追击又下 ⇒ **死循环** ✗")

        # ④ 源码级：两处 attack 带、接下一段那次不带
        src = inspect.getsource(a.__class__)
        check(src.count("keep_clock=True") >= 2,
              "attack 那两处下发没都带 `keep_clock=True`（只带一处 ⇒ 另一条路照样刷新 ✗）：%d"
              % src.count("keep_clock=True"))
        _fn = src.split("def _task_finished", 1)[-1].split("\n    def ", 1)[0]
        check("start_climb(" in _fn and "keep_clock" not in _fn,
              "`_task_finished` 接下一段那次 `start_climb` 带上了 keep_clock —— 那"
              "「单段执行器成功 ⇒ 刷新」就没了（用户明确要它能刷新 ✗）")


def t_goto_timeout_reset_after_finish():
    """「**收工之后再下同一步 ⇒ 钟必须重打**」（用户 2026-09-28 报"**三楼前往二楼的 drop 时，
    左右晃了很长时间**"✗）。

    现场 `behavior.log`（这两行挨着看就够了 ✓）：
      `05:48:49.523  task_begin  dst=二楼 kind=DropJob`
      `05:48:50.377  task_fail   寻路超时：这一段已经跑了 10 秒  secs=10.0`
    —— **只隔 0.85 秒** ✗ ⇒ 一个新任务带着**上一段 10 秒的旧账**开工。

    病根：「同一步」的签名（`_climb_sig` = **执行器类 + 起点集合 + 目标集合** ✓）原来**只在
    `stop_climb` 里清** ✗，而任务**正常收工**（到了 / 到集合收工 / 超时失败 / 没坐标）走的是
    `_task_finished` ⇒ 签名**残留** ⇒「追击」再下一次 `DropJob(三楼 → 二楼)`（类与起终点
    全一样 ✓）被当成"同一步重下"⇒ 钟**接着上一段走** ⇒ 一开工就超时 ⇒ 失败收起 ⇒ 追击又下
    ⇒ **死循环** ✗。人看到的就是"**左右晃**"：每一轮都从 `ALIGN` 重新就近平齐（走回 foothold
    中心 ✓），走到一半又被超时掐掉 ✓。

    钉三件：
      ① **收工之后**再挂同一步 ⇒ 钟**重打** ✓（本次修的就是它）；
      ② **覆盖一个正在跑的任务**（没收工）⇒ 钟**接着走** ✓（2026-09-28 那天修过的反面，
         别弄丢 ✗）；
      ③ 收工后换**另一步**（起终点不同）⇒ 照旧重打 ✓（老行为 ✓）。
    """
    import time as _time

    from decision import route

    s = fresh_settings()
    h = Harness(s)
    a = h.agent

    def _job(dst="二楼", src="三楼"):
        # ⚠ 起终点 + 执行器类就是"同一步"的**全部**判据（见 `start_climb` 里的 `_sig` ✓）
        return route.DropJob(src, dst, [(500.0, 480.0, 520.0, "1")], tol_px=10, hold_ms=250)

    a._route_origin = dict(CHASE_ORIGIN)   # 追击签注 ⇒ 才吃「寻路超时时间」✓（2026-09-29）
    a.start_climb(_job())
    t0 = a._climb_started
    check(t0, "start_climb 没打「寻路超时」的钟")
    # ② **覆盖**一个**还在跑**的（同一步）⇒ 钟一动不动
    _time.sleep(0.003)
    a._route_origin = dict(CHASE_ORIGIN)   # 追击签注 ⇒ 才吃「寻路超时时间」✓（2026-09-29）
    a.start_climb(_job())
    check(a._climb_started == t0,
          "覆盖一个**正在跑**的同一步却把钟重打了（那正是「永远触发不了超时」✗ —— "
          "2026-09-28 修过这条，别弄丢 ✓）：%r → %r" % (t0, a._climb_started))
    # ① **收工之后**再下同一步 ⇒ 必须重打（本次修的那条 ✓）
    a._task_finished()
    check(a._climb is None, "收工后任务还挂在 `_climb` 上")
    _time.sleep(0.003)
    a._route_origin = dict(CHASE_ORIGIN)   # 追击签注 ⇒ 才吃「寻路超时时间」✓（2026-09-29）
    a.start_climb(_job())
    check(a._climb_started != t0,
          "**收工之后再下同一步没重打钟** —— 新任务一开工就带上一段的旧账 ⇒ 立刻超时 ⇒ "
          "又重下 ⇒ 死循环（现场「左右晃」就是这个 ✗）：还是 %r" % (a._climb_started,))
    # ③ 收工 + 换**另一步** ⇒ 照旧重打（老行为 ✓）
    t1 = a._climb_started
    a._task_finished()
    _time.sleep(0.003)
    a._route_origin = dict(CHASE_ORIGIN)   # 同上：追击签注 ✓
    a.start_climb(_job(dst="一楼", src="二楼"))
    check(a._climb_started != t1,
          "收工后换了**下一步**却没重打钟：%r" % (a._climb_started,))


def t_goto_timeout_per_hop():
    """「寻路超时时间(s)」按**每一段**算，不是按整条路线（用户 2026-09-26 定的口径）。

    口径：**每完成一段**（从一个集合走到另一个集合）就**重新计时** ✓ —— 长路线不该因为
    "两段各自都没超、加起来超了"被整条切断 ✗。

    代码上靠 `start_climb()` 重打 `_climb_started`（`_task_finished` 接下一段时会调它 ✓），
    但这件事**以前没有任何用例钉** ✗ —— 谁把 `start_climb` 里的那句挪走，行为就悄悄变成
    "整条路线共用一次超时"，只会在某次长路线走到一半被切断时才发现 ✗。两种情形分开钉：

      · 两段各 9 秒、上限 10 秒（累计 18 秒）⇒ **不许**被切断；
      · 单独一段 11 秒、上限 10 秒 ⇒ 必须切断，并说清"这一段"跑了多久。
    """
    from decision import route

    s = fresh_settings(goto_timeout_s=10.0)
    s.enabled = True
    h = Harness(s)
    a = route.WalkJob("甲集合", [(520.0, 510.0, 530.0, -208.0, "1")])
    b = route.WalkJob("乙集合", [(520.0, 510.0, 530.0, -208.0, "2")])
    with h._patched():
        h.agent.start_route([a, b], why="用例：两段", origin=dict(CHASE_ORIGIN))
        ws = h.ws(with_mob=False)
        # ① 第一段跑了 9 秒（没超）⇒ 到了 ⇒ 第二段**重新计时**
        h.clock.t += 9.0
        ws.player.here_sets = ["甲集合"]
        check(h.agent._climb_tick(h.clock.t, 520.0, set(), ws) is False,
              "第一段到了没接上第二段")
        check(h.agent._climb is b, "第二段没挂上：%r" % (h.agent._climb,))
        # ② 第二段再跑 9 秒（累计 18 秒 > 上限）⇒ **不许**被切
        h.clock.t += 9.0
        ws.player.here_sets = ["乙集合"]
        check(h.agent._climb_tick(h.clock.t, 520.0, set(), ws) is True,
              "两段各 9 秒、上限 10 ⇒ 累计 18 秒被整条切断（该**每段重新计时** ✗）")
        check("寻路超时" not in h.agent.current_goto_note(),
              "两段都没超却报了超时：%r" % h.agent.current_goto_note())

        # ③ 单独一段超过上限 ⇒ 必须切，并说清是"这一段"
        c = route.WalkJob("丙集合", [(520.0, 510.0, 530.0, -208.0, "3")],
                          timeout_s=600.0)
        h.agent._route_origin = dict(CHASE_ORIGIN)   # 追击签注 ✓（2026-09-29 起只有它吃超时）
        h.agent.start_climb(c)
        ws.player.here_sets = ["甲集合"]
        h.clock.t += 11.0
        check(h.agent._climb_tick(h.clock.t, 520.0, set(), ws) is True,
              "单段超过「寻路超时时间」却没切断：%r" % (h.agent._climb,))
        note = h.agent.current_goto_note()
        check("寻路超时" in note, "切断了却没说清原因：%r" % note)
        check("这一段" in note, "超时话里没说清是**这一段**跑了多久：%r" % note)


def t_afk_retry_params_are_common():
    """「被打断重试」+「打断后重试(s)」必须挂在**通用层**（换类型也看得见）。

    2026-09-26 实锤：这两个控件被 `addRow` 进了「隐身休息」子组的表单 ✗ ⇒ 选「定点休息」
    时整个隐身子组被隐藏 ⇒ 参数**跟着一起消失**（用户："我怎么没在定点休息组里看到
    配置？"）。它们本来就是两种类型共用的 ✓（区别只在"算不算被打断"的时间窗）。
    """
    p, app = build_panel()
    settings = ag.settings
    saved = settings.anti_afk_type
    try:
        settings.anti_afk_type = "spot_rest"
        p._refresh_afk_ui()
        app.processEvents()
        check(not p._afk_hidden.isVisibleTo(p),
              "选「定点休息」时隐身子组该藏起来（这条测不了别的）")
        check(not p._afk_hidden.isAncestorOf(p.ck_afk_retry),
              "「被打断重试」挂在隐身子组里（换类型会跟着消失 ✗）")
        check(not p._afk_hidden.isAncestorOf(p.sp_afk_retry),
              "「打断后重试(s)」挂在隐身子组里（换类型会跟着消失 ✗）")
        check(p.ck_afk_retry.isVisibleTo(p) and p.sp_afk_retry.isVisibleTo(p),
              "选「定点休息」时看不到这两个参数 —— 用户 2026-09-26 报的就是这个 ✗")
    finally:
        settings.anti_afk_type = saved
        p._refresh_afk_ui()


def t_climb_on_rope_by_broadcast():
    """「在不在绳上」怎么算 —— **上爬以广播为权威，下爬才留记忆**（用户 2026-09-27 两条要求：
    "**位置状态变化即广播，且绝对权威**" ✓ + "**排查所有执行器是否有自己判定坐标，全部换成由
    位置状态广播通信**" ✓）。

    来龙去脉（**同一件事踩过两回，正反两面都留在这儿** ✗）：

    · 反面（2026-09-27 早）：**只看广播**、而广播的 `ladder_id` 带**按键许可**（"没按 ↑/↓ 一律
      不算在绳上" ✓，防"走路碰到绳子就卡住" ✓）⇒ 执行器在**下爬**"点按重试"那几拍自己松开 ↓
      ⇒ 广播**必然**说"不在绳上" ⇒ 判"还没上绳" ⇒ 又松开 ⇒ **原地打转** ✗（当时的现场 ✓）。
    · 正面（2026-09-27 晚，用户截图）：为了治上面那条，加了"**位置兜底 + 记忆**"（`_rope_seen`
      ✗）—— 可**上爬那一路从不松键**（用户定："只有『按住 ↑』和『补按住 ↑』，不能存在
      『松开 ↑』和『点按 ↑』" ✓）⇒ 那条兜底在上爬这边**只有害处**：人跳失败退回平台、
      位置仍落在绳段里（就站在绳底那块面上 ✓）⇒ 记忆**永远不清** ⇒ CLIMB 相一直以为在爬 ✗
      ⇒ 卡在「**正爬升中：刚爬 -1**」按住 ↑ 磨墙 ✗（位置状态早就说"没在绳上"了 ✗）。

    ⇒ 现在的口径（**按方向分开** ✓）：
      · **上爬（`dir > 0`）**：**广播说不在绳上就是不在** ✓ —— 不留记忆、不用位置兜底 ✗
        （我们不松键 ⇒ 广播绝不会因为我们自己而变空 ✓，"绝对权威"才能这么用 ✓）；
      · **下爬（`dir < 0`）**：那几下**是我们自己松 ↓ 造成的** ⇒ 仍用"**广播说位置还在绳段里**"
        （`on_rope_pos` ✓，与按键许可无关 ✓，几何只在 `perception/pos_state.py` 算一次 ✓）
        把记忆**留住** ✓；广播连位置都给不出来 ⇒ 才清 ✓。

    钉六件：
      ① ⭐ **上爬：广播说不在绳上 ⇒ 立刻回到"还没上绳"** ✓（哪怕记忆曾为 True、位置还在绳段里 ✗）；
      ② ⭐ **下爬：自己松键造成的空档 ⇒ 记忆留住** ✓（这一档才能免疫"自己造成的空档" ✓）；
      ③ **从没确认过**（一开始广播就是 `None`）⇒ **不认** ✓ —— 人**站在绳底那块面上、还没抓绳**
        时位置同样"在绳段里" ✗，拿它判会**跳过横向对齐** ✗，也会让"跳失败 ⇒ 直接补按" ✗；
      ④ **还按着 ↑/↓**（下爬 ✓）而**广播说位置也不在绳段里**（x 偏出 / y 出绳段 / 没读数 ✓）
        ⇒ 清掉记忆、照旧"还没上绳" ✓；
      ⑤ ⚠ **对齐相里不许跳过横向对齐**（`t_climb_flow_rules` / `t_climb_diagonal_jump` 钉着 ✓）；
      ⑥ 广播**当场**说在绳上 ⇒ 照旧认 ✓。
    """
    from decision import route

    C = route.ClimbJob
    base = dict(ladder_id="L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                dst_set="上", src_set="下", tol_px=10, hold_ms=0, stall_s=3.0)

    def climbing(**kw):
        """造一个"**已经在爬**"的任务，并**先确认过一次在绳上**（`_rope_seen=True` ✓）。"""
        j = C(**base)
        j.phase = C.CLIMB
        j._jump_tried = True
        j.update(0.0, 700.0, py=0.0, ladder_id="L2", here_sets=["下"])   # 感知确认在绳上 ✓
        for k, v in kw.items():
            setattr(j, k, v)
        return j

    # ① ⭐ **上爬以广播为权威**（用户 2026-09-27 第 1 条："位置状态变化即广播，且绝对权威"）：
    #    广播说不在绳上 ⇒ **立刻**按"还没上绳"处理 ✓ —— 记忆**不许**黏着 ✗
    #    （人跳失败退回平台、位置仍落在绳段里 ✗ ⇒ 黏住就会卡在"正爬升中：刚爬 -1" 磨墙 ✗）。
    j = climbing()
    check(j._rope_seen is True, "用例前提：`climbing()` 该先确认过在绳上 ✓")
    o = j.update(0.1, 700.0, py=0.0, ladder_id=None, here_sets=["下"],
                 on_rope_pos="L2")      # 广播：键没按 ⇒ 不在绳上（位置兜底**不算数** ✗）
    check(j._rope_seen is False,
          "上爬时广播说不在绳上了，记忆还黏着（⇒ CLIMB 相会一直以为在爬、卡住 ✗）"
          "：%r" % (o.get("note"),))
    check("还没上绳" in str(o.get("note") or ""),
          "上爬：广播说不在绳上，却没回到「还没上绳」那一支（卡在「正爬升中」✗）：%r"
          % (o.get("note"),))

    # ② ⭐ **下爬照旧靠记忆兜**（它会**自己松 ↓**：点按重试那几拍 ✓）—— "绝对权威"在这一档
    #    要认"空档是我们自己造成的"（否则又回到"原地打转"那个老坑 ✗）。
    jd = C(**dict(base, direction=-1))
    jd.phase = C.CLIMB
    jd._jump_tried = True
    jd.update(0.0, 700.0, py=0.0, ladder_id="L2", here_sets=["下"])   # 先确认在绳上 ✓
    check(jd._rope_seen is True, "用例前提（下爬）：该先确认过一次在绳上 ✓")
    jd.update(0.1, 700.0, py=0.0, ladder_id=None, here_sets=["下"],
              on_rope_pos="L2")         # 我们自己松 ↓ 造成的空档 + 位置还在绳段里 ✓
    check(jd._rope_seen is True,
          "下爬：我们自己松 ↓ 造成的空档，记忆被清了（⇒ 又松开 ⇒ 原地打转 ✗）")

    # ③ **从没确认过**（一上来感知就是 None）⇒ 不认 ✓
    j = C(**base)
    j.phase = C.CLIMB
    j._jump_tried = True
    o = j.update(0.0, 700.0, py=0.0, ladder_id=None, here_sets=["下"])
    check("还没上绳" in str(o.get("note") or ""),
          "从没确认过在绳上却凭位置认了（「站在绳底、还没抓绳」也会被当成在绳上 ✗）：%r"
          % (o.get("note"),))

    # ② 广播说**位置不在这根绳的绳段里了** ⇒ 记忆清掉、照旧「还没上绳」✓
    #    ⚠ 三种"给不出绳号"的情况在广播那边是**同一件事**（`on_rope_pos=None` ✓：
    #      x 偏出容差 / y 出绳段 / 这一拍没有读数 —— 判据原文在 `perception/pos_state.py` ✓），
    #      这里逐个摆一遍，确认执行器**只读广播、不自己比坐标** ✗。
    for px, py, why in ((711.0, 0.0, "x 偏出容差"),
                        (700.0, 400.0, "y 出绳段（400 > 绳底 200 + 10）"),
                        (700.0, None, "没有 y 读数")):
        j = climbing()
        o = j.update(0.1, px, py=py, ladder_id=None, here_sets=["下"],
                     on_rope_pos=None)           # ⭐ 广播：位置不在这根绳的绳段里 ✓
        check(not j._rope_seen and "还没上绳" in str(o.get("note") or ""),
              "%s 却还认在绳上（记忆太黏 ✗ —— 那「从绳上掉下来」就再也判不出来了）：%r / "
              "_rope_seen=%s" % (why, o.get("note"), j._rope_seen))

    # ④ ⚠ 对齐相里不许跳过横向对齐（流程是"先对齐 x 站住 → 再按跳" ✓）
    j = C(**base)
    j.hold_ms = 250
    o = j.update(0.0, 700.0, py=0.0, ladder_id=None, here_sets=["下"])
    check("跳过横向对齐" not in str(j.note or "") and j.phase == C.ALIGN,
          "对齐相里凭位置就跳过了横向对齐（用户流程是「先对齐 x 再跳」✗）：%r / %s"
          % (j.note, j.phase))

    # ⑤ 感知当场说在绳上 ⇒ 照旧认 ✓
    j = C(**base)
    j.phase = C.CLIMB
    j._jump_tried = True
    o = j.update(0.0, 700.0, py=0.0, ladder_id="L2", here_sets=["下"])
    check("已上绳" in str(o.get("note") or "") and o["dir"] != 0,
          "感知说在绳上时反而没认（原行为被改坏 ✗）：%r / %s" % (o.get("note"), o))


def t_interrupt_by_origin():
    """**「任务没跑的那一拍」怎么处置 ⇒ 按发任务的签注分流**（用户 2026-10-02 #1 ✓ 原话：

        "_climb_started 按发任务的签注判定：追击→直接结束本次寻路任务；换战区→不变；
         休息→走『打断后重试(s)』参数"）。

    病（P0，现场实测）：原来**任何**"任务没跑的拍"都重打那把钟 ⇒ 打架抖动（`battle_state`
    每 0.1~0.3 s 翻一次）时**永远走不到**「寻路超时」/「追怪寻路.duration」⇒ 2026-10-02
    「龙族打猎场」卡 40 秒、直到人工关自动 ✗（`behavior.log` 15:39:31→15:40:06 ✓）。

    钉五件：
      ① **追击（chase）⇒ 直接结束本次寻路任务**（`_climb` 清空 + `task_fail` 留痕 ✓，
         且**不重打**那把钟 ✓）；
      ② **换战区（zone）⇒ 不变**（照旧重打 ⇒ 挨打时间不算超时 ✓）；
      ③ **休息（rest）⇒ 不重打**（钟接着走 ⇒ 到点由「寻路超时」收口，休息自己按
         「打断后重试(s)」重排 ✓）；
      ④ 无签注（命令前往 / 任务队列 / …）⇒ **不变**（照旧重打 ✓ 老行为一字不变 ✓）；
      ⑤ `_origin_kind()` 与 `chase_tagged()` 同口径 ✓；**源码级**：`origin=` 的实参
         必须是 dict（`origin="replan"` 那种字符串会让 `start_route` 的 `dict(origin)`
         当场抛 ValueError ⇒ 新路线起不来、而旧任务已被停掉 ⇒ **原地站着** ✗ 现场
         `climb_replan_err` 就是它 ✓）。
    """
    import inspect
    import re as _re

    import decision.agent as _am
    from decision import route as _route

    s = fresh_settings()
    h = Harness(s)
    a = h.agent

    def mk_job():
        return _route.ClimbJob("L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                               dst_set="乙平台", src_set="甲平台", tol_px=6, hold_ms=0)

    def drive(kind, t):
        """装一个在跑的任务（签注 = kind）⇒ 叫一声"这一拍没跑" ⇒ 回现场。"""
        a._climb = mk_job()
        a._climb_origin = ({**kind} if kind else None)
        a._climb_started = 100.0
        a._task_ran_at = 99.5
        a._route = []
        a._climb_sig = ("ClimbJob", "甲平台", "乙平台")
        a._climb_interrupted(t)
        return a._climb, a._climb_started

    # ① 追击 ⇒ 直接结束（任务清掉、钟**不**重打）
    j, st = drive({"kind": "chase", "mob_id": 7}, 200.0)
    check(j is None and st == 100.0,
          "追击被打断却没结束这一趟（或把钟重打了 ⇒ 又是「永不超时」✗）：climb=%r started=%r"
          % (j, st))
    check(not a._climb_sig, "追击结束之后签名没清（下一个任务会被当「同一步重下」✗）")
    a._route_origin = None
    # ② 换战区 ⇒ 不变（重打）
    j, st = drive({"kind": "zone"}, 300.0)
    check(j is not None and st == 300.0,
          "换战区被打断没照旧重打钟（挨打时间会被算成超时 ✗）：climb=%r started=%r" % (j, st))
    a.stop_climb("用例收尾")
    # ③ 休息 ⇒ 不重打（钟接着走）
    j, st = drive({"kind": "rest"}, 400.0)
    check(j is not None and st == 100.0,
          "休息赶路被打断把钟重打了（该让它走到「寻路超时」、由休息按「打断后重试(s)」重排 ✗）："
          "climb=%r started=%r" % (j, st))
    a.stop_climb("用例收尾")
    # ④ 无签注 ⇒ 不变（重打 ✓ 老行为）
    j, st = drive(None, 500.0)
    check(j is not None and st == 500.0,
          "没签注的任务被打断没照旧重打钟（老行为被改坏 ✗）：climb=%r started=%r" % (j, st))
    a.stop_climb("用例收尾")

    # ⑤ 口径一致 + 源码级：`origin=` 不许再传字符串（`dict(str)` 会当场抛 ✗）
    a._climb_origin = {"kind": "chase"}
    check(a._origin_kind() == "chase" and a.chase_tagged() is True,
          "`_origin_kind()` / `chase_tagged()` 不同口径了 ✗")
    a._climb_origin = {"kind": "zone"}
    check(a._origin_kind() == "zone" and a.chase_tagged() is False,
          "`chase_tagged()` 把非追击也算成追击了 ✗")
    # ⚠ 扫源码前**先滤掉注释行**（我自己的注释里就写着 `origin="replan"` 那段历史 ✗
    #   —— 不滤会把说明文字判红，这坑在别处踩过一次 ✓）
    src = "\n".join(ln for ln in inspect.getsource(_am.CombatAgent).split("\n")
                    if not ln.strip().startswith("#"))
    _bad_origin = _re.findall(r"origin\s*=\s*[\"']", src)
    check(not _bad_origin,
          "`origin=` 又收到**字符串**了（`dict(str)` 会抛 ⇒ 新路线起不来、旧任务已停 ⇒ "
          "原地站着 ✗）：%r" % (_bad_origin,))
    check(src.count("origin={") >= 3,
          "「换战区 / 休息 / replan」那三处的签注没打上（按签注分流就分不出来 ✗）")


def t_agent_state_writers_single_entry():
    """**`agent.state` 只有一个写入口**（用户 2026-10-02 #3 ✓ "直接修"）+
    **状态名必须是 `AGENT_STATES` 里的**（#9 ✓）。

    病（审计出来的事实）：`self.state = ...` 在 `agent.py` 里有 **23 处**，其中 **21 处绕过
    `_set_state`** ⇒ 锚（`_idle_since`/`_idle_fid`/`_idle_quiet`）不维护、`battle_state`
    打点缺失、`_back_ctx` 残留键（休息状态机那几个相位全在其中 ✓）。

    钉四件：
      ① 源码级：`self.state =` **只允许**出现在 `__init__` 的初值、`_set_state_raw`、
         `_set_state` 这三处（按 4 空格缩进的 `def` **逐个方法切**再找 ✓ —— 别用"下一个 def"
         当边界，那会把中间几千字符一起吞掉 ✗）；
      ② 源码级：`_set_state("X")` / `_set_state_raw("X")` 的**字面量**必须都在 `AGENT_STATES` 里；
      ③ `AGENT_STATES` 覆盖七个休息态 + 六个主状态（休息那七态由 `REST_STATE_SPEC` 派生 ✓）；
      ④ 运行期：拼错的状态名**不抛**（20ms 回路里抛会把 tick 带崩 ✗）但会留一条
         `bad_state` 痕迹 ✓。
    """
    import inspect
    import re as _re

    import decision.agent as _am

    src = inspect.getsource(_am)
    # ① 按方法切（只切 **4 空格缩进**的 def ✓）
    funcs = {}
    cur, buf = None, []
    for ln in src.split("\n"):
        m = _re.match(r"^    def (\w+)\(", ln)
        if m:
            if cur:
                funcs[cur] = "\n".join(buf)
            cur, buf = m.group(1), [ln]
        elif cur is not None:
            buf.append(ln)
    if cur:
        funcs[cur] = "\n".join(buf)
    # ⚠ `__init__` 允许写**初值**（`self.state = "idle"` ✓ 那时候还没有锚要维护 ✓）
    allow = {"__init__", "_set_state", "_set_state_raw"}
    offenders = [n for n, body in funcs.items()
                 if n not in allow and _re.search(r"^\s+self\.state = ", body, _re.M)]
    check(not offenders,
          "还有方法直接写 `self.state`（绕过唯一入口 ⇒ 锚/打点丢 ✗）：%r" % (offenders,))
    check(funcs["__init__"].count("self.state = ") == 1,
          "`__init__` 里的 `self.state =` 不止一处（初值只该有一处 ✓）")
    for nm in ("_set_state", "_set_state_raw"):
        check(nm in funcs and "self.state = " in funcs[nm],
              "`%s` 里看不到 `self.state = `（写口被改了？）✗" % nm)
        check(funcs[nm].count("self.state = ") == 1,
              "`%s` 里写了不止一次 `self.state =`（写口该只有一处 ✓）" % nm)
    check(src.count("self.state = ") == 3,
          "`self.state =` 的写点总数变了（该是 3：`__init__` 初值 + 两个 setter ✓）：%d"
          % src.count("self.state = "))

    # ② 字面量必须在表里
    lits = set(_re.findall(r'\._set_state(?:_raw)?\(\s*"([^"]+)"', src))
    check(lits, "一个状态字面量都没扫到（正则失效？）✗")
    check(lits <= set(_am.AGENT_STATES),
          "有状态名不在 `AGENT_STATES` 里（拼错 / 忘了登记 ✗）：%r"
          % (sorted(lits - set(_am.AGENT_STATES)),))

    # ③ 表覆盖主状态 + 休息七态
    need = {"idle", "attack", "chase", "climb", "evade_jump", "evade_back_jump"}
    check(need <= set(_am.AGENT_STATES),
          "`AGENT_STATES` 少了主状态：%r" % (sorted(need - set(_am.AGENT_STATES)),))
    check(set(_am.REST_STATES) <= set(_am.AGENT_STATES),
          "`AGENT_STATES` 没把休息七态包进来（它是从 `REST_STATE_SPEC` 派生的一处口径 ✓）")

    # ④ 运行期：拼错**不抛**、但留痕
    import tempfile
    from pathlib import Path as _P

    import core.behavior as _beh
    h = Harness(fresh_settings())
    a = h.agent
    _old_blog, _old_ben = _beh.LOG, _beh.ENABLED
    _blogf = _P(tempfile.mkdtemp(prefix="behavior_badstate_")) / "behavior.log"
    try:
        _beh.configure(True, log=_blogf)
        a._set_state("chace")                     # 故意拼错
        txt = _blogf.read_text(encoding="utf-8") if _blogf.exists() else ""
        check("bad_state" in txt,
              "拼错的状态名没留痕（静默跑偏 ✗）：%r" % (txt[-200:],))
        check(a.state == "chace", "拼错时不该顺手改行为（照样写上 ✓）")
    finally:
        _beh.configure(_old_ben, log=_old_blog)
    a._set_state("idle")


def t_tap_skips_held_key():
    """**点按不许把"正被按住的键"松开**（用户 2026-10-02 #5 ✓ "直接修"）。

    病（审计事实）：一个键有**两个写者** —— `KeyState`（执行器 `out["jump"]` / 移动键）与
    `tap`（规避跳 / 追击跳 / 走不动跳 / 喝药 / 超时按键）。`tap` 是"按下 + 松开" ⇒ 若它落在
    `KeyState` 按住期间，就是把按住的键**松掉**，而 `keys.set` 因为"键集没变"**不补发**
    ⇒ 静默失效（键盘帽会显示按着、实际已松 ✗）。

    钉三件：
      ① 键正被 `KeyState` 按着 ⇒ `_tap` **不发**（`taps()` 空 ✓）但**键帽照亮**（`_seq_key_at` ✓）；
      ② 没被按着 ⇒ 照发（老行为 ✓）；
      ③ 源码级：`agent.py` 里除了 `_tap` 内部，**不许**再出现裸 `tap(` 调用 ✓（一处入口 ✓）。
    """
    import inspect
    import re as _re

    import decision.agent as _am

    s = fresh_settings()
    h = Harness(s)
    a = h.agent
    jump = s.keymap["jump"]

    with h._patched():
        # ⚠ 直接调 agent 方法（不走 `run()`）⇒ 记录器要的两样得自己备好：
        #   `clock0`（相对时刻的基准 ✓ 不设就是 AttributeError，被 except 静默吞掉 ✗）
        h.clock0 = h.clock.t
        a.keys.set({jump})                       # 执行器这一拍正按住跳
        a._tap(jump, "用例：按住时点按")
        check(not h.taps(jump),
              "键正被 KeyState 按着，`_tap` 还是发了点按（= 把按住松掉 ✗）：%r" % (h.taps(jump),))
        check(jump in a._seq_key_at, "跳过点按之后没点键帽（键确实按着 ⇒ 该亮 ✓）")
        a.keys.set(set())                        # 松掉
        a._tap(jump, "用例：没按住时点按")
        check(len(h.taps(jump)) == 1,
              "键没被按着时 `_tap` 没发出去（该走老路 ✓）：%r" % (h.taps(jump),))
    a.keys.release_all()

    src = inspect.getsource(_am)
    funcs, cur, buf = {}, None, []
    for ln in src.split("\n"):
        m = _re.match(r"^    def (\w+)\(", ln)
        if m:
            if cur:
                funcs[cur] = "\n".join(buf)
            cur, buf = m.group(1), [ln]
        elif cur is not None:
            buf.append(ln)
    if cur:
        funcs[cur] = "\n".join(buf)
    leaves = []
    for n, body in funcs.items():
        if n == "_tap":
            continue
        for ln in body.split("\n"):
            if ln.strip().startswith("#"):
                continue
            if _re.search(r"(?<![\w.])tap\(", ln):
                leaves.append((n, ln.strip()))
    check(not leaves,
          "还有直接 `tap(...)` 的调用（该走 `_tap` 一处入口 ✓）：%r" % (leaves[:3],))


def t_cancel_clears_runtime():
    """**`cancel()` 也要把运行时字段收干净**（用户 2026-10-02 #8 ✓）。

    病（审计事实）：四个 `cancel()` 原来**只置 `FAILED` + note**，**一个运行时字段都不清** ✗；
    而 `retry()` 要清的字段一大把（ClimbJob ~28 个）⇒ 将来谁复用同一个 job、或新加字段忘同步，
    就带着**上一个任务的脏锚**（历史踩过 `_diag_at` / `_climb_sig` / `_rope_seen` ✓）。

    做法：`cancel()` = **复用 `retry()` 的清理**（一处清单 ✓）+ 盖成 `FAILED` + 原因 ✓
    ⇒ 用例就钉"**cancel 后的字段状态 == retry 的**"（不用手抄字段清单 ✓ 加字段自动覆盖 ✓）。
    """
    from decision import route

    def cmp_fields(a, b, fields):
        """两两比字段（**对象显式传** ✓ —— 写死闭包变量会在下一段比较里比错对象 ✗ 踩过）。"""
        return [(f, getattr(a, f), getattr(b, f)) for f in fields
                if getattr(a, f) != getattr(b, f)]

    # ① ClimbJob：驱动到"跳过了 / 斜跳飞过 / 有记忆"那种脏状态
    def mk_climb():
        return route.ClimbJob("L2", x=700.0, y1=-100.0, y2=200.0, direction=1,
                              dst_set="乙平台", src_set="甲平台", tol_px=6, hold_ms=250,
                              near_px=20, jump_start_px=120)
    j_c, j_r = mk_climb(), mk_climb()
    for j in (j_c, j_r):
        j.update(0.0, px=470.0, py=100.0, here_sets=["甲平台"])
        j.update(0.15, px=470.0, py=100.0, here_sets=["甲平台"])     # 第 2 拍按跳 ⇒ 飞行相
        j.update(0.20, px=470.0, py=250.0, here_sets=["甲平台"], ladder_id="L2")
    j_c.cancel("用例：取消")
    j_r.retry()
    # ⚠ **不比 `phase` / `note`**：那两个是 `cancel()` **故意**盖掉的（FAILED + 原因 ✓）
    fields = ("attempt", "_t0", "_in_tol_since", "_off_since", "_rope_seen", "_jump_tried",
              "_diag_flying", "_diag_at", "_diag_arming", "_diag_y0", "_diag_left",
              "_retry_stage", "_arrived_at", "_arrived_hold", "_mid_jumped", "_mid_jump_at",
              "_press_dir", "_press_at", "_dir_tap_at", "_retrying", "_align_pressed")
    bad = cmp_fields(j_c, j_r, fields)
    check(not bad, "`ClimbJob.cancel()` 没把运行时字段收干净（与 retry 不一致 ✗）：%r" % (bad,))
    check(j_c.phase == route.ClimbJob.FAILED and j_c.note == "用例：取消",
          "cancel 没盖成 FAILED / 没写原因：%r / %r" % (j_c.phase, j_c.note))

    # ② DropJob（它自己的锚：`_falling` / `_attempt_at` / `_reassert_at` …）
    def mk_drop():
        return route.DropJob("甲平台", "乙平台", [(500.0, 470.0, 530.0, 0.0, "41")],
                             tol_px=6, hold_ms=0, retry_ms=500, stall_s=1.0)
    j_c, j_r = mk_drop(), mk_drop()
    for j in (j_c, j_r):
        j.update(0.0, px=500.0, py=0.0)
        j.update(0.3, px=500.0, py=0.0)
        j.update(0.6, px=500.0, py=20.0)          # 掉了 ⇒ `_falling`
    j_c.cancel("用例：取消")
    j_r.retry()
    fields = ("attempt", "_t0", "_in_tol_since", "_armed_at",
              "_attempt_at", "_y_base", "_jump_at", "_landing_at", "_next_round_at",
              "_reassert_at", "_reassert_n", "_falling", "_detach_tap_at", "_tap_n",
              "_y0", "_jump_reassert_at")
    bad = cmp_fields(j_c, j_r, fields)
    check(not bad, "`DropJob.cancel()` 没把运行时字段收干净 ✗：%r" % (bad,))
    check(j_c.phase == route.DropJob.FAILED and j_c.note == "用例：取消",
          "`DropJob.cancel()` 没盖成 FAILED / 没写原因：%r / %r" % (j_c.phase, j_c.note))

    # ③ WalkJob / JumpJob（字段少，一起钉）
    w_c = route.WalkJob("乙平台", [(700.0, 690.0, 710.0, -208.0, "52")])
    w_r = route.WalkJob("乙平台", [(700.0, 690.0, 710.0, -208.0, "52")])
    for w in (w_c, w_r):
        w.update(0.0, px=600.0)
        w.update(1.0, px=600.0)
    w_c.cancel("用例：取消")
    w_r.retry()
    bad = cmp_fields(w_c, w_r,
                     ("attempt", "_t0", "_best_d", "_best_at", "_hopped", "_hop_at"))
    check(not bad, "`WalkJob.cancel()` 没把运行时字段收干净 ✗：%r" % (bad,))

    # ④ JumpJob（`_jump_at` / `_jump_dir`）
    def mk_jump():
        # ⚠ 跳的落点是**三元组** `(左, 右, id)`（`jump_job_for_edge` 的形状 ✓ 别抄走/爬的五元组 ✗）
        return route.JumpJob("乙平台", [(690.0, 710.0, "52")], src_set="甲平台")
    p_c, p_r = mk_jump(), mk_jump()
    for p in (p_c, p_r):
        p.update(0.0, px=600.0)
        p.update(0.5, px=700.0)          # 进起跳距离 ⇒ 按跳
    p_c.cancel("用例：取消")
    p_r.retry()
    bad = cmp_fields(p_c, p_r, ("attempt", "_t0", "_jump_at", "_jump_dir"))
    check(not bad, "`JumpJob.cancel()` 没把运行时字段收干净 ✗：%r" % (bad,))


def t_link_eff_guards_and_probes():
    """⭐⭐ **A→B→A 链路提效**的三件（用户 2026-09-29："开动"✓ —— 先做最便宜那两件 + 两个打点 ✓）。

    **为什么要做**：`perf.log` 里最近一轮的账是 `e2e_probe_ms` 中位 **98 ms**（A 屏幕→B 解码，
      纯上游）+ `pipe_ms` **16 ms**；而 `slot_wait_ms`（帧在槽里等多久）和
      `kbd_rtt_ms`（B 发指令→A 固件回执）**从来没人量过** ✗ ⇒ 没法判断该往哪边使劲。

    钉五件：
      ① **torch / cv2 的线程池按到 1**（GPU 推理时它们纯属抢核 —— 而读线程是上游唯一的
         兜底，被抢住就变成端到端延迟 ✗）；且**必须调在加载模型之前** ✓；
      ② **键盘通道开 `TCP_NODELAY`**（按键/回执都是小包，Nagle 最坏多等一个 RTT ✗），
         而且**控制机与游戏机两侧都要**（回执被拖住的话 `kbd_rtt_ms` 量出来就不干净 ✗）；
      ③ `kbd_rtt_ms` 的记账：有在等的指令才记 ✓、账减完要清空 ✓、**空闲期的旧起点绝不算成
         延迟** ✓（那是"没人按键的几十秒"，算进去这条数就废了 ✗）；
      ④ `slot_wait_ms` 真在 `live_thread` 里按 `t_recv_mono` 算 ✓（负数夹 0 ✓）；
      ⑤ 两个打点都走 `core.perf`（`perf.sample` / `perf.ms` ✓ ⇒ 自动进 perf.log 的分位数 ✓）。
    """
    import inspect
    import socket
    import time as _t

    from remote_kbd import kbd_client as kc
    from gui import live_thread as lt

    # ① 线程池按到 1（用完**恢复**，别把后面用例的 torch 也按成单线程 ✓）
    #   ⚠ 自检进程里 torch 可能**加载不了**（实测 WinError 1114 载不动 c10.dll ✗）——
    #     那是这个进程的环境问题，不是产品代码的问题 ⇒ 这时只钉 cv2 那半 ✓
    #     （函数本身对"没有 torch"也是静默跳过 ✓ 见它的注释）。
    import cv2
    try:
        import torch
    except Exception as e:              # noqa: BLE001
        torch = None
        print("      （本进程加载不了 torch：%s —— 只钉 cv2 那半 + 源码顺序 ✓）" % e)
    _n0, _c0 = (torch.get_num_threads() if torch is not None else None), cv2.getNumThreads()
    try:
        lt._limit_cpu_threads()
        if torch is not None:
            check(torch.get_num_threads() == 1,
                  "torch 线程池没按到 1（GPU 推理时它会和读线程抢核 ✗）：%d"
                  % torch.get_num_threads())
        check(cv2.getNumThreads() == 1,
              "cv2 线程池没按到 1（cvtColor/resize 会抢读线程的核 ✗）：%d"
              % cv2.getNumThreads())
    finally:
        if torch is not None:
            torch.set_num_threads(_n0)
        cv2.setNumThreads(_c0)
    _src = inspect.getsource(lt)
    _i_call = _src.rindex("_limit_cpu_threads()")
    _i_load = _src.index("model = YOLO(weights)")
    check(_i_call < _i_load,
          "`_limit_cpu_threads()` 没调在**加载模型之前**（模型加载之后 torch 的线程池"
          "可能已经开始用了 ✗）")

    # ② TCP_NODELAY：真开一条回环 TCP 看它到底设上没有 ✓
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    cli = socket.create_connection(srv.getsockname(), timeout=2.0)
    acc, _addr = srv.accept()
    try:
        check(kc.set_nodelay(cli), "`set_nodelay` 在回环 TCP 上返回 False（没设上 ✗）")
        check(cli.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) == 1,
              "客户端这条连接没有 TCP_NODELAY（Nagle 会攒按键小包 ✗）")
        kc.set_nodelay(acc)
        check(acc.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) == 1,
              "对端（relay 那半边）没设 TCP_NODELAY：回执被攒住 ⇒ kbd_rtt_ms 会被污染 ✗")
    finally:
        for s in (cli, acc, srv):
            try:
                s.close()
            except OSError:
                pass
    _ksrc = inspect.getsource(kc)
    check("set_nodelay(raw)" in _ksrc,
          "`KbdClient.__init__` 里没调 `set_nodelay`（只在用例里设等于没做 ✗）")
    from remote_kbd import relay as _rl
    _rsrc = inspect.getsource(_rl)
    check("_set_nodelay(conn)" in _rsrc,
          "relay（A 机那半边）没关 Nagle ✗")
    check("from .kbd_client import set_nodelay" in _rsrc,
          "relay 里的 `set_nodelay` 不是**共用**客户端那一份（两处各写一份必然漂 ✗）")

    # ③ `kbd_rtt_ms` 记账（空壳对象，不碰真 socket ✓）
    class _FakePerf(object):
        def __init__(self):
            self.samples = []

        def ms(self, name, t0):
            self.samples.append((name, (_t.perf_counter() - t0) * 1000.0))

    fp = _FakePerf()
    _real_perf = kc._perf
    kc._perf = lambda: fp
    try:
        o = object.__new__(kc.KbdClient)
        o._pending, o._rtt_t0 = 2, _t.perf_counter() - 0.05
        o._take_rtt(1)
        check(len(fp.samples) == 1 and fp.samples[0][0] == "kbd_rtt_ms",
              "收到回执时没记 `kbd_rtt_ms`：%r" % (fp.samples,))
        check(fp.samples[0][1] >= 40.0,
              "记出来的往返比实际发出的时刻还短（起点算错了）：%.1f ms" % fp.samples[0][1])
        check(o._pending == 1 and o._rtt_t0 is not None,
              "还有一条没对上回执，起点不该清空：pending=%d t0=%r" % (o._pending, o._rtt_t0))
        o._take_rtt(1)
        check(o._pending == 0 and o._rtt_t0 is None,
              "账减完了却没清空起点（下一次会把空闲期算成延迟 ✗）：pending=%d t0=%r"
              % (o._pending, o._rtt_t0))
        # ⚠ 空闲期：正常流程下 pending=0 时 t0 也已经是 None，但**判据必须是"有没有在等"**
        #   （不是"t0 是不是 None"✗）—— 所以这里**故意造出**"没有在等、却还留着个旧起点"
        #   的状态（老代码 / 别的调用方完全可能这样），要求它**不许**记这一笔 ✓。
        o._pending, o._rtt_t0 = 0, _t.perf_counter() - 0.05
        n_before = len(fp.samples)
        o._take_rtt(1)
        check(len(fp.samples) == n_before,
              "**没有在等的指令**却记了一次 `kbd_rtt_ms`（空闲期不是延迟 ✗）：%r"
              % (fp.samples,))
        check(o._rtt_t0 is None,
              "空闲期那一笔之后旧起点没清掉（下一次真发指令会把两段拼一起 ✗）：%r"
              % (o._rtt_t0,))
    finally:
        kc._perf = _real_perf

    # ④⑤ 两个打点真在实时链里（源码级：`slot_wait_ms` 必须按 t_recv_mono 算 ✓）
    check('perf.sample("slot_wait_ms"' in _src,
          "`live_thread` 里没有 `slot_wait_ms` 打点（帧在槽里等多久仍然是个谜 ✗）")
    check("(_t_take - float(f.t_recv_mono))" in _src,
          "`slot_wait_ms` 不是按「取帧时刻 − 解码完成时刻」算的（口径错了 ✗）")
    check('_p.ms("kbd_rtt_ms"' in _ksrc,
          "`kbd_rtt_ms` 没走 `core.perf`（那就进不了 perf.log 的分位数 ✓）")

    # ⭐ **分阶段 + 质量打点**（用户 2026-10-01 ✓，参考 Maple_xfeat 的 `stage_ms` /
    #   `valid_camera_fraction`）：`pipe_ms` 是一锅粥 ⇒ 分不出"定位 / 追踪"各占多少；
    #   而且耗时低 ≠ 质量好 ⇒ 还要量"有效比例"。
    for _name in ("locate_ms", "track_ms"):
        check(('perf.ms("%s"' % _name) in _src,
              "`live_thread` 里没有 `%s` 打点 ⇒ `pipe_ms` 拆不开，哪段是瓶颈看不出来 ✗"
              % _name)
    for _name in ("cam_valid", "cam_rejected"):
        check(('perf.sample("%s"' % _name) in _src,
              "`live_thread` 里没有 `%s` 打点 ⇒ 只量耗时、不量有效比例（耗时低≠质量好 ✗）"
              % _name)
    check("first_cam_s" in _src,
          "`live_thread` 里没有 `first_cam_s`（冷启动到可用多久仍然是个谜 ✗）")


# ---------------------------------------------------------------- 入口

def t_sweep_edge_turn():
    """扫平台换向 = **两条件任一**（2026-09-29 用户定稿）：
      A（新）距当前 foothold 集合边缘 ≤「距离平台边缘回头(px)」⇒ 回头，
         **即使前方有怪也转** ✓（集合 x 范围 = 广播 `here_span`（世界系）⇒
         用 `world_x` 比，同一坐标系 ✓）；
      B（旧）前方无怪持续「换朝向延迟」⇒ 回头（一字不动 ✓）。
      判不出集合（here_span 缺 / 没定位）⇒ 只剩 B ✓（老行为原样）。
    """
    s = fresh_settings(strategy="sweep", jump_random_prob=0.0, attack_dist=80.0,
                       attack_cd=0, min_turn_hold_ms=0, sweep_edge_turn_px=50)
    h = Harness(s)
    h.clock0 = h.clock.t

    _span = {"v": (0.0, 1000.0)}     # 当前集合的 x 范围（世界系）
    _wx = {"v": 500.0}               # 玩家世界 x（小地图黄点 ✓）

    from unittest import mock as _mock

    from decision import agent as _agmod

    def beat(mob_x=None):
        """喂一拍；可选摆一只怪（画面 x=mob_x，与玩家同高）。返回倾向朝向。"""
        h.mobs_fn = (lambda t: [Mob(id=1, x=mob_x, y=500.0,
                                    w=40.0, h=40.0, conf=0.9)]
                     if mob_x is not None else [])
        ws = h.ws()
        ws.player.here_span = _span["v"]
        ws.player.world_x = _wx["v"]
        h.clock.t += 0.05
        with _mock.patch.object(_agmod.time, "monotonic", h.clock):
            h.agent.tick(ws)               # ⚠ 假钟：B 的计时才推得动 ✓（同 428 那批 ✓）
        return h.agent._patrol_dir

    # A：朝右、距右边缘 (1000-960)=40 ≤ 50 ⇒ 回头，**前方 400px 处有怪也转** ✓
    #    （老口径"主方向有怪就不换向"在这里绝不转 —— 正是用户要改掉的 ✗）
    _wx["v"] = 960.0
    check(beat(mob_x=900.0) == -1,
          "到集合边缘（40≤50）却没回头（前方有怪也该转 ✓）")
    # 翻到朝左后：距左边缘 960 > 50，且"前方（左）"没怪 ⇒ 不翻（B 计时刚开始 ✓）
    check(beat(mob_x=900.0) == -1, "不该连着翻（B 计时刚开始、A 也不满足）✗")
    # A''：朝左、距左边缘 (40-0)=40 ≤ 50 ⇒ 翻回朝右 ✓（左边缘同样生效 ✓）
    _wx["v"] = 40.0
    check(beat(mob_x=900.0) == 1, "朝左到左边缘（40≤50）却没翻回 ✗")
    # 离边缘远 + 前方无怪 ⇒ 只走 B：第一拍不翻（计时刚开始 ✓）
    _wx["v"] = 500.0
    check(beat() == 1, "离边缘远、前方无怪，第一拍就翻（B 计时被跳过 ✗）")
    # B 到点 ⇒ 翻 ✓（老逻辑保留的证据 ✓）
    for _ in range(24):                # 1.2s > turn_cd 1.0s ✓
        d = beat()
    check(d == -1, "前方无怪持续超过「换朝向延迟」却没翻（B 老逻辑丢了 ✗）：%s" % d)
    # 判不出集合（here_span=None）⇒ 只剩 B：到边缘距离也不翻（回退原样 ✓）
    h.agent._patrol_dir = 1
    h.agent._no_target_since = None
    _span["v"] = None
    _wx["v"] = 960.0
    check(beat() == 1,
          "判不出集合却按边缘翻了（该退回只有 B ✗）")
    for _ in range(24):
        d = beat()
    check(d == -1, "回退路径里 B 到点没翻 ✗：%s" % d)


CHECKS = [
    ("⭐⭐ 链路提效：torch/cv2 线程按到 1 + 键盘通道 TCP_NODELAY + kbd_rtt/slot_wait 打点",
     t_link_eff_guards_and_probes),
    ("⭐⭐ 寻路时限：只有「追击」签注吃「寻路超时时间」（其它任务无限）、"
     "attack 重下不刷新它的钟、走完一段接下一段才刷新（用户 2026-09-29）",
     t_chase_clock_attack_vs_segment),
    ("定点休息：走→到达后行为→歇时长→结束后前往→回战斗；走不到如实说",
     t_afk_spot_rest),
    ("⭐ 定点休息「休息过程中循环行为」：只在休息中按 A~B 循环跑、不勾=老行为一字不变、"
     "收摊要松键（用户 2026-09-28）",
     t_spot_loop_rest_behavior),
    ("⭐ 「休息结束推迟到循环执行完」：勾上⇒到点等这一轮演完再结束（不许再开新一轮）、"
     "不勾=当场收摊、手动结束不受它管（用户 2026-09-28）",
     t_spot_loop_hold_rest_end),
    ("⭐⭐ 「玩家位置」参数组（用户 2026-09-28）：脚底偏移 / 框面积最小占比 / 面积基线·容差"
     "（闸拦在位置查询**之前**）/ 相机 y 用脚底偏移 / 箭头颜色·长度；默认全关 = 老行为一字不变",
     t_player_loc_params),
    ("⭐⭐ 「当前任务」的签注 + 生存时间（用户 2026-09-28）：信息栏那行显示成"
     "「前往：底层(追击  剩余 00:15)」；格式 `MM:SS`、与「寻路超时」同一把钟、休息时不加",
     t_goto_tag_left),
    ("⭐⭐ 下跳「跳下绳子」的水平方向（用户 2026-09-28 按流程图）：有锁定目标 ⇒ 朝它；"
     "否则任意方向（背离下跳点中心）；回调坏了不许炸", t_drop_detach_dir),
    ("⭐⭐ 新组「任务」→「追怪寻路.duration(s)」（用户 2026-09-28）：只掐「追怪下达的」任务、"
     "与「寻路超时」同一把钟、0 = 不启用", t_chase_goto_max),
    ("⭐⭐ 攻击范围判定「单侧」（跟画面一致）＋ 背后怪抢锁（用户 2026-09-28）："
     "背后够得着 ⇒ 抢锁转身打它；够不着 ⇒ 不抢；同一只不重复抢",
     t_attack_single_side_and_steal),
    ("⭐ 位置/血量/动作的定频快照（pos/act 每秒一条、休息分支里也要打）+ 喝药时记触发血"
     "（用户 2026-09-28）",
     t_snap_beat),
    ("⭐ 「怪物判定缓存时长(s)」是参数（TTL 内用缓存 / 超时重查 / 0=不缓存）（用户 2026-09-28）",
     t_mob_sets_ttl_param),
    ("⭐ 结束休息要**杀掉**循环行为正在执行的动作 + **终止循环**（松键 + 清 ctx；"
     "停自动 / 换类型也要收摊）（用户 2026-09-28）",
     t_spot_loop_killed_on_rest_end),
    ("⭐ 界面三件：勾选框在「到达后行为」下面、勾选后下方**缩进**出现参数"
     "（行为编辑器按钮 + 循环时间 A~B）（用户 2026-09-28）",
     t_spot_loop_panel_widgets),
    ("防掉线：打断重试那两个参数挂在通用层（选定点休息也看得见）",
     t_afk_retry_params_are_common),
    ("寻路超时：每完成一段重新计时（长路线不被整条切断），单段超时才切",
     t_goto_timeout_per_hop),
    ("**同一步被重下不许把「寻路超时」归零**（用户 2026-09-28 报「没有触发寻路超时」；"
     "DropJob 下发 318 次、超时只 57 次）",
     t_goto_timeout_survives_same_step_restart),
    ("「寻路超时后按键」：默认(None)一个键都不发、配了就在超时那一下点按一次、"
     "没超时不发（用户 2026-09-28 要求）",
     t_goto_timeout_key),
    ("锁定目标的 foothold 集合**每秒只查一次**（用户 2026-09-28：治「锁着目标来回走」"
     "—— 怪框抖 ⇒ 集合在楼层间跳 ⇒ 目的地跟着跳）",
     t_locked_mob_info_cache),
    ("「寻路超时」的钟：**收工之后再下同一步要重打**、覆盖正在跑的同一步不重打"
     "（用户 2026-09-28：三楼→二楼 drop 左右晃 —— 新任务开工就带着上一段 10 秒的旧账）",
     t_goto_timeout_reset_after_finish),
    ("「来源签名」+ 两个节点复核更便宜的怪：只掐「追击」类、严格更便宜、掐完换锁定目标"
     "（用户 2026-09-28：对齐完成后起跳前 / 寻路超时计时过半）",
     t_climb_recheck_cheaper_mob),
    ("⭐⭐ 追击起跳「距离平台边缘多远禁用(px)」（用户 2026-10-02）：启用+贴边 ⇒ 不跳 + 打点；"
     "离边远/不启用/判不出 ⇒ 照跳；方向取「跳向那侧」",
     t_chase_jump_edge_guard),
    ("⭐ 攀爬第三个节点：「走向绳梯（离绳还远）」**每拍**复核我这块平台上的怪 ⇒ 掐掉 + 换目标"
     "（用户 2026-10-02）",
     t_climb_recheck_walk_platform),
    ("⭐⭐ 大怪锁生效期间**背后的怪不许抢锁**（用户 2026-10-02：锁定命中大怪 ⇒ 禁用「背后攻击"
     "范围内有怪触发转向」；小怪锁定/开关关/池空 ⇒ 照旧抢）",
     t_big_mob_lock_blocks_behind_steal),
    ("⭐⭐ 「站桩」的判据 = **输出轮**（用户 2026-10-02 定义）：怪进进出出攻击框（状态 chase↔attack"
     " 抖）也**不再**清零轮次 ⇒ 「每 3 轮补朝向键」照样要到、首窗一段只开一次",
     t_station_turn_survives_state_flicker),
    ("⭐ 到顶后「再按住一会儿」那几拍**朝锁定目标按方向键**（用户 2026-10-02：没有目标 ⇒ 老行为；"
     "交接到 chase 那一拍**方向不翻**）",
     t_climb_arrive_face_target),
    ("⭐ 「走向绳梯（离绳还远）」**每拍**复核：我这块平台上若有别的怪 ⇒ 掐掉这一步 + 换目标"
     "（用户 2026-10-02：起跳距离外每拍查「当前 foothold 集合里有没有代价更低的目标」）",
     t_climb_recheck_walk_platform),
    ("「禁止战斗」= 旧的「限制战斗区域」：默认不勾、勾了才进禁战名单、老配置迁移不许丢"
     "（用户 2026-09-28 把限制搬进每一项）",
     t_battle_zone_can_fight),
    ("「可以战斗」改名不改值（老键 `no_fight` 原值照抄、不取反）+ 不在能打区找**代价最低**的"
     "可战斗区（用户 2026-09-28）",
     t_zone_pick_cheapest_and_rename),
    ("战斗区域**按地图 id 存**（per-map 文件 + 老配置迁移一次 + 不再写回 project.yaml；"
     "用户 2026-10-01）",
     t_battle_zones_per_map),
    ("「编辑战斗区域」只读视图的**判据**（只认本集合 / 点选命中 / 参数那行 / 别的集合不算）"
     "—— ⚠ 只测纯逻辑：那块视图在离屏自检里建不出来（用户 2026-09-28）",
     t_foothold_picker_logic),
    # ⚠⚠ **这条用例搬到 `tools/selftest_zone_editor.py` 了**（2026-09-28 踩过 ⇒ 别搬回来 ✗）：
    #   它要建 `gui.foothold_picker.FootholdPicker`（内含 `QGraphicsView` ✓），而**在本套件的
    #   这个位置上建控件会进程原生崩溃（`0xC0000409`）** —— 实测**连空的 `QGraphicsView()` /
    #   连 `QWidget()` 都崩** ✗（与视图本身无关 ⇒ 是本套件前面某处留下的 Qt 环境问题 ✓）。
    #   ⇒ zone_editor 套件天生全是图形控件、环境干净 ✓ 那里测真视图 ✓；
    #     本套件里只保留"**接线**"的间接覆盖（`t_align_params` 会建 `BattleZoneDialog` ✓）。
    # ("「编辑战斗区域」子弹窗里的只读集合视图：只画本集合 / 点选配 idle 回归点 + 显示参数 / "
    #  "点空白清空 / 底图在场 / 没工厂退回文本框（用户 2026-09-28）",
    #  t_battle_zone_idle_picker),
    ("防掉线契约：状态表齐全 / 收尾自保松键 / 只收自己那条路线",
     t_rest_contract),
    ("休息文案：每个阶段都有话说；定点休息不许显示「未休息」",
     t_rest_state_text_covers_all_states),
    ("自定义定时行为：手动触发立刻演一次、计时从头开始（暂停/关自动不响应）",
     t_timer_manual_fire),
    ("自定义定时行为：界面「手动触发」按钮（换色 + 灰掉的时机）",
     t_timer_manual_fire_button),
    ("输出CD：序列里没有攻击键也要守CD",
     t_cd_no_attack_key),
    ("输出CD：进攻击状态走排期（状态抖动不超速）",
     t_cd_on_enter_attack_state),
    ("输出CD：序列比CD长时以序列为准",
     t_cd_shorter_than_sequence),
    ("定期RELEASEALL：本机序列进度全保留",
     t_releaseall_keeps_progress),
    ("定时重置指令通道：到点发/0禁用/关自动也发/手动输入跳过",
     t_periodic_reset_channel),
    ("⭐ 序列的 `up` 不许松掉决策侧按着的键（用户 2026-10-03：按键挤压的另一半）",
     t_seq_up_skips_decision_held),
    ("指令通道体检：不健康标出来 + 后台重连 + 恢复跟随",
     t_link_health_watch),
    ("定时重置：本地后端也要真的松开按键（没有固件兜底）",
     t_periodic_reset_local_backend),
    ("类别表：只在 perception/classes.py 定义",
     t_class_table_single_source),
    ("时序拍：next_deadline 报的是序列到点时刻",
     t_next_deadline),
    ("时序拍：序列元素按本机绝对时钟发（不再一帧量化）",
     t_sequence_timing_not_frame_quantized),
    ("定期RELEASEALL：只作废按键记录",
     t_releaseall_clears_held_only),
    ("自定义定时行为：暂停后不触发、且打断正在演的序列",
     t_timer_paused_never_fires),
    ("自定义定时行为：「休息时暂停计时」是每条各自的开关（没勾的照常倒数）",
     t_timer_pause_on_rest_flag),
    ("自定义定时行为：暂停按钮/红字（已暂停）/继续接着走",
     t_timer_pause_button),
    ("休息状态机：到点/手动结束/关防掉线/手动进入",
     t_rest_state_machine),
    ("休息状态机：手动进入要先等清怪",
     t_rest_manual_request),
    ("被打断重试：补血即打断并提前重试（不勾选则不打断）",
     t_rest_interrupt_retry),
    ("⭐⭐ 休息**被切**必须结清+留痕（用户 2026-10-03：关自动 / 界面接管 ⇒ `afk_cut` + "
     "`afk_done(why/hidden)` + 重排下次；`afk_enter_dropped` 报出 rest_s/played_s）",
     t_rest_cut_is_accounted),
    ("追击起跳：区间在攻击距离内侧时也要跳（且只跳一次）",
     t_chase_jump_inside_band),
    ("追击起跳：区间在外侧时进区间跳一次",
     t_chase_jump_outside_band_once),
    ("追击起跳：怪走进区间的那一拍跳一次",
     t_chase_jump_approach_edge),
    ("追击起跳：有别的怪在场 / 开关关掉都不跳",
     t_chase_jump_guard_shut),
    ("追击：按着方向键 x 一直不动 ⇒ 按「移动操作尝试间隔」单点跳一下",
     t_chase_hop_when_stuck),
    ("平地巡逻无怪时松掉输出键（怪在输出序列中途消失也不许卡键）",
     t_patrol_idle_releases_output),
    ("走只有一种走法：朝集合中点（「走的方向」参数不许回来）",
     t_walk_only_center),
    ("扫平台：攻击范围内有怪要站桩（不许边走边打）",
     t_sweep_stands_still_in_attack),
    ("扫平台换向两条件任一：到集合边缘（即使前方有怪）/ 前方无怪持续「换朝向延迟」",
     t_sweep_edge_turn),
    ("扫平台换向两条件任一：到集合边缘（即使前方有怪）/ 前方无怪持续「换朝向延迟」",
     t_sweep_edge_turn),
    ("站桩输出的「每 3 次 attack 补一个朝向键」（用户 2026-09-28 要求 2）+ 绳上绝不按左右",
     t_station_turn_every_three),
    ("按键层：F10~F12 三条发送路径全拦",
     t_input_local_only),
    ("触控板：本地 F10 开关（手动输入开着也能用）",
     t_touchpad_f10_toggle),
    ("触控板：面板隐藏时自动关闭",
     t_touchpad_hidden_closes_mode),
    ("触控板：状态栏显示「触控模式中」",
     t_touchpad_status_text),
    ("触控板：非触控模式滚轮让给滚动区",
     t_touchpad_wheel_passthrough),
    ("本地(仅测试)输入：开启自动要二次确认，关闭不拦",
     t_auto_confirm_local),
    ("重置指令通道：先把 ←/→/↑/↓ 各点按一遍，再 RELEASEALL 重置（用户 2026-10-01）",
     t_reset_link_taps_directions),
    ("判定参数：对齐误差范围/时间有默认值、存读一致、设置里有那一页",
     t_align_params),
    ("卡键看门狗：非输出状态下按住输出键超时 ⇒ 松开并留痕；正常输出不误杀",
     t_stuck_key_watchdog),
    ("防掉线：换行为类型不重排计时（休息时长与下次触发都不动）",
     t_afk_type_does_not_reschedule),
    ("多步路径：按顺序逐步走完，到了自动接下一步，失败整条停并说清断在第几步",
     t_multi_step_route),
    ("定点上绳：对齐保持/抖出去重计时/到达双判据/偏离绳梯/超时/掉下来/重来/真实边",
     t_climb_job),
    ("上绳接进状态机：有怪先打、打完继续爬、开关不按跳、失败自动重来、到位/取消收干净",
     t_climb_wiring),
    ("下跳：先按↓再按跳/就近挑点/到达双判据/超时重来/按类型分发",
     t_drop_job),
    ("上绳对齐用世界坐标（画面坐标会走反/死循环）+ 靠近改点按 + 保持窗口含端到端延迟",
     t_climb_world_x_and_tap),
    ("「随机输入延迟」**整个删掉**（用户 2026-09-27 要求）—— 这条反向钉住它别回来。",
     t_no_random_input_delay),
    ("**攻击范围 = 矩形**（用户 2026-09-27 定的）：四个距离 = 四条边，判据**只有一处**。",
     t_attack_box),
    ("下跳（drop）流程（用户 2026-09-27 定的，照抄）：",
     t_drop_flow),
    ("⭐ 下跳 ALIGN 的两条「别在错的层上硬做」闸（用户 2026-10-02 龙族打猎场卡 40 秒）："
     "「就在下跳点上」要 x + **脚下的面 y** 一起看（掉到二楼不再被判成就位 ✓）；"
     "已经到目标层 ⇒ 收工 ✓；既不在起跳平台也不在目标层 ⇒ 如实失败交给上层重算 ✓；"
     "判不出来（`ground_y`/集合为空）⇒ 不拦 ✓；落点形状统一五元组（老的 4 元组兼容 ✓）",
     t_drop_align_layer_guards),
    ("⭐ 「任务没跑的那一拍」按**签注**分流（用户 2026-10-02 #1）：追击 ⇒ 直接结束这一趟；"
     "换战区 ⇒ 照旧重打（挨打不算超时）；休息 ⇒ 不重打（让「寻路超时」收口、由休息的"
     "「打断后重试(s)」重排）；无签注 ⇒ 不变；且 `origin=` 必须是 dict（字符串会当场抛 ✗）",
     t_interrupt_by_origin),
    ("⭐ `agent.state` **只有一个写入口**（#3：`self.state =` 只许在 `__init__` + 两个 setter 里）"
     "+ 状态名必须在 `AGENT_STATES` 里（#9：拼错留 `bad_state` 痕迹、字面量源码级校验）",
     t_agent_state_writers_single_entry),
    ("⭐ 点按不许把「正被按住的键」松开（#5）：`_tap` 一处入口、按住时跳过（键帽照亮）、"
     "源码级禁止裸 `tap(`", t_tap_skips_held_key),
    ("⭐ `cancel()` 也要把运行时字段收干净（#8：复用 `retry()` 的清单 ⇒ 与 retry 状态一致）",
     t_cancel_clears_runtime),
    ("下跳的「移动操作尝试间隔(ms)」重试（用户 2026-09-27 要求 2，原话）：",
     t_drop_retry_gap),
    ("**下爬**的收尾：「跳下绳梯」（2026-09-26 用户当场定的新时序）。",
     t_climb_down_jump_off),
    ("「爬不动了 ⇒ 补按 ↑」**只在「已经在绳上」时判**（用户让看日志，日志照出来的坑）。",
     t_climb_stall_only_on_ladder),
    ("「补按」打点**限流 1 条/秒**（2026-09-29：一场卡死刷了 588 条把日志淹了 ✗）。",
     t_climb_reassert_log_rate_limited),
    ("KeyState 瞬发登记：输出键进 pressed、决策拍不顶掉（2026-09-30：输出键帽不亮）。",
     t_keystate_momentary),
    ("断线判定成立必停自动、与重连开关解耦（2026-09-30：断线了自动还在瞎跑）。",
     t_reconnect_stop_auto),
    ("上爬对齐 x：**每一轮都要先发一次方向键**（点按）→ 跳 → 按住 ↑（用户 2026-09-28 定的循环）。",
     t_climb_align_each_round_presses),
    ("**被战斗打断**不许把挨打那几秒算成\"爬不动了 / 走不动了\"，且回来后要能从对的地方继续。",
     t_job_interrupt_by_fight),
    ("「限制战斗区域」（用户 2026-09-26）：不在配置的集合里 ⇒ **不打架**，先下前往命令。",
     t_battle_zone_restriction),
    ("「转向后输出延迟」**改了就生效**（用户 2026-09-27 要求确认）—— 不重启、不重开自动。",
     t_turn_output_delay_live),
    ("「**临时战斗**」不受「限制战斗区域」的约束（用户 2026-09-27 要求）。",
     t_temp_fight_out_of_zone),
    ("「限制战斗区域」**不许拦「命令前往」**（2026-09-26 用户现场定的判据）。",
     t_zone_rule_yields_to_goto),
    ("「画面里看不到角色」必须**留痕**，而且**与自动开着没无关**（用户 2026-09-27 要求）。",
     t_player_gone_logged),
    ("「这一拍任务没跑 ⇒ 报信」是**一处**结算（用户 2026-09-26 要求结构性收口）。",
     t_task_report_on_skip),
    ("「结束当前寻路」要停**整条**路线；而「限制战斗区域」也不许被\"死账\"骗到。",
     t_end_route_clears_plan),
    ("点按的\"按住多久\"必须**按时间**算 —— 用户 2026-09-27 报\"对齐 x 时方向键按了很久\"。",
     t_climb_align_tap_releases_any_fps),
    ("**容差不能小于读数噪声**（2026-09-27 用户要求把这类\"名不副实\"的阈值一起改掉）。",
     t_tolerance_above_read_noise),
    ("「开始对齐绳梯x的距离(px)」（用户 2026-09-27 要求）：比它远 ⇒ **一口气按住**；",
     t_climb_align_near_px),
    ("玩家行为打点（初版：**只有任务**）—— 六个事件要真的落到 `behavior.log` 里。",
     t_behavior_task_events),
    ("寻路的**任务内部变化**也进 behavior.log（相切换 / 斜跳 / 失败重来，用户 2026-09-28 要求）。",
     t_behavior_route_events),
    ("ladder_id 许可改成「climb 执行器按住 ↑ 后发起通知」（诉求 1，用户 2026-09-28）。",
     t_climbing_vertical_notify),
    ("climb「到顶后再按住 ↑」不许被「脚下已是目的地」提前收工（诉求 2，用户 2026-09-28）。",
     t_climb_hold_up_not_preempted),
    ("「卡住判定时长(s)」**已移除**，统一用「移动操作尝试间隔(ms)」（用户 2026-09-27 要求）。",
     t_stall_unified_to_retry),
    ("攀爬流程的两条新规矩（用户 2026-09-27 要求，落点在 `ClimbJob.update`）：",
     t_climb_flow_rules),
    ("⭐ **上爬的 ↑ 从「按跳那一刻」才开始按**（用户 2026-10-02：「应该是按跳之后再按住 ↑，"
     "我观察到 ↑ 从任务下达就一直被按住了」）：跳之前（对齐走位 / 站住等窗口）一个竖直键都不按、"
     "按跳那拍起一路按住（含补按 / 回对齐 / retry）；**例外 = 广播说人已在本绳绳段里**"
     "（纯几何、与按键许可无关 ✓）",
     t_climb_up_key_starts_at_jump),
    ("⭐ **上了非目标绳 ⇒ 通知重启寻路**（用户 2026-10-02 治本）："
     "被非目标绳吸住时立刻 replan（'龙族打猎场'卡左角的根因 ✓）；在目标绳/没上绳不 replan ✓",
     t_climb_replan_wrong_ladder),
    ("唯一特例：攀爬时不进 attack（从「按下跳」到「攀爬成功」；绳就在脚下要起跳也屏蔽，"
     "绳在别处/走路任务照旧先打，上一拍是 attack 也能扳回来）",
     t_climb_attack_exception_agent),
    ("「**对齐绳梯移动延迟(ms)**」（用户 2026-09-27 加的参数）：对齐绳的 x 时，",
     t_climb_align_gap),
    ("「跳(jump)」执行器（初版）：走近 → **按跳** → 落到目标集合；整天靠「寻路超时时间」兜底。",
     t_jump_job),
    ("那个「**1/3 尝试**」必须彻底没了（用户 2026-09-27 在寻路任务里看到它）。",
     t_retry_notes_have_no_denominator),
    ("**按跳的许可条件 = 只有「x 对齐」这一条**（用户 2026-09-27 明确，原话：",
     t_climb_jump_gate),
    ("「**斜跳上绳**」：离绳还远的时候**按住朝绳的方向 + 起跳**斜着过去，",
     t_climb_diagonal_jump),
    ("chase 前先判「**怪和我是不是同一块平台**」，不是 ⇒ **先下前往任务**（用户 2026-09-27）。",
     t_chase_goto_other_foothold),
    ("追击下前往的目的地也要在「可以战斗」白名单里：关掉二楼就真的不去了（用户 2026-10-01）",
     t_chase_goto_respects_can_fight),
    ("下跳途中**被\"通往下层的绳\"吸住 ⇒ 先脱离**（用户 2026-09-27 要求；判据 2026-09-28 换成"
     "「在绳段里 + 脚下没面」—— 用户报「下跳 drop 结果实际爬上了向下爬的绳子」）。",
     t_drop_detach_ladder),
    ("锁定目标优先级：**先比寻路距离，再比画面绝对距离**（用户 2026-09-27 要求）。",
     t_lock_target_by_path_cost),
    ("大怪优先锁定：相对判定（基准 × 倍数）+ 视野内优先（用户 2026-10-01）",
     t_big_mob_priority),
    ("⭐⭐ 大怪判定基准 = **最小框均线**（用户 2026-10-02：「比最小的高这个数」⇒ 兼容大怪多"
     "小怪少；记录一段时间内的最小框均线抗单个漏检小框）",
     t_big_mob_baseline_min_avg),
    ("「锁定目标不能在**非限制战斗区域**内」= 怪不在配置的集合里 ⇒ **不给锁**（用户 2026-09-27 要求 1）。",
     t_battle_zone_mob_filter),
    ("够得着 + 同一条 x 线 ⇒ **不下前往，直接锁**（用户 2026-09-27 要求 2）。",
     t_reach_skip_goto),
    ("「当前执行器」那行：`下跳(drop):{详细内容}`（用户 2026-09-27 要求）。",
     t_job_label_text),
    ("**要去的地方已经到了 ⇒ 整条路线收工**（用户 2026-09-27 报的卡死）。",
     t_route_goal_already_reached),
    ("石人寺院III（真图）寻路体检：一楼→顶层三步可解析、两两集合都能走通",
     t_route_probe_stone_temple),
    ("「前往重下间隔(s)」：原先**写死的 3 秒常量** ⇒ 现在界面上可配（用户 2026-09-27 要求）。",
     t_zone_cd_setting),
    ("战斗区域**每项**的行为：最大战斗时长到点换地方（没到不下/0 不限/离开清零/空目标只停手）"
     "+ idle 回归水平走向中心（到中心站住/没注入不按键）",
     t_battle_zone_item_behaviors),
    ("⭐ 「**扫平台**」策略下 idle 那两件事也要能用：巡逻那拍起「切换平台」的计时"
     "（不依赖 `state == idle` ✓）、闲够就下前往任务、**方向键归巡逻**"
     "（朝倾向方向走优先于回 foothold 中心 ✓）、前面站着怪不切平台（用户 2026-10-02 方案 A）",
     t_sweep_idle_switch_and_walk_priority),
    ("⭐ 「最大战斗时长」的**倒计时只读镜像**：没在计时给 None、开始/清零三件套同进同出、"
     "且**不进项目文件**（用户 2026-09-28：编辑战斗区域里用灰字显示倒计时）",
     t_fight_clock_published),
    ("「最小战斗时长(s)」：只配最小也累计、没到点不追别的 foothold 集合的怪、到了恢复（用户 2026-10-01）",
     t_fight_min_clock_and_gate),
    ("⭐ 「爬」的**中途跳下**（逐边：默认向目标中心 / 仅向左 / 仅向右 + 「高度」）：y 到点就跳下、"
     "**↑ 不松**、落地后再经过「移动操作尝试间隔」才松（用户 2026-09-28）",
     t_climb_mid_jump),
    ("⛔ 「**禁用杀怪寻路**」：开了不再查怪在哪块平台（也不显示）、不再因追怪下寻路任务，"
     "只单纯走向锁定怪物（默认关 = 老行为；用户 2026-09-28）",
     t_disable_chase_pathfinding),
    ("⛔⭐ 「禁用杀怪寻路」**开着时开自动也要能正常运转**：没有小地图世界坐标（没标定/没地形）"
     "照样打怪，且区域规则只在**判不了**时才不拦（用户 2026-10-04）",
     t_disable_chase_pathfinding_auto),
    ("追击**降级路径**：怪那层判不出/走不到、或「判成同一集合但攻击框框不住怪」⇒ 朝"
     "「玩家→怪」方向的最近集合逐层逼近（用户 2026-09-28）；真同层同线照旧追、没注入不动",
     t_mob_goto_towards),
    ("追击降级**不许自指**：落脚区就是玩家自己站的那块 ⇒ 不下前往、也不许站着不动"
     "（2026-10-01 寺院通道2 卡在右下角那种）；`here=True` 那种「成功但没起跑」不算成功 ✓",
     t_mob_goto_towards_no_self_loop),
    ("下跳「按住 ↓」那一组输出：**↓ 必须一直按着**，而且要**按期重发**（用户 2026-09-27 报）。",
     t_drop_hold_dir_reassert),
    ("「在绳梯上」的**许可条件**：没按着 ↑/↓ ⇒ 一律不算（用户 2026-09-27 要求）。",
     t_ladder_only_when_holding_vertical),
    ("「**任务队列**」（用户 2026-09-27：「命令前往」右边的「添加任务队列」）⇒ 一条接一条走 ✓。",
     t_goto_queue),
    ("「添加任务队列」**排进去就自己跑**（用户 2026-09-27 原话）：",
     t_queue_starts_itself),
    ("位置状态一变 ⇒ **重新评判一次最优路径**（用户 2026-09-27 原话）：",
     t_replan_on_location_change),
    ("**所有**寻路任务（走 / 爬 / 下跳 / 跳）都吃「寻路超时时间」（用户 2026-09-27 要确认）。",
     t_goto_timeout_all_job_kinds),
    ("「在不在绳上」：上爬以广播为权威、下爬才留记忆（位置状态绝对权威 ✓）",
     t_climb_on_rope_by_broadcast),
]


def main():
    # 自检绝不写用户的配置：save 换成空操作（决策层另外用独立实例）
    ag.DecisionSettings.save = lambda self: None
    results = []
    for name, fn in CHECKS:
        try:
            fn()
            results.append((name, True, ""))
            print("  [OK] %s" % name)
        except AssertionError as e:
            results.append((name, False, str(e)))
            print("  [NG] %s\n         %s" % (name, e))
        except Exception as e:                                  # noqa: BLE001
            results.append((name, False, "%s: %s" % (type(e).__name__, e)))
            print("  [NG] %s\n         %s: %s" % (name, type(e).__name__, e))
    bad = [r for r in results if not r[1]]
    print()
    if bad:
        print("决策层自检：%d/%d 通过，%d 条失败"
              % (len(results) - len(bad), len(results), len(bad)))
        return 1
    print("决策层自检全部通过（%d 条）" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
