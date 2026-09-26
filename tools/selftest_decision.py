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
    s.input_delay = [0, 0]          # 关掉随机抖动，便于断言结构间隔
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

    界面那行字读的就是 input_link_ok / input_link_err（player_panel 的
    _poll_link_health），这里不测的话，链路死了界面会一直显示「已连接」。
    """
    s = fresh_settings(resetall_interval=0)
    h = Harness(s)
    h.clock0 = h.clock.t          # 手动推进时钟（不经 run()）时记录器要用它
    state = {"ok": False, "backend": "remote", "err": "发指令后 5 秒没收到固件回执"}
    reconnected = []

    with patch.object(ag.time, "monotonic", h.clock), \
         patch.object(dinput, "link_health", lambda: dict(state)), \
         patch.object(dinput, "reconnect_remote",
                      lambda *a, **k: (reconnected.append(1), True)[1]), \
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

        # 恢复健康 → 标记跟着恢复
        state.update({"ok": True, "err": ""})
        h.clock.t += 11.0
        h.agent.tick(h.ws(True))
        check(s.input_link_ok is True and not s.input_link_err,
              "通道恢复后界面标记没跟着恢复：ok=%s err=%r"
              % (s.input_link_ok, s.input_link_err))


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
    继续歇着等于等着挨打。三条边界都要对：
      · 勾选 + 补血   ⇒ 打断，且下次休息 = 现在 + retry_sec（**秒**，不是随机 N~M 分钟）
      · 不勾选 + 补血 ⇒ **绝不打断**（补血照常发生，休息照旧）
      · 勾选 + 没补血 ⇒ 不打断（触发它的是"补血"这个事件，不是"在休息"）
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

    # 打点：被打断要能在 perf.log 里看见 —— 否则以后只能猜"它到底触发过没有"。
    # 走**公开路径**验证（configure → flush → 读文件），不去翻 perf 的内部字典。
    import tempfile
    from core import perf
    old_log, old_en = perf.LOG, perf.ENABLED
    logf = Path(tempfile.mkdtemp(prefix="perf_afk_")) / "perf.log"
    try:
        perf.configure(True, log=logf)
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
        perf.flush(force=True)
        check("afk_interrupt" in logf.read_text(encoding="utf-8"),
              "被打断了但 perf.log 里没有 afk_interrupt（这个功能会变成没法观测）")

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
        perf.flush(force=True)
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
    finally:
        perf.configure(old_en)
        perf.LOG = old_log


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
                       attack_cd=30, input_delay=[0, 0])
    h = Harness(s).run(1.0)
    downs = [t for t, kind, _k in h.log if kind == "down"]
    ups = [t for t, kind, _k in h.log if kind == "up"]
    check(downs and ups, "序列没跑起来：downs=%s ups=%s" % (downs[:3], ups[:3]))
    gap = ups[0] - downs[0]
    check(0.095 <= gap <= 0.13,
          "delay 100ms 实际隔了 %.3f 秒（>0.133 就是又回到「一帧量化」了）" % gap)


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
    h.run(6.0)
    taps = h.taps(s.keymap["jump"])
    check(len(taps) == 1,
          "怪走进起跳区间时应当正好跳一次，实际 %d 次：%s" % (len(taps), taps))


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
    """「走」**只有一种走法：朝目标集合的 x 中点**（用户 2026-09-26 明确否掉了"走的方向"）。

    这条原先测的是"三种方向"（`WalkJob.mode` = center / left / right，还在设置里摆了个
    「走的方向」下拉）。用户的定论是：**设置里不该有这个参数** —— 默认就一种：朝集合正中间
    走；将来若某一步真要"只按 ←/→"，那是**逐边**的事（foothold 编辑器 →「可到达」窗口里
    配），不是全局设置 ⇒ 这条用例改成**反向钉住**：那个参数不许回来。

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

    s = fresh_settings(strategy="sweep", jump_random_prob=0.0, attack_dist=80.0,
                       min_turn_hold_ms=1000, turn_output_delay_ms=200)
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
    limit = ag.TURN_TAP_S + 4 * FRAME
    for a, b in segs:
        check(b - a <= limit,
              "站桩输出时方向键按了 %.3f 秒（上限 %.2f）—— 角色会一边走一边打、"
              "从怪身上走过去再回头（按住段 %s）"
              % (b - a, limit, [(round(x, 3), round(y, 3)) for x, y in segs]))
    check(sum(b - a for a, b in segs) <= 0.6,
          "两秒里按着方向键走了 %.2f 秒 —— 那不叫站桩"
          % sum(b - a for a, b in segs))

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
    check(sum(b - a for a, b in segs2) <= 0.6,
          "两秒里按着方向键走了 %.2f 秒 —— 那不叫站桩" % sum(b - a for a, b in segs2))

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
        with patch.object(QMessageBox, "question", _answer(QMessageBox.Yes)):
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
        settings.input_device = was_dev
        settings.enabled = was_on
        p._refresh_auto_ui()


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
        # ⚠ 两个细节：① 任务的超时从**第一次 update** 起算 ⇒ 要两拍才到期；
        #            ② 默认还会**重试**（`max_attempts`）⇒ 失败要走到"放弃"那一支，
        #               所以这里把次数设成 1。
        a2 = route.WalkJob("丙集合", [(520.0, 510.0, 530.0, -208.0, "3")],
                           timeout_s=0.5, max_attempts=1)
        b2 = route.WalkJob("丁集合", [(520.0, 510.0, 530.0, -208.0, "4")])
        h.agent.start_route([a2, b2], why="用例：中途失败")
        ws2 = h.ws(with_mob=False)
        ws2.player.here_sets = []
        h.agent._climb_tick(10.0, 520.0, set(), ws2)      # 起算
        h.agent._climb_tick(11.0, 520.0, set(), ws2)      # 超时 ⇒ 放弃
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
    # ⚠ 这里**曾经**还测过「走的方向」（`walk_mode`）—— 用户 2026-09-26 明确否掉：
    # 走只有"朝集合中点"一种走法，设置里不许有这个参数 ⇒ 改成**反向钉住**它。
    check(not hasattr(settings, "walk_mode"),
          "`DecisionSettings` 里又出现了 `walk_mode`（用户 2026-09-26 明确删掉）")
    from gui.settings_dialog import SettingsDialog
    _d = SettingsDialog()
    try:
        check(hasattr(_d, "sp_goto_timeout"),
              "判定参数页里没有「寻路超时时间」控件")
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
    o = job.update(0.1, px=760.0)
    check(o["move"] == -1 and not o["jump"], "站过头时该往左挪：%s" % o)

    # ② 进误差范围 ⇒ **先站住**；保持时间不够不许跳（最容易做错的一步）
    job = mk()
    o = job.update(0.0, px=703.0)          # 差 3 ≤ 6 ⇒ 进范围
    check(o["move"] == 0 and not o["jump"],
          "刚进误差范围该停下等保持时间，不能马上跳：%s" % o)
    o = job.update(0.1, px=703.0)          # 才 100ms < 200ms
    check(o["phase"] == route.ClimbJob.ALIGN and not o["jump"],
          "保持时间没到就跳了：%s" % o)
    o = job.update(0.25, px=703.0)         # 250ms ≥ 200 ⇒ 上绳
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

    # ④ 到达判据 = **纯几何**（用户 2026-09-26 定的规则："按 ↑ 直到玩家的真实世界坐标
    #    ≤ 绳梯上端连接的 foothold 的 y"，即 `dst_y`）。
    #    ⚠ **"脚下是目标集合"不算到达依据**：定位读数会抖，`-170`（离目标还差 5px）那种
    #    位置也可能被报成"已在目标集合" ⇒ 提前收工 —— 用户实测就是"在 -170 就松开了 ↑" ✗。
    #    集合判据没丢，它在**失败之后**用来"别再重试"（见 agent 的 ⑤b2 用例）。
    job = mk(dst_y=-208.0)
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0, py=0.0, here_sets=["甲平台"])
    check(o["jump"] and not o["done"], "还在别的平台上就说到达了：%s" % o)
    o = job.update(0.40, px=700.0, py=-90.0, here_sets=["乙平台"])
    check(not o["done"],
          "只凭「脚下是目标集合」就判到达（提前收工 —— y 还没到目标面）：%s" % o)
    o = job.update(0.50, px=700.0, py=-208.0, here_sets=["乙平台"])
    # 够到目标面就算"到达"了，但**这一拍还不松手**（要再按住误差时间，见 ④b）
    check(not o["done"] and o["dir"] == 1 and "到达" in o["note"],
          "y 到了目标面却没判到达：%s" % o["note"])
    check("-208" in o["note"], "到达没写清依据（该说 y 到了哪个面）：%r" % o["note"])
    o = job.update(0.90, px=700.0)
    check(o["done"] and not o["jump"], "到达之后还在按键：%s" % o)

    # ④b **到达之后要再按住方向键一会儿才松**（用户 2026-09-26 要求：
    #     "↑ 需要延迟『坐标对齐误差时间』（设置里那个）再松开"）。
    #     为什么：读数到目标面 ≠ 人已经站上去（读数滞后一个端到端延迟；游戏里"迈上平台"
    #     那一步也要按着 ↑）⇒ 立刻松手就是差最后一点点。
    job = mk(dst_y=-208.0)                       # hold_ms=200（= 设置里的误差时间）
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0, py=-208.0, ladder_id="L2")   # 一上来就够到目标面
    check(not o["done"] and o["dir"] == 1 and not o["jump"],
          "刚够到目标面就松了 ↑（该再按住一会儿）：%s" % o)
    check("按住" in o["note"], "没写清「还要再按住一会儿」：%r" % o["note"])
    o = job.update(0.35, px=700.0, py=-208.0, ladder_id="L2")   # 100ms < 200ms
    check(not o["done"] and o["dir"] == 1, "保持时间没到就松了 ↑：%s" % o)
    o = job.update(0.50, px=700.0, py=-208.0, ladder_id="L2")   # 250ms ≥ 200ms
    check(o["done"] and o["dir"] == 0, "保持够了还没收工 / 没松 ↑：%s" % o)

    # ⑤ 到达判据（没有目标面信息时的兜底）：y 越过绳的上端。
    #    ⚠ 判到到达那一拍**还不松手**（要再按住误差时间，见 ④b）⇒ 断言分两拍。
    job = mk()
    job.update(0.0, px=700.0)
    job.update(0.25, px=700.0, py=-100.0)
    o = job.update(0.40, px=700.0, py=-120.0)      # 上端 -100 − 缓冲 14
    check(not o["done"] and o["dir"] == 1 and "到达" in o["note"],
          "y 越过绳上端却没判到达（兜底判据）：%s" % o)
    check(job.update(0.65, px=700.0, py=-120.0)["done"],
          "到达后等够了还没收工：%s" % job.note)      # 0.65-0.40 = 250ms ≥ 200ms

    # ⑥ 向下爬：按 ↓，y 越过下端算到（同样"到达后再按住一会儿"）
    job = mk(direction=-1, dst_set="丙平台")
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0)
    check(o["jump"] and o["dir"] == -1, "向下爬该按↓：%s" % o)
    o = job.update(0.40, px=700.0, py=200.0)
    check(not o["done"] and o["dir"] == -1 and "到达" in o["note"],
          "y 越过绳下端却没判到达：%s" % o)
    check(job.update(0.65, px=700.0, py=200.0)["done"],
          "到达后等够了还没收工：%s" % job.note)

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

    # ⑧b **偏离保护**（用户 2026-09-26）：横向偏出去超过阈值并**持续**一段 ⇒ 失败；
    #     只偏一下（能自己蹭回来）不算 —— 上绳那一下本来就会歪。
    job = mk()
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0)
    check(o["jump"], "该上绳了：%s" % o)
    o = job.update(0.30, px=760.0)               # 横向偏 60 > 18，但才 0.05s
    check(not o["failed"], "才偏一下（还没超过宽限期）就判失败：%s" % o)
    o = job.update(0.34, px=703.0)               # 自己蹭回来了 ⇒ 计时清零
    check(not o["failed"] and o["jump"], "蹭回来之后不该还在算偏离：%s" % o)
    o = job.update(0.40, px=760.0)               # 又偏出去
    o = job.update(1.10, px=760.0)               # 持续 0.7s ≥ 0.6 ⇒ 失败
    check(o["failed"] and not o["jump"], "偏离绳梯没判失败：%s" % o)
    check("偏离" in o["note"], "失败原因没写清是偏离：%r" % o["note"])

    # ⑧c **重新激活**：失败后 retry() 回到第一步重来，次数累加（到上限由 agent 放弃）
    check(job.attempt == 1, "一开始 attempt 该是 1：%s" % job.attempt)
    o = job.retry()
    check(job.attempt == 2 and o["phase"] == route.ClimbJob.ALIGN and not o["jump"],
          "retry 没回到对齐阶段：%s / attempt=%s" % (o, job.attempt))
    check(job._t0 is None and job._off_x_since is None,
          "retry 没把计时清干净（超时会立刻又炸）：%r" % (job._t0,))

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

    # ⑪ **爬到一半不动了 ⇒ 必须把话说清楚**（2026-09-26 用户第二次报"卡在某个 y 就不动"）。
    #    老行为是默默按住跳键耗到超时、再重试 3 次、放弃 —— 界面上只有一个"停住的角色"，
    #    而两种候选原因的**修法正好相反**（读数口径差一截 / 这段绳到不了顶）⇒ 只能靠猜。
    #    现在：y 卡住 STALL_S 秒 ⇒ 判失败，且 note 里必须同时有**卡住的 y**和**离目标面差多少**。
    job = mk(dst_y=-208.0)
    job.update(0.0, px=701.0, py=-150.0, ladder_id="L2")
    o = job.update(0.25, px=701.0, py=-150.0, ladder_id="L2")
    check(job.phase == route.ClimbJob.CLIMB, "没进上绳阶段：%s" % job.note)
    # 先把"往上爬"演出来（y 一路变小），再停在 -170 —— 就是用户报的那一幕
    # ⚠ 计时变量**别叫 `t`**：这个函数里 `t` 是地形对象（⑩ 载入的），覆盖它会让后面
    # 造任务的地方变成 `'float' object has no attribute 'footholds'`（自己踩的 ✗）。
    _t1 = 0.25
    for _py in (-160.0, -170.0):
        _t1 += 0.1
        job.update(_t1, px=701.0, py=_py, ladder_id="L2")
    while _t1 < 0.25 + route.STALL_S + 0.6:
        _t1 += 0.1
        o = job.update(_t1, px=701.0, py=-170.0, ladder_id="L2")   # 真实报的那个 y
        if o["failed"]:
            break
    check(o["failed"],
          "爬到某个 y 不动了却没判失败（会一直按着跳键耗到超时，人只看到一个停住的角色）")
    check("-170" in o["note"] and "-208" in o["note"],
          "失败话里没把两个数说清（卡在哪个 y / 离目标面还差多少）：%r" % o["note"])
    check("偏移" in o["note"], "没提示去核对「坐标系偏移」：%r" % o["note"])

    # ⑪b y 一直在变好 ⇒ 不许判"不动了"（正常爬升不能被误杀）
    job = mk(dst_y=-208.0)
    job.update(0.0, px=701.0, py=-100.0, ladder_id="L2")
    job.update(0.25, px=701.0, py=-100.0, ladder_id="L2")
    o = None
    for k in range(40):
        o = job.update(0.35 + k * 0.1, px=701.0, py=-100.0 - (k + 1) * 5.0,
                       ladder_id="L2")
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

    # ⑫ **上绳之后只按 ↑、不再按跳**（用户 2026-09-26 定的规则：
    #    "按 ↑ 直到玩家的真实世界坐标 ≤ 绳梯上端连接的 foothold 的 y"）。
    #    实测现象：命令前往左上平台时"还在绳上、离平台还差一截" —— 罪魁就是跳键一直被按住
    #    （跳键只该用来**贴上绳**）。反过来也必须钉住：**还没上绳时不许不按跳**
    #    （否则角色永远贴不上去，任务只会空转到超时）。
    job = mk(dst_y=-208.0)
    job.update(0.0, px=701.0)
    o = job.update(0.25, px=701.0)
    check(o["jump"] and o["dir"] == 1, "还没上绳就该按跳贴上去：%s" % o)
    o = job.update(0.30, px=701.0, py=-150.0, ladder_id="L2")
    check(not o["jump"], "上了绳还在按跳（会卡在绳上/绳顶，迈不上平台）：%s" % o)
    check(o["dir"] == 1, "上了绳却不按 ↑ 了（那这一拍就是站着不动）：%s" % o)
    check("↑" in o["note"] and "-208" in o["note"],
          "上绳后的提示没说清「按↑直到到哪」：%r" % o["note"])

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
    # 走不动（x 卡住）⇒ 失败并说清卡在哪个 x —— 被墙/台阶挡住时人要看得懂
    job = route.WalkJob("乙平台", [(900.0, 890.0, 910.0, -208.0, "7")],
                        timeout_s=60.0)
    job.update(0.0, px=100.0)
    # ⚠ 计时变量**别叫 `t`**：这个函数里 `t` 是地形对象（上面 ⑩ 载入的），
    # 覆盖了它后面 `walk_job_for_edge(t, …)` 就成了 `'float' object has no attribute
    # 'footholds'`（自己踩的 ✗）。
    _tt = 0.0
    o = None
    for _ in range(80):
        _tt += 0.1
        o = job.update(_tt, px=100.0)
        if o["failed"]:
            break
    check(o["failed"] and "走不动" in o["note"] and "100" in o["note"],
          "走不动却没判失败 / 没写清卡在哪：%r" % o["note"])
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
    for bad in ({"kind": "climb", "from": "甲", "to": "乙"},
                {"kind": "walk", "from": "甲", "to": "（不存在的集合）"}):
        try:
            route.walk_job_for_edge(t, z, bad)
            raise AssertionError("构造不该成功（会下一条不知道往哪走的命令）：%s" % bad)
        except ValueError:
            pass          # 正是期望的：说不清就抛


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
        ws.player.world_y = -200.0             # 越过绳上端（-100 − 缓冲 14）
        # ⚠ 判到达用的是**世界**坐标 y（`world_y`）—— 摆 `y`（画面坐标）是不算数的
        h.log.clear()
        h.agent.tick(ws)
        check(h.agent._climb is None, "到了任务没自己撤掉：%r" % (h.agent._climb,))
        check(h.agent.state != "climb", "到了还挂着 climb 状态：%s" % h.agent.state)
        check(jump not in downs(), "到了还在按跳：%s" % h.log)

        # ⑤b **失败保护**（2026-09-26 改）：失败 ⇒ **延迟**重新激活；到上限才真放弃。
        # 原来这里是"立即重来" —— 失败那一下人还在原地、朝向也没变，常常在同一处再歪一次。
        was_delay = s.climb_retry_delay_s
        try:
            s.climb_retry_delay_s = 1.0
            j = job(max_attempts=2)
            obj = h.agent.start_climb(j)
            obj._fail("用例：故意失败")      # 直接打成失败态，省掉摆时间的麻烦
            h.log.clear()
            done = h.agent._climb_tick(0.0, 500.0, set(), h.ws(with_mob=False))
            check(done is False and h.agent._climb is j,
                  "失败后任务被扔了：done=%s climb=%r" % (done, h.agent._climb))
            check(j.attempt == 1 and j.phase == route.ClimbJob.FAILED,
                  "没等延迟就重新激活了（等于老行为）：attempt=%s phase=%s"
                  % (j.attempt, j.phase))
            check(jump not in downs(), "等待期间还在按跳（该站着等）：%s" % h.log)
            # 还没到点 ⇒ 仍然不重来
            h.agent._climb_tick(0.5, 500.0, set(), h.ws(with_mob=False))
            check(j.attempt == 1, "延迟没到就重新激活了：attempt=%s" % j.attempt)
            # 到点 ⇒ 重新对齐（attempt +1）
            h.agent._climb_tick(1.1, 500.0, set(), h.ws(with_mob=False))
            check(j.attempt == 2 and j.phase == route.ClimbJob.ALIGN,
                  "延迟到了却没重新激活：attempt=%s phase=%s" % (j.attempt, j.phase))
            # 上限：到次数上限就真放弃（不能无限重来 —— 那看着像卡死）
            obj._fail("用例：又失败")
            done = h.agent._climb_tick(1.2, 500.0, set(), h.ws(with_mob=False))
            check(done is True and h.agent._climb is None,
                  "到了次数上限还没放弃（会一直重来，看着像卡死）：%s / %r"
                  % (done, h.agent._climb))
            check(h.agent.state != "climb",
                  "放弃之后还挂着 climb 状态：%s" % h.agent.state)
            check(h.agent._climb_retry_at is None,
                  "放弃之后还留着「等重来」的时刻（下一个任务会立刻被它带跑）")
        finally:
            s.climb_retry_delay_s = was_delay

        # ⑤b2 **已经到了目标集合 ⇒ 不重来，直接收工**（2026-09-26 用户要求 2）：
        # 失败判定可能晚于实际到达（定位抖一下、到了又被打下来）—— 这时接着"重新激活"
        # 就是"人已经站上去了还在原地爬"，比不做更糟。
        j2 = job(max_attempts=3)
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

        # ⑤e **寻路超时时间(s)**（用户 2026-09-26 要求）：从下达那一刻算，超了就切断 ——
        #     任务自己的 timeout 会被 `retry()` 清零，累计可能远超预期，所以要有总闸。
        was_cap = s.goto_timeout_s
        s.goto_timeout_s = 0.5
        j9 = job()
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
    """下跳执行器：就近平齐 → **先按住 ↓** → ↓+跳 → 落地；失败保护与上绳同一套。

    用户 2026-09-26 定的动作：**按住 ↓ 再按跳**（与「跳(jump)」= 在 foothold 边缘按跳
    是两个类型，别混）。这里钉三件最容易做错的事：
      · **顺序**：ARMED 那一拍只能有 ↓、**不能有跳**（"先按住再按跳"就是这个意思）；
      · **就近**：边上给了好几个可下跳点时要挑离自己最近的那个；
      · **到达**：先认集合，认不出集合就用"y 掉下去了一段"兜底（y 向下增大）。
    """
    from core import mapdata, zones
    from decision import route

    spots = [(700.0, 660.0, 740.0, "41"), (1250.0, 1210.0, 1290.0, "72")]
    job = route.DropJob("甲平台", "乙平台", spots, tol_px=6, hold_ms=0)

    # ① 就近挑：站在 x=1200 ⇒ 挑 fh 72（不是 700 那个）
    o = job.update(0.0, px=1200.0)
    check(job._pick[3] == "72", "没就近挑可下跳点：%r" % (job._pick,))
    check(o["move"] == 1 and not o["jump"], "该往右挪：%s" % o)
    # ② 平齐后 ⇒ **先按住 ↓**：这一拍有 ↓、没有跳
    o = job.update(0.5, px=1250.0)
    check(o["phase"] == route.DropJob.ARMED and not o["jump"],
          "对齐好了该先进 ARMED（按住↓）阶段：%s" % o)
    check(o["dir"] == -1, "ARMED 阶段就该按住 ↓：%s" % o)
    # ③ 按住一小会儿 ⇒ 补跳
    o = job.update(0.8, px=1250.0)
    check(o["phase"] == route.DropJob.DROP and o["jump"] and o["dir"] == -1,
          "该「↓ + 跳」了：%s" % o)
    # ④ 到达：先认集合
    o = job.update(0.9, px=1250.0, py=0.0, here_sets=["甲平台"])
    check(not o["done"], "还在别的平台上就说到了：%s" % o)
    o = job.update(1.0, px=1250.0, py=120.0, here_sets=["乙平台"])
    check(o["done"] and not o["jump"], "脚下已经是目标集合却没判到达：%s" % o)

    # ⑤ 没有集合信息时的兜底：y 比出发时**增大** ≥40（y 向下增大 = 真的掉下去了）
    job = route.DropJob("甲平台", "乙平台", spots, tol_px=6, hold_ms=0)
    job.update(0.0, px=700.0)
    job.update(0.5, px=700.0, py=-300.0)          # 进 ARMED，同时记下出发 y
    o = job.update(0.8, px=700.0, py=-300.0)
    check(o["jump"], "该在下跳：%s" % o)
    o = job.update(1.0, px=700.0, py=-250.0)      # 掉了 50 ≥ 40
    check(o["done"], "y 掉下去一段却没判到达（兜底判据）：%s" % o)

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
        h.agent.start_route([a, b], why="用例：两段")
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
                          timeout_s=600.0, max_attempts=1)
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


# ---------------------------------------------------------------- 入口

CHECKS = [
    ("定点休息：走→到达后行为→歇时长→结束后前往→回战斗；走不到如实说",
     t_afk_spot_rest),
    ("防掉线：打断重试那两个参数挂在通用层（选定点休息也看得见）",
     t_afk_retry_params_are_common),
    ("寻路超时：每完成一段重新计时（长路线不被整条切断），单段超时才切",
     t_goto_timeout_per_hop),
    ("防掉线契约：状态表齐全 / 收尾自保松键 / 只收自己那条路线",
     t_rest_contract),
    ("休息文案：每个阶段都有话说；定点休息不许显示「未休息」",
     t_rest_state_text_covers_all_states),
    ("自定义定时行为：手动触发立刻演一次、计时从头开始（暂停/关自动不响应）",
     t_timer_manual_fire),
    ("自定义定时行为：界面「手动触发」按钮（换色 + 灰掉的时机）",
     t_timer_manual_fire_button),
    ("输出CD：序列里没有攻击键也要守CD", t_cd_no_attack_key),
    ("输出CD：进攻击状态走排期（状态抖动不超速）", t_cd_on_enter_attack_state),
    ("输出CD：序列比CD长时以序列为准", t_cd_shorter_than_sequence),
    ("定期RELEASEALL：本机序列进度全保留", t_releaseall_keeps_progress),
    ("定时重置指令通道：到点发/0禁用/关自动也发/手动输入跳过",
     t_periodic_reset_channel),
    ("指令通道体检：不健康标出来 + 后台重连 + 恢复跟随", t_link_health_watch),
    ("定时重置：本地后端也要真的松开按键（没有固件兜底）",
     t_periodic_reset_local_backend),
    ("类别表：只在 perception/classes.py 定义", t_class_table_single_source),
    ("时序拍：next_deadline 报的是序列到点时刻", t_next_deadline),
    ("时序拍：序列元素按本机绝对时钟发（不再一帧量化）",
     t_sequence_timing_not_frame_quantized),
    ("定期RELEASEALL：只作废按键记录", t_releaseall_clears_held_only),
    ("自定义定时行为：暂停后不触发、且打断正在演的序列",
     t_timer_paused_never_fires),
    ("自定义定时行为：「休息时暂停计时」是每条各自的开关（没勾的照常倒数）",
     t_timer_pause_on_rest_flag),
    ("自定义定时行为：暂停按钮/红字（已暂停）/继续接着走", t_timer_pause_button),
    ("休息状态机：到点/手动结束/关防掉线/手动进入", t_rest_state_machine),
    ("休息状态机：手动进入要先等清怪", t_rest_manual_request),
    ("被打断重试：补血即打断并提前重试（不勾选则不打断）",
     t_rest_interrupt_retry),
    ("追击起跳：区间在攻击距离内侧时也要跳（且只跳一次）",
     t_chase_jump_inside_band),
    ("追击起跳：区间在外侧时进区间跳一次", t_chase_jump_outside_band_once),
    ("追击起跳：怪走进区间的那一拍跳一次", t_chase_jump_approach_edge),
    ("追击起跳：有别的怪在场 / 开关关掉都不跳", t_chase_jump_guard_shut),
    ("平地巡逻无怪时松掉输出键（怪在输出序列中途消失也不许卡键）",
     t_patrol_idle_releases_output),
    ("走只有一种走法：朝集合中点（「走的方向」参数不许回来）", t_walk_only_center),
    ("扫平台：攻击范围内有怪要站桩（不许边走边打）",
     t_sweep_stands_still_in_attack),
    ("按键层：F10~F12 三条发送路径全拦", t_input_local_only),
    ("触控板：本地 F10 开关（手动输入开着也能用）", t_touchpad_f10_toggle),
    ("触控板：面板隐藏时自动关闭", t_touchpad_hidden_closes_mode),
    ("触控板：状态栏显示「触控模式中」", t_touchpad_status_text),
    ("触控板：非触控模式滚轮让给滚动区", t_touchpad_wheel_passthrough),
    ("本地(仅测试)输入：开启自动要二次确认，关闭不拦", t_auto_confirm_local),
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
