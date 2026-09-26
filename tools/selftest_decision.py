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
        #   ③ `tap`（点按类：追击起跳 / 规避跳 / 喝药 / 喂宠）—— 它**直接调
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
        """某个键被 `tap` 点按的时刻列表（起跳 / 规避跳 / 喝药 / 喂宠走这条路）。"""
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
    s.auto_feed_pet = False
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
        early = [t for t in downs if t < delay_s - 0.2]
        check(len(early) == 1,
              "%s：delay 还没走完就重放了 %d 次（进度被打回起点）：%s"
              % (name, len(early), downs[:6]))
        late = [t for t in (downs + ups) if t >= delay_s - 0.2]
        check(late, "%s：delay 之后没有任何动作（delay 永远走不完）" % name)


def t_releaseall_clears_held_only():
    """RELEASEALL 之后上下文里记的按键要作废（固件那边已经松了）。"""
    s = fresh_settings(resetall_interval=1, output_seq=[
        {"type": "down", "key": "attack"}, {"type": "delay", "ms": 3000},
        {"type": "up", "key": "attack"}])
    h = Harness(s)
    h.clock0 = h.clock.t                      # 直接手动推进，不经 run()
    with h._patched():
        h.agent.tick(h.ws(True))              # 起跑：按住 attack
        h.clock.t += 2.0                      # 跨过一次定期 RELEASEALL
        h.agent.tick(h.ws(True))
        check(any(k == "RELEASEALL" for _, k, _ in h.log), "没触发定期 RELEASEALL")
        ctx = h.agent._output_ctx
        check(ctx is not None, "输出序列上下文被整块清掉了（进度应保留）")
        check(not ctx[3], "上下文里还记着按下的键：%s" % (ctx[3],))


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
        p._tick_feed_cd()
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

    # ④ 到达判据一：脚下就是**目标集合**（最贴近"到了哪块平台"）
    job = mk()
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0, py=0.0, here_sets=["甲平台"])
    check(o["jump"] and not o["done"], "还在别的平台上就说到达了：%s" % o)
    o = job.update(0.40, px=700.0, py=-90.0, here_sets=["乙平台"])
    check(o["done"] and not o["jump"], "脚下已是目标集合却没判到达：%s" % o)
    o = job.update(0.50, px=700.0)
    check(o["done"] and not o["jump"], "到达之后还在按键：%s" % o)

    # ⑤ 到达判据二（没有集合信息时的兜底）：y 越过绳的上端
    job = mk()
    job.update(0.0, px=700.0)
    job.update(0.25, px=700.0, py=-100.0)
    o = job.update(0.40, px=700.0, py=-120.0)      # 上端 -100 − 缓冲 14
    check(o["done"], "y 越过绳上端却没判到达（兜底判据）：%s" % o)

    # ⑥ 向下爬：按 ↓，y 越过下端算到
    job = mk(direction=-1, dst_set="丙平台")
    job.update(0.0, px=700.0)
    o = job.update(0.25, px=700.0)
    check(o["jump"] and o["dir"] == -1, "向下爬该按↓：%s" % o)
    o = job.update(0.40, px=700.0, py=200.0)
    check(o["done"], "y 越过绳下端却没判到达：%s" % o)

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
    for bad in ({"kind": "walk", "from": "右下", "to": "上下过渡平台"},
                {"kind": "climb", "from": "右下", "to": "上下过渡平台",
                 "ladder": "L9"},
                {"kind": "climb", "from": "右下", "to": "左上", "ladder": "L2"}):
        try:
            route.job_for_edge(t, z, bad)
            raise AssertionError("构造不该成功（会往反方向爬）：%s" % bad)
        except ValueError:
            pass          # 正是期望的：说不清就抛，**不许猜**


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


# ---------------------------------------------------------------- 入口

CHECKS = [
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
