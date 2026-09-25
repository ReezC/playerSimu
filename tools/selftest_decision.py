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
        self.log = []                       # [(相对秒, "down"/"up"/"RELEASEALL", 键)]
        self.agent = CombatAgent(settings)

    # ---- 世界状态 ----
    def ws(self, with_mob=True):
        w = WorldState(width=1920, height=1080)
        w.player = Player(x=500.0, y=500.0, w=60.0, h=100.0, bottom=600.0,
                          hp=1.0, mp=1.0, found=True)
        w.mobs = ([Mob(id=1, x=560.0, y=500.0, w=40.0, h=40.0, conf=0.9, reachable=True)]
                  if with_mob else [])
        return w

    # ---- 记录器 ----
    def _rec(self, kind):
        def f(*a):
            self.log.append((round(self.clock.t - self.clock0, 3), kind,
                             a[0] if a else "-"))
        return f

    @contextmanager
    def _patched(self):
        # **两条按键路径都要挡**：agent 自己发序列用的 key_down/key_up 在 ag 命名空间，
        # 而 KeyState（移动键）走的是 decision.input 里的那两个 —— 只挡前者的话，
        # 走 KeyState 的用例会**真的往本机发键**（往当前窗口打字，实测踩过）。
        with patch.object(ag.time, "monotonic", self.clock), \
             patch.object(ag, "key_down", self._rec("down")), \
             patch.object(ag, "key_up", self._rec("up")), \
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
    ("按键层：F10~F12 三条发送路径全拦", t_input_local_only),
    ("触控板：本地 F10 开关（手动输入开着也能用）", t_touchpad_f10_toggle),
    ("触控板：面板隐藏时自动关闭", t_touchpad_hidden_closes_mode),
    ("触控板：状态栏显示「触控模式中」", t_touchpad_status_text),
    ("触控板：非触控模式滚轮让给滚动区", t_touchpad_wheel_passthrough),
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
