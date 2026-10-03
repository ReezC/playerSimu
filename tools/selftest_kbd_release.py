"""自检：**"把键松干净"那条路**（用户 2026-10-01 ✓）。

现场有两件事，必须分开看，别混成一个问题：

  · **"停了自动，它还在自己按输出键、左右走"** —— 那是**已经发出去**的命令还排在后
    面（固件逐条执行，`TAP`/`FIX`/`RND` 内部还会原地等）⇒ 收尾靠**停自动时补一条
    `RELEASEALL`**（就是本文件守着的那条路 ✓）。
  · 曾经试过从**源头限流**（按键背压：同一时刻只让少量指令在飞）来掐掉那个队列 ——
    **连着三次把按键通道搞到不可用**（特别卡 / 输入全无效 / 方向键发不出去 ✗）
    ⇒ **整个撤掉了**（2026-10-01，见 `decision/agent.py` 里那段说明）。
    ⚠ 要重做的话，**先量 `perf.log` 里的 `kbd_rtt_ms`**（B 发指令 → A 固件回执的真实
      往返和抖动），拿数据说话 —— 前三次全是"照代码推断"拍出来的，都错了 ✗。

跑法：
    python -m tools.selftest_kbd_release
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认是 **GBK**：用例名里带 ⇒ / ✗ 这类字符时 `print` 会抛
# `UnicodeEncodeError` —— 那和被测的东西无关，是终端的编码问题 ⇒ 切成 UTF-8 ✓。
try:
    sys.stdout.reconfigure(encoding="utf-8")            # type: ignore[attr-defined]
except Exception:                                       # noqa: BLE001
    pass


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ══════════════════════ 用例 ══════════════════════

def t_stop_automation_releases_keys():
    """**停自动 ⇒ 真的发一条 RELEASEALL**：固件那边这才第一次被告知"松手" ✓。

    为什么钉它：老路子是"不再发新命令"就完事 —— 可**已经进管道的照走不误** ⇒ 现场就是
    "关了自动还在自己按"。这条收尾一旦被谁删掉，那个现象会**原样回来而且没有任何报错**
    （所以必须钉住 ✓）。
    """
    src = (ROOT / "decision" / "agent.py").read_text(encoding="utf-8")
    check("if not s.enabled and self._was_enabled:" in src,
          "找不到「关闭自动」的下降沿分支（被删了？）⇒ 会重现「停了还在自己按」✗")
    check("self._release_keys_now()" in src,
          "下降沿没有调用抽出后的松键入口 ⇒ 本地/远程两种后端会漏一种 ✗")
    # ⚠ 停自动**不许**把键重按回去（`_reassert_ctx`）——那等于停了又按起来 ✗
    _i = src.find("if not s.enabled and self._was_enabled:")
    _seg = src[_i:_i + 1600]
    check("_release_keys_now" in _seg, "下降沿分支里看不到松键调用 ✗")
    check("_reassert_ctx" not in _seg,
          "停自动的分支里出现了 `_reassert_ctx` ⇒ 松完又按回去 ✗（那等于没停）")


def t_release_keys_covers_both_backends():
    """松键那唯一一处写法必须**同时认两种后端**（2026-09-26 现场踩过 ✗）。

      · 远程/串口：固件一条 `RELEASEALL` 全清 ⇒ 本机**只**清记录（再补发 up 是错的）；
      · 本地(仅测试)：没有固件，`release_all_remote()` 是空操作 ⇒ 必须逐键发 up，
        只清记录的话那个键在本机就**永远按着** ✗（实测只看得见 down、没有 up）。
    """
    src = (ROOT / "decision" / "agent.py").read_text(encoding="utf-8")
    _i = src.find("def _release_keys_now(self):")
    check(_i >= 0, "找不到 `_release_keys_now`（松键的唯一写法被拆散了？）✗")
    _seg = src[_i:_i + 2200]
    check('== "local"' in _seg, "没按后端分岔 —— 两种后端会被当成一种 ✗")
    check("keys.release_all()" in _seg, "本地后端那条路没了 ⇒ 键在本机永远按着 ✗")
    check("release_all_remote()" in _seg and "keys.clear()" in _seg,
          "远程后端那条路不全（固件清完之后本机记录也得同步清 ✓）✗")


def t_backpressure_is_gone():
    """⭐ **限流（背压）已经被整个撤掉** —— 这条是"别再悄悄加回来"的闸门 ✗。

    它连着三次把按键通道搞到不可用（特别卡 / 输入全无效 / 方向键发不出去）。撤掉是
    用户 2026-10-01 定的 ✓。谁要再加，得先拿 `perf.log` 的 `kbd_rtt_ms` 说话 ✗。
    """
    import inspect

    from remote_kbd import kbd_client as kc

    check(not hasattr(kc.KbdClient, "_backpressure"),
          "`_backpressure` 又回来了？要重做请先量 `kbd_rtt_ms` 再谈 ✗")
    check(not hasattr(kc.KbdClient, "MAX_INFLIGHT"),
          "`MAX_INFLIGHT` 又回来了 ✗")
    _src = inspect.getsource(kc.KbdClient.send)
    check("_backpressure" not in _src and "MAX_INFLIGHT" not in _src,
          "`send()` 里又在做限流了 ✗")
    # 参数也别留着（留着就会有人以为它有用 ✗）
    from decision.agent import DecisionSettings
    check(not hasattr(DecisionSettings(), "kbd_max_inflight"),
          "设置对象上还留着 `kbd_max_inflight` ⇒ 界面上那个旋钮的残骸 ✗")


def t_mouse_move_is_instrumented():
    """⭐ B 侧：每条 `MOVE` 都要留下「发得匀不匀」的可量证据（用户 2026-10-03 ✓ 现场：
    「看起来现象是在 A 机上一段一段一顿一顿的指令汇报很离散，B 机卡不卡其实没关系」）。

    为什么必须有这条：鼠标这条链以前**没有专用计数** ✗（`MOVE` 和键盘混在 `send` 里
    ✓）⇒「实际命令率 / 间隔 / 每步几像素」全都量不出来、只能照代码猜 ✗ —— 而"照代码
    猜"在这个项目上已经被坑过三次（见本文件开头那段限流史 ✓）。所以本轮**只加打点、
    先不改代码**：先拿数定案"是 B **发**得就不匀"还是"A **转**得不匀" ✓。

    钉三件：
      ① 真发出去的仍是 `MOVE dx dy`（位移不许在链路层被改 ✗）；
      ② `mouse_send` 计数跟着条数走；
      ③ `move_gap_ms` / `move_px` 都真的落了样本 —— 前者判「一撮一撮」、后者判
         「量化台阶（中位 1~2 px）」✓，少任何一个都定不了案 ✓。
    """
    import tempfile
    import time as _time
    from core import perf
    from decision import input as dinput

    class _R:
        def __init__(self):
            self.lines = []

        def send(self, line):
            self.lines.append(line)
            return True

    logf = Path(tempfile.mkdtemp(prefix="perf_mouse_")) / "perf.log"
    old_en, old_log, old_r = perf.ENABLED, perf.LOG, dinput._remote
    try:
        perf.configure(True, log=logf)
        r = _R()
        dinput._remote = r
        dinput._last_move_t = None
        for dx, dy in ((3, 0), (1, 1), (0, -2)):
            dinput.mouse_move(dx, dy)
            _time.sleep(0.004)
        perf.flush(force=True)
        got = logf.read_text(encoding="utf-8")
    finally:
        dinput._remote = old_r
        dinput._last_move_t = None
        perf.configure(old_en)
        perf.LOG = old_log

    check(r.lines == ["MOVE 3 0", "MOVE 1 1", "MOVE 0 -2"],
          "鼠标没按 `MOVE dx dy` 发（位移被改了？）：%r" % (r.lines,))
    for key in ("mouse_send", "move_gap_ms", "move_px"):
        check([ln for ln in got.splitlines() if ln.startswith(key)],
              "`perf` 里没有 `%s` ⇒ 这件事又回到「只能猜」✗：%r" % (key, got[-200:]))


def t_move_stats_reports_burstiness():
    """⭐ A 侧 relay：`MOVE 节拍` 那行必须能把「一段一段」量出来（同一现场 ✓）。

    现场要判的就是「**成批** + 有**缝**」：喂两批（一批 2 条、一批 3 条）+ 中间隔
    30 ms ⇒ 报出来必须是「批内最大 3、间隔最大 ≈30 ms」✗（这正是「A 机上一段一段
    一顿一顿」的形状 ✓；理想是「批内 ≈1、间隔绕节拍小抖动」✓）。

    顺带钉三个"一写错数就变假"的地方：
      · **半行**（TCP 把一条 `MOVE …\\n` 切两半）不许算成两条命令、也不许整条丢掉 ✗
        ⇒ 半行要留到下一批 ✓；
      · 键盘指令**不许**混进鼠标率 ✗（混进来率直接算歪 ✓）；
      · **到点才报**（没到 `report_sec` 就返回 None ✓）⇒ 不许每条都写一行把日志刷爆 ✗。
    """
    from remote_kbd.relay import MoveStats

    st = MoveStats(report_sec=0.0)
    t = 100.0
    st.feed(b"MOVE 3 0\nMOVE 2 1\n", t)             # 一批 2 条
    t += 0.030
    st.feed(b"PRESS A\nMOVE 1 0\n", t)              # 键盘混杂：只该算 1 条 MOVE
    t += 0.002
    st.feed(b"MOVE 1 2\nMOVE 0 1\nMOVE 4 0\n", t)   # 一批 3 条
    line = st.maybe_report(now=t + 0.001)
    check(line and line.startswith("MOVE 节拍 "), "没报出 `MOVE 节拍` 那行：%r" % (line,))
    check("n=6" in line, "MOVE 条数算错了（键盘混进来了？）：%s" % line)
    check("批内中位2最大3" in line, "批内条数没量出来（「成批」就靠它 ✗）：%s" % line)
    check("最大30.0ms" in line, "间隔最大值没量出来（「有缝」就靠它 ✗）：%s" % line)
    check("步长中位" in line and "最大" in line, "步长（量化台阶）没报：%s" % line)

    # 半行（被 TCP 切两半）必须留到下一批，且**只算一条、步长要拼回 12+5=17**
    # ⚠ 光断言 `n=1` 抓不住「不留半行」那种写法 ✗（切半的那半行也被当一条 ⇒ 数正好也是 1 ✓
    #   反向验证实测：把 `_carry` 去掉，`n=1` 照样成立 ⇒ 假绿 ✗）⇒ **必须看步长** ✓：
    #   不留半行时 `MOVE 12` 缺第三个字段 ⇒ 步长记成 0 ✗（而 0 会把"量化台阶"那一眼糊掉 ✓）。
    st2 = MoveStats(report_sec=0.0)
    st2.feed(b"MOVE 12", 1.0)
    st2.feed(b" 5\n", 1.01)
    line2 = st2.maybe_report(now=1.02)
    check("n=1" in line2 and "批=1" in line2,
          "半行被算成两条命令 / 或者整条丢了（TCP 切包是常态 ✗）：%s" % line2)
    check("步长中位17" in line2,
          "半行没拼回去（步长该是 12+5=17；记成 0 说明把残行也当命令了 ✗）：%s" % line2)

    # 没到点不许报（不然 5 秒的窗口会被刷成一堆行 ✗）
    st3 = MoveStats(report_sec=5.0)
    st3.feed(b"MOVE 1 0\n", 10.0)
    check(st3.maybe_report(now=10.5) is None, "没到 `report_sec` 就报了一行 ✗")
    check(st3.maybe_report(now=15.5) is not None, "到点了却没报 ✗")


def t_relay_trace_has_ms_timestamp():
    """A 侧 `relay_trace.log` 的时间戳必须到**毫秒**（2026-10-03 ✓）。

    为什么：要判「离散」就得看 **20~30 ms** 的缝 ✗ —— 秒级时间戳（原来就是
    `time.strftime("%H:%M:%S")` ✗）下那些缝**根本看不见** ⇒ "拿日志也答不出来" ✓。
    """
    src = (ROOT / "remote_kbd" / "relay.py").read_text(encoding="utf-8")
    check("def _ts()" in src, "找不到毫秒时间戳入口 `_ts()`（被删了？）⇒ 日志又只剩秒 ✗")
    check("%03d" in src, "`_ts()` 里没算毫秒那三位 ⇒ 还是秒级 ✗")
    check("(_ts(), text)" in src, "`trace()` 没走 `_ts()`（又写回秒级 ✗）")
    check("% (time.strftime(" not in src,
          "`trace()` 里还留着老的秒级写法（两处并存 ⇒ 有一条永远不带毫秒 ✗）")


def t_a_logs_are_forwarded_to_b():
    """⭐⭐ **A 机的日志要回传到 B 机**（用户 2026-10-03 ✓ 原话："你能让 A 机的log都往B机发吗？
    这样我就不用老切换了"）。

    做法：不新开端口 —— relay 本来就有到 B 机的**回程**（固件回执走那条 ✓）⇒ A 侧
    `trace()` 顺手挂上去（`#LOG <毫秒时间戳> <原文>` ✓），B 侧 `KbdClient._take_a_logs`
    挑出来落进 `remote_kbd/A_relay_trace.log` ✓（名字带 `A_`，免得和 A 机自己那份混 ✗）。

    钉五件（每件都对应一个"会静默坏掉"的坑 ✓）：
      ① A 侧真的推、且**带毫秒**（要判 20~30 ms 的缝，秒级不行 ✗）；
      ② **回传与本地文件无关** ✗ —— `_trace_fh` 打不开（目录没了/磁盘满）时也要推 ✓
         （探针实测过：原来 `trace()` 先判文件、`None` 就 return ⇒ 回传一起死 ✗）；
      ③ B 侧**只挑 `#LOG ` 行**，剩下的（`DONE`/`ERR`）一个字不改 ⇒ **回执计数不受影响** ✓
         （日志里出现 DONE 字眼也不许被算成回执 ✗ —— 那会把 `kbd_rtt_ms` 和"死链判据"骗到 ✓）；
      ④ **半行规则**：日志半行留到下一批 ✓，**普通半行（`DON`）绝不许扣住** ✗（扣住 = 拖慢
         回执计数 = 把 RTT 量歪 ✓）；
      ⑤ 落盘内容里有抬头行 + 原文（超限砍半 ✓）。
    """
    import tempfile
    from remote_kbd import relay as relay_mod
    from remote_kbd import kbd_client as kc

    class _Sink:
        def __init__(self):
            self.buf = b""

        def sendall(self, data):
            self.buf += data

    # ① / ②：A 侧推 ——**故意把文件置成 None**（这个状态本来会让回传一起死 ✗）
    old_fh, old_sink = relay_mod._trace_fh, relay_mod._LOG_SINK
    sink = _Sink()
    try:
        relay_mod._trace_fh = None
        relay_mod._LOG_SINK = sink
        relay_mod.trace("MOVE 节拍 5s n=42")
    finally:
        relay_mod._trace_fh = old_fh
        relay_mod._LOG_SINK = old_sink
    check(sink.buf.startswith(b"#LOG "), "A 侧没把日志推给回程：%r" % (sink.buf,))
    _ts_part = sink.buf.split(b" ", 2)[1]           # `#LOG <时间戳> <正文>` ✓
    check(_ts_part.count(b".") == 1 and len(_ts_part.split(b".")[1]) == 3,
          "回传行没带**毫秒**时间戳（判不了 20~30 ms 的缝 ✗）：%r" % (sink.buf,))
    src = (ROOT / "remote_kbd" / "relay.py").read_text(encoding="utf-8")
    _tr = src[src.index("def trace(text):"):]
    check("_push_to_client(text)" in _tr[:600],
          "`trace()` 里没看到回传调用（被删了？）⇒ A 机日志又只能在 A 机看 ✗")
    check(_tr.index("_push_to_client(text)") < _tr.index("if _trace_fh is None:"),
          "回传被写在「文件打不开就 return」**之后** ✗ ⇒ 那种情况下人还是看不到日志 ✓")

    # ③④⑤：B 侧挑走 + 落盘 + 计数不受影响
    tmp = Path(tempfile.mkdtemp(prefix="alog_")) / "A_relay_trace.log"
    old_path = kc.A_LOG_PATH
    try:
        kc.A_LOG_PATH = tmp
        c = kc.KbdClient.__new__(kc.KbdClient)          # 不连网 ✓
        c._log_carry, c.a_log_lines, c._a_log_banner = b"", 0, False
        rest = c._take_a_logs(b"#LOG 12:00:00.123 A\n"
                              b"#LOG 12:00:00.130 tcp->serial MOVE 3 0|DONE|\n"
                              b"DONE\nERR\n")
        check(rest.count(b"DONE") + rest.count(b"ERR") == 2,
              "回执计数被日志行污染了（日志里有个 DONE 字眼 ✗）：%r" % (rest,))
        check(c.a_log_lines == 2, "没挑出两条日志：%r" % (c.a_log_lines,))
        got = tmp.read_text(encoding="utf-8")
        check("A 机 relay 日志（转发" in got and "12:00:00.130 tcp->serial" in got,
              "落盘内容不对（缺抬头行或原文）：%r" % (got,))

        c2 = kc.KbdClient.__new__(kc.KbdClient)
        c2._log_carry, c2.a_log_lines, c2._a_log_banner = b"", 0, True
        check(c2._take_a_logs(b"#LOG 12:00:00.1") == b"",
              "日志半行没留到下一批：%r" % (c2._take_a_logs,))
        r2 = c2._take_a_logs(b"23 MOVE 1 0\nDON")
        check(c2.a_log_lines == 1, "半行拼回来后条数不对：%r" % (c2.a_log_lines,))
        check(b"DON" in r2,
              "普通半行（DON）被扣住了 ✗ ⇒ 回执计数要等下一批才动 = 把 RTT 量歪 ✓：%r" % (r2,))
    finally:
        kc.A_LOG_PATH = old_path


def t_pad_move_is_coalesced():
    """⭐⭐ 触控板位移**累积 + 8 ms 合并成一条**（用户 2026-10-03 ✓ 现场："卡卡的" +
    "A 机上一段一段一顿一顿"）。

    病：原来**每个事件直接发一条** ✗（触控板 60~125 Hz、快速滑还有突发）⇒ 每条都要走
    ≈11 ms 往返（`perf.log` 的 `kbd_rtt_ms` ✓）⇒ 事件比往返快 ⇒ 必积压 ✓；且发送在
    **GUI 主线程**、排在 `draw_ms`（中位 5 / p99 15~22 ms）后面 ⇒ 一帧攒的位移**同一拍
    连发几条** ✗ = "一段一段"的形状来源 ✓（链路无缓冲：relay 原样转发 + 固件一批做完 ✓）。

    钉四件：
      ① `_on_pad_moved` **不再直接发** ✗（只累积 + 起节拍 ✓）；
      ② `_flush_pad` 把累积量**合并成一条**发出去 ✓；
      ③ **位移守恒**：整数发走、**小数留着**（慢速滑不会丢位移 ✓）；
      ④ 没整像素时**停表**（不许空转 ✓）。
    """
    from types import SimpleNamespace

    from gui.player_panel import PlayerPanel
    from decision import input as dinput

    sent = []
    _old = dinput.mouse_move
    dinput.mouse_move = lambda dx, dy: sent.append((dx, dy))
    try:
        class _T:
            def __init__(self):
                self.stopped = 0

            def stop(self):
                self.stopped += 1

        # ③ 整数发走、小数留着（3.4px ⇒ 发 3、留 0.4 ✓）
        fake = SimpleNamespace(_pad_acc=[3.4, -1.7], _pad_idle=0, _pad_timer=_T())
        PlayerPanel._flush_pad(fake)
        check(sent == [(3, -1)], "累积位移没有合并成一条发出去：%r" % (sent,))
        check(abs(fake._pad_acc[0] - 0.4) < 1e-9 and abs(fake._pad_acc[1] + 0.7) < 1e-9,
              "小数部分被丢了（慢速滑动会丢位移 ✗）：%r" % (fake._pad_acc,))

        # ④ 没整像素 ⇒ 表停掉，不许空转
        fake2 = SimpleNamespace(_pad_acc=[0.3, 0.4], _pad_idle=7, _pad_timer=_T())
        before = len(sent)
        PlayerPanel._flush_pad(fake2)
        check(len(sent) == before, "不足 1px 也发了命令（空放 ✗）：%r" % (sent[before:],))
        check(fake2._pad_timer.stopped == 1, "静下来没停表（会一直空转 ✗）")

        # ① 每事件直发那条路必须**没了**（源码级，防止改回去 ✗）
        src = (ROOT / "gui" / "player_panel.py").read_text(encoding="utf-8")
        _body = src.split("def _on_pad_moved", 1)[1].split("def _flush_pad", 1)[0]
        check("dinput.mouse_move(" not in _body,
              "`_on_pad_moved` 又直接发命令了（每个事件一条 ⇒ 积压/一顿一顿回来 ✗）")
        check("_flush_pad" in _body and "QTimer" in _body,
              "没有起 8 ms 合并节拍（那就等于没改 ✗）")
    finally:
        dinput.mouse_move = _old


TESTS = (
    ("停自动要真的松干净（下降沿 + 不许把键按回去）", t_stop_automation_releases_keys),
    ("松键那条唯一写法同时认两种后端", t_release_keys_covers_both_backends),
    ("限流（背压）已整个撤掉、别悄悄加回来", t_backpressure_is_gone),
    ("⭐ 鼠标链打点：`mouse_send` / `move_gap_ms` / `move_px`（先量后改 ✓）",
     t_mouse_move_is_instrumented),
    ("⭐ A 侧 `MOVE 节拍`：批内条数 + 间隔 + 步长（半行 / 键盘混杂都要分清）",
     t_move_stats_reports_burstiness),
    ("A 侧 relay 日志时间戳到毫秒（秒级看不出 20~30 ms 的缝）",
     t_relay_trace_has_ms_timestamp),
    ("⭐⭐ A 机日志回传到 B 机（`#LOG` ⇒ `A_relay_trace.log`；不许污染回执计数）",
     t_a_logs_are_forwarded_to_b),
    ("⭐⭐ 触控板位移累积 + 8 ms 合并成一条（位移守恒、静下来停表）",
     t_pad_move_is_coalesced),
)


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:                             # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
