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
    import time as _t
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
        # ⚠ 回传现在**排队 + 独立线程发**（2026-10-04 ✓ 见 `_push_to_client`：不许阻塞
        #   转发那条路 ✗）⇒ 这里要**等一下真发出去**，不然会误判成"没推"✗。
        _dl = _t.time() + 2.0
        while not sink.buf and _t.time() < _dl:
            _t.sleep(0.01)
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


def t_pad_delta_from_tracker():
    """⭐⭐⭐ 触控板位移**由跟踪线程按固定节拍产生**（用户 2026-10-05 ✓ 原话："直接2"）。

    病（实测，`perf.log` 18:40 那段按 F10 滑了两下）：位移原来只在 `TouchPad.mouseMoveEvent`
    里产生 ✗（Qt 的 move 事件 ⇒ 跑在 GUI 主线程上）⇒ 主线程一被重绘 / 主回路拖住，Qt 就把连续
    移动**合并**成一次：
        `move_gap_ms` 中位 20 / p95 76 / p99 201 / **最大 467 ms**
        `move_px`                             **最大 514**（一次跳半个屏幕 ✓）
    ⇒ A 机"停一下、猛跳一下" ✓；而同一段 `send_ms` 中位 **0.15 ms**、`kbd_pending` ≤ 4
    ⇒ **瓶颈不在链路也不在发送，在位移的产生** ✓。
    方子（用户挑的"直接②"）：产生挪进 `decision/input.py::PointerTracker` 那条独立线程
    （固定节拍 `GetCursorPos` ✓ + `SetCursorPos` 拨回钉点 ✓），出口**直接**喂 `_PadSender`
    （不经 Qt 事件循环 ✓）。

    钉七件：
      ① 每拍都算位移、并**把指针拨回钉点**（无限滑动 ✓）；
      ② **位移守恒**：逐拍增量之和 == 总移动（不丢、不重复 ✓）；
      ③ `on_delta` 在**跟踪线程**里被调（线程名 `pad-track` ✓ —— 这是"不经 Qt 事件循环"的
         铁证：走 `moved` 信号会排队回到主线程 ✗）；
      ④ **拨回失败也算对**：用的是 `pos - 上一次`，不是 `pos - 钉点` ✗
         （后者会一次跳一大截、之后一路算 0 ✓）；
      ⑤ `stop()` 之后线程真的退出、位移不再增加 ✓；
      ⑥ 真 Win32 那条路能读能拨（**原样拨回去** ⇒ 不打扰正在用机器的人 ✓）；
      ⑦ 源码钉：位移**不再是** Qt move 事件产生的 ✗ ＋ 面板出口接在跟踪线程上 ✓ ＋
         跟踪线程里**不碰 Qt** ✗ ＋ **不自己调 `timeBeginPeriod`** ✗（精度只该 `core/winperf` 一处 ✓）。
    """
    import threading as _thr
    import time as _t
    from types import SimpleNamespace

    from decision import input as dinput
    from gui import player_panel as _pp

    # ⑥ 真 Win32：`GetCursorPos` 读得到 ✓；`SetCursorPos` **只钉契约**（不抛异常 + 回布尔 ✓）
    #   ⚠ **不许**断言"拨得动"：锁屏 / 非交互桌面下 `SetCursorPos` 返回 0、指针纹丝不动、
    #     `GetLastError` 还是 0（本机 2026-10-05 实测 ✓ 见 `decision/input.py::pointer_warp` ✓）
    #     ⇒ 那样会让用例随"屏幕锁没锁"红绿 ✗。真拨动/拨不动两条路都在 ④ 用假指针钉住了 ✓。
    _p0 = dinput.pointer_pos()
    check(_p0 is not None, "`GetCursorPos` 读不到（真 Win32 这条路 ✗）")
    _r = dinput.pointer_warp(_p0[0], _p0[1])
    check(isinstance(_r, bool),
          "`pointer_warp` 没回布尔（拨不动时上层没法判断 / 留痕 ✗）：%r" % (_r,))

    pos = [100, 100]                    # 假指针：位置由测试说了算 ✓
    warps = []
    got = []                            # 每条回调：(dx, dy, 线程名)
    warp_ok = [True]
    _op, _ow = dinput.pointer_pos, dinput.pointer_warp
    dinput.pointer_pos = lambda: (pos[0], pos[1])

    def _fake_warp(x, y):
        warps.append((x, y))
        if warp_ok[0]:                  # 真拨回 ⇒ 假指针也跟着回钉点（模仿真实 ✓）
            pos[0], pos[1] = int(x), int(y)
        return warp_ok[0]               # ⚠ 拨不动就如实回 False（真 Windows 就这样 ✓）

    dinput.pointer_warp = _fake_warp

    def _wait(n, seq, tmo=2.0):
        """等 `seq` 攒够 n 条：位置由测试摆好，线程下一拍就会看到 ✓
        （确定性 ✓ 不靠猜时序 ✓）。

        ⚠ 回调**记完位移之后**才拨回指针 ⇒ 断言 `warps` 前要**单独再等一拍** ✗
          （只等 `got` 会在"回调已记、拨回还没发生"那一瞬看歪 ⇒ 用例随机红 ✓）。
        """
        t0 = _t.perf_counter()
        while len(seq) < n and _t.perf_counter() - t0 < tmo:
            _t.sleep(0.002)

    t = dinput.PointerTracker(
        (100, 100),
        lambda dx, dy: got.append((dx, dy, _thr.current_thread().name)),
        scale=2.0, interval=0.003)

    def _move(dx, dy):
        """把假指针从**当前位置**挪 (dx,dy) 物理像素。

        ⚠ 必须这样建模（而不是写绝对坐标 ✓）：拨回成功时指针**已经回到钉点** ✓，
          写绝对坐标会把"回到钉点"那一段也算进去 ⇒ 期望值看着像写错了 ✓（第一版就这么绕进去的 ✓）。
        """
        pos[0] += dx
        pos[1] += dy

    try:
        check(t.start(), "跟踪线程没起来（`GetCursorPos` 有值却起不来 ✗）")
        _wait(len(got) + 1, got)                # 先空转一拍（第一拍没有"上一次"✓ settle ✓）

        # ①② 挪两次 ⇒ 两条位移（按 scale=2 除回逻辑像素 ✓），每次都要拨回钉点
        n0 = len(got)
        _move(6, 0)                             # 6 物理像素 ⇒ 3 逻辑像素
        _wait(n0 + 1, got)
        _move(0, 10)                            # 再 10 ⇒ 5（拨回已生效 ⇒ 这次是纯 +y ✓）
        _wait(n0 + 2, got)
        _wait(2, warps)                         # 拨回发生在回调之后 ⇒ 单独等 ✓
        # ⚠ 再晾几拍：**"我们拨回指针"这件事本身绝不许被读成位移** ✗ ——
        #   `last` 没跟着挪到钉点的话，下一拍就会多一条反向位移（这里是 (-3,-5) ✓），
        #   远程鼠标会每拍抖一下 ✓。假指针会立刻跟到钉点 ⇒ 这是**确定的红**、不是碰运气 ✓。
        _t.sleep(0.05)
        _d = [(dx, dy) for dx, dy, _n in got[n0:]]
        check(_d == [(3.0, 0.0), (0.0, 5.0)],
              "位移算得不对（该按 scale=2 除回逻辑像素 [(3,0),(0,5)]，实得 %r；"
              "多出来的那条多半是「拨回指针」被读成了位移 ✗）" % (_d,))
        check(len(warps) >= 2 and all(w == (100, 100) for w in warps),
              "没把指针拨回钉点（那就成了「有限大的板子」✗，滑到屏幕边就动不了 ✓）：%r"
              % (warps,))
        _sum = (sum(dx for dx, _y in _d), sum(dy for _x, dy in _d))
        check(_sum == (3.0, 5.0), "位移没守恒（丢/重复像素 ⇒ 鼠标会漏会跳 ✗）：%r" % (_sum,))

        # ③ 回调在跟踪线程里（不是主线程 ✓ 更不是"排队回 GUI"✗）
        _names = sorted({n for _dx, _dy, n in got})
        check(_names == ["pad-track"],
              "回调不在跟踪线程里（线程名 %r）—— 那说明位移又绕回 Qt 事件循环了 ✗"
              "（`moved` 信号是排队的 ⇒ 回 GUI 线程 ⇒ 又排在重绘后面 ✓）" % (_names,))

        # ④ 拨回失败 ⇒ 照样只算"这一拍真动了多少"（不许一次跳一大截、也不许一路算 0）
        warp_ok[0] = False
        n1 = len(got)
        _move(10, 0)
        _wait(n1 + 1, got)
        _move(10, 0)
        _wait(n1 + 2, got)
        _d2 = [(dx, dy) for dx, dy, _n in got[n1:]]
        check(_d2 == [(5.0, 0.0), (5.0, 0.0)],
              "拨回失败时算的是 `pos - 钉点`（会一次跳一大截 ✗）：%r" % (_d2,))

        # ⑤ stop 之后线程真的退出、也不再产生位移
        _th_alive = t._th
        t.stop()
        _t.sleep(0.15)
        check(_th_alive is not None and not _th_alive.is_alive(),
              "`stop()` 之后跟踪线程还活着（会继续摸指针 ✗）")
        _n_stop = len(got)
        _t.sleep(0.05)
        check(len(got) == _n_stop, "`stop()` 之后还在产生位移 ✗")
    finally:
        t.stop()                        # 幂等（已经停过就什么都不做 ✓）
        dinput.pointer_pos, dinput.pointer_warp = _op, _ow

    # ⑦ 面板出口：跟踪线程的原始位移 ⇒ 乘灵敏度 ⇒ **浮点**交给发送器（小数不许截 ✗）
    class _S:
        def __init__(self):
            self.pushed = []

        def push(self, dx, dy):
            self.pushed.append((dx, dy))

    _old_spd = _pp.settings.mouse_speed
    try:
        _pp.settings.mouse_speed = 2.0
        fake = SimpleNamespace(_pad_sender=_S())
        _pp.PlayerPanel._on_pad_tracked(fake, 3.4, -1.7)
        _p = fake._pad_sender.pushed
        check(len(_p) == 1 and abs(_p[0][0] - 6.8) < 1e-9 and abs(_p[0][1] + 3.4) < 1e-9,
              "面板出口没乘灵敏度 / 把小数截了（慢速滑动会丢位移 ✗）：%r" % (_p,))
    finally:
        _pp.settings.mouse_speed = _old_spd

    # ⑧ 源码钉（防哪天又悄悄把位移挪回 Qt 事件 / 挪回主线程 ✗）
    #   ⚠ "**不许出现**某某"这类判据必须查 **AST 里的真实调用**，不能字符串匹配 ✗：
    #     函数/类的说明文为了讲清病根**会写**`moved.emit` / `QCursor.setPos`（那是文字 ✓
    #     不是代码 ✓）⇒ 字符串匹配会被自己的说明打红 ✓（第一次就是这么红的 ✓）。
    import ast as _ast

    def _code_calls(text, name=None):
        """`text` 里**代码级**的调用（说明文字 / 注释不算 ✓）；给 `name` 就只查那个函数/类 ✓。"""
        tree = _ast.parse(text)
        if name is not None:
            tree = next(n for n in _ast.walk(tree)
                        if isinstance(n, (_ast.FunctionDef, _ast.ClassDef))
                        and n.name == name)
        return " ".join(_ast.dump(c) for c in _ast.walk(tree) if isinstance(c, _ast.Call))

    tp = (ROOT / "gui" / "touchpad.py").read_text(encoding="utf-8")
    _mv_calls = _code_calls(tp, "mouseMoveEvent")
    check("moved" not in _mv_calls and "setPos" not in _mv_calls,
          "`mouseMoveEvent` 里又**真的**调了 `moved.emit` / `QCursor.setPos` ⇒ 位移回到 Qt 事件里"
          "产生了（GUI 一卡就成撮 ⇒ A 机一顿一顿回来 ✗）")
    check("_start_track" in tp and "_stop_track" in tp and "set_delta_sink" in tp,
          "触控板没有起停跟踪线程 / 没有那个「实时出口」✗")
    check("timeBeginPeriod" not in _code_calls(tp),
          "`gui/touchpad.py` 里调了 `timeBeginPeriod` —— 定时器精度只该 `core/winperf.py` 一处 ✓")

    inp = (ROOT / "decision" / "input.py").read_text(encoding="utf-8")
    _ptr = inp.split("def pointer_pos", 1)[1].split("class PointerTracker", 1)[0]
    check("GetCursorPos" in _ptr and "SetCursorPos" in _ptr,
          "指针读写没走 Win32（`GetCursorPos` / `SetCursorPos` ✓ 那是唯一能在非 GUI 线程跑的 ✓）")
    check("QCursor" not in _ptr, "指针读写那里碰了 Qt（它要在非 GUI 线程里跑 ✗）")
    _trk_calls = _code_calls(inp, "PointerTracker")
    check("QCursor" not in _trk_calls,
          "跟踪线程里**真的**碰了 Qt（非 GUI 线程调 `QCursor` 会偶发崩 ✗）")
    check("timeBeginPeriod" not in _code_calls(inp),
          "`decision/input.py` 里调了 `timeBeginPeriod` —— 定时器精度只该 `core/winperf.py` 一处 ✓")
    check("pos[0] - last[0]" in inp,
          "位移不是 `pos - 上一次`（拨回失败会一次跳一大截 ✗）")
    check("pad_warp_fail" in inp,
          "拨回失败没留痕（真出问题时「指针为什么跑掉了」会查不出来 ✗）")

    pp = (ROOT / "gui" / "player_panel.py").read_text(encoding="utf-8")
    check("set_delta_sink(self._on_pad_tracked)" in pp,
          "面板没把位移出口接到跟踪线程上（那就又走 Qt 排队 ⇒ GUI 一卡就成撮 ✗）")
    check("_on_pad_moved" not in pp and "_flush_pad" not in pp,
          "面板还留着旧口径（经 Qt 事件的 8 ms 合并那条）✗")

    # ⑨ 控件那条**接线**真起一次（`devicePixelRatioF` / 起 / 停 / 不留线程 ✓）——
    #   ①②③④ 测的是**跟踪线程本身** ✓，这条测"触控板到底接上它没有" ✓
    #   ⚠ 源码钉只能证明"写着"，证明不了"跑得起来" ✗（`devicePixelRatioF` 这类会当场炸 ✓）。
    from PyQt5.QtWidgets import QApplication, QWidget

    from gui.touchpad import TouchPad

    # ⚠⚠ `QApplication` 必须**留个引用**（`_app = ...`）✗：写成
    #   `QApplication.instance() or QApplication([])` 而不接住的话，PyQt 会在那条表达式
    #   结束时把 app **析构掉** ✗ ⇒ 紧接着建 `QWidget()` 就是
    #   `qFatal("Cannot create a QWidget without QApplication")` ⇒ 进程直接
    #   **`0xC0000409`**（fail-fast ✓ **没有 traceback** ✓ 实测踩过 ✓ 靠逐行插桩才夹出来 ✓）。
    _app = QApplication.instance() or QApplication([])
    _host = QWidget()
    _host.resize(800, 600)
    _host.show()                     # 离屏平台 ⇒ 看不见 ✓ 只为让子控件"可见"（`grabMouse` 要 ✓）
    pad = TouchPad(_host)
    pad.set_delta_sink(lambda dx, dy: None)
    try:
        pad.set_active(True)
        check(pad._tracker is not None and not pad._track_fail,
              "触控板激活后没接上跟踪线程（板子会像坏的一样、一个字都不说 ✗）")
        check("pad-track" in [t.name for t in _thr.enumerate()],
              "没有 `pad-track` 线程在跑（激活等于没做 ✗）")
    finally:
        pad.set_active(False)
    _t.sleep(0.2)
    check(pad._tracker is None, "停用后没清掉 `_tracker`（下次激活会漏掉旧线程 ✗）")
    check("pad-track" not in [t.name for t in _thr.enumerate()],
          "停用后 `pad-track` 线程还在（关了还在后台摸指针 ✗）")


def t_kbd_stats_reports_backlog():
    """⭐⭐⭐ **A 侧把「键盘这条通道忙不忙」汇报给 B**（用户 2026-10-04 ✓ 原话："现在老有
    指令堆积的问题，能否做一些 A 向 B 汇报的东西？"）。

    为什么单开一条：`MoveStats` **只认 `MOVE`** ✓（键盘混进去会把鼠标率算歪 ✗）⇒ 键盘这条
    在 A 侧**一个字都不统计** ✗ —— 而"堆积"正是在它上面（B 发的 `PRESS/TAP/…` 走 relay
    原样写进 115200 串口，固件**逐条执行**且 `TAP` 里还有 `delay` ✓ ⇒ 发得比执行快就积压 ✓）。

    一行里两个数就能定案：`n`（**发送率**）与 `完成`（**固件完成率**）——发 N 条/秒、完成
    M 条/秒、N>M **就是在积压** ✓；`写`（写串口耗时）告诉你"下游堵到什么程度" ✓。

    钉五件：
      ① 只认键盘（`MOVE` 不许混进来 ✗）；
      ② 一批几条 / 到达间隔都量出来（成批、一顿一顿的证据 ✓）；
      ③ `写` / `完成` 两个数都在（**本次要的就是它们** ✓）；
      ④ **半行**（TCP 切包）留到下一批 ⇒ 不许把残行当命令 ✗；
      ⑤ 没到 `report_sec` 不报（别把日志刷爆 ✓）。
    """
    from remote_kbd.relay import KbdStats

    st = KbdStats(report_sec=0.0)
    t = 100.0
    st.feed(b"PRESS LEFT\nTAP d 30\n", t, wr_ms=0.4, done=1)      # 一批 2 条
    t += 0.025
    st.feed(b"MOVE 3 0\nRELEASE LEFT\n", t, wr_ms=0.6, done=2)    # 鼠标混进来：只算 1 条
    line = st.maybe_report(now=t + 0.001)
    check(line and line.startswith("KBD 节拍 "), "没报出 `KBD 节拍` 那行：%r" % (line,))
    check("n=3" in line, "键盘条数算错了（`MOVE` 混进来了 / 漏算了？）：%s" % line)
    check("批内中位2最大2" in line, "批内条数没量出来（「成批」就靠它 ✗）：%s" % line)
    check("间隔中位" in line and "最大25.0ms" in line,
          "到达间隔没量出来（「一顿一顿」就靠它 ✗）：%s" % line)
    check("写中位0.6 最大0.6ms" in line,
          "写串口耗时没报（**下游堵没堵就看它** ✗）：%s" % line)
    check("完成 3（" in line and "条/秒" in line,
          "固件完成率没报（发送率 vs 完成率才是积压判据 ✗）：%s" % line)

    # ④ 半行：`TAP d` 与 ` 30\n` 分两批到 ⇒ 只算 1 条（不许把残行当命令 ✗）
    st2 = KbdStats(report_sec=0.0)
    st2.feed(b"TAP d", 1.0)
    st2.feed(b" 30\n", 1.01, wr_ms=0.2)
    line2 = st2.maybe_report(now=1.02)
    check("n=1" in line2 and "批=1" in line2, "半行被算成两条命令了：%s" % line2)
    check("写中位0.2" in line2, "写耗时丢了：%s" % line2)

    # ⑤ 没到点不报（不然 5 秒的窗口会被刷成一堆行 ✗）
    st3 = KbdStats(report_sec=5.0)
    st3.feed(b"PRESS A\n", 10.0)
    check(st3.maybe_report(now=10.5) is None, "没到 `report_sec` 就报了一行 ✗")
    check(st3.maybe_report(now=15.5) is not None, "到点了却没报 ✗")


def t_trace_push_is_async_and_once():
    """⭐⭐ A 侧 `trace()`：**只推一次** + **绝不阻塞命令转发**（用户 2026-10-04 ✓）。

    现场两件事（都在 `remote_kbd/A_relay_trace.log` 与代码里看出来的 ✓）：
      · B 收到的那份日志里**每行都是双份** ✗ —— `trace()` 头上推一次、收尾**又推一次** ✗
        （回程和固件回执**同一条 TLS** ⇒ 白占一半回程带宽 ✓）；
      · 更要紧：回传原来是**同步 `sendall`**，而 `trace()` 就在转发那条路上（`link.write`
        **之前** ✓）⇒ 客户端读得慢时 `sendall` 最长阻塞 **5 秒**（那 socket 的写超时 ✓）
        ⇒ **这一拍的命令根本发不到固件** ✗ = "指令堆积 / 按键突然很迟"的一种真成因 ✓。
      ⇒ 现在：**只推一次** + **排队交给独立线程**（满了丢最旧 ✓）。

    钉四件：
      ① `trace()` 只推**一次**（数 sink 里带标记的行 ✓）；
      ② 回传**不阻塞**：sink 每次 `sendall` 故意慢 20 ms ⇒ `trace()` 必须**立刻返回** ✓；
      ③ 队列满了**丢最旧**且**不抛**（`_QDROP` 涨 ✓、长度不超上限 ✓）；
      ④ 没客户端连着 ⇒ 一个字都不排队（老行为 ✓）。
    """
    import time as _t
    from remote_kbd import relay as relay_mod

    class _SlowSink:
        # ⚠ 故意慢到 **80 ms**：它必须**大于**下面那个 50 ms 的判据 ✗ —— 不然"同步发送"
        #   那种写法（20 ms）也会通过 ⇒ 用例白写 ✓（反向验证实测就是这么发现的 ✓）。
        def __init__(self, block=0.08):
            self.block = float(block)
            self.buf = b""

        def sendall(self, raw):
            _t.sleep(self.block)
            self.buf += raw

    old = relay_mod._LOG_SINK
    sink = _SlowSink()
    _drop0 = relay_mod._QDROP
    try:
        relay_mod._LOG_SINK = sink
        with relay_mod._QLOCK:
            relay_mod._QUEUE.clear()            # 别把上一轮的残留算进来 ✓
        # ② **先**验"不阻塞"（⚠ 顺序有讲究：它排在"只推一次"**前面**，两条性质才各自
        #   能被独立验到 ✓ —— 反过来时"同步发送"那种写法会先被 ① 挂住、② 根本没跑到 ✗）
        _t0 = _t.perf_counter()
        relay_mod.trace("排队不阻塞")
        _dt = _t.perf_counter() - _t0
        check(_dt < 0.05,
              "回传把调用方阻塞住了（它就在转发那条路上 ⇒ 命令会发不出去 ✗）：%.3fs" % _dt)
        # ① 只推一次
        relay_mod.trace("tcp->serial PRESS LEFT|")
        _dl = _t.time() + 2.0
        while sink.buf.count(b"PRESS LEFT|") < 1 and _t.time() < _dl:
            _t.sleep(0.01)
        _t.sleep(0.15)                          # 再等一会儿，看有没有第二份 ✗
        check(sink.buf.count(b"PRESS LEFT|") == 1,
              "`trace()` 推了不止一次（现场实测 `A_relay_trace.log` 里每行双份 ✗）：%r"
              % (sink.buf,))
        # ③ 队列满 ⇒ 丢最旧、不抛
        for _i in range(relay_mod.LOG_BACKLOG_MAX + 50):
            relay_mod.trace("y%d" % _i)
        check(len(relay_mod._QUEUE) <= relay_mod.LOG_BACKLOG_MAX,
              "队列长度超过上限（回传会把内存吃光 ✗）：%d" % (len(relay_mod._QUEUE),))
        check(relay_mod._QDROP > _drop0,
              "队列满了**没丢**（该丢最旧 ✓）：%d → %d"
              % (_drop0, relay_mod._QDROP))
    finally:
        relay_mod._LOG_SINK = old               # 让发送线程把剩下的丢掉 ✓
        relay_mod._QUEUE.clear()

    # ④ 没客户端 ⇒ 不排队
    relay_mod._LOG_SINK = None
    _n0 = len(relay_mod._QUEUE)
    relay_mod.trace("没有客户端")
    check(len(relay_mod._QUEUE) <= _n0, "没有客户端时还在排队（白占内存 ✗）")

    # 源码级：`trace()` 里只许出现**一处**回传调用（双份那个坑别再回来 ✗）
    src = (ROOT / "remote_kbd" / "relay.py").read_text(encoding="utf-8")
    _tr = src[src.index("def trace(text):"):]
    _tr = _tr[: _tr.index("\nclass ") if "\nclass " in _tr else len(_tr)]
    check(_tr.count("_push_to_client(text)") == 1,
          "`trace()` 里回传调用了 %d 处（要 1 处 ✗ —— 两处就是每行双份 ✓）"
          % _tr.count("_push_to_client(text)"))


def t_kbd_pending_is_sampled():
    """⭐ B 侧把「**还没等到回执的指令条数**」打点进 `perf.log`（用户 2026-10-04 ✓）。

    为什么：`kbd_rtt_ms` 只说"**最老的**那条等了多久"，看不出**攒了几条** ✗ —— 而"指令堆积"
    最直接的量化就是**在飞条数** ✓（配 A 侧 `KBD 节拍` 那行的"发送率 vs 完成率"一起看 ✓）。

    钉三件：
      ① 打点的是在飞条数（`kbd_pending` ✓）；
      ② **带 `min_gap`** —— 它是连续量（每发一条就变、~10 条/秒 ✗）⇒ 不定频会把日志刷爆 ✓；
      ③ `send()`（+1）与 `_take_rtt()`（回落）**两处都记** ✓（只记一处 ⇒ 涨上去就下不来 ✗）。
    """
    from unittest.mock import patch

    from core import perf as _pf
    from remote_kbd import kbd_client as kc

    rec = []
    c = kc.KbdClient.__new__(kc.KbdClient)          # 不连网 ✓
    c._pending = 3
    with patch.object(_pf, "sample", lambda k, v, **kw: rec.append((k, v, kw))):
        c._sample_pending()
    check(rec and rec[-1][0] == "kbd_pending" and rec[-1][1] == 3,
          "在飞条数没进 `perf`：%r" % (rec,))
    # ⚠⚠ 2026-10-05 修正：**这里原来断言 `min_gap=1.0`，是错的** ✗ —— `core.perf.sample`
    #   只吃 `(name, value)`（**没有 kwargs** ✓ 见它的签名 ✓），写成 `min_gap=` 会抛
    #   `TypeError`，而调用处包着 `try/except` ⇒ 从写出来那天起**一条样本都没进过**
    #   （全历史 1497 段里 `kbd_pending` 0 条 ✗）—— 这正是 `_sample_pending` docstring 里
    #   那个"哑弹藏了三个月"✓。⇒ 现在钉的变成**相反的两件**：样本进得去 + **源码里不许再写
    #   `min_gap=`**（防哑弹回来 ✓）。
    check(rec[-1][2] == {},
          "`perf.sample` 收到了多余关键字（`core.perf` 不吃 `min_gap=` ⇒ 会被静默吞掉 ✗）：%r"
          % (rec[-1],))
    _kc_src = (ROOT / "remote_kbd" / "kbd_client.py").read_text(encoding="utf-8")
    _body = _kc_src.split("def _sample_pending(self):", 1)[1]
    _body = _body.split("\n    def ", 1)[0]
    # ⚠ 只扫**代码**：那段历史说明（docstring）里**故意写着** `min_gap=` 这几个字 ✓
    #   （"别写它、为什么" ✓）⇒ 连注释一起扫会误报 ✗
    _code = _body.split('"""', 2)[2] if _body.count('"""') >= 2 else _body
    check("min_gap" not in _code,
          "`_sample_pending` 里又写了 `min_gap=`（`core.perf.sample` 只吃 (name, value) ⇒ "
          "TypeError 被吞 ⇒ 这条曲线重新变哑弹 ✗）")

    src = (ROOT / "remote_kbd" / "kbd_client.py").read_text(encoding="utf-8")
    check("def _sample_pending(self):" in src, "`_sample_pending` 没定义 ✗")
    check(src.count("self._sample_pending()") == 2,
          "调用点不是 2 处（`send()` 记涨 / `_take_rtt()` 记落 ⇒ 缺一处就只涨不落 ✗）：%d"
          % src.count("self._sample_pending()"))


def t_pad_sender_off_gui_thread():
    """⭐⭐ 触控板位移**由独立线程发**（用户 2026-10-05 ✓ 原话："B机F10控制A机鼠标，A机的鼠标
    移动不连续"）。

    病根（2026-10-03 那次只治了一半 ✗）：那时做了"累积 + 8 ms 合并" ✓，但**发送仍在 GUI
    主线程**里同步调 `dinput.mouse_move` ✗ —— 它要等 socket + 背压
    （`kbd_rtt_ms` 中位 **11 ms**、p95 更大 ✓），而主线程还要画（`draw_ms` 中位 5 /
    p99 15~22 ms ✓）⇒ 那个"8 ms 节拍"实际是"**何时腾出手何时发**" ✗ ⇒ 一次发一大坨
    ⇒ **A 机鼠标一顿一顿** ✓。
    方子同 `decision/manual_input` 2026-10-02 那次 ✓：**上游只入队，发送归独立线程** ✓
    （⚠ 2026-10-05 又查一层：上游"产生位移"那段也是挂在 GUI 事件上的 ✗ ⇒ 一并挪进
    `PointerTracker` 独立线程 ✓ 见 `t_pad_delta_from_tracker` ✓）。

    钉四件：
      ① **`push()` 绝不阻塞**（假发送睡 120 ms ⇒ push 十次总耗时 < 5 ms ✓）—— 这是
         "GUI 不被拖住"的全部保证 ✓；
      ② **位移守恒**（多慢都一像素不丢：发出去的和 == 推进去的和 ✓）；
      ③ **合并生效**（推得快 + 发得慢 ⇒ 发出条数 < 推入次数 ✓，但总量仍然守恒 ✓）；
      ④ `stop()` 之后线程真的退出 ✓。
    """
    import time as _t

    from gui.player_panel import _PadSender
    from decision import input as dinput

    sent = []
    _old = dinput.mouse_move
    dinput.mouse_move = lambda dx, dy: (sent.append((dx, dy)), _t.sleep(0.12))
    s = _PadSender()
    try:
        # ① push 绝不阻塞（发送慢 120ms/条也拖不住它 ✓）
        _t0 = _t.perf_counter()
        for _i in range(10):
            s.push(3, 1)                  # 共 30/10 px
        _push_ms = (_t.perf_counter() - _t0) * 1000.0
        check(_push_ms < 5.0,
              "`push()` 被发送拖住了（%.1f ms ⇒ GUI 照样卡 ⇒ A 机一顿一顿 ✗）" % _push_ms)
        # ②③ 等它把活干完（慢发送 120ms/条 ⇒ 给足时间），位移必须守恒、且确实合并过
        _t.sleep(1.2)
        check(sum(x for x, _y in sent) == 30 and sum(y for _x, y in sent) == 10,
              "位移没守恒（丢像素 ⇒ 鼠标会跳/漏 ✗）：%r" % (sent,))
        check(s.n_sent < s.n_push,
              "没有合并（推 %d 次发了 %d 条 —— 等于每条单独发 ✗）" % (s.n_push, s.n_sent))
        # ④ stop 之后线程退出（别留守护线程 ✓）
        s.stop()
        _t.sleep(0.3)
        check(not s._th.is_alive(), "`stop()` 之后发送线程还活着 ✗")
    finally:
        dinput.mouse_move = _old
        s.stop()


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
    ("⭐⭐⭐ 触控板位移由**跟踪线程**产生（固定节拍 / 守恒 / 不经 Qt 事件；用户 2026-10-05）",
     t_pad_delta_from_tracker),
    ("⭐⭐ 触控板位移**由独立线程发**（push 不阻塞 / 守恒 / 合并 / stop 退出；用户 2026-10-05）",
     t_pad_sender_off_gui_thread),
    ("⭐⭐⭐ A 侧把「键盘这条」也汇报给 B（发送率 / 写串口耗时 / 固件完成率 ⇒ 积压判据）",
     t_kbd_stats_reports_backlog),
    ("⭐⭐ A 侧 `trace()` 只推一次 + 回传绝不阻塞命令转发（队列满丢最旧）",
     t_trace_push_is_async_and_once),
    ("⭐ B 侧打点「还没等到回执的条数」（kbd_pending：积压最直接的量化）",
     t_kbd_pending_is_sampled),
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
