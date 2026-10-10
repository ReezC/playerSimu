"""性能保活自检：**实测**「睡 10 ms 到底睡了多久」，而不是只看"设置成功了"。

**为什么要有它**
    「窗口一失焦就卡」的根因是 Windows 默认定时器粒度约 15.6 ms，而工作台的
    关键回路是 10 ms 级的时序拍（`decision/agent.py` 的 TIMING_TICK）。
    这类问题的坑在于：**调用成功不等于生效** —— `timeBeginPeriod(1)` 会返回 0，
    但系统仍可能因为进程被电源节流（EcoQoS）而**忽略**它。所以这里的判据是
    实测数字：提升精度后，`sleep(10ms)` 应当量到 10~12 ms，而不是 15.6 ms。

跑法（B 机 / 工作台那台）：

    python -m tools.selftest_winperf        # 全过返回 0，有失败返回 1

只读：不写任何配置，只改本进程自己的优先级/定时器精度（退出即失效）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import winperf                                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# 判据：提到 1 ms 之后 10 ms 的等待不该超过这个数。
# （留了余量：真机上通常量到 10.0~10.6 ms，15.6 ms 是没生效时的典型值。）
WAIT_MAX_MS = 13.0


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _steps(d):
    return {s[0]: s for s in d["steps"]}


# ---------------------------------------------------------------- 四件事

def t_apply_all_steps():
    """四个步骤都要跑一遍，且各自带结果（失败也只记不抛）。"""
    d = winperf.apply(priority="above_normal", probe=0)
    names = [s[0] for s in d["steps"]]
    for want in ("电源节流", "进程优先级", "定时器精度", "防挂起"):
        check(want in names, "步骤 %s 没跑：%s" % (want, names))
    if sys.platform != "win32":
        check(all(not s[1] for s in d["steps"]),
              "非 Windows 上不该报告成功：%s" % (d["steps"],))
        return
    bad = [s for s in d["steps"] if not s[1]]
    check(not bad, "这些步骤没生效：%s" % (bad,))
    check(winperf.describe().startswith("性能保活"), "describe() 没输出摘要")


def t_priority_really_raised():
    """优先级要**独立核实**（不看 winperf 自己的报告）。"""
    if sys.platform != "win32":
        return
    import ctypes

    winperf.apply()
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k.GetPriorityClass.restype = ctypes.c_ulong
    got = k.GetPriorityClass(k.GetCurrentProcess())
    check(got == winperf.PRIORITY["above_normal"],
          "进程优先级是 0x%X，应当是 ABOVE_NORMAL(0x%X)"
          % (got, winperf.PRIORITY["above_normal"]))


def t_timer_precision_measured():
    """**核心判据**：提升精度前后各量一次 10 ms 的等待，把数字打出来。

    这是唯一能证明"真的生效"的办法：设置成功但被系统忽略时，
    数字仍然停在 15.6 ms —— 那才是"失焦就卡"的真凶。
    """
    if sys.platform != "win32":
        return
    winperf.release()                       # 回到默认精度，量基线
    before, _ = winperf.measure_wait(10.0, 20)
    winperf.apply()
    after, p95 = winperf.measure_wait(10.0, 20)
    best = winperf.measure_wait(10.0, 20)[0]
    # 判据用**最小值**而不是中位：机器忙的时候偶发几次迟到是正常的，
    # 但"定时器粒度"决定的是**最快能多快醒** —— 粒度 15.6 ms 时连最小值
    # 也下不来。用中位判会被负载抖动误伤（自检自己跑一堆东西，正是这种环境）。
    print("       要等 10 ms，实际等了：改前 %5.1f ms → 改后 中位 %.1f / 最小 %.1f"
          "（p95 %.1f）" % (before, after, best, p95))
    if before <= WAIT_MAX_MS:
        # **把结论说出来**：这一项在本机不是瓶颈时，别让人以为"保活白开了"，
        # 也别让人接着往这条线上查（本机实测过就是 10 ms 级）。
        # ⚠ 这里只能用 GBK 也有的字符：控制台是 936 代码页，`↳`/`✓` 这类会直接
        # 抛 UnicodeEncodeError（自检把自己搞崩，比不输出还糟）。
        print("       -> 本机改前就已经是 1 ms 级（%.1f ms）→ **定时器精度不是本机"
              "的瓶颈**；\n"
              "          「失焦变卡」要往别处查：GUI 主线程每帧绘制/信号积压 ——"
              "看实时预览状态行里的\n"
              "          「绘制 x ms 合并丢弃 n」（丢帧数涨就是主线程跟不上）。"
              % before)
    check(min(after, best) <= WAIT_MAX_MS,
          "定时器精度没生效：10 ms 的等待最快也要 %.1f ms（> %.0f ms）——\n"
          "       多半是电源节流没关掉（系统忽略了 timeBeginPeriod）。\n"
          "       检查「设置 → 性能保活」是否为开，或看上面「电源节流」那条的说明。"
          % (min(after, best), WAIT_MAX_MS))


def t_thread_boost():
    """关键回路线程的提权：Windows 上要有回报，非 Windows 上也只是 False。"""
    got = winperf.boost_thread()
    if sys.platform == "win32":
        check(got is True, "SetThreadPriority 没成功")
    else:
        check(got is False, "非 Windows 上应当返回 False")


def t_never_raises():
    """保活绝不该把界面搞挂：参数离谱也只记不抛。"""
    d = winperf.apply(priority="离谱的档位", timer_ms=1)
    st = _steps(d)
    check(st["进程优先级"][1], "档位不认识时应当退回「高于正常」")
    d2 = winperf.apply(timer_ms="不是数字")      # int() 会抛 → 要被兜住
    check(_steps(d2)["定时器精度"][1] is False,
          "参数不合法时该记成失败，而不是抛异常")
    winperf.apply()                              # 恢复成正常的一套
    winperf.release()
    winperf.release()                            # 重复 release 不该抛
    winperf.apply()


def t_probe_and_default_on():
    """apply(probe=n) 要带回实测数字；配置缺省时默认开。"""
    d = winperf.apply(probe=3)
    if sys.platform == "win32":
        check(isinstance(d.get("probe_ms"), float)
              and 5.0 < d["probe_ms"] < 40.0,
              "probe 数字不合理：%r" % (d.get("probe_ms"),))

    import core.config as tcfg
    old = tcfg.load_live
    try:
        tcfg.load_live = lambda: {}              # 配置里没有这个键
        check(winperf.enabled_by_config() is True,
              "缺少 perf_keepalive 时应当默认开（关掉的现象是「失焦就卡」，"
              "很难归因）")
        tcfg.load_live = lambda: {"perf_keepalive": False}
        check(winperf.enabled_by_config() is False, "显式关掉时应当读成 False")
    finally:
        tcfg.load_live = old


# ---------------------------------------------------------------- 接线

def t_wired_into_app():
    """启动接线是**源码约定**：少一处就等于没做，而且现场看不出来。"""
    app = (ROOT / "gui" / "app.py").read_text(encoding="utf-8")
    check("winperf.apply(" in app, "工作台启动时没有应用性能保活")
    check(app.index("winperf.apply(") < app.index("QApplication(sys.argv)"),
          "保活必须在建 QApplication 之前应用（进程级设置越早越好）")
    check("winperf.release()" in app or "_wp.release()" in app,
          "退出时没有 timeEndPeriod（占着 1 ms 名额不放，别的程序吃不到）")

    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("boost_thread()" in lt,
          "关键回路线程（收流→推理→决策）没提线程优先级")

    # 定时器精度只该有一处实现：wgc_capture 原先自己调过一份，现在走 winperf
    wgc = (ROOT / "core" / "wgc_capture.py").read_text(encoding="utf-8")
    check("timeBeginPeriod" not in wgc,
          "wgc_capture 又自己调 timeBeginPeriod 了 —— 定时器精度应当只有 "
          "core/winperf.py 一处实现")
    # ⚠ 取窗口必须按**框的中心点**（2026-10-09 修 ✗）：以前用 rect 的**左上角**那个点 ✗
    #   ⇒ 只要那个角压到工作台 / 任务栏 / 别的窗口，就**抓到别的窗口** ✗（用户现场："本地的
    #   实时窗口……好像是捕获程序窗口" ✓）⇒ 钉住"中心点优先、左上角兜底" ✓。
    check("_geom_at(x + w // 2, y + h // 2)" in wgc,
          "wgc_capture 又按 rect 左上角找窗口了 —— 那个角压到别的窗口就会抓错窗口 ✗"
          "（用户 2026-10-09 报的现场）")

    live = (ROOT / "config" / "live.yaml").read_text(encoding="utf-8")
    check("perf_keepalive" in live, "config/live.yaml 里没有 perf_keepalive 开关")

    sd = (ROOT / "gui" / "settings_dialog.py").read_text(encoding="utf-8")
    check("ck_keepalive" in sd and "winperf.apply()" in sd,
          "设置里没有性能保活开关（或改了没立即生效）")

    # 改 config/live.yaml 一律走 update_live：save_live 是整文件覆盖，
    # 谁直接用它写几个键，就会把别的键（含本文件的 perf_keepalive）静默抹掉
    for name in ("live_panel.py", "settings_dialog.py"):
        src = (ROOT / "gui" / name).read_text(encoding="utf-8")
        check("save_live(" not in src,
              "%s 又用 save_live 整文件覆盖了 —— 会把 perf_log / perf_keepalive "
              "一起抹掉（改用 core.config.update_live）" % name)


def t_update_live_keeps_keys():
    """`update_live` 不许把**别的键**抹掉。

    `save_live()` 是整文件覆盖：谁直接拿几个键去调它，就会把别处写的键
    （`perf_log` / `perf_keepalive`）一起抹掉 —— 现象是「设置里明明开着，
    重启后文件里没了、选项变回默认」。这条把那个坑钉住。
    """
    import shutil
    import tempfile

    import core.config as tcfg

    tmp = Path(tempfile.mkdtemp(prefix="livecfg_"))
    old = tcfg.LIVE_CONFIG
    tcfg.LIVE_CONFIG = tmp / "live.yaml"
    try:
        tcfg.save_live({"perf_log": True, "perf_keepalive": True, "imgsz": 800})
        tcfg.update_live(imgsz=960)
        got = tcfg.load_live()
        check(got.get("perf_keepalive") is True,
              "update_live 把 perf_keepalive 抹掉了：%s" % (got,))
        check(got.get("perf_log") is True, "update_live 把 perf_log 抹掉了：%s" % (got,))
        check(got.get("imgsz") == 960, "要改的键没改上：%s" % (got,))
        check(winperf.enabled_by_config() is True, "配置里开着却被读成关")
    finally:
        tcfg.LIVE_CONFIG = old
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_docs_mention():
    """winperf 得自己说清楚"为什么"，否则下一个人会把它当玄学调优删掉。"""
    src = (ROOT / "core" / "winperf.py").read_text(encoding="utf-8")
    for kw in ("timeBeginPeriod", "EcoQoS", "TIMING_TICK", "15.6"):
        check(kw in src, "winperf.py 的说明里缺 %s（解释不清就会被删）" % kw)


def t_covered_switches_to_screen():
    """⭐⭐ 源窗口**被别的窗口盖住** ⇒ 这一拍改走「整屏合成」（用户 2026-10-10 选的口径 ✓）。

    病（用户报的「本地实时窗口来源被遮挡」✓）：`window_capture: wgc` 是**单窗口采集** ✗ ⇒
    盖在源窗口上的**外接登录窗 / 断线提示框不进画面** ✗（断线重连、测谎判不出界面 ✓
    —— 用户 2026-10-09 原话就是为这个 ✓）；WGC 还每 ≤0.3s 按**框中心那个点**重挑窗口 ✗
    ⇒ 中心被遮挡物占了就**抓成遮挡物** ✗。

    钉四件：
      ① **被盖住** ⇒ **不走 WGC**、改走整屏合成，并留下「被谁盖住」的人话 ✓
         （界面状态行 / `behavior.log` 全靠它 ✓ 不然又是"静默换路"✗）；
      ② **没被盖** ⇒ **照旧走 WGC** ✓（不许一律改成整屏合成 ✗ 那是白扔性能 ✓）；
      ③ ⚠ **不许认死窗口** ✗：判据只按**标题**（那段源码里不许出现窗口句柄那套 ✓）——
         那是用户**明确否掉过**的方向 ✓（"盖住的窗口也要显示" ✓）；
      ④ 判据本身：9 个采样点同一个窗口 ⇒ 没被盖 ✓；中间插进来一个别的标题 ⇒ 被它盖住 ✓。
    """
    import numpy as np

    import core.dxgi_capture as dx
    import core.wgc_capture as wg
    import core.wincap as wc

    rect = (100, 100, 400, 300)
    img = np.zeros((2, 2, 3), np.uint8)
    ev = []
    _o = ((dx, "available"), (dx, "grab_rect"), (wg, "available"), (wg, "grab_rect"),
          (wc, "grab_rect_fast"), (wc, "title_at"), (wc, "class_at"))
    _saved = [(m, n, getattr(m, n)) for m, n in _o]
    try:
        # DXGI 用不了（大部分机器上 dxcam 不可用 ✓）⇒ 才会走到 WGC 那一档 ✓
        dx.available = lambda: (False, "")
        dx.grab_rect = lambda r: (ev.append("dxgi"), None)[1]
        wg.available = lambda: True
        wg.grab_rect = lambda r: (ev.append("wgc"), img)[1]
        wc.grab_rect_fast = lambda r: (ev.append("bitblt"), img)[1]
        # ⚠ `class_at` 也要 mock 成"非桌面/任务栏" ✓：它是**另一个探针**（排掉 `Progman` 那种
        #   桌面窗口 ✓ 见 `_SHELL_CLASSES` ✓）—— 真机跑用例时那个矩形通常正落在**桌面**上 ✗
        #   ⇒ 9 个点全被排掉 ⇒ 判成"没被盖"✗（本用例第一版就是这么假红的 ✓ 记一笔 ✓）。
        wc.class_at = lambda x, y: ""

        # ---- ① 被盖住：上半屏是「登录窗」、下半屏是「游戏」⇒ 整块跨了两个窗口 ✓ ----
        wc.title_at = lambda x, y: "登录窗" if y < 220 else "游戏"
        got = wc.grab_rect(rect, use_wgc=True)
        check(got is not None, "被盖住时抓帧直接失败了（该退到整屏合成 ✓）")
        check("wgc" not in ev and ev and ev[-1] == "bitblt",
              "被别的窗口盖住却还在用 WGC（单窗口采集 ⇒ 盖住的登录窗不进画面 ✗）：%r" % (ev,))
        _n = wc.last_note()
        check(_n.get("how") == "screen",
              "换抓法这件事**要留下痕迹**（`how` 该是 screen ✓）：%r" % (_n,))
        check("登录窗" in (_n.get("why") or ""),
              "没说清**被谁盖住**（状态行 / behavior.log 就只剩\"莫名其妙换了抓法\"✗）：%r"
              % (_n,))

        # ---- ②′ **迟滞**：刚判过"被盖" ⇒ 立刻又不被盖，也**不许马上切回** ✗ ----
        #   （真机日志实测：边缘噪声让判定 8 秒翻 4 次、`screen↔wgc` 只隔 14 ms ✗ ⇒
        #    同一路流一会儿单窗口一会儿整屏 ✗ —— 外接断线窗就"时有时无"⇒ 识别/警报跟着丢 ✓）
        ev.clear()
        wc.title_at = lambda x, y: "游戏"                 # 突然"不被盖"了 ✓
        got_h = wc.grab_rect(rect, use_wgc=True)
        check(got_h is not None and "wgc" not in ev and ev and ev[-1] == "bitblt",
              "刚判过\"被盖\"却**立刻切回 WGC** ✗ ⇒ 同一路流一会儿一个窗口一会儿整屏"
              "（外接断线窗时有时无 ⇒ 警报/识别会丢 ✓）：%r" % (ev,))
        check("继续抓整屏" in (wc.last_note().get("why") or ""),
              "迟滞期间没说清\"为什么还在抓整屏\"（复盘看不出发生过什么 ✗）：%r"
              % (wc.last_note(),))

        # ---- ② 没被盖（且迟滞已过）：9 个点全是同一个窗口 ⇒ 照旧 WGC ✓ ----
        ev.clear()
        wc._COVER["last"] = 0.0            # 模拟"已经很久没判到被盖"⇒ 迟滞窗口过去 ✓
        wc.title_at = lambda x, y: "游戏"
        got2 = wc.grab_rect(rect, use_wgc=True)
        check(got2 is not None and ev == ["wgc"],
              "没被盖住却不用 WGC 了（一律整屏合成 = 白扔性能 ✗）：%r" % (ev,))
        check(wc.last_note().get("why") == "",
              "没被盖住却留了「换路理由」（状态行会挂着一条假提示 ✗）：%r" % (wc.last_note(),))

        # ---- ③ 判据只按标题、不认窗口句柄（用户否掉过的方向 ✗）----
        _src = (ROOT / "core" / "wincap.py").read_text(encoding="utf-8")
        _body = _src.split("def covered_by(rect):", 1)[-1].split("\ndef ", 1)[0]
        check("hwnd" not in _body and "GetWindowThreadProcessId" not in _body,
              "「被谁盖住」的判据里又混进了窗口句柄那套 ✗（采集**不许认死窗口** ✓ "
              "盖住的窗口也要显示 ✓ 见 `covered_by` 的说明 ✓）")
        check("_OCCL_POINTS" in _body and len(wc._OCCL_POINTS) >= 5,
              "判据只取一个点 ⇒ 「正好压在中心的小窗」会被漏掉 ✗（WGC 就是按中心点挑的 ✓）")

        # ---- ④ 判据本身：中心窗口 + 一个别的标题 ⇒ 报出"被它盖住" ----
        wc.title_at = lambda x, y: "别的窗" if (x, y) == (int(rect[0] + rect[2] * 0.94),
                                                         int(rect[1] + rect[3] * 0.5)) else "游戏"
        _c, _why = wc.covered_by(rect)
        check(_c == "别的窗" and "别的窗" in _why,
              "角落里插进来一个别的窗口却判成\"没被盖\"：%r / %r" % (_c, _why))
        wc.title_at = lambda x, y: "游戏"
        check(wc.covered_by(rect) == (None, ""),
              "全都属于同一个窗口却判成\"被盖住\"（会一直走整屏合成 ✗）：%r"
              % (wc.covered_by(rect),))
    finally:
        for m, n, v in _saved:
            setattr(m, n, v)
        wc._NOTE["how"], wc._NOTE["why"] = "", ""
        wc._COVER["last"] = 0.0            # ⚠ 迟滞状态也要还原（不然会漏到别的用例 ✓）


def t_binding_rect_falls_back():
    """⭐⭐⭐ 「框选区域」这一拍抓哪块屏 —— **窗口问不到时必须退回你框的那块** ✓
    （用户 2026-10-10 ✓ 原话："现在本地窗口实时**并不是用的我框选的屏幕区域**，而是把
    **2 个显示器的全部内容**展示了"✓）。

    病因（本机实测 ✓）：`rect_rel` 是**相对那个窗口**的比例（≈ 0,0,1,1 = "整个窗口" ✓），
    而窗口问不到时老代码把它按**虚拟屏幕**折算 ✗ —— 本机虚拟屏幕 = `(-2560, 0, 5120, 1440)`
    = **两台显示器拼起来** ✓ ⇒ 一折就是 5117×1400 的"整块桌面" ✓ 用户看到的就是它 ✓。

    钉四件：
      ① 没记过窗口 / 没记过比例 ⇒ **绝对区域**（老行为 ✓ 一个字不变 ✓）；
      ② **窗口问不到**（关了 / 最小化 / 句柄失效 ✓）⇒ **绝对区域** ＋ 一句人话 ✓
         （⚠ 不许静默退回 ✗）；
      ③ 窗口**问得到** ⇒ 按它现在的几何折算 ✓（"挪了不用重框"这个功能不许被误伤 ✗）；
      ④ ⭐ **最小化**要被当成"问不到"（`window_geom` 那条 ✓ —— 否则拿到的是一块
         `(-31993, -32000, 146, 28)` 的屏幕外占位矩形 ✓ 实测值 ✓）。
    """
    import core.wincap as wc

    fall = (3, 33, 1921, 1081)                  # 用户框的那块（实测存值 ✓）
    rel = [0.00052, 0.027878, 0.99948, 0.972122]   # 相对窗口 ≈ 整个窗口 ✓（实测存值 ✓）
    _o = wc.window_geom
    try:
        # ---- ① 没绑定 ⇒ 绝对区域、且不留话 ✓ ----
        for args in ((None, None), (rel, None), (None, 399802)):
            got, why = wc.binding_rect(*args, fall)
            check(tuple(got) == tuple(fall) and why == "",
                  "没绑定窗口时该**原样**用绝对区域（老行为 ✗）：%r / %r" % (got, why))

        # ---- ② 窗口问不到 ⇒ 绝对区域 + 说清（⚠ 不是把比例按虚拟屏幕折算 ✗）----
        wc.window_geom = lambda h: None
        got, why = wc.binding_rect(rel, 399802, fall)
        check(tuple(got) == tuple(fall),
              "窗口问不到时**没退回你框的那块**（按虚拟屏幕折算就会变成两台显示器全屏 ✗）：%r"
              % (got,))
        check(why and "固定区域" in why,
              "退回绝对区域却**不说一声**（人只看到画面突然变了 ✗）：%r" % (why,))
        # ⚠ 顺手钉住"按虚拟屏幕折算"这个错法的结果长什么样（防再犯 ✓）：
        #   多显示器机器上它必然**远大于**单个屏宽 ✓（本机 5120 ⇒ 折出 5117 ✓ 实测 ✓）
        if wc.virtual_screen()[2] > 2000:
            _vs = wc.rect_from_rel(rel, None)
            check(_vs[2] > 1.5 * wc.virtual_screen()[2] / 2.0,
                  "多显示器上\"按虚拟屏幕折算\"居然没放大（那这条就失去意义了 ✓）：%r" % (_vs,))

        # ---- ③ 窗口问得到 ⇒ 跟着窗口算（功能不许被误伤 ✗）----
        wc.window_geom = lambda h: (-8, -8, 1936, 1096)      # 窗口在别的分辨率下 ✓
        got2, why2 = wc.binding_rect(rel, 399802, fall)
        check(tuple(got2) != tuple(fall) and abs(got2[2] - 1935) <= 3,
              "窗口问得到时该**按窗口现在的几何**折算（挪了/换分辨率不用重框 ✓）：%r" % (got2,))
        check(why2 == "", "一切正常却还挂着提示（状态行会一直闪一条假警告 ✗）：%r" % (why2,))

        # ---- ④ 最小化 ⇒ window_geom 必须判"问不到"（占位矩形不算几何 ✓）----
        _src = (ROOT / "core" / "wincap.py").read_text(encoding="utf-8")
        _body = _src.split("def window_geom(hwnd):", 1)[-1].split("\ndef ", 1)[0]
        check("IsIconic" in _body,
              "`window_geom` 没把**最小化**当\"问不到\" ✗ ⇒ 它的占位矩形 `(-32000,…)` 会被"
              "当成\"窗口在哪\"⇒ 落点算到屏幕外 ✓（用户那条\"不是用我框的区域\"的另一半 ✓）")
        check("-30000" in _body,
              "`window_geom` 没兜\"最小化占位矩形\"那道（`IsIconic` 偶尔问不到 ✓ 实测值 "
              "`(-31993, -32000, 146, 28)` ✓）")
    finally:
        wc.window_geom = _o


TESTS = (
    ("四个步骤都跑且有结果", t_apply_all_steps),
    ("进程优先级真的提上去了", t_priority_really_raised),
    ("定时器精度实测（10 ms 的等待真的只睡 10 ms）", t_timer_precision_measured),
    ("关键线程提权", t_thread_boost),
    ("失败只记不抛", t_never_raises),
    ("probe 实测数字 + 默认开", t_probe_and_default_on),
    ("接进工作台启动/线程/设置（源码约定）", t_wired_into_app),
    ("update_live 不丢别的键", t_update_live_keeps_keys),
    ("模块内解释了「为什么」", t_docs_mention),
    ("⭐⭐ 源窗口被盖住 ⇒ 改走整屏合成（不许认死窗口）", t_covered_switches_to_screen),
    ("⭐⭐⭐ 框选区域：窗口问不到 ⇒ 退回「你框的那块」（不许按虚拟屏幕放大）",
     t_binding_rect_falls_back),
)


def main() -> int:
    if sys.platform != "win32":
        print("（非 Windows：只验证不抛异常与源码接线，实测项跳过）\n")
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    if failed:
        print("提示：保活没生效时，工作台跑到一半失焦就会开始掉拍/发卡。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
