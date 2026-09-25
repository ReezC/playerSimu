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

    import tools.config as tcfg
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
              "一起抹掉（改用 tools.config.update_live）" % name)


def t_update_live_keeps_keys():
    """`update_live` 不许把**别的键**抹掉。

    `save_live()` 是整文件覆盖：谁直接拿几个键去调它，就会把别处写的键
    （`perf_log` / `perf_keepalive`）一起抹掉 —— 现象是「设置里明明开着，
    重启后文件里没了、选项变回默认」。这条把那个坑钉住。
    """
    import shutil
    import tempfile

    import tools.config as tcfg

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
