"""Windows 性能保活：让工作台**不是焦点窗口**时，照样拿到它需要的调度与定时器精度。

**为什么需要**（不是"优化"，是补一个本来就缺的声明）
    工作台的关键回路是 **10 ms 级**的时序拍：`decision/agent.py` 的
    `TIMING_TICK = 0.01`，由 `gui/live_thread.py` 主循环的
    `slot.take(timeout)` → `Event.wait()` 驱动。而 Windows 的默认定时器粒度约
    **15.6 ms** —— 想睡 10 ms，实际睡 15.6 ms，拍子被拉长一半。再叠上系统对
    **非前台进程**的电源节流（EcoQoS：降频 + 更激进的定时器合并），表现就是
    「窗口一失焦就卡」。

    这两件事都必须由**进程自己声明**：
      · `timeBeginPeriod(1)` 提高定时器精度 —— Win10 2004 起是**按进程**生效的，
        别的程序调过不等于自己也有；
      · `SetProcessInformation(ProcessPowerThrottling)` 明确告诉系统"别节流我"。
    而项目原先只有 `core/wgc_capture.py` 被 import 时调过一次
    `timeBeginPeriod(1)`（且从没 `timeEndPeriod`）—— 那条路只有「来源 = 本地窗口」
    抓屏才走，**收流模式全程没提过精度**。

    ⚠ **但别把 15.6 ms 当成本机事实**：有的机器（实测过一台 B 机就是）默认已经
    是 1 ms 级 —— 别的程序早把系统定时器分辨率提上去了，于是这一项成了空操作。
    判据永远是 `tools/selftest_winperf` 实测的「改前 / 改后」，不是上面的理论值；
    「失焦就卡」如果压在这上面，那是猜的。

**这里做四件事**（每件独立失败，失败只记不抛 —— 保活绝不该把界面搞挂）
    1. 关电源节流：不降频、不忽略我们的定时器精度请求；
    2. 进程优先级 → 高于正常：和前台程序抢 CPU 时不被饿着；
    3. 定时器精度 → 1 ms：10 ms 的等待真的只睡 10 ms；
    4. 阻止系统进入空闲省电（可选）：长时间挂机不被系统挂起。
    另外给关键回路提供 `boost_thread()`：把实时/决策线程单独提一档。

**开关**：「设置 → 性能保活」，落在 `config/live.yaml` 的 `perf_keepalive`（默认开）。
**验证**：`python -m tools.selftest_winperf` —— 它会**实测**"睡 10 ms 到底睡了多久"，
把改前/改后的数字摆出来（这是唯一能证明真的生效的办法）。

**不做什么**：不改 Qt 的事件循环、不动渲染、不占 CPU 自旋。定时器精度提到 1 ms
之后 `Event.wait(0.01)` 本身就是准的，没必要用"睡到只差 2 ms 再忙等"那种手法
（那要白烧一两个点的 CPU，本项目不需要）。
"""

import ctypes
import statistics
import sys
import time

IS_WIN = sys.platform == "win32"

# 进程优先级类（winbase.h）
PRIORITY = {
    "normal": 0x00000020,        # NORMAL_PRIORITY_CLASS
    "above_normal": 0x00008000,  # ABOVE_NORMAL_PRIORITY_CLASS
    "high": 0x00000080,          # HIGH_PRIORITY_CLASS
}
# 线程优先级（winbase.h）：只给关键线程用，别动 GUI 线程
THREAD_PRIORITY = {
    "above_normal": 1,           # THREAD_PRIORITY_ABOVE_NORMAL
    "highest": 2,                # THREAD_PRIORITY_HIGHEST
}

# PROCESS_POWER_THROTTLING_STATE（processthreadsapi.h）
_PROCESS_POWER_THROTTLING = 4
_THROTTLE_EXECUTION_SPEED = 0x1
_THROTTLE_IGNORE_TIMER_RESOLUTION = 0x4

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001

# 最近一次 apply() 的结果：[(标签, 成功?, 说明)]，界面/日志读它
STEPS = []
# 已经生效的东西（release() 只回退真的设过的）
_state = {"timer_ms": 0, "keep_awake": False}


class _ThrottleState(ctypes.Structure):
    """SetProcessInformation 的入参结构。"""

    _fields_ = [("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong)]


def _k32():
    """kernel32（带 last_error，失败时能报出 Windows 错误码）。"""
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _proc():
    k = _k32()
    k.GetCurrentProcess.restype = ctypes.c_void_p
    return k.GetCurrentProcess()


def _err():
    return "错误码 %d" % ctypes.get_last_error()


def _record(label, ok, why=""):
    STEPS.append((label, bool(ok), why))
    return bool(ok)


# ---------------- 四件事 ----------------

def disable_power_throttling():
    """关掉电源节流（EcoQoS）。

    两个位都要**清**（StateMask=0）：
      · EXECUTION_SPEED —— 不清的话，系统可以在我们"不在前台"时降频；
      · IGNORE_TIMER_RESOLUTION —— 不清的话，系统可以**忽略**我们的
        timeBeginPeriod 请求（那样第 3 件事等于白做）。
    这正是"失焦就卡"的直接来源，也是浏览器/播放器都会显式声明的一项。
    """
    if not IS_WIN:
        return _record("电源节流", False, "非 Windows")
    try:
        k = _k32()
        # ⚠ argtypes 必须显式写：不写的话 ctypes 按 C int 传参，而进程句柄是
        # 64 位指针 → 直接 OverflowError: int too long to convert（自检里逮到的）。
        k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_ulong]
        k.SetProcessInformation.restype = ctypes.c_int
        st = _ThrottleState(1, _THROTTLE_EXECUTION_SPEED
                            | _THROTTLE_IGNORE_TIMER_RESOLUTION, 0)
        ok = k.SetProcessInformation(_proc(), _PROCESS_POWER_THROTTLING,
                                    ctypes.byref(st), ctypes.sizeof(st))
        return _record("电源节流", ok, "已关" if ok else _err())
    except Exception as e:
        return _record("电源节流", False, "%s: %s" % (type(e).__name__, e))


def set_priority(level="above_normal"):
    """进程优先级。默认「高于正常」而不是「高」：够抢到 CPU，又不会把同机的
    其它程序（含 A 机那份对时/串口进程的对端）饿着。"""
    if not IS_WIN:
        return _record("进程优先级", False, "非 Windows")
    try:
        k = _k32()
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k.GetPriorityClass.argtypes = [ctypes.c_void_p]
        k.GetPriorityClass.restype = ctypes.c_ulong
        want = PRIORITY.get(level, PRIORITY["above_normal"])
        ok = k.SetPriorityClass(_proc(), want)
        got = k.GetPriorityClass(_proc())
        return _record("进程优先级", ok and got == want,
                       "0x%X" % got if ok else _err())
    except Exception as e:
        return _record("进程优先级", False, "%s: %s" % (type(e).__name__, e))


def begin_timer_period(ms=1):
    """把定时器精度提到 ms 毫秒（Win10 2004+ 按进程生效）。

    **为什么是这一步在治"卡"**：主循环用 `Event.wait(0.01)` 等 10 ms 的时序拍，
    默认精度下实际是 15.6 ms —— 每拍多 56%，越积越偏。调到 1 ms 之后
    `sleep(0.01)`/`wait(0.01)` 就是 10 ms 级。
    """
    if not IS_WIN:
        return _record("定时器精度", False, "非 Windows")
    try:
        rc = ctypes.windll.winmm.timeBeginPeriod(int(ms))
        if rc == 0:
            _state["timer_ms"] = int(ms)
        return _record("定时器精度", rc == 0, "%d ms" % ms if rc == 0
                       else "winmm 返回 %s" % rc)
    except Exception as e:
        return _record("定时器精度", False, "%s: %s" % (type(e).__name__, e))


def keep_awake(on=True):
    """阻止系统进入空闲省电状态（长时间挂机不被挂起）。

    **只声明「系统别睡」，不声明「显示器别关」**：关屏不影响后台回路，
    而要求屏幕常亮会白白费电、也容易让人以为程序在抢资源。
    """
    if not IS_WIN:
        return _record("防挂起", False, "非 Windows")
    try:
        flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0)
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        # 0x80000001 超出有符号 int，必须显式声明成 ULONG（同 电源节流 那条的教训）
        k.SetThreadExecutionState.argtypes = [ctypes.c_ulong]
        k.SetThreadExecutionState.restype = ctypes.c_ulong
        prev = k.SetThreadExecutionState(flags)
        if on and not prev:
            return _record("防挂起", False, _err())
        _state["keep_awake"] = bool(on)
        return _record("防挂起", True, "开" if on else "关")
    except Exception as e:
        return _record("防挂起", False, "%s: %s" % (type(e).__name__, e))


def boost_thread(level="above_normal"):
    """把**当前线程**提一档（给实时/决策线程用）。

    为什么单独要这一下：进程优先级管的是"和别的进程抢"，而 Windows 会给
    **前台进程的线程**额外的调度优待 —— 失焦时被压下去的往往正是这条要
    10 ms 一拍、还不能迟到的回路。别拿它提 GUI 线程（GUI 慢一点看得出，
    决策拍晚了是操作事故）。
    """
    if not IS_WIN:
        return False
    try:
        k = _k32()
        k.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        k.GetCurrentThread.restype = ctypes.c_void_p
        return bool(k.SetThreadPriority(k.GetCurrentThread(),
                                        THREAD_PRIORITY.get(level, 1)))
    except Exception:
        return False


# ---------------- 总开关 ----------------

def enabled_by_config():
    """读 `config/live.yaml` 的 `perf_keepalive`（默认开）。

    默认开是有意的：**关掉它的现象是"失焦就卡"，那是个很难归因的故障**，
    没人会想到是自己关的。想省电再自己去设置里关。
    """
    try:
        # 用 load_live()（tools.config.get 读的是 link.yaml，单键会被当成 section）
        from tools.config import load_live
        return bool(load_live().get("perf_keepalive", True))
    except Exception:
        return True


def apply(priority="above_normal", timer_ms=1, awake=True, probe=0):
    """一口气做完四件事；返回步骤结果（供日志/界面显示）。

    probe > 0 时顺带**实测**一次"睡 10 ms 实际睡了多久"（n=probe 次采样，
    每次约 10 ms，所以 probe=5 只花 50 ms）。数字留在结果里 ——
    "声称设好了"和"真的准了"是两回事，这一项是唯一能自己证明自己的。
    """
    del STEPS[:]
    disable_power_throttling()
    set_priority(priority)
    begin_timer_period(timer_ms)
    keep_awake(awake)
    out = {"steps": list(STEPS), "probe_ms": None}
    if probe > 0:
        med, _p95 = measure_wait(10.0, probe)
        out["probe_ms"] = med
    return out


def release():
    """退出时回退（`timeEndPeriod` 不还回去，别的程序就吃不到高精度定时器）。"""
    if IS_WIN:
        try:
            if _state.get("timer_ms"):
                ctypes.windll.winmm.timeEndPeriod(int(_state["timer_ms"]))
                _state["timer_ms"] = 0
            if _state.get("keep_awake"):
                ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
                _state["keep_awake"] = False
        except Exception:
            pass
    del STEPS[:]


# ---------------- 诊断 ----------------

def measure_wait(ms=10.0, n=60):
    """实测「要等 ms 毫秒，实际等了多久」→ (中位, p95)，单位毫秒。

    这是判断"保活有没有真的生效"的标准动作：默认精度下 10 ms 会量到 ~15.6 ms，
    提到 1 ms 之后应该落到 10~12 ms。
    """
    xs = []
    for _ in range(max(1, int(n))):
        t0 = time.perf_counter()
        time.sleep(ms / 1000.0)
        xs.append((time.perf_counter() - t0) * 1000.0)
    xs.sort()
    med = statistics.median(xs)
    p95 = xs[min(len(xs) - 1, int(len(xs) * 0.95))]
    return round(med, 2), round(p95, 2)


def describe():
    """一行摘要（给日志/状态行）。"""
    if not STEPS:
        return "性能保活：未应用"
    ok = [s for s in STEPS if s[1]]
    bad = [s for s in STEPS if not s[1]]
    parts = ["%s=%s" % (s[0], s[2]) for s in ok]
    txt = "性能保活 %s：%s" % ("✓" if not bad else "部分", "  ".join(parts))
    if bad:
        txt += "　|　未生效：" + "，".join("%s(%s)" % (s[0], s[2]) for s in bad)
    return txt
