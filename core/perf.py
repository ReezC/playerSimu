"""性能打点：把关键路径的耗时/频率记下来，定期写成一段能直接读的日志。

**这是给「性能评估」用的，不是行为统计**
    要回答的是：每帧的预算是多少、实际花了多少、余量还剩多少、尾延迟（p99/最大）
    有多坏、有没有超预算的帧、端到端延迟多大、资源有没有在涨。据此才能判断
    「哪里有风险」和「改哪里才有收益」。所以指标偏**耗时/吞吐/资源**。

**为什么内置**
    诊断「为什么慢 / 卡在哪一段 / 会不会撑不住」时，现场加 print 或另写脚本，
    既慢又常常错过现象那一瞬间。这里常开，把数据留在文件里，事后直接看。

**开销怎么控**（目标：常开着也不影响功能）
    · 一次打点 = 两次 `time.perf_counter()`（各几十纳秒）+ 一次环形缓冲写入；
    · 每个指标一个**固定大小**的环形缓冲（`RING` 个样本），内存不随时间增长；
    · 落盘默认每 `FLUSH_SEC` 秒追加一段汇总；**文件有上限** —— 超过 `MAX_BYTES`
      就截掉前半（保留最近一段历史），**不会无限增长**；
    · 进程资源（RSS/句柄）只在落盘那一刻取一次，不影响帧循环；
    · 关掉（`设置 → 性能日志`，落在 config/live.yaml 的 `perf_log`）之后，
      每条打点只剩一个 bool 判断。

**怎么读**（按性能评估的口径）
    每段：`n / 中位 / p95 / p99 / 最大 / 均值`，另有该段结束时的 RSS/句柄。
    ① 看**占比**：`infer_ms / pipe_ms` 就是推理在单帧处理里占多少 —— 优化只值得
       动占比大的那段；
    ② 看**尾延迟**：中位好看不算数，`p99 / 最大` 决定「会不会偶尔卡一下」；
    ③ 看**超预算**：`over_budget`（单帧处理时间超过帧间隔的帧数）与 `drop`
       （抓帧侧覆盖掉的帧）一涨，实时性就已经在丢；
    ④ 看**端到端**：`e2e_probe_ms`（A 机屏幕上的时间码 → 本机解码，走画面探针）
       才是控制质量的真实延迟；`out_key_ms`（取到帧 → 真的把键发出去）是本机这
       一段的处理+排期延迟；
    ⑤ 看**跨段趋势**：`rss_mb`/`handles` 单调上涨 = 泄漏；`gap_ms` 变大 = 链路恶化。
    指标名自带单位：`_ms` 毫秒、`*_fps` 每秒次数、`*_mb` 兆字节。

日志：仓库根 `perf.log`（和 crash.log / stdout.log 一起，`*.log` 已在 .gitignore）。
"""

import ctypes
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "perf.log"

RING = 256              # 每个指标保留最近多少样本（算分位数）
FLUSH_SEC = 30.0        # 每段多长
MAX_BYTES = 512 * 1024  # 日志超过它就把前半截掉（保留最近的一段历史）

#: 总开关：关着的时候所有打点都是「读一个 bool 就返回」。
ENABLED = False

_stats = {}
_flush_at = 0.0
_window_start = None
_segments = 0
_frame_t = None         # 最近一帧被取走的时刻（算「取帧 → 出键」用）
_notes = {}             # 这一段的环境注记（预算多少、什么设备…），写在段头


# ---------------------------------------------------------------- 进程资源
#
# 与 gui/stability.py 共用同一份实现（那边是手动点按钮，这里是每段落盘时自动采一次）。
# ctypes 的签名必须显式声明：`GetCurrentProcess()` 返回的伪句柄是 -1，默认 c_int
# 会把它截成 0xFFFFFFFF，和真正的 INVALID_HANDLE_VALUE 不是一个值，调用直接失败。

def _procs():
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    return k32.GetCurrentProcess(), k32, ctypes.windll.psapi


def rss_mb():
    """当前进程常驻内存（MB）；拿不到返回 None。"""
    try:
        class _PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        h, _k32, psapi = _procs()
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return None
        return pmc.WorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        return None


def handles():
    """进程当前的内核句柄数（socket / 文件 / 线程漏关都记在这）。"""
    try:
        h, k32, _psapi = _procs()
        k32.GetProcessHandleCount.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_ulong)]
        k32.GetProcessHandleCount.restype = ctypes.c_int
        n = ctypes.c_ulong()
        if not k32.GetProcessHandleCount(h, ctypes.byref(n)):
            return None
        return int(n.value)
    except Exception:
        return None


class Stat:
    """一个指标的固定容量环形缓冲 + 计数。"""
    __slots__ = ("n", "total", "mn", "mx", "buf", "i")

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.mn = None
        self.mx = None
        self.buf = [0.0] * RING
        self.i = 0

    def add(self, v):
        self.n += 1
        self.total += v
        self.mn = v if self.mn is None else (v if v < self.mn else self.mn)
        self.mx = v if self.mx is None else (v if v > self.mx else self.mx)
        self.buf[self.i] = v
        self.i = (self.i + 1) % RING

    def samples(self):
        return self.buf[:self.n] if self.n < RING else self.buf

    def percentile(self, q):
        s = sorted(self.samples())
        if not s:
            return 0.0
        return s[min(len(s) - 1, int(len(s) * q))]


def _stat(name):
    st = _stats.get(name)
    if st is None:
        st = _stats[name] = Stat()
    return st


# ---------------------------------------------------------------- 打点

def ms(name, t0):
    """记一段耗时：`t0` 是 `time.perf_counter()` 的起点（热路径就用这个）。"""
    if not ENABLED:
        return
    _stat(name).add((time.perf_counter() - t0) * 1000.0)


def sample(name, value):
    """记一个数值（帧间隔、每轮输出间隔、框数……）。"""
    if not ENABLED:
        return
    try:
        _stat(name).add(float(value))
    except (TypeError, ValueError):
        pass


def count(name, n=1):
    """记次数（调用次数、跳的次数、发键次数……）。"""
    if not ENABLED:
        return
    _stat(name).add(float(n))


def frame_arrived(t=None):
    """记下「最新一帧被取走」的时刻（实时线程每帧调一次，用来算控制延迟）。"""
    global _frame_t
    if not ENABLED:
        return
    _frame_t = time.perf_counter() if t is None else t


def since_frame(name):
    """从「最新一帧取到」到现在的毫秒数 —— 即本机这一段对**这一帧**的响应延迟。

    在真的把输出键发出去的地方调它（agent 的输出序列），得到的就是
    「拿到画面 → 出键」的延迟。它包含处理时间 + 排期等待，是控制质量的关键量：
    推理尖刺、排期拖后，都会直接体现在这里。
    """
    if not ENABLED or _frame_t is None:
        return
    _stat(name).add((time.perf_counter() - _frame_t) * 1000.0)


def note(key, value):
    """给这一段记一条环境注记（帧预算、设备、分辨率……），写在段头。

    有了它，日志本身就能自证「这次评估是在什么条件下降发生的」—— 预算不同、
    设备不同，同样的毫秒数含义完全不同。
    """
    if not ENABLED:
        return
    _notes[str(key)] = value


# ---------------------------------------------------------------- 生命周期

def set_enabled(on):
    """只翻开关，**不清已有统计** —— 给「设置」里改开关时用。

    和 `configure` 的区别：`configure` 是实时线程启动时调的，会把窗口/样本全部
    重置（等于重新开一段）；这里只是停止/开始打点，当前这一段的统计留着，
    改完开关不用重开实时预览。
    """
    global ENABLED
    ENABLED = bool(on)
    return ENABLED


def configure(on, log=None):
    """设置开关与日志路径（实时线程启动时调一次）。会清掉上一轮的数据。"""
    global ENABLED, LOG, _flush_at, _window_start, _segments, _frame_t
    ENABLED = bool(on)
    if log is not None:
        LOG = Path(log)
    _stats.clear()
    _notes.clear()
    _frame_t = None
    _window_start = None
    _flush_at = 0.0
    _segments = 0
    return ENABLED


def _render(span):
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    ctx = "  ".join("%s=%s" % (k, v) for k, v in sorted(_notes.items()))
    head = "=== %s  窗口 %.1fs" % (when, span)
    lines = ["", (head + "  " + ctx) if ctx else head]
    for name in sorted(_stats):
        st = _stats[name]
        if not st.n:
            continue
        lines.append("%-14s n=%-6d 中位 %8.2f  p95 %8.2f  p99 %8.2f  最大 %9.2f  均值 %8.2f"
                     % (name, st.n, st.percentile(0.5), st.percentile(0.95),
                        st.percentile(0.99), st.mx or 0.0, st.total / st.n))
    return "\n".join(lines) + "\n"


def _add_proc_gauges():
    """把进程 RSS / 句柄数记进本段（每段一个样本，看的是跨段趋势）。"""
    v = rss_mb()
    if v is not None:
        _stat("rss_mb").add(v)
    v = handles()
    if v is not None:
        _stat("handles").add(v)


def _trim():
    """文件太大就截掉前半（只在超限时发生，代价可忽略）。"""
    try:
        if LOG.stat().st_size <= MAX_BYTES:
            return
        blocks = LOG.read_text(encoding="utf-8").split("\n=== ")
        keep = blocks[len(blocks) // 2:]
        LOG.write_text("=== " + "\n=== ".join(keep), encoding="utf-8")
    except Exception:
        pass


def flush(now=None, force=False):
    """到点就把这一段落盘。**由调用方定期调**（实时面板每 0.5 秒一次那种）。"""
    global _flush_at, _window_start, _segments
    if not ENABLED:
        return False
    now = time.monotonic() if now is None else now
    if _window_start is None:
        _window_start = now
        _flush_at = now + FLUSH_SEC
        if not force:
            return False        # 非强制：第一次只用来起头，等下一个窗口再落盘
    if not force and now < _flush_at:
        return False

    span = now - _window_start
    _window_start = now
    _flush_at = now + FLUSH_SEC
    # 资源随每段一起落盘：只看耗时看不出「内存/句柄在涨」这类风险，而它们正是
    # 「挂了一夜之后变慢」的典型来源。落盘时才取一次，完全不进帧循环。
    _add_proc_gauges()
    text = _render(span)
    _stats.clear()          # 每段独立：清了才看得出「这一段」和「上一段」的差别
    _segments += 1
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(text)
        _trim()
    except Exception:
        pass
    return True


def summary(limit=10):
    """当前这一段里最慢的几个指标（一行一个），给界面显示用。"""
    rows = []
    for name in sorted(_stats, key=lambda k: -(_stats[k].percentile(0.95))):
        st = _stats[name]
        if st.n >= 3:
            rows.append("%-14s 中位 %8.2f  p95 %8.2f  n=%d"
                        % (name, st.percentile(0.5), st.percentile(0.95), st.n))
        if len(rows) >= limit:
            break
    return "\n".join(rows)
