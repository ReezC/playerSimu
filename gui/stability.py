"""稳定性自检：长时间开自动之后，一眼看出「有没有东西在涨」。

**为什么需要它**
    内存 / 句柄 / 线程的泄漏不会报错，只会让程序越跑越慢，最后卡顿或吃满内存。
    任务管理器能看内存，但看不到「内核句柄数」「线程数」「Python 对象数」——
    而它们恰恰是「挂了一夜之后不对劲」的常见原因（漏关的 socket、没退出的线程、
    只增不减的容器都各自记在某一项上）。

**怎么用**
    第一次点「稳定性自检」= 记基线（只显示当前值）；
    隔十几分钟再点一次 = 给出各项的变化量。**一次涨完然后走平是正常的**
    （模型预热、缓存），持续单调上涨才是泄漏。

**开销**：只读，毫秒级。**不启 tracemalloc** —— 那个能让程序慢一倍，不适合常开。
拿不到某项（非 Windows / 系统调用失败）就显示「—」，不影响其他项。
"""

import gc
import threading
import time

from core import perf

_START = time.monotonic()
_last = None            # 上一次快照（同一进程内前后对比用）

#: 各项超过这些变化量就提示一句（不是判死，只是让你别忽略）
_LIMITS = {"rss_mb": 50.0, "handles": 50, "threads": 2, "gc_objects": 5000}

# RSS / 句柄的取法只有一份实现（core/perf.py）—— 那边每段落盘时也自动采一次，
# 这里只是手动点按钮时用同一份。以前这里是复制的一份 ctypes 代码。
_rss_mb = perf.rss_mb
_handles = perf.handles


def snapshot():
    """取一份当前状态（纯读、无副作用）。"""
    from decision.agent import settings
    return {
        "t": time.monotonic(),
        "rss_mb": _rss_mb(),
        "handles": _handles(),
        "threads": threading.active_count(),
        "thread_names": sorted(t.name for t in threading.enumerate()),
        "gc_objects": len(gc.get_objects()),
        "定时行为": len(settings.custom_timers),
        "定时行为计时项": len(settings.custom_timer_next),
        "自定义按键": len(settings.custom_keys or {}),
    }


def _fmt(v, unit=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return "%.1f%s" % (v, unit)
    return "%d%s" % (v, unit)


def _fmt_delta(k, d):
    if d is None:
        return "—"
    if isinstance(d, float):
        return "%+.1f" % d
    return "%+d" % d


def check():
    """取快照；有基线就给出对比。

    返回 `(详细文本, 需要留意的行 list, 是不是第一次记基线)`。
    """
    global _last
    cur = snapshot()
    prev, _last = _last, cur

    up = time.monotonic() - _START
    head = "稳定性自检  ·  本进程已运行 %d 时 %d 分" % (up // 3600,
                                                        (up % 3600) // 60)
    keys = [("rss_mb", "常驻内存", " MB"),
            ("handles", "内核句柄数", ""),
            ("threads", "线程数", ""),
            ("gc_objects", "Python 对象数", ""),
            ("定时行为", "定时行为", ""),
            ("定时行为计时项", "定时行为计时项", ""),
            ("自定义按键", "自定义按键", "")]

    if prev is None:
        rows = ["%s：%s" % (label, _fmt(cur[k], unit)) for k, label, unit in keys]
        text = "\n".join([head, "", "已记下基线：", ""] + rows + [
            "",
            "隔十几分钟再点一次就能看出涨没涨。",
            "一次涨完然后走平是正常的（模型预热、缓存），持续单调上涨才要留意。",
            "线程：%s" % ", ".join(cur["thread_names"])])
        return text, [], True

    mins = max(0.1, (cur["t"] - prev["t"]) / 60.0)
    warn = []
    lines = [head, "",
             "%-14s %12s %12s %12s" % ("", "现在", "上次", "变化"),
             "-" * 54]
    for k, label, unit in keys:
        v, p = cur[k], prev[k]
        d = None if (v is None or p is None) else v - p
        lines.append("%-14s %12s %12s %12s"
                     % (label, _fmt(v, unit), _fmt(p, unit), _fmt_delta(k, d)))
        lim = _LIMITS.get(k)
        if lim is not None and d is not None and d > lim:
            warn.append("%s 涨了 %s%s（%.1f 分钟内）—— 值得留意"
                        % (label, _fmt_delta(k, d), unit, mins))

    add = sorted(set(cur["thread_names"]) - set(prev["thread_names"]))
    gone = sorted(set(prev["thread_names"]) - set(cur["thread_names"]))
    if add:
        warn.append("多了线程：%s" % ", ".join(add))
    if len(add) > 1 or (add and len(cur["thread_names"]) > 12):
        warn.append("线程只增不减通常是线程泄漏（见 stability.py 顶部说明）")

    lines += ["", "间隔 %.1f 分钟；线程：%s" % (mins, ", ".join(cur["thread_names"]))]
    if gone:
        lines.append("（少了线程：%s）" % ", ".join(gone))
    lines += ["", "判断方法：隔十几分钟再点一次。一次涨完然后走平是正常的"
                  "（模型预热、缓存），各项持续单调上涨才是泄漏。"]
    return "\n".join(lines), warn, False
