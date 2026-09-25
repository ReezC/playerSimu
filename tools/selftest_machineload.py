"""本机负载清点自检：这是「推理为什么慢」的**第一个**判据，必须是能算的数。

**为什么要它**
    2026-09-25 实测：同一份权重、同一份 `imgsz=800`，机器干净时全图前向
    10.0~10.7 ms；而同机跑着 YOLO 训练的数据加载子进程、可用内存只剩 2.2 GB
    时是 23.7 ms —— 而 GPU 全程 P0 / 2692 MHz / 62.9 W，**不是降频**。
    所以「推理莫名变慢」第一个该看的是本机负载，不是模型，也不是 imgsz。

    这条判据有三件必须钉住的性质，正好对应下面三条用例：
      1. 数字得**算得出来**，且没有 psutil 时降级成「只报内存」而不是抛异常；
      2. **不能把自己数成占用者** —— 判定逻辑的源码里就写着 `spawn_main` /
         `gui.app` 这两个字面量，不排除自身的话，机器明明干净也会报告警
         （`tools/infer_probe.py` 第一版就在自己身上误报过，是真踩的坑）；
      3. 接线要在场：状态行显示它、工具复用同一份判据（两边各写一份迟早漂）。

跑法（不连流、不起 GUI）：

    python -m tools.selftest_machineload      # 全过返回 0，有失败返回 1
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import machineload                                # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: 告警要压在状态行最前面，太长会把帧率/推理耗时那些数挤出去
WARN_MAX_CHARS = 60


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


CLEAN = {"avail_gb": 15.0, "total_gb": 31.8, "workers": 0, "worker_mb": 0.0,
         "pids": [], "app_mb": 900.0, "has_psutil": True}


def t_probe_shape():
    """probe() 的键要齐、数值要自洽（GUI 与工具都按这些键取）。"""
    d = machineload.probe()
    for k in ("avail_gb", "total_gb", "workers", "worker_mb", "pids",
              "app_mb", "has_psutil"):
        check(k in d, "probe() 少了键 %s（GUI 和工具都按它取，不能少）" % k)
    if d["avail_gb"] is not None:
        check(d["total_gb"] and d["total_gb"] > 0, "总物理内存读不出来：%s" % d)
        check(0 < d["avail_gb"] <= d["total_gb"] + 0.01,
              "可用内存不在 (0, 总] 区间里：%s" % d)
    check(d["workers"] == len(d["pids"]), "子进程个数与 pid 列表对不上：%s" % d)
    check(d["worker_mb"] >= 0, "子进程占用不可能是负数")


def t_not_counting_itself():
    """**关键**：判定逻辑不能把本进程数成占用者。

    本文件自己的源码里带着 `spawn_main` 这个字面量（就在这个函数下面几行的
    字符串里），命令行把整份源码带上了 —— 不排除自身的话，机器干干净净也会
    报「有并行子进程」。
    """
    src = (ROOT / "core" / "machineload.py").read_text(encoding="utf-8")
    check("os.getpid()" in src,
          "core/machineload.py 不再排除本进程 —— 干净机器上也会误报告警")
    me = (ROOT / "tools" / "selftest_machineload.py").read_text(encoding="utf-8")
    check("spawn_main" in me,
          "这个用例失去意义了：本文件里已经没有会误导自己的字面量")
    d = machineload.probe()
    check(os.getpid() not in d["pids"],
          "把自己（pid %d）数成了并行子进程" % os.getpid())


def t_warn_is_numbers():
    """告警只看数：内存见底 / 有并行子进程；正常时必须一个字都不说。"""
    check(machineload.warn(CLEAN) == "",
          "干净机器上不该告警：%r" % machineload.warn(CLEAN))

    low = dict(CLEAN, avail_gb=2.2)
    w = machineload.warn(low)
    check(w and "内存" in w, "内存见底必须告警：%r" % w)

    wk = dict(CLEAN, workers=11, worker_mb=11.8 * 1024, pids=list(range(11)))
    w = machineload.warn(wk)
    check(w and "并行子进程" in w, "有并行子进程必须告警：%r" % w)

    both = machineload.warn(dict(low, workers=11, worker_mb=11.8 * 1024))
    check("内存" in both and "并行子进程" in both,
          "两条都占的时候要一起报出来：%r" % both)

    # 边界：阈值上不算见底（判据是「小于」阈值，不是「小于等于」）
    edge = machineload.warn(dict(CLEAN, avail_gb=machineload.AVAIL_MIN_GB))
    check(edge == "",
          "刚好等于阈值（%.1f）不该告警：%r" % (machineload.AVAIL_MIN_GB, edge))

    # 内存读不到时：既不崩，也不能凭空告警
    check(machineload.warn(dict(CLEAN, avail_gb=None, total_gb=None)) == "",
          "读不到内存时不该告警")
    check(machineload.warn({}) == "", "空字典不该崩，也不该告警")


def t_warn_fits_status_line():
    """告警必须短 —— 它压在那条已经塞满的状态行最前面。"""
    d = dict(CLEAN, avail_gb=2.2, workers=19, worker_mb=19 * 1024)
    w = machineload.warn(d)
    check(len(w) <= WARN_MAX_CHARS,
          "告警 %d 字太长（上限 %d），会把帧率/推理耗时挤出状态行：%r"
          % (len(w), WARN_MAX_CHARS, w))


def t_size_unit_by_magnitude():
    """占用要按量级选单位 —— 别出现「并行子进程 1 个(0.0GB)」这种等于没写的数。"""
    check(machineload.fmt_mb(11.8 * 1024) == "11.8GB",
          "GB 量级读错了：%s" % machineload.fmt_mb(11.8 * 1024))
    check(machineload.fmt_mb(940) == "940MB",
          "MB 量级读错了：%s" % machineload.fmt_mb(940))
    check(machineload.fmt_mb(1023) == "1023MB",
          "边界（1023 MB）应当是 MB：%s" % machineload.fmt_mb(1023))
    check(machineload.fmt_mb(1024) == "1.0GB",
          "边界（1024 MB）应当是 GB：%s" % machineload.fmt_mb(1024))
    w = machineload.warn(dict(CLEAN, workers=1, worker_mb=47, pids=[1]))
    check("MB" in w and "0.0GB" not in w,
          "刚起来的单个小进程被读成了 0.0GB：%r" % w)


def t_detail_has_next_step():
    """明细不能只把数字重复一遍，要给「所以该干什么」。"""
    bad = dict(CLEAN, avail_gb=2.2, workers=11, worker_mb=11.8 * 1024,
               pids=[1, 2, 3])
    t = machineload.detail(bad)
    check("2.2" in t and "11" in t, "明细里没把关键数字写出来：%r" % t)
    check("imgsz" in t,
          "明细没写明「别去动 imgsz / 模型」—— 那正是最容易走错的一步")
    check("imgsz" not in machineload.detail(CLEAN),
          "干净机器上不该提示去动模型")
    check(machineload.detail({}) == "", "空字典不该崩")


def t_degrades_without_psutil():
    """没有 psutil 也要能报内存（部署机不一定装得上）。"""
    if not machineload.probe()["has_psutil"]:
        return                      # 本机没装：probe() 本身已经跑通了降级路径
    import builtins
    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "psutil":
            raise ImportError("模拟没装 psutil")
        return real(name, *a, **kw)

    builtins.__import__ = fake
    try:
        d = machineload.probe()
    finally:
        builtins.__import__ = real
    check(d["has_psutil"] is False, "没有 psutil 时要标记 has_psutil=False")
    check(d["workers"] == 0 and d["pids"] == [],
          "没有 psutil 时不许编出子进程：%s" % d)
    check(d["avail_gb"] is not None,
          "没有 psutil 也必须能报内存（那是 ctypes 直接读的，不依赖它）")
    check(machineload.warn(d) == "", "只剩内存且内存充足时不该告警")


def t_census_does_not_read_all_memory():
    """清点**不许对每个进程取内存** —— 那一下会把同进程的推理拖到几百毫秒。

    **为什么它排在前面**（2026-09-25 现场踩到的真事故）：改之前 `_census()` 用
    `process_iter([..., "memory_info"])` 枚举全部 360 个进程 —— 本机量到
    **1.3 秒**（光取 RSS 就 ~1.0 秒，而 名字+命令行 只要 45ms）。代价还不只是 CPU：
    这 1.3 秒的 psutil 风暴和**同进程**里的 `model.predict` 撞在一起时，推理的墙钟
    耗时会被抬到 **400~2500ms**（实测：单独 predict 最大 28ms；加上每 10 秒一次
    probe() 之后，60 秒里出现 6 次超 200ms，而且**每次都卡在 probe 刚启动那一刻**）。
    现象就是工作台「不间断地卡顿一下」—— 查了半天才查到是自己加的诊断干的。

    判据用**调用次数**、不用耗时：耗时在忙的机器上会飘，次数很稳。
    """
    import psutil
    import unittest.mock as mock

    # 判据一（主）：**枚举时不许要 memory_info**。这是那条 bug 的契约本身 ——
    # 一旦塞回去，psutil 就会对每个进程取一遍 RSS。
    # 为什么不去数 memory_info 的调用次数（试过两种写法都不行）：psutil 内部
    # 那条路会绕过 `psutil.Process.memory_info`（实测：枚举列表里明明带着
    # memory_info、probe() 也确实涨到 1420ms，计数却只有 1 次），而用 MagicMock
    # 换方法更糟 —— 它不是描述符，psutil 按实例取属性时拿到类上那个 mock、
    # 调用不带 self，直接抛 TypeError，用例"通过"了其实什么都没验。
    # 断言自己传出去的那份契约，最准。
    seen = []
    real_pi = psutil.process_iter

    def spy(attrs=None, *a, **kw):
        seen.append(list(attrs or []))
        return real_pi(attrs, *a, **kw)

    total = len(psutil.pids())
    t0 = time.perf_counter()
    with mock.patch.object(psutil, "process_iter", spy):
        machineload.probe()
    ms = (time.perf_counter() - t0) * 1000.0
    print("      （本机 %d 个进程，枚举字段 %s，probe() %.0f ms）"
          % (total, seen, ms))
    for heavy in ("memory_info", "cmdline"):
        bad = [a for a in seen if heavy in a]
        check(not bad,
              "清点把 %s 塞进 process_iter 的枚举列表了（%s）—— 那是**对全机 "
              "%d 个进程**逐个取：memory_info ~1.0 秒（会把同进程的推理拖到 "
              "400~2500ms）、cmdline ~45ms（每个都要读对方的 PEB）。"
              "先用便宜的字段筛出 python 候选，再对候选逐个取贵的。"
              % (heavy, bad, total))
    # 判据二（粗）：万一以后换个写法绕开这条契约，耗时也会露馅。
    # 改前实测 1308~1420ms，改后 56~70ms，留 10 倍余量仍能抓住。
    check(ms < 800.0,
          "清点一次用了 %.0f ms（改前 ~1300ms、改后 ~60ms）—— 太慢就是又去"
          "枚举全机内存了" % ms)


def t_wired_into_gui_and_tool():
    """接线：状态行显示它、采在单独线程、工具复用同一份判据。"""
    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("machineload" in lt and '"load_warn"' in lt,
          "实时线程没把负载告警带上统计（状态行就没得显示）")
    check("Thread(target=_load_watch" in lt and "daemon=True" in lt,
          "负载清点必须在**单独线程**里采 —— 放进主回路就是每 10 秒卡一帧，"
          "正好污染我们在盯的 infer_ms / pipe_ms")
    check("self._stop.wait(LOAD_WATCH_SEC)" in lt,
          "守望线程要用停止事件等待，否则点「停止」会白等一个采集周期")
    check("负载清点失败" in lt,
          "守望线程里的异常又被静默吞了 —— 清点一直失败却什么都不显示，"
          "等于这个功能根本没做（而它正是「推理为什么慢」的唯一解释）")

    lp = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("load_warn" in lp, "状态行没接负载告警")
    check("load_detail" in lp, "明细应当进 tooltip，别硬挤进状态行")

    probe = (ROOT / "tools" / "infer_probe.py").read_text(encoding="utf-8")
    check("from core import machineload" in probe,
          "tools/infer_probe.py 没有复用 core.machineload —— 两边判据会漂")
    check("_python_census" not in probe and "def _mem_gb" not in probe,
          "tools/infer_probe.py 又自己写了一份清点（判据就有两份了）")

    rep = (ROOT / "tools" / "perf_report.py").read_text(encoding="utf-8")
    for k in ("avail_gb", "mp_workers"):
        check(k in rep,
              "perf_report 的 ROLE 里没给 %s 写说明 —— 日志里出现时没人看得懂" % k)


TESTS = (
    ("probe() 键齐全、数值自洽", t_probe_shape),
    ("不把自己数成占用者", t_not_counting_itself),
    ("告警只看数：内存见底 / 有并行子进程", t_warn_is_numbers),
    ("告警短到能塞进状态行", t_warn_fits_status_line),
    ("占用按量级选单位（不出现 0.0GB）", t_size_unit_by_magnitude),
    ("明细给出下一步（别去动 imgsz）", t_detail_has_next_step),
    ("没有 psutil 时降级成只报内存", t_degrades_without_psutil),
    ("清点不许对所有进程取内存（会拖住推理）", t_census_does_not_read_all_memory),
    ("接线：状态行 + 单独线程 + 工具复用同一份", t_wired_into_gui_and_tool),
)


def main() -> int:
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
