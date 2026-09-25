"""本机负载清点：谁在跟实时推理抢这台机器。

**为什么要有它**（2026-09-25 实测定下来的根因，不是猜的）
    同一份权重、同一份 `imgsz=800` 配置，在 B 机上量到过两种结果：

        · 机器干净（预览与训练都停、可用内存 15 GB）→ 全图前向 10.0~10.7 ms
        · 同机跑着 YOLO 训练的数据加载子进程、可用内存只剩 2.2 GB → 23.7 ms

    而 GPU 全程 P0 / 2692 MHz / 62.9 W / 52℃（`clocks_throttle_reasons=0x0`）——
    **不是降频**；ultralytics 自报的 `inference` 也与输入尺寸无关（64×64 也要
    8~12 ms）。也就是说被放大的是**每一层的固定开销**，不是算力不够。
    所以「推理莫名变慢」第一个该看的是本机负载，不是模型，也不是 imgsz。

判据（都是能算的数，不是体感）
    · 可用物理内存 < 4 GB                 → 见底，会把每次 op 的固定开销整体放大
    · 有 `multiprocessing.spawn` 子进程   → 训练取数 / 并行标注，每个 ~1 GB 且一直在吃 CPU

    **GPU 利用率不在这里查**：`nvidia-smi` 一次要 100~200 ms，而负载清点要每
    10 秒跑一次、结果还要进状态行 —— 那点开销会把我们要量的东西自己污染掉。
    要看 GPU 就用 `python -m tools.infer_probe`（它是一次性的旁路工具）。

用法（**只给这一份入口**，GUI 与命令行工具都走它，别再各写一份）
    d = probe()          # 采一次，几十毫秒（没有 psutil 时自动降级成「只报内存」）
    warn(d)              # 没问题返回 ""，有问题返回一句给状态行用的话
    detail(d)            # 体检明细，给 tooltip 用

只读：不改配置、不写日志、不碰网络。
"""

import ctypes
import os

#: 可用物理内存低于它就算「见底」。
#: 4 GB 这个数留了余量：实测 2.2 GB 时前向从 10.5 拖到 23.7 ms，而 15 GB 时正常。
AVAIL_MIN_GB = 4.0


def _mem_gb():
    """→ (可用, 总) 物理内存 GB。用 ctypes，不引第三方依赖（永远能报出来）。"""
    class _MSX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        st = _MSX()
        st.dwLength = ctypes.sizeof(st)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullAvailPhys / 1073741824.0, st.ullTotalPhys / 1073741824.0
    except Exception:
        return None


def _census():
    """列出本机 python 进程；没有 psutil 返回 None（调用方按「清不出来」处理）。

    `multiprocessing.spawn` 出来的子进程命令行里都带 `spawn_main`
    （`--multiprocessing-fork` 那一串），训练取数与并行标注都是它开的。

    **必须排除本进程**：调用方的源码里就写着 "spawn_main" / "gui.app" 这两个
    字面量，命令行里带着整份源码 —— 不排除的话，机器明明很干净也会把自己数成
    「一个工作台 + 一批子进程」。（这个坑在 tools/infer_probe.py 上真踩过。）

    **绝不要对所有进程取 `memory_info()`**（2026-09-25 现场踩到）：
        · 代价：这台机器上 360 个进程取一遍 RSS 要 **~1.0 秒**（每个 ~2.6ms），
          而只取 名字+命令行 只要 **45ms**；
        · **更严重的是它有副作用**：这 1.3 秒的 psutil 风暴和**同进程**里的
          `model.predict` 撞在一起时，推理的墙钟耗时会从 ~14ms 被抬到
          **400~2500ms**（实测复现：单独 predict 最大 28ms；加上每 10 秒一次
          probe() 之后，60 秒里出现 6 次 400~2510ms，而且**每次都卡在 probe
          刚启动那一刻**）。现象就是工作台「不间断地卡顿一下」。
    所以改成：先按 名字+命令行 挑出可疑的，**只对那几个取内存**。
    """
    try:
        import psutil
    except Exception:
        return None
    me = os.getpid()
    # 第一遍：**只要 名字**（最便宜）。顺带说明为什么分三遍：
    #   · 全机 359 个进程取 cmdline 要 ~45ms —— 每个都要读一遍对方的 PEB；
    #   · 取 memory_info 要 ~1.0s（上面那段注释里的事故就是它）。
    #   而我们真正关心的只有 python 那几个（本机通常 1~3 个），
    #   所以先用最便宜的字段筛出候选，再对候选逐个取贵的字段。
    cand = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            nm = (p.info.get("name") or "").lower()
            if nm.startswith("python") and p.info["pid"] != me:
                cand.append(p.info["pid"])
        except Exception:
            continue        # 进程刚好退出 / 权限不够：跳过，不影响其它
    out = []
    for pid in cand:
        try:
            cl = " ".join(psutil.Process(pid).cmdline() or [])
        except Exception:
            cl = ""
        out.append({"pid": pid,
                    "worker": "spawn_main" in cl,
                    "app": "gui.app" in cl,
                    "mb": 0.0})
    # 内存只对**疑似占用者**取（通常 0~2 个）
    for r in out:
        if r["worker"] or r["app"]:
            try:
                r["mb"] = psutil.Process(r["pid"]).memory_info().rss / 1048576.0
            except Exception:
                pass
    return out


def probe():
    """采一次本机负载。

    键固定这几个（GUI 与工具都按它取，别各取各的）：
        avail_gb / total_gb    可用、总物理内存（GB）；读不到是 None
        workers / worker_mb    并行子进程个数、合计占用（MB）
        pids                   那些子进程的 pid（排查时要能指着看）
        app_mb                 工作台自身常驻（MB），拿不到是 None
        has_psutil             进程清点是否可用（False = 只有内存那两项可信）
    """
    mem = _mem_gb()
    rows = _census()
    wk = [] if rows is None else [r for r in rows if r["worker"]]
    app = next((r for r in (rows or []) if r["app"]), None)
    return {
        "avail_gb": mem[0] if mem else None,
        "total_gb": mem[1] if mem else None,
        "workers": len(wk),
        "worker_mb": sum(r["mb"] for r in wk),
        "pids": [r["pid"] for r in wk],
        "app_mb": app["mb"] if app else None,
        "has_psutil": rows is not None,
    }


def fmt_mb(mb):
    """MB → 「11.8GB」/「940MB」。公开给命令行工具一起用。

    按量级选单位：并行标注的子进程是 ~1.1 GB/个，而刚起来的进程只有几十 MB ——
    一律写成 GB 就会出现「并行子进程 1 个(0.0GB)」这种看了等于没看的数。
    """
    return "%.1fGB" % (mb / 1024.0) if mb >= 1024 else "%dMB" % mb


def warn(d):
    """→ 给状态行用的一句话；没问题时是空串。

    **必须短**：它要塞进已经很长的那条状态行最前面，长了会把帧率/推理耗时那
    几个数挤出去 —— 而那些数正是要看的东西。
    """
    if not d:
        return ""
    bits = []
    av, tot = d.get("avail_gb"), d.get("total_gb")
    if av is not None and av < AVAIL_MIN_GB:
        bits.append("内存 %.1f/%.1fGB" % (av, tot))
    if d.get("workers"):
        bits.append("并行子进程 %d 个(%s)"
                    % (d["workers"], fmt_mb(d.get("worker_mb", 0.0))))
    return ("负载告警：" + "、".join(bits)) if bits else ""


def detail(d):
    """体检明细，给 tooltip 用（一行一件事，最后告诉人该干什么）。"""
    if not d:
        return ""
    lines = []
    av, tot = d.get("avail_gb"), d.get("total_gb")
    if av is not None:
        lines.append("可用物理内存 %.1f / %.1f GB" % (av, tot))
    if not d.get("has_psutil"):
        lines.append("没装 psutil —— 只能看内存，清不出并行子进程"
                     "（pip install psutil）")
    elif d.get("workers"):
        pids = ", ".join(str(p) for p in d.get("pids", [])[:8])
        more = " 等" if len(d.get("pids", [])) > 8 else ""
        lines.append("并行子进程 %d 个（合计 %s；pid %s%s）"
                     % (d["workers"], fmt_mb(d.get("worker_mb", 0.0)), pids, more))
    else:
        lines.append("没有并行子进程（训练取数 / 并行标注都会开出它们）")
    if d.get("app_mb") is not None:
        lines.append("工作台自身常驻 %.0f MB" % d["app_mb"])
    if warn(d):
        lines += [
            "",
            "这些会把**每次 op 的固定开销**整体放大：推理的前向耗时与输入尺寸",
            "无关地被拖慢（同一份配置实测 10.0 → 23.7 ms），而 GPU 的时钟、功耗",
            "都正常 —— 所以**不是降频、也不是模型的问题**。",
            "等它们结束再复测；别为此去动 imgsz 或换模型。",
        ]
    return "\n".join(lines)
