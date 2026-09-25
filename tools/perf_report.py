"""性能评估报告：读 perf.log（core/perf.py 落盘的），按性能口径给出结论。

**它回答什么**（不是行为统计）
    · 余量：单帧预算多少、实际花多少、还剩多少
    · 占比：单帧处理时间花在哪一段（谁占大头，才值得优化谁）
    · 尾延迟：中位好看不算数，p95/p99/最大决定「会不会偶尔卡」
    · 尖刺：逐段普查「某一帧超过 1 秒」的卡顿，并区分两类 ——
        · `infer_ms` 也跟着大 → 本机被挂起（系统睡眠 / GPU 掉 / 被抢占）
        · `infer_ms` 正常、只有 `gap_ms` 大 → 是**输入**停了（A 机推流 / 网络）
    · 退化：前 1/4 段 vs 后 1/4 段的中位（并剔掉最差 2 段复核，防止被尖刺带偏）
    · 资源：RSS / 句柄跨段有没有单调上涨（泄漏）

用法：
    python -m tools.perf_report                 # 读仓库根 perf.log
    python -m tools.perf_report D:\\path\\perf.log
"""

import argparse
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 认为「卡了一下」的阈值（毫秒）：超过它就逐段点出来
STALL_MS = 1000.0

#: 各阶段的角色说明（打印时带上，报告自解释）
ROLE = {
    "infer_ms": "推理（模型前向）",
    "pipe_ms": "单帧处理总耗时（取帧→决策完）",
    "agent_ms": "决策（规则/状态机）",
    "timing_ms": "时序拍（推进输出序列）",
    "gap_ms": "帧到达间隔（= 输入速度）",
    "draw_ms": "画框（只在显示帧）",
    "send_ms": "发键（通道写）",
    "e2e_probe_ms": "端到端延迟（A机屏幕时间码→本机解码）",
    "out_key_ms": "取帧→出键（本机处理+排期）",
    "over_budget": "单帧处理超过预算的次数",
    "drop": "抓帧侧被覆盖掉的帧数",
    "recv_fps": "链路输入速度",
    "proc_fps": "实际处理速度",
    "rss_mb": "常驻内存",
    "handles": "内核句柄数",
    "boxes": "每帧框数（负载）",
    # 本机负载（core/machineload.py 每 10 秒采一次）。有它们才能事后对账：
    # 「那段时间推理为什么慢」= 把 infer_ms 和 avail_gb / mp_workers 对齐着看。
    "avail_gb": "可用物理内存（GB）——低于 4 会拖慢每次 op",
    "mp_workers": "并行子进程数（训练取数/并行标注）",
    "mp_worker_mb": "并行子进程占用（MB）",
    # 小地图定位（S3，perception/minimap.PlayerLocator）：这两个是**每帧计数**，
    # 比值 = 定位成功率。掉了先看世界坐标那行写的 reason（没标定/没认出黄点）。
    "mmap_ok": "小地图定位成功（帧）",
    "mmap_miss": "小地图定位失败（帧）",
}


#: 日志里的中文列名 → 内部字段名（新版多一列 p99，旧版没有 —— 都要能读）
_COLS = {"中位": "p50", "p95": "p95", "p99": "p99", "最大": "max", "均值": "mean"}


def parse(path):
    """→ [{'t': 时间戳, 'notes': {...}, 'm': {指标: {...}}}]"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    segs = []
    for raw in text.split("\n=== "):
        lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith("==="):          # 文件首块可能带前导 "==="
            lines[0] = lines[0][3:].strip()
        m = re.match(r"(\S+ \S+)\s+窗口 ([\d.]+)s(.*)", lines[0])
        notes = {}
        if m:
            for kv in m.group(3).split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    # 只认正经的键名，别把 "===" 这类残留当注记
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
                        notes[k] = v
        seg = {"t": m.group(1) if m else lines[0], "notes": notes, "m": {}}
        for ln in lines[1:]:
            mm = re.match(r"^(\S+)\s+n=(\d+)\s+(.*)$", ln)
            if not mm:
                continue
            nums = {}
            for label, val in re.findall(r"(\S+)\s+([\d.]+)", mm.group(3)):
                if label in _COLS:
                    try:
                        nums[_COLS[label]] = float(val)
                    except ValueError:
                        pass
            seg["m"][mm.group(1)] = {"n": int(mm.group(2)), **nums}
        if seg["m"]:
            segs.append(seg)
    return segs


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))] if s else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="按性能口径评估 perf.log")
    ap.add_argument("path", nargs="?",
                    default=str(ROOT / "perf.log"), help="perf.log 路径")
    ap.add_argument("--stall-ms", type=float, default=STALL_MS,
                    help="判定「卡了一下」的阈值（毫秒）")
    ap.add_argument("--since", default=None, metavar="HH:MM",
                    help="只看这个时刻之后的段（当天内比较，便于排除旧数据/自己测试的干扰）")
    ap.add_argument("--last", type=float, default=None, metavar="MIN",
                    help="只看最后 N 分钟")
    args = ap.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print("找不到 %s —— 先在实时面板打开「性能日志」，跑一段再看。" % path)
        return 2
    segs = [s for s in parse(path) if s["m"]]
    if not segs:
        print("%s 里没有可解析的段。" % path)
        return 2

    def _secs(t):
        """段头里的当天时刻 → 秒（跨天不处理：同一天内的日志足够用）。"""
        p = t[11:].split(":")
        return int(p[0]) * 3600 + int(p[1]) * 60 + (int(p[2]) if len(p) > 2 else 0)

    total = len(segs)
    if args.since:
        cut = args.since if ":" in args.since else args.since + ":00"
        hh, mm = cut.split(":")[:2]
        cut = "%s:%s" % (hh.zfill(2), mm.zfill(2))
        segs = [s for s in segs if s["t"][11:16] >= cut]
    if args.last:
        end = _secs(segs[-1]["t"]) if segs else 0
        segs = [s for s in segs if _secs(s["t"]) >= end - args.last * 60]
    if not segs:
        print("按这个时间条件筛完没有段了（原 %d 段）。" % total)
        return 2

    print("perf.log 性能评估：%d 段，%s → %s%s"
          % (len(segs), segs[0]["t"], segs[-1]["t"],
             "" if len(segs) == total else "（从 %d 段里筛出）" % total))
    if segs[-1]["notes"]:
        print("运行条件：" + "  ".join("%s=%s" % kv
                                       for kv in sorted(segs[-1]["notes"].items())))

    def col(name, field):
        return [s["m"][name][field] for s in segs
                if name in s["m"] and field in s["m"][name]]

    # ---- 1. 各阶段 ----
    print()
    print("== 1. 各阶段（段中位的分布；p95/最大取段内最坏）==")
    print("   %-14s %8s %8s %8s | %8s | %9s" %
          ("指标", "最好段", "中位段", "最坏段", "p95最坏", "尖刺最大"))
    for name in ("infer_ms", "pipe_ms", "agent_ms", "timing_ms", "gap_ms",
                 "draw_ms", "send_ms", "e2e_probe_ms", "out_key_ms",
                 "recv_fps", "proc_fps", "boxes", "rss_mb", "handles",
                 "over_budget", "drop"):
        p50, p95, mx = col(name, "p50"), col(name, "p95"), col(name, "max")
        if not p50:
            continue
        print("   %-14s %8.2f %8.2f %8.2f | %8.2f | %9.2f   %s"
              % (name, min(p50), st.median(p50), max(p50), max(p95), max(mx),
                 ROLE.get(name, "")))

    # ---- 1b. 计数类 ----
    #
    # 计数类（`count()` 记的）**百分位没有意义** —— 每次贡献的值都是 1，所以
    # 中位/p95/最大 全是 1，看不出发生了多少次。要的是「每段总计」= n × 均值。
    print()
    print("== 1b. 计数类（每段总计）==")

    def _tot(seg, name):
        d = seg["m"].get(name) or {}
        n, m = d.get("n"), d.get("mean")
        return n * m if (n and m is not None) else None

    # send / send_fail 是**指令通道**的收发次数：失败率一涨就是链路（relay/串口）出问题，
    # 比看界面那行字可靠 —— 界面只在「连接那一刻」写过字的时候会骗人。
    for name in ("over_budget", "drop", "timing_calls", "out_round",
                 "send", "send_fail",
                 "probe_ok", "probe_miss", "probe_reject"):
        tot = [v for v in (_tot(s, name) for s in segs) if v is not None]
        if not tot:
            continue
        print("   %-14s 每段 中位 %7.1f 次 / 最大 %7.1f 次（%d 段有数据）"
              % (name, st.median(tot), max(tot), len(tot)))

    # 探针健康度：e2e_probe_ms 只有在解码成功率够高时才可信 —— 失败时样本变少、
    # 读出来会偏小，看着像「延迟变好了」。
    rate = []
    for s in segs:
        # 用 or 0.0 而不是要求两者都在：**全失败**（一个 ok 都没有）恰恰是最需要
        # 提示「读数不可信」的情况，而那时 probe_ok 这个指标根本不存在。
        ok = _tot(s, "probe_ok") or 0.0
        miss = _tot(s, "probe_miss") or 0.0
        if ok + miss > 0:
            rate.append(ok / (ok + miss))
    if rate:
        med = st.median(rate)
        print("   探针解码成功率 每段 中位 %.0f%%%s"
              % (100 * med,
                 "   ← 偏低：e2e_probe_ms 读数不可信（探针位置/位数/A机脚本）"
                 if med < 0.9 else ""))

    # ---- 2. 余量 ----
    gap = st.median(col("gap_ms", "p50")) if col("gap_ms", "p50") else 0.0
    pip = st.median(col("pipe_ms", "p50")) if col("pipe_ms", "p50") else 0.0
    inf = st.median(col("infer_ms", "p50")) if col("infer_ms", "p50") else 0.0
    agt = st.median(col("agent_ms", "p50")) if col("agent_ms", "p50") else 0.0
    if gap and pip:
        print()
        print("== 2. 余量核算（段中位）==")
        print("   单帧预算（帧到达间隔） %7.1f ms" % gap)
        print("   单帧实际处理           %7.1f ms  → 占 %.0f%%，余量 %.1f ms"
              % (pip, 100.0 * pip / gap, gap - pip))
        if pip:
            print("   ├ 推理                 %7.1f ms  → 占处理 %.0f%%%s"
                  % (inf, 100.0 * inf / pip,
                     "   ← 唯一大块，优化只值得动它" if inf / pip > 0.6 else ""))
            print("   ├ 决策                 %7.1f ms  → 占处理 %.1f%%"
                  % (agt, 100.0 * agt / pip))
            print("   └ 追踪/地形/组装（推定）%7.1f ms" % max(0.0, pip - inf - agt))

    # ---- 3. 尾延迟 ----
    print()
    print("== 3. 尾延迟 vs 预算 ==")
    # 只对「本机单帧处理耗时」比预算。
    #   gap_ms 是输入间隔本身，拿它比自己没意义；
    #   out_key_ms 是「延迟」不是「每帧计算量」（含排期），拿帧预算比必然全超。
    for name in ("pipe_ms", "infer_ms", "draw_ms", "timing_ms"):
        p95, mx = col(name, "p95"), col(name, "max")
        if not p95 or not mx:
            continue
        over = sum(1 for v in p95 if gap and v > gap)
        print("   %-14s p95 超预算的段 %2d/%d ；单段最大 最大 %8.1f ms"
              % (name, over, len(p95), max(mx)))

    # ---- 4. 尖刺普查（区分「本机被挂起」和「输入停了」）----
    print()
    print("== 4. 尖刺普查（段内最大 > %.0f ms）==" % args.stall_ms)
    for name in ("gap_ms", "infer_ms", "pipe_ms", "agent_ms", "timing_ms",
                 "draw_ms", "send_ms", "out_key_ms", "e2e_probe_ms"):
        bad = [(s["t"][-8:], s["m"][name]["max"]) for s in segs
               if name in s["m"] and s["m"][name].get("max", 0) > args.stall_ms]
        if bad:
            print("   %-14s %d 处：%s" % (name, len(bad),
                                         "  ".join("%s %.0fms" % b
                                                   for b in bad[:5])))
    gbad = {s["t"] for s in segs if s["m"].get("gap_ms", {}).get("max", 0) > args.stall_ms}
    ibad = {s["t"] for s in segs if s["m"].get("infer_ms", {}).get("max", 0) > args.stall_ms}
    both, only_in = gbad & ibad, gbad - ibad
    if both:
        print("   → 本机被挂起（gap 与 infer 同时大，墙钟跨过去了）：%s"
              % "  ".join(sorted(t[-8:] for t in both)))
    if only_in:
        print("   → 输入停了（只有 gap 大、infer 正常，说明是 A 机推流/网络）：%s"
              % "  ".join(sorted(t[-8:] for t in only_in)))

    # ---- 5. 退化 ----
    #
    # **必须保持时间序**。「剔掉最差 N 段」要按索引剔，不能先 sorted 再切片 ——
    # 那样切片出来的根本不是前后两段时间，判出来的「退化」是假的（这里踩过：
    # 一度报了 infer +38% 的退化，其实是脚本自己的 bug）。
    print()
    print("== 5. 随时间退化（前 1/4 段 → 后 1/4 段）==")
    for name in ("infer_ms", "pipe_ms", "gap_ms", "send_ms", "e2e_probe_ms",
                 "out_key_ms", "rss_mb", "handles"):
        vals = col(name, "p50")
        if len(vals) < 8:
            continue
        q = max(1, len(vals) // 4)
        a, b = st.median(vals[:q]), st.median(vals[-q:])
        # 尖刺段会同时抬高两边，按**值**挑出最差的 2 段、按**索引**剔除（保时间序）
        worst = set(sorted(range(len(vals)), key=lambda i: -vals[i])[:2])
        trim = [v for i, v in enumerate(vals) if i not in worst]
        tq = max(1, len(trim) // 4)
        ta, tb = st.median(trim[:tq]), st.median(trim[-tq:])
        # 幅度 + 绝对值双门限：0.06→0.07ms 这种是噪声，不该当风险报
        flag = ""
        if ta and (tb - ta) > 0.3 and tb / ta > 1.25:
            flag = "   ← 值得查"
        elif a and (b - a) > 0.3 and b / a > 1.15:
            flag = "   （被少数段带偏，剔除后不明显）"
        noisy = sum(1 for v in vals if v > 2 * min(vals))
        print("   %-12s %8.2f→%8.2f | 剔最差2段 %8.2f→%8.2f | 超「最好段×2」%d段%s"
              % (name, a, b, ta, tb, noisy, flag))

    # ---- 6. 资源 ----
    rss, hd = col("rss_mb", "p50"), col("handles", "p50")
    if len(rss) >= 4 or len(hd) >= 4:
        print()
        print("== 6. 资源趋势 ==")
        if len(rss) >= 4:
            print("   RSS    %8.1f → %8.1f MB（%+.1f）"
                  % (rss[0], rss[-1], rss[-1] - rss[0]))
        if len(hd) >= 4:
            print("   句柄   %8.0f → %8.0f（%+.0f）"
                  % (hd[0], hd[-1], hd[-1] - hd[0]))
        print("   （缓慢单调上涨才是泄漏；一次涨完走平是缓存/预热）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
