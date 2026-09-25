"""【B 机运行】推流自检 · 测量侧：按候选清单逐条量端到端延迟。

配套：A 机在部署台点工具栏的「推流自检」（它按**同一份清单**逐条换参试推）。
两边各点一次、各跑一轮，最后把 A 机那份 `perf_push_A.json` 拷过来合并出表。

用法：
    python -m tools.stream_sweep --precheck          # 只做预检（几何/链路），10 秒
    python -m tools.stream_sweep                     # 测一轮，写 perf_push_B.json
    python -m tools.stream_sweep --merge perf_push_A.json   # 合并出表并给结论
    python -m tools.stream_sweep --seconds 8 --settle 2     # 先短跑一轮试流程

**为什么先预检、不先扫 18 套**：延迟数只在**探针几何判据通过**时才算数
（`probe_codec.Verdict`）。几何不对的时候，18 套跑完得到的是一张"好看但全假"的
表 —— 2026-09-25 就是这么白花了半天。所以这里第一步就把几何判死：不过就**拒绝
开扫**，并告诉你去 `python -m tools.probe_tune` 手工推框。

**和 A 机怎么对上**：靠**握手**（tools/sweep_link.py：A 机每段开始发一条公告，B 机按
公告量、并回一条回执）。所以**谁先跑都行** —— B 机可以先把这条命令跑起来等 A 机，
中途重启也不影响；A 机把该段的参数一起公告过来，连"两边清单必须一致"都不再是前提。

收不到公告（A 机是旧版 / 端口被防火墙挡了 / 加 `--no-link`）时**退回老流程**：两边按清单
**同一顺序**走，各记起止**墙钟**（双机已对时，差 ~1ms），合并时用它核对 —— 那条路要求
**先点 A 机开始、紧接着（几秒内）跑这个命令**，晚几分钟就整轮对不上（2026-09-25 实测）。
"""

import argparse
import json
import socket
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2                                                      # noqa: E402
import numpy as np                                              # noqa: E402

from link import PyAVSource                                     # noqa: E402
from tools import probe_codec, push_presets as pp, sweep_link    # noqa: E402
from tools.config import ROOT, get                              # noqa: E402

#: 每段等流等多久（A 机换参数会重启 ffmpeg，UDP 流会断一下）
WAIT_S = 15.0
#: 等 A 机把报告发过来最多多久（B 机自己还有最后一段要量完，所以给得宽）
REPORT_WAIT_S = 200.0
#: 最近一次开流失败的原因 —— 用来分辨"该重试"还是"要人动手"（端口被占）
_LAST_ERR = []
#: 这类错误重试没有意义：UDP 5000 被别人占着（工作台的实时预览最常干这事）
_FATAL = ("10048", "address already in use")


def _offset_s():
    """时钟偏移（秒）：启动时对一次时，失败就用文件里的旧值。

    **顺手看一眼旧值有多旧**：两台机器各漂各的（实测 ~45ms/小时），拿一小时前的偏移
    去算延迟，那段漂移会**整片**加到每个数上。所以旧了就明说 —— 同一轮里的数还能横向
    比（大家用同一个偏移），但不能拿它去和别的时间点的数比。
    """
    try:
        from tools.probe_recv import load_offset_ms
        off = float(load_offset_ms(None)) / 1000.0
    except Exception as e:                   # noqa: BLE001
        print("[sweep] 拿不到时钟偏移（%s）—— 先跑 tools.clock_sync --save" % e)
        return 0.0
    try:
        age = time.time() - (ROOT / "config" / "clock_offset.txt").stat().st_mtime
        if age > 300:
            print("[sweep] 提醒：这份时钟偏移是 %.0f 分钟前的（漂移 ~45ms/小时）——"
                  " 想和别的时间点的数比，先跑一次 tools.clock_sync --save"
                  % (age / 60.0))
    except Exception:                        # noqa: BLE001
        pass
    return off


def _geometry(shape):
    """探针几何：**与工作台同一套来源**（项目标定 → 全局 → link.yaml）。"""
    from tools.probe_tune import initial_geo
    bits = int(get("probe", "bits", 40))
    geo, src = initial_geo(shape, bits)
    return geo, bits, src


def open_stream(url, fmt, wait_s):
    """开流并拿到第一帧；开不起来就重试，直到 wait_s 用尽 → (src, frame) / (None, None)。

    为什么要重试：A 机每换一次参数都会重启 ffmpeg，UDP 流会断一下再回来；
    而 PyAV 的 open 是"探到流头才算成功"（`analyzeduration=1s`），
    在断流期间会直接抛异常。所以这里包一层等待，别让一次重启判成"这一套失败"。
    """
    t0 = time.monotonic()
    last = None
    del _LAST_ERR[:]
    while time.monotonic() - t0 < wait_s:
        src = None
        try:
            src = PyAVSource(url, container_format=fmt)
            src.open()
            f = src.read()
            if f is not None:
                return src, f
        except Exception as e:               # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, e)
            _LAST_ERR.append(str(e))
            if any(k in str(e).lower() or k in str(e) for k in _FATAL):
                return None, None            # 端口被占：重试没用，交给调用方说人话
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
        time.sleep(0.4)
    if last:
        print("[sweep] 等流超时（%.0fs）：%s" % (wait_s, last))
    return None, None


def fatal_open_error():
    """最近一次开流失败是不是"要人动手"的那类（端口被占）→ 人话，否则 None。"""
    if any(any(k in m.lower() or k in m for k in _FATAL) for m in _LAST_ERR):
        from tools.probe_tune import human_open_error
        return human_open_error(Exception(_LAST_ERR[-1]))
    return None


#: 握手模式下每段最多等多久拿到公告（拿不到就认为 A 机跑完了 / 没在握手）。
#: 比一段的时长宽得多 —— A 机换参数时 ffmpeg 起停本身要一两秒。
SEG_WAIT_S = 40.0


def iter_segments(link, presets, seconds, settle, on_line=None):
    """产出 (段号, 参数, seconds, settle, 总段数) —— **握手优先**。

    有公告就按 A 机的公告走（段号、参数、测量窗口全以 A 机为准）：B 机晚起、中途重启、
    两边清单不一致都不影响对上。一条公告都收不到就退回"按清单顺序"那套老流程，
    并且**明说**退回去了（不说的话现场会以为握手生效了）。

    这段逻辑单独抽出来是为了**不起流就能把它测了**（见 tools/selftest_sweep_link.py）。
    """
    say = on_line or (lambda _t: None)
    if link is not None:
        n_ann = 0
        while True:
            seg = link.wait_seg(SEG_WAIT_S)
            if seg is None:
                break
            n_ann += 1
            link.ack(seg)                       # 回执：A 机的日志里会显示"B 机已跟上"
            preset = dict(seg.get("preset") or {})
            if not preset.get("name"):
                preset["name"] = "段%d" % (int(seg.get("i") or 0) + 1)
            yield (int(seg.get("i") or 0), preset,
                   float(seg.get("seconds") or seconds),
                   float(seg.get("settle") or settle),
                   int(seg.get("n") or 0) or len(presets))
        if n_ann:
            say("[sweep] 握手结束：按 A 机的公告量了 %d 段" % n_ann)
            return
        say("[sweep] 没收到 A 机的段公告（A 机没在跑自检？端口对不上？）—— "
            "退回按清单顺序走，**这一轮的对齐要靠人掐时间**")
    for i, p in enumerate(presets):
        yield i, p, seconds, settle, len(presets)


def open_link(no_link=False):
    """建握手监听（端口来自 link.yaml 的 sweep.port）→ Listener / None。

    拿不到就返回 None，外面退回老流程 —— **握手不该是跑不起来的理由**。
    """
    if no_link:
        print("[sweep] --no-link：按清单顺序走（老流程，需要人掐时间）")
        return None
    port = int(get("sweep", "port", sweep_link.DEFAULT_PORT))
    try:
        lis = sweep_link.Listener(port)
    except OSError as e:
        print("[sweep] 握手端口 UDP %d 绑不上（%s）—— 退回按清单顺序走" % (port, e))
        return None
    print("[sweep] 握手：在 UDP %d 上等 A 机的段公告"
          "（A 机在你之前或之后开始都行，不用掐时间）" % port)
    return lis


def measure_segment(src, geo, bits, offset_s, seconds, settle):
    """量一段 → meas dict。前 settle 秒丢弃（推流刚起、解码器刚锁）。

    返回的键见 `push_presets.row_b`。**延迟只在几何判据通过时才认**：
    `probe_mono` 为 False 时那串数就是乱码凑出来的（见 probe_codec.Verdict），
    合并阶段的 `classify` 会据此淘汰整行。
    """
    vd = probe_codec.Verdict()
    delays, gaps, reads = [], [], []
    n = n_settle = n_err = 0
    err_first = None
    last_mono = None
    t0 = time.monotonic()
    while True:
        el = time.monotonic() - t0
        if el >= settle + seconds:
            break
        _t0 = time.perf_counter()
        try:
            f = src.read()
        except Exception as e:               # noqa: BLE001
            # **一次读超时不许炸掉整轮**：A 机每段都会重启 ffmpeg，那一下流本来就是断的。
            # 实测踩过（2026-09-25 23:50）：第 18 段的一次 read 超时把整轮 16 段全废了 ——
            # 那时结果还没落盘、结论也没回传，A 机还在干等。所以这里记一笔就继续量。
            n_err += 1
            if err_first is None:
                err_first = "%s: %s" % (type(e).__name__, e)
            time.sleep(0.01)
            continue
        _t1 = time.perf_counter()
        if f is None:
            time.sleep(0.002)
            continue
        now = time.monotonic() - t0
        if last_mono is not None:
            g = (f.t_recv_mono - last_mono) * 1000.0
            if now >= settle:
                gaps.append(g)
        last_mono = f.t_recv_mono
        if now < settle:
            n_settle += 1
            continue
        n += 1
        reads.append((_t1 - _t0) * 1000.0)

        gray = cv2.cvtColor(np.asarray(f.image), cv2.COLOR_RGB2GRAY)
        ts = probe_codec.decode_ms(gray, geo["x"], geo["y"], geo["cell"],
                                  geo["gap"], bits)
        if ts is not None:
            t_recv_ms = (f.t_recv_wall + offset_s) * 1000.0
            # 原始差也喂进判据：**整片平移**那种错位（单调、值也在一天内、只是位置
            # 被挪了）只有它能识破 —— 预检要拦住它，否则每段都"没量到延迟"，
            # 18 段跑完才发现白跑。
            vd.add(ts, 1000.0 / 60.0,
                   delay_raw_ms=t_recv_ms - (probe_codec.day_start_ms() + ts))
            d = probe_codec.resolve_delay_ms(t_recv_ms, ts)
            if d is not None and d < 5000:
                delays.append(d)

    span = max(1e-6, (time.monotonic() - t0) - settle)

    def pct(xs, q):
        if not xs:
            return None
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * q))]

    ok, why = vd.summary()
    return {"frames": n, "settle_frames": n_settle,
            "read_err": n_err, "read_err_first": err_first,
            "recv_fps": (n / span) if span > 0 else None,
            "proc_fps": None,                    # 测量侧不做推理（这是"轻消费者"）
            "lat_p50": pct(delays, 0.5), "lat_p95": pct(delays, 0.95),
            "lat_max": max(delays) if delays else None, "lat_n": len(delays),
            "jitter": (statistics.pstdev(delays) if len(delays) > 2 else None),
            "read_p50": pct(reads, 0.5), "gap_p50": pct(gaps, 0.5),
            "probe_mono": bool(vd.mono), "probe_why": why,
            "bad_packets": int(getattr(src, "bad_packets", 0) or 0)}


def precheck(src, geo, bits, offset_s, seconds=4.0):
    """几何/链路预检 → (能不能开扫, 原因)。**不过就不扫**（省得出一张全假的表）。"""
    m = measure_segment(src, geo, bits, offset_s, seconds=seconds, settle=1.0)
    if m["lat_n"] == 0:
        return False, ("这一小会儿一帧延迟都没量到（解码数 %d / 收到 %d 帧）—— "
                       "先确认探针在画、几何对得上" % (m["lat_n"], m["frames"]))
    if not m["probe_mono"]:
        return False, ("探针几何判据没过：%s\n"
                       "    延迟数现在**不可信**（错位采样照样能凑出看着合法的值）。\n"
                       "    先在 B 机修几何：python -m tools.probe_tune --project <项目目录>\n"
                       "    （或用工作台实时页的「框选探针」；几何不对就别浪费 6 分钟扫参数）"
                       % m["probe_why"])
    return True, ("预检通过：延迟 p50 %.0fms、收到 %.1f fps、几何单调（步长≈帧周期）"
                  % (m["lat_p50"] or -1, m["recv_fps"] or -1))


def result_text(rows_a, rows_b):
    """A/B 两份记账 → 出表 + 结论 + 一行"怎么用" → **返回文本**。

    抽成返回文本，是为了同一份结论能**打印给 B 机看，也能回传给 A 机**（全自动那条路
    就是靠它：结论回给部署台弹出来，人不用去 B 机抄数字）。
    """
    rows = pp.merge(rows_a, rows_b)
    rows, best, why = pp.rank(rows)
    out = [pp.render(rows, best, why=why)]
    if best:
        out.append("\n把上面那条写回部署台：")
        out.append("  config/deploy.json 的 push 段改成 —— fps=%s 码率=%s GOP=%s "
                   "passthrough=%s" % (best.get("fps"), best.get("bitrate"),
                                       best.get("gop"), best.get("passthrough")))
        if not any(r.get("pass") for r in rows):
            out.append("（注意：它只是**折中建议** —— 上面那句前提没满足，"
                       "先解决它再重跑一轮）")
    else:
        out.append("\n（没有任何可比的段：B 机那份里没量到延迟的段？）")
    return "\n".join(out)


def read_rows(path):
    """读一份记账 → rows（读不了返回 None）。"""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d.get("rows", d if isinstance(d, list) else [])
    except Exception:                        # noqa: BLE001
        return None


def merge_and_report(a_path, b_path):
    """（命令行 `--merge` 用）读两份文件 → 打印结果。"""
    rows_a = read_rows(a_path)
    if rows_a is None:
        print("[sweep] 读不了 A 机那份 %s（先把它拷过来）" % a_path)
        return 1
    rows_b = read_rows(b_path)
    if rows_b is None:
        print("[sweep] 读不了 %s：%s（先跑一轮 `python -m tools.stream_sweep`）"
              % (b_path, b_path))
        return 1
    print(result_text(rows_a, rows_b))
    print("\n（A 机那份：%s；B 机那份：%s）" % (Path(a_path).name, Path(b_path).name))
    return 0


def finish_and_report(link, b_path, say=None):
    """收尾：等 A 机的报告 → 合并出表 → **把结论回给 A 机** → 返回结论文本。

    这是"两个开关"里 B 机那一半的关键：报告走同一条握手通道（UDP 5002），
    所以不需要有人去 A 机拷 `perf_push_A.json`、也不需要记住路径。

    拿不到报告也照样出成绩（B 侧那几个数自己成立），只是会说清"少了 A 机那半"。
    """
    say = say or (lambda t: print(t, flush=True))
    rows_b = read_rows(b_path) or []
    rows_a = None
    if link is not None:
        say("[sweep] 等 A 机把报告发过来…（走握手通道，不用手工拷文件）")
        blob = link.wait_blob("rep", REPORT_WAIT_S)
        if blob:
            try:
                Path(pp.A_REPORT).write_text(blob, encoding="utf-8")
                say("[sweep] 收到 A 机报告 → 已写 %s" % pp.A_REPORT)
                rows_a = read_rows(pp.A_REPORT)
            except Exception as e:            # noqa: BLE001
                say("[sweep] 写 %s 失败：%s" % (pp.A_REPORT, e))
        else:
            say("[sweep] 没等到 A 机的报告（%.0f 秒）—— A 机可能是旧版本、"
                "或它那侧没跑自检。下面只有 B 机侧的数。" % REPORT_WAIT_S)
    text = result_text(rows_a or [], rows_b)
    say("\n" + text)
    if link is not None:
        n = link.send_blob("res", text)
        say("[sweep] 结论已回给 A 机（%d 片）—— 部署台会弹出来" % n if n
            else "[sweep] 结论回传失败（A 机的地址没学到？）")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=None)
    ap.add_argument("--format", default=None, help="强制输入格式（裸流必须给）")
    ap.add_argument("--presets", default=None, help="候选清单路径（默认 config/push_presets.json）")
    ap.add_argument("--seconds", type=float, default=None, help="每段测量时长")
    ap.add_argument("--settle", type=float, default=None, help="每段丢弃的起始秒数")
    ap.add_argument("--wait", type=float, default=WAIT_S, help="每段等流超时")
    ap.add_argument("--precheck", action="store_true", help="只做预检")
    ap.add_argument("--merge", default=None, help="合并 A 机那份并出表")
    ap.add_argument("--no-link", action="store_true", dest="no_link",
                    help="不走握手，按清单顺序量（老流程：要在 A 机点开始后几秒内跑）")
    ap.add_argument("--auto", action="store_true", dest="auto",
                    help="全自动：等 A 机公告 → 逐段量 → 收 A 机报告 → 出表"
                         " → 结论回传给 A 机（B 机这边只要跑这一个命令）")
    args = ap.parse_args()

    b_path = ROOT / pp.B_REPORT
    if args.merge:
        return merge_and_report(args.merge, b_path)

    cfg = pp.load(args.presets)
    seconds = float(args.seconds if args.seconds is not None else cfg["seconds"])
    settle = float(args.settle if args.settle is not None else cfg["settle_seconds"])
    presets = cfg["presets"]
    url = args.url or get("stream", "url")
    fmt = args.format or get("stream", "format", None)
    offset_s = _offset_s()

    print("[sweep] 候选 %d 条 × (%.0fs 测量 + %.0fs 丢弃) ≈ %.0f 分钟"
          % (len(presets), seconds, settle,
             len(presets) * (seconds + settle + 3) / 60.0))
    print("[sweep] 命令：在 A 机部署台点「推流自检」——**你这条命令先跑还是后跑都行**；"
          "每段开始 A 机会公告过来，按公告对齐（收不到公告才退回『按清单顺序』那套，"
          "那才要掐时间）")

    src, frame = open_stream(url, fmt, args.wait)
    if src is None:
        fatal = fatal_open_error()
        if fatal:
            print("[sweep] " + fatal)
            return 1
        print("[sweep] 等不到流：A 机在推吗？（%s）" % url)
        print("       排查看这三样：A 机「屏幕推流」卡在跑？目标是不是 %s？"
              "netstat -ano -p UDP | findstr :5000" % url)
        return 1
    geo, bits, geo_src = _geometry(np.asarray(frame.image).shape)
    print("[sweep] 流 %sx%s  探针几何来源=%s x=%.1f y=%.1f cell=%.2f gap=%.2f"
          % (src.size[0], src.size[1], geo_src, geo["x"], geo["y"],
             geo["cell"], geo["gap"]))

    # --auto：握手监听要**早于预检**建起来 —— 预检不过时得把原因回给 A 机，
    # 否则部署台那边不知道 B 已经放弃，只能干等超时。
    link = open_link(args.no_link) if args.auto else None

    ok, why = precheck(src, geo, bits, offset_s)
    print("[sweep] 预检：%s" % why)
    if not ok:
        # 光说"没量到延迟"没法行动 —— 用刚拿到的那一帧直接判是哪一种毛病
        # （没有码带 / 码带在但几何错 / 压在边缘，见 probe_codec.no_decode_hint）。
        hint = ""
        try:
            g0 = cv2.cvtColor(np.asarray(frame.image), cv2.COLOR_RGB2GRAY)
            hint = probe_codec.no_decode_hint(g0, geo["x"], geo["y"],
                                              geo["cell"], geo["gap"], bits)
            print("[sweep] 诊断：%s" % hint)
        except Exception as e:                # noqa: BLE001
            hint = "看不了画面（%s）" % e
            print("[sweep] 诊断：%s" % hint)
        if link is not None:
            # 没开扫就退出，也要**说话**：A 机的部署台在等结论，等超时不如现在就知道
            link.send_blob("res", "B 机预检没过，**没有开扫**（没白跑 8 分钟）：\n"
                                  "  %s\n[sweep] 诊断：%s\n\n"
                                  "按上面的诊断修好（多在 B 机侧），A 机再点一次"
                                  "「推流自检」即可。" % (why, hint))
            link.close()
        src.close()
        return 2
    if args.precheck:
        src.close()
        return 0
    # ★ 必须现在放手：UDP 5000 同时只能有一个收流者，而下面每段都要重新开一次
    # （A 机换参数会重启 ffmpeg，流会断一下）。不关的话后面每一段都会"绑不上端口"。
    src.close()

    if link is None and not args.auto:        # --auto 在前头已经建过了（别建两次）
        link = open_link(args.no_link)
    rows = []

    def _dump():
        """把当前 rows 落盘 —— **每段都写一次**。

        为什么不是最后写一次：一轮 8 分钟、18 段。实测踩过（2026-09-25 23:50）：
        第 18 段一次读超时就让整轮 16 段的结果全丢了（都在内存里）。写 18 次小 JSON
        的开销可以忽略，而"崩了也留下已测的"是刚需。
        """
        out = {"meta": {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "host": socket.gethostname(),
                        "geometry": geo, "geometry_src": geo_src, "bits": bits,
                        "offset_s": offset_s, "seconds": seconds, "settle": settle,
                        "source": url, "presets": [q["name"] for q in presets]},
               "rows": rows}
        Path(b_path).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    try:
        for i, p, sec, stl, total in iter_segments(
                link, presets, seconds, settle,
                on_line=lambda t: print(t, flush=True)):
            print("\n[sweep] 第 %d/%d 段：%s —— 等流…"
                  % (i + 1, total, p["name"]), flush=True)
            s2, f2 = open_stream(url, fmt, args.wait)
            if s2 is None:
                fatal = fatal_open_error()
                if fatal:
                    # 端口被占这类重试没意义：继续扫下去只会浪费 17 段的时间
                    print("[sweep] " + fatal)
                    print("[sweep] 停在这里（已测 %d 段）—— 让出端口后重跑整轮"
                          % len(rows))
                    break
                t = time.time()
                rows.append(pp.row_b(p["name"], i, p, t, t, {}, extra={
                    "ok": False, "why": "这一段没等到流（A 机没切过来？）"}))
                continue
            t0 = time.time()
            try:
                meas = measure_segment(s2, geo, bits, offset_s, sec, stl)
            except Exception as e:                # noqa: BLE001
                # 这一段读挂了也不许带走整轮：记成"这一段没量到"，继续下一段
                print("[sweep]   这一段读挂了（%s: %s）—— 记为失败，继续下一段"
                      % (type(e).__name__, e), flush=True)
                meas = {}
            finally:
                try:
                    s2.close()
                except Exception:                 # noqa: BLE001
                    pass
            t1 = time.time()
            r = pp.row_b(p["name"], i, p, t0, t1, meas)
            rows.append(r)
            print("[sweep]   → 收到 %.1f fps、延迟 p50 %.0f / p95 %.0f ms（几何%s）"
                  % (meas.get("recv_fps") or -1, meas.get("lat_p50") or -1,
                     meas.get("lat_p95") or -1,
                     "单调" if meas.get("probe_mono") else "**可疑**"))
            _dump()                               # 每段都落盘（见 _dump 的说明）
    except Exception as e:                        # noqa: BLE001
        # 兜底：整轮中途出事也要**把已测的段落盘、并且出表**（别让 A 机干等）
        print("[sweep] 第 %d 段附近出错（%s: %s）—— 已测的 %d 段照样出表。"
              % (locals().get("i", -1) + 1, type(e).__name__, e, len(rows)),
              flush=True)
        _dump()
    finally:
        # --auto 不能在这儿关：后面还要用它收 A 机的报告、再把结论回过去
        if link is not None and not args.auto:
            link.close()

    _dump()                                   # 与逐段落盘同一份实现（每次覆盖写）
    print("\n[sweep] 已写 %s（%d 段）" % (b_path, len(rows)))
    if args.auto:
        try:
            finish_and_report(link, b_path)
        finally:
            if link is not None:
                link.close()
    else:
        if link is not None:
            link.close()
        print("[sweep] 把 A 机的 %s 拷过来，然后：\n"
              "        python -m tools.stream_sweep --merge %s"
              % (pp.A_REPORT, pp.A_REPORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
