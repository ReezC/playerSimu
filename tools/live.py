"""【B 机运行】实时收流主力工具：自动定位探针条 + FPS/延迟统计。

用法:
    python -m tools.live
    python -m tools.live --show --seconds 20
"""

import argparse
import statistics
import time

import cv2
import numpy as np

from link import PyAVSource
from tools.config import ROOT, get
from tools.probe_codec import resolve_delay_ms


def locate_probe(gray, cell, gap, bits, y_search=60, x_search=600):
    """暴力搜索探针条起点。评分=采样点黑白确信度，约束=前两位为 1,0。"""
    p = cell + gap
    n = 2 + bits
    offsets = (cell // 2 + np.arange(n) * p).astype(np.int32)
    h, w = gray.shape[:2]
    y_lim = min(y_search, h)
    x_lim = max(0, min(x_search, w - int(offsets[-1])))
    if x_lim <= 0:
        return None

    xs = np.arange(x_lim)
    best = None
    for y in range(y_lim):
        row = gray[y].astype(np.int32)
        v = row[offsets[None, :] + xs[:, None]]     # (x_lim, n)
        b = v > 128
        ok = b[:, 0] & ~b[:, 1]
        ones = b[:, 2:].sum(axis=1)
        ok = ok & (ones >= 3) & (ones <= 34)
        if not ok.any():
            continue
        conf = np.abs(v - 128).mean(axis=1)
        conf = np.where(ok, conf, -1.0)
        j = int(np.argmax(conf))
        if conf[j] > 0 and (best is None or conf[j] > best[0]):
            best = (float(conf[j]), int(j), y)
    return best


def load_offset_ms():
    p = ROOT / "config" / "clock_offset.txt"
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return 0.0


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--report", type=float, default=2.0)
    ap.add_argument("--max-delay", type=float, default=5000.0)
    ap.add_argument("--no-auto", action="store_true", help="不自动定位，用配置坐标")
    args = ap.parse_args()

    cell = get("probe", "cell", 16)
    gap = get("probe", "gap", 2)
    bits = get("probe", "bits", 40)
    px, py = get("probe", "x", 100), get("probe", "y", 8)
    offset_ms = load_offset_ms()

    src = PyAVSource(args.url or get("stream", "url"))
    src.open()
    print("[live] source=%s  clock_offset=%.3fms  cell=%d gap=%d bits=%d"
          % (args.url or get("stream", "url"), offset_ms, cell, gap, bits), flush=True)

    gaps, delays = [], []
    miss = 0
    n = 0
    dups = 0
    last = None
    last_pts = None
    locked = None
    t0 = time.perf_counter()
    t_report = t0

    try:
        while True:
            f = src.read()
            if f is None:
                print("[live] 流结束")
                break

            if f.pts is not None and f.pts == last_pts:
                dups += 1
                continue
            last_pts = f.pts

            n += 1
            if last is not None:
                gaps.append((f.t_recv_mono - last) * 1000.0)
            last = f.t_recv_mono

            gray = cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY)

            if not args.no_auto and locked is None and n >= 20:
                r = locate_probe(gray, cell, gap, bits)
                if r is not None:
                    conf, x0, y = r
                    locked = (x0, y)
                    print("[live] 自动定位成功: x0=%d y=%d conf=%.1f   (配置 x=%d y=%d)"
                          % (x0, y, conf, px, py), flush=True)

            rx, ry = locked if locked else (px, py)
            p = cell + gap
            base = rx + cell // 2
            vals = [int(gray[ry, int(base + i * p)]) for i in range(2 + bits)]
            b = [1 if v > 128 else 0 for v in vals]
            if b[0] == 1 and b[1] == 0:
                ts = 0
                for bit in b[2:]:
                    ts = (ts << 1) | bit
                d = resolve_delay_ms((f.t_recv_wall + offset_ms / 1000.0) * 1000.0, ts)
                if d is not None and d < args.max_delay:
                    delays.append(d)
            else:
                miss += 1

            if args.show:
                vis = cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR)
                for i in range(2 + bits):
                    cx = rx + cell // 2 + i * p
                    cv2.circle(vis, (cx, ry), 3, (0, 255, 0) if b[i] else (0, 0, 255), -1)
                if delays:
                    cv2.putText(vis, "lat %.1f ms" % delays[-1], (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
                small = cv2.resize(vis, None, fx=0.5, fy=0.5)
                cv2.imshow("live", small)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.seconds > 0 and (time.perf_counter() - t0) >= args.seconds:
                break

            if time.perf_counter() - t_report >= args.report:
                t_report = time.perf_counter()
                el = time.perf_counter() - t0
                line = "[%6.1fs] frames=%6d fps=%5.1f" % (el, n, n / el)
                if gaps:
                    line += " | gap p50=%5.2f p95=%5.2fms" % (pct(gaps, .5), pct(gaps, .95))
                if delays:
                    line += " | lat p50=%5.1f p95=%5.1fms miss=%d" % (
                        pct(delays, .5), pct(delays, .95), miss)
                else:
                    line += " | lat -- miss=%d" % miss
                print(line, flush=True)

    except KeyboardInterrupt:
        print("\n[live] 中断")
    finally:
        src.close()
        cv2.destroyAllWindows()

    el = time.perf_counter() - t0
    print("\n========== 汇总 ==========")
    print("时长        : %.1fs" % el)
    print("帧数        : %d  (均值 %.2f fps)" % (n, n / el if el else 0))
    if gaps:
        print("帧间隔      : p50=%.2f p95=%.2f p99=%.2f max=%.2f ms (jitter=%.2f)"
              % (pct(gaps, .5), pct(gaps, .95), pct(gaps, .99), max(gaps), statistics.pstdev(gaps)))
    if delays:
        print("端到端延迟  : p50=%.1f p95=%.1f p99=%.1f max=%.1f ms (jitter=%.2f)"
              % (pct(delays, .5), pct(delays, .95), pct(delays, .99), max(delays),
                 statistics.pstdev(delays)))
    print("探针解码失败: %d / %d" % (miss, n))
    print("重复帧跳过  : %d" % dups)
    print("坏包跳过    : %d" % src.bad_packets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
