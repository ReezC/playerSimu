"""【B 机运行】收流自检：实时显示 + FPS / 帧间隔 / 端到端延迟统计。

用法：
    python -m tools.probe_recv --show
    python -m tools.probe_recv --show --offset 3.42          # 手动指定时钟偏移(ms)
    python -m tools.probe_recv --file data/recordings/s1.mkv # 离线回放
"""

import argparse
import statistics
import sys
import time

import cv2
import numpy as np

from link import FileSource, PyAVSource
from tools.config import ROOT, get
from tools.probe_codec import decode_ms, resolve_delay_ms


def load_offset_ms(cli_offset: float | None) -> float:
    if cli_offset is not None:
        return cli_offset
    p = ROOT / "config" / "clock_offset.txt"
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    print("[probe_recv] 未找到时钟偏移，延迟统计不可用。"
          "请先运行 `python -m tools.clock_sync --host <A机IP> --save`")
    return 0.0


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=None)
    ap.add_argument("--file", type=str, default=None, help="离线回放文件（与 --url 二选一）")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--probe", action="store_true", default=None, help="启用屏幕时间码解码")
    ap.add_argument("--offset", type=float, default=None, help="时钟偏移 ms（t_A = t_B + offset）")
    ap.add_argument("--max-delay", type=float, default=5000.0,
                    help="延迟有效上限 ms，超出视为异常丢弃（离线回放自测时调大）")
    ap.add_argument("--report", type=float, default=2.0, help="统计打印间隔(秒)")
    ap.add_argument("--seconds", type=float, default=0.0, help="运行时长(秒)，0=不限（Ctrl+C 停）")
    args = ap.parse_args()

    cfg_probe = bool(get("probe", "enabled", True))
    use_probe = cfg_probe if args.probe is None else args.probe
    px, py = get("probe", "x", 100), get("probe", "y", 8)
    cell, gap, bits = get("probe", "cell", 16), get("probe", "gap", 2), get("probe", "bits", 40)
    offset_ms = load_offset_ms(args.offset) if use_probe else 0.0

    if args.file:
        src = FileSource(args.file, realtime=True)
    else:
        src = PyAVSource(args.url or get("stream", "url"))

    src.open()
    w, h = src.size or (0, 0)
    print(f"[probe_recv] source={args.file or src.url} {w}x{h} fps={src.fps}")
    print(f"[probe_recv] probe={'on' if use_probe else 'off'} offset={offset_ms:.3f}ms "
          f"region=({px},{py}) cell={cell}")

    gaps: list[float] = []
    delays: list[float] = []
    miss = 0
    n = 0
    last_mono = None
    t_report = time.perf_counter()
    t_start = time.perf_counter()

    try:
        while True:
            f = src.read()
            if f is None:
                break
            n += 1

            if last_mono is not None:
                gaps.append((f.t_recv_mono - last_mono) * 1000.0)
            last_mono = f.t_recv_mono

            if use_probe:
                gray = cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY)
                ts_a = decode_ms(gray, px, py, cell, gap, bits)
                if ts_a is None:
                    miss += 1
                else:
                    d = resolve_delay_ms((f.t_recv_wall + offset_ms / 1000.0) * 1000.0, ts_a)
                    if d is not None and d < args.max_delay:
                        delays.append(d)

            if args.show:
                vis = cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR)
                if use_probe:
                    cv2.rectangle(vis, (px - 2, py - 2),
                                  (px + (2 + bits) * (cell + gap) + 2, py + cell + 4),
                                  (0, 255, 0), 1)
                    if delays:
                        cv2.putText(vis, f"lat {delays[-1]:.1f}ms", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(vis, "no probe", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("probe_recv", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.seconds > 0 and (time.perf_counter() - t_start) >= args.seconds:
                break

            if time.perf_counter() - t_report >= args.report:
                t_report = time.perf_counter()
                el = time.perf_counter() - t_start
                line = (f"[{el:7.1f}s] frames={n:6d} fps={n / el:6.2f}")
                if gaps:
                    line += (f" | gap p50={pct(gaps,.5):6.2f} p95={pct(gaps,.95):6.2f} "
                             f"max={max(gaps):7.2f}ms")
                if delays:
                    line += (f" | lat p50={pct(delays,.5):6.1f} p95={pct(delays,.95):6.1f} "
                             f"p99={pct(delays,.99):6.1f}ms miss={miss}")
                print(line, flush=True)

    except KeyboardInterrupt:
        print("\n[probe_recv] interrupted")
    finally:
        src.close()
        cv2.destroyAllWindows()

    el = time.perf_counter() - t_start
    print("\n========== 汇总 ==========")
    print(f"时长        : {el:.1f}s")
    print(f"帧数        : {n}  (均值 {n / el:.2f} fps)")
    if gaps:
        print(f"帧间隔      : p50={pct(gaps,.5):.2f} p95={pct(gaps,.95):.2f} "
              f"p99={pct(gaps,.99):.2f} max={max(gaps):.2f} ms "
              f"(jitter={statistics.pstdev(gaps):.2f})")
    print(f"探针解码失败: {miss} / {n}")
    if delays:
        print(f"端到端延迟  : p50={pct(delays,.5):.1f} p95={pct(delays,.95):.1f} "
              f"p99={pct(delays,.99):.1f} max={max(delays):.1f} ms "
              f"(jitter={statistics.pstdev(delays):.2f})")
    else:
        print("端到端延迟  : 无有效样本（探针未解码或超出 --max-delay）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
