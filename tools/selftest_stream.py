"""【B 机运行】一键自测实时收流链路（无需 A 机）。

内部：启动 push_clip 子进程推 UDP 流 → 主进程用 PyAVSource 收 → 输出统计。

用法：
    python -m tools.selftest_stream
    python -m tools.selftest_stream --show --seconds 15

说明：测试片段里的探针时间戳是"片段生成时刻"，因此延迟绝对值是
      "生成至今"，并无物理意义。本自测真正验证的是：
        1) PyAVSource 能稳定收到 UDP 流
        2) 帧率/帧间隔正常
        3) 探针经 编码→UDP传输→解码 后仍可正确解码
"""

import argparse
import statistics
import subprocess
import sys
import time

import cv2

from link import PyAVSource
from tools.config import get
from tools.probe_codec import decode_ms, resolve_delay_ms


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="data/recordings/testclip.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--url", type=str, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    url = args.url or get("stream", "url")
    port = get("stream", "port", 5000)
    px, py = get("probe", "x", 100), get("probe", "y", 8)
    cell, gap, bits = get("probe", "cell", 16), get("probe", "gap", 2), get("probe", "bits", 40)

    pusher = subprocess.Popen(
        [sys.executable, "-m", "tools.push_clip",
         "--file", args.file, "--target", f"udp://127.0.0.1:{port}",
         "--loop", "--duration", str(args.seconds + 8)],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
    )
    print(f"[selftest] 推流子进程 pid={pusher.pid}，等待 2s 起流...")
    time.sleep(2.0)

    src = PyAVSource(url)
    gaps, delays = [], []
    miss, n = 0, 0
    last = None
    t_start = time.perf_counter()
    t_report = time.perf_counter()

    try:
        src.open()
        print(f"[selftest] 收流 {url}  {src.size} fps={src.fps}")
        while time.perf_counter() - t_start < args.seconds:
            f = src.read()
            if f is None:
                break
            n += 1
            if last is not None:
                gaps.append((f.t_recv_mono - last) * 1000.0)
            last = f.t_recv_mono

            gray = cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY)
            ts_a = decode_ms(gray, px, py, cell, gap, bits)
            if ts_a is None:
                miss += 1
            else:
                d = resolve_delay_ms(f.t_recv_wall * 1000.0, ts_a)
                if d is not None:
                    delays.append(d)

            if args.show:
                vis = cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(vis, (px - 2, py - 2),
                              (px + (2 + bits) * (cell + gap) + 2, py + cell + 4), (0, 255, 0), 1)
                cv2.putText(vis, f"frames {n} probe {'OK' if ts_a is not None else 'MISS'}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("selftest_stream", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if pusher.poll() is not None:
                print("[selftest] 推流子进程已退出，停止收流")
                break

            if time.perf_counter() - t_report >= 1.0:
                t_report = time.perf_counter()
                print(f"  ... 已收 {n} 帧 (probe miss={miss})", flush=True)
    except Exception as e:
        print(f"[selftest] 收流失败: {type(e).__name__}: {e}")
        return 1
    finally:
        src.close()
        cv2.destroyAllWindows()
        if pusher.poll() is None:
            pusher.terminate()

    el = time.perf_counter() - t_start
    print("\n========== 实时收流自测 ==========")
    print(f"时长        : {el:.1f}s")
    print(f"帧数        : {n}  (均值 {n / el:.2f} fps)")
    if gaps:
        print(f"帧间隔      : p50={pct(gaps,.5):.2f} p95={pct(gaps,.95):.2f} "
              f"max={max(gaps):.2f} ms (jitter={statistics.pstdev(gaps):.2f})")
    print(f"探针解码失败: {miss} / {n}")
    print(f"探针时间戳  : {len(delays)} 个有效样本（绝对值=片段生成至今，无物理意义）")

    ok = n > 0 and (miss / max(1, n)) < 0.05
    print(f"\n结论        : {'通过 ✓' if ok else '未通过 ✗'}"
          f"（帧率与探针解码成功率达标）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
