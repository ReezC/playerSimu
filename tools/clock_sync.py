"""【B 机运行】与 A 机做时钟偏移估计（NTP 风格 + min-delay filter）。

offset 定义：t_A = t_B + offset （offset > 0 表示 A 机时钟更快）

四个时刻：
  T0 = B 机发出（B 钟）
  T1 = A 机收到（A 钟）
  T2 = A 机发出（A 钟）
  T3 = B 机收到（B 钟）
  offset = ((T1 - T0) + (T2 - T3)) / 2
  delay  = (T3 - T0) - (T2 - T1)

min-delay filter：取 RTT 最小的若干样本求均值，削弱排队抖动。

用法（B 机）：
    python -m tools.clock_sync --host 192.168.1.8
"""

import argparse
import socket
import statistics
import sys
import time

from tools.config import ROOT, get


def measure(host: str, port: int, samples: int, interval_ms: int, timeout: float = 0.5):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    server = (host, port)
    recs = []

    for i in range(samples):
        t0 = time.time_ns() // 1_000_000
        try:
            sock.sendto(f"PING,{i}".encode(), server)
            data, _ = sock.recvfrom(1024)
            t3 = time.time_ns() // 1_000_000
        except socket.timeout:
            continue

        parts = data.decode().strip().split(",")
        if len(parts) < 4 or parts[0] != "PONG":
            continue
        try:
            t1, t2 = int(parts[2]), int(parts[3])
        except ValueError:
            continue

        offset = ((t1 - t0) + (t2 - t3)) / 2.0
        delay = (t3 - t0) - (t2 - t1)
        recs.append((delay, offset))

        if interval_ms > 0:
            time.sleep(interval_ms / 1000.0)

    sock.close()
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default=None, help="A 机 IP")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--interval", type=int, default=None, help="采样间隔 ms")
    ap.add_argument("--save", action="store_true", help="把 offset 写入 config/clock_offset.txt")
    args = ap.parse_args()

    host = args.host or get("a_host")
    port = args.port or get("clock_sync", "server_port", 5001)
    samples = args.samples or get("clock_sync", "samples", 200)
    interval = args.interval or get("clock_sync", "interval_ms", 10)

    print(f"[clock_sync] {host}:{port} samples={samples} interval={interval}ms")
    recs = measure(host, port, samples, interval)
    if not recs:
        print("[clock_sync] 无响应：检查 A 机 clock_server 是否已启动、防火墙是否放行 UDP")
        return 1

    recs.sort(key=lambda r: r[0])
    best = recs[: max(1, len(recs) // 10)]  # 取 RTT 最小的 10%
    offsets = [o for _, o in best]
    delays = [d for d, _ in recs]

    offset = statistics.fmean(offsets)
    jitter = statistics.pstdev(offsets) if len(offsets) > 1 else 0.0

    print(f"[clock_sync] 有效样本 {len(recs)}/{samples}")
    print(f"[clock_sync] RTT   min={min(delays):.3f}ms  median={statistics.median(delays):.3f}ms")
    print(f"[clock_sync] offset = {offset:.3f} ms  (jitter={jitter:.3f} ms)")
    print(f"[clock_sync] t_A = t_B + {offset:.3f}ms")

    if args.save:
        p = ROOT / "config" / "clock_offset.txt"
        p.write_text(f"{offset:.6f}\n", encoding="utf-8")
        print(f"[clock_sync] 已写入 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
