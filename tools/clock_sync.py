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


def sync_offset(host=None, port=None, samples=None, interval_ms=None,
                save=True, quiet=False):
    """自动对时：探测 A 机 clock_server 并返回 offset（毫秒）。

    失败（A 机没跑 clock_server / 网络不通）返回 None，不抛异常。

    **快速失败**：先发一个 PING 探测，0.6s 内不回就直接放弃，
    避免 A 机没跑时钟服务时干等几十秒（200 样本 × 0.5s 超时）。
    """
    host = host or get("a_host")
    port = port or get("clock_sync", "server_port", 5001)
    if samples is None:
        samples = get("clock_sync", "samples", 200)
    if interval_ms is None:
        interval_ms = get("clock_sync", "interval_ms", 10)

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.settimeout(0.6)
    try:
        probe.sendto(b"PING,-1", (host, port))
        probe.recvfrom(1024)
    except socket.timeout:
        if not quiet:
            print("[clock_sync] A 机 clock_server 无响应，跳过自动对时", flush=True)
        return None
    finally:
        probe.close()

    recs = measure(host, port, samples, interval_ms)
    if not recs:
        return None
    recs.sort(key=lambda r: r[0])
    best = recs[: max(1, len(recs) // 10)]
    offset = statistics.fmean([o for _, o in best])

    if save:
        (ROOT / "config" / "clock_offset.txt").write_text(
            f"{offset:.6f}\n", encoding="utf-8")
    if not quiet:
        jitter = statistics.pstdev([o for _, o in best]) if len(best) > 1 else 0.0
        print(f"[clock_sync] 自动对时 offset={offset:.3f}ms (jitter={jitter:.3f}ms)",
              flush=True)
    return offset


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
