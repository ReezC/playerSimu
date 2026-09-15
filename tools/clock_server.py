"""【A 机运行】UDP 对时服务端（echo）。

协议：B 机发 "PING,<seq>" → 本端回 "PONG,<seq>,<t1_ms>,<t2_ms>"
  t1 = 服务端收到时刻，t2 = 服务端发出时刻（均为 A 机墙钟，毫秒）

用法（A 机）：
    python -m tools.clock_server --port 5001

注意：需放行防火墙 UDP 入站。
"""

import argparse
import socket
import sys
import time

from tools.config import get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()

    port = args.port or get("clock_sync", "server_port", 5001)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, port))
    print(f"[clock_server] listening udp://{args.host}:{port}", flush=True)

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            t1 = time.time_ns() // 1_000_000
            try:
                seq = data.decode().strip().split(",")[1]
            except Exception:
                seq = "0"
            t2 = time.time_ns() // 1_000_000
            sock.sendto(f"PONG,{seq},{t1},{t2}".encode(), addr)
    except KeyboardInterrupt:
        print("\n[clock_server] bye")
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
