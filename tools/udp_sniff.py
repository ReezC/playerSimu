"""UDP 收包计数探针（不依赖 PyAV）。

用途：判断 A 机的流到底有没有到达 B 机。
  - 收到包 -> 网络通，问题在解码/配置
  - 收不到 -> A 机没发出来，或 B 机防火墙拦了

用法:
    python -m tools.udp_sniff --seconds 10
"""

import argparse
import socket
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", args.port))
    except OSError as e:
        print(f"[udp_sniff] 绑定失败 {e}：端口被占用，先关掉其他 probe_recv")
        return 1
    s.settimeout(1.0)

    print(f"[udp_sniff] 监听 0.0.0.0:{args.port}，持续 {args.seconds}s ...", flush=True)

    n = 0
    total = 0
    senders = {}
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        try:
            data, addr = s.recvfrom(65535)
        except socket.timeout:
            continue
        n += 1
        total += len(data)
        senders[addr] = senders.get(addr, 0) + 1
        if n % 200 == 0:
            print(f"  ... {n} 包 / {total / 1024:.0f} KB", flush=True)

    s.close()
    print(f"\n[udp_sniff] 总计 {n} 包 / {total} 字节")
    for a, c in sorted(senders.items(), key=lambda x: -x[1]):
        print(f"  来自 {a[0]}:{a[1]}  ->  {c} 包")

    if n == 0:
        print("\n=> 一个包都没收到")
        print("   可能原因：")
        print("   1) B 机防火墙没放行 UDP 5000（管理员 PowerShell 加规则）")
        print("   2) OBS 的 FFmpeg 输出类型不是「输出到 URL」")
        print("   3) OBS 的 URL 写错了（IP / 端口）")
        print("   4) OBS 根本没在录")
    else:
        print("\n=> 网络通畅，A 机的流已到达 B 机")
        print("   问题在解码侧（PyAV 参数 / 流格式）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
