"""远程注入 agent：在检测测试机上运行，监听 TCP 端口，收到指令后转发到 Pro Micro 串口。

用法（在测试机上跑）:
    python remote_agent.py COM7 9000

之后控制端用 remote_send.py 连到 <测试机IP>:9000 发指令。
"""
import socket
import sys
import time

import serial


def main():
    if len(sys.argv) < 2:
        print("用法: python remote_agent.py <COM端口> [监听端口=9000]")
        return
    com = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9000

    ser = serial.Serial(com, 115200, timeout=1)
    time.sleep(2.0)  # 等 Pro Micro 复位完成

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"串口 {com} 就绪，监听 0.0.0.0:{port}，等待控制端...", flush=True)

    while True:
        conn, addr = srv.accept()
        print(f"控制端连接: {addr}", flush=True)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            cmd = data.decode(errors="ignore").strip()
            print(f"收到指令: {cmd}", flush=True)

            ser.write((cmd + "\n").encode())
            ser.flush()

            # 等固件回 DONE
            deadline = time.time() + 30
            resp = ""
            while time.time() < deadline:
                resp += ser.read_all().decode(errors="ignore")
                if "DONE" in resp:
                    break
                time.sleep(0.1)
            conn.sendall(resp.encode())
        conn.close()
        print("控制端断开", flush=True)


if __name__ == "__main__":
    main()
