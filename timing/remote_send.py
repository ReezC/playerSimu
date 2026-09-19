"""远程发送注入指令到测试机上的 remote_agent。

用法（在控制端跑）:
    python remote_send.py 192.168.1.100 9000 "FIX a 100 100"
    python remote_send.py 192.168.1.100 9000 "RND a 60 140 100"
"""
import socket
import sys


def main():
    if len(sys.argv) < 4:
        print('用法: python remote_send.py <测试机IP> <端口> "<指令>"')
        return
    ip = sys.argv[1]
    port = int(sys.argv[2])
    cmd = " ".join(sys.argv[3:])

    conn = socket.create_connection((ip, port), timeout=10)
    conn.sendall(cmd.encode())
    conn.shutdown(socket.SHUT_WR)  # 表示发送完毕，等 agent 回传结果
    resp = conn.recv(4096).decode(errors="ignore")
    print(f"发送: {cmd}")
    print(f"响应: {resp.strip()}")
    conn.close()


if __name__ == "__main__":
    main()
