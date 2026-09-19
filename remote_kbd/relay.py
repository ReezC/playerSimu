"""远程键盘中继（跑在游戏机 / 被控机）。

TLS 服务端：收控制机 agent 发来的按键指令，原样转发到 Pro Micro 的 USB 串口，
并把 Pro Micro 的应答（DONE/ERR）回传给控制机。

用法：
    python relay.py --serial COM5 --port 9000 --cert certs/cert.pem --key certs/key.pem
"""

import argparse
import socket
import ssl
import threading
import time

import serial


def bridge(conn, ser):
    """双向桥：TCP -> 串口，串口 -> TCP。"""
    stop = threading.Event()

    def ser_to_tcp():
        while not stop.is_set():
            try:
                data = ser.read(256)
            except Exception:
                break
            if data:
                try:
                    conn.sendall(data)
                except Exception:
                    break

    t = threading.Thread(target=ser_to_tcp, daemon=True)
    t.start()
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            ser.write(data)
    except Exception:
        pass
    finally:
        stop.set()
        try:
            conn.close()
        except Exception:
            pass


def wait_ready(ser, timeout=3.0):
    """等固件上电打印 READY（可选，方便确认串口连对了）。"""
    line = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        chunk = ser.read(1)
        if chunk:
            line += chunk
            if chunk == b"\n":
                print("[relay] firmware:", line.decode(errors="replace").strip())
                return
    print("[relay] warning: 未收到固件 READY，检查串口号/固件")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="COM5", help="Pro Micro 的串口号")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--cert", default="certs/cert.pem")
    ap.add_argument("--key", default="certs/key.pem")
    args = ap.parse_args()

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(args.cert, args.key)

    ser = serial.Serial(args.serial, 115200, timeout=0.2)
    print(f"[relay] serial {args.serial} open")
    wait_ready(ser)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"[relay] listening on :{args.port}")

    while True:
        conn, addr = server.accept()
        try:
            tls = ctx.wrap_socket(conn, server_side=True)
        except Exception as e:
            print(f"[relay] TLS handshake failed: {e}")
            conn.close()
            continue
        print(f"[relay] client {addr[0]} connected")
        bridge(tls, ser)
        print("[relay] client disconnected")


if __name__ == "__main__":
    main()
