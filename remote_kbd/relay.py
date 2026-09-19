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


def ping(ser, timeout=2.0):
    """主动验证：发一条无副作用指令，等 Pro Micro 回 DONE。

    不用 READY 判活：32U4 打开串口会复位，READY 只在 setup 打一次，
    常常在复位/重枚举期间就漏掉了。主动发指令看回执才可靠。
    """
    try:
        ser.reset_input_buffer()
        ser.write(b"RELEASEALL\n")
    except Exception:
        return False
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            buf += ser.read(64)
        except Exception:
            break
        if b"DONE" in buf:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="COM5", help="Pro Micro 的串口号")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--cert", default="certs/cert.pem")
    ap.add_argument("--key", default="certs/key.pem")
    args = ap.parse_args()

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(args.cert, args.key)

    ser = serial.Serial(args.serial, 115200, timeout=0.3)
    ser.dtr = False          # 关键：禁用 DTR，避免打开串口触发 32U4 复位
    ser.rts = False
    print(f"[relay] serial {args.serial} open")
    time.sleep(0.5)

    if ping(ser):
        print("[relay] Pro Micro 在线，串口通信正常")
    else:
        print("[relay] warning: Pro Micro 无响应 —— 逐项检查：")
        print("  1) 固件是否真烧进 Pro Micro（板子/端口选对了吗）")
        print("  2) 板子频率是否 8MHz（选错会导致 USB 异常）")
        print("  3) 串口号是否正确")

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
