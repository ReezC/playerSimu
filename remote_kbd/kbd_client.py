"""控制机侧 TLS 客户端：agent 用它把按键指令发给游戏机的 relay。

用法：
    from remote_kbd.kbd_client import KbdClient
    kbd = KbdClient("192.168.1.200", 9000, "certs/cert.pem")
    kbd.send("PRESS LEFT")
    kbd.send("TAP CTRL 30")
"""

import socket
import ssl
import threading


class KbdClient:
    def __init__(self, host, port, cafile, timeout=5.0):
        ctx = ssl.create_default_context(cafile=cafile)
        ctx.check_hostname = False          # 自签证书不校主机名
        raw = socket.create_connection((host, port), timeout=timeout)
        self._sock = ctx.wrap_socket(raw, server_hostname=host)
        self._lock = threading.Lock()

    def send(self, line):
        """发一条指令（fire-and-forget，不等 DONE）。"""
        with self._lock:
            self._sock.sendall((line + "\n").encode())

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass
