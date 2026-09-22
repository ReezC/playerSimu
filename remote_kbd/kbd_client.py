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
        self._stop = threading.Event()
        # 后台读线程：relay 会把固件的 DONE 应答回传，这里持续读走丢弃，
        # 否则回传缓冲被 DONE 塞满后，relay/固件/控制机整条链路会连锁卡死
        # （表现为：按键发不出、推理丢帧暴增、RELEASE 丢失导致卡键）。
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def send(self, line):
        """发一条指令（fire-and-forget，不等 DONE）。"""
        with self._lock:
            try:
                self._sock.settimeout(2.0)
                self._sock.sendall((line + "\n").encode())
            except Exception:
                pass

    def _drain(self):
        """读走 relay 回传的应答，避免回传缓冲堆积。"""
        while not self._stop.is_set():
            try:
                data = self._sock.recv(4096)
                if not data:
                    break               # 连接被对端关闭
            except socket.timeout:
                continue                # 空闲超时：继续等，别退出
            except Exception:
                break

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass
