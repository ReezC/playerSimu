"""临时：验证 ①relay 用 select 等可读（不打断 TLS 记录）②B 侧链路坏了自动重连。"""
import os
import socket
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, ".")

from remote_kbd import relay

ok = []


class FakeSerial:
    def __init__(self, port, baud, timeout=0.3, write_timeout=1.0):
        self.written = b""

    def read(self, n):
        time.sleep(0.02)
        return b""

    def write(self, data):
        self.written += data
        return len(data)

    def close(self):
        pass

    dtr = rts = False

    def reset_input_buffer(self):
        pass


relay.serial = types.SimpleNamespace(Serial=FakeSerial)

print("== 1. relay：select 等可读 → 空闲不打断连接，有数据照常转发 ==")
link = relay.SerialLink("COMFAKE")
a, b = socket.socketpair()
t = threading.Thread(target=relay.bridge, args=(b, link), daemon=True)
t.start()
time.sleep(0.2)
res = {}
link2 = link


class Wrap:
    """包一层，记录 bridge 有没有因为「空闲 5 秒」把连接关掉。"""

    def __init__(self, inner):
        self._i = inner

    def read(self, n):
        return self._i.read(n)

    def write(self, data):
        return self._i.write(data)


gw = Wrap(link)
a.sendall(b"PRESS A\n")
time.sleep(0.3)
print("   串口收到: %r" % link._ser.written)
ok.append(("select 版本仍然正常转发", link._ser.written == b"PRESS A\n"))

# 空闲 6 秒（> 原来的 recv 超时 5s）：不该断开、也不该报错
time.sleep(6.0)
a.sendall(b"RELEASE A\n")
time.sleep(0.3)
print("   空闲 6 秒后再发一条，串口收到: %r" % link._ser.written[-14:])
print("   桥还活着: %s" % t.is_alive())
ok.append(("空闲超过原超时后连接依然可用",
           link._ser.written.endswith(b"RELEASE A\n")))
ok.append(("桥没有断开", t.is_alive()))
a.close()
time.sleep(0.3)

print()
print("== 2. B 侧：链路坏了会自己重连（且节流）==")
import ctypes as _ctypes
for _d in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
           "vcruntime140.dll", "vcruntime140_1.dll"):
    try:
        _ctypes.CDLL(_d)
    except OSError:
        pass
from PyQt5.QtWidgets import QApplication
import gui.player_panel as pp

app = QApplication(sys.argv)
panel = pp.PlayerPanel()

calls = []
pp.dinput.reconnect_remote = lambda: (calls.append(time.monotonic()), True)[1]
pp.dinput.link_health = lambda: {"backend": "remote", "ok": False,
                                 "err": "SSLEOFError: EOF occurred in violation of protocol"}

panel._link_key = None
panel._poll_link_health()
time.sleep(0.4)
print("   坏链路一次轮询 → 重连调用数 %d；状态行：%s"
      % (len(calls), panel.lbl_device_state.text().splitlines()[0]))
ok.append(("坏链路触发了自动重连", len(calls) == 1))
ok.append(("状态行说明在自动重连", "自动重连" in panel.lbl_device_state.text()))

for _ in range(6):          # 5 秒内反复轮询：不该反复重连
    panel._link_key = None
    panel._poll_link_health()
    time.sleep(0.05)
time.sleep(0.4)
print("   0.7 秒内轮询 7 次 → 重连调用数 %d（应仍是 1）" % len(calls))
ok.append(("重连有节流（不会每 500ms 重连一次）", len(calls) == 1))

time.sleep(4.8)             # 过 5 秒后允许再来一次
panel._link_key = None
panel._poll_link_health()
time.sleep(0.4)
print("   再过 5 秒 → 重连调用数 %d（应变成 2）" % len(calls))
ok.append(("超过 5 秒才允许下一次重连", len(calls) == 2))

pp.dinput.link_health = lambda: {"backend": "remote", "ok": True, "err": ""}
panel._link_key = None
panel._poll_link_health()
print("   恢复正常后状态行：%s" % panel.lbl_device_state.text())
ok.append(("恢复后状态行变绿字", "已连接 ProMicro" in panel.lbl_device_state.text()))

print()
for n, good in ok:
    print("   %-46s [%s]" % (n, "OK" if good else "NG"))
print()
print("共 %d 项，NG %d 项" % (len(ok), sum(1 for _n, g in ok if not g)))
