"""本地直连 Pro Micro：通过 USB CDC 串口发指令，固件模拟成 HID 键盘。

与 kbd_client.py（网络客户端）对称：agent 直接开串口发 PRESS/RELEASE/TAP，
不需要 relay 中转（relay 是「串口↔网络」的桥，双机才需要）。

用法：
    from remote_kbd.serial_kbd import SerialKbd
    kbd = SerialKbd("COM5")
    kbd.send("PRESS LEFT")
    kbd.close()

依赖：pip install pyserial
"""

import threading
import time

# Pro Micro / Leonardo / Micro 的 USB VID:PID（换 USB 口后串口号会变，靠它自动发现）
_PRO_MICRO_VIDPIDS = ("2341:8036", "2341:8037", "1B4F:9205", "1B4F:9206")


def find_pro_micro_port():
    """自动找 Pro Micro（Arduino Leonardo/Micro、SparkFun Pro Micro）的串口。

    返回串口号（如 "COM7"），找不到返回 None。
    """
    try:
        import serial.tools.list_ports as lp
    except Exception:
        return None
    for p in lp.comports():
        hwid = (p.hwid or "").upper()
        if any(vp in hwid for vp in _PRO_MICRO_VIDPIDS):
            return p.device
    return None


class SerialKbd:
    def __init__(self, port, baudrate=115200, timeout=0.3):
        import serial
        self._ser = serial.Serial(port, baudrate, timeout=timeout)
        self._ser.dtr = False   # 关键：禁用 DTR，避免打开串口触发 32U4 复位
        self._ser.rts = False
        time.sleep(0.5)          # 等 Pro Micro 稳定（打开串口可能已触发一次复位）
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # 后台读线程：读走固件回传的 DONE/READY，否则串口输入缓冲堆积、
        # 固件 Serial.println 阻塞，整条链路卡死（表现为按键发不出）。
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def send(self, line):
        """发一条指令（fire-and-forget，不等 DONE）。"""
        with self._lock:
            try:
                self._ser.write((line + "\n").encode())
                self._ser.flush()
            except Exception:
                pass

    def _drain(self):
        """持续读走固件回传，避免输入缓冲堆积。"""
        while not self._stop.is_set():
            try:
                if not self._ser.read(256):
                    continue
            except Exception:
                break

    def close(self):
        self._stop.set()
        try:
            self._ser.close()
        except Exception:
            pass
