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

# 「最老那条还没回执的指令等了多久」——**判据与 TLS 那条共用一份** ✓（`kbd_client.py` ✓）。
#   两种跑法都要能用（部署时当脚本 / 从仓库根当包 ✓ 同 `relay.py` 那个 try/except ✓）。
try:
    from .kbd_client import silence_seconds as _silence_seconds
except ImportError:                         # noqa: BLE001 —— 当脚本跑时走这条 ✓
    from kbd_client import silence_seconds as _silence_seconds

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
    def __init__(self, port, baudrate=115200, timeout=0.3, write_timeout=1.0):
        import serial
        # write_timeout 必须有：固件不读的时候 write/flush 会**永久阻塞**，而这是
        # 跑在决策线程里的 —— 一次卡住就是「自动还在跑，但一个键都发不出去」，
        # 而且永远不会自愈。relay 那边早就设了 1.0，本地直连这份漏了。
        # 超时会抛 SerialTimeoutException，被 send() 的 except 接住当成发送失败。
        self._ser = serial.Serial(port, baudrate, timeout=timeout,
                                  write_timeout=write_timeout)
        self._ser.dtr = False   # 关键：禁用 DTR，避免打开串口触发 32U4 复位
        self._ser.rts = False
        time.sleep(0.5)          # 等 Pro Micro 稳定（打开串口可能已触发一次复位）
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.ok = True           # 最近一次发送成功了吗（链路健康度）
        self.fails = 0           # 累计失败次数（排查用）
        self.last_err = ""
        # 回执计数（同 kbd_client）：固件对每条指令都回 DONE/ERR，
        # 「发得出去但等不到回执」= 固件没在处理，是更隐蔽的卡死
        self.sent = 0
        self.replies = 0
        self.last_send_at = 0.0
        self.last_reply_at = 0.0
        #: ⭐ **还没等到回执的条数 + 最早那条发出的时刻**（同 `kbd_client` ✓）——
        #:   `silent_for()` 的判据就靠它俩（"最老那条等了多久" ✓ 见 `silence_seconds` ✓）。
        #:   ⚠ 2026-10-04 之前这里没有这两个字段 ⇒ 判据用的是"距上次**发送**多久" ✗
        #:   ⇒ 一直发就永远判不出死链（现场卡键 7 分 46 秒 ✓）。
        self._pending = 0
        self._rtt_t0 = None
        # 后台读线程：读走固件回传的 DONE/READY，否则串口输入缓冲堆积、
        # 固件 Serial.println 阻塞，整条链路卡死（表现为按键发不出）。
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def send(self, line):
        """发一条指令。返回 True = 已写出，False = 这次没发出去（见 kbd_client.send）。"""
        with self._lock:
            try:
                self._ser.write((line + "\n").encode())
                self._ser.flush()
                self.ok = True
                self.sent += 1
                self.last_send_at = time.monotonic()
                # 回执起点：只记**最早那条还没对上回执的**（口径同 `kbd_client.send` ✓）
                self._pending += 1
                if self._rtt_t0 is None:
                    self._rtt_t0 = time.perf_counter()
                return True
            except Exception as e:
                self.ok = False
                self.fails += 1
                self.last_err = "%s: %s" % (type(e).__name__, e)
                return False

    def silent_for(self):
        """多久没收到固件回执了（秒）。0 = 正常（或无从判断）。

        ⚠ **判据与 TLS 那条共用同一段**（`kbd_client.silence_seconds` ✓）—— 这个文件原来
          自己抄了一份，抄的还是**错的**那版（"距上次**发送**多久" ✗ 一直发就恒 0.1 秒 ⇒
          判不出死链 ✗ 2026-10-04 现场卡键 7 分 46 秒 ✓）⇒ 两处各写一份必然漂 ✓。
        """
        return _silence_seconds(self.sent, self._rtt_t0, time.perf_counter())

    def _drain(self):
        """持续读走固件回传，避免输入缓冲堆积；顺带统计回执（并推进 `silent_for` 的账 ✓）。"""
        while not self._stop.is_set():
            try:
                data = self._ser.read(256)
            except Exception:
                break
            if data:
                n = data.count(b"DONE") + data.count(b"ERR")
                if n:
                    self.replies += n
                    self.last_reply_at = time.monotonic()
                    # 账减掉（口径同 `kbd_client._take_rtt` ✓）：减完还有 ⇒ 起点挪到"现在"，
                    # 没有 ⇒ 清空 ✓（不清的话空闲期会被算成延迟 ✗）
                    self._pending = max(0, self._pending - n)
                    self._rtt_t0 = time.perf_counter() if self._pending > 0 else None

    def close(self):
        self._stop.set()
        try:
            self._ser.close()
        except Exception:
            pass
