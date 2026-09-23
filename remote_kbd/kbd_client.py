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
import time


class KbdClient:
    def __init__(self, host, port, cafile, timeout=5.0):
        ctx = ssl.create_default_context(cafile=cafile)
        ctx.check_hostname = False          # 自签证书不校主机名
        raw = socket.create_connection((host, port), timeout=timeout)
        self._sock = ctx.wrap_socket(raw, server_hostname=host)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dirty = False      # 上一条可能只发出去半行（超时中断）→ 下一条要重新对齐
        self.ok = True           # 最近一次发送成功了吗（链路健康度）
        self.fails = 0           # 累计失败次数（排查用）
        self.last_err = ""
        # 回执计数：固件对每条指令都会回 DONE/ERR。靠它判断「TCP 发得出去、
        # 但固件根本没在处理」这种更隐蔽的卡死（relay 卡在串口写上时就是这样，
        # 发送不报错、命令却全都石沉大海）。
        self.sent = 0
        self.replies = 0
        self.last_send_at = 0.0
        self.last_reply_at = 0.0
        # 后台读线程：relay 会把固件的 DONE 应答回传，这里持续读走丢弃，
        # 否则回传缓冲被 DONE 塞满后，relay/固件/控制机整条链路会连锁卡死
        # （表现为：按键发不出、推理丢帧暴增、RELEASE 丢失导致卡键）。
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def send(self, line):
        """发一条指令。返回 True = 已交给内核，False = 这次没发出去。

        **为什么不吞异常就完事**：发送超时可能只发出去**半行**。对端是按行解析的，
        残留的半个命令会和下一条命令粘成一条无效指令 —— 那条命令就白丢了，
        而丢一条 RELEASE 在游戏里就是「那个键一直按着」。所以失败时标记 dirty，
        下一条命令前面补一个换行，把对端的解析重新对齐。

        返回值是链路健康度的来源：之前一律 `pass`，通道死了上层根本不知道，
        还在那儿一直发 —— 表现就是「停自动没用、手动也没用，只能关 GUI」。
        """
        data = (line + "\n").encode()
        with self._lock:
            try:
                self._sock.settimeout(2.0)
                if self._dirty:
                    data = b"\n" + data     # 切掉对端残留的半行
                self._sock.sendall(data)
                self._dirty = False
                self.ok = True
                self.sent += 1
                self.last_send_at = time.monotonic()
                return True
            except Exception as e:
                self._dirty = True
                self.ok = False
                self.fails += 1
                self.last_err = "%s: %s" % (type(e).__name__, e)
                return False

    def silent_for(self):
        """多久没收到固件回执了（秒）。0 = 正常（或无从判断）。

        判定「链路卡死」的核心：**发过指令、但迟迟等不到回执**。
        只靠「发送有没有报错」是不够的 —— relay 卡在串口写上时，我们的 TCP
        发送照样成功，命令却全都没到固件，那就是「无限左走 + 停自动无效」
        的真实场景（发什么都当成功，实际什么也没发生）。
        """
        if self.sent == 0:
            return 0.0
        now = time.monotonic()
        if now - self.last_send_at > 30.0:
            return 0.0          # 近期没发过指令，无从判断，别拿旧账报错
        if self.last_reply_at >= self.last_send_at:
            return 0.0          # 最后一条指令有回执 → 正常
        return now - self.last_send_at

    def _drain(self):
        """读走 relay 回传的应答，避免回传缓冲堆积。

        线程一退出，回传缓冲就会堆满，最后把 relay 的串口->TCP 方向堵死 ——
        整条链连锁卡住。所以退出时**必须把链路标记成不健康**，让上层（agent 的
        周期自检 / 手动重置按钮）知道要重连，而不是继续对着一条死链发命令。
        """
        while not self._stop.is_set():
            try:
                data = self._sock.recv(4096)
                if not data:
                    self.ok = False     # 连接被对端关闭
                    self.last_err = "连接被对端关闭"
                    break
                n = data.count(b"DONE") + data.count(b"ERR")
                if n:
                    self.replies += n
                    self.last_reply_at = time.monotonic()
            except socket.timeout:
                continue                # 空闲超时：继续等，别退出
            except Exception as e:
                self.ok = False
                self.last_err = "读线程退出：%s: %s" % (type(e).__name__, e)
                break

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass
