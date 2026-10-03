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
from pathlib import Path

#: ⭐ **A 机日志在 B 机的落点**（用户 2026-10-03 ✓ 原话："你能让 A 机的log都往B机发吗？
#: 这样我就不用老切换了"）。A 机的 `relay_trace.log` 里正是"鼠标节拍 / tcp->serial /
#: 串口错误"这些**只能在被控机上看到**的东西 ✓ ⇒ 现在 relay 顺手回传、B 侧落在这 ✓。
#: ⚠ 名字带 `A_` 前缀：两台机器上都有一个同名文件最容易看错 ✗（A 机那份还叫
#:   `relay_trace.log` ✓ 这份是"从 A 转发来的" ✓）。
A_LOG_PATH = Path(__file__).resolve().parent / "A_relay_trace.log"
#: 单文件上限（超了砍半 ✓ 与 A 侧 `relay.py` 的 `_trim` 同口径 ✓）。
A_LOG_MAX = 4 * 1024 * 1024
#: A 侧回传行的前缀（`relay.py::_push_to_client` 一处实现 ✓ 两边必须一致 ✗）。
A_LOG_PREFIX = b"#LOG "


def _perf():
    """性能打点（**可选**）：`core.perf` 只在控制机（B）上有 —— 这个模块要保持能单独
    在别的机器上跑（它是"发指令"那半边 ✓），拿不到就静默不算 ✓（绝不因此报错 ✗）。"""
    try:
        from core import perf
        return perf
    except Exception:                       # noqa: BLE001
        return None


def set_nodelay(sock):
    """给一条 TCP 连接打开 **`TCP_NODELAY`** —— 按键这条通道专治 Nagle 攒包 ✓。

    为什么必须有（用户 2026-09-29 链路提效 ✓）：这是**小包 + 实时**的通道（`PRESS L` 就
    十来个字节 ✗），而 Nagle 的算法正是"小包先攒着、等前面那个 ACK 回来再发" ⇒ 最坏多等
    **一个 RTT（局域网也有几毫秒，跨交换机几十毫秒级）** ✗，而且它还是**串行累积**的
    （每条指令都等前一条的回执 ✓）。这条链路上每一条指令都是"现在就要按下"，
    没有任何理由攒 ✓。

    `OSError` 一律吞掉：老系统 / 非常规套接字上设不上也不该让连接失败 ✓
    （`kbd_client` 是"发指令"那半边，宁可没有 NODELAY 也不能不发 ✗）。
    """
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return True
    except (OSError, AttributeError):
        return False


class KbdClient:
    def __init__(self, host, port, cafile, timeout=5.0):
        ctx = ssl.create_default_context(cafile=cafile)
        ctx.check_hostname = False          # 自签证书不校主机名
        raw = socket.create_connection((host, port), timeout=timeout)
        # ⭐ **在包 TLS 之前设**（`setsockopt` 对 TLS 包装后的对象也能用，但放在这儿更直白 ✓）
        set_nodelay(raw)
        self._sock = ctx.wrap_socket(raw, server_hostname=host)
        # **超时只在这里设一次**：原来 send() 每次都调 settimeout(2.0)，而读线程
        # 同时在这个 socket 上 recv —— 多线程改同一个 socket 的超时会打乱正在进行的
        # 收发（TLS 层尤其敏感）。收发都用 2 秒，读线程本来就把超时当「空闲」处理。
        self._sock.settimeout(2.0)
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
        #: ⭐ **还没等到回执的指令条数 + 最早那条发出的时刻** —— 用来算
        #: 「B 发出 → A 机固件执行完回执」的**往返延迟**（用户 2026-09-29 链路提效 ✓）。
        #: 这是 A→B→A 闭环里**唯一从没量过**的一段：`send_ms` 只说明"交给内核"花了多久，
        #: 真正决定"按键什么时候生效"的是这段往返（+ 游戏的输入响应）✓。
        #: 固件对**每条**指令回一条 `DONE`/`ERR`（见 `_drain` ✓）⇒ 一对一，账能对上 ✓。
        self._pending = 0
        self._rtt_t0 = None
        #: ⭐ A 机日志回传（2026-10-03 ✓ 见 `A_LOG_PATH`）：半行缓冲 + 计数 + 每连接一条
        #:   抬头行（不然日志里两个连接的行会接在一起分不清 ✓）。
        self._log_carry = b""
        self.a_log_lines = 0
        self._a_log_banner = False
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
                if self._dirty:
                    data = b"\n" + data     # 切掉对端残留的半行
                self._sock.sendall(data)
                self._dirty = False
                self.ok = True
                self.sent += 1
                self.last_send_at = time.monotonic()
                # RTT 起点：只记**最早那条还没对上回执的**（这样量到的就是窗口里最久的
                # 那一条 ⇒ 天然偏保守 ✓，不会把"刚发出去"当成整段往返 ✗）
                self._pending += 1
                if self._rtt_t0 is None:
                    self._rtt_t0 = time.perf_counter()
                self._sample_pending()
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

    def _take_rtt(self, n):
        """收到 n 条回执 ⇒ 记一次往返延迟 + 把"还没对上回执"的账减掉 ✓（一处口径 ✓）。

        · 有在等的指令 ⇒ `perf.ms("kbd_rtt_ms", 最早那条发出时刻)` ✓；
        · 减完之后**还有**没对的 ⇒ 起点挪到"现在"（剩下的最早那条 ≈ 此刻刚发 ✓，
          误差 ≤ 一次采样间隔，可接受 ✓）；没有 ⇒ 清空，等下一次真发指令再起算 ✓
          （**绝不用空闲期的旧起点** ✗：那会把"没人按键的几十秒"算成延迟 ✗）。
        """
        if n <= 0:
            return
        if self._pending > 0 and self._rtt_t0 is not None:
            _p = _perf()
            if _p is not None:
                _p.ms("kbd_rtt_ms", self._rtt_t0)
        self._pending = max(0, self._pending - n)
        self._rtt_t0 = time.perf_counter() if self._pending > 0 else None
        self._sample_pending()

    def _sample_pending(self):
        """把「**还没等到回执的指令条数**」打点进 `perf.log`（用户 2026-10-04 ✓ 见 `_pending`）。

        ⭐ 这是"**指令堆积**"在 B 侧最直接的量化：固件/串口忙时，B 发得出去、回执回不来
        ⇒ 这个数一路涨 ✓；而 `kbd_rtt_ms` 只说"**最老的**那条等了多久"、看不出**攒了几条** ✗
        ⇒ 两个一起看才能定案（配 A 侧 `KBD 节拍` 那行的"发送率 vs 完成率"✓）。

        ⚠ `min_gap=1.0`：它是**连续量**（每发一条就变，~10 条/秒 ✗）⇒ 不定频就是每拍一条、
        把 `perf.log` 刷爆 ✓（口径同 `chase_dist` / `mmap_age_ms` ✓）。
        ⚠ 吞异常：打点不许把发指令这条路搞挂 ✗（拿不到 `core.perf` 的老环境也是静默 ✓）。
        """
        _p = _perf()
        if _p is None:
            return
        try:
            _p.sample("kbd_pending", int(self._pending), min_gap=1.0)
        except Exception:                       # noqa: BLE001
            pass

    def _take_a_logs(self, data):
        """把 A 机**回传的日志行**（`#LOG …`）挑出来落盘，返回**剩下的**给回执计数用 ✓。

        用户原话："你能让 A 机的log都往B机发吗？这样我就不用老切换了" ✓ —— A 机 relay 的
        `trace()` 现在顺手回传（`relay.py::_push_to_client` ✓），这里落进 `A_LOG_PATH` ✓。

        ⚠ 三件必须做对（少一件就会把**别的**功能弄坏 ✗）：
          · **挑走的行不再参与 `DONE`/`ERR` 计数** ✗ —— 否则 `kbd_rtt_ms` 和"链路卡死"判据
            （`silent_for`，看的就是回执）会被**日志里的字眼**骗到 ✓；
          · **只留"可能是日志行开头"的半行**（TCP 切包是常态 ✓）—— 普通半行（比如 `DON`）
            **绝不能扣住** ✗（扣住就拖慢回执计数 = 把 RTT 量歪 ✓）；
          · 落盘失败**不许把读线程搞挂** ✗（它一退，回程缓冲堆满 ⇒ 整条链连锁卡死 ✓
            见本模块 `_drain` 的文档 ✓）。
        """
        if not data:
            return data
        try:
            body = self._log_carry + bytes(data)
        except Exception:                           # noqa: BLE001
            return data
        if A_LOG_PREFIX not in body and not body.startswith(b"#"):
            self._log_carry = b""
            return data                             # 一条日志都没有 ⇒ 一个字不改 ✓
        lines = body.split(b"\n")
        keep = b""
        if lines and lines[-1] and not body.endswith(b"\n"):
            tail = lines[-1]
            if tail.startswith(b"#") or A_LOG_PREFIX.startswith(tail):
                keep = lines.pop()                  # 只可能是半条日志行 ✓
        self._log_carry = keep
        rest = []
        for raw in lines:
            if raw.startswith(A_LOG_PREFIX):
                self._append_a_log(raw[len(A_LOG_PREFIX):])
            elif raw:
                rest.append(raw)
        return b"\n".join(rest)                     # 回执计数在**剩下的**上做 ✓

    def _append_a_log(self, raw):
        """写一行到 `A_LOG_PATH`（超限砍半 ✓；失败静默 ✓ —— 只是方便，不是功能 ✗）。"""
        try:
            text = raw.decode("utf-8", "replace")
        except Exception:                           # noqa: BLE001
            return
        try:
            p = A_LOG_PATH
            with open(p, "a", encoding="utf-8") as fh:
                if not self._a_log_banner:
                    self._a_log_banner = True
                    fh.write("=== A 机 relay 日志（转发 · 本连接 %s）===\n"
                             % time.strftime("%Y-%m-%d %H:%M:%S"))
                fh.write(text + "\n")
            self.a_log_lines += 1
            if p.stat().st_size > A_LOG_MAX:
                blob = p.read_bytes()
                p.write_bytes(blob[len(blob) // 2:])    # 砍半（同 A 侧口径 ✓）
        except Exception:                           # noqa: BLE001
            pass

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
                # ⭐ **先把 A 机日志挑走**再数回执（2026-10-03 ✓ 见 `_take_a_logs`）——
                #   挑走的行不许参与 `DONE`/`ERR` 计数 ✗（否则日志字眼会骗到 RTT / 死链判据 ✓）
                data = self._take_a_logs(data)
                n = data.count(b"DONE") + data.count(b"ERR")
                if n:
                    self.replies += n
                    self.last_reply_at = time.monotonic()
                    self._take_rtt(n)
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
