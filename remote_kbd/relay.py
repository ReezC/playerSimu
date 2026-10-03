"""远程键盘中继（跑在游戏机 / 被控机）。

TLS 服务端：收控制机 agent 发来的按键指令，原样转发到 Pro Micro 的 USB 串口，
并把 Pro Micro 的应答（DONE/ERR）回传给控制机。

用法：
    python relay.py --serial COM5 --port 9000 --cert certs/cert.pem --key certs/key.pem
"""

import argparse
import faulthandler
import select
import socket
import ssl
import threading
import time
from collections import deque
from pathlib import Path

import serial

# `set_nodelay` 与客户端**共用一份实现** ✓。两种跑法都要能用：
#   · 部署时是 `python relay.py …`（脚本，同目录 import ✓）；
#   · 从仓库根 `python -m remote_kbd.relay`（包内相对 import ✓）。
try:
    from .kbd_client import set_nodelay as _set_nodelay
except ImportError:                       # noqa: BLE001 —— 当脚本跑时走这条 ✓
    from kbd_client import set_nodelay as _set_nodelay

# ---- 崩溃取证 ---------------------------------------------------------------
#
# 这个进程曾经以 3221225477（0xC0000005 访问违规）**原生崩死**过：Python 的
# try/except 抓不住这类崩溃（崩在 C 扩展/驱动里），进程当场消失，控制台只剩
# 「意外退出」四个字 —— 现象是「界面还显示连接成功，但手动输入和鼠标都传不过去」。
# 所以补两件事：
#   · faulthandler 在信号层面把 Python 栈写进 relay_crash.log（哪一行死的）
#   · 每条指令 + 每次串口错误追加进 relay_trace.log（只留最近一段，不会长胖）
#     —— 崩溃前最后在做什么，看它。
LOG_DIR = Path(__file__).resolve().parent
TRACE_MAX = 256 * 1024          # 超了就砍掉前半
_trace_fh = None
_crash_fh = None


def _trim(path, limit):
    """文件超过 limit 就把前半砍掉（保留最近的历史）。"""
    try:
        if path.stat().st_size <= limit:
            return
        blocks = path.read_text(encoding="utf-8", errors="replace").splitlines()
        path.write_text("\n".join(blocks[len(blocks) // 2:]) + "\n",
                        encoding="utf-8")
    except Exception:
        pass


def enable_forensics(log_dir=None):
    """打开崩溃取证（启动时调一次）。log_dir 只给测试用。"""
    global _trace_fh, _crash_fh, LOG_DIR
    if log_dir is not None:
        LOG_DIR = Path(log_dir)
    try:
        _crash_fh = open(LOG_DIR / "relay_crash.log", "a", buffering=1,
                         encoding="utf-8")
        _crash_fh.write("\n=== relay start %s ===\n"
                        % time.strftime("%Y-%m-%d %H:%M:%S"))
        faulthandler.enable(_crash_fh)      # 原生崩溃时把栈写这里
        _trace_fh = open(LOG_DIR / "relay_trace.log", "a", buffering=1,
                         encoding="utf-8")
    except Exception:
        pass
    return LOG_DIR


#: ⭐ **A 机日志的回传口**：当前连着的那个客户端（`bridge` 里设 / 清 ✓）。
#: 用户 2026-10-03 ✓ 原话："你能让 A 机的log都往B机发吗？这样我就不用老切换了" ——
#: A 机的 `relay_trace.log` 只在被控机上 ✓，每次要看都得切过去 ✗ ⇒ 顺手挂到**已有的
#: 回程**（固件回执那条 TCP ✓）上，B 侧收到就落盘（`remote_kbd/kbd_client.py` ✓）。
_LOG_SINK = None
#: 所有对客户端的 `sendall` **共用这把锁**：回程现在有**两个写者** —— `ser_to_tcp`
#: 写固件回执 ✓、`trace()` 写日志回传 ✓ —— 而 **TLS 记录不许交错** ✗（两个线程同时
#: `sendall` 会写出坏记录 ⇒ 整条链烂掉 ✓）。
_SEND_LOCK = threading.Lock()


#: 回传**队列**（见 `_push_to_client`）：**绝不许让回传挡住命令转发** ✗
#:   —— 用户 2026-10-04 ✓："现在老有指令堆积的问题"。
#:   原来 `_push_to_client` 是**同步 `sendall`** ✗，而 `trace()` 就在转发那条路上
#:   （`bridge()` 里 `link.write(data)` **之前** ✓）⇒ 客户端（B）读得慢时 `sendall`
#:   最长要阻塞 5 秒（那 socket 的写超时 ✓）⇒ **这一拍的命令根本发不到固件** ✗
#:   ⇒ 表现就是"指令堆积 / 按键延迟突然很大"✓。
#:   ⇒ 改成：**排队 + 独立线程发**，队列满了**丢最旧**（回传只是方便，宁可少几行 ✗
#:     也绝不拖住命令 ✓）。
LOG_BACKLOG_MAX = 200
_QUEUE = deque()
_QLOCK = threading.Lock()
_QWAKE = threading.Event()
_QDROP = 0                # 因队列满被丢掉的日志行数（给用例/排查看 ✓）
_QSENT = 0                # 真发出去的日志行数
_Q_THREAD = None


def _sender_loop():
    """把队列里的日志行一条条发给客户端（**独立线程** ✓，慢也只慢它自己 ✓）。"""
    global _QSENT
    while True:
        try:
            _QWAKE.wait(0.5)
            _QWAKE.clear()
            while True:
                with _QLOCK:
                    if not _QUEUE:
                        break
                    line = _QUEUE.popleft()
                conn = _LOG_SINK
                if conn is None:
                    continue                       # 客户端走了 ⇒ 丢掉这几行 ✓
                try:
                    with _SEND_LOCK:
                        conn.sendall(line)
                    _QSENT += 1
                except Exception:                  # noqa: BLE001 —— 回传失败不算事 ✓
                    pass
        except Exception:                          # noqa: BLE001 —— 发送线程绝不许死 ✓
            time.sleep(0.05)


def _push_to_client(text):
    """把一行日志**排队**回传给 B 机（用户 2026-10-03 ✓ 见 `_LOG_SINK`）。

    ⚠ 四条纪律：
      · 只在**有客户端连着**时排队 ✓（没连就只是本地文件 ✓，`relay_trace.log` 照旧写全 ✓）；
      · **排队 + 独立线程发 ⇒ 绝不阻塞调用方** ✓（调用方是转发那条路 ✗ 见 `LOG_BACKLOG_MAX`）；
      · 队列满了**丢最旧**（`_QDROP` 计数 ✓）—— 回传丢几行无害，拖住命令有害 ✓；
      · **不许在这里再调 `trace()`** ✗（会自激 ✓）。
    """
    global _QDROP, _Q_THREAD
    if _LOG_SINK is None:
        return
    if _Q_THREAD is None or not _Q_THREAD.is_alive():
        _Q_THREAD = threading.Thread(target=_sender_loop, daemon=True,
                                     name="relay-log-return")
        _Q_THREAD.start()
    line = ("#LOG %s %s\n" % (_ts(), text)).encode("utf-8", "replace")
    with _QLOCK:
        if len(_QUEUE) >= LOG_BACKLOG_MAX:
            _QUEUE.popleft()
            _QDROP += 1
        _QUEUE.append(line)
    _QWAKE.set()


def _ts():
    """日志时间戳：**到毫秒**（2026-10-03 ✓）。

    为什么必须毫秒：原来只写 `%H:%M:%S` ✗ ⇒ 「一段一段一顿一顿」那种现象发生在
    **20~30 ms** 的尺度上 ✓ —— 秒级精度下那些缝**完全看不见** ✗（现场问的就是这件事 ✓）。
    """
    return time.strftime("%H:%M:%S") + ".%03d" % (int(time.time() * 1000) % 1000)


def trace(text):
    """记一条指令级日志（超限自动瘦身）+ **顺手回传给 B 机**（2026-10-03 ✓）。"""
    global _trace_fh
    # ⭐ **先回传**（用户 2026-10-03 ✓："这样我就不用老切换了"）——
    #   ⚠ 必须**与本地文件无关** ✗：`_trace_fh` 为 None 时（目录建不出来 / 打开失败 ✓）
    #   原来会直接 `return` ⇒ 回传也一起死 ✗ ⇒ 人还是得切到 A 机去看（甚至没得看 ✓）。
    #   实测就是这么发现的（探针里把 `_trace_fh` 设成 None，"推出"是空的 ✓）。
    #   ⚠⚠ **只推这一次**（2026-10-04 ✓ 现场实测：B 收到的 `A_relay_trace.log` 里
    #      **每行都是双份** ✗）—— 原来下面收尾处**又推了一次** ✗ ⇒ 回程（和固件回执
    #      同一条 TLS ✓）白占一半带宽，而回程一堵就会拖住命令转发（见 `LOG_BACKLOG_MAX` ✓）。
    _push_to_client(text)
    if _trace_fh is None:
        return
    try:
        _trace_fh.write("%s %s\n" % (_ts(), text))
        _trace_fh.flush()
        if _trace_fh.tell() > TRACE_MAX:
            _trace_fh.close()
            _trim(LOG_DIR / "relay_trace.log", TRACE_MAX // 2)
            _trace_fh = open(LOG_DIR / "relay_trace.log", "a", buffering=1,
                             encoding="utf-8")
    except Exception:
        pass


class SerialLink:
    """串口 + 一把锁 + 出错自动重开。

    **为什么要锁**：relay 里有两个线程同时用它 —— 读线程（Pro Micro → TCP）和
    主线程（TCP → Pro Micro）。pyserial 的 Serial 不是线程安全的，Windows 上
    两个线程同时读写同一个句柄可能直接原生崩溃（0xC0000005，Python 层抓不住，
    进程当场消失）。用 RLock：read/write/reopen 互相不重叠。

    **为什么要自动重开**：32U4 一旦被重新枚举（**重新烧固件**、拔插、复位），
    原来的句柄就失效了，继续读写它正是上面那种崩溃的高发场景。重开之后最多丢
    一条指令，而不是让整个中继死掉。
    """

    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self._lock = threading.RLock()
        self._opened_at = 0.0
        self._ser = self._open()

    def _open(self):
        ser = serial.Serial(self.port, self.baud, timeout=0.3, write_timeout=1.0)
        try:
            ser.dtr = False      # 关键：禁用 DTR，避免打开串口触发 32U4 复位
            ser.rts = False
        except Exception:
            pass
        self._opened_at = time.monotonic()
        return ser

    def reopen(self, why=""):
        """关掉旧句柄重开一个。1 秒内不重复重开，免得陷入「开-失败-开」循环。"""
        with self._lock:
            if time.monotonic() - self._opened_at < 1.0:
                return False
            try:
                self._ser.close()
            except Exception:
                pass
            try:
                self._ser = self._open()
                trace("serial reopened (%s)" % why)
                return True
            except Exception as e:
                trace("serial reopen FAILED (%s): %s" % (why, e))
                return False

    def read(self, n):
        with self._lock:
            try:
                # **非阻塞读**：先问驱动有多少字节，没有就立刻返回。
                #
                # 绝不能占着锁去 read() —— read 在没数据时会**阻塞到 timeout
                # （0.3 秒）**，而写路径用的是同一把锁，于是：
                #   · 主线程写一条指令要排队最长 0.3 秒 → 鼠标延迟暴涨
                #   · RLock 不保证公平，读线程循环抢锁能把写**饿死** →
                #     B 侧 2 秒发送超时 → PRESS 白丢
                #   · 现象就是「连接是通的（RELEASEALL 挤进去了），但按键全无效、
                #     鼠标延迟巨大、A 机光标一直一点点地爬」
                #   （实测踩过；改动前读写互不相关，是我加锁时带进来的。）
                waiting = self._ser.in_waiting
                if not waiting:
                    return b""
                return self._ser.read(min(n, waiting))
            except Exception as e:
                trace("serial read error: %s" % e)
                self.reopen("read")
                return b""

    def write(self, data):
        with self._lock:
            try:
                return self._ser.write(data)
            except Exception as e:
                trace("serial write error: %s" % e)
                if self.reopen("write"):
                    try:
                        return self._ser.write(data)   # 重开后补发一次
                    except Exception as e2:
                        trace("serial write retry failed: %s" % e2)
                raise

    def close(self):
        with self._lock:
            try:
                self._ser.close()
            except Exception:
                pass


def _med(xs):
    """中位数（空样本 ⇒ 0 ✓）—— `MoveStats` / `KbdStats` **共用这三件** ✓。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[len(s) // 2])


def _maxv(xs):
    return float(max(xs)) if xs else 0.0


def _p95(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(0.95 * (len(s) - 1)))])


class MoveStats:
    """⭐ 「**MOVE 命令**以什么节拍到达 A 机」——只统计鼠标，不掺键盘（2026-10-03 ✓）。

    用户现场原话："看起来现象是在 A 机上一段一段一顿一顿的指令汇报很离散，B 机卡不卡
    其实没关系"。**判据得先有**：relay 原来只把 `tcp->serial <数据>` 按**秒**记进
    `relay_trace.log` ✗ ⇒ 那个"离散"根本量不出来 ✓；`perf.log` 里也没有鼠标专用计数 ✗
    （`MOVE` 和键盘混在 `send` 里 ✓）⇒ 只能靠猜 ✓。

    每 `report_sec` 秒往 `relay_trace.log` 写**一行**（现场只需抓这一个文件 ✓）：

      · `n`         这一段 MOVE **条数 + 条/秒** ⇒ 实际命令率（触控板正常 60~125/s ✓
                    远低于它 ⇒ B 侧在**攒着发** ✗ 见 B 侧 `int()` 量化 + 主线程节流 ✓）
      · `per_batch` **一次 recv 里带几条 MOVE** ⇒ 「**成批**」的直接证据 ✓（理想 ≈1；
                    出现 3~5 ⇒ 那就是「一段一段」✗）
      · `gap_ms`    相邻两条 MOVE 的**到达间隔** ⇒ 「**一顿一顿**」的直接证据 ✓
                    （中位小而 p95/最大 大 = 一撮一撮 ✗ 理想是绕节拍的小抖动 ✓）
      · `step_px`   每条 MOVE 的 `|dx|+|dy|` ⇒ **中位 1~2 = 量化台阶** ✓
                    （B 侧 `int()` + 余数累积 ⇒ 慢速时几个事件才凑够 1 像素 ✓）

    ⚠ **只看完整行**：一条 `MOVE …\n` 可能被 TCP 切成两半 ⇒ 半行留到下一批再算 ✗
      （不然"批内条数 / 间隔"全歪 ✓ 见 `_carry`）。
    ⚠ 统计**不许改变转发行为** ✗：它只读 `data`；`link.write(data)` 照旧原样 ✓。
    ⚠ 只认 `MOVE` ✓：键指令混进来会把率算歪 ✗。
    """

    def __init__(self, report_sec=5.0):
        self.report_sec = float(report_sec)
        self._carry = b""
        self._n = 0
        self._batches = 0            # 这一段里"带 MOVE 的 recv"次数 ✓
        self._per_batch = []
        self._gaps = []
        self._steps = []
        self._last_t = None
        self._t0 = None

    # ---- 喂数据（在 relay 的 tcp->serial 那条路上调 ✓）----
    def feed(self, data, now):
        """把一次 `recv` 拿到的原始字节喂进来（**只统计 MOVE** ✓）。"""
        if self._t0 is None:
            self._t0 = now
        try:
            body = self._carry + bytes(data)
        except Exception:                           # noqa: BLE001 —— 统计不许把桥搞挂 ✗
            return
        lines, _, rest = body.rpartition(b"\n")
        self._carry = rest if rest and len(rest) < 4096 else b""
        n_in_batch = 0
        for raw in lines.split(b"\n"):
            parts = raw.strip().split()
            if not parts or parts[0] != b"MOVE":
                continue
            try:
                step = abs(int(parts[1])) + abs(int(parts[2]))
            except Exception:                       # noqa: BLE001
                step = 0
            if self._last_t is not None:
                self._gaps.append((now - self._last_t) * 1000.0)
            self._last_t = now
            self._n += 1
            n_in_batch += 1
            self._steps.append(step)
        if n_in_batch:
            self._batches += 1
            self._per_batch.append(n_in_batch)

    # ---- 取报告（到点了返回一行文本，否则 None ✓）----
    def maybe_report(self, now=None, force=False):
        now = time.perf_counter() if now is None else now
        if self._t0 is None:
            return None
        if not force and (now - self._t0) < self.report_sec:
            return None
        span = max(1e-6, now - self._t0)
        line = ("MOVE 节拍 %.0fs n=%d（%.1f 条/秒）批=%d 批内中位%d最大%d "
                "间隔中位%.1f p95 %.1f 最大%.1fms 步长中位%d最大%d"
                % (span, self._n, self._n / span, self._batches,
                   int(_med(self._per_batch)), int(_maxv(self._per_batch)),
                   _med(self._gaps), _p95(self._gaps), _maxv(self._gaps),
                   int(_med(self._steps)), int(_maxv(self._steps))))
        self._n = 0
        self._batches = 0
        self._per_batch = []
        self._gaps = []
        self._steps = []
        self._t0 = now
        return line

    def reset(self):
        """别把"上一段"的样本带进下一段 ✓（换客户端 / 重连时调 ✓）。"""
        self.report_sec = float(self.report_sec)
        self._carry = b""
        self._n = 0
        self._batches = 0
        self._per_batch = []
        self._gaps = []
        self._steps = []
        self._last_t = None
        self._t0 = None

    # ---- 小工具（空样本安全 ✓）----


class KbdStats:
    """⭐⭐ 「**键盘指令**这条通道忙不忙、积压在哪一段」——`MoveStats` 的键盘版（2026-10-04 ✓）。

    用户原话："**现在老有指令堆积的问题，能否做一些 A 向 B 汇报的东西？**" ✓。

    为什么单开一条（原来一个字都不统计 ✗）：`MoveStats` **只认 `MOVE`** ✓（键盘混进去会把
    鼠标率算歪 ✗），于是**键盘这条**在 A 侧完全看不见 —— 而"堆积"正是在这条上：
    B 发的 `PRESS/TAP/…` 走 relay 原样写进 115200 串口，固件**逐条执行**（`TAP` 里还有
    `delay` ✓）⇒ 发得比执行快时，命令就积压在**OS 串口缓冲 / 固件 RX** 里 ✓。

    每 `report_sec` 秒往 trace 写一行（⇒ **自动回传 B** ✓，落 `remote_kbd/A_relay_trace.log`）：

      · `n`        这一段键盘**条数 + 条/秒**（B 的发包率 ✓）
      · `批内`     一次 `recv` 里带几条 ⇒ 「**成批发**」的证据 ✓（理想 ≈1~2 ✓）
      · `间隔`     相邻两条**到达间隔** ⇒ 是否"一撮一撮" ✗
      · `写` ⭐     `link.write()`（写串口）的**耗时** 中位/最大 —— **这就是"下游堵没堵"的
                   直接证据** ✓：串口/固件忙 ⇒ 写变慢（`write_timeout=1.0`，最坏整秒 ✗）
      · `完成` ⭐   回程收到的 `DONE/ERR` 条数 + 条/秒 —— **固件真实完成率** ✓
                   ⇒ 与 `n`（发送率）一对比就能定案：**发 N 条/秒、完成 M 条/秒，N>M 就是在积压** ✓

    ⚠ 只看完整行（TCP 切包是常态 ✓ 半行留到下一批 ✓）；⚠ 统计**不许改变转发行为** ✗；
    ⚠ 只认**键盘**（`MOVE` 归 `MoveStats` ✓，两边都不许互相污染 ✓）。
    """

    #: 键盘指令的第一个词（与固件 `pro_micro.ino` 的那几个 `head == "…"` 对齐 ✓）。
    #: ⚠ **`MOVE` 归 `MoveStats`** ✓；`PRESSM`/`RELEASEM`（**鼠标左右键**）两边都**不**算 ✓
    #:   —— 它们是鼠标那条、量极小；混进"键盘节拍"会把"键盘积压"的故事搅浑 ✗
    #:   （真要量它们，加一条 MouseBtnStats 比塞进这里清楚 ✓）。
    KEYS = (b"PRESS", b"RELEASE", b"RELEASEALL", b"TAP", b"FIX", b"RND", b"HOLD")

    def __init__(self, report_sec=5.0):
        self.report_sec = float(report_sec)
        self._carry = b""
        self._n = 0
        self._batches = 0
        self._per_batch = []
        self._gaps = []
        self._wrs = []
        self._done = 0
        self._last_t = None
        self._t0 = None

    def feed(self, data, now, wr_ms=None, done=0):
        """喂一次 `recv` 的原始字节（**只统计键盘** ✓）+ 这次写串口花了多少毫秒 + 收到的回执数。"""
        if self._t0 is None:
            self._t0 = now
        if wr_ms is not None:
            try:
                self._wrs.append(float(wr_ms))
            except (TypeError, ValueError):
                pass
        if done:
            self._done += int(done)
        try:
            body = self._carry + bytes(data)
        except Exception:                           # noqa: BLE001 —— 统计不许把桥搞挂 ✗
            return
        lines, _, rest = body.rpartition(b"\n")
        self._carry = rest if rest and len(rest) < 4096 else b""
        n_in_batch = 0
        for raw in lines.split(b"\n"):
            parts = raw.strip().split()
            if not parts or parts[0] not in self.KEYS:
                continue
            if self._last_t is not None:
                self._gaps.append((now - self._last_t) * 1000.0)
            self._last_t = now
            self._n += 1
            n_in_batch += 1
        if n_in_batch:
            self._batches += 1
            self._per_batch.append(n_in_batch)

    def maybe_report(self, now=None, force=False):
        now = time.perf_counter() if now is None else now
        if self._t0 is None:
            return None
        if not force and (now - self._t0) < self.report_sec:
            return None
        span = max(1e-6, now - self._t0)
        line = ("KBD 节拍 %.0fs n=%d（%.1f 条/秒）批=%d 批内中位%d最大%d "
                "间隔中位%.1f p95 %.1f 最大%.1fms 写中位%.1f 最大%.1fms "
                "完成 %d（%.1f 条/秒）"
                % (span, self._n, self._n / span, self._batches,
                   int(_med(self._per_batch)), int(_maxv(self._per_batch)),
                   _med(self._gaps), _p95(self._gaps), _maxv(self._gaps),
                   _med(self._wrs), _maxv(self._wrs), self._done, self._done / span))
        self._n = 0
        self._batches = 0
        self._per_batch = []
        self._gaps = []
        self._wrs = []
        self._done = 0
        self._t0 = now
        return line

    def reset(self):
        """别把"上一段"的样本带进下一段 ✓（换客户端 / 重连时调 ✓）。"""
        self._carry = b""
        self._n = 0
        self._batches = 0
        self._per_batch = []
        self._gaps = []
        self._wrs = []
        self._done = 0
        self._last_t = None
        self._t0 = None


#: 全进程一份（relay 只管一个客户端 ✓）
MOVE_STATS = MoveStats()
#: 键盘那条（见 `KbdStats` ✓）
KBD_STATS = KbdStats()
#: 回程收到的 `DONE/ERR` 条数**暂存**：由 `ser_to_tcp` 累加、主循环**取走清零** ✓。
#:   为什么不直接喂给 `KBD_STATS`：那会变成**两个线程同时改它** ✗（`feed` 在动列表、
#:   `maybe_report` 在排序 ⇒ 采样本会互相踩 ✓）⇒ 这里只用"整数的 `+=` / 取走"
#:   （CPython 下够原子 ✓），统计对象**永远只由主循环一个线程碰** ✓。
_DONE_PENDING = 0


def _take_done():
    """取走暂存的固件回执条数并清零（见 `_DONE_PENDING` ✓）。"""
    global _DONE_PENDING
    n, _DONE_PENDING = _DONE_PENDING, 0
    return n


def bridge(conn, link):
    """双向桥：TCP -> 串口，串口 -> TCP。

    **两个写方向都设了超时，这是必须的。** 以前没有超时：只要有一头对方不读，
    写就会**永久阻塞**在那个线程里 —— 串口->TCP 卡住后这个线程再也不读串口，
    Pro Micro 的 Serial.println 跟着阻塞，固件就不再处理任何指令（包括
    RELEASEALL），整条链死透且**不会自愈**，只能断开重连。用户看到的「无限
    左走、停自动无效、只能关 GUI」就是这条链死透了。

    设了超时之后：写卡住 → 抛异常 → 退出桥循环 → 关连接等下一个客户端，
    链自己恢复（断开时还会补一条 RELEASEALL 松开卡住的键）。
    """
    stop = threading.Event()
    conn.settimeout(5.0)          # 客户端不读时别永久阻塞
    # ⭐ 回执（DONE/ERR）也是**小包 + 实时**：不关 Nagle 就会攒着等 ACK（最坏 ~40 ms ✗），
    #   而控制机正是拿"发指令→收到回执"在算 `kbd_rtt_ms`（用户 2026-09-29 链路提效 ✓）
    #   —— 让回执被 Nagle 拖住，量出来的往返就不干净了 ✗（`set_nodelay` 一处实现 ✓）。
    _set_nodelay(conn)
    # 换客户端 ⇒ 统计从头开始（别把上一条连接的样本并进来 ✓ 见 `MoveStats.reset`）
    MOVE_STATS.reset()
    KBD_STATS.reset()             # 键盘那条同理（见 `KbdStats` ✓）
    # ⭐ 日志回传口：本条连接期间，`trace()` 写的每一行也发给这个客户端
    #   （用户 2026-10-03 ✓ 见 `_LOG_SINK` / `_push_to_client`）。
    global _LOG_SINK, _DONE_PENDING
    _LOG_SINK = conn
    _DONE_PENDING = 0

    def ser_to_tcp():
        global _DONE_PENDING
        while not stop.is_set():
            try:
                data = link.read(256)
            except Exception as e:
                trace("serial->tcp 读失败，结束本连接: %s" % e)
                break
            if data:
                # ⭐ **固件的完成率**（`KbdStats` 要它）：回程这批里数 `DONE/ERR` ⇒ 暂存整数，
                #   主循环取走（统计对象只由主循环一个线程碰 ✓ 见 `_DONE_PENDING` ✓）。
                try:
                    _DONE_PENDING += data.count(b"DONE") + data.count(b"ERR")
                except Exception:                    # noqa: BLE001
                    pass
                try:
                    # ⚠ 回程**两个写者**（这里的固件回执 + `trace` 的日志回传）⇒ 必须
                    #   同一把锁 ✗（TLS 记录交错 = 整条链烂掉 ✓ 见 `_SEND_LOCK`）
                    with _SEND_LOCK:
                        conn.sendall(data)
                except Exception:
                    break
            else:
                # 没数据：让出 CPU（read 现在是非阻塞的，不然这里会空转满核）
                time.sleep(0.002)

    t = threading.Thread(target=ser_to_tcp, daemon=True)
    t.start()
    try:
        while True:
            # **先用 select 等可读、再 recv**，不要靠 recv 的超时来轮询空闲。
            # 原因：这是 TLS 连接，超时如果卡在**一条 TLS 记录中间**，SSL 状态就
            # 作废了，之后只能重连 —— B 机看到的正是
            #   SSLEOFError: EOF occurred in violation of protocol
            # （现象：休息几分钟后一按键就报这个，链再也回不来）。
            # select 只等「有数据可读」，空闲时直接回头继续，不打断任何记录。
            try:
                ready, _, _ = select.select([conn], [], [], 5.0)
            except Exception:
                break
            if not ready:
                continue          # 空闲：自动打怪停下时本来就没键要发
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue
            if not data:
                break
            trace("tcp->serial %s"
                  % data.decode("ascii", "replace").replace("\n", "|")[:60])
            # ⭐ **写串口耗时**（`KbdStats` 要它）：这就是"下游堵没堵"的直接证据 ✓
            #   （固件忙 ⇒ OS 串口缓冲满 ⇒ `write` 变慢，最坏撞上 `write_timeout=1.0` ✓）。
            _t_wr = time.perf_counter()
            link.write(data)
            _wr_ms = (time.perf_counter() - _t_wr) * 1000.0
            # ⭐ **鼠标节拍统计**（2026-10-03 ✓ 用户现场："A 机上一段一段一顿一顿的指令
            #   汇报很离散"）+ ⭐⭐ **键盘节拍/积压统计**（2026-10-04 ✓ 用户现场："现在老有
            #   指令堆积的问题，能否做一些 A 向 B 汇报的东西？"）：都只**读** `data` ✓、
            #   不动转发（`link.write` 上一行照旧原样 ✓）；到点就往 trace 里补一行
            #   ⇒ **自动回传 B** ✓ ⇒ 现场只需抓 `remote_kbd/A_relay_trace.log` ✓。
            #   ⚠ 包在 try 里：统计是"观测"，绝不许把桥搞挂 ✗。
            try:
                _now_stat = time.perf_counter()
                MOVE_STATS.feed(data, _now_stat)
                _stat_line = MOVE_STATS.maybe_report(now=_now_stat)
                if _stat_line:
                    trace(_stat_line)
                KBD_STATS.feed(data, _now_stat, wr_ms=_wr_ms, done=_take_done())
                _kbd_line = KBD_STATS.maybe_report(now=_now_stat)
                if _kbd_line:
                    trace(_kbd_line)
            except Exception:
                pass
    except Exception as e:
        trace("tcp->serial 写失败，结束本连接: %s" % e)
    finally:
        stop.set()
        # 客户端走了 ⇒ 日志回传口一起清掉 ✓（下一个连上来时再设 ✓）
        _LOG_SINK = None
        # ⭐⭐ 连接断开时**清空串口缓冲 + 重开句柄**（2026-10-02 ✓ —— 用户报
        #   "必须断开 ProMicro 才拯救"）：光发 RELEASEALL 不够 —— 它只清固件侧
        #   held 表，但**积压在 OS 串口驱动缓冲里的命令还在**（固件不读时 write
        #   照样成功返回 ✗）⇒ B 机重连后 relay 又把积压的写出去 ⇒ 还是卡 ✗。
        #   reopen 关掉旧句柄重开 ⇒ OS 缓冲清空 ⇒ 积压的命令全丢 ✓（客户端都
        #   断开了，丢积压是对的 —— 本来就是异常退出 ✓）。然后在干净的新句柄
        #   上发 RELEASEALL，让固件回到空状态。
        link.reopen("client disconnect")
        try:
            link.write(b"RELEASEALL\n")
        except Exception:
            pass
        trace("client disconnected")
        try:
            conn.close()
        except Exception:
            pass


def ping(link, timeout=2.0):
    """主动验证：发一条无副作用指令，等 Pro Micro 回 DONE。

    不用 READY 判活：32U4 打开串口会复位，READY 只在 setup 打一次，
    常常在复位/重枚举期间就漏掉了。主动发指令看回执才可靠。
    """
    try:
        link.write(b"RELEASEALL\n")
    except Exception:
        return False
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            buf += link.read(64)
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

    log_dir = enable_forensics()
    try:
        link = SerialLink(args.serial)
    except Exception as e:
        print(f"[relay] 串口 {args.serial} 打不开：{e}")
        print("        检查：板子插好了吗 / 串口号对不对（部署台的「环境自检」"
              "能列出当前设备）")
        return 1
    print(f"[relay] serial {args.serial} open")
    print(f"[relay] 崩溃取证写到 {log_dir}（relay_crash.log / relay_trace.log）")
    time.sleep(0.5)

    if ping(link):
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
            # 握手阶段对端就走了：多数情况是 B 侧「重连」的中间态（它撤掉了刚建的
            # 连接，或两个重连撞在一起）—— 偶发一条不用管，relay 会继续等下一个。
            # **持续刷屏**才是问题（B 侧在反复重连、一次都没成），那时要查：
            # 证书/端口、以及 B 侧那行状态栏写的原因。
            print(f"[relay] 客户端在 TLS 握手阶段断开（{e}）"
                  f" —— 偶发可忽略；一直刷就是 B 侧连不上")
            conn.close()
            continue
        print(f"[relay] client {addr[0]} connected")
        trace("client %s connected" % addr[0])
        bridge(tls, link)
        print("[relay] client disconnected")


if __name__ == "__main__":
    main()
