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
from pathlib import Path

import serial

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


def trace(text):
    """记一条指令级日志（超限自动瘦身）。"""
    global _trace_fh
    if _trace_fh is None:
        return
    try:
        _trace_fh.write("%s %s\n" % (time.strftime("%H:%M:%S"), text))
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
                return self._ser.read(n)
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

    def ser_to_tcp():
        while not stop.is_set():
            try:
                data = link.read(256)
            except Exception as e:
                trace("serial->tcp 读失败，结束本连接: %s" % e)
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
            link.write(data)
    except Exception as e:
        trace("tcp->serial 写失败，结束本连接: %s" % e)
    finally:
        stop.set()
        # 连接断开时清空所有按住的键，防止「释放包丢失」导致卡键
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
            print(f"[relay] TLS handshake failed: {e}")
            conn.close()
            continue
        print(f"[relay] client {addr[0]} connected")
        trace("client %s connected" % addr[0])
        bridge(tls, link)
        print("[relay] client disconnected")


if __name__ == "__main__":
    main()
