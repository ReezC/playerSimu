# -*- coding: utf-8 -*-
"""小地图收流客户端（socket 协议）—— 从 `perception/minimap.py` 拆出（2026-10-18 ✓）。

一眼看懂：**只负责"把 A 机的小地图帧收过来"** ✓
  · `MiniMapClient`：连 relay 的 5003 端口、只留最新一帧、帧尺寸/区域按 `HEAD` 头解析；
  · `_recv_exact`：长度前缀协议的底座；
  · `stop_all_clients()`：进程退出 / 自检收尾统一停（**幂等** ✓）。

⚠ 不在这里做的事：找点、标定、换算、画图 —— 那些在 `perception/minimap.py` ✓
  （依赖方向：`minimap` → `map_client` ✓ 单向 ✓）。
"""
import socket
import struct
import threading
import time
import weakref
from collections import deque
from pathlib import Path

import cv2
import numpy as np





#: ⭐⭐ **还活着的收流客户端**（弱引用 ✓ 不拦 GC ✓）：只为**进程退出前把它们停掉** ✗。
#: 为什么非要有（2026-10-09 ✓ 实测定位）：
#:   `start()` 起的是 **daemon 线程**（`minimap-recv` ✓），它常年**阻塞在 `recv` 上** ✗；
#:   解释器退出时不打招呼就拆线程脚下的 socket / numpy 缓冲 ⇒ **access violation
#:   （`0xC0000005`）** ✗ —— 表现是"**跑完所有用例之后不清不楚地崩掉、连失败清单都不打**" ✗
#:   （`tools/selftest_decision` 就是这样 ✓ 因为里面有几条用例**真建** `RoutePanel` /
#:    `PlayerPanel` ✓ 它们会惰性起收流线程 ✓）。
#: ⚠ 生产里是同一件事（关工作台时偶发 ✓）⇒ 出口统一收一遍：`stop_all_clients()` ✓
#:   （工作台关窗那条路见 `gui/player_panel.shutdown` ✓）。
_CLIENTS = weakref.WeakSet()




def stop_all_clients():
    """把还活着的收流客户端**全部停掉**（进程退出 / 自检收尾各调一次 ✓）。

    幂等 ✓；单条停失败不影响别的 ✓（收尾动作不该把退出流程弄崩 ✗）。
    """
    for c in list(_CLIENTS):
        try:
            c.stop()
        except Exception:                       # noqa: BLE001
            pass


try:
    import atexit as _atexit
    _atexit.register(stop_all_clients)
except Exception:                               # noqa: BLE001
    pass

# ⭐ 2026-10-21：工作台/部署台关窗后是 **`os._exit` 硬退**（`core/appexit.py`，为治
#   「窗口关掉但是实例清不掉」✓）—— 那条路**不跑 atexit** ✗ ⇒ 在这儿**自己登记**一份 ✓
#   （方向是 `perception → core` ✓ 合层 ✓；反过来让 `core` 去 import 本模块会被
#     分层扫描当场按"函数内向上依赖"打红 ✗ 试过了 ✓）。
try:
    from core.appexit import at_exit_hook
    at_exit_hook(stop_all_clients)
except Exception:                               # noqa: BLE001
    pass




class MiniMapClient:
    """收 A 机推来的小地图帧；只保留最新一帧（旧的直接丢）。"""

    def __init__(self, host, port=5003, timeout=5.0):
        self.host, self.port = host, int(port)
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._frame = None          # 最新的 BGR 帧（numpy）
        self._t = 0.0               # 收到它的时刻（perf_counter）
        # 最近若干帧的到达时刻：算"拍/秒"要用它。**不能用 1/(现在-最后一帧时刻)**
        # ——那算的是"距上一帧过了多久"，界面每 100ms 取一次值，会看到 52 这种
        # 假数字（默认推 **60** fps，见 tools.minimap_push.DEFAULTS —— 2026-10-03 从 30 提上来 ✓）。
        self._times = deque(maxlen=40)
        self._stop = threading.Event()
        self._th = None
        #: 当前那条 TCP（`stop()` 要能**从外面掐断它** ✓ 否则 `recv` 卡到超时才退出 ✗）
        self._sock = None
        self.n_recv = 0
        self.n_drop = 0             # 没被取走就被下一帧覆盖的次数
        self.err = ""
        self.connected = False

    # ---------------- 线程 ----------------

    def start(self):
        if self._th is None:
            self._th = threading.Thread(target=self._run, daemon=True,
                                        name="minimap-recv")
            self._th.start()
            _CLIENTS.add(self)          # 登记 ⇒ 进程退出前 `stop_all_clients()` 收得到 ✓
        return self

    def stop(self):
        """停掉收流线程 —— **要真的等它退出**（用户 2026-10-17 ✓ 排查到的崩溃 ✓）。

        ⚠⚠ 老写法只 `_stop.set()` + `join(2)` ✗：线程正卡在 `recv` 里（超时 5s ✓）⇒
          join 等不到 ⇒ 解释器开始拆 numpy/socket 缓冲 ⇒ **`0xC0000005`（access violation）**
          ✓ ⇒ 现场就是"跑完全部用例、进程却非 0 退出"✓（很容易被当成"套件挂了" ✗）。
        ⇒ 现在先把**那条 TCP 掐断**（`shutdown` ✓ 让阻塞的 `recv` 立刻返回 ✓）再 join ✓。
        """
        self._stop.set()
        s = self._sock
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except Exception:                   # noqa: BLE001 —— 已经断了 / 还没连上 ✓
                pass
            try:
                s.close()
            except Exception:                   # noqa: BLE001
                pass
        if self._th is not None:
            self._th.join(timeout=2)
        _CLIENTS.discard(self)

    def _run(self):
        while not self._stop.is_set():
            got0 = self.n_recv      # 这次连接里收过帧没有（首帧超时 vs 推到一半停）
            try:
                with socket.create_connection((self.host, self.port),
                                              timeout=self.timeout) as s:
                    s.settimeout(self.timeout)
                    # ⭐ 让 `stop()` 能从**外面**掐断这条连接（否则它要等 recv 超时 ✗
                    #   见 `stop()` 的说明：解释器退出时那一下就是 access violation ✓）
                    self._sock = s
                    self.connected = True
                    self.err = ""
                    while not self._stop.is_set():
                        hdr = _recv_exact(s, 4)
                        if not hdr:
                            raise ConnectionError("对端关闭")
                        n = struct.unpack(">I", hdr)[0]
                        if n <= 0 or n > 20 * 1024 * 1024:
                            raise ValueError("帧长异常: %d" % n)
                        buf = _recv_exact(s, n)
                        if not buf:
                            raise ConnectionError("读帧中断")
                        jpg = np.frombuffer(buf, np.uint8)
                        img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        with self._lock:
                            if self._frame is not None:
                                self.n_drop += 1
                            self._frame = img
                            self._t = time.perf_counter()
                            self._times.append(self._t)
                        self.n_recv += 1
            except (socket.timeout, TimeoutError):
                # **超时**（不是"对端关闭"）：分两种情况说清该去查什么
                if self._stop.is_set():
                    break
                self.connected = False
                self.err = self._timeout_note(self.n_recv - got0)
                time.sleep(1.0)
            except ConnectionRefusedError:
                if self._stop.is_set():
                    break
                self.connected = False
                self.err = ("A 机 %s:%d **拒绝连接** —— 那边没在跑「小地图推流」，"
                            "或者端口和 config/link.yaml 的 minimap.port 不一致"
                            % (self.host, self.port))
                time.sleep(1.0)
            except Exception as e:
                if self._stop.is_set():
                    break
                self.connected = False
                self.err = "%s: %s" % (type(e).__name__, e)
                time.sleep(1.0)

    def _timeout_note(self, got):
        """超时那句话：分「一帧都没来」和「推到一半停了」两种（2026-09-26）。

        为什么必须分：前者要人去 A 机上查"推流起来没有 / 端口 / 防火墙"；后者说明
        链路本来是通的，是**推到一半卡住**（典型：小地图被别的窗口盖住、抓屏卡住）。
        老实现把两者都报成「对端关闭」，等于把人往错方向带。
        """
        if got > 0:
            return ("推到一半停了：这次连接已经收过 %d 帧，然后 %.0f 秒没有新帧 —— "
                    "去看 A 机那条推流日志（小地图是不是被别的窗口盖住 / 抓屏卡住）"
                    % (got, self.timeout))
        return ("等帧超时（%.0f 秒一帧都没来）：确认 A 机「小地图推流」在跑、"
                "监听端口 %d 与 config/link.yaml 的 minimap.port 一致、"
                "A 机防火墙放行入站 TCP %d"
                % (self.timeout, self.port, self.port))

    # ---------------- 取帧 ----------------

    def latest(self, clear=False):
        """最新一帧 → (frame, t)；还没有返回 (None, 0)。"""
        with self._lock:
            f, t = self._frame, self._t
            if clear:
                self._frame = None
        return f, t

    @property
    def fps(self):
        """最近的**真实**收帧速率（拍/秒）。

        用「最近 N 帧的时间跨度」算，而不是「距最后一帧过了多久」——
        后者界面每 100ms 取一次值时，会在几毫秒内给出 52 这种假数字。
        断流超过 1 秒直接算 0（别把上一轮的速率一直挂着）。
        """
        with self._lock:
            ts = list(self._times)
        if len(ts) < 2:
            return 0.0
        span = ts[-1] - ts[0]
        if span <= 0 or (time.perf_counter() - ts[-1]) > 1.0:
            return 0.0
        return (len(ts) - 1) / span

    # ---------------- 换图：叫 A 机把推流换成这一张图 ----------------

    def switch_map(self, map_id, timeout=3.0):
        """叫 A 机把推流区域换成这张图 ⇒ `(ok, info)`（**不用重启它** ✓）。

        ⭐ 这是「推流开着不停、内容自己变更」的那一句：另开一条到**同一个端口**的
           连接，先报 `HELLO mmap-ctl`（协议常量都在 `tools/mmap_regions`，A/B 共用
           那一份 ✓ —— 各写一份迟早讲成两种方言 ✗），再发 `MAP <id>`。
           A 机查到就 **下一帧**生效，并把**实际在用的区域和 zoom 回过来**。

        `info`：成功是个 dict（`map_id` / `box` / `zoom`）；失败是一句人话（str）。

        ⚠ **必须另开一条连接，不碰收帧那条**：收帧线程正阻塞在 `recv(4)` 上等着读
           帧头，拿它的 socket 说话等于往帧流里灌字，两边都不讨好 ✗。

        ⚠ `zoom` 要搭回来是有讲究的：标定是**对着某个 zoom 标出来的**
           （`mapdata.zoom_of`），换图后 zoom 可能不一样 ⇒ 上层拿这个 zoom 去问
           `calib_for(..., zoom=...)`，才能让"几何对不上"当场暴露，而不是闷头
           给出整倍数错的坐标 ✗。
        """
        mid = str(map_id or "").strip()
        if not mid:
            return False, "没给地图 id —— 不知道该叫 A 机推哪一张"
        if not self.host:
            return False, "不知道 A 机的地址（link.yaml 里没读到 a_host）"
        end = time.time() + max(0.5, float(timeout))
        try:
            from tools import mmap_regions                 # 只在用到处局部 import ✓
        except Exception as e:                             # noqa: BLE001
            return False, "加载换图协议失败：%s: %s" % (type(e).__name__, e)
        try:
            with socket.create_connection((self.host, self.port),
                                          timeout=max(0.5, float(timeout))) as s:
                s.settimeout(max(0.2, end - time.time()))
                # ⭐ **握手与命令合成一次 `sendall`**（2026-10-04 ✓）：分开写时这两个小包
                #   几乎总是**被 A 那边一口气收成一段**（我们连上就发，A 还没轮到 `accept()`
                #   ⇒ 都在它的内核接收缓冲里 ✓）—— A 侧现在能就地处理同一段里的命令
                #   （`tools/minimap_push._ctl_feed` ✓），合成一条只是让"**一段 = 一条完整
                #   命令**"更确定、也少一次系统调用 ✓。
                #   ⚠ **不会让旧版 A 更糟**：旧 A 不认控制连接（不问卷手还是分段都当收帧
                #     客户端 ✓）⇒ 两种写法它一样不理我们 ✓。
                s.sendall(mmap_regions.CTL_HELLO
                          + ("%s %s\n" % (mmap_regions.CMD_MAP, mid)).encode("utf-8"))
                buf = b""
                while b"\n" not in buf and time.time() < end:
                    try:
                        chunk = s.recv(4096)
                    except socket.timeout:
                        break
                    if not chunk:
                        break
                    buf += chunk
        except Exception as e:                             # noqa: BLE001
            return False, ("叫 A 机换图失败：%s: %s\n"
                           "（A 机的「小地图推流」在跑吗？它要是还没支持 MAP 命令，"
                           "连上会受理但一帧也读不回来 —— 那种场合重启一次推流最省事）"
                           % (type(e).__name__, e))
        head = (buf.split(b"\n", 1)[0] or b"").decode("utf-8", "replace")
        rep = mmap_regions.parse_reply(head)
        if rep is None:
            return False, ("A 机的回执看不懂：%r\n"
                           "（多半是那边的推流还是旧版 —— 旧版不支持 MAP 命令，"
                           "会把这条连接当成收流客户端，于是永远不回话）" % (head[:80],))
        if not rep.get("ok"):
            return False, "A 机拒绝了：%s %s" % (rep.get("code") or "",
                                                 rep.get("why") or "")
        return True, rep




def _recv_exact(sock, n):
    """读满 n 字节。**对端关闭**返回 None；**超时照原样往上抛**（别吞）。

    2026-09-26 改：以前这里是 `except Exception: return None` —— 于是超时被当成
    "对端关闭"，界面/命令行永远显示「ConnectionError: 对端关闭」：A 机明明活得好好的
    （只是没在推 / 卡住了），人却跑去查网络和进程。**超时和关闭是两回事**，
    必须分开说，上层才可能报出"卡在哪一步"（见 `MiniMapClient._timeout_note`）。
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))      # 超时 ⇒ socket.timeout 往上抛
        if not chunk:
            return None                      # 真关了（FIN）
        buf += chunk
    return bytes(buf)
