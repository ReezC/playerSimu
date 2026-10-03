"""小地图定位（寻路用）：收 A 机单独推来的小地图流 → 换算成世界坐标。

数据链路（见 `docs/寻路设计.md`）：

    A 机 tools/minimap_push.py   小地图矩形截图 → JPEG → TCP 长度帧
    ↓
    MiniMapClient（本模块）      收到 → 只留**最新一帧**（和主画面同一个原则：
                                 宁可丢帧、绝不排队）
    ↓
    locate_canvas()              在流画面里找 `datasets/map/<id>.png`（底图），
                                 定出「面板怎么显示底图」：缩放 s + 偏移 (ox, oy)
    ↓
    panel_to_canvas()            流像素 → 底图像素 → （mapdata）世界坐标

**为什么单独一路**：主画面是 H.264 压过的，小地图上的黄点只有 2~5 像素，
压完就糊了；这一路是原始像素（JPEG q=100），黄点位置几乎无损。

**把 s/偏移量出来**（标定）在工作台里做：「路线识别 → 小地图定位 → 标定…」
（gui/minimap_calib.py）—— 那边看得见面板和底图重合不重合，这是唯一的判据。
本模块的命令行是同一条路的等价做法（留给排查/脚本用）：
    python -m perception.minimap --map 105090600

**显示方式（全局 / 局部）由用户人工选**（工作台「路线识别 → 小地图定位」，
每张图一份，存在 `datasets/map/<id>.mapcalib.json` 的 `mode`）——
**不按匹配分自动挑**：那两种画法在不同图上确实不一样（要进游戏走两步看地形
动不动才知道），分数高低说明不了该用哪种。这里的工具一律**沿用你选的那个**
（`--mode saved`，默认），分数只用来判断"这一次定位可不可信"。
"""

import socket
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core import mapdata, zones
from core.imgio import imread, imwrite   # 支持中文路径（cv2.imread 遇中文静默失败）

ROOT = Path(__file__).resolve().parent.parent


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
        return self

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            got0 = self.n_recv      # 这次连接里收过帧没有（首帧超时 vs 推到一半停）
            try:
                with socket.create_connection((self.host, self.port),
                                              timeout=self.timeout) as s:
                    s.settimeout(self.timeout)
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


def is_blackout(frame, ratio=0.98, max_val=24):
    """这一帧是不是**几乎全黑**（「进传送门 → 黑屏」那一瞬）→ bool。

    ⭐ **切图的判据**（用户 2026-10-01 定）：「进传送门 → 黑屏 → 亮起」算进了新图，
      由 **B 机**认这个瞬间，再把新图的 id 发给 A 机（`MiniMapClient.switch_map`）
      ⇒ 推流内容自动换到新图那块区域。

      ⚠ 这里**只判"黑不黑"**。什么时候算"亮起"、新图是哪张，得由**编排层**决定：
      「黑 → 亮」是个**过程**，两次采样之间到底算不算换完了，只有知道目标图的
      那一层说得清 —— 在这一层猜，就会把"半黑"判成"已经进图"✗。

    `ratio`：多大比例的像素算黑，就算黑屏（默认 98% —— 面板边框残留几个亮点不算
      "已经亮起"，别把"还剩 1% 有内容"当成已经进图 ✗）。
    `max_val`：多暗算黑（0~255，默认 24 —— 给 JPEG 噪点和暗部 UI 留点余量）。

    ⚠ **空帧 / 取不到帧 ⇒ False**：没有证据就说"进黑屏了"是猜 —— 那会把
      "还没收到帧"（断流的症状 ✗）误判成"正在换图"，正好把人往错方向带。
    """
    try:
        arr = np.asarray(frame)
    except Exception:                                      # noqa: BLE001
        return False
    if getattr(arr, "ndim", 0) < 2 or getattr(arr, "size", 0) == 0:
        return False
    if arr.ndim == 3:
        arr = arr.mean(axis=2)          # 三通道取平均 —— 比只看某一个通道稳 ✓
    try:
        return bool((arr <= float(max_val)).mean() >= float(ratio))
    except Exception:                                      # noqa: BLE001
        return False


def offset_of(calib):
    """从**标定字典**里读「坐标系偏移」→ `(x, y)`（没有 / 坏了给 `(0, 0)`）。

    ⚠ **口径**（2026-09-26 用户澄清，**别搞反**）：`定位结果 + 这个偏移` 才是玩家的
    **真实世界坐标** —— 那正是**游戏自己的世界坐标系**。地形数据（`core/mapdata` 的
    foothold / ladder 的 x、y）**本来就在这个系里**（例：105090600 的 fh44 地形 y=-208，
    它的世界 y 就是 -208）。⇒ 判据里拿玩家读数与地形坐标比大小是**直接比**，
    **不要**给地形坐标再加偏移（我犯过这个错：把"到达目标平台面"从 -208 放宽成 -175 ✗）。
    这个偏移修的是"黄点上的那个锚点 ↔ 玩家原点"那层对应，不是坐标系换算。

    ⚠ **2026-09-27 起，锚点是黄点的「下沿（脚底）」**（`dot_feet`，见那儿的口径说明）：
    双点标定采样的也是下沿 ⇒ 这一项从"重心 ↔ 原点"（≈ 半个点高，现场是 (7,33)）
    变成"下沿 ↔ 原点"的**残差**，正常应该接近 **(0, 0)** ✓。
    ⇒ 老的 (7,33) 那份是**重心口径**的，别搬进新标定（那是双重补偿，会凭白偏 33 世界像素 ✗）。
    """
    v = (calib or {}).get("world_offset") or []
    if len(v) == 2:
        try:
            return (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            pass
    return (0.0, 0.0)


def world_offset(map_id, src=None):
    """某张图 + 某个来源的「坐标系偏移」（见 `offset_of` 的口径说明）。

    ⚠ `src` 的默认值**不能在签名里写成 `SRC_STREAM`**：这几个常量定义在本文件靠后的
    位置，默认值在 `def` 那一刻就要取值 ⇒ 会直接 `NameError`（写这条时就是这么炸的 ✗）。
    所以这里 `src=None`，调用时再兜底。
    """
    if not map_id:
        return (0.0, 0.0)
    try:
        return offset_of(mapdata.load_calib(map_id, src or SRC_STREAM))
    except Exception:                                # noqa: BLE001
        return (0.0, 0.0)


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


# ══════════════════════════════════════════════════════════════
# 面板 → 底图 → 世界
# ══════════════════════════════════════════════════════════════

#: 面板显示底图的两种方式（**每张图可能不同**，所以要标定并记下来）：
#:   fit  —— 整张底图缩放到面板里（窗口里永远看得见全图，地形不随人移动）
#:   crop —— 面板 1:1 显示底图的一块、随玩家滚动（底图比面板大时只能这样）
MODE_FIT = "fit"
MODE_CROP = "crop"

#: 面板画面**从哪来**（工作台「路线识别 → 小地图来源」；存在 config/live.yaml）：
#:   stream —— A 机的「小地图推流」：独立一路、原始像素（JPEG q100），画面最清晰；
#:   live   —— 从**实时预览那一帧**里裁一块：不用多推一路，但那是 H.264 压过的画面，
#:             黄点只剩几个像素（**实验**：面板↔底图匹配问题不大，黄点识别待实测）。
#: 为什么放在 live.yaml 而不是每张图的标定里：它取决于本机的推流/画面几何，与地图无关。
SRC_STREAM = "stream"
SRC_LIVE = "live"

#: **匹配分到多少才算"这把尺子可信"**（0~1）。两个地方共用这一份，别再各写一个数 ✗：
#:   · `region_match_score`（框完当场验证：这个框对不对）；
#:   · `check_calib`（核对当前标定：**尺子自己可信吗**）。
#: 0.8 是原来就写着的值（`locate_fit`/`locate_crop` 那条 `min_score` 0.55 是"低于它
#: 连试都不用试"，比它高一档才是"可信"✓）；实测参考：面板里混着游戏 UI、或底图压过时，
#: 分数会掉到 0.62~0.70 一带 ⇒ 那种分数下的定位结果只能当参考 ✗。
TRUST_SCORE = 0.8

#: **匹配分低到多少就"量不出来"**（0~1）—— 「标定时能不能量出来」与「运行时跟不跟得住」
#: 共用这一份，别再各写一个 0.55 ✗（两处各写一个数迟早分叉，而这两件事说的是同一件事：
#: "这块面板和底图对上了没有" ✓）。它比 `TRUST_SCORE` 低一档：0.55 是"低于它连试都
#: 不用试"，0.8 才是"这份几何可信" ✓。
LOCATE_MIN_SCORE = 0.55

#: `locate_crop` 的模板（缩回底图尺度之后）**最多取底图的多少**（0~1，按边长）。
#: **为什么要有这个上限**：面板比底图宽时（crop 的常态 —— 实测 105040303 = 502×408 的
#: 面板 + **82×218** 的底图，真倍数 5.645 缩回来 **87.5 > 82** ✗），整块模板**放不进底图**；
#: 能放进去的位置只剩 0 个 ⇒ 位置根本量不出来 ✗。取中间一块、留够可放的位置才行 ✓。
CROP_TPL_FRAC = 0.6

#: 同上，**备一档更小的**：面板贴到底图外沿时（面板的可见部分很窄），0.6 那块
#: 仍然"放不进去"（实测：116×79 的底图 + 100 高的面板，可见的只有 43 行 ⇒ 0.6×79 = 47
#: 的块放不下 ✗）⇒ 小一档才有块落得进去 ✓（大小只影响"搜得到搜不到"，不影响谁胜出 ✓）。
CROP_TPL_FRAC_SMALL = 0.35

#: 模板**最小的边长**（底图像素）—— 再小就"哪儿都像"：`TM_CCOEFF_NORMED` 在
#: 42×34 那种小模板上碰巧就能到 0.48 ✗（实测：真倍数 5.645 的面板，自动定位给出
#: **scale=12.000** —— 就是"小模板碰巧高分"把候选带跑的 ✓）。小于这个数不参选。
CROP_TPL_MIN = 12

#: 面板缩回底图之后**最多允许比底图大多少倍**才算候选（超过 ⇒ 面板覆盖的范围远超
#: 底图，几何上讲不通：不是这个倍数）。比 1 大是因为面板里常含游戏 UI 边框，
#: 它映到底图上是"负数那一圈"（实测用户那份：x ∈ [-12.2, 76.7] ✓）。
CROP_FOOT_MAX = 1.6

#: 来源的中文名 —— 界面、提示、报告共用这一份，别一处写「独立推流」另一处写
#: 「收流」：标定是按来源分开存的，名字对不上就没人搞得清哪份是哪份。
SRC_LABEL = {SRC_STREAM: "独立推流", SRC_LIVE: "从实时画面"}

#: 标定 dict 的形状（存 datasets/map/<id>.mapcalib.json）
#:   mode/scale/offset  —— 面板像素 → 底图像素：canvas = (panel - offset) / scale
#:   view               —— crop 模式专用：当前显示的是底图从 (view) 开始的那一块
#:   score              —— 标定时匹配得有多好（>0.8 才可信）
#:   scale_y            —— **可选**：y 轴自己的缩放（2026-09-27「双点标定」加的）。
#:                         ⚠ 为什么是"另开一个键"而不是把 `scale` 改成 `[sx, sy]`：
#:                         改类型会让**所有老读法**（`float(cal["scale"])`、模板匹配
#:                         存下来的那些、命令行工具）在读到新文件时当场炸 ✗；而
#:                         "`scale` = x 轴、`scale_y` 缺省 = 跟 x 一样"这套写法，
#:                         两个方向都兼容 —— 老代码读新文件只是把 y 也当成 x（两轴
#:                         本来就该几乎相等），新代码读老文件**精确**（缺省即同轴）✓。
#:                         取两轴一律走 `scales_of()`，别各处自己读（少一处就是
#:                         "换算用 x、画图用 y"这种最难查的分歧 ✗）。


def _crop_scales(canvas_wh, panel_wh):
    """crop 的候选放大倍数：整数为主（像素画放大就是复制像素），再补半档。"""
    cw, ch = canvas_wh
    pw, ph = panel_wh
    cand = set(float(v) for v in range(1, 13))
    cand |= {x * 0.5 for x in range(2, 25)}
    if cw > 0 and ch > 0:
        # 「面板刚好装下底图」的那个比例：客户端常在它附近取值
        cand.add(round(min(pw / cw, ph / ch), 3))
    return sorted(c for c in cand if 0.5 <= c <= 12.0)


#: 标定文件里没写 `alpha` 时的叠图浓淡（%）。**默认值只有这一处** —— 设置窗口、
#: 标定弹窗、路线识别面板都走 `overlay_alpha_pct()`，谁也别再写 `or 55` ✗。
DEFAULT_ALPHA_PCT = 55


def overlay_alpha_pct(calib):
    """标定里的叠图浓淡（0~100）。**0 就是 0** ✓。

    ⚠ 2026-09-26 修：原来是 `int(cal.get("alpha") or 55)` ⇒ **0 是 falsy** ⇒ 滑块拉到
    最左端（0 = 完全看不见）会被 `or` 换成 55 ✗（用户报"拖动条在最左端时透明度是
    55% 而不是 0%"）。判"有没有这个键"只能用 `is None`，不能用 `or` —— 这个值**特意
    要 0**，而 `or` 恰恰把 0 当"没填"。
    """
    v = (calib or {}).get("alpha")
    if v is None:
        return DEFAULT_ALPHA_PCT
    try:
        return max(0, min(100, int(round(float(v)))))
    except (TypeError, ValueError):
        return DEFAULT_ALPHA_PCT


def overlay_alpha(calib):
    """同上，但给 Qt / 绘制用的 0.0~1.0 透明度（`setOpacity` 那种）。"""
    return overlay_alpha_pct(calib) / 100.0


def live_src(cfg=None):
    """`config/live.yaml` 里的 `mmap_src`（非法或缺省 ⇒ 收流）。

    **一份口径**：路线识别面板和设置窗口（浓淡那一项按来源存 ✓）都从这儿取，
    不然两边各判一次"哪个来源合法"迟早分叉。
    """
    v = (cfg or {}).get("mmap_src")
    return v if v in (SRC_STREAM, SRC_LIVE) else SRC_STREAM


def crop_of(project, cfg=None):
    """小地图面板在**实时画面**里的位置 `[x, y, w, h]`；取不到给 None。

    **按项目（地图）存** —— `projects/<名>/project.yaml` 的 `mmap_crop`。用户 2026-09-27
    现场：不同地图的小地图面板**尺寸/位置完全不同**，而这个值原来只有全局一份
    （`config/live.yaml`）⇒ 换个项目还是上一张图的框 ⇒ 叠图/定位全错 ✗。所以它和
    「寻路编辑器 / 集合 / 地形图」是同一类东西（跟项目走 ✓），入口也在那一页
    （「路线识别 → 寻路配置 → 框选小地图」，就在「寻路编辑器」下面 ✓）。

    ⚠ **兜底**：本项目**还没框过**时读老的那份全局值（`cfg` 里的 `mmap_crop`，那是
      2026-09-27 之前唯一的一份）—— 升级后不用重框就能接着用；**按本项目框过一次之后
      它就不再起作用** ✓。这是"读"的便利，**不是第二处存储**：谁都不许再往
      `live.yaml` 写它 ✗（那正是"换项目就错"的来源）。
    ⚠ `project` 传 `gui.project.Project`（或任何有 `.get()` 的映射）；没打开项目就传 None
      ⇒ 只剩兜底那条路 ✓。
    ⚠ **一条口径只有这一处**：路线识别面板（`_mmap_crop`）、命令行取证工具都用它，
      别再各写一份"项目优先还是全局优先" ✗。
    """
    for src in (project, cfg):
        if src is None:
            continue
        r = src.get("mmap_crop")
        if isinstance(r, (list, tuple)) and len(r) == 4:
            return [int(v) for v in r]
    return None


def _subpixel_peak(res, ml):
    """`matchTemplate` 结果图上的峰值 → **亚像素**位置 `(fx, fy)`。

    **为什么不能只用 `minMaxLoc`**（2026-09-27）：它给的是**整数**像素，而这张图上
    1 个面板像素 = 8.5 世界像素 ⇒ 取整等于白送最多半个像素的系统偏差（4 个世界像素
    —— 现场"看着像点没点准"那个量级 ✗）。做法是标准的三点抛物线插值（x / y 各一次），
    只在峰值邻域 ±1 里做；邻域越界（贴边）就退回整数 ✓。
    插出来的偏移**夹在 ±0.5 内**：噪声大时抛物线会外推到离谱的位置 ✗。
    """
    mx, my = int(ml[0]), int(ml[1])
    h, w = res.shape[:2]
    fx, fy = float(mx), float(my)

    def _one(a, b, c):
        """三点 (a, b, c) 中间那个是最大 ⇒ 峰相对中间偏多少（±0.5 内）。"""
        den = a - 2.0 * b + c
        if abs(den) < 1e-12:
            return 0.0
        return max(-0.5, min(0.5, 0.5 * (a - c) / den))

    if 0 < mx < w - 1:
        fx = mx + _one(float(res[my, mx - 1]), float(res[my, mx]),
                       float(res[my, mx + 1]))
    if 0 < my < h - 1:
        fy = my + _one(float(res[my - 1, mx]), float(res[my, mx]),
                       float(res[my + 1, mx]))
    return fx, fy


def _fit_at(fg, cg, s):
    """按缩放 `s` 把底图缩到面板大小匹配一次 → `(分数, 峰值, 结果图)`；放不下 ⇒ None。"""
    if not s or s <= 0:
        return None
    tw, th = int(round(cg.shape[1] * s)), int(round(cg.shape[0] * s))
    if tw < 8 or th < 8 or tw > fg.shape[1] or th > fg.shape[0]:
        return None
    t = cv2.resize(cg, (tw, th), interpolation=cv2.INTER_AREA)
    r = np.nan_to_num(cv2.matchTemplate(fg, t, cv2.TM_CCOEFF_NORMED))
    _, mx, _, ml = cv2.minMaxLoc(r)
    return float(mx), ml, r


def refine_fit(fg, cg, best, rounds=6, span0=0.25, steps=7):
    """把 `locate_fit` 的**粗**结果细化：连续尺度 + 亚像素偏移（2026-09-27 新增）。

    **为什么非有不可**（"把全局小地图弄准"的第一步就是先把尺子弄准）：
      · 粗搜的尺度是**离散**的（`1.00~6.00` 每 0.25 一档，往上每 0.5 一档）；
      · 峰值是**整数**像素（`minMaxLoc`）。
      合成面板实测：真值 `scale=1.874` 时粗搜给 1.933（差 3.1%）⇒ **最远角差 11 个面板
      像素 ≈ 93 世界像素**。拿这种尺子去说"标定偏了多少"，尺子自己就是噪声 ✗。
    做法：先把偏移换成同一档的**亚像素**结果；再逐轮以**上一轮最优为中心**均匀取
    `steps` 档（`span` 每轮减半），取分最高的那档 —— **分数是单调判据，改进不了就停**
    （不然细化可能越弄越差 ✗）。**代价**：约 `rounds × steps` 次匹配（默认 6×7 ≈ 40 次，
    几十毫秒，一次性按钮操作 ✓）。
    ⚠ **不许离开粗搜那一档的窗口**（`±span0/2` = 粗搜格子的一半）：分数面很平时
      （面板里混着游戏 UI / 底图压过 / 分辨率对不上）"只挑更高的分"会**一路漂走** ——
      实测踩过：从 1.25 漂到 0.84，两套几何最远角差 6600 世界像素，全是噪声 ✗。
      窗口 = "真值确实在粗搜那一档附近"这个前提的边界 ✓。
    ⚠ **粗搜分数低也要细化**（`locate_fit` 里细化在门槛之前判 ✓）：真值落在两档中间时，
      粗搜每一档都差 6% 以上 ⇒ 分数能低到 0.27，但细化到 0.03 的格子就接近满分 ✓。
    """
    if best is None:
        return best
    steps = max(3, int(steps))
    base_s = float(best["scale"])          # 粗搜那一档（窗口的中心）
    lo, hi = base_s - span0 / 2.0, base_s + span0 / 2.0
    b_score = float(best["score"])
    b_off = [float(v) for v in best["offset"]]
    s0 = base_s

    def _eval(s):
        got = _fit_at(fg, cg, s)
        if got is None:
            return None
        sc, ml, r = got
        return sc, list(_subpixel_peak(r, ml))

    # ① 同一档，偏移换成亚像素（尺度不动也可能立刻改善 ✓）
    cur = _eval(s0)
    if cur is not None and cur[0] >= b_score:
        b_score, b_off = cur[0], cur[1]
    # ② 逐轮细化尺度：以上一轮最优为中心，但不许出窗口 ✓
    for rd in range(int(rounds)):
        span = span0 / (2.0 ** rd)
        center = min(max(s0, lo), hi)
        cand = None
        for i in range(steps):
            s = min(max(center + span * (2.0 * i / (steps - 1) - 1.0), lo), hi)
            got = _eval(s)
            if got is None:
                continue
            if cand is None or got[0] > cand[0]:
                cand = (got[0], s, got[1])
        if cand is None or cand[0] <= b_score + 1e-9:
            break                      # 这一轮没有改进 ⇒ 停（别越弄越差 ✗）
        b_score, s0, b_off = cand
    return {"mode": best.get("mode", MODE_FIT), "scale": float(s0),
            "offset": [float(v) for v in b_off], "score": float(b_score),
            "view": list(best.get("view") or [0, 0]), "refined": True}


def locate_fit(frame, canvas, scales=None, min_score=LOCATE_MIN_SCORE, refine=True):
    """方式 1：整张底图缩放到面板 → 标定 dict 或 None。

    ⚠ 默认**细化**（`refine_fit`，2026-09-27 加）：粗搜是离散尺度 + 整数峰值，不够当
      一把"量精度的尺子"（见 `refine_fit` 的说明）。要老行为（纯粗搜、整数偏移）
      就传 `refine=False` ✓（`locate_crop` 那份暂时仍是粗搜 —— crop 的自动标定
      用户明确暂缓 ✓）。
    """
    fg = _gray(frame)
    cg = _gray(canvas)
    if scales is None:
        fh, fw = fg.shape[:2]
        ch, cw = cg.shape[:2]
        base = min(fw / float(cw), fh / float(ch))
        # 候选尺度要**够密**：客户端把小地图放大几倍是不确定的（2x / 3x / 正好嵌进面板）。
        # 上限放到 12：小底图（几十像素）被客户端放大 6~10 倍是实测见过的，
        # 原来只到 5，那种图上的自动定位会一路失败（人手动拖也不够，见 locate_crop）。
        cand = [round(1.0 + 0.25 * i, 3) for i in range(21)]      # 1.00 ~ 6.00
        cand += [round(6.0 + 0.5 * i, 3) for i in range(13)]      # 6.0 ~ 12.0
        cand += [round(base, 3), round(base * 0.5, 3), round(base * 2.0, 3)]
        scales = sorted({c for c in cand if 0.25 <= c <= 12.0})

    best = None
    for s in scales:
        got = _fit_at(fg, cg, s)
        if got is None:
            continue
        mx, ml = got[0], got[1]
        if best is None or mx > best["score"]:
            best = {"mode": MODE_FIT, "scale": float(s),
                    "offset": [int(ml[0]), int(ml[1])], "score": float(mx),
                    "view": [0, 0]}
    if best is None:
        return None
    if refine:
        # ⚠ **先细化再判门槛**（2026-09-27）：粗搜的格子是 0.25 ⇒ 真值落在两档中间
        #   时**每一档都差 6% 以上**，粗搜的分数可以低到 0.27（合成面板实测：真值
        #   1.874、粗搜 1.75/2.0 都只有 0.2~0.3）⇒ 拿粗分数去卡 0.55 会直接判"量不出来" ✗。
        #   先细化到 0.0625 的格子 ⇒ 真值那一档分数接近 1.0 ✓，再拿它和市场门槛比才有意义。
        best = refine_fit(fg, cg, best)
    if best["score"] < min_score:
        return None
    return best


def _crop_blocks(f, inset, s, cw, ch):
    """按倍数 `s` 从面板上取**几块**当模板 → `[(面板上的左上角, 模板, 真正实现的倍数), ...]`。

    每块都缩回**底图尺度**（`cv2.resize(INTER_AREA)` ✓），且缩回后 ≤
    `CROP_TPL_FRAC`×底图（大块才有约束力、且要留得出可放的位置 ✓）。

    ⚠⚠ **为什么要好几块**（用户 2026-09-29："自动定位按钮 好像对于局部地图来说不好用"✗）：
    **到底哪一块落在底图里，事先不知道** —— 那正是要求的东西 ✗（实测用户那份：
    面板映到底图 x ∈ [−12.2, 76.7]，而底图只有 82 宽 ⇒ 面板右侧一大截在底图之外，
    只有**靠左**的那块才落得进去 ⇒ 只取"正中间那块"会**整块放不进底图**、
    匹配位置贴着边界 ✗ ⇒ 量出来的倍数是错的 8.4（真 5.645）✗✗）。
    ⇒ 面板在某个轴上比底图宽（或差不多）时，那个轴给**左 / 中 / 右**三个位置 ✓，
    轴向上装得下就只给中间一个 ✓（省时间：多数情况只有 1~3 块 ✓）。
    """
    h, w = f.shape[:2]
    ax0, ay0 = inset, inset
    ax1, ay1 = w - inset, h - inset
    if ax1 - ax0 < 8 or ay1 - ay0 < 8:
        ax0, ay0, ax1, ay1 = 0, 0, w, h
    if ax1 - ax0 < 8 or ay1 - ay0 < 8:
        return []
    foot_w = (ax1 - ax0) / float(s)          # 这个倍数下面板映到底图有多大
    foot_h = (ay1 - ay0) / float(s)
    # 那个轴上**装不下整个面板**（crop 的常态）⇒ 只能挪着试（哪块在底图里事先不知道 ✓）；
    # 装得下就只取中间一块（省时间 ✓）
    out = []
    for frac in ((CROP_TPL_FRAC, CROP_TPL_FRAC_SMALL)
                 if (foot_w >= cw * 0.98 or foot_h >= ch * 0.98)
                 else (CROP_TPL_FRAC,)):
        tw = int(min(foot_w, cw * frac))
        th = int(min(foot_h, ch * frac))
        if tw < CROP_TPL_MIN or th < CROP_TPL_MIN:
            continue
        if tw >= cw or th >= ch:        # 必须**严格小于**底图，否则只有一个位置可放
            continue
        wq = max(CROP_TPL_MIN, min(int(round(tw * s)), ax1 - ax0))
        hq = max(CROP_TPL_MIN, min(int(round(th * s)), ay1 - ay0))
        xs = [ax0 + ((ax1 - ax0) - wq) // 2]
        ys = [ay0 + ((ay1 - ay0) - hq) // 2]
        # ⚠ **面板**装不下那个轴 ⇒ 左 / 中 / 右都试：面板贴到底图外沿时，只有靠边那块
        #   落得进去（实测用户那份面板映到底图 x ∈ [−12.2, 76.7] ⇒ 只有靠左那块 ✓）
        if foot_w >= cw * 0.98:
            xs = sorted({ax0, ax0 + ((ax1 - ax0) - wq) // 2, ax1 - wq})
        if foot_h >= ch * 0.98:
            ys = sorted({ay0, ay0 + ((ay1 - ay0) - hq) // 2, ay1 - hq})
        for px0 in xs:
            for py0 in ys:
                if any(v[0] == (px0, py0) for v in out):
                    continue                     # 两档 frac 撞在一起 ⇒ 别重复算 ✓
                sub = f[py0:py0 + hq, px0:px0 + wq]
                if getattr(sub, "size", 0) == 0:
                    continue
                tpl = cv2.resize(sub, (max(1, int(round(wq / s))),
                                       max(1, int(round(hq / s)))),
                                 interpolation=cv2.INTER_AREA)
                if tpl.shape[0] < CROP_TPL_MIN or tpl.shape[1] < CROP_TPL_MIN:
                    continue
                if tpl.shape[0] >= ch or tpl.shape[1] >= cw:
                    continue
                # ⚠ `eff` 一律取**真正实现出来的倍数**（按实际输出尺寸算），不写"想要的那个 s" ——
                #   否则 view 是对的、scale 是错的（`t_crop_scale_not_faked` 钉着 ✓）
                out.append(((px0, py0), tpl, wq / float(tpl.shape[1])))
    return out


def _peak_subpix(r, x, y):
    """相关图峰值的**亚像素**位置（抛物线拟合）→ `(fx, fy)` 浮点。

    面板放大 5~10 倍很常见 ⇒ **1 个底图像素的量化误差 = 5~10 个面板像素** ✗
    （实测 105040303：一个底图像素 ≈ 7.4 个世界像素 ⇒ 那点误差直接进世界坐标 ✓）。
    峰值贴着边界（拟合不了）就返回整数 ✓。
    """
    out = [float(x), float(y)]
    h, w = r.shape[:2]
    if 0 < x < w - 1:
        l, m, rr = float(r[y, x - 1]), float(r[y, x]), float(r[y, x + 1])
        d = l - 2.0 * m + rr
        if abs(d) > 1e-9:
            out[0] = x + max(-0.5, min(0.5, 0.5 * (l - rr) / d))
    if 0 < y < h - 1:
        l, m, rr = float(r[y - 1, x]), float(r[y, x]), float(r[y + 1, x])
        d = l - 2.0 * m + rr
        if abs(d) > 1e-9:
            out[1] = y + max(-0.5, min(0.5, 0.5 * (l - rr) / d))
    return out[0], out[1]


#: ⭐ **「滚动方向」** = 这条小地图面板**会往哪个轴滚**（用户 2026-09-29 追加：
#: "局部小地图也应该分滚动类型：'仅X''仅Y''双轴'，这样在对于单轴滚动的小地图，
#: 实时匹配时可以提高速度"✓）。
#: 它**不改换算**（`panel_to_canvas` 一字不变 ✓），只改两件事：
#:   · **搜索只在会动的那根轴上做** ⇒ 单轴图上快好几倍 ✓（全图找只扫一条**带**
#:     hmm：仅 X ⇒ 高只有模板那么高的一条横带 ✓、仅 Y ⇒ 竖带 ✓；拟合也只试那一轴 ✓）；
#:   · **不会动的那根轴"不许跟歪"** ⇒ 那一轴由标定定死 ✓（面板在那一轴上的小抖动
#:     不会变成显示区漂移 ✓）。
#: ⚠ 老标定里**没有这个键** ⇒ 双轴（= 今天的行为一字不变 ✓）。
SCROLL_XY = "xy"
SCROLL_X = "x"
SCROLL_Y = "y"
#: 界面上那个下拉的文案（`gui/minimap_calib.py` 用它 ✓ 一处口径 ✓）
SCROLL_LABEL = ((SCROLL_XY, "双轴（上下左右都会滚）"),
                (SCROLL_X, "仅 X（地图只左右滚）"),
                (SCROLL_Y, "仅 Y（地图只上下滚）"))


def scroll_of(calib):
    """标定里写的滚动方向 → `"x" | "y" | "xy"`（没写 / 写错 ⇒ 双轴 = 老行为 ✓）。"""
    v = str((calib or {}).get("scroll") or SCROLL_XY).strip().lower()
    if v in ("x", "h", "horizontal", "仅x"):
        return SCROLL_X
    if v in ("y", "v", "vertical", "仅y"):
        return SCROLL_Y
    return SCROLL_XY


def scroll_axes(calib):
    """→ `(能动的轴 X, 能动的轴 Y)`（`(True, True)` = 双轴 ✓）。

    调用方一律用这一对布尔判"这根轴要不要搜 / 要不要动" ✓（别在别处再解析一遍字符串 ✗）。
    """
    s = scroll_of(calib)
    return (s in (SCROLL_XY, SCROLL_X), s in (SCROLL_XY, SCROLL_Y))


def _crop_calib_from(tpl_pos, sub_xy, eff, inset):
    """"模板落在底图哪里" → 标定 dict（`view` = 面板 (inset, inset) 对应的底图坐标 ✓）。

    ⚠ 口径照抄 `panel_to_canvas`：`canvas = (panel − offset)/scale + view`
      ⇒ 面板上 `sub_xy` 那个点映到 `tpl_pos` ⇒ `view = tpl_pos − (sub_xy − offset)/scale` ✓
      （整块模板时 `sub_xy == offset` ⇒ `view == tpl_pos`，和老行为**一字不差** ✓）。
    """
    fx, fy = tpl_pos
    vx = fx - (sub_xy[0] - inset) / eff
    vy = fy - (sub_xy[1] - inset) / eff
    return {"mode": MODE_CROP, "scale": float(eff), "offset": [inset, inset],
            "view": [float(vx), float(vy)]}


def _crop_score(frame, canvas, calib, max_samples=2000):
    """一套 crop 几何的贴合分（**只在重叠够多时才算数**）→ `(score|None, overlap)`。

    ⚠ 搜索时采样取 **2000**（不是显示那个 6000）：拟合要算几百次，2000 点足够分辨
      "哪套几何更好" ✓，反而快 ~2.5 倍 ✓；**给人看的那个分**仍走 `align_score` 的 6000 ✓
      （两者同一个函数、同一个几何族，只是采样密度不同 ⇒ 差别在小数点后第三位 ✓）。
    """
    sc, ov, _why = overlap_pearson(frame, canvas, calib, max_samples=max_samples)
    if sc is None or ov < 0.08:
        # 面板几乎全在底图外面 ⇒ 那几个采样点说明不了什么 ✗（容易碰巧高分）
        return None, ov
    return sc, ov


def _crop_pos_fit(frame, canvas, calib, steps=(8.0, 4.0, 2.0, 1.0, 0.5, 0.25),
                  axes=None):
    """**固定缩放**，只挪位移（每一步试 8 个方向，逐级细化到 0.25 底图像素）→ 最好的那份。

    ⚠ 为什么要"固定缩放"分开做（实测踩到）：位移和缩放**不是可分的** —— 缩放差 2%
      时，把它单独往真值挪是**下坡**（位移能补偿一部分）⇒ 一起坐标下降会**卡在这条山脊上**
      （实测：9.0/7.0 那份真几何贴合分 1.0000，可从 9.54/7.41 出发一起爬只能到
      9.25/7.41 的 0.9298 就动不了了 ✗）。固定缩放时位移那一维是**单峰**的 ⇒ 稳 ✓。

    ⚠⚠ **细档（≤1 像素）要用高采样**（6000），粗档（≥2）才用 2000：采样稀了分数面会
       "阶梯化"（实测：1:1 的合成面板在 2000 采样下，8.00 / 8.25 / 7.75 都是 1.0000、
       8.50 掉到 0.8606 ⇒ 爬坡最多只能保证 ±0.25，再叠加"只许严格变好"就**停在偏
       0.5 像素的地方** ✗ —— 那 0.5 底图像素在这张图上是 ~4 个世界像素 ✓）。
      细档的eval 只有几十次 ⇒ 多花十几毫秒，换回来的是"跟出来的显示区真的准" ✓。
    """
    ms = 6000 if max(steps) <= 1.0 else 2000
    # ⭐ **单轴的图只试那根轴**（"仅 X" ⇒ 不试 dy ✓）：那些 dy ≠ 0 的探针要么被算出来是
    #    下坡（白花），要么会把"根本不会动的那根轴"跟歪 ✗ ⇒ 2/3 的探针直接省掉 ✓
    ax_on, ay_on = (True, True) if axes is None else axes
    best_sc, _ov = _crop_score(frame, canvas, calib, max_samples=ms)
    if best_sc is None:
        return None
    best = {"calib": dict(calib), "score": float(best_sc)}
    for step in steps:
        moved = True
        while moved:
            moved = False
            base = best["calib"]
            v = list(base.get("view") or (0, 0))
            for dx in ((step, 0.0, -step) if ax_on else (0.0,)):
                for dy in ((step, 0.0, -step) if ay_on else (0.0,)):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    probe = dict(base, view=[v[0] + dx, v[1] + dy])
                    sc, _ov = _crop_score(frame, canvas, probe, max_samples=ms)
                    if sc is not None and sc > best["score"] + 1e-4:
                        best = {"calib": probe, "score": float(sc)}
                        moved = True
                        break
                if moved:
                    break
    return best


def _fit_crop(frame, canvas, calib, axes=None):
    """位移 + 缩放一起定：**缩放扫网格、每个缩放都把位移归位**，再收窄重来一遍 ✓。

    为什么这么绕（见 `_crop_pos_fit` 的说明）：两者不可分 ⇒ 只能"固定一个、优化另一个"
    交替 ✓。两轮足够（第一轮 ±8% 步 1%，第二轮在最好的附近 ±1% 步 0.12% ✓）。

    ⚠ 为什么不用模板匹配把这件事做完：它只能给**整像素**峰值，而面板放大 5 倍时
      **1 底图像素 = 5~10 个面板像素** ✗；贴合分是连续的 ⇒ 能定到 0.25 底图像素 ✓
      （而且它**就是判据那行显示的那个数** —— 把它做到最好，就是"自动定位"该干的事 ✓）。
    """
    fit = _crop_pos_fit(frame, canvas, calib, axes=axes)
    if fit is None:
        return None
    best = fit
    # 三段：先 ±20%（粗，位置只粗定）→ ±4% → ±0.8%（细，位置抠到 0.25 ✓）
    # ⚠ 第一段要**宽**：种子可能离真值 10%~20%（模板匹配在小块上给不准 ✓），
    #   而收窄之后就只能在本山头上找 ⇒ 太窄会把对的那份关在外面 ✗
    for span, step, steps in ((0.20, 0.04, (4.0, 2.0)),
                              (0.04, 0.008, (2.0, 1.0)),
                              (0.008, 0.0016, (1.0, 0.5, 0.25))):
        s0 = float(scales_of(best["calib"])[0])
        if s0 <= 0:
            break
        n = max(2, int(round(span / step)))
        for k in range(-n, n + 1):
            if k == 0:
                continue
            cand = with_scales(best["calib"], s0 * (1.0 + step * k),
                               s0 * (1.0 + step * k))
            got = _crop_pos_fit(frame, canvas, cand, steps=steps, axes=axes)
            if got is not None and got["score"] > best["score"] + 1e-4:
                best = got
    return best


def refine_crop(panel, canvas, calib, move_scale=True):
    """在**现有几何**附近抠一抠 ⇒ 更好的那一份（`{"calib", "score", "before", "gain"}`）。

    用途：手工对齐之后想"再抠一点"（用户 2026-09-29："我已经手工对的很好了"✓ —— 那就
    量一量**还能不能更好**：好多少、往哪边 ✓）。**只接受更好的** ⟹ 绝不会把对的改坏 ✓
    （所以它可以放心地做成一个按钮 ✓）。算不出来 ⇒ `None`。

    `move_scale=False` ⇒ **只挪位移、缩放一个字不动** —— 界面上那个「不改缩放」就是它 ✓
    （"我已经量好缩放、只想知道位置"那条路 ✓）。
    """
    cal = dict(calib or {})
    if cal.get("mode") != MODE_CROP or not has_geometry(cal):
        return None
    if float(scales_of(cal)[0]) <= 0:
        return None
    axes = scroll_axes(cal)          # ⭐ 单轴 ⇒ 只跟会动的那根轴（见 `scroll_axes` ✓）
    got = _fit_crop(panel, canvas, cal, axes=axes) if move_scale \
        else _crop_pos_fit(panel, canvas, cal, axes=axes)
    if not move_scale:
        # ⚠ 只挪位移时**光靠"手上那套"当起点不够**：它可能整块是错的（首次标定时就是）
        #   ⇒ 再把"就按这个缩放、用模板匹配粗定一遍"那份当第二号起点 ✓
        #   ⛔ 但**缩放必须锁死在调用方给的那个**：粗定那份的 `eff` 是"真正实现出来的
        #     倍数"（可能 5.652 ≠ 输入的 5.645）⇒ 直接用它就等于偷偷改了缩放 ✗✗
        #     （`t_crop_locate_real_shape` 里钉着这一条 ✓）。
        s_lock = float(scales_of(cal)[0])
        alt = crop_seed_at(panel, canvas, s_lock,
                           int(cal.get("offset", [4, 4])[0]))
        if alt is not None:
            alt2 = _crop_pos_fit(panel, canvas,
                                 with_scales(alt["calib"], s_lock, s_lock),
                                 axes=axes)
            if alt2 is not None and (got is None or alt2["score"] > got["score"]):
                got = {"calib": with_scales(alt2["calib"], s_lock, s_lock),
                       "score": alt2["score"]}
    if got is None:
        return None
    before, _ov = _crop_score(panel, canvas, cal)
    before = 0.0 if before is None else float(before)
    return {"calib": got["calib"], "score": float(got["score"]), "before": before,
            "gain": float(got["score"] - before)}


def crop_seed_at(frame, canvas, s, inset=4):
    """**某个倍数**下用模板匹配粗定一份几何 → `{"calib", "score"}`（都不行 ⇒ `None`）。

    ⚠ 块取自**面板**（`_crop_blocks` ✓）、分数一律是**整张面板的贴合分**（`_crop_score`
    ✓）—— 于是"取哪块模板"只影响**能不能搜到**，不影响"谁胜出" ✓（全世界同一把尺 ✓）。
    `locate_crop`（撒种子）与 `refine_crop`（"不改缩放"那条路也用它）共用这一处 ✓。
    """
    f = _gray(frame) if getattr(frame, "ndim", 3) == 3 else frame
    c = _gray(canvas) if getattr(canvas, "ndim", 3) == 3 else canvas
    ch, cw = c.shape[:2]
    h, w = f.shape[:2]
    if not (s > 0):
        return None
    # 面板缩回底图后盖住多大：远超底图 ⇒ 几何上讲不通，不是这个倍数
    if w / float(s) > cw * CROP_FOOT_MAX or h / float(s) > ch * CROP_FOOT_MAX:
        return None
    out = None
    for sub_xy, tpl, eff in _crop_blocks(f, inset, s, cw, ch):
        r = np.nan_to_num(cv2.matchTemplate(c, tpl, cv2.TM_CCOEFF_NORMED))
        _mx, _y, _z, ml = cv2.minMaxLoc(r)
        cal = _crop_calib_from(_peak_subpix(r, ml[0], ml[1]), sub_xy, eff, inset)
        sc, _ov = _crop_score(frame, canvas, cal)
        if sc is None:
            continue
        if out is None or sc > out["score"]:
            out = {"calib": cal, "score": float(sc)}
    return out


def locate_crop(frame, canvas, inset=4, min_score=LOCATE_MIN_SCORE, scales=None,
                refine=True, seed=None, scroll=None):
    """方式 2：面板是底图的**一块**（可能先放大了若干倍）→ 标定 dict 或 None。

    **为什么不能只按 1:1 找**（实测踩到）：客户端把小地图放大后再切块很常见。
    猴林迷宫I 就是：底图 78×203，而面板 753×612 —— 光宽度就差 9.7 倍。这种图上
    「整张缩放进面板」（fit）和「1:1 取一块」**两种都量不出来**，人手动把叠加层
    拖到 4 倍也不够（缩放上限），表现就是「匹配分 0.7 上下、怎么看都不对」。

    **怎么做**（用户 2026-09-29 报"自动定位对局部地图不好用"✗ 之后重写的一版）：
      ① **撒种子**（`_crop_blocks` + `matchTemplate`）：对每个候选倍数，从面板上取
         **中间 / 左 / 右**几块缩回底图尺度、在底图里找它 ⇒ 一批**粗略**的 (倍数, 位置)；
         有 `seed`（**手上现有那套几何**）时它就是第一号种子 ✓ —— 手工对齐过的那份
         离答案最近，从它爬最省事 ✓；
      ② **按贴合分拟合**（`_crop_pos_fit` + `_fit_crop`）：沿着**判据那行显示的那个分**
         （位移 → ±0.25 底图像素、缩放 → ±0.2% ✓）。⚠ 模板匹配只能给**整像素**峰值，
         而面板放大 5 倍时 **1 底图像素 = 5~10 个面板像素** ⇒ 只靠它，几何天生差那么
         一截 ✗；贴合分是连续的 ⇒ 能爬到那一截里面 ✓；
      ③ 谁胜出**一律由贴合分说了算**（`overlap_pearson` = 判据那行**同一个函数** ✓）。

    ⚠⚠ 老版（2026-09-29 之前）**三条毛病凑成一件事**：自动定位对 crop 图**永远给不出
    正确的那份几何**（实测森林迷宫III / 105040303：给的是 scale=12.000，真值 5.645 ✗）：
      ① **缩放只有整数 + 半档**（`abs(s − round(s)) > 0.3 ⇒ 跳过`）⇒ 5.645 这种值
         **一个候选都够不着** ✗（而差 3% 就能让分数从 0.95 掉到 0.2 一带 ✓）；
      ② **整块模板放不进底图就跳过**（`tt.shape[1] > c.shape[1]: continue`）——
         而那份真倍数缩回来是 **87.5 宽 > 底图 82** ✗ ⇒ **唯一正确的倍数被扔掉** ✗✗；
      ③ **候选之间比分**：`TM_CCOEFF_NORMED` 在不同大小的模板之间**不可比**（实测
         42×34 的小模板碰巧 0.479 > 真倍数那块 0.2 一带）⇒ 一比就选那个离谱的 12 ✗。
      ⇒ 现在：**连续缩放 + 多块模板 + 统一口径排序 + 爬坡细化** ✓。

    于是标定里同时有 `scale`（面板像素 / 底图像素）和 `view`（面板左上角对应的
    底图坐标），换算仍是 `canvas = (panel - offset) / scale + view`。

    ⭐ `scroll`（`"x"|"y"|"xy"`，默认从 `seed` 里读 ✓）—— **单轴的图只跟会动的那根轴**
    （用户 2026-09-29：滚动类型"仅X/仅Y/双轴"，单轴图上实时匹配要快 ✓）：拟合少 2/3 的
    探针 ✓，而且**不会动的那根轴一个字都不改**（由标定定死 ✓）。第一次标这张图（没种子）
    时不带它 ⇒ 两轴都搜 ✓。
    """
    f = _gray(frame)
    c = _gray(canvas)
    h, w = f.shape[:2]
    ch, cw = c.shape[:2]
    if scales is None:
        scales = _crop_scales((cw, ch), (w, h))
    # ⭐ **滚动方向**：手上这套几何（`seed`）或调用方显式给的都算数（用户 2026-09-29 ✓）。
    #   单轴 ⇒ 拟合只在会动的那根轴上走（少 2/3 的探针 ✓），而且**不会动的那根轴一个
    #   字都不改** ✓（它由标定定死；seed 给的 y 就是对的 ✓）。
    #   ⚠ 没有种子（第一次标这张图）时不带 `scroll` ⇒ 两根轴都得搜 ✓（那时还不知道该定死哪轴 ✗）。
    axes = scroll_axes(seed if seed is not None else {"scroll": scroll})

    # ⭐ **单轴图：不动的那根轴"一个字都不改"** —— 种子阶段就要钉住（`_fit_crop` 只保证
    #   "拟合时不往那根轴动"，可粗搜出来的那些种子**那根轴是随便的** ✗ ⇒ 它们一旦比分
    #   胜出，y（比如）就被悄悄改了 ✗✗。参考值 = 调用方给的 `seed` 里那个 ✓
    #   （没有 seed 就没有参考 —— 第一次标这张图时两轴都得搜 ✓）。
    ref = None
    if (seed and seed.get("mode") == MODE_CROP and has_geometry(seed)
            and not (axes[0] and axes[1])):
        _v = seed.get("view") or [0.0, 0.0]
        ref = (float(_v[0]), float(_v[1]))

    def _pin(cal):
        """把"不动的那根轴"钉回 `ref` ✓（双轴 / 没参考 ⇒ 原样返回 ✓）。"""
        if ref is None:
            return cal
        v = list(cal.get("view") or (0.0, 0.0))
        if not axes[0]:
            v[0] = ref[0]
        if not axes[1]:
            v[1] = ref[1]
        return dict(cal, view=[float(v[0]), float(v[1])])

    seeds = []
    # ① 手上这套几何（有的话）**最先**：手工对着的那份离答案最近 ✓
    if seed and seed.get("mode") == MODE_CROP and float(scales_of(seed)[0]) > 0:
        seeds.append(dict(seed))
        # 它附近也撒两个（缩放 ±3%、±6%）：手工缩放差一点点很常见 ✓
        s0 = float(scales_of(seed)[0])
        for fct in (0.94, 0.97, 1.03, 1.06):
            seeds.append(with_scales(seed, s0 * fct, s0 * fct))
    # ② 再靠粗搜：多块模板 + 贴合分排序 ✓（第一次标定这张图时只走这一路 ✓）
    got = []
    for s in scales:
        if s <= 0:
            continue
        ev = crop_seed_at(f, c, s, inset)
        if ev is not None:
            got.append(ev)
    got.sort(key=lambda v: -v["score"])
    seeds.extend(_pin(v["calib"]) for v in got[:6])
    if not seeds:
        return None
    # ③ 排名靠前的几号种子做**完整拟合**（位移 + 缩放分开定，见 `_fit_crop` ✓），
    #   其余的只按原样比分 —— 拟合一次 ~200 ms ⇒ 只给前两号，免得点一下等两秒 ✗
    ranked = []
    for cal in seeds:
        sc, _ov = _crop_score(frame, canvas, cal)
        if sc is not None:
            ranked.append({"calib": dict(cal), "score": float(sc)})
    if not ranked:
        return None
    ranked.sort(key=lambda v: -v["score"])
    best = ranked[0]
    for seed_cal in ([v["calib"] for v in ranked[:2]] if refine else []):
        got = _fit_crop(frame, canvas, seed_cal, axes=axes)
        if got is not None and got["score"] > best["score"]:
            best = got
    if best is None or best["score"] < min_score:
        return None
    out = dict(best["calib"])
    out["score"] = float(best["score"])       # ⚠ 报**贴合分**（口径同判据那行 ✓）
    return out


def locate(frame, canvas, mode):
    """定出「面板 → 底图」的换算。**mode 必填**（MODE_FIT / MODE_CROP）。

    **故意不给默认值**：以前默认是"两种都试、取分高的" —— 那等于程序替你选了。
    可是这两种画法在不同图上确实不一样（要进游戏走两步看地形动不动才知道），
    分数高低说明不了该用哪种（项目规范，见本模块开头）。所以这里强制调用方
    说清是哪个方式，值就是用户在工作台里选的那个。
    """
    if mode == MODE_FIT:
        return locate_fit(frame, canvas)
    if mode == MODE_CROP:
        return locate_crop(frame, canvas)
    return None


class CropViewTracker:
    """crop（局部小地图）**运行时跟踪**「现在显示的是底图哪一块」→ `view`（底图像素）。

    **为什么非有不可**（用户 2026-09-29 指定的下一步："开发「**局部小地图**」的世界坐标
    相关功能" ✓）：
      · `fit`（全局小地图）**没这个问题** —— 整张底图永远都在面板里、**地形不随人动**，
        标定文件里那份几何量一次就永久有效 ✓；
      · `crop`（局部小地图）**面板只是底图的一块、而且随玩家滚动** ✓（本模块顶部与
        `docs/寻路设计.md` §5 都写着"随玩家滚动 / 随人滚动"）⇒ 标定文件里存的 `view`
        只是**标定那一刻**的位置 ⇒ 人一走，`panel_to_world` 就整体偏"滚了多少 ×
        底图刻度"（量级：这张图 **1 底图像素 ≈ 7.4 世界像素** ⇒ 滚 50 像素就偏 ~370
        世界像素 ✗）⇒ 世界坐标 / `here_sets` /"怪在哪一层"**全错**，
        而现场看着像"标定没量准"或"怪在天上" ✗。
      ⇒ 所以每拍得重新问一遍"现在显示的是哪一块"，这就是本类 ✓。

    **怎么跟**：拿**面板正中间那一块**当模板（⚠ 为什么是正中间、而不是"起点算出来的那一段"，
    见 `_patch` 的说明 —— 那是最容易写错、且**一错就再也回不来**的一处 ✗），缩回
    **底图尺度**后在底图上找它落在哪 ⇒ 那就是新的 `view` ✓ ——
    换算与 `locate_crop` 同一套：`view = 匹配到的位置 − 模板在面板里的偏移 / scale` ✓
    （`offset` 在 crop 下就是模板从面板边缘往里缩了多少，见 `locate_crop` 的说明 ✓）。

    **在哪找**：
      · 有历史 ⇒ **先在上一拍附近找**（`±pad`，`pad = max(PAD_MIN, PAD_GAIN × 上一拍滚了多少)`）：
        滚动是连续的 ⇒ 附近找又快、又不容易认错到另一处相似的地形上 ✓；
        `PAD_GAIN` 是留给"突然加速"的余量 —— 真追不上时匹配分会掉 ⇒ **这一拍就自动
        全图重找** ✓（自愈，不需要额外的状态机 ✓）；
      · 没有历史（第一拍）/ 附近没找到 ⇒ **全图重找**，而且**门槛分两档**（见 `update` ②）：
        第一拍用"量得出来"那条线（`LOCATE_MIN_SCORE` = 标定时同一把尺 ✓），
        已经在跟的时候重找要 `TRUST_SCORE`（0.8）—— 因为此刻**错一次就会被当成"跟住了"、
        之后都在那一带附近找 ⇒ 再也不会自我纠正** ✗（实测：拿另一张图的底图能到 0.555、
        面板被挡一半也有 0.6 上下 ⇒ 0.55 挡不住"像但不是"）。
      实测（本机基准）：82×218 的底图 + 41×36 的模板，全图找 **~1 ms**、附近找 **~0.5 ms**
      ⇒ 30 拍/秒的实时回路扛得住 ✓（比一帧推理便宜几十倍 ✓）。

    **跟不住时绝不猜** ✗（本项目反复吃过的亏）：保留上一拍跟住的那个（没有就用标定里
    那个），`ok=False` + `why` 说清 —— `PlayerLocator.update` 会把这条写进 `note`/`short`，
    界面那一行看得见 ✓（"我这一拍的世界坐标可不可信"必须当场说得出口 ✓）。
    """

    #: 「附近找」最少给多少余量（底图像素）。2 ≈ 黄点本身的分辨率（亚像素 ±0.5 面板像素
    #: ÷ 缩放）—— 再小就成了"要求零位移"，人一走就掉进全图重找 ✓。
    PAD_MIN = 2.0
    #: 速度余量：上一拍滚了 d ⇒ 这一拍允许 3d（滚动是连续的，允许它加速到 3 倍）
    PAD_GAIN = 3.0
    #: 模板最小边长（底图像素）：太小的模板"配什么都能得高分" ⇒ 等于没量 ✗
    MIN_TPL = 6

    def __init__(self, min_score=None):
        self.min_score = float(LOCATE_MIN_SCORE if min_score is None else min_score)
        #: 最近一次**跟住**的 view（底图像素，float 二元组）；**没跟住过 = None**
        #: （⚠ 只由"跟住"那一拍写 —— 没跟住时给的"标定里那个"只进返回值，不进这里 ✗，
        #:   否则下一拍会以为"已经在跟"，全图重找的门槛就抬不起来了 ✓）
        self.view = None
        #: 这一轮**跟住过**没有（`reset` 清掉）—— 决定全图重找用哪条门槛（见 `update` ② ✓）
        self.tracked = False
        #: 最近一次的匹配分（0 = 还没跟住过）
        self.score = 0.0
        #: 连续没跟住几拍（跟住就归零）—— 界面/打点用它说"已经晃了多久"
        self.missed = 0
        #: 这一拍为什么没跟住（跟住时是空串）
        self.why = ""
        #: 这一拍在哪找到的（near / full；排查用）—— 老在 full 上说明"附近找"失灵了
        self.where = ""
        #: 上一拍 view 滚了多少（底图像素）—— 给"附近找"定余量
        self._step = None
        #: 底图灰度缓存（每拍转一次纯属浪费；底图对象在一次会话里不变 ✓）
        self._cg = None
        self._cg_key = None

    def reset(self, view=None):
        """换图 / 重标定：忘掉跟过的位置（给了 `view` 就以它当起点 ✓）。"""
        self.view = (float(view[0]), float(view[1])) if view else None
        self.tracked = bool(view)
        self.score = 0.0
        self.missed = 0
        self.why = ""
        self.where = ""
        self._step = None
        return self

    def _gray_canvas(self, canvas):
        """底图灰度（**按对象缓存** —— 底图对象在一次会话里不变 ✓）。"""
        key = (id(canvas), canvas.shape[0], canvas.shape[1])
        if self._cg is None or self._cg_key != key:
            self._cg = _gray(canvas)
            self._cg_key = key
        return self._cg

    @staticmethod
    def _inset(calib):
        """crop 下的面板内缩量 `(x, y)`（= `offset`，见 `locate_crop` 的口径 ✓）。"""
        try:
            ox, oy = calib.get("offset") or (0, 0)
            return float(ox), float(oy)
        except (TypeError, ValueError):
            return 0.0, 0.0

    def _patch(self, panel, canvas, calib, seed, clip=True):
        """取模板 → `(模板灰度, 面板左上角, 底图上的期望位置)`；取不出 ⇒ `None`。

        两种取法（`clip`），**都必须有**（各有各挡不住的情况 ✓，见 `update`）：
          · `clip=True` —— **按起点算"面板哪一段落在底图上"**，再往里收掉"起点可能已经
            错掉"的那点余量、并**不超过底图的一半**，在剩下那段里居中取。它最干净
            （一点面板边/底色都不含 ✓），但**依赖起点**：起点旧了就会把"面板边 / 底图外的
            底色"取进来 ✗（那种像素在底图上没有对应 ⇒ 分数掉下去 ⇒ 全图重找也救不回来 ✗）；
          · `clip=False` —— **面板正中间那一半**，完全不看起点 ✓（起点很旧时靠它救回来 ✓），
            代价是：面板边/底色落在正中间时（人贴着地图上/下边缘走）它会含到 ✗。

        ⚠ **上限：不超过底图的一半**（两种取法都要）—— 再大，能放它的位置就只剩几像素 ⇒
        位置根本量不出来（实测：面板比底图**宽**时，77×77 的模板在 82×218 的底图上只有
        **5 个水平位置** ⇒ 匹配永远贴着边界、x 一动不动 ✗ —— 而"面板比底图宽"正是局部
        小地图的常态：实测 105040303 是 82×218 的底图 + 502×408 的面板 ✓）。
        """
        sx, sy = scales_of(calib)
        if sx <= 0 or sy <= 0:
            return None
        ph, pw = panel.shape[:2]
        ch, cw = canvas.shape[:2]
        vx, vy = float(seed[0]), float(seed[1])
        ox, oy = self._inset(calib)
        if clip:
            # 面板四边换到底图坐标、和底图求交（超出底图的那一圈在底图上没有对应 ✓）
            bx0 = max(0.0, vx - ox / sx)
            bx1 = min(float(cw), vx + (pw - ox) / sx)
            by0 = max(0.0, vy - oy / sy)
            by1 = min(float(ch), vy + (ph - oy) / sy)
            # **收掉"起点可能已经错掉"的那点余量**：上一拍滚过多少就是它最可能错多少
            # （附近找的窗口也是按它开的 ✓ 两处同一个量 ✓）。
            mpx = (max(self.PAD_MIN, self.PAD_GAIN * abs(self._step[0]))
                   if self._step else self.PAD_MIN)
            mpy = (max(self.PAD_MIN, self.PAD_GAIN * abs(self._step[1]))
                   if self._step else self.PAD_MIN)
            bx0, bx1 = bx0 + mpx, bx1 - mpx
            by0, by1 = by0 + mpy, by1 - mpy
            if bx1 - bx0 < self.MIN_TPL or by1 - by0 < self.MIN_TPL:
                return None
        else:
            bx0, bx1 = vx, vx + pw / sx
            by0, by1 = vy, vy + ph / sy
        # 上限 + 居中
        w = min(bx1 - bx0, max(float(self.MIN_TPL), cw / 2.0))
        h = min(by1 - by0, max(float(self.MIN_TPL), ch / 2.0))
        tw = int(round(w))
        th = int(round(h))
        if tw < self.MIN_TPL or th < self.MIN_TPL:
            return None
        if tw >= cw or th >= ch:      # 必须**严格小于**底图：否则只有一个位置可放 ⇒ 量不出来
            return None
        cx0 = bx0 + (bx1 - bx0 - w) / 2.0
        cy0 = by0 + (by1 - by0 - h) / 2.0
        px0 = int(max(0, min(pw - 1, int(np.floor((cx0 - vx) * sx + ox)))))
        px1 = int(max(px0 + 1, min(pw, int(np.ceil((cx0 + w - vx) * sx + ox)))))
        py0 = int(max(0, min(ph - 1, int(np.floor((cy0 - vy) * sy + oy)))))
        py1 = int(max(py0 + 1, min(ph, int(np.ceil((cy0 + h - vy) * sy + oy)))))
        sub = panel[py0:py1, px0:px1]
        if getattr(sub, "size", 0) == 0:
            return None
        tpl = cv2.resize(_gray(sub), (tw, th), interpolation=cv2.INTER_AREA)
        # 这个模板在**底图上的期望位置**（按起点算）：附近找的窗口围着它 ✓
        ax = (px0 - ox) / sx + vx
        ay = (py0 - oy) / sy + vy
        return tpl, (px0, py0), (ax, ay)

    def _match(self, region, tpl, ox, oy, where):
        """在 `region`（底图灰度的一段）里找 `tpl` → `(分数, 绝对位置 x, y, 在哪找的)`。"""
        rh, rw = region.shape[:2]
        th, tw = tpl.shape[:2]
        if tw > rw or th > rh or tw < 1 or th < 1:
            return None
        res = np.nan_to_num(cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED))
        _, mx, _, ml = cv2.minMaxLoc(res)
        fx, fy = _subpixel_peak(res, ml)
        return float(mx), float(fx) + float(ox), float(fy) + float(oy), where

    def _search(self, cg, patch, cw, ch, axes=(True, True)):
        """在一个模板上跑「**附近找 → 全图重找**」→ `((分数, x, y, where)|None, 见过的最好分)`。

        `axes` = `(能动 X, 能动 Y)`（`scroll_axes(calib)` ✓ **单轴的图只搜那根轴** ——
        用户 2026-09-29："仅X/仅Y"的图上实时匹配要快 ✓）：
          · **附近找**：不动的那根轴 `pad = 0` ⇒ 窗口在那一轴上**只有模板那么宽** ✓
            （面板在那一轴上的小抖动不会把结果带偏 ✓），而且那根轴的搜索面积直接没了 ✓；
          · **全图重找**：不动的那根轴收成一条**带**（横带 / 竖带，高/宽 = 模板 + 一点余量 ✓）
            ⇒ matchTemplate 的面积少掉一大截（实测：底图 82×218、模板 41×36 时，
            仅 X 那条横带的面积是整图的 ~20% ✓ ~5 倍 ✓）。
          · ⚠ 只在**有先验**时才能收成带（`ax/ay` = 模板按当前 view 该落在哪儿 ✓）；
            先验不可信（还没跟住过 / 落在底图外）⇒ 老实全图找 ✓（宁慢不猜 ✗）。

        门槛**分两档**（用户 2026-09-29 ✓ 实测定的）：
          · **附近找**（有历史才走）⇒ 用「量得出来」那条线（`min_score` = 标定时同一把尺 ✓）。
            滚动是连续的 ⇒ 附近找**认错也只能错在 pad 之内**（几个底图像素 ✓）；
          · **全图重找** ⇒ 第一拍（还没跟住过）仍是 `min_score`（标定那一刻人刚好站在
            那儿 ⇒ 标定里那个 `view` 就是先验 ✓）；**已经在跟**的时候要 `TRUST_SCORE`
            （0.8，"这份几何可信"那档）。为什么必须抬：此刻错一次就会被当成"跟住了"、
            之后每拍都围着那一带找 ⇒ **再也不会自我纠正** ✗。
            实测：拿**另一张图**的底图去匹配能到 **0.555**、面板被挡掉一半（重复花纹那种
            图）也有 **0.6** 上下 ⇒ 0.55 这条线挡不住"像但不是" ✗。

        ⚠⚠ **附近找的结果"不太像"时要再全图找一次**（2026-09-29 实测踩到）：窗口只有
          `pad`（冷启动是 `PAD_MIN` = 2 个底图像素）⇒ 人**站着不动之后猛一走**
          （或传送/跳一下）真位置落在窗口外 ⇒ 匹配就贴着窗口边给个 0.6~0.7 的结果
          （过得了 0.55 ✓）⇒ 那一拍的世界坐标偏 pad 那么多，而且**看着像跟住了** ✗。
          ⇒ 采纳条件加一条：**已经在跟**的时候，附近找的分要到 `TRUST_SCORE` 才直接算数；
          不够就**再全图找一次**，谁分高用谁 ✓（全图找 ~1 ms，只在"不太像"那拍多花 ✓）。
        """
        ax_on, ay_on = axes
        tpl, _pq, (ax, ay) = patch
        th, tw = tpl.shape[:2]
        best, seen = None, 0.0
        # 门槛：附近找要够像才算数（不够 ⇒ 走下面那次全图找 ✓）
        bar = (self.min_score if not self.tracked
               else max(self.min_score, TRUST_SCORE))
        if self._step is not None and self.view is not None:
            # ⭐ 不动的那根轴**余量为 0**（单轴的图：面板在那一轴上不会滚 ✓）
            padx = (max(self.PAD_MIN, self.PAD_GAIN * abs(self._step[0]))
                    if ax_on else 0.0)
            pady = (max(self.PAD_MIN, self.PAD_GAIN * abs(self._step[1]))
                    if ay_on else 0.0)
            wx0 = int(max(0, np.floor(ax - padx)))
            wy0 = int(max(0, np.floor(ay - pady)))
            wx1 = int(min(cw, np.ceil(ax + padx) + tw))
            wy1 = int(min(ch, np.ceil(ay + pady) + th))
            if wx1 - wx0 >= tw and wy1 - wy0 >= th:
                near = self._match(cg[wy0:wy1, wx0:wx1], tpl, wx0, wy0, "near")
                if near is not None:
                    seen = max(seen, near[0])
                    if near[0] >= bar:
                        best = near
        if best is None:
            # ⭐ 全图重找：不动的那根轴收成**一条带**（先验 = 模板按当前 view 该落在哪儿 ✓，
            #    由 `_patch` 给的那个 `ax/ay` ✓）。带子比整图小一大截 ⇒ 单轴图上快好几倍 ✓。
            #    ⚠ **带里没找到像的 ⇒ 再老实全图找一次**（多花 ~1 ms，但"先验恰好是错的"
            #    —— 标定里的位置不准 / 刚换图 —— 那一拍就不会白丢 ✓ 宁慢不猜 ✗）。
            for _try_band in (True, False):
                got, sc = self._search_full(cg, tpl, cw, ch, bar,
                                            (ax, ay) if _try_band else None, axes)
                seen = max(seen, sc)
                if got is not None:
                    best = got
                    break
                if not _try_band:
                    break
        return best, seen

    def _search_full(self, cg, tpl, cw, ch, bar, prior, axes):
        """全图（或单轴时的**一条带**）里找 → `(分数, x, y, where)|None`（不够好 ⇒ None ✓）。

        `prior = (ax, ay)`（模板按当前 view 该落在哪儿 ✓）：给了它、而且是**单轴**的图
        ⇒ 只扫不动的那根轴上的一条带 ✓；`prior=None` ⇒ 整图 ✓（宁慢不猜 ✗）。
        """
        ax_on, ay_on = axes
        th, tw = tpl.shape[:2]
        reg, rx0, ry0 = cg, 0, 0
        if prior is not None and not (ax_on and ay_on):
            ax, ay = prior
            band = int(self.PAD_MIN) + 2          # 容忍先验自己差几个像素 ✓
            if not ay_on:
                ry0 = int(max(0, np.floor(ay) - band))
                ry1 = int(min(ch, np.ceil(ay) + th + band))
                if ry1 - ry0 >= th:
                    reg = cg[ry0:ry1, :]
            if not ax_on:
                rx0 = int(max(0, np.floor(ax) - band))
                rx1 = int(min(cw, np.ceil(ax) + tw + band))
                if rx1 - rx0 >= tw:
                    reg = reg[:, rx0:rx1]
            if (not ay_on and reg.shape[0] < th) or (not ax_on and reg.shape[1] < tw):
                reg, rx0, ry0 = cg, 0, 0            # 带子装不下 ⇒ 退回整图 ✓
        full = self._match(reg, tpl, rx0, ry0, "full")
        if full is None:
            return None, 0.0
        return (full if full[0] >= bar else None), float(full[0])

    def update(self, panel, canvas, calib=None):
        """这一拍的面板 → `view` 跟踪结论 dict。

        返回 `{"ok", "view": [x, y]|None, "score", "trust", "why", "missed", "where"}`；
        `view` 是**可以用的那个**：跟住 ⇒ 新跟出来的；没跟住 ⇒ 上一拍跟住的那个
        （都没有 ⇒ 标定里那个）⇒ 调用方只在 `ok` 时才该改几何 ✓（没跟住时
        标定里那份就是今天的行为，一字不变 ✓）。
        `trust` = 这一拍的匹配分到没到**可信**那一档（`TRUST_SCORE`）—— 面板被挡掉一块时
        分会落在中间那带（`trust=False`），**照样能用**，但界面该说一句"不太稳" ✓。
        """
        out = {"ok": False, "view": None, "score": 0.0, "why": "", "where": ""}
        if panel is None or canvas is None or getattr(canvas, "size", 0) == 0:
            out["why"] = "没有面板画面 / 没有底图"
            return self._done(out, calib)
        if (calib or {}).get("mode") != MODE_CROP:
            out["why"] = "这份标定不是「局部小地图」"
            return self._done(out, calib)
        if not has_geometry(calib):
            out["why"] = "标定里还没有几何（缩放 / 显示区起点）"
            return self._done(out, calib)
        cg = self._gray_canvas(canvas)
        ch, cw = cg.shape[:2]
        sx, sy = scales_of(calib)
        ox, oy = self._inset(calib)
        # 起点候选：优先"上一拍跟住的那个"（连续 ✓），其次**标定里那个**（第一拍 / 人刚
        # 重标过 ✓）。它决定两件事：模板取哪一段（`clip=True` 那路 ✓）与"附近找"的窗口 ✓。
        seeds = []
        for cand in (self.view, (calib.get("view") or (0, 0))):
            if cand is None:
                continue
            cc = (float(cand[0]), float(cand[1]))
            if cc not in seeds:
                seeds.append(cc)
        # 模板候选（顺序 = 优先尝试的顺序）：先按起点裁出来的（最干净），再**面板正中间**
        # 那块（不看起点 ⇒ 起点很旧 / 跟丢一阵时靠它救回来 ✓）。两者可能一样 ⇒ 去重 ✓
        # （按"面板上哪个矩形"去重 —— 同样的方块跑两遍纯属浪费 ✗）。
        patches = []
        for seed in seeds:
            pt = self._patch(panel, canvas, calib, seed, clip=True)
            if pt is not None and all(p[1] != pt[1] for p in patches):
                patches.append(pt)
        pt2 = self._patch(panel, canvas, calib, seeds[0] if seeds else (0, 0),
                          clip=False)
        if pt2 is not None and all(p[1] != pt2[1] for p in patches):
            patches.append(pt2)
        if not patches:
            out["why"] = ("面板里能对到底图的那一段太小 —— 面板没框全 / 标定的缩放或"
                          "显示区起点不对")
            return self._done(out, calib)
        seen = 0.0               # 这一拍见过的最好分数（只为把"差多少"说清楚 ✓）
        best = None
        pick = None
        # ⭐ **滚动方向**（用户 2026-09-29 ✓）：单轴的图只搜会动的那根轴 ⇒ 又快、又不会
        #   把"根本不动的轴"跟歪 ✓（面板在那一轴上的抖动 / 匹配噪声都进不来 ✓）。
        axes = scroll_axes(calib)
        for pt in patches:
            got, sc = self._search(cg, pt, cw, ch, axes=axes)
            seen = max(seen, sc)
            if got is not None:
                best, pick = got, pt
                break
        if best is None:
            out["score"] = float(seen)
            out["why"] = ("没在底图上找到这块面板（最好的一次匹配分只有 %.2f）—— 面板被挡住 / "
                          "换了图 / 「显示方式」选错，或标定的缩放不对"
                          % seen)
            return self._done(out, calib)
        tpl, (px0, py0), (ax, ay) = pick
        score, mx, my, where = best
        # 匹配到的位置 → view：**模板左上角对应面板的 (px0, py0)**（同一套换算，
        # ⚠ 别再加一次 inset —— `locate_crop` 那条"view 里已经含 inset"的约定在这边
        #   由 `(px0 - ox)` 抵消掉 ✓，`t_crop_roundtrip` 钉着同一个口径 ✓）。
        nvx = float(mx) - (float(px0) - ox) / sx
        nvy = float(my) - (float(py0) - oy) / sy
        # ⭐ 单轴：不动的那根轴**一个字都不改**（照上一拍/标定里那个 ✓）—— 带子搜索给回来的
        #   那个坐标本来就带着几个像素的余量 ⇒ 直接采纳就会让显示区在那一轴上慢慢漂 ✗。
        ax_on, ay_on = axes
        if seeds and not (ax_on and ay_on):
            if not ay_on:
                nvy = float(seeds[0][1])
            if not ax_on:
                nvx = float(seeds[0][0])
        prev = self.view
        self.view = (nvx, nvy)
        self.tracked = True
        self.score = float(score)
        self.missed = 0
        self.why = ""
        self.where = where
        # 下一拍的"附近"余量按这一拍真滚了多少来定（自己标定自己 ⇒ 不用拍一个速度常数 ✓）
        self._step = ((nvx - prev[0], nvy - prev[1]) if prev is not None else None)
        # 「可信吗」也一起报（口径同 `TRUST_SCORE`：0.55 那档只说明"量得出来"，0.8 才是
        # "这份几何可信"✓）。面板被挡掉一块时分数就落在中间那一带，而**平掉的那半不贡献
        # 方差** ⇒ 分数会虚高（实测：上半被挡 ⇒ 0.6~0.7，位置还差几个底图像素）
        # ⇒ 界面那行要能说"跟住但不太稳"，别让人以为这拍的位置和干净那拍一样准 ✓。
        out.update(ok=True, view=[nvx, nvy], score=float(score), where=where,
                   trust=bool(float(score) >= TRUST_SCORE))
        return out

    def _done(self, out, calib):
        """没跟住这一拍：**保留**上一拍跟住的那个（没有就用标定里那个），只记一笔 ✓。

        ⚠ `self.view` **只由"跟住"那一拍写** —— 这里只是把"这一拍该用哪个 view"
        填进返回值（标定里那个是兜底，不是"跟住了"✗，见 `tracked` 的说明 ✓）。
        """
        if out.get("ok"):
            return out
        self.missed += 1
        self.why = out.get("why") or ""
        self.where = out.get("where") or ""
        if out.get("view") is None:
            v = self.view
            if v is None:
                v = (calib or {}).get("view")
                try:
                    v = (float(v[0]), float(v[1])) if v else None
                except (TypeError, ValueError, IndexError):
                    v = None
            if v is not None:
                out["view"] = [v[0], v[1]]
        out["missed"] = self.missed
        return out


def region_match_score(frame, rect, terrain, mode=None, ok_score=TRUST_SCORE):
    """框出来的那一块画面 ↔ 底图，对一次分（**框完当场验证**的唯一一份实现）。

    **为什么抽到这里**（2026-09-27）：「框选小地图」按钮按用户要求从「路线识别」页搬进
    设置的「实时画面 · 地形叠加」组，而这个"当场验证"原来长在
    `route_panel._verify_mmap_region` 里 —— 不抽出来就是两处各写一份（README
    「口径只有一处」那条 ✗）。现在设置窗（和任何要验证的调用方）都走它。

    验证什么：框大了（把血条/聊天栏框进去）、面板被游戏 UI 挡住、「显示方式」选错 ——
    这三种的现象都是"寻路看起来坏了"，而在这里（匹配分）一眼能看出来。

    `ok_score` 沿用的就是 `locate_fit` / `locate_crop` 内部那条 `min_score` 0.8，
    **不是这里新拍的一个数** ✓。
    返回 `{"ok", "score": float|None, "why": str, "mode": str}`；`why` 是人话。
    """
    mode = mode or MODE_FIT
    if terrain is None or getattr(terrain, "canvas", None) is None:
        return {"ok": False, "score": None, "mode": mode,
                "why": "这张图还没有底图（先「生成地形图」）"}
    if not rect or len(list(rect)) != 4:
        return {"ok": False, "score": None, "mode": mode, "why": "还没框选区域"}
    x, y, w, h = (int(v) for v in rect)
    if w < 1 or h < 1:
        return {"ok": False, "score": None, "mode": mode, "why": "框选区域是空的"}
    fh_, fw_ = frame.shape[:2]
    if x < 0 or y < 0 or x + w > fw_ or y + h > fh_:
        # 画面尺寸变过（换分辨率 / 改推流参数）⇒ 旧框选区越界，说清并让它重框 ✓
        return {"ok": False, "score": None, "mode": mode,
                "why": "框选区域超出当前画面（%d×%d）—— 画面尺寸变过，重框一次" % (fw_, fh_)}
    loc = locate(frame[y:y + h, x:x + w], terrain.canvas, mode)
    if loc is None:
        return {"ok": False, "score": None, "mode": mode,
                "why": "匹配不上 —— 可能框大了（含血条/聊天栏）、面板被游戏 UI 挡住，"
                       "或「显示方式」选错了"}
    sc = float(loc["score"])
    return {"ok": sc >= ok_score, "score": sc, "mode": mode,
            "why": "" if sc >= ok_score else "偏低（%.1f 以上才算对上）" % ok_score}


def check_calib(panel, terrain, calib, mode=None, min_score=LOCATE_MIN_SCORE,
                ok_world_px=10.0):
    """量一遍「**当前这份标定到底差多少**」—— 拿真帧，一律报**世界像素**。

    用户 2026-09-27 要求"先把全局小地图弄准"，而**在此之前根本问不出这个数** ✗：
      · `score` 是无量纲的"像不像"（模板匹配才有，两点/手工标定一律没有）；
      · `resid_px` / `axis_gap_pct` 只长在两点法里、而且是**面板像素**；
      · `tools/mmap_dot_probe.py` 量的是**黄点认不认得出来**（识别率），不是标定误差。
    ⇒ 这里补上那把尺子。用户决策④：容差按像素定值、**表述一律世界坐标** ✓。

    **怎么量**：模板匹配当**独立的**尺子 —— 同一张真帧，把底图压进去另解一份几何
    （`locate` → `refine_fit` ✓），再看"**同一个面板像素，两套几何映射到世界差多少**"，
    四个角都算、报最坏的那个 ✓。尺子自己的可信度一并报出来（`score` / `px_per_world` /
    `refined`）：不报的话，人会把"尺子抖"当成"标定偏" ✗。

    ⚠ **只报数、不改任何东西**（核对不是标定 —— 要改走「双点标定」✓）。
    ⚠ `mode` 缺省取标定里那份；**crop 的尺子仍是粗搜**（`locate_crop` 没细化，用户明确
      把 crop 的自动标定暂缓 ✓）⇒ 那种情况下 `refined=False`，读数时要心里有数 ✓。
    ⚠ `ok_world_px` 默认 **10** = 和「坐标对齐误差范围」（`route.ALIGN_TOL_PX`）**同一把
      尺**：寻路判据能容忍的偏差，标定就不该更差 ✓（不是这里新拍的数 ✓）。

    返回（除带 `_px` 的，其余都是**世界像素**）：
      ok / why / verdict（人话，界面/命令行直接用）/ score / refined
      err_world（四角最大偏差）/ err_world_x / err_world_y（那个角的分量）
      err_px（换成**实时面板像素** —— 现场最直观 ✓）/ worst_panel（那个角的面板像素）
      px_per_world（1 面板像素 = 多少世界像素）
      scale_cur / scale_ref / scale_delta_pct、offset_cur / offset_ref
    """
    #: 四角的名字（只为了把"最坏在哪个角"说成人话 ✓）
    _corner_name = {(0, 0): "左上", (1, 0): "右上", (0, 1): "左下", (1, 1): "右下"}
    base = {"ok": False, "why": "", "verdict": "", "score": None,
            "refined": False, "err_world": None, "err_world_x": None,
            "err_world_y": None, "err_px": None, "worst_panel": None,
            "px_per_world": None, "scale_cur": None, "scale_ref": None,
            "scale_delta_pct": None, "offset_cur": None, "offset_ref": None}
    if panel is None or terrain is None or getattr(terrain, "canvas", None) is None:
        return dict(base, why="没有画面 / 这张图还没有底图（先「生成地形图」）")
    if not calib or not has_geometry(calib):
        return dict(base, why="这份标定还没有几何 —— 先标一次（双点标定 / 自动定位）")
    mode = mode or calib.get("mode") or MODE_FIT
    ref = locate(panel, terrain.canvas, mode)
    if ref is None:
        return dict(base, why="尺子没量出来（匹配分低于 %.2f）：画面里小地图没框全、"
                              "被游戏 UI 挡住，或「显示方式」选错了" % min_score)

    fh, fw = panel.shape[:2]
    ppw = float(terrain.px_per_world) or 1.0
    worst = None                                  # (世界误差, 角名, dx, dy, (px,py))
    for kx in (0, 1):
        for ky in (0, 1):
            px, py = float(kx * fw), float(ky * fh)
            ax, ay = panel_to_world(px, py, calib, terrain)
            bx, by = panel_to_world(px, py, ref, terrain)
            dx, dy = float(bx - ax), float(by - ay)
            d = float(np.hypot(dx, dy))
            if worst is None or d > worst[0]:
                worst = (d, _corner_name[(kx, ky)], dx, dy, (px, py))
    err, cname, dx, dy, wpt = worst
    sx_c, _sy_c = scales_of(calib)
    sx_r, _sy_r = scales_of(ref)
    err_px = err / ppw
    dlt = None if not sx_c else (sx_r - sx_c) / sx_c * 100.0
    # 「主要差在哪个方向」：分量差值说话（只报数不说话，人还得自己算一遍 ✗）
    axis = "x" if abs(dx) >= abs(dy) else "y"
    # ⚠ **尺子自己也可能是错的**：匹配分低（画面里混着游戏 UI / 面板被挡 / 显示方式选错）
    #   时上面那套搜索只是"矮子里拔高个" ⇒ 必须把可信度说出来，否则人会把"尺子抖"
    #   当成"标定偏"（`TRUST_SCORE` 与 `region_match_score` 的 `ok_score` 同一个口径 ✓）。
    score = float(ref["score"])
    trust = score >= TRUST_SCORE
    verdict = ("最大偏差 **%.0f 世界像素**（≈ %.1f 实时像素，%s角，主要差在 %s）；"
               "尺子匹配分 %.2f%s"
               % (err, err_px, cname, axis, score,
                  "" if trust else "（**偏低，这个数只能当参考**）"))
    ok = trust and err <= float(ok_world_px)
    if not trust:
        why = ("尺子本身不可信（匹配分 %.2f < %.2f）⇒ 偏差只能当参考：画面里小地图"
               "没框全、被游戏 UI 挡住，或「显示方式」选错了" % (score, TRUST_SCORE))
    elif not ok:
        why = "偏差 %.0f 世界像素 > %.0f（≈1 个实时像素）" % (err, ok_world_px)
    else:
        why = ""
    return {"ok": ok,
            "why": why,
            "verdict": verdict,
            "trust": trust,
            "score": float(ref["score"]),
            "refined": bool(ref.get("refined")),
            "err_world": err, "err_world_x": dx, "err_world_y": dy,
            "err_px": err_px, "worst_panel": [wpt[0], wpt[1]],
            "px_per_world": ppw,
            "scale_cur": float(sx_c), "scale_ref": float(sx_r),
            "scale_delta_pct": dlt,
            "offset_cur": [float(v) for v in (calib.get("offset") or (0, 0))],
            "offset_ref": [float(v) for v in (ref.get("offset") or (0, 0))]}


def stream_panel(timeout=5.0):
    """从 A 机的「小地图推流」取**一帧**面板（BGR ndarray）→ `(帧, 原因)`。

    **一处实现**：工作台的「实测精度」按钮（来源选「独立推流」时要拿一帧来核对 ✓）和
    命令行工具（`tools/mmap_dot_probe.grab_stream` / `tools/mmap_calib_check`）都要它 ——
    各写一遍"连哪口 / 最多等多久 / 清了没有"迟早分叉 ✗（和 `crop_of` 同一个理由 ✓）。
    取不到时 `原因` 是人话（没配 a_host / 超时没收到 ✓），调用方直接显示 ✓。
    """
    from core.config import get                    # 本模块只在用到处局部 import ✓
    host = get("a_host")                            # link.yaml 顶层
    port = get("minimap", "port", 5003)
    if not host:
        return None, "link.yaml 里没读到 a_host（A 机地址）"
    cli = MiniMapClient(host, port=port, timeout=float(timeout)).start()
    deadline = time.time() + max(0.5, float(timeout))
    try:
        while time.time() < deadline:
            f, _t = cli.latest(clear=True)
            if f is not None:
                return f, ""
            time.sleep(0.02)
    finally:
        try:
            cli.stop()
        except Exception:                           # noqa: BLE001
            pass
    return None, ("%.0f 秒内没收到小地图推流 —— A 机的「小地图推流」开了吗？"
                  % float(timeout))


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def scales_of(calib):
    """标定里的**两轴缩放** → `(sx, sy)`。

    `scale` = x 轴；`scale_y` **缺省 = 跟 x 一样**（老文件就是这么写的，见
    `perception/minimap.py` 顶部那个形状说明）。读两轴一律走这里 —— 各处自己读
    迟早出现"换算用 x、画图用 y"的分歧，而那种偏只有拿尺子量才看得出来 ✗。
    """
    cal = calib or {}
    try:
        sx = float(cal.get("scale") or 1.0)
    except (TypeError, ValueError):
        sx = 1.0
    sy = cal.get("scale_y")
    if sy is None:
        return sx, sx
    try:
        sy = float(sy)
    except (TypeError, ValueError):
        return sx, sx
    return (sx, sy) if sy > 0 else (sx, sx)


def with_scales(calib, sx, sy=None):
    """把两轴缩放写进一份标定（copy）—— **两轴一样时不留 `scale_y`**。

    为什么不留：老文件里就没有这个键，而"缺省 = 同 x"这条口径已经够表达"等比" ✓；
    留着 `scale_y == scale` 只会让 diff 看起来像改过、也让人以为这份标定是量过两轴的。
    """
    out = dict(calib or {})
    out["scale"] = float(sx)
    if sy is None or abs(float(sy) - float(sx)) < 1e-9:
        out.pop("scale_y", None)
    else:
        out["scale_y"] = float(sy)
    return out


def panel_to_canvas(px, py, calib):
    """面板像素 → 底图像素（calib 来自 locate / 标定文件）。"""
    sx, sy = scales_of(calib)
    ox, oy = calib.get("offset") or (0, 0)
    cx, cy = (px - ox) / sx, (py - oy) / sy
    if calib.get("mode") == MODE_CROP:
        vx, vy = calib.get("view") or (0, 0)
        cx, cy = cx + vx, cy + vy
    return cx, cy


def panel_to_world(px, py, calib, terrain):
    """面板像素 → **世界坐标**。"""
    return terrain.canvas_to_world(*panel_to_canvas(px, py, calib))


def world_to_panel(wx, wy, calib, terrain):
    """世界坐标 → **面板像素**（`panel_to_world` 的逆；**一处实现**）。

    为什么要单开一个（2026-09-27）：反向换算原来**散在三处** —— 双点标定窗的 `_uv`、
    地形视图 `image_xy`、编辑器的 `_add_background` ⇒ 谁改一处就分叉（约定 10 ✗）。
    而"把已知地标（portal / 绳梯 / 集合）投到面板像素上做核对"要用它（用户 2026-09-27
    决策③：投影产物按项目持久存 ✓）。
    ⚠ 与 `panel_to_world` **必须互逆**（crop 下 `view` 的加减号最容易写反）——
      用例 `t_world_to_panel_roundtrip` 钉着这条 ✓。
    """
    cx, cy = terrain.world_to_canvas(wx, wy)
    sx, sy = scales_of(calib)
    ox, oy = calib.get("offset") or (0, 0)
    if calib.get("mode") == MODE_CROP:
        vx, vy = calib.get("view") or (0, 0)
        cx, cy = cx - vx, cy - vy
    return cx * sx + ox, cy * sy + oy


def dot_feet(r):
    """黄点结论（`find_player_dot` 的返回）→ 面板像素 **(x, 下沿 y)**。

    ⚠ **这是"黄点在面板里算哪个位置"的唯一口径：采样（双点标定）和读数
      （`PlayerLocator.update`）必须都走这里**。用户 2026-09-27 现场验过：
      只统一一边就凭空差半个点 —— 这张图 1 个面板像素 = 7.9 世界像素，6 像素高的点
      差 12 世界像素（实测：站在 L1 上端报 (48,-222)，而真值是 (56,-205)）。

    为什么是**下沿**：游戏把小地图上的玩家标记画成一个小黄点、**下沿对着脚下**，
    所以下沿才是"这个人在地图上的位置"；重心比它高半个点，直接拿重心换算会整体抬高，
    而且和「坐标系偏移」补的那点差撞在一起、谁也说不清哪边错了 ✗。

    ⚠⚠ **不许取整**（2026-09-27 改；用户报的"同一个位置读数差 9"）：
      `find_player_dot` 给的 `x/y` 本来就是**亚像素**浮点（`cv2` 的质心），取整等于把
      读数量化成"整面板像素"一格一格跳 —— 而这张图上
      **1 个面板像素 = 8.55 世界单位**（`px_per_world / scale` = 16.082 / 1.880076）
      ⇒ **形状半个像素的变化**（抗锯齿、压缩噪声、**人在绳上/地上时那个点的画法略有
      差别** —— 眼睛看着就是"没动"）能让读数跳 **9 个世界单位** ✗✗。
      **现场**：站在 L1 上（x=56）顺着绳下来后显示 x=47（正好差一格），而他盯着收流
      小地图确认黄点**没有水平位移** —— 差的正是这一格。
      分辨率的上限消不掉（±0.5 面板像素 ≈ ±4.3 世界单位，那是**测量**本身的分辨率）；
      能消的是这层**白送的量化** ✓。
    ⚠ 脚底锚点本身不变：还是"重心 + (h-1)/2"（h=6 的点占 88..93 行 ⇒ 91+2.5 = 93.5，
      正好压在最后一行上 ✓）。原来这里 `int()` 向下取整的理由是"重心已经被取整过、
      +半个点会落到 x.5 上 ⇒ `round` 会多走一格"，可现在重心**不再被取整**了，
      那条理由也就不成立了 ✓。
    ⚠ 采样那一侧（双点标定）填的是**整数框**，`int(round())` 在那儿照旧 —— 那不是口径
      分叉：口径管的是"取哪个锚点"（下沿），取整只是**界面**的显示精度 ✓。
    """
    h = int(r.get("h") or 1)
    return (float(r.get("x") or 0.0), float(r.get("y") or 0.0) + (h - 1) / 2.0)


def solve_two_point(c1, p1, c2, p2):
    """两点标定：两对「底图像素 ↔ 面板像素」→ **两轴缩放 + 偏移 + 自查**。

    这是标定文件里 `scale` / `scale_y` / `offset` 的算法（用户 2026-09-27 要的
    「双点标定」；也就是 `docs/寻路设计.md` §5「地标法」那条的落地）。

    公式就是 `panel_to_canvas` 的反函数：`面板 = 底图 × scale + offset`，两轴各一套 ⇒
    每轴两个点给两个方程、解两个未知量（缩放、偏移）—— **恰好定解**。

    ⚠ **所以这里不能拿残差当自查**（两轴各自解的话它恒为 0，等于没查）—— 这条是
      写这个函数时才想透的，记在这儿：真正的自查是
        · `axis_gap_pct` = **两轴一致性**：游戏把整张小地图**等比**缩放 ⇒ `sx` 与
          `sy` 本该几乎相等；差得多就说明**两点没点在同一个地标上**（或画面被非等比
          拉过）⇒ 这就是用户要的"x/y 缩放不一致"那个提示 ✓；
        · `resid_px` = **按等比再拟合一次**（三个未知量、四个方程）的最大残差 ——
          这个数才有诊断价值（"如果两轴相同，最合适的几何差几个像素"）✓。

    **退化要拦住**（不能给个假数）：两点的底图 x 相同 ⇒ x 轴解不出来（分母 0）；
    y 相同 ⇒ y 轴解不出来；两点完全相同 ⇒ 两条都不成立。

    返回 `{"ok", "why", "scale", "scale_y", "offset", "iso_scale", "iso_offset",
    "resid_px", "axis_gap_pct"}`；`ok=False` 时 `why` 是人话（直接可以显示给人看）。
    """
    try:
        c1, p1 = [float(v) for v in c1], [float(v) for v in p1]
        c2, p2 = [float(v) for v in c2], [float(v) for v in p2]
        if not (len(c1) == len(p1) == len(c2) == len(p2) == 2):
            raise ValueError("不是两个坐标")
    except (TypeError, ValueError, IndexError):
        return {"ok": False, "why": "坐标要是四个数（x、y）：%r %r %r %r"
                                    % (c1, p1, c2, p2)}
    dcx, dcy = c2[0] - c1[0], c2[1] - c1[1]
    if abs(dcx) < 1e-6 and abs(dcy) < 1e-6:
        return {"ok": False, "why": "两点的底图坐标完全一样 —— 这是同一个点，量不出缩放"}
    if abs(dcx) < 1e-6:
        return {"ok": False, "why": "两点的底图 x 相同（同一列）⇒ x 轴的缩放解不出来，"
                                    "请把两点取成既不同列、也不同行"}
    if abs(dcy) < 1e-6:
        return {"ok": False, "why": "两点的底图 y 相同（同一行）⇒ y 轴的缩放解不出来，"
                                    "请把两点取成既不同列、也不同行"}
    sx = (p2[0] - p1[0]) / dcx
    sy = (p2[1] - p1[1]) / dcy
    if sx <= 0 or sy <= 0:
        return {"ok": False, "why": "算出来的缩放是负的（x %.3f / y %.3f）—— 两点多半"
                                    "左右（上下）对调了：实时图上那个地标应当由底图上"
                                    "**同一个**地标放大而来" % (sx, sy)}
    ox, oy = p1[0] - c1[0] * sx, p1[1] - c1[1] * sy
    # 按「等比」再拟合一次（3 个未知量、4 个方程 ⇒ 有残差）——抵消共用的那一个 s，
    # 偏移取两轴各自的均值（给定 s 时这才是最小二乘解）。
    mcx, mcy = (c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0
    mpx, mpy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    num = ((c1[0] - mcx) * (p1[0] - mpx) + (c2[0] - mcx) * (p2[0] - mpx)
           + (c1[1] - mcy) * (p1[1] - mpy) + (c2[1] - mcy) * (p2[1] - mpy))
    den = ((c1[0] - mcx) ** 2 + (c2[0] - mcx) ** 2
           + (c1[1] - mcy) ** 2 + (c2[1] - mcy) ** 2)
    s_iso = (num / den) if den > 0 else sx
    iox, ioy = mpx - s_iso * mcx, mpy - s_iso * mcy
    resid = max(float(np.hypot(p1[0] - (c1[0] * s_iso + iox),
                               p1[1] - (c1[1] * s_iso + ioy))),
                float(np.hypot(p2[0] - (c2[0] * s_iso + iox),
                               p2[1] - (c2[1] * s_iso + ioy))))
    return {"ok": True, "why": "",
            "scale": float(sx), "scale_y": float(sy), "offset": [float(ox), float(oy)],
            "iso_scale": float(s_iso), "iso_offset": [float(iox), float(ioy)],
            "resid_px": resid,
            "axis_gap_pct": float(abs(sy - sx) / sx * 100.0)}


def drop_stale_world_offset(calib):
    """**重新量几何**之前：把遗留的「坐标系偏移」清掉（下沿锚点之后它该是 (0, 0) ✓）。

    为什么必须清（2026-09-27 查出来的一个真 bug，33 个世界像素就这么来的）：
      · 2026-09-27 之前锚点是黄点**重心** ⇒ 那时量出来的 `world_offset` 是"重心 ↔ 原点"
        的补偿（105040303 里现存 `[7, 33]` ✓）；
      · 现在锚点是**下沿（脚底）**（见 `dot_feet`）⇒ 同一项的正常值就是 (0, 0)；
      · 而"保存新几何"那条路是 `out = dict(老的 calib)` 再改几个键（`calib_from_two_point` /
        `minimap_calib._on_save` 都是）⇒ **老值会被原样带过去** ✗ ⇒ 读完标定、算世界坐标时
        凭空偏 33 像素 —— 而 33 世界像素 ≈ 4 个面板像素，正好落在"看着像标定没量准"的量级 ✗。
    ⇒ 结论：任何一次**重新量几何**（双点标定 / 对着底图手工对齐 / 自动匹配）都把这一项归零；
      真要单独量它得走专门的采样办法（`world_offset_advice`，目前不在界面上 ✓）。
    """
    out = dict(calib or {})
    if out.get("world_offset"):
        out.pop("world_offset", None)
    return out


def calib_from_two_point(calib, r, inset=4):
    """把 `solve_two_point` 的结果写进一份标定 dict（**保留** mode/alpha/ref/…）。

    ⚠ 两种显示方式的写法**不一样**，这是这个函数唯一存在的理由：
      · fit ：`canvas = (面板 - offset) / scale` ⇒ 解出来的 `offset` 就是它，直接用 ✓；
      · crop：同一层关系，但 `offset` 在这套约定里**必须是 inset**（它是"模板从面板
        边缘往里缩了多少"，见 `locate_crop`/`panel_to_canvas`）⇒ 得把解出来的"面板系
        截距"换算成 `view`：`(面板 - inset)/scale + view = (面板 - o)/scale`
        ⇒ `view = (inset - o) / scale`（逐轴）✓。
    写错这一处的现象很隐蔽：读数**整体平移** inset 个底图像素（≈ 几十世界像素），
    看着"差不多对"（`t_two_point_save_roundtrip` 钉着这条）。

    顺带：`src` 标成 `"two_point"`（这份几何是两点解出来的，不是模板匹配也不是目测），
    **不带 `score`**（它没有匹配分 —— 留着上一次的分数只会让人以为这份是自动量出来的）。

    ⚠ 走 `drop_stale_world_offset`：**重新量几何时把遗留的「坐标系偏移」清掉**（锚点换成
      下沿之后它该是 (0, 0)；老的 (7, 33) 是重心口径的，带过去就是凭空偏 33 像素 ✗）。
    """
    out = drop_stale_world_offset(calib)
    sx, sy = float(r["scale"]), float(r.get("scale_y", r["scale"]))
    ox, oy = (float(v) for v in r["offset"])
    out = with_scales(out, sx, sy)
    if out.get("mode") == MODE_CROP:
        out["offset"] = [int(inset), int(inset)]
        out["view"] = [int(round((inset - ox) / sx)), int(round((inset - oy) / sy))]
    else:
        out["mode"] = out.get("mode") or MODE_FIT
        # ⚠ **别取整**（2026-09-27 改）：这张图上 1 个面板像素 = 7.9~8.6 世界像素 ⇒
        #   取整就是系统性偏最多半个像素。实测：解出来的 -0.452 / -10.507 被存成
        #   0 / -11 ⇒ 读数的 x 永远差 4 个世界像素、y 永远差 4 个，**看着像点没点准**
        #   （用户 2026-09-27 那个 (48,-222) 里就有这 4 个像素 ✗）。存 3 位小数就够
        #   （0.001 面板像素 = 0.009 世界像素，远小于任何读数误差）。
        out["offset"] = [round(ox, 3), round(oy, 3)]
        out["view"] = [0, 0]
    out["src"] = "two_point"
    out.pop("score", None)
    return out


def world_offset_advice(pairs, calib, cur_offset, terrain):
    """「坐标系偏移」核对：**黄点采样**（面板坐标 + 那块像素站着的世界坐标）→ 建议值。

    ⚠ **必须是"面板坐标 + 世界坐标"，而且面板坐标要是黄点（玩家标记）的位置** ——
      这条是写第一版时实算过的，记在这儿免得以后有人再走一遍：
        · 「坐标系偏移」加在**世界坐标**那一层（`panel_to_canvas` 之后）；
        · 而"底图像素 ↔ 世界坐标"那层由**地图数据**自己说了算（`world_to_canvas` 的逆）；
        · 拿**平台角那种地标**（底图坐标）+ 它的世界坐标去算 ⇒ 两者都由地图数据定死
          ⇒ 结果**恒等于 0**，是个死结论 ✗（除非你是要查"点歪了没"，那是另一件事）。
      黄点不一样：它是"玩家原点 + 偏移"画出来的 ⇒ 才带得出偏移 ✓。

    算式（就是「坐标系偏移」那个 tooltip 里让人手动做的那件事）：

        偏移_i = 世界_i - panel_to_world(黄点面板_i, 标定, 地图数据)

    两次采样给两个**独立**估计：互相一致 ⇒ 就是真实偏移，和现在存的一比就知道
    "该改成多少"（`suggest` / `delta`）✓；互相差很多 ⇒ 采样那一下人不在你说的那个
    地方（或标定不对）⇒ 把 `spread`（差多少世界像素）摆出来让人自己判 ✓。

    ⚠ 本函数**不设**"差多少算不一致"的阈值 —— 那是拍脑袋的常数（用户规矩：加常数要
      先说明来历）。所以它把数全摆出来、由界面决定怎么说。
    ⚠ **只给建议，不写配置**（用户 2026-09-27 定的口径：核对但不自动改）。

    `pairs` = `[((面板 x, 面板 y), (世界 x, 世界 y)), …]`。
    返回 `{"ok", "why", "per_point": [(ox, oy), …], "suggest": (ox, oy) | None,
    "cur": (cx, cy) | None, "delta": (dx, dy) | None, "spread": float}`。
    """
    if terrain is None or getattr(terrain, "px_per_world", None) in (None, 0):
        return {"ok": False, "why": "这张图没有底图/换算数据（先「生成地形图」）",
                "per_point": [], "suggest": None, "cur": None, "delta": None,
                "spread": 0.0}
    pts = []
    for item in pairs or []:
        try:
            p, w = item
            p = [float(v) for v in p]
            w = [float(v) for v in w]
            if len(p) != 2 or len(w) != 2:
                raise ValueError("不是两个坐标")
        except (TypeError, ValueError, IndexError):
            continue
        wx, wy = panel_to_world(p[0], p[1], calib, terrain)
        pts.append((w[0] - wx, w[1] - wy))
    if not pts:
        return {"ok": False, "why": "没填黄点的位置和它站着的世界坐标（或格式不对）"
                                    "⇒ 没有可核对的", "per_point": [], "suggest": None,
                "cur": None, "delta": None, "spread": 0.0}
    spread = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            spread = max(spread, float(np.hypot(pts[i][0] - pts[j][0],
                                                pts[i][1] - pts[j][1])))
    suggest = (int(round(sum(p[0] for p in pts) / len(pts))),
               int(round(sum(p[1] for p in pts) / len(pts))))   # 设置里存整数 ✓
    cur = delta = None
    try:
        if cur_offset is not None and len(list(cur_offset)) == 2:
            cur = (int(cur_offset[0]), int(cur_offset[1]))
            delta = (suggest[0] - cur[0], suggest[1] - cur[1])
    except (TypeError, ValueError):
        cur = delta = None
    return {"ok": True, "why": "", "per_point": pts, "suggest": suggest,
            "cur": cur, "delta": delta, "spread": spread}


def has_geometry(calib):
    """这份标定里有没有**可用的几何** —— 「标定过没有」都该用它判断。

    **为什么不看 `score`**（现场踩到）：`score` 是「模板匹配有多像」，而
    **人工对齐没有匹配分**（人拿眼睛对出来的，分数就是 0）。以前拿 score 当
    「标过没有」，于是手工对齐存下去之后仍被判成「没标过」：

      · 标定弹窗重开时把刚存好的几何**覆盖**成粗略猜测 —— 现象就是
        「我明明点了保存，重开却变回去了」；
      · 工作台那行一直显示「未标定」。

    两条都能在 `tools/selftest_minimap.py` 的 `t_save_then_reopen` 里复现。
    """
    if not isinstance(calib, dict) or not calib.get("mode"):
        return False
    if not calib.get("scale"):
        return False
    need = "view" if calib.get("mode") == MODE_CROP else "offset"
    return calib.get(need) is not None


# ══════════════════════════════════════════════════════════════
# 玩家标记（「黄点」）识别 —— S3 的门槛
# ══════════════════════════════════════════════════════════════
#
# **为什么这里宁可说「认不出」也不硬猜**：这个位置会被 `segment_of` 拿去判
# 「我在哪块平台上」，猜错比认不出糟得多 —— 认不出只是这一拍没有定位，猜错会
# 让寻路朝反方向走。所以下面的结论有三种，不是两种：
#     认得出 / 认不出 / **这张图的颜色层根本不可用**（要换来源，或改用底图相减）
#
# 标记的画法见 `docs/寻路设计.md` §2.2：
#   · WZ 素材 `Map/MapHelper.img/minimap/user`（外观随客户端，实测是亮黄点）
#   · 无素材兜底 **5×5 青色方块** `Color(0,255,255)`
#   · NPC / portal 点 `Color(132,216,243)` —— **要排除**，它不是玩家
#
# 实测（2026-09-25，寺院通道2 的采集帧）：玩家点**亮黄、6×6、22~25 像素、
# 单连通块**，中心色 **BGR≈(65,243,245)** —— B 分量 65 是压缩/抗锯齿渗出来的，
# 所以阈值**不能按纯黄 (0,255,255) 卡死**。同一天把同一套阈值放到「东部岩山V」
# 的图上，直接命中 2000~7000 像素 / 87~110 块（那张图底图本身就到处黄褐色）
# → 「颜色层够不够用」必须**当场判出来**，这正是下面 `flood` 那一路的由来。

#: 两个「颜色家族」：黄（WZ 素材 `user`，实测玩家点就是它）与青（无素材兜底方块）。
#: **分开判、黄优先**：这两族在同一张图上经常只有一族干净 —— 实测寺院通道2 的
#: **地图本身就是饱和青色**，青族被淹了，而黄族只有玩家那一个点。
#: 判据直接在 RGB 上卡「两个通道很亮、第三个很暗」，不用 HSV：
#:   黄：(R,G ≥ 200) 且 (B ≤ 150) 且 |R−G| ≤ 60
#:   青：(G,B ≥ 200) 且 (R ≤ 120) 且 |G−B| ≤ 60
#: 这样一刀同时挡掉三类捣乱的：NPC/portal 的蓝 RGB(132,216,243)（R=132 > 120）、
#: 底图的黄褐纹理 RGB(218,168,103)（G=168 < 200）、白字白框 (B/R 都很高)。
DOT_YELLOW = ("R>=200", "G>=200", "B<=150", "|R-G|<=60")
DOT_CYAN = ("G>=200", "B>=200", "R<=120", "|G-B|<=60")
#: 家族顺序 = 优先级（玩家标记实测是黄的；青是无素材时的兜底画法）。
DOT_FAMILIES = ("yellow", "cyan")
#: 亮度下限：两族判据里"亮通道 ≥ 200"已经隐含了，这个值只给文档/调参用。
DOT_V_MIN = 170
#: 候选点的边长下限 / 上限。上限按**面板宽度**给 —— 换分辨率/换窗口，面板会大一倍。
DOT_SIDE_MIN = 3
DOT_SIDE_MAX_DIV = 16          # 上限 = max(6, 面板宽 // 这个数)
#: 方形度下限（面积 / 外接矩形）：点近似方块；细长条（描边、地形线）不是点。
DOT_FILL_MIN = 0.45
DOT_ASPECT_MAX = 2.6
#: 面板边缘这些像素不算 —— 面板有边框和地图名，那圈亮色会冒充标记。
DOT_INSET = 3
#: 颜色层「被淹」的两条线（见 _is_flood）：像点的块个数上限，以及
#: 成片像素的占比 / 绝对下限（取大的那个）。**不要用掩码总像素判**：压缩噪声
#: 会撒出成百上千个 1~2 像素的碎点，谁也不像标记，却能顶到几千。
#:
#: 绝对值给得**小**（60）是实测定的：面板大小差 5.6 倍（独立推流 753×612、
#: 从实时画面 134×109），而标记在两种面板里都只占 **0.09%** —— 也就是说
#: 「多大算成片」必须跟着面板走，绝对值只兜住"面板特别小"那种情况。
#: 踩过的坑：绝对值原来写 400，而独立推流那颗 28×24 的点就有 411 像素 ——
#: 差一点就把正常的一帧判成「颜色层不可用」。
DOT_MAX_CANDS = 12
DOT_FLOOD_RATIO = 0.01
DOT_FLOOD_PX = 60
#: 跨帧：相邻两拍位移上限（超过就当噪声，把上一拍位置丢掉重捕），
#: 以及「连续几拍都找到」才算确认（单帧亮斑不算，屏幕上到处是亮斑）。
DOT_MAX_JUMP_PX = 40.0
DOT_CONFIRM_FRAMES = 2

#: 漏检之后**沿用上一帧位置**的时间上限（毫秒）—— 口径同 `perception/tracker.py`
#: 的 `debounce_ms`（那边是"漏检期间保留幽灵框"）。为什么不干脆一直沿用：那位置
#: 会越来越像真的（人跑远了它还在原地），而且寻路会拿它当"我在哪块平台上"。
DOT_HOLD_MS = 500.0
#: 有上一帧位置时，只在它周围这么大一块里搜（面板像素，半边长）。
#: **这么做有两个好处**：① 快 —— 收流那条来源的面板是 753×612，全图掩码 +
#: 连通域每拍要几毫秒，缩到 37×37 基本免费；② 稳 —— 底图上的杂色根本进不来，
#: 「被淹」那个否决条件也就用不上了（见 find_player_dot 的 `near`）。
DOT_ROI_PAD = 18
#: 漏检期间按（衰减的）速度外推：每漏一拍乘一次这个系数，总位移有上限
#: （同 MobTracker._drift：不能冻住，也不能飘到地图另一头）。
DOT_GHOST_DECAY = 0.75
DOT_GHOST_MAX_SHIFT = 8.0        # 幽灵外推的总位移上限（面板像素）

#: 上面这四个可以**从 config/live.yaml 覆盖**（界面在「路线识别 → 小地图定位」里调）。
#: 键名 → 默认值：改键名要一起改 `track_params()` 和界面那排输入框。
TRACK_KEYS = (("mmap_hold_ms", DOT_HOLD_MS),
              ("mmap_roi_pad", DOT_ROI_PAD),
              ("mmap_max_jump", DOT_MAX_JUMP_PX),
              ("mmap_ghost_shift", DOT_GHOST_MAX_SHIFT))


def track_params(live_cfg=None):
    """从 config/live.yaml 的 dict 里取出四个跟踪参数（缺的用默认值）。

    **只有这一处知道键名与默认值**：界面、实时线程、命令行工具都调它，
    免得三处各写一份、改了一处另外两处还是老值。
    """
    cfg = live_cfg or {}
    out = {}
    for key, dflt in TRACK_KEYS:
        raw = cfg.get(key)
        try:
            out[key] = float(dflt) if raw is None else float(raw)
        except (TypeError, ValueError):
            out[key] = float(dflt)          # 配置写坏了 → 退回默认，不炸
    return out


def tracker_kwargs(track):
    """`track_params()` 的 dict → `PlayerDotTracker(**…)` 的关键字。"""
    return {"hold_ms": track.get("mmap_hold_ms"),
            "roi_pad": track.get("mmap_roi_pad"),
            "max_jump": track.get("mmap_max_jump"),
            "ghost_max_shift": track.get("mmap_ghost_shift")}


def dot_color(panel, family=None):
    """族判据本身（RGB 上直接卡），**不含面板边缘的 inset** —— 也用来量颜色。

    `family` 取 "yellow" / "cyan" 只取一族；None = 两族并集。
    """
    b = panel[:, :, 0].astype(np.int16)
    g = panel[:, :, 1].astype(np.int16)
    r = panel[:, :, 2].astype(np.int16)
    yellow = (r >= 200) & (g >= 200) & (b <= 150) & (np.abs(r - g) <= 60)
    cyan = (g >= 200) & (b >= 200) & (r <= 120) & (np.abs(g - b) <= 60)
    if family == "yellow":
        return yellow
    if family == "cyan":
        return cyan
    return yellow | cyan


def dot_mask(panel, family=None):
    """「玩家标记色」掩码（uint8 0/1），已挖掉面板边缘。

    `family` 取 "yellow" / "cyan" 只取一族；None = 两族并集（**只用于看证据图**，
    判断该用哪族请走 find_player_dot —— 两族分开判才不会互相拖累）。
    """
    m = dot_color(panel, family).astype(np.uint8)
    ins = DOT_INSET
    if m.shape[0] > 2 * ins and m.shape[1] > 2 * ins:
        m[:ins, :] = 0
        m[-ins:, :] = 0
        m[:, :ins] = 0
        m[:, -ins:] = 0
    return m


def dot_side_max(panel_w):
    """「像点」的边长上限：按**面板宽度**给（换分辨率/换窗口，面板会大一倍）。"""
    return max(6, int(panel_w) // DOT_SIDE_MAX_DIV)


def _dot_candidates(m, panel_w):
    """掩码 → (像「点」的连通块列表, 成片像素数)。

    每项：{x, y（重心）, w, h, area, fill}。**不做「选一个」** —— 多候选时该选谁
    要跨帧才看得出来（见 PlayerDotTracker），单帧选一个等于瞎猜。

    第二个返回值 `dense` 是「边长达到最小点尺寸的块」的像素总和 —— **判断"颜色层
    有没有被淹"要用它，不能用掩码总像素**：压缩噪声会撒出成百上千个 1~2 像素的
    碎点，它们谁也不像标记，却能把总像素顶到几千。
    """
    side_max = dot_side_max(panel_w)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    dense = 0
    for i in range(1, n):
        x, y, w, h, a = (int(t) for t in stats[i])
        if w < DOT_SIDE_MIN or h < DOT_SIDE_MIN:
            continue                       # 碎点：不算候选，也不算"成片"
        dense += a
        if w > side_max or h > side_max:
            continue
        if a < DOT_SIDE_MIN * DOT_SIDE_MIN - 2:      # 3×3 至少 7 个像素
            continue
        fill = a / float(max(1, w * h))
        if fill < DOT_FILL_MIN:
            continue
        if max(w, h) > DOT_ASPECT_MAX * min(w, h):
            continue
        out.append({"x": float(cent[i][0]), "y": float(cent[i][1]),
                    "w": w, "h": h, "area": a, "fill": round(fill, 2)})
    # 越大越方越像点：给个稳定顺序（多候选时先看像的那个）
    out.sort(key=lambda c: (-c["area"], -c["fill"]))
    return out, dense


def roi_around(center, shape, pad):
    """以 center 为中心、半边长 pad 的矩形（裁到画面内）→ (x0,y0,x1,y1) 或 None。

    画面太小（裁出来没意义）返回 None，让调用方退回全画面搜。
    """
    if center is None:
        return None
    h, w = shape[:2]
    cx, cy = float(center[0]), float(center[1])
    x0, x1 = int(cx - pad), int(cx + pad) + 1
    y0, y1 = int(cy - pad), int(cy + pad) + 1
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 2 * DOT_SIDE_MIN or y1 - y0 < 2 * DOT_SIDE_MIN:
        return None
    return (x0, y0, x1, y1)


def _is_flood(n_cands, dense, area_search):
    """这个搜索区里的颜色层还能不能信 → True = 不可用。

    两条独立的信号，命中任一条就算淹：
      · 像点的块**个数**太多（一片碎块里总有一个"最像的"，但选它是瞎猜）；
      · 成片像素占比太高（底图自己就一大片标记色，比如东部岩山V）。
    """
    if n_cands > DOT_MAX_CANDS:
        return True
    return dense > max(DOT_FLOOD_PX, DOT_FLOOD_RATIO * max(1, area_search))


def _calib_roi(panel_shape, calib, terrain):
    """「底图覆盖到的那块面板区域」(x0,y0,x1,y1) —— 把面板标题/边框排除在外。

    **为什么要它**（实测踩到）：面板上方有一条「地图名 + 图标」的标题带，颜色是
    暖色 —— 颜色层把它整块命中，一块 214×272 的面板光那条就吃掉 4000+ 像素，
    于是每帧都被判成「颜色层不可用」。底图只覆盖地图区，所以**拿底图在面板上的
    范围当搜索区**，标题自然落在外面。

    只在 fit 方式下算得出来：crop 方式下面板显示的整块就是底图的一部分（标题在
    面板自己的边框里，靠 DOT_INSET 那一圈挡掉）。
    """
    if not has_geometry(calib) or calib.get("mode") != MODE_FIT:
        return None
    cv_ = getattr(terrain, "canvas", None)
    if cv_ is None:
        return None
    sx, sy = scales_of(calib)                  # 两轴各自（老文件 ⇒ 两个一样 ✓）
    ox, oy = calib.get("offset") or (0, 0)
    ch, cw = cv_.shape[:2]
    # ⚠ 参数是 numpy 的 `.shape`（**高在前**）：按 (宽, 高) 解会把宽高弄反，
    # 搜索结果区被裁成正方形的一角 —— 实测表现是「底图相减明明减掉了，却找不到点」。
    h, w = panel_shape[:2]
    x0, y0 = max(0, int(ox)), max(0, int(oy))
    x1, y1 = min(w, int(ox + cw * sx)), min(h, int(oy + ch * sy))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def _apply_roi(m, roi):
    """把掩码裁到 roi 里（其余置 0）。roi 为 None 就原样返回。"""
    if not roi:
        return m
    out = np.zeros_like(m)
    x0, y0, x1, y1 = [int(v) for v in roi]
    np.copyto(out[y0:y1, x0:x1], m[y0:y1, x0:x1])
    return out


def _basemap_extra_mask(panel, calib, terrain, m, family=None):
    """「底图上这块也是标记色」的掩码 → 从面板掩码里减掉它，只留多出来的。

    **为什么要这一层**：有的图底图本身就到处黄褐色（实测东部岩山V 那张图，
    纯颜色层命中 7000+ 像素）。玩家标记是**画在底图之上**的东西，所以
    「面板命中 − 底图在同一个位置也命中」才是标记（NPC 点同样会被留下，靠
    尺寸/方形度和跨帧跟踪再筛）。底图与面板有几何误差，减之前把底图掩码**胀**开
    几个像素，宁可信底图（少留几个候选）也别把底图的纹理当标记。
    """
    sx, sy = scales_of(calib)
    ox, oy = calib.get("offset") or (0, 0)
    vx, vy = (calib.get("view") or (0, 0)) if calib.get("mode") == MODE_CROP else (0, 0)
    h, w = panel.shape[:2]
    # canvas = (panel - offset)/scale + view  ⇒  panel = (canvas - view)*scale + offset
    mat = np.float32([[sx, 0, ox - vx * sx], [0, sy, oy - vy * sy]])
    warped = cv2.warpAffine(terrain.canvas, mat, (w, h),
                            flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
    base = dot_mask(warped, family)
    # 膨胀核跟着缩放走（2026-09-27 改两轴后取两轴里**大的那个**：宁可多胀一点，
    # 也不要比原来胀得少 —— 这条本来就是"宁可信底图"）
    k = max(3, int(round(max(sx, sy) * 2))) | 1        # 奇数核
    base = cv2.dilate(base, np.ones((k, k), np.uint8))
    return (m & (1 - base)).astype(np.uint8)


def find_player_dot(panel, calib=None, terrain=None, roi=None, near=None,
                    ref_w=None):
    """面板里找玩家标记 → 结论 dict（**认不出也是结论，不要硬凑**）。

    搜索顺序 = **黄族 → 青族**，每族先颜色层、再底图相减层；第一个「有候选且不算
    被淹」的组合胜出。分开判很要紧：实测寺院通道2 的**地图本身就是饱和青色**，
    青族被淹，而黄族只有玩家那一个点 —— 混在一起判会把这条结论一起丢掉。

    返回：
        ok         找到像标记的点
        x, y       面板像素坐标（重心；ok 时才有意义）
        w, h, area 外接矩形与像素数
        bgr        重心附近的实际颜色（中位）
        mask_px    胜出族的命中像素数
        dense_px   胜出族的「成片」像素数（判断被淹用它，不是 mask_px）
        candidates 胜出族的候选个数
        family     "yellow" / "cyan"
        layer      "color" / "basemap"（结论来自哪一层）
        flood      True = 颜色层没帮上忙（靠底图层给的，或干脆给不出）
        diag       每族的诊断句（拒绝时拼进 reason，便于人判断该怎么救）
        roi        实际用的搜索区（有 fit 标定时会自动排除面板标题带）
        reason     一句话，能直接贴进日志/状态行

    `roi` 不传时：有 fit 标定就自动用「底图覆盖区」（排除面板标题/边框）。
    `near`（面板坐标）不传时是**首次捕获**语义；传了表示"我上一帧在这儿" ——
    那时多候选按**离 near 最近**挑，而且"被淹"不再是否决条件（见 loop 里那段）。
    """
    used_roi = roi or _calib_roi(panel.shape[:2], calib, terrain)
    if used_roi:
        rx0, ry0, rx1, ry1 = [int(v) for v in used_roi]
        area_search = max(1, (rx1 - rx0) * (ry1 - ry0))
    else:
        area_search = max(1, int(panel.shape[0] * panel.shape[1]))

    diag = []
    any_flood = False
    hit = None
    for fam in DOT_FAMILIES:                   # 黄族优先（实测玩家标记就是黄的）
        m0 = _apply_roi(dot_mask(panel, fam), used_roi)
        mask_px = int(m0.sum())
        for layer in ("color", "basemap"):
            if layer == "basemap":
                # 颜色层救不回来时才用「面板 − 底图」：这一层要标定 + 底图
                if not has_geometry(calib) or terrain is None \
                        or getattr(terrain, "canvas", None) is None:
                    continue
                m = _basemap_extra_mask(panel, calib, terrain, m0, fam)
                if int(m.sum()) >= mask_px * 0.5:   # 没减掉多少 → 这层没意义
                    continue
            else:
                m = m0
            # `ref_w`：**给裁剪图用**。像点的边长上限是按面板宽度定的，而局部
            # 搜索传进来的是一小块（37×37）—— 按那块的宽度算，上限只剩 6px，
            # 28×24 的玩家点会被尺寸筛选直接挡掉（实测就是这么"局部搜索没生效"的）。
            cands, dense = _dot_candidates(m, ref_w or panel.shape[1])
            flooded = _is_flood(len(cands), dense, area_search)
            # `near`（有上一帧位置）时**被淹不再是否决条件**：那会儿搜的就是
            # 上一帧周围一小块，"离上一帧最近"比"整张图大不大"可靠得多。
            if cands and (not flooded or near is not None):
                hit = {"family": fam, "layer": layer, "mask_px": mask_px,
                       "dense_px": dense, "candidates": len(cands),
                       "all": cands, "flooded": bool(flooded)}
                break
            any_flood = any_flood or flooded
            diag.append("%s族/%s层：像点的块 %d 个、成片 %d 像素%s"
                        % ("黄" if fam == "yellow" else "青", layer, len(cands),
                           dense, "（被淹）" if flooded else ""))
        if hit is not None:
            break

    base = {"mask_px": 0, "dense_px": 0, "flood": False, "candidates": 0,
            "layer": "", "family": "", "all": [], "roi": used_roi,
            "diag": diag, "w": 0, "h": 0, "area": 0,
            "x": 0.0, "y": 0.0, "bgr": None}

    if hit is None:
        # 认不出就**认不出**：这时候给一个"最像的块"，下游会当成"我在哪块平台上"，
        # 比认不出糟得多（实测东部岩山V 那张图：底图本身就到处黄褐色）。
        why = "；".join(diag) if diag else "两族都没有像点的块"
        pre = "颜色层不可用" if any_flood else "认不出玩家标记"
        return dict(base, ok=False, flood=any_flood, reason=(
            "%s（%s）—— 换来源（独立推流），或给它一份底图标定走「底图相减」"
            % (pre, why)))

    cands = hit["all"]
    if near is not None:
        # 有上一帧位置 → **离它最近的那个**就是它（距离比"哪块更大"可靠得多）
        c = min(cands, key=lambda t: (t["x"] - near[0]) ** 2
                + (t["y"] - near[1]) ** 2)
    else:
        c = cands[0]
    cx, cy = int(round(c["x"])), int(round(c["y"]))
    patch = panel[max(0, cy - 3):cy + 4, max(0, cx - 3):cx + 4]
    # 颜色取**这个族判据命中的那些像素**的中位 —— 直接取补丁中位会被周围压暗，
    # 报出来是个"看着不像标记"的颜色（实测：标记核心 R,G≈245，补丁中位只有 167）。
    pm = dot_color(patch, hit["family"]) if patch.size else None
    sel = patch[pm] if pm is not None else None
    if sel is not None and len(sel):
        med = np.median(sel, axis=0).astype(int)
    elif patch.size:
        med = np.median(patch.reshape(-1, 3), axis=0).astype(int)
    else:
        med = np.zeros(3, int)
    reason = ("找到玩家标记（%s 族 / %s 层）" % (hit["family"], hit["layer"])
              if len(cands) == 1 else
              "找到 %d 个候选（%s 族 / %s 层；跨帧跟踪会挑）"
              % (len(cands), hit["family"], hit["layer"]))
    out = dict(base, ok=True, x=c["x"], y=c["y"], w=c["w"], h=c["h"],
               area=c["area"], bgr=tuple(int(v) for v in med), reason=reason)
    out.update(hit)
    # flood 的含义：**颜色层没帮上忙**（结论要么是靠底图层得到的，要么根本给不出）
    out["flood"] = (hit["layer"] == "basemap")
    return out


class PlayerDotTracker:
    """跨帧跟踪玩家标记：多候选挑最近的、跳变丢掉重捕、连续两拍才算确认。

    **为什么非要跨帧**：单帧里"亮黄的方块"不止玩家一个（NPC 点、特效、地图纹理），
    而玩家点是**连续移动**的：用上一拍位置当下先验，一跳 40px 以上的直接判噪声。
    三条做法都照 `perception/tracker.py` 那套（怪物/角色的防抖）：

      ① **有上一帧位置就只在它周围搜一小块**（`DOT_ROI_PAD`）—— 又快又稳：
         收流那条来源的面板 753×612，全图掩码+连通域每拍几毫秒，缩到 37×37 基本
         免费；底图上的杂色也根本进不来；
      ② **漏检就沿用上一帧位置**（`DOT_HOLD_MS` 内），口径同那边的"幽灵框"——
         不然读数一秒里闪好几次"认不出"，寻路也白丢一小段惯性；
      ③ 幽灵期间按**衰减的速度**往前推一点（总位移有上限），同 `_drift`：既不能
         冻在原地（人在跑，位置越差越远），也不能一直推（会飘到地图另一头）。
    """
    def __init__(self, hold_ms=None, roi_pad=None, max_jump=None,
                 ghost_max_shift=None):
        self.x = None
        self.y = None
        self.hits = 0
        self.frames = 0
        # 这四个是**界面参数**（config/live.yaml，见 track_params）：
        #   沿用窗口 / 搜索半径 / 跳变上限 / 外推上限 —— 都能在「路线识别」里调。
        self.hold_ms = float(DOT_HOLD_MS if hold_ms is None else hold_ms)
        self.roi_pad = int(DOT_ROI_PAD if roi_pad is None else roi_pad)
        self.max_jump = float(DOT_MAX_JUMP_PX if max_jump is None else max_jump)
        self.ghost_max_shift = float(DOT_GHOST_MAX_SHIFT if ghost_max_shift is None
                                     else ghost_max_shift)
        self.missed = 0            # 连续漏检拍数（口径同 MobTracker 的 missed）
        self.last_roi = None       # 上一次实际搜的区域（诊断/自检用）
        self._vx = self._vy = 0.0  # 速度（面板像素/拍，平滑过）
        self._shifted = 0.0        # 幽灵外推累计走了多远
        self._last_seen = 0.0
        self._last = None          # 上一次成功的完整结论（补位时原样复用）

    def update(self, panel, calib=None, terrain=None, roi=None, now=None):
        now = time.monotonic() if now is None else now
        self.frames += 1
        r = self._search(panel, calib, terrain, roi)

        if not r["ok"]:
            # 漏检：防抖窗口内**沿用上一帧位置**（同 MobTracker 的幽灵框做法）。
            # `hold_ms <= 0` = 关掉这条（口径同项目里其它"0 = 关"的参数）。
            # ⚠ 不判这个的话，"窗口设 0"在同一拍里仍然算没过期 —— Windows 的
            # `time.monotonic()` 粒度约 15.6 ms，两次调用会返回同一个值。
            if (self.hold_ms > 0 and self._last is not None
                    and (now - self._last_seen) * 1000.0 <= self.hold_ms):
                self.missed += 1
                hx, hy = self._drift()
                self.x, self.y = hx, hy
                return dict(self._last, x=hx, y=hy, held=True,
                            missed=self.missed,
                            confirmed=self.hits >= DOT_CONFIRM_FRAMES,
                            reason="上一帧位置（漏检 %d 拍）" % self.missed)
            self.hits = 0
            self.missed = 0
            self._last = None
            return dict(r, hits=0, confirmed=False, held=False, missed=0)

        if self.x is not None:
            d = ((r["x"] - self.x) ** 2 + (r["y"] - self.y) ** 2) ** 0.5
            if self.max_jump > 0 and d > self.max_jump:
                # 跳变：**丢掉上一拍位置**再报认不出 —— 下一拍从零重捕（真传送也
                # 只丢一拍；留着旧位置的话，人真换了地方就永远追不上了）
                self.x = self.y = None
                self.hits = 0
                self.missed = 0
                self._last = None
                return dict(r, ok=False, hits=0, confirmed=False, held=False,
                            missed=0, reason=(
                                "候选位置跳变 %.0f px（阈值 %.0f）—— 当噪声丢弃，"
                                "下一拍重捕" % (d, self.max_jump)))

        # 速度：够算"漏检期间大概往哪边走"就行，平滑一下免得被单拍抖动带偏
        if self.x is not None:
            self._vx = 0.5 * self._vx + 0.5 * (r["x"] - self.x)
            self._vy = 0.5 * self._vy + 0.5 * (r["y"] - self.y)
        else:
            self._vx = self._vy = 0.0
        self.x, self.y = r["x"], r["y"]
        self.hits += 1
        self.missed = 0
        self._shifted = 0.0
        self._last_seen = now
        self._last = dict(r)
        return dict(r, hits=self.hits, held=False, missed=0,
                    confirmed=self.hits >= DOT_CONFIRM_FRAMES)

    # ---------------- 内部 ----------------

    def _search(self, panel, calib, terrain, roi):
        """这一拍在哪搜：有上一帧位置 → 它周围一小块（`near` 语义）；否则全画面。"""
        near = (self.x, self.y) if self._last is not None else None
        if near is not None:
            # 半径要**盖得住玩家点本身**（面板大，点也大：收流那条 753×612 上
            # 点是 28×24），不然点会被裁掉半截、重心也跟着偏。
            pad = max(self.roi_pad, dot_side_max(panel.shape[1]))
            used = roi_around(near, panel.shape, pad)
            self.last_roi = used
            if used is not None:
                x0, y0, x1, y1 = used
                rr = find_player_dot(panel[y0:y1, x0:x1],
                                     near=(near[0] - x0, near[1] - y0),
                                     ref_w=panel.shape[1])
                if rr["ok"]:
                    # **坐标要加回偏移**，候选列表也一样（下游按面板坐标用）
                    return dict(rr, x=rr["x"] + x0, y=rr["y"] + y0,
                                layer="roi",
                                all=[dict(c, x=c["x"] + x0, y=c["y"] + y0)
                                     for c in rr["all"]])
                # 这一小块里没有 → 退回全画面（可能真换了地方 / 标定变了）
        self.last_roi = roi
        return find_player_dot(panel, calib, terrain, roi=roi)

    def _drift(self):
        """漏检时按（衰减的）速度往前推一点，总位移有上限（同 MobTracker._drift）。"""
        step = DOT_GHOST_DECAY ** max(0, self.missed - 1)
        dx, dy = self._vx * step, self._vy * step
        d = (dx * dx + dy * dy) ** 0.5
        left = max(0.0, self.ghost_max_shift) - self._shifted
        if d > left and d > 1e-9:                 # 到了上限就不走了
            k = max(0.0, left) / d
            dx, dy = dx * k, dy * k
        self._shifted += (dx * dx + dy * dy) ** 0.5
        return (self.x or 0.0) + dx, (self.y or 0.0) + dy


# ══════════════════════════════════════════════════════════════
# 玩家定位（S3 的成品）：面板画面 → 世界坐标 → 在哪条段
# ══════════════════════════════════════════════════════════════

class PlayerLocator:
    """把三件事串成**一次调用**：黄点识别 → 世界坐标 → 站在哪条段。

        PlayerDotTracker      面板里找玩家标记（黄点，含跨帧确认）
          ↓ panel_to_world    面板像素 → 底图像素 → 世界坐标（用标定 + miniMap 的 mag）
          ↓ segment_of        世界坐标 → 地形里的哪条段（Segment.index）

    **为什么非得有人把它串起来**：这三步以前分别散在「标定弹窗」和「命令行诊断」
    里，各自只做一段 —— 真接进实时回路时又得再拼一遍，而拼错的方式很隐蔽
    （面板↔底图漏一次换算，或者把段号当成别的什么编号）：算出来的位置整体错，
    下游却当成"我在哪块平台上"照用。所以这里把口径固定死，谁都别再拼第二份。

    **算不出来是常态，也是结论**：`world_x/world_y/segment_id` 一律是 None 而不是
    0（0 是合法的世界坐标 —— 地图西北角），`note` 写清为什么。
    """

    def __init__(self, map_id=None, track=None):
        #: 跟踪参数（`PlayerDotTracker` 的关键字：hold_ms/roi_pad/max_jump/
        #: ghost_max_shift）。**必须记在 locator 上**：换图会重建 tracker，
        #: 不记的话"界面里调过一次、一换图又回默认"。
        self.track = {k: v for k, v in (track or {}).items() if v is not None}
        self.tracker = None
        self.map_id = None
        self.terrain = None
        self._calib_cache = {}
        #: 「坐标系偏移」缓存（见 _world_offset）：(时刻, (x, y))
        self._off_cache = None
        #: ⭐ **局部小地图（crop）的运行时 view 跟踪**（用户 2026-09-29 任务 1 ✓）——
        #: crop 的面板**随玩家滚动**，标定文件里那个 `view` 只在标定那一刻成立
        #: ⇒ 每拍得重新问"现在显示的是哪一块"（理由详见 `CropViewTracker` ✓）。
        #: 挂在 locator 上（**有状态**：跨帧记"上一拍滚了多少"给附近找定余量 ✓），
        #: 与 locator 同寿命；`load()`（换图）时清掉 ✓。
        self._view_track = CropViewTracker()
        self._new_tracker()
        self.load(map_id)

    def _new_tracker(self):
        self.tracker = PlayerDotTracker(**self.track)

    def set_track(self, **kw):
        """改跟踪参数（界面拨一下立刻生效，并**换图也保留**）。"""
        for k, v in kw.items():
            if v is None:
                self.track.pop(k, None)
            else:
                self.track[k] = v
        self._new_tracker()
        return self

    def use_track_config(self, live_cfg):
        """按 `config/live.yaml` 那份配置更新跟踪参数（界面那排输入框改完就叫它）。

        键名与默认值只在 `tracker_kwargs()` / `track_params()` 里写一份。
        """
        return self.set_track(**tracker_kwargs(track_params(live_cfg)))

    def load(self, map_id):
        """（换）加载某张图的地形。传 None 就是"不定位"。"""
        map_id = map_id or None
        if map_id == self.map_id and self.terrain is not None:
            return self
        self.map_id = map_id
        self.terrain = (mapdata.load(map_id, with_canvas=True)
                        if map_id else None)
        # 换图 → 位置不连续，别拿上一张图的点去跟踪（跳变判据会乱丢）。
        # ⚠ 重建时把界面调的跟踪参数带上（见 self.track）。
        self._new_tracker()
        self._calib_cache = {}
        # 换图 ⇒ "现在显示的是底图哪一块"整个换了一套（底图都不一样）⇒ 忘掉跟过的位置 ✓
        self._view_track.reset()
        return self

    def calib_for(self, src, ttl=1.0, zoom=None):
        """读某条来源的标定，**带 1 秒缓存**。

        为什么不每次读文件：实时回路 30 拍/秒，每拍读一次 JSON 纯属浪费；
        为什么不只在启动时读一次：标定在弹窗里随时会被改，读死了要重启才生效。
        1 秒的折中足够（弹窗保存后本来就会自己刷新一次）。

        ⭐ `zoom`（**可选**，不传 = 与从前完全一样）：只肯要**在这个 zoom 下标出来的**
           那份（见 `mapdata.zoom_of` / `mapdata.load_calib`）。zoom 变了 ⇒ 返回
           **None**（不是"拿旧的凑合"）⇒ 上层会说"这份标定不适用"，而不是闷头给出
           整倍数错的坐标 ✗。
           ⇒ **换图之后一定要带上 `switch_map` 回回来的那个 zoom**：同一张图换个 zoom
           推，面板像素尺寸就变了，而标定里的 `scale`/`offset` 是对着当时那个尺寸
           量出来的（这正是 2026-10-01 修的那个坑）。
        """
        now = time.monotonic()
        key = (src, int(zoom) if zoom is not None else None)
        hit = self._calib_cache.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
        # ⚠ **不传 zoom 时必须用"老调用形式"**（2026-10-01 修 ✗）：`tools/selftest_minimap`
        #   的「实测精度」用例把 `load_calib` 换成了只认 `(_m, src=None)` 的替身 ⇒
        #   一律多传一个 `zoom=None` 会当场 `TypeError`（那个用例就这么挂的 ✓）。
        #   ⇒ **可选参数不该改变老调用形式** —— 这条对所有可能被替身/被 mock 的函数都成立 ✓。
        if not self.map_id:
            cal = None
        elif zoom is not None:
            cal = mapdata.load_calib(self.map_id, src, zoom=zoom)
        else:
            cal = mapdata.load_calib(self.map_id, src)
        self._calib_cache[key] = (now, cal)
        return cal

    def forget_calib(self):
        """忘掉标定缓存 —— **标定刚被改过**（弹窗保存/重标）时叫它，下一拍立刻用新的。

        为什么要有这个入口：`calib_for` 带 1 秒缓存（实时回路 30 拍/秒，每拍读一次 JSON
        不值当）。那是给"一直跑着的回路"用的；而**人刚点完保存**时，那 1 秒的延迟会让人
        以为"保存没生效 / 读数根本没变"（用户 2026-09-27 报的就是这个形状），所以保存
        那条路上显式忘一次 ✓。别改成"每次都不缓存"：那是把 30 拍/秒的读盘成本还给实时回路。
        """
        self._calib_cache = {}
        return self

    def _world_offset(self, calib):
        """「坐标系偏移」(x, y)：算出来的世界坐标**加上**它（用户 2026-09-26 要求）。

        为什么要它：坐标是按**黄点的下沿（脚底）**算的（见 `dot_feet`），而
        "那个下沿 ↔ 游戏里的玩家原点"这层对应是游戏自己定的 —— 与其猜，不如给一个
        显式偏移让人自己量一次。读数、段号、foothold、绳梯判据都在**加完之后**算
        （见 `update`），所以寻路的"我在哪块平台上"也跟着准。
        ⚠ 锚点是下沿时（2026-09-27 起），这一项的正常值接近 **(0, 0)**（见 `offset_of`）。

        **按地图 id 存**（用户 2026-09-26 定的口径）：它跟标定走 —— 存在
        `datasets/map/<id>.mapcalib.json` 的 `sources.<来源>.world_offset`，
        和 `scale/offset/view/alpha` 同一份（每张图、每个来源各自一套 ✓）。
        直接读 `calib`（调用方已经按来源取好了），所以面板和实时线程天然一致。
        """
        return offset_of(calib)

    def update(self, panel, src=None, calib=None, terrain=None, fh_xtol=0):
        """面板画面（BGR）→ 定位结论 dict。

        返回：
            ok         这一拍**拿到了世界坐标**（认不出点 / 没标定 → False）
            confirmed  黄点是"连续两拍都在附近"确认过的（决策层该只认这种）
            px, py     黄点在面板里的像素（**重心** —— 画标记用；认不出时是上一次的残值，别用）
            world_x/world_y  世界坐标；没算出来是 None
                       ⚠ 它取的是黄点**下沿（脚底）**换算出来的那个点（`dot_feet`）+
                       「坐标系偏移」，**不是** `px/py` 直接换算 —— 采样（双点标定）
                       用的也是下沿，两边同一口径读数才对得上（见 `dot_feet`）
            segment_id 站在哪条段（None = 没落在平台上）
            foothold_id      踩着哪条 foothold（字符串 id；None = 没落在平台上）。
                       **"我在不在某个 foothold 集合里"用这个，不用 segment_id**：
                       自动串段会把"要跳/攀才能互通"的并成同一段（实测 105090600 的
                       第 0 段把一面 388 像素高的悬崖当成了平台边缘），集合是人工圈的，
                       判据只能是 foothold id ∈ 集合（见 core/zones.py、docs/寻路设计.md §12）
            ladder_id        踩在**哪根绳梯**上（"L2" 这种编号；None = 不在绳上）。
                       编号来自 `zones.ladder_ids` —— 与编辑器画在绳上的、以及「爬」那条
                       边里写的绳号**同一套**。判据是几何的（`mapdata.ladder_at`），
                       和执行器"对齐到绳的 x"用的**判定参数**是两回事（见设置→判定参数）
            dot        黄点那一层给的说明（认不出时就是原因）
            view/view_ok/view_score/view_trust/view_why
                       ⭐ **局部小地图（crop）的运行时跟踪**（用户 2026-09-29 任务 1 ✓）：
                       `view` = **这一拍算坐标时用的**「显示区起点」（底图像素）；`view_ok` =
                       这一拍真的**跟住了**吗（`None` = 不是 crop，没这回事 ✓）；
                       `view_score` = 跟的匹配分；`view_trust` = 到没到 `TRUST_SCORE`
                       那档（`False` = 跟住了不太稳，典型是面板被挡掉一块 ✓）；
                       `view_why` = 没跟住的原因。
                       ⚠ 跟不住时用的是**标定里那份**（= 老行为）⇒ `view_ok=False` 就意味着
                       "这一拍的世界坐标可能整体偏"（crop 会滚动！见 `CropViewTracker` ✓）。
            note       没算出坐标 / 没落平台的原因（**可能很长**，给 tooltip/日志）
            short      `note` 的一句话版本（正常时是空串）——给**贴在画面上**的
                       读数用：那是一行不换行的字，塞下 `note` 会横穿整个画面
        """
        src = src or SRC_STREAM
        terrain = terrain if terrain is not None else self.terrain
        # `fh_xtol`：「脚下是哪条 foothold / 哪条段」的 **x 方向容差**，由调用方从
        # 设置里的「坐标对齐误差范围」传进来（0 = 老行为）。只兜"原本找不到"的情形，
        # 详见 `core.mapdata.Terrain.foothold_below` 的 xtol 说明 ✓。
        _xt = max(0, int(fh_xtol or 0))
        if calib is None:
            calib = self.calib_for(src)
        # ⭐⭐ **局部小地图（crop）：先把"现在显示的是底图哪一块"量出来**（用户 2026-09-29
        #   任务 1 ✓）。crop 的面板**随玩家滚动** ⇒ 标定文件里那个 `view` 只在标定那一刻
        #   成立，人一走后面所有换算都是整体平移的 ✗（理由与量级见 `CropViewTracker` ✓）。
        #   ⚠ **跟住才改几何**：没跟住时用标定里那份 = 今天的行为一字不变 ✓，只把
        #     "这一拍没跟住"写进 note/short（否则人会拿着一个偏了 370 世界像素的读数
        #     当准的用 ✗ —— 那正是"世界坐标整体偏"那类问题的样子 ✓）。
        #   ⚠ 必须在**黄点识别之前**跑：黄点的搜索 ROI / 底图遮罩都吃 `view`
        #     （`_calib_roi` / `_basemap_extra_mask` ✓）⇒ 跟住了连"认黄点"都更准 ✓。
        _vt = {"ok": None, "view": None, "score": None, "why": ""}
        _crop_note = ""
        if (calib or {}).get("mode") == MODE_CROP \
                and terrain is not None and getattr(terrain, "canvas", None) is not None:
            _vt = self._view_track.update(panel, terrain.canvas, calib)
            if _vt.get("ok") and _vt.get("view"):
                calib = dict(calib, view=[float(v) for v in _vt["view"]])
                if not _vt.get("trust"):
                    # 跟住了但不够可信（典型：面板被挡掉一块 —— 平掉的那半不贡献方差，
                    # 分数会虚高 ✓）。**照样用**（比拿标定里那个旧位置强 ✓），但要说出来 ✓。
                    _crop_note = ("局部小地图：这一拍跟住了，但匹配分只有 %.2f（不太稳，"
                                  "面板被挡住 / 画面糊了？）"
                                  % float(_vt.get("score") or 0.0))
            else:
                _crop_note = ("局部小地图：这一拍没跟住显示区（%s）⇒ 用的是标定里那个"
                              "位置，世界坐标可能整体偏" % (_vt.get("why") or "说不清"))
        r = self.tracker.update(panel, calib=calib, terrain=terrain)
        out = {"ok": False, "confirmed": bool(r.get("confirmed")),
               # held：这一拍没认出黄点，位置是**上一帧**的（防抖窗口内沿用）。
               # 决策层可以用（两三拍之内还算得准），但要能分辨出来。
               "held": bool(r.get("held")), "missed": int(r.get("missed") or 0),
               "px": r["x"], "py": r["y"], "world_x": None, "world_y": None,
               "segment_id": None, "foothold_id": None, "src": src,
               # 局部小地图（crop）的运行时跟踪（见上面那段；不是 crop 时一律 None ✓）
               "view": _vt.get("view"), "view_ok": _vt.get("ok"),
               "view_score": _vt.get("score"), "view_why": _vt.get("why") or "",
               "view_trust": _vt.get("trust"),
               "dot": r["reason"], "note": "", "short": "认不出黄点"}
        if not r["ok"]:
            out["note"] = r["reason"]
            return out
        if terrain is None:
            out["note"] = "没有这张图的地形/底图（先「生成地形图」）"
            out["short"] = "没有地形数据"
            return out
        if not has_geometry(calib or {}):
            out["note"] = ("还没量过这张图的「面板 → 底图」换算"
                           "（来源「%s」下点「标定…」）"
                           % SRC_LABEL.get(src, src))
            out["short"] = "还没标定"
            return out

        # ⚠ **取黄点的下沿（脚底），不是重心**（2026-09-27 改，见 `dot_feet`）：
        #   双点标定采样时填的就是下沿 ⇒ 两边不同口径的话，站回同一个地方读数却
        #   差 (h-1)/2 个面板像素（实测 6 像素高的点 = 12 世界像素，用户报过），
        #   而那个差**看着像标定不准**、最难查 ✗。`px/py`（画标记用）仍是重心 ✓。
        fx, fy = dot_feet(r)
        wx, wy = panel_to_world(fx, fy, calib, terrain)
        # 「坐标系偏移」：**在这一步加**（用户自己精确标定用）—— 后面的范围检查、
        # 段号、foothold、绳梯判据全都用加完之后的坐标，口径才是一份。
        # 锚点换成下沿之后，这一项的含义就是"下沿 → 玩家原点"的**残差**（正常接近 0）。
        ox, oy = self._world_offset(calib)
        if ox or oy:
            wx, wy = wx + ox, wy + oy
        out["world_x"], out["world_y"] = wx, wy
        b = terrain.bounds
        if b and not (b[0] - 1 <= wx <= b[2] + 1 and b[1] - 1 <= wy <= b[3] + 1):
            # 落在图外基本只有一个原因：标定（或底图）不对。说出来，别把
            # 一个地图外的坐标交给寻路。
            out["note"] = ("算出来在底图范围外 (%.0f, %.0f) —— 标定或底图不对？"
                           % (wx, wy))
            out["short"] = "算到图外了"
            return out
        seg = terrain.segment_of(wx, wy, xtol=_xt)
        out["segment_id"] = seg.index if seg is not None else None
        # 脚下**那条** foothold 的 id（字符串，和 zones 文件里的写法一致）。
        # 与 segment_of 是同一套口径（都走 foothold_below），多算一次可忽略。
        f = terrain.foothold_below(wx, wy, xtol=_xt)
        out["foothold_id"] = str(f.fid) if f is not None else None
        # **在不在绳梯上**（2026-09-26 用户要求：小地图那行要能报「绳梯：L2」）。
        # 判据用 mapdata 现成的 `ladder_at`（绳的 x ± LADDER_DX、y 在绳段 ± LADDER_PAD）；
        # 编号用 `zones.ladder_ids` —— 与编辑器画在绳上的 L1/L2、以及「爬」那条边里写的
        # 绳号**必须是同一套**（不然同一个 "L2" 在两处指两根绳，人会被带沟里）。
        L = terrain.ladder_at(wx, wy)
        out["ladder_id"] = (zones.ladder_ids(terrain).get(id(L))
                            if L is not None else None)
        out["ok"] = True
        if seg is None:
            out["note"] = "坐标算出来了，但脚下没有平台（半空 / 墙里？）"
            out["short"] = "脚下没有平台"
        elif out["held"]:
            # 位置是上一帧的（这一拍没认出黄点，防抖窗口内沿用）—— 说出来，
            # 别让人以为这是这一拍量出来的
            out["note"] = ("这一拍没认出黄点，用的是上一帧的位置（漏检 %d 拍）"
                           % out["missed"])
            out["short"] = "上一帧位置"
        else:
            out["short"] = ""
        if _crop_note:
            # 局部小地图没跟住 ⇒ **必须说出来**（这一拍的世界坐标可能整体偏了"滚了多少"✗）：
            # 拼在已有 note 后面（原来那句可能是"脚下没有平台"，两条都要看得见 ✓）；
            # short 只在还没有话说的时候顶上（画面上那行越短越好，见它的说明 ✓）。
            out["note"] = (out["note"] + "；" if out["note"] else "") + _crop_note
            if not out["short"]:
                out["short"] = "局部小地图没跟住"
        return out


def screen_to_world(player, sx, sy):
    """**游戏画面坐标 → 世界坐标**（用**玩家做锚**：`画面 = 世界 − Camera` ✓）。

    为什么要有它（用户 2026-09-27："chase状态需要判定一下怪物位于的foothold集合，若不与
    玩家处于同一个，需要先下达前往任务"）：**怪只有画面坐标**（`perception.world_state.Mob`
    的 `x/y` = 框中心、画面像素 ✓），而查"怪站哪块 foothold / 属于哪些集合"（
    `core.mapdata.Terrain.foothold_below` + `core.zones.Zones.set_of` ✓）**要世界坐标**
    ⇒ 中间这一跳就是本函数 ✓（全仓**唯一**一处"画面 → 世界" ✓）。

    原理（`docs/寻路设计.md`）：世界 → 屏幕是 `屏幕 = 世界 − Camera`；
    ⚠ 那条公式**没有比例尺** ⇒ 它的前提就是「**画面 1 像素 = 世界 1 像素**」（推流按原始
      分辨率抓、没有缩放 ✓）。这条前提不成立时本函数只是**刻度不对** ⇒ 调用方要把推出来的
      世界坐标**留痕**（写进 note / 打点 ✓），别当精密测量用 ✗。
    Camera 不用另外找：玩家**同时**有画面坐标（`player.x` / `player.bottom`）和世界坐标
    （`player.world_x/world_y`，来自小地图黄点 ✓）⇒ `Camera = 世界 − 画面`，直接相减 ✓。
    ⚠ 口径与「黄点脚底」对齐（约定 10）：`sy` 要给**脚底**的画面 y —— 怪是
      `mob.y + mob.h / 2` ✓、玩家那头用 `player.bottom`（框底 ✓，`world_y` 也是脚底 ✓）；
      x 两边都用中心 ✓（`player.x` ↔ `world_x` ✓）。
    ⚠ 拿不到玩家的世界坐标（没定位）⇒ 返回 `None`（**不猜** ✓，调用方退回老行为 ✓）。

    ⭐⭐ **优先用「可信相机」**（`player.cam_x/cam_y`，用户 2026-09-29 任务 2 ✓）：
    上面那个"现算"的相机**每一拍都在动**，而它的两个来源一个是权威的（黄点世界坐标 ✓）、
    一个不稳（YOLO 玩家框 ⇒ 框抖 Δ ⇒ 每只怪的世界坐标同量平移 Δ ✗）。
    `PlayerTracker` 已经把"跟黄点对得上的那几拍"挑出来存在 `player.cam_x/cam_y` 上
    （口径与这里**完全一致** ✓ 见 `perception/tracker._cam_of`）⇒ 这里照用，全仓
    "画面 → 世界"因此走**同一份相机** ✓。
    ⚠ 没有它（`None` ⇒ 还没建立）⇒ **老口径现算**，行为一字不变 ✓。
    """
    wx0 = getattr(player, "world_x", None)
    wy0 = getattr(player, "world_y", None)
    if wx0 is None or wy0 is None:
        return None
    # ⭐ **脚底偏移**（用户 2026-09-28 ✓）：框底不一定正好压在脚底（鞋底阴影 / 披风 /
    #   特效会让框多出一截 ✗）⇒ 用镜像值补正 ✓（`live_thread` 每帧灌进 `world_state` ✓
    #   —— 这边**不能**反过来 import decision，会绕成循环依赖 ✗）。
    from perception import world_state as _ws
    cam_x = getattr(player, "cam_x", None)
    cam_y = getattr(player, "cam_y", None)
    if cam_x is None or cam_y is None:
        cam_x = float(wx0) - float(getattr(player, "x", 0.0) or 0.0)
        cam_y = (float(wy0) - (float(getattr(player, "bottom", 0.0) or 0.0)
                               + float(_ws.FOOT_OFFSET_PX)))
    return (float(sx) + float(cam_x), float(sy) + float(cam_y))


def apply_to_player(player, loc):
    """把定位结论写进 `WorldState` 的 `Player` 那一页（**一处口径**）。

    三个字段的含义见 `perception/world_state.py`；`segment_id` 是**地形**的段号
    （`core/mapdata.Terrain.segment_of` 给的）—— 寻路要的就是它。
    """
    player.world_x = loc.get("world_x")
    player.world_y = loc.get("world_y")
    player.segment_id = loc.get("segment_id")
    # held = 这一拍没认出黄点、位置沿用上一帧（防抖窗口内）。决策层可以照用，
    # 但要能分辨出来 —— 将来做"精细落点"时这类位置不该当新鲜观测。
    player.world_held = bool(loc.get("held"))
    player.world_note = loc.get("note") or ""
    return player


# ══════════════════════════════════════════════════════════════
# 「现在显示的是底图哪一块」（crop / 局部小地图专用）
# ══════════════════════════════════════════════════════════════

#: tools/map_terrain_view.py 把底图放大到大约这个宽度（更宽的底图不缩小）
_OVERLAY_W = 1280


def overlay_zoom(canvas_w):
    """底图 → 叠加图 的放大倍数（和 tools/map_terrain_view.render 同一套算法）。"""
    return max(1, int(round(float(_OVERLAY_W) / max(1, int(canvas_w)))))


def view_rects(loc, panel_wh, canvas_wh):
    """crop 模式：算出「面板现在显示的是底图哪一块」→ (底图矩形, 叠加图矩形)。

    面板是**放大过**的（scale = 面板像素 / 底图像素），所以它在底图坐标里覆盖
    的大小是 `面板 / scale`，起点是 `view - inset/scale`。

    注意 inset 只能减一次：模板是从面板里裁掉 inset 再去匹配的，匹配位置本身
    就是底图坐标（见 locate_crop / panel_to_canvas 的注释）；scale 也是同理 ——
    匹配时面板已经按 1/s 缩回去了，所以 view 就是底图坐标，**不要再折算一次**。
    """
    pw, ph = panel_wh
    sx, sy = scales_of(loc)
    inset = float((loc.get("offset") or [4, 4])[0])
    vx, vy = loc.get("view") or (0, 0)
    x0, y0 = vx - inset / sx, vy - inset / sy
    x1, y1 = x0 + pw / sx, y0 + ph / sy
    z = overlay_zoom(canvas_wh[0])
    return ((int(x0), int(y0), int(x1), int(y1)),
            (int(x0 * z), int(y0 * z), int(x1 * z), int(y1 * z)))


def overlay_draw_rects(loc, panel_wh, canvas_wh):
    """→（叠加图上的**源矩形**（None = 整张）, 面板里的**目标矩形**）。

    把标定好的叠加画到别处时就靠这一对：源矩形从叠加图上取一块，
    贴到目标矩形（面板坐标）。

      fit  ：面板里放的是**整张**底图 → 源 = 整张叠加图，目标 = 面板里从 `offset`
             起、`底图 × scale` 那么大的那一块（面板比它大的地方本来就该留白）。
      crop ：面板里放的是底图的**一块** → 源 = 那一块（`view_rects` 算好的），
             目标 = 整个面板。

    **为什么抽成一份**：标定弹窗的叠加层、以及「把叠加画到实时画面上」都要用
    同一个映射。各写一份迟早对不上，而这种偏最阴 —— 两边看着都"差不多"，
    只有拿尺子量（或者进游戏走两步）才发现差一截。
    """
    pw, ph = panel_wh
    cw, ch = canvas_wh
    if loc.get("mode") == MODE_CROP:
        _canvas_rect, img_rect = view_rects(loc, panel_wh, canvas_wh)
        return img_rect, (0, 0, int(pw), int(ph))
    sx, sy = scales_of(loc)
    ox, oy = loc.get("offset") or (0, 0)
    return None, (int(ox), int(oy),
                  int(round(cw * sx)), int(round(ch * sy)))


def frame_overlay_rects(loc, frame_rect, canvas_wh, calib_panel=None):
    """→（源矩形, **实时画面坐标**下的目标矩形）。

    面板在实时画面里的位置由 `frame_rect=(fx, fy, fw, fh)` 给出（工作台
    「小地图定位 → 框选小地图」框出来的那一块，存 config/live.yaml 的 `mmap_crop`），
    标定说面板里该怎么放 —— 两者拼一下就是"画在画面的哪儿"。

    **收流来源同样适用**：那一路小地图虽然另有独立推流（画质好、黄点不糊），
    但小地图面板本身**也在主画面里**（只是被 H.264 压过），所以"它在画面的哪儿"
    照样是框出来的那一块，和来源无关。

    `calib_panel` = **标定当时那块面板的尺寸** `(w, h)`（不传 = 不折算，老行为）。
    ⚠ 为什么必须传（2026-09-26 实测踩过）：标定记的是"**那块**面板像素 ↔ 底图像素"的
    换算，而这里要往"**现在**框出来的那一块"上画 —— 两者尺寸可能差好几倍（典型：A 机
    推流带 zoom，标定面板 754px 宽，而主画面里的小地图只有 251px）⇒ 不折算的话，
    叠图会被整块放大：`底图 134 × scale 5.63 = 754` 的方块糊在 251px 的小地图上。
    传了它，就按 `标定面板 ÷ 当前那块画面` 的比例把几何先换算过去再画 ——
    于是 A 机改 zoom、或面板尺寸变了，叠图都不会再被放大。

    ⚠⚠ **折算哪些量**（2026-09-29 修，用户报"**刚标定完叠加图显示就不对**"✓）：
      标定那份说的是 `底图 = (p标定 − offset) / scale + view`，而框选那块面板与它
      **看的是同一块地图**、只是像素尺寸差 z 倍（`p标定 = z · p框选`）⇒ 代进去：
      `底图 = (p框选 − offset/z) / (scale/z) + view` ⇒
      **`scale` 与 `offset` 要除以 z，`view` 不动** ✓（`view` 是**底图坐标**，跟面板
      放大几倍无关）。
      ⛔ 以前把 `view` 也除了 z ✗ ⇒ 叠图整块平移 `view × (1 − 1/z)` 个底图像素
      （实测 105040303：偏 (7.3, 69.8) 个底图像素，而那块面板一共才装得下
      (58.1, 60.8) 个 ⇒ **叠图跑到地图别处去了** ✗）。
      **为什么这个错一直没露**：`fit` 的 `view` 恒为 `[0, 0]`（整张底图不动）⇒ 除不除
      都一样 ✗；只有 `crop` 的 `view` 是"现在显示在地图哪儿"、必然非 0 ⇒ 一除就错 ✓
      —— 于是现象恰好是"**只有局部小地图**、刚标定完叠图就不对"。
    """
    fx, fy, fw, fh = (int(v) for v in frame_rect)
    loc2 = loc
    if calib_panel:
        try:
            pw = float(calib_panel[0] or 0)
        except (TypeError, ValueError, IndexError):
            pw = 0.0
        if pw > 0 and fw > 0 and abs(pw / float(fw) - 1.0) > 1e-6:
            z = pw / float(fw)                  # 标定面板 → 当前那块画面 的比例
            sx, sy = scales_of(loc)             # 两轴都要折算（比例 z 本身是等比的 ✓）
            loc2 = with_scales(loc, sx / z, sy / z)
            loc2["offset"] = [float(v) / z for v in (loc.get("offset") or (0, 0))]
            # ⛔ **`view` 不折算**（`底图 = (p − offset/z)/(scale/z) + view` ⇒ view 原样 ✓）
            #    —— 这里原来写的是 `loc2["view"] = [v / z ...]`，crop 图上会让叠图整体
            #    平移几个到几十个底图像素（见上面那段 ⚠⚠ ✓）。
    src, (dx, dy, dw, dh) = overlay_draw_rects(loc2, (fw, fh), canvas_wh)
    return src, (fx + dx, fy + dy, dw, dh)


def overlap_pearson(panel, canvas, calib, max_samples=6000, min_samples=64):
    """**按重叠区**算面板↔底图的相关 → `(score|None, overlap, why)`。

    **一处口径**：`align_score`（给人看的那行）与 `locate_crop`（自动定位挑候选）都用它 ✓
    —— 两处各写一个"像不像"的算法，迟早变成"自动定位说 0.9、判据那行说 0.4" ✗。
    采样点**落在底图里**的才参与 ⇒ 面板边上的游戏 UI / 底图外那圈自然被排除 ✓。

    ⚠ **传灰度图（2 维）也认**：自动定位要算几百次，每次现转一次灰度就是几百毫秒 ✗
      （实测 675×600 的面板转一次 ~1 ms）⇒ 调用方**自己转一次、反复用** ✓。
    """
    if panel is None or getattr(panel, "size", 0) == 0:
        return None, 0.0, "没有面板画面"
    if canvas is None or getattr(canvas, "size", 0) == 0:
        return None, 0.0, "这张图还没有底图（先「生成地形图」）"
    if not has_geometry(calib):
        return None, 0.0, "这份标定还没有几何（缩放 / 偏移）"
    panel = panel if getattr(panel, "ndim", 3) == 2 else _gray(panel)
    canvas = canvas if getattr(canvas, "ndim", 3) == 2 else _gray(canvas)
    ch, cw = canvas.shape[:2]
    sx, sy = scales_of(calib)
    if sx <= 0 or sy <= 0:
        return None, 0.0, "标定里的缩放不是正数"
    # ⚠ 字段口径照抄 `panel_to_canvas`（**唯一那处** ✓）：crop 才加 `view`
    ox, oy = calib.get("offset") or (0, 0)
    vx, vy = ((calib.get("view") or (0, 0)) if calib.get("mode") == MODE_CROP
              else (0, 0))
    ph, pw = panel.shape[:2]
    # 采样步长：面板可能 750×800 ⇒ 别一次算几十万像素（1 底图像素就够 ✓）
    step = max(1, int(round((pw * ph / float(max(1, max_samples))) ** 0.5)))
    xg, yg = np.meshgrid(np.arange(0, pw, step), np.arange(0, ph, step))
    bx = np.rint((xg - ox) / sx + vx).astype(np.int64)
    by = np.rint((yg - oy) / sy + vy).astype(np.int64)
    inside = (bx >= 0) & (bx < cw) & (by >= 0) & (by < ch)
    n = int(inside.sum())
    ov = float(n) / float(max(1, xg.size))
    if n < max(16, int(min_samples)):
        return None, ov, ("面板几乎没有落在底图上（只有 %.1f%% 的采样点在里面）—— "
                          "缩放 / 偏移 / 显示区起点不对？" % (ov * 100.0))
    try:
        a = panel[yg[inside], xg[inside]].astype(np.float32)
        b = canvas[by[inside], bx[inside]].astype(np.float32)
    except Exception as e:                              # noqa: BLE001
        return None, ov, "采样失败：%s" % e
    if float(a.std()) < 1e-3 or float(b.std()) < 1e-3:
        return None, ov, "面板或底图那一块几乎是纯色（没有可比的纹理）"
    sc = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(sc):
        return None, ov, "算不出相关（那一块是平的）"
    return max(0.0, min(1.0, sc)), ov, ""


def align_score(panel, terrain, calib, max_samples=6000, min_samples=64,
                near=False, near_px=3, near_max=9):
    """**当前这套几何贴不贴** → `{"score", "overlap", "ok", "why", "near"}`（**不做搜索** ✓）。

    和 `locate*`（找位置）是两件事：这里固定用你**手上这套几何**，只问"面板和底图对得上吗"。
    为什么要它（用户 2026-09-29 第 ⑤ 条 ✓ 原话："对于局部地图，评判分应该按局部来，
    必须制定剪裁范围才能让评分合理"）：
      · `locate` 给的是"**整张面板**在底图里最像哪儿"——crop（局部小地图）下，面板只是
        底图的一小块，那个分跟"你对齐了没有"没关系 ✗（手工对齐的更干脆：没有分 ✓）；
      · 这里改成**只算重叠区**：把面板像素按当前几何映到底图上，**落在底图里**的那些
        才参与打分 ⇒ "剪裁范围"由**这套几何自己定出来**，不需要人再去框一块 ✓
        （要靠人剪裁才能算分，等于把这件客观的事变主观 ✗）。
    分数 = 两串像素的**归一化相关**（和 `TM_CCOEFF_NORMED` 的 0~1 同一个口径 ✓）：
      1.0 = 完全重合；≥ `TRUST_SCORE`(0.8) 基本就是"对上了" ✓。
    `overlap` = 面板里落在底图上的像素占比 —— **这个数小就说明"框多了"**（把游戏 UI /
      面板外那圈框进了 crop 区域）⇒ 分再高也别全信 ✓（这也是 `_refresh_judge` 要一起
      报它的原因 ✓）。

    ⭐⭐ `near=True` ⇒ 多算一圈**邻域**，返回 `near = {"score", "dx", "dy"}`（**相对现在这套
      几何**平移 `(dx, dy)` 底图像素之后最好的那一档；`(0, 0)` 就是现在 ✓）。

      **为什么非有不可**（用户 2026-09-29 下一句就是："森林迷宫III项目，我已经手工对得很
      好了**匹配分还是很低**！"✗）：**绝对分不是一把通用的尺** —— 有的图客户端小地图
      的画法与底图**不同源**（自己另画一套线/底色），那类图**对得再准也只有 0.3~0.5** ✗
      ⇒ 光看"0.42 低于 0.55"会让人以为"我没对齐"，于是把对的几何改坏 ✗✗（判断贴合分
      **低不低**，只能和**它自己**比：附近有没有明显更高的一档 ✓）。
      `near["score"] - score <= 0.02` ⇒ **已经在这一档最好** ✓（这时分数低是"这张图
      天生如此"，不是你没对齐 ✓）；差得多 ⇒ `near` 里那对 `(dx, dy)` 就是**该往哪挪** ✓。

    ⚠ **只读**：不搜位置、不改标定、不写文件；`near=False` 时每拍 ~0.3 ms，
      开邻域约 9~13 次 ⇒ ~4 ms（标定弹窗每拍算一次，够用 ✓）。
    """
    out = {"score": None, "overlap": 0.0, "ok": False, "why": "", "near": None}
    canvas = getattr(terrain, "canvas", None) if terrain is not None else None
    sc, ov, why = overlap_pearson(panel, canvas, calib, max_samples, min_samples)
    out["overlap"] = ov
    if sc is None:
        out["why"] = why
        return out
    out["score"] = sc
    out["ok"] = True
    if near:
        out["near"] = _near_best(panel, canvas, calib, sc,
                                 near_px, near_max, max_samples, min_samples)
    return out


def _near_best(panel, canvas, calib, base, near_px, near_max, max_samples, min_samples):
    """邻域里最好的一档（**爬坡**：先 3×3，最好的落在边上就再往外走一档）→
    `{"score", "dx", "dy"}`（`(0,0)` = 现在这套 ✓）。

    ⚠ 只平移 `view`（crop）/ `offset`（fit）—— 也就是**只回答"往哪挪"**；缩放不在里面 ✓
    （缩放的微调是另一件事，见 `refine_calib` ✓；两者混在一起搜，人读不懂那个数）。
    """
    if not has_geometry(calib):
        return None
    # 邻域要算十几次 ⇒ 灰度**只转一次**（不然每转一次 675×600 就是 ~1 ms ✗）
    panel = panel if getattr(panel, "ndim", 3) == 2 else _gray(panel)
    canvas = canvas if getattr(canvas, "ndim", 3) == 2 else _gray(canvas)

    def at(dx, dy):
        c = dict(calib)
        if calib.get("mode") == MODE_CROP:
            v = list(calib.get("view") or (0, 0))
            c["view"] = [float(v[0]) + dx, float(v[1]) + dy]
        else:
            o = list(calib.get("offset") or (0, 0))
            c["offset"] = [float(o[0]) - dx * float(scales_of(calib)[0]),
                           float(o[1]) - dy * float(scales_of(calib)[1])]
        s, _ov, _w = overlap_pearson(panel, canvas, c, max_samples, min_samples)
        return s

    best = {"score": float(base), "dx": 0, "dy": 0}
    for k in range(1, max(1, int(near_max / max(1, near_px))) + 1):
        r = k * near_px
        found = False
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                if dx == 0 and dy == 0:
                    continue
                s = at(dx, dy)
                if s is not None and s > best["score"]:
                    best = {"score": float(s), "dx": dx, "dy": dy}
                    found = True
        if not found:
            break                      # 这一圈没有更好的 ⇒ 不再往外走（省时间 ✓）
    return best


def crop_compare(frame, canvas, rect):
    """左 = 实时小地图面板，右 = 底图上匹配到的那一块（并排看对不对）。"""
    x0, y0, x1, y1 = rect
    h, w = frame.shape[:2]
    right = np.full((h, w, 3), 40, np.uint8)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if cx1 > cx0 and cy1 > cy0:
        piece = canvas[cy0:cy1, cx0:cx1]
        right[:] = cv2.resize(piece, (w, h), interpolation=cv2.INTER_NEAREST)
    pad = np.full((h, 6, 3), 30, np.uint8)
    return np.hstack([frame, pad, right])


def draw_on_overlay(map_id, rect, out=None):
    """在 `<id>_overlay.png` 上框出「当前是这一块」→ 返回写出的路径（没有叠加图返回 None）。

    这就是"当前小地图是叠加图的哪一块"最直观的答案：直接把框画在那张图上。
    （用了 `--dx/--dy` 微调过叠加图的话，框会差那几个像素。）
    """
    src = mapdata.map_dir() / ("%s_overlay.png" % map_id)
    if not src.exists():
        return None
    img = imread(str(src))
    if img is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in rect)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 3)
    org = (max(4, x0 + 6), max(26, y0 - 10))
    cv2.putText(img, "live view", org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(img, "live view", org, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2)
    dst = Path(out) if out else (mapdata.map_dir() / ("%s_overlay_live.png" % map_id))
    imwrite(dst, img)
    return dst


# ══════════════════════════════════════════════════════════════
# 诊断（B 机跑一次，把面板显示方式量出来）
# ══════════════════════════════════════════════════════════════

def _desc(calib):
    sx, sy = scales_of(calib)
    s = ("scale=%.3f" % sx if abs(sy - sx) < 1e-9
         else "scale x=%.3f y=%.3f" % (sx, sy))     # 两轴不同时必须说出来（不然像量过一样）
    if calib["mode"] == MODE_FIT:
        return "fit  %s offset=%s" % (s, calib["offset"])
    return "crop %s view=%s" % (s, calib["view"])


def _mode_zh(mode):
    """内部值 → 界面上的说法（和「路线识别」下拉里一致）。"""
    return {MODE_FIT: "全局小地图", MODE_CROP: "局部小地图"}.get(mode, str(mode))


def diagnose(map_id, host, port=5003, wait=8.0, out=None, mode="saved", save=True):
    t = mapdata.load(map_id, with_canvas=True)
    if t is None or t.canvas is None:
        print("没有 %s 的地形/底图，先导出："
              "WzProbe.exe dump-terrain <WZ目录> datasets/map --only %s"
              % (map_id, map_id))
        return 2

    # **方式由用户自己在工作台里选**（每张图一份，见 datasets/map/<id>.mapcalib.json）：
    # 默认（mode="saved"）就读他选的那个，**不按匹配分自动挑**（项目规范）。
    # 放在连流之前：没选就直接告诉他去选，不必先等一帧。
    cal = mapdata.load_calib(map_id) or {}
    use = cal.get("mode") if mode == "saved" else mode
    if not use:
        print("这张图还没选显示方式 —— 到「路线识别 → 小地图定位」的「显示方式」里选")
        print("（全局小地图 / 局部小地图），选完这里就按选的那个跑。")
        return 2

    cli = MiniMapClient(host, port).start()
    print("连 %s:%s …（等最多 %.0f 秒）" % (host, port, wait))
    frame = None
    t0 = time.time()
    while time.time() - t0 < wait:
        f, _ = cli.latest()
        if f is not None:
            frame = f
            break
        time.sleep(0.2)
    if frame is None:
        print("没收到帧。%s" % (cli.err or "检查 A 机是否在跑 minimap_push、"
                                          "端口是否放行"))
        cli.stop()
        return 1

    fh, fw = frame.shape[:2]
    print("收到一帧 %dx%d（%.1f KB）" % (fw, fh, frame.nbytes / 1024.0))

    # 方式已经在连流之前定好了（use）；匹配分只用来判断"这一次定位可不可信"。
    loc = locate(frame, t.canvas, mode=use)
    if loc is None:
        print("按你选的「%s」没对上 —— 可能：" % _mode_zh(use))
        print("  · A 机框的区域不只是小地图（框多了/框错了）")
        print("  · 推的不是这张图（现在是 %s）" % map_id)
        print("  · 面板被游戏 UI 挡住了一部分")
        # 两种方式的分数只当**诊断线索**（不是让分数决定方式；改方式去工作台改）
        print("  参考：", end="")
        for tag, r in (("全局小地图", locate_fit(frame, t.canvas)),
                       ("局部小地图", locate_crop(frame, t.canvas))):
            print(" %s=%s" % (tag, "没对上" if r is None
                            else "%.3f" % r["score"]), end="")
        print("（方式不在这里改 —— 到「路线识别」里换）")
        cli.stop()
        return 1

    sx, sy = scales_of(loc)          # 模板匹配只有**一个**尺度 ⇒ 这里两轴必然相同 ✓
    s = sx
    ox, oy = loc["offset"]
    cw, ch = t.canvas.shape[1], t.canvas.shape[0]
    print("采用: %s  匹配分=%.3f" % (_desc(loc), loc["score"]))
    if loc["mode"] == MODE_FIT:
        print("  底图 %dx%d → 面板里 %dx%d，位置 (%d,%d)"
              % (cw, ch, int(cw * sx), int(ch * sy), ox, oy))
        print("  换算：底图像素 = (面板像素 - 偏移) / %.3f" % s)
    else:
        print("  面板 1:1 显示底图的一块，当前那块从底图的 %s 开始" % (loc["view"],))
        print("  换算：底图像素 = 面板像素 - 偏移 + %s" % (loc["view"],))
    print("  世界坐标 = 底图像素 * %.2f - (%s, %s)"
          % (t.px_per_world, t.mini.get("centerX"), t.mini.get("centerY")))

    # 角点对照：面板里底图的四角 → 世界坐标（应当等于世界范围四角）
    b = t.bounds
    for name, (px, py) in (("左上", (ox, oy)), ("右下", (ox + cw * s, oy + ch * s))):
        wx, wy = panel_to_world(px, py, loc, t)
        print("  面板%s角 → 世界 (%.0f, %.0f)" % (name, wx, wy))
    print("  世界范围应为 %s" % (tuple(round(v) for v in b),))

    if save:
        # 只把**量出来的几何**存进去，**方式保持你选的那个**（不覆盖）。
        # ⚠ 走 `with_scales`：这个工具量不出两轴差别（模板匹配只有一个尺度）⇒ 存完
        #   两轴视为相同、顺手清掉旧的 `scale_y`（留着就是一份**改过的**几何里混着
        #   上一轮的 y 缩放 ✗）。
        cal = with_scales(cal, sx, sy)
        cal.update({k: loc[k] for k in ("offset", "view", "score")})
        cal["mode"] = loc["mode"]
        mapdata.save_calib(map_id, cal)
        print("已保存标定: %s（显示方式 = %s，你在 GUI 里选的那个）"
              % (mapdata.calib_path(map_id), loc["mode"]))

    p = Path(out) if out else (mapdata.map_dir() / "_minimap_live.png")
    if loc["mode"] == MODE_CROP:
        # **crop（局部小地图）下"现在显示的是底图哪一块"就是 view** ——
        # 原来这里也画绿框，但画的是"整张底图铺在面板上"，对 crop 是错的
        # （底图比面板大，框会整个跑到画面外）。改成两张真正看得懂的对照图。
        rect_c, rect_o = view_rects(loc, (frame.shape[1], frame.shape[0]), (cw, ch))
        print("  底图上那一块: %s  （面板 %dx%d）"
              % (tuple(rect_c[:2]), frame.shape[1], frame.shape[0]))
        print("  在叠加图上大致是: %s   （底图放大 %d 倍；用过 --dx/--dy 要自己加）"
              % (tuple(int(v) for v in rect_o), overlay_zoom(cw)))
        imwrite(p, crop_compare(frame, t.canvas, rect_c))
        print("已写出 %s（左 = 实时面板，右 = 底图上匹配到的那一块）" % p)
        op = draw_on_overlay(map_id, rect_o)
        if op:
            print("已写出 %s（叠加图上框出「当前是这一块」）" % op)
        else:
            print("（还没有 %s_overlay.png —— 在工作台「路线识别」点一下"
                  "「生成地形图」就有了）" % map_id)
    else:
        vis = frame.copy()
        cv2.rectangle(vis, (ox, oy), (int(ox + cw * s), int(oy + ch * s)),
                      (0, 255, 0), 2)
        cv2.putText(vis, "scale=%.3f off=(%d,%d) score=%.2f"
                    % (s, ox, oy, loc["score"]),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        imwrite(p, vis)
        print("已写出 %s（绿框 = 匹配到的底图，拿去和游戏里的小地图对照）" % p)
    cli.stop()
    return 0


def watch(map_id, host, port=5003, mode=None, seconds=0.0, fps=5.0,
          show=True, out=None):
    """连续看「面板 ↔ 底图」的对应关系（crop / 局部小地图的主力验证手段）。

    每拍：定位一次 → 打印 view / 匹配分 / 与上一拍的**位移** → 更新两张对照图：

        datasets/map/_minimap_live.png        左 = 实时面板，右 = 底图那一块
        datasets/map/<id>_overlay_live.png    叠加图上框出「当前是这一块」

    **为什么非得连续看**：crop 下"现在显示的是底图的哪一块"完全由 matchTemplate
    匹配出来（`view`），单帧结果对错是看不出来的。走两步再看才有判据：
      · 往右走 → view 应该**朝同一个方向**平移（地形图跟着人滑动）；
      · 匹配分不该掉（掉了说明面板被 UI 挡住 / 换图了 / 选错方式）；
      · 位移量级要和走的距离相符（`view` 的单位就是底图像素）。
    按 q（或 Ctrl+C）退出。
    """
    from core.config import get

    t = mapdata.load(map_id, with_canvas=True)
    if t is None or t.canvas is None:
        print("没有 %s 的地形/底图，先导出：" % map_id)
        print("  WzProbe.exe dump-terrain <WZ目录> datasets/map --only %s" % map_id)
        return 2

    # 方式沿用**你在工作台里选的那个**（不按分数自动挑，见项目规范）；
    # 没选就让人先去选 —— 这个工具不该替他决定。
    if mode in (None, "", "saved"):
        mode = (mapdata.load_calib(map_id) or {}).get("mode")
    if not mode:
        print("这张图还没选显示方式 —— 到「路线识别 → 小地图定位」里选"
              "（全局小地图 / 局部小地图）")
        return 2
    if mode != MODE_CROP:
        print("这张图选的是「%s」—— --watch 是给**局部小地图（crop）**看 view 用的；"
              % _mode_zh(mode))
        print("全局小地图整张都显示，没有「现在是哪一块」这回事。")
        return 2

    host = host or get("a_host")
    cli = MiniMapClient(host, port).start()
    print("连 %s:%s …  方式=%s  %.0f 拍/秒  （q 退出）" % (host, port, mode, fps))
    print("  判据：往右走 → view 也朝右平移；匹配分不掉；位移量级和走的距离相符")

    cw, ch = t.canvas.shape[1], t.canvas.shape[0]
    prev = None                      # 上一拍的底图块左上角
    last_frame = None
    t0 = time.time()
    n = 0
    try:
        while True:
            if seconds and time.time() - t0 >= seconds:
                break
            frame, _ts = cli.latest()
            if frame is None or frame is last_frame:
                time.sleep(0.05)
                continue
            last_frame = frame

            loc = locate(frame, t.canvas, mode=mode)
            if loc is None:
                print("  没对上（面板被挡住？换图了？方式选错？）")
                time.sleep(1.0 / fps)
                continue
            if loc["mode"] != MODE_CROP:
                print("  ⚠ 这次定位更像「%s」—— 方式要改去「路线识别」里改，"
                      "这里不自动换" % _mode_zh(loc["mode"]))
                break

            rect_c, rect_o = view_rects(loc, (frame.shape[1], frame.shape[0]),
                                        (cw, ch))
            mv = ""
            if prev is not None:
                mv = "  位移 (%+d, %+d)" % (rect_c[0] - prev[0], rect_c[1] - prev[1])
            prev = rect_c[:2]
            n += 1
            print("  [%3d] view=%-16s 匹配分=%.3f  底图块起于 %-14s%s"
                  % (n, tuple(loc["view"]), loc["score"], rect_c[:2], mv))

            imwrite(out or (mapdata.map_dir() / "_minimap_live.png"),
                    crop_compare(frame, t.canvas, rect_c))
            draw_on_overlay(map_id, rect_o)

            if show:
                try:
                    vis = crop_compare(frame, t.canvas, rect_c)
                    cv2.imshow("minimap live (left=panel right=base map)",
                               cv2.resize(vis, None, fx=0.75, fy=0.75))
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                except Exception as e:
                    show = False      # 没有 GUI 后端（headless）→ 只写文件
                    print("  （打不开预览窗口：%s；每拍的对照图仍在写 "
                          "_minimap_live.png）" % e)
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    finally:
        cli.stop()
        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    print("退出（共 %d 拍）" % n)
    return 0


def main() -> int:
    import argparse
    from core.config import get
    ap = argparse.ArgumentParser(description="小地图定位诊断（B 机）")
    ap.add_argument("--map", dest="map_id", default="105090600")
    ap.add_argument("--host", default=None, help="A 机 IP，默认读 link.yaml 的 a_host")
    ap.add_argument("--port", type=int, default=None,
                    help="A 机小地图推流端口，默认读 link.yaml 的 minimap.port")
    ap.add_argument("--wait", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=("saved", MODE_FIT, MODE_CROP),
                    default="saved",
                    help="saved（默认）= 用你在工作台「路线识别 → 小地图定位」里"
                         "选的那个；fit / crop = 本次临时按这个跑（方式由你定，"
                         "程序不按匹配分自动挑）")
    ap.add_argument("--no-save", dest="save", action="store_false",
                    help="不写标定文件（只想看看时用）")
    ap.add_argument("--watch", type=float, default=0.0, metavar="秒",
                    help="连续确认模式：跑这么多秒（0 = 一直跑到按 q / Ctrl+C）。"
                         "crop（局部小地图）用它边走边看 view 对不对")
    ap.add_argument("--fps", type=float, default=5.0, help="--watch 的刷新率")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="--watch 时不开预览窗口（只写对照图）")
    a = ap.parse_args()
    # 端口和 host 一样，默认从**双机共用**的 link.yaml 读 —— A 机部署台那张
    # 「小地图推流」卡片也是照它对（自检里会比对，不一致当场说）。
    port = a.port or int(get("minimap", "port", 5003))
    if a.watch:
        # watch 自己会去读工作台里选的那个（mode=None → 读标定文件）
        return watch(a.map_id, a.host or get("a_host"), port,
                     None if a.mode == "saved" else a.mode,
                     a.watch, a.fps, a.show, a.out)
    return diagnose(a.map_id, a.host or get("a_host"), port, a.wait, a.out,
                    a.mode, a.save)


if __name__ == "__main__":
    raise SystemExit(main())
