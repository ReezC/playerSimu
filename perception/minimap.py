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
        # 假数字（默认推 30 fps，见 tools.minimap_push.DEFAULTS）。
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


def locate_fit(frame, canvas, scales=None, min_score=0.55, refine=True):
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


def locate_crop(frame, canvas, inset=4, min_score=0.55, scales=None):
    """方式 2：面板是底图的**一块**（可能先放大了若干倍）→ 标定 dict 或 None。

    **为什么不能只按 1:1 找**（实测踩到）：客户端把小地图放大后再切块很常见。
    猴林迷宫I 就是：底图 78×203，而面板 753×612 —— 光宽度就差 9.7 倍。这种图上
    「整张缩放进面板」（fit）和「1:1 取一块」**两种都量不出来**，人手动把叠加层
    拖到 4 倍也不够（缩放上限），表现就是「匹配分 0.7 上下、怎么看都不对」。

    **做法**：对每个候选放大倍数 s，把**面板按 1/s 缩回去**（缩回底图尺度），
    再在**原尺寸底图**里找它 —— 一次 matchTemplate 得到「显示的是底图哪一块」。

    缩回去有两种实现，**按 s 分开用**（这里踩过一个静默的坑）：
      · s ≥ 1（客户端把地图**放大**了，最常见）：**抽样**（`[::step]`），不插值 ——
        客户端放大就是复制像素，抽样正好把它还原回去，计算量还降到 1/s²；
      · s < 1（客户端把地图**缩小**着显示）：只能**插值**（`cv2.resize`）。
        ⚠ 以前不分这两种：`step = max(1, round(0.54)) = 1` → 模板其实还是 1:1 那张，
        分数照样 1.0，却把 scale 记成 0.54 —— view 是对的、scale 是错的，
        换算整体偏 1/s 倍（`t_crop_roundtrip` 系的一条回归逮到过）。
        另外 `eff` 一律取**真正实现出来的倍数**（抽样就是 step，插值就按实际输出尺寸算），
        不写"想要的那个 s"。

    于是标定里同时有 `scale`（面板像素 / 底图像素）和 `view`（面板左上角对应的
    底图坐标），换算仍是 `canvas = (panel - offset) / scale + view`。
    """
    f = _gray(frame)
    c = _gray(canvas)
    h, w = f.shape[:2]
    # 面板可能有边框/外圈，用中间部分当模板更稳（分数不会被边框拉低）
    t = f[inset:h - inset, inset:w - inset] if (h > 2 * inset and w > 2 * inset) else f

    if scales is None:
        scales = _crop_scales((c.shape[1], c.shape[0]), (w, h))

    best = None
    for s in scales:
        if s <= 0:
            continue
        if s < 1.0:
            # 缩小：抽样做不到（step 最小是 1），必须插值。
            # ⚠ 别在这里偷懒用 step=1 顶替 —— 那样模板是 1:1 的、分数满分，
            # 却把 scale 记成 s，等于给出一份"看着很准其实偏一倍"的换算。
            tt = cv2.resize(t, (max(1, int(round(t.shape[1] * s))),
                                max(1, int(round(t.shape[0] * s)))),
                            interpolation=cv2.INTER_AREA)
            eff = t.shape[1] / float(tt.shape[1])      # 真正实现出来的倍数
        else:
            step = max(1, int(round(s)))
            if abs(s - step) > 0.3:
                continue        # 抽样实现不出这个倍数（相邻整数档已经在候选里了）
            tt = t[::step, ::step] if step > 1 else t
            eff = float(step)
        if tt.shape[0] < 6 or tt.shape[1] < 6:
            continue
        if tt.shape[0] > c.shape[0] or tt.shape[1] > c.shape[1]:
            continue                    # 缩回去还是比底图大 → 不是这个倍数
        r = np.nan_to_num(cv2.matchTemplate(c, tt, cv2.TM_CCOEFF_NORMED))
        _, mx, _, ml = cv2.minMaxLoc(r)
        if best is None or mx > best["score"]:
            # view 是「面板左上角(去掉 inset 之后)对应的底图坐标」——
            # 匹配位置本身就是这个坐标，所以**不要再按 s 折算**（换算里已经除了 s）。
            best = {"mode": MODE_CROP, "scale": eff,
                    "offset": [inset, inset],
                    "view": [int(ml[0]), int(ml[1])], "score": float(mx)}
    if best is None or best["score"] < min_score:
        return None
    return best


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


def check_calib(panel, terrain, calib, mode=None, min_score=0.55,
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
    from tools.config import get                    # 本模块只在用到处局部 import ✓
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
        return self

    def calib_for(self, src, ttl=1.0):
        """读某条来源的标定，**带 1 秒缓存**。

        为什么不每次读文件：实时回路 30 拍/秒，每拍读一次 JSON 纯属浪费；
        为什么不只在启动时读一次：标定在弹窗里随时会被改，读死了要重启才生效。
        1 秒的折中足够（弹窗保存后本来就会自己刷新一次）。
        """
        now = time.monotonic()
        hit = self._calib_cache.get(src)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
        cal = mapdata.load_calib(self.map_id, src) if self.map_id else None
        self._calib_cache[src] = (now, cal)
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
        r = self.tracker.update(panel, calib=calib, terrain=terrain)
        out = {"ok": False, "confirmed": bool(r.get("confirmed")),
               # held：这一拍没认出黄点，位置是**上一帧**的（防抖窗口内沿用）。
               # 决策层可以用（两三拍之内还算得准），但要能分辨出来。
               "held": bool(r.get("held")), "missed": int(r.get("missed") or 0),
               "px": r["x"], "py": r["y"], "world_x": None, "world_y": None,
               "segment_id": None, "foothold_id": None, "src": src,
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
    """
    wx0 = getattr(player, "world_x", None)
    wy0 = getattr(player, "world_y", None)
    if wx0 is None or wy0 is None:
        return None
    cam_x = float(wx0) - float(getattr(player, "x", 0.0) or 0.0)
    # ⭐ **脚底偏移**（用户 2026-09-28 ✓）：框底不一定正好压在脚底（鞋底阴影 / 披风 /
    #   特效会让框多出一截 ✗）⇒ 用镜像值补正 ✓（`live_thread` 每帧灌进 `world_state` ✓
    #   —— 这边**不能**反过来 import decision，会绕成循环依赖 ✗）。
    from perception import world_state as _ws
    cam_y = (float(wy0) - (float(getattr(player, "bottom", 0.0) or 0.0)
                           + float(_ws.FOOT_OFFSET_PX)))
    return (float(sx) + cam_x, float(sy) + cam_y)


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
            loc2["view"] = [float(v) / z for v in (loc.get("view") or (0, 0))]
    src, (dx, dy, dw, dh) = overlay_draw_rects(loc2, (fw, fh), canvas_wh)
    return src, (fx + dx, fy + dy, dw, dh)


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
    from tools.config import get
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
    from tools.config import get

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
    from tools.config import get
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
