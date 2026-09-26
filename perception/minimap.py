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
        # 假数字（真实推流是 10 fps）。
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
#: 来源的中文名 —— 界面、提示、报告共用这一份，别一处写「独立推流」另一处写
#: 「收流」：标定是按来源分开存的，名字对不上就没人搞得清哪份是哪份。
SRC_LABEL = {SRC_STREAM: "独立推流", SRC_LIVE: "从实时画面"}

#: 标定 dict 的形状（存 datasets/map/<id>.mapcalib.json）
#:   mode/scale/offset  —— 面板像素 → 底图像素：canvas = (panel - offset) / scale
#:   view               —— crop 模式专用：当前显示的是底图从 (view) 开始的那一块
#:   score              —— 标定时匹配得有多好（>0.8 才可信）


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


def locate_fit(frame, canvas, scales=None, min_score=0.55):
    """方式 1：整张底图缩放到面板 → 标定 dict 或 None。"""
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
        if s <= 0:
            continue
        tw, th = int(round(cg.shape[1] * s)), int(round(cg.shape[0] * s))
        if tw < 8 or th < 8 or tw > fg.shape[1] or th > fg.shape[0]:
            continue
        t = cv2.resize(cg, (tw, th), interpolation=cv2.INTER_AREA)
        r = np.nan_to_num(cv2.matchTemplate(fg, t, cv2.TM_CCOEFF_NORMED))
        _, mx, _, ml = cv2.minMaxLoc(r)
        if best is None or mx > best["score"]:
            best = {"mode": MODE_FIT, "scale": float(s),
                    "offset": [int(ml[0]), int(ml[1])], "score": float(mx),
                    "view": [0, 0]}
    if best is None or best["score"] < min_score:
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


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def panel_to_canvas(px, py, calib):
    """面板像素 → 底图像素（calib 来自 locate / 标定文件）。"""
    s = float(calib.get("scale") or 1.0)
    ox, oy = calib.get("offset") or (0, 0)
    cx, cy = (px - ox) / s, (py - oy) / s
    if calib.get("mode") == MODE_CROP:
        vx, vy = calib.get("view") or (0, 0)
        cx, cy = cx + vx, cy + vy
    return cx, cy


def panel_to_world(px, py, calib, terrain):
    """面板像素 → **世界坐标**。"""
    return terrain.canvas_to_world(*panel_to_canvas(px, py, calib))


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
    s = float(calib.get("scale") or 1.0)
    ox, oy = calib.get("offset") or (0, 0)
    ch, cw = cv_.shape[:2]
    # ⚠ 参数是 numpy 的 `.shape`（**高在前**）：按 (宽, 高) 解会把宽高弄反，
    # 搜索结果区被裁成正方形的一角 —— 实测表现是「底图相减明明减掉了，却找不到点」。
    h, w = panel_shape[:2]
    x0, y0 = max(0, int(ox)), max(0, int(oy))
    x1, y1 = min(w, int(ox + cw * s)), min(h, int(oy + ch * s))
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
    s = float(calib.get("scale") or 1.0)
    ox, oy = calib.get("offset") or (0, 0)
    vx, vy = (calib.get("view") or (0, 0)) if calib.get("mode") == MODE_CROP else (0, 0)
    h, w = panel.shape[:2]
    # canvas = (panel - offset)/scale + view  ⇒  panel = (canvas - view)*scale + offset
    mat = np.float32([[s, 0, ox - vx * s], [0, s, oy - vy * s]])
    warped = cv2.warpAffine(terrain.canvas, mat, (w, h),
                            flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
    base = dot_mask(warped, family)
    k = max(3, int(round(s * 2))) | 1        # 奇数核
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

    def _world_offset(self, calib):
        """「坐标系偏移」(x, y)：算出来的世界坐标**加上**它（用户 2026-09-26 要求）。

        为什么要它：坐标是按**黄点重心**算的（见 `find_player_dot` 的 `x, y`），而
        "黄点重心 ↔ 游戏里的玩家原点"这层对应是游戏自己定的 —— 与其猜（按中心还是
        按脚下），不如给一个显式偏移让人自己量一次。读数、段号、foothold、绳梯判据
        都在**加完之后**算（见 `update`），所以寻路的"我在哪块平台上"也跟着准。

        **按地图 id 存**（用户 2026-09-26 定的口径）：它跟标定走 —— 存在
        `datasets/map/<id>.mapcalib.json` 的 `sources.<来源>.world_offset`，
        和 `scale/offset/view/alpha` 同一份（每张图、每个来源各自一套 ✓）。
        直接读 `calib`（调用方已经按来源取好了），所以面板和实时线程天然一致。
        """
        v = (calib or {}).get("world_offset") or []
        if len(v) == 2:
            try:
                return (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                pass
        return (0.0, 0.0)

    def update(self, panel, src=None, calib=None, terrain=None):
        """面板画面（BGR）→ 定位结论 dict。

        返回：
            ok         这一拍**拿到了世界坐标**（认不出点 / 没标定 → False）
            confirmed  黄点是"连续两拍都在附近"确认过的（决策层该只认这种）
            px, py     黄点在面板里的像素（重心；认不出时是上一次的残值，别用）
            world_x/world_y  世界坐标；没算出来是 None
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

        wx, wy = panel_to_world(r["x"], r["y"], calib, terrain)
        # 「坐标系偏移」：**在这一步加**（用户自己精确标定用）—— 后面的范围检查、
        # 段号、foothold、绳梯判据全都用加完之后的坐标，口径才是一份。
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
        seg = terrain.segment_of(wx, wy)
        out["segment_id"] = seg.index if seg is not None else None
        # 脚下**那条** foothold 的 id（字符串，和 zones 文件里的写法一致）。
        # 与 segment_of 是同一套口径（都走 foothold_below），多算一次可忽略。
        f = terrain.foothold_below(wx, wy)
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
    s = float(loc.get("scale") or 1.0)
    inset = float((loc.get("offset") or [4, 4])[0])
    vx, vy = loc.get("view") or (0, 0)
    x0, y0 = vx - inset / s, vy - inset / s
    x1, y1 = x0 + pw / s, y0 + ph / s
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
    s = float(loc.get("scale") or 1.0)
    ox, oy = loc.get("offset") or (0, 0)
    return None, (int(ox), int(oy),
                  int(round(cw * s)), int(round(ch * s)))


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
            loc2 = dict(loc)
            loc2["scale"] = (float(loc.get("scale") or 1.0) / z)
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
    if calib["mode"] == MODE_FIT:
        return "fit  scale=%.3f offset=%s" % (calib["scale"], calib["offset"])
    return "crop view=%s" % (calib["view"],)


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

    s = float(loc["scale"])
    ox, oy = loc["offset"]
    cw, ch = t.canvas.shape[1], t.canvas.shape[0]
    print("采用: %s  匹配分=%.3f" % (_desc(loc), loc["score"]))
    if loc["mode"] == MODE_FIT:
        print("  底图 %dx%d → 面板里 %dx%d，位置 (%d,%d)"
              % (cw, ch, int(cw * s), int(ch * s), ox, oy))
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
        cal.update({k: loc[k] for k in ("scale", "offset", "view", "score")})
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
