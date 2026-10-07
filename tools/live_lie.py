# -*- coding: utf-8 -*-
"""实时测谎测试（**独立工具窗** ✓ 用户 2026-10-03 ✓ 原话："能不能做个**实时模式测试**，用**框选
窗口的模式**（参考数据工作台框选相关设计）" ✓ ＋ "落点 = **独立工具窗口**" ✓ ＋ "**不用选择素材**，
我的需求是**实时计算**，类似数据工作台的**框选本地屏幕区域**或者**收流**" ✓）。

它干什么 ✗✗：**接实时画面**（本地屏幕的一块 / 收流 ✓）⇒ 每一拍**现抓现算**（不是"整段预演"✗），
而且**只在框选出来的那一块**里跑测谎 ✓ —— 现场要测的往往只是屏幕上一小块 ✓，全图跑既慢又容易被
别的东西带偏 ✗。

为什么**不碰主链路** ✓（用户选的"独立工具窗口" ✓）：实时面板那条链（`LivePanel` / `LiveThread`）
**一行不改** ✗ ⇒ 零风险 ✓ 可以慢慢验 ✓。

三种"实时源"（`FrameSource` ✓ 纯逻辑 ✓ 好测 ✓）：
  · **`ScreenSource`** —— **本地屏幕的一块矩形**（用户点名的那种 ✓）：框选 = `gui.region_selector`
    的**抓屏框选**（带放大镜 ✓ docs/UI规范.md §8 ✓）⇒ 每拍 `core.wincap.grab_rect` 抓那块 ✓；
  · **`StreamSource`** —— **收流**（用户点名的那种 ✓）：`link.PyAVSource`（udp/srt/rtsp ✓）⇒
    收来整帧、按 ROI 裁一块 ✓；
  · **`VideoSource`** —— 录像当实时（⚠ 界面上**不再暴露** ✗ 用户明说"不用选择素材" ✓ —— 只留给
    **自检 / 命令行调试** ✓ 它也是唯一能在离屏、无流、无屏幕的环境里被完整测到的源 ✓）。
⇒ `grab()` 一律回"**能给 tracker 吃的那块图**"，下游不用关心它是哪来的 ✓（也不用再裁 ✗）。

⚠ **不跑检测器**（`dets=None` ✓）：测谎的核心是"**目标本身的运动**" ✓ —— 先把"实时 + 框选 +
  测谎"这条主线打通 ✓（要接检测器再说 ✗）。
⚠ **尺寸**：屏幕/流抓来是**原始分辨率**（1080p+ ✗）而 `LieTracker` 是 CPU 图像处理 ⇒ 直接喂会慢 ✗
  ⇒ 引擎里先**缩到加工尺度**（宽 ≤ `PROC_W` ✓ 与 `lie_demo` 同一量级 ✓）再跑 ✓，显示也用这一份 ✓。
⚠ 参数**跟着 `lie_demo` 那份界面配置走**（`load_lie_cfg` ✓ 在演示窗里调好的那套直接生效 ✓）。

用法（项目根目录；界面上一律走「`实时测谎.bat`」）：
    python -m tools.live_lie                                  # 空窗：选来源 → 框选 → 开始
    python -m tools.live_lie --stream udp://0.0.0.0:5000      # 直接收流
    python -m tools.live_lie --video <素材> --rect 100,60,300,240   # 调试/自检用（非实时）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lie_demo import Runner, draw, load_source          # noqa: E402

#: **加工尺度**：喂给 `LieTracker` 的那份图，宽度缩到不超过它（高度按比例 ✓）。
#: ⚠ 为什么必须缩 ✗✗：屏幕/流抓来是 1080p/2K，而 `LieTracker` 是**纯 CPU** 的图像处理
#:   （白块 / 背景 / 直方图 …… ✓）⇒ 直接喂动辄几十 ms/拍 ⇒ "实时"就没了 ✗；`lie_demo` 那条路
#:   本来就在这个量级上跑（`PROC_H` ✓）⇒ 用同一档最稳 ✓。
PROC_W = 750
#: **ROI / 屏幕矩形的最小边**（px ✓）：几个像素的"区域"喂给追踪器没有意义 ✗（白块都放不下 ✓）
#: —— 比 `region_picker` 那个"< 4px 当误点"的门大得多 ✓（那个管"手抖" ✓ 这个管"够不够用" ✓）。
MIN_SIDE = 48
#: 显示刷新上限（fps ✓）：**追踪每帧都跑** ✓，只是**画面**别刷太猛（每帧重建 QPixmap 很贵 ✗）。
SHOW_FPS = 30.0
#: `Runner`（= `LieTracker` + 控制器 ✓）能吃的参数名 ✓ —— 从 `lie_demo` 那份界面配置里挑这些 ✓
#: （键名与 `Runner.__init__` 一致 ✓ 对不上的自然落空 ⇒ 用追踪器默认 ✓ 不硬编 ✗）。
_LIE_CFG_KEYS = ("path_ms", "merge_iou", "merge_iou_out", "sep_ratio", "ring_cov_sep",
                 "brick_iou_sep", "ring_cov_fuse", "brick_iou_fuse", "inherit_dist",
                 "allow_back_ratio", "edge_max", "sep_vel_max_ratio", "noise_tol",
                 "board_s", "board_overlap", "follow_gain", "iou_brick")
#: 收流地址的默认提示（用户在 `live_panel` 那边也是这个写法 ✓）
DEFAULT_URL = "udp://0.0.0.0:5000"


# ══════════════════════════════════════════════════════════
# 纯逻辑（不 import Qt ✓ 自检直接钉 ✓）
# ══════════════════════════════════════════════════════════

def clamp_rect(rect, w, h):
    """把矩形**夹进** `w×h` 的画面/屏幕并取整 ⇒ `(x, y, w, h)`（尺寸为 0 / 退化 / 太小 ⇒ `None` ✓）。

    ⚠ **屏幕矩形**和**画面里的 ROI** 都走它 ✓（两处的"夹"是一个意思 ✓ 别各写一份 ✗）：
      · 屏幕那条：`w/h` 传**虚拟屏幕**的宽高 ✓（多显示器时 x/y 可能为负 ✓ 夹完照旧可能为负 ✓
        —— 那是对的 ✓ `grab_rect` 就吃这套坐标 ✓）；
      · 画面那条：`w/h` 传这一帧的宽高 ✓。
    ⚠ **太小**（任一边 < `MIN_SIDE`）⇒ `None` ✓：那点地方连白块都放不下 ⇒ 硬跑只会白算 ✗。
    """
    if rect is None:
        return None
    try:
        _x, _y, _rw, _rh = (int(round(float(v))) for v in tuple(rect)[:4])
    except (TypeError, ValueError):
        return None
    _w, _h = int(w or 0), int(h or 0)
    if _w <= 0 or _h <= 0:
        return None
    _x = max(-_w, min(_x, _w))
    _y = max(-_h, min(_y, _h))
    _rw = min(_rw, _w - max(0, _x))
    _rh = min(_rh, _h - max(0, _y))
    if _rw < MIN_SIDE or _rh < MIN_SIDE:
        return None
    return (_x, _y, _rw, _rh)


def crop_roi(frame, rect):
    """按 `rect` 从**画面**里裁一块（`rect` 应当已经过 `clamp_rect` ✓）；裁不出东西 ⇒ `None` ✓。"""
    if frame is None or rect is None:
        return None
    _x, _y, _w, _h = rect
    if _x < 0 or _y < 0:
        return None                       # 画面里的 ROI 不该有负起点（那是屏幕坐标的事 ✓）
    _f = frame[_y:_y + _h, _x:_x + _w]
    return _f if _f is not None and _f.size else None


def fit_proc(frame, proc_w=PROC_W):
    """把任意分辨率的图**缩到加工尺度**（宽 ≤ `proc_w` ✓ 高度按比例 ✓ 已经够小 ⇒ 原样 ✓）。

    ⚠ 只**缩小**不放大 ✓（放大既慢又没有信息 ✓）；太小（宽 < 2×`MIN_SIDE`）也**不动** ✓
      —— 硬放大会把插值噪声喂给追踪器 ✗。
    """
    if frame is None:
        return None
    _h, _w = frame.shape[:2]
    if _w <= 0 or _h <= 0:
        return None
    if _w <= _LIMIT_W or _w < 2 * MIN_SIDE:
        return frame
    _k = _LIMIT_W / float(_w)
    return cv2.resize(frame, (_LIMIT_W, max(MIN_SIDE, int(round(_h * _k)))),
                      interpolation=cv2.INTER_AREA)


class LiveLieEngine:
    """一块画面 → （缩到加工尺度 →）`Runner`（`LieTracker` + 控制器 + 光标 ✓）→ 结果 ✓。

    · `reset()` ⇒ **新一局**：状态清零 ✓（换来源 / 重开都要 ✓）—— `Runner` 的 `assume`（控制器 /
      光标的**初始位置**）用**加工尺度图的中心** ✓（给错了初始光标会先"飞"过去 ⇒ 头几帧的命中率
      全是假的 ✗）；
    · `step(frame, ts)` ⇒ 回 `(o, pos, rad, hit, boxes, motion)` ✓（**坐标在加工尺度图上** ✓，
      与 `draw(..., scale)` 那套直接对得上 ✓）；没建起来 / 尺寸不行 ⇒ `None` ✓。
    ⚠ **惰性建**（第一帧才知道加工尺度的实际高宽 ✓）：所以 `reset()` 之后要先喂一帧 ✓。
    """

    def __init__(self, cfg=None, gain=None, proc_w=PROC_W, detector=None, mode="motion"):
        self._cfg = dict(cfg or {})
        self.gain = gain
        self.proc_w = int(proc_w or PROC_W)
        #: ⭐⭐⭐⭐⭐ **走哪条管线**（用户 2026-10-05 ✓ 原话："**把我们当前的进展做进实时测谎.bat
        #:   用于实时验证**" ✓✓）——
        #:     · `"motion"`（**默认** ✓）= `MotionRunner`（运动分离 ✓ 我们这几轮**所有**改动都在这条
        #:       上：`pred` 由绿圈推 / 夹取 / 夹取增益 / 拆框 / 融合必含真目标 ✓）；
        #:     · `"classic"` = 老的 `Runner`（`LieTracker` ✓ 留着对照 ✓）。
        #:   ⚠ 参数分别取两套 ✓（`load_motion_cfg` / `load_lie_cfg` ✓ 都是**你界面上调好的那些** ✓）。
        self.mode = str(mode or "motion")
        self.mcfg = (dict(load_motion_cfg()) if self.mode == "motion" else {})
        #: ⭐⭐⭐ **检测器**（用户 2026-10-03 ✓ 原话："实时测谎：**没有任何检出框**" ✓）——
        #:   任何有 `.detect(bgr) -> [(cls,cx,cy,w,h,conf), …]` 的东西都行 ✓
        #:   （真货 = `lie_demo.DetsWorker` ✓ 自检里塞**桩** ✓ ⇒ 这条链在离屏也能测 ✓）。
        #:   ⚠ **没有它，测谎就只剩"白块跟一跟"** ✗✗：融合 / 分离 / 砖表 / 点选 …… 全靠**检出框** ✓
        #:     （这是我上一轮"先不跑检测器"那个取舍的代价 ✗ —— 现在接上 ✓）。
        self.detector = detector
        self.runner = None
        self.n = 0
        self.size = None
        self.reset()

    def reset(self):
        """新一局：**状态清零，但检测器留着** ✓（重建模型要好几秒 ✗ 没必要 ✓）。"""
        self.runner = None
        self.n = 0
        self.size = None

    def step(self, frame, ts=None):
        _p = fit_proc(frame, self.proc_w)
        if _p is None or _p.size == 0:
            return None
        if self.runner is None:
            self.size = (_p.shape[1], _p.shape[0])
            if self.mode == "motion":
                # ⚠ `MotionRunner` 与 `Runner` **同签名、同 6 元组** ✓（见它那段注释 ✓）
                #   ⇒ 上面"渲染 / 点选 / 拖动"那一整套**一行都不用改** ✓。
                from perception.lie_motion import MotionRunner
                self.runner = MotionRunner(dets=None, **self.mcfg)
            else:
                self.runner = Runner(dets=None,
                                     assume=(self.size[0] / 2.0, self.size[1] / 2.0),
                                     gain=self.gain, **self._cfg)
        _ts = float(ts) if ts is not None else time.time()
        # ⚠ 检出跑在**加工尺度那份图**上 ✓（与 `lie_demo` / `ab_compare` 那条路**同一把尺** ✓）
        #   ⇒ 检出框坐标就是"加工域"✓，与 tracker / 画面 / 点选**全对得上** ✓
        #   （⚠ 别在原生 1080p 上检：既慢又和"喂给 tracker 的那份"不是同一个坐标 ✗）。
        _raw = None
        if self.detector is not None:
            try:
                _raw = self.detector.detect(_p)
            except Exception:                 # noqa: BLE001 —— 检测失败不该把整条链带死 ✓
                _raw = None
        # ⚠⚠⚠ **两条管线对 `dets` 的要求不一样** ✗✗（**踩过** ✓ 见 `MotionRunner` 那段注释 ✓）：
        #   · 经典 `Runner`：**没有框 ⇒ `None`** ✓（它自己判"本拍没观测" ✓）；
        #   · 运动 `MotionRunner`：**必须是个列表** ✗ —— 它按拍取 `self.dets[i]` ✓
        #     ⇒ 给 `None` 会 `TypeError` ⇒ PyQt5 里就是**闪退** ✗✗
        #     ⇒ 一律塞**单元素列表** ✓（没检出就 `[None]` ✓ 它当"这拍没框" ✓），`i` 恒传 0 ✓。
        if self.mode == "motion":
            self.runner.dets = [_raw]
        else:
            self.runner.dets = [_raw] if _raw else None
        _out = self.runner.step(_p, 0, _ts)
        self.n += 1
        return _out

    @property
    def cursor(self):
        return None if self.runner is None else tuple(self.runner.cursor)


def load_lie_cfg():
    """`lie_demo` 那份界面配置里的**追踪器参数**（用户调好的那套 ✓ 直接生效 ✓）。

    ⚠ 只挑 `_LIE_CFG_KEYS` 里**有值**的 ✓ —— 其余不传 ⇒ 用追踪器自己的默认 ✓（不硬编 ✗）；
      `theme` / 配置文件不在（命令行环境 ✓）⇒ 回空 ✓ 照样能跑（全用默认 ✓）。
    """
    try:
        from gui import theme
        _c = dict(theme.load_section("lie_demo") or {})
    except Exception:                       # noqa: BLE001 —— 读不到配置不算错误 ✓ 用默认就是 ✓
        return {}
    return {k: _c.get(k) for k in _LIE_CFG_KEYS if _c.get(k) is not None}


#: ⭐⭐⭐⭐⭐ **运动分离那条的参数**（用户 2026-10-05 ✓ 原话："**把我们当前的进展做进实时测谎.bat
#:   用于实时验证**" ✓✓）—— 键名与 `lie_demo._motion_kw` **同一套** ✓（`config/ui.yaml` 里
#:   就是 `motion_*` ✓ 别各叫一套 ✗）；左 = 配置键 ✓ 右 = `MotionTracker` 的形参名 ✓。
#:   ⚠ **一个参数都不硬编** ✗：`ui.yaml` 里没有的键 ⇒ 不传 ⇒ 用追踪器自己的默认 ✓
#:     （与 `load_lie_cfg` 同一条纪律 ✓）。
_MOTION_CFG_MAP = (("motion_pair_gate", "pair_gate"),
                   ("motion_score_decay", "score_decay"),
                   ("motion_switch_margin", "switch_margin"),
                   ("motion_min_hits", "min_hits"),
                   ("motion_white_w", "white_w"),
                   ("motion_smooth", "smooth"),
                   ("motion_q_min", "q_min"),
                   ("motion_q_need", "q_need"),
                   ("motion_stuck_ratio", "stuck_ratio"),
                   ("motion_pred_max", "kf_pred_max"),
                   ("motion_dir_n", "dir_n"),
                   ("motion_fuse_pull", "fuse_pull"),
                   ("motion_clamp_gain", "clamp_gain"))
#: ⚠ 这几个是**整数**（其余是浮点 ✓）—— ⚠⚠ 别一律 `float(...)` ✗：`min_hits` / `q_need` /
#:   `dir_n` 收 `float` 也能跑，但**日志/界面上就不是整数**了 ✓（`MotionTracker` 里会 `int()` ✓
#:   所以不算错 ✗；这里写清楚只是别踩"看起来像 4.0 拍"这种困惑 ✓）。
_MOTION_INT_KEYS = ("min_hits", "q_need", "dir_n")


def load_motion_cfg():
    """**运动分离**那条的参数（同 `load_lie_cfg` ✓ 只是键名不同 + 带类型 ✓）。"""
    try:
        from gui import theme
        _c = dict(theme.load_section("lie_demo") or {})
    except Exception:                       # noqa: BLE001 —— 读不到就全用默认 ✓
        return {}
    _out = {}
    for _ck, _pk in _MOTION_CFG_MAP:
        _v = _c.get(_ck)
        if _v is None:
            continue
        try:
            _out[_pk] = (int(float(_v)) if _pk in _MOTION_INT_KEYS else float(_v))
        except (TypeError, ValueError):
            continue
    #   ⭐⭐⭐⭐⭐ **限定区域（ROI ✓ 用户 2026-10-07 ✓ 原话："**只在限定的区域计算（因为这是
    #     部分弹窗）**" ✓✓）** —— ⚠⚠ **它是个 4 元组** ✗ ⇒ **不能进上面那张"标量表"** ✗
    #     （那张表一律 `float(...)` ✓ ⇒ 列表会被打散 / 直接报错 ✓）⇒ 单独读 ✓：
    #     `motion_roi: [x0, y0, x1, y1]`（**加工域**坐标 ✓）；非法（数量不对 / 反了 / 退化）
    #     ⇒ **当没给** ✓（= 整幅 ✓ 与改动前逐位一致 ✓）。
    _roi = _c.get("motion_roi")
    if isinstance(_roi, (list, tuple)) and len(_roi) == 4:
        try:
            _v4 = [float(_x) for _x in _roi]
            if _v4[2] > _v4[0] and _v4[3] > _v4[1]:
                _out["roi"] = tuple(_v4)
        except (TypeError, ValueError):
            pass
    return _out


# ══════════════════════════════════════════════════════════
# 三种"实时源"（纯逻辑 ✓ 只有 Screen/Stream 依赖系统库 ✓ 且**惰性** import ✓）
# ══════════════════════════════════════════════════════════

class FrameSource:
    """取帧的统一口子 ✓。

    `grab()` 回**这一源的原生那一幅**（BGR ✓ 还没裁、没画过任何东西 ✓）；暂时没帧 ⇒ `None` ✓。
    ⚠ **裁 ROI 是调用方的事**（`crop_roi` ✓）—— 源不管 ROI ✗（屏幕那条更特殊：它**抓的时候**
      就已经限定范围了 ✓ 压根没有"整帧"可裁 ✓）。
    `done()` = "这个源**再也不会**有新帧了"（录像放完 ✓）⇒ 调用方据此收尾 ✓（其它源恒 `False` ✓）。
    """

    def grab(self):
        raise NotImplementedError

    def done(self):
        return False

    def close(self):
        pass


class ScreenSource(FrameSource):
    """**本地屏幕的一块矩形**（用户点名的那种 ✓ "类似数据工作台的框选本地屏幕区域" ✓）。

    ⚠ `rect` 是**屏幕绝对坐标**（多显示器时可能为负 ✓ 与 `wincap.virtual_screen` /
      `region_selector.select_region` **同一套** ✓）—— 与"画面里的 ROI"**不是一回事** ✗：
      那个是"抓完在画面里裁" ✓ 这个是"抓之前就限定范围" ✓（更省也更快 ✓）。
    ⚠ `grab_rect` 走 WGC / BitBlt（只在 B 机有 ✓）⇒ 起不来就**如实抛** ✗ 不静默黑屏 ✗
      （`live_thread` 那边也是这个取向 ✓）。
    """

    def __init__(self, rect, use_wgc=True):
        self.rect = tuple(int(v) for v in rect)
        self.use_wgc = bool(use_wgc)

    def grab(self):
        from core import wincap
        return wincap.grab_rect(self.rect, use_wgc=self.use_wgc)


class StreamSource(FrameSource):
    """**收流**（用户点名的那种 ✓）—— `link.PyAVSource`（udp / srt / rtsp ✓）+ 按 ROI 裁一块 ✓。

    ⚠ `read()` 是**阻塞**的；`TimeoutError` = "**暂时**没帧"（udp 超时 ✓ 不是错误 ✓）⇒ 原样抛
      ✓ 调用方 `continue` ✓（照 `live_thread` 的收流回路 ✓ 别把它当失败 ✗）。
    ⚠ 惰性 `open()`（第一次 `grab` 才开 ✓）—— 开流会阻塞（udp 超时 5 秒 ✓）⇒ 放在取帧线程里
      别卡 UI 线程 ✓。
    """

    def __init__(self, url, fmt=None):
        self.url = str(url or DEFAULT_URL)
        self.fmt = fmt
        self._src = None

    def grab(self):
        if self._src is None:
            from link import PyAVSource
            self._src = PyAVSource(self.url, decode_format="bgr24",
                                   container_format=self.fmt)
            self._src.open()
        _f = self._src.read()
        return None if _f is None else _f.image      # ⚠ 整帧 ✓ 裁不裁由调用方定 ✓

    def close(self):
        try:
            if self._src is not None:
                self._src.close()
        except Exception:                   # noqa: BLE001
            pass
        self._src = None


class VideoSource(FrameSource):
    """录像当实时（⚠ **只在自检 / 命令行调试里用** ✗ —— 用户明说"不用选择素材" ✓）。

    按素材**自己的时间戳**逐帧放（`frames` / `ts` ✓ 来自 `lie_demo.load_source` ✓）：没到点 ⇒
    `None`（这一拍不给帧 ✓）⇒ 看起来就是实时的 ✓。
    ⚠ 它是**唯一**能在离屏 / 无流 / 无屏幕的环境里被完整测到的源 ✓ ⇒ 自检全靠它 ✓。
    ⚠ 用的是**加工帧**（`frames[i][1]` ✓ 与 `lie_demo` 那条路同一份 ✓ 快 ✓）；`crop` 按它的坐标 ✓。
    """

    def __init__(self, path, loop=False):
        self.frames, self.ts, self.proc, self.scale, self.fps = load_source(str(path))
        self.loop = bool(loop)
        self.i = 0
        self._t0 = None

    def reset(self):
        self.i = 0
        self._t0 = None

    def done(self):
        return (not self.frames) or (self.i >= len(self.frames) and not self.loop)

    def grab(self):
        if not self.frames:
            return None
        if self.i >= len(self.frames):
            if not self.loop:
                return None
            self.reset()
        _now = time.time()
        if self._t0 is None:
            self._t0 = _now
        if (_now - self._t0) < float(self.ts[self.i]):
            return None                     # 还没到这一拍的时刻 ✓（这就是"实时"那把尺 ✓）
        _img = self.frames[self.i][1]
        self.i += 1
        return _img                          # ⚠ 加工帧 ✓ 裁不裁由调用方定 ✓


#: `fit_proc` 用的目标宽度（模块级常量 ✓ 免得函数里再查一次 ✓）
_LIMIT_W = PROC_W


# ══════════════════════════════════════════════════════════
# 窗口（PyQt5 ✓ 取帧 / 计算 / 绘制**全在后台线程** ✓ UI 只贴图 ✓）
# ══════════════════════════════════════════════════════════

def run_window(args):
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout,
                                 QLabel, QLineEdit, QMainWindow, QPushButton,
                                 QVBoxLayout, QWidget)

    from gui.region_selector import select_region, select_region_on_image
    from gui.widgets import NoWheelDoubleSpinBox        # ⚠ 滚轮不许改值（UI规范 §1 ✓）

    def _to_qimage(bgr):
        """BGR ndarray ⇒ `QImage`（**拷一份** ✓ —— 不拷的话底层 buffer 会被下一次抓帧改掉 ✗）。"""
        _h, _w = bgr.shape[:2]
        return QImage(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).data, _w, _h,
                      3 * _w, QImage.Format_RGB888).copy()

    class GrabWorker(QThread):
        """**取帧 → 裁 → 算 → 画**（全在后台 ✓）⇒ 发 `QImage` + 一撮统计 ✓。

        ⚠ 为什么整条都放后台 ✗：`grab_rect` 约 20ms、`read()` 直接**阻塞**、tracker 再几毫秒
          ⇒ 放 UI 线程就是"拖窗口都卡" ✗（`live_thread` 那条路也是全在后台 ✓）。
        ⚠ 跨线程只发 **QImage**（后台就 `copy()` 好 ✓）；发 numpy 再在 UI 侧转有"buffer 被下一次
          抓帧改掉"的隐患 ✗。
        ⚠ **抽帧不排队**：源回 `None`（没到点 / 没帧）就睡一下再看 ✓ —— 实时系统的铁律是
          "宁可丢帧、绝不排队" ✓（`live_thread._LatestSlot` 那段注释同理 ✓）。
        """

        frame_ready = pyqtSignal(object, dict)
        failed = pyqtSignal(str)
        finished_all = pyqtSignal()

        def __init__(self, src, crop, engine, ui_fps=SHOW_FPS):
            super().__init__()
            self._src = src
            self._crop = crop
            self._eng = engine
            self._stop = False
            self._want_raw = False            # 界面要一帧"原生"（给"从画面框选"用 ✓）
            self._min_dt = 1.0 / max(1.0, float(ui_fps))
            self._last_show = 0.0
            self._t_win = time.time()
            self._n_win = 0
            self._fps = 0.0
            self._failed = None

        def stop(self):
            self._stop = True

        def request_raw(self):
            """界面喊一声 ⇒ 下一帧额外发一张**没画过任何东西**的原生帧 ✓（UI规范 §8 那条 ✓）。"""
            self._want_raw = True

        def run(self):
            _blank = 0
            while not self._stop:
                try:
                    _f = self._src.grab()
                except TimeoutError:            # udp 暂时没帧 ✓ 不是错误 ✓
                    time.sleep(0.002)
                    continue
                except Exception as e:          # noqa: BLE001 —— 如实报上去 ✓ 不静默黑屏 ✗
                    self._failed = "%s: %s" % (type(e).__name__, e)
                    break
                if _f is None:
                    if getattr(self._src, "done", None) is not None and self._src.done():
                        break
                    _blank += 1
                    time.sleep(0.002 if _blank < 250 else 0.02)
                    continue
                _blank = 0
                if self._want_raw:
                    self._want_raw = False
                    self.frame_ready.emit(_to_qimage(_f), {"raw": True})
                _roi = crop_roi(_f, self._crop) if self._crop is not None else _f
                if _roi is None:
                    continue
                _res = self._eng.step(_roi, time.time())
                self._n_win += 1
                _now = time.time()
                _span = _now - self._t_win
                if _span >= 1.0:
                    self._fps = self._n_win / _span
                    self._t_win, self._n_win = _now, 0
                if _res is None or (_now - self._last_show) < self._min_dt:
                    continue
                self._last_show = _now
                _o, _pos, _rad, _hit, _boxes, _motion = _res
                _hud = "帧 %d ｜ %s ｜ %.1f fps" % (self._eng.n,
                                                   (_o or {}).get("state", "-"), self._fps)
                _vis = draw(_roi.copy(), (_o, _pos, _rad, _hit, _boxes), 1.0, 1.0,
                            hud=_hud, cursor=self._eng.cursor, motion=_motion)
                self.frame_ready.emit(_to_qimage(_vis), {
                    "state": (_o or {}).get("state"), "pos": _pos,
                    "fps": self._fps, "n": self._eng.n, "size": self._eng.size})
            self._src.close()
            if self._failed:
                self.failed.emit(self._failed)
            self.finished_all.emit()

    class Win(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("实时测谎测试（框选屏幕区域 / 收流）· 独立工具")
            self.kind = "stream" if args.stream else "screen"
            self.screen_rect = None           # **屏幕坐标**（屏幕那条 ✓）
            self.crop = None                  # **画面里的 ROI**（收流那条 ✓）
            self.url = args.stream or DEFAULT_URL
            # ⚠ 默认**运动分离**（= 我们当前的进展 ✓ 见 `LiveLieEngine` 那段 ✓）
            self.engine = LiveLieEngine(cfg=load_lie_cfg(),
                                        mode=str(getattr(args, "mode", "motion") or "motion"))
            self.worker = None
            self._det = None                  # 检测器（**常驻子进程** ✓ ⇒ 停的时候必须关 ✗）
            self._raw_q = None
            self._build()
            # ⚠ `--rect` 的**坐标含义随来源变** ✗：屏幕那条是**屏幕坐标** ✓（这里就设 ✓）；
            #   `--video` 调试那条是**画面里的 ROI** ✗ ⇒ 留给 `play()` 去解 ✓（别在这儿当屏幕坐标用 ✗）。
            if args.rect and not getattr(args, "video", ""):
                self._set_screen_rect(args.rect)

        # ---------------- UI ----------------
        def _build(self):
            root = QWidget()
            lay = QVBoxLayout(root)
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(4)
            bar = QHBoxLayout()
            bar.addWidget(QLabel("来源"))
            self.cmb_src = QComboBox()
            self.cmb_src.addItem("本地屏幕区域", "screen")
            self.cmb_src.addItem("收流（udp/srt/rtsp）", "stream")
            self.cmb_src.setCurrentIndex(0 if self.kind == "screen" else 1)
            self.cmb_src.currentIndexChanged.connect(self._on_src_changed)
            bar.addWidget(self.cmb_src)
            self.ed_url = QLineEdit(self.url)
            self.ed_url.setMinimumWidth(260)
            self.ed_url.setToolTip("收流地址，例如：\n"
                                   "  udp://0.0.0.0:5000\n"
                                   "  srt://0.0.0.0:9000?mode=listener\n"
                                   "  rtsp://192.168.1.8:8554/live")
            bar.addWidget(self.ed_url)
            self.btn_rect = QPushButton("框选屏幕区域…")
            self.btn_rect.setToolTip(
                "抓整个虚拟屏幕、在上面框一块 ⇒ **只有这块**会被实时抓取并送进测谎。\n\n"
                "· 带放大镜（+/- 调倍数、Esc 取消）—— 与数据工作台那个框选**同一个实现**；\n"
                "· 抓屏前会先把本窗口藏起来（不然框到的会是自己）；\n"
                "· 框太小（任一边 < %d px）会被拒绝。" % MIN_SIDE)
            self.btn_rect.clicked.connect(self.pick_rect)
            bar.addWidget(self.btn_rect)
            self.btn_pick_crop = QPushButton("从画面框选…")
            self.btn_pick_crop.setToolTip(
                "先「开始」收到画面，再点这里 ⇒ 在这一帧上框一块作为**处理区** ✓\n"
                "（同样带放大镜 ✓；框的必须是**原生帧** ⇒ 所以会专门取一帧没画过标注的 ✓）")
            self.btn_pick_crop.clicked.connect(self.pick_crop)
            bar.addWidget(self.btn_pick_crop)
            self.lbl_rect = QLabel("区域：（未框选）")
            self.lbl_rect.setStyleSheet("color:#9ad;")
            bar.addWidget(self.lbl_rect)
            bar.addStretch(1)
            # ⭐⭐⭐ **检测器开关**（用户 2026-10-03 ✓ 原话："实时测谎：**没有任何检出框**" ✓）：
            #   **检出框是测谎判据的命根子** ✗✗（融合 / 分离 / 砖表 / 点选全靠它 ✓）⇒ 默认**开** ✓。
            self.chk_dets = QCheckBox("用检测器")
            self.chk_dets.setChecked(not bool(getattr(args, "no_dets", False)))
            self.chk_dets.setToolTip(
                "在**框选那块图**上跑 YOLO ⇒ 有检出框 ✓ ⇒ 融合 / 分离 / 砖表 / 点选才工作。\n\n"
                "⚠ 关掉就只剩「白块跟一跟」（上一轮那个取舍 ✗）—— 会看不到任何检出框。\n"
                "⚠ 开着会慢一些：每拍一次推理（约二三十 ms）⇒ 帧率会掉。\n"
                "⚠ 改这个要**停一下再开始**才生效。")
            bar.addWidget(self.chk_dets)
            # ⭐⭐⭐⭐⭐ **管线选择**（用户 2026-10-05 ✓ 原话："**把我们当前的进展做进实时测谎.bat
            #   用于实时验证**" ✓✓）—— **默认「运动分离」** ✓（我们这几轮所有改动都在这条上 ✓）；
            #   「经典」留着对照 ✓。⚠ 换这个要**停一下再开始**才生效 ✓（与检测器开关同一条 ✓）。
            bar.addWidget(QLabel("管线"))
            self.cmb_mode = QComboBox()
            self.cmb_mode.addItem("运动分离（当前进展）", "motion")
            self.cmb_mode.addItem("经典 LieTracker", "classic")
            _m0 = str(getattr(args, "mode", "motion") or "motion")
            self.cmb_mode.setCurrentIndex(0 if _m0 != "classic" else 1)
            self.cmb_mode.setToolTip(
                "**运动分离**：这几轮的所有改动（绿圈推预测 / 融合框夹取 / 夹取增益 / 拆框 /\n"
                "融合必含真目标）全在这条管线上 ⇒ **实时验证用它** ✓。\n"
                "**经典**：老的 `LieTracker`（对照用 ✓）。\n"
                "⚠ 改这个要**停一下再开始**才生效。")
            bar.addWidget(self.cmb_mode)
            bar.addWidget(QLabel("置信度"))
            self.sp_conf = NoWheelDoubleSpinBox()
            self.sp_conf.setRange(0.05, 0.95)
            self.sp_conf.setSingleStep(0.05)
            self.sp_conf.setDecimals(2)
            self.sp_conf.setValue(float(getattr(args, "conf", 0.25) or 0.25))
            bar.addWidget(self.sp_conf)
            self.btn_play = QPushButton("开始")
            self.btn_play.clicked.connect(self.toggle)
            bar.addWidget(self.btn_play)
            lay.addLayout(bar)
            self.view = QLabel("选好来源 → 框一块区域 → 点「开始」")
            self.view.setAlignment(Qt.AlignCenter)
            self.view.setMinimumSize(640, 400)
            self.view.setStyleSheet("background:#101010; color:#888;")
            lay.addWidget(self.view, 1)
            self.setCentralWidget(root)
            self.resize(1100, 640)
            self._on_src_changed()

        def _on_src_changed(self):
            self.kind = self.cmb_src.currentData() or "screen"
            _screen = (self.kind == "screen")
            self.ed_url.setVisible(not _screen)
            self.btn_rect.setVisible(_screen)
            self.btn_pick_crop.setVisible(not _screen)
            self.stop()
            self.screen_rect = None
            self.crop = None
            self.lbl_rect.setText("区域：（未框选）")
            self.statusBar().showMessage(
                "框选一块**屏幕区域**（会实时抓那块）✓" if _screen
                else "填好收流地址 → 点「开始」收到画面后，再点「从画面框选…」✓")

        # ---------------- 框选 ----------------
        def _set_screen_rect(self, rect):
            _vx, _vy, _vw, _vh = self._virtual_screen()
            _c = clamp_rect(rect, _vw, _vh)
            self.screen_rect = _c                 # ⚠ 屏幕坐标**可能是负的**（多显示器 ✓）⇒ 不再夹正 ✓
            if _c is None:
                self.lbl_rect.setText("区域：（框太小，拒绝）")
                return False
            self.lbl_rect.setText("区域：%d,%d  %d×%d（屏幕）" % _c)
            return True

        @staticmethod
        def _virtual_screen():
            try:
                from core import wincap
                return wincap.virtual_screen()
            except Exception:                     # noqa: BLE001 —— 没有 wincap（非 Windows / 离屏）⇒ 退主屏
                _a = QApplication.instance()
                _s = _a.primaryScreen() if _a is not None else None
                if _s is None:
                    return (0, 0, 1920, 1080)
                _g = _s.geometry()
                return (_g.x(), _g.y(), _g.width(), _g.height())

        def pick_rect(self):
            """**抓屏框选**（带放大镜 ✓ 走 `region_selector` ✓ 见 UI规范 §8 ✓）⇒ 屏幕坐标 ✓。"""
            _r = select_region(self)
            if _r is None:
                return                            # 取消 / 误点（< 4px ✓）⇒ 保持原样 ✓ 不猜 ✗
            if not self._set_screen_rect(_r):
                self.statusBar().showMessage(
                    "这块太小了（任一边要 ≥ %d px）⇒ 没采用 ✓ 请重新框 ✓" % MIN_SIDE, 8000)
                return
            self.statusBar().showMessage(
                "已框定屏幕区域 %s ✓ 点「开始」就开始实时抓取 + 测谎 ✓" % (self.screen_rect,))

        def pick_crop(self):
            """**在画面里框**处理区（收流那条用 ✓ 带放大镜 ✓）—— 先要有一帧**原生**画面 ✓。"""
            if self.kind == "screen":
                return
            if self.worker is None or self._raw_q is None:
                self.statusBar().showMessage("先点「开始」收到画面，再框 ✓", 8000)
                return
            _pm = QPixmap.fromImage(self._raw_q)
            _img = self._qimage_to_bgr()
            if _img is None:
                return
            _r = select_region_on_image(_img, self)
            if _r is None:
                return
            _c = clamp_rect(_r, _img.shape[1], _img.shape[0])
            if _c is None:
                self.statusBar().showMessage(
                    "这块太小了（任一边要 ≥ %d px）⇒ 没采用 ✓" % MIN_SIDE, 8000)
                return
            self.crop = _c
            self.lbl_rect.setText("区域：%d,%d  %d×%d（画面）" % _c)
            self.statusBar().showMessage("已框定画面区域 %d,%d %d×%d ✓" % _c)
            self._restart_worker()                # 换 ROI ⇒ 重开那一路（状态清零 ✓）
            del _pm

        def _qimage_to_bgr(self):
            if self._raw_q is None:
                return None
            _q = self._raw_q.convertToFormat(QImage.Format_RGB888)
            _w, _h = _q.width(), _q.height()
            _ptr = _q.constBits()
            _ptr.setsize(_q.byteCount())
            _arr = np.frombuffer(_ptr, np.uint8).reshape((_h, _q.bytesPerLine()))[:, :3 * _w]
            _arr = _arr.reshape((_h, _w, 3))
            return cv2.cvtColor(_arr, cv2.COLOR_RGB2BGR).copy()

        # ---------------- 开关 ----------------
        def toggle(self):
            if self.worker is not None:
                self.stop()
            else:
                self.play()

        def _make_source(self):
            if self.kind == "screen":
                if self.screen_rect is None:
                    self.statusBar().showMessage("先在屏幕上框一块区域 ✓", 8000)
                    return None
                return ScreenSource(self.screen_rect)
            if self.kind == "stream":
                _u = self.ed_url.text().strip() or DEFAULT_URL
                self.url = _u
                return StreamSource(_u, fmt=getattr(args, "format", None))
            return None

        def play(self):
            if getattr(args, "video", ""):        # ⚠ 调试/自检那条（命令行才用 ✓ 界面上没有 ✓）
                _src = VideoSource(args.video, loop=bool(args.loop))
                if self.screen_rect is not None or args.rect:
                    self.crop = clamp_rect(
                        [int(v) for v in str(args.rect).split(",")] if args.rect else None,
                        _src.proc[0], _src.proc[1])
            else:
                _src = self._make_source()
            if _src is None:
                return
            self._open_detector()                 # ⭐ 先接检测器（见它 ✓ 关着就只有白块跟踪 ✓）
            # ⚠ 管线开关在**这里**生效 ✓（改完要"停一下再开始" ✓ 与检测器开关同一条 ✓）——
            #   `reset()` 会把 `runner` 清掉 ✓ ⇒ 下次 `step` 按新 `mode` 重建 ✓（参数也重读 ✓）。
            self.engine.mode = str(self.cmb_mode.currentData() or "motion")
            self.engine.mcfg = (load_motion_cfg() if self.engine.mode == "motion" else {})
            if self.engine.mode == "motion" and not self.chk_dets.isChecked():
                # ⚠⚠ **运动分离对检出框是"硬依赖"** ✗✗（**实测**：检测器关着 ⇒ 状态永远是 `init`
                #    ✓ 屏幕上什么都不动 ✓ 与经典那条不同 ✗ —— 经典还有"白块跟一跟"兜底 ✓）
                #   ⇒ 必须**明说** ✓，不然看着就像"这个工具坏了" ✗。
                self.statusBar().showMessage(
                    "⚠ 运动分离**必须有检出框**（整条判据都建立在框上 ✗）—— 请勾上「用检测器」"
                    "（经典那条才有「白块跟一跟」的兜底 ✗）", 20000)
            self.engine.reset()
            self.worker = GrabWorker(_src, self.crop, self.engine)
            self.worker.frame_ready.connect(self._on_frame)
            self.worker.failed.connect(self._on_failed)
            self.worker.finished_all.connect(self._on_finished)
            self.worker.start()
            self.btn_play.setText("停止")
            self.statusBar().showMessage("跑起来了 ✓（追踪每帧都算 ✓ 画面最多 %.0f fps ✓）" % SHOW_FPS)

        def _open_detector(self):
            """按界面开关建检测器（**常驻子进程** ✓）—— 先关旧的 ✓（不然越开越多 ✗ 会漏进程 ✗）。"""
            self._close_detector()
            if not self.chk_dets.isChecked():
                self.statusBar().showMessage(
                    "⚠ 检测器**关着** ⇒ 不会有任何检出框（融合 / 分离 / 砖表 都不工作 ✓）", 12000)
                return
            try:
                from tools.lie_demo import DetsWorker
                self._det = DetsWorker(getattr(args, "weights", "") or None,
                                       conf=float(self.sp_conf.value()))
            except Exception as _e:               # noqa: BLE001 —— 建不起来必须**说出来** ✗ 不静默 ✗
                self._det = None
                self.statusBar().showMessage("检测器起不来：%s" % _e, 20000)
                return
            self.engine.detector = self._det

        def _close_detector(self):
            if self._det is not None:
                try:
                    self._det.close()
                except Exception:                 # noqa: BLE001
                    pass
                self._det = None
            self.engine.detector = None

        def _restart_worker(self):
            if self.worker is None:
                return
            self.stop()
            self.play()

        def stop(self):
            _w = self.worker
            self.worker = None
            if _w is not None:
                _w.stop()
                _w.wait(2000)
            self._close_detector()                # ⭐ 子进程要收 ✓
            self.btn_play.setText("开始")

        def _on_failed(self, msg):
            self.statusBar().showMessage("取帧失败：%s" % msg, 15000)

        def _on_finished(self):
            self.worker = None
            self.btn_play.setText("开始")
            self.statusBar().showMessage("取帧结束 ✓", 8000)

        # ---------------- 显示 ----------------
        def _on_frame(self, qimg, info):
            if info.get("raw"):
                self._raw_q = qimg
                return
            self.view.setPixmap(QPixmap.fromImage(qimg))
            self.statusBar().showMessage(
                "状态 %s ｜ %.1f fps ｜ 第 %s 拍 ｜ 处理尺度 %s ｜ %s"
                % (info.get("state"), float(info.get("fps") or 0.0), info.get("n"),
                   info.get("size"),
                   ("屏幕 %s" % (self.screen_rect,)) if self.kind == "screen"
                   else ("画面 ROI %s" % (self.crop,))))

    app = QApplication.instance() or QApplication(sys.argv)
    win = Win()
    win.show()
    if getattr(args, "autostart", False):
        win.play()
    return app.exec_()


def main(argv=None):
    ap = argparse.ArgumentParser(description="实时测谎测试（独立工具窗 ✓ 框选屏幕区域 / 收流 ✓）")
    ap.add_argument("--stream", default="", help="收流地址（udp/srt/rtsp ✓ 给了就直接收流）")
    ap.add_argument("--format", default=None, help="容器格式（少数流要显式给 ✓ 照 live_panel）")
    ap.add_argument("--rect", default="", help="屏幕区域 x,y,w,h（**屏幕坐标** ✓ 给了就免手框）")
    ap.add_argument("--autostart", action="store_true", help="开窗即开始（调试用 ✓）")
    ap.add_argument("--no-dets", action="store_true",
                    help="**关掉检测器**（只剩白块跟踪 ⇒ 没有任何检出框 ✓ 一般不用 ✓）")
    ap.add_argument("--conf", type=float, default=0.25, help="检测置信度（默认 0.25 ✓ 同 lie_demo）")
    ap.add_argument("--weights", default="", help="YOLO 权重（不给就自动找 ✓）")
    ap.add_argument("--video", default="", help="[调试/自检] 录像当实时源（**界面上不暴露** ✓）")
    ap.add_argument("--loop", action="store_true", help="[调试] 录像放完循环")
    ap.add_argument("--mode", default="motion", choices=("motion", "classic"),
                    help="哪条管线：motion=运动分离（默认 ✓ 我们当前进展）/ classic=老 LieTracker")
    return run_window(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
