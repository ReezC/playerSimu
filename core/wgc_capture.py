"""WGC 窗口采集：走 Windows Graphics Capture（DWM 合成层），替代 BitBlt 抓屏。

基于 wgc-python（封装 Windows.Graphics.Capture 的 C++ DLL + Python 绑定）。

与 BitBlt 的区别：
  - 不经过 GDI（GetWindowDC/BitBlt），走 DWM 合成层，反作弊更难检测
  - 能抓被遮挡 / 后台的窗口（BitBlt 要求窗口可见）

为什么不用 wgc-python 的 client_area_only=True：它算客户区有偏差（实测
1920x1080 的客户区只抓到 1906x1072，还带 7px 水平偏移）。这里改成抓**整个
窗口**（client_area_only=False，画面 = DWM 扩展边框区域），再自己按屏幕坐标
裁出 rect —— 尺寸和框选精确一致。

接口：grab_rect((x, y, w, h)) -> BGR ndarray（屏幕坐标，和 wincap.grab_rect 一致）。
失败返回 None，调用方回退 BitBlt。
"""

import ctypes
import threading
import time
from ctypes import wintypes

import numpy as np

# 提高 Windows timer 分辨率到 1ms（否则等帧循环的 sleep 会睡到 15.6ms）
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

_DWMWA_EXTENDED_FRAME_BOUNDS = 9
_GA_ROOT = 2

_caps = {}        # hwnd -> WindowCapture（跨帧复用：WGC 会话创建开销大）
_lock = threading.Lock()

# 当前 rect 的窗口几何缓存（避免每帧重复 win32 调用；窗口移动靠 TTL 兜底）
_state = {"key": None, "geom": None, "t": 0.0}
_GEOM_TTL = 0.3   # 几何缓存有效期（秒）


def available():
    """WGC 是否可用（wgc-python 已安装）。"""
    try:
        import wgc_python  # noqa: F401
        return True
    except Exception:
        return False


def _geom_at(x, y):
    """(x, y) 所在顶层窗口的几何。

    返回 (hwnd, title, class_name, ex, ey)：ex/ey 是 WGC 窗口帧原点对应的
    屏幕坐标（DWM 扩展边框左上角，即 WGC 抓窗口画面的 (0,0) 落在屏幕哪）。
    找不到返回 None。
    """
    try:
        import win32gui
    except Exception:
        return None
    try:
        hwnd = win32gui.WindowFromPoint((int(x), int(y)))
        if not hwnd:
            return None
        root = win32gui.GetAncestor(hwnd, _GA_ROOT)
        title = win32gui.GetWindowText(root)
        if not title.strip():
            return None   # 无标题窗口 wgc-python 按 title 定位不到，回退 BitBlt
        cls = win32gui.GetClassName(root)
        rc = wintypes.RECT()
        ok = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(root), _DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rc), ctypes.sizeof(rc))
        if ok == 0:
            ex, ey = rc.left, rc.top
        else:
            ex, ey = win32gui.GetWindowRect(root)[:2]
        return root, title, cls, ex, ey
    except Exception:
        return None


def _get_cap(root, title, cls):
    """取（或建）某窗口的采集会话（整个窗口，非客户区）。失败返回 None。"""
    with _lock:
        cap = _caps.get(root)
        if cap is None:
            try:
                from wgc_python import WindowCapture
                cap = WindowCapture(title, cls, client_area_only=False)
            except Exception:
                return None
            _caps[root] = cap
        return cap


def grab_rect(rect):
    """WGC 抓屏幕矩形 rect=(x, y, w, h)，返回 BGR ndarray；失败返回 None。"""
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        return None

    st = _state
    now = time.monotonic()
    if (st["geom"] is None or st["key"] != (x, y, w, h)
            or now - st["t"] > _GEOM_TTL):
        geom = _geom_at(x, y)
        if geom is None:
            return None
        st["key"] = (x, y, w, h)
        st["geom"] = geom
        st["t"] = now
    root, title, cls, ex, ey = st["geom"]
    cap = _get_cap(root, title, cls)
    if cap is None:
        return None

    # 取最新帧（零拷贝映射，只拷贝需要的区域）。窗口无新帧时短暂等待。
    deadline = time.monotonic() + 0.05
    while True:
        try:
            r = cap.get_frame()
        except Exception:
            r = None
        if r:
            break
        if time.monotonic() >= deadline:
            st["geom"] = None   # 会话可能失效，下次重查
            return None
        time.sleep(0.001)

    ptr, fw, fh, rp = r
    try:
        # rect 在窗口帧内的偏移 = rect 屏幕坐标 - 帧原点屏幕坐标
        ox, oy = x - ex, y - ey
        ox = max(0, min(ox, fw - 1))
        oy = max(0, min(oy, fh - 1))
        cw = min(w, fw - ox)
        ch = min(h, fh - oy)
        if cw <= 0 or ch <= 0:
            return None
        buf = (ctypes.c_ubyte * (fh * rp)).from_address(ptr)
        arr = np.ndarray((fh, fw, 4), dtype=np.uint8,
                         buffer=buf, strides=(rp, 4, 1))
        return np.ascontiguousarray(arr[oy:oy + ch, ox:ox + cw, :3])
    except Exception:
        return None
    finally:
        try:
            cap.release_frame()
        except Exception:
            pass
