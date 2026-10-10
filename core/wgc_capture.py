"""WGC 窗口采集：走 Windows Graphics Capture（DWM 合成层），替代 BitBlt 抓屏。

基于 wgc-python（封装 Windows.Graphics.Capture 的 C++ DLL + Python 绑定）。

与 BitBlt 的区别（2026-10-10 校正 ✓ 原来那句"反作弊更难检测"是**判断不是实测** ✗）：
  - 走 DWM 合成层（系统 API ✓ 按**单个窗口**采 ✓）：**盖在该窗口上面的别的窗口不进画面** ✗；
  - ⚠⚠ 反作弊**一般不管抓屏** ✓（它管进程 / 模块 / 内存 / 注入输入 ✓）—— 本仓库两款实现
    读的都是"系统合成出来的像素"（BitBlt = 桌面 DC ✓ 与录屏软件同类 ✓），**不需要**为
    "会不会被察觉"在两者之间取舍 ✓；真正会发生的只有两类 ✗：① 窗口设了
    `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` ⇒ **抓到黑** ✗（"防录制" ✓ 不是"抓你" ✓）；
    ② 少数反作弊**在游戏进程内** hook GDI 抓取色器 ⇒ 那只作用于游戏自己那个进程 ✓
    （我们是另一个进程读桌面 DC ✓ 走不到它的 hook ✓）。
    口径原文在 `gui/live_panel.py` 的「抓屏」tooltip（**一处说清** ✓ 别各处再写一份 ✗）。

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

# 提高 Windows timer 分辨率到 1ms（否则等帧循环的 sleep 会睡到 15.6ms）。
# 走 core/winperf 的统一入口：同一件事只该有一处实现（工作台启动时也会调用它，
# 见 core/winperf.py —— 那边还负责退出时 timeEndPeriod，别在这里各写一份）。
from core import winperf

try:
    winperf.begin_timer_period(1)
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
    """WGC 抓屏幕矩形 rect=(x, y, w, h)，返回 BGR ndarray；失败返回 None。

    ⚠⚠ **它抓的是"那个点上的窗口自己"**（单窗口采集 ✗）—— 所以：
      · **盖在它上面的别的窗口不会进画面** ✗（外接的登录窗 / 断线提示框等都看不到 ✓
        见 `gui/live_panel.py` 里「抓屏方式」那格 tooltip ✓）⇒ 靠**看画面**判界面的
        功能（断线重连 / 测谎）会**判不出来** ✗ ⇒ 那种用法要用 **BitBlt** ✓；
      · 窗口被别的窗口盖住时，这里按"框中心那个点"找到的是**盖它的那个窗口** ✓
        —— 2026-10-09 试过"认死 hwnd 只抓源窗口" ✗ **当场被用户否掉** ✓
        （用户原话："**我需要的是盖住的窗口和游戏都显示，因为登陆界面是外接的窗口，
        你不显示断线重连会有问题**" ✓）⇒ **已改回本行为** ✓ 别再往回改 ✗。
    """
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        return None

    st = _state
    now = time.monotonic()
    if (st["geom"] is None or st["key"] != (x, y, w, h)
            or now - st["t"] > _GEOM_TTL):
        # ⚠⚠ **按"框的中心点"找窗口**（2026-10-09 修 ✗ 用户现场："本地的实时窗口不是框选屏幕
        #   区域实现的，好像是捕获程序窗口"）：以前用 rect 的**左上角**那个点 ✗ ⇒ 只要那个角
        #   压到别的窗口 / 任务栏 / 工作台，就**抓到别的窗口** ✗；中心点更符合"我框的是哪块屏幕" ✓。
        #   中心点也取不到（窗口无标题那种 ✓）⇒ 退回左上角（老行为 ✓ 不倒退）。
        geom = _geom_at(x + w // 2, y + h // 2) or _geom_at(x, y)
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
