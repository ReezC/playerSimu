"""窗口枚举的 **ctypes 兜底**（用户 2026-10-17 ✓ 现场：A 机部署台点「跟随窗口」⇒
「没找到窗口」，可那台机器明明开着十几个程序 ✓）。

**为什么要这个模块**：
  · `core/wincap.list_windows` 原来**只在 pywin32 在的时候**才列得出窗口 ✗，
    缺了就直接 `return []` ✓ —— **静默** ✓ 用户只看到「没找到窗口」✓
    而那明明是假话 ✓（正是最难查的那种 ✓）；
  · `deploy/requirements.txt` 里**没有 pywin32** ✗（A 机只装 PyQt5 / pyserial / PyYAML ✓）
    ⇒ A 机上**必然**踩 ✓。

**做法**：只用**标准库 `ctypes`** ✓ **一个包都不用装** ✓
（A 机原则：能不上 pip 就不上 ✓ 见 `deploy/requirements.txt` 顶部 ✓）。

**口径与 pywin32 那条完全一致**（谁也别改单边 ✗）：
  · 只要**可见 / 有标题**的顶层窗口 ✓；
  · `rect` 是**客户区**（`GetClientRect` + `ClientToScreen` ✓ 不含标题栏边框 ✓
    —— 推流要抓的就是这块 ✓）；
  · 按面积**从大到小** ✓（游戏窗口一般排第一 ✓）；
  · `window_geom` 是**外框**（DWM 扩展边框优先 ✓ 同 `wincap.window_geom` ✓）——
    相对比例就是拿它算的 ✓（`rel_from_rect` / `binding_rect` ✓）。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

#: `DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS)`（9）—— 与 pywin32 那条同一个数 ✓
_DWMWA_EXTENDED_FRAME_BOUNDS = 9

_user32 = None
_dwmapi = None


def _u():
    """user32 句柄（声明好 argtypes ⇒ 64 位下句柄不会被截断 ✓）。"""
    global _user32
    if _user32 is not None:
        return _user32
    u = ctypes.WinDLL("user32", use_last_error=True)
    u.IsWindow.argtypes = [wintypes.HWND]
    u.IsWindowVisible.argtypes = [wintypes.HWND]
    u.IsIconic.argtypes = [wintypes.HWND]
    u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    u.GetWindowThreadProcessId.argtypes = [wintypes.HWND,
                                           ctypes.POINTER(wintypes.DWORD)]
    _user32 = u
    return u


def _dwm():
    global _dwmapi
    if _dwmapi is None:
        _dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    return _dwmapi


def available():
    """`ctypes` 这条路永远可用（user32 是系统库 ✓）—— 只用来给界面说人话 ✓。"""
    try:
        _u()
        return True
    except Exception:                      # noqa: BLE001
        return False


def pid_of(hwnd):
    """窗口的进程 ID（拿不到给 0 ✓）—— 用来把**部署台自己**排掉 ✓。"""
    try:
        pid = wintypes.DWORD(0)
        _u().GetWindowThreadProcessId(wintypes.HWND(int(hwnd)), ctypes.byref(pid))
        return int(pid.value)
    except Exception:                      # noqa: BLE001
        return 0


def is_minimized(hwnd):
    """最小化了没有（最小化的窗口没有可用的屏幕几何 ✓）。"""
    try:
        return bool(_u().IsIconic(wintypes.HWND(int(hwnd))))
    except Exception:                      # noqa: BLE001
        return False


def window_geom(hwnd):
    """窗口**外框** `(x, y, w, h)` ✓ / `None` ✓（口径同 `wincap.window_geom` ✓）。

    ⚠ 最小化 / 问不到 / 明显是"最小化占位矩形"（-32000 那种）⇒ `None` ✓ ——
      调用方（`binding_rect`）会按标题重绑或退回绝对区域 ✓。
    """
    try:
        h = wintypes.HWND(int(hwnd))
        u = _u()
        if not u.IsWindow(h) or u.IsIconic(h):
            return None
        rc = wintypes.RECT()
        ok = _dwm().DwmGetWindowAttribute(h, _DWMWA_EXTENDED_FRAME_BOUNDS,
                                          ctypes.byref(rc), ctypes.sizeof(rc))
        if ok != 0:
            if not u.GetWindowRect(h, ctypes.byref(rc)):
                return None
        l, t, r, b = rc.left, rc.top, rc.right, rc.bottom
        if (r - l) <= 0 or (b - t) <= 0:
            return None
        if l <= -30000 or t <= -30000 or (r - l) <= 60 or (b - t) <= 60:
            return None
        return (int(l), int(t), int(r - l), int(b - t))
    except Exception:                      # noqa: BLE001
        return None


def list_windows(min_w=200, min_h=150):
    """`[{hwnd, title, rect(客户区), size}]` ✓（面积从大到小 ✓ 同 pywin32 那条 ✓）。"""
    out = []
    try:
        u = _u()
    except Exception:                      # noqa: BLE001
        return out

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lparam):
        try:
            if not u.IsWindowVisible(hwnd) or u.IsIconic(hwnd):
                return True
            n = int(u.GetWindowTextLengthW(hwnd))
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value
            if not title.strip():
                return True
            rc = wintypes.RECT()
            if not u.GetClientRect(hwnd, ctypes.byref(rc)):
                return True
            pt = wintypes.POINT(0, 0)
            if not u.ClientToScreen(hwnd, ctypes.byref(pt)):
                return True
            w = int(rc.right - rc.left)
            h = int(rc.bottom - rc.top)
            if w < int(min_w) or h < int(min_h):
                return True
            out.append({"hwnd": int(hwnd), "title": title,
                        "rect": (int(pt.x), int(pt.y), w, h), "size": (w, h)})
        except Exception:                  # noqa: BLE001 —— 单个窗口坏了不影响整轮 ✓
            pass
        return True

    try:
        u.EnumWindows(WNDENUMPROC(_cb), 0)
    except Exception:                      # noqa: BLE001
        return out
    out.sort(key=lambda x: (-x["size"][0] * x["size"][1]))
    return out
