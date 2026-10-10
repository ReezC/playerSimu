"""窗口捕获：只抓游戏窗口，不抓整个桌面。

**为什么需要**：直接录屏会把桌面图标、任务栏、聊天窗口一起录进去。
后果不只是"画面不干净"——标注时会把这些无关内容也匹配上，
训练时模型会学到"任务栏 = 目标"，而且每次标注都要额外设 region 去排除。

**抓取方式**：取窗口**客户区**（不含标题栏和边框）的屏幕坐标，
再用 PIL.ImageGrab 抓那个矩形。

为什么不用 PrintWindow（让窗口自己渲染到位图，被遮挡也能抓）：
很多游戏用 DirectX / 硬件加速渲染，PrintWindow 会返回一张黑图。
屏幕抓取的代价是窗口必须可见、不能被完全遮挡，但对本场景完全够用。
"""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab

from core.context import TaskContext
from core.imgio import imwrite


# ══════════════════════════════════════════════════════════════
# 窗口枚举
# ══════════════════════════════════════════════════════════════
def available():
    """窗口捕获是否可用（需要 pywin32）。"""
    try:
        import win32gui  # noqa: F401
        return True
    except Exception:
        return False


def list_windows(min_w=200, min_h=150):
    """列出可见的顶层窗口。

    返回 [{hwnd, title, rect, size}]，rect 是**客户区**的屏幕坐标 (x, y, w, h)。
    过滤掉太小的窗口（工具提示、浮动面板之类）。
    """
    try:
        import win32gui
    except Exception:
        return []

    out = []

    def cb(hwnd, _):
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True

            title = win32gui.GetWindowText(hwnd)
            if not title.strip():
                return True

            # 客户区 = 游戏真正绘制的区域，不含标题栏和窗口边框
            cl, ct, cr, cb_ = win32gui.GetClientRect(hwnd)
            sx, sy = win32gui.ClientToScreen(hwnd, (cl, ct))
            ex, ey = win32gui.ClientToScreen(hwnd, (cr, cb_))

            w, h = ex - sx, ey - sy
            if w < min_w or h < min_h:
                return True

            out.append({"hwnd": hwnd, "title": title,
                        "rect": (sx, sy, w, h), "size": (w, h)})
        except Exception:
            pass
        return True

    win32gui.EnumWindows(cb, None)
    out.sort(key=lambda x: (-x["size"][0] * x["size"][1]))
    return out


def find_window(keyword):
    """按标题关键词找窗口（找最大的那个）。找不到返回 None。"""
    kw = (keyword or "").lower()
    hits = [w for w in list_windows() if kw in w["title"].lower()]
    return hits[0] if hits else None


def list_windows_sorted():
    """按面积从大到小列出（一般游戏窗口最大，排最前）。"""
    return list_windows()


# ══════════════════════════════════════════════════════════════
# 抓取
# ══════════════════════════════════════════════════════════════
def grab_rect_fast(rect):
    """win32 BitBlt 抓屏，返回 BGR ndarray。

    **为什么不用 PIL ImageGrab**：ImageGrab 抓 1920x1080 要 60ms+，
    实时窗口识别只能跑到 15fps，设 60fps 也白搭。BitBlt 约 20ms，
    能到 48fps（1366x768 约 80fps）。坐标体系和 ImageGrab(all_screens=True)
    一致，都是虚拟屏幕坐标。
    """
    import win32con
    import win32gui
    import win32ui

    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        raise ValueError("矩形无效: %s" % (rect,))

    hwnd = win32gui.GetDesktopWindow()
    hdc = win32gui.GetWindowDC(hwnd)
    mfc_dc = save_dc = bmp = None
    try:
        mfc_dc = win32ui.CreateDCFromHandle(hdc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bmp)
        save_dc.BitBlt((0, 0), (w, h), mfc_dc, (x, y), win32con.SRCCOPY)
        bmpstr = bmp.GetBitmapBits(True)
        # GetBitmapBits 返回 BGRA、top-down（实测和 PIL 像素一致，无需翻转）
        img = np.frombuffer(bmpstr, dtype="uint8").reshape(h, w, 4)[:, :, :3]
        return np.ascontiguousarray(img)
    finally:
        for obj in (bmp, save_dc, mfc_dc):
            if obj is not None:
                try:
                    if hasattr(obj, "GetHandle"):
                        win32gui.DeleteObject(obj.GetHandle())
                    else:
                        obj.DeleteDC()
                except Exception:
                    pass
        win32gui.ReleaseDC(hwnd, hdc)


def virtual_screen():
    """虚拟屏幕（所有显示器合并）边界 (x, y, w, h)，屏幕坐标。

    多显示器时 x/y 可能为负（左边/上边的显示器）。框选区域和抓屏都用
    这套坐标，才能跨显示器一致。
    """
    import ctypes

    u = ctypes.windll.user32
    return (u.GetSystemMetrics(76), u.GetSystemMetrics(77),   # SM_X/YVIRTUALSCREEN
            u.GetSystemMetrics(78), u.GetSystemMetrics(79))   # SM_CX/CYVIRTUALSCREEN


def rect_to_ratio(rect):
    """屏幕绝对坐标 (x, y, w, h) → 相对虚拟屏幕的比例 [nx, ny, nw, nh]（0~1）。

    用比例存储，屏幕分辨率 / 显示器布局变化后仍能正确定位（如 HP/MP 条区域）。
    """
    vx, vy, vw, vh = virtual_screen()
    x, y, w, h = (float(v) for v in rect)
    return [round((x - vx) / vw, 6), round((y - vy) / vh, 6),
            round(w / vw, 6), round(h / vh, 6)]


def rect_from_ratio(ratio):
    """相对虚拟屏幕的比例 [nx, ny, nw, nh] → 屏幕绝对坐标 (x, y, w, h)。"""
    vx, vy, vw, vh = virtual_screen()
    nx, ny, nw, nh = (float(v) for v in ratio)
    return (int(vx + nx * vw), int(vy + ny * vh), int(nw * vw), int(nh * vh))


def window_at(x, y):
    """屏幕坐标 `(x, y)` 上是哪个**顶层窗口** ⇒ `(hwnd, 标题)`；拿不到 ⇒ None ✓。

    框选那一刻记一次 ✓（那会儿窗口还看得见 ✓）；之后只用它问"这个窗口现在在哪"
    （`window_geom` ✓）⇒ **框跟着窗口走** ✓。⚠ 注意：**不用它决定"抓谁的画面"** ✗ ——
    那条会漏掉盖在窗口上面的外接弹窗 ✗（见 `grab_rect` 的说明 ✓ 用户 2026-10-09 明确否过 ✓）。
    """
    try:
        import win32gui
        h = win32gui.WindowFromPoint((int(x), int(y)))
        if not h:
            return None
        root = win32gui.GetAncestor(h, 2)          # GA_ROOT ✓
        return root, (win32gui.GetWindowText(root) or "")
    except Exception:                              # noqa: BLE001
        return None


def client_rect(hwnd):
    """`hwnd` 的**客户区**在屏幕上的 `(x, y, w, h)`；拿不到 ⇒ `None` ✓。

    ⭐⭐ 为什么专门要它（2026-10-10 ✓ 用户原话："**框选窗口能做个自适应吗？就是我鼠标指着
      的时候它自动匹配窗口画面的尺寸 —— 不然我每次自己框都会有偏差**"✓）：
      「窗口画面的尺寸」= **客户区** ✓（游戏真正绘制的区域 ✓ 不含标题栏 / 边框 ✓）
      —— 手拖永远差几个像素 ✗（用户那句"每次都会有偏差"就是这个 ✓）。
    ⚠ 与 `window_geom`（**DWM 外框** ✓ 含标题栏）**不是一回事** ✗：外框是"窗口在哪"（给
      `rect_rel` 当基准 ✓ 见 `binding_rect` ✓）；客户区才是"画面从哪儿开始" ✓ 两个都要 ✓。
    """
    try:
        import win32gui
        if not hwnd or not win32gui.IsWindow(hwnd):
            return None
        cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
        sx, sy = win32gui.ClientToScreen(hwnd, (cl, ct))
        ex, ey = win32gui.ClientToScreen(hwnd, (cr, cb))
        w, h = int(ex - sx), int(ey - sy)
        if w <= 0 or h <= 0:
            return None
        return (int(sx), int(sy), w, h)
    except Exception:                          # noqa: BLE001
        return None


def window_under_cursor(skip_pid=None, min_w=200, min_h=150):
    """**鼠标底下**那个顶层窗口 ⇒ `{"hwnd", "title", "rect"(客户区), "pid"}` | `None` ✓。

    ⭐ 给"自动框选"用的（用户 2026-10-10 ✓ 见 `client_rect` 那段 ✓）。
    `skip_pid`：**排掉这个进程自己的窗口** ✓ —— 工作台自己**绝不许**被抓进来 ✗
      （不然"指着工作台"就抓成工作台 ✓ 那比手框还糟 ✗）。
    ⚠ 判据与 `window_at` **同一口径**（`WindowFromPoint` + 爬到 `GA_ROOT` ✓ 见那边 ✓）——
      区别只是这里还要标题 / 客户区 / pid ✓。
    """
    try:
        import win32api
        import win32con
        import win32gui
        import win32process
        x, y = win32api.GetCursorPos()
        win = win32gui.WindowFromPoint((int(x), int(y)))
        if not win:
            return None
        top = win32gui.GetAncestor(win, win32con.GA_ROOT) or win
        if not win32gui.IsWindowVisible(top) or win32gui.IsIconic(top):
            return None
        title = win32gui.GetWindowText(top) or ""
        if not title.strip():
            return None
        pid = 0
        try:
            _t, pid = win32process.GetWindowThreadProcessId(top)
        except Exception:                      # noqa: BLE001
            pid = 0
        if skip_pid is not None and int(pid) == int(skip_pid):
            return None
        r = client_rect(top)
        if r is None or r[2] < int(min_w) or r[3] < int(min_h):
            return None
        return {"hwnd": int(top), "title": title, "rect": r, "pid": int(pid)}
    except Exception:                          # noqa: BLE001
        return None


def window_geom(hwnd):
    """`hwnd` 现在的屏幕几何 `(x, y, w, h)`（DWM 扩展边框 ✓）；**问不到 ⇒ None** ✓。

    ⚠⚠ **最小化也要算"问不到"** ✗（2026-10-10 ✓ 用户报"本地窗口实时并不是用我框选的区域，
      而是把两台显示器的全部内容展示了"✓ 查出来的第一处）：窗口最小化时 `IsWindow`
      照样是真 ✓、`GetWindowRect` 也照样给数 ✗ —— 但那是 Windows 的**最小化占位矩形**
      `(-32000, -32000, 160, 31)` 之类 ✓（本机实测 `(-31993, -32000, 146, 28)` ✓）
      ⇒ 拿它当"窗口现在在哪" ⇒ 落点算到屏幕外 ✗（更坏的是：调用方一旦判定"窗口不可用"，
      就会去走别的退路 ✓ 见 `binding_rect` ✓）。
    """
    if not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        import win32gui
        if not win32gui.IsWindow(hwnd):
            return None
        if win32gui.IsIconic(hwnd):                 # 已最小化 ⇒ 没有可用的屏幕几何 ✓
            return None
        rc = wintypes.RECT()
        ok = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd), 9, ctypes.byref(rc), ctypes.sizeof(rc))   # 9 = 扩展边框 ✓
        if ok != 0:
            l, t, r, b = win32gui.GetWindowRect(hwnd)
        else:
            l, t, r, b = rc.left, rc.top, rc.right, rc.bottom
        if r - l <= 0 or b - t <= 0:
            return None
        # ⚠ 兜一道"最小化占位矩形"（`IsIconic` 偶尔问不到时会漏过来 ✓ —— 本机实测过那种值 ✓）
        if l <= -30000 or t <= -30000 or (r - l) <= 60 or (b - t) <= 60:
            return None
        return (int(l), int(t), int(r - l), int(b - t))
    except Exception:                              # noqa: BLE001
        return None


def rel_from_rect(rect, geom=None):
    """屏幕 rect `(x, y, w, h)` ⇒ 比例 `[nx, ny, nw, nh]` ✓。

    `geom`（窗口几何）= **相对那个窗口** ✓（窗口挪了/换分辨率都跟得上 ✓ 这才是"不用重框"的关键 ✓）；
    不给 ⇒ 相对**虚拟屏幕**（老行为 ✓ 同 `rect_to_ratio` ✓）。
    """
    x, y, w, h = (float(v) for v in rect)
    gx, gy, gw, gh = [float(v) for v in (geom if geom else virtual_screen())]
    gw, gh = max(1e-6, gw), max(1e-6, gh)
    return [round((x - gx) / gw, 6), round((y - gy) / gh, 6),
            round(w / gw, 6), round(h / gh, 6)]


def rect_from_rel(rel, geom=None):
    """比例 `[nx, ny, nw, nh]` ⇒ 屏幕 rect `(x, y, w, h)` ✓（**纯函数** ⇒ 能单测 ✓）。

    `geom`（窗口几何 ✓）优先 ✓；没有（窗口关了 / 没记过 ✓）⇒ 按**虚拟屏幕**折算 ✓（老行为 ✓）。
    """
    nx, ny, nw, nh = (float(v) for v in rel)
    gx, gy, gw, gh = [float(v) for v in (geom if geom else virtual_screen())]
    return (int(round(gx + nx * gw)), int(round(gy + ny * gh)),
            int(round(nw * gw)), int(round(nh * gh)))


def binding_rect(rel, win, fallback_rect):
    """「框选区域」这一拍**该抓哪块屏** ⇒ `(rect, why)` —— `why` = 一句**短**人话（没退路 ⇒ ""）。

    · `rel` = 框选那一刻记的**相对那个窗口**的比例 ✓（见 `rel_from_rect` ✓）；
    · `win` = 那个窗口的**句柄** ✓（`gui/live_panel._pick_rect` 记的 ✓）；
    · `fallback_rect` = 你框的那块**绝对屏幕区域** ✓（老行为 ✓）。

    规则（2026-10-10 ✓ 用户原话：**"现在本地窗口实时并不是用的我框选的屏幕区域，而是把
    2 个显示器的全部内容展示了"** ✓）：
      · 没记过窗口 / 没记过比例 ⇒ **绝对区域** ✓（老行为一字不变 ✓）；
      · 窗口还在、几何问得到 ⇒ 按它**现在的**几何折算 ✓（挪了 / 换分辨率不用重框 ✓）；
      · 窗口**问不到**（关了 / 最小化 / 句柄失效 ✓ 见 `window_geom` ✓）⇒ **绝对区域** ✓
        ＋ 一句"为什么退回你框的那块" ✓（⚠ 不许静默 ✓）。

    ⚠⚠⚠ **绝不许**在"窗口问不到"时把这份比例按**虚拟屏幕**折算 ✗：
      它是**相对窗口**的（≈ 0,0,1,1 ⇒ "整个窗口" ✓），而虚拟屏幕是**两台显示器拼起来的** ✓
      ⇒ 一折就把"整窗口"放大成"整块桌面" ✓（本机实测：5117×1400 ✓ 用户看到的就是它 ✓）。
    """
    if not rel or not win:
        return tuple(fallback_rect), ""
    try:
        h = int(win)
    except (TypeError, ValueError):
        return tuple(fallback_rect), ""
    g = window_geom(h)
    if g is None:
        return tuple(fallback_rect), (
            "框绑定的那个窗口现在**问不到**（关了 / 最小化 / 句柄失效）"
            "⇒ 已按**你框的那块固定区域**抓（想跟着窗口走 ⇒ 重框一次 ✓）")
    return tuple(rect_from_rel(rel, g)), ""


def resolve_rect(v):
    """兼容解析区域：新格式比例 [nx,ny,nw,nh]（值都在 0~1）反算成绝对坐标；
    旧格式绝对坐标 [x,y,w,h] 原样返回。"""
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        nums = [float(n) for n in v]
    except Exception:
        return None
    if all(0.0 <= n <= 1.0 for n in nums):
        return rect_from_ratio(nums)
    return [int(n) for n in nums]


# ══════════════════════════════════════════════════════════════
# 「这块矩形被别的窗口盖住了没有」
# ══════════════════════════════════════════════════════════════
#: 采样点（相对矩形的比例 ✓）：**中心 + 四角偏内 + 四边中点 = 9 个** ✓。
#: ⚠ 只看中心那一个点会被"正好压在中心的小窗"骗过 ✗ —— 而 WGC 恰恰就是按中心点找窗口的
#:   （`core/wgc_capture.py` ✓）⇒ 判"要不要换抓法"必须**多取几个点**，才看得出
#:   "这块矩形其实跨了不止一个窗口" ✓。
_OCCL_POINTS = ((0.5, 0.5), (0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
                (0.5, 0.06), (0.5, 0.94), (0.06, 0.5), (0.94, 0.5))


def title_at(x, y):
    """屏幕上那个点**最顶层**窗口的标题（取不到 ⇒ "" ✓）。

    ⚠⚠ 常量要用 **`win32con`** ✗（`win32gui.GA_ROOT` **不存在** ✓ —— 2026-10-10 真 API
      冒烟当场抓到：它抛 `AttributeError`、被下面的 `except` 吞掉 ⇒ **标题永远是空** ✗
      ⇒ `covered_by` 永远判"没被盖"✗。⚠ mock 的用例**抓不到这种错** ✓
      ⇒ 这一层必须**用真 API 冒烟一次** ✓ 记在这儿 ✓）。
    ⚠ `WindowFromPoint` 给的是**最深的那个子窗口**（实测：编辑器里是
      `Chrome_RenderWidgetHostHWND` / "Chrome Legacy Window" ✓）⇒ 必须 `GA_ROOT` 爬到
      **顶层**才是人认得的那份标题（"playerSimu - CodeBuddy CN" ✓）。
    """
    try:
        import win32con
        import win32gui
        win = win32gui.WindowFromPoint((int(x), int(y)))
        if not win:
            return ""
        top = win32gui.GetAncestor(win, win32con.GA_ROOT) or win
        return win32gui.GetWindowText(top) or ""
    except Exception:                            # noqa: BLE001 —— 问不到就当"没窗口" ✓
        return ""


def class_at(x, y):
    """屏幕上那个点**最顶层**窗口的**类名**（取不到 ⇒ "" ✓）—— 用来认出"桌面 / 任务栏" ✓。"""
    try:
        import win32con
        import win32gui
        win = win32gui.WindowFromPoint((int(x), int(y)))
        if not win:
            return ""
        top = win32gui.GetAncestor(win, win32con.GA_ROOT) or win
        return win32gui.GetClassName(top) or ""
    except Exception:                            # noqa: BLE001
        return ""


#: **桌面/任务栏**那几个窗口的类名 —— 采样点落在它们上面**不算"被盖住"** ✗。
#: ⚠ 为什么必须排掉（2026-10-10 真 API 冒烟当场看出来的 ✗）：矩形的四个角常常落在**空桌面**
#:   上（`Progman` / "Program Manager" ✓），而桌面/任务栏是在**所有窗口底下** ✓ —— 它是
#:   "源窗口没铺到那儿"，**不是**"挡在源窗口前面" ✗ ⇒ 不排掉的话，只要框比窗口大一点点，
#:   就会永远判"被盖住" ⇒ 永远走整屏合成（等于把 WGC 那档废掉 ✗ 白扔性能 ✓）。
_SHELL_CLASSES = ("Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd")


def covered_by(rect):
    """这块矩形**被哪个别的窗口盖住了** ⇒ `(遮挡物标题, 人话)`；没被盖 ⇒ `(None, "")` ✓。

    判据（刻意做得简单、可解释 ✓）：在矩形里取 9 个点（见 `_OCCL_POINTS` ✓），看
    **最顶层窗口的标题**是不是同一个 —— 全都一样 ⇒ 没被盖 ✓（整块都属于同一个窗口 ✓）；
    有不一致的 ⇒ 说明这 9 个点跨了**不止一个窗口** ⇒ 源窗口被别的窗口压住了 ✓
    ⇒ 把"**除中心那个窗口之外、出现最多**"的那个标题当遮挡物 ✓（拿中心那个当"源窗口" ✓
    与 WGC 自己的挑法一致 ✓）。

    ⚠⚠ **只认标题、不认句柄** ✗（这是本仓库的**明确纪律** ✓ 用户 2026-10-09 原话：
      "我需要的是**盖住的窗口和游戏都显示**，因为登陆界面是外接的窗口，你不显示断线重连
      会有问题" ✓）。⇒ 这个函数**只用来决定"要不要改用整屏合成"** ✓，
      **绝不许**拿它把采集**认死到某一个窗口**上 ✗（认死窗口 = 盖住它的登录窗不进画面 ✗
      ⇒ 断线重连 / 测谎判不出界面 ✓ 那一版用户当场否掉了 ✓ 见
      `tools/selftest_live_panel.py` 里那条反向钉 ✓）。
    """
    try:
        x, y, w, h = (int(v) for v in rect)
    except (TypeError, ValueError):
        return None, ""
    if w <= 0 or h <= 0:
        return None, ""
    titles = [title_at(x + w * px, y + h * py) for px, py in _OCCL_POINTS]
    cnt = {}
    for t in titles:
        cnt[t] = cnt.get(t, 0) + 1
    if len(cnt) <= 1:
        return None, ""                       # 9 个点同一个窗口（或同一个"没窗口"）⇒ 没被盖 ✓
    src = titles[0]                           # 中心那个 = WGC 会挑的那个 ✓
    # ⚠ 桌面 / 任务栏那几个点不算"被盖" ✗（它们在**所有窗口底下** ✓ 见 `_SHELL_CLASSES` ✓）：
    #   否则"框比窗口大一点点"就会永远判被盖 ⇒ WGC 那档等于废掉 ✓。
    others = []
    for (px, py), t in zip(_OCCL_POINTS, titles):
        if t == src:
            continue
        if class_at(x + w * px, y + h * py) in _SHELL_CLASSES:
            continue
        others.append(t)
    if not others:
        return None, ""
    # 出现最多的那个（"没标题的窗口"排最后 ✓ 但也要说得出话 ✓）
    cnt2 = {}
    for t in others:
        cnt2[t] = cnt2.get(t, 0) + 1
    cover = sorted(cnt2, key=lambda k: (cnt2[k], bool(k)), reverse=True)[0]
    cover = cover or "一个没有标题的窗口"
    return cover, ("画面里**不止一个窗口**（「%s」之外还有「%s」）⇒ 这一拍**改抓整屏**"
                   "（WGC 只抓一个窗口 ✗ 盖住它的登录窗 / 断线提示框就不进画面 ✓）"
                   % (src or "源窗口", cover))


#: 最近一次 `grab_rect` **实际走了哪条路**（`dxgi` / `wgc` / `screen` / `bitblt` / `pil` ✓）
#: 以及"**为什么要换路**"（人话 ✓ 没有就是 "" ✓）—— 给界面状态行 / `behavior.log` 用 ✓。
#: ⚠ 写只发生在**调用 `grab_rect` 的那条线程**里 ✓；读的是另一条线程（主回路 ✓）
#:   ⇒ 读走 `last_note()` 的快照 ✓（一次 dict 拷贝就够 ✓ 不追求严格同步 ✓）。
_NOTE = {"how": "", "why": ""}

#: ⭐⭐ 「上一次判到**被别的窗口盖住**」的时刻 —— 给"切回单窗口抓法"加**迟滞** ✓。
#: ⚠⚠ 为什么要迟滞（2026-10-10 ✓ 真机日志实测踩到 ✗）：采样点贴着窗口边缘时 `covered_by`
#:   会**来回翻** —— 那段日志里 **8 秒翻了 4 次**（`screen → wgc` 只隔 **14 ms** ✗）
#:   ⇒ 同一路流一会儿是"单窗口"、一会儿是"整屏合成" ✗ = 本文件 warning 过的那种抖动
#:   （断线重连 / 测谎**会被它骗到** ✓）—— 而且用户报的「**本地实时断线后没有重连警报**」✓
#:   正落在这上面：外接的断线/登录窗是**另一个窗口** ✗，只有"整屏合成"那条路才拍得到它 ✓
#:   ⇒ 只要有一拍的判定翻回 WGC ✗，那 1 秒一次的识别就可能正好撞上"没有那个窗"的帧 ✗
#:   ⇒ 判不出界面 ⇒ **警报不响** ✓✓。
#:   规则（照"进去立刻、出来要等"这条通用做法 ✓）：**判到被盖 ⇒ 立刻切** ✓（外接窗要马上
#:   进画面 ✓）；**判到不被盖 ⇒ 再坚持 `COVER_HOLD_S` 秒**才切回 ✓（边缘噪声翻不动它 ✓）。
_COVER = {"last": 0.0}
COVER_HOLD_S = 2.0


def last_note():
    """最近一次抓帧**实际走的抓法 + 换路的理由**（人话 ✓ 见 `_NOTE` ✓）。"""
    return dict(_NOTE)


def grab_rect(rect, use_wgc=True, prefer_dxgi=True):
    """抓屏幕上指定的矩形，返回 BGR ndarray（**画面没变 / 全失败 ⇒ None** ✓）。

    取法**从快到慢**排队（前三条读的都是"屏幕上看到的样子" ✓ 只有 WGC 那条按窗口采 ✗）：

      1. **DXGI 桌面复制**（`core/dxgi_capture` ✓ **最快** ✓）：本机实测 1920×1080 **16.9 ms**、
         960×540 **5.7 ms** ✓（对照：BitBlt 同尺寸 50.5 / 23.4 ms ✓）—— 它读的是
         **DWM 合成结果** ⇒ **外接登录窗照样进画面** ✓✓（断线重连 / 测谎要的就是这个 ✓）；
         ⚠ DXGI **只在画面有变化时给新帧** ⇒ 这里拿到 `None` 就把机会让给下面的路 ✓
         （**调用方**也该把 `None` 当"这拍没变化、沿用上一帧" ✓ 见 `gui/live_thread` ✓）；
      2. **WGC**（`use_wgc=True` 才试 ✓）：按**单个窗口**采 ✗ ⇒ 盖在它上面的窗口**不进画面** ✗
         （用户 2026-10-09 原话："**我需要的是盖住的窗口和游戏都显示**，因为登陆界面是外接的
         窗口，你不显示断线重连会有问题" ✓ ⇒ 那种用法**别选它** ✓ 选 `bitblt` / 让 DXGI 兜住 ✓）；
         ⭐⭐ **被别的窗口盖住时，这一档也自动让路**（2026-10-10 ✓ 用户选的口径 ✓ 见
         `covered_by` 上面那段 ✓）：`covered_by()` 判出"这块矩形跨了不止一个窗口" ⇒
         **跳过 WGC、直接走整屏合成** ✓ ⇒ 盖住的登录窗 / 断线提示框照样进画面 ✓
         （不这么做的话：WGC 按"框中心那个点"找窗口 ✓ ⇒ 中心被遮挡物占了就**抓成遮挡物** ✗）；
      3. **win32 BitBlt**（桌面 DC ✓ 慢但最稳 ✓ 同样读合成结果 ✓）；
      4. **PIL ImageGrab**（最慢，最后兜底 ✓）。
    """
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        raise ValueError("矩形无效: %s" % (rect,))
    _NOTE["how"], _NOTE["why"] = "", ""

    # 0) DXGI 桌面复制：最快，而且同样是"屏幕合成结果"（外接弹窗照样进画面 ✓）
    _dxgi_ok = False
    if prefer_dxgi:
        try:
            from core import dxgi_capture
            _dxgi_ok, _why = dxgi_capture.available()
            if _dxgi_ok:
                img = dxgi_capture.grab_rect(rect)
                if img is not None:
                    _NOTE["how"] = "dxgi"
                    return img
        except Exception:
            _dxgi_ok = False

    # 1) WGC：走 DWM 合成层（⚠ **按单个窗口**采 ✗ 盖在它上面的窗口不进画面 ✗）
    #    ⭐⭐ **只在 DXGI 用不了时才试它**（2026-10-10 ✓）：两条路的"画面含义"不一样 ✗
    #      —— DXGI/BitBlt 给的是**合成结果** ✓、WGC 给的是**那一个窗口** ✗ ⇒ 同一路流里
    #      混着来会出现"这一帧有外接弹窗、下一帧没有"✗（断线重连 / 测谎会被这种抖动骗到 ✓）。
    #      ⇒ 既然 DXGI 又快又是合成结果 ✓，那只要它在，就**不用** WGC ✓
    #        （用户选 `wgc` 也只是"允许用 WGC"✓ —— DXGI 不可用的机器上才会走到这里 ✓）。
    if use_wgc and not _dxgi_ok:
        # ⭐⭐ **被盖住 ⇒ 不试 WGC、直接走整屏合成**（用户 2026-10-10 选的口径 ✓）：
        #   这一档是**单窗口**采集 ✗ ⇒ 盖在源窗口上的外接登录窗 / 断线提示框**不进画面** ✗
        #   ⇒ 断线重连、测谎判不出界面 ✓（用户 258 条那次的原话就是这个 ✓）。
        _cover, _cover_why = covered_by(rect)
        if _cover:
            _COVER["last"] = time.monotonic()
        # 迟滞：**进去立刻、出来要等** ✓（见 `_COVER` 上面那段实测说明 ✓）
        _hold = (time.monotonic() - _COVER["last"]) < COVER_HOLD_S
        if _cover or _hold:
            _NOTE["how"] = "screen"
            _NOTE["why"] = _cover_why or (
                "刚刚还在被别的窗口盖住 ⇒ 这一拍**继续抓整屏**（防抖 ✓ 不翻来翻去 ✗）")
        else:
            try:
                from core import wgc_capture
                if wgc_capture.available():
                    img = wgc_capture.grab_rect(rect)
                    if img is not None:
                        _NOTE["how"] = "wgc"
                        return img
            except Exception:
                pass

    # 2) win32 BitBlt（快）
    try:
        img = grab_rect_fast(rect)
        _NOTE["how"] = _NOTE["how"] or "bitblt"
        return img
    except Exception:
        # 3) PIL ImageGrab（最慢，兜底）
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
        _NOTE["how"] = _NOTE["how"] or "pil"
        return np.array(img)[:, :, ::-1].copy()      # PIL 是 RGB，OpenCV 要 BGR


# ══════════════════════════════════════════════════════════════
# 采集任务
# ══════════════════════════════════════════════════════════════
def _sleep_to(t0, n, interval):
    """睡到「第 n 帧应该在的时刻」。落后了就立刻返回，不补偿。"""
    d = t0 + n * interval - time.perf_counter()
    if d > 0:
        time.sleep(d)


def _small(gray_src):
    return cv2.cvtColor(cv2.resize(gray_src, (80, 45)),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)


def run_capture(params, ctx=None):
    """实时抓取窗口画面并存成 PNG。

    params:
        rect     (x, y, w, h)，也可传 "x,y,w,h" 字符串
        out      输出目录
        fps      每秒抓几帧
        seconds  持续秒数，0 = 一直抓到取消
        dedup    去重阈值（与上一张已保存帧的平均像素差），0 = 不去重
        limit    最多张数，0 = 不限
        prefix   文件名前缀
    """
    ctx = ctx or TaskContext()

    rect = params.get("rect")
    if isinstance(rect, str):
        try:
            rect = tuple(int(v) for v in rect.split(","))
        except Exception:
            raise ValueError('区域格式应为 "x,y,w,h"，收到: %s' % rect)

    if not rect or len(rect) != 4 or int(rect[2]) <= 0 or int(rect[3]) <= 0:
        raise ValueError("窗口区域无效 —— 请先选择窗口")
    rect = tuple(int(v) for v in rect)

    out = Path(params["out"])
    out.mkdir(parents=True, exist_ok=True)

    fps = max(0.5, float(params.get("fps", 5.0)))
    seconds = float(params.get("seconds", 120.0))
    dedup = float(params.get("dedup", 0.0))
    limit = int(params.get("limit", 0))
    prefix = params.get("prefix", "frame")

    old = list(out.glob(prefix + "_*.png"))
    if old:
        ctx.log("输出目录已有 %d 张同名文件，将被覆盖" % len(old), "warn")

    ctx.log("抓取区域 %d,%d  %d×%d" % (rect[0], rect[1], rect[2], rect[3]))
    ctx.log("频率 %.1f fps   时长 %s   去重 %s"
            % (fps, ("%.0f 秒" % seconds) if seconds > 0 else "不限（手动停）",
               ("%.3f" % dedup) if dedup > 0 else "关"))

    interval = 1.0 / fps
    t0 = time.perf_counter()
    saved = skipped = 0
    last = None
    canceled = False

    while True:
        if ctx.canceled():
            canceled = True
            break

        elapsed = time.perf_counter() - t0
        if seconds > 0 and elapsed >= seconds:
            break
        if limit and saved >= limit:
            break

        try:
            frame = grab_rect(rect)
        except Exception as e:
            ctx.log("抓帧失败: %s: %s" % (type(e).__name__, e), "error")
            time.sleep(0.3)
            continue

        if dedup > 0:
            cur = _small(frame)
            if last is not None and float(np.abs(cur - last).mean()) / 255.0 < dedup:
                skipped += 1
                _sleep_to(t0, saved + skipped, interval)
                continue
            last = cur

        imwrite(out / ("%s_%05d.png" % (prefix, saved)), frame)
        saved += 1

        if saved % 10 == 0:
            ctx.progress(int(seconds), int(seconds), "已抓 %d 张" % saved) \
                if seconds > 0 else ctx.progress(saved, 0, "已抓 %d 张" % saved)
            ctx.log("  已抓 %d 张（去重跳过 %d）  用时 %.0fs"
                    % (saved, skipped, elapsed))

        _sleep_to(t0, saved + skipped, interval)

    dt = time.perf_counter() - t0
    summary = "抓取 %d 张" % saved
    if skipped:
        summary += "（去重跳过 %d）" % skipped
    if canceled:
        summary += "  [已取消]"

    ctx.log(summary, "ok" if not canceled else "warn")
    if not saved:
        ctx.log("一张都没抓到 —— 检查窗口是否被最小化，或区域坐标是否正确", "warn")

    ctx.progress(saved, saved, summary)
    return {
        "frames": saved,
        "skipped": skipped,
        "seconds": dt,
        "out_dir": str(out),
        "summary": summary,
    }
