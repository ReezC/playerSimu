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


def grab_rect(rect, use_wgc=True):
    """抓屏幕上指定的矩形，返回 BGR ndarray。

    优先 WGC（DWM 合成层，不经过 GDI BitBlt），失败回退 win32 BitBlt，
    再失败回退 PIL ImageGrab。use_wgc=False 时跳过 WGC（抓整个虚拟屏幕等
    跨窗口场景，WGC 单窗口采集会 clamp 错）。
    """
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        raise ValueError("矩形无效: %s" % (rect,))

    # 1) WGC：走 DWM 合成层，反作弊更难检测，且能抓被遮挡窗口
    if use_wgc:
        try:
            from core import wgc_capture
            if wgc_capture.available():
                img = wgc_capture.grab_rect(rect)
                if img is not None:
                    return img
        except Exception:
            pass

    # 2) win32 BitBlt（快）
    try:
        return grab_rect_fast(rect)
    except Exception:
        # 3) PIL ImageGrab（最慢，兜底）
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
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
