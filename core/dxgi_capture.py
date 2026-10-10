"""DXGI **桌面复制**抓屏（Windows 上最快的那条路，`dxcam` 封装 ✓）。

**为什么加它**（2026-10-10 ✓ 用户口径："我们当然选用**性能最好**的、能满足**截取部分屏幕**需求的"✓）：
本机实测（`data/_bench_dxgi.py` ✓ 用完即删 ✓，1920×1080）：

| 抓法 | 整屏 | 960×540 | 240×79 |
|---|---|---|---|
| **DXGI（本模块 ✓）** | **16.9 ms** | **5.7 ms** | 全 None（那块画面没变 ✓） |
| BitBlt（原来那条 ✓） | 50.5 ms | 23.4 ms | 9.1 ms |

⇒ 整屏快 **约 3 倍**、中等框快 **约 4 倍** ✓；原来那条在 15 fps 下就吃掉 **67%** 的预算 ✗。

**它和我们已有的两档的关系**（不是替代 ✓ 是"多一档、最快那档"✓）：
  · **DXGI 桌面复制**（本模块）：读的是 **DWM 合成结果**（= 屏幕上看到什么就有什么 ✓
    ⇒ **外接登录窗 / 断线提示框照样在画面里** ✓✓ 断线重连、测谎都靠这个 ✓）；
  · **BitBlt 桌面 DC**（`wincap.grab_rect_fast` ✓）：同样读合成结果 ✓ 只是慢 ✓ ⇒ 作**兜底** ✓；
  · **WGC**（`core/wgc_capture.py` ✓）：**按单个窗口**采 ✗（盖在它上面的窗口不进画面 ✗）⇒
    只在"只想看游戏本体"时才该选它 ✓ 见 `gui/live_panel.py`「抓屏」那格 tooltip ✓。

⚠ **两条使用纪律**：
  1. DXGI **只在画面有变化时给新帧** ⇒ `grab_rect` 返回 `None` 是**正常的**（不是错误 ✓）
     调用方要**沿用上一帧** ✓（正好也更省 CPU ✓）；
  2. `dxcam` 的对象**有线程亲和**（COM ✗）⇒ 这里**每线程一份**（`threading.local` ✓），
     谁在用就在哪个线程建 ✓ 别跨线程共享 ✗。
"""

import threading

_lock = threading.local()
_state = {"ok": None, "why": ""}


def available():
    """这台机器能不能用 DXGI（`dxcam` 装了 + 建得起来）⇒ `(bool, 人话)`。结果缓存 ✓。"""
    if _state["ok"] is not None:
        return _state["ok"], _state["why"]
    try:
        import dxcam  # noqa: F401
    except Exception as e:                       # noqa: BLE001 —— 没装 ✓ 正常走兜底 ✓
        _state["ok"], _state["why"] = False, "没装 dxcam（%s）" % type(e).__name__
        return _state["ok"], _state["why"]
    cam = _cam(create=True)
    if cam is None:
        _state["ok"], _state["why"] = False, "dxcam 建不起来（独占全屏 / 无显示器？）"
    else:
        _state["ok"], _state["why"] = True, ""
    return _state["ok"], _state["why"]


def _cam(create=False):
    """取**本线程**那台相机；`create=True` 时没有就建一台 ✓（失败给 None ✓）。"""
    cam = getattr(_lock, "cam", None)
    if cam is not None or not create:
        return cam
    try:
        import dxcam
        cam = dxcam.create(output_idx=0, output_color="BGR")
    except Exception:                            # noqa: BLE001 —— 建不起来就照实说 ✓
        cam = None
    _lock.cam = cam
    return cam


def grab_rect(rect):
    """抓屏幕矩形 `(x, y, w, h)` → BGR ndarray；**画面没变 / 不可用 ⇒ None** ✓。

    ⚠ `None` 不是错误 ✓（见模块头第 1 条 ✓）：调用方沿用上一帧即可 ✓。
    """
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        return None
    cam = _cam(create=True)
    if cam is None:
        return None
    try:
        # dxcam 的 region 口径是 (left, top, right, bottom) ✓ 别搞混（我们对外是 x,y,w,h ✓）
        img = cam.grab(region=(x, y, x + w, y + h))
    except Exception:                            # noqa: BLE001 —— 抓失败就当"这拍没有" ✓
        return None
    if img is None:
        return None
    # dxcam 在 region 下偶尔给"未裁"的整屏（不同版本行为不一致 ✓）⇒ 自己再裁一刀保险 ✓
    if img.shape[0] != h or img.shape[1] != w:
        ox, oy = x, y
        ih, iw = img.shape[:2]
        # 只有在"给的是整屏"时才按原坐标裁 ✓（否则宁可不裁，别裁错 ✗）
        if iw >= ox + w and ih >= oy + h:
            img = img[oy:oy + h, ox:ox + w]
        else:
            return None
    import numpy as np
    return np.ascontiguousarray(img)


def release():
    """放掉本线程那台相机（换图 / 退出时调 ✓ 失败不抛 ✓）。"""
    cam = getattr(_lock, "cam", None)
    if cam is None:
        return
    try:
        cam.release()
    except Exception:                            # noqa: BLE001
        try:
            del cam
        except Exception:                        # noqa: BLE001
            pass
    _lock.cam = None
