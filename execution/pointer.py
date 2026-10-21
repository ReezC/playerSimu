# -*- coding: utf-8 -*-
"""指针：读位置 / 移动 / 轮询线程（**不负责发键** ✓）。

从 `execution/input.py` 拆出来的（2026-10-17 ✓ 第 1 波 ✓）：
原来"发键 + 指针 + 远端 relay"挤在一个 983 行的文件里 ✗，
其中指针这块只依赖 `win32.py` 与 `perf` ✓ 是最好拆的一块 ✓。
"""
import ctypes
import threading
import time

from ctypes import wintypes

from core import perf
from execution.win32 import user32 as _user32




# ---- ⭐⭐ 指针跟踪：触控板位移的**产生**侧（用户 2026-10-05 ✓ 原话："直接2"）----
#
# **为什么要这一块**：触控板的位移原来只在 `TouchPad.mouseMoveEvent` 里产生 ✗（Qt 的
#   move 事件 ⇒ 跑在 GUI 主线程上）—— 主线程一被重绘 / 主回路拖住，Qt 就会把连续移动
#   **合并**成一次。实测（`perf.log` 2026-10-05 18:40 那段，按 F10 滑了两下）：
#     `move_gap_ms` 中位 20 / p95 76 / p99 201 / **最大 467 ms**
#     `move_px`                             **最大 514**（一次跳半个屏幕 ✓）
#   ⇒ A 机看到的就是"停一下、猛跳一下" ✓；而同一段里 `send_ms` 中位才 **0.15 ms**、
#   `kbd_pending` 最大 4 ⇒ **瓶颈既不在链路也不在发送，在位移的产生** ✓
#   （发送那条早已挪出主线程 ✓ 见 `gui/player_panel.py::_PadSender`）。
# ⇒ 现在由下面这条**独立线程**按固定节拍读系统指针、算位移 ⇒ 节拍只受"线程能不能被调度"
#   影响，与 GUI 卡不卡无关 ✓。

#: 轮询节拍（秒）：250 Hz（比触控板本身的事件率还高 ⇒ 不丢节拍 ✓）。
#: ⚠ 能不能真的到 4 ms 取决于**系统定时器精度**：Windows 默认 ~15.6 ms ⇒ 实际约 64 Hz，
#:   但**仍然均匀** ✓ —— 而"均匀"正是要害（用户要的不是"更快"，是"别一顿一顿" ✓）。
#:   本仓库只在 `core/winperf.py` **一处**声明精度（工作台启动时按「性能保活」设置应用 ✓，
#:   实测见 `tools/selftest_winperf.py` ✓）⇒ **这里绝不自己再调 `timeBeginPeriod`** ✗
#:   （同一件事只该有一处实现 ✓）。实际节奏看打点 **`pad_poll_gap_ms`** ✓。
TRACK_INTERVAL_S = 0.004




def pointer_pos():
    """当前指针位置（**物理像素**，Win32 屏幕坐标 ✓）；读不到返回 None ✓。"""
    try:
        pt = wintypes.POINT()
        if _user32.GetCursorPos(ctypes.byref(pt)):
            return (int(pt.x), int(pt.y))
    except Exception:                   # noqa: BLE001 —— 读不到就当"这拍没数据"✓ 不炸 ✓
        pass
    return None




def pointer_warp(x, y):
    """把指针摆到 (x, y)（**物理像素** ✓）。返回是否成功 ✓。

    ⚠ **会失败，而且是"无声失败"**：本机实测（2026-10-05）在**锁屏 / 非交互桌面**下
      `SetCursorPos` 返回 0、指针纹丝不动，而 `GetLastError` **还是 0** ✗（查不出原因 ✓）
      —— 这不是缺陷，是桌面的规矩 ✓。所以调用方必须**按返回值行事**、不能想当然 ✓
      （`PointerTracker` 因此用 `pos - 上一次` 算位移 ✓：拨回成不成功都不影响位移正确 ✓）。
    """
    try:
        return bool(_user32.SetCursorPos(int(x), int(y)))
    except Exception:                   # noqa: BLE001
        return False




class PointerTracker:
    """⭐ 按**固定节拍**轮询系统指针、算位移的线程（用户 2026-10-05 ✓ "直接2"）。

    它干的活（每拍）：读位置 → 减掉**上一次读到的位置**得位移 → 位移非零就
      `on_delta(dx, dy)` → 把指针 `SetCursorPos` 拨回钉点（钉点 = 进触控模式时的板心 ✓）。

    ⚠⚠ 三条纪律（违反了就是另一个 bug ✗）：
      · **只用 Win32**（`GetCursorPos` / `SetCursorPos`）—— **绝不碰 Qt** ✗：
        `QCursor.setPos()` 只许在 GUI 线程调，而这条线程不是 ✓；
      · `on_delta` **在这条线程里被调** ⇒ 回调只许做线程安全的事（加法 / 入队 / 置事件 ✓），
        **绝不许碰界面** ✗ —— 要通知界面就走 Qt 信号（那是排队的 ✓ 见 `gui/touchpad.py`）；
      · 位移用 **`pos - 上一次`**，**不是** `pos - 钉点` ✗ —— 后者在"拨回失败 / 指针被别的
        东西挪走"那一下会**算出一大跳**（而且之后一路算 0 ✗）；前者不管拨回成不成功，
        都只算"这一拍真的动了多少" ✓。

    ⚠ 钉点是**物理像素**（`GetCursorPos` 的原生坐标 ✓）⇒ 高分屏下与 Qt 的逻辑坐标差一个
      比例 ⇒ 位移由调用方除回去（`scale` ✓ = `devicePixelRatio` ✓）；钉点本身不用换算 ✓
      （它来自 `GetCursorPos` 自己 ✓）。
    """

    def __init__(self, ref, on_delta, scale=1.0, interval=TRACK_INTERVAL_S):
        self.ref = (int(ref[0]), int(ref[1]))
        self.on_delta = on_delta
        self.scale = max(1e-6, float(scale))
        self.interval = max(0.001, float(interval))
        self.n_poll = 0          # 轮询拍数（自检/诊断用 ✓）
        self.n_delta = 0         # 其中有位移的拍数 ✓
        self._stop = False
        self._th = None

    def start(self):
        """起线程。返回 True/False：`GetCursorPos` 读不到 ⇒ False（调用方自己给提示 ✓）。"""
        if pointer_pos() is None:
            return False
        self._th = threading.Thread(target=self._run, daemon=True, name="pad-track")
        self._th.start()
        return True

    def stop(self, timeout=0.3):
        """收工：置停止 + **等线程真的退出**（别留一条守护线程在后台摸指针 ✗）。"""
        self._stop = True
        th = self._th
        self._th = None
        if th is not None:
            th.join(timeout)             # 一拍之内必退（interval 只有几 ms ✓）

    def _run(self):
        last = None                      # 上一次读到的位置（`pos - last` ✓ 见纪律第三条）
        last_t = None
        while not self._stop:
            pos = pointer_pos()
            if pos is not None:
                now = time.perf_counter()
                if last_t is not None:
                    perf.sample("pad_poll_gap_ms", (now - last_t) * 1000.0)
                last_t = now
                self.n_poll += 1
                if last is not None:
                    dx = pos[0] - last[0]
                    dy = pos[1] - last[1]
                    if dx or dy:
                        self.n_delta += 1
                        perf.sample("pad_poll_px",
                                    (abs(dx) + abs(dy)) / self.scale)
                        try:
                            self.on_delta(dx / self.scale, dy / self.scale)
                        except Exception:       # noqa: BLE001 —— 回调坏了别把跟踪线程弄死 ✗
                            pass
                        # ⚠⚠ 拨回成功 ⇒ **指针此刻就在钉点** ⇒ `last` 必须跟着改成钉点 ✗：
                        #   不改，下一拍就会把"**我们自己拨回去**"读成一次**反向位移** ✗
                        #   （(-dx,-dy) ⇒ 远程鼠标每拍抖一下 ✓ 实测就是这么抓出来的 ✓
                        #    用例里假指针会立刻跟到钉点 ✓ 所以它是**确定的红**、不是碰运气 ✓）。
                        # ⚠ 拨回可能失败（锁屏 / 非交互桌面：返回 0 且 `GetLastError` 也是 0 ✓
                        #   见 `pointer_warp` 的说明 ✓）—— **失败不算错**（位移照旧正确 ✓ 因为
                        #   用的是 `pos - 上一次` ✓），但要**留痕**：否则"指针跑掉了"查不出来 ✗
                        if pointer_warp(self.ref[0], self.ref[1]):
                            pos = self.ref
                        else:
                            perf.count("pad_warp_fail")
                last = pos
            time.sleep(self.interval)
