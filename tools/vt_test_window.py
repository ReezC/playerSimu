# -*- coding: utf-8 -*-
"""测谎自动跟踪 · **独立测试窗**（用户 2026-10-17 ✓ 原话：
「**我们不用需要授权的版本，我们只用它的功能做一个测试版的自己的窗口**」✓）。

它是什么：
  · 一个**单窗口、无授权、无账号**的试验台：抓屏幕一块 → 喂 `perception.vt_sdk`
    （= 我们自己的 `visual_tracking_sdk_20260920` 薄封装 ✓）→ 画 ROI / 目标框 / 瞄点
    → **可选**把真实光标移过去；
  · **默认不移动鼠标** ✓（勾上才动 ✓ 且左键按住 = 人工接管、立刻让路 ✓ 同 SDK 示例口径）；
  · 支持**视频回放**：没有游戏窗口也能验"锁定 / 跟丢 / 成功"这些状态 ✓。

它不是什么（别记错 ✗）：
  · 不是主程序的一部分（不进工作台、不写 `config/*.yaml`、不发串口 ✓）；
  · 不是那个「冒险岛测谎.exe」（那个按机器码授权 ✓ 未授权只能看到验证窗 ✓）；
  · **不改 SDK 目录里任何文件** ✓（`vt_sdk` 顶部那条纪律 ✓ 只把根目录塞进 `sys.path` ✓）。

用法：
    python -m tools.vt_test_window                        # 开窗，先框选区域
    python -m tools.vt_test_window --rect 0 0 1280 720    # 直接给屏幕矩形
    python -m tools.vt_test_window --rect 0 0 1280 720 --move   # 允许移动真实光标
    python -m tools.vt_test_window --video D:/x.mp4       # 视频回放（不会动鼠标）
    python -m tools.vt_test_window --rect 0 0 800 600 --content # 矩形已是内容区（跳过定位）

⚠ 想看"它到底有没有在动"就盯这三处：状态行的 `阶段 / 推理ms / 瞄点`、
  画面上的**品红目标框**、以及勾上「移动鼠标」之后光标有没有跟上。
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

#: 允许 `python tools/vt_test_window.py` 直接跑（不必先 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ⚠⚠ **必须在 import PyQt5 之前**把系统那份 MSVC 运行库预装进进程 ✓
#   （`core/win_dlls.preload_msvc_runtime` ✓ 见 `gui/app.py` 那段说明 ✓）
#   不做的后果（本窗第一次跑就当场踩中 ✓）：
#     `PyQt5/Qt5/bin` 自带一套旧 `msvcp140.dll` ⇒ 它先落进进程 ⇒ 之后
#     **`import torch` 报 `WinError 1114`（DLL 初始化例程失败）** ✓ ⇒
#     SDK 的 `[liescd] 本机模型 FAILED (CPU)` ✓ ⇒ **看着像"没有目标/不跟踪"** ✗
#     （实际是模型压根没加载 ✓ 极容易被当成算法问题 ✗）。
from core.win_dlls import preload_msvc_runtime as _preload_msvc   # noqa: E402
try:
    _preload_msvc()
except Exception:                            # noqa: BLE001 —— 预载失败不拦启动 ✓
    pass

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal     # noqa: E402
from PyQt5.QtGui import QImage, QPixmap                     # noqa: E402
from PyQt5.QtWidgets import (QApplication, QCheckBox, QFileDialog,  # noqa: E402
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)

from perception.vt_sdk import VtLieSdk, frame_rect_of      # noqa: E402

#: 抓帧节拍上限（Hz ✓）。SDK 实测 `process` 中位 37~39ms、p90 172ms ✓
#: ⚠ 从 20 提到 30（用户 2026-10-17 ✓「它好像特别的慢」）：录像多是 30fps ✓
#:   而"一拍一帧"时 20Hz 只能播到 **0.67 倍速** ✗ —— 那是**回放慢**，不是算法慢 ✓
#:   （真正跑不动时循环会自己降下来 ✓ 因为一拍要等 `tick()` 返回 ✓）。
MAX_FPS = 30

# ---- 叠加颜色（BGR ✓ 与实时预览那套同源：橙黄 ROI / 品红目标框 ✓）----
C_ROI = (0, 200, 255)
C_BOX = (255, 0, 255)
C_AIM = (0, 255, 255)
C_TXT = (255, 255, 255)


def _set_dpi_aware():
    """**必须先做**：不声明 DPI 感知的话，`SetCursorPos` 拿到的物理坐标会被系统再缩放一次
    ⇒ 表现是"框对了、鼠标却总差一截"（按 100% 缩放开发时还看不出来 ✗）。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:                       # noqa: BLE001 —— 老系统没 shcore ✓
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:                   # noqa: BLE001
            pass


def _left_down():
    """左键是不是按着（**人工接管**的判据 ✓ 同 SDK 示例 ✓）。"""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)
    except Exception:                       # noqa: BLE001
        return False


def _move_cursor(x, y):
    """把真实光标移到屏幕坐标（物理 ✓）⇒ `bool`（成功与否 ✓）。"""
    if _left_down():                        # 人工接管 ⇒ 一律让路 ✓
        return False
    try:
        return bool(ctypes.windll.user32.SetCursorPos(int(x), int(y)))
    except Exception:                       # noqa: BLE001
        return False


class Runner:
    """抓帧 → 喂 SDK → 留一份"最新结果"给界面画（**不用信号传图** ✓ 免得排队 ✗）。

    ⚠ 为什么是"留最新"而不是 emit：图像 1920×1080×3 ≈ 6MB/帧 ✓
      跨线程 emit 会进事件队列 ⇒ 界面一慢就积压 ⇒ 越看越延迟 ✗（同 `LiveThread` 的口径 ✓）。
    """

    def __init__(self, vt: VtLieSdk, rect=None, video=None, allow_move=False,
                 content=False):
        self.vt = vt
        self.rect = tuple(int(v) for v in rect) if rect else None
        self.video = str(video) if video else ""
        self.allow_move = bool(allow_move)
        self.content = bool(content)
        self.running = False
        self.moved = 0                        # 真正发出去的鼠标指令数（诊断用 ✓）
        self.moved_blocked = 0                # 被"人工接管"挡掉的次数 ✓
        self._t_stop = 0.0
        self._last = None                     # 最新一帧的结果（dict ✓ 见 `snapshot` ✓）
        self._cap = None
        self._fps_hist = []
        #: 视频回放的"按真实时间走"用（见 `_grab` ✓）：片子的 fps / 开播时刻 / 已读到第几帧
        self._vid_fps = 0.0
        self._t_vid0 = None
        self._vid_i = 0

    # ---------------- 生命周期 ----------------
    def start(self):
        self.running = True
        self.moved = 0
        self.moved_blocked = 0
        self._t_stop = 0.0
        # 每次开播都从"第 0 帧 / 此刻"重新对时（见 `_grab` 的真实时间回放 ✓）
        self._vid_fps = 0.0
        self._t_vid0 = None
        self._vid_i = 0
        if self.video:
            import os

            import cv2
            # ⚠ 先分辨"文件没了"和"解码不了"（本窗第一次冒烟就撞上：用的是一份已经被删的录像 ✓
            #   ⇒ 只报"视频打不开"会让人去查解码器 ✗ 白查半天 ✓）。
            if not os.path.isfile(self.video):
                self.running = False
                return "视频不存在（可能已经被删/改名）：%s" % self.video
            self._cap = cv2.VideoCapture(self.video)
            if not self._cap.isOpened():
                self._cap = None
                self.running = False
                return "视频打不开（解码器/编码不支持）：%s" % self.video
        # 内容区模式 ⇒ 直接告诉 SDK "整幅就是内容区"（省掉双锚点定位 ✓ 见 SDK 文档 ✓）
        if self.content and self.rect:
            self.vt.reset(roi=(0, 0, self.rect[2], self.rect[3]))
        else:
            self.vt.reset()
        return ""

    def stop(self):
        self.running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:                # noqa: BLE001
                pass
            self._cap = None

    # ---------------- 抓帧 ----------------
    def _grab(self):
        """⇒ `(frame_bgr, origin, kind)`：`frame_bgr=None` ⇒ 这拍没新画面（沿用上一帧 ✓）。

        ⭐ **视频按真实时间走**（用户 2026-10-17 ✓「它好像特别的慢」）：老写法"一拍读一帧"⇒
          喂帧上限 20Hz 而录像 30fps ⇒ **回放只有 0.67 倍速** ✓ ⇒ 看着就是"特别慢"✗
          （而且动作变慢会让跟踪显得更迟钝 ✓ 两件事被混成一件 ✗）。
          现在按"开播到现在该看到第几帧"来跳帧：落后就 `grab()` 丢几帧 ✓ 追到当前时刻 ✓
          （**不是**把落后的帧排队补上 ✗ 那会越积越多 ✓ 同实时那套口径 ✓）。
        """
        if self._cap is not None:
            import cv2
            now = time.monotonic()
            if not self._vid_fps:
                self._vid_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
            if self._t_vid0 is None:
                self._t_vid0 = now
                self._vid_i = 0
            want = int((now - self._t_vid0) * self._vid_fps)
            _skip = 0
            while self._vid_i < want - 1 and _skip < 60:
                if not self._cap.grab():
                    return None, (0, 0), "eof"
                self._vid_i += 1
                _skip += 1
            ok, frame = self._cap.read()
            if not ok:
                return None, (0, 0), "eof"
            self._vid_i += 1
            return frame, (0, 0), "video"
        from core import wincap
        img = wincap.grab_rect(self.rect)
        return img, (self.rect[0], self.rect[1]), "screen"

    # ---------------- 一拍 ----------------
    def tick(self):
        """跑一拍（**在工作线程里跑** ✓ 见 `TickThread`）。返回"本轮是否结束"的一句话。

        ⚠ 为什么从"界面定时器直接调"改成"工作线程"（用户 2026-10-17 ✓ 原话：
          「**它好像特别的慢**」✗）：抓屏 + 推理 + 绘制**串在主线程上**时，
          一帧要 40~170ms（SDK 中位 37ms、p90 172ms ✓）⇒ 这期间窗口**收不到任何事件**
          ⇒ 拖不动、按钮没反应、画面一顿一顿 ✓ —— 观感就是"特别慢"✗
          （实际帧率可能还有 10~15fps ✓ 是**界面被堵住**了 ✓ 两者容易混 ✗）。
        """
        if not self.running:
            return ""
        t0 = time.monotonic()
        frame, origin, kind = self._grab()
        grab_ms = (time.monotonic() - t0) * 1000.0
        if kind == "eof":
            self.running = False
            return "视频播完了"
        if frame is None:
            # DXGI 没变化 ⇒ 没有新帧；**不喂**（喂同一帧会让 SDK 以为目标静止 ✓ 反而伤跟踪 ✗）
            return ""
        obs = self.vt.feed(frame, t0, origin)
        if obs is None:
            return ""

        self._last = self._pack(frame, obs, t0)
        self._last["grab_ms"] = grab_ms

        # ---- 鼠标：**默认不动** ✓；勾了才动，且左键按住一律让路 ✓ ----
        if self.allow_move and kind == "screen" and self._last["aim_screen"]:
            if _left_down():
                self.moved_blocked += 1
            elif _move_cursor(*self._last["aim_screen"]):
                self.moved += 1

        phase = self._last["phase"]
        if phase in ("SUCCESS", "ENDED"):
            self.running = False
            why = getattr(obs, "why", "") or ""
            return "本轮结束：%s%s" % (phase, ("（%s）" % why) if why else "")
        return ""

    def _pack(self, frame, obs, t0):
        """把 `obs` 里**界面要用的那几样**抠出来（**别存 obs 本身** ✗：SDK 可能复用对象 ✓）。"""
        box = frame_rect_of(obs)
        aim = getattr(obs, "aim_frame", None)
        aim_s = getattr(obs, "aim_screen", None)
        roi = getattr(obs, "roi", None)
        d = {
            "frame": frame,                        # 只读（画的时候先 copy ✓）
            "phase": str(getattr(obs, "phase", "-")),
            "roi": tuple(int(v) for v in roi) if roi else None,
            "box": tuple(int(round(float(v))) for v in box) if box else None,
            "aim": (int(aim[0]), int(aim[1])) if aim is not None else None,
            "aim_screen": (float(aim_s[0]), float(aim_s[1])) if aim_s is not None else None,
            "ms": (time.monotonic() - t0) * 1000.0,
            "frames": int(getattr(self.vt, "frames", 0)),
        }
        self._fps_hist.append(t0)
        if len(self._fps_hist) > 40:
            self._fps_hist.pop(0)
        span = self._fps_hist[-1] - self._fps_hist[0] if len(self._fps_hist) > 1 else 0.0
        d["fps"] = (len(self._fps_hist) - 1) / span if span > 0.05 else 0.0
        return d

    def snapshot(self):
        return self._last


class TickThread(QThread):
    """在**工作线程**里跑 `Runner.tick()`（抓帧 + 推理 + 送鼠标 ✓）。

    ⭐ 为什么必须有它（用户 2026-10-17 ✓「它好像特别的慢」）：这三件事**一件都不该在
      界面线程里做** ✓ —— SDK 单帧中位 37ms、p90 172ms ✓，串在界面线程上时这段时间
      **窗口一点事件都收不到** ⇒ 拖动/按钮全卡 ✓（观感"特别慢"✗，实际帧率并不低 ✓）。
    它只负责跑；**画**交给界面那边的 `_paint()`（每 33ms 拿一次"最新那份" ✓
    拿不到新的就重画旧的 ✓ 不排队 ✓ —— 同 `LiveThread` 的口径 ✓）。
    """

    done = pyqtSignal(str)          # 本轮结束（SUCCESS/ENDED/视频播完）⇒ 界面收尾 ✓

    def __init__(self, runner, max_fps=MAX_FPS, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.max_fps = max(1, int(max_fps))
        self._stop = False

    def stop(self):
        """叫停 —— **只置标志，不 join**（join 由调用方决定要不要等 ✓）。"""
        self._stop = True
        self.runner.running = False

    def run(self):
        period = 1.0 / float(self.max_fps)
        while not self._stop and self.runner.running:
            t0 = time.monotonic()
            msg = ""
            try:
                msg = self.runner.tick()
            except Exception as e:                  # noqa: BLE001 —— 线程里炸了要说出来 ✓
                msg = "出错了：%s: %s" % (type(e).__name__, e)
            if msg:
                self.done.emit(msg)
                return
            left = period - (time.monotonic() - t0)
            if left > 0:
                time.sleep(left)


class VtTestWindow(QMainWindow):
    """测试窗本体：上面参数、中间画面、下面状态。"""

    def __init__(self, rect=None, video="", allow_move=False, content=False,
                 auto=False):
        super().__init__()
        self.setWindowTitle("测谎自动跟踪 · 测试窗（不移动鼠标 = 默认）")
        self.resize(1100, 760)

        self.vt = VtLieSdk(confidence=0.15, region="CN", success_detection=not content)
        self.runner = Runner(self.vt, rect=rect, video=video,
                             allow_move=allow_move, content=content)
        self.rect = tuple(rect) if rect else None
        self.video = video
        #: **画**的定时器（33ms ⇒ 30fps ✓）—— 只做"把最新那份画出来"，**不抓帧、不推理** ✓
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._paint)
        self._load_timer = QTimer(self)
        self._load_timer.timeout.connect(self._poll_ready)
        #: **跑**的线程（抓帧 + 推理 + 送鼠标 ✓ 见 `TickThread`）
        self._thread = None
        #: 已经画过的帧号（同样就不重画 ✓）/ 绘制耗时的滑动平均
        self._painted_frames = -1
        self._draw_ema = 0.0

        self._build_ui()
        if allow_move:
            self.chk_move.setChecked(True)
        self._refresh_src_label()
        if auto:
            #: `--auto`：开窗就自己按「开始」（**冒烟/录屏用** ✓ 省得手点 ✓）
            QTimer.singleShot(400, self._toggle)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QWidget()
        lay = QVBoxLayout(root)

        row1 = QHBoxLayout()
        self.btn_pick = QPushButton("框选区域…")
        self.btn_pick.setToolTip("在屏幕上拖一个矩形（**游戏窗口里的内容区**最好）\n"
                                 "⚠ 想省掉定位那一步就勾「矩形已是内容区」")
        self.btn_pick.clicked.connect(self._pick_rect)
        row1.addWidget(self.btn_pick)

        self.btn_video = QPushButton("打开视频…")
        self.btn_video.setToolTip("没有游戏窗口时用它验状态机（**不会移动鼠标** ✓）")
        self.btn_video.clicked.connect(self._pick_video)
        row1.addWidget(self.btn_video)

        self.chk_content = QCheckBox("矩形已是内容区")
        self.chk_content.setToolTip("勾上 ⇒ 跳过双锚点定位（`reset(roi=全矩形)` ✓）；\n"
                                    "不勾 ⇒ SDK 自己在画面里找 ROI（`LOCATING`）")
        self.chk_content.setChecked(bool(self.runner.content))
        row1.addWidget(self.chk_content)

        self.chk_move = QCheckBox("移动鼠标")
        self.chk_move.setToolTip("**默认关** ✓\n"
                                 "勾上 = 允许 `SetCursorPos` 把真实光标送到瞄点；\n"
                                 "⚠ 按住左键 = 人工接管，本窗立刻让路（不会跟你抢 ✓）")
        self.chk_move.toggled.connect(self._on_move_toggled)
        row1.addWidget(self.chk_move)

        self.chk_draw = QCheckBox("显示叠加")
        self.chk_draw.setChecked(True)
        row1.addWidget(self.chk_draw)
        row1.addStretch(1)

        self.btn_start = QPushButton("开始")
        self.btn_start.clicked.connect(self._toggle)
        row1.addWidget(self.btn_start)

        self.btn_reset = QPushButton("结束本轮")
        self.btn_reset.setToolTip("清身份/轨迹/成功计数 ⇒ 下一轮重新从开场开始（`reset()` ✓）")
        self.btn_reset.clicked.connect(self._reset_round)
        row1.addWidget(self.btn_reset)
        lay.addLayout(row1)

        self.lbl_src = QLabel()
        self.lbl_src.setStyleSheet("color:#5f6368;")
        self.lbl_src.setWordWrap(True)
        lay.addWidget(self.lbl_src)

        self.view = QLabel("（还没有画面 —— 先框选区域或打开视频）")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumHeight(480)
        self.view.setStyleSheet("background:#202124; color:#9aa0a6;")
        lay.addWidget(self.view, 1)

        self.lbl_state = QLabel("未开始")
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setStyleSheet("color:#80868b;")
        lay.addWidget(self.lbl_state)

        self.setCentralWidget(root)

    # ---------------- 源 ----------------
    def _refresh_src_label(self):
        if self.video:
            self.lbl_src.setText("源：**视频** %s" % self.video)
        elif self.rect:
            self.lbl_src.setText("源：屏幕矩形 x=%d y=%d %d×%d（**物理像素** ✓）"
                                 % self.rect)
        else:
            self.lbl_src.setText("源：还没选 —— 点「框选区域…」或「打开视频…」")

    def _pick_rect(self):
        from gui.region_picker import select_screen_region
        r = select_screen_region(self, hide_owner=True)
        if not r:
            return
        self.rect = tuple(int(v) for v in r)
        self.video = ""
        self.runner.rect = self.rect
        self.runner.video = ""
        self._refresh_src_label()

    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mkv)")
        if not p:
            return
        self.video = str(p)
        self.rect = None
        self.runner.video = self.video
        self.runner.rect = None
        self.chk_move.setChecked(False)     # 视频回放移动鼠标没有意义 ✓
        self._refresh_src_label()

    # ---------------- 运行 ----------------
    def _toggle(self):
        if self.runner.running:
            self._pause()
            return
        if not (self.rect or self.video):
            QMessageBox.information(self, "先选源", "先「框选区域…」或「打开视频…」。")
            return
        if not self.vt.ready:
            self.vt.load(blocking=False)     # 权重加载 ~11s ⇒ 后台 ✓ 界面别卡死 ✓
            self._load_timer.start(200)
            self._say("正在预热模型（~11 秒，第一次才有）…")
            return
        err = self.runner.start()
        if err:
            QMessageBox.warning(self, "起不来", err)
            return
        # ⭐ 跑在**工作线程**里（抓帧/推理/送鼠标 ✓），界面只按 30fps 画最新那份 ✓
        self._thread = TickThread(self.runner, MAX_FPS, self)
        self._thread.done.connect(self._on_round_done)
        self._thread.start()
        self.btn_start.setText("暂停")
        self._timer.start(33)
        self._say("跑起来了（鼠标：%s）" % ("会动" if self.chk_move.isChecked() else "不动"))

    def _pause(self):
        """暂停：停线程 + 停重绘（**等它真的退出** ✓ 否则下一次「开始」会两个线程抢同一个会话 ✗）。"""
        self.runner.stop()
        th, self._thread = self._thread, None
        if th is not None:
            th.stop()
            th.wait(1500)
        self._timer.stop()
        self.btn_start.setText("开始")

    def _on_round_done(self, msg):
        """工作线程说"本轮结束"（SUCCESS/ENDED/视频播完/出错）⇒ 界面收尾 ✓。"""
        self._pause()
        self._paint()                    # 把最后一帧留下来（别停在半张 ✓）
        self._say(msg)

    def _poll_ready(self):
        if self.vt.loading:
            return
        self._load_timer.stop()
        if not self.vt.ready:
            self._say("⚠ 模型没加载起来：%s" % (self.vt.error or "原因不明"))
            return
        self._say("预热完成（%.1f 秒）—— 再点「开始」" % (self.vt.load_ms / 1000.0))
        self._toggle()

    def _reset_round(self):
        self._pause()
        self.vt.reset()
        self._say("本轮已结束（身份/轨迹已清 ✓）")

    def _on_move_toggled(self, on):
        """运行中也能切 ✓（改的就是 `runner.allow_move` 这一个值 ✓）。"""
        self.runner.allow_move = bool(on)
        if on and self.video:
            self.chk_move.setChecked(False)
            QMessageBox.information(self, "视频回放不动鼠标",
                                    "视频回放没有真实屏幕坐标，移动鼠标没有意义。")
            return
        self._say("鼠标：**%s**" % ("会动（按住左键立刻让路）" if on else "不动（只预览）"))

    # ---------------- 画（30fps，只画最新那份 ✓） ----------------
    def _paint(self):
        snap = self.runner.snapshot()
        if not snap:
            return
        # ⭐ **没有新帧就不重画**（喂帧 10~30fps，而重绘定时器 30fps ⇒ 省掉三分之一到一半的白工 ✓）
        if int(snap.get("frames", 0)) == self._painted_frames:
            return
        self._painted_frames = int(snap.get("frames", 0))
        t0 = time.monotonic()
        rgb = self._compose(snap)
        h, w = rgb.shape[:2]
        # ⚠ 这里**不再 `QPixmap.scaled`**（用户 2026-10-17 ✓「它好像特别的慢」）：
        #   1080p 每帧 `SmoothTransformation` 要 30~60ms ✗ ⇒ 已在 `_compose` 里按
        #   控件大小 `cv2.resize`（`INTER_AREA`，**先缩再画框** ⇒ 叠加和拷贝都只在小图上 ✓）。
        #   `QPixmap.fromImage` 自己会拷一份 ⇒ `QImage` 那边**不必**再 `.copy()` ✓。
        self.view.setPixmap(QPixmap.fromImage(
            QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)))
        # 用**滑动平均**报（单帧抖动很大：1080p 缩放 ±80ms ✓ 看瞬时值只会让人误判 ✗）
        _ms = (time.monotonic() - t0) * 1000.0
        self._draw_ema = (_ms if not self._draw_ema
                          else self._draw_ema * 0.8 + _ms * 0.2)
        snap["draw_ms"] = self._draw_ema
        self._say(self._state_text(snap))

    def _compose(self, snap):
        """最新一帧 ⇒ **缩到控件大小 + 画叠加 + 转 RGB**（一步到位，别来回拷 ✗）。"""
        import cv2
        frame = snap["frame"]
        fh, fw = frame.shape[:2]
        vw = max(64, self.view.width() - 6)
        vh = max(64, self.view.height() - 6)
        k = min(vw / float(fw), vh / float(fh), 1.0)     # **不放大**（放大只会变糊 ✗）
        if k < 0.999:
            small = cv2.resize(frame, (max(1, int(fw * k)), max(1, int(fh * k))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame.copy()                         # 不缩也得拷（要在上面画框 ✓）
        if self.chk_draw.isChecked():
            self._draw(small, snap, k)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def _draw(self, img, snap, k=1.0):
        """把 ROI / 目标框 / 瞄点画到（可能已缩小的）图上 —— 坐标一律乘 `k` ✓。"""
        import cv2

        def _p(x, y):
            return (int(round(x * k)), int(round(y * k)))

        if snap.get("roi"):
            x, y, w, h = snap["roi"]
            cv2.rectangle(img, _p(x, y), _p(x + w, y + h), C_ROI, 1)
            cv2.putText(img, "ROI", _p(x + 2, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_ROI, 1, cv2.LINE_AA)
        if snap.get("box"):
            x0, y0, x1, y1 = snap["box"]
            cv2.rectangle(img, _p(x0, y0), _p(x1, y1), C_BOX, 2)
        if snap.get("aim"):
            ax, ay = _p(*snap["aim"])
            cv2.circle(img, (ax, ay), 8, C_AIM, 1)
            cv2.line(img, (ax - 12, ay), (ax + 12, ay), C_AIM, 1)
            cv2.line(img, (ax, ay - 12), (ax, ay + 12), C_AIM, 1)
        return img

    def _state_text(self, snap):
        # ⭐ 三段耗时**分开报**（用户 2026-10-17 ✓「它好像特别的慢」）——
        #   只报一个"单帧 ms"时分不清是**抓屏慢**、**推理慢**还是**画图慢** ✗
        #   （画图慢还会把界面一起堵住 ✓ 看着像整体都慢 ✓）。
        parts = ["阶段 **%s**" % snap.get("phase", "-"),
                 "帧 %d" % snap.get("frames", 0),
                 "%.1f fps" % snap.get("fps", 0.0),
                 "抓 %.0f+推 %.0f ms" % (snap.get("grab_ms", 0.0),
                                        max(0.0, snap.get("ms", 0.0)
                                            - snap.get("grab_ms", 0.0))),
                 "画 %.1f ms" % snap.get("draw_ms", 0.0)]
        if snap.get("roi"):
            parts.append("ROI %s" % (snap["roi"],))
        if snap.get("box"):
            parts.append("目标框 %s" % (snap["box"],))
        if snap.get("aim"):
            parts.append("瞄点(帧) %s" % (snap["aim"],))
        if snap.get("aim_screen"):
            parts.append("瞄点(屏) (%.0f, %.0f)" % snap["aim_screen"])
        parts.append("鼠标指令 %d 条%s" % (
            self.runner.moved,
            "（被人工接管挡下 %d）" % self.runner.moved_blocked
            if self.runner.moved_blocked else ""))
        return "　·　".join(parts)

    def _say(self, text):
        self.lbl_state.setText(text)

    # ---------------- 收尾 ----------------
    def closeEvent(self, e):
        try:
            self._pause()                   # 先把工作线程收掉（别让它在解释器退出时还在喂帧 ✗）
            self.vt.close()
        except Exception:                   # noqa: BLE001
            pass
        super().closeEvent(e)


def main(argv=None):
    ap = argparse.ArgumentParser(description="测谎自动跟踪 · 独立测试窗")
    ap.add_argument("--rect", nargs=4, type=int, default=None,
                    metavar=("L", "T", "W", "H"), help="屏幕矩形（物理像素）")
    ap.add_argument("--video", default="", help="视频回放（不会移动鼠标）")
    ap.add_argument("--move", action="store_true", help="允许移动真实光标（默认不动）")
    ap.add_argument("--content", action="store_true", help="矩形已经是内容区（跳过定位）")
    ap.add_argument("--auto", action="store_true",
                    help="开窗就自动点「开始」（冒烟/录屏用 ✓ 鼠标该动还是会动 ✗ 注意 --move）")
    args = ap.parse_args(argv)

    _set_dpi_aware()
    app = QApplication(sys.argv[:1])
    w = VtTestWindow(rect=args.rect, video=args.video, allow_move=args.move,
                     content=args.content, auto=args.auto)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
