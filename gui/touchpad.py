"""虚拟触控板：按住 Ctrl 在方块上滑动鼠标，位移映射成远程鼠标移动指令。

交互：
    本地鼠标移到方块上 → 按住 Ctrl → 进入触控模式 → 滑动鼠标 → 松开 Ctrl 退出
    触控模式期间，在这块板上点击 / 滚轮，会一并转发给远程鼠标
    （左键、右键、滚轮中键点击、滚轮上下滚动）。

实现要点 —— **光标钉住（warp）**：
    本地鼠标在屏幕上是有限的，滑到边缘就滑不动了。进入触控模式后记下光标
    位置，每次拿到位移就 `QCursor.setPos()` 把光标复位回原点，等于一块「无限
    大的触控板」；同时光标视觉上保持不动，正好符合「鼠标指着这块板」。

    光标被钉住还有个好处：点击 / 滚轮都发生在原点这一小块区域内，不会因为
    光标跑远而点丢。

信号：
    moved(dx, dy)   —— 原始位移（未乘灵敏度），由调用方决定映射比例
    clicked(btn)    —— 点击（"left" / "right" / "middle"）
    scrolled(n)     —— 滚轮格数（正数向上、负数向下）
    active_changed  —— 是否处于滑动中
"""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget


class TouchPad(QWidget):
    moved = pyqtSignal(int, int)
    clicked = pyqtSignal(str)     # 触控模式下的点击："left" / "right" / "middle"
    scrolled = pyqtSignal(int)    # 触控模式下的滚轮：正数向上、负数向下
    active_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(210, 132)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip(
            "按住 Ctrl，在这块板上滑动鼠标 → 远程鼠标跟着动（松开 Ctrl 停止）。\n"
            "Ctrl 期间：左键 / 右键 / 按下滚轮 会点击，上下滚轮会滚动。")
        self._enabled = True
        self._active = False
        self._ref = None          # 光标原点（全局坐标）
        self._warping = False     # 标记「下一次 move 是 warp 触发的」，要跳过
        self._wheel_rem = 0       # 滚轮余数：触摸板会给出不足一格（120）的增量
        # 鼠标不动时收不到 move 事件，用定时器兜底检测 Ctrl 松开
        self._mod_timer = QTimer(self)
        self._mod_timer.setInterval(50)
        self._mod_timer.timeout.connect(self._check_modifier)

    # ---------------- 对外 ----------------

    def set_capture_enabled(self, on):
        """ProMicro 未连接时禁用（本地模式没有硬件鼠标）。"""
        on = bool(on)
        if on == self._enabled:
            return
        self._enabled = on
        if not on:
            self._deactivate()
        self.update()

    @property
    def active(self):
        return self._active

    # ---------------- 内部 ----------------

    def _activate(self):
        self._active = True
        self._ref = QCursor.pos()
        self._wheel_rem = 0
        self.grabMouse()          # 捕获鼠标：滑出方块也能继续收到事件
        self._mod_timer.start()
        self.active_changed.emit(True)
        self.update()

    def _deactivate(self):
        if not self._active:
            return
        self._active = False
        self._mod_timer.stop()
        try:
            self.releaseMouse()
        except Exception:
            pass
        self._ref = None
        self._warping = False
        self._wheel_rem = 0
        self.active_changed.emit(False)
        self.update()

    def _check_modifier(self):
        if self._active and not (QApplication.keyboardModifiers() & Qt.ControlModifier):
            self._deactivate()

    # ---------------- 事件 ----------------

    def mouseMoveEvent(self, ev):
        if self._warping:
            self._warping = False     # 这次 move 是 setPos 触发的，忽略
            return
        ctrl = bool(ev.modifiers() & Qt.ControlModifier)
        if ctrl and self._enabled:
            if not self._active:
                self._activate()
                return                # 刚激活这一帧不计位移
            gp = ev.globalPos()
            dx = gp.x() - self._ref.x()
            dy = gp.y() - self._ref.y()
            if dx or dy:
                self.moved.emit(dx, dy)
                self._warping = True
                QCursor.setPos(self._ref)     # 光标钉回原点 → 无限滑动
        elif self._active:
            self._deactivate()

    def _ensure_active(self, mods):
        """按住 Ctrl 但还没进触控模式时，补一次激活。

        正常路径是「先滑动一下」触发激活，但「按住 Ctrl 直接点击 / 滚轮」不产生
        move 事件、不会激活 —— 不补这一下，第一下点击就被吞掉，用起来像坏了。
        """
        if self._enabled and not self._active and (mods & Qt.ControlModifier):
            self._activate()

    def mousePressEvent(self, ev):
        # 触控模式（按住 Ctrl）下：左/右/中键点击一并转发给远程鼠标。
        # 用「点击」而不是「按下-松开」：固件 CLICK 是一次原子动作，不会因为
        # 松开事件丢失而卡住按钮。要拖拽请用右边的「左键」按钮（按住不放）。
        self._ensure_active(ev.modifiers())
        if self._enabled and self._active:
            btn = {Qt.LeftButton: "left",
                   Qt.RightButton: "right",
                   Qt.MiddleButton: "middle"}.get(ev.button())
            if btn:
                self.clicked.emit(btn)
        ev.accept()   # 触控板不吃点击，不会顺带触发界面上的其它控件

    def wheelEvent(self, ev):
        """触控模式下把滚轮转发给远程鼠标（非触控模式吞掉，避免滚动外层面板）。"""
        self._ensure_active(ev.modifiers())
        if not (self._enabled and self._active):
            ev.accept()
            return
        # Qt 一格滚轮 = 120；触摸板/高精度滚轮会给不足一格的增量，攒够再发
        self._wheel_rem += ev.angleDelta().y()
        n = int(self._wheel_rem / 120)
        if n:
            self._wheel_rem -= n * 120
            self.scrolled.emit(n)
        ev.accept()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        if not self._enabled:
            bg, border, fg = QColor("#f1f3f4"), QColor("#dadce0"), QColor("#bdc1c6")
        elif self._active:
            bg, border, fg = QColor("#e8f0fe"), QColor("#1a73e8"), QColor("#1a73e8")
        else:
            bg, border, fg = QColor("#f8f9fa"), QColor("#dadce0"), QColor("#5f6368")
        p.setPen(QPen(border, 2))
        p.setBrush(bg)
        p.drawRoundedRect(r, 8, 8)
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(fg)
        if not self._enabled:
            txt = "鼠标控制\n需要 ProMicro"
        elif self._active:
            txt = "滑动中…\n点击 / 滚轮 已接通"
        else:
            txt = "触控板\n按住 Ctrl 滑动鼠标\n（期间可点击 / 滚轮）"
        p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, txt)
