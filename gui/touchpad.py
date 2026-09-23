"""虚拟触控板：本地按 F10 开 / 关触控模式，在方块上滑动鼠标 → 远程鼠标跟着动。

交互：
    本地按 F10 → 进入触控模式（板子变蓝）→ 在这块板上滑动鼠标 → 远程鼠标跟着动
    再按一次 F10（或 ProMicro 断开）→ 退出触控模式
    触控模式期间，在这块板上点击 / 滚轮，会一并转发给远程鼠标
    （左键、右键、滚轮中键点击、滚轮上下滚动）。

**为什么不用「按住 Ctrl 滑动」**（原来的做法）：
    Ctrl 是默认的攻击键，而「开启手动输入」会把本机按键实时转发给游戏 ——
    按住 Ctrl 用触控板时，这个 Ctrl 本身就是一次攻击，游戏里会跟着出手。
    改成 F10 开关：本地一个键、只走本窗口的按键事件，不经过任何转发路径
    （F10 也在 `decision/input.py` 的「本机操作键」名单里，永不发给游戏）。

实现要点 —— **光标钉住（warp）**：
    本地鼠标在屏幕上是有限的，滑到边缘就滑不动了。进入触控模式时把光标挪到
    板子中心并记下位置，之后每次拿到位移就 `QCursor.setPos()` 复位回原点，
    等于一块「无限大的触控板」；同时光标视觉上停在板子上，正好符合「鼠标指着
    这块板」，点击 / 滚轮也都发生在这一小块区域内，不会因为光标跑远而点丢。

    先挪到**中心**是为了四个方向都有余量：F10 是随时按的，光标原本可能就在
    屏幕最边上，往那一侧滑就没反应了。

信号：
    moved(dx, dy)   —— 原始位移（未乘灵敏度），由调用方决定映射比例
    clicked(btn)    —— 点击（"left" / "right" / "middle"）
    scrolled(n)     —— 滚轮格数（正数向上、负数向下）
    active_changed  —— 是否处于触控模式（开 / 关）
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from gui.widgets import forward_wheel


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
            "本地按 F10 开启 / 关闭触控模式。\n"
            "开启后在这块板上滑动鼠标 → 远程鼠标跟着动（板子会变蓝）。\n"
            "触控模式期间：左键 / 右键 / 按下滚轮 会点击，上下滚轮会滚动。\n"
            "用 F10 而不是按住 Ctrl：Ctrl 是默认攻击键，按住它会打出去。")
        self._enabled = True
        self._active = False
        self._ref = None          # 光标原点（全局坐标）
        self._warping = False     # 标记「下一次 move 是 warp 触发的」，要跳过
        self._wheel_rem = 0       # 滚轮余数：触摸板会给出不足一格（120）的增量

    # ---------------- 对外 ----------------

    def set_capture_enabled(self, on):
        """ProMicro 未连接时禁用（本地模式没有硬件鼠标）。"""
        on = bool(on)
        if on == self._enabled:
            return
        self._enabled = on
        if not on:
            self.set_active(False)
        self.update()

    def toggle(self):
        """本地按 F10：开 / 关触控模式。"""
        self.set_active(not self._active)

    def set_active(self, on):
        """进入 / 退出触控模式（面板的 F10 处理走这里）。"""
        if on and self._enabled:
            self._activate()
        elif not on:
            self._deactivate()

    @property
    def active(self):
        return self._active

    # ---------------- 内部 ----------------

    def _activate(self):
        if self._active:
            return
        self._active = True
        self._wheel_rem = 0
        # 光标挪到板子中心再钉住：四个方向都留出滑动余量（见模块文档）
        self._ref = self.mapToGlobal(self.rect().center())
        self._warping = True           # 这次 setPos 会带回一个 move 事件，忽略掉
        QCursor.setPos(self._ref)
        self.grabMouse()               # 捕获鼠标：滑出方块也能继续收到事件
        self.active_changed.emit(True)
        self.update()

    def _deactivate(self):
        if not self._active:
            return
        self._active = False
        try:
            self.releaseMouse()
        except Exception:
            pass
        self._ref = None
        self._warping = False
        self._wheel_rem = 0
        self.active_changed.emit(False)
        self.update()

    # ---------------- 事件 ----------------

    def mouseMoveEvent(self, ev):
        if self._warping:
            self._warping = False     # 这次 move 是 setPos 触发的，忽略
            return
        if not self._active:
            return                    # 非触控模式：板子上滑动不做事
        gp = ev.globalPos()
        dx = gp.x() - self._ref.x()
        dy = gp.y() - self._ref.y()
        if dx or dy:
            self.moved.emit(dx, dy)
            self._warping = True
            QCursor.setPos(self._ref)     # 光标钉回原点 → 无限滑动

    def mousePressEvent(self, ev):
        # 触控模式下：左/右/中键点击一并转发给远程鼠标。
        # 用「点击」而不是「按下-松开」：固件 CLICK 是一次原子动作，不会因为
        # 松开事件丢失而卡住按钮。要拖拽请用右边的「左键」按钮（按住不放）。
        if self._enabled and self._active:
            btn = {Qt.LeftButton: "left",
                   Qt.RightButton: "right",
                   Qt.MiddleButton: "middle"}.get(ev.button())
            if btn:
                self.clicked.emit(btn)
        ev.accept()   # 触控板不吃点击，不会顺带触发界面上的其它控件

    def wheelEvent(self, ev):
        """触控模式：滚轮转发给远程鼠标；非触控模式：让给外层滚动区滚页面。"""
        if not (self._enabled and self._active):
            if forward_wheel(ev, self):
                ev.accept()
            else:
                ev.ignore()
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
            txt = "触控模式已开\n滑动 / 点击 / 滚轮\n再按 F10 关闭"
        else:
            txt = "触控板\n本地按 F10 开启\n（开启后可点击 / 滚轮）"
        p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, txt)
