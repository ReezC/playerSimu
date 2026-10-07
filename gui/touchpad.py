"""虚拟触控板：本地按 F10 开 / 关触控模式，在方块上滑动鼠标 → 远程鼠标跟着动。

交互：
    本地按 F10 → 进入触控模式（板子变蓝）→ 在这块板上滑动鼠标 → 远程鼠标跟着动
    再按一次 F10（或 ProMicro 断开）→ 退出触控模式
    触控模式期间，在这块板上点击 / 按住拖 / 滚轮，会一并转发给远程鼠标
    （左键、右键、滚轮中键、滚轮上下滚动）。

**点击 vs 拖拽**（两者都要，所以不能一律用「按下-松开」）：
    · 点一下就松 = 走固件的原子 CLICK —— 一次动作完成，不会因为丢一个松开事件
      把远程按钮卡在按下状态（这是原来只用 CLICK 的理由，保留）；
    · 按住并滑动超过 HOLD_PX = 发 PRESS 按住不放 → 滑动 → 松开时发 RELEASE =
      真正的拖拽（拖窗口 / 框选 / 拖滚动条）。**拖拽必须一开始就按住**，
      所以判定是「移动超过阈值的那一刻补发 PRESS」，那几像素的位移照常发出去，
      不丢。

**为什么不用「按住 Ctrl 滑动」**（原来的做法）：
    Ctrl 是默认的攻击键，而「开启手动输入」会把本机按键实时转发给游戏 ——
    按住 Ctrl 用触控板时，这个 Ctrl 本身就是一次攻击，游戏里会跟着出手。
    改成 F10 开关：本地一个键、只走本窗口的按键事件，不经过任何转发路径
    （F10 也在 `decision/input.py` 的「本机操作键」名单里，永不发给游戏）。

实现要点 —— **位移由跟踪线程按固定节拍产生**：
    ⭐⭐ 2026-10-05（用户原话："B机F10控制A机鼠标，A机的鼠标移动不连续" ⇒ 选"直接②" ✓）：
    位移**不再由 Qt 的 move 事件产生** ✗ —— 那跑在 GUI 主线程上，主线程一被重绘 /
    主回路拖住，Qt 就把连续移动**合并**成一次 ⇒ 实测（`perf.log` 18:40 那段）
    `move_gap_ms` 最大 **467 ms**、`move_px` 最大 **514** ⇒ "停一下、猛跳一下" ✓
    （而同一段 `send_ms` 中位才 **0.15 ms** ⇒ 瓶颈不在发送 ✓）。
    现在：`decision/input.py::PointerTracker` 起一条**独立线程**，按固定节拍
    （`TRACK_INTERVAL_S` ✓）`GetCursorPos` 算位移、再把指针 `SetCursorPos` 拨回钉点
    ⇒ 节拍只受"线程能不能被调度"影响，与 GUI 卡不卡无关 ✓。

    位移的**去向**分两路（这是本模块最要紧的一处口径 ✓）：
      · **要发出去的那部分**走 `set_delta_sink`（面板接它直接喂发送器 ✓）——
        **在跟踪线程里同步调**，不经 Qt 事件循环 ✓✓（这是"均匀"的全部保证 ✓）；
      · `moved` 信号（Qt 排队投递 ⇒ 回到 GUI 线程 ✓）只给**拖拽阈值**这类判定用 ✓
        —— 它晚几毫秒到没关系（判的是累计距离 ✓），位移本身一个像素都不会少 ✓。

实现要点 —— **光标钉住（warp）**：
    本地鼠标在屏幕上是有限的，滑到边缘就滑不动了。进入触控模式时把光标挪到
    板子中心并记下位置，之后每一拍（跟踪线程 ✓）把指针 `SetCursorPos` 拨回原点，
    等于一块「无限大的触控板」；同时光标视觉上停在板子上，正好符合「鼠标指着
    这块板」，点击 / 滚轮也都发生在这一小块区域内，不会因为光标跑远而点丢。

    先挪到**中心**是为了四个方向都有余量：F10 是随时按的，光标原本可能就在
    屏幕最边上，往那一侧滑就没反应了。⚠ 现在每 4 ms 就拨回一次（原来只在
    "下一个事件到来时"拨 ✗）⇒ 手指能跑出去的空间**极小** ✓ ⇒ 屏幕边缘那点
    限制基本不可能碰到 ✓。

信号：
    moved(dx, dy)   —— 原始位移（**未乘灵敏度、浮点** ✓）：**由跟踪线程发出** ⇒ 连接方
                       拿到的是排队投递（在 GUI 线程执行 ✓）；只给"拖拽阈值"这类判定用 ✓。
                       真正要发出去的位移走 `set_delta_sink`（不经 Qt ✓ 见上）。
    clicked(btn)    —— 点击（"left" / "right" / "middle"）：按一下没怎么动就松开
    pressed(btn)    —— 按住不放（拖拽开始）：按下后滑动超过阈值
    released(btn)   —— 松开拖拽中的按钮
    scrolled(n)     —— 滚轮格数（正数向上、负数向下）
    active_changed  —— 是否处于触控模式（开 / 关）
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from gui.widgets import forward_wheel


class TouchPad(QWidget):
    #: 按下后滑动超过这么多像素才算「拖拽」（否则松开当点击）—— 真实触控板也是这个逻辑
    HOLD_PX = 4

    moved = pyqtSignal(float, float)
    clicked = pyqtSignal(str)     # 点击："left" / "right" / "middle"
    pressed = pyqtSignal(str)     # 按住不放（拖拽开始）
    released = pyqtSignal(str)    # 松开拖拽中的按钮
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
            "触控模式期间：左键 / 右键 / 按下滚轮 会点击，按住滑动 = 拖拽，\n"
            "上下滚轮会滚动。\n"
            "用 F10 而不是按住 Ctrl：Ctrl 是默认攻击键，按住它会打出去。")
        self._enabled = True
        self._active = False
        self._ref = None          # 光标钉点（**物理像素** ✓ 由 GetCursorPos 读回 ✓ 见 _start_track）
        self._wheel_rem = 0       # 滚轮余数：触摸板会给出不足一格（120）的增量
        # ⭐⭐ 位移的**产生**：跟踪线程 + 它的实时出口（用户 2026-10-05 ✓ "直接②" ✓ 见模块文档）
        self._tracker = None      # `decision.input.PointerTracker`（没进触控模式时为 None ✓）
        self._delta_sink = None   # 面板接的出口（**在跟踪线程里被调** ✓ 只许线程安全的事 ✓）
        self._track_fail = False  # 跟踪线程起不来（界面会说清 ✓ 别静默 ✗）
        # 拖拽状态：按下先不发，滑动超过 HOLD_PX 那一刻才补发 PRESS
        self._down = None         # 按着没松的按钮名（还没进入拖拽）
        self._dragging = False    # 已经发过 PRESS，松开时要发 RELEASE
        self._drag_moved = 0.0    # 按下之后累计走了多少像素（判阈值用）
        self._drag_btn = None     # 拖拽中的按钮名
        # `moved` **由跟踪线程发出** ⇒ 这里自己再收一份（排队投递 ⇒ 回到 GUI 线程 ✓）
        # 只用来判拖拽阈值 ✓（真发送走 `_delta_sink` ✓ 见 `_on_tracked`）
        self.moved.connect(self._note_drag_move)

    # ---------------- 对外 ----------------

    def set_delta_sink(self, fn):
        """接上位移的**实时出口**（面板调 ✓）：`fn(dx, dy)` —— **在跟踪线程里被同步调** ✓。

        为什么要这么个口子（而不是用 `moved` 信号）：信号是**排队**投递的 ⇒ 回到 GUI 线程
        执行 ⇒ 又会排在重绘 / 主回路后面 ✗ —— 那正是"位移的节拍挂在主线程上"的老毛病 ✓
        （用户 2026-10-05 ✓ 见模块文档）。所以"要发出去的位移"必须由跟踪线程**直接**交出去 ✓。

        ⚠ `fn` 只许做线程安全的事（加法 / 入队 / 置事件 ✓）；碰界面一律走 `moved` 信号 ✓。
        """
        self._delta_sink = fn

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
        self._forget_press()
        # 光标挪到板子中心再钉住：四个方向都留出滑动余量（见模块文档）
        # ⚠ 这一次 `setPos` 必须走 Qt（GUI 线程 ✓）；之后每一拍由**跟踪线程**用
        #   `SetCursorPos` 拨回（非 GUI 线程不能用 `QCursor` ✗ 见 `PointerTracker` 的纪律 ✓）
        self._ref = self.mapToGlobal(self.rect().center())
        QCursor.setPos(self._ref)
        self.grabMouse()               # 捕获鼠标：滑出方块也能继续收到事件
        self._start_track()            # ⭐ 位移的产生交给跟踪线程（见模块文档 ✓）
        self.active_changed.emit(True)
        self.update()

    def _deactivate(self):
        if not self._active:
            return
        # **先松开拖拽中的按钮**再退出：F10 关掉 / 窗口失焦 / 切页签都会走这里，
        # 那一刻若正按着，远程那个按钮会永远留在按下状态（只能靠固件的
        # RELEASEALL 兜底，见 remote_kbd/pro_micro/pro_micro.ino）。
        self._end_drag()
        self._forget_press()
        self._active = False
        self._stop_track()             # ⭐ 先收掉跟踪线程（别让它继续摸指针 ✗ 见模块文档）
        try:
            self.releaseMouse()
        except Exception:
            pass
        self._ref = None
        self._wheel_rem = 0
        self.active_changed.emit(False)
        self.update()

    def _end_drag(self):
        """正在拖拽就补一条 RELEASE（退出触控模式时的安全网）。"""
        if self._dragging and self._drag_btn:
            self.released.emit(self._drag_btn)

    def _forget_press(self):
        """忘掉按下/拖拽状态（不发信号）。"""
        self._down = None
        self._dragging = False
        self._drag_moved = 0.0
        self._drag_btn = None

    # ---------------- 位移的产生（跟踪线程，见模块文档） ----------------

    def _start_track(self):
        """起**指针跟踪线程**（位移从这儿来 ✓）：钉点用 `GetCursorPos` 读回**物理像素** ✓。

        ⚠ 必须在 `QCursor.setPos(板心)` **之后**读：这样钉点是物理像素，高分屏下不用换算 ✓
          （位移由 `PointerTracker` 按 `scale=devicePixelRatio` 除回逻辑像素 ✓）；
          起不来就 `_track_fail`（界面会说清 ✓ 别静默 —— 静默的后果是"板子像坏的"✗）。
        """
        from decision import input as dinput

        self._stop_track()                          # 幂等（重复激活不会漏掉旧线程 ✓）
        pos = dinput.pointer_pos()
        t = None
        if pos is not None:
            t = dinput.PointerTracker(
                pos, self._on_tracked,
                scale=max(1e-6, float(self.devicePixelRatioF() or 1.0)))
            if not t.start():
                t = None
        self._tracker = t
        self._track_fail = t is None
        if self._track_fail:
            try:
                from core import perf
                perf.count("pad_track_fail")        # 记一笔：这条功能没在工作 ✓
            except Exception:                       # noqa: BLE001 —— 打点坏了别影响行为 ✗
                pass

    def _stop_track(self):
        """收工：停跟踪线程 + **等它真的退出**（别留一条守护线程在后台摸指针 ✗）。"""
        t = self._tracker
        self._tracker = None
        if t is not None:
            t.stop()

    def _on_tracked(self, dx, dy):
        """跟踪线程每拍的回调 —— **在跟踪线程里执行** ✗⏰ ⇒ 只做两件线程安全的事：

        ① 位移交给**实时出口**（`set_delta_sink` ✓）：面板接它直接喂发送器 ⇒
           **不经 Qt 事件循环** ✓✓（"GUI 卡不卡都均匀"的全部保证 ✓）；
        ② `moved` 信号（Qt 排队投递 ⇒ 回到 GUI 线程 ✓）：只给拖拽阈值这类判定用 ✓。

        ⚠ 这里**绝对不许碰界面** ✗（`self.xxx.setText()` / `QCursor` / `update()` 都不行 ✓）——
          非 GUI 线程碰 Qt 控件会偶发崩 ✓；要改界面就发信号让 GUI 线程去改 ✓。
        """
        sink = self._delta_sink
        if sink is not None:
            try:
                sink(dx, dy)
            except Exception:                       # noqa: BLE001 —— 出口坏了别弄死跟踪线程 ✗
                pass
        try:
            self.moved.emit(float(dx), float(dy))
        except Exception:                           # noqa: BLE001 —— 窗口正在销毁时 emit 会抛 ✓
            pass

    def _note_drag_move(self, dx, dy):
        """拖拽阈值：按下后累计滑动超过 `HOLD_PX` 才补发 PRESS（进入拖拽 ✓）—— GUI 线程里跑 ✓。

        ⚠ 位移是跟踪线程**排队**送来的（会晚几毫秒 ✓）—— 没关系：这里判的是**累计距离**
          （阈值 4 px ✓），迟到只意味着"拖拽晚几毫秒开始" ✓；位移本身一个像素都不会少
          （它走的是实时出口 ✓ 见模块文档）。那几像素也照常发出去，不丢 ✓。
        """
        if self._down is None or self._dragging:
            return
        self._drag_moved += (dx * dx + dy * dy) ** 0.5
        if self._drag_moved >= self.HOLD_PX:
            self._dragging = True
            self._drag_btn = self._down
            self.pressed.emit(self._drag_btn)

    # ---------------- 事件 ----------------

    def mouseMoveEvent(self, ev):
        """⚠ 位移**不在这里产生**（用户 2026-10-05 ✓ "直接2"）：这里只是"跟踪线程每拍把指针
        拨回板心"带回来的**回声** ⇒ 什么都不做 ✓。

        原来位移在这里算（`ev.globalPos() - 钉点` ⇒ `moved` ⇒ `QCursor.setPos` 拨回 ✗）——
        那等于把位移的节拍**挂在 Qt 的事件投递上**：GUI 主线程一被重绘 / 主回路拖住，Qt 就把
        连续移动**合并**成一次 ⇒ 一次跳几百像素 ✗（实测 `move_px` 最大 **514** / `move_gap_ms`
        最大 **467 ms** ✓）⇒ A 机"停一下、猛跳一下" ✓。见模块文档 ✓。
        """
        ev.accept()

    def mousePressEvent(self, ev):
        # 触控模式下：左/右/中键**先只记下来**，不立刻发 —— 等滑动超过阈值再发
        # PRESS（拖拽）；一直没滑动就松开则走固件的原子 CLICK。
        # 这样「点击」仍然是原子的（丢一个松开事件不会卡住按钮），
        # 而「按住拖」这条原来没有的路也能走（见模块文档的「点击 vs 拖拽」）。
        if self._enabled and self._active:
            btn = {Qt.LeftButton: "left",
                   Qt.RightButton: "right",
                   Qt.MiddleButton: "middle"}.get(ev.button())
            if btn:
                self._down = btn
                self._dragging = False
                self._drag_moved = 0.0
                self._drag_btn = None
        ev.accept()   # 触控板不吃点击，不会顺带触发界面上的其它控件

    def mouseReleaseEvent(self, ev):
        """松开：拖拽过 → RELEASE；没滑过阈值 → 点一下（原子 CLICK）。"""
        btn = {Qt.LeftButton: "left", Qt.RightButton: "right",
               Qt.MiddleButton: "middle"}.get(ev.button())
        if self._down is not None and btn == self._down:
            if self._dragging:
                self.released.emit(self._drag_btn)
            else:
                self.clicked.emit(self._down)
            self._forget_press()
        ev.accept()

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
        elif self._active and self._track_fail:
            # ⚠ 跟踪线程起不来（`GetCursorPos` 不可用那种）⇒ **说清楚**：别让人以为板子坏了 ✓
            txt = "触控模式已开\n⚠ 指针跟踪起不来\n（滑动不会动鼠标）"
        elif self._active:
            txt = "触控模式已开\n滑动 / 点击 / 拖拽 / 滚轮\n再按 F10 关闭"
        else:
            txt = "触控板\n本地按 F10 开启\n（开启后可点击 / 滚轮）"
        p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, txt)
