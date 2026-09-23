"""共用的基础控件。

**为什么需要这些子类**

    QSpinBox / QDoubleSpinBox / QComboBox / QSlider 默认「指针悬停就响应滚轮」。
    在滚动区里这是灾难：想翻页，指针顺路划过参数框，数字就被改了，
    而且改完没有任何提示。

    更隐蔽的是「聚焦残留」：点过一次数字框后焦点还留在它上面，
    之后再滚页面，滚轮仍会被当成"在调这个参数"—— 表现为
    「没聚焦时不误改了，但只要点过一下就又开始了」。

**规则：这些控件永远不响应滚轮，但页面必须照滚**

    滚轮专心滚页面 / 缩放画布；参数只用上下箭头、下拉框、键盘来改。
    这些控件都有明确的非滚轮调节方式，禁用滚轮不损失任何操作效率，
    反而彻底根除了"到底是谁在抢滚轮"的纠结。

**关键：不要靠 `ev.ignore()` 把滚轮「让给父级」**

    「`ignore()` 之后事件冒泡给父级滚动区」是 Qt 桌面开发的常见说法，但它依赖
    平台层怎么派发滚轮（控件树下谁先拿到、忽略后往上走几层）—— 实测无头环境下
    合成事件 + `ignore()`，外层 QScrollArea 纹丝不动；真实环境里这条链路的
    细节也不由我们控制。

    所以改成**自己驱动滚动条**：见 `forward_wheel()` —— 找最近的外层滚动区，
    按「系统滚轮行数 × singleStep」直接改它的滚动条值，再 `accept()` 事件。
    这样滚不滚、滚多少完全由我们决定，也不依赖任何冒泡细节。
"""

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import (QAbstractScrollArea, QAbstractSlider,
                             QAbstractSpinBox, QApplication, QComboBox,
                             QDoubleSpinBox, QScrollBar, QSlider, QSpinBox,
                             QTabBar)

# 守卫只吃这几类控件上的滚轮：数字框 / 下拉框 / 滑块 / 标签栏。
# （QSpinBox、QDoubleSpinBox ⊂ QAbstractSpinBox；QSlider ⊂ QAbstractSlider）
# 其它一律放行：滚动区、列表、树、文本区、画面……滚轮该干的活照干。
# 注意 **QScrollBar 要单独放行**（见 _WheelGuard）：它也是 QAbstractSlider
# 的子类，但它滚的就是页面，吃掉它 = 指针压在最右边的滚动条上时页面不滚。
_VALUE_WIDGETS = (QAbstractSpinBox, QComboBox, QAbstractSlider, QTabBar)


def forward_wheel(ev, widget):
    """把落在 widget 上的滚轮事件改成「滚动最近的外层滚动区」。

    真的滚动了返回 True（调用方应 `ev.accept()`，别让它再往上冒一次，
    否则可能会被外层再滚一遍）。

    **为什么不用 `ev.ignore()`**：见模块文档 —— 别依赖平台层的冒泡。
    这里从 widget 自己开始往上找 QAbstractScrollArea 并驱动它的滚动条；内层已经
    滚到头（到顶/到底）就继续往外找，嵌套滚动区也能用。路上谁挂了
    `_wheel_target`（指向滚动区）就用谁 —— 见 gui/player_panel.py 的常驻块。
    """
    px = ev.pixelDelta()
    ang = ev.angleDelta()
    if not (px.x() or px.y() or ang.x() or ang.y()):
        return False
    # Shift + 滚轮 = 横向滚（跟系统习惯一致）
    hx, hy = px.x(), px.y()
    ax, ay = ang.x(), ang.y()
    if (ev.modifiers() & Qt.ShiftModifier) and (hy or ay) and not (hx or ax):
        hx, hy, ax, ay = 0, 0, ay, 0
    w = widget
    while w is not None:
        if isinstance(w, QAbstractScrollArea):
            if _apply_wheel(w, hx, hy, ax, ay):
                return True
        # 指路：有些控件（比如「置顶常驻」的组）不在滚动区**里面**，往上找永远
        # 找不到 —— 谁把它放在外面，就在它上面挂一个 `_wheel_target` 指向那个
        # 滚动区，滚轮照样滚页面（见 gui/player_panel.py 的「操控」组）。
        tgt = getattr(w, "_wheel_target", None)
        if isinstance(tgt, QAbstractScrollArea) and _apply_wheel(tgt, hx, hy, ax, ay):
            return True
        w = w.parentWidget()
    return False


def _apply_wheel(area, hx, hy, ax, ay):
    """按滚轮增量滚 area 的两个滚动条；滚动了返回 True。"""
    moved = False
    for sb, dpx, dang in ((area.horizontalScrollBar(), hx, ax),
                          (area.verticalScrollBar(), hy, ay)):
        if sb is None or sb.maximum() <= sb.minimum():
            continue
        if dpx:                      # 触摸板 / 平滑滚动：给的就是像素
            amount = float(dpx)
        elif dang:                   # 普通滚轮：一格 = 120，换算成「行 × 行高」
            lines = QApplication.wheelScrollLines()
            if lines <= 0:
                lines = 3            # 系统取不到（-1/0）时用 Qt 的常规默认
            amount = angle_to_units(dang, lines, sb.singleStep() or 1)
        else:
            continue
        v = sb.value() + int(round(amount))
        v = max(sb.minimum(), min(sb.maximum(), v))
        if v != sb.value():
            sb.setValue(v)
            moved = True
    return moved


def angle_to_units(angle, lines, step):
    """angleDelta（一格 120，向上为正）→ 滚动条单位数（正数 = 值变大 = 内容往下/往右）。

    向上滚（angle > 0）应该往回看 → 值变小，所以整体取负。
    """
    return -float(angle) / 120.0 * lines * step


class _NoWheel:
    """滚轮事件永远不改本控件的值；有外层滚动区就改成滚外层。"""

    def wheelEvent(self, ev):
        if forward_wheel(ev, self):
            ev.accept()          # 已代为滚动，别再往上传（否则外层可能再滚一遍）
        else:
            ev.ignore()


class NoWheelSpinBox(_NoWheel, QSpinBox):
    pass


class NoWheelDoubleSpinBox(_NoWheel, QDoubleSpinBox):
    pass


class NoWheelComboBox(_NoWheel, QComboBox):
    pass


class NoWheelSlider(_NoWheel, QSlider):
    pass


class _WheelGuard(QObject):
    """应用级守卫：滚轮不许改任何参数控件。

    **为什么不能只靠 NoWheel* 子类**
        那要求每个写 UI 的人都记得换类。实测漏了十几处（calib_manual /
        player_panel / settings_dialog / cards 里都有裸控件），漏一处就是一个
        「滚着滚着参数被改了、而且不知道是谁改的」的坑 —— 排查起来极其难受。

    守卫装在 QApplication 上，看到滚轮事件落在参数控件上就**代为滚动外层
    滚动区**（而不是单纯吃掉 —— 吃掉会让「指针正好压在它上面时页面不滚」）。
    已经用 NoWheel* 写过的控件跳过（它们自己做同一件事，别做两遍）。
    """

    def eventFilter(self, obj, ev):
        if ev.type() != QEvent.Wheel:
            return False
        if isinstance(obj, _NoWheel):
            # NoWheel* 自己会转发，但**这里也得转一次**：QAbstractSpinBox（数字框）
            # 这类控件 Qt 在 event() 里就把滚轮忽略了 —— 我们的 wheelEvent 压根
            # 不会被调用，事件只会一路往父级冒。父级要是没滚动区（比如置顶常驻
            # 在滚动区外面的组），滚轮就白滚了。守卫在事件到达控件之前跑，
            # 正好补上这一下；转成功就吃掉，免得冒上去再滚一遍。
            if forward_wheel(ev, obj):
                ev.accept()
                return True
            return False
        if isinstance(obj, QScrollBar):
            return False          # 滚动条滚的就是页面，放行让它自己滚
        if isinstance(obj, _VALUE_WIDGETS):
            # 参数不变，但页面照滚；accept 是为了别让平台层再往上冒一次（那样会滚两下）
            forward_wheel(ev, obj)
            ev.accept()
            return True
        return False


_guard = None


def install_wheel_guard(app=None):
    """给整个应用装上「滚轮不改参数」的守卫。启动时调一次即可（幂等）。"""
    global _guard
    if _guard is not None:
        return _guard
    from PyQt5.QtWidgets import QApplication
    app = app or QApplication.instance()
    if app is None:
        return None
    _guard = _WheelGuard(app)
    app.installEventFilter(_guard)
    return _guard
