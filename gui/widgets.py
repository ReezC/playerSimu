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

from pathlib import Path

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import (QAbstractScrollArea, QAbstractSlider,
                             QAbstractSpinBox, QApplication, QComboBox,
                             QDoubleSpinBox, QFrame, QScrollArea, QScrollBar,
                             QSlider, QSpinBox, QTabBar, QVBoxLayout, QWidget)

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
    """⚠⚠ **修掉 `QDoubleSpinBox`「小上限时打不进第一位」那个坑**（用户 2026-10-04 ✓ 报的就是它：
    「**我无法将框心力度小数点左边的数字改成 1**」✓）。

    病根（**离屏实测** ✓）：上限 `1.0`、当前 `0.15` 时，**选中整数位那个 `0` 再敲 `1`** ⇒ Qt 把它
    当 **`1.15`**（后面 `.15` 还在 ✗）⇒ **超上限** ⇒ `QDoubleValidator` 报 `Invalid` ⇒ **那一下
    就是敲不进去** ✗✗（实测 `text` 原地不动 ✓）；**全选**后敲 `1` 才行 ✓ —— 但没人该知道要这么做 ✗。
    ⇒ 两道补丁：
      · `validate`：**"是个数字、但越界"** 时报 **`Acceptable`** ✓ ⇒ 文字**进得去** ✓
        （⚠ **字母 / 格式错的仍 `Invalid`** ✗ 不放行 ✓）；
      · `valueFromText`：**自己把值夹进范围** ✓✗ —— ⚠⚠ **不能省** ✗✗：越界文本丢给基类**实测拿到
        的是 `0.0`**（不是上限 `1.0` ✗）⇒ 用户敲 `1.15` 回车会**跳到 `0.00`** ✗✗（比"敲不进去"还糟 ✓）；
        ⚠ 也别指望 `setCorrectionMode` 那两档 ✗ —— 实测（`Intermediate` / `CorrectToNearestValue`）
          **平台相关**（`offscreen` 恰好给 `1.0`、真实平台给 `0.00` ✗）⇒ 只有自己夹才**确定** ✓。
    ⚠ 再送一条"**焦点进来先全选**"（Tab 过来直接敲一个字 = **整段替换** ✓ 最省事 ✓）。
    """

    def validate(self, text, pos):
        _st, _tx, _ps = super().validate(text, pos)
        if _st == QValidator.Acceptable or not text:
            return _st, _tx, _ps
        try:
            float(text.replace(",", "."))   # ⚠ 只有"数字但越界"才放行 ✗（字母照旧不放 ✓）
        except ValueError:
            return _st, _tx, _ps
        return QValidator.Acceptable, _tx, _ps

    def valueFromText(self, text):
        try:
            _v = float(str(text).replace(",", "."))
        except ValueError:
            return self.value()             # 解析不了 ⇒ 保持原值 ✓（不猜）
        return min(self.maximum(), max(self.minimum(), round(_v, self.decimals())))
        #                        ↑ 夹进范围：`1.15 → 1.00` ✓ = 用户要的「改成 1」✓

    def focusInEvent(self, ev):
        super().focusInEvent(ev)
        _le = self.lineEdit()
        if _le is not None:
            _le.selectAll()


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


def scroll_page(widget, margins=(12, 12, 12, 12), spacing=8,
                h_scroll=False, min_width=None):
    """把**长参数页**装进滚动区，返回「页面布局」给调用方照旧 `addWidget` ✓。

    为什么要有这一处（`docs/UI规范.md`：**长面板放进 `QScrollArea`**）：

        直接 `QVBoxLayout(self)` 时，内容比页签高 ⇒ Qt 只能**硬挤** ⇒ 卡片里相邻的行被
        压到**互相重叠** —— "不支持滚动"和"行和行重叠"其实是**同一个病** ✗。
        （2026-09-27 用户就是这么报的："路线识别页签不支持滚动？现在攀爬参数组的行和行都
        重叠了"。）**一处实现**：谁做长面板都走这里，别再各写一遍 `QScrollArea` 那三行 ✗。

    做法照 `gui/player_panel.py` 那份样板 ✓：
        外层 `QVBoxLayout(widget)`（零边距）→ `QScrollArea`
        （`setWidgetResizable(True)` = **内容跟着视口走**，窗口拉大时卡片跟着变宽 ✓；
          `NoFrame` = 不画多余边框 ✓；**横向滚动条关掉** = 参数页只竖着滚 ✓）
        → 里面一个空 `QWidget` 当 holder ⇒ **返回 holder 的布局** ✓。

    ⚠ 参数里那些 `NoWheel*` 控件是**顺着父级**找最近滚动区来转发滚轮的
      （`forward_wheel` ✓ UI规范 §5）—— 以前面板里没滚动区，指针压在参数框上时
      滚轮就是"改不了参数、也滚不动页面"；装了这个之后才真正按规范工作 ✓。
    """
    outer = QVBoxLayout(widget)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    holder = QWidget()
    page = QVBoxLayout(holder)
    page.setContentsMargins(*margins)
    page.setSpacing(spacing)
    outer.addWidget(scroll_area(holder, widget, h_scroll=h_scroll,
                                min_width=min_width), 1)
    return page


def mount_scroll(layout, content, h_scroll=False, min_width=None, stretch=1):
    """把**已有的** `content` 套进滚动区，加到 `layout` 里 —— 返回滚动区控件。

    什么时候用 `scroll_page`、什么时候用它：

      · `scroll_page(widget)`：页面是"**空容器 + 一堆卡片**"（参数页那种 ✓）——
        它替你建 holder，返回**布局**，照旧 `addWidget` ✓；
      · `mount_scroll(layout, content)`：内容**已经有自己的控件、自己的样式**
        （A 机部署台的 `QFrame#Card` 明细框、左栏卡片列表 ✓）—— 只套滚动区、
        **不动内容本身** ✓。

    两个都走同一个 `_scroll_area`（**`QScrollArea` 只在这一处 new** ✓）——
    "一处实现"这条硬要求就靠它 ✓（`tools/check_ui.py` 会查裸写 ✗）。
    """
    scroll = scroll_area(content, layout.parentWidget(),
                         h_scroll=h_scroll, min_width=min_width)
    layout.addWidget(scroll, stretch)
    return scroll


def scroll_area(content, parent=None, h_scroll=False, min_width=None):
    """`QScrollArea` 的**唯一 new 处**（`scroll_page` / `mount_scroll` / 分栏都用它 ✓）。

    直接用它的时候只有一种：**滚动区自己要当"一个控件"交出去**（`QSplitter` 的某一栏、
    对话框的一侧 ✓ —— 那种地方没有"现成的布局"可以传，见 `gui/zone_editor.py::_panel`）。
    其余一律走 `scroll_page`（空容器 + 卡片 ✓）或 `mount_scroll`（内容自带样式 ✓）。

    口径（别再各写一份 ✗）：
      · `setWidgetResizable(True)`：内容跟着视口走（窗口拉宽时卡片跟着变宽 ✓）；
      · `NoFrame`：页面/页签自己已经有边框了，别套两层 ✓；
      · 横向默认**关掉**（参数页只竖着滚 ✓）；`h_scroll=True` 才按需出横向滚动条
        —— 多栏（`QSplitter`）那类"栏宽可以拖窄"的场景需要它 ✓；
      · `min_width`：给分栏用的最小宽度（**由人定**，别让内容算 —— UI规范 §4 ✓）。
    """
    scroll = QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(
        Qt.ScrollBarAsNeeded if h_scroll else Qt.ScrollBarAlwaysOff)
    if min_width:
        scroll.setMinimumWidth(int(min_width))
    scroll.setWidget(content)
    return scroll


_sound_player = None          # QMediaPlayer 常驻引用（局部变量会被 GC 掉 ⇒ 放不出声 ✗）


def sound_player():
    """常驻的 QMediaPlayer（「试听」的**停止**要读它的播放状态 ✓）。

    只在 **GUI 线程**用 ✓（收流线程要响就发信号给面板 ✓）。
    """
    global _sound_player
    if _sound_player is None:
        from PyQt5.QtMultimedia import QMediaPlayer

        _sound_player = QMediaPlayer()
    return _sound_player


def make_copyable(w):
    """让这个标签的文字**能被选中复制**（用户 2026-10-04 ✓ 第 4 条）→ 返回它自己 ✓。

    ⚠ 大部分地方**不用手调**：窗口收尾（`theme.bind_window_state` ✓）会把整窗标签扫一遍 ✓
      （见 `theme.enable_label_copy` 里那段"为什么按窗口扫" ✓）。这个函数给**当场就要**的
      场景用（比如字段标题刚建好 ✓）。
    """
    from PyQt5.QtCore import Qt

    try:
        w.setTextInteractionFlags(Qt.TextSelectableByMouse)
    except Exception:                                # noqa: BLE001
        pass
    return w


def field_tip(label, tip):
    """把提示（tooltip）**挂到"字段标题"上**（用户 2026-10-04 ✓ 第 3 条原话："现在是在哪里编辑
    （输入框、按钮、触控板等）就在哪里弹出，改成**指着字段标题时弹出**。例如「要标注的掉落物」"）
    → 返回那个标题 ✓。

    规矩见 `docs/UI规范.md` §6 ✓。两条要点：
      · **提示归标题**：鼠标指标题看"这个字段是干嘛的" ✓ —— 指输入框/下拉只剩"怎么填"本身 ✓；
      · ⚠ 搬完**必须把控件上那句删掉** ✗：两处都留 ⇒ 指控件弹两个框叠在一起 ✓ 比不搬还糟 ✓。
    """
    if label is not None:
        try:
            label.setToolTip(str(tip or ""))
        except Exception:                            # noqa: BLE001
            pass
        make_copyable(label)
    return label


def title_label(text, tip="", parent=None):
    """**字段标题标签**：文字**可复制** ✓ + 提示挂在它身上 ✓（新代码优先用它 ✓）。

    ⚠ 为什么要有这个工厂：标题到处手写 `QLabel("…")` ⇒ "能不能复制""提示挂哪"两件事就
      永远各写各的 ✗（正是用户 2026-10-04 第 3、4 条要收的那个口子 ✓）。
    """
    from PyQt5.QtWidgets import QLabel

    lbl = QLabel(text, parent)
    make_copyable(lbl)
    if tip:
        lbl.setToolTip(str(tip))
    return lbl


def stop_sound():
    """停止播放（「试听」点成「停止」✓）。没在播 = 空操作 ✓。"""
    if _sound_player is not None:
        _sound_player.stop()


#: 上一次「响警报」**实际怎么响的**（给人看 ✓ 状态行/日志用 ✓）：`"mp3"` / `"beep"` / `""` ✓
_last_sound = {"how": "", "path": ""}


def last_sound_note():
    """最近一次 `play_sound` **到底响没响、走的是哪条路** ⇒ `{"how": …, "path": …}` ✓。

    ⚠ 为什么非要留这个（2026-10-10 ✓ 用户："**没有任何警报**" ✓ 查出来的第二层）：
      `QMediaPlayer.play()` **失败不抛异常** ✗（编解码器缺失 / 音频后端插件不在 ✓ 都会静默失败 ✓）
      ⇒ 只看返回值**根本不知道响没响** ✓ ⇒ 加这一份"实际怎么响的"给日志 ✓。
    """
    return dict(_last_sound)


def _beep():
    """**保证听得见**的兜底（Windows 优先走 `winsound` ✓ 返回走了哪条路）。

    ⚠⚠ 为什么不能只用 `QApplication.beep()`（2026-10-10 ✓ 实测踩到）：Qt 在 Windows 上
      这个调用**经常是空操作** ✗（用户"没有警报"就是这么哑掉的 ✓）⇒ 换成
      `winsound.MessageBeep` ✓（系统警告音 ✓ 一定出声 ✓）。
    """
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        return "beep"
    except Exception:                     # noqa: BLE001 —— 非 Windows / 没 winsound ✓
        pass
    try:
        QApplication.beep()
        return "beep"
    except Exception:                     # noqa: BLE001
        return ""


def play_sound(path):
    """播放音效文件（测谎/掉线弹窗的「触发音效」+「试听」✓ **一处实现**）。

    相对路径按仓库根解析 ✓；文件缺失 / QtMultimedia 不可用 ⇒ 退回系统提示音 ✓
    —— **报警不能因为音效配置坏了就哑掉** ✓。返回 **True = 音效文件起播了**（「试听」
    按钮据它决定要不要变「停止」✓）；`QMediaPlayer` 必须在 **GUI 线程** ✓。

    ⚠⚠⚠ **2026-10-10 起：先响一声系统提示音，再起音效文件** ✓（用户："本地实时断线后
    没有重连警报" ✓ 现场查到底：**识别是对的**（日志 21:19:55 `st=login` ✓），
    哑在**声音这一层** ✗）。理由：
      · `QMediaPlayer.play()` **失败不抛** ✗（后端插件缺失 / 编解码器缺失 ⇒ 静默哑 ✗）
        ⇒ 光靠返回值判断不了 ✓；
      · 旧的兜底 `QApplication.beep()` 在 Windows 上**常常是空操作** ✗。
    ⇒ **报警这类"必须吵醒人"的声音，宁可多响一声，也不能哑** ✓（故不加开关 ✓）。
    """
    p = Path(path) if path else None
    if p is not None and not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    # ① 先保证"一定出声" ✓（见上面那段：Qt 的媒体管线能静默失败 ✗）
    _how = _beep()
    if p is not None and p.exists():
        try:
            from PyQt5.QtCore import QUrl
            from PyQt5.QtMultimedia import QMediaContent

            player = sound_player()
            player.setMedia(QMediaContent(QUrl.fromLocalFile(str(p))))
            player.play()
            _last_sound.update({"how": "mp3+%s" % _how if _how else "mp3",
                                "path": str(p)})
            return True
        except Exception:                 # noqa: BLE001 —— 编解码器缺失等 ✓
            pass
    _last_sound.update({"how": _how or "（连蜂鸣都没响成 ✗）", "path": str(p or "")})
    return False
