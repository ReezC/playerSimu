"""可编辑的图片画布：显示标注框，支持拖拽修改和拉新框。

**坐标约定**（这是最容易写错的地方，所以定死）：

    场景坐标 == 图片像素坐标，图片左上角是 (0, 0)
    每个框的 rect 固定为 (0, 0, w, h)，位置由 pos() 表达
        → 框的像素坐标 = pos() + rect()

这样"场景坐标 → 像素坐标"是恒等映射，保存标注时不用做任何换算，
不会出现"改完保存坐标偏了"这种难查的 bug。
"""

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QKeySequence, QPainter, QPainterPath,
                         QPen, QTransform)
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsPathItem,
                             QGraphicsPixmapItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsSimpleTextItem,
                             QGraphicsView)

from gui import theme
from perception.classes import (CLASS_MOB, CLASS_PLAYER, ZH_NAMES, bgr_to_hex,
                                bgr_to_rgb)

HANDLE = 10        # 控制点判定半径（图片像素）
MIN_SIZE = 8       # 框的最小边长

#: 左键在**空白处**拖时干什么（用户 2026-10-04 ✓ 第 3 条：质检台右上角那个「适应窗口」
#: 按钮改成"操作模式"下拉 ✓ 默认标注模式 ✓ 快捷键 Q/W ✓）：
MODE_LABEL = "label"    # 拉出一个**新框**（老行为 ✓ 默认 ✓）
MODE_SELECT = "select"  # 只拉一个**黑虚线**框来**选中**里面的框（**不建框** ✓ 批量删误检用 ✓）
#: 框选范围的颜色 —— 用户指定"**黑色虚线**"✓（标注模式那个是新框预览，所以按类别上色 ✓，
#: 两种模式一眼能分出来 ✓）
SELECT_BAND_COLOR = "#000000"

# 类别名/颜色都来自 perception/classes.py（唯一定义处）—— 类别清单只在那里维护
CLASS_NAMES = ZH_NAMES


def _box_colors():
    """质检台框颜色：{cls: (hex_str, (r, g, b))}。

    **每个类别的颜色都在设置里可改**（theme.class_colors 统一从 config/ui.yaml 读）。
    """
    return {cls: (bgr_to_hex(bgr), bgr_to_rgb(bgr))
            for cls, bgr in theme.class_colors().items()}


class BBoxItem(QGraphicsRectItem):
    """一个可移动 / 可缩放的标注框。"""

    def __init__(self, x, y, w, h, cls=1, manual=False, colors=None,
                 names=None):
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self.cls = cls
        self.manual = manual   # 是否人工框（新建/修改过 = 人工，重标时保留）
        self._colors = colors or _box_colors()

        hex_, rgb = self._colors.get(cls, self._colors[1])
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setPen(QPen(QColor(hex_), 3 if manual else 2))
        # ⭐ **只有描边、不填充**（2026-09-29 用户要求 ✓）：填充会让框中间蒙上一层色，
        #    看"框和目标对不对"反而费眼 ✗ —— 描边已足够标出边界 ✓。
        self.setBrush(QBrush(Qt.NoBrush))
        self.setZValue(10)

        # 类名标签：贴在框左上角上方
        # ⭐ **工作台配的类别名优先**（2026-09-30 用户报 ✓ 原话："为什么 YOLO 提案后
        #   框的名称是'玩家'？上面配的不是'shape'嘛"）—— 画布原来写死质检台那套
        #   中文名（0=玩家/1=怪物 ✗），通用工作台配的"shape"根本显示不出来 ✗。
        self.label = QGraphicsSimpleTextItem(self._label_text(names), self)
        self.label.setBrush(QBrush(QColor(hex_)))
        self.label.setZValue(1)
        self.label.setPos(-1, -18)

        self._mode = None
        self._press_scene = None
        self._orig_rect = None
        self._orig_pos = None
        self.on_changed = None   # 拖动/缩放后由画布注入的回调
        self.on_press = None     # 拖动/缩放开始前由画布注入的回调（撤销记录用）
        #: 按下**之前**由画布注入的回调（收 `e.modifiers()` ✓）—— 「选择模式」要借它把
        #: 上一次的选中清掉（点一下就换成只选这一个 ✓ 标准手感 ✓）；标注模式里画布不做啥 ✓。
        self.on_select = None

    def _label_text(self, names=None):
        """框上的类名文本：**注入的类别名表优先** ✓（`names` = 按 id 的名字列表 ✓）；
        没注入 / id 越界（比如拿别的数据集训的权重预测出类别 7 ✓）⇒ 退回质检台
        那套中文名（0=玩家/1=怪物 ✓），再不行就显示数字本身 ✓。"""
        if names and 0 <= self.cls < len(names):
            return str(names[self.cls])
        return CLASS_NAMES.get(self.cls, str(self.cls))

    # ---------------- 光标形状 ----------------

    def hoverMoveEvent(self, e):
        p = e.pos()
        r = self.rect()

        on_l = p.x() <= r.left() + HANDLE
        on_r = p.x() >= r.right() - HANDLE
        on_t = p.y() <= r.top() + HANDLE
        on_b = p.y() >= r.bottom() - HANDLE

        if (on_l and on_t) or (on_r and on_b):
            self.setCursor(Qt.SizeFDiagCursor)
        elif (on_r and on_t) or (on_l and on_b):
            self.setCursor(Qt.SizeBDiagCursor)
        elif on_l or on_r:
            self.setCursor(Qt.SizeHorCursor)
        elif on_t or on_b:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)

        super().hoverMoveEvent(e)

    def hoverLeaveEvent(self, e):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(e)

    # ---------------- 拖拽 / 缩放 ----------------

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton or not self.isEnabled():
            e.ignore()
            return

        p = e.pos()
        r = self.rect()

        mode = ""
        if p.x() <= r.left() + HANDLE:
            mode += "l"
        elif p.x() >= r.right() - HANDLE:
            mode += "r"
        if p.y() <= r.top() + HANDLE:
            mode += "t"
        elif p.y() >= r.bottom() - HANDLE:
            mode += "b"
        self._mode = mode or "move"

        self._press_scene = e.scenePos()
        self._orig_rect = QRectF(r)
        self._orig_pos = QPointF(self.pos())

        # ⭐ 按下之前先问画布一声（「选择模式」= 一次干净的单选：不按 Ctrl 就清掉别的 ✓）
        if self.on_select:
            self.on_select(e.modifiers())
        self.setSelected(True)
        if self.on_press:
            self.on_press()   # 拖动开始，通知画布记录撤销快照
        e.accept()

    def mouseMoveEvent(self, e):
        if self._mode is None:
            e.ignore()
            return

        d = e.scenePos() - self._press_scene
        r = QRectF(self._orig_rect)
        pos = QPointF(self._orig_pos)

        if self._mode == "move":
            pos += d
        else:
            if "l" in self._mode:
                r.setLeft(r.left() + d.x())
            if "r" in self._mode:
                r.setRight(r.right() + d.x())
            if "t" in self._mode:
                r.setTop(r.top() + d.y())
            if "b" in self._mode:
                r.setBottom(r.bottom() + d.y())

        if r.width() < MIN_SIZE:
            r.setWidth(MIN_SIZE)
        if r.height() < MIN_SIZE:
            r.setHeight(MIN_SIZE)

        # normalized() 处理"拖过头变成负宽高"的情况
        nr = r.normalized()
        self.prepareGeometryChange()
        self.setRect(0, 0, nr.width(), nr.height())
        self.setPos(pos + nr.topLeft())
        e.accept()

    def mouseReleaseEvent(self, e):
        self._mode = None
        # 位置或尺寸真的变了才通知 —— 否则点一下（没拖动）也会标记为已修改
        if self._orig_pos is not None:
            if self.pos() != self._orig_pos or self.rect() != self._orig_rect:
                self.manual = True   # 被拖动/缩放过的框，视为人工框
                hex_ = self._colors.get(self.cls, self._colors[1])[0]
                self.setPen(QPen(QColor(hex_), 3))   # 人工框加粗
                if self.on_changed:
                    self.on_changed()
        super().mouseReleaseEvent(e)


class ZoomPanView(QGraphicsView):
    """看图的三种基本操作：滚轮=缩放、中键拖=平移、双击=适应窗口。

    **为什么抽成一个基类**：质检台（ImageCanvas）和「手动目测标定」的视图都要这套
    交互，以前各写了一份 —— 于是同一个操作在两处手感不同（缩进步进 1.15 / 1.25、
    标定那边干脆没有双击适应），用起来就像 bug。现在只有这一份实现，谁要改一起改。

    子类的鼠标事件只要**不处理中键**、并且最终 `super()` 上来即可复用；左键留给
    子类自己（质检台用左键拉框、标定弹窗用左键拖模板）。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # 缩放时以鼠标位置为锚点，手感自然
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self._panning = False
        self._pan_start = None

    def fit(self):
        """适应窗口。子类有更好的基准（比如只框住图片）时覆盖它。

        **空场景必须直接返回**：`fitInView` 拿到 0×0 的矩形会算出非法缩放，
        Qt 内部随后越界 —— 实测表现为**偶发进程崩溃**（gui_smoke 0xC0000005，
        8 次里崩 2 次），而且只在"画布是一张空图"时才出现，很难联想到这里。
        """
        r = self.scene().sceneRect()
        if r.width() < 1 or r.height() < 1:
            return
        self.fitInView(r, Qt.KeepAspectRatio)

    # ---------------- 事件 ----------------

    def mousePressEvent(self, e):
        # 中键拖动 = 平移画布。放大看细节时，边角的框/模板经常在视图外，
        # 没有平移就只能反复缩小放大，很难用。
        if e.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._panning and self._pan_start is not None:
            d = e.pos() - self._pan_start
            self._pan_start = e.pos()

            h = self.horizontalScrollBar()
            v = self.verticalScrollBar()
            h.setValue(h.value() - d.x())
            v.setValue(v.value() - d.y())

            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else (1.0 / 1.15)
        self.scale(f, f)
        e.accept()

    def mouseDoubleClickEvent(self, e):
        # 双击 = 适应窗口，比去点按钮快
        self.fit()
        e.accept()


class ImageCanvas(ZoomPanView):
    """显示图片 + 可编辑的框（质检台）。

    交互（缩放/平移/双击见 ZoomPanView）：
        在框上拖        → 移动
        在框边缘拖      → 缩放（8 个方向）
        在空白处拖      → **标注模式**：拉一个新框 ✓
                          **选择模式**：拉一个黑虚线框 ⇒ 选中被框住的框（不建框 ✓）
        Del             → 删除选中的框（选择模式框一批 ⇒ 一次删掉 ✓）
        滚轮            → 缩放视图
        中键拖          → 平移画布（放大后看边角用）
        双击            → 适应窗口

    两种模式见 `set_mode` / `MODE_LABEL` / `MODE_SELECT` ✓（快捷键 Q / W 在质检台上 ✓）。
    """

    boxes_changed = pyqtSignal()
    before_change = pyqtSignal()     # 破坏性操作前发（撤销记录快照）
    copy_requested = pyqtSignal()    # Ctrl+C
    paste_requested = pyqtSignal()   # Ctrl+V
    undo_requested = pyqtSignal()    # Ctrl+Z
    #: 选中数变了（框选/点框都发 ✓）—— 面板拿它写一句"选中 N 个框" ✓。
    #: ⚠ **选中不算修改**：不许借此发 `boxes_changed` ✗（那会把"只看了一遍"的帧标成
    #:   人工改过 ✓ 见 `_select_in_rect` ✓）。
    selection_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QBrush(QColor("#202124")))

        self.pix_item = None
        self.boxes = []
        self.editable = True
        self.current_cls = CLASS_MOB   # 新建框的默认类别（下拉框选的）
        #: ⭐ **类别名表**（id → 名 ✓，通用工作台注入 ✓）—— `None` = 质检台那套
        #:   中文名（0=玩家/1=怪物 ✓）。见 `set_class_names` ✓。
        self.class_names = None
        self._colors = _box_colors()   # 框颜色缓存（load 时刷新）
        #: ⭐ **质检台蒙版**层（2026-09-29 用户要求 ✓）：画面上罩一层灰、**框在蒙版上面**
        #:   —— 观察"框和目标对不对"时把画面压暗、框更跳 ✓。0 = 不罩 ✓。
        self._mask_alpha = 0.0
        self._mask_item = None

        self._draw_start = None
        self._rubber = None
        self._draw_cls = None     # 正在拉的框用哪个类（按下时定，Ctrl 会临时改）
        #: ⭐ **操作模式**（用户 2026-10-04 ✓ 第 3 条）—— `MODE_LABEL` 拉新框（老行为 ✓）、
        #:   `MODE_SELECT` 拉黑虚线框**选中**里面的框（不建框 ✓）。默认标注模式 ✓。
        self.mode = MODE_LABEL
        self._select_band = False   # 手上这个橡皮筋是"框选"还是"建框"（按下时定 ✓）
        self._band_add = False      # 框选时按住了 Ctrl（= 加选，别清旧的 ✓）
        self._apply_mode_cursor()
        #: ⭐ 「实时小地图」那一层（`set_live_patch` ✓）—— 只在**路线识别 → 地形图**
        #: 用得上：把 A 机实时小地图面板**按标定摆到地形图上**，方便放大看对齐 ✓。
        #: ⚠ `load()` 会 `scene_.clear()` ⇒ 每一轮都要**重新挂**（见 load ✓），
        #: 所以这里只当"有没有"的哨兵，别把它当常驻对象 ✗。
        self._live_item = None
        #: 那一层的**浓淡**（0~1；用户 2026-09-29 要的「透明度」参数 ✓）。
        #: ⚠ 存在这里、`load()` 重挂时**再贴一次** —— 否则换图/重画地形图之后
        #: 它悄悄回到不透明（看着像"参数没生效" ✗）。
        self._live_alpha = 0.8

    @staticmethod
    def _tag_helper(item):
        """标成**辅助层**（蒙版 / 实时小地图贴图 ✓）：

        · **不吃鼠标**（`setAcceptedMouseButtons(NoButton)` ✓）—— 它们只是"看"的 ✓；
        · **不参与"空白"判定**（`setData(0, "helper")` ✓，`mousePressEvent` 里跳过 ✓）
          —— ⚠ 否则蒙版盖满全图 ⇒ 永远判不出"空白" ⇒ **拉不出框** ✗（实测 ✓）。
        """
        item.setData(0, "helper")
        item.setAcceptedMouseButtons(Qt.NoButton)

    def _apply_mask(self):
        """重铺**蒙版路径**：整幅画面、**挖掉所有框**（2026-09-30 用户报 ✓ 原话：
        "框里的怪还是受了灰蒙版影响"）—— 蒙版只压暗**框外**的画面 ✓，**框里**保持
        原亮度 ✓；框挪/缩/增/删都跟着重铺（见各调用点 ✓）。没画面 = 全空路径 ✓。"""
        if self._mask_item is None:
            return
        w, h = self.img_size()
        path = QPainterPath()
        if w > 0 and h > 0:
            path.addRect(0.0, 0.0, float(w), float(h))
        for it in self.boxes:
            hole = QPainterPath()
            hole.addRect(it.sceneBoundingRect())
            path = path.subtracted(hole)
        self._mask_item.setPath(path)

    def _on_item_changed(self):
        """框几何变了（拖动/缩放）⇒ **蒙版洞跟着走** + 照常发 `boxes_changed` ✓。"""
        self._apply_mask()
        self.boxes_changed.emit()

    # ---------------- 操作模式（用户 2026-10-04 ✓ 第 3 条）----------------

    def set_mode(self, mode):
        """切**操作模式**（`MODE_LABEL` / `MODE_SELECT` ✓）→ 返回生效的那个 ✓。

        `MODE_SELECT`：空白处拖**不建框** ✓，只拉一个**黑色虚线**框 ⇒ 松手**选中被框住的框**
        （拖完按 Del 就能一次删掉一批 ✗ 误检 ✓ —— 这正是用户要它解决的事 ✓）。
        🚫 传别的值一律当 `MODE_LABEL`（老行为 ✓）—— 宁可回到最熟的那个，也别进一个
        "什么都不做"的僵尸模式 ✗。
        """
        self.mode = MODE_SELECT if str(mode) == MODE_SELECT else MODE_LABEL
        self._apply_mode_cursor()
        return self.mode

    def _apply_mode_cursor(self):
        """光标：标注模式十字（要画框 ✓）、选择模式箭头（要挑框 ✓）。"""
        try:
            self.viewport().setCursor(
                Qt.ArrowCursor if self.mode == MODE_SELECT else Qt.CrossCursor)
        except Exception:                            # noqa: BLE001
            pass

    def _on_box_pressed(self, modifiers):
        """某个框被按下**之前**（`BBoxItem.on_select` ✓）。

        **选择模式**：不按 Ctrl ⇒ 先清掉别的选中（点一下就换成只选这一个 ✓ 是"选择"该有的
          手感 ✓）；按住 Ctrl ⇒ 加选 ✓。
        **标注模式**：什么都不做（保持老行为：点几个就选几个 ✓ 别在本轮顺手改掉它 ✗）。
        """
        if self.mode == MODE_SELECT and not (modifiers & Qt.ControlModifier):
            self.clear_selection()

    def clear_selection(self):
        """取消所有选中 → 返回被清掉的个数 ✓（框选新起一次时先清 ✓）。"""
        n = 0
        for it in self.boxes:
            if it.isSelected():
                it.setSelected(False)
                n += 1
        if n:
            self.selection_changed.emit(0)
        return n

    def selected_count(self):
        return sum(1 for it in self.boxes if it.isSelected())

    def _select_in_rect(self, rect, add=False):
        """把**与 rect 相交**的框选中 → 返回选中总数 ✓（框选那一下的实际动作 ✓）。

        ⚠⚠ **绝不发 `boxes_changed`** ✗：选中不是修改 —— 发了的话"只是框选看了一眼"的帧
          会被当成**人工改过**（`ReviewPanel._on_boxes_changed` ⇒ `_dirty` ⇒ 翻页时写盘 ✗），
          于是「只看未经人工修改」里忽进忽出 ✓（同 `_save_current` 那条"以落盘为准"的教训 ✓）。
        """
        r = QRectF(rect)
        for it in self.boxes:
            hit = it.sceneBoundingRect().intersects(r)
            if hit:
                it.setSelected(True)
            elif not add:
                it.setSelected(False)
        n = self.selected_count()
        self.selection_changed.emit(n)
        return n

    # ---------------- 载入 ----------------

    def load(self, pixmap, boxes=(), editable=True, fit=True):
        self.scene_.clear()
        self.boxes = []
        self.pix_item = None
        self._rubber = None
        self._draw_start = None
        self._select_band = False     # 换帧时手上那根橡皮筋一律作废 ✓（别把上一帧的框选带过来 ✗）
        self._band_add = False

        self.pix_item = self.scene_.addPixmap(pixmap)
        self.pix_item.setPos(0, 0)
        self.pix_item.setZValue(0)

        w, h = pixmap.width(), pixmap.height()
        self.scene_.setSceneRect(0, 0, w, h)

        # ⭐ **蒙版层**：插在画面**之后**、框**之前**（同一 z 下按插入顺序堆叠 ⇒
        #   天然"画面 < 蒙版 < 框" ✓）—— alpha=0 时也照建（透明不可见 ✓），
        #   这样 `set_mask_alpha` 拖滑条时不用重新 load ✓。
        # ⭐ **蒙版层**：插在画面**之后**、框**之前**，z=5 显式压序 ✓。
        #   ⚠⚠ **是"挖了洞"的路径层**，不是一整块矩形（2026-09-30 用户报 ✓ 原话：
        #   "框里的怪还是受了灰蒙版影响"）—— 蒙版只压暗**框外**的画面 ✓，**框里**
        #   保持原亮度（看"框里的目标"才不费眼 ✓）；洞随框的增删/挪/缩实时重铺
        #   （`_apply_mask` ✓）。alpha=0 时也照建（透明不可见 ✓），拖滑条不用重 load ✓。
        #   ⚠⚠ **必须标成辅助层**（`_tag_helper` ✓）：它是盖满全图的一块，不标的话
        #     `mousePressEvent` 的"空白"判定永远命中的是它 ⇒ **哪儿都拉不出框** ✗
        #     （2026-09-30 用户报"质检台不能标框了"的根因 ✓）。
        self._mask_item = QGraphicsPathItem()
        self._mask_item.setPen(QPen(Qt.NoPen))
        self._mask_item.setBrush(QBrush(QColor(96, 96, 96)))
        self._mask_item.setZValue(5)          # 画面(0) < 蒙版(5) < 框(10) ✓
        self._tag_helper(self._mask_item)
        self.scene_.addItem(self._mask_item)
        self._mask_item.setOpacity(self._mask_alpha)

        # ⭐ 实时小地图那层要**重新挂**（上面 `scene_.clear()` 连它一起删了 ✗）——
        #   落在最上面（ZValue 10）、默认不显示，等 `set_live_patch` 来放 ✓。
        self._live_item = QGraphicsPixmapItem()
        self._live_item.setZValue(10)
        self._live_item.setVisible(False)
        self._live_item.setOpacity(self._live_alpha)   # 重挂也带着浓淡 ✓（见那个属性）
        self._tag_helper(self._live_item)     # 贴图层只是"看"的 ✓ 别挡交互/空白判定 ✓
        self.scene_.addItem(self._live_item)

        self.editable = editable
        self._colors = _box_colors()   # 读一次可视化配置，所有框共用
        for box in boxes:
            cls, x, y, bw, bh = box[0], box[1], box[2], box[3], box[4]
            manual = box[5] if len(box) > 5 else False
            self.add_box(x, y, bw, bh, cls, manual, self._colors)

        self._apply_mask()            # 载入即铺（框里的画面不蒙灰 ✓）

        if fit:
            self.fit()      # 走 fit() 里的空场景守卫（0×0 上 fitInView 会崩）

    def img_size(self):
        if self.pix_item is None:
            return 0, 0
        pm = self.pix_item.pixmap()
        return pm.width(), pm.height()

    def set_class_names(self, names):
        """换**类别名表**（id → 名 ✓，通用工作台用 ✓）—— 已有的框**当场刷新** ✓；
        `None`/空 = 回质检台那套中文名 ✓（框标签的取值规则见 `BBoxItem._label_text` ✓）。"""
        self.class_names = [str(n) for n in (names or []) if str(n)] or None
        for it in self.boxes:
            it.label.setText(it._label_text(self.class_names))

    def set_mask_alpha(self, alpha):
        """只改**蒙版层**的浓淡（不重取帧 ✓）—— 拖滑条要**跟手** ✓。返回生效值。

        蒙版罩在画面上、**框画在蒙版上面**（插入顺序天然保证 ✓）——
        看帧时压暗画面、框更跳 ✓；0 = 不罩（老观感 ✓）。
        """
        self._mask_alpha = max(0.0, min(1.0, float(alpha)))
        if self._mask_item is not None:
            self._mask_item.setOpacity(self._mask_alpha)
        return self._mask_alpha

    def set_live_patch_alpha(self, alpha):
        """只改那一层的**浓淡**（不重取帧、不挪位）—— 拖动条要**跟手** ✓。

        为什么单独一条：`set_live_patch` 要重新摆一遍（调用方还得先取一帧面板 ✗）；
        拖浓淡时人只想要"立刻看得清一点"，不该去动帧 ✓。返回值 = 生效后的浓淡 ✓。
        """
        self._live_alpha = max(0.0, min(1.0, float(alpha)))
        if self._live_item is not None:
            self._live_item.setOpacity(self._live_alpha)
        return self._live_alpha

    def set_live_patch(self, pixmap, origin=(0.0, 0.0), kx=1.0, ky=1.0,
                       opacity=None):
        """⭐ 放/换/收「**实时小地图**」那一层（用户 2026-09-29 第 ⑦ 条 ✓）。

        用途（原话："现在实时画面小地图比较小很难看清是否对准，可以在路线识别页签→
        地形图里的图里实时滚动收流图（我自己控制底图），这样我就方便观察局部了"）：
        把 A 机那块小地图面板**按标定摆到地形图上** —— 于是"面板里的内容"和"它该在
        底图的哪一块"在**同一张图**上重叠，放大一看就知道标定/滚动对不对 ✓。

        参数都是**这张画布的坐标系**（= 地形图/叠加图像素，见调用方 `_live_map_rect` ✓）：
        `origin` = 面板图左上角落在哪、`kx/ky` = 面板像素 → 画布像素的倍数。
        `pixmap=None`（或空图）⇒ **收起来**（不显示，不是留一张空的 ✗）。
        """
        if self._live_item is None:          # 还没 load 过地形图：没处放 ✓
            return False
        if pixmap is None or pixmap.isNull():
            self._live_item.setVisible(False)
            return False
        self._live_item.setPixmap(pixmap)
        self._live_item.setTransform(QTransform().scale(float(kx), float(ky)))
        self._live_item.setPos(float(origin[0]), float(origin[1]))
        # `opacity=None` ⇒ 用**上次设的那个**（拖动条拖过的 ✓）—— 调用方不必每次都带 ✓
        if opacity is not None:
            self.set_live_patch_alpha(opacity)
        self._live_item.setVisible(True)
        return True

    # ---------------- 框操作 ----------------

    def add_box(self, x, y, w, h, cls=1, manual=False, colors=None):
        it = BBoxItem(x, y, w, h, cls, manual, colors or self._colors,
                      self.class_names)
        it.setEnabled(self.editable)
        it.on_changed = self._on_item_changed   # 框挪/缩 ⇒ 蒙版洞跟手 + 照常发信号 ✓
        it.on_press = self.before_change.emit   # 拖动前记录撤销快照
        it.on_select = self._on_box_pressed     # 选择模式：按下一框就换成只选它 ✓
        self.scene_.addItem(it)
        self.boxes.append(it)
        self._apply_mask()                      # 新框 = 新洞 ✓
        return it

    def get_boxes(self):
        """返回 [(cls, x, y, w, h, manual), ...]，像素坐标。"""
        out = []
        for it in self.boxes:
            r = it.rect()
            p = it.pos()
            out.append((it.cls, p.x(), p.y(), r.width(), r.height(), it.manual))
        return out

    def get_selected_boxes(self):
        """返回选中框的 [(cls, x, y, w, h, manual)]（复制用）。"""
        out = []
        for it in self.boxes:
            if it.isSelected():
                r = it.rect()
                p = it.pos()
                out.append((it.cls, p.x(), p.y(), r.width(), r.height(), it.manual))
        return out

    def replace_boxes(self, boxes):
        """清掉现有框，用 boxes 重建（撤销/粘贴恢复用）。"""
        for it in list(self.boxes):
            self.scene_.removeItem(it)
        self.boxes = []
        for cls, x, y, w, h, manual in boxes:
            self.add_box(x, y, w, h, cls, manual)

    def count(self):
        return len(self.boxes)

    def remove_selected(self):
        n = 0
        for it in list(self.boxes):
            if it.isSelected():
                n += 1
        if n:
            self.before_change.emit()   # 删除前记录撤销快照
            for it in list(self.boxes):
                if it.isSelected():
                    self.scene_.removeItem(it)
                    self.boxes.remove(it)
            self.boxes_changed.emit()
            self._apply_mask()                  # 删框 = 补回那块蒙版 ✓
        return n

    def clear_boxes(self):
        for it in list(self.boxes):
            self.scene_.removeItem(it)
        self.boxes = []
        self.boxes_changed.emit()
        self._apply_mask()                      # 全删 = 整幅重新蒙上 ✓

    def set_editable(self, on):
        self.editable = bool(on)
        for it in self.boxes:
            it.setEnabled(self.editable)

    def fit(self):
        # 基准就是图片本身（sceneRect 在 load 里按图片尺寸设）；
        # 空画布那一下由 ZoomPanView.fit 直接返回（见那里的崩溃说明）。
        super().fit()

    # ---------------- 事件 ----------------
    # 中键平移 / 滚轮缩放 / 双击适应窗口 都在 ZoomPanView 里，这里只管左键：
    # 空白处拖 = 拉新框，框上拖 = 移动/缩放。

    def mousePressEvent(self, e):
        # 空白处按下 = 开始拉新框（不用切换工具，符合直觉）。
        # **按住 Ctrl = 这一框按「玩家」画**：质检时玩家框少但要准，为了它来回切
        # 下拉框很烦；按 Ctrl 画完就回到下拉框里选的类别，不用切回去。
        if e.button() == Qt.LeftButton and self.editable and self.pix_item is not None:
            # ⭐ **辅助层不参与"空白"判定**（蒙版/贴图只是"看"的 ✓）—— 蒙版盖满
            #   全图，`itemAt`（只看最顶那个）永远命中的是它 ⇒ 拉不出框 ✗（实测 ✓）。
            #   真空白 = 除辅助层外只剩底图 ✓；点在框上 ⇒ 这儿不进、事件交给框 ✓。
            item = self.itemAt(e.pos())
            if (item is None or item is self.pix_item
                    or (item.data(0) == "helper"
                        and all(it is self.pix_item
                                for it in self.items(e.pos())
                                if it.data(0) != "helper"))):
                self._draw_start = self.mapToScene(e.pos())
                self._rubber = QGraphicsRectItem()
                self._rubber.setZValue(20)
                if self.mode == MODE_SELECT:
                    # ⭐ **选择模式**（用户 2026-10-04 ✓ 第 3 条）：只拉一个**黑色虚线**框，
                    #   松手**选中被框住的框**、**不建框** ✓（批量删误检用 ✓）。
                    #   ⚠ 不按 Ctrl ⇒ 先把旧的选中清掉（"框一次 = 选这一批" ✓）；
                    #     按住 Ctrl ⇒ 加选 ✓。
                    self._band_add = bool(e.modifiers() & Qt.ControlModifier)
                    self._select_band = True
                    self._draw_cls = None
                    if not self._band_add:
                        self.clear_selection()
                    self._rubber.setPen(QPen(QColor(SELECT_BAND_COLOR), 2, Qt.DashLine))
                else:
                    self._draw_cls = (CLASS_PLAYER
                                      if (e.modifiers() & Qt.ControlModifier)
                                      else self.current_cls)
                    self._select_band = False
                    hex_, _rgb = self._colors.get(self._draw_cls,
                                                  self._colors[CLASS_MOB])
                    self._rubber.setPen(QPen(QColor(hex_), 2, Qt.DashLine))
                self.scene_.addItem(self._rubber)
                e.accept()
                return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._rubber is not None and self._draw_start is not None:
            cur = self.mapToScene(e.pos())
            self._rubber.setRect(QRectF(self._draw_start, cur).normalized())
            e.accept()
            return

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._rubber is not None:
            r = self._rubber.rect()
            band, add = self._select_band, self._band_add
            self._select_band = False
            self._band_add = False
            self.scene_.removeItem(self._rubber)
            self._rubber = None
            self._draw_start = None

            if band:
                # ⭐ 选择模式：松手只**选中**（不建框 ✓）—— 太小的一下当"点空白" ⇒ 只清选择 ✓
                #   ⚠ 这里**不发 `boxes_changed`**（选中不算修改 ✓ 见 `_select_in_rect` ✓）
                if r.width() >= MIN_SIZE and r.height() >= MIN_SIZE:
                    self._select_in_rect(r, add)
                self._draw_cls = None
                e.accept()
                return

            if r.width() >= MIN_SIZE and r.height() >= MIN_SIZE:
                self.before_change.emit()   # 加框前记录撤销快照
                # 类别按按下那一刻定的（Ctrl 按住时是玩家，见 mousePressEvent）
                self.add_box(r.x(), r.y(), r.width(), r.height(),
                             self._draw_cls if self._draw_cls is not None
                             else self.current_cls, manual=True)
                self.boxes_changed.emit()
            self._draw_cls = None
            e.accept()
            return

        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.matches(QKeySequence.Copy) and self.editable:
            self.copy_requested.emit()
            e.accept()
            return
        if e.matches(QKeySequence.Paste) and self.editable:
            self.paste_requested.emit()
            e.accept()
            return
        if e.matches(QKeySequence.Undo) and self.editable:
            self.undo_requested.emit()
            e.accept()
            return
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.editable:
            self.remove_selected()
            e.accept()
            return
        super().keyPressEvent(e)
