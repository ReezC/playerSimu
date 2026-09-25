"""可编辑的图片画布：显示标注框，支持拖拽修改和拉新框。

**坐标约定**（这是最容易写错的地方，所以定死）：

    场景坐标 == 图片像素坐标，图片左上角是 (0, 0)
    每个框的 rect 固定为 (0, 0, w, h)，位置由 pos() 表达
        → 框的像素坐标 = pos() + rect()

这样"场景坐标 → 像素坐标"是恒等映射，保存标注时不用做任何换算，
不会出现"改完保存坐标偏了"这种难查的 bug。
"""

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QKeySequence, QPainter, QPen
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsSimpleTextItem,
                             QGraphicsView)

from gui import theme
from perception.classes import (CLASS_MOB, CLASS_PLAYER, ZH_NAMES, bgr_to_hex,
                                bgr_to_rgb)

HANDLE = 10        # 控制点判定半径（图片像素）
MIN_SIZE = 8       # 框的最小边长

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

    def __init__(self, x, y, w, h, cls=1, manual=False, colors=None):
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self.cls = cls
        self.manual = manual   # 是否人工框（新建/修改过 = 人工，重标时保留）
        self._colors = colors or _box_colors()

        hex_, rgb = self._colors.get(cls, self._colors[1])
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setPen(QPen(QColor(hex_), 3 if manual else 2))
        self.setBrush(QBrush(QColor(*rgb, 36)))
        self.setZValue(10)

        # 类名标签：贴在框左上角上方
        self.label = QGraphicsSimpleTextItem(
            CLASS_NAMES.get(cls, str(cls)), self)
        self.label.setBrush(QBrush(QColor(hex_)))
        self.label.setZValue(1)
        self.label.setPos(-1, -18)

        self._mode = None
        self._press_scene = None
        self._orig_rect = None
        self._orig_pos = None
        self.on_changed = None   # 拖动/缩放后由画布注入的回调
        self.on_press = None     # 拖动/缩放开始前由画布注入的回调（撤销记录用）

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
        在空白处拖      → 拉一个新框
        Del             → 删除选中的框
        滚轮            → 缩放视图
        中键拖          → 平移画布（放大后看边角用）
        双击            → 适应窗口
    """

    boxes_changed = pyqtSignal()
    before_change = pyqtSignal()     # 破坏性操作前发（撤销记录快照）
    copy_requested = pyqtSignal()    # Ctrl+C
    paste_requested = pyqtSignal()   # Ctrl+V
    undo_requested = pyqtSignal()    # Ctrl+Z

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
        self._colors = _box_colors()   # 框颜色缓存（load 时刷新）

        self._draw_start = None
        self._rubber = None
        self._draw_cls = None     # 正在拉的框用哪个类（按下时定，Ctrl 会临时改）

    # ---------------- 载入 ----------------

    def load(self, pixmap, boxes=(), editable=True, fit=True):
        self.scene_.clear()
        self.boxes = []
        self.pix_item = None
        self._rubber = None
        self._draw_start = None

        self.pix_item = self.scene_.addPixmap(pixmap)
        self.pix_item.setPos(0, 0)
        self.pix_item.setZValue(0)

        w, h = pixmap.width(), pixmap.height()
        self.scene_.setSceneRect(0, 0, w, h)

        self.editable = editable
        self._colors = _box_colors()   # 读一次可视化配置，所有框共用
        for box in boxes:
            cls, x, y, bw, bh = box[0], box[1], box[2], box[3], box[4]
            manual = box[5] if len(box) > 5 else False
            self.add_box(x, y, bw, bh, cls, manual, self._colors)

        if fit:
            self.fit()      # 走 fit() 里的空场景守卫（0×0 上 fitInView 会崩）

    def img_size(self):
        if self.pix_item is None:
            return 0, 0
        pm = self.pix_item.pixmap()
        return pm.width(), pm.height()

    # ---------------- 框操作 ----------------

    def add_box(self, x, y, w, h, cls=1, manual=False, colors=None):
        it = BBoxItem(x, y, w, h, cls, manual, colors or self._colors)
        it.setEnabled(self.editable)
        it.on_changed = self.boxes_changed.emit
        it.on_press = self.before_change.emit   # 拖动前记录撤销快照
        self.scene_.addItem(it)
        self.boxes.append(it)
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
        return n

    def clear_boxes(self):
        for it in list(self.boxes):
            self.scene_.removeItem(it)
        self.boxes = []
        self.boxes_changed.emit()

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
            item = self.itemAt(e.pos())
            if item is None or item is self.pix_item:
                self._draw_cls = (CLASS_PLAYER
                                  if (e.modifiers() & Qt.ControlModifier)
                                  else self.current_cls)
                self._draw_start = self.mapToScene(e.pos())
                self._rubber = QGraphicsRectItem()
                hex_, _rgb = self._colors.get(self._draw_cls,
                                              self._colors[CLASS_MOB])
                self._rubber.setPen(QPen(QColor(hex_), 2, Qt.DashLine))
                self._rubber.setZValue(20)
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
            self.scene_.removeItem(self._rubber)
            self._rubber = None
            self._draw_start = None

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
