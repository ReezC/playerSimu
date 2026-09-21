"""可编辑的图片画布：显示标注框，支持拖拽修改和拉新框。

**坐标约定**（这是最容易写错的地方，所以定死）：

    场景坐标 == 图片像素坐标，图片左上角是 (0, 0)
    每个框的 rect 固定为 (0, 0, w, h)，位置由 pos() 表达
        → 框的像素坐标 = pos() + rect()

这样"场景坐标 → 像素坐标"是恒等映射，保存标注时不用做任何换算，
不会出现"改完保存坐标偏了"这种难查的 bug。
"""

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsView)

HANDLE = 10        # 控制点判定半径（图片像素）
MIN_SIZE = 8       # 框的最小边长


class BBoxItem(QGraphicsRectItem):
    """一个可移动 / 可缩放的标注框。"""

    # 类别 → 框颜色（和 data.yaml 的 class 对齐）
    COLORS = {
        0: ("#4285f4", (66, 133, 244)),    # player 蓝
        1: ("#34a853", (52, 168, 83)),      # mob 绿
        2: ("#fbbc04", (251, 188, 4)),      # drop 黄
        3: ("#ea4335", (234, 67, 53)),      # npc 红
    }

    def __init__(self, x, y, w, h, cls=1):
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self.cls = cls

        hex_, rgb = self.COLORS.get(cls, self.COLORS[1])
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setPen(QPen(QColor(hex_), 2))
        self.setBrush(QBrush(QColor(*rgb, 36)))
        self.setZValue(10)

        self._mode = None
        self._press_scene = None
        self._orig_rect = None
        self._orig_pos = None
        self.on_changed = None   # 拖动/缩放后由画布注入的回调

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
                if self.on_changed:
                    self.on_changed()
        super().mouseReleaseEvent(e)


class ImageCanvas(QGraphicsView):
    """显示图片 + 可编辑的框。

    交互：
        在框上拖        → 移动
        在框边缘拖      → 缩放（8 个方向）
        在空白处拖      → 拉一个新框
        Del             → 删除选中的框
        滚轮            → 缩放视图
        中键拖          → 平移画布（放大后看边角用）
        双击            → 适应窗口
    """

    boxes_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        # 缩放时以鼠标位置为锚点，手感自然
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#202124")))

        self.pix_item = None
        self.boxes = []
        self.editable = True
        self.current_cls = 1      # 新建框的默认类别（1=怪物，0=玩家）

        self._draw_start = None
        self._rubber = None

        self._panning = False
        self._pan_start = None

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
        for box in boxes:
            cls, x, y, bw, bh = box
            self.add_box(x, y, bw, bh, cls)

        if fit:
            self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)

    def img_size(self):
        if self.pix_item is None:
            return 0, 0
        pm = self.pix_item.pixmap()
        return pm.width(), pm.height()

    # ---------------- 框操作 ----------------

    def add_box(self, x, y, w, h, cls=1):
        it = BBoxItem(x, y, w, h, cls)
        it.setEnabled(self.editable)
        it.on_changed = self.boxes_changed.emit
        self.scene_.addItem(it)
        self.boxes.append(it)
        return it

    def get_boxes(self):
        """返回 [(cls, x, y, w, h), ...]，像素坐标。"""
        out = []
        for it in self.boxes:
            r = it.rect()
            p = it.pos()
            out.append((it.cls, p.x(), p.y(), r.width(), r.height()))
        return out

    def count(self):
        return len(self.boxes)

    def remove_selected(self):
        n = 0
        for it in list(self.boxes):
            if it.isSelected():
                self.scene_.removeItem(it)
                self.boxes.remove(it)
                n += 1
        if n:
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
        if self.pix_item is not None:
            self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)

    # ---------------- 事件 ----------------

    def mousePressEvent(self, e):
        # 中键拖动 = 平移画布。放大看细节时，边角的框经常在视图外，
        # 没有平移就只能反复缩小放大，很难用。
        if e.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return

        # 空白处按下 = 开始拉新框（不用切换工具，符合直觉）
        if e.button() == Qt.LeftButton and self.editable and self.pix_item is not None:
            item = self.itemAt(e.pos())
            if item is None or item is self.pix_item:
                self._draw_start = self.mapToScene(e.pos())
                self._rubber = QGraphicsRectItem()
                self._rubber.setPen(QPen(QColor("#fbbc04"), 2, Qt.DashLine))
                self._rubber.setZValue(20)
                self.scene_.addItem(self._rubber)
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

        if self._rubber is not None and self._draw_start is not None:
            cur = self.mapToScene(e.pos())
            self._rubber.setRect(QRectF(self._draw_start, cur).normalized())
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

        if self._rubber is not None:
            r = self._rubber.rect()
            self.scene_.removeItem(self._rubber)
            self._rubber = None
            self._draw_start = None

            if r.width() >= MIN_SIZE and r.height() >= MIN_SIZE:
                self.add_box(r.x(), r.y(), r.width(), r.height(), self.current_cls)
                self.boxes_changed.emit()
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

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.editable:
            self.remove_selected()
            e.accept()
            return
        super().keyPressEvent(e)
