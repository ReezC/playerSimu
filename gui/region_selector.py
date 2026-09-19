"""全屏拖拽框选屏幕区域。

用户点「框选区域」后，抓一张虚拟屏幕（所有显示器合并）截图铺满屏幕，
拖拽出一个矩形，松开返回矩形在虚拟屏幕坐标系的 (x, y, w, h)。
Esc 取消返回 None。
"""

import numpy as np
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QDialog

from core import wincap


def _bgr_to_pixmap(img):
    img = np.ascontiguousarray(img)
    h, w, ch = img.shape
    qimg = QImage(img.tobytes(), w, h, img.strides[0], QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class RegionSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint
                            | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

        self._origin = None
        self._current = None

        vx, vy, vw, vh = wincap.virtual_screen()
        self._vx, self._vy = vx, vy

        try:
            full = wincap.grab_rect((vx, vy, vw, vh))
            self._pix = _bgr_to_pixmap(full)
        except Exception:
            self._pix = QPixmap(vw, vh)
            self._pix.fill(QColor("#000000"))

        self.setGeometry(vx, vy, vw, vh)

    # ---------------- 绘制 ----------------

    def paintEvent(self, ev):
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pix)

        if self._origin is None or self._current is None:
            return

        r = QRect(self._origin, self._current).normalized()
        # 框选区域之外压暗，框内保持原亮，一眼看清选了什么
        dim = QColor(0, 0, 0, 110)
        top = QRect(0, 0, self.width(), r.top())
        bottom = QRect(0, r.bottom() + 1, self.width(),
                       self.height() - r.bottom() - 1)
        left = QRect(0, r.top(), r.left(), r.height())
        right = QRect(r.right() + 1, r.top(),
                      self.width() - r.right() - 1, r.height())
        for qr in (top, bottom, left, right):
            if qr.width() > 0 and qr.height() > 0:
                p.fillRect(qr, dim)

        p.setPen(QPen(QColor("#4285f4"), 2))
        p.drawRect(r)
        p.drawText(r.left() + 4, max(14, r.top() - 6),
                   "%d×%d" % (r.width(), r.height()))

    # ---------------- 鼠标 ----------------

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._origin = ev.pos()
            self._current = ev.pos()
            self.update()

    def mouseMoveEvent(self, ev):
        if self._origin is not None:
            self._current = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._origin is None:
            return
        r = QRect(self._origin, ev.pos()).normalized()
        if r.width() >= 4 and r.height() >= 4:
            self.accept()
        else:
            self._origin = self._current = None
            self.update()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.reject()

    # ---------------- 结果 ----------------

    def result_rect(self):
        """返回屏幕坐标 (x, y, w, h)；未框选返回 None。"""
        if self._origin is None or self._current is None:
            return None
        r = QRect(self._origin, self._current).normalized()
        return (self._vx + r.x(), self._vy + r.y(), r.width(), r.height())


def select_region(parent=None):
    """弹框选区，返回 (x, y, w, h) 屏幕坐标；取消返回 None。"""
    sel = RegionSelector(parent)
    if sel.exec_() == QDialog.Accepted:
        return sel.result_rect()
    return None
