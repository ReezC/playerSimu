"""全屏（或在参考图上）拖拽框选区域。

用户点「框选区域」后，抓一张虚拟屏幕（所有显示器合并）截图铺满屏幕，
拖拽出一个矩形，松开返回矩形在虚拟屏幕坐标系的 (x, y, w, h)。
Esc 取消返回 None。

**指针旁边有放大镜**（跟着鼠标走）：显示光标附近 N×N 个像素、按当前倍数放大，
带像素网格和准心，并标出光标落在源图的哪个像素（`x,y`）。
框选这种东西常常只差一两个像素（探针方块带、HP/MP 条边缘），
没有放大镜时只能靠感觉 —— 所以放大镜是贴着光标、且在拖拽时自动挪到框的外侧，
不会挡住正在定的边。

倍数用 `+` / `-` 调（4~16）；本仓库所有框选都走这里
（`select_region` 框屏幕、`select_region_on_image` 框实时画面）。
"""

import numpy as np
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QDialog

from core import wincap

#: 放大镜里显示多少个源图像素（正方形边长，单位=像素）
LOUPE_PX = 21
#: 放大倍数范围与默认值
ZOOM_MIN, ZOOM_MAX, ZOOM_DEFAULT = 4, 16, 8
#: 放大镜上方那行说明（画在图像上面，不压住像素）
CAPTION_H = 16


def _bgr_to_pixmap(img):
    img = np.ascontiguousarray(img)
    h, w, ch = img.shape
    qimg = QImage(img.tobytes(), w, h, img.strides[0], QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class RegionSelector(QDialog):
    def __init__(self, parent=None, image=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint
                            | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

        self._origin = None
        self._current = None
        self._cursor = None          # 光标当前位置（没按键时也要跟，放大镜要用）
        self._zoom = ZOOM_DEFAULT
        self._loupe_rect = None      # 上一次画放大镜的地方（只重画这块）

        if image is not None:
            # 在给定参考图（BGR）上框选：坐标相对这张图
            self._vx, self._vy = 0, 0
            self._pix = _bgr_to_pixmap(image)
            vw, vh = image.shape[1], image.shape[0]
        else:
            vx, vy, vw, vh = wincap.virtual_screen()
            self._vx, self._vy = vx, vy

            try:
                # 抓整个虚拟屏幕（跨窗口）：WGC 只能抓单个窗口，这里走 BitBlt
                full = wincap.grab_rect((vx, vy, vw, vh), use_wgc=False)
                self._pix = _bgr_to_pixmap(full)
            except Exception:
                self._pix = QPixmap(vw, vh)
                self._pix.fill(QColor("#000000"))

        self.setGeometry(self._vx, self._vy, vw, vh)

    # ---------------- 放大镜 ----------------

    def _loupe_parts(self, pos=None):
        """算出放大镜该画在哪 → (整体矩形, 图像矩形, 取源图的矩形, 准心矩形, 说明条)。

        准心 = 光标所在的那**一个像素**放大后的方块；贴边时源矩形会整体平移，
        准心跟着偏 —— 这样它始终指着真实像素，不会骗人。

        摆位两条规矩：**不压住光标**（左右两侧都试，挑不盖住光标那一侧的）、
        **整体不越界**（说明条贴在图像下方，避免贴底边时跑到窗口外）。
        """
        pos = self._cursor if pos is None else pos
        if pos is None or self._pix.isNull():
            return None
        z = self._zoom
        n = LOUPE_PX
        pw, ph = self._pix.width(), self._pix.height()
        if pw <= 0 or ph <= 0:
            return None
        sx, sy = pos.x() + self._vx, pos.y() + self._vy
        left = max(0, min(pw - n, sx - n // 2))
        top = max(0, min(ph - n, sy - n // 2))
        src = QRect(left, top, min(n, pw), min(n, ph))

        img = QRect(0, 0, src.width() * z, src.height() * z)
        w, h = self.width(), self.height()
        box_h = img.height() + CAPTION_H

        # 左右：优先侧随场景变（拖拽时放框的外侧），但两侧都算，挑不压住光标的
        if self._origin is not None and pos.x() >= self._origin.x():
            pref = (pos.x() - 16 - img.width(), pos.x() + 16)     # 往右拖 → 放左边
        elif self._origin is not None:
            pref = (pos.x() + 16, pos.x() - 16 - img.width())
        else:
            pref = (pos.x() + 18, pos.x() - 18 - img.width())
        cands = [max(0, min(w - img.width(), v)) for v in pref]
        left_px = cands[0]
        for v in cands:
            if not (v <= pos.x() <= v + img.width()):     # 这一侧不盖住光标
                left_px = v
                break
        img.moveLeft(left_px)

        # 上下：默认下方，下方放不下就翻到上方，再夹进窗口
        top_px = pos.y() + 18
        if top_px + box_h > h:
            top_px = pos.y() - 18 - box_h
        img.moveTop(max(0, min(h - box_h, top_px)))

        cap = QRect(img.left(), img.top() + img.height(), img.width(), CAPTION_H)
        mark = QRect(img.left() + (sx - src.x()) * z,
                     img.top() + (sy - src.y()) * z, z, z)
        return img.united(cap), img, src, mark, cap

    def _paint_loupe(self, p, parts):
        box, img, src, mark, caption = parts
        z = self._zoom
        p.save()
        # 放大要**看得到像素**：绝不能平滑（平滑会把两个像素糊成一个）
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.fillRect(caption, QColor(20, 22, 26, 235))
        p.fillRect(img, QColor(0, 0, 0))
        p.drawPixmap(img, self._pix, src)

        if z >= 6:      # 像素网格：一眼数得清差几个像素
            p.setPen(QPen(QColor(255, 255, 255, 45), 1))
            for i in range(1, src.width()):
                x = img.left() + i * z
                p.drawLine(x, img.top(), x, img.bottom())
            for j in range(1, src.height()):
                y = img.top() + j * z
                p.drawLine(img.left(), y, img.right(), y)

        p.setPen(QPen(QColor("#ff5252"), 2))     # 准心 = 光标那一个像素
        p.drawRect(mark)
        p.setPen(QPen(QColor("#5f6368"), 1))
        p.drawRect(box.adjusted(0, 0, -1, -1))

        p.setPen(QColor("#e8eaed"))
        p.drawText(caption.adjusted(5, 0, -5, 0),
                   Qt.AlignVCenter | Qt.AlignLeft,
                   "%d, %d    x%d" % (src.x() + (mark.x() - img.left()) // z,
                                      src.y() + (mark.y() - img.top()) // z, z))
        p.restore()

    def _repaint_loupe(self, new_box=None):
        """只重画放大镜那一小块（跟着鼠标跑，整屏重画太浪费）。"""
        old = self._loupe_rect
        if old is None and new_box is None:
            self.update()
            return
        if old is None:
            r = new_box
        elif new_box is None:
            r = old
        else:
            r = old.united(new_box)
        self.update(r.adjusted(-2, -2, 2, 2))

    # ---------------- 绘制 ----------------

    def paintEvent(self, ev):
        p = QPainter(self)
        reg = ev.rect()
        # 只重画被要求重画的那块（放大镜移动时通常就一两百像素见方）
        p.drawPixmap(reg, self._pix, reg)

        if self._origin is not None and self._current is not None:
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

        parts = self._loupe_parts()
        self._loupe_rect = parts[0] if parts else None
        if parts:
            self._paint_loupe(p, parts)

    # ---------------- 鼠标 ----------------

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._cursor = ev.pos()
            self._origin = ev.pos()
            self._current = ev.pos()
            self.update()

    def mouseMoveEvent(self, ev):
        self._cursor = ev.pos()          # 没按键也要记（放大镜跟手）
        if self._origin is not None:
            self._current = ev.pos()
            self.update()                # 拖拽中压暗区域在变，得整屏重画
            return
        parts = self._loupe_parts()
        self._repaint_loupe(parts[0] if parts else None)

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._origin is None:
            return
        r = QRect(self._origin, ev.pos()).normalized()
        if r.width() >= 4 and r.height() >= 4:
            self.accept()
        else:
            self._origin = self._current = None
            self.update()

    def leaveEvent(self, ev):
        self._cursor = None
        self.update()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_Escape:
            self.reject()
            return
        if k in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom = min(ZOOM_MAX, self._zoom + 1)
        elif k in (Qt.Key_Minus, Qt.Key_Underscore):
            self._zoom = max(ZOOM_MIN, self._zoom - 1)
        else:
            return
        self.update()

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


def select_region_on_image(image, parent=None):
    """在给定参考图（BGR）上框选，返回 (x, y, w, h) 相对该图的坐标；取消返回 None。"""
    sel = RegionSelector(parent, image=image)
    if sel.exec_() == QDialog.Accepted:
        return sel.result_rect()
    return None
