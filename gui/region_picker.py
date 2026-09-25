"""框选区域（**本仓库唯一实现**）：拖一个矩形，带放大镜。

**为什么所有框选都走这里**（见 docs/UI规范.md §9）
    框选这种东西常常只差一两个像素：探针方块带的边、HP/MP 条的上沿、小地图
    面板的框 —— 没有放大镜就只能靠感觉，而"差一像素"的后果是 decode 错位 /
    血条采样到背景 / B 机匹配不上。以前这里一份、A 机部署台又一份（还没有
    放大镜），同一个操作在两处手感还不一样。现在只有这一个：

        select_on_pixmap()      在给定图（实时画面 / 截图）上框
        select_screen_region()  抓屏再框（可注入抓屏方式、可自动藏起调用方窗口）

**放大镜**（跟着鼠标走）：显示光标附近 21×21 个像素、按当前倍数放大，带像素
网格和准心，并标出光标落在源图的哪个像素、当前几倍。摆位两条规矩：**不压住
光标**、**整体不越界**。倍数用 `+` / `-`（4~16）—— **不用滚轮**：滚轮不许改
任何东西是硬规则（UI 规范 §1），框选窗口里也一样。

**只依赖 PyQt5**：A 机的部署台只有 PyQt5（不需要 numpy/opencv），所以这里
不能 import numpy、cv2、core.wincap —— 那些抓屏的事由调用方通过 `grab` 注入。
"""
import time

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog

#: 放大镜里显示多少个源图像素（正方形边长，单位=像素）
LOUPE_PX = 21
#: 放大倍数范围与默认值
ZOOM_MIN, ZOOM_MAX, ZOOM_DEFAULT = 4, 16, 8
#: 放大镜上方那行说明（画在图像上面，不压住像素）
CAPTION_H = 16
#: 小于这个尺寸的框当成误点（不返回结果，等于取消）
MIN_SIDE = 4


class LoupeSelector(QDialog):
    """全屏/全图的框选窗口，带放大镜。

    pix    要框的图（QPixmap）
    origin 这张图的左上角落在**结果坐标系**里的位置：
              · 在实时画面上框（画面坐标）      → (0, 0)
              · 抓了整屏框（屏幕坐标）          → 屏幕原点 (vx, vy)
           放大镜内部一律用**图像坐标**（不再加减 origin）—— 以前这里把屏幕
           原点加进图像坐标去索引 pixmap，多显示器（原点为负）时放大镜会指错
           地方、准心也跟着偏。
    zoom   起始倍数（细长目标，比如 HP/MP 条，可以给大一点，例如 12）
    """

    def __init__(self, pix, origin=(0, 0), zoom=ZOOM_DEFAULT, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint
                            | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

        self._pix = pix if pix is not None else QPixmap()
        self._ox, self._oy = int(origin[0]), int(origin[1])
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, int(zoom)))

        self._origin = None
        self._current = None
        self._cursor = None          # 光标当前位置（没按键时也要跟，放大镜要用）
        self._loupe_rect = None      # 上一次画放大镜的地方（只重画这块）

        w = max(1, self._pix.width())
        h = max(1, self._pix.height())
        self.setGeometry(self._ox, self._oy, w, h)

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
        sx, sy = pos.x(), pos.y()              # 图像坐标 == 窗口内坐标
        left = max(0, min(max(0, pw - n), sx - n // 2))
        top = max(0, min(max(0, ph - n), sy - n // 2))
        src = QRect(left, top, min(n, pw - left), min(n, ph - top))

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

    def _paint_hint(self, p):
        """左上角那行按键提示（`+/-` 是唯一的倍数入口，不写出来没人知道）。"""
        txt = "拖拽=框选    +/- = 倍数（当前 %d×）    Esc = 取消" % self._zoom
        r = QRect(0, 0, self.width(), 26)
        p.fillRect(r, QColor(20, 22, 26, 200))
        p.setPen(QColor("#e8eaed"))
        p.drawText(r.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, txt)

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
            p.drawText(r.left() + 4, max(40, r.top() - 6),
                       "%d×%d" % (r.width(), r.height()))

        # 提示画在提示条自己那块，别被放大镜的局部重画擦掉
        if reg.intersects(QRect(0, 0, self.width(), 26)):
            self._paint_hint(p)

        parts = self._loupe_parts()
        self._loupe_rect = parts[0] if parts else None
        if parts:
            self._paint_loupe(p, parts)

    # ---------------- 鼠标 / 键盘 ----------------

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
        if r.width() >= MIN_SIDE and r.height() >= MIN_SIDE:
            self.accept()
        else:
            # 误点（手一抖点一下）：清掉重框，而不是返回一个几像素的"区域"
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
        """结果坐标 (x, y, w, h)（= 图像坐标 + origin）；未框选返回 None。

        尺寸按 `QRect` 的**含两端**约定：从 (100,40) 拖到 (300,190) 覆盖的是
        201×151 个像素 —— 和抓屏/裁剪时真正取到的范围一致。
        """
        if self._origin is None or self._current is None:
            return None
        r = QRect(self._origin, self._current).normalized()
        return (self._ox + r.x(), self._oy + r.y(), r.width(), r.height())


# ══════════════════════════════════════════════════════════════
# 便捷入口
# ══════════════════════════════════════════════════════════════

def select_on_pixmap(pix, origin=(0, 0), parent=None, zoom=ZOOM_DEFAULT):
    """在给定图上框选 → (x, y, w, h)（加过 origin）；取消返回 None。"""
    sel = LoupeSelector(pix, origin=origin, zoom=zoom, parent=parent)
    return sel.result_rect() if sel.exec_() == QDialog.Accepted else None


def grab_primary_pixmap():
    """抓主屏 → (QPixmap, 屏幕原点)。

    默认抓屏方式只依赖 PyQt5 —— A 机部署台那边就够用（那里没装 numpy/opencv）。
    B 机想用能跨显示器的抓屏（core.wincap），通过 `grab=` 传进来即可。
    """
    from PyQt5.QtGui import QGuiApplication
    scr = QGuiApplication.primaryScreen()
    pix = scr.grabWindow(0) if scr is not None else QPixmap()
    return pix, (0, 0)


def select_screen_region(parent=None, zoom=ZOOM_DEFAULT, hide_owner=True, grab=None):
    """抓屏 → 框选 → **屏幕坐标** (x, y, w, h)；取消返回 None。

    hide_owner：抓屏前先把 parent 所在窗口藏起来 —— 不藏的话抓到的截图里
    盖着调用方自己的界面，人对着自己的窗口框游戏里的东西（部署台那条路上
    实测踩过）。窗口在 finally 里 show() 回来，中途取消也不会把界面弄丢。

    grab 返回值是 (QPixmap, origin)：默认 `grab_primary_pixmap`（Qt 抓主屏）。
    """
    if grab is None:
        grab = grab_primary_pixmap

    hidden = None
    if hide_owner and parent is not None:
        hidden = parent.window()
        if hidden.isVisible():
            hidden.hide()
            QApplication.processEvents()   # 先把重绘/隐藏做完
            time.sleep(0.25)               # 再等窗口真的从屏幕上消失

    try:
        pix, origin = grab()
        # parent 传 None：父窗口马上要被藏起来，挂上去 Qt 会连对话框一起隐掉
        return select_on_pixmap(pix, origin=origin, parent=None, zoom=zoom)
    finally:
        if hidden is not None:
            hidden.show()
