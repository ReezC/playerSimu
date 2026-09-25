"""框选区域（B 机侧入口）：在实时画面上框 / 抓虚拟屏幕框。

**交互实现都在 `gui/region_picker.py`**（放大镜 + 像素网格 + `+/-` 倍数那一套，
见 docs/UI规范.md §9）—— 这里只负责两件 B 机特有的事：

    · 在**实时画面**（numpy BGR）上框 → 转成 QPixmap（numpy 只在 B 机装）；
    · 抓**虚拟屏幕**（多显示器合并）用 `core.wincap`，比 Qt 只抓主屏更全。

函数名保持不变（`select_region` / `select_region_on_image`），调用方不用改；
A 机部署台那边不能走这里（没有 numpy/wincap），它直接用 region_picker 的 Qt 抓屏。
"""

import numpy as np
from PyQt5.QtGui import QColor, QImage, QPixmap

from core import wincap
from gui.region_picker import (LOUPE_PX, ZOOM_DEFAULT, ZOOM_MAX, ZOOM_MIN,  # noqa: F401
                               LoupeSelector, select_on_pixmap,
                               select_screen_region)

#: 兼容旧名字：以前这个模块里有个 RegionSelector 类，现在就是 LoupeSelector。
#: （注意构造参数不同：它收 QPixmap + origin，不再是 numpy 图。）
RegionSelector = LoupeSelector


def _bgr_to_pixmap(img):
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    qimg = QImage(img.tobytes(), w, h, img.strides[0], QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


def _grab_virtual_screen():
    """抓整个虚拟屏幕（跨显示器）→ (QPixmap, 屏幕原点)。"""
    vx, vy, vw, vh = wincap.virtual_screen()
    try:
        # 跨窗口抓只能走 BitBlt（WGC 只能抓单个窗口）
        full = wincap.grab_rect((vx, vy, vw, vh), use_wgc=False)
        pix = _bgr_to_pixmap(full)
    except Exception:
        pix = QPixmap(vw, vh)
        pix.fill(QColor("#000000"))
    return pix, (vx, vy)


def select_region(parent=None, zoom=ZOOM_DEFAULT):
    """抓屏框选 → (x, y, w, h) 屏幕坐标；取消返回 None。

    抓屏前会把 `parent` 所在窗口**藏起来**（不然截到的图里盖着工作台自己）。
    """
    return select_screen_region(parent, zoom=zoom, grab=_grab_virtual_screen)


def select_region_on_image(image, parent=None, zoom=ZOOM_DEFAULT):
    """在给定参考图（numpy BGR）上框选 → (x, y, w, h) 相对该图；取消返回 None。

    zoom 给起始倍数：细长目标（HP/MP 条、探针方块带）可以给大一点，默认 8。
    """
    return select_on_pixmap(_bgr_to_pixmap(image), parent=parent, zoom=zoom)
