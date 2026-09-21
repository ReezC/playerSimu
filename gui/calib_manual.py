"""手动目测标定尺度弹窗。

自动标定在「目标太小/太少」时会失败（尺度离散度大、判定不可信）。
这里提供人工方式：把一只怪（或玩家）的 stand 帧当半透明模板叠加在画面上，
用户拖动模板到目标上、调缩放滑块让模板轮廓和目标重合，目测确定尺度。

原理：scale = 画面里目标尺寸 / 素材尺寸，正是模板匹配要用的缩放。
"""

from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap, QTransform
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsView, QHBoxLayout, QLabel, QSlider,
                             QVBoxLayout)

from core import wzexport
from core.imgio import imread   # 支持中文路径，cv2.imread 遇中文会静默失败

SLIDER_MIN = 200    # 0.200
SLIDER_MAX = 4000   # 4.000


def _load_frame_pixmap(path):
    """读一帧画面 → QPixmap（BGR→RGB）。"""
    img = imread(str(path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _load_trimmed_pixmap(path):
    """读带 alpha 的图并裁到 alpha 包围盒 → QPixmap。读不到返回 None。"""
    img = imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    b, al = img[:, :, :3], img[:, :, 3]
    ys, xs = np.nonzero(al > 128)
    if len(xs) < 20:
        return None
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rgba = cv2.cvtColor(b, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = al
    h, w = rgba.shape[:2]
    qimg = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class ZoomableView(QGraphicsView):
    """支持滚轮缩放 + 中键按住拖拽平移的视图。

    左键仍留给「拖动模板」用（ItemIsMovable），互不冲突。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._panning = False
        self._pan_start = None

    def wheelEvent(self, ev):
        factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning and self._pan_start is not None:
            delta = ev.pos() - self._pan_start
            self._pan_start = ev.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class CalibManualDialog(QDialog):
    def __init__(self, mobs, player_id, frames_dir, mob_scales=None,
                 mob_default=1.41, player_default=1.41, parent=None):
        super().__init__(parent)
        self.setWindowTitle("手动目测标定尺度")
        self.setMinimumSize(720, 620)

        self._mobs = [str(m) for m in (mobs or [])]
        self._player_id = (player_id or "").strip()
        self._frames_dir = Path(frames_dir)
        self._mob_scales = dict(mob_scales or {})   # {id:帧stem -> scale}
        self._mob_default = float(mob_default or 1.41)
        self._player_default = float(player_default or 1.41)
        self._scale = max(0.2, min(4.0, self._mob_default))

        self.name_map = wzexport.build_mob_name_map()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ---- 目标 / 模板 / 帧 ----
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(QLabel("标定目标"))
        self.cmb_target = QComboBox()
        self.cmb_target.addItem("怪物", "mob")
        self.cmb_target.addItem("玩家", "player")
        self.cmb_target.currentIndexChanged.connect(self._on_target)
        top.addWidget(self.cmb_target)

        top.addWidget(QLabel("模板"))
        self.cmb_tpl = QComboBox()
        self.cmb_tpl.setMaxVisibleItems(20)
        self.cmb_tpl.currentIndexChanged.connect(self._reload_template)
        top.addWidget(self.cmb_tpl, 1)

        top.addWidget(QLabel("画面帧"))
        self.cmb_frame = QComboBox()
        self.cmb_frame.setMaxVisibleItems(20)
        self.cmb_frame.currentIndexChanged.connect(self._reload_frame)
        top.addWidget(self.cmb_frame, 1)
        root.addLayout(top)

        # ---- 视图 ----
        self.view = ZoomableView()
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setStyleSheet("background:#202124; border:1px solid #3c4043;")
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        root.addWidget(self.view, 1)

        # ---- 缩放滑块 + 数字输入 ----
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("缩放"))
        self.sld = QSlider(Qt.Horizontal)
        self.sld.setRange(SLIDER_MIN, SLIDER_MAX)
        self.sld.setValue(int(round(self._scale * 100)))
        self.sld.valueChanged.connect(self._on_slider)
        row.addWidget(self.sld, 1)
        self.sp_scale = QDoubleSpinBox()
        self.sp_scale.setRange(0.05, 8.0)
        self.sp_scale.setDecimals(3)
        self.sp_scale.setSingleStep(0.01)
        self.sp_scale.setValue(self._scale)
        self.sp_scale.setSuffix("×")
        self.sp_scale.setMinimumWidth(90)
        self.sp_scale.valueChanged.connect(self._on_spin)
        row.addWidget(self.sp_scale)
        self.lbl_scale = QLabel()
        self.lbl_scale.setMinimumWidth(90)
        self.lbl_scale.setStyleSheet("color:#80868b;")
        row.addWidget(self.lbl_scale)
        self.ck_mirror = QCheckBox("镜像")
        self.ck_mirror.setToolTip("水平翻转模板（目标朝向和模板相反时用）")
        self.ck_mirror.toggled.connect(self._on_mirror)
        row.addWidget(self.ck_mirror)
        root.addLayout(row)

        hint = QLabel("拖动模板到目标上，调缩放让轮廓重合，确定即取当前缩放为尺度。")
        hint.setStyleSheet("color:#80868b;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("确定")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # 场景里的两个 item
        self._mirror = False
        self._tpl_orig = None        # 未翻转的原始模板
        self._bg_item = QGraphicsPixmapItem()
        self._tpl_item = QGraphicsPixmapItem()
        self._tpl_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)
        self._tpl_item.setOpacity(0.65)
        self.scene.addItem(self._bg_item)
        self.scene.addItem(self._tpl_item)

        self._fill_frames()
        self._on_target()
        self._reload_frame()
        self._apply_scale()

    # ---------------- 填充 / 刷新 ----------------

    def _fill_frames(self):
        frames = (sorted(self._frames_dir.glob("*.png"))
                  if self._frames_dir.is_dir() else [])
        self._frames = frames
        self.cmb_frame.blockSignals(True)
        self.cmb_frame.clear()
        for f in frames:
            self.cmb_frame.addItem(f.name, str(f))
        self.cmb_frame.blockSignals(False)

    def _on_target(self):
        self.cmb_tpl.blockSignals(True)
        self.cmb_tpl.clear()
        self._tpl_meta = {}     # 模板路径 -> (怪/玩家 id, 帧 stem)
        if self.cmb_target.currentData() == "player":
            root = Path("datasets/sprites/player") / self._player_id
            for f in sorted(root.glob("*.png")):
                frame = f.stem
                key = "%s:%s" % (self._player_id, frame)
                sc = self._mob_scales.get(key, self._player_default)
                self.cmb_tpl.addItem("%s · %.2f×" % (frame, sc), str(f))
                self._tpl_meta[str(f)] = (self._player_id, frame)
        else:
            sprite_dir = wzexport.sprite_dir_path()
            for mid in self._mobs:
                d = sprite_dir / mid
                if not d.is_dir():
                    continue
                name = self.name_map.get(mid)
                nm = ("%s (%s)" % (name, mid)) if name else mid
                for f in sorted(d.glob("*.png")):
                    frame = f.stem
                    act = frame.rsplit("_", 1)[0]
                    if act.startswith("die"):
                        continue
                    key = "%s:%s" % (mid, frame)
                    sc = self._mob_scales.get(key, self._mob_default)
                    self.cmb_tpl.addItem("%s · %s · %.2f×" % (nm, frame, sc), str(f))
                    self._tpl_meta[str(f)] = (mid, frame)
        self.cmb_tpl.blockSignals(False)
        self._reload_template()

    def _reload_template(self):
        path = self.cmb_tpl.currentData()
        self._tpl_orig = _load_trimmed_pixmap(path) if path else None
        # 回填当前选中帧的尺度（帧级覆盖 → 默认值）
        meta = self._tpl_meta.get(path) if path else None
        if meta:
            mid, frame = meta
            default = (self._player_default
                       if self.cmb_target.currentData() == "player"
                       else self._mob_default)
            sc = self._mob_scales.get("%s:%s" % (mid, frame), default)
            self._scale = max(0.2, min(4.0, sc))
            self.sld.blockSignals(True)
            self.sld.setValue(int(round(self._scale * 1000)))
            self.sld.blockSignals(False)
            self.sp_scale.blockSignals(True)
            self.sp_scale.setValue(self._scale)
            self.sp_scale.blockSignals(False)
        self._apply_mirror()
        self._apply_scale()

    def _on_mirror(self, checked):
        self._mirror = checked
        self._apply_mirror()

    def _apply_mirror(self):
        """把原始模板按镜像开关翻转（或原样）设到模板 item 上。"""
        pm = self._tpl_orig
        if pm is None or pm.isNull():
            self._tpl_item.setPixmap(QPixmap())
            return
        if self._mirror:
            pm = pm.transformed(QTransform().scale(-1, 1))
        self._tpl_item.setPixmap(pm)
        self._tpl_item.setOffset(-pm.width() / 2, -pm.height() / 2)
        self._center_template()

    def _reload_frame(self):
        path = self.cmb_frame.currentData()
        pm = _load_frame_pixmap(path) if path else None
        if pm is None:
            self._bg_item.setPixmap(QPixmap())
            return
        self._bg_item.setPixmap(pm)
        self.scene.setSceneRect(0, 0, pm.width(), pm.height())
        self._fit_view()
        self._center_template()

    def _center_template(self):
        bg = self._bg_item.pixmap()
        if bg.isNull():
            return
        self._tpl_item.setPos(bg.width() / 2, bg.height() / 2)

    def _fit_view(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _on_slider(self, val):
        self._scale = val / 1000.0
        self.sp_scale.blockSignals(True)
        self.sp_scale.setValue(self._scale)
        self.sp_scale.blockSignals(False)
        self._apply_scale()

    def _on_spin(self, val):
        self._scale = val
        self.sld.blockSignals(True)
        self.sld.setValue(int(round(val * 1000)))
        self.sld.blockSignals(False)
        self._apply_scale()

    def _apply_scale(self):
        self._tpl_item.setScale(self._scale)
        pm = self._tpl_item.pixmap()
        if not pm.isNull():
            w = pm.width() * self._scale
            h = pm.height() * self._scale
            self.lbl_scale.setText("%d×%d px" % (int(w), int(h)))
        else:
            self.lbl_scale.setText("")

    # ---------------- 结果 ----------------

    def scale_value(self):
        return self._scale

    def result_target(self):
        return self.cmb_target.currentData()

    def result_mob(self):
        """返回当前选中模板对应的怪/玩家 id（无则空串）。"""
        path = self.cmb_tpl.currentData()
        meta = self._tpl_meta.get(path) if path else None
        return meta[0] if meta else ""

    def result_frame(self):
        """返回当前选中模板对应的帧 stem（无则空串）。"""
        path = self.cmb_tpl.currentData()
        meta = self._tpl_meta.get(path) if path else None
        return meta[1] if meta else ""
