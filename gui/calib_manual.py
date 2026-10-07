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
                             QHBoxLayout, QLabel, QPushButton, QSlider,
                             QVBoxLayout)

from core import petlib
from core import wzexport
from core.imgio import imread   # 支持中文路径，cv2.imread 遇中文会静默失败
from gui import theme           # 弹窗几何按客户端存（config/ui.yaml）✓
from gui.canvas import ZoomPanView   # 和质检台**同一份**看图交互（滚轮/中键/双击）
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox,
                         NoWheelSlider)

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


class CalibManualDialog(QDialog):
    def __init__(self, mobs, player_id, frames_dir, mob_scales=None,
                 mob_default=1.41, player_default=1.41, parent=None,
                 *, drops=None, pets=None, drop_default=1.41, pet_default=1.41):
        """⭐ 后四个走**关键字**（用户 2026-10-05 ✓ 原话："卡片3标定尺度 也要支持标定掉落物、
        宠物类"）—— 老调用方是那 7 个位置参数 ✓ 加在末尾当 kw ⇒ 一处老写法都不用动 ✓。

        `drops` / `pets`：④ 里那一类的清单（`label.drops` / `label.pets` ✓ 元素是
        `{"id"/"pet", "name", "equips"}` 那种 ✓）；**只用来列模板** ✓ —— 结果按"类"存
        （`drop_scale` / `pet_scale` ✓ 见 `CalibCard._save_manual_scale` ✓）。
        """
        super().__init__(parent)
        self.setWindowTitle("手动目测标定尺度")
        self.setMinimumSize(720, 620)

        self._mobs = [str(m) for m in (mobs or [])]
        self._player_id = (player_id or "").strip()
        self._frames_dir = Path(frames_dir)
        self._mob_scales = dict(mob_scales or {})   # {id:帧stem -> scale}
        self._mob_default = float(mob_default or 1.41)
        self._player_default = float(player_default or 1.41)
        #: 掉落物 / 宠物的清单与"起点尺度"（没标过 ⇒ ③ 把总尺度递进来 ✓ 见 `_manual_calib` ✓）
        self._drops = list(drops or [])
        self._pets = list(pets or [])
        self._drop_default = float(drop_default or 1.41)
        self._pet_default = float(pet_default or 1.41)
        self._scale = max(0.2, min(4.0, self._mob_default))

        self.name_map = wzexport.build_mob_name_map()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ---- 目标 / 模板 / 帧 ----
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(QLabel("标定目标"))
        self.cmb_target = NoWheelComboBox()
        self.cmb_target.addItem("怪物", "mob")
        self.cmb_target.addItem("玩家", "player")
        # ⭐ 多两类（用户 2026-10-05 ✓）：它们的模板来自**另外两套图库** ✓（掉落物图标 /
        #   宠物组合外观 ✓ 见 `_on_target` ✓），结果按"类"记一个自己的尺度 ✓。
        self.cmb_target.addItem("掉落物", "drop")
        self.cmb_target.addItem("宠物", "pet")
        self.cmb_target.currentIndexChanged.connect(self._on_target)
        top.addWidget(self.cmb_target)

        top.addWidget(QLabel("模板"))
        self.cmb_tpl = NoWheelComboBox()
        self.cmb_tpl.setMaxVisibleItems(20)
        self.cmb_tpl.currentIndexChanged.connect(self._reload_template)
        top.addWidget(self.cmb_tpl, 1)

        top.addWidget(QLabel("画面帧"))
        self.cmb_frame = NoWheelComboBox()
        self.cmb_frame.setMaxVisibleItems(20)
        self.cmb_frame.currentIndexChanged.connect(self._reload_frame)
        top.addWidget(self.cmb_frame, 1)
        root.addLayout(top)

        # ---- 翻帧：和质检台同一套（◀　位置　拖动滑块　▶）----
        # 为什么不能只有下拉框：对齐模板常要在**连续几帧**里挑目标最清楚的那帧，
        # 滑一下就翻一帧，比"展开下拉再选"快得多（质检台就是这么用的）。
        # 下拉框保留 —— 帧很多时它能直接跳到指定帧，两者是互补的。
        nav = QHBoxLayout()
        nav.setSpacing(6)
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(36)
        self.btn_prev.setStyleSheet("padding: 2px 4px;")
        self.btn_prev.setToolTip("上一帧（也可以直接拖右边的滑块）")
        self.btn_prev.clicked.connect(lambda: self._step_frame(-1))
        nav.addWidget(self.btn_prev)

        self.lbl_pos = QLabel("0 / 0")
        self.lbl_pos.setMinimumWidth(96)
        self.lbl_pos.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.lbl_pos)

        self.sld_frame = NoWheelSlider(Qt.Horizontal)
        self.sld_frame.setMinimum(0)
        self.sld_frame.valueChanged.connect(self._on_frame_slider)
        nav.addWidget(self.sld_frame, 1)

        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(36)
        self.btn_next.setStyleSheet("padding: 2px 4px;")
        self.btn_next.setToolTip("下一帧（也可以直接拖左边的滑块）")
        self.btn_next.clicked.connect(lambda: self._step_frame(1))
        nav.addWidget(self.btn_next)
        root.addLayout(nav)

        # ---- 视图 ----
        # 用**质检台同一个基类**（gui.canvas.ZoomPanView）：滚轮缩放、中键平移、
        # 双击适应窗口 —— 以前这里自己写了一份，于是没有双击适应、缩进步进也不同。
        self.view = ZoomPanView()
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setStyleSheet("background:#202124; border:1px solid #3c4043;")
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        root.addWidget(self.view, 1)

        # ---- 缩放滑块 + 数字输入 ----
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("缩放"))
        self.sld = NoWheelSlider(Qt.Horizontal)
        self.sld.setRange(SLIDER_MIN, SLIDER_MAX)
        self.sld.setValue(int(round(self._scale * 100)))
        self.sld.valueChanged.connect(self._on_slider)
        row.addWidget(self.sld, 1)
        self.sp_scale = NoWheelDoubleSpinBox()
        self.sp_scale.setRange(0.05, 8.0)
        self.sp_scale.setDecimals(3)
        self.sp_scale.setSingleStep(0.01)
        self.sp_scale.setValue(self._scale)
        self.sp_scale.setMinimumWidth(90)
        self.sp_scale.valueChanged.connect(self._on_spin)
        row.addWidget(self.sp_scale)
        # 单位写在框**外面**（UI 规范 §9：不写进编辑框）
        row.addWidget(QLabel("×"))
        self.lbl_scale = QLabel()
        self.lbl_scale.setMinimumWidth(90)
        self.lbl_scale.setStyleSheet("color:#80868b;")
        row.addWidget(self.lbl_scale)
        self.ck_mirror = QCheckBox("镜像")
        self.ck_mirror.setToolTip("水平翻转模板（目标朝向和模板相反时用）")
        self.ck_mirror.toggled.connect(self._on_mirror)
        row.addWidget(self.ck_mirror)
        root.addLayout(row)

        hint = QLabel("拖动模板到目标上，调缩放让轮廓重合，确定即取当前缩放为尺度。"
                      "　滚轮=缩放　中键拖=平移　双击=适应窗口　◀▶/滑块=翻帧")
        hint.setStyleSheet("color:#80868b;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("确定")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        theme.unify_ok_cancel(btns.button(QDialogButtonBox.Ok),
                              btns.button(QDialogButtonBox.Cancel))
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
        theme.bind_window_state(self, "calib_manual")  # 拉过的大小/位置按客户端记住 ✓

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

        # 滑块范围跟着帧数走（帧 0 个时也要能显示 0 / 0，不能停在旧帧数上）
        self.sld_frame.blockSignals(True)
        self.sld_frame.setMaximum(max(0, len(frames) - 1))
        self.sld_frame.setValue(0)
        self.sld_frame.blockSignals(False)
        self._update_pos()

    def _update_pos(self):
        """帧下拉 / 滑块 / 「i / n」标签三者对齐（和质检台一样显示 1-based）。"""
        n = len(getattr(self, "_frames", None) or [])
        i = self.cmb_frame.currentIndex()
        self.lbl_pos.setText("%d / %d" % (i + 1 if (n and i >= 0) else 0, n))

        self.sld_frame.blockSignals(True)
        self.sld_frame.setValue(max(0, i))
        self.sld_frame.blockSignals(False)

    def _step_frame(self, delta):
        """◀ / ▶：翻一帧。到头就停住，不绕回（免得手快翻到不知道哪去了）。"""
        n = self.cmb_frame.count()
        if not n:
            return
        i = max(0, min(self.cmb_frame.currentIndex() + delta, n - 1))
        if i != self.cmb_frame.currentIndex():
            self.cmb_frame.setCurrentIndex(i)       # 信号 → _reload_frame
        else:
            self._update_pos()

    def _on_frame_slider(self, val):
        """拖动滑块翻帧（质检台里就是拖这条）。"""
        if (0 <= val < self.cmb_frame.count()
                and val != self.cmb_frame.currentIndex()):
            self.cmb_frame.setCurrentIndex(val)     # 信号 → _reload_frame
        else:
            self._update_pos()

    def _on_target(self):
        self.cmb_tpl.blockSignals(True)
        self.cmb_tpl.clear()
        self._tpl_meta = {}     # 模板路径 -> (怪/玩家/掉落物/宠物 id, 帧 stem)
        _tgt = self.cmb_target.currentData()
        if _tgt in ("drop", "pet"):
            # ⭐⭐ 掉落物 / 宠物（用户 2026-10-05 ✓）—— 模板来自**另外两套图库**：
            #   · 掉落物：`datasets/sprites/drop/<id>/*.png`（`wzexport.drop_frame_files` ✓）；
            #   · 宠物：**现算的组合外观**（宠物 + 它戴的装备 ✓ `petlib.compose_pet` ✓ ——
            #     ④ 标的时候用的就是这一份 ✓，标定当然也得拿同一份 ✓ 不然量的是别的图 ✗）。
            #   ⚠ 这两类都是**按类记一个尺度** ✓（没有"逐帧标定"那套 ✓ 见
            #     `CalibCard._save_manual_scale` ✓）⇒ 这里列帧只是让你挑一张**最清楚的**来对齐 ✓。
            self._fill_class_templates(_tgt)
            self.cmb_tpl.blockSignals(False)
            self._reload_template()
            return
        if _tgt == "player":
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

    def _fill_class_templates(self, kind):
        """把「掉落物 / 宠物」这一类**能拿来对齐的模板**填进模板下拉 ⇒ 顺便记好 meta ✓。

        ⚠ 一个都没有时**说清为什么**（不静默留个空下拉 ✗ —— "选不到东西"和"弹窗坏了"
        得分得清 ✓）：占位项写在模板下拉里 ✓（它的 data 为空 ⇒ 画布上什么都不叠 ✓）。
        ⚠ 这两类是**按类记一个尺度** ✓ ⇒ 列帧只是让你挑一张**最清楚的**来对齐 ✓
          （`_tpl_meta` 里的 id 只用于显示 ✓ 写回时不看它 ✓ 见 `CalibCard._save_manual_scale` ✓）。
        """
        if kind == "drop":
            for d in self._drops:
                did = str(d.get("id") or "").strip()
                if not did:
                    continue
                nm = str(d.get("name") or did)
                files = wzexport.drop_frame_files(did)
                if not files:
                    continue
                for f in files:
                    self.cmb_tpl.addItem("%s (%s) · %s · %.2f×"
                                         % (nm, did, f.stem, self._drop_default), str(f))
                    self._tpl_meta[str(f)] = (did, f.stem)
            if self.cmb_tpl.count() == 0:
                self.cmb_tpl.addItem("（掉落物图库还是空的：先用 WzProbe dump-drops 导出，"
                                     "并在 ④ 里勾上「标注掉落物」）", "")
            return
        for e in self._pets:
            pid = str(e.get("pet") or "").strip()
            if not pid:
                continue
            nm = str(e.get("name") or pid)
            try:
                # 宠物模板**现算组合外观**（宠物 + 它戴的装备 ✓）—— ④ 标的时候用的就是这一份 ✓
                root = petlib.compose_pet(pid, e.get("equips") or ())
            except Exception:                    # noqa: BLE001 —— 组合做不出来 ⇒ 如实跳过 ✓
                root = None
            if root is None:
                continue
            for f in sorted(Path(root).glob("*.png")):
                if f.stem == "icon":
                    continue
                self.cmb_tpl.addItem("%s (%s) · %s · %.2f×"
                                     % (nm, pid, f.stem, self._pet_default), str(f))
                self._tpl_meta[str(f)] = (pid, f.stem)
        if self.cmb_tpl.count() == 0:
            self.cmb_tpl.addItem("（这一只宠物的组合外观还没做出来：先在 ④ 里勾上「标注宠物」，"
                                 "并确认宠物图库已用 dump-pets / dump-petequips 导出）", "")

    def _reload_template(self):
        path = self.cmb_tpl.currentData()
        self._tpl_orig = _load_trimmed_pixmap(path) if path else None
        # 回填当前选中帧的尺度（帧级覆盖 → 默认值）
        meta = self._tpl_meta.get(path) if path else None
        if meta:
            mid, frame = meta
            _t = self.cmb_target.currentData()
            if _t == "player":
                default = self._player_default
            elif _t == "drop":
                default = self._drop_default
            elif _t == "pet":
                default = self._pet_default
            else:
                default = self._mob_default
            # ⚠ 帧级覆盖（`mob_scales`）**只对 怪 / 玩家** 有意义 ✓ —— 掉落物 / 宠物按类存 ✓
            #   （硬查那张表只会查到别人的账 ✓ 见 `CalibCard._save_manual_scale` ✓）
            if _t in ("drop", "pet"):
                sc = default
            else:
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
        # 先对齐滑块/标签：这个函数可能是滑块拖动触发的，不能让它们自己不同步
        self._update_pos()
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
        # 走 ZoomPanView.fit()：和质检台的「适应窗口」是同一份实现
        # （双击画面也是调它，两边行为必然一致）
        self.view.fit()

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
