"""标注质检台 + 编辑器。

两个身份：
    质检 —— 翻看标注，靠筛选快速定位问题帧
    编辑 —— 补标漏检的怪、删掉误检的框、剔除整帧

**为什么"只看不改"不够**

    自动标注的错误分两类，后果完全不同：
        误检 / 框偏移  → 教模型学错
        漏检           → 那块区域被当背景训练，等于教模型"这里没有怪"

    漏检不是"少学一点"，而是"学到了反面"。而漏检恰恰是模板匹配
    最主要的失败模式（怪被挡住、姿态不符、特效遮盖）。所以人工修正
    的入口是必需的，不是锦上添花。

显示的是 frames/ 里的**原图**，框是活的 —— 不是烧好框的 vis 图，
否则编辑时会和已有框重叠，看不出哪个是自己在拖。
"""

import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from gui import labelio
from gui.canvas import ImageCanvas
from gui.widgets import NoWheelComboBox, NoWheelSlider


class ReviewPanel(QWidget):
    FILTERS = (("全部", "all"),
               ("只看空帧（疑似漏检）", "zero"),
               ("只看多框（疑似误检）", "many"),
               ("有新框出现（疑似误检）", "new"),
               ("有旧框消失（疑似漏检）", "lost"),
               ("只看玩家框丢失", "player_missing"),
               ("只看人工改过", "manual"))

    MANY_THRESHOLD = 8      # 一帧超过这么多框，多半是误检

    def __init__(self, parent=None):
        super().__init__(parent)

        self.project = None
        self.items = []         # [(stem, 框数, 是否人工, 上一帧框数)]，上一帧为 -1 表示无
        self.index = 0
        self._dirty = False

        self._build()

    # ══════════════════════════════════════════════════
    # 界面
    # ══════════════════════════════════════════════════
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- 导航 ----
        bar = QHBoxLayout()
        bar.setSpacing(6)

        for text, delta in (("◀", -1), ("▶", 1)):
            b = QPushButton(text)
            b.setFixedWidth(36)
            b.setStyleSheet("padding: 2px 4px;")
            b.clicked.connect(lambda _, d=delta: self.step(d))
            bar.addWidget(b)
            if delta < 0:
                self.lbl_pos = QLabel("0 / 0")
                self.lbl_pos.setMinimumWidth(96)
                self.lbl_pos.setAlignment(Qt.AlignCenter)
                bar.addWidget(self.lbl_pos)

        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._on_slider)
        bar.addWidget(self.slider, 1)

        bar.addWidget(QLabel("筛选"))
        self.cmb_filter = NoWheelComboBox()
        for text, key in self.FILTERS:
            self.cmb_filter.addItem(text, key)
        self.cmb_filter.currentIndexChanged.connect(self.reload)
        bar.addWidget(self.cmb_filter)

        b = QPushButton("刷新")
        b.clicked.connect(self.reload)
        bar.addWidget(b)

        b = QPushButton("适应窗口")
        b.clicked.connect(lambda: self.canvas.fit())
        bar.addWidget(b)

        root.addLayout(bar)

        # ---- 画布 ----
        self.canvas = ImageCanvas()
        self.canvas.boxes_changed.connect(self._on_boxes_changed)
        root.addWidget(self.canvas, 1)

        # ---- 操作区 ----
        ops = QHBoxLayout()
        ops.setSpacing(6)

        self.lbl_status = QLabel("—")
        self.lbl_status.setStyleSheet("color: #5f6368;")
        # 状态文字切帧时长短会变，若按文字撑宽，会顶到 QSplitter 挤压右边配置区。
        # Ignored 让它不参与宽度计算，文字过长就自然裁剪。
        self.lbl_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        ops.addWidget(self.lbl_status, 1)

        ops.addWidget(QLabel("新建框"))
        self.cmb_cls = NoWheelComboBox()
        self.cmb_cls.addItem("怪物", 1)
        self.cmb_cls.addItem("玩家", 0)
        self.cmb_cls.currentIndexChanged.connect(self._on_cls_changed)
        ops.addWidget(self.cmb_cls)

        hint = QLabel("空白拖=建框　框边拖=缩放　Del=删框　滚轮=缩放　中键拖=平移　双击=适应")
        hint.setStyleSheet("color: #80868b;")
        ops.addWidget(hint)

        for text, slot, tip in (
                ("删选中框", self._delete_selected, "删除选中的框（也可按 Del）"),
                ("恢复自动", self._revert, "丢掉人工修改，回到自动标注的结果"),
                ("剔除整帧", self._drop_frame, "这张图不参与训练 —— 删除它所有相关文件"),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            ops.addWidget(b)

        root.addLayout(ops)

    # ══════════════════════════════════════════════════
    # 数据
    # ══════════════════════════════════════════════════
    def bind(self, project):
        self.project = project
        self.reload()

    def _img_size(self):
        """画面尺寸。相邻帧对比要换算像素坐标，但逐帧解图太慢 ——
        先看 ④ 落盘的 stats.json，没有再退回读第一张图。"""
        if self.project is None:
            return 0, 0

        f = self.project.dir_of("labels_auto") / "stats.json"
        try:
            with open(f, "r", encoding="utf-8") as fh:
                sz = json.load(fh).get("img_size") or []
            if len(sz) == 2 and sz[0] and sz[1]:
                return int(sz[0]), int(sz[1])
        except Exception:
            pass

        first = next(iter(sorted(self.project.frames.glob("*.png"))), None)
        if first is not None:
            pm = QPixmap(str(first))
            if not pm.isNull():
                return pm.width(), pm.height()
        return 0, 0

    def reload(self):
        self._save_current()        # 保住当前帧的改动再重建列表

        self.items = []
        self.index = 0

        if self.project is None:
            self.canvas.load(QPixmap(), [])
            self.lbl_status.setText("未选择项目")
            self._update_pos()
            return

        mode = self.cmb_filter.currentData()
        need_match = mode in ("new", "lost")

        # 画面尺寸所有帧都一样，优先从 stats.json 读，免得逐帧解图
        w, h = self._img_size() if need_match else (0, 0)

        # 相邻帧对比必须基于**完整序列**，不能只比较筛选后的结果 ——
        # 否则"上一帧"会跳到几十帧之前，配出来的对是无意义的。
        prev_boxes = None
        total = 0

        for f in sorted(self.project.frames.glob("*.png")):
            stem = f.stem
            total += 1
            n, manual = labelio.count_boxes(self.project, stem)

            # -1 = 没有上一帧可比（第一帧）
            n_new = n_lost = -1
            if need_match and w and h:
                cur = labelio.load_boxes(self.project, stem, w, h)
                if prev_boxes is not None:
                    new_ids, lost_ids = labelio.match_boxes(prev_boxes, cur)
                    n_new, n_lost = len(new_ids), len(lost_ids)
                prev_boxes = cur

            if mode == "zero" and n != 0:
                continue
            if mode == "many" and n < self.MANY_THRESHOLD:
                continue
            if mode == "manual" and not manual:
                continue
            if mode == "new" and n_new <= 0:
                continue
            if mode == "lost" and n_lost <= 0:
                continue
            if mode == "player_missing":
                by_cls = labelio.count_by_class(self.project, stem)
                if by_cls.get(labelio.CLASS_PLAYER, 0) > 0:
                    continue

            self.items.append((stem, n, manual, n_new, n_lost))

        if not self.items:
            self.canvas.load(QPixmap(), [])
            if not total:
                msg = "项目里还没有画面 —— 先跑 ② 采集"
            elif not need_match:
                msg = "没有符合条件的帧（项目共 %d 帧）" % total
            elif not (w and h):
                msg = "拿不到画面尺寸，无法做相邻帧对比"
            else:
                what = "新出现的框" if mode == "new" else "消失的框"
                msg = ("相邻帧之间没有%s（项目共 %d 帧）—— "
                       "说明标注在时间上是连贯的" % (what, total))
            self.lbl_status.setText(msg)

        self._update_pos()
        self._load_current()

    def _load_current(self):
        if not self.items or self.project is None:
            self._dirty = False
            return

        stem, n, manual, n_new, n_lost = self.items[self.index]
        path = self.project.frames / (stem + ".png")

        pm = QPixmap(str(path))
        if pm.isNull():
            self.lbl_status.setText("读不到 %s" % path.name)
            self._dirty = False
            return

        w, h = pm.width(), pm.height()
        boxes = labelio.load_boxes(self.project, stem, w, h)

        self.canvas.load(pm, boxes, editable=True, fit=True)
        self._dirty = False

        # 与上一帧的配对结果是质检最关键的线索：
        # 新框冒出多半是误检，旧框消失多半是漏检（怪被打 / 被遮挡 / 走到边缘）。
        diff = ""
        if n_new >= 0:
            diff = "   ·   新增 %d / 消失 %d" % (n_new, n_lost)

        self._set_status("%s   ·   %d 个框%s%s   ·   %d×%d"
                         % (stem, len(boxes),
                            "（人工）" if manual else "",
                            diff, w, h))

    def _save_current(self):
        """把当前帧的改动写回 labels/。

        只在真的改过时才写 —— 否则每翻一页都要生成一个文件，
        等于把 labels_auto 全量复制一遍，人工/自动的区分就没意义了。
        """
        if not self._dirty or not self.items or self.project is None:
            return

        stem = self.items[self.index][0]
        w, h = self.canvas.img_size()
        if not w or not h:
            return

        boxes = self.canvas.get_boxes()
        n = labelio.save_boxes(self.project, stem, boxes, w, h)

        # 保留原有的配对结果 —— 它描述的是时间序列上的邻帧，不随人工修改而变
        old = self.items[self.index]
        self.items[self.index] = (stem, len(boxes), True, old[3], old[4])
        self._dirty = False
        self._set_status("已保存 %s（%d 个框）" % (stem, n), ok=True)

    # ══════════════════════════════════════════════════
    # 操作
    # ══════════════════════════════════════════════════
    def _on_boxes_changed(self):
        self._dirty = True
        self._set_status("有未保存的修改（翻页会自动保存）")

    def _on_cls_changed(self):
        self.canvas.current_cls = self.cmb_cls.currentData()
        self._set_status("新建框类别：%s" % self.cmb_cls.currentText())

    def _delete_selected(self):
        n = self.canvas.remove_selected()
        if n:
            self._set_status("删掉 %d 个框" % n, ok=True)
        else:
            self._set_status("没有选中的框 —— 先点一下框")

    def _revert(self):
        if not self.items:
            return

        stem, _n, manual, _new, _lost = self.items[self.index]
        if not manual:
            QMessageBox.information(self, "无需恢复", "这一帧没有人工修改过")
            return

        labelio.revert_frame(self.project, stem)
        self._dirty = False
        self._load_current()
        self._set_status("已恢复为自动标注的结果", ok=True)

    def _drop_frame(self):
        if not self.items:
            return

        stem = self.items[self.index][0]
        r = QMessageBox.question(
            self, "剔除这一帧",
            "将删除 %s 的图片、自动标注、人工标注和可视化图。\n\n"
            "这张图不再参与训练。\n\n"
            "适合处理：被窗口/UI 大面积遮挡、或怪被遮到认不出的帧。" % stem,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if r != QMessageBox.Yes:
            return

        n = labelio.delete_frame(self.project, stem)
        self._dirty = False
        self.items.pop(self.index)
        self.index = max(0, min(self.index, len(self.items) - 1))
        self._update_pos()
        self._load_current()
        self._set_status("已剔除 %s（删除 %d 个文件）" % (stem, n), ok=True)

    # ══════════════════════════════════════════════════
    # 导航
    # ══════════════════════════════════════════════════
    def _update_pos(self):
        n = len(self.items)
        self.lbl_pos.setText("%d / %d" % (self.index + 1 if n else 0, n))

        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, n - 1))
        self.slider.setValue(self.index)
        self.slider.blockSignals(False)

    def step(self, delta):
        if not self.items:
            return
        self._save_current()
        self.index = max(0, min(self.index + delta, len(self.items) - 1))
        self._update_pos()
        self._load_current()

    def _on_slider(self, value):
        if not self.items:
            return
        self._save_current()
        self.index = value
        self._update_pos()
        self._load_current()

    def _set_status(self, text, ok=False):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            "color: %s;" % ("#137333" if ok else "#5f6368"))
