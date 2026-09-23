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

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from gui import labelio
from gui.canvas import ImageCanvas
from gui.widgets import NoWheelComboBox, NoWheelSlider


class ReviewPanel(QWidget):
    # 筛选项就这几条，都是**只看产物本身就能判断**的：
    #     空帧 / 多框 / 玩家框数量不对（0 个或 ≥2 个）  ← 看一帧的框数与类别
    #     未经人工修改 / 人工改过                      ← 看 labels/ 里有没有这帧
    #
    # **已移除「有新框出现 / 有旧框消失」**：它们靠相邻帧 IoU 配对来推断误检/漏检，
    # 而目标一直在动，配出来的「新增/消失」很容易是伪影，反而把人引到没问题的帧上。
    FILTERS = (("全部", "all"),
               ("只看空帧（疑似漏检）", "zero"),
               ("只看多框（疑似误检）", "many"),
               ("只看玩家框丢失或重复", "player_count_bad"),
               ("只看未经人工修改", "auto_only"),
               ("只看人工改过", "manual"))

    MANY_THRESHOLD = 8      # 一帧超过这么多框，多半是误检

    def __init__(self, parent=None):
        super().__init__(parent)

        self.project = None
        self.items = []         # [(stem, 框数, 是否人工, 上一帧框数)]，上一帧为 -1 表示无
        self.index = 0
        self._dirty = False
        self._undo = []         # 撤销栈 [(stem, snapshot), ...]，snapshot 是操作前框列表
        self._clipboard = []    # 复制的框 [(cls, x, y, w, h, manual), ...]

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
        self.canvas.before_change.connect(self._on_before_change)
        self.canvas.copy_requested.connect(self._copy)
        self.canvas.paste_requested.connect(self._paste)
        self.canvas.undo_requested.connect(self._undo_edit)
        root.addWidget(self.canvas, 1)

        # ---- 本帧信息行（独占一行）----
        # 「这一帧有什么」是质检时一直在看的东西：帧名 / 各类框数 / 是否人工改过 /
        # 与上一帧比多了少了。以前它和一堆按钮挤在同一行，被 Ignored 策略裁得只剩
        # 半截 —— 所以单独给一行，宽度随便让它占。
        self.lbl_frame = QLabel("—")
        self.lbl_frame.setStyleSheet("color: #202124;")
        self.lbl_frame.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_frame.setMinimumWidth(0)
        root.addWidget(self.lbl_frame)

        # ---- 编辑行：新建框类型 + 编辑按钮 + 最近一次操作的反馈 ----
        ops = QHBoxLayout()
        ops.setSpacing(6)

        ops.addWidget(QLabel("新建框"))
        self.cmb_cls = NoWheelComboBox()
        # 顺序按常用度：怪物最常用（自动标注也产它），玩家次之 —— 而**按住 Ctrl
        # 拖出来的一律是玩家框**（不用来回切下拉框，见下面那行提示）。
        # 「其他玩家」只人工标（自动标注不产这个类，见 perception/classes.py）。
        for _cls in (labelio.CLASS_MOB, labelio.CLASS_PLAYER,
                     labelio.CLASS_OTHER_PLAYER):
            self.cmb_cls.addItem(labelio.label(_cls), _cls)
        self.cmb_cls.currentIndexChanged.connect(self._on_cls_changed)
        ops.addWidget(self.cmb_cls)

        ops.addSpacing(10)
        for text, slot, tip in (
                ("复制", self._copy, "复制选中的框（Ctrl+C）"),
                ("粘贴", self._paste, "粘贴剪贴板里的框（Ctrl+V）"),
                ("撤销", self._undo_edit, "撤销上一次编辑（Ctrl+Z）"),
                ("删选中框", self._delete_selected, "删除选中的框（也可按 Del）"),
                ("恢复自动", self._revert, "丢掉人工修改，回到自动标注的结果"),
                ("剔除整帧", self._drop_frame, "这张图不参与训练 —— 删除它所有相关文件"),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            ops.addWidget(b)

        # 最近一次操作的反馈（「已保存 / 有未保存的修改 / 已恢复…」）跟在按钮右边：
        # 它和「本帧有什么」是两件事，别再混进上面那行里。
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #5f6368;")
        self.lbl_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        ops.addWidget(self.lbl_status, 1)

        root.addLayout(ops)

        # ---- 操作提示：只留「手上要用的」，快捷键细节放按钮 tooltip，别挤成一长条 ----
        hint = QLabel("空白拖=建框（按住 Ctrl 拖 = 玩家框）　框边拖=缩放　"
                      "Del=删框　滚轮=缩放　中键拖=平移　双击=适应窗口")
        hint.setStyleSheet("color: #80868b;")
        hint.setToolTip("复制 / 粘贴 / 撤销：Ctrl+C / Ctrl+V / Ctrl+Z（也可用上面的按钮）\n"
                        "按住 Ctrl 拖空白处 = 按「玩家」类建框，松开 Ctrl 后回到下拉框选的类")
        root.addWidget(hint)

    # ══════════════════════════════════════════════════
    # 数据
    # ══════════════════════════════════════════════════
    def bind(self, project):
        self.project = project
        self.reload()

    def reload(self):
        self._save_current()        # 保住当前帧的改动再重建列表
        self._undo = []             # 帧列表重建后，旧撤销快照失效

        self.items = []
        self.index = 0

        if self.project is None:
            self.canvas.load(QPixmap(), [])
            self.lbl_frame.setText("未选择项目")
            self.lbl_status.setText("")
            self._update_pos()
            return

        mode = self.cmb_filter.currentData()
        total = 0

        for f in sorted(self.project.frames.glob("*.png")):
            stem = f.stem
            total += 1
            n, manual = labelio.count_boxes(self.project, stem)

            if mode == "zero" and n != 0:
                continue
            if mode == "many" and n < self.MANY_THRESHOLD:
                continue
            if mode == "auto_only" and manual:
                continue
            if mode == "manual" and not manual:
                continue
            if mode == "player_count_bad":
                # 一帧**恰好 1 个**玩家框才算正常：
                #     0 个  = 丢失（被遮挡 / 走出视野 / 阈值太高）
                #     ≥2 个 = 重复或误标（同一角色被检出两次，或把别人标成了玩家）
                # 两种都值得看一眼，所以判据是「不等于 1」，不是「等于 0」。
                by_cls = labelio.count_by_class(self.project, stem)
                if by_cls.get(labelio.CLASS_PLAYER, 0) == 1:
                    continue

            self.items.append((stem, n, manual))

        if not self.items:
            self.canvas.load(QPixmap(), [])
            if not total:
                msg = "项目里还没有画面 —— 先跑 ② 采集"
            else:
                msg = "没有符合条件的帧（项目共 %d 帧）" % total
            self.lbl_frame.setText(msg)

        self._update_pos()
        self._load_current()

    def _load_current(self):
        if not self.items or self.project is None:
            self._dirty = False
            return

        stem, n, manual = self.items[self.index]
        path = self.project.frames / (stem + ".png")

        pm = QPixmap(str(path))
        if pm.isNull():
            self.lbl_frame.setText("读不到 %s" % path.name)
            self._dirty = False
            return

        w, h = pm.width(), pm.height()
        boxes = labelio.load_boxes(self.project, stem, w, h)

        self.canvas.load(pm, boxes, editable=True, fit=True)
        self._dirty = False

        # ---- 本帧信息（独占一行，见 _build 的说明）----
        counts = labelio.count_by_class(self.project, stem)
        order = labelio.ORDER
        # 基数只列「玩家 / 怪物」，其余类别**有才列** —— 否则一行五个 0 很吵
        shown = order[:2] + [c for c in order[2:] if counts.get(c)]
        by_cls = " · ".join("%s %d" % (labelio.label(c), counts.get(c, 0))
                            for c in shown)

        self.lbl_frame.setText("%s　%s%s"
                               % (path.name, by_cls,
                                  "　（人工改过）" if manual else ""))
        self.lbl_frame.setToolTip("画面 %d×%d，本帧共 %d 个框" % (w, h, len(boxes)))
        self.lbl_status.setText("")      # 翻帧后清掉上一帧的操作反馈

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

        # 「人工改过」以**落盘为准**：save_boxes 返回人工框数，只有写得下
        # labels/<stem>.txt 才 >0。不能无条件写 True —— 比如「把自动框全删了」
        # 这种编辑不产生人工框，标 True 的话列表说已改、磁盘说没改，
        # 一重载就悄悄变回去，「只看未经人工修改」会忽进忽出。
        self.items[self.index] = (stem, len(boxes), bool(n))
        self._dirty = False
        self._set_status("已保存 %s（%d 个框）" % (stem, n), ok=True)

    # ══════════════════════════════════════════════════
    # 操作
    # ══════════════════════════════════════════════════
    def _on_boxes_changed(self):
        self._dirty = True
        self._set_status("有未保存的修改（翻页会自动保存）")

    def _on_before_change(self):
        """破坏性操作（加框/删框/拖动）前记录撤销快照。"""
        if self.project is None or not self.items:
            return
        stem = self.items[self.index][0]
        self._undo.append((stem, self.canvas.get_boxes()))
        if len(self._undo) > 100:
            del self._undo[0]

    def _copy(self):
        sel = self.canvas.get_selected_boxes()
        if not sel:
            self._set_status("没有选中的框 —— 先点一下框再复制")
            return
        self._clipboard = sel
        self._set_status("已复制 %d 个框" % len(sel), ok=True)

    def _paste(self):
        if not self._clipboard:
            self._set_status("剪贴板为空 —— 先复制框")
            return
        if self.project is None or not self.items:
            return
        self.canvas.before_change.emit()   # 记录撤销快照
        for cls, x, y, w, h, manual in self._clipboard:
            # 偏移 12px，避免和原框完全重叠看不清
            self.canvas.add_box(x + 12, y + 12, w, h, cls, True)
        self.canvas.boxes_changed.emit()
        self._set_status("已粘贴 %d 个框" % len(self._clipboard), ok=True)

    def _undo_edit(self):
        if not self._undo:
            self._set_status("没有可撤销的操作")
            return
        stem, snap = self._undo.pop()
        # 撤销的可能不是当前帧（复制后切到别的帧粘贴），先跳到对应帧
        idx = next((i for i, it in enumerate(self.items) if it[0] == stem), None)
        if idx is not None and idx != self.index:
            self._save_current()
            self.index = idx
            self._update_pos()
            self._load_current()
        self.canvas.replace_boxes(snap)
        self._dirty = True
        self._set_status("已撤销", ok=True)

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

        stem, _n, manual = self.items[self.index]
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
        # 处理台账里也把它划掉：图都没了，选帧弹窗不该再列它（否则勾上会报
        # 「选中的帧找不到」）。台账的键是各标注目标（mob/player），一并清。
        labelio.unmark_processed(self.project.dir_of("labels_auto"), [stem])
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
