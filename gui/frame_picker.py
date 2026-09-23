"""选帧弹窗：挑「这一次自动标注要处理哪些帧」。

**从哪来**
    「标注怪物 / 标注玩家」点下去先弹它 —— 脑子里要清楚「哪些已经标过、哪些没有」。
    数据集导出、迭代也能复用（各自的默认筛选不同）。

**为什么需要「处理台账」**
    光看标注文件在不在判断不了：怪物 pass 和玩家 pass 写的是同一个 txt（各覆盖
    自己那一类、保留另一类），零框的帧也会写一个空文件。台账（labels_auto/
    _state.json，见 gui/labelio.py）记的是「哪一类在哪几帧上跑过」，才分得清。

**交互（重点是批量）**
    · 每行一个复选框；列表是 ExtendedSelection：Shift 点/拖选一段、Ctrl 点加选；
    · 按**空格** = 把「当前选中的那些行」整体勾上/取消（Qt 原生不这么做，这里
      自己接快捷键）—— 「Shift 选 200 行 → 空格」就是你要的批量选 / 批量不选；
    · 一排按钮：全选 / 全不选 / 反选 / 只选未处理 / 只选已处理 / 只勾这些；
    · 顶部筛选（只看某一类）只影响**显示**，不动勾选状态；
    · 帧号段输入（如 `0-499,1200`）用于「只勾这些」。

**默认全选**
    「重标」场景的默认就是全部重跑一遍。要只补没跑过的，点一下「只选未处理」——
    配合抽帧的「取消勾选重抽前清空 = 追加素材」，新加的帧正好都在「未处理」里。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QShortcut, QVBoxLayout, QWidget)

from gui import labelio
from gui.widgets import NoWheelComboBox

# 显示用：框数只列「玩家 / 怪物」，其余有才列（和质检台信息行同一套规矩）
_BASE_CLS = 2


class FramePickDialog(QDialog):
    """选帧。`project` 给目录，`target` 是 "mob"/"player"（决定台账与文案）。"""

    FILTERS = (("全部", "all"), ("未勾选", "unchecked"), ("已勾选", "checked"),
               ("未处理", "todo"), ("已处理", "done"))

    def __init__(self, parent, project, target):
        super().__init__(parent)
        self.project = project
        self.target = target
        self._items = []          # [(stem, QListWidgetItem)]

        labels_dir = project.dir_of("labels_auto")
        self.processed = labelio.processed_set(labels_dir, target)
        self.stems = [p.stem for p in sorted(project.frames.glob("*.png"))]

        what = "怪物" if target == "mob" else "玩家"
        self.setWindowTitle("选帧 —— 标注%s" % what)
        root = QVBoxLayout(self)

        head = QLabel("要处理哪些帧？（默认全选；Shift 选一段后按空格可批量勾选 / 取消）")
        head.setStyleSheet("color: #202124;")
        root.addWidget(head)

        # ---- 筛选 + 帧号段 ----
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("只看"))
        self.cmb_filter = NoWheelComboBox()
        for text, key in self.FILTERS:
            self.cmb_filter.addItem(text, key)
        self.cmb_filter.setToolTip("只影响显示，不动已经勾好的状态")
        self.cmb_filter.currentIndexChanged.connect(self._apply_filter)
        bar.addWidget(self.cmb_filter)

        bar.addWidget(QLabel("帧号"))
        self.ed_range = QLineEdit()
        self.ed_range.setPlaceholderText("如 0-499,1200")
        self.ed_range.setToolTip(
            "按帧号（frame_00012.png 的 12）填，支持逗号和区间。\n"
            "点右边按钮 = 「只勾这些」（会清掉别的勾选）")
        self.ed_range.returnPressed.connect(self._check_range_only)
        bar.addWidget(self.ed_range, 1)

        b = QPushButton("只勾这些")
        b.setToolTip("按上面的帧号段勾选，其余全部取消")
        b.clicked.connect(self._check_range_only)
        bar.addWidget(b)
        root.addLayout(bar)

        # ---- 批量按钮 ----
        btns = QHBoxLayout()
        btns.setSpacing(6)

        for text, slot, tip in (
                ("全选", self._check_all, "把所有帧都勾上"),
                ("全不选", self._uncheck_all, "一个都不勾"),
                ("反选", self._invert, "勾选状态反转"),
                ("只选未处理", self._check_todo,
                 "只勾「这一类还没跑过」的帧 —— 补标新素材时最常用"),
                ("只选已处理", self._check_done, "只勾已经跑过的帧（重标一遍）"),
                ("勾选所选", None, "把列表里选中的行勾上（等于按空格）"),
                ("取消勾选所选", None, "把列表里选中的行取消勾选（等于按空格）"),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            if slot is not None:
                b.clicked.connect(slot)
            btns.addWidget(b)
            setattr(self, "_b_%s" % text, b)
        self._b_勾选所选.clicked.connect(lambda: self._toggle_selected(True))
        self._b_取消勾选所选.clicked.connect(lambda: self._toggle_selected(False))
        btns.addStretch(1)
        root.addLayout(btns)

        # ---- 列表 ----
        self.lst = QListWidget()
        self.lst.setSelectionMode(QListWidget.ExtendedSelection)
        self.lst.setUniformItemSizes(True)
        root.addWidget(self.lst, 1)

        # 空格 = 切换「选中行」的勾选（批量操作的主入口）
        for key in (Qt.Key_Space, Qt.Key_Return):
            sc = QShortcut(key, self.lst)
            sc.setContext(Qt.WidgetShortcut)
            sc.activated.connect(lambda: self._toggle_selected(None))
        # Ctrl+A 反而不常用（要全选有按钮），但别让列表的默认行为把勾选带跑
        self.lst.itemChanged.connect(lambda _i: self._update_count())

        foot = QHBoxLayout()
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color: #5f6368;")
        foot.addWidget(self.lbl_count, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("确定")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        foot.addWidget(bb)
        root.addLayout(foot)

        self.resize(720, 560)
        self._populate()

    # ---------------- 列表 ----------------

    def _row_text(self, stem):
        counts, manual = labelio.frame_summary(self.project, stem)
        order = labelio.ORDER
        shown = order[:_BASE_CLS] + [c for c in order[_BASE_CLS:] if counts.get(c)]
        by_cls = " · ".join("%s %d" % (labelio.label(c), counts.get(c, 0))
                            for c in shown)
        marks = []
        marks.append("已处理" if stem in self.processed else "未处理")
        if manual:
            marks.append("人工改过")
        return "%s　%s　%s" % (stem, by_cls, "　".join(marks))

    def _populate(self):
        self.lst.blockSignals(True)
        for stem in self.stems:
            it = QListWidgetItem(self._row_text(stem))
            it.setData(Qt.UserRole, stem)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)          # 默认全选
            self.lst.addItem(it)
            self._items.append((stem, it))
        self.lst.blockSignals(False)
        self._apply_filter()
        self._update_count()

    def _apply_filter(self):
        mode = self.cmb_filter.currentData() or "all"
        for stem, it in self._items:
            if mode == "all":
                it.setHidden(False)
            elif mode == "checked":
                it.setHidden(it.checkState() != Qt.Checked)
            elif mode == "unchecked":
                it.setHidden(it.checkState() == Qt.Checked)
            elif mode == "done":
                it.setHidden(stem not in self.processed)
            else:                                  # todo
                it.setHidden(stem in self.processed)
        self._update_count()

    def _update_count(self):
        n = sum(1 for _s, it in self._items if it.checkState() == Qt.Checked)
        vis = sum(1 for _s, it in self._items if not it.isHidden())
        self.lbl_count.setText("已勾选 %d / 共 %d%s"
                               % (n, len(self._items),
                                  "" if vis == len(self._items)
                                  else "（当前显示 %d）" % vis))

    # ---------------- 批量 ----------------

    def _set_checked(self, want_checked):
        self.lst.blockSignals(True)
        for _stem, it in self._items:
            it.setCheckState(Qt.Checked if want_checked(it) else Qt.Unchecked)
        self.lst.blockSignals(False)
        self._apply_filter()

    def _check_all(self):
        self._set_checked(lambda _it: True)

    def _uncheck_all(self):
        self._set_checked(lambda _it: False)

    def _invert(self):
        self._set_checked(lambda it: it.checkState() != Qt.Checked)

    def _check_todo(self):
        self._set_checked(lambda it: it.data(Qt.UserRole) not in self.processed)

    def _check_done(self):
        self._set_checked(lambda it: it.data(Qt.UserRole) in self.processed)

    def _toggle_selected(self, force=None):
        """空格 / 两个「所选」按钮：切换当前选中行的勾选。

        force=None → 取反（按第一行的状态决定方向，多行状态不一致时以「有没勾上的
        就全勾上」为准，符合直觉）；True/False → 全部设成该状态。
        """
        sel = self.lst.selectedItems()
        if not sel:
            return
        if force is None:
            force = any(it.checkState() != Qt.Checked for it in sel)
        self.lst.blockSignals(True)
        for it in sel:
            it.setCheckState(Qt.Checked if force else Qt.Unchecked)
        self.lst.blockSignals(False)
        self._apply_filter()

    def _parse_range(self):
        """帧号段文本 → 帧号集合。`0-499,1200` 这样写。"""
        out = set()
        for part in (self.ed_range.text() or "").replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, _, b = part.partition("-")
                if a.strip().isdigit() and b.strip().isdigit():
                    lo, hi = int(a), int(b)
                    out.update(range(min(lo, hi), max(lo, hi) + 1))
            elif part.isdigit():
                out.add(int(part))
        return out

    def _check_range_only(self):
        rng = self._parse_range()
        if not rng:
            return
        self.lst.blockSignals(True)
        for stem, it in self._items:
            tail = stem.rsplit("_", 1)[-1]
            it.setCheckState(Qt.Checked if (tail.isdigit() and int(tail) in rng)
                             else Qt.Unchecked)
        self.lst.blockSignals(False)
        self._apply_filter()

    # ---------------- 结果 ----------------

    def selected_stems(self):
        """返回勾选的帧名列表。"""
        return [stem for stem, it in self._items
                if it.checkState() == Qt.Checked]
