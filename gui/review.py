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

import re                     # ⭐ 批量剔除弹窗的"帧号段"要按数字匹配 ✓（见 `_stem_in_ranges` ✓）

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                             QShortcut, QSizePolicy, QVBoxLayout, QWidget)

from gui import labelio, theme
from gui.canvas import MIN_SIZE, MODE_LABEL, MODE_REGION, MODE_SELECT, ImageCanvas
from perception.world import region as region_mod        # ⭐ 框选区域一处口径 ✓（见 `region.py` ✓）
from gui.widgets import NoWheelComboBox, NoWheelSlider


class BatchDropDialog(QDialog):
    """「批量剔除」选帧弹窗（用户 2026-10-16 ✓ 原话："**给质检台加一个批量选帧删除的弹窗功能
    （类似自动标注的选帧）**"✓）。

    ⭐ 交互**照现成那套**（`yolo_workbench._SelFramesDialog` ✓ 它本身又仿数据集工作台的
      `FramePickDialog` ✓）—— 会用那个的人自然会用这个 ✓：
      · 每行一个复选框；Shift / Ctrl 选一段；**空格**把所选行批量勾上 / 取消 ✓；
      · 全选 / 全不选 / 反选 / 勾选所选 / 取消勾选所选 ✓
        （⚠ 四个按钮都只作用于**当前显示**的帧 ✓ —— 屏幕上看得到几行就管几行 ✓）；
      · 「只看」筛选：**只动显示、不动勾选** ✓（勾了再切筛选，勾还在 ✓）；
      · 「帧号」段「只勾这些」：把名字里的数字段落进区间 ✓（`0-499,1200` ✓）。
    ⚠⚠ **默认一个都不勾** ✗（剔除是**破坏性**的 ⇒ 绝不能预设"全勾"✗）；
    ⚠ 本弹窗**不动任何文件** ✗：它只负责"挑出哪些帧" ✓，真删在调用方
      （`ReviewPanel._drop_frames_batch` ✓ 且删前还有**二次确认** ✓）。
    """

    FILTERS = (("全部", "all"),
               ("只看未勾选", "unchecked"),
               ("只看已勾选", "checked"),
               ("只看空帧（疑似漏检）", "zero"),
               ("只看多框（疑似误检）", "many"),
               ("只看人工改过", "manual"))

    def __init__(self, parent, items):
        super().__init__(parent)
        self.setWindowTitle("批量剔除 —— 勾选要删掉的帧")
        self._rows = [(str(it[0]), int(it[1]), bool(it[2])) for it in items]
        self._checked = set()                 #: 勾选状态**自己拿着** ✓ ⇒ 切筛选不丢 ✓

        root = QVBoxLayout(self)
        head = QLabel(
            "勾选**要剔除**的帧（默认一个都不勾 ✓；Shift 选一段后按空格可批量勾 / 取消）\n"
            "⚠ 剔除 = 删除这一帧的 **图片 + 自动标注 + 人工标注 + 可视化图**，"
            "不可撤销（点「剔除」后还会再问一次 ✓）")
        head.setWordWrap(True)
        root.addWidget(head)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("只看"))
        self.cmb_filter = NoWheelComboBox()
        for text, key in self.FILTERS:
            self.cmb_filter.addItem(text, key)
        self.cmb_filter.currentIndexChanged.connect(self._populate)
        bar.addWidget(self.cmb_filter)
        bar.addWidget(QLabel("帧号"))
        self.ed_range = QLineEdit()
        self.ed_range.setPlaceholderText("如 0-499,1200（按名字里的数字段 ✓）")
        self.ed_range.returnPressed.connect(self._check_range_only)
        bar.addWidget(self.ed_range, 1)
        b = QPushButton("只勾这些")
        b.setToolTip("按上面的帧号段勾选，其余全部取消")
        b.clicked.connect(self._check_range_only)
        bar.addWidget(b)
        root.addLayout(bar)

        btns = QHBoxLayout()
        btns.setSpacing(6)
        for text, slot in (("全选", self._check_all), ("全不选", self._uncheck_all),
                           ("反选", self._invert)):
            btns.addWidget(QPushButton(text, clicked=slot))
        for text, force in (("勾选所选", True), ("取消勾选所选", False)):
            btns.addWidget(QPushButton(text, clicked=lambda _=False, f=force:
                                       self._toggle_selected(f)))
        btns.addStretch(1)
        root.addLayout(btns)

        self.lst = QListWidget()
        self.lst.setSelectionMode(QListWidget.ExtendedSelection)
        self.lst.setUniformItemSizes(True)
        self.lst.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.lst, 1)
        for key in (Qt.Key_Space, Qt.Key_Return):
            sc = QShortcut(key, self.lst)
            sc.setContext(Qt.WidgetShortcut)
            sc.activated.connect(lambda: self._toggle_selected(None))

        foot = QHBoxLayout()
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color: #5f6368;")
        foot.addWidget(self.lbl_count, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("剔除选中的帧")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        theme.unify_ok_cancel(bb.button(QDialogButtonBox.Ok),
                              bb.button(QDialogButtonBox.Cancel))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        foot.addWidget(bb)
        root.addLayout(foot)

        self.resize(680, 560)
        self._populate()
        theme.bind_window_state(self, "review_batch_drop")

    # ---------------- 列表 ----------------
    def _match(self, row):
        """该行落不落进当前「只看」筛选（**只影响显示** ✓）。"""
        key = self.cmb_filter.currentData() or "all"
        stem, boxes, manual = row
        if key == "checked":
            return stem in self._checked
        if key == "unchecked":
            return stem not in self._checked
        if key == "zero":
            return boxes == 0
        if key == "many":
            return boxes >= 2
        if key == "manual":
            return manual
        return True

    def _populate(self, *_a):
        self.lst.blockSignals(True)
        self.lst.clear()
        shown = 0
        for row in self._rows:
            if not self._match(row):
                continue
            shown += 1
            stem, boxes, manual = row
            it = QListWidgetItem("%s   %s%d 框%s"
                                 % (stem, "" if boxes else "⚠ ", boxes,
                                    "（人工改过）" if manual else ""))
            it.setData(Qt.UserRole, stem)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if stem in self._checked else Qt.Unchecked)
            self.lst.addItem(it)
        self.lst.blockSignals(False)
        self._refresh_count(shown)

    def _on_item_changed(self, it):
        stem = it.data(Qt.UserRole)
        if it.checkState() == Qt.Checked:
            self._checked.add(stem)
        else:
            self._checked.discard(stem)
        self._refresh_count()

    def _refresh_count(self, shown=None):
        if shown is None:
            shown = self.lst.count()
        self.lbl_count.setText("已勾 **%d** 帧 / 共 %d 帧（当前显示 %d 帧）"
                               % (len(self._checked), len(self._rows), shown))

    # ---------------- 批量勾选（都只作用于**当前显示** ✓）----------------
    def _visible_rows(self):
        return [self.lst.item(i).data(Qt.UserRole) for i in range(self.lst.count())]

    def _set_visible(self, want):
        for stem in self._visible_rows():
            if want:
                self._checked.add(stem)
            else:
                self._checked.discard(stem)
        self._populate()

    def _check_all(self):
        self._set_visible(True)

    def _uncheck_all(self):
        self._set_visible(False)

    def _invert(self):
        for stem in self._visible_rows():
            if stem in self._checked:
                self._checked.discard(stem)
            else:
                self._checked.add(stem)
        self._populate()

    def _toggle_selected(self, force=None):
        rows = [it.data(Qt.UserRole) for it in self.lst.selectedItems()]
        if not rows:
            return
        for stem in rows:
            want = (stem not in self._checked) if force is None else bool(force)
            if want:
                self._checked.add(stem)
            else:
                self._checked.discard(stem)
        self._populate()

    # ---------------- 帧号段 ----------------
    @staticmethod
    def _parse_ranges(text):
        """`"0-499,1200"` ⇒ `[(0, 499), (1200, 1200)]` ✓（空 / 看不懂 ⇒ 空表 ✓）。"""
        out = []
        for seg in str(text or "").replace("，", ",").split(","):
            seg = seg.strip()
            if not seg:
                continue
            m = re.match(r"^(\d+)\s*-\s*(\d+)$", seg)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                out.append((min(a, b), max(a, b)))
                continue
            if seg.isdigit():
                out.append((int(seg), int(seg)))
        return out

    @classmethod
    def _stem_in_ranges(cls, stem, rngs):
        """名字里**任意**一段数字落进区间就算中 ✓。

        ⚠ 为什么不是"只看末尾数字" ✗：这套帧名形如
          `cycle_00001_0000001_000.012605s` ⇒ 帧号在**中间** ✓、末尾是 `s` ✗
          ⇒ 只要"任意段命中"就能同时兼容两种命名 ✓（用户不必记格式 ✓）。
        """
        if not rngs:
            return False
        for num in re.findall(r"\d+", stem):
            v = int(num)
            if any(a <= v <= b for a, b in rngs):
                return True
        return False

    def _check_range_only(self):
        rngs = self._parse_ranges(self.ed_range.text())
        if not rngs:
            QMessageBox.information(self, "帧号段",
                                    "没看懂这段帧号 —— 写法形如 `0-499,1200` ✓")
            return
        self._checked = {stem for stem, _b, _m in self._rows
                         if self._stem_in_ranges(stem, rngs)}
        self._populate()

    # ---------------- 结果 ----------------
    def selected(self):
        """勾中的帧（按**列表原顺序** ✓ —— 删起来的日志好读 ✓）。"""
        return [stem for stem, _b, _m in self._rows if stem in self._checked]


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
               # ⭐ 「只看有掉落物」（用户 2026-10-04 ✓ 第 2 条）：掉落物是**人工标的**
               #   （自动标注只产 玩家/怪物 ✓ 见 `tools/label_drops`），所以这条实际上是
               #   "**挑出标过掉落物的帧**" —— 复核/补漏掉落物标注时用得上 ✓。
               ("只看有掉落物", "has_drop"),
               # ⭐ 「只看有宠物」（用户 2026-10-05 ✓）：宠物是**干扰类**（class 5 ✓ 见
               #   `perception/detect/classes.py` ✓）—— 自动标注**会**产它（`tools/label_pets` ✓ 模板匹配
               #   或 YOLO 补框 ✓），所以这条既是"挑出标过宠物的帧"、也是**复核宠物框**的入口 ✓
               #   （宠物最烦的错误是"把宠物误检成怪"⇒ 逐帧看一遍那几帧就够了 ✓）。
               ("只看有宠物", "has_pet"),
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

        # 「聚焦质检台」= 焦点在**本面板或其子控件**上（见 `_build` 里那对方向键快捷键）。
        # QWidget 默认是 NoFocus ⇒ 不设这个的话，**点空白处**焦点还留在别的页签上，
        # 人在面板里按 ←→ 却是别处在响应 ✗（画布本身有焦点 ✓，但空手点一下就丢了）。
        self.setFocusPolicy(Qt.StrongFocus)

        self._build()

    # ══════════════════════════════════════════════════
    # 界面
    # ══════════════════════════════════════════════════
    def _build(self):
        # ⚠ **有意不放滚动区**（§4「有意为之」标注）：本页主体 = 画布（QGraphicsView，
        #    **不许再套外层滚动区** ✗ 见 §4）；导航 / 本帧信息 / 编辑行已按「一件事一行」
        #    分好（见下面各行注释），窗口变窄时信息行与反馈走 Ignored 策略缩 ✓。
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
            # 快捷键写在按钮 tooltip 上（和这一页其它按钮一个规矩 ✓）：不写出来没人知道
            b.setToolTip("上一帧（←）" if delta < 0 else "下一帧（→）")
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

        # ⭐⭐ **操作模式**（用户 2026-10-04 ✓ 第 3 条：原来这儿是「适应窗口」按钮 ⇒ 换成它 ✓）。
        #   · **标注模式**（默认 ✓ 快捷键 Q）：空白处拖 = 拉新框（老手感 ✓）；
        #   · **选择模式**（快捷键 W）：空白处拖 = **黑色虚线**框选 ⇒ 松手**选中被框住的框**，
        #     **不建框** ✓ ⇒ 点「删选中框」或按 Del 一次删掉一批（清理误检 ✓）。
        #   ⚠ 「适应窗口」没消失：**双击画面**就是它 ✓（画布老规矩 ✓ 见 `ZoomPanView`），
        #     提示行里也写着 ✓ —— 位置让给了更常用的模式开关 ✓。
        bar.addWidget(QLabel("操作"))
        self.cmb_mode = NoWheelComboBox()
        self.cmb_mode.addItem("标注模式", MODE_LABEL)
        self.cmb_mode.addItem("选择模式", MODE_SELECT)
        # ⭐⭐ **区域自动标注**（用户 2026-10-11 ✓ 原话："给质检台新增「区域自动标注」操作模式，
        #   在该模式，框选一个区域后出现弹窗…点击各个标注按钮，**只会在当前帧框选的区域内**
        #   进行自动标注"✓）—— 拉一个**青色实线**框（跟选择模式的黑虚线一眼分得开 ✓），
        #   松手弹窗 ⇒ 里面照「自动标注」卡片那一套选参数、点按钮 ✓。
        self.cmb_mode.addItem("区域自动标注", MODE_REGION)
        self.cmb_mode.setToolTip(
            "**标注模式**（默认 / 按 Q）：空白处拖 = **拉一个新框** ✓\n"
            "**选择模式**（按 W）：空白处拖 = **黑色虚线**框选 ⇒ 松手**选中被框住的框** ✓\n"
            "　　**不会建框** ✓ ⇒ 按 Del 或点「删选中框」一次删掉一批（清误检用 ✓）；\n"
            "　　不按 Ctrl = 只选这一批，按住 Ctrl = **加选** ✓。\n"
            "**区域自动标注**：空白处拖 = **青色实线**框一块 ⇒ **松手弹窗** ✓\n"
            "　　弹窗里勾「标注玩家 / 标注怪物 / 标准掉落物 / 标注宠物」⇒ 点对应按钮 ⇒\n"
            "　　**只在这一块里、只对当前帧**跑自动标注 ✓（区域外的目标一个都不标 ✓）。\n\n"
            "三种模式下都一样：框上拖 = 移动 / 框边拖 = 缩放 / 滚轮 = 缩放 /\n"
            "中键拖 = 平移 / **双击 = 适应窗口** / ←→ = 上一帧下一帧 ✓")
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        bar.addWidget(self.cmb_mode)

        root.addLayout(bar)

        # ---- 键盘：← / → 切帧（2026-09-27 用户要求："聚焦质检台时，希望能按←→切换帧"）----
        # 为什么是 `QShortcut` + `WidgetWithChildrenShortcut`，而不是 `keyPressEvent`：
        #   这一页是**多控件**面板（画布 / 筛选下拉 / 滑块 / 按钮），焦点几乎从来不在面板
        #   本身 ⇒ `keyPressEvent` 基本收不到 ✗（症状正是"有时候能切、有时候不能"）。
        #   而这个 context 的语义恰好是"焦点在本面板**或其任何子控件**上都算聚焦质检台" ✓，
        #   它**不会溢到别的页签**（主窗口还挂着实时 / 路线识别那些页 ✓）—— 这正是用户
        #   说的"**聚焦质检台时**" ✓（`WindowShortcut` 就是错的：在别的页签上也会抢 ←→ ✗）。
        # ⚠ 交互取舍：焦点在**滑块**上时 ←→ 也归切帧 —— 滑块本来就用 ←→ 挪一格，而那一格
        #   正好也是一帧 ⇒ 行为一致 ✓；焦点在**筛选下拉**上时同理（它还有上下键和鼠标 ✓）。
        #   质检是"一帧一帧过"的活，切帧是主意图 ✓。
        # ⚠ 按住不放会**自动重复**（`QShortcut` 默认行为）= 连切 ✓。
        for key, delta in ((Qt.Key_Left, -1), (Qt.Key_Right, 1)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda d=delta: self.step(d))

        # ---- 画布 ----
        self.canvas = ImageCanvas()
        self.canvas.boxes_changed.connect(self._on_boxes_changed)
        self.canvas.before_change.connect(self._on_before_change)
        self.canvas.copy_requested.connect(self._copy)
        self.canvas.paste_requested.connect(self._paste)
        self.canvas.undo_requested.connect(self._undo_edit)
        # ⭐ 框选/点框选中了几个，写一句给用户看（「选中 3 个框 ⇒ 按 Del 一起删」✓）
        self.canvas.selection_changed.connect(self._on_selection_changed)
        # ⭐⭐ **区域自动标注**：框完松手 ⇒ 弹窗 ✓（用户 2026-10-11 ✓ 见 `cmb_mode` 那条 ✓）
        self.canvas.region_picked.connect(self._on_region_picked)
        root.addWidget(self.canvas, 1)

        # ⭐ 操作模式快捷键（用户 2026-10-04 ✓ 第 3 条）：**Q = 标注模式、W = 选择模式** ✓
        #   ⚠ 与 ←/→ 用**同一套** context（`WidgetWithChildrenShortcut` ✓ 见上面那段说明）：
        #     只在质检台里生效，不去别的页签抢 Q/W 那两个字母键 ✗。
        for key, mode in ((Qt.Key_Q, MODE_LABEL), (Qt.Key_W, MODE_SELECT)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda m=mode: self.set_mode(m))

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
        # 「其他玩家」只人工标（自动标注不产这个类，见 perception/detect/classes.py）。
        # ⭐ **掉落物**（用户 2026-10-04 ✓ 第 1 条："质检台现在需要能手动标注掉落物类"）——
        #   排第二：自动标注产的是 玩家/怪物（见 `tools/label_drops` ✓），而**掉落物全靠人工补**
        #   ⇒ 它和怪物一样常用 ✓（`CLASS_DROP = 2` 见 `perception/detect/classes.py` ✓ 颜色/名字都在那儿 ✓）。
        for _cls in (labelio.CLASS_MOB, labelio.CLASS_DROP, labelio.CLASS_PLAYER,
                     labelio.CLASS_OTHER_PLAYER, labelio.CLASS_PET):
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
                # ⭐⭐ **批量剔除…**（用户 2026-10-16 ✓ 原话："**给质检台加一个批量选帧删除的
                #   弹窗功能（类似自动标注的选帧）**"✓）：
                #   · 为什么要它 —— 坏帧（被窗口/UI 大面积遮挡、怪被遮到认不出）通常**成段出现** ✗
                #     ⇒ 一帧一帧点「剔除整帧」太慢 ✓；
                #   · 交互照现成那套（`yolo_workbench._SelFramesDialog` ✓ 它本身又仿
                #     数据集工作台的 `FramePickDialog` ✓）：每行复选框 / Shift 选一段 /
                #     空格批量切换 / 全选·全不选·反选 / 只看筛选 / 帧号段"只勾这些" ✓；
                #   · 语义**沿用本页的单帧剔除** ✓（删的是同一组文件 ✓ 见 `labelio.delete_frame` ✓）
                #     —— 不新造第二种"删除" ✗（那种迟早两边对不上 ✓）。
                ("批量剔除…", self._drop_frames_batch,
                 "一次剔掉多帧（弹窗里勾选 / 借筛选批量勾）—— 删除所选帧的全部相关文件"),
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
        hint = QLabel("空白拖=建框（按住 Ctrl 拖 = 玩家框）　Q/W=标注/选择模式　框边拖=缩放　"
                      "Del=删框　滚轮=缩放　中键拖=平移　双击=适应窗口　"
                      "←/→=上一帧/下一帧")
        hint.setStyleSheet("color: #80868b;")
        hint.setToolTip("复制 / 粘贴 / 撤销：Ctrl+C / Ctrl+V / Ctrl+Z（也可用上面的按钮）\n"
                        "按住 Ctrl 拖空白处 = 按「玩家」类建框，松开 Ctrl 后回到下拉框选的类\n"
                        "Q = 标注模式（空白拖 = 拉新框）；W = 选择模式（空白拖 = 黑虚线框选，"
                        "选中被框住的框 ⇒ Del 批量删 ✓）\n"
                        "← / →：上一帧 / 下一帧（焦点在这个面板里就行，按着不放会连切 ✓）")
        root.addWidget(hint)

        # ⚠ 这一句要放在**最后**：`_on_mode_changed` 会写 `lbl_status`（上面那行才建出来 ✗）
        #   —— 放早了就是 `AttributeError: 'ReviewPanel' object has no attribute 'lbl_status'`
        #   （本轮踩过 ✓ 离屏探针当场抓到 ✓）。
        self._on_mode_changed()      # 按下拉框的默认值同步一次（默认 = 标注模式 ✓ 光标也对上 ✓）

    # ══════════════════════════════════════════════════
    # 数据
    # ══════════════════════════════════════════════════
    def bind(self, project):
        self.project = project
        self.reload()

    def reload(self):
        self._save_current()        # 保住当前帧的改动再重建列表
        self._undo = []             # 帧列表重建后，旧撤销快照失效

        # ⭐⭐ **先记住"我停在哪一帧"**（用户 2026-10-16 ✓ 第 2 条："质检台**刷新后**，需要
        #   **定位到上次停留的那一帧**（若不存在则是**最近的向上相邻帧**）"✓）：
        #     ① 当前帧名 ✓；② **刷新前那份顺序** ✓ —— 原帧要是没了，就靠它回溯"向上相邻" ✓
        #     （⚠ 不能只记下标 ✗：刷完列表长度/内容都可能变，下标对不上 ✓）。
        _prev_stem = self.items[self.index][0] if 0 <= self.index < len(self.items) else ""
        _prev_order = [it[0] for it in self.items]

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
            if mode == "has_drop":
                # ⭐ 「只看有掉落物」（用户 2026-10-04 ✓ 第 2 条）—— 一个掉落框都没有就跳过 ✓。
                #   ⚠ 取的是**合并后**的框（labels_auto + labels ✓ 见 `labelio.count_by_class`）：
                #     将来自动标注也产掉落物时，这条筛选不用改 ✓。
                if not labelio.count_by_class(self.project, stem).get(labelio.CLASS_DROP, 0):
                    continue
            if mode == "has_pet":
                # ⭐ 「只看有宠物」（用户 2026-10-05 ✓）—— 与上一条**同款** ✓：
                #   一个宠物框都没有就跳过 ✓；同样取**合并后**的框（`labels_auto` + `labels` ✓
                #   见 `labelio.count_by_class` ✓）⇒ 人工补的宠物框也算 ✓。
                #   ⚠ 为什么值得有这么一条：宠物是**干扰类**（class 5 ✓）—— 它最烦的错误是
                #     "宠物被误检成怪 / 怪被误检成宠物" ✓ ⇒ 有这一条就能把"标过宠物的那几帧"
                #     一次挑出来逐帧核对 ✓（不用在几百帧里翻 ✓）。
                if not labelio.count_by_class(self.project, stem).get(labelio.CLASS_PET, 0):
                    continue

            self.items.append((stem, n, manual))

        if not self.items:
            self.canvas.load(QPixmap(), [])
            if not total:
                msg = "项目里还没有画面 —— 先跑 ② 采集"
            else:
                msg = "没有符合条件的帧（项目共 %d 帧）" % total
            self.lbl_frame.setText(msg)

        # ⭐ **重建完 ⇒ 定位回原处**（用户 2026-10-16 ✓ 第 2 条 ✓）——⚠ 必须放在
        #   `_update_pos` / `_load_current` **之前** ✗（那两位读的就是 `self.index` ✓
        #   放后面等于白定位一遍、还得再刷一次 ✓）。
        self._restore_pos(_prev_stem, _prev_order)
        self._update_pos()
        self._load_current()

    def _restore_pos(self, prev_stem, prev_order):
        """刷新后**定位到上次那一帧**；它不在了 ⇒ **最近的向上相邻帧** ✓（用户 2026-10-16 ✓）。

        · 原帧还在 ⇒ 直接选它 ✓；
        · 原帧没了（被剔掉 / 换了筛选 ✓）⇒ 顺着**刷新前那份顺序**往回找（`prev_order` ✓）
          第一个"现在还活着"的帧 ✓ —— 那就是**向上相邻** ✓；
        · 往回一个都没剩（原先是第一帧 ✓）⇒ 落到列表开头 ✓（这是唯一合理的落点 ✓）。
        """
        stems = [it[0] for it in self.items]
        if not stems:
            return
        if prev_stem and prev_stem in stems:
            self.index = stems.index(prev_stem)
            return
        try:
            cut = prev_order.index(prev_stem)
        except ValueError:                     # 连旧顺序里都没有（第一次刷 / 换了项目 ✓）
            cut = len(prev_order)
        for s in reversed(prev_order[:cut]):
            if s in stems:
                self.index = stems.index(s)
                return
        self.index = 0

    def _load_current(self):
        if not self.items or self.project is None:
            self._dirty = False
            return

        # ⭐ **区域自动标注弹窗跟着当前帧走**（用户 2026-10-16 ✓ 第 1 条："弹窗打开时需要能
        #   交互主窗口"✓）—— 交互的前提是"**要标哪一帧**"始终等于**你正看着的这一帧** ✓：
        #   每载入一帧就把新帧名推给它 ✓（`set_target` ✓ 弹窗那边标题上常显 ✓）。
        #   ⚠ 不做这一步而只是把 `exec_` 换成 `show` ✗ ⇒ 人换了帧再按按钮会**以为标的是新帧** ✗
        #     —— 那正是原来把它做成模态的理由 ✓ 见 `_on_region_picked` 那段 ✓。
        _rd = getattr(self, "_region_dlg", None)
        if _rd is not None and _rd.isVisible():
            try:
                _rd.set_target(self.items[self.index][0])
            except Exception:                    # noqa: BLE001 —— 推不动不该炸主回路 ✓
                pass

        stem, n, manual = self.items[self.index]
        path = self.project.frames / (stem + ".png")

        pm = QPixmap(str(path))
        if pm.isNull():
            self.lbl_frame.setText("读不到 %s" % path.name)
            self._dirty = False
            return

        w, h = pm.width(), pm.height()
        boxes = labelio.load_boxes(self.project, stem, w, h)

        # ⭐ **蒙版透明度**（用户 2026-09-29 ✓）：每次载帧都从界面偏好读 ⇒
        #   设置里改完、下一帧就生效（当前帧由设置弹窗的信号即时推 ✓）
        self.canvas.set_mask_alpha(theme.review_mask_alpha())
        self.canvas.load(pm, boxes, editable=True, fit=True)
        self._dirty = False

        # ---- 本帧信息（独占一行，见 _build 的说明）----
        counts = labelio.count_by_class(self.project, stem)
        order = labelio.ORDER
        # 基数只列「玩家 / 怪物」，其余类别**有才列** —— 否则一行五个 0 很吵
        shown = order[:2] + [c for c in order[2:] if counts.get(c)]
        by_cls = " · ".join("%s %d" % (labelio.label(c), counts.get(c, 0))
                            for c in shown)

        # ⭐⭐ **分辨率写在帧名后面**（用户 2026-10-05 ✓ 原话："在质检台帧名后面显示分辨率"）。
        #   为什么值得放在这一行：**同一批里面混进别的分辨率**是真会发生的 ✓，而且后果很隐蔽 ✗ ——
        #   实测现场（阳光沙滩那个项目）：前 158 帧 1920×1080、后 28 帧 1280×720（重采时窗口变小 ✓）
        #   ⇒ 标定是按 1080p 做的（`scale_at` ✓）⇒ 720p 那批里目标小 1.5 倍 ⇒ 模板大 1.5 倍
        #   ⇒ **那批帧一个都检不出** ✗（日志只会泛泛说"scale 不对" ✓ 上一轮就是被它带偏的 ✓）。
        #   ⚠ 顺序要紧：**警告紧跟帧名 + 分辨率** ✗ —— 这一行是 `QSizePolicy.Ignored` ⇒ 太长会被
        #     裁掉 ✗，把"最该看见的"塞在末尾等于没有 ✓。
        res_txt = "%d×%d" % (w, h)
        at = self.project.get("scale_at") or {}
        warn = ""
        try:
            aw, ah = int(at.get("width") or 0), int(at.get("height") or 0)
        except (TypeError, ValueError):
            aw = ah = 0
        if aw and ah and (aw, ah) != (w, h):
            warn = "　⚠ 与标定不符（标定是 %d×%d）" % (aw, ah)

        self.lbl_frame.setText("%s　%s%s　%s%s"
                               % (path.name, res_txt, warn, by_cls,
                                  "　（人工改过）" if manual else ""))
        tip = "画面 %d×%d，本帧共 %d 个框" % (w, h, len(boxes))
        if warn:
            tip += ("\n\n⚠ 这一帧的分辨率与**标定时的画面**不一样（标定是 %d×%d ✓）。"
                    "模板是按标定那套尺寸缩放出来的 ⇒ 分辨率对不上的帧，"
                    "模板匹配**一个都检不出** ✓。\n"
                    "处理办法：把这批帧单独放一个项目，先在 ③ 重标一遍再标 ✓" % (aw, ah))
        elif aw and ah:
            tip += "\n（分辨率与标定一致 %d×%d ✓）" % (aw, ah)
        self.lbl_frame.setToolTip(tip)
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

    # ══════════════════════════════════════════════════
    # 操作模式（用户 2026-10-04 ✓ 第 3 条）
    # ══════════════════════════════════════════════════
    def set_mode(self, mode):
        """切**操作模式** —— 下拉 / **Q** / **W** 三条路都走这里 ✓（口径只此一处 ✓）。

        ⚠ 快捷键也要**同步下拉框**：按了 W 而下拉还写着"标注模式"，人会以为快捷键没生效 ✗
          （`setCurrentIndex` 自己会触发 `_on_mode_changed` ⇒ 剩下的活由它干，不重复 ✓）。
        ⚠ 传未知值 ⇒ 回到标注模式（同 `ImageCanvas.set_mode` 的取舍：宁可回到最熟的那个 ✓）。
        """
        i = self.cmb_mode.findData(mode)
        if i >= 0 and i != self.cmb_mode.currentIndex():
            self.cmb_mode.setCurrentIndex(i)
            return self.canvas.mode
        return self._on_mode_changed()

    def _on_mode_changed(self, *_):
        """下拉变了（或按了 Q/W）⇒ 同步画布 + 写一句反馈 ✓。返回生效的模式 ✓。"""
        mode = str(self.cmb_mode.currentData() or MODE_LABEL)
        self.canvas.set_mode(mode)
        if mode == MODE_REGION:
            # ⭐ 说清"下一步干什么" ✗（不然人切过来只会乱拖 ✓ 用户 2026-10-11 ✓）
            self._set_status("区域自动标注：在画面上拖一个**青框**圈出要标的地方 ⇒ 松手弹窗 ✓")
        else:
            self._set_status("操作模式：%s" % self.cmb_mode.currentText())
        return self.canvas.mode

    def _on_region_picked(self, rect):
        """框完一块 ⇒ 弹「区域自动标注」（用户 2026-10-11 ✓ 这一屏的正题 ✓）。

        ⚠ 三件事缺一不可：
          · **要有一帧**：没有当前帧就没有"只标这一张"的落点 ⇒ 老实说一声 ✓；
          · **框不能太小**：`region.to_spec` 对 `w<=0 / h<=0` 会返回空串 ⇒ 老实说一声 ✓
            （比弹出一个"没用的弹窗"强 ✓）；
          · ~~**模态**（`exec_`）~~ ⇒ ⭐⭐ **2026-10-16 改成非模态**（用户："**弹窗打开时需要能
            交互主窗口**"✓）：那条"以为标的是新帧"的风险 ✗ 现在改由**目标帧跟着当前帧走**
            来消（`set_target` + `_load_current` 里那次推送 ✓）—— 比"靠模态挡住"更结实 ✓。
            细节见下面动手处的注释 ✓。
        """
        if self.project is None or not self.items:
            self._set_status("先打开项目、选到一帧，再框 ✓")
            return
        spec = region_mod.to_spec(rect)
        # ⚠ 太小的一下**不算"框选"** ✗（画布那边也是这个口径 ⇒ 用同一个常数 ✓
        #   不然会出现"画布说可以、面板说不行"或反过来 ⇒ 两边各说各话 ✓ 见 `MIN_SIZE` ✓）
        _r = region_mod.parse(spec)
        if not _r or _r[2] < MIN_SIZE or _r[3] < MIN_SIZE:
            self._set_status("这个框太小了（至少 %d×%d 像素 ✓ 拖大一点再松手 ✓）"
                             % (MIN_SIZE, MIN_SIZE))
            return
        stem = self.items[self.index][0]
        from gui.region_label_dialog import RegionLabelDialog
        # ⭐⭐ **改成非模态**（用户 2026-10-16 ✓ 第 1 条："区域自动标注 弹窗打开时需要能
        #   交互主窗口"✓）—— ⚠ 上面那段"必须模态"的三条理由**仍然成立** ✓（弹窗按钮标的是
        #   "那一刻的那一帧" ✓）⇒ 所以**不是**简单把 `exec_` 换掉 ✗，而是**同时**把
        #   "要标哪一帧"变成**活的** ✓（两条一起才安全 ✓）：
        #     ① 弹窗标题常显「**目标帧：xxx**」✓（`set_target` ✓）；
        #     ② 质检台**每载入一帧**就把它推给弹窗 ✓（`_load_current` 里那三行 ✓）
        #   ⇒ 弹窗标的**永远是你正看着的那一帧** ✓ ⇒ 原来防的那件事（以为标的是新帧 ✗）
        #     从"靠模态挡住"变成"**靠同一份事实**"✓ 更结实 ✓。
        #   ⚠ 另外两条与"批量剔除"弹窗**同款**的讲究：**不许开第二个**（已开着就拎到前面 ✓）、
        #     结果改从**信号**收（非模态没有"等返回值"那回事 ✓ 见 `_on_region_closed` ✓）。
        _rd = getattr(self, "_region_dlg", None)
        if _rd is not None and _rd.isVisible():
            # ⭐⭐ **开着窗再框一块 ⇒ 就地换区域**（用户 2026-10-16 ✓ 原话："**我不能开着区域
            #   自动标注反复循环框选标注吗？**"✓）—— 以前这里只同步**帧名** + 把窗拎到前面 ✗
            #   ⇒ 刚框的那一块**被丢掉** ✗（再点按钮还是按**开窗时那块**标 ✓ = 白框 ✗，
            #   而人以为"这次标的是新框"✓）。
            #   ⚠ 与 `set_target` 是一对 ✓：**帧**跟当前帧走 ✓、**区域**跟刚框的那块走 ✓
            #   ⇒ 于是"框一块 → 标 → 再框一块 → 再标"这条循环成立 ✓
            #   （`_on_run` 是**点按钮那一刻**才读这两份 ✓ 见它里面那次兜底 ✓）。
            _rd.set_target(stem)
            _rd.set_region(spec)
            _rd.raise_()
            _rd.activateWindow()
            self._set_status("区域已更新成刚框的那块 ✓ 开着窗就能一直「框→标→再框」✓")
            return
        dlg = RegionLabelDialog(self, self.project, stem, spec)
        dlg.setModal(False)
        dlg.setWindowModality(Qt.NonModal)
        self._region_dlg = dlg                     # ⚠ 存引用：不然会被 GC 掉、窗口一闪就没 ✗
        dlg.finished.connect(lambda _r: self._on_region_closed(dlg))
        # ⭐⭐⭐ **每跑完一趟就刷一次**（用户 2026-10-16 ✓ 原话："**显示 ✓ 完成：1 帧 / 本次 1 框 /
        #   有检出 100%，但是实际并没看到有框**"✓）—— 见 `_on_region_labeled` ✓。
        dlg.labeled.connect(self._on_region_labeled)
        dlg.show()
        self._set_status("区域自动标注：弹窗开着也能翻帧 ✓ —— 它标的**永远是你正看着的那一帧** ✓")

    def _on_region_labeled(self, stem):
        """区域标注**跑完一趟** ⇒ 立刻重载一次（用户 2026-10-16 ✓ 见弹窗那个 `labeled` 信号 ✓）。

        ⚠ 为什么不能只靠"关窗才重载"✗：弹窗现在是**非模态** ✓，人可以开着它连标好几帧 ✓
          ⇒ 只挂在 `finished` 那条路上 ⇒ **屏幕一直是旧的** ✗
          （用户实测：日志显示"✓ 完成：1 帧 / 本次 1 框"，屏幕上却没框 ✓
           磁盘上框其实已经写了 ✓）。
        ⚠ 两件讲究：
          · **只在你正看着这一帧时**才重载 ✓ —— 你已经翻到别的帧了，就别把画面抢回来 ✗
            （那会让人以为"我翻的那帧被改了"✓）；这时**在状态行说清**刚标的是哪一帧 ✓；
          · 重载走 `reload()` ✓ ⇒ 顺带吃上"**定位回原帧**"那条 ✓（见 `_restore_pos` ✓）。
        """
        cur = self.items[self.index][0] if 0 <= self.index < len(self.items) else ""
        if stem and stem == cur:
            self.reload()
            self._set_status("区域自动标注完成 ⇒ 已重新载入 %s ✓" % stem, ok=True)
        else:
            self._set_status(
                "区域自动标注完成：标的是 %s（你现在看的是 %s ✓ 切过去就能看到框 ✓）"
                % (stem or "?", cur or "—"), ok=True)

    def _on_region_closed(self, dlg):
        """区域弹窗关掉之后（**非模态 ⇒ 结果从信号来** ✓ 见 `_on_region_picked` ✓）。

        ⚠ 与原来 `exec_` 之后那段**一模一样**的口径 ✓：弹窗里跑过标注 ⇒ 框已经变了 ⇒
          **重新载入**（不重载 ⇒ 屏幕上是旧框 ✓ 人会以为"没标上" ✗）；只是打开看了看
          ⇒ 别白重载一遍 ✓。
        ⚠ 重载走 `reload()` ✓ ⇒ 顺带吃上"**定位回原帧**"那条 ✓（用户 2026-10-16 第 2 条 ✓）。
        """
        if getattr(dlg, "_done_once", False):
            self.reload()
            self._set_status("区域自动标注已跑完 ⇒ 已重新载入当前帧 ✓", ok=True)

    def _on_selection_changed(self, n):
        """选中数变了 ⇒ 写一句 —— **批量删**这条路就靠这句告诉人下一步按什么 ✓。"""
        if n > 0:
            self._set_status("已选 %d 个框 —— 按 Del 或点「删选中框」一起删" % n, ok=True)
        elif self.canvas.mode == MODE_SELECT:
            self._set_status("没有选中的框 —— 在空白处拖一个黑框，被它框住的框会被选中 ✓")

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

    def _drop_frames_batch(self):
        """**批量剔除**（用户 2026-10-16 ✓ 原话："**给质检台加一个批量选帧删除的弹窗功能
        （类似自动标注的选帧）**"✓）：弹窗勾选 → **二次确认** → 一次删掉 ✓。

        ⭐⭐ **非模态**（用户 2026-10-16 ✓ 原话："**这个弹窗打开时，主窗口需要能交互**"✓）：
          · 所以走 `show()` **不是** `exec_()` ✗ —— 主窗口（翻帧、看框、改框）**照常能点** ✓；
          · ⚠ 非模态带来两个新坑，都在下面处理了：
            ① **能开两个** ✗ ⇒ 已开着就把老的**拎到前面**、不再新建 ✓；
            ② **开着的那段时间列表可能变** ✗（人在主窗口翻帧/改框/甚至单帧剔除 ✓）
              ⇒ 真正删之前**只认"现在还在这份列表里"的帧** ✓（其余跳过并如实报数 ✓）。
        ⚠ 语义**完全沿用**上面那个单帧的（`_drop_frame` ✓）：删的是**同一组文件**
          （图片 / 自动标注 / 人工标注 / 可视化 ✓ 见 `labelio.delete_frame` ✓）
          + 把这几帧从**所有**类的处理台账里划掉 ✓（图都没了，选帧弹窗不该再列它 ✓
          否则勾上会报"找不到"✗ —— 同 `_drop_frame` 那条注释 ✓）。
        ⚠ **不新造第二种"删除"** ✗（那种迟早两边对不上 ✓）。
        """
        if not self.items:
            return

        # ① 已经开着 ⇒ 拎到前面（**不开第二个** ✗）
        dlg = getattr(self, "_batch_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return

        self._save_current()          # 当前帧的改动先落盘 ✓（别顺手把它丢了 ✗）
        dlg = BatchDropDialog(self, self.items)
        dlg.setModal(False)           # ⭐ 主窗口必须还能点 ✓
        dlg.setWindowModality(Qt.NonModal)
        self._batch_dlg = dlg         # ⚠ 存一份引用：不然会被 GC 掉、窗口一闪就没 ✗
        dlg.accepted.connect(self._on_batch_drop_accept)
        dlg.show()                    # ⭐ **非模态**：不等结果、不挡主窗口 ✓

    def _on_batch_drop_accept(self):
        """弹窗按了「剔除选中的帧」⇒ 二次确认 + 真删（非模态 ⇒ 结果从信号来 ✓）。"""
        dlg = getattr(self, "_batch_dlg", None)
        stems = dlg.selected() if dlg is not None else []
        if not stems:
            QMessageBox.information(self, "批量剔除", "一帧都没勾 ✓")
            return

        # ② ⚠ 弹窗开着这段时间，主窗口可能已经动过列表（翻帧/改框/单帧剔除 ✓）
        #    ⇒ **只认还在这份列表里的** ✓（否则会去删一个"已经不在清单里"的名字 ✗
        #      —— `delete_frame` 对不存在的文件是 0 个，但台账/计数会变得难解释 ✓）。
        alive = {it[0] for it in self.items}
        use = [s for s in stems if s in alive]
        skipped = len(stems) - len(use)
        if not use:
            QMessageBox.information(
                self, "批量剔除",
                "勾的这些帧**已经不在当前列表里**了（可能刚被剔过 / 换了筛选 ✓）")
            return

        extra = ("\n\n⚠ 另有 %d 帧已经不在列表里 ⇒ **会跳过** ✓" % skipped) if skipped else ""
        r = QMessageBox.question(
            self, "批量剔除 %d 帧" % len(use),
            "将删除这 %d 帧的图片、自动标注、人工标注和可视化图。%s\n\n"
            "这些图不再参与训练。\n\n"
            "适合处理：被窗口/UI 大面积遮挡、或怪被遮到认不出的帧。"
            % (len(use), extra),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return

        n_files = 0
        for s in use:
            n_files += labelio.delete_frame(self.project, s)
        # ⚠ **台账一次划掉**（`unmark_processed` 收的是 list ✓）：图片都没了，
        #   选帧弹窗再列出来就会"勾了却找不到" ✗。
        labelio.unmark_processed(self.project.dir_of("labels_auto"), use)

        gone = set(use)
        self.items = [it for it in self.items if it[0] not in gone]
        self._dirty = False
        self.index = max(0, min(self.index, len(self.items) - 1))
        self._update_pos()
        self._load_current()
        self._set_status("已批量剔除 %d 帧（删除 %d 个文件%s）"
                         % (len(use), n_files,
                            "，跳过 %d" % skipped if skipped else ""), ok=True)

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
