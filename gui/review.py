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
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QShortcut, QSizePolicy, QVBoxLayout, QWidget)

from gui import labelio, theme
from gui.canvas import MODE_LABEL, MODE_SELECT, ImageCanvas
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
               # ⭐ 「只看有掉落物」（用户 2026-10-04 ✓ 第 2 条）：掉落物是**人工标的**
               #   （自动标注只产 玩家/怪物 ✓ 见 `tools/label_drops`），所以这条实际上是
               #   "**挑出标过掉落物的帧**" —— 复核/补漏掉落物标注时用得上 ✓。
               ("只看有掉落物", "has_drop"),
               # ⭐ 「只看有宠物」（用户 2026-10-05 ✓）：宠物是**干扰类**（class 5 ✓ 见
               #   `perception/classes.py` ✓）—— 自动标注**会**产它（`tools/label_pets` ✓ 模板匹配
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
        self.cmb_mode.setToolTip(
            "**标注模式**（默认 / 按 Q）：空白处拖 = **拉一个新框** ✓\n"
            "**选择模式**（按 W）：空白处拖 = **黑色虚线**框选 ⇒ 松手**选中被框住的框** ✓\n"
            "　　**不会建框** ✓ ⇒ 按 Del 或点「删选中框」一次删掉一批（清误检用 ✓）；\n"
            "　　不按 Ctrl = 只选这一批，按住 Ctrl = **加选** ✓。\n\n"
            "两种模式下都一样：框上拖 = 移动 / 框边拖 = 缩放 / 滚轮 = 缩放 /\n"
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
        # 「其他玩家」只人工标（自动标注不产这个类，见 perception/classes.py）。
        # ⭐ **掉落物**（用户 2026-10-04 ✓ 第 1 条："质检台现在需要能手动标注掉落物类"）——
        #   排第二：自动标注产的是 玩家/怪物（见 `tools/label_drops` ✓），而**掉落物全靠人工补**
        #   ⇒ 它和怪物一样常用 ✓（`CLASS_DROP = 2` 见 `perception/classes.py` ✓ 颜色/名字都在那儿 ✓）。
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
        self._set_status("操作模式：%s" % self.cmb_mode.currentText())
        return self.canvas.mode

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
