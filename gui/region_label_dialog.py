"""「区域自动标注」弹窗 —— 框选一块，**只在那块里标**（用户 2026-10-11 ✓）。

用户原话（这一屏就是照它做的 ✓）：

    给质检台新增「区域自动标注」操作模式
    在该模式，框选一个区域后出现弹窗，弹窗里的内容类似模型训练页签的自动标注卡片，参数布置如：
      YOLO权重→列表<权重>
      「标注玩家」→勾选项
      「标注怪物」→勾选项
      「标准掉落物」→勾选项
      「标注宠物」→勾选项
    每个勾选项被勾选时，在该勾选项下面显示出详细的配置内容及按钮，**直接照抄自动标注卡片的相关内容**
    ... 点击各个标注按钮，**只会在当前帧框选的区域内**进行自动标注
    注意弹窗要支持滚动

⚠⚠ 怎么做到"**直接照抄**"而不抄出两份（那必然越抄越歪 ✗）：
  弹窗内部持有一张**真的** `LabelCard`（`gui/steps/cards.py` ✓ 不显示 ✓），并把它的
  **四块原件**（`player_box` / `mob_box` / `drop_box` / `pet_box` ✓）与 **YOLO 权重下拉**
  （`cmb_yolo` ✓）**搬进**弹窗 ✓ —— Qt 的 `addWidget` 会把控件**重新认爹** ✓
  ⇒ 参数、提示、按钮、按钮接线**全是同一份** ✓（卡片改了这里自动跟着改 ✓）。

三条纪律（都写在对应函数的说明里 ✓）：
  · **只处理当前帧** ✗：把卡片的 `_pick_frames` 覆盖成"就这一帧" ✓（卡片那几个按钮本来会弹
    "选帧窗" ⇒ 这里不该弹 ✓ 用户要的是"当前帧框选的区域" ✓）；
  · **只标框选区域** ✗：把 `card._region` 设成这次的矩形 ✓（四条链各自读它 ✓ 见
    `perception/region.py` ✓ 一处口径 ✓）；
  · **不许阻塞界面** ✗：跑标注走 `gui.worker.TaskThread` ✓（与主窗口同一套读条/日志/取消 ✓）。
"""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel,
                             QPlainTextEdit, QProgressBar, QPushButton,
                             QVBoxLayout, QWidget)

from gui.steps.cards import LabelCard
from gui.widgets import scroll_page, title_label
from gui.worker import TaskThread, safe_slot
from perception import region as region_mod

#: 四个勾选项（`(键, 中文, 那一块在卡片里的属性名 ✓)`）—— **一处口径** ✓
#: ⚠ 顺序照用户给的那份（玩家 → 怪物 → 掉落物 → 宠物 ✓）。
TARGETS = (
    ("player", "标注玩家", "player_box"),
    ("mob", "标注怪物", "mob_box"),
    ("drop", "标准掉落物", "drop_box"),
    ("pet", "标注宠物", "pet_box"),
)


class RegionLabelDialog(QDialog):
    """见模块开头那段（照用户原话做的 ✓）。"""

    #: ⭐⭐⭐ **跑完一次就报一次**（用户 2026-10-16 ✓ 原话："**显示 ✓ 完成：1 帧 / 本次 1 框 /
    #:   有检出 100%，但是实际并没看到有框**"✓）—— 参数是**刚标完的那一帧** ✓。
    #:   ⚠ 为什么必须有它：本弹窗现在是**非模态** ✓，而"重载当前帧"原来只挂在**关闭**那一下
    #:     （`finished` ⇒ `review._on_region_closed` ✓）⇒ **弹窗不关就一直不重载** ✗
    #:     ⇒ 磁盘上框已经有了（`labels_auto/<帧>.txt` ✓ 实测确实写了 ✓）、屏幕上却是旧的 ✗
    #:     = 用户看到的那一幕 ✓。⇒ 改成"**每跑完一次就通知质检台刷一次**" ✓。
    labeled = pyqtSignal(str)

    def __init__(self, parent=None, project=None, frame_stem="", region="",
                 title="区域自动标注"):
        super().__init__(parent)
        self.setWindowTitle(title)
        #: ⭐ 标题基名（`set_target` 会往标题上缀「目标帧：xxx」✓ 见那边 ✓）
        self._base_title = str(title or "区域自动标注")
        #: ⭐ 「只标哪儿」那行 QLabel 的句柄（**非模态**时要随当前帧改写 ✓ 见 `_render_target` ✓）
        self.lbl_target = None
        self.setSizeGripEnabled(True)
        self.project = project
        self.frame_stem = str(frame_stem or "")
        #: ⭐ 这一趟**只标这个矩形**（`"x,y,w,h"` ✓ 画面像素 ✓）—— 见 `perception/region.py` ✓
        self.region_spec = region_mod.to_spec(region)
        self.task = None
        self._done_once = False

        # ⚠ 卡片**不显示**（它是"零件库" ✓），但必须 `bind` 一次 ⇒ 参数回填成当前项目的值 ✓
        #   （不回填 ⇒ 弹窗里的参数是默认值 ⇒ 和训练页看到的对不上 ✓ 那就是两个真相 ✗）
        self.card = LabelCard()
        self.card.setParent(self)
        try:
            self.card.bind(project)
        except Exception:                        # noqa: BLE001 —— 回填失败不该打不开弹窗 ✗
            pass
        # ⭐⭐ **只标框选区域**（四条链各自读它 ✓ 一处口径 ✓）
        self.card._region = self.region_spec or None
        # ⭐⭐ **只处理当前帧**（覆盖"选帧窗"✓）：卡片那几个按钮会问"跑哪些帧" ✓
        #   而这一屏的语义是**就这一帧** ✓（用户："只会在**当前帧**框选的区域内进行自动标注"✓）。
        self.card._pick_frames = lambda _target: (True, [self.frame_stem] or None)
        self.card.run_clicked.connect(safe_slot(self._on_run))

        self._build()
        for _k, _c, _attr in TARGETS:
            self._set_section_visible(_attr, False)      # 一开始都收着 ✓（照用户说的"勾上才出现" ✓）

    # ---------------- 目标帧（非模态 ⇒ 得跟着当前帧走 ✓）----------------

    def set_target(self, stem):
        """把"**要标哪一帧**"改成 `stem`（用户 2026-10-16 ✓ 由质检台每载入一帧推过来 ✓）。

        ⚠ 为什么必须有它：本弹窗原来是**模态** ✓，理由是"那几个按钮标的是**打开弹窗那一刻**
          的那一帧"✓ —— 一旦允许主窗口交互（非模态 ✓），**不把目标帧跟着当前帧走** ✗
          就会出现"人在背后换了帧、再按按钮、以为标的是新帧" ✗（正是原来要防的那件事 ✓）。
        ⇒ 现在：目标帧 = **你正看着的那一帧** ✓（标题与顶部那行都实时显示 ✓ 看得见才敢按 ✓）。
        """
        self.frame_stem = str(stem or "")
        self._sync_target_title()
        self._render_target()

    # ---------------- ⭐ 当前框选区域的预览（只读 ✓）----------------

    def _refresh_preview(self):
        """把**当前帧 + 当前区域**裁一块显示出来（用户 2026-10-16 ✓ 原话："**在区域自动标注
        弹窗里，下面加一个当前框选区域显示的视图区**"✓）。

        · 作用：**一眼确认"这次到底要标哪一块"** ✓ —— 以前只能靠顶上一行字
          `只标框选区域 (x, y) w×h` 去想象 ✗（框偏了 / 框小了都看不出来 ✗）；
        · 它跟着**同一份来源**走 ✓：`self.frame_stem` + `self.region_spec` ✓
          （由 `set_target` / `set_region` ⇒ `_render_target` 一路叫过来 ✓）
          ⇒ **不存在"预览一块、实际标另一块"** ✗（那是这种预览最容易出的错 ✓）；
        · ⚠ **只读** ✗：不写文件、不改区域、不改帧 ✓（要改还是回画布上框 ✓）；
        · ⚠ **每一步都可能缺**（没帧 / 没区域 / 图读不出来 ✓）⇒ 一律显示**一句人话**、
          绝不抛异常 ✗ —— 它是"帮着确认"的控件，不该把弹窗搞崩 ✓。
        """
        lbl = getattr(self, "lbl_region_view", None)
        if lbl is None:
            return

        def _say(msg):
            lbl.setPixmap(QPixmap())
            lbl.setText(msg)

        _r = region_mod.parse(self.region_spec) if self.region_spec else None
        if not self.frame_stem or _r is None:
            _say("（还没框区域 / 还没选帧）")
            return
        try:
            pm = QPixmap(str(Path(self.project.frames) / (self.frame_stem + ".png")))
        except Exception:                        # noqa: BLE001
            pm = QPixmap()
        if pm.isNull():
            _say("读不到这一帧的图：%s" % self.frame_stem)
            return

        x, y, w, h = (int(v) for v in _r)
        # ⚠ **向外各带 15% 的周边**（至少 8px ✓）：只看区域内部会**分不清边界在哪** ✗
        #   ⇒ 带上一点周围，一眼看得出"框到哪儿为止" ✓（连着框几块时尤其有用 ✓）。
        pad_x, pad_y = max(8, int(w * 0.15)), max(8, int(h * 0.15))
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(pm.width(), x + w + pad_x), min(pm.height(), y + h + pad_y)
        if x1 <= x0 or y1 <= y0:
            _say("这块区域不在画面里（%s）" % self.region_spec)
            return

        crop = QPixmap(pm.copy(x0, y0, x1 - x0, y1 - y0))
        # ⭐ 把**区域框**画在裁出来的那块上（青色实线 ✓ —— 与画布上那根"区域自动标注"
        #   橡皮筋**同一个色** ✓ 见 `gui/canvas` 里那条说明 ✓）：一眼就是"这块要标" ✓。
        p = QPainter(crop)
        try:
            pen = QPen(QColor(0, 200, 255))
            pen.setWidth(2)
            p.setPen(pen)
            p.drawRect(int(x - x0), int(y - y0),
                       max(1, min(w, x1 - x)), max(1, min(h, y1 - y)))
        finally:
            p.end()

        lbl.setText("")
        lbl.setToolTip("当前要标的那一块：%d×%d @ (%d, %d)（含周边各 %d px ✓ 只读）"
                       % (w, h, x, y, pad_x))
        lbl.setPixmap(crop.scaled(max(1, lbl.width() - 4), max(1, lbl.height() - 4),
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_region(self, region):
        """把"**只标哪一块**"换成刚框的那一块（用户 2026-10-16 ✓ 原话："**我不能开着区域
        自动标注反复循环框选标注吗？**"✓）。

        ⚠ 为什么必须有它：区域原来是**开窗那一刻**定死的 ✗（`__init__` 里存一次 ✓）
          ⇒ 弹窗开着时再拖一个新框，质检台只能"把窗拎到前面" ✗ ⇒ 那一框**白框了** ✗
          （人以为"这次标新框"✓，其实拿**旧区域**又标了一遍 ✗）。
        ⇒ 现在：开着窗再框 ⇒ **就地换区域** ✓ ⇒ 可以一直开着窗
          "**框一块 → 点按钮标 → 再框一块 → 再标**" ✓（区域与目标帧都是**按按钮那一刻**读的 ✓
          见 `_on_run` ✓ ⇒ 每次点都吃最新的那份 ✓）。
        ⚠ 与 `set_target` 是**一对** ✓：一个换"标哪一帧"✓、一个换"标哪一块"✓
          —— 两个都跟着当前状态走，才能"开着窗连着改" ✓。
        """
        self.region_spec = region_mod.to_spec(region)
        self._render_target()
        try:
            self._log("区域已更新 ✓ 直接点按钮就按**这一块**标 ✓")
        except Exception:                        # noqa: BLE001 —— 写日志失败不该炸接线 ✓
            pass

    def _sync_target_title(self):
        """标题上常显「目标帧」（非模态时它是最显眼的那个锚 ✓）。"""
        self.setWindowTitle("%s —— 目标帧：%s"
                            % (self._base_title, self.frame_stem or "（没选帧）"))

    def _render_target(self):
        """改写顶部那行"这次只标哪儿"（区域 + **目标帧** ✓ 一份格式串 ✓）。"""
        if self.lbl_target is None:
            return
        _r = region_mod.parse(self.region_spec)
        # ⚠⚠ **必须防 `None`** ✗（2026-10-16 ✓ 桩测逮到的真 bug ✓）：区域为空 / 解析不出来时
        #   `region_mod.parse()` 返回 `None` ✓，而下面直接下标 ⇒ `TypeError` ⇒
        #   **整个质检台会炸** ✗（⚠ `set_target` 每换一帧都会走到这儿 ✓ —— 也就是说
        #   "非模态推目标帧"这条新链一旦遇到空区域，就是**翻一帧崩一次** ✗）。
        if _r is None:
            self.lbl_target.setText(
                "⚠ 还没框区域（或这块读不出来）⇒ 回画布上框一块再点按钮 ✓")
            # ⚠⚠ **这里也得刷预览** ✗（2026-10-16 ✓ 桩测第 ⑤ 条逮到 ✓）：
            #   早退会**留着上一块的旧图** ✗ ⇒ 顶上写着"还没框区域"、下面却还挂着一张
            #   旧区域的截图 ✓ —— 最坏的情况是**照着旧图点按钮** ✗（正好是这种"确认类"
            #   控件最不该有的状态 ✓）。⇒ 让预览也走到"没区域"那条分支 ✓。
            self._refresh_preview()
            return
        self.lbl_target.setText(
            "⭐ **只标框选区域** (%d, %d) %d×%d ✓ ／ **只处理当前帧** `%s` ✓"
            % (_r[0], _r[1], _r[2], _r[3], self.frame_stem or "（没选帧）"))
        # ⭐ 顺手把**预览**也刷了（用户 2026-10-16 ✓）：`set_region` / `set_target` 都会走到
        #   这儿 ⇒ 一处接线、两处生效 ✓（换块、换帧，预览都跟着 ✓ 不用各写一遍 ✗）。
        self._refresh_preview()

    # ---------------- 界面 ----------------

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # ---- 顶部：这次"只标哪儿"说在最前面（人要一眼看见自己框的是哪块 ✓）----
        # ⭐ 2026-10-16：这行改成**活的** ✓ —— 存下句柄、文案交给 `_render_target`（**一份格式串**
        #   ✓ 别再抄第二遍 ✗）；非模态时质检台每换一帧都会把它改写 ✓（见 `set_target` ✓）。
        _hint = QLabel("")
        self.lbl_target = _hint
        self._render_target()
        _hint.setWordWrap(True)
        _hint.setFrameShape(QFrame.StyledPanel)
        _hint.setToolTip("区域外的目标**一个都不会标** ✓ —— 判据是「框的中心点在不在区域里」✓\n"
                         "（见 `perception/region.py`：四条标注链共用同一套 ✓）")
        outer.addWidget(_hint)

        # ---- YOLO 权重：**搬卡片那个下拉原件** ✓（选择结果卡片自己也读得到 ✓ 一处口径 ✓）----
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(title_label("YOLO 权重", "「YOLO 标注」那几个按钮用的模型 ✓"))
        self.card.cmb_yolo.setParent(self)
        row.addWidget(self.card.cmb_yolo, 1)
        outer.addLayout(row)

        # ---- 四个勾选项（勾上 ⇒ 它下面展开那一块 ✓）----
        cks = QHBoxLayout()
        cks.setSpacing(14)
        self.checks = {}
        for key, label, _attr in TARGETS:
            ck = QCheckBox(label)
            ck.setToolTip("勾上 ⇒ 下面展开「%s」的详细配置与按钮 ✓（内容与「自动标注」卡片**同一份** ✓）"
                          % label)
            ck.stateChanged.connect(lambda _s, k=key: self._on_check(k))
            self.checks[key] = ck
            cks.addWidget(ck)
        cks.addStretch(1)
        outer.addLayout(cks)

        # ---- 中间：四块（**滚动区** ✓ 参数+按钮比屏幕高 ✓）----
        holder = QWidget()
        page = scroll_page(holder)
        self.sections = {}
        for key, label, attr in TARGETS:
            box = QWidget()
            lay = QVBoxLayout(box)
            lay.setContentsMargins(0, 4, 0, 8)
            lay.setSpacing(6)
            lay.addWidget(title_label("「%s」的配置（与「自动标注」卡片同一份 ✓）" % label))
            _w = getattr(self.card, attr, None)
            if _w is not None:
                _w.setParent(box)
                lay.addWidget(_w)                # ⭐ 搬原件 ⇒ 参数/按钮/接线全是同一份 ✓
            self.sections[attr] = box
            page.addWidget(box)
        page.addStretch(1)
        outer.addWidget(holder, 1)

        # ---- 底部：读条 + 日志 + 取消/关闭（与主窗口同一套体验 ✓）----
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(400)
        self.log_view.setMinimumHeight(84)
        self.log_view.setPlaceholderText("点上面任意一个标注按钮 ⇒ 这里出日志 ✓")
        outer.addWidget(self.log_view)

        # ⭐⭐⭐ **当前框选区域的视图区**（用户 2026-10-16 ✓ 原话："**在区域自动标注弹窗里，
        #   下面加一个当前框选区域显示的视图区**"✓）—— 摆在**日志下面、按钮上面** ✓。
        #   · 作用：**一眼确认"这次要标哪一块"** ✓（顶上一行字只能靠想象 ✗）；
        #   · 内容 = 当前帧裁到那一块 + 各带一点周边 ✓ + **青色框**标出区域范围 ✓
        #     （与画布上那根橡皮筋同色 ✓）；
        #   · ⚠ **只读** ✗：不改区域、不写文件 ✓（这是"确认"，不是"编辑"✗）。
        self.lbl_region_view = QLabel()
        self.lbl_region_view.setMinimumHeight(150)
        self.lbl_region_view.setAlignment(Qt.AlignCenter)
        self.lbl_region_view.setFrameShape(QFrame.StyledPanel)
        self.lbl_region_view.setStyleSheet("background: #fafafa;")
        self.lbl_region_view.setToolTip(
            "当前要标的那一块（含一点周边 ✓ 青色框 = 区域边界 ✓）\n"
            "⚠ 只读：想改区域就回画布上重新框一次 ✓")
        outer.addWidget(self.lbl_region_view)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_close)
        outer.addLayout(btns)

        # ⭐ 建完先刷一次预览（用户 2026-10-16 ✓ 见上面那个视图区 ✓）——
        #   ⚠ 放在**最后**：那会儿控件才有尺寸 ✓（`scaled()` 按当前尺寸缩 ✓）。
        self._refresh_preview()
        self.resize(760, 640)

    def _set_section_visible(self, attr, on):
        w = self.sections.get(attr)
        if w is not None:
            w.setVisible(bool(on))

    def _on_check(self, key):
        """勾上/取消 ⇒ 展开/收起那一块 ✓（**只是显隐** ✓ 不动卡片的勾选状态 ✗）。"""
        for k, _label, attr in TARGETS:
            if k == key:
                self._set_section_visible(attr, self.checks[k].isChecked())

    # ---------------- 跑标注 ----------------

    def _on_run(self, card):
        """卡片里点了一个标注按钮 ⇒ **在这儿起任务**（与 `main_window._on_run` 同一套 ✓）。"""
        if self.task is not None:
            self._log("已有任务在跑，等它结束或点「取消」✓", "warn")
            return
        if self.project is None:
            self._log("还没打开项目 ✓", "warn")
            return
        if not self.frame_stem:
            self._log("没选帧 ⇒ 不知道该标哪一张 ✓", "warn")
            return
        # ① 参数回写项目（**必须** ✓ 不然 `make_task` 读到的是项目里的旧值 ⇒ 弹窗里改了等于没改 ✗）
        try:
            card.sync(self.project)
            self.project.save()
        except Exception as e:                   # noqa: BLE001
            self._log("参数回写失败：%s: %s" % (type(e).__name__, e), "error")
            return
        # ② ⚠⚠ **兜底再钉一次"只这一帧 + 只这块区域"** ✗：卡片那几个 `_emit_*` 会先弹"选帧窗"
        #   （已经被我们覆盖成"就这一帧" ✓ 见 `__init__` ✓），照理 `_only[target]` 就有了 ✓；
        #   但**将来有人加一条不走那个入口的按钮** ⇒ `only` 是空 ⇒ **整份画面重标一遍** ✗✗ ——
        #   这一屏的语义是"**当前帧**框选区域"✓ ⇒ 在最靠近任务的地方再钉一次最可靠 ✓
        #   （"只标区域"同理 ⇒ `_region` 也再设一遍 ✓）。
        _tgt = str(getattr(card, "_label_target", "") or "")
        card._region = self.region_spec or None
        if _tgt:
            card._only[_tgt] = [self.frame_stem]
        try:
            made = card.make_task(self.project)
        except Exception as e:                   # noqa: BLE001
            self._log("没法运行：%s: %s" % (type(e).__name__, e), "error")
            return
        if made is None:
            self._log("这一步没有可跑的任务 ✓", "warn")
            return
        fn, params = made
        # ③ 起后台任务（⚠ 只读线程池 / 进度条都在 `TaskThread` 里 ✓ 界面不会假死 ✓）
        self._log("── 开始：%s（%s）" % (getattr(fn, "__name__", fn),
                                        region_mod.note(self.region_spec) or "整幅画面"), "info")
        self.progress.setVisible(True)
        self.btn_cancel.setEnabled(True)
        self.btn_close.setEnabled(False)
        self.task = TaskThread(fn, params, self)
        self.task.sig_log.connect(safe_slot(self._log))
        self.task.sig_progress.connect(safe_slot(self._on_progress))
        self.task.sig_done.connect(safe_slot(self._on_done))
        self.task.finished.connect(self._on_finished)
        self.task.start()

    def _on_progress(self, cur, total, text):
        if total > 0:
            self.progress.setRange(0, int(total))
            self.progress.setValue(int(cur))
            self.progress.setFormat("%v/%m %p%%  %s" % (text or ""))
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat(text or "运行中…")

    def _on_done(self, ok, summary, result):
        self._done_once = True
        self._log(("✓ 完成：" if ok else "✗ 失败：") + str(summary or ""),
                  "ok" if ok else "error")
        # ⭐ **通知质检台刷一次**（用户 2026-10-16 ✓ 见上面那个信号 ✓）：
        #   ⚠ 只在**成功**时报 ✓ —— 失败那趟磁盘上什么都没变 ⇒ 刷屏只会让人以为"框变了" ✗。
        #   ⚠ 报的是**这一趟标的那一帧** ✓（`self.frame_stem` 就是这一趟的目标 ✓
        #     见 `_on_run` 里那次 `card._only[_tgt] = [self.frame_stem]` ✓）。
        if ok:
            try:
                self.labeled.emit(self.frame_stem)
            except Exception:                    # noqa: BLE001 —— 报不动不该炸任务回调 ✓
                pass
        try:
            self.card.refresh(force=True)        # 跑完刷新卡片自己的产物摘要 ✓
        except Exception:                        # noqa: BLE001
            pass

    def _on_finished(self):
        self.task = None
        self.progress.setVisible(False)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)

    def _cancel(self):
        if self.task is not None:
            self.task.cancel()
            self._log("已请求取消（跑到当前这一帧结束就停 ✓）", "warn")

    def _log(self, msg, level="info"):
        self.log_view.appendPlainText(str(msg))

    # ---------------- 关闭 ----------------

    def closeEvent(self, e):
        """⚠ 有任务在跑 ⇒ **先取消再关** ✗（不然界面关了、后台还在动磁盘 ✓ 用户会以为卡住 ✓）。"""
        if self.task is not None:
            try:
                self.task.cancel()
                self.task.wait(3000)
            except Exception:                    # noqa: BLE001
                pass
        e.accept()
