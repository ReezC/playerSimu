"""步骤卡片基类。

一张卡片的布局：

    ┌───────────────────────────────────────┐
    │ ● ② 采集                      [提示]   │  header：状态灯 + 标题
    │ ───────────────────────────────────── │
    │ 来源    [文件 ▾]                       │  body：参数区（子类填）
    │ 文件    [____________] [浏览]          │
    │ ───────────────────────────────────── │
    │ 843 帧                        [运行]    │  footer：结果摘要 + 动作
    └───────────────────────────────────────┘

子类需要实现：
    build_params(form)          往参数区塞控件
    load_from_project(project)  从项目配置回填控件
    make_task(project)          返回 (任务函数, 参数字典)，None 表示暂不可用
    check_deps(project)         返回 (是否就绪, 原因)
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QFileDialog, QFormLayout, QFrame,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout)

from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox,   # noqa: F401
                         NoWheelSpinBox, field_tip)

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"

# 状态 → (圆点颜色, 说明)
STATES = {
    "idle":    ("#9aa0a6", "未开始"),
    "running": ("#4285f4", "进行中"),
    "done":    ("#34a853", "完成"),
    "warn":    ("#fbbc04", "完成但存疑"),
    "fail":    ("#ea4335", "失败"),
}


class StepCard(QFrame):
    run_clicked = pyqtSignal(object)     # 参数：本卡片（主窗口负责启动任务）
    view_clicked = pyqtSignal(object)    # 请求主视区显示本步骤的产物

    def __init__(self, num, key, title, hint="", parent=None):
        super().__init__(parent)
        self.num = num
        self.key = key
        self.title = title
        self.hint = hint

        self.project = None
        self.widgets = {}       # key -> (widget, kind)
        self._extra = {}        # key -> 额外控件（如 path 的浏览按钮），隐藏整行时用
        self.state = "idle"

        self.setObjectName("StepCard")
        self.setFrameShape(QFrame.StyledPanel)

        self._build()

    # ---------------- 构建界面 ----------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # --- header ---
        head = QHBoxLayout()
        head.setSpacing(6)

        self.lamp = QLabel("●")
        self.lamp.setFixedWidth(14)
        self.lamp.setStyleSheet("color: %s;" % STATES["idle"][0])

        label = QLabel("%s %s" % (CIRCLED[self.num - 1] if self.num <= len(CIRCLED) else self.num,
                                  self.title))
        label.setStyleSheet("font-weight: 600;")

        head.addWidget(self.lamp)
        head.addWidget(label)
        head.addStretch(1)

        self.state_text = QLabel(STATES["idle"][1])
        self.state_text.setStyleSheet("color: #5f6368;")
        head.addWidget(self.state_text)
        root.addLayout(head)

        # --- hint：说明小字，紧跟标题，比放卡片底部更好读 ---
        if self.hint:
            note = QLabel(self.hint)
            note.setStyleSheet("color: #80868b;")
            note.setWordWrap(True)
            root.addWidget(note)

        # --- body：参数区 ---
        self.form = QFormLayout()
        self.form.setContentsMargins(0, 2, 0, 2)
        self.form.setSpacing(4)
        self.form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.build_params(self.form)
        root.addLayout(self.form)

        # --- footer：结果摘要 + 动作按钮 ---
        foot = QHBoxLayout()
        foot.setSpacing(6)

        self.result = QLabel("—")
        self.result.setStyleSheet("color: #5f6368;")
        self.result.setWordWrap(True)
        foot.addWidget(self.result, 1)

        self.btn_view = QPushButton("查看")
        self.btn_view.setVisible(False)
        self.btn_view.clicked.connect(lambda: self.view_clicked.emit(self))
        foot.addWidget(self.btn_view)

        self.btn_run = QPushButton("运行")
        self.btn_run.setStyleSheet(
            "QPushButton { background:#1a73e8; color:#ffffff; border:none;"
            " border-radius:5px; font-weight:600; padding:6px 16px; }"
            "QPushButton:hover { background:#4285f4; }"
            "QPushButton:disabled { background:#dadce0; color:#ffffff; }")
        self.btn_run.clicked.connect(lambda: self.run_clicked.emit(self))
        foot.addWidget(self.btn_run)

        root.addLayout(foot)

    # ---------------- 状态 ----------------

    def set_state(self, state, text=None):
        self.state = state
        color, desc = STATES.get(state, STATES["idle"])
        self.lamp.setStyleSheet("color: %s;" % color)
        self.state_text.setText(text if text is not None else desc)

    def set_result(self, text, allow_view=False):
        self.result.setText(text or "—")
        self.btn_view.setVisible(bool(allow_view))

    def set_busy(self, busy):
        self.btn_run.setEnabled(not busy)
        self.btn_view.setEnabled(not busy)

    # ---------------- 参数控件工厂 ----------------

    def field(self, form, key, label, kind="str", default=None, **kw):
        """创建并登记一个参数控件。kind: str|int|float|bool|choice|path

        ⭐ `tip="…"`（用户 2026-10-04 ✓ 第 3 条）：提示**挂在字段标题上** ✓ —— 传它就不用
          再写 `self.widgets[key][0].setToolTip(...)` 了（挂控件上的写法见规范 §6 ✗）。
        """
        # ⭐⭐ 记下「这个 key 是在**哪个表单**里建的」✓ —— `tip()`（单独一句那种 ✗）得
        #   靠它才找得到这一行的标题：参数搬进「段」自己那个 `QFormLayout` 之后，
        #   `self.form.labelForField(w)` 返回 None ✗（实测：`per_mob` 搬进怪物段 ⇒ 三条用例红 ✓）。
        #   ⚠ 不用 `QFormLayout` 的 buddy 反查 ✗ —— 实测那行 `addRow("标题", w)` 并**没**把
        #   buddy 指到控件上 ✗（兜底不生效 ✓）。这里存一份是最直白的 ✓。
        _forms = getattr(self, "_field_forms", None)
        if _forms is None:
            _forms = self._field_forms = {}
        _forms[key] = form

        def _finish(target):
            """收尾：把 `tip=` 挂到这一行的**标题**上（`QFormLayout.labelForField` ✓
            控件和行布局都吃 ✓ 见 `set_row_visible` 那处同样的用法 ✓）。"""
            if kw.get("tip"):
                field_tip(form.labelForField(target), kw["tip"])
            return target

        if kind == "int":
            w = NoWheelSpinBox()
            w.setRange(int(kw.get("minimum", 0)), int(kw.get("maximum", 10 ** 9)))
            w.setValue(int(default or 0))
        elif kind == "float":
            w = NoWheelDoubleSpinBox()
            w.setDecimals(int(kw.get("decimals", 3)))
            w.setRange(float(kw.get("minimum", -1e9)), float(kw.get("maximum", 1e9)))
            w.setSingleStep(float(kw.get("step", 0.01)))
            w.setValue(float(default if default is not None else 0.0))
        elif kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
        elif kind == "choice":
            w = NoWheelComboBox()
            w.addItems([str(c) for c in kw.get("choices", [])])
            if default is not None:
                i = w.findText(str(default))
                if i >= 0:
                    w.setCurrentIndex(i)
        elif kind == "combo":
            # 选项由调用方动态填充（如地图列表），value() 取的是 userData 而不是显示文本
            w = NoWheelComboBox()
            w.setMaxVisibleItems(int(kw.get("max_visible", 20)))
            form.addRow(label, w)
            self.widgets[key] = (w, kind)
            return _finish(w)
        elif kind == "path":
            w = QLineEdit(str(default or ""))
            w.setReadOnly(True)
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(w, 1)
            btn = QPushButton("浏览")
            btn.setFixedWidth(52)
            btn.setStyleSheet("padding: 2px 6px;")
            # mode="dir" 时选目录而不是文件
            # ⭐ `on_pick`（用户 2026-09-29 ✓）：选完文件后回调一次 —— 采集卡片用它
            #   算「预估多少帧」（见 `CaptureCard._on_file_picked` ✓）。
            btn.clicked.connect(lambda: self._pick_path(
                key, w, kw.get("filter", ""), kw.get("mode", "file"),
                on_pick=kw.get("on_pick")))
            row.addWidget(btn)
            form.addRow(label, row)
            self.widgets[key] = (w, kind)
            self._extra[key] = btn
            return _finish(row)      # 行布局也能 `labelForField` ✓
        else:
            w = QLineEdit(str(default or ""))
        form.addRow(label, w)
        self.widgets[key] = (w, kind)
        return _finish(w)

    def tip(self, key, text):
        """把提示挂到**这一行的字段标题**上（用户 2026-10-04 ✓ 第 3 条）✓。

        ⚠ 为什么要有它：老写法是 `self.widgets[key][0].setToolTip(...)`（**挂在控件上** ✗ 见
          规范 §6 ✓）。迁移时把
              `self.widgets["x"][0].setToolTip(`  ⇒  `self.tip("x", `
          一行改一处即可（**不用**去动上面 `self.field(...)` 那一串长参数 —— 手改容易打错 ✗）。
        ⚠ 搬完**别两处都留** ✗：控件上和标题上各一份 ⇒ 鼠标一指弹两个框 ✓ 比不搬还糟 ✓。
        """
        holder = (getattr(self, "widgets", None) or {}).get(key)
        w = holder[0] if holder else None
        if w is None:
            return
        form = getattr(self, "form", None)
        # ⭐⭐ 2026-10-05：这一行的表单由 `field()` 记下来了 ✓ —— 参数**搬进「段」之后**
        #   控件就不在主表单里 ✗ ⇒ 老写法 `self.form.labelForField(w)` 返回 None
        #   ⇒ `field_tip(None, …)` = **一个提示都没有** ✗（实测：`per_mob` 搬进怪物段就红 ✓）。
        #   ⚠ 别想用 `QLabel.buddy()` 反查兜底 ✗ —— 实测 `QFormLayout.addRow("标题", w)`
        #   **没有**把 buddy 指到控件上 ✗（那条路走不通 ✓ 别绕回去了 ✓）。
        _form = (getattr(self, "_field_forms", None) or {}).get(key) or form
        field_tip(_form.labelForField(w) if _form is not None else None, text)

    def set_row_visible(self, key, on):
        """隐藏/显示一行（label + 控件 + 可能的额外按钮）。

        给「按来源切换参数」这类场景用 —— 比 setEnabled 更彻底，
        不相关的参数直接藏掉，界面干净。
        """
        w, _kind = self.widgets.get(key, (None, None))
        if w is None:
            return
        w.setVisible(bool(on))
        lbl = self.form.labelForField(w)
        if lbl is not None:
            lbl.setVisible(bool(on))
        extra = self._extra.get(key)
        if extra is not None:
            extra.setVisible(bool(on))

    def _pick_path(self, key, line, filt, mode="file", on_pick=None):
        start = line.text() or ""
        if mode == "dir":
            path = QFileDialog.getExistingDirectory(self.window(), "选择目录", start)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self.window(), "选择文件", start,
                filt or "视频文件 (*.mkv *.mp4 *.avi);;所有文件 (*)")
        if path:
            line.setText(path)
            # ⭐ 选完就回调（`on_pick(path)` ✓）—— 采集卡片靠它显示「预估多少帧」✓。
            # ⚠ **取消选择不进这里**（`if path:` ✓）⇒ 不会把上一次的预估刷成空 ✗。
            # ⚠ 回调里别做重活（它在 UI 线程 ✓ 见 `CaptureCard._on_file_picked` 的说明 ✓）。
            if callable(on_pick):
                try:
                    on_pick(path)
                except Exception:
                    pass

    def value(self, key):
        w, kind = self.widgets[key]
        if kind == "int":
            return w.value()
        if kind == "float":
            return w.value()
        if kind == "bool":
            return w.isChecked()
        if kind == "choice":
            return w.currentText()
        if kind == "combo":
            return w.currentData()      # 显示文本给用户看，真正取值用 userData
        return w.text().strip()

    def set_value(self, key, val):
        if key not in self.widgets or val is None:
            return
        w, kind = self.widgets[key]
        if kind == "int":
            w.setValue(int(val))
        elif kind == "float":
            w.setValue(float(val))
        elif kind == "bool":
            w.setChecked(bool(val))
        elif kind == "choice":
            i = w.findText(str(val))
            if i >= 0:
                w.setCurrentIndex(i)
        elif kind == "combo":
            i = w.findData(val)
            if i >= 0:
                w.setCurrentIndex(i)
        else:
            w.setText(str(val))

    def values(self, keys):
        return {k: self.value(k) for k in keys if k in self.widgets}

    # ---------------- 与项目交互 ----------------

    def bind(self, project):
        """切换项目时调用：回填参数 + 刷新产物摘要。"""
        self.project = project

        if project is None:
            self.set_state("idle", STATES["idle"][1])
            self.set_result("—")
            self.set_busy(True)
            return

        self.set_busy(False)
        try:
            self.load_from_project(project)
        except Exception as e:
            self.set_result("参数回填失败: %s" % e)

        # force：换了项目，卡片上残留的 running 已经不属于当前项目了
        self.refresh(force=True)

    def refresh(self, force=False):
        """刷新产物摘要 + 按产物推断状态。

        **为什么状态要"推断"而不是"记住"**：
            以前 bind() 里无条件 set_state("idle")，切换项目时完成过的
            绿点全被抹成灰点 —— 但"这一步做到哪了"本来就是个能从产物
            读出来的客观事实。只要产物还在，绿灯就该自己亮回来。

        force=True 用于切换项目：必须重算，哪怕卡片还标着 running。
        """
        if self.project is None:
            return

        self.set_result(self.summarize(self.project))

        if self.state == "running" and not force:
            return                      # 正在跑，别让推断盖掉

        got = self.detect_state(self.project)
        if got is not None:
            self.set_state(*got)

    def _running_guard(self):
        return self.state == "running"

    # ---------------- 子类接口 ----------------

    def build_params(self, form):
        pass

    def load_from_project(self, project):
        """从项目配置回填界面控件。"""

    def sync(self, project):
        """把界面上的参数写回项目配置（点「运行」时自动调用）。"""

    def summarize(self, project):
        return "—"

    def detect_state(self, project):
        """按已有产物推断这一步的状态，返回 (state, 说明文字)。

        未实现时返回 None，表示"认不出来" —— 保持现状，别乱改灯，
        否则会把刚跑出来的 done 覆盖成 idle。

        state 取值见模块顶部的 STATES：idle / running / done / warn / fail。
        """
        return None

    def check_deps(self, project):
        """返回 (True, "") 或 (False, "缺少 XXX，请先完成 YY")。"""
        return True, ""

    def make_task(self, project):
        """返回 (fn, params)。None 表示本步骤暂未实现。"""
        return None
