"""部署台（A 机）设置弹窗：两个页签，交互与工作台那份（gui/settings_dialog.py）一致。

| 页签 | 装什么 | 落在哪 |
|---|---|---|
| **界面** | 界面字号 | `config/ui.yaml` —— **与工作台共用一份** |
| **日志** | 右下角「运行日志」保留多少行 | `config/deploy.json` 的 `log.max_lines`（A 机本地） |

**为什么字号不另存一份**：`gui/theme.py` 本来就是共享的，部署台启动时已经在用它
（`deploy/app.py` 的 `main()`）。另存一份必然出现"工作台调大了、部署台没动"这种怪事 ——
而这两个界面常常是**同一双眼睛**在轮着看。

**为什么不直接收 `DeployWindow`**：那个窗口一构造就会建五张服务卡片、起两个定时器、
跑一次环境自检 —— 弹窗只该要它真正用到的东西。所以这里收 `cfg`（活的配置 dict，就地改）
和 `on_saved`（回调：落盘与界面同步交回给窗口）。附带的好处是它**能被自检直接构造**，
不必把整个部署台界面拉起来（`tools/selftest_deploy.py` 就是这么用的）。

**字号改动立即生效、不用重启**：底层是 `QApplication.setFont()`，所有没写死字号的控件
都会跟随（见 `gui/theme.py` 的说明）。但**只有点确定才写配置** —— 拖动过程中只更新
预览，取消就等于没动过。

代码顺序 = 页签顺序 = 视觉顺序（docs/UI规范.md §4）。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QFrame, QHBoxLayout,
                             QLabel, QScrollArea, QTabWidget, QVBoxLayout,
                             QWidget)

from gui import theme
# NoWheel* 必须模块级导入：控件在 __init__ 里建，方法里的懒导入到不了那儿。
# （详见 docs/UI规范.md：滚轮不许改参数）
from gui.widgets import NoWheelSlider, NoWheelSpinBox

#: 页签名（顺序 = 显示顺序）。
TAB_NAMES = ("界面", "日志")

#: 日志保留行数的范围与默认值。**默认值必须与 deploy/config.py 的 log.max_lines 一致**
#: （那里是"允许有哪些键"的唯一定义，这里是"界面能填多少"）。
LOG_MIN, LOG_MAX, LOG_DEFAULT = 200, 20000, 4000


class DeploySettingsDialog(QDialog):
    def __init__(self, cfg, on_saved=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumSize(430, 320)

        #: 活的配置 dict（就地改；落盘由 on_saved 回调交回窗口统一走）
        self.cfg = cfg
        self._on_saved = on_saved
        # 字号改动的对比基准。**必须在建控件之前取** —— 控件 setValue 时若已连上
        # valueChanged，_on_changed 会读它（和 gui/settings_dialog.py 同一个理由）。
        self._orig = theme.load_size()

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 10)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._page_appearance(), TAB_NAMES[0])
        self.tabs.addTab(self._page_log(), TAB_NAMES[1])
        self.tabs.setStyleSheet(theme.TAB_QSS)
        root.addWidget(self.tabs, 1)

        # ---- 按钮（在页签之外：无论在哪一页都能点确定）----
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("确定")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        root.addWidget(box)

        self._on_changed(self.slider.value())

    # ---------------- 页面骨架 ----------------

    def _page(self):
        """建一个可滚动的页面，返回 (页 widget, 往里面加东西的布局)。

        套一层 QScrollArea：字号调到最大时内容会明显变高，不套就可能超出小屏
        （docs/UI规范.md §4：长面板放进 QScrollArea）。
        """
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)      # 页签已经有边框了，别套两层
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(8)
        lay.setContentsMargins(10, 10, 10, 10)
        area.setWidget(inner)
        outer.addWidget(area)
        return page, lay

    @staticmethod
    def _head(lay, title, note=""):
        """一个页内小标题（+ 可选说明），返回标签供测试/复用。"""
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(lbl)
        if note:
            sub = QLabel(note)
            sub.setStyleSheet("color: #5f6368;")
            sub.setWordWrap(True)
            lay.addWidget(sub)
        return lbl

    # ---------------- 页签 1：界面 ----------------

    def _page_appearance(self):
        """外观：只有字号。部署台不画检测框，所以工作台那半页"可视化"这里没有。"""
        page, lay = self._page()

        self._head(lay, "界面字号",
                   "两个界面**共用 config/ui.yaml 这一份字号** —— 这里改了，工作台"
                   "（B 机）那边也是这个字号，反之亦然。\n"
                   "点确定后立即生效，不用重启。")

        head = QHBoxLayout()
        head.addWidget(QLabel("字号"))
        head.addStretch(1)
        self.lbl_val = QLabel()
        self.lbl_val.setStyleSheet("color: #5f6368;")
        head.addWidget(self.lbl_val)
        lay.addLayout(head)

        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setRange(theme.MIN_SIZE, theme.MAX_SIZE)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(1)
        self.slider.setValue(self._orig)
        self.slider.valueChanged.connect(self._on_changed)
        lay.addWidget(self.slider)

        scale = QHBoxLayout()
        scale.addWidget(QLabel("小"))
        scale.addStretch(1)
        scale.addWidget(QLabel("大"))
        lay.addLayout(scale)

        self.lbl_preview = QLabel("预览：屏幕推流 · h264_nvenc · 1366×768 @ 60fps")
        self.lbl_preview.setStyleSheet(
            "background: #ffffff; border: 1px solid #dadce0;"
            " border-radius: 4px; padding: 8px; color: #202124;")
        lay.addWidget(self.lbl_preview)

        self.lbl_note = QLabel()
        self.lbl_note.setStyleSheet("color: #80868b;")
        self.lbl_note.setWordWrap(True)
        lay.addWidget(self.lbl_note)

        lay.addStretch(1)
        return page

    # ---------------- 页签 2：日志 ----------------

    def _page_log(self):
        """日志：只影响界面显示，不影响正在跑的服务、也不影响落盘的 txt。"""
        page, lay = self._page()

        self._head(lay, "界面日志",
                   "右下角「运行日志」面板保留多少行历史。\n"
                   "调大：往前翻能看到更多（A 机的 ffmpeg 报错常常是隔一阵才复现，"
                   "历史长一点才抓得到）。调小：界面更省、长时间挂着也不会越堆越大。\n"
                   "「清空」「保存…」两个按钮不受影响。")

        self.sp_lines = NoWheelSpinBox()
        self.sp_lines.setRange(LOG_MIN, LOG_MAX)
        self.sp_lines.setSingleStep(500)
        self.sp_lines.setSuffix(" 行")
        self.sp_lines.setValue(int((self.cfg.get("log") or {}).get("max_lines")
                                   or LOG_DEFAULT))
        self.sp_lines.setToolTip(
            "**调小会立刻丢掉较旧的行**（保留最近的这么多行），不是等下次启动才生效。")
        row = QHBoxLayout()
        row.addWidget(QLabel("保留行数"))
        row.addWidget(self.sp_lines)
        row.addStretch(1)
        lay.addLayout(row)

        lay.addStretch(1)
        return page

    # ---------------- 字号 ----------------

    def _on_changed(self, v):
        self.lbl_val.setText("%d px" % v)
        self.lbl_preview.setStyleSheet(
            "background: #ffffff; border: 1px solid #dadce0;"
            " border-radius: 4px; padding: 8px; color: #202124;"
            " font-size: %dpx;" % v)
        if v < self._orig:
            self.lbl_note.setText("比当前小 %d px。" % (self._orig - v))
        elif v > self._orig:
            self.lbl_note.setText("比当前大 %d px。" % (v - self._orig))
        else:
            self.lbl_note.setText("和当前一样。")

    # ---------------- 确定 ----------------

    def _accept(self):
        v = self.slider.value()
        if v != self._orig:
            theme.save_size(v)
            from PyQt5.QtWidgets import QApplication
            theme.apply(QApplication.instance(), v)     # 立即生效，不用重启
        self.cfg["log"]["max_lines"] = int(self.sp_lines.value())
        if self._on_saved is not None:
            self._on_saved()
        self.accept()
