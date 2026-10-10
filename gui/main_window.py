"""主窗口。

布局：
    ┌──────────────────────────────────────────────────────┐
    │ [项目 ▾] [新建] [打开] [刷新]            ● 就绪        │  工具栏
    ├──────────────────────────┬───────────────────────────┤
    │                          │  ┌─① 地图 / 怪种 ─── ⚪ ─┐ │
    │      主视区               │  └───────────────────────┘ │
    │  （图片 / 日志 / 曲线）    │  ┌─② 采集 ───────── ⚪ ─┐ │
    │                          │  └───────────────────────┘ │
    │                          │  ... 共 8 张卡片           │
    ├──────────────────────────┴───────────────────────────┤
    │ 日志                                                  │
    ├──────────────────────────────────────────────────────┤
    │ [进度条]                                    [取消]     │
    └──────────────────────────────────────────────────────┘
"""

import html
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QFileDialog, QHBoxLayout, QInputDialog,
                             QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                             QProgressBar, QPushButton, QScrollArea, QShortcut,
                             QSizePolicy, QSplitter, QTabWidget,
                             QToolBar, QVBoxLayout, QWidget)

from gui import theme
from gui.export_dialog import ExportDialog
from gui.live_panel import LivePanel
from gui.player_panel import PlayerPanel
from gui.route_panel import RoutePanel
from gui.settings_dialog import SettingsDialog
from gui.widgets import NoWheelComboBox, scroll_page
from gui.project import Project, last_opened, remember_open, sanitize
from gui.review import ReviewPanel
from gui.steps import ALL_CARDS, CIRCLED
from gui.verify_viewer import VerifyViewer
from gui.worker import TaskThread, safe_slot

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"

# 应用名（也是窗口标题的基础部分）。放在这里而不是 app.py：窗口标题要拼上
# 项目名（见 MainWindow._update_title），而 QApplication 的 applicationName
# 也用同一个字符串 —— 定义两处早晚会不一致。
APP_NAME = "playerSimu 数据集工作台"

QSS = """
/* ===== 基础 ===== */
QMainWindow, QWidget#Central { background: #f0f2f5; }
QWidget { color: #202124; }

/* ===== 卡片 ===== */
QFrame#StepCard {
    background: #ffffff;
    border: 1px solid #e2e5ea;
    border-radius: 8px;
}
QFrame#StepCard:hover { border-color: #c9cdd4; }

/* ===== 分组框 ===== */
QGroupBox {
    background: #ffffff;
    border: 1px solid #e2e5ea;
    border-radius: 8px;
    margin-top: 14px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: #5f6368;
}

/* 「操控」组（gui/player_panel.py）：它在**置顶常驻区**（滚动区外面），
   和下面能滚的参数区**都是白底** ⇒ 扫一眼分不清哪块不跟着滚（用户 2026-09-26
   要求"加个背景色区分一下"）。只按 objectName 命中这一组：别的分组框、以及
   本组里嵌套的「前往平台」都照旧白底 ✓（那一块白底正好读成"卡在操控面板里"）。
   取色不新造：`#e8f0fe` 是调色板里既有的"选中/强调底"（见上面 QComboBox
   下拉的 selection 色），配 Tab 选中同一个蓝 `#1a73e8` 当标题色 ✓。 */
QGroupBox#CtrlGroup { background: #e8f0fe; }
QGroupBox#CtrlGroup::title { color: #1a73e8; }

/* ===== 按钮 ===== */
QPushButton {
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 5px;
    padding: 4px 10px;
    color: #202124;
}
QPushButton:hover { background: #f8f9fa; border-color: #c9cdd4; }
QPushButton:pressed { background: #f1f3f4; }
QPushButton:disabled { background: #f8f9fa; color: #9aa0a6; border-color: #e2e5ea; }

/* ===== 输入框 ===== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 20px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #4285f4;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background: #f8f9fa; color: #9aa0a6;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #dadce0;
    selection-background-color: #e8f0fe; selection-color: #202124;
}

/* ===== Tab ===== */
QTabWidget::pane {
    border: 1px solid #e2e5ea;
    border-radius: 6px;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background: transparent;
    padding: 8px 18px;
    color: #5f6368;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
}
QTabBar::tab:selected {
    color: #1a73e8;
    border-bottom: 2px solid #1a73e8;
    font-weight: 600;
}
QTabBar::tab:hover:!selected { color: #202124; }

/* ===== 滚动条 ===== */
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #c9cdd4; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #a8adb6; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #c9cdd4; border-radius: 5px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #a8adb6; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ===== 进度条 ===== */
QProgressBar {
    background: #e2e5ea;
    border: none;
    border-radius: 5px;
    height: 14px;
    text-align: center;
    color: #202124;
}
QProgressBar::chunk { background: #4285f4; border-radius: 5px; }

/* ===== 滑块 ===== */
QSlider::groove:horizontal {
    background: #e2e5ea; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #4285f4; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #1a73e8; }

/* ===== 日志 ===== */
QPlainTextEdit#Log {
    background: #202124; color: #e8eaed;
    border: 1px solid #3c4043; border-radius: 6px;
    font-family: Consolas, monospace;
}

/* ===== 提示 ===== */
QToolTip {
    background: #202124; color: #e8eaed;
    border: 1px solid #3c4043; padding: 4px 8px;
    border-radius: 4px;
}
"""


class InfoPage(QWidget):
    """纯文字详情页。

    内容可以更新，而不是重建控件 —— 每次「查看」都往主视区塞一个新页，越用越乱。
    ⚠ 主视区 2026-09-27 起是**页签容器**（见 `_build_viewer`）⇒ 这里不再有"页号"这回事：
      以前那几个固定页号常量（`PAGE_REVIEW` / `PAGE_VERIFY` / `PAGE_LIVE` / `PAGE_PLAYER`）
      **全部删掉** ✗ —— 页号会随"开了几个 / 关了几个页签"变化，拿固定页号去
      `setCurrentIndex` 迟早错位。现在一律用**控件本身**说话：
      `open_view(widget, 标题)` / `close_view(widget)` ✓。
    """

    def __init__(self, title, body=""):
        super().__init__()

        # ⚠ 页面**要能滚**（`gui.widgets.scroll_page` = 滚动区**唯一实现**，2026-09-27 收口）：
        #   步骤详情页的正文可能很长（`setWordWrap` 之后占很多行），以前没有滚动区 ⇒
        #   窗口一矮就**直接被裁掉**、还滚不到 ✗（起始页那几行短文案看不出来）。
        lay = scroll_page(self, margins=(24, 20, 24, 20), spacing=6)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: 600; color: #202124;")

        self.lbl_body = QLabel(body)
        self.lbl_body.setWordWrap(True)
        self.lbl_body.setStyleSheet("color: #5f6368;")
        self.lbl_body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_body.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_body)
        lay.addStretch(1)

    def set_body(self, text):
        self.lbl_body.setText(text)


class PlayerMatchPage(QWidget):
    """角色匹配验证结果页：画面 + 说明文字。

    匹配验证的结果直接显示在主视区，不在角色模板页里再开一个画面窗口。

    ⚠ **有意不放滚动区**（§4「有意为之」标注）：内容就三样（标题 / 一段说明 / 画面），
    画面必须常驻占满 ⇒ 整页滚动没有收益 ✗（真要放长文字时再给「说明」那截单独接滚动 ✓）。
    """

    def __init__(self):
        super().__init__()
        self._pm = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(6)

        self.lbl_title = QLabel("角色匹配验证")
        self.lbl_title.setStyleSheet("font-weight: 600; color: #202124;")
        lay.addWidget(self.lbl_title)

        self.lbl_text = QLabel("（点「角色模板 → 匹配验证」后这里显示结果）")
        self.lbl_text.setStyleSheet("color:#5f6368;")
        self.lbl_text.setWordWrap(True)
        lay.addWidget(self.lbl_text)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumHeight(300)
        self.view.setStyleSheet(
            "background:#202124; color:#9aa0a6; border:1px solid #dadce0;")
        lay.addWidget(self.view, 1)

    def set_result(self, pm, text):
        self.lbl_text.setText(text)
        self._pm = pm
        if pm is not None:
            self._apply()
        else:
            self.view.setText("（无结果画面）")

    def _apply(self):
        if self._pm is not None:
            self.view.setPixmap(self._pm.scaled(
                self.view.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._apply()


class MainWindow(QMainWindow):
    LOG_COLORS = {"info": "#e8eaed", "ok": "#81c995",
                  "warn": "#fdd663", "error": "#f28b82"}

    def __init__(self):
        super().__init__()
        self.resize(1460, 920)

        self.project = None
        self._update_title()        # 还没有项目 → 只有基础标题
        self.task = None
        self.current_card = None
        #: 卡片之外的**小任务**跑完时的收尾回调（`run_export` ✓ 见那段的说明 ✓）。
        self._mini_done = None
        self.cards = []

        self._build()
        self.setStyleSheet(QSS)
        self._reload_projects()

        if not PROJECTS_DIR.exists():
            PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

        self.log("工作台已就绪。先「新建」或「打开」一个项目。", "ok")
        self.log("项目根目录: %s" % PROJECTS_DIR)

        # ⭐⭐ **启动就把「上次那个项目」打开**（用户 2026-10-04 ✓ 现场原话："决策参数页签→战斗参数→
        #   追击起跳需要的冲刺时间，没有保存数据，每次重开 gui 都要重新填"）。
        #   根因**不是**"没存" ✗ —— `chase_jump_dash_ms` 在 `DecisionSettings` 的 `to_dict` /
        #   `from_dict` 里都有 ✓（实测往返 137 也对 ✓、项目文件里也真存着 ✓）；
        #   而是**启动时没有项目** ✗：决策参数**只按项目存**（保存钩子见 `_bind_decision_params` ✓）
        #   ⇒ 没项目时 `settings.save()` **一个字节都不写** ✗ ⇒ 人在"还没开项目"时改的参数，
        #   一重开就回默认 ✓；而界面显示的偏偏是「最近打开那个项目」那份 ✗
        #   ⇒ 看着特别像"保存没生效" ✓（用户那句话就是这么来的 ✓）。
        #   ⇒ 启动直接开上次那个：改动从此有归属 ✓、标题里也看得见改的是谁 ✓。
        #   ⚠ 没有项目时下面那句"绑一次决策参数"照样要跑 ✓ —— 就是它按"最近项目那份"回填控件 ✓
        #     （连全局热键都吃它，不然会用默认 F11 ✗）。
        if not self._open_last_project_at_startup():
            self._bind_cards()

        # 全局热键：任何窗口聚焦时都能开关自动打怪。键 = 决策参数里
        # 「开关自动」的映射（默认 F11），改了映射会动态重新注册。
        self._hotkey_id = 1
        self._auto_shortcut = None   # 全局热键注册失败时的窗口内快捷键降级
        #: 现在哪一层在生效：`hotkey`（注册制 ✓）/ `hook`（键盘钩子 ✓）/
        #: `window`（只在本窗口 ✓）/ `none`（键无效 ✓）—— 见 `_register_auto_hotkey` ✓
        self._hotkey_how = ""
        self._register_auto_hotkey()

        # ⭐⭐ **窗口收尾：提示搬到字段标题 + 标签可复制**（用户 2026-10-04 ✓ 第 3、4 条）。
        #   ⚠⚠ 页签面板**不走** `theme.bind_window_state`（那是弹窗专用的收尾 ✗）—— 用户反馈
        #   "实测没有实现提示挂字段标题……决策参数页签→输出行为CD"就是这么来的：
        #   面板这条路上**根本没人调** ✗ ⇒ 现在主窗口建完补一次 ✓（把整棵树都扫到 ✓：
        #   决策参数 / 路线识别 / 模型训练 / 挂机保护 / 起始页 ✓），后开的页签走 `open_view` ✓。
        try:
            _tips, _lbls = theme.finish_window(self)
            self.log("窗口收尾：提示搬到字段标题 %d 处 · 标签可复制 %d 个" % (_tips, _lbls))
        except Exception as e:                        # noqa: BLE001 —— 收尾失败不该拖垮启动 ✓
            self.log("窗口收尾（提示挂标题 / 可复制）没跑成：%s" % e, "warn")

    # ══════════════════════════════════════════════════
    # 构建界面
    # ══════════════════════════════════════════════════
    def _build(self):
        central = QWidget()
        central.setObjectName("Central")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addWidget(self._build_toolbar())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_viewer())
        split.addWidget(self._build_right_tabs())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([900, 520])
        split.setCollapsible(0, False)
        split.setCollapsible(1, False)
        root.addWidget(split, 1)

        root.addWidget(self._build_log(), 0)
        root.addLayout(self._build_statusbar())

    def _build_toolbar(self):
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(QLabel("项目"))

        self.cmb_project = NoWheelComboBox()
        self.cmb_project.setMinimumWidth(300)
        self.cmb_project.currentIndexChanged.connect(self._on_project_changed)
        lay.addWidget(self.cmb_project)

        for text, slot in (("新建", self._new_project),
                           ("打开…", self._open_project_dialog),
                           ("刷新", self._reload_projects)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            lay.addWidget(b)

        sep = QLabel("│")
        sep.setStyleSheet("color: #dadce0;")
        lay.addWidget(sep)

        btn_wz = QPushButton("WZ 导出…")
        btn_wz.setToolTip("解析 Map.wz 生成地图清单 —— ① 地图卡片的下拉列表从这里来")
        btn_wz.clicked.connect(self._on_export)
        lay.addWidget(btn_wz)

        sep3 = QLabel("│")
        sep3.setStyleSheet("color: #dadce0;")
        lay.addWidget(sep3)

        # ⛔ **顶栏的「实时」按钮已移除**（用户 2026-09-28 要求 ✓ 原话："做好后把最顶部的
        #   实时按钮移除"）—— 因为「实时」现在是**常驻页签**（关不掉、且页签栏上一直看得见 ✓）
        #   ⇒ 顶栏那个按钮就是**重复入口**了 ✗（本文件反复强调"唯一入口"✓ 少一个旁路更好 ✓）。
        # ⚠ `self._show_live` 方法**保留**（`open_view(self.live_panel, "实时")` 还用它 ✓），
        #   只是不再从这个按钮进来 ✓。

        sep4 = QLabel("│")
        sep4.setStyleSheet("color: #dadce0;")
        lay.addWidget(sep4)

        btn_set = QPushButton("设置")
        btn_set.setToolTip("调整界面字号（立即生效，无需重启）")
        btn_set.clicked.connect(self._on_settings)
        lay.addWidget(btn_set)

        lay.addStretch(1)

        self.lbl_status = QLabel("未选择项目")
        self.lbl_status.setStyleSheet("color: #5f6368;")
        lay.addWidget(self.lbl_status)

        return bar

    def _build_viewer(self):
        """主视区 = **页签容器**（2026-09-27 用户要求："主视区也能分页签，只要会占用主视区
        的都应该走打开、关闭页签流程"）。

        为什么从 `QStackedWidget` 换过来：
          · 以前是"一个固定页号一个页面"：占用者**一直挂着** ⇒ "打开 / 关闭"这件事根本
            不存在 ✗，而动态页（步骤详情）一来，页号还得靠人肉维护（老的 `viewer_pages`）
            ⇒ 迟早错位（`InfoPage` 那句注释记的就是这类坑）；
          · 现在：**要占用主视区 ⇒ 调 `open_view(widget, 标题)`**，关掉走 `close_view` ✓
            —— 那是**唯一的开合入口**，谁都不许再直接 `viewer.setCurrentIndex` /
            `addTab` ✗（用例 `t_viewer_has_single_entry` 钉着这条）。
        ⚠ 尺寸策略照旧（`Ignored`）：实时页每帧刷新会带动 sizeHint 变化，传导到
          `QSplitter` 就会挤压右侧卡片 ✗（原来踩过）。
        ⚠ **起始页不参与开合**：它永远是第一个标签、关不掉 —— 它不占用主视区去干活，
          只是"把所有工作页都关掉之后总得有个落脚的地方" ✓（不然主视区会变成空白 ✗）。
        """
        self.viewer = QTabWidget()
        self.viewer.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.viewer.setTabsClosable(True)     # 页签可关（用户要的"关闭流程"）
        self.viewer.setMovable(True)          # 能拖着重排（纯界面，不牵动任何逻辑 ✓）
        self.viewer.setDocumentMode(True)     # 主视区是"工作区"，不是一叠设置卡片
        self.viewer.tabCloseRequested.connect(self._on_view_close)

        self.home_view = self._placeholder(
            "主视区",
            "这里会显示当前步骤的产物：\n\n"
            "  ·   抽帧画面预览（②）\n"
            "  ·   尺度标定对比图（③）\n"
            "  ·   质检台：翻看 / 筛选 / 修正标注（⑤）\n"
            "  ·   训练曲线与日志（⑦）\n"
            "  ·   检测结果图（⑧）\n\n"
            "点右侧卡片下方的「查看」—— 它会在**这里新开一个页签**（页签上的 × 关闭 ✓）。")
        self.viewer.addTab(self.home_view, "起始")
        # ⭐ **常驻页签不显示「×」**（用户 2026-09-28 ✓）—— 「起始」是这里**直接 addTab** 的
        #   （不走 `open_view` ✓）⇒ 得手动刷一次，否则它启动时还带着 × ✗（点它没用、看着别扭 ✓）。
        self._refresh_tab_close_buttons()

        # ---- 四个**常驻工作页**：先不建标签，谁用谁 `open_view` ✓ ----
        # 控件本身**不销毁**：关掉页签只是"从主视区摘下来"，下次打开还是原来那个对象
        #（标定/画面/滚动位置都还在 ✓）。
        self.review = ReviewPanel()           # 质检台（⑤「查看」）
        self.verify_viewer = VerifyViewer()   # 验证结果浏览器（⑧「查看」）
        self.live_panel = LivePanel()         # 实时预览（工具栏「实时」）
        # ⭐⭐ **三个常驻页签从"启动"就都在**（用户 2026-09-28 ✓ 原话："起始页签、实时页签、
        #   质检台页签现在是 **3 个常驻页签**"）。
        #   ⚠⚠ 这一步**必须**有：以前「实时」「质检台」是**按需打开**的（「实时」靠**顶栏那个
        #     按钮** ✓、「质检台」靠卡片上的"查看"✓）⇒ 顶栏按钮一移除，「实时」就**没有入口**了
        #     ✗（用户当场就问"**实时页签呢？质检台页签呢？**"✗）⇒ 现在启动就全建出来 ✓
        #     —— **"常驻" = 一直在** ✓，不能只靠"关不掉" ✗。
        #   ⚠ 顺序：`open_view` 会 `setCurrentIndex` ⇒ **最后必须落回「起始」**（默认那一页 ✓）。
        try:
            self._show_live()                   # 建「实时」并 bind 当前项目 ✓
        except Exception:                        # noqa: BLE001 —— 项目还没绑好也**不能**少了这个页签 ✓
            self.open_view(self.live_panel, "实时")
        self.open_view(self.review, "质检台")       # 建「质检台」✓（内容仍由卡片"查看"填 ✓）
        self.open_view(self.home_view, "起始")      # ⭐ 最后切回「起始」✓

        self.player_view = PlayerMatchPage()  # 角色匹配验证结果（角色模板 → 匹配验证）
        self.view_widgets = {}                # card.key -> 步骤详情页控件
        return self.viewer

    def _placeholder(self, title, body):
        return InfoPage(title, body)

    def _build_cards(self):
        host = QWidget()
        # 滚动区**只有一处实现**（`gui.widgets.scroll_page`，2026-09-27 收口）：这里原来是
        # 手写的一份 —— 和「决策参数」页、「路线识别」页、两个设置弹窗各写一版 = 五份重复 ✗
        # ⇒ 改一处忘一处就是"有的页能滚、有的页不能滚"（路线识别页就是这么被漏掉的 ✓）。
        # 右边距 8 是给滚动条留的位置（跟原来一样 ✓）。
        inner_lay = scroll_page(host, margins=(2, 2, 8, 2), spacing=8)

        for cls in ALL_CARDS:
            card = cls()
            card.run_clicked.connect(self._on_run)
            card.view_clicked.connect(self._on_view)
            # ⭐⭐ 卡片**把"项目现在是哪张图"改掉了** ⇒ 交给 `_on_map_changed` 重推跨面板那几件
            #   （用户 2026-10-04 ✓ 现场：从「识别目标」卡的下拉换图后，A 机「当前地图」不变、
            #    小地图还按上一张图取区域 ✗）。`MapCard` 与 `RoutePanel` **信号名故意一致** ✓，
            #   所以这里一条 `hasattr` 就把两个写口都接上 ✓（谁再加一个写口也自动接 ✓）。
            if hasattr(card, "map_changed"):
                card.map_changed.connect(self._on_map_changed)
            self.cards.append(card)
            inner_lay.addWidget(card)

        inner_lay.addStretch(1)
        return host

    def _build_right_tabs(self):
        """右侧：模型训练流程 + 决策参数 + 路线识别 + 挂机保护，四个页签。"""
        tabs = QTabWidget()
        tabs.addTab(self._build_cards(), "模型训练")
        self.player_panel = PlayerPanel()
        self.player_panel.verify_started.connect(self._on_player_verify_started)
        self.player_panel.verify_result.connect(self._on_player_verify)
        self.player_panel.auto_key_changed.connect(self._on_auto_key_changed)
        # HP/MP 条框选要在实时预览画面上做，给它实时页引用
        self.player_panel.live_panel = self.live_panel
        # 实时推理识别出的血/蓝比例 → 决策参数页可视化
        self.live_panel.potions_ready.connect(self.player_panel._on_potions)
        tabs.addTab(self.player_panel, "决策参数")
        self.route_panel = RoutePanel()
        # 框选小地图、以及「把叠图画到实时画面上」，都要拿实时那个面板
        # （理由同上面 HP/MP 条：框选得在画面上做）。
        self.route_panel.live_panel = self.live_panel
        # 面板自己建的时候还没有 live_panel，叠图挂不上去；这里补一次。
        # 不补的话：上次退出时开关是开着的用户，要先去「路线识别」页点一下
        # 才会看到叠图 —— 而它明明是开着的（"勾了没反应"就长这样）。
        self.route_panel._refresh_overlay()
        # ⭐ 路线识别页**就地换地图**（用户 2026-10-02 的「手动更换」✓）之后，把玩家的
        #   「定点休息」集合下拉重新推一遍 —— 集合按**地图 id** 存，不推就还列着上一张图的
        #   集合（比不列更糟：选得到、但站在那儿永远不成立 ✗）。这一步和 `_bind_cards`
        #   切项目时**同一个动作** ✓（那处见下面 `set_zone_sets`）。
        self.route_panel.map_changed.connect(self._on_map_changed)
        tabs.addTab(self.route_panel, "路线识别")
        # ⭐ 「挂机保护」页（2026-09-29，用户要求：在路线识别右边 ✓）：
        #    「防挂机」新组（触发音效 + 试听 ✓）+ 从「决策参数」**整体搬来**的「防掉线」组 ✓。
        #    ⚠ 防掉线的控件/槽函数仍归 PlayerPanel 所有（信号早连好 ✓）——
        #      build_protection_page 只是把那个组 **re-parent** 到本页 ✓（搬视觉不搬逻辑 ✓）。
        self.protection_panel = self.player_panel.build_protection_page()
        tabs.addTab(self.protection_panel, "挂机保护")
        # 最小宽度兜底：主视区尺寸波动时不把配置区挤没
        tabs.setMinimumWidth(460)
        # 忽略 sizeHint：卡片状态文字更新会改变 sizeHint，传导到 QSplitter
        # 会让右侧宽度跟着抖。固定按 minimumWidth + stretch 分配即可。
        tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        return tabs

    def _build_log(self):
        self.txt_log = QPlainTextEdit()
        self.txt_log.setObjectName("Log")
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(3000)
        self.txt_log.setFixedHeight(150)
        self.txt_log.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        return self.txt_log

    def _build_statusbar(self):
        lay = QHBoxLayout()
        lay.setSpacing(6)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("空闲")
        lay.addWidget(self.progress, 1)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        lay.addWidget(self.btn_cancel)

        b = QPushButton("清空日志")
        b.clicked.connect(lambda: self.txt_log.clear())
        lay.addWidget(b)

        return lay

    # ══════════════════════════════════════════════════
    # 项目管理
    # ══════════════════════════════════════════════════
    def _reload_projects(self):
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        cur = self.cmb_project.currentData()

        self.cmb_project.blockSignals(True)
        self.cmb_project.clear()
        self.cmb_project.addItem("（未选择项目）", None)

        for d in sorted(PROJECTS_DIR.iterdir()):
            if (d / "project.yaml").exists():
                self.cmb_project.addItem(d.name, str(d))

        idx = self.cmb_project.findData(cur) if cur else 0
        self.cmb_project.setCurrentIndex(max(0, idx))
        self.cmb_project.blockSignals(False)

    def _on_project_changed(self, _idx):
        data = self.cmb_project.currentData()
        if not data:
            self.project = None
            self._bind_cards()
            self.lbl_status.setText("未选择项目")
            return
        try:
            self.open_project(Path(data))
        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))

    def _open_last_project_at_startup(self):
        """启动时打开**上次那个项目**（成不成都不抛 ✓）。返回是否真的打开了。

        ⚠ 为什么要有它（用户 2026-10-04 ✓ 现场："追击起跳需要的冲刺时间，没有保存数据，
          每次重开 gui 都要重新填"）：决策参数**只按项目存** ✓ ⇒ **没有项目时
          `settings.save()` 什么都不写** ✗（见 `decision/agent.py` 的 `set_save_hook` ✓）
          ⇒ 人在"还没开项目"时改的参数，一重开就没了 ✓；而界面显示的又是
          「最近打开那个项目」那份 ✗ ⇒ 看着就像"没保存" ✓（其实那个参数在 JSON 里
          存/读两处都写着 ✓ 见 `DecisionSettings.to_dict` / `from_dict` ✓）。
          ⇒ 启动把上次那个项目打开，改动就有归属了 ✓。
        ⚠ 项目打不开（被删 / 改名 / 文件坏了）⇒ **保持"没有项目"** ✓ 只记一条日志 ✗
          别让一个陈旧路径把工作台挡在门外 ✗。
        """
        proj = last_opened()
        if proj is None:
            return False
        try:
            self.open_project(Path(proj.root))     # 它自己会 `_bind_cards()` ✓
            return True
        except Exception as e:                     # noqa: BLE001
            self.project = None
            self.log("⚠ 上次的项目打不开（%s：%s）⇒ 按「没有项目」启动。"
                     % (type(e).__name__, e), "warn")
            return False

    def open_project(self, path):
        self.project = Project.open(path)
        self._clear_pages()
        self._bind_cards()

        snap = self.project.snapshot()
        self.lbl_status.setText("画面 %d  /  自动标注 %d  /  怪种 %d"
                                % (snap["frames"], snap["labels_auto"], snap["mobs"]))
        self.log("已打开项目: %s" % self.project.root.name, "ok")
        self.log("  地图 %s   scale %s   怪种 %s"
                 % (self.project.get("map_id") or "-",
                    self.project.get("scale"),
                    self.project.get("mobs") or "-"))

        idx = self.cmb_project.findData(str(path))
        if idx >= 0 and idx != self.cmb_project.currentIndex():
            self.cmb_project.blockSignals(True)
            self.cmb_project.setCurrentIndex(idx)
            self.cmb_project.blockSignals(False)

        # **最后**才记「最近打开的项目」：上面的 _bind_cards 要用**上一个**最近项目
        # 当播种源（新建项目沿用上一套参数），提前记下的话播种源就变成它自己了。
        remember_open(self.project.root)

    def _update_title(self):
        """窗口标题 = 「项目名-」+ 基础标题。

        为什么不只写固定的应用名：同时开几个项目的窗口时，任务栏/Alt+Tab 里
        一排「playerSimu 数据集工作台」根本分不清哪个是哪个。
        没有项目时保持基础标题（别留个空的「-playerSimu…」）。
        """
        base = APP_NAME
        name = self.project.root.name if self.project is not None else ""
        self.setWindowTitle("%s-%s" % (name, base) if name else base)

    def _bind_decision_params(self, project):
        """把决策参数切到当前项目那一份（**必须在各面板 bind 之前**跑）。

        顺序为什么关键：面板（决策参数页、路线识别页）在 bind 时是从 settings
        读值回填控件的 —— 先换参数再回填，控件才显示当前项目的值。同样地，
        「最近打开的项目」是在**换完之后**才更新的（见 open_project 末尾），
        否则播种源会变成刚打开的这个项目自己。

        三条规则（对应「参数只认项目」这件事）：
          · 有项目且存过 decision 段 —— 用它自己的；
          · 有项目但没存过（新建 / 老项目）—— 用**最近打开的那个项目**那份播种
            并当场固化。键位映射、行为序列、血蓝条这些东西各图通用，让每个新项目
            都重配一遍不合理；固化之后它就有自己的一份，之后互不影响；
          · 没打开项目 —— 读**最近打开的那个项目**那份，**只读**：界面照样能改
            （实时预览立刻生效），但没有归属、不落盘（save() 没有钩子就什么都不写），
            一打开项目就被项目的值覆盖。
        """
        from decision.agent import set_save_hook, settings

        if project is None:
            set_save_hook(None)
            src = last_opened()
            settings.from_dict((src.get("decision") if src is not None else None) or {})
            return

        data = project.get("decision")
        if not isinstance(data, dict) or not data:
            src = last_opened()
            # 最近项目正好就是这个项目（比如它上次打开时被清空了 decision）→ 只能
            # 落到默认值，别拿它自己那份空字典当种子。
            seed = None
            if src is not None and src.root != project.root:
                seed = src.get("decision")
            settings.from_dict(seed or {})
            # 存**整份**（缺的键由 from_dict 的默认值补齐）—— 项目文件一打开就是
            # 完整的一份，手工翻/手工改都看得全，不用去猜哪些键是默认值。
            project.set("decision", settings.to_dict(), save=True)
        else:
            settings.from_dict(data)
        # ⭐⭐ **顺手写回「当前这张图」那份**（用户 2026-10-03 ✓）：「路线识别」页那批配置
        #   改成**按地图 id** 存（`datasets/map/<id>.route.json` ✓ 见 `core/route_cfg.py`），
        #   而 `project.yaml` 里那份**留着当"新图第一次打开时的播种值"**（用户选的 ✓）。
        #   ⚠ 挂在这一处 ⇒ **以后任何人加一个"改完就 `settings.save()`"的控件，都会自动
        #     带上"写回这张图"** ✓（逐个 handler 去加必然漏 ✗ 而漏了就是"改了没存"✗）。
        #   ⚠ 面板可能还没建好（首次构造 / 换项目途中）⇒ `getattr` 兜底 + 吞异常：参数存不
        #     下去是坏事，但不能连界面一起炸 ✗（同 `DecisionSettings.save` 的纪律 ✓）。
        def _save_everywhere(d, p=project):
            p.set("decision", d)
            p.save()
            rp = getattr(self, "route_panel", None)
            if rp is not None:
                try:
                    rp._save_route_cfg()
                except Exception:                   # noqa: BLE001
                    pass
            # ⭐⭐ 多实例（用户 2026-10-09："**必须支持能开 2 个工作台**"✓）：上面两处落盘都是
            #   **读-改-写、只盖本窗口改过的那几处**（`Project.save` / `core.route_cfg.save` ✓）
            #   ⇒ 两个窗口改**不同**参数**互不覆盖** ✓
            #   （原来整份覆盖 ⇒ 现场症状："这边选了平台站桩、那边一存变成 patrol" ✗）。
            #   ⚠ 剩下"**同一个**项被两边都改过"无法两全（后写的赢 ✓）⇒ **必须说出来** ✗，
            #     别让人以为"我改的怎么没了"（记账：`Project._conflicts` ＋ 面板的
            #     `_route_cfg_conflicts` ✓）。⚠ 这段放在**两处都存完之后** ✓（先读的话路线那半还没产出 ✗）。
            _cf = list(getattr(p, "_conflicts", None) or [])
            if rp is not None:
                _cf += list(getattr(rp, "_route_cfg_conflicts", None) or [])
            if _cf:
                _more = "（还有 %d 项）" % (len(_cf) - 6) if len(_cf) > 6 else ""
                self.log("⚠ 另一个工作台也改过这些项 ⇒ 已按**本窗口**的值写：%s%s"
                         % ("、".join(_cf[:6]), _more), "warn")
        set_save_hook(_save_everywhere)

    def _on_map_changed(self, map_id):
        """路线识别页**就地换了地图**（「手动更换」✓）⇒ 把依赖"哪张图"的跨面板东西重推。

        ⚠ 只有**跨面板**的那几件归这里（本面板自己的刷新在 `_on_manual_map_change` 里
          当场做过 ✓ 别重复一套 ✗）：目前就是玩家面板那两个「定点休息」集合下拉 ——
          集合按**地图 id** 存，不重推的话它们还列着上一张图的集合名
          （比"空着"更糟：选得到、但站在那儿永远不成立 ✗）。
        `map_id` 只用来打日志（谁需要它自己从 `route_panel._map_id()` 取 ✓）。
        """
        self.player_panel.set_zone_sets(self.route_panel.zone_sets())
        # ⭐⭐ **把新地图 id 推给实时面板/线程**（用户 2026-10-04 ✓ 现场：换图/换项目后 A 的
        #   「当前地图」不跟着变，而且小地图框选**还按上一张图**取 ⇒ `perf.log` 的 `mmap_note`
        #   一直报「框选区超出当前画面，重框一次」✗ —— 同一个根因 ✓）。
        #   ⚠ 两个写口（「识别目标」卡的下拉 ✓ / 路线识别页的「手动更换」✓）现在都会走到这里 ✓
        #     （卡片那条是 2026-10-04 补的 ✓ 它原来**一个信号都不发** ✗）。
        #   ⚠ 实时线程在跑 ⇒ `set_mmap` 只写几个字段、主回路每拍现读 ⇒ **当场生效** ✓
        #     （不用停/开实时 ✓）；没跑 ⇒ 下一次 `start()` 由参数带进去 ✓ 两条路都对 ✓。
        lp = getattr(self, "live_panel", None)
        if lp is not None and hasattr(lp, "set_mmap"):
            try:
                lp.set_mmap(map_id=(self.route_panel._map_id() or ""))
            except Exception as e:                        # noqa: BLE001 —— 推不动也别崩 ✓
                self.log("新地图 id 没推进实时（%s: %s）" % (type(e).__name__, e), "warn")
        # ⚠ 「前往平台」那一组也是按地图列集合的（控件归 RoutePanel ✓）——它自己会在
        #   `_refresh_goto()` 里重读 ✓（`_on_manual_map_change` 已经调过 ✓），这里不用管 ✗。

    def _bind_cards(self):
        # 先换决策参数，再让各面板回填控件（它们从 settings 读数）
        self._bind_decision_params(self.project)
        for c in self.cards:
            c.bind(self.project)
        self.review.bind(self.project)
        self.live_panel.bind(self.project)
        # HP/MP 条框选结果存在 project.yaml 里，切项目要跟着换
        self.player_panel.bind(self.project)
        # 路线识别面板：换项目要重读**地形图 + 集合下拉**（那些都按地图 id 存）
        self.route_panel.bind(self.project)
        # 「定点休息」的两个集合下拉：集合名属于**地图** ⇒ 名单只能由路线识别面板给
        # （玩家面板不持有地图 id）。以前写好了 `set_zone_sets()` 却**没人调** ⇒
        # 那两个下拉一直只有「（未选）」✗（2026-09-26 补）。
        self.player_panel.set_zone_sets(self.route_panel.zone_sets())
        # ⭐ 「**地区选择**」通用弹窗的工厂（用户 2026-10-05 ✓）：玩家面板要选**地图元素**
        #   （「站桩地点」/「拾取掉落地区」），而弹窗要地形与集合、只有**路线识别页**有
        #   （`make_element_picker` ✓）⇒ 这里推一次（同上面 `set_zone_sets` 那条 ✓）。
        self.player_panel.set_element_picker_factory(self.route_panel.make_element_picker)
        # ⛔ 「编辑战斗区域」2026-10-01 整个搬到「路线识别 → 寻路配置 → 路线规划」✓ ——
        #   只读 foothold 视图的工厂不再跨面板推（那边自己就有 `make_foothold_picker` ✓）。
        # 「前往平台」（选择平台 / 命令前往 / 结束当前寻路）**显示在决策参数页**：
        # 控件还是 RoutePanel 造的那一份（逻辑要在那边 ✓），这里只负责搬位置 ✓。
        self.player_panel.mount_goto(getattr(self.route_panel, "goto_box", None))
        self.lbl_status.setText("未选择项目" if self.project is None else
                                self.lbl_status.text())
        # 标题跟着项目走（切项目 / 新建 / 打开都走这里）
        self._update_title()

    def _new_project(self):
        name, ok = QInputDialog.getText(self, "新建项目", "项目名（建议：地图名或地图ID）")
        if not ok or not name.strip():
            return

        map_id, _ = QInputDialog.getText(
            self, "新建项目", "地图 ID（可留空，稍后①再填）")

        root = PROJECTS_DIR / sanitize(name)
        if root.exists():
            QMessageBox.warning(self, "已存在", "项目目录已存在：\n%s" % root)
            return

        try:
            Project.create(root, name.strip(), map_id.strip())
        except Exception as e:
            QMessageBox.critical(self, "创建失败", str(e))
            return

        self.log("已创建项目: %s" % root, "ok")
        self._reload_projects()
        idx = self.cmb_project.findData(str(root))
        if idx >= 0:
            self.cmb_project.setCurrentIndex(idx)

    def _open_project_dialog(self):
        d = QFileDialog.getExistingDirectory(
            self, "选择项目目录（含 project.yaml）", str(PROJECTS_DIR))
        if not d:
            return
        try:
            self.open_project(Path(d))
        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))

    # ---------------- WZ 导出 ----------------

    def _show_live(self):
        """切到实时预览（没有这个页签就开一个 ✓）。"""
        self.live_panel.bind(self.project)
        self.open_view(self.live_panel, "实时")

    def _on_settings(self):
        """开设置（**非模态**，2026-09-27 用户定）。

        为什么不再用 `exec_()`：改完「浓淡 / 叠图」要对着**实时画面**（或路线识别页）
        看效果 ⇒ 开着设置得能去「实时」页看 —— 模态会把整个工作台锁住，只能
        "关窗 → 看 → 重开设置" ✗（用户就是这么定的；原来那条理由是"要在设置里框选
        小地图"，那个按钮 2026-09-27 已搬回路线识别页 ✓）。
        非模态之后只剩一个实例的问题：**关过窗就重建**（打开时要重读基准值，不然会拿
        旧基准去比"这次改了什么" ✗），还开着就直接抬到前面 ✓。
        """
        dlg = getattr(self, "_settings_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()
        dlg = SettingsDialog(self)
        # 「设置 → 界面」里改了叠图开关 / 浓淡 ⇒ 立刻让路线识别面板重画那一层
        # （那一层是它画的，见 route_panel.apply_overlay_settings ✓）
        dlg.overlay_changed.connect(self.route_panel.apply_overlay_settings)
        # 「质检台蒙版透明度」改了 ⇒ 立刻推给质检台画布（不重取帧 ✓）
        dlg.mask_changed.connect(self.review.canvas.set_mask_alpha)
        self._settings_dlg = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_export(self):
        dlg = ExportDialog(self)
        dlg.exported.connect(self._on_exported)
        dlg.exec_()

    def _on_exported(self):
        """导出完成后刷新地图下拉 —— 否则卡片还在用旧的清单。"""
        for c in self.cards:
            if hasattr(c, "reload_maps"):
                c.reload_maps()
                c.refresh()
        self.log("① 地图卡片的下拉列表已刷新", "ok")

    def _on_player_verify_started(self):
        """点「匹配验证」→ 主视区切到角色匹配页（没有就开一个），先显示进行中。"""
        self.player_view.set_result(None, "连流抓帧 + 匹配中…")
        self.open_view(self.player_view, "角色匹配")

    def _on_player_verify(self, pm, text):
        """匹配验证出结果 → 主视区显示画面 + 说明。"""
        self.player_view.set_result(pm, text)
        self.open_view(self.player_view, "角色匹配")

    # ══════════════════════════════════════════════════
    # 运行任务
    # ══════════════════════════════════════════════════
    def _on_run(self, card):
        if self.project is None:
            QMessageBox.warning(self, "提示", "请先新建或打开一个项目")
            return
        if self.task is not None:
            QMessageBox.information(self, "提示", "已有任务在运行，请等它结束或点「取消」")
            return

        # 1) 参数回写项目 —— 这样界面上的改动不会丢
        try:
            card.sync(self.project)
            self.project.save()
        except Exception as e:
            QMessageBox.warning(self, "参数错误", str(e))
            return

        # 2) 上游依赖检查（比强制顺序灵活，但不会让人迷失）
        ok, why = card.check_deps(self.project)
        if not ok:
            QMessageBox.warning(self, "尚未就绪", why)
            return

        # 3) 组装任务
        try:
            made = card.make_task(self.project)
        except Exception as e:
            QMessageBox.warning(self, "无法运行", str(e))
            return
        if made is None:
            QMessageBox.information(
                self, "尚未实现",
                getattr(card, "pending", "该步骤将在后续版本实现"))
            return

        fn, params = made
        self._start(card, fn, params)

    # ⭐⭐ 卡片之外的**小任务**（用户 2026-10-05 ✓）
    def run_export(self, fn, params, title, on_done=None):
        """在**工作台**上跑一个小任务（读条 + 日志 + 取消 ✓）—— 卡片之外发起的那些 ✓。

        为什么要它（用户 2026-10-05 ✓ 原话："如果没有导出，在添加的带装备的宠物、关闭弹窗后，
        在数据集工作台显式读条导出"）：补导宠物图库这件事**不属于任何一张卡片**（它是在「自动
        标注」卡里加宠物时触发的 ✓），但它必须**看得见** ✓ —— 同一条读条、同一份日志、同样能
        取消 ✓（别做成"点了没反应"或"偷偷在后台跑"✗）。

        ⚠ 与 `_on_run` 的区别：那条是"**跑某张卡片**"，要先做参数回写 / 上游依赖检查 /
          选帧确认 ✗ —— 小任务这三件都不该有 ✓（它是补数据，不是跑步骤 ✓）。
        ⚠ `on_done(ok, summary)` 在任务结算时（`_on_done` 里 ✓）被调 ⇒ 调用方在那儿刷新自己
          （比如把宠物列表重新铺一遍 ⇒ 图标按新图重画 ✓）。

        返回 False = **没起成**（已有任务在跑 ✓）⇒ 调用方自己决定怎么如实提示 ✓ 别静默 ✗。
        """
        if self.task is not None:
            return False
        self._mini_done = on_done
        self._start(None, fn, params, title=title)
        return True

    def _start(self, card, fn, params, title=None):
        """起一个后台任务。`card` 允许传 None ⇒ 卡片之外的**小任务**（见 `run_export` ✓）。"""
        self.current_card = card
        head = title or (("%s %s" % (CIRCLED[card.num - 1], card.title))
                         if card is not None else "任务")
        if card is not None:
            card.set_state("running")
            card.set_result("运行中…")
            card.set_busy(True)

        for c in self.cards:
            c.set_busy(True)

        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("运行中…")
        self.log("── 开始 %s ──" % head, "info")

        self.task = TaskThread(fn, params, self)
        # 全部过 safe_slot —— 槽函数抛异常会让 PyQt5 调 qFatal() 直接 abort()
        self.task.sig_log.connect(safe_slot(self.log))
        self.task.sig_progress.connect(safe_slot(self._on_progress))
        self.task.sig_done.connect(safe_slot(self._on_done))
        # 必须在 finished 里才释放引用 —— 见 _on_task_finished 的说明
        self.task.finished.connect(self._on_task_finished)
        self.task.start()

    def _on_task_finished(self):
        """线程真正结束后才丢引用。

        **不能**在 sig_done 里直接 self.task = None：
        那是个跨线程排队信号，主线程收到它时工作线程的 run() 可能还没返回，
        此时 Python 包装对象被回收，Qt 的 C++ 侧就悬空了 ——
        表现是无征兆闪退（Windows 事件日志里是 Qt5Core.dll + 0xc0000409）。
        """
        t = self.task
        self.task = None
        if t is not None:
            t.deleteLater()

    def _on_progress(self, cur, total, text):
        if total > 0:
            if self.progress.maximum() != total:
                self.progress.setRange(0, total)
            self.progress.setValue(cur)

            # setFormat 把 % 当格式符，所以 text 里的 % 必须先转义。
            # 原来写成 "%s   %p%%" % text —— %p 会被 Python 当格式符，
            # 文本里再带个 % 就直接抛异常。而槽函数抛异常 = abort() = 闪退。
            fmt = (text or "").replace("%", "%%")
            self.progress.setFormat((fmt + "   " if fmt else "") + "%p%")
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat((text or "运行中…").replace("%", "%%"))

    def _on_done(self, ok, summary, result=None):
        card = self.current_card

        if card is not None:
            if ok:
                card.set_state("done", "完成")
            else:
                card.set_state("fail", "失败")

        # 有些步骤的结果要写回项目配置（比如 ③ 标定出的缩放值）。
        # 必须在下面 refresh() 之前做，否则卡片读到的还是旧值。
        if ok and card is not None and hasattr(card, "on_result"):
            try:
                card.on_result(result)
            except Exception as e:
                self.log("结果写回失败: %s: %s" % (type(e).__name__, e), "warn")

        self.progress.setRange(0, 100)
        self.progress.setValue(100 if ok else 0)
        self.progress.setFormat("空闲" if ok else "失败")
        self.btn_cancel.setEnabled(False)

        for c in self.cards:
            c.set_busy(False)
            c.refresh()

        if card is not None:
            if ok:
                self.log("── %s 完成: %s ──" % (card.title, summary or ""), "ok")
            else:
                card.set_result(summary)
                self.log("── %s 失败: %s ──" % (card.title, summary), "error")

        # 详情页跟着更新；④ 完成后质检台要重新载入新出的可视化图
        if card is not None:
            if card.key == "label":
                self.review.bind(self.project)
            elif card.key == "verify":
                self.verify_viewer.bind(self.project)
            else:
                self._rebuild_page(card)

        # ⭐ 小任务（`run_export` ✓）的收尾：**先摘下来再调**（回调里可能又发起别的任务 ✓
        #   连清两遍也不会重复调 ✓）。卡片任务没有这个回调 ⇒ `_mini_done` 是 None ✓ 什么都不做 ✓。
        cb, self._mini_done = self._mini_done, None
        if cb is not None:
            try:
                cb(ok, summary)
            except Exception as e:          # noqa: BLE001 —— 收尾坏了别把结算弄挂 ✗
                self.log("小任务收尾失败: %s: %s" % (type(e).__name__, e), "warn")

        # 注意：这里**不要** self.task = None —— 线程可能还没跑完，
        # 交给 _on_task_finished（由 finished 信号触发）来释放。
        self.current_card = None

        if self.project is not None:
            snap = self.project.snapshot()
            self.lbl_status.setText("画面 %d  /  自动标注 %d  /  怪种 %d"
                                    % (snap["frames"], snap["labels_auto"], snap["mobs"]))

    def _cancel(self):
        if self.task is not None:
            self.task.cancel()
            # 点了要**当场有反应**，否则用户会以为没点上而反复点。任务本身不一定
            # 立刻停 —— 它在各阶段的检查点上看 `canceled()` 收工（`tools/` 里那些 ✓）。
            # ⭐⭐ 用户 2026-10-05 ✓ 原话："能不等他返回吗？" —— **能的那几处已经能了** ✓：
            #   模板加载以前是**整块**动作 ✗（逐张 `imread` 几百张精灵 ✓）⇒ 现在改成
            #   "每读一张看一眼" ✓（`detect_mobs.load_templates` / `PlayerLocator` 的
            #   `should_stop` ✓ 见 SKILL 258 ✓）⇒ 收工从"几十秒"降到"几百毫秒" ✓。
            #   ⚠ 真正拦不住的只剩**第三方库内部**那一下（`import torch` / `YOLO(weights)`
            #     读权重 ✓ —— 我们进不去 ✓）。所以话要说准：**常见情况是马上停**，
            #     只有"载入 YOLO 权重"那一步要等它返回（通常几秒 ✓），别把话说成大喘气 ✓。
            self.btn_cancel.setEnabled(False)
            self.progress.setFormat("取消中…")
            self.log("已请求取消：正在收尾（各步检查点上收工；"
                     "只有「载入 YOLO 权重」那一下要等它返回，通常几秒）", "warn")

    # ══════════════════════════════════════════════════
    # 主视区
    # ══════════════════════════════════════════════════
    #: ⭐⭐ **常驻页签**（用户 2026-09-28 要求 ✓ 原话："起始页签、实时页签、质检台页签现在是
    #:   3 个常驻页签，**不需要关闭按钮**"）—— 它们**关不掉**，而且页签上**不显示「×」** ✓。
    #:   ⚠ 用**控件身份**判（不是标题 ✗）：标题是可以变的（`_view_title` 那类 ✓），身份不会 ✗。
    def _resident_views(self):
        return (self.home_view, self.live_panel, self.review)

    def _refresh_tab_close_buttons(self):
        """把**常驻页签**上那个「× 关闭按钮」摘掉（用户 2026-09-28 要求 ✓）。

        ⚠ 为什么只能这么干：`QTabWidget.setTabsClosable(True)` 是**整条页签栏**的开关 ✗，
          **没法**只让某几页不可关 ✗ ⇒ 只能逐页把右侧那个按钮**换掉**
          （`tabBar().setTabButton(i, QTabBar.RightSide, None)` ✓）。
        ⚠ 所以**页签一增减就必须重刷**（`open_view` / `close_view` / `_on_view_close` 之后都调 ✓）：
          页号会随开合变 ⇒ 谁都不能只刷一次 ✗（这正是本文件反复强调的"别按页号做人肉账"✓）。
        """
        bar = self.viewer.tabBar()
        if bar is None:
            return
        try:
            from PyQt5.QtWidgets import QTabBar

            _res = self._resident_views()
            for _i in range(self.viewer.count()):
                if self.viewer.widget(_i) in _res:
                    bar.setTabButton(_i, QTabBar.RightSide, None)
        except Exception:                      # noqa: BLE001 —— 摘不掉按钮不该拖垮界面 ✓
            pass

    def open_view(self, widget, title):
        """**让一个控件占用主视区**：已经在 ⇒ 切过去；没有 ⇒ 新开一个页签并切过去。

        这是主视区**唯一的打开入口**（用户 2026-09-27 要求："只要会占用主视区的都应该走
        打开、关闭页签流程"）—— 别再直接 `self.viewer.setCurrentIndex(...)` /
        `self.viewer.addTab(...)` ✗：页号会随开合变化，任何人肉算页号的地方迟早错位
        （用例 `t_viewer_has_single_entry` 钉着这条 ✓）。
        """
        idx = self.viewer.indexOf(widget)
        if idx < 0:
            idx = self.viewer.addTab(widget, title)
            # ⭐ **新开的页签也收尾一次**（提示搬标题 + 可复制 ✓）：这些控件大多是**懒建**的
            #   （点「查看」才造出来 ✗）⇒ 主窗口 `__init__` 那一次扫不到它们 ✓（用户第 3、4 条 ✓）
            try:
                theme.finish_window(widget)
            except Exception as e:                    # noqa: BLE001
                self.log("页签收尾没跑成（%s）：%s" % (title, e), "warn")
        self.viewer.setCurrentIndex(idx)
        self._refresh_tab_close_buttons()   # ⭐ 页签增减 ⇒ 重刷「×」的显隐 ✓（页号会变 ✗）
        return idx

    def close_view(self, widget):
        """关掉某个控件的页签（**不销毁控件**：下次 `open_view` 还接着用原来那个 ✓）。"""
        idx = self.viewer.indexOf(widget)
        if idx >= 0:
            self.viewer.removeTab(idx)      # removeTab 只是摘下来，不删控件 ✓
            self._refresh_tab_close_buttons()   # ⭐ 摘掉一个 ⇒ 页号全变 ⇒ 重刷 ✓

    def _on_view_close(self, idx):
        """点了页签上的 × ⇒ 关掉它。

        ⭐⭐ **三个常驻页签一律关不掉**（用户 2026-09-28 要求 ✓ 原话："起始页签、实时页签、
        质检台页签现在是 **3 个常驻页签，不需要关闭按钮**"）：
          · 「起始」—— 所有工作页关掉之后总得有个落脚处 ✓（老规矩 ✓）；
          · 「实时」—— 它是这套东西的**主界面**（顶栏那个「实时」按钮也一起去掉了 ✓）；
          · 「质检台」—— 同上，常驻 ✓。
        ⚠ 这三个页签上**也不显示「×」**（`_refresh_tab_close_buttons` ✓）—— 这里是**第二道保险**
          （万一还有别的路径触发 `tabCloseRequested` ✓，照样关不掉 ✓）。

        ⚠ 「实时」页签**关掉不会停**收流 / 推理（页签只管"看不看得见"✓）—— 现在既然关不掉了，
          这条提醒也就**不再需要** ✓（原来的那句已随"可关"一起去掉 ✓）。
        """
        w = self.viewer.widget(idx)
        if w in self._resident_views():
            self.log("「起始 / 实时 / 质检台」是常驻页签，不能关 ✓", "warn")
            return
        self.viewer.removeTab(idx)
        self._refresh_tab_close_buttons()   # ⭐ 关掉一个 ⇒ 页号全变 ⇒ 重刷 ✓

    def _view_title(self, card):
        """步骤详情页的**页签标题**（和页面标题同一份说法 ✓，别两处各写一个）。"""
        return "%s %s" % (CIRCLED[card.num - 1], card.title)

    def _on_view(self, card):
        # ⑤ 标注校验的「查看」= 打开质检台。
        # 原来是挂在 ④ 上的，但质检台做的是校验修正，属于 ⑤ 的职责 ——
        # 挂在 ④ 上会让人以为"标注跑完就算完事了"。
        if card.key == "editor":
            self.review.bind(self.project)
            self.open_view(self.review, "质检台")
            return

        # ⑧ 验证的「查看」= 打开验证结果浏览器
        if card.key == "verify":
            self.verify_viewer.bind(self.project)
            self.open_view(self.verify_viewer, "验证结果")
            return

        self._rebuild_page(card)
        w = self.view_widgets.get(card.key)
        if w is not None:
            self.open_view(w, self._view_title(card))

    def _rebuild_page(self, card):
        """详情页：首次访问时创建，之后只更新文字（**不重建控件** ✓）。

        ⚠ 这里**只负责"内容"**，页签由 `open_view` 开（开合一入口 ✓）—— 在这里
          `addTab` 就等于又长出一条旁路 ✗。
        """
        body = "%s\n\n%s" % (
            card.hint or "",
            card.summarize(self.project) if self.project else "未选择项目")

        page = self.view_widgets.get(card.key)
        if isinstance(page, InfoPage):
            page.set_body(body)
            return
        self.view_widgets[card.key] = InfoPage(self._view_title(card), body)

    def _clear_pages(self):
        """清掉**步骤详情页**（换项目时用）：四个常驻工作页与起始页**不在这里关**。

        为什么常驻工作页不关：它们是"工作台的一部分"（实时 / 质检台 / 验证结果 /
        角色匹配）—— 换项目后各自 `bind()` 换数据 ✓；把人正在看的页面抽走反而更糟 ✗。
        """
        for w in list(self.view_widgets.values()):
            self.close_view(w)      # 先摘页签，再销毁控件（顺序反了会留下空标签 ✗）
            w.deleteLater()         # 详情页是"用完即弃"的：内容跟着项目变 ✓
        self.view_widgets.clear()
        self.open_view(self.home_view, "起始")

    # ══════════════════════════════════════════════════
    # 日志
    # ══════════════════════════════════════════════════
    def log(self, msg, level="info"):
        color = self.LOG_COLORS.get(level, self.LOG_COLORS["info"])
        stamp = time.strftime("%H:%M:%S")
        pad = "&nbsp;" * len(stamp)

        lines = str(msg).splitlines() or [""]
        for i, line in enumerate(lines):
            prefix = stamp if i == 0 else pad
            self.txt_log.appendHtml(
                '<span style="color:%s">%s&nbsp;%s</span>'
                % (color, prefix, html.escape(line)))

        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ══════════════════════════════════════════════════
    def closeEvent(self, e):
        # 实时预览的线程必须显式收掉。QThread 在「还在运行」的状态下被销毁，
        # Qt 不会给警告，而是直接 abort —— 表现是关窗口时程序无征兆闪退。
        if self.live_panel.thread is not None:
            self.live_panel.shutdown()
        if self.player_panel.thread is not None:
            self.player_panel.shutdown()

        if self.task is not None and self.task.isRunning():
            r = QMessageBox.question(self, "确认退出", "有任务正在运行，确定要退出吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.task.cancel()
            self.task.wait(5000)

        # 注销 F11 全局热键，避免残留占着系统快捷键
        from decision import hotkey as _hotkey
        _hotkey.unregister(int(self.winId()), self._hotkey_id)
        # ⚠⚠ 键盘钩子也**必须卸** ✗（它挂在**系统钩子链**上 ✓ 不卸 ⇒ 关掉工作台后
        #   还会被叫进来、拖慢整机按键 ✓ —— 而且进程退出时系统才收，体验很差 ✓）
        _hotkey.remove_hook()
        e.accept()

    def _register_auto_hotkey(self):
        """按「开关自动」的映射注册全局热键 —— **三层**：注册制 → 键盘钩子 → 窗口内快捷键。

        ⚠⚠ 为什么要三层（2026-10-10 ✓ 用户原话：**"我需要 F11 全局生效"**）：
          · `RegisterHotKey`（第 1 层）是**独占**的 ✗ —— 同一个键**只允许一个进程**注册 ✓
            现场实测 `RegisterHotKey(…, VK_F11)` **返回 0** ✗（`F12` 也被占着 ✓ 只有 `F10` 能 ✓）
            ⇒ 本进程只能退到"本窗口聚焦时"✗ ⇒ 用户把焦点切到游戏后按 F11 **毫无反应** ✓；
          · `WH_KEYBOARD_LL`（第 2 层 ✓ 见 `decision/hotkey.install_hook`）是**观察型** ✓
            —— 别人占着同一个键，我们**照样收得到** ✓（**不吞键** ✓ 游戏那边照旧 ✓）；
          · 最后一层仍是窗口内快捷键 ✓（两层都失败时才用 ✓ 语义 = "本窗口聚焦时有效" ✓）。
        ⚠ 第 1 层成功时要**把钩子卸掉** ✗（两路都活着 = 按一下触发两回 ⇒ 开了又关 ✓）。
        """
        from decision import hotkey as _hotkey
        from decision.agent import settings
        from decision.input import resolve_vk
        key_name = settings.keymap.get("auto", "f11")
        vk = resolve_vk(key_name)
        if vk is None:
            self._hotkey_how = "none"
            self.log("「开关自动」映射的键无效，全局热键未注册", "warn")
            return
        _hotkey.unregister(int(self.winId()), self._hotkey_id)
        if _hotkey.register(int(self.winId()), self._hotkey_id, vk):
            _hotkey.remove_hook()               # ⚠ 别两路都活着（不然按一下 = 开+关 ✓）
            self._hotkey_how = "hotkey"
            if self._auto_shortcut is not None:
                self._auto_shortcut.setEnabled(False)
            self._refresh_hotkey_ui()
            return
        # ---- 第 2 层：键盘钩子（**不独占** ⇒ 别人占着也能全局生效 ✓）----
        _ok, _why = _hotkey.install_hook(vk, self._on_auto_hotkey_fired)
        if _ok:
            self._hotkey_how = "hook"
            if self._auto_shortcut is not None:
                self._auto_shortcut.setEnabled(False)
            self.log("`%s` 已被别的进程注册 ⇒ 改用键盘钩子（不独占，照样在任何窗口生效）"
                     % key_name, "info")
            self._note_hotkey_fallback("hook", key_name, "")
            self._refresh_hotkey_ui()
            return
        # ---- 第 3 层：窗口内快捷键 ----
        self._hotkey_how = "window"
        if self._auto_shortcut is None:
            self._auto_shortcut = QShortcut(QKeySequence(key_name), self)
            # ⚠ 显式写出来（§10）：这里**故意**用 WindowShortcut（默认值）—— 降级语义就是
            #   "本窗口聚焦时"仍能用同一键开关自动；不写出来后人会以为是漏了 ✗
            self._auto_shortcut.setContext(Qt.WindowShortcut)
            self._auto_shortcut.activated.connect(self.player_panel.toggle_auto)
        else:
            self._auto_shortcut.setKey(QKeySequence(key_name))
        self._auto_shortcut.setEnabled(True)
        # ⚠⚠ 这句要**说清"怎么办"** ✗（原来只说"被占用"⇒ 人只能猜 ✓ 用户 2026-10-10 现场 ✓）：
        #   多半是**另一个本程序实例**还开着（它先注册的 ✓ 见 295 条那次"两个工作台"✓）。
        self.log("全局热键 `%s` 被别的进程占用，键盘钩子也没装上（%s）"
                 "⇒ 现在**只在本窗口聚焦时**有效。"
                 "想让它在游戏里也生效：关掉多余的**本程序实例**（多半是它占着），"
                 "或在「决策参数 → 键位」里换一个键 ✓" % (key_name, _why), "warn")
        self._note_hotkey_fallback("window", key_name, _why)
        self._refresh_hotkey_ui()

    def _note_hotkey_fallback(self, how, key_name, why):
        """留痕：全局热键降级了（**别静默** ✗ —— 人在游戏里按 F11 没反应会以为程序坏了 ✓）。"""
        try:
            from core import behavior            # ⚠ 是 `core.behavior` ✓（写成裸 `import behavior`
            behavior.event("hotkey_fallback", how=str(how), key=str(key_name),
                           why=str(why or "")[:120])
        except Exception:                        # noqa: BLE001 —— 打点坏了别影响启动 ✓
            pass

    def _refresh_hotkey_ui(self):
        """把"热键现在哪种生效"写进按钮 tooltip（**让人看得见** ✓ —— 见 `_register_auto_hotkey`）。"""
        try:
            _how = getattr(self, "_hotkey_how", "")
            _txt = {"hotkey": "全局热键已注册：任何窗口聚焦都生效 ✓",
                    "hook": ("全局热键被别的进程占着 ⇒ 已改用**键盘钩子**："
                             "照样在任何窗口生效 ✓（它不独占 ✓）"),
                    "window": ("⚠ 全局热键和键盘钩子都被挡住了 ⇒ **只在本窗口聚焦时**有效。"
                               "想让游戏里也生效：关掉多余的本程序实例，或换个键 ✓"),
                    "none": "⚠ 映射的键无效，没注册 ✓"}.get(_how)
            if _txt:
                self.player_panel.btn_auto.setToolTip(
                    "开始/停止自动打怪。\n%s" % _txt)
        except Exception:                        # noqa: BLE001 —— 只是提示，别把启动弄崩 ✓
            pass

    def _on_auto_key_changed(self, key):
        self._register_auto_hotkey()

    def _on_auto_hotkey_fired(self):
        """热键**真的被按下**了 —— 注册制与键盘钩子**两路共用的唯一出口** ✓。

        ⚠⚠ 钩子那一路是在**系统钩子链里**被叫的 ✗ ⇒ 这里**不许做重活**（慢了会拖全系统的
          按键 ✓ 见 `decision/hotkey.install_hook` 的说明 ✓）⇒ 丢回 Qt 事件循环再开关自动 ✓。
        """
        QTimer.singleShot(0, self.player_panel.toggle_auto)

    def nativeEvent(self, eventType, message):
        """接收系统级 WM_HOTKEY 全局热键，任何窗口聚焦都能触发。"""
        from decision import hotkey as _hotkey
        if _hotkey.is_hotkey(message, self._hotkey_id):
            self._on_auto_hotkey_fired()
            return True, 0
        return super().nativeEvent(eventType, message)
