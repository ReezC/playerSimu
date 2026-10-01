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

from PyQt5.QtCore import Qt
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
        self.cards = []

        self._build()
        self.setStyleSheet(QSS)
        self._reload_projects()

        if not PROJECTS_DIR.exists():
            PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

        self.log("工作台已就绪。先「新建」或「打开」一个项目。", "ok")
        self.log("项目根目录: %s" % PROJECTS_DIR)

        # 启动也要绑一次决策参数：没有项目时用的是「最近打开的那个项目」那份，
        # 不绑的话参数面板显示默认值，连下面注册的全局热键都会用默认 F11，
        # 而不是那个项目里映射的键。
        self._bind_cards()

        # 全局热键：任何窗口聚焦时都能开关自动打怪。键 = 决策参数里
        # 「开关自动」的映射（默认 F11），改了映射会动态重新注册。
        self._hotkey_id = 1
        self._auto_shortcut = None   # 全局热键注册失败时的窗口内快捷键降级
        self._register_auto_hotkey()

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
        set_save_hook(lambda d, p=project: (p.set("decision", d), p.save()))

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

    def _start(self, card, fn, params):
        self.current_card = card
        card.set_state("running")
        card.set_result("运行中…")
        card.set_busy(True)

        for c in self.cards:
            c.set_busy(True)

        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("运行中…")
        self.log("── 开始 %s %s ──" % (CIRCLED[card.num - 1], card.title), "info")

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
            # 立刻停 —— 模型/模板加载这类步骤拦不住，得等它返回（各阶段的检查点
            # 见 tools/ 里的 ctx.canceled()）。
            self.btn_cancel.setEnabled(False)
            self.progress.setFormat("取消中…")
            self.log("已请求取消：正在收尾（模型/模板加载这类步骤要等它返回）", "warn")

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
        e.accept()

    def _register_auto_hotkey(self):
        """按「开关自动」映射动态注册全局热键；失败降级为窗口内快捷键。"""
        from decision import hotkey as _hotkey
        from decision.agent import settings
        from decision.input import resolve_vk
        key_name = settings.keymap.get("auto", "f11")
        vk = resolve_vk(key_name)
        if vk is None:
            self.log("「开关自动」映射的键无效，全局热键未注册", "warn")
            return
        _hotkey.unregister(int(self.winId()), self._hotkey_id)
        if _hotkey.register(int(self.winId()), self._hotkey_id, vk):
            # 注册成功：禁用降级用的窗口内快捷键（如果有）
            if self._auto_shortcut is not None:
                self._auto_shortcut.setEnabled(False)
            return
        # 注册失败（多半是另一个本程序实例已占用这个全局热键）：
        # 降级为窗口内快捷键，保证本窗口聚焦时仍能用同一键开关自动。
        if self._auto_shortcut is None:
            self._auto_shortcut = QShortcut(QKeySequence(key_name), self)
            # ⚠ 显式写出来（§10）：这里**故意**用 WindowShortcut（默认值）—— 降级语义就是
            #   "本窗口聚焦时"仍能用同一键开关自动；不写出来后人会以为是漏了 ✗
            self._auto_shortcut.setContext(Qt.WindowShortcut)
            self._auto_shortcut.activated.connect(self.player_panel.toggle_auto)
        else:
            self._auto_shortcut.setKey(QKeySequence(key_name))
        self._auto_shortcut.setEnabled(True)
        self.log("全局热键被占用（可能是另一个本程序实例），已降级为窗口内快捷键", "info")

    def _on_auto_key_changed(self, key):
        self._register_auto_hotkey()

    def nativeEvent(self, eventType, message):
        """接收系统级 WM_HOTKEY 全局热键，任何窗口聚焦都能触发。"""
        from decision import hotkey as _hotkey
        if _hotkey.is_hotkey(message, self._hotkey_id):
            self.player_panel.toggle_auto()
            return True, 0
        return super().nativeEvent(eventType, message)
