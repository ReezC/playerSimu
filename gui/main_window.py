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
from PyQt5.QtWidgets import (QFileDialog, QHBoxLayout, QInputDialog,
                             QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                             QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                             QSplitter, QStackedWidget, QToolBar, QVBoxLayout,
                             QWidget)

from gui import theme
from gui.export_dialog import ExportDialog
from gui.live_panel import LivePanel
from gui.settings_dialog import SettingsDialog
from gui.widgets import NoWheelComboBox
from gui.project import Project, sanitize
from gui.review import ReviewPanel
from gui.steps import ALL_CARDS, CIRCLED
from gui.verify_viewer import VerifyViewer
from gui.worker import TaskThread, safe_slot

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"

QSS = """
QMainWindow, QWidget#Central { background: #f1f3f4; }
QFrame#StepCard {
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 6px;
}
QFrame#StepCard QLabel { color: #202124; }
QPlainTextEdit#Log {
    background: #202124; color: #e8eaed;
    border: none; font-family: Consolas, monospace;
}
QPushButton { padding: 4px 10px; }
"""


PAGE_REVIEW = 1     # 质检台在主视区里的固定页号
PAGE_VERIFY = 2     # 验证结果浏览器的固定页号
PAGE_LIVE = 3       # 实时预览（收流 + 推理）的固定页号


class InfoPage(QWidget):
    """纯文字详情页。

    内容可以更新，而不是重建控件 —— 重建会让 QStackedWidget 的页号错位，
    每次「查看」都往里面塞新页，越用越乱。
    """

    def __init__(self, title, body=""):
        super().__init__()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)

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


class MainWindow(QMainWindow):
    LOG_COLORS = {"info": "#e8eaed", "ok": "#81c995",
                  "warn": "#fdd663", "error": "#f28b82"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("playerSimu 数据集工作台")
        self.resize(1460, 920)

        self.project = None
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
        split.addWidget(self._build_cards())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([900, 520])
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

        sep2 = QLabel("│")
        sep2.setStyleSheet("color: #dadce0;")
        lay.addWidget(sep2)

        self.btn_qa = QPushButton("质检台")
        self.btn_qa.setToolTip("翻看 ⑤ 的标注：筛选异常帧、修正错框 —— 判断框得准不准")
        self.btn_qa.clicked.connect(self._show_review)
        lay.addWidget(self.btn_qa)

        sep3 = QLabel("│")
        sep3.setStyleSheet("color: #dadce0;")
        lay.addWidget(sep3)

        btn_live = QPushButton("实时")
        btn_live.setToolTip("收 A 机推流并实时推理 —— 在窗口里直接看检测效果和帧率")
        btn_live.clicked.connect(self._show_live)
        lay.addWidget(btn_live)

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
        self.viewer = QStackedWidget()
        self.viewer.addWidget(self._placeholder(
            "主视区",
            "这里会显示当前步骤的产物：\n\n"
            "  ·   抽帧画面预览（②）\n"
            "  ·   尺度标定对比图（③）\n"
            "  ·   质检台：翻看 / 筛选 / 修正标注（⑤）\n"
            "  ·   训练曲线与日志（⑦）\n"
            "  ·   检测结果图（⑧）\n\n"
            "点右侧卡片下方的「查看」可切换到这里。"))
        # 质检台固定在 index 1 —— 它是流程里最常看的页面，不参与动态创建
        self.review = ReviewPanel()
        self.viewer.addWidget(self.review)

        # 验证结果浏览器固定在 index 2
        self.verify_viewer = VerifyViewer()
        self.viewer.addWidget(self.verify_viewer)

        # 实时预览固定在 index 3
        self.live_panel = LivePanel()
        self.viewer.addWidget(self.live_panel)

        self.viewer_pages = {}      # card.key -> 动态页页号（从 4 起）
        return self.viewer

    def _placeholder(self, title, body):
        return InfoPage(title, body)

    def _build_cards(self):
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(2, 2, 8, 2)
        inner_lay.setSpacing(8)

        for cls in ALL_CARDS:
            card = cls()
            card.run_clicked.connect(self._on_run)
            card.view_clicked.connect(self._on_view)
            self.cards.append(card)
            inner_lay.addWidget(card)

        inner_lay.addStretch(1)
        scroll.setWidget(inner)

        lay.addWidget(scroll)
        return host

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

    def _bind_cards(self):
        for c in self.cards:
            c.bind(self.project)
        self.review.bind(self.project)
        self.live_panel.bind(self.project)
        self.lbl_status.setText("未选择项目" if self.project is None else
                                self.lbl_status.text())

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
        """切到实时预览。"""
        self.live_panel.bind(self.project)
        self.viewer.setCurrentIndex(PAGE_LIVE)

    def _on_settings(self):
        SettingsDialog(self).exec_()

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

    def _show_review(self):
        """切到质检台。标注还在跑也可以看，已经生成的图会先列出来。"""
        if self.project is None:
            QMessageBox.warning(self, "提示", "请先打开一个项目")
            return

        self.review.bind(self.project)
        self.viewer.setCurrentIndex(PAGE_REVIEW)
        self.log("质检台：%d 张可视化图" % self.review.items.__len__())

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
            self.log("已请求取消，等待当前步骤结束…", "warn")

    # ══════════════════════════════════════════════════
    # 主视区
    # ══════════════════════════════════════════════════
    def _on_view(self, card):
        # ⑤ 标注校验的「查看」= 打开质检台。
        # 原来是挂在 ④ 上的，但质检台做的是校验修正，属于 ⑤ 的职责 ——
        # 挂在 ④ 上会让人以为"标注跑完就算完事了"。
        if card.key == "editor":
            self.review.bind(self.project)
            self.viewer.setCurrentIndex(PAGE_REVIEW)
            return

        # ⑧ 验证的「查看」= 打开验证结果浏览器
        if card.key == "verify":
            self.verify_viewer.bind(self.project)
            self.viewer.setCurrentIndex(PAGE_VERIFY)
            return

        self._rebuild_page(card)
        idx = self.viewer_pages.get(card.key)
        if idx is not None:
            self.viewer.setCurrentIndex(idx)

    def _rebuild_page(self, card):
        """详情页：首次访问时创建，之后只更新文字。"""
        body = "%s\n\n%s" % (
            card.hint or "",
            card.summarize(self.project) if self.project else "未选择项目")

        idx = self.viewer_pages.get(card.key)
        page = self.viewer.widget(idx) if idx is not None else None

        if isinstance(page, InfoPage):
            page.set_body(body)
            return

        page = InfoPage("%s %s" % (CIRCLED[card.num - 1], card.title), body)
        self.viewer.addWidget(page)
        self.viewer_pages[card.key] = self.viewer.count() - 1

    def _clear_pages(self):
        """清掉动态详情页，保留欢迎页、质检台、验证结果浏览器、实时预览。"""
        while self.viewer.count() > 4:
            w = self.viewer.widget(self.viewer.count() - 1)
            self.viewer.removeWidget(w)
            w.deleteLater()
        self.viewer_pages.clear()
        self.viewer.setCurrentIndex(0)

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

        if self.task is not None and self.task.isRunning():
            r = QMessageBox.question(self, "确认退出", "有任务正在运行，确定要退出吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.task.cancel()
            self.task.wait(5000)
        e.accept()
