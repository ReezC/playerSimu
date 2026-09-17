"""WZ 导出对话框：解析 Map.wz + String.wz，生成地图清单。

这是**全局操作**，不属于任何项目 —— 导出一次，所有项目都能从下拉列表
里选地图。所以它挂在工具栏上，而不是做成流程中的一张卡片。
"""

import html
import time
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPlainTextEdit,
                             QProgressBar, QPushButton, QVBoxLayout)

from core import wzexport
from gui.worker import TaskThread, safe_slot

ROOT = Path(__file__).resolve().parent.parent


class ExportDialog(QDialog):
    exported = pyqtSignal()      # 导出成功，主窗口据此刷新地图下拉

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WZ 导出 · 地图清单")
        self.resize(780, 600)

        self.task = None
        self._build()
        self._load()

    # ---------------- 界面 ----------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        tip = QLabel(
            "解析 Map.wz 得到「地图 → 怪种」对应关系，供 ① 地图卡片的"
            "下拉列表使用。\n地图数据基本不变，导一次可以长期用；换客户端版本时才需要重导。")
        tip.setStyleSheet("color: #5f6368;")
        tip.setWordWrap(True)
        root.addWidget(tip)

        self.ed_wz = self._path_row(root, "WZ 目录", "", True)
        self.ed_exe = self._path_row(root, "WzProbe", "", False)
        self.ed_out = self._path_row(root, "输出文件", "", False)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color: #80868b;")
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint)

        root.addWidget(QLabel("日志"))
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumBlockCount(2000)
        self.txt.setStyleSheet(
            "background:#202124; color:#e8eaed; font-family:Consolas,monospace;")
        root.addWidget(self.txt, 1)

        bar = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("空闲")
        bar.addWidget(self.progress, 1)

        self.btn_start = QPushButton("开始导出")
        self.btn_start.clicked.connect(self._start)
        bar.addWidget(self.btn_start)

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        bar.addWidget(self.btn_close)

        root.addLayout(bar)

    def _path_row(self, layout, label, value, is_dir):
        row = QHBoxLayout()
        lab = QLabel(label)
        lab.setFixedWidth(72)
        row.addWidget(lab)

        edit = QLineEdit(value)
        row.addWidget(edit, 1)

        btn = QPushButton("浏览")
        btn.setFixedWidth(56)
        btn.clicked.connect(lambda: self._pick(edit, is_dir))
        row.addWidget(btn)

        layout.addLayout(row)
        return edit

    def _pick(self, edit, is_dir):
        cur = edit.text().strip()
        if is_dir:
            p = QFileDialog.getExistingDirectory(self, "选择目录", cur or str(ROOT))
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "选择文件", cur or str(ROOT),
                "可执行文件 (*.exe);;JSON (*.json);;所有文件 (*)")
        if p:
            edit.setText(p)

    # ---------------- 配置读写 ----------------

    def _load(self):
        cfg = wzexport.load_cfg()

        self.ed_wz.setText(cfg.get("wz_dir", ""))

        exe = wzexport.find_probe_exe(cfg.get("probe_exe", ""))
        self.ed_exe.setText(exe)
        if not exe:
            self.lbl_hint.setText(
                "未找到 WzProbe.exe。请先编译：\n"
                "  cd E:\\MyPrograms\\RippleRogue\\MapleNecrocer\n"
                "  dotnet build WzProbe/WzProbe.csproj -c Release")

        self.ed_out.setText(str(wzexport.maps_json_path(cfg)))

        idx = wzexport.load_index()
        if idx:
            self._append("已有清单: %d 张地图（有怪 %d），%s"
                         % (idx.get("count", 0), idx.get("with_mob", 0),
                            idx.get("exported_at", "?")))

    def _save_cfg(self):
        cfg = wzexport.load_cfg()

        def rel(p):
            """项目根内的路径存相对，方便整体搬迁。"""
            try:
                return Path(p).resolve().relative_to(ROOT).as_posix()
            except Exception:
                return p

        cfg["wz_dir"] = self.ed_wz.text().strip()
        cfg["probe_exe"] = self.ed_exe.text().strip()
        cfg["maps_json"] = rel(self.ed_out.text().strip())
        wzexport.save_cfg(cfg)

    # ---------------- 执行 ----------------

    def _start(self):
        if self.task is not None:
            return

        wz_dir = self.ed_wz.text().strip()
        exe = self.ed_exe.text().strip()
        out = self.ed_out.text().strip()

        if not wz_dir:
            QMessageBox.warning(self, "缺少参数", "请先选择 WZ 目录")
            return
        if not out:
            QMessageBox.warning(self, "缺少参数", "请先指定输出文件")
            return

        self._save_cfg()

        self.btn_start.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("导出中…")
        self.txt.clear()

        params = {"exe": exe, "wz_dir": wz_dir, "out": out}

        self.task = TaskThread(wzexport.run_export_maps, params, self)
        # 过 safe_slot：槽里抛异常会让 PyQt5 abort() 掉整个程序
        self.task.sig_log.connect(safe_slot(self._append))
        self.task.sig_progress.connect(safe_slot(self._on_progress))
        self.task.sig_done.connect(safe_slot(self._on_done))
        # 线程真正结束才释放引用，理由同 main_window._on_task_finished
        self.task.finished.connect(self._on_task_finished)
        self.task.start()

    def _on_task_finished(self):
        t = self.task
        self.task = None
        if t is not None:
            t.deleteLater()

    def _on_progress(self, cur, total, text):
        if total > 0:
            if self.progress.maximum() != total:
                self.progress.setRange(0, total)
                self.progress.setFormat("%v / %m   %p%")
            self.progress.setValue(cur)
        else:
            self.progress.setRange(0, 0)

    def _on_done(self, ok, summary, result=None):
        self.btn_start.setEnabled(True)
        self.btn_close.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if ok else 0)
        self.progress.setFormat("完成" if ok else "失败")

        if ok:
            self._append("完成: " + summary, "ok")
            self.exported.emit()
        else:
            self._append("失败: " + summary, "error")
            QMessageBox.critical(self, "导出失败", summary)

    # ---------------- 日志 ----------------

    COLORS = {"info": "#e8eaed", "ok": "#81c995",
              "warn": "#fdd663", "error": "#f28b82"}

    def _append(self, msg, level="info"):
        color = self.COLORS.get(level, self.COLORS["info"])
        stamp = time.strftime("%H:%M:%S")
        pad = "&nbsp;" * len(stamp)

        for i, line in enumerate(str(msg).splitlines() or [""]):
            self.txt.appendHtml('<span style="color:%s">%s&nbsp;%s</span>'
                                % (color, stamp if i == 0 else pad,
                                   html.escape(line)))

        sb = self.txt.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, e):
        if self.task is not None and self.task.isRunning():
            r = QMessageBox.question(self, "确认", "导出正在进行，确定要中止吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.task.cancel()
            self.task.wait(3000)
        e.accept()
