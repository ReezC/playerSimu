"""训练报告弹窗：左栏**本项目所有权重**，右栏那一版的评估 + 建议（不同颜色区分重点）。

2026-09-26 用户要求：把原来贴在「训练」卡片结果区的那段建议**移位**到这里 ——
由卡片上的「查看报告」按钮弹出，并且能**逐个权重**翻看。

**为什么独立成文件**：`gui/` 下的弹窗一律一个文件（`export_dialog` / `zone_editor` /
`settings_dialog`…），卡片只用**懒导入**把它叫起来 —— 卡片只管收参数、发运行请求
（见 `gui/steps/__init__.py` 的分工），不该在已经近 2000 行的 `cards.py` 里长出弹窗。

**数据全部派生自产物**（`runs/detect_vN/run.json` + `models/*.pt`），不另存任何状态 ⇒
重开工作台、换项目，看到的都是磁盘上真实的那一份（和卡片摘要同源，不会两处对不上）。

上色沿用仓库既有做法（仓库里没有 QTextBrowser）：
`QPlainTextEdit.appendHtml('<span style="color:…">…</span>')` —— 同
`gui/export_dialog.py` 的日志与 `main_window.log`。
"""

import html as _html
from pathlib import Path

from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QPlainTextEdit, QPushButton,
                             QSplitter, QVBoxLayout)
from PyQt5.QtCore import Qt

from gui.steps.cards import _fmt3, _model_files, _run_dirs
from perception.metrics import (ACCEPT, BALANCED, BOX, COMPARE, FALSE, FIRST,
                                METRICS, MISSED, NOBASE, advise_items)

#: 条目种类 → 整行颜色。**颜色只属于界面**（`perception/metrics.py` 保持不认颜色，
#: 它要能脱离 PyQt 自检）。取值沿用仓库日志那套调色板（见 export_dialog.COLORS）。
_KIND_COLOR = {
    METRICS: "#e8eaed",
    MISSED: "#f28b82",      # 漏检：红 —— 最直接拖后腿的那个
    FALSE: "#fdd663",       # 误检：黄
    BALANCED: "#81c995",    # 持平：绿
    BOX: "#fdd663",         # 框不够准：黄
    COMPARE: "#8ab4f8",     # 与上一版比：蓝
    NOBASE: "#9aa0a6",      # 没法比：灰（别给它抢眼的颜色）
    FIRST: "#9aa0a6",
    ACCEPT: "#81c995",      # 验收口径：绿
}

#: 关键词高亮色（要能压在各自底色上读得出来）
_HL_COLOR = "#ffd54f"
_MUTED = "#9aa0a6"
_FG = "#e8eaed"


def _proj_name(project):
    """项目显示名：`gui.Project` **只保证有 `root`** —— 先试名字类属性，最后退回目录名。

    （踩过：直接 `getattr(project, "name", None)` 拿到 None ⇒ 标题变成「（未选项目）」，
    明明选着项目 —— 实测第一版就是这个问题。）
    """
    for attr in ("name", "title"):
        v = getattr(project, attr, None)
        if v:
            return str(v)
    root = getattr(project, "root", None) or getattr(project, "path", None)
    return Path(str(root)).name if root else "（未选项目）"


class TrainReportDialog(QDialog):
    """看**当前项目**每一个权重的评估与建议。"""

    def __init__(self, project, parent=None, current=None):
        super().__init__(parent)
        self.project = project
        self._items = []
        self._html = []                 # 累积 HTML（用例要断言"确实上色了"）
        self.setWindowTitle("训练报告 · %s" % _proj_name(project))
        self.resize(760, 560)
        self._build()
        self._fill(current)

    # ---------------- 界面 ----------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        self.lbl_top = QLabel("左边是**本项目训练过的权重**（归档的 + 还在 runs 里的）。"
                              "选一个，右边说它这一版读到什么、下一步做什么。")
        self.lbl_top.setWordWrap(True)
        self.lbl_top.setStyleSheet("color: #5f6368;")
        root.addWidget(self.lbl_top)

        split = QSplitter(Qt.Horizontal)
        self.lst = QListWidget()
        self.lst.setMinimumWidth(220)
        self.lst.currentRowChanged.connect(self._show)
        split.addWidget(self.lst)

        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setStyleSheet(
            "background:#202124; color:%s; font-family:Consolas,'Microsoft YaHei',monospace;"
            % _FG)
        split.addWidget(self.txt)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        bar = QHBoxLayout()
        bar.addStretch(1)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        bar.addWidget(self.btn_close)
        root.addLayout(bar)

    # ---------------- 数据 ----------------

    def _fill(self, current=None):
        """重填左栏（权重列表），并选中 `current`（默认第一个 = 最新）。"""
        self.lst.blockSignals(True)
        self.lst.clear()
        self._items = _model_files(self.project) if self.project is not None else []
        for name, path, _ver, info in self._items:
            m50 = ((info or {}).get("metrics") or {}).get("map50")
            tag = "已归档" if path.parent.name == "models" else "未归档"
            self.lst.addItem(QListWidgetItem("%s\nmAP50 %s · %s"
                                             % (name, _fmt3(m50), tag)))
        self.lst.blockSignals(False)
        if not self._items:
            self._render_empty()
            return
        row = 0
        if current:
            for i, (name, _p, _v, _i) in enumerate(self._items):
                if name == current:
                    row = i
                    break
        self.lst.setCurrentRow(row)

    def _prev_for(self, name):
        """上一版 = **版本号紧挨着它、更小的那个 run**（目录名当标签）。

        ⚠ 用目录名而不是 `run.json` 里的 `name`：同名重训时 ultralytics 会自己加后缀，
        两者不一定一致，写错了就是"跟哪一版比"标错（比不显示更糟）。
        """
        runs = _run_dirs(self.project) if self.project is not None else []
        for i, (_v, nm, _d, _info) in enumerate(runs):
            if nm == name and i + 1 < len(runs):
                prev = dict(runs[i + 1][3] or {})
                prev["name"] = runs[i + 1][1]
                return prev
        return None

    def _show(self, row):
        self.txt.clear()
        self._html = []
        if not (0 <= row < len(self._items)):
            return
        name, path, _ver, info = self._items[row]
        self._render_head(name, path, info)
        if not info or not (info.get("metrics") or {}):
            # **没有指标就别硬凑一段建议**：满屏 `—` 再补一句"第一版"，只会让人以为
            # 这一版考砸了 —— 如实说"没留下记录"就够了（也顺带说明了它为什么没得比）。
            self._append('<span style="color:%s">这一版没留下指标记录（没有 run.json，'
                         '或它是手工塞进 models/ 的权重）—— 没法给评估，也就没得比涨跌。'
                         '</span>' % _MUTED)
            self._append("")
            return
        self._render_advice(info, self._prev_for(name))

    def _render_empty(self):
        self.txt.clear()
        self._html = []
        self._append('<span style="color:%s">本项目还没有训练过的权重 —— 先在上面的'
                     '「训练」步骤跑一次。</span>' % _MUTED)

    # ---------------- 渲染 ----------------

    def _append(self, html_line):
        self._html.append(html_line)
        self.txt.appendHtml(html_line)

    def rendered_html(self):
        """累积的 HTML（自检用：断言"确实按 kind 上色了"）。"""
        return "\n".join(self._html)

    def _render_head(self, name, path, info):
        info = info or {}
        self._append('<b style="color:%s">%s</b>' % (_FG, _html.escape(str(name))))
        sec = info.get("seconds")
        rows = [("权重", str(path)),
                ("基础权重", info.get("base") or "—"),
                ("轮数", info.get("epochs", "—")),
                ("输入尺寸", info.get("imgsz", "—")),
                ("batch / 设备", "%s / %s" % (info.get("batch", "—"),
                                              info.get("device", "—"))),
                ("用时", ("%.1f 分钟" % (sec / 60.0)) if sec else "—"),
                ("完成于", info.get("finished_at") or "—")]
        for k, v in rows:
            self._append('<span style="color:%s">%s：</span>'
                         '<span style="color:%s">%s</span>'
                         % (_MUTED, k, _FG, _html.escape(str(v))))
        self._append("")

    def _render_advice(self, info, prev):
        """条目 → 整行颜色（按 `kind`）+ 关键词加粗高亮（按 `hl`）。

        ⚠ 关键词按**长到短**替换：不然先替掉 `mAP50`，`mAP50-95` 就再也匹配不上了。
        高亮词必须真的出现在句子里才替换（`perception/metrics.py` 的 docstring 写了
        这条约定），所以这里不用怕"白高亮"。
        """
        for it in advise_items(info, prev):
            color = _KIND_COLOR.get(it["kind"], _KIND_COLOR[METRICS])
            text = _html.escape(it["text"])
            for kw in sorted(it.get("hl") or [], key=len, reverse=True):
                k = _html.escape(str(kw))
                if k and k in text:
                    text = text.replace(
                        k, '<b style="color:%s">%s</b>' % (_HL_COLOR, k))
            self._append('<span style="color:%s">%s</span>' % (color, text))
