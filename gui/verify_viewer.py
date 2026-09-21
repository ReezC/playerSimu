"""浏览 ⑧ 验证产出的带框图。

只读浏览：翻页、缩放、平移。不做任何编辑 —— 验证结果就是最终产物，
不需要像质检台那样改框。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                             QWidget)

from gui.canvas import ImageCanvas


class VerifyViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.files = []
        self.index = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(6)

        for text, delta in (("◀", -1), ("▶", 1)):
            b = QPushButton(text)
            b.setFixedWidth(36)
            b.setStyleSheet("padding: 2px 4px;")
            b.clicked.connect(lambda _, d=delta: self.step(d))
            bar.addWidget(b)
            if delta < 0:
                self.lbl = QLabel("0 / 0")
                self.lbl.setMinimumWidth(180)
                self.lbl.setAlignment(Qt.AlignCenter)
                bar.addWidget(self.lbl)

        bar.addStretch(1)

        btn_fit = QPushButton("适应窗口")
        btn_fit.clicked.connect(lambda: self.canvas.fit())
        bar.addWidget(btn_fit)

        root.addLayout(bar)

        self.canvas = ImageCanvas()
        root.addWidget(self.canvas, 1)

        hint = QLabel("滚轮=缩放　中键拖=平移　双击=适应")
        hint.setStyleSheet("color: #80868b;")
        root.addWidget(hint)

    def bind(self, project):
        """载入项目的验证产物（verify/ 目录里的带框图）。"""
        self.files = sorted(project.dir_of("verify").glob("*.jpg"))
        self.index = 0
        self._load()

    def step(self, delta):
        if not self.files:
            return
        self.index = max(0, min(self.index + delta, len(self.files) - 1))
        self._load()

    def _load(self):
        if not self.files:
            self.canvas.load(QPixmap(), [])
            self.lbl.setText("0 / 0")
            return

        f = self.files[self.index]
        pm = QPixmap(str(f))
        if pm.isNull():
            self.lbl.setText("%d / %d   读不到 %s" % (self.index + 1, len(self.files), f.name))
            return

        self.canvas.load(pm, [], editable=False, fit=True)
        self.lbl.setText("%d / %d   %s" % (self.index + 1, len(self.files), f.name))
