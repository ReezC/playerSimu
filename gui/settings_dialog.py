"""设置弹窗：界面字号 + 朝向超时。

字号改动**立即生效**，不需要重启 —— 底层是 QApplication.setFont()，
所有没写死字号的控件都会跟随。朝向超时是决策参数，点确定后写入 decision.json。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QColorDialog, QDialog, QDialogButtonBox,
                             QFormLayout, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QSpinBox, QVBoxLayout)

from decision.agent import settings
from gui import theme


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(360)

        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 12)

        # ---- 字号 ----
        head = QHBoxLayout()
        lab = QLabel("界面字号")
        head.addWidget(lab)
        head.addStretch(1)

        self.lbl_val = QLabel()
        self.lbl_val.setStyleSheet("color: #5f6368;")
        head.addWidget(self.lbl_val)
        root.addLayout(head)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(theme.MIN_SIZE, theme.MAX_SIZE)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(1)
        self.slider.setValue(theme.load_size())
        self.slider.valueChanged.connect(self._on_changed)
        root.addWidget(self.slider)

        scale = QHBoxLayout()
        scale.addWidget(QLabel("小"))
        scale.addStretch(1)
        scale.addWidget(QLabel("大"))
        root.addLayout(scale)

        self.lbl_preview = QLabel("预览：怪物检出 12 框 · 置信度 0.92")
        self.lbl_preview.setStyleSheet(
            "background: #ffffff; border: 1px solid #dadce0;"
            " border-radius: 4px; padding: 8px; color: #202124;")
        root.addWidget(self.lbl_preview)

        self.lbl_note = QLabel()
        self.lbl_note.setStyleSheet("color: #80868b;")
        self.lbl_note.setWordWrap(True)
        root.addWidget(self.lbl_note)

        # ---- 朝向超时 ----
        root.addSpacing(6)
        lbl_to = QLabel("朝向无变化停止自动")
        lbl_to.setStyleSheet("font-weight: 600;")
        root.addWidget(lbl_to)

        note_to = QLabel("角色朝向超过该时长没变化，自动停止（0 = 禁用）。")
        note_to.setStyleSheet("color: #5f6368;")
        note_to.setWordWrap(True)
        root.addWidget(note_to)

        self.sp_timeout = QSpinBox()
        self.sp_timeout.setRange(0, 1440)
        self.sp_timeout.setSuffix(" min")
        self.sp_timeout.setValue(int(settings.facing_timeout_min))
        root.addWidget(self.sp_timeout)

        # ---- 找不到玩家超时 ----
        root.addSpacing(6)
        lbl_lost = QLabel("找不到玩家停止自动")
        lbl_lost.setStyleSheet("font-weight: 600;")
        root.addWidget(lbl_lost)

        note_lost = QLabel("连续找不到玩家超过该时长，自动停止（0 = 禁用）。")
        note_lost.setStyleSheet("color: #5f6368;")
        note_lost.setWordWrap(True)
        root.addWidget(note_lost)

        self.sp_player_lost = QSpinBox()
        self.sp_player_lost.setRange(0, 1440)
        self.sp_player_lost.setSuffix(" min")
        self.sp_player_lost.setValue(int(settings.player_lost_timeout_min))
        root.addWidget(self.sp_player_lost)

        # ---- 定时清空按键 ----
        root.addSpacing(6)
        lbl_reset = QLabel("定时清空按键（防卡键）")
        lbl_reset.setStyleSheet("font-weight: 600;")
        root.addWidget(lbl_reset)

        note_reset = QLabel("每隔该秒数向 Pro Micro 发一次 RELEASEALL，清空可能卡住的键（0 = 禁用）。")
        note_reset.setStyleSheet("color: #5f6368;")
        note_reset.setWordWrap(True)
        root.addWidget(note_reset)

        self.sp_resetall = QSpinBox()
        self.sp_resetall.setRange(0, 3600)
        self.sp_resetall.setSuffix(" s")
        self.sp_resetall.setValue(int(settings.resetall_interval))
        root.addWidget(self.sp_resetall)

        # ---- 可视化 ----
        root.addSpacing(6)
        lbl_vis = QLabel("可视化（实时预览）")
        lbl_vis.setStyleSheet("font-weight: 600;")
        root.addWidget(lbl_vis)

        note_vis = QLabel("实时预览里框和线条的颜色。保存后重新开始实时预览生效。")
        note_vis.setStyleSheet("color: #5f6368;")
        note_vis.setWordWrap(True)
        root.addWidget(note_vis)

        self._vis = theme.load_vis()
        vf = QFormLayout()
        vf.setLabelAlignment(Qt.AlignLeft)
        self._color_btns = {}
        for key, label in (("mob_color", "怪物框颜色"),
                           ("player_color", "玩家框颜色"),
                           ("lock_color", "锁定框颜色"),
                           ("attack_color", "最大攻击距离线颜色"),
                           ("min_attack_color", "最小攻击距离线颜色"),
                           ("vision_color", "视野线颜色")):
            self._color_btns[key] = self._color_btn(self._vis[key])
            vf.addRow(label, self._color_btns[key])

        self.sp_vision_width = QSpinBox()
        self.sp_vision_width.setRange(1, 10)
        self.sp_vision_width.setSuffix(" px")
        self.sp_vision_width.setValue(int(self._vis["vision_width"]))
        vf.addRow("视野线宽度", self.sp_vision_width)
        root.addLayout(vf)

        # ---- 按钮 ----
        box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("确定")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        root.addWidget(box)

        self._orig = theme.load_size()
        self._on_changed(self.slider.value())

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

    def _color_btn(self, color):
        """一个显示当前颜色、点击弹颜色选择器的按钮。"""
        btn = QPushButton(color)
        btn.setFixedWidth(110)
        btn._color = color
        btn.setStyleSheet(
            "QPushButton { background: %s; color: #202124; border: 1px solid #dadce0;"
            " border-radius: 4px; padding: 4px 8px; }" % color)
        btn.clicked.connect(lambda: self._pick_color(btn))
        return btn

    def _pick_color(self, btn):
        from PyQt5.QtGui import QColor
        c = QColorDialog.getColor(QColor(btn._color), self, "选择颜色")
        if c.isValid():
            btn._color = c.name()
            btn.setText(c.name())
            btn.setStyleSheet(
                "QPushButton { background: %s; color: #202124;"
                " border: 1px solid #dadce0; border-radius: 4px; padding: 4px 8px; }"
                % c.name())

    def _accept(self):
        v = self.slider.value()
        if v != self._orig:
            theme.save_size(v)
            from PyQt5.QtWidgets import QApplication
            theme.apply(QApplication.instance(), v)
        # 朝向超时
        t = self.sp_timeout.value()
        if t != settings.facing_timeout_min:
            settings.facing_timeout_min = t
            settings.save()
        # 找不到玩家超时
        t2 = self.sp_player_lost.value()
        if t2 != settings.player_lost_timeout_min:
            settings.player_lost_timeout_min = t2
            settings.save()
        # 定时清空按键
        t3 = self.sp_resetall.value()
        if t3 != settings.resetall_interval:
            settings.resetall_interval = t3
            settings.save()
        # 可视化
        vis_cfg = {k: b._color for k, b in self._color_btns.items()}
        vis_cfg["vision_width"] = self.sp_vision_width.value()
        theme.save_vis(vis_cfg)
        self.accept()
