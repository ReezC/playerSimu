"""设置弹窗：界面字号 + 各种「停止自动」条件 + 断线自动重连。

字号改动**立即生效**，不需要重启 —— 底层是 QApplication.setFont()，
所有没写死字号的控件都会跟随。其余都是决策参数：点确定后写回**当前项目**
（没打开项目时写 config/decision.json，见 decision/agent.py 的 set_save_hook）。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QColorDialog, QDialog, QDialogButtonBox,
                             QFormLayout, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QSpinBox, QVBoxLayout)

from decision.agent import settings
from gui import theme
from perception import classes
# NoWheel* 必须模块级导入：控件在 __init__ 里建，方法里的懒导入到不了那儿。
# （详见 docs/UI规范.md：滚轮不许改参数）
from gui.widgets import (NoWheelDoubleSpinBox, NoWheelSlider,
                         NoWheelSpinBox)


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

        self.slider = NoWheelSlider(Qt.Horizontal)
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

        # 分钟支持小数：用 DoubleSpinBox（0 = 禁用，所以下限仍是 0）
        self.sp_timeout = NoWheelDoubleSpinBox()
        self.sp_timeout.setRange(0.0, 1440.0)
        self.sp_timeout.setDecimals(1)
        self.sp_timeout.setSingleStep(0.5)
        self.sp_timeout.setSuffix(" min")
        self.sp_timeout.setValue(float(settings.facing_timeout_min))
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

        self.sp_player_lost = NoWheelDoubleSpinBox()
        self.sp_player_lost.setRange(0.0, 1440.0)
        self.sp_player_lost.setDecimals(1)
        self.sp_player_lost.setSingleStep(0.5)
        self.sp_player_lost.setSuffix(" min")
        self.sp_player_lost.setValue(float(settings.player_lost_timeout_min))
        root.addWidget(self.sp_player_lost)

        # ---- 断线自动重连 ----
        root.addSpacing(6)
        lbl_rc = QLabel("断线自动重连")
        lbl_rc.setStyleSheet("font-weight: 600;")
        root.addWidget(lbl_rc)

        note_rc = QLabel(
            "玩家框丢失后自动判别界面（断线提示框 / 登录 / 选频道 / 排队 / 选角），"
            "确认是断线就**停止自动**并按步骤走回游戏，回到游戏后恢复自动。\n"
            "鼠标点服务器 / 点频道还没接（要先做鼠标标定），走到那一步会停下并提示。\n"
            "做判断用模板锚点，不读文字、不训模型。")
        note_rc.setStyleSheet("color: #5f6368;")
        note_rc.setWordWrap(True)
        root.addWidget(note_rc)

        self.ck_reconnect = QCheckBox("检测到断线后自动重连")
        self.ck_reconnect.setChecked(bool(settings.reconnect_enabled))
        root.addWidget(self.ck_reconnect)

        # ---- 定时清空按键 ----
        root.addSpacing(6)
        lbl_reset = QLabel("定时清空按键（防卡键）")
        lbl_reset.setStyleSheet("font-weight: 600;")
        root.addWidget(lbl_reset)

        note_reset = QLabel("每隔该秒数向 Pro Micro 发一次 RELEASEALL，清空可能卡住的键（0 = 禁用）。")
        note_reset.setStyleSheet("color: #5f6368;")
        note_reset.setWordWrap(True)
        root.addWidget(note_reset)

        self.sp_resetall = NoWheelSpinBox()
        self.sp_resetall.setRange(0, 3600)
        self.sp_resetall.setSuffix(" s")
        self.sp_resetall.setValue(int(settings.resetall_interval))
        root.addWidget(self.sp_resetall)

        # ---- 可视化 ----
        root.addSpacing(6)
        lbl_vis = QLabel("可视化（实时预览）")
        lbl_vis.setStyleSheet("font-weight: 600;")
        root.addWidget(lbl_vis)

        note_vis = QLabel(
            "实时预览 / 质检台 / 推理结果里，每个类别的框颜色和辅助线颜色。\n"
            "点确定立刻生效（实时预览每秒重读一次配置，不用重开）。")
        note_vis.setStyleSheet("color: #5f6368;")
        note_vis.setWordWrap(True)
        root.addWidget(note_vis)

        self._vis = theme.load_vis()
        vf = QFormLayout()
        vf.setLabelAlignment(Qt.AlignLeft)
        self._color_btns = {}

        def add_row(label, key, tip=""):
            btn = self._color_btn(self._vis[key])
            if tip:
                btn.setToolTip(tip)
            self._color_btns[key] = btn
            vf.addRow(label, btn)

        # 类别框颜色**按类别表生成**：有几个类别就有几行，以后加类别这里自动多一行，
        # 不会出现「新类别没地方改颜色」。配置键 = 英文名 + "_color"（见 gui/theme.py）。
        for cid, en, zh, _bgr in classes.CLASSES:
            add_row("%s框颜色" % zh, "%s_color" % en,
                    "类别 %d（%s）—— 检测框颜色" % (cid, en))

        # 下面几条不是类别，是辅助线与标记
        add_row("锁定框颜色", "lock_color", "锁定的那个攻击目标")
        add_row("最大攻击距离线颜色", "attack_color", "")
        add_row("最小攻击距离线颜色", "min_attack_color", "规避范围")
        add_row("视野线颜色", "vision_color", "")

        self.sp_vision_width = NoWheelSpinBox()
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
        # 断线自动重连
        rc = self.ck_reconnect.isChecked()
        if rc != bool(settings.reconnect_enabled):
            settings.reconnect_enabled = rc
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
