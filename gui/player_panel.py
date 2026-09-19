"""决策参数面板：自动打怪的开关、攻击距离、键盘映射。

决策逻辑在 decision/agent.py，实时推理循环驱动它。这里只负责：
    · 开关自动（按钮 + F11）
    · 攻击距离
    · 键盘映射（7 个键）

所有改动直接写进全局 settings（decision.agent.settings），实时线程每帧
读到最新值，无需重启。
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QVBoxLayout, QWidget)

from decision import input as dinput
from decision.agent import settings

# 键盘映射的每一行：(key, 标签, 默认键名)
# 注意："auto" 就是「开启自动」按钮的快捷键，默认 F11，和按钮/全局热键是同一个东西。
KEY_ROWS = [
    ("auto", "开关自动", "f11"),
    ("left", "移动←", "left"),
    ("right", "移动→", "right"),
    ("up", "移动↑", "up"),
    ("down", "移动↓", "down"),
    ("attack", "输出", "ctrl"),
    ("jump", "跳跃", "alt"),
]


class PlayerPanel(QWidget):
    verify_started = pyqtSignal()            # 保留：主窗口既有连接
    verify_result = pyqtSignal(object, str)  # 保留：主窗口既有连接
    auto_key_changed = pyqtSignal(str)       # 「开关自动」映射变了 → 主窗口重新注册全局热键

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self._build()
        self._sync_from_settings()

    # ---------------- 界面 ----------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("决策参数")
        title.setStyleSheet("font-weight: 600; font-size: 14px; color: #202124;")
        root.addWidget(title)

        # ---- 按钮组 ----
        self.btn_auto = QPushButton("开启自动（F11）")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setMinimumHeight(34)
        self.btn_auto.clicked.connect(self._toggle_auto)
        root.addWidget(self.btn_auto)

        self.lbl_state = QLabel("当前：未开启")
        self.lbl_state.setStyleSheet("color: #80868b;")
        root.addWidget(self.lbl_state)

        # ---- 参数组 ----
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)

        self.sp_attack = self._spin(0, 2000, 80, 0)
        self.sp_attack.valueChanged.connect(self._on_attack_dist)
        form.addRow("攻击距离", self.sp_attack)

        # 目标切换 CD [min, max]（毫秒）
        self.sp_cd_min = self._spin(0, 10000, 500, 0)
        self.sp_cd_max = self._spin(0, 10000, 1000, 0)
        self.sp_cd_min.valueChanged.connect(self._on_target_cd)
        self.sp_cd_max.valueChanged.connect(self._on_target_cd)
        cd_row = QHBoxLayout()
        cd_row.addWidget(self.sp_cd_min)
        cd_row.addWidget(QLabel("~"))
        cd_row.addWidget(self.sp_cd_max)
        cd_row.addWidget(QLabel("ms"))
        form.addRow("目标切换CD", cd_row)

        root.addLayout(form)

        # ---- 键盘映射组 ----
        grp = QGroupBox("键盘映射")
        g = QFormLayout(grp)
        g.setLabelAlignment(Qt.AlignLeft)

        self._key_combos = {}
        for key, label, default in KEY_ROWS:
            cb = self._key_combo()
            cb.currentIndexChanged.connect(
                lambda _i, k=key, c=cb: self._on_key(k, c))
            self._key_combos[key] = cb
            g.addRow(label, cb)

        root.addWidget(grp)
        root.addStretch(1)

        # F11 开关自动（窗口内快捷键；全局热键后续接 RegisterHotKey）
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------------- 控件构造 ----------------

    @staticmethod
    def _spin(lo, hi, val, decimals):
        from gui.widgets import NoWheelDoubleSpinBox
        w = NoWheelDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(val)
        return w

    @staticmethod
    def _key_combo():
        cb = QComboBox()
        for key in dinput.CHOICES:
            cb.addItem(dinput.display_name(key), key)
        return cb

    # ---------------- 事件 ----------------

    def _toggle_auto(self, checked=None):
        on = self.btn_auto.isChecked()
        settings.enabled = on
        self._refresh_auto_ui()

    def _refresh_auto_ui(self):
        on = settings.enabled
        hotkey = dinput.display_name(settings.keymap.get("auto", "f11"))
        self.btn_auto.setChecked(on)
        self.btn_auto.setText("停止自动（%s）" % hotkey if on else "开启自动（%s）" % hotkey)
        self.lbl_state.setText("当前：自动打怪运行中" if on else "当前：未开启")

    def _on_attack_dist(self, val):
        settings.attack_dist = float(val)
        settings.save()

    def _on_target_cd(self, _val=None):
        settings.target_cd = [int(self.sp_cd_min.value()), int(self.sp_cd_max.value())]
        settings.save()

    def _on_key(self, key, cb):
        settings.set_key(key, cb.currentData())
        settings.save()
        if key == "auto":
            # 「开关自动」就是开启按钮的快捷键：改了映射要同步按钮文字 +
            # 通知主窗口重新注册全局热键。
            self._refresh_auto_ui()
            self.auto_key_changed.emit(cb.currentData())

    def _sync_from_settings(self):
        """启动时把 settings 里的值灌到控件上。"""
        self.sp_attack.blockSignals(True)
        self.sp_attack.setValue(settings.attack_dist)
        self.sp_attack.blockSignals(False)

        lo, hi = settings.target_cd
        self.sp_cd_min.blockSignals(True)
        self.sp_cd_min.setValue(lo)
        self.sp_cd_min.blockSignals(False)
        self.sp_cd_max.blockSignals(True)
        self.sp_cd_max.setValue(hi)
        self.sp_cd_max.blockSignals(False)

        for key, cb in self._key_combos.items():
            cb.blockSignals(True)
            idx = cb.findData(settings.keymap.get(key))
            if idx >= 0:
                cb.setCurrentIndex(idx)
            cb.blockSignals(False)

        self._refresh_auto_ui()

    def toggle_auto(self):
        """切换自动开关（由全局热键触发）。"""
        self.btn_auto.setChecked(not self.btn_auto.isChecked())
        self._toggle_auto()

    # ---------------- 清理 ----------------

    def shutdown(self):
        settings.enabled = False
        if self.thread is not None:
            self.thread.wait(2000)
