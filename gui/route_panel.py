"""路线识别面板：平台识别 / 跳跃落点预测 / 扫平台 的总开关。

这些功能每帧做图像处理（平台顶边检测、玩家运动追踪），比较吃 CPU，
默认关闭。需要时再开。
"""

from PyQt5.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from decision.agent import settings


class RoutePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("路线识别")
        title.setStyleSheet("font-weight: 600;")
        root.addWidget(title)

        note = QLabel("平台识别、跳跃落点预测、扫平台等寻路功能的总开关。\n"
                      "这些功能每帧做图像处理，比较吃 CPU，默认关闭。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f6368;")
        root.addWidget(note)

        self.ck_enabled = QCheckBox("启用路线识别")
        self.ck_enabled.setChecked(bool(settings.route_enabled))
        self.ck_enabled.toggled.connect(self._on_toggle)
        root.addWidget(self.ck_enabled)

        self.lbl_hint = QLabel()
        self.lbl_hint.setStyleSheet("color: #80868b;")
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint)
        self._refresh_hint()

        root.addStretch(1)

    def bind(self, project):
        """切项目：把开关刷成当前项目的值。

        `route_enabled` 也是一个决策参数，**按项目存**（见
        MainWindow._bind_decision_params），所以切项目必须重新读一遍，
        否则显示的是上一个项目的值。

        这里必须 blockSignals：不挡的话 setChecked 会触发 _on_toggle，
        把刚读出来的值又 save 回去 —— 而且 save 的是新项目，
        等于拿旧项目的开关覆盖新项目。
        """
        self.ck_enabled.blockSignals(True)
        self.ck_enabled.setChecked(bool(settings.route_enabled))
        self.ck_enabled.blockSignals(False)
        self._refresh_hint()

    def _refresh_hint(self):
        self.lbl_hint.setText(
            "已启用：实时预览会每帧检测平台顶边、预测跳跃落点（更耗 CPU）。"
            if self.ck_enabled.isChecked() else
            "已关闭：跳过平台识别和跳跃预测，实时预览更流畅。")

    def _on_toggle(self, on):
        settings.route_enabled = bool(on)
        settings.save()
        self._refresh_hint()
