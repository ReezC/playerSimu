"""决策参数面板（占位）。

角色识别已并入「模型训练」流程（YOLO 2 类检测），玩家选择挪到了
① 识别目标选项卡片。这个页签预留给后续的「决策参数」
（HFSM 阈值、攻击距离、逃跑 HP、技能 CD 等），现在只放占位说明。

保留 PlayerPanel 类名、verify_* 信号和 shutdown() 接口，是为了主窗口
（main_window.py）的既有连接和关闭清理不报错 —— 后续在这里填决策参数时
能直接复用这些挂点。
"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlayerPanel(QWidget):
    verify_started = pyqtSignal()            # 预留：决策参数改动触发主视区
    verify_result = pyqtSignal(object, str)  # 预留：(QPixmap 或 None, 说明文字)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("决策参数")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        root.addWidget(title)

        note = QLabel(
            "决策参数（HFSM 阈值、攻击距离、逃跑 HP、技能 CD 等）将在后续开发。\n\n"
            "角色识别已并入「模型训练」流程：\n"
            "  · ① 识别目标选项 里选择要训练的玩家模板\n"
            "  · ④ 自动标注 同时标怪物和玩家\n"
            "  · ⑦ 训练 产出 2 类检测模型\n"
            "  · 实时 里同时显示怪物框和角色框")
        note.setStyleSheet("color: #80868b;")
        note.setWordWrap(True)
        root.addWidget(note)
        root.addStretch(1)

    def shutdown(self):
        if self.thread is not None:
            self.thread.wait(2000)
