"""挑「要标注的玩家（角色）」的小弹窗（用户 2026-10-05 ✓ 原话："标注{类名}…要标注的{类名}…

    背包窗格（玩家、怪物也用同样的背包窗格的形式添加，怪物用stand或fly当icon，玩家用stand当icon）
    …背包窗格的添加、移出选中按钮"）。

⚠ 与 `gui/mob_picker.MobPickDialog` 的分工：那个是**多选**（一张图有好几种怪 ✓）；
  这个是**单选**（项目里就一个 `player_id` ✓）⇒ 别硬套那个（多选反而会让人以为能标好几个角色 ✗）。

⚠ 列表口径 = `datasets/sprites/player/` 下**有 png 的子目录**（与卡片 1 原来那个下拉**同一份** ✓
  —— 它 2026-10-05 搬进「标注玩家」段了 ✓）。
"""
from pathlib import Path

from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout)

from gui import theme

# ⚠ 这里用**普通 `QLineEdit`**：`gui.widgets` 里没有 `NoWheelLineEdit` ✗（本轮踩过）；
#   而 UI 规范那条"滚轮不许改参数"针对的是**数值控件**（spin/combo/slider ✓
#   见 `tools/check_ui.py::check_raw_widgets` ✓）—— 文本输入框不吃滚轮 ✓ 不受那条管 ✓。

PLAYER_ROOT = Path("datasets/sprites/player")


def list_players(root=None):
    """可用角色 id（**有 png 的子目录** ✓ 排序 ✓）—— 口径只此一处 ✓。"""
    root = Path(root or PLAYER_ROOT)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and any(d.glob("*.png")):
            out.append(d.name)
    return out


class PlayerPickDialog(QDialog):
    """单选一个角色 id ⇒ `result_id()` ✓。"""

    def __init__(self, current="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择要标注的角色")
        self.setMinimumSize(360, 420)
        self._ids = list_players()
        self._cur = str(current or "").strip()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        hint = QLabel("角色模板库：%s\n选一个角色 ⇒ 用它那套外观模板做「标注玩家」（class 0）✓"
                      % PLAYER_ROOT)
        hint.setStyleSheet("color:#80868b;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("筛选（按角色 id 搜）")
        self.ed_filter.textChanged.connect(lambda _t: self._fill())
        row.addWidget(self.ed_filter, 1)
        root.addLayout(row)

        self.lst = QListWidget()
        self.lst.setSelectionMode(QListWidget.SingleSelection)
        self.lst.itemDoubleClicked.connect(lambda _i: self.accept())
        root.addWidget(self.lst, 1)

        # ⚠ 这行必须**先建**再 `_fill()` ✗ —— `_fill()` 里要写它（本轮就栽在这：
        #   顺序反了 ⇒ 第一次填充直接 `AttributeError` ✓）
        self.lbl_n = QLabel("")
        self.lbl_n.setStyleSheet("color:#5f6368;")
        root.addWidget(self.lbl_n)
        self._fill()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("确定")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        theme.unify_ok_cancel(btns.button(QDialogButtonBox.Ok),
                              btns.button(QDialogButtonBox.Cancel))
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)
        theme.bind_window_state(self, "player_picker")    # 几何按客户端记住 ✓（UI规范 §11 ✓）

    def _fill(self):
        kw = (self.ed_filter.text() or "").strip().lower()
        self.lst.clear()
        for pid in self._ids:
            if kw and kw not in pid.lower():
                continue
            it = QListWidgetItem(pid)
            it.setData(256, pid)                      # Qt.UserRole
            self.lst.addItem(it)
            if pid == self._cur:
                self.lst.setCurrentItem(it)
        self.lbl_n.setText("共 %d 个角色%s" % (self.lst.count(),
                                              "（库里是空的：先用 WzProbe 导出角色模板 ✓）"
                                              if not self._ids else ""))

    def result_id(self):
        it = self.lst.currentItem()
        return str(it.data(256) or "") if it is not None else ""
