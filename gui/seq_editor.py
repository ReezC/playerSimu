"""行为序列编辑器（通用）。

用于编辑一个有序的行为列表，每个元素是「某键按下」「某键松开」或
「额外延迟」。执行时每个键动作之间自动插入随机输入延迟，额外延迟是
固定的时间间隔。回身输出、以及其他行为配置都可复用此编辑器。

序列元素结构（和 decision.agent 的 back_jump_seq 一致）：
    {"type": "down",  "key": "left"}    键按下
    {"type": "up",    "key": "jump"}    键松开
    {"type": "delay", "ms": 200}        额外延迟
"""

from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QInputDialog, QLabel,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QVBoxLayout)

# 可选的键（显示名 → 序列键名）。back/forward 是特殊键，执行时按朝向解析。
SEQ_KEYS = [
    ("反方向", "back"),
    ("目标方向", "forward"),
    ("移动←", "left"),
    ("移动→", "right"),
    ("移动↑", "up"),
    ("移动↓", "down"),
    ("输出", "attack"),
    ("跳跃", "jump"),
    ("补血", "hp_pot"),
    ("补蓝", "mp_pot"),
    ("喂宠", "feed_pet"),
    ("商城", "shop"),
    ("回车", "enter"),
    ("Esc", "esc"),
]
_KEY_DISPLAY = dict(SEQ_KEYS)

# 键类型可视化配色：同键的「按下/松开」共色，不同键不同色，延迟单独灰。
_KEY_STYLE = {
    "back":     {"color": "#1a73e8", "bg": "#e8f0fe"},
    "forward":  {"color": "#1a73e8", "bg": "#e8f0fe"},
    "left":     {"color": "#1a73e8", "bg": "#e8f0fe"},
    "right":    {"color": "#1a73e8", "bg": "#e8f0fe"},
    "up":       {"color": "#1a73e8", "bg": "#e8f0fe"},
    "down":     {"color": "#1a73e8", "bg": "#e8f0fe"},
    "attack":   {"color": "#d93025", "bg": "#fce8e6"},
    "jump":     {"color": "#188038", "bg": "#e6f4ea"},
    "hp_pot":   {"color": "#c2185b", "bg": "#fce8ef"},
    "mp_pot":   {"color": "#7b1fa2", "bg": "#f3e8fd"},
    "feed_pet": {"color": "#e37400", "bg": "#fef3e0"},
    "shop":     {"color": "#00695c", "bg": "#e0f2f1"},
    "enter":    {"color": "#00838f", "bg": "#e0f7fa"},
    "esc":      {"color": "#455a64", "bg": "#eceff1"},
    "delay":    {"color": "#5f6368", "bg": "#f1f3f4"},
}

_QSS = """
QDialog { background: #ffffff; }
QLabel#tip { color: #5f6368; font-size: 12px; }
QListWidget {
    border: 1px solid #dadce0;
    border-radius: 8px;
    background: #fafbfc;
    font-size: 13px;
    selection-background-color: #cfe0fb;
    selection-color: #202124;
    outline: none;
}
QListWidget::item { padding: 5px 6px; }
QPushButton {
    border: 1px solid #dadce0;
    border-radius: 6px;
    padding: 7px 12px;
    background: #f8f9fa;
    color: #3c4043;
    font-size: 13px;
}
QPushButton:hover { background: #f1f3f4; }
QPushButton:pressed { background: #e8eaed; }
QPushButton:disabled { color: #9aa0a6; background: #f8f9fa; }
QPushButton#addKey { background: #e8f0fe; border-color: #aecbfa; color: #1967d2; font-weight: 600; }
QPushButton#addKey:hover { background: #d2e3fc; }
QPushButton#addDelay { background: #f1f3f4; border-color: #dadce0; color: #5f6368; font-weight: 600; }
QPushButton#addDelay:hover { background: #e8eaed; }
QPushButton#okBtn { background: #1a73e8; border-color: #1a73e8; color: #ffffff; font-weight: 600; }
QPushButton#okBtn:hover { background: #1765cc; }
"""


def _elem_text(elem):
    if elem["type"] == "delay":
        return "额外延迟 %dms" % int(elem.get("ms", 0))
    name = _KEY_DISPLAY.get(elem.get("key"), elem.get("key", "?"))
    return ("按下 %s" if elem["type"] == "down" else "松开 %s") % name


class SeqEditorDialog(QDialog):
    """行为序列编辑器。传入序列 list，确定后通过 seq() 取回。"""

    def __init__(self, seq, parent=None, title="行为编辑器"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 520)
        self.setStyleSheet(_QSS)
        self._seq = [dict(e) for e in seq]

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        tip = QLabel("每个键动作之间自动加入随机输入延迟；「额外延迟」是固定间隔。\n"
                     "「反方向 / 目标方向」会按角色当前朝向自动解析成 ← / →。\n"
                     "新增元素会插到当前选中项的下方；未选中则追加到末尾。")
        tip.setObjectName("tip")
        tip.setWordWrap(True)
        root.addWidget(tip)

        self.lst = QListWidget()
        root.addWidget(self.lst, 1)

        # 添加按钮：按类型着色
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        for text, obj, slot in (("＋键按下", "addKey", self._add_down),
                                ("＋键松开", "addKey", self._add_up),
                                ("＋额外延迟", "addDelay", self._add_delay)):
            b = QPushButton(text)
            b.setObjectName(obj)
            b.clicked.connect(slot)
            row1.addWidget(b)
        root.addLayout(row1)

        # 操作按钮
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        for text, slot in (("删除", self._del),
                           ("上移", self._up),
                           ("下移", self._down),
                           ("清空", self._clear)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row2.addWidget(b)
        root.addLayout(row2)

        # 确定 / 取消
        row3 = QHBoxLayout()
        row3.addStretch(1)
        btn_ok = QPushButton("确定")
        btn_ok.setObjectName("okBtn")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        row3.addWidget(btn_ok)
        row3.addWidget(btn_cancel)
        root.addLayout(row3)

        self._refresh()

    # ---------------- 序列操作 ----------------

    def _apply_style(self, item, elem):
        key = "delay" if elem["type"] == "delay" else elem.get("key")
        st = _KEY_STYLE.get(key) or {"color": "#3c4043", "bg": "#f8f9fa"}  # 自定义键默认灰
        item.setForeground(QBrush(QColor(st["color"])))
        item.setBackground(QBrush(QColor(st["bg"])))

    def _refresh(self):
        cur = self.lst.currentRow()
        self.lst.clear()
        for e in self._seq:
            item = QListWidgetItem(_elem_text(e))
            self._apply_style(item, e)
            self.lst.addItem(item)
        if 0 <= cur < self.lst.count():
            self.lst.setCurrentRow(cur)

    def _all_keys(self):
        """固定键 + 自定义按键，组成可选键列表 [(显示名, 键名), ...]。"""
        from decision.agent import settings
        keys = list(SEQ_KEYS)
        for name in settings.custom_keys:
            keys.append(("%s" % name, name))
        return keys

    def _pick_key(self):
        keys = self._all_keys()
        names = [d for d, _k in keys]
        name, ok = QInputDialog.getItem(self, "选择键", "选择键：", names, 0, False)
        if not ok:
            return None
        for d, k in keys:
            if d == name:
                return k
        return None

    def _insert(self, elem):
        """插入到当前选中项下方；未选中则追加到末尾，并选中新元素。"""
        row = self.lst.currentRow()
        if row < 0:
            self._seq.append(elem)
            self._refresh()
            self.lst.setCurrentRow(len(self._seq) - 1)
        else:
            self._seq.insert(row + 1, elem)
            self._refresh()
            self.lst.setCurrentRow(row + 1)

    def _add_down(self):
        key = self._pick_key()
        if key is None:
            return
        self._insert({"type": "down", "key": key})

    def _add_up(self):
        key = self._pick_key()
        if key is None:
            return
        self._insert({"type": "up", "key": key})

    def _add_delay(self):
        ms, ok = QInputDialog.getInt(self, "额外延迟", "延迟毫秒数：",
                                     200, 0, 100000, 10)
        if not ok:
            return
        self._insert({"type": "delay", "ms": ms})

    def _del(self):
        row = self.lst.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "先选中要删除的元素")
            return
        del self._seq[row]
        self._refresh()

    def _clear(self):
        self._seq.clear()
        self._refresh()

    def _up(self):
        row = self.lst.currentRow()
        if row <= 0:
            return
        self._seq[row], self._seq[row - 1] = self._seq[row - 1], self._seq[row]
        self._refresh()
        self.lst.setCurrentRow(row - 1)

    def _down(self):
        row = self.lst.currentRow()
        if row < 0 or row >= len(self._seq) - 1:
            return
        self._seq[row], self._seq[row + 1] = self._seq[row + 1], self._seq[row]
        self._refresh()
        self.lst.setCurrentRow(row + 1)

    def seq(self):
        return [dict(e) for e in self._seq]
