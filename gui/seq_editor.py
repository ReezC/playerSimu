"""行为序列编辑器（通用，支持嵌套）。

用于编辑一个有序的行为列表，每个元素是「某键按下」「某键松开」或
「额外延迟」。执行时每个键动作之间自动插入随机输入延迟，额外延迟是
固定的时间间隔。回身输出、输出行为、防掉线、自定义定时行为都可复用。

元素可带「执行几率」（右键→修改几率），非 100% 时界面显示百分比；
元素还可带「触发后执行」子列表（右键→编辑触发后行为），形成嵌套结构，
树形控件用缩进表示父子关系。

序列元素结构：
    {"type": "down",  "key": "left", "prob": 100, "then": [...]}   键按下
    {"type": "up",    "key": "jump"}                                键松开
    {"type": "delay", "ms": 200}                                    额外延迟

序列编辑核心抽成 SeqEditorWidget（可嵌入任意弹窗）；SeqEditorDialog 是
带确定/取消的独立弹窗版本，用于单独编辑一个序列；TimerEditDialog 是
自定义定时行为的一体化编辑弹窗（名字 + 时间区间 + 序列）。
"""

import copy

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QDialog, QFormLayout, QHBoxLayout, QInputDialog,
                             QLabel, QLineEdit, QMenu, QMessageBox, QPushButton,
                             QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

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
QTreeWidget {
    border: 1px solid #dadce0;
    border-radius: 8px;
    background: #fafbfc;
    font-size: 13px;
    selection-background-color: #cfe0fb;
    selection-color: #202124;
    outline: none;
}
QTreeWidget::item { padding: 5px 6px; }
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
        base = "额外延迟 %dms" % int(elem.get("ms", 0))
    else:
        name = _KEY_DISPLAY.get(elem.get("key"), elem.get("key", "?"))
        base = ("按下 %s" if elem["type"] == "down" else "松开 %s") % name
    prob = elem.get("prob", 100)
    if prob != 100:
        base += "  %d%%" % prob
    if elem.get("then"):
        base += "  ▸"
    return base


class SeqEditorWidget(QWidget):
    """行为序列编辑核心（树 + 按钮 + 编辑逻辑），可嵌入任意容器。"""

    def __init__(self, seq, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(0, 0, 0, 0)

        tip = QLabel("每个键动作之间自动加入随机输入延迟；「额外延迟」是固定间隔。\n"
                     "「反方向 / 目标方向」会按角色当前朝向自动解析成 ← / →。\n"
                     "右键元素可「修改几率」「编辑触发后执行」；子元素缩进显示。\n"
                     "新增元素会插到当前选中项的同级下方；未选中则追加到顶层末尾。")
        tip.setObjectName("tip")
        tip.setWordWrap(True)
        root.addWidget(tip)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        root.addWidget(self.tree, 1)

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

        # 初始化树（深拷贝，避免污染传入序列）
        self._add_children(self.tree.invisibleRootItem(), copy.deepcopy(seq))
        self.tree.expandAll()

    # ---------------- 树 ↔ 序列 ----------------

    def _apply_style(self, item, elem):
        key = "delay" if elem["type"] == "delay" else elem.get("key")
        st = _KEY_STYLE.get(key) or {"color": "#3c4043", "bg": "#f8f9fa"}  # 自定义键默认灰
        item.setForeground(0, QBrush(QColor(st["color"])))
        item.setBackground(0, QBrush(QColor(st["bg"])))

    def _add_children(self, parent_item, seq):
        for elem in seq:
            item = QTreeWidgetItem(parent_item)
            item.setText(0, _elem_text(elem))
            item.setData(0, Qt.UserRole, elem)
            self._apply_style(item, elem)
            then = elem.get("then")
            if isinstance(then, list) and then:
                self._add_children(item, then)

    def _collect(self, parent_item):
        out = []
        for i in range(parent_item.childCount()):
            item = parent_item.child(i)
            elem = dict(item.data(0, Qt.UserRole))
            sub = self._collect(item)
            if sub:
                elem["then"] = sub
            else:
                elem.pop("then", None)
            out.append(elem)
        return out

    def seq(self):
        return self._collect(self.tree.invisibleRootItem())

    # ---------------- 序列操作 ----------------

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
        """插入到当前选中项的同级下方；未选中则追加到顶层末尾。"""
        item = self.tree.currentItem()
        if item is None:
            parent = self.tree.invisibleRootItem()
            row = parent.childCount()
        else:
            parent = item.parent() or self.tree.invisibleRootItem()
            row = parent.indexOfChild(item) + 1
        new_item = QTreeWidgetItem(parent)
        new_item.setText(0, _elem_text(elem))
        new_item.setData(0, Qt.UserRole, elem)
        self._apply_style(new_item, elem)
        parent.insertChild(row, new_item)
        self.tree.setCurrentItem(new_item)
        if parent is not self.tree.invisibleRootItem():
            self.tree.expandItem(parent)

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
        item = self.tree.currentItem()
        if item is None:
            QMessageBox.information(self, "提示", "先选中要删除的元素")
            return
        parent = item.parent() or self.tree.invisibleRootItem()
        parent.removeChild(item)

    def _clear(self):
        self.tree.clear()

    def _up(self):
        item = self.tree.currentItem()
        if item is None:
            return
        parent = item.parent() or self.tree.invisibleRootItem()
        row = parent.indexOfChild(item)
        if row <= 0:
            return
        parent.takeChild(row)
        parent.insertChild(row - 1, item)
        self.tree.setCurrentItem(item)

    def _down(self):
        item = self.tree.currentItem()
        if item is None:
            return
        parent = item.parent() or self.tree.invisibleRootItem()
        row = parent.indexOfChild(item)
        if row < 0 or row >= parent.childCount() - 1:
            return
        parent.takeChild(row)
        parent.insertChild(row + 1, item)
        self.tree.setCurrentItem(item)

    # ---------------- 右键菜单 ----------------

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        menu = QMenu(self)
        act_prob = menu.addAction("修改几率")
        act_prob.triggered.connect(self._edit_prob)
        act_then = menu.addAction("编辑触发后行为")
        act_then.triggered.connect(self._edit_then)
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def _edit_prob(self):
        item = self.tree.currentItem()
        if item is None:
            return
        elem = dict(item.data(0, Qt.UserRole))   # data() 返回副本，改完要写回
        cur = int(elem.get("prob", 100))
        prob, ok = QInputDialog.getInt(self, "修改几率", "执行几率（%）：",
                                       cur, 0, 100, 5)
        if not ok:
            return
        if prob >= 100:
            elem.pop("prob", None)
        else:
            elem["prob"] = prob
        item.setData(0, Qt.UserRole, elem)        # 写回
        item.setText(0, _elem_text(elem))

    def _edit_then(self):
        """编辑当前元素的「触发后执行」子列表。"""
        item = self.tree.currentItem()
        if item is None:
            return
        elem = dict(item.data(0, Qt.UserRole))   # data() 返回副本，改完要写回
        dlg = SeqEditorDialog(elem.get("then", []), self, title="触发后执行")
        if not dlg.exec_():
            return
        sub = dlg.seq()
        if sub:
            elem["then"] = sub
        else:
            elem.pop("then", None)
        item.setData(0, Qt.UserRole, elem)        # 写回
        # 重建该节点的子节点
        item.takeChildren()
        if sub:
            self._add_children(item, sub)
        item.setText(0, _elem_text(elem))
        self.tree.expandItem(item)


class SeqEditorDialog(QDialog):
    """行为序列编辑器（独立弹窗）。传入序列 list，确定后通过 seq() 取回。"""

    def __init__(self, seq, parent=None, title="行为编辑器"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(460, 560)
        self.setStyleSheet(_QSS)

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        self.editor = SeqEditorWidget(seq, self)
        root.addWidget(self.editor, 1)

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

    def seq(self):
        return self.editor.seq()


class TimerEditDialog(QDialog):
    """自定义定时行为的一体化编辑弹窗：名字 + 时间区间 + 行为序列。

    timer 为 None 表示新增；否则传入现有 dict 预填。
    确定后通过 result() 取回 {name, interval:[lo,hi], seq}。
    """

    _DEFAULT_SEQ = [{"type": "down", "key": "attack"}, {"type": "up", "key": "attack"}]

    def __init__(self, timer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑定时行为" if timer else "添加定时行为")
        self.resize(480, 620)
        self.setStyleSheet(_QSS)

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        # 名字 + 时间区间
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("例如：自动喊话")
        self.sp_lo = QSpinBox()
        self.sp_lo.setRange(1, 600)
        self.sp_lo.setSuffix(" 分钟")
        self.sp_hi = QSpinBox()
        self.sp_hi.setRange(1, 600)
        self.sp_hi.setSuffix(" 分钟")
        form.addRow("行为名", self.ed_name)
        form.addRow("触发下限", self.sp_lo)
        form.addRow("触发上限", self.sp_hi)
        root.addLayout(form)

        # 序列编辑核心（嵌入）
        seq = (timer.get("seq") or list(self._DEFAULT_SEQ)) if timer else list(self._DEFAULT_SEQ)
        self.editor = SeqEditorWidget(seq, self)
        root.addWidget(self.editor, 1)

        # 确定 / 取消
        row = QHBoxLayout()
        row.addStretch(1)
        btn_ok = QPushButton("确定")
        btn_ok.setObjectName("okBtn")
        btn_ok.clicked.connect(self._on_ok)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        root.addLayout(row)

        # 预填
        if timer:
            self.ed_name.setText(timer.get("name", ""))
            lo, hi = timer.get("interval", [5, 10])
            self.sp_lo.setValue(lo)
            self.sp_hi.setValue(max(lo, hi))
        else:
            self.sp_lo.setValue(5)
            self.sp_hi.setValue(10)
        self.sp_lo.valueChanged.connect(self._clamp_hi)
        self._clamp_hi()

    def _clamp_hi(self):
        if self.sp_hi.value() < self.sp_lo.value():
            self.sp_hi.setValue(self.sp_lo.value())

    def _on_ok(self):
        if not self.ed_name.text().strip():
            QMessageBox.warning(self, "提示", "行为名不能为空")
            return
        self.accept()

    def result(self):
        return {
            "name": self.ed_name.text().strip(),
            "interval": [self.sp_lo.value(), self.sp_hi.value()],
            "seq": self.editor.seq(),
        }
