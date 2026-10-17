"""行为序列编辑器（通用，支持嵌套）。

用于编辑一个有序的行为列表，每个元素是「某键按下」「某键松开」或
「额外延迟」。执行时**一帧推进一步**（元素之间不再插随机延迟，2026-09-27 删掉），
所以「额外延迟」才是那个固定的时间间隔。回身输出、输出行为、防掉线、自定义定时行为都可复用。

元素可带「执行几率」（右键→修改几率），非 100% 时界面显示百分比；
元素还可带「触发后执行」子列表（右键→编辑触发后行为），形成嵌套结构，
树形控件用缩进表示父子关系。

序列元素结构：
    {"type": "down",  "key": "left", "prob": 100, "then": [...]}   键按下
    {"type": "up",    "key": "jump"}                                键松开
    {"type": "delay", "ms": 200}                                    额外延迟

⚠ 「**以秒为单位**」那个勾选**不在这个文件**（2026-10-15 挪走了 ✓）：它只属于
「**添加 / 编辑定时行为**」那个弹窗（本文件末尾的 `TimerEditDialog` ✓），控制的是
「**触发下限 / 触发上限**」的单位（`min` ⇄ `s`），与上面这个「额外延迟」**无关** ✗ ——
用户 2026-10-15 ✓ 原话："**不是所有的行为编辑器都要加「以秒为单位」，只在添加定时行为弹窗里加，
它只是控制「触发下限(分钟)」「触发上限(分钟)」的单位**"✓。

序列编辑核心抽成 SeqEditorWidget（可嵌入任意弹窗）；SeqEditorDialog 是
带确定/取消的独立弹窗版本，用于单独编辑一个序列；TimerEditDialog 是
自定义定时行为的一体化编辑弹窗（名字 + 时间区间 + 序列）。
"""

import copy
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QCheckBox, QDialog, QFileDialog, QFormLayout,
                             QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu,
                             QMessageBox, QPushButton, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from core import seq_presets as sp             # 预设的存取（纯逻辑，见那边的判据）
from core.seq_presets import MAX_DELAY_MS      # 延迟上限的真源搬到了 core（数据不是 UI）
from gui import theme                          # 弹窗几何按客户端存（config/ui.yaml）✓
from gui.widgets import NoWheelDoubleSpinBox   # 必须模块级：控件在 __init__ 里建

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
# ⚠ 方向**是键名 → 显示名**（`_elem_text` 就是那么用的：`_KEY_DISPLAY.get(key)`）——
# 原来写成 `dict(SEQ_KEYS)` 了（那是显示名 → 键名 ✗）⇒ 查谁都查不到 ⇒ 界面上一直显示**裸键名**
# （`按下 esc` / `按下 attack` ✗，本该是「按下 Esc」「按下 输出」）。加「双击改键」时用例当场
# 抓到（`键改了但行文本没变：'按下 esc'` ✓）。别再把方向反过来 ✗。
_KEY_DISPLAY = {k: d for d, k in SEQ_KEYS}

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
QPushButton#savePreset { background: #e6f4ea; border-color: #a8dab5; color: #137333; font-weight: 600; }
QPushButton#savePreset:hover { background: #ceead6; }
QPushButton#loadPreset { background: #fef7e0; border-color: #fadf8e; color: #a05c00; font-weight: 600; }
QPushButton#loadPreset:hover { background: #feefc3; }
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
    """行为序列编辑核心（树 + 按钮 + 编辑逻辑），可嵌入任意容器。

    `preset_name`：**保存**预设时的默认文件名（不传就是 `行为预设.json`）——
    由行为名 / 弹窗标题带进来，省得每存一次都要重新打一遍名字。
    """

    def __init__(self, seq, parent=None, preset_name=None):
        super().__init__(parent)
        self.preset_name = sp.safe_name(preset_name or "行为预设")
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(0, 0, 0, 0)

        # 预设：保存 / 加载（2026-09-26 用户要求，T1）。放**最上面**，和"往序列里
        # 加东西"那一排分开 —— 这是整条序列的存读，不是改某一个元素。
        row_save = QHBoxLayout()
        row_save.setSpacing(8)
        self.btn_save = QPushButton("保存")
        self.btn_save.setObjectName("savePreset")
        self.btn_save.setToolTip("把当前整条序列存成预设文件（人能读的 JSON；\n"
                                 "默认放 config/sequences/，也可以自己另选路径）")
        self.btn_save.clicked.connect(self._save_preset)
        self.btn_load = QPushButton("加载")
        self.btn_load.setObjectName("loadPreset")
        self.btn_load.setToolTip("从预设文件读回一条序列，**整体替换**当前内容。\n"
                                 "文件坏了只会报错，当前序列一个字都不会动。")
        self.btn_load.clicked.connect(self._load_preset)
        row_save.addWidget(self.btn_save)
        row_save.addWidget(self.btn_load)
        row_save.addStretch(1)
        root.addLayout(row_save)

        tip = QLabel("一帧推进一步（元素之间没有随机延迟了）；「额外延迟」才是固定间隔。\n"
                     "「反方向 / 目标方向」会按角色当前朝向自动解析成 ← / →。\n"
                     "加「键按下」会**自动配一条同键的「键松开」**；\n"
                     "右键元素可「修改几率」「编辑触发后执行」；"
                     "**双击**元素按类型进编辑（键 / 毫秒）；子元素缩进显示。\n"
                     "新增元素会插到当前选中项的同级下方；未选中则追加到顶层末尾。")
        tip.setObjectName("tip")
        tip.setWordWrap(True)
        root.addWidget(tip)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        # **双击一个元素 ⇒ 按它的类型弹对应的编辑窗**（用户 2026-09-26 要求）——
        # 和「可到达」列表双击 = 改那条边是同一个手势（见 `gui/zone_editor.py`）✓。
        self.tree.itemDoubleClicked.connect(self._on_double_click)
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
        self.set_seq(seq)

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

    def set_seq(self, seq):
        """**整体替换**树里的序列（加载预设用）。

        传进来的东西调用方已经校验过；这里仍然深拷贝一份 —— 树里的元素随时会被右键改
        （`_edit_prob` / `_edit_then` 都是"取副本、改完写回"），不能让外面那份跟着变。
        """
        self.tree.clear()
        self._add_children(self.tree.invisibleRootItem(),
                           copy.deepcopy(list(seq or [])))
        self.tree.expandAll()

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
        """加一个「键按下」⇒ **自动在它下面配一个同键的「键松开」**（用户 2026-09-26 要求）。

        为什么：只按不松是最常见的坑（键卡住、后面的动作按不出来 ✗），而手敲一对本来就要
        点两次、选两次。位置不用自己算：`_insert` 插完会把新项**选中** ✓ ⇒ 紧接着插的
        「松开」正好落在它下面 ⇒ 天然的 down → up →（下一条）✓。
        """
        key = self._pick_key()
        if key is None:
            return
        self._insert({"type": "down", "key": key})
        self._insert({"type": "up", "key": key})     # ← 紧跟一条**同键**的松开 ✓

    def _add_up(self):
        key = self._pick_key()
        if key is None:
            return
        self._insert({"type": "up", "key": key})

    def _add_delay(self):
        ms, ok = QInputDialog.getInt(
            self, "额外延迟", "延迟毫秒数（1000 = 1 秒，60000 = 1 分钟）：",
            200, 0, MAX_DELAY_MS, 10)
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

    # ---------------- 预设：保存 / 加载 ----------------

    def _save_preset(self):
        """整条序列存成预设文件（默认 `config/sequences/<默认名>.json`）。

        校验不过 / 写不进去 ⇒ **只报错**，界面上的序列一个字不动。
        """
        d = sp.default_dir()
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as ex:                     # 目录建不了（权限 / 盘满）
            QMessageBox.warning(self, "保存失败", "建不了目录：\n%s\n%s" % (d, ex))
            return
        start = str(d / ("%s.json" % self.preset_name))
        path, _flt = QFileDialog.getSaveFileName(self, "保存行为预设", start,
                                                 "预设文件 (*.json)")
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")            # 没打后缀就补上（否则下次"加载"里看不到）
        try:
            info = sp.save_preset(p, self.seq(), name=p.stem)
        except (OSError, ValueError) as ex:
            QMessageBox.warning(self, "保存失败",
                                "没写成功（**当前序列没有改动**）：\n%s" % ex)
            return
        QMessageBox.information(self, "已保存",
                                "存到：\n%s\n（%d 个元素）"
                                % (p, sp.count(info["seq"])))

    def _load_preset(self):
        """从预设文件读回一条序列（**整体替换**当前内容）。"""
        d = sp.default_dir()
        path, _flt = QFileDialog.getOpenFileName(
            self, "加载行为预设", str(d if d.is_dir() else sp.ROOT),
            "预设文件 (*.json)")
        if not path:
            return
        self._apply_preset_file(path)

    def _apply_preset_file(self, path):
        """真正做事的那一半（**不弹文件框**，便于自检直接喂路径）⇒ 成功 True。

        坏文件 ⇒ 警告 + `False`，**当前序列一个字都不动**。
        判据来自用户 2026-09-26：静默清空 = 以为加载成功了、其实序列没了
        ⇒ 按键行为**直接变了** ✗（那种坑比报错难查一百倍）。
        """
        try:
            info = sp.load_preset(path)
        except (OSError, ValueError) as ex:
            QMessageBox.warning(self, "加载失败",
                                "这个文件不能用，当前序列**没有改动**：\n%s" % ex)
            return False
        if self.seq() and QMessageBox.question(
                self, "确认加载",
                "加载会**整体替换**当前序列（现在有 %d 个元素），继续？"
                % sp.count(self.seq())) != QMessageBox.Yes:
            return False
        self.set_seq(info["seq"])
        self.preset_name = sp.safe_name(info["name"])
        QMessageBox.information(self, "已加载",
                                "读了「%s」：%d 个元素%s"
                                % (info["name"], sp.count(info["seq"]),
                                   ("\n（存于 %s）" % info["saved_at"]
                                    if info["saved_at"] else "")))
        return True

    # ---------------- 双击：按类型进编辑 ----------------

    def _on_double_click(self, item, _col=0):
        """**双击一个元素** ⇒ 按它的**类型**弹对应的编辑窗（用户 2026-09-26 要求）。

        · `delay`       ⇒ 改毫秒数；
        · `down` / `up` ⇒ 改键（重选一个键）。
        这两个"改"以前**只能删了重加** ✗（右键菜单里只有几率 / 触发后执行 ✓）。

        改完就地更新那一条的**三处**：存储的字典 + 行文本 + 配色 ✓ ——
        `item.data()` 返回的是**副本**，忘记 `setData` 写回就是"看着改了、其实没改" ✗。
        """
        if item is None:
            return
        elem = dict(item.data(0, Qt.UserRole))
        if elem.get("type") == "delay":
            ms, ok = QInputDialog.getInt(
                self, "额外延迟", "延迟毫秒数（1000 = 1 秒，60000 = 1 分钟）：",
                int(elem.get("ms", 0)), 0, MAX_DELAY_MS, 10)
            if not ok:
                return
            elem["ms"] = ms
        else:
            key = self._pick_key()
            if key is None:
                return
            elem["key"] = key
        item.setData(0, Qt.UserRole, elem)        # 写回（data() 是副本）
        item.setText(0, _elem_text(elem))
        self._apply_style(item, elem)

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

        # 默认预设名跟着弹窗标题走（"回身输出" ⇒ 存出来就是 回身输出.json）
        self.editor = SeqEditorWidget(seq, self, preset_name=title)
        root.addWidget(self.editor, 1)

        row3 = QHBoxLayout()
        row3.addStretch(1)
        btn_ok = QPushButton("确定")
        btn_ok.setObjectName("okBtn")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消")
        theme.unify_ok_cancel(btn_ok, btn_cancel)
        btn_cancel.clicked.connect(self.reject)
        row3.addWidget(btn_ok)
        row3.addWidget(btn_cancel)
        root.addLayout(row3)
        theme.bind_window_state(self, "seq_editor")    # 拉过的大小/位置按客户端记住 ✓

    def seq(self):
        return self.editor.seq()


#: 「触发下限 / 上限」的**真源单位 = 分钟** ✓（同 `custom_timers[].interval`，
#: 见 `decision/agent.py` 里 `random.uniform(lo, hi) * 60.0` ✓）。
#: 界面上的 `min` / `s` 只是**显示与填写的口径** ✓ —— 勾「以秒为单位」换的只是它 ✓。
TIMER_LO_MIN, TIMER_HI_MIN = 0.01, 600.0       # 分钟（= 0.6 秒 ~ 10 小时 ✓ 见下面那段）
TIMER_DEFAULT_LO, TIMER_DEFAULT_HI = 5.0, 10.0  # 新增时的默认区间（分钟 ✓ 同老行为 ✓）
#: ⭐⭐ **秒口径的下限（秒 ✓）**（用户 2026-10-16 ✓ 原话："**编辑定时行为的触发下限、上限
#: 不能配置 6.0 以下**"✓）。
#: ⚠⚠ **先把"为什么原来正好卡在 6.0"记下来**（免得后人又"顺手放宽"✗）：
#:   真源是**分钟** ✓、而控件原来只留 **1 位小数** ✓ ⇒ 最小可表示的一格 = **0.1 分钟 = 6 秒** ✓
#:   ⇒ 秒口径 `× 60` 之后就成了 **6.0** ✓ —— **那个 6 不是谁定的，是"存储精度"的产物** ✓
#:   （注释里原先还写着"不许放宽"✗ —— 那条是把副作用当成了设定 ✓）。
#: ⇒ 要真能配 6 秒以下，**两处都得动** ✓：① 分钟口径小数位 1 → 3
#:   （最小 0.001 分钟 ≈ 0.06 秒 ✓ 够用 ✓ 见 `TIMER_DECIMALS`）；② 秒口径下限单独给
#:   （**1.0 秒** ✓ 见 `TIMER_LO_SEC`）—— 只动一处都会"填得进、存不下"✗。
#: ⚠ 为什么不干脆"0.1 秒起"✗：实时回路一拍 ~15~30fps ✓ ⇒ 亚秒定时 = **每拍都按** ✗
#:   那不是配置、是抖 ✓ ⇒ 下限卡在 **1 秒** ✓（要更小得先说明用途 ✓）。
#: ⚠ 分钟口径也给了更小的下限（0.01 分钟 ≈ 0.6 秒 ✓），但那要**手打**（步进仍是 0.1 ✓
#:   保住老手感 ✓）；要填秒数就勾「以秒为单位」✓ 那才是它的用途 ✓。
TIMER_DECIMALS = 3
TIMER_LO_SEC = 1.0                              # 秒口径下限（秒 ✓ 见上面那段）

#: 「以秒为单位」这个**本机偏好**存哪：`config/ui.yaml` 的**顶层** `timer_edit_prefs` 段 ✓
#: （同 `gui/theme.load_section` / `save_section` 那套"窗口自己的上次配置"✓）。
#: ⚠ 段名**别叫 `timer_edit`** ✗：那个名字已经被「**窗口几何**」占着当键了（在 `windows:` 段
#:   底下 ✓，见 `gui/theme.py` 里负责把弹窗几何记进 `windows:` 的那个函数 ✓）—— 同名不同层，
#:   将来查配置的人一定会看错 ✓。加 `_prefs` 后缀区分开 ✓。
#: ⚠⚠ **别把那个函数名原样写进这段注释** ✗：本文件里有一条**按出现次数数**的源码钉 ——
#:   注释里抄一遍就多一次、当场假红 ✓（上一版就是这么被抓住的 ✓ 见开发日志 341/342 ✓）。
_UI_SECTION = "timer_edit_prefs"
_UI_SECONDS_KEY = "interval_in_seconds"


class TimerEditDialog(QDialog):
    """自定义定时行为的一体化编辑弹窗：名字 + 时间区间 + 行为序列。

    timer 为 None 表示新增；否则传入现有 dict 预填。
    确定后通过 result() 取回 {name, interval:[lo,hi], seq} —— **`interval` 永远是分钟** ✓。

    ⭐ 「**以秒为单位**」（用户 2026-10-15 ✓ 原话："**不是所有的行为编辑器都要加「以秒为单位」，
    只在添加定时行为弹窗里加，它只是控制「触发下限(分钟)」「触发上限(分钟)」的单位**"✓）：
    勾上 ⇒「触发下限 / 上限」按**秒**填、行标签变成 `(s)`；不勾 ⇒ 按**分钟**（老口径 ✓）。
    ⚠ 它**只管界面** ✗：`interval` 存下去**永远是分钟** ✓ ⇒ 切来切去**不会改动已配好的值** ✓。
    """

    _DEFAULT_SEQ = [{"type": "down", "key": "attack"}, {"type": "up", "key": "attack"}]

    def __init__(self, timer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑定时行为" if timer else "添加定时行为")
        self.resize(480, 620)
        self.setStyleSheet(_QSS)
        # ⭐ 跟**其它弹窗同一套**（用户 2026-09-28："居中、放弃位置记忆，我期望的是调整过的
        #   缩放数据记忆要有"✓ ⇒ 每次显示都居中到主窗口 ✓、尺寸沿用拉过的那份 ✓）。
        #   ⚠ 这个窗原来**漏在规范外**（既没接几何、也没人发现 ✗）—— 这一轮补齐 ✓。
        theme.bind_window_state(self, "timer_edit")

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        # 名字 + 时间区间
        form = QFormLayout()
        self.form = form                        # 换单位时要改**行标签**的文案 ⇒ 得留一份 ✓
        form.setLabelAlignment(Qt.AlignLeft)
        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("例如：自动喊话")
        # 分钟支持小数：用 DoubleSpinBox（和参数面板同一套，滚轮不误触）
        # ⚠ 这两个控件的**单位会变**（min ⇄ s ✓ 见 `set_seconds_mode`）⇒ 范围 / 小数位 / 步进 /
        #   行标签一律**不在这里写死**，全部交给 `set_seconds_mode()` **一处**设 ✓
        #   （两处各写一份必然分叉 ✗）。
        self.sp_lo = NoWheelDoubleSpinBox()
        self.sp_hi = NoWheelDoubleSpinBox()
        #: ⭐ **真源（分钟 ✓** 与 `custom_timers[].interval` 同口径 ✓）—— 控件里放的只是
        #:   "当前单位下的那个数" ✓；换单位**只重算显示**、不动这里 ✓（切来切去不会改值 ✓）。
        self._lo_min = TIMER_DEFAULT_LO
        self._hi_min = TIMER_DEFAULT_HI
        self._seconds = False
        # 单位写在**行标签**里（UI 规范 §9②：不写进编辑框 ✓；§6：单位用 `(min)` / `(s)` 短写法 ✓）
        form.addRow("行为名", self.ed_name)
        form.addRow("触发下限", self.sp_lo)
        form.addRow("触发上限", self.sp_hi)
        # ⭐ 「以秒为单位」（用户 2026-10-15 ✓）：**只属于这个弹窗** ✓（通用序列编辑器里没有 ✗），
        #   而且只管下面那两条的**单位** ✓。
        self.ck_seconds = QCheckBox("以秒为单位")
        self.ck_seconds.setToolTip(
            "勾上：下面两条按**秒**填（`30` = 30 秒），单位变成 `(s)`；\n"
            "不勾（默认）：按**分钟**填（可写小数，`0.5` = 30 秒），单位是 `(min)`。\n\n"
            "⚠ 只改**填法 / 显示**：定时行为存下来**永远是分钟** ⇒ 跟面板上的倒计时、\n"
            "项目文件同一口径，切来切去**不会改动已配好的值**。\n"
            "这个勾选记在**本机**（config/ui.yaml），跟项目无关。")
        form.addRow("", self.ck_seconds)
        # 「休息时暂停计时」（用户 2026-09-26 要求）—— **每条行为各自一份**的开关：
        #   勾上（默认）= 防掉线休息期间这条**冻住计时**（老行为 ✓，剩余时间不走）；
        #   不勾 = 休息期间**照常倒数、到点照演** ✓（给"休息时也得按的键"用：喂宠 / 喊话…）。
        self.ck_pause_rest = QCheckBox("休息时暂停计时")
        self.ck_pause_rest.setToolTip(
            "勾上（默认）：防掉线休息期间**这条行为冻住计时**\n"
            "（剩余时间不走，休息结束后接着倒 ✓）。\n\n"
            "不勾：休息期间它**照常倒计时、到点照演** —— 给「休息时也得按的键」用\n"
            "（喂宠、喊话、报点之类）。⚠ 它和休息自己的序列是**各按各的**（各自按自己的\n"
            "时刻表走 ✓，同一根键被两边同时操作时没有仲裁 ✗）。")
        form.addRow("", self.ck_pause_rest)
        root.addLayout(form)

        # 序列编辑核心（嵌入）
        seq = (timer.get("seq") or list(self._DEFAULT_SEQ)) if timer else list(self._DEFAULT_SEQ)
        self.editor = SeqEditorWidget(seq, self, preset_name="定时行为")
        root.addWidget(self.editor, 1)

        # 确定 / 取消
        row = QHBoxLayout()
        row.addStretch(1)
        btn_ok = QPushButton("确定")
        btn_ok.setObjectName("okBtn")
        btn_ok.clicked.connect(self._on_ok)
        btn_cancel = QPushButton("取消")
        theme.unify_ok_cancel(btn_ok, btn_cancel)
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        root.addLayout(row)

        # 预填（**分钟** ✓ —— 存的就是这个单位 ✓）
        if timer:
            self.ed_name.setText(timer.get("name", ""))
            lo, hi = timer.get("interval", [TIMER_DEFAULT_LO, TIMER_DEFAULT_HI])
            self._lo_min = max(0.0, float(lo))
            self._hi_min = max(self._lo_min, float(hi))
        # 「休息时暂停计时」也要预填（缺省 = 勾上 = 老行为：休息时冻住 ✓）
        self.ck_pause_rest.setChecked(bool((timer or {}).get("pause_on_rest", True)))
        self.sp_lo.valueChanged.connect(self._sync_from_spins)
        self.sp_hi.valueChanged.connect(self._sync_from_spins)
        self.ck_seconds.toggled.connect(self._on_seconds_toggled)
        # ⚠ 口径**最后**摆：`set_seconds_mode` 会按真源把两个控件的值写一遍（内部 `blockSignals` ✓）
        #   —— 放在接信号之前的话，那几次写入会被当成"用户改了值"、把真源覆盖掉 ✓ 踩过 ✓。
        #   读本机偏好（缺省 = 不勾 = **老口径（分钟）** ✓）；初始化**不落盘** ✓。
        self.set_seconds_mode(self._load_seconds(), persist=False)
        # 预设默认文件名跟着「行为名」走（改名字时同步，省得存出一堆"行为预设.json"）
        self.ed_name.textChanged.connect(self._sync_preset_name)
        self._sync_preset_name()

    # ---------------- 「以秒为单位」（2026-10-15） ----------------

    def seconds_mode(self):
        """现在是不是「以秒为单位」（给用例 / 调用方看 ✓）。"""
        return bool(self._seconds)

    def set_seconds_mode(self, on, persist=True):
        """**摆**「以秒为单位」⇒ 按真源重设那两个控件 + 行标签 ✓。

        · **真源（`_lo_min` / `_hi_min`，单位永远是分钟）一个字都不动** ✓ —— 这里只是把
          "同一个量"换算成当前单位显示出来 ✓ ⇒ 来回切**不会改值** ✓；
        · `persist=False` = **不写** `config/ui.yaml` ✓（**用例专用**：不许动用户真实的本机偏好 ✗；
          顺便让用例不受用户本机勾过的状态影响 ✓ —— 否则老用例会假红 ✓）。
        """
        self._seconds = bool(on)
        self.ck_seconds.blockSignals(True)      # 摆的时候别再触发一次 `toggled`（会重复落盘 ✓）
        self.ck_seconds.setChecked(self._seconds)
        self.ck_seconds.blockSignals(False)
        self._apply_unit()
        if persist:
            self._save_seconds()

    def _apply_unit(self):
        """按当前口径把两个控件重设一遍（范围 / 小数位 / 步进 / 值 / 行标签 ✓）。

        · 单位换算：秒口径 = 分钟口径 **× 60** ✓（同一量 ⇒ 来回切**只换显示** ✓）；
        · ⭐ **下限按口径各给**（用户 2026-10-16 ✓ 原话："**编辑定时行为的触发下限、上限
          不能配置 6.0 以下**"✓）：分钟 `TIMER_LO_MIN`（0.01 分钟 ≈ 0.6 秒 ✓）/
          秒 **`TIMER_LO_SEC`（1.0 秒 ✓）**；
          ⚠ 原来秒口径是"分钟下限 × 60"⇒ **正好 6.0** ✗ —— 那不是设定的下限，
            是"分钟只留 1 位小数"的**精度副作用** ✓（0.1 分钟 = 6 秒 ✓ 详见 `TIMER_LO_SEC` 那段 ✓）；
        · **小数位 `TIMER_DECIMALS`（3 ✓）**：1 位时最小一格 = 6 秒 ⇒ 秒数**根本填不进去** ✗
          ⇒ 要与下限一起放 ✓（只放一处会"填得进、存不下"✗）；
        · 步进：分钟 0.1（**老手感一字不动** ✓）/ 秒 1（1 秒一步 ✓）；
        · ⚠ 换单位时控件值按 3 位小数显示 ⇒ 像 2 秒（= 0.03333 分钟）这种会显示成 `0.033` ✓
          —— 那只是**显示**四舍五入 ✓，**真源 `_lo_min` / `_hi_min` 一个字不动** ✓
          （下面 `blockSignals` 就是为这个 ✓；配合 3 位小数 ⇒ 秒口径的值切回来不会丢 ✓）。
        """
        sc = 60.0 if self._seconds else 1.0
        _lo = TIMER_LO_SEC if self._seconds else TIMER_LO_MIN
        _hi = TIMER_HI_MIN * sc
        for sp, v_min in ((self.sp_lo, self._lo_min), (self.sp_hi, self._hi_min)):
            sp.blockSignals(True)               # 换范围/换值会发 `valueChanged` ⇒ 别回头改真源 ✓
            sp.setRange(_lo, _hi)
            sp.setDecimals(TIMER_DECIMALS)
            sp.setSingleStep(1.0 if self._seconds else 0.1)
            sp.setValue(min(max(v_min * sc, _lo), _hi))
            sp.blockSignals(False)
        unit = "s" if self._seconds else "min"
        self.form.labelForField(self.sp_lo).setText("触发下限(%s)" % unit)
        self.form.labelForField(self.sp_hi).setText("触发上限(%s)" % unit)

    def _on_seconds_toggled(self, on):
        self.set_seconds_mode(on)

    def _sync_from_spins(self, *_a):
        """控件一变 ⇒ 换回**分钟**写进真源 ✓（真源的**唯一写口** ✓），并顺手钳上限 ✓。"""
        sc = 60.0 if self._seconds else 1.0
        self._lo_min = float(self.sp_lo.value()) / sc
        self._hi_min = float(self.sp_hi.value()) / sc
        self._clamp_hi()

    def _clamp_hi(self):
        """上限不许小于下限（**在分钟上比** ✓ 与当前显示单位无关 ✓）。

        ⚠ 下限一律用**当前口径**那一份（`TIMER_LO_SEC` / `TIMER_LO_MIN` ✓ 见 `_apply_unit` ✓）
          —— 两处各写一份必然分叉 ✗（这正是先前那个"秒口径正好 6.0"的来源 ✓）。
        """
        if self._hi_min >= self._lo_min:
            return
        self._hi_min = self._lo_min
        sc = 60.0 if self._seconds else 1.0
        _lo = TIMER_LO_SEC if self._seconds else TIMER_LO_MIN
        self.sp_hi.blockSignals(True)
        self.sp_hi.setValue(min(max(self._hi_min * sc, _lo), TIMER_HI_MIN * sc))
        self.sp_hi.blockSignals(False)

    def _load_seconds(self):
        """读本机偏好 ⇒ 默认 **False**（不勾 = 分钟 = **老口径** ✓）。

        ⚠ 任何异常一律当"没存过"✓ —— 界面偏好读不出来**不是**错误，
        不许把弹窗拦在门外 ✗（同 `gui/theme.load_section` 自己的口径 ✓）。
        """
        try:
            return bool(theme.load_section(_UI_SECTION).get(_UI_SECONDS_KEY, False))
        except Exception:                       # noqa: BLE001
            return False

    def _save_seconds(self):
        """把这个勾选记进本机（`config/ui.yaml` ✓）；存不下就只在本次会话生效 ✓（不报错）。"""
        try:
            d = theme.load_section(_UI_SECTION)
            d[_UI_SECONDS_KEY] = bool(self._seconds)
            theme.save_section(_UI_SECTION, d)
        except Exception:                       # noqa: BLE001
            pass

    def _sync_preset_name(self):
        """把「行为名」同步成预设的默认文件名（空名时退回"定时行为"）。"""
        self.editor.preset_name = sp.safe_name(self.ed_name.text().strip()
                                               or "定时行为")

    def _on_ok(self):
        if not self.ed_name.text().strip():
            QMessageBox.warning(self, "提示", "行为名不能为空")
            return
        self.accept()

    def result(self):
        return {
            "name": self.ed_name.text().strip(),
            # ⚠ **取真源（分钟 ✓）**，不是控件的值 ✗ —— 控件里放的是"当前单位下的数"
            #   （勾了「以秒为单位」时它是秒 ✓ 直接交出去就慢了 60 倍 ✗）。
            "interval": [self._lo_min, self._hi_min],
            "seq": self.editor.seq(),
            # 「休息时暂停计时」（2026-09-26 新增的项目级开关 ✓）
            "pause_on_rest": bool(self.ck_pause_rest.isChecked()),
        }
