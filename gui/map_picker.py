"""「手动更换」地图弹窗（用户 2026-10-02 ✓ 原话："增加按钮『手动更换』，用以修改这个 id"）。

**为什么要有它**：路线识别 →「寻路配置」顶部那行「当前地图」默认跟着
「模型训练 → 识别目标」那个下拉走（同一份 `project.map_id` ✓），但那个下拉在**另一个
页签**里 —— 想在这一页换图就得来回切页签，而且切回来还要等本页重读一次（`showEvent`
才刷新 ✓）。所以就地给一个入口 ✓。

**口径只有一份**（别在这儿另写一套 ✗）：
  · 候选清单 = `core.wzexport.list_maps`（和模型训练那个下拉**同一份实现**，
    连筛选都是调它 ✓ —— 含"逐字重合度"那套兜底，记不全地图名也能找到 ✓）；
  · 写项目 = `core.wzexport.apply_map_choice`（地图 + 怪列表四件一起写 ✓ 同那个下拉 ✓）。

⚠ 版式与「确定/取消」配色照 `gui/mob_picker.MobPickDialog`（同类东西长成一个样 ✓
见 docs/UI规范.md §6 / §11）；窗口几何也按客户端记住（`theme.bind_window_state` ✓）。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QVBoxLayout)

from core import wzexport
from gui import theme

#: 单次筛选最多展示多少项（清单里几百张图，全列出来谁也翻不动 ✓ 同 mob_picker 的思路）
_MAX_SHOWN = 400


class MapPickDialog(QDialog):
    """挑一张地图 → `accept()`；`self.chosen` 是选中的 id（没选就是空串 ✓）。"""

    def __init__(self, current_id="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("更换当前地图")
        self.setMinimumSize(560, 520)

        self._current = str(current_id or "").strip()
        self.chosen = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        cur = wzexport.map_entry(self._current)
        note = QLabel(
            "选一张地图作为**当前地图**（改的是这个项目的地图参数，"
            "和「模型训练 → 识别目标」那个下拉是同一个 ✓）。\n"
            "当前：%s\n"
            "换了之后：地形/集合/标定/战斗区域都跟着这张图走；"
            "「实时」若在跑，会立刻改用新图（不用重启）。"
            % (wzexport.short_label(self._current,
                                    (cur or {}).get("name", "")) or "（还没选）"))
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f6368;")
        root.addWidget(note)

        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("输入地图名 / id / 怪名筛选，如「森林迷宫」或「105040303」…")
        self.ed_filter.textChanged.connect(self._refilter)
        root.addWidget(self.ed_filter)

        self.lst = QListWidget()
        self.lst.setStyleSheet(_LIST_QSS)
        self.lst.itemDoubleClicked.connect(lambda _i: self._on_ok())
        self.lst.currentItemChanged.connect(lambda *_a: self._sync_enabled())
        root.addWidget(self.lst, 1)

        self.lbl_count = QLabel()
        self.lbl_count.setStyleSheet("color: #5f6368;")
        self.lbl_count.setWordWrap(True)
        root.addWidget(self.lbl_count)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_ok = btns.button(QDialogButtonBox.Ok)
        self.btn_ok.setText("换成这张")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        theme.unify_ok_cancel(self.btn_ok, btns.button(QDialogButtonBox.Cancel))
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._refilter()
        theme.bind_window_state(self, "map_picker")     # 拉过的大小/位置按客户端记住 ✓

    # ---------------- 列表 ----------------

    def _refilter(self):
        """按关键词重列候选（**筛选口径复用 `list_maps`** ✓ 别在这儿自己写匹配 ✗）。

        ⚠ 关键词为空时**不列**：几百张图全塞进列表，人只会觉得卡（同 mob_picker：
        "输入筛选后显示候选" ✓）。但**当前那张图总要显示** —— 否则看着像它不存在 ✗。
        """
        kw = self.ed_filter.text().strip()
        self.lst.clear()
        rows = wzexport.list_maps(only_with_mob=False, keyword=kw) if kw else []
        if not kw and self._current:
            cur = wzexport.map_entry(self._current)
            if cur is not None:
                self._add(cur)
        for m in rows[:_MAX_SHOWN]:
            self._add(m)
        n = self.lst.count()
        if not kw:
            self.lbl_count.setText(
                "输入关键词开始筛选（清单共 %d 张图）" % len(
                    wzexport.list_maps(only_with_mob=False, keyword="")))
        elif n >= _MAX_SHOWN:
            self.lbl_count.setText("匹配很多，只列出前 %d 张 —— 多打几个字缩小范围" % _MAX_SHOWN)
        else:
            self.lbl_count.setText("匹配 %d 张" % n if n else "没有匹配的地图 —— 换个词试试")
        self._sync_enabled()

    def _add(self, m):
        item = QListWidgetItem(m["label"])
        item.setData(Qt.UserRole, m["id"])
        item.setToolTip("id：%s%s" % (m["id"],
                                     ("\n地图名：%s" % m["name"]) if m["name"] else ""))
        if m["id"] == self._current:
            item.setText(item.text() + "    ← 当前")
        self.lst.addItem(item)

    def _sync_enabled(self):
        """没选中就不许「换成这张」（省得点了确定却什么都没换 ✗）。"""
        self.btn_ok.setEnabled(self._selected() != "")

    def _selected(self):
        it = self.lst.currentItem()
        return str(it.data(Qt.UserRole)) if it is not None else ""

    def _on_ok(self):
        self.chosen = self._selected()
        if not self.chosen:
            return                                      # 没选就什么都不做（按钮本来就灰着 ✓）
        self.accept()


_LIST_QSS = ("QListWidget { border: 1px solid #dadce0; border-radius: 6px;"
             " background: #fafbfc; }"
             "QListWidget::item { padding: 3px; }"
             "QListWidget::item:selected { background: #cfe0fb; color: #1967d2; }")
