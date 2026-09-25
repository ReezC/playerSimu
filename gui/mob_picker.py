"""确认要识别的怪物弹窗。

上方列表是当前会处理的怪（名称 + stand 第 0 帧缩略图），可删除。
下方是添加区：输入名称或 id 筛选候选（候选 = 精灵库里所有有 stand 帧的怪），
从过滤结果里选中一个「添加」到上方。

怪物名优先用地图清单里的名字，没有就只显示 id。
"""

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSplitter, QVBoxLayout, QWidget)

from core import wzexport

_THUMB = 48
_MAX_CANDIDATES = 200   # 单次筛选最多展示的候选数，避免候选爆炸拖慢界面


def _mob_display(mid, name_map):
    name = name_map.get(mid)
    return ("%s  (%s)" % (name, mid)) if name else mid


def _mob_icon(mid, size=_THUMB):
    """缩略图：优先 stand 第 0 帧，没有就用 fly —— 很多飞的怪只有 fly。"""
    _act, frames = wzexport.mob_action_frames(wzexport.sprite_dir_path() / mid)
    if frames:
        pm = QPixmap(str(frames[0]))
        if not pm.isNull():
            return QIcon(pm.scaled(size, size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation))
    return QIcon()


class MobPickDialog(QDialog):
    def __init__(self, current_mobs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("确认要识别的怪物")
        self.setMinimumSize(500, 680)

        self.name_map = wzexport.build_mob_name_map()
        self._mobs = list(current_mobs)     # 有序，保持用户原有顺序
        self._all = wzexport.list_sprite_mobs()   # 全部候选 id（排序）

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        note = QLabel("删掉画面里没有的怪，添加漏掉的怪；确定后保存。"
                      "标定/标注只会处理这里列出的怪。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f6368;")
        root.addWidget(note)

        # ---- 上下可拖动的分割区：上 = 已选列表，下 = 添加区 ----
        self.split = QSplitter(Qt.Vertical)
        self.split.setChildrenCollapsible(False)   # 防止拖到底把某块拖没

        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(4)
        lbl = QLabel("当前会处理的怪物")
        lbl.setStyleSheet("font-weight: 600;")
        tv.addWidget(lbl)

        self.lst = QListWidget()
        self.lst.setIconSize(QSize(_THUMB, _THUMB))
        self.lst.setStyleSheet(self._LIST_QSS)
        tv.addWidget(self.lst, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_del = QPushButton("删除选中")
        self.btn_del.clicked.connect(self._remove)
        row.addWidget(self.btn_del)
        self.lbl_sel = QLabel()
        self.lbl_sel.setStyleSheet("color: #80868b;")
        row.addWidget(self.lbl_sel, 1)
        tv.addLayout(row)
        self.split.addWidget(top)

        bot = QWidget()
        bv = QVBoxLayout(bot)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(4)
        lbl2 = QLabel("添加怪物（输入名称或 id 筛选）")
        lbl2.setStyleSheet("font-weight: 600;")
        bv.addWidget(lbl2)

        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("如「青蛇」或「2130」…")
        self.ed_filter.textChanged.connect(self._refilter)
        bv.addWidget(self.ed_filter)

        self.cand = QListWidget()
        self.cand.setIconSize(QSize(_THUMB, _THUMB))
        self.cand.setStyleSheet(self._LIST_QSS)
        self.cand.itemDoubleClicked.connect(lambda _i: self._add())
        bv.addWidget(self.cand, 1)

        addrow = QHBoxLayout()
        addrow.setSpacing(6)
        self.btn_add = QPushButton("添加选中")
        self.btn_add.clicked.connect(self._add)
        addrow.addWidget(self.btn_add)
        self.lbl_match = QLabel()
        self.lbl_match.setStyleSheet("color: #80868b;")
        addrow.addWidget(self.lbl_match, 1)
        bv.addLayout(addrow)
        self.split.addWidget(bot)

        # 已选列表默认占更多（原来太短），拉伸比例也偏上；拖分隔条可随时调整
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        self.split.setSizes([320, 200])
        root.addWidget(self.split, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("确定")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._refresh_list()
        self._refilter()

    _LIST_QSS = ("QListWidget { border: 1px solid #dadce0; border-radius: 6px;"
                 " background: #fafbfc; }"
                 "QListWidget::item { padding: 3px; }"
                 "QListWidget::item:selected { background: #cfe0fb; color: #1967d2; }")

    # ---------------- 刷新 ----------------

    def _refresh_list(self):
        self.lst.clear()
        for mid in self._mobs:
            item = QListWidgetItem(_mob_display(mid, self.name_map))
            item.setData(Qt.UserRole, mid)
            item.setIcon(_mob_icon(mid))
            self.lst.addItem(item)
        n = len(self._mobs)
        self.lbl_sel.setText("共 %d 种" % n if n else "还没有选怪")

    def _refilter(self):
        kw = self.ed_filter.text().strip()
        self.cand.clear()
        if not kw:
            self.lbl_match.setText("输入筛选后显示候选（共 %d 种怪）" % len(self._all))
            return

        kw = kw.lower()
        shown = 0
        for mid in self._all:
            if mid in self._mobs:
                continue
            disp = _mob_display(mid, self.name_map)
            if kw not in disp.lower():
                continue
            item = QListWidgetItem(disp)
            item.setData(Qt.UserRole, mid)
            item.setIcon(_mob_icon(mid))
            self.cand.addItem(item)
            shown += 1
            if shown >= _MAX_CANDIDATES:
                break

        self.lbl_match.setText("%d 个匹配" % shown
                               if shown else "无匹配，换个关键词试试")

    # ---------------- 操作 ----------------

    def _remove(self):
        row = self.lst.currentRow()
        if row < 0:
            return
        mid = self.lst.item(row).data(Qt.UserRole)
        if mid in self._mobs:
            self._mobs.remove(mid)
        self._refresh_list()
        self._refilter()

    def _add(self):
        row = self.cand.currentRow()
        if row < 0:
            return
        mid = self.cand.item(row).data(Qt.UserRole)
        if not mid or mid in self._mobs:
            return
        self._mobs.append(mid)
        self._refresh_list()
        self._refilter()

    def result_mobs(self):
        return list(self._mobs)
