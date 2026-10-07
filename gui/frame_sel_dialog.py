"""「这条目用哪些帧当模板」的勾选弹窗（用户 2026-10-05 ✓ 原话：

    "在背包窗格中双击选中的怪物、玩家、掉落物、宠物，显示弹窗展示所有的帧，
     用户可以自由决定选中哪些帧用于匹配（默认是旧逻辑中的那些）"

⭐⭐ 2026-10-05 二轮（用户原话："预览模板帧需要能看到图，光看名字判断不出来"）：
  · 列表改成**缩略图格子**（每格 = 图 + 帧名 + 勾选框 ✓）；
  · 右边一块**大图预览**（点哪一帧看哪一帧 ✓）—— 光看 `stand_3` 这种名字确实判断不了
    这一帧是不是"怪被技能糊住的那张" ✗，得看图 ✓；
  · ⚠ 图从哪来：调用方把 `{stem: 图片路径}` 一并传进来 ✓（`files=` ✓ 见
    `LabelCard._entry_frame_files` ✓ —— 四段的图片位置各不相同，界面层不自己猜 ✗）。
    没传 / 读不出图 ⇒ **退化成只看名字** ✓（老调用方一字不改 ✓ 不会崩 ✓）。

⚠ 与 `gui/frame_picker.FramePickDialog` **不是一回事**（别混 ✗）：
  · 那个选的是**画面帧**（这一趟要处理哪些素材 ✓ 键在 `_only[target]` ✓）；
  · 这个选的是**模板帧**（这个条目拿哪几张图去匹配 ✓ 键在 `label.frames_sel` ✓）。
  两条轴正交 ✓ 可以同时用 ✓。

⚠ 默认 = **旧逻辑会用的那些**（调用方算好传进来 ✓ 见 `LabelCard._default_frames` ✓）：
  用户"只是看看、不改"就点确定 ⇒ 落盘的正是老口径 ⇒ 行为一字不变 ✓。
"""
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QLabel,
                             QListView, QListWidget, QListWidgetItem, QPushButton,
                             QVBoxLayout)

from gui import theme


class FrameSelDialog(QDialog):
    """列出某个条目的**全部**帧（带图 ✓）⇒ 勾选参与匹配的那些 ✓。"""

    #: 格子里缩略图的边长（px ✓）—— 88 够看出"这是哪一帧" ✓ 又不会让格子太宽 ✓。
    THUMB = 88
    #: 右侧大图的边长上限（px ✓）—— 想看清细节时看它 ✓。
    PREVIEW = 240

    def __init__(self, parent, title, stems, selected=None, files=None):
        """`stems` = 帧名（顺序就是显示顺序 ✓）；`files` = `{stem: 图片路径}`（可选 ✓）。"""
        super().__init__(parent)
        self.setWindowTitle("选择模板帧 —— %s" % (title or ""))
        self.setMinimumSize(760, 560)
        self.resize(880, 620)
        self._stems = [str(s) for s in (stems or [])]
        self._files = {str(k): str(v) for k, v in (files or {}).items()}
        _sel = set(str(s) for s in (selected or []))
        _noimg = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        hint = QLabel("勾上「参与模板匹配」的帧 ✓　·　不勾的帧这一趟**不会被当模板** ✓\n"
                      "默认 = 现在这条逻辑本来的选择（不改也能直接确定 ✓）"
                      "　·　**点一格看右边大图** ✓")
        hint.setStyleSheet("color:#80868b;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        mid = QHBoxLayout()
        mid.setSpacing(10)

        # ---- 左：缩略图格子（复选框照旧在格子上 ✓ 点勾选圈才改勾选 ✓）----
        self.lst = QListWidget()
        self.lst.setViewMode(QListView.IconMode)          # ⭐ 图 + 文件名 的格子 ✓
        self.lst.setIconSize(QSize(self.THUMB, self.THUMB))
        self.lst.setGridSize(QSize(self.THUMB + 48, self.THUMB + 46))
        self.lst.setResizeMode(QListView.Adjust)          # 窗口一变 ⇒ 自动重排 ✓
        self.lst.setMovement(QListView.Static)            # 不许拖动格子 ✗
        self.lst.setWordWrap(True)
        self.lst.setSpacing(4)
        self.lst.setSelectionMode(QListWidget.SingleSelection)
        for s in self._stems:
            it = QListWidgetItem(s)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if s in _sel else Qt.Unchecked)
            it.setTextAlignment(Qt.AlignHCenter)
            _p = self._files.get(s)
            pm = QPixmap(_p) if _p else QPixmap()
            if pm.isNull():
                # ⚠ 读不出图也**照常列出来** ✓（只看名字 ✓ 别把这一帧藏了 ✗ ——
                #   藏了的话用户会以为"库里没有这一帧" ✓ 那是另一回事 ✗）。
                _noimg += 1
                it.setToolTip("%s\n（没找到这一帧的图，只能看名字 ✓）" % s)
            else:
                it.setIcon(QIcon(pm.scaled(self.THUMB, self.THUMB,
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                it.setToolTip("%s\n%s" % (s, _p))
            self.lst.addItem(it)
        mid.addWidget(self.lst, 3)

        # ---- 右：大图预览（判断"这一帧到底长什么样"就靠它 ✓）----
        side = QVBoxLayout()
        side.setSpacing(6)
        box = QFrame()
        box.setFrameShape(QFrame.StyledPanel)
        bl = QVBoxLayout(box)
        bl.setContentsMargins(6, 6, 6, 6)
        self.preview = QLabel("（点左边一格看大图）")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(self.PREVIEW, self.PREVIEW)
        self.preview.setWordWrap(True)
        bl.addWidget(self.preview)
        side.addWidget(box)
        self.lbl_name = QLabel("—")
        self.lbl_name.setStyleSheet("color:#5f6368;")
        self.lbl_name.setWordWrap(True)
        self.lbl_name.setMinimumWidth(self.PREVIEW)
        side.addWidget(self.lbl_name)
        if _noimg:
            _miss = QLabel("⚠ 有 %d 帧没读到图（那几格只能看名字 ✓）" % _noimg)
            _miss.setStyleSheet("color:#b06000;")
            _miss.setWordWrap(True)
            side.addWidget(_miss)
        side.addStretch(1)
        mid.addLayout(side, 2)
        root.addLayout(mid, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        for text, fn in (("全选", lambda: self._set_all(True)),
                         ("全不选", lambda: self._set_all(False)),
                         ("反选", self._invert)):
            b = QPushButton(text)
            b.setToolTip("%s（只动勾选，不改帧本身 ✓）" % text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        self.lbl_n = QLabel("")
        self.lbl_n.setStyleSheet("color:#5f6368;")
        row.addWidget(self.lbl_n)
        root.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("确定")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        theme.unify_ok_cancel(btns.button(QDialogButtonBox.Ok),
                              btns.button(QDialogButtonBox.Cancel))
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self.lst.itemChanged.connect(lambda _i: self._refresh_n())
        self.lst.currentItemChanged.connect(self._on_current)
        if self.lst.count():
            self.lst.setCurrentRow(0)                     # 一进来就有大图看 ✓
        self._refresh_n()
        theme.bind_window_state(self, "frame_sel")   # 几何按客户端记住 ✓（UI规范 §11 ✓）

    # ---------------- 交互 ----------------

    def _on_current(self, cur, _prev=None):
        """点哪一帧 ⇒ 右边看哪一帧 ✓（读不出图就如实说 ✓ 别显示上一帧的图 ✗）。"""
        s = str(cur.text()) if cur is not None else ""
        p = self._files.get(s)
        pm = QPixmap(p) if p else QPixmap()
        if pm.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText("（这一帧读不出图）\n%s" % (s or ""))
            self.lbl_name.setText(s or "—")
            return
        self.preview.setPixmap(pm.scaled(self.PREVIEW, self.PREVIEW,
                                         Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.lbl_name.setText("%s　（原图 %d×%d）" % (s, pm.width(), pm.height()))

    def _set_all(self, on):
        for i in range(self.lst.count()):
            self.lst.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _invert(self):
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            it.setCheckState(Qt.Unchecked if it.checkState() == Qt.Checked
                             else Qt.Checked)

    def _refresh_n(self):
        self.lbl_n.setText("已勾 %d / %d 帧" % (len(self.selected_stems()),
                                                len(self._stems)))

    # ---------------- 结果 ----------------

    def selected_stems(self):
        """勾上的帧 stem（**保持原顺序** ✓）—— 空 = 一个都没勾 ✓（调用方如实落盘 ✓）。"""
        out = []
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if it.checkState() == Qt.Checked:
                out.append(str(it.text()))
        return out
