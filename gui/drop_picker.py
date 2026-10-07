"""选择要标注的**掉落物（道具）**弹窗（用户 2026-10-04 ✓）。

版式与 `gui/mob_picker.py` **同一套**（上方已选 + 下方"输入名称或 id 筛选"候选 ✓，
可照抄的既有规范：搜索框 + 确定/取消 + `theme.bind_window_state` ✓）。

⭐ **两区都是背包窗格**（`gui/icon_grid.IconGrid` ✓ 用户 2026-10-04 追加要求："以背包窗格的
  形式展示项目，鼠标指着的时候显示信息弹窗（MapleNecrocer 应该有相关实现）"）：格子 + 悬停
  信息窗（名称 / id / 类别 / 几帧 ✓ 口径在 `icon_grid.tip_lines`，**只此一处** ✓）——
  所以这里**不再自己拼行文本** ✗（以前那套 `_drop_row_text` / `_drop_tip` / `_drop_icon`
  已删：同一件事两处实现，迟早一边缺东西 ✓）。

**数据从哪来**（与「怪」那条线**对称**，见 `core/wzexport.py` 里"掉落物图库"那一段 ✓）：
  · 名称 / 类别 / 几帧 ⇒ 清单 `datasets/drops.json` ✓；
  · 图标 ⇒ 精灵库 `datasets/sprites/drop/<id>/*.png` ✓（金币那几个 id **WZ 里就没名字** ✗
    ⇒ 界面如实写「（WZ 里没有名字）」+ 告诉人怎么手工补 ✓ 见 SKILL 约定 190 ✓）。
⚠ 图库整个不在时（清单和图标目录都没有）⇒ **如实说清是哪一步缺**（不是"没搜到" ✗ ——
  后者会让人以为名字打错了 ✓），并把「添加选中」灰掉 ✓。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSplitter, QVBoxLayout,
                             QWidget)

from core import wzexport
from gui import theme                  # 弹窗几何按客户端存（config/ui.yaml）✓
from gui.icon_grid import IconGrid     # ⭐ 背包窗格 + 悬停信息窗（口径只此一处 ✓）

_MAX_CANDIDATES = 200   # 单次最多展示的候选数，避免候选爆炸拖慢界面（同怪那份 ✓）


class DropPickDialog(QDialog):
    """返回 `[{id, name}]`（保持用户顺序 ✓ 见 `result_drops` ✓）。"""

    def __init__(self, current_drops, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择要标注的掉落物")
        self.setMinimumSize(520, 660)

        # 已选：统一成 `[{id, name}]`（老配置可能只存了 id 字符串 ⇒ `norm_drops` 兜住 ✓、
        # 名字缺了就查清单补上 ✓）
        self._names = wzexport.drop_name_map()
        self._drops = wzexport.norm_drops(current_drops)
        for d in self._drops:
            if not d["name"]:
                d["name"] = self._names.get(d["id"], "")
        self._ok, self._why = wzexport.drop_library_state()
        self._all = wzexport.list_sprite_drops() if self._ok else []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        note = QLabel("标定/标注只会处理这里列出的掉落物。\n"
                      "两区都是**背包窗格**：**鼠标指到格子上**就弹出名称 / id / 类别 / 几帧 ✓"
                      "（有些没有名字 —— 那是 WZ 里就没有，金币整类都没有，\n"
                      "可以手工往 `datasets/drops.json` 里补 ✓）。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f6368;")
        root.addWidget(note)
        if not self._ok:
            # ⚠ 图库没导出 ⇒ **说清是哪一步缺**（不是"搜不到" ✗），并把添加区灰掉 ✓
            warn = QLabel(self._why)
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #b06000;")
            root.addWidget(warn)

        self.split = QSplitter(Qt.Vertical)
        self.split.setChildrenCollapsible(False)

        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(4)
        lbl = QLabel("当前要标注的掉落物")
        lbl.setStyleSheet("font-weight: 600;")
        tv.addWidget(lbl)

        self.lst = IconGrid()
        tv.addWidget(self.lst, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_del = QPushButton("移出选中")
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
        lbl2 = QLabel("添加掉落物（输入名称或 id 筛选）")
        lbl2.setStyleSheet("font-weight: 600;")
        bv.addWidget(lbl2)

        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("如「金币」或「04030001」…")
        self.ed_filter.textChanged.connect(self._refilter)
        self.ed_filter.setEnabled(self._ok)
        bv.addWidget(self.ed_filter)

        self.cand = IconGrid()
        # ⭐ **双击直接加进来**（背包那种手感 ✓）—— 悬停看信息、双击进包、也可选中后点按钮 ✓
        self.cand.activated_id.connect(lambda _i: self._add())
        bv.addWidget(self.cand, 1)

        addrow = QHBoxLayout()
        addrow.setSpacing(6)
        self.btn_add = QPushButton("添加选中")
        self.btn_add.clicked.connect(self._add)
        self.btn_add.setEnabled(self._ok)      # 图库没导出 ⇒ 没东西可加 ✓（不假装 ✗）
        addrow.addWidget(self.btn_add)
        self.lbl_match = QLabel()
        self.lbl_match.setStyleSheet("color: #80868b;")
        addrow.addWidget(self.lbl_match, 1)
        bv.addLayout(addrow)
        self.split.addWidget(bot)

        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        self.split.setSizes([300, 220])
        root.addWidget(self.split, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("确定")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        theme.unify_ok_cancel(btns.button(QDialogButtonBox.Ok),
                              btns.button(QDialogButtonBox.Cancel))
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._refresh_list()
        self._refilter()
        theme.bind_window_state(self, "drop_picker")   # 拉过的大小/位置按客户端记住 ✓

    # ---------------- 刷新 ----------------

    def _refresh_list(self):
        """已选区：背包窗格（名字/类别/几帧由 `IconGrid` 自己去查清单 ✓ 悬停就能看到 ✓）。"""
        self.lst.set_entries(self._drops)
        n = len(self._drops)
        self.lbl_sel.setText("共 %d 项" % n if n else "还没有选掉落物")

    def _refilter(self):
        """重列候选 —— ⚠⭐ **不打字也列**（用户 2026-10-04 ✓）。

        **为什么要破 `mob_picker` 那条"必须先输入"的规矩**（那个弹窗是"输入才显示候选" ✓）：
          怪有名字（`maps.json` 里有 ✓）⇒ 输入「青蛇」就能找到 ✓；而**金币在 `String.wz` 里
          根本没有名字** ✗（整个 Special 分支不存在 ✓）⇒ 逼人先打字＝**逼人去猜 8 位 id** ✗。
          ⇒ 空关键词时就把图库列出来（截至 `_MAX_CANDIDATES` ✓ 335 项一次列完不卡 ✓），
          让人**用眼睛挑** ✓；要缩小范围再打字 ✓。
        """
        if not self._ok:
            self.cand.set_entries([])
            self.lbl_match.setText("图库还没导出 ⇒ 没法列候选（见上面的说明）")
            return
        kw = self.ed_filter.text().strip()
        # 关键词空 ⇒ 全列（受 `_MAX_CANDIDATES` 限制）；有词 ⇒ 走 `list_sprite_drops` 的筛选 ✓
        pool = self._all if not kw else wzexport.list_sprite_drops(kw)
        picked = {d["id"] for d in self._drops}
        rows = [d for d in pool if d["id"] not in picked][:_MAX_CANDIDATES]
        self.cand.set_entries(rows)          # ⭐ 格子 ✓（悬停看名称/id ✓）
        shown = len(rows)
        if not kw:
            self.lbl_match.setText(
                ("图库共 %d 项 · 已列出 %d 项%s（输入 id / 名称 / 类别 可缩小范围）"
                 % (len(self._all), shown,
                    "（只列前 %d）" % _MAX_CANDIDATES
                    if len(self._all) > shown else ""))
                if shown else "图库里什么都没有")
        else:
            self.lbl_match.setText("%d 个匹配" % shown if shown else "无匹配，换个关键词试试")

    # ---------------- 操作 ----------------

    def _remove(self):
        did = self.lst.current_id()          # ⚠ 唯一出口（格子版式换过 ⇒ 别再读行号 ✗）
        if not did:
            return
        self._drops = [d for d in self._drops if d["id"] != did]
        self._refresh_list()
        self._refilter()

    def _add(self):
        did = self.cand.current_id()         # ⚠ 同上（双击也走这里 ✓ 见 `activated_id` ✓）
        if not did or any(d["id"] == did for d in self._drops):
            return
        # 名字直接取**候选那一份**（它带着清单里的名字 ✓）；缺了再查一次名字表 ✓
        nm = ""
        for d in self._all:
            if d["id"] == did:
                nm = str(d.get("name") or "")
                break
        self._drops.append({"id": did, "name": nm or self._names.get(did, "")})
        self._refresh_list()
        self._refilter()

    def result_drops(self):
        return [dict(d) for d in self._drops]
