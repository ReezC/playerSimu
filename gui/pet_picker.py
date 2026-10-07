"""选择要标注的**宠物**弹窗（用户 2026-10-04 ✓ 方案 C ✓ 照 MapleNecrocer 的 `PetForm` ✓）。

流程与用户说的一模一样 ✓：**搜索筛选宠物 → 选这只宠物戴的装备 → 加入"要标注的宠物列表"** ✓。

⭐ **这里存的是"宠物 + 装备"这一套**（不是单只宠物 ✗）—— 因为要标的是**组合外观** ✓：
   宠物戴了帽子/丝带之后样子就变了 ✓，模板必须按"戴着的样子"做 ✓（见 `core/petlib.compose_pet` ✓）。

数据从 `core/petlib.py` 来（真源 = `datasets/pets.json` / `datasets/pet_equips.json` ✓，
由外部 `WzProbe dump-pets` / `dump-petequips` 导出 ✓）：
  · 左栏 = 宠物清单 ✓（带 `icon.png` ✓）；
  · 右栏 = **这只宠物能戴的装备** ✓ —— 判据是导出时记下的 `pets:[能戴的宠物]` ✓
    （源头 = `Character.wz/PetEquip/<装备>/<宠物id>` 那个子目录 ✓ 见 SKILL 约定 206 ✓）；
  · 中下那个小窗 = **组合预览**（现算 ✓ `petlib.compose_pet` 有缓存 ✓）。

⚠ 图库没导出时**如实说缺哪一步**（`petlib.pet_library_state()` ✓ 点名那两个通道 ✓），
  并把「加入列表」灰掉 ✓ —— 同掉落物那个弹窗的纪律 ✓。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QSplitter, QVBoxLayout, QWidget)

from core import petlib
from gui import theme
from gui.icon_grid import IconGrid

_MAX_CANDIDATES = 300      # 宠物/装备都不多（实测 50 只 / 52 件 ✓）⇒ 上限只当保险 ✓


def combo_entry(pet_row, equips):
    """把「宠物 + 装备」拼成 `IconGrid` 吃的那种条目 ✓（**口径只此一处** ✓）。

    ⚠ 复用掉落物那套字段名（`id/name/desc/icon` ✓）⇒ `icon_grid.tip_lines` 一个字都不用改 ✓：
      · `id`   = 宠物 id（模板名里它在前 ✓）；
      · `name` = 「宠物名 + 装备名」✓（拿不到名字就退回 id ✓ 别显示空 ✓）；
      · `desc` = 「宠物 id + 装备 id」那一行 ✓（查得清是哪一套 ✓）；
      · `icon` = **组合图** `stand0_0.png` ✓（就是"戴着的样子" ✓）；
      · `frames` = **组合出来有几帧** ✓（⭐ 用户 2026-10-04 ✓ 原话："小白雪人有很多帧，为什么
        添加后写的只有1帧？"✗）—— 不填的话 `icon_grid.normalize_entry` 按默认 **1** 算 ✗ ⇒
        信息窗写「1 张静态图」✗（截图里那句 ✓），而实际 `stand0` 有 3 帧 ✓、
        模板匹配**这 3 帧都会当模板** ✓ ⇒ 界面说的是假话 ✗。
        ⚠ 数的是**组合目录里**、`petlib.COMBO_STATES` 那几个姿态的帧 ✓（= 真拿去匹配的那几张 ✓）——
          宠物本体有几帧不等于最后用几帧 ✗（组合按那几个姿态走 ✓ 见 `petlib` 模块说明 ✓）。
          ⭐ 2026-10-04 扩姿态之后（`stand0/stand1/move/jump/hungry` ✓ 用户要求 ✓）这里
          自然跟着变成"五姿态总帧数" ✓（原来只数 `stand0_*` ✗ ⇒ 界面少报 ✓）。
      · `states` = 那几个姿态的**中文短语**（"站立/走路/跳/饿" ✓）—— 给信息窗直说模板覆盖了
        哪些姿态 ✓（用户那句疑问的正解 ✓）。
    """
    pet_row = pet_row or {}
    pid = str(pet_row.get("id") or "")
    eq_ids = [str(x).strip() for x in (equips or []) if str(x).strip()]
    eq_names = {str(e.get("id")): str(e.get("name") or "") for e in petlib.load_pet_equip_index()}
    pname = str(pet_row.get("name") or "").strip() or pid
    if eq_ids:
        nm = "%s + %s" % (pname, "、".join(eq_names.get(e, e) for e in eq_ids))
    else:
        nm = pname
    d = petlib.compose_pet(pid, eq_ids)          # 现算（有缓存 ✓）
    icon = str(d / "stand0_0.png") if d is not None else ""
    desc = "宠物 %s" % pid + ("　装备 %s" % "、".join(eq_ids) if eq_ids else "（没戴装备）")
    # ⭐ **几帧**：数组合目录里真拿去匹配的那几张 ✓（见上面 `frames` 那段说明 ✓）
    #   ⚠ 必须按 `petlib.COMBO_STATES`（**姿态列表**）数 ✗ 别写死 `stand0_*` ——
    #     2026-10-04 扩姿态后（站立/走路/跳/饿 ✓ 用户要求 ✓）只数 stand0 会**少报** ✓。
    n_fr = 0
    if d is not None:
        for _st in petlib.COMBO_STATES:
            n_fr += len(list(d.glob("%s_*.png" % _st)))
    if not n_fr:                                  # 组合做不出来 ⇒ 退回"本体有几帧" ✓
        n_fr = len(petlib.sprite_frames(petlib.pet_dir(), pid)) or 1
    # ⭐ 姿态的中文短语（给信息窗直说"模板覆盖哪几个姿态" ✓ 用户那句疑问的正解 ✓）：
    #   ⚠ 顺序 = `COMBO_STATES` 的顺序 ✓ 只列**这只宠物真有帧**的那些 ✓（别写没数据的 ✓）。
    _ZH = {"stand0": "站立", "stand1": "站立(呼吸)", "move": "走路",
           "jump": "跳", "hungry": "饿", "rest0": "休息", "sit": "坐", "angry": "生气"}
    _sts = []
    if d is not None:
        for _st in petlib.COMBO_STATES:
            if list(d.glob("%s_*.png" % _st)):
                _sts.append(_ZH.get(_st, _st))
    return {"id": pid, "name": nm, "desc": desc, "icon": icon, "has_img": bool(icon),
            "frames": max(1, int(n_fr)),
            "states": "、".join(_sts),
            "pet": pid, "equips": eq_ids}


class PetPickDialog(QDialog):
    """「选择要标注的宠物」弹窗 ✓（`current_pets` = 卡片上那张列表 ✓）。"""

    def __init__(self, current_pets, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择要标注的宠物")
        self.setMinimumSize(760, 620)

        self._ok, self._why = petlib.pet_library_state()
        self._pets = petlib.load_pet_index()
        self._equips = petlib.load_pet_equip_index()
        self._name_of = {str(p["id"]): str(p.get("name") or "") for p in self._pets}
        self._picked = []
        for d in (current_pets or []):
            if isinstance(d, dict) and str(d.get("pet") or d.get("id") or "").strip():
                self._picked.append({
                    "pet": str(d.get("pet") or d.get("id")).strip(),
                    "name": str(d.get("name") or "").strip(),
                    "equips": [str(x).strip() for x in (d.get("equips") or []) if str(x).strip()],
                })
            elif d:
                self._picked.append({"pet": str(d).strip(), "name": "", "equips": []})

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        note = QLabel("标的是**宠物戴着的样子**（组合外观 ✓）：先挑宠物，再挑它戴的装备，"
                      "然后「加入列表」✓。\n"
                      "鼠标指到格子上会弹出信息窗（名称 / id / 装备 ✓）。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f6368;")
        root.addWidget(note)
        if not self._ok:
            warn = QLabel(self._why)
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #b06000;")
            root.addWidget(warn)

        # ---------------- 已选（要标注的宠物列表 ✓）----------------
        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(4)
        lbl_sel = QLabel("要标注的宠物")
        lbl_sel.setStyleSheet("font-weight: 600;")
        tv.addWidget(lbl_sel)
        self.lst = IconGrid()
        self.lst.setMinimumHeight(37 * 3 + 12)
        tv.addWidget(self.lst)
        srow = QHBoxLayout()
        srow.setSpacing(6)
        self.btn_del = QPushButton("移出选中")
        self.btn_del.clicked.connect(self._remove)
        srow.addWidget(self.btn_del)
        self.lbl_sel = QLabel()
        self.lbl_sel.setStyleSheet("color: #80868b;")
        srow.addWidget(self.lbl_sel, 1)
        tv.addLayout(srow)

        # ---------------- 挑：左宠物 / 右装备 / 预览 ----------------
        bot = QWidget()
        bv = QVBoxLayout(bot)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(4)

        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("输入宠物名或 id 筛选（如「小猫」或「5000000」）…")
        self.ed_filter.textChanged.connect(self._refilter)
        self.ed_filter.setEnabled(self._ok)
        bv.addWidget(self.ed_filter)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)
        l1 = QLabel("宠物")
        l1.setStyleSheet("font-weight: 600;")
        lv.addWidget(l1)
        self.cand_pet = IconGrid()
        # ⚠ `IconGrid` **没有** `selection_changed` 这种信号 ✗（它只有 `hovered` / `activated_id` ✓）
        #   ⇒ 用 QListWidget 自带的 `currentItemChanged` / `itemSelectionChanged` ✓
        #   （本轮第一版就是照想象写了 `selection_changed` ✗ 建窗就 AttributeError ✓）。
        self.cand_pet.currentItemChanged.connect(lambda *_a: self._on_pet_picked())
        lv.addWidget(self.cand_pet, 1)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(4)
        self.lbl_eq = QLabel("它戴的装备（可多选：Ctrl 点/框选 ✓）")
        self.lbl_eq.setStyleSheet("font-weight: 600;")
        rv.addWidget(self.lbl_eq)
        self.cand_eq = IconGrid()
        # 多选 ✓（一只能戴好几件：帽子 + 丝带 ✓）—— 用 Qt 自带的扩展选择 ✓
        #   ⚠ 别自己发明 `setSelectionMode2` 这种 API ✗（本轮第一版就这么写的 ✓）
        self.cand_eq.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cand_eq.itemSelectionChanged.connect(lambda: self._preview())
        rv.addWidget(self.cand_eq, 1)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([380, 380])
        bv.addWidget(split, 1)

        prow = QHBoxLayout()
        prow.setSpacing(8)
        self.lbl_preview = QLabel("（选一只宠物就能看到组合预览）")
        self.lbl_preview.setMinimumWidth(120)
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet("border: 1px solid #dadce0; background: #fafbfc;")
        prow.addWidget(self.lbl_preview)
        self.lbl_match = QLabel()
        self.lbl_match.setStyleSheet("color: #80868b;")
        self.lbl_match.setWordWrap(True)
        prow.addWidget(self.lbl_match, 1)
        bv.addLayout(prow)

        self.split = QSplitter(Qt.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(top)
        self.split.addWidget(bot)
        self.split.setSizes([200, 400])
        root.addWidget(self.split, 1)

        btns = QDialogButtonBox()
        self.btn_add = QPushButton("加入列表")
        self.btn_add.setDefault(True)
        self.btn_add.clicked.connect(self._add)
        btns.addButton(self.btn_add, QDialogButtonBox.AcceptRole)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)
        btns.addButton(self.btn_close, QDialogButtonBox.RejectRole)
        root.addWidget(btns)

        theme.bind_window_state(self, "dlg_pet_picker")
        self._refresh_picked()
        self._refilter()

    # ---------------- 刷新 ----------------
    def _refresh_picked(self):
        rows = [combo_entry({"id": p["pet"], "name": self._name_of.get(p["pet"], p.get("name"))},
                            p["equips"]) for p in self._picked]
        self.lst.set_entries(rows)
        n = len(self._picked)
        self.lbl_sel.setText("共 %d 套" % n if n else "还没有选宠物")
        self._refresh_buttons()

    def _refresh_buttons(self):
        # ⚠ `current_id()` 没选时给的是**空串**（不是 None ✗）⇒ 判真假、别 `is not None` ✓
        self.btn_add.setEnabled(bool(self._ok) and bool(self.cand_pet.current_id()))
        self.btn_del.setEnabled(bool(self.lst.current_id()))

    def _refilter(self):
        self.cand_pet.set_entries([])
        self.cand_eq.set_entries([])
        self._preview()
        if not self._ok:
            self.lbl_match.setText("图库还没导出 ⇒ 没法列宠物（见上面的说明）")
            return
        kw = self.ed_filter.text().strip().lower()
        rows = []
        for p in self._pets[:_MAX_CANDIDATES]:
            nm = str(p.get("name") or "")
            if kw and kw not in str(p["id"]).lower() and kw not in nm.lower():
                continue
            icon = petlib.sprite_icon(petlib.pet_dir(), p["id"])
            rows.append({"id": p["id"], "name": nm or ("（没名字）%s" % p["id"]),
                         "desc": "宠物 %s" % p["id"],
                         "icon": str(icon) if icon else "", "has_img": bool(icon)})
        self.cand_pet.set_entries(rows)
        self.lbl_match.setText("宠物共 %d 只 · 筛出 %d 只（左栏选中后右栏列出它能戴的装备 ✓）"
                               % (len(self._pets), len(rows)) if rows else "没有匹配的宠物")

    def _on_pet_picked(self):
        """左栏选了宠物 ⇒ 右栏换成**它能戴的**那些装备 ✓ + 刷新预览 ✓。"""
        pid = self.cand_pet.current_id()
        rows = []
        if pid and self._ok:
            for e in self._equips:
                if pid not in (e.get("pets") or []):
                    continue                     # ⭐ 戴不了的不列（真源在 `pets` 里 ✓）
                icon = petlib.sprite_icon(petlib.pet_equip_dir(), e["id"])
                rows.append({"id": e["id"], "name": str(e.get("name") or e["id"]),
                             "desc": "宠物装备 %s" % e["id"],
                             "icon": str(icon) if icon else "", "has_img": bool(icon)})
        self.cand_eq.set_entries(rows)
        self.lbl_eq.setText("它戴的装备（%d 件可选 · Ctrl/框选可多选 ✓）" % len(rows)
                            if pid else "它戴的装备（先在左边选一只宠物 ✓）")
        self._preview()
        self._refresh_buttons()

    def _selected_equips(self):
        """右栏**选中**的那几件（多选 ✓）—— 唯一出口 ✓（别在别处各读一遍 ✗）。

        ⚠⚠ **必须读选中项，不能读 `ids()`** ✗ —— `IconGrid.ids()` 给的是**格子里全部** id ✓
          （掉落物那边拿它验"候选列了哪些" ✓），拿它当"选中"就会**一加加一串** ✗
          （本轮第一版就是这么写的 ✓ 冒烟一试就发现"1 件选了却加进去 11 件" ✓）。
        """
        return [str(it.data(Qt.UserRole)) for it in self.cand_eq.selectedItems()
                if it.data(Qt.UserRole)]

    def _preview(self):
        """组合预览（左栏宠物 + 右栏选中的装备 ✓ 现算 ✓ 有缓存 ✓）。"""
        pid = self.cand_pet.current_id()
        eqs = self._selected_equips()
        if not pid or not self._ok:
            self.lbl_preview.setText("（选一只宠物就能看到组合预览）")
            self.lbl_preview.setPixmap(QPixmap())
            return
        d = petlib.compose_pet(pid, eqs)
        if d is None:
            self.lbl_preview.setText("这只做不出组合（没有 stand0 帧 ✗）")
            return
        pm = QPixmap(str(d / "stand0_0.png"))
        if pm.isNull():
            self.lbl_preview.setText("组合图读不出来 ✗")
            return
        self.lbl_preview.setText("")
        self.lbl_preview.setPixmap(pm.scaled(104, 104, Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation))

    # ---------------- 增删 ----------------
    def _add(self):
        pid = self.cand_pet.current_id()
        if not pid:
            return
        eqs = self._selected_equips()
        if any(p["pet"] == pid and p["equips"] == sorted(eqs) for p in self._picked):
            return                               # 同一套（宠物 + 同一组装备）不重复加 ✓
        self._picked.append({"pet": pid, "name": self._name_of.get(pid, ""),
                             "equips": sorted(eqs)})
        self._refresh_picked()

    def _remove(self):
        e = self.lst.current_entry() or {}
        pid = str(e.get("pet") or self.lst.current_id() or "")
        if not pid:
            return
        # ⚠ 同一只宠物可能配了**好几套**（戴不同装备 ✓）⇒ 必须按"**当前选中那一套**"删 ✓
        #   （只按宠物 id 删会删错那一条 ✗ —— 格子里的条目带着 `equips` ✓ 见 `combo_entry` ✓）
        want = sorted(str(x) for x in (e.get("equips") or []))
        for i, p in enumerate(self._picked):
            if p["pet"] == pid and sorted(p["equips"]) == want:
                self._picked.pop(i)
                break
        else:
            for i, p in enumerate(self._picked):
                if p["pet"] == pid:
                    self._picked.pop(i)
                    break
        self._refresh_picked()

    def picked_pets(self):
        """这一趟选完的列表（卡片拿它存进项目 ✓）。"""
        return [dict(p) for p in self._picked]
