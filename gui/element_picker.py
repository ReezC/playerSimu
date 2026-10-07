"""「**地区选择**」通用弹窗（用户 2026-10-05 ✓）—— 选一个（或多选）**地图元素**。

用户原话："选择「平台站桩」策略类型后，出现「选择站桩地点」→ 地区选择弹窗（**一个新的
通用弹窗类**，用于选择 foothold 集合、单个 foothold 或绳梯、传送门等地图元素）"✓
—— 「通用」两个字是重点：**站桩地点**（单选）和**拾取掉落地区**（多选）都用它 ✓，
以后别的功能要选地图元素也复用它 ✓。

**四类元素**（与 `decision.agent.DecisionSettings.SPOT_KINDS` / `core.zones` 的「地点」
结构**同一份口径** ✓）：
  · **集合**（`{"kind":"set","name":"二楼"}`）；
  · **单条 foothold**（`{"kind":"foothold","fid":"41","ratio":100.0}` —— `ratio` = 走到的百分比）；
  · **绳梯**（`{"kind":"ladder","lid":"L2","x":1234}`）；
  · **传送门**（`{"kind":"portal","pn":"sp","x":100,"y":200}`）。

⚠ **"按地点把元素找回来"那几件全在 `core.zones`**（`find_foothold` / `find_ladder` /
  `find_portal` / `element_span` / `element_label` ✓ **一处实现** ✓）—— 本文件只做
  "界面"那一层（列表 / 高亮 / 点选），**不许**再写一份"怎么找"✗（`gui/live_thread.py`
  的解析器也走那一份 ✓）。

⚠ 只读：本弹窗**不改地形**（要改去「寻路编辑器」✓）。
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from core import zones as zones_mod
from gui import theme
from gui.foothold_picker import FootholdPicker
from gui.widgets import NoWheelComboBox
from gui.zone_editor import C_PORTAL, C_SEL, dist_point_seg, dist_to_ladder

#: 四类的**固定顺序**（列表分组顺序 = 集合 → foothold → 绳梯 → 传送门 ✓）。
KIND_ORDER = ("set", "foothold", "ladder", "portal")

#: 图上点选容差（**屏幕像素** ⇒ 按当前缩放折成世界距离 ✓，同 `foothold_picker` 的口径 ✓）。
PICK_SCREEN_PX = 14.0


# ══════════════════════════════════════════
# 纯逻辑（**不碰 Qt** ⇒ 可以在离屏自检里用真地形直接测 ✓）
# ══════════════════════════════════════════

def spot_key(spot):
    """一个「地点」的**比较键**（判断"是不是同一个地点"用 ✓ —— 存/读回、列表去重都靠它）。

    ⚠ 坐标**取整到 1 个世界像素**再比：几何量存的是浮点（口径见 SKILL 约定 11 ✓），
      而"同一个地点"这件事在 1 像素的尺度上本来就分不出来 ✓。
    """
    spot = spot or {}
    k = str(spot.get("kind") or "")
    try:
        if k == "set":
            return ("set", str(spot.get("name") or "").strip())
        if k == "foothold":
            return ("foothold", str(spot.get("fid") or "").strip())
        if k == "ladder":
            return ("ladder", str(spot.get("lid") or "").strip(),
                    round(float(spot.get("x") or 0.0)))
        if k == "portal":
            return ("portal", str(spot.get("pn") or "").strip(),
                    round(float(spot.get("x") or 0.0)), round(float(spot.get("y") or 0.0)))
    except (TypeError, ValueError):
        return (k, "")
    return (k, "")


def _entry(spot, terrain, zones):
    """一个「地点」→ 列表条目 `{"spot", "kind", "label", "x", "y"}`（排序/显示用 ✓）。"""
    kind = str(spot.get("kind") or "")
    x, y = 0.0, 0.0
    try:
        if kind == "set":
            sp = zones_mod.set_span(terrain, ((zones.sets or {}).get(
                str(spot.get("name") or "")) or {}).get("footholds") or [])
            if sp is not None:
                x, y = (float(sp[0]) + float(sp[1])) / 2.0, (float(sp[2]) + float(sp[3])) / 2.0
        elif kind == "foothold":
            f = zones_mod.find_foothold(terrain, spot.get("fid"))
            if f is not None:
                x = (float(f.left) + float(f.right)) / 2.0
                y = float(f.y_at(x))
        elif kind == "ladder":
            L = zones_mod.find_ladder(terrain, spot)
            if L is not None:
                x = float(L.x)
                y = (float(L.y1) + float(L.y2)) / 2.0
        elif kind == "portal":
            p = zones_mod.find_portal(terrain, spot)
            if p is not None:
                x, y = float(p.x), float(p.y)
    except Exception:                            # noqa: BLE001 —— 数据坏了照旧列出来 ✓
        pass
    return {"spot": dict(spot or {}), "kind": kind, "x": x, "y": y,
            "label": zones_mod.element_label(spot, terrain, zones)}


def list_elements(terrain, zones, kinds=None):
    """地图上**可选的元素** → `[_entry…]`（按 类型顺序 → x 升序 ✓）。

    `kinds` = 只要哪几类（`None` = 四类全要 ✓）。
    ⚠ 墙（`is_wall`）**不给选**（站不上去 / 不是"地点"✓）；没圈进集合的元素**照样列出来**
      （"走不到"是使用时的事，不是"看不见"✗ —— 但列表里会照实写上 ✓）。
    """
    ks = tuple(kinds or KIND_ORDER)
    if terrain is None:
        return []
    out = []
    if "set" in ks and zones is not None:
        for name in (zones.sets or {}):
            out.extend([_entry({"kind": "set", "name": str(name)}, terrain, zones)])
        out.sort(key=lambda e: e["x"])
    if "foothold" in ks:
        fhs = []
        for f in (getattr(terrain, "footholds", None) or ()):
            if getattr(f, "is_wall", False):
                continue
            fhs.append(_entry({"kind": "foothold", "fid": str(f.fid),
                               "ratio": 100.0}, terrain, zones))
        fhs.sort(key=lambda e: e["x"])
        out.extend(fhs)
    if "ladder" in ks:
        lads = []
        try:
            lids = zones_mod.ladder_ids(terrain)
        except Exception:                        # noqa: BLE001
            lids = {}
        for L in (getattr(terrain, "ladders", None) or ()):
            lads.append(_entry({"kind": "ladder", "lid": str(lids.get(id(L)) or ""),
                                "x": float(L.x)}, terrain, zones))
        lads.sort(key=lambda e: e["x"])
        out.extend(lads)
    if "portal" in ks:
        pors = []
        for p in (getattr(terrain, "portals", None) or ()):
            pors.append(_entry({"kind": "portal", "pn": str(getattr(p, "pn", "") or ""),
                                "x": float(p.x), "y": float(p.y)}, terrain, zones))
        pors.sort(key=lambda e: e["x"])
        out.extend(pors)
    return out


def nearest_element(terrain, zones, wx, wy, tol, kinds=None):
    """图上一点 `(wx, wy)`（**世界坐标** ✓）→ **最近的可选元素**的「地点」/ `None`。

    ⚠ 只比 **foothold（点到线段）/ 绳梯（点到竖线段）/ 传送门（点到点）** ——
      **集合不进这一层**（它是若干 foothold 的集合体，图上点中的一定是某条 foothold ✓；
      要选集合请从**列表**里点 ✓）。距离 > `tol` ⇒ `None`（点空白 = 没点中 ✓）。
    """
    ks = set(kinds or KIND_ORDER)
    best = None
    try:
        if "foothold" in ks:
            for f in (getattr(terrain, "footholds", None) or ()):
                if getattr(f, "is_wall", False):
                    continue
                d = dist_point_seg(wx, wy, float(f.x1), float(f.y1), float(f.x2), float(f.y2))
                if best is None or d < best[0]:
                    best = (d, {"kind": "foothold", "fid": str(f.fid), "ratio": 100.0})
        if "ladder" in ks:
            for L in (getattr(terrain, "ladders", None) or ()):
                d = dist_to_ladder(L, wx, wy)
                if best is None or d < best[0]:
                    best = (d, {"kind": "ladder", "lid": "", "x": float(L.x)})
        if "portal" in ks:
            for p in (getattr(terrain, "portals", None) or ()):
                d = ((wx - float(p.x)) ** 2 + (wy - float(p.y)) ** 2) ** 0.5
                if best is None or d < best[0]:
                    best = (d, {"kind": "portal", "pn": str(getattr(p, "pn", "") or ""),
                                "x": float(p.x), "y": float(p.y)})
    except (TypeError, ValueError):
        return None
    if best is None or best[0] > float(tol):
        return None
    # 绳梯/传送门把**稳定 id** 补上（存下去才有"显示名" ✓ 见 `core.zones` 的说明 ✓）
    spot = best[1]
    if spot["kind"] == "ladder":
        try:
            lids = zones_mod.ladder_ids(terrain)
        except Exception:                        # noqa: BLE001
            lids = {}
        for L in (getattr(terrain, "ladders", None) or ()):
            if abs(float(L.x) - float(spot["x"])) < 0.5:
                spot["lid"] = str(lids.get(id(L)) or "")
                break
    elif spot["kind"] == "portal":
        for p in (getattr(terrain, "portals", None) or ()):
            if (abs(float(p.x) - float(spot["x"])) < 0.5
                    and abs(float(p.y) - float(spot["y"])) < 0.5):
                spot["pn"] = str(getattr(p, "pn", "") or "")
                break
    return spot


# ══════════════════════════════════════════
# 只读地图视图（复用「idle 回归」那块视图的全部画法 ✓）
# ══════════════════════════════════════════

class ElementView(FootholdPicker):
    """**只读**地图视图：画底图 + 全部 foothold + 集合名 + 绳梯 + **传送门** + 选中高亮。

    ⭐ 继承 `FootholdPicker`（`gui/foothold_picker.py` ✓）而不是另写一份：那块视图已经把
      「底图换算 / 全部 foothold / 集合名 / 绳梯 / 笔宽随缩放 / 呼吸 / 缩放平移」都做好了
      （用户 2026-09-28 亲口要过"要跟 foothold 编辑器**一模一样**"✓）⇒ 这里只**多画一层
      传送门** + 把"选中的那一条"换成**任意地点** ✓（少写一份必然又漂 ✗）。
    """

    #: 图上点选到一个元素 ⇒ 发它的「地点」dict（点空白 ⇒ `None` ✓）。
    picked_spot = pyqtSignal(object)

    def __init__(self, terrain, zones, parent=None):
        # ⚠ `set_name=""` ⇒ 父类不把任何 foothold 当"本集合"着色（这里没有"本集合" ✓）
        super().__init__(terrain, zones, "", parent=parent)
        self._sel_spot = None
        self._portal_items = []
        self.setToolTip("**点一个元素**（foothold / 绳梯 / 传送门）⇒ 选中它 ✓（点空白 = 清空）。\n"
                        "滚轮缩放 / 中键拖 = 平移 / 双击 = 适应窗口 ✓。\n"
                        "⚠ 只读：改地形请去「寻路编辑器」✓；选**集合**请从左边列表里点 ✓。")
        self._rebuild()

    # ---------------- 画 ----------------

    def _rebuild(self):
        """父类那份画法（底图 / foothold / 集合名 / 绳梯）**原样**跑一遍，再补两层 ✓。"""
        FootholdPicker._rebuild(self)
        self._add_portals()
        sp = getattr(self, "_sel_spot", None)
        if sp:
            self._mark_spot(sp)

    def _add_portals(self):
        """画**传送门**（紫实心圆 + 名字 ✓ —— 父类原来不画它 ✓ 口径抄 `zone_editor` ✓）。"""
        self._portal_items = []
        for p in (getattr(self.terrain, "portals", None) or []):
            try:
                it = self._scene.addEllipse(float(p.x) - 7, float(p.y) - 7, 14, 14,
                                            self._pen(C_PORTAL),
                                            QBrush(QColor(C_PORTAL)))
            except Exception:                    # noqa: BLE001
                continue
            it.setZValue(6)
            self._portal_items.append((p, it))
            name = str(getattr(p, "pn", "") or "")
            if not name:
                continue
            from PyQt5.QtWidgets import QGraphicsSimpleTextItem
            t = QGraphicsSimpleTextItem(name)
            fnt = QFont()
            fnt.setPixelSize(34)
            t.setFont(fnt)
            t.setBrush(QBrush(QColor(C_PORTAL)))
            t.setPos(float(p.x) + 10, float(p.y))
            t.setZValue(7)
            self._scene.addItem(t)

    def _item_for(self, spot):
        """某个「地点」对应的图元（集合没有单个图元 ⇒ `None` ✓）。"""
        kind = str((spot or {}).get("kind") or "")
        if kind == "foothold":
            return self._items.get(str((spot or {}).get("fid") or ""))
        if kind == "ladder":
            L = zones_mod.find_ladder(self.terrain, spot)
            if L is None:
                return None
            for it, _b, _s, _h in self._lines:
                try:
                    ln = it.line()
                except Exception:                # noqa: BLE001
                    continue
                if (abs(ln.x1() - float(L.x)) < 0.5 and abs(ln.x2() - float(L.x)) < 0.5
                        and abs(min(ln.y1(), ln.y2())
                                - min(float(L.y1), float(L.y2))) < 0.5):
                    return it
            return None
        if kind == "portal":
            p = zones_mod.find_portal(self.terrain, spot)
            for obj, it in self._portal_items:
                if obj is p and p is not None:
                    return it
            return None
        return None

    def _mark_spot(self, spot):
        """把选中那个元素标出来（**只动"基准色 + 呼吸"这两样** ✓ 不重建场景 ✓）。

        · **集合** ⇒ 它那一批 foothold **只进呼吸表、不改基准色**（口径同父类的
          `highlight_set` ✓ —— 两个语义不打架 ✓）；
        · 其余三类 ⇒ 那一个图元改成 `C_SEL` + 压顶 + 呼吸 ✓。
        ⚠ 这里**不能**调父类的 `highlight_set()`：它会再 `_rebuild()` ⇒ 与我的 `_rebuild`
          互相调 = 无限递归 ✗（写这条时就得避开 ✓）。
        """
        kind = str((spot or {}).get("kind") or "")
        if kind == "set":
            fids = set(str(x) for x in ((self.zones.sets or {}).get(
                str(spot.get("name") or "")) or {}).get("footholds") or [])
            items = set(self._items[f] for f in fids if f in self._items)
            self._lines = [(i, b, s_, (i in items)) for i, b, s_, _ in self._lines]
            return
        it = self._item_for(spot)
        if it is None:
            return
        it.setZValue(9)
        self._lines = [(i, (C_SEL if i is it else b), s_, (i is it))
                       for i, b, s_, _ in self._lines]
        self.viewport().update()

    # ---------------- 选 ----------------

    def current_spot(self):
        """当前选中的「地点」/ `None` ✓。"""
        return self._sel_spot

    def set_spot(self, spot, focus=False):
        """外部设选中（`None` = 清空 ✓）；没变就不重画 ✓；`focus=True` ⇒ 顺带居中放大 ✓。"""
        a = spot_key(spot)
        b = spot_key(self._sel_spot) if self._sel_spot else None
        if (spot is None) == (self._sel_spot is None) and a == b:
            if focus and spot:
                self.focus_spot(spot)
            return
        self._sel_spot = dict(spot) if spot else None
        self._rebuild()
        if focus and self._sel_spot:
            self.focus_spot(self._sel_spot)

    def focus_spot(self, spot):
        """居中到某个地点（缩放**只往大调** ✓ —— 同父类 `focus_on` 的口径 ✓）。"""
        ent = _entry(spot, self.terrain, self.zones)
        cx, cy = float(ent["x"]), float(ent["y"])
        try:
            k = max(1e-6, float(self.transform().m11()))
        except Exception:                        # noqa: BLE001
            k = 1.0
        self.centerOn(cx, cy)
        if k < 2.2:
            self.scale(2.2 / k, 2.2 / k)
            self.centerOn(cx, cy)
        self.viewport().update()

    def mousePressEvent(self, e):
        """左键点图 ⇒ **就近选中一个元素**（屏幕像素容差 ⇒ 按缩放折成世界距离 ✓）。"""
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        """⚠ 用 release 而不是 press：父类的 press 会先处理"点 foothold"（它自己那套 ✓）
        而我们这里要点**任意元素** ⇒ 放在 release 上、按世界坐标就近判一次 ✓（点一下只判一次 ✓）。"""
        super().mouseReleaseEvent(e)
        try:
            if e.button() != Qt.LeftButton:
                return
            p = self.mapToScene(e.pos())
            tol = PICK_SCREEN_PX / max(1e-6, float(self._px()))
            spot = nearest_element(self.terrain, self.zones, float(p.x()), float(p.y()), tol)
            # ⭐ **点中元素 ⇒ 聚焦 + 呼吸**（用户 2026-10-06 ✓ 原话："选中地图元素时视图区要
            #   **聚焦 + 呼吸高亮**"✓）。呼吸本来就有（`_mark_spot` 把那个图元换选中色并进
            #   呼吸表 ✓）；缺的是**聚焦**（`focus_spot` 居中 + 放大到 ≥2.2 倍 ✓ 早就写好了 ✓
            #   只是没人调 ✗）。⚠ 点空白（`spot is None`）**不聚焦** ✗ —— 那一下是"取消选中"，
            #   不该把视图拽走 ✓。
            self.set_spot(spot, focus=spot is not None)
            self.picked_spot.emit(spot)
        except Exception:                        # noqa: BLE001 —— 点选坏了别把视图弄崩 ✗
            pass


# ══════════════════════════════════════════
# 弹窗
# ══════════════════════════════════════════

class ElementPickerDialog(QDialog):
    """「地区选择」：左列表（可筛选/搜索/勾选）+ 右地图（点选/高亮）+ 明细行 ✓。

    · `multi=False`（**默认**）⇒ 单选，给「站桩地点」用 ✓；
    · `multi=True` ⇒ 每行一个勾选框，给「拾取掉落地区」（**多个地点** ✓）用 ✓；
    · `init` = 打开时该选中/勾上的地点（单个 dict 或一串 ✓）。
    结果：`selected_spots()` → `[spot…]`（单选模式最多一个 ✓；没选 = `[]` ✓）。
    """

    def __init__(self, parent, terrain, zones, title="选择地点",
                 multi=False, init=None, kinds=None):
        super().__init__(parent)
        self.setWindowTitle(str(title))
        self._terrain, self._zones = terrain, zones
        self._multi = bool(multi)
        if isinstance(init, dict):
            self._init = [init]
        else:
            self._init = [x for x in (init or []) if isinstance(x, dict)]
        self._entries = list_elements(terrain, zones, kinds=kinds)

        root = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("类型"))
        self.cmb_kind = NoWheelComboBox()
        self.cmb_kind.addItem("（全部）", "")
        for k in (kinds or KIND_ORDER):
            self.cmb_kind.addItem(zones_mod.ELEMENT_LABELS.get(k, k), k)
        self.cmb_kind.currentIndexChanged.connect(lambda _i: self._fill())
        row.addWidget(self.cmb_kind)
        row.addWidget(QLabel("搜索"))
        self.ed_find = QLineEdit()
        self.ed_find.setPlaceholderText("按名字 / 编号过滤…")
        self.ed_find.textChanged.connect(lambda _t: self._fill())
        row.addWidget(self.ed_find, 1)
        root.addLayout(row)

        body = QHBoxLayout()
        self.lst = QListWidget()
        self.lst.setMinimumWidth(300)
        self.lst.currentItemChanged.connect(lambda *a: self._on_row())
        self.lst.itemChanged.connect(self._on_item_changed)
        body.addWidget(self.lst, 2)
        self.view = ElementView(terrain, zones)
        self.view.picked_spot.connect(self._on_picked_spot)
        body.addWidget(self.view, 3)

        # ⭐⭐ **多选模式的「已选」列表**（用户 2026-10-05 ✓ 原话：「掉落地区」的顺序）——
        #   顺序 = **依次走的顺序** ✓（`agent` 那边就是按这个列表一路 `plan_and_start_route` ✓）
        #   ⇒ 必须能**上移 / 下移**（勾选的先后不等于想要的走法顺序 ✗）。
        #   ⚠ 只在多选模式显示（单选模式"顺序"没有意义 ✓）。
        self.box_chosen = QWidget()
        _cv = QVBoxLayout(self.box_chosen)
        _cv.setContentsMargins(0, 0, 0, 0)
        _cv.addWidget(QLabel("已选（顺序 = 依次走的顺序）"))
        self.lst_chosen = QListWidget()
        self.lst_chosen.setMinimumWidth(220)
        self.lst_chosen.currentItemChanged.connect(lambda *a: self._on_chosen_row())
        _cv.addWidget(self.lst_chosen, 1)
        _cb = QHBoxLayout()
        for _txt, _slot, _tip in (
                ("上移", lambda: self._move_chosen(-1), "往前挪一位（先走它）"),
                ("下移", lambda: self._move_chosen(1), "往后挪一位"),
                ("移除", self._remove_chosen, "从「已选」里去掉（主列表的勾也跟着取消）")):
            _b = QPushButton(_txt)
            _b.setToolTip(_tip)
            _b.clicked.connect(_slot)
            _cb.addWidget(_b)
        _cb.addStretch(1)
        _cv.addLayout(_cb)
        body.addWidget(self.box_chosen, 2)
        self.box_chosen.setVisible(self._multi)
        root.addLayout(body, 1)

        self.lbl_detail = QLabel("")
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setStyleSheet("color: #80868b;")
        root.addWidget(self.lbl_detail)

        btns = QHBoxLayout()
        self.btn_clear = QPushButton("清除选择")
        self.btn_clear.setToolTip("什么都不选 ⇒ 调用方按「没选」处理 ✓")
        self.btn_clear.clicked.connect(self._clear)
        btns.addWidget(self.btn_clear)
        btns.addStretch(1)
        self.btn_ok = QPushButton("确定")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        root.addLayout(btns)

        self._fill_chosen()          # ⭐ 先灌「已选」（顺序 = `init` 给的顺序 ✓）
        self._fill()                 # 再刷主列表（勾选状态读「已选」✓ 两边永远一致 ✓）
        if self._init:
            self._select_first_init()
        self.resize(860, 560)
        theme.bind_window_state(self, "element_picker")

    # ---------------- 列表 ----------------

    def _row_item(self, spot, label=None):
        """造一行（`UserRole` 存那份「地点」dict ✓）。"""
        it = QListWidgetItem(label if label is not None
                             else zones_mod.element_label(spot, self._terrain, self._zones))
        it.setData(Qt.UserRole, dict(spot))
        return it

    def _fill_chosen(self):
        """把 `init` 按**给定顺序**灌进「已选」（只在打开时用一次 ✓）。

        ⚠ 之后这个列表由**勾选**（追加）/ **移除** / **上移下移**维护 ✓ —— 不再重灌 ✗
          （重灌会把用户刚调好的顺序冲掉 ✓）。
        """
        self.lst_chosen.blockSignals(True)
        try:
            self.lst_chosen.clear()
            if not self._multi:
                return
            for sp in self._init:
                self.lst_chosen.addItem(self._row_item(sp))
        finally:
            self.lst_chosen.blockSignals(False)

    def _fill(self):
        """按「类型 + 搜索词」重建**主**列表；**勾选状态跟着「已选」** ✓。"""
        keep = {spot_key(x) for x in self._chosen()}
        k = str(self.cmb_kind.currentData() or "")
        q = str(self.ed_find.text() or "").strip().lower()
        self.lst.blockSignals(True)
        try:
            self.lst.clear()
            for e in self._entries:
                if k and e["kind"] != k:
                    continue
                if q and q not in str(e["label"]).lower():
                    continue
                it = self._row_item(e["spot"], label=e["label"])
                if self._multi:
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if spot_key(e["spot"]) in keep
                                     else Qt.Unchecked)
                self.lst.addItem(it)
        finally:
            self.lst.blockSignals(False)
        self._refresh_detail()

    def _chosen(self):
        """**选中的地点，按"依次走的顺序"** ✓。

        · 单选模式 = 主列表的当前行（最多一个 ✓）；
        · 多选模式 = 右边那个「**已选**」列表，**顺序就是它的行序** ✓
          （上移 / 下移改的就是它 ✓ —— 不是勾选的先后 ✗）。
        """
        if not self._multi:
            it = self.lst.currentItem()
            return [it.data(Qt.UserRole)] if it is not None else []
        out = []
        for i in range(self.lst_chosen.count()):
            sp = self.lst_chosen.item(i).data(Qt.UserRole)
            if isinstance(sp, dict):
                out.append(sp)
        return out

    # ---------------- 多选：维护「已选」的顺序 ----------------

    def _append_chosen(self, spot):
        self.lst_chosen.blockSignals(True)
        try:
            self.lst_chosen.addItem(self._row_item(spot))
        finally:
            self.lst_chosen.blockSignals(False)

    def _remove_chosen_key(self, key):
        self.lst_chosen.blockSignals(True)
        try:
            for i in range(self.lst_chosen.count()):
                if spot_key(self.lst_chosen.item(i).data(Qt.UserRole)) == key:
                    self.lst_chosen.takeItem(i)
                    break
        finally:
            self.lst_chosen.blockSignals(False)

    def _on_item_changed(self, it):
        """主列表**勾上 / 取消** ⇒ 同步「已选」（新勾的**追加在末尾** ✓ 再自己上下移 ✓）。"""
        if not self._multi:
            self._refresh_detail()
            return
        spot = it.data(Qt.UserRole)
        if not isinstance(spot, dict):
            return
        key = spot_key(spot)
        have = {spot_key(self.lst_chosen.item(i).data(Qt.UserRole))
                for i in range(self.lst_chosen.count())}
        if it.checkState() == Qt.Checked:
            if key not in have:
                self._append_chosen(spot)
        elif key in have:
            self._remove_chosen_key(key)
        self._refresh_detail()

    def _sync_main_checks(self):
        """把主列表的勾**按「已选」重新对一遍**（移除 / 清空之后用 ✓）。"""
        want = {spot_key(self.lst_chosen.item(i).data(Qt.UserRole))
                for i in range(self.lst_chosen.count())}
        self.lst.blockSignals(True)
        try:
            for i in range(self.lst.count()):
                it = self.lst.item(i)
                it.setCheckState(Qt.Checked if spot_key(it.data(Qt.UserRole)) in want
                                 else Qt.Unchecked)
        finally:
            self.lst.blockSignals(False)

    def _remove_chosen(self):
        it = self.lst_chosen.currentItem()
        if it is None:
            return
        self._remove_chosen_key(spot_key(it.data(Qt.UserRole)))
        self._sync_main_checks()
        self._refresh_detail()

    def _move_chosen(self, delta):
        """上移 / 下移一行（**顺序 = 依次走的顺序** ✓）。到头就不动（不绕圈 ✓）。"""
        i = int(self.lst_chosen.currentRow())
        j = i + int(delta)
        if i < 0 or j < 0 or j >= self.lst_chosen.count():
            return
        it = self.lst_chosen.takeItem(i)
        self.lst_chosen.insertItem(j, it)
        self.lst_chosen.setCurrentRow(j)
        self._refresh_detail()

    def _on_chosen_row(self):
        it = self.lst_chosen.currentItem()
        sp = it.data(Qt.UserRole) if it is not None else None
        if isinstance(sp, dict):
            self.view.set_spot(sp)
        self._refresh_detail()

    def _select_first_init(self):
        """打开时把 `init` 第一项**在列表里选出来**（可能在筛掉的范围里 ⇒ 尽力而为 ✓）。"""
        want = spot_key(self._init[0])
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if spot_key(it.data(Qt.UserRole)) == want:
                self.lst.setCurrentItem(it)
                return
        self.view.set_spot(self._init[0])

    def _on_row(self):
        it = self.lst.currentItem()
        spot = it.data(Qt.UserRole) if it is not None else None
        # ⭐ 选行 ⇒ 也**聚焦 + 呼吸**（用户 2026-10-06 ✓ 同 `mouseReleaseEvent` 那条口径 ✓）：
        #   列表里挑一行、图上立刻"飞过去并闪给你看" ✓；取消选中（`None`）不聚焦 ✓。
        self.view.set_spot(spot, focus=isinstance(spot, dict))
        self._refresh_detail()

    def _on_picked_spot(self, spot):
        """图上点中一个元素 ⇒ 把列表同步到那一行（**屏蔽信号**防回去又设一遍 ✓）。"""
        want = spot_key(spot)
        self.lst.blockSignals(True)
        try:
            for i in range(self.lst.count()):
                it = self.lst.item(i)
                if spot_key(it.data(Qt.UserRole)) == want:
                    self.lst.setCurrentItem(it)
                    break
        finally:
            self.lst.blockSignals(False)
        self._refresh_detail()

    def _refresh_detail(self):
        """明细行：选中的那几个（多选时**按顺序**全列）+ 一句"用它时会怎样" ✓。"""
        pick = self._chosen()
        if not pick:
            self.lbl_detail.setText("没选任何地点 ⇒ 调用方按「未配置」处理 ✓")
            return
        lines = [zones_mod.element_label(x, self._terrain, self._zones) for x in pick]
        if self._multi:
            self.lbl_detail.setText(
                "已选 %d 个（**按「已选」的顺序依次走**）：%s"
                % (len(pick), " → ".join(lines)))
        else:
            self.lbl_detail.setText("已选：%s" % lines[0])

    def _clear(self):
        """清空（多选 ⇒ 「已选」与主列表的勾一起清 ✓）。"""
        self.lst_chosen.blockSignals(True)
        try:
            self.lst_chosen.clear()
        finally:
            self.lst_chosen.blockSignals(False)
        self.lst.blockSignals(True)
        try:
            for i in range(self.lst.count()):
                if self._multi:
                    self.lst.item(i).setCheckState(Qt.Unchecked)
            self.lst.setCurrentItem(None)
        finally:
            self.lst.blockSignals(False)
        self.view.set_spot(None)
        self._refresh_detail()

    # ---------------- 结果 ----------------

    def selected_spots(self):
        """选中的地点（**没选 = 空列表** ✓）。"""
        return list(self._chosen())

    def selected_spot(self):
        """单选模式下的那一个（没有 ⇒ `None` ✓）。"""
        got = self._chosen()
        return got[0] if got else None


__all__ = ["ElementPickerDialog", "ElementView", "list_elements",
           "nearest_element", "spot_key", "KIND_ORDER", "PICK_SCREEN_PX"]
