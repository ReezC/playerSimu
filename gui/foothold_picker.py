"""「**idle 回归 foothold**」的**只读集合视图**（用户 2026-09-28 要求 ✓）。

用户原话："编辑战斗区域的**子弹窗**需要有『foothold 集合编辑器』的**同款视图（只读）**，
可以通过**点选**来**查看 foothold 参数**、**配置「idle回归foothold」**"✓。

⇒ 这里两个职责：
  · **看**：把这个**集合**包含的 foothold 画出来（**底图打底** ✓，换算与
    `gui/zone_editor._add_background` / `tools/map_terrain_view.render` **同一套** ✓），
    点一条 ⇒ 旁边那行显示它的参数（id / 墙或地板 / x 范围 / 长 / 面 y ✓ —— 口径抄
    `gui/zone_editor._refresh_status` 里那一段 ✓）；
  · **配**：**点中的那条就是 `idle_foothold`** ✓（再点一次空白 = 清空 ✓）。
    ⚠ 为什么"点选即配置"而不是"点选只看、另按按钮才配"：用户那句话是
      "通过点选来**查看**参数、**配置** idle 回归 foothold" ⇒ 一次点选同时满足两件事最省 ✓；
      真想要"只看不配"，把 `point_to_config` 关掉即可（留给以后 ✓）。

⚠ **只读**（这是与「寻路编辑器」最重要的区别）：本视图**不改地形** —— 不移动、不新增、
  不删 foothold ✓，只画 + 选 ✓。真正的编辑仍在「寻路编辑器」里 ✓。

⚠ 为什么**单独一个文件**（而不是塞进 `zone_editor.py`）：那个文件两千多行、被
  `tools/selftest_zone_editor.py` 大量钉着 ⇒ 只读摊位放这儿，**两边互不影响** ✓；
  只从它那儿拿两样现成的（`ZoomPanView` 的缩放/平移、`dist_to` 的点线距离 ✓）。
"""

import math

from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QGraphicsLineItem, QGraphicsPixmapItem,
                             QGraphicsScene, QHBoxLayout, QLabel, QVBoxLayout,
                             QWidget)

import math as _math  # noqa: F401  （呼吸公式在下面的 `_breath_amount` 里用 ✓）

from gui import theme
from gui.canvas import ZoomPanView
# ⚠ **不许裸用 `QComboBox`**（docs/UI规范.md：滚轮会误改参数 ✗）⇒ 用项目那个不吃滚轮的 ✓
#   （`tools/check_ui` 会当场抓到 ✓ —— 这次就是它抓出来的 ✓）。
from gui.widgets import NoWheelComboBox
# ⭐⭐ **配色 / 字号 / 呼吸节奏一律从 `zone_editor` 拿**（用户 2026-09-28：
#   "**字体线段的配色、粗细等还是跟 foothold 编辑器不一样**"✗）——
#   ⛔ **绝不在这里再写一份色值** ✗：两处一旦各写一份，必然漂（§12 的老教训 ✓），
#   而"要跟编辑器一模一样"这件事**只有共用常量**才做得稳 ✓。
from gui.zone_editor import (C_FLOOR, C_IN_SET, C_LADDER, C_LADDER_ID, C_SEL,
                             C_SETNAME, C_WALL, LABEL_PX, LADDER_PX, PULSE_MS,
                             PULSE_STEP)
from gui.zone_editor import dist_to          # 点线距离：一处实现（`zone_editor` 里那份 ✓）

#: 点选时"屏幕像素"级别的容差（会自动按当前缩放折成世界距离 ✓）。
#: 为什么按屏幕像素定：放大看细节时，"看着点中了"和"真的点中了"总得一致 ✓
#: （同 `zone_editor.pick_at` 的 tol 口径 ✓）。
PICK_SCREEN_PX = 14.0

#: ⚠ **集合名 / 绳梯编号的字高与呼吸节奏都不在这里定** —— 直接用 `zone_editor` 的
#: `LABEL_PX` / `LADDER_PX` / `PULSE_MS` / `PULSE_STEP`（**一处口径** ✓ 见上面的 import ✓）：
#: 用户的判据就是"**看着跟 foothold 编辑器一样**"✓，各写一份数字必然漂 ✗。

#: ⭐ 「聚焦到某条 foothold」时至少放大到多少倍（已经更大就不缩回去 ✓ 见 `focus_on` ✓）。
#: ⚠ 这个是**本视图自己**的交互手感（编辑器里没有对应功能 ✓）⇒ 不进那边的口径 ✓。
FOCUS_SCALE = 2.2


# ══════════════════════════════════════════
# 纯逻辑（**不碰 Qt** ⇒ 可以在自检里用真地形直接测 ✓）
# ══════════════════════════════════════════
# ⚠ 为什么单拎出来（2026-09-28 踩过 ✗）：本文件那个 `QGraphicsView` 子类**在离屏自检里建不出来**
#   （进程原生崩溃 `0xC0000409` ✗，两个套件都一样 ⇒ 是环境限制 ✓），
#   所以"本集合有哪几条 / 点中哪一条 / 那条的参数"这几件**判据**必须能**脱离 Qt** 测 ✓。

def set_fids(zones, set_name):
    """集合 `set_name` 里有哪几条 foothold（**id 是字符串** ✓ 见 `core/zones` 的 `sets` 结构 ✓）。"""
    try:
        s = (zones.sets or {}).get(str(set_name or "")) or {}
        return [str(x) for x in (s.get("footholds") or [])]
    except Exception:                       # noqa: BLE001 —— 数据坏了当空集合（别炸弹窗 ✗）
        return []


def footholds_of(terrain, zones, set_name):
    """本集合的 foothold 对象（顺序 = 地形里的顺序 ✓）。"""
    want = set(set_fids(zones, set_name))
    return [f for f in (terrain.footholds or []) if str(f.fid) in want]


def fid_info(terrain, zones, set_name, fid):
    """一条 foothold 的**参数**一行字（口径抄 `gui/zone_editor._refresh_status` ✓）；找不到给 `""` ✓。"""
    fid = str(fid or "")
    for f in footholds_of(terrain, zones, set_name):
        if str(f.fid) != fid:
            continue
        mid = (float(f.x1) + float(f.x2)) / 2.0
        try:
            yy = float(f.y_at(mid))
        except Exception:                   # noqa: BLE001
            yy = float(f.y1)
        return ("#%s（%s）　x %d..%d　长 %d　y=%d"
                % (f.fid, "墙" if f.is_wall else "地板",
                   round(f.x1), round(f.x2),
                   round(abs(float(f.x2) - float(f.x1))), round(yy)))
    return ""


def pick_fid(terrain, zones, set_name, x, y, tol):
    """场景（= 世界 ✓）坐标附近最近的那条 foothold ⇒ 它的 id（没命中给 `""` ✓）。

    ⚠ 只有**本集合**里的才参与 ✓（点到别的集合的线不该改 idle 回归点 ✗）。
    """
    want = set(set_fids(zones, set_name))
    best, best_d = None, None
    for f in (terrain.footholds or []):
        if str(f.fid) not in want:
            continue
        d = dist_to(f, float(x), float(y))
        if d <= float(tol) and (best_d is None or d < best_d):
            best, best_d = f, d
    return str(best.fid) if best is not None else ""


class FootholdPicker(ZoomPanView):
    """只读的「集合 foothold 视图」：点一条 ⇒ 它就是 `idle_foothold` ✓。

    信号 `picked(fid)`：选中的 foothold id（`""` = 点空白 ⇒ **清空** ✓）。
    """

    picked = pyqtSignal(str)

    def sizeHint(self):
        """**别拿场景尺寸当"我想要多大"**（2026-09-28 现场修 ✗ —— 用户报"**编辑战斗区域子窗口
        也太大了，还不让缩小**"✓）。

        `QGraphicsView` 默认把**场景尺寸**当 sizeHint，而本视图的场景 = **整张图的底图 +
        foothold**（约 2000×1000）⇒ 装它的 `BattleZoneDialog` 一显示就被撑到**上千高**，
        之后**任何一次布局重算**又会把它拽回去 ⇒ 用户看到的正是"**太大、想拖小还拖不动**"✗。

        ⚠ 这和 `gui/zone_editor._ZoneView.sizeHint` 是**同一个坑**（那边 2026-09-26 用户
        报过第 3 次"最小高度太大"✓，结论一字不差地记在它的 docstring 里 ✓）⇒ 口径照抄：
        **视图的语义是"给多大画多大"**（sizePolicy 已经是 Expanding ✓），sizeHint 只表个意，
        窗口尺寸交给布局与用户拖拽（能缩多小由 `setMinimumSize` 说话 ✓）。
        """
        return QSize(320, 200)

    def minimumSizeHint(self):
        """**跟 `sizeHint` 一个道理**（2026-09-28 补 ✓）：`QGraphicsView` 的默认
        `minimumSizeHint` 也会**跟着场景尺寸走**（场景 = 整张图 ≈ 2600×1400 ✗）⇒
        装它的窗口**最小宽度**就下不来 ⇒ 用户想拖窄都拖不动 ✗。

        ⚠ `setMinimumSize(300,160)` 虽然压得住它，但**显式覆写**才是根治 ✓
          （同一个坑在 `sizeHint` 上已经踩过一次 ✓ 见上面那段 ✓）。
        """
        return QSize(300, 160)

    def __init__(self, terrain, zones, set_name, current="", parent=None):
        super().__init__(parent)
        self.terrain = terrain
        self.zones = zones
        self.set_name = str(set_name or "")
        self._cur = str(current or "")
        #: ⭐ 要**呼吸高亮的另一个集合**名（用户 2026-10-02 ✓ —— "切换平台"模式下高亮
        #:   idle 去哪选中的集合）：和 `_cur`（选中的**一条**）不同，这个高亮**整个集合**
        #:   的所有 foothold（只进呼吸表、不改基准色 ✓）。`""` = 不高亮 ✓。
        self._highlight_set = ""
        self._items = {}                    # fid(str) -> QGraphicsLineItem（高亮要用 ✓）
        #: ⭐ **每一条线的画法**：`[(item, 基准色, 线型, 归不归呼吸), …]` —— 由 `_rebuild` 填 ✓
        #:   ⚠ **笔宽不在建的时候定死**（那样缩放一变就粗细不对 ✗）⇒ 统一在 `_apply_pens`
        #:   里按**当前缩放**算（口径抄 `zone_editor._apply_widths` ✓ 它挂在 `paintEvent` ✓）。
        self._lines = []
        #: 线宽（**屏幕像素**，跟着编辑器的设置走 ✓）—— 乘当前缩放才是世界单位 ✓
        self._lw = float(theme.load_foothold_width())
        self._pulse = 0.0                   # 呼吸相位（0~1 循环 ✓ 同 `zone_editor._pulse` ✓）
        self._fitted = False
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        # ⚠ 最小尺寸是**有意**定的（太矮/太窄就点不准了 ✓），但**别定大**（2026-09-28：
        #   原来是 380×230 ⇒ 加上子弹窗的表单，整个窗最小就 ~490 高，用户要"能缩小"✗）
        #   ⇒ 收到 300×160：照样能滚轮缩放/中键平移，窗口却缩得下去 ✓
        #   （分母是用户拖拽的**下限**，sizeHint 才是"它想多大"—— 那个见上面的 `sizeHint` ✓）
        self.setMinimumSize(300, 160)
        self.setToolTip("**点一条 foothold** ⇒ 它就是「idle 回归 foothold」✓"
                        "（再点空白 = 清空）。\n"
                        "滚轮缩放 / 中键拖 = 平移 / 双击 = 适应窗口 ✓。\n"
                        "⚠ 这里是**只读**的：改地形请去「寻路编辑器」✓。")
        self._rebuild()
        # 呼吸节拍：只在**有高亮项**时真干活（`_pulse_tick` 里早退 ✓）⇒ 常开也不费 ✓
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(PULSE_MS)
        self._pulse_timer.timeout.connect(self._pulse_tick)
        self._pulse_timer.start()

    @property
    def _hl(self):
        """**正在呼吸**的那几条 → `[(item, 基准色), …]`（只读派生 ✓）。

        ⚠ 保留它是因为**用例钉着这个名字**（`t_idle_foothold_picker_panel` ✓）—— 内部真源
          现在是 `_lines` 的第 4 项（要不要呼吸 ✓ 见 `_rebuild` / `_apply_pens` ✓）。
        """
        return [(it, base) for it, base, _st, hl in self._lines if hl]

    # ---------------- 读 ----------------

    def _fids(self):
        """本集合里的 foothold id（转调模块级纯函数 ✓ —— 那几件判据要能脱离 Qt 测 ✓）。"""
        return set_fids(self.zones, self.set_name)

    def footholds(self):
        """本集合的 foothold 对象（顺序 = 地形里的顺序 ✓）—— 只读 ✓。"""
        return footholds_of(self.terrain, self.zones, self.set_name)

    def current(self):
        """当前选中的 foothold id（`""` = 没选 ✓）。"""
        return self._cur

    def info(self, fid):
        """一条 foothold 的**参数**一行字（转调纯函数 ✓ 口径见它 ✓）。"""
        return fid_info(self.terrain, self.zones, self.set_name, fid)

    # ---------------- 选 ----------------

    def set_current(self, fid, focus=False):
        """外部设选中（`""` = 清空 ✓）；没变就不重画 ✓。

        `focus=True` ⇒ 顺带**居中放大到它**（给"下拉选了一条"用 ✓ 用户要求："选中之后
        可视图**聚焦**到该 foothold 并**呼吸高亮**"✓）。⚠ 图上点选**不**传它 ——
        那是"就近点中"，把人看的地方挪走反而突兀 ✗。
        """
        fid = str(fid or "")
        if fid == self._cur:
            if focus:
                self.focus_on(fid)
            return
        self._cur = fid
        self._rebuild()
        if focus:
            self.focus_on(fid)

    def highlight_set(self, set_name):
        """**呼吸高亮另一个集合**的所有 foothold（用户 2026-10-02 ✓ —— "切换平台"模式下
        高亮 idle 去哪选中的集合）。

        和 `set_current` 的区别：那个选**一条** foothold（改基准色为 `C_SEL` + 压顶 + 呼吸）；
        这个高亮**整个集合**的所有 foothold —— **只进呼吸表、不改基准色** ✓（"能选的"
        还是 `C_IN_SET` 绿，"要去的那块"只是呼吸一明一暗，颜色不换，两个语义不打架 ✓）。

        ⚠ `set_name=None / ""` = 清掉集合高亮 ✓（切回"回归 foothold"模式时调一次清掉 ✓）。
        """
        sn = str(set_name or "")
        if sn == self._highlight_set:
            return
        self._highlight_set = sn
        self._rebuild()

    # ---------------- 画 ----------------

    def _rebuild(self):
        """把场景重画一遍：底图 → **全部** foothold → 集合名 → 选中的那条高亮 ✓。

        ⭐⭐ **2026-09-28 改（用户要求）**："可视图希望就显示**寻路编辑器里配好的结果图**，
        **直接照搬**"✓ ⇒ 原来是**只画本集合那几条线**（"哪条能选"用得着），看不出"这张图
        长什么样" ✗；现在照 `gui/zone_editor.py::ZoneEditorDialog._rebuild_scene` 的口径画：

          · **底图**（`_add_background` ✓ 与那边同一套换算 ✓）；
          · **全部 foothold**：墙 = 橙虚线 / 地板 = 蓝实线（同 `C_WALL`/`C_FLOOR` 的观感 ✓）；
            ⭐ **本集合那几条**用**绿**（`#0b8043` = 那边的 `C_IN_SET` ✓）⇒ 一眼看出"能选的是哪些"
            ✓（这是本视图相对寻路编辑器**唯一**多出来的一层信息 ✓）；
          · **集合名标签**（画在集合包围盒上边中点 ✓ 同 `LABEL_PX` 的口径：字高按**场景单位**
            = 世界像素，跟着缩放走 ✓）；
          · **选中的那条**压最上面高亮 ✓ + **进呼吸表**（`_hl` ✓ 见 `_pulse_tick`）。

        ⚠ 线与字都用 **`width 0`（cosmetic）**：本视图只求"看得清、点得准"，
          **不抄**那边"线宽随缩放折算"整套（那要每次重建下发笔宽 ✓ 这里更省 ✓ 也不影响观感 ✓）。
        ⚠ **只读**：这里只画 + 选，不改地形 ✓。
        """
        self._scene.clear()
        self._items = {}
        self._lines = []                         # [(item, 基准色, 归不归呼吸), …]（见 `_apply_pens` ✓）
        self._lw = float(theme.load_foothold_width())   # ⭐ 线宽**跟着编辑器的设置走** ✓
        self._add_background()
        mine = set(self._fids())                 # 本集合的 foothold id（要单独着色 ✓）
        for f in (self.terrain.footholds or []):
            is_mine = str(f.fid) in mine
            # ⭐ **配色全用 `zone_editor` 的常量**（用户要的就是"跟编辑器一样" ✓）：
            #   墙 = `C_WALL` 灰虚线；**地板 = `C_FLOOR` 亮绿**；**本集合 = `C_IN_SET` 深绿**
            #   （那边 `C_IN_SET` 的语义是"当前高亮的集合" ✓ 这里语义正好对应"能选的" ✓）。
            if f.is_wall:
                base, style = C_WALL, Qt.DashLine
            elif is_mine:
                base, style = C_IN_SET, None
            else:
                base, style = C_FLOOR, None
            it = QGraphicsLineItem(float(f.x1), float(f.y1), float(f.x2), float(f.y2))
            it.setPen(self._pen(base, style=style))
            it.setZValue(2 if f.is_wall else 3)
            self._scene.addItem(it)
            self._items[str(f.fid)] = it
            self._lines.append((it, base, style, False))
        self._add_ladders()
        self._add_set_names()
        it = self._items.get(self._cur)
        if it is not None:
            it.setZValue(9)                     # 压在最上面（一眼看出选的是哪条 ✓）
            self._lines = [(i, (C_SEL if i is it else b), s_, (i is it))
                           for i, b, s_, _ in self._lines]
        # ⭐ **集合级呼吸高亮**（用户 2026-10-02 ✓）：把 `_highlight_set` 集合的所有 foothold
        #   也进呼吸表（**不改基准色** ✓ —— 和"选中的那条"区别开：那个改色压顶，这个只呼吸）。
        #   用途："切换平台"模式下让用户**一眼看到「idle 去哪」选的那块平台在地图哪里** ✓。
        if self._highlight_set:
            _hl_fids = set(set_fids(self.zones, self._highlight_set))
            _hl_items = set(self._items[fid] for fid in _hl_fids if fid in self._items)
            self._lines = [(i, b, s_, (hl or (i in _hl_items)))
                           for i, b, s_, hl in self._lines]
        r = self._scene.itemsBoundingRect()
        if r.width() > 1 and r.height() > 1:
            self._scene.setSceneRect(r.adjusted(-40, -40, 40, 40))

    def _add_ladders(self):
        """画**绳梯**（蓝点线 ✓ + 它的编号字 ✓）—— 口径全抄 `zone_editor` 那段 ✓。

        ⚠ **原来这里压根没画绳梯** ✗ —— 而"寻路编辑器里配好的结果图"上绳梯是很显眼的一层
          （用户一眼就发现少了 ✓）。配色/字号/字色都用那边的常量 ✓（`C_LADDER` / `LADDER_PX` /
          `C_LADDER_ID` ✓）—— ⚠ 编号**不是蓝字**：那边定过规矩"**蓝字专指已注册的集合名**"✓。
        """
        from PyQt5.QtGui import QBrush, QFont
        from PyQt5.QtWidgets import QGraphicsSimpleTextItem

        from core import zones as _zones

        try:
            lids = _zones.ladder_ids(self.terrain)
        except Exception:                       # noqa: BLE001
            return
        for L in (getattr(self.terrain, "ladders", None) or []):
            it = QGraphicsLineItem(float(L.x), float(L.y1), float(L.x), float(L.y2))
            pen = self._pen(C_LADDER)
            pen.setStyle(Qt.DotLine)            # ⚠ 绳梯在那边是**点线** ✓
            it.setPen(pen)
            it.setZValue(6)                     # 压在 foothold 上面（它是"通道"✓）
            self._scene.addItem(it)
            self._lines.append((it, C_LADDER, Qt.DotLine, False))
            txt = str(lids.get(id(L)) or "")
            if not txt:
                continue
            t = QGraphicsSimpleTextItem(txt)
            fnt = QFont()
            fnt.setPixelSize(LADDER_PX)         # ⭐ 字号照搬（那边 38 ✓）
            t.setFont(fnt)
            t.setBrush(QBrush(QColor(C_LADDER_ID)))     # 近白（**不是**蓝 ✓ 见那边的规矩 ✓）
            t.setPos(float(L.x) + LADDER_PX * 0.35, min(float(L.y1), float(L.y2)))
            t.setZValue(7)
            self._scene.addItem(t)

    def _add_set_names(self):
        """画**集合名**（每块集合一行字，落在它包围盒上边中点 ✓ 口径抄 `zone_editor` ✓）。

        ⚠ 字高用**场景单位**（= 世界像素 ✓）：本视图的 scene 就是世界坐标（见 `_add_background`
          ✓）⇒ 跟着缩放变粗变细，和寻路编辑器里**同一种观感** ✓（`.setFlag(ItemIgnoresTransformations)`
          ✗ 别加 —— 加了字就不跟缩放了，反而不像那张图 ✓）。
        ⚠ 一个集合的包围盒由它的 foothold 拼出来（**只算非墙的** —— 墙是竖线，算进去会把盒子
          拉得很怪 ✓）；一条 foothold 都没有的集合跳过 ✓（fid 指向的空集合很常见 ✓）。
        """
        from PyQt5.QtGui import QBrush, QFont
        from PyQt5.QtWidgets import QGraphicsSimpleTextItem

        for name, s in (self.zones.sets or {}).items():
            xs, ys = [], []
            fids = set(str(x) for x in ((s or {}).get("footholds") or []))
            for f in (self.terrain.footholds or []):
                if str(f.fid) not in fids or f.is_wall:
                    continue
                xs += [float(f.x1), float(f.x2)]
                ys += [float(f.y1), float(f.y2)]
            if not xs:
                continue
            t = QGraphicsSimpleTextItem(str(name))
            fnt = QFont()
            fnt.setPixelSize(LABEL_PX)           # ⭐ 字号照搬（那边 46 ✓ 原来我写 42 ✗）
            t.setFont(fnt)
            # ⭐ **集合名一律蓝字**（`C_SETNAME` ✓ 那边的规矩：蓝字专指"已注册的集合名" ✓
            #   —— 我原来用近白 ✗ 跟编辑器对不上 ✓）。
            t.setBrush(QBrush(QColor(C_SETNAME)))
            # 字宽用场景单位量不出来（字体是像素单位）⇒ 按字数估个半宽把它居中 ✓
            t.setPos((min(xs) + max(xs)) / 2.0 - LABEL_PX * 0.55 * len(str(name)),
                     min(ys) - LABEL_PX * 1.6)
            t.setZValue(4)
            self._scene.addItem(t)

    # ---------------- 画（笔宽随缩放 ✓） ----------------

    def _px(self):
        """当前缩放（1 个**世界像素** = 几个**屏幕像素** ✓ 口径抄 `zone_editor._px` ✓）。"""
        try:
            return max(1e-6, float(self.transform().m11()))
        except Exception:                       # noqa: BLE001
            return 1.0

    def _pen(self, color, extra=0.0, style=None):
        """按**当前缩放**折出这枝笔（口径抄 `zone_editor._pen` ✓）。

        ⚠⚠ **原来这里是 `setWidth(0)`（cosmetic）** ✗ —— 那会让线在**任何缩放**下都一个粗
           ⇒ 用户一眼就看出"跟编辑器不一样" ✗（他的判据原话："字体线段的配色、**粗细**等"✓）。
           ⇒ 现在 = `(它设的线宽 + extra) × 当前缩放` ✓（extra 只给呼吸用 ✓）。
        """
        pen = QPen(QColor(color))
        pen.setWidthF(max(0.0, (self._lw + float(extra)) * self._px()))
        pen.setCapStyle(Qt.RoundCap)
        if style is not None:
            pen.setStyle(style)
        return pen

    def _breath_amount(self):
        """呼吸的明暗系数 0~1（口径抄 `zone_editor._on_pulse` 的公式 ✓）。"""
        return 0.5 - 0.5 * math.cos(self._pulse * 2.0 * math.pi)

    def _apply_pens(self):
        """把 `_lines` 里每条线的笔**按当前缩放 + 当前呼吸相位**重算一遍 ✓。

        ⚠ 为什么每次都重算（而不是"建的时候定死 + 呼吸时改"）：**缩放会变**（滚轮 / 聚焦 ✓）
          ⇒ 笔宽必须跟着变；而呼吸又要**加粗**（`extra` ✓）⇒ 两者叠在一起算才不打架 ✓
          —— 所以**只有这一处算笔**、它挂在 `paintEvent` 上（同 `zone_editor` 的做法 ✓）。
        """
        a = self._breath_amount()
        for it, base, style, hl in self._lines:
            extra = 3.0 * a if hl else 0.0
            color = QColor(base)
            if hl:
                # 呼吸 = 改 HSV 的 V（那边的口径 ✓），最多加粗 3 屏幕像素 ✓
                color.setHsv(color.hue(), color.saturation(), int(150 + 105 * a))
            it.setPen(self._pen(color, extra=extra, style=style))

    def paintEvent(self, ev):
        """每帧先把笔按当前缩放/呼吸相位刷新一遍，再交给基类画 ✓。

        ⚠ 挂在这里**最省心**：滚轮缩放、聚焦放大、窗口尺寸变 —— 任何一个都会触发重绘，
          笔宽就自动跟上了（不用在每个改缩放的地方都记得调一次 ✓ 同 `zone_editor` ✓）。
        """
        try:
            self._apply_pens()
        except Exception:                       # noqa: BLE001 —— 画笔坏了也别把绘制整个搞崩 ✗
            pass
        super().paintEvent(ev)

    # ---------------- 呼吸（高亮那条一明一暗 ✓） ----------------

    def _pulse_tick(self):
        """呼吸节拍：推一下相位就够 ✓（**笔在 `paintEvent` 里算** ✓ 见 `_apply_pens`）。

        ⚠ 没有高亮项时**什么都不做**（省 CPU ✓ 同那边那句早退 ✓）。
        """
        if not any(hl for _i, _b, _s, hl in self._lines):
            return
        self._pulse = (self._pulse + PULSE_STEP) % 1.0
        self.viewport().update()

    def focus_on(self, fid):
        """**居中到某一条 foothold**（并适当放大到看得清 ✓）—— 给"下拉选中"用 ✓。

        ⚠ 缩放**只往大调**（已经比 `FOCUS_SCALE` 看得更大时不缩回去 ✗ —— 用户刚手动放大
          想看细节，一下拉就被缩小会很难受 ✓）。
        """
        fid = str(fid or "")
        r = None
        for f in (self.terrain.footholds or []):
            if str(f.fid) == fid:
                x1, x2 = float(f.x1), float(f.x2)
                y1, y2 = float(f.y1), float(f.y2)
                r = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                break
        if r is None:
            return
        cx, cy = (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
        try:
            k = float(self.transform().m11())
        except Exception:                       # noqa: BLE001
            k = 0.0
        self.centerOn(cx, cy)
        if k < FOCUS_SCALE:
            self.scale(FOCUS_SCALE / max(1e-6, k), FOCUS_SCALE / max(1e-6, k))
            self.centerOn(cx, cy)               # 缩完再居中一次（缩放会挪视口 ✓）
        self.viewport().update()

    def _add_background(self):
        """把**小地图底图**铺在几何下面（换算同 `zone_editor._add_background` ✓）。

            图像像素 = (世界坐标 + centerX/Y) / px_per_world
        ⇒ 图 (0,0) 对应世界 `(-centerX, -centerY)`，一个图像像素 = `px_per_world` 个世界像素 ✓
        ⇒ 所以"位置 = (-centerX,-centerY) + 缩放 = px_per_world" ✓（不自己另算一套 ✗）。

        ⚠ 底图**直接取 `terrain.canvas`**（`mapdata.load(..., with_canvas=True)` 已经读好了 ✓）
          —— 本模块**不自己读文件** ✓（中文路径 / alpha 那套坑留给 `mapdata` 一处 ✓）。
        ⚠ 透明区：`canvas` 可能是 3 通道（BGR）+ 单独一份 `canvas_alpha` ✓ 也可能直接 4 通道
          （BGRA ✓）⇒ 两种都认 ✓（见 `core/mapdata` 里 `canvas` / `canvas_alpha` 的说明 ✓）。
        """
        cv = getattr(self.terrain, "canvas", None)
        if cv is None or getattr(cv, "size", 0) == 0:
            return
        try:
            import numpy as np
            a = getattr(self.terrain, "canvas_alpha", None)
            if (getattr(cv, "ndim", 0) == 3 and cv.shape[2] == 3
                    and getattr(a, "ndim", 0) == 2 and a.shape[:2] == cv.shape[:2]):
                cv = np.dstack([cv, a])
            h, w = cv.shape[:2]
            fmt = (QImage.Format_ARGB32
                   if (getattr(cv, "ndim", 0) == 3 and cv.shape[2] == 4)
                   else QImage.Format_BGR888)
            buf = np.ascontiguousarray(cv)
            qimg = QImage(buf.data, w, h, buf.strides[0], fmt)
            pm = QPixmap.fromImage(qimg.copy())     # ⚠ copy()：`buf` 是临时对象 ✗
        except Exception:                   # noqa: BLE001 —— 底图坏了就只画线（不影响点选 ✓）
            return
        it = QGraphicsPixmapItem(pm)
        it.setPos(-float((self.terrain.mini or {}).get("centerX") or 0),
                  -float((self.terrain.mini or {}).get("centerY") or 0))
        try:
            it.setScale(float(self.terrain.px_per_world))
        except Exception:                   # noqa: BLE001
            pass
        it.setZValue(0)
        self._scene.addItem(it)

    # ---------------- 交互 ----------------

    def showEvent(self, ev):
        """第一次露头 ⇒ 适应窗口（此时才有真实视口尺寸 ✓）。"""
        super().showEvent(ev)
        # ⚠⚠ **视口还没有真实尺寸时不许 fit**（2026-09-28 ✓）：`fitInView` 拿到 0×0 的矩形
        #   会算出非法缩放，Qt 内部随后越界 ⇒ **进程原生崩溃** ✗ —— 这正是本视图**在离屏
        #   自检里建不出来**的原因（`tools/selftest_zone_editor` 里那条用例因此被停用 ✓）。
        #   ⇒ 视口是 0 就先不 fit，留给下一次 showEvent / 首次 paint 之后（`_fitted` **不置位**
        #      ⇒ 下次露头还会再试 ✓）。
        if not self._fitted:
            vp = self.viewport()
            if vp is None or vp.width() < 2 or vp.height() < 2:
                return
            self._fitted = True
            self.fit()

    def pick_at_scene(self, x, y, tol=None):
        """场景坐标 `(x, y)` 附近最近的那条 foothold ⇒ **选中它**并返回 fid（`""` = 没命中 ⇒ 清空 ✓）。

        ⚠ 抽成**公共方法**有两个好处：① `mousePressEvent` 调它 ✓；② **用例可以直接喂世界坐标**
          测命中 —— 离屏窗口尺寸是 0，`mapToScene()` 算出来的东西**根本不可信** ✗
          （造一个假的 `QMouseEvent` 也测不出真行为 ✓）。
        ⚠ 场景坐标**就是世界坐标**（底图按世界坐标摆的 ✓ 见 `_add_background`）✓。

        `tol` 不给 ⇒ 按**屏幕像素**折算（`PICK_SCREEN_PX / 当前缩放` ✓ 放大时自动变小 ✓
        —— 同 `zone_editor.pick_at` 的口径：那里由调用方按缩放换算 ✓）。
        """
        if tol is None:
            try:
                k = max(1e-6, float(self.transform().m11()))
            except Exception:               # noqa: BLE001
                k = 1.0
            tol = PICK_SCREEN_PX / k
        self.set_current(pick_fid(self.terrain, self.zones, self.set_name, x, y, tol))
        self.picked.emit(self._cur)
        return self._cur

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            sp = self.mapToScene(e.pos())
            self.pick_at_scene(float(sp.x()), float(sp.y()))
            e.accept()
            return
        super().mousePressEvent(e)


# ══════════════════════════════════════════
# ⭐ 带**下拉**的包装（用户 2026-09-28 要求 ③ ✓）
# ══════════════════════════════════════════
# 用户原话："这个 idle 回归 foothold 配置用**下拉裂变**选择吧，选中之后可视图**聚焦**到该
# foothold 并**呼吸高亮**（跟寻路编辑器里的模式一样）"✓。
#
# ⚠ **为什么是独立 widget 而不是给视图加控件**：`FootholdPicker` 是 `QGraphicsView`，
#   往里塞 `QComboBox` 得走 `QGraphicsProxyWidget`（重、还要管坐标系 ✗）⇒ 一个
#   `QWidget`（下拉在上、视图在下）最省 ✓。
# ⚠ **对外接口与 `FootholdPicker` 完全一致**（`picked` / `current()` / `info()` /
#   `set_current()` ✓）⇒ `gui/player_panel.py::BattleZoneDialog` **一行都不用改** ✓
#   （它只依赖这四个 ✓ 见那边的 `_make_picker` / `_refresh_idle_label` / `idle_fid` ✓）。
class FootholdPickerPanel(QWidget):
    """**只读视图 + 下拉**：下拉选一条 ⇒ 视图聚焦并呼吸高亮 ✓；图上点选 ⇒ 下拉同步 ✓。

    ⚠ **图在上、下拉在下**（用户 2026-09-28 明确："可视图放在最上面"✓）。
    ⚠ 面板自己有**最小高度**：太矮的话下拉一展开就超出窗口、什么都看不到 ✗
      （用户报过："下拉列表展开啥也看不到"✓）—— 落地时算上"下拉 + 那行说明 + 清空按钮"
      要的地方，别只按视图算 ✓。
    """

    picked = pyqtSignal(str)
    #: 面板的最小高度（px）：够放下"视图 + 下拉 + 说明行" ⇒ 下拉展开才有地方 ✓
    MIN_H = 260

    def __init__(self, terrain, zones, set_name, current="", parent=None):
        super().__init__(parent)
        self._view = FootholdPicker(terrain, zones, set_name, current=current)
        self.setMinimumHeight(self.MIN_H)
        # ⚠ 视图自己也会发 `picked`（图上点选 ✓）⇒ 转出去 + 把下拉同步上 ✓
        self._view.picked.connect(self._on_view_picked)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        # ⭐ **图在上、下拉在它下面**（用户 2026-09-28："可视图放在最上面吧"✓）——
        #   原来是"下拉在上" ✗：图被压在下面、又跟表单挤在一起 ⇒ 又窄又矮，下拉一展开
        #   还看不到几项 ✗（用户原话："现在下拉列表展开啥也看不到，图又窄"✓）。
        root.addWidget(self._view, 1)        # 视图吃满剩余高度（它是主角 ✓）
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        # ⚠ 这块从 `QFormLayout` 里**拿出来**了（占满整宽 ✓）⇒ 表单那行"idle 回归 foothold"
        #   的标签也就没了 ⇒ 在这儿补上（否则用户看不出这一大块是干什么的 ✗）。
        row.addWidget(QLabel("idle 回归 foothold"))
        self.cmb = NoWheelComboBox()
        self.cmb.setMinimumWidth(150)
        # ⚠⚠ **别让下拉按"最长的那一项"决定宽度**（2026-09-28 实测踩到 ✗）：项文本里带着
        #   `x 123..456　长 333　y=-208` 这种长句 ⇒ `QComboBox` 默认按最长项算 `sizeHint`
        #   ⇒ **整个子弹窗被顶到 860+ 宽**（用户会说"窗口太宽"，之前已经修过一次同类问题 ✓）。
        #   ⇒ 宽度改由 `minimumContentsLength` 说话（够看 `#14 · 地板 · y=-208` 就行 ✓），
        #     完整参数在**下面那行说明**里（它会自动换行 ✓）。
        self.cmb.setSizeAdjustPolicy(
            NoWheelComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb.setMinimumContentsLength(22)
        self.cmb.setToolTip(
            "**本集合**里的 foothold（在「寻路编辑器」里圈进这块集合的那些 ✓）。\n"
            "选一条 ⇒ 下面的图**聚焦**到它、并**呼吸高亮** ✓。\n"
            "⚠ 只列**本集合**的 —— 点到别的集合的线不该改 idle 回归点 ✗。")
        self._fill_combo(current)
        self.cmb.currentIndexChanged.connect(self._on_combo)
        row.addWidget(self.cmb, 1)
        root.addLayout(row)                  # 下拉在视图**下面** ✓

    # ---------------- 内部 ----------------

    def _fill_combo(self, current):
        """候选 = 本集合的 foothold（`#id 类型 面y` ✓）；**空的一项**在最前 = 不设 ✓。"""
        self.cmb.blockSignals(True)
        try:
            self.cmb.clear()
            self.cmb.addItem("（不设 —— 不跨层水平走）", "")
            for f in self._view.footholds():
                # ⚠ 项文本**要短**（宽度由它算 ✓ 见 `setMinimumContentsLength` 那段）；
                #   完整参数（x 范围 / 长 / 面 y）在下面那行说明里，选完就能看到 ✓。
                self.cmb.addItem(self._short_label(f), str(f.fid))
            j = self.cmb.findData(str(current or ""))
            self.cmb.setCurrentIndex(j if j >= 0 else 0)
        finally:
            self.cmb.blockSignals(False)
        if not self._view.footholds():
            self.cmb.addItem("（这块集合里还没有 foothold）", "")
            self.cmb.setCurrentIndex(0)
            self.cmb.setEnabled(False)

    @staticmethod
    def _short_label(f):
        """下拉里那一行的**短标签**：`#14 · 地板 · y=-208`（够认人，又不撑宽 ✓）。"""
        try:
            mid = (float(f.x1) + float(f.x2)) / 2.0
            yy = float(f.y_at(mid))
        except Exception:                       # noqa: BLE001
            yy = float(getattr(f, "y1", 0.0) or 0.0)
        return "#%s · %s · y=%d" % (f.fid, "墙" if f.is_wall else "地板", round(yy))

    def _on_combo(self, _i):
        """下拉选了一条 ⇒ 设选中 + **聚焦**（`focus=True` ✓）⇒ 视图重画并呼吸 ✓。"""
        fid = str(self.cmb.currentData() or "")
        self._view.set_current(fid, focus=bool(fid))
        self.picked.emit(self._view.current())

    def _on_view_picked(self, fid):
        """图上点选 ⇒ **把下拉也切过去**（两边永远一致 ✓，同寻路编辑器的手感 ✓）。"""
        fid = str(fid or "")
        j = self.cmb.findData(fid)
        if j >= 0 and j != self.cmb.currentIndex():
            self.cmb.blockSignals(True)
            try:
                self.cmb.setCurrentIndex(j)
            finally:
                self.cmb.blockSignals(False)
        self.picked.emit(fid)

    # ---------------- 对外（与 `FootholdPicker` 同接口 ✓） ----------------

    def current(self):
        return self._view.current()

    def info(self, fid):
        return self._view.info(fid)

    def set_current(self, fid, focus=False):
        fid = str(fid or "")
        j = self.cmb.findData(fid)
        if j >= 0:
            self.cmb.blockSignals(True)
            try:
                self.cmb.setCurrentIndex(j)
            finally:
                self.cmb.blockSignals(False)
        self._view.set_current(fid, focus=focus)

    def highlight_set(self, set_name):
        """**呼吸高亮另一个集合**（转发给内部视图 ✓ —— 与 `FootholdPicker` 同接口）。

        ⭐ 用户 2026-10-02 ✓ —— "切换平台"模式下高亮 idle 去哪选中的集合：下拉是 foothold
        列表（不是集合列表）⇒ 这里**不同步下拉**，只把视图的集合高亮转过去 ✓。
        """
        self._view.highlight_set(set_name)
