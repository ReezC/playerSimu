"""foothold 集合编辑器（V1）：把地形画出来，人工点选/框选 → 注册成命名集合。

**为什么必须有它**（见 `docs/寻路设计.md` §12）：自动串段会把"要跳/攀才能互通"的
foothold 并成同一段 —— 实测 `105090600` 的第 0 段把一面 **388 像素高**的悬崖当成了
平台边缘（43 条 foothold、其中 12 条长竖条）。所以"哪几条属于 A 平台"这件事
**只能由人圈**，判定一律用「脚下的 foothold id ∈ 集合」。

**V1 范围**：几何渲染 + 点选/框选 + 集合注册/改名/删除 + 存盘与校验 + 撤销/重做。
（边 walk/climb/drop、路径点、绳梯/传送门自动归属、可走建议 = V2，见 §12.3。）

**交互**（与仓库既有的看图操作一致，见 `gui/canvas.ZoomPanView`）：

    滚轮 = 缩放        中键拖 = 平移        双击 = 适应窗口
    左键点 = 选中（Shift/Ctrl = 加选，Alt = 减选）
    左键拖 = 框选       **左→右 = 只选完全包含；右→左 = 相交即选**
                        （AutoCAD 的 window / crossing 那套，用熟了很省事）
    Ctrl+A 全选        Esc 清空            Ctrl+Z / Ctrl+Y 撤销 / 重做

**哪些能选**：只有**可站立的** foothold（横向/斜坡）能被选进集合 —— 竖向的墙
（`x1==x2`）画出来只作参照，`Terrain.foothold_below()` 本来也跳过它们，
把墙圈进集合没有任何意义（玩家永远站不上去）。
"""

import json
import math
from pathlib import Path

from html import escape

from PyQt5.QtCore import (QLineF, QPointF, QRectF, QSize, Qt, QTimer,
                          pyqtSignal)
from PyQt5.QtGui import (QAbstractTextDocumentLayout, QBrush, QColor, QFont,
                         QIcon, QImage, QPainter, QPainterPath,
                         QPainterPathStroker, QPen, QPixmap, QPolygonF,
                         QTextDocument, QTextOption)
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                             QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QFrame,
                             QGraphicsLineItem, QGraphicsPixmapItem,
                             QGraphicsScene, QGraphicsSimpleTextItem,
                             QGridLayout, QGroupBox, QHBoxLayout, QInputDialog,
                             QLabel, QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QScrollArea, QSizePolicy, QSplitter,
                             QStyle, QStyledItemDelegate, QStyleOptionViewItem,
                             QVBoxLayout, QWidget)

from core import mapdata, zones
from gui import theme
from gui.canvas import ZoomPanView
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSlider,
                         scroll_area)

#: 点选的拾取半径（**屏幕像素 ÷ 当前缩放**，见 _ZoneView）：线段很细，
#: 要求"正好点在线上"是不可能的 —— 矢量编辑器都带这个容差。
PICK_PX = 6.0
#: 框选时判定"包含"用的宽松量（世界像素）：foothold 是线，严格包含几乎选不中
BOX_PAD = 2.0

#: 绘制配色（内联样式是本项目的既定风格，见 docs/UI规范.md §7 的说明）
#: **配色是照着"压在底图上"挑的**：底图是深蓝底 + 蓝色平台（小地图那种），
#: 原先的暗绿（#2f6f4f）压上去几乎看不见 ✗ —— 实测渲染出来才发现，所以改亮绿。
C_FLOOR = QColor("#8ce99a")        # 可站立（横向/斜坡）
C_WALL = QColor("#c8ccd0")         # 竖向的墙：只作参照，不可选
C_LADDER = QColor("#1a73e8")       # 绳梯
C_PORTAL = QColor("#8e24aa")       # 传送点
C_LIFE = QColor("#c5221f")         # 刷怪点
C_SEL = QColor("#ffd54f")          # 选中
C_IN_SET = QColor("#0b8043")       # 属于当前高亮的集合

#: 传送点圆点的半径（世界像素）×2 种尺寸 + 名字的字号（用户 2026-10-06 ✓ 原话两条：
#:   "**在 foothold 集合编辑器里，传送点的名称没有标出来**" ＋
#:   "**选中传送点后，要放大传送点的图形呼吸高亮，不然看不出来**"）。
#: ⚠ 为什么"选中要**画大**"而不是只换颜色：门的图形本来就小（半径 6 世界像素 ✓ 图上几像素
#:   ✗），只换颜色在小缩放下**照样看不出来** —— 用户的原话就是"不然看不出来"✓ ⇒ 半径翻近一倍
#:   ＋ 换选中色（`C_SEL` 同"选中的 foothold / 绳梯"✓）＋ 一起呼吸 ✓ 三样一起才够醒目 ✓。
PORTAL_R = 6.0
PORTAL_SEL_R = 11.0
#: 门名文字的字号（世界像素；绳梯编号那一套 `LADDER_PX` 的同学 ✓ —— 门名比绳号长，
#: 稍微小一点免得糊住旁边的线 ✓）。
PORTAL_PX = 13

#: 呼吸高亮的节拍（毫秒）与相位步长。**为什么用 QTimer 而不是 QPropertyAnimation**：
#: 高亮的是一批 QGraphicsLineItem（不是单个控件），按节拍统一改笔刷最直接，
#: 而且停止逻辑只有一处（closeEvent）。
PULSE_MS = 60
PULSE_STEP = 0.14

#: 底图透明度（0~100，百分比）的默认值与取值范围。
#: 为什么要有这个滑条：底图是深蓝的示意图，而线是亮绿 —— 每张图、每个显示器
#: 看着都不太一样，固定一个值必然有人嫌"底图太抢眼/太黑看不出地形"。实测把
#: 0.6 写死时就卡在这个取舍上，所以交给用的人。
BG_OPACITY_DEF = 60

#: 集合名在界面上的**统一颜色**。为什么统一成蓝的：界面上本来就有一堆颜色
#: （画布按类型着色、悬空边标红、集合各自的配色），名字再跟着花就更难认了 ——
#: "这个字段是一个注册过的集合"这件事只该有一种视觉。悬空边那种**错误**仍然标红。
C_SETNAME = "#1a73e8"

#: 集合之间那条**连线**的颜色 + 线型（2026-09-26 要求 3）：黑色细虚线。
#: 为什么不再按类型上色（原来 走=绿实线 / 爬=蓝虚线 / 跳=橙虚线 / 门=紫）：
#: 同一片画面上已经有亮绿地形线、深蓝底图、蓝色集合名、黄色呼吸高亮 —— 箭头再各挑
#: 一种颜色，就没法一眼认出"这是箭头"。**类型不丢**：箭头那个三角仍然按类型着色，
#: 列表里也写着 [走]/[爬]…，所以线只负责"有没有关系"，颜色交给箭头。
C_ARROW = "#202124"

#: 集合名的字高（**场景单位 = 世界像素**）。场景就是世界坐标系（约 2630×1400），
#: 所以这个数在整图 fit（≈0.27 倍）下约合 12 屏幕像素，放大时跟着变粗 ——
#: 和 `tools/map_terrain_view` 烘进图里的段号是同一种观感（它就是跟着图缩放的）。
LABEL_PX = 46

#: 绳梯编号（`L1`/`L2`…）的字高，比集合名小一号 —— 它是**参照信息**（"这条边用的是
#: 哪根绳"），不该和集合名抢注意力。编号来自 `core.zones.ladder_ids`，与边里写的
#: 绳号、日志、将来的执行器**同一口径**（都是按 (x, page) 排序编号）。
LADDER_PX = 38

#: 列表行里"分色"用的角色：`DisplayRole` 留**纯文本**（tooltip、自检、`text()` 都从
#: 它取），富文本放这里给 `_RichRowDelegate` 画。见那条类注释里的理由。
ROLE_HTML = Qt.UserRole + 2

def _natural(s):
    """自然序的排序键：`L10` 要排在 `L2` **后面**（按数字段比，不是按字符串比）。

    只给下拉/列表排序用（绳号 `L1..L10`、门 `west00/west01` 这种）。
    """
    out, num = [], ""
    for ch in str(s):
        if ch.isdigit():
            num += ch
            continue
        if num:
            out.append((0, int(num), ""))
            num = ""
        out.append((1, 0, ch))
    if num:
        out.append((0, int(num), ""))
    return out


#: 列表自动排序时**通行方式**的先后（2026-09-26 要求"自动 sort"）：走 → 爬 → 跳 → 下跳
#: → 门 → 待确认。定死在这里，别用 `zones.EDGE_KINDS` 的顺序 —— 那个是**数据层**的顺序
#: （`drop` 在前是为了历史兼容），改它会影响别处。
_KIND_ORDER = {k: i for i, k in enumerate(zones.EDGE_LABELS)}


def edge_row_key(e, focus):
    """「可到达 / 可被到达」两栏的排序键。

    为什么按**另一端**的集合名排：这两栏回答的是"能去哪儿 / 谁要来"，找的是**那个集合**；
    而文件里的顺序是"谁先被加进来"，编辑一次就乱一次（用户 2026-09-26 报的问题）——
    按它排等于每次都要从头扫一遍。后面依次比类型、绳号、门：同一个终点有多条时，
    同类的挨在一起、顺序也稳定（两次打开的排列不会跳）。
    中文按**码点**排（不是拼音）：只求稳定可预期，不求像字典。
    """
    other = e.get("from") if e.get("to") == focus else e.get("to")
    # ⭐ `portal`（走哪个门）2026-09-27 随「传送门」移除过 ✓、**2026-10-06 加回** ✓ ——
    #   它和 `ladder` 是同一回事：同一对集合、同类型、**不同门/不同绳**是几条**不同的边** ✓，
    #   排序键里少了它 ⇒ 两条边的相对次序由"谁先被加进来"决定 ⇒ 编辑一次就跳一次 ✗。
    return (str(other or ""), _KIND_ORDER.get(e.get("kind") or "?", 99),
            str(e.get("ladder") or ""), str(e.get("portal") or ""))


#: 列表行里"非名字"部分的文字颜色：交给调色板（**不是**写死的黑）——
#: 选中那一行时底色变深，字得跟着反色才读得清，调色板会自己处理这件事。
#: 需要写死的只有"集合名"（蓝）和"悬空"（红）这两种**含义**色。

#: 绳梯编号的**字色**：中性浅色，**不是蓝**。
#: 2026-09-26 用户定的一条规矩：蓝字在这个窗口里专指"**已注册的平台集合名**"
#: （见 C_SETNAME）。编号不是集合名，用蓝就会误导（第一版就是蓝的 ✗）。
#: 注：绳梯那根**线**仍是蓝的点线（C_LADDER）—— 规矩管的是**字**。
C_LADDER_ID = "#f1f3f4"

#: 同一对集合里**后续行**的缩进（两个全角空格 —— 和组头 `名字 → ` 后面那个同宽 ✓）。
#: 为什么用全角：这一栏是中文界面，半角空格在比例字体里几乎看不见（等于没缩进 ✗）。
EDGE_INDENT = "　　"


# ══════════════════════════════════════════
# 纯几何（不碰 Qt，自检直接覆盖）
# ══════════════════════════════════════════

def dist_point_seg(px, py, x1, y1, x2, y2):
    """点到线段的距离（世界像素）。"""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = x1 + t * dx, y1 + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def dist_to(f, x, y):
    """点到某条 foothold 的距离。"""
    return dist_point_seg(x, y, f.x1, f.y1, f.x2, f.y2)


def pick_at(terrain, x, y, tol):
    """(x, y) 附近最近的可站立 foothold → Foothold / None。

    `tol` 由调用方按当前缩放换算（放大时 tol 小、缩小时 tol 大）——
    这样"看着点中了"和"真的点中了"总是一致。
    """
    best, best_d = None, None
    for f in terrain.footholds:
        if f.is_wall:
            continue
        d = dist_to(f, x, y)
        if d <= tol and (best_d is None or d < best_d):
            best, best_d = f, d
    return best


def dist_to_ladder(L, x, y):
    """点到某根绳梯的距离（绳梯是**竖直线段** `(L.x, L.y1)…(L.x, L.y2)` ✓）。"""
    return dist_point_seg(x, y, float(L.x), float(L.y1), float(L.x), float(L.y2))


def pick_ladder(terrain, x, y, tol):
    """(x, y) 附近最近的**绳梯** → Ladder / None（口径与 `pick_at` **同一把尺** ✓）。

    用户 2026-09-27 要求："foothold 编辑器希望能**点选绳梯**（为了快速查看它的信息），
    **不用接逻辑**" ⇒ 它只用来"**看**" ✓：不进选择集、不写文件、不进撤销栈 ✗。
    `tol` 与 `pick_at` 同源 —— 由调用方按当前缩放换算（`PICK_PX` ✓），
    保证"看着点中了 = 真的点中了" ✓（细线本来点不中，容差是功能不是手感 ✓）。
    """
    best, best_d = None, None
    for L in (getattr(terrain, "ladders", None) or []):
        d = dist_to_ladder(L, x, y)
        if d <= tol and (best_d is None or d < best_d):
            best, best_d = L, d
    return best


def dist_to_portal(p, x, y):
    """点到某个传送点的距离（门是个**点** ✓ 所以就是两点距离 ✓）。"""
    return ((x - float(p.x)) ** 2 + (y - float(p.y)) ** 2) ** 0.5


def pick_portal(terrain, x, y, tol):
    """(x, y) 附近最近的**传送点** → Portal / None（口径与 `pick_at` / `pick_ladder` 同一把尺 ✓）。

    用户 2026-10-06 要求："**选中传送点后，要放大传送点的图形呼吸高亮，不然看不出来**"
    ⇒ 先得**点得中**它 ✓：门是个小圆点（半径 6 世界像素 ✓），缩小时更小 ⇒ 和绳梯同理，
    `tol` 由调用方按当前缩放换算（`PICK_PX` ✓）保证"看着点中了 = 真的点中了" ✓。
    ⚠ 与绳梯**同一套语义**：只用来"**看**"（状态行 + 画布高亮 ✓），
      不进选择集、不写文件、不进撤销栈 ✗。
    """
    best, best_d = None, None
    for p in (getattr(terrain, "portals", None) or []):
        d = dist_to_portal(p, x, y)
        if d <= tol and (best_d is None or d < best_d):
            best, best_d = p, d
    return best


def _seg_in_rect(f, rect, mode):
    """线段与矩形的关系 → 是否命中。mode: "contains" / "crossing"。"""
    x0, y0, x1, y1 = rect
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    fx0, fx1 = min(f.x1, f.x2), max(f.x1, f.x2)
    fy0, fy1 = min(f.y1, f.y2), max(f.y1, f.y2)
    if mode == "contains":
        # 线段**整个**落在框里（端点带一点宽松量：贴边也该算中）
        return (lo_x - BOX_PAD <= fx0 and fx1 <= hi_x + BOX_PAD
                and lo_y - BOX_PAD <= fy0 and fy1 <= hi_y + BOX_PAD)
    # crossing：包围盒相交 + 真的碰到（水平/竖直/一般线段都用采样判定，够用且稳）
    if fx1 < lo_x or fx0 > hi_x or fy1 < lo_y or fy0 > hi_y:
        return False
    n = 24
    for i in range(n + 1):
        t = i / float(n)
        px = f.x1 + t * (f.x2 - f.x1)
        py = f.y1 + t * (f.y2 - f.y1)
        if lo_x <= px <= hi_x and lo_y <= py <= hi_y:
            return True
    return False


def pick_in_rect(terrain, rect, mode):
    """框选命中的可站立 foothold → [Foothold…]（`mode` 见 `_seg_in_rect`）。"""
    return [f for f in terrain.footholds
            if not f.is_wall and _seg_in_rect(f, rect, mode)]


def ids_bbox(terrain, ids):
    """一组 foothold id 的世界包围盒 → (x, y, w, h) / None（"缩放到集合"用）。

    实现在 `core/zones.set_span`（**一份换算**：归属判定、聚焦、将来的边都取同一处）。
    """
    sp = zones.set_span(terrain, ids)
    if sp is None:
        return None
    x0, x1, y0, y1 = sp
    return (x0, y0, x1 - x0, y1 - y0)


def new_selection(cur, ids, mode):
    """选择集运算（**纯函数**，交互与自检共用一份）→ 新的 id 集合。

    mode: `replace`（普通点击/框选）/ `add`（Shift/Ctrl）/ `sub`（Alt）
    """
    cur = set(cur)
    ids = set(str(i) for i in ids)
    if mode == "replace":
        return ids
    if mode == "add":
        return cur | ids
    if mode == "sub":
        return cur - ids
    raise ValueError("未知的选择模式：%s" % mode)


# ══════════════════════════════════════════
# 画布
# ══════════════════════════════════════════

class FootholdItem(QGraphicsLineItem):
    """一条 foothold。**拾取范围比线本身宽**（`shape()` 用描边路径），否则细线根本点不中。"""

    def __init__(self, f, pick_world):
        super().__init__(float(f.x1), float(f.y1), float(f.x2), float(f.y2))
        self.f = f
        self._pick = float(pick_world)
        # 只定颜色和线型 —— **线宽不在这里定**：它随缩放折算（见 _ZoneView._pen）；
        # 这里先给一支 0 宽的笔，真正宽度由 _ZoneView._apply_widths 在重建末尾统一写。
        if f.is_wall:
            pen = QPen(QColor(C_WALL))
            pen.setStyle(Qt.DashLine)
        else:
            pen = QPen(QColor(C_FLOOR))
        super().setPen(pen)
        self.setZValue(1 if f.is_wall else 2)

    def shape(self):
        p = QPainterPath()
        p.moveTo(self.line().p1())
        p.lineTo(self.line().p2())
        st = QPainterPathStroker()
        st.setWidth(self._pick)
        return st.createStroke(p)


class _ZoneView(ZoomPanView):
    """左键留给选择（基类只管滚轮缩放 / 中键平移 / 双击适应，见 ZoomPanView 的说明）。"""

    picked = pyqtSignal(object, object)     # (id 集合, 模式) —— 交给窗口更新选择
    hovered = pyqtSignal(float, float)      # 场景坐标（状态行显示）

    def sizeHint(self):
        """**别拿场景尺寸当"我想要多大"**（2026-09-26 用户第 3 次报"窗口最小高度太大"）。

        `QGraphicsView` 默认把**场景尺寸**当 sizeHint —— 这张图的场景约 2237×1080，
        于是窗口的 sizeHint 被抬成 **2521×1182**：一显示、或任何一个布局重算
        （连"鼠标移动改状态行"都会触发），窗口就被拽成 **1222 高**（实测：明明
        `resize(760)` 了，实际变成 1222）。用户看到的正是"想拖小、它自己变回去"。

        视图的语义本来就是"**给多大画多大**"（sizePolicy 已经是 Expanding），
        sizeHint 不该表示"我需要多大"。这里只表个意：窗口尺寸由布局的 stretch
        和用户的拖拽决定（`_drop_stale_min_height` 那边保证地板只来自布局）。
        """
        return QSize(320, 200)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._press = None
        self._mode = "replace"
        #: 线条宽度（**屏幕像素**，见 set_line_width）与"上次真正写进 item 的值"
        self._line_w = float(theme.FOOTHOLD_W_DEFAULT)
        self._applied = None
        #: 所有线：[(item, 颜色, 线型, 是否归呼吸高亮管)]。**由 _rebuild_scene 填**
        #: （view 自己持一份：线宽要在缩放变化时统一重写，而 item 的字典在对话框上）
        self._lines = []

    # ---- 线宽：屏幕像素 → 场景单位 ----

    def set_line_width(self, px):
        """设置线条宽度（屏幕像素）→ 立刻按当前缩放重画一遍。

        **为什么单位是屏幕像素**（而不是场景单位）：编辑器的缩放跨度很大 ——
        整图 fit 时约 0.35 倍、放大看细节能到 8 倍以上。按场景单位给宽度的话，
        同一个值在两种视图下差二十多倍：整图时细到看不见、放大时糊成一片。
        """
        self._line_w = max(float(theme.FOOTHOLD_W_MIN),
                           min(float(theme.FOOTHOLD_W_MAX), float(px)))
        self._applied = None
        self._apply_widths()
        self.viewport().update()

    def _px(self):
        """1 屏幕像素 = 多少场景单位（缩放越大，同一个屏幕宽度对应的场景宽度越小）。"""
        s = self.transform().m11() or 1.0
        return 1.0 / max(1e-6, s)

    def _pen(self, color, extra=0.0, style=None):
        """按当前缩放造一支笔：**屏幕上正好 (_line_w + extra) 像素粗**。

        为什么不用 `QPen(color, 0)`（原先的写法）：0 宽是 cosmetic 笔，粗细**写死
        1 像素**、改不了 —— 这正是"想调粗一点"没法调的原因。而直接给场景宽度又会
        随缩放变粗变细。所以每次重绘把屏幕像素折算成场景单位（见 paintEvent）。
        """
        pen = QPen(QColor(color))
        pen.setWidthF(max(0.0, (self._line_w + extra) * self._px()))
        pen.setCapStyle(Qt.RoundCap)        # 粗线时线头不缺口（1px 时看不出来）
        if style is not None:
            pen.setStyle(style)
        return pen

    def _apply_widths(self):
        """把当前宽度写进所有线（呼吸高亮的那些除外 —— 它们由 `_on_pulse` 每 60ms 写）。"""
        for it, color, style, pulsed in self._lines:
            if pulsed:
                continue
            it.setPen(self._pen(color, style=style))
        self._applied = (self._line_w, round(self.transform().m11(), 4))

    def paintEvent(self, e):
        """重绘前核对一次：**缩放变了就重算笔宽**（屏幕上粗细保持不变）。

        挂在 paintEvent 是最省心的位置：任何缩放路径（滚轮 / 双击适应 / 聚焦集合 /
        fitInView）都会经过这里，不用逐个去挂钩子。`_applied` 保证只有真的变了才重写 ——
        否则每次 setPen 都会让 item 重新排队重绘，来回刷新停不下来。
        """
        if self._applied != (self._line_w, round(self.transform().m11(), 4)):
            self._apply_widths()
        super().paintEvent(e)

    def _axis_tol(self):
        """把屏幕上的 6px 换算成当前缩放下的世界像素（放大时更精确）。"""
        s = self.transform().m11() or 1.0
        return PICK_PX / max(1e-6, s)

    def _modifier_mode(self, e):
        if e.modifiers() & Qt.ShiftModifier or e.modifiers() & Qt.ControlModifier:
            return "add"
        if e.modifiers() & Qt.AltModifier:
            return "sub"
        return "replace"

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            p = self.mapToScene(e.pos())
            self._press = (p.x(), p.y())
            self._mode = self._modifier_mode(e)
            self.viewport().update()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        p = self.mapToScene(e.pos())
        self.hovered.emit(p.x(), p.y())
        if self._press is not None:
            self.viewport().update()        # 重画框选矩形
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._press is not None:
            x0, y0 = self._press
            self._press = None
            x1, y1 = self.mapToScene(e.pos()).x(), self.mapToScene(e.pos()).y()
            moved = abs(x1 - x0) + abs(y1 - y0)
            t = self._axis_tol()
            if moved <= t:
                # 视为点击：按"点选"处理（点空白 = 清空，除非在加选/减选）
                self.picked.emit(("click", x0, y0, t), self._mode)
            else:
                # 框选：左→右 = 包含；右→左 = 相交
                mode = "contains" if x1 >= x0 else "crossing"
                self.picked.emit(("box", (x0, y0, x1, y1), mode), self._mode)
            self.viewport().update()
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def drawForeground(self, painter, rect):
        """画框选矩形（左→右 实线蓝框 / 右→左 虚线绿框，和 CAD 的习惯一致）。"""
        if self._press is None:
            return
        cur = self.mapFromGlobal(self.cursor().pos())
        p0 = self.mapToScene(cur)
        x0, y0 = self._press
        r = QRectF(QPointF(x0, y0), p0).normalized()
        contain = p0.x() >= x0
        # 框选矩形比线粗 1 px：它是"正在操作"的东西，得压过底下的线
        painter.setPen(self._pen("#1a73e8" if contain else "#0b8043", extra=1.0,
                                 style=Qt.SolidLine if contain else Qt.DashLine))
        painter.setBrush(QBrush(QColor(26, 115, 232, 30) if contain
                                else QColor(11, 128, 67, 30)))
        painter.drawRect(r)


class _RichRowDelegate(QStyledItemDelegate):
    """列表行按**富文本**画 —— 一行里只给"集合名"上色时必须这样（Qt 默认只画纯文本）。

    用户 2026-09-26 的原话：可达项要显示成「**左上** → **右下** [走(walk)]」——
    两个名字蓝、`→` 和 `[走(walk)]` 是正常黑字。一根列表项只有一份纯文本，
    所以只能自己画。

    **为什么用委托而不是 `setItemWidget(QLabel)`**：列表每刷新一次都是 clear + 重填
    （`_refresh_edges` 一次动作就能触发好几回），item widget 会跟着反复建/销毁控件；
    而且控件会吃掉鼠标事件，选中、双击都要再转发一次。委托只接管"怎么把字画出来"，
    点击 / 双击 / tooltip / 选中底色全是 Qt 默认行为。

    约定：`DisplayRole` 存**纯文本**（tooltip 与自检从它取），富文本存 `ROLE_HTML`。
    **没有富文本就退回默认画法** —— 以后新加的行不会因为我这份委托变得怪。
    富文本里**不带颜色**的部分用调色板色：平时是正常黑字，选中时自动反色。
    """

    def paint(self, painter, opt, idx):
        rich = idx.data(ROLE_HTML)
        if not rich:
            super().paint(painter, opt, idx)
            return
        painter.save()
        o = QStyleOptionViewItem(opt)
        self.initStyleOption(o, idx)
        w = o.widget
        style = w.style() if w is not None else QApplication.style()
        o.text = ""                       # 背景（含选中底色）、图标、焦点框仍由样式画
        style.drawControl(QStyle.CE_ItemViewItem, o, painter, w)
        tr = style.subElementRect(QStyle.SE_ItemViewItemText, o, w)
        doc = QTextDocument()
        doc.setDefaultFont(o.font)
        doc.setDocumentMargin(0)
        t = QTextOption()
        t.setWrapMode(QTextOption.NoWrap)   # 行高是固定的：换行等于第二行看不见，干脆裁掉
        doc.setDefaultTextOption(t)
        doc.setHtml(rich)
        doc.setTextWidth(max(1.0, tr.width() - 4))
        ctx = QAbstractTextDocumentLayout.PaintContext()
        ctx.palette = o.palette            # 不带颜色的字跟着调色板 ⇒ 选中时自动反色
        painter.translate(tr.left() + 2, tr.top())
        doc.documentLayout().draw(painter, ctx)
        painter.restore()


class AddReachDialog(QDialog):
    """一次问完「从哪儿能到哪儿」：**终点 + 通行方式 + 它要求的东西（绳/门）**。

    为什么合成一个弹窗（2026-09-26 要求）：原来是 2~4 个 `QInputDialog` 串着弹
    （终点 → 类型 → 绳/门）—— 每弹一个都得重新回想"上一步选了啥"，选错了只能整个
    退出重来；而这几个参数本来就该**一起看**（选了「爬」才知道要挑绳、选了「传送门」
    才知道要挑门）。

    读法：`exec_()` 返回 Accepted 后取 `result_dict`：
        {"dst": 名字, "kind": "climb"|…, "ladder": "L2"|None, "portal": "west00"|None}

    **两种用法**（2026-09-26 加编辑）：
      · 增加：`init=None` ⇒ 三个下拉从零选，确认按钮写「增加」；
      · 编辑：`init=` 那条边现在的样子 ⇒ 预填到当前值、确认按钮写「保存」。
    同一个弹窗承担两件事，是因为"终点 + 类型 + 绳/门"在两种场景下**是同一组问题**：
    合成两套界面必然漂（改了一处忘另一处）。**起点两种模式都不改**（见 _on_edit_edge）。

    **非模态**（2026-09-26 用户要求）：开着它的时候还要能**缩放细看**编辑器那张图
    （"这条 foothold 在不在集合里、这个下跳点该不该留"），模态会把工作台锁住。
    非模态之后 `exec_()` 那条返回值没有了 ⇒ 结果靠 `applied` 信号交给调用方
    （和 `ZoneEditorDialog.saved_now` 同一个路子）。
    """

    #: 点了确定 → 结果字典（见 `_accept`）。**单实例 + 非模态**下调用方拿不到返回值**，
    #: 所以必须主动通知外面（外面据此真的加/改那条边）。
    applied = pyqtSignal(dict)

    def __init__(self, parent, src, dst_all, dst_related=(), ladders=(),
                 drop_choices=(), init=None, all_ladders=(),
                 portals=(), all_portals=()):
        super().__init__(parent)
        self.src = str(src)                 # 起点（外面加边时要用）
        self.mode = "add"                   # "add" / "edit" —— 由调用方设（见 _on_edit_edge）
        self.edge = None                    # 编辑模式下绑的那条边
        self.setModal(False)                # 非模态：开着它还能缩放看编辑器的图
        self._init = dict(init) if init else None
        self.setWindowTitle(("%s可达 —— %s" % ("编辑" if self._init else "增加", src)))
        self.setMinimumWidth(440)
        # ⭐ 跟**其它弹窗同一套**（用户 2026-09-28："居中、放弃位置记忆，我期望的是调整过的
        #   缩放数据记忆要有"✓ ⇒ 每次显示都**居中到主窗口** ✓、尺寸沿用拉过的那份 ✓）。
        #   ⚠ 这个窗原来**漏在规范外**（既没接几何、也没人发现 ✗）—— 这一轮补齐 ✓。
        theme.bind_window_state(self, "add_reach")
        self.result_dict = None
        self._src = str(src)
        # 终点候选**排序**（2026-09-26 要求"自动 sort"）：原来按注册顺序列，
        # 集合一多就得从头找；排序之后位置固定，找起来是"扫一眼"而不是"扫一遍"。
        self._all = sorted(str(n) for n in dst_all)
        self._rel = sorted(str(n) for n in dst_related)

        root = QVBoxLayout(self)
        head = QLabel("从「%s」能到哪儿：" % self._src)
        head.setWordWrap(True)
        root.addWidget(head)
        if self._init:
            # 起点**不许改**得说出来（不然人会找不到那一格，以为程序少做了功能）
            tip = QLabel("正在**改这条边**：起点固定是「%s」（要换起点就删掉这条、"
                         "再「增加可达」）。" % self._src)
            tip.setStyleSheet("color: #5f6368;")
            tip.setWordWrap(True)
            root.addWidget(tip)

        form = QFormLayout()

        # ---- 终点（默认只列与起点有关的；勾「全部」放开）----
        dst_row = QHBoxLayout()
        self.cmb_dst = NoWheelComboBox()
        self.cmb_dst.setMinimumWidth(190)
        dst_row.addWidget(self.cmb_dst, 1)
        self.ck_all = QCheckBox("全部")
        self.ck_all.setToolTip(
            "默认只列**与起点有关的**终点（建议涉及的 + 已经连着的）——\n"
            "集合一多，全列出来根本找不着。勾上就列这张图的全部集合。")
        dst_row.addWidget(self.ck_all)
        form.addRow("终点", dst_row)

        # ---- 通行方式（2026-09-26 定的术语：问的是"从 A 到 B **怎么过去**"）----
        self.cmb_kind = NoWheelComboBox()
        for k in zones.EDGE_LABELS:
            # 类型名走 `zones.kind_label`（唯一口径）：中文名里已经带了英文键的
            # （如「走(walk)」）不会再被拼一遍 —— 否则就是「走(walk)（walk）」。
            self.cmb_kind.addItem(zones.kind_label(k), k)
        self.cmb_kind.setToolTip(
            "从起点到终点**怎么过去**：\n"
            "  走       —— 走过去（同层、无缝）\n"
            "  爬（绳梯）—— 爬绳 / 梯子（要指名哪根绳）\n"
            "  跳       —— 站在 foothold **边缘按跳键**，平着 / 斜着蹦过去\n"
            "  下跳     —— **按住 ↓ 再按跳**，从平台上穿下去、落到下一层\n"
            "  传送点   —— **走到那扇门、按一下 ↑**（要指名**哪扇门**；2026-10-06 加的）\n\n"
            "⚠ 「跳」和「下跳」是**两种不同的按法**，别混：跳是往前蹦，下跳是往下穿。\n"
            "这两种都要用到跳键 ⇒ 真执行前得先做**跳跃标定**（现在先把位置标出来即可）。\n\n"
            "⚠ **传送点不支持跨图**（明确的口径）：这里只能选**本图内有出口**的门 ✓\n"
            "（跨图的门在下拉里会标出来、选不了它就没意义 ✓ 导出数据里也只给得到目标地图号 ✓）。")
        form.addRow("通行方式", self.cmb_kind)

        # ---- 类型要求的东西：按类型显隐（这正是一个弹窗才做得到的事）----
        self.lbl_lad = QLabel("爬哪根绳")
        self.cmb_lad = NoWheelComboBox()
        # ⭐⭐ **「全部」勾选**（用户 2026-10-06 ✓ 原话（问答里定的）："加全部勾选"）——
        #   照**终点**那一格的同款做法（`ck_all` ✓ 见 `_fill_dst` ✓）：默认只列
        #   **与起点集合有关**的绳（`ladders_touching` ✓ 见 `_reach_ladder_choices` ✓），
        #   勾上就列**本图全部**绳 ✓ —— 并且把"**跟起点集合无关**"的那些**在行里标出来** ✓
        #   （`（⚠ … 从「起点」走不到它）` ✓）。
        #   ⚠ 起因（用户 2026-10-06 现场）："为什么从 7 去 6 不能选 L6 绳子？" ✓ ——
        #     候选按**起点集合**筛（穿过它 / 某端落在它上 ✓）⇒ L6 与 7 无关就**根本不出现** ✗
        #     而界面上**一句话都没说** ⇒ 只能猜"是不是程序少做了功能" ✓。
        #   ⚠ 勾上之后**不**改变校验口径 ✗：真要走不到那根绳，运行时照样会失败 ✓
        #     （行里那句"走不到它"就是提醒 ✓ 见 `_fill_lad` ✓）。
        self._lad_rel = sorted(ladders, key=lambda x: _natural(x[0]))
        self._lad_all = sorted(all_ladders or ladders, key=lambda x: _natural(x[0]))
        lad_row = QHBoxLayout()
        lad_row.addWidget(self.cmb_lad, 1)
        self.ck_lad_all = QCheckBox("全部")
        self.ck_lad_all.setToolTip(
            "默认只列**与起点集合有关**的绳（穿过起点集合的、或某一端落在它上面的）——\n"
            "集合一多，全列出来根本找不着。勾上就列**这张图的全部绳** ✓。\n\n"
            "⚠ 与起点集合**无关**的那些会**在行里标出来**："
            "「（⚠ 从「…」走不到它）」—— 选了它，运行时多半会走不过去 ✗\n"
            "（正确做法一般是拆成两步：先「走」到那根绳所在的那个集合，再「爬」它 ✓）。")
        self.ck_lad_all.toggled.connect(lambda _v: self._fill_lad())
        lad_row.addWidget(self.ck_lad_all)
        form.addRow(self.lbl_lad, lad_row)
        self._fill_lad()

        # ---- 「走哪个门」（传送点 ✓ 2026-10-06 加回 ✓）----
        # 用户原话："现在增加一种新的通行方式'传送点(portal)'供寻路编辑器→增加可达 弹窗里配置；
        #   选中后，可以选择传送点（例如黄金沙滩项目，地形叠加图里的"h009"），操作方式是走到该
        #   点位按↑" ✓ ⇒ 这一行就是"选哪扇门" ✓。
        # 写法与**绳那一行同款**（默认只列与起点集合有关的 + 「全部」放开 + 无关的标出来 ✓）。
        self.lbl_gate = QLabel("走哪扇门")
        self.cmb_gate = NoWheelComboBox()
        self._gate_rel = sorted(portals, key=lambda x: _natural(x[0]))
        self._gate_all = sorted(all_portals or portals, key=lambda x: _natural(x[0]))
        gate_row = QHBoxLayout()
        gate_row.addWidget(self.cmb_gate, 1)
        self.ck_gate_all = QCheckBox("全部")
        self.ck_gate_all.setToolTip(
            "默认只列**与起点集合有关**的门（落在起点集合那一带 ±24px 内的 ✓）——\n"
            "集合一多，全列出来根本找不着。勾上就列**这张图的全部门** ✓。\n\n"
            "⚠ 与起点集合**无关**的、以及**跨图**的门会**在行里标出来**：\n"
            "「（⚠ 从「…」走不到它）」/「（⚠ 跨图 —— 不支持）」—— 选了它们，运行时走不过去 ✗\n"
            "（跨图是**明确不支持**的：导出数据里只给得到目标**地图号**，目标点位没有 ✓）。")
        self.ck_gate_all.toggled.connect(lambda _v: self._fill_gate())
        gate_row.addWidget(self.ck_gate_all)
        form.addRow(self.lbl_gate, gate_row)
        self._fill_gate()

        # ---- 走(walk) 的方向类型（2026-09-26 用户要求）----
        self.lbl_wd = QLabel("类型")
        self.cmb_wd = NoWheelComboBox()
        for v in zones.WALK_DIRS:
            self.cmb_wd.addItem(zones.WALK_DIR_LABELS[v], v)
        self.cmb_wd.setToolTip(
            "「走」的方向类型（**逐边**配，2026-09-27 起真的生效）：\n"
            "· 默认方向 = 朝**目标集合所有落点 x 的中点**走；\n"
            "· 仅向左 / 仅向右 = **只按那一边**，一直按到踏上目标集合（不看中点）——\n"
            "  适合「从平台边缘往左走出去、掉到下面那层」这种边。\n\n"
            "⚠ 只按一边时若 x 一直不动（被墙挡住 / 方向配反了），会在「走不动了」\n"
            "那一刻如实报出来，不会干等到超时。")
        form.addRow(self.lbl_wd, self.cmb_wd)

        # ---- 「爬（绳梯）」的**中途跳下**（2026-09-28 用户要求 ✓）----
        # ⭐ **二级联动**：先选方向（`cmb_mid`）⇒ 选了**非空**才出现「高度」（`spn_mid_y`）✓。
        # ⚠ 这是全项目**第一个"下拉引起的二级显隐"** ⇒ 一级样板是 `cmb_kind`/`_sync_kind` ✓
        #   （⚠ 别忘把 `cmb_mid` 的变化也接进 `_sync_kind` ✗ —— 不然改了下拉、"高度"那行不跟着动 ✓）。
        self.lbl_mid = QLabel("中途跳下")
        self.cmb_mid = NoWheelComboBox()
        for _v in zones.MID_JUMP_DIRS:
            self.cmb_mid.addItem(zones.MID_JUMP_LABELS[_v], _v)
        self.cmb_mid.setToolTip(
            "「爬」的**中途跳下**（**逐边**配，2026-09-28 用户要求）：\n"
            "· **（不中途跳下）** = 老行为，整段爬到底（一个字不变 ✓）；\n"
            "· **默认（向目标中心）/ 仅向左 / 仅向右** = 爬到某一点就跳下，方向按这里选的 ✓"
            "（向目标中心 = 朝目标集合的 x 中点；仅向左 / 仅向右 = 只按那一边 ✓）。\n"
            "选了非空的方向后，下面会多出「**高度**」一格 ✓。")
        form.addRow(self.lbl_mid, self.cmb_mid)

        # 「高度」= 中途跳下那一处的**世界坐标 y（⚠ 越小越靠上）**；只有方向非空才出现 ✓。
        self.lbl_mid_y = QLabel("高度")
        # ⚠ 必须用 `NoWheelDoubleSpinBox`（UI 规范：**滚轮不许改参数** ✗ `check_ui` 会拦 ✓）
        self.spn_mid_y = NoWheelDoubleSpinBox()
        self.spn_mid_y.setRange(-4000.0, 4000.0)
        self.spn_mid_y.setDecimals(0)
        self.spn_mid_y.setValue(0.0)
        self.spn_mid_y.setToolTip(
            "「中途跳下」的**高度** —— 世界坐标 **y**（⚠ **越小越靠上**）\n"
            "（就是小地图算出来的那套坐标 ✓，和「寻路编辑器」里 foothold 的 y 同一套 ✓）。\n\n"
            "执行器那边的判据是「**人已经爬到该高度或更高**」（`py ≤ 高度` ✓ —— 世界 y 越小越靠上）：\n"
            "· 往上爬时人**从绳下端**出发 ⇒ 爬到你这儿配的高度**就跳下去** ✓；\n"
            "· 想「**永不触发**」得配得比这根绳的**上端**还小 ✓。\n\n"
            "⚠ 这条判据**改过一次**（2026-09-28）：原来是「人的 y ≥ 高度」（= 人在这点的**下方**）"
            "⇒ 一上绳就满足 ⇒ 一触发就把机会用掉 ⇒ **之后爬到顶也不跳** ✗（现场就是这么坏的 ✓）。")
        form.addRow(self.lbl_mid_y, self.spn_mid_y)
        root.addLayout(form)

        # ---- 下跳：**从哪些 foothold 起跳**（2026-09-26 用户要求）----
        # 为什么是列表而不是下拉：一条下跳边上往往有好几个能下去的点，要能多选、
        # 能删掉不该有的。**默认 = 起点集合的全部**（`zones.drop_footholds` 同一口径）——
        # 只有人工增删过，才会把这一格写进文件。
        self.drop_host = QWidget()
        lay_d = QVBoxLayout(self.drop_host)
        lay_d.setContentsMargins(0, 0, 0, 0)
        lbl_d = QLabel("可下跳 foothold（默认 = 起点集合的全部）")
        lbl_d.setWordWrap(True)
        lay_d.addWidget(lbl_d)
        self.lst_drop = QListWidget()
        self.lst_drop.setMaximumHeight(130)
        self.lst_drop.setToolTip(
            "**这条下跳边能从哪些 foothold 起跳**。\n\n"
            "默认是起点集合的全部 foothold；把不该有的删掉（比如那里下面是墙 /\n"
            "掉下去回不来），或者把删掉的加回来。\n\n"
            "**选中一行 ⇒ 集合编辑器里那条 foothold 会呼吸**（其它高亮临时让位），\n"
            "照着画面判断该不该留它 —— 比看坐标快得多。")
        self.lst_drop.currentItemChanged.connect(
            lambda *_: self._preview_drop())
        lay_d.addWidget(self.lst_drop)
        row_d = QHBoxLayout()
        self.btn_drop_add = QPushButton("加一个…")
        self.btn_drop_add.setToolTip(
            "从**起点集合的 foothold** 里挑一个加回来（下跳的起点必须在起点集合里）。")
        self.btn_drop_add.clicked.connect(self._drop_add)
        self.btn_drop_del = QPushButton("移出")
        self.btn_drop_del.setToolTip("把选中的那条移出（不写进文件 ⇒ 执行器不会从它下跳）。")
        self.btn_drop_del.clicked.connect(self._drop_del)
        row_d.addWidget(self.btn_drop_add)
        row_d.addWidget(self.btn_drop_del)
        row_d.addStretch(1)
        lay_d.addLayout(row_d)
        root.addWidget(self.drop_host)
        self._drop_all = [(str(i), t) for i, t in drop_choices]   # 候选（起点集合的）
        init_ids = (init or {}).get("footholds")
        self._drop_ids = ([str(x) for x in init_ids] if init_ids is not None
                          else [i for i, _t in self._drop_all])

        self.lbl_note = QLabel("")
        self.lbl_note.setStyleSheet("color: #b06000;")
        self.lbl_note.setWordWrap(True)
        root.addWidget(self.lbl_note)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_ok = box.button(QDialogButtonBox.Ok)
        self.btn_ok.setText("保存" if self._init else "增加")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        theme.unify_ok_cancel(box.button(QDialogButtonBox.Ok),
                              box.button(QDialogButtonBox.Cancel))
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        root.addWidget(box)

        self.ck_all.toggled.connect(lambda *_: self._fill_dst())
        self.cmb_kind.currentIndexChanged.connect(lambda *_: self._sync_kind())
        # ⚠ 「中途跳下」**自己变了也要重算显隐**（二级联动 ✓）：它决定「高度」那一行出不出现 ✓
        #   （只接 `cmb_kind` 的话，改完下拉「高度」不跟着动 ✗ —— 这类漏接只能靠这条说明防 ✓）
        self.cmb_mid.currentIndexChanged.connect(lambda *_: self._sync_kind())
        self._fill_dst()
        self._fill_drop()
        # ⭐ 爬的「中途跳下」**回填**（2026-09-28 ✓）—— 放在 `_sync_kind()` **之前** ✗：
        #   那样它会按**刚填进去**的值决定「高度」那一行显示不显示 ✓（顺序反了会一直是隐藏 ✓）
        if self._init:
            _d0 = (self._init.get("mid_dir")
                   if isinstance(self._init, dict) else None)
            _k0 = self.cmb_mid.findData(zones.mid_jump_dir({"mid_dir": _d0}))
            if _k0 >= 0:
                self.cmb_mid.setCurrentIndex(_k0)
            try:
                if _d0:
                    self.spn_mid_y.setValue(float(self._init.get("mid_y") or 0.0))
            except (TypeError, ValueError):
                pass
        self._sync_kind()
        if self._init:
            self._apply_init()

    def _apply_init(self):
        """编辑模式：把这条边**现在的样子**填进三个下拉。

        ⚠ 终点可能是**悬空边**指向的那个集合（已经改名/删掉）—— 那时它在候选里不存在，
        下拉就停在第一项：保存等于把这条悬空边**修好**，而那正是最该能做的事（不然只能
        删掉重加）。所以要**先放开「全部」再找一次**，别让"相关集合"把它藏起来。
        """
        d = self._init
        # `init` 的来源有两种写法，这里都得认：**原始边**用 `to`（zones 数据结构），
        # 而本弹窗自己的结果字典用 `dst`（见 _accept）。踩过：只认 `dst` ⇒ 传原始边时
        # 静悄悄预填不上，终点停在第一项。
        dst = d.get("dst") or d.get("to")
        if dst:
            j = self.cmb_dst.findData(dst)
            if j < 0 and not self.ck_all.isChecked():
                self.ck_all.setChecked(True)     # 会触发 _fill_dst 重填候选
                j = self.cmb_dst.findData(dst)
            if j >= 0:
                self.cmb_dst.setCurrentIndex(j)
        j = self.cmb_kind.findData(d.get("kind"))
        if j >= 0:
            self.cmb_kind.setCurrentIndex(j)     # 触发 _sync_kind ⇒ 绳那一行跟着显隐
        # ⭐ 绳 / 门**同一套**（2026-10-06 ✓）：编辑一条**它不在"相关"里**的边时，自动勾上
        #   「全部」再找一次 ✓（照终点那格对 `ck_all` 的做法 ✓ —— 否则预填不上、界面停在
        #   第一项上 ✗，而人看到的只是"打开时选错了"，根本联想不到是候选被收窄了 ✓）。
        for cmb, key, ck, avail in (
                (self.cmb_lad, "ladder", self.ck_lad_all, "_lad_all_avail"),
                (self.cmb_gate, "portal", self.ck_gate_all, "_gate_all_avail")):
            v = d.get(key)
            if not v:
                continue
            k = cmb.findData(v)
            if k < 0 and getattr(self, avail, False):
                ck.setChecked(True)                  # 触发重填
                k = cmb.findData(v)
            if k >= 0:
                cmb.setCurrentIndex(k)
        # 「走」的方向类型（默认方向 = 不写这一格 ⇒ 找不到就用第 0 项）
        k = self.cmb_wd.findData(zones.walk_dir(d))
        if k >= 0:
            self.cmb_wd.setCurrentIndex(k)

    def _fill_lad(self):
        """按「全部」勾选重填绳候选（照 `_fill_dst` 的同款做法 ✓ 用户 2026-10-06 ✓ "加全部勾选"）。

        ⚠ 与起点集合**无关**的绳要**在行里标出来** ✗ —— 只列出来不吭声，等于递给人一个陷阱 ✓
          （用户现场那句"为什么从 7 去 6 不能选 L6"就是"看不见 / 说不清"逼出来的 ✓）。
          判据只此一处：**在不在** `self._lad_rel` 里（= `ladders_touching(起点集合)` ✓）。
        ⚠ 重填时**保住当前选中**（预填 / 编辑模式靠它 ✓ 同 `_fill_dst` ✓）。
        """
        cur = self.cmb_lad.currentData()
        rel = {lid for lid, _t in self._lad_rel}
        use = self._lad_all if self.ck_lad_all.isChecked() else self._lad_rel
        self.cmb_lad.blockSignals(True)
        self.cmb_lad.clear()
        for lid, text in use:
            if lid not in rel:
                text = ("%s（⚠ 从「%s」走不到它 —— 一般先「走」到它所在的集合、再「爬」它）"
                        % (text, self._src))
            self.cmb_lad.addItem(text, lid)
        j = self.cmb_lad.findData(cur)
        self.cmb_lad.setCurrentIndex(j if j >= 0 else 0)
        self.cmb_lad.blockSignals(False)
        # 没有"相关"可收窄时就没有这个开关（勾了也没意义 ✓ 同 `_fill_dst` ✓）
        # ⚠⚠ 判据**必须**存成显式标志 ✗ —— 别用 `self.ck_lad_all.isVisible()`：控件在
        #   弹窗 `show()` 之前 `isVisible()` **恒为 False** ✓（实测：构造完就问它 ⇒ 永远 False，
        #   "该隐藏"那条分支根本不执行 ✗；而"该显示"的那次 `setVisible(True)` 其实已经记下了 ✓
        #   —— 于是行为看着"时对时不对"，最难查 ✓）。
        self._lad_all_avail = len(self._lad_all) > len(self._lad_rel)
        self.ck_lad_all.setVisible(self._lad_all_avail)
        if not self._lad_all_avail and self.ck_lad_all.isChecked():
            self.ck_lad_all.blockSignals(True)
            self.ck_lad_all.setChecked(False)
            self.ck_lad_all.blockSignals(False)

    def _fill_gate(self):
        """按「全部」勾选重填门候选（照 `_fill_lad` 同款 ✓ 用户 2026-10-06 ✓ "走哪个门"）。

        ⚠ 三类都要**在行里说清**（别静默 ✗）：与起点集合**无关**的 ✓、**跨图**的 ✗、
          落点查不到的 ✗ —— 后两类的标记由调用方揉进行文（`ZoneEditorDialog._portal_row` ✓），
          这里只补"与起点无关"那一句 ✓（判据只此一处：在不在 `self._gate_rel` ✓）。
        ⚠ 重填时**保住当前选中** ✓（预填 / 编辑模式靠它 ✓）。
        ⚠ 判据用显式标志 `_gate_all_avail`（**别用 `isVisible()`** ✗ —— 弹窗 `show()` 之前
          恒为 False ✓ 同 `_fill_lad` 那个坑 ✓）。
        """
        cur = self.cmb_gate.currentData()
        rel = {pn for pn, _t in self._gate_rel}
        use = self._gate_all if self.ck_gate_all.isChecked() else self._gate_rel
        self.cmb_gate.blockSignals(True)
        self.cmb_gate.clear()
        for pn, text in use:
            if pn not in rel:
                text = ("%s（⚠ 从「%s」走不到它 —— 一般先「走」到它所在的那个集合）"
                        % (text, self._src))
            self.cmb_gate.addItem(text, pn)
        j = self.cmb_gate.findData(cur)
        self.cmb_gate.setCurrentIndex(j if j >= 0 else 0)
        self.cmb_gate.blockSignals(False)
        self._gate_all_avail = len(self._gate_all) > len(self._gate_rel)
        self.ck_gate_all.setVisible(self._gate_all_avail)
        if not self._gate_all_avail and self.ck_gate_all.isChecked():
            self.ck_gate_all.blockSignals(True)
            self.ck_gate_all.setChecked(False)
            self.ck_gate_all.blockSignals(False)

    def _fill_dst(self):
        cur = self.cmb_dst.currentData()
        use = self._all if (self.ck_all.isChecked() or not self._rel) else self._rel
        self.cmb_dst.blockSignals(True)
        self.cmb_dst.clear()
        for n in use:
            self.cmb_dst.addItem(n, n)
        j = self.cmb_dst.findData(cur)
        self.cmb_dst.setCurrentIndex(j if j >= 0 else 0)
        self.cmb_dst.blockSignals(False)
        # 没有"相关"可收窄时就没有这个开关（勾了也没意义）
        self.ck_all.setVisible(bool(self._rel) and len(self._all) > len(self._rel))
        self.ck_all.blockSignals(True)
        self.ck_all.setChecked(self.ck_all.isChecked() or not self._rel)
        self.ck_all.blockSignals(False)

    def _sync_kind(self):
        """按类型显隐"绳 / 可下跳 foothold"那几行，并在缺料时说清**为什么不能加**。"""
        kind = self.cmb_kind.currentData()
        self.lbl_lad.setVisible(kind == "climb")
        self.cmb_lad.setVisible(kind == "climb")
        # ⭐ 「走哪扇门」只对「传送点」有意义（2026-10-06 加回 ✓）
        self.lbl_gate.setVisible(kind == "portal")
        self.cmb_gate.setVisible(kind == "portal")
        self.lbl_wd.setVisible(kind == "walk")      # 「类型」只对「走」有意义
        self.cmb_wd.setVisible(kind == "walk")
        # ⭐ 「中途跳下」只对「爬」有意义；⚠ 「高度」还要**方向非空**才出现（**二级联动** ✓ ——
        #   全项目第一处：一级是 `cmb_kind`，二级是 `cmb_mid` ✓）
        self.lbl_mid.setVisible(kind == "climb")
        self.cmb_mid.setVisible(kind == "climb")
        _show_y = (kind == "climb" and bool(self.cmb_mid.currentData()))
        self.lbl_mid_y.setVisible(_show_y)
        self.spn_mid_y.setVisible(_show_y)
        self.drop_host.setVisible(kind == "drop")
        why = ""
        if kind == "climb" and self.cmb_lad.count() == 0:
            why = ("「%s」附近没有可爬的绳 —— 换个类型，或者先把绳另一端那块地形"
                   "圈成集合。" % self._src)
        if kind == "portal" and self.cmb_gate.count() == 0:
            # ⭐ 一条都没得选 ⇒ 告诉人**为什么**（别让人对着空下拉猜 ✗ 2026-10-06 ✓）：
            #   这张图**根本没有传送点**、或者（可本图的都在别的集合那一带 ⇒ 勾「全部」能看到 ✓）。
            why = ("这张图（或「%s」这一带）没有可选的传送点 —— 换个类型；"
                   "或者勾上门的「全部」看看这张图到底有哪些门（与起点无关的会标出来 ✓）。"
                   % self._src)
        if kind == "drop" and not self._drop_ids:
            why = "一个可下跳的 foothold 都没有了 —— 至少留一个（点「加一个…」）。"
        self.lbl_note.setText(why)
        self.btn_ok.setEnabled(not why)

    # ---- 可下跳 foothold 那张表 ----

    def _fill_drop(self):
        self.lst_drop.blockSignals(True)
        self.lst_drop.clear()
        texts = dict(self._drop_all)
        for i in self._drop_ids:
            it = QListWidgetItem(texts.get(i, "fh %s（不在起点集合里）" % i))
            it.setData(Qt.UserRole, i)
            self.lst_drop.addItem(it)
        self.lst_drop.blockSignals(False)
        self._sync_kind()

    def _preview_drop(self):
        """选中一行 ⇒ 让编辑器只呼吸这一条 foothold（其它高亮临时让位）。

        为什么挂在父窗口上：这个弹窗是编辑器开的（parent = 编辑器）⇒ 直接问它要。
        自检里 parent 可能是 None（不传）⇒ 什么都不做。
        """
        it = self.lst_drop.currentItem()
        fn = getattr(self.parent(), "preview_foothold", None)
        if fn is not None:
            fn(it.data(Qt.UserRole) if it is not None else None)

    def _drop_add(self):
        have = set(self._drop_ids)
        left = [(i, t) for i, t in self._drop_all if i not in have]
        if not left:
            QMessageBox.information(self, "没有可加的",
                                    "起点集合里的 foothold 都在这张表里了。")
            return
        pick, ok = QInputDialog.getItem(self, "加一个可下跳 foothold",
                                        "从起点集合里挑：",
                                        ["%s　%s" % (t, i) for i, t in left], 0, False)
        if not ok or not pick:
            return
        fid = left[[("%s　%s" % (t, i)) for i, t in left].index(pick)][0]
        self._drop_ids.append(fid)
        self._fill_drop()

    def _drop_del(self):
        it = self.lst_drop.currentItem()
        if it is None:
            return
        idx = self.lst_drop.currentRow()
        fid = it.data(Qt.UserRole)
        self._drop_ids = [x for x in self._drop_ids if x != fid]
        self._fill_drop()
        # 连着删几条时要顺手：**删完自动选下一条**（不然每删一条都得再点一下，
        # 而且"没选中"时再点「移出」会没反应，像是坏了）。
        if self.lst_drop.count():
            self.lst_drop.setCurrentRow(min(idx, self.lst_drop.count() - 1))
        self._preview_drop()

    def _drop_changed(self):
        """人工增删过没有 ⇒ 决定要不要把这一格写进文件（没动过就留空 = 用默认）。"""
        return sorted(self._drop_ids) != sorted(i for i, _t in self._drop_all)

    def done(self, r):
        """关窗（确定 / 取消都算）⇒ **把预览还回去**，别让编辑器一直只亮着一条。"""
        fn = getattr(self.parent(), "clear_foothold_preview", None)
        if fn is not None:
            fn()
        super().done(r)

    def _accept(self):
        kind = self.cmb_kind.currentData()
        if self.cmb_dst.currentData() is None:
            return                              # 一个终点都没有（理论上到不了这儿）
        self.result_dict = {
            "dst": self.cmb_dst.currentData(),
            "kind": kind,
            "ladder": (self.cmb_lad.currentData() if kind == "climb" else None),
            # ⭐ 走哪扇门（2026-10-06 ✓）：与 `ladder` 同款 —— **非传送点一律 None** ✓
            #   （`edit_edge` 那边见 None 就把这一格删掉 ⇒ 换类型不会留下上一任的门 ✓）。
            "portal": (self.cmb_gate.currentData() if kind == "portal" else None),
            # 下跳：**只有人工改过才带这一格**（没改 = 用起点集合的全部 ⇒ 不写文件）
            "footholds": (list(self._drop_ids)
                          if kind == "drop" and self._drop_changed() else None),
            # 走：方向类型（默认方向 = None ⇒ 不写文件）
            "dir": (self.cmb_wd.currentData() or None) if kind == "walk" else None,
            # ⭐ 爬：**中途跳下**（2026-09-28 ✓）—— 与 `dir` 同款写法：**非爬一律 None** ✓；
            # ⚠ 「高度」只在**方向非空**时才算数（方向空 = 不启用 ⇒ 高度没意义 ✓）
            "mid_dir": ((self.cmb_mid.currentData() or None)
                        if kind == "climb" else None),
            "mid_y": ((float(self.spn_mid_y.value())
                       if self.cmb_mid.currentData() else None)
                      if kind == "climb" else None),
        }
        # 非模态 ⇒ 调用方拿不到 `exec_()` 的返回值，用信号把结果送出去
        self.applied.emit(dict(self.result_dict))
        self.accept()


# ══════════════════════════════════════════
# 编辑器窗口
# ══════════════════════════════════════════

class ZoneEditorDialog(QDialog):
    """集合编辑器。`zones_path` 只有自检用（默认写 datasets/map/<id>.zones.json）。

    **非模态、单实例**（由 `gui/route_panel._on_edit_zones` 管理）：
    圈集合时要照着实时页的画面判断平台、还要反复跑一下看世界坐标对不对，
    模态对话框会把整个工作台锁住。
    """

    #: 保存成功 → (地图 id, 集合个数)。**非模态下没法靠返回值知道"后来存过没有"**，
    #: 所以用信号把这件事告诉面板（它据此更新那句提示）。
    saved_now = pyqtSignal(str, int)

    #: 窗口最小**宽度**（560）：再窄按钮上的字就挤没了。高度**一律不钉** ——
    #: 高度只由布局算（见 __init__ 的说明与 _drop_stale_min_height）。
    MIN_W = 560

    def __init__(self, map_id, terrain=None, parent=None, zones_path=None):
        super().__init__(parent)
        self.setWindowTitle("foothold 集合编辑器 —— %s" % map_id)
        self.resize(1180, 760)
        # **只钉宽度，不钉高度**：高度交给布局自己算（2026-09-26 在真实 Windows 平台上
        # 实测 = 222px，离屏环境 211px）。
        # ⚠ 这里踩过一次：上一版写了 `setMinimumSize(720, 420)` —— 本意是"允许缩到比较小"，
        # 实际是给纵向钉了 420 的**地板**，比布局需要的 222 大一倍，于是"还是缩不下去"。
        # 窗口能缩多矮由**子控件的最小尺寸**决定（现在三栏：另外两栏在滚动区里，
        # 画布那栏最小 120，见 `_build` 里的 `_panel`），
        # 想放宽就放宽那边，别在这里加硬地板。
        # ⚠⚠ 更坑的是：删掉这行**旧窗口照样缩不下去** —— 编辑器非模态单实例，`close()`
        # 只是隐藏、对象还活着，那个 420 一直挂在它身上（见 _drop_stale_min_height）。
        self.setMinimumWidth(self.MIN_W)
        self.map_id = str(map_id)
        self.terrain = terrain if terrain is not None else mapdata.load(self.map_id)
        if self.terrain is None:
            raise RuntimeError("没有 %s 的地形数据 —— 先在「路线识别」里点「生成地形图」"
                               % self.map_id)
        self._zones_path = Path(zones_path) if zones_path else zones.zones_path(self.map_id)
        self.zones = self._load_zones()
        self._sel = set()                   # 当前选中的 foothold id（字符串）
        #: 「**点选着的绳梯**」（存 `Ladder` 对象本身；`None` = 没选）——
        #: 用户 2026-09-27 要求："foothold 编辑器希望能**点选绳梯**（为了快速查看它的信息），
        #: **不用接逻辑**"。
        #: ⚠ 它是**只读的"看着它"**：**不进 `_sel`**（那套是 foothold 编辑用的 ✓）、
        #: 不参与 `register` / 增删成员 / 写文件 / 撤销栈 ✗ —— 只影响状态行那几行字
        #: 与画布上那根绳的颜色/呼吸 ✓（见 `_pick_ladder` / `_refresh_status`）。
        self._sel_ladder = None
        #: 「**点选着的传送点**」（存 `Portal` 对象本身；`None` = 没选）—— 用户 2026-10-06 ✓
        #: 原话："**选中传送点后，要放大传送点的图形呼吸高亮，不然看不出来**"✓。
        #: 与 `_sel_ladder` **完全同款**：只影响状态行 + 画布那扇门（画大 + 选中色 + 呼吸 ✓），
        #: **不碰** `_sel` / 集合高亮 / 文件 / 撤销栈 ✗；两者**互斥**（一次只亮一样 ✓）。
        self._sel_portal = None
        self._undo, self._redo = [], []
        self._highlight = None              # 当前高亮的集合名
        self._items = {}
        self._labels = {}                   # {集合名: 名字那个 item}（标在包围盒上方）
        self._ladder_items = []
        self._hl_items = []                 # [(item, 底色)] —— 呼吸高亮的对象
        self._arrow_items = []              # [(边, 连线item, 三角item, 类型色)]
        self._edge_hl = None                # 列表里选中的那条边（它的箭头一起呼吸）
        self._pulse = 0.0
        self._bg_item = None                # 底图 item（透明度滑条直接改它）
        #: 「增加可达」里选中"可下跳 foothold"时**只亮这一条**（其余高亮临时让位）——
        #: 详见 preview_foothold / _rebuild_scene
        self._preview_fid = None
        #: 当前开着的「增加/编辑可达」窗口（**单实例**，见 _open_reach）
        self._reach_dlg = None
        self._bg_opacity = BG_OPACITY_DEF / 100.0
        self.saved = False
        # 非模态：圈集合时要照着实时页的画面判断平台，模态会把整个工作台锁住、
        # 只能不停地关窗开窗（见 gui/route_panel._on_edit_zones 的单实例管理）。
        self.setModal(False)
        self._build()
        self._rebuild_scene()
        self._refresh_list()
        theme.bind_window_state(self, "zone_editor")   # 拉过的大小/位置按客户端记住 ✓
        # 画布的**缩放**也记住（用户 2026-09-27："每次打开弹窗都要重新调缩放"✗ ——
        # 窗口尺寸那件事早就记了，他每次重调的是画布里的 zoom ✓ 见 `theme.bind_view_zoom`）
        theme.bind_view_zoom(self.view, "zone_editor_view")

    # ---------------- 数据 ----------------

    def _load_zones(self):
        if not self._zones_path.exists():
            return zones.Zones(self.map_id)
        return zones.Zones.from_dict(
            json.loads(self._zones_path.read_text(encoding="utf-8")))

    def _snapshot(self):
        """改动**之前**拍一张（集合文件很小，整份存最省心）。"""
        return json.dumps(self.zones.to_dict(), ensure_ascii=False)

    def _commit(self, snap):
        """一次改动**成功之后**才登记撤销点。

        **失败的操作不许占一格 Ctrl+Z**：否则用户按撤销会觉得"没反应" ——
        实际是被那次失败（比如集合重名）吃掉了。实测踩过：自检里"重名注册失败 →
        撤销"竟然什么都没撤销，就是因为失败路径已经把快照压进去了。
        """
        self._undo.append(snap)
        if len(self._undo) > 100:
            del self._undo[0]
        self._redo = []
        self._after_change()

    def undo(self):
        if not self._undo:
            return
        self._redo.append(json.dumps(self.zones.to_dict(), ensure_ascii=False))
        self.zones = zones.Zones.from_dict(json.loads(self._undo.pop()))
        self._after_change()

    def redo(self):
        if not self._redo:
            return
        self._undo.append(json.dumps(self.zones.to_dict(), ensure_ascii=False))
        self.zones = zones.Zones.from_dict(json.loads(self._redo.pop()))
        self._after_change()

    def _after_change(self):
        self._refresh_list()
        self._rebuild_scene()

    # ---------------- 选择 ----------------

    def select(self, ids, mode="replace"):
        self._sel = new_selection(self._sel, ids, mode)
        self._refresh_status()
        self._rebuild_scene()
        # 「可到达」是按"在编辑谁"收窄的，而**选 foothold 也会改变那个焦点**
        # （一批只属于一个集合时，就用它当焦点）—— 所以这里必须跟着刷。
        self._refresh_edges()

    def selection_ids(self):
        return set(self._sel)

    # ---------------- 集合操作 ----------------

    def register(self, name):
        """把当前选中注册成集合 → 名字（重复/空名会抛 ValueError，界面负责提示）。"""
        if not self._sel:
            raise ValueError("先在地图上选中 foothold（点选或框选）")
        snap = self._snapshot()
        out = self.zones.add_set(name, sorted(self._sel))   # 失败会抛，此时 zones 没动
        # **先记高亮，再 _commit**：_commit 里会重建场景，晚设等于"刚注册完看不见高亮"，
        # 得再去点一下列表才亮（实测就是这么别扭）。
        self._highlight = out
        # 注册完**清掉视窗选择**：注册这个动作已经让"集合"成为当前编辑对象了（同
        # _on_pick_set 的规则）。留着选择会让「选中的」（C_SEL）和「集合的」（C_IN_SET）
        # 两套颜色同时在呼吸，看不出谁是谁。
        self._sel = set()
        # 「看着的绳梯」同理要收掉：它用的是**同一个选中色**（`C_SEL` ✓），留着就又是
        # "两样东西一起亮" ✗（用户 2026-09-26 定的"一次只编辑一样"✓）。
        self._sel_ladder = None
        self._sel_portal = None                 # 传送点同理（2026-10-06 ✓）
        self._commit(snap)
        self._refresh_status()
        return out

    def delete_set(self, name):
        snap = self._snapshot()
        gone = self.zones.remove_set(name)
        if self._highlight == name:
            self._highlight = None
        self._commit(snap)
        return gone

    def rename(self, old, new):
        snap = self._snapshot()
        out = self.zones.rename_set(old, new)
        self._highlight = out          # 同上：必须在 _commit（会重建场景）之前
        self._commit(snap)
        return out

    def add_to_set(self, name):
        if not self._sel:
            raise ValueError("先选中 foothold（在视窗里点选/框选几条）。\n"
                             "注意：点集合会清掉视窗里的选择，所以若是「先选 foothold "
                             "再点集合」的顺序，选择已经没了 —— 按住 Shift 再选就不会清。")
        snap = self._snapshot()
        self.zones.sets[name]["footholds"] = sorted(
            set(self.zones.sets[name]["footholds"]) | set(self._sel))
        self._commit(snap)

    def remove_from_set(self, name):
        if not self._sel:
            raise ValueError("先选中 foothold（在视窗里点选/框选几条）。\n"
                             "注意：点集合会清掉视窗里的选择，所以若是「先选 foothold "
                             "再点集合」的顺序，选择已经没了 —— 按住 Shift 再选就不会清。")
        snap = self._snapshot()
        self.zones.sets[name]["footholds"] = sorted(
            set(self.zones.sets[name]["footholds"]) - set(self._sel))
        self._commit(snap)

    def save(self):
        """校验后落盘 → (路径, 问题列表)。有问题**不挡着保存**，但要人知道。"""
        probs = self.zones.validate(self.terrain)
        self._zones_path.parent.mkdir(parents=True, exist_ok=True)
        self.zones.save(self._zones_path)
        self.saved = True
        # 通知外面（面板据此更新提示）—— 非模态下调用方拿不到返回值
        self.saved_now.emit(self.map_id, len(self.zones.sets))
        return self._zones_path, probs

    # ---------------- 界面 ----------------

    def _panel(self, inner, min_w):
        """把一栏的控件装进 `QScrollArea` ⇒ 返回那个滚动区（给分栏用）。

        docs/UI规范.md §4：**长面板进滚动区**。三栏各一个滚动区，好处有两条：
          · 窗口矮下来时**各栏自己出滚动条**，而不是把按钮压没（看不见的按钮没法点 ✗）；
          · 滚动区是**可压缩**的 ⇒ 窗口的最小高度只由画布那栏决定
            （地板只许来自当前布局，见 `_drop_stale_min_height`）。
        `min_w` 是这一栏的**最小宽度**（分栏拖到最窄就停在这）：它由**人**定、不是由内容
        定 —— 内容需要的宽度（集合栏 ≈256、可达栏 ≈276，探针量过）比它能拖到的最窄要宽，
        所以拖到 160/170 时**栏里出横向滚动条**（而不是把按钮切掉看不见 ✗）。
        为什么必须自己定小一点：`QSplitter` 的最小宽度 = **各栏最小宽度之和**，三栏最小
        宽度加起来要留在对话框的 620 以内（用例 `t_window_can_shrink_vertically` 钉着
        `minimumWidth() <= 620`）；按内容算就会到 770+ ✗。
        """
        # 滚动区**只有一处实现**：`gui.widgets.scroll_area`（2026-09-27 收口）——
        # 这里是唯一"滚动区自己要当控件交出去"的场景（它是 `QSplitter` 的一栏 ✓），
        # 所以直接拿它，而不是 `scroll_page`（那个要一个容器 widget ✓）。
        # 纵向按需；横向**也按需**（拖窄了要给得出滚动条，见上面的 `min_w` 说明）
        return scroll_area(inner, h_scroll=True, min_width=min_w)

    def _row_tag(self, text):
        """表单行的**行首小标签**（灰、右对齐）—— 说明"这一行在干什么"。"""
        lb = QLabel(text)
        lb.setStyleSheet("color: #5f6368;")
        lb.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lb

    def _build(self):
        # 最上面一条**常驻**的顶栏：放**文档级**动作 —— 保存 / 撤销 / 重做。
        # 为什么搬上来：保存原来在右栏**最底下**、而右栏是滚动区 ⇒ 窗口一矮、或者往下滚
        # 两下，这个最要紧的按钮就**看不见**了 ✗（"找不到保存"比任何交互细节都糟）；
        # 顶栏在滚动区**外面** ⇒ 永远看得见 ✓。
        # 撤销/重做 **2026-09-26 布局重构时一起搬上来**：它们和「保存」是**一类**
        # （对整个文档动手，不属于"集合"也不属于"可达"任何一栏），原来挤在右栏末尾
        # 既不好找、又白占三栏本就不多的纵向空间（用户："信息全挤在一起"）。
        outer = QVBoxLayout(self)
        # ⚠ 顶栏要**够矮**：它长在滚动区外面 ⇒ 高度会直接顶起窗口的最小高度，
        # 而本窗口有一条既要满足的要求 —— 窗口能一路缩到 **240** 高（用例钉着：
        # `t_dialog_background_breathing_and_size` 会 resize 到 240 并断言不被顶回去 ✗）。
        # 所以这里零边距、零间距、按钮内边距也压到最小，只留醒目配色 ✓；
        # 几个按钮**横着排**（横排不加高度 ✓）。
        outer.setSpacing(2)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        self.btn_save = QPushButton("保存")
        self.btn_save.setObjectName("saveZones")
        # 换色：深绿底 + 白字（本仓库的主操作/成功色 —— 和"跳"、"已保存"同一档绿）
        self.btn_save.setStyleSheet(
            "QPushButton { background: #188038; color: #ffffff;"
            " border: 1px solid #0d652d; border-radius: 6px;"
            " padding: 2px 18px; font-weight: 600; }"
            "QPushButton:hover { background: #146c2e; }"
            "QPushButton:pressed { background: #0d652d; }")
        self.btn_save.setToolTip("写入 %s（快捷键 Ctrl+S）" % self._zones_path.name)
        self.btn_save.clicked.connect(self._on_save)
        top.addWidget(self.btn_save)
        # 撤销 / 重做（和保存同一类：文档级动作；快捷键 Ctrl+Z / Ctrl+Y 在 keyPressEvent）
        self.btn_undo = QPushButton("撤销")
        self.btn_undo.setToolTip(
            "回退上一步改动（Ctrl+Z）。\n\n"
            "放在顶栏：它和「保存」一样是**文档级**动作 —— 既不属于「集合」那一栏，\n"
            "也不属于「可达」那一栏。")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo = QPushButton("重做")
        self.btn_redo.setToolTip("重做刚被撤销的那一步（Ctrl+Y）。")
        self.btn_redo.clicked.connect(self.redo)
        top.addWidget(self.btn_undo)
        top.addWidget(self.btn_redo)
        top.addStretch(1)
        outer.addLayout(top)

        # ---- 三栏：画布 ｜ 集合 ｜ 可达·可被到达 ----
        # 2026-09-26 用户要求（原话："现在感觉信息全挤在一起，也不能缩放调整，希望将集合
        # 编辑与可达、可被到达的布局分列，以省去滑动过程"）：
        #   · **分列**：原来是**一根**竖排滚动条 —— 集合在上面、可达 / 可被到达堆在下面
        #     ⇒ 窗口一矮就得上下滚着对照（"我选中这个集合，它能去哪儿 / 谁来过"本来就该
        #     同屏 ✗）。现在「集合」一栏、「可达 / 可被到达」一栏，各自独立滚动 ✓。
        #   · **能缩放调整**：用 `QSplitter` —— 栏宽可以用鼠标拖（原来右栏钉死 250~420 宽，
        #     画布和它之间没有分隔条，用户只能忍着 ✗）。
        # 分工按"手上正在改什么"分：画布 = 看与选，集合 = 圈范围，关系 = 连边。
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(6)               # 拖得动、又不吃掉太多栏宽
        # 不许把某一栏拖成 0 宽：拖没了得靠拖回来才发现，而且那栏里的按钮就点不到了 ✗
        split.setChildrenCollapsible(False)

        # 画布那栏：**不套滚动区** —— 视图自己会缩放 / 平移（滚轮 / 中键）
        col_view = QWidget()
        left = QVBoxLayout(col_view)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)
        self.scene = QGraphicsScene(self)
        self.view = _ZoneView(self)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.picked.connect(self._on_picked)
        self.view.hovered.connect(self._on_hover)
        # **别把窗口顶大**：视图原来 setMinimumWidth(700)，和右边面板一加，
        # 对话框的最小尺寸反过来推着窗口长 —— 用户看到的就是"点个集合窗口变大了"。
        # 视图只要最小尺寸很小 + 可伸缩，聚焦（fitInView）就纯粹是视口内的事。
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 画布的最小尺寸压小：它是"窗口能缩多矮"的实际决定者（另外两栏都在滚动区里）。
        # 宽 200（原 240）：三栏的最小宽度之和必须留在 620 以内，否则
        # `t_window_can_shrink_vertically` 那条「残留最小宽度清掉后不许超过 620」会红 ✗
        #（200 + 170 + 190 + 分隔条与边距 ≈ 590 ✓）。
        self.view.setMinimumSize(200, 120)
        # 线条宽度来自「设置 → foothold 编辑器线条宽度」（config/ui.yaml）
        self.view.set_line_width(theme.load_foothold_width())
        left.addWidget(self.view, 1)

        # 底图透明度：实时改 `_bg_item` 的 opacity，**不重建场景**（重建会打断选择
        # 与呼吸高亮，滑条拖起来会一顿一顿的）。
        row_bg = QHBoxLayout()
        row_bg.setSpacing(6)
        row_bg.addWidget(QLabel("底图透明度"))
        self.sl_bg = NoWheelSlider(Qt.Horizontal)
        self.sl_bg.setRange(0, 100)
        self.sl_bg.setValue(BG_OPACITY_DEF)
        # ⚠ 用「最小 + 最大」而不是 `setFixedWidth(160)`（2026-09-26 三栏重构）：
        # 固定宽度会把**这一行**的最小宽度钉成 160+标签+百分数 ≈ 300，而画布那栏的最小
        # 宽度又直接决定整个对话框的最小宽度（探针实测：这一处就把对话框顶到 702，
        # 而 `t_window_can_shrink_vertically` 要求 ≤ 620 ✗）。
        # 现在：宽的时候还是 160（和以前一样），栏被拖窄时滑条跟着缩到 96 ✓。
        self.sl_bg.setMinimumWidth(96)
        self.sl_bg.setMaximumWidth(160)
        self.sl_bg.setToolTip(
            "小地图底图的不透明度（0~100%%）。\n\n"
            "底图是深蓝的示意图、线是亮绿：调低 = 底图更淡、线段更清楚；\n"
            "调高 = 地形看得更全，但绿线容易被蓝色块吃掉。\n"
            "0 = 完全隐藏底图（只看几何）。")
        self.sl_bg.valueChanged.connect(self._on_bg_opacity)
        row_bg.addWidget(self.sl_bg)
        self.lbl_bg = QLabel("%d%%" % BG_OPACITY_DEF)
        self.lbl_bg.setStyleSheet("color: #5f6368;")
        row_bg.addWidget(self.lbl_bg)
        row_bg.addStretch(1)
        left.addLayout(row_bg)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #5f6368;")
        self.lbl_status.setWordWrap(True)      # 长文字换行，不撑宽窗口
        left.addWidget(self.lbl_status)
        hint = QLabel("滚轮=缩放　中键拖=平移　双击=适应　｜　左键点=选　拖=框选"
                      "（左→右=包含，右→左=相交）　Shift/Ctrl=加选　Alt=减选　"
                      "Ctrl+A 全选　Esc 清空　Ctrl+Z/Y 撤销重做　｜　"
                      "绳子上的 L1/L2… 就是「爬」那条边里写的**绳号**（同一套编号）")
        hint.setStyleSheet("color: #80868b;")
        hint.setWordWrap(True)
        left.addWidget(hint)
        split.addWidget(col_view)             # 第 1 栏：画布

        # 第 2 栏「集合」：**只放集合自己的事**（列表 + 建 / 改 / 删）。
        # 标题交给 QGroupBox（白底圆角 + 灰标题由主窗口的全局 QSS 给，见 gui/main_window.py）
        # —— 三栏各自一个分组框，一眼看出"这一栏管什么"，比原来那三行裸 QLabel 标题清楚 ✓。
        col_sets = QWidget()
        cs = QVBoxLayout(col_sets)
        cs.setContentsMargins(0, 0, 0, 0)
        cs.setSpacing(6)
        grp_sets = QGroupBox("集合")
        cs.addWidget(grp_sets, 1)
        right = QVBoxLayout(grp_sets)         # ⚠ 本节里 `right` = 「集合」组**内部**的布局
        # 三个列表共用一份委托：只给"集合名"上色，其它字保持正常色（见 _RichRowDelegate）
        self._rich = _RichRowDelegate(self)
        self.lst = QListWidget()
        self.lst.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst.setItemDelegate(self._rich)
        self.lst.currentItemChanged.connect(lambda *_: self._on_pick_set())
        self.lst.itemDoubleClicked.connect(lambda *_: self._on_rename())
        self.lst.setToolTip(
            "点一个集合 = 在图上**高亮它圈住的地形**（视图居中到它，**不缩放**）。\n\n"
            "注意：点集合会**清掉视窗里正选着的 foothold**（一次只编辑一样东西 ——\n"
            "两边同时亮着时集合自己的 foothold 也在呼吸，根本分不出哪个是「已选中」）。\n"
            "要在某个集合上继续加选：先点集合，再**按住 Shift** 在视窗里点选/框选。")
        # 给列表一个像样的最小高度：滚动区里它们会按内容伸缩，太矮就连一行都看不全
        self.lst.setMinimumHeight(110)
        right.addWidget(self.lst, 1)

        # 按钮按**功能**分三行（2026-09-26 布局重构）：
        #   新建 = 造一个新集合；成员 = 往集合里加 / 减 foothold；管理 = 改名 / 删除。
        # 原来五个按钮一字排开往下叠 —— 既看不出"谁和谁是一件事"，又白占纵向 ✗。
        # ⚠ 用 QGridLayout（docs/UI规范.md §4：多列用 QGridLayout，别靠固定宽度对齐）。
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)          # 按钮那一列吃掉宽度，标签列只占需要的

        def _btn(text, slot, tip):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            return b

        self.btn_register = _btn(
            "注册为集合…", self._on_register,
            "把当前选中的 foothold 注册成一个命名集合（按地图存 id 列表）。")
        grid.addWidget(self._row_tag("新建"), 0, 0)
        grid.addWidget(self.btn_register, 0, 1)

        # 成员那一行：加 / 减 是一件事的两个方向 ⇒ 并排，一眼看出是一对
        self.btn_add_to = _btn(
            "加入集合…", self._on_add_to,
            "把当前选中的 foothold 加入某个集合（弹窗默认当前高亮的那个）。\n\n"
            "两种顺序都行：① 先在视窗里选 foothold → 这里选集合；\n"
            "② 先点集合 → **按住 Shift** 在视窗里加选 → 这里选集合。")
        self.btn_remove = _btn(
            "移出集合…", self._on_remove_from,
            "把当前选中的 foothold 从某个集合里移出（弹窗默认当前高亮的那个）。\n"
            "顺序同「加入集合…」。")
        row_members = QHBoxLayout()
        row_members.setSpacing(6)
        row_members.addWidget(self.btn_add_to)
        row_members.addWidget(self.btn_remove)
        grid.addWidget(self._row_tag("成员"), 1, 0)
        grid.addLayout(row_members, 1, 1)

        # 管理那一行：改名 / 删除 —— 动的是"集合"这个东西本身，和成员编辑不是一回事
        self.btn_rename = _btn("改名…", self._on_rename,
                               "改集合名（引用它的地方会一起改）。")
        self.btn_delete = _btn("删除", self._on_delete,
                               "删掉这个集合（**会二次确认**）。")
        row_manage = QHBoxLayout()
        row_manage.setSpacing(6)
        row_manage.addWidget(self.btn_rename)
        row_manage.addWidget(self.btn_delete)
        grid.addWidget(self._row_tag("管理"), 2, 0)
        grid.addLayout(row_manage, 2, 1)
        right.addLayout(grid)

        # ---- 第 3 栏：可到达 / 可被到达 ----
        # ⚠ 「集合」栏到上面为止 —— 两栏都建好之后**一起**装滚动区、一起进分栏（见本节末尾）
        #   ⇒「三栏是什么、什么顺序、各占多宽」集中在一处，不用来回找 ✓。
        # 为什么和「集合」分成两栏（见上面 split 的说明）：选中一个集合之后，"它能去哪儿"
        # 和"谁来过"就是**紧接着要看的下一件事** —— 和集合列表放同屏才叫对照 ✓。
        # 叫「可到达」是因为"边"是图论词：这一栏回答的是"从哪个平台能到哪个平台"。
        # 有向（§12.3 B5）：**出去的一栏可编辑、进来的一栏只读**（见 in_host）。
        col_edges = QWidget()
        ce = QVBoxLayout(col_edges)
        ce.setContentsMargins(0, 0, 0, 0)
        ce.setSpacing(6)
        grp_e = QGroupBox("可到达 · 从这个集合出去")
        grp_e.setToolTip(
            "从**当前在编辑的集合**出去的边（进来的边在下面「可被到达」栏）。\n"
            "没选中任何集合时列全部边 —— 那是「总览」的位置 ✓。\n"
            "类型：走 / 爬（绳梯）/ 跳 / 下跳 / 传送门。")
        ce.addWidget(grp_e, 2)            # 这一组比「可被到达」多，所以分得多一点
        right = QVBoxLayout(grp_e)        # ⚠ 本节里 `right` = 「可到达」组**内部**的布局
        right.setSpacing(6)
        # 有没有按"当前在编辑的集合"收窄（收窄时必须**说出来**：静悄悄少几条最难查）
        self.lbl_e_note = QLabel("")
        self.lbl_e_note.setStyleSheet("color: #5f6368;")
        self.lbl_e_note.setWordWrap(True)
        right.addWidget(self.lbl_e_note)
        self.lst_e = QListWidget()
        self.lst_e.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst_e.setItemDelegate(self._rich)      # 行内分色（见 _RichRowDelegate）
        self.lst_e.setMinimumHeight(110)
        self.lst_e.setToolTip(
            "**从当前在编辑的集合出去**的边（别的边在下面「可被到达」栏）。\n"
            "没选中任何集合时列**全部边**（总览）。\n"
            "类型：走 / 爬（绳梯）/ 跳 / 下跳 / 传送门。\n"
            "画布上：连线一律**黑色细虚线**，**箭头三角**按类型着色 ——\n"
            "绿=走、蓝=爬、橙=跳、紫=传送门。\n\n"
            "「反向」是给**选中的这条**加一条反方向的边；它属于对面那个集合，\n"
            "所以不会在这一栏里冒出来（想找它：选中对面那个集合）。\n\n"
            "**边是人确认过的**：自动只产出建议（点「建议…」看），不会自己变成可用边 ——\n"
            "毕竟「能不能过去」是玩法知识，判据只能给候选。\n\n"
            "红色的那几条是**悬空边**（指向已删除或改名的集合）—— 双击它就能把终点"
            "改成对的（或删掉）。\n\n"
            "选中一行 ⇒ 画布上**对应的那条箭头会呼吸**（一眼认出改的是哪条）；\n"
            "**双击一行 = 改这条边**（终点 / 类型 / 绳 / 门；起点固定是当前集合）。")
        # 选中一行 ⇒ 画布上那条箭头呼吸高亮（两栏共用同一个槽，见 _on_pick_edge）
        self.lst_e.currentItemChanged.connect(lambda *_: self._on_pick_edge(self.lst_e))
        # 双击一行 ⇒ 改这条边（和集合那栏"双击改名"是同一个手势）
        self.lst_e.itemDoubleClicked.connect(self._on_edit_edge)
        right.addWidget(self.lst_e, 1)

        row_e = QHBoxLayout()
        row_e.setSpacing(6)
        for text, slot, tip in (
                ("增加可达", self._on_add_edge,
                 "加一条「从哪儿能到哪儿」：**起点 = 当前高亮的集合**，然后选终点、选类型。\n\n"
                 "终点**默认只列与起点有关的**（建议涉及的 + 已经连着的）—— 集合一多，\n"
                 "全列出来根本找不着；想选别的就挑最后那项「显示全部集合…」。\n\n"
                 "爬（climb）还要选**哪根绳**、传送门还要选**哪个门** ——\n"
                 "这两类缺了就说不清**怎么过去**（保存时校验会拦）。"),
                ("反向", self._on_rev_edge,
                 "把选中的那条复制成反方向。图是**有向**的：爬上去和爬下来要两条。"),
                ("删除", self._on_del_edge, "删掉选中的那条（可撤销）。")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row_e.addWidget(b)
        right.addLayout(row_e)

        self.btn_sug = QPushButton("建议…")
        self.btn_sug.setToolTip(
            "按数据算出**候选边**，由你逐条确认（自动只产建议，不会自己变成边）：\n"
            "  · 走（walk）：两组地形严丝合缝（§12.2 的 Δy=0 且 gap=0）；\n"
            "  · 爬（climb）：同一根绳两端各有一个集合；\n"
            "  · 「另一端还没圈」也会列出来 —— 那不是边，是提示你**该圈哪块地形**。\n\n"
            "⚠ **传送点（portal）给不出信任的建议**，要自己用「增加可达」加 ✗ ——\n"
            "  「哪扇门通向哪儿」是导出数据里的 `tn`，但「值不值得用它」是玩法判断，\n"
            "  而且**跨图的门明确不支持**（用户 2026-10-06 定）✓ 所以不自动产候选 ✓。")
        self.btn_sug.clicked.connect(self._on_suggest)
        right.addWidget(self.btn_sug)

        # ---- 可被到达（进来的边；**只读**，2026-09-26 要求 2）----
        # 和「可到达」各占一个分组框：选中 A 时"我从哪儿来"和"A 能去哪儿"是两件事，
        # 混在一栏里谁都得逐行认方向。
        # 这里**不放编辑按钮** —— 要改就把**起点**那个集合选中，它的"可到达"里就有这条边
        #（一次只编辑一样东西，见 _refresh_edges）。
        # 整段套一个控件：用例据此断言"这一栏里一个按钮都没有"（`t_reach_ui_filter_and_style`）。
        self.in_host = QGroupBox("可被到达 · 进入这个集合")
        self.in_host.setToolTip(
            "哪几个集合**能到**当前在编辑的这个集合（进入它的边）。\n\n"
            "这一栏**纯看**：要改就先把**起点**那个集合选中，它的「可到达」里就有这条边。")
        lay_in = QVBoxLayout(self.in_host)
        lay_in.setSpacing(6)
        self.lbl_in_note = QLabel("")
        self.lbl_in_note.setStyleSheet("color: #5f6368;")
        self.lbl_in_note.setWordWrap(True)
        lay_in.addWidget(self.lbl_in_note)
        self.lst_in = QListWidget()
        self.lst_in.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst_in.setItemDelegate(self._rich)
        self.lst_in.setMinimumHeight(90)
        self.lst_in.setToolTip(
            "哪几个集合**能到**当前在编辑的这个集合（进入它的边）。\n\n"
            "这里**纯看**：没有编辑按钮、双击也不进编辑 —— 要改就把**起点**那个集合"
            "选中，\n它的「可到达」里就有这条边（一次只编辑一样东西）。\n\n"
            "画布上按类型着色的是**箭头**（绿=走、蓝=爬、橙=跳、紫=传送门），\n"
            "连线一律是黑细虚线。")
        # 这一栏里选中一行，画布上那条箭头也呼吸（纯看也能定位到是哪条）
        self.lst_in.currentItemChanged.connect(lambda *_: self._on_pick_edge(self.lst_in))
        lay_in.addWidget(self.lst_in)
        ce.addWidget(self.in_host, 1)

        # 两栏面板装进滚动区，按「画布 → 集合 → 关系」的顺序进分栏
        #（这个顺序 = 干活的顺序：先看图、再圈范围、最后连边 ✓）
        self.area_sets = self._panel(col_sets, 160)      # 留着：用例断言"进滚动区了"
        self.area_edges = self._panel(col_edges, 170)    # 同上
        split.addWidget(self.area_sets)
        split.addWidget(self.area_edges)
        # 只有**画布**那栏跟着窗口长（另外两栏是"工具面板"，宽度由人拖）
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setStretchFactor(2, 0)
        # 初始宽度：**按各栏内容的实际需要给**（探针量的：集合栏内容 280、可达栏的边列表
        # 329~346 —— 后者随集合名的长短变）⇒ 开局就不该出现横向滚动条 ✓
        #（用户拖窄了才出，那是他自己的选择）。画布那栏仍然最大，而且窗口变大时
        # **只有它跟着长**（见上面的 stretchFactor）✓。
        split.setSizes([480, 290, 390])
        self.split = split                      # 留着：用例断言"栏宽真的能拖"
        outer.addWidget(split, 1)               # 三栏塞进「顶栏 + 内容」这个外层
        # ⚠ 「保存 / 撤销 / 重做」都在**最上面的顶栏**（见 `_build` 开头）—— 这里不再放一份：
        #   它们原来在右栏最底下、而右栏是滚动区 ⇒ 窗口一矮、或者往下滚两下就看不见 ✗。

        # 呼吸高亮的节拍器：只改笔刷，不重排场景（见 _on_pulse）
        self.timer = QTimer(self)
        self.timer.setInterval(PULSE_MS)
        self.timer.timeout.connect(self._on_pulse)
        self.timer.start()
        # 快捷键统一在 keyPressEvent 里处理（与上面那行提示严格一致）；
        # 不额外挂 QShortcut —— 两处写容易漂，而且藏在按钮上的快捷键没人找得到。

    def keyPressEvent(self, e):
        if e.modifiers() & Qt.ControlModifier and e.key() == Qt.Key_A:
            self.select([str(f.fid) for f in self.terrain.footholds if not f.is_wall])
            return e.accept()
        if e.modifiers() & Qt.ControlModifier and e.key() == Qt.Key_Z:
            self.undo()
            return e.accept()
        if e.modifiers() & Qt.ControlModifier and e.key() == Qt.Key_Y:
            self.redo()
            return e.accept()
        if e.modifiers() & Qt.ControlModifier and e.key() == Qt.Key_S:
            self._on_save()
            return e.accept()
        if e.key() == Qt.Key_Escape:
            self.select([])
            return e.accept()
        super().keyPressEvent(e)

    # ---------------- 画 ----------------

    def _add_background(self):
        """把**小地图底图**铺在几何下面 —— 编辑器看起来就是那张地形图，而不是一堆线。

        换算与 `tools/map_terrain_view.render` 的 `P()` **同一套**（换算是底线：
        各写一份必然差几像素，而"线压在地形上"正是靠这几像素判断对错的）：
            图像像素 = (世界坐标 + centerX/Y) / px_per_world
        ⇒ 图 (0,0) 对应世界 (-centerX, -centerY)，一个图像像素 = px_per_world 个世界像素。
        """
        name = (self.terrain.mini or {}).get("canvas")
        if not name:
            return None
        path = mapdata.map_dir() / name
        if not path.exists():
            return None
        try:
            import cv2

            from core.imgio import imread          # 走仓库的读图（中文路径安全）
            # ⭐ **带 alpha 读**（用户 2026-09-27："小地图底图…的背景应该透明吧？你自己加的黑色？"）
            #   —— 底图 PNG 有近一半像素是全透明的（面板之外的圆角 ✓，四角 alpha=0 ✓）；
            #   原来按 `IMREAD_COLOR` 读 ⇒ 透明区变**纯黑** ✗，看起来就像"给底图加了个黑底" ✗。
            img = imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:                          # noqa: BLE001
            return None
        if img is None or getattr(img, "size", 0) == 0:
            return None
        h, w = img.shape[:2]
        # ⚠ 带 alpha 的底图（BGRA ✓）必须走 `Format_ARGB32`（内存序就是 BGRA ✓）；否则透明区变黑 ✗
        _fmt = (QImage.Format_ARGB32
                if (getattr(img, "ndim", 0) == 3 and img.shape[2] == 4)
                else QImage.Format_BGR888)
        qimg = QImage(img.tobytes(), w, h, img.strides[0], _fmt)
        item = QGraphicsPixmapItem(QPixmap.fromImage(qimg.copy()))
        # 最近邻放大：底图本来只有一百多像素宽，插值只会把它糊成一片
        item.setTransformationMode(Qt.FastTransformation)
        ox = float(self.terrain.mini.get("centerX") or 0)
        oy = float(self.terrain.mini.get("centerY") or 0)
        k = float(self.terrain.px_per_world or 16.0)
        item.setPos(-ox, -oy)
        item.setScale(k)
        item.setZValue(-10)
        # 按滑条当前值压暗（默认 0.6 = `tools/map_terrain_view` 那边同样的亮度；
        # 不压的话底图那些亮蓝色块会把线条吃掉）。
        item.setOpacity(self._bg_opacity)
        self._bg_item = item
        return item

    def _on_bg_opacity(self, val):
        """滑条动了 → 只改底图那一个 item 的透明度 + 更新百分比文字。"""
        self._bg_opacity = max(0.0, min(1.0, float(val) / 100.0))
        self.lbl_bg.setText("%d%%" % int(val))
        if self._bg_item is not None:
            self._bg_item.setOpacity(self._bg_opacity)

    def _rebuild_scene(self):
        scene = self.scene
        scene.clear()
        self._items = {}
        self._ladder_items = []
        self._hl_items = []            # 呼吸高亮的对象（选中的 + 选中集合涉及的）
        # view 侧的线条清单（线宽随缩放统一重写，见 _ZoneView._apply_widths）：
        # 第 4 项 = 笔归呼吸高亮管（宽度由 _on_pulse 每 60ms 写，这里别插手）
        lines = self.view._lines = []
        pick = PICK_PX / max(1e-6, self.view.transform().m11())
        in_hl = set()
        # **临时预览**（「增加可达」里选中"可下跳 foothold"那一行，2026-09-26 要求）：
        # 取消**外面所有**呼吸高亮，只让那一条呼吸 —— 重点在于"看的是这一条"。
        pv = self._preview_fid
        if not pv and self._highlight and self._highlight in self.zones.sets:
            in_hl = set(self.zones.sets[self._highlight]["footholds"])
        # 集合"涉及"的绳梯/传送门：**数据层算**（core/zones.py，与将来做边用同一判据）
        hl_lad = (set() if pv else
                  (set(zones.ladders_of(self.terrain, in_hl)) if in_hl else set()))
        hl_por = (set() if pv else
                  (set(zones.portals_of(self.terrain, in_hl)) if in_hl else set()))

        bg = self._add_background()
        if bg is not None:
            scene.addItem(bg)          # 先铺底图（z=-10），后面画的都在它上面
        for f in self.terrain.footholds:
            it = FootholdItem(f, pick)
            style = Qt.DashLine if f.is_wall else None
            color = C_WALL if f.is_wall else C_FLOOR
            pulsed = False
            if pv:                      # 预览时只亮这一条（其余一律常态色）
                if not f.is_wall and str(f.fid) == pv:
                    color, pulsed = C_SEL, True
                    it.setZValue(5)
                    self._hl_items.append((it, C_SEL))
            elif not f.is_wall and str(f.fid) in self._sel:
                color, pulsed = C_SEL, True
                it.setZValue(5)
                self._hl_items.append((it, C_SEL))
            elif not f.is_wall and str(f.fid) in in_hl:
                color, pulsed = C_IN_SET, True
                it.setZValue(4)
                self._hl_items.append((it, C_IN_SET))
            if pulsed:
                it.setPen(self.view._pen(color))
            scene.addItem(it)
            lines.append((it, color, style, pulsed))
            self._items[str(f.fid)] = it
        # 绳梯编号：和「爬」那条边里写的绳号是同一套（见 core.zones.ladder_ids）
        ladders = self.terrain.ladders
        lids = zones.ladder_ids(self.terrain)
        for L in ladders:
            # 点选着的那根（用户 2026-09-27 ✓）⇒ 用**选中色**（`C_SEL`，和"选中的 foothold"
            # 同一个黄 ✓，一眼能认出"现在看的是这根"）+ 一起呼吸 ✓。
            # ⚠ 颜色必须走 `lines` 那一份（`_apply_widths` 每次缩放都会按它重写笔 ✗）——
            #   所以这里改的是 `color` 变量，而不是画完再 setPen ✓。
            sel = (L is self._sel_ladder)
            color = C_SEL if sel else C_LADDER
            it = scene.addLine(L.x, min(L.y1, L.y2), L.x, max(L.y1, L.y2),
                               self.view._pen(color, style=Qt.DotLine))
            it.setZValue(3 if sel else 1)
            self._ladder_items.append((L, it))
            # 编号标在绳子的**上端右侧**：绳是竖线，标在线上会盖住线；上端是"爬上去
            # 到了哪儿"，和走路的方向一致。编不出来（理论上不会）就不标，别画个空字。
            lid = lids.get(id(L))
            hit = L in hl_lad          # 选中集合时，它的绳子一起呼吸
            if lid:
                # ⚠ 编号**不进 `_hl_items`**：呼吸会把字染成绳子的蓝，而蓝字在这里专指
                # "已注册的平台集合名"（用户 2026-09-26 定的规矩）。绳子本身在闪已经够
                # 定位了 —— 要找的是"哪根绳"，编号只需**读得清**。
                self._text_item(lid, LADDER_PX, C_LADDER_ID,
                                L.x + LADDER_PX * 0.28, min(L.y1, L.y2))
            lines.append((it, color, Qt.DotLine, hit or sel))
            if hit or sel:
                self._hl_items.append((it, color))
        for p in self.terrain.portals:
            # ⭐ 传送点（用户 2026-10-06 ✓ 两条要求都在这儿）：
            #   · **名称一直标出来** ✓（原来只画个圆点，图上根本读不出是哪扇门 ✗）；
            #   · **点选的那扇 ⇒ 画大 + 换选中色 + 呼吸** ✓（照绳梯那套 ✓ 见 `_pick_portal`）。
            #   ⚠ 颜色必须走 `lines` 那一份（`_apply_widths` 每次缩放会按它重写笔 ✗）。
            sel = (p is self._sel_portal)
            color = C_SEL if sel else C_PORTAL
            r = PORTAL_SEL_R if sel else PORTAL_R
            it = scene.addEllipse(p.x - r, p.y - r, r * 2, r * 2,
                                  self.view._pen(color), QBrush(color))
            it.setZValue(7 if sel else 6)
            hit = p in hl_por
            lines.append((it, color, None, hit or sel))
            if hit or sel:
                self._hl_items.append((it, color))
            # 名字标在**右上**（圆点右上角外一点 ✓）：压在圆点/线上会读不清 ✗。
            # ⚠ 名字**不进 `_hl_items`**（同绳梯编号那条理由 ✓）：呼吸会把字染色，而"染色"
            #   在这儿专门表示状态；名字只需要**读得清** ✓。
            if p.pn:
                self._text_item(p.pn, PORTAL_PX, color,
                                p.x + PORTAL_PX * 0.45, p.y - PORTAL_PX * 1.15)
        for lf in self.terrain.life:
            try:
                x, y = float(lf.get("x")), float(lf.get("cy"))
            except (TypeError, ValueError):
                continue
            it = scene.addRect(x - 4, y - 4, 8, 8,
                               self.view._pen(C_LIFE), QBrush(C_LIFE))
            it.setZValue(6)
            lines.append((it, C_LIFE, None, False))
        # ---- 集合名（标在每个集合**包围盒的上边中点**）----
        # 为什么标在包围盒上方而不是质心：质心常常正落在平台上，字会压住线
        # （要求 1：像地形叠加图那样把注册名标上去）。
        self._labels = {}
        for name, s in self.zones.sets.items():
            sp = zones.set_span(self.terrain, s.get("footholds") or [])
            if sp is None:
                continue
            x0, x1, y0, _y1 = sp
            col = QColor(s.get("color") or zones.PALETTE[0])
            txt = self._text_item(name, LABEL_PX, col,
                                  x0 + (x1 - x0) / 2.0, y0 - LABEL_PX * 1.3,
                                  center_x=True)
            self._labels[name] = txt
            # 选中的集合（或**选中了它的成员**）→ 名字一起呼吸高亮（要求 1）
            # （预览的时候不亮：那会儿在看某一条 foothold，名字跟着闪会分心）
            if not pv and (name == self._highlight or (self._sel & set(s["footholds"]))):
                self._hl_items.append((txt, col))
        # ---- 边（箭头）：从起点集合的包围盒顶边中点 → 终点集合的顶边中点 ----
        # 有向图两条反向边会**完全重叠**，所以按名字顺序给一个垂直偏移把两条分开。
        self._arrow_items = []              # 重建后 item 是新的 ⇒ 三个高亮状态一起重记
        for e in self.zones.edges:
            sa = self.zones.sets.get(e.get("from"))
            sb = self.zones.sets.get(e.get("to"))
            if sa is None or sb is None:
                continue                    # 悬空边不画（边列表里标红）
            sp_a = zones.set_span(self.terrain, sa.get("footholds") or [])
            sp_b = zones.set_span(self.terrain, sb.get("footholds") or [])
            if sp_a is None or sp_b is None:
                continue
            zh, col = zones.EDGE_LABELS.get(e.get("kind"), ("?", "#9aa0a6"))
            sign = 1.0 if str(e.get("from")) <= str(e.get("to")) else -1.0
            ln, ar = self._add_arrow(
                QPointF((sp_a[0] + sp_a[1]) / 2.0, sp_a[2]),
                QPointF((sp_b[0] + sp_b[1]) / 2.0, sp_b[2]),
                col, sign=sign, extra=lines)
            # 记下来：列表里选中这条边时，要在画布上把它点亮（见 _on_pulse）。
            # 键是**边对象本身**（同一批 dict，按身份找），不是 (from,to,kind) ——
            # 那两个字段相同的边可以并存（比如同向两条不同绳），键会撞。
            self._arrow_items.append((e, ln, ar, col))
        r = scene.itemsBoundingRect()
        if r.width() >= 1 and r.height() >= 1:
            scene.setSceneRect(r.adjusted(-40, -40, 40, 40))
        # 收尾统一上宽度：新 item 先带 0 宽的笔（cosmetic 1px），这里按设置重写一遍
        self.view._apply_widths()
        self._refresh_status()

    def _add_arrow(self, p0, p1, color, sign=1.0, extra=None):
        """画一条带箭头的边：**黑细虚线** + **按类型着色的箭头**（见 C_ARROW）。

        线宽按**屏幕像素**走，和地形线一套 —— 见 `_ZoneView._pen`。线本身不再用
        类型的颜色（那样画面上四种彩色线谁也认不出是箭头），所以这里只收一个 `color`
        ——它是**箭头三角**的颜色。
        """
        dx, dy = p1.x() - p0.x(), p1.y() - p0.y()
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        ox, oy = -dy / n * 6.0 * sign, dx / n * 6.0 * sign   # 垂直偏移：反向两条不叠
        a = QPointF(p0.x() + ox, p0.y() + oy)
        b = QPointF(p1.x() + ox, p1.y() + oy)
        ln = self.scene.addLine(QLineF(a, b),
                                self.view._pen(C_ARROW, style=Qt.DashLine))
        ln.setZValue(7)
        # 箭头：终点那头一个小三角（长度也按屏幕像素算，缩放时不变形）
        ux, uy = (b.x() - a.x()) / n, (b.y() - a.y()) / n
        L = 18.0 * self.view._px()
        W = 9.0 * self.view._px()
        tip = QPointF(b.x(), b.y())
        tri = QPolygonF([tip,
                         QPointF(b.x() - ux * L - uy * W, b.y() - uy * L + ux * W),
                         QPointF(b.x() - ux * L + uy * W, b.y() - uy * L - ux * W)])
        # 三角**实线**、填类型色：小三角再套一层虚线边就只剩一团噪点
        ar = self.scene.addPolygon(tri, self.view._pen(color, style=Qt.SolidLine),
                                   QBrush(QColor(color)))
        ar.setZValue(7)
        if extra is not None:               # 一并交给线宽那套（缩放时不粗不细）
            extra.append((ln, C_ARROW, Qt.DashLine, False))
            extra.append((ar, color, Qt.SolidLine, False))
        return ln, ar

    def _text_item(self, text, px, color, x, y, center_x=False):
        """画一个**带黑描边**的文字（集合名 / 绳梯编号）→ 返回彩色那一份。

        两个约定：
          · **描边**：底图是深蓝的示意图，亮色字直接压上去读不清 —— 黑色副本偏移
            2px 衬在底下。描边**不参与呼吸**（跟着变色反而糊），所以只进场景；
          · **不接鼠标**：选择是几何判据（见 `_on_picked`），文字不该抢事件。

        返回彩色那份：呼吸高亮只改它（`_on_pulse` 里文字走 `setBrush`）。
        """
        f = QFont(theme.FAMILY)
        f.setPixelSize(px)
        f.setBold(True)
        sh = QGraphicsSimpleTextItem(text)
        sh.setFont(f)
        sh.setBrush(QBrush(QColor(0, 0, 0)))
        txt = QGraphicsSimpleTextItem(text)
        txt.setFont(f)
        # `QColor(color)` 两种写法都吃（C_SETNAME / C_LADDER_ID 是字符串，集合自己的
        # 配色是 QColor）—— 踩过：只传 QBrush(color) 时传字符串会 TypeError 崩掉。
        txt.setBrush(QBrush(QColor(color)))
        if center_x:                        # 以 x 为中心（集合名要落在包围盒上方正中）
            x -= txt.boundingRect().width() / 2.0
        sh.setPos(x + 2, y + 2)
        txt.setPos(x, y)
        for it in (sh, txt):
            it.setZValue(20)
            it.setAcceptedMouseButtons(Qt.NoButton)
            self.scene.addItem(it)
        return txt

    def _restore_edge_hl(self):
        """把箭头恢复成常态画法（黑细虚线 + 类型色三角）。

        取消选中、或改选另一条时必须调 —— 每 60ms 只写"当前选中那条"，不主动恢复的话
        上一条会**一直亮着**（两条都亮就分不出在看哪条了）。
        """
        for _e, ln, ar, col in self._arrow_items:
            ln.setPen(self.view._pen(C_ARROW, style=Qt.DashLine))
            ar.setPen(self.view._pen(col, style=Qt.SolidLine))
            ar.setBrush(QBrush(QColor(col)))

    def _on_pick_edge(self, lst):
        """边列表换了选中项 → 让**画布上那条箭头**呼吸（两个列表共用这一个槽）。

        为什么要有：画布上可以同时有十几条箭头，光看列表里那行字对不上哪条是哪条
        （反向的两条只差一个方向，而且本来就重叠着画）。点亮它才知道"改的是哪条"。

        **一次只亮一条**：在另一个列表里选中时，把这边也清掉 —— 两栏同时"选中"会
        让人以为它们指向同一条。清的时候 blockSignals，免得来回触发。
        """
        it = lst.currentItem()
        edge = it.data(Qt.UserRole) if it is not None else None
        other = self.lst_in if lst is self.lst_e else self.lst_e
        if edge is not None and other.currentItem() is not None:
            other.blockSignals(True)
            other.setCurrentRow(-1)
            other.blockSignals(False)
        self._restore_edge_hl()
        # ⚠ 存下来的是 Qt 给的**副本**（dict 进 UserRole 会转成 QVariantMap，取出来不是
        # 同一个对象）⇒ 后面一律按**值**比较，别用 `is`（踩过：身份比较永远不成立）。
        self._edge_hl = edge
        self.view.viewport().update()

    def _on_pulse(self):
        """呼吸高亮：只改笔刷颜色/线宽，不重排场景（几百个 item 也就毫秒级）。

        选中的 foothold 用**黄色**呼吸；选中某个集合时，**它涉及的全部线条一起呼吸**
        （自己的 foothold + 归它的绳梯 + 归它的传送门）—— 一眼就知道"这个集合到底
        圈住了什么"，不用逐条去认。列表里选中的那条可达，**它的箭头也一起呼吸**。
        """
        if not self._hl_items and self._edge_hl is None:
            return
        self._pulse = (self._pulse + PULSE_STEP) % 1.0
        a = 0.5 - 0.5 * math.cos(self._pulse * 2 * math.pi)      # 0~1
        for it, base in self._hl_items:
            c = QColor(base)
            c.setHsv(c.hue(), c.saturation(), int(150 + 105 * a))
            if isinstance(it, QGraphicsSimpleTextItem):
                # **集合名**也跟着呼吸（要求 1）—— 文字用 brush 上色，它没有 pen，
                # 所以这里必须分开走，否则 setPen 一调就 AttributeError 崩掉。
                # 描边那份是固定黑的，所以压暗到最低时字仍读得清。
                it.setBrush(QBrush(c))
                continue
            # 呼吸时比设置值粗最多 3 px（**屏幕像素**，和缩放无关 —— 原来写死的是
            # 场景单位 1~4，整图 0.35 倍下折算出来不到 1.5 px，几乎看不出在闪）
            it.setPen(self.view._pen(c, extra=3.0 * a))
        # 列表里选中的那条可达：箭头也呼吸。**连线保持虚线**（虚线是它的身份，
        # 呼吸把它变成实线就认不出是箭头了）；颜色走"选中"的黄，比单纯加粗显眼。
        if self._edge_hl is not None:
            c = QColor(C_SEL)
            c.setHsv(c.hue(), c.saturation(), int(150 + 105 * a))
            for e, ln, ar, _col in self._arrow_items:
                if e != self._edge_hl:
                    continue
                ln.setPen(self.view._pen(c, extra=1.0 + 3.0 * a, style=Qt.DashLine))
                ar.setPen(self.view._pen(c, style=Qt.SolidLine))
                ar.setBrush(QBrush(c))
                break
        self.view.viewport().update()

    def reload_line_width(self):
        """重读「设置 → foothold 编辑器线条宽度」并应用到画布。

        为什么要手动调：设置弹窗和编辑器是**两个独立的窗口**，那边改完没人通知这边。
        调用点有两处 —— `showEvent`（重开编辑器）和 `_on_edit_zones`（点「寻路编辑器」
        把已在开着的窗口提到前面时），所以"改完设置 → 点一次那个按钮"就能生效。
        """
        self.view.set_line_width(theme.load_foothold_width())

    def showEvent(self, e):
        """重新显示 → 接回节拍器、重读线条宽度、**清掉残留的纵向地板**。

        三件事都只放在这里，因为非模态下这扇窗会反复开关 —— 而且 `close()` 只是**隐藏**、
        **不销毁对象**（挂在主窗口上复用），所以"上次启动时建的窗口"可能一直活着。
        """
        self.timer.start()
        self.reload_line_width()          # 设置可能在这期间改过（不重读就一直用旧值）
        self._drop_stale_min_height()
        super().showEvent(e)

    def _drop_stale_min_height(self):
        """把**比布局要求更高的**尺寸地板清掉。

        ⚠ 为什么需要这一步（2026-09-26 用户第二次报"纵向还是不能缩放"）：
        `__init__` 里曾经写过 `setMinimumSize(720, 420)`，本意是"允许缩到比较小"，
        实际是给纵向钉了 420 的地板。代码删掉之后**旧窗口照样缩不下去** —— 编辑器是
        非模态单实例，`close()` 只是隐藏、对象还活着（`win._zone_editor`），那个 420
        一直挂在它身上，**只有重启工作台才会消失**（实测：删掉那行之后新建的窗口
        minimumHeight=0、布局只要 222px，而旧窗口仍是 720×420）。

        地板的唯一合法来源是**当前布局**（`minimumSizeHint()`），比它大的都是残留。
        每次显示时对一遍 —— 以后改布局也不会再冒出这种"幽灵地板"。
        宽度同理，但不能低于 `MIN_W`（那是**有意**钉的：再窄按钮上的字就挤没了）。
        """
        auto = self.minimumSizeHint()
        want_w = max(self.MIN_W, auto.width())
        if self.minimumWidth() > want_w:
            self.setMinimumWidth(want_w)
        if self.minimumHeight() > auto.height():
            self.setMinimumHeight(auto.height())

    def closeEvent(self, e):
        """关掉编辑器要**停掉节拍器** —— 留着它会在窗口没了以后继续跑（白烧 CPU）。"""
        self.timer.stop()
        super().closeEvent(e)

    def fit(self):
        r = self.scene.sceneRect()
        if r.width() >= 1:
            self.view.fitInView(r, Qt.KeepAspectRatio)

    def focus_ids(self, ids):
        """把视图**居中**到这些 foothold —— **不改缩放**（缩放由滚轮/双击自己定）。

        为什么只居中不缩放（2026-09-26 要求 2）：原来是 `fitInView` 放大到刚好装下，
        于是 ①"整张图现在在哪一块"这个上下文没了；②**每个集合一个比例**，前后两次
        看的比例对不上，想比两处的距离/大小都没法比。居中同样能把它找出来，
        但不打断手里的比例。要整图按双击（适应窗口）。
        """
        b = ids_bbox(self.terrain, ids)
        if b is None:
            return
        x, y, w, h = b
        self.view.centerOn(QPointF(x + w / 2.0, y + h / 2.0))
        self.view.viewport().update()

    # ---------------- 事件 ----------------

    def _pick_ladder(self, L):
        """点中一根绳梯 ⇒ **只记着它**（状态行出信息 + 画布上那根换选中色并呼吸 ✓）。

        用户 2026-09-27："点选绳梯（为了快速查看它的信息），**不用接逻辑**" ✓ ——
        所以这里**不碰** `_sel` / 集合高亮 / 文件 / 撤销栈 ✗，只刷新显示 ✓。
        """
        if self._sel_ladder is L:
            self._refresh_status()
            return
        self._sel_ladder = L
        self._rebuild_scene()               # 那根绳要换色 + 呼吸 ✓
        self._refresh_status()

    def _clear_ladder_sel(self):
        """取消"点选着的绳梯"（点到别处 / 框选时 ✓）；本来就没选 ⇒ **什么都不做**（别白重建 ✗）。"""
        if self._sel_ladder is None:
            return
        self._sel_ladder = None
        self._rebuild_scene()
        self._refresh_status()

    # ⭐⭐ 传送点的点选（用户 2026-10-06 ✓ 原话："**选中传送点后，要放大传送点的图形呼吸高亮，
    #   不然看不出来**"）—— 与绳梯那套**完全同款**（`_pick_ladder` / `_clear_ladder_sel` ✓）：
    #   只影响显示（状态行 + 画布上那扇门 ✓），**不碰** `_sel` / 集合高亮 / 文件 / 撤销栈 ✗。
    #   ⚠ 两样**互斥**（"一次只亮一样" ✓ 本仓库的老规矩 ✓）：点门 ⇒ 清绳；点绳 ⇒ 清门 ✓。
    def _pick_portal(self, p):
        """点中一扇传送点 ⇒ 记着它（状态行出信息 + 那扇门**画大 + 选中色 + 呼吸** ✓）。"""
        self._sel_ladder = None             # 一次只亮一样（同上 ✓）
        if self._sel_portal is p:
            self._refresh_status()
            return
        self._sel_portal = p
        self._rebuild_scene()               # 那扇门要放大 + 换色 + 呼吸 ✓
        self._refresh_status()

    def _clear_portal_sel(self):
        """取消"点选着的传送点"（点到别处 / 框选时 ✓）；没选 ⇒ **什么都不做**（别白重建 ✗）。"""
        if self._sel_portal is None:
            return
        self._sel_portal = None
        self._rebuild_scene()
        self._refresh_status()

    def _on_picked(self, what, mode):
        """视窗里的主动选择。**replace 模式（没按修饰键）会让出集合高亮**。

        规则（2026-09-26 要求 1）：**一次只编辑一样东西**。
          · 视窗里真的选中了 foothold 且没按修饰键 ⇒ 取消集合高亮；
          · 按住 Shift/Ctrl（加选）或 Alt（减选）⇒ 那是"往当前这批里加减"，
            集合高亮正是他据以加减的参照 ⇒ **保留**（这就是"按住 shift 批量选择"）；
          · 点空白 / 空框选（replace）⇒ 只清空选择，**不动集合高亮** —— 那不是
            "在编辑 foothold"，顺手点一下空白不该把正看着的集合弄丢。

        反向的同一条规则在 `_on_pick_set`（点集合 ⇒ 清掉视窗里的选择）。
        为什么非要互斥：两边同时亮着时"现在在编辑哪个"没法判断 —— 集合自己的
        foothold 也在呼吸，和"已选中"的混在一起完全分不出来。
        """
        if what[0] == "click":
            _t, x, y, tol = what
            # ⭐ **先看绳梯 / 传送点**（用户 2026-09-27 ✓ / 2026-10-06 ✓）：⚠ 只在**没点到
            #   foothold** 时才认它们 —— 绳脚下那两条 foothold 才是这张图的主业（编辑集合 ✓），
            #   别被绳子/门抢走 ✗；而且它们**只用来"看信息"** ✓，不动任何选择 ✓。
            #   ⚠ 顺序：**绳梯在前**（老行为不变 ✓）；门排在它后面（两者很少叠在一起 ✓
            #     真叠在一起时，先点的那个优先 —— 有 `_pick_*` 里的互斥收拾 ✓）。
            if pick_at(self.terrain, x, y, tol) is None:
                L = pick_ladder(self.terrain, x, y, tol)
                if L is not None:
                    self._pick_portal(None)     # 一次只亮一样 ✓
                    self._pick_ladder(L)
                    return
                p = pick_portal(self.terrain, x, y, tol)
                if p is not None:
                    self._pick_portal(p)
                    return
            self._clear_ladder_sel()        # 点到别处 ⇒ 取消"看着的绳梯"（点选语义 ✓）
            self._clear_portal_sel()        # 传送点同理 ✓
            f = pick_at(self.terrain, x, y, tol)
            if f is None:
                if mode == "replace":
                    self.select([])
                return
            if mode == "replace":
                self._clear_set_highlight()
            self.select([str(f.fid)], mode)
            return
        _t, rect, box_mode = what
        self._clear_ladder_sel()            # 框选是明确的编辑动作 ⇒ 取消"看着的绳梯" ✓
        self._clear_portal_sel()            # 传送点同理 ✓
        fs = pick_in_rect(self.terrain, rect, box_mode)
        ids = [str(f.fid) for f in fs]
        if not ids and mode == "replace":
            self.select([])
            return
        if mode == "replace":
            self._clear_set_highlight()
        self.select(ids, mode)

    def preview_foothold(self, fid):
        """**临时只呼吸这一条** foothold（`None` = 恢复正常）。

        谁在用：「增加可达」弹窗里选中"可下跳 foothold"那一行时（2026-09-26 用户要求）。
        为什么要"取消外面所有呼吸"：那是**在看**一条具体的 foothold（它该不该留在下跳
        列表里），而不是在编辑某个集合 —— 集合自己那批一起呼吸就分不清看的是哪条。
        """
        fid = str(fid) if fid is not None else None
        if fid == self._preview_fid:
            return
        self._preview_fid = fid
        self._rebuild_scene()

    def clear_foothold_preview(self):
        """撤回临时预览（弹窗关了 / 取消选中）。"""
        self.preview_foothold(None)

    def _clear_set_highlight(self):
        """取消集合高亮（**列表里那行的选中态一起复位**）。

        为什么必须连列表一起复位：`_highlight` 是唯一事实，列表那行只是它的显示 ——
        只改一边就会出现"列表里那行还亮着、画布上已经不呼吸了"的错位。
        （`_refresh_list` 内部 blockSignals，所以复位不会反弹回 `_on_pick_set`。）
        """
        if self._highlight is None:
            return
        self._highlight = None
        self._refresh_list()

    def _on_hover(self, x, y):
        self._hover = (x, y)
        self._refresh_status()

    def _refresh_status(self):
        """状态行：选中了几条 + **选中那几条的坐标** + 鼠标位置 + 集合数。

        为什么要把坐标写出来（2026-09-26 用户要求）：圈集合、对绳梯、核 `fh id` 全靠
        这几对数 —— 以前只能看鼠标那一格的坐标，选中了哪条、它跨多远、在哪一层，
        都得自己拿眼睛量。多选时一行放不下 ⇒ 面板上给**包围盒**，逐条清单进 tooltip。
        """
        n = len(self._sel)
        hx, hy = getattr(self, "_hover", (0.0, 0.0))
        where, tip = "", ""
        fs = [f for f in self.terrain.footholds if str(f.fid) in self._sel]
        if len(fs) == 1:
            f = fs[0]
            mid = (f.x1 + f.x2) / 2.0
            where = ("　｜　#%s（%s）x %d..%d　长 %d　y=%d"
                     % (f.fid, "墙" if f.is_wall else "地板",
                        round(f.x1), round(f.x2), round(abs(f.x2 - f.x1)),
                        round(f.y_at(mid))))
            tip = ("选中 foothold #%s\n  x %d .. %d（长 %d）\n  y=%d（中点）"
                   "\n  左端 (%d, %d)　右端 (%d, %d)"
                   % (f.fid, round(f.x1), round(f.x2), round(abs(f.x2 - f.x1)),
                      round(f.y_at(mid)),
                      round(f.x1), round(f.y_at(f.x1)),
                      round(f.x2), round(f.y_at(f.x2))))
        elif fs:
            sp = zones.set_span(self.terrain, [str(f.fid) for f in fs])
            if sp:
                where = ("　｜　包围盒 x %d..%d　y %d..%d"
                         % (round(sp[0]), round(sp[1]), round(sp[2]), round(sp[3])))
            tip = "选中 %d 条 foothold：\n%s" % (
                len(fs), "\n".join(
                    "  #%s（%s）x %d..%d　y=%d"
                    % (f.fid, "墙" if f.is_wall else "地板", round(f.x1),
                       round(f.x2), round(f.y_at((f.x1 + f.x2) / 2.0)))
                    for f in fs[:40]))
        # **点选着的绳梯**（用户 2026-09-27："点选绳梯为了快速查看它的信息" ✓）：
        # 把"它的信息"直接摆出来 —— 绳号 / 位置 / **两端各压着哪条 foothold** /
        # 那些 foothold 又属于哪些集合（这几样正是要看的 ✓，纯只读 ✓，改集合还是走 foothold ✓）。
        # ⭐ **点选着的传送点**（用户 2026-10-06 ✓："选中传送点后要放大图形呼吸高亮"＋
        #   他后面那条疑问"h009 是连着 h010 的…为什么写着跨图？"）：把**名字 / 坐标 /
        #   WZ 那两个字段 / 出口是谁**都摆出来 ✓ —— 判据走修好的 `zones.portal_exit`
        #   （**按 `tn` 在本图找** ✓ 不再看 `tm` ✗ 见那儿的说明 ✓）。
        if self._sel_portal is not None:
            _P = self._sel_portal
            _ex = zones.portal_exit(self.terrain, _P.pn)
            where += ("　｜　传送点 %s：x=%d　y=%d" % (_P.pn, round(_P.x), round(_P.y)))
            _pt = ["点选传送点 %s（只读：不动选择集、不写文件 ✓）" % _P.pn,
                   "  x=%d　y=%d　pt=%d（0=出生点 / 1=普通 / 2、7=小地图上的）"
                   % (round(_P.x), round(_P.y), int(getattr(_P, "pt", 0) or 0)),
                   "  WZ 字段：tm=%s　tn=%s" % (getattr(_P, "tm", "?"),
                                              getattr(_P, "tn", "") or "（空）")]
            if _ex is not None:
                _pt.append("  ⭐ **通向**本图的 %s（x=%d　y=%d）" % (_ex.pn, _ex.x, _ex.y))
            elif not str(getattr(_P, "tn", "") or "").strip():
                _pt.append("  ⚠ 没配目标（WZ 里 `tn` 是空的）⇒ 这扇门**没有去向** ✓")
            else:
                _pt.append("  ⚠ 目标门 `%s` **不在本图** ⇒ 跨图（本功能不支持 ✗）"
                           % getattr(_P, "tn", ""))
            tip = (tip + "\n\n" if tip else "") + "\n".join(_pt)
        # **点选着的绳梯**（用户 2026-09-27："点选绳梯为了快速查看它的信息" ✓）：
        # 把"它的信息"直接摆出来 —— 绳号 / 位置 / **两端各压着哪条 foothold** /
        # 那些 foothold 又属于哪些集合（这几样正是要看的 ✓，纯只读 ✓，改集合还是走 foothold ✓）。
        if self._sel_ladder is not None:
            _L = self._sel_ladder
            _lid = zones.ladder_ids(self.terrain).get(id(_L)) or "?"
            where += ("　｜　绳梯 %s：x=%d　y %d..%d（长 %d）"
                      % (_lid, round(_L.x), round(min(_L.y1, _L.y2)),
                         round(max(_L.y1, _L.y2)), round(abs(_L.y2 - _L.y1))))
            _lt = ["点选绳梯 %s（只读：不动选择集、不写文件 ✓）" % _lid,
                   "  x=%d　y %d .. %d（长 %d）"
                   % (round(_L.x), round(min(_L.y1, _L.y2)), round(max(_L.y1, _L.y2)),
                      round(abs(_L.y2 - _L.y1)))]
            for _tag, _e in zip(("上端", "下端"),
                                zones.ladder_ends(self.terrain, _L)):
                if _e is None:
                    _lt.append("  %s：**没压到任何 foothold**（该补数据了）" % _tag)
                    continue
                _sets = self.zones.set_of(str(_e.fid))
                _lt.append("  %s：fh #%s　y=%d　集合：%s"
                           % (_tag, _e.fid, round(_e.y_at((_e.left + _e.right) / 2.0)),
                              "、".join(_sets) if _sets else "（没圈进任何集合）"))
            tip = (tip + "\n\n" if tip else "") + "\n".join(_lt)
        self.lbl_status.setText(
            "已选 %d 条 foothold%s%s　｜　鼠标 (%.0f, %.0f)　｜　集合 %d 个"
            % (n, "（点「注册为集合…」给它起名）" if n else "", where,
               hx, hy, len(self.zones.sets)))
        self.lbl_status.setToolTip(tip)

    # ---------------- 边 ----------------

    def add_edge(self, src, dst, kind, ladder=None, footholds=None,
                 walk_dir=None, mid_dir=None, mid_y=None, why="", portal=None):
        """加一条边（走撤销栈）。失败**不登记撤销点**（见 _commit 的说明）。"""
        snap = self._snapshot()
        e = self.zones.add_edge(src, dst, kind, ladder=ladder,
                                footholds=footholds, walk_dir=walk_dir,
                                mid_dir=mid_dir, mid_y=mid_y, why=why,
                                portal=portal)
        self._commit(snap)
        return e

    def del_edge(self, edge):
        """删掉一条边（按 from/to/kind/**ladder/portal** 匹配，避免拿错对象）。

        ⚠ 绳号也要进判据（2026-09-27）：同一对集合、同一类型、**不同绳**是两条不同的边
          （`二楼 →(爬 L3)→ 三楼` 与 `二楼 →(爬 L7)→ 三楼`）—— 少了这一格，删其中一条
          会把**两条一起删掉** ✗（与 `core.zones.add_edge` 的去重判据必须是同一套 ✓）。
        ⭐ **传送点的门**同理（2026-10-06 ✓）：`h009` 与 `h010` 是两条不同的边 ✓。
        """
        snap = self._snapshot()
        key = self._edge_key(edge)
        self.zones.edges = [e for e in self.zones.edges if self._edge_key(e) != key]
        self._commit(snap)

    @staticmethod
    def _edge_key(e):
        """边的**身份**：`(from, to, kind, ladder, portal)`。

        **一处实现**：`del_edge` / `edit_edge` 的匹配都用它 —— 各写一份必然分叉
        （`core.zones.add_edge` 的去重判据与此同义 ✓）。
        ⭐ `portal` 那一格 2026-09-27 随「传送门」移除过 ✓、**2026-10-06 加回** ✓ ——
        ⚠ 加回时**必须同时进这个身份** ✗：同一对集合、同是传送点、**不同门**（`h009` vs
        `h010`）是**两条不同的边** ✓ —— 少了这一格，删其中一条会把**两条一起删掉** ✗、
        改一条会改到**另一条** ✗（与 `add_edge` 的去重判据同一套 ✓）。
        """
        return (e.get("from"), e.get("to"), e.get("kind"),
                str(e.get("ladder") or ""), str(e.get("portal") or ""))

    def rev_edge(self, edge):
        """反向复制一条边 → 新边（已经有了就返回 None）。"""
        src, dst = edge.get("to"), edge.get("from")
        if src not in self.zones.sets or dst not in self.zones.sets:
            return None
        snap = self._snapshot()
        e = self.zones.add_edge(src, dst, edge.get("kind"),
                                ladder=edge.get("ladder"),
                                # ⭐ 传送点反向复制时**哪扇门也要跟着** ✗ —— 漏了它，
                                #   反向那条边就没指名门 ⇒ 保存时校验会拦（说"没指定哪扇门"）
                                #   而人会觉得"明明是照原样复制的" ✓（2026-10-06 加回 portal 时一起补 ✓）。
                                portal=edge.get("portal"),
                                why=(edge.get("why") or "") + "（反向复制）")
        self._commit(snap)
        return e

    def edit_edge(self, edge, dst, kind, ladder=None, footholds=None,
                  walk_dir=None, mid_dir=None, mid_y=None, portal=None):
        """改一条边（**终点 / 类型 / 绳 / 门**）。失败抛 ValueError。

        按 `_edge_key`（from/to/kind/**ladder/portal**）匹配（与 `del_edge` 同一套）：
        列表里拿到的是 Qt 转过的**副本**，不能按对象身份找。**起点在这里不动** ——
        见 `_on_edit_edge`。

        类型换了要把**不再需要的那一格删掉**：走/跳不该留着上一任的绳号、也不该留着
        上一任的门（2026-10-06 加回 portal 时一起补 ✓）。保存时的校验只认当前类型，
        留着不至于报错，但会在下次改回"爬"/"传送点"时**突然复活**（那是上次的绳/门）。
        """
        snap = self._snapshot()
        key = self._edge_key(edge)
        hit = None
        for e in self.zones.edges:
            if self._edge_key(e) == key:
                hit = e
                break
        if hit is None:
            raise ValueError("那条可达已经不在了（可能被删掉、或被改过名）—— "
                             "刷新一下再看看。")
        if dst not in self.zones.sets:
            raise ValueError("终点集合不存在：%s" % dst)
        if dst == hit.get("from"):
            raise ValueError("终点不能是起点自己（那就成了自环）。")
        for e in self.zones.edges:
            if e is hit:
                continue
            # 「改成和另一条一模一样」才算重复：**绳号/门不同不算**（那是两条不同的边 ✓）
            if (e.get("from") == hit.get("from") and e.get("to") == dst
                    and e.get("kind") == kind
                    and str(e.get("ladder") or "") == str(ladder or "")
                    and str(e.get("portal") or "") == str(portal or "")):
                raise ValueError("已经有一条一样的了（%s → %s，%s%s）—— 不用改。"
                                 % (hit.get("from"), dst,
                                    zones.kind_label(kind),
                                    ("　%s" % (ladder or portal)) if (ladder or portal) else ""))
        hit["to"] = dst
        hit["kind"] = kind
        # ⚠ `portal` 与 `ladder` **同一套写法**：给了就存、没给（None / 空）就**删掉** ✓ ——
        #   类型从「传送点」换成别的时，旧门名必须消失 ✗（不然下次换回"传送点"会**突然复活** ✓）。
        for k, v in (("ladder", ladder), ("portal", portal)):
            if v:
                hit[k] = str(v)
            else:
                hit.pop(k, None)
        # 下跳的「可下跳 foothold」：给了就存，**没给（None）就删掉** —— 删掉 = 回到
        # "起点集合的全部"那个默认（与 `zones.drop_footholds` 同一口径）。
        if footholds:
            hit["footholds"] = [str(x) for x in footholds]
        else:
            hit.pop("footholds", None)
        # 「走」的方向类型：给了就存，**没给（None）就删** = 回到"默认方向"。
        if walk_dir:
            hit["dir"] = str(walk_dir)
        else:
            hit.pop("dir", None)
        # ⭐ 爬的**中途跳下**：给了就存、**没给（None / 空）就删** ✓（= 回到"不中途跳下" ✓
        #   与上面 `dir` 那段同一套写法 ✓）
        if mid_dir:
            hit["mid_dir"] = str(mid_dir)
            if mid_y is not None:
                hit["mid_y"] = float(mid_y)
        else:
            hit.pop("mid_dir", None)
            hit.pop("mid_y", None)
        self._commit(snap)
        return hit

    def _cur_edge(self):
        it = self.lst_e.currentItem()
        return it.data(Qt.UserRole) if it is not None else None

    def _edge_row(self, e, first=True):
        """一条边 → 列表行（**一行仍是一条边**：选中 / 高亮 / 删除都按行 ✓）。

        ⚠ 2026-09-27 用户要求：**同一对集合的多种走法合成一组** —— 组里**第一条写全**
          （`小平台 → 一楼　[跳(jump)]`），后面的行**缩进 + 只写方式和条件**
          （`　　[走(walk)]（仅向左）`）；「可到达」与「可被到达」两栏用**同一套**写法 ✓，
          连参数都只剩一个 `first`（两栏原来各传一次 `fmt`，其实永远是 `%s → %s` ✗）。
        为什么要合并：同一对集合常有好几条边（走 + 跳、两根绳…），原来一个终点铺开三四行、
          每行都要从头读一遍名字 ✗；而**条件**（方向 / 绳号 / 起跳点）原来在列表里
          **看不见**（用户要的"扩展更多信息"就是它 —— 见 `zones.edge_cond` ✓）。
        **只有两个集合名是蓝字**，`→`、类型、条件都是正常黑字（用户 2026-09-26 的要求 ✓）；
        悬空边（指向已删除/改名的集合）那两个名字标红 —— 那种边看着没事、跑起来才会在
        路径里断掉，必须在界面上就扎眼 ✓。
        `DisplayRole` 仍是**纯文本**（tooltip / 自检 / `text()` 从它取），
        分色那版放 `ROLE_HTML` 给 `_RichRowDelegate` 画 ✓。
        """
        kind = e.get("kind") or "?"
        zh, _color = zones.EDGE_LABELS.get(kind, (kind, "#9aa0a6"))
        cond = zones.edge_cond(e)
        tail = "[%s]%s" % (zh, "（%s）" % cond if cond else "")
        bad = (e.get("from") not in self.zones.sets
               or e.get("to") not in self.zones.sets)
        warn = "　⚠ 悬空" if bad else ""
        head = "" if first else EDGE_INDENT          # 同组后续行：缩进、不重复写名字 ✓
        if first:
            text = "%s → %s　%s%s" % (e.get("from"), e.get("to"), tail, warn)
        else:
            text = "%s%s%s" % (head, tail, warn)
        it = QListWidgetItem(text)
        it.setData(Qt.UserRole, e)
        nm_col = "#c5221f" if bad else C_SETNAME          # 悬空⇒红；正常⇒蓝（集合名专属）

        def nm(s):
            return '<span style="color:%s">%s</span>' % (nm_col, escape(str(s)))

        warn_html = ('　<span style="color:#c5221f">⚠ 悬空</span>' if bad else "")
        it.setData(ROLE_HTML,
                   ("%s → %s　%s%s" % (nm(e.get("from")), nm(e.get("to")), tail,
                                       warn_html)) if first
                   else ("%s%s%s" % (head, tail, warn_html)))
        it.setToolTip((e.get("why") or "（没写理由）")
                      + ("\n\n⚠ 悬空边：指向的集合已经不在 —— 删掉它，或把集合名改回来。"
                         if bad else ""))
        return it

    def _refresh_edges(self):
        """填两栏：**可到达**（从焦点集合出去的边）/ **可被到达**（能到它的边）。

        为什么"可到达"只列**出去**的（2026-09-26 要求 1）：这一栏是"我在编辑谁"的
        延伸 —— 选中 A 时它回答的是"A 能去哪儿"。原来把进来的边也一起列，于是点
        「反向」之后，那条反向边（**起点是别人**）会当场多出一行，而它并不属于
        "A 能去哪儿"，看着就像"反向生成了一个不该出现在这儿的东西"。
        ⚠ **2026-09-27 一度改成"有多少列多少"（列全部边、相关的排最前），当天用户判定
          是**误修** ⇒ 已改回收窄**。别再往"列全部"改 ✗ —— 用户要的是"这一栏 = 它能去哪儿"，
          总览看"没选中任何集合"那种状态（那时两栏都列全部 ✓）。改回收窄后，
          加边/改边**要把那一行选出来并滚到可见**（`_focus_edge_row` ✓）——
          不然新加的那条可能不在收窄后的列表里，看着又像"没加进去" ✗。

        进来的边在下面**「可被到达」**栏里看（2026-09-26 要求 2）—— 那一栏
        **没有编辑按钮**：要改就选中那边的集合，它的"可到达"里就有这条边。
        一次只编辑一样东西，和"选择 / 集合高亮互斥"是同一条规矩。
        悬空边永远列在"可到达"栏（它是错误，不管在编辑谁）。
        """
        focus = self._focus_set()
        prev = self._edge_hl            # 记住"正在看的那条边"（重填会把选中项清掉）
        for lst in (self.lst_e, self.lst_in):
            lst.blockSignals(True)
            lst.clear()
        out_edges, in_edges = [], []
        for e in self.zones.edges:
            bad = (e.get("from") not in self.zones.sets
                   or e.get("to") not in self.zones.sets)
            if (not focus) or bad or e.get("from") == focus:
                out_edges.append(e)
            if (not focus) or (not bad and e.get("to") == focus):
                in_edges.append(e)
        # **两栏都自动排序**（2026-09-26 要求）：按"另一端的集合名 → 类型 → 绳/门"。
        # 原来是文件顺序 ⇒ 编辑一次（改终点/改类型）行的位置就乱一次，每次都得重新找。
        # **同一对集合合成一组**（组头写全、后续行缩进简写，见 `_edge_row` ✓）。
        # 分组键就是排序键的头一段（"另一端的集合名"✓）再补上 `(from, to)` ——
        # 排序键本身只保证"同另一端的相邻"，补上这一对才保证**同一对**连续
        # （总览时同一个终点可能有不同起点，不补就会把 A→B 和 C→B 混在一组里 ✗）。
        def _rows(edges, lst):
            pair = None
            for e in sorted(edges, key=lambda x: (edge_row_key(x, focus),
                                                  str(x.get("from") or ""),
                                                  str(x.get("to") or ""))):
                key = (e.get("from"), e.get("to"))
                lst.addItem(self._edge_row(e, first=(key != pair)))
                pair = key

        _rows(out_edges, self.lst_e)
        # 进来的边也写成「起点 → 终点」：这一栏的标题已经说了终点是谁
        # （能到「X」的），行里再写一遍反而绕。
        _rows(in_edges, self.lst_in)
        n_out, n_in = len(out_edges), len(in_edges)
        for lst in (self.lst_e, self.lst_in):
            lst.blockSignals(False)
        # 重填会把选中项丢掉 ⇒ **找回来**。不找回来的话，随便点一下 foothold（也会走到
        # 这里）就把"正在看的那条边"丢了、画布上的呼吸也跟着停，看着就像 bug。
        # 比对用 `==`（两边都是 dict）：列表里那份是 Qt 转换过的副本，`is` 永远不成立。
        hit = None
        if prev is not None:
            for lst in (self.lst_e, self.lst_in):
                for i in range(lst.count()):
                    if lst.item(i).data(Qt.UserRole) == prev:
                        hit = (lst, lst.item(i))
                        break
                if hit is not None:
                    break
        if hit is not None:
            hit[0].setCurrentItem(hit[1])   # 走 _on_pick_edge ⇒ 高亮状态跟着一致
        elif prev is not None:
            # 那条边已经不在了（被删掉、或被收窄挡在外面）⇒ 收掉高亮，
            # 别留一条"看不见的亮线"（列表里没这行，画面上却在闪）。
            self._edge_hl = None
            self._restore_edge_hl()
        # 收窄了就**说出来**（静悄悄少几条是最难查的那种"东西不见了"）
        n_all = len(self.zones.edges)
        if not focus:
            self.lbl_e_note.setText("")
            self.lbl_in_note.setText("")
            return
        if n_out != n_all:
            self.lbl_e_note.setText(
                "只显示从「%s」出去的 %d 条（共 %d 条）—— 取消选中就显示全部。"
                % (focus, n_out, n_all))
        else:
            self.lbl_e_note.setText("从「%s」出去的 %d 条。" % (focus, n_out))
        self.lbl_in_note.setText(
            "能到「%s」的 %d 条%s" % (focus, n_in,
                                    "　（只读：要改就选中上边那个集合）"
                                    if n_in else "　（没有）"))


    def _sug_label(self, s):
        """一条建议 → 列表里那行字。"""
        if s.get("to") is None:
            return "【待圈地形】%s　(x=%.0f, y=%.0f)" % (s["why"], s["x"], s["y"])
        zh = zones.EDGE_LABELS.get(s["kind"], (s["kind"], ""))[0]
        tail = ("　绳 %s" % s["ladder"]) if s.get("ladder") else (
            ("　门 %s" % s["portal"]) if s.get("portal") else "")
        return "%s → %s　[%s]%s" % (s["from"], s["to"], zh, tail)

    def _on_suggest(self):
        """建议队列：算一遍候选边，**由人逐条采纳**（自动只产建议，边不会自动可用）。"""
        # ⚠ 2026-09-27：「传送门」这种通行方式连同 `portal_suggestions()` 一起**移除**了
        # ⇒ 建议只剩两种「走」和「爬」（四种通行方式里，跳/下跳都给不出可信建议 ✓）。
        sug = list(zones.walk_suggestions(self.zones, self.terrain))
        sug += zones.climb_suggestions(self.zones, self.terrain)
        # ⚠ 判"有没有采纳过"的判据要和 `core.zones.add_edge` **同一套**
        #（from/to/kind/ladder）：少了**绳号**，"同一对集合、另一根绳"的建议会被
        # 当成"已经采纳过了"吞掉 ✗（爬边尤其 —— 一张图上两根绳接同一对平台很常见）。
        have = {(e.get("from"), e.get("to"), e.get("kind"),
                 str(e.get("ladder") or "")) for e in self.zones.edges}
        sug = [s for s in sug
               if s.get("to") is None or (s["from"], s["to"], s["kind"],
                                          str(s.get("ladder") or "")) not in have]
        # **收窄**：选中了某个集合（或它的一段 foothold）⇒ 只给与它有关的候选。
        # 收窄后一条都没有就退回全部 —— 那也是"总览"的入口（不然人会以为建议没了）。
        focus = self._focus_set()
        if focus:
            keep = [s for s in sug if focus in (s.get("from"), s.get("to"))]
            if keep:
                sug = keep
            else:
                focus = None
        if not sug:
            QMessageBox.information(
                self, "没有新建议",
                "按现在的集合算，没有还没采纳的候选边。\n\n"
                "没有建议通常意味着**还缺集合**：绳的另一端、平台之间的落点\n"
                "（「待圈地形」那种提示就是）—— 先在视窗里把它们圈出来。")
            return
        while sug:
            items = [self._sug_label(s) for s in sug]
            pick, ok = QInputDialog.getItem(
                self, "建议（采纳一条）",
                ("候选可达（只列与「%s」有关的；选一条采纳，取消 = 结束）："
                 % focus) if focus else
                "候选可达（选一条采纳；取消 = 结束）：", items, 0, False)
            if not ok or not pick:
                return
            s = sug.pop(items.index(pick))
            if s.get("to") is None:
                # 它不是边，是"还差一块地形"的提示：说清坐标，别让人以为程序算错了
                QMessageBox.information(
                    self, "这一条不是边", "%s\n\n先把 (x=%.0f, y=%.0f) 那一带圈成集合，"
                    "这条爬升才接得上。" % (s["why"], s["x"], s["y"]))
                continue
            try:
                self.add_edge(s["from"], s["to"], s["kind"],
                              ladder=s.get("ladder"),
                              why="采纳建议：" + (s.get("why") or ""))
            except ValueError as ex:                    # noqa: BLE001
                QMessageBox.warning(self, "采纳失败", str(ex))

    def _reach_ladder_choices(self, src):
        """「爬哪根绳」的候选 → [(绳号, 界面上那行字)]，给「增加可达」那个弹窗用。

        用 `ladders_touching`（本集合的绳 ∪ **绳端通到本集合**的绳）：这里问的是
        "能爬上去的有哪些"，绳端离平台几十像素的那种也要能选到 ✓。
        ⚠ 这里原来还返回"走哪个门"（传送门）那一串 —— **2026-09-27 随「传送门」一起
          移除**了（`zones.EDGE_KINDS` 现在只有走/爬/下跳/跳 ✓）。
        """
        lids = zones.ladder_ids(self.terrain)
        return [(lids.get(id(L), "?"),
                 "%s  x=%d  y[%d..%d]  %s"
                 % (lids.get(id(L), "?"), L.x, min(L.y1, L.y2), max(L.y1, L.y2),
                    "绳子" if L.l else "梯子"))
                for L in zones.ladders_touching(self.terrain,
                                                self.zones.sets[src]["footholds"])]

    def _all_ladder_choices(self):
        """**本图全部**绳 → `[(绳号, 那行字)]`（用户 2026-10-06 ✓ "加全部勾选" 用 ✓）。

        行里**不**带"有没有关系"的标记 ✓ —— 那件事由弹窗按 `self._lad_rel` 判 ✓
        （一处口径：`AddReachDialog._fill_lad` ✓ 别在两边各写一份"算不算相关" ✗）。
        ⚠ 排序照 `_reach_ladder_choices` 同款（绳号**自然序** ✓ 弹窗那边也会再排一次 ✓）。
        """
        lids = zones.ladder_ids(self.terrain)
        return [(lids.get(id(L), "?"),
                 "%s  x=%d  y[%d..%d]  %s"
                 % (lids.get(id(L), "?"), L.x, min(L.y1, L.y2), max(L.y1, L.y2),
                    "绳子" if L.l else "梯子"))
                for L in (getattr(self.terrain, "ladders", None) or [])]

    def _portal_row(self, p):
        """一个传送点 → 「增加可达」下拉里那行字（2026-10-06 ✓）。

        ⭐ **能走到哪儿就写在行里** ✓（用户那条口径："我们应该能知道这个传送点通向哪里" ✓）：
        同图门写出**出口门名 + 坐标** ✓；跨图 / 落点查不到的**当场说清为什么** ✗
        （别让人选了个跑不通的门，等运行时才发现 ✓）。
        """
        ex = zones.portal_exit(self.terrain, p.pn)
        if ex is not None:
            return ("%s  x=%d y=%d　→ 出口 %s(x=%d y=%d)"
                    % (p.pn, p.x, p.y, ex.pn, ex.x, ex.y))
        if int(getattr(p, "tm", 0) or 0) != 999999999:
            return ("%s  x=%d y=%d　⚠ 跨图（tm=%s）—— **不支持**（只给得到目标地图号）"
                    % (p.pn, p.x, p.y, getattr(p, "tm", "?")))
        return ("%s  x=%d y=%d　⚠ 落点查不到（tn=%r 在本图里没有那扇门）"
                % (p.pn, p.x, p.y, getattr(p, "tn", "") or ""))

    def _reach_portal_choices(self, src):
        """「走哪扇门」的候选 → [(门名, 界面上那行字)]，给「增加可达」那个弹窗用。

        用 `zones.portals_of`（落在**起点集合**包围盒 ±`ATTACH_PAD` 内的门 ✓）：这里问的是
        "从我这一层够得着哪扇门" ✓ —— 够不着的由弹窗的「全部」勾选放开 ✓（同绳那条 ✓）。
        ⚠ 与绳那条**同一套写法**（默认收窄 + 全部放开 + 无关的标出来 ✓）—— 见 `_fill_gate` ✓。
        """
        return [(str(p.pn), self._portal_row(p))
                for p in zones.portals_of(self.terrain,
                                          self.zones.sets[src]["footholds"])]

    def _all_portal_choices(self):
        """**本图全部**门 → `[(门名, 那行字)]`（"全部"勾选用 ✓ 用户 2026-10-06 ✓）。"""
        return [(str(p.pn), self._portal_row(p))
                for p in (getattr(self.terrain, "portals", None) or [])]

    def _open_reach(self, dlg, key):
        """开「增加 / 编辑可达」窗口：**非模态 + 单实例**（2026-09-26 用户要求）。

        非模态：开着它还要**缩放细看**编辑器那张图（判断这条 foothold 在不在集合里、
        这个下跳点该不该留它）—— 模态会把整个工作台锁住，只能关窗重开。
        单实例：两个窗口改同一份文件，谁覆盖谁说不清 ⇒ **同一件事再点一次只把它提到
        前面来**（`key` 相同就复用）；换了一条边（`key` 不同）才把旧的换掉。
        """
        old = getattr(self, "_reach_dlg", None)
        if old is not None and getattr(old, "_reach_key", None) == key:
            old.show()
            old.raise_()
            old.activateWindow()
            dlg.deleteLater()           # 新造的没用上，扔掉（别留着抢焦点）
            return
        if old is not None:
            old.close()
            old.deleteLater()           # 绑的是另一条边/另一个起点 ⇒ 旧的必须消失
        dlg._reach_key = key
        dlg.applied.connect(self._apply_reach)
        dlg.finished.connect(lambda *_: self._on_reach_closed(dlg))
        self._reach_dlg = dlg
        dlg.show()                      # show() 而不是 exec_() —— 非模态
        dlg.raise_()
        dlg.activateWindow()

    def _on_reach_closed(self, dlg):
        """窗口关了就把引用忘掉（不然下次点会去 raise 一个已经没了的窗口）。"""
        if getattr(self, "_reach_dlg", None) is dlg:
            self._reach_dlg = None

    def _apply_reach(self, r):
        """弹窗点了确定（非模态 ⇒ 靠信号回来）→ 真的加/改那条边。

        `exec_()` 那条返回值在非模态下没有了，所以"确定"之后干什么必须挪到这里
        （和 `ZoneEditorDialog.saved_now` 同一个路子）。

        ⚠ **加了没加都要说话**（2026-09-27 用户报"点了增加后无添加项目"）：`add_edge`
          对**完全一样**的边是幂等的（返回老那条、不新增）⇒ 原来界面上一点动静都没有，
          看着就像程序坏了 ✗。这里比对条数：没新增就弹一句说清"已经有这条了、想改绳
          就双击那一行"，并且**把那条选出来**（一眼看见它到底在哪 ✓）。
        """
        dlg = self.sender()
        try:
            if getattr(dlg, "mode", "add") == "edit" and getattr(dlg, "edge", None):
                e = self.edit_edge(dlg.edge, r["dst"], r["kind"], ladder=r["ladder"],
                                   footholds=r.get("footholds"),
                                   walk_dir=r.get("dir"),
                                   mid_dir=r.get("mid_dir"),
                                   mid_y=r.get("mid_y"),
                                   # ⭐ 走哪扇门（2026-10-06 ✓）：和 `ladder` 同一层 —— 漏传
                                   #   就等于"编辑时把门弄丢了"（保存后校验会拦 ✓ 但人会觉得莫名 ✓）。
                                   portal=r.get("portal"))
            else:
                n0 = len(self.zones.edges)
                e = self.add_edge(dlg.src, r["dst"], r["kind"], ladder=r["ladder"],
                                  footholds=r.get("footholds"),
                                  walk_dir=r.get("dir"),
                                  mid_dir=r.get("mid_dir"),
                                  mid_y=r.get("mid_y"),
                                  portal=r.get("portal"),
                                  why="手工加的（%s → %s）" % (dlg.src, r["dst"]))
                if len(self.zones.edges) == n0:
                    extra = str(e.get("ladder") or e.get("portal") or "")
                    QMessageBox.information(
                        self, "这一条已经有了",
                        "「%s → %s」已经有一条「%s」了（%s）—— 所以没有重复加。\n\n"
                        "· 想把绳/门**换成**你刚选的那个：在上面的「可到达」栏里"
                        "**双击那一行**改（起点不会变）；\n"
                        "· 想要**另一根绳**（两条不同的走法）：绳号不一样就会各自加一条 ✓ "
                        "—— 如果刚才选的绳号确实不同却看到这句话，说明列表里那份还没刷新，"
                        "点一下集合或重开这一栏看看。"
                        % (dlg.src, r["dst"], zones.kind_label(r["kind"]),
                           ("绳 " + extra) if r.get("ladder") else
                           (("门 " + extra) if extra else "没有绳/门")))
            self._focus_edge_row(e)     # 加/改完把那一行**选中并滚到可见**（不然在长列表里找不着 ✗）
        except ValueError as ex:                        # noqa: BLE001
            QMessageBox.warning(self, "没加成", str(ex))

    def _focus_edge_row(self, e):
        """把某条边在「可到达」列表里**选中并滚到看得见**。

        为什么需要：那一栏现在列**全部边**（"有多少列多少" ✓），新建/刚改的那条很可能在
        十几行里靠下 ⇒ 不选出来、不滚过去，人只会看到"点了没反应" ✗（2026-09-27 用户
        报的就是这个形状）。选中走既有的 `_edge_hl`（`_refresh_edges` 会按它把那行选回来 ✓）。
        """
        if e is None:
            return
        self._edge_hl = dict(e)
        self._refresh_edges()
        it = self.lst_e.currentItem()
        if it is not None:
            self.lst_e.scrollToItem(it)

    def _drop_choices(self, src):
        """起点集合的 foothold → [(id, 界面上那行字)]，给「可下跳 foothold」那张表用。

        只列**起点集合里的**：下跳的起点必须属于这条边的起点集合，否则"从那儿下去"
        就无从谈起（要加别的地方，先把那块地形圈进这个集合）。
        """
        out, order = [], {}
        for fid in (self.zones.sets.get(src) or {}).get("footholds") or []:
            f = next((x for x in self.terrain.footholds
                      if str(x.fid) == str(fid)), None)
            if f is None:
                # 本图没这条（换过图 / 重导过地形）⇒ 排最后，别混在正常行之间
                order[str(fid)] = (1, 0.0, 0.0)
                out.append((str(fid), "fh %s（本图没有这条）" % fid))
                continue
            y = f.y_at((f.left + f.right) / 2.0)
            order[str(fid)] = (0, y, f.left)
            out.append((str(fid), "fh %s　y=%d　x %d..%d"
                        % (fid, y, f.left, f.right)))
        # **按地图上的上下顺序排**（y 小的在上，同高再按 x）—— 这张表是拿来"挑一个能
        # 下去的点"，按画面顺序排最直观（2026-09-26 要求"自动 sort"）；原来按 foothold
        # id 排，和画面上的位置对不上，得逐行读坐标去找。
        out.sort(key=lambda t: order.get(str(t[0]), (1, 0.0, 0.0)))
        return out

    def _on_add_edge(self):
        """增加一条可达：**一个弹窗问完**（终点 / 通行方式 / 绳或门）。

        为什么不像以前那样串 2~4 个 QInputDialog：每弹一个都得重新回想"上一步选了啥"，
        选错只能整个退出重来；而这几个参数本来就该一起看（选「爬」才知道要挑绳、
        选「传送门」才知道要挑门）。弹窗那边按类型显隐那两行，见 AddReachDialog。
        """
        # 起点：高亮的集合 > 选中的 foothold 所属的那个集合（"我在编辑谁"就是起点）
        src = self._focus_set() or self._cur_set()
        if not src:
            QMessageBox.information(self, "先选起点",
                                    "先在右边列表里点一个集合（或在视窗里选一段 "
                                    "foothold）当**起点**。")
            return
        others = [n for n in self.zones.sets if n != src]
        if not others:
            QMessageBox.information(self, "只有一个集合",
                                    "至少要两个集合才谈得上可达。")
            return
        lads = self._reach_ladder_choices(src)
        dlg = AddReachDialog(self, src, others, self._related_sets(src),
                             ladders=lads,
                             # ⭐ 本图**全部**绳（用户 2026-10-06 ✓ "加全部勾选"）——
                             #   勾上就能选到与起点集合**无关**的那些（行里会标出来 ✓）
                             all_ladders=self._all_ladder_choices(),
                             # ⭐ 门：与起点集合有关的 / 本图全部（2026-10-06 ✓ 同绳那套 ✓）
                             #   ⚠ 「全部」里会把**跨图**的门也列出来（行里标"不支持" ✓）——
                             #     列出来才说得清"这张图有哪些门"，标出来才不至于让人选错 ✓。
                             portals=self._reach_portal_choices(src),
                             all_portals=self._all_portal_choices(),
                             drop_choices=self._drop_choices(src))
        # **非模态 + 单实例**（见 _open_reach）：同一个起点再点一次 ⇒ 只聚焦，不新开
        self._open_reach(dlg, ("add", src))

    def _on_del_edge(self):
        e = self._cur_edge()
        if e is None:
            return
        self.del_edge(e)

    def _on_rev_edge(self):
        e = self._cur_edge()
        if e is None:
            return
        if self.rev_edge(e) is None:
            QMessageBox.information(self, "没法反向",
                                    "反向那条的集合已经不在（悬空边）—— 先处理它。")

    def _on_edit_edge(self, it=None):
        """**双击**「可到达」里一行 → 改这条边（终点 / 类型 / 绳 / 门）。

        为什么是双击：这一栏已经有 4 个按钮，再加一个"编辑…"就挤了；而"双击列表项编辑"
        是本窗口的通用手势（集合那一栏双击就是改名，见 `_on_rename`）。
        为什么**起点不可改**：起点是"我在编辑谁"（当前集合），改起点等于换一条边 ——
        用「删除」+「增加可达」就够了；而且换起点就得重算绳/门的候选（它们都依附于
        起点集合），一个弹窗里塞两件事更容易搞错。
        悬空边也能双击：那时终点预选不上，**保存就等于把它修好**（这是最该能做的事）。
        """
        e = it.data(Qt.UserRole) if it is not None else None
        if e is None:
            e = self._cur_edge()
        if e is None:
            return
        src = e.get("from")
        if src not in self.zones.sets:
            QMessageBox.information(
                self, "这条边悬空",
                "起点集合「%s」已经不在（被删掉或改过名）—— 先把集合名改回来，"
                "或者把这条边删掉。" % src)
            return
        others = [n for n in self.zones.sets if n != src]
        lads = self._reach_ladder_choices(src)
        dlg = AddReachDialog(self, src, others, self._related_sets(src),
                             ladders=lads,
                             # ⭐ 本图**全部**绳（用户 2026-10-06 ✓ "加全部勾选"）——
                             #   勾上就能选到与起点集合**无关**的那些（行里会标出来 ✓）
                             all_ladders=self._all_ladder_choices(),
                             # ⭐ 门：与起点集合有关的 / 本图全部（2026-10-06 ✓ 同绳那套 ✓）
                             #   ⚠ 「全部」里会把**跨图**的门也列出来（行里标"不支持" ✓）——
                             #     列出来才说得清"这张图有哪些门"，标出来才不至于让人选错 ✓。
                             portals=self._reach_portal_choices(src),
                             all_portals=self._all_portal_choices(),
                             drop_choices=self._drop_choices(src), init=e)
        dlg.mode = "edit"
        dlg.edge = e
        # 非模态 + 单实例：**这条边**再双击一次只聚焦；换一条边才会把旧的换掉。
        # ⚠ key 必须用**值**（from/to/kind/portal），不能用 `id(e)` —— 双击拿到的是 Qt
        # 转过来的**副本**，每点一次都是新对象、id 都不同 ⇒ 同一条边也会被当成"另一条"
        # 反复重开窗口（那就违背"再点只切换聚焦"了）。
        self._open_reach(dlg, ("edit", e.get("from"), e.get("to"),
                               e.get("kind"), str(e.get("portal") or "")))

    def _refresh_list(self):
        self.lst.blockSignals(True)
        self.lst.clear()
        for name, s in self.zones.sets.items():
            it = QListWidgetItem("%s  (%d)" % (name, len(s["footholds"])))
            it.setData(Qt.UserRole, name)
            # **只有名字是蓝字**，(14) 那个条数是正常黑字（用户 2026-09-26 的规矩）
            it.setData(ROLE_HTML, '<span style="color:%s">%s</span>  (%d)'
                       % (C_SETNAME, escape(str(name)), len(s["footholds"])))
            # 它自己的配色放在左边那个小色块里（和画布上该集合的颜色一致）。
            # ⚠ **别在这里 setForeground**：那会让**整行**（含 `(14)` 那个条数）都变成
            # 蓝的 —— 分色由上面的 ROLE_HTML 决定，这里的角色只管"没有富文本的行"
            # 的兜底（见 _RichRowDelegate）。
            c = QColor(s.get("color") or zones.PALETTE[0])
            px = QPixmap(12, 12)
            px.fill(c)
            it.setIcon(QIcon(px))
            self.lst.addItem(it)
            if name == self._highlight:
                self.lst.setCurrentItem(it)
        self.lst.blockSignals(False)
        self._refresh_edges()               # 边列表与集合列表同一个刷新入口
        self._refresh_status()
        self.btn_undo.setEnabled(bool(self._undo))
        self.btn_redo.setEnabled(bool(self._redo))

    def _cur_set(self):
        it = self.lst.currentItem()
        return it.data(Qt.UserRole) if it is not None else None

    def _focus_set(self):
        """现在"在编辑哪个集合" → 名字 / None。**可到达、建议都按它收窄。**

        顺序：高亮的集合 > 选中的 foothold 所属的集合（**同一批只属于一个集合**时才算）。
        两者都没有 ⇒ None ⇒ 那时显示全部（那是"总览"场景，也是逃生口）。
        """
        if self._highlight:
            return self._highlight
        owners = []
        for fid in sorted(self._sel):
            for n in self.zones.set_of(fid):
                if n not in owners:
                    owners.append(n)
        return owners[0] if len(owners) == 1 else None

    def _related_sets(self, name):
        """和 `name` 有关的集合 → 名字列表（**建议在前、已连着的在后**）。

        给「增加可达」收窄候选用：集合一多，全列出来根本找不着；而真正要加的边，
        九成就在这几条里（自动建议给出的候选，或者"反向加一条/换个走法"）。
        """
        out = []

        def add(n):
            if n and n != name and n in self.zones.sets and n not in out:
                out.append(n)

        have = {(e.get("from"), e.get("to"), e.get("kind"))
                for e in self.zones.edges}
        for s in (zones.walk_suggestions(self.zones, self.terrain)
                  + zones.climb_suggestions(self.zones, self.terrain)):
            if not s.get("to") or (s["from"], s["to"], s["kind"]) in have:
                continue            # 「另一端没圈」不是边；已采纳的也不用再列
            if s["from"] == name:
                add(s["to"])
            elif s["to"] == name:
                add(s["from"])
        for e in self.zones.edges:
            if e.get("from") == name:
                add(e.get("to"))
            elif e.get("to") == name:
                add(e.get("from"))
        return out

    def _on_pick_set(self):
        name = self._cur_set()
        self._highlight = name
        # **点选集合 ⇒ 清掉视窗里的选择**（反向的同一条规则，见 _on_picked）：
        # 一次只编辑一样东西。取消列表选中（name 为空）时**不清** —— 那只是"不再
        # 高亮某个集合"，手上正在框的那批 foothold 不该跟着没了。
        #
        # ⚠ 由此带来的顺序：要在某个集合上「加入/移出」，先点集合、再按住 Shift
        # 在视窗里选（Shift 不清高亮）。「加入集合…」那个弹窗默认就是当前这一行，
        # 所以两种顺序都能用。
        if name:
            self._sel = set()
            self._sel_ladder = None         # 同上：别让"看着的绳梯"和集合高亮一起亮 ✓
            self._sel_portal = None         # 传送点同理（2026-10-06 ✓）
        self._rebuild_scene()
        self._refresh_status()
        self._refresh_edges()       # 焦点变了 ⇒ 可达列表跟着收窄/放开
        if name:
            self.focus_ids(self.zones.sets[name]["footholds"])

    def _on_register(self):
        if not self._sel:
            QMessageBox.information(self, "先选 foothold",
                                    "先在地图上点选或框选几条 foothold。")
            return
        name, ok = QInputDialog.getText(self, "注册集合", "集合名（例如 A平台）：")
        if not ok or not name.strip():
            return
        try:
            self.register(name.strip())
        except ValueError as e:
            QMessageBox.warning(self, "注册失败", str(e))

    def _ask_set_name(self, title, verb):
        """挑一个集合（**默认当前高亮的那行**）→ 名字；没有集合/取消给 None。

        为什么不再"直接用列表里选中的那一行"：现在点集合会**清掉视窗里的 foothold
        选择**（一次只编辑一样东西，见 _on_pick_set），而"加入/移出"恰恰需要
        「一批 foothold + 一个集合」同时在场。弹一个默认当前行的选择框，两种顺序就
        都能用了：
            ① 先在视窗里选 foothold → 「加入集合…」→ 直接回车（默认就是刚才那个集合）
            ② 先点集合看清它圈了什么 → 按住 Shift 在视窗里加选 → 「加入集合…」→ 回车
        """
        names = list(self.zones.sets)
        if not names:
            QMessageBox.information(self, "还没有集合",
                                    "先用「注册为集合…」建一个。")
            return None
        cur = self._cur_set()
        idx = names.index(cur) if cur in names else 0
        name, ok = QInputDialog.getItem(self, title, "%s哪个集合：" % verb,
                                        names, idx, False)
        return name if ok and name else None

    def _on_add_to(self):
        name = self._ask_set_name("加入集合", "加入")
        if not name:
            return
        try:
            self.add_to_set(name)
        except ValueError as e:
            QMessageBox.warning(self, "加入失败", str(e))

    def _on_remove_from(self):
        name = self._ask_set_name("移出集合", "移出")
        if not name:
            return
        try:
            self.remove_from_set(name)
        except ValueError as e:
            QMessageBox.warning(self, "移出失败", str(e))

    def _on_rename(self):
        name = self._cur_set()
        if not name:
            return
        new, ok = QInputDialog.getText(self, "改集合名", "新名字：", text=name)
        if not ok or not new.strip() or new.strip() == name:
            return
        try:
            self.rename(name, new.strip())
        except ValueError as e:
            QMessageBox.warning(self, "改名失败", str(e))

    def _on_delete(self):
        name = self._cur_set()
        if not name:
            return
        r = QMessageBox.question(
            self, "删除集合",
            "删掉集合「%s」？\n（引用它的地方会一起清掉，可以用「撤销」找回）" % name,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r == QMessageBox.Yes:
            self.delete_set(name)

    def _on_save(self):
        path, probs = self.save()
        if probs:
            QMessageBox.warning(self, "保存了，但有需要注意的地方",
                                "\n\n".join(probs) + "\n\n文件：%s" % path)
        else:
            QMessageBox.information(self, "已保存", "写入 %s" % path)
