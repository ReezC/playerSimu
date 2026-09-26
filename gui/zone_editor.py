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
                             QDialog, QDialogButtonBox, QFormLayout, QFrame,
                             QGraphicsLineItem, QGraphicsPixmapItem,
                             QGraphicsScene, QGraphicsSimpleTextItem,
                             QHBoxLayout, QInputDialog, QLabel, QListWidget,
                             QListWidgetItem, QMessageBox, QPushButton,
                             QScrollArea, QSizePolicy, QStyle,
                             QStyledItemDelegate, QStyleOptionViewItem,
                             QVBoxLayout, QWidget)

from core import mapdata, zones
from gui import theme
from gui.canvas import ZoomPanView
from gui.widgets import NoWheelComboBox, NoWheelSlider

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

    def __init__(self, parent, src, dst_all, dst_related=(), ladders=(), portals=(),
                 drop_choices=(), init=None):
        super().__init__(parent)
        self.src = str(src)                 # 起点（外面加边时要用）
        self.mode = "add"                   # "add" / "edit" —— 由调用方设（见 _on_edit_edge）
        self.edge = None                    # 编辑模式下绑的那条边
        self.setModal(False)                # 非模态：开着它还能缩放看编辑器的图
        self._init = dict(init) if init else None
        self.setWindowTitle(("%s可达 —— %s" % ("编辑" if self._init else "增加", src)))
        self.setMinimumWidth(440)
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
            "  传送门   —— 走哪个门（要指名哪个门）\n\n"
            "⚠ 「跳」和「下跳」是**两种不同的按法**，别混：跳是往前蹦，下跳是往下穿。\n"
            "这两种都要用到跳键 ⇒ 真执行前得先做**跳跃标定**（现在先把位置标出来即可）。")
        form.addRow("通行方式", self.cmb_kind)

        # ---- 类型要求的东西：按类型显隐（这正是一个弹窗才做得到的事）----
        self.lbl_lad = QLabel("爬哪根绳")
        self.cmb_lad = NoWheelComboBox()
        # 排序（2026-09-26 要求"自动 sort"）：绳号按**自然序**（L2 在 L10 前面）、
        # 门按 id。候选是外面给的，顺序不该由"谁先被扫描到"决定。
        for lid, text in sorted(ladders, key=lambda x: _natural(x[0])):
            self.cmb_lad.addItem(text, lid)
        form.addRow(self.lbl_lad, self.cmb_lad)

        self.lbl_por = QLabel("走哪个门")
        self.cmb_por = NoWheelComboBox()
        for pn, text in sorted(portals, key=lambda x: _natural(x[0])):
            self.cmb_por.addItem(text, pn)
        form.addRow(self.lbl_por, self.cmb_por)

        # ---- 走(walk) 的方向类型（2026-09-26 用户要求）----
        self.lbl_wd = QLabel("类型")
        self.cmb_wd = NoWheelComboBox()
        for v in zones.WALK_DIRS:
            self.cmb_wd.addItem(zones.WALK_DIR_LABELS[v], v)
        self.cmb_wd.setToolTip(
            "「走」的方向类型：**默认方向**（朝目标走）/ **仅向左** / **仅向右**。\n\n"
            "⚠ 现在**只是配置占位**：执行器还没有「走到 x」那一步，这三个值暂时\n"
            "不影响任何行为 —— 先填着、能存住，逻辑后面再做。")
        form.addRow(self.lbl_wd, self.cmb_wd)
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
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        root.addWidget(box)

        self.ck_all.toggled.connect(lambda *_: self._fill_dst())
        self.cmb_kind.currentIndexChanged.connect(lambda *_: self._sync_kind())
        self._fill_dst()
        self._fill_drop()
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
            self.cmb_kind.setCurrentIndex(j)     # 触发 _sync_kind ⇒ 绳/门行跟着显隐
        for cmb, key in ((self.cmb_lad, "ladder"), (self.cmb_por, "portal")):
            v = d.get(key)
            if v:
                k = cmb.findData(v)
                if k >= 0:
                    cmb.setCurrentIndex(k)
        # 「走」的方向类型（默认方向 = 不写这一格 ⇒ 找不到就用第 0 项）
        k = self.cmb_wd.findData(zones.walk_dir(d))
        if k >= 0:
            self.cmb_wd.setCurrentIndex(k)

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
        """按类型显隐"绳 / 门 / 可下跳 foothold"那几行，并在缺料时说清**为什么不能加**。"""
        kind = self.cmb_kind.currentData()
        self.lbl_lad.setVisible(kind == "climb")
        self.cmb_lad.setVisible(kind == "climb")
        self.lbl_por.setVisible(kind == "portal")
        self.cmb_por.setVisible(kind == "portal")
        self.lbl_wd.setVisible(kind == "walk")      # 「类型」只对「走」有意义
        self.cmb_wd.setVisible(kind == "walk")
        self.drop_host.setVisible(kind == "drop")
        why = ""
        if kind == "climb" and self.cmb_lad.count() == 0:
            why = ("「%s」附近没有可爬的绳 —— 换个类型，或者先把绳另一端那块地形"
                   "圈成集合。" % self._src)
        if kind == "portal" and self.cmb_por.count() == 0:
            why = "「%s」范围内没有可用的传送门（出生点不算门）。" % self._src
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
            "portal": (self.cmb_por.currentData() if kind == "portal" else None),
            # 下跳：**只有人工改过才带这一格**（没改 = 用起点集合的全部 ⇒ 不写文件）
            "footholds": (list(self._drop_ids)
                          if kind == "drop" and self._drop_changed() else None),
            # 走：方向类型（默认方向 = None ⇒ 不写文件）
            "dir": (self.cmb_wd.currentData() or None) if kind == "walk" else None,
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
        # 窗口能缩多矮由**子控件的最小尺寸**决定（右栏已在滚动区里，画布最小 120），
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
        self._commit(snap)
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

    def _build(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
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
        # 画布的最小尺寸压小：它是"窗口能缩多矮"的实际决定者（右栏已经进滚动区了）
        self.view.setMinimumSize(240, 120)
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
        self.sl_bg.setFixedWidth(160)
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
        root.addLayout(left, 1)

        # 右侧那一列放进 QScrollArea（docs/UI规范.md §4：长面板进滚动区）——
        # 窗口一矮，这十几个控件就会被压扁/挤没（按钮看不见就没法干活了），
        # 有了滚动区是"出滚动条"，而不是"控件消失"。
        right_host = QWidget()
        right = QVBoxLayout(right_host)
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(QLabel("集合"))
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

        for text, slot, tip in (
                ("注册为集合…", self._on_register,
                 "把当前选中的 foothold 注册成一个命名集合（按地图存 id 列表）。"),
                ("加入集合…", self._on_add_to,
                 "把当前选中的 foothold 加入某个集合（弹窗默认当前高亮的那个）。\n\n"
                 "两种顺序都行：① 先在视窗里选 foothold → 这里选集合；\n"
                 "② 先点集合 → **按住 Shift** 在视窗里加选 → 这里选集合。"),
                ("移出集合…", self._on_remove_from,
                 "把当前选中的 foothold 从某个集合里移出（弹窗默认当前高亮的那个）。\n"
                 "顺序同「加入集合…」。"),
                ("改名…", self._on_rename, "改集合名（引用它的地方会一起改）。"),
                ("删除", self._on_delete, "删掉这个集合（**会二次确认**）。"),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            right.addWidget(b)

        # ---- 可到达（有向；§12.3 B5）----
        # 摆在集合下面：**先圈集合、再把它们连起来**，这个顺序就是干活的顺序。
        # 叫「可到达」是因为"边"是图论词：这一栏回答的是"从哪个平台能到哪个平台"。
        right.addSpacing(8)
        lbl_e = QLabel("可到达：")
        lbl_e.setStyleSheet("font-weight: 600;")
        right.addWidget(lbl_e)
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
            "类型：走 / 爬（绳梯）/ 跳 / 传送门。\n"
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
            "  · 传送门（portal）：**本图内**的门，落点在数据里（tn）；\n"
            "  · 「另一端还没圈」也会列出来 —— 那不是边，是提示你**该圈哪块地形**。")
        self.btn_sug.clicked.connect(self._on_suggest)
        right.addWidget(self.btn_sug)

        # ---- 可被到达（进来的边；**只读**，2026-09-26 要求 2）----
        # 为什么单独一栏：选中 A 时"我从哪儿来"和"A 能去哪儿"是两件事，混在一栏里
        # 谁都得逐行认方向。这里**不放编辑按钮** —— 要改就选中上面那个集合，
        # 它的"可到达"里就有这条边（一次只编辑一样东西，见 _refresh_edges）。
        # 整段套一个 QWidget：用例据此断言"这一栏里一个按钮都没有"。
        right.addSpacing(8)
        self.in_host = QWidget()
        lay_in = QVBoxLayout(self.in_host)
        lay_in.setContentsMargins(0, 0, 0, 0)
        lbl_in = QLabel("可被到达：")
        lbl_in.setStyleSheet("font-weight: 600;")
        lay_in.addWidget(lbl_in)
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
            "这里**纯看**：没有编辑按钮、双击也不进编辑 —— 要改就先把上面那个集合"
            "选中，\n它的「可到达」里就有这条边（一次只编辑一样东西）。\n\n"
            "画布上按类型着色的是**箭头**（绿=走、蓝=爬、橙=跳、紫=传送门），\n"
            "连线一律是黑细虚线。")
        # 这一栏里选中一行，画布上那条箭头也呼吸（纯看也能定位到是哪条）
        self.lst_in.currentItemChanged.connect(lambda *_: self._on_pick_edge(self.lst_in))
        lay_in.addWidget(self.lst_in)
        right.addWidget(self.in_host)

        row = QHBoxLayout()
        self.btn_undo = QPushButton("撤销")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo = QPushButton("重做")
        self.btn_redo.clicked.connect(self.redo)
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_redo)
        right.addLayout(row)

        self.btn_save = QPushButton("保存")
        self.btn_save.setToolTip("写入 %s" % self._zones_path.name)
        self.btn_save.clicked.connect(self._on_save)
        right.addWidget(self.btn_save)
        right.addStretch(1)

        # 滚动区：窗口矮下来时**出滚动条**，而不是把按钮压没
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)      # 别和主布局的边距套两层
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # 只滚纵向
        area.setWidget(right_host)
        area.setMinimumWidth(250)               # 再窄按钮上的字就挤没了
        area.setMaximumWidth(420)
        self.right_area = area                  # 留着（用例要断言"右栏真的在滚动区里"）
        root.addWidget(area, 0)

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
            from core.imgio import imread          # 走仓库的读图（中文路径安全）
            img = imread(path)
        except Exception:                          # noqa: BLE001
            return None
        if img is None or getattr(img, "size", 0) == 0:
            return None
        h, w = img.shape[:2]
        qimg = QImage(img.tobytes(), w, h, img.strides[0], QImage.Format_BGR888)
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
            it = scene.addLine(L.x, min(L.y1, L.y2), L.x, max(L.y1, L.y2),
                               self.view._pen(C_LADDER, style=Qt.DotLine))
            it.setZValue(1)
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
            lines.append((it, C_LADDER, Qt.DotLine, hit))
            if hit:
                self._hl_items.append((it, C_LADDER))
        for p in self.terrain.portals:
            it = scene.addEllipse(p.x - 6, p.y - 6, 12, 12,
                                  self.view._pen(C_PORTAL), QBrush(C_PORTAL))
            it.setZValue(6)
            hit = p in hl_por
            lines.append((it, C_PORTAL, None, hit))
            if hit:
                self._hl_items.append((it, C_PORTAL))
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
        调用点有两处 —— `showEvent`（重开编辑器）和 `_on_edit_zones`（点「编辑集合…」
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
        self.lbl_status.setText(
            "已选 %d 条 foothold%s%s　｜　鼠标 (%.0f, %.0f)　｜　集合 %d 个"
            % (n, "（点「注册为集合…」给它起名）" if n else "", where,
               hx, hy, len(self.zones.sets)))
        self.lbl_status.setToolTip(tip)

    # ---------------- 边 ----------------

    def add_edge(self, src, dst, kind, ladder=None, portal=None, footholds=None,
                 walk_dir=None, why=""):
        """加一条边（走撤销栈）。失败**不登记撤销点**（见 _commit 的说明）。"""
        snap = self._snapshot()
        e = self.zones.add_edge(src, dst, kind, ladder=ladder, portal=portal,
                                footholds=footholds, walk_dir=walk_dir, why=why)
        self._commit(snap)
        return e

    def del_edge(self, edge):
        """删掉一条边（按 from/to/kind/portal 匹配，避免拿错对象）。"""
        snap = self._snapshot()
        key = (edge.get("from"), edge.get("to"), edge.get("kind"),
               str(edge.get("portal") or ""))
        self.zones.edges = [
            e for e in self.zones.edges
            if (e.get("from"), e.get("to"), e.get("kind"),
                str(e.get("portal") or "")) != key]
        self._commit(snap)

    def rev_edge(self, edge):
        """反向复制一条边 → 新边（已经有了就返回 None）。"""
        src, dst = edge.get("to"), edge.get("from")
        if src not in self.zones.sets or dst not in self.zones.sets:
            return None
        snap = self._snapshot()
        e = self.zones.add_edge(src, dst, edge.get("kind"),
                                ladder=edge.get("ladder"),
                                portal=edge.get("portal"),
                                why=(edge.get("why") or "") + "（反向复制）")
        self._commit(snap)
        return e

    def edit_edge(self, edge, dst, kind, ladder=None, portal=None, footholds=None,
                  walk_dir=None):
        """改一条边（**终点 / 类型 / 绳 / 门**）。失败抛 ValueError。

        按 (from,to,kind,portal) 匹配（与 `del_edge` 同一套）：列表里拿到的是 Qt 转过
        的**副本**，不能按对象身份找。**起点在这里不动** —— 见 `_on_edit_edge`。

        类型换了要把**不再需要的那一格删掉**：走/跳不该留着上一任的绳号。保存时的校验
        只认当前类型，留着不至于报错，但会在下次改回"爬"时**突然复活**（那是上次的绳）。
        """
        snap = self._snapshot()
        key = (edge.get("from"), edge.get("to"), edge.get("kind"),
               str(edge.get("portal") or ""))
        hit = None
        for e in self.zones.edges:
            if (e.get("from"), e.get("to"), e.get("kind"),
                    str(e.get("portal") or "")) == key:
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
            if (e.get("from") == hit.get("from") and e.get("to") == dst
                    and e.get("kind") == kind
                    and str(e.get("portal") or "") == str(portal or "")):
                raise ValueError("已经有一条一样的了（%s → %s，%s）—— 不用改。"
                                 % (hit.get("from"), dst,
                                    zones.kind_label(kind)))
        hit["to"] = dst
        hit["kind"] = kind
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
        self._commit(snap)
        return hit

    def _cur_edge(self):
        it = self.lst_e.currentItem()
        return it.data(Qt.UserRole) if it is not None else None

    def _edge_row(self, e, fmt):
        """一条边 → 列表行。**只有两个集合名是蓝字**，`→`、类型、绳号都是正常黑字
        （用户 2026-09-26 的要求）；悬空边（指向已删除/改名的集合）那两个名字标红 ——
        那种边看着没事、跑起来才会在路径里断掉，必须在界面上就扎眼。

        `DisplayRole` 仍是**纯文本**（tooltip / 自检 / `text()` 从它取），
        分色那版放 `ROLE_HTML` 给 `_RichRowDelegate` 画。
        """
        kind = e.get("kind") or "?"
        zh, _color = zones.EDGE_LABELS.get(kind, (kind, "#9aa0a6"))
        extra = ""
        if kind == "climb":
            extra = "　绳 %s" % (e.get("ladder") or "?")
        elif kind == "portal":
            extra = "　门 %s" % (e.get("portal") or "?")
        bad = (e.get("from") not in self.zones.sets
               or e.get("to") not in self.zones.sets)
        it = QListWidgetItem("%s　[%s]%s%s"
                             % (fmt % (e.get("from"), e.get("to")), zh, extra,
                                "　⚠ 悬空" if bad else ""))
        it.setData(Qt.UserRole, e)
        nm_col = "#c5221f" if bad else C_SETNAME          # 悬空⇒红；正常⇒蓝（集合名专属）

        def nm(s):
            return '<span style="color:%s">%s</span>' % (nm_col, escape(str(s)))

        tail = '[%s]%s' % (zh, extra)                     # 不带颜色 ⇒ 正常黑字
        it.setData(ROLE_HTML, "%s → %s　%s%s"
                   % (nm(e.get("from")), nm(e.get("to")), tail,
                      '　<span style="color:#c5221f">⚠ 悬空</span>' if bad else ""))
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
        for e in sorted(out_edges, key=lambda x: edge_row_key(x, focus)):
            self.lst_e.addItem(self._edge_row(e, "%s → %s"))
        for e in sorted(in_edges, key=lambda x: edge_row_key(x, focus)):
            # 进来的边也写成「起点 → 终点」：这一栏的标题已经说了终点是谁
            # （能到「X」的），行里再写一遍反而绕。
            self.lst_in.addItem(self._edge_row(e, "%s → %s"))
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
        sug = list(zones.walk_suggestions(self.zones, self.terrain))
        sug += zones.climb_suggestions(self.zones, self.terrain)
        sug += zones.portal_suggestions(self.zones, self.terrain)
        have = {(e.get("from"), e.get("to"), e.get("kind")) for e in self.zones.edges}
        sug = [s for s in sug
               if s.get("to") is None or (s["from"], s["to"], s["kind"]) not in have]
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
                              ladder=s.get("ladder"), portal=s.get("portal"),
                              why="采纳建议：" + (s.get("why") or ""))
            except ValueError as ex:                    # noqa: BLE001
                QMessageBox.warning(self, "采纳失败", str(ex))

    def _reach_ladder_choices(self, src):
        """(绳, 门) 两串候选 → [(值, 界面上那行字)]，给「增加可达」那个弹窗用。

        绳用 `ladders_touching`（本集合的绳 ∪ **绳端通到本集合**的绳）：这里问的是
        "能爬上去的有哪些"，绳端离平台几十像素的那种也要能选到。
        门排除 `pt=0`（那是**出生点**，不是门）。
        """
        lids = zones.ladder_ids(self.terrain)
        lads = [(lids.get(id(L), "?"),
                 "%s  x=%d  y[%d..%d]  %s"
                 % (lids.get(id(L), "?"), L.x, min(L.y1, L.y2), max(L.y1, L.y2),
                    "绳子" if L.l else "梯子"))
                for L in zones.ladders_touching(self.terrain,
                                                self.zones.sets[src]["footholds"])]
        pors = [(p.pn, "%s  (%d,%d)%s"
                 % (p.pn, p.x, p.y,
                    "  → 跨图 %s" % p.tm if p.tm != 999999999 else "  → 本图内"))
                for p in zones.portals_of(self.terrain,
                                          self.zones.sets[src]["footholds"])
                if p.pt != 0]
        return lads, pors

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
        """
        dlg = self.sender()
        try:
            if getattr(dlg, "mode", "add") == "edit" and getattr(dlg, "edge", None):
                self.edit_edge(dlg.edge, r["dst"], r["kind"], ladder=r["ladder"],
                               portal=r["portal"], footholds=r.get("footholds"),
                               walk_dir=r.get("dir"))
            else:
                self.add_edge(dlg.src, r["dst"], r["kind"], ladder=r["ladder"],
                              portal=r["portal"], footholds=r.get("footholds"),
                              walk_dir=r.get("dir"),
                              why="手工加的（%s → %s）" % (dlg.src, r["dst"]))
        except ValueError as ex:                        # noqa: BLE001
            QMessageBox.warning(self, "没加成", str(ex))

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
        lads, pors = self._reach_ladder_choices(src)
        dlg = AddReachDialog(self, src, others, self._related_sets(src),
                             ladders=lads, portals=pors,
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
        lads, pors = self._reach_ladder_choices(src)
        dlg = AddReachDialog(self, src, others, self._related_sets(src),
                             ladders=lads, portals=pors,
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
                  + zones.climb_suggestions(self.zones, self.terrain)
                  + zones.portal_suggestions(self.zones, self.terrain)):
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
        self._rebuild_scene()
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
