"""路线识别面板：平台识别 / 跳跃落点预测 的总开关 + 小地图定位 + 地形图。

「总开关」这几项每帧做图像处理（平台顶边检测、玩家运动追踪），比较吃 CPU，
默认关闭。**关掉 = 这块一点活都不干**（包括地形关系识别，见 live_thread），
所以它是一条真的性能开关，不是只影响画面。

**小地图定位**：寻路要用小地图上的黄点算出世界坐标，而「小地图面板怎么显示
底图」有两种方式，**不同地图可能不一样**（所以界面上的名字按"看见的是什么"来）：

    全局小地图（fit）   整张底图缩放到面板里     → 地形图不随人移动，只有黄点在动
    局部小地图（crop）  面板 1:1 显示底图的一块  → 地形图跟着人滑动（黄点基本在中间）

这里**让你选**（每张图存一份，见 datasets/map/<id>.mapcalib.json），
程序不替你猜。判断方法就写在面板上：进游戏左右走两步看一眼即可。

**地形图**：显示 `datasets/map/<id>_zones.png` —— **地形编辑器的结果**（圈进集合的
平台按集合颜色画 + 名字标在平台上方，没圈的画暗灰）。"程序认得的平台"就是它，
所以寻路要往哪走、小地图定位对不对，看着这张图才有概念。

同目录还留一张 `<id>_overlay.png`（每段一色 + 段号，`tools/map_terrain_view.py`
生成）：**叠到实时画面上用的仍是它** —— 那里要看的是"所有几何位置对不对"，
不是"我圈了哪几块"。两张都由「生成地形图」一起画。
"""

import time

import numpy as np

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGroupBox, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget)

from core import mapdata
from decision.agent import ZONE_GOTO_RETRY_S, settings
from gui import theme
from gui.canvas import ImageCanvas
from gui.minimap_calib import MinimapCalibDialog   # 量「面板 ↔ 底图」的弹窗
from gui.two_point_calib import TwoPointCalibDialog   # 双点标定（填坐标解几何）
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox,
                         NoWheelSlider, NoWheelSpinBox,
                         scroll_area)   # 滚轮不许改参数（UI规范 §5）+ 参数区当分割器一栏
from gui.worker import safe_slot        # 槽里抛异常 = 整个工作台 abort（见 worker.py）
from perception import minimap as mm
# ⚠ 小地图框选区域**不在这里存**：它**按项目**存（`project.yaml` 的 `mmap_crop`）——
#   取它的口径只有 `perception.minimap.crop_of` 一处（本项目 → 没框过时回退老的那份 ✓）。
#   `live.yaml` 这里只剩**来源**（`mmap_src`）和叠图那几项 ✓。
from core.config import load_live, update_live

#: 「地形图」那一栏的**显示类型**（用户 2026-10-06 ✓ 原话：把「叠加实时小地图」那个勾选框
#: 改成「显示类型」下拉，含 **仅地形图 / 实时小地图 / 像素差分地图** ✓）。
#: 短键是**存进 `live.yaml` 的那个值**（`live_map_view` ✓ 本机外观偏好，和 `live_map_alpha`
#: 同一处 ✓）；界面名字只是给人看的 ✓。
LIVE_VIEWS = (("仅地形图", "terrain"),
              ("实时小地图", "live"),
              ("像素差分地图", "diff"))
#: 默认 = **实时小地图**：那会儿这个勾选框**默认就是勾上的**（用户 2026-09-29 追加"怎么没在
#: 路线识别页签→地形图里的视图区看到实时滚动的小地图"✗ ⇒ 默认关着等于没做 ✓）⇒ 不许改 ✗。
LIVE_VIEW_DEFAULT = "live"
#: 「像素差分地图」里"**背景档**"（整幅差的中位）大到这个灰阶 ⇒ 说明底图与游戏里那张小地图
#: **不同源**（底色/线条不一样 ✓）⇒ 差分会整幅发亮、只能当辅助看。**那时必须在状态行说清**
#: 并染橙色 ✗：静默的话人会拿它当判据去调标定 ✓（那方向是错的 ✓）。
#: ⚠ **定义已经搬到 `perception/minimap.py`**（用户 2026-10-06「下一步」✓：同一根线现在还要
#:   当"**差分层能不能拿来定位**"的闸 ✓ ⇒ 感知层是它的家 ✓，界面反过来 import 它 ✓
#:   约定 10：一处实现 —— 两边各写一个 30 迟早分叉 ✗）。
DIFF_FLOOR_WARN = mm.DIFF_FLOOR_WARN


def norm_live_view(v):
    """`live_map_view` 的值 → `LIVE_VIEWS` 里的短键（坏的 / 缺的 ⇒ 默认 ✓ 看的东西别抛 ✗）。"""
    v = str(v or "")
    for _zh, _k in LIVE_VIEWS:
        if v == _k:
            return _k
    return LIVE_VIEW_DEFAULT


def _mmss(sec):
    """秒 → `M:SS`（倒计时统一这个写法，和 player_panel 那边一致）。"""
    sec = max(0, int(sec))
    return "%d:%02d" % (sec // 60, sec % 60)


def _clear_layout(layout):
    """清空一个布局（连里面的子控件一起销毁）—— 「编辑战斗区域」那条摘要要**重画**。

    ⚠ 为什么要销毁而不是只摘出来：布局里的 `QLabel` 摘出来如果不删，会**堆在父
    widget 上越积越多**（每次重画留一批不可见的孤儿 ✗）。
    """
    while layout.count():
        item = layout.takeAt(0)
        child = item.layout()
        if child is not None:
            _clear_layout(child)
        w = item.widget()
        if w is not None:
            w.deleteLater()


def _mmss2(sec):
    """秒 → **`MM:SS`**（分也补零 ✓）—— 「当前任务」那条**生存时间**专用（用户 2026-09-28 ✓
    举例：`前往：底层(追击  剩余 00:15)` ⇒ 分是**两位** ✓）。

    ⚠ **不复用 `_mmss`** ✗：那个是 `M:SS`（`0:15`），被 `_timer_lines` 好几处用着（休息 /
    定时行为 ✓），改它会**动到已有的显示** ✗。两套并存、各管一行 ✓。
    """
    sec = max(0, int(sec))
    return "%02d:%02d" % (sec // 60, sec % 60)


def _generate_task(params, ctx):
    """后台：缺地形数据就先从 WZ 导，再画叠加图。**不碰任何 Qt 控件。**

    （跑在 gui.worker.TaskThread 里，见那里的铁律：工作线程只 emit 信号。）
    """
    mid = str(params["map_id"])
    out = mapdata.map_dir()
    json_path = out / ("%s.json" % mid)

    if json_path.exists() and not params.get("force_export"):
        # 地形数据已经有了：只重画叠加图（毫秒级）—— 大部分时候用户只是想看
        # 最新的那张，没必要每次去解析一遍 Map.wz（那是几十秒）。
        ctx.log("已有地形数据 %s，只重画叠加图" % json_path.name)
    else:
        from core import wzexport
        wzexport.run_export_terrain({"map_id": mid, "out": str(out)}, ctx)

    if ctx.canceled():
        return {"summary": "已取消"}

    from core import zones as zones_mod
    from tools.map_terrain_view import render
    t = mapdata.load(mid, with_canvas=True)
    if t is None:
        raise RuntimeError("读不到地形 JSON：%s" % json_path)

    target = out / ("%s_overlay.png" % mid)
    # render 返回的是 foothold 条数（其中墙多少），不是"串成的段数" —— 别写错标签
    w, h, n_fh, n_wall = render(t, target)
    ctx.log("叠加图 %s  %d×%d（foothold %d / 其中墙 %d，串成 %d 段）"
            % (target.name, w, h, n_fh, n_wall, len(t.segments)), "ok")

    # 第二张：**地形编辑器的结果**（颜色=集合、名字标在平台上）——「路线识别」的
    # 「地形图」显示的是它。两张都画：叠图那版仍要留着（叠到实时画面上时，要看的是
    # "所有几何位置对不对"，不是"我圈了哪几块"）。都只有毫秒级。
    z = zones_mod.load(mid)
    ztarget = out / ("%s_zones.png" % mid)
    render(t, ztarget, zones=z)
    ctx.log("集合图 %s（%d 个集合%s）"
            % (ztarget.name, len(z.sets),
               "；还没圈集合，点「寻路编辑器」" if not z.sets else ""), "ok")
    return {"summary": "已生成 %s" % ztarget.name, "path": str(ztarget)}


class RoutePanel(QWidget):
    #: ⭐ **本面板把「项目的地图 id」改掉了**（用户 2026-10-02 的「手动更换」✓）——
    #: 参数是新地图 id ✓。
    #: 谁接谁刷新（**不强推给谁** ✓）：主窗口接上重新推 `player_panel.set_zone_sets` ——
    #: 那正是 `_bind_cards` 在"切项目"时做的同一件事（集合按**地图 id** 存 ✓）；
    #: 本面板自己的刷新不靠信号（`_on_manual_map_change` 里当场做 ✓）。
    map_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # ⚠ **长面板必须能滚**（用户 2026-09-27 报："路线识别页签不支持滚动？现在攀爬参数组的
        #   行和行都重叠了"）—— 这两个症状是**同一个病**：原来这里直接 `QVBoxLayout(self)` ✗
        #   ⇒ 内容比页签高时 Qt 只能**硬挤**，卡片里相邻的行被压到**互相重叠** ✗。
        # 版面：**上栏 = 参数区（滚动）、下栏 = 地形图（常驻）**，中间一条**可拖动的分隔条** ✓。
        # ⚠ 用户 2026-09-27 报："寻路配置的地形图……**太占位置了**" —— 原来那版是
        #   `card("地形图", stretch=1, into=self.layout())` ⇒ 它**吃掉全部剩余高度** ✗，
        #   用户没法把它压小。现在换成 `QSplitter(Qt.Vertical)`（项目口径见
        #   `docs/UI规范.md` §4：多栏用 `QSplitter` + **`setChildrenCollapsible(False)`**）：
        #   **拖分隔条**就能决定地形图占多少 ✓（初始只给它 300px ✓），
        #   ⚠ `setChildrenCollapsible(False)` = 不许拖成 0（规范硬要求 ✗ 不是"隐藏"）。
        # 为什么不做"折叠"控件：项目里没有折叠先例 ✗；而且这页的两条硬约束是
        #   "参数区能滚 + 画布不进滚动区"（`t_route_panel_scrolls` 钉着 ✓），
        #   分割条两栏正好各管一条 ✓。
        # ⚠ **画布不许进滚动区**（2026-09-27 实测踩到的）：③「地形图」里那张
        #   `ImageCanvas` 是 `QGraphicsView`（**它自己就是个滚动视图** ⇒ 再套一层
        #   `QScrollArea` = 滚动视图套滚动视图）—— 这样一放，工作台**退出时会偶发
        #   0xC0000005**（access violation；实测 8/8 崩 ✗，把画布换成普通控件立刻 8/8 好 ✓）。
        #   所以参数区才单独进滚动区、地形图单独占一栏 ✓。
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.split_page = QSplitter(Qt.Vertical)
        self.split_page.setHandleWidth(6)
        self.split_page.setChildrenCollapsible(False)      # 不许拖成 0（UI规范 §4 ✓）
        outer.addWidget(self.split_page)

        # 上栏：参数区（`scroll_area` 是 `QScrollArea` 的**唯一 new 处** ✓；它专门给
        # "滚动区自己要当一栏交出去"这种场景用 —— 见 `gui/widgets.py::scroll_area` 的说明 ✓）。
        # 边距 12 / 间距 8 与原来（`scroll_page(self)`）**一模一样** ⇒ 观感不变 ✓。
        holder = QWidget()
        page = QVBoxLayout(holder)
        page.setContentsMargins(12, 12, 12, 12)
        page.setSpacing(8)
        _pscroll = scroll_area(holder, self)
        # 栏的最小尺寸**由人定**（UI规范 §4 ✓）：参数区至少留这么高 ——
        # 不然窗口一矮，Qt 会把参数区压成一条缝、地形图却还稳稳占着 300px（实测踩到 ✓）。
        _pscroll.setMinimumHeight(240)
        self.split_page.addWidget(_pscroll)
        # 下栏：地形图（常驻 ✓ —— 卡片加进这一栏的布局，见下面 `card(..., into=...)` ✓）。
        map_holder = QWidget()
        map_holder.setMinimumHeight(170)                   # 画布 140 + 卡片上下留白 ✓
        self.maplayout = QVBoxLayout(map_holder)
        self.maplayout.setContentsMargins(12, 8, 12, 12)
        self.maplayout.setSpacing(8)
        self.split_page.addWidget(map_holder)
        self.split_page.setStretchFactor(0, 1)             # 参数区先长 ✓
        self.split_page.setStretchFactor(1, 0)
        self.split_page.setSizes([640, 300])               # 初始别让地形图占太多 ✓

        def card(title, tip="", stretch=0, into=None):
            """按 UI 规范把**一组**控件装进一张卡片 → 返回卡片内的布局。

            为什么是 `QGroupBox` 而不是工作流那套 `StepCard`：那个自带状态灯和运行
            按钮，是给"一步一步的任务"用的；这里是**参数页** —— 项目里参数分组一律
            写 `QGroupBox("标题")`（`gui/player_panel.py` 里 12 处都是），白底圆角 +
            灰色小标题的样式由 `main_window` 的全局 QSS 给（`QGroupBox` /
            `QGroupBox::title`）。**别在这儿自己写样式**：写了就跟着主题走不动了。

            ⚠ 顺序仍然是**代码顺序 = 视觉顺序**（docs/UI规范.md §4）：卡片按下面
            出现的先后从上到下排，要挪位置就整块搬（见下面三次 `root = card(...)`）。
            """
            box = QGroupBox(title)
            if tip:
                box.setToolTip(tip)
            lay = QVBoxLayout(box)
            lay.setContentsMargins(10, 6, 10, 8)
            lay.setSpacing(6)
            # `into`：**常驻**的卡片加在指定布局上（滚动区**外面** ✓）——
            # 现在只有 ③「地形图」用它：加进 `self.maplayout`（分割器的下栏 ✓），
            # 那张画布是 `QGraphicsView`，**不能**进滚动区 ✗（见 `__init__` 那段说明 ✓）。
            # ⚠ **不许写成 `(into or page)`**（2026-09-27 实测踩到 ✗）：PyQt 的 **空布局是"假"的**
            #   （`QLayout` 有 `__len__` ⇒ 刚建好、还没装东西时 `bool() == False` ✗）⇒
            #   新栏那个空 `maplayout` 会被当成 falsy、卡片又落回参数页（画布就这么进了滚动区 ✗，
            #   正是 `t_route_panel_scrolls` ⑥ 那条把这件事钉出来 ✓）。⇒ **显式判 `None`** ✓。
            if into is not None:
                into.addWidget(box, stretch)
            else:
                page.addWidget(box, stretch)
            return lay

        # ---- ① 寻路配置：集合 + 执行器参数 ----
        # 下面这段所有 `root.add*` 都落进这张卡片（`root` 依次指向各卡片的内布局）。
        # 组名 2026-09-26 由「路线识别」改成「寻路配置」（用户要求；页签名不变 ✓）。
        root = card("寻路配置")
        #: ⭐ **记住这张卡的布局**（用户 2026-10-03 ✓ 原话："「小地图定位」的来源、黄点跟踪
        #:   参数挪到寻路配置最上方"）：那两组控件的**定义**还在「小地图定位」那一节里
        #:   （紧挨着它们说明的上下文 ✓ 搬定义要连 tooltip 一起抄 40 行 ✗ 不值），
        #:   但**它们的布局插到这张卡的顶部** ✓（`insertLayout` ✓ 见下面两处）。
        #:   为什么这么做：这三样**按地图 id 存**（`core/route_cfg.py` ✓）⇒ 和"现在是哪张图"
        #:   是同一件事的两半，放一起才看得出是一组 ✓。
        #:   ⚠ 插入位置从 **1** 起（不是 0）：第 0 位是「当前地图」那行 —— 那一行是全卡的
        #:     **前提**（用户 2026-10-02 定的"在寻路配置最顶部"✓ 下面每一件都以它为前提 ✓）。
        root_pf = root

        # ---- ⭐ 「当前地图」+「手动更换」（用户 2026-10-02 ✓ 原话："在路线识别→寻路配置
        #      最顶部 增加只读参数『当前地图』，默认为 模型训练 页签→识别目标选项里的地图参数；
        #      增加按钮『手动更换』，用以修改这个 id"）----
        # 为什么必须在**这一组的最顶上**：下面每一件（寻路编辑器 / 框选小地图 / 标定 /
        #   集合下拉 / 战斗区域 / 攀爬参数）**全都以"现在是哪张图"为前提** ✓ ——
        #   不先把这件事说清，下面所有操作都在猜（而"猜错了图"这一类最难查 ✗）。
        # · **只读**（`QLabel` ✓ 与本页「小地图定位」卡里那个同名标签同一写法、同一个文本 ✓
        #   见 `_map_display_text`）：它默认就是「模型训练 → 识别目标」那个地图参数 ——
        #   一份数据两个显示（`project.map_id` ✓），切页签/换项目靠 `_refresh_mmap()`
        #   重读（`bind` / `showEvent` 都会调 ✓）。
        # · 「手动更换」= 就地改它：候选与写口都复用同一份实现
        #   （`gui/map_picker.MapPickDialog` + `core.wzexport.apply_map_choice` ✓）——
        #   走的是和那个下拉**同一个写口**（地图 + 怪列表四件一起写 ✓ 各写一份迟早漏 ✗）。
        # · 样式：**普通按钮**（同旁边「框选小地图」「实测精度」✓）—— 它不是"打开编辑器"
        #   那种入口（那个才用 `theme.ENTRY_BTN_QSS` 拎出来 ✓ 见 UI规范 §6）。
        mrow = QHBoxLayout()
        mrow.setSpacing(8)
        mrow.addWidget(QLabel("当前地图"))
        self.lbl_cur_map = QLabel("—")
        self.lbl_cur_map.setStyleSheet("color: #5f6368;")
        self.lbl_cur_map.setWordWrap(True)
        mrow.addWidget(self.lbl_cur_map, 1)
        self.btn_map_change = QPushButton("手动更换")
        self.btn_map_change.setObjectName("mapChange")
        self.btn_map_change.setToolTip(
            "就地换掉这个项目的**地图参数**（和「模型训练 → 识别目标」那个下拉"
            "是同一个 ✓）。\n\n"
            "换了之后：地形图 / 集合 / 标定 / 战斗区域都跟着这张图走；\n"
            "「实时」如果在跑，会**立刻改用新图**（不用重启 ✓）。\n\n"
            "为什么要有这个按钮：那个下拉在**另一个页签**里，本来要来回切页签 +\n"
            "等本页重读一次才看得到变化。\n\n"
            "⚠ 换图会连带把「要识别的怪」换成这张图的（和那个下拉的行为一致 ✓）——\n"
            "上一张图手动增删过的怪列表**不会**跟过来。")
        self.btn_map_change.clicked.connect(safe_slot(self._on_manual_map_change))
        mrow.addWidget(self.btn_map_change)
        root.addLayout(mrow)
        self._refresh_current_map()

        # ---- ⭐ 「小地图来源」**搬到这里**（用户 2026-10-03 ✓ 原话："「小地图定位」的
        #      来源、黄点跟踪参数挪到寻路配置最上方"）----
        # 为什么挪：三样（来源 / 跟踪参数 / 框选区域）**按地图 id 存**（`core/route_cfg.py`
        #   ✓ 见 `docs/开发日志.md` 第 124 条）⇒ 和"现在是哪张图"是同一件事的两半，
        #   放一起才看得出是一组 ✓。
        # ⚠ 位置在「当前地图」那行**下面**（不是它上面 ✗）：那一行是**全卡的前提**
        #   （用户 2026-10-02 定"在寻路配置最顶部"✓ 下面每一件都以它为前提 ✓）。
        # ⚠ 为什么把**代码整段搬过来**、而不是"定义留在原地、只把布局插过来"（我先试了 ✗）：
        #   实测那个 combo 会**丢父**（父链为空 ⇒ 变成顶级控件 ⇒ 之后 `mapTo` 直接
        #   **访问违例崩进程** ✗ 见 `t_mmap_rows_split` 那次 0xC0000005 ✓）——
        #   布局这样借来借去不可靠 ✗，控件建在**它真正要待的那张卡**里最稳 ✓。
        # ⚠ 摆放：一行只放一个参数组（UI 规范 §4）—— 这一行**只有来源** ✓（「标定…」
        #   仍在「小地图定位」卡里挨着它读的那份几何 ✓）。
        srow = QHBoxLayout()
        srow.setSpacing(6)
        srow.addWidget(QLabel("小地图来源"))
        self.cmb_mmap_src = NoWheelComboBox()
        self.cmb_mmap_src.addItem("收流（A 机小地图推流）", mm.SRC_STREAM)
        self.cmb_mmap_src.addItem("从实时画面框选（实验）", mm.SRC_LIVE)
        self.cmb_mmap_src.setToolTip(
            "小地图面板的画面从哪来。\n\n"
            "收流（默认）：A 机「被控机部署台 → 小地图推流」单独推一路原始像素 ——\n"
            "  黄点（玩家点）只有几个像素，这一路是唯一能保证它不糊的画质。\n\n"
            "从实时画面框选（实验）：不另推一路，直接在「实时」页那一帧上裁一块。\n"
            "  省掉 A 机一次截屏 + 一路 TCP，但画面是压过的 —— 面板和底图还能对上，\n"
            "  黄点识别能不能稳还没实测。\n\n"
            "（「框选小地图」和来源**无关**，是必做的一步：两种来源都要知道\n"
            "  小地图面板在实时画面的哪儿 —— 收流时主画面里也有小地图，只是压过。）\n"
            "（2026-09-27 起它在「设置 → 界面 → 实时画面 · 地形叠加」里。）\n\n"
            "两者只在「画面从哪来」这一步不同：标定、换算、世界坐标都一样。\n\n"
            "⭐ 2026-10-03 起它**按地图 id 存**（`datasets/map/<id>.route.json` ✓）——\n"
            "   每张图各记一份（换图就跟着换 ✓）；`config/live.yaml` 那份只剩"
            "「新图第一次打开时的播种值」。")
        self.cmb_mmap_src.currentIndexChanged.connect(self._on_mmap_src)
        srow.addWidget(self.cmb_mmap_src)
        srow.addStretch(1)
        root.addLayout(srow)

        # 这一组**只剩「寻路编辑器」**（2026-09-26 用户定：其余都没意义）。
        # 原来上面还有一段说明 + `启用路线识别` 开关 + 一行状态提示 —— 那个开关门控的是
        # **感知**（平台识别 / 落点预测 / 小地图定位），关掉时「命令前往」连"你在哪个
        # 平台"都答不出来，正是"点一下没反应"的典型来源。现在这些感知**一直跑**
        # （见 gui/live_thread.py），页面上只留真正要人动手的那一件事。
        # 它打开的是 **foothold 集合编辑器**那个窗口（窗口标题仍是那个名字）：寻路模块的
        # 第一块（「我在不在 A 平台」靠它）。
        # ⚠ 按钮文案 **2026-09-26 用户定：叫「寻路编辑器」**（原来叫「编辑集合…」）。
        #   连带把**所有指着这个按钮**的文案一起改了（同一批：状态行 / tooltip /
        #   玩家面板的「限制战斗区域」提示 / 设置里那条说明 / 解析失败那句 why）——
        #   只改按钮不改它们，界面上就会让人去点一个**不存在的**按钮 ✗。
        #   窗口自己的标题（`gui/zone_editor.py` 的 "foothold 集合编辑器 —— <地图>"）
        #   **没动**：那是"编辑器"这个名字，和"按钮叫什么"是两件事（用户没提过）。
        self.btn_zones = QPushButton("寻路编辑器")
        self.btn_zones.setToolTip(
            "打开 foothold 集合编辑器：把这张图的地形画出来，点选/框选 foothold\n"
            "注册成命名集合（存 datasets/map/<id>.zones.json，**按地图 id 一份**）。\n\n"
            "「我在不在 A 平台」这条判据就靠它：自动串段会把「要跳/攀才能互通」的\n"
            "并成同一段（实测 105090600 的第 0 段把一面 388 像素高的悬崖当成了平台\n"
            "边缘），所以分组只能由人圈。详见 docs/寻路设计.md §12。")
        # **换个颜色把它拎出来**（用户 2026-09-26 要求）：这一组里它是**唯一的入口**
        # （打开 foothold 集合编辑器 = 寻路的第一块地基），和旁边那些纯参数控件长成
        # 一个样时没人找得到 ✗。靛蓝是这套配色里没被占用的色位（蓝=主按钮、
        # 绿=通过、橙=提醒、红=错误，都各有用处 ✓）。
        self.btn_zones.setObjectName("editZones")
        # 样式来自 `gui.theme.ENTRY_BTN_QSS`（**一处**）：2026-09-27 起「打开质检台」也用
        # 同一个角色 ⇒ 颜色值别在这里再写一份 ✗（约定 10）。
        self.btn_zones.setStyleSheet(theme.ENTRY_BTN_QSS)
        self.btn_zones.clicked.connect(self._on_edit_zones)
        root.addWidget(self.btn_zones)

        # ---- ⭐⭐ 「**禁用杀怪寻路**」从「设置 → 判定参数」**搬到这里**（用户 2026-10-06 ✓
        #      原话："把禁用杀怪寻路从设置里移出来，移到路线识别→寻路配置→寻路编辑器按钮下面"）----
        # 为什么搬：它**只服务寻路这一条链**（追击那条 ✓ 见 `_on_no_chase_path` 与
        #   `DecisionSettings.disable_chase_pathfinding` 的说明 ✓）⇒ 摆在"判定参数"里，
        #   人根本想不到它和"寻路编辑器"是同一件事 ✗。
        # ⭐⭐ **存储口径也是这一轮定的**（用户 2026-10-06 ✓ 原话："**从此 禁用杀怪寻路就是按
        #   地图id存的数据，而不是在设置里全局一份**"✓）：它落在
        #   `datasets/map/<id>.route.json`（`core/route_cfg.PARAM_KEYS` ✓ 与那 6 个攀爬/重试
        #   参数同一份 ✓）⇒ **每张图各一份** ✓ —— 因为它管的是"**这张图**要不要用寻路追怪"
        #   （寻路成不成立是**图**的属性 ✗ 不是项目的 ✓）。
        #   ⚠ `project.yaml` 里那一格**留着当播种值**（老项目已经配过的会在某图第一次打开时
        #     种进按图那份 ✓ 不丢配置 ✓ 同那 6 个 ✓）。
        # ⚠ **旧入口已删**（`settings_dialog._page_judge` 里那一格 + 它的写回 ✓）——
        #   两个入口各存一份迟早分叉 ✗（本仓库"唯一入口"的纪律 ✓）。
        # ⚠ 保存时机随**这一组**的规矩：**即改即存**（`_on_no_chase_path` ✓）；
        #   设置面板那边是"点确定才写"✗ —— 两种时机别混 ✓。
        self.ck_no_chase_path = QCheckBox("禁用杀怪寻路")
        self.ck_no_chase_path.setChecked(
            bool(getattr(settings, "disable_chase_pathfinding", False)))
        self.ck_no_chase_path.setToolTip(
            "**开了之后，追怪不再走寻路**：\n\n"
            "· **不再查询**「怪物框底下是哪块 foothold 集合」（画面上那行\n"
            "  `查#怪号 集合名` 也不再出现 ✓）；\n"
            "· **不再因为追怪而下达寻路任务**（含「朝方向逐层逼近」那条降级 ✓）；\n"
            "· 只是**单纯地朝锁定怪物走**（= 没有寻路时的老逻辑：只按 ←/→ ✓）。\n\n"
            "什么时候开：寻路那条线总判不准、或者只想让它先凑过去打的时候 ✓。\n"
            "关了（默认）就是老行为：判得出怪在哪块平台就先下「前往」过去 ✓。\n\n"
            "⚠ 它**只管「追击」这一条链**（用户 2026-10-06 澄清 ✓）—— 下面这三件\n"
            "  **照旧生效**：\n"
            "· 「编辑战斗区域」里勾了「可以战斗」时的**区域筛选**（那也在查集合，\n"
            "  但它服务的是「别追出禁战区」，不是追怪寻路 ✗）；\n"
            "· **「人不在允许战斗的区域 ⇒ 先回去」**（回区那条寻路任务**照下** ✓\n"
            "  —— 它不是追击 ✗）；\n"
            "· 追击起跳（由那个独立开关管 ✓）。\n\n"
            "⚠ 开着它时**开自动前的体检会放宽**（地图 id / 标定 / 地形图那三件不再拦你 ✓\n"
            "  —— 这个模式不读世界坐标 ✓ 见 `player_panel._precheck_problems` ✓）。")
        self.ck_no_chase_path.toggled.connect(self._on_no_chase_path)
        root.addWidget(self.ck_no_chase_path)

        # ---- 「框选小地图」（2026-09-27 从「设置 → 界面」搬来，用户要求）----
        # 为什么必须是**这一页、这一组、这个位置**：
        #   · **按项目**（地图）—— 不同地图的小地图面板尺寸/位置完全不同 ⇒ 它天生跟项目走，
        #     放"设置"（全局一份）里就是错的：换个项目还是上一张图的框 ⇒ 叠图/定位全错 ✗
        #     （用户 2026-09-27 现场报的就是这个）；
        #   · 和「寻路编辑器」是**连续的同一件事**（先圈地形 → 再框面板位置），
        #     ⚠ 原来"紧贴按钮正下方"，2026-10-06 起中间隔了「禁用杀怪寻路」那一格
        #     （用户要求把它挪到按钮下面 ✓）—— 仍是"接着寻路编辑器的那一串" ✓。
        # ⚠ 框选要**当前实时画面**：拿 `self.live_panel.current_frame()`（主窗口在
        #   `_bind_cards` 里塞进来的，和叠图同一处 ✓）；没开始预览会说清怎么开。
        # ⚠ 走 gui/region_selector（放大镜 + Esc + <4px 当误点，docs/UI规范.md §8）——
        #   框选全仓库只有那一份，不许在这儿再实现一遍。
        crow = QHBoxLayout()
        crow.setSpacing(8)
        self.btn_mmap_crop = QPushButton("框选小地图")
        self.btn_mmap_crop.setObjectName("mmapCrop")
        self.btn_mmap_crop.setToolTip(
            "在**实时画面**上把游戏的小地图面板框出来。\n\n"
            "存哪儿：**本项目**（projects/<项目>/project.yaml 的 `mmap_crop`）——\n"
            "不同地图的小地图面板尺寸/位置完全不同，所以**每个项目各框一份** ✓。\n"
            "（本项目还没框过时，会先用着老的那份全局值，框一次就归到本项目。）\n\n"
            "两个用途：\n"
            "  · 「在实时画面上叠地形图」得知道往画面的哪儿画；\n"
            "  · 来源选「从实时画面框选」时，还要靠它把面板裁出来喂给标定弹窗。\n\n"
            "先在「实时」页点开始、看到画面里的游戏小地图再回来框。\n"
            "只框**面板本身**：多框进来的血条/聊天栏会一起算进去，匹配分会掉下来。\n"
            "框完立刻存（不用等确定 ✓），当场和底图核对一次，匹配分写在下面那行。\n\n"
            "画面尺寸变了（换分辨率 / 改推流参数）要重框一次。")
        self.btn_mmap_crop.clicked.connect(safe_slot(self._pick_mmap_crop))
        crow.addWidget(self.btn_mmap_crop)
        self.lbl_mmap_crop = QLabel()
        self.lbl_mmap_crop.setStyleSheet("color: #5f6368;")
        self.lbl_mmap_crop.setWordWrap(True)
        crow.addWidget(self.lbl_mmap_crop, 1)
        root.addLayout(crow)
        self.lbl_crop_note = QLabel()
        self.lbl_crop_note.setStyleSheet("color: #5f6368;")
        self.lbl_crop_note.setWordWrap(True)
        root.addWidget(self.lbl_crop_note)
        self._refresh_crop_status()

        # ---- 「实测精度」（2026-09-27 用户要求：把核对做成一次点击）----
        # 它回答这一页最要紧的那个问题：**"我这份标定到底差多少世界像素？"** ——
        # 在这之前全仓库都问不出来（匹配分是无量纲的"像不像"、两点法的残差是**面板像素**、
        # 取证工具量的是黄点识别率 ✗）⇒ 只能靠"寻路看起来对不对"猜。
        # ⚠ **只读**：不改标定、不写任何文件（要改走「标定…」/ 双点标定 ✓）—— 所以文案是
        #   "实测"而不是"校准"：别让人以为点一下会改数据 ✗。
        # ⚠ **算法只有一份**：`perception.minimap.check_calib`（命令行工具
        #   `tools/mmap_calib_check.py` 走的是同一份 ✓，别在这儿再写一套 ✗）。
        # ⚠ 拿的面板是：来源=独立推流 ⇒ A 机那一帧（`mm.stream_panel` ✓，不用实时页）；
        #   来源=从实时画面 ⇒ 按本项目框选区域裁**原生帧**（`current_frame()`，
        #   身上没有检测框/视野虚线 ✓，见 docs/UI规范.md §8）。
        krow = QHBoxLayout()
        krow.setSpacing(8)
        self.btn_mmap_check = QPushButton("实测精度")
        self.btn_mmap_check.setObjectName("mmapCheck")
        self.btn_mmap_check.setToolTip(
            "量一遍**当前这份标定**差多少：拿一帧真画面 + 底图，用模板匹配当一把独立的\n"
            "尺子（同一个面板像素，两套几何映射到世界差多少），四个角取最坏那个。\n\n"
            "读数**一律世界像素**（决策：容差按实时像素定值、表述用世界坐标 ✓）：\n"
            "  · 偏差 ≈ 多少世界像素 / ≈ 多少实时像素 —— 判据 10 世界像素 ≈ 1 个实时像素，\n"
            "    和「坐标对齐误差范围」是同一把尺；\n"
            "  · **尺子匹配分**：低于 0.80 就说明「画面里那块钱不干净」（框选混进了血条/\n"
            "    聊天栏、被别的 UI 挡住、「显示方式」选错）⇒ 那个偏差数只能当参考。\n\n"
            "**只读，不改任何东西**（要改标定走「标定…」或双点标定）。\n"
            "前提：本项目框过小地图（来源=从实时画面时）、这条来源标过。")
        self.btn_mmap_check.clicked.connect(safe_slot(self._check_mmap_calib))
        krow.addWidget(self.btn_mmap_check)
        self.lbl_check_note = QLabel()
        self.lbl_check_note.setStyleSheet("color: #5f6368;")
        self.lbl_check_note.setWordWrap(True)
        krow.addWidget(self.lbl_check_note, 1)
        root.addLayout(krow)
        self.lbl_check_detail = QLabel()
        self.lbl_check_detail.setStyleSheet("color: #5f6368;")
        self.lbl_check_detail.setWordWrap(True)
        root.addWidget(self.lbl_check_detail)

        # ---- 子组：**攀爬参数**（2026-09-26 成组；2026-09-27 用户改名）----
        # 为什么单独成组：里面全是"上绳这一步"的参数（对齐的节奏 + 失败后怎么重来），
        # 和上面那些"寻路怎么走 / 跑多快"不是一回事 ✓（缩进 + 边框，一眼看出归属 ✓）。
        # ⚠ 2026-09-27 用户把组名从「攀爬失败保护」改成「**攀爬参数**」—— 组里现在既有
        #   "失败保护"，也有"对齐节奏"（`climb_align_gap_ms`），老名字已经装不下 ✓。
        # ---- 子组：**攀爬参数**（2026-09-26 成组；2026-09-27 用户改名）----
        grp_guard = QGroupBox("攀爬参数")
        gv = QVBoxLayout(grp_guard)
        gv.setSpacing(6)
        # 第一行：**对齐绳梯移动延迟(ms)**（用户 2026-09-27 加的参数）——
        # 它管的是流程里**最早**的一步（对齐绳的 x），所以排在最前 ✓
        row_gap = QHBoxLayout()
        row_gap.setSpacing(6)
        row_gap.addWidget(QLabel("对齐绳梯移动延迟(ms)"))
        self.sp_align_gap = NoWheelSpinBox()
        self.sp_align_gap.setRange(0, 3000)
        self.sp_align_gap.setSingleStep(10)
        self.sp_align_gap.setMinimumWidth(90)
        self.sp_align_gap.setValue(int(getattr(settings, "climb_align_gap_ms", 180) or 0))
        self.sp_align_gap.setToolTip(
            "**对齐绳梯的 x** 时，两次按下方向键之间至少要隔这么久（毫秒）。\n\n"
            "为什么要它：按住方向键在游戏里就是「一直走」，快到绳那儿必然冲过头、\n"
            "然后往回走 ⇒ 表现是「在绳两边来回抖」。所以进了「开始对齐绳梯x的距离」\n"
            "以内就改成**一下一下地点按**，这个参数是那两下之间**至少**等多久 ✓。\n\n"
            "0 = 不限制（想按就按，最激进）。默认 180 = 老行为（原来写死的点按周期）。\n"
            "调大 = 走得更碎、更不易过冲（代价：贴到绳上更慢）。\n\n"
            "⚠ 比那个距离**远**时仍然是**一口气按住走**（那本来就只有「一次按下」）；\n"
            "   它约束的是「松开过之后再按」的那一下 ✓。")
        self.sp_align_gap.valueChanged.connect(self._on_align_gap)
        row_gap.addWidget(self.sp_align_gap)
        row_gap.addStretch(1)
        gv.addLayout(row_gap)

        # 第二行：**开始对齐绳梯x的距离(px)**（用户 2026-09-27 加的参数）——
        # 和上面那条是**同一个步骤**（对齐绳的 x）的两个参数：这条决定"离多远开始点按"，
        # 上面那条决定"点按之间至少隔多久" ✓ 所以紧跟着它排，不另起一组 ✓。
        row_near = QHBoxLayout()
        row_near.setSpacing(6)
        row_near.addWidget(QLabel("开始对齐绳梯x的距离(px)"))
        self.sp_align_near = NoWheelSpinBox()
        self.sp_align_near.setRange(1, 2000)
        self.sp_align_near.setSingleStep(5)
        self.sp_align_near.setMinimumWidth(90)
        self.sp_align_near.setValue(
            int(getattr(settings, "climb_align_near_px", 20) or 20))
        self.sp_align_near.setToolTip(
            "**对齐绳梯的 x** 时，离绳还差这么多像素以内 ⇒ 改成**点按**（一下一下）。\n\n"
            "· 差得比它**远** ⇒ 一口气**按住**方向键（按住 = 一直走，走得快）；\n"
            "· 进到它**以内** ⇒ 点按（按一下、松一下），别冲过头 —— 上面那条\n"
            "  「对齐绳梯移动延迟(ms)」管的就是这两下之间至少隔多久 ✓。\n\n"
            "默认 20 = 老行为（原来写死的值）。**调大** = 更早进入「一下一下」的精细对齐，\n"
            "不容易冲过头（代价：贴到绳上慢一点）；调小 = 只在最后一小段才点按。\n\n"
            "⚠ 用户 2026-09-27 说 20 太近了（要能调大）—— 这一格就是那个数 ✓。")
        self.sp_align_near.valueChanged.connect(self._on_align_near)
        row_near.addWidget(self.sp_align_near)
        row_near.addStretch(1)
        gv.addLayout(row_near)

        # 第三行：失败后延迟激活时间
        row_retry = QHBoxLayout()
        row_retry.setSpacing(6)
        # 单位写在标签里（UI 规范 §9：不写进编辑框）
        row_retry.addWidget(QLabel("攀爬失败后延迟激活时间(s)"))
        self.sp_retry = NoWheelDoubleSpinBox()
        self.sp_retry.setRange(0.0, 30.0)
        self.sp_retry.setDecimals(1)
        self.sp_retry.setSingleStep(0.5)
        self.sp_retry.setMinimumWidth(90)
        self.sp_retry.setValue(float(getattr(settings, "climb_retry_delay_s", 1.0)))
        self.sp_retry.setToolTip(
            "上绳梯 / 下跳**失败后等多久**再重新激活（就是原来的「重新对齐再来一次」）。\n\n"
            "0 = 立即重来（老行为）。\n"
            "等一会儿的好处：失败那一下人往往还在原地、朝向也没变，立刻重来容易在同一处\n"
            "再歪一次；先站稳一小会儿再重来，成功率更高。\n\n"
            "⚠ 等待期间**不按键**（角色站着不动）；如果那时已经站在目标集合里，\n"
            "任务会**直接收工**，不再重来。")
        self.sp_retry.valueChanged.connect(self._on_retry_delay)
        row_retry.addWidget(self.sp_retry)
        row_retry.addStretch(1)
        gv.addLayout(row_retry)

        # 第四行：**延迟增量**（缩进 ⇒ 看起来是上面那个的子参数 ✓）
        row_inc = QHBoxLayout()
        row_inc.setSpacing(6)
        row_inc.addSpacing(18)                    # ← 缩进
        row_inc.addWidget(QLabel("延迟增量(s)"))
        self.sp_retry_inc = NoWheelDoubleSpinBox()
        self.sp_retry_inc.setRange(0.0, 30.0)
        self.sp_retry_inc.setDecimals(1)
        self.sp_retry_inc.setSingleStep(0.5)
        self.sp_retry_inc.setMinimumWidth(90)
        self.sp_retry_inc.setValue(
            float(getattr(settings, "climb_retry_delay_inc_s", 1.0)))
        self.sp_retry_inc.setToolTip(
            "**每次失败**，下一次的等待就多出这么多（秒）：\n"
            "  第 1 次失败等「延迟激活时间」；第 2 次等「激活 + 增量×1」；\n"
            "  第 3 次等「激活 + 增量×2」……\n\n"
            "为什么要：上绳失败常常就是「这次歪了」—— 越往后越该多稳一会儿再重来。\n"
            "0 = 每次都用同一个等待（老行为）。\n"
            "⚠ 总次数仍由「上绳最多试几次」那道上限卡着（不会无限重来 ✗）。")
        self.sp_retry_inc.valueChanged.connect(self._on_retry_delay_inc)
        row_inc.addWidget(self.sp_retry_inc)
        row_inc.addStretch(1)
        gv.addLayout(row_inc)

        root.addWidget(grp_guard)

        # ---- 「**移动操作尝试间隔(ms)**」（2026-09-27 用户要求：改名 + **搬出**子组）----
        # 原名「爬不动时先补按 ↑ 观察(s)」，原来挂在「攀爬参数」子组里（`gv`），现在：
        #   · **改名**：它不再是"爬不动"专用 —— 它是**移动操作**的通用重试间隔；
        #   · **搬出来**：放在外面的「寻路配置」组里（`root` 还是这张卡的内布局 ✓），
        #     紧挨着子组之前 ⇒ 一眼看出它是**两个通行方式共用**的，不属于"上绳这一步" ✗；
        #   · **单位从秒改成毫秒（整数）**：毫秒没有小数意义，而且整数才能用 `NoWheelSpinBox`
        #     （`docs/UI规范.md` §9；`tools/check_ui.py` 的 `check_ms_is_integer` 钉着 ✓）。
        # 它管的两件事（同一个语义："一段移动操作发出去，隔多久没看到预期变化就补发/重试"）：
        #   · **爬绳**：y 不再变好 ⇒ 先**补按一次 ↑**，观察这么久还不好才判失败 ✓
        #     （用户口径"在绳上、不偏离、↑ 没松就一定能上" ⇒ "不动"首先是"键没按上" ✗）；
        #   · **下跳**：按住 ↓ + 点按跳之后这么久 **Y 还没动** ⇒ **补发一个「松开 ↓」**，
        #     再重新按住 ↓ + 点按跳 ✓（不"补按 ↓"而"先松开"：`KeyState` 只在键集**变化**
        #     时才发键，一直按着再按一次，对面根本收不到新的按下 ✗）。
        row_re = QHBoxLayout()
        row_re.setSpacing(6)
        row_re.addWidget(QLabel("移动操作尝试间隔(ms)"))
        self.sp_retry_gap = NoWheelSpinBox()
        self.sp_retry_gap.setRange(0, 60000)
        self.sp_retry_gap.setSingleStep(100)
        self.sp_retry_gap.setMinimumWidth(90)
        self.sp_retry_gap.setValue(int(getattr(settings, "move_retry_ms", 3000) or 0))
        self.sp_retry_gap.setToolTip(
            "一段**移动操作**发出去之后，隔多久**看不到预期变化**就补发/重试一遍"
            "（毫秒）。它管两件事：\n\n"
            "· **爬绳**：y 不再变好 ⇒ 先**补按一次 ↑**，观察这么久还不好才判失败。\n"
            "  为什么要补按：`KeyState` 只在键集**变化**时才发键 ⇒ 对面（中继/固件）把键\n"
            "  丢了，本机是**不知道**的 ✗ —— 表现是「人停在绳上、任务却自信地按着 ↑」。\n"
            "  用户口径：在绳上、不偏离、↑ 没松就一定能上去 ⇒「不动」首先是「键没按上」✗。\n"
            "· **下跳**：按住 ↓ + 点按跳之后这么久 **Y 还没动** ⇒ **补发一个「松开 ↓」**，\n"
            "  然后重新按住 ↓ + 点按跳（一直按着 ↓ 再「按一次」，对面收不到新的按下 ✗）。\n\n"
            "0 = 不重试（爬绳直接判失败 / 下跳一直按着等）—— 老行为。\n"
            "调大 = 更愿意把「没反应」当成按键问题（代价：真卡住时要多等这么久才收手）。")
        self.sp_retry_gap.valueChanged.connect(self._on_retry_gap)
        row_re.addWidget(self.sp_retry_gap)
        row_re.addStretch(1)
        root.addLayout(row_re)

        # ⛔ 「**前往重下间隔(s)**」这一格 **2026-09-28 移除了**（用户要求：它搬进**每一个
        #   战斗区域项**、改名「**区域查询CD(s)**」✓）⇒ 现在在「**决策参数 → 路线脚本 →
        #   战斗区域**」那个弹窗里**逐项**配 ✓（见 `DecisionSettings.battle_zones` ✓）。
        #   ⚠ **别再把这一格加回来** ✗（一个全局值 + 每项一个值同时存在，取值就成了掷骰子 ✗）。

        # ---- 子组：**物理参数**（用户 2026-09-27 要求）----
        # 为什么不塞进「攀爬参数」：这里放的是**跟移动物理有关**的量，而且**是临时的** ——
        # 用户原话："先配一个起跳距离测试用，后续物理相关逻辑实现后，移除起跳距离参数，
        # 增加角色移动速度、跳跃力的配置后台换算" ✓ ⇒ 那套换算做出来之后，这一组里的
        # 「起跳距离」就该**删掉**（所以它不该混进"上绳这一步"的常驻参数里 ✗）。
        # 现在只有一项，但**两个地方**吃它（2026-09-27 起）：
        #   ① 「跳(jump)」这条通行方式：从离目标 foothold 多远开始按跳 ✓；
        #   ② **上绳的「斜跳」**（用户 2026-09-27 要求）：它是斜跳区间的**上界** ——
        #      「开始对齐绳梯x的距离」< 离绳 x 距离 ≤ 它 ⇒ 按住朝绳方向 + 起跳，斜着上绳 ✓
        #      （≤ 那个近距就还是老的"对齐 x 原地起跳"⇒ **0 = 完全老行为** ✓）。
        grp_phys = QGroupBox("物理参数")
        pv = QVBoxLayout(grp_phys)
        pv.setSpacing(6)
        row_js = QHBoxLayout()
        row_js.setSpacing(6)
        row_js.addWidget(QLabel("起跳距离(px)"))
        self.sp_jump_start = NoWheelSpinBox()
        self.sp_jump_start.setRange(0, 2000)
        self.sp_jump_start.setSingleStep(5)
        self.sp_jump_start.setMinimumWidth(90)
        self.sp_jump_start.setValue(int(getattr(settings, "jump_start_px", 0) or 0))
        self.sp_jump_start.setToolTip(
            "「**跳(jump)**」这条通行方式**从离目标 foothold 多远开始按跳**（像素）。\n\n"
            "⚠ 它还管**上绳的「斜跳」**（2026-09-27 用户要求）：离绳子还远的时候\n"
            "不用先站到绳正下方 —— 当「**开始对齐绳梯x的距离**」< 离绳的 x 距离 ≤\n"
            "**这一格**，而且人正站在绳要抓的那块 foothold 上 ⇒ 直接**按住朝绳的方向\n"
            "+ 起跳**斜着过去 ✓（一轮只斜跳一次，跳完继续朝绳走）。\n"
            "距离 ≤「开始对齐绳梯x的距离」时，仍然是老流程：**点按对齐 → 站住 → 原地跳** ✓；\n"
            "**0 = 斜跳整个关掉**（默认值 ⇒ 与加这个功能之前完全一样 ✓）。\n\n"
            "口径：角色 x 到**最近那条目标 foothold 的 x 范围**（也就是最近端点）的距离\n"
            "≤ 它 ⇒ 按跳 ✓；**0 = 只有 x 真的进了目标 foothold 范围才跳**（默认值 ✓，\n"
            "不拍脑袋补数 —— 现场试出来该提前多少再填 ✓）。\n\n"
            "⚠ **临时参数**（用户 2026-09-27 定的）：先拿它把「跳」调通；等物理算得出来\n"
            "（角色移动速度 + 跳跃力 ⇒ 该在离平台边缘多远起跳），它**会被删掉**、\n"
            "换成后台换算 ✓ —— 所以别往它身上挂别的东西 ✗。\n\n"
            "⚠ 和「追击起跳」（决策参数 → 战斗参数 → 攻击；战斗里追怪用的那个跳）\n"
            "**不是一回事** ✓。")
        self.sp_jump_start.valueChanged.connect(self._on_jump_start)
        row_js.addWidget(self.sp_jump_start)
        row_js.addStretch(1)
        pv.addLayout(row_js)
        root.addWidget(grp_phys)

        # ---- 子组：**路线规划**（2026-10-01 用户要求："把「编辑战斗区域」移到路线识别页签
        #      寻路配置的新子组『路线规划』"）----
        #
        # 为什么它该在这一页、而不是决策参数页（用户那句"为什么龙族打猎场不能编辑战斗
        # 区域"的根因就在这 ✓）：
        #   · 战斗区域**每一项都是地图里的东西** —— `set` 是**本图已注册的集合名**、
        #     `idle_foothold` 是**本图的 foothold 编号**、`fight_dst` 又是集合名 ⇒
        #     只有本页（持有地图 id）能**当场**问出"这张图有哪些集合"；
        #   · 原来挂在决策参数页 ⇒ 候选集合名只能由本页在**开项目那一刻推一次**
        #     （`main_window._bind_cards` → `player_panel.set_zone_sets`）⇒ 那张图的
        #     地形/集合若是**会话中途才导出来**的，那份候选就是**一张空表、用到关掉工作台**
        #     ⇒ 「添加」里一个集合都没有 = "不能编辑"（龙族打猎场正是这样 ✗）。
        #   ⇒ 搬到本页后候选**每次点开现读**（`_bz_candidate_names` ✓），不再看推送时机。
        grp_plan = QGroupBox("路线规划")
        plv = QVBoxLayout(grp_plan)
        plv.setSpacing(6)
        # 这一行现在只有：标签 + 「编辑」按钮 + **只读的已配摘要**（添加 / 删除 / 改参数
        # **全在弹窗里** ✓ —— 行内再来一套按钮就是重复 ✗ 用户 2026-09-28 第 1 条 ✓）。
        self._bz_rows = QVBoxLayout()
        self._bz_rows.setSpacing(4)
        self._bz_names = []                 # 当前显示的名字（顺序 = 界面顺序 ✓）
        self.btn_battle_zone_edit = QPushButton("编辑战斗区域")
        self.btn_battle_zone_edit.setToolTip(
            "打开「编辑战斗区域」：**添加 / 删除 / 双击一行改参数** ✓\n"
            "每一项可以勾「可以战斗」—— 那就是旧的「限制战斗区域」✓\n\n"
            "候选集合 = **当前这张图已注册的集合**（每次点开现读 ✓ 2026-10-01）。")
        self.btn_battle_zone_edit.clicked.connect(
            safe_slot(self._on_battle_zone_edit_clicked))
        _bz_top = QHBoxLayout()
        _bz_top.setSpacing(6)
        _bz_top.addWidget(self.btn_battle_zone_edit)
        _bz_top.addStretch(1)
        plv.addLayout(_bz_top)
        plv.addLayout(self._bz_rows)
        plv.addWidget(self._bz_hint())
        root.addWidget(grp_plan)

        # ---- ② 小地图定位（寻路用；方式由你选，程序不猜）----
        root = card("小地图定位")

        # 面板上只留「参数 + 状态」；怎么判断选哪种、标定怎么跑，都放 tooltip
        # （docs/UI规范.md：正文别塞说明，细节交给 tooltip）
        #
        # **分两行**（docs/UI规范.md §4：一行只放一个参数组，放不下就另起一行，
        # 不许在行尾继续 addWidget 堆）：
        #   第一行「这是哪张图 · 怎么显示」
        #   第二行「画面从哪来 · 框选/标定动作」
        # 要加新参数时先想它属于哪一组，再决定放哪一行 —— 一路往右加的话，
        # 窗口一窄就是所有标签一起被压没，而且再也说不清哪几个是一组。
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("当前地图"))
        self.lbl_mmap = QLabel("—")
        row.addWidget(self.lbl_mmap)
        row.addSpacing(12)
        row.addWidget(QLabel("显示方式"))
        self.cmb_mmap_mode = NoWheelComboBox()
        # 名字按「你在小地图上看到的是什么」取：整张都在 = 全局；
        # 只看到一块、跟着人滑动 = 局部。（fit/crop 是实现里的叫法，别摆到界面上）
        self.cmb_mmap_mode.addItem("全局小地图", mm.MODE_FIT)
        self.cmb_mmap_mode.addItem("局部小地图", mm.MODE_CROP)
        self.cmb_mmap_mode.setToolTip(
            "小地图面板怎么显示底图，两种方式每张图可能不一样，所以由你选"
            "（每张图存一份）。\n\n"
            "怎么判断：进游戏左右走两步，盯着小地图看 ——\n"
            "  · 地形图**不动**，只有黄点在移动 → 全局小地图\n"
            "  · 地形图**跟着你滑动**（黄点差不多在中间） → 局部小地图")
        self.cmb_mmap_mode.currentIndexChanged.connect(self._on_mmap_mode)
        row.addWidget(self.cmb_mmap_mode)
        # 第一行到这儿结束。**别在这行后面继续加** —— 加上去的东西会一路往右堆，
        # 窗口一窄先被压没的就是标签（§4 就是这么被违反的）。要加先归组。
        row.addStretch(1)
        root.addLayout(row)

        # ---- 第二行：画面从哪来 + 框选/标定动作 ----
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        # ⚠ 「小地图来源」**2026-10-03 已搬去「寻路配置」卡**（「当前地图」那行下面 ✓
        #   用户原话："「小地图定位」的来源、黄点跟踪参数挪到寻路配置最上方"✓）——
        #   控件的**定义也跟着走了** ✓（别在这儿再建一个 ✗ 见那张卡里的长注释：
        #   "布局借来借去会让 combo 丢父 ⇒ `mapTo` 访问违例崩进程"✓）。

        # ---- 「坐标系偏移」(x, y)：算出来的世界坐标**加上**它（2026-09-26 用户要求）----
        # 单独一行（docs/UI规范.md §4：一行只放一个参数组）；按**地图 id + 来源**存在
        # `datasets/map/<id>.mapcalib.json` 的 `sources.<来源>.world_offset`，
        # 和 scale/offset 同一份（面板与实时线程天然一致）。
        row_off = QHBoxLayout()
        row_off.setSpacing(6)
        row_off.addWidget(QLabel("坐标系偏移"))
        self._sp_off = []
        for _lbl in ("x", "y"):
            row_off.addWidget(QLabel(_lbl))
            sp = NoWheelSpinBox()
            sp.setRange(-5000, 5000)
            sp.setMinimumWidth(84)
            sp.setToolTip(
                "算出来的世界坐标会**加上**这个向量 —— 用来把"
                "「黄点**下沿（脚底）**」和你要的玩家原点（脚下那一点）对齐。\n\n"
                "怎么量：走到一个你知道确切世界坐标的点，看那行读数差多少，"
                "把差值填进来（读数是 算出来的 + 这里）。\n"
                "⚠ 2026-09-27 起标定的锚点是**下沿**（双点标定采样的也是下沿）⇒"
                "这一项是**残差**、正常应该接近 (0, 0)。\n"
                "   老那份 (7, 33) 是按**重心**量的，别照抄（那是双重补偿，读数会偏 33 世界像素）。\n\n"
                "它只影响读数与寻路判断，**不碰标定**（面板↔底图那套照旧）。")
            sp.valueChanged.connect(self._on_world_offset)
            row_off.addWidget(sp)
            self._sp_off.append(sp)
        row_off.addStretch(1)
        root.addLayout(row_off)

        # ⚠ 「框选小地图」**2026-09-27 用户定：留在本页、放在「寻路配置」里**
        #   （「寻路编辑器」正下方 ✓）。它记的是「小地图面板在实时画面里的哪个位置」，
        #   而那**按项目（地图）各一份** —— 不同地图的小地图面板尺寸/位置完全不同，
        #   放"设置"（全局一份）里换个项目就是错的 ✗（用户 2026-09-27 现场）。
        #   ⇒ 和"叠不叠、浓淡多少"不是一类东西：那两个才是本机全局的外观偏好 ✓。
        #   这一页另外还留**现场状态**（框没框 / 叠图画没画 ✓）。

        # 标定做成弹窗（照「手动目测标定尺度」那套交互）：量出来的是
        # 「面板 ↔ 底图」的缩放/偏移，而判据是**看得见**的重合程度 ——
        # 摆在这块面板里放不下，也不该让人去命令行跑一遍。
        self.btn_mmap_calib = QPushButton("标定…")
        self.btn_mmap_calib.setToolTip(
            "量这张图的「面板 ↔ 底图」换算（存 datasets/map/<id>.mapcalib.json）。\n\n"
            "弹窗里：A 机推来的实时面板 + 半透明底图叠在一起 ——\n"
            "「自动定位」用模板匹配量（准）；匹配不上时可以手动拖动对齐，\n"
            "目测重合后「保存标定」。\n\n"
            "前置：A 机的「被控机部署台 → 小地图推流」已经在跑。")
        self.btn_mmap_calib.clicked.connect(self._open_mmap_calib)
        row2.addWidget(self.btn_mmap_calib)
        # 第二行到这儿结束（同上：要加先归组，别往右续）
        row2.addStretch(1)
        root.addLayout(row2)

        # ---- 第三行：「双点标定」单人一行 ----
        # （用户 2026-09-27 要求加这个按钮）填两对坐标把几何**解出来** —— 和上面那个
        # 「标定…」是**同一份**标定的两条量法（互相补：一个量得准、一个看得见）。
        # 为什么要单开一个：真实面板带标题栏/边框时模板匹配会偏（合成面板实测 3%，
        # 最远那个角 ≈ 93 世界像素），而两点法是解析解、还能报"两轴一致不一致"。
        # ⚠ **别把它并回 row2**：并进去面板最小宽度 580 → 646，`t_track_rows_layout`
        #   当场红（UI 规范 §4 量出来的那条线），窗口一窄就是标签先被压没 ✗。
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        self.btn_mmap_two = QPushButton("双点标定")
        self.btn_mmap_two.setToolTip(
            "走两个地方各采一次，把「面板 ↔ 底图」的缩放与偏移解出来"
            "（存 datasets/map/<id>.mapcalib.json，和「标定…」同一份）。\n\n"
            "怎么做：\n"
            "  1. 走到一个你知道确切位置的地方（寻路编辑器里悬停能读到「鼠标 (x, y)」）；\n"
            "  2. 弹窗左边填那个位置在地图上的坐标（默认世界坐标）、右边点"
            "「实时填入当前黄点脚底」；\n"
            "  3. 换个地方再来一次（A / B 两行，别在同一行或同一列）；\n"
            "  4. 「算一算」→「保存标定」。\n\n"
            "它比「标定…」准的地方：解出来的缩放是连续值（模板匹配只能从 1.0/1.25/"
            "1.5/… 这种候选里挑），而且能表达 x/y 两轴各自的缩放。\n\n"
            "⚠ 别把「坐标系偏移」清零：定位读的是黄点重心、这里填的是脚底，"
            "那半个点高的差正是它在补的。")
        self.btn_mmap_two.clicked.connect(self._open_two_point_calib)
        row3.addWidget(self.btn_mmap_two)
        row3.addStretch(1)
        # 第三行到这儿结束。**代码顺序 = 视觉顺序**（UI 规范 §4）：row2 那一句必须
        # 排在 row3 之前，不然「双点标定」会跑到「标定…」上面去。
        root.addLayout(row3)

        # ---- 行3/4：标记跟踪（黄点的四个容差）----
        # 这一组**分两行**（同「小地图定位」那一组，docs/UI规范.md §4：一行放不下
        # 就另起一行，别顺着行尾续 —— 四个框挤一行时面板最小宽度会从 521 涨到 714，
        # 窗口一窄标签就先被压没）：
        #   行3「漏检怎么办」：沿用窗口 / 外推上限
        #   行4「怎么搜、怎么判噪声」：搜索半径 / 跳变上限
        # 四个都是"这一帧的黄点要不要信"：小 → 反应快、抗噪差；大 → 稳、但会跟丢。
        # ⚠ 存哪儿（2026-10-03 改）：**按地图 id 存**（`core/route_cfg.py` ✓
        #   `datasets/map/<id>.route.json` ✓）—— 以前一律写 `config/live.yaml` ✗；
        #   那份现在只是"**新图第一次打开时的播种值**"（用户选的 ① ✓）。
        #   为什么该按图：面板尺寸/压缩比逐图不同 ⇒ "多大的跳变算噪声"本来就该逐图调 ✓。
        #   （布局上也已搬进「寻路配置」卡顶部 ✓ 用户 2026-10-03 ✓ 见 `root_pf.insertLayout` ✓。）
        self._sp_track = {}          # 配置键名 → 输入框（键名以 mm.TRACK_KEYS 为准）

        def _track_spin(key, name, lo, hi, tip):
            """造一个容差输入框：滚轮不改值（UI规范 §5）、说明进 tooltip。

            单位**不写进框里**（UI 规范 §9）：写在旁边那个 QLabel 上
            （「沿用(ms)」「跳变上限(px)」这种）。
            """
            sp = NoWheelSpinBox()
            sp.setRange(lo, hi)
            sp.setFixedWidth(84)
            sp.setToolTip("%s：\n%s" % (name, tip))
            sp.valueChanged.connect(self._on_mmap_track)
            self._sp_track[key] = sp
            return sp

        # 行3：漏检怎么办（沿用 + 外推）
        row4 = QHBoxLayout()
        row4.setSpacing(6)
        row4.addWidget(QLabel("标记跟踪"))
        row4.addWidget(QLabel("沿用(ms)"))
        row4.addWidget(_track_spin(
            "mmap_hold_ms", "沿用", 0, 5000,
            "黄点这一拍认不出时，**沿用上一帧位置**多久（毫秒）。\n"
            "窗口内位置会按速度外推一点点；窗口过了才当真跟丢。\n"
            "0 = 不用这条（认不出就直接说认不出）。"))
        row4.addWidget(QLabel("外推上限(px)"))
        row4.addWidget(_track_spin(
            "mmap_ghost_shift", "外推上限", 0, 100,
            "沿用期间位置最多往外推这么多像素（面板像素，1 px ≈ 16 世界像素）。"))
        row4.addStretch(1)
        # ⭐ **插进「寻路配置」卡**（用户 2026-10-03 ✓）：第 2 位 = 「来源」那行下面 ✓
        root_pf.insertLayout(2, row4)

        # 行4：怎么搜、怎么判噪声（搜索半径 + 跳变上限）。**别往上面那行续加**。
        row5 = QHBoxLayout()
        row5.setSpacing(6)
        row5.addWidget(QLabel(""))
        row5.addWidget(QLabel("搜索半径(px)"))
        row5.addWidget(_track_spin(
            "mmap_roi_pad", "搜索半径", 0, 200,
            "有上一帧位置时，只在它周围这么大一块里找（像素）。\n"
            "实际半径还会自动放大到能盖住黄点本身（面板大、点也大）。\n"
            "越小越快；人跑得快时找不到会自动退回全画面重找。"))
        row5.addWidget(QLabel("跳变上限(px)"))
        row5.addWidget(_track_spin(
            "mmap_max_jump", "跳变上限", 0, 500,
            "两拍之间位置跳超过这么多像素就当噪声：丢掉位置、下一拍重捕。\n"
            "调小 = 更不信突变（适合跟丢少、噪声多的画面）；0 = 不判跳变。"))
        row5.addStretch(1)
        # ⭐ 同上（第 3 位 = 「沿用/外推」那行下面 ✓ 两行**不许挤一行** —— UI 规范 §4 ✓
        #   `t_track_rows_layout` 按坐标量着 ✓）
        root_pf.insertLayout(3, row5)

        # ---- 世界坐标那行 ----
        # 紧挨着「小地图面板」那一组（原来「框选小地图」按钮就在这行上面；按钮
        # 2026-09-27 搬去设置了 ✓）：它读的就是画面里框出来的那块面板，
        # 挨着放才看得出"这行数是从那块画面算出来的"。
        # 只在勾上「在实时画面上叠地形图」时才显示（要读数就得先看那块面板对不对）。
        self.lbl_mmap_world = QLabel()
        self.lbl_mmap_world.setStyleSheet("color: #80868b;")
        self.lbl_mmap_world.setWordWrap(True)
        self.lbl_mmap_world.setVisible(False)
        root.addWidget(self.lbl_mmap_world)
        #: ⭐⭐ 「**小地图链路快不快**」那一行（用户 2026-10-03 ✓ 原话："我能在路线识别页签→
        #:   地形图看到验证结果吗？"）—— 以前这些数只有 `perf.log` 里有 ✗：
        #:     `收帧 58.2 fps ・ 丢帧 12（本次 +3）・ 定位 0.5 ms`
        #:   收帧 = `MiniMapClient.fps`（**真实**收帧率 ⇒ "A 机到底推了多少" ✓）；
        #:   丢帧 = `n_drop` 增量（没被取走就被覆盖 ✓ 持续 >0 ⇒ 消费侧比推流慢 ✓）；
        #:   定位 = 这一拍 `locator.update` 的耗时（`locate_ms` 的界面版 ✓）。
        #:   ⚠ 它和上面那行「玩家世界坐标」**分工不同**：那行回答"**对不对**"（几何/标定 ✓），
        #:     这行回答"**快不快/新不新**"（链路 ✓）—— 地形图那层叠加看的是前者 ✓。
        #:   ⚠ 只在**收流**来源有意义（live 来源没有这条推流链 ⇒ 明说一句，别留旧数骗人 ✗）。
        self.lbl_mmap_rate = QLabel()
        self.lbl_mmap_rate.setStyleSheet("color: #80868b;")
        self.lbl_mmap_rate.setWordWrap(True)
        root.addWidget(self.lbl_mmap_rate)
        #: 玩家定位器：面板画面 → 世界坐标 → 在哪条段（perception.minimap）。
        #: **和实时线程各持一份**：那一份写进 WorldState（决策用），这一份只做读数；
        #: 共用一份会让两边的跨帧跟踪互相打乱（同一套跳变判据被两边各推一次）。
        self._locator = mm.PlayerLocator()
        #: 玩家世界坐标的**最近一次读数**（`_tick_world` 写）——「命令前往」择路要用它：
        #: 同一对集合有多条绳时**挑离我最近的那根**（`decision.route.pick_edge` ✓）。
        #: 读不到黄点 / 实时没跑的时候是 None ⇒ 择路退回文件顺序（**不猜** ✗）。
        self._world_at = None
        self._world_note = ""           # 这行字（也贴到画面里那块框下面）
        self._ov_pix = None             # 叠图源图缓存（见 _overlay_pix）
        #: 叠图**当前是按哪个显示区画的**（局部小地图：这一拍跟出来的那个 ✓）。
        #: 用来"**变了才重画**" —— 250ms 一拍，显示区没动就别白重画一遍 ✓（见 _tick_world）。
        self._ov_view = None
        #: ⭐ 「小地图链路」那一行的两个账（见 `_refresh_mmap_rate` ✓）：
        #:   `_rate_drop_last` = 上次记的累计丢帧（算**增量** ✓）；`_loc_ms_last` = 这一拍定位耗时。
        self._rate_drop_last = None
        self._loc_ms_last = None
        #: ⭐ 「当前这张图」的小地图框选区域（2026-10-03 ✓）：`_apply_route_cfg` 灌进来；
        #:   `None` = 还没灌过（没选图 / 老会话）⇒ `_mmap_crop()` 走**原来的口径**（项目优先 ✓）。
        self._crop_override = None
        #: ⭐ 「当前这张图」的小地图来源（2026-10-03 ✓）：`_apply_route_cfg` 灌进来；
        #:   `None` = 还没灌过 ⇒ `_mmap_src()` 走**原来的口径**（读全局 live.yaml ✓）。
        self._src_override = None
        self._world_timer = QTimer(self)
        self._world_timer.setInterval(250)      # 4 次/秒：够看，又不占主线程
        self._world_timer.timeout.connect(self._tick_world)

        # 「地形图里叠实时小地图」那条的节拍（用户 2026-09-29 第 ⑦ 条 ✓）：单开一个定时器，
        # 因为它跟"画面上那层叠图"（`mmap_draw`）是**两件事** —— 那个关着也该能用这个 ✓。
        self._live_map_timer = QTimer(self)
        self._live_map_timer.setInterval(250)   # 同上：4 次/秒够看 ✓
        self._live_map_timer.timeout.connect(safe_slot(self._live_map_tick))
        #: 那一层的**透明度**（% ；用户 2026-09-29 追加："需要加个参数'透明度'"✓）。
        #: 落在 `config/live.yaml`（本机外观偏好，和 `mmap_draw` 同一处 ✓）。
        self._live_alpha = int(load_live().get("live_map_alpha", 75))
        #: ⭐ **显示类型**（用户 2026-10-06 ✓ 从"一个勾选框"升级成三选一 ✓）：
        #: `terrain` 只看地形图 / `live` 叠实时小地图（老行为 ✓）/ `diff` **像素差分地图**
        #: （面板 − 底图那一块，看"底图上没有的东西" ✓ 见 `mm.diff_panel_vs_canvas` ✓）。
        self._live_disp = norm_live_view(load_live().get("live_map_view"))
        #: ⭐ **这一层自己的「显示区」跟踪**（局部小地图 ✓ 用户 2026-09-29 追加："实时小地图的
        #: 位置没有跟着我的移动变化"✗）—— 以前它借的是**画面那层叠图**跟出来的 `_ov_view`
        #: （见 `_tick_world`），而那条路要求"叠图开着 + 世界坐标那行算得出来"才更新
        #: ⇒ 认不出黄点时它**根本不更新** ⇒ 面板钉在标定那一刻的位置上 ✗✗。
        #: 现在这一层自己跟：它手里**本来就有这一拍的面板**（`_live_map_tick` 每 250ms 取一张 ✓）
        #: ⇒ 不依赖别处的开关与状态 ✓（跟不住才退回 `_ov_view` / 标定里那份 ✓）。
        self._live_view_track = mm.CropViewTracker()

        # ---- 叠图这一行：**只留状态**（开关与浓淡已搬到 设置 → 界面）----
        # 为什么搬走（用户 2026-09-26 要求）："叠不叠、浓淡多少"是**外观偏好**，归设置窗口 ✓；
        # 这一页只留**现场**要盯的：画没画、画在哪儿、没画卡在哪一步 ✓。
        # ⚠ 但**读取与绘制仍在本面板**（`_mmap_draw_on()` / `_refresh_overlay()` ✓）——
        #   实时画面上那一层是它画的；设置窗口只改配置，改完由主窗口叫一句
        #   `apply_overlay_settings()` ✓（见那个方法）。
        # ⚠ 还要记住：这一层是**纯显示层**（只在 Qt 那边往缩小后的画面上补一层，
        #   numpy 帧一个字节不改 ✓）—— 烘进帧会让标定弹窗拿叠图和它自己匹配 ✗。
        self.lbl_mmap_draw = QLabel()
        self.lbl_mmap_draw.setStyleSheet("color: #80868b;")
        self.lbl_mmap_draw.setWordWrap(True)
        root.addWidget(self.lbl_mmap_draw)

        self.lbl_mmap_hint = QLabel()
        self.lbl_mmap_hint.setStyleSheet("color: #80868b;")
        self.lbl_mmap_hint.setWordWrap(True)
        root.addWidget(self.lbl_mmap_hint)

        # ---- ③ 地形图（看得见才好判断小地图定位对不对）----
        # ⚠ **常驻**（`into=self.maplayout` = 加在滚动区**外面**的那一栏 ✓）：这张卡片里有
        #   `ImageCanvas`（`QGraphicsView`）—— 把它放进滚动区实测会让工作台退出时偶发
        #   0xC0000005（8/8 崩 ✗），换掉画布立刻好（8/8 ✓）⇒ 见 `__init__` 那段说明 ✓。
        # 用户 2026-09-27："寻路配置的地形图……太占位置了" ⇒ 空间由**上面的分隔条**决定
        #   （拖它就能把这一栏压小 ✓ 见 `__init__`）；卡片自身的两项交互也写进 tooltip，
        #   免得用户不知道（画布**早就**支持滚轮缩放 / 中键平移 / 双击适应 ✓）。
        root = card("地形图", stretch=1, into=self.maplayout,
                    tip="地形图。上面那条**分隔条可以拖** ⇒ 决定这一栏占多高（太占位置就往上拖）✓\n\n"
                        "画布里：**滚轮** = 缩放 / **按住中键拖** = 平移 / **双击** = 适应窗口 ✓。")

        self.lbl_map_img = QLabel()
        self.lbl_map_img.setStyleSheet("color: #80868b;")
        self.lbl_map_img.setWordWrap(True)
        root.addWidget(self.lbl_map_img)

        # ⭐⭐ **「显示类型」**（用户 2026-10-06 ✓ 原话：把「叠加实时小地图」改成「显示类型」
        #   坐在里面：**仅地形图 / 实时小地图 / 像素差分地图** ✓ —— 起因是他那天问的
        #   "是不是颜色相减"，想亲眼看一眼"面板减底图"长什么样 ✓）：
        #     · **实时小地图**（= 老行为 ✓ 原来那个勾选框勾上）：把那块面板**按标定摆到
        #       地形图上**（`ImageCanvas.set_live_patch` ✓）—— "面板里画的是什么"和"它该在
        #       地图哪一块"**在同一张图上重叠**，放大（滚轮）一看就知道标定 / 显示区跟踪
        #       对不对 ✓；
        #     · **像素差分地图**：面板 **减** 底图上它盖住的那一块
        #       （`mm.diff_panel_vs_canvas` ✓）⇒ 差出来的就是"**底图上没有的东西**"
        #       （玩家点 / 实时元素 ✓）。
        #   ⚠ 底图始终由你自己缩放平移（滚轮 / 中键拖 ✓），这里只管"叠什么" ✓。
        row_live = QHBoxLayout()
        row_live.setSpacing(6)
        row_live.addWidget(QLabel("显示类型"))
        self.cmb_live_view = NoWheelComboBox()
        for _zh, _k in LIVE_VIEWS:
            self.cmb_live_view.addItem(_zh, _k)
        self.cmb_live_view.setMinimumWidth(120)
        self.cmb_live_view.setToolTip(
            "地形图上叠什么：\n"
            "  实时小地图 —— 把当前那块小地图面板按标定摆到这张地形图上（半透明），"
            "随时跟着滚动 ✓。滚轮放大到那一带：面板里的地形和底图处处重合就是对的；"
            "整块偏、或者越走越偏，就是标定或「显示区跟踪」的问题 ✓。\n"
            "  像素差分地图 —— 面板 减 底图上它盖住的那一块，专门看「底图上没有的东西」"
            "（玩家点 / 其它实时元素 ✓）。差分先减掉整幅的背景档再放大，"
            "所以整幅都偏一点时也不至于糊成一片 ✓。\n"
            "  仅地形图 —— 不叠，只看底图 ✓。\n\n"
            "⚠ 差分只是「看」的：不参与定位、不写盘 ✓。\n"
            "⚠ 底图是解码出来的地形画布、和游戏里那张小地图不一定同源 ⇒ "
            "差分的背景档偏大就说明这张图上它只能当辅助（右边那行会把数字报出来 ✓）。\n\n"
            "⚠ 面板从哪来跟「小地图来源」走：收流用 A 机那一口（画质好），"
            "从实时画面则按本项目的框选区域裁。")
        # ⚠ `setCurrentIndex` **必须在 `connect` 之前**（同老勾选框那条理由 ✓）：
        #   不然构造期就触一次 `currentIndexChanged` ⇒ ① 把 250ms 的取帧节拍在"这一页
        #   还没显示"的时候就开起来 ✗ ② 顺手往 `live.yaml` 写一次盘 ✗。
        self.cmb_live_view.setCurrentIndex(
            max(0, self.cmb_live_view.findData(self._live_disp)))
        self.cmb_live_view.currentIndexChanged.connect(
            safe_slot(self._on_live_view))
        row_live.addWidget(self.cmb_live_view)
        # ⭐ **透明度**（用户 2026-09-29 追加："需要加个参数'透明度'，表示实时小地图的透明度"✓）
        #   —— 它压在底图上：太实看不清底图、太淡看不清面板，所以给一根**跟手**的拖动条 ✓。
        #   ⚠ 单位（%）写在**框外**（UI规范 ✓ 不许 `setSuffix`）、滚轮不改参数
        #     （`NoWheelSlider` ✓）。
        row_live.addWidget(QLabel("透明度"))
        self.sld_live_alpha = NoWheelSlider(Qt.Horizontal)
        self.sld_live_alpha.setRange(0, 100)
        self.sld_live_alpha.setValue(self._live_alpha)
        self.sld_live_alpha.setMinimumWidth(90)
        self.sld_live_alpha.setMaximumWidth(150)
        self.sld_live_alpha.setToolTip(
            "「显示类型」那一层的透明度（%，实时小地图 / 像素差分地图都用它）。\n\n"
            "0% = 完全看不见（等于关掉那一层）；100% = 完全不透明（会盖住底图）。\n"
            "拖着调，立刻生效（不用等下一拍 ✓）。\n"
            "默认 75%：既看得清面板里的地形，也还能看见底图对不对得上 ✓；\n"
            "看差分时想看得更清楚，拉到 100% 即可 ✓。")
        # 「仅地形图」= 没有那一层 ⇒ 这条跟着灰掉（同老勾选框那条口径 ✓；
        # ⚠ 构造期 `setCurrentIndex` 在 `connect` 之前 ⇒ 得**自己**摆一次状态 ✓）
        self.sld_live_alpha.setEnabled(self._live_disp != "terrain")
        self.sld_live_alpha.valueChanged.connect(safe_slot(self._on_live_alpha))
        # 松手才落盘（见 `_on_live_alpha` 的说明 ✓）
        self.sld_live_alpha.sliderReleased.connect(safe_slot(self._save_live_alpha))
        row_live.addWidget(self.sld_live_alpha)
        self.lbl_live_alpha = QLabel("%d%%" % self._live_alpha)
        self.lbl_live_alpha.setMinimumWidth(40)
        self.lbl_live_alpha.setStyleSheet("color: #5f6368;")
        row_live.addWidget(self.lbl_live_alpha)
        self.lbl_live_map = QLabel()
        self.lbl_live_map.setStyleSheet("color: #80868b;")
        self.lbl_live_map.setWordWrap(True)
        row_live.addWidget(self.lbl_live_map, 1)
        root.addLayout(row_live)

        # ---- 选择平台 + 命令前往 ----
        # ⚠ 这一整块**不在本页显示**（2026-09-26 用户要求：搬到「决策参数 → 操控」区 ✓）：
        #   命令一条路线是**决策**动作，该和「开自动 / 停自动」挨着；以前它藏在另一个
        #   页签的图下面，想"看一眼决策参数再点前往"就得来回切页 ✗。
        # ⚠ 但**控件和逻辑都留在这边**（由主窗口 `mount_goto` 搬过去显示 ✓）：
        #   「预览」要把包围盒画在本页那张**地形图**上 ✓，「命令前往」要用本页手里的
        #   地形 / 集合 / 地图 id ✓ —— 搬去那边就得把这些也复制一份（必然分叉 ✗）。
        # 所以这里**不往 root 里加**，装进 `self.goto_box` 等主窗口来取 ✓。
        self.goto_box = QWidget()
        gbox = QVBoxLayout(self.goto_box)
        gbox.setContentsMargins(0, 0, 0, 0)
        gbox.setSpacing(4)
        row_goto = QHBoxLayout()
        row_goto.setSpacing(6)
        row_goto.addWidget(QLabel("选择平台"))
        self.cmb_goto = NoWheelComboBox()
        self.cmb_goto.setMinimumWidth(150)
        self.cmb_goto.setToolTip(
            "要去的平台（下拉里是**地形编辑器注册过的集合**）。\n\n"
            "选中就把它在下面的地形图上框出来 —— 先看清是哪块，再谈路怎么走。\n"
            "空项 = 不预览（图上不叠框）。\n\n"
            "⚠ 还没圈集合时这里是空的：先到「寻路编辑器」里圈一个。")
        self.cmb_goto.currentIndexChanged.connect(self._on_goto_pick)
        row_goto.addWidget(self.cmb_goto)
        self.btn_goto = QPushButton("命令前往")
        self.btn_goto.setToolTip(
            "按边图算一遍「从现在所在平台 → 预览的平台」通不通，\n"
            "把路线（哪一步是走 / 爬绳 / 下跳 / **跳**）写在下面那行里，\n"
            "**并真的命令角色走过去**（2026-09-26 起）：到了自动接下一步。\n"
            "走不到时会说出**边界**（从起点能到哪些集合）—— 那就是缺边的位置。\n\n"
            "⚠ 四种通行方式**都有执行器**（2026-09-27 起「传送门 / 待确认」已不是通行\n"
            "方式 ✓）⇒ 能算出路就一定下得去；算不出来时会说清是缺哪条边。\n"
            "「跳」的初版执行器已接（2026-09-27）：它用的「起跳距离」是**临时参数**，\n"
            "见「路线识别 → 物理参数」那一组 ✓。途中遇到怪先打（攻击优先仲裁），超时报警。")
        self.btn_goto.clicked.connect(self._on_goto)
        row_goto.addWidget(self.btn_goto)
        # 「**添加任务队列**」（用户 2026-09-27 要求：就放在「命令前往」**右边**）：
        # 把下拉里选的平台**排进队列**；当前这条路线整条走完，自动接着下一条 ✓。
        # 队列显示在**小地图下面那几行**（「当前任务」的下方，一行一个 ✓）。
        self.btn_goto_queue = QPushButton("添加任务队列")
        self.btn_goto_queue.setToolTip(
            "把上面选中的平台**排进任务队列**（不马上出发）✓\n\n"
            "队列里的每一条会在**当前这条路线整条走完**之后自动接着跑（一条接一条 ✓）；\n"
            "每一条出发时才按「我现在站哪」重新算路 —— 所以顺序里夹着「现在到不了」的地方\n"
            "一定会在那一步如实报错 ✓。\n\n"
            "队列显示在**小地图下面那几行**：「当前任务」下方的「队列 N：前往 X」，一行一个 ✓。\n\n"
            "⚠ 「结束当前寻路」会**连队列一起清掉**（那是「别走了」的意思 ✓）。\n"
            "⚠ 队列排在**实时里跑着的 agent** 身上 ⇒ 要先在「实时」页开始（和「命令前往」一样 ✓）。")
        self.btn_goto_queue.clicked.connect(self._on_goto_queue)
        row_goto.addWidget(self.btn_goto_queue)
        # 「结束当前寻路」（2026-09-26 用户要求 3）：撤掉挂着的上绳/下跳任务 ——
        # 也就是让画面那行「当前任务」回到"战斗"。挨着「命令前往」放：一开一关一对。
        self.btn_stop_goto = QPushButton("结束当前寻路")
        self.btn_stop_goto.setToolTip(
            "撤掉**当前挂着的寻路任务**（上绳 / 下跳），角色立刻回到全权战斗。\n\n"
            "挂任务的是「命令前往」；这个按钮就是它的对手 —— 命令走错了、或者\n"
            "你不想让它爬了，点这里（不用去关自动）。\n\n"
            "没有任务在跑时会明说，不会静悄悄什么都不做。")
        self.btn_stop_goto.clicked.connect(self._on_stop_goto)
        row_goto.addWidget(self.btn_stop_goto)
        row_goto.addStretch(1)
        gbox.addLayout(row_goto)

        self.lbl_goto = QLabel()
        self.lbl_goto.setStyleSheet("color: #80868b;")
        self.lbl_goto.setWordWrap(True)
        gbox.addWidget(self.lbl_goto)

        # 只读看图画布：和质检台同一套交互（滚轮缩放 / 中键平移 / 双击适应）
        self.canvas = ImageCanvas()
        # 只设**下限**（用户可以靠分隔条把它压到这么小 ✓ —— 用户 2026-09-27 嫌它占地方；
        # 原来 320 太"霸道" ✗，140 够看清缩略图，要看细节滚轮放大就行 ✓）。
        self.canvas.setMinimumHeight(140)
        root.addWidget(self.canvas, 1)

        # ---- 生成：一条命令干完「导出 WZ 地形 → 画叠加图」----
        # 数据缺就导（几十秒，后台跑），已有就只重画（毫秒级）。
        self.task = None
        gen = QHBoxLayout()
        gen.setSpacing(6)
        self.btn_gen = QPushButton("生成地形图")
        self.btn_gen.setToolTip(
            "自动跑完导出地形这件事：\n"
            "  1. datasets/map/<地图id>.json 不存在 → 起 WzProbe.exe dump-terrain\n"
            "     只导这一张（解析 Map.wz 要几十秒，期间界面可以继续用）；\n"
            "  2. 已经有地形数据 → 跳过第 1 步，只重画叠加图（毫秒级）；\n"
            "  3. 画完自动刷新上面的图，不用重启工作台。\n\n"
            "资源/客户端换过之后要**重新导出**地形数据的话，先删掉\n"
            "datasets/map/<地图id>.json 再点这个按钮。")
        self.btn_gen.clicked.connect(self._generate)
        gen.addWidget(self.btn_gen)

        self.lbl_gen = QLabel("")
        self.lbl_gen.setStyleSheet("color: #80868b;")
        self.lbl_gen.setWordWrap(True)
        gen.addWidget(self.lbl_gen, 1)
        root.addLayout(gen)

        self._refresh_mmap()
        self._refresh_map_image()
        self._refresh_goto()

    # ---------------- 选择平台 / 命令前往（路线测试）----------------

    # ---------------- 选择平台 / 命令前往（路线测试）----------------

    def _refresh_goto(self):
        """「选择平台」下拉：按**地形编辑器注册过的集合**填（空项永远在最前）。

        也跟着集合文件走 —— 编辑器保存后（`_on_zones_saved`）与重开面板时都重读，
        所以新圈的集合不用重启就能选。
        """
        mid = self._map_id()
        cur = self._goto_name() or str(getattr(settings, "route_goto_set", "") or "")
        names = []
        if mid:
            try:
                from core import zones as zones_mod
                names = list(zones_mod.load(mid).sets)
            except Exception:                       # noqa: BLE001
                names = []      # 文件坏了不该让面板起不来（下面那句提示会说明）
        self.cmb_goto.blockSignals(True)
        self.cmb_goto.clear()
        self.cmb_goto.addItem("（不预览）", "")
        for n in names:
            self.cmb_goto.addItem(n, n)
        j = self.cmb_goto.findData(cur)
        self.cmb_goto.setCurrentIndex(j if j >= 0 else 0)
        self.cmb_goto.blockSignals(False)
        self.cmb_goto.setEnabled(bool(names))
        self.btn_goto.setEnabled(bool(names))
        if not names:
            self._say_goto("还没有平台集合 —— 点上面的「寻路编辑器」圈一个（路线按集合走）。",
                           "#b06000")

    def _goto_name(self):
        v = self.cmb_goto.currentData()
        return str(v or "")

    def _preview_boxes(self, mid):
        """选择平台在图上那个框 → [(cls, x, y, w, h, manual)]，空 = 没选/算不出来。

        坐标走 `tools.map_terrain_view.image_xy` —— 和**画那张图**用的是同一套换算。
        自己再算一遍迟早会漂（漂了就是"框画在别处"，看着像集合圈错了，最难查）。
        """
        name = self._goto_name()
        if not (name and mid):
            return []
        try:
            from core import zones as zones_mod
            from tools.map_terrain_view import image_xy
            t = mapdata.load(mid, with_canvas=True)
            z = zones_mod.load(mid)
            s = z.sets.get(name)
            if t is None or s is None:
                return []
            sp = zones_mod.set_span(t, s.get("footholds") or [])
            if sp is None:
                return []
            x0, y0 = image_xy(t, sp[0], sp[2])
            x1, y1 = image_xy(t, sp[1], sp[3])
            # **最少 8 像素**：平台是一根横线（包围盒高度可能是 0），按原样框出来就是
            # 一条 1px 的发丝 —— 等于没框（预览的作用就是"看得见是哪块"）。不够就在
            # 中心两侧补到 8px。
            if x1 - x0 < 8:
                cx = (x0 + x1) / 2.0
                x0, x1 = cx - 4, cx + 4
            if y1 - y0 < 8:
                cy = (y0 + y1) / 2.0
                y0, y1 = cy - 4, cy + 4
            return [(1, x0, y0, x1 - x0, y1 - y0, False)]
        except Exception:                           # noqa: BLE001
            return []

    def _on_goto_pick(self):
        """选了平台 → 存进配置（跟着项目走）+ 在图上把它框出来。"""
        name = self._goto_name()
        if str(getattr(settings, "route_goto_set", "") or "") != name:
            settings.route_goto_set = name
            settings.save()     # 没打开项目时不落盘（见 decision/agent 的 set_save_hook）
        self._say_goto("" if name else "（不预览：图上看全集）")
        self._refresh_map_image()

    def _goto_queue_names(self):
        """**任务队列**里还没跑的目的地（拿不到实时 agent ⇒ 空 ✓）。

        给画面那几行用（`_osd_lines`）—— 队列存在 agent 身上（`agent.goto_queue()` ✓），
        界面**不许**去读它的私有字段（同 `current_goto_set` 那条规矩 ✓）。
        """
        ag = self._live_agent()
        if ag is None:
            return []
        try:
            return list(ag.goto_queue() or [])
        except Exception:                       # noqa: BLE001
            return []

    def _on_goto_queue(self):
        """「添加任务队列」：把当前选中的平台**排进队列**（用户 2026-09-27）。

        ⚠ **排进去就自己跑**（用户 2026-09-27："我期望的是：战斗是最低优先级的任务，
        寻路任务队列要依次执行，现在我排队列都没反应"）：手上没任务 ⇒ 下一拍 `tick` 就
        出发第一条（不用再按「命令前往」✓）；有任务在跑 ⇒ 老实排在它后面，收工自动接
        下一棒 ✓。起跑那笔活儿在决策线程里做（`agent._queue_kick` ✓）。

        队列**跟着实时里跑着的 agent** 走 —— 和「命令前往」同一个理由：起点是"我现在站
        哪个集合"，每一刻都可能变 ⇒ 队列里只存**目的地**，真正出发时才解析路线 ✓。
        所以实时没在跑时**如实说**（别静默什么都不做 ✗）。
        """
        name = self._goto_name()
        if not name:
            self._say_goto("先在左边选一个平台，再排进任务队列。", "#b06000")
            return
        ag = self._live_agent()
        if ag is None:
            self._say_goto("　（队列要排给**实时**里跑着的角色 —— 先去「实时」页开始；"
                           "现在没法排）", "#b06000")
            return
        try:
            n = int(ag.queue_goto(name))
        except Exception as ex:                 # noqa: BLE001
            self._say_goto("排进队列时出错：%s" % ex, "#b06000")
            return
        # ⚠ 自动关着时队列**留不住**：`tick` 开头 `if not s.enabled:` 会
        #   `stop_route("关自动")` 连队列一起清掉（"关自动 = 停手"的口径 ✓）⇒ 这里如实
        #   说一句，别让人以为排上了却又"没反应" ✗（用户 2026-09-27 报的就是这种观感）。
        if not bool(getattr(settings, "enabled", False)):
            self._say_goto("已排队列（第 %d 个）：前往「%s」—— ⚠ 自动**没开**，"
                           "关自动会把队列一起清掉：先开自动" % (n, name), "#b06000")
            return
        self._say_goto("已排进任务队列（第 %d 个）：前往「%s」"
                       "（手上没任务就立刻出发，否则跑完这条接它）" % (n, name), "#1e8e3e")

    def _here_set(self, z, fresh_s=3.0):
        """现在所在的集合名（拿不到给空串）。

        `fresh_s`：定位读数的新鲜度上限。**过期的读数不能当起点** —— 人会走，
        拿几分钟前的位置算出来的路是错的，而且看着像"边图坏了"。
        """
        fid, ts = getattr(self, "_fh_seen", ("", 0.0))
        if not fid or time.monotonic() - ts > fresh_s:
            return ""
        names = z.set_of(str(fid))
        return names[0] if names else ""

    def _on_goto(self):
        """算一遍「现在所在平台 → 选择平台」，**并把第一步交给执行器**。

        两件事一起做：
          · 算路：通不通 + 沿途每一步靠什么过去（走/爬绳/跳/传送门）+ 走不到时的**边界**
            （从起点能到哪些集合）—— 那正是路线测试要的信息（缺哪条边）；
          · 下命令：**整条路线**交给执行器（走 / 爬 / 下跳 / 跳四类都下发 ✓；
            2026-09-26 用户定、2026-09-27 接进「跳」，见 `_command_first_step`）。
            以前这里**只算不走**，点了按钮角色一动不动。
        """
        mid = self._map_id()
        dst = self._goto_name()
        if not mid or not dst:
            self._say_goto("先在左边选一个平台。", "#b06000")
            return
        try:
            from core import zones as zones_mod
            z = zones_mod.load(mid)
        except Exception as e:                      # noqa: BLE001
            self._say_goto("集合文件读不出来：%s" % e, "#c5221f")
            return
        src = self._here_set(z)
        if not src:
            self._say_goto("还不知道你在哪个平台：先勾「在实时画面上叠地形图」、"
                           "到「实时」页跑一小会儿（下面那行要显示得出「位于fh：…」）。",
                           "#b06000")
            return
        path, why = zones_mod.find_path(z, src, dst)
        if path is None:
            self._say_goto("%s" % why, "#c5221f")
            return
        cmd, sent, plan = self._command_first_step(z, mid, path)
        steps = " → ".join(path)
        # 每一步的走法：**显示的就是规划真的选中的那条边**（同一对集合有两条绳时，规划按
        # "离我最近的"选了一条 —— 这行要是还写另一条，就成了"说的和做的不一样" ✗）
        chosen = list((plan or {}).get("edges") or [])
        detail = "；".join(
            "%s --%s-->" % (a, zones_mod.edge_text(
                z, a, b, edge=(chosen[i] if i < len(chosen) else None)))
            for i, (a, b) in enumerate(zip(path, path[1:])))
        here = z.set_of(str(getattr(self, "_fh_seen", ("", 0.0))[0]))
        note = ("（你现在同时属于 %d 个集合：%s；按「%s」当起点）"
                % (len(here), "、".join(here), src) if len(here) > 1 else "")
        # 命令没发出去 ⇒ 用**橙色**：那是在说"这一步现在做不到"，不是路线本身不对。
        self._say_goto("能走到（%s）：%s%s%s%s"
                       % (why, steps, ("　｜　" + detail) if detail else "",
                          note, cmd),
                       "#0b8043" if sent else "#b06000")

    def _on_stop_goto(self):
        """「结束当前寻路」：撤掉挂着的任务，让「当前任务」回到"战斗"。

        判定用 `current_goto_set()`（agent 的公开口径），**不去摸 `_climb`** ——
        那是私有的运行时状态。

        ⚠ 停的是 `agent.stop_route()`（**整条路线**），不是 `stop_climb()`（只当前一步）
        —— 2026-09-26 用户现场定的。差别**看得见**：「命令前往」是**多步**的，
        只停当前一步的话，后面没走的那几步会留在 `_route` 里变成**死账**；而
        `_leave_battle_zone_tick` 原来拿 `_route` 当"我已经在回去的路上了"的证据 ⇒
        此后**一条命令都不下**、角色站着不动（那行已改成只看 `_climb` ✓）。
        两处都堵上：这里停干净、那边判据也不再信死账 ✓。
        用例：`tools/selftest_decision.py` 的 `t_end_route_clears_plan`（直接调这个方法）。
        """
        ag = self._live_agent()
        if ag is None:
            self._say_goto("现在没有在跑的实时推理（先去「实时」页开始）—— "
                           "没有寻路任务可结束。", "#b06000")
            return
        if not self._current_goto_set():
            self._say_goto("现在没有寻路任务（当前任务：战斗）。", "#b06000")
            return
        ag.stop_route("手动结束寻路")
        self._say_goto("已结束当前寻路任务（当前任务：战斗）。", "#0b8043")

    def _current_goto_set(self):
        """当前寻路任务的目标集合名（没有任务 / 实时没在跑时给空串）。"""
        ag = self._live_agent()
        if ag is None:
            return ""
        try:
            return str(ag.current_goto_set() or "")
        except Exception:                           # noqa: BLE001
            return ""

    def _timer_lines(self):
        """所有**计时任务**的名称 + 计时信息（一行一个）：休息最优先，其次自定义定时行为。

        数据全是 `decision.agent.settings` 上的**运行时状态**（agent 写、界面只读），
        口径和「决策参数」页那张休息卡片一致（见 `gui/player_panel._rest_text`、
        `_tick_timer_cd`）。⚠ 时间是 `time.monotonic()` 秒 —— 两侧同进程同一时钟源，
        所以这里直接减（别换成 wall clock，那会被系统对时带偏）。
        """
        out = []

        def left(until):
            if not until or until <= 0:
                return ""
            return "　剩余 %s" % _mmss(until - time.monotonic())

        st = getattr(settings, "rest_state", "")
        # 阶段 → 短句**共用 agent 那张表**（`rest_state_text`）：两处各写一份必然漂 ✗
        #（2026-09-26 实锤：定点休息进了休息，这边**一行都不显示**，而玩家面板写着
        #  「未休息」—— 同一次、两个界面、两种错法 ✗）。
        from decision import agent as agent_mod
        _rt = agent_mod.rest_state_text(st)
        if _rt:
            out.append("休息　%s%s" % (_rt, left(settings.rest_until_monotonic)))
        elif getattr(settings, "rest_pending", False):
            # 到点了但攻击范围内还有怪：卡在这步最容易被当成"坏了"，写出来
            out.append("休息　待休息（等清空攻击范围内的怪）")
        elif getattr(settings, "next_afk_monotonic", 0.0) > 0:
            out.append("休息　下次%s" % left(settings.next_afk_monotonic))
        else:
            out.append("休息　未排期（防掉线关着）")
        # ⭐⭐ **血条读空 ⇒ 定时行为整体被暂停**（用户 2026-10-05 ✓ 三轮原话："「血条读空」期间
        #   连所有自定义定时行为也一起显式暂停（并留痕/显示）"）—— 这时**不许再写「剩余 M:SS」** ✗：
        #   那是"马上要跑"的意思，可它其实已经停用 ✓（真实剩余在恢复那一刻会被**重摇** ✓
        #   见 `agent._reseed_overdue_timers` ✓）。⚠ 判据**只问 agent**（`timers_suspended()` ✓
        #   一处口径 ✓）—— 界面层不自己判断 ✗。
        _t_sus = False
        try:
            _ag1 = self._live_agent()
            _t_sus = bool(_ag1.timers_suspended()) if _ag1 is not None else False
        except Exception:                       # noqa: BLE001 —— 老环境 / 替身 ⇒ 当没停 ✓
            _t_sus = False
        for t in (getattr(settings, "custom_timers", None) or []):
            if t.get("paused"):
                # **暂停的不列**（2026-09-26 用户要求）：它现在不会触发，列出来只会
                # 让人以为"还有一项在跑"。暂停状态在「决策参数」页的列表里看得到。
                # ⚠ 那是**用户自己按的**暂停（与"血条读空"这次整体暂停**两回事** ✓）。
                continue
            name = str(t.get("name") or "?")
            nx = float((getattr(settings, "custom_timer_next", None) or {})
                       .get(name, 0.0))
            # 还没排期（刚加/刚编辑过）⇒ 按区间下限占位，别显示 0:00
            tail = ("　已暂停（血条读空）" if _t_sus else
                    (left(nx) if nx > 0 else
                     "　剩余 %s" % _mmss((t.get("interval") or [5, 10])[0] * 60.0)))
            out.append("定时行为「%s」%s" % (name, tail))
        # ⚠ 「自动喂宠」那一行**已移除**（用户 2026-09-26 去掉整个功能：他会用「自定义定时
        #    行为」自己实现 ✓ ⇒ 它会作为一条普通定时行为出现在上面那段循环里 ✓）。
        if int(getattr(settings, "resetall_interval", 0) or 0) > 0:
            # **要倒计时**，不写"每 N s"（2026-09-26 用户要求）：排期在 agent 的 tick
            # 里，所以它没在跑（或刚开自动还没排到）时只有"未排期"——
            # 公开口径见 `CombatAgent.resetall_left`，别在这儿摸 `_next_resetall`。
            ag = self._live_agent()
            try:
                sec = ag.resetall_left() if ag is not None else None
            except Exception:                       # noqa: BLE001
                sec = None
            out.append("定时清键%s" % ("　剩余 %s" % _mmss(sec) if sec is not None
                                     else "　未排期"))
        return out

    def _fight_line(self):
        """「本集合还能打多久」那一行（**没有在计时 ⇒ ""** ✓ 一个字都不画）。

        用户 2026-09-29 原话："将 foothold 集合的**最大战斗时间倒计时**显示在小地图下面的
        信息栏（当前任务及说明下面，自定义定时行为上面）"✓。

        ⚠ **口径一处**：剩余时间只从 `agent.fight_remain()` 拿 ✓（它和 `_fight_beat` /
        编辑战斗区域里那行灰字倒计时**同一份账** ✓）—— 这里**不许**自己拿墙钟去减 ✗：
        时间在寻路打断期间是**暂停累计**的（不清零 ✓），墙钟减会把打断那段也算进去 ⇒
        显示会莫名跳 ✗（2026-09-28 就是这么错过一次 ✓）。
        ⚠ **没有**在计时（不在任何一个 `can_fight` 区域 / 那一项 `fight_max_s = 0` = 不限 /
        自动没在跑）⇒ 返回 `""` ✓ **绝不写 `0:00`** ✗（那会让人以为"马上要换地方了"✓）。
        """
        ag = self._live_agent()
        if ag is None:
            return ""
        try:
            r = ag.fight_remain()
        except Exception:                       # noqa: BLE001 —— 老环境 / 替身 ⇒ 当没有 ✓
            return ""
        if not r:
            return ""
        try:
            name, left, cap = r
        except (TypeError, ValueError):
            return ""
        return "战斗时长　「%s」剩余 %s / %s" % (name, _mmss(left), _mmss(cap))

    def _osd_lines(self, first):
        """画面那几行（小地图框下方，从上到下）：世界坐标 → **当前任务** → （队列）→
        **战斗时长倒计时** → 定时任务。

        2026-09-26 用户要求 1、2：
          · 「当前任务」默认「战斗」（全权战斗 Agent），有寻路任务时「前往：{集合名}」；
          · 再往下**换行**罗列所有计时任务（休息最优先，其次自定义定时行为）；
          · 这几行**不要背景色**，字色取设置里的「定时任务颜色」。
        **合成一个列表**交给 live_panel（而不是分几次推）：它们要连成一片贴在同一处，
        分开推会出现"上一行的位置被下一行占掉"。
        """
        from gui import theme
        _vis = theme.load_vis()
        col = _vis.get("timer_color") or "#ffeb3b"
        # 「定时任务」那一项的**显示开关**（用户 2026-09-27：辅助线与标记组里每项前面
        # 加开关）—— 关掉 ⇒ **这几行一个字都不画** ✓。
        # ⚠ 它只管自己那几行（当前任务 / 定时任务 / 任务说明）：上面那个 `first` 是
        #   **世界坐标等读数**，不归这个开关管（颜色本来也不是 `timer_color` ✓）。
        # ⭐ **决策状态行**（2026-09-29 用户要求：把画面左上角的「决策:attack」
        #   **搬进信息栏**、放在世界坐标**下面**、与世界坐标**统一格式** ——
        #   就是普通 `str` ⇒ 黑底白字、基础字号，同一份样式 ✓；左上角那行 cv2 白字已删 ✓）。
        #   追击带锁定目标时带上怪号 ✓；拿不到 agent / 状态（老环境、替身）⇒
        #   **一个字都不画**（不画「决策：?」这种废行 ✗）。
        #   ⚠ 它和世界坐标一样是**读数** ⇒ 不归「定时任务」开关管 ✓（关掉也常驻 ✓）。
        _dec = ""
        try:
            _ag = self._live_agent()
            _st = str(getattr(_ag, "state", "") or "") if _ag is not None else ""
            if _st:
                _dec = "决策：%s" % _st
                _tid = getattr(_ag, "_target_id", None)
                if _st == "chase" and _tid is not None:
                    _dec += "（怪%d）" % _tid
                # ⭐ **站桩 attack 的轮次提示**（用户 2026-10-02 ✓ 原话："给站桩 attack 的计数也
                #   加上显示，显示在信息栏，例如 决策：attack (2次后补朝向)"）—— 文案由
                #   `agent.station_turn_hint()` **一处**给 ✓（界面层只贴字符串，**别自己算轮次** ✗
                #   —— 那是第二份口径 ✗）；没在站桩 / 判不出 ⇒ 空串 ⇒ 这行**一个字不变** ✓。
                _hint = ""
                try:
                    _hint = str(_ag.station_turn_hint() or "")
                except Exception:                 # noqa: BLE001 —— 少一行提示别把界面带崩 ✗
                    _hint = ""
                if _hint:
                    _dec += " (%s)" % _hint
                # ⭐ 当前区域配了 idle 回归 foothold ⇒「决策：idle → foothold#N」
                #   （用户 2026-09-29 ✓；目标 id 由 `_idle_walk_beat` 每拍写 ✓ 一处口径）
                _ifid = getattr(_ag, "_idle_fid", None)
                if _st == "idle" and _ifid:
                    _dec += " → foothold#%s" % _ifid
        except Exception:                     # noqa: BLE001 —— 老环境/替身 ⇒ 不画 ✓
            _dec = ""
        # ⭐⭐ **角色死了吗**（用户 2026-10-05 ✓ 原话："角色死了之后画面中心会出现这个弹窗
        #   （原地复活），并且 HP 是 0%？…我们能否检查出角色死了？"）——
        #   判据与"停手 + 留痕"全在 `agent._death_beat` ✓（血条连续几拍读出空 ✓）；
        #   这里**只读数、只贴字符串** ✗（界面层别自己判断 ✓ 那是第二份口径 ✗）。
        _dead = False
        try:
            _ag0 = self._live_agent()
            _dead = bool(getattr(_ag0, "_dead", False)) if _ag0 is not None else False
        except Exception:                     # noqa: BLE001 —— 老环境/替身 ⇒ 当没死 ✓
            _dead = False
        # ⚠ 它和世界坐标/决策那两行一样是**读数** ⇒ **不归「定时任务」开关管** ✓
        #   （关掉那几行时它照样要露出来 ✓ —— 人死了却什么都不显示是最糟的 ✓）。
        _death_line = ("⚠ 角色已死亡（等复活 / 回城再继续）·　自动喝药与定时行为已暂停",
                       "#ff5252", False, 11)
        _head = ([_death_line] if _dead else [])
        if not bool(_vis.get("timer_on", True)):
            return [first] + ([_dec] if _dec else []) + _head
        dst = self._current_goto_set()
        # 休息时「当前任务」写「**休息**」（用户 2026-09-26 要求）—— 休息**优先**于寻路：
        # 「定点休息」本来就会挂着一条"走过去"的任务，那行若还写「前往：X」，会让人以为
        # 在执行「命令前往」（目的地确实是休息点 ✓，但主人是休息机器 ✓；目的地由下面
        # 那行「前往休息点…「X」」说清 ✓）。
        from decision import agent as agent_mod
        _rest = agent_mod.rest_state_text(getattr(settings, "rest_state", ""))
        # ⭐⭐ 「当前任务」那一行：**签注 + 生存时间**（用户 2026-09-28 ✓ 原话："下任务的签注，
        #   需要有生存时间，并在信息栏…的当前任务后面标注出来，例如 `前往：底层(追击  剩余 00:15)`"）。
        #   · **签注** = `agent.current_goto_tag()`（就是 `_climb_origin["kind"]` 那个**短词** ✓
        #     不另编词表 ✓）；**剩余** = `agent.goto_time_left()`（与"寻路超时"**同一把钟** ✓
        #     ⇒ 与真到点那一刻**必然一致** ✓）。两者任一拿不到 ⇒ **整段括号都不出现** ✓（不硬塞 ✗）。
        #   · ⚠ **休息时不加**（`_rest` ✓）：休息的剩余**已经**在下面「定时任务」那几行里显示着
        #     （`休息　休息中　剩余 2:05` ✓ 那是 `_timer_lines` 管的 ✓）⇒ 再加一遍是重复 ✗。
        #   · ⚠ 行元组仍是 **4 项**、字号仍 `11` ✓ ⇒ `gui/live_panel._draw_note` **一个字都不用改** ✓。
        #   · ⭐⭐ **所有任务都显示剩余**（2026-09-29 晚用户澄清 ✓："其他所有的任务都走
        #     「寻路超时时间」，都有出口"）—— 判据在 `goto_time_left()` 里（**一处** ✓）：
        #     追击额外吃「追怪寻路.duration」取更早者 ✓；「寻路超时时间」= 0（不限时）才不显示 ✓。
        _tail = ""
        if not _rest:
            try:
                _ag2 = self._live_agent()
                _tag = str(_ag2.current_goto_tag() or "") if _ag2 is not None else ""
                _left = _ag2.goto_time_left() if _ag2 is not None else None
            except Exception:               # noqa: BLE001 —— 老环境 / 替身对象 ⇒ 就当没有 ✓
                _tag, _left = "", None
            if _left is not None:
                _tail = "(%s剩余 %s)" % ((_tag + "  ") if _tag else "", _mmss2(_left))
        # ⭐ 「当前任务」这一行**字大一号**（用户 2026-09-28 ✓ 原话："当前任务以及下一行
        #   缩进的说明字体稍微大点"）—— 行元组第 4 项 = 字号标记（非 0 ⇒ 用大字 ✓
        #   具体字号由 `gui/live_panel._draw_note` 的 `_TASK_PT` 定 ✓ **只此一处**✗别各写各的）。
        # ⚠ 上面那行 `first`（世界坐标等读数）**不标** ⇒ 保持原字号 ✓。
        lines = [first]                       # str ⇒ 保持黑底白字（读数是排查用的）
        if _dec:
            lines.append(_dec)                # 决策状态：与世界坐标同款式 ✓（见上）
        # ⭐⭐ 角色死亡那行（构造在**上面** ✓ `_head`）—— 放「当前任务」**之前**：
        #   那一行这时还写着"战斗"，得先说清"为什么不打了" ✓（红字 + 大字 ✓）。
        lines += _head
        lines.append(("当前任务　%s%s" % ("休息" if _rest else
                                          ("前往：%s" % dst if dst else "战斗"), _tail),
                      col, False, 11))
        # 任务**现在这一步在干什么 / 为什么失败**（2026-09-26 补）：任务里一直写着原因
        #（对齐中差几像素 / 偏离绳 / 爬不动了 / 拿不到世界坐标），以前没人看得到 ⇒
        # 连着两次"角色爬到某个 y 就不动了"都只能靠猜。挂在任务名下面一行，任务结束就消失。
        # 任务结束后还要挂一会儿（`agent.GOTO_NOTE_KEEP_S`）：用户问的是"为什么在 -170
        # 就松开了 ↑"，而任务一结束那行原本会立刻消失 ⇒ 看见时已无从查证 ✗。
        _note = ""
        try:
            ag = self._live_agent()
            # ⚠ 2026-09-27 起这一行**带上执行器名**（`下跳(drop):…` ✓ 用户要求）——
            #   原来只有一句自由文本，卡住时看不出跑的是哪个执行器 ✗。
            #   （`current_goto_text` 没 job 时会退回 `current_goto_note` ⇒ 老行为不变 ✓）
            _note = str(ag.current_goto_text() or "") if ag is not None else ""
        except Exception:                           # noqa: BLE001
            _note = ""
        if _note:
            # ⭐ **就是这句**（用户 2026-09-28 说的"下一行**缩进**的说明"✓）：`　` 是个全角
            #   空格 ⇒ 画面上看出"缩进"= 它是「当前任务」的子行 ✓。字号也**跟着大一号** ✓
            #   （第 4 项非 0 ⇒ `_draw_note` 用 `_TASK_PT` ✓）。
            lines.append(("　%s" % _note, col, False, 11))
        # ⭐⭐ **「最大战斗时长」的倒计时**（用户 2026-09-29 ✓ 原话："将 foothold 集合的
        #   最大战斗时间倒计时显示在小地图下面的信息栏（当前任务及说明下面，自定义定时
        #   行为上面）"✓）—— 就插在「当前任务 + 它那行说明」**下面**、「休息 / 定时行为」
        #   那几行**上面** ✓（口径见 `_fight_line` ✓）。
        _fight = self._fight_line()
        if _fight:
            lines.append((_fight, col, False))
        # 「**任务队列**」：排在「当前任务」下方、**一行一个**（用户 2026-09-27 要求 ✓）。
        # ⚠ 只在真有队列时才多这几行（没排队时画面上不多一行 ✓）；实时没在跑时队列是空的
        #   （队列本来就存在 agent 身上 ✓）。
        for _i, _dst in enumerate(self._goto_queue_names(), 1):
            lines.append(("队列 %d：前往 %s" % (_i, _dst), col, False))
        lines += [(t, col, False) for t in self._timer_lines()]
        return lines

    def _live_agent(self):
        """当前在跑的 `CombatAgent`（没在跑给 None）——「命令前往」用它下发任务。

        它是 `decision/agent.py` 的模块级 `CURRENT`（「实时」线程启动时登记、退出时注销）：
        以前 agent 只是实时线程里的局部变量 ⇒ 面板够不着，只能"算路给你看"。
        """
        from decision import agent as agent_mod
        return getattr(agent_mod, "CURRENT", None)

    def _command_first_step(self, z, mid, path):
        """把**整条路线**交给执行器 ⇒ (一句给人看的话, 是否真的下发了命令)。

        2026-09-26 改：以前只下**第一步**（后面几步不会自动接着走 ✗）。现在整条交给
        `agent.start_route()`：每段用 `route.job_for_edge` 造任务（走 / 爬 / 下跳 / **跳**），
        到了自动接下一段；**任何一段失败就整条停**并说清断在第几步（见 `agent._task_finished`）。

        只接「走」「爬」「下跳」「跳」四种执行器 —— **四种都有** ✓（2026-09-27 起
        「传送门 / 待确认」已经不是通行方式了 ✓，所以"路径里有跑不了的边"这种情形不存在）
        并说清是第几步、靠什么 —— 半途停在一个上不去/下不来的平台上，比不下更糟 ✗
       （"跳"要先做跳跃标定，"传送门"只差接线；两条都在 `docs/开发计划.md`）。
        """
        ag = self._live_agent()
        if ag is None:
            return ("　（这条命令要发给**实时**里跑着的角色 —— 先去「实时」页开始；"
                    "现在只算给你看）"), False, None
        try:
            from core import mapdata
            from decision import route as route_mod
            t = mapdata.load(mid, with_canvas=True)
            if t is None:
                return "　（读不到地形数据，造不出任务）", False, None
            # ⚠ 判据（整条路径逐段查、跳/门整条拒发、坐标口径、tol/hold 怎么传）**只有一处**：
            # `decision.route.plan_jobs` —— 「定点休息」那边（实时线程注入的解析器）走的是
            # 同一个函数。以前这里自己写了一遍，两边分叉的话会出现"面板能下发、休息却走
            # 不过去"这种最难查的怪事 ✗。
            # `at` = 玩家最近一次世界坐标：同一对集合有多条绳时**挑最近的**（见 pick_edge ✓）
            plan = route_mod.plan_jobs(t, z, path[0], path[-1],
                                       tol_px=int(settings.align_tol_px),
                                       hold_ms=int(settings.align_hold_ms),
                                       # 「多久没进展算卡住」= 「移动操作尝试间隔」换算成秒 ✓
                                       # （2026-09-27 起统一；一处实现 `stall_s_from_retry_ms` ✓，
                                       #  与实时线程那条解析器同源 ✓）
                                       stall_s=route_mod.stall_s_from_retry_ms(
                                           getattr(settings, "move_retry_ms", 3000)),
                                       # 「起跳距离(px)」：只有「跳」那条边用得上（临时参数 ✓）
                                       jump_start_px=int(getattr(settings, "jump_start_px", 0) or 0),
                                       at=getattr(self, "_world_at", None))
        except ValueError as ex:            # 绳找不到 / 说不清上下 ⇒ 如实说，不猜
            return "　（没法下这条命令：%s）" % ex, False, None
        except Exception as ex:             # noqa: BLE001
            return "　（下命令时出错：%s）" % ex, False, None
        if plan.get("here"):
            return "　（已经在「%s」上了，不用走）" % path[-1], False, plan
        jobs = plan["jobs"]
        if not jobs:
            return "　（整条都没下发：%s）" % plan["why"], False, plan
        # 整条交给执行器：到了自动接下一段，中途失败整条停（见 agent._task_finished）
        ag.start_route(jobs, why="命令前往：%s" % path[-1])
        return ("　｜　**已命令**（整条 %d 步）：%s"
                % (len(jobs), " → ".join(path))), True, plan

    def _say_goto(self, text, color="#80868b"):
        self.lbl_goto.setText(text)
        self.lbl_goto.setStyleSheet("color: %s;" % color)

    def _map_id(self):
        p = getattr(self, "project", None)
        return (p.get("map_id") or "").strip() if p is not None else ""

    def _map_display_text(self):
        """「当前地图」那一行的文字：`名字_id`（用户 2026-10-02 给的样式 ✓ 例：`森林迷宫III_105040303`）。

        ⚠ **一处口径、两个显示**（本组顶部那行 + 「小地图定位」卡里的「当前地图」✓）——
          同一件事写两种格式，人就会以为它是两个东西 ✗（`_refresh_current_map` 一起刷 ✓）。
        ⚠ 名字来自 `core.wzexport.map_entry`（和"选地图"那份清单**同一份** ✓）；
          清单读不出来就退回**只给 id** ✓（不编 ✗ 本套资源里几百张图没有名字 ✓）。
        ⚠ 「没打开项目」和「项目里还没选地图」是两回事（以前一律写"没打开项目"，
          项目明明开着却看到这句，像项目丢了 ✗）。
        """
        p = getattr(self, "project", None)
        if p is None:
            return "（没打开项目）"
        mid = self._map_id()
        if not mid:
            return "（这个项目还没选地图）"
        try:
            from core import wzexport
            m = wzexport.map_entry(mid)
            return wzexport.short_label(mid, (m or {}).get("name", ""))
        except Exception:                       # noqa: BLE001 —— 清单读不了也别说不出话 ✓
            return mid

    def _refresh_current_map(self):
        """把「当前地图」**两处**只读文字一起刷（口径见 `_map_display_text` ✓）。

        谁调：`_refresh_mmap()`（`bind` / `showEvent` 都会走到 ✓）、构造函数 ✓、
        以及「手动更换」改完之后 ✓ —— 所以它**不需要**自己盯配置变化 ✓。
        """
        t = self._map_display_text()
        for lbl in (getattr(self, "lbl_cur_map", None),
                    getattr(self, "lbl_mmap", None)):
            if lbl is not None:
                lbl.setText(t)

    def _on_manual_map_change(self):
        """「手动更换」：就地改项目的**地图参数**（用户 2026-10-02 ✓）。

        候选与写口都复用现成那份（`gui/map_picker.MapPickDialog` +
        `core.wzexport.apply_map_choice` ✓ 和「模型训练 → 识别目标」那个下拉**同一个写口**）。

        改完照 `bind()` 末尾那张"换图该刷新什么"的清单做一遍 ✓（漏一处就是"改了没生效"，
        而那类最难查 ✗），并**推给实时线程**（`_push_mmap` ✓）—— 不然"实时还在用上一张图"
        要等到重启才发现 ✗。
        ⚠ 别忘 `map_changed` 信号：玩家的「定点休息」集合下拉是按**地图 id** 存的，
          主窗口靠它重新推（同 `_bind_cards` 在切项目时那一步 ✓）。
        """
        from core import wzexport
        from gui.map_picker import MapPickDialog

        p = getattr(self, "project", None)
        if p is None:
            self._set_hint("先打开一个项目，才能换地图", "#d93025")
            return
        before = self._map_id()
        dlg = MapPickDialog(before, parent=self)
        if dlg.exec_() != dlg.Accepted:
            return
        mid = str(dlg.chosen or "").strip()
        if not mid or mid == before:
            return                                  # 没选/选了同一张 ⇒ 什么都不做 ✓
        m = wzexport.apply_map_choice(p, mid)
        if m is None:
            # 只可能是清单变了（少了这张图）—— 说清、**一个字节都没写** ✓
            self._set_hint("地图清单里找不到 %s —— 换个候选再试一次" % mid, "#d93025")
            return
        # ① 本面板：只读那两行 + 小地图定位那一套（来源/标定/框选状态都在里面 ✓）
        self._refresh_mmap()
        # ② 依赖"是哪张图"的那些：地形图、集合相关下拉、战斗区域（照 `bind()` 的清单 ✓）
        self._refresh_map_image()
        self._refresh_goto()
        self._load_battle_zones_for_map(p)
        # ⭐ 同一件事的第二处：**这张图**的「路线识别」配置也在此刻灌进来（用户 2026-10-03 ✓）
        self._apply_route_cfg(mid)
        # ③ 实时线程（在跑就当场生效 ✓ 见 live_panel.set_mmap）
        self._push_mmap()
        # ④ 别的面板（玩家的集合下拉 ✓ 按地图 id 存的那一批）
        self.map_changed.emit(mid)
        self._set_hint("当前地图已换成 %s ✓"
                       % wzexport.short_label(mid, m.get("name", "")), "#0b8043")

    def zone_sets(self):
        """当前地图里**已注册的集合名**（给「定点休息」那两个下拉用）。

        谁来调：主窗口 `_bind_cards`（切项目 / 换地图）时把它推给玩家面板
        （`player_panel.set_zone_sets(...)`）。为什么由这里提供：集合按**地图 id** 存，
        只有本面板知道当前地图（`_map_id()` ✓），玩家面板不持有它 —— 走仓库既有的
        「面板间推送」（同 `live_panel.set_mmap` ✓）。
        读不到（没项目 / 没集合文件）⇒ 给空列表，让那边只显示「（未选）」——
        **不许**编几个名字出来 ✗。
        """
        mid = self._map_id()
        if not mid:
            return []
        try:
            from core import zones as zones_mod
            return sorted(zones_mod.load(mid).sets)
        except Exception:                       # noqa: BLE001
            return []

    def make_foothold_picker(self, set_name, current_fid=""):
        """造一块**只读的集合 foothold 视图**（给「编辑战斗区域」子弹窗用 ✓ 用户 2026-09-28 要求 ✓）。

        用户原话："编辑战斗区域的**子弹窗**需要有『foothold 集合编辑器』的**同款视图（只读）**，
        可以通过**点选**来**查看 foothold 参数**、**配置「idle回归foothold」**"✓。

        谁调：**本面板的「路线规划 → 编辑战斗区域」**（`_on_battle_zone_edit_clicked` ✓
        把它当 `picker_factory` 传给列表弹窗 ✓）。⭐ 2026-10-01 起**不再跨面板推**：原来
        "玩家面板 → 主窗口 `set_foothold_picker_factory` → 本方法"那条路已经删掉（编辑战斗
        区域整块搬到本页 ✓）—— 弹窗不持有地图 id，而**本页有** ⇒ 直接自己调就行 ✓。

        为什么给**工厂**而不是"造好一个视图推过去"：
          · 视图要按**集合名**造（一个区域项一个集合 ✓ 造的那一刻才知道是哪个 ✓）；
          · 弹窗会开很多次（几个集合就要几块 ✓）⇒ 现造现给最省事 ✓。
        ⚠ **只读**：视图只画 + 选，**不改地形** ✓（要编辑仍去「寻路编辑器」✓）。

        ⭐ **2026-09-28 升级成 `FootholdPickerPanel`**（用户要求 ✓ 原话："可视图希望就显示
        **寻路编辑器里配好的结果图，直接照搬**" + "这个 idle 回归 foothold 配置用**下拉**选择吧，
        选中之后可视图**聚焦**到该 foothold 并**呼吸高亮**（跟寻路编辑器里的模式一样）"✓）：
          · 图**照搬结果图**：底图 + **全部** foothold（本集合那几条**绿**）+ 集合名（见 `_rebuild` ✓）；
          · **下拉**选一条 ⇒ 视图**居中放大**到它 + **呼吸高亮** ✓；
          · 图上点选 ⇒ 下拉**同步** ✓（两边永远一致 ✓）。
        ⚠ 接口与原来的 `FootholdPicker` **完全一致**（`picked` / `current` / `info` /
          `set_current` ✓）⇒ 子弹窗那边的代码**一行没改** ✓。

        读不到（没项目 / 没地形 / 没集合文件）⇒ 回 `None` ⇒ 弹窗**退回文本框** ✓（不许硬造 ✗）。
        """
        mid = self._map_id()
        if not mid:
            return None
        try:
            from gui.foothold_picker import FootholdPickerPanel
            t = mapdata.load(mid, with_canvas=True)     # ⚠ 要底图 ⇒ `with_canvas=True` ✓
            if t is None:
                return None
            z = self._load_zones(mid)
            if z is None:
                return None
            return FootholdPickerPanel(t, z, set_name, current=current_fid)
        except Exception:                       # noqa: BLE001 —— 造不出来就让弹窗退回文本框 ✓
            return None

    def make_element_picker(self, parent=None, multi=False, init=None):
        """造「**地区选择**」通用弹窗（用户 2026-10-05 ✓）。

        谁调：**玩家面板**的「站桩地点」（单选 ✓）与「拾取掉落地区」（多选 ✓）——
        它俩走 `player_panel.set_element_picker_factory`（主窗口把这一个方法推过去 ✓）。
        为什么由**本页**提供：弹窗要**地图与集合**（`terrain` / `zones`），只有本页知道
        当前地图 id（`_map_id()` ✓）⇒ 同 `zone_sets` 那条"面板间推送"（✓）。

        回值：选中的**地点列表**（空列表 = 用户清空了 ✓）/ `None` = **取消**
        （调用方据此**一个字都不改** ✓）。
        ⚠ 没打开项目 / 读不到地形 ⇒ 弹窗里**说清怎么办**并返回 `None`（**别开一个空的** ✗
          —— 同 `_on_battle_zone_edit_clicked` 那条"不能编辑战斗区域"的教训 ✓）。
        """
        mid = self._map_id()
        if not mid:
            QMessageBox.information(
                parent, "还没有地图",
                "选地点要用**地图里的元素**，而现在没打开项目。\n\n"
                "先在「路线识别」页**打开一个项目** ✓")
            return None
        try:
            from gui.element_picker import ElementPickerDialog
            t = mapdata.load(mid, with_canvas=True)     # ⚠ 要底图 ⇒ `with_canvas=True` ✓
            if t is None:
                QMessageBox.information(parent, "读不到地形",
                                        "这张图（%s）的地形没加载出来，先确认文件在不在 ✓" % mid)
                return None
            z = self._load_zones(mid)
            if z is None:
                QMessageBox.information(parent, "读不到集合",
                                        "这张图（%s）的集合文件没读出来 ✓" % mid)
                return None
            dlg = ElementPickerDialog(parent, t, z, multi=bool(multi), init=init)
            if not dlg.exec_():
                return None                              # 取消 ⇒ 不改设置 ✓
            return dlg.selected_spots()
        except Exception as ex:                          # noqa: BLE001
            QMessageBox.warning(parent, "打不开「地区选择」",
                                "%s: %s" % (type(ex).__name__, ex))
            return None

    # ---------------- 路线规划：编辑战斗区域（2026-10-01 从决策参数页搬来 ✓）----------------

    def _bz_hint(self):
        """「编辑战斗区域」下面那行短说明（2026-09-28 用户重构后重写 ✓；2026-10-01 搬来）。"""
        lbl = QLabel("点上面的「编辑战斗区域」增删改 ✓；每一项**勾了「可以战斗」**才算数 —— "
                     "**没勾的项整个不生效**（它那几项参数也不跑 ✓）。")
        lbl.setStyleSheet("color: #80868b;")
        lbl.setWordWrap(True)
        lbl.setToolTip(
            "**每一项 = 一块集合（平台）的配置**（区域查询CD / idle 回归 foothold /\n"
            "最大战斗时长 / 到点去哪 / **可以战斗** ✓）。\n\n"
            "「可以战斗」= **这一项生效不生效的总开关**（用户 2026-10-04 ✓ 原话：\n"
            "  \"刚才我没有勾选底层，但是他还是在底层打了30s\" —— 那次就是老口径的锅 ✗）：\n"
            "  勾上 ⇒ ① 进「能打名单」：人**不在这块集合上**时这一拍**不打架**、先下\n"
            "          「前往」回去 ✓，而且只打**本集合里**的怪 ✓；\n"
            "        ② 它自己那几项参数**照样跑**（区域查询CD / idle 回归 / 最大战斗时长 ✓）。\n"
            "  不勾 ⇒ **这一项整个不生效** ✓：不进能打名单 ✓，而且**它那几项参数也不跑** ✗\n"
            "          —— 不会在这儿计「最大战斗时长」、也不会被 idle 回归/切换平台管 ✓\n"
            "          （**新加的项默认勾上** ✓）。\n\n"
            "一个都没勾 = **不限制**（任何地方都打 ✓ 老行为）**且所有项的参数都不生效** ✓。\n\n"
            "怎么配：点「编辑战斗区域」⇒ 打开列表 ——\n"
            "  点「添加」选一块集合 ⇒ 立刻弹出它的参数窗 ✓；\n"
            "  **双击一行**同样是改它 ✓；选中后点「删除」移除 ✓。\n\n"
            "⚠ 集合候选 = **当前这张图已注册的集合**（每次点开现读 ✓）：\n"
            "   这张图还没圈集合时，先去「**寻路编辑器**」圈一块再来 ✓。\n\n"
            "⚠ 拿不到定位（不知道自己在哪块平台）时按「不在禁战区里」处理 ⇒ 先回去 ✓：\n"
            "   宁可先归位，也不要在不知道自己在哪的时候开打 ✗。")
        return lbl

    def _bz_items(self):
        """**真源**：`settings.battle_zones` 的每一项 → `[(名字, 项), …]`（顺序 = 界面顺序 ✓）。

        ⚠ **界面一律读它**（`battle_zone_sets` 只是**派生副本** ✗）：用户 2026-09-28 原话
          "按钮和弹窗呢？你做的我没法测" —— 那时界面只读写副本 ⇒ `cd_s` /
          `idle_foothold` / `fight_max_s` / `fight_dst` 四个字段**一个入口都没有** ✗。
        """
        out = []
        for z in (getattr(settings, "battle_zones", None) or []):
            if isinstance(z, dict) and str(z.get("set") or ""):
                out.append((str(z["set"]), z))
        return out

    def _bz_commit(self, zones):
        """写回 `battle_zones` + **同步派生副本** + 存盘 + 重画（唯一出口 ✓）。

        ⭐ 存盘目标 2026-10-01 起是**这张图的 `datasets/map/<id>.battle.json`** ✓（不再是
          project.yaml ✗）—— 战斗区域按地图 id 存（见 `core.battle` ✓）。
        ⚠ `sync_battle_zone_sets()` **必须调**（见 `decision/agent.py` 的说明 ✓）：
          `battle_zone_sets` **没有自动同步**（`__getattr__` 代理 + property 会栈溢出 ✗）
          ⇒ 不调的话"筛怪 / 回区域"还会按**老名单**跑 ✗。
        """
        settings.battle_zones = [dict(z) for z in zones]
        try:
            settings.sync_battle_zone_sets()
        except Exception:                       # noqa: BLE001 —— 老设置对象没这个方法也照存 ✓
            pass
        from core import battle
        mid = self._map_id()
        if mid:
            battle.save(mid, settings.battle_zones)     # 写**清洗后**的列表 ✓
        self._refresh_battle_zones()

    def _load_battle_zones_for_map(self, project):
        """按**地图 id** 把战斗区域读进 `settings.battle_zones`（换图 / 开项目时调 ✓）。

        2026-10-01 起战斗区域改存 `datasets/map/<id>.battle.json`（`core.battle` ✓）——
        所以这里是**新的"读"那一边**（`from_dict` 不再读 project.yaml 里的老键 ✓）。

        ⭐ **老配置迁移（只发生一次，显式）**：这张图**还没有** per-map 文件、而旧 project.yaml
          里还留着 `decision.battle_zones` / `battle_zone_sets`（升级前存的）⇒ 把旧值**迁进
          文件一次**（照 `_load_battle_zones` 的老口径：`no_fight` 照抄 / 老名单 = 能打 / 老
          `goto_retry_s` 当 CD ✓），旧行为一点不变 ✓。⚠ 迁移**只在文件不存在时**发生：
          文件一旦写过（哪怕空列表）就**不再**拿旧 project.yaml 覆盖它 ✓（"清空了"是有意的 ✗）。
        """
        from core import battle
        mid = self._map_id()
        if not mid:
            settings.battle_zones = []
            settings.sync_battle_zone_sets()
            self._refresh_battle_zones()
            return
        raw = battle.load(mid)
        if raw is None:
            legacy = (project or {}).get("decision") or {}
            if (isinstance(legacy, dict)
                    and (legacy.get("battle_zones") or legacy.get("battle_zone_sets"))):
                try:
                    _cd = max(0.5, float(legacy.get("goto_retry_s", ZONE_GOTO_RETRY_S)))
                except (TypeError, ValueError):
                    _cd = ZONE_GOTO_RETRY_S
                raw = settings._load_battle_zones(legacy, _cd)
                battle.save(mid, raw)
            else:
                raw = []
        settings.battle_zones = settings._load_battle_zones({"battle_zones": raw})
        settings.sync_battle_zone_sets()
        self._refresh_battle_zones()

    def _bz_candidate_names(self):
        """候选集合名 = **这张图已注册的集合**（`zone_sets()`，**每次点开现读** ✓）。

        ⭐ 这就是搬到本页的意义（用户 2026-10-01 的根因 ✓）：原来在决策参数页时，候选
          只能由本页在**开项目那一刻推一次**（`main_window._bind_cards` →
          `player_panel.set_zone_sets`）⇒ 地形/集合若是会话中途才导出来的，那份候选
          **永远是空表** ⇒ 「添加」里一个集合都没有 ✗（龙族打猎场就是这样）。
          本页持有地图 id ⇒ 现读一遍即可，而且**读不到就明说**（见
          `_on_battle_zone_edit_clicked` ✓）。
        """
        return [str(n) for n in self.zone_sets() if str(n)]

    def _refresh_battle_zones(self):
        """把 `settings.battle_zones` 灌成**一条一行**：**勾选框（可以战斗）+ 摘要** ✓。

        ⭐ 2026-10-01 用户要求：把「可以战斗」的勾选框**从「编辑战斗区域」弹窗挪到主窗口**、
          放在每一个已配项目的前面 ✓ —— 勾 / 取消**立刻**写回（`_on_bz_can_fight` ✓）。

        ⚠ 这里**只显示 + 勾选**：增删、双击改参数都在「编辑战斗区域」弹窗里 ✓（不重复 ✓）。
        """
        _clear_layout(self._bz_rows)
        items = self._bz_items()
        self._bz_names = [n for n, _z in items]
        if not items:
            empty = QLabel("（还没配 —— 点上面的「编辑战斗区域」添加）")
            empty.setStyleSheet("color: #80868b;")
            self._bz_rows.addWidget(empty)
            return
        for n, z in items:
            bits = []
            bits.append("CD %gs" % float(z.get("cd_s") or ZONE_GOTO_RETRY_S))
            _ids = [str(x) for x in (z.get("idle_footholds") or []) if str(x).strip()]
            if _ids:
                bits.append("idle 回 fh %s（随机）" % "、".join(_ids[:3])
                            + ("…等%d" % len(_ids) if len(_ids) > 3 else ""))
            if float(z.get("fight_min_s") or 0.0) > 0:
                bits.append("至少打 %gs" % z.get("fight_min_s"))
            if float(z.get("fight_max_s") or 0.0) > 0:
                bits.append("最多打 %gs → %s"
                            % (z.get("fight_max_s"), z.get("fight_dst") or "（不前往）"))
            _ids_t = "、".join(_ids) if _ids else "（无）"
            # ⭐ 勾选框**就是**「可以战斗」（文本里不再重复写它 ✓ 免得两处表达同一件事 ✗）。
            #   ⚠ 先 `setChecked` 再 `connect`（否则重画时 `toggled` 会误写一次盘 ✗）。
            # ⭐⭐ 没勾 ⇒ **当场**在行里说一句 + 变色（用户 2026-10-04 ✓ 原话："如果不生效动态
            #   显示当前的配置有没有用并简短例如『该区域未启用，不会生效』"✓）——
            #   措辞与「编辑战斗区域」弹窗里那句**同一套** ✓（改口令只改一处 ✓）。
            _off = "" if z.get("can_fight") else "　⚠ 未启用，不会生效"
            cb = QCheckBox(("%s　·　%s" % (n, "，".join(bits)) if bits else n) + _off)
            cb.setChecked(bool(z.get("can_fight")))
            if not z.get("can_fight"):
                cb.setStyleSheet("color:#b06000;")      # 一眼看出哪几项现在不生效 ✓
            cb.setToolTip("这一项的完整设置（**改参数请点上面的「编辑战斗区域」** ✓）：\n"
                          "  区域查询CD(s) = %s\n"
                          "  idle 回归 foothold（随机池） = %s\n"
                          "  最小战斗时长(s) = %s\n"
                          "  最大战斗时长(s) = %s\n  到点去哪 = %s"
                          % (z.get("cd_s"), _ids_t,
                             z.get("fight_min_s") or 0.0,
                             z.get("fight_max_s"), z.get("fight_dst") or "（不前往）"))
            cb.toggled.connect(lambda on, _n=n: self._on_bz_can_fight(_n, on))
            self._bz_rows.addWidget(cb)

    def _on_bz_can_fight(self, name, on):
        """主窗口列表项前面的**勾选框**变动 ⇒ 写回 `can_fight`（用户 2026-10-01 ✓ 从弹窗挪来）。

        ⚠ **状态没变就直接返回**（重画 / 重复触发不该白写一次盘 ✓）。
        """
        zones = [dict(z) for _n, z in self._bz_items()]
        for z in zones:
            if str(z.get("set") or "") == name:
                if bool(z.get("can_fight")) == bool(on):
                    return
                z["can_fight"] = bool(on)
                break
        else:
            return                        # 那一项已经不在配置里了 ⇒ 不动 ✓
        self._bz_commit(zones)

    def _on_battle_zone_edit_clicked(self):
        """点「**编辑战斗区域**」⇒ 打开列表弹窗（用户 2026-09-28 第 2 条 ✓）。

        弹窗里：**添加 / 删除 / 双击一行改参数** ✓；每次增删改都**立刻**经 `_bz_commit`
        写回（存盘 + 同步派生副本 ✓）⇒ 关掉弹窗不需要额外的"确定"语义 ✓。

        ⭐ **候选集合名本页现读**（见 `_bz_candidate_names` ✓）；读不到（这张图还没圈
        集合 / 没打开项目）⇒ **明说怎么补**，别开一个"什么都选不了"的空弹窗 ✗
        （用户 2026-10-01 报的"不能编辑战斗区域"就是这么来的 ✓）。

        ⚠ 只读 foothold 视图的工厂就是本页的 `make_foothold_picker`（**不再跨面板推** ✓）——
          弹窗不持有地图 id，而本页有 ✓。
        """
        names = self._bz_candidate_names()
        if not names:
            QMessageBox.information(
                self, "这张图还没有集合",
                "战斗区域是按**集合（平台）**配的，而现在这张图（%s）里一条集合都没有。\n\n"
                "先点本页上面的「**寻路编辑器**」圈几块集合（存 "
                "datasets/map/<地图id>.zones.json），\n回来就能在这里添加区域了 ✓"
                % (self._map_id() or "没打开项目 / 没选地图"))
            return
        from gui.player_panel import BattleZoneListDialog   # 懒加载（避免构造面板时的循环 ✓）
        dlg = BattleZoneListDialog([z for _n, z in self._bz_items()],
                                   names=names,
                                   on_save=self._bz_commit, parent=self,
                                   picker_factory=self.make_foothold_picker)
        dlg.exec_()
        self._refresh_battle_zones()

    def _on_edit_zones(self):
        """打开/聚焦 foothold 集合编辑器（按当前项目的地图）。

        **非模态 + 单实例**（2026-09-26 改）：
          · 非模态：圈集合时要照着「实时」页的画面判断哪块是 A 平台、还要反复跑一下
            看世界坐标对不对 —— 模态对话框会把整个工作台锁住，只能不停地关窗开窗；
          · 单实例：编辑器挂在**主窗口**上（`win._zone_editor`），再点一次就把它提到
            前面来，不新开第二个窗口（两个窗口改同一份文件，谁覆盖谁说不清）。
        换地图时会关掉旧窗口重建 —— 否则那扇窗画的还是上一张图，人会以为"集合丢了"。

        **槽里抛异常 = 整个工作台 abort**（见 gui/worker.py 的说明），所以自己兜住：
        地形没导、文件坏了、地图 id 不对，都变成一句能看懂的话，而不是进程消失。
        """
        mid = self._map_id()
        if not mid:
            QMessageBox.information(self, "先选地图",
                                    "先在①「识别目标选项」里选地图 —— 集合是按地图存的。")
            return
        win = self.window()
        try:
            from gui.zone_editor import ZoneEditorDialog
            dlg = getattr(win, "_zone_editor", None)
            if dlg is not None and getattr(dlg, "map_id", "") != mid:
                dlg.close()
                dlg.deleteLater()
                dlg = None
            if dlg is None:
                dlg = ZoneEditorDialog(mid, parent=win)
                setattr(win, "_zone_editor", dlg)
            # 面板可能被重建过（换项目）⇒ 提示要接到**当前**这块面板上。
            # 记一下接到谁，避免同一个面板重复连接、旧的连接自然失效。
            if getattr(dlg, "_hint_to", None) is not self:
                dlg.saved_now.connect(self._on_zones_saved)
                dlg._hint_to = self
            # 窗口已经开着时（`show()` 对可见窗口是空操作、`raise_()` 也不触发
            # showEvent）要手动重读一次设置 —— 否则「设置 → 线条宽度」改完再点这个
            # 按钮，还是按旧的宽度画，人会以为设置没生效。
            if hasattr(dlg, "reload_line_width"):
                dlg.reload_line_width()
            dlg.show()                     # show() 而不是 exec_() —— 非模态
            dlg.raise_()                   # 已经在开着就提到前面来（切换焦点）
            dlg.activateWindow()
        except Exception as e:                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "编辑器打不开", "%s: %s" % (type(e).__name__, e))

    # ---------------- 小地图框选（**按项目存**，2026-09-27 从设置搬回本页）----------------

    def _mmap_crop(self):
        """**当前这张图**的小地图框选区域 `[x, y, w, h]`；取不到给 None。

        ⭐ 2026-10-03 起**先看这张图那份**（`datasets/map/<id>.route.json` ✓ 用户要求
        "路线识别页签所有配置按地图 id 存"✓）⇒ 换图后不会再拿上一张图的框 ✗
        （那正是用户 2026-09-27 在 B 机侧踩过的坑 ✓）。
        没灌过（`_crop_override` 为 None：还没选图 / 老会话）⇒ 走**原来的口径**
        `perception.minimap.crop_of`（**项目优先，回退老的那份全局值** ✓）—— 行为一字不变 ✓。
        """
        if getattr(self, "_crop_override", None):
            return list(self._crop_override)
        return mm.crop_of(getattr(self, "project", None), load_live())

    # ---------------- ⭐ 「路线识别」配置按地图 id 存（用户 2026-10-03 ✓）----------------
    # 口径与键清单见 `core/route_cfg.py` ✓：这批键（攀爬参数 / 移动重试 / 起跳距离 /
    # 小地图框选 / 来源 / 黄点跟踪）从"按项目、全局"改成**按地图 id**；而 `project.yaml`
    # （6 个参数 + `mmap_crop`）与 `config/live.yaml`（`mmap_src` / `mmap_track`）
    # **留着当"新图第一次打开时的播种值"** ✓（用户 2026-10-03 选的 ✓）。
    # ⚠ 落盘时机**不在这里逐个 handler 挂** ✗：`main_window` 注册的保存钩子会在**任何**
    #   `settings.save()` 之后顺手调 `_save_route_cfg()` ✓（一处 ✓ 以后加控件不会漏 ✓）。
    def _route_cfg_values(self):
        """面板上按图存的那 9 个键的当前值 → `(map_id, vals)`。"""
        from core import route_cfg
        vals = {}
        for k in route_cfg.PARAM_KEYS:
            v = getattr(settings, k, None)
            if v is not None:
                vals[k] = v
        crop = self._mmap_crop()
        if crop:
            vals["mmap_crop"] = [int(v) for v in crop]
        vals["mmap_src"] = self._mmap_src()
        vals["mmap_track"] = dict(getattr(self, "_mmap_track", None) or {})
        return self._map_id(), vals

    def _save_route_cfg(self):
        """把当前这 9 个键写回**当前这张图**那份（没选图 ⇒ 一个字都不写 ✓ 别瞎存 ✗）。

        ⭐⭐ **多实例**（用户 2026-10-09："必须支持开 2 个工作台" ✓）：带 `_route_cfg_base`
        （= 本窗口**上次见到的**那份 ✓ 见 `_load_route_cfg` ✓）⇒ `route_cfg.save` 只盖
        "相对它改过的键" ✓ —— 另一个窗口刚改的**留住** ✓（这一路尤其要紧：保存钩子在**任何**
        `settings.save()` 之后都会顺手调本方法 ✓ ⇒ 不放基准就是"动个别处的参数也把这 9 键整份盖掉"✗）。
        两边改了**同一个**键 ⇒ 记进 `_route_cfg_conflicts` ✓，由 `main_window` 写进日志 ✓。
        """
        from core import route_cfg
        mid, vals = self._route_cfg_values()
        if not mid:
            return None
        cf = []
        try:
            p = route_cfg.save(mid, vals,
                               base=getattr(self, "_route_cfg_base", None), conflicts=cf)
        except Exception:                       # noqa: BLE001 —— 存不下不该炸界面 ✗
            return None
        # ⚠ 存完把基准刷成"刚写下去的那份" ✓（不刷的话，下次会把"我刚写的"也当成"我改过的"再盖
        #   一遍 ✓ 幂等，但冲突记账会失真 ✗ —— 同 `Project.save` 那条纪律 ✓）
        self._route_cfg_base = dict(vals)
        self._route_cfg_conflicts = cf
        return p

    def _seed_route_cfg(self):
        """这张图**第一次打开**时的播种值（用户 2026-10-03 选的 ① ✓）—— 来源就是**老家**：
          · 6 个参数 ⇒ `project.yaml` 的 `decision` 段（那份空 ⇒ 用 settings 当前值 = 默认 ✓）；
          · `mmap_crop` ⇒ `project.yaml` 顶层（它的老家 ✓）；
          · `mmap_src` / `mmap_track` ⇒ `config/live.yaml`（它们的另一个老家 ✓）。
        ⚠ **别拿 settings 当前值当那 6 个的来源** ✗：手动换图那一刻，settings 里装的是
          **上一张图**的值 ⇒ 那样就把 A 图的参数种给 B 图 ✗（正是本次要治的"换图串参数"✓）。
        """
        from core import route_cfg
        p = getattr(self, "project", None)
        dec = {}
        if p is not None and isinstance(p.get("decision"), dict):
            dec = p.get("decision")
        out = {}
        for k in route_cfg.PARAM_KEYS:
            if k in dec:
                out[k] = dec[k]
            else:
                v = getattr(settings, k, None)
                if v is not None:
                    out[k] = v
        crop = p.get("mmap_crop") if p is not None else None
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            out["mmap_crop"] = [int(v) for v in crop]
        live = load_live() or {}
        if live.get("mmap_src"):
            out["mmap_src"] = live.get("mmap_src")
        if isinstance(live.get("mmap_track"), dict):
            out["mmap_track"] = dict(live.get("mmap_track"))
        return out

    def _apply_route_cfg(self, mid=None):
        """切图 / 换项目：把**这张图**那份灌进来；文件不存在 ⇒ **先播种再灌** ✓。

        灌两拨：6 个参数交给 `settings.apply_route_cfg`（钳位/容错在**那一处** ✓）；
        `mmap_crop` / `mmap_src` / `mmap_track` 是本页控件的事 ⇒ 在这里灌（**一处** ✓）。
        ⚠ `mmap_src` 走 `setCurrentIndex` —— 会触发它自己的 handler（重连收流客户端、
          刷状态 ✓ 那正是换图需要的 ✓）；`mmap_track` 那排用 **blockSignals** 免得逐格
          触发一串下行（值最后一起 `_push_mmap` 推 ✓）。
        """
        from core import route_cfg
        mid = mid or self._map_id()
        if not mid:
            return 0
        vals = route_cfg.load(mid)
        if vals is None:                        # 这张图还没配过 ⇒ 播种
            # ⚠⚠ **只灌值，绝不落盘** ✗（2026-10-03 现场踩了 ✓）：切图 / 换项目是**只读**
            #   动作 —— 一旦在这里写文件，任何"跑一遍自检 / 探针"都会把**当时的假值**
            #   （用例 mock 的 live.yaml ✓）写成真实配置 ✗✗ ⇒ 之后所有用例都被它污染
            #   （实测：3 个 `<id>.route.json` 被写出来，其中 `climb_retry_delay_s=1.0`
            #   把"换项目刚绑好的 3.0"顶掉 ⇒ `selftest_minimap` 当场红 ✓ 见
            #   "bind 里的 setValue 又写回配置了" 那条断言 ✓）。
            #   为什么不写也**不影响**功能：播种源就是"老家"（`project.yaml` / `live.yaml` ✓），
            #   而它们会被用户的每次改动同步更新 ✓ ⇒ 下次读不到文件时再播一次，结果**一样** ✓；
            #   真正落盘的时刻只有一个 —— 用户**改了**任何一个参数（走 `_save_route_cfg` ✓
            #   经保存钩子 / 三个写口 ✓），那时整份 9 键一起写 ✓。
            vals = self._seed_route_cfg()
        # ⭐⭐ 记**基准**：我这次"从文件里见到的 / 播下去的"那份 ⇒ 落盘时只盖"相对它改过的
        #   键" ✓（多实例 ✓ 用户 2026-10-09："必须支持开 2 个工作台" ✓ 见 `core.route_cfg.save`）。
        self._route_cfg_base = {str(k): v for k, v in (vals or {}).items()}
        n = settings.apply_route_cfg(vals)
        # ⭐ 「**禁用杀怪寻路**」（2026-10-06 起**按地图 id 存** ✓ 用户口径："从此 禁用杀怪寻路
        #   就是按地图id存的数据"✓）⇒ 换图/换项目要**把控件摆成这张图的值** ✓。
        #   ⚠⚠ **必须 `blockSignals`** ✗：它接的是 `_on_no_chase_path`（会 `settings.save()` ✓）
        #     ⇒ 不挡信号就等于"切图时**伪装成用户改了参数** ⇒ 当场写盘" ✗ —— 那正是
        #     `_apply_route_cfg` 上面那段血泪（"切图竟然写盘"✗ / `0xC0000409` 硬崩 ✗）里的
        #     同一类坑 ✓。摆值**不是**用户操作 ✓ 落盘只发生在人真改的时候 ✓。
        _ck = getattr(self, "ck_no_chase_path", None)
        # ⚠ **用 `getattr` 护住** ✗：自检里这个"面板"常常是**替身**（`SimpleNamespace` ✓ 只带
        #   本方法真正要动的那几个控件 ✓）⇒ 硬取会当场 `AttributeError` 把用例带崩 ✗
        #   （同 `live_thread` 那条"两层都要 `getattr`"的教训 ✓）。
        if _ck is not None:
            _ck.blockSignals(True)
            _ck.setChecked(bool(getattr(settings, "disable_chase_pathfinding", False)))
            _ck.blockSignals(False)
        _crop = vals.get("mmap_crop")
        self._crop_override = ([int(v) for v in _crop]
                              if isinstance(_crop, (list, tuple)) and len(_crop) == 4
                              else None)
        # ② 来源：**只摆控件，绝不触发它的 handler** ✗（用户 2026-10-03 这条崩过 ✓ 见下）
        #    ✗ 老写法 `self.cmb_mmap_src.setCurrentIndex(...)`（**没挡信号**）=
        #      在 `bind()` 跑到一半时**伪装成"用户手动换了来源"** ⇒ 钻进
        #      `_on_mmap_src → _refresh_mmap（会 load 标定 / 重建 locator / 推到实时）`
        #      整条链 ⇒ 现场实测**进程硬崩**（`0xC0000409` fail-fast ✗ 连 traceback 都没有，
        #      是靠"逐步打成空操作"二分才定到就是这一句 ✓✗）。
        #    ⇒ 程序性恢复配置**不该走"用户交互"那条路**：这里只摆值 + 记覆盖值 ✓，
        #      副作用（收掉本页收流客户端 ✓ / 刷状态 ✓）**由流程显式做**
        #      （`bind()` 末尾本来就会 `_refresh_mmap()` ✓，而它现在读的是覆盖值 ✓）。
        #    ⚠ 也**不写 `config/live.yaml`** —— 那份是"新图的播种值"（用户选的 ① ✓），
        #      切图不该动它 ✗（只有用户自己动下拉时才更新 ✓ 见 `_on_mmap_src` ✓）。
        src = vals.get("mmap_src")
        self._src_override = src or None
        _cmb = self.cmb_mmap_src
        _want = _cmb.findData(src) if src else -1
        if _want >= 0 and _want != _cmb.currentIndex():
            _cmb.blockSignals(True)
            try:
                _cmb.setCurrentIndex(_want)
            finally:
                _cmb.blockSignals(False)
            # 来源换了 ⇒ 本页那条收流客户端要收掉（形状同 `_on_mmap_src` ✓ 下次按需重连）
            _cli = getattr(self, "_mmap_cli", None)
            if _cli is not None:
                try:
                    _cli.stop()
                except Exception:                   # noqa: BLE001
                    pass
                self._mmap_cli = None
        trk = vals.get("mmap_track")
        if isinstance(trk, dict):
            for k, sp in getattr(self, "_sp_track", {}).items():
                if k not in trk:
                    continue
                sp.blockSignals(True)
                try:
                    sp.setValue(float(trk[k]))
                except (TypeError, ValueError):  # 坏值 ⇒ 保留控件现值（不猜 ✗）
                    pass
                sp.blockSignals(False)
            self._mmap_track = dict(trk)
        self._refresh_crop_status()
        return n

    def _refresh_crop_status(self):
        """按钮右边那行「本项目：…」—— 把三个状态**分开说清楚**（别都说成"没框" ✗）：

        · 没打开项目 / 没地图 → 说清"先打开项目"（框选按项目存，没处可存 ✓）；
          ⚠ 这一条要**先判**：没项目时小地图定位压根不跑（实时线程拿不到地图 id），
            此时说"暂用老的那份"是误导 —— 那份值并没有在任何地方被用上 ✗；
        · 本项目框过     → 值；
        · 本项目没框过、但有老的那份全局值 → 值 + 明说"暂用老的那份"（框一次就归本项目 ✓）。
        """
        p = getattr(self, "project", None)
        own = p.get("mmap_crop") if p is not None else None
        crop = self._mmap_crop()
        # 分支顺序按"能确定的事"排：**没有值就绝不去 unpack**（写反了就是一句崩溃 ✗）
        if p is None:
            self.lbl_mmap_crop.setText("还没框（先打开项目 —— 框选是**按项目存**的）")
        elif crop is None:
            self.lbl_mmap_crop.setText("本项目：还没框")
        elif isinstance(own, (list, tuple)) and len(own) == 4:
            self.lbl_mmap_crop.setText("本项目：x=%d y=%d　%d×%d" % tuple(crop))
        else:
            self.lbl_mmap_crop.setText(
                "本项目还没框过　暂用老的那份（全局）：x=%d y=%d　%d×%d"
                % tuple(crop))

    def _pick_mmap_crop(self):
        """在**实时画面**上框出小地图面板 ⇒ 存进**本项目**（`project.yaml` 的 `mmap_crop`）。

        规范（docs/UI规范.md §8）：框选一律走 `gui.region_selector`（放大镜、Esc、
        **<4px 当误点**），并且**框完当场验证** —— 验证 = "裁出来那块和底图能对上多少分"
        （`perception.minimap.region_match_score`，一处实现 ✓）。

        ⚠ **存哪**：项目，不是 `config/live.yaml` ✗ —— 用户 2026-09-27 现场：不同地图的
          小地图面板尺寸完全不同，全局一份的话换个项目还是上一张图的框 ⇒ 定位全错 ✗。
          没打开项目 / 没选地图 ⇒ **一个字都不写**，明说一句（同 `_on_edit_zones` ✓）。
        ⚠ 只写位置，**不动来源**（`mmap_src`）：面板位置和"画面从哪来"是两回事 ——
          顺手改来源等于替用户改了另一个设置 ✗（老代码踩过，用例钉着这条）。
        ⚠ 槽里抛异常 = 整个工作台 abort（见 gui/worker.py）⇒ `safe_slot` 兜住。
        """
        p = getattr(self, "project", None)
        if p is None or not self._map_id():
            QMessageBox.information(
                self, "先选地图",
                "框选小地图是**按项目（地图）存**的 ——\n"
                "先在①「识别目标选项」里选地图，再回来框（每个项目各框一份）。")
            return
        lp = getattr(self, "live_panel", None)
        frame = None if lp is None else lp.current_frame()
        if frame is None:
            QMessageBox.information(
                self, "先开始预览",
                "框选要在实时画面上做 ——\n"
                "去「实时」页点「开始」，看到画面（含游戏的小地图面板），再回来点。")
            return
        from gui.region_selector import select_region_on_image
        rect = select_region_on_image(frame, self)
        if rect is None:
            return                  # 取消（<4px 的框在框选里就按误点丢掉了）
        p.set("mmap_crop", [int(v) for v in rect], save=True)
        # ⭐ 同时写进**这张图**那份（2026-10-03 ✓）：`project.yaml` 那份从此只是"新图的
        #   播种值"✓ ⇒ 内存里那个覆盖值也要跟着更新（本页 `_mmap_crop()` 读的就是它 ✓）。
        self._crop_override = [int(v) for v in rect]
        self._save_route_cfg()
        self._refresh_crop_status()
        # 立刻生效：实时线程拿新框去裁面板（`_push_mmap`）+ 叠图那一层按新位置重画。
        # 以前这两步靠"通知设置窗"绕一圈（`overlay_changed`），现在按钮就在本页 ⇒ 直接叫 ✓。
        self._push_mmap()
        self._refresh_overlay()
        self._verify_mmap_crop(frame, rect)

    def _verify_mmap_crop(self, frame, rect):
        """框完当场和底图核对一次，结论写在按钮下面那行。

        **为什么必须当场验**：框大了（把血条/聊天栏框进去）、面板被游戏 UI 挡住、
        「显示方式」选错 —— 这三种的现象都是"寻路看起来坏了"，而在这里一眼能看出来。
        """
        x, y, w, h = (int(v) for v in rect)
        head = "已框选 (x=%d, y=%d, %d×%d)" % (x, y, w, h)
        mid = self._map_id()
        t = mapdata.load(mid, with_canvas=True) if mid else None
        if t is None or t.canvas is None:
            self.lbl_crop_note.setText(
                "%s　（这张图还没有底图 —— 先点上面的「生成地形图」，之后能在这儿核对）"
                % head)
            self.lbl_crop_note.setStyleSheet("color: #b06000;")
            return
        # 「显示方式」（fit / crop）取**当前来源**那份标定里的 —— 这里没有那个下拉，而
        # 验证用的方式必须和实际换算用的一致，否则分低是假的 ✗。
        mode = (mapdata.load_calib(mid, self._mmap_src()) or {}).get("mode")
        self.lbl_crop_note.setText(head + "　正在和底图核对…")
        self.lbl_crop_note.setStyleSheet("color: #5f6368;")
        QApplication.processEvents()    # 先把上面那行画出来（核对要几十毫秒）
        r = mm.region_match_score(frame, rect, t, mode=mode)
        if r["score"] is None:
            self.lbl_crop_note.setText("%s　%s" % (head, r["why"]))
            self.lbl_crop_note.setStyleSheet("color: #c5221f;")
            return
        self.lbl_crop_note.setText("%s　匹配分 %.2f%s"
                                   % (head, r["score"],
                                      "" if r["ok"] else "（%s）" % r["why"]))
        self.lbl_crop_note.setStyleSheet(
            "color: %s;" % ("#188038" if r["ok"] else "#b06000"))

    # ---------------- 实测精度（只读核对，2026-09-27 用户要求做成一次点击）----------------

    def _mmap_panel_for_check(self, blocking=True):
        """取一帧「小地图面板」给核对用 → `(panel, 问题)`；没问题时 `问题=""`。

        **按来源走两条路**（和叠图 / 标定弹窗同一套判断 ✓）：
          · **独立推流** ⇒ 问 A 机要一帧（`mm.stream_panel` ✓，**不需要**实时页在跑）；
          · **从实时画面** ⇒ 按本项目的框选区域，从**原生帧**上裁一块
            （`current_frame()` = 没画过检测框/视野虚线的那份 ✓，见 docs/UI规范.md §8）。
        每条早退都带回**一句人话**（调用方直接显示 ✓）—— 不许"点了没反应" ✗。

        ⚠ `blocking=False`（**"叠加实时小地图"那条每 250ms 走一次的路** ✓）：收流那一路
        要用**已经连着的那条客户端**（`_stream_client()`：`latest()` 是非阻塞的 ✓）——
        `mm.stream_panel()` 每次都会**新建客户端 + 最多等 5 秒** ✗，放到定时器里必卡 ✓。
        不阻塞时取不到就返回"还没收到帧"，下一拍再来 ✓。
        """
        if self._mmap_src() == mm.SRC_STREAM:
            if not blocking:
                cli = self._stream_client()
                if cli is None:
                    return None, "小地图推流连不上（link.yaml 里读到 a_host 了吗）"
                f, _t = cli.latest()
                if f is None:
                    return None, "还没收到小地图推流（%s）" % (cli.err or "A 机那一路起了吗？")
                return f, ""
            panel, why = mm.stream_panel()
            return (None, why) if panel is None else (panel, "")
        lp = getattr(self, "live_panel", None)
        frame = None if lp is None else lp.current_frame()
        if frame is None:
            return None, ("来源是「从实时画面框选」，核对要拿实时画面里那一块 ——\n"
                          "去「实时」页点「开始」，看到画面再回来点。")
        rect = self._mmap_crop()
        if not rect or len(rect) != 4:
            return None, ("来源是「从实时画面框选」时需要框选区域 ——\n"
                          "点上面那个「框选小地图」（**按项目存** ✓）框一次再来核对。")
        x, y, w, h = (int(v) for v in rect)
        fh, fw = frame.shape[:2]
        if x < 0 or y < 0 or x + w > fw or y + h > fh:
            return None, ("框选区域 %s 超出当前画面（%d×%d）—— 画面尺寸变过"
                          "（换分辨率 / 改推流参数），重框一次再来核对。"
                          % (rect, fw, fh))
        if w < 8 or h < 8:
            return None, "框选区域太小（%d×%d），核对不了" % (w, h)
        return frame[y:y + h, x:x + w], ""

    def _check_mmap_calib(self):
        """「实测精度」：量一遍当前标定差多少 **世界像素**（**只读**，不改任何东西）。

        用户 2026-09-27 要求把它做成**一次点击**（命令行那份
        `tools/mmap_calib_check.py` 算的是同一件事、走的是**同一份** `mm.check_calib` ✓）。

        ⚠ 读数**一律世界像素**（决策④：容差按实时像素定值、**表述**用世界坐标 ✓）。
        ⚠ **尺子可不可信必须一起报**：匹配分 < `mm.TRUST_SCORE` 时明说"只能当参考" ——
          不报的话，人会把"尺子抖"当成"标定偏"（实测：面板混着游戏 UI 时分数只有
          0.62~0.70 ⇒ 那种数看着像结论，其实什么都不是 ✗）。
        ⚠ 槽里抛异常 = 整个工作台 abort（见 gui/worker.py）⇒ `safe_slot` 兜住 ✓。
        """
        mid = self._map_id()
        if not mid:
            QMessageBox.information(
                self, "先选地图",
                "实测精度要拿**本项目的标定**比 ——\n"
                "先在①「识别目标选项」里选地图，再回来点。")
            return
        t = mapdata.load(mid, with_canvas=True)
        if t is None or t.canvas is None:
            QMessageBox.information(
                self, "还没有底图",
                "这张图还没有底图 —— 先点上面那个「生成地形图」，之后才能核对。")
            return
        src = self._mmap_src()
        cal = mapdata.load_calib(mid, src)
        if not mm.has_geometry(cal):
            QMessageBox.information(
                self, "还没标定",
                "「%s」这条来源还没有标定 —— 先用「标定…」或双点标定量一次，"
                "再回来核对。" % mm.SRC_LABEL.get(src, src))
            return
        self.lbl_check_note.setText("正在取一帧画面核对…")
        self.lbl_check_note.setStyleSheet("color: #5f6368;")
        self.lbl_check_detail.setText("")
        QApplication.processEvents()     # 先把上面那行画出来（取帧 + 匹配要几十~几百毫秒）
        panel, why = self._mmap_panel_for_check()
        if panel is None:
            self.lbl_check_note.setText("核对不了：%s" % why.splitlines()[0])
            self.lbl_check_note.setStyleSheet("color: #b06000;")
            self.lbl_check_detail.setText(why)
            return
        r = mm.check_calib(panel, t, cal)
        if r["err_world"] is None:
            # ⭐⭐ **先把现场存下来再说话**（用户 2026-10-09 ✓ 原话："这些提示是不正确的，
            #   因为我的手动标定对齐地图已经接近完美了"）—— 旧版这里只写面板一行 ✗ ⇒
            #   事后谁也分不清"压糊了 / 框多了 / 底图不对" ✗（三件事看一眼图就分得清 ✓）。
            _dir = mm.dump_diag(panel, getattr(t, "canvas", None), cal, tag="check",
                                extra={"map_id": str(mid), "src": str(src),
                                       "why": r.get("why"),
                                       "raw_scores": r.get("raw_scores")})
            # ⭐⭐ 报的是**重合率**那条结论（模板匹配在本图天生够不着门槛 ✗ 见
            #   `perception.minimap.calib_overlap` ✓）—— 它是回答"我标得准不准"的那条路 ✓，
            #   也是用户 2026-10-09 要的那句（他手工对齐近乎完美 ✓ 工具却说量不出来 ✗）。
            _ov_txt = str(r.get("overlap_text") or "")
            _ok = bool(r.get("overlap")) and float((r["overlap"] or {}).get("iou") or 0.0) >= mm.OVERLAP_OK
            if _ov_txt:
                self.lbl_check_note.setText(
                    "模板匹配不适用（本图天花板低）；按**地形重合**看：%s" % _ov_txt)
                self.lbl_check_note.setStyleSheet(
                    "color: %s;" % ("#188038" if _ok else "#b06000"))
            else:
                self.lbl_check_note.setText("核对不了：%s" % r["why"])
                self.lbl_check_note.setStyleSheet("color: #b06000;")
            self.lbl_check_detail.setText(
                ("%s\n" % (r.get("hint") or ""))
                + ("现场已存：%s（panel.png / canvas.png / meta.json）" % _dir if _dir else ""))
            try:
                from core import behavior as _bh
                _bh.event("mmap_check_fail", map_id=str(mid), src=str(src),
                          why=str(r.get("why") or "")[:180], diag=str(_dir))
            except Exception:                        # noqa: BLE001 —— 记账坏了别影响界面 ✓
                pass
            return
        # `verdict` 里那对 `**` 是给命令行看的（Qlabel 不认 markdown）⇒ 去掉 ✓
        self.lbl_check_note.setText(r["verdict"].replace("**", ""))
        self.lbl_check_note.setStyleSheet(
            "color: %s;" % ("#188038" if r["ok"]
                            else ("#b06000" if r["trust"] else "#c5221f")))
        self.lbl_check_detail.setText(
            "偏差 x/y = %+.0f / %+.0f 世界像素（最坏在面板 (%d, %d)）；"
            "标定 scale=%.4f vs 尺子 %.4f（%+.2f%%）；offset %s vs %s；"
            "1 实时像素 = %.2f 世界像素%s"
            % (r["err_world_x"], r["err_world_y"],
               int(r["worst_panel"][0]), int(r["worst_panel"][1]),
               r["scale_cur"], r["scale_ref"], r["scale_delta_pct"] or 0.0,
               [round(float(v), 2) for v in r["offset_cur"]],
               [round(float(v), 2) for v in r["offset_ref"]],
               r["px_per_world"],
               "" if r["trust"] else "；⚠ 尺子匹配分只有 %.2f（< %.2f）⇒ 只能当参考"
               % (r["score"], mm.TRUST_SCORE))
            # ⭐ 顺带把"地形重合率"也写上（2026-10-09 ✓ 用户要看的就是"我标得准不准" ✓）：
            #   它是**对覆盖层/渲染差异不敏感**的那个判据 ✓（见 `mm.calib_overlap` ✓）。
            + ("\n%s" % r["overlap_text"] if r.get("overlap_text") else ""))

    def _on_zones_saved(self, map_id, n_sets):
        """编辑器保存后更新面板提示 + **立刻重画集合图**。

        为什么在这里重画：编辑器保存后，下面那张「地形图」显示的就是旧结果了
        （少一个集合、颜色也变了），而它正是用来看"圈得对不对"的 ——
        所以保存即刷新，不用人再去点一次「生成地形图」。
        """
        self._set_hint("集合已保存（%s：%d 个集合）" % (map_id, n_sets), "#0b8043")
        self._render_zones_png(map_id)
        self._refresh_map_image()
        self._refresh_goto()        # 新圈/改名的集合要能立刻在下拉里选到

    def _render_zones_png(self, mid):
        """把「地形编辑器的结果」画成 `<id>_zones.png`（同步跑，上百毫秒）。

        为什么不丢后台线程：这是一张 1280 宽的小图（实测上百毫秒），而它要**立刻**
        反映到上面的图里；后台线程是留给"导出 WZ"那种几十秒的活的。
        出错只打印 —— 画不出图不该把人挡在"保存成功"之外。
        """
        try:
            from core import zones as zones_mod
            from tools.map_terrain_view import render
            t = mapdata.load(mid, with_canvas=True)
            if t is None:
                return
            render(t, mapdata.map_dir() / ("%s_zones.png" % mid),
                   zones=zones_mod.load(mid))
        except Exception:                   # noqa: BLE001
            import traceback
            traceback.print_exc()

    def _load_zones(self, mid):
        """读这张图的 foothold 集合（**按 mtime 缓存**）→ `core.zones.Zones`。

        为什么看 mtime：编辑器和这块面板是**两个窗口**，那边一保存，这边 4 次/秒的
        读数要立刻跟着变 —— 但也没必要每次重读文件。
        """
        from core import zones
        p = zones.zones_path(mid)
        try:
            mt = p.stat().st_mtime if p.exists() else 0.0
        except OSError:
            mt = 0.0
        c = getattr(self, "_zones_cache", None)
        if c is not None and c[0] == mid and c[1] == mt:
            return c[2]
        z = zones.load(mid)
        self._zones_cache = (mid, mt, z)
        return z

    def _fh_zone(self, loc):
        """脚下那条 foothold 属于哪些集合 → 名字（空串 = 这条判据用不上）。

        **判据是「脚下的 foothold id ∈ 集合」，不是段号**（见 docs/寻路设计.md §12）：
        自动串段会把"要跳/攀才能互通"的并成同一段（实测 105090600 的第 0 段把一面
        388 像素高的悬崖当成了平台边缘），所以"我在不在 A 平台"只能靠人工圈的集合。
        没圈过集合时明说「未分组」，**不显示成空白** —— 空白会被当成程序坏了。
        """
        fid = loc.get("foothold_id")
        if not fid:
            return ""
        try:
            names = self._load_zones(self._map_id()).set_of(fid)
        except Exception:                   # noqa: BLE001
            return ""
        return "、".join(names) if names else "未分组"

    @staticmethod
    def _mode_label(mode):
        """内部值 → 界面上的名字（fit = 全局 / crop = 局部）。"""
        if mode == mm.MODE_FIT:
            return "全局小地图"
        if mode == mm.MODE_CROP:
            return "局部小地图"
        return str(mode)

    # ---------------- 小地图来源（面板画面从哪来）----------------

    def _mmap_src(self):
        """**当前这张图**的小地图来源（没灌过 ⇒ 回退全局 `config/live.yaml` ✓）。

        ⭐ 2026-10-03 起**按地图 id 存**（用户要求："路线识别页签所有配置按地图 id 存" ✓）：
        以前一律读 live.yaml ✗ ⇒ 换图后来源跟着上一条走的（而**标定是按来源分开存的** ⇒
        来源错 = 读错那份几何、世界坐标整体错 ✗ 见 `core/mapdata.calib_path` ✓）。

        ⚠ 为什么用"覆盖值"而不是**改写** live.yaml ✗：那份是**新图的播种值**
          （用户 2026-10-03 选的 ① ✓）⇒ **切图不该动它** ✓ —— 只有用户自己动那个下拉时
          才更新它（见 `_on_mmap_src` ✓）。
        """
        if getattr(self, "_src_override", None):
            return self._src_override
        # 口径只有一处：`perception.minimap.source_for_map`（**按地图 id**那份优先 ✓，
        # 没灌过才回退全局 `live.yaml` ✓）—— 它和上面 `_src_override` 的来路是同一份数据 ✓，
        # 所以两条路给出同一个答案 ✓。⚠ 2026-10-10：体检那边原来读的是**全局** `live_src()`
        # ✗ ⇒ 与本页分叉（"我标定过了还显示没标定"✓）⇒ 那边已改走这个函数 ✓。
        return mm.source_for_map(self._map_id(), load_live())

    def _track_now(self):
        """**当前这张图**的黄点跟踪参数（没灌过 ⇒ 回退全局 `config/live.yaml` ✓）。

        ⭐ 2026-10-03 起按地图 id 存（见 `core/route_cfg.py`）：以前三处一律
        `mm.track_params(load_live())` ✗ ⇒ 换图后还拿上一张图的跟踪参数（面板尺寸/压缩
        都不一样 ⇒ 认黄点的阈值也该不一样 ✓）。

        ⚠ "缺哪个键补默认"的口径**仍在 `mm.track_params` 一处** ✓：这里只决定**读哪一份**
          + **按它的键清单过滤**（这张图那份多写的键忽略 ✗、少的键用默认 ✓）。
        """
        base = mm.track_params(load_live())
        mine = getattr(self, "_mmap_track", None)
        if not mine:
            return base
        out = dict(base)
        for k, v in mine.items():
            if k in out:
                out[k] = v
        return out

    def _on_world_offset(self, _v=None):
        """改了「坐标系偏移」⇒ 写进**这张图的标定文件**（按地图 id + 来源，2026-09-26 用户定）。

        和 `scale/offset/view/alpha` 同一份：面板与实时线程都从标定里读（`PlayerLocator`
        那边已经按来源取好了）⇒ 天然一致，不会再出现"两边各存一份、看着没变"。
        """
        mid = self._map_id()
        if not mid:
            return
        try:
            src = self._mmap_src()
            cal = mapdata.load_calib(mid, src) or {}
            cal["world_offset"] = [int(sp.value()) for sp in self._sp_off]
            mapdata.save_calib(mid, cal, src)
        except Exception:                           # noqa: BLE001
            pass

    def _stream_client(self):
        """收流那条客户端（懒建、复用）—— 来源=收流时定位就用它。

        为什么必须跟来源走：两条来源的**面板尺寸差好几倍**（A 机推流还带 zoom），
        标定也是**按来源分开存**的（`sources.stream` / `sources.live`）——
        拿 A 那份几何去量 B 那块画面，结果就是"标定过了却没坐标"（2026-09-26 踩过）。
        """
        if getattr(self, "_mmap_cli", None) is None:
            try:
                from core.config import get
                self._mmap_cli = mm.MiniMapClient(
                    get("a_host"), port=get("minimap", "port", 5003)).start()
            except Exception:                       # noqa: BLE001
                self._mmap_cli = None
        return self._mmap_cli

    def _on_mmap_src(self, _i):
        """换来源：立刻存下来，并刷一遍状态。

        「框选小地图」**和来源无关**（它就在本页上面）—— 它记的是面板在画面里的位置，
        两种来源都要用（收流时主画面里也有小地图，只是被压过 ✓）。

        换来源要把收流那条客户端**收掉**：留着它白占 A 机一路连接（现在是广播、
        不会再饿死别人，但没必要），下次切回来按时会重连。
        """
        src = self.cmb_mmap_src.currentData()
        update_live(mmap_src=src)      # 全局那份从此只是"新图的播种值"（2026-10-03 ✓）
        # ⚠ 覆盖值要在**存之前**设（`_route_cfg_values` 读的就是 `_mmap_src()` ✓ 见 `_mmap_src`）
        self._src_override = src
        self._save_route_cfg()         # ⭐ 同时写进**这张图**那份 ✓（用户要求按图存 ✓）
        cli = getattr(self, "_mmap_cli", None)
        if cli is not None:
            try:
                cli.stop()
            except Exception:                       # noqa: BLE001
                pass
            self._mmap_cli = None
        self._refresh_mmap()

    def _set_hint(self, text, color="#80868b"):
        """状态行即时反馈（带颜色：中性灰 / 提示橙 / 对不上红 / 对上了绿）。"""
        self.lbl_mmap_hint.setText(text)
        self.lbl_mmap_hint.setStyleSheet("color: %s;" % color)

    def _mmap_dialog_prereq(self):
        """开标定那两扇窗（「标定…」/「双点标定」）的**共同前置** → `(mid, client)`。

        抽出来一处写的原因：「标定…」和「双点标定」的前置**完全一样**，各写一份迟早
        一边漏一条 —— 最典型的是"来源=从实时画面却忘了要求先框选/先开预览" ⇒ 弹窗开着
        却永远没有画面，人以为是标定坏了 ✗。

        任一条不满足：弹提示 + 返回 `(None, None)`（调用方直接 return）。
        """
        mid = self._map_id()
        if not mid:
            QMessageBox.information(
                self, "先选地图",
                "先在①「识别目标选项」里选地图 —— 标定是按每张图各存一份的。")
            return (None, None)
        t = mapdata.load(mid, with_canvas=True)
        if t is None or t.canvas is None:
            QMessageBox.information(
                self, "这张图没有小地图底图",
                "标定要有底图（datasets/map/%s.png）才能对齐。\n\n"
                "先在下面点「生成地形图」把地形和底图导出来。" % mid)
            return (None, None)
        # 画面来源按上面选的那个：
        #   收流       → client=None，弹窗自己连 A 机那一口（原路不变）
        #   从实时画面 → 塞一个"裁实时帧"的适配器进去（弹窗那边一行都不用改）
        client = None
        if self._mmap_src() == mm.SRC_LIVE:
            region = self._mmap_crop()
            lp = getattr(self, "live_panel", None)
            if not region or len(region) != 4:
                QMessageBox.information(
                    self, "先框选区域",
                    "来源是「从实时画面框选」—— 先在实时画面上把小地图面板框出来：\n"
                    "本页「寻路配置」里那个「框选小地图」（就在「寻路编辑器」下面）✓。")
                return (None, None)
            if lp is None or lp.current_frame() is None:
                QMessageBox.information(
                    self, "先开始预览",
                    "标定要拿实时画面里那一块 —— 先到「实时」页点「开始」，\n"
                    "看到画面再回来。")
                return (None, None)
            from gui.live_panel import LiveFrameRegionClient
            client = LiveFrameRegionClient(lp, region)
        return (mid, client)

    def _open_mmap_calib(self):
        """打开标定弹窗（量「面板 ↔ 底图」的换算）。

        显示方式**按上面选的那个**带进去：方式要进游戏走两步才知道该用哪种，
        弹窗里不替你改（和命令行工具的态度一致，见 perception/minimap.py）。
        """
        mid, client = self._mmap_dialog_prereq()
        if not mid:
            return
        dlg = MinimapCalibDialog(mid, mode=self.cmb_mmap_mode.currentData(),
                                 client=client, parent=self,
                                 src=self._mmap_src())
        dlg.exec_()
        if dlg.saved:
            self._refresh_mmap()      # 状态行立刻变成「已标定 … 匹配分 …」

    def _open_two_point_calib(self):
        """打开「双点标定」（两次采样 → 解出缩放/偏移，见 gui/two_point_calib.py）。

        显示方式同样**按上面选的那个**带进去（它决定这份几何是"整张底图铺在面板上"
        还是"面板是底图的一块"—— 两种写法的 `offset`/`view` 不一样，见
        `perception.minimap.calib_from_two_point`）。

        **非模态 + 单实例**（2026-09-27 用户要求："开双点标定窗口时可以操作数据集工作台
        主窗口"）：量这个的过程本来就要一边动工作台（到「实时」页开预览、看世界读数、
        对着叠图核），模态会把工作台锁住 ✗。做法照 `_on_edit_zones` 那套 ——
        窗口挂在**主窗口**上（`win._two_point_dlg`），再点一次只提到前面，不新开第二个
        （两个窗改同一份标定，谁覆盖谁说不清 ✗）；换地图/换来源就关掉重建。
        ⚠ 非模态就没有 `exec_()` 的返回值了 ⇒ 保存结果靠 `saved_now` 信号回来
          （接在 `_refresh_mmap` 上，状态行与叠图立刻跟着变 ✓）。
        """
        mid, client = self._mmap_dialog_prereq()
        if not mid:
            return
        src = self._mmap_src()
        win = self.window()
        dlg = getattr(win, "_two_point_dlg", None)
        # 地图或来源变了 ⇒ 重建一扇（旧的那扇量的是上一张图/另一条来源的几何）。
        # ⚠ **关掉过就不要再重建**（2026-09-27 用户报："关掉双点标定弹窗后，上次的数据
        #   就丢了"）：关窗只是隐藏（`close()`/Esc 都只是 hide），实例留着 ⇒ 你填过的
        #   那几格数字还在；重新打开时由弹窗自己把收帧那一路接回来（见它的 `showEvent`）。
        if dlg is not None and ((getattr(dlg, "map_id", "") != mid)
                                or (getattr(dlg, "src", src) != src)):
            dlg.close()
            dlg.deleteLater()
            dlg = None
        if dlg is None:
            dlg = TwoPointCalibDialog(mid, src=src, client=client,
                                      mode=self.cmb_mmap_mode.currentData(),
                                      parent=win)
            # 面板可能被重建过（换项目）⇒ 信号要接到**当前**这块面板上；
            # 记一下接到谁，避免同一个面板重复连接。
            dlg.saved_now.connect(self._refresh_mmap)
            dlg._hint_to = self
            setattr(win, "_two_point_dlg", dlg)
        elif getattr(dlg, "_hint_to", None) is not self:
            dlg.saved_now.connect(self._refresh_mmap)
            dlg._hint_to = self
        dlg.show()                     # show() 而不是 exec_() —— 非模态
        dlg.raise_()                   # 已经在开着就提到前面来
        dlg.activateWindow()

    def _refresh_mmap(self):
        mid = self._map_id()
        p = getattr(self, "project", None)
        # 来源要先读：**标定是按来源分开存的**，读错那一份就是错的几何
        # （两条来源的面板尺寸差 5.6 倍，见 core/mapdata.calib_path）。
        src_kind = self._mmap_src()
        cal = mapdata.load_calib(mid, src_kind) if mid else None

        # 状态行颜色归位：叠图那条警告会把它染成橙/红，任何一次刷新都该回到中性灰
        # （原来"框完当场验"那条也会染色，2026-09-27 随按钮搬去设置了 ✓）
        self.lbl_mmap_hint.setStyleSheet("color: #80868b;")
        # 来源跟着配置走（blockSignals：不然 setCurrentIndex 会反过来写回去）
        self.cmb_mmap_src.blockSignals(True)
        j = self.cmb_mmap_src.findData(src_kind)
        self.cmb_mmap_src.setCurrentIndex(j if j >= 0 else 0)
        self.cmb_mmap_src.blockSignals(False)
        # 「框选小地图」按钮就在**这一组**（上面「寻路编辑器」正下方 ✓）—— 它记的
        # `mmap_crop` 是必做的一步（叠图要知道往哪画；「从实时画面框选」那条还要靠它
        # 裁面板 ✓）。**状态行这里也要跟着报**（框没框 / 是"本项目框的"还是"暂用老的" ✓）。
        self._refresh_crop_status()

        # 「当前地图」那两行只读文字（本组顶部那行 + 本卡里那个「当前地图」）——
        # **一处口径一处写**（`_map_display_text` ✓ 2026-10-02 起带地图名，例
        # `森林迷宫III_105040303`）；「没打开项目」和「项目里还没选地图」是两回事
        # （以前一律写"没打开项目"，项目明明开着却看到这句，像项目丢了 ✗）。
        self._refresh_current_map()

        self.cmb_mmap_mode.setEnabled(bool(mid))
        self.btn_mmap_calib.setEnabled(bool(mid))
        self.cmb_mmap_mode.blockSignals(True)      # 不挡信号会把刚读的值又写回去
        i = self.cmb_mmap_mode.findData((cal or {}).get("mode"))
        self.cmb_mmap_mode.setCurrentIndex(i if i >= 0 else 0)
        self.cmb_mmap_mode.blockSignals(False)

        if not mid:
            self.lbl_mmap_hint.setText("先在①选地图" if p is not None
                                       else "打开项目后再设")
            self.lbl_mmap_hint.setToolTip("")
        elif not self._mmap_crop():
            # 框选是**必做的一步**（叠图得知道往哪画），而且流程上先框位置、再量
            # 换算 —— 所以它在「未标定」前面报，免得人来回跑两趟。
            self.lbl_mmap_hint.setText("本项目还没框选小地图在画面里的位置")
            self.lbl_mmap_hint.setToolTip(
                "「在实时画面上叠地形图」要知道小地图面板在画面的哪儿；\n"
                "来源选「从实时画面框选」时，还要靠它把面板裁出来喂给标定弹窗。\n\n"
                "点上面「寻路编辑器」下面那个「框选小地图」，在实时画面上框出游戏的\n"
                "小地图面板（框完当场报匹配分 ✓）。\n"
                "⚠ 它是**按项目（地图）存**的 —— 不同地图的小地图面板尺寸/位置完全不同，\n"
                "   所以每个项目各框一份（本项目没框过时会先用着老的那份全局值）。")
        elif not mm.has_geometry(cal):
            # **判的是「有没有几何」，不是「score 有没有」**：手工对齐没有
            # 匹配分（score=0），但它照样是一份能用的标定 —— 拿 score 判会让人
            # 看到「未标定」，以为自己白标了（见 perception/minimap.has_geometry）。
            #
            # 标定按来源分开存：**另一条来源标过，不代表这条标过** —— 那样算出来的
            # 世界坐标整体错。所以这里要把「另一条标过」直接说出来，人才知道
            # 不是自己白标了、而是切了来源。
            others = [mm.SRC_LABEL.get(s, s) for s, c in
                      mapdata.load_calibs(mid).items()
                      if s and s != src_kind and mm.has_geometry(c)]
            self.lbl_mmap_hint.setText(
                "未标定（当前来源：%s%s）"
                % (mm.SRC_LABEL.get(src_kind, src_kind),
                   "；%s 已标过" % "、".join(others) if others else ""))
            if src_kind == mm.SRC_LIVE:
                # 来源是"从实时画面框选"时，前置条件完全不同 —— 别让人去 A 机
                # 找那口推流（他可能压根没在用）。
                tip = ("这张图还没量过「面板像素 → 底图像素」的换算。\n"
                       "来源是「从实时画面框选」：小地图那块已经从实时画面上框出来了\n"
                       "（就是本页「寻路配置」里那个「框选小地图」框的），\n"
                       "点「标定…」量一次就能对上（看得见重合）。")
            else:
                tip = ("这张图还没量过「面板像素 → 底图像素」的换算。\n"
                       "在 A 机「被控机部署台 → 小地图推流」启动之后，\n"
                       "点上面的「标定…」量一次（弹窗里看得见重合不重合）。\n\n"
                       "等价的命令行做法（排查用）：\n"
                       "    python -m perception.minimap --map %s" % mid)
            self.lbl_mmap_hint.setToolTip(tip)
        else:
            # ⚠ 两轴不同时要**分开写**（`scale_y`，见 perception.minimap 顶部那个形状说明）：
            # 只报一个数的话，人以为这份几何是等比的，而它其实 x/y 不一样 ✗
            _sx, _sy = mm.scales_of(cal)
            _s = ("缩放 %.2f/%.2f(x/y)" % (_sx, _sy) if abs(_sy - _sx) > 1e-9
                  else "缩放 %.2f" % _sx)
            geo = ("%s　偏移 %s" % (_s, cal.get("offset"))
                   if cal.get("mode") == mm.MODE_FIT
                   else "%s　显示区起点 %s" % (_s, cal.get("view"),))
            # 手工对齐的没有匹配分：写「手工对齐」，别摆一个 0.00 让人以为坏了
            how = ("手工对齐" if (cal.get("src") == "manual"
                                  or not cal.get("score"))
                   else "匹配分 %.2f" % cal["score"])
            # 老格式（没记来源）**不能**写成「来源：从实时画面」—— 那是猜的。
            # 这份几何只对当时那条来源成立，说成当前来源就是让人放心用错的数。
            label = ("来源未记·老格式" if cal.get("legacy")
                     else mm.SRC_LABEL.get(src_kind, src_kind))
            self.lbl_mmap_hint.setText(
                "已标定（%s）　%s　%s　%s"
                % (label, self._mode_label(cal["mode"]), geo, how))
            crop = self._mmap_crop()
            legacy_note = (
                "⚠ 这份标定是**老格式**（没有记来源）—— 它只对当时那条来源成立。\n"
                "如果你换过小地图来源，请重量一次并保存（保存后就会归到当前来源）。\n\n"
                if cal.get("legacy") else "")
            self.lbl_mmap_hint.setToolTip(
                legacy_note +
                "量出来的几何参数：缩放 / 偏移 / 显示区起点。\n"
                "改「显示方式」只换方式，这些数不会丢。\n\n"
                "画面来源：%s（**标定按来源分开存**，换来源要重量一次 —— "
                "两条来源的面板尺寸不一样，同一份几何在另一条上算出来的位置是错的）\n"
                "小地图在画面里的位置（**本项目**的 mmap_crop）：%s"
                % (mm.SRC_LABEL.get(src_kind, src_kind),
                   crop or "（还没框）"))

        # 叠图那一层：开关/浓淡现在在**设置 → 界面** ✓ ⇒ 这里只按当前配置重算一次
        # 「画什么、往哪画、没画的话卡在哪一步」。
        self._refresh_overlay()
        # 跟踪参数回填（blockSignals —— 回填别把配置又写一遍）
        # ⚠ 2026-10-03：读**这张图**那份（`_track_now` ✓ 按地图 id 存），不再是全局 live.yaml ✗
        trk = self._track_now()
        for k, sp in self._sp_track.items():
            sp.blockSignals(True)
            sp.setValue(int(round(trk[k])))
            sp.blockSignals(False)
        # 「坐标系偏移」从**这张图的标定**里回填（按地图 id + 来源，别把 setValue 当用户改动）
        #
        # ⚠ **2026-09-27 拆掉了"从 config/live.yaml 的 `mmap_world_offset` 搬过来"那条**
        #   （原来在标定里没有这个键时执行一次、还会顺手写进标定文件）。拆的原因：
        #   那天标定的锚点从"黄点重心"改成了"黄点**下沿（脚底）**"（`perception.minimap.dot_feet`），
        #   而 live.yaml 里那份 `[7, 33]` 是**重心口径**的旧值 ⇒ 再搬进来就是**双重补偿**：
        #   每张还没量过的图的读数凭白偏 33 个世界像素，而且是**静悄悄写进文件**的 ✗
        #   （用户 2026-09-27 就是踩在这条形状上：读数差 17 像素）。
        #   现在：标定里有就用它，没有就按 (0, 0)（脚底口径下这才是对的 ✓），
        #   并且**不再往文件里写** —— 只有用户自己在下面那两格里改才会写（见 `_on_world_offset`）。
        off = (cal or {}).get("world_offset") or []
        if len(off) != 2:
            off = [0, 0]
        else:
            off = [off[0], off[1]]
        if len(off) == 2:
            for sp, v in zip(self._sp_off, off):
                sp.blockSignals(True)
                sp.setValue(int(v))
                sp.blockSignals(False)
        # 世界坐标那行（勾上叠图才显示）：跟着地图一起刷新 —— 换了图，地形和
        # 标定都换了，读数必须重算，否则显示的是上一张图的段号。
        self._locator.load(mid or None)
        # ⚠ **顺手把标定缓存忘掉**：这条刷新既在换图时走，也在"弹窗保存完"时走
        #   （`saved_now` → 这里），而 `calib_for` 带 1 秒缓存 ⇒ 不忘的话刚保存的那份
        #   最多 1 秒后才生效，人看着就像"保存没生效/读数不变"（用户 2026-09-27 报的
        #   现象）。`load()` 只在**换图**时清缓存，同图刷新是直接 return 的 ✗，所以这句
        #   不能省。
        self._locator.forget_calib()
        self._locator.use_track_config(trk)   # load() 会重建 tracker，参数要再喂一次
        self._refresh_world()
        self._push_mmap()

    # ---------------- 把叠图画到实时画面上（显示层）----------------

    def _mmap_draw_on(self):
        """要不要把叠图画到实时画面上（config/live.yaml 的 `mmap_draw`，默认关）。"""
        return bool(load_live().get("mmap_draw", False))

    def apply_overlay_settings(self, *_a):
        """设置窗口改了「叠地形图 / 浓淡」之后叫一声 ⇒ 这一层立刻按新配置重画。

        谁调：主窗口（`SettingsDialog.overlay_changed` ✓）。为什么不在本面板留一套开关：
        那属于**外观偏好**，归设置窗口 ✓（用户 2026-09-26 要求搬走）。参数收 `*_a`：
        它接的是 Qt 信号，多给几个参数也不该炸 ✓。
        """
        self._refresh_overlay()
        self._refresh_world()       # 世界坐标那行跟着这个开关显隐

    # ---------------- 玩家世界坐标（读数）----------------

    def _mmap_track_values(self):
        """那排输入框现在的值 → {配置键名: 数值}。"""
        return {k: sp.value() for k, sp in self._sp_track.items()}

    def _on_mmap_track(self, _v=None):
        """改黄点容差：**立刻存配置 + 立刻生效**（不用重开预览）。

        面板自己那份 locator 当场改；实时线程那份靠 `_push_mmap` 推过去 ——
        两边各持一份（见 _tick_world 的说明），参数必须一起同步。
        """
        vals = self._mmap_track_values()
        update_live(**vals)            # 全局那份从此只是"新图的播种值"（2026-10-03 ✓）
        self._mmap_track = dict(vals)  # 本页那份（`_route_cfg_values` 读它 ✓）
        self._save_route_cfg()         # ⭐ 同时写进**这张图**那份 ✓
        self._locator.use_track_config(vals)
        self._push_mmap()

    def _push_mmap(self):
        """把「小地图定位」要的三样推给实时线程（切项目 / 换来源 / 重框都要叫）。

        实时线程那边靠它才知道：用哪张图的地形与标定、面板从哪来、裁哪一块。
        没跑就只存在实时面板上，等下次 start() 由参数带进去。
        """
        lp = getattr(self, "live_panel", None)
        if lp is None or not hasattr(lp, "set_mmap"):
            return
        # ⭐ 2026-10-03：三样**全按地图 id**取（用户要求 ✓ 见 `core/route_cfg.py`）——
        #   框选区域 `_mmap_crop()` ✓、来源 `_mmap_src()` ✓、跟踪参数 `_track_now()` ✓；
        #   每一样都是"这张图那份 → 没灌过才回退全局（`project.yaml` / `live.yaml` ✓）"。
        #   （不再需要 `load_live()` 了 ✗ —— 别把全局那份又读回来拼进来 ✓。）
        lp.set_mmap(map_id=self._map_id() or "",
                    src=self._mmap_src(),
                    crop=self._mmap_crop(),
                    track=self._track_now())

    def _refresh_world(self):
        """世界坐标那行的显隐与节拍：只在勾上「叠地形图」时才跑。"""
        # 换图 / 重标定 ⇒ 几何换了 ⇒ 这一层那份显示区跟踪**要重新开始**（别带着上一张图的
        # 状态/上一份几何的先验 ✗）；种子给"现在最可信的那个"（画面那层跟出来的 → 标定里那份）
        try:
            _cal0 = (mapdata.load_calib(self._map_id(), self._mmap_src()) or {})
        except Exception:                       # noqa: BLE001 —— 刷新而已，不许崩 ✗
            _cal0 = {}
        if self._ov_view:
            _seed0 = self._ov_view
        else:
            _v0 = _cal0.get("view") or (0.0, 0.0)
            _seed0 = (float(_v0[0]), float(_v0[1]))
        self._live_view_track.reset(_seed0)
        on = self._mmap_draw_on()
        self.lbl_mmap_world.setVisible(on)
        if on:
            if not self._world_timer.isActive():
                self._world_timer.start()
            self._tick_world()              # 立刻出一次数，别让人等 250ms
        else:
            if self._world_timer.isActive():
                self._world_timer.stop()
            self._world_note = ""

    def _refresh_mmap_rate(self, src):
        """⭐ 「**小地图链路快不快**」那一行（用户 2026-10-03 ✓）—— 口径见 `lbl_mmap_rate` ✓。

        为什么要有它：本轮把"小地图高帧率更新"改完之后（定位挪去高频回路 ✓ 推流默认
        60fps ✓ 见 SKILL 170），验收数只在 `perf.log` 里 ✗ —— 而**这一页本来就是看链路
        的地方**（它自己就有一条 `MiniMapClient` ✓ `_stream_client` ✓）⇒ 把数摆在这儿 ✓。
        ⚠ 只报**事实**，不猜：没连上就说没连上；来源=live 就说明"没有这一路" ✓。
        """
        if str(src) != mm.SRC_STREAM:
            self.lbl_mmap_rate.setText("链路：来源＝实时画面（没有小地图推流这一路，"
                                       "帧率看「实时」页那行）")
            return
        cli = getattr(self, "_mmap_cli", None)
        if cli is None or not bool(getattr(cli, "connected", False)):
            self.lbl_mmap_rate.setText("链路：小地图推流还没连上（A 机那一路起了吗？）")
            return
        try:
            fps = float(cli.fps)
        except Exception:                       # noqa: BLE001 —— 读数而已，不许崩 ✗
            fps = 0.0
        n_drop = int(getattr(cli, "n_drop", 0) or 0)
        n_recv = int(getattr(cli, "n_recv", 0) or 0)
        _tot = n_recv + n_drop
        pct = (100.0 * n_drop / _tot) if _tot else 0.0
        ms = "—" if self._loc_ms_last is None else "%.2f ms" % self._loc_ms_last
        self.lbl_mmap_rate.setText(
            "链路：收帧 %5.1f fps ・ 丢帧 %d/%d（%.1f%%）・ 定位 %s"
            % (fps, n_drop, _tot, pct, ms))
        self.lbl_mmap_rate.setToolTip(
            "收帧 = A 机**真正推出来**的帧率（`MiniMapClient.fps`，不是配置里那个数）；\n"
            "丢帧 = 我们**没来得及取走**就被下一帧覆盖的次数（占收到总数的比例）——\n"
            "  一直大于百分之几 ⇒ 消费侧比推流慢 ⇒ 该把定位/取用再提快一点；\n"
            "定位 = 这一拍 `PlayerLocator.update` 的耗时（与 `perf.log` 的 `locate_ms` 同源）。\n\n"
            "⚠ 这一行回答「**快不快/新不新**」；上面那行「玩家世界坐标」回答「**对不对**」\n"
            "（几何/标定），地形图里那层叠加也是用来看「对不对」的 ✓。")

    def _tick_world(self):
        """算一次玩家世界坐标并显示（只在这行可见时跑）。

        **面板取的是「实时画面里框出来的那一块」**（`mmap_crop`），所以标定也取
        「从实时画面」那一份 —— 和叠图看的是同一块像素。读数与叠图因此永远对得上；
        换成别条来源的面板尺寸都不一样（实测差 5.6 倍），标定会错位。
        """
        def _say(text, color="#80868b", tip="", osd=None):
            """`text` 给面板那一行，`osd` 给**贴在画面上**的那行（默认同 text）。

            **两行分开**：画面那行是一行不换行的字，长一点就横穿整个画面（实测
            "认不出黄点"那条诊断有 150+ 字，贴上去就是一条黑带盖住半屏、还看不清）。
            所以面板上可以写全，画面上只用一句话版本，详情进 tooltip。
            """
            self._world_note = self._osd_lines(osd if osd is not None else text)
            self.lbl_mmap_world.setText(text)
            self.lbl_mmap_world.setStyleSheet("color: %s;" % color)
            self.lbl_mmap_world.setToolTip(tip)
            # 同一句话贴在画面里那块框的下面（见 live_panel._draw_note）。
            # **只换那行字**：这条路每 250ms 走一次，整算叠图会连地形 JSON 和
            # 底图 PNG 一起重读（几十毫秒 × 4 次/秒，全在 GUI 主线程上）。
            lp2 = getattr(self, "live_panel", None)
            if lp2 is None or not lp2.set_overlay_note(self._world_note):
                self._refresh_overlay()

        if not self._mmap_draw_on():
            return
        lp = getattr(self, "live_panel", None)
        mid = self._map_id()
        if lp is None or not mid:
            return _say("玩家世界坐标：（没打开项目 / 没选地图）")
        # **画面按「小地图来源」走**（2026-09-26 踩过）：这里原来写死"从主画面裁一块 +
        # `src=SRC_LIVE`" ⇒ 来源选**收流**时读的是**另一份标定**（`sources.live`，
        # 多半压根没存过）⇒ 表现就是"标定完了却没坐标"；而且喂的是 H.264 压过的画面，
        # 黄点更糊 —— 跟"选收流"的初衷正好相反。实时线程那份（`live_thread`）一直是对的，
        # 这里照它分一次流。
        src = self._mmap_src()
        # ⭐ 「链路快不快」那一行**先刷**（用户 2026-10-03 ✓）：它用**上一拍**的定位耗时 ✓，
        #   而且下面有好几条早退（连不上 / 没框选 / 认不出黄点 ✓）—— 那些时候这一行**更该**
        #   说清链路的状况 ✓（早退之后再刷就永远是上一句话了 ✗）。
        self._refresh_mmap_rate(src)
        panel = None
        if src == mm.SRC_STREAM:
            cli = self._stream_client()
            if cli is None:
                return _say("玩家世界坐标：小地图推流连不上（link.yaml 里读到 a_host 了吗）",
                            "#b06000")
            panel, _t = cli.latest()
            if panel is None:
                return _say("玩家世界坐标：还没收到小地图推流（%s）"
                            % (cli.err or "A 机那一路起了吗？"), "#b06000")
        else:
            frame = lp.current_frame()
            # ⚠ 框选区域**一处口径**（`_mmap_crop`：本项目优先 ✓，同 `_push_mmap`）——
            #   原来读 `load_live()["mmap_crop"]` ✗ ⇒ 只框了本项目时这里会裁到"老那份"
            #   的位置上，读数就是错的。
            crop = self._mmap_crop() or []
            if frame is None:
                return _say("玩家世界坐标：先到「实时」页点「开始」预览")
            if len(crop) != 4:
                return _say("玩家世界坐标：先框选小地图（读数要从那块画面算）"
                        "　→ 路线识别 → 小地图定位 → 「框选小地图」")
            x, y, w, h = [int(v) for v in crop]
            H, W = frame.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
                return _say("玩家世界坐标：框选的区域超出当前画面 %d×%d —— 重框一次"
                            % (W, H))
            # 复制一份再用：那一帧是实时线程的就地画布（上面有玩家框/平台线），而且
            # 随时会被下一帧覆盖 —— 直接切视图会读到写了一半的像素。
            panel = frame[y:y + h, x:x + w].copy()
        loc = self._locator.load(mid)
        # 和实时线程**同一口径**：脚下 foothold 的 x 容差取设置里的「坐标对齐误差范围」
        # （站平台边上读数会偏出边界几像素 —— 见 `core.mapdata.foothold_below` 的 xtol 说明）
        # ⭐ 这一拍定位花了多久（`_refresh_mmap_rate` 下一拍会把它显示出来 ✓）——
        #   与 `perf.log` 的 `locate_ms` 同一件事，只是摆在界面上 ✓（用户 2026-10-03 ✓）。
        _t_loc0 = time.perf_counter()
        r = loc.update(panel, src=src,
                       fh_xtol=int(getattr(settings, "align_tol_px", 0) or 0))
        self._loc_ms_last = (time.perf_counter() - _t_loc0) * 1000.0
        # ⭐⭐ **「显示区跟住了没有」要先取出来**（用户 2026-09-29："实时小地图的位置没有跟着
        #   我的移动变化"✗ 的直接原因之一）：
        #   它和"认不认得出黄点"是**两件事** —— 面板随人滚动 ⇒ 这一拍显示的底图范围变了没，
        #   光看面板就知道 ✓（`PlayerLocator.update` 里**在黄点之前**就跟了一遍 ✓）。
        #   而下面 `if not r["ok"]` 会**提前 return** ⇒ 认不出黄点时这条信息被丢掉 ✗
        #   ⇒ `_ov_view` 一直是 None ⇒ 画面那层叠图 + 地形图里那块实时小地图都**钉死**
        #   在标定那一刻的位置上 ✗✗。所以这里先处理、再管黄点 ✓。
        _vw_now = r.get("view")
        if r.get("view_ok") and _vw_now:
            _vw_now = (float(_vw_now[0]), float(_vw_now[1]))
            if _vw_now != self._ov_view:
                self._ov_view = _vw_now
                self._refresh_overlay(view=_vw_now)
        # ⭐⭐⭐ **坐标只认「唯一那一份」**（用户 2026-10-06 ✓ 原话："**为什么还是两条链？？**"
        #   ✗ ⇒ "**全做完**"✓）：
        #   `live_thread`（唯一写者 ✓）每拍把 `pos_state` 的广播抄到 `decision.agent.LAST_POS`
        #   ✓ ⇒ **这一行从此读它** ✓，不再用下面那份 `self._locator` 的坐标 ✗。
        #   为什么要收：两份 `PlayerLocator` **各自带跨帧锁定** ⇒ 同一帧"两套历史"⇒ 一个挑对、
        #     一个锁假点 ✗（现场："你看的 1176 是对的、它用的 −1911 是错的"✓ 就是这么来的 ✓）。
        #   ⚠⚠ **那份 locator 不能删** ✗ —— 它还在算"**显示区跟踪**"（`view` ✓ 叠图画在哪儿要用 ✓
        #     那不是坐标 ✓，见上面 `_ov_view` 那一段 ✓）。**删的只是"它算出来的坐标"** ✓。
        #   ⚠ 实时没在跑 ⇒ `LAST_POS is None` ⇒ **照实说"没有位置状态"** ✓（绝不自己再算一份 ✗
        #     —— 那正是要拆掉的东西 ✓）。
        from decision import agent as _agent_mod
        _lp = getattr(_agent_mod, "LAST_POS", None)
        # ⭐⭐ **把三种状态分清楚**（2026-10-10 ✓ 用户报："刚开始推流 + 推理**大概率一直**显示
        #   「世界坐标：实时没在跑」"✗）：以前只有"有坐标"和"没在跑"两种说法 ✗ ⇒
        #   明明在跑、只是**还没定上位**（刚开 / 黄点还没锁住 ✓）也被说成"没在跑" ✓ = 假话 ✗。
        #   ⇒ 现在分：①没在跑（从没收到过）②在跑但这一拍没定上位 ③在跑且有坐标 ✓。
        _age = None
        if _lp:
            try:
                _age = time.monotonic() - float(_lp.get("at") or 0.0)
            except (TypeError, ValueError):
                _age = None
        if _lp and str(_lp.get("map_id") or "") == str(mid):
            r = dict(r,
                     world_x=_lp.get("world_x"), world_y=_lp.get("world_y"),
                     foothold_id=_lp.get("foothold_id"),
                     segment_id=_lp.get("segment_id"),
                     # 唯一那一份的"新鲜度"也说出来（`at` = `time.monotonic()` ✓）
                     note="（唯一那份：实时回路的位置状态 ✓ 人 y 与门 y 差多少按它判 ✓）",
                     short="")
            if r.get("world_x") is None:
                r = dict(r, ok=False,
                         short="实时在跑，但**这一拍还没定上位**（刚开 / 黄点还没锁住）")
            elif _age is not None and _age > 2.0:
                # 有坐标、但**很久没更新** ⇒ 实时多半已经停了（`LAST_POS` 从来不清 ✓）
                r = dict(r, ok=False,
                         short="位置状态已经 %.0f 秒没更新（实时可能停了）" % _age)
        elif _lp:
            # 收到过、但**是另一张图**的（换图之后实时还在按旧图算 ⇒ 别拿它当这一张的坐标 ✗）
            r = dict(r, ok=False, world_x=None, world_y=None,
                     foothold_id=None, segment_id=None,
                     short="实时在算的是另一张图（%s）⇒ 这一行不显示它"
                           % (_lp.get("map_id") or "?"),
                     note="这一行只显示「实时回路的位置状态」（唯一那一份 ✓）——"
                          "它现在给的 map_id 与这张图不一致，按「换图还没跟上」处理 ✓。")
        else:
            r = dict(r, ok=False, world_x=None, world_y=None,
                     foothold_id=None, segment_id=None,
                     short="实时没在跑 ⇒ 没有位置状态（这一行只显示 Agent 用的那份）",
                     note="这一行现在只显示「实时回路的位置状态」（唯一那一份 ✓）——"
                          "不实时跑就没有它。")
        if not r["ok"]:
            # 面板上写一句话，**详情进 tooltip**；画面上那行更短（见 _say 的说明）——
            # 完整诊断有一百多字，贴到画面上就是一条横穿半屏的黑带。
            return _say("玩家世界坐标：%s（鼠标放上去看详情）" % r["short"],
                        "#b06000",
                        "%s\n\n黄点那一层：%s" % (r["note"], r["dot"]),
                        osd="世界坐标：%s" % r["short"])
        # 记下"现在在哪条 foothold + 什么时候读到的"：命令前往要用它当起点，
        # 而**过期的读数不能当起点**（人会走）—— 见 _here_set。
        self._fh_seen = (r.get("foothold_id") or "", time.monotonic())
        # 世界坐标也留一份：择路"挑最近的绳"要用（见 `__init__` 的 `_world_at` ✓）
        self._world_at = (r["world_x"], r["world_y"])
        seg = r["segment_id"]
        head = "玩家世界坐标 (%d, %d)" % (round(r["world_x"]), round(r["world_y"]))
        if seg is None:
            return _say("%s　脚下没有平台" % head, "#b06000", r["note"],
                        osd="世界 (%d, %d)　没站在平台上"
                            % (round(r["world_x"]), round(r["world_y"])))
        color = "#188038" if r["confirmed"] else "#b06000"
        # 「上一帧」= 这一拍没认出黄点、位置沿用上一帧（防抖窗口内）—— 标出来，
        # 别让人以为这是这一拍量到的
        held = "（上一帧）" if r.get("held") else ""
        tail = "" if r["confirmed"] else "（还没连续确认）"
        # 显示的**不再是「第 N 段」**（段号是自动串出来的、语义上不可靠，见 _fh_zone），
        # 而是"脚下的 foothold 属于哪个人工圈的集合" —— 那才是寻路要用的判据。
        zone = self._fh_zone(r)
        where = "　位于fh：%s" % zone if zone else ""
        # ⭐ **「绳梯」「到顶」都读位置状态广播**（2026-09-28 用户核心思路：**坐标 → 状态
        #   的解析全归位置状态机，界面只读广播、不再自己判** ✓）—— 经实时线程 `pos_state()` 拿 ✓。
        # ⚠ **`绳梯` 不再读定位那份 `r["ladder_id"]`** ✗：定位那份是"离哪根绳最近"、**不管按没按
        #   ↑/↓** ⇒ 不按 ↑ 也会显示「绳梯：Lx」（用户 2026-09-28 报：坐标(1193,61) 显示
        #   "一楼 绳梯：L3"是错的，正确是"一楼" ✓）。广播的 `ladder_id` 带"按住 ↑/↓ 才算"的
        #   许可 ✓ ⇒ 拿它才符合用户那条"只有按住 ↑ 时才允许有绳梯判定" ✓。
        #   三态（`at_ladder_top`）：`"L2"` = 已到 L2 上端 ⇒ 显示绳号；`""` = 判过了没到 ⇒
        #   在绳上时写「否」；`None` = 判不了 ⇒ 不写 ✓。
        lad = top = None
        try:
            _th = getattr(getattr(self, "live_panel", None), "thread", None)
            _ps = _th.pos_state() if (_th is not None and hasattr(_th, "pos_state")) else None
            lad = getattr(_ps, "ladder_id", None)
            top = getattr(_ps, "at_ladder_top", None)
        except Exception:                       # noqa: BLE001 —— 显示而已，不许把这一行弄崩 ✗
            lad = top = None
        lad_s = "　绳梯：%s" % lad if lad else ""
        if top:
            top_s = "　到顶：%s" % top
        elif top == "" and lad:
            top_s = "　到顶：否"
        else:
            top_s = ""
        # 贴到画面上的那行**必须短**（它不换行，长了横穿半屏）—— 集合名可以很长
        # （"右下休息平台"就是 6 个字），所以那里截断，面板那一行保留全名
        zone_s = zone if len(zone) <= 8 else zone[:7] + "…"
        # ⭐⭐ **局部小地图（crop）的「显示区」跟踪状态**（用户 2026-09-29 任务 1 ✓）：
        #   crop 的面板**随玩家滚动** ⇒ 标定文件里那个显示区起点只在**标定那一刻**成立 ✗
        #   ⇒ 实时回路每拍都重新跟一次（`perception.minimap.CropViewTracker` ✓）。
        #   这里把"这一拍跟住没有"摆出来 —— 它是"世界坐标整体偏"那类现场的第一现场证据 ✓：
        #     · 跟住了 ⇒ 报**现在**那块底图的起点 + 匹配分（起点随着人走而变，这是对的 ✓）；
        #     · 没跟住 ⇒ **橙色警告**（那一拍用的是标定里那个位置，可能整体偏 ✗）。
        #   ⚠ 不是 crop 时 `view_ok` 是 `None` ⇒ 一个字都不加（fit 没这回事 ✓）。
        crop_seg = ""
        crop_tip = ""
        crop_warn = False
        if r.get("view_ok") is not None:
            _vw = r.get("view") or []
            if r.get("view_ok"):
                crop_seg = "　显示区 %s（跟住 %.2f%s）" % (
                    "(%s)" % ",".join("%.0f" % float(v) for v in _vw) if _vw else "?",
                    float(r.get("view_score") or 0.0),
                    "" if r.get("view_trust") else "·不太稳")
                if not r.get("view_trust"):
                    crop_warn = True
                    crop_tip = (
                        "局部小地图（crop）这一拍**跟住了**，但匹配分没到可信那一档"
                        "（%.2f < %.2f）。\n"
                        "常见原因：面板被游戏 UI 挡掉一块 / 画面糊了。⚠ 被挡掉的那一块"
                        "**是平的**⇒ 匹配分会虚高、位置也可能差几个底图像素\n"
                        "（1 个底图像素 = 好几到十几个世界像素）⇒ 这一拍的读数比干净那拍"
                        "糙一点。" % (float(r.get("view_score") or 0.0), mm.TRUST_SCORE))
            else:
                crop_warn = True
                crop_seg = "　⚠ 显示区没跟住"
                crop_tip = (
                    "局部小地图（crop）的面板**随玩家滚动** ⇒ 程序每拍都要重新问"
                    "「现在显示的是底图哪一块」。\n"
                    "这一拍**没跟上** ⇒ 世界坐标用的是**标定里那个位置**算的，可能整体偏"
                    "（偏的多少 = 滚了多少 × 底图刻度）。\n\n"
                    "跟不住常见原因：面板被游戏 UI 挡住 / 画面糊了 / 「显示方式」选错 /\n"
                    "标定的缩放或显示区起点不对。%s"
                    % ("\n\n具体：%s" % r.get("view_why") if r.get("view_why") else ""))
        # ⚠ 工具提示原来**写死**「来源：从实时画面」（`SRC_LABEL[SRC_LIVE]`）—— 就算
        #   来源选的是「收流」也这么说 ✗。来源说错，人会去错的标定里找问题，而这份几何
        #   只对当时那条来源成立（用户 2026-09-27 查"同一个黄点为什么读数差 9"时就踩在
        #   这句上）。顺带把**黄点的原始像素**摆出来：世界坐标对 1 个面板像素的敏感度 =
        #   `px_per_world / scale`（这张图 8.55 世界单位 ⇒ 差 9 恰好是**一格**）——
        #   有争议时先看这行 x/y 有没有变：变了是"解析/跟踪"，没变就是"标定/来源" ✓。
        return _say("%s%s%s%s%s%s%s" % (head, where, lad_s, top_s, held, tail, crop_seg),
                    "#b06000" if (crop_warn or not r["confirmed"]) else color,
                    ("来源：%s　标定：%s\n"
                     "黄点原始像素（面板）：x=%.2f　y=%.2f（重心，亚像素）"
                     "　脚底锚点 y=%.2f　%s族·%s层%s%s"
                     % (mm.SRC_LABEL.get(src, src), mm.SRC_LABEL.get(src, src),
                        float(r.get("px") or 0.0), float(r.get("py") or 0.0),
                        mm.dot_feet(r)[1],
                        "黄" if r.get("family") == "yellow" else "青",
                        r.get("layer") or "?",
                        "（上一帧沿用）" if r.get("held") else "",
                        ("\n\n" + r["note"]) if r.get("held") else ""))
                    + (("\n\n" + crop_tip) if crop_tip else ""),
                    osd="世界 (%d, %d)%s%s%s"
                        % (round(r["world_x"]), round(r["world_y"]),
                           ("　fh：%s" % zone_s) if zone_s else "",
                           lad_s + top_s, held))

    def _overlay_blocker(self):
        """现在画不了的话卡在哪一步（空串 = 能画）。

        五种卡法都要能说出来 —— 这一块最容易变成"勾了没反应"，而人对着一个
        没反应的勾选框只能猜。

        ⚠ **顺序 = 提示的优先级**：前面几条是"要人去做什么"，最后那条（等收流来帧）
        是"等一会儿它自己就好" —— 所以人该动手的原因排在前面，别让一句"等着"盖住
        "你还没标定"。
        """
        mid = self._map_id()
        if not mid:
            return "还没选地图"
        if not mm.has_geometry(mapdata.load_calib(mid, self._mmap_src()) or {}):
            return ("这张图在当前小地图来源下还没标定（点右边「标定…」量一次并保存）")
        # ⚠ 框选区域**一处口径**（`_mmap_crop` = `mm.crop_of`：本项目优先、没框过才回退
        #   老的那份全局值 ✓）。这里原来读 `load_live()["mmap_crop"]` ✗ —— 框选区域
        #   2026-09-27 起**按项目存**（project.yaml）⇒ 只框了本项目的图上，这一层会一直
        #   说"还没框选小地图"、叠图压根画不出来 ✗（用户 2026-09-29 问"叠图跟不跟"时
        #   顺手查出来的同族问题）。
        if not self._mmap_crop():
            return ("还没框选小地图在画面里的位置（路线识别 → 小地图定位 → "
                    "「框选小地图」，按项目存）")
        if not (mapdata.map_dir() / ("%s_overlay.png" % mid)).exists():
            return "还没有 %s_overlay.png（先点下面的「生成地形图」）" % mid
        t = mapdata.load(mid, with_canvas=True)
        if t is None or t.canvas is None:
            return "这张图还没有小地图底图（先点「生成地形图」）"
        # ⚠ 来源=收流、且**还不知道"标定当时那块面板多大"** ⇒ **先别画**（2026-09-26 修，
        #   用户报："叠图刚打开实时时尺寸不对，点一下「路线识别」页签才正常"）。
        #   根因（已复现，现场数字：底图 134×101、标定 scale=1.874、mmap_crop=[6,72,134,109]）：
        #     · 那个尺寸只能问**当前收流帧**（标定文件里的 `panel` 字段是空话 —— 全项目
        #       没有任何地方写它，见 `_calib_panel_wh`），而收流客户端是**懒建**的
        #       （`_stream_client()`）⇒ 第一次刷新必然拿不到帧 ⇒ 给 None；
        #     · `frame_overlay_rects(..., calib_panel=None)` 就**不折算**（老行为）⇒
        #       直接拿标定的 scale 去画 ⇒ 实测画成 251×189 糊在 134×109 的框上（宽 1.873 倍）✗；
        #     · 更糟的是**画上之后没人再重算**：250ms 那个节拍（`_tick_world`）只在
        #       "叠图还没挂上"时才整算一次（`live_panel.set_overlay_note()` 返回 False 的那条路）
        #       ⇒ 尺寸就**冻结在第一次** ✗ —— 这正好解释了"点一下页签才正常"（`showEvent`
        #       → `_refresh_mmap` → `_refresh_overlay`，那时收流早就来帧了）。
        #   拦在这里之后：这两拍不画（状态行说明白）⇒ `set_minimap_overlay(None)` 让
        #   `set_overlay_note` 返回 False ⇒ 上面那个 250ms 节拍会一直重试 ✓ ⇒ 收流第一帧
        #   一到，250ms 内自动画**对** ✓（不重标定、不碰标定文件、不加参数）。
        #   判据只认"来源=收流 且 折算比例算不出来"：来源=「从实时画面」时那块面板**就是**
        #   框出来的这一块（标定弹窗裁的就是它）⇒ 折算比例本来就是 1，不能拦 ✗
        #   （拿 `_calib_panel_wh()` 单一返回值当判据会连它一起拦住，见用例 ③）。
        if self._mmap_src() == mm.SRC_STREAM and self._calib_panel_wh() is None:
            return ("等 A 机小地图推流来第一帧（还不知道那块面板多大 —— "
                    "现在画会大/小一截，它来了会自动画上）")
        return ""

    def _overlay_pix(self, mid):
        """叠图源图（`<id>_overlay.png`）的 QPixmap，**带缓存**。

        为什么要缓存：世界坐标那行每 250ms 重贴一次，而这条路会重画叠图 ——
        每次都从磁盘读一张 200KB 级的 PNG（4 次/秒）纯属浪费，而且卡主线程。
        """
        p = str(mapdata.map_dir() / ("%s_overlay.png" % mid))
        if self._ov_pix is None or self._ov_pix[0] != p:
            self._ov_pix = (p, QPixmap(p))
        return self._ov_pix[1]

    def _on_live_view(self, *_a):
        """「显示类型」变了（用户 2026-10-06 ✓）：记住 → 落盘 → 立刻按新类型来一拍 ✓。

        ⚠ 签名吃 `*_a`：`currentIndexChanged` 会给槽带一个 int（不接住就是静默 TypeError ✓）。
        """
        self._live_disp = norm_live_view(self.cmb_live_view.currentData())
        self._save_live_view()
        self._apply_live_view()

    def _apply_live_view(self):
        """按当前「显示类型」开/停那条 250ms 的节拍，并立刻来一拍 ✓（= 老 `_on_live_map_toggle` ✓）。

        「仅地形图」⇒ 那一层**收起来**（= 老版本"去勾" ✓），节拍也停（没人看就别取帧 ✓）。
        """
        on = self._live_disp != "terrain"
        # 透明度那条只对"看得见的那一层"有意义 ⇒ 「仅地形图」时跟着**灰掉**
        # （不藏掉：布局不跳 ✓，和设置窗里「浓淡」对「叠图」的做法一致 ✓）
        self.sld_live_alpha.setEnabled(on)
        if on:
            self.canvas.set_live_patch_alpha(self._live_alpha / 100.0)
            self._live_map_timer.start()
            self._live_map_tick()
        else:
            self._live_map_timer.stop()
            self.canvas.set_live_patch(None)        # 收起来（不是留一张空的 ✗）
            self.lbl_live_map.setText("")

    def _save_live_view(self):
        """把显示类型存进 `config/live.yaml`（本机外观偏好，和 `live_map_alpha` 同一处 ✓）。

        ⚠ 与透明度那条不同：这个是**下拉**（一次点击一个事件），所以当场写 ✓；
        （透明度是拖动条，拖一次几十个事件 ⇒ 那才要攒到松手 ✗。）
        """
        try:
            update_live(live_map_view=str(self._live_disp))
        except Exception:                       # noqa: BLE001 —— 存不下也不该把界面弄崩 ✗
            pass

    def _on_live_alpha(self, v):
        """拖动透明度：**立刻**生效（跟手 ✓）。

        ⚠ **不在这里写盘**：拖一次会来几十个事件 ⇒ 写几十次 YAML（每次重写整个文件 ✗）；
        落盘在**松手**（`sliderReleased`）和**离开这一页**（`hideEvent`）各一次 ✓。
        """
        self._live_alpha = int(v)
        self.lbl_live_alpha.setText("%d%%" % self._live_alpha)
        self.canvas.set_live_patch_alpha(self._live_alpha / 100.0)

    def _save_live_alpha(self):
        """把透明度存进 `config/live.yaml`（本机外观偏好，和 `mmap_draw` 同一处 ✓）。"""
        try:
            update_live(live_map_alpha=int(self._live_alpha))
        except Exception:                       # noqa: BLE001 —— 存不下也不该把界面弄崩 ✗
            pass

    def _live_map_rect(self, cal, canvas_wh):
        """面板图摆在**地形图坐标系**里的哪儿 → `(origin_x, origin_y, kx, ky)`。

        地形图 / 叠加图那一层是底图的 `overlay_zoom` 倍（`<id>_zones.png` 同理 ✓），
        所以要把"面板像素 → 底图像素"（`mm.panel_to_canvas`，**唯一口径** ✓）再乘回去。
        两种模式同一个式子：fit 的 `offset`、crop 的 `view` 都由它自己处理 ✓。
        """
        bx, by = mm.panel_to_canvas(0.0, 0.0, cal)
        z = float(mm.overlay_zoom(canvas_wh[0]))
        sx, sy = mm.scales_of(cal)
        return (bx * z, by * z,
                z / max(1e-6, float(sx)), z / max(1e-6, float(sy)))

    #: 差分视图里那个"圈"的**平滑系数**（一阶低通 ✓ 越大越跟手、越小越稳 ✓）。
    #: ⚠ 用户 2026-10-10："你的锁定标记会**乱飘**到别的点上去，根本没坐标偏差有做任何平滑"✗
    #: ⇒ 取 0.55：跟得上走、又不跟着单像素噪声跳 ✓。
    #: ⚠ 口径出处：`E:\MyPrograms\Maple_xfeat\src\vision\tracker.py` 里那套也是"连通域**质心** +
    #:   跨拍平滑"✓（那边 EMA α=0.85 是**喂速度**用的、不拿去当显示坐标 ✓ 见它的注释 ✓）。
    _DOT_EMA = 0.55
    #: 一跳超过这么多（面板像素 ✓）⇒ 判"真的换了个点" ⇒ **重锚**（不平滑过去 ✓）。
    #: 量级照 `Maple_xfeat` 的模板搜索半径（20px ✓ 见 `tracker.py` ✓）。
    _DOT_JUMP_PX = 20.0

    def _diff_dot_center(self, mask, di, cal, cw):
        """差分掩模里挑"最像玩家点"的那一块 ⇒ **质心**（面板像素 ✓）；挑不出来 ⇒ None ✓。

        为什么不直接用 `di["at"]`（= 差得最厉害的那个**像素** ✗）：那是 **argmax** ✗
        ⇒ ① 不是点的中心（用户："你**瞄准的不是黄点正中心**"✗）；
           ② 每拍在块内最亮的那个像素上跳 ⇒ **看着乱飘** ✗（用户："锁定标记会乱飘"✗）。
        挑法照 `Maple_xfeat` 的 `MinimapTracker`（`src/vision/tracker.py` ✓）那一套思路：
          · **连通域**（8 邻域 ⇒ 一个点不会碎成几块 ✓），太碎的（面积<3）和成片的（>900）都丢 ✓；
          · 打分 = **方**（长宽比）+ **实**（面积/外接框密度）− **离上一拍多远**（连续性 ✓
            ⇒ "锁住就不许乱跳" ✓）；
          · 取**质心**（`np.mean` ✓ 亚像素 ✓）当这一拍的观测 ✓，再做一阶低通（`_DOT_EMA` ✓）。
        """
        try:
            import cv2
            n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        except Exception:                           # noqa: BLE001 —— 算不动就不画圈（别乱画 ✗）
            return None
        prev = getattr(self, "_dot_ema", None)
        _pw = float(mask.shape[1]) if getattr(mask, "shape", None) else 240.0
        best, best_s = None, -1e9
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 3 or w < 3 or h < 3 or area > 900:
                continue
            dens = area / float(max(1, w * h))
            sq = 1.0 - abs(w - h) / float(max(1, max(w, h)))
            cx, cy = float(cents[i][0]), float(cents[i][1])
            d = 0.0 if prev is None else float(np.hypot(cx - prev[0], cy - prev[1]))
            # ⭐⭐ **"像不像玩家那个点"也算进来**（2026-10-10 ✓ 与识别层同一把尺子 ✓ 见
            #   `perception/minimap.dot_player_like` ✓）：这样圈出来的那个点，与
            #   Agent 锁的那个点是**同一套判据**挑的 ✓（不然会出现"圈的是 A、锁的是 B"✗）。
            like = mm.dot_player_like({"w": w, "h": h, "area": area}, _pw)
            s = 1.6 * like + 1.2 * dens + 1.0 * sq - 0.04 * d
            if s > best_s:
                best_s, best = s, (cx, cy, d)
        if best is None:
            return None
        cx, cy, d = best
        if prev is None or d > self._DOT_JUMP_PX * 3:
            self._dot_ema = (cx, cy)                   # 首次 / 真换点了 ⇒ 重锚 ✓
        else:
            a = self._DOT_EMA
            self._dot_ema = (a * cx + (1 - a) * prev[0], a * cy + (1 - a) * prev[1])
        return self._dot_ema

    def _live_map_tick(self):
        """把当前那块小地图面板按标定摆到地形图上（每 250ms 一拍 ✓）。

        跟「实测精度」那一路共用一个取帧口（`_mmap_panel_for_check(blocking=False)` ✓，
        **不阻塞** —— 定时器里不许等网络 ✗）。

        ⭐ **局部小地图这一层自己跟显示区**（`self._live_view_track` ✓）：它手里本来就有
        这一拍的面板 ⇒ 不依赖"画面那层叠图开着 / 世界坐标算得出来"那些条件 ✓。
        跟住 ⇒ 用新位置摆；没跟住 ⇒ 退回画面那层跟出来的 / 标定里那份，并在状态行**说清**
        （橙色 `⚠`，不是静默照标定 ✗ —— 那会让人以为"它在跟着走"）。
        """
        # ⚠ **不要**先判"有没有 live_panel"：来源=收流时那块面板是从 A 机那一口来的，
        #   跟本机实时页在不在跑没关系 ✓（那条路要 `live_panel` 的是"从实时画面框选"，
        #   由 `_mmap_panel_for_check` 自己按来源判 ✓）。
        # ⭐⭐ **每拍先把上一拍圈出来的点清掉**（2026-10-10 ✓）：不然那些**早退分支**
        #   （取不到面板 / 没标定 / 没底图 ✓）会把**上一拍那个圈**留在图上 ⇒ 看着像"它还在动"✗
        #   —— 而那正是"看得见的假结果"，比没有更坏 ✓。成功那条路会在本拍再画上去 ✓。
        self._live_mark = []
        self.canvas.set_markers([])
        mid = self._map_id()
        if not mid:
            self.canvas.set_live_patch(None)
            self.lbl_live_map.setText("先选地图 + 打开项目")
            return
        t = mapdata.load(mid, with_canvas=True)
        if t is None or t.canvas is None:
            self.canvas.set_live_patch(None)
            self.lbl_live_map.setText("这张图还没有底图 —— 先点「生成地形图」")
            return
        panel, why = self._mmap_panel_for_check(blocking=False)
        if panel is None:
            self.canvas.set_live_patch(None)
            self.lbl_live_map.setText("取不到面板：%s" % why)
            return
        cal = mapdata.load_calib(mid, self._mmap_src()) or {}
        if not mm.has_geometry(cal):
            self.canvas.set_live_patch(None)
            self.lbl_live_map.setText("这张图还没有标定（面板摆不上去）")
            return
        # ⭐⭐ **局部小地图：这一层自己跟"现在显示的是底图哪一块"**（用户 2026-09-29：
        #   "实时小地图的位置没有跟着我的移动变化"✗）。
        #   ⚠ 以前这里借的是 `self._ov_view`（**画面那层叠图**跟出来的），而那条路只在
        #     "叠图开着 **且** 世界坐标那行算得出来"时才更新 ⇒ **认不出黄点 / 没开叠图**
        #     的时候它一直是 None ⇒ 面板钉在标定那一刻的位置上，人走了也不动 ✗✗。
        #   现在：这一拍的面板就在手上 ⇒ 直接喂给自己那份 `CropViewTracker` ✓
        #     （`_refresh_world` 换图/重标定时 `reset` ✓）—— 跟住就用它，没跟住再退回
        #     `_ov_view`，都没有就照标定里那份摆（= 老行为 ✓，并在状态行说清 ✓）。
        follow, how = None, "照标定"
        if cal.get("mode") == mm.MODE_CROP:
            seed = self._ov_view or (cal.get("view") or (0.0, 0.0))
            vt = self._live_view_track.update(panel, t.canvas,
                                              dict(cal, view=[float(seed[0]),
                                                              float(seed[1])]))
            if vt.get("ok") and vt.get("view"):
                follow = [float(v) for v in vt["view"]]
                how = ("跟住了显示区 %.2f%s" % (float(vt.get("score") or 0.0),
                                              "" if vt.get("trust") else "·不太稳"))
            elif self._ov_view:
                follow = [float(v) for v in self._ov_view]
                how = "用画面那层跟出来的显示区"
            else:
                follow = None
                how = "⚠ 显示区没跟住 ⇒ 照标定里那个位置摆"
            if follow is not None:
                cal = dict(cal, view=follow)
        from gui.minimap_calib import np_to_pixmap      # 一处实现（别各写一份 ✓）
        ch, cw = t.canvas.shape[:2]
        ox, oy, kx, ky = self._live_map_rect(cal, (cw, ch))
        hp, wp = panel.shape[:2]
        # ⭐ **像素差分地图**（用户 2026-10-06 ✓）：摆上去的是"**面板 − 底图那一块**"，
        #   不是面板本身 ✓ —— 于是"底图上没有的东西"（玩家点 / 实时元素 ✓）会跳出来 ✓。
        #   ⚠ 算不出来（面板落在底图外 / 标定读不出 / 形状怪 ✓）⇒ **照旧摆面板**
        #     （把画面弄空才是更坏的 ✗）＋ 一行说清 ✓ 不许静默 ✓。
        show, diff_txt, diff_warn = panel, "", False
        #: 差分那一层的**差异掩模**（给"透明底"和"找点"两处共用 ✓ 别各算一遍 ✗）
        _diff_alpha = None
        if self._live_disp == "diff":
            # ⭐ `color=True`：**把点的颜色显示出来**（用户 2026-10-06 ✓ 原话："像素差分地图
            #   能把点的颜色显示出来而不是纯白吗？"✓）—— 差得够亮的那块按**面板本来的颜色**
            #   画 ✓（玩家点还是黄的 ✓）；默认那份灰阶留给**定位**（`dot_candidates_diff` ✓
            #   它的门限是"峰值比例"⇒ 换颜色就变味 ✗ 见那边说明 ✓）。
            vis, di = mm.diff_panel_vs_canvas(panel, t.canvas, cal, color=True)
            if vis is None:
                diff_txt = "　⚠ 差分算不出来（%s）⇒ 照旧摆面板" % (di.get("why") or "?")
                diff_warn = True
            else:
                wx, wy = mm.panel_to_world(di["at"][0], di["at"][1], cal, t)
                diff_txt = ("　差分 ×%g：背景档 %g 灰阶（整幅均值 %g · 最大 %g）· "
                            "最突出 %g 在面板 (%d, %d) ⇒ 世界 (%d, %d)"
                            % (di["gain"], di["floor"], di["raw_mean"], di["raw_peak"],
                               di["peak"], di["at"][0], di["at"][1],
                               round(wx), round(wy)))
                show = vis
                # ⭐⭐⭐ **把"这一拍找到的那个点"圈到地形图上**（用户 2026-10-10 ✓ 原话：
                #   "你是应该以**差分地图**来找玩家坐标的，那我需要**在这里看到你的结果**，
                #    例如把你**找到的黄点圈出来**" ✓✓）。
                #   ⚠⚠ 但**不能用 `di["at"]`** ✗ —— 它是"差得最**厉害**那个**像素**"（argmax ✓）：
                #     ① 不是点的**中心**（用户："你**瞄准的不是黄点正中心**"✗）；
                #     ② 每拍在最亮的那个像素上跳来跳去 ⇒ **看着乱飘** ✗（用户第二条 ✓）；
                #   ⇒ 改成"**连通块质心 + 跨拍平滑**"（口径与 `E:\MyPrograms\Maple_xfeat`
                #     `src/vision/tracker.py` 一致 ✓ 那也是**连通域质心**而不是 argmax ✓）：
                #     ① 取"够亮"的像素做连通域（阈值 = 峰值的一半，且不低于噪声底线 ✓）；
                #     ② 挑"最像玩家点"的那一块（**离上一拍最近** + 块**圆一点、大一点** ✓）；
                #     ③ 用**质心**（`np.mean` ✓ 亚像素 ✓）当这一拍的观测；
                #     ④ 再对位置做**一阶低通**（`_DOT_EMA` ✓）—— 人眼看的就是它 ✓ 不能飘 ✓。
                # 差异掩模：阈值 = 峰值一半、且不低于噪声底线（`mm.DOT_DIFF_MIN_CONTRAST` ✓）
                #   —— 同一张掩模**两用** ✓：① 给这层加 alpha（只画差异、其余透明 ✓ 用户第四条 ✓）；
                #   ② 给"挑点"用（连通域 ✓ 见 `_diff_dot_center` ✓）。
                _pk = float(di.get("peak") or 0.0)
                _thr = max(float(mm.DOT_DIFF_MIN_CONTRAST), 0.5 * _pk)
                _diff_alpha = ((vis.max(axis=2).astype(np.float32) >= _thr)
                               .astype(np.uint8) * 255)
                # ⛔⛔ **这里原来把"差分自己挑的点"当玩家圈上去** —— **2026-10-10 用户否掉** ✗
                #   （原话："**地形图都锁到红点上去了，报的坐标也与信息栏不一致，这个不应该
                #    自己走独立逻辑，它就应该根据最终信息栏读到的玩家世界坐标来绘制**"✓）：
                #   · `_diff_dot_center` 挑的是"**差得够亮的那一块**"（连通域质心 ✓）——
                #     底图/地图上只要"与面板不同源"的东西（**墙的红点**、地形线、集合色块 ✓）
                #     差得够亮就会被当成玩家 ⇒ **锁到红点上** ✗（截图现场 ✓）；
                #   · 而它报的世界坐标与信息栏那行**必然是两个数** ✗（两条链 ✓ ——
                #     这正是 2026-10-06 拆过一次的老毛病 ✓ 见 `decision/agent.LAST_POS` ✓）。
                #   ⇒ 玩家圈**只认唯一那一份**（下面那段 ✓）；差分**照旧画**（它是给人看
                #     "差在哪"的一层 ✓ 用户 2026-10-06 定的 ✓）—— 但它**不再代表玩家** ✗。
                #   ⚠ 那一层自己挑出来的块**照样算出来、只是只写成一句文字** ✓（用户 2026-10-10
                #     要过"我要在这里看到你的结果"✓ ⇒ 不能干脆删掉 ✓；但它**不许再当玩家位置** ✗）
                #     —— 这样 `_diff_dot_center` 也不必变成死代码 ✓。
                _dpt = self._diff_dot_center(_diff_alpha, di, cal, cw)
                if _dpt is not None:
                    _dwx, _dwy = mm.panel_to_world(_dpt[0], _dpt[1], cal, t) \
                        if cal.get("mode") != mm.MODE_CROP else (None, None)
                    diff_txt += "　（差分那层自己挑的块在面板 (%.0f, %.0f)%s —— 仅供看，**不代表玩家** ✗）" \
                        % (_dpt[0], _dpt[1],
                           "" if _dwx is None else
                           "，按标定算出来是 (%d, %d)" % (round(_dwx), round(_dwy)))
                if di["floor"] >= DIFF_FLOOR_WARN:
                    diff_txt += "（背景档偏大 ⇒ 这张图上底图与游戏小地图不同源，只能当辅助看）"
                    diff_warn = True
        # ⭐⭐⭐⭐⭐ **玩家那个圈 = 唯一那一份世界坐标**（用户 2026-10-10 ✓ 原话："**地形图都锁到
        #   红点上去了，报的坐标也与信息栏不一致，这个不应该自己走独立逻辑，它就应该根据
        #   最终信息栏读到的玩家世界坐标来绘制**"✓）。
        #   · 来源：`decision.agent.LAST_POS` ✓（**实时回路是唯一写者** ✓ 见
        #     `gui/live_thread._fill_route_ctx` ✓）—— 与信息栏那行、Agent 决策**同一个数** ✓；
        #   · 换算：只走**地形**（`terrain.world_to_canvas` ✓ ⇒ 与标定/显示区那套**无关** ✓
        #     —— 所以它不会因为 `view` 跟丢而漂 ✗）；
        #   · ⚠ 换图 / 实时没在跑 / 这一拍还没定上位 ⇒ **不画** ✓（绝不自己再算一份 ✗
        #     —— 那正是要拆掉的东西 ✓）；沿用上一帧的（`held` ✓）照画 ✓ 但**如实标注** ✓。
        _lp = None
        try:
            from decision import agent as _agent_mod
            _lp = getattr(_agent_mod, "LAST_POS", None)
        except Exception:                       # noqa: BLE001 —— 公开失败不该弄坏绘制 ✓
            _lp = None
        if isinstance(_lp, dict) and str(_lp.get("map_id") or "") == str(mid):
            _wx, _wy = _lp.get("world_x"), _lp.get("world_y")
            if _wx is not None and _wy is not None:
                try:
                    _cx, _cy = t.world_to_canvas(float(_wx), float(_wy))
                    _z = float(mm.overlay_zoom(cw))
                    self._live_mark = [(
                        _cx * _z, _cy * _z,
                        "玩家 世界(%d, %d)%s" % (round(float(_wx)), round(float(_wy)),
                                                "·沿用上一帧" if _lp.get("held") else ""),
                        "#ffd400", 12.0)]
                except Exception:               # noqa: BLE001
                    pass
        # ⭐⭐ **差分那一层要"只把差异画出来、其余透明"**（用户 2026-10-10 ✓ 第四条：
        #   "还是没有在像素差分地图汇总**看到我编辑的地形**（foothold 集合、楼梯、传送点…）"✗）：
        #   原因很具体 —— `mode=fit`（全局小地图）时**面板正好盖住整张底图** ✗，而差分图
        #   背景是黑的 ⇒ 黑底一铺 ⇒ **底下那张地形图（集合颜色 / 楼梯 / 传送点）全被盖掉** ✓✓。
        #   ⇒ 这里给它加 **alpha 通道**：差异像素不透明（点看得见 ✓）、其它全透明（地形透出来 ✓）。
        _patch = np_to_pixmap(show)
        if self._live_disp == "diff" and not diff_warn and _diff_alpha is not None:
            try:
                if _diff_alpha.shape[:2] == show.shape[:2]:
                    # ⚠ 必须走**四通道**那条（`np_to_pixmap` 认 BGRA ⇒ `Format_ARGB32` ✓）——
                    #   三通道的图会被当成 BGR 逐像素错读 ✗（见 `gui/minimap_calib.np_to_pixmap` ✓）。
                    _patch = np_to_pixmap(np.dstack([show, _diff_alpha]))
            except Exception:                       # noqa: BLE001 —— 加不上就照旧（别把画面弄空 ✗）
                pass
        ok = self.canvas.set_live_patch(_patch, (ox, oy), kx, ky)
        if not ok:
            self.lbl_live_map.setText("地形图还没画出来（先点「生成地形图」）")
            return
        # ⭐⭐ 把这一拍算出来的点**圈**上去（`set_markers` ✓ 见它说明 ✓）；
        #   ⚠ 放在 `set_live_patch` **之后** ✓ —— 它要场景里已经有地形图才画得上 ✓。
        n_mark = self.canvas.set_markers(self._live_mark)
        if n_mark:
            diff_txt += "　· 已在地形图上**圈出 %d 个点**（黄圈 = 差分找到的那个 ✓）" % n_mark
        warn = how.startswith("⚠") or diff_warn
        self.lbl_live_map.setText(
            "面板 %d×%d 摆在地图 (%d, %d) 起、%.2f×（%s）%s%s"
            % (wp, hp, round(ox), round(oy), kx, how,
               "" if cal.get("mode") == mm.MODE_CROP
               else "　（全局小地图：整张底图都在面板里、不滚动 ✓）",
               diff_txt))
        self.lbl_live_map.setStyleSheet("color: %s;" % ("#b06000" if warn else "#80868b"))

    def _calib_panel_wh(self, cal=None):
        """**标定当时那块面板的尺寸** `(w, h)` —— 画叠图时要把标定几何折算到当前画面。

        为什么需要（2026-09-26 用户报「叠图被放大」）：标定记的是"那块面板像素 ↔ 底图
        像素"的换算，而叠图要画在"**现在**框出来的那一块"上 —— 两者尺寸可能差好几倍
        （典型：A 机推流带 zoom）⇒ 不折算就把整块叠图放大。用户的原话也是这个意思：
        「那个倍率只是算坐标用的」。

        取法（按可靠程度）：
          1. 标定文件里的 `panel`（新存的有；老标定没有）；
          2. **当前收流帧**的尺寸 —— 来源=收流时它正是标定那块面板（zoom=1 时 =
             真实面板大小）；
          3. 都拿不到 ⇒ None（退回老行为，不折算；状态行会提示"比面板框还大"）。
        """
        if cal is None:
            try:
                cal = mapdata.load_calib(self._map_id(), self._mmap_src()) or {}
            except Exception:                       # noqa: BLE001
                cal = {}
        p = cal.get("panel")
        if p and len(p) == 2:
            try:
                if int(p[0]) > 0 and int(p[1]) > 0:
                    return (int(p[0]), int(p[1]))
            except (TypeError, ValueError):
                pass
        if self._mmap_src() == mm.SRC_STREAM:
            cli = self._stream_client()
            if cli is not None:
                f, _t = cli.latest()
                if f is not None:
                    return (int(f.shape[1]), int(f.shape[0]))
        return None

    def _refresh_overlay(self, view=None):
        """照标定算出「叠图的哪一块画到画面的哪里」，交给实时面板去画。

        算出来的矩形是**画面坐标**：面板在画面里的位置来自 `mmap_crop`，
        面板里怎么摆来自标定（`perception/minimap.frame_overlay_rects` 一份公式，
        和标定弹窗的叠加层同源）。

        `view` = **这一拍跟出来的**「显示区起点」（底图像素；只有局部小地图才用得上 ✓，
        见 `perception.minimap.CropViewTracker`）—— 传了就**压过标定里那份** ✓：
        crop 的面板随玩家滚动，标定里那个只在标定那一刻成立，不压的话叠图会停在
        标定那一刻的位置、和人看到的小地图对不上 ✗（用户 2026-09-29 问"现在已经能
        实时地滚动底图了？" ✓）。不传 = 照标定里那份画（老行为 ✓）。

        画/没画、画的哪儿、没画卡在哪 —— 都写在勾选框旁边那行上。
        """
        if not view:
            # 没带"这一拍跟出来的" ⇒ 这一层回到"照标定画"的状态；把记住的那个清掉，
            # 下一拍（`_tick_world`）好照新跟出来的重画一次 ✓（不清的话它会以为
            # "已经画在当前那个显示区上了"、于是不再更新 ✗）。
            self._ov_view = None
        lp = getattr(self, "live_panel", None)
        on = self._mmap_draw_on()
        why = self._overlay_blocker()
        if on and not why and lp is not None:
            mid = self._map_id()
            t = mapdata.load(mid, with_canvas=True)
            # ⚠ **框选区域取一处口径**（`_mmap_crop` = `mm.crop_of`：本项目优先、
            #   没框过才回退老的那份全局值 ✓）。这里原来直接读 `load_live()["mmap_crop"]`
            #   ✗ —— 而框选区域 2026-09-27 起**按项目存**（project.yaml）⇒ 只框了本项目
            #   的图上，叠图会被摆到"老那份全局值"的位置（多半是上一张图的框）✗。
            crop = self._mmap_crop() or []
            if len(crop) == 4 and t is not None and t.canvas is not None:
                cal = mapdata.load_calib(mid, self._mmap_src()) or {}
                if view:
                    cal = dict(cal, view=[float(v) for v in view])
                src, dst = mm.frame_overlay_rects(
                    cal, [int(v) for v in crop],
                    (t.canvas.shape[1], t.canvas.shape[0]),
                    calib_panel=self._calib_panel_wh(cal))
                pix = self._overlay_pix(mid)
                # ⚠ 浓淡只能从 `overlay_alpha()` 取：原来写 `cal.get("alpha") or 55`，
                #   而 **0 是 falsy** ⇒ 用户把滑块拉到最左端（0 = 看不见），显示出来的却是
                #   55% ✗（2026-09-26 用户报的那个 bug ✓）。默认值也只有一处实现 ✓。
                alpha = mm.overlay_alpha(cal)
                # `note` = 玩家世界坐标那行：贴在画面里那块框的下面（见
                # gui/live_panel._draw_note）。每次重画都要重贴 —— 帧是新的。
                lp.set_minimap_overlay(pix, src, dst, alpha,
                                       note=self._world_note)
                # ⚠ 叠图比**面板框**还大 ⇒ 标定和这块框对不上（2026-09-26 实测踩过：
                # 标定文件里 `scale=5.63`（在 A 机 zoom=3 的收流帧上按"整图 fit"
                # 拟合出来的、匹配分才 0.70）⇒ 底下算出来 134×101×5.63 = 754×569，
                # **整屏都是它**。说在明面上，别让人对着画面猜「是不是程序画错了」。
                # ⚠ 这句**现在真的只意味着"标定不对"**（2026-09-26 改）：以前还会在
                #   "收流第一帧还没来、折算比例算不出来"时冒出来喊"标定八成不对" ——
                #   那时标定其实是好的、只是尺寸还不知道 ✗（用户报的那个现象就是这么
                #   被冤枉的）；现在那种情况已经被 `_overlay_blocker` 拦在前面 ✓。
                #   所以文案里要**把两个尺寸都摆出来**（叠图几×几、框几×几）：人一眼
                #   就能看出是差一截还是差好几倍，而不是只被告知"八成不对"。
                warn = ""
                if dst[2] > int(crop[2]) * 1.2 or dst[3] > int(crop[3]) * 1.2:
                    hint = ("确认 A 机推流的 zoom 是 1、「显示方式」全局/局部选对了，再标一次"
                            if self._mmap_src() == mm.SRC_STREAM else
                            "来源是「从实时画面」，面板就是框出来的这一块 ⇒ "
                            "打开「标定…」对着它重标一次")
                    warn = ("　⚠ 叠图 %d×%d 比面板框 %d×%d 还大 —— 标定和这块框对不上：%s"
                            % (dst[2], dst[3], int(crop[2]), int(crop[3]), hint))
                self.lbl_mmap_draw.setText(
                    "已画在画面 (%d, %d) %d×%d　浓淡 %.0f%%（在 设置 → 界面 里调）%s"
                    % (dst[0], dst[1], dst[2], dst[3], alpha * 100, warn))
                self.lbl_mmap_draw.setStyleSheet(
                    "color:#b06000;" if warn else "color:#188038;")
                return
            why = "算不出往画面的哪儿画（框选/标定/底图不齐全）"
        if lp is not None:
            lp.set_minimap_overlay(None)
        if not on:
            self.lbl_mmap_draw.setText("（没开）")
            self.lbl_mmap_draw.setStyleSheet("color:#80868b;")
        else:
            self.lbl_mmap_draw.setText("没画：%s" % why)
            self.lbl_mmap_draw.setStyleSheet("color:#b06000;")

    # ---------------- 地形图 ----------------

    def _map_image_path(self, mid):
        """这张图该显示哪张图 → (路径或 None, 标题, 附注)。

        优先 `<id>_zones.png`（**地形编辑器的结果**：颜色 = 集合、名字标在平台上方、
        灰 = 还没圈进任何集合）—— 那才是"程序认得的平台"，寻路要照它走。
        没有就退到 `<id>_overlay.png`（每段一色 + 段号；**叠到实时画面上用的仍是它**），
        再退到小地图底图 `<id>.png`；都没有说明地形还没导出，把该跑的命令写出来。

        **分三段返回**（不是拼成一整句）：中间那段「几×几 像素」得**量了文件**才知道，
        而那只 QPixmap 在调用方（它本来就要为显示加载一次，不多花一次解码）。
        拼成一整句的话，"几×几"要么掉到命令行提示后面，要么得在这儿再解码一遍。
        """
        d = mapdata.map_dir()
        zi = d / ("%s_zones.png" % mid)
        if zi.exists():
            from core import zones as zones_mod
            try:
                n = len(zones_mod.load(mid).sets)
            except Exception:               # noqa: BLE001
                n = -1
            tip = ("（地形编辑器的结果：颜色 = 集合，名字标在平台上方）"
                   if n != 0 else
                   "（还没有集合 —— 点「寻路编辑器」圈一个，颜色和名字才出得来）")
            return zi, ("集合图 %s" % zi.name), tip
        over = d / ("%s_overlay.png" % mid)
        if over.exists():
            return over, ("地形叠加图 %s" % over.name), (
                "（每段一色 + 段号。想要**集合版**就点下面的「生成地形图」重画一张）")
        base = d / ("%s.png" % mid)
        if base.exists():
            return base, ("小地图底图 %s" % base.name), (
                "（还没有叠加图 —— 点下面的「生成地形图」画一张看得更清楚的；"
                "也可以手跑：\n    python -m tools.map_terrain_view %s）" % mid)
        return None, "这张图还没有地形数据", (
            "（datasets/map/%s.json 不存在。）\n"
            "点下面的「生成地形图」会自动导出并画出来（也可以手跑：\n"
            "    WzProbe.exe dump-terrain <WZ目录> datasets/map --only %s）"
            % (mid, mid))

    def _refresh_map_image(self):
        mid = self._map_id()
        if not mid:
            self.lbl_map_img.setText(
                "先在①选地图。" if getattr(self, "project", None) is not None
                else "打开一个项目后再看。")
            self.canvas.load(QPixmap(), [], editable=False, fit=True)
        else:
            # 集合图是**派生数据**（实测 38ms），缺了就顺手补上 —— 否则打开面板看到的
            # 还是旧那版叠加图，"地形图显示编辑器结果"这件事就只改了一半，
            # 还得人自己去点一次「生成地形图」。写不出来就退回叠加图（下面那条路）。
            if not (mapdata.map_dir() / ("%s_zones.png" % mid)).exists():
                self._render_zones_png(mid)
            path, title, extra = self._map_image_path(mid)
            pm = QPixmap(str(path)) if path is not None else QPixmap()
            # 「几×几」放在**标题后面、附注前面**：它是这张图的第一眼信息，
            # 掉到那串命令行提示后面就没人看得见了。数用**真文件**量出来的
            # （`pm` 本来就要为显示加载，不多花一次解码）—— **不是推算**：
            # 推算一旦和实际文件不一致（换过图、改过生成参数），报出来的就是
            # 假数，而"看着像对的假数"最难发现。
            size = "" if pm.isNull() else "%d×%d 像素" % (pm.width(), pm.height())
            self.lbl_map_img.setText("　".join(
                x for x in (title, size, extra) if x))
            # 选了「选择平台」就把它框出来（只读覆盖层，图本身不变）
            _pv = self._preview_boxes(mid)
            self.canvas.load(pm, _pv, editable=False, fit=True)
            # ⭐⭐ **预览框上的字改成「平台名」**（用户 2026-10-06 ✓ 原话："它上面写着"怪物"
            #   ⇒ 换成平台名吧"✓）。
            #   为什么会显示"怪物"：这个框走的是**通用画布 / 质检台同一套** `BBoxItem` ✓
            #   （用户现场看到的就是"**就像一个标注框，跟质检台那个一样**"✓），而它的标签默认
            #   取**类别名表**（`class_names=None` ⇒ "0=玩家 / 1=怪物" ✓）⇒ 一块**平台预览框**
            #   被标成 **"怪物"** ✗ ⇒ 现场真的被当成"怪物检出框"来问 ✓（误导实锤 ✓）。
            #   ⚠ 只在**这块预览框**上改字 ✗：不动类别、不动颜色表、不动质检台/工作台那边的
            #     任何框（它们的标签仍归 `set_class_names` 管 ✓ 一处口径还在 ✓）。
            _pv_name = self._goto_name()
            if _pv and _pv_name:
                for _it in self.canvas.boxes:
                    _it.label.setText(_pv_name)

        # 没地图就没得生成；正在跑的时候也不让重复点
        self.btn_gen.setEnabled(bool(mid) and self.task is None)

    # ---------------- 生成地形图 ----------------

    def _generate(self):
        """点「生成地形图」：缺 WZ 数据就先导出，然后画叠加图。

        整件事丢到后台线程（见 gui/worker.py 的铁律：工作线程只 emit 信号，
        绝不碰控件）—— 导出要解析 Map.wz，几十秒，卡住界面就没法用了。
        """
        mid = self._map_id()
        if not mid:
            QMessageBox.information(
                self, "先选地图",
                "先在①「识别目标选项」里选地图 —— 才知道要导哪一张。")
            return
        if self.task is not None:
            return

        self.btn_gen.setEnabled(False)
        self.lbl_gen.setStyleSheet("color: #5f6368;")
        self.lbl_gen.setText("开始…")

        from gui.worker import TaskThread, safe_slot
        self.task = TaskThread(_generate_task, {"map_id": mid}, self)
        # 槽函数一律过 safe_slot：槽里抛异常会让 PyQt5 直接 abort() 整个程序
        self.task.sig_log.connect(safe_slot(self._on_gen_log))
        self.task.sig_done.connect(safe_slot(self._on_gen_done))
        self.task.finished.connect(safe_slot(self._on_gen_finished))
        self.task.start()

    def _on_gen_log(self, msg, _level="info"):
        """把子进程最后一行输出显示在按钮旁边（长路径截断）。"""
        line = " ".join(str(msg).split())
        self.lbl_gen.setText(line[:120] + ("…" if len(line) > 120 else ""))

    def _on_gen_done(self, ok, summary, _result=None):
        self._ov_pix = None                # 叠加图重画了 → 丢掉缓存的旧图
        self._refresh_map_image()          # 画好了立刻换图
        self._refresh_overlay()            # 叠到实时画面那一层也要换成新的
        self.lbl_gen.setStyleSheet("color: %s;" % ("#137333" if ok else "#c5221f"))
        self.lbl_gen.setText(("✓ " if ok else "✗ ") + (summary or ""))

    def _on_gen_finished(self):
        # 线程真正结束才释放引用（和 export_dialog / main_window 同一套写法）
        t = self.task
        self.task = None
        if t is not None:
            t.deleteLater()
        self.btn_gen.setEnabled(bool(self._map_id()))

    def _on_mmap_mode(self, _i):
        mid = self._map_id()
        if not mid:
            return
        src_kind = self._mmap_src()
        cal = mapdata.load_calib(mid, src_kind) or {}
        # **只改方式**，量出来的几何参数（scale/offset/view）保留
        cal["mode"] = self.cmb_mmap_mode.currentData()
        cal["picked_by"] = "gui"
        cal.pop("legacy", None)
        # 带来源：不带就会整份覆盖，把另一条来源的标定抹掉（人看不出来）
        mapdata.save_calib(mid, cal, src_kind)
        self._refresh_mmap()        # 里面会顺带按新方式重算叠图（fit/crop 画法不同）

    def _on_align_gap(self, v):
        """改了「对齐绳梯移动延迟(ms)」⇒ 写进配置（同样是决策参数、跟着项目存）。"""
        settings.climb_align_gap_ms = int(v)
        settings.save()


    def _on_align_near(self, v):
        """改了「开始对齐绳梯x的距离(px)」⇒ 写进配置（同样是决策参数、跟着项目存）。"""
        settings.climb_align_near_px = int(v)
        settings.save()

    def _on_retry_delay(self, v):
        """改了"失败后延迟激活时间" ⇒ 写进配置（它是决策参数，跟着项目存）。"""
        settings.climb_retry_delay_s = float(v)
        settings.save()

    def _on_retry_delay_inc(self, v):
        """改了"延迟增量" ⇒ 写进配置（同样是决策参数、跟着项目存）。"""
        settings.climb_retry_delay_inc_s = float(v)
        settings.save()

    def _on_retry_gap(self, v):
        """改了「移动操作尝试间隔(ms)」⇒ 写进配置（同样是决策参数、跟着项目存）。

        一个参数管两处（爬绳补按 / 下跳重试，见 `DecisionSettings.move_retry_ms` 的说明）：
        **毫秒整数**存一份，agent 在灌进任务时各折算一次（爬绳要秒、下跳要毫秒 ✓）。
        """
        settings.move_retry_ms = max(0, int(v))
        settings.save()

    def _on_no_chase_path(self, v):
        """改了「**禁用杀怪寻路**」⇒ 即改即存（同这一组其它参数 ✓）。

        ⚠ 它**不按地图 id 存**（与上面那几个攀爬参数不同 ✗）：它是 `DecisionSettings` 的字段
          ⇒ 跟着**当前项目**走（`project.yaml` 的 `decision:` 段 ✓ 见 `to_dict` ✓）。
        ⚠ 写的是**同一个 `settings` 单例**（agent 每拍读它 ✓）⇒ 改完**立刻生效**、
          不用重启、也不用重开自动 ✓（同 `_on_retry_gap` 那批 ✓）。
        """
        b = bool(v)
        if b != bool(getattr(settings, "disable_chase_pathfinding", False)):
            settings.disable_chase_pathfinding = b
            settings.save()     # 没打开项目时不落盘（见 decision/agent 的 set_save_hook）

    def _on_jump_start(self, v):
        """改了「起跳距离(px)」⇒ 写进配置（决策参数、跟着项目存 ✓）。

        ⚠ 它是「跳」的**临时参数**：物理换算做好之后会被删掉 ✓
          （见控件 tooltip / `DecisionSettings.jump_start_px`）。
        """
        settings.jump_start_px = max(0, int(v))
        settings.save()

    def bind(self, project):
        """切项目：本页要跟着换的是**图**和**集合下拉**（都按地图 id 存）。

        以前这里还要回填一个"启用路线识别"的开关（还专门处理过 blockSignals 的坑）——
        那个开关和它门控的感知一起撤掉了（见 __init__ 的说明），所以只剩这几件。
        """
        self.project = project
        # 延迟激活时间也是决策参数（按项目存）⇒ 换项目重读一遍；blockSignals 别把
        # 刚读出来的值又写回去（和以前那个开关踩过的坑是同一个）。
        self.sp_align_gap.blockSignals(True)
        self.sp_align_gap.setValue(int(getattr(settings, "climb_align_gap_ms", 180) or 0))
        self.sp_align_gap.blockSignals(False)
        self.sp_align_near.blockSignals(True)
        self.sp_align_near.setValue(
            int(getattr(settings, "climb_align_near_px", 20) or 20))
        self.sp_align_near.blockSignals(False)
        self.sp_retry.blockSignals(True)
        self.sp_retry.setValue(float(getattr(settings, "climb_retry_delay_s", 1.0)))
        self.sp_retry.blockSignals(False)
        self.sp_retry_inc.blockSignals(True)
        self.sp_retry_inc.setValue(
            float(getattr(settings, "climb_retry_delay_inc_s", 1.0)))
        self.sp_retry_inc.blockSignals(False)
        self.sp_retry_gap.blockSignals(True)
        self.sp_retry_gap.setValue(int(getattr(settings, "move_retry_ms", 3000) or 0))
        self.sp_retry_gap.blockSignals(False)
        # ⛔ 这里原来回填「前往重下间隔(s)」—— 那一格 2026-09-28 移除了（搬进「战斗区域」
        #   弹窗、逐项配 ✓ 见 `DecisionSettings.battle_zones` ✓）。
        # ⚠ 「禁用杀怪寻路」**不在这里回填** ✗ —— 它 2026-10-06 起是**按地图 id 存**的
        #   （`datasets/map/<id>.route.json` ✓ 用户口径："从此…按地图id存的数据"✓）⇒ 它的值
        #   由 `_apply_route_cfg()` 在**按图灌值之后**摆 ✓（在本函数**末尾**调 ✓ 见那儿 ✓）。
        #   在这里回填读到的会是**上一个项目/上一张图**的值 ⇒ 正好串图 ✗（同 `_seed_route_cfg`
        #   那条纪律："别拿 settings 内存值当按图参数的来源"✗）。
        # 「起跳距离(px)」（「跳」的临时参数 ✓）也是决策参数 ⇒ 换项目重读一遍 ✓
        self.sp_jump_start.blockSignals(True)
        self.sp_jump_start.setValue(int(getattr(settings, "jump_start_px", 0) or 0))
        self.sp_jump_start.blockSignals(False)
        self._refresh_mmap()
        self._refresh_map_image()
        self._refresh_goto()        # 换图/换项目：下拉要跟着换成这张图的集合
        # 「路线规划」的战斗区域**按地图 id 存** ⇒ 换图时从这里读进 settings（必要时先从旧
        #   project.yaml 迁移一次 ✓）—— 它内部会顺手重画摘要 ✓。
        self._load_battle_zones_for_map(project)
        # ⭐⭐ 「路线识别」那批配置**也按地图 id 存**（用户 2026-10-03 ✓ 见 `core/route_cfg.py`）：
        #   换项目/换图时把**这张图**那份灌进来；文件不存在 ⇒ 从老家**播种**一份（口径 ① ✓）。
        #   ⚠ 必须排在 `_load_battle_zones_for_map` **之后**（两者都是"按图"的 ✓ 顺序无依赖 ✓
        #     但一起做、一起读文件更好排查 ✓）；它内部会顺带刷「小地图定位」那几行 ✓。
        self._apply_route_cfg(self._map_id())

    def showEvent(self, e):
        """切到「路线识别」页签时重读一次。

        为什么需要：地图是在①「识别目标选项」里选的，而这个页签**不会**跟着刷新 ——
        选完地图切过来还是旧值（甚至停在「（这个项目还没选地图）」），
        看着像①的选择没生效。按页签刷新的时机来重读，代价只是一次文件读取。
        """
        super().showEvent(e)
        if getattr(self, "project", None) is not None:
            self._refresh_mmap()
            self._refresh_map_image()
        # ⭐ 「显示类型」那条 **只在看得见的时候跑**（切到别的页签就停 ——
        #   4 次/秒的取帧+重绘虽然便宜，但没人看的时候白花 ✓）。默认是「实时小地图」⇒
        #   进这一页就该看见它在滚（取不到面板时右边那行会说清为什么 ✓）。
        if (getattr(self, "cmb_live_view", None) is not None
                and self._live_disp != "terrain"):
            if not self._live_map_timer.isActive():
                self._live_map_timer.start()
            self._live_map_tick()

    def hideEvent(self, e):
        """切走这个页签 ⇒ 停掉「叠加实时小地图」那条节拍（没人看就不取帧 ✓）。"""
        super().hideEvent(e)
        if getattr(self, "_live_map_timer", None) is not None:
            self._live_map_timer.stop()
        # 透明度可能拖完没松手就切页了 ⇒ 这里补一次落盘（拖的时候不写盘，见 `_on_live_alpha`）
        if getattr(self, "sld_live_alpha", None) is not None:
            self._save_live_alpha()


