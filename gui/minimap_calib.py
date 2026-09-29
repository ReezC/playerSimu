"""小地图标定弹窗：量出「实时小地图面板 ↔ 底图 ↔ 世界坐标」的换算。

**为什么要做成看得见的弹窗**（而不是让你在命令行跑一遍）
    标定这件事的正确判据是**看得见**：左边是 A 机推来的实时面板，叠着半透明的
    底图，重合不重合一眼就能判断。只有一串 `scale=1.406 offset=(12,7)` 的数字，
    对错全凭猜 —— 而这几个数直接决定「我在哪块平台上」。

**交互照「手动目测标定尺度」（gui/calib_manual.py）那套**：
    拖半透明叠加层对齐 → 目测确认 → 「保存标定」记下当前几何
    （那边是拖模板量怪的大小，这边是拖底图量面板的缩放/偏移，动作是同一个）。

**两条路都能走**：
    「自动定位」  用模板匹配算出 scale/offset/view（准，1 像素级）
    「手动拖动」  匹配分低/面板被 UI 挡住时，人把底图拖到与面板重合的位置
                 （方向键 1 像素微调，Shift+方向键 5 像素）

**两轴（x/y 各自一个缩放）**：本窗**认**双点标定量出来的两轴几何 —— 读进来、按各自的
缩放画叠加层、保存时原样写回；人拖动/改缩放时两轴**按同一比例**一起变（比例要改，用
「双点标定」重量一次）。⚠ 这条是 2026-09-27 修的：以前本窗只认一个缩放，双点标定的
几何一进一出就被抹成等比（`scale_y` 被丢掉），界面还只显示一个数 ✗。

**显示方式（全局 / 局部）不在这里改**：那要进游戏走两步看「地形动不动」才知道，
所以沿用「路线识别」下拉里你选的那个 —— 和命令行工具的态度一致（程序不替你猜）。
"""

import time

from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QImage, QPainter, QPixmap, QTransform
from PyQt5.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QDialog,
                             QFrame, QGraphicsPixmapItem, QGraphicsScene,
                             QGridLayout, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)

from core import mapdata
from gui import theme                  # 弹窗几何按客户端存（config/ui.yaml）✓
from gui.canvas import ZoomPanView    # 看图交互在几个窗口里是同一份
from gui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSlider
from gui.worker import safe_slot      # 槽里抛异常 = 整个工作台 abort（见 worker.py）
from perception import minimap as mm
from tools.config import load_live

#: 版面：一行里"参数名"那格与"单位"那格的**统一宽度** ⇒ 各行的拖动条/数字框能对齐 ✓
#: （对齐是"读起来像一张表"的关键 —— 参差的行让人没法竖着比对两轴 ✓）。
ROW_LABEL_W = 84
ROW_UNIT_W = 74

#: 「轴偏移」两条拖动条：滑块是整数 ⇒ 用 ×10 当刻度（**0.1 一个格**，见 `_offset_row`）。
#: 值域 = `[-2000, 2000]`（面板像素 / 底图像素都够用：fit 的 offset 是几百像素级、
#: crop 的 view 是底图坐标、也就几百）。
OFF_MIN, OFF_MAX = -20000, 20000
OFF_STEP = 10

#: 缩放滑块：0.200 ~ 12.000。
#: 上限从 4.0 放到 12.0：小底图（几十像素）被客户端放大若干倍、A 机的推流 zoom
#: 又乘一次，面板里的实际倍数能到 9~10 倍（实测猴林迷宫I：底图 78×203、面板
#: 753×612 → 光宽度就差 9.7 倍）。卡在 4 倍时，人手动也拖不到正确位置。
SLIDER_MIN = 200
SLIDER_MAX = 12000
SPIN_MAX = 12.0

#: 匹配分低于这个值时，「保存」会先问一句。0.55 = locate 内部那条 min_score
SCORE_WARN = 0.55

#: 邻域比现在**高出多少**才算"附近还有更贴的一档"（见 `_refresh_judge` ✓）。
#: 0.02 是"比采样噪声大、又不至于把 0.002 的抖动当指路"那个量级 ✓。
NEAR_MARGIN = 0.02

#: 存回文件时 offset 保留的小数位（0.001 面板像素 ≈ 0.01 世界像素，远小于读数误差）
OFF_DEC = 3


def off_store(v):
    """标定里的一个 offset 分量 → 存文件用的值。

    ⚠ **别再 `int()` 截断**（2026-09-27 改）：这张图上 1 个面板像素 ≈ 8 世界像素 ⇒
      截断是**系统性**偏最多半个像素 = 4 个世界像素。实测：双点标定解出来的 -10.507
      一进这扇窗再保存就变成 -10，读数永远差 4 个世界像素、**看着像"点没点准"** ✗
      （用户 2026-09-27 那个 (48,-222) 里就有这 4 个像素）。
      ⚠ crop 的 `offset` 是 **inset 整数**（见 `calib()`），那边照旧用 int ✓。
    """
    try:
        return round(float(v), OFF_DEC)
    except (TypeError, ValueError):
        return 0


def _with_canvas_alpha(t):
    """底图 + 它自己的 alpha ⇒ **BGRA**（底图没有透明区就原样 3 通道 ✓）。

    用户 2026-09-27："**小地图底图**、地形叠加图的背景应该透明吧？你自己加的黑色？" ——
    底图 PNG 本来有近一半像素是**透明**的（面板之外的圆角 ✓），以前读图时把 alpha 丢了 ✗
    ⇒ 透明区变成**纯黑** ✗（见 `core/mapdata.load` 的说明 ✓）。`canvas` 本身仍是 3 通道（全仓
    的匹配/拼接都按 BGR 写 ✓），alpha 单独放在 `Terrain.canvas_alpha` ✓，要用的人自己合 ✓。
    """
    import numpy as np

    c = getattr(t, "canvas", None)
    a = getattr(t, "canvas_alpha", None)
    if c is None or a is None:
        return c
    return np.dstack([c, a])


def np_to_pixmap(img):
    """numpy **BGR / BGRA** → QPixmap（小地图帧是 BGR ✓；底图**带 alpha** 时是 BGRA ✓）。

    ⚠ 4 通道那条必须走 `Format_ARGB32`（它在内存里就是 BGRA 序 ✓）：以前只有 3 通道这一路 ⇒
      带 alpha 的底图会被当成 BGR 逐像素错读（通道错位 ✗ + 透明区变黑 ✗）——这正是用户问的
      "小地图底图的背景为什么是黑的"的来源之一 ✓（见 `core/mapdata.load` 里读底图那段的说明 ✓）。
    """
    import cv2
    import numpy as np

    if getattr(img, "ndim", 0) == 3 and img.shape[2] == 4:
        bgra = np.ascontiguousarray(img)                # 原样喂 Qt（BGRA = Format_ARGB32 ✓）
        h, w = bgra.shape[:2]
        qimg = QImage(bgra.data, w, h, bgra.strides[0], QImage.Format_ARGB32)
        return QPixmap.fromImage(qimg.copy())
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class MinimapCalibDialog(QDialog):
    """量「面板 → 底图」的缩放/偏移（存 datasets/map/<id>.mapcalib.json）。

    ⭐ **2026-09-29 用户七条改造**（原话见各处注释）：
      ① 去掉"在图上拖半透明底图"那种改法 —— 位置只由「x/y 轴偏移」两条拖动条 +
         方向键定 ✓（拖动**看不准也测不出数**：人眼对不齐半个像素，而方向键/数字框
         是**可复现**的 ✓）；
      ② 两条偏移拖动条（模式不同、单位不同：fit 是**面板像素**、crop 是**底图像素** ✓）；
      ③ 方向键的单位 = **1 个面板像素**（≈ 0.2 个底图像素）—— 原来 1 个**底图像素**
         太粗（这张图 1 底图像素 ≈ 7~16 世界像素 ✗）；
      ④ 「锁定 xy 缩放」勾上后两轴一起改 ✓；
      ⑤ crop 的评分**按重叠区算**（`mm.align_score` 一处实现）—— 原来报的是
         "整张面板 vs 整张底图"，crop 下那个数没有意义 ✗；
      ⑥ 自动定位旁的「不改缩放」：只改位移、保留当前缩放 ✓。
    """

    def __init__(self, map_id, mode=None, host=None, port=None, client=None,
                 parent=None, src=None):
        super().__init__(parent)
        self.setWindowTitle("小地图标定")
        self.setMinimumSize(900, 720)
        #: 点过「保存标定」没有 —— 外层（路线识别）据此决定要不要刷新状态
        self.saved = False
        #: **这份标定是给哪条来源量的**。标定按来源分开存（见 mapdata.calib_path）：
        #: 两条来源的面板尺寸差 5.6 倍，一份几何只对一条成立。外层（路线识别）知道
        #: 自己喂的是哪条来源，会显式传进来；没传就按配置里的当前来源算。
        self.src = src or (load_live().get("mmap_src") or mm.SRC_STREAM)

        self.map_id = str(map_id)
        self.terrain = mapdata.load(self.map_id, with_canvas=True)
        self.canvas = None if self.terrain is None else self.terrain.canvas

        from tools.config import get
        self.host = host or get("a_host")
        self.port = int(port or get("minimap", "port", 5003))
        #: client 传进来就**不接管它的生命周期**（将来实时那路已经在跑时共用一条）；
        #: 没传就自己连一路，关窗时自己停 —— 不能让弹窗留下一个后台线程。
        self._own_client = client is None

        cal = mapdata.load_calib(self.map_id, self.src) or {}
        self.mode = mode or cal.get("mode") or mm.MODE_FIT
        #: 文件里那份几何是**另一种显示方式**量的吗（下面 `__init__` 末尾会说一句）
        self._mode_note = ""
        if mode and cal.get("mode") and str(cal.get("mode")) != str(mode):
            # fit / crop 两种写法的 `offset`/`view` **含义不一样**（crop 的 offset 是
            # inset、截距进 `view`，见 `calib_from_two_point`）⇒ 拿「局部」的几何在
            # 「全局」下摆，位置会整个不对 —— 表现和"没保存"一模一样 ✗（2026-09-27）
            self._mode_note = (
                "⚠ 文件里那份几何是**「%s」**方式量的，而这扇窗现在是**「%s」**"
                "（「显示方式」下拉）—— 两种写法的 offset/view 含义不同，直接拿来对齐会"
                "看着完全不对（像是没保存）✗。要么把下拉换回「%s」，要么按现在这种方式"
                "重量一次。" % (cal["mode"], mode, cal["mode"]))
        # ⚠ **两轴**（2026-09-27 修）：这扇窗原来只认**一个**缩放 —— 读 x 轴、
        #   叠加层也按等比画、`calib()` 连 `scale_y` 都不写 ⇒ 用「双点标定」量出来的
        #   两轴几何一进这扇窗（a）看着差一点、（b）**保存时 y 轴被抹成 x** ✗
        #   （用户 2026-09-27 报的"双点标定完点开标定窗发现不太对"就是这条）。
        #   现在两轴都读进来、按各自的缩放画、保存时原样写回 ✓。
        #   ⚠ 这扇窗的拖动/缩放只会**等比**改整体大小（`_ratio` 保留载入时的两轴比例）
        #     —— 要改两轴**比例**仍然只能用「双点标定」重量一次 ✓。
        self.scale = float(cal.get("scale") or 1.0)
        self.scale_y = float(mm.scales_of(cal)[1])
        self._two_axis_note = ""
        if abs(self.scale_y - self.scale) > 1e-9:
            self._two_axis_note = (
                "这份标定**两轴不同**（x %.4f / y %.4f）—— 下面「x 轴缩放」「y 轴缩放」"
                "两条分别对应它们，改哪条动哪条、保存照写 ✓"
                "（两轴本该几乎相等：差得多说明标定或采样有问题）。"
                % (self.scale, self.scale_y))
        self.offset = [off_store(v) for v in (cal.get("offset") or [0, 0])]
        # crop 专用：现在显示的是底图从**这一块**开始的内容。
        # ⚠ 不叫 self.view —— 那个名字留给画布控件（和 calib_manual 一致），
        # 两边同名过一次，结果控件被几何字段覆盖掉（AttributeError 一大串）。
        # ⚠ **存小数、不许取整**（2026-09-29 改，用户第 ③ 条）：方向键/拖动条的一步是
        #   **1 个面板像素** = `1/scale` 个底图像素（这张图 ≈0.18）⇒ 取整就把人调的东西
        #   直接吃掉 ✗（"调了 0.2、保存一看没变"）。全仓库规矩也是几何量不取整 ✓。
        self.block = [off_store(v) for v in (cal.get("view") or [0, 0])]
        self.score = float(cal.get("score") or 0.0)
        #: ⭐ **滚动方向**（用户 2026-09-29 追加："局部小地图也应该分滚动类型：仅X / 仅Y /
        #: 双轴，这样在对于单轴滚动的小地图，实时匹配时可以提高速度"✓）——
        #: 它是**几何的一部分**（存在标定文件里、按图各一份 ✓），默认「双轴」= 老行为 ✓。
        self.scroll = mm.scroll_of(cal)
        self.inset = 4                     # crop 的边框内缩（与 locate_crop 一致）
        # crop 下 `offset` **就是 inset**（模板从面板边缘往里缩了多少，
        # 与 locate_crop 的返回一致）。老文件里可能根本没有这个键（默认 [0,0]）
        # —— 那样存下去的东西会比人工对齐差 inset 个底图像素（在
        # px_per_world≈16 的图上是 63 世界单位，现场就是被这个坑到）。
        # 归一化放这里：从文件读来的这个值不可信，唯一可信的是 inset 本身。
        if self.mode == mm.MODE_CROP:
            self.offset = [self.inset, self.inset]
        #: 叠加层透明度（%）。对齐时要能同时看清「面板」和「底图」，全靠它。
        # ⚠ 别写 `cal.get("alpha") or 55`：**0 是 falsy** ⇒ 存着 0（完全看不见）也会
        #   显示成 55% ✗（2026-09-26 修，默认值现在只有 `overlay_alpha_pct` 一处 ✓）。
        self.alpha = mm.overlay_alpha_pct(cal)
        # 叠加参照：canvas（WZ 那张底图）/ terrain（我们自己渲染的地形叠加图）。
        # **为什么要有得选**：实测有些客户端的小地图是**按 foothold 现画**的，
        # 跟 WZ 里那张 `miniMap/canvas` 完全是两套画（白底绿棕 vs 黑底蓝地形），
        # 叠那张图人工也永远对不上；叠「平台形状一致」的地形图才对得上。
        self.ref = (cal.get("ref") or "canvas")
        self._ref_pix = None
        self._ref_zoom = 1                 # 参照图 1 像素 = 多少底图像素
        self._frame = None
        self._scene_wh = None
        #: 这一拍按**重叠区**算出来的贴合分（`mm.align_score` 的返回值；没画面时 None ✓）。
        #: 判据那行显示它、保存前的提醒也用它 —— `self.score` 是"载入/自动定位那次"的分，
        #: 在 crop 下说明不了"你对齐了没有" ✗（见 `_refresh_judge` ✓）。
        self._live_score = None
        # 有旧标定就别再自己摆 —— 判「**有没有几何**」而不是「有没有分数」：
        # 手工对齐的 score=0，拿分数判会把刚存好的几何覆盖成粗略猜测，
        # 这正是现场「点了保存、重开又变回去」的直接原因。
        self._geom_ready = mm.has_geometry(cal)
        #: 当前几何是**人眼对齐**的还是自动量出来的（存进标定的 src 字段）
        self.manual = (cal.get("src") == "manual")
        self._prev_anchor = None           # 上一拍的锚点（看位移用）
        self._last_locate = 0.0            # 上次定位时刻（自动模式节流用）
        #: 「保存标定」成功那一刻的几何指纹（None = 还没量过任何东西）。
        #: 关窗时拿它比：有差异就是"有没保存的改动" —— 见 reject()。
        self._saved_snap = None
        #: 载入时那份几何长什么样 + 它是**怎么来的**（`src` / 有没有 `score`）。
        #: 用来判"人动过没有"：**没动过就原样保留**，不然「打开看一眼、点一下保存」会把
        #: `src=two_point` 降级成 `manual`、还给一份没有匹配分的几何安上 `score: 0`，
        #: 顺带把双点标定量出来的 `scale_y` 抹掉 ✗（2026-09-27 修）。
        self._src_loaded = str(cal.get("src") or "")
        self._score_loaded = cal.get("score")
        self._geom_loaded = self._geom_now()

        self._build()
        self._load_ref()

        self._locate_cost = 0.0            # 上次定位耗时（自动模式据此退让）

        self.client = client or mm.MiniMapClient(self.host, self.port).start()
        # 标题行左边：这是哪张图、**画面从哪来**。来源不同时，"等帧该去哪等"
        # 完全不一样（收流等 A 机那一口；从实时画面框选要本机「实时」页在跑），
        # 不写出来人就只能猜 —— 这块以前是空着的。
        self.lbl_where.setText(
            "地图 %s　%s" % (self.map_id,
                             getattr(self.client, "label", "")
                             or "A 机小地图推流"))
        self._timer = QTimer(self)
        self._timer.setInterval(100)       # 取帧很便宜（只是读一个引用）
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self._sync_overlay()
        self._refresh_loaded()
        self._refresh_status()
        self._refresh_judge()
        # 两类"一打开就看着不对劲、容易被当成没保存"的情况，开头就说清（见上面两处 note）：
        #   ① 文件里的几何是**另一种显示方式**量的（fit ↔ crop 的 offset/view 含义不同）；
        #   ② 透明度是 **0%**（叠图完全透明 ⇒ 一片空白）。
        if self._mode_note:
            self._say(self._mode_note, bad=True)
        elif self.alpha <= 0:
            self._say("⚠ 这份标定的叠加层透明度是 **0%** ⇒ 打开时什么都看不见"
                      "（几何是**存着的**，不是没保存）—— 把「透明度」往右拖就能看见 ✓。",
                      bad=True)
        theme.bind_window_state(self, "minimap_calib")  # 拉过的大小/位置按客户端记住 ✓

    # ---------------- 界面 ----------------

    def _build(self):
        """版面（顺序 = 代码顺序，UI 规范 §4）：**标题 → 这份几何从哪来 → §画面 → 画布
        → §几何 → 判据 → 动作结果 → 按钮 → 提示**。

        **归纳的三条**（2026-09-29 用户要求"整理信息、归纳布局"）：
          · 按"**你在干什么**"分小节：画面（看什么）/ 叠加层（那层图什么样，跟着画面走 ✓）/
            几何（往哪摆）/ 判据动作（算得对不对、存哪）—— 小标题 + 细分割线（`_section` ✓）；
          · **一对 x/y 并排成一行**（缩放一行、偏移一行）：它是**一个**参数组的两个分量 ✓
            ⇒ 省下两行高度，全给画布（这扇窗最缺的就是高度 ✓）；
          · 参数名 / 单位统一宽度对齐（`_row_label` / `_unit_label` ✓），读数不再"参差"。
        说明文字一律**收进 tooltip**（正文只留一句提示 ✓ —— 本窗以前底部堆了三行长提示 ✗）。
        """
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)
        self.setMinimumWidth(940)           # 一行两组「拖动条+数字框」也不挤（见下 ✓）

        # ---- 标题行：这是哪张图 · 哪种显示方式 ----
        top = QHBoxLayout()
        top.setSpacing(10)
        self.lbl_where = QLabel()
        self.lbl_where.setStyleSheet("color:#202124; font-weight:600;")
        top.addWidget(self.lbl_where)
        self.lbl_mode = QLabel()
        self.lbl_mode.setStyleSheet("color:#5f6368;")
        self.lbl_mode.setToolTip(
            "显示方式在「路线识别 → 小地图定位」里选（要进游戏走两步看地形动不动\n"
            "才知道该用哪种），这里只是沿用你选的那个，不替你改。")
        top.addWidget(self.lbl_mode)
        top.addStretch(1)
        root.addLayout(top)

        # ---- 这份几何是**从哪来的**（单独一行，不往上面那行续 —— UI 规范 §4）----
        # 为什么非要写出来：不写的话，「还原了上次标定」和「程序刚摆了个大致位置」
        # 在屏幕上**长得一模一样**，于是人只能猜"我标过没有"。现场就是这么卡住的：
        # 标定压根没存上（文件都不存在），弹窗却每次都默默摆一个大概位置 ——
        # 看着像"还原了一份错的"，实际是"没有可还原的东西"。
        self.lbl_loaded = QLabel()
        self.lbl_loaded.setWordWrap(True)
        self.lbl_loaded.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_loaded)

        # ---- §画面：收帧状态 + 两个开关 + 叠加层那两样（参照 / 浓淡）----
        sec = self._section(
            root, "画面",
            tip="这一节管「你在看什么」：\n"
                "· 实时画面：跟着流刷新（不勾 = 停在当前这张，方便慢慢量 ✓）；\n"
                "· 持续自动定位：每来一帧重算一次（边走边看它最有用 ✓）；\n"
                "· 叠加参照 / 透明度：那层半透明图用哪张、多浓 —— 只影响你**看到**的，\n"
                "  不参与换算（自动定位只认小地图底图 ✓）。")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.lbl_conn = QLabel()
        self.lbl_conn.setWordWrap(True)
        row.addWidget(self.lbl_conn, 1)
        self.ck_live = QCheckBox("实时画面")
        self.ck_live.setChecked(True)
        self.ck_live.setToolTip("一直跟着流刷新画面。\n"
                                "不勾就停在「抓一帧」抓到的那张上（想专门量某一帧时用）")
        self.ck_live.toggled.connect(lambda _c: self._on_tick())
        row.addWidget(self.ck_live)
        self.ck_auto = QCheckBox("持续自动定位")
        self.ck_auto.setToolTip(
            "每来一帧就重新定位一次（相当于命令行的 --watch）。\n"
            "**边走边看**它最有价值：往右走 → 底图块也朝右平移，说明量对了。")
        row.addWidget(self.ck_auto)
        sec.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self._row_label("叠加参照"))
        self.cmb_ref = NoWheelComboBox()
        self.cmb_ref.addItem("小地图底图", "canvas")
        self.cmb_ref.addItem("地形叠加图", "terrain")
        self.cmb_ref.setToolTip(
            "画面里那层半透明图用哪一张跟面板对。\n\n"
            "· 小地图底图：WZ 里的 `miniMap/canvas`（游戏真用它当小地图时才叠得准）\n"
            "· 地形叠加图：我们自己按 foothold 画的那张（**平台形状**一致）——\n"
            "  有些客户端的小地图是按地形现画的，跟底图完全是两套画，那时候只能拿\n"
            "  这张来人工目测对齐（平台边缘对平台边缘）。\n\n"
            "（自动定位只认底图；这张只影响你眼睛看到的叠加层。）")
        self.cmb_ref.currentIndexChanged.connect(safe_slot(self._on_ref))
        row.addWidget(self.cmb_ref)
        row.addSpacing(10)
        row.addWidget(self._row_label("透明度", 60))
        self.sld_alpha = NoWheelSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(int(self.alpha))
        # 宽度给个**上下限**：太窄不好拖、拉满又会横穿半个窗口（这一行右边没别的东西，
        # 让它长到 700px 只是显得空 ✓）。
        self.sld_alpha.setMinimumWidth(150)
        self.sld_alpha.setMaximumWidth(260)
        self.sld_alpha.setToolTip(
            "面板上面那层「WZ 素材/地形图」的浓淡（%）。\n"
            "对齐时两头都要看得见：太浓只看见底图，太淡又看不见地形 ——\n"
            "数值跟着标定存，下次打开还是它。")
        self.sld_alpha.valueChanged.connect(safe_slot(self._on_alpha))
        row.addWidget(self.sld_alpha, 1)
        self.lbl_alpha = QLabel()
        # ⚠ 0% 时要写成「叠图看不见」（见 `_sync_alpha_label`）：字更长，宽度留够
        self.lbl_alpha.setMinimumWidth(96)
        self.lbl_alpha.setStyleSheet("color:#80868b;")
        row.addWidget(self.lbl_alpha)
        self._sync_alpha_label()
        sec.addLayout(row)

        # ---- 画布：面板帧（底）+ 半透明底图（位置只由下面两条偏移条 + 方向键决定 ✓）----
        self.view = ZoomPanView()
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setStyleSheet("background:#202124; border:1px solid #3c4043;")
        self.view.setFocusPolicy(Qt.NoFocus)   # 方向键留给本窗口做微调（见 keyPressEvent）
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self._bg_item = QGraphicsPixmapItem()
        # 整张叠加图那个 item 留着（隐藏）：按钮已经去掉 —— 现在看那张图走
        # 「叠加参照 = 地形叠加图」。留着是免得 _draw_overlay 变成会炸的死代码。
        self._ov_img_item = QGraphicsPixmapItem()
        self._ov_img_item.setVisible(False)
        self.scene.addItem(self._ov_img_item)
        # ⚠ **不给人拖**（用户 2026-09-29 第 ① 条："移除手动在图示区拖动修改的功能"）。
        #   拖动的问题不是难用，是"**量不出数**"：人眼对不齐半个像素，而且拖一下就
        #   把 scale/offset/view 全按鼠标位置**重算**一遍（原来 `_on_overlay_moved` 就是
        #   干这个）⇒ 想复现刚才那一下根本做不到 ✗。现在位置只由「x/y 轴偏移」两条
        #   拖动条 + 方向键决定 —— 都是**可复现的整数/小数** ✓。
        self._ov_item = QGraphicsPixmapItem()
        self._ov_item.setOpacity(max(0, min(100, self.alpha)) / 100.0)
        self.scene.addItem(self._bg_item)
        self.scene.addItem(self._ov_item)
        root.addWidget(self.view, 1)

        # ---- §几何：一对 x/y 并排一行（"缩放"是一个参数组 ✓）----
        sec = self._section(
            root, "几何（面板 ↔ 底图）",
            tip="这一节决定「面板里的像素对应底图哪儿」——\n"
                "换算 = 底图 = (面板 − 偏移) ÷ 缩放（局部小地图再加「显示区起点」）。\n\n"
                "拖动条给手感、数字框给准数；方向键挪 1 个面板像素。\n"
                "⚠ 图上不再能拖底图：拖量不出数（对不齐半个像素、还不可复现 ✗）。")
        # ⚠ **网格**（不是两行 `QHBoxLayout`）：列跨行共享 ⇒ "缩放 y" 与 "y 轴偏移"
        #   必定在同一个 x 上 ✓（见 `_num_cell` 的 ⚠⚠ —— 各写各的行会差 63 个像素 ✗）。
        #   ⭐ 一对 x/y **并排**成一行：它是**一个**参数组的两个分量 ✓（省下两行高度，
        #   全给上面那张画布 ✓）。
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        self.sld, self.sp_scale = self._num_cell(grid, 0, 0, "缩放 x", 0, "scale")
        self.sld_y, self.sp_scale_y = self._num_cell(grid, 0, 5, "缩放 y", 1, "scale")
        # ⭐ **「锁定 xy 缩放」**（用户 2026-09-29 第 ④ 条）：勾上后改任一轴 ⇒ 两轴一起变 ✓
        #   （游戏是把小地图**等比**缩放的，两轴本该相等；只有双点标定那种"两轴分别量"
        #    的场合才需要单独调 —— 那时把这个勾去掉 ✓）。
        self.ck_lock_xy = QCheckBox("锁定 xy")
        self.ck_lock_xy.setChecked(True)    # 默认勾上：等比是常态 ✓
        self.ck_lock_xy.setToolTip(
            "勾上：拖 x 轴或 y 轴任一条，两轴**一起**变（等比）—— 游戏把小地图等比缩放，\n"
            "平时就该这样 ✓。\n"
            "不勾：两轴各调各的（只有「双点标定」量出两轴不同时才需要）。")
        grid.addWidget(self.ck_lock_xy, 0, 10)
        # ⭐ **两条「轴偏移」并排**（用户 2026-09-29 第 ② 条）：
        #   ⚠ 单位**随模式变**（这是同一条命令在两套写法里的意思不同，必须写在单位那格 ✗）：
        #     · fit（全局）：**面板像素** —— 底图从面板的哪个像素开始摆（`offset`）；
        #     · crop（局部）：**底图像素** —— 现在显示的是底图从哪一块开始（`view`）。
        self.sld_ox, self.sp_ox = self._offset_cell(grid, 1, 0, 0)
        self.sld_oy, self.sp_oy = self._offset_cell(grid, 1, 5, 1)
        grid.setColumnStretch(11, 1)        # 尾巴留白都归最后一列（不撑坏两组 ✓）
        sec.addLayout(grid)

        # ⭐ **「滚动方向」**（用户 2026-09-29 追加 ✓）：这条面板会往哪个轴滚。
        #   ⚠ 它**不改换算**（`panel_to_canvas` 一字不变 ✓），只影响"实时怎么找"：
        #     · 单轴 ⇒ 只在**会动的那根轴**上搜 ⇒ 快好几倍 ✓（全图找收成一条带、
        #       拟合少 2/3 的探针 ✓）；
        #     · 而且**不会动的那根轴一个字都不改** ⇒ 面板在那一轴上的抖动不会变成
        #       显示区漂移 ✓（"跟出来的世界坐标在某一轴上慢慢跑"那类怪事就是它防的 ✓）。
        #   ⚠ 顺带：**第一次标这张图**时别指望它省事 —— 那时还不知道该定死哪根轴 ⇒
        #     两轴都会搜 ✓（量出来之后再选，之后每一次搜索都省 ✓）。
        row_sc = QHBoxLayout()
        row_sc.setSpacing(6)
        row_sc.addWidget(QLabel("滚动方向"))
        self.cmb_scroll = NoWheelComboBox()
        for _key, _txt in mm.SCROLL_LABEL:
            self.cmb_scroll.addItem(_txt, _key)
        _i = self.cmb_scroll.findData(self.scroll)
        self.cmb_scroll.setCurrentIndex(_i if _i >= 0 else 0)
        self.cmb_scroll.setToolTip(
            "这条小地图面板会往哪个方向滚（进游戏走两步就知道：地形动不动）。\n\n"
            "· 双轴：上下左右都会滚（默认，和以前一样 ✓）；\n"
            "· 仅 X / 仅 Y：只左右 / 只上下滚。\n\n"
            "它有两个作用：\n"
            "① 快：实时匹配只在会滚的那根轴上搜（全图找收成一条窄带、拟合少 2/3 的探针）\n"
            "   —— 单轴的小地图上快好几倍；\n"
            "② 稳：不会滚的那根轴一个字都不改（由标定定死）⇒ 面板在那根轴上的\n"
            "   抖动不会变成「显示区慢慢漂」⇒ 世界坐标不会在那一轴上悄悄跑偏。\n\n"
            "⚠ 选错了会怎样：把双轴标成仅 X ⇒ 那根轴的滚动跟不上（世界坐标会偏，\n"
            "状态行会报「显示区没跟住」）；把仅 X 标成双轴 ⇒ 只是慢一点，结果一样 ✓。")
        row_sc.addWidget(self.cmb_scroll)
        self.lbl_scroll = QLabel(
            "这一项只影响找得快不快、和「不动的那根轴不许跟歪」——换算一字不变 ✓")
        self.lbl_scroll.setStyleSheet("color:#5f6368;")
        self.lbl_scroll.setWordWrap(True)
        row_sc.addWidget(self.lbl_scroll, 1)
        self.cmb_scroll.currentIndexChanged.connect(
            safe_slot(self._on_scroll_changed))
        sec.addLayout(row_sc)

        # 两轴不同的标定要在这里说出来（见 `__init__` 里的 `_two_axis_note`）：
        # 人得知道"这份几何是两轴的、这扇窗会怎么对待它" —— 以前那行写的是"保存会把
        # y 轴改成 x"（现在两轴都保留了 ✓），留着是因为**两轴比例只能用双点标定改**。
        self.lbl_xy = QLabel(self._two_axis_note)
        self.lbl_xy.setStyleSheet("color:#3c4043;")
        self.lbl_xy.setWordWrap(True)
        self.lbl_xy.setVisible(bool(self._two_axis_note))
        sec.addWidget(self.lbl_xy)

        # ---- 判据（贴合分 / 世界坐标）+ 动作结果（定位成功、保存到哪）----
        # 两行分开：一行是**判据**（现在这套几何贴不贴），一行是**刚才那一下的结果** ——
        # 挤在一个 label 里会互相覆盖（定位结果刚写上就被下一拍冲掉，像没定位过 ✗）。
        self.lbl_judge = QLabel()
        self.lbl_judge.setWordWrap(True)
        self.lbl_judge.setStyleSheet("color:#3c4043;")
        root.addWidget(self.lbl_judge)

        self.lbl_say = QLabel()
        self.lbl_say.setWordWrap(True)
        self.lbl_say.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_say)

        # ---- 按钮 ----
        # 槽一律过 safe_slot：PyQt5 里槽函数抛未捕获异常会 qFatal() → 整个工作台
        # 无征兆闪退（见 gui/worker.py）。以前这里没包，出问题只会看到"点了没反应"。
        brow = QHBoxLayout()
        brow.setSpacing(6)
        self.btn_locate = QPushButton("自动定位")
        self.btn_locate.setToolTip(
            "按**当前这一帧**算一次「面板 ↔ 底图」的缩放/偏移（1 像素级）。\n"
            "和「持续自动定位」是同一条计算，只是一个算一次、一个跟着帧反复算。\n"
            "全局小地图点这一下就够了（倍数本来就能算出来，附近微调即可）。")
        self.btn_locate.clicked.connect(safe_slot(self._on_locate_clicked))
        brow.addWidget(self.btn_locate)
        # ⭐ **「不改缩放」**（用户 2026-09-29 第 ⑥ 条）：自动定位只改位移、保留现在的缩放 ✓。
        #   为什么需要：模板匹配给的缩放只能从**档位**里挑（`locate_crop` 是整数倍、
        #   `locate_fit` 是当前值附近 ±10% 的 5 个候选），而双点标定/人工量出来的缩放
        #   更准 ⇒ "只对齐位置、别动我的缩放"是一个**真实需求** ✓。
        self.ck_no_scale = QCheckBox("不改缩放")
        self.ck_no_scale.setToolTip(
            "勾上：点「自动定位」只把**位置**对齐过来，缩放保留现在这个值 ✓。\n"
            "（不勾 = 连缩放一起按匹配结果改，和以前一样。）")
        brow.addWidget(self.ck_no_scale)

        brow.addStretch(1)
        self.btn_save = QPushButton("保存标定")
        self.btn_save.clicked.connect(safe_slot(self._on_save))
        brow.addWidget(self.btn_save)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)
        brow.addWidget(btn_close)
        root.addLayout(brow)

        # 按钮配色走**全项目那一份**（`theme.unify_ok_cancel`）：「保存标定」= 主按钮
        # （蓝底白字 `#1a73e8` ✓ 和"确定"同一个语义），「关闭」= 次要按钮 ✓。
        # ⚠ 别在这儿自己写色值 ✗（docs/UI规范.md 的按钮那一条：新建带"确定/取消"的对话框
        #   都必须调它一次 —— 本窗是"保存/关闭"，语义相同 ✓）。
        theme.unify_ok_cancel(self.btn_save, btn_close)

        # ---- 提示：**一行**（细节全在 tooltip 里 —— 正文塞说明会挤掉画布 ✗）----
        hint = QLabel("方向键 = 挪 1 个**面板像素**（Shift=5，焦点在哪都一样）　"
                      "·　滚轮缩放 / 中键平移 / 双击适应　·　图上不能拖（拖量不出数 ✗）")
        hint.setToolTip(
            "怎么对齐：用上面「x/y 轴偏移」两条拖动条（或方向键）把那层半透明底图挪到\n"
            "和面板重合，然后「保存标定」✓。数字框可以直接输精确值。\n\n"
            "为什么不给在图上拖底图：拖**量不出数** —— 人眼对不齐半个像素，而且一下就把\n"
            "缩放/偏移全按鼠标位置重算一遍，想复现刚才那一下做不到 ✗。\n"
            "拖动条 / 数字框 / 方向键都是**可复现**的值（存进标定文件，下次打开还是它 ✓）。\n\n"
            "不勾「实时画面」＝画面停住，方便对着量 ✓。")
        hint.setStyleSheet("color:#80868b;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # ⭐ **统一把 `**` 从界面上剥掉**（见 `_plain`）：本文件的说明文字是按文档那种
        #    "**强调**"写法写的，而 Qt 的 `QLabel` / tooltip **不认 markdown** ⇒ 直接显示
        #    就是两个星号 ✗（2026-09-29 眼过版面时在提示行上逮到的）。
        #    走一遍**所有子控件**：文案（`QLabel` / 按钮、勾选框的 `text`）+ 所有 `toolTip`
        #    一起处理 ⇒ 以后新加的控件也自动盖到 ✓，不用每写一句话就想着去掉星号 ✓。
        #    ⚠ **只碰 QLabel / QAbstractButton 的 text**：`QAbstractSpinBox` 也有 `text()`，
        #      但它那个 `setText()` 是"**按字符串设值**" ⇒ 顺手一改就把数值动了 ✗。
        for _w in self.findChildren(QWidget):
            if _w.toolTip():
                _w.setToolTip(self._plain(_w.toolTip()))
        for _w in self.findChildren((QLabel, QAbstractButton)):
            if _w.text():
                _w.setText(self._plain(_w.text()))
        # （动态那几行——`lbl_loaded` / `lbl_say` / `lbl_conn` / `lbl_judge`——在各自的
        #   setter 里过 `_plain`：它们是在 `_build` **之后**才写内容的 ✓。）

        # ⭐ **方向键永远只干"挪底图"这一件事**（用户 2026-09-27 报："期望按方向键或 Shift+
        #   方向键可以**按像素移动底图**，但是现在是在**操作透明度**" ✗）：
        #   这扇窗里的拖动条 / 数字框 / 下拉**默认都会抢方向键** —— 拖动条：←→ 改值、
        #   Shift+←→ 按 pageStep 跳；数字框：↑↓ 增减；下拉：↑↓ 换项 ✗；焦点落在谁身上谁就
        #   吃掉它（实测落在「透明度」那条上 ⇒ 按方向键变成"在调透明度" ✗，可下面那行提示
        #   明明写着"方向键=挪 1 个底图像素" ✗）。
        #   ⇒ 给**所有子控件**装事件过滤器：方向键在**到达控件之前**就被本窗口拿走
        #     （见 `eventFilter` ✓）⇒ 不管焦点在哪，方向键都是"挪底图" ✓；
        #     而鼠标拖/点照旧 ✓、数字框里**用键盘输精确值**也照旧 ✓（只让出方向键 ✓）。
        for _w in self.findChildren(QWidget):
            _w.installEventFilter(self)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    # ---------------- 「这份几何从哪来」/「有没有没存的改动」 ----------------

    def _refresh_loaded(self):
        """顶部那行：现在的几何是**上次存的**，还是**程序刚摆的**。

        判据用 `mm.has_geometry`（不是 score）—— 手工对齐没有匹配分，拿分数判会
        把"已经标过"说成"还没标定"（这正是以前「点了保存、重开又变回去」的由来）。
        """
        p = mapdata.calib_path(self.map_id)
        # **只读这条来源的那一份**：两条来源的面板尺寸差 5.6 倍，串了就是错的几何
        cal = mapdata.load_calib(self.map_id, self.src) or {}
        label = mm.SRC_LABEL.get(self.src, self.src)
        if mm.has_geometry(cal):
            when = ""
            try:                        # 文件时间＝"上次标定是什么时候"最直接的证据
                when = time.strftime("　存于 %m-%d %H:%M",
                                     time.localtime(p.stat().st_mtime))
            except Exception:
                pass
            _src = str(cal.get("src") or "")
            if _src == "two_point":
                # 双点标定的几何是**两轴**的（而且没有匹配分）—— 只报一个数会让人以为
                # 这份是等比的、或者以为它是自动匹配出来的 ✗（2026-09-27 修）
                how = "双点标定（两轴 x %.4f / y %.4f）" % mm.scales_of(cal)
            elif _src == "manual":
                how = "手工对齐"
            else:
                how = "自动匹配 %.2f" % float(cal.get("score") or 0.0)
            # 老格式没记来源 → **不能**写成「来源：从实时画面」：那是猜的，而这份
            # 几何只对当时那条来源成立（说成当前来源就是让人放心用错的数）。
            shown = "未记·老格式" if cal.get("legacy") else label
            self.lbl_loaded.setText(self._plain("已载入上次标定（来源：%s）：%s%s"
                                                % (shown, how, when)))
            self.lbl_loaded.setStyleSheet("color:#188038;")
            tip = ("这份几何是从文件里读回来的，关掉再打开还是它：\n%s\n\n"
                   "**它只对「%s」这条来源成立** —— 换来源后面板尺寸就变了，"
                   "要按新来源重量一次。\n"
                   "改了之后要点「保存标定」才会覆盖它（关窗时也会提醒）。"
                   % (p, label))
            if cal.get("legacy"):
                tip += ("\n\n⚠ 这份文件是**老格式**（没记来源 —— 它是「按来源分开存」"
                        "之前量的）—— 如果你换过来源，这份几何多半对不上，"
                        "请重量一次再保存。")
            self.lbl_loaded.setToolTip(self._plain(tip))
        else:
            self.lbl_loaded.setText(self._plain(
                "这张图在「%s」下还没标定过：现在摆的是程序猜的粗略位置 —— "
                "调好重合后点「保存标定」，否则下次打开还是这样。" % label))
            self.lbl_loaded.setStyleSheet("color:#b06000;")
            self.lbl_loaded.setToolTip(self._plain(
                "还没有可用的几何：\n%s\n\n"
                "（文件不存在，或者里面只有「显示方式」—— 那不算标定过。）\n"
                "标定是**按来源分开存**的：这里只认「%s」那一份，标的是另一条"
                "来源也照样算没标。\n"
                "「保存标定」会写进上面这个文件。" % (p, label)))

    def _geom_now(self):
        """当前几何的指纹（两轴 / 偏移 / crop 的起始块）—— 判"人动过没有"用。"""
        return (self.mode,
                round(float(self.scale), 6), round(float(self.scale_y), 6),
                off_store(self.offset[0]), off_store(self.offset[1]),
                off_store(self.block[0]), off_store(self.block[1]))

    def _untouched(self):
        """几何还是载入时那一份吗（没动过 ⇒ 保存时**原样保留**来源与分数）。"""
        return self._geom_now() == self._geom_loaded

    def _snapshot(self):
        """几何指纹：只取决定换算的那几项（透明度、参照图这些不算改动）。

        ⚠ **`scale_y` 也算**（2026-09-27）：它现在真的决定 y 轴换算，漏了它的话
        "只改了两轴比例"会被当成"没改动"，关窗时就不提醒保存了 ✗。
        """
        c = self.calib()
        return (c.get("mode"),
                round(float(c.get("scale") or 0.0), 4),
                round(float(mm.scales_of(c)[1]), 4),
                tuple(off_store(v) for v in (c.get("offset") or [0, 0])),
                tuple(off_store(v) for v in (c.get("view") or [0, 0])))

    def _unsaved(self):
        """有没保存的改动吗（还没量过东西时一律 False：那时确实没什么可存）。"""
        return (self._saved_snap is not None
                and self._snapshot() != self._saved_snap)

    # ---------------- 收帧 ----------------

    def _on_tick(self):
        frame, _t = self.client.latest()
        # 「实时画面」勾着才自动换帧；不勾就把画面**停住**（帧还在流里滚，
        # 只是不覆盖你在看的那张）—— 想对着某一帧慢慢量就把它取消掉。
        if (self.ck_live.isChecked() and frame is not None
                and frame is not self._frame):
            self._take(frame)
        # 基线取在**第一拍摆完位之后**：程序自己那次粗略摆位（自动定位还开着时
        # 会再跟一次自动量）不算"你改过" —— 否则一开一关就弹"还没保存"。
        if self._saved_snap is None and self._frame is not None:
            self._saved_snap = self._snapshot()
        self._refresh_status()

    def _take(self, frame):
        """把这一帧拿来看/用来定位。"""
        self._frame = frame
        self._apply_frame(frame)
        if self.ck_auto.isChecked():
            self._locate_now(auto=True)

    def _apply_frame(self, frame):
        """新帧到了：换背景图；第一帧顺便把叠加层摆到一个大致对的位置。"""
        self._bg_item.setPixmap(np_to_pixmap(frame))
        h, w = frame.shape[:2]
        if not self._geom_ready:
            self._geom_ready = True
            # 起始位置：整张底图按「刚好嵌进面板」的比例居中摆上，人再微调。
            # 直接给 1.0 的话，底图比面板大时人一上来就看不见全貌（crop 常见）。
            cw, ch = self.canvas.shape[1], self.canvas.shape[0]
            fit = min(w / float(cw), h / float(ch))
            if self.mode == mm.MODE_CROP:
                self.scale = 1.0
                self.block = [(cw - w) // 2, (ch - h) // 2]
            else:
                self.scale = round(fit, 3)
                self.offset = [int((w - cw * self.scale) / 2),
                               int((h - ch * self.scale) / 2)]
            # 还没标定过时摆的是**等比**的粗略位置 ⇒ 两轴相同 ✓
            #（有标定时根本走不到这里：`_geom_ready` 开局就是 True ✓）
            self.scale_y = float(self.scale)
            self._sync_overlay()
        if self._scene_wh != (w, h):
            self._scene_wh = (w, h)
            self._update_scene_rect()
            self.view.fit()
            # 面板尺寸变了（A 机重新框过区域）：判据里的角点世界坐标要跟着重算，
            # 不然那行还是按旧面板尺寸算的 —— 看着像"标定飘了"。
            self._refresh_judge()

    def _update_scene_rect(self):
        """场景 = 面板帧 ∪ 叠加层（crop 下底图比面板大，不扩就说不了话）。"""
        if self._frame is None:
            return
        h, w = self._frame.shape[:2]
        r = self._ov_item.sceneBoundingRect()
        x0 = min(0.0, r.left())
        y0 = min(0.0, r.top())
        x1 = max(float(w), r.right())
        y1 = max(float(h), r.bottom())
        self.scene.setSceneRect(x0, y0, x1 - x0, y1 - y0)

    # ---------------- 几何：标定字典 ↔ 叠加层 ----------------

    def _on_scroll_changed(self, _i=None):
        """换「滚动方向」：只更新手上的值 + 把影响说清楚（不重算几何 ✓）。

        ⚠ **下次搜索就生效**（`calib()` 里带着它 ✓）：所以这里不用立刻重跑自动定位 ——
        人要是想立刻看效果，点一下「自动定位」即可 ✓。
        """
        self.scroll = self.cmb_scroll.currentData() or mm.SCROLL_XY
        if self.scroll == mm.SCROLL_XY:
            self.lbl_scroll.setText(
                "「双轴」：两轴都会搜（和以前一样 ✓）。图上只往一个方向滚的话，"
                "选「仅 X / 仅 Y」会快好几倍 ✓")
        else:
            _zh = "左右" if self.scroll == mm.SCROLL_X else "上下"
            self.lbl_scroll.setText(
                "「仅 %s」（只%s滚）：实时匹配只在那一根轴上搜 ✓ 更快，"
                "另一根轴一个字都不改（防漂 ✓）"
                % (self.scroll.upper(), _zh))

    def calib(self):
        """当前几何 → 标定 dict（和命令行工具存的是同一套字段）。

        ⚠ crop 的 `offset` **必须写成 inset**（不能用从文件里读来的那个数）：
        它是「模板从面板边缘往里缩了多少」，`panel_to_canvas` 拿它做换算 ——
        写成 0 等于说「没缩」，于是程序按文件算出来的世界坐标和人工对齐差
        inset 个底图像素（`t_save_then_reopen` 钉着这条）。

        ⚠ **`scale_y` 照写**（2026-09-27 改）：这扇窗以前只写一个 `scale` ⇒ 双点标定
        量出来的两轴几何一进一出就被抹成等比 ✗。两轴相同时 `with_scales` 不写这个键
        （老文件里就没有它 ✓），所以等比的老标定不会被塞一个新字段 ✓。
        ⚠ **人没动过 ⇒ 来源与分数原样保留**：不然"打开看一眼、点一下保存"会把
        `src=two_point` 降级成 `manual`、还给没有匹配分的几何安上 `score: 0` ✗。
        """
        # fit 的 offset 走 `off_store`（**不许取整**，见那儿）；crop 的 offset 是 inset 整数 ✓
        off = ([int(self.inset), int(self.inset)] if self.mode == mm.MODE_CROP
               else [off_store(self.offset[0]), off_store(self.offset[1])])
        out = {"mode": self.mode,
               "offset": off,
               # ⚠ `view` 也**不许取整**（2026-09-29）：方向键/拖动条的一步是 1 个面板像素
               #   = `1/scale` 个底图像素 ⇒ 取整等于"调了没反应" ✗（见 `self.block` 那段）。
               "view": [off_store(self.block[0]), off_store(self.block[1])],
               # 记下人工对齐时用的是哪张参照图 + 那层叠多浓：下次打开接着用
               "ref": self.ref,
               "alpha": int(self.alpha)}
        # ⭐ **滚动方向**（用户 2026-09-29 ✓）：只在**不是默认**（双轴）时写进文件 ——
        #   老文件里没有这个键 ⇒ 一律按双轴算 ✓（和 `with_scales` 两轴相同时不写一个道理：
        #   别给老标定塞它本来没有的字段 ✓）。
        if self.scroll != mm.SCROLL_XY:
            out["scroll"] = self.scroll
        if self._untouched():
            # 几何一个字没动 ⇒ 两个数**照抄载入的那份**（别再 `round(…, 4)` 把双点标定
            # 量出来的 1.880076 改成 1.8801："看一眼再保存"不该改数据 ✗）
            out["scale"] = float(self.scale)
            out = mm.with_scales(out, float(self.scale), float(self.scale_y))
            # 来源/分数照抄载入的那份：`two_point` 的几何**没有** `score` 这个键，
            # 别给它安一个 0（那会让人以为它是自动匹配出来的 ✗）
            out["src"] = self._src_loaded or ("manual" if self.manual else "auto")
            if self._score_loaded is not None:
                out["score"] = self._score_loaded
        else:
            out["scale"] = round(float(self.scale), 6)
            out = mm.with_scales(out, round(float(self.scale), 6),
                                 round(float(self.scale_y), 6))
            out["score"] = round(float(self.score), 6)
            # 这份几何是**量出来的**还是**人眼对齐的**：手工对齐没有匹配分，
            # 而「标过没有」不能拿 score 判（见 mm.has_geometry）。
            out["src"] = "manual" if self.manual else "auto"
        return out

    def _load_ref(self):
        """按 self.ref 选叠加参照图。返回是否用上了期望的那张。"""
        self._ref_pix, self._ref_zoom = None, 1
        if self.ref == "terrain" and self.canvas is not None:
            p = mapdata.map_dir() / ("%s_overlay.png" % self.map_id)
            pm = QPixmap(str(p)) if p.exists() else QPixmap()
            if not pm.isNull():
                self._ref_pix = pm
                # 叠加图是把底图放大 overlay_zoom 倍画的（见 tools/map_terrain_view）
                self._ref_zoom = mm.overlay_zoom(self.canvas.shape[1])
            else:
                self.ref = "canvas"        # 没有叠加图 → 退回底图（下拉也跟着回）
        if self._ref_pix is None and self.canvas is not None:
            self._ref_pix = np_to_pixmap(_with_canvas_alpha(self.terrain))
            self._ref_zoom = 1
        if self._ref_pix is not None:
            self._ov_item.setPixmap(self._ref_pix)
        return self.ref == self.cmb_ref.currentData()

    def _on_ref(self, _i=None):
        """换参照图：几何（scale/offset/view）不变，只是换一层图给人看。"""
        want = self.cmb_ref.currentData() or "canvas"
        self.ref = want
        self._load_ref()
        # 没叠加图时 _load_ref 会把 ref 退回 canvas —— 下拉跟着回，别显示不一致
        j = self.cmb_ref.findData(self.ref)
        if j >= 0 and j != self.cmb_ref.currentIndex():
            self.cmb_ref.blockSignals(True)
            self.cmb_ref.setCurrentIndex(j)
            self.cmb_ref.blockSignals(False)
        self._sync_overlay()
        if self.ref != want:
            self._say("没有 %s_overlay.png —— 先在「路线识别」里点「生成地形图」，"
                      "暂时按小地图底图叠着看" % self.map_id, bad=True)

    def _sync_overlay(self):
        """标定字段 → 叠加层的位置/缩放（fit 用两轴 scale+offset，crop 用 view）。

        `self.scale` / `self.scale_y` 一律是**面板像素 / 底图像素**（两轴）；参照图可能是
        放大过的（地形叠加图是底图的 overlay_zoom 倍），所以 item 的缩放要除以 `_ref_zoom`。
        """
        if self.canvas is None:
            return
        # ⚠ **两轴各自缩**（2026-09-27 改）：item 的变换走 `QTransform` —— `setScale`
        #   只有一个数，会把 y 轴也按 x 画，双点标定量出来的那点差别在叠加层上就
        #   **看不出来** ✗（而它正是 y 轴换算用的数）。
        kx = float(self.scale) / max(1, self._ref_zoom)
        ky = float(self.scale_y) / max(1, self._ref_zoom)
        self._ov_item.setTransform(QTransform().scale(kx, ky))
        if self.mode == mm.MODE_CROP:
            # crop = 面板显示底图的**一块**，而这一块可能被放大了 scale 倍
            # （小底图被客户端放大，实测能到 9~10 倍）。所以叠加层也跟着缩放：
            # 让「底图坐标 block」那一点正好落在面板的 (inset, inset) 上。
            self._ov_item.setPos(-self.block[0] * self.scale + self.inset,
                                 -self.block[1] * self.scale + self.inset)
        else:
            self._ov_item.setPos(self.offset[0], self.offset[1])
        self._update_scene_rect()
        self._sync_widgets()
        self._refresh_judge()

    def _pull_from_overlay(self):
        """从叠加层**现在的位置**反算回标定字段（方向键用它 ✓，鼠标拖动已去掉 ✗）。

        ⚠ 为什么留着这个方向（item → 字段）：方向键的一步是"挪**画面** 1 个像素"，
        而画面像素 ↔ 标定字段之间要按模式、按各轴缩放换算 —— 放这里一处实现 ✓。
        ⚠ **一律存小数**（`off_store`）：1 个面板像素 = `1/scale` 个底图像素（≈0.18）
        ⇒ 取整就把人调的东西吃掉了 ✗（用户第 ③ 条）。
        """
        self.manual = True            # 人调的 → 存进标定时记 src=manual
        pos = self._ov_item.pos()
        if self.mode == mm.MODE_CROP:
            s = max(1e-6, float(self.scale))
            self.block = [off_store((self.inset - pos.x()) / s),
                          off_store((self.inset - pos.y()) / s)]
        else:
            # 只改位置；缩放从 item 的变换里**逐轴**读回来（别拿一个数当两个 ✗）
            t = self._ov_item.transform()
            self.scale = float(t.m11()) * max(1, self._ref_zoom)
            self.scale_y = float(t.m22()) * max(1, self._ref_zoom)
            self.offset = [off_store(pos.x()), off_store(pos.y())]
        self._sync_widgets()
        self._refresh_judge()

    def _on_alpha(self, val):
        """透明度：只改叠加层的浓淡，几何一点不动（对齐时反复调的就是它）。"""
        self.alpha = int(val)
        self._sync_alpha_label()
        self._ov_item.setOpacity(max(0, min(100, self.alpha)) / 100.0)

    def _sync_alpha_label(self):
        """透明度那行的读数。**0% 必须写成"叠图看不见"**：

        0% 时叠图完全透明 ⇒ 一打开这扇窗就是"一片空白、像没标定过"✗（用户 2026-09-27
        报"保存后再打开发现没保存"，而他文件里正是 `alpha: 0` —— 几何存着、只是看不见）。
        0 是**合法的存档值**（设置那一侧专门修过"0 被 `or` 吃掉"的 bug，别在这儿禁掉它 ✗）
        ⇒ 能做的就是把话说清楚 ✓。
        """
        if self.alpha <= 0:
            self.lbl_alpha.setText(self._plain("0%（叠图看不见）"))
            self.lbl_alpha.setStyleSheet("color:#b06000;")
        else:
            self.lbl_alpha.setText(self._plain("%d%%" % self.alpha))
            self.lbl_alpha.setStyleSheet("color:#80868b;")

    def _sync_widgets(self):
        """把**两轴缩放 + 两轴偏移**回填到各自的拖动条/数字框（别让界面和几何不一致）。"""
        for sp, v in ((self.sp_scale, self.scale), (self.sp_scale_y, self.scale_y)):
            sp.blockSignals(True)
            sp.setValue(float(v))
            sp.blockSignals(False)
        for sld, v in ((self.sld, self.scale), (self.sld_y, self.scale_y)):
            sld.blockSignals(True)
            sld.setValue(max(SLIDER_MIN, min(SLIDER_MAX,
                                             int(round(float(v) * 1000)))))
            sld.blockSignals(False)
        # 两条「轴偏移」（值 = 模式对应的字段：fit 的 offset / crop 的 view ✓）
        for i, (sld, sp) in enumerate(((self.sld_ox, self.sp_ox),
                                       (self.sld_oy, self.sp_oy))):
            v = self._off_get(i)
            sp.blockSignals(True)
            sp.setValue(float(v))
            sp.blockSignals(False)
            sld.blockSignals(True)
            sld.setValue(int(max(OFF_MIN, min(OFF_MAX, round(v * OFF_STEP)))))
            sld.blockSignals(False)
        # 两种方式都要能调缩放：crop 也可能是「放大后取一块」（实测小底图会到
        # 9~10 倍）—— 以前这里把 crop 的缩放禁掉了，正好把唯一的手动出路堵死。
        for w in (self.sld, self.sp_scale, self.sld_y, self.sp_scale_y):
            w.setEnabled(True)

    @staticmethod
    def _plain(text):
        """把给**用户看**的文案里的 `**` 去掉 —— Qt 的 `QLabel` / tooltip **不认 markdown** ✗。

        为什么要有这一道（2026-09-29 眼过版面时逮到的）：本文件的说明性文字是按
        `docs/寻路设计.md` 那种"**强调**"写法写的，直接塞进 `QLabel` 就会在界面上
        原样显示两个星号 ✗（实测那句话是"方向键 = 挪 1 个**面板像素**" ⇒ 屏幕上真长这样 ✗）。
        与其一个个改字符串（改完以后新写的又会犯 ✗），不如在**显示口**统一剥掉这一层 ✓。
        `⚠` / `「」` / 中文标点照旧（它们在 Qt 里都能正常显示 ✓）。
        """
        return str(text or "").replace("**", "")

    def _row_label(self, text, width=ROW_LABEL_W):
        """一行最左边那个**参数名**标签（统一宽度 ⇒ 各行控件能对齐 ✓）。"""
        lab = QLabel(text)
        lab.setFixedWidth(int(width))
        lab.setToolTip(text)
        return lab

    def _unit_label(self, text, width=ROW_UNIT_W):
        """控件右边那个**单位/倍数**标签（灰字、固定宽度 ⇒ 数字框右边缘齐 ✓）。

        UI 规范 §9：**单位写在框外**（不许 `setSuffix` ✗）—— 写进框里数字会被挤窄、
        而且 `value()` 的文本会混进单位，读起来像"值的一部分" ✗。
        """
        lab = QLabel(text)
        lab.setStyleSheet("color:#80868b;")
        lab.setFixedWidth(int(width))
        return lab

    @staticmethod
    def _section(root, title, tip=""):
        """一个**小节**（灰字标题 + 一条细分割线）→ 返回里面的布局（往里 `addLayout` ✓）。

        为什么不 `QGroupBox`：工作台卡片那套样式（白底圆角 + 灰色小标题）是设在
        **主窗口自己身上**的（`main_window.setStyleSheet(QSS)` ⇒ **不继承给顶层弹窗** ✗），
        而把那份 QSS 从主窗口搬出来是另一件事（会动到主窗口观感）。这里用**同一套调色板**
        （标题 `#5f6368` / 分割线 `#e2e5ea`）自己画一条 —— 观感一致、又不牵扯主窗口 ✓
        （这扇窗本来就自带样式：画布那条 `background:#202124` 就是它自己设的 ✓）。

        **顺序 = 代码顺序**（UI 规范 §4）：小节按这里出现的先后从上到下排。
        """
        lab = QLabel(title)
        lab.setStyleSheet("color:#5f6368; font-weight:600;")
        if tip:
            lab.setToolTip(tip)
        root.addWidget(lab)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#e2e5ea; background:#e2e5ea; max-height:1px;")
        root.addWidget(line)
        lay = QVBoxLayout()
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(6)
        root.addLayout(lay)
        return lay

    def _num_cell(self, grid, r, c, label, axis, kind):
        """往网格里放**一个参数单元格**：`[参数名][拖动条][数字框][单位]`（占 4 列 ✓）。

        `kind` = `"scale"` / `"offset"`（两种的**范围 / 精度 / 单位 / 回调**不同 ✓，
        共同的那套摆放与提示只此一处 —— 以前是 `_scale_row` + `_offset_row` 两份 ✗）。

        ⚠⚠ **为什么用 `QGridLayout` 而不是"一行里塞两个 `QHBoxLayout`"**（2026-09-29 实测）：
        网格的**列是跨行共享的** ⇒ "缩放 y" 和 "y 轴偏移"必定落在**同一个 x** 上 ✓；
        而各写各的 `QHBoxLayout` 时，只要某一行多挂了一个控件（这里就是那个「锁定 xy」勾），
        两行的**右半组就会错开**（实测差了 **63 个像素** ✗）—— 竖着扫一眼就是"不像一张表"✗。
        """
        grid.addWidget(self._row_label(label), r, c)
        sld = NoWheelSlider(Qt.Horizontal)
        sp = NoWheelDoubleSpinBox()
        if kind == "scale":
            sld.setRange(SLIDER_MIN, SLIDER_MAX)
            sld.setToolTip(
                "这一轴的放大倍数（面板像素 / 底图像素），拖动 = 粗调。\n"
                "两轴本该**几乎相等**（游戏把小地图等比缩放）；差得多说明标定或采样有问题。\n"
                "拖动只改几何，不改透明度 —— 想让人看得清用上面那条「透明度」。")
            sld.valueChanged.connect(safe_slot(
                lambda v, a=axis: self._set_scale(a, v / 1000.0)))
            sp.setRange(0.05, SPIN_MAX)
            sp.setDecimals(3)
            sp.setSingleStep(0.01)
            sp.setToolTip("这一轴缩放的精确值（0.05~%.0f）—— 数字框给准数，拖动条给手感。"
                          % SPIN_MAX)
            sp.valueChanged.connect(safe_slot(lambda v, a=axis: self._set_scale(a, v)))
            unit = "×"
        else:
            sld.setRange(OFF_MIN, OFF_MAX)
            sld.setToolTip(
                "往哪边挪（单位见右边：全局小地图是面板像素、局部小地图是底图像素）。\n"
                "拖动 = 粗调，想要准数用右边那个数字框；方向键也能挪（1 个**面板像素**/下）。\n"
                "它是「这次显示的是哪一块」这件事的**唯一**输入 —— 图上不再能拖 ✗。")
            sld.valueChanged.connect(safe_slot(
                lambda v, a=axis: self._set_offset(a, v / float(OFF_STEP))))
            sp.setRange(OFF_MIN / float(OFF_STEP), OFF_MAX / float(OFF_STEP))
            sp.setDecimals(2)
            sp.setSingleStep(0.1)
            sp.setToolTip("这一轴偏移的精确值（0.01 一个格）—— 数字框给准数，拖动条给手感。")
            sp.valueChanged.connect(safe_slot(lambda v, a=axis: self._set_offset(a, v)))
            unit = self._off_unit()
        sld.setMinimumWidth(120)
        sp.setMinimumWidth(96)
        grid.addWidget(sld, r, c + 1)
        grid.addWidget(sp, r, c + 2)
        grid.addWidget(self._unit_label(unit, ROW_UNIT_W), r, c + 3)
        # 只有"拖动条"那一列吃剩余宽度 ⇒ 两个单元格组等宽、四条列对齐 ✓
        grid.setColumnStretch(c + 1, 1)
        return sld, sp

    def _set_scale(self, axis, v):
        """改**某一轴**的缩放（`axis` 0 = x、1 = y）。

        两轴默认**锁在一起**（「锁定 xy 缩放」勾着 ⇒ 改一条两轴一起变 ✓，游戏本来就
        等比缩放）；去掉勾才各调各的（只有「双点标定」量出两轴不同时才需要 ✓）。
        人改的 ⇒ 存进标定时记 `src=manual`（和拖动/偏移同一个口径：都不是程序量的 ✓）。
        """
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        lock = bool(getattr(self, "ck_lock_xy", None) is not None
                    and self.ck_lock_xy.isChecked())
        if axis == 0:
            self.scale = v
            if lock:
                self.scale_y = v
        else:
            self.scale_y = v
            if lock:
                self.scale = v
        self.manual = True
        self._sync_overlay()

    # ---------------- 「轴偏移」两条（单位随模式变） ----------------

    def _off_unit(self):
        """「轴偏移」的单位：fit = **面板像素**、crop = **底图像素**。

        ⚠ 必须显式写在标签上：同一个数在两种模式下差 `scale` 倍（≈5.6）✗ ——
        crop 的"偏移"是 `view`（现在显示的是底图哪一块），而 fit 的是 `offset`
        （底图从面板哪个像素开始摆）。
        """
        return "底图像素" if self.mode == mm.MODE_CROP else "面板像素"

    def _off_label(self, axis):
        """那一轴的偏移标签（**不含单位** —— 单位单独放控件右边那格 ✓，
        免得同一个单位在一行里出现两次）。保留这个口子：调用方要"带轴名的完整叫法"时用它 ✓。
        """
        return "%s 轴偏移" % ("x" if axis == 0 else "y")

    def _off_get(self, axis):
        """现在这一轴的偏移（**模式对应的那个字段**，float ✓）。"""
        v = (self.block if self.mode == mm.MODE_CROP else self.offset)[axis]
        return float(v)

    def _offset_cell(self, grid, r, c, axis):
        """「轴偏移」那一格（摆放共用 `_num_cell` ✓，单位随模式变 ✓）。"""
        return self._num_cell(grid, r, c, self._off_label(axis), axis, "offset")

    def _set_offset(self, axis, v):
        """改**某一轴**的偏移（模式对应的那个字段）—— 人改的 ⇒ `src=manual` ✓。"""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if self.mode == mm.MODE_CROP:
            self.block[axis] = off_store(v)
        else:
            self.offset[axis] = off_store(v)
        self.manual = True
        self._sync_overlay()

    # ---------------- 定位 ----------------

    def _locate_now(self, auto=False):
        if self.canvas is None:
            return
        if self._frame is None:
            self._say(getattr(self.client, "hint", None)
                      or "还没收到帧 —— A 机上的「小地图推流」启动了吗？",
                      bad=True)
            return
        now = time.perf_counter()
        if auto:
            # 节流：**自适应**。固定 250ms 不够 —— 大面板上一次定位可能要几秒，
            # 每帧都算会把工作台主线程（渲染/显示也在这个线程）拖住，
            # 观感就是「画面慢动作、延迟数字却变得好看」。这里要求定位最多占
            # 1/5 的时间：上次花了 3 秒，就 12 秒才算一次。
            gap = max(0.25, float(getattr(self, "_locate_cost", 0.0)) * 4.0)
            if (now - self._last_locate) < gap:
                return
        self._last_locate = now

        # **手动点和持续模式走完全同一条计算**（只是重复次数不同）：
        # 用户视角就该是这样 —— 「自动定位」= 按当前帧算一次，「持续」= 每秒多算几次。
        # 之前手动那次跑的是"完整搜索"（几十秒），结果人只会看到"点了没反应"。
        #
        # 尺度从哪来：全局小地图（fit）的倍数本来就能**算出来**——
        # 面板刚好装下整张底图，那就是 min(面板宽/底图宽, 面板高/底图高)；
        # 只需要在它附近微调（±10%，5 个候选）。crop 则由 locate_crop 自己
        # 搜放大倍数（每个候选都很便宜）。
        scales = self._near_scales()

        t0 = time.perf_counter()
        if scales is not None:
            loc = mm.locate_fit(self._frame, self.canvas, scales=scales)
        elif self.mode == mm.MODE_CROP:
            # ⭐ crop 两条路都**拿手上这套几何当种子**（用户 2026-09-29："我已经手工对得
            #   很好了"✓ —— 手工那份离答案最近，从它开始量又准又省 ✓）：
            #     · 「不改缩放」⇒ 只按**当前缩放**把位移抠好（`refine_crop(move_scale=False)`
            #       ✓）—— 以前这条路是拿"整数档搜索"的结果去取位移 ✗，缩放本来是对的、
            #       位移却是从错的倍数算出来的 ✗✗；
            #     · 否则 ⇒ 完整定一遍（位移 + 缩放 ✓）。
            cur = self.calib()
            loc = None
            if self.ck_no_scale.isChecked():
                got = mm.refine_crop(self._frame, self.canvas, cur, move_scale=False)
                if got is not None:
                    loc = dict(got["calib"], score=float(got["score"]))
            if loc is None:
                loc = mm.locate_crop(self._frame, self.canvas, seed=cur)
        else:
            loc = mm.locate(self._frame, self.canvas, self.mode)
        dt = time.perf_counter() - t0
        self._locate_cost = float(dt)      # 给自动模式的自适应节流用

        if loc is None:
            self._say_fail()
            return
        anchor = loc.get("view") if self.mode == mm.MODE_CROP else loc.get("offset")
        move = ""
        if auto and self._prev_anchor is not None and anchor is not None:
            move = "　位移 (%+d, %+d)" % (anchor[0] - self._prev_anchor[0],
                                        anchor[1] - self._prev_anchor[1])
        self._prev_anchor = list(anchor) if anchor is not None else None

        if self.ck_no_scale.isChecked():
            # ⭐ 「不改缩放」：**只把位移拿过来**（用户 2026-09-29 第 ⑥ 条 ✓）。
            #   做法：算"面板左上角那个点落在底图哪儿"——两套几何各算一次，都走
            #   `mm.panel_to_canvas` 这个**唯一口径** ✓ —— 再解出**我们这套缩放下的**偏移。
            #   fit：`底图 = (面板 − offset) / scale`  ⇒  offset = −底图(0,0) × scale
            #   crop：`底图 = (面板 − inset) / scale + view` ⇒ view = 底图(0,0) + inset/scale
            bx, by = mm.panel_to_canvas(0.0, 0.0, loc)
            sx = max(1e-6, float(self.scale))
            sy = max(1e-6, float(self.scale_y))
            if self.mode == mm.MODE_CROP:
                self.block = [off_store(bx + self.inset / sx),
                              off_store(by + self.inset / sy)]
            else:
                self.offset = [off_store(-bx * sx), off_store(-by * sy)]
            self.score = float(loc["score"])
            # 缩放是**保留的**（可能来自双点标定/人工）⇒ 记 `manual`：这份几何确实
            # 不是"这一次自动匹配量出来的" ✓（不然 `src=auto` 会让人以为缩放也是匹配的 ✗）。
            self.manual = True
        else:
            self.scale = float(loc["scale"])
            # 自动定位量出来的是**等比**几何（模板匹配只有一个倍数）⇒ y 轴跟着 x
            #（双点标定的两轴比例这样就被这次定位取代了 —— 这是对的：这份几何确实变成
            #  "匹配出来的"了，`src` 也会跟着变成 auto ✓；之后还能拿 y 那条拖动条单独调 ✓）
            self.scale_y = self.scale
            self.offset = [off_store(v) for v in loc["offset"]]
            self.block = [off_store(v) for v in (loc.get("view") or [0, 0])]
            self.score = float(loc["score"])
            self.manual = False       # 自动量出来的 → 存进标定时记 src=auto
        self._sync_overlay()
        # 顺带报一下这次定位花了多久：这一路以后要接进实时线程，耗时是硬指标
        #（B 机关键路径本来就没余量，见 docs/寻路设计.md §5 的性能一节）。
        self._say("定位成功%s　匹配分 %.3f　%.0f ms%s"
                  % ("（只改位移，缩放没动）" if self.ck_no_scale.isChecked() else "",
                     self.score, dt * 1000, move),
                  bad=(self.score < SCORE_WARN or dt > 0.25))

    def _near_scales(self):
        """fit 用的候选倍数：当前值附近 ±10%（5 个）—— 一发即中型。

        全局小地图的倍数本来就能算出来（面板刚好装下整张底图 = 两个方向比例的
        较小者），这里只是在它附近微调；**不做全范围搜索**：
        大面板上全搜索要几十秒，手动点一下根本等不到，看着就是"没反应"。
        crop 不用这个（locate_crop 自己搜放大倍数，且每个候选都很便宜）。
        """
        if self.mode != mm.MODE_FIT or not (self.scale > 0):
            return None
        out = [round(self.scale * f, 4) for f in (0.90, 0.95, 1.0, 1.05, 1.10)]
        return [s for s in out if 0.25 <= s <= 12.0] or None

    def _on_locate_clicked(self):
        """「自动定位」按钮：**同步算一次**，和持续模式同一条计算。

        先把提示写上再算：这一下通常几十毫秒（fit 只用当前倍数附近 5 个候选），
        但万一遇到大底图也要让人看到"它在算"，而不是"点了没反应"。
        """
        if self.canvas is None or self._frame is None:
            self._say("还没收到画面 —— A 机「被控机部署台 → 小地图推流」启动了吗？"
                      "（当前 %s）" % (getattr(self.client, "err", "") or "没连上"),
                      bad=True)
            return
        self._say("正在算…")
        QApplication.processEvents()       # 让上面这行先画出来
        self._locate_now()

    def _say_fail(self):
        """没对上：把两种方式的分数都摆出来当线索（和命令行工具一致）。

        ⚠ 别在这里做"完整搜索"：那是几十秒起步（见 _locate_now 的注释），
        失败时再卡一次，人只会以为按钮坏了。用当前尺度附近的候选给个参考分就够。
        """
        bits = []
        for tag, r in (("全局小地图", mm.locate_fit(
                            self._frame, self.canvas,
                            scales=self._near_scales())),
                       # ⚠ 参考分也**带着手上这套几何当种子**（crop ✓）：不带的那个
                       #   在"面板比底图宽"的图上常常直接归零，给不出任何线索 ✗
                       ("局部小地图", (mm.locate_crop(self._frame, self.canvas,
                                                     seed=self.calib())
                                       if self.mode == mm.MODE_CROP
                                       else mm.locate_crop(self._frame, self.canvas)))):
            bits.append("%s%s" % (tag, "没对上" if r is None else "%.3f" % r["score"]))
        extra = ""
        h, w = self._frame.shape[:2]
        cw, ch = self.canvas.shape[1], self.canvas.shape[0]
        if self.mode == mm.MODE_CROP and (w > cw or h > ch):
            extra = ("\n底图只有 %dx%d、面板 %dx%d 更大 —— 面板装不下整张底图，"
                     "所以不可能是「局部小地图」，多半是「全局小地图」"
                     "（去「路线识别」里换，这里不替你改）。" % (cw, ch, w, h))
        self._say("没对上 —— 可能：画面来源那块框得不对（A 机推流区域 / 本机框选区域"
                  "不只是小地图）/ 画面不是这张图 / 面板被游戏 UI 挡住。"
                  "参考分：%s%s" % ("、".join(bits), extra),
                  bad=True)

    def _say(self, text, bad=False):
        """动作结果（定位/保存/抓帧）。**只写这一行**，不和连接状态抢地方。"""
        self.lbl_say.setText(self._plain(text))
        self.lbl_say.setStyleSheet("color:%s;" % ("#c5221f" if bad else "#188038"))

    # ---------------- 状态 / 判据 ----------------

    def _refresh_status(self):
        if self._frame is None:
            err = getattr(self.client, "err", "") or ""
            self.lbl_conn.setStyleSheet("color:#c5221f;")
            # 等帧的说法随来源变：收流等的是 A 机那一口，从实时画面框选等的是
            # 本机「实时」页 —— 说错会让人跑去 A 机上瞎找（那边可能压根没开）。
            base = getattr(self.client, "wait_hint", None) or (
                "等帧中…（连 %s:%d）" % (self.host, self.port))
            self.lbl_conn.setText(self._plain("%s　%s" % (base, err) if err else base))
        elif not self.ck_live.isChecked():
            self.lbl_conn.setStyleSheet("color:#5f6368;")
            self.lbl_conn.setText(self._plain("画面已停住（「实时画面」没勾）"))
        else:
            # 只报**速率**，不报累计帧数：那是个只会一直变大的数字，看它没意义
            self.lbl_conn.setStyleSheet("color:#188038;")
            self.lbl_conn.setText("收帧中 · %.1f 拍/秒"
                                  % (getattr(self.client, "fps", 0.0) or 0.0))

    def _refresh_judge(self):
        if self.canvas is None or self._frame is None or self.terrain is None:
            self.lbl_judge.setText("")
            return
        h, w = self._frame.shape[:2]
        c = self.calib()
        try:
            lx, ly = mm.panel_to_world(0, 0, c, self.terrain)
            rx, ry = mm.panel_to_world(w, h, c, self.terrain)
        except Exception as e:
            self.lbl_judge.setText("换算失败：%s" % e)
            return
        b = self.terrain.bounds
        if self.mode == mm.MODE_CROP:
            # 局部小地图只显示底图的**一块**：判据是「这一块落在世界范围内」，
            # 而不是「四角贴住世界范围的四角」—— 后者在 crop 下永远不成立，
            # 于是标得再准那行也是橙的，人就开始怀疑自己标错了。
            pad = 16.0
            ok = (b[0] - pad <= lx and rx <= b[2] + pad
                  and b[1] - pad <= ly and ry <= b[3] + pad)
        else:
            ok = (abs(lx - b[0]) < 80 and abs(ly - b[1]) < 80
                  and abs(rx - b[2]) < 80 and abs(ry - b[3]) < 80)
        # ⭐ **贴合分：按"面板与该块底图的重叠区"算**（用户 2026-09-29 第 ⑤ 条 ✓）——
        #   载入/自动定位那个 `self.score` 在 crop 下是"整张面板 vs 整张底图"，跟你手上
        #   这套几何对不对**没关系** ✗（手工对齐的干脆没有分）。`mm.align_score` 一处实现 ✓
        #   （它自己算重叠范围，不需要人去框一块 ✓）。
        # ⭐⭐ `near=True` **必须开**（用户 2026-09-29："我已经手工对的很好了匹配分还是很低！"✗）：
        #   **绝对分不是通用的尺** —— 有的图客户端小地图的画法与底图不同源，对得再准也只有
        #   0.3~0.5 ✗；光看"0.42 低于 0.55"会让人以为没对齐，于是把**对的**几何改坏 ✗✗。
        #   低不低只能和**它自己**比：附近一圈里有没有明显更贴的一档 ✓（有 ⇒ 该往哪挪 ✓）
        sc = mm.align_score(self._frame, self.terrain, c, near=True)
        self._live_score = sc
        nb = sc.get("near") if sc.get("ok") else None
        better = bool(nb is not None
                      and float(nb["score"]) > float(sc["score"]) + NEAR_MARGIN)
        if sc.get("ok"):
            head = "贴合分 %.3f（重叠 %d%%）" % (sc["score"], round(sc["overlap"] * 100))
            if better:
                head += "　附近 (%+g, %+g) 更贴 %.3f ⇒ 往那边挪" % (
                    nb["dx"], nb["dy"], nb["score"])
            else:
                head += "　已在这一档最好"
            good = sc["score"] >= mm.TRUST_SCORE
        else:
            head = "贴合分：算不出来（%s）" % (sc.get("why") or "说不清")
            good = False
        # 行文收一收（"世界"说一次就够）：这行要**一眼扫过**，重复的词只是占宽度 ✓
        self.lbl_judge.setText(
            "%s　世界：左上 (%.0f, %.0f)　右下 (%.0f, %.0f)　范围 (%.0f, %.0f) ~ (%.0f, %.0f)"
            % (head, lx, ly, rx, ry, b[0], b[1], b[2], b[3]))
        self.lbl_judge.setToolTip(self._plain(
            "「贴合分」= 按**现在这套几何**，面板里那块和底图**重叠的部分**像不像"
            "（归一化相关，1.0 = 完全重合）。\n"
            "和「自动定位」那个匹配分不是一回事：那个是「整张面板最像底图的哪儿」，"
            "局部小地图下面板只是底图一小块，那个数没有意义 ✓。\n"
            "「重叠 %」= 面板里落在底图上的像素占比 —— **这个数小说明框多了**"
            "（把游戏 UI / 面板外那圈框进 crop 区域了）⇒ 分再高也别全信。\n\n"
            "⚠ **这个分不能跨图比、也不能只和 0.55 那条线比**：有的图客户端小地图的"
            "画法和底图**不同源**（自己另画一套底色/线条），对得再准也只有 0.3~0.5。\n"
            "所以后面那句更要紧：\n"
            "  · 「**已在这一档最好**」= 附近挪几像素都没有更贴的 ⇒ 分低是**这张图天生**"
            "如此，不是你没对齐 ✓（再用眼睛确认一眼半透明底图和面板处处重合即可）；\n"
            "  · 「**附近 (x, y) 更贴**」= 有得可挪 ⇒ 照它说的方向挪（或用方向键 / 拖动条），"
            "那才是真的没对齐 ✓。"))
        # 颜色三态：分够高 = 绿 ✓；分低但**已是这一档最好** = 灰（别再吓人 ✗）；
        #          附近还有更贴的 = 橙（有得可挪 ✓）
        if ok and good:
            color = "#188038"
        elif sc.get("ok") and not better:
            color = "#5f6368"
        else:
            color = "#b06000"
        self.lbl_judge.setStyleSheet("color:%s;" % color)

    # ---------------- 动作 ----------------

    def _draw_overlay(self):
        """「打开 / 关闭叠加图」：把那**一整张**地形图就地切到画面里看。

        要看到的东西就是 `<id>_overlay.png` 本身 —— **地形画在 WZ 素材小地图上**
        （`tools/map_terrain_view.py` 的 `use_canvas=True` 默认就是这么画的），
        再白框标出「当前小地图是它的哪一块」。

        就地切换（而不是弹新窗口）：按钮文字跟着变成「关闭叠加图」，
        再点一下回到"面板 + 半透明底图"的标定视图 —— 这样调标定和看结果
        不用来回找窗口。
        """
        if self._ov_img_item.isVisible():
            self._show_overlay_view(False)
            return
        if self.canvas is None:
            self._say("没有底图 —— 先在「路线识别」里点「生成地形图」", bad=True)
            return
        # 这一整段包起来：safe_slot 只把异常打到看不见的 stderr，表现就是
        # "点了没反应"。宁可把原因写在状态行上。
        try:
            p = None
            if self._frame is None:
                # 没画面也要能看那张图（只是没有白框）：它就是「地形画在 WZ 素材上」
                src = mapdata.map_dir() / ("%s_overlay.png" % self.map_id)
                if src.exists():
                    p = str(src)
                else:
                    self._say("还没有 %s_overlay.png —— 先在「路线识别」里点"
                              "「生成地形图」；另外现在也没收到画面（%s）"
                              % (self.map_id,
                                 getattr(self.client, "hint", None)
                                 or "A 机的「小地图推流」启动了吗？"), bad=True)
                    return
            else:
                h, w = self._frame.shape[:2]
                cw, ch = self.canvas.shape[1], self.canvas.shape[0]
                c = self.calib()
                if self.mode == mm.MODE_CROP:
                    _rc, rect_o = mm.view_rects(c, (w, h), (cw, ch))
                else:
                    z = mm.overlay_zoom(cw)
                    ox, oy = c["offset"]
                    _sx, _sy = mm.scales_of(c)      # 两轴各自（这份 dict 来自 calib() 是等比，
                    rect_o = (ox * z, oy * z,        # 但别在这儿写死一个 scale：那是"两处
                              (ox + cw * _sx) * z,   # 各判一次"的老毛病，迟早分叉 ✗）
                              (oy + ch * _sy) * z)
                p = mm.draw_on_overlay(self.map_id, rect_o)
                if p is None:
                    self._say("还没有 %s_overlay.png —— 先在「路线识别」里点"
                              "「生成地形图」" % self.map_id, bad=True)
                    return
            pm = QPixmap(str(p))
            if pm.isNull():
                self._say("图读不出来：%s" % p, bad=True)
                return
        except Exception as e:
            self._say("叠加图出错：%s: %s" % (type(e).__name__, e), bad=True)
            return
        self._ov_img_item.setPixmap(pm)
        self._show_overlay_view(True)
        self._say("叠加图 = 地形画在 WZ 小地图素材上；白框 = 当前那一块（%s）" % p)

    def _show_overlay_view(self, on):
        """切换画面：True = 只看那张叠加图；False = 回到标定视图。"""
        self._ov_img_item.setVisible(bool(on))
        self._bg_item.setVisible(not on)
        self._ov_item.setVisible(not on)
        self.btn_overlay.setText(self._plain("关闭叠加图" if on else "打开叠加图"))
        if on:
            self.scene.setSceneRect(self._ov_img_item.boundingRect())
        else:
            self._scene_wh = None          # 让下一帧重新按面板尺寸设场景
        self.view.fit()

    def _on_save(self):
        """存下当前几何。**每条早退都要说话**（见下面两处 `_say`）。

        以前有两条"静默返回"：没底图直接 return、提示里选了「否」也直接 return
        —— 人都以为自己存上了（这正是「我点了保存，结果没保存」的由来）。
        """
        if self.canvas is None:
            self._say("没有底图，存不了 —— 先在「路线识别」里点「生成地形图」"
                      "（没有底图就没法换算世界坐标）", bad=True)
            return
        # ⚠ 双点标定量的几何**没有匹配分**，可它是"量出来的"、不是目测的 ⇒ 别拿
        #   「匹配分偏低」去吓人：那段话的建议是"先点自动定位"，一点就把量好的两轴几何
        #   换成等比匹配的结果 ✗（2026-09-27）。只有**人眼对齐 / 自动匹配**那两份才提醒 ✓。
        # ⭐ 判据用**按重叠区算的贴合分**（用户 2026-09-29 第 ⑤ 条 ✓）：`self.score` 在
        #   crop 下是"整张面板 vs 整张底图"，跟你手上这套几何**对不对没关系** ✗
        #   （手工对齐更干脆：它压根没有分 ⇒ 以前每次保存都弹窗 ✗）。
        _sc = self._live_score or {}
        live = _sc.get("score")
        show = float(self.score) if live is None else float(live)
        # ⭐ 只是"分低"还不够**下结论**（用户 2026-09-29："我已经手工对的很好了匹配分还是
        #   很低！"✗）：还得看**附近有没有更贴的** —— 没有 ⇒ 这个低分是"这张图的画法与底图
        #   不同源"造成的，不是你标歪了 ✓（那时提醒的**说法**要跟着变，别一直喊"对不上" ✗）。
        _nb = _sc.get("near") if _sc.get("ok") else None
        _better = bool(_nb is not None
                       and float(_nb["score"]) > float(live or 0.0) + NEAR_MARGIN)
        if show < SCORE_WARN and not (self._untouched()
                                      and self._src_loaded == "two_point"):
            if live is None:
                title = "手工对齐"
                text = ("这份几何是**你用眼睛对齐**的（这一拍没算出贴合分）。\n\n"
                        "存下去之后，寻路完全按你的对齐算世界坐标 —— 人在这块平台、\n"
                        "程序以为在另一块，是这类标定最典型的错法。\n\n"
                        "再确认一眼：半透明底图和下面的面板**处处重合**吗？")
            elif _better:
                title = "贴合分偏低"
                text = ("当前贴合分 %.3f（低于 %.2f）—— 它按**面板与底图重叠的那一块**"
                        "算的（重叠 %d%%，拟合局部小地图就该这么算）。\n\n"
                        "而且**附近有更贴的一档**：把 view 挪 (%+g, %+g) 能到 %.3f\n"
                        "⇒ 这一套确实是歪的，先挪过去（方向键一步 = 1 个面板像素，\n"
                        "或者直接点「自动定位」—— 想保住现在的缩放就勾上「不改缩放」）。"
                        % (float(live), SCORE_WARN,
                           round(float(_sc.get("overlap") or 0.0) * 100),
                           _nb["dx"], _nb["dy"], float(_nb["score"])))
            else:
                title = "贴合分偏低（但已经贴到头了）"
                text = ("当前贴合分 %.3f（低于 %.2f，重叠 %d%%）**但它已经是这一档最好的**"
                        "—— 附近挪几个像素都没有更贴的。\n\n"
                        "⚠ 这个分不能跨图比：有的图客户端小地图的画法和底图**不同源**"
                        "（自己另画一套底色/线条），对得再准也只有 0.3~0.5。\n"
                        "所以要你自己拍板：半透明底图和下面的面板**处处重合**吗？\n\n"
                        "（重合 ⇒ 放心存 ✓；不重合 ⇒ 那是「这段地形本来就长得很像」，"
                        "别拿分当依据，用方向键/拖动条对着地标抠。）"
                        % (float(live), SCORE_WARN,
                           round(float(_sc.get("overlap") or 0.0) * 100)))
            # 默认给「是」：人已经点过一次「保存标定」，那次点击就是他的意图，
            # 这里只是提醒 —— 不该让他再猜一次默认值（以前默认「否」，手一抖
            # 就静默什么都没发生）。
            r = QMessageBox.question(self, title, text + "\n\n还是要保存吗？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes)
            if r != QMessageBox.Yes:
                self._say("没有保存（你在提示里选了「否」—— 几何还在，可以再点"
                          "「保存标定」）", bad=True)
                return
        # ⚠ 先把遗留的「坐标系偏移」清掉（`mm.drop_stale_world_offset`）：锚点换成下沿之后
        #   它该是 (0, 0)，而老的 (7, 33) 是**重心口径**的 —— 不清就会被原样带进新几何，
        #   读数凭空偏 33 个世界像素（≈4 个面板像素，看着像标定没量准 ✗）。
        cal = mm.drop_stale_world_offset(mapdata.load_calib(self.map_id, self.src))
        # ⭐ **先把旧的几何键清掉再 update**（2026-09-28 修，用户报："标定改完没生效 / 两种
        #   标定法数据不一样"）：`self.calib()` 在两轴相同时**不写 `scale_y`**（`with_scales`
        #   抹掉 ✓），而 `dict.update` **不删旧键** ⇒ 旧的双点标定 `scale_y` 会**残留**下来
        #   ⇒ 世界坐标 y 轴按**错的缩放**算（实测 106010105：自动定位后 scale=1.874 等比，
        #   却残留 scale_y=1.940285539…，两轴差 3.5%，正是用户看到的"两种标定法数据不一样" ✗）。
        #   同理 `src`/`score` 也该被这次的几何口径整体接管，别让旧值糊着 ✓。
        for _k in ("scale", "scale_y", "offset", "view", "src", "score"):
            cal.pop(_k, None)
        # 只更新量出来的几何 + 方式：显示方式仍算「你在界面里选的」
        cal.update(self.calib())
        cal["picked_by"] = "gui"
        cal.pop("legacy", None)         # 老格式的标记不写回文件（见 mapdata.save_calib）
        # 写文件这步**必须自己兜住异常**：这个槽外面还包着 safe_slot，而它只
        # `traceback.print_exc()` —— 正常启动走的是 pythonw（没有控制台），
        # 于是写失败会**一点痕迹都没有**：人点了保存、界面什么都没说，以为存上了。
        p = mapdata.calib_path(self.map_id)
        try:
            # **带上来源**：不带就等于整份覆盖 —— 会把另一条来源的标定抹掉，
            # 而人完全看不出来（然后在那条来源下算出错的世界坐标）。
            mapdata.save_calib(self.map_id, cal, self.src)
        except Exception as e:
            self._say("**保存失败**：%s: %s　（目标文件 %s —— 检查目录是否可写）"
                      % (type(e).__name__, e, p), bad=True)
            return
        if not p.exists():
            # 写没报错、文件却不在（同步盘/权限怪问题）：也要说出来，别让人以为存上了
            self._say("**保存失败**：写完却找不到文件 %s" % p, bad=True)
            return
        self.saved = True
        self._saved_snap = self._snapshot()     # 这一份就是"已保存"的基线
        self._refresh_loaded()                  # 顶部那行立刻变成「已载入上次标定…」
        self._say("已保存标定（%s）→ %s"
                  % ("手工对齐" if self.manual else "自动匹配", p))

    # ---------------- 键盘微调 ----------------

    def eventFilter(self, obj, ev):
        """**方向键在到达控件之前就被本窗口拿走**（装过滤器的原因见 `__init__` 里那段 ⚠ ✓）。

        为什么必须拦在**控件之前**：拖动条 / 数字框 / 下拉**自己就吃方向键**（改值 / 增减 /
        换项 ✗）—— 光"别给它们焦点"治不住：人一点它们（比如要用鼠标调透明度）焦点就过去了 ✗
        （用户 2026-09-27 报的"按方向键在操作透明度"正是这一种 ✗）。
        """
        if (ev.type() == QEvent.KeyPress
                and ev.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down)):
            self.keyPressEvent(ev)          # 挪叠加层（1 个**面板像素** / Shift 5 个 ✓）
            return True                     # **吞掉**：别再往下发给那个控件 ✗
        return super().eventFilter(obj, ev)

    def keyPressEvent(self, e):
        """方向键挪**叠加层**：一格 = **1 个面板像素**（Shift = 5 个）。

        ⚠ **为什么不按"1 个底图像素"**（用户 2026-09-29 第 ③ 条："方向键的单位不能是
        底图像素，要尽量小"✓）：叠加层画在**面板像素**这个坐标系里（`setPos` 就是它），
        而 1 个底图像素在放大后的面板上是 `scale` 个像素（这张图 ≈5.6 个）⇒ 想挪半个
        底图像素根本做不到 ✗。改成挪 1 个**面板像素**：一步 = `1/scale` 个底图像素
        （≈0.18），换算成世界坐标是 1~3 个 —— **小于读数噪声**（±4.3 ✓），
        也就是"能稳定调出来的最小一步" ✓（再小就只剩显示抖动，没有意义）。
        方向语义照旧：按左 = 叠加层往左走（于是「看到的是底图哪一块」右移）。
        """
        if self.canvas is None or e.key() not in (Qt.Key_Left, Qt.Key_Right,
                                                  Qt.Key_Up, Qt.Key_Down):
            super().keyPressEvent(e)
            return
        step = 5.0 if (e.modifiers() & Qt.ShiftModifier) else 1.0
        d = {Qt.Key_Left: (-step, 0.0), Qt.Key_Right: (step, 0.0),
             Qt.Key_Up: (0.0, -step), Qt.Key_Down: (0.0, step)}[e.key()]
        pos = self._ov_item.pos()
        self._ov_item.setPos(pos.x() + d[0], pos.y() + d[1])
        # ⚠ **必须显式叫**：叠加层已经不能拖了（没有 itemChange 回调 ✓），setPos 本身
        #   什么都不通知 ⇒ 忘了这句就是"按方向键画面动了、存下来的数没变" ✗。
        self._pull_from_overlay()
        e.accept()

    # ---------------- 收尾 ----------------

    def closeEvent(self, e):
        self._shutdown()
        super().closeEvent(e)

    def reject(self):
        """关闭（按钮 / Esc / 右上角）前：有没保存的改动就问一句。

        **为什么要在出口上拦**：「调了半天、结果没存」是这个弹窗最容易发生、
        而且**事后无法自证**的事故 —— 数据没了，屏幕上却和"存过了"长得一样。
        （判据是几何指纹，不是"点没点过保存"。）
        """
        if self._unsaved():
            r = QMessageBox.question(
                self, "还有没保存的改动",
                "现在的几何和上次保存的不一样，关掉就没了。\n\n"
                "「是」= 先保存再关（匹配分偏低时会再确认一次）\n"
                "「否」= 直接关掉，放弃这次的改动",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if r == QMessageBox.Yes:
                self._on_save()
                if self._unsaved():
                    # 没存成（比如在分数提示里选了「否」，或写文件失败）→ 别关，
                    # 状态行已经写明原因了，人看得见。
                    return
        super().reject()

    def done(self, code):
        self._shutdown()
        super().done(code)

    def _shutdown(self):
        """关窗必须停掉自己那路收流 —— 不然弹窗关了线程还在，端口还占着。"""
        self._timer.stop()
        if self._own_client:
            try:
                self.client.stop()
            except Exception:
                pass
