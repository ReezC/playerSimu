"""双点标定弹窗：走两个地方各采一次 → 解出「面板 ↔ 底图」的缩放与偏移。

**怎么用**（用户 2026-09-27 定的布局与交互：一个下拉 + A/B 两行 + 每行右边一个按钮）：
    1. 走到一个**你知道确切位置**的地方；
    2. 左边「离线」那两格填**那个位置**在地图上的坐标（默认口径 = 世界坐标，从
       「寻路编辑器」悬停时那行 `鼠标 (x, y)` 读；也可以切成底图像素）；
    3. 右边点「实时填入当前黄点脚底」⇒ 程序认一次黄点，把它**下沿**（脚底）填进「实时」两格；
    4. 换个地方再做一遍（两行）；
    5. 点「算一算」→ 看缩放/偏移 →「保存标定」。

**为什么是"黄点脚底"而不是重心**（这条是这扇窗最要紧的口径，别改成重心）：
    游戏把小地图上的玩家标记画成一个小黄点，**它的下沿对着玩家的脚下**；而
    `find_player_dot` 给的是那个点的**重心**，比脚底高约半个点。⇒ 这里填**脚底**
    （`perception.minimap.dot_feet`），标出来的几何就是"地图内容 ↔ 面板"的真映射
    （叠图也对得上 ✓）。

    ⚠ **读数那条路也走同一个 `dot_feet`**（2026-09-27 改）：以前它拿的是**重心** ⇒
    采样填脚底、读数读重心，站回同一个地方读数就差半个点高（这张图 1 个面板像素 =
    7.9 世界像素，6 像素高的点差 12 世界像素 —— 用户现场报"站在 L1 上端该是
    (56,-205)、显示 (48,-222)"就是这个 ✗）。两边同一口径之后，「坐标系偏移」也从
    "重心 ↔ 玩家原点"（≈ 半个点高，老值是 (7,33)）变成"下沿 ↔ 玩家原点"的**残差**，
    正常接近 **(0, 0)** ✓。

**和「标定…」那扇窗的分工**：两扇窗写**同一份**标定（按来源分开存），谁最后保存谁说了算。
这扇窗量出来的 `src` 是 `two_point`（不是模板匹配、也不是目测）。
"""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout)

from core import mapdata
from gui import theme                  # 弹窗几何按客户端存（config/ui.yaml）✓
from gui.widgets import NoWheelComboBox, NoWheelSpinBox
from gui.worker import safe_slot                # 槽里抛异常 = 整个工作台 abort（见 worker.py）
from perception import minimap as mm

#: 表格里那两列的顺序（离线在前、实时在后）—— `_build` 与 `_uv` 共用一份：
#: 顺序写两遍就会"填进去的和读出来的不是同一列"，而那种错**看着像标定不准** ✗。
OFF_LIVE = ("off", "live")


class TwoPointCalibDialog(QDialog):
    """双点标定：两次采样（离线位置 + 实时黄点脚底）→ 缩放/偏移。

    **非模态**（2026-09-27 用户要求："开双点标定窗口时可以操作数据集工作台主窗口"）：
    量这个要一边操作工作台 —— 到「实时」页开预览/看画面、切到「路线识别」看世界读数、
    去「设置」调叠图浓淡对着看……模态框会把整个工作台锁住，只能反复关窗开窗 ✗。
    非模态之后 `exec_()` 那条返回值没有了 ⇒ 保存结果靠 `saved_now` 信号交给调用方
    （和 `ZoneEditorDialog.saved_now` 同一个路子 ✓）；窗口由调用方按**单实例**挂住。
    """

    #: 保存成功 → 叫一声（非模态下调用方拿不到返回值，只能靠信号知道"存过了"）。
    #: 不带参数：接的人（路线面板的 `_refresh_mmap`）只用它当"该重读一遍"的提示 ✓。
    saved_now = pyqtSignal()

    def __init__(self, map_id, src=None, client=None, parent=None,
                 mode=None, host=None, port=None):
        super().__init__(parent)
        self.setWindowTitle("双点标定")
        self.setModal(False)                # 模态会把工作台锁住（见类说明）
        self.map_id = str(map_id or "")
        # ⚠ 兜底来源必须和「标定…」那扇窗**同一套规则**（`config/live.yaml` 的 `mmap_src`）：
        #   两扇窗写/读的是同一份文件的同一条来源（`sources.<来源>`），兜底规则不一致就会
        #   出现"我在普通标定里存的，双点标定里看不见"（用户 2026-09-27 说的"应该引用的
        #   是一套数据"）。调用方（路线识别）本来就会显式传 src ✓，这里只是把兜底对齐 ✓。
        from tools.config import load_live
        self.src = src or (load_live().get("mmap_src") or mm.SRC_STREAM)
        #: 「保存标定」成功过 → 调用方拿它决定要不要立刻刷新状态行（同 `MinimapCalibDialog`）
        self.saved = False
        #: 现在这份标定：保存时保留 mode / alpha / ref / world_offset … 这些字段。
        #: ⚠ `mode` 由调用方（「显示方式」下拉）带进来 —— 它决定这份几何的**写法**
        #:   （fit 直接存 offset；crop 的 offset 那套约定是 inset、截距进 `view`，
        #:   见 `perception.minimap.calib_from_two_point`）。没带就沿用文件里的 / fit。
        self._cal = dict(mapdata.load_calib(self.map_id, self.src) or {})
        if mode:
            self._cal["mode"] = mode
        elif not self._cal.get("mode"):
            self._cal["mode"] = mm.MODE_FIT
        self._result = None                 # 最近一次「算一算」的结果
        self._frame = None                  # 实时小地图那一帧（认黄点要用）
        #: 地形/底图：换"世界坐标 ↔ 底图像素"要用它（拿不到就老实说，别给个"填了没用"的入口）
        self._terrain = mapdata.load(self.map_id, with_canvas=True)
        self._dot = None                    # 最近一次认到的黄点结论（报脚底用）
        from tools.config import get
        self.host = host or get("a_host")
        self.port = int(port or get("minimap", "port", 5003))
        #: client 传进来就**不接管它的生命周期**（实时那路可能已经有一条在跑）；
        #: 没传就自己连一路，关窗自己停 —— 弹窗不许留下后台线程（同 MinimapCalibDialog）。
        self._own_client = client is None
        #: 我们那路收流被 `_shutdown` 停过（关窗停的）⇒ 下次 `showEvent` 要重建（见那儿）
        self._client_off = False
        self._build()
        self.resize(620, 380)               # 一屏放得下（用户那块屏 1366×768）
        self.client = client or mm.MiniMapClient(self.host, self.port).start()
        self._timer = QTimer(self)
        self._timer.setInterval(200)        # 只为"认黄点"能拿到当前这一帧
        self._timer.timeout.connect(safe_slot(self._tick))
        self._timer.start()
        self._refresh_now()
        self._tick()
        theme.bind_window_state(self, "two_point_calib")  # 拉过的大小/位置按客户端记住 ✓

    # ---------------- 界面 ----------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self.lbl_where = QLabel(
            "地图 %s　来源：%s" % (self.map_id or "（未选）",
                                mm.SRC_LABEL.get(self.src, self.src)))
        root.addWidget(self.lbl_where)
        self.lbl_now = QLabel()
        self.lbl_now.setWordWrap(True)
        self.lbl_now.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_now)
        self.lbl_live = QLabel()
        self.lbl_live.setStyleSheet("color:#5f6368;")
        root.addWidget(self.lbl_live)

        # ---- 离线那两格填的是什么坐标（用户叫它「离线点坐标类型」）----
        # 默认**世界坐标**：那是「寻路编辑器」悬停时直接读得到的那套（`鼠标 (x, y)`），
        # 而底图像素没有哪个界面会报给你（要自己数像素）⇒ 默认给好用的那个 ✓。
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("离线点坐标类型"))
        self.cmb_unit = NoWheelComboBox()
        self.cmb_unit.addItem("世界坐标", "world")
        self.cmb_unit.addItem("像素", "px")
        self.cmb_unit.setToolTip(
            "「离线 x / 离线 y」那两格填的是哪种坐标：\n\n"
            "世界坐标（默认）：WZ 数据那套 —— foothold / ladderRope 的 x/y，\n"
            "  也就是「寻路编辑器」里悬停时那行 `鼠标 (x, y)`、以及选中某条 foothold\n"
            "  时状态行报的 x/y（绳子上端就是 (x, y1)）。\n"
            "  程序按地图数据自己换算成底图像素（`world_to_canvas`），这一步是精确的。\n\n"
            "像素：那个位置在 `datasets/map/<id>.png` 上的第几像素（要自己数）。\n\n"
            "⚠ 两种都行，但来源要对应：世界坐标要从 WZ / 编辑器里读（那是原值）。\n"
            "⛔ 别从「路线识别」那行的「玩家世界坐标」抄 —— 那套已经加了「坐标系偏移」，\n"
            "   抄进来等于把偏移也算进标定里，之后读数会变成两倍偏移。\n"
            "⛔ 也别把世界坐标当像素填：缩放会算出一个离谱的数（结果行看得出来）。")
        self.cmb_unit.currentIndexChanged.connect(safe_slot(self._on_unit))
        row.addWidget(self.cmb_unit)
        row.addStretch(1)
        root.addLayout(row)

        # ---- 表：两行（A / B）× 四列（离线 x/y、实时 x/y）+ 每行一个「认黄点」----
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        for col, name in enumerate(("离线 x", "离线 y", "实时 x", "实时 y")):
            lab = QLabel(name)
            lab.setStyleSheet("color:#5f6368;")
            grid.addWidget(lab, 0, col + 1)
        self._sp = {}
        self._btn_dot = []
        for i, tag in enumerate(("A", "B")):
            grid.addWidget(QLabel(tag), i + 1, 0)
            for j, kind in enumerate(OFF_LIVE):
                bx, by = NoWheelSpinBox(), NoWheelSpinBox()
                for b in (bx, by):
                    # 离线那两格可能装世界坐标（一张大图的跨度能到几万）⇒ 范围放宽；
                    # 实时那两格永远是面板像素 ⇒ 小范围就够（也更好按上下键微调）
                    b.setRange(-99999 if kind == "off" else -9999,
                               99999 if kind == "off" else 9999)
                    b.setMinimumWidth(64)   # 最小宽度够用即可，不用 setFixedWidth（规范 §4）
                grid.addWidget(bx, i + 1, j * 2 + 1)
                grid.addWidget(by, i + 1, j * 2 + 2)
                for b in (bx, by):
                    b.valueChanged.connect(safe_slot(self._on_value_changed))
                self._sp[(tag, kind)] = (bx, by)
            btn = QPushButton("实时填入当前黄点脚底")
            btn.setToolTip(
                "用「当前这一帧」实时小地图认一次黄点，把它「下沿（脚底）」填进这一行的\n"
                "「实时 x / 实时 y」—— 游戏把玩家标记画成一个小黄点、下沿对着脚下，\n"
                "所以脚底才是「这个位置」该在的像素（读数那条路取的**也是**下沿，\n"
                "两边同一口径，站回同一个地方读数才对得上 ✓）。\n\n"
                "⚠ 「坐标系偏移」现在补的是「下沿 → 玩家原点」那点**残差**，正常接近 (0, 0)；\n"
                "   老那份 (7,33) 是按**重心**量的，别照抄进来（那是双重补偿）。\n\n"
                "认不出会说明原因（面板没对上 / 黄点被挡 / 颜色层被淹等），不瞎填。")
            btn.clicked.connect(safe_slot(lambda _c=False, k=i: self._on_pick_dot(k)))
            grid.addWidget(btn, i + 1, 5)
            self._btn_dot.append(btn)
        grid.setColumnStretch(6, 1)
        root.addLayout(grid)

        hint = QLabel(
            "走到一个你知道确切位置的地方 → 左边填那个位置在地图上的坐标"
            "（从寻路编辑器/WZ 里读，别从「玩家世界坐标」那行抄）、"
            "右边点按钮读黄点脚底 → 换个地方再来一次 → 「算一算」→「保存标定」。\n"
            "⚠ 两个位置别在同一行或同一列（那样有一轴的缩放解不出来）。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#5f6368;")
        root.addWidget(hint)

        self.lbl_say = QLabel("")
        self.lbl_say.setWordWrap(True)
        self.lbl_say.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_say.setStyleSheet("color:#80868b;")
        root.addWidget(self.lbl_say)

        brow = QHBoxLayout()
        brow.setSpacing(6)
        self.btn_calc = QPushButton("算一算")
        self.btn_calc.setToolTip("按填的四格坐标解出缩放与偏移。\n"
                                 "退化输入（两个点同一行/同一列/坐标一样）会被拦下来说清楚。")
        self.btn_calc.clicked.connect(safe_slot(self._on_calc))
        brow.addWidget(self.btn_calc)
        brow.addStretch(1)
        self.btn_save = QPushButton("保存标定")
        self.btn_save.clicked.connect(safe_slot(self._on_save))
        brow.addWidget(self.btn_save)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)
        brow.addWidget(btn_close)
        root.addLayout(brow)

    # ---------------- 收帧 / 口径 ----------------

    def _tick(self):
        """取一帧实时小地图（只为"认黄点"用；没有就把状态写明白）。"""
        frame, _t = self.client.latest()
        if frame is None:
            self.lbl_live.setText(
                "实时画面：还没收到（%s）"
                % (getattr(self.client, "hint", None)
                   or ("A 机的「小地图推流」启动了吗？"
                       if self.src == mm.SRC_STREAM else "先到「实时」页开始预览")))
            return
        self._frame = frame
        self.lbl_live.setText("实时画面：%d×%d" % (frame.shape[1], frame.shape[0]))

    def _off_is_world(self):
        """离线那两格现在填的是世界坐标吗。"""
        return (self.cmb_unit.currentData() or "world") == "world"

    def _on_unit(self, *_a):
        """换口径时把已有数字一并换算过去 —— 不换算的话，格子里留着的就是**另一种口径**
        的数，而界面显示"现在填世界坐标"⇒ 人会以为它已经换好了 ✗（这类"看着对其实错"
        最难查）。换不了（没地图数据）就原样留着并说一句。"""
        if self._terrain is None:
            self._say("这张图没有底图/换算数据，坐标口径换不了（先「生成地形图」）。",
                      bad=True)
            return
        for tag in ("A", "B"):
            bx, by = self._sp[(tag, "off")]
            x, y = float(bx.value()), float(by.value())
            if (x, y) == (0.0, 0.0):
                continue                     # 还没填，别把 0 换成别的数
            if self._off_is_world():
                wx, wy = self._terrain.canvas_to_world(x, y)
            else:
                wx, wy = self._terrain.world_to_canvas(x, y)
            bx.setValue(int(round(wx)))
            by.setValue(int(round(wy)))
        self._say("离线那两格现在按「%s」填。" % ("世界坐标" if self._off_is_world()
                                              else "底图像素"))

    # ---------------- 认黄点 ----------------

    def _on_pick_dot(self, i):
        """「实时填入当前黄点脚底」：认一次黄点，把**下沿**填进第 `i` 行的实时两格。

        为什么值得做：程序**本来就要**认这个点（定位全靠它），而这扇窗要的正是"人现在
        在哪" —— 让人拿鼠标去描一个 2~5 像素的点，既是白费功夫也是误差来源 ✗。
        几何优先用**刚解出来的**那份（ROI/底图相减都更准），没有就用文件里那份。
        """
        if self._frame is None:
            self._say("还没有实时画面 —— 先让「实时画面」那行显示出尺寸再认。", bad=True)
            return
        cal = self._cal
        if self._result and self._result.get("ok"):
            cal = mm.calib_from_two_point(self._cal, self._result)
        r = mm.find_player_dot(self._frame, cal, self._terrain)
        if not r["ok"]:
            self._dot = None
            self._say("黄点没认出来：%s" % r.get("reason", ""), bad=True)
            return
        self._dot = r
        x, y = mm.dot_feet(r)               # 口径只有一处（`perception.minimap.dot_feet`）
        sp = self._samp_row(i)
        # 采样格是**整数**框 ⇒ 在这里取整（`round` 到最近，别 `int()` 截断）——
        # `dot_feet` 给的是**亚像素**（2026-09-27 起读数那条路直接用小数 ✓）
        sp[0].setValue(int(round(x)))
        sp[1].setValue(int(round(y)))
        self._say("采样%s：黄点认在 (%d, %d)　下沿（脚底）填 (%d, %d)　%s族/%s层　像点 %d×%d"
                  % ("一二"[i], int(round(r["x"])), int(round(r["y"])), x, y,
                     "黄" if r["family"] == "yellow" else "青",
                     "颜色" if r["layer"] == "color" else "底图相减",
                     int(r["w"]), int(r["h"])))

    def _samp_row(self, i):
        """第 `i` 行的四个格子（顺序见 `OFF_LIVE`）。"""
        tag = "AB"[i]
        return (self._sp[(tag, "live")][0], self._sp[(tag, "live")][1])

    # ---------------- 算 / 存 ----------------

    def _uv(self, tag, kind):
        """取一格的两个数（离线那侧按当前口径换算成**底图像素**）。"""
        bx, by = self._sp[(tag, kind)]
        x, y = float(bx.value()), float(by.value())
        if kind != "off" or not self._off_is_world():
            return (x, y)
        if self._terrain is None:
            return (x, y)                       # 没底图数据 ⇒ 交给核心去判退化/说不清
        return tuple(float(v) for v in self._terrain.world_to_canvas(x, y))

    @staticmethod
    def _round2(p):
        return "(%.1f, %.1f)" % (p[0], p[1])

    def _on_value_changed(self, *_a):
        """任何一格被改动 ⇒ 上次「算一算」的结果**作废**。

        ⚠ 为什么必须：这扇窗现在**关掉不销毁**（数字留着，见 `showEvent`）⇒ `_result`
          也会跟着留 ✗。不设这道闸的话，"改了两个数、忘了重新算、直接点保存"就会把
          **上一版**几何写进标定 —— 而界面看着一切正常（最难查的那类）✗。
        """
        if self._result is not None:
            self._result = None
            self._say("坐标改过了 ⇒ 重新点「算一算」再保存。")

    def _on_calc(self):
        cA, pA = self._uv("A", "off"), self._uv("A", "live")
        cB, pB = self._uv("B", "off"), self._uv("B", "live")
        r = mm.solve_two_point(cA, pA, cB, pB)
        # ⚠ 下面的 `cA…pB` 与上面同一份（解与跨度都要用）；别删上面那两行 ——
        #   解退化输入之前就得有值（`solve_two_point` 拿的就是它们）。
        if not r["ok"]:
            self._result = None
            self._say("算不出来：%s" % r["why"], bad=True)
            return
        self._result = r
        # 采样几何：两轴各跨了多少、1 个实时像素的读数差等于多少缩放误差 —— 这几个数
        # 一起决定**这份标定能有多准**。用户 2026-09-27 那份：y 方向只隔 27 个实时像素
        # ⇒ 1 像素读数差 = 3.7% 的 y 缩放误差，报出来的"两轴差 8.43%"基本是噪声；而当时
        # 只看得到那个百分比，分不清"两点没对准"还是"两点太近" ✗。
        dcx, dcy = abs(cB[0] - cA[0]), abs(cB[1] - cA[1])      # 底图像素跨度
        dpx, dpy = abs(pB[0] - pA[0]), abs(pB[1] - pA[1])      # 实时像素跨度
        lines = ["缩放 x %.4f　y %.4f（两轴差 %.2f%%）"
                 % (r["scale"], r["scale_y"], r["axis_gap_pct"]),
                 "偏移 (%.1f, %.1f)" % (r["offset"][0], r["offset"][1]),
                 "两点跨度：底图 %.1f×%.1f 像素、实时 %.0f×%.0f 像素"
                 % (dcx, dcy, dpx, dpy),
                 "实时那两格每错 1 像素 ⇒ x 缩放差 %.2f%%、y 缩放差 %.2f%%"
                 % ((100.0 / dpx) if dpx else 0.0, (100.0 / dpy) if dpy else 0.0),
                 "按等比再拟合的最大残差 %.2f 实时像素（这个数才是「两点对没对准」的自查）"
                 % r["resid_px"]]
        k = float(getattr(self._terrain, "px_per_world", 0) or 0)
        if k > 0:
            lines.append("1 个实时像素 = %.1f 世界像素(x) / %.1f 世界像素(y)；残差 = %.0f 世界像素"
                         % (k / r["scale"], k / r["scale_y"],
                            r["resid_px"] * k / r["iso_scale"]))
        lines.append("（游戏把小地图**等比**缩放 ⇒ 两个缩放本该几乎相等；差得多说明两次"
                     "采样没对准同一块地标 —— 也可能只是**两点太近**：跨度越小越不准，"
                     "取到**对角**上最稳。）")
        if self._off_is_world():
            # 填的是世界坐标 ⇒ 把**换算后**的底图像素也摆出来：万一填错口径/地图不对，
            # 这一行是唯一能看出来的地方（不然只看到一个"看着像对的"缩放）
            lines.insert(1, "（离线按世界坐标换算成底图像素：A %s　B %s）"
                            % (self._round2(cA), self._round2(cB)))
        if self._dot is not None:
            lines.append("（注：这里填的是黄点下沿（脚底），定位读数走的**也是**同一个"
                         "下沿 —— 两边同一口径，站回同一个地方就对得上；"
                         "「坐标系偏移」补的是「下沿 → 玩家原点」那点残差，正常接近 (0, 0)。）")
        self._say("\n".join(lines))

    def _on_save(self):
        if self._result is None or not self._result.get("ok"):
            self._say("先「算一算」，算出几何了才能保存。")
            return
        cal = mm.calib_from_two_point(self._cal, self._result)
        try:
            mapdata.save_calib(self.map_id, cal, self.src)
        except Exception as e:                      # noqa: BLE001
            self._say("写文件失败：%s" % e, bad=True)
            return
        self.saved = True
        self._cal = cal
        sx, sy = mm.scales_of(cal)
        self._say("已保存到 %s（来源：%s）：缩放 x %.4f / y %.4f，偏移 %s。\n"
                  "「路线识别」那页会立刻重读这份几何（这扇窗是非模态的，你可以在那边"
                  "继续操作）。"
                  % (mapdata.calib_path(self.map_id),
                     mm.SRC_LABEL.get(self.src, self.src), sx, sy, cal.get("offset")))
        self._refresh_now()
        # 非模态 ⇒ 调用方拿不到返回值，只能靠信号知道"存过了"（它据此刷状态行/叠图）
        self.saved_now.emit()

    def _refresh_now(self):
        """顶部那行：这份标定现在长什么样（**两轴不同要分开写**，别只报一个数）。"""
        cal = mapdata.load_calib(self.map_id, self.src) or {}
        if not mm.has_geometry(cal):
            self.lbl_now.setText("现在这份：这张图在「%s」下还没标定过。"
                                 % mm.SRC_LABEL.get(self.src, self.src))
            self.lbl_now.setStyleSheet("color:#b06000;")
            return
        sx, sy = mm.scales_of(cal)
        src_tag = cal.get("src")
        if src_tag == "manual":
            how = "人眼对齐"
        elif src_tag == "two_point":
            how = "双点标定"
        elif src_tag:
            how = "模板匹配 %.2f" % float(cal.get("score") or 0.0)
        else:
            how = "没记来源"
        self.lbl_now.setText(
            "现在这份：%s　缩放 %s　偏移 %s　（%s）"
            % (mm.SRC_LABEL.get(self.src, self.src),
               ("x %.3f / y %.3f" % (sx, sy)) if abs(sy - sx) > 1e-9 else "%.3f" % sx,
               cal.get("offset"), how))
        self.lbl_now.setStyleSheet("color:#3c4043;")

    def _say(self, text, bad=False):
        self.lbl_say.setText(text)
        self.lbl_say.setStyleSheet("color:#c5221f;" if bad else "color:#188038;")

    # ---------------- 收尾 ----------------

    def showEvent(self, e):
        """重新显示（非模态下这扇窗会反复开关）⇒ 把收帧那一路接回来。

        ⚠ 关窗时 `_shutdown()` 把那一路停了（不留后台线程）⇒ 再打开必须重建，
        否则「认黄点」永远说"还没收到画面"，而界面上没有任何迹象 ✗。
        ⚠ 反过来，**别在关窗时把实例扔掉**：你填的那几格数字就记在这些控件里
        （用户 2026-09-27 报："关掉后上次的数据就丢了"）。
        """
        super().showEvent(e)
        if not self._timer.isActive():
            self._timer.start()
        if self._own_client and self._client_off:
            self.client = mm.MiniMapClient(self.host, self.port).start()
            self._client_off = False
        self._tick()

    def closeEvent(self, e):
        self._shutdown()
        super().closeEvent(e)

    def done(self, code):
        self._shutdown()
        super().done(code)

    def _shutdown(self):
        """关窗必须停掉自己那路收流（弹窗关了线程还在、端口还占着就麻烦了）。

        ⚠ 只**停**、不销毁：控件里的数字要留着（见 `showEvent`），下次 `show()` 接回来。
        """
        self._timer.stop()
        if self._own_client:
            try:
                self.client.stop()
                self._client_off = True
            except Exception:                       # noqa: BLE001
                pass
