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

**显示方式（全局 / 局部）不在这里改**：那要进游戏走两步看「地形动不动」才知道，
所以沿用「路线识别」下拉里你选的那个 —— 和命令行工具的态度一致（程序不替你猜）。
"""

import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QDialog,
                             QGraphicsPixmapItem, QGraphicsScene, QHBoxLayout,
                             QLabel, QMessageBox, QPushButton, QVBoxLayout)

from core import mapdata
from gui.canvas import ZoomPanView    # 看图交互在几个窗口里是同一份
from gui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSlider
from gui.worker import safe_slot      # 槽里抛异常 = 整个工作台 abort（见 worker.py）
from perception import minimap as mm
from tools.config import load_live

#: 缩放滑块：0.200 ~ 12.000。
#: 上限从 4.0 放到 12.0：小底图（几十像素）被客户端放大若干倍、A 机的推流 zoom
#: 又乘一次，面板里的实际倍数能到 9~10 倍（实测猴林迷宫I：底图 78×203、面板
#: 753×612 → 光宽度就差 9.7 倍）。卡在 4 倍时，人手动也拖不到正确位置。
SLIDER_MIN = 200
SLIDER_MAX = 12000
SPIN_MAX = 12.0

#: 匹配分低于这个值时，「保存」会先问一句。0.55 = locate 内部那条 min_score
SCORE_WARN = 0.55


def np_to_pixmap(img):
    """numpy BGR → QPixmap（小地图帧/底图都是 BGR）。"""
    import cv2
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class _OverlayItem(QGraphicsPixmapItem):
    """半透明底图叠加层：可拖动，拖完通知外层重算几何。

    `_busy` 是防重入的：外层收到通知后会把 item 的位置**规范化**（取整、
    或按模式换算 offset/view）再 setPos 回来 —— 那又会触发一次 itemChange，
    没有这个标记就会自己叫自己，一路递归下去。
    """

    def __init__(self, on_moved):
        super().__init__()
        self._on_moved = on_moved
        self._busy = False

    def itemChange(self, change, value):
        if (change == QGraphicsPixmapItem.ItemPositionHasChanged
                and self._on_moved is not None and not self._busy):
            self._on_moved()
        return super().itemChange(change, value)


class MinimapCalibDialog(QDialog):
    """量「面板 → 底图」的缩放/偏移（存 datasets/map/<id>.mapcalib.json）。"""

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
        self.scale = float(cal.get("scale") or 1.0)
        self.offset = [int(v) for v in (cal.get("offset") or [0, 0])]
        # crop 专用：现在显示的是底图从**这一块**开始的内容。
        # ⚠ 不叫 self.view —— 那个名字留给画布控件（和 calib_manual 一致），
        # 两边同名过一次，结果控件被几何字段覆盖掉（AttributeError 一大串）。
        self.block = [int(v) for v in (cal.get("view") or [0, 0])]
        self.score = float(cal.get("score") or 0.0)
        self.inset = 4                     # crop 的边框内缩（与 locate_crop 一致）
        # crop 下 `offset` **就是 inset**（模板从面板边缘往里缩了多少，
        # 与 locate_crop 的返回一致）。老文件里可能根本没有这个键（默认 [0,0]）
        # —— 那样存下去的东西会比人工对齐差 inset 个底图像素（在
        # px_per_world≈16 的图上是 63 世界单位，现场就是被这个坑到）。
        # 归一化放这里：从文件读来的这个值不可信，唯一可信的是 inset 本身。
        if self.mode == mm.MODE_CROP:
            self.offset = [self.inset, self.inset]
        #: 叠加层透明度（%）。对齐时要能同时看清「面板」和「底图」，全靠它。
        self.alpha = int(cal.get("alpha") or 55)
        # 叠加参照：canvas（WZ 那张底图）/ terrain（我们自己渲染的地形叠加图）。
        # **为什么要有得选**：实测有些客户端的小地图是**按 foothold 现画**的，
        # 跟 WZ 里那张 `miniMap/canvas` 完全是两套画（白底绿棕 vs 黑底蓝地形），
        # 叠那张图人工也永远对不上；叠「平台形状一致」的地形图才对得上。
        self.ref = (cal.get("ref") or "canvas")
        self._ref_pix = None
        self._ref_zoom = 1                 # 参照图 1 像素 = 多少底图像素
        self._frame = None
        self._scene_wh = None
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

    # ---------------- 界面 ----------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.lbl_where = QLabel()
        top.addWidget(self.lbl_where)
        top.addStretch(1)
        self.lbl_mode = QLabel()
        self.lbl_mode.setStyleSheet("color:#5f6368;")
        self.lbl_mode.setToolTip(
            "显示方式在「路线识别 → 小地图定位」里选（要进游戏走两步看地形动不动\n"
            "才知道该用哪种），这里只是沿用你选的那个，不替你改。")
        top.addWidget(self.lbl_mode)

        top.addSpacing(10)
        top.addWidget(QLabel("叠加参照"))
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
        top.addWidget(self.cmb_ref)
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

        # ---- 状态 + 收流控制 ----
        # 两行分开写：一行是**连接状态**（收帧中 / 等帧），一行是**动作结果**
        # （定位成功/失败、保存到哪）。挤在一个 label 里会互相覆盖 —— 定位结果
        # 刚写上就被下一拍的状态刷新冲掉，人只看到"收帧中"，像没定位过。
        row = QHBoxLayout()
        row.setSpacing(6)
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
        root.addLayout(row)

        self.lbl_say = QLabel()
        self.lbl_say.setWordWrap(True)
        self.lbl_say.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_say)

        # ---- 画布：面板帧（底）+ 半透明底图（可拖）----
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
        self._ov_item = _OverlayItem(self._on_overlay_moved)
        self._ov_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)
        # ItemSendsGeometryChanges **必须开**：不开的话 Qt 会把位置变化的
        # itemChange 优化掉 —— 表现就是"拖了、画面动了、量出来的数一个没变"
        # （自检里就是这么逮到的）。
        self._ov_item.setFlag(QGraphicsPixmapItem.ItemSendsGeometryChanges, True)
        self._ov_item.setOpacity(max(0, min(100, self.alpha)) / 100.0)
        self.scene.addItem(self._bg_item)
        self.scene.addItem(self._ov_item)
        root.addWidget(self.view, 1)

        # ---- 缩放（fit 有效；crop 固定 1:1）----
        srow = QHBoxLayout()
        srow.setSpacing(8)
        srow.addWidget(QLabel("缩放"))
        self.sld = NoWheelSlider(Qt.Horizontal)
        self.sld.setRange(SLIDER_MIN, SLIDER_MAX)
        self.sld.valueChanged.connect(self._on_slider)
        srow.addWidget(self.sld, 1)
        self.sp_scale = NoWheelDoubleSpinBox()
        self.sp_scale.setRange(0.05, SPIN_MAX)
        self.sp_scale.setDecimals(3)
        self.sp_scale.setSingleStep(0.01)
        self.sp_scale.setSuffix("×")
        self.sp_scale.setMinimumWidth(96)
        self.sp_scale.valueChanged.connect(self._on_spin)
        srow.addWidget(self.sp_scale)

        # ---- 叠加层透明度（WZ 素材那层的浓淡）----
        srow.addSpacing(10)
        srow.addWidget(QLabel("透明度"))
        self.sld_alpha = NoWheelSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(int(self.alpha))
        self.sld_alpha.setFixedWidth(150)
        self.sld_alpha.setToolTip(
            "面板上面那层「WZ 素材/地形图」的浓淡（%）。\n"
            "对齐时两头都要看得见：太浓只看见底图，太淡又看不见地形 ——\n"
            "数值跟着标定存，下次打开还是它。")
        self.sld_alpha.valueChanged.connect(safe_slot(self._on_alpha))
        srow.addWidget(self.sld_alpha)
        self.lbl_alpha = QLabel("%d%%" % self.alpha)
        self.lbl_alpha.setMinimumWidth(42)
        self.lbl_alpha.setStyleSheet("color:#80868b;")
        srow.addWidget(self.lbl_alpha)
        root.addLayout(srow)

        # ---- 判据：角点世界坐标 vs 世界范围 ----
        self.lbl_judge = QLabel()
        self.lbl_judge.setWordWrap(True)
        self.lbl_judge.setStyleSheet("color:#3c4043;")
        root.addWidget(self.lbl_judge)

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

        brow.addStretch(1)
        self.btn_save = QPushButton("保存标定")
        self.btn_save.clicked.connect(safe_slot(self._on_save))
        brow.addWidget(self.btn_save)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)
        brow.addWidget(btn_close)
        root.addLayout(brow)

        hint = QLabel("拖动画面里的半透明底图，让它和下面的面板重合，然后「保存标定」。"
                      "　方向键=挪 1 个底图像素（Shift=5 个）　滚轮=缩放视图　"
                      "中键拖=平移视图　双击=适应窗口　"
                      "不勾「实时画面」＝画面停住，方便对着量。")
        hint.setStyleSheet("color:#80868b;")
        hint.setWordWrap(True)
        root.addWidget(hint)

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
            how = ("手工对齐" if cal.get("src") == "manual"
                   else "自动匹配 %.2f" % float(cal.get("score") or 0.0))
            # 老格式没记来源 → **不能**写成「来源：从实时画面」：那是猜的，而这份
            # 几何只对当时那条来源成立（说成当前来源就是让人放心用错的数）。
            shown = "未记·老格式" if cal.get("legacy") else label
            self.lbl_loaded.setText("已载入上次标定（来源：%s）：%s%s"
                                    % (shown, how, when))
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
            self.lbl_loaded.setToolTip(tip)
        else:
            self.lbl_loaded.setText(
                "这张图在「%s」下还没标定过：现在摆的是程序猜的粗略位置 —— "
                "调好重合后点「保存标定」，否则下次打开还是这样。" % label)
            self.lbl_loaded.setStyleSheet("color:#b06000;")
            self.lbl_loaded.setToolTip(
                "还没有可用的几何：\n%s\n\n"
                "（文件不存在，或者里面只有「显示方式」—— 那不算标定过。）\n"
                "标定是**按来源分开存**的：这里只认「%s」那一份，标的是另一条"
                "来源也照样算没标。\n"
                "「保存标定」会写进上面这个文件。" % (p, label))

    def _snapshot(self):
        """几何指纹：只取决定换算的那四项（透明度、参照图这些不算改动）。"""
        c = self.calib()
        return (c.get("mode"),
                round(float(c.get("scale") or 0.0), 4),
                tuple(int(v) for v in (c.get("offset") or [0, 0])),
                tuple(int(v) for v in (c.get("view") or [0, 0])))

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

    def calib(self):
        """当前几何 → 标定 dict（和命令行工具存的是同一套字段）。

        ⚠ crop 的 `offset` **必须写成 inset**（不能用从文件里读来的那个数）：
        它是「模板从面板边缘往里缩了多少」，`panel_to_canvas` 拿它做换算 ——
        写成 0 等于说「没缩」，于是程序按文件算出来的世界坐标和人工对齐差
        inset 个底图像素（`t_save_then_reopen` 钉着这条）。
        """
        off = ([int(self.inset), int(self.inset)] if self.mode == mm.MODE_CROP
               else [int(self.offset[0]), int(self.offset[1])])
        return {"mode": self.mode,
                "scale": round(float(self.scale), 4),
                "offset": off,
                "view": [int(self.block[0]), int(self.block[1])],
                "score": round(float(self.score), 4),
                # 这份几何是**量出来的**还是**人眼对齐的**：手工对齐没有匹配分，
                # 而「标过没有」不能拿 score 判（见 mm.has_geometry）。
                "src": "manual" if self.manual else "auto",
                # 记下人工对齐时用的是哪张参照图 + 那层叠多浓：下次打开接着用
                "ref": self.ref,
                "alpha": int(self.alpha)}

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
            self._ref_pix = np_to_pixmap(self.canvas)
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
        """标定字段 → 叠加层的位置/缩放（fit 用 scale+offset，crop 用 view）。

        `self.scale` 一律是**面板像素 / 底图像素**；参照图可能是放大过的
        （地形叠加图是底图的 overlay_zoom 倍），所以 item 的缩放要除以 `_ref_zoom`。
        """
        if self.canvas is None:
            return
        k = float(self.scale) / max(1, self._ref_zoom)
        self._ov_item._busy = True
        try:
            if self.mode == mm.MODE_CROP:
                # crop = 面板显示底图的**一块**，而这一块可能被放大了 scale 倍
                # （小底图被客户端放大，实测能到 9~10 倍）。所以叠加层也跟着缩放：
                # 让「底图坐标 block」那一点正好落在面板的 (inset, inset) 上。
                self._ov_item.setScale(k)
                self._ov_item.setPos(-self.block[0] * self.scale + self.inset,
                                     -self.block[1] * self.scale + self.inset)
            else:
                self._ov_item.setScale(k)
                self._ov_item.setPos(self.offset[0], self.offset[1])
        finally:
            self._ov_item._busy = False
        self._update_scene_rect()
        self._sync_widgets()
        self._refresh_judge()

    def _on_overlay_moved(self):
        """人拖了叠加层 → 反算回标定字段（拖动就是唯一的输入方式之一）。

        能走到这里就说明是**人**动的：程序自己摆位时 `_sync_overlay` 会挂
        `_busy` 挡住回调（见 `_OverlayItem`），所以这里标 manual 不会误标。
        """
        self.manual = True            # 人眼对齐 → 存进标定时记 src=manual
        pos = self._ov_item.pos()
        if self.mode == mm.MODE_CROP:
            s = max(1e-6, float(self.scale))
            self.block = [int(round((self.inset - pos.x()) / s)),
                          int(round((self.inset - pos.y()) / s))]
        else:
            self.scale = float(self._ov_item.scale()) * max(1, self._ref_zoom)
            self.offset = [int(round(pos.x())), int(round(pos.y()))]
        self._sync_widgets()
        self._refresh_judge()

    def _on_alpha(self, val):
        """透明度：只改叠加层的浓淡，几何一点不动（对齐时反复调的就是它）。"""
        self.alpha = int(val)
        self.lbl_alpha.setText("%d%%" % self.alpha)
        self._ov_item.setOpacity(max(0, min(100, self.alpha)) / 100.0)

    def _sync_widgets(self):
        """把 scale 回填到滑块/数字框（别让界面和实际几何不一致）。"""
        self.sp_scale.blockSignals(True)
        self.sp_scale.setValue(float(self.scale))
        self.sp_scale.blockSignals(False)
        self.sld.blockSignals(True)
        self.sld.setValue(max(SLIDER_MIN, min(SLIDER_MAX,
                                              int(round(self.scale * 1000)))))
        self.sld.blockSignals(False)
        # 两种方式都要能调缩放：crop 也可能是「放大后取一块」（实测小底图会到
        # 9~10 倍）—— 以前这里把 crop 的缩放禁掉了，正好把唯一的手动出路堵死。
        self.sld.setEnabled(True)
        self.sp_scale.setEnabled(True)

    def _on_slider(self, val):
        self.scale = val / 1000.0
        self._sync_overlay()

    def _on_spin(self, val):
        self.scale = float(val)
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

        self.scale = float(loc["scale"])
        self.offset = [int(v) for v in loc["offset"]]
        self.block = [int(v) for v in (loc.get("view") or [0, 0])]
        self.score = float(loc["score"])
        self.manual = False           # 自动量出来的 → 存进标定时记 src=auto
        self._sync_overlay()
        # 顺带报一下这次定位花了多久：这一路以后要接进实时线程，耗时是硬指标
        #（B 机关键路径本来就没余量，见 docs/寻路设计.md §5 的性能一节）。
        self._say("定位成功　匹配分 %.3f　%.0f ms%s"
                  % (self.score, dt * 1000, move),
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
                       ("局部小地图", mm.locate_crop(self._frame, self.canvas))):
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
        self.lbl_say.setText(text)
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
            self.lbl_conn.setText("%s　%s" % (base, err) if err else base)
        elif not self.ck_live.isChecked():
            self.lbl_conn.setStyleSheet("color:#5f6368;")
            self.lbl_conn.setText("画面已停住（「实时画面」没勾）")
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
        self.lbl_judge.setText(
            "匹配分 %.3f　面板左上 → 世界 (%.0f, %.0f)　右下 → 世界 (%.0f, %.0f)　"
            "世界范围 (%.0f, %.0f) ~ (%.0f, %.0f)"
            % (self.score, lx, ly, rx, ry, b[0], b[1], b[2], b[3]))
        self.lbl_judge.setStyleSheet("color:%s;"
                                     % ("#188038" if ok else "#b06000"))

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
                    rect_o = (ox * z, oy * z, (ox + cw * c["scale"]) * z,
                              (oy + ch * c["scale"]) * z)
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
        self.btn_overlay.setText("关闭叠加图" if on else "打开叠加图")
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
        if self.score < SCORE_WARN:
            if self.manual:
                title = "手工对齐"
                text = ("这份几何是**你用眼睛对齐**的（没有匹配分）。\n\n"
                        "存下去之后，寻路完全按你的对齐算世界坐标 —— 人在这块平台、\n"
                        "程序以为在另一块，是这类标定最典型的错法。\n\n"
                        "再确认一眼：半透明底图和下面的面板**处处重合**吗？")
            else:
                title = "匹配分偏低"
                text = ("当前匹配分 %.3f（低于 %.2f）。\n\n"
                        "分数低说明面板和底图对不上 —— 存下去的话，寻路会按这个\n"
                        "错位的换算算世界坐标（表现是「人在这块平台，程序以为在\n"
                        "另一块」）。\n\n"
                        "建议先点「自动定位」；实在匹配不上（面板被 UI 挡住等）\n"
                        "再手动拖动对齐。" % (self.score, SCORE_WARN))
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
        cal = mapdata.load_calib(self.map_id, self.src) or {}
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

    def keyPressEvent(self, e):
        """方向键 1 像素、Shift+方向键 5 像素地挪**叠加层**（目测对齐最后那一两像素）。

        方向语义和拖动一致：按左 = 叠加层往左走（于是「看到的是底图哪一块」右移）。
        别按「数字往哪边走」去理解 —— 人对着画面按，手感才是对的。
        """
        # 一格 = **1 个底图像素**（在放大后的面板上就是 scale 个像素）：
        # 直接按 1 个屏幕像素挪，在 9 倍放大的图上根本动不了几何（block 是整数）。
        base = max(1.0, float(self.scale))
        step = (5.0 if (e.modifiers() & Qt.ShiftModifier) else 1.0) * base
        d = {Qt.Key_Left: (-step, 0), Qt.Key_Right: (step, 0),
             Qt.Key_Up: (0, -step), Qt.Key_Down: (0, step)}.get(e.key())
        if d is None or self.canvas is None:
            super().keyPressEvent(e)
            return
        pos = self._ov_item.pos()
        self._ov_item.setPos(pos.x() + d[0], pos.y() + d[1])   # → _on_overlay_moved
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
