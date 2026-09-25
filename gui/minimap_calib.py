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
from PyQt5.QtWidgets import (QCheckBox, QDialog, QGraphicsPixmapItem,
                             QGraphicsScene, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout)

from core import mapdata
from gui.canvas import ZoomPanView     # 和质检台/标定弹窗**同一份**看图交互
from gui.widgets import NoWheelDoubleSpinBox, NoWheelSlider
from perception import minimap as mm

#: 缩放滑块：0.200 ~ 4.000（和「手动目测标定尺度」同量级）
SLIDER_MIN = 200
SLIDER_MAX = 4000

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
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("小地图标定")
        self.setMinimumSize(900, 720)
        #: 点过「保存标定」没有 —— 外层（路线识别）据此决定要不要刷新状态
        self.saved = False

        self.map_id = str(map_id)
        self.terrain = mapdata.load(self.map_id, with_canvas=True)
        self.canvas = None if self.terrain is None else self.terrain.canvas

        from tools.config import get
        self.host = host or get("a_host")
        self.port = int(port or get("minimap", "port", 5003))
        #: client 传进来就**不接管它的生命周期**（将来实时那路已经在跑时共用一条）；
        #: 没传就自己连一路，关窗时自己停 —— 不能让弹窗留下一个后台线程。
        self._own_client = client is None

        cal = mapdata.load_calib(self.map_id) or {}
        self.mode = mode or cal.get("mode") or mm.MODE_FIT
        self.scale = float(cal.get("scale") or 1.0)
        self.offset = [int(v) for v in (cal.get("offset") or [0, 0])]
        # crop 专用：现在显示的是底图从**这一块**开始的内容。
        # ⚠ 不叫 self.view —— 那个名字留给画布控件（和 calib_manual 一致），
        # 两边同名过一次，结果控件被几何字段覆盖掉（AttributeError 一大串）。
        self.block = [int(v) for v in (cal.get("view") or [0, 0])]
        self.score = float(cal.get("score") or 0.0)
        self.inset = 4                     # crop 的边框内缩（与 locate_crop 一致）
        self._frame = None
        self._scene_wh = None
        self._geom_ready = bool(cal.get("score"))   # 有旧标定就别再自己摆
        self._prev_anchor = None           # 上一拍的锚点（看位移用）
        self._last_locate = 0.0            # 上次定位时刻（自动模式节流用）

        self._build()

        self.client = client or mm.MiniMapClient(self.host, self.port).start()
        self._timer = QTimer(self)
        self._timer.setInterval(100)       # 取帧很便宜（只是读一个引用）
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self._sync_overlay()
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
        root.addLayout(top)

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
        self._ov_item = _OverlayItem(self._on_overlay_moved)
        self._ov_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)
        # ItemSendsGeometryChanges **必须开**：不开的话 Qt 会把位置变化的
        # itemChange 优化掉 —— 表现就是"拖了、画面动了、量出来的数一个没变"
        # （自检里就是这么逮到的）。
        self._ov_item.setFlag(QGraphicsPixmapItem.ItemSendsGeometryChanges, True)
        self._ov_item.setOpacity(0.55)
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
        self.sp_scale.setRange(0.05, 8.0)
        self.sp_scale.setDecimals(3)
        self.sp_scale.setSingleStep(0.01)
        self.sp_scale.setSuffix("×")
        self.sp_scale.setMinimumWidth(96)
        self.sp_scale.valueChanged.connect(self._on_spin)
        srow.addWidget(self.sp_scale)
        root.addLayout(srow)

        # ---- 判据：角点世界坐标 vs 世界范围 ----
        self.lbl_judge = QLabel()
        self.lbl_judge.setWordWrap(True)
        self.lbl_judge.setStyleSheet("color:#3c4043;")
        root.addWidget(self.lbl_judge)

        # ---- 按钮 ----
        brow = QHBoxLayout()
        brow.setSpacing(6)
        self.btn_grab = QPushButton("抓一帧")
        self.btn_grab.setToolTip("从 A 机推来的流里取最新一帧（A 机没在推时这里会说明）")
        self.btn_grab.clicked.connect(self._on_grab)
        brow.addWidget(self.btn_grab)

        self.btn_locate = QPushButton("自动定位")
        self.btn_locate.setToolTip("用模板匹配算出缩放/偏移（比手拖准，1 像素级）")
        self.btn_locate.clicked.connect(lambda: self._locate_now())
        brow.addWidget(self.btn_locate)

        self.btn_overlay = QPushButton("打开叠加图")
        self.btn_overlay.setToolTip(
            "把「当前小地图对应叠加图的哪一块」画到 <id>_overlay.png 上\n"
            "（寻路要往哪走、定位对不对，看那张图最直观）")
        self.btn_overlay.clicked.connect(self._draw_overlay)
        brow.addWidget(self.btn_overlay)

        brow.addStretch(1)
        self.btn_save = QPushButton("保存标定")
        self.btn_save.clicked.connect(self._on_save)
        brow.addWidget(self.btn_save)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)
        brow.addWidget(btn_close)
        root.addLayout(brow)

        hint = QLabel("拖动画面里的半透明底图，让它和下面的面板重合，然后「保存标定」。"
                      "　方向键=把叠加层挪 1 像素（Shift=5 像素）　"
                      "滚轮=缩放视图　中键拖=平移视图　双击=适应窗口")
        hint.setStyleSheet("color:#80868b;")
        hint.setWordWrap(True)
        root.addWidget(hint)

    # ---------------- 收帧 ----------------

    def _on_tick(self):
        frame, _t = self.client.latest()
        # 「实时画面」勾着才自动换帧；不勾就停在「抓一帧」抓到的那张上
        # （帧还在流里滚，只是不覆盖你在看的那张 —— 想专门量某一帧时用得上）
        if (self.ck_live.isChecked() and frame is not None
                and frame is not self._frame):
            self._take(frame)
        self._refresh_status()

    def _take(self, frame):
        """把这一帧拿来看/用来定位（实时画面与「抓一帧」都走这里）。"""
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
        """当前几何 → 标定 dict（和命令行工具存的是同一套字段）。"""
        return {"mode": self.mode,
                "scale": round(float(self.scale), 4),
                "offset": [int(self.offset[0]), int(self.offset[1])],
                "view": [int(self.block[0]), int(self.block[1])],
                "score": round(float(self.score), 4)}

    def _sync_overlay(self):
        """标定字段 → 叠加层的位置/缩放（fit 用 scale+offset，crop 用 view）。"""
        if self.canvas is None:
            return
        pm = self._ov_item.pixmap()
        if pm.isNull() or pm.width() != self.canvas.shape[1]:
            self._ov_item.setPixmap(np_to_pixmap(self.canvas))
        self._ov_item._busy = True
        try:
            if self.mode == mm.MODE_CROP:
                # 面板 1:1 显示底图的一块：把整张底图挪到「那一块的左上角对上 (0,0)」
                self._ov_item.setScale(1.0)
                self._ov_item.setPos(-self.block[0], -self.block[1])
            else:
                self._ov_item.setScale(self.scale)
                self._ov_item.setPos(self.offset[0], self.offset[1])
        finally:
            self._ov_item._busy = False
        self._update_scene_rect()
        self._sync_widgets()
        self._refresh_judge()

    def _on_overlay_moved(self):
        """人拖了叠加层 → 反算回标定字段（拖动就是唯一的输入方式之一）。"""
        pos = self._ov_item.pos()
        if self.mode == mm.MODE_CROP:
            self.block = [-int(round(pos.x())), -int(round(pos.y()))]
        else:
            self.scale = float(self._ov_item.scale())
            self.offset = [int(round(pos.x())), int(round(pos.y()))]
        self._sync_widgets()
        self._refresh_judge()

    def _sync_widgets(self):
        """把 scale 回填到滑块/数字框（别让界面和实际几何不一致）。"""
        self.sp_scale.blockSignals(True)
        self.sp_scale.setValue(float(self.scale))
        self.sp_scale.blockSignals(False)
        self.sld.blockSignals(True)
        self.sld.setValue(max(SLIDER_MIN, min(SLIDER_MAX,
                                              int(round(self.scale * 1000)))))
        self.sld.blockSignals(False)
        # crop 是 1:1 显示，改缩放没有意义 —— 直接禁掉，免得人白调半天
        crop = (self.mode == mm.MODE_CROP)
        self.sld.setEnabled(not crop)
        self.sp_scale.setEnabled(not crop)

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
            self._say("还没收到帧 —— A 机上的「小地图推流」启动了吗？", bad=True)
            return
        now = time.perf_counter()
        if auto and (now - self._last_locate) < 0.25:
            return                        # 节流：定位比帧率慢时不至于把界面堵死
        self._last_locate = now

        # 「持续自动定位」是**跟踪**，不是重新搜一遍：
        # 全局小地图（fit）的完整搜索要试二十来个尺度，大底图上一次能跑几百毫秒
        # —— 每帧都那样跑，10 fps 的流会把界面线程堵死。跟踪只试当前尺度附近
        # 那一两个（人走两步不会让客户端换缩放比例），代价立刻降到一次匹配。
        scales = None
        if auto and self.mode == mm.MODE_FIT and self.scale > 0:
            scales = [round(self.scale * f, 4) for f in (0.98, 1.0, 1.02)]
            scales = [s for s in scales if 0.25 <= s <= 8.0]

        t0 = time.perf_counter()
        if scales is not None:
            loc = mm.locate_fit(self._frame, self.canvas, scales=scales)
            if loc is None:
                loc = mm.locate_fit(self._frame, self.canvas)  # 跟丢了就整体重搜一次
        else:
            loc = mm.locate(self._frame, self.canvas, self.mode)
        dt = time.perf_counter() - t0

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
        self._sync_overlay()
        # 顺带报一下这次定位花了多久：这一路以后要接进实时线程，耗时是硬指标
        #（B 机关键路径本来就没余量，见 docs/寻路设计.md §5 的性能一节）。
        self._say("定位成功　匹配分 %.3f　%.0f ms%s"
                  % (self.score, dt * 1000, move),
                  bad=(self.score < SCORE_WARN or dt > 0.25))

    def _say_fail(self):
        """没对上：把两种方式的分数都摆出来当线索（和命令行工具一致）。"""
        bits = []
        for tag, r in (("全局小地图", mm.locate_fit(self._frame, self.canvas)),
                       ("局部小地图", mm.locate_crop(self._frame, self.canvas))):
            bits.append("%s%s" % (tag, "没对上" if r is None else "%.3f" % r["score"]))
        extra = ""
        h, w = self._frame.shape[:2]
        cw, ch = self.canvas.shape[1], self.canvas.shape[0]
        if self.mode == mm.MODE_CROP and (w > cw or h > ch):
            extra = ("\n底图只有 %dx%d、面板 %dx%d 更大 —— 面板装不下整张底图，"
                     "所以不可能是「局部小地图」，多半是「全局小地图」"
                     "（去「路线识别」里换，这里不替你改）。" % (cw, ch, w, h))
        self._say("没对上 —— 可能：A 机框选区域不只是小地图 / 推的不是这张图 / "
                  "面板被游戏 UI 挡住。参考分：%s%s" % ("、".join(bits), extra),
                  bad=True)

    def _say(self, text, bad=False):
        """动作结果（定位/保存/抓帧）。**只写这一行**，不和连接状态抢地方。"""
        self.lbl_say.setText(text)
        self.lbl_say.setStyleSheet("color:%s;" % ("#c5221f" if bad else "#188038"))

    # ---------------- 状态 / 判据 ----------------

    def _refresh_status(self):
        cli = self.client
        n = getattr(cli, "n_recv", 0)
        if self._frame is None:
            err = getattr(cli, "err", "") or ""
            self.lbl_conn.setStyleSheet("color:#c5221f;")
            self.lbl_conn.setText("等帧中…（连 %s:%d）%s" % (self.host, self.port, err))
        elif not self.ck_live.isChecked():
            self.lbl_conn.setStyleSheet("color:#5f6368;")
            self.lbl_conn.setText("已抓一帧（实时画面关着）")
        else:
            self.lbl_conn.setStyleSheet("color:#188038;")
            self.lbl_conn.setText("收帧中 · %.1f 拍/秒 · 已收 %d 帧"
                                  % (getattr(cli, "fps", 0.0) or 0.0, n))

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
        ok = (abs(lx - b[0]) < 80 and abs(ly - b[1]) < 80
              and abs(rx - b[2]) < 80 and abs(ry - b[3]) < 80)
        self.lbl_judge.setText(
            "匹配分 %.3f　面板左上 → 世界 (%.0f, %.0f)　右下 → 世界 (%.0f, %.0f)　"
            "世界范围 (%.0f, %.0f) ~ (%.0f, %.0f)"
            % (self.score, lx, ly, rx, ry, b[0], b[1], b[2], b[3]))
        self.lbl_judge.setStyleSheet("color:%s;"
                                     % ("#188038" if ok else "#b06000"))

    # ---------------- 动作 ----------------

    def _on_grab(self):
        frame, _t = self.client.latest()
        if frame is None:
            self._say("还没收到帧 —— A 机「被控机部署台 → 小地图推流」启动了吗？"
                      "（当前 %s）" % (getattr(self.client, "err", "") or "没连上"),
                      bad=True)
            return
        self._frame = frame
        self._apply_frame(frame)
        self._say("已抓一帧 %dx%d" % (frame.shape[1], frame.shape[0]))

    def _draw_overlay(self):
        """把「当前是叠加图哪一块」画到 <id>_overlay.png 上（和命令行同一函数）。"""
        if self.canvas is None or self._frame is None:
            self._say("先抓一帧再画", bad=True)
            return
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
            self._say("还没有 %s_overlay.png —— 先在「路线识别」里点「生成地形图」"
                      % self.map_id, bad=True)
        else:
            self._say("已写出 %s（框 = 当前小地图对应的那一块）" % p)

    def _on_save(self):
        if self.canvas is None:
            return
        if self.score < SCORE_WARN:
            r = QMessageBox.question(
                self, "匹配分偏低",
                "当前匹配分 %.3f（低于 %.2f）。\n\n"
                "分数低说明面板和底图对不上 —— 存下去的话，寻路会按这个错位的\n"
                "换算算世界坐标（表现是「人在这块平台，程序以为在另一块」）。\n\n"
                "建议先点「自动定位」；实在匹配不上（面板被 UI 挡住等）再用\n"
                "手动拖动对齐 —— 但那样只有你的眼睛能保证它对。\n\n"
                "还是要保存吗？" % (self.score, SCORE_WARN),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        cal = mapdata.load_calib(self.map_id) or {}
        # 只更新量出来的几何 + 方式：显示方式仍算「你在界面里选的」
        cal.update(self.calib())
        cal["picked_by"] = "gui"
        mapdata.save_calib(self.map_id, cal)
        self.saved = True
        self._say("已保存标定 → %s" % mapdata.calib_path(self.map_id))

    # ---------------- 键盘微调 ----------------

    def keyPressEvent(self, e):
        """方向键 1 像素、Shift+方向键 5 像素地挪**叠加层**（目测对齐最后那一两像素）。

        方向语义和拖动一致：按左 = 叠加层往左走（于是「看到的是底图哪一块」右移）。
        别按「数字往哪边走」去理解 —— 人对着画面按，手感才是对的。
        """
        step = 5 if (e.modifiers() & Qt.ShiftModifier) else 1
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
