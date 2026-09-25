"""实时预览面板：收流 → YOLO 推理 → 画面实时显示。

设计要点见 live_thread.py 顶部说明。这一层只负责：
    · 收参数、起停线程
    · 把线程推来的帧画到界面上
    · 显示三个分开的速度指标
"""

from PyQt5.QtCore import QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QSizePolicy,
                             QVBoxLayout, QWidget)

import time

import numpy as np

from decision.agent import settings
from gui.live_thread import LiveThread
from gui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox
from tools.config import ROOT, get, load_live, update_live


def _bgr_to_pixmap(img):
    """numpy BGR → QPixmap。

    三个坑（都是真实踩过的）：
      1. 必须 copy()：QImage 只引用那段内存不持有所有权，numpy 数组一回收，
         界面拿到野指针 → 花屏或崩溃。
      2. to_ndarray(bgr24) 返回的帧带行 padding（非连续，stride 比 3*w 大），
         QImage 按 3*w 读会整幅错位 —— 先 ascontiguousarray 去掉 padding。
      3. numpy 的 .data 在新版返回 memoryview，PyQt5 的 QImage 不认它，
         要用 tobytes() 拿 bytes 再构造。
    """
    img = np.ascontiguousarray(img)
    h, w, ch = img.shape
    qimg = QImage(img.tobytes(), w, h, img.strides[0], QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class LiveFrameRegionClient:
    """把「实时画面里框出来的一块」喂给标定弹窗（代替 A 机的小地图推流）。

    **为什么要有它**：小地图标定要的只是"一张面板画面"，从哪来其实无所谓 ——
    以前只能等 A 机那条独立推流（清晰，但要多推一路、多一份 A 机开销）。这个
    适配器直接从**实时预览那一帧**上裁一块（工作台「小地图来源 = 从实时画面框选」，
    实验做法）。接口和 `perception.minimap.MiniMapClient` 一致，所以标定弹窗
    **一行都不用改** —— 换来源 = 换喂帧的人。

    ⚠ 为什么标"实验"：实时画面是 H.264 压过的，小地图黄点只有几个像素，能不能
    稳定认出玩家点**还没实测**（面板 ↔ 底图的匹配问题不大：底图本来就比面板小，
    匹配是在粗结构上做的）。判据还是标定弹窗里那个匹配分 + 以后黄点识别的实测。
    """

    def __init__(self, panel, region):
        self.panel = panel
        self.region = [int(v) for v in region]
        self.n_recv = 0
        self.fps = 0.0
        self.err = ""
        self.connected = True
        # 给标定弹窗的文字：换来源之后，"等帧中…（连 A机:5003）"那类提示会说错
        self.label = "实时画面框选 %d×%d" % (self.region[2], self.region[3])
        self.wait_hint = ("还没有实时画面 —— 先到「实时」页点开始预览，再回来标定")
        self.hint = "实时画面里那块区域没取到（先到「实时」页开始预览）"
        self._src = None        # 上一帧（身份比较：同一帧不重复裁）
        self._crop = None
        self._t = 0.0
        self._t0 = time.time()
        self._n0 = 0

    def start(self):
        return self

    def stop(self):
        pass                    # 借的是实时预览那一路，不能停它（和弹窗的约定一致）

    def latest(self, clear=False):
        """→ (那一块的 BGR 图, 时间戳)；取不到给 (None, 0.0)，原因写进 `err`。

        **同一帧返回同一个对象**：标定弹窗靠 `frame is not self._frame` 判断
        "来了新帧"，每次重建对象会让它以为帧在不停刷新。
        """
        img = None if self.panel is None else self.panel.current_frame()
        if img is None:
            # 报错要写"怎么修"（UI规范 §6）：不然人只会看到"没有画面"然后卡住
            self.err = "还没有实时画面 —— 先到「实时」页点「开始」预览"
            return None, 0.0
        x, y, w, h = self.region
        H, W = img.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
            # 画面尺寸变了（换分辨率 / 改推流参数）→ 说清楚，别给一块错位的图
            self.err = ("框选的区域 %s 超出当前画面 %dx%d —— 重新框一次"
                        % (self.region, W, H))
            return None, 0.0
        if img is not self._src:
            self._src = img
            self._crop = np.ascontiguousarray(img[y:y + h, x:x + w])
            self._t = time.time()
            self.n_recv += 1
            span = self._t - self._t0
            if span >= 1.0:
                self.fps = (self.n_recv - self._n0) / span
                self._t0, self._n0 = self._t, self.n_recv
            self.err = ""
        return self._crop, self._t


class LivePanel(QWidget):
    potions_ready = pyqtSignal(float, float)   # (hp, mp) 比例，转发给决策参数页

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.thread = None
        self._last_pix = None
        self._last_bgr = None   # 最近一帧画面（BGR），供 HP/MP 条在画面上框选
        self._rect = None       # 本地窗口模式下框选的区域 (x, y, w, h)
        # 绘制合并（见 _on_frame）：只保留最新一帧，渲染慢时丢中间帧而不是排队
        self._pending = None
        self._render_pending = False
        self._disp_drawn = 0      # 真画出来的帧数
        self._disp_merged = 0     # 还没画就又来新帧 → 被合并掉的帧数
        self._disp_skipped = 0    # 面板不可见 → 整帧不画（省主线程）
        self._draw_ms = 0.0       # 最近一次绘制耗时（含 QImage/QPixmap + 缩放）
        #: 挂在实时画面上的「地形叠加图」那一层：None = 不画。
        #: 由「路线识别 → 小地图定位」算好后塞进来（set_minimap_overlay），
        #: **纯显示层** —— 帧数据一个字节都不改，见那个方法的说明。
        self._ov_minimap = None
        #: 小地图定位（S3）用的三样：地图 id / 来源 / 面板在画面里的矩形。
        #: 由「路线识别」面板推过来（set_mmap），随线程参数一起带进实时线程。
        self._mmap_mid = ""
        self._mmap_src = "stream"
        self._mmap_crop = None
        self._mmap_track = None     # 黄点容差（界面上那排，键名见 mm.TRACK_KEYS）
        self._build()

    # ---------------- 界面 ----------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- 控制条 ----
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self.btn_start = QPushButton("▶ 开始")
        self.btn_start.clicked.connect(self.start)
        bar.addWidget(self.btn_start)

        self.btn_infer = QPushButton("▶ 开始推理")
        self.btn_infer.setEnabled(False)
        self.btn_infer.setToolTip("先点「开始」收画面，再点这里开启 YOLO 识别与决策")
        self.btn_infer.clicked.connect(self.start_infer)
        bar.addWidget(self.btn_infer)

        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        bar.addWidget(self.btn_stop)

        # 流状态灯：等待推流 / 已连接 / 无流
        self.lbl_stream = QLabel("—")
        self.lbl_stream.setStyleSheet("color:#80868b; font-weight:600;")
        bar.addWidget(self.lbl_stream)

        bar.addSpacing(10)
        bar.addWidget(QLabel("来源"))
        self.cmb_source = NoWheelComboBox()
        self.cmb_source.addItem("收流", "stream")
        self.cmb_source.addItem("本地窗口", "window")
        self.cmb_source.currentIndexChanged.connect(self._on_source)
        bar.addWidget(self.cmb_source)

        # 收流：UDP/RTSP 地址
        self.ed_url = QLineEdit(get("stream", "url", "udp://0.0.0.0:5000"))
        self.ed_url.setMinimumWidth(220)
        bar.addWidget(self.ed_url, 1)

        # 本地窗口：全屏框选区域
        self.btn_pick_rect = QPushButton("框选区域")
        self.btn_pick_rect.setToolTip("全屏拖拽框选要识别的屏幕区域")
        self.btn_pick_rect.clicked.connect(self._pick_rect)
        bar.addWidget(self.btn_pick_rect)

        self.lbl_rect = QLabel("（未选）")
        self.lbl_rect.setStyleSheet("color:#80868b;")
        bar.addWidget(self.lbl_rect, 1)

        root.addLayout(bar)

        # ---- 参数行 ----
        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel("权重"))
        self.ed_weights = QLineEdit()
        self.ed_weights.setReadOnly(True)
        self.ed_weights.setMinimumWidth(200)
        row.addWidget(self.ed_weights, 1)

        _live = load_live()

        row.addWidget(QLabel("怪conf"))
        self.sp_conf_mob = NoWheelDoubleSpinBox()
        self.sp_conf_mob.setRange(0.0, 1.0)
        self.sp_conf_mob.setDecimals(2)
        self.sp_conf_mob.setSingleStep(0.05)
        self.sp_conf_mob.setValue(float(_live.get("conf_mob", 0.30)))
        self.sp_conf_mob.valueChanged.connect(self._save_live_params)
        row.addWidget(self.sp_conf_mob)

        row.addWidget(QLabel("玩家conf"))
        self.sp_conf_player = NoWheelDoubleSpinBox()
        self.sp_conf_player.setRange(0.0, 1.0)
        self.sp_conf_player.setDecimals(2)
        self.sp_conf_player.setSingleStep(0.05)
        self.sp_conf_player.setValue(float(_live.get("conf_player", 0.50)))
        self.sp_conf_player.valueChanged.connect(self._save_live_params)
        row.addWidget(self.sp_conf_player)

        row.addWidget(QLabel("imgsz"))
        self.sp_imgsz = NoWheelSpinBox()
        self.sp_imgsz.setRange(320, 2048)
        self.sp_imgsz.setValue(int(_live.get("imgsz", 960)))
        row.addWidget(self.sp_imgsz)

        row.addWidget(QLabel("设备"))
        self.ed_device = QLineEdit(str(_live.get("device", "0")))
        self.ed_device.setFixedWidth(50)
        row.addWidget(self.ed_device)

        # 本地窗口：抓帧频率
        self.lbl_capfps = QLabel("抓帧fps")
        self.sp_capfps = NoWheelSpinBox()
        self.sp_capfps.setRange(1, 60)
        self.sp_capfps.setValue(int(_live.get("capture_fps", 60)))
        self.sp_capfps.setToolTip("本地窗口每秒抓几帧去推理。\n窗口画面本身可能只有 60fps，抓太高浪费，\n15~30 对角色/怪物识别通常足够。")
        row.addWidget(self.lbl_capfps)
        row.addWidget(self.sp_capfps)

        self.ck_draw = QCheckBox("画框")
        self.ck_draw.setChecked(bool(_live.get("draw", True)))
        row.addWidget(self.ck_draw)

        self.ck_probe = QCheckBox("延迟探针")
        self.ck_probe.setChecked(bool(get("probe", "enabled", True)))
        self.ck_probe.setToolTip(
            "解码 A 机屏幕上的时间码，测真实端到端延迟。\n"
            "需要：A 机跑 python -m tools.probe_gen\n"
            "      B 机跑过一次 python -m tools.clock_sync --host 192.168.1.8 --save\n\n"
            "不启用时延迟显示为 ——，因为 pts 推算只能反映网络抖动，测不出真实延迟。")
        row.addWidget(self.ck_probe)

        self.ck_probe_box = QCheckBox("探针框")
        self.ck_probe_box.setChecked(True)
        self.ck_probe_box.setToolTip(
            "在实时画面上把**采样几何**画出来（框 + 每个采样点），不用去命令行看：\n\n"
            "  · 框的颜色就是判据的结论 ——\n"
            "      绿 = 几何可用（时间戳单调、值也合理）；\n"
            "      红 = 几何可疑（时间戳乱跳 / 整片挪位）；\n"
            "      黄 = 判据还在攒样本；灰 = 探针没启用。\n"
            "  · 每个十字是该块的采样点，颜色跟着判成 0/1 变（黄=白块、蓝=黑块）；\n"
            "      变紫 = 这个点压在黑白之间，也就是**没对准**。\n"
            "  · 框角上有两个圆圈 = 头两个固定标记块（白、黑）。\n\n"
            "框没括住整条码带、或尾部十字开始发紫，就是几何不对 —— 点「框选探针」重框。")
        row.addWidget(self.ck_probe_box)

        # 探针几何：人工框选（秒级、当场验证）—— 「调完探针大小，收流位置全靠猜」的解药。
        # 与 HP/MP 条同一套框选交互（在实时画面上拖矩形），但换算不同：
        # 方块带是 n = 2+bits 个等大等距方块，头两个是固定标记（白、黑），
        # 所以框完能**当场解码验证**；解不出就不保存（宁可不改，也别改坏）。
        self.btn_probe_box = QPushButton("框选探针")
        self.btn_probe_box.setToolTip(
            "在实时画面上框住整条时间码方块带（含最左边那个常亮的白块）。\n"
            "框完当场解码验证：解得出时间码才保存。\n"
            "存进 config/probe_calib.json，存的是**画面比例** —— 换分辨率/窗口不用重标；\n"
            "保存后每秒自动生效，不用重开预览。解不出会提示检查什么，且不修改任何配置。")
        self.btn_probe_box.clicked.connect(self._pick_probe)
        row.addWidget(self.btn_probe_box)

        row.addStretch(1)
        root.addLayout(row)

        # 探针几何一行：现在用的是哪套（人工标定 / 配置值）、具体数值是多少。
        # 有这一行就不用再靠猜 —— 改完立刻看得见。
        self.lbl_probe = QLabel()
        self.lbl_probe.setStyleSheet("color:#5f6368;")
        self._refresh_probe_label()
        root.addWidget(self.lbl_probe)

        # ---- 画面 ----
        self.view = QLabel("（点「开始」后这里显示实时画面）")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumHeight(300)
        # 忽略内容（pixmap）的 sizeHint：否则每帧 setPixmap 会改变 QLabel 的
        # 建议尺寸，传导到 QSplitter，主视区宽度跟着抖、把右侧卡片压来压去。
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setStyleSheet(
            "background:#202124; color:#9aa0a6; border:1px solid #dadce0;")
        root.addWidget(self.view, 1)

        # ---- 统计条 ----
        self.lbl_stats = QLabel("未开始")
        self.lbl_stats.setStyleSheet("color:#5f6368;")
        root.addWidget(self.lbl_stats)

        self.lbl_hint = QLabel(
            "输入 fps = 来源给到的速度（收流=链路，窗口=抓帧频率）｜ 处理 fps = 我们实际跑完的帧率（低于输入就会丢帧）｜ "
            "丢帧 = 为了不积压延迟而主动丢掉的帧数（正常，实时系统宁可丢帧也不排队）｜\n"
            "推理 ms = 模型耗时，要跑 60fps 得 ≤16ms ｜ 显示 fps = 界面刷新，故意低于推理，不参与性能判断")
        self.lbl_hint.setStyleSheet("color:#80868b;")
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint)

        self._on_source()     # 初始可见性：默认收流模式

    # ---------------- 来源切换 ----------------

    def _on_source(self, _idx=None):
        is_stream = self.cmb_source.currentData() == "stream"
        is_win = not is_stream
        self.ed_url.setVisible(is_stream)
        self.btn_pick_rect.setVisible(is_win)
        self.lbl_rect.setVisible(is_win)
        self.lbl_capfps.setVisible(is_win)
        self.sp_capfps.setVisible(is_win)

    # ---------------- 探针几何（人工框选标定） ----------------

    def _probe_label_text(self):
        """当前探针几何 + 来源，一行说完。

        来源有三种，**项目标定优先**（标定跟项目走，见 pick_calib 的说明）。
        """
        from tools import probe_codec
        bits = int(get("probe", "bits", 40))
        cal, src = probe_codec.pick_calib(settings.probe_calib)
        if cal:
            if self._last_bgr is not None:
                h, w = self._last_bgr.shape[:2]
                x, y, cell, gap = probe_codec.calib_to_px(cal, (h, w))
                return ("探针：%s  x=%.0f y=%.0f cell=%.1f gap=%.1f bits=%d"
                        "（按当前画面 %d×%d 换算）" % (src, x, y, cell, gap,
                                                     bits, w, h))
            return ("探针：%s（画面比例 x=%.4f y=%.4f cell=%.4fW gap=%.4fW bits=%d）"
                    % (src, cal["x_ratio"], cal["y_ratio"], cal["cell_ratio"],
                       cal["gap_ratio"], bits))
        return ("探针：用 link.yaml 的配置值  x=%s y=%s cell=%s gap=%s bits=%d"
                "　（位置不对就点「框选探针」）"
                % (get("probe", "x", 100), get("probe", "y", 8),
                   get("probe", "cell", 16), get("probe", "gap", 2), bits))

    def _refresh_probe_label(self, extra=""):
        self.lbl_probe.setText(self._probe_label_text() + extra)

    # ---------------- 在实时画面上画采样框（判"框得对不对"的眼睛） ----------------

    def _probe_geo_px(self, shape):
        """当前生效的探针几何（像素）→ (x, y, cell, gap, bits, 来源)，拿不到给 None。

        **与实时回路、框选、命令行工具同一个来源**（项目标定 → 全局 → link.yaml）：
        屏幕上画的框必须就是真正采样的那几个点，否则人看到的和机器采的会悄悄错开。
        """
        from tools import probe_codec

        bits = int(get("probe", "bits", 40))
        cal, src = probe_codec.pick_calib(settings.probe_calib)
        if cal:
            x, y, cell, gap = probe_codec.calib_to_px(cal, shape)
            return x, y, cell, gap, bits, src
        return (float(get("probe", "x", 100)), float(get("probe", "y", 8)),
                float(get("probe", "cell", 16)), float(get("probe", "gap", 2)),
                bits, "link.yaml")

    def _probe_box_color(self):
        """框该用什么颜色 → (BGR, 一句话)。**颜色就是判据的结论**。

        实测背景（2026-09-25）：几何和屏幕上真正画的码对不上时，延迟读数是一坨乱数，
        而界面上一切正常 —— 人能一眼看出的只有"绿框没框全"。所以把颜色直接绑到判据上。
        """
        s = getattr(self, "_last_stats", None) or {}
        if not s.get("probe_on", True):
            return (128, 128, 128), "探针未启用"
        n = int(s.get("probe_samples") or 0)
        if n < 6:
            return (0, 200, 255), "判据采样中（%d 帧）" % n
        if s.get("probe_mono") is False and int(s.get("probe_jumpy") or 0) > 0:
            return (60, 60, 240), ("几何可疑：时间戳乱跳 %d 帧"
                                   % int(s.get("probe_jumpy") or 0))
        if s.get("probe_value_ok") is False:
            return (60, 60, 240), ("几何可疑：解出的时刻差了 %.0f 秒"
                                   % abs(s.get("probe_value_off_s") or 0))
        d = s.get("delay_ms")
        if d is None:
            return (0, 200, 255), "判据 OK，但算不出延迟（多半是时钟偏移）"
        return (80, 220, 80), "几何 OK：延迟 %.0f ms" % d

    def _draw_probe_overlay(self, bgr):
        """在**副本**上画采样框 + 每个采样点 → 返回它（颜色见 _probe_box_color）。

        **为什么必须画在副本上**：`_on_frame` 里 `_pending` 与 `_last_bgr` 是同一个
        数组，而 `current_frame()` 把它交给探针框选 / HP 条框选用 —— 在原图上画框线，
        线会落在方块上改变灰度均值，等于**自己污染自己的采样**。
        """
        import cv2

        from tools import probe_codec

        h, w = bgr.shape[:2]
        geo = self._probe_geo_px((h, w))
        if geo is None:
            return bgr
        # **自己拷一份再画**（不依赖调用方记得 copy）：`_on_frame` 里 `_pending` 与
        # `_last_bgr` 是同一个数组，而 `current_frame()` 要把它交给框选用 ——
        # 在它上面画框线，线落进方块就会改变灰度均值，自己污染自己的采样。
        # 自检 (`t_probe_box_overlay`) 直接钉了"输入数组一个像素都不许动"。
        bgr = bgr.copy()
        x, y, cell, gap, bits, _src = geo
        color, why = self._probe_box_color()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        seq, means = probe_codec.read_cells(gray, x, y, cell, gap, bits)
        n = 2 + bits
        cv2.rectangle(bgr, (int(round(x)) - 2, int(round(y)) - 2),
                      (int(round(x + n * (cell + gap))) + 2,
                       int(round(y + cell)) + 2), color, 2)
        for i in range(n):
            cx = int(round(x + i * (cell + gap) + cell / 2.0))
            cy = int(round(y + cell / 2.0))
            if means[i] < 0:
                c = (0, 0, 255)                       # 采样点跑出画面了
            elif probe_codec.AMBIG_LO <= means[i] <= probe_codec.AMBIG_HI:
                c = (255, 0, 255)                     # 压在黑白之间 = 没对准
            else:
                c = (0, 255, 255) if seq[i] else (255, 120, 0)
            cv2.drawMarker(bgr, (cx, cy), c, cv2.MARKER_CROSS, 7, 1)
            if i < 2:                                 # 头两个是固定标记（白、黑）
                cv2.circle(bgr, (cx, cy), int(cell / 2), (255, 255, 0), 1)
        cv2.putText(bgr, why, (max(6, int(x) - 2), max(18, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return bgr

    def _arm_probe_verify(self):
        """布置「存后自检」：2.5 秒后用实时统计的**增量**回头看通没通。

        存下来 ≠ 生效：框选/量出的几何只保证"当时那一帧解得出来"，实时是每帧
        在当前画面上采样。所以这里记下当前的计数，稍后比对（见 _verify_calib_if_due）。
        """
        st = getattr(self, "_last_stats", None) or {}
        self._verify = [time.monotonic() + 2.5,
                        int(st.get("probe_ok") or 0),
                        int(st.get("probe_invalid") or 0)
                        + int(st.get("probe_miss") or 0)]

    def _verify_geo_live(self, geo, bits, seconds=1.5, want=8):
        """拿实时流跑 ~0.8 秒，判这份几何解出的时间戳是否**单调** → (可信?, 说明)。

        **为什么不能只"再解一帧"**：单帧解出一个"接近此刻"的值**证明不了几何对**
        —— 40 位码的低位是快变化的毫秒，错位采样只把低位采错，凑出一个仍落在
        ±10 分钟窗口里的值完全可能（实测踩过：比实际画的小 2.8px/块的几何，42 块
        累积偏 118px、低位全错，解出的值"看着合法"，延迟被报成一坨乱数 p95 4.7s）。

        单调性是**时间轴上的性质**：时间戳每帧只该往前走一小步，错位必然破坏它。
        所以这里真的等几百毫秒收几帧 —— 短到人感觉不出来，但足够识破"低位采错"。
        帧从 `current_frame()` 取（预览在跑，它每帧都换）。
        """
        from PyQt5.QtWidgets import QApplication

        import cv2
        from tools import probe_codec as pc

        # 步长上限的参考周期**故意放宽到 ≥66ms**（≈15fps）：这里收的是**显示帧**，
        # 显示层掉一帧就是两倍步长（实测见过 195ms 的一跳）——那不等于错位。
        # 这道闸是"存之前的一个粗筛"，细判留给实时的连续判据（24 帧窗口）。
        period = max(1000.0 / max(1.0, float((getattr(self, "_last_stats", None) or {})
                                             .get("show_fps") or 30.0)),
                     1000.0 / 15.0)
        vd = pc.Verdict()
        app = QApplication.instance()
        t0 = time.monotonic()
        seen = set()
        while time.monotonic() - t0 < seconds and len(vd.ts_hist) < want:
            if app is not None:
                app.processEvents()      # 让收流线程推来的新帧进 _on_frame
            f = self.current_frame()
            if f is None or id(f) in seen:
                time.sleep(0.005)
                continue
            seen.add(id(f))
            try:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                ts = pc.decode_ms(gray, geo["x"], geo["y"], geo["cell"],
                                  geo["gap"], bits)
            except Exception:
                ts = None
            if ts is not None:
                vd.add(ts, period)
        if len(vd.ts_hist) < 3:
            # 帧不够（预览没在跑 / 刚起）：**不下结论、也不拦**，如实说一句
            return True, "只收到 %d 帧，判据没跑起来" % len(vd.ts_hist)
        return vd.summary()

    def _pick_probe(self):
        """在实时画面上框选探针方块带 → 当场解码验证 → 通过才保存标定。

        **为什么必须验证**：这是 40 位码，差一个方块宽度就整个错位，而错位采样
        也能解出一个「合法」的 40 位数（看着完全正常）。所以判据是「解出来的时刻
        接近现在」（见 probe_codec.ts_plausible）。解不出就什么都不改。

        但**光有那一条不够**（见 `_verify_geo_live`）：保存前还要拿实时流跑 0.8 秒
        看时间戳**单不单调**，不单调就默认不保存 —— 这是 2026-09-25 那半天查错
        换来的闸：错的几何一旦静默存进项目，工作台从此按它采样，延迟读数全是假的
        而界面上一切正常。
        """
        from gui.region_selector import select_region_on_image
        from tools import probe_codec

        frame = self.current_frame()
        if frame is None:
            QMessageBox.information(
                self, "提示",
                "请先在「实时」页开始预览、看到画面（含 A 机的探针方块带）之后再框选。")
            return
        rect = select_region_on_image(frame, self)
        if rect is None:
            return

        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bits = int(get("probe", "bits", 40))
        # 没对过时的机器上，「接近此刻」这条判据用不了（差多少全看钟差），
        # 这时退一步：能解出合法码就先标上，并明确告诉用户去对时。
        offset_ok = (ROOT / "config" / "clock_offset.txt").exists()
        # 两个 hint 都来自 link.yaml：框得偏小时，用这份**已知能出数**的几何兜底。
        # 判据不因此放宽（照样要解出合理时间码），所以只是更宽容、不会存坏值。
        geo = probe_codec.solve_from_rect(gray, rect, bits, strict=offset_ok,
                                          gap_hint=get("probe", "gap", 2),
                                          cell_hint=get("probe", "cell", 20))
        if geo is None:
            QMessageBox.warning(
                self, "没解出时间码",
                "框选里没能解出时间码，所以**没有保存**（宁可不改，也别改坏）。\n\n"
                "按顺序检查：\n"
                "  1. A 机是不是在跑  python -m tools.probe_gen\n"
                "  2. 要框住**整条方块带**：包括最左边那个常亮的白块，"
                "和它后面那个常黑的块（头两个是编码的起始标记）\n"
                "  3. 方块要够大：cell ≥ 16px，而且**框的高度要正好贴住方块高度**\n"
                "     （框紧了 cell 会小一截，再往下每个方块都错位 —— 40 位码必错）\n"
                "  4. bits 要和 A 机一致（现在是 %d）\n"
                "  5. 画面里那条带子如果发虚/被 UI 压住，先在 A 机挪开探针再标" % bits)
            return

        # ★ **单调性闸**：解得出"接近此刻"的值不等于几何对（低位采错照样能凑）。
        # 所以保存前拿实时流跑 0.8 秒看时间戳单不单调，不单调就默认**不保存**。
        mono_ok, mono_why = self._verify_geo_live(geo, bits)
        if not mono_ok:
            r = QMessageBox.question(
                self, "这份几何可能不对",
                "框选能解出时间码，但**实时这一小会儿解出的时间戳在乱跳**：\n\n"
                "    %s\n\n"
                "时间戳本该每帧只往前走一小步。乱跳说明采样点没落在方块中心、"
                "低位被采错了 ——\n这种错**光看能不能解出是看不出来的**，而它会让"
                "「端到端延迟」变成一个假数。\n\n"
                "「否」= 不保存，回去重框（推荐）\n"
                "「是」= 仍然保存（我知道自己在做什么）\n\n"
                "重框要点：高度**贴住方块**、宽度**量到整条带的外缘**。\n"
                "也可以改用  python -m tools.probe_tune  手工推框 —— 它按同一个判据"
                "当场告诉你\n对没对，判据没过还不让保存。" % mono_why,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                self._refresh_probe_label("　⨯ 已取消保存（几何没通过单调性判据）")
                return

        dx, dy, dcell = geo["snap"]
        # **存进项目**（settings 按项目持久化，和 HP/MP 条同一套机制）：
        # 换项目就换一份标定；没打开项目时只改内存（会被项目里的值覆盖）。
        settings.probe_calib = probe_codec.calib_from_geo(geo, frame.shape, bits)
        settings.save()
        # 存了不等于生效：过 2.5 秒拿实时统计的**增量**回头验一次
        # （见 _verify_calib_if_due；"那一帧能解出"和"每帧都能解出"是两回事）。
        self._arm_probe_verify()
        h, w = frame.shape[:2]
        # 修正量要写清楚：框小了会被「已知好值」救回来，但那不等于没问题 ——
        # 40 位码靠**节距**对齐，框选高度差几像素就该重框（这里只是兜底）。
        tail = ""
        if dx or dy or dcell:
            tail = "　自动对齐修正 dx=%+d dy=%+d dcell=%+d" % (dx, dy, dcell)
            if abs(dcell) >= 2:
                tail += ("（框选高度差 %dpx，已按 link.yaml 的 cell=%.0f 对齐 ——\n"
                         "建议重框：高度贴住方块）" % (abs(dcell), geo["cell"]))
        if geo["plausible"]:
            extra = "　✓ 解出 ts=%d ms，与本地时刻相符" % geo["ts"]
            head = "✓ 标定成功"
            body = ("解出的时间码是 %d ms（A 机当天时刻），与本地时刻相符 —— 对准了。"
                    % geo["ts"])
        else:
            extra = "　⚠ 解出 ts=%d ms，但对不上本地时刻" % geo["ts"]
            head = "✓ 已标出位置（还没对时）"
            body = ("在框选附近解出了时间码 %d ms，几何应该是对的；但它和本地时刻对不上，"
                    "说明**双机还没对时**。\n\n"
                    "**注意**：在这种情况下，上面那行「端到端延迟」仍会显示解不出 ——\n"
                    "因为「解出时间码」和「算出延迟」是两件事：后者还要双机时钟对齐在\n"
                    "1 分钟以内。跑一次对时就都通了：\n"
                    "    python -m tools.clock_sync --host <A机IP> --save"
                    % geo["ts"])
        self._refresh_probe_label(extra)
        QMessageBox.information(
            self, head,
            body + "\n\n"
            "几何：x=%.1f y=%.1f cell=%.2f gap=%.2f bits=%d%s\n"
            "画面 %d×%d；存的是比例，换分辨率不用重标。\n"
            "每秒自动生效，不用重开预览。"
            % (geo["x"], geo["y"], geo["cell"], geo["gap"], bits, tail, w, h))

    def _pick_rect(self):
        from gui.region_selector import select_region
        rect = select_region(self)
        if rect is None:
            return
        self._rect = rect
        self.lbl_rect.setText("%d,%d  %d×%d" % rect)

    # ---------------- 项目 ----------------

    def set_mmap(self, map_id=None, src=None, crop=None, track=None):
        """把「小地图定位」要的东西转给实时线程（切项目/换来源/重框/改容差时叫它）。

        实时线程在跑就当场生效（`set_mmap` 只写几个字段，主回路每拍现读），
        没跑就等下一次 start() 由参数带进去 —— 两条路都在下面的 start() 里对齐。
        """
        if map_id is not None:
            self._mmap_mid = str(map_id)
        if src is not None:
            self._mmap_src = src
        if crop is not None:
            self._mmap_crop = [int(v) for v in crop]
        if track is not None:
            self._mmap_track = dict(track)
        if self.thread is not None:
            self.thread.set_mmap(map_id=map_id, src=src, crop=crop,
                                 track=track)

    def bind(self, project):
        """绑定项目：自动找最新的模型权重。"""
        self.project = project
        # 地图跟着项目走：小地图定位要用它去读地形与标定（见 set_mmap）
        self.set_mmap(map_id=((project.get("map_id") or "").strip()
                              if project is not None else ""))
        if self.thread is not None:
            self.stop()

        if project is None:
            self.ed_weights.setText("")
            return

        # 按修改时间取最新的模型（detect_v1 比旧的 mob_v1 新）。
        # 不能按文件名排序：detect_v1 < mob_v1，got[-1] 会选中旧的单类模型。
        got = sorted(project.dir_of("models").glob("*.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        if not got:
            got = sorted(project.dir_of("runs").glob("**/weights/best.pt"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        self.ed_weights.setText(str(got[0]) if got else "")

    @staticmethod
    def _load_offset_ms():
        """读双机时钟偏移。没有它探针算不出延迟。"""
        p = ROOT / "config" / "clock_offset.txt"
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except Exception:
            return 0.0

    # ---------------- 起停 ----------------

    def _save_live_params(self, *_):
        """conf 改动实时写 config，让运行中的实时预览立即生效（其余参数下次启动生效）。"""
        try:
            # 走 update_live（不是 save_live）：后者是整文件覆盖，会把别处写的键
            # （perf_log / perf_keepalive）一起抹掉 —— 现象是「设置里明明开着，
            # 重启后文件里没了、选项变回默认」，很难归因。
            update_live(
                conf_mob=self.sp_conf_mob.value(),
                conf_player=self.sp_conf_player.value(),
                imgsz=self.sp_imgsz.value(),
                device=self.ed_device.text().strip() or "0",
                capture_fps=self.sp_capfps.value(),
                draw=self.ck_draw.isChecked(),
            )
        except Exception:
            pass

    def start(self):
        # 旧线程可能还在后台退出（已 stop），直接丢弃引用重开
        if self.thread is not None:
            self.thread.stop()
            self.thread = None

        w = self.ed_weights.text().strip()
        if not w:
            self.lbl_stats.setText("没有可用的模型权重 —— 请先完成 ⑦ 训练")
            return

        source = self.cmb_source.currentData()

        # 玩家也走 YOLO（每个玩家独占一个类，当前固定 class 0）。
        # 多玩家时把 player_id → class 映射做成可配置项（见 live_thread.PLAYER_CLASS_MAP）。
        pid = ""
        if self.project is not None:
            pid = self.project.get("player_id") or ""

        # 保存实时参数，下次开 GUI 不用再调。
        # 走 update_live：其余键（perf_log / perf_keepalive）原样留着 ——
        # 以前这里整文件覆盖，还得手工把 perf_log 抄一遍补回来，那正是
        # 「整文件覆盖丢键」这个坑的症状。
        update_live(
            conf_mob=self.sp_conf_mob.value(),
            conf_player=self.sp_conf_player.value(),
            imgsz=self.sp_imgsz.value(),
            device=self.ed_device.text().strip() or "0",
            capture_fps=self.sp_capfps.value(),
            draw=self.ck_draw.isChecked(),
        )

        common = {
            "weights": w,
            "conf_mob": self.sp_conf_mob.value(),
            "conf_player": self.sp_conf_player.value(),
            "imgsz": self.sp_imgsz.value(),
            "device": self.ed_device.text().strip() or "0",
            "draw": self.ck_draw.isChecked(),
            "perf_log": bool(load_live().get("perf_log", True)),
            "show_fps": 30.0,
            "player_id": pid,
            # 小地图定位（S3）：把玩家世界坐标写进 WorldState（见 live_thread）
            "mmap_map_id": self._mmap_mid,
            "mmap_src": self._mmap_src,
            "mmap_crop": self._mmap_crop,
            "mmap_track": self._mmap_track,
        }

        if source == "window":
            rect = self._rect
            if not rect:
                self.lbl_stats.setText("请先点「框选区域」拖拽选择屏幕区域")
                return
            self.thread = LiveThread({
                **common,
                "source": "window",
                "rect": rect,
                "capture_fps": self.sp_capfps.value(),
                "probe": False,      # 本地窗口没有跨机传输延迟
            })
        else:
            url = self.ed_url.text().strip()
            if not url:
                self.lbl_stats.setText("请填写收流地址")
                return
            self.thread = LiveThread({
                **common,
                "source": "stream",
                "url": url,
                "format": get("stream", "format", None),
                "probe": self.ck_probe.isChecked(),
                "probe_x": get("probe", "x", 100),
                "probe_y": get("probe", "y", 8),
                "probe_cell": get("probe", "cell", 16),
                "probe_gap": get("probe", "gap", 2),
                "probe_bits": get("probe", "bits", 40),
                "clock_offset_ms": self._load_offset_ms(),
            })
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.stats_ready.connect(self._on_stats)
        self.thread.potions_ready.connect(self.potions_ready)
        self.thread.failed.connect(self._on_failed)
        self.thread.stream_status.connect(self._on_stream_status)
        self.thread.finished.connect(
            lambda _t=self.thread: self._on_finished(_t))
        self.thread.start()

        self.btn_start.setEnabled(False)
        self.btn_infer.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.lbl_stream.setText("—")
        self.lbl_stream.setStyleSheet("color:#80868b; font-weight:600;")
        self.lbl_stats.setText("正在收画面…（点「开始推理」开启识别）")

    def start_infer(self):
        """在已收画面的基础上开启推理（YOLO + 决策 + 画框）。"""
        if self.thread is None:
            return
        self.thread.set_infer(True)
        self.btn_infer.setEnabled(False)
        self.lbl_stats.setText("推理已开启（首次会加载模型，几秒）…")

    def stop(self):
        if self.thread is None:
            return
        self.thread.stop()
        # 等线程真正退出（内部要 join reader、关闭流 socket）再恢复「开始」。
        # 不等的话，快速「停止→开始」时旧线程的 UDP socket 还没释放，
        # 新线程 bind 端口会报 Errno 10048（端口被占用）。
        self.lbl_stats.setText("正在停止…（等收流线程释放端口）")
        self.thread.wait(5000)
        self.thread = None
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_infer.setEnabled(False)
        self.lbl_stats.setText("已停止")
        self.lbl_stream.setText("—")
        self.lbl_stream.setStyleSheet("color:#80868b; font-weight:600;")

    def shutdown(self):
        """窗口关闭时调用：等线程真正退出，避免 QThread 被销毁时报错。"""
        if self.thread is None:
            return
        self.thread.stop()
        self.thread.wait(3000)
        self.thread = None

    # ---------------- 回调 ----------------

    def _on_frame(self, img):
        """收到新帧：**只留最新一帧**，渲染跟不上就丢中间帧，绝不排队。

        **为什么要合并**（"失焦就卡"的主因之一）：这个槽跑在 GUI 主线程上，
        每帧要做两次整幅拷贝（QImage + QPixmap）加一次缩放 —— 1920×1080 大约
        10~20 ms；而线程按 30 fps 推信号。主线程只要慢一点（**失焦时被系统降级**、
        窗口正在缩放、机器在跑训练），队列就**越堆越长**：画面越来越滞后、
        鼠标点一下半天才响应。这个延迟发生在主线程队列里，工作线程侧的统计
        根本看不到（见 `_render` 的注释）—— 所以只看监控数字一切正常。

        合并办法：最新帧放 `_pending`，只挂**一个**零延时任务去画；画之前又来
        新帧就只覆盖 `_pending`（等于"追赶时不补旧帧"）。面板不可见（切到别的
        页签、窗口被藏起来）连画都不画：画面没人看，省下的时间留给决策回路。
        """
        self._last_bgr = img        # 取帧类功能（探针/HP 条框选）永远要最新的
        self._pending = img
        if self._render_pending:
            self._disp_merged += 1  # 上一帧还没画完 → 这一帧被合并掉
            return
        self._render_pending = True
        QTimer.singleShot(0, self._draw_pending)

    def _draw_pending(self):
        """把 `_pending` 画出来（零延时任务，和上一条 _on_frame 同一个合并节拍）。"""
        self._render_pending = False
        img = self._pending
        self._pending = None
        if img is None:
            return
        if not self.isVisible():
            self._disp_skipped += 1
            return
        t0 = time.perf_counter()
        vis = img
        if getattr(self, "ck_probe_box", None) is not None and self.ck_probe_box.isChecked():
            # 画采样框（`_draw_probe_overlay` 内部自己拷副本，见那里的说明）
            vis = self._draw_probe_overlay(img)
        self._last_pix = _bgr_to_pixmap(vis)
        self._render()
        self._draw_ms = (time.perf_counter() - t0) * 1000.0
        self._disp_drawn += 1

    def current_frame(self):
        """返回最近一帧画面（BGR ndarray）；还没有画面时返回 None。

        **不受绘制合并影响**：被合并/因为不可见没画的帧，这里照样拿得到 ——
        探针标定、HP/MP 条框选都靠它。
        """
        return self._last_bgr

    def set_minimap_overlay(self, pix, src_rect=None, frame_rect=None,
                            alpha=0.55, note=""):
        """在实时画面上挂一层「地形叠加图」；传 `pix=None` 卸掉。

        **为什么必须画在显示层、不能烘进帧**：`current_frame()`（= `_last_bgr`）
        还要喂给探针标定、HP/MP 条框选，以及小地图「从实时画面框选」的标定弹窗
        （`LiveFrameRegionClient` 拿它裁出面板去做模板匹配）。叠加一旦烘进帧，
        标定弹窗就会**拿叠加图和它自己匹配** —— 匹配分虚高，框选还会框到画上去的
        东西。所以这一层只在 `_render` 里往 QPixmap 上补，帧数据一个字节不动。

        成本（实测：画面 1366×768、面板 134×109、叠加源图 1340×1010）：
        `setPixmap(缩放)` 1.150 ms → 加这一层 1.212 ms，**多 0.06 ms**；
        同样的效果烘进 numpy 帧要 2.7 ms（差 25 倍），还不算它污染帧的代价。
        快的原因是 Qt 只往目标区写，不会去重采样整张源图 —— **别改成 Smooth**。

        参数一律是**画面坐标 / 画面像素**，由调用方照标定算好
        （`perception/minimap.frame_overlay_rects`）：面板这边不读文件、不算几何。
        """
        if pix is None or pix.isNull() or frame_rect is None:
            self._ov_minimap = None
            self._render()          # 立刻擦掉上一层，别等下一帧
            return
        # `note`：叠在**框选那块的下方**的一行字（玩家世界坐标，见 路线识别面板）。
        # 一起存进来，因为每次重画时都要重新贴上 —— 帧是新的，文字也得跟着重贴。
        self._ov_minimap = (pix, src_rect, tuple(int(v) for v in frame_rect),
                            float(alpha), str(note or ""))
        self._render()

    def set_overlay_note(self, note):
        """只换叠图那行读数 → 换成 True；现在没挂叠图 → False。

        **为什么要单开一个入口**：世界坐标那行每 250ms 更新一次，走
        `set_minimap_overlay` 会把地形 JSON 和底图 PNG 一起重读一遍（那是几十
        毫秒级的活，还全在 GUI 主线程上）—— 而这里要换的只是一个字符串。
        调用方拿 False 时再整算一次（见 route_panel._tick_world）。
        """
        ov = self._ov_minimap
        if ov is None:
            return False
        self._ov_minimap = tuple(ov[:4]) + (str(note or ""),)
        self._render()
        return True

    def _render(self):
        if self._last_pix is None:
            return
        # 必须用 FastTransformation。
        # SmoothTransformation 缩放一张 1920x1080 要几十毫秒，而这个槽跑在
        # GUI 主线程上 —— 线程按 30fps 推信号，主线程却每帧花 40~60ms 处理，
        # 信号队列就会越堆越长，画面看起来"延迟几百毫秒"。
        #
        # 这个延迟发生在主线程队列里，工作线程侧的统计根本看不到 ——
        # 所以监控数字一切正常，体感却明显滞后。
        pix = self._last_pix.scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        if self._ov_minimap is not None and not pix.isNull():
            self._paint_minimap_overlay(pix, self._ov_minimap)
        self.view.setPixmap(pix)

    def _paint_minimap_overlay(self, pix, ov):
        """把叠加层画到**已经缩放到标签尺寸**的那张图上。

        倍率要用 `pix.width() / 原始帧宽` 反推，**不能**用 `min(标签宽/画面宽, ...)`：
        `scaled()` 会取整，两者差的那一点点在这个几十像素的叠加区上正好看得见。
        """
        img, src, rect, alpha, note = (list(ov) + [""])[:5]
        fw = self._last_pix.width() or 1
        fh = self._last_pix.height() or 1
        sx, sy = pix.width() / float(fw), pix.height() / float(fh)
        if sx <= 0 or sy <= 0:
            return
        x, y, w, h = rect
        dst = QRect(int(round(x * sx)), int(round(y * sy)),
                    max(1, int(round(w * sx))), max(1, int(round(h * sy))))
        if not dst.intersects(pix.rect()):
            return                  # 整块在画面外（换了分辨率还没重框）：不白画
        p = QPainter(pix)
        try:
            p.setOpacity(max(0.0, min(1.0, alpha)))
            if src is None:         # None = 整张叠加图（fit 模式）
                p.drawPixmap(dst, img)
                self._draw_note(p, dst, note)       # 画完叠图再贴读数
                return
            sx0, sy0, sx1, sy1 = (int(round(v)) for v in src)
            # 源矩形**超出叠加图**是正常情况：面板框得比底图还高时就会这样
            # （实测就是这么框的：面板 134×109，而底图只有 101 高 —— 多出来的
            # 那 8 行底图里本来就没有内容）。
            #
            # 这种情况**交给 Qt 就行**：实测它会自己剪掉源、并把目标矩形按比例
            # 调好 —— 画出来的范围和"手工求交集再按比例裁目标"逐像素一致
            # （差 ≤1px 是取整）。**别在这儿自己再写一遍裁剪**：多余，而且比例
            # 系数很容易写错，错了就是整层叠图偏一小截、看着像"标定偏了"。
            p.drawPixmap(dst, img, QRect(sx0, sy0, max(1, sx1 - sx0),
                                         max(1, sy1 - sy0)))
            self._draw_note(p, dst, note)
        finally:
            p.end()

    @staticmethod
    def _draw_note(p, dst, note):
        """在小地图那块框的**下方**贴一行读数（玩家世界坐标）。

        为什么画在叠加层**之后**：它是读数，被半透明的地形图压住就看不清了。
        放不下就翻到框的上方 —— 框本来就贴着画面下沿时，写字会掉出画面外
        （这个项目里标签一律这么处理）。带描边是为了在任何底色上都读得出来。
        """
        from PyQt5.QtGui import QColor, QFont
        if not note:
            return
        p.setOpacity(1.0)
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        x = max(0, int(dst.left()))
        y = int(dst.bottom()) + 4
        # 兜一道：这行字是**一行不换行**的，太长会横穿整个画面还看不清。
        # 调用方本来就该只给短句（见 route_panel._tick_world 的 _say），
        # 但这里也不能靠"上游会给短的"活着 —— 超出就省略号截断。
        max_w = max(60, p.device().width() - x - 2)
        if fm.horizontalAdvance(note) + 8 > max_w:
            note = fm.elidedText(note, Qt.ElideRight, max(20, max_w - 8))
        tw, th = fm.horizontalAdvance(note) + 8, fm.height() + 2
        if y + th > p.device().height():
            y = max(0, int(dst.top()) - th - 4)     # 下方放不下 → 翻到上面
        box = QRect(x, y, tw, th)
        p.fillRect(box, QColor(0, 0, 0, 150))
        p.setPen(QColor(0, 0, 0))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):   # 描边
            p.drawText(box.adjusted(dx + 4, dy, dx + 4, dy),
                       Qt.AlignLeft | Qt.AlignVCenter, note)
        p.setPen(QColor(255, 255, 255))
        p.drawText(box.adjusted(4, 0, 4, 0),
                   Qt.AlignLeft | Qt.AlignVCenter, note)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._render()          # 窗口变大时把画面重新贴满

    def _verify_calib_if_due(self, s):
        """框选保存后过 2.5 秒回头看一眼：实时那边到底通没通。

        「保存成功」只说明**当时那一帧**解得出来；实时是每帧都在当前画面上采样，
        几何差一点就会解出乱码（而乱码偶尔也能落在值域内骗过判据）。所以这里用
        **增量**判断：这几秒里有没有解出过。不通过就当场说清楚，
        别让人对着两行矛盾的信息自己猜（实测为这个来回折腾了好几轮）。
        """
        v = getattr(self, "_verify", None)
        if not v or time.monotonic() < v[0]:
            return
        self._verify = None
        # ② 解得出、但**时间戳在乱跳** → 这份几何按单调性判据不可信。
        #    比"一帧都解不出"隐蔽得多：屏幕上延迟数照样在跳，看着像在工作。
        if (int(s.get("probe_samples") or 0) >= 6
                and ((s.get("probe_mono") is False
                      and int(s.get("probe_jumpy") or 0) > 0)
                     or s.get("probe_value_ok") is False)):
            QMessageBox.warning(
                self, "这份标定不可信",
                "标定已保存、也**解得出时间码**，但实时这几秒解出的时间戳在乱跳"
                "（%d 帧）——\n说明采样点没落在方块中心、低位被采错了。这种错"
                "**光看能不能解出是看不出来的**\n（错位照样能凑出一个「接近此刻」"
                "的值），所以那个延迟数一律不可信。\n\n"
                "两个办法：\n"
                "  1. 点「框选探针」重框：高度贴住方块、宽度量到整条带外缘；\n"
                "  2. python -m tools.probe_tune 手工推框（同一判据、当场告诉你对没对，\n"
                "     判据没过还不让保存）。\n\n"
                "参考：A 机按 cell=23 / gap=3（逻辑）画，换算到 1366x768 流里约"
                " cell=18.4 gap=2.4。"
                % int(s.get("probe_jumpy") or 0)
                + "\\n\\n（这份标定已经存了；重框会覆盖它。）")
            return
        ok = int(s.get("probe_ok") or 0) - v[1]
        bad = (int(s.get("probe_invalid") or 0)
               + int(s.get("probe_miss") or 0)) - v[2]
        if not s.get("probe_on") or s.get("delay_ms") is not None or ok > 0:
            return                      # 通了：不打扰
        if bad <= 0:
            return                      # 这几秒没解过（可能没画面）：不下结论
        QMessageBox.warning(
            self, "标定没生效",
            "标定已保存，但**实时那边这几秒一帧都没解出来**（无效 %d 帧）——\n"
            "说明这份几何在连续画面上不对，多半是框选时没框住方块。\n\n"
            "重框时注意：\n"
            "  1. 高度要**正好贴住方块**（小一格，40 位码往后全错位）\n"
            "  2. 宽度要**量到整条带的外缘**（最后一块的右边）\n"
            "  3. 对照 link.yaml 的 probe.cell / probe.gap（现在是 %s / %s）\n\n"
            "（这份标定已经存了，但它不生效；重框会覆盖它。）"
            % (bad, get("probe", "cell", 20), get("probe", "gap", 2)))

    def _on_stats(self, s):
        self._last_stats = s
        self._verify_calib_if_due(s)
        w, h = s.get("size") or (0, 0)
        d = s.get("delay_ms")
        self.lbl_stats.setToolTip("")
        # **几何可疑优先于一切**：时间戳在乱跳时，那个延迟数就是乱数（而且可能
        # 正好"看着正常"）—— 必须抢在它前面说清楚，否则又把人引到链路上去查。
        mono_bad = (s.get("probe_mono") is False
                    and int(s.get("probe_jumpy") or 0) > 0)
        value_bad = (s.get("probe_value_ok") is False
                     and s.get("probe_value_off_s") is not None)
        jumpy = (s.get("probe_on") and int(s.get("probe_samples") or 0) >= 6
                 and (mono_bad or value_bad))
        if jumpy:
            if mono_bad:
                why = "时间戳在乱跳 %d 帧" % int(s.get("probe_jumpy") or 0)
            else:
                why = ("解出的时刻差了 %.0f 秒（整片挪位）"
                       % abs(s.get("probe_value_off_s") or 0))
            d_txt = "端到端延迟  ——  （几何可疑：%s）" % why
            self.lbl_stats.setToolTip(
                "解得出时间码，但**这份几何不可信**：\n"
                "  · 乱跳那条 —— 解出的时间戳忽大忽小（步长本该≈帧周期）；\n"
                "  · 挪位那条 —— 时间戳**单调**、也在一天之内，但整片被挪了几秒~\n"
                "    几分钟：错位采样把低位采成了另一个位置，值照样「看着正常」。\n\n"
                "**这两种错光看「能不能解出」都看不出来**（错位照样能凑出一个接近\n"
                "此刻的值），所以上面那个延迟数一律不可信，宁可不显示。\n\n"
                "怎么修：\n"
                "  1. 点「框选探针」重框：高度**贴住方块**、宽度**量到整条带外缘**\n"
                "  2. python -m tools.probe_tune 手工推框（同一判据、当场看对没对）\n\n"
                "参考：A 机按 cell=23 / gap=3（逻辑）画，换算到 1366x768 流里约\n"
                "cell=18.4 gap=2.4 —— 如果你现在的 cell/gap 明显比这小，就是它了。\n"
                "（判据：probe_codec.Verdict，与 tools/probe_tune 共用一份。）")
        elif d is not None:
            d_txt = "端到端延迟 %6.0f ms" % d
        elif not s.get("probe_on"):
            d_txt = "端到端延迟  ——  （未启用探针）"
        elif s.get("probe_invalid"):
            # 解出来了、但值是**不可能的**（≥ 一天，例如 306001920ms ≈ 85 小时）
            # ＝采样点没落在方块上，**几何错了**。这时绝不能提示「去对时」——
            # 实测踩过：界面让人去对时，其实是框选没框准，方向全错。
            d_txt = "端到端延迟  ——  （解出的时间码无效：探针几何不对）"
            self.lbl_stats.setToolTip(
                "从框选区里解出的值不在一天之内（本次运行 %d 帧），说明采样点没落在\n"
                "方块上 —— **是位置/大小标错了，不是时钟问题**。点「框选探针」重框：\n"
                "  1. 要框住**整条方块带**（含最左边常亮白块 + 后面那个常黑块）\n"
                "  2. **框的高度要正好贴住方块**（框紧了 cell 会小一截，40 位码当场错位）\n"
                "  3. 对照 link.yaml 的 probe.cell（现在是 %s）看是不是差一截"
                % (int(s.get("probe_invalid", 0)), get("probe", "cell", 20)))
        elif s.get("probe_ok"):
            rej = int(s.get("probe_reject", 0))
            d_txt = "端到端延迟  ——  （已解出、但时钟对不上）"
            self.lbl_stats.setToolTip(
                "探针**解码是成功的、值也在一天之内**（本次运行已丢弃 %d 帧），算不出\n"
                "延迟是因为双机时钟差得超出可接受范围（对时后应只差几毫秒）。跑一次对时：\n"
                "    python -m tools.clock_sync --host <A机IP> --save\n"
                "当前用的时钟偏移 = %.0f ms（config/clock_offset.txt，每 5 秒自动重读，\n"
                "所以对完时不用重开预览）。" % (rej, s.get("clock_offset_ms", 0.0)))
        else:
            d_txt = "端到端延迟  ——  （探针解不出，共 %d 帧）" % s.get("probe_miss", 0)
            self.lbl_stats.setToolTip(
                "连时间码都没解出来（不是时钟问题）。按顺序查：\n"
                "  1. A 机在跑 python -m tools.probe_gen 吗\n"
                "  2. 点「框选探针」重新框一次（框住整条方块带）\n"
                "  3. 探针方块带有没有被游戏 UI 压住 / 画得太小（cell ≥ 16px）")
        # 断线重连在做的事放在最前面 —— 这时候帧率数字意义不大，状态才是要紧的
        note = (s.get("reconnect") or "").strip()
        head = "【%s】 " % note if note else ""
        # 本机负载告警再压一层到最前面：它是「上面这些数字为什么变差」的解释，
        # 比数字本身更要紧 —— 实测同时跑训练/并行标注时，推理的前向会与输入尺寸
        # 无关地整体变慢（同一份配置 10.0 → 23.7 ms），而 GPU 时钟功耗都正常。
        lw = (s.get("load_warn") or "").strip()
        if lw:
            head = "【%s】 " % lw + head
        # 明细放 tooltip：状态行那一行已经塞满了，硬挤进去反而看不清数字
        ld = (s.get("load_detail") or "").strip()
        if ld:
            cur = self.lbl_stats.toolTip()
            self.lbl_stats.setToolTip((cur + "\n\n" if cur else "") + ld)
        # 绘制那一段单独报：「显示 fps」只说明**推**了多少，看不出主线程画得
        # 动不动 —— 而"失焦就卡"恰恰卡在这里（合并丢弃的帧数一涨，就说明主线程
        # 跟不上推送、开始在丢中间帧；不丢帧时界面才不会滞后）。
        self.lbl_stats.setText(
            "%s%s ｜ 输入 %5.1f fps ｜ 处理 %5.1f fps ｜ 丢帧 %d ｜ 推理 %5.1f ms ｜ "
            "显示 %4.1f fps ｜ 绘制 %4.1f ms 合并丢弃 %d ｜ 检出 %d ｜ %d×%d"
            % (head, d_txt, s.get("recv_fps", 0), s.get("proc_fps", 0),
               s.get("dropped", 0), s.get("infer_ms", 0), s.get("show_fps", 0),
               self._draw_ms, self._disp_merged,
               s.get("boxes", 0), w, h))

    def _on_stream_status(self, status):
        if status == "waiting":
            self.lbl_stream.setText("等待推流…")
            self.lbl_stream.setStyleSheet("color:#b06000; font-weight:600;")
        elif status == "connected":
            self.lbl_stream.setText("已连接")
            self.lbl_stream.setStyleSheet("color:#137333; font-weight:600;")
        elif status == "no_stream":
            self.lbl_stream.setText("无流")
            self.lbl_stream.setStyleSheet("color:#c5221f; font-weight:600;")

    def _on_failed(self, msg):
        self.lbl_stats.setText("失败：%s" % msg)

    def _on_finished(self, t=None):
        # 只清理「当前线程」；若用户已重开新线程，旧线程退出不该动新线程的引用
        if t is not None and self.thread is not t:
            return
        self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_infer.setEnabled(False)
        if not self.lbl_stats.text().startswith("失败"):
            self.lbl_stats.setText("已停止")
            self.lbl_stream.setText("—")
            self.lbl_stream.setStyleSheet("color:#80868b; font-weight:600;")
