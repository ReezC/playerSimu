"""实时预览面板：收流 → YOLO 推理 → 画面实时显示。

设计要点见 live_thread.py 顶部说明。这一层只负责：
    · 收参数、起停线程
    · 把线程推来的帧画到界面上
    · 显示三个分开的速度指标
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSizePolicy, QVBoxLayout,
                             QWidget)

import numpy as np

from gui.live_thread import LiveThread
from gui.widgets import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox
from tools.config import ROOT, get, load_live, save_live


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


class LivePanel(QWidget):
    potions_ready = pyqtSignal(float, float)   # (hp, mp) 比例，转发给决策参数页

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.thread = None
        self._last_pix = None
        self._last_bgr = None   # 最近一帧画面（BGR），供 HP/MP 条在画面上框选
        self._rect = None       # 本地窗口模式下框选的区域 (x, y, w, h)
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

        row.addStretch(1)
        root.addLayout(row)

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

    def _pick_rect(self):
        from gui.region_selector import select_region
        rect = select_region(self)
        if rect is None:
            return
        self._rect = rect
        self.lbl_rect.setText("%d,%d  %d×%d" % rect)

    # ---------------- 项目 ----------------

    def bind(self, project):
        """绑定项目：自动找最新的模型权重。"""
        self.project = project
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
            save_live({
                "conf_mob": self.sp_conf_mob.value(),
                "conf_player": self.sp_conf_player.value(),
                "imgsz": self.sp_imgsz.value(),
                "device": self.ed_device.text().strip() or "0",
                "capture_fps": self.sp_capfps.value(),
                "draw": self.ck_draw.isChecked(),
            })
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

        # 保存实时参数，下次开 GUI 不用再调
        save_live({
            "conf_mob": self.sp_conf_mob.value(),
            "conf_player": self.sp_conf_player.value(),
            "imgsz": self.sp_imgsz.value(),
            "device": self.ed_device.text().strip() or "0",
            "capture_fps": self.sp_capfps.value(),
            "draw": self.ck_draw.isChecked(),
        })

        common = {
            "weights": w,
            "conf_mob": self.sp_conf_mob.value(),
            "conf_player": self.sp_conf_player.value(),
            "imgsz": self.sp_imgsz.value(),
            "device": self.ed_device.text().strip() or "0",
            "draw": self.ck_draw.isChecked(),
            "show_fps": 30.0,
            "player_id": pid,
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
        pm = _bgr_to_pixmap(img)
        self._last_pix = pm
        self._last_bgr = img
        self._render()

    def current_frame(self):
        """返回最近一帧画面（BGR ndarray）；还没有画面时返回 None。"""
        return self._last_bgr

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
        self.view.setPixmap(self._last_pix.scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._render()          # 窗口变大时把画面重新贴满

    def _on_stats(self, s):
        w, h = s.get("size") or (0, 0)
        d = s.get("delay_ms")
        if d is not None:
            d_txt = "端到端延迟 %6.0f ms" % d
        elif s.get("probe_on"):
            d_txt = "端到端延迟  ——  （探针未解出，共 %d 帧）" % s.get("probe_miss", 0)
        else:
            d_txt = "端到端延迟  ——  （未启用探针）"
        # 断线重连在做的事放在最前面 —— 这时候帧率数字意义不大，状态才是要紧的
        note = (s.get("reconnect") or "").strip()
        head = "【%s】 " % note if note else ""
        self.lbl_stats.setText(
            "%s%s ｜ 输入 %5.1f fps ｜ 处理 %5.1f fps ｜ 丢帧 %d ｜ 推理 %5.1f ms ｜ "
            "显示 %4.1f fps ｜ 检出 %d ｜ %d×%d"
            % (head, d_txt, s.get("recv_fps", 0), s.get("proc_fps", 0),
               s.get("dropped", 0), s.get("infer_ms", 0), s.get("show_fps", 0),
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
