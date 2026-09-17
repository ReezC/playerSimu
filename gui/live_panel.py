"""实时预览面板：收流 → YOLO 推理 → 画面实时显示。

设计要点见 live_thread.py 顶部说明。这一层只负责：
    · 收参数、起停线程
    · 把线程推来的帧画到界面上
    · 显示三个分开的速度指标
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout, QWidget)

from gui.live_thread import LiveThread
from gui.widgets import NoWheelDoubleSpinBox, NoWheelSpinBox
from tools.config import ROOT, get


def _bgr_to_pixmap(img):
    """numpy BGR → QPixmap。

    必须 copy()：QImage 只引用那段内存，不持有所有权。numpy 数组一旦被回收，
    界面拿到的就是野指针 —— 表现是花屏或者直接崩。
    """
    h, w, ch = img.shape
    qimg = QImage(img.data, w, h, ch * w, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class LivePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.thread = None
        self._last_pix = None
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

        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        bar.addWidget(self.btn_stop)

        bar.addSpacing(10)
        bar.addWidget(QLabel("源"))
        self.ed_url = QLineEdit(get("stream", "url", "udp://0.0.0.0:5000"))
        self.ed_url.setMinimumWidth(220)
        bar.addWidget(self.ed_url, 1)

        root.addLayout(bar)

        # ---- 参数行 ----
        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel("权重"))
        self.ed_weights = QLineEdit()
        self.ed_weights.setReadOnly(True)
        self.ed_weights.setMinimumWidth(200)
        row.addWidget(self.ed_weights, 1)

        row.addWidget(QLabel("conf"))
        self.sp_conf = NoWheelDoubleSpinBox()
        self.sp_conf.setRange(0.0, 1.0)
        self.sp_conf.setDecimals(2)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(0.30)
        row.addWidget(self.sp_conf)

        row.addWidget(QLabel("imgsz"))
        self.sp_imgsz = NoWheelSpinBox()
        self.sp_imgsz.setRange(320, 2048)
        self.sp_imgsz.setValue(960)
        row.addWidget(self.sp_imgsz)

        row.addWidget(QLabel("设备"))
        self.ed_device = QLineEdit("0")
        self.ed_device.setFixedWidth(50)
        row.addWidget(self.ed_device)

        self.ck_draw = QCheckBox("画框")
        self.ck_draw.setChecked(True)
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
        self.view.setStyleSheet(
            "background:#202124; color:#9aa0a6; border:1px solid #dadce0;")
        root.addWidget(self.view, 1)

        # ---- 统计条 ----
        self.lbl_stats = QLabel("未开始")
        self.lbl_stats.setStyleSheet("color:#5f6368;")
        root.addWidget(self.lbl_stats)

        self.lbl_hint = QLabel(
            "收流 fps = 链路给到的输入速度 ｜ 处理 fps = 我们实际跑完的帧率（低于收流就会丢帧）｜ "
            "丢帧 = 为了不积压延迟而主动丢掉的帧数（正常，实时系统宁可丢帧也不排队）｜\n"
            "推理 ms = 模型耗时，要跑 60fps 得 ≤16ms ｜ 显示 fps = 界面刷新，故意低于推理，不参与性能判断")
        self.lbl_hint.setStyleSheet("color:#80868b;")
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint)

    # ---------------- 项目 ----------------

    def bind(self, project):
        """绑定项目：自动找最新的模型权重。"""
        self.project = project
        if self.thread is not None:
            self.stop()

        if project is None:
            self.ed_weights.setText("")
            return

        got = sorted(project.dir_of("models").glob("*.pt"))
        if not got:
            got = sorted(project.dir_of("runs").glob("**/weights/best.pt"))
        self.ed_weights.setText(str(got[-1]) if got else "")

    @staticmethod
    def _load_offset_ms():
        """读双机时钟偏移。没有它探针算不出延迟。"""
        p = ROOT / "config" / "clock_offset.txt"
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except Exception:
            return 0.0

    # ---------------- 起停 ----------------

    def start(self):
        if self.thread is not None:
            return

        w = self.ed_weights.text().strip()
        if not w:
            self.lbl_stats.setText("没有可用的模型权重 —— 请先完成 ⑦ 训练")
            return

        url = self.ed_url.text().strip()
        if not url:
            self.lbl_stats.setText("请填写收流地址")
            return

        self.thread = LiveThread({
            "url": url,
            "weights": w,
            "conf": self.sp_conf.value(),
            "imgsz": self.sp_imgsz.value(),
            "device": self.ed_device.text().strip() or "0",
            "draw": self.ck_draw.isChecked(),
            "show_fps": 30.0,
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
        self.thread.failed.connect(self._on_failed)
        self.thread.finished.connect(self._on_finished)
        self.thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_stats.setText("正在启动（首次会加载模型，几秒）…")

    def stop(self):
        if self.thread is None:
            return
        self.thread.stop()
        # 不在这里 wait()：线程可能在等一帧（收流阻塞），wait 会把界面也卡住。
        # 让它自己在后台退出，finished 信号回来再收拾。
        self.btn_stop.setEnabled(False)
        self.lbl_stats.setText("正在停止…")

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
        self._render()

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
        self.lbl_stats.setText(
            "%s ｜ 收流 %5.1f fps ｜ 处理 %5.1f fps ｜ 丢帧 %d ｜ 推理 %5.1f ms ｜ "
            "显示 %4.1f fps ｜ 检出 %d ｜ %d×%d"
            % (d_txt, s.get("recv_fps", 0), s.get("proc_fps", 0),
               s.get("dropped", 0), s.get("infer_ms", 0), s.get("show_fps", 0),
               s.get("boxes", 0), w, h))

    def _on_failed(self, msg):
        self.lbl_stats.setText("失败：%s" % msg)

    def _on_finished(self):
        self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if not self.lbl_stats.text().startswith("失败"):
            self.lbl_stats.setText("已停止")
