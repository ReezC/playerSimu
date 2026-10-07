"""实时预览面板：收流 → YOLO 推理 → 画面实时显示。

设计要点见 live_thread.py 顶部说明。这一层只负责：
    · 收参数、起停线程
    · 把线程推来的帧画到界面上
    · 显示三个分开的速度指标
"""

from PyQt5.QtCore import QRect, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QDialog, QFileDialog, QFormLayout,
                             QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QPlainTextEdit, QProgressBar, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

import time
from pathlib import Path

import numpy as np

from decision.agent import settings
from gui import theme
from gui.live_thread import LiveThread
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox,
                         play_sound)
from core.config import ROOT, get, load_live, update_live

#: ⭐⭐ **预览重绘的最低间隔**（毫秒 ✓ 用户 2026-10-06 ✓ 见 `_render` 的说明 ✓）：
#: 取 **20**（= 50 fps 上限 ✓）—— 比管线实测的 `proc_fps`(**39** ✓) 略快一点 ✓
#: ⇒ **正常时它不是限速者** ✓，只在"一小段时间里被请求多次"（换帧 + 叠图 + 那行读数 ✓）
#: 时把重复的那些**合并掉** ✓ ⇒ 主线程开销封顶 ✓、画面永远是最新那一帧 ✓。
RENDER_MIN_MS = 20.0


def _bgr_to_pixmap(img):
    """numpy BGR → QPixmap。

    三个坑（都是真实踩过的）：
      1. QImage 只引用那段内存、**不持有所有权** ⇒ 它引的那段必须在
         `fromImage` 期间活着；`data` 是本函数的局部变量，够用。
         ⚠ 曾经还有一次 `qimg.copy()`（怕 numpy 数组被回收）—— 那是多余的：
         保活的是 `data`，而 `fromImage` 会立刻把像素拷进 QPixmap。
         1920×1080 实测：去掉它 **9.43 → 7.02 ms/帧**（30fps 下每秒省 72ms，
         而这笔开销跑在 **GUI 主线程**上 ⇒ 直接少抢推理的 CPU）。
      2. to_ndarray(bgr24) 返回的帧带行 padding（非连续，stride 比 3*w 大），
         QImage 按 3*w 读会整幅错位 —— 先 ascontiguousarray 去掉 padding。
         ⚠ 别改成"直接 tobytes + 3*w"：那样**更慢**（实测 7.02 → 14.77 ms/帧，
         跨步拷贝走的是逐行通用路径），仍然要 ascontiguousarray。
      3. numpy 的 .data 在新版返回 memoryview，PyQt5 的 QImage 不认它，
         要用 tobytes() 拿 bytes 再构造。
    """
    img = np.ascontiguousarray(img)
    h, w, ch = img.shape
    data = img.tobytes()
    qimg = QImage(data, w, h, img.strides[0], QImage.Format_BGR888)
    return QPixmap.fromImage(qimg)


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


class _TeeLog:
    """导出的打印 —— **原 stdout 和界面日志各写一份**（一件都不许少 ✓）。

    用户 2026-10-04 ✓ 原话："新的imgsz导出时，数据工作台**没有日志**和进度条" ⇒ 导出过程的
    那堆输出（ultralytics / 我们自己的 print ✓）必须**实时**进到界面日志框里 ✓。

    ⚠⚠ 为什么必须 **tee**（不能只往界面塞）：`redirect_stdout` 改的是**进程级**的
      `sys.stdout`（全局只有一个 ✓），而实时线程 / 别的页也可能在打印 ⇒ 只塞界面、
      不写回原 stdout 的话，**那些行就永远进不了 `stdout.log`** ✗
      —— 这个项目排查问题几乎全靠日志 ⇒ 那是拿功能换功能 ✗。
    ⚠ `sys.stdout` 在 `pythonw` 下可能是 `None` ⇒ 原样吞掉（只发界面 ✓ 不许抛 ✗）。
    ⚠ `write` 会被 `print` **分几次**调（`print(a, b)` ⇒ 多次 write ✓）⇒ 自己攒到换行才
      发一条（不然界面日志会被拆成半行 ✗）。
    """

    def __init__(self, orig, emit):
        self._orig = orig
        self._emit = emit
        self._buf = ""

    def write(self, s):
        try:
            if self._orig is not None:
                self._orig.write(s)
        except Exception:                            # noqa: BLE001
            pass
        try:
            self._buf += str(s)
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._emit(line)
        except Exception:                            # noqa: BLE001
            pass
        return len(str(s))

    def flush(self):
        try:
            if self._orig is not None:
                self._orig.flush()
        except Exception:                            # noqa: BLE001
            pass

    def isatty(self):
        return False


class EngineExportThread(QThread):
    """后台导 TensorRT 引擎（用户 2026-10-03 ✓："不合适的话弹提示指引然后搞个按钮一键导出也行"）。

    ⚠ **必须后台** ✗：一次导出要**几分钟**（TensorRT 在自动调优 kernel ✓）⇒ 放主线程就是整个
      界面假死 ✓（这个项目里"界面假死"已经被报过好几次 ✓）。
    ⚠ **只复用已有那条实现**（`tools/export_engine.export_for` ✓ 约定"一处实现"✓）：它自己
      处理了"**权重路径含中文**"那个坑 ✓（TensorRT / onnx 的 C 扩展读不了中文路径 ✗ ——
      而本项目的地图名/项目名**全是中文** ⇒ 那条路必然踩 ✓）。
    ⭐ **`line` 信号 = 导出过程的每一行**（用户 2026-10-04 ✓ 见 `_TeeLog`）⇒ 界面上那个
      「导出」进度窗拿它刷日志 ✓（原来这几分钟**一个字都看不到** ✗）。
    """

    done = pyqtSignal(str)      # 成功：最终引擎路径 ✓
    failed = pyqtSignal(str)    # 失败：原因（直接给对话框显示 ✓）
    line = pyqtSignal(str)      # ⭐ 导出过程的一行输出（界面日志用 ✓）

    def __init__(self, weights, imgsz, device="0", parent=None):
        super().__init__(parent)
        self._weights = str(weights)
        self._imgsz = int(imgsz)
        self._device = str(device)

    def run(self):
        import sys
        from contextlib import redirect_stderr, redirect_stdout
        from pathlib import Path

        def say(text):
            try:
                self.line.emit(str(text))
            except Exception:                        # noqa: BLE001 —— 日志发不出去不许影响导出 ✗
                pass

        say("=== 开始导出：%s → %s_%d.engine（imgsz=%d，device=%s）"
            % (Path(self._weights).name, Path(self._weights).stem, self._imgsz,
               self._imgsz, self._device))
        say("（TensorRT 会自己调优 kernel ⇒ 这几分钟**没有百分比**，看日志在动就说明没死 ✓）")
        try:
            with redirect_stdout(_TeeLog(sys.stdout, say)), \
                    redirect_stderr(_TeeLog(sys.stderr, say)):
                from tools.export_engine import export_for
                out = str(export_for(self._weights, self._imgsz,
                                     device=self._device))
        except Exception as e:                       # noqa: BLE001
            say("✗ 导出失败：%s: %s" % (type(e).__name__, e))
            self.failed.emit("%s: %s" % (type(e).__name__, e))
            return
        say("✓ 导出完成：%s" % out)
        self.done.emit(out)


class EngineExportDialog(QDialog):
    """导出引擎的**进度窗**（用户 2026-10-04 ✓ 原话："新的imgsz导出时，数据工作台没有日志
    和进度条"）。

    为什么要有它：导出一次要**几分钟**（TensorRT 调优 kernel ✓），而原来界面上只有"按钮变成
    导出中… + 状态行一句话" ⇒ 这**几分钟里没有日志、也没有任何进度** ✗ —— 人根本不知道
    它是在跑还是已经死了 ✓（对照 `gui/yolo_workbench.py` 的训练/验证页：那两处早就是
    「日志区 + 忙式进度条」✓，此处只是把同一套搬过来 ✓）。

    三件：
      · **忙式进度条**（`setRange(0, 0)`）：TensorRT **不给真实百分比** ⇒ **不编** ✗
        （编个"40%"比不给更坏 ✓），配「已用 mm:ss」的计时器 ⇒ "在动"看得见 ✓；干完
        自己收起来 ✓（同 `yolo_workbench` 的口径 ✓）；
      · **日志框**（`QPlainTextEdit`，只读、`objectName="Log"` 走全局样式 ✓）：导出过程
        **实时**追加（吃 `EngineExportThread.line` ✓），只追加、自动滚到最新 ✓；
      · **关闭按钮**：跑着时写「隐藏（后台继续导出）」✓ —— 关窗**不中止**导出
        （中止没有干净做法：TensorRT 中途被杀会留半份引擎 ✗）；结果照旧由
        `_on_export_done/failed` 弹框报 ✓。
    """

    def __init__(self, imgsz, parent=None):
        super().__init__(parent)
        self._t0 = time.monotonic()
        self._imgsz = int(imgsz)
        self._done = False
        self.setWindowTitle("导出 TensorRT 引擎（imgsz=%d）" % self._imgsz)
        root = QVBoxLayout(self)
        self.lbl_tip = QLabel("正在导出 imgsz=%d 的引擎…（几分钟；期间**界面仍能用** ✓）"
                              % self._imgsz)
        root.addWidget(self.lbl_tip)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)          # 忙式：拿不到真百分比 ⇒ 不编 ✓
        self.bar.setToolTip(
            "导出过程**拿不到真实百分比**（TensorRT 自己在调优 kernel）⇒ 这里只表示"
            "「**还在跑**」✓；旁边的「已用」才是进度参考 ✓。")
        root.addWidget(self.bar)
        self.lbl_time = QLabel("已用 0:00")
        self.lbl_time.setStyleSheet("color: #80868b;")
        root.addWidget(self.lbl_time)
        self.log = QPlainTextEdit()
        self.log.setObjectName("Log")            # 同标注/训练页那套样式 ✓
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)      # 超长截头（别无限吃内存 ✓）
        self.log.setTextInteractionFlags(Qt.TextSelectableByMouse
                                         | Qt.TextSelectableByKeyboard)
        self.log.setMinimumHeight(180)
        self.log.setToolTip("导出过程的输出（ultralytics / TensorRT / 本程序 ✓）——"
                            "实时追加、只追加 ✓。")
        root.addWidget(self.log, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_hide = QPushButton("隐藏（后台继续导出）")
        self.btn_hide.setToolTip("关掉这个窗**不会中止导出** —— 跑完照旧弹框报结果 ✓。")
        self.btn_hide.clicked.connect(self.hide)
        row.addWidget(self.btn_hide)
        root.addLayout(row)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self.resize(760, 460)
        theme.bind_window_state(self, "engine_export")

    # ---- 给 `EngineExportThread.line` 用的入口 ----

    def append(self, text):
        """追一行日志（带时刻、只追加、自动滚到最新 ✓ —— 同 `yolo_workbench` 的日志区 ✓）。"""
        s = str(text).rstrip()
        if not s:
            return
        self.log.appendPlainText(time.strftime("%H:%M:%S  ") + s)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _tick(self):
        m, s = divmod(int(time.monotonic() - self._t0), 60)
        self.lbl_time.setText("已用 %d:%02d" % (m, s))

    def mark_finished(self, ok=True, why=""):
        """导出结束：进度条收起来 + 计时停 + 按钮改成「关闭」✓（如实标成功/失败 ✓）。"""
        self._done = True
        self._timer.stop()
        self.bar.setVisible(False)
        m, s = divmod(int(time.monotonic() - self._t0), 60)
        self.lbl_time.setText("%s（用时 %d:%02d）%s"
                              % ("✓ 完成" if ok else "✗ 失败", m, s,
                                 ("　" + str(why)[:80]) if why else ""))
        self.lbl_tip.setText("导出已结束 —— 可以关掉这个窗了 ✓"
                             if ok else "导出失败了 —— 看上面的日志 ✓")
        self.btn_hide.setText("关闭")
        try:
            self.btn_hide.clicked.disconnect()
        except Exception:                            # noqa: BLE001
            pass
        self.btn_hide.clicked.connect(self.close)


class LivePanel(QWidget):
    potions_ready = pyqtSignal(float, float)   # (hp, mp) 比例，转发给决策参数页

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.thread = None
        self._last_pix = None
        #: ⭐⭐ 预览重绘**限频**用的两个小状态（用户 2026-10-06 ✓ 见 `_render` 的说明 ✓）：
        #: `_last_render_at` = 上一次真画上去的时刻；`_render_pending` = 已经排了一次
        #: 延迟重绘（同一拍里再来请求就只更新 `_last_pix` ✓ 合并掉 ✓）。
        self._last_render_at = 0.0
        self._render_pending = False
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

        # ---- 参数区（2026-09-29 按 §4「一行只放一个参数组」拆行：原来 10 个控件挤一行，
        #      窗口一窄所有标签一起被压没 ✗。按「你在干什么」分三行 ✓）----
        # ⚠ 本页**有意不放滚动区**：实时画面必须常驻占满（预览页出滚动条是反交互 ✗）——
        #   取舍同 player_panel 的「参数 + 常驻画布」；view 是 QLabel 虽然能进滚动区 ✗。
        #
        # 行 1（推理参数：权重 / 置信度阈值 / 分辨率 / 设备）—— 这一行到这儿结束
        row = QHBoxLayout()
        row.setSpacing(6)

        # ⭐⭐ **「权重」改成能自己选的下拉**（用户 2026-10-04 ✓ 原话："开始推理按钮下面的权重
        #   为什么不能自己选（就像在 训练模型页签→训练卡片→基础权重 一样）"）。
        #   原来是个**只读** `QLineEdit`：内容只能由 `bind(project)` 自动挑（`.engine` 优先 ✓）
        #   —— 于是"**别的项目训好的模型想直接拿来试**""手头这份实验权重想跑一下""导出了
        #   新引擎想立刻切过去"统统做不到 ✗（只能去改目录、或等它按 mtime 自己挑 ✓）。
        #   候选口径**与训练卡片/④ 那张卡的「YOLO 权重」一致**（同一份 `list_weights()` ✓，
        #   不另发明一套扫描 ✗）；默认仍是**自动挑最新**（老行为没变 ✓），
        #   一旦你自己选了 ⇒ **按项目记住**（`project.yaml` 的 `live.weights` ✓，
        #   同「基础权重」记在 `train.model` 的口径 ✓），下拉里的「（自动…）」随时切回来 ✓。
        row.addWidget(QLabel("权重"))
        self.cmb_weights = NoWheelComboBox()
        self.cmb_weights.setMinimumWidth(200)
        self.cmb_weights.setMaxVisibleItems(24)
        self.cmb_weights.setToolTip(
            "这次推理**用哪份模型**。\n\n"
            "· （自动：本项目最新）—— 老行为 ✓：优先用最新导出的 **TensorRT 引擎**\n"
            "  （`models/*.engine`，推理快 3-5 倍），没有才用 `models/*.pt`，再退回\n"
            "  `runs/**/weights/best.pt`（按修改时间取最新的 ✓）；\n"
            "· 本项目训好的：`.engine`（引擎 · 尺寸焊死）/ `.pt`（尺寸随便填 ✓）；\n"
            "· 别的项目训好的：换张图先拿旧模型试很有用 ✓（会提示「这份权重不是本项目训的」）；\n"
            "· （自定义 / 浏览…）：随便挑一个 `.pt` / `.engine`。\n\n"
            "⚠ **改完要重新点「开始」才生效**（正在跑的实时不会中途换模型 —— 同其它参数 ✓）。\n"
            "⚠ 一旦你手动选过，就会**按项目记住**、不再自动跟最新 ⇒ 想回到自动挑，\n"
            "   选上面那项「（自动：本项目最新）」即可。")
        self.cmb_weights.activated[int].connect(self._on_weight_pick)
        row.addWidget(self.cmb_weights, 1)

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
        self.sp_imgsz.setToolTip(
            "推理时把画面缩到多大再喂网络：越大越准、越慢（960→640 约省 56% 推理时间）。\n\n"
            "⚠ 如果这个项目的权重是 **TensorRT 引擎**（`models/*.engine` —— 实时会优先选它 ✓），\n"
            "   那它的输入尺寸是**导出时焊死**的（比如 640）⇒ 这里填别的**不会生效**：\n"
            "   实时会**按引擎的尺寸跑**（所以不会再报错起不来 ✓），并在控制台和\n"
            "   `perf.log` 的段头里写明实际用的是多少（`imgsz=640(引擎·已对齐)`）。\n"
            "   想真换成别的尺寸：点右边「导出该尺寸引擎」（一键 ✓ 要等几分钟），\n"
            "   或改用 `.pt` 权重（`.pt` 没有这个限制 —— 320~2048 随便填都能跑 ✓）。")
        row.addWidget(self.sp_imgsz)

        # ⭐ **一键导出「上面那个 imgsz」的引擎**（用户 2026-10-03 ✓ 原话："想真用别的尺寸：
        #   重新导出那个尺寸的引擎 —— 就不能填写后自动导出吗？不合适的话弹提示指引然后搞个
        #   按钮一键导出也行" ✓）。
        #   ⚠ **不做"填完自动导出"** ✗（这也是用户的备选方案 ✓）：一次导出 = 几分钟的重编译
        #     （TensorRT 调优 kernel ✓），而 `imgsz` 是**逐位敲**进去的 ⇒ `1`/`12`/`128`
        #     每一击都触发一次全量导出 ✓ 而且会把正在跑的实时卡死 ✗ ⇒ 走"确认 + 按钮" ✓。
        self.btn_export_engine = QPushButton("导出该尺寸引擎")
        self.btn_export_engine.setToolTip(
            "按左边「imgsz」里的尺寸**导出一份 TensorRT 引擎**（要等几分钟，界面不卡 ✓）。\n\n"
            "为什么需要它：TensorRT 引擎的输入尺寸是**导出时焊死**的 ⇒ 换尺寸只能重导 ✓。\n\n"
            "导出从哪来：\n"
            "  · 训练从 `models/<名字>.pt` 来 ⇒ 引擎就是它导出的 ⇒ 同目录同名那份 .pt 必须在；\n"
            "  · 只有 `.engine`（没有同名 .pt）时会**指路**、不会瞎猜导哪份 ✗。\n\n"
            "导出的文件名**带尺寸**（`<名字>_<尺寸>.engine` ✓）⇒ 几个尺寸可以共存 ✓；\n"
            "而实时挑权重是**按修改时间取最新的 .engine** ✓ ⇒ 刚导出的那个下次启动自动被选中 ✓。\n\n"
            "⚠ 导出期间**请先停掉实时**：它会重写 `<名字>.engine`，被占用会失败 ✓。")
        self.btn_export_engine.clicked.connect(self._on_export_engine)
        row.addWidget(self.btn_export_engine)

        row.addWidget(QLabel("显示"))
        self.sp_show_fps = NoWheelSpinBox()
        self.sp_show_fps.setRange(1, 120)
        self.sp_show_fps.setValue(int(_live.get("show_fps", 30)))
        self.sp_show_fps.setToolTip(
            "界面画面每秒推多少帧。\n\n"
            "  调高 = 画面更顺滑，但**内容帧率受推理速度限制** —— 推理 33fps 时\n"
            "  显示设 60 也只有 33 个不同帧（其余是重复上一帧）。\n\n"
            "  真想让画面快 ⇒ 先降「imgsz」（960→640 约省 56% ⇒ 推理 22ms→~10ms）。\n"
            "  30 就够看；想顺滑点设 60。")
        self.sp_show_fps.valueChanged.connect(self._save_live_params)
        row.addWidget(self.sp_show_fps)

        row.addWidget(QLabel("设备"))
        self.ed_device = QLineEdit(str(_live.get("device", "0")))
        # ⚠ 别用 setFixedWidth 钉死（§4）：钉死会把整行的最小宽度顶上去
        self.ed_device.setMinimumWidth(40)
        self.ed_device.setMaximumWidth(64)
        row.addWidget(self.ed_device)
        row.addStretch(1)
        root.addLayout(row)

        # 行 2（抓帧与显示；「抓帧fps」只在窗口来源显示 ✓）—— 这一行到这儿结束
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        # 本地窗口：抓帧频率
        self.lbl_capfps = QLabel("抓帧fps")
        self.sp_capfps = NoWheelSpinBox()
        self.sp_capfps.setRange(1, 60)
        self.sp_capfps.setValue(int(_live.get("capture_fps", 60)))
        self.sp_capfps.setToolTip("本地窗口每秒抓几帧去推理。\n窗口画面本身可能只有 60fps，抓太高浪费，\n15~30 对角色/怪物识别通常足够。")
        row2.addWidget(self.lbl_capfps)
        row2.addWidget(self.sp_capfps)

        self.ck_draw = QCheckBox("画框")
        self.ck_draw.setChecked(bool(_live.get("draw", True)))
        row2.addWidget(self.ck_draw)

        self.ck_half = QCheckBox("FP16")
        self.ck_half.setChecked(bool(_live.get("half", True)))
        self.ck_half.setToolTip(
            "半精度推理（FP16）：GPU 上比 FP32 快 30~50%，精度几乎无损。\n"
            "  老显卡 / 某些模型不支持时取消勾选（退回 FP32）。\n"
            "  和「imgsz」降低**叠加生效**：640 + FP16 ⇒ 推理可能从 22ms 降到 ~6ms。")
        self.ck_half.stateChanged.connect(self._save_live_params)
        row2.addWidget(self.ck_half)

        row2.addStretch(1)
        root.addLayout(row2)

        # 行 3（延迟探针：开关 / 采样几何显示 / 框选）—— 这一行到这儿结束
        row3 = QHBoxLayout()
        row3.setSpacing(6)

        self.ck_probe = QCheckBox("延迟探针")
        self.ck_probe.setChecked(bool(get("probe", "enabled", True)))
        self.ck_probe.setToolTip(
            "解码 A 机屏幕上的时间码，测真实端到端延迟。\n"
            "需要：A 机跑 python -m tools.probe_gen\n"
            "      B 机跑过一次对时（A 机 IP 见 config/link.yaml 的 a_host）：\n"
            "        python -m tools.clock_sync --host <A机IP> --save\n"
            "      ⚠ 对时偏置会整段加进延迟里（偏置旧了延迟数就整体偏高/偏低 ✗）——\n"
            "        状态行上会把它一并显示出来，体感对不上时先重对一次时。\n\n"
            "不启用时延迟显示为 ——，因为 pts 推算只能反映网络抖动，测不出真实延迟。")
        row3.addWidget(self.ck_probe)

        self.ck_probe_box = QCheckBox("探针框")
        self.ck_probe_box.setChecked(True)
        self.ck_probe_box.setToolTip(
            "在实时画面上把采样几何画出来（框 + 每个采样点），不用去命令行看：\n\n"
            "  · 框的颜色就是判据的结论 ——\n"
            "      绿 = 几何可用（时间戳单调、值也合理）；\n"
            "      红 = 几何可疑（时间戳乱跳 / 整片挪位）；\n"
            "      黄 = 判据还在攒样本；灰 = 探针没启用。\n"
            "  · 每个十字是该块的采样点，颜色跟着判成 0/1 变（黄=白块、蓝=黑块）；\n"
            "      变紫 = 这个点压在黑白之间，也就是没对准。\n"
            "  · 框角上有两个圆圈 = 头两个固定标记块（白、黑）。\n\n"
            "框没括住整条码带、或尾部十字开始发紫，就是几何不对 —— 点「框选探针」重框。")
        row3.addWidget(self.ck_probe_box)

        # 探针几何：人工框选（秒级、当场验证）—— 「调完探针大小，收流位置全靠猜」的解药。
        # 与 HP/MP 条同一套框选交互（在实时画面上拖矩形），但换算不同：
        # 方块带是 n = 2+bits 个等大等距方块，头两个是固定标记（白、黑），
        # 所以框完能当场解码验证；解不出就不保存（宁可不改，也别改坏）。
        self.btn_probe_box = QPushButton("框选探针")
        self.btn_probe_box.setToolTip(
            "在实时画面上框住整条时间码方块带（含最左边那个常亮的白块）。\n"
            "框完当场解码验证：解得出时间码才保存。\n"
            "存进 config/probe_calib.json，存的是画面比例 —— 换分辨率/窗口不用重标；\n"
            "保存后每秒自动生效，不用重开预览。解不出会提示检查什么，且不修改任何配置。")
        self.btn_probe_box.clicked.connect(self._pick_probe)
        row3.addWidget(self.btn_probe_box)

        row3.addStretch(1)
        root.addLayout(row3)

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
        # ⭐⭐ **推图那一行常显**（用户 2026-10-04 ✓ 原话："实时面板常显『推出的是哪张图 /
        #   A 收没收下』"✓）。为什么要单独一行（现场教训 ✓）：以前只在**被拒**时才吭声 ✗，
        #   而且只 `print` 到 stdout（重定向后还会被**缓冲**住 ✗）⇒ 人以为"我改了项目地图、
        #   A 应该收到了"✓ 实际 A 一直收的是**旧图**、回回被拒 ✓ ⇒ 全程"没反应" ✓。
        #   ⚠ 只读、不参与逻辑 ✓（数据来自线程的 `mmap_push_state()` ✓ 见那边说明 ✓）。
        self.lbl_mmap = QLabel("小地图：—")
        self.lbl_mmap.setStyleSheet("color:#5f6368;")
        self.lbl_mmap.setWordWrap(True)
        root.addWidget(self.lbl_mmap)

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
            self.cmb_weights.clear()
            self.cmb_weights.addItem("（先打开一个项目）", "")
            return

        # ⭐ **手选的优先、没选过才自动挑**（用户 2026-10-04 ✓ 见那个下拉的说明）：
        #   存的地方是项目自己的 `live.weights` ✓（同「基础权重」记在 `train.model` 的口径 ✓）
        #   —— 不存成"全局一份"：换个项目就该换模型 ✓。
        self._fill_weight_options(str(project.sec("live").get("weights") or ""))

    #: 「权重」下拉里**自动那项**的 userData（选它 = 清掉手选、回到"自动挑最新" ✓）
    WEIGHTS_AUTO = "\x00auto"
    #: ……「自定义 / 浏览…」那项（选中它会弹文件选择器 —— 它不是个真的权重路径 ✓）
    WEIGHTS_CUSTOM = "\x00custom"

    def _auto_weights(self, project=None):
        """**自动挑**用哪份权重（老行为，口径只此一处 ✓）。

        ⭐ `.engine` 优先于 `.pt`（2026-10-02 ✓）：导出过 TensorRT 引擎的项目自动选它
        （推理快 3-5 倍）；没导出过的退回 `.pt`；再退回 `runs/**/weights/best.pt` ✓。
        一律按**修改时间**取最新（不能按文件名排：`detect_v1` < `mob_v1`，
        取 `[-1]` 会选中旧的单类模型 ✗）。
        """
        p = self.project if project is None else project
        if p is None:
            return ""
        for pat in ("models/*.engine", "models/*.pt", "runs/**/weights/best.pt"):
            got = sorted(p.root.glob(pat), key=lambda q: q.stat().st_mtime,
                         reverse=True)
            if got:
                return str(got[0])
        return ""

    def _weight_candidates(self, project=None):
        """「权重」下拉的候选，按"你会想用哪一份"排；**同一路径只出现一次** ✓。

        ⚠ 口径与训练卡片「基础权重」/ ④ 那张卡的「YOLO 权重」**同一处**（都用
          `tools.yolo_augment.list_weights()` ✓）—— 不另写一份扫描 ✗。
        ⚠ `.engine` **单列一组**：`list_weights()` 只列 `.pt` ✗，而实时这边引擎才是首选 ✓
          （几个尺寸的引擎是**共存**的：`<名字>_<尺寸>.engine` ✓，都得能选到 ✓）。
        回 `[(label, path), …]`（不含「自动 / 自定义」那两项 ✓）。
        """
        p = self.project if project is None else project
        out, seen = [], set()

        def _add(label, path):
            key = str(path)
            if key and key not in seen:
                seen.add(key)
                out.append((label, key))

        if p is not None:
            for q in sorted(p.root.glob("models/*.engine"),
                            key=lambda x: x.stat().st_mtime, reverse=True):
                _add("%s　（引擎 · 本项目）" % q.name, q)
            _mine = sorted(p.root.glob("models/*.pt"),
                           key=lambda x: x.stat().st_mtime, reverse=True)
            _mine += sorted(p.root.glob("runs/**/weights/best.pt"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
            for q in _mine:
                _add("%s　（本项目）" % q.name, q)
        try:
            from tools.yolo_augment import list_weights
            trained = list_weights()
        except Exception:                            # noqa: BLE001
            trained = []
        for label, path, pid in trained:
            _add(label + (("　· 玩家:%s" % pid) if pid else ""), path)
        for q in sorted(ROOT.glob("*.pt")):
            _add("%s　（官方预训练）" % q.name, q)
        return out

    def _fill_weight_options(self, cur=None):
        """填「权重」下拉 —— ⚠ **先填候选、再回填取值**：`findData` 找不到会**静默不选中** ✗
        ⇒ 界面显示的是别的权重、实际用的却是存的那个（这类不一致最难查 ✓；
        同训练卡片 `_fill_model_options` 那条说明 ✓）。
        """
        if cur is None:
            cur = self._weights_chosen()
        self.cmb_weights.blockSignals(True)
        try:
            self.cmb_weights.clear()
            _auto = self._auto_weights()
            self.cmb_weights.addItem(
                "（自动：本项目最新）" + ("　" + Path(_auto).name if _auto
                                        else "　· 还没有可用的"),
                self.WEIGHTS_AUTO)
            _cands = self._weight_candidates()
            if _cands:
                self.cmb_weights.insertSeparator(self.cmb_weights.count())
            for label, path in _cands:
                self.cmb_weights.addItem(label, path)
            self.cmb_weights.insertSeparator(self.cmb_weights.count())
            self.cmb_weights.addItem("（自定义 / 浏览…）", self.WEIGHTS_CUSTOM)
            if cur and self.cmb_weights.findData(cur) < 0:
                # 存着的那份**不在候选里**（被删了 / 从别的机器拷来的）⇒ 补一项、**别吞掉** ✗
                # （⚠ 补的这项要标明"文件可能不在了"，不然人以为它还在用 ✓）
                _miss = "" if Path(cur).is_file() else "　（文件不在了）"
                self.cmb_weights.insertItem(
                    0, "（当前填写）%s%s" % (Path(cur).name, _miss), cur)
            i = self.cmb_weights.findData(cur) if cur else 0
            self.cmb_weights.setCurrentIndex(i if i >= 0 else 0)
        finally:
            self.cmb_weights.blockSignals(False)
        self._weights_prev = cur

    def _weights_chosen(self):
        """项目里**手选**的那份（没选过 = `""` ⇒ 走自动挑 ✓）。"""
        if self.project is None:
            return ""
        try:
            return str(self.project.sec("live").get("weights") or "")
        except Exception:                            # noqa: BLE001
            return ""

    def cur_weights(self):
        """**这次推理实际会用哪份权重** —— 唯一出口 ✓（`start()` 与导出引擎都读它 ✓）。

        ⚠ 只许这一处：两处各读一个控件（或各写一遍自动挑的逻辑）迟早会不一致 ✗ ——
          那正是"界面显示 A、实际用 B"这类最难查的毛病的来源 ✓。
        """
        w = str(self.cmb_weights.currentData() or "")
        if w == self.WEIGHTS_AUTO or w in (self.WEIGHTS_CUSTOM, ""):
            # 停在「自动」/「自定义」（没选完）/ 还没填过候选 ⇒ 都算自动 ✓
            return self._auto_weights()
        return w

    def _on_weight_pick(self, _idx=None):
        """下拉选择变化：**自定义** ⇒ 弹文件选择；**自动** ⇒ 清掉项目里手选的那份；
        其它 ⇒ 按项目记住它 ✓（`project.yaml` 的 `live.weights` ✓）。
        """
        w = self.cmb_weights.currentData()
        if w == self.WEIGHTS_CUSTOM:
            start = str(self._weights_prev or self._auto_weights() or "")
            path, _f = QFileDialog.getOpenFileName(
                self, "选一份模型权重",
                (str(Path(start).parent) if start else str(ROOT)),
                "模型权重 (*.pt *.engine);;PyTorch 权重 (*.pt);;"
                "TensorRT 引擎 (*.engine);;所有文件 (*)")
            if not path:
                i = self.cmb_weights.findData(self._weights_prev)   # 取消 ⇒ 还原
                self.cmb_weights.setCurrentIndex(i if i >= 0 else 0)
                return
            if self.cmb_weights.findData(path) < 0:
                self.cmb_weights.insertItem(self.cmb_weights.count() - 1,
                                            "（自定义）%s" % Path(path).name, path)
            self.cmb_weights.setCurrentIndex(self.cmb_weights.findData(path))
            self._weights_prev = path
            self._remember_weights(path)
            return
        if w == self.WEIGHTS_AUTO:
            self._weights_prev = ""
            self._remember_weights("")
            return
        self._weights_prev = str(w or "")
        self._remember_weights(str(w or ""))

    def _remember_weights(self, path):
        """把手选结果**按项目**记住（`""` = 清掉、回到自动 ✓）。

        ⚠ 存不上**不许拦人**（只读的项目目录 / yaml 坏了 ✓）：本次选择照旧生效
          （`cur_weights` 读的是控件 ✓），只是下次打开会退回自动挑 ✓。
        """
        p = self.project
        if p is None:
            return
        try:
            p.sec("live")["weights"] = str(path or "")
            p.save()
        except Exception as e:                       # noqa: BLE001
            print("⚠ 权重选择没存进项目（%s: %s）" % (type(e).__name__, e))

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
                show_fps=self.sp_show_fps.value(),
                half=self.ck_half.isChecked(),
            )
        except Exception:
            pass

    def start(self):
        # ⭐⭐ **先把地图 id 从项目重读一次**（用户 2026-10-04 ✓ 现场：项目里改了 map_id、
        #   面板却还拿旧图去推 ⇒ A 一直回 `no-such-map` ⇒ 全程没反应 ✓ 见 `_sync_mmap_map_id`）
        self._sync_mmap_map_id()
        # 那一行**开局先填上**（不等第一份统计 ✓）：人一点「开始」就能看见"要推的是哪张图" ✓
        self._show_mmap_push({"want": self._mmap_mid, "ok": "", "reply": "",
                              "src": self._mmap_src})
        # 旧线程可能还在后台退出（已 stop），直接丢弃引用重开
        if self.thread is not None:
            self.thread.stop()
            self.thread = None

        w = self.cur_weights()                       # ⭐ 唯一出口（见那边的说明 ✓）
        if not w:
            self.lbl_stats.setText("没有可用的模型权重 —— 请先完成 ⑦ 训练")
            return
        # ⚠ 手选的那份可能已经**被删 / 被移走**（下拉里会给它标「文件不在了」✓）⇒ 在这儿
        #   就把话说清，别等 ultralytics 抛一长串栈 ✗。⚠ 只查"**带目录**的路径"：裸名字是
        #   **官方权重名**（`yolo26n.pt` 这种），ultralytics 自己会去联网下载 ✓，
        #   拿 `is_file()` 拦它会误伤 ✓。
        _wp = Path(w)
        if not _wp.is_file() and str(_wp.parent) not in ("", "."):
            self.lbl_stats.setText("这份权重不在了：%s —— 请重新选一份（或先完成 ⑦ 训练）"
                                   % _wp.name)
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
            show_fps=self.sp_show_fps.value(),
            half=self.ck_half.isChecked(),
        )

        common = {
            "weights": w,
            "conf_mob": self.sp_conf_mob.value(),
            "conf_player": self.sp_conf_player.value(),
            "imgsz": self.sp_imgsz.value(),
            "device": self.ed_device.text().strip() or "0",
            "draw": self.ck_draw.isChecked(),
            "half": self.ck_half.isChecked(),
            "perf_log": bool(load_live().get("perf_log", True)),
            "show_fps": float(self.sp_show_fps.value()),
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
        # 两路帧信号（见 live_thread 里那两条说明）：
        #   · `raw_frame_ready` = **原生帧**（没画过任何东西）⇒ 存起来给 `current_frame()`
        #     用，框选/测量/叠图核对都读它 ✓；
        #   · `frame_ready` = **显示帧**（画好框的那份）⇒ 只用来画，**不许**进 `current_frame()` ✗。
        self.thread.raw_frame_ready.connect(self._on_raw_frame)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.stats_ready.connect(self._on_stats)
        self.thread.potions_ready.connect(self.potions_ready)
        self.thread.failed.connect(self._on_failed)
        # ⭐ 「这份权重不是本项目训的」⇒ 状态行 + 弹一次（2026-10-03 ✓ 见那个槽的说明 ✓）
        self.thread.weights_warn.connect(self._on_weights_warn)
        # ⭐ 「imgsz 填的值与引擎输入不符 ⇒ 已按引擎的跑」⇒ 弹**指引**（用户 2026-10-03 ✓
        #   见 `LiveThread.imgsz_mismatch`：光 print 进控制台人看不到 ✗）。
        self.thread.imgsz_mismatch.connect(self._on_imgsz_mismatch)
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
        # ⭐ 顺手再重读一次地图 id（用户 2026-10-04 ✓ 同 `start` ✓）：万一"开始"之后
        #   才改了项目地图，第二次重读能把新的推过去 ✓（成本只是一次 dict 读 ✓）。
        self._sync_mmap_map_id()
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

    def _on_raw_frame(self, img):
        """收到**原生帧**（从流里解出来、**一个字都没画**）⇒ 存引用，给 `current_frame()`。

        **为什么单开一路**（用户 2026-09-27 现场 bug）：取帧做测量的那几个功能量的都是
        **像素** —— 探针标定（灰度解码）、HP/MP 条框选、小地图框选、双点/模板标定弹窗、
        叠图匹配分核对。以前它们拿到的和显示用的是**同一份**（上面画着玩家蓝框、怪物绿框、
        攻击线，以及**视野灰色虚线**）⇒ 框选把那条虚线一起框了进去、探针采样被框线污染、
        匹配分虚高 ✗。现在线程分两路发（`raw_frame_ready` / `frame_ready`），这里只存
        **原生**那份 ✓。

        ⚠ 不画、不拷、不排队：只存个引用（微秒级）⇒ 不受显示限流与绘制合并影响，
          **面板不可见时也照更新** —— 不然切回来看一眼再框选，框到的是旧画面 ✗。
        """
        self._last_bgr = img

    def _on_frame(self, img):
        """收到**显示帧**（已画好检测框/攻击线/视野虚线）：**只留最新一帧**，渲染跟不上就丢中间帧，绝不排队。

        **为什么要合并**（"失焦就卡"的主因之一）：这个槽跑在 GUI 主线程上，
        每帧要做两次整幅拷贝（QImage + QPixmap）加一次缩放 —— 1920×1080 大约
        10~20 ms；而线程按 30 fps 推信号。主线程只要慢一点（**失焦时被系统降级**、
        窗口正在缩放、机器在跑训练），队列就**越堆越长**：画面越来越滞后、
        鼠标点一下半天才响应。这个延迟发生在主线程队列里，工作线程侧的统计
        根本看不到（见 `_render` 的注释）—— 所以只看监控数字一切正常。

        合并办法：最新帧放 `_pending`，只挂**一个**零延时任务去画；画之前又来
        新帧就只覆盖 `_pending`（等于"追赶时不补旧帧"）。面板不可见（切到别的
        页签、窗口被藏起来）连画都不画：画面没人看，省下的时间留给决策回路。

        ⚠ 这一份**只用来显示**（它上面画着框和虚线）。**别在这里写 `_last_bgr = img`** ✗
          —— 那是"框选框到视野虚线"那个 bug 的写法；取帧一律走 `_on_raw_frame` ✓。
        """
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
        # ⭐⭐ **帧驱动这条才限频**（用户 2026-10-06 ✓ 见 `RENDER_MIN_MS` 与 `_render` 的说明 ✓）：
        #   工作线程按 `proc_fps`(**39** ✓ 实测 ✓) 往主线程推帧 ✓，主线程还要管面板/日志/菜单 ✓
        #   ⇒ 追不上就**排队** ✗（现场正是"预览 23.5fps + 整个窗口发木"✓，而 `draw_ms`
        #   只有 0.02 ms ✓ ⇒ **不是画得慢** ✗）。
        #   ⇒ 这里把"画的次数"压到 `RENDER_MIN_MS` 一次 ✓：中间来的帧**照旧更新 `_last_pix`** ✓
        #     （下面几行统计也照旧走 ✓），只是**攒到点再画一次** ✓ ⇒ 主线程开销封顶 ✓、
        #     画面永远是最新那一帧 ✓（同收流那条"只取最新"口径 ✓）。
        #   ⚠ 被跳过的那次由 `QTimer.singleShot` 兜底 ✓（保证最后一帧一定画得出去 ✓）。
        if (time.monotonic() - self._last_render_at) * 1000.0 < RENDER_MIN_MS:
            if not self._render_pending:
                self._render_pending = True
                QTimer.singleShot(int(RENDER_MIN_MS) + 1, self._render_flush)
        else:
            self._render()
        self._draw_ms = (time.perf_counter() - t0) * 1000.0
        self._disp_drawn += 1

    def current_frame(self):
        """返回最近一帧**原生画面**（BGR ndarray）；还没有画面时返回 None。

        ⚠ **原生 = 从推流里解出来、没画过任何东西的那一份**（用户 2026-09-27 要求）：
          它喂的都是"拿像素做测量"的功能 —— 探针标定（灰度解码）、HP/MP 条框选、
          小地图框选、双点/模板标定弹窗、叠图匹配分核对。显示帧（画着检测框/攻击线/
          **视野虚线**）**不许**从这里出去 ✗：那正是"框选小地图把虚线一起框进去"的来源。

        **不受绘制合并/显示限流影响**：被合并、因为不可见没画的帧，这里照样是最新的
        —— 由 `_on_raw_frame` 单独维护（见那儿）。
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
            # ⚠ **这里必须"立刻画"**（`_render_now` ✗ 不能用限频那个 `_render` ✓）：
            #   限频是给**帧驱动**那条路（`_on_frame` ✓ 39 次/秒 ✓）准备的 ✓；
            #   而"挂/卸叠图、换那行读数"是**人的动作**（几十次/秒都不到 ✓）⇒
            #   推迟它只会让"点了没反应"或"测试看到的还是上一张"✗。
            self._render_now()      # 立刻擦掉上一层，别等下一帧
            return
        # `note`：叠在**框选那块的下方**的一行字（玩家世界坐标，见 路线识别面板）。
        # 一起存进来，因为每次重画时都要重新贴上 —— 帧是新的，文字也得跟着重贴。
        # ⚠ note **原样**存（别 str()）：它可能是多行列表（当前任务 / 定时任务，
        # 见 _draw_note）—— 套了 str 就会把整个列表画成一行字面量。
        self._ov_minimap = (pix, src_rect, tuple(int(v) for v in frame_rect),
                            float(alpha), note or "")
        self._render_now()          # ⚠ 人的动作 ⇒ 立刻画（同上，不限频 ✓）

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
        self._ov_minimap = tuple(ov[:4]) + (note or "",)   # 同上：原样存，可能是列表
        self._render_now()          # ⚠ 同上：这是"换一行字"，立刻画 ✓（不限频 ✓）
        return True

    def _render(self):
        """把**最新那一帧**画上去 —— 带**限频**（同一小段时间里的多次请求合并成一次 ✓）。

        ⭐⭐ 为什么要限频（用户 2026-10-06 ✓ 现场原话："**视频预览只有 10~13fps，
          整个窗口都发木**"✓；重启调优后被拾到 **23.5fps** ✓，还是不够流畅 ✓）：
            实测（`perf.log` ✓）`proc_fps` 中位 **39** ✓ ⇒ **管线产出没问题** ✓；
            而 `draw_ms` 中位 **0.02 ms** ✓ ⇒ **也不是画得慢** ✗ ——
            是工作线程按 39 帧/秒往主线程**推信号** ✓，而主线程还要处理面板/日志/菜单 ✓
            ⇒ 追不上就**排队** ✗ ⇒ 两个现象一起来：预览只剩二十来帧 ✓、窗口发木 ✓
            （下面那段注释里 2026-09 就记过同一件事："信号队列越堆越长，画面延迟几百毫秒"✓）。
          ⇒ 这里把重绘**限到 `RENDER_MIN_MS` 一次** ✓：期间来的请求**只更新 `_last_pix`**
            （`_on_frame` 照旧写 ✓）、到点**合并成一次**重绘 ✓ ⇒ 主线程每帧开销**有上限** ✓，
            而且每次画的都是**当时最新**那一帧 ✓（陈旧帧直接丢 ✓ —— 与收流那条"只取最新"同一口径 ✓）。
          ⚠ 保证"最后一帧一定画得出去" ✓：被推迟的那次由 `QTimer.singleShot` 兜底 ✓
            （没有新帧也会补画 ✓）。
        """
        if self._last_pix is None:
            return
        # ⚠⚠ **限频不放在这儿**（如实记 ✓）：我第一版就放这儿 ⇒ 当场把 `selftest_main_window`
        #   从 **12/12** 打到 **6/12** ✗ —— 因为这个函数**很多调用方**都用（挂/卸叠图、
        #   换那行读数、可见性切换 ✓），放这儿等于**连"人动了立刻要看见"的那些也一起推迟** ✗。
        #   ⇒ 限频只放**帧驱动那一條**（`_on_frame` ✓ 39 次/秒那条 ✓ 见它那几行 ✓）；
        #     本函数**保持"立刻画"** ✓ —— 老行为一个字不变 ✓。
        self._render_now()

    def _render_flush(self):
        """被推迟的那次到点了 ⇒ 照常画（`_render` 里会重新判一次限频 ✓）。"""
        self._render_pending = False
        self._render_now()

    def _render_now(self):
        """真正画一次（**只有这里**碰 Qt ✓）。

        ⚠⚠ **必须自己守一道"还没有帧"**（如实记 ✓）：推迟重绘那条（`_render_flush` ✓）
        是**直接**调本函数的 ✓ ⇒ 它绕过了 `_render` 顶上那道 `_last_pix is None` 的判断 ✗
        ⇒ 实测当场炸 `'NoneType' object has no attribute 'scaled'` ✓（`selftest_main_window`
        6/12 ✗）。守在这儿，两条入口就都安全了 ✓。
        """
        if self._last_pix is None:
            return
        self._last_render_at = time.monotonic()
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
        """在小地图那块框的**下方**贴读数（世界坐标 / 当前任务 / 定时任务）。

        **两种形态**，由 `note` 的类型决定（2026-09-26 用户要求）：
          · `str` —— 一行**黑底白字**（世界坐标那行：花画面上要有底才读得清）；
          · `list[(文本, 颜色, 是否铺底)]` —— 多行、**默认不铺底色**（"这些文本不要
            背景色"）⇒ 靠四向描边保证可读性，字色由「设置 → 定时任务颜色」给。

        为什么画在叠加层**之后**：它是读数，被半透明的地形图压住就看不清了。
        放不下就翻到框的上方 —— 框本来就贴着画面下沿时，写字会掉出画面外
        （这个项目里标签一律这么处理）。
        """
        from PyQt5.QtGui import QColor, QFont
        if not note:
            return
        lines = list(note) if isinstance(note, (list, tuple)) else [note]
        p.setOpacity(1.0)
        #: 基准字号（原来的那个，**没点名要变大的行**都用它 ✓）。
        _BASE_PT = 9
        #: ⭐ **「当前任务」+ 它下面那行说明的字号**（用户 2026-09-28 要求 ✓ 原话：
        #:   "当前任务以及下一行缩进的说明字体稍微大点"）—— 原 9 ⇒ 抬到 11（"稍微"✓，
        #:   再大就顶到画面了 ✗）。
        _TASK_PT = 11
        f = QFont()
        f.setPointSize(_BASE_PT)
        f.setBold(True)
        # ⭐ **逐行字号**（用户 2026-09-28 ✓）：行元组从 3 项变 **4 项** ——
        #   `(文本, 颜色, 是否铺底, 字号pt)` ✓；老的三项 / 纯 `str` 照旧 ✓（没给字号 ⇒
        #   用基准 9 ✓ **向后兼容**，别的调用点一个字都不用改 ✓）。
        #   · 只让**任务那两行**变大：世界坐标那行是**读数**（黑底白字、用来排查），
        #     跟着变大反而是噪音 ✗（用户只点名了任务那两行 ✓）。
        #   · 行高 `lh` 按**大字**算 ⇒ 所有行的格子一样高、各自垂直居中 ✓
        #     上下不会挤在一起 ✓（不这么算的话，大字那行会压住下面那行 ✗）。
        fbig = QFont(f)
        fbig.setPointSize(_TASK_PT)
        p.setFont(fbig)
        lh = p.fontMetrics().height() + 2
        p.setFont(f)
        fm = p.fontMetrics()
        _cur_pt = _BASE_PT
        x = max(0, int(dst.left()))
        y = int(dst.bottom()) + 4
        # 兜一道：每行都**不换行**，太长会横穿整个画面还看不清。调用方本来就该只给
        # 短句（见 route_panel._tick_world 的 _say），但这里也不能靠"上游会给短的"活着。
        max_w = max(60, p.device().width() - x - 2)
        if y + lh * len(lines) > p.device().height():
            y = max(0, int(dst.top()) - lh * len(lines) - 4)   # 下方放不下 → 翻到上面
        for i, row in enumerate(lines):
            if isinstance(row, str):
                text, color, bg, pt = row, None, True, 0
            else:
                text = str(row[0])
                color = row[1] if len(row) > 1 else None
                bg = bool(row[2]) if len(row) > 2 else True
                pt = int(row[3]) if len(row) > 3 else 0
            if not text:
                continue
            if (pt or _BASE_PT) != _cur_pt:         # ← 这行换字号了 ⇒ 换字体 + 重量宽度 ✓
                p.setFont(fbig if pt else f)
                fm = p.fontMetrics()
                _cur_pt = pt or _BASE_PT
            if fm.horizontalAdvance(text) + 8 > max_w:
                text = fm.elidedText(text, Qt.ElideRight, max(20, max_w - 8))
            box = QRect(x, y + i * lh, fm.horizontalAdvance(text) + 8, lh)
            if bg:
                p.fillRect(box, QColor(0, 0, 0, 150))
            p.setPen(QColor(0, 0, 0))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):   # 描边
                p.drawText(box.adjusted(dx + 4, dy, dx + 4, dy),
                           Qt.AlignLeft | Qt.AlignVCenter, text)
            p.setPen(QColor(color) if color else QColor(255, 255, 255))
            p.drawText(box.adjusted(4, 0, 4, 0),
                       Qt.AlignLeft | Qt.AlignVCenter, text)

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
        # ⭐ 测谎/掉线弹窗的**报警音**（M1，2026-09-29 从 live_thread 迁来 ✓）：
        #    「combat → 非combat」的过渡播一次「触发音效」（挂机保护页可配 ✓）。
        #    QMediaPlayer 必须在 **GUI 线程** ⇒ 收流线程发不了，放这里正好 ✓；
        #    流程内的界面切换（登录→选频道→排队…）不连响 ✓。
        _scr = str(s.get("screen") or "combat")
        _prev = getattr(self, "_alarm_prev_screen", "combat")
        self._alarm_prev_screen = _scr
        if _scr != "combat" and _prev == "combat":
            play_sound(getattr(settings, "lie_alarm_sound", ""))
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
            off = s.get("clock_offset_ms")
            if off is not None:
                # **把"对时偏置"一并摆出来**（用户 2026-09-26 要求）。
                #
                # 为什么必须露出来：这个延迟是「A 机画码那一刻 → 本机收到并解出这一刻」，
                # 计算时**整段加上了对时偏置**（`t_A = t_B + offset`）⇒ 偏置旧了或那次测得
                # 不准，延迟数就**整体**高/低那么多 ✗。实测就栽在这：偏置 1144.5 ms 而显示
                # 1255 ms ⇒ 那个"锁死的 1.25 秒"里几乎全是偏置，链路其实只有百毫秒级 ✗。
                # 摆出来之后"1255 ≈ 1144"一眼可见 ✓ —— 比事后猜半天有用得多 ✓。
                d_txt += "　（含对时偏置 %+.0f ms）" % float(off)
                self.lbl_stats.setToolTip(
                    "端到端延迟 = A 机「画时间码那一刻」→ 本机收到并解出这一刻。\n"
                    "计算时**整段加上了对时偏置**（当前 %+.0f ms，来自 config/clock_offset.txt）：\n"
                    "  · 偏置是准的 ⇒ 上面这个数就是真延迟 ✓；\n"
                    "  · 偏置旧了 / 那次测得不准 ⇒ 延迟数会**整体**偏高或偏低那么多 ✗\n"
                    "    （实测踩过：偏置 1144 ms，把百毫秒级的链路显示成 1255 ms ✗）。\n\n"
                    "怎么确认偏置值不值得信（10 秒）：\n"
                    "  1. A 机屏和本机预览**同框**比一眼（差多少就是真延迟）；\n"
                    "  2. 或重跑一次对时 —— 数字变了就说明原来那份旧了：\n"
                    "     python -m tools.clock_sync --host <A机IP> --save\n"
                    "     （对时结果 5 秒内自动生效，**不用重开预览**。）" % float(off))
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
        # 载荷告警挂上去（先挂它、后挂积压 —— **代码顺序与显示顺序是反的**：
        # 这里都是往 head 前面拼，所以后拼的显示在最左边）。
        lw = (s.get("load_warn") or "").strip()
        if lw:
            head = "【%s】 " % lw + head
        # 积压锁死放**最前面**（比负载告警还优先）：它不是"为什么变慢"的解释，
        # 而是一个**已经发生的故障状态**（延迟几秒、自己回不来），不处置会一直是坏的。
        # 判据在 gui/live_thread.lag_watchdog（纯函数，已自检）。
        lag = (s.get("lag_warn") or "").strip()
        if lag:
            head = "【%s】 " % lag + head
        # ⭐ 测谎/防挂机弹窗接管中（M1「测谎报警」）：这是「**现在必须人来玩小游戏**」
        #    的报警，比负载/积压都优先 —— 放最前面 ✓（往 head 前拼 = 显示在最左 ✓）。
        #    配套的报警音在 live_thread（进入弹窗状态那一下响 ✓）。
        _scr = str(s.get("screen") or "combat")
        if _scr.startswith("lie"):
            head = "【测谎弹窗 %s：请人工完成小游戏】 " % _scr + head
        elif _scr != "combat":
            # 掉线/登录系（ui_state 的界面 id ⇒ 中文名 ✓）：重连开着时 reconnect_note
            # 会另报它在做哪一步，这里报的是"现在画面在哪个界面" ✓
            from perception import ui_state as _uist
            head = "【界面：%s】 " % _uist.UI_NAMES.get(_scr, _scr) + head
        # ⭐⭐ **小地图权威坐标的"新鲜度"先算好**（用户 2026-10-03 ✓ 他问"我能在…看到验证
        #   结果吗"）：`mmap_age_ms` = 这份世界坐标背后的那一帧是**多久以前收到的** ⇒
        #   提高小地图帧率到底有没有用，看这一个数最直接 ✓（口径见 SKILL **约定 170** ✓）。
        #   `None`（还没定位过 / 来源是实时画面 ⇒ 没有这条推流链）⇒ 显 `—`，别编个 0 ✗。
        #   ⚠ 必须**定义在使用之前**：下面 tooltip 那一段也要用它 ✓（第一版写在状态行那儿 ⇒
        #     pyflakes 当场抓出 `undefined name '_age'` ✗ `selftest_live_panel` 也红了 ✓）。
        _age = s.get("mmap_age_ms")
        _age_txt = ("%5.1f ms" % float(_age)) if isinstance(_age, (int, float)) else "    —  "
        # 明细放 tooltip：状态行那一行已经塞满了，硬挤进去反而看不清数字
        for key in ("lag_detail", "load_detail", "src_lag_detail"):
            d = (s.get(key) or "").strip()
            if d:
                cur = self.lbl_stats.toolTip()
                self.lbl_stats.setToolTip((cur + "\n\n" if cur else "") + d)
        # ⭐ 小地图那一路的明细也挂 tooltip（用户 2026-10-03 ✓）：状态行里只放最要紧的
        #   「坐标 xx ms」✓，其余（收帧率 / 累计丢帧）放这儿 —— 状态行已经塞满了 ✗。
        if _age is not None or s.get("mmap_rate") is not None:
            _mr = s.get("mmap_rate")
            cur = self.lbl_stats.toolTip()
            _mline = ("小地图（世界坐标的来源）：\n"
                      "  · 坐标延迟 %s —— 这份坐标背后的那一帧**多久以前**收到的\n"
                      "    （越小越新；提到 60fps 之后应该稳定在半帧~一帧，约 8~17 ms ✓）\n"
                      "  · 收帧 %s fps —— **A 机真正推出来**的帧率（不是配置里那个数）\n"
                      "  · 累计丢帧 %s —— 我们没来得及取走就被下一帧覆盖的次数\n"
                      "    （一直涨 ⇒ 消费侧比推流慢，见 SKILL 170）"
                      % (_age_txt.strip() + " ms" if _age is not None else "—",
                         ("%.1f" % float(_mr)) if isinstance(_mr, (int, float)) else "—",
                         int(s.get("mmap_drop", 0) or 0)))
            self.lbl_stats.setToolTip((cur + "\n\n" if cur else "") + _mline)
        # 绘制那一段单独报：「显示 fps」只说明**推**了多少，看不出主线程画得
        # 动不动 —— 而"失焦就卡"恰恰卡在这里（合并丢弃的帧数一涨，就说明主线程
        # 跟不上推送、开始在丢中间帧；不丢帧时界面才不会滞后）。
        # 「卡在谁身上」（2026-09-26）：输入/处理两个数在**不丢帧时必然相等** ⇒
        # "处理速度掉到 20"看着像 B 机算不动，其实常常是 A 机只推了 20。判据在
        # `live_thread.limit_reason`（纯函数、有自检）：说不清时**不贴标签**。
        _limit_txt = {"input": "　⚠ 受 A 机推流限制",
                      "self": "　⚠ 本机算不过来"}.get(s.get("limit") or "", "")
        # ⭐ 「跟不上源 ⇒ 已自动降显示刷新」（用户 2026-10-05 ✓ 见 `live_thread.
        #   src_lag_watchdog`）：**说清它做了什么** —— 否则人看到帧率从 50 掉到 15
        #   只会以为是程序坏了 ✗（明细在 tooltip ✓ 短句只写"降了预览" ✓）。
        if s.get("preview_capped") and _limit_txt:
            _limit_txt += "（预览已自动降到 %g fps）" % float(s.get("preview_fps") or 0.0)
        # ⭐ 推图那一行每来一份状态就刷新一次（用户 2026-10-04 ✓ 常显 ✓ 见 `_show_mmap_push`）
        self._show_mmap_push(s.get("mmap_push"))
        self.lbl_stats.setText(
            "%s%s ｜ 输入 %5.1f fps ｜ 处理 %5.1f fps ｜ 丢帧 %d ｜ 推理 %5.1f ms ｜ "
            "显示 %4.1f fps ｜ 绘制 %4.1f ms 合并丢弃 %d ｜ 检出 %d ｜ 坐标 %s ｜ %d×%d%s"
            % (head, d_txt, s.get("recv_fps", 0), s.get("proc_fps", 0),
               s.get("dropped", 0), s.get("infer_ms", 0), s.get("show_fps", 0),
               self._draw_ms, self._disp_merged,
               s.get("boxes", 0), _age_txt, w, h, _limit_txt))

    def _sync_mmap_map_id(self):
        """把「要给 A 机发的地图 id」从**项目**重读一次 ⇒ 变了就推给实时线程（返回是否变了 ✓）。

        ⚠⚠ 为什么必须有它（用户 2026-10-04 ✓ 现场：手改了项目里的 `map_id`，A 那边"没有任何
          变化"✓）——**权威源只有一个：`project.map_id`** ✓，而面板缓存的 `_mmap_mid` 只在
          `bind(project)` 和"路线页手动推"时才更新 ✗ ⇒ **项目里改了它不知道** ✓ ⇒ 实时线程
          一直拿**旧图**给 A 发 `MAP <旧 id>` ⇒ A 回 `no-such-map` ⇒ 收不到小地图 ⇒
          定位不了玩家 ⇒ 自动不动作 ✓。
          （查现场时的铁证：项目里明明写着 `105040306` ✓，而 `perf.log` 里 B 一直推的是
            **105040303** ✓ 每 5 秒重试一次 ✓。）
        ⚠ 路线页那个 `_map_id()` 读的也是 `project.map_id` ⇒ 重读是**收敛**的 ✓ 不会跟它打架 ✓；
          项目没打开 ⇒ 什么都不做 ✓（也**不写项目** ✗）。
        """
        p = getattr(self, "project", None)
        if p is None:
            return False
        mid = str(p.get("map_id") or "").strip()
        if mid == str(getattr(self, "_mmap_mid", "") or ""):
            return False
        self._mmap_mid = mid
        th = getattr(self, "thread", None)
        if th is not None and hasattr(th, "set_mmap"):
            try:
                th.set_mmap(map_id=mid)   # 实时在跑 ⇒ 当场生效 ✓（见 set_mmap 的说明 ✓）
            except Exception:             # noqa: BLE001 —— 推不动也别崩 ✗
                pass
        return True

    def _show_mmap_push(self, st):
        """把「推给 A 的是哪张图 / A 收没收下」写到常显那一行 ✓（纯展示，不参与判断 ✓）。"""
        st = st if isinstance(st, dict) else {}
        want = str(st.get("want") or "")
        ok = str(st.get("ok") or "")
        reply = str(st.get("reply") or "")
        src = str(st.get("src") or "")
        # ⚠ 这一段是**纯文本**（QLabel 不认 markdown ✗）⇒ 不许写 `**加粗**` ✗（写了就原样显示）
        if src == "live":
            self.lbl_mmap.setText("小地图：来源＝从实时画面取（不走 A 机的推流）")
            self.lbl_mmap.setStyleSheet("color:#5f6368;")
        elif ok and ok == want:
            self.lbl_mmap.setText("✓ 已推给 A 机：%s（A 收下了 ✓）" % ok)
            self.lbl_mmap.setStyleSheet("color:#137333;")
        elif want:
            self.lbl_mmap.setText("⚠ 要给 A 机推的是 %s —— A 说：%s"
                                  % (want, reply or "（还没回话）"))
            self.lbl_mmap.setStyleSheet("color:#b06000; font-weight:600;")
        else:
            self.lbl_mmap.setText("小地图：还没定要推哪张图（打开项目后自动带过来 ✓）")
            self.lbl_mmap.setStyleSheet("color:#5f6368;")
        self.lbl_mmap.setToolTip(
            "这一行说的是「B→A 推图」那条路（小地图）现在的样子：\n"
            "  · 要推的图 = 项目里的 map_id（打开项目/开始实时都会重读 ✓）；\n"
            "  · 收下没 = A 机**明确回过 ok** 才算 ✓（失败每 5 秒重试一次，修好就自愈 ✓）；\n"
            "  · 被拒时那句话就是『缺什么』—— 照着去 A 机「小地图推流 → 框选…」框一次 ✓。")

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
        # ⚠ 这行是**单行标签**（长了会被截 ✗）⇒ 全文另挂 tooltip ✓：像"权重文件不在了…"
        #   那类消息已经把"怎么办"排在**前面**（见 `gui/live_thread.missing_weights_msg` ✓）
        #   ⇒ 被截掉的只是尾巴上的路径，hover 一下照样看全 ✓。
        self.lbl_stats.setText("失败：%s" % msg)
        self.lbl_stats.setToolTip(msg)

    def _on_imgsz_mismatch(self, want, used):
        """引擎输入尺寸 ≠ 你填的 ⇒ **弹窗，点「确定」就把导出跑起来**（用户 2026-10-05 ✓ 原话：
        "现在调成 640imgsz 会弹窗，期望这个弹窗不管写什么，点确定就自动后台完成导出步骤"）。

        ⚠ **非致命**：实时**照常在跑**（按引擎的尺寸 ✓，框的坐标仍然正确 ✓）⇒ 不停 ✓。
        ⭐ **口径已改**（2026-10-03 → 2026-10-05 ✓）：原来这里只"指到按钮、由人再点一次" ✗
          ⇒ 现在**点确定 = 直接后台开导** ✓（用户要的就是"一次点击"✓）。"要不要花几分钟重编译"
          这件事**在这个弹窗里就已经问过了** ✓（所以不再叠一层确认 ✗ —— 那是"点两下"✗）。
        ⚠ 推不出该导哪份 `.pt` / 已有导出在跑 ⇒ 由 `_start_engine_export` **如实说清**、**不瞎猜** ✗。
        """
        self.lbl_stats.setText("⚠ imgsz 填的 %d 不生效（这个引擎是 %d）" % (want, used))
        self.lbl_stats.setToolTip(
            "这个项目用的是 TensorRT 引擎（`models/*.engine` ✓ 实时优先选它 ✓），\n"
            "它的输入尺寸在**导出时**就焊死了：%d ✓。" % used)
        _r = QMessageBox.question(
            self, "imgsz 不生效：这个项目用的是 TensorRT 引擎",
            "引擎的输入尺寸在**导出时**就固定了：**%d**。\n\n"
            "⇒ 你在「imgsz」里填的 **%d 不会生效**；实时已按 **%d** 跑"
            "（框的坐标仍然正确 ✓，只是「缩到多大再喂网络」这个选择没生效）。\n\n"
            "**点「确定」现在就开始导出 %d 的引擎**（后台跑，要几分钟；期间界面仍能用 ✓\n"
            "导出完**下次启动实时会自动选它** ✓）。\n\n"
            "⚠ 导出会重写 `models` 里那份引擎文件；**实时正在用它的话建议先停实时**"
            "（被占用会失败 ✓）。\n"
            "不想现在导 ⇒ 点「取消」，之后随时可点本页「导出该尺寸引擎」✓，\n"
            "或把「权重」改选成 `.pt`（`.pt` 没有尺寸限制 —— 320~2048 随便填 ✓）。\n\n"
            "（本提示只弹这一次；`perf.log` 段头里也会写 `imgsz=%d(引擎·已对齐)` ✓）"
            % (used, want, used, want, used),
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if _r == QMessageBox.Ok:
            self._start_engine_export(want)

    def _engine_export_plan(self, imgsz):
        """「该导哪份 `.pt` / 出什么文件名」⇒ `(pt, out, why)`。

        ⚠⭐ 这是**导出这条路上唯一读权重的地方**（读权重的出口总共两处：`start()` 和这里 ✓
          用例钉着这个数 ✓ 别再加 ✗ —— 说明见 `cur_weights` 那一处 ✓）。
        `pt is None` 时 `why` 是**人话的原因**（由调用方弹框 ✓）—— **推不出来就指路、不猜** ✗
        （口径同 `tools.export_engine.pt_for` ✓）。
        """
        try:
            from tools.export_engine import pt_for
            _pt, _why = pt_for(self.cur_weights())
        except Exception as e:                       # noqa: BLE001
            _pt, _why = None, "%s: %s" % (type(e).__name__, e)
        if _pt is None:
            return None, None, _why
        return (_pt, _pt.with_name("%s_%d.engine" % (_pt.stem, int(imgsz))), "")

    def _start_engine_export(self, imgsz):
        """⭐ **把导出真的跑起来**（后台线程 + 进度窗）→ `True` = 起了 ✓。

        两个调用方**共用这一段**（"一处实现" ✓）：
          · `_on_export_engine`（按钮 ✓ —— 它自己先弹一次确认 ✓）；
          · `_on_imgsz_mismatch`（**点确定 = 直接开导** ✓ 用户 2026-10-05 要求 ✓ ——
            那次确认已经在那个弹窗里问过了 ⇒ **不再叠一层** ✗ 叠了就是"点两下"✗）。
        ⚠ 推不出 `.pt` / 已有导出在跑 ⇒ **如实弹框说清** + 返回 `False` ✓（不静默 ✗）。
        """
        imgsz = int(imgsz)
        _pt, _out, _why = self._engine_export_plan(imgsz)   # ⭐ 读权重只在这一处（见它 ✓）
        if _pt is None:
            QMessageBox.information(self, "先说明这份引擎是从哪份 .pt 导出来的", _why)
            return False
        _t_old = getattr(self, "_engine_thread", None)
        if _t_old is not None and _t_old.isRunning():
            QMessageBox.information(self, "正在导出", "上一次导出还没跑完，等它结束再点。")
            return False
        self.btn_export_engine.setEnabled(False)
        self.btn_export_engine.setText("导出中…")
        self.lbl_stats.setText("正在导出 %d 的引擎…（几分钟，别关窗口）" % imgsz)
        _t = EngineExportThread(_pt, imgsz,
                                device=str(load_live().get("device", "0")))
        _t.done.connect(self._on_export_done)
        _t.failed.connect(self._on_export_failed)
        _t.finished.connect(self._on_export_finished)
        # ⭐⭐ **进度窗**（用户 2026-10-04 ✓ 原话："新的imgsz导出时，数据工作台没有日志和
        #   进度条"）：这几分钟里必须有**日志 + 进度**，不然人不知道它是在跑还是死了 ✗
        #   （见 `EngineExportDialog` ✓）。非模态 ⇒ 界面照旧能用 ✓；
        #   ⚠ 窗开不出来也**不许拦导出** ✗（导出才是正事 ✓）。
        try:
            _old = getattr(self, "_engine_dlg", None)
            if _old is not None:
                _old.close()
            _dlg = EngineExportDialog(imgsz, self)
            _t.line.connect(_dlg.append)
            self._engine_dlg = _dlg
            _dlg.show()
        except Exception as e:                       # noqa: BLE001
            self._engine_dlg = None
            print("⚠ 导出进度窗建不起来（%s: %s）—— 导出照跑，日志看 stdout.log ✓"
                  % (type(e).__name__, e))
        self._export_ok = None
        self._engine_thread = _t
        _t.start()
        return True

    def _on_export_engine(self):
        """⭐ 一键导出「当前 imgsz」对应尺寸的引擎（用户 2026-10-03 ✓ 见那个按钮的说明）。

        三步都**如实**做（不许假装 ✗）：
          ① 从**当前权重**推出该导哪份 `.pt`（`pt_for` ✓ 推不出来就**指路**，不猜 ✗）；
          ② 弹确认框：尺寸 / 输出文件名 / 要等几分钟 / **请先停实时**（引擎会被重写 ✓）；
          ③ 交给 `_start_engine_export`（后台线程 + 进度窗 ✓ 与"imgsz 不生效"那个弹窗**共用** ✓）；
          成功 / 失败**都**弹框 ✓（成功还要说清"什么时候生效" ✓）。
        ⚠ 2026-10-05：**确认只在这里做**（按钮这条 ✓）；`_on_imgsz_mismatch` 那次点击本身就是
          确认 ⇒ 那种情况**不再**走到这里（见那两个方法的说明 ✓）。
        """
        imgsz = int(self.sp_imgsz.value())
        _pt, _out, _why = self._engine_export_plan(imgsz)
        if _pt is None:
            QMessageBox.information(self, "先说明这份引擎是从哪份 .pt 导出来的", _why)
            return
        _ok = QMessageBox.question(
            self, "导出 TensorRT 引擎",
            "按「imgsz」里的 **%d** 导一份引擎：\n\n"
            "  从：%s\n  出：%s\n\n"
            "⚠ 要等**几分钟**（TensorRT 在自动调优 kernel）；期间界面仍能用 ✓。\n"
            "⚠ **请先停掉实时** —— 导出会先重写 `%s.engine`，被占用就会失败。\n\n"
            "开始导出？" % (imgsz, _pt.name, _out.name, _pt.stem),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if _ok != QMessageBox.Yes:
            return
        self._start_engine_export(imgsz)

    def _finish_export_dlg(self, ok, why=""):
        """把进度窗收尾（**窗没建起来 / 已被关掉**都算正常 ⇒ 一律吞异常 ✓）。"""
        dlg = getattr(self, "_engine_dlg", None)
        if dlg is None:
            return
        try:
            dlg.mark_finished(ok=bool(ok), why=why)
        except Exception:                            # noqa: BLE001
            pass

    def _on_export_done(self, path):
        """导出成功：① **当场把下拉切到刚导出的那份**；② 说清**怎么才算生效**。

        **为什么这两件事都得做**（用户 2026-10-04 ✓ 现场：重导 960 之后点「开始推理」⇒
        `FileNotFoundError: '…\\detect_v5.engine' does not exist` ✗）：引擎文件名带尺寸，
        是靠 `tools/export_engine.export_for` 把 ultralytics 写出来的 `<权重名>.engine`
        **`os.replace` 改名**成 `<权重名>_<尺寸>.engine` 得来的 ✓（不这么做几个尺寸没法共存 ✓）
        ⇒ **旧的那个名字当场就没了** ✗。而权重路径是**点「开始」那一刻**定下、带进线程的 ✓
        ⇒ 已经在跑的实时还捏着旧路径 ⇒ 首次加载（点「开始推理」）必然找不到文件 ✗。
        ⇒ 三件一次收口：
          · 「自动」时**刷新下拉**（自动那项此时就指向刚导出的引擎 ✓）；
          · 手选的那份**正是被改名的那个**（`detect_v5.engine` ⇒ `detect_v5_960.engine` ✓）
            ⇒ 把它改成新引擎并存进项目（否则那份选择会**永远指向一个不存在的文件** ✗）；
          · 提示语从含糊的"下次启动生效"改成"**重新点「开始」**"（这正是这次踩空的来源 ✗）。
        """
        self._export_ok = True
        _new = Path(path)
        _cur = str(self.cmb_weights.currentData() or "")
        if _cur in ("", self.WEIGHTS_AUTO):
            self._fill_weight_options()              # 自动 ⇒ 刷新后自动项就指向新的 ✓
        else:
            _old = Path(_cur)
            if not _old.is_file() and _new.name.startswith(_old.stem + "_"):
                # 你选的那份**被改名成了这一份**（`detect_v5.engine` ⇒
                # `detect_v5_960.engine` ✓）⇒ 跟着它（否则那份选择会**永远指向一个
                # 不存在的文件** ✗），并落盘 ✓
                self._remember_weights(path)
                self._fill_weight_options(path)
            else:
                # ⚠ 别的**一律不许动**：你手选的是另一份引擎（它还在）、或者你选的就是
                #   这份 ✓ —— 抢走选择是另一类毛病 ✗（只刷新列表 + 选回原选择 ✓）
                self._fill_weight_options(_cur)
        self.lbl_stats.setText("已导出引擎：%s —— 重新点「开始」即用它" % _new.name)
        QMessageBox.information(
            self, "导出完成",
            "已生成：\n  %s\n\n"
            "**「权重」那一栏已经切到它了** ✓（名字带尺寸 `<名字>_<尺寸>.engine` ✓）。\n"
            "⚠ **要重新点一次「开始」才用它**：权重是在点「开始」那一刻带进推理线程的，\n"
            "   正在跑的实时**不会中途换模型** ✗（还开着就先「■ 停止」再「▶ 开始」）。\n\n"
            "   ⭐ 刚才那个 `FileNotFoundError: … does not exist` 就是这么来的：\n"
            "      重新导出会把**旧的** `<权重名>.engine` **改名**成带尺寸的新名字 ⇒\n"
            "      还在跑的那份实时捏着的旧路径就没了 ✗（现在它会直接说清该怎么办 ✓）。\n\n"
            "⚠ 想同时留着别的尺寸：不用管，文件名带尺寸、互不覆盖 ✓。" % path)

    def _on_export_failed(self, msg):
        self._export_ok = False
        self.lbl_stats.setText("导出失败：%s" % msg[:60])
        QMessageBox.warning(
            self, "导出失败",
            "没导出来：\n\n%s\n\n"
            "常见原因：\n"
            "  · **实时还开着**、`<权重名>.engine` 被占用 ⇒ 先停实时再导 ✓；\n"
            "  · 没装 TensorRT / 显存不够 ⇒ 看控制台那几行（`stdout.log` ✓）。" % msg)

    def _on_export_finished(self):
        self.btn_export_engine.setEnabled(True)
        self.btn_export_engine.setText("导出该尺寸引擎")
        self._engine_thread = None
        # ⭐ 进度窗收尾（**成功/失败都收** ✓）：`_export_ok` 由 `_on_export_done/failed`
        #   设好（信号是队列投递、顺序保持 ✓）⇒ 这里只负责"停下来 + 换成关闭按钮" ✓。
        self._finish_export_dlg(getattr(self, "_export_ok", None) is not False)
        self._export_ok = None

    def _on_weights_warn(self, msg):
        """⭐ 权重**不是本项目训的** ⇒ 状态行写清 + 弹一次（2026-10-03 ✓ 用户现场
        "怎么没有检出框了"）。

        为什么要弹：那个现象（**一个框都没有**）以前**没有任何提示** ✗ —— 人只能猜是
        抓屏坏了 / 实时没开 / 阈值太高 ✓（现场那次真因是"续训退化成了 coco8 ⇒ 发布成
        项目权重 ⇒ 类名全对不上" ✗）。这条只在**开始推理那一下**发一次 ✓，不刷屏 ✓，
        而且**不阻止**实时继续跑 ✓（有的权重只是类名顺序不同，照旧能用 ✓）。
        """
        self.lbl_stats.setText("⚠ 权重与项目对不上（不会检出框）")
        self.lbl_stats.setToolTip(msg)
        QMessageBox.warning(self, "这份权重不是本项目的", msg)

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
