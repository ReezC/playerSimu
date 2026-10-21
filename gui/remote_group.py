"""「远程操控」那一组（设置 → 链路 ✓ 用户 2026-10-15 要求 ✓）：**鼠标标定**。

为什么单独一个文件 ✗（2026-10-15 ✓）：它原来是塞在 `gui/settings_dialog.py` 里的 ✗ ——
那一份本来就已经 1400 多行 ✓，我加完这一组直接顶过 **1500 行** ⇒ `selftest_arch` 的 **E-②**
（不许再冒巨型运行期文件 ✓）当场红 ✓。
⚠ 那条钉子明说的两条出路是「**拆成两件事**」或者「连为什么一起写进 `GIANT_BASELINE`」✓ ——
这里选**拆** ✓：`settings_dialog.py` 管的是"**`link.yaml` 那些格**" ✓，而这一组**一个字节都不写
`link.yaml`** ✗（它写的是 `config/mouse_gain.json` ✓）⇒ 它本来就不属于那一页的数据驱动表格 ✓
（见 `_page_link` 里那段"为什么不塞进字段表"✓）。

⚠ 这里**只管界面** ✓：读写在 `execution/mouse_aim.py`（`calib_status` / `set_auto_measure` /
`auto_measure_async` ✓），而且**唯一写者**是它 ✓ —— 界面自己 `json.dump` 会把 `by_frame` 里
那几份标定**抹掉** ✗（抹了就得每种尺寸重量一次 ✓）。
"""

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QGroupBox, QLabel,
                             QPushButton)

from execution import aim_convert as ma_c

from execution import aim_convert as ma_c

from execution import mouse_aim as ma
from gui.widgets import title_label


class RemoteControlBox(QGroupBox):
    """「远程操控」= 状态行 + 「自动量」勾选框 + 「立即量一次」按钮 ✓。

    ⚠⚠ 这一组的两个控件各有**一层语义**，都写进提示里了（不然用户会以为按钮坏了 ✗）：
      · **「自动量」那一格的真正含义 = 「允许它动鼠标」** ✓ —— 既有设计里，关着的时候
        `auto_measure_async` **一个字节都不发** ✗（见 `mouse_aim.AUTO_MEASURE_DEFAULT` ✓）；
      · **按钮量的是"实时回路正收到的那一帧"的尺寸** ✓ ⇒ 不用用户填数字 ✓，
        但前提是**先开着实时预览** ✓（没有帧就明说，别静静什么都不做 ✗）。
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__("远程操控", parent)
        self._main = main_window          # ⚠ 只用来问"实时回路现在收到多大的帧" ✓
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        tip = QLabel("点服务器 / 点频道 / 自动测谎瞄准全靠它：**这个画面尺寸没量过 ⇒ 鼠标"
                     "一点都不点**（不是点偏 ✗）。\n"
                     "⚠「自动量」会**动鼠标**（撞左上角 → 走一步 → 走回来，全程不点 ✓）"
                     "⇒ 所以默认**关** ✓。")
        tip.setStyleSheet("color: #5f6368;")
        tip.setWordWrap(True)
        form.addRow("", tip)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        form.addRow("标定状态", self.status)

        self.auto = QCheckBox("自动量（这个尺寸没量过就自己量一次）")
        # ⚠ 勾选框的**自己的文字就是字段名** ⇒ 提示挂它身上 ✓（同链路页那张表的规矩 ✓）
        self.auto.setToolTip("勾上 = 允许它在你没量过的画面上**自己量一次**（会动鼠标 ✓，"
                             "量完这个尺寸就一直有效 ⇒ 来回切来源都自动 ✓）。"
                             "不勾 = 只提示「没量过 ⇒ 不点」✓，绝不自己动鼠标 ✓。")
        self.auto.toggled.connect(self._on_auto)
        form.addRow("", self.auto)

        self.btn = QPushButton("立即量一次")
        form.addRow(title_label("立即量一次",
                                "拿**实时回路正收到的那一帧**的尺寸量一次 ✓（量的是它自己，"
                                "不用你填数字 ✓）。⚠ 会动鼠标（不点）✓；⚠ 前提是上面那格已勾上 "
                                "—— 那格的真正含义就是「允许它动鼠标」✓。"),
                    self.btn)
        self.btn.clicked.connect(self._on_measure)

        self.refresh()                    # 建的时候就把现状灌进来 ✓

    # ---- 灌值 / 交互 ----

    def refresh(self):
        """把 `config/mouse_gain.json` 的**现状**灌进这一组 ✓（建组时 + 露头时 ✓）。"""
        try:
            self.status.setText(ma_c.calib_status())
        except Exception as e:            # noqa: BLE001 —— 状态行读不出来也不许把设置页弄崩 ✗
            self.status.setText("读标定状态失败：%s" % e)
        try:
            _on = bool(ma_c.auto_measure_enabled())
        except Exception:                 # noqa: BLE001
            _on = False
        self.auto.blockSignals(True)      # ⚠ 灌值**不许**触发写回 ✗（设置对话框的既有规矩 ✓）
        self.auto.setChecked(_on)
        self.auto.blockSignals(False)

    def _on_auto(self, on):
        """勾选框 ⇒ 写 `mouse_gain.json` 的 `auto_measure` ✓（**只改这一个键** ✓）。"""
        ok, why = ma_c.set_auto_measure(bool(on))
        if not ok:
            # ⚠ 写不进 ⇒ 把勾**退回去** ✗（别让界面显示一个没落盘的开关 ✓）
            self.auto.blockSignals(True)
            self.auto.setChecked(not on)
            self.auto.blockSignals(False)
            self.status.setText(why)
            return
        self.status.setText("%s　｜　%s" % (ma_c.calib_status(), why))

    def _on_measure(self):
        """**立即量一次**：尺寸取"实时回路正收到的那一帧" ✓ ⇒ **后台**量 ✓（不卡界面 / 不卡回路 ✓）。

        ⚠⚠ 2026-10-15 现场报「**点了一点反应都没有**」✗ ⇒ 这一版改成"**点了一定看得见**" ✓：
          ① **先亮明**（按钮置灰 + 文字改成"量中…" ✓ + 状态行写清"会动一下鼠标"✓）；
          ② ⭐ **已经标定过的尺寸也要能量** ✗ —— 这是根子 ✓：当前尺寸多半**早就量过**
             （`load_gain` ✓）⇒ 走"自动量"那条路只会回一句"这个尺寸已经有标定 ⇒ 不用量" ✓
             ⇒ **鼠标不会动** ✗ ⇒ 在用户眼里就是"没反应" ✓。
             ⇒ 量过就用 `auto_measure(…, force=True)` **强制重新量一次** ✓
             （`force` 本来就是给"人明确要求"准备的 ✓ 会跳过冷却 ✓）；
          ③ **量完把结果写回状态行** ✓（走 Qt 信号 ✓ —— 后台线程不许碰控件 ✗）。
        ⚠ 量本身**必须在后台** ✗：它要读两次远端光标（每次最多等 0.8 秒 ✓）⇒ 放界面线程里就是
          "点下去界面卡一两秒" ✓（同 `mouse_aim.auto_measure_async` 那段说明 ✓）。
        """
        _lp = getattr(self._main, "live_panel", None)
        _st = (getattr(_lp, "_last_stats", None) or {}) if _lp is not None else {}
        _w, _h = (_st.get("size") or (0, 0))
        # ⭐⭐ 兜底（2026-10-15 现场 ✓）：`_last_stats` 属于"**设置对话框那一台工作台**"✗ ——
        #   而**开着两个工作台**时，预览完全可能跑在**另一台**上 ⇒ 现场连着两次留痕都是
        #   `size=[0, 0]` ✗ ⇒ 一个字节都没发 ⇒ 用户看到的就是"点了没反应" ✓。
        #   ⇒ 兜底问**实时回路自己报上来的那一份** ✓（`mouse_aim.note_frame_shape` ✓：
        #     谁在收帧谁就报 ✓ 与"哪台工作台的设置对话框"无关 ✓）。
        if not (_w and _h):
            _fs = ma_c.last_frame_shape()
            if _fs:
                _h, _w = int(_fs[0]), int(_fs[1])
        # ⭐⭐ **点一下必须留下痕**（2026-10-15 ✓ 用户连着报了三轮"点了没反应"之后加的 ✗）：
        #   现场最难办的是**分不清**这两件事 ✗ —— 它们在界面上**都是"点了没反应"**：
        #     ① 点击**根本没到**这个函数（还在跑改动前的旧进程 ✗ / 按钮被禁用 ✗）；
        #     ② 到了、但被某道闸挡下（没勾「自动量」/ 拿不到画面尺寸 / 折算比算不出… ✓）。
        #   ⇒ 记一条 `remote_measure_click` ✓ —— 一次点击就能定性 ✓：
        #     · **日志里没有它** ⇒ 就是①（旧进程 / 按钮压根没触发 ✓）；
        #     · `size=[0,0]` ⇒ ②里的"拿不到画面尺寸"✓；
        #     · `auto=false` ⇒ ②里的"没勾「自动量」"✓；
        #     · `auto=true` 且尺寸正常 ⇒ 真去量了 ⇒ 那日志里**必有** `mouse_gain_probe` ✓
        #       （没有 ⇒ 说明量那句压根没被调到 ✗ —— 那是更深的问题 ✓）。
        #   ⚠ 它自己炸了**绝不许**影响这次点击 ✗（同 `mouse_aim._log_probe` 的规矩 ✓）；
        #   ⚠ 本文件的 import 一律在**模块顶部** ✗ —— `core.behavior` 在这个函数里是**局部导入** ✓
        #     （`check_ui` 只挑 `gui.*` 那种自造模块的局部导入 ✓ 见 `docs/UI规范.md` ✓）。
        try:
            from core import behavior
            behavior.event("remote_measure_click",
                           size=[int(_w), int(_h)],
                           auto=bool(self.auto.isChecked()),
                           has_main=bool(self._main is not None),
                           has_panel=bool(_lp is not None))
        except Exception:              # noqa: BLE001
            pass
        if not (_w and _h):
            self.status.setText("⚠ **拿不到画面尺寸** ⇒ 先把「实时」预览开起来 ✓"
                                "（量的是实时回路正收到的那个尺寸 ✓ 没有帧就不知道要量多大 ✓）")
            return
        # ⚠ 形状按 `(h, w)` 传 ✓ —— `mouse_aim.gain_frame_key` 收的就是这个口径 ✓
        _shape = (int(_h), int(_w))
        if not ma_c.auto_measure_enabled():
            # ⚠⚠ **不许静静地什么都不做** ✗（那最容易让人以为"按钮坏了" ✓）：
            #   这一下没发指令的**真正原因**是上面那格没勾 ✓ ⇒ 明说 ✓。
            self.status.setText("⚠ 上面那格「自动量」**没勾** ⇒ 这一下不发指令 ✓"
                                "（那格的真正含义就是「允许它动鼠标」✓）"
                                "—— 想现在量：先勾上，再点这个按钮 ✓")
            return
        if ma_c.load_gain(_shape):
            self._start_force(_shape)      # 已经量过 ⇒ **强制重量一次** ✓（见 docstring ② ✓）
            return
        _started, why = ma_c.auto_measure_async(_shape)
        if _started:
            self.btn.setEnabled(False)
            self.btn.setText("量中…")
            self.status.setText("正在量 %d×%d …（会动一下鼠标 ✓）" % (_w, _h))
            return
        self.status.setText(why)           # 起不来 ⇒ 把原因原样说出来 ✓（别静默 ✗）

    def _start_force(self, shape):
        """**强制重量一次**（已标定过的尺寸也照量 ✓）⇒ 后台线程 + 信号回界面 ✓。"""
        self.btn.setEnabled(False)
        self.btn.setText("量中…")
        self.status.setText("正在**重量** %d×%d：会动一下鼠标（撞左上角 → 走一步 → 走回来，"
                            "全程不点 ✓）…" % (shape[1], shape[0]))
        self._task = _ForceMeasure(shape)
        self._task.done.connect(self._on_forced)
        self._task.start()

    def _on_forced(self, ok, why):
        self.btn.setEnabled(True)
        self.btn.setText("立即量一次")
        self.status.setText("%s　｜　%s" % (why, ma_c.calib_status()))


class _ForceMeasure(QThread):
    """后台"重量一次" ✓（`auto_measure(…, force=True)` ✓ ⇒ 发 `done(ok, 人话)` ✓）。

    ⚠⚠ **后台线程不许碰控件** ✗ ⇒ 结果一律走信号 ✓（Qt 的规矩 ✓）。
    ⚠ `force=True` 只有"**人明确按了按钮**"才有资格用 ✓（它会跳过失败冷却 ✓
      见 `mouse_aim.auto_measure` 的说明 ✓）—— 自动那条路永远不许带 force ✗。
    """

    done = pyqtSignal(bool, str)

    def __init__(self, shape):
        super().__init__()
        self._shape = shape

    def run(self):
        try:
            ok, why = ma_c.auto_measure(self._shape, force=True)
        except Exception as e:            # noqa: BLE001 —— 量炸了也要把按钮放回来 ✓
            ok, why = False, "量的时候抛异常：%s: %s" % (type(e).__name__, e)
        self.done.emit(bool(ok), str(why))
