"""设置弹窗：按用途分成三个页签。

| 页签 | 装什么 | 判断标准 |
|---|---|---|
| **界面** | 界面字号、可视化颜色/线宽 | 只管**长什么样**，改了立刻看得见 |
| **保护与恢复** | 停止自动的条件、防卡键、断线重连 | 管**出问题怎么办** |
| **诊断** | 性能日志开关、性能保活 | 只在排查问题时动 |

**为什么分页签**：原来是一根长列表，找一项得一路往下扫；而且「外观」和「安全」
混在一起容易看错上下文（比如把字号当成决策参数）。分页之后每页只有一件事，
页签名就是目录。

字号改动**立即生效**，不需要重启 —— 底层是 QApplication.setFont()，所有没写死
字号的控件都会跟随。保护与恢复里的都是决策参数：点确定后写回**当前项目**
（没打开项目时不落盘，只改内存 —— 见 decision/agent.py 的 set_save_hook）；
性能日志开关不是决策参数，它落在 config/live.yaml（和实时预览同一份配置）；
性能保活也是（进程优先级/电源节流/定时器精度，见 core/winperf.py）。

代码顺序 = 页签顺序 = 视觉顺序（docs/UI规范.md §4）。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QColorDialog, QDialog, QDialogButtonBox,
                             QFormLayout, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QTabWidget,
                             QVBoxLayout, QWidget)

from core import perf, winperf
from decision.agent import settings
from gui import theme
from perception import classes
# NoWheel* 必须模块级导入：控件在 __init__ 里建，方法里的懒导入到不了那儿。
# （详见 docs/UI规范.md：滚轮不许改参数）
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSlider,
                         NoWheelSpinBox)
from tools.config import load_live, update_live

#: 页签名（顺序 = 显示顺序）。测试和文档都按这份来。
TAB_NAMES = ("界面", "保护与恢复", "判定参数", "诊断")

#: 页签栏样式：**与部署台（A 机）的设置弹窗共用一份**（gui/theme.TAB_QSS）。
#: 对话框不继承主窗口的 QSS，两边都得自己带 —— 但只该有一处定义。
_TAB_QSS = theme.TAB_QSS


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumSize(470, 430)

        # 字号改动的对比基准。**必须在建控件之前取** —— 控件 setValue 时若已连上
        # valueChanged，_on_changed 会读它（早先靠「先 setValue 后 connect」绕开，
        # 那是隐式的，容易在重排代码时踩到）。
        self._orig = theme.load_size()

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 10)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._page_appearance(), TAB_NAMES[0])
        self.tabs.addTab(self._page_protect(), TAB_NAMES[1])
        self.tabs.addTab(self._page_judge(), TAB_NAMES[2])
        self.tabs.addTab(self._page_diagnose(), TAB_NAMES[3])
        self.tabs.setStyleSheet(_TAB_QSS)
        root.addWidget(self.tabs, 1)

        # ---- 按钮（在页签之外：无论在哪一页都能点确定）----
        box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("确定")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        root.addWidget(box)

        self._on_changed(self.slider.value())

    # ---------------- 页面骨架 ----------------

    def _page(self):
        """建一个可滚动的页面，返回 (页 widget, 往里面加东西的布局)。

        每个页签都套一层 QScrollArea：字号调到最大时内容会明显变高，
        不套就可能超出小屏（docs/UI规范.md §4：长面板放进 QScrollArea）。
        """
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)      # 页签已经有边框了，别套两层
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(8)
        lay.setContentsMargins(10, 10, 10, 10)
        area.setWidget(inner)
        outer.addWidget(area)
        return page, lay

    @staticmethod
    def _head(lay, title, note=""):
        """一个页内小标题（+ 可选说明），返回标签供测试/复用。"""
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(lbl)
        if note:
            sub = QLabel(note)
            sub.setStyleSheet("color: #5f6368;")
            sub.setWordWrap(True)
            lay.addWidget(sub)
        return lbl

    # ---------------- 页签 1：界面 ----------------

    def _page_appearance(self):
        """外观：字号 + 可视化颜色。都只影响「看到什么」，不影响行为。"""
        page, lay = self._page()

        # ---- 字号 ----
        head = QHBoxLayout()
        head.addWidget(QLabel("界面字号"))
        head.addStretch(1)
        self.lbl_val = QLabel()
        self.lbl_val.setStyleSheet("color: #5f6368;")
        head.addWidget(self.lbl_val)
        lay.addLayout(head)

        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setRange(theme.MIN_SIZE, theme.MAX_SIZE)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(1)
        self.slider.setValue(self._orig)
        self.slider.valueChanged.connect(self._on_changed)
        lay.addWidget(self.slider)

        scale = QHBoxLayout()
        scale.addWidget(QLabel("小"))
        scale.addStretch(1)
        scale.addWidget(QLabel("大"))
        lay.addLayout(scale)

        self.lbl_preview = QLabel("预览：怪物检出 12 框 · 置信度 0.92")
        self.lbl_preview.setStyleSheet(
            "background: #ffffff; border: 1px solid #dadce0;"
            " border-radius: 4px; padding: 8px; color: #202124;")
        lay.addWidget(self.lbl_preview)

        self.lbl_note = QLabel()
        self.lbl_note.setStyleSheet("color: #80868b;")
        self.lbl_note.setWordWrap(True)
        lay.addWidget(self.lbl_note)

        # ---- 可视化 ----
        lay.addSpacing(8)
        self._head(
            lay, "可视化（实时预览）",
            "实时预览 / 质检台 / 推理结果里，每个类别的框颜色和辅助线颜色。\n"
            "点确定立刻生效（实时预览每秒重读一次配置，不用重开）。")

        self._vis = theme.load_vis()
        vf = QFormLayout()
        vf.setLabelAlignment(Qt.AlignLeft)
        self._color_btns = {}

        def add_row(label, key, tip=""):
            btn = self._color_btn(self._vis[key])
            if tip:
                btn.setToolTip(tip)
            self._color_btns[key] = btn
            vf.addRow(label, btn)

        # 类别框颜色**按类别表生成**：有几个类别就有几行，以后加类别这里自动多一行，
        # 不会出现「新类别没地方改颜色」。配置键 = 英文名 + "_color"（见 gui/theme.py）。
        for cid, en, zh, _bgr in classes.CLASSES:
            add_row("%s框颜色" % zh, "%s_color" % en,
                    "类别 %d（%s）—— 检测框颜色" % (cid, en))

        # 下面几条不是类别，是辅助线与标记
        add_row("锁定框颜色", "lock_color", "锁定的那个攻击目标")
        add_row("最大攻击距离线颜色", "attack_color", "")
        add_row("最小攻击距离线颜色", "min_attack_color", "规避范围")
        add_row("视野线颜色", "vision_color", "")
        add_row("定时任务颜色", "timer_color",
                "实时画面上「当前任务 / 定时任务」那几行的字色。\n"
                "那里**没有底色**（只有描边），所以别选太暗的 —— 会压不住游戏画面。")

        self.sp_vision_width = NoWheelSpinBox()
        self.sp_vision_width.setRange(1, 10)
        self.sp_vision_width.setValue(int(self._vis["vision_width"]))
        # 单位写在行标签里（UI 规范 §9：不写进编辑框）
        vf.addRow("视野线宽度(px)", self.sp_vision_width)
        lay.addLayout(vf)

        # ---- foothold 集合编辑器 ----
        # 单独一组：上面那组的说明写着"实时预览"，而这项只管编辑器那个窗口。
        lay.addSpacing(8)
        self._head(lay, "foothold 集合编辑器（线宽，px）",
                   "地形线条在编辑器里画多粗。\n"
                   "点确定后**再打开一次**编辑器就按新值画（窗口一直开着的话，"
                   "点一次「编辑集合…」也会刷新）。")
        self.sp_fh_width = NoWheelSpinBox()
        self.sp_fh_width.setRange(theme.FOOTHOLD_W_MIN, theme.FOOTHOLD_W_MAX)
        self.sp_fh_width.setValue(theme.load_foothold_width())
        self.sp_fh_width.setToolTip(
            "线宽单位是**屏幕像素**（和缩放无关）：编辑器整图看时约缩小到 0.35 倍、\n"
            "放大看细节能到 8 倍 —— 若按「场景单位」给宽度，同一个值在两种视图下\n"
            "差二十多倍（整图时看不见线，放大时线糊成一片），所以这里定的是\n"
            "「屏幕上多粗」。\n\n"
            "觉得线太细看不清就调大；嫌糊住底图就调回 1。\n"
            "选中/集合内的线会呼吸高亮，它的粗细也跟着这个值走。")
        lay.addWidget(self.sp_fh_width)

        lay.addStretch(1)
        return page

    # ---------------- 页签 2：保护与恢复 ----------------

    def _page_protect(self):
        """出问题怎么办：什么时候停下、怎么清干净、断了怎么回来。"""
        page, lay = self._page()

        self._head(lay, "朝向无变化停止自动（min）",
                   "角色朝向超过该时长没变化，自动停止（0 = 禁用）。")
        self.sp_timeout = NoWheelDoubleSpinBox()
        self.sp_timeout.setRange(0.0, 1440.0)
        self.sp_timeout.setDecimals(1)
        self.sp_timeout.setSingleStep(0.5)
        self.sp_timeout.setValue(float(settings.facing_timeout_min))
        lay.addWidget(self.sp_timeout)

        lay.addSpacing(8)
        self._head(lay, "找不到玩家停止自动（min）",
                   "连续找不到玩家超过该时长，自动停止（0 = 禁用）。")
        self.sp_player_lost = NoWheelDoubleSpinBox()
        self.sp_player_lost.setRange(0.0, 1440.0)
        self.sp_player_lost.setDecimals(1)
        self.sp_player_lost.setSingleStep(0.5)
        self.sp_player_lost.setValue(float(settings.player_lost_timeout_min))
        lay.addWidget(self.sp_player_lost)

        lay.addSpacing(8)
        self._head(lay, "定时清空按键（s，防卡键）",
                   "每隔该秒数向 Pro Micro 发一次 RELEASEALL，"
                   "清空可能卡住的键（0 = 禁用）。")
        self.sp_resetall = NoWheelSpinBox()
        self.sp_resetall.setRange(0, 3600)
        self.sp_resetall.setValue(int(settings.resetall_interval))
        lay.addWidget(self.sp_resetall)

        lay.addSpacing(8)
        self._head(
            lay, "断线自动重连",
            "玩家框丢失后自动判别界面（断线提示框 / 登录 / 选频道 / 排队 / 选角），"
            "确认是断线就停止自动并按步骤走回游戏，回到游戏后恢复自动。\n"
            "鼠标点服务器 / 点频道还没接（要先做鼠标标定），走到那一步会停下并提示。\n"
            "做判断用模板锚点，不读文字、不训模型。")
        self.ck_reconnect = QCheckBox("检测到断线后自动重连")
        self.ck_reconnect.setChecked(bool(settings.reconnect_enabled))
        lay.addWidget(self.ck_reconnect)

        # 重连的**子参数**（2026-09-26 补：审计发现这 5 个"只能在配置文件里改" ✗ ——
        # 页面上原本只有上面那个总开关）。版式同样遵守 UI 规范 §9：一行一个、单位进标签、
        # 说明进 tooltip、毫秒一律**整数**控件。
        rc_form = QFormLayout()
        rc_form.setLabelAlignment(Qt.AlignLeft)
        rc_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        def rc_row(label, w, tip):
            w.setToolTip(tip)
            rc_form.addRow(label, w)
            return w

        self.sp_rc_probe = NoWheelDoubleSpinBox()
        self.sp_rc_probe.setRange(0.0, 600.0)
        self.sp_rc_probe.setDecimals(1)
        self.sp_rc_probe.setSingleStep(0.5)
        self.sp_rc_probe.setValue(float(settings.reconnect_probe_after_lost_sec))
        rc_row("丢失多久后开始探界面(s)", self.sp_rc_probe,
               "玩家框丢多久之后开始判别界面（0 = 立刻）。\n\n"
               "默认 0：断线提示框只显示两三秒，等 5 秒再探就错过它了，客户端会一直卡在\n"
               "提示框上等人按确定。探一次只要约 4 ms，不必为省这点开销推迟。")

        self.sp_rc_step = NoWheelSpinBox()
        self.sp_rc_step.setRange(0, 60000)
        self.sp_rc_step.setSingleStep(500)
        self.sp_rc_step.setValue(int(settings.reconnect_step_timeout_ms))
        rc_row("每步等待超时(ms)", self.sp_rc_step,
               "重连每一步（点服务器 / 点频道 / 排队 / 选角）等「界面真的变了」的超时。\n"
               "到点还没变就按下面的次数重试。")

        self.sp_rc_queue = NoWheelSpinBox()
        self.sp_rc_queue.setRange(0, 600000)
        self.sp_rc_queue.setSingleStep(1000)
        self.sp_rc_queue.setValue(int(settings.reconnect_queue_timeout_ms))
        rc_row("排队弹窗超时(ms)", self.sp_rc_queue,
               "排队那一步专用的长超时：排队可能要等很久，用上面那个 3 秒会一直重试。")

        self.sp_rc_retry = NoWheelSpinBox()
        self.sp_rc_retry.setRange(0, 20)
        self.sp_rc_retry.setValue(int(settings.reconnect_max_retry))
        rc_row("同一步最多重试(次)", self.sp_rc_retry,
               "同一步骤最多重试几次；到上限就停下并提示（不会无限重连）。")

        self.ck_rc_resume = QCheckBox("回到游戏后自动恢复自动打怪")
        self.ck_rc_resume.setChecked(bool(settings.reconnect_resume_auto))
        self.ck_rc_resume.setToolTip(
            "重连成功、回到游戏画面之后，自动把「自动打怪」重新打开。\n"
            "不勾就停在「已回到游戏、自动仍是关的」，由你自己决定。")
        rc_form.addRow("", self.ck_rc_resume)
        lay.addLayout(rc_form)

        # 总开关关着时子参数灰掉（改它没意义）—— 和「追击起跳」那一组同一个做法
        def _rc_enable(on):
            for w in (self.sp_rc_probe, self.sp_rc_step, self.sp_rc_queue,
                      self.sp_rc_retry, self.ck_rc_resume):
                w.setEnabled(bool(on))
        self.ck_reconnect.toggled.connect(_rc_enable)
        _rc_enable(self.ck_reconnect.isChecked())

        lay.addStretch(1)
        return page

    # ---------------- 页签 3：判定参数 ----------------

    def _page_judge(self):
        """**判定**用参数：哪些事算"成立"。

        为什么单独一页：这一页回答的既不是"长什么样"（界面）、也不是"出问题怎么办"
        （保护与恢复），而是"**怎么算数**" —— 同一个动作（对齐到某个坐标）成不成功、
        由哪几个数说了算。以后爬绳 / 寻路的判据都挂在这儿，塞进别页会找不着。
        """
        page, lay = self._page()

        # 版式（2026-09-26 用户要求，见 docs/UI规范.md §9）：
        #   · **一行一个参数**：左边参数名（**带单位**）、右边配置 —— 扫一眼就找得到；
        #   · 参数说明**全部进 tooltip**（页面上不再摆大段文字：那是查参数时的噪音）；
        #   · 单位写在**标签**里，不写进编辑框（`setSuffix` 在规范的禁用清单里）。
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        def row(label, w, tip):
            w.setToolTip(tip)
            form.addRow(label, w)
            return w

        # ① 坐标对齐误差范围
        self.sp_align_tol = NoWheelSpinBox()
        self.sp_align_tol.setRange(1, 200)
        self.sp_align_tol.setValue(int(settings.align_tol_px))
        row("坐标对齐误差范围(px)", self.sp_align_tol,
            "需要对齐坐标的功能（寻路走到某个 x、上绳前对准绳的 x）容许的偏差。\n"
            "单位是**游戏世界像素**（就是小地图算出来的那套坐标），不是屏幕像素。\n"
            "调小 ⇒ 对得更准但要磨一会儿（甚至走过头来回摆）；\n"
            "调大 ⇒ 快，但站偏了也算数（上绳会按不上）。\n\n"
            "默认 6：绳在数据里就是一条线（宽度 0），而世界坐标本身有几像素抖动 ——\n"
            "6 够吸住，又不会宽到把隔壁平台的边也算进来。")

        # ② 坐标对齐误差时间
        self.sp_align_hold = NoWheelSpinBox()
        self.sp_align_hold.setRange(0, 5000)
        self.sp_align_hold.setValue(int(settings.align_hold_ms))
        row("坐标对齐误差时间(ms)", self.sp_align_hold,
            "进误差范围后要**保持这么久**才算对齐成功；\n"
            "上绳**够到目标平台面之后**也要再按住 ↑ 这么久才松（2026-09-26 用户要求）。\n"
            "为什么不能只看一帧：小地图定位与按键下发不是同一时刻（有延迟、还会抖），\n"
            "单帧落进误差范围不代表真站住了。\n"
            "**实际等待 = 画面 / 指令延迟 + 这个值**，所以它不是「总超时」。\n\n"
            "默认 250：约 3~4 个视频帧，够滤掉抖动，又不至于每步都干等。")

        # ③ 寻路超时时间（2026-09-26 新增；当天又明确了口径：按**每一段**算）
        self.sp_goto_timeout = NoWheelSpinBox()
        self.sp_goto_timeout.setRange(0, 3600)
        self.sp_goto_timeout.setValue(int(round(float(
            getattr(settings, "goto_timeout_s", 30.0) or 0.0))))
        row("寻路超时时间(s)", self.sp_goto_timeout,
            "「**每一段**」寻路（从当前集合走到下一个集合）最多花这么久，超了就切断它\n"
            "（画面上那行会写明「寻路超时：这一段已经跑了 N 秒」）。\n\n"
            "⚠ **每完成一段就重新计时**：一条多段路线里每段各自算 ⇒ 整条路线的**总**耗时\n"
            "可以远超这个值（那**不算**超时 ✓）。它挡的是「某一段卡住了」或者\n"
            "「在一段里反复重试」。\n\n"
            "为什么除了任务自己的超时还要它：失败重来会把任务内部的计时**清零**，\n"
            "同一段里累计下来可能远超预期。\n\n"
            "0 = 不限时（老行为）。默认 30。")

        # ⚠ 这里**曾经**摆过一个「走的方向」下拉（2026-09-26 用户明确删掉）：
        # 走只有一种走法 —— 朝目标集合的 x 中点；设置是**全局参数**，
        # 用户没提过的参数一律不许往这里加。将来若某一步需要"只按 ←/→"，
        # 那是**逐边**配置（foothold 编辑器 →「可到达」窗口）。

        lay.addLayout(form)
        lay.addStretch(1)
        return page

    # ---------------- 页签 4：诊断 ----------------

    def _page_diagnose(self):
        """排查问题用的开关。平常不用动这一页。"""
        page, lay = self._page()

        self._head(
            lay, "性能日志",
            "把关键路径的耗时/吞吐/延迟/资源记进仓库根 perf.log（每 30 秒一段）。\n"
            "日志不会无限增长：超过 %d KB 就自动截掉前半，只留最近一段历史。\n"
            "开销极低（一个打点约 0.3 微秒），不排查问题时可以关掉。\n"
            "评估报告：python -m tools.perf_report"
            % (perf.MAX_BYTES // 1024))
        self.ck_perf = QCheckBox("记录性能日志到仓库根 perf.log")
        self.ck_perf.setChecked(bool(load_live().get("perf_log", True)))
        lay.addWidget(self.ck_perf)

        self._head(
            lay, "性能保活",
            "向 Windows 声明「别把我当后台程序降级」，四件事：\n"
            "  · 关电源节流（EcoQoS 降频）　· 进程优先级 → 高于正常\n"
            "  · 定时器精度 → 1 ms　　　　　· 阻止系统空闲时挂起\n\n"
            "**默认开，建议一直开着**：关键回路是 10 ms 级的时序拍，而 Windows 默认\n"
            "定时器粒度约 15.6 ms —— 关掉它的现象是「窗口一失焦就卡」，很难归因\n"
            "（没人会想到是自己关的）。放开手去做别的事时，这一项就是「别掉拍」的保证。\n\n"
            "自己验一下数字（会实测 sleep(10ms) 到底睡多久）：\n"
            "    python -m tools.selftest_winperf")
        self.ck_keepalive = QCheckBox("性能保活（失焦时也不被系统降级）")
        self.ck_keepalive.setChecked(bool(load_live().get("perf_keepalive", True)))
        lay.addWidget(self.ck_keepalive)

        lay.addStretch(1)
        return page

    # ---------------- 颜色按钮 ----------------

    def _color_btn(self, color):
        """一个显示当前颜色、点击弹颜色选择器的按钮。"""
        btn = QPushButton(color)
        btn.setFixedWidth(110)
        btn._color = color
        btn.setStyleSheet(
            "QPushButton { background: %s; color: #202124; border: 1px solid #dadce0;"
            " border-radius: 4px; padding: 4px 8px; }" % color)
        btn.clicked.connect(lambda: self._pick_color(btn))
        return btn

    def _pick_color(self, btn):
        from PyQt5.QtGui import QColor
        c = QColorDialog.getColor(QColor(btn._color), self, "选择颜色")
        if c.isValid():
            btn._color = c.name()
            btn.setText(c.name())
            btn.setStyleSheet(
                "QPushButton { background: %s; color: #202124;"
                " border: 1px solid #dadce0; border-radius: 4px; padding: 4px 8px; }"
                % c.name())

    # ---------------- 字号 ----------------

    def _on_changed(self, v):
        self.lbl_val.setText("%d px" % v)
        self.lbl_preview.setStyleSheet(
            "background: #ffffff; border: 1px solid #dadce0;"
            " border-radius: 4px; padding: 8px; color: #202124;"
            " font-size: %dpx;" % v)
        if v < self._orig:
            self.lbl_note.setText("比当前小 %d px。" % (self._orig - v))
        elif v > self._orig:
            self.lbl_note.setText("比当前大 %d px。" % (v - self._orig))
        else:
            self.lbl_note.setText("和当前一样。")

    # ---------------- 确定 ----------------

    def _accept(self):
        v = self.slider.value()
        if v != self._orig:
            theme.save_size(v)
            from PyQt5.QtWidgets import QApplication
            theme.apply(QApplication.instance(), v)
        # 朝向超时
        t = self.sp_timeout.value()
        if t != settings.facing_timeout_min:
            settings.facing_timeout_min = t
            settings.save()
        # 找不到玩家超时
        t2 = self.sp_player_lost.value()
        if t2 != settings.player_lost_timeout_min:
            settings.player_lost_timeout_min = t2
            settings.save()
        # 断线自动重连
        rc = self.ck_reconnect.isChecked()
        if rc != bool(settings.reconnect_enabled):
            settings.reconnect_enabled = rc
            settings.save()
        # 重连子参数（2026-09-26 新增到界面上）
        rp = float(self.sp_rc_probe.value())
        if rp != float(settings.reconnect_probe_after_lost_sec):
            settings.reconnect_probe_after_lost_sec = rp
            settings.save()
        rs = int(self.sp_rc_step.value())
        if rs != int(settings.reconnect_step_timeout_ms):
            settings.reconnect_step_timeout_ms = rs
            settings.save()
        rq = int(self.sp_rc_queue.value())
        if rq != int(settings.reconnect_queue_timeout_ms):
            settings.reconnect_queue_timeout_ms = rq
            settings.save()
        rr = int(self.sp_rc_retry.value())
        if rr != int(settings.reconnect_max_retry):
            settings.reconnect_max_retry = rr
            settings.save()
        ra = bool(self.ck_rc_resume.isChecked())
        if ra != bool(settings.reconnect_resume_auto):
            settings.reconnect_resume_auto = ra
            settings.save()
        # 定时清空按键
        t3 = self.sp_resetall.value()
        if t3 != settings.resetall_interval:
            settings.resetall_interval = t3
            settings.save()
        # 判定参数（和上面那些一样：跟着**当前项目**存）
        at = int(self.sp_align_tol.value())
        if at != settings.align_tol_px:
            settings.align_tol_px = at
            settings.save()
        ah = int(self.sp_align_hold.value())
        if ah != settings.align_hold_ms:
            settings.align_hold_ms = ah
            settings.save()
        # 寻路超时时间（2026-09-26 新增，跟着项目存）
        gt = int(self.sp_goto_timeout.value())
        if gt != int(getattr(settings, "goto_timeout_s", 30) or 0):
            settings.goto_timeout_s = gt
            settings.save()
        # 可视化
        vis_cfg = {k: b._color for k, b in self._color_btns.items()}
        vis_cfg["vision_width"] = self.sp_vision_width.value()
        theme.save_vis(vis_cfg)
        # foothold 编辑器的线条宽度（和上面那份配置同一文件，但语义上不属于"实时预览"）
        fw = self.sp_fh_width.value()
        if fw != theme.load_foothold_width():
            theme.save_foothold_width(fw)
        # 性能日志 / 性能保活：都落在 config/live.yaml（save_live 是整文件覆盖，
        # 所以先读整份、只改这两个键、再写回 —— 别把实时预览那几项冲掉）。
        pl = self.ck_perf.isChecked()
        ka = self.ck_keepalive.isChecked()
        if pl != bool(load_live().get("perf_log", True)):
            update_live(perf_log=pl)
            perf.set_enabled(pl)
        if ka != bool(load_live().get("perf_keepalive", True)):
            update_live(perf_keepalive=ka)
        # 保活立即生效：开 → 马上声明（幂等，重复调无害）；
        # 关 → 把定时器精度还回去（timeEndPeriod）。**进程优先级不主动降回去**：
        # "关掉保活"的意图是「别再动系统」，不是「把我降到最低」。
        if ka:
            winperf.apply()
        else:
            winperf.release()
        self.accept()
