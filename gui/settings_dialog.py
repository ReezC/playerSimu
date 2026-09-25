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
from gui.widgets import (NoWheelDoubleSpinBox, NoWheelSlider,
                         NoWheelSpinBox)
from tools.config import load_live, update_live

#: 页签名（顺序 = 显示顺序）。测试和文档都按这份来。
TAB_NAMES = ("界面", "保护与恢复", "诊断")

#: 页签栏样式。主窗口那份 QSS 是 `self.setStyleSheet()` 打在主窗口上的，
#: 对话框不继承 —— 所以这里单独来一份，好让设置弹窗和主窗口长得一致。
#: 只写页签相关几条；其余控件沿用各自的内联样式。
_TAB_QSS = """
QTabWidget::pane { border: 1px solid #e2e5ea; border-radius: 6px;
                   background: #ffffff; top: -1px; }
QTabBar::tab { background: #f0f2f5; color: #5f6368; padding: 6px 14px;
               border: 1px solid #e2e5ea; border-bottom: none;
               border-top-left-radius: 6px; border-top-right-radius: 6px;
               margin-right: 2px; }
QTabBar::tab:selected { background: #ffffff; color: #202124; font-weight: 600; }
QTabBar::tab:hover { color: #202124; }
"""


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
        self.tabs.addTab(self._page_diagnose(), TAB_NAMES[2])
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

        self.sp_vision_width = NoWheelSpinBox()
        self.sp_vision_width.setRange(1, 10)
        self.sp_vision_width.setSuffix(" px")
        self.sp_vision_width.setValue(int(self._vis["vision_width"]))
        vf.addRow("视野线宽度", self.sp_vision_width)
        lay.addLayout(vf)

        lay.addStretch(1)
        return page

    # ---------------- 页签 2：保护与恢复 ----------------

    def _page_protect(self):
        """出问题怎么办：什么时候停下、怎么清干净、断了怎么回来。"""
        page, lay = self._page()

        self._head(lay, "朝向无变化停止自动",
                   "角色朝向超过该时长没变化，自动停止（0 = 禁用）。")
        self.sp_timeout = NoWheelDoubleSpinBox()
        self.sp_timeout.setRange(0.0, 1440.0)
        self.sp_timeout.setDecimals(1)
        self.sp_timeout.setSingleStep(0.5)
        self.sp_timeout.setSuffix(" min")
        self.sp_timeout.setValue(float(settings.facing_timeout_min))
        lay.addWidget(self.sp_timeout)

        lay.addSpacing(8)
        self._head(lay, "找不到玩家停止自动",
                   "连续找不到玩家超过该时长，自动停止（0 = 禁用）。")
        self.sp_player_lost = NoWheelDoubleSpinBox()
        self.sp_player_lost.setRange(0.0, 1440.0)
        self.sp_player_lost.setDecimals(1)
        self.sp_player_lost.setSingleStep(0.5)
        self.sp_player_lost.setSuffix(" min")
        self.sp_player_lost.setValue(float(settings.player_lost_timeout_min))
        lay.addWidget(self.sp_player_lost)

        lay.addSpacing(8)
        self._head(lay, "定时清空按键（防卡键）",
                   "每隔该秒数向 Pro Micro 发一次 RELEASEALL，"
                   "清空可能卡住的键（0 = 禁用）。")
        self.sp_resetall = NoWheelSpinBox()
        self.sp_resetall.setRange(0, 3600)
        self.sp_resetall.setSuffix(" s")
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

        lay.addStretch(1)
        return page

    # ---------------- 页签 3：诊断 ----------------

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
        # 定时清空按键
        t3 = self.sp_resetall.value()
        if t3 != settings.resetall_interval:
            settings.resetall_interval = t3
            settings.save()
        # 可视化
        vis_cfg = {k: b._color for k, b in self._color_btns.items()}
        vis_cfg["vision_width"] = self.sp_vision_width.value()
        theme.save_vis(vis_cfg)
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
