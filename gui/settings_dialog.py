"""设置弹窗：按用途分成三个页签。

| 页签 | 装什么 | 判断标准 |
|---|---|---|
| **界面** | 界面字号、可视化颜色/线宽、**实时画面那块小地图**（叠不叠 / 浓淡） | 只管**长什么样**、只动**本机配置**，改了立刻看得见 |
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

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QColorDialog, QDialog,
                             QDialogButtonBox, QFormLayout, QFrame, QGroupBox,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QTabWidget, QVBoxLayout, QWidget)

from core import mapdata, perf, winperf
from decision.agent import settings
from gui import theme
from gui.project import last_opened
from perception import classes
from perception import minimap as mm
# NoWheel* 必须模块级导入：控件在 __init__ 里建，方法里的懒导入到不了那儿。
# （详见 docs/UI规范.md：滚轮不许改参数）
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSlider,
                         NoWheelSpinBox, scroll_page)
from tools.config import load_live, update_live

#: 页签名（顺序 = 显示顺序）。测试和文档都按这份来。
TAB_NAMES = ("界面", "保护与恢复", "判定参数", "诊断")

#: 页签栏样式：**与部署台（A 机）的设置弹窗共用一份**（gui/theme.TAB_QSS）。
#: 对话框不继承主窗口的 QSS，两边都得自己带 —— 但只该有一处定义。
_TAB_QSS = theme.TAB_QSS


class SettingsDialog(QDialog):
    #: 「叠图开关 / 浓淡」改了 ⇒ 通知**路线识别面板**重画实时画面上那一层
    #: （那一层是它画的，见 `route_panel.apply_overlay_settings` ✓）。主窗口接线 ✓。
    #: （「框选小地图」2026-09-27 搬回那一页了 ⇒ 重框不再走这个信号 ✓。）
    overlay_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumSize(470, 430)
        #: **非模态**（2026-09-27 用户定，同「双点标定」窗）：改完「浓淡 / 叠图」要对着
        #: **实时画面 / 路线识别页**看效果 —— 模态会把整个工作台锁住，只能关窗→看→重开 ✗。
        #: ⇒ 主窗口用 `show()` 而不是 `exec_()`，并且**单实例**（见 `_on_settings`）。
        #: （原来是"框选小地图要拿实时画面"这条理由；那个按钮 2026-09-27 已按用户要求
        #:   搬回「路线识别 → 寻路配置」（**按项目存** ✓），但这扇窗仍然保持非模态 ✓。）
        self.setModal(False)

        # 「实时画面 · 地形叠加」那两项的**打开时基准值**（点确定时只写变了的 ✓）。
        # 浓淡存在**当前地图的标定文件**里（按地图 id + 来源各一份 ✓）—— 这个窗口不是
        # 路线识别面板，所以地图 id 只能从"最近打开的项目"取（和那边同一处来源 ✓）。
        _p = last_opened()
        self._map_id = str((_p.get("map_id") if _p else "") or "").strip()
        self._alpha0 = mm.overlay_alpha_pct(
            mapdata.load_calib(self._map_id, mm.live_src(load_live()))
            if self._map_id else None)
        self._draw0 = bool(load_live().get("mmap_draw", False))

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
        theme.unify_ok_cancel(box.button(QDialogButtonBox.Ok),
                              box.button(QDialogButtonBox.Cancel))
        # ⚠ 连的是**包了一层**的 `_accept_safe`（不是裸 `_accept` ✓）—— 写回中途出错时
        #   ①**保证对话框关得掉** ②**把"坏在哪一项"报出来**（用户 2026-09-28 报"点确定
        #   修改无效、好多次"✓ 以前是静默跳过 + 关不掉 ✗ 见那个方法的说明 ✓）。
        box.accepted.connect(self._accept_safe)
        box.rejected.connect(self.reject)
        root.addWidget(box)

        self._on_changed(self.slider.value())
        theme.bind_window_state(self, "settings")      # 拉过的大小/位置按客户端记住 ✓

    # ---------------- 页面骨架 ----------------

    def _page(self):
        """建一个可滚动的页面，返回 (页 widget, 往里面加东西的布局)。

        每个页签都套一层 QScrollArea：字号调到最大时内容会明显变高，
        不套就可能超出小屏（docs/UI规范.md §4：长面板放进 QScrollArea）。
        ⚠ 滚动区**只有一处实现**：`gui.widgets.scroll_page`（2026-09-27 收口）——
        以前这里手写了一份，工作台主窗口、A 机部署台又各写一份 ⇒ **四份重复** ✗，
        改一处忘一处就是"有的页能滚、有的页不能滚"（用户报过路线识别页不能滚 ✓）。
        """
        page = QWidget()
        lay = scroll_page(page, margins=(10, 10, 10, 10), spacing=8)
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
        page_lay = lay          # 页面布局：下面每个分组都挂在它上面（每组开头的套路 ✓）

        # ---- ① 界面字号 ----
        grp = QGroupBox("界面字号")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        page_lay.addWidget(grp)
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

        # ---- ② 实时画面 · 地形叠加（2026-09-26 从「路线识别」页搬来）----
        # 为什么搬：叠不叠、浓淡多少是**外观偏好**，不是寻路参数 ✓；而"画在哪儿、
        # 没画卡在哪一步"那行**现场状态**仍留在路线识别页 ✓（那才是那一页要看的 ✓）。
        # ⚠ 浓淡按**每张图**各存一份（在标定文件里，和「标定…」弹窗那个滑块同一处 ✓）
        #   ⇒ 这里显示/写的是**当前项目那张图**的值 ✓。
        grp = QGroupBox("实时画面 · 地形叠加")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        page_lay.addWidget(grp)

        self.ck_mmap_draw = QCheckBox("在实时画面上叠地形图")
        self.ck_mmap_draw.setToolTip(
            "把这张图的 `<id>_overlay.png`（按 foothold 画的地形图，平台形状）\n"
            "按**你标定出来的换算**半透明地画到实时画面的小地图面板上。\n\n"
            "看什么：叠上去和游戏小地图里的地形**重合不重合** —— 不重合就说明标定\n"
            "不对（或者小地图面板的位置框错了）。这一层是**静止**的：几何用标定时\n"
            "那一份，不会跟着人物走（跟着走要每帧重新定位，见 docs/寻路设计.md）。\n\n"
            "为什么只叠地形图：平台形状和 foothold 是同一套坐标，能直接看出「程序\n"
            "以为你在哪块平台上」；小地图底图是 WZ 的素材画，没有这个信息。\n\n"
            "点确定立刻生效；画没画、画在哪儿，去「路线识别」页看那一行状态 ✓。")
        self.ck_mmap_draw.setChecked(self._draw0)
        lay.addWidget(self.ck_mmap_draw)

        arow = QHBoxLayout()
        arow.setSpacing(8)
        arow.addWidget(QLabel("浓淡"))
        self.sld_alpha = NoWheelSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(int(self._alpha0))
        self.sld_alpha.setMinimumWidth(150)
        self.sld_alpha.setToolTip(
            "地形叠加层的浓淡（%，0 = 完全看不见，100 = 完全实）。\n\n"
            "存进**这张图的标定文件**（按来源各一份），和「标定…」弹窗里那个滑块\n"
            "是同一处 —— 在哪儿调都一样 ✓。\n\n"
            "看不清楚就调低一点：小地图面板本来就小，太实会盖住底图上的细节。")
        arow.addWidget(self.sld_alpha)
        self.lbl_alpha_val = QLabel("%d%%" % int(self._alpha0))
        self.lbl_alpha_val.setMinimumWidth(42)
        self.lbl_alpha_val.setStyleSheet("color: #5f6368;")
        arow.addWidget(self.lbl_alpha_val)
        arow.addStretch(1)
        lay.addLayout(arow)
        self.sld_alpha.valueChanged.connect(self._on_alpha_changed)

        self.lbl_alpha_note = QLabel()
        self.lbl_alpha_note.setStyleSheet("color: #5f6368;")
        self.lbl_alpha_note.setWordWrap(True)
        lay.addWidget(self.lbl_alpha_note)

        # 叠图关着时浓淡没有意义 ⇒ 灰掉而不是藏掉（布局不跳，旁边就是那句话 ✓）
        self.ck_mmap_draw.toggled.connect(
            lambda on: self.sld_alpha.setEnabled(bool(on)))
        self.sld_alpha.setEnabled(self.ck_mmap_draw.isChecked())
        if not self._map_id:
            # 没有地图就没有"这张图的标定"可写 ⇒ 明说，别让人以为调了没生效 ✗
            self.sld_alpha.setEnabled(False)
        self._on_alpha_changed(int(self._alpha0))

        # ⚠ 「框选小地图」**2026-09-27 用户要求搬回「路线识别 → 寻路配置」**
        #   （就在「寻路编辑器」正下方 ✓）—— 理由是它必须**按项目（地图）**保存：
        #   不同地图的小地图面板尺寸/位置完全不同，而这个窗口里的东西是**全局**的，
        #   换个项目还是上一张图的框 ⇒ 叠图/定位全错 ✗。这里只留"往那块上画什么"
        #   （叠不叠 / 浓淡多少），那是本机的外观偏好 ✓。

        # ---- ③ 检测框颜色（按类别表生成）----
        self._vis = theme.load_vis()
        self._color_btns = {}
        #: ⭐ **各框线宽的 spin**（用户 2026-09-28 ✓"给其他的粗细也加配置"）→ {vis 键: QSpinBox}
        self._width_spins = {}
        #: 「辅助线与标记」每项前面的**显示开关**（用户 2026-09-27 要求）→ {键: QCheckBox}
        self._vis_on = {}

        _grid_rows = [0]          # 网格版的行号（闭包内递增 ✓）

        def add_row(form, label, key, tip="", on_key=None, on_tip="",
                    w_key=None, w_tip=""):
            """一行「颜色（+ 粗细）」（可选：**前面再加一个开关** ✓，用户 2026-09-27 起）。

            ⭐⭐ 2026-09-28 用户四条要求（照办 ✓ 原话）："**粗细调整参数放颜色配置右边**，
               并**给其他的粗细也加配置**，并且用**网格对齐**，要求**参数名显示完整**"。
            · **网格版**（`form` 是 `QGridLayout`）⇒ 按**三列**摆：
              列 0 = 开关（名字写在上头 ✓）／列 1 = 色块／列 2 = 粗细框 ✓
              —— 网格保证**各行的色块、粗细框上下对齐** ✓，列 0 的**最小宽度**由调用方
              抬到够长的那个名字 ⇒ **参数名完整显示**（不截断、不省略 ✓）。
            · **老版**（`form` 是 `QFormLayout`，类别框那一组 ✓）⇒ 走原样（那组没有粗细 ✓）。
            · `w_key` 给了 ⇒ **多一格粗细**（照样存 `config/ui.yaml` 的 `vis:` 段 ✓，
              与颜色**同一处**、**同一套** ✓）；没给 ⇒ 只两列 —— 有的项本来就没有线宽
              （「定时任务」那是**字色** ✗），不该硬塞一格 ✗。
            · `on_key` 给了 ⇒ 关掉 = **不画这一项**（颜色/粗细都留着，开回来还在 ✓）。
            """
            btn = self._color_btn(self._vis[key])
            if tip:
                btn.setToolTip(tip)
            self._color_btns[key] = btn
            ws = None
            if w_key is not None:
                ws = NoWheelSpinBox()
                ws.setRange(1, 20)
                ws.setValue(int(self._vis.get(w_key, 1)))
                ws.setFixedWidth(56)
                ws.setToolTip(w_tip or "这一项的**线条粗细(px)**（1 = 最细）。")
                self._width_spins[w_key] = ws
            if not isinstance(form, QGridLayout):
                # ── 老写法（`QFormLayout`：类别框那一组 ✓ 没有粗细）
                if not on_key:
                    form.addRow(label, btn)
                    return
                ck = QCheckBox(label)
                ck.setChecked(bool(self._vis.get(on_key, True)))
                ck.setToolTip(on_tip or "关掉 = **不画**这一项（颜色留着，开回来还在 ✓）")
                self._vis_on[on_key] = ck
                row = QHBoxLayout()
                row.setSpacing(6)
                row.addWidget(ck)
                row.addWidget(btn)
                row.addStretch(1)
                form.addRow("", row)
                return
            # ── ⭐ 网格版（辅助线与标记这一组 ✓）
            r = _grid_rows[0]
            _grid_rows[0] += 1
            if on_key:
                ck = QCheckBox(label)
                ck.setChecked(bool(self._vis.get(on_key, True)))
                ck.setToolTip(on_tip or "关掉 = **不画**这一项（颜色留着，开回来还在 ✓）")
                self._vis_on[on_key] = ck
                form.addWidget(ck, r, 0)
            else:
                form.addWidget(QLabel(label), r, 0)
            form.addWidget(btn, r, 1)
            if ws is not None:
                form.addWidget(ws, r, 2)

        grp = QGroupBox("检测框颜色")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        page_lay.addWidget(grp)
        vf = QFormLayout()
        vf.setLabelAlignment(Qt.AlignLeft)
        lay.addLayout(vf)

        # 类别框颜色**按类别表生成**：有几个类别就有几行，以后加类别这里自动多一行，
        # 不会出现「新类别没地方改颜色」。配置键 = 英文名 + "_color"（见 gui/theme.py）。

        # 类别框颜色**按类别表生成**：有几个类别就有几行，以后加类别这里自动多一行，
        # 不会出现「新类别没地方改颜色」。配置键 = 英文名 + "_color"（见 gui/theme.py）。
        for cid, en, zh, _bgr in classes.CLASSES:
            add_row(vf, "%s框颜色" % zh, "%s_color" % en,
                    "类别 %d（%s）—— 检测框颜色" % (cid, en))

        # ---- ④ 辅助线与标记（不是类别，是给操作者看的参照 ✓）----
        grp = QGroupBox("辅助线与标记（实时预览）")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        page_lay.addWidget(grp)
        # ⭐⭐ 这一组用 **QGridLayout**（用户 2026-09-28 四条 ✓ 原话："粗细调整参数放**颜色配置
        #   右边**，并**给其他的粗细也加配置**，并且用**网格对齐**，要求**参数名显示完整**"）。
        #   三列：**开关（名字写在上头）| 色块 | 粗细框** ⇒ 各行的色块 / 粗细框**上下对齐** ✓；
        #   列 0 的最小宽度给够 ⇒ 最长的名字也**完整显示**（不截断、不省略 ✓）。
        vf2 = QGridLayout()
        vf2.setHorizontalSpacing(8)
        vf2.setVerticalSpacing(6)
        vf2.setColumnMinimumWidth(0, 190)      # ⚠ 名字那一列：够长 ⇒ 不截断 ✓
        vf2.setColumnStretch(3, 1)             # 右边留白吸收 ✓
        lay.addLayout(vf2)
        # ⚠ 2026-09-27 用户改的三件事（照办）：
        #   ① 名字末尾的「颜色」两字**去掉**（废话、占地方）—— 名字交给左边的**开关**，
        #      右边只留色块 ✓；
        #   ② **每项前面加开关**（`*_on`：关掉 = 不画这一项，颜色留着 ✓）；
        #   ③ 「最大/最小攻击距离线颜色」两项**换掉**：攻击范围从"一条线"变成了**矩形**
        #      ⇒ 现在是「攻击范围框 / 攻击盲区框 / 跳跃攻击范围框」三项 ✓
        #      （前两个框的画法见 `gui/live_thread.py`；第三个是占位，逻辑待跳跃物理 ✓）。
        add_row(vf2, "锁定框", "lock_color",
                "锁定的那个攻击目标（只有它会被红框圈出来 ✓）。",
                on_key="lock_on", w_key="lock_width")
        add_row(vf2, "攻击范围框", "attack_color",
                "**攻击范围 = 矩形**（2026-09-27 起）：四个距离在「决策参数 → 战斗参数 →\n"
                "攻击」里（最大 / 最小 / 向上 / 向下攻击距离）。\n\n"
                "这一段画的是「**可攻击区**」= 从「**最小攻击距离**」到「**最大攻击距离**」\n"
                "（**不含**盲区那一块 —— 盲区由上面那个框画 ✓，两个框并排不重叠）。\n"
                "竖直那两条的取值：**负数 = 不限**（画到画面边）、**0 = 就是 0**、\n"
                "上下都配 0（或水平跨度 0）⇒ 框是空集 ⇒ **不画也不判** ✓。",
                on_key="attack_on", w_key="attack_width")
        add_row(vf2, "攻击盲区框", "min_attack_color",
                "「最小攻击距离」以内那一块（太近 ⇒ 不算可攻击、并触发规避）。\n"
                "最小攻击距离 = 0（默认）时**没有盲区** ⇒ 这个框不画 ✓。",
                on_key="min_attack_on", w_key="min_attack_width")
        add_row(vf2, "跳跃攻击范围框", "jump_attack_color",
                "**占位项**（用户 2026-09-27 记的一笔）：等跳跃物理做好后，它表示\n"
                "「怪框落在这个范围内 ⇒ 按跳就能把它带进攻击范围框 ⇒ 触发 attack」。\n"
                "⚠ 逻辑还没实现 ⇒ 现在画面上**不会**出现这个框（先把颜色定下来 ✓）。",
                on_key="jump_attack_on", w_key="jump_attack_width")
        add_row(vf2, "追击起跳框", "chase_jump_color",
                "「**追击起跳**」的起跳区间（决策参数 → 战斗参数 → 攻击 → 追击起跳）：\n"
                "区间 = [最大攻击距离 + min, 最大攻击距离 + max] ✓。\n\n"
                "⚠ 现在的**高度跟「攻击范围框」一样**；用户 2026-09-27 说：后续要\n"
                "**算上「跳跃攻击范围」的高度**（框会长高），**当前先不算** ✓。\n"
                "⚠ 它的**判定**仍然是**水平距离**（不看高度）✓ —— 画成框只是让人看得见 ✓。",
                on_key="chase_jump_on", w_key="chase_jump_width")
        add_row(vf2, "视野线", "vision_color", "视野矩形那四条虚线。",
                on_key="vision_on", w_key="vision_width")
        add_row(vf2, "定时任务", "timer_color",
                "实时画面上「当前任务 / 定时任务」那几行的字色。\n"
                "那里**没有底色**（只有描边），所以别选太暗的 —— 会压不住游戏画面。",
                on_key="timer_on")      # ⚠ 这是**字色** ⇒ 没有粗细，不塞那一格 ✓

        # ⭐⭐ **玩家坐标箭头**（用户 2026-09-28 ✓ 原话："放在**设置 → 界面页签 → 辅助线与标记
        #   （实时预览）**"）—— ⚠ 它属于**这一组**，不属于「决策参数」✗：
        #   规范 §4 的判据就是"**画面上每个可显示的东西**都该能在这一组里单独关掉" ✅
        #   （所以它有 `player_arrow_on` 开关 ✓，颜色走同一套带 alpha 的取色弹窗 ✓）。
        #   ⚠ 颜色 + **粗细**都照 `add_row`（粗细就在色块右边那一格 ✓ 用户 2026-09-28 要求 ✓）；
        #     只有「长度」多占一行（网格里跨 3 列 ✓）。
        add_row(vf2, "玩家坐标箭头", "player_arrow_color",
                "以**玩家脚底**为原点画两根：一根朝画面右（世界 x 正方向）、一根朝上\n"
                "（世界 y 正方向）。\n\n"
                "⚠ 它**不参与任何决策**，就是拿来对着调「决策参数 → 玩家位置 → 脚底偏移(px)」\n"
                "的：箭头根部该正好落在脚底 ✓（偏上 / 偏下就是偏移没调对 ✓）。",
                on_key="player_arrow_on", w_key="player_arrow_width")

        self.sp_arrow_len = NoWheelSpinBox()
        self.sp_arrow_len.setRange(0, 2000)
        self.sp_arrow_len.setValue(int(self._vis["player_arrow_len"]))
        # ⚠ 长度另起一行：网格版要**跨 3 列**摆（`addRow` 是 `QFormLayout` 的 API ✗）
        _lr = _grid_rows[0]
        _grid_rows[0] += 1
        vf2.addWidget(QLabel("箭头长度(px)（0 = 不画）"), _lr, 0)
        vf2.addWidget(self.sp_arrow_len, _lr, 2)

        # ⭐ **箭头尖大小**（用户 2026-09-28 追加 ✓ 原话："再加个箭头 size 配置"）——
        #   就是**尖端那个三角头**的大小 ✓，单位 = 占**线段长**的百分比（`cv2.arrowedLine`
        #   的 `tipLength` 只吃比例 ✗ 不吃像素 ✓ 所以按百分比存）。
        #   ⚠ 与「长度」**同列**摆（列 0 标签 / 列 2 数值 ✓）⇒ 和上面所有行的数值**一条竖线**
        #     对齐 ✓（用户上一条要求"网格对齐 + 参数名显示完整" ✓ 这行也守着 ✓）。
        self.sp_arrow_tip = NoWheelSpinBox()
        self.sp_arrow_tip.setRange(5, 100)
        self.sp_arrow_tip.setValue(int(self._vis["player_arrow_tip_pct"]))
        self.sp_arrow_tip.setToolTip(
            "箭头**尖端那个三角头**的大小，单位 = **占线段长的百分比**\n"
            "（25 = 尖长正好是线段的 1/4）。调大 ⇒ 箭头越「胖头」；\n"
            "线段本身太短时这个尖会显得很大 ⇒ 两者搭配着调。")
        _tr = _grid_rows[0]
        _grid_rows[0] += 1
        vf2.addWidget(QLabel("箭头尖大小(%)（占线长）"), _tr, 0)
        vf2.addWidget(self.sp_arrow_tip, _tr, 2)

        # ---- ⑤ foothold 集合编辑器（只管它那个窗口，和实时预览无关 ⇒ 单独一组）----
        grp = QGroupBox("foothold 集合编辑器 · 线宽(px)")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        page_lay.addWidget(grp)
        _note = QLabel("地形线条在编辑器里画多粗。\n"
                       "点确定后**再打开一次**编辑器就按新值画（窗口一直开着的话，"
                       "点一次「寻路编辑器」也会刷新）。")
        _note.setStyleSheet("color: #5f6368;")
        _note.setWordWrap(True)
        lay.addWidget(_note)
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
                   "**依据 = 小地图找不到黄点**（用户 2026-09-28 ✓）：连续这么多分钟\n"
                   "**拿不到可用位置**（黄点没认出来 / 位置只能沿用上一帧）⇒ 自动停止。\n"
                   "⚠ 它**不看**主画面里认没认出人物框 —— 框好好的但黄点糊了 / 被挡住，\n"
                   "照样没有可用位置 ⇒ 照样算「找不到」✓（0 = 禁用）。\n"
                   "⚠ 这一格同时决定**留痕**：超过该时长时，perf.log 段头会写上"
                   "「画面里看不到角色 N 分钟（死亡界面/弹窗/切图？）」，并记一次 "
                   "player_gone；\n"
                   "—— 它**与「自动开着没」无关**（自动已经关着也照样记），"
                   "因为事后最常问的就是「那段时间到底是死了、卡了、还是没开自动」。\n"
                   "0 = 连这条注记也不记。")
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

    def _fill_goto_timeout_key(self):
        """填「**寻路超时后按键**」下拉：**（不按）** + 固定键 + **自定义按键** ✓。

        ⚠ **每次显示设置窗都要重填**（见 `showEvent` ✓）：自定义按键是**可增删**的
        （「自定义按键」组里随时加/删 ✓）⇒ 只在 `__init__` 里填一次的话，
        新加的键选不到、删掉的键还留在列表里 ✗。

        ⚠ 「反方向 / 目标方向」**排除**：`SEQ_KEYS` 里那两个是"执行时按**当时朝向**解析"的
        特殊键 ✗（见 `decision/agent.py::_run_seq` 的 `back`/`forward` ✓）——
        超时那一刻没有朝向可说，选了也执行不了 ✗。
        """
        from decision.agent import settings as _s
        from gui.seq_editor import SEQ_KEYS

        cur = getattr(_s, "goto_timeout_key", None)
        self.cb_goto_timeout_key.clear()
        self.cb_goto_timeout_key.addItem("（不按）", None)     # 默认项 ✓ data = None ✓
        for disp, key in SEQ_KEYS:
            if key in ("back", "forward"):
                continue
            self.cb_goto_timeout_key.addItem(disp, key)
        for name in (_s.custom_keys or {}):
            self.cb_goto_timeout_key.addItem("自定义：%s" % name, name)
        idx = self.cb_goto_timeout_key.findData(cur)
        self.cb_goto_timeout_key.setCurrentIndex(idx if idx >= 0 else 0)

    def showEvent(self, ev):
        """设置窗每次露头就把「寻路超时后按键」的选项重填一遍 ✓
        （自定义按键可能刚加/刚删 —— 单选一份会让列表过期 ✗ 见 `_fill_goto_timeout_key` ✓）。"""
        super().showEvent(ev)
        try:
            self._fill_goto_timeout_key()
        except Exception:      # noqa: BLE001 —— 填选项这种小事不该拦住整个设置窗 ✗
            pass

    def _page_judge(self):
        """**判定**用参数：哪些事算"成立"。

        为什么单独一页：这一页回答的既不是"长什么样"（界面）、也不是"出问题怎么办"
        （保护与恢复），而是"**怎么算数**" —— 同一个动作（对齐到某个坐标）成不成功、
        由哪几个数说了算。以后爬绳 / 寻路的判据都挂在这儿，塞进别页会找不着。
        """
        page, lay = self._page()

        # ⭐ **顶层开关：禁用杀怪寻路**（用户 2026-09-28 要求 ✓ 原话："设置里**判定参数顶层**加个
        #   开关「禁用杀怪寻路」：开启后**不再查询怪物框所属的 foothold 集合**（**也不显示**），
        #   **不再因追怪而下达寻路任务**，而只是**单纯地走向锁定怪物**（**无寻路时的老逻辑**）"✓）。
        # ⚠ 放在**表单之外的最上面**（"顶层"✓）：它管的是**这一页那些判据要不要参与**
        #   （追击寻路整条链 ✓），不属于"某个判据的数值" ⇒ 塞进 `form` 里会和参数混在一起 ✗。
        self.ck_no_chase_path = QCheckBox("禁用杀怪寻路")
        self.ck_no_chase_path.setChecked(
            bool(getattr(settings, "disable_chase_pathfinding", False)))
        self.ck_no_chase_path.setToolTip(
            "**开了之后，追怪不再走寻路**：\n\n"
            "· **不再查询**「怪物框底下是哪块 foothold 集合」（画面上那行\n"
            "  `查#怪号 集合名` 也不再出现 ✓）；\n"
            "· **不再因为追怪而下达寻路任务**（含「朝方向逐层逼近」那条降级 ✓）；\n"
            "· 只是**单纯地朝锁定怪物走**（= 没有寻路时的老逻辑：只按 ←/→ ✓）。\n\n"
            "什么时候开：寻路那条线总判不准、或者只想让它先凑过去打的时候 ✓。\n"
            "关了（默认）就是老行为：判得出怪在哪块平台就先下「前往」过去 ✓。\n\n"
            "⚠ 它**不影响**这两件事（那是别的功能）：\n"
            "· 「编辑战斗区域」里勾了「禁止战斗」时的**区域筛选**（那也在查集合，\n"
            "  但它服务的是「别追出禁战区」，不是追怪寻路 ✗）；\n"
            "· 追击起跳（由那个独立开关管 ✓）。")
        lay.addWidget(self.ck_no_chase_path)

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

        # ---- ⭐⭐ 「玩家位置」（用户 2026-09-28 ✓：它属于**判定参数**）----
        #   为什么在这一页：这几个数决定"**这一拍算不算拿到了玩家位置**" ——
        #   ① 脚底偏移改的是「画面 → 世界」的换算（相机 y ✓）；②③④ 是**框面积闸**
        #   （框太小 / 比近期基线小太多 ⇒ 这一拍**不做位置查询** ✓）。
        #   两条理由都成立：既是"**怎么算数**"（本页的题 ✓）、又是"**跟项目走**"（不同地图的
        #   面 y、人物框大小都不一样 ✓）⇒ 该在这一页。
        #   ⛔ **别再留在「决策参数」面板**（2026-09-28 犯过一次 ✗ 用户指出）✓。
        #   ⚠ 箭头（颜色 / 粗细 / 长度）**不在这儿** —— 那是"本机外观"（不参与决策 ✓），
        #     归 **设置 → 界面 → 辅助线与标记（实时预览）** ✓（分界判据见 `docs/UI规范.md` §4）。
        _ploc = QGroupBox("玩家位置")
        _plf = QFormLayout(_ploc)
        _plf.setLabelAlignment(Qt.AlignLeft)
        _plf.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        def pl_row(label, w, tip):
            w.setToolTip(tip)
            _plf.addRow(label, w)
            return w

        self.sp_foot_off = NoWheelSpinBox()
        self.sp_foot_off.setRange(-500, 500)
        self.sp_foot_off.setValue(
            int(getattr(settings, "player_foot_offset_px", 0) or 0))
        pl_row("脚底偏移(px)", self.sp_foot_off,
               "算「脚底 / 相机 y」时用「**框底 + 这个值**」（正数 = 往下挪）。\n\n"
               "人物框的底边**不一定**正好压在脚底（鞋底阴影 / 披风 / 特效会让框多出一截）。\n"
               "在实时预览上对着「辅助线与标记 → **玩家坐标箭头**」调：\n"
               "箭头根部该正好落在脚底 ✓")

        # ⭐⭐ **「面积基线窗口(s)」紧挨着「脚底偏移」**（用户 2026-09-28 ✓ 原话："该参数应该
        #   放在『**脚底偏移**』参数下面"）；⚠ 单位**从「拍」改成「秒」**（用户原话："这个『拍』
        #   是什么？是多久？**需要可量化的描述**"✓）—— 「拍」依赖帧率：30fps 的 30 拍是 1 秒、
        #   15fps 就变成 2 秒 ⇒ **说不清到底多久** ✗ ⇒ 只有**秒**是可量化的 ✓。
        self.sp_area_base_s = NoWheelDoubleSpinBox()
        self.sp_area_base_s.setRange(0.5, 60.0)
        self.sp_area_base_s.setDecimals(1)
        self.sp_area_base_s.setSingleStep(0.5)
        self.sp_area_base_s.setValue(float(
            getattr(settings, "player_box_area_base_s", 3.0) or 3.0))
        pl_row("面积基线窗口(s)", self.sp_area_base_s,
               "拿**过去这么多秒**里采到的「玩家框面积」求均值，当「**正常大小**」✓。\n\n"
               "⚠ 单位是**秒**，不是「拍」✗ —— 拍数依赖帧率（30fps 的 30 拍 = 1 秒，"
               "15fps 就成 2 秒），说不清到底多久 ✓。\n"
               "默认 3 秒：够长能抹平抖动，又不至于把「人物走近 / 走远导致框变大变小」算进去 ✓。")

        self.sp_box_min_area = NoWheelDoubleSpinBox()
        self.sp_box_min_area.setRange(0.0, 100.0)
        self.sp_box_min_area.setDecimals(3)
        self.sp_box_min_area.setSingleStep(0.1)
        self.sp_box_min_area.setValue(float(
            getattr(settings, "player_box_min_area_pct", 0.0) or 0.0))
        pl_row("框面积最小占比(%)", self.sp_box_min_area,
               "框面积小于「**上面那个基线 × 这个比例**」⇒ 这一拍**不算有效检测** ✓。\n\n"
               "⚠ 是**跟基线比**（= 过去若干秒的平均框面积 ✓），**不是**跟画面面积比 ✗ ——\n"
               "不同地图 / 不同分辨率下人物框本来就不一样大，跟画面比没有可比性 ✓。\n"
               "人太小 / 框抖掉了 ⇒ 位置不可信，宁可这一拍不定位 ✓。\n0 = 不启用。")

        self.sp_area_tol = NoWheelDoubleSpinBox()
        self.sp_area_tol.setRange(0.0, 90.0)
        self.sp_area_tol.setDecimals(1)
        self.sp_area_tol.setSingleStep(1.0)
        self.sp_area_tol.setValue(float(
            getattr(settings, "player_box_area_tol_pct", 0.0) or 0.0))
        pl_row("面积容差(%)", self.sp_area_tol,
               "当前框面积 ≤ 基线 ×(1−容差%) ⇒ 这一拍**推迟 / 不做**位置查询。\n\n"
               "⚠ 与「框面积最小占比」的分工：**最小占比**是**更狠的硬下限**"
               "（低于它连检测都不算 ✓）；**容差**是**更松的一档**（只让这一拍先别查 ✓）。\n"
               "一般「最小占比」要**小于**「容差」才有意义（先判更狠的那道 ✓）。\n"
               "0 = 关掉这道闸（老行为）。")

        form.addRow("", _ploc)

        # ---- ⭐ 子组：**任务**（用户 2026-09-28 要求 ✓ 原话："在设置 → 判定参数**单开一组
        #      『任务』**"）----
        # ⚠ 为什么单开一组：它管的**不是某个执行器怎么走**，而是"**这一趟任务该不该继续**"✗
        #   —— 跟上面那些"怎么走 / 走多快 / 走多久"不是一类 ✓（用户点名要单开 ✓）。
        # ⚠ 只放**判定参数**这一页（就是本页 ✓）—— 2026-09-28 我自己犯过一次"加到了主界面的
        #   `route_panel`（决策参数面板）"✗，用户当场找不到 ✓ 别再犯 ✗。
        _task_grp = QGroupBox("任务")
        _tf = QFormLayout(_task_grp)
        _tf.setLabelAlignment(Qt.AlignLeft)
        _tf.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.sp_chase_max = NoWheelDoubleSpinBox()
        self.sp_chase_max.setRange(0.0, 600.0)
        self.sp_chase_max.setDecimals(1)
        self.sp_chase_max.setSingleStep(1.0)
        # 🔴🔴 **回填**（2026-09-28 补 ✓ —— 我第一版**漏了这一行**，用户当场报"改不了
        #   追怪寻路.duration、也没有任何提示"✗）。
        #   ⚠ 漏了它的症状**特别像"写回坏了"**：点确定**其实存进去了**（所以不会弹框 ✓），
        #     但**下次打开又显示 0** ⇒ 看起来就是"改了没用"✗。
        #   ⚠ **教训**：新加一个控件，**回填**和**写回**是**两件事**，缺一不可 ✗ ——
        #     只测"设值 → 点确定 → settings 变了"**是测不出漏回填的** ✓
        #     必须再测一条："**配置里是非默认值 → 打开对话框 → 控件显示它**" ✓。
        self.sp_chase_max.setValue(
            float(getattr(settings, "chase_goto_max_s", 0.0) or 0.0))
        self.sp_chase_max.setToolTip(
            "**追怪下达的**寻路任务最多跑这么久（秒），超了就**结束这趟任务**。\n\n"
            "为什么需要：怪会跑、会换平台，追击有时会变成「一层一层追下去」，\n"
            "把时间都耗在这一趟上。这条**只掐「追怪」那一类任务** ✓ ——\n"
            "「命令前往」「定点休息」「回战斗区域」等**都不受它管** ✓。\n\n"
            "0 = 不启用（**默认**，老行为一字不变）。\n"
            "⚠ 与「寻路超时时间」是**两道独立的闸**：这道更专（只掐追击），通常设得更短；\n"
            "   两个用的是**同一把钟**（这一段任务的起点）⇒ 不会互相错位 ✓。")
        _tf.addRow("追怪寻路.duration(s)", self.sp_chase_max)
        form.addRow("", _task_grp)

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
            "6 够吸住，又不会宽到把隔壁平台的边也算进来。\n\n"
            "**还管一处**（2026-09-26 加）：判「**脚下是哪条平台 / 在不在某个集合**」时，\n"
            "先用 0 容差找，**一条都找不到**才按这个值放宽一次 —— 人站在平台**边上**时，\n"
            "小地图黄点（画的是玩家中心）会读到边界外几个像素，不兜这一下就会报\n"
            "「脚下没有平台 / 未分组 / 起点不知道」（实测 105090600 的 (650,283)：\n"
            "离平台左边界 7px，而这里填的是 10px ⇒ 现在能认出来 ✓）。\n"
            "⚠ 只影响**原本找不到**的情形 ⇒ 已经判对的结果一个都不会变 ✓。")

        # ② 坐标对齐误差时间
        self.sp_align_hold = NoWheelSpinBox()
        self.sp_align_hold.setRange(0, 5000)
        self.sp_align_hold.setValue(int(settings.align_hold_ms))
        row("坐标对齐误差时间(ms)", self.sp_align_hold,
            "进误差范围后要**保持这么久**才算对齐成功；\n"
            "上绳**到顶收工后**也要再按住 ↑ 这么久才松（2026-09-26 用户要求）。\n"
            "为什么不能只看一帧：小地图定位与按键下发不是同一时刻（有延迟、还会抖），\n"
            "单帧落进误差范围不代表真站住了。\n"
            "⚠ 两处口径不同（2026-09-28 用户重新审视）：\n"
            "· **对齐稳住** = 这个值 **+ 端到端延迟**（对齐靠读数，要补延迟）；\n"
            "· **收工后按住 ↑** = **纯这个值**（游戏机制：迈上平台那一步要按着 ↑，与延迟无关）。\n"
            "默认 250：约 3~4 个视频帧，够滤掉抖动，又不至于每步都干等。")

        # ③ 寻路超时时间（2026-09-26 新增；当天又明确了口径：按**每一段**算）
        self.sp_goto_timeout = NoWheelSpinBox()
        self.sp_goto_timeout.setRange(0, 3600)
        self.sp_goto_timeout.setValue(int(round(float(
            getattr(settings, "goto_timeout_s", 30.0) or 0.0))))
        row("寻路超时时间(s)", self.sp_goto_timeout,
            "⭐ **只对「追击」签注的任务生效**（用户 2026-09-29 定的口径 ✓）：\n"
            "追击是「追不上就别追了」那种语义 ⇒ 到点切断它，画面上会写明\n"
            "「寻路超时：这一段已经跑了 N 秒」。\n\n"
            "⛔ **其它任务一律不限时**（命令前往 / 定点休息 / 回战斗区域 / 战斗时长到点\n"
            "换地方…）：一条长路线（走→爬→走）跑两分钟很正常，拿 30 秒去切它是**误伤** ✗\n"
            "⇒ 那些任务的「当前任务」那行也**不再显示剩余**（显示剩余 = 让人以为它要被切 ✗）。\n\n"
            "⚠ 追击任务里**每完成一段就重新计时**（走完一段接下一段 ⇒ 预算是新的 ✓），\n"
            "但**攻击逻辑重下同一趟追击不重新计时**（怪换层了要改目的地，那不算新任务 ✓）\n"
            "—— 不然一趟永远追不上的追击会被「重下」无限续命 ✗。\n\n"
            "⚠ 各执行器自己的「卡住重试 / 放弃」（爬绳补按 ↑、下跳重试…）**不受这里影响** ✓\n"
            "⇒ 真卡住仍然各自处理 ✓；这道闸只管**总时长**。\n\n"
            "0 = 不限时。默认 30。")

        # ④ ⭐ **寻路超时后按键**（用户 2026-09-28 要求）：寻路**超时切断**那一下，额外
        #    **点按**这个键。类型 = **自定义按键下拉列表**（同行为编辑器：固定键 + 你自己
        #    在「自定义按键」组里加的那些键 ✓）。
        #    ⚠ 「反方向 / 目标方向」**不进这个列表**：那两个是行为序列里"执行时按当时朝向
        #      解析"的特殊键 ✗ —— 超时那一刻没有朝向可言，选了也没法执行 ✗。
        self.cb_goto_timeout_key = NoWheelComboBox()
        self.cb_goto_timeout_key.setMinimumWidth(120)
        self._fill_goto_timeout_key()
        row("寻路超时后按键", self.cb_goto_timeout_key,
            "「寻路超时时间」到点、任务被切断的那一下，额外**点按**这个键（按下再松开 ✓）。\n"
            "用途：卡住时给游戏一个信号（打断当前动作 / 取消某个状态 之类）。\n\n"
            "· **（不按）** = 老行为（默认 ✓ —— 一个键都不多发）；\n"
            "· 列表 = 固定键 + 你在「自定义按键」组里加的那些键 ✓\n"
            "  （自定义键显示成「自定义：<名字>」✓）。\n\n"
            "⚠ 「反方向 / 目标方向」不在列表里 —— 那两个是行为序列里「按当时朝向解析」的\n"
            "特殊键，超时那一刻没有朝向可说，选了也执行不了。\n"
            "⚠ 只在**超时切断**那一下发一次（不是每拍发 ✓）。")

        # ⚠ 这里原来有「**卡住判定时长(s)**」那一格 —— **2026-09-27 用户要求移除** ✓：
        #   原话："移除卡住判定时长(s)，**统一采用移动操作尝试间隔(ms)**" ✓。
        #   那些"多久没进展就算卡住"的判据现在统一读 **「移动操作尝试间隔(ms)」**
        #   （在「路线识别 → 寻路配置」组里 ✓，一处配置、多处判据 ✓）；
        #   走（`WalkJob`）照旧用 `route.STALL_S` 常量 + 「寻路超时时间」兜底 ✓。

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

    @staticmethod
    def _rgba_css(color):
        """颜色串 → CSS `rgba(r, g, b, a)` —— 色块按钮要能看出**透明度** ✓。

        ⚠ 不能直接把颜色串塞进样式表：`#AARRGGBB` 这种写法 **Qt 样式表不认**
          （它只认 `#RRGGBB` / `rgba(...)`）⇒ 配了透明度的按钮会变成默认灰 ✗。
        """
        from PyQt5.QtGui import QColor
        c = QColor(color)
        return "rgba(%d, %d, %d, %d)" % (c.red(), c.green(), c.blue(), c.alpha())

    def _color_btn(self, color):
        """一个显示当前颜色（含透明度）、点击弹颜色选择器的按钮。"""
        from PyQt5.QtGui import QColor
        c0 = QColor(color)
        btn = QPushButton(c0.name(QColor.HexArgb).upper())
        btn.setFixedWidth(110)
        btn._color = c0.name(QColor.HexArgb)
        btn.setToolTip("当前颜色（含透明度）：点击选择 —— 弹窗里有**透明度拖动条** ✓")
        btn.setStyleSheet(
            "QPushButton { background: %s; color: #202124; border: 1px solid #dadce0;"
            " border-radius: 4px; padding: 4px 8px; }" % self._rgba_css(color))
        btn.clicked.connect(lambda: self._pick_color(btn))
        return btn

    def _pick_color(self, btn):
        """弹颜色选择器 —— **带透明度拖动条**（用户 2026-09-27 要求）。

        ⚠ 两个选项缺一不可：
          · `ShowAlphaChannel` —— 那个"Alpha 通道"**拖动条**就是它给的 ✓；
          · `DontUseNativeDialog` —— **必须**！Windows 上默认走系统颜色弹窗，
            而系统那个**没有** alpha 条 ✗（加了 ShowAlphaChannel 也不显示 ⇒ 用户会
            以为"说好的拖动条呢"）。
        选完存 `#AARRGGBB`（`QColor.name(HexArgb)` ✓）—— 老写法 `name()` 只有 6 位，
        透明度会**当场丢掉** ✗。
        """
        from PyQt5.QtGui import QColor
        c = QColorDialog.getColor(
            QColor(btn._color), self, "选择颜色（可调透明度）",
            QColorDialog.ShowAlphaChannel | QColorDialog.DontUseNativeDialog)
        if c.isValid():
            btn._color = c.name(QColor.HexArgb)
            btn.setText(c.name(QColor.HexArgb).upper())
            btn.setStyleSheet(
                "QPushButton { background: %s; color: #202124;"
                " border: 1px solid #dadce0; border-radius: 4px; padding: 4px 8px; }"
                % self._rgba_css(btn._color))

    # ---------------- 实时画面叠图 ----------------

    def _on_alpha_changed(self, v):
        """拖动浓淡 ⇒ 只更新那两个标签（**点确定才写**，和这一页其它项一致 ✓）。"""
        self.lbl_alpha_val.setText("%d%%" % int(v))
        if not self._map_id:
            self.lbl_alpha_note.setText(
                "⚠ 还没打开项目 / 项目里还没选地图 —— 浓淡是**按图存**的，现在改了没处存。")
        elif self.ck_mmap_draw.isChecked():
            self.lbl_alpha_note.setText(
                "当前地图「%s」的叠加层浓淡：%d%%（按图各存一份，改完点确定 ✓）"
                % (self._map_id, int(v)))
        else:
            self.lbl_alpha_note.setText("叠图关着 —— 勾上「在实时画面上叠地形图」才有意义。")

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

    def _accept_safe(self):
        """⚠ **包一层**：保证「确定」**一定关得掉**、而且写回失败**不许静默**（用户 2026-09-28 报
        "今天设置里点确定却修改无效的事情发生好多次了"✓）。

        为什么非要有它：`_accept` 从第一项写回一路到最后的 `self.accept()`，**全是裸语句**
        （没有 try ✗）⇒ 中途**任何一步抛异常**就会：
          ① 它**后面**那些项的写回**全被跳过** ⇒ 用户看到的"**改了没用**"✗
          ② `self.accept()` 在**最后** ⇒ **对话框关不掉** ⇒ "点了确定没反应"✗
        两种症状**交替出现**（取决于哪一项坏了）⇒ 正是用户说的"**好多次、时好时坏**" ✓。

        ⇒ 现在：异常**抓下来**（控制台打堆栈 + 弹框说清是哪一项 ✓）、**并且照样把对话框关掉** ✓。
        ⚠ 写回**到哪儿为止**是**没法回滚**的（前面几项已经写了 ✓）—— 所以弹框里要提醒用户
          "后面那几项这次没生效，再点一次确定 / 重设一遍" ✗（别让他以为全都存好了 ✓）。
        """
        _ok = False
        try:
            self._accept()
            _ok = True
        except Exception as _e:                      # noqa: BLE001 —— 坏在哪一项都要报出来 ✓
            import traceback

            traceback.print_exc()
            try:
                from PyQt5.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self, "设置没能全部写回",
                    "点「确定」时有一项写回失败了：\n\n%s\n\n"
                    "⚠ 它**后面**那些项的修改**这次没生效** —— 对话框关掉后再点一次确定，\n"
                    "或者把那几项重新设一遍。\n\n"
                    "（详细堆栈在控制台里）" % (_e,))
            except Exception:                        # noqa: BLE001
                pass
        if not _ok:
            self.accept()                            # ⚠ **出错也必须关得掉** ✓

    def _accept(self):
        # ---- 实时画面 · 地形叠加（2026-09-26 从路线识别页搬来的两项）----
        _draw = bool(self.ck_mmap_draw.isChecked())
        _alpha = int(self.sld_alpha.value())
        if _draw != self._draw0:
            update_live(mmap_draw=_draw)
        if self._map_id and _alpha != self._alpha0:
            try:
                _src = mm.live_src(load_live())
                _cal = mapdata.load_calib(self._map_id, _src) or {}
                _cal["alpha"] = _alpha
                mapdata.save_calib(self._map_id, _cal, _src)
            except Exception:                       # noqa: BLE001
                pass
        if _draw != self._draw0 or _alpha != self._alpha0:
            # 那一层是路线识别面板画的 ⇒ 叫它一声，立刻按新配置重画 ✓
            self.overlay_changed.emit()

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
        # ⭐ 「玩家位置」（用户 2026-09-28 ✓：搬进**判定参数**页 ✓ —— 这几个数跟项目走 ✓）
        _fo = int(self.sp_foot_off.value())
        if _fo != int(getattr(settings, "player_foot_offset_px", 0) or 0):
            settings.player_foot_offset_px = _fo
            settings.save()
        # ⭐ 「**追怪寻路.duration(s)**」（用户 2026-09-28 ✓）—— 单位**秒**（`float` ✓），
        #   `0` = 不启用 ✓；跟项目存 ✓（与「寻路超时时间」同一口径 ✓）。
        _cm = float(self.sp_chase_max.value())
        if abs(_cm - float(getattr(settings, "chase_goto_max_s", 0.0) or 0.0)) > 1e-9:
            settings.chase_goto_max_s = max(0.0, _cm)
            settings.save()
        _ma = float(self.sp_box_min_area.value())
        if abs(_ma - float(getattr(settings, "player_box_min_area_pct", 0.0) or 0.0)) > 1e-9:
            settings.player_box_min_area_pct = _ma
            settings.save()
        # ⭐ 「面积基线窗口」**单位改成秒**（用户 2026-09-28 ✓："『拍』是什么？是多久？需要可
        #   量化的描述"✓）⇒ 键名也换成 `player_box_area_base_s` ✓（老键 `_n` 由 `from_dict`
        #   兜底读一次 ✓ 不让老项目丢值 ✗）。
        _bs = float(self.sp_area_base_s.value())
        if abs(_bs - float(getattr(settings, "player_box_area_base_s", 3.0) or 3.0)) > 1e-9:
            settings.player_box_area_base_s = max(0.5, _bs)
            settings.save()
        _at = float(self.sp_area_tol.value())
        if abs(_at - float(getattr(settings, "player_box_area_tol_pct", 0.0) or 0.0)) > 1e-9:
            settings.player_box_area_tol_pct = _at
            settings.save()
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
        # ⭐ 「寻路超时后按键」（2026-09-28 加 ✓，跟着项目存 ✓）
        #    `currentData()` 就是键名（"（不按）"那一项的 data 是 `None` ✓ 见 `_fill_...`）
        gk = self.cb_goto_timeout_key.currentData()
        if gk != getattr(settings, "goto_timeout_key", None):
            settings.goto_timeout_key = gk
            settings.save()
        # ⭐ 「**禁用杀怪寻路**」（2026-09-28 加 ✓）—— 和上面那些一样**跟着当前项目存** ✓
        #   （字段进 `DecisionSettings.to_dict` ⇒ 自动跟着 `project.yaml` 的 `decision:` 段 ✓）。
        ncp = bool(self.ck_no_chase_path.isChecked())
        if ncp != bool(getattr(settings, "disable_chase_pathfinding", False)):
            settings.disable_chase_pathfinding = ncp
            settings.save()
        # ⚠ 「卡住判定时长(s)」那一格的写回**已移除**（2026-09-27 用户要求：统一用
        #   「移动操作尝试间隔(ms)」✓）—— 那个键以后不再写；老配置里残留的值也不再读 ✓。
        # 可视化（颜色现在可能是 `#AARRGGBB` —— 带透明度 ✓，见 `_pick_color`）
        vis_cfg = {k: b._color for k, b in self._color_btns.items()}
        # ⭐⭐ **各框线宽**（用户 2026-09-28 ✓"给其他的粗细也加配置"）—— 网格里色块右边那一格
        #   就是它（`add_row(..., w_key=...)` 造好的 spin，见 `self._width_spins` ✓）；
        #   键名与 `gui/theme.VIS_DEFAULTS` **一一对应** ✓（`load_vis` 那边也加了同一批的
        #   "读回 + 夹范围" ✓ —— 只加默认值不加读回 = 改了没用 ✗，上一轮踩过 ✓）。
        for _wk, _ws in self._width_spins.items():
            vis_cfg[_wk] = int(_ws.value())
        vis_cfg["player_arrow_len"] = int(self.sp_arrow_len.value())
        # ⭐ 箭头**尖大小**（用户 2026-09-28 ✓"再加个箭头 size 配置"）✓ 与长度同一段存 ✓
        vis_cfg["player_arrow_tip_pct"] = int(self.sp_arrow_tip.value())
        # 每项的**显示开关**（用户 2026-09-27："辅助线与标记组里每项参数前加开关"）：
        # 和颜色存在同一段（`config/ui.yaml` 的 `vis:` ✓），`theme.load_vis` 一并读回 ✓
        vis_cfg.update({k: bool(ck.isChecked()) for k, ck in self._vis_on.items()})
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
