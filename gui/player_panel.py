"""决策参数面板：自动打怪的开关、攻击距离、键盘映射。

决策逻辑在 decision/agent.py，实时推理循环驱动它。这里只负责：
    · 开关自动（按钮 + F11）
    · 触控板 / 鼠标左右键 / 灵敏度 / 输入设备（「操控」组，本地 F10 开关触控模式）
    · 攻击距离
    · 键盘映射（7 个键）

F10 / F11 / F12 是**本机操作键**：只在本窗口起作用，永不转发给游戏
（见 decision/input.py 的 LOCAL_ONLY）。

所有改动直接写进 settings 单例（decision.agent.settings），实时线程每帧读到
最新值，无需重启。**这份参数按项目各存一份**：主窗口打开项目时注册保存钩子，
每次 save() 就整份写回该项目的 project.yaml（见 MainWindow._bind_decision_params）。
"""

from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QFormLayout, QFrame, QGridLayout, QGroupBox,
                             QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                             QMessageBox, QProgressBar, QPushButton, QScrollArea,
                             QSlider, QVBoxLayout, QWidget)

from decision import input as dinput
from decision.agent import ANTI_AFK_TYPES, load_rect, load_saved, settings
# NoWheel* 必须模块级导入：控件在 _build() 里建，懒导入到不了那儿。
# （详见 docs/UI规范.md：滚轮不许改参数）
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox,
                         NoWheelSlider, NoWheelSpinBox, forward_wheel)
from gui.touchpad import TouchPad

# 键盘映射的每一行：(key, 标签, 默认键名)
# 注意："auto" 就是「开启自动」按钮的快捷键，默认 F11，和按钮/全局热键是同一个东西。
KEY_ROWS = [
    ("auto", "开关自动", "f11"),
    ("left", "移动←", "left"),
    ("right", "移动→", "right"),
    ("up", "移动↑", "up"),
    ("down", "移动↓", "down"),
    ("attack", "输出", "ctrl"),
    ("jump", "跳跃", "alt"),
    ("hp_pot", "补血", None),
    ("mp_pot", "补蓝", None),
    ("feed_pet", "喂宠", None),
    ("shop", "商城", None),
    ("enter", "回车", "enter"),
]

# Qt.Key → input.py 键名（按键监听时用）
_QT_TO_NAME = {
    Qt.Key_Left: "left", Qt.Key_Right: "right", Qt.Key_Up: "up", Qt.Key_Down: "down",
    Qt.Key_Control: "ctrl", Qt.Key_Alt: "alt", Qt.Key_Shift: "shift",
    Qt.Key_Space: "space", Qt.Key_Return: "enter", Qt.Key_Enter: "enter",
    Qt.Key_Escape: "esc", Qt.Key_Tab: "tab",
    Qt.Key_Delete: "del", Qt.Key_Insert: "insert", Qt.Key_Home: "home",
    Qt.Key_End: "end", Qt.Key_PageUp: "pageup", Qt.Key_PageDown: "pagedown",
    Qt.Key_QuoteLeft: "grave", Qt.Key_AsciiTilde: "grave",
}
for _i in range(1, 13):
    _QT_TO_NAME[getattr(Qt, "Key_F%d" % _i)] = "f%d" % _i
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _QT_TO_NAME[getattr(Qt, "Key_" + _c)] = _c.lower()
for _i in range(10):
    _QT_TO_NAME[getattr(Qt, "Key_%d" % _i)] = str(_i)


def _sample_bar_color(img):
    """从一张 BGR 图里采样血条「填充色」，返回 [[B,G,R],[B,G,R]]；失败返回 None。

    关键：血条上通常有白色数字文本（如 "224/366"）和灰色背景——按亮度采样
    会采到文字而不是填充色（实际踩过）。所以**优先取有彩色的像素**（HSV
    饱和度 > 40，排除白/灰），几乎没有彩色像素才回退到亮度 top 30%。
    """
    import cv2
    import numpy as np

    if img is None or img.size == 0:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    # 1) 优先：有彩色的像素（红/蓝填充），排除白字和灰背景
    colorful = s > 40
    if np.count_nonzero(colorful) >= colorful.size * 0.05:
        mean = img[colorful].mean(axis=0)
    else:
        # 2) 回退：几乎没有彩色像素，取亮度 top 30%
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = gray.flatten()
        n = max(1, int(len(flat) * 0.3))
        idx = np.argsort(flat)[-n:]
        mean = img.reshape(-1, 3)[idx].mean(axis=0)

    tol = 60
    low = tuple(np.clip(mean - tol, 0, 255).astype(np.uint8).tolist())
    high = tuple(np.clip(mean + tol, 0, 255).astype(np.uint8).tolist())
    return [list(low), list(high)]


class PlayerPanel(QWidget):
    verify_started = pyqtSignal()            # 保留：主窗口既有连接
    verify_result = pyqtSignal(object, str)  # 保留：主窗口既有连接
    auto_key_changed = pyqtSignal(object)    # 「开关自动」映射变了（键名或 None）→ 重新注册全局热键
    device_connected = pyqtSignal(str)       # Pro Micro 连接结果："ok" / "fail"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.project = None
        self._build()
        self.device_connected.connect(self._on_device_connected)
        self._sync_from_settings()

    # ---------------- 界面 ----------------

    def _build(self):
        # 版面分两段：上面「操控」组常驻不滚动，下面 QScrollArea 装其余参数。
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 0)
        outer.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 「操控」组压在滚动内容上方，边框要和滚动内容左右对齐：
        # 滚动内容会被右边的滚动条往左挤 sb_w 像素，所以组这边也要补同样的
        # 右边距（含内容自己的 pad），否则两个 groupbox 的边框一宽一窄。
        pad = 10
        sb_w = scroll.verticalScrollBar().sizeHint().width()

        # ---- 置顶常驻区（放在滚动区**外面**，不参与滚动）----
        # 上半：开启自动 / 开启手动输入 —— 最常用的两个开关，滚没了最难受。
        # 下半：「操控」组（触控板 / 左右键 / 灵敏度 / 输入设备+状态 / 重置指令通道）。
        # 「以「操控」组的下边界为基准在本页面置顶」= 这一整块的下边界就是滚动区的
        # 上边界：开关和这一组始终可见、可交互 —— 卡键、掉设备这类事往往正是滚到
        # 一半才发现的，那时候够不着就很别扭。
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.btn_auto = QPushButton("开启自动（F11）")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setMinimumHeight(34)
        self.btn_auto.setToolTip("开始/停止自动打怪。也可用 F11 快捷键开关。")
        self.btn_auto.clicked.connect(self._toggle_auto)
        ctrl_row.addWidget(self.btn_auto)

        self.btn_manual = QPushButton("开启手动输入")
        self.btn_manual.setCheckable(True)
        self.btn_manual.setMinimumHeight(34)
        self.btn_manual.setToolTip(
            "开启后，本机键盘的按键会实时转发到游戏（手动插手）。\n"
            "**不会**关掉自动，可以和自动同时开着。\n"
            "两边抢同一个键时：你松开的那个键固件也就松了，但决策层会在下一帧\n"
            "重新把它按回去（每次手动按键都会让决策层重同步一次）。\n"
            "\n"
            "触控板用**本地按 F10**开 / 关（不是按住 Ctrl —— Ctrl 是默认攻击键，\n"
            "按住它用触控板会让角色打出去）。F10 / F11 / F12 是本机操作键，\n"
            "永不转发给游戏。")
        self.btn_manual.clicked.connect(self._toggle_manual)
        ctrl_row.addWidget(self.btn_manual)

        ctrl_grp = QGroupBox("操控")
        cg = QHBoxLayout(ctrl_grp)
        cg.setSpacing(10)

        # 左：触控板（TouchPad 固定 210x132）
        self.touchpad = TouchPad()
        cg.addWidget(self.touchpad)

        # 触控板位移的余数累积（小位移不丢：0.3px 攒到 1px 才发）
        self._pad_rem = [0.0, 0.0]
        self.touchpad.moved.connect(self._on_pad_moved)
        # 触控模式（本地按 F10 开 / 关，见 eventFilter）下的点击 / 滚轮也转发给远程鼠标
        self.touchpad.clicked.connect(dinput.mouse_click)
        self.touchpad.scrolled.connect(dinput.mouse_scroll)

        # 右：一列，全部压进触控板的纵向范围（4 行 ≈ 122px ≤ 触控板 132px）。
        # 本地按 F10 开触控模式 → 在板子上滑动，位移按灵敏度映射发给远程鼠标。
        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        # （1）左右键：按住 = 按下不放（支持拖拽），松开 = 弹起。只对 ProMicro 生效。
        self.btn_mouse_l = QPushButton("左键")
        self.btn_mouse_r = QPushButton("右键")
        for b in (self.btn_mouse_l, self.btn_mouse_r):
            b.setMinimumHeight(28)
            b.setToolTip("按住 = 鼠标按下不放（可拖拽），松开 = 弹起")
        self.btn_mouse_l.pressed.connect(lambda: dinput.mouse_press("left"))
        self.btn_mouse_l.released.connect(lambda: dinput.mouse_release("left"))
        self.btn_mouse_r.pressed.connect(lambda: dinput.mouse_press("right"))
        self.btn_mouse_r.released.connect(lambda: dinput.mouse_release("right"))
        mbtn_row = QHBoxLayout()
        mbtn_row.setSpacing(6)
        mbtn_row.addWidget(self.btn_mouse_l)
        mbtn_row.addWidget(self.btn_mouse_r)
        mbtn_row.addStretch(1)
        right_col.addLayout(mbtn_row)

        # （2）灵敏度：本地鼠标位移 × 该比例 = 远程鼠标位移
        self.sp_mouse_speed = self._spin(0.1, 10.0, 1.0, 2)
        self.sp_mouse_speed.setSingleStep(0.1)
        self.sp_mouse_speed.setToolTip(
            "灵敏度：本地鼠标位移 × 该比例 = 远程鼠标位移（1.0 = 1:1）。")
        self.sp_mouse_speed.valueChanged.connect(self._on_mouse_speed)
        spd_row = QHBoxLayout()
        spd_row.setSpacing(6)
        spd_row.addWidget(QLabel("灵敏度"))
        spd_row.addWidget(self.sp_mouse_speed)
        spd_row.addStretch(1)
        right_col.addLayout(spd_row)

        # （3）输入设备 + 重置指令通道：同一行 —— 一个是「走哪条通道」，
        #      一个是「通道卡住时的逃生口」，放一起才好找。
        self.cmb_device = NoWheelComboBox()
        self.cmb_device.addItem("本地(仅测试)", "local")
        self.cmb_device.addItem("ProMicro(远程)", "remote")
        self.cmb_device.addItem("ProMicro(本地)", "serial")
        self.cmb_device.setToolTip(
            "本地(仅测试)：本机键盘模拟（SendInput）；\n"
            "ProMicro(远程)：网络连游戏机的 Pro Micro 硬件键盘；\n"
            "ProMicro(本地)：本机 USB 直连 Pro Micro，无需 relay。")
        self.cmb_device.currentIndexChanged.connect(self._on_input_device)

        # 重置指令通道：卡键 / 按键发不出去时的逃生口。
        # 自动关着也能用 —— 卡键往往正是「发现卡了去关自动」之后才发现的。
        self.btn_reset_link = QPushButton("重置指令通道")
        self.btn_reset_link.setToolTip(
            "按键卡住（角色自己一直走 / 一直攻击）或按键发不出去时点这里。\n"
            "① 重连远程通道 —— 连接断开时 relay 会直接往固件写一条 RELEASEALL，\n"
            "   把卡住的键一次松开（这条路不依赖我们的网络还通不通）；\n"
            "② 再补发一轮 RELEASEALL + 所有映射键的 RELEASE；\n"
            "③ 让决策层重同步按键状态（否则它以为键还按着，之后不再补发）。")
        self.btn_reset_link.clicked.connect(self._reset_link)
        dev_row = QHBoxLayout()
        dev_row.setSpacing(6)
        dev_row.addWidget(QLabel("输入设备"))
        dev_row.addWidget(self.cmb_device)
        dev_row.addWidget(self.btn_reset_link)
        dev_row.addStretch(1)
        right_col.addLayout(dev_row)

        # （4）设备状态：这几个字最长（「通道仍不通：…（已停止发指令…）」），
        #      单独一行并允许换行，免得把这一行撑宽、连带触控板那一行的宽度。
        self.lbl_device_state = QLabel("本地键盘")
        self.lbl_device_state.setStyleSheet("color: #80868b;")
        self.lbl_device_state.setWordWrap(True)
        right_col.addWidget(self.lbl_device_state)

        right_col.addStretch(1)
        cg.addLayout(right_col, 1)

        # 常驻块外面套一层：纵向两行（开关行 + 操控组），右侧补出
        # 「滚动条 + 内容边距」，让边框和滚动内容对齐
        top_bar = QWidget()
        tb = QVBoxLayout(top_bar)
        tb.setContentsMargins(0, 0, sb_w + pad, 0)
        tb.setSpacing(6)
        tb.addLayout(ctrl_row)
        tb.addWidget(ctrl_grp)
        # 这一整块在滚动区外面，往上找不到滚动区 —— 指个路，指针停在这里时
        # 滚轮仍然滚下面的参数（详见 gui/widgets.py 的 forward_wheel）。
        top_bar._wheel_target = scroll
        # 块里大多是「不吃滚轮」的普通控件（按钮、空白、状态文字）：它们不处理
        # 滚轮，事件会一路冒到 top_bar —— 在这里接住，滚轮才真的哪儿都能滚
        # （点完「开启自动」手不用挪就能继续滚页面）。
        self._top_bar = top_bar
        top_bar.installEventFilter(self)

        outer.addWidget(top_bar)
        outer.addWidget(scroll, 1)      # 剩余高度全给滚动区

        content = QWidget()
        scroll.setWidget(content)

        root = QVBoxLayout(content)
        # 左右边距交给外层（outer 的 10px），这样「操控」组的左边框和滚动内容
        # 完全对齐；右边留 pad，与组那边补的 sb_w + pad 对上。
        root.setContentsMargins(0, 8, pad, pad)
        root.setSpacing(8)

        title = QLabel("决策参数")
        title.setStyleSheet("font-weight: 600; font-size: 14px; color: #202124;")
        root.addWidget(title)

        # 开启自动 / 开启手动输入已挪到滚动区外面（见上面常驻块）
        self.lbl_state = QLabel("当前：未开启")
        self.lbl_state.setStyleSheet("color: #80868b;")
        root.addWidget(self.lbl_state)

        # ---- 其余参数（输入设备 / 状态 / 重置指令通道已挪进上面的「操控」组）----
        top_form = QFormLayout()
        top_form.setLabelAlignment(Qt.AlignLeft)

        # 随机输入延迟 [min, max]（毫秒）：点按类按键之间的随机间隔
        self.sp_delay_min = self._spin(0, 1000, 70, 0)
        self.sp_delay_max = self._spin(0, 1000, 130, 0)
        self.sp_delay_min.setToolTip("按键之间的随机间隔（毫秒），模拟真人手速，越小越快。")
        self.sp_delay_max.setToolTip("按键之间的随机间隔（毫秒），模拟真人手速，越小越快。")
        self.sp_delay_min.valueChanged.connect(self._on_input_delay)
        self.sp_delay_max.valueChanged.connect(self._on_input_delay)
        delay_row = QHBoxLayout()
        delay_row.addWidget(self.sp_delay_min)
        delay_row.addWidget(QLabel("~"))
        delay_row.addWidget(self.sp_delay_max)
        delay_row.addWidget(QLabel("ms"))
        top_form.addRow("随机输入延迟", delay_row)

        # 输出后锁定防抖（毫秒）：这段时间内 attack 目标保持锁定，不切换到范围内其他框
        self.sp_attack_lock_db = self._spin(0, 10000, 300, 0)
        self.sp_attack_lock_db.setToolTip("攻击出手后，这段时间内不换目标，避免来回切换。")
        self.sp_attack_lock_db.valueChanged.connect(self._on_attack_lock_db)
        top_form.addRow("输出后锁定防抖(ms)", self.sp_attack_lock_db)

        # 玩家追踪阈值（像素）：新玩家框离预测位置超此距离就不认（防跟错）。
        self.sp_track_jump = self._spin(0, 2000, 150, 0)
        self.sp_track_jump.setToolTip(
            "玩家追踪阈值（像素）：新检出的玩家框离「预测位置」超过这个距离就不认，\n"
            "防止跟错到别的玩家身上。\n"
            "调大更宽容（跟丢少，但可能被别的玩家抢锁）；调小更严格。\n"
            "帧率掉帧时一帧里位移变大，可能需要放宽。")
        self.sp_track_jump.valueChanged.connect(self._on_track_jump)
        top_form.addRow("玩家追踪阈值", self.sp_track_jump)

        root.addLayout(top_form)

        # ---- 参数模板（轻量一行，不套 groupbox：只有两个按钮，边框标题纯占地方）----
        tpl_row = QHBoxLayout()
        tpl_row.setSpacing(8)
        btn_save_tpl = QPushButton("保存模板")
        btn_save_tpl.setToolTip("把当前所有决策参数存成模板。")
        btn_save_tpl.clicked.connect(self._save_template)
        btn_load_tpl = QPushButton("加载模板")
        btn_load_tpl.setToolTip("从模板载入决策参数。")
        btn_load_tpl.clicked.connect(self._load_template)
        tpl_row.addWidget(btn_save_tpl)
        tpl_row.addWidget(btn_load_tpl)
        tpl_row.addStretch(1)
        root.addLayout(tpl_row)

        # ---- 视野组（2 列网格：6 项 → 3 行，省一半垂直空间）----
        vision_grp = QGroupBox("视野")
        vg = QVBoxLayout(vision_grp)
        vg.setSpacing(6)

        self.ck_vision_center = QCheckBox("基于画面中心")
        self.ck_vision_center.setToolTip("勾选：以画面屏幕中心为视野中心；\n不勾：以角色位置为中心。")
        self.ck_vision_center.stateChanged.connect(self._on_vision)
        vg.addWidget(self.ck_vision_center)

        self.sp_vision_top = self._spin(-1, 2000, 200, 0)
        self.sp_vision_bottom = self._spin(-1, 2000, 200, 0)
        self.sp_vision_left = self._spin(-1, 2000, 200, 0)
        self.sp_vision_right = self._spin(-1, 2000, 200, 0)
        for _sp in (self.sp_vision_top, self.sp_vision_bottom,
                    self.sp_vision_left, self.sp_vision_right):
            _sp.setToolTip("该方向的视野范围（像素）。设为 -1 表示该方向不限制。")
            _sp.valueChanged.connect(self._on_vision)

        self.sp_vision_off_x = self._spin(-2000, 2000, 0, 0)
        self.sp_vision_off_y = self._spin(-2000, 2000, 0, 0)
        self.sp_vision_off_x.setToolTip("视野框基于中心在 x 方向的偏移（像素）。")
        self.sp_vision_off_y.setToolTip("视野框基于中心在 y 方向的偏移（像素）。")
        self.sp_vision_off_x.valueChanged.connect(self._on_vision)
        self.sp_vision_off_y.valueChanged.connect(self._on_vision)

        vgrid = QGridLayout()
        vgrid.setHorizontalSpacing(10)
        vgrid.setVerticalSpacing(6)

        def _vcell(text, widget):
            """一行「标签 + 输入框」，标签定宽对齐。返回 (layout, label)。

            宽度按最长的动态标签「向前(左)视野」定，否则切换中心/角色模式时
            文字会被截断。"""
            h = QHBoxLayout()
            h.setSpacing(6)
            lbl = QLabel(text)
            lbl.setFixedWidth(82)
            h.addWidget(lbl)
            h.addWidget(widget, 1)
            return h, lbl

        cell, _ = _vcell("向上视野", self.sp_vision_top)
        vgrid.addLayout(cell, 0, 0)
        cell, _ = _vcell("向下视野", self.sp_vision_bottom)
        vgrid.addLayout(cell, 0, 1)
        cell, self._lbl_vision_left = _vcell("向左视野", self.sp_vision_left)
        vgrid.addLayout(cell, 1, 0)
        cell, self._lbl_vision_right = _vcell("向右视野", self.sp_vision_right)
        vgrid.addLayout(cell, 1, 1)
        cell, _ = _vcell("x偏移", self.sp_vision_off_x)
        vgrid.addLayout(cell, 2, 0)
        cell, _ = _vcell("y偏移", self.sp_vision_off_y)
        vgrid.addLayout(cell, 2, 1)
        vg.addLayout(vgrid)

        root.addWidget(vision_grp)

        # ---- 战斗参数组 ----
        battle = QGroupBox("战斗参数")
        bf = QFormLayout(battle)
        bf.setLabelAlignment(Qt.AlignLeft)

        # 输出行为 CD（毫秒）：两次输出之间的最小间隔（从上次输出时刻起算）
        self.sp_attack_cd = self._spin(0, 10000, 0, 0)
        self.sp_attack_cd.setToolTip(
            "两次输出之间的最小间隔（毫秒），从上次输出那一刻起算，越小打得越快。\n"
            "实际间隔 = 本值 + 随机延迟；输出行为序列本身更长时以序列为准。\n"
            "（主循环一帧只推进一步序列，帧率低时序列耗时会按帧向上取整）")
        self.sp_attack_cd.valueChanged.connect(self._on_attack_cd)
        bf.addRow("输出行为CD(ms)", self.sp_attack_cd)

        self.sp_attack = self._spin(0, 2000, 80, 0)
        self.sp_attack.setToolTip(
            "怪离角色多近才开始攻击（只算前方）。\n"
            "距离从角色中心点量到怪框最近的边，不是从玩家框的边缘起算。")
        self.sp_attack.valueChanged.connect(self._on_attack_dist)
        bf.addRow("最大攻击距离", self.sp_attack)

        # 最小切换朝向时间（毫秒）：换向后方向键至少按住这么久
        self.sp_min_turn_hold = self._spin(0, 3000, 0, 0)
        self.sp_min_turn_hold.setToolTip(
            "换朝向时，方向键至少要按住这么久（毫秒）。0 = 不约束。\n"
            "按下去立刻就松开的话，角色的转身动作可能还没做完 —— 这时候打出去\n"
            "的方向是错的。这段按住时间只能被「又换一次朝向」打断。")
        self.sp_min_turn_hold.valueChanged.connect(self._on_turn_params)
        bf.addRow("最小切换朝向时间(ms)", self.sp_min_turn_hold)

        # 转向后输出延迟（毫秒）：换向后推迟这么久才开始输出
        self.sp_turn_output_delay = self._spin(0, 3000, 0, 0)
        self.sp_turn_output_delay.setToolTip(
            "换朝向后，输出行为要额外推迟这么久（毫秒）才开始。0 = 不延迟。\n"
            "和上面那条配合用：那条保证方向键按住够久，这条保证输出等转身做完。\n"
            "已经在跑的输出序列不会被掐断（半截掐断会留下按着的键），\n"
            "只把「新开一轮输出」往后推。")
        self.sp_turn_output_delay.valueChanged.connect(self._on_turn_params)
        bf.addRow("转向后输出延迟(ms)", self.sp_turn_output_delay)

        self.sp_min_attack = self._spin(0, 2000, 0, 0)
        self.sp_min_attack.setToolTip(
            "怪贴脸到此距离内就触发规避（跳/后退）。设 0 表示不规避。\n"
            "口径同最大攻击距离：角色中心点 → 怪框最近的边。")
        self.sp_min_attack.valueChanged.connect(self._on_min_attack)
        bf.addRow("最小攻击距离", self.sp_min_attack)

        self.ck_chase_jump = QCheckBox("启用")
        self.ck_chase_jump.setToolTip(
            "追击起跳：攻击范围内没有其他怪时，锁定目标落在起跳区间内就按跳。\n"
            "范围内还有其他怪则不跳 —— 先打那些，跳会打断输出。\n"
            "区间 = [最大攻击距离 + min, 最大攻击距离 + max]，\n"
            "min/max 都是相对最大攻击距离的偏移：\n"
            "  0 ~ 50  → 纯外侧（还差一点够不着时跳）\n"
            "  -30 ~ 0 → 纯内侧（已进范围但偏远时跳）\n"
            "  -30 ~ 50 → 跨攻击距离两侧")
        self.ck_chase_jump.stateChanged.connect(self._on_chase_jump)
        self.sp_chase_jump_min = self._spin(-2000, 2000, 0, 0)
        self.sp_chase_jump_min.setToolTip("起跳区间下限：相对最大攻击距离的偏移（可为负）。")
        self.sp_chase_jump_min.valueChanged.connect(self._on_chase_jump)
        self.sp_chase_jump_max = self._spin(-2000, 2000, 50, 0)
        self.sp_chase_jump_max.setToolTip("起跳区间上限：相对最大攻击距离的偏移。")
        self.sp_chase_jump_max.valueChanged.connect(self._on_chase_jump)
        cj_row = QHBoxLayout()
        cj_row.setSpacing(4)
        cj_row.addWidget(self.ck_chase_jump)
        cj_row.addWidget(self.sp_chase_jump_min)
        cj_row.addWidget(QLabel("~"))
        cj_row.addWidget(self.sp_chase_jump_max)
        cj_row.addStretch(1)
        bf.addRow("追击起跳", cj_row)

        # 规避策略（仅最小攻击距离 > 0 时显示）
        evade_grp = QGroupBox("规避策略")
        ef = QFormLayout(evade_grp)
        ef.setLabelAlignment(Qt.AlignLeft)
        self.cmb_evade = NoWheelComboBox()
        self.cmb_evade.addItem("跳", "jump")
        self.cmb_evade.addItem("后退", "back")
        self.cmb_evade.setToolTip("怪贴脸时的应对方式：跳起来打 / 往后退。")
        self.cmb_evade.currentIndexChanged.connect(self._on_evade_type)
        ef.addRow("规避类型", self.cmb_evade)
        self.sp_jump_interval = self._spin(0, 10000, 200, 0)
        self.sp_jump_interval.setToolTip("跳起来后隔多久再攻击，太短会打断起跳。")
        self.sp_jump_interval.valueChanged.connect(self._on_jump_interval)
        ef.addRow("跳与输出延迟(ms)", self.sp_jump_interval)
        self._lbl_jump_interval = ef.labelForField(self.sp_jump_interval)

        self.sp_jump_random = self._spin(0.0, 1.0, 0.1, 2)
        self.sp_jump_random.setSingleStep(0.05)
        self.sp_jump_random.setToolTip("正常攻击时随机跳+输出的概率，让动作更像真人。")
        self.sp_jump_random.valueChanged.connect(self._on_jump_random)
        ef.addRow("乱跳几率", self.sp_jump_random)
        self._lbl_jump_random = ef.labelForField(self.sp_jump_random)

        self.btn_edit_seq = QPushButton("调整回身输出")
        self.btn_edit_seq.setToolTip("打开行为编辑器，配置反向跳回身的动作序列")
        self.btn_edit_seq.setStyleSheet(self._BTN_EDIT_SEQ)
        self.btn_edit_seq.clicked.connect(self._edit_back_jump_seq)
        ef.addRow("", self.btn_edit_seq)
        self._lbl_edit_seq = ef.labelForField(self.btn_edit_seq)

        bf.addRow("", evade_grp)
        self._evade_grp = evade_grp

        # 目标切换 CD [min, max]（毫秒）
        self.sp_cd_min = self._spin(0, 10000, 500, 0)
        self.sp_cd_max = self._spin(0, 10000, 1000, 0)
        self.sp_cd_min.setToolTip("锁定目标后，至少/至多隔多久才允许换目标（随机取中间值）。")
        self.sp_cd_max.setToolTip("锁定目标后，至少/至多隔多久才允许换目标（随机取中间值）。")
        self.sp_cd_min.valueChanged.connect(self._on_target_cd)
        self.sp_cd_max.valueChanged.connect(self._on_target_cd)
        cd_row = QHBoxLayout()
        cd_row.addWidget(self.sp_cd_min)
        cd_row.addWidget(QLabel("~"))
        cd_row.addWidget(self.sp_cd_max)
        cd_row.addWidget(QLabel("ms"))
        bf.addRow("目标切换CD", cd_row)

        # 防抖：高置信度怪框消失后保留位置（独立于策略）
        self.sp_debounce_conf = self._spin(0.0, 1.0, 0.5, 2)
        self.sp_debounce_conf.setSingleStep(0.05)
        self.sp_debounce_conf.setToolTip("置信度高于此值的怪，框短暂消失会保留位置，防漏检。")
        self.sp_debounce_conf.valueChanged.connect(self._on_debounce)
        bf.addRow("防抖置信度", self.sp_debounce_conf)

        self.sp_debounce_ms = self._spin(0, 10000, 300, 0)
        self.sp_debounce_ms.setToolTip("怪框消失后保留位置的时长。")
        self.sp_debounce_ms.valueChanged.connect(self._on_debounce)
        bf.addRow("防抖时间(ms)", self.sp_debounce_ms)

        # 定义输出行为：呼出行为编辑器，配置 attack 状态执行的输出序列（默认一个输出键）
        self.btn_edit_output = QPushButton("定义输出行为")
        self.btn_edit_output.setToolTip("打开行为编辑器，配置输出动作序列（默认一个输出键）")
        self.btn_edit_output.setStyleSheet(self._BTN_EDIT_SEQ)
        self.btn_edit_output.clicked.connect(self._edit_output_seq)
        bf.addRow("", self.btn_edit_output)

        root.addWidget(battle)

        # ---- 自动喝药组 ----
        pot = QGroupBox("自动喝药")
        pg = QFormLayout(pot)
        pg.setLabelAlignment(Qt.AlignLeft)

        self.btn_hp_bar = QPushButton("框选 HP 条")
        self.btn_hp_bar.setToolTip("在游戏画面上框选血条，用于识别当前血量。")
        self.btn_hp_bar.clicked.connect(lambda: self._pick_bar("hp"))
        self.lbl_hp_bar = QLabel("未选")
        self.lbl_hp_bar.setStyleSheet("color: #80868b;")
        hp_row = QHBoxLayout()
        hp_row.addWidget(self.btn_hp_bar)
        hp_row.addWidget(self.lbl_hp_bar, 1)
        pg.addRow("HP条", hp_row)

        self.btn_mp_bar = QPushButton("框选 MP 条")
        self.btn_mp_bar.setToolTip("在游戏画面上框选蓝条，用于识别当前蓝量。")
        self.btn_mp_bar.clicked.connect(lambda: self._pick_bar("mp"))
        self.lbl_mp_bar = QLabel("未选")
        self.lbl_mp_bar.setStyleSheet("color: #80868b;")
        mp_row = QHBoxLayout()
        mp_row.addWidget(self.btn_mp_bar)
        mp_row.addWidget(self.lbl_mp_bar, 1)
        pg.addRow("MP条", mp_row)

        self.ck_auto_hp = QCheckBox("自动补血")
        self.ck_auto_hp.setToolTip("勾选后血量低于阈值自动喝药。")
        self.ck_auto_hp.stateChanged.connect(self._on_auto_hp)
        self.sl_hp = NoWheelSlider(Qt.Horizontal)
        self.sl_hp.setRange(0, 100)
        self.sl_hp.setValue(30)
        self.sl_hp.setToolTip("血量低于此百分比就喝药。")
        self.sl_hp.valueChanged.connect(self._on_hp_threshold)
        self.lbl_hp_th = QLabel("30%")
        self.lbl_hp_th.setFixedWidth(40)
        hp_th = QHBoxLayout()
        hp_th.addWidget(self.ck_auto_hp)
        hp_th.addWidget(self.sl_hp, 1)
        hp_th.addWidget(self.lbl_hp_th)
        pg.addRow("HP阈值", hp_th)

        self.ck_auto_mp = QCheckBox("自动补蓝")
        self.ck_auto_mp.setToolTip("勾选后蓝量低于阈值自动喝药。")
        self.ck_auto_mp.stateChanged.connect(self._on_auto_mp)
        self.sl_mp = NoWheelSlider(Qt.Horizontal)
        self.sl_mp.setRange(0, 100)
        self.sl_mp.setValue(20)
        self.sl_mp.setToolTip("蓝量低于此百分比就喝药。")
        self.sl_mp.valueChanged.connect(self._on_mp_threshold)
        self.lbl_mp_th = QLabel("20%")
        self.lbl_mp_th.setFixedWidth(40)
        mp_th = QHBoxLayout()
        mp_th.addWidget(self.ck_auto_mp)
        mp_th.addWidget(self.sl_mp, 1)
        mp_th.addWidget(self.lbl_mp_th)
        pg.addRow("MP阈值", mp_th)

        # 喝药冷却（ms）
        self.sp_pot_cd = self._spin(0, 10000, 1000, 0)
        self.sp_pot_cd.setToolTip("喝一次药后隔多久才允许再喝，防连喝。")
        self.sp_pot_cd.valueChanged.connect(self._on_pot_cd)
        pg.addRow("喝药冷却", self.sp_pot_cd)

        # 实时识别出的血/蓝比例（初始隐藏，框选后才显示）
        self.pb_hp = QProgressBar()
        self.pb_hp.setRange(0, 100)
        self.pb_hp.setValue(0)
        self.pb_hp.setFormat("HP %p%")
        pg.addRow("HP当前", self.pb_hp)
        self._lbl_hp_cur = pg.labelForField(self.pb_hp)

        self.pb_mp = QProgressBar()
        self.pb_mp.setRange(0, 100)
        self.pb_mp.setValue(0)
        self.pb_mp.setFormat("MP %p%")
        pg.addRow("MP当前", self.pb_mp)
        self._lbl_mp_cur = pg.labelForField(self.pb_mp)

        # 自动喂宠
        feed_row = QHBoxLayout()
        self.ck_feed = QCheckBox("自动喂宠")
        self.ck_feed.setToolTip("定时自动喂宠物。")
        self.ck_feed.stateChanged.connect(self._on_auto_feed)
        self.lbl_feed_cd = QLabel("")
        self.lbl_feed_cd.setStyleSheet("color:#80868b;")
        feed_row.addWidget(self.ck_feed)
        feed_row.addWidget(self.lbl_feed_cd)
        feed_row.addStretch(1)
        pg.addRow("", feed_row)

        # 喂宠间隔：随机区间 [下限, 上限]（分钟）
        self.sp_feed_interval_min = self._spin(0.1, 600, 5, 1, 0.1)
        self.sp_feed_interval_max = self._spin(0.1, 600, 10, 1, 0.1)
        self.sp_feed_interval_min.setToolTip("每隔随机 N~M 分钟喂一次宠物（支持小数，如 1.5）。")
        self.sp_feed_interval_max.setToolTip("每隔随机 N~M 分钟喂一次宠物（支持小数，如 1.5）。")
        self.sp_feed_interval_min.valueChanged.connect(self._on_feed_interval)
        self.sp_feed_interval_max.valueChanged.connect(self._on_feed_interval)
        feed_iv_row = QHBoxLayout()
        feed_iv_row.addWidget(self.sp_feed_interval_min)
        feed_iv_row.addWidget(QLabel("~"))
        feed_iv_row.addWidget(self.sp_feed_interval_max)
        pg.addRow("喂宠间隔(min)", feed_iv_row)

        self._pot_form = pg

        root.addWidget(pot)

        # ---- 策略参数组 ----
        strategy_grp = QGroupBox("策略参数")
        sf = QFormLayout(strategy_grp)
        sf.setLabelAlignment(Qt.AlignLeft)

        # 策略类型
        self.cmb_strategy = NoWheelComboBox()
        self.cmb_strategy.addItem("平地巡逻", "patrol")
        self.cmb_strategy.addItem("扫平台", "sweep")
        self.cmb_strategy.currentIndexChanged.connect(self._on_strategy)
        self.cmb_strategy.setToolTip(
            "平地巡逻：锁定全部怪，就近优先。\n"
            "扫平台：优先朝向方向，背后一定距离内的怪按就近锁定；\n"
            "当前朝向没怪持续一段时间就换向。")
        sf.addRow("策略类型", self.cmb_strategy)

        self.lbl_strategy_note = QLabel("")
        self.lbl_strategy_note.setStyleSheet("color: #80868b;")
        self.lbl_strategy_note.setWordWrap(True)
        sf.addRow("", self.lbl_strategy_note)

        # 换朝向延迟（仅「扫平台」策略用）：当前朝向没怪持续此时间才换
        self.sp_turn_cd = self._spin(0, 10000, 1000, 0)
        self.sp_turn_cd.valueChanged.connect(self._on_turn_cd)
        self.sp_turn_cd.setToolTip(
            "当前朝向没怪持续此时间（毫秒）才换朝向。\n"
            "防止某帧漏检/抖动导致频繁换向。")
        sf.addRow("换朝向延迟(ms)", self.sp_turn_cd)
        self._lbl_turn_cd = sf.labelForField(self.sp_turn_cd)

        # 背后锁定距离（仅「扫平台」策略用）
        self.sp_back_range = self._spin(0, 2000, 100, 0)
        self.sp_back_range.valueChanged.connect(self._on_back_range)
        self.sp_back_range.setToolTip(
            "允许锁定背后此距离（像素）内的怪。\n"
            "扫平台优先朝向方向，但背后很近的怪也会被就近锁定。")
        sf.addRow("背后锁定距离", self.sp_back_range)
        self._lbl_back_range = sf.labelForField(self.sp_back_range)

        root.addWidget(strategy_grp)

        # ---- 键盘映射组 ----
        grp = QGroupBox("键盘映射")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        self._key_buttons = {}
        self._listening_key = None
        self._listening_custom = None   # 正在监听的自定义按键名
        COLS = 2      # 每行排 2 组，横向铺开省空间
        for i, (key, label, default) in enumerate(KEY_ROWS):
            btn = QPushButton()
            btn.setFixedWidth(100)
            btn.clicked.connect(lambda _c, k=key: self._start_listen(k))
            self._key_buttons[key] = btn

            rm = QPushButton("移除")
            rm.setFixedWidth(42)
            rm.setStyleSheet("padding: 2px 6px;")
            rm.setToolTip("清空这个键的映射")
            rm.clicked.connect(lambda _c, k=key: self._remove_key(k))

            cell = QHBoxLayout()
            cell.setSpacing(6)
            lbl = QLabel(label)
            lbl.setFixedWidth(52)
            cell.addWidget(lbl)
            cell.addWidget(btn)
            cell.addWidget(rm)
            cell.addStretch(1)

            r, c = divmod(i, COLS)
            grid.addLayout(cell, r, c)

        root.addWidget(grp)

        # ---- 自定义按键组 ----
        custom_grp = QGroupBox("自定义按键")
        cv = QVBoxLayout(custom_grp)
        self._custom_list = QVBoxLayout()
        self._custom_list.setSpacing(4)
        cv.addLayout(self._custom_list)
        btn_add_custom = QPushButton("＋ 添加自定义按键")
        btn_add_custom.setToolTip("新增一个自定义按键，可在行为序列里使用。")
        btn_add_custom.clicked.connect(self._add_custom_key)
        cv.addWidget(btn_add_custom)
        root.addWidget(custom_grp)

        # ---- 自定义定时行为组 ----
        timer_grp = QGroupBox("自定义定时行为")
        tv = QVBoxLayout(timer_grp)
        tv.setSpacing(6)
        self._timer_cd_labels = {}       # {name: 倒计时 QLabel}
        self._timer_list = QVBoxLayout()
        self._timer_list.setSpacing(4)
        tv.addLayout(self._timer_list)
        btn_add_timer = QPushButton("＋ 添加行为")
        btn_add_timer.setToolTip("新增一个定时执行的按键行为（名字 + 序列 + 间隔）。")
        btn_add_timer.clicked.connect(self._add_timer)
        tv.addWidget(btn_add_timer)
        root.addWidget(timer_grp)

        # ---- 防掉线组 ----
        afk = QGroupBox("防掉线")
        af = QFormLayout(afk)

        # 手动进入/结束休息：立刻休息一次、或休息中提前拉回来。
        # 左右顺序 = 一次休息的时间顺序（先进后出）。两个按钮都只看实时线程写的
        # 状态（_poll_auto_state 轮询刷新），自己不做判断。
        # 右边紧跟休息状态/倒计时，同一行放省垂直空间（实测三个控件合计约 500px，
        # 面板 612px，放得下）。
        self.btn_start_rest = QPushButton("手动进入休息")
        self.btn_start_rest.setToolTip(
            "立刻按「隐身休息」流程休息一次：\n"
            "  进入隐身行为 → 歇完（防掉线里设的休息时长）→ 退出隐身行为 → 继续打怪，\n"
            "  并按防掉线的间隔重新排下一次自动休息。\n"
            "和自动防掉线走**同一条流程**：攻击范围内还有怪时会先等它们清空\n"
            "（状态显示「待休息：等清空攻击范围内的怪」），免得正打着怪突然站住。\n"
            "自动关着、防掉线没开、或已经在休息时不可点。")
        self.btn_start_rest.setEnabled(False)
        self.btn_start_rest.clicked.connect(self._on_start_rest)

        self.btn_end_rest = QPushButton("手动结束休息")
        self.btn_end_rest.setToolTip(
            "休息中点了立刻结束休息：执行「退出隐身行为」，然后恢复正常打怪。\n"
            "没在休息时不可点。")
        self.btn_end_rest.setEnabled(False)
        self.btn_end_rest.clicked.connect(self._on_end_rest)

        self.lbl_rest = QLabel("")
        self.lbl_rest.setStyleSheet("color: #80868b;")
        rest_row = QHBoxLayout()
        rest_row.setSpacing(8)
        rest_row.addWidget(self.btn_start_rest)
        rest_row.addWidget(self.btn_end_rest)
        rest_row.addWidget(self.lbl_rest, 1)
        af.addRow("", rest_row)

        self.ck_afk = QCheckBox("自动防掉线")
        self.ck_afk.setToolTip("定时执行防掉线行为（目前：隐身休息）。")
        self.ck_afk.stateChanged.connect(self._on_anti_afk)
        af.addRow("", self.ck_afk)

        self.sp_afk_min = self._spin(0.1, 600, 5, 1, 0.1)
        self.sp_afk_max = self._spin(0.1, 600, 10, 1, 0.1)
        self.sp_afk_min.setToolTip("每隔随机 N~M 分钟触发一次防掉线（支持小数，如 1.5）。")
        self.sp_afk_max.setToolTip("每隔随机 N~M 分钟触发一次防掉线（支持小数，如 1.5）。")
        self.sp_afk_min.valueChanged.connect(self._on_afk_time)
        self.sp_afk_max.valueChanged.connect(self._on_afk_time)
        afk_iv_row = QHBoxLayout()
        afk_iv_row.addWidget(self.sp_afk_min)
        afk_iv_row.addWidget(QLabel("~"))
        afk_iv_row.addWidget(self.sp_afk_max)
        af.addRow("触发时间(min)", afk_iv_row)

        # 行为类型：不同类型展开不同的参数子组（目前只有「隐身休息」）
        self.cmb_afk_type = NoWheelComboBox()
        for tid, tname in ANTI_AFK_TYPES:
            self.cmb_afk_type.addItem(tname, tid)
        self.cmb_afk_type.setToolTip("防掉线触发时做什么。")
        self.cmb_afk_type.currentIndexChanged.connect(self._on_afk_type)
        af.addRow("行为类型", self.cmb_afk_type)

        # 隐身休息子组：触发时间到 → 等攻击范围内的怪清空 → 执行进入隐身 →
        # 休息随机时长 → 执行退出隐身 → 继续打怪
        self._afk_hidden = QGroupBox("隐身休息")
        hf = QFormLayout(self._afk_hidden)
        hf.setLabelAlignment(Qt.AlignLeft)

        self.btn_afk_enter = QPushButton("编辑进入隐身行为")
        self.btn_afk_enter.setToolTip("进隐身时执行的动作序列（例如按隐身术快捷键）。")
        self.btn_afk_enter.setStyleSheet(self._BTN_EDIT_SEQ)
        self.btn_afk_enter.clicked.connect(lambda: self._edit_afk_seq("enter"))
        hf.addRow("进入隐身", self.btn_afk_enter)

        self.btn_afk_exit = QPushButton("编辑退出隐身行为")
        self.btn_afk_exit.setToolTip("退出隐身时执行的动作序列。")
        self.btn_afk_exit.setStyleSheet(self._BTN_EDIT_SEQ)
        self.btn_afk_exit.clicked.connect(lambda: self._edit_afk_seq("exit"))
        hf.addRow("退出隐身", self.btn_afk_exit)

        self.sp_rest_min = self._spin(0.1, 600, 10, 1, 0.1)
        self.sp_rest_max = self._spin(0.1, 600, 20, 1, 0.1)
        self.sp_rest_min.setToolTip("隐身休息随机 N~M 分钟（支持小数，如 7.5）。")
        self.sp_rest_max.setToolTip("隐身休息随机 N~M 分钟（支持小数，如 7.5）。")
        self.sp_rest_min.valueChanged.connect(self._on_afk_rest_time)
        self.sp_rest_max.valueChanged.connect(self._on_afk_rest_time)
        rest_iv_row = QHBoxLayout()
        rest_iv_row.addWidget(self.sp_rest_min)
        rest_iv_row.addWidget(QLabel("~"))
        rest_iv_row.addWidget(self.sp_rest_max)
        hf.addRow("休息时长(min)", rest_iv_row)

        af.addRow(self._afk_hidden)

        root.addWidget(afk)
        root.addStretch(1)

        # 喂宠倒计时：每秒刷新一次「距离下次」
        self._feed_timer = QTimer(self)
        self._feed_timer.setInterval(1000)
        self._feed_timer.timeout.connect(self._tick_feed_cd)

        # 自动开关状态轮询：agent 后台可能因朝向超时等把 enabled 关掉，定时同步 UI
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(500)
        self._state_timer.timeout.connect(self._poll_auto_state)
        self._state_timer.start()

        # F11 开关自动（窗口内快捷键；全局热键后续接 RegisterHotKey）
        self.setFocusPolicy(Qt.StrongFocus)

        # 手动输入开启时，屏蔽键盘对 UI 参数界面的影响（全局事件过滤器）
        QApplication.instance().installEventFilter(self)

    # ---------------- 控件构造 ----------------

    @staticmethod
    def _spin(lo, hi, val, decimals, step=None):
        w = NoWheelDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        if step is not None:      # 小数位下默认步进 1.0 太粗（5.0 直接跳 6.0）
            w.setSingleStep(step)
        w.setValue(val)
        return w

    # ---------------- 事件 ----------------

    def _toggle_auto(self, checked=None):
        on = self.btn_auto.isChecked()
        settings.enabled = on
        if not on:
            self._force_release_all()   # 关闭自动立即释放所有按键，防卡键
        self._refresh_auto_ui()

    def _toggle_manual(self, checked=None):
        from decision import manual_input
        on = self.btn_manual.isChecked()
        if on:
            # 注意：**不动自动**。手动输入可以和自动同时开着（抢同一个键的问题
            # 由 manual_input 的「按键状态重同步」兜着，见那个模块的文档）。
            try:
                manual_input.start()
            except Exception as e:
                self.btn_manual.setChecked(False)
                QMessageBox.warning(self, "手动输入启动失败", str(e))
                return
        else:
            manual_input.stop()
        self._refresh_manual_ui()

    def _refresh_manual_ui(self):
        from decision import manual_input
        on = manual_input.active()
        self.btn_manual.setChecked(on)
        self.btn_manual.setText("停止手动输入" if on else "开启手动输入")
        self.btn_manual.setStyleSheet(self._BTN_MANUAL_ON if on else self._BTN_MANUAL_OFF)

    # ---------------- 鼠标控制（摇杆 + 左右键） ----------------

    def _on_pad_moved(self, dx, dy):
        """触控板原始位移 → 乘灵敏度 + 余数累积 → 发远程鼠标移动。"""
        spd = max(0.01, float(settings.mouse_speed))
        self._pad_rem[0] += dx * spd
        self._pad_rem[1] += dy * spd
        mx = int(self._pad_rem[0])
        my = int(self._pad_rem[1])
        self._pad_rem[0] -= mx
        self._pad_rem[1] -= my
        if mx or my:
            dinput.mouse_move(mx, my)

    def _on_mouse_speed(self, _val=None):
        settings.mouse_speed = float(self.sp_mouse_speed.value())
        settings.save()

    def _refresh_mouse_ui(self):
        """鼠标控制只对 ProMicro 生效；本地模式禁用（本机鼠标要操作界面）。"""
        ok = dinput.mouse_available()
        for w in (self.btn_mouse_l, self.btn_mouse_r, self.sp_mouse_speed):
            w.setEnabled(ok)
        self.touchpad.set_capture_enabled(ok)

    def _reset_link(self):
        """手动重置指令通道：卡键 / 发不出指令时的逃生口（见按钮 tooltip）。"""
        from decision import input as dinput
        h = dinput.link_health()
        backend = h.get("backend")
        reconnected = False
        if backend in ("remote", "serial"):
            if not h.get("ok", True):
                self.lbl_device_state.setText("通道异常（%s），正在重连…" % (h.get("err") or "? "))
            else:
                self.lbl_device_state.setText("正在重连通道…")
            self.lbl_device_state.setStyleSheet("color: #b06000;")
            QApplication.processEvents()          # 先把上面这句画出来，重连会阻塞一下
            # 无条件重连：链路「看起来正常」但实际已经冻住的情况（relay 卡在串口
            # 写上）只有断开这一下能解，断开时 relay 会给固件发 RELEASEALL。
            reconnected = dinput.reconnect_remote()
        self._force_release_all()
        settings.input_resync = True              # 让决策层也清掉本地按键状态
        h2 = dinput.link_health()
        if h2.get("ok", True):
            self.lbl_device_state.setText("已重连通道并释放按键" if reconnected
                                          else "已释放所有按键")
            self.lbl_device_state.setStyleSheet("color: #137333;")
        else:
            self.lbl_device_state.setText(
                "通道仍不通：%s（已停止发指令，不会再乱按键）"
                % (h2.get("err") or "原因未知"))
            self.lbl_device_state.setStyleSheet("color: #c5221f;")

    def _force_release_all(self):
        """关闭自动时，立即对所有映射键发 key_up（RELEASE），不等 agent 下一帧。

        背景：按键通过 Pro Micro 固件保持，若 RELEASE 命令丢失（网络瞬断）或
        实时线程卡住没走到 release_all，键会卡在按下状态。这里兜底强制释放所有
        映射键；对未按下的键发 RELEASE 无害（固件/SendInput 都忽略）。
        """
        from decision import input as dinput
        # 远程模式：发一条 RELEASEALL 让固件清空所有按键，比逐键 RELEASE 可靠，
        # 能救回「RELEASE 丢失导致卡键」的情况
        dinput.release_all_remote()
        # 逐键兜底（本地 SendInput + 远程补充）
        keys = set(settings.keymap.values()) | set(settings.custom_keys.values())
        for key in keys:
            if key:
                try:
                    dinput.key_up(key)
                except Exception:
                    pass

    _BTN_AUTO_OFF = ("QPushButton { background:#1a73e8; color:#ffffff; border:none;"
                     " border-radius:5px; font-weight:600; }"
                     "QPushButton:hover { background:#4285f4; }")
    _BTN_AUTO_ON = ("QPushButton { background:#ea4335; color:#ffffff; border:none;"
                    " border-radius:5px; font-weight:600; }"
                    "QPushButton:hover { background:#f25a4b; }")
    # 手动输入按钮：关闭=绿色（可开启），开启=红色（正在转发）
    _BTN_MANUAL_OFF = ("QPushButton { background:#188038; color:#ffffff; border:none;"
                       " border-radius:5px; font-weight:600; }"
                       "QPushButton:hover { background:#1e9e4a; }")
    _BTN_MANUAL_ON = ("QPushButton { background:#ea4335; color:#ffffff; border:none;"
                      " border-radius:5px; font-weight:600; }"
                      "QPushButton:hover { background:#f25a4b; }")
    # 行为编辑器按钮的统一配色（所有「呼出行为编辑器」的按钮都用这个）
    _BTN_EDIT_SEQ = ("QPushButton { background:#f3e8fd; color:#7627bb;"
                     " border:1px solid #d7aefb; border-radius:5px; font-weight:600; }"
                     "QPushButton:hover { background:#e9d5fd; }")

    def _refresh_auto_ui(self):
        on = settings.enabled
        hotkey = dinput.display_name(settings.keymap.get("auto", "f11"))
        self.btn_auto.setChecked(on)
        self.btn_auto.setText("停止自动（%s）" % hotkey if on else "开启自动（%s）" % hotkey)
        self.btn_auto.setStyleSheet(self._BTN_AUTO_ON if on else self._BTN_AUTO_OFF)
        self.lbl_state.setText("当前：自动打怪运行中" if on else "当前：未开启")

    def _poll_auto_state(self):
        """定时把 settings.enabled 同步到开关 UI（agent 后台停自动时也能反映）。

        顺带刷新休息状态：rest_state 由实时线程里的 agent 写，只能轮询。
        """
        # 兜底：面板被藏起来时不该还占着鼠标。hideEvent 管切页签那条路，
        # 这里再兜一层（父级被整体隐藏之类不会走我们的 hideEvent）。
        if self.touchpad.active and not self.isVisible():
            self.touchpad.set_active(False)
        if self.btn_auto.isChecked() != settings.enabled:
            self._refresh_auto_ui()
        resting = bool(settings.rest_state)
        self.btn_end_rest.setEnabled(resting)
        # 「手动进入休息」：自动开着 + 防掉线开着 + 没在休息 + 没在等清怪。
        # 已经在「待休息」时再点没意义（本来就是等清怪），禁掉更清楚。
        self.btn_start_rest.setEnabled(
            bool(settings.enabled) and bool(settings.anti_afk_enabled)
            and not resting and not settings.rest_pending)
        self.lbl_rest.setText(self._rest_text())
        # 休息状态优先判断：能进休息就说明自动必然是开着的，反过来写会出现
        # 「休息中」却显示「未开启」的自相矛盾。
        if resting:
            txt = "当前：隐身休息中"
        elif settings.enabled:
            txt = "当前：自动打怪运行中"
        else:
            txt = "当前：未开启"
        from decision import manual_input
        if manual_input.active():
            txt += "（手动输入中）"     # 现在两者可以同时开着，要说清楚
        if self.touchpad.active:
            txt += "（触控模式中）"     # 板子会变蓝，但滚到别处时就看不见了
        self.lbl_state.setText(txt)

    @staticmethod
    def _rest_text():
        """休息状态文字：当前阶段 / 休息剩余时间 / 下次休息倒计时。

        数据都由实时线程里的 agent 写（rest_state / rest_until_monotonic /
        next_afk_monotonic / rest_pending），这里只读不算；用 time.monotonic
        算剩余是因为两侧同进程、同一时钟源。
        """
        import time

        def left(until):
            """「　剩余 M:SS」后缀；没有计时基准（<=0）时返回空串。"""
            if until <= 0:
                return ""
            sec = max(0, int(until - time.monotonic()))
            return "　剩余 %d:%02d" % (sec // 60, sec % 60)

        st = settings.rest_state
        # 休息时长从「开始进入隐身」起算，所以这个阶段剩余时间也已经在走了
        if st == "afk_enter":
            return "进入隐身…" + left(settings.rest_until_monotonic)
        if st == "afk_rest":
            return "休息中" + left(settings.rest_until_monotonic)
        if st == "afk_exit":
            return "退出隐身…"
        # 到点了但攻击范围内还有怪：卡在这一步时最容易被误认为「坏了」
        if settings.rest_pending:
            return "待休息：等清空攻击范围内的怪"
        nxt = settings.next_afk_monotonic
        if nxt > 0:
            return "下次休息" + left(nxt)
        return "未休息"

    def _on_start_rest(self):
        """手动进入休息：置一次性请求，agent 下一帧标记「待休息」。

        这里**不直接改** rest_pending —— 那是 agent 的运行时状态，界面只读不写；
        而且请求要能被「停自动 / 关掉防掉线」干净地丢掉，那由 agent 负责。
        """
        settings.rest_request = True
        self.lbl_rest.setText("已请求休息：等清空攻击范围内的怪…")

    def _on_end_rest(self):
        """手动结束休息：置一次性请求，agent 下一帧转去执行「退出隐身行为」再恢复打怪。"""
        settings.rest_abort = True

    def _on_attack_dist(self, val):
        settings.attack_dist = float(val)
        settings.save()

    def _on_min_attack(self, val):
        settings.min_attack_dist = int(val)
        settings.save()
        self._refresh_evade_ui()

    def _on_chase_jump(self, _val=None):
        settings.chase_jump_enabled = bool(self.ck_chase_jump.isChecked())
        settings.chase_jump_min = int(self.sp_chase_jump_min.value())
        settings.chase_jump_max = int(self.sp_chase_jump_max.value())
        settings.save()
        self._refresh_chase_jump_ui()

    def _refresh_chase_jump_ui(self):
        """开关关掉时把两个距离框灰掉。"""
        on = self.ck_chase_jump.isChecked()
        self.sp_chase_jump_min.setEnabled(on)
        self.sp_chase_jump_max.setEnabled(on)

    def _on_evade_type(self, _idx=None):
        settings.evade_type = self.cmb_evade.currentData()
        settings.save()
        self._refresh_evade_ui()

    def _on_jump_interval(self, val):
        settings.jump_interval = int(val)
        settings.save()

    def _on_jump_random(self, val):
        settings.jump_random_prob = float(val)
        settings.save()

    def _edit_back_jump_seq(self):
        """打开回身输出行为编辑器。"""
        from gui.seq_editor import SeqEditorDialog
        dlg = SeqEditorDialog(settings.back_jump_seq, self)
        if dlg.exec_():
            settings.back_jump_seq = dlg.seq()
            settings.save()

    def _edit_output_seq(self):
        """打开输出行为编辑器。"""
        from gui.seq_editor import SeqEditorDialog
        dlg = SeqEditorDialog(settings.output_seq, self, title="输出行为编辑器")
        if dlg.exec_():
            settings.output_seq = dlg.seq()
            settings.save()

    def _on_anti_afk(self, state):
        settings.anti_afk_enabled = bool(state)
        settings.save()

    def _on_afk_time(self, _val=None):
        settings.anti_afk_min = float(self.sp_afk_min.value())
        settings.anti_afk_max = float(self.sp_afk_max.value())
        if settings.anti_afk_max < settings.anti_afk_min:
            settings.anti_afk_max = settings.anti_afk_min
        settings.save()

    def _on_afk_type(self, _idx=None):
        settings.anti_afk_type = self.cmb_afk_type.currentData() or ANTI_AFK_TYPES[0][0]
        settings.save()
        self._refresh_afk_ui()

    def _on_afk_rest_time(self, _val=None):
        settings.anti_afk_rest_min = float(self.sp_rest_min.value())
        settings.anti_afk_rest_max = float(self.sp_rest_max.value())
        if settings.anti_afk_rest_max < settings.anti_afk_rest_min:
            settings.anti_afk_rest_max = settings.anti_afk_rest_min
        settings.save()

    def _refresh_afk_ui(self):
        """按行为类型显示对应的参数子组。"""
        self._afk_hidden.setVisible(settings.anti_afk_type == "hidden_rest")

    def _edit_afk_seq(self, which):
        """打开「进入隐身」/「退出隐身」的行为编辑器。"""
        from gui.seq_editor import SeqEditorDialog
        attr = "anti_afk_enter_seq" if which == "enter" else "anti_afk_exit_seq"
        title = ("进入隐身" if which == "enter" else "退出隐身") + "行为编辑器"
        dlg = SeqEditorDialog(getattr(settings, attr) or [], self, title=title)
        if dlg.exec_():
            setattr(settings, attr, dlg.seq())
            settings.save()

    def _refresh_evade_ui(self):
        """规避策略子组：最小攻击距离 > 0 才显示；跳间隔/乱跳几率仅「跳」类型显示。"""
        show = settings.min_attack_dist > 0
        self._evade_grp.setVisible(show)
        if show:
            is_jump = settings.evade_type == "jump"
            self.sp_jump_interval.setVisible(is_jump)
            if getattr(self, "_lbl_jump_interval", None) is not None:
                self._lbl_jump_interval.setVisible(is_jump)
            self.sp_jump_random.setVisible(is_jump)
            if getattr(self, "_lbl_jump_random", None) is not None:
                self._lbl_jump_random.setVisible(is_jump)
            self.btn_edit_seq.setVisible(is_jump)
            if getattr(self, "_lbl_edit_seq", None) is not None:
                self._lbl_edit_seq.setVisible(is_jump)

    def _on_target_cd(self, _val=None):
        settings.target_cd = [int(self.sp_cd_min.value()), int(self.sp_cd_max.value())]
        settings.save()

    def _on_input_delay(self, _val=None):
        settings.input_delay = [int(self.sp_delay_min.value()), int(self.sp_delay_max.value())]
        settings.save()

    def _on_attack_cd(self, val):
        settings.attack_cd = int(val)
        settings.save()

    def _on_attack_lock_db(self, val):
        settings.attack_lock_debounce_ms = int(val)
        settings.save()

    def _on_track_jump(self, val):
        settings.player_track_jump = int(val)
        settings.save()

    def _on_turn_params(self, _val=None):
        """换向相关的两个时间参数一起写（同一组，一个处理器够了）。"""
        settings.min_turn_hold_ms = int(self.sp_min_turn_hold.value())
        settings.turn_output_delay_ms = int(self.sp_turn_output_delay.value())
        settings.save()

    def _on_hp_threshold(self, val):
        settings.hp_threshold = int(val)
        self.lbl_hp_th.setText("%d%%" % val)
        settings.save()

    def _on_mp_threshold(self, val):
        settings.mp_threshold = int(val)
        self.lbl_mp_th.setText("%d%%" % val)
        settings.save()

    def _on_pot_cd(self, val):
        settings.pot_cd = int(val)
        settings.save()

    def _on_strategy(self, _idx=None):
        settings.strategy = self.cmb_strategy.currentData()
        settings.save()
        self._refresh_strategy_ui()

    def _on_turn_cd(self, val):
        settings.sweep_turn_cd = int(val)
        settings.save()

    def _on_back_range(self, val):
        settings.back_range = int(val)
        settings.save()

    def _on_debounce(self, _val=None):
        settings.debounce_conf = float(self.sp_debounce_conf.value())
        settings.debounce_ms = int(self.sp_debounce_ms.value())
        settings.save()

    def _refresh_strategy_ui(self):
        """按策略切换说明文字，并控制扫平台专属参数是否可见。"""
        is_sweep = settings.strategy == "sweep"
        if is_sweep:
            self.lbl_strategy_note.setText(
                "扫平台：优先朝向方向，背后一定距离内的怪按就近锁定；"
                "当前朝向没怪持续一段时间就换向。继承上/下阈值过滤。")
        else:
            self.lbl_strategy_note.setText(
                "以角色脚底为基准：上阈值往上、下阈值往下，范围外的怪不追踪")
        self.sp_turn_cd.setVisible(is_sweep)
        if getattr(self, "_lbl_turn_cd", None) is not None:
            self._lbl_turn_cd.setVisible(is_sweep)
        self.sp_back_range.setVisible(is_sweep)
        if getattr(self, "_lbl_back_range", None) is not None:
            self._lbl_back_range.setVisible(is_sweep)

    def _on_vision(self, _val=None):
        settings.vision_top = int(self.sp_vision_top.value())
        settings.vision_bottom = int(self.sp_vision_bottom.value())
        settings.vision_left = int(self.sp_vision_left.value())
        settings.vision_right = int(self.sp_vision_right.value())
        settings.vision_center = bool(self.ck_vision_center.isChecked())
        settings.vision_off_x = int(self.sp_vision_off_x.value())
        settings.vision_off_y = int(self.sp_vision_off_y.value())
        settings.save()
        self._refresh_vision_labels()

    def _refresh_vision_labels(self):
        """左右视野标签随「基于画面中心」切换。

        基于角色（不勾）：左右是「朝向相对」的前后，标成「向前(左)/向后(右)」；
        基于画面中心（勾）：左右是画面绝对方向，标成「向左/向右」。
        """
        center = bool(self.ck_vision_center.isChecked())
        left = "向左视野" if center else "向前(左)视野"
        right = "向右视野" if center else "向后(右)视野"
        if getattr(self, "_lbl_vision_left", None) is not None:
            self._lbl_vision_left.setText(left)
        if getattr(self, "_lbl_vision_right", None) is not None:
            self._lbl_vision_right.setText(right)

    def _on_auto_hp(self, state):
        settings.auto_hp_pot = bool(state)
        settings.save()

    def _on_auto_mp(self, state):
        settings.auto_mp_pot = bool(state)
        settings.save()

    def _on_auto_feed(self, state):
        settings.auto_feed_pet = bool(state)
        settings.save()
        # 开关切换都清计时：打开后等一个间隔再喂（不立即吃）、关掉不留残值
        settings.feed_next_monotonic = 0.0
        if settings.auto_feed_pet:
            self._feed_timer.start()
            self._tick_feed_cd()
        else:
            self._feed_timer.stop()
            self.lbl_feed_cd.setText("")

    def _tick_feed_cd(self):
        """每秒刷新喂宠 + 自定义定时行为的倒计时（读 agent 维护的下次触发时刻）。"""
        import time
        now = time.monotonic()
        next_ts = settings.feed_next_monotonic
        if next_ts > 0:
            remaining = next_ts - now
        else:
            remaining = settings.feed_interval_min * 60.0   # agent 还没跑过：按最短间隔（下限）占位
        if remaining < 0:
            remaining = 0
        m = int(remaining) // 60
        s = int(remaining) % 60
        self.lbl_feed_cd.setText("距离下次：%d:%02d" % (m, s))
        # 自定义定时行为倒计时
        for name, lbl in self._timer_cd_labels.items():
            next_ts = settings.custom_timer_next.get(name, 0.0)
            if next_ts > 0:
                remaining = next_ts - now
            else:
                t = next((x for x in settings.custom_timers if x.get("name") == name), None)
                lo = (t.get("interval", [5, 10])[0] if t else 5) * 60.0
                remaining = lo
            if remaining < 0:
                remaining = 0
            m = int(remaining) // 60
            s = int(remaining) % 60
            lbl.setText("距离下次：%d:%02d" % (m, s))

    # ---------------- 自定义定时行为 ----------------

    def _refresh_timers(self):
        """重建自定义定时行为列表。"""
        self._clear_layout(self._timer_list)
        self._timer_cd_labels.clear()
        for t in settings.custom_timers:
            name = t.get("name", "")
            lo, hi = t.get("interval", [5, 10])
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl_name = QLabel(name)
            lbl_name.setFixedWidth(90)
            # %g：整数不显示小数点（5 而不是 5.0），小数原样显示（7.5）
            lbl_iv = QLabel("%g~%gmin" % (lo, hi))
            lbl_iv.setFixedWidth(80)
            lbl_cd = QLabel("")
            lbl_cd.setStyleSheet("color:#80868b;")
            row.addWidget(lbl_name)
            row.addWidget(lbl_iv)
            row.addWidget(lbl_cd, 1)
            btn_edit = QPushButton("编辑")
            btn_edit.setFixedWidth(52)
            btn_edit.setToolTip("修改名字 / 时间区间 / 行为序列")
            btn_edit.clicked.connect(lambda _c, n=name: self._edit_timer(n))
            btn_del = QPushButton("删除")
            btn_del.setFixedWidth(52)
            btn_del.clicked.connect(lambda _c, n=name: self._del_timer(n))
            row.addWidget(btn_edit)
            row.addWidget(btn_del)
            self._timer_list.addLayout(row)
            self._timer_cd_labels[name] = lbl_cd
        self._tick_feed_cd()

    def _add_timer(self):
        """添加一个自定义定时行为：一个弹窗搞定名字 + 时间区间 + 行为序列。"""
        from gui.seq_editor import TimerEditDialog
        dlg = TimerEditDialog(None, self)
        if not dlg.exec_():
            return
        r = dlg.result()
        if any(t.get("name") == r["name"] for t in settings.custom_timers):
            QMessageBox.warning(self, "添加失败", "行为名已存在")
            return
        settings.custom_timers.append(r)
        settings.custom_timer_next.pop(r["name"], None)   # 重新计时
        settings.save()
        self._refresh_timers()

    def _edit_timer(self, name):
        """编辑一个自定义定时行为：一个弹窗搞定名字 + 时间区间 + 行为序列。"""
        t = next((x for x in settings.custom_timers if x.get("name") == name), None)
        if t is None:
            return
        from gui.seq_editor import TimerEditDialog
        dlg = TimerEditDialog(t, self)
        if not dlg.exec_():
            return
        r = dlg.result()
        if r["name"] != name and any(x.get("name") == r["name"] for x in settings.custom_timers):
            QMessageBox.warning(self, "编辑失败", "行为名已存在")
            return
        t["name"] = r["name"]
        t["interval"] = r["interval"]
        t["seq"] = r["seq"]
        # 编辑后重新计时：无论名字变没变，都按新间隔重新随机
        settings.custom_timer_next.pop(name, None)
        settings.custom_timer_next.pop(r["name"], None)
        settings.save()
        self._refresh_timers()

    def _del_timer(self, name):
        """删除一个自定义定时行为。"""
        settings.custom_timers = [t for t in settings.custom_timers if t.get("name") != name]
        settings.custom_timer_next.pop(name, None)
        settings.save()
        self._refresh_timers()

    def _on_feed_interval(self, _val=None):
        settings.feed_interval_min = float(self.sp_feed_interval_min.value())
        settings.feed_interval_max = float(self.sp_feed_interval_max.value())
        if settings.feed_interval_max < settings.feed_interval_min:
            settings.feed_interval_max = settings.feed_interval_min
        settings.save()

    def _pick_bar(self, kind):
        """在实时画面上框选 HP/MP 条区域，并采样该条的填充色。"""
        from gui.region_selector import select_region_on_image
        frame = None
        lp = getattr(self, "live_panel", None)
        if lp is not None:
            frame = lp.current_frame()
        if frame is None:
            QMessageBox.information(
                self, "提示", "请先在「实时」页开始预览、看到画面后，再框选 HP/MP 条")
            return
        rect = select_region_on_image(frame, self)
        if rect is None:
            return
        x, y, rw, rh = rect
        region = frame[y:y + rh, x:x + rw]
        color = _sample_bar_color(region)
        # 存「画面比例」（相对画面帧），收流/窗口分辨率变化自动适配
        h, w = frame.shape[:2]
        ratio = [round(x / w, 6), round(y / h, 6),
                 round(rw / w, 6), round(rh / h, 6)]
        if kind == "hp":
            settings.hp_bar = ratio
            settings.hp_color = color
        else:
            settings.mp_bar = ratio
            settings.mp_color = color
        settings.save()
        # 按项目保存：写进当前项目的 project.yaml。没打开项目时只落全局，
        # 下次开项目仍能从全局兜底读到。
        if self.project is not None:
            bars = dict(self.project.get("bars") or {})
            bars[kind] = ratio
            bars[kind + "_color"] = color
            self.project.set("bars", bars, save=True)
        self._sync_bar_ui()

    # ---------------- HP/MP 条（按项目保存） ----------------

    def bind(self, project):
        """切换项目：刷新控件显示 + 载入该项目的 HP/MP 条框选结果。

        **决策参数本身（整份）由主窗口在更早的时候换好**（见
        MainWindow._bind_decision_params，按项目存在 project.yaml 的 decision 段），
        这里只负责把换完之后的 settings 回填到界面上。

        HP/MP 条：项目存过就用项目的；没存过回退到 config/decision.json 里的全局值
        （= 最后一次框选 / 加载模板的位置）—— 同一套游戏 UI 通常通用，
        一律清空会逼着每个项目都重框一次。
        """
        self.project = project
        proj = (project.get("bars") if project is not None else None) or {}
        fallback = load_saved()      # 直接读文件，天然跟着最新一次 save()

        def pick(key, saved_key):
            return proj.get(key) or fallback.get(saved_key)

        settings.hp_bar = load_rect(pick("hp", "hp_bar"))
        settings.hp_color = pick("hp_color", "hp_color")
        settings.mp_bar = load_rect(pick("mp", "mp_bar"))
        settings.mp_color = pick("mp_color", "mp_color")
        self._sync_bar_ui()
        # 决策参数已经换成当前项目那份了，把控件重新回填一遍（切项目必走）
        self._sync_from_settings()

    def _sync_bar_ui(self):
        """按 settings 里的 HP/MP 条刷新标签文案、可见性和填充色。"""
        # 归一化：全零 / 零尺寸框当成「没框」—— [0,0,0,0] 是非空列表（真值），
        # 不处理会显示成一条空进度条。
        settings.hp_bar = load_rect(settings.hp_bar)
        settings.mp_bar = load_rect(settings.mp_bar)
        for rect, lbl in ((settings.hp_bar, self.lbl_hp_bar),
                          (settings.mp_bar, self.lbl_mp_bar)):
            if rect:
                nx, ny, nw, nh = rect
                lbl.setText("x%.0f%% y%.0f%%  %.0f%%×%.0f%%"
                            % (nx * 100, ny * 100, nw * 100, nh * 100))
            else:
                lbl.setText("未选")
        self._refresh_bar_visibility()
        self._apply_bar_chunk_color()

    def _refresh_bar_visibility(self):
        """未框选时不显示填充条（含 label），框选后才显示。"""
        show_hp = bool(settings.hp_bar)
        show_mp = bool(settings.mp_bar)
        self.pb_hp.setVisible(show_hp)
        self.pb_mp.setVisible(show_mp)
        if getattr(self, "_lbl_hp_cur", None) is not None:
            self._lbl_hp_cur.setVisible(show_hp)
            self._lbl_mp_cur.setVisible(show_mp)

    def _apply_bar_chunk_color(self):
        """进度条填充色按血条采样色显示（不再是系统默认绿）。"""
        for color, pb in ((settings.hp_color, self.pb_hp),
                          (settings.mp_color, self.pb_mp)):
            if color:
                b, g, r = color[1]      # 取范围高端（更亮的填充色）
                pb.setStyleSheet(
                    "QProgressBar::chunk { background: rgb(%d,%d,%d); }"
                    % (r, g, b))

    def _on_potions(self, hp, mp):
        """实时更新识别出的血/蓝比例。"""
        self.pb_hp.setValue(int(hp * 100))
        self.pb_mp.setValue(int(mp * 100))

    def _on_input_device(self, _idx=None):
        dev = self.cmb_device.currentData()
        settings.input_device = dev
        settings.save()
        self._apply_input_device(dev)

    def _apply_input_device(self, dev):
        """切换输入设备：本地 SendInput / 远程 Pro Micro / 本地 Pro Micro（后台异步连接）。"""
        from decision import input as dinput
        import threading
        if dev == "remote":
            from tools.config import get
            host = get("kbd", "host")
            port = int(get("kbd", "port", 9000))
            cert = get("kbd", "cert", "remote_kbd/certs/cert.pem")
            self.lbl_device_state.setText("正在连接 ProMicro(远程)…")
            def _do_connect():
                try:
                    dinput.use_network(host, port, cert)
                    self.device_connected.emit("ok")
                except Exception:
                    dinput.use_local()
                    self.device_connected.emit("fail")
            threading.Thread(target=_do_connect, daemon=True).start()
        elif dev == "serial":
            from tools.config import get
            ser_port = get("kbd", "serial_local", "COM5")
            self.lbl_device_state.setText("正在连接 ProMicro(本地)…")
            def _do_connect():
                try:
                    dinput.use_serial(ser_port)
                    self.device_connected.emit("ok")
                except Exception:
                    dinput.use_local()
                    self.device_connected.emit("fail")
            threading.Thread(target=_do_connect, daemon=True).start()
        else:
            dinput.use_local()
            self.lbl_device_state.setText("本地键盘")
        self._refresh_mouse_ui()

    def _on_device_connected(self, result):
        """Pro Micro 连接结果回调（后台线程 → 主线程）。"""
        if result == "ok":
            if settings.input_device == "serial":
                self.lbl_device_state.setText("已连接 ProMicro(本地)")
            else:
                self.lbl_device_state.setText("已连接 ProMicro(远程)")
        else:
            settings.input_device = "local"
            settings.save()
            self.lbl_device_state.setText("Pro Micro 连接失败，已回退本地")
            self.cmb_device.blockSignals(True)
            idx = self.cmb_device.findData("local")
            if idx >= 0:
                self.cmb_device.setCurrentIndex(idx)
            self.cmb_device.blockSignals(False)
        self._refresh_mouse_ui()

    # ---------------- 键盘映射（按键监听） ----------------

    def _refresh_key_buttons(self):
        """按 settings.keymap 刷新所有键按钮的文字。"""
        for key, btn in self._key_buttons.items():
            k = settings.keymap.get(key)
            btn.setText(dinput.display_name(k) if k else "无")

    def _start_listen(self, key):
        """进入监听：下一个按键作为新映射，Esc 取消。"""
        self._listening_key = key
        btn = self._key_buttons[key]
        btn.setText("请按键…")
        # 焦点给面板自身（而非按钮）：方向键才不会被 QScrollArea 吃掉
        self.setFocus()

    def _finish_listen(self, name=None):
        """结束监听并应用（name=None 表示取消）。"""
        key = self._listening_key
        self._listening_key = None
        if key is not None and name is not None:
            settings.set_key(key, name)
            settings.save()
            if key == "auto":
                self._refresh_auto_ui()
                self.auto_key_changed.emit(name)
        self._refresh_key_buttons()

    def _remove_key(self, key):
        """清空某个键的映射（设为 None = 不发键）。"""
        settings.set_key(key, None)
        settings.save()
        if key == "auto":
            self._refresh_auto_ui()
            self.auto_key_changed.emit(None)
        self._refresh_key_buttons()

    # ---------------- 自定义按键 ----------------

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            child = item.layout()
            if child is not None:
                PlayerPanel._clear_layout(child)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _next_custom_name(self):
        i = 1
        while ("custom%d" % i) in settings.custom_keys:
            i += 1
        return "custom%d" % i

    def _add_custom_key(self):
        name = self._next_custom_name()
        settings.custom_keys[name] = None
        settings.save()
        self._refresh_custom_keys()
        self._start_listen_custom(name)

    def _start_listen_custom(self, name):
        self._listening_custom = name
        self.setFocus()
        self._refresh_custom_keys()

    def _finish_listen_custom(self, physical):
        name = self._listening_custom
        self._listening_custom = None
        if name is not None and physical is not None:
            settings.custom_keys[name] = physical
            settings.save()
        self._refresh_custom_keys()

    def _remove_custom_key(self, name):
        settings.custom_keys.pop(name, None)
        settings.save()
        self._refresh_custom_keys()

    def _refresh_custom_keys(self):
        self._clear_layout(self._custom_list)
        for name in settings.custom_keys:
            row = QHBoxLayout()
            ed = QLineEdit(name)
            ed.setFixedWidth(100)
            ed.setToolTip("可编辑名称，改完回车生效；行为编辑器里会用到这个名字")
            ed.editingFinished.connect(
                lambda n=name, e=ed: self._rename_custom_key(n, e.text()))
            btn = QPushButton()
            btn.setFixedWidth(110)
            k = settings.custom_keys.get(name)
            if self._listening_custom == name:
                btn.setText("请按键…")
            else:
                btn.setText(dinput.display_name(k) if k else "无")
            btn.clicked.connect(lambda _c, n=name: self._start_listen_custom(n))
            rm = QPushButton("删除")
            rm.setFixedWidth(48)
            rm.clicked.connect(lambda _c, n=name: self._remove_custom_key(n))
            row.addWidget(ed)
            row.addWidget(btn)
            row.addWidget(rm)
            row.addStretch(1)
            self._custom_list.addLayout(row)

    def _rename_custom_key(self, old_name, new_name):
        """重命名自定义按键，并同步行为序列里的引用。"""
        new_name = (new_name or "").strip()
        if not new_name or new_name == old_name:
            self._refresh_custom_keys()   # 恢复原名
            return
        if new_name in settings.custom_keys:
            QMessageBox.warning(self, "重名", "名字「%s」已存在" % new_name)
            self._refresh_custom_keys()
            return
        physical = settings.custom_keys.pop(old_name)
        settings.custom_keys[new_name] = physical
        # 同步行为序列（回身输出/进入隐身/退出隐身）里对该键的引用
        for attr in ("back_jump_seq", "anti_afk_enter_seq", "anti_afk_exit_seq"):
            seq = getattr(settings, attr, None) or []
            for e in seq:
                if isinstance(e, dict) and e.get("key") == old_name:
                    e["key"] = new_name
            setattr(settings, attr, seq)
        settings.save()
        self._refresh_custom_keys()

    # ---------------- 参数模板 ----------------

    def _save_template(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存参数模板",
                                              "参数模板.json", "JSON (*.json)")
        if not path:
            return
        try:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(settings.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载参数模板", "", "JSON (*.json)")
        if not path:
            return
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            settings.from_dict(data)
            settings.save()
            self._sync_from_settings()
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))

    def eventFilter(self, obj, ev):
        """手动输入开启时，吃掉所有键盘事件，防止按键作用到 UI。

        方向键/空格/字母等本该转发到游戏，但也会改变 spinbox 数值、
        触发按钮、往输入框打字 —— 开启手动输入后一律屏蔽，让键盘只去游戏。
        「开关自动」的 F11 是系统级全局热键（WM_HOTKEY），不走这里，不受影响。
        """
        # 本地 F10：开 / 关触控板的触控模式。必须放在「手动输入吃掉键盘」**之前** ——
        # 两者经常同时开着，而 F10 恰恰是给手动操作配的。这里也顺手吃掉 F10 的
        # 按下和松开（不给界面、也不给游戏；F10 在本机操作键名单里，见 input.py）。
        if ev.type() in (QEvent.KeyPress, QEvent.KeyRelease) and ev.key() == Qt.Key_F10:
            if ev.type() == QEvent.KeyPress and not ev.isAutoRepeat():
                self.touchpad.toggle()
            return True

        # 触控模式用 grabMouse 独占鼠标：窗口失去焦点（切到别的程序 / 最小化）时
        # 自动关掉，否则回来时指针一沾到板子就继续发远程鼠标指令、点击变成游戏里的
        # 点击，看着像界面卡死。面板被藏起来（切页签）走的是 hideEvent。
        if ev.type() in (QEvent.ApplicationDeactivate, QEvent.WindowDeactivate):
            if self.touchpad.active:
                self.touchpad.set_active(False)
            return False

        from decision import manual_input
        if manual_input.active() and ev.type() in (QEvent.KeyPress, QEvent.KeyRelease):
            return True
        # 置顶常驻块（开启自动 + 「操控」组）在滚动区外面，块里不吃滚轮的控件
        # 会把事件冒到这里 —— 转给下面的滚动区，指针停在这一块上时页面照样滚。
        if ev.type() == QEvent.Wheel and obj is getattr(self, "_top_bar", None):
            if forward_wheel(ev, obj):
                ev.accept()
                return True
            return False
        return super().eventFilter(obj, ev)

    def hideEvent(self, ev):
        """面板被藏起来（切到别的页签）：关掉触控板的触控模式。

        触控模式用 grabMouse 独占鼠标，留着它切走的话：指针一沾到板子就继续发
        远程鼠标指令、点击变成游戏里的点击，看着像界面卡死 —— 而板子那点蓝色
        提示也在别的页签上，看不见。宁可你切回来重新按 F10。
        """
        if self.touchpad.active:
            self.touchpad.set_active(False)
        super().hideEvent(ev)

    def keyPressEvent(self, ev):
        """监听态下捕获按键；Esc 取消，其余键作为新映射。"""
        if self._listening_key is not None:
            name = None if ev.key() == Qt.Key_Escape else _QT_TO_NAME.get(ev.key())
            self._finish_listen(name)
        elif self._listening_custom is not None:
            name = None if ev.key() == Qt.Key_Escape else _QT_TO_NAME.get(ev.key())
            self._finish_listen_custom(name)
        else:
            super().keyPressEvent(ev)

    def _sync_from_settings(self):
        """启动时把 settings 里的值灌到控件上。"""
        self.sp_attack.blockSignals(True)
        self.sp_attack.setValue(settings.attack_dist)
        self.sp_attack.blockSignals(False)

        self.sp_min_attack.blockSignals(True)
        self.sp_min_attack.setValue(settings.min_attack_dist)
        self.sp_min_attack.blockSignals(False)

        self.ck_chase_jump.blockSignals(True)
        self.ck_chase_jump.setChecked(bool(settings.chase_jump_enabled))
        self.ck_chase_jump.blockSignals(False)
        for sp, v in ((self.sp_chase_jump_min, settings.chase_jump_min),
                      (self.sp_chase_jump_max, settings.chase_jump_max)):
            sp.blockSignals(True)
            sp.setValue(int(v))
            sp.blockSignals(False)
        self._refresh_chase_jump_ui()

        self.cmb_evade.blockSignals(True)
        _idx = self.cmb_evade.findData(settings.evade_type)
        if _idx >= 0:
            self.cmb_evade.setCurrentIndex(_idx)
        self.cmb_evade.blockSignals(False)

        self.sp_jump_interval.blockSignals(True)
        self.sp_jump_interval.setValue(settings.jump_interval)
        self.sp_jump_interval.blockSignals(False)

        self.sp_jump_random.blockSignals(True)
        self.sp_jump_random.setValue(settings.jump_random_prob)
        self.sp_jump_random.blockSignals(False)

        self._refresh_evade_ui()

        lo, hi = settings.target_cd
        self.sp_cd_min.blockSignals(True)
        self.sp_cd_min.setValue(lo)
        self.sp_cd_min.blockSignals(False)
        self.sp_cd_max.blockSignals(True)
        self.sp_cd_max.setValue(hi)
        self.sp_cd_max.blockSignals(False)

        self._refresh_key_buttons()

        lo, hi = settings.input_delay
        self.sp_delay_min.blockSignals(True)
        self.sp_delay_min.setValue(lo)
        self.sp_delay_min.blockSignals(False)
        self.sp_delay_max.blockSignals(True)
        self.sp_delay_max.setValue(hi)
        self.sp_delay_max.blockSignals(False)

        self.sp_attack_cd.blockSignals(True)
        self.sp_attack_cd.setValue(settings.attack_cd)
        self.sp_attack_cd.blockSignals(False)

        self.sp_attack_lock_db.blockSignals(True)
        self.sp_attack_lock_db.setValue(settings.attack_lock_debounce_ms)
        self.sp_attack_lock_db.blockSignals(False)

        self.sp_track_jump.blockSignals(True)
        self.sp_track_jump.setValue(settings.player_track_jump)
        self.sp_track_jump.blockSignals(False)

        self.sp_min_turn_hold.blockSignals(True)
        self.sp_min_turn_hold.setValue(settings.min_turn_hold_ms)
        self.sp_min_turn_hold.blockSignals(False)

        self.sp_turn_output_delay.blockSignals(True)
        self.sp_turn_output_delay.setValue(settings.turn_output_delay_ms)
        self.sp_turn_output_delay.blockSignals(False)

        self.sl_hp.blockSignals(True)
        self.sl_hp.setValue(settings.hp_threshold)
        self.sl_hp.blockSignals(False)
        self.lbl_hp_th.setText("%d%%" % settings.hp_threshold)

        self.sl_mp.blockSignals(True)
        self.sl_mp.setValue(settings.mp_threshold)
        self.sl_mp.blockSignals(False)
        self.lbl_mp_th.setText("%d%%" % settings.mp_threshold)

        self.sp_pot_cd.blockSignals(True)
        self.sp_pot_cd.setValue(settings.pot_cd)
        self.sp_pot_cd.blockSignals(False)

        self.cmb_strategy.blockSignals(True)
        idx = self.cmb_strategy.findData(settings.strategy)
        if idx >= 0:
            self.cmb_strategy.setCurrentIndex(idx)
        self.cmb_strategy.blockSignals(False)

        self.ck_vision_center.blockSignals(True)
        self.ck_vision_center.setChecked(settings.vision_center)
        self.ck_vision_center.blockSignals(False)

        self.sp_vision_top.blockSignals(True)
        self.sp_vision_top.setValue(settings.vision_top)
        self.sp_vision_top.blockSignals(False)
        self.sp_vision_bottom.blockSignals(True)
        self.sp_vision_bottom.setValue(settings.vision_bottom)
        self.sp_vision_bottom.blockSignals(False)
        self.sp_vision_left.blockSignals(True)
        self.sp_vision_left.setValue(settings.vision_left)
        self.sp_vision_left.blockSignals(False)
        self.sp_vision_right.blockSignals(True)
        self.sp_vision_right.setValue(settings.vision_right)
        self.sp_vision_right.blockSignals(False)
        self.sp_vision_off_x.blockSignals(True)
        self.sp_vision_off_x.setValue(settings.vision_off_x)
        self.sp_vision_off_x.blockSignals(False)
        self.sp_vision_off_y.blockSignals(True)
        self.sp_vision_off_y.setValue(settings.vision_off_y)
        self.sp_vision_off_y.blockSignals(False)
        self._refresh_vision_labels()

        self.sp_turn_cd.blockSignals(True)
        self.sp_turn_cd.setValue(settings.sweep_turn_cd)
        self.sp_turn_cd.blockSignals(False)

        self.sp_back_range.blockSignals(True)
        self.sp_back_range.setValue(settings.back_range)
        self.sp_back_range.blockSignals(False)

        self.sp_debounce_conf.blockSignals(True)
        self.sp_debounce_conf.setValue(settings.debounce_conf)
        self.sp_debounce_conf.blockSignals(False)

        self.sp_debounce_ms.blockSignals(True)
        self.sp_debounce_ms.setValue(settings.debounce_ms)
        self.sp_debounce_ms.blockSignals(False)

        self._refresh_strategy_ui()

        self.ck_auto_hp.blockSignals(True)
        self.ck_auto_hp.setChecked(settings.auto_hp_pot)
        self.ck_auto_hp.blockSignals(False)

        self.ck_auto_mp.blockSignals(True)
        self.ck_auto_mp.setChecked(settings.auto_mp_pot)
        self.ck_auto_mp.blockSignals(False)

        self.ck_feed.blockSignals(True)
        self.ck_feed.setChecked(settings.auto_feed_pet)
        self.ck_feed.blockSignals(False)

        if settings.auto_feed_pet:
            self._feed_timer.start()
            self._tick_feed_cd()

        self.sp_feed_interval_min.blockSignals(True)
        self.sp_feed_interval_min.setValue(settings.feed_interval_min)
        self.sp_feed_interval_min.blockSignals(False)

        self.sp_feed_interval_max.blockSignals(True)
        self.sp_feed_interval_max.setValue(settings.feed_interval_max)
        self.sp_feed_interval_max.blockSignals(False)

        # 自定义定时行为列表
        self._refresh_timers()
        if settings.custom_timers:
            self._feed_timer.start()

        self.ck_afk.blockSignals(True)
        self.ck_afk.setChecked(settings.anti_afk_enabled)
        self.ck_afk.blockSignals(False)

        self.sp_afk_min.blockSignals(True)
        self.sp_afk_min.setValue(settings.anti_afk_min)
        self.sp_afk_min.blockSignals(False)

        self.sp_afk_max.blockSignals(True)
        self.sp_afk_max.setValue(settings.anti_afk_max)
        self.sp_afk_max.blockSignals(False)

        self.cmb_afk_type.blockSignals(True)
        ai = self.cmb_afk_type.findData(settings.anti_afk_type)
        self.cmb_afk_type.setCurrentIndex(max(0, ai))
        self.cmb_afk_type.blockSignals(False)

        self.sp_rest_min.blockSignals(True)
        self.sp_rest_min.setValue(settings.anti_afk_rest_min)
        self.sp_rest_min.blockSignals(False)

        self.sp_rest_max.blockSignals(True)
        self.sp_rest_max.setValue(settings.anti_afk_rest_max)
        self.sp_rest_max.blockSignals(False)

        self._refresh_afk_ui()

        # HP/MP 条：先显示全局值；打开项目后由 bind() 用项目里存的覆盖
        self._sync_bar_ui()

        dev = settings.input_device
        self.cmb_device.blockSignals(True)
        idx = self.cmb_device.findData(dev)
        if idx >= 0:
            self.cmb_device.setCurrentIndex(idx)
        self.cmb_device.blockSignals(False)

        self.sp_mouse_speed.blockSignals(True)
        self.sp_mouse_speed.setValue(float(settings.mouse_speed))
        self.sp_mouse_speed.blockSignals(False)
        self._apply_input_device(dev)

        self._refresh_auto_ui()
        self._refresh_custom_keys()

    def toggle_auto(self):
        """切换自动开关（由全局热键触发）。"""
        self.btn_auto.setChecked(not self.btn_auto.isChecked())
        self._toggle_auto()

    # ---------------- 清理 ----------------

    def shutdown(self):
        settings.enabled = False
        # 释放所有按键 + 关闭远程键盘连接，停止指令传输
        self._force_release_all()
        from decision import input as dinput
        dinput.shutdown()
        from decision import manual_input
        manual_input.stop()
        if self.thread is not None:
            self.thread.wait(2000)
