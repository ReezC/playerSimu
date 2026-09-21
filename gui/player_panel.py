"""决策参数面板：自动打怪的开关、攻击距离、键盘映射。

决策逻辑在 decision/agent.py，实时推理循环驱动它。这里只负责：
    · 开关自动（按钮 + F11）
    · 攻击距离
    · 键盘映射（7 个键）

所有改动直接写进全局 settings（decision.agent.settings），实时线程每帧
读到最新值，无需重启。
"""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QMessageBox, QProgressBar, QPushButton,
                             QScrollArea, QSlider, QVBoxLayout, QWidget)

from decision import input as dinput
from decision.agent import settings

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
}
for _i in range(1, 13):
    _QT_TO_NAME[getattr(Qt, "Key_F%d" % _i)] = "f%d" % _i
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _QT_TO_NAME[getattr(Qt, "Key_" + _c)] = _c.lower()
for _i in range(10):
    _QT_TO_NAME[getattr(Qt, "Key_%d" % _i)] = str(_i)


def _sample_bar_color(rect):
    """截取框选区域，采样血条「填充色」，返回 [[B,G,R],[B,G,R]]；失败返回 None。

    关键：血条上通常有白色数字文本（如 "224/366"）和灰色背景——按亮度采样
    会采到文字而不是填充色（实际踩过）。所以**优先取有彩色的像素**（HSV
    饱和度 > 40，排除白/灰），几乎没有彩色像素才回退到亮度 top 30%。
    """
    import cv2
    import numpy as np
    from core import wincap

    try:
        img = wincap.grab_rect(rect)
    except Exception:
        return None
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
        self._build()
        self.device_connected.connect(self._on_device_connected)
        self._sync_from_settings()

    # ---------------- 界面 ----------------

    def _build(self):
        # 外层：QScrollArea 包裹内容 —— 参数太多时可垂直滚动，不再撑大主窗口
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("决策参数")
        title.setStyleSheet("font-weight: 600; font-size: 14px; color: #202124;")
        root.addWidget(title)

        # ---- 按钮组 ----
        self.btn_auto = QPushButton("开启自动（F11）")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setMinimumHeight(34)
        self.btn_auto.clicked.connect(self._toggle_auto)
        root.addWidget(self.btn_auto)

        self.lbl_state = QLabel("当前：未开启")
        self.lbl_state.setStyleSheet("color: #80868b;")
        root.addWidget(self.lbl_state)

        # ---- 输入设备（置顶）----
        top_form = QFormLayout()
        top_form.setLabelAlignment(Qt.AlignLeft)

        self.cmb_device = QComboBox()
        self.cmb_device.addItem("本地", "local")
        self.cmb_device.addItem("Pro Micro", "remote")
        self.cmb_device.currentIndexChanged.connect(self._on_input_device)
        top_form.addRow("输入设备", self.cmb_device)

        self.lbl_device_state = QLabel("本地键盘")
        self.lbl_device_state.setStyleSheet("color: #80868b;")
        top_form.addRow("", self.lbl_device_state)

        # 随机输入延迟 [min, max]（毫秒）：点按类按键之间的随机间隔
        self.sp_delay_min = self._spin(0, 1000, 70, 0)
        self.sp_delay_max = self._spin(0, 1000, 130, 0)
        self.sp_delay_min.valueChanged.connect(self._on_input_delay)
        self.sp_delay_max.valueChanged.connect(self._on_input_delay)
        delay_row = QHBoxLayout()
        delay_row.addWidget(self.sp_delay_min)
        delay_row.addWidget(QLabel("~"))
        delay_row.addWidget(self.sp_delay_max)
        delay_row.addWidget(QLabel("ms"))
        top_form.addRow("随机输入延迟", delay_row)

        # 输出行为 CD（毫秒）：攻击/跳输出/反向跳回身等输出行为的节奏 = CD + 随机延迟
        self.sp_attack_cd = self._spin(0, 10000, 0, 0)
        self.sp_attack_cd.valueChanged.connect(self._on_attack_cd)
        top_form.addRow("输出行为CD(ms)", self.sp_attack_cd)

        root.addLayout(top_form)

        # ---- 参数模板 ----
        tpl = QGroupBox("参数模板")
        tf = QHBoxLayout(tpl)
        btn_save_tpl = QPushButton("保存模板")
        btn_save_tpl.clicked.connect(self._save_template)
        btn_load_tpl = QPushButton("加载模板")
        btn_load_tpl.clicked.connect(self._load_template)
        tf.addWidget(btn_save_tpl)
        tf.addWidget(btn_load_tpl)
        tf.addStretch(1)
        root.addWidget(tpl)

        # ---- 视野组 ----
        vision_grp = QGroupBox("视野")
        vf = QFormLayout(vision_grp)
        vf.setLabelAlignment(Qt.AlignLeft)

        self.ck_vision_center = QCheckBox("基于画面中心")
        self.ck_vision_center.setToolTip("勾选：以画面屏幕中心为视野中心；\n不勾：以角色位置为中心。")
        self.ck_vision_center.stateChanged.connect(self._on_vision)
        vf.addRow("", self.ck_vision_center)

        self.sp_vision_top = self._spin(-1, 2000, 200, 0)
        self.sp_vision_bottom = self._spin(-1, 2000, 200, 0)
        self.sp_vision_left = self._spin(-1, 2000, 200, 0)
        self.sp_vision_right = self._spin(-1, 2000, 200, 0)
        for _sp in (self.sp_vision_top, self.sp_vision_bottom,
                    self.sp_vision_left, self.sp_vision_right):
            _sp.setToolTip("该方向的视野范围（像素）。设为 -1 表示该方向不限制。")
            _sp.valueChanged.connect(self._on_vision)

        vf.addRow("向上视野", self.sp_vision_top)
        vf.addRow("向下视野", self.sp_vision_bottom)
        vf.addRow("向左视野", self.sp_vision_left)
        vf.addRow("向右视野", self.sp_vision_right)

        self.sp_vision_off_x = self._spin(-2000, 2000, 0, 0)
        self.sp_vision_off_y = self._spin(-2000, 2000, 0, 0)
        self.sp_vision_off_x.setToolTip("视野框基于中心在 x 方向的偏移（像素）。")
        self.sp_vision_off_y.setToolTip("视野框基于中心在 y 方向的偏移（像素）。")
        self.sp_vision_off_x.valueChanged.connect(self._on_vision)
        self.sp_vision_off_y.valueChanged.connect(self._on_vision)
        vf.addRow("x偏移", self.sp_vision_off_x)
        vf.addRow("y偏移", self.sp_vision_off_y)

        root.addWidget(vision_grp)

        # ---- 战斗参数组 ----
        battle = QGroupBox("战斗参数")
        bf = QFormLayout(battle)
        bf.setLabelAlignment(Qt.AlignLeft)

        self.sp_attack = self._spin(0, 2000, 80, 0)
        self.sp_attack.valueChanged.connect(self._on_attack_dist)
        bf.addRow("最大攻击距离", self.sp_attack)

        self.sp_min_attack = self._spin(0, 2000, 0, 0)
        self.sp_min_attack.valueChanged.connect(self._on_min_attack)
        bf.addRow("最小攻击距离", self.sp_min_attack)

        # 规避策略（仅最小攻击距离 > 0 时显示）
        evade_grp = QGroupBox("规避策略")
        ef = QFormLayout(evade_grp)
        ef.setLabelAlignment(Qt.AlignLeft)
        self.cmb_evade = QComboBox()
        self.cmb_evade.addItem("跳", "jump")
        self.cmb_evade.addItem("后退", "back")
        self.cmb_evade.currentIndexChanged.connect(self._on_evade_type)
        ef.addRow("规避类型", self.cmb_evade)
        self.sp_jump_interval = self._spin(0, 10000, 200, 0)
        self.sp_jump_interval.valueChanged.connect(self._on_jump_interval)
        ef.addRow("跳间隔(ms)", self.sp_jump_interval)
        self._lbl_jump_interval = ef.labelForField(self.sp_jump_interval)

        self.sp_jump_random = self._spin(0.0, 1.0, 0.1, 2)
        self.sp_jump_random.setSingleStep(0.05)
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
        self.sp_debounce_conf.valueChanged.connect(self._on_debounce)
        bf.addRow("防抖置信度", self.sp_debounce_conf)

        self.sp_debounce_ms = self._spin(0, 10000, 300, 0)
        self.sp_debounce_ms.valueChanged.connect(self._on_debounce)
        bf.addRow("防抖时间(ms)", self.sp_debounce_ms)

        root.addWidget(battle)

        # ---- 策略参数组 ----
        strategy_grp = QGroupBox("策略参数")
        sf = QFormLayout(strategy_grp)
        sf.setLabelAlignment(Qt.AlignLeft)

        # 策略类型
        self.cmb_strategy = QComboBox()
        self.cmb_strategy.addItem("平地巡逻", "patrol")
        self.cmb_strategy.addItem("扫平台", "sweep")
        self.cmb_strategy.currentIndexChanged.connect(self._on_strategy)
        sf.addRow("策略类型", self.cmb_strategy)

        self.lbl_strategy_note = QLabel("")
        self.lbl_strategy_note.setStyleSheet("color: #80868b;")
        self.lbl_strategy_note.setWordWrap(True)
        sf.addRow("", self.lbl_strategy_note)

        # 换朝向 CD（仅「扫平台」策略用）
        self.sp_turn_cd = self._spin(0, 10000, 1000, 0)
        self.sp_turn_cd.valueChanged.connect(self._on_turn_cd)
        sf.addRow("换朝向CD(ms)", self.sp_turn_cd)
        self._lbl_turn_cd = sf.labelForField(self.sp_turn_cd)

        # 换朝向防抖（仅「扫平台」策略用）
        self.sp_turn_debounce = self._spin(0, 10000, 0, 0)
        self.sp_turn_debounce.valueChanged.connect(self._on_turn_debounce)
        sf.addRow("换朝向防抖(ms)", self.sp_turn_debounce)
        self._lbl_turn_debounce = sf.labelForField(self.sp_turn_debounce)

        # 背后锁定距离（仅「扫平台」策略用）
        self.sp_back_range = self._spin(0, 2000, 100, 0)
        self.sp_back_range.valueChanged.connect(self._on_back_range)
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
            btn.setFixedWidth(120)
            btn.clicked.connect(lambda _c, k=key: self._start_listen(k))
            self._key_buttons[key] = btn

            rm = QPushButton("移除")
            rm.setFixedWidth(48)
            rm.setStyleSheet("padding: 2px 6px;")
            rm.setToolTip("清空这个键的映射")
            rm.clicked.connect(lambda _c, k=key: self._remove_key(k))

            cell = QHBoxLayout()
            cell.setSpacing(6)
            lbl = QLabel(label)
            lbl.setFixedWidth(56)
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
        btn_add_custom.clicked.connect(self._add_custom_key)
        cv.addWidget(btn_add_custom)
        root.addWidget(custom_grp)

        # ---- 自动喝药组 ----
        pot = QGroupBox("自动喝药")
        pg = QFormLayout(pot)
        pg.setLabelAlignment(Qt.AlignLeft)

        self.btn_hp_bar = QPushButton("框选 HP 条")
        self.btn_hp_bar.clicked.connect(lambda: self._pick_bar("hp"))
        self.lbl_hp_bar = QLabel("未选")
        self.lbl_hp_bar.setStyleSheet("color: #80868b;")
        hp_row = QHBoxLayout()
        hp_row.addWidget(self.btn_hp_bar)
        hp_row.addWidget(self.lbl_hp_bar, 1)
        pg.addRow("HP条", hp_row)

        self.btn_mp_bar = QPushButton("框选 MP 条")
        self.btn_mp_bar.clicked.connect(lambda: self._pick_bar("mp"))
        self.lbl_mp_bar = QLabel("未选")
        self.lbl_mp_bar.setStyleSheet("color: #80868b;")
        mp_row = QHBoxLayout()
        mp_row.addWidget(self.btn_mp_bar)
        mp_row.addWidget(self.lbl_mp_bar, 1)
        pg.addRow("MP条", mp_row)

        self.ck_auto_hp = QCheckBox("自动补血")
        self.ck_auto_hp.stateChanged.connect(self._on_auto_hp)
        self.sl_hp = QSlider(Qt.Horizontal)
        self.sl_hp.setRange(0, 100)
        self.sl_hp.setValue(30)
        self.sl_hp.valueChanged.connect(self._on_hp_threshold)
        self.lbl_hp_th = QLabel("30%")
        self.lbl_hp_th.setFixedWidth(40)
        hp_th = QHBoxLayout()
        hp_th.addWidget(self.ck_auto_hp)
        hp_th.addWidget(self.sl_hp, 1)
        hp_th.addWidget(self.lbl_hp_th)
        pg.addRow("HP阈值", hp_th)

        self.ck_auto_mp = QCheckBox("自动补蓝")
        self.ck_auto_mp.stateChanged.connect(self._on_auto_mp)
        self.sl_mp = QSlider(Qt.Horizontal)
        self.sl_mp.setRange(0, 100)
        self.sl_mp.setValue(20)
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
        self.ck_feed.stateChanged.connect(self._on_auto_feed)
        self.lbl_feed_cd = QLabel("")
        self.lbl_feed_cd.setStyleSheet("color:#80868b;")
        feed_row.addWidget(self.ck_feed)
        feed_row.addWidget(self.lbl_feed_cd)
        feed_row.addStretch(1)
        pg.addRow("", feed_row)

        # 喂宠间隔：随机区间 [下限, 上限]（分钟）
        self.sp_feed_interval_min = self._spin(1, 600, 5, 0)
        self.sp_feed_interval_max = self._spin(1, 600, 10, 0)
        self.sp_feed_interval_min.valueChanged.connect(self._on_feed_interval)
        self.sp_feed_interval_max.valueChanged.connect(self._on_feed_interval)
        feed_iv_row = QHBoxLayout()
        feed_iv_row.addWidget(self.sp_feed_interval_min)
        feed_iv_row.addWidget(QLabel("~"))
        feed_iv_row.addWidget(self.sp_feed_interval_max)
        pg.addRow("喂宠间隔(min)", feed_iv_row)

        self._pot_form = pg

        root.addWidget(pot)

        # ---- 防掉线组 ----
        afk = QGroupBox("防掉线")
        af = QFormLayout(afk)

        self.ck_afk = QCheckBox("自动防掉线")
        self.ck_afk.stateChanged.connect(self._on_anti_afk)
        af.addRow("", self.ck_afk)

        self.sp_afk_min = self._spin(1, 600, 5, 0)
        self.sp_afk_max = self._spin(1, 600, 10, 0)
        self.sp_afk_min.valueChanged.connect(self._on_afk_time)
        self.sp_afk_max.valueChanged.connect(self._on_afk_time)
        afk_iv_row = QHBoxLayout()
        afk_iv_row.addWidget(self.sp_afk_min)
        afk_iv_row.addWidget(QLabel("~"))
        afk_iv_row.addWidget(self.sp_afk_max)
        af.addRow("触发时间(min)", afk_iv_row)

        self.btn_edit_afk = QPushButton("编辑防掉线行为")
        self.btn_edit_afk.setStyleSheet(self._BTN_EDIT_SEQ)
        self.btn_edit_afk.clicked.connect(self._edit_anti_afk_seq)
        af.addRow("行为", self.btn_edit_afk)

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

    # ---------------- 控件构造 ----------------

    @staticmethod
    def _spin(lo, hi, val, decimals):
        from gui.widgets import NoWheelDoubleSpinBox
        w = NoWheelDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(val)
        return w

    # ---------------- 事件 ----------------

    def _toggle_auto(self, checked=None):
        on = self.btn_auto.isChecked()
        settings.enabled = on
        if not on:
            self._force_release_all()   # 关闭自动立即释放所有按键，防卡键
        self._refresh_auto_ui()

    def _force_release_all(self):
        """关闭自动时，立即对所有映射键发 key_up（RELEASE），不等 agent 下一帧。

        背景：按键通过 Pro Micro 固件保持，若 RELEASE 命令丢失（网络瞬断）或
        实时线程卡住没走到 release_all，键会卡在按下状态。这里兜底强制释放所有
        映射键；对未按下的键发 RELEASE 无害（固件/SendInput 都忽略）。
        """
        from decision import input as dinput
        for key in settings.keymap.values():
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
        """定时把 settings.enabled 同步到开关 UI（agent 后台停自动时也能反映）。"""
        if self.btn_auto.isChecked() != settings.enabled:
            self._refresh_auto_ui()

    def _on_attack_dist(self, val):
        settings.attack_dist = float(val)
        settings.save()

    def _on_min_attack(self, val):
        settings.min_attack_dist = int(val)
        settings.save()
        self._refresh_evade_ui()

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

    def _on_anti_afk(self, state):
        settings.anti_afk_enabled = bool(state)
        settings.save()

    def _on_afk_time(self, _val=None):
        settings.anti_afk_min = int(self.sp_afk_min.value())
        settings.anti_afk_max = int(self.sp_afk_max.value())
        if settings.anti_afk_max < settings.anti_afk_min:
            settings.anti_afk_max = settings.anti_afk_min
        settings.save()

    def _edit_anti_afk_seq(self):
        """打开防掉线行为编辑器。"""
        from gui.seq_editor import SeqEditorDialog
        dlg = SeqEditorDialog(settings.anti_afk_seq, self, title="防掉线行为编辑器")
        if dlg.exec_():
            settings.anti_afk_seq = dlg.seq()
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

    def _on_turn_debounce(self, val):
        settings.turn_debounce_ms = int(val)
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
                "朝向方向没怪就换向（有 CD）。继承上/下阈值过滤。")
        else:
            self.lbl_strategy_note.setText(
                "以角色脚底为基准：上阈值往上、下阈值往下，范围外的怪不追踪")
        self.sp_turn_cd.setVisible(is_sweep)
        if getattr(self, "_lbl_turn_cd", None) is not None:
            self._lbl_turn_cd.setVisible(is_sweep)
        self.sp_back_range.setVisible(is_sweep)
        if getattr(self, "_lbl_back_range", None) is not None:
            self._lbl_back_range.setVisible(is_sweep)
        self.sp_turn_debounce.setVisible(is_sweep)
        if getattr(self, "_lbl_turn_debounce", None) is not None:
            self._lbl_turn_debounce.setVisible(is_sweep)

    def _on_vision(self, _val=None):
        settings.vision_top = int(self.sp_vision_top.value())
        settings.vision_bottom = int(self.sp_vision_bottom.value())
        settings.vision_left = int(self.sp_vision_left.value())
        settings.vision_right = int(self.sp_vision_right.value())
        settings.vision_center = bool(self.ck_vision_center.isChecked())
        settings.vision_off_x = int(self.sp_vision_off_x.value())
        settings.vision_off_y = int(self.sp_vision_off_y.value())
        settings.save()

    def _on_auto_hp(self, state):
        settings.auto_hp_pot = bool(state)
        settings.save()

    def _on_auto_mp(self, state):
        settings.auto_mp_pot = bool(state)
        settings.save()

    def _on_auto_feed(self, state):
        settings.auto_feed_pet = bool(state)
        settings.save()
        # 开关切换都清计时：打开立即吃一次、关掉不留残值（倒计时重新走）
        settings.feed_next_monotonic = 0.0
        if settings.auto_feed_pet:
            self._feed_timer.start()
            self._tick_feed_cd()
        else:
            self._feed_timer.stop()
            self.lbl_feed_cd.setText("")

    def _tick_feed_cd(self):
        """每秒刷新喂宠倒计时（读 agent 维护的下次喂宠时刻）。"""
        import time
        next_ts = settings.feed_next_monotonic
        if next_ts > 0:
            remaining = next_ts - time.monotonic()
        else:
            remaining = settings.feed_interval_min * 60.0   # agent 还没跑过：按最短间隔（下限）占位
        if remaining < 0:
            remaining = 0
        m = int(remaining) // 60
        s = int(remaining) % 60
        self.lbl_feed_cd.setText("距离下次：%d:%02d" % (m, s))

    def _on_feed_interval(self, _val=None):
        settings.feed_interval_min = int(self.sp_feed_interval_min.value())
        settings.feed_interval_max = int(self.sp_feed_interval_max.value())
        if settings.feed_interval_max < settings.feed_interval_min:
            settings.feed_interval_max = settings.feed_interval_min
        settings.save()

    def _pick_bar(self, kind):
        """框选 HP/MP 条区域，并采样该条的填充色。"""
        from gui.region_selector import select_region
        rect = select_region(self)
        if rect is None:
            return
        color = _sample_bar_color(rect)
        if kind == "hp":
            settings.hp_bar = list(rect)
            settings.hp_color = color
            self.lbl_hp_bar.setText("%d,%d  %dx%d" % rect)
        else:
            settings.mp_bar = list(rect)
            settings.mp_color = color
            self.lbl_mp_bar.setText("%d,%d  %dx%d" % rect)
        settings.save()
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
        """切换输入设备：本地 SendInput 或远程 Pro Micro（后台异步连接）。"""
        from decision import input as dinput
        if dev == "remote":
            from tools.config import get
            host = get("kbd", "host")
            port = int(get("kbd", "port", 9000))
            cert = get("kbd", "cert", "remote_kbd/certs/cert.pem")
            self.lbl_device_state.setText("正在连接 Pro Micro…")
            import threading
            def _do_connect():
                try:
                    dinput.use_network(host, port, cert)
                    self.device_connected.emit("ok")
                except Exception:
                    dinput.use_local()
                    self.device_connected.emit("fail")
            threading.Thread(target=_do_connect, daemon=True).start()
        else:
            dinput.use_local()
            self.lbl_device_state.setText("本地键盘")

    def _on_device_connected(self, result):
        """Pro Micro 连接结果回调（后台线程 → 主线程）。"""
        if result == "ok":
            self.lbl_device_state.setText("已连接 Pro Micro")
        else:
            settings.input_device = "local"
            settings.save()
            self.lbl_device_state.setText("Pro Micro 连接失败，已回退本地")
            self.cmb_device.blockSignals(True)
            idx = self.cmb_device.findData("local")
            if idx >= 0:
                self.cmb_device.setCurrentIndex(idx)
            self.cmb_device.blockSignals(False)

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
            lbl = QLabel(name)
            lbl.setFixedWidth(80)
            btn = QPushButton()
            btn.setFixedWidth(120)
            k = settings.custom_keys.get(name)
            if self._listening_custom == name:
                btn.setText("请按键…")
            else:
                btn.setText(dinput.display_name(k) if k else "无")
            btn.clicked.connect(lambda _c, n=name: self._start_listen_custom(n))
            rm = QPushButton("删除")
            rm.setFixedWidth(48)
            rm.clicked.connect(lambda _c, n=name: self._remove_custom_key(n))
            row.addWidget(lbl)
            row.addWidget(btn)
            row.addWidget(rm)
            row.addStretch(1)
            self._custom_list.addLayout(row)

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

        self.sp_turn_cd.blockSignals(True)
        self.sp_turn_cd.setValue(settings.sweep_turn_cd)
        self.sp_turn_cd.blockSignals(False)

        self.sp_turn_debounce.blockSignals(True)
        self.sp_turn_debounce.setValue(settings.turn_debounce_ms)
        self.sp_turn_debounce.blockSignals(False)

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

        self.ck_afk.blockSignals(True)
        self.ck_afk.setChecked(settings.anti_afk_enabled)
        self.ck_afk.blockSignals(False)

        self.sp_afk_min.blockSignals(True)
        self.sp_afk_min.setValue(settings.anti_afk_min)
        self.sp_afk_min.blockSignals(False)

        self.sp_afk_max.blockSignals(True)
        self.sp_afk_max.setValue(settings.anti_afk_max)
        self.sp_afk_max.blockSignals(False)

        if settings.hp_bar:
            self.lbl_hp_bar.setText("%d,%d  %dx%d" % tuple(settings.hp_bar))
        if settings.mp_bar:
            self.lbl_mp_bar.setText("%d,%d  %dx%d" % tuple(settings.mp_bar))
        self._refresh_bar_visibility()
        self._apply_bar_chunk_color()

        dev = settings.input_device
        self.cmb_device.blockSignals(True)
        idx = self.cmb_device.findData(dev)
        if idx >= 0:
            self.cmb_device.setCurrentIndex(idx)
        self.cmb_device.blockSignals(False)
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
        if self.thread is not None:
            self.thread.wait(2000)
