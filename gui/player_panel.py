"""决策参数面板：自动打怪的开关、攻击距离、键盘映射。

决策逻辑在 decision/agent.py，实时推理循环驱动它。这里只负责：
    · 开关自动（按钮 + F11）
    · 触控板 / 鼠标左右键 / 灵敏度 / 输入设备（「操控」组，本地 F10 开关触控模式）
    · 攻击距离
    · 键盘映射（7 个键）

F10 / F11 / F12 是**本机操作键**：只在本窗口起作用，永不转发给游戏
（见 decision/input.py 的 LOCAL_ONLY）。

所有改动直接写进 settings 单例（decision.agent.settings），实时线程每帧读到
最新值，无需重启。**这份参数只按项目存**：主窗口打开项目时注册保存钩子，每次
save() 就整份写回该项目的 project.yaml（见 MainWindow._bind_decision_params）。
没打开项目时参数取自**最近打开的那个项目**（只读不落盘）：界面照样能调、实时
生效，但一打开项目就被项目的值覆盖 —— 参数必须有明确归属。
"""

import time

from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                             QDialog, QFileDialog, QFormLayout, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
                             QPushButton, QScrollArea, QSizePolicy, QSlider,
                             QVBoxLayout, QWidget)

from core import behavior           # 重连回来"没恢复自动"要落一条日志（`reconnect_resume` ✓）
from core import zones as zones_mod
# ⭐ 「自动测谎（视觉追踪）」组的参数落 `config/live.yaml`（本机偏好 ✓ 2026-10-09）
from core.config import load_live, update_live
from decision import input as dinput
from decision.agent import (ANTI_AFK_TYPES, ZONE_GOTO_RETRY_S, load_rect,
                            settings)
from gui import theme
from gui.project import last_opened
# NoWheel* 必须模块级导入：控件在 _build() 里建，懒导入到不了那儿。
# （详见 docs/UI规范.md：滚轮不许改参数）
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox,
                         NoWheelSlider, NoWheelSpinBox, forward_wheel,
                         play_sound, scroll_page, sound_player, stop_sound)
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
    # ⭐ 「拾取」（用户 2026-10-06 ✓ 原话："在键盘映射组里加一项默认键盘映射『拾取』"）——
    #   默认 `z`（这类游戏的常见拾取键 ✓，不对就点一下按钮再按实际那个键 ✓）；
    #   ⚠ 它**只在「战斗参数 → 自动拾取」开着时**才被点按 ✓（见 `_pickup_beat` ✓）。
    ("pickup", "拾取", "z"),
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


class BattleZoneListDialog(QDialog):
    """「**编辑战斗区域**」列表弹窗（用户 2026-09-28 第 2 条 ✓ 原话："点击『编辑』就打开了
    『编辑战斗区域』弹窗，**初始是空列表**，可以往里面**添加项目、删除项目、双击编辑项目**。
    点**添加**才是现在已经做好的『编辑战斗区域{foothold集合名}』编辑弹窗"✓）。

    · **列表** = `settings.battle_zones` 的每一项（**真源** ✓，顺序 = 界面顺序 ✓）；
    · **添加** ⇒ **先选集合名**（`QInputDialog`，候选由外面推 ✓）⇒ 再开 `BattleZoneDialog`
      配这一项 ✓（"点添加才是那个编辑弹窗" ✓）；
    · **双击一行 = 编辑它**（同一个 `BattleZoneDialog` ✓）；
    · **删除** = 选中后删（有确认 ✓）；
    · ⚠ **每次增删改立刻写回**（`on_save` 回调 ⇒ `PlayerPanel._bz_commit` ⇒ 存盘 + 同步派生
      副本 ✓）⇒ "关掉弹窗"没有额外确定/取消语义 ✓（老界面就是"改完即存" ✓ 保持不变 ✓）。
    · ⚠ 走**回调**而不是直接抓 `settings`：本类**不认识** settings ⇒ 好测 ✓（用例塞一个假的
      `on_save` 就能收下每次结果 ✓）。
    """

    def __init__(self, zones, names=None, on_save=None, parent=None,
                 picker_factory=None):
        super().__init__(parent)
        self.setWindowTitle("编辑战斗区域")
        # ⚠ 只钉**下限**（比这更小按钮就挤没了 ✓），并给一个**第一次打开**的尺寸
        #   —— 之后按客户端记住（见 __init__ 末尾的 `bind_window_state` ✓，docs/UI规范.md §11）
        self.setMinimumSize(470, 360)
        self.resize(560, 460)
        self._names = [str(n) for n in (names or []) if str(n)]
        #: 「idle 回归 foothold」那块**只读视图**的工厂（**透传给子弹窗** ✓
        #: —— 见 `BattleZoneDialog.__init__` 里那段说明 ✓）
        self._picker_factory = picker_factory
        self._on_save = on_save
        self._zones = [dict(z) for z in (zones or []) if isinstance(z, dict)]

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        self.lst = QListWidget()
        # ⭐ 框选多选 + Ctrl 连续选中/反选（用户 2026-10-02 ✓）：
        #   `ExtendedSelection` 一次到位支持三种手势 ——
        #   · **Ctrl+点击** = 切换那一项的选中（反选 ✓）；
        #   · **Shift+点击** = 从当前项到点击项的范围选中 ✓；
        #   · **鼠标拖框**（空白处按下拖）= 框选范围内的所有项 ✓。
        #   双击编辑不受影响（双击会让那项变成 current，然后开编辑 ✓）。
        self.lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # 双击 = 编辑（同 `gui/zone_editor.py` 的列表范式 ✓）
        self.lst.itemDoubleClicked.connect(self._on_edit)
        # ⭐⭐ 2026-10-01：「可以战斗」的勾选框**挪到主窗口**了（用户要求 ✓）⇒ 这里列表
        #   只显示**只读摘要**（含"可以战斗"字样）；勾选入口在主窗口每项前面 ✓。
        self.lst.setToolTip(
            "**双击一行**改它的参数 ✓（区域查询CD / idle 回归 foothold / "
            "最小战斗时长 / 最大战斗时长 / 到点去哪）。\n\n"
            "**多选**（用户 2026-10-02 ✓）：\n"
            "· **Ctrl+点击** = 切换选中（反选 ✓）；\n"
            "· **Shift+点击** = 范围选中；\n"
            "· **空白处拖框** = 框选多项 ✓。\n"
            "选了多项后**删除**会一次删干净 ✓。\n\n"
            "「**可以战斗**」在主窗口「路线规划」组的每项前面勾 ✓。")
        root.addWidget(self.lst, 1)

        btns = QHBoxLayout()
        btns.setSpacing(6)
        self.btn_add = QPushButton("添加")
        self.btn_add.setToolTip("选一个**集合**加进来 ⇒ 立刻打开它的参数窗 ✓")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_del = QPushButton("删除")
        self.btn_del.setToolTip("删掉**选中**的那一项（有确认 ✓）")
        self.btn_del.clicked.connect(self._on_del)
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        btns.addStretch(1)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)
        btns.addWidget(self.btn_close)
        root.addLayout(btns)

        hint = QLabel("「**可以战斗**」的勾选在**主窗口「路线规划」组的每项前面**勾 ✓（这里只显示）。\n"
                      "**双击一行**改它的其他参数 ✓。改完**立刻生效并跟着项目存** ✓。")
        hint.setStyleSheet("color: #80868b;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # ⭐ 「**最大战斗时长**」的**倒计时**（用户 2026-09-28 要求 ✓ 原话："编辑战斗区域里，
        #   要用**灰字**显示**最大战斗时长的倒计时**"✓）。
        #   数据来自 `agent._publish_fight_clock` 写在 `settings` 上的那三件套
        #   （`fight_zone_name` / `fight_elapsed_s` / `fight_cap_s` ✓）—— ⚠ **只读** ✓
        #   面板**不许**当第二个写者（同 `pos_state()` 那条纪律 ✓）。
        #   ⚠⚠ **`fight_elapsed_s` = "已累计在打的秒数"**（用户 2026-09-28 改口径 ✓）⇒ 剩余
        #      **直接** `cap - elapsed` ✓、**不许**再拿时钟去减 ✗ —— 寻路打断期间时间是**暂停**
        #      的（不清零 ✓），拿墙钟减会把打断那段也算进去 ⇒ 显示会跳 ✗（改之前就是这么错的）。
        #   ⚠ 只有**某个区域真的在计时**时才显示 ✓（不在区域 / 那项不限 ⇒ 什么都不显示 ✓
        #     别显示 `0:00` ✗ —— 那会让人以为"马上要走了"）。
        self.lbl_cd = QLabel("")
        self.lbl_cd.setStyleSheet("color: #80868b;")      # ⭐ **灰字**（用户明确要求 ✓）
        self.lbl_cd.setWordWrap(True)
        root.addWidget(self.lbl_cd)
        # 每 0.5 秒刷一次（够跟手、又不费 ✓）。⚠ 弹窗关掉后这个 timer 跟着对象一起没 ✓
        # （本弹窗是**每次点「编辑」新建**的 ✓ 不是常驻单例）⇒ 不用额外接 showEvent/hideEvent ✓。
        self._cd_timer = QTimer(self)
        self._cd_timer.setInterval(500)
        self._cd_timer.timeout.connect(self._refresh_fight_cd)
        self._cd_timer.start()

        self._reload()
        # 拉过的大小/位置按客户端记住（docs/UI规范.md §11 ✓ —— 用户 2026-09-28 报的
        # "子窗口太大、还不让缩小"里，也有"尺寸根本留不住、每次都回默认"那一半 ✓）
        theme.bind_window_state(self, "battle_zone_list")

    # ---------------- 内部 ----------------

    def _refresh_fight_cd(self):
        """刷新那行**灰字倒计时**（用户 2026-09-28 ✓）—— ⚠ **只读** `settings` 上那三件套 ✓。

        ⚠ 口径全在 `decision/agent._publish_fight_clock` / `_fight_beat`（**一处** ✓ 别在这儿
          另算一套"什么时候算在打 / 从哪一刻算" ✗ —— 那是 agent 的事实的判断 ✓）。
        ⚠⚠ **剩余是"减去已累计的秒数"**（用户 2026-09-28 改口径后 ✓）：`cap - fight_elapsed_s`
          ✓ —— **不是** `cap - (monotonic - started_at)` ✗。原因：现在**只有寻路会打断时间**
          （用户原话："只要一直战斗就不应该有任何理由停时间"✓），打断期间是**暂停累计**
          （不清零 ✓）⇒ 拿墙钟减会把打断的那段算进去 ⇒ 显示会莫名跳 ✗。
        """
        try:
            name = getattr(settings, "fight_zone_name", None)
            el = float(getattr(settings, "fight_elapsed_s", 0.0) or 0.0)
            cap = float(getattr(settings, "fight_cap_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            name, el, cap = None, 0.0, 0.0
        if not name or cap <= 0.0:
            self.lbl_cd.setText("")           # 没在计时 ⇒ 什么都不显示（别显示 0:00 ✗）
            return
        left = max(0.0, cap - el)
        self.lbl_cd.setText(
            "⏱ 「%s」本轮已连续打 %.0f s，还剩 %.0f s（上限 %.0f s）—— 到点会按这一项的"
            "「到点去哪」换地方" % (name, cap - left, left, cap))

    def _reload(self):
        """把 `self._zones` 灌进列表 —— 一行 = 一个区域项的**只读摘要**（含"可以战斗"字样 ✓）。

        ⭐ 2026-10-01：「可以战斗」的**勾选框**挪到主窗口了 ⇒ 这里不再有 `ItemIsUserCheckable`
          和 `setCheckState`（勾选入口在主窗口每项前面 ✓），文本里把"可以战斗"读回来 ✓。
        """
        # ⚠⚠ **重画期间仍屏蔽信号**（经典坑 ✗）：`clear()` / `addItem()` 都会发 `itemChanged`
        #   ⇒ 不屏蔽的话，重画一遍就等于"把每一行的状态再写一遍配置"（白写盘 / 写乱 ✗）。
        self.lst.blockSignals(True)
        try:
            self.lst.clear()
            for z in self._zones:
                nm = str(z.get("set") or "")
                bits = []
                if z.get("can_fight"):
                    # ⚠ 纯文本控件（`QListWidgetItem`）**不认 markdown** ✗ —— 原来写着
                    #   `"**可以战斗**"` ⇒ 界面上原样显示四个星号 ✓（2026-10-04 顺手修 ✓；
                    #   同一行的另一个说法是没勾时的 `⚠ 未启用，不会生效` ✓）。
                    bits.append("可以战斗")
                try:
                    bits.append("CD %.1fs" % float(z.get("cd_s") or ZONE_GOTO_RETRY_S))
                except (TypeError, ValueError):
                    pass
                _ids = [str(x) for x in (z.get("idle_footholds") or []) if str(x).strip()]
                if _ids:
                    bits.append("idle #%s" % "、#".join(_ids[:3])
                                + ("…等%d块" % len(_ids) if len(_ids) > 3 else ""))
                try:
                    _fn = float(z.get("fight_min_s") or 0.0)
                except (TypeError, ValueError):
                    _fn = 0.0
                if _fn > 0:
                    bits.append("至少打 %.0fs" % _fn)
                try:
                    _fm = float(z.get("fight_max_s") or 0.0)
                except (TypeError, ValueError):
                    _fm = 0.0
                if _fm > 0:
                    bits.append("最多打 %.0fs" % _fm)
                # ⭐ 没勾「可以战斗」⇒ 当场标一句"不生效"（用户 2026-10-04 ✓ 原话："如果不生效
                #   动态显示当前的配置有没有用并简短例如『该区域未启用，不会生效』"✓）。
                _off = "" if z.get("can_fight") else "    ⚠ 未启用，不会生效"
                item = QListWidgetItem(("%s    ·    %s" % (nm, "，".join(bits))
                                        if bits else nm) + _off)
                item.setData(Qt.UserRole, nm)
                # ⭐ 去掉勾选框：`QListWidgetItem` **默认就带** `ItemIsUserCheckable` ✗
                #   （勾选框挪到主窗口了）⇒ 显式把这一位摘掉 ✓。
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                self.lst.addItem(item)
        finally:
            self.lst.blockSignals(False)

    def zones(self):
        """当前这一份（`list of dict` ✓ —— 用例 / 外面都能直接读 ✓）。"""
        return [dict(z) for z in self._zones]

    def _enabled_map(self):
        """`{集合名: 勾没勾「可以战斗」}` —— **推给子弹窗**判断"目的地会不会去" ✓。

        ⚠ 为什么从这儿推：子弹窗不认识 `settings`（见类说明 ✓），而"哪一项没启用"是
          **同一份 `battle_zones` 里别项的状态** ⇒ 只有持有整份的调用方知道 ✓
          （同 `names` / `picker_factory` 那套"面板间推送" ✓）。
        """
        return {str(z.get("set") or ""): bool(z.get("can_fight"))
                for z in self._zones if str(z.get("set") or "")}

    def _commit(self, reload=True):
        """**唯一出口**：写回一笔 ✓（`reload=True` 时顺手重画列表 ✓）。"""
        if reload:
            self._reload()
        if callable(self._on_save):
            self._on_save([dict(z) for z in self._zones])

    def _picked(self):
        it = self.lst.currentItem()
        return str(it.data(Qt.UserRole) or "") if it is not None else ""

    def choose_name(self):
        """弹那个"选集合"的小窗 ⇒ 返回选中的名字（取消 / 没有候选 ⇒ `""` ✓）。

        ⚠ 独立成一个方法：用例可以**替身掉它**（免得弹真窗口 ✗ 也让"添加"这条能测 ✓）。
        """
        cand = [n for n in self._names
                if n not in set(str(z.get("set") or "") for z in self._zones)]
        if not cand:
            QMessageBox.information(
                self, "没有可加的集合",
                "候选集合都加过了（或者这张图还没注册集合）。\n\n"
                "集合是在「寻路编辑器」里圈出来的 —— 先确认那边已经有了 ✓。")
            return ""
        nm, ok = QInputDialog.getItem(self, "添加战斗区域", "选一个集合：", cand, 0, False)
        return str(nm) if (ok and nm) else ""

    def _on_add(self):
        nm = self.choose_name()
        if not nm:
            return
        dlg = BattleZoneDialog({"set": nm, "cd_s": ZONE_GOTO_RETRY_S},
                               names=self._names, parent=self,
                               picker_factory=self._picker_factory,
                               enabled_map=self._enabled_map())
        if dlg.exec_() != QDialog.Accepted:
            return
        self._zones.append(dlg.zone())
        self._commit()

    def _on_edit(self, *_a):
        nm = self._picked()
        cur = [z for z in self._zones if str(z.get("set") or "") == nm]
        if not cur:
            return
        dlg = BattleZoneDialog(cur[0], names=self._names, parent=self,
                               picker_factory=self._picker_factory,
                               enabled_map=self._enabled_map())
        if dlg.exec_() != QDialog.Accepted:
            return
        new = dlg.zone()
        self._zones = [z for z in self._zones if str(z.get("set") or "") != nm]
        self._zones.append(new)               # 顺序无语义（同老界面 ✓）
        self._commit()

    def _on_del(self):
        # ⭐ 批量删除（用户 2026-10-02 ✓）：多选了就一次删干净。
        #   `selectedItems()` = 框选 / Ctrl 多选的所有项；**空 ⇒ 退化到 `currentItem()`**
        #   （老行为 ✓ —— 单选时 selectedItems 也是空的，那一下还能用 currentItem 删）。
        items = self.lst.selectedItems() or ([self.lst.currentItem()]
                                             if self.lst.currentItem() else [])
        names = [str(it.data(Qt.UserRole) or "") for it in items
                 if it is not None]
        names = [n for n in names if n]
        if not names:
            QMessageBox.information(self, "先选一项",
                                    "在列表里点一下要删的那一项 ✓（**Ctrl+点击** 可多选）")
            return
        if len(names) == 1:
            msg = "确定把「%s」从战斗区域里删掉吗？" % names[0]
        else:
            msg = "确定把这 %d 项从战斗区域里删掉吗？\n\n%s" % (
                len(names), "、".join(names))
        if QMessageBox.question(self, "删除战斗区域", msg) != QMessageBox.Yes:
            return
        _kill = set(names)
        self._zones = [z for z in self._zones
                       if str(z.get("set") or "") not in _kill]
        self._commit()


class BattleZoneDialog(QDialog):
    """**战斗区域**每一项的编辑弹窗（用户 2026-09-28 原话："**按钮和弹窗呢？你做的我没法测**"）。

    为什么要有它：`settings.battle_zones` 每项是 **5 个字段**（`set` / `cd_s` /
    `idle_foothold` / `fight_max_s` / `fight_dst` ✓，见 `decision/agent.py` 的 `__init__` ✓），
    可原来界面上那个「限制战斗区域」**只读写派生副本 `battle_zone_sets`**（= 一串集合名 ✗）
    ⇒ 后四个字段**一个入口都没有** ⇒ 配不了、也就**测不了** ✗（用户当场就是这么说的 ✓）。

    ⚠ **`set`（集合名）在这里是只读的**：它是这一项的**身份**（`battle_zones` 里唯一 ✓），
      也是派生副本 `battle_zone_sets` 的来源 —— 允许改名就要连带处理副本、以及别处
      已经写了这个名字的地方（`fight_dst` / 已配的边……✗）⇒ 要换区域就**删了重加** ✓。
    ⚠ 候选集合名**从外面推**进来（`names` ✓，同 `set_zone_sets` 那套"面板间推送" ✓）——
      本面板不持有地图 id（集合是按 map id 存在 `core/zones` 里的 ✓）。
    """

    def __init__(self, zone, names=None, parent=None, picker_factory=None,
                 enabled_map=None):
        super().__init__(parent)
        z0 = dict(zone or {})
        #: ⭐ **只读视图工厂**（用户 2026-09-28 要求 ✓ 原话："编辑战斗区域的**子弹窗**需要有
        #:   『foothold 集合编辑器』的**同款视图（只读）**，可以通过**点选**来**查看 foothold
        #:   参数**、**配置「idle回归foothold」**"✓）。
        #:   签名 `fn(set_name, current_fid) -> QWidget | None` ✓ —— **由外面推**进来
        #:   （本面板不持有地图 id ✗，同 `names` 那套"面板间推送" ✓）；
        #:   **没推 / 推来 None ⇒ 退回原来的文本框** ✓（老环境、以及用例里直接 new 的
        #:   那些实例都不受影响 ✓）。视图本体见 `gui/foothold_picker.py` ✓。
        self._picker_factory = picker_factory
        self._picker = None
        self._set = str(z0.get("set") or "")
        #: ⚠ **「可以战斗」不在这里改**（用户要求挪出去勾 ✓；2026-10-01 起在
        #:   **主窗口「路线规划」组每项前面**勾 ✓）—— 这里只把它**记下来**，
        #:   `zone()` 时**原样带回** ✓ ⇒ 免得"进来编辑一次就把勾选弄丢" ✗✗。
        # ⚠ 新增区域（列表"添加"传来的 z0 没有 can_fight 键）默认「可以战斗」；
        # 否则新加的战斗区域会被排除出 battle_zone_sets 白名单，导致玩家站在它上面、
        # 场景里又有怪时 tick 提前返回 leave_battle_zone，idle 回归（以及战斗）都不触发。
        # 编辑已有区域时，显式的值会被原样保留（get 的第二参数只兜底"没有该键"的情况）。
        self._can_fight = bool(z0.get("can_fight", True))
        #: ⭐⭐ **每个区域项的「可以战斗」现状**（`{集合名: bool}` ✓）—— 由外面推进来 ✓。
        #:   用途：本项的目的地（idle 切换平台 / 到点去哪）若指向**没启用的项** ⇒ 弹窗要
        #:   **当场**说一句"配了也不会去" ✓（用户 2026-10-04 ✓ 原话："修功能，然后如果不生效
        #:   动态显示当前的配置有没有用并简短例如『该区域未启用，不会生效』"✓）。
        #:   ⚠ 没推 / 推来 None ⇒ 只报"本项自己启不启用" ✓（老环境与直接 new 的用例不受影响 ✓）。
        self._enabled_map = dict(enabled_map or {})
        self.setWindowTitle(("编辑战斗区域「%s」" % self._set) if self._set
                            else "添加战斗区域")
        # ⚠ 只钉**宽度**、不钉高度，并给一个**打开的尺寸**（2026-09-28 现场修 ✗ ——
        #   用户报"**编辑战斗区域子窗口也太大了，还不让缩小**"✓）。
        #   原来这里连 `resize` 都没有 ⇒ 窗口多大**完全由布局的 sizeHint 决定** ✗，
        #   而里面那块只读视图是 `QGraphicsView`（`FootholdPicker`）—— 它默认把
        #   **场景尺寸**（整张图的底图 + foothold ≈ 2000×1000）当 sizeHint ⇒
        #   一显示就被撑到**上千高**，之后任何一次布局重算还会拽回去 ⇒ "太大 + 拖不动"✗。
        #   ⇒ 根治在视图那边（`gui/foothold_picker.py` 的 `sizeHint` ✓ 现在只表 320×200 ✓）；
        #     这里再补一个合理初值，免得"每次打开位置尺寸都飘"✓。
        self.resize(560, 620)
        self.setMinimumWidth(430)

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        # ⭐⭐ **这行配置到底会不会生效** —— 一行短话，随内容实时变（用户 2026-10-04 ✓ 原话：
        #   "修功能，然后如果不生效动态显示当前的配置有没有用并简短例如『该区域未启用，
        #   不会生效』"✓）。判据与措辞全在 `_update_effect_hint` ✓（唯一出口 ✓）。
        #   ⚠ 纯文本，别写 markdown ✗（QLabel 不认 ✓ 同 `live_panel._show_mmap_push` 那条纪律 ✓）。
        self.lbl_effect = QLabel("")
        self.lbl_effect.setWordWrap(True)
        root.addWidget(self.lbl_effect)

        form_top = QFormLayout()
        form_top.setLabelAlignment(Qt.AlignLeft)
        # ⭐ 拆成上下两段表单（用户 2026-10-02 ✓ 要求"idle 回归 foothold 块放在 idle 模式
        #   下方"⇒ 视图块要夹在「idle 去哪」与「最小战斗时长」之间 ✓ —— 表单按 addRow 顺序
        #   排，没法在中间插一段 ⇒ 拆成 `form_top` / `form_bottom`，中间塞 `_idle_box`）。
        form_bottom = QFormLayout()
        form_bottom.setLabelAlignment(Qt.AlignLeft)

        # ① 区域集合（只读，见类说明）
        lb = QLabel(self._set or "（未选）")
        lb.setStyleSheet("font-weight: 600;")
        lb.setToolTip("这一项管的是哪块平台（集合名 ✓）。\n"
                      "⚠ 这里不给改：它是这一项的**身份** —— 要换区域请**删了重加** ✓。")
        form_top.addRow("区域集合", lb)

        # ⚠ **「可以战斗」不在这里**（用户要求挪出去勾 ✓；2026-10-01 起在**主窗口
        #   「路线规划」组每项前面**勾 ✓）—— 这里删掉控件，但 `__init__` 把它记进
        #   `self._can_fight`、`zone()` **原样带回** ✓（否则"进来编辑一次就把勾选弄丢" ✗✗）。

        # ② 区域查询CD(s)（旧「前往重下间隔(s)」的**逐项版** ✓）
        self.sp_cd = NoWheelDoubleSpinBox()
        self.sp_cd.setRange(0.5, 600.0)
        self.sp_cd.setDecimals(1)
        self.sp_cd.setSingleStep(0.5)
        self.sp_cd.setValue(max(0.5, float(z0.get("cd_s") or ZONE_GOTO_RETRY_S)))
        tip_cd = ("**这一项**的「多久才能再来一次」：追击下前往、区域筛缓存都以它为准 ✓\n"
                  "（三处调用点各传各的：回区域重下传目标区域项、追击/筛缓存传"
                  "「怪那块 foothold 命中的区域项」✓）。\n\n"
                  "⚠ 下限 0.5 —— 再小就等于「每拍重下 / 每拍重算」✗。\n"
                  "默认 %.1f（归属不到任何区域项时也用它兜底 ✓）。" % ZONE_GOTO_RETRY_S)
        self.sp_cd.setToolTip(tip_cd)
        form_top.addRow("区域查询CD(s)", self.sp_cd)

        # ⭐⭐ idle 模式下拉已移除（用户 2026-10-02 ✓ 取消互斥）：
        #   idle 回归 foothold 和 切换平台**同时**生效，不再二选一 ✓。
        #   切换平台的参数（idle持续n秒切换平台 / idle 去哪）移到了下面 form_bottom ✓。
        _idle_dst_val = str(z0.get("idle_dst_set") or "").strip()

        # ③ idle 回归 foothold（**列表随机**，2026-09-29 用户要求 ✓）
        #    交互流程（用户定的 ✓）：**走选中**（在上面的只读视图里点一条）→ **点添加** →
        #    **显示在列表里**；列表里可**删除**。运行时每次触发回归**随机抽一个** ✓。
        #    ⭐ 2026-09-28 用户要求（原话）："编辑战斗区域的**子弹窗**需要有『foothold 集合
        #    编辑器』的**同款视图（只读）**，可以通过**点选**来**查看 foothold 参数**"✓ ⇒
        #    有工厂就摆**只读视图**，没有就**退回文本框** ✓（视图本体：`gui/foothold_picker.py` ✓）。
        _legacy_ids = [str(x) for x in (z0.get("idle_footholds") or []) if str(x).strip()]
        if not _legacy_ids and str(z0.get("idle_foothold") or "").strip():
            _legacy_ids = [str(z0["idle_foothold"]).strip()]     # 老单值配置 ✓
        self.ed_idle = QLineEdit(" ".join(_legacy_ids))          # ← 兜底（没工厂时用 ✓）
        self.ed_idle.setPlaceholderText("例如 41 或 41 5 27（空 = 不做 idle 回归）")
        self.ed_idle.setToolTip(
            "**闲着没事的时候回哪些砖上站着**（**列表随机** ✓）—— 填 **foothold 编号**，\n"
            "多个用空格隔开；每次触发回归随机抽一块 ✓。\n\n"
            "· 空 = **不做这件事**（老行为：不跨层水平走 ✓）。\n\n"
            "⚠ 只填**编号**，不要填集合名（集合是上面那一格的事 ✓）。")
        try:
            self._picker = self._make_picker()
        except Exception:                   # noqa: BLE001 —— 视图建不出来就退回文本框 ✓（别炸弹窗 ✗）
            self._picker = None
        if self._picker is None:
            form_top.addRow("idle 回归 foothold", self.ed_idle)
        else:
            # ⭐ 有视图 ⇒ **把文本框收起来**（那套"手填编号"就不摆了 ✓ 免得两处表达同一件事 ✗）
            box = QVBoxLayout()
            box.setSpacing(4)
            box.addWidget(self._picker, 1)
            self.lbl_fh = QLabel("")
            self.lbl_fh.setStyleSheet("color: #5f6368;")
            self.lbl_fh.setWordWrap(True)
            box.addWidget(self.lbl_fh)
            # ---- 列表（回归点池）+ 增删行（§4：一件东西一行 ✓）----
            # ⚠⚠ 2026-09-30 用户报「编辑战斗区域弹窗双击打不开编辑」的根因就在这行：
            #   之前调试时把它换成 `= None` 忘了还原 ⇒ `__init__` 在下一行
            #   `setMaximumHeight` 当场 AttributeError ⇒ 弹窗建一半炸掉、
            #   信号槽里异常被 PyQt 吞掉 = 双击「没反应」✗。
            self.lst_idle = QListWidget()
            self.lst_idle.setMaximumHeight(76)      # ~3 行够用；长了弹窗被顶高 ✗
            self.lst_idle.setToolTip(
                "**回归点池**：每次触发 idle 回归，**随机抽一块**去站 ✓。\n"
                "点一行 = 在上面的视图里看那块的参数 ✓；删除见下面按钮 ✓。")
            self.lst_idle.itemClicked.connect(self._on_idle_row_clicked)
            for fid in _legacy_ids:
                self.lst_idle.addItem("#%s" % fid)
            box.addWidget(self.lst_idle)
            row_i = QHBoxLayout()
            self.btn_idle_add = QPushButton("添加选中")
            self.btn_idle_add.setToolTip(
                "把上面视图里**当前选中的 foothold** 加进回归点池 ✓\n"
                "（池里的每一块都可能被随机抽中 ✓）")
            self.btn_idle_add.clicked.connect(self._on_add_idle)
            row_i.addWidget(self.btn_idle_add)
            self.btn_idle_del = QPushButton("删除选中")
            self.btn_idle_del.setEnabled(False)
            self.btn_idle_del.setToolTip("把列表里**选中的那行**从回归点池里删掉 ✓")
            self.btn_idle_del.clicked.connect(self._on_del_idle)
            row_i.addWidget(self.btn_idle_del)
            self.btn_idle_clear = QPushButton("清空")
            self.btn_idle_clear.setToolTip(
                "清空回归点池（= 不做 idle 回归 ✓ 老行为：不跨层水平走 ✓）")
            self.btn_idle_clear.clicked.connect(self._on_clear_idle)
            row_i.addWidget(self.btn_idle_clear)
            row_i.addStretch(1)
            box.addLayout(row_i)
            # ⭐⭐ **不塞进 `QFormLayout`**（用户 2026-09-28 报："图又窄"✓）：表单会给它让出
            #   **标签列**那一竖条的宽度 ✗，而这块是**图**、宽度就是一切 ⇒ 单独拿出来占满整宽 ✓。
            #   ⚠ 位置见下面 `root.insertWidget(0, box, 1)` —— 用户要求"可视图放在**最上面**"✓。
            self._idle_box = box
            self._refresh_idle_label()

        # ⭐ idle 切换平台参数（用户 2026-10-02 ✓ —— 移到这，最小战斗时长之前）：
        #   取消互斥后，idle 回归 foothold（上面 _idle_box）和切换平台**同时**生效 ✓。
        #   idle 持续 N 秒没怪才下切换平台任务；delay=0 ⇒ 不启用、idle 去哪置灰 ✓。
        self.sp_idle_delay = NoWheelDoubleSpinBox()
        self.sp_idle_delay.setRange(0.0, 60.0)
        self.sp_idle_delay.setDecimals(1)
        self.sp_idle_delay.setSingleStep(0.5)
        _isd0 = z0.get("idle_switch_delay_s")
        self.sp_idle_delay.setValue(max(0.0, float(_isd0) if _isd0 is not None else 3.0))
        self.sp_idle_delay.setToolTip(
            "**idle 持续多久没怪**才切换平台（秒 ✓）。\n\n"
            "· **0 = 不启用**切换平台逻辑（下面的「idle 去哪」置灰 ✓）；\n"
            "· >0 = idle 持续这么久**确实没怪了**才下前往任务 ✓"
            "（中途有怪 ⇒ 计时清零重来 ✓）。\n\n"
            "⭐ idle 期间**同时**走 idle 回归 foothold（上面的视图+列表）✓ —— 两者不再互斥 ✓。")
        self.sp_idle_delay.valueChanged.connect(self._on_idle_delay_changed)
        form_bottom.addRow("idle持续n秒切换平台", self.sp_idle_delay)

        self.cmb_idle_dst = NoWheelComboBox()
        self.cmb_idle_dst.setMinimumWidth(150)
        self.cmb_idle_dst.addItem("（不前往）", "")
        for n in sorted(str(x) for x in (names or [])):
            if n and n != self._set:                   # 别把自己列成目的地 ✗（同 cmb_dst ✓）
                self.cmb_idle_dst.addItem(n, n)
        _i_dst = self.cmb_idle_dst.findData(_idle_dst_val)
        self.cmb_idle_dst.setCurrentIndex(_i_dst if _i_dst >= 0 else 0)
        self.cmb_idle_dst.setToolTip(
            "idle 持续够「idle持续n秒切换平台」后**前往哪块平台**（集合名 ✓）。\n"
            "· 候选 = 这张图已注册的集合（不含这一项自己 ✓）。\n"
            "· 上面的「idle持续n秒切换平台」= 0 时这格**置灰**（不启用 ✓）。")
        self.cmb_idle_dst.currentIndexChanged.connect(self._on_idle_dst_changed)
        self._idle_dst_widget = QWidget()
        _h_dst = QHBoxLayout(self._idle_dst_widget)
        _h_dst.setContentsMargins(0, 0, 0, 0)
        _h_dst.setSpacing(6)
        _h_dst.addWidget(QLabel("idle 去哪"))
        _h_dst.addWidget(self.cmb_idle_dst, 1)
        form_bottom.addRow(self._idle_dst_widget)

        # ④ 最小战斗时长(s) + ⑤ 最大战斗时长(s) + ⑥ 到点去哪
        self.sp_fight_min = NoWheelDoubleSpinBox()
        self.sp_fight_min.setRange(0.0, 3600.0)
        self.sp_fight_min.setDecimals(1)
        self.sp_fight_min.setSingleStep(5.0)
        self.sp_fight_min.setValue(max(0.0, float(z0.get("fight_min_s") or 0.0)))
        self.sp_fight_min.setToolTip(
            "在这块区域里**至少连着打多久**，才肯去追**别的 foothold 集合**的怪（秒 ✓）。\n\n"
            "· **0 = 不限**（老行为 ✓ —— 一进这块平台，别的平台的怪也照追）；\n"
            "· 填了 ⇒ 在本集合累计战斗**还没到**这个秒数时，**不追别的集合的怪** ✓\n"
            "  （只在本集合里打；到点之后才恢复跨集合追击 ✓）。\n\n"
            "⚠ 计时口径见 `decision/agent.py` 里 `fight_min_s` 字段的说明（和「最大战斗时长」\n"
            "   共用同一把钟：只有寻路会暂停、换区域才归零 ✓）。")
        form_bottom.addRow("最小战斗时长(s)", self.sp_fight_min)

        self.sp_fight = NoWheelDoubleSpinBox()
        self.sp_fight.setRange(0.0, 3600.0)
        self.sp_fight.setDecimals(1)
        self.sp_fight.setSingleStep(5.0)
        self.sp_fight.setValue(max(0.0, float(z0.get("fight_max_s") or 0.0)))
        self.sp_fight.setToolTip(
            "在这块区域里**连着打多久就换地方**（秒 ✓）。\n\n"
            "· **0 = 不限**（老行为 ✓ —— 打到没怪为止）；\n"
            "· 填了 ⇒ 到点就去下面的「到点去哪」（没填就不动 ✓）。\n\n"
            "⚠ 计时口径见 `decision/agent.py` 里那个字段的说明。")
        form_bottom.addRow("最大战斗时长(s)", self.sp_fight)

        self.cmb_dst = NoWheelComboBox()
        self.cmb_dst.setMinimumWidth(150)
        self.cmb_dst.addItem("（不前往）", "")          # data = "" ⇒ 空 = 不做这件事 ✓
        for n in sorted(str(x) for x in (names or [])):
            if n and n != self._set:                   # 别把自己列成目的地 ✗
                self.cmb_dst.addItem(n, n)
        cur = str(z0.get("fight_dst") or "")
        _i = self.cmb_dst.findData(cur)
        self.cmb_dst.setCurrentIndex(_i if _i >= 0 else 0)
        self.cmb_dst.setToolTip(
            "「最大战斗时长」到点之后**去哪块平台**（集合名 ✓）。\n"
            "· **（不前往）** = 到点只停手、不换地方（默认 ✓）；\n"
            "· 候选 = 这张图已注册的集合（不含这一项自己 ✓）。")
        form_bottom.addRow("到点去哪", self.cmb_dst)

        root.addLayout(form_top)
        # ⭐ **视图块夹在两段表单之间**（用户 2026-10-02 ✓ —— "idle 回归 foothold 块放在
        #   idle 模式下方"）：`form_top`（区域集合/CD/idle 模式/idle 去哪）→ `_idle_box`
        #   （视图+列表+按钮）→ `form_bottom`（最小/最大战斗时长/到点去哪）。
        #   · 有视图 ⇒ `_idle_box` 吃满剩余高度（窗口纵向缩放 ✓ 用户 2026-09-28 那条仍成立）；
        #   · 没视图 ⇒ `form_top` 的"idle 回归 foothold"文本框已经够了 ⇒ 直接接 `form_bottom`。
        if getattr(self, "_idle_box", None) is not None:
            root.addLayout(self._idle_box, 1)
            root.addLayout(form_bottom)
        else:
            root.addLayout(form_bottom)
            root.addStretch(1)

        row = QHBoxLayout()
        row.addStretch(1)
        btn_ok = QPushButton("确定")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消")
        theme.unify_ok_cancel(btn_ok, btn_cancel)
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel)
        row.addWidget(btn_ok)
        root.addLayout(row)
        # 拉过的大小/位置按客户端记住（docs/UI规范.md §11 ✓）—— 第一次打开用上面那个 `resize`
        # ⭐ 目的地一变就重算那句"到底会不会生效"（用户 2026-10-04 ✓ 见 `_update_effect_hint`）
        self.cmb_idle_dst.currentIndexChanged.connect(
            lambda _i: self._update_effect_hint())
        self.cmb_dst.currentIndexChanged.connect(lambda _i: self._update_effect_hint())
        self._update_effect_hint()
        theme.bind_window_state(self, "battle_zone")
        # ⭐ 初始化「idle 去哪」的置灰状态（delay=0 ⇒ 置灰 ✓ 建完所有控件后调一次 ✓）
        self._on_idle_delay_changed(self.sp_idle_delay.value())

    def _on_idle_delay_changed(self, val):
        """「idle持续n秒切换平台」改值 ⇒ delay=0 时把「idle 去哪」置灰（用户 2026-10-02 ✓）。

        · val=0 ⇒ 不启用切换平台逻辑 ⇒ idle 去哪 置灰（无法配置 ✓）；
        · val>0 ⇒ 启用 ⇒ idle 去哪 可配置 ✓ + 更新视图高亮 ✓。
        """
        _on = float(val) > 0
        self.cmb_idle_dst.setEnabled(_on)
        self._apply_set_highlight()

    def _on_idle_dst_changed(self):
        """idle 去哪集合下拉切换 ⇒ 更新视图的集合高亮（用户 2026-10-02 ✓）。"""
        self._apply_set_highlight()

    def _apply_set_highlight(self):
        """delay>0 时把"idle 去哪"选的集合喂给视图 `highlight_set` ✓；delay=0 清掉 ✓。"""
        _picker = getattr(self, "_picker", None)
        if _picker is None:
            return
        _delay = float(self.sp_idle_delay.value()) if hasattr(self, "sp_idle_delay") else 0.0
        _hl = str(self.cmb_idle_dst.currentData() or "") if _delay > 0 else ""
        try:
            _picker.highlight_set(_hl)
        except Exception:                               # noqa: BLE001
            pass        # 视图没这个方法（旧版/测试替身）也不许炸弹窗 ✗

    def _make_picker(self):
        """按工厂建那块**只读视图**（建不出来 / 没工厂 ⇒ `None` ✓ 调用方退回文本框 ✓）。"""
        fn = self._picker_factory
        if not callable(fn):
            return None
        w = fn(self._set, str(self.ed_idle.text() or "").strip())
        if w is None:
            return None
        # 点选之后刷新下面那行参数（视图自己已经改好选中态了 ✓）
        w.picked.connect(lambda *_a: self._refresh_idle_label())
        return w

    def _update_effect_hint(self):
        """那一行短话：**这份配置现在到底会不会生效**（唯一出口 ✓ 纯展示 ✓ 不参与判断 ✓）。

        用户 2026-10-04 ✓ 原话："修功能，然后如果不生效动态显示当前的配置有没有用并简短
        例如『该区域未启用，不会生效』"。
        三条（按优先级 ✓，短 ✓）：
          ① 本项没勾「可以战斗」⇒ **整个不生效**（`decision/agent._zone_enabled` 的口径 ✓）；
          ② 本项勾了、但**目的地**（idle 切换平台 / 到点去哪）指向**没启用的项** ⇒ 到时候
             **不会去**（`_zone_dst_blocked` ✓ 2026-10-04 修的那条 ✓）；
          ③ 都没问题 ⇒ 说一句"会生效"（**看得见才算数** ✗ 别让人猜 ✓）。
        ⚠ 目的地是不是"没启用"靠外面推的 `_enabled_map` ✓；**没推 ⇒ 不猜**（② 整条不报 ✓）——
          "图上压根没这一项"（中性平台 ✓）不是"未启用" ✗，别混 ✓。
        """
        lbl = getattr(self, "lbl_effect", None)
        if lbl is None:
            return
        if not self._can_fight:
            lbl.setText("⚠ 该区域未启用（主窗口「路线规划」里「可以战斗」没勾）"
                        "—— 这里的配置不会生效")
            lbl.setStyleSheet("color:#b06000; font-weight:600;")
            return
        _bad = []
        for _cmb, _what in ((getattr(self, "cmb_idle_dst", None), "idle 切换平台"),
                            (getattr(self, "cmb_dst", None), "到点去哪")):
            _dst = str(_cmb.currentData() or "") if _cmb is not None else ""
            if _dst and self._enabled_map.get(_dst) is False:
                _bad.append("「%s」的目的地「%s」未启用" % (_what, _dst))
        if _bad:
            lbl.setText("⚠ " + "；".join(_bad) + " ⇒ 到时候不会去")
            lbl.setStyleSheet("color:#b06000; font-weight:600;")
            return
        lbl.setText("✓ 该区域已启用：这里的配置会生效")
        lbl.setStyleSheet("color:#5f6368;")

    def _refresh_idle_label(self):
        """刷新"当前选中的 foothold"那行（**顺带显示它的参数** —— 用户点选要看的就是它 ✓）。"""
        if self._picker is None or not hasattr(self, "lbl_fh"):
            return
        fid = str(self._picker.current() or "")
        info = self._picker.info(fid) if fid else ""
        self.lbl_fh.setText(("已选：%s" % info) if info
                            else "还没选 —— 点上面一条 foothold ⇒ 再点「添加选中」入池 ✓")
        if hasattr(self, "btn_idle_add"):
            self.btn_idle_add.setEnabled(bool(fid))
        if hasattr(self, "btn_idle_del"):
            self.btn_idle_del.setEnabled(self.lst_idle.currentRow() >= 0)
        if hasattr(self, "btn_idle_clear"):
            self.btn_idle_clear.setEnabled(self.lst_idle.count() > 0)

    def _on_idle_row_clicked(self, item):
        """点列表一行 = 在视图里**看那块**的参数（只切选中，不加不加错 ✗）。"""
        fid = str(item.text()).lstrip("#").strip()
        if self._picker is not None and fid:
            self._picker.set_current(fid)
            self._refresh_idle_label()

    def _on_add_idle(self):
        """把视图里当前选中的 foothold **加进回归点池**（重复添加 = 忽略 ✓）。"""
        if self._picker is None:
            return
        fid = str(self._picker.current() or "").strip()
        if not fid:
            return
        if fid in self.idle_fids():
            return                      # 已在池里 ⇒ 不重复加（列表会越叠越长 ✗）
        self.lst_idle.addItem("#%s" % fid)
        self._refresh_idle_label()

    def _on_del_idle(self):
        """删掉列表里**选中的那行**（没选中 = 不动 ✓）。"""
        row = self.lst_idle.currentRow()
        if row >= 0:
            self.lst_idle.takeItem(row)
        self._refresh_idle_label()

    def _on_clear_idle(self):
        """清空回归点池（= 不做 idle 回归 ✓ 老行为）。"""
        self.lst_idle.clear()
        if self._picker is not None:
            self._picker.set_current("")
        self._refresh_idle_label()

    def idle_fids(self):
        """回归点池（**有视图就读列表 ✓ 否则解析文本框 ✓**；保序去重 ✓）。

        ⚠ 一处取值（`zone()` 也走它 ✓）：别让"列表"和"文本框"两处各取一次 ✗
          —— 那样迟早出现"界面看着加了、存下去是空的"✗。
        """
        if self._picker is not None:
            fids = []
            for row in range(self.lst_idle.count()):
                fid = str(self.lst_idle.item(row).text()).lstrip("#").strip()
                if fid and fid not in fids:
                    fids.append(fid)
            return fids
        # 兜底文本框：空格/逗号分隔都能吃 ✓
        out = []
        for tok in str(self.ed_idle.text() or "").replace(",", " ").split():
            fid = tok.strip()
            if fid and fid not in out:
                out.append(fid)
        return out

    def zone(self):
        """确定之后取回这一项（`dict` ✓ 字段与 `settings.battle_zones` 的项一一对应 ✓）。"""
        return {"set": self._set,
                "cd_s": float(self.sp_cd.value()),
                # ⭐ 两个字段**都存**（不管模式 ✓ —— 模式只控制 UI 显隐，不控制保存）：
                #   决策层靠 `idle_dst_set` **优先**判断（有值 ⇒ 切换平台，不看 footholds ✓）；
                #   这样从"回归 foothold"切到"切换平台"时，旧 footholds 不丢（下次切回来还在 ✓）。
                "idle_footholds": self.idle_fids(),
                "idle_dst_set": str(self.cmb_idle_dst.currentData() or ""),
                "idle_switch_delay_s": float(self.sp_idle_delay.value()),
                "fight_min_s": float(self.sp_fight_min.value()),
                "fight_max_s": float(self.sp_fight.value()),
                "fight_dst": str(self.cmb_dst.currentData() or ""),
                # ⭐ 「可以战斗」**不在这个弹窗里改**（2026-10-01 起在**主窗口每项前面**勾 ✓）——
                #   这里只是**原样带回**（见 `__init__` 里 `self._can_fight` ✓）
                #   ⇒ 编辑别的参数不会顺手把勾选清掉 ✓✓（这一条最容易漏 ✗）。
                "can_fight": bool(self._can_fight)}


class _PadSender:
    """触控板位移的**发送器**：单槽累积 + **独立线程**发（用户 2026-10-05 ✓ 原话：
    "B机F10控制A机鼠标，A机的鼠标移动不连续"）。

    它在整条链里的位置（三段各自解耦 ✓，缺一段就会"一顿一顿" ✗）：
      ① **产生**位移：`decision/input.py::PointerTracker` 独立线程按固定节拍轮询指针 ✓
         （挂在 Qt move 事件上的老写法会让节拍跟着 GUI 卡 ✗）；
      ② **入队**（本类的 `push()`）：跟踪线程直接调 ⇒ 只加法 + 置事件 ✓（不碰网络 ✓）；
      ③ **发送**（本类的线程）：真去 `dinput.mouse_move` ⇒ **要阻塞就阻塞在这里** ✓。

    **为什么第 ③ 步必须独立线程**（和 `decision/manual_input` 2026-10-02 那次同一个病 ✓）：
      `dinput.mouse_move` → `_send_remote` **会阻塞**（socket + 背压 ✓；实测
      `kbd_rtt_ms` 中位 **11 ms**、p95 更大 ✓）。若在**跟踪线程里直接发** ✗ ⇒ 那条约 250 Hz
      的轮询节拍会被往返拖成"什么时候回来什么时候再轮询" ✗ ⇒ 一次攒一大坨 ⇒ A 机照样
      "一段一段一顿一顿" ✓（2026-10-03 只做了"累积 + 合并"、发送仍在主线程 ⇒ 病根只治了一半 ✗）。
      挪到独立线程后：**发送节奏只受链路限制**，而且永远拿最新累积量 ✓。

    ⚠⚠ **单槽累积、不是覆盖** ✗：位移必须守恒 —— 覆盖（丢帧那套）在这里会**丢像素** ✓
      ⇒ `push()` 是 `+=`，`take()` 发完再扣掉 ✓（小数留着 ✓ 慢速滑不丢 ✓）。
    ⚠ `push()` **绝不碰网络**（只加法 + 置事件 ✓）—— 这是"调用方（跟踪线程 / GUI）不被拖住"
      的全部保证 ✓。
    ⚠ 谁调 `push()`：**只有** `_on_pad_tracked`（跟踪线程 ✓）；收尾只有 `shutdown` ✓
      ⇒ 惰性创建才不用加锁 ✓（见 `_on_pad_tracked` 的说明 ✓）。
    """

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._acc = [0.0, 0.0]
        self._evt = threading.Event()
        self._stop = False
        self.n_push = 0          # 入队次数（打点/自检用 ✓）
        self.n_sent = 0          # 真发出去几条 ✓（< n_push 说明合并生效 ✓）
        self._th = threading.Thread(target=self._run, daemon=True, name="pad-send")
        self._th.start()

    def push(self, dx, dy):
        """入队（**立刻返回** ✓）—— GUI 侧只做加法 + 置事件 ✓。"""
        with self._lock:
            self._acc[0] += float(dx)
            self._acc[1] += float(dy)
            self.n_push += 1
        self._evt.set()

    def take(self):
        """取走当前**整像素**累积（小数留着 ✓）—— 给发送线程用，也便于自检 ✓。"""
        with self._lock:
            mx, my = int(self._acc[0]), int(self._acc[1])
            self._acc[0] -= mx
            self._acc[1] -= my
            return mx, my

    def _run(self):
        while not self._stop:
            self._evt.wait(0.5)
            if self._stop:
                return
            mx, my = self.take()
            if not (mx or my):
                self._evt.clear()        # 不足 1px ⇒ 等下一次 push（不空转 ✓）
                continue
            try:
                dinput.mouse_move(mx, my)   # ⚠ 要阻塞就阻塞在**这条线程**里 ✓
                self.n_sent += 1
            except Exception:            # noqa: BLE001 —— 链路坏了别把发送线程弄死 ✗
                pass

    def stop(self):
        """收尾（面板关闭时调 ✓）：置停止 + 唤醒线程，别留一条守护线程空等 ✓。"""
        self._stop = True
        self._evt.set()


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
        self._update_save_hint()

    # ---------------- 界面 ----------------

    def _update_save_hint(self):
        """把"这份参数存到哪儿 / 现在改了会不会存"写进顶部那一行 ✓（纯展示 ✓ 不参与判断 ✓）。

        ⚠⚠ 为什么要有一行（用户 2026-10-04 ✓ 原话："追击起跳需要的冲刺时间，没有保存数据，
          每次重开 gui 都要重新填"）：**决策参数只按项目存** ✓ ⇒ 没打开项目时
          `settings.save()` **什么都不写** ✗ ⇒ 那时候改的参数一重开就回默认 ✓。
          以前界面上**一句都不说** ✗ ⇒ 人自然以为"存了" ⇒ 报的就是"没保存"✓。
        ⚠ 有项目时也写一句（说清**写回哪个项目** ✓）—— "写进谁"这件事本来就该看得见 ✓。
        """
        lbl = getattr(self, "lbl_save_hint", None)
        if lbl is None:
            return
        if self.project is None:
            lbl.setText("⚠ 还没打开项目：这里改的参数不会保存（重开就回默认值）"
                        "—— 先用上面的「打开…」选一个项目，之后改动会立即写回它。")
            lbl.setStyleSheet("color:#b06000; font-weight:600;")
        else:
            lbl.setText("参数按项目保存：%s（改完立即写回它的 project.yaml）"
                        % self.project.root.name)
            lbl.setStyleSheet("color:#5f6368;")

    def mount_goto(self, widget):
        """把「前往平台」那一块（`RoutePanel` 造的）放进本页操控区 ⇒ **幂等** ✓。

        谁调：`MainWindow._bind_cards`（它同时握着两个面板 ✓）。为什么是"搬控件"而不是
        "在这边另造一套"：那块要用 `RoutePanel` 的地形图 / 地形数据 / 地图 id，
        复制一份必然分叉（改一边忘一边 ✗）。重复调用不会插两次 ✓。
        """
        if widget is None or self._goto_mounted is widget:
            return
        widget.setParent(self.goto_holder)
        # 插到**最上面**：这一组里「前往平台」是主内容，手工休息那行在下（2026-09-26 ✓）。
        self._goto_lay.insertWidget(0, widget)
        self._goto_mounted = widget

    def _build(self):
        # 版面分两段：上面「操控」组常驻不滚动，下面 QScrollArea 装其余参数。
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 0)
        outer.setSpacing(6)

        # ⭐⭐ **这份参数存到哪儿** 的一行常显提示（用户 2026-10-04 ✓ 现场："追击起跳需要的
        #   冲刺时间，没有保存数据，每次重开 gui 都要重新填"）。决策参数**只按项目存** ✓
        #   ⇒ **没打开项目时 `settings.save()` 一个字节都不写** ✗（见 `decision/agent.py`
        #   的 `set_save_hook` ✓）⇒ 必须**明说**，别让人以为"改了就存上了" ✗
        #   （同理：显示的是"最近打开那个项目"那份 ⇒ 尤其容易误会成"没保存"✗）。
        #   ⚠ 纯文本，别写 markdown ✗（QLabel 不认 ✓ 同 `live_panel._show_mmap_push` 那条纪律 ✓）。
        self.lbl_save_hint = QLabel("")
        self.lbl_save_hint.setWordWrap(True)
        outer.addWidget(self.lbl_save_hint)

        # ⚠ 这是**唯一一处**"自己 new 滚动区"（`# ui-allow-scroll：<理由>` 是给
        #   `tools/check_ui.py` 认的标记 ✓）：本页要在滚动区**外面**压一条**常驻**的
        #   「操控」栏（开关 + 触控板），而且那条栏的滚轮要**指路**到这个滚动区实例
        #   （`top_bar._wheel_target = scroll`，见下面 ✓）—— 这两件事 `scroll_page`
        #   表达不了（它不给滚动区实例 ✓）。除此之外一律走 `gui.widgets` 里的那三个 ✓。
        scroll = QScrollArea()          # ui-allow-scroll：常驻栏要拿滚动区实例当 _wheel_target
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
        self.btn_auto.setToolTip(
            "开始/停止自动打怪。也可用 F11 快捷键开关（全局热键，在别的程序里也生效）。\n"
            "输入设备是「本地(仅测试)」时，**开启**会自动先弹一次确认 —— 那个模式下\n"
            "按键是打给这台机器自己的，工作台一失焦就会落到别的窗口里。关闭不弹。")
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
        # objectName **必须给**（2026-09-26 用户要求"给操控组加个背景色区分一下"）：
        # 主窗口那份 QSS 里的底色规则是按 `QGroupBox#CtrlGroup` 选的 ⇒ 不给名字就选
        # 不中（表现为"改了没反应"）。为什么不直接写 `QGroupBox` 全局：那会把**所有**
        # 分组框（含下面滚动的参数区、以及本组里嵌套的「前往平台」）一起染色 ✗。
        # 颜色定义在主窗口 QSS 的「分组框」那一段（样式只有一处 ✓）。
        ctrl_grp.setObjectName("CtrlGroup")
        # 组内**纵向两段**（docs/UI规范.md §4：一行只放一组，放不下就另起一行 ✓）：
        #   第一段：触控板 + 右列（左右键 / 灵敏度 / 输入设备 / 状态）—— 原样不动 ✓
        #   第二段：「前往平台」—— 2026-09-26 用户要求从「路线识别 → 地形图」搬来 ✓
        # 为什么放第二段而不是塞进右列：右列是**按触控板高度配的**（4 行 ≈122px ≤ 132px ✓），
        # 硬塞会把那一列的"正好对齐"破坏掉 ✗。
        cv = QVBoxLayout(ctrl_grp)
        cv.setSpacing(8)
        cg = QHBoxLayout()
        cg.setSpacing(10)

        # 左：触控板（TouchPad 固定 210x132）
        self.touchpad = TouchPad()
        cg.addWidget(self.touchpad)

        # ⭐⭐ 触控板位移的**实时出口**（用户 2026-10-05 ✓ 原话："直接2"）：跟踪线程每拍**直接
        #   调它** ⇒ **不经 Qt 事件循环** ✓✓ —— 这正是"GUI 卡不卡都均匀"的全部保证 ✓
        #   （`moved` 信号是排队的 ⇒ 回到 GUI 线程 ⇒ 又会排在重绘/主回路后面 ✗ 那正是老毛病 ✓）。
        #   ⚠ 它会在**跟踪线程**里被调 ⇒ 只许做线程安全的事（见 `_on_pad_tracked` ✓）。
        self.touchpad.set_delta_sink(self._on_pad_tracked)
        # 触控模式（本地按 F10 开 / 关，见 eventFilter）下的点击 / 拖拽 / 滚轮都转发给远程鼠标：
        #   点一下就松 → clicked → 固件原子 CLICK；
        #   按住并滑动 → pressed/released → 固件 PRESSM/RELEASEM（拖窗口、框选）。
        self.touchpad.clicked.connect(dinput.mouse_click)
        self.touchpad.pressed.connect(dinput.mouse_press)
        self.touchpad.released.connect(dinput.mouse_release)
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
            "本地(仅测试)：本机键盘模拟（SendInput）。**键打到这台机器自己**，"
            "不是游戏机 ——\n"
            "  工作台一失焦就会落到别的窗口里，所以这个模式下开自动前会先确认一次。\n"
            "ProMicro(远程)：网络连游戏机的 Pro Micro 硬件键盘；\n"
            "ProMicro(本地)：本机 USB 直连 Pro Micro，无需 relay。")
        self.cmb_device.currentIndexChanged.connect(self._on_input_device)

        # 重置指令通道：卡键 / 按键发不出去时的逃生口。
        # 自动关着也能用 —— 卡键往往正是「发现卡了去关自动」之后才发现的。
        self.btn_reset_link = QPushButton("重置指令通道")
        self.btn_reset_link.setToolTip(
            "按键卡住（角色自己一直走 / 一直攻击）或按键发不出去时点这里。\n"
            "⓪ 先把 ←/→/↑/↓ 四个方向键**各点按一遍**（按下→松开）：补发这一下能把\n"
            "   游戏侧「卡住的方向键」刷成松开（光靠 RELEASEALL 只清得了固件侧 ✗）；\n"
            "① 重连远程通道 —— 连接断开时 relay 会直接往固件写一条 RELEASEALL，\n"
            "   把卡住的键一次松开（这条路不依赖我们的网络还通不通）；\n"
            "② 再补发一轮 RELEASEALL + 所有映射键的 RELEASE；\n"
            "③ 让决策层重同步按键状态（否则它以为键还按着，之后不再补发）。")
        self.btn_reset_link.clicked.connect(self._reset_link)

        # 稳定性自检：长时间开自动之后，看内存 / 句柄 / 线程有没有在涨。
        self.btn_stability = QPushButton("稳定性自检")
        self.btn_stability.setToolTip(
            "长时间开着自动之后点一下：看常驻内存 / 内核句柄数 / 线程数 / Python\n"
            "对象数有没有持续上涨 —— 泄漏不会报错，只会让程序越跑越慢。\n"
            "第一次点是记基线，隔十几分钟再点一次才看得出趋势。\n"
            "只读、毫秒级，开关着自动都能点。")
        self.btn_stability.clicked.connect(self._stability_check)

        dev_row = QHBoxLayout()
        dev_row.setSpacing(6)
        dev_row.addWidget(QLabel("输入设备"))
        dev_row.addWidget(self.cmb_device)
        dev_row.addWidget(self.btn_reset_link)
        dev_row.addWidget(self.btn_stability)
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
        cv.addLayout(cg)

        # ---- 「命令」组（用户 2026-09-26 改名 + 扩组）----
        # 「命令」= 一切"让角色**现在**去做某件事"的入口，两类东西：
        #   · 「前往平台」那一块（选择平台 / 命令前往 / 结束当前寻路）—— 由主窗口
        #     `mount_goto` 从「路线识别」面板搬进来（**控件与逻辑留在那边** ✓）；
        #   · 「手动进入休息 / 手动结束休息 / 休息状态」三件 —— 同一天从「防掉线」组搬来
        #     （它们本来就是命令 ✓，挂在防掉线里想"让角色去歇会儿"时找不到 ✗）。
        self.goto_holder = QGroupBox("命令")
        self._goto_lay = QVBoxLayout(self.goto_holder)
        self._goto_lay.setSpacing(6)
        self._goto_mounted = None
        # 「手动进入/结束休息」：立刻休息一次、或休息中提前拉回来。
        # 左右顺序 = 一次休息的时间顺序（先进后出）。两个按钮都只看实时线程写的
        # 状态（_poll_auto_state 轮询刷新），自己不做判断。
        # 右边紧跟休息状态/倒计时，同一行放省垂直空间（实测三个控件合计约 500px ✓）。
        self.btn_start_rest = QPushButton("手动进入休息")
        self.btn_start_rest.setToolTip(
            "立刻按防掉线的流程休息一次：\n"
            "  进入行为 → 歇完（防掉线里设的休息时长）→ 退出行为 → 继续打怪，\n"
            "  并按防掉线的间隔重新排下一次自动休息。\n"
            "和自动防掉线走**同一条流程**：攻击范围内还有怪时会先等它们清空\n"
            "（状态显示「待休息：等清空攻击范围内的怪」），免得正打着怪突然站住。\n"
            "自动关着、防掉线没开、或已经在休息时不可点。")
        self.btn_start_rest.setEnabled(False)
        self.btn_start_rest.clicked.connect(self._on_start_rest)

        self.btn_end_rest = QPushButton("手动结束休息")
        self.btn_end_rest.setToolTip(
            "休息中点了立刻结束休息：执行「退出」那套行为，然后恢复正常打怪。\n"
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
        # ⚠ 先加这一行，`mount_goto` 再用 `insertWidget(0, …)` 把「前往平台」块插到**上面**
        #   ⇒ 组内顺序 = 先"去哪儿"，再"歇不歇" ✓（挂载是后面才发生的 ✓）。
        self._goto_lay.addLayout(rest_row)
        cv.addWidget(self.goto_holder)

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

        # ⚠ 这里原来是「随机输入延迟」那一对（`sp_delay_min/max` + `_on_input_delay`）——
        #   **2026-09-27 按用户要求整个功能与配置都删掉**（决策层同步删了 `DecisionSettings.
        #   input_delay` 与 `_random_input_delay`）。**不许加回来**：`tools/selftest_decision.py`
        #   里两条反向钉盯着（参数对象上没有这个字段、本面板上没有那两个控件）。

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

        # ⭐⭐ 「玩家位置」（脚底偏移 / 框面积最小占比 / 面积基线·容差）**已搬到
        #   「**设置 → 判定参数页**」**（用户 2026-09-28 ✓ 原话："「玩家位置」也应该在
        #   设置 → 判定参数页签"）。
        #   为什么它不属于**这一页**（决策参数面板）：那几个数既是"**这一拍算不算拿到了
        #   玩家位置**"（脚底偏移改「画面 → 世界」的换算 ✓；面积闸决定这一拍要不要**跳过
        #   定位查询** ✓）、又该**跟项目走**（不同地图的面 y / 人物框大小都不一样 ✓）
        #   ⇒ 两条都指向「**判定参数**」页 ✓。
        #   ⚠ 判据（规范 §4）：**"怎么说算数" + "跟项目走" ⇒ 进设置弹窗的判定参数页**；
        #     **"本机外观"（只是画给人看）⇒ 进 界面 → 辅助线与标记** ✓。
        #   ⛔ **别再搬回来**（2026-09-28 犯过一次 ✗）；后端字段没动
        #     （仍是 `decision.settings.player_*` ⇒ 仍跟着项目存 ✓）。

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
            "实际间隔 = 本值；输出行为序列本身更长时以序列为准。\n"
            "（主循环一帧只推进一步序列，帧率低时序列耗时会按帧向上取整）")
        self.sp_attack_cd.valueChanged.connect(self._on_attack_cd)
        bf.addRow("输出行为CD(ms)", self.sp_attack_cd)

        # ---- 子组「**攻击**」（用户 2026-09-27 要求）----
        # 放进来的东西（他点名的那两批）：**攻击范围那四个距离** + **追击起跳**的全部参数 ✓。
        # 为什么不留在「战斗参数」外面：它们是"打谁 / 打得到谁"这一件事的四个边 + 一个补充
        # 动作 ⇒ 收进子组之后，"攻击"这个词下面一眼看全 ✓（版式照抄下面的 `evade_grp` ✓）。
        attack_grp = QGroupBox("攻击")
        af = QFormLayout(attack_grp)
        af.setLabelAlignment(Qt.AlignLeft)

        # ① **攻击范围 = 矩形**（用户 2026-09-27 定的）：四个距离 = 四条边 ✓
        #   水平：最小攻击距离 ~ 最大攻击距离（前者以内是**盲区**）；
        #   竖直：往上「向上攻击距离」、往下「向下攻击距离」。
        # ⚠ **0 = 该方向不限**（= 以前完全不看 y 的老行为 ✓）⇒ 老项目升级后行为不变 ✓。
        #   判据只有一处：`decision/agent.py::DecisionAgent._in_box` ✓。
        self.sp_attack = self._spin(0, 2000, 80, 0)
        self.sp_attack.setToolTip(
            "怪离角色多近才开始攻击（**只算前方**）。\n"
            "距离从角色中心点量到怪框最近的边，不是从玩家框的边缘起算。\n\n"
            "⚠ 这是**攻击范围矩形**的四条边之一（2026-09-27 起攻击范围是矩形）：\n"
            "  水平 = 「最小攻击距离」~「最大攻击距离」；竖直 = 上下那两个距离。\n"
            "  实机效果：设置 → 外观 → 辅助线与标记里的「攻击范围框」就是它 ✓\n\n"
            "⚠ 配 **0** ⇒ 水平可攻击区为空 ⇒ **不打任何怪、也不画框**（按 0 算，不是无限 ✓）。")
        self.sp_attack.valueChanged.connect(self._on_attack_dist)
        af.addRow("最大攻击距离", self.sp_attack)

        self.sp_min_attack = self._spin(0, 2000, 0, 0)
        self.sp_min_attack.setToolTip(
            "怪贴脸到此距离内 = **攻击盲区**：不算可攻击，并触发规避（跳/后退）。\n"
            "设 0 表示没有盲区（老行为）。\n"
            "口径同最大攻击距离：角色中心点 → 怪框最近的边；竖直同样受上下攻击距离约束。\n"
            "实机效果：设置 → 外观 → 辅助线与标记里的「攻击盲区框」就是它 ✓")
        self.sp_min_attack.valueChanged.connect(self._on_min_attack)
        af.addRow("最小攻击距离", self.sp_min_attack)

        self.sp_attack_up = self._spin(-1, 2000, -1, 0)
        self.sp_attack_up.setToolTip(
            "**向上攻击距离**（像素）：攻击范围矩形从角色中心**往上**能延伸多少。\n\n"
            "取值（用户 2026-09-27 定的口径，别混）：\n"
            "  · **-1（或任何负数）= 上方不限** ⇒ 不限制（= 老行为，以前根本不看 y ✓）；\n"
            "  · **0 = 上方就是 0** ⇒ 上方的怪**一律打不到**（不是「无限」✗）；\n"
            "  · **正数** = 具体距离：怪框离角色中心的竖直距离 ≤ 它才算够得着 ✓。\n\n"
            "什么时候要设它：上下两层平台靠得近时，**不打上面那层的怪**（只打同一层的）✓\n"
            "⚠ 上下**都配 0** ⇒ 整个攻击范围框面积为 0 ⇒ **不打任何怪、也不画那个框** ✓。")
        self.sp_attack_up.valueChanged.connect(self._on_attack_vertical)
        af.addRow("向上攻击距离", self.sp_attack_up)

        self.sp_attack_down = self._spin(-1, 2000, -1, 0)
        self.sp_attack_down.setToolTip(
            "**向下攻击距离**（像素）：攻击范围矩形从角色中心**往下**能延伸多少。\n\n"
            "取值同上（**-1/负数 = 下方不限**、**0 = 下方就是 0（打不到）**、正数 = 具体距离 ✓）。\n\n"
            "⚠ 这里说的是**怪框**离角色中心的竖直距离，和角色自己会不会掉下去无关。")
        self.sp_attack_down.valueChanged.connect(self._on_attack_vertical)
        af.addRow("向下攻击距离", self.sp_attack_down)

        # ⭐⭐ **「扇形角度」**（用户 2026-10-04 ✓ 原话："在「向下攻击距离」参数下面加个参数
        #   「扇形角度」，代表攻击距离矩形的扇形化角度，默认 0。当不是 0 时，攻击距离框的
        #   上下两条边要向上及向下倾斜这个角度"✓）。
        #   **几何**：把矩形上下两条边、以**近端那条竖边**（=「最小攻击距离」处 ✓）为支点
        #   向外倾斜 α ⇒ **远端更宽、近端高度不变** ✓（就是他那张示意图：左边那个窄矩形、
        #   往右张开 ✓）。⚠ **0 = 老行为一字不差** ✓（`tan0 = 0` ✓）。
        #   判据与绘制**同一处**：`decision/agent.py::DecisionAgent._in_box` / `attack_box_poly` ✓。
        self.sp_attack_fan = self._spin(0, 60, 0, 1, 0.5)
        self.sp_attack_fan.setToolTip(
            "**扇形角度**（度）：把攻击范围框的**上下两条边**向外倾斜这么多。\n\n"
            "· **0（默认）= 矩形**（老行为，一字不变 ✓）；\n"
            "· **> 0 = 扇形（梯形）**：以「**最小攻击距离**」那条竖边为支点、远端向外张开 ——\n"
            "  近端高度还是「向上/向下攻击距离」，离角色越远允许的竖直距离越大\n"
            "  （水平距离 d 处 = 该方向距离 + max(0, d − 最小攻击距离) × tanα ✓）。\n\n"
            "为什么要它：有些技能的判定是**扇形**（近处窄、远处宽）—— 用矩形框会在远处\n"
            "把够不着的怪也算进来（或者反过来，把远处能打的漏掉 ✗）。\n\n"
            "⚠ 上限 60°（再大就接近「上下不限」了 ✗）；负数一律按 0 算 ✓。\n"
            "⚠ 「上下都配 0」的老规矩：**α = 0** 时整个框面积为 0 ⇒ 不判也不画；\n"
            "   而 **α > 0 时远端有高度** ⇒ 框是有效的（判定与画面都照做 ✓）。\n"
            "实机效果：设置 → 外观 → 辅助线与标记里的「攻击范围框」会从矩形变成梯形 ✓。")
        self.sp_attack_fan.valueChanged.connect(self._on_attack_fan)
        af.addRow("扇形角度(°)", self.sp_attack_fan)

        # 「**跳跃攻击范围**」（用户 2026-09-27 记的一笔，**本次不实现**）：等跳跃物理做好后，
        # 它表示"怪框落在这个范围内 ⇒ 按跳就能把它带进攻击范围框 ⇒ 触发 attack" ✓。
        # 所以这里**故意不放参数框**（没有物理逻辑可依据 ✗），但设置里已经有它的
        # **显示颜色 + 开关**（「外观 → 辅助线与标记 → 跳跃攻击范围框」✓）。
        lbl_jump_box = QLabel("「跳跃攻击范围」：等跳跃物理做好后再加参数，"
                              "现在设置里只有它的显示颜色")
        lbl_jump_box.setWordWrap(True)
        lbl_jump_box.setStyleSheet("color: #5f6368;")
        af.addRow("", lbl_jump_box)

        # ⭐⭐ **「走不动按跳」**（用户 2026-10-06 ✓ 原话："现在走路走不动会跳一下，能做成开关
        #   放到战斗参数里吗「走不动按跳」，放在「追击起跳」上面"）—— 就摆在**追击起跳上面** ✓。
        #   它管的是**走执行器**（`route.WalkJob` ✓）：判定"走不动了"之后先**单点跳一下**
        #   （跳过小台阶 ✓ 2026-09-28 加 ✓；跳过还走不动才判失败 ✓）。
        #   ⚠ **默认勾上** = **现在正在跑的行为** ✓（要的是"能关掉" ✓ 不是"默认关" ✗）。
        self.ck_walk_hop = QCheckBox("走不动时跳一下")
        self.ck_walk_hop.setChecked(bool(getattr(settings, "walk_stall_jump", True)))
        self.ck_walk_hop.setToolTip(
            "走（`走路`执行器）判定「走不动了」之后，**先单点跳一下**试试能不能跳过小台阶/\n"
            "小障碍；跳完还走不动才判失败（并说清卡在哪）。\n\n"
            "· **勾上（默认）= 现在正在跑的行为** ✓：被小台阶挡住时，跳一下就过去了 ✓；\n"
            "· **取消** ⇒ 走不动**直接判失败**（= 加这个单点跳之前的老行为 ✓）——\n"
            "  适合「一被挡住就让它早点放弃、别在那儿跳」的场合 ✓。\n\n"
            "⚠ 一轮只跳一次（跳完重新计时，再走不动就失败 ⇒ 被真墙挡住时不会无限跳 ✓）。\n"
            "⚠ 跳的那一下**只跳、不横挪**（跳起来挪会挪歪 ✓）；按 60ms 就松 ✓。\n"
            "⚠ 和「追击起跳」（下面那个）**不是一回事** ✗：那个是**战斗中追怪**用的跳，\n"
            "  这个只管**走路被挡住**时的那一下 ✓。")
        self.ck_walk_hop.stateChanged.connect(self._on_walk_hop)
        bf.addRow("走不动按跳", self.ck_walk_hop)

        # ② **追击起跳**（原来散在「战斗参数」里，2026-09-27 一起收进「攻击」子组 ✓）
        self.ck_chase_jump = QCheckBox("启用")

        # 最小切换朝向时间（毫秒）：换向后方向键至少按住这么久
        self.sp_min_turn_hold = self._spin(0, 3000, 0, 0)
        self.sp_min_turn_hold.setToolTip(
            "换朝向时，方向键至少要按住这么久（毫秒）。0 = 不约束。\n"
            "按下去立刻就松开的话，角色的转身动作可能还没做完 —— 这时候打出去\n"
            "的方向是错的。这段按住时间只能被「又换一次朝向」打断。\n"
            "\n"
            "【站桩输出（怪在攻击范围内、停住打）时】\n"
            "默认只点一下方向键（0.15 秒，够转身、不会走位）；\n"
            "但**每 3 次攻击**会补一次「朝目标」的方向键、并按满上面这个时间\n"
            "（就是为了把角色确实转过去 —— 按满会朝怪小走一段，属预期）。\n"
            "退出站桩（怪离开攻击范围 / 停自动等）后，这个「每 3 次」的计数清零。\n"
            "⚠ 正在爬绳梯时**绝不按左右**（在绳上按左右 = 松手掉下来），\n"
            "  那时只用内部朝向判断，不补方向键。")
        self.sp_min_turn_hold.valueChanged.connect(self._on_turn_params)
        bf.addRow("最小切换朝向时间(ms)", self.sp_min_turn_hold)

        # ⭐⭐ 「**站桩补朝向间隔(ms)**」（用户 2026-10-02 ✓ 原话："把『站桩 attack n 次后补朝向』
        #   改成『站桩补朝向间隔(ms)』，**新增在『最小切换朝向时间(ms)』下面**。逻辑是**每站桩
        #   这么久就补**"）—— 站桩那一段里，距上一次补键 ≥ 它就补一次（照旧点一下 `TURN_TAP_S` ✓；
        #   进站桩的**首窗**照旧按满上面那个「最小切换朝向时间」✓ 并当作一次补键 ✓）。
        #   ⚠ **0 = 不补**（只留首窗 ✓）；**按项目存** ✓（`to_dict`/`from_dict` ✓ ⇒ 项目保存 +
        #     保存/加载模板三处一起生效 ✓）。
        self.sp_station_turn_iv = self._spin(
            0, 60000, int(getattr(settings, "station_turn_interval_ms", 1000) or 0), 0, 100)
        self.sp_station_turn_iv.setToolTip(
            "站桩输出时，**每隔这么久补一次「朝目标」的方向键**（毫秒）。\n"
            "补的那一下**按满上面那个「最小切换朝向时间」**（用户 2026-10-02 定 ✓）；\n"
            "⚠ 自动保护：**实际按多长 = min(最小切换朝向时间, 这个间隔)**\n"
            "   —— 否则时长 ≥ 间隔时，方向键会一直按着（角色一边打一边朝怪走）。\n\n"
            "0 = 不补（只留刚进站桩那一下）。\n"
            "默认 1000（≈ 原来「每 3 次攻击补一次」的实际节奏）。\n"
            "信息栏那行会按 0.1 秒精度显示「还有多久补」（例：1.2s后补朝向）。")
        self.sp_station_turn_iv.valueChanged.connect(self._on_turn_params)
        bf.addRow("站桩补朝向间隔(ms)", self.sp_station_turn_iv)

        # 转向后输出延迟（毫秒）：换向后推迟这么久才开始输出
        self.sp_turn_output_delay = self._spin(0, 3000, 0, 0)
        self.sp_turn_output_delay.setToolTip(
            "换朝向后，输出行为要额外推迟这么久（毫秒）才开始。0 = 不延迟。\n"
            "和上面那条配合用：那条保证方向键按住够久，这条保证输出等转身做完。\n"
            "已经在跑的输出序列不会被掐断（半截掐断会留下按着的键），\n"
            "只把「新开一轮输出」往后推。\n"
            "站桩时补朝向键（上一项）也吃它 —— 顺序固定是"
            "「输出 → 转向 → 本项延迟 → 输出」（用户 2026-10-06 定）。\n"
            "改完立刻生效（不用重启、不用重开自动）。")
        self.sp_turn_output_delay.valueChanged.connect(self._on_turn_params)
        bf.addRow("转向后输出延迟(ms)", self.sp_turn_output_delay)

        # ⭐⭐ 「**自动拾取**」（用户 2026-10-06 ✓ 原话："在战斗参数的『转向后输出延迟』**下面**
        #   加一个参数『自动拾取』→ bool 开关，逻辑是**当玩家与掉落物接触时连续点按拾取**"）——
        #   ⚠ 位置就是**这一行**（紧跟那个 spin ✓ 用例量坐标钉着"在它下面" ✓ 别挪 ✗）。
        #   ⚠ 它跟「键盘映射 → 拾取」是一对：那个决定**点哪个键**，这个决定**点不点** ✓。
        self.ck_auto_pickup = QCheckBox("自动拾取")
        self.ck_auto_pickup.setChecked(bool(getattr(settings, "auto_pickup", False)))
        self.ck_auto_pickup.setToolTip(
            "**玩家与掉落物接触时，连续点按「拾取」键**（默认每 0.15 秒一下）。\n\n"
            "· 判据 = 「玩家框」和「掉落物框」**相交**（感知层每帧算好 ⇒ 决策只读那个结论 ✓）；\n"
            "· 点的是「键盘映射」组里那个「**拾取**」键（默认 Z）—— 不对就去那儿改一下；\n"
            "· 没接触 / 没配拾取键 ⇒ **一下都不点** ✓。\n\n"
            "⚠ **默认关**：老项目升上来行为一个字都不变 ✓；\n"
            "⚠ 它**不占**输出CD、也不参与「转向」那套（拾取键不是方向键）⇒ 打架走路时照捡 ✓。\n"
            "⚠ 模型里**要有掉落类**（class 2）：权重没这一类 ⇒ 永远判不出「接触」（框都没有 ✗）；\n"
            "   「界面 → 检测框颜色 → 掉落框颜色」那个勾只影响**看不看得见**、不影响这里 ✓。\n"
            "⚠ 要跟「平台站桩 → 定时拾取掉落」配合用：那套负责**走过去**，这个负责**碰到了就点** ✓。")
        self.ck_auto_pickup.stateChanged.connect(self._on_auto_pickup)
        bf.addRow("自动拾取", self.ck_auto_pickup)

        self.ck_chase_jump.setToolTip(
            "追击起跳：起跳范围内「从无怪变成有怪」的那一拍才跳一次。\n"
            "所以同一只怪一直挂在区间里不会反复跳（它只在刚进来时跳一下）。\n"
            "若攻击范围内还有其他怪则不跳 —— 先打那些，跳会打断输出。\n"
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
        af.addRow("追击起跳", cj_row)

        # 「追击起跳需要的冲刺时间(ms)」（用户 2026-09-26 要求）：放在它**下面一行**。
        # 含义 = `chase` 状态**连续**维持了这么久才准起跳（中途进别的状态就归零）：
        # 刚进追击 / 刚打完 / 刚转身那会儿就跳，常常是"为了跳而跳" —— 人还没冲起来，
        # 跳出去够不着、还把节奏打断。0 = 不额外要求（老行为）。
        # ⚠ `_spin(lo, hi, val, **decimals**, step=None)` —— 第 4 个参数是**小数位**。
        # 这里要的是"毫秒、整数、步进 50" ⇒ `_spin(0, 5000, 0, 0, 50)`；
        # 曾把 50 写在第 4 位 ⇒ decimals=50 ⇒ 框里显示 0.000…0（50 个 0）✗。
        # 单位也不写进框里（UI 规范：单位写在框外）—— 已经在下面那行标签的 "(ms)" 里了。
        self.sp_chase_dash = self._spin(0, 5000, 0, 0, 50)
        self.sp_chase_dash.setToolTip(
            "追击起跳需要的**冲刺时间**（毫秒）：\n"
            "`chase`（追击）状态必须**连续**维持这么久，才会真的起跳；\n"
            "只要中途进了别的状态（攻击 / 站桩 / 规避 / 休息…）计时就**归零**。\n\n"
            "例：填 300 ⇒ 冲了 300ms 以上，进起跳区间那一拍才跳。\n"
            "0 = 不额外要求（老行为：一到区间就跳）。")
        self.sp_chase_dash.valueChanged.connect(self._on_chase_jump)
        af.addRow("追击起跳需要的冲刺时间(ms)", self.sp_chase_dash)

        # ⭐⭐ 「**距离平台边缘多远禁用(px)**」（用户 2026-10-02 ✓ 原话："追击起跳功能加个参数
        #   「距离平台边缘多远禁用(px)」（**不启用时不能配置**），代表如果距离 foothold 集边缘
        #   距离小于等于这个值即使满足条件也不按跳"）。
        #   样式**照「大怪优先」**（用户点名 ✓）：勾选框 + 一个 px 框；**不勾 ⇒ px 框灰掉不可配** ✓
        #   且这条闸**整个不启用** ✓（= 老行为 ✓）。
        self.ck_chase_edge_guard = QCheckBox("距离平台边缘禁用")
        self.ck_chase_edge_guard.setChecked(bool(
            getattr(settings, "chase_jump_edge_guard_enabled", False)))
        self.ck_chase_edge_guard.setToolTip(
            "**起跳贴边防掉**：人到「**跳向那侧**」的平台边缘距离 ≤ 下面这个值时，\n"
            "哪怕起跳条件全部满足也**不按跳**（再跳出去就掉下平台了）。\n"
            "「跳向那侧」= 当前朝向那侧（起跳本来就是朝目标飞）。\n"
            "⚠ 判不出边缘距离时（没定位 / 位置状态没给平台宽度）→ **不禁用**（宁缺勿错）。\n"
            "不勾 = 这条闸不启用，下面那个值也不可配（老行为）。")
        self.ck_chase_edge_guard.stateChanged.connect(self._on_chase_jump)
        self.sp_chase_edge_px = self._spin(
            0, 2000, int(getattr(settings, "chase_jump_edge_guard_px", 100) or 0), 0, 10)
        self.sp_chase_edge_px.setToolTip(
            "阈值（像素）：距「跳向那侧」的平台边缘 ≤ 它 ⇒ 不跳。\n"
            "0 = 等于没开（只禁掉「已经贴在边缘上」那一瞬间）。")
        self.sp_chase_edge_px.valueChanged.connect(self._on_chase_jump)
        eg_row = QHBoxLayout()
        eg_row.setSpacing(4)
        eg_row.addWidget(self.ck_chase_edge_guard)
        eg_row.addWidget(self.sp_chase_edge_px)
        eg_row.addStretch(1)
        af.addRow("距离平台边缘禁用(px)", eg_row)

        # ③ **大怪优先**（用户 2026-10-01 ✓；当天又要求挪进「攻击」子组 ✓）：
        #   锁定目标时优先「明显更高」的大怪 —— 这是"**打谁**"这一件事 ⇒ 收进「攻击」✓。
        #   2026-10-01 再要求：加**总开关**「大怪优先」勾选框 —— 不勾 = 逻辑不启用、
        #   下面「大怪判定倍数」「大怪优先半径」两个参数也**灰掉不可配** ✓。
        self.ck_big_mob = QCheckBox("大怪优先")
        self.ck_big_mob.setChecked(bool(getattr(settings, "big_mob_enabled", True)))
        self.ck_big_mob.setToolTip(
            "**总开关**：勾上才启用「大怪优先」（锁定目标时优先明显更高的大怪 ✓）。\n"
            "不勾 = 逻辑不启用、下面「大怪判定倍数」「大怪优先半径」也**不可配** ✓。")
        af.addRow("大怪优先", self.ck_big_mob)

        self.sp_big_mob_ratio = self._spin(1.0, 5.0, 1.5, 1)
        self.sp_big_mob_ratio.setSingleStep(0.1)
        self.sp_big_mob_ratio.setToolTip(
            "锁定目标时优先「明显更高」的大怪（**相对判定** ✓）。\n"
            "判据：怪框高度 ≥ **最小框均线 × 这个倍数**就算大怪\n"
            "（最小框均线 = 最近 10 秒里「每拍最小的那个怪框高」的平均\n"
            " —— 最小那一头永远站在小怪那边，**大怪多、小怪少**时也认得出 ✓；\n"
            " 单个漏检的小框被均线摊平，不会把基准拖低 ✓）。\n"
            "设为 1.0 = 关闭（人人都是大怪，等于不优先）。")
        af.addRow("大怪判定倍数", self.sp_big_mob_ratio)

        self.sp_big_mob_range_px = self._spin(0, 5000, 600, 0)
        self.sp_big_mob_range_px.setToolTip(
            "大怪优先的**半径**（画面像素 ✓）。\n"
            "只有「画面距离 ≤ 这个值」的大怪才会插队优先，否则仍按距离排\n"
            "（避免为了远处的大怪放弃眼前的小怪 ✓）。\n"
            "设为 0 = 关闭大怪优先。")
        af.addRow("大怪优先半径(px)", self.sp_big_mob_range_px)

        # 三个控件接同一个回调（开关 + 两个参数），并按开关初值启用/禁用参数 ✓
        self.ck_big_mob.stateChanged.connect(self._on_big_mob)
        self.sp_big_mob_ratio.valueChanged.connect(self._on_big_mob)
        self.sp_big_mob_range_px.valueChanged.connect(self._on_big_mob)
        self._apply_big_mob_enabled()

        # 子组挂到「战斗参数」组上（版式同下面的 `evade_grp`：标签留空、整块占一行 ✓）。
        # ⚠ **必须挂**：只建 QGroupBox 不加进来，这个框永远不会显示（Qt 里没有父级的
        #   控件不参与布局 ✗）—— 自检 `t_attack_group` 会当场红 ✓。
        bf.addRow("", attack_grp)

        # ⛔ 「编辑战斗区域」这一行 2026-10-01 搬走了（用户要求：移到「路线识别 →
        #   寻路配置 → 路线规划」✓）—— 战斗区域的每一项都是**地图里的东西**（集合名 /
        #   idle 回归 foothold 编号），只有持有地图 id 的路线识别页能**每次现读**候选集合名。
        #   它原来在这里靠"面板间推送"（开项目那一刻推一次），地形/集合会话中途才导出来时
        #   就是**空候选、用到关工作台** ⇒ "不能编辑战斗区域" ✗。
        #   （弹窗类 `BattleZoneListDialog` / `BattleZoneDialog` 还留在这个文件里，
        #   由 `gui/route_panel.py` 懒加载调用 ✓ —— 它们读写的是 settings，不持有地图 id ✓。）

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

        # 定义输出行为：呼出行为编辑器，配置 attack 状态执行的输出序列（默认一个输出键）
        self.btn_edit_output = QPushButton("定义输出行为")
        self.btn_edit_output.setToolTip("打开行为编辑器，配置输出动作序列（默认一个输出键）")
        self.btn_edit_output.setStyleSheet(self._BTN_EDIT_SEQ)
        self.btn_edit_output.clicked.connect(self._edit_output_seq)
        bf.addRow("", self.btn_edit_output)

        root.addWidget(battle)

        # ---- ⭐⭐ 「目标参数」组（用户 2026-10-06 ✓）----
        #   用户原话："决策参数页签→战斗参数 的「目标切换CD」、「防抖置信度」、「防抖时间」
        #   挪出到**下面**新的分组「目标参数」" ✓ ⇒ 它们仨从「战斗参数」**搬出来**、
        #   在这一组里单开（`root.addWidget(battle)` 之后 = 主区里 battle 的**下面** ✓），
        #   再补上「**目标被攻击CD(ms)**」（新参数 ✓ 同一条链上的东西 ⇒ 摆一起 ✓）。
        #   ⚠ 这三个控件的**名字 / 键 / 默认值 / tooltip / 信号一个字都没改** ✓
        #     （纯版式搬移 ⇒ `_sync_from_settings` 那几处回填、`_on_target_cd` /
        #      `_on_debounce` 两个写回**都不用动** ✓）；存盘仍按**项目**走（同旁边那些 ✓）。
        tgt = QGroupBox("目标参数")
        tf = QFormLayout(tgt)
        tf.setLabelAlignment(Qt.AlignLeft)

        # ① 目标切换 CD [min, max]（毫秒）—— 从「战斗参数」搬来（键/默认值/提示全原样 ✓）
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
        tf.addRow("目标切换CD", cd_row)

        # ②a ⭐⭐ 「攻击目标数量」（用户 2026-10-06 ✓ 原话："在「目标被攻击CD」**上方**增加
        #    配置「攻击目标数量」→ int。代表一次攻击最多只能使几个怪物框进被攻击CD。
        #    **填 0 代表不启用「目标被攻击CD」（配置置灰）**，**填 -1 代表不限制**"✓）。
        #    ⇒ 它是下面那格的**闸 + 名额**：`0` = 不启用（下面那格**置灰** ✓）、
        #      `-1` = 不限制（默认 ✓ 与上一版一字不差 ✓）、`N>0` = **同时最多 N 只怪**在 CD 里 ✓。
        #    ⚠ **默认 -1**（不是 0 ✗）：0 会把"已经调过「目标被攻击CD」的老项目"**默默关掉** ✗。
        self.sp_atk_target_count = self._spin(-1, 50, -1, 0)
        self.sp_atk_target_count.setToolTip(
            "**一次攻击最多让几只怪进「被攻击CD」**（`0` / `-1` / 正数是三种意思）：\n\n"
            "· **0 = 不启用「目标被攻击CD」** —— 下面那一格会**置灰**，整个冷却机制关掉\n"
            "  （不记 CD、不摘怪、预览里也不画灰框 ✓ 跟没有这个功能一样）。\n"
            "· **-1 = 不限制**（默认 ✓ = 下面那格自己的行为，跟以前一样）。\n"
            "· **正数 N = 名额**：同一时间**最多 N 只怪**能待在「目标被攻击CD」里。\n"
            "  名额满了 ⇒ 新怪**不再被框进 CD**（它照旧能被选、照旧挨打 ✓）——\n"
            "  也就是「别一次攻击就把一堆怪全框进去、结果没怪可打」。\n"
            "  有怪冷却到点出表之后，新怪自然就能顶上来 ✓。\n\n"
            "· 只在**攻击范围内触发了 attack** 那一刻才占名额（转身 / 追击本身不算 ✓）。\n"
            "· 按项目存（跟旁边这些参数一样 ✓）。")
        self.sp_atk_target_count.valueChanged.connect(self._on_atk_target_count)
        tf.addRow("攻击目标数量", self.sp_atk_target_count)

        # ②b ⭐⭐ 「目标被攻击CD」（毫秒；用户 2026-10-06 ✓ 原话："代表每只怪使玩家触发进 attack、
        #    触发转朝向后，需要冷却该时间才可再次触发" + 追问定稿："**只有当其在攻击范围内触发了
        #    attack 后才进CD**（我打过这只怪 1 次了）① 选为目标（不会拉你进 attack、不会让你为它
        #    转身、chase），也 ② 不参与『背后怪抢锁 / 回身』"✓）。
        #    `0`（默认）= **不限制** ✓ ⇒ 老项目一字不变 ✓。
        #    ⚠ 上面那格填 0 ⇒ 这一格**置灰**且不生效 ✓（用户原话："配置置灰"✓）——
        #      置灰那把开关在 `_sync_mob_cd_enabled` **一处** ✓（别在别处再 setEnabled ✗）。
        self.sp_mob_atk_cd = self._spin(0, 60000, 0, 0)
        self.sp_mob_atk_cd.setToolTip(
            "**按怪**记的冷却（毫秒）：某只怪在**攻击范围内触发了 attack**（= 我打过它一次）\n"
            "之后，这段时间里它：\n"
            "· **不能被选为目标** —— 不会把我拉进 attack、不会让我为它转身、也不会去 chase 它；\n"
            "· **不参与「背后怪抢锁 / 回身」**。\n"
            "⇒ 效果就是「打过一次 ⇒ 先歇一会儿再打它」（有几只怪时会先打别的）。\n\n"
            "· **0 = 不限制**（默认，= 以前的行为 ✓）；填得越大，同一只怪被再次触发得越慢。\n"
            "· 只在**攻击范围内触发了 attack** 那一刻才进冷却（转身 / 追击本身不算）。\n"
            "· 冷却中的怪在**实时预览里画成灰框**，右下角写着还剩几秒（一位小数）✓\n"
            "  —— 它只是**不被选中**，不是看不见了 ✓。\n"
            "· ⚠ 上面「**攻击目标数量**」填 **0** ⇒ 这一格**置灰、不生效** ✓（那是总开关）；\n"
            "  填正数 ⇒ 只放那么多只怪同时进 CD ✓。\n"
            "· 按项目存（跟旁边这些参数一样 ✓）。")
        self.sp_mob_atk_cd.valueChanged.connect(self._on_mob_atk_cd)
        tf.addRow("目标被攻击CD(ms)", self.sp_mob_atk_cd)

        # ③④ 防抖（怪框消失后保留位置）—— 同样从「战斗参数」搬来（键/默认值/提示全原样 ✓）
        self.sp_debounce_conf = self._spin(0.0, 1.0, 0.5, 2)
        self.sp_debounce_conf.setSingleStep(0.05)
        self.sp_debounce_conf.setToolTip(
            "置信度高于此值的怪，框短暂消失会保留位置，防漏检。\n"
            "保留期间框会按消失前的速度继续走一小段（逐渐收住），\n"
            "不会粘在原地 —— 移动中的怪也跟得上。")
        self.sp_debounce_conf.valueChanged.connect(self._on_debounce)
        tf.addRow("防抖置信度", self.sp_debounce_conf)

        self.sp_debounce_ms = self._spin(0, 10000, 300, 0)
        self.sp_debounce_ms.setToolTip(
            "怪框消失后保留位置的时长。\n"
            "这段时间里框按消失前的速度继续走、每帧收一点，所以是「滑过去」\n"
            "而不是「跳一下」；到期就不再输出。\n"
            "调大的代价：角色可能朝一个已经消失的框走过去（性能面板的\n"
            "chase_ghost 就是这个占比）。")
        self.sp_debounce_ms.valueChanged.connect(self._on_debounce)
        tf.addRow("防抖时间(ms)", self.sp_debounce_ms)

        # ⚠ 这一组的控件**不进 `self.widgets`**（那是 `steps.base` 那套卡片的登记表 ✗
        #   本面板的控件一律按成员名访问、回填走 `_sync_from_settings` ✓ 同上面 battle 里那批 ✓）。
        root.addWidget(tgt)
        # ⚠ 起手就按"攻击目标数量"把「目标被攻击CD」那格的灰/亮摆对一次 ✓（值本身由
        #   `_sync_from_settings` 填 ✓）—— 只加 setEnabled 而没人调 = 没接上 ✗（约定 121 的教训 ✓）。
        self._sync_mob_cd_enabled()

        # ---- 自动喝药组 ----
        pot = QGroupBox("自动喝药")
        pg = QFormLayout(pot)
        pg.setLabelAlignment(Qt.AlignLeft)

        self.btn_hp_bar = QPushButton("框选 HP 条")
        self.btn_hp_bar.setToolTip(
            "在实时画面上框选血条，用于识别当前血量。\n"
            "框选时跟着光标的放大镜是 12×（+/- 可调 4~16 倍）：\n"
            "血条上沿差一两个像素，采样就会吃到背景色。")
        self.btn_hp_bar.clicked.connect(lambda: self._pick_bar("hp"))
        self.lbl_hp_bar = QLabel("未选")
        self.lbl_hp_bar.setStyleSheet("color: #80868b;")
        hp_row = QHBoxLayout()
        hp_row.addWidget(self.btn_hp_bar)
        hp_row.addWidget(self.lbl_hp_bar, 1)
        pg.addRow("HP条", hp_row)

        self.btn_mp_bar = QPushButton("框选 MP 条")
        self.btn_mp_bar.setToolTip(
            "在实时画面上框选蓝条，用于识别当前蓝量。\n"
            "框选时跟着光标的放大镜是 12×（+/- 可调 4~16 倍）：\n"
            "蓝条上沿差一两个像素，采样就会吃到背景色。")
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

        # ⭐⭐ **「血条读空」的红字直接盖在 HP 条上、不占任何行**（用户 2026-10-05 ✓ 第二轮原话：
        #   "不要把红字用额外行写出来，直接覆盖 Hp 红条"）。
        #   ⛔ 它原来是「自动喝药」组里 `pg.addRow("", lbl)` 的**独立一行** ✗ ⇒ 一亮一灭就
        #      **多一行 / 少一行** ✗；而血条**受击会闪**（那几拍采样出来就是空 ⇒ 判据翻来翻去 ✓）
        #      ⇒ 版面**频繁抽动** ✓（用户现场："频繁的让 gui 多一行少一行" ✓）。
        #   ⇒ 现在做成 `pb_hp` 的**子控件**、用 grid 贴在条上居中 ✓（红字压在红条上 = 他要的样子 ✓）；
        #      亮的时候把条自带的 `HP xx%` 让开 ✓（`_on_potions` 里 `setTextVisible` ✓ 不然两段字叠）。
        #   ⚠ 判据与 Agent **同源**：都是"血条读空 = 0" ✓（数据就是 `_on_potions` 收到的那两个数 ✓
        #     —— 实时层识别出来的比例 ✓），界面层不另立规则 ✗。
        self.lbl_hp_off = QLabel("读空：喝药/定时已停", self.pb_hp)
        self.lbl_hp_off.setAlignment(Qt.AlignCenter)
        self.lbl_hp_off.setStyleSheet("color: #d93025; background: transparent;")
        # ⚠ 文案要**短**（实测 sizeHint：这句 170px；"血条读空 ⇒ 喝药 / 定时已停" 要 272px ✗
        #   ⇒ 条一窄就被裁 ✓）。
        # ⚠⚠ `Ignored` 策略不能省 ✗：QLabel 的 minimumSizeHint 就是文字宽 ⇒ 不压它的话
        #   **条的最小宽会被这段红字顶大** ⇒ 面板那一列跟着变宽 ✗（= 版面还是在动 ✓，只不过从
        #   "多一行"变成了"变宽" ✗）。`Ignored` = "最小 0、给我多少用多少" ✓ ⇒ 条多宽它多宽 ✓。
        self.lbl_hp_off.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_hp_off.setMinimumSize(0, 0)
        self.lbl_hp_off.setToolTip(
            "已停用：血条读空（角色死亡 / 复活界面）。\n"
            "自动喝药与自定义定时行为都会停；血条一读出数就自动恢复。")
        _ovl = QGridLayout(self.pb_hp)
        _ovl.setContentsMargins(0, 0, 0, 0)
        _ovl.addWidget(self.lbl_hp_off, 0, 0)
        _ovl.setColumnStretch(0, 1)      # 撑满整条（不然只拿 sizeHint，字会被挤在左边 ✓）
        _ovl.setRowStretch(0, 1)
        self.lbl_hp_off.setVisible(False)

        self.pb_mp = QProgressBar()
        self.pb_mp.setRange(0, 100)
        self.pb_mp.setValue(0)
        self.pb_mp.setFormat("MP %p%")
        pg.addRow("MP当前", self.pb_mp)
        self._lbl_mp_cur = pg.labelForField(self.pb_mp)

        # ⚠ 「自动喂宠」整块**已移除**（用户 2026-09-26）。
        #    他要的那个效果改用「自定义定时行为」实现 ✓ —— 按键那层的「喂宠」键**保留** ✓，
        #    在那里配一条定时行为（间隔 + 喂宠键）就是原来那个功能 ✓，而且能自己调序列。
        #    别把这一块又加回来 ✗。

        self._pot_form = pg

        root.addWidget(pot)

        # ---- 策略参数组 ----
        strategy_grp = QGroupBox("策略参数")
        sf = QFormLayout(strategy_grp)
        sf.setLabelAlignment(Qt.AlignLeft)

        # 策略类型
        #: 三种策略的**中文名**（只此一处 ✓ 标签 / 摘要 / "还没启用"那句都读它 ✗ 别各写一份）
        #: ⚠ 与 `decision/agent.py` 的 `strategy` 取值**必须字字对应**（用例钉着 ✓）。
        self.STRATEGY_NAMES = {"patrol": "平地巡逻",
                               "sweep": "扫平台",
                               "platform": "平台站桩",
                               # ⭐ 「多点巡逻」（用户 2026-10-06 ✓）：**依次循环**走点位 ✓
                               "multi": "多点巡逻"}

        # ⭐ 「**当前策略类型**」放在策略参数组**最上方**（用户 2026-10-05 ✓ 原话："在策略参数组
        #   最上方加「当前策略类型」，右下角加「启用」"✓）—— 它说的是**正在生效**那一个，
        #   与下面下拉选中的那个**可能不同**（选了还没点「启用」✓）。
        self.lbl_strategy_cur = QLabel("")
        self.lbl_strategy_cur.setWordWrap(True)
        sf.addRow("当前策略类型", self.lbl_strategy_cur)

        # 策略类型（**选择**）—— ⚠ 换它**不生效**（只切下面的参数预览 ✓）；要生效点右下角「启用」✓。
        self.cmb_strategy = NoWheelComboBox()
        self.cmb_strategy.addItem("平地巡逻", "patrol")
        self.cmb_strategy.addItem("扫平台", "sweep")
        # ⛔⛔ **这一行上一轮漏了**（用户 2026-10-05 报"真机策略类型选不了站桩"✗）——
        #   当时只把下面那四行控件加上了、忘了把第三项塞进下拉 ⇒ 下拉里**根本没有**它 ✗。
        #   教训：**用例要"真调一次"控件**（我那条是直接改 `settings.strategy` ⇒ 抓不到 ✗
        #   见 `t_platform_panel_widgets` ✓ 现在改成走下拉 ✓）。
        self.cmb_strategy.addItem("平台站桩", "platform")
        # ⭐ 「多点巡逻」（用户 2026-10-06 ✓）—— ⚠ 上一轮**漏过一行**（"平台站桩"那次 ✓ 见下面那段
        #   教训注释 ✓）：**加了策略就必须同时塞进这个下拉** ✗，否则"面板选不了"✗
        #   （用例 `t_platform_panel_widgets` 现在是**走下拉**钉的 ✓ 抓得到 ✓）。
        self.cmb_strategy.addItem("多点巡逻", "multi")
        self.cmb_strategy.currentIndexChanged.connect(self._on_strategy)
        self.cmb_strategy.setToolTip(
            "平地巡逻：锁定全部怪，就近优先。\n"
            "扫平台：优先朝向方向，背后一定距离内的怪按就近锁定；\n"
            "到集合边缘附近（「距离平台边缘回头」）或当前朝向没怪持续\n"
            "一段时间就换向。\n"
            "平台站桩：站在「站桩地点」打（够不着的怪只在站点范围内水平逼近）。\n"
            "多点巡逻：在多个「点位」之间**依次循环**走，**到了直接去下一个**；\n"
            "          路上/到点遇怪照打（攻击范围内有怪优先 attack ✓）。\n"
            "⚠ 这里只是**选择**：换它只切下面的参数预览，要生效请点本组右下角的「启用」✓。")
        sf.addRow("策略类型", self.cmb_strategy)

        self.lbl_strategy_note = QLabel("")
        self.lbl_strategy_note.setStyleSheet("color: #80868b;")
        self.lbl_strategy_note.setWordWrap(True)
        sf.addRow("", self.lbl_strategy_note)

        # 换朝向延迟（仅「扫平台」策略用）：当前朝向没怪持续此时间才换
        # ⭐⭐ **这一行的"标签"就是勾选框**（用户 2026-10-09 ✓ 原话："在「换朝向延迟」、
        #   「距离平台边缘回头」**前面加勾选**"✓）⇒ 勾选框就落在数字格**前面** ✓。
        self.ck_turn_cd = QCheckBox("换朝向延迟(ms)")
        self.ck_turn_cd.toggled.connect(self._on_turn_cd_on)
        self.ck_turn_cd.setToolTip(
            "勾上 ⇒ 「当前朝向没怪持续这么多毫秒就换向」这条生效 ✓；\n"
            "不勾 ⇒ **这条关掉** ✗（不会因为前方暂时没怪就掉头 ✓）。\n"
            "⚠ 两条都不勾 ⇒ 按**内置 100px** 走到平台边缘回头（见下一行说明 ✓）。")
        self.sp_turn_cd = self._spin(0, 10000, 1000, 0)
        self.sp_turn_cd.valueChanged.connect(self._on_turn_cd)
        self.sp_turn_cd.setToolTip(
            "当前朝向没怪持续此时间（毫秒）才换朝向。\n"
            "防止某帧漏检/抖动导致频繁换向。\n"
            "⚠ 前面那个勾**不勾**时这一格不生效（会置灰 ✓）。")
        sf.addRow(self.ck_turn_cd, self.sp_turn_cd)
        self._lbl_turn_cd = self.ck_turn_cd        # ⚠ 老名字继续可用（显隐照旧按它走 ✓）

        # 背后锁定距离（仅「扫平台」策略用）
        self.sp_back_range = self._spin(0, 2000, 100, 0)
        self.sp_back_range.valueChanged.connect(self._on_back_range)
        self.sp_back_range.setToolTip(
            "允许锁定背后此距离（像素）内的怪。\n"
            "扫平台优先朝向方向，但背后很近的怪也会被就近锁定。")
        sf.addRow("背后锁定距离", self.sp_back_range)
        self._lbl_back_range = sf.labelForField(self.sp_back_range)

        # 距离平台边缘多远回头（仅「扫平台」策略用，2026-09-29 新增 ✓）
        # ⭐⭐ 同上一行：**勾选框就是这一行的标签**（用户 2026-10-09 ✓ 原话同上 ✓）。
        self.ck_edge_turn = QCheckBox("距离平台边缘回头(px)")
        self.ck_edge_turn.toggled.connect(self._on_edge_turn_on)
        self.ck_edge_turn.setToolTip(
            "勾上 ⇒ 「离当前 foothold 集合边缘 ≤ 这么多像素就回头」这条生效 ✓；\n"
            "不勾 ⇒ **这条关掉** ✗（哪怕一直走到边缘也不按它转）。\n"
            "⚠ 两条都不勾 ⇒ 按**内置 100px** 走到平台边缘回头（用户 2026-10-09 定的默认 ✓\n"
            "   这时**下面那格填多少都不看** ✓）。")
        self.sp_edge_turn = self._spin(0, 2000, 100, 0)
        self.sp_edge_turn.valueChanged.connect(self._on_edge_turn)
        self.sp_edge_turn.setToolTip(
            "扫平台：距离当前 foothold 集合边缘 ≤ 此距离（像素）就回头，\n"
            "即使该方向有怪也转（旧口径会被赖在另一头不走的怪一直拽向边缘）。\n"
            "判不出当前集合（平台没圈集合 / 定位缺）时这条不生效，\n"
            "退回「换朝向延迟」的老逻辑。\n"
            "⚠ 前面那个勾不勾、以及两个都不勾时的兜底（100px），见上一行说明 ✓。")
        sf.addRow(self.ck_edge_turn, self.sp_edge_turn)
        self._lbl_edge_turn = self.ck_edge_turn    # ⚠ 老名字继续可用（显隐照旧按它走 ✓）

        # ⭐ 两条都不勾时的**兜底说明**（用户 2026-10-09 ✓："默认走到平台边缘 100px 回头"✓）
        #   ⚠ 纯文本，别写 markdown ✗（QLabel 不认 ✓ 同本页那条纪律 ✓）
        self.lbl_sweep_hint = QLabel("")
        self.lbl_sweep_hint.setStyleSheet("color: #80868b;")
        self.lbl_sweep_hint.setWordWrap(True)
        sf.addRow("", self.lbl_sweep_hint)

        # ---- 「平台站桩」专属（用户 2026-10-05 ✓）----
        # ⚠ 这几行**只在策略 = 「平台站桩」时显示**（见 `_refresh_strategy_ui` 的**三态** ✓）
        # ① 「站桩地点」= 一个**地图元素**（集合 / 单 foothold / 绳梯 / 传送门 ✓），
        #    由「地区选择」通用弹窗（`gui/element_picker.py`）选 —— **单选** ✓。
        #    ⚠ 本面板**不持有地图 id** ⇒ 弹窗由**工厂**推过来（同 `set_zone_sets` 那套 ✓）。
        self.btn_station_spot = QPushButton("选择站桩地点…")
        self.btn_station_spot.setStyleSheet(theme.ENTRY_BTN_QSS)
        self.btn_station_spot.clicked.connect(self._on_station_spot_pick)
        self.btn_station_spot.setToolTip(
            "选一个**地图元素**当地点（集合 / 单个 foothold / 绳梯 / 传送门）✓。\n\n"
            "· 人不在那儿 ⇒ 先下「前往」走过去（**可跨层** ✓）；\n"
            "· 到位之后**就在那儿打**：够不着的怪只在站点 x 范围内水平逼近、到边缘就停\n"
            "  （不出平台、不跨层 ✓）；\n"
            "· 完全没怪 ⇒ 原地站住（不回归、不换平台 ✓）；\n"
            "· **不选** = 这个策略什么都不做（站着不动 ✓，**不是**退回平地巡逻 ✗）。")
        sf.addRow("站桩地点", self.btn_station_spot)
        _r_spot = sf.labelForField(self.btn_station_spot)
        self.lbl_station_spot = QLabel("")
        self.lbl_station_spot.setWordWrap(True)
        sf.addRow("", self.lbl_station_spot)
        _r_spot2 = sf.labelForField(self.lbl_station_spot)

        # ② 「定时拾取掉落」（用户 2026-10-05 ✓）：开关 ⇒ 时间区间 ⇒ 掉落地区；
        #    **不勾 ⇒ 后面三个参数置灰不可配置** ✓（用户原话 ✓）。
        self.ck_pickup = QCheckBox("定时拾取掉落")
        self.ck_pickup.setToolTip(
            "勾上之后：**到站桩地点**那一刻起随机一个倒计时（下面那段时间区间 ✓），\n"
            "到点后按「拾取掉落地区」的顺序用寻路**依次全部走到**，走完**回站桩地点**、\n"
            "再重新抽一个（循环 ✓）。\n\n"
            "· 到点时攻击范围内还有怪 ⇒ **先打**（推迟到没怪、没任务那一刻才出发 ✓）；\n"
            "· **不勾** = 老行为（不拾取 ✓），下面三个参数置灰。")
        self.ck_pickup.stateChanged.connect(self._on_pickup_enabled)
        sf.addRow("", self.ck_pickup)
        _r_ck = sf.labelForField(self.ck_pickup)

        self.sp_pickup_min = self._spin(0.0, 3600, 0, 1, 0.5)
        self.sp_pickup_max = self._spin(0.0, 3600, 0, 1, 0.5)
        self.sp_pickup_min.valueChanged.connect(self._on_pickup_time)
        self.sp_pickup_max.valueChanged.connect(self._on_pickup_time)
        self.sp_pickup_min.setToolTip("随机倒计时的**下限（秒）**；0~0 = 关。")
        self.sp_pickup_max.setToolTip(
            "随机倒计时的**上限（秒）**。\n"
            "⚠ **上限 <= 0 ⇒ 整组不生效**（用户 2026-10-05：「配<=0不生效」✓）。")
        _pk = QWidget()
        _pk_row = QHBoxLayout(_pk)
        _pk_row.setContentsMargins(0, 0, 0, 0)
        _pk_row.addWidget(self.sp_pickup_min)
        _pk_row.addWidget(QLabel("~"))
        _pk_row.addWidget(self.sp_pickup_max)
        sf.addRow("拾取掉落时间(s)", _pk)
        _r_time = sf.labelForField(_pk)

        self.btn_pickup_spots = QPushButton("选择掉落地区…")
        self.btn_pickup_spots.setStyleSheet(theme.ENTRY_BTN_QSS)
        self.btn_pickup_spots.clicked.connect(self._on_pickup_spots_pick)
        self.btn_pickup_spots.setToolTip(
            "选**多个地点**（可以选好几条 foothold / 集合 ✓）：到点后**按弹窗里列出的顺序**\n"
            "（同类型按 x 从左到右 ✓）用寻路**依次全部走到**，走完算一轮 ✓。\n"
            "⚠ 一个地点都解析不出可走的集合 ⇒ 这一轮**不出发**（log 里留一条 `pickup_abort` ✓）。")
        sf.addRow("拾取掉落地区", self.btn_pickup_spots)
        _r_ps = sf.labelForField(self.btn_pickup_spots)
        self.lbl_pickup_spots = QLabel("")
        self.lbl_pickup_spots.setWordWrap(True)
        sf.addRow("", self.lbl_pickup_spots)
        _r_ps2 = sf.labelForField(self.lbl_pickup_spots)

        # ---- ⭐ 「多点巡逻」专属（用户 2026-10-06 ✓）----
        # 「点位」= **多个地点**（同一个「地区选择」通用弹窗 ✓ **多选** ✓ 与「拾取掉落地区」
        #   完全同款 ✓）；**顺序 = 依次走的顺序**（弹窗里列出的顺序 ✓）。
        # ⚠ 少于 2 个 ⇒ **不许启用**（用户第 4 条：**爆红字** ✓ 见 `_apply_strategy` 的拦截 ✓）。
        self.btn_multi_spots = QPushButton("选择点位…")
        self.btn_multi_spots.setStyleSheet(theme.ENTRY_BTN_QSS)
        self.btn_multi_spots.clicked.connect(self._on_multi_spots_pick)
        self.btn_multi_spots.setToolTip(
            "选**多个地点**当巡逻点位（集合 / 单条 foothold / 绳梯 / 传送门 ✓）。\n\n"
            "· **顺序 = 依次走的顺序**（同弹窗里列出的顺序 ✓）；\n"
            "· 启用后角色**依次循环**走过去（走完最后一个 ⇒ 回到第一个 ✓）；\n"
            "· ⭐ **到了就直接去下一个**（不在那儿停留等条件 ✓）；\n"
            "· 路上 / 到点时攻击范围内有怪 ⇒ **照打**（那是既有机制 ✓），打完接着走 ✓；\n"
            "· ⚠ **至少选 2 个**才能启用这个策略（只选 1 个 = 站着不动 ✓）。")
        sf.addRow("点位", self.btn_multi_spots)
        _r_ms = sf.labelForField(self.btn_multi_spots)
        self.lbl_multi_spots = QLabel("")
        self.lbl_multi_spots.setWordWrap(True)
        sf.addRow("", self.lbl_multi_spots)
        _r_ms2 = sf.labelForField(self.lbl_multi_spots)

        #: 「平台站桩」那几行的 `(控件, 左边那格标签)` —— 显隐**一处收口** ✓
        #: （`QFormLayout` 里"标题"和"控件"是**两样东西** ⇒ 必须成对显隐 ✗ 只藏一个会留半行 ✓）
        self._station_rows = [(self.btn_station_spot, _r_spot),
                              (self.lbl_station_spot, _r_spot2),
                              (self.ck_pickup, _r_ck),
                              (_pk, _r_time),
                              (self.btn_pickup_spots, _r_ps),
                              (self.lbl_pickup_spots, _r_ps2)]
        #: 「多点巡逻」那两行 —— 同上，成对显隐 ✓（只在**选中**「多点巡逻」时出现 ✓）
        self._multi_rows = [(self.btn_multi_spots, _r_ms),
                            (self.lbl_multi_spots, _r_ms2)]
        #: 「地区选择」弹窗的工厂（`fn(parent, multi, init) -> [地点] | None` ✓
        #:   —— 由主窗口从路线识别面板推过来 ✓ 见 `set_element_picker_factory` ✓）
        self._element_factory = None

        # ⭐ 右上那一行是「当前策略类型」、右下角这一个「**启用**」（用户 2026-10-05 ✓
        #   原话："在策略参数组最上方加「当前策略类型」，右下角加「启用」"✓）。
        #   为什么要有它：换策略**不是**改一个数字那么轻（它会换掉整套行为 ✓）⇒ 让"选"
        #   和"生效"分开，用户可以先看参数、配好站桩地点再按下去 ✓。
        _ap = QHBoxLayout()
        _ap.addStretch(1)
        self.btn_strategy_apply = QPushButton("启用")
        self.btn_strategy_apply.clicked.connect(self._apply_strategy)
        self.btn_strategy_apply.setToolTip(
            "把上面「策略类型」里**选中的**那个真正启用（写进项目参数 ✓）。\n\n"
            "· 换策略只改「这一拍之后怎么打」，**不会**丢别的配置 ✓；\n"
            "· 「当前策略类型」那行显示的是**正在生效**的那一个 —— 与上面选的不一样时，\n"
            "  下面那些参数只是**预览**（还没生效 ✓），这个按钮这时才是可点的 ✓。")
        _ap.addWidget(self.btn_strategy_apply)
        sf.addRow("", _ap)

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
        self._timer_cd_state = {}        # {name: 上次是不是"已暂停"}（省掉每秒重复设样式）
        self._timer_pause_btns = {}      # {name: 暂停/继续 QPushButton}
        self._timer_list = QVBoxLayout()
        self._timer_list.setSpacing(4)
        tv.addLayout(self._timer_list)
        # ⚠⚠ **这里原来有一行红字**（"已暂停：血条读空…" ✓ `tv.addWidget(lbl)` ✗）——
        #   用户 2026-10-05 第二轮要求**不再用额外行** ✓（原话："不要把红字用额外行写出来，直接
        #   覆盖 Hp 红条"）：血条受击会闪 ⇒ 那行一亮一灭 ⇒ 版面**频繁多一行少一行** ✗。
        #   ⇒ 显示统一挪到 **HP 条上**那一个（`self.lbl_hp_off` ✓ 不占行 ✓ 文案里就写着"定时已停" ✓），
        #     这里**别再往回加** ✗。⛔ 注意：暂停本身**照旧有效** ✓（闸在 `agent._custom_timers` ✓
        #     与 `agent.timers_suspended()` ✓）—— 本条只是"怎么给人看"改了位置 ✓。
        timer_grp.setToolTip("血条读空（角色死亡 / 复活界面）期间整组暂停；血条一读出数就恢复。")
        btn_add_timer = QPushButton("＋ 添加行为")
        btn_add_timer.setToolTip("新增一个定时执行的按键行为（名字 + 序列 + 间隔）。")
        btn_add_timer.clicked.connect(self._add_timer)
        tv.addWidget(btn_add_timer)
        root.addWidget(timer_grp)

        # ---- 防掉线组 ----
        afk = QGroupBox("防掉线")
        self.afk = afk      # ⭐ 存成员引用：「挂机保护」页要把它 **re-parent** 过去 ✓（2026-09-29）
        af = QFormLayout(afk)

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

        # **休息时长**（2026-09-26 用户要求）：从「隐身休息」子组**上移到通用层**，
        # 紧跟触发时间之后 —— 它是**每个防掉线行为类型**通用的参数
        # （隐身休息 / 定点休息… 都是"歇这么久"）。上移后它不再跟着类型显隐 ✓。
        self.sp_rest_min = self._spin(0.1, 600, 10, 1, 0.1)
        self.sp_rest_max = self._spin(0.1, 600, 20, 1, 0.1)
        self.sp_rest_min.setToolTip("**本次休息**随机 N~M 分钟（支持小数，如 7.5）。\n"
                                    "所有防掉线行为类型通用。")
        self.sp_rest_max.setToolTip("**本次休息**随机 N~M 分钟（支持小数，如 7.5）。\n"
                                    "所有防掉线行为类型通用。")
        self.sp_rest_min.valueChanged.connect(self._on_afk_rest_time)
        self.sp_rest_max.valueChanged.connect(self._on_afk_rest_time)
        rest_iv_row = QHBoxLayout()
        rest_iv_row.addWidget(self.sp_rest_min)
        rest_iv_row.addWidget(QLabel("~"))
        rest_iv_row.addWidget(self.sp_rest_max)
        af.addRow("休息时长(min)", rest_iv_row)

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

        # ⚠ 「休息时长」**已上移到通用层**（触发时间下面，2026-09-26 用户要求）：
        # 它是**每个防掉线行为类型**通用的参数，不再是「隐身休息」专属 ⇒ 别加回这个子组 ✗。

        # 「被打断重试」：休息期间**触发了自动补血**就算被打断 —— 补血说明没兜住
        # （隐身到期 / 被范围技能扫到 / 有东西在打我们），继续歇着等于等着挨打。
        # ⚠ 两种休息类型共用这一个开关**和**下面那个秒数（2026-09-26 用户确认）：
        #    只是"算不算被打断"的时间窗不一样，见 `agent.REST_STATE_SPEC` 的 interrupt 列。
        # ⚠⚠ 它们**必须挂在通用层**（`af`），不能挂进「隐身休息」子组（`hf`）——
        #    挂错了的话：选「定点休息」时整个隐身子组被隐藏 ⇒ **这两个参数也跟着消失** ✗
        #    （用户 2026-09-26 就是这么问的："我怎么没在定点休息组里看到配置？"）。
        #    ⚠ 但**代码顺序不等于显示顺序**：`af.addRow` 的调用顺序才是 ⇒ 下面这两个
        #    `af.addRow` 特意放在**两个子组之前**（那两处各有一行注释标着）。
        self.ck_afk_retry = QCheckBox("被打断重试")
        self.ck_afk_retry.setToolTip(
            "休息期间**触发了自动补血**就算被打断：立刻收工回战斗，并把下次休息提前到\n"
            "下面设的秒数之后（不再是随机的那 N~M 分钟）。\n\n"
            "为什么补血算被打断：它说明没兜住 —— 隐身到期、被范围技能扫到，或者有东西\n"
            "在打我们。这时候继续歇着只是等着挨打。\n\n"
            "**从哪一刻开算**：\n"
            "  · 隐身休息：进入隐身之后到退出隐身之前；\n"
            "  · 定点休息：**从开始前往指定地点那一刻就开算**（到达后行为、正歇着也算）\n"
            "    —— 去休息点路上被打断，这次就不作数，等下面那个秒数后重来。\n"
            "不勾选时：补血照常发生，但休息不被打断。")
        self.ck_afk_retry.stateChanged.connect(self._on_afk_retry)

        self.sp_afk_retry = self._spin(1, 3600, 60, 0, 1)
        self.sp_afk_retry.setToolTip(
            "被打断后隔多少秒**再试一次**休息（1~3600 秒）—— 两种休息类型都适用。\n"
            "设小一点 = 被打了就很快再试着休息（容易反复被打断）；\n"
            "设大一点 = 干脆先打一会儿再说。")
        self.sp_afk_retry.valueChanged.connect(self._on_afk_retry_sec)

        # ---- **通用层**：两种休息类型共用的参数，放子组**外面** ⇒ 换类型也看得见 ✓ ----
        # （见上面那段注释：挂进「隐身休息」子组的话，选「定点休息」时会一起消失 ✗）
        af.addRow("", self.ck_afk_retry)
        af.addRow("打断后重试(s)", self.sp_afk_retry)

        af.addRow(self._afk_hidden)

        # 定点休息子组（用户 2026-09-26 要求）：走到指定集合 → 到达后行为 → 歇**通用休息时长**
        # → 「结束后前往」填了就先走过去 → 回战斗。
        # ⚠ 行进流程（用路径解析 + 多步执行）记在 docs/开发计划.md 的 P2，还没接完 ——
        #    所以这个子组的 tooltip 里写明了现状，别让人以为选了就已经能走 ✗。
        self._afk_spot = QGroupBox("定点休息")
        sf = QFormLayout(self._afk_spot)
        sf.setLabelAlignment(Qt.AlignLeft)

        self.cmb_spot_set = NoWheelComboBox()
        self.cmb_spot_set.setToolTip(
            "休息要去的地点：当前地图**已注册的 foothold 集合**。\n\n"
            "⚠ 现状：这一类型的**行进流程还在做**（见 docs/开发计划.md P2）——\n"
            "现在它会像隐身休息那样原地歇，还不会自己走过去。")
        self.cmb_spot_set.currentIndexChanged.connect(self._on_spot_changed)
        sf.addRow("指定地点", self.cmb_spot_set)

        self.btn_spot_seq = QPushButton("编辑到达后行为")
        self.btn_spot_seq.setToolTip("走到指定地点之后执行的动作序列（例如坐下、用道具）。")
        self.btn_spot_seq.setStyleSheet(self._BTN_EDIT_SEQ)
        self.btn_spot_seq.clicked.connect(lambda: self._edit_afk_seq("spot"))
        sf.addRow("到达后行为", self.btn_spot_seq)

        # ⭐⭐ **「休息过程中循环行为」**（用户 2026-09-28 要求 ✓ 原话："定点休息类型，到达后行为
        #   按钮下面加个配置，布局为：勾选框「休息过程中循环行为」；勾选后，下方缩进出现参数：
        #   「循环行为编辑」→行为编辑器按钮 / 循环时间(s) A ~ B"）。
        #   ⚠ 位置就是"到达后行为的**下面**"（用户点名的 ✓）。
        self.ck_spot_loop = QCheckBox("休息过程中循环行为")
        self.ck_spot_loop.setToolTip(
            "勾上之后，**休息中**（已经到地方、正在歇着的那段）会按下面的间隔**反复**执行"
            "「循环行为」。\n\n"
            "· 与上面的「到达后行为」**互不影响**：那条是**一次性**的（到了先做一遍 ✓），"
            "这条是**循环**的 ✓；\n"
            "· **不勾** = 老行为一字不变 ✓。")
        self.ck_spot_loop.stateChanged.connect(self._on_spot_loop_toggle)
        sf.addRow("", self.ck_spot_loop)     # 空标签 ⇒ 勾选框单独占一行 ✓

        # 勾选后出现的参数（**下方缩进** ✓ 见 `_refresh_spot_loop_ui`）
        self._spot_loop_box = QWidget()
        _lb = QFormLayout(self._spot_loop_box)
        _lb.setContentsMargins(24, 0, 0, 0)  # ⭐ 缩进（"下方缩进出现参数" ✓）
        _lb.setSpacing(6)
        # ⭐⭐ **循环行为是一个"列表配置"**（用户 2026-09-28 升级 ✓ 原话："把循环行为编辑做成
        #   列表配置，行为编辑器里点击确定后，向列表里加一项，每项可以**重命名**、**删除**、
        #   **双击打开行为编辑器**编辑。实际的执行**每个循环里会按照每个项目依次执行**"）。
        #   ⇒ 一排"列表 + 三个按钮"，双击行 = 打开行为编辑器 ✓（用户指定的手势 ✓）。
        self.ck_loop_items = QListWidget()
        self.ck_loop_items.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ck_loop_items.setToolTip(
            "休息中每隔「循环时间」就把这个列表**从头到尾依次执行一遍** ✓。\n\n"
            "· **双击**一行 = 打开行为编辑器改它 ✓；\n"
            "· 「重命名」改显示名（只影响这一行怎么显示 ✓）；\n"
            "· 一项都没编（或全是空行为）⇒ 这个循环**什么都不做** ✓。")
        self.ck_loop_items.itemDoubleClicked.connect(self._on_loop_item_edit)
        _lb.addRow("循环行为", self.ck_loop_items)
        _lbrow = QHBoxLayout()
        _lbrow.setContentsMargins(0, 0, 0, 0)
        for _txt, _slot, _tip in (
                ("添加", self._on_loop_item_add, "新建一项：先起个名，再打开行为编辑器编它 ✓"),
                ("重命名", self._on_loop_item_rename, "改这一项的显示名 ✓"),
                ("删除", self._on_loop_item_del, "删掉这一项 ✓"),
                ("上移", lambda: self._on_loop_item_move(-1), "往前挪一位（执行顺序）✓"),
                ("下移", lambda: self._on_loop_item_move(1), "往后挪一位（执行顺序）✓")):
            _b = QPushButton(_txt)
            _b.setToolTip(_tip)
            _b.clicked.connect(_slot)
            _lbrow.addWidget(_b)
        _lbrow.addStretch(1)
        _lb.addRow("", _lbrow)
        self.sp_loop_min = self._spin(0.1, 3600, 30, 1, 0.5)
        self.sp_loop_max = self._spin(0.1, 3600, 60, 1, 0.5)
        self.sp_loop_min.valueChanged.connect(self._on_spot_loop_time)
        self.sp_loop_max.valueChanged.connect(self._on_spot_loop_time)
        _loop_row = QHBoxLayout()
        _loop_row.addWidget(self.sp_loop_min)
        _loop_row.addWidget(QLabel("~"))
        _loop_row.addWidget(self.sp_loop_max)
        _lb.addRow("循环时间(s)", _loop_row)
        # ⭐⭐ **「休息结束推迟到循环执行完」**（用户 2026-09-28 要求 ✓ 原话："勾上休息过程中
        #   循环行为时，再加一个开关子参数『休息结束推迟到循环执行完』"）。
        #   ⚠ 位置：**「循环时间(s)」的下面**、同一块缩进里 ⇒ 只有勾了「休息过程中循环行为」
        #     才看得见 ✓（它本来就是那一块的**子参数** ✓）。
        self.ck_loop_hold = QCheckBox("休息结束推迟到循环执行完")
        self.ck_loop_hold.setToolTip(
            "休息时间到了、而**这一轮循环还在演**时，先**等它演完**再结束休息。\n\n"
            "· **不勾**（默认）= 老行为：到点**当场收摊**（手上那段半截就断 ✗）、"
            "立刻去「结束后前往」/ 回战斗 ✓；\n"
            "· **勾上** = 只等**当前这一轮**演完（演完就结束 ✓、**不再开新一轮** ✓）；\n"
            "· 这样才不会出现「一边按着循环的键、一边下发寻路的键」（两边打架 ✗）；\n"
            "· ⚠ **手动点「结束休息」不受它管** —— 那种情况仍是「手上这一项演完就收」✓。")
        self.ck_loop_hold.stateChanged.connect(self._on_loop_hold_toggle)
        _lb.addRow("", self.ck_loop_hold)
        sf.addRow("", self._spot_loop_box)

        self.cmb_spot_after = NoWheelComboBox()
        self.cmb_spot_after.setToolTip("休息**结束之后**先走去哪个集合，再回去打怪。\n"
                                       "选「（不前往）」= 休息完直接继续打怪。")
        self.cmb_spot_after.currentIndexChanged.connect(self._on_spot_changed)
        sf.addRow("结束后前往", self.cmb_spot_after)

        af.addRow(self._afk_spot)

        root.addWidget(afk)
        root.addStretch(1)

        # 倒计时刷新（每秒）：自定义定时行为的「剩余 M:SS」（读 agent 维护的下次触发时刻）。
        # ⚠ 名字原来是 `_feed_timer`（那会儿它只管喂宠）——「自动喂宠」2026-09-26 整块移除后
        #    它就是**通用的倒计时节拍**了 ⇒ 一并改名，别留着旧名字误导人 ✗。
        self._cd_timer = QTimer(self)
        self._cd_timer.setInterval(1000)
        self._cd_timer.timeout.connect(self._tick_timer_cd)

        # 自动开关状态轮询：agent 后台可能因朝向超时等把 enabled 关掉，定时同步 UI
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(500)
        self._state_timer.timeout.connect(self._poll_auto_state)
        # ⭐ 循环项的"当前状态"也搭这趟车刷（500ms ✓ 足够跟手、又不费 ✓）
        self._state_timer.timeout.connect(self._tick_loop_state)
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

    #: ⭐ **开自动前的体检**：上一次的结论（`None` 还没查过 / `True` 通过 / `False` 用户选了否）
    _precheck_ok = None

    @staticmethod
    def _precheck_problems(map_id, src=None, need_mmap=True):
        """⭐ 开自动前的**条件体检**（用户 2026-09-29 ✓ 原话："开启自动前，检查一下条件吧，
        然后出弹窗提示"）。返回**问题清单**（空列表 = 通过 ✓）。

        ⛔ `need_mmap=False`（= 设置里「**禁用杀怪寻路**」开着 ✓）⇒ **一条都不报** ✓：
          这一页查的三件（地图 id / 标定几何 / 地形图）**全都只服务"世界坐标那一套"**，
          而那个模式**根本不读世界坐标**（锁定 / 追击 / 走位全用画面坐标 ✓，见
          `decision/agent.py::_player_located` 的说明 ✓）⇒ 它们**不再**构成"开自动不动、
          或者过一会儿自己停掉" ✓。⚠ 不这么做的后果正是用户 2026-10-04 报的：弹窗默认
          按钮是「否」⇒ 点下去**自动压根没开起来** ⇒ 看着就是"这个开关一开就不干活"✗。

        **为什么要有它**：2026-09-29 那次「开自动却不打怪、只站着」查了半天，根因是
        `森林迷宫III` 的小地图标定**只做了一半** —— `datasets/map/105040303.mapcalib.json`
        里只有 `world_offset`/`alpha`/`mode`（叠加显示用的），**缺 `scale`/`offset`**
        （=「面板 → 底图」换算没做 ✗）⇒ 玩家世界坐标恒为 `None` ⇒ 决策层判"没有玩家位置"
        （`decision/agent.py:4789`）⇒ 你设的 3 分钟一到就**自己把自动停了** ✗。
        而这些事**开之前就查得出来** ✓ ⇒ 所以做在这儿 ✓。

        ⚠ **只判"纯读文件、毫秒级"的项**（地图 id / 标定几何 / 地形底图）✓ —— 能在 UI 线程
        直接调 ✓。**故意不去抓一帧跑定位** ✗：独立推流那条路要**阻塞最多 5 秒**
        （`mm.stream_panel`）⇒ 开个自动先卡 5 秒不能接受 ✓（"黄点认不认得出"那类信息，
          实时页小地图那行本来就在实时显示 ✓ 见 `minimap.PlayerLocator` 的 `short`/`note`）。

        ⚠ 抽成 **`@staticmethod` 且只吃 `map_id`** 是为了**能单测**（不吃 self / 不弹窗 ✓）；
        弹窗那步交给调用方 `_precheck_auto` ✓。
        """
        probs = []
        if not need_mmap:
            return probs                    # ⛔ 「禁用杀怪寻路」开着 ⇒ 这些都不是拦路虎 ✓
        mid = str(map_id or "").strip()
        if not mid:
            probs.append("· **没选地图** —— 本项目还没在 ①「识别目标选项」里选地图。")
            return probs
        try:
            from core import mapdata
            from perception import minimap as mm
            # ⚠⚠ 来源必须与**路线识别页**同一处口径（`mm.source_for_map` ✓）——
            #   2026-10-10 用户现场："**为什么我标定过了还显示这个**" ✗：
            #   这张图按 id 存的是 `live`（从实时画面 ✓ 标定也在那条下 ✓），而这里原来读
            #   **全局** `live.yaml` 的 `mmap_src`（= `stream` 独立推流 ✗）⇒ 去问另一条来源
            #   ⇒ 明明刚标过却报"还没标定" ✓（标定是**按来源分开存**的 ✓ 见
            #   `core/mapdata.calib_path` ✓）。⇒ 现在统一走 `source_for_map` ✓。
            _src = str(src or mm.source_for_map(mid))   # `mid` 已非空 ✓（上面查过 ✓）
            # ① 标定几何：⚠ 就用 `has_geometry`（它**不看 score**、只问"几何量出来没有"✓，
            #    缺 `scale`/`offset` 即 False ⇒ 正是本次踩的坑 ✓）
            calib = None
            try:
                calib = mapdata.load_calib(mid, _src)
            except Exception:
                calib = None
            if not mm.has_geometry(calib or {}):
                # ⭐⭐ **另一条来源下量过就直接点名** ✓（别让人对着"没标定"发呆 ✓）：
                #   这是本次现场最费解的一点 —— 人刚在「从实时画面」下量完，提示却说
                #   当前来源没量 ✗ ⇒ 把"你这张图在**哪条来源**下是量过的"写在同一条提示里 ✓。
                _others = []
                try:
                    for _s, _c in (mapdata.load_calibs(mid) or {}).items():
                        if _s and _s != _src and mm.has_geometry(_c or {}):
                            _others.append("「%s」" % mm.SRC_LABEL.get(_s, _s))
                except Exception:              # noqa: BLE001 —— 这条只是加个提示，别弄坏体检 ✗
                    _others = []
                _hint = ("\n  ⚠ 这张图在 %s 下**是量过的** ✓ ⇒ 要么把「小地图来源」切到那条"
                         "（路线识别页那个下拉 ✓），要么就按当前来源「%s」重量一次 ✓"
                         % ("、".join(_others), mm.SRC_LABEL.get(_src, _src))) if _others else ""
                probs.append(
                    "· **地图「%s」在来源「%s」下还没标定**（缺「面板 → 底图」换算）\n"
                    "  ⇒ 玩家坐标会一直是空的 ⇒ 开一会儿自动自己就会停掉\n"
                    "  → 去「路线识别」页点「标定…」量一次并**保存**。%s"
                    % (mid, mm.SRC_LABEL.get(_src, _src), _hint))
            # ② 地形 / 底图：`canvas is None` = 没生成过地形图（同 route_panel 的口径 ✓）
            try:
                t = mapdata.load(mid, with_canvas=True)
            except Exception:
                t = None
            if t is None or getattr(t, "canvas", None) is None:
                probs.append(
                    "· **地图「%s」还没有地形图 / 底图**\n"
                    "  → 去「路线识别」页点「生成地形图」。" % mid)
        except Exception as e:                  # noqa: BLE001 —— 体检本身不许把开自动搞崩 ✗
            probs.append("· 检查标定 / 地形时出错：%s" % e)
        return probs

    def _auto_precheck_problems(self):
        """开自动要查的那份问题清单 —— **唯一一处**（前台开自动的弹窗 ✓ + 后台重连恢复 ✓）。

        ⚠ 两处**不许各写一遍**参数（`map_id` + `need_mmap`）✗：那个「禁用杀怪寻路」的口径
          2026-10-04 刚错过一次（见 `selftest_live_panel.t_auto_precheck` ④ ✓）——
          再分岔就是两份口径，早晚一处对一处错 ✓。
        """
        return self._precheck_problems(
            self._map_id_for_check(),
            # ⛔ 「禁用杀怪寻路」开着 ⇒ 体检里那三件（地图/标定/地形）**都不构成拦路虎**
            #   （那个模式不读世界坐标 ✓，见 `_precheck_problems` 的说明 ✓）
            need_mmap=not bool(getattr(settings, "disable_chase_pathfinding", False)))

    def _precheck_auto(self):
        """开自动前的体检 + 弹窗。返回 `True` = 放行 ✓。

        ⚠ **只"提示"不"禁止"**（用户说的是"**出弹窗提示**"✓）：提示完让人自己决定 ——
        但**默认按钮给「否」** ✓（这次要拦的是"条件没凑齐就开"✓，安全的一侧才是默认 ✓，
        和 `_confirm_local_auto` 同一个道理 ✓）。
        ⚠ 判据本身在 `_auto_precheck_problems`（**两处共用的唯一一份** ✓）—— 后台那条
          （重连回到游戏 ✓）**不弹窗**，走 `_poll_reconnect_resume` ✓。
        """
        if self._auto_confirming:
            return True                         # 已经在问别的了，别叠对话框 ✗
        probs = self._auto_precheck_problems()
        if not probs:
            return True
        self._auto_confirming = True
        try:
            r = QMessageBox.warning(
                self, "开自动前的检查没通过",
                "下面这些条件没凑齐，开了自动很可能**不动**、或者过一会儿**自己停掉**：\n\n"
                + "\n\n".join(probs)
                + "\n\n（这只是提醒，不是禁止 —— 确认现在就要开吗？）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        finally:
            self._auto_confirming = False
        return r == QMessageBox.Yes

    def _map_id_for_check(self):
        """本项目的地图 id（体检用 ✓ 口径同 `route_panel._map_id`：`project.get("map_id")`）。"""
        try:
            p = getattr(self, "project", None)
            if p is None:
                return ""
            get = getattr(p, "get", None)
            return str((get("map_id") if callable(get) else "") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _resume_auto_verdict(probs):
        """断线重连回来「能不能恢复自动」的判据（**纯函数** ✓ 能单测 ✓ 不吃 self / 不碰 Qt）。

        回 `(要不要开自动, 人话原因)`：有问题 ⇒ `(False, 拼起来的问题清单)`；没问题 ⇒ `(True, "")`。

        ⚠ 与前台那条（`_precheck_auto`）**同一份判据**、**不同处置**：
          · 前台：**弹窗提醒**，由人决定（默认按钮「否」✓ 人就在旁边 ✓）；
          · 后台（这里）：**不弹窗** ✗，直接**不恢复 + 写清原因** —— 人可能不在屏幕前，
            弹一个模态框只会卡住界面，还把"没人点"变成"永久阻塞" ✗。
        """
        if probs:
            return False, "；".join(str(p).strip() for p in probs)
        return True, ""

    def _poll_reconnect_resume(self):
        """⭐⭐ 断线重连回到游戏 ⇒ **体检通过才恢复自动**（用户 2026-10-07 ✓ 三选一里选了"体检"）。

        **为什么要多这一拍**：`reconnect.py::_finish` 原来直接 `settings.enabled = True` ✗
        ⇒ **绕过了「开自动前体检」**（2026-09-29 ✓ `_precheck_problems`）⇒ 标定只做了一半 /
        没有地形图时，重连回来照样把自动打开 ⇒ 表现成"开了却只站着不打、过几分钟自己停" ✗
        —— 那正是 2026-09-29 查了半天的病根。所以状态机只**置旗子** ✓，由这里（**界面线程** ✓
        `_state_timer` 500ms ✓）消费 ✓：`decision/` 里读不了项目 / 标定 / 地形（那是界面的事 ✓），
        实时线程里也不许弹模态框 ✗。

        ⚠ 三条口径：
          · **用户已经自己开着** ⇒ 什么都不做（尊重人 ✓ 别因为体检不过又给他关掉 ✗）；
          · **体检不过** ⇒ **不恢复** ✓ + 状态栏写清"为什么不恢复" ✓ + 落一条日志 ✓；
          · 旗子**没人消费**（headless / 别的宿主 ✓）⇒ 自动就一直不开 ✓（安全的一侧 ✓）。
        """
        # ⭐ 顺手刷新「断线重连 → 鼠标标定」那一行（**只在这一组看得见时**才读那个 json ✓）：
        #   标定是外部 CLI 做的（`tools.mouse_aim_calib` ✓）⇒ 不刷的话界面上永远停在打开
        #   工作台那一刻的状态 ✗ —— 而这一行存在的意义就是"它为什么不点"**在界面上就能看见** ✓
        #   （否则又得去翻文件 / 翻日志 ✗ 用户 2026-10-09 报的正是这类"看不出来" ✓）。
        # ⚠⚠ **两道闸防"控件已经析构"** ✗（2026-10-09 ✓ 自检里踩到）：有些用例会用
        #   `sip.delete` **当场析构**面板（`tests/_kill_qt` ✓ 本仓库的老手法 ✓），而本函数
        #   挂在 500ms 定时器上 ✓ —— 定时器虽然是面板的子对象、本该一起没 ✓，但**已经排队
        #   的那次事件**仍可能落到一个**已经析构的 C++ 对象**上 ⇒ 直接 fail-fast
        #   （`0xC0000409`，**连 Python traceback 都没有** ✗；自检表现就是"跑到某条用例
        #   突然死掉、失败清单都不打" ✓）。⇒ `sip.isdeleted` + try 各一道 ✓。
        try:
            from PyQt5 import sip
            _grp = getattr(self, "_rc_group", None)
            if _grp is not None and not sip.isdeleted(_grp) and _grp.isVisible():
                self._refresh_rc_gain()
        except Exception:                        # noqa: BLE001 —— 刷新失败不该打断这一拍 ✓
            pass
        if not bool(getattr(settings, "reconnect_resume_pending", False)):
            return
        # ⚠ **先清旗子、再体检**：体检自己抛了也不至于每 500ms 重来一遍 ✓（幂等 ✓）
        settings.reconnect_resume_pending = False
        if bool(settings.enabled):
            return                      # 人自己已经开了 ⇒ 不动他 ✓（也不必白跑一趟体检 ✓）
        probs = self._auto_precheck_problems()
        ok, why = self._resume_auto_verdict(probs)
        if not ok:
            # 状态栏那句要**短**（它挤在统计行最前面 ✓）：只取前两条的**第一行**，
            # 完整原因落进日志（`behavior.event` ✓ 复盘时能查全 ✓）。
            _short = "；".join(x.strip().splitlines()[0] for x in probs[:2])
            settings.reconnect_note = (
                "已回到游戏 —— **没恢复自动**：%s（共 %d 条）→ 修好后自己点「开启自动」"
                % (_short, len(probs)))
            behavior.event("reconnect_resume", ok=False, why=why[:120])
            return
        settings.enabled = True
        settings.reconnect_note = "已回到游戏 —— 检查通过，已恢复自动"
        behavior.event("reconnect_resume", ok=True, why="")
        self._refresh_auto_ui()

    def _toggle_auto(self, checked=None):
        on = self.btn_auto.isChecked()
        if on and not self._confirm_local_auto():
            # 在确认框上选了「否」：把按钮拨回去，**不碰 settings.enabled**。
            # blockSignals 只是防御（clicked 不会因为 setChecked 再发一次，
            # 但以后万一有人把它接到 toggled 上，这里就会自己叫自己）。
            self.btn_auto.blockSignals(True)
            self.btn_auto.setChecked(False)
            self.btn_auto.blockSignals(False)
            self._refresh_auto_ui()
            return
        # ⭐⭐ **开之前先体检**（用户 2026-09-29 ✓ 原话："开启自动前，检查一下条件吧，然后
        #   出弹窗提示"）。少了这一步，条件没凑齐时会表现成"开自动却只站着不打"、
        #   或者过几分钟**自己停掉**（`player_lost_stop`）✗ —— 那时候再回头查就要花掉半天
        #   （2026-09-29 就是这么查的 ✓ 根因是"小地图标定只做了一半"）。
        #   顺序放在本地输入确认**之后**：那个是安全确认（先问），这个是条件说明 ✓。
        if on and not self._precheck_auto():
            self.btn_auto.blockSignals(True)
            self.btn_auto.setChecked(False)
            self.btn_auto.blockSignals(False)
            self._refresh_auto_ui()
            return
        settings.enabled = on
        if not on:
            self._force_release_all()   # 关闭自动立即释放所有按键，防卡键
        self._refresh_auto_ui()

    #: 本地输入二次确认是否正在显示。全局热键（WM_HOTKEY）在**任何**程序里都生效，
    #: 连按两下会在模态框自己的事件循环里再进来一次，把确认框叠成一摞。
    _auto_confirming = False

    def _confirm_local_auto(self):
        """输入设备是「本地(仅测试)」时，开自动前先让用户确认一次。

        **为什么非要有**：本地模式走的是本机 SendInput —— 按键打到的是**这台
        机器自己**的键盘输入上，不是游戏机。工作台一失焦、或你切到别的窗口，
        这些键就落进当时的前台程序（浏览器、聊天窗口，也可能是工作台自己）。
        而「开启自动」恰恰是最容易被顺手点一下的那个按钮，还挂着 F11 全局热键
        （在别的程序里按也生效）。

        默认按钮给**「否」**：这里拦的是"手快点了一下"，安全的那一侧才是默认 ——
        和「保存标定」那种（人的意图已经表达了、只是提醒）正相反。

        关自动**不问**：越顺手能停下来越好。
        """
        if settings.input_device != "local":
            return True
        if self._auto_confirming:
            return False        # 已经在问了，别叠第二个
        self._auto_confirming = True
        try:
            r = QMessageBox.question(
                self, "输入设备是「本地(仅测试)」",
                "本地模式的按键是「本机模拟」的：它打到这台机器自己的键盘输入上，"
                "不是游戏机。\n\n"
                "工作台一失焦、或你切到别的窗口，这些键就会落进当时的前台程序"
                "（浏览器、聊天窗口，也可能是工作台自己）。\n\n"
                "确定现在开启自动吗？\n\n"
                "（要挂机跑，请先把「输入设备」切到 ProMicro(远程) 或 ProMicro(本地)。）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        finally:
            self._auto_confirming = False
        return r == QMessageBox.Yes

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

    def _on_pad_tracked(self, dx, dy):
        """触控板位移 → 乘灵敏度 → 交给发送器（**在指针跟踪线程里被调** ✗⏰ 用户 2026-10-05 ✓）。

        为什么是它、而不是 Qt 信号（用户选的"直接2" ✓）：位移的**产生**已经挪进
        `decision/input.py::PointerTracker` 那条独立线程（固定节拍轮询系统指针 ✓），
        它的 `on_delta` 回调就直接落到这里 ⇒ 从"手指动"到"进发送队列"这条路上
        **没有任何 Qt 事件循环** ✓✓（原来两段都挂在 GUI 线程上 ⇒ 一卡就成撮 ✓）。
        位移的**发送**也早就在独立线程里（`_PadSender` ✓）⇒ 现在整条链只剩"链路本身"的
        节奏限制 ✓（这才是该有的样子 ✓）。

        ⚠ **不加锁也安全**（说明白，免得后人加一把没用的锁 ✓）：
          · 这个方法只被**一条**线程（`pad-track` ✓）调 ⇒ 惰性创建 `_pad_sender`
            也只有一个写者 ✓；
          · 另一处会碰它的是 `shutdown`，而它**先停跟踪线程、再停发送器** ✓（见 `shutdown` ✓）
            ⇒ 两者不会同时进行 ✓。
        ⚠ 这里**不许碰界面、不许碰网络** ✗：`_PadSender.push()` 只做加法 + 置事件 ✓
          （真发送在它自己的线程里 ✓）。
        """
        spd = max(0.01, float(settings.mouse_speed))
        s = getattr(self, "_pad_sender", None)
        if s is None:
            s = self._pad_sender = _PadSender()
        s.push(dx * spd, dy * spd)

    def _on_mouse_speed(self, _val=None):
        settings.mouse_speed = float(self.sp_mouse_speed.value())
        settings.save()

    def _refresh_mouse_ui(self):
        """鼠标控制只对 ProMicro 生效；本地模式禁用（本机鼠标要操作界面）。"""
        ok = dinput.mouse_available()
        for w in (self.btn_mouse_l, self.btn_mouse_r, self.sp_mouse_speed):
            w.setEnabled(ok)
        self.touchpad.set_capture_enabled(ok)

    def _stability_check(self):
        """稳定性自检：报内存 / 句柄 / 线程 / Python 对象，并和上次比。

        数值落在「详情」里（点开看）；上面只放需要留意的行 —— 没有就明说
        「没有异常增长」，别让人以为要看一堆数字。
        """
        from gui import stability
        text, warn, first = stability.check()
        box = QMessageBox(self)
        box.setWindowTitle("稳定性自检")
        box.setIcon(QMessageBox.Warning if warn else QMessageBox.Information)
        if warn:
            box.setText("发现值得留意的地方：\n\n" + "\n".join(warn))
        elif first:
            box.setText("已记下基线（详情里是当前值）。\n\n"
                        "隔十几分钟再点一次，就能看出各项涨没涨。")
        else:
            box.setText("没有发现异常增长。\n\n"
                        "一次涨完然后走平是正常的（模型预热、缓存）；"
                        "各项持续单调上涨才是泄漏 —— 隔十几分钟再点一次对照。")
        box.setDetailedText(text)
        box.exec_()

    def _reset_link(self):
        """手动重置指令通道：卡键 / 发不出指令时的逃生口（见按钮 tooltip）。"""
        from decision import input as dinput
        # ⭐ **先把所有方向键点按一遍，再重置**（用户 2026-10-01 ✓）：
        #   重连 / RELEASEALL 只能清掉**固件侧**按住的键；但若一条 RELEASE 在链路里
        #   丢了，**游戏侧**会一直以为某个方向键还按着（角色往一个方向一直走、按啥
        #   都不停）—— 那条"卡住的键"根本不经过固件了。逐个方向键 `tap` 一下 =
        #   往游戏里补一个「按下 → 松开」，把游戏侧的键状态钉回"松开" ✓。
        #   ⚠ 顺序**必须先按、后重置**（用户定的 ✓）：先补的这四下能把游戏侧的方向键
        #   刷干净；随后 RELEASEALL / 重连再清固件侧，两边就都不留卡键 ✓。
        #   ⚠ 通道死了这几下会静默失败（`_send_remote` 吞掉、不抛 ✗）⇒ 无副作用，
        #   后面的重连照走 ✓。
        for _name in ("left", "right", "up", "down"):
            _k = (settings.keymap or {}).get(_name)
            if _k:
                try:
                    dinput.tap(_k)
                except Exception:                       # noqa: BLE001
                    pass
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
        # ⭐⭐ **断线重连请求恢复自动**（用户 2026-10-07 ✓）：**必须排在最前** ——
        #   它可能刚把 `settings.enabled` 打开 ⇒ 下面那句"同步到开关"才会跟着把按钮点亮 ✓
        #   （放在后面就要等下一拍才亮 ✓ 白闪 500ms ✗）。
        self._poll_reconnect_resume()
        # 兜底：面板被藏起来时不该还占着鼠标。hideEvent 管切页签那条路，
        # 这里再兜一层（父级被整体隐藏之类不会走我们的 hideEvent）。
        if self.touchpad.active and not self.isVisible():
            self.touchpad.set_active(False)
        if self.btn_auto.isChecked() != settings.enabled:
            self._refresh_auto_ui()
        resting = bool(settings.rest_state)
        self.btn_end_rest.setEnabled(resting)
        # 「手动触发」跟着自动开关变灰/变亮（自动关着时序列活不过一帧，见 `_fire_timer`）。
        # 每秒设一次 enabled：值没变时 Qt 不会重绘，代价可以忽略。
        self._sync_timer_fire_btns()
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
        self._poll_link_health()

    def _poll_link_health(self):
        """定时核对**指令通道**的真实健康度。

        **为什么必须有这一层**：设备状态那行原来只在「连接成功」那一刻写一次，
        之后再也不更新 —— 于是 relay 崩了（本仓库真实踩过：原生崩溃
        0xC0000005，进程当场消失）、串口掉了、链路卡死了，界面上照旧写着
        「已连接 ProMicro(远程)」，而实际上一条指令都发不出去。
        现在每 500ms 对一次账，不正常就标红说明原因。

        判据来自 `dinput.link_health()`：既看「发送有没有报错」，也看
        「发出去后 5 秒内有没有固件回执」—— 后者能抓到「发送看着成功、
        实际谁也没收到」的情况。恢复动作不在这里做（重连会阻塞、也可能
        和实时线程抢连接），界面上点「重置指令通道」即可。
        """
        h = dinput.link_health()
        backend, ok = h.get("backend"), bool(h.get("ok", True))
        # 坏着就**持续**重试（节流在 _auto_reconnect_link 里，5 秒一次）。
        # **不能只在「状态变化」那一下试一次**：第一次赶巧失败（relay 正忙、
        # 板子还在重枚举）就再也不试了 —— 界面永远红着、指令一条都发不出去，
        # 表现就是「远程输出全无效」。这是我上一版写错的地方。
        if backend in ("remote", "serial") and not ok:
            self._auto_reconnect_link()
        key = (backend, ok, str(h.get("err") or ""))
        if key == getattr(self, "_link_key", None):
            return                      # 状态没变就别每 500ms 重写控件
        self._link_key = key
        if backend in ("remote", "serial"):
            if ok:
                self.lbl_device_state.setText(
                    "已连接 ProMicro(本地)" if settings.input_device == "serial"
                    else "已连接 ProMicro(远程)")
                self.lbl_device_state.setStyleSheet("color: #137333;")
            else:
                self.lbl_device_state.setText(
                    "指令发不出去：%s\n（正在自动重连；也可点「重置指令通道」）"
                    % (h.get("err") or "链路无回执"))
                self.lbl_device_state.setStyleSheet("color: #c5221f;")
        elif backend == "blocked":
            self.lbl_device_state.setText(
                "通道断了又没接上（已停止发指令）—— 正在自动重连")
            self.lbl_device_state.setStyleSheet("color: #c5221f;")
            self._auto_reconnect_link()
        else:
            self.lbl_device_state.setText("本地键盘")
            self.lbl_device_state.setStyleSheet("color: #80868b;")

    def _auto_reconnect_link(self):
        """链路坏了自动重连：后台线程做，最多 5 秒一次。

        **为什么现在敢自动做**：以前只有手动按钮 —— 结果是 relay 一崩或一重启，
        B 机就一直红着、什么都发不出去，得人去点一下（实测踩过：A 机 relay 挂掉后
        界面还写着「已连接」，现在它会变红并自己重连）。

        放后台线程是为了不卡界面；重连本身是安全的：断开那一下 relay 会补一条
        RELEASEALL 松开按住的键（见 decision/input.py 的 reconnect_remote）。
        手动按钮保留 —— 自动重连失败时它还在。
        """
        now = time.monotonic()
        if getattr(self, "_link_rc_busy", False):
            return
        if now - getattr(self, "_link_rc_at", 0.0) < 5.0:
            return
        self._link_rc_at = now
        self._link_rc_busy = True

        import threading

        def _do():
            try:
                dinput.reconnect_remote()
            except Exception:
                pass
            finally:
                self._link_rc_busy = False

        threading.Thread(target=_do, daemon=True, name="link-reconnect").start()

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

        st = str(settings.rest_state or "")
        # 阶段 → 短句走 **agent 里那张共用表**（`REST_STATE_TEXT`）。
        # ⚠ 这里原来是三个 `if`，只认隐身那三个状态 ⇒ 「定点休息」进来之后一个都匹配不上，
        #   于是卡片上写着 **「未休息」** ✗（2026-09-26 用户当场就问了：是没读到休息时长，
        #   还是显示错？—— 是**显示**错，时长一直读得到 ✓）。
        from decision import agent as agent_mod
        txt = agent_mod.rest_state_text(st)
        if txt:
            # 去休息点 / 结束后前往：顺带说清**去哪个集合**（不然只看到"前往休息点…"）
            if st in ("afk_spot_walk", "afk_spot_act"):
                dst = str(getattr(settings, "anti_afk_spot_set", "") or "")
                txt += ("「%s」" % dst) if dst else ""
            elif st == "afk_spot_back":
                dst = str(getattr(settings, "anti_afk_spot_after", "") or "")
                txt += ("「%s」" % dst) if dst else ""
            # ⭐ **循环行为的进度**（用户 2026-09-28："休息过程中希望能看到**每一个循环项目**的
            #   当前状态信息"✓）—— 只在「休息中」那一段显示 ✓（别的阶段这一格是 0 ✓）。
            if st == "afk_spot_rest":
                _tot = int(getattr(settings, "spot_loop_total", 0) or 0)
                _idx = int(getattr(settings, "spot_loop_idx", 0) or 0)
                _nm = str(getattr(settings, "spot_loop_name", "") or "")
                if _tot > 0:
                    if _idx > 0:
                        txt += " · 循环 %d/%d「%s」" % (_idx, _tot, _nm)
                    elif _nm:
                        # 一轮跑完、正在等下一次 ⇒ 说清"还有几秒重来"（不然看着像卡住 ✓）
                        _at = float(getattr(settings, "spot_loop_next_at", 0.0) or 0.0)
                        _w = max(0, int(round(_at - time.monotonic()))) if _at > 0 else 0
                        txt += (" · 循环一轮完成，" + ("%d 秒后重来" % _w if _w > 0
                                                   else "马上重来"))
            # 倒计时只在**真的在计时**的阶段加（其余阶段它是 0 ⇒ `left()` 给空串 ✓）
            return txt + left(settings.rest_until_monotonic)
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

    def _on_attack_vertical(self, _val=None):
        """改了「向上攻击距离 / 向下攻击距离」⇒ 写进配置（决策参数、跟着项目存 ✓）。

        这两条是 2026-09-27 新加的"竖边"（攻击范围从**只看 x** 改成**矩形** ✓）：
        **0 = 该方向不限** ✓ —— 老项目文件里没有这两个键 ⇒ 读进来是 0 ⇒ 行为一点不变 ✓。
        判据只有一处：`decision/agent.py::DecisionAgent._in_box` ✓。
        """
        settings.attack_up_dist = int(self.sp_attack_up.value())
        settings.attack_down_dist = int(self.sp_attack_down.value())
        settings.save()

    def _on_attack_fan(self, _val=None):
        """改了「扇形角度」⇒ 写进配置（决策参数、**跟着项目存** ✓）。

        ⭐ 用户 2026-10-04 ✓：把攻击范围矩形的**上下两条边**以**近端竖边**（最小攻击距离处）
        为支点向外倾斜这个角度 ⇒ 远端更宽（判据与绘制**同一处**：`agent._in_box` /
        `attack_box_poly` ✓）。⚠ **0 = 老行为一字不差** ✓（老项目文件里没有这个键 ⇒ 兜底 0 ✓）。
        """
        settings.attack_fan_deg = float(self.sp_attack_fan.value())
        settings.save()

    def _on_walk_hop(self, _val=None):
        """「走不动按跳」开关写回设置（用户 2026-10-06 ✓）。

        ⚠ 和别的决策参数一样**按项目存** ✓（`walk_stall_jump` 在 `DecisionSettings.to_dict` /
          `from_dict` 里 ✓ ⇒ 跟着 `project.yaml` 的 `decision:` 段走 ✓）；没打开项目时
          `settings.save()` 一个字节都不写 ✓（顶部那行提示已经说清了 ✓）。
        ⚠ 只管**走路被挡住**时那一下跳 ✗ —— 战斗里的「追击起跳」是另一个开关（下面那个 ✓）。
        """
        settings.walk_stall_jump = bool(self.ck_walk_hop.isChecked())
        settings.save()

    def _on_auto_pickup(self, _val=None):
        """「自动拾取」开关写回设置（用户 2026-10-06 ✓）。

        ⚠ 和别的决策参数一样**按项目存** ✓（`auto_pickup` 在 `DecisionSettings.to_dict` /
          `from_dict` 里 ✓ ⇒ 跟着 `project.yaml` 的 `decision:` 段走 ✓）。
        ⚠ 它只管**点不点** ✗ —— 点哪个键在「键盘映射 → 拾取」（`settings.keymap["pickup"]` ✓）。
        """
        settings.auto_pickup = bool(self.ck_auto_pickup.isChecked())
        settings.save()

    def _on_chase_jump(self, _val=None):
        settings.chase_jump_enabled = bool(self.ck_chase_jump.isChecked())
        settings.chase_jump_min = int(self.sp_chase_jump_min.value())
        settings.chase_jump_max = int(self.sp_chase_jump_max.value())
        settings.chase_jump_dash_ms = int(self.sp_chase_dash.value())
        # ⭐ 「距离平台边缘禁用(px)」（用户 2026-10-02 ✓）：开关 + 阈值（勾选框控制可配 ✓）
        if hasattr(self, "ck_chase_edge_guard"):
            settings.chase_jump_edge_guard_enabled = bool(
                self.ck_chase_edge_guard.isChecked())
            settings.chase_jump_edge_guard_px = int(self.sp_chase_edge_px.value())
        settings.save()
        self._refresh_chase_jump_ui()

    def _refresh_chase_jump_ui(self):
        """开关关掉时把参数框灰掉（两个距离 + 冲刺时间）。"""
        on = self.ck_chase_jump.isChecked()
        # 冲刺时间也跟着灰：开关关着时改它没意义（用户要求"勾选追击起跳时才有这个参数"）。
        # `hasattr` 兜底：老配置文件/半初始化的面板里可能还没这个框。
        if hasattr(self, "sp_chase_dash"):
            self.sp_chase_dash.setEnabled(on)
        self.sp_chase_jump_min.setEnabled(on)
        self.sp_chase_jump_max.setEnabled(on)
        # ⭐ 「距离平台边缘禁用」那个 px 框：**两个开关都要开**才可配
        #   （追击起跳开 ✓ + 它自己的勾选框开 ✓）—— 就是用户那句"**不启用时不能配置**" ✓
        #   （样式同「大怪优先」的两个参数框 ✓）。
        if hasattr(self, "ck_chase_edge_guard") and hasattr(self, "sp_chase_edge_px"):
            self.sp_chase_edge_px.setEnabled(
                bool(on) and self.ck_chase_edge_guard.isChecked())

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

    # ---------------- 循环行为**列表**（用户 2026-09-28 ✓） ----------------

    def _loop_items(self):
        """设置里那份列表（**永远返回可写的那一份** ✓ 改完自己调 `_save_loop_items` ✓）。"""
        return settings.anti_afk_spot_loop_items

    def _save_loop_items(self):
        """写回 + 存盘 + 重画列表（**唯一出口** ✓ 免得五处各写一遍 ✗）。"""
        settings.save()
        self._refresh_loop_list()

    def _tick_loop_state(self):
        """按运行态给**列表里的当前项**加个 `▶` 前缀（用户 2026-09-28 要"每个循环项目的
        **当前状态**"✓）—— 挂在 500ms 的状态轮询里（`_state_timer` ✓ 不额外开定时器 ✓）。

        ⚠⚠ **只改文本、不动选中行**（用户正拿这个列表编辑 ✓ 抢走选中会很难受 ✗）——
          也不重灌列表（重灌会清掉选中 ✗）；纯 `setText` ✓。
        ⚠ 只在**休息中**显示（其余时候把前缀摘掉 ✓ 免得配置界面一直挂个 ▶ 看不懂 ✓）。
        """
        if not hasattr(self, "ck_loop_items"):
            return
        idx = 0
        if str(getattr(settings, "rest_state", "") or "") == "afk_spot_rest":
            idx = int(getattr(settings, "spot_loop_idx", 0) or 0)
        self.ck_loop_items.blockSignals(True)
        try:
            for i in range(self.ck_loop_items.count()):
                it = self.ck_loop_items.item(i)
                base = str(it.text()).lstrip("▶ ").strip()
                it.setText(("▶ %s" % base) if (idx > 0 and i == idx - 1) else base)
        finally:
            self.ck_loop_items.blockSignals(False)

    def _refresh_loop_list(self):
        """把列表灌进控件（一行 = 一项的名字 ✓）；顺便按勾选框显示/隐藏那一块 ✓。"""
        self.ck_loop_items.blockSignals(True)
        try:
            self.ck_loop_items.clear()
            for it in self._loop_items():
                self.ck_loop_items.addItem(str((it or {}).get("name") or "（未命名）"))
        finally:
            self.ck_loop_items.blockSignals(False)
        self._refresh_spot_loop_ui()

    def _loop_sel(self):
        """当前选中的那一项的**下标**（没选给 -1 ✓）。"""
        return int(self.ck_loop_items.currentRow())

    def _on_loop_item_add(self):
        """**添加**一项：先起名字 ⇒ 再打开行为编辑器编它 ✓（用户说"点击确定后向列表里加一项"✓）。"""
        name, ok = QInputDialog.getText(self, "添加循环行为", "这一项叫什么名字？",
                                        text="循环行为")
        if not ok:
            return
        self._loop_items().append({"name": str(name or "（未命名）").strip() or "（未命名）",
                                  "seq": []})
        self._refresh_loop_list()
        self.ck_loop_items.setCurrentRow(len(self._loop_items()) - 1)
        self._save_loop_items()
        self._on_loop_item_edit()          # ⇐ 顺手把行为编辑器打开（用户要"点了确定就加一项"✓）

    def _on_loop_item_edit(self, *_a):
        """**双击**（或刚添完）⇒ 打开行为编辑器编**这一项**的行为 ✓。"""
        i = self._loop_sel()
        if i < 0:
            QMessageBox.information(self, "先选一项", "在列表里点一下要编辑的那一项 ✓")
            return
        from gui.seq_editor import SeqEditorDialog

        items = self._loop_items()
        dlg = SeqEditorDialog(list(items[i].get("seq") or []), self,
                              title="「%s」行为编辑器" % (items[i].get("name") or "循环行为"))
        if dlg.exec_():
            items[i]["seq"] = dlg.seq()
            self._save_loop_items()

    def _on_loop_item_rename(self):
        """**重命名**（只改显示名 ✓ 不影响行为 ✓）。"""
        i = self._loop_sel()
        if i < 0:
            QMessageBox.information(self, "先选一项", "在列表里点一下要改名的那一项 ✓")
            return
        items = self._loop_items()
        name, ok = QInputDialog.getText(self, "重命名", "新名字：",
                                        text=str(items[i].get("name") or ""))
        if not ok:
            return
        items[i]["name"] = str(name or "（未命名）").strip() or "（未命名）"
        self._refresh_loop_list()
        self.ck_loop_items.setCurrentRow(i)
        self._save_loop_items()

    def _on_loop_item_del(self):
        """**删除**一项（有确认 ✓）。"""
        i = self._loop_sel()
        if i < 0:
            QMessageBox.information(self, "先选一项", "在列表里点一下要删的那一项 ✓")
            return
        items = self._loop_items()
        if QMessageBox.question(
                self, "删除循环行为",
                "确定删掉「%s」吗？" % (items[i].get("name") or "（未命名）")) != QMessageBox.Yes:
            return
        del items[i]
        self._refresh_loop_list()
        self.ck_loop_items.setCurrentRow(min(i, len(items) - 1))
        self._save_loop_items()

    def _on_loop_item_move(self, step):
        """上移 / 下移一位（**执行顺序**就是列表顺序 ✓ 用户要"依次执行"⇒ 顺序得能调 ✓）。"""
        i = self._loop_sel()
        j = i + int(step)
        items = self._loop_items()
        if i < 0 or j < 0 or j >= len(items):
            return
        items[i], items[j] = items[j], items[i]
        self._refresh_loop_list()
        self.ck_loop_items.setCurrentRow(j)
        self._save_loop_items()

    def _on_spot_loop_toggle(self, _state=None):
        """勾选框变化 ⇒ 写回设置 + 刷新缩进那块的显隐 ✓。"""
        settings.anti_afk_spot_loop = bool(self.ck_spot_loop.isChecked())
        self._refresh_spot_loop_ui()
        settings.save()

    def _on_loop_hold_toggle(self, _state=None):
        """「休息结束推迟到循环执行完」变化 ⇒ 写回 + 存盘 ✓。

        ⚠ 它**只影响"自然到点"那一条**（口径在 `decision/agent._hold_rest_for_loop` ✓）——
          界面这边不做任何"立刻生效"的动作（不需要：到点那一刻 agent 现读现判 ✓）。
        """
        settings.anti_afk_spot_loop_hold = bool(self.ck_loop_hold.isChecked())
        settings.save()

    def _on_spot_loop_time(self, _val=None):
        """「循环时间(s) A~B」的 A/B 任一变化 ⇒ 写回 + 夹下限（A 不许大于 B ✓）。"""
        settings.anti_afk_spot_loop_min = float(self.sp_loop_min.value())
        settings.anti_afk_spot_loop_max = float(self.sp_loop_max.value())
        if settings.anti_afk_spot_loop_max < settings.anti_afk_spot_loop_min:
            settings.anti_afk_spot_loop_max = settings.anti_afk_spot_loop_min
        settings.save()

    def _refresh_spot_loop_ui(self):
        """按勾选框显示/隐藏「循环行为编辑 + 循环时间(s)」那一块（**下方缩进** ✓）。

        ⚠ 用 `setVisible` 而不是 `setEnabled`：用户要的是"**勾选后下方缩进出现**参数"✓
          —— 藏起来才是"出现" ✓（灰掉仍是"一直在那儿"✗）。
        """
        if not hasattr(self, "_spot_loop_box"):
            return
        self._spot_loop_box.setVisible(bool(self.ck_spot_loop.isChecked()))
        # ⚠ **这里不许再调 `_refresh_loop_list`** ✗ —— 那个函数结尾会反过来调本函数
        #   ⇒ 无限递归（`RecursionError`，当天踩过 ✓）。分工：**列表内容**归
        #   `_refresh_loop_list`、**整块显隐**归本函数 ✓（回填处先刷列表再刷这里 ✓）。

    def _on_afk_rest_time(self, _val=None):
        settings.anti_afk_rest_min = float(self.sp_rest_min.value())
        settings.anti_afk_rest_max = float(self.sp_rest_max.value())
        if settings.anti_afk_rest_max < settings.anti_afk_rest_min:
            settings.anti_afk_rest_max = settings.anti_afk_rest_min
        settings.save()

    def _on_afk_retry(self, state):
        settings.anti_afk_retry_on_interrupt = bool(state)
        settings.save()
        self._refresh_afk_ui()      # 勾了才让"秒数"可调

    def _on_afk_retry_sec(self, _val=None):
        settings.anti_afk_retry_sec = max(1.0, float(self.sp_afk_retry.value()))
        settings.save()

    def _refresh_afk_ui(self):
        """按行为类型显示对应的参数子组；「打断后重试(s)」只在勾选时可调。

        用**灰掉**而不是藏掉：布局不跳（藏掉会让下面的控件往上蹦一下），
        而且灰着的框旁边就是那句 tooltip，比"消失了"更好解释。
        """
        self._afk_hidden.setVisible(settings.anti_afk_type == "hidden_rest")
        self._afk_spot.setVisible(settings.anti_afk_type == "spot_rest")
        self.sp_afk_retry.setEnabled(bool(settings.anti_afk_retry_on_interrupt))
        # ⚠ 那块"缩进参数"的显隐**只归勾选框管**（子组一藏它也跟着看不见 ✓）——
        #   这里再刷一次是为了"切回定点休息类型时状态是对的"（不刷也不会错，但便宜 ✓）。
        self._refresh_spot_loop_ui()

    def set_zone_sets(self, names):
        """把**当前地图已注册的集合名**灌进「定点休息」那两个下拉（由路线识别面板推过来）。

        为什么是"推"而不是自己去读：集合属于**地图**（`core/zones` 按 map id 存），
        而玩家面板不持有地图 id —— 路线识别面板那边才有（`_map_id()` / `bind(project)`），
        所以走它已经用的那套"面板间推送"（同 live_panel.set_mmap ✓）。
        保留当前选择：refill 之后按名字重新选回去；选的名字没了就回到"未选"。
        """
        names = [str(n) for n in (names or [])]
        for cmb, empty_label, cur in (
                (self.cmb_spot_set, "（未选）", settings.anti_afk_spot_set),
                (self.cmb_spot_after, "（不前往）", settings.anti_afk_spot_after)):
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItem(empty_label, "")
            for n in sorted(names):
                cmb.addItem(n, n)
            j = cmb.findData(str(cur or ""))
            cmb.setCurrentIndex(j if j >= 0 else 0)
            cmb.blockSignals(False)

    def _on_spot_changed(self, _i=None):
        """「定点休息」的三个参数写回配置。

        ⚠ 单独一个槽（不接 `_on_anti_afk`）：那个槽的第一个参数是**勾选框状态**，
        下拉的 `currentIndexChanged(int)` 传进去会被当成 bool ⇒ 选到 index 0 就把
        "自动防掉线"关掉了 ✗（这类签名串台的坑，本仓库踩过好几次）。
        """
        settings.anti_afk_spot_set = str(self.cmb_spot_set.currentData() or "")
        settings.anti_afk_spot_after = str(self.cmb_spot_after.currentData() or "")
        settings.save()

    def _edit_afk_seq(self, which):
        """打开「进入隐身」/「退出隐身」的行为编辑器。"""
        from gui.seq_editor import SeqEditorDialog
        # 三个入口共用：进入隐身 / 退出隐身 / 定点休息的「到达后行为」
        attr = {"enter": "anti_afk_enter_seq",
                "exit": "anti_afk_exit_seq",
                "spot": "anti_afk_spot_seq"}[which]
        # ⚠ 标题原来只判了 enter/exit ⇒ `spot` 会得到"**退出隐身**行为编辑器"这个错标题 ✗
        #   （用户 2026-09-28 加第三个入口时顺手改成查表 ✓）。
        title = {"enter": "进入隐身", "exit": "退出隐身",
                 "spot": "到达后行为"}[which] + "行为编辑器"
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

    def _on_mob_atk_cd(self, _val=None):
        """「**目标被攻击CD**」写回设置（用户 2026-10-06 ✓）。

        ⚠ 它是**按怪**记的冷却（`DecisionSettings.mob_atk_cd_ms` ✓）：`0` = 不限制 ✓；
          agent 那边一张账、盖戳只在一处（见 `CombatAgent._mark_mob_atk_cd` ✓）。
        ⚠ 跟旁边参数一样**按项目存** ✓（在 `to_dict` / `from_dict` 里 ✓）；没打开项目时
          `settings.save()` 一个字节都不写 ✓（顶部那行提示已经说清 ✓）。
        """
        settings.mob_atk_cd_ms = int(self.sp_mob_atk_cd.value())
        settings.save()

    def _on_atk_target_count(self, _val=None):
        """「**攻击目标数量**」写回设置（用户 2026-10-06 ✓）。

        ⚠ 三种意思（`DecisionSettings.atk_target_count` ✓）：`0` = **不启用**「目标被攻击CD」
          （⇒ 顺手把那一格置灰 ✓）、`-1` = **不限制**、`N>0` = **同时最多 N 只怪**在 CD 里 ✓。
          内核判"开没开"只在 `CombatAgent._mob_atk_cd_on` **一处** ✓（这儿只管界面 ✓）。
        ⚠ 跟旁边参数一样**按项目存** ✓（在 `to_dict` / `from_dict` 里 ✓）。
        """
        settings.atk_target_count = int(self.sp_atk_target_count.value())
        # 0 ⇒ 下面那格置灰（用户原话："配置置灰"✓）；回 -1 / 正数 ⇒ 亮回来 ✓（值一直留着 ✓）
        self._sync_mob_cd_enabled()
        settings.save()

    def _sync_mob_cd_enabled(self):
        """按「攻击目标数量」把「目标被攻击CD」那格**置灰 / 亮回来** —— **只此一处** ✓。

        ⚠ 两处都要调它：① 新建/回填（`_sync_from_settings` ✓）② 用户改上面那格
          （`_on_atk_target_count` ✓）—— 只在一处调 = 另一个入口的灰/亮是错的 ✗。
        ⚠ **不动那格的值** ✓：用户填的 300 一直留着（把「攻击目标数量」调回正数就照旧生效 ✓）
          —— 千万别在置灰时顺手清零 ✗。
        """
        self.sp_mob_atk_cd.setEnabled(
            int(self.sp_atk_target_count.value()) != 0)

    def _on_attack_cd(self, val):
        settings.attack_cd = int(val)
        settings.save()

    def _on_attack_lock_db(self, val):
        settings.attack_lock_debounce_ms = int(val)
        settings.save()

    def _on_track_jump(self, val):
        settings.player_track_jump = int(val)
        settings.save()

    # ⚠ 原来这里有个槽（`_on_player_loc_params`）—— 「玩家位置」搬去**设置 → 判定参数页**
    #   之后**整块删掉** ✓：那几项的写回在 `settings_dialog._accept` 里（和「坐标对齐误差
    #   范围」那批**同一段** ✓ 都是跟着项目存 ✓ 一份实现、一处维护 ✓）。

    # ⚠ 原来这里有两个方法（`_refresh_arrow_btn` / `_pick_arrow_color`）—— 箭头搬去
    #   「设置 → 界面 → 辅助线与标记（实时预览）」之后**整块删掉** ✓：那边的色块、取色弹窗
    #   （带 alpha 拖动条 ✓）、开关都走 `settings_dialog` 里那套现成的 `add_row` ✓
    #   —— 一份实现、一处维护（规范 §4 ✓）。

    def _on_turn_params(self, _val=None):
        """换向相关的两个时间参数一起写（同一组，一个处理器够了）。"""
        settings.min_turn_hold_ms = int(self.sp_min_turn_hold.value())
        # ⭐ 「站桩补朝向间隔(ms)」（2026-10-02 ✓）：与上面那条同一个处理器 ⇒ 改完即时落盘 ✓
        #   （`settings.save()` 在下面统一调 ✓）—— 它随项目存 ✓，也会跟着"保存/加载模板"走 ✓。
        if hasattr(self, "sp_station_turn_iv"):
            settings.station_turn_interval_ms = int(self.sp_station_turn_iv.value())
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
        """策略类型下拉**变了** ⇒ 只切下面的参数**预览**（**不生效** ✓ 用户 2026-10-05）。

        ⚠ 生效要按右下角的「启用」（`_apply_strategy` ✓）—— 用户要求"手动启用"（原话：
          "需要新加一个手动启用策略类型：在策略参数组最上方加「当前策略类型」，右下角加
          「启用」"✓）：换策略换的是**整套行为**，让"选"与"生效"分开，才看得清现在跑的是哪一套 ✓。
        """
        self._refresh_strategy_ui()

    def _apply_strategy(self):
        """「启用」：把下拉里选中的那个策略**真正生效**（写进项目参数 ✓）。

        已经是它 ⇒ 一个字节都不动 ✓（按钮那时本来就是灰的 ✓ 见 `_refresh_strategy_ui`）。
        """
        sel = str(self.cmb_strategy.currentData() or "patrol")
        if sel == str(getattr(settings, "strategy", "patrol") or "patrol"):
            return
        # ⛔ 「**多点巡逻**」的点位少于 2 个 ⇒ **不许启用**（用户 2026-10-06 ✓ 第 4 条原话：
        #   "**当未选择点位时，无法启用（爆红字）**"✓）⇒ 就地下红字、**什么都不改** ✓
        #   （连"选中的那个"也保持原样 ✓ —— 让人先补齐点位再按一次 ✓）。
        if sel == "multi":
            n = len([x for x in (getattr(settings, "multi_spots", None) or [])
                     if isinstance(x, dict) and x])
            if n < 2:
                self.lbl_strategy_note.setText(
                    "⚠「多点巡逻」**至少要选 2 个点位**才能启用（现在 %d 个）—— "
                    "先点上面的「选择点位…」再回来按「启用」。" % n)
                self.lbl_strategy_note.setStyleSheet("color: #c5221f;")
                return
        settings.strategy = sel
        settings.save()
        self._refresh_strategy_ui()

    def _on_turn_cd(self, val):
        settings.sweep_turn_cd = int(val)
        settings.save()

    def _on_back_range(self, val):
        settings.back_range = int(val)
        settings.save()

    def _on_edge_turn(self, val):
        settings.sweep_edge_turn_px = int(val)
        settings.save()

    # ---- ⭐⭐ 「扫平台」两条换向规则的开关（用户 2026-10-09 ✓ 原话："在「换朝向延迟」、
    #      「距离平台边缘回头」**前面加勾选**；**当都不勾选时默认走到平台边缘 100px 回头**"✓）----

    def _on_turn_cd_on(self, on):
        """「换朝向延迟」前面那个勾：不勾 ⇒ 这条规则**关掉** ✗（不会因为前方暂时没怪掉头 ✓）。"""
        settings.sweep_turn_cd_enabled = bool(on)
        self._sync_sweep_switches()
        settings.save()

    def _on_edge_turn_on(self, on):
        """「距离平台边缘回头」前面那个勾：不勾 ⇒ 这条规则**关掉** ✗。"""
        settings.sweep_edge_turn_enabled = bool(on)
        self._sync_sweep_switches()
        settings.save()

    def _sync_sweep_switches(self):
        """两个勾 ↔ 两个数字格 ＋ 说明行（**一处口径** ✓ 构造回填与勾选都调它 ✓）。

        ⚠⚠ **都不勾不是"永远不回头"** ✗ —— 按**内置 100px** 走到平台边缘回头 ✓
          （用户 2026-10-09 原话 ✓ 实现见 `decision/agent.py::SWEEP_EDGE_FALLBACK_PX` ✓）；
          这时「距离平台边缘回头」那格**填多少都不看** ✓（所以置灰 ✓）。
        ⚠ 说明行**纯文本** ✗（QLabel 不认 markdown ✓ 同本页那条纪律 ✓）。
        """
        _cd_on = bool(self.ck_turn_cd.isChecked())
        _edge_on = bool(self.ck_edge_turn.isChecked())
        self.sp_turn_cd.setEnabled(_cd_on)
        self.sp_edge_turn.setEnabled(_edge_on)
        if not _cd_on and not _edge_on:
            self.lbl_sweep_hint.setText(
                "两条都不勾 ⇒ 按内置 100px 走到平台边缘回头（上面那格填多少都不看）")
        elif not _cd_on:
            self.lbl_sweep_hint.setText(
                "只按「距离平台边缘回头」转：走到边缘附近就回头，前方没怪也不掉头")
        elif not _edge_on:
            self.lbl_sweep_hint.setText(
                "只按「换朝向延迟」转：前方没怪够久才掉头，走到边缘也不按它转")
        else:
            self.lbl_sweep_hint.setText("")

    # ---- 「挂机保护」页（2026-09-29：右侧新页签，见 main_window ✓）----

    def build_protection_page(self):
        """构建「挂机保护」页（main_window 把返回值 `addTab` 成新页签 ✓）。

        内容 = 「防挂机」新组（触发音效 + 试听 ✓）+ ⭐「断线重连」组（点名要的那个参数 ✓）
        + **整体搬来**的「防掉线」组 ✓。
        ⚠⚠ 只是 **re-parent**：防掉线的控件与全部槽函数仍归**本面板**所有
        （信号早就连在它身上 ✓）—— 搬的是视觉位置，不是重写 ✗；
        `lay.addWidget(self.afk)` 会把它从决策参数页的布局里**自动摘走** ✓。
        """
        page = QWidget()
        lay = scroll_page(page, margins=(12, 12, 12, 12), spacing=8)
        lay.addWidget(self._build_antihang_group())
        # ⭐⭐ 「断线重连」组（用户 2026-10-07 ✓ 原话："是数据工作台主窗口「挂机保护页签」"
        #   —— 一开始做进了**设置弹窗**，位置不对 ✗ 在这儿才对 ✓）。
        lay.addWidget(self._build_reconnect_group())
        # ⭐⭐ 「自动测谎（视觉追踪）」组（用户 2026-10-09 ✓ 原话："勾选框参数放到挂机保护
        #   页签，单独一个组" ✓ —— 一开始做在「实时」页那一行，位置不对 ✗ 在这儿才对 ✓）
        lay.addWidget(self._build_vt_group())
        lay.addWidget(self.afk)               # ⭐ re-parent（见上 ✓）
        lay.addStretch(1)
        return page

    def _build_vt_group(self):
        """⭐ 「自动测谎（视觉追踪）」组（2026-10-09 ✓ 用户要求：挂机保护页签 · 单独一个组）。

        这一组管的是**接进外部 `visual_tracking` 包**的那条路（见 `perception/vt_sdk.py` ✓）：
        把「实时画面」那一块（**画框之前的原生帧** ✓）交给它跟踪，报成功时用鼠标点一下「确定」。

        参数落 `config/live.yaml`（**本机偏好** ✓ 同「保留测谎录屏」那两项 ✓，不进项目文件 ✗）：
        实时线程**每拍热读**它（见 `live_thread` 里那段 `load_live()` ✓）⇒ 这里改完
        **最多 1 秒生效**、**不用重开预览** ✓（也就不用把线程引用塞到这里来 ✓）。
        """
        _lv = load_live()
        grp = QGroupBox("自动测谎（视觉追踪）")
        f = QFormLayout(grp)

        # ⭐⭐ **状态行**（用户 2026-10-09 ✓ 原话："这个状态肯定要一起迁移过去啊" ✓）——
        #   从「实时」页整块搬来 ✓（那边只剩一句"已搬走"的注释 ✓）。
        #   ⚠ 文字**只有实时线程发得出来**（`LiveThread.vt_status` ✓）：那一路先到「实时」页的
        #     `_on_vt_status`，它**原样转发**（`LivePanel.vt_status` ✓）⇒ 这里显示 ✓。
        #     **一处显示** ✓ —— 两边各显示一份就是"两份状态、迟早不一致" ✗。
        self.lbl_vt = QLabel("测谎：未启用")
        self.lbl_vt.setStyleSheet("color: #5f6368;")
        self.lbl_vt.setWordWrap(True)
        self.lbl_vt.setToolTip(
            "它是跟踪包的**实时状态**（跟着实时线程走 ✓），按这几种循环：\n"
            "  · `加载模型…` / `模型就绪` —— 第一次启用时的模型加载（约 10 秒 ✓）；\n"
            "  · `LOCATING` —— 还在画面里找弹窗 ⇒ 这时描边和框都**不会**出现 ✓；\n"
            "  · `WAITING` / `LOCKED` / `COAST` —— 已找到内容区，锁定目标 / 预测延续；\n"
            "  · `LOST` —— 跟丢了（**不代表任务结束** ✓ 它会自己找回来）；\n"
            "  · `SUCCESS_PENDING` / `SUCCESS` —— 成功面板出现 ⇒ 按上面那格去点「确定」；\n"
            "  · `不可用 —— …` —— 起不来（原因就写在这行文字里 ✓）。")
        f.addRow("当前状态", self.lbl_vt)
        # ⚠ **不在这里自己起线程 / 碰线程**：那条线归「实时」页 ✓ —— 它转发过来就行 ✓
        #   （`main_window` 建这一页之前，已经把实时面板挂在 `PlayerPanel.live_panel` 上了 ✓）。
        _lp = getattr(self, "live_panel", None)
        if _lp is not None:
            try:
                _lp.vt_status.connect(self.set_vt_status)
            except Exception:                # noqa: BLE001 —— 连不上也不该让页面建不出来 ✓
                pass

        self.ck_vt = QCheckBox("启用（把实时画面交给 visual_tracking 跟踪）")
        self.ck_vt.setChecked(bool(_lv.get("vt_on", False)))
        self.ck_vt.setToolTip(
            "勾上就把「实时画面」里**框选的那块窗口**（画框之前的原生帧 ✓）喂给外部视觉追踪包\n"
            "`visual_tracking_sdk_20260920`：它自己会在画面里找游戏弹窗、并跟踪那个目标。\n\n"
            "  · 它**只检测、不点击**（`INTEGRATION.md:105` ✓）—— 报成功之后点不点「确定」\n"
            "    由下面那一格决定 ✓；\n"
            "  · 第一次开会加载模型（YOLO 权重，约 10 秒）；加载/推理都在**独立线程**里，\n"
            "    **不影响实时画面** ✓；\n"
            "  · ⚠ 要鼠标点得动，得**标定过鼠标**（`config/mouse_gain.json` ✓ 同断线重连那套 ✓）——\n"
            "    鼠标指令是**经 relay 打到游戏机**的 ✓ **收流 / 本机窗口都行** ✓（跟画面从哪来无关 ✓）；\n\n"
            "状态看本组最上面那一行（定位中 / 跟踪中 / 成功⇒已点确定 / 不可用的原因 ✓）。")
        self.ck_vt.stateChanged.connect(self._on_vt_changed)
        f.addRow("", self.ck_vt)

        self.ck_vt_click = QCheckBox("成功后自动用鼠标点「确定」")
        self.ck_vt_click.setChecked(bool(_lv.get("vt_click", True)))
        self.ck_vt_click.setToolTip(
            "跟踪包报「成功面板出现了」之后，**我们**去找面板上那个「确定」按钮并点一下\n"
            "（按钮位置是拿它自带的按钮模板在面板范围里匹配出来的 ✓ 不猜坐标 ✓）。\n\n"
            "  · 匹配不到（阈值 0.86 内没有）⇒ **不点**（点错比不点糟 ✗），日志里留一条\n"
            "    `vt_confirm_miss` ✓；\n"
            "  · 取消勾选 ⇒ 只报不点（想自己确认时用 ✓）。\n\n"
            "⚠ 点一下走的是断线重连那套硬件鼠标（`mouse_aim`）：会先**撞角归零**再走位 ✓。")
        self.ck_vt_click.stateChanged.connect(self._on_vt_changed)
        f.addRow("", self.ck_vt_click)

        self.ck_vt_aim = QCheckBox("鼠标跟着目标走（指着它）")
        self.ck_vt_aim.setChecked(bool(_lv.get("vt_aim", True)))
        self.ck_vt_aim.setToolTip(
            "跟踪状态是 `LOCKED` / `COAST`、而且点**不超过 0.2 秒新**时，让硬件鼠标**指过去**\n"
            "（撞角归零 → 走位 —— **不点** ✓ 同断线重连那套后端 ✓）。\n\n"
            "  · 「什么时候该动鼠标」这套规矩**照抄跟踪包自己的适配器**（`MouseOutput` ✓\n"
            "    只认锁定/预测态、超时点不补发、成功界面一律停手 ✓）—— 不是我们自己定的 ✓；\n"
            "  · ⚠ 绝对值定位每次都要**撞角归零** ⇒ 一格点只发一次，不会疯狂甩动 ✓；\n"
            "  · ⚠ 鼠标指令**经 relay 打到游戏机** ✓ ⇒ **收流 / 本机窗口都有效** ✓（前提：标定是在\n"
            "    **同一画面几何**下做的 —— 你那份 `mouse_gain.json` 记着 `source: stream` ✓ 正是如此 ✓）；\n"
            "  · ⚠ 要**标定过鼠标**（`config/mouse_gain.json` ✓）。\n\n"
            "想自己接管鼠标时把它关掉 ⇒ 只跟不指 ✓。")
        self.ck_vt_aim.stateChanged.connect(self._on_vt_changed)
        f.addRow("", self.ck_vt_aim)

        self.ck_vt_draw = QCheckBox("在画面上描边（ROI / 目标框 / 瞄点）")
        self.ck_vt_draw.setChecked(bool(_lv.get("vt_draw", True)))
        self.ck_vt_draw.setToolTip(
            "在「实时画面」上画三件（都在**显示帧**上 ✓ 喂给跟踪包的那份一个像素都不动 ✓）：\n"
            "  · **橙黄矩形 + `VT <阶段>`** = 跟踪包自己定位到的**弹窗内容区**（ROI ✓）——\n"
            "    没看到它就说明它还没定位到（阶段会写 LOCATING ✓）；\n"
            "  · **品红粗框** = 它认的**目标框**；\n"
            "  · **品红十字** = 鼠标正要指的**瞄点**。\n\n"
            "调试期建议开着（「它到底跟没跟上」一眼就看出来 ✓）；不想让画面花就关掉 ✓。\n"
            "⚠ 要看到这些 ⇒ 「实时画面」那边的「**画框**」也得开着（同其它标记 ✓）。")
        self.ck_vt_draw.stateChanged.connect(self._on_vt_changed)
        f.addRow("", self.ck_vt_draw)

        self.sp_vt_aim_ms = NoWheelSpinBox()
        self.sp_vt_aim_ms.setRange(4, 200)
        self.sp_vt_aim_ms.setSingleStep(4)
        self.sp_vt_aim_ms.setValue(int(_lv.get("vt_aim_ms", 16)))
        self.sp_vt_aim_ms.setToolTip(
            "平滑走位的**插值间隔**（毫秒，默认 **16** ≈ 60 次/秒）。\n\n"
            "它管的是「光标走得多细」：**调小 = 更顺滑**（每拍走更小的一步 ✓，越接近连续移动 ✓）、\n"
            "调大 = 更省固件带宽但会看出「一顿一顿」✗。\n\n"
            "⚠ 它**不是**「多久指一次」✗：目标每一拍都在刷新（跟着跟踪结果 ✓），鼠标每拍走一小步 ✓；\n"
            "⚠ 只有「位置不可信」时才归零一次（每轮开头 / 停顿超过 1 秒 ⇒ 人可能碰过鼠标 ✓），\n"
            "   所以不会再出现「在左上角和目标之间来回跳」✓。\n\n"
            "  · 还是觉得顿 ⇒ **调到 8**（更顺 ✓，代价是固件要收更多指令）；\n"
            "  · 固件带宽吃紧 / 卡 ⇒ 调大到 24~33 ✓。")
        self.sp_vt_aim_ms.valueChanged.connect(self._on_vt_changed)
        f.addRow("鼠标跟随间隔(ms)", self.sp_vt_aim_ms)

        self.sp_vt_conf = NoWheelDoubleSpinBox()
        self.sp_vt_conf.setRange(0.05, 0.90)
        self.sp_vt_conf.setDecimals(2)
        self.sp_vt_conf.setSingleStep(0.05)
        self.sp_vt_conf.setValue(float(_lv.get("vt_conf", 0.15)))
        self.sp_vt_conf.setToolTip(
            "跟踪包内部检测的置信度门限（默认 **0.15**，与它交付时验过的那套一致 ✓）。\n\n"
            "  · 调**高** ⇒ 只认更像的框：漏检变多，但误检少；\n"
            "  · 调**低** ⇒ 更不容易漏，但会有低分框混进来。\n\n"
            "⚠ 它的算法内部另有一条低分通道（BYTE 辅助）用于跟踪 ⇒ 这一格不是「越低越好 / "
            "越高越好」⇒ **没特殊原因别动它** ✓；要动就先跑一遍素材对照（同一目标、同一段）。")
        self.sp_vt_conf.valueChanged.connect(self._on_vt_changed)
        f.addRow("检测置信度", self.sp_vt_conf)

        self.cmb_vt_region = NoWheelComboBox()
        self.cmb_vt_region.addItem("国服（CN）", "CN")
        self.cmb_vt_region.addItem("繁中（TW）", "TW")
        _rg = str(_lv.get("vt_region", "CN") or "CN").upper()
        self.cmb_vt_region.setCurrentIndex(0 if _rg != "TW" else 1)
        self.cmb_vt_region.setToolTip(
            "「成功面板」模板用哪一套（跟游戏客户端语言走 ✓）：\n"
            "  · **国服** = `liescc/cg.png`（默认 ✓）；\n"
            "  · **繁中** = `liescc/tw/tw-cg.png`。\n\n"
            "⚠ 选错不会有任何反应（永远等不到成功）—— 语言拿不准时先按默认开一次，\n"
            "   若日志里一直是 LOCKED 却从不出现 SUCCESS，再来这儿换 ✓。")
        self.cmb_vt_region.currentIndexChanged.connect(self._on_vt_changed)
        f.addRow("成功模板地区", self.cmb_vt_region)

        self.sp_vt_cool = NoWheelDoubleSpinBox()
        self.sp_vt_cool.setRange(1.0, 60.0)
        self.sp_vt_cool.setDecimals(1)
        self.sp_vt_cool.setSingleStep(1.0)
        self.sp_vt_cool.setValue(float(_lv.get("vt_cool_s", 3.0)))
        self.sp_vt_cool.setToolTip(
            "点完「确定」之后，隔多少秒才**重开一轮**（默认 **3.0 秒**）。\n\n"
            "为什么要它：成功面板关掉需要一两秒，立刻重开一轮的话，新一轮会在**旧画面**上\n"
            "又认出同一个面板 ⇒ **连点两下** ✗（可能把游戏的下一步操作也点掉）。\n\n"
            "  · 觉得开下一局太慢 ⇒ 调小（但别小于 1 秒）；\n"
            "  · 面板消失得慢（网络卡）⇒ 调大。")
        self.sp_vt_cool.valueChanged.connect(self._on_vt_changed)
        f.addRow("成功冷却（秒）", self.sp_vt_cool)

        note = QLabel("改完**最多 1 秒生效**（不用重开预览 ✓）。第一次启用会加载模型约 10 秒，"
                      "加载/推理都在独立线程，不影响实时画面。")
        note.setStyleSheet("color: #80868b;")
        note.setWordWrap(True)
        f.addRow("", note)
        # ⚠ **最后才开闸**（见 `_on_vt_changed` ✓）：上面那几行 `setValue/setCurrentIndex` 已经
        #   把初始值灌进去了（那几次 `valueChanged` 要**被闸挡住** ✓），从这一刻起人才改得动 ✓。
        self._vt_built = True
        return grp

    def set_vt_status(self, text):
        """「实时」页转发过来的测谎状态 ⇒ 写进组里那一行 ✓（见 `_build_vt_group` ✓）。"""
        try:
            self.lbl_vt.setText(str(text))
        except Exception:                    # noqa: BLE001 —— 显示而已，出错不许冒泡 ✓
            pass

    def _on_vt_changed(self, *_):
        """「自动测谎」组任一格改动 ⇒ 写回 `config/live.yaml`（实时线程热读 ✓）。

        ⚠⚠ **构造期不许写**（`_vt_built` 闸 ✓）：`setValue` / `setCurrentIndex` 在**建控件时**
          就会触发 `valueChanged` ✗ —— 那一刻后面几个控件还没建，读它们就是 `AttributeError`
          （被 except 吞掉、只留一句 ⚠），而且会**拿半套默认值先写一次盘** ✗。
        ⚠ 只写这几个键（`update_live` 是**合并**写 ✓）—— 别整份覆盖，否则会把
          `perf_log` / 录屏开关那些键冲掉 ✗（本仓库在这上面栽过，见 `live_panel._save_live_params` 的注释 ✓）。
        """
        if not getattr(self, "_vt_built", False):
            return
        try:
            update_live(vt_on=bool(self.ck_vt.isChecked()),
                        vt_click=bool(self.ck_vt_click.isChecked()),
                        vt_aim=bool(self.ck_vt_aim.isChecked()),
                        vt_draw=bool(self.ck_vt_draw.isChecked()),
                        vt_conf=float(self.sp_vt_conf.value()),
                        vt_region=str(self.cmb_vt_region.currentData() or "CN"),
                        vt_cool_s=float(self.sp_vt_cool.value()),
                        vt_aim_ms=int(self.sp_vt_aim_ms.value()))
        except Exception as e:                # noqa: BLE001 —— 存不上不该炸面板 ✓
            print("⚠ 自动测谎参数没存进 live.yaml（%s: %s）" % (type(e).__name__, e))

    def _build_antihang_group(self):
        """「防挂机」组：测谎/掉线弹窗出现时的**触发音效**（用户 2026-09-29 ✓）。"""
        grp = QGroupBox("防挂机")
        f = QFormLayout(grp)

        row = QHBoxLayout()
        self.ed_alarm_sound = QLineEdit(
            getattr(settings, "lie_alarm_sound",
                    "datasets/sound/Neotokyo.Effect.alert.mp3"))
        self.ed_alarm_sound.editingFinished.connect(self._on_alarm_sound)
        self.ed_alarm_sound.setToolTip(
            "测谎/掉线等弹窗出现时播放的音效文件（mp3/wav）。\n"
            "相对路径按仓库根解析；留空 = 系统提示音。")
        row.addWidget(self.ed_alarm_sound, 1)
        _b = QPushButton("浏览…")
        _b.clicked.connect(self._browse_alarm_sound)
        row.addWidget(_b)
        _p = QPushButton("试听")
        _p.setToolTip("播一遍当前配置的音效 ✓")
        _p.clicked.connect(self._preview_alarm_sound)
        row.addWidget(_p)
        f.addRow("触发音效", row)

        note = QLabel("测谎 / 掉线等弹窗出现时播放的提示音（mp3 / wav）；"
                      "留空或文件缺失时退回系统提示音。")
        note.setStyleSheet("color: #80868b;")
        note.setWordWrap(True)
        f.addRow("", note)
        return grp

    def _build_reconnect_group(self):
        """⭐⭐ 「断线重连」组（**主窗口 → 挂机保护页签** ✓ 用户 2026-10-07 点名的位置 ✓）。

        **2026-10-09 全搬过来**（用户原话："把「检测到断线后自动走回游戏」这几个开关+子参数
        也搬到「挂机保护 → 断线重连」组" ✓）：原来这一组只有「频道」一个参数 ✓，其余十格
        （总开关 / 探测延时 / 步骤超时 / 排队超时 / 重试上限 / 恢复自动 / 四个点击比例 ✓）
        全散在**设置弹窗 → 保护与恢复**里 ✗ ⇒ 断线这块东西要**跨两个窗口**找 ✗。
        ⚠ 仓库纪律是「**一处控件**」✓ ⇒ 是**搬**、不是复制 ✗（设置窗那份已删 ✓ 只留一行指路 ✓）。

        ⚠ 落点怎么算**不在这儿** ✗：`decision/reconnect.py::CHANNEL_GRID_*` 一处实现 ✓
          （界面只收一个整数 ✓ 省得两处口径分叉 ✗）。
        """
        grp = QGroupBox("断线重连")
        self._rc_group = grp                      # 给"只在看得见时才刷新标定那行"用 ✓
        f = QFormLayout(grp)

        # ---- ① 总开关 ----
        self.ck_reconnect = QCheckBox("检测到断线后自动走回游戏")
        self.ck_reconnect.setChecked(bool(settings.reconnect_enabled))
        self.ck_reconnect.setToolTip(
            "⚠ **判到断线界面就停止自动** —— 这一半永远生效（2026-09-30 ✓），不受本开关影响 ✓；\n"
            "本开关管的是要不要**自动按回车 / 点鼠标走回游戏** ✓。\n\n"
            "⭐ **自动本来就关着**时也会接手（2026-10-09 放宽 ✓ —— 常见现场：掉线把血条读空 ⇒\n"
            "  被当成「角色死亡」⇒ 自动被停 ✗ ⇒ 老口径「自动没开就不插手」⇒ 报警响了却什么都不做 ✗）：\n"
            "  · 判到**登录界面 / 断线提示框**（人正常玩**到不了**的界面 ✓）⇒ **照样接手** ✓；\n"
            "  · **选频道 / 选角 / 排队**（人自己换频道时也会到 ✓）⇒ 自动关着就**不抢** ✓\n"
            "    （抢了就是跟手动操作打架 ✗）。\n\n"
            "回到游戏后按下面「回到游戏后自动恢复自动打怪」决定要不要接着打 ✓\n"
            "（⚠ 还要过一趟「开自动前的检查」—— 条件没凑齐就**不恢复**，不弹窗，只写在状态栏 ✓）。")
        self.ck_reconnect.toggled.connect(self._on_reconnect_param)
        f.addRow("", self.ck_reconnect)

        # ---- ② 子参数（版式遵守 UI 规范 §9：一行一个、单位进标签、说明进 tooltip ✓）----
        self.sp_rc_probe = NoWheelDoubleSpinBox()
        self.sp_rc_probe.setRange(0.0, 600.0)
        self.sp_rc_probe.setDecimals(1)
        self.sp_rc_probe.setSingleStep(0.5)
        self.sp_rc_probe.setValue(float(settings.reconnect_probe_after_lost_sec))
        self.sp_rc_probe.valueChanged.connect(self._on_reconnect_param)
        self.sp_rc_probe.setToolTip(
            "玩家框丢多久之后开始判别界面（0 = 立刻）。\n\n"
            "默认 0：断线提示框只显示两三秒，等 5 秒再探就错过它了，客户端会一直卡在\n"
            "提示框上等人按确定。探一次只要约 4 ms，不必为省这点开销推迟。")
        f.addRow("丢失多久后开始探界面(s)", self.sp_rc_probe)

        self.sp_rc_step = NoWheelSpinBox()
        self.sp_rc_step.setRange(0, 60000)
        self.sp_rc_step.setSingleStep(500)
        self.sp_rc_step.setValue(int(settings.reconnect_step_timeout_ms))
        self.sp_rc_step.valueChanged.connect(self._on_reconnect_param)
        self.sp_rc_step.setToolTip(
            "重连每一步（点服务器 / 点频道 / 排队 / 选角）等「界面真的变了」的超时。\n"
            "到点还没变就按下面的次数重试。")
        f.addRow("每步等待超时(ms)", self.sp_rc_step)

        self.sp_rc_queue = NoWheelSpinBox()
        self.sp_rc_queue.setRange(0, 600000)
        self.sp_rc_queue.setSingleStep(1000)
        self.sp_rc_queue.setValue(int(settings.reconnect_queue_timeout_ms))
        self.sp_rc_queue.valueChanged.connect(self._on_reconnect_param)
        self.sp_rc_queue.setToolTip(
            "排队那一步专用的长超时：排队可能要等很久，用上面那个 3 秒会一直重试。")
        f.addRow("排队弹窗超时(ms)", self.sp_rc_queue)

        self.sp_rc_retry = NoWheelSpinBox()
        self.sp_rc_retry.setRange(0, 20)
        self.sp_rc_retry.setValue(int(settings.reconnect_max_retry))
        self.sp_rc_retry.valueChanged.connect(self._on_reconnect_param)
        self.sp_rc_retry.setToolTip(
            "同一步骤最多重试几次；到上限就停下并提示（不会无限重连）。")
        f.addRow("同一步最多重试(次)", self.sp_rc_retry)

        self.ck_rc_resume = QCheckBox("回到游戏后自动恢复自动打怪")
        self.ck_rc_resume.setChecked(bool(settings.reconnect_resume_auto))
        self.ck_rc_resume.toggled.connect(self._on_reconnect_param)
        self.ck_rc_resume.setToolTip(
            "重连成功、回到游戏画面之后，自动把「自动打怪」重新打开。\n\n"
            "⚠ 打开前会做一次「开自动前的检查」（**与手动开自动是同一份**：地图 / 标定\n"
            "几何 / 地形图）：条件没凑齐就**不恢复**，并在状态栏写清原因（不会弹窗 ——\n"
            "挂机时人往往不在屏幕前 ✓）。修好后自己点「开启自动」就行。\n\n"
            "不勾就停在「已回到游戏、自动仍是关的」，由你自己决定。\n"
            "⭐ 另外：**接手时自动本来就是关的**（掉线把自动停了 ✓ 见上 ✓）⇒ 回到游戏也**不会**\n"
            "自作主张打开 ✓（那属于越权 ✓ 2026-10-09 修的 ✓）。")
        f.addRow("", self.ck_rc_resume)

        # ---- ③ 「频道」（想进第几格 ✓ 用户 2026-10-07 点名要放这儿 ✓）----
        self.sp_rc_channel = NoWheelSpinBox()
        self.sp_rc_channel.setRange(1, 60)         # 1~60 ✓ 见 reconnect.CHANNEL_TOTAL ✓
        self.sp_rc_channel.setValue(int(getattr(settings, "reconnect_channel", 1) or 1))
        self.sp_rc_channel.setToolTip(
            "断线重连时想进**第几个频道**（1~60 ✓）。\n\n"
            "**1 = 面板左上角那一格** ✓，从左到右、从上到下数 ✓（顺序已真机验过 ✓）。\n"
            "一页看得见 20 格（4 列 × 5 行）✓ —— 第 **21~60** 个不在第一页里：\n"
            "工具会**先把光标放到列表上、再往下滚「需要的那几行」**，让目标正好落在\n"
            "**最下面一行**，然后点那一格（滚轮 ✓ 不会乱点 ✓）。\n"
            "例：**22 ⇒ 往下滚 1 行**（22 是第 6 行 ⇒ 滚 1 行后它落在最下面一行 ✓）；\n"
            "≤ 20 的频道在第一页里 ⇒ **一行都不滚** ✓。\n"
            "落点 = 那对「频道 X/Y 比例」（= 第 1 格的位置）＋ 格距 × 行号 ✓\n"
            "—— 存的是**比例** ⇒ 换分辨率不用重配 ✓。\n\n"
            "⚠ 一次**只发 1 格**（人也是这么滚的 ✓）：连发一大串滚轮会被客户端当成一下、\n"
            "  列表**根本不动** ⇒ 接着就点错频道（2026-10-10 真机踩过：up=20/down=5 ⇒ 点了频道 2 ✗）。\n"
            "⚠ 它**假定频道面板一打开就在顶部** ✓（真机如此 ✓）；要是列表在这之前被人手动滚过，\n"
            "  会点错行 ⇒ 重开一次频道面板即可 ✓。\n"
            "⚠ 服务器那一格不受这里影响（永远是第 1 个服务器 + 单击 ✓）。\n"
            "⚠ 填成超出 1~60 ⇒ **一个字节都不点**、只在状态栏说清 ✓（宁可不做，也不乱点 ✓）。")
        self.sp_rc_channel.valueChanged.connect(self._on_reconnect_channel)
        f.addRow("频道", self.sp_rc_channel)

        # ---- ④ 「点服务器 / 点频道」要点的**画面比例位置**（用户 2026-10-06 ✓）----
        #   **存比例（0~1）、不存像素** ✓ ⇒ 换分辨率 / 窗口大小自动适配（同 HP/MP 条那套 ✓）。
        #   默认值是 2026-10-06 **从断线素材里量出来的**（不是目测 ✗）：服务器第 1 格
        #   (698,245)@1920×1080、频道面板第 1 格 (788,550) ⇒ 见
        #   `decision/reconnect.py::CLICK_TARGETS` 上面那段说明 ✓。
        #   客户端布局不一样就改这里（重量：`python -X utf8 -m tools._probe_disc_frames
        #   --bars --t 15.0 --region 660,480,640,300` ✓）。
        def _ratio_spin(val, tip):
            w = NoWheelDoubleSpinBox()
            w.setRange(0.0, 1.0)
            w.setDecimals(4)
            w.setSingleStep(0.005)
            w.setValue(float(val))
            w.setToolTip(tip)
            w.valueChanged.connect(self._on_reconnect_param)
            return w

        self.sp_rc_srv_x = _ratio_spin(
            settings.reconnect_server_x,
            "「选择频道（服务器列表）」界面里，要点的那一格在**画面上的横向位置**，\n"
            "写成 0~1 的比例（0 = 最左、1 = 最右）。\n\n"
            "默认 0.3635 = 素材里「1.蓝蜗牛」那一格的中心 x=698（1920 宽）。\n"
            "⚠ 存比例不存像素：换分辨率 / 窗口大小不用重配 ✓。")
        f.addRow("服务器 X 比例", self.sp_rc_srv_x)
        self.sp_rc_srv_y = _ratio_spin(
            settings.reconnect_server_y,
            "同上，纵向比例（0 = 最上、1 = 最下）。\n\n"
            "默认 0.2074 = 真机校正过的**格子中心**（1080 高下 y≈224）。\n"
            "⚠ 旧默认 0.2269 是素材里**名字那一行**的中心（y=245）⇒ 落点偏下约 15px ✗"
            "（2026-10-07 真机真点发现的 ✓ 见设计文档 §11）。")
        f.addRow("服务器 Y 比例", self.sp_rc_srv_y)
        self.sp_rc_chan_x = _ratio_spin(
            settings.reconnect_channel_x,
            "频道面板弹出的界面里，要点的那一格（默认 = 第 1 格「频道1」）的横向比例。\n\n"
            "默认 0.4104 = 素材里第 1 格的中心 x=788。\n"
            "「频道」选了第 2~60 个时，落点 = 这一格 ＋ 格距 × 格号 ✓。")
        f.addRow("频道 X 比例", self.sp_rc_chan_x)
        self.sp_rc_chan_y = _ratio_spin(
            settings.reconnect_channel_y,
            "同上，纵向比例。默认 0.5093 = 素材里面板第 1 行的中心 y=550。")
        f.addRow("频道 Y 比例", self.sp_rc_chan_y)

        # ---- ⑤ 「鼠标标定」现状（**只读** ✓）：它是"能不能点"的另一半 ——
        #   比例再准，没标定也**一个字节都不会发**（`decision/mouse_aim.py` ✓）。
        #   放一行在这儿，是为了让"它为什么不点"**在界面上就能看见** ✓（否则只能去翻日志 ✗）。
        self.lbl_rc_gain = QLabel()
        self.lbl_rc_gain.setWordWrap(True)
        f.addRow("鼠标标定", self.lbl_rc_gain)
        self._refresh_rc_gain()

        # ---- ⑥ 总开关关着 ⇒ 子参数灰掉（改它没意义 ✓ 同「追击起跳」那一组的做法 ✓）----
        self.ck_reconnect.toggled.connect(self._rc_enable)
        self._rc_enable(self.ck_reconnect.isChecked())

        note = QLabel("断线时不读文字、不训模型（模板锚点判界面 ✓）。"
                      "「点服务器 / 点频道」要先撞到屏幕左上角再按比例走位 —— "
                      "所以**鼠标必须标定过**，没标定就会停在原地不动手（不会乱点 ✓）。")
        note.setStyleSheet("color: #80868b;")
        note.setWordWrap(True)
        f.addRow("", note)
        return grp

    def _rc_enable(self, on):
        """总开关关着 ⇒ 这一组子参数灰掉（`_sync_reconnect_group` 回填时也会调 ✓）。"""
        for w in (self.sp_rc_probe, self.sp_rc_step, self.sp_rc_queue,
                  self.sp_rc_retry, self.ck_rc_resume, self.sp_rc_channel,
                  self.sp_rc_srv_x, self.sp_rc_srv_y,
                  self.sp_rc_chan_x, self.sp_rc_chan_y):
            w.setEnabled(bool(on))

    def _on_reconnect_channel(self, _val=None):
        """「频道」改了 ⇒ 写回设置并落盘 ✓（与会话里的其它参数同一个做法 ✓）。"""
        settings.reconnect_channel = int(self.sp_rc_channel.value())
        settings.save()

    def _on_reconnect_param(self, *_a):
        """断线重连这一组**任何一个**参数改了 ⇒ 整组写回设置 + 落盘 ✓。

        为什么整组写、不按改动那个写：`settings` 是本进程唯一的真相 ✓，而这些控件的初值
        **就是从它来的** ✓（见 `_sync_reconnect_group` ✓）⇒ 整组回写的值 == 控件上的值 ✓
        不会串号 ✓；省掉十个一模一样的槽函数 ✓（少写一处就少一处"改了没落盘" ✗）。
        ⚠ 「频道」不走这儿 ✗ —— 它有自己的槽（`_on_reconnect_channel` ✓ 一处控件一处落盘 ✓）。
        """
        settings.reconnect_enabled = bool(self.ck_reconnect.isChecked())
        settings.reconnect_probe_after_lost_sec = float(self.sp_rc_probe.value())
        settings.reconnect_step_timeout_ms = int(self.sp_rc_step.value())
        settings.reconnect_queue_timeout_ms = int(self.sp_rc_queue.value())
        settings.reconnect_max_retry = int(self.sp_rc_retry.value())
        settings.reconnect_resume_auto = bool(self.ck_rc_resume.isChecked())
        settings.reconnect_server_x = float(self.sp_rc_srv_x.value())
        settings.reconnect_server_y = float(self.sp_rc_srv_y.value())
        settings.reconnect_channel_x = float(self.sp_rc_chan_x.value())
        settings.reconnect_channel_y = float(self.sp_rc_chan_y.value())
        settings.save()

    def _sync_reconnect_group(self):
        """打开项目 / 设置变了 ⇒ 把「断线重连」这组控件**按当前 settings 回填** ✓。

        ⚠⚠ **必须判 hasattr** ✗：`_sync_from_settings()` 在 `__init__` 里就会跑 ✓，而本页是
          **按需**才建（`build_protection_page` ✓ 主窗口 addTab 时才调 ✓）⇒ 那一刻这些控件
          还不存在 ✗（2026-10-07 真栽过：不判就直接 `AttributeError` ⇒ **整个面板建不起来** ✗✗）。
        ⚠ `blockSignals` 必须包住 ✗：`setValue` / `setChecked` 会触发信号 ⇒ 又走
          `_on_reconnect_param` ⇒ 把"回填动作"当成人改的**再写一遍盘** ✓（无害但白写 ✓
          更糟的是"打开项目"变成"改设置" ✗）。
        """
        if not hasattr(self, "sp_rc_channel"):
            return
        _ws = (self.ck_reconnect, self.sp_rc_probe, self.sp_rc_step, self.sp_rc_queue,
               self.sp_rc_retry, self.ck_rc_resume, self.sp_rc_channel,
               self.sp_rc_srv_x, self.sp_rc_srv_y, self.sp_rc_chan_x, self.sp_rc_chan_y)
        for _w in _ws:
            _w.blockSignals(True)
        try:
            self.ck_reconnect.setChecked(bool(settings.reconnect_enabled))
            self.sp_rc_probe.setValue(float(settings.reconnect_probe_after_lost_sec))
            self.sp_rc_step.setValue(int(settings.reconnect_step_timeout_ms))
            self.sp_rc_queue.setValue(int(settings.reconnect_queue_timeout_ms))
            self.sp_rc_retry.setValue(int(settings.reconnect_max_retry))
            self.ck_rc_resume.setChecked(bool(settings.reconnect_resume_auto))
            self.sp_rc_channel.setValue(int(getattr(settings, "reconnect_channel", 1) or 1))
            self.sp_rc_srv_x.setValue(float(settings.reconnect_server_x))
            self.sp_rc_srv_y.setValue(float(settings.reconnect_server_y))
            self.sp_rc_chan_x.setValue(float(settings.reconnect_channel_x))
            self.sp_rc_chan_y.setValue(float(settings.reconnect_channel_y))
            self._rc_enable(bool(settings.reconnect_enabled))
        finally:
            for _w in _ws:
                _w.blockSignals(False)

    def _refresh_rc_gain(self):
        """把「鼠标标定」现状写进那一行（**只读** ✓）。

        判据只有一处：`decision.mouse_aim` 的**文件在不在 + 两个分量是不是正数** ✓
        （与真正点击时读的是同一份 ✓）—— 这里**不许自己再读一遍那份 json** ✗
        （两处口径迟早分叉，本仓库踩过 ✓）。
        """
        try:
            from decision import mouse_aim
            g = mouse_aim.load_gain()
            if g:
                self.lbl_rc_gain.setText("已标定：%.4f / %.4f 画面像素每指令单位 ✓"
                                         % (float(g[0]), float(g[1])))
            else:
                self.lbl_rc_gain.setText(
                    "未标定 —— 点服务器 / 点频道这一步会**停在这里不动手**（不会乱点 ✗）。\n"
                    "⚠ 在**跑工作台这台机器**上跑（不是游戏机 ✗ —— 鼠标指令会经 relay 打到游戏机 ✓）：\n"
                    "    python -X utf8 -m tools.mouse_aim_calib\n"
                    "跑之前：① **先停掉实时预览**（标定和它抢同一个 UDP 端口 ✗）；\n"
                    "        ② 游戏停在**画面不动**的界面 + **鼠标指针可见**；\n"
                    "        ③ 游戏机指针速度 6/11、关掉「提高指针精确度」。")
        except Exception as e:                   # noqa: BLE001 —— 读不到就说读不到 ✓
            self.lbl_rc_gain.setText("读不到标定状态：%s" % e)

    def _on_alarm_sound(self):
        settings.lie_alarm_sound = self.ed_alarm_sound.text().strip()
        settings.save()

    def _browse_alarm_sound(self):
        from PyQt5.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(self, "选择触发音效", "",
                                           "音频 (*.mp3 *.wav);;所有文件 (*)")
        if p:
            self.ed_alarm_sound.setText(p)
            self._on_alarm_sound()

    def _preview_alarm_sound(self):
        play_sound(self.ed_alarm_sound.text().strip())

    def _toggle_preview(self):
        """试听 ⇄ 停止（2026-09-29 用户要求 ✓：点击后变「停止」，播放完或点停止再变回）。"""
        if self._previewing:
            stop_sound()
            self._set_preview_text("试听")
            return
        if play_sound(self.ed_alarm_sound.text().strip()):
            self._set_preview_text("停止")   # 真的在播才变 ✓（文件缺失只会 beep ✓）

    def _set_preview_text(self, text):
        self._previewing = (text == "停止")
        self._btn_preview.setText(text)

    def _on_player_state(self, state):
        """共享播放器的状态信号：播完（EndOfMedia）/ 被停 ⇒ 按钮弹回「试听」✓。"""
        from PyQt5.QtMultimedia import QMediaPlayer
        if state == QMediaPlayer.StoppedState and self._previewing:
            self._previewing = False
            self._btn_preview.setText("试听")

    def _on_debounce(self, _val=None):
        settings.debounce_conf = float(self.sp_debounce_conf.value())
        settings.debounce_ms = int(self.sp_debounce_ms.value())
        settings.save()

    def _on_big_mob(self, _val=None):
        settings.big_mob_enabled = bool(self.ck_big_mob.isChecked())
        settings.big_mob_ratio = float(self.sp_big_mob_ratio.value())
        settings.big_mob_range_px = int(self.sp_big_mob_range_px.value())
        self._apply_big_mob_enabled()
        settings.save()

    def _apply_big_mob_enabled(self):
        """「大怪优先」勾选开关 ⇒ 下面两个参数**可用 / 禁用**（用户 2026-10-01 ✓）。"""
        on = bool(self.ck_big_mob.isChecked())
        self.sp_big_mob_ratio.setEnabled(on)
        self.sp_big_mob_range_px.setEnabled(on)

    def _set_rows_visible(self, rows, on):
        """成对显隐「控件 + 它左边那格标签」。

        ⚠ `QFormLayout` 里"标题"和"控件"是**两样东西** ✗ ⇒ 只藏控件会留半个空标题
          （用户一眼就看出来 ✓ 同 `_refresh_strategy_ui` 里那几处 `_lbl_*` 的写法 ✓）。
        """
        on = bool(on)
        for w, lbl in (rows or ()):
            w.setVisible(on)
            if lbl is not None:
                lbl.setVisible(on)

    def _refresh_strategy_ui(self):
        """切说明 / 参数预览，并报告「**当前生效**的是哪一个」（用户 2026-10-05 ✓）。

        ⚠⚠ **参数显隐看的是"下拉选中的那个"**（= 预览 ✓ 让用户先配好站桩地点再启用 ✓）；
          而"正在跑的是哪一套"由最上面那行 `当前策略类型` 说清楚 —— 两者不一致时那行会
          **明写「还没启用」**（红字 ✓）⇒ 所以这**不是**在骗人 ✓（若参数跟着"生效的那个"走，
          用户在启用前根本配不了站桩地点 ✗）。
        ⛔ 参数显隐本身仍是**三态** ✗：原来"扫平台 / 其他"两态会让平台站桩下看到扫平台那三个
          **不生效**的参数 ✓。
        """
        sel = str(self.cmb_strategy.currentData() or "patrol")
        cur = str(getattr(settings, "strategy", "patrol") or "patrol")
        names = getattr(self, "STRATEGY_NAMES", {}) or {}
        same = (sel == cur)
        if same:
            self.lbl_strategy_cur.setText(
                "%s（已启用）" % names.get(cur, cur))
            self.lbl_strategy_cur.setStyleSheet("color: #0b8043;")
        else:
            self.lbl_strategy_cur.setText(
                "%s（正在生效）；下面选的是「%s」—— 还没启用，"
                "按右下角「启用」才生效"
                % (names.get(cur, cur), names.get(sel, sel)))
            self.lbl_strategy_cur.setStyleSheet("color: #c5221f;")
        if getattr(self, "btn_strategy_apply", None) is not None:
            self.btn_strategy_apply.setEnabled(not same)
        is_sweep = sel == "sweep"
        is_plat = sel == "platform"
        is_multi = sel == "multi"
        # ⚠ 说明行每次重设成**正常灰**（`_apply_strategy` 拦截时会把它染红 ✓ 见那儿 ✓）——
        #   不在这儿复原的话，红字会一直挂着（哪怕后来补好了点位 ✓）。
        self.lbl_strategy_note.setStyleSheet("color: #80868b;")
        if is_sweep:
            self.lbl_strategy_note.setText(
                "扫平台：优先朝向方向，背后一定距离内的怪按就近锁定；"
                "到集合边缘（「距离平台边缘回头」）或当前朝向没怪持续一段时间"
                "就换向。继承上/下阈值过滤。")
        elif is_plat:
            self.lbl_strategy_note.setText(
                "平台站桩：站在「站桩地点」打 —— 够不着的怪只在站点 x 范围内水平逼近、"
                "到边缘就停（不出平台、不跨层）；完全没怪就原地站住；"
                "「最大战斗时长」在这种策略下不生效。")
        elif is_multi:
            _n = len([x for x in (getattr(settings, "multi_spots", None) or [])
                      if isinstance(x, dict) and x])
            self.lbl_strategy_note.setText(
                "多点巡逻：在 %d 个点位之间**依次循环**走 —— **到了就直接去下一个**；"
                "路上/到点遇怪照打（攻击范围内有怪优先 attack ✓）。%s"
                % (_n, "" if _n >= 2 else "　⚠ 少于 2 个点位 ⇒ **不能启用**（先「选择点位…」）"))
        else:
            self.lbl_strategy_note.setText(
                "以角色脚底为基准：上阈值往上、下阈值往下，范围外的怪不追踪")
        # 扫平台那三个（只在 sweep 显示 ✓）
        # ⚠ `sp_*` 那三格 + 说明行（`lbl_sweep_hint` ✓）归这里；两行的"标签"就是勾选框 ✓
        #   ⇒ 它们走下面那个 `_lbl_*` 循环（`_lbl_turn_cd` / `_lbl_edge_turn` 就是勾选框本身 ✓）
        for w in (self.sp_turn_cd, self.sp_back_range, self.sp_edge_turn,
                  self.lbl_sweep_hint):
            w.setVisible(is_sweep)
        for a in ("_lbl_turn_cd", "_lbl_back_range", "_lbl_edge_turn"):
            lb = getattr(self, a, None)
            if lb is not None:
                lb.setVisible(is_sweep)
        # 「平台站桩」那几行（只在 platform 显示 ✓）
        self._set_rows_visible(getattr(self, "_station_rows", ()), is_plat)
        # ⭐ 「多点巡逻」那两行（只在 multi 显示 ✓ 用户 2026-10-06 ✓）
        self._set_rows_visible(getattr(self, "_multi_rows", ()), is_multi)
        self._refresh_station_texts()

    def set_element_picker_factory(self, fn):
        """把「地区选择」弹窗的**工厂**装进来（由主窗口从路线识别面板推 ✓ 同 `set_zone_sets`）。

        `fn(parent, multi, init) -> [地点…] | None`（`None` = 用户取消 ⇒ **不改设置** ✓）。
        ⚠ 为什么是"推一个工厂"：弹窗要**地图与集合**（`terrain` / `zones`），而本面板
          **不持有地图 id** ✗（集合按 map id 存 ✓）—— 这正是仓库里"面板间推送"那条老规矩
          （同 `set_zone_sets` / 当年的 `set_foothold_picker_factory` ✓；后者的"编辑战斗区域"
          已整块搬到路线识别页，**本功能是它之后新的一条跨面板推送** ✓）。
        """
        self._element_factory = fn

    def _refresh_station_texts(self):
        """「站桩地点」/「拾取掉落地区」两行摘要 + 拾取参数的**灰不灰** ✓（一处收口）。"""
        spot = settings.station_spot if isinstance(settings.station_spot, dict) else {}
        if spot:
            self.lbl_station_spot.setText(zones_mod.element_label(spot))
            self.lbl_station_spot.setStyleSheet("color: #80868b;")
        else:
            self.lbl_station_spot.setText("（未选）不选 = 这个策略什么都不做（站着不动）")
            self.lbl_station_spot.setStyleSheet("color: #c5221f;")
        on = bool(getattr(settings, "station_pickup_enabled", False))
        for w in (self.sp_pickup_min, self.sp_pickup_max, self.btn_pickup_spots):
            w.setEnabled(on)
        spots = [x for x in (getattr(settings, "station_pickup_spots", None) or [])
                 if isinstance(x, dict)]
        if not spots:
            txt = "（未选地区）"
        else:
            txt = " → ".join(zones_mod.element_label(x) for x in spots)
        try:
            cap = float(getattr(settings, "station_pickup_max_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            cap = 0.0
        if on and cap <= 0.0:
            txt += "　⚠ 时间上限 <= 0 ⇒ 整组不生效"
        elif on and not spots:
            txt += "　⚠ 没选掉落地区 ⇒ 整组不生效"
        self.lbl_pickup_spots.setText(txt)
        # ⭐ 「多点巡逻」的点位摘要也在这儿刷（用户 2026-10-06 ✓）——**一处收口** ✓：
        #   本方法已经被 `_refresh_strategy_ui` / 各写口 / `bind` 调着 ✓ 挂这儿不会漏 ✓。
        self._refresh_multi_texts()

    def _on_station_spot_pick(self):
        """「选择站桩地点」⇒ 开「地区选择」弹窗（**单选** ✓）⇒ 写回 + 刷摘要 ✓。"""
        got = self._open_element_picker(multi=False,
                                        init=[settings.station_spot]
                                        if isinstance(settings.station_spot, dict)
                                        and settings.station_spot else [])
        if got is None:
            return                               # 取消 ⇒ 一个字都不改 ✓
        settings.station_spot = dict(got[0]) if got else {}
        settings.save()
        self._refresh_station_texts()

    def _on_multi_spots_pick(self):
        """「选择点位」⇒ 开「地区选择」弹窗（**多选** ✓）⇒ 写回 + 刷摘要 ✓。

        与「选择掉落地区」**完全同款**（同一工厂 / 同一弹窗 / 同一套写回 ✓）—— 用户 2026-10-06 ✓
        第 1 条原话："选中后出现「选择点位」→ **平台站桩类型里已经实现了的通用地点选择弹窗**（多选）"✓。
        """
        got = self._open_element_picker(
            multi=True,
            init=[x for x in (getattr(settings, "multi_spots", None) or [])
                  if isinstance(x, dict)])
        if got is None:
            return
        settings.multi_spots = [dict(x) for x in got]
        settings.save()
        self._refresh_station_texts()       # 一处收口：它末尾会转调 `_refresh_multi_texts` ✓
        self._refresh_strategy_ui()         # 点位数量变了 ⇒ 那句"能不能启用"的说明跟着变 ✓

    def _refresh_multi_texts(self):
        """「点位」那行摘要（`顺序 → 顺序` ✓；**< 2 个** ⇒ 红字说明"不能启用" ✓）。"""
        spots = [x for x in (getattr(settings, "multi_spots", None) or [])
                 if isinstance(x, dict) and x]
        lbl = getattr(self, "lbl_multi_spots", None)
        if lbl is None:
            return
        if not spots:
            lbl.setText("（未选）至少要选 2 个点位，否则这个策略不能启用")
            lbl.setStyleSheet("color: #c5221f;")
            return
        txt = " → ".join(zones_mod.element_label(x) for x in spots)
        if len(spots) < 2:
            lbl.setText("%s　⚠ 只有 1 个 ⇒ **不能启用**（再选一个 ✓）" % txt)
            lbl.setStyleSheet("color: #c5221f;")
        else:
            lbl.setText("%s　（共 %d 个，走完回到第一个 ⇒ 循环 ✓）" % (txt, len(spots)))
            lbl.setStyleSheet("color: #80868b;")

    def _on_pickup_spots_pick(self):
        """「选择掉落地区」⇒ 开「地区选择」弹窗（**多选** ✓）⇒ 写回 + 刷摘要 ✓。"""
        got = self._open_element_picker(
            multi=True,
            init=[x for x in (getattr(settings, "station_pickup_spots", None) or [])
                  if isinstance(x, dict)])
        if got is None:
            return
        settings.station_pickup_spots = [dict(x) for x in got]
        settings.save()
        self._refresh_station_texts()

    def _open_element_picker(self, multi, init):
        """调那个**工厂**开弹窗；没装工厂（老环境 / 没打开项目）⇒ 说清怎么办 + `None` ✓。"""
        fn = getattr(self, "_element_factory", None)
        if not callable(fn):
            QMessageBox.information(
                self, "还没法选地点",
                "选地点要用**地图里的元素**，而现在没有地图可用。\n\n"
                "先在「路线识别」页**打开一个项目**（并在那页把地图/集合准备好）✓")
            return None
        try:
            return fn(self, bool(multi), init)
        except Exception as ex:                  # noqa: BLE001 —— 弹窗坏了别把面板弄崩 ✗
            QMessageBox.warning(self, "选地点失败", "打开「地区选择」时出错：%s" % ex)
            return None

    def _on_pickup_enabled(self, _state=None):
        """「定时拾取掉落」开关 ⇒ 写回 + 刷参数灰不灰 ✓。"""
        settings.station_pickup_enabled = bool(self.ck_pickup.isChecked())
        settings.save()
        self._refresh_station_texts()

    def _on_pickup_time(self, _val=None):
        """「拾取掉落时间(s)」A~B 任一变化 ⇒ 写回 + 夹下限（A 不许大于 B ✓）。"""
        settings.station_pickup_min_s = float(self.sp_pickup_min.value())
        settings.station_pickup_max_s = float(self.sp_pickup_max.value())
        if settings.station_pickup_max_s < settings.station_pickup_min_s:
            settings.station_pickup_max_s = settings.station_pickup_min_s
        settings.save()
        self._refresh_station_texts()

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

    # ⚠ `_on_auto_feed`（自动喂宠开关的槽）**已随功能一起移除**（用户 2026-09-26）——
    #    想喂宠就用「自定义定时行为」配一条 ✓（按键那层的「喂宠」键还在 ✓）。

    def _tick_timer_cd(self):
        """每秒刷新**自定义定时行为**的倒计时（读 agent 维护的下次触发时刻）。

        ⚠ 名字原来是 `_tick_feed_cd`（那会儿它还管喂宠）：「自动喂宠」2026-09-26 整块移除后
          它就只管定时行为了 ⇒ 一并改名，别留着旧名字误导人 ✗。
        """
        import time
        now = time.monotonic()
        # 自定义定时行为倒计时
        for name, lbl in self._timer_cd_labels.items():
            t = next((x for x in settings.custom_timers if x.get("name") == name), None)
            paused = bool(t.get("paused")) if t else False
            if paused:
                # 暂停时倒计时冻在"停下那一刻的剩余"上（agent 那边已经不再排期）
                remaining = max(0.0, float(t.get("paused_left") or 0.0))
            else:
                next_ts = settings.custom_timer_next.get(name, 0.0)
                if next_ts > 0:
                    remaining = next_ts - now
                else:
                    lo = (t.get("interval", [5, 10])[0] if t else 5) * 60.0
                    remaining = lo
                if remaining < 0:
                    remaining = 0
            m = int(remaining) // 60
            s = int(remaining) % 60
            lbl.setText("距离下次：%d:%02d%s"
                        % (m, s, "（已暂停）" if paused else ""))
            # 样式只在状态变化时设一次（每秒重复 setStyleSheet 会白白触发重绘）
            if self._timer_cd_state.get(name) != paused:
                self._timer_cd_state[name] = paused
                lbl.setStyleSheet("color:%s;" % ("#c5221f" if paused
                                                 else "#80868b"))

    # ---------------- 自定义定时行为 ----------------

    def _refresh_timers(self):
        """重建自定义定时行为列表。"""
        self._clear_layout(self._timer_list)
        self._timer_cd_labels.clear()
        self._timer_cd_state.clear()
        self._timer_pause_btns.clear()
        self._timer_fire_btns = {}       # {name: 「手动触发」按钮}
        for t in settings.custom_timers:
            name = t.get("name", "")
            lo, hi = t.get("interval", [5, 10])
            paused = bool(t.get("paused"))
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
            # 「手动触发」（2026-09-26 用户要求）：立刻演一次 + 计时从头开始。
            # 颜色**故意和旁边三个不一样**（琥珀色）—— 它是"会让角色立刻动起来"的那一个，
            # 混在「暂停 / 编辑 / 删除」里最容易点错（用户明确要求"换个按钮颜色"）。
            btn_fire = QPushButton("手动触发")
            btn_fire.setFixedWidth(78)
            btn_fire.setStyleSheet(
                "QPushButton { background:#fef7e0; color:#a05c00;"
                " border:1px solid #fadf8e; border-radius:5px; font-weight:600; }"
                "QPushButton:hover { background:#feefc3; }"
                "QPushButton:disabled { color:#9aa0a6; background:#f8f9fa;"
                " border-color:#dadce0; }")
            btn_fire.setToolTip(
                "**立刻演一次**这个行为，并把**计时从头开始**"
                "（下一次触发从这一刻重新随机）。\n\n"
                "灰着 = 现在点不了：自动没开（序列活不下来），"
                "或者这一项已暂停（先点「继续」）。")
            btn_fire.clicked.connect(lambda _c, n=name: self._fire_timer(n))
            self._timer_fire_btns[name] = btn_fire
            row.addWidget(btn_fire)
            # 暂停 / 继续：把某一个定时行为单独停掉（其它的照常），
            # 点「继续」从停下的地方接着倒计时
            btn_pause = QPushButton("继续" if paused else "暂停")
            btn_pause.setFixedWidth(52)
            btn_pause.setToolTip(
                "暂停：这个行为不再触发（正在演的序列立刻停下并松开按键），\n"
                "倒计时冻在当前剩余时间上（红字 +「已暂停」）。\n"
                "其它的定时行为不受影响。" if not paused else
                "继续：从暂停时的剩余时间接着倒计时。")
            btn_pause.clicked.connect(lambda _c, n=name: self._toggle_timer_pause(n))
            self._timer_pause_btns[name] = btn_pause
            row.addWidget(btn_pause)
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
        # 有定时行为就得让倒计时在跑：新增/编辑后也保证这一秒表在转
        # （以前只在"打开项目时已有定时行为"和"开喂宠"时启动，新增完不会动）
        if settings.custom_timers and not self._cd_timer.isActive():
            self._cd_timer.start()
        self._tick_timer_cd()
        self._sync_timer_fire_btns()

    def _fire_timer(self, name):
        """手动触发一次这个定时行为（**计时从头开始**，用户 2026-09-26 要求）。

        为什么只往 `settings.custom_timer_fire` 里塞个名字：序列只能在**实时线程**里
        演（按键、计时都在 agent 那边），界面里直接演会跟主回路抢按键 ✗ —— 走仓库里
        已有的"请求通道"做法（同 `rest_request` / `rest_abort` ✓）。
        """
        if not settings.enabled:
            return          # 按钮本来就是灰的（自动关着时序列活不过一帧）
        settings.custom_timer_fire.append(str(name))

    def _sync_timer_fire_btns(self):
        """「手动触发」按钮的可用性：**自动没开 / 这一项已暂停**时点不了。

        判据和 agent 那边**对齐**（不然就是"点了没反应" ✗）：
          · 自动没开 ⇒ tick 每拍都会 `_release_held_keys()`（它会把 `_timer_states`
            全清掉）⇒ 序列活不过一帧；
          · 已暂停 ⇒ agent 那边本来就跳过这一项（见 `_custom_timers` 的 paused 分支）。
        """
        for name, btn in getattr(self, "_timer_fire_btns", {}).items():
            t = next((x for x in settings.custom_timers
                      if x.get("name") == name), None)
            btn.setEnabled(bool(settings.enabled)
                           and not bool((t or {}).get("paused")))

    def _toggle_timer_pause(self, name):
        """暂停 / 继续一个自定义定时行为。

        暂停：不再触发 + 正在演的序列停下（agent 那边处理），倒计时冻住；
        继续：把冻住的剩余时间写回 `custom_timer_next`，agent 看到 next > 0
        就不会重新随机 —— 也就是**接着走**，而不是把等待清零从头开始。
        """
        import time
        t = next((x for x in settings.custom_timers if x.get("name") == name), None)
        if t is None:
            return
        now = time.monotonic()
        if t.get("paused"):
            left = max(0.0, float(t.pop("paused_left", 0.0) or 0.0))
            t["paused"] = False
            settings.custom_timer_next[name] = now + left
        else:
            next_ts = settings.custom_timer_next.get(name, 0.0)
            if next_ts > now:
                left = next_ts - now
            else:
                # agent 还没排期（刚加/刚编辑过）：按区间下限占位，别显示 0:00
                lo = (t.get("interval") or [5, 10])[0]
                left = max(0.0, float(lo) * 60.0)
            t["paused"] = True
            t["paused_left"] = left
            settings.custom_timer_next.pop(name, None)
        settings.save()
        self._refresh_timers()

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
        # 「休息时暂停计时」（2026-09-26 新增）：编辑窗里那个勾 ⇒ 写回这一条 ✓
        t["pause_on_rest"] = bool(r.get("pause_on_rest", True))
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

    # ⚠ `_on_feed_interval`（喂宠间隔的槽）**已随功能一起移除**（用户 2026-09-26）。

    def _pick_bar(self, kind):
        """在实时画面上框选 HP/MP 条区域，并采样该条的填充色。

        框选走全仓库唯一那份实现（带放大镜，见 docs/UI规范.md §8）：血条上沿
        差一两个像素，采样就会吃到背景色。条又细又长，所以起始倍数给到 12×。
        """
        from gui.region_selector import select_region_on_image
        frame = None
        lp = getattr(self, "live_panel", None)
        if lp is not None:
            frame = lp.current_frame()
        if frame is None:
            QMessageBox.information(
                self, "提示", "请先在「实时」页开始预览、看到画面后，再框选 HP/MP 条")
            return
        rect = select_region_on_image(frame, self, zoom=12)
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

        HP/MP 条：项目存过就用项目的；没存过回退到**最近打开的那个项目**那份
        （= 你上一次框选 / 加载模板的位置）—— 同一套游戏 UI 通常通用，
        一律清空会逼着每个项目都重框一次。顺序上「最近打开的项目」是主窗口在
        本函数**之后**才更新的，所以这里拿到的正是上一个项目那套。
        """
        self.project = project
        proj = (project.get("bars") if project is not None else None) or {}
        prev = last_opened()
        fallback = (prev.get("bars") if prev is not None else None) or {}

        def pick(key, saved_key):
            return proj.get(key) or fallback.get(saved_key)

        settings.hp_bar = load_rect(pick("hp", "hp_bar"))
        settings.hp_color = pick("hp_color", "hp_color")
        settings.mp_bar = load_rect(pick("mp", "mp_bar"))
        settings.mp_color = pick("mp_color", "mp_color")
        self._sync_bar_ui()
        # 决策参数已经换成当前项目那份了，把控件重新回填一遍（切项目必走）
        self._sync_from_settings()
        # ⭐ 顶部那行"存到哪儿"也跟着换（切项目 / 关掉项目都要重说一遍 ✓ 别留着上一句 ✗）
        self._update_save_hint()

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
        """实时更新识别出的血/蓝比例。

        ⭐ 血条读出来是 **0** ⇒ 顺手把「自动喝药已停用」那行亮出来（用户 2026-10-05 ✓）——
        判据**与 Agent 那道闸同源**：都是"血条读空"（`agent._pot_off_empty` ✓
        `ws.player.dead` ✓ 见 `_drink_potions` ✓），界面层不另立规则 ✗。
        """
        self.pb_hp.setValue(int(hp * 100))
        self.pb_mp.setValue(int(mp * 100))
        # ⚠ **一处判据、一处显示**（"血条读出来是 0" ✓ 与 Agent 那道闸同源 ✓）：
        #   红字**盖在 HP 条上**（`lbl_hp_off` ✓ **不占任何行** ✓ 用户 2026-10-05 第二轮要求 ✓）——
        #   原来两处各一行（`lbl_pot_off` / `lbl_timers_off` ✗）⇒ 血条一闪就**多一行少一行** ✗。
        #   ⚠ 亮的时候把条自带的 `HP xx%` **让开** ✓：不然两段字会叠在一起（红字压着自己的百分比 ✓）。
        _off = float(hp) <= 0.0
        self.pb_hp.setTextVisible(not _off)
        self.lbl_hp_off.setVisible(_off)

    def _on_input_device(self, _idx=None):
        dev = self.cmb_device.currentData()
        settings.input_device = dev
        settings.save()
        self._apply_input_device(dev)

    def _apply_input_device(self, dev):
        """切换输入设备：本地 SendInput / 远程 Pro Micro / 本地 Pro Micro（后台异步连接）。

        ⭐⭐ 2026-10-05（用户"**从源头彻底封死**"✓）：**"起线程去连被控机"这件事挪进了
          `decision.input.connect_async`** —— 那儿是**唯一**去连被控机的地方，也是**闸**
          （`input.net_allowed` ✓）所在 ⇒ 谁调都绕不过去 ✓。
          原来两个 `threading.Thread(...)` 就写在本方法里 ✗ ⇒ 只有**记得打桩的那个套件**
          （`selftest_main_window` ✓）才不连，`gui_smoke` / `minimap` / `screen_state` /
          `decision` 里那几个 `PlayerPanel()` 都照样连 ✗ ⇒ 那条线程回来碰已销毁的对象 ⇒
          **偶发原生崩 `0xC0000005`** ✗（病根与判据见 `decision/input.py::net_allowed` ✓）。
        """
        from decision import input as dinput
        if dev in ("remote", "serial") and not dinput.net_allowed():
            # ⭐ 离屏自检 / 显式关闸：**连线程都不起**、也不发（`use_blocked` ✓ 什么都不发）。
            # ⚠ 这里**绝不 emit("fail")**：`_on_device_connected` 收到非 ok 会把
            #   `settings.input_device` 写回 `local` 并 `save()` ✗（自检写脏用户配置 ✗），
            #   而且那会把"离屏跳过"说成"连接失败"—— 两件事不一样 ✓。
            dinput.use_blocked(dev)
            self.lbl_device_state.setText("已跳过：离屏运行，不连被控机")
            self._refresh_mouse_ui()
            return
        if dev == "remote":
            from core.config import get
            host = get("kbd", "host")
            port = int(get("kbd", "port", 9000))
            cert = get("kbd", "cert", "remote_kbd/certs/cert.pem")
            self.lbl_device_state.setText("正在连接 ProMicro(远程)…")
            dinput.connect_async("remote", self._on_device_link_result,
                                 host=host, port=port, cert=cert)
        elif dev == "serial":
            from core.config import get
            ser_port = get("kbd", "serial_local", "COM5")
            self.lbl_device_state.setText("正在连接 ProMicro(本地)…")
            dinput.connect_async("serial", self._on_device_link_result, port=ser_port)
        else:
            dinput.use_local()
            self.lbl_device_state.setText("本地键盘")
        self._refresh_mouse_ui()

    def _on_device_link_result(self, status):
        """后台连接的回调（**在别的线程里**跑 ✓）：连不上就回退本地，再把结果发回主线程。

        ⚠⚠ **`emit` 必须兜住 `RuntimeError`**（2026-09-28 查明 ✓ 这是**真实隐患**，
          不只在自检里）：这条线程在**连不上时要等好几秒超时**（TCP ✓），而那时
          `PlayerPanel` 的 C++ 对象**可能已经销毁**了 —— 用户**把工作台关了** /
          离屏自检**跑完了** ✓ ⇒ `self.device_connected.emit(...)` 抛
          `RuntimeError: wrapped C/C++ object of type PlayerPanel has been deleted` ✗
          ⇒ 而它**在后台线程里、没人接** ⇒ **进程直接段错误 `0xC0000005`** ✗
          （症状极迷惑：自检"所有断言都过了"却崩、`faulthandler` 只给别的线程栈 ✓，
          而且它是**时机性**的 —— 对面秒拒就连不上、碰巧不崩 ✗）。

        ⚠ 2026-10-05：这段原来在 `remote` / `serial` 两条路里**各抄一份** ✗ ⇒ 现在**一处** ✓
          （`status` 只有 `"ok"` / `"fail"`：`"skip"`（离屏不连）在 `_apply_input_device`
          里就返回了、**不走这条** ✓）。
        """
        from decision import input as dinput
        if status != "ok":
            dinput.use_local()          # 连不上 ⇒ 回退本地（老口径一字不变 ✓）
        try:
            self.device_connected.emit(status)
        except RuntimeError:
            pass                        # 窗口已销毁 ⇒ 没人听，别崩 ✓

    def _on_device_connected(self, result):
        """Pro Micro 连接结果回调（后台线程 → 主线程）。"""
        if result == "ok":
            if settings.input_device == "serial":
                self.lbl_device_state.setText("已连接 ProMicro(本地)")
            else:
                self.lbl_device_state.setText("已连接 ProMicro(远程)")
        else:
            # ⚠⚠ **把原因一起说出来** ✗（2026-10-10 ✓ 用户现场："本地ProMicro连接失败" ✗
            #   而真正原因是**串口被另一个工作台占着** ✓ —— 界面只有八个字，等于没说 ✓
            #   见 `decision.input.last_connect_err` 的说明 ✓）。
            from decision import input as _din
            _why = str(_din.last_connect_err() or "").strip()
            settings.input_device = "local"
            settings.save()
            self.lbl_device_state.setText(
                "Pro Micro 连接失败，已回退本地" + (" —— %s" % _why if _why else ""))
            # ⚠ 长原因挂 tooltip ✓（状态栏一行放不下全部 ✓ 但别让它消失 ✗）
            self.lbl_device_state.setToolTip(_why)
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

        # 攻击范围那两条**竖边**（2026-09-27 新增；老项目里是 0 = 不限 ✓）
        for sp, v in ((self.sp_attack_up, settings.attack_up_dist),
                      (self.sp_attack_down, settings.attack_down_dist)):
            sp.blockSignals(True)
            sp.setValue(int(v))
            sp.blockSignals(False)

        # ⭐ 「**扇形角度**」（度；用户 2026-10-04 ✓ 见它的 tooltip ✓）：
        #   ⚠ `getattr` 兜底 0.0 —— 老配置 / 半初始化的 settings 里可能还没这个字段 ✓
        #   （= 矩形 = 老行为 ✓，界面与判定都不变 ✓）。
        self.sp_attack_fan.blockSignals(True)
        self.sp_attack_fan.setValue(
            float(getattr(settings, "attack_fan_deg", 0.0) or 0.0))
        self.sp_attack_fan.blockSignals(False)

        # ⭐ 「走不动按跳」（用户 2026-10-06 ✓）：老项目没这个键 ⇒ 显示 **True**（= 现在行为 ✓）
        self.ck_walk_hop.setChecked(bool(getattr(settings, "walk_stall_jump", True)))
        # ⭐ 「自动拾取」（用户 2026-10-06 ✓）：老项目没这个键 ⇒ 显示 **False**（= 关着 ✓
        #   和 `DecisionSettings.from_dict` 的兜底同一个方向 ✓ 别一个显示开、一个读成关 ✗）
        self.ck_auto_pickup.blockSignals(True)
        self.ck_auto_pickup.setChecked(bool(getattr(settings, "auto_pickup", False)))
        self.ck_auto_pickup.blockSignals(False)
        self.ck_chase_jump.blockSignals(True)
        self.ck_chase_jump.setChecked(bool(settings.chase_jump_enabled))
        self.ck_chase_jump.blockSignals(False)
        for sp, v in ((self.sp_chase_jump_min, settings.chase_jump_min),
                      (self.sp_chase_jump_max, settings.chase_jump_max),
                      (self.sp_chase_dash, settings.chase_jump_dash_ms)):
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

        # ⭐ 「攻击目标数量」（用户 2026-10-06 ✓）—— `getattr` 兜底 **-1** ✓（**老配置没有这个键**
        #   ⇒ 显示 -1 = 不限制 ✓ 与 agent 那边的兜底一致 ✓，**不是 0** ✗：0 = 不启用 = 会把
        #   老项目已经调好的「目标被攻击CD」关掉 ✓）。
        self.sp_atk_target_count.blockSignals(True)
        self.sp_atk_target_count.setValue(
            int(getattr(settings, "atk_target_count", -1)))
        self.sp_atk_target_count.blockSignals(False)

        # ⭐ 「目标被攻击CD」（用户 2026-10-06 ✓）—— `getattr` 兜底 0 ✓（**老配置没有这个键**
        #   ⇒ 显示 0 = 不限制 ✓ 与 agent 那边的兜底一致 ✓）。
        self.sp_mob_atk_cd.blockSignals(True)
        self.sp_mob_atk_cd.setValue(int(getattr(settings, "mob_atk_cd_ms", 0) or 0))
        self.sp_mob_atk_cd.blockSignals(False)
        # ⚠ 回填完**必须**跟着刷一次灰/亮（上面那格是 0 的话，这一格得是灰的 ✓ —— 只回填值、
        #   不刷灰亮，就会出现"界面上是亮的、其实不生效"✗ 约定 121 的教训 ✓）。
        self._sync_mob_cd_enabled()

        self._refresh_key_buttons()

        self.sp_attack_cd.blockSignals(True)
        self.sp_attack_cd.setValue(settings.attack_cd)
        self.sp_attack_cd.blockSignals(False)

        self.sp_attack_lock_db.blockSignals(True)
        self.sp_attack_lock_db.setValue(settings.attack_lock_debounce_ms)
        self.sp_attack_lock_db.blockSignals(False)

        self.sp_track_jump.blockSignals(True)
        self.sp_track_jump.setValue(settings.player_track_jump)
        self.sp_track_jump.blockSignals(False)

        # ⚠ 「玩家位置」（脚底偏移 / 框面积闸）已搬到「**设置 → 判定参数页**」✓，
        #   不在这儿回填（那几项的初值在 `settings_dialog._page_judge` 里现读 settings ✓）。
        # ⚠ 箭头（颜色 / 粗细 / 长度）已搬到「设置 → 界面 → 辅助线与标记（实时预览）」✓，
        #   不在这儿回填（本机外观那套在 `settings_dialog` 里读 `theme.load_vis()` ✓）。

        self.sp_min_turn_hold.blockSignals(True)
        self.sp_min_turn_hold.setValue(settings.min_turn_hold_ms)
        self.sp_min_turn_hold.blockSignals(False)
        # ⭐ 「站桩补朝向间隔(ms)」（2026-10-02 ✓）—— **切项目 / 加载模板之后必须回填** ✓
        #   （不回填的话界面还显示上一个项目的值 ⇒ 一改就把新项目的值写错 ✗）。
        if hasattr(self, "sp_station_turn_iv"):
            self.sp_station_turn_iv.blockSignals(True)
            self.sp_station_turn_iv.setValue(
                int(getattr(settings, "station_turn_interval_ms", 1000) or 0))
            self.sp_station_turn_iv.blockSignals(False)

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
        self.sp_edge_turn.blockSignals(True)
        self.sp_edge_turn.setValue(getattr(settings, "sweep_edge_turn_px", 100))
        self.sp_edge_turn.blockSignals(False)
        # ⭐ 两个勾：回填 + 同步"格子置灰 / 说明行"（用户 2026-10-09 ✓ 见 `_sync_sweep_switches` ✓）
        #   ⚠ 一律 `blockSignals`：摆值**不是**用户操作 ⇒ 不许顺手落盘 ✗（本页那条纪律 ✓）。
        for _ck, _attr, _dflt in ((self.ck_turn_cd, "sweep_turn_cd_enabled", True),
                                  (self.ck_edge_turn, "sweep_edge_turn_enabled", True)):
            _ck.blockSignals(True)
            _ck.setChecked(bool(getattr(settings, _attr, _dflt)))
            _ck.blockSignals(False)
        self._sync_sweep_switches()

        self.sp_debounce_conf.blockSignals(True)
        self.sp_debounce_conf.setValue(settings.debounce_conf)
        self.sp_debounce_conf.blockSignals(False)

        self.sp_debounce_ms.blockSignals(True)
        self.sp_debounce_ms.setValue(settings.debounce_ms)
        self.sp_debounce_ms.blockSignals(False)

        self.sp_big_mob_ratio.blockSignals(True)
        self.sp_big_mob_ratio.setValue(getattr(settings, "big_mob_ratio", 1.5))
        self.sp_big_mob_ratio.blockSignals(False)

        # ⭐⭐ 「断线重连」整组（挂机保护页 → 断线重连 ✓）：打开项目时得**跟着项目值走** ✓
        #   （⚠ 不接这一步的后果很具体：界面上显示的还是上一个项目的数 / 默认值 ✗
        #     —— 而真正花的是 `settings.reconnect_*` ✓ ⇒ "看到的不是用的" ✗）
        # ⚠⚠ **实现里必须判 hasattr** ✗：`_sync_from_settings()` 是在 `__init__` 里调的 ✓，
        #   而「挂机保护页」是**按需**才建（`build_protection_page` ✓ 主窗口 addTab 时才调 ✓）
        #   ⇒ 这一步跑的时候那些控件可能**还不存在** ✗
        #   —— 2026-10-07 真栽过：不判就直接 `AttributeError` ⇒ **整个面板建不起来** ✗✗
        #   （而控件建好之后它照旧会被同步 ✓ 见 `_build_reconnect_group` 里的初值 ✓）。
        self._sync_reconnect_group()

        self.sp_big_mob_range_px.blockSignals(True)
        self.sp_big_mob_range_px.setValue(getattr(settings, "big_mob_range_px", 600))
        self.sp_big_mob_range_px.blockSignals(False)

        # ⭐ 「平台站桩」三组（用户 2026-10-05 ✓）：拾取开关 / 时间区间 回填；
        #   **站桩地点与掉落地区是"结构体"**（不是控件自己存得下的）⇒ 由
        #   `_refresh_strategy_ui() → _refresh_station_texts()` 现读现画 ✓（一处口径 ✓）。
        self.ck_pickup.blockSignals(True)
        self.ck_pickup.setChecked(bool(getattr(settings, "station_pickup_enabled", False)))
        self.ck_pickup.blockSignals(False)
        self.sp_pickup_min.blockSignals(True)
        self.sp_pickup_min.setValue(float(getattr(settings, "station_pickup_min_s", 0.0) or 0.0))
        self.sp_pickup_min.blockSignals(False)
        self.sp_pickup_max.blockSignals(True)
        self.sp_pickup_max.setValue(float(getattr(settings, "station_pickup_max_s", 0.0) or 0.0))
        self.sp_pickup_max.blockSignals(False)

        self._refresh_strategy_ui()

        self.ck_auto_hp.blockSignals(True)
        self.ck_auto_hp.setChecked(settings.auto_hp_pot)
        self.ck_auto_hp.blockSignals(False)

        self.ck_auto_mp.blockSignals(True)
        self.ck_auto_mp.setChecked(settings.auto_mp_pot)
        self.ck_auto_mp.blockSignals(False)

        # ⚠ 「自动喂宠」的控件回填**已移除**（2026-09-26 整块删除）—— 别再加回来 ✗。

        # 自定义定时行为列表
        self._refresh_timers()
        if settings.custom_timers:
            self._cd_timer.start()

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

        self.ck_afk_retry.blockSignals(True)
        self.ck_afk_retry.setChecked(bool(settings.anti_afk_retry_on_interrupt))
        self.ck_afk_retry.blockSignals(False)

        self.sp_afk_retry.blockSignals(True)
        self.sp_afk_retry.setValue(max(1.0, float(settings.anti_afk_retry_sec)))
        self.sp_afk_retry.blockSignals(False)

        # ⭐ 「休息过程中循环行为」（用户 2026-09-28 ✓）—— 回填三件 + 缩进那块显隐 ✓
        self.ck_spot_loop.blockSignals(True)
        self.ck_spot_loop.setChecked(bool(settings.anti_afk_spot_loop))
        self.ck_spot_loop.blockSignals(False)
        self.sp_loop_min.blockSignals(True)
        self.sp_loop_min.setValue(float(settings.anti_afk_spot_loop_min))
        self.sp_loop_min.blockSignals(False)
        self.sp_loop_max.blockSignals(True)
        self.sp_loop_max.setValue(float(settings.anti_afk_spot_loop_max))
        self.sp_loop_max.blockSignals(False)
        # ⭐ 「休息结束推迟到循环执行完」（用户 2026-09-28 ✓）—— 回填（老项目里没这格 ⇒ 关 ✓）
        self.ck_loop_hold.blockSignals(True)
        self.ck_loop_hold.setChecked(bool(settings.anti_afk_spot_loop_hold))
        self.ck_loop_hold.blockSignals(False)

        if hasattr(self, "ck_loop_items"):
            self._refresh_loop_list()
        self._refresh_afk_ui()
        self._refresh_spot_loop_ui()

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
        # ⭐⭐ **顺手收掉小地图收流线程**（2026-10-09 ✓）：它们是 **daemon** ✓ 且常年阻塞在
        #   `recv` 上 ✗ —— 不显式停，解释器退出时脚下的 socket 缓冲会被拆 ⇒ 偶发
        #   **access violation（`0xC0000005`）** ✗（"关工作台时不清不楚地崩一下" ✓）。
        #   一处出口统一收（同套件收尾那条 ✓ 见 `perception.minimap.stop_all_clients` ✓）。
        try:
            from perception.minimap import stop_all_clients
            stop_all_clients()
        except Exception:                        # noqa: BLE001 —— 收尾不该拦住关窗 ✓
            pass
        # ⭐⭐ 触控板这两条线程都要收掉，**顺序不能反**（用户 2026-10-05 ✓）：
        #   ① 先停**跟踪线程**（`pad-track` ✓ 它每拍都在往发送器里 push ✓）；
        #   ② 再停**发送线程**（`pad-send` ✓）。
        #   ⚠ 反了的话：发送器已经没了、跟踪线程还在 push ⇒ 又会惰性造一个出来 ✗
        #     （那就是"关了还留着一条守护线程" ✓ 正是这两条注释要防的事 ✓）。
        try:
            self.touchpad.set_active(False)      # ⇒ `_deactivate` ⇒ `_stop_track` ✓
        except Exception:                        # noqa: BLE001 —— 收尾别因小失大 ✗
            pass
        _ps = getattr(self, "_pad_sender", None)
        if _ps is not None:
            _ps.stop()
            self._pad_sender = None
        # 释放所有按键 + 关闭远程键盘连接，停止指令传输
        self._force_release_all()
        from decision import input as dinput
        dinput.shutdown()
        from decision import manual_input
        manual_input.stop()
        if self.thread is not None:
            self.thread.wait(2000)
