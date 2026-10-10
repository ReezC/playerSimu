"""键盘模拟：键名映射 + SendInput 底层发送。

游戏通常读 GetAsyncKeyState / DirectInput。SendInput 的虚拟键码对
GetAsyncKeyState 有效（大多数 ARPG）；个别用 DirectInput 读扫描码的
游戏才需要 KEYEVENTF_SCANCODE，这里先走虚拟键码，遇到不生效再换。

键名用可读字符串（"left"/"ctrl"/"f10"），UI 层只管键名，不碰 VK 码。
"""

import ctypes
import os
import threading
import time
from ctypes import wintypes

from core import perf

# ---- 键名 → 虚拟键码 ----
_VK = {
    "left": 0x25, "right": 0x27, "up": 0x26, "down": 0x28,
    "ctrl": 0x11, "alt": 0x12, "shift": 0x10,
    "enter": 0x0D, "space": 0x20, "esc": 0x1B, "tab": 0x09,
    "backspace": 0x08, "del": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22,
    "grave": 0xC0,   # ` / ~（反引号键，VK_OEM_3）
    # 标点符号键（VK_OEM_*，手动输入 / 自定义按键要用）
    ",": 0xBC, ".": 0xBE, "/": 0xBF, ";": 0xBA, "'": 0xDE,
    "[": 0xDB, "]": 0xDD, "\\": 0xDC, "-": 0xBD, "=": 0xBB,
}
_VK.update({"f%d" % i: 0x70 + i - 1 for i in range(1, 13)})
_VK.update({chr(c): c for c in range(ord("A"), ord("Z") + 1)})   # a~z
_VK.update({str(i): ord("0") + i for i in range(10)})            # 0~9

# 键名 → 给 UI 显示的友好名（下拉框里用）
DISPLAY = {
    "left": "←", "right": "→", "up": "↑", "down": "↓",
    "ctrl": "Ctrl", "alt": "Alt", "shift": "Shift",
    "space": "空格", "enter": "回车", "esc": "Esc", "tab": "Tab",
    "del": "Del", "insert": "Ins", "home": "Home", "end": "End",
    "pageup": "PgUp", "pagedown": "PgDn",
    "grave": "~",
}
DISPLAY.update({"f%d" % i: "F%d" % i for i in range(1, 13)})
DISPLAY.update({chr(c): chr(c).upper() for c in range(ord("A"), ord("Z") + 1)})
DISPLAY.update({str(i): str(i) for i in range(10)})

# 键盘映射里可以选的键（按常用度排序）
CHOICES = ["left", "right", "up", "down", "ctrl", "alt", "shift", "space"] + \
          ["f%d" % i for i in range(1, 13)] + \
          [chr(c) for c in range(ord("A"), ord("Z") + 1)] + \
          [str(i) for i in range(10)] + \
          [",", ".", "/", ";", "'", "[", "]", "\\", "-", "=", "grave"]

# 本机操作键：**硬编码，永不发给游戏**。
#   F10 = 触控板触控模式开关（gui/touchpad.py）
#   F11 = 开关自动（默认热键；全局热键注册以外的地方也不该发它）
#   F12 = 预留（调试 / 截图之类）
# 拦在所有发送路径的最里面 —— 决策层、手动输入、断线重连都从 key_down/key_up/tap
# 出去，所以在这里拦一次就够。不拦的话：按一下本地开关，游戏里会跟着触发一次
# （最典型的是「按住 Ctrl 用触控板」——Ctrl 是默认攻击键，一按就打出去了）。
LOCAL_ONLY = ("f10", "f11", "f12")


def is_local_only(name):
    """是本机操作键吗（是的话，任何发送路径都不该把它发出去）。"""
    return str(name).lower() in LOCAL_ONLY


# 默认键位
DEFAULT_KEYMAP = {
    "auto": "f11",      # 开关自动
    "left": "left",     # 移动←
    "right": "right",   # 移动→
    "up": "up",         # 移动↑
    "down": "down",     # 移动↓
    "attack": "ctrl",   # 输出（攻击）
    "jump": "alt",      # 跳跃
    #: ⭐ 「拾取」（用户 2026-10-06 ✓）：**只给「战斗参数 → 自动拾取」用** ✓（那个开关开了
    #:   才会被点按 ✓）。默认 `"z"` = 这类游戏的常见拾取键 ✓ —— 不对就在
    #:   「键盘映射」组里点一下那个按钮、再按你游戏里的拾取键（Esc 取消 ✓）。
    "pickup": "z",      # 拾取（自动拾取）
    "hp_pot": None,     # 补血（自动喝药）
    "mp_pot": None,     # 补蓝（自动喝药）
    "feed_pet": None,   # 喂宠
    "shop": None,       # 商城
    "enter": "enter",   # 回车
}


def display_name(key):
    return DISPLAY.get(key, key)


def resolve_vk(key):
    """键名 → VK 码；不认识返回 None。"""
    if isinstance(key, int):
        return key
    n = str(key).lower()
    vk = _VK.get(n)
    if vk is None:
        # 字母键在 _VK 里存的是大写（'A'~'Z'），小写查不到时回退大写
        vk = _VK.get(n.upper())
    return vk


# ---- SendInput ----
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTunion)]


_user32 = ctypes.windll.user32


def _send(vk, up=False):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def key_down(name):
    if is_local_only(name):
        return      # 本机操作键（F10~F12）：永不发给游戏，见 LOCAL_ONLY
    if _remote is not None:
        _send_remote("PRESS " + _remote_key(name))
        return
    if _blocked:
        return      # 通道断了又没接上：宁可什么都不发，也别打到控制机上（见 _blocked）
    vk = resolve_vk(name)
    if vk is not None:
        _send(vk, False)


def key_up(name):
    if is_local_only(name):
        return      # 同上：本机操作键的松开也不发（否则游戏侧会收到一个孤儿 RELEASE）
    if _remote is not None:
        _send_remote("RELEASE " + _remote_key(name))
        return
    if _blocked:
        return
    vk = resolve_vk(name)
    if vk is not None:
        _send(vk, True)


def tap(name, duration=0.06):
    """点按一下（按下 → 松开）。给「开关自动」这类瞬发键用。"""
    if is_local_only(name):
        return      # 同上：本机操作键不点
    if _remote is not None:
        _send_remote("TAP %s %d" % (_remote_key(name), int(duration * 1000)))
        return
    if _blocked:
        return
    vk = resolve_vk(name)
    if vk is None:
        return
    _send(vk, False)
    time.sleep(duration)
    _send(vk, True)


class KeyState:
    """跟踪当前按下的键，只在集合变化时发 key_up/key_down。

    移动键要「按住不放」，攻击键也要持续 —— 不能每帧都重新按下/松开，
    否则游戏里表现为断断续续。这个类记着上次按了哪些，对比本次目标集合，
    只对差异发按键。
    """

    def __init__(self):
        self._pressed = set()
        #: ⭐ **瞬发键**（输出序列 / 攻击 tap ✓ 2026-09-30）：它们不能走 `set()`
        #:   —— 决策拍是**全量重算** ✗ 会把序列正按着的键当场顶掉 ⇒ 单独登记、
        #:   `pressed()` 取**并集** ✓（流画面左下的「按键帽」据此点亮输出键 ✓）。
        self._momentary = set()

    def pressed(self):
        """当前按着的键（**只读快照** ✓）—— 给"现在按着 ↑/↓ 吗"这类**许可条件**用 ✓。
        ⭐ 2026-09-30 起含**瞬发键**（序列/攻击 ✓ 见 `_momentary` ✓）—— 流画面
        的按键帽据此点亮 ✓；`holding_vertical` 等许可判定不受影响 ✓（瞬发的攻击键
        本来就不是攀爬许可的输入 ✗ 语义不冲突 ✓）。

        用户 2026-09-27 要求："位置状态判定优化：除非按住了 ↑ 或 ↓，不能主动判定为在绳梯上"
        ⇒ `agent.holding_vertical()` 读它，再由感知层决定要不要写 `player.ladder_id` ✓。
        """
        return set(self._pressed) | set(self._momentary)

    def set(self, keys):
        """keys：本次应该按下的键名集合。"""
        keys = set(k for k in keys if k)
        for k in self._pressed - keys:
            key_up(k)
        for k in keys - self._pressed:
            key_down(k)
        self._pressed = keys

    def momentary_mark(self, name):
        """**瞬发键记账**（只记账、**不发键** ✓ —— 键已由调用方发出，再发就是双按
        ✗ 实测输出 CD 用例当场炸 ✓）。⚠ 该键已被**决策**按着（∈ `_pressed`）⇒
        不记账（决策侧本来就在 `pressed()` 里 ✓ 摘账时也不能发松开 ✓）。"""
        if name and name not in self._momentary and name not in self._pressed:
            self._momentary.add(name)

    def momentary_discard(self, name):
        """**瞬发键摘账**（只摘账、**不发松开** ✓ 同上 ✓）。"""
        self._momentary.discard(name)

    def held_by_decision(self, name):
        """这个键**这一拍正被决策侧按着**吗（∈ `_pressed`，**不含**瞬发账 ✓）。

        谁问：输出序列里那个 `up` 步（用户 2026-10-03 ✓ —— "按键挤压"的另一半）。
        为什么需要它：序列 `down` 时若这个键**已经被决策按着**，`momentary_mark` **故意
        不登记**（决策侧本来就在 `pressed()` 里 ✓）；到 `up` 那一步要是照发 `key_up`，
        松掉的其实是**决策侧**按着的那个键 ✗ —— 而决策侧 `keys.set` 因为"键集没变"
        **不会补发** PRESS ⇒ 这个键**静默丢掉**（按住的动作莫名断一下 = 现场说的挤压 ✓）。
        ⚠ 与 `_tap` 用的 `pressed()`（**并集** ✓）**不是一回事，别合并** ✗：
          点按不能碰任何"别人按着的"键（并集 ✓）；而"松开"只该让开**决策侧**的键 ——
          序列**自己**按下去的那个必须照松（不然序列自己就卡住了 ✗）。
        """
        return str(name) in self._pressed

    def release_all(self):
        self.set(set())

    def clear(self):
        """只清空按键状态、不发 key_up。配合固件 RELEASEALL 使用：
        固件侧已一次性释放所有键，本地只需同步状态，下一帧重新按需要的键。"""
        self._pressed = set()


# ---- 远程后端（agent 跑在控制机，通过 TLS 把按键发到游戏机的 Pro Micro）----
# 默认 _remote=None 走本地 SendInput；调用 use_network() 后切到远程硬件键盘。
_remote = None
# 连接参数记下来，「重连通道」要用（见 reconnect_remote）
_cfg = None
# 通道断了又接不上时置 True：这时**绝不能**退化成本机 SendInput ——
# 控制机正是人在用的机器，把游戏按键打到本机上是会出事的（比如往别的窗口里打字）。
# 断开期间宁可什么都不发，等重连成功再恢复。
_blocked = False

#: 最近一次**连后端失败**的原因（人话 ✓；空串 = 没失败过 ✓）—— 给界面 / `behavior.log` 用。
#: ⚠⚠ 为什么必须留（2026-10-10 ✓ 用户报「**本地ProMicro连接失败**」✗ 而界面只有
#:   「Pro Micro 连接失败，已回退本地」八股 ✓）：`connect_async` 把异常**整个吞掉**了 ✗
#:   （`except Exception: status = "fail"` ✓）⇒ **为什么失败**在哪儿都查不到 ✓ 人只能猜 ✓
#:   —— 现场原因是 `PermissionError(13, 拒绝访问)` = **串口被另一个工作台占着** ✓
#:   （一个串口只能被一个进程独占 ✓ 而当时有**两个工作台**在跑 ✓）。
_connect_err = ""

# 换后端（连 / 断 / 重连）**必须串行**：这几件事都是「先关旧的、再建新的」。
# 两个线程同时做就会出现「一边正在 TLS 握手，另一边把那个 socket 关掉」——
# 表现是 relay 侧刷「客户端在 TLS 握手阶段断开」（实测踩过：手动的「重置指令通道」
# 和界面上新加的自动重连撞在一起）。用 RLock：reconnect_remote 内部要调
# use_network（同一线程重入），普通 Lock 会把自己锁死。
_link_rlock = threading.RLock()

# ---- ⭐⭐ 「离屏（自检 / 冒烟）**绝不连被控机**」的唯一闸（用户 2026-10-05 ✓ 治根）----
#: 显式"我就是要连"（专门的链路自检 `tools/selftest_link.py`、无头部署那种场合 ✓）
_ALLOW_NET_ENV = "PSIMU_ALLOW_NET"
#: 闸关着时的留痕标记（同一 kind 只记一次，别刷爆日志 ✗ 见 `_note_net_skip`）
_net_skip_kind = None


def net_allowed():
    """这条进程**允不允许**去连被控机（远端 relay / 串口硬件键盘）。

    ⭐⭐ 为什么要有它（用户 2026-10-05 ✓ 原话："**从源头彻底封死**"）：
      病根是**打桩打在了用例里、而且只打了一个套件** ✗ —— 原来只有
      `tools/selftest_main_window._win()` 在 monkeypatch 面板方法 ✓，而
      `gui_smoke` / `selftest_minimap` / `selftest_screen_state` / `selftest_decision`
      里那几个 `PlayerPanel()` **都没打** ✗ ⇒ 只要项目配置是 `input_device: remote`
      （`config/decision.json` 与好几个项目就是 ✓）它们照样起那条**后台连接线程** ✗
      ⇒ 线程在**几秒后**才超时返回，而那时用例早跑完、Qt 对象已销毁 ⇒
      **偶发原生崩 `0xC0000005`** ✗（时机性 ⇒ "多跑几次"不算验证 ✗）。

    判据（先显式、后推断 ✓；**每次都读环境**，不缓存 ✓ 用例要能当场摆判据）：
      · `PSIMU_ALLOW_NET=1` ⇒ **一律允许**（`tools/selftest_link.py` 这种"就是要连"的 ✓）；
      · 否则 `QT_QPA_PLATFORM=offscreen` ⇒ **不允许**：离屏 = 没人看得见画面，此时去连被控机
        **只可能**是自检 / 冒烟在跑（~20 个套件 + `gui_smoke` 都用它 ✓）；
      · 其余（正常工作台 / **部署台**，真平台）⇒ 允许 ✓ —— `deploy/` 侧不设 offscreen（已核 ✓）。

    ⚠ 只管一件事：**连不连被控机**。本地 SendInput 那条路不归它管 ✓。
    """
    if str(os.environ.get(_ALLOW_NET_ENV, "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    return str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower() != "offscreen"


def _note_net_skip(kind):
    """留痕：闸关着、**没去连**（不许静默 ✗）—— 同一 kind 只记一次 ✓。"""
    global _net_skip_kind
    if _net_skip_kind != kind:
        _net_skip_kind = kind
        try:
            perf.count("net_skip_" + str(kind or "?"))
        except Exception:                       # noqa: BLE001 —— 打点坏了别影响行为 ✗
            pass


def use_blocked(why=""):
    """切到「**什么都不发**」的后端（`_blocked` ✓）：连不上被控机时**绝不能**退化成把按键
    打到控制机上（控制机正是人在用的机器 ✗ —— 老口径见 `_blocked` 的说明 ✓）。

    ⭐ 新用途（用户 2026-10-05 ✓）：闸关着（离屏自检 / 显式关闸 ✓）时走这儿 ——
      **不连、也不发**（比"退化成本地"安全 ✓ 也比"静默"诚实 ✓）。
    """
    global _blocked
    with _link_rlock:
        _drop_remote()
        _blocked = True
        _note_net_skip(why or "blocked")


def last_connect_err():
    """最近一次**连后端失败**的原因（人话 ✓；空串 = 没失败过 ✓）—— 见 `_connect_err` ✓。"""
    return _connect_err


def cursor_probe():
    """**现在光标在哪**（＋ 那台机器的屏幕信息）⇒ `(x, y, vx, vy, vw, vh, cap)`；问不到 ⇒ None ✓。

    · **本地后端**（本机 SendInput / 串口 ProMicro ✓）⇒ 本机 `GetCursorPos` ✓（`cap = None` ✓）；
    · **网络 relay**（鼠标动在**被控机** ✓）⇒ 问 A 机（`KbdClient.cursor` ✓ 见 relay 那边 ✓）。

    ⚠ 这一条就是"**能不能自己量鼠标标定**"的闸 ✓（`decision/mouse_aim.auto_measure` ✓）：
      问不到 ⇒ 只能请人**手工量一次** ✓（当帧尺寸变了的时候 ✓ 见 `gain_mismatch_block` ✓）。
    ⚠ 为什么非要能"看见光标"：gain 是「帧像素 / 指令单位」✓ ⇒ 只能**发已知单位、看走了多少像素**
      量出来 ✓ ⇒ 看不见光标 ⇒ 量不出来 ✓（这是用户 2026-10-10："**不能自动吗？**" 的那道坎 ✓）。
    """
    if _remote is not None and hasattr(_remote, "cursor"):
        # 网络后端（`KbdClient` ✓）：**问 A 机** ✓ 见 `remote_kbd/relay.py` 的 `CURSOR_QUERY` ✓
        try:
            return _remote.cursor()
        except Exception:                        # noqa: BLE001 —— 问不到就当我们量不了 ✓
            return None
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        pt = wintypes.POINT()
        if not u.GetCursorPos(ctypes.byref(pt)):
            return None
        return (int(pt.x), int(pt.y),
                int(u.GetSystemMetrics(76)), int(u.GetSystemMetrics(77)),
                int(u.GetSystemMetrics(78)), int(u.GetSystemMetrics(79)), None)
    except Exception:                            # noqa: BLE001
        return None


def _note_connect_fail(kind, why):
    """留痕：连不上时**把原因记下来**（日志 + 打点 ✓ 不许静默 ✗ 见 `_connect_err` ✓）。"""
    try:
        # ⚠ 是 `core.behavior` ✗（原来写成裸 `import behavior` ⇒ 这里**每次都抛** ✓
        #   被下面那句 except 吃掉 ⇒ **这条日志一直没写出去** ✓ 2026-10-10 顺手修 ✓）。
        from core import behavior
        behavior.event("connect_fail", kind=str(kind or "?"), why=str(why)[:160])
    except Exception:                           # noqa: BLE001 —— 打点坏了别影响连接 ✗
        pass
    try:
        perf.count("connect_fail_" + str(kind or "?"))
    except Exception:                           # noqa: BLE001
        pass


def connect_async(kind, on_done=None, **kw):
    """按 `kind` 起一条**后台连接线程**（GUI 用，`PlayerPanel._apply_input_device` 调它）。

    ⚠⭐ **全仓库唯一**去连被控机的地方（原来那两处 `threading.Thread(...)` 在面板里 ✗）——
      用户 2026-10-05 要求"从源头彻底封死"：**线程的出生地和闸放同一处** ✓ ⇒ 谁调都绕不过去 ✓。

    闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开 ✓ 见 `net_allowed`）⇒ **连线程都不起** ✓
    （**不是**"起一条马上返回"✗ —— 那样它仍可能在解释器收尾时活着 ✗ 正是要治的病 ✓）
    ＋ 转 `use_blocked`（什么都不发 ✓）。

    返回 `True` = 起了线程；`False` = **没起**（`on_done` 也不会被叫 ✓ 调用方自己给提示 ✓）。

    `on_done(status)` 在**后台线程**里被叫：`"ok"` / `"fail"` —— 回调里碰 Qt 对象必须自己兜
    `RuntimeError`（窗口可能已经销毁 ✓ 见 `PlayerPanel._on_device_connected` ✓）。
    """
    if not net_allowed():
        use_blocked(str(kind))
        return False

    def _run():
        global _connect_err
        status = "ok"
        try:
            if kind == "serial":
                use_serial(kw.get("port"))
            else:
                use_network(kw.get("host"), kw.get("port"), kw.get("cert"))
        except Exception as e:                  # noqa: BLE001 —— 连不上是常态（对面没开机 ✓）
            status = "fail"
            # ⚠⚠ **不许静默**（2026-10-10 ✓ 用户现场）：原来这里把异常整个吞掉 ✗ ⇒ 界面只剩
            #   "Pro Micro 连接失败" ✗、`behavior.log` 里也没线索 ⇒ 只能靠人猜 ✓
            #   （实际原因是**串口被另一个工作台占着** ✓ 见 `_connect_err` ✓）。
            if not _connect_err:
                _connect_err = "%s: %s" % (type(e).__name__, e)
            _note_connect_fail(kind, _connect_err)
        if on_done is not None:
            on_done(status)

    threading.Thread(target=_run, daemon=True,
                     name="input-connect-%s" % kind).start()
    return True


_CMD_KEY = {
    "left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN",
    "ctrl": "CTRL", "alt": "ALT", "shift": "SHIFT",
    "enter": "ENTER", "space": "SPACE", "esc": "ESC", "tab": "TAB",
    "backspace": "BACKSPACE",
    "del": "DEL", "insert": "INSERT", "home": "HOME", "end": "END",
    "pageup": "PGUP", "pagedown": "PGDN",
    "grave": "GRAVE",
}
_CMD_KEY.update({"f%d" % i: "F%d" % i for i in range(1, 13)})


def _remote_key(name):
    """input.py 键名 → 固件键名。"""
    n = str(name).lower()
    return _CMD_KEY.get(n, n)


def _send_remote(line):
    """发一条远程指令。返回 True/False（False 或不返回都表示这条没发出去）。

    以前这里 `except: pass` 把失败全吞了 —— 通道死了上层一点感觉都没有，
    还在不停发。表现就是「停自动没用、手动也没用，只能关 GUI」。
    """
    if _remote is None:
        return False
    _t = time.perf_counter()
    try:
        ok = bool(_remote.send(line))
    except Exception:
        ok = False
    # 打点：一次发键在通道上花了多久（远程是 socket / 串口写，链路卡住时这里最先变大）
    perf.ms("send_ms", _t)
    if ok:
        perf.count("send")
    else:
        perf.count("send_fail")
        # 失败按指令名分开计（fail_PRESS / fail_RELEASEALL / fail_TAP…）：
        # 「所有指令都在丢」和「某一条特定指令在丢」是完全不同的问题
        # —— 后者往往是固件正卡在某个 delay 里没读串口。
        head = (str(line).split() or ["?"])[0][:10]
        perf.count("fail_" + head)
    return ok


# 发过指令后，超过这么久收不到固件回执就判定链路卡死（秒）。
# 固件对每条指令都会回 DONE/ERR，所以「静默」是可靠的死链信号。
REPLY_SILENCE_LIMIT = 5.0


def link_health():
    """指令通道当前状态：{"backend", "ok", "err", "fails", "silent"}。

    backend: "local"（本机 SendInput）/ "remote"（网络 Pro Micro）/
             "serial"（直连）/ "blocked"（断了又接不上，已停止发送）。

    ok 同时对「发送有没有报错」和「固件还有没有回执」两件事下判断 ——
    后者能抓到 relay 卡在串口写上这种「发送成功、实际什么都没发生」的情况。
    """
    if _remote is None:
        # ⚠ 连不上时 `_remote` 就是 None ✗ ⇒ 这里**必须**把 `_connect_err` 带出去 ✓
        #   （不然界面 / 日志只有"连接失败"四个字 ✓ 见 `_connect_err` 的说明 ✓）。
        return {"backend": "blocked" if _blocked else "local",
                "ok": not _blocked, "err": _connect_err, "fails": 0, "silent": 0.0}
    kind = _cfg[0] if _cfg else "remote"
    send_ok = bool(getattr(_remote, "ok", True))
    silent = 0.0
    try:
        silent = float(_remote.silent_for())
    except Exception:
        pass
    quiet_ok = silent < REPLY_SILENCE_LIMIT
    err = str(getattr(_remote, "last_err", ""))
    if not quiet_ok and not err:
        err = "发指令后 %.0f 秒没收到固件回执（链路卡死）" % silent
    return {"backend": kind,
            "ok": send_ok and quiet_ok,
            "err": err,
            "fails": int(getattr(_remote, "fails", 0)),
            "silent": silent}


def reconnect_remote():
    """丢掉当前连接重开。返回 True = 重开后通道可用。

    **为什么重连能解开卡键**：relay 在客户端断开时会往串口直接写一条 RELEASEALL
    （`remote_kbd/relay.py` 的 bridge() 里的 finally）。也就是说「断开」这个动作
    本身就是一次硬复位 —— 而且它**不依赖我们这边的 TCP 通道是否还通**。
    这正是上次「关掉 GUI」能让卡住的键松开的原因：那一下是 relay 替我们松的。

    重连失败**不会**退化成本机 SendInput（那会把按键打到控制机上），而是进入
    _blocked：什么都不发，等下次重连成功。

    ⚠ 闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开 ✓ 见 `net_allowed`）⇒ **不重连**、
      直接转"什么都不发"并返回 `False`（"通道不可用"是**实话** ✓ 别报 True 骗上层 ✗）。
    """
    global _remote, _blocked
    if not net_allowed():
        use_blocked("retry")
        return False
    with _link_rlock:               # 与其它换后端的调用串行（见 _link_rlock 说明）
        if _remote is None or _cfg is None:
            _blocked = False
            return True
        kind = _cfg[0]
        try:
            if kind == "remote":
                use_network(_cfg[1], _cfg[2], _cfg[3])
            else:
                use_serial(_cfg[1])
            return True
        except Exception:
            _drop_remote()
            _blocked = True
            return False


def release_all_remote():
    """发 RELEASEALL 到固件，一次性清空固件侧所有按住的键（防长时间运行卡键）。"""
    if _remote is not None:
        _send_remote("RELEASEALL")


# ---- 鼠标（远程模式：由 Pro Micro 固件作为硬件 HID 鼠标输出，同键盘一样走固件）----
#: 上一条 `mouse_move` 的时刻（算「命令间隔」用 ✓ 见 `mouse_move`）。
_last_move_t = None


def mouse_move(dx, dy):
    """相对移动鼠标：dx/dy 像素（正数向右/向下）。

    ⭐ **打点**（2026-10-03 ✓ 用户现场："看起来现象是在 A 机上一段一段一顿一顿的指令
    汇报很离散，B 机卡不卡其实没关系"）：
    鼠标这条链以前**没有任何专用计数** ✗（`MOVE` 和键盘混在 `send` 里 ✓）⇒ 「实际命令率
    多少 / 间隔匀不匀 / 每步几像素」全都量不出来、只能猜 ✓。现在补三个：

      · `mouse_send`   条数 ⇒ 命令率（触控板正常 60~125/s；远低于它 = 在攒着发 ✗）
      · `move_gap_ms`  相邻两条 MOVE 的间隔 ⇒ 中位小而 p95 大 = **一撮一撮** ✗
      · `move_px`      `|dx|+|dy|` ⇒ **中位 1~2 = 量化台阶**（`_on_pad_moved` 的 `int()`
                        + 余数累积：慢速时几个事件才凑够 1 像素 ✓）

    ⚠ 另一半在 **A 侧**（`remote_kbd/relay.py` 的 `MOVE 节拍` 那行 ✓）⇒ 两边一对比就能
      定案「是 B **发**得就不匀」还是「A **转**得不匀」✓（这正是本轮只加打点、不先改代码
      的原因 ✓）。
    """
    global _last_move_t
    if _remote is not None:
        _now = time.perf_counter()
        if _last_move_t is not None:
            perf.sample("move_gap_ms", (_now - _last_move_t) * 1000.0)
        _last_move_t = _now
        perf.count("mouse_send")
        perf.sample("move_px", abs(int(dx)) + abs(int(dy)))
        _send_remote("MOVE %d %d" % (int(dx), int(dy)))


def mouse_click(btn="left"):
    """点击鼠标：left / right / middle。"""
    if _remote is not None:
        _send_remote("CLICK " + str(btn).upper())


def mouse_press(btn="left"):
    """按住鼠标按钮不松开。"""
    if _remote is not None:
        _send_remote("PRESSM " + str(btn).upper())


def mouse_release(btn="left"):
    """松开鼠标按钮。"""
    if _remote is not None:
        _send_remote("RELEASEM " + str(btn).upper())


def mouse_scroll(n):
    """滚轮：正数向上、负数向下。"""
    if _remote is not None:
        _send_remote("SCROLL %d" % int(n))


def mouse_drag(dx, dy, btn="left", steps=8):
    """拖拽：按下 -> 分段移动 -> 松开。**交给固件一条指令原子完成**。

    **为什么不在这里发 PRESSM / MOVE×n / RELEASEM 三条**：中途串口或 TLS 断一次，
    那条 RELEASEM 就永远到不了 —— 左键会一直按着（只能拔板子，或者关掉界面让
    relay 替我们松）。合成一条发过去，最坏后果只是「这次拖拽没发生」，
    不会留下按住的键。

    steps：移动分几段（固件侧 1~40，段间隔几毫秒）。**必须分段** ——
    瞬移式的拖拽游戏看不到中间位置，不会被当成拖动。

    **发送失败补一发 RELEASEM**：`_send_remote` 返回 False 说明这条没发出去，
    但它也可能是「发出去了一半才断」，固件那边已经按下了。宁可多松一次
    （松一个没按的键无害），也别留一个按住的左键。

    注：这是给**程序化拖拽**（脚本 / 重连流程 / 一次性的拖到位）用的。
    触控板上那种跟着手指走的拖拽是**人机交互**，必须按住期间连续发 MOVE，
    所以那边仍然用 mouse_press / mouse_move / mouse_release 三件套。
    """
    if _remote is None:
        return
    n = max(1, min(40, int(steps)))
    if not _send_remote("DRAG %s %d %d %d"
                        % (str(btn).upper(), int(dx), int(dy), n)):
        mouse_release(btn)


def mouse_available():
    """鼠标控制是否可用：只有 ProMicro 后端（远程 / 本地串口）能输出硬件鼠标。

    本地 SendInput 模式不实现鼠标 —— 本机鼠标正用于操作界面，
    再让程序模拟鼠标会和人手打架。
    """
    return _remote is not None


# ---- ⭐⭐ 指针跟踪：触控板位移的**产生**侧（用户 2026-10-05 ✓ 原话："直接2"）----
#
# **为什么要这一块**：触控板的位移原来只在 `TouchPad.mouseMoveEvent` 里产生 ✗（Qt 的
#   move 事件 ⇒ 跑在 GUI 主线程上）—— 主线程一被重绘 / 主回路拖住，Qt 就会把连续移动
#   **合并**成一次。实测（`perf.log` 2026-10-05 18:40 那段，按 F10 滑了两下）：
#     `move_gap_ms` 中位 20 / p95 76 / p99 201 / **最大 467 ms**
#     `move_px`                             **最大 514**（一次跳半个屏幕 ✓）
#   ⇒ A 机看到的就是"停一下、猛跳一下" ✓；而同一段里 `send_ms` 中位才 **0.15 ms**、
#   `kbd_pending` 最大 4 ⇒ **瓶颈既不在链路也不在发送，在位移的产生** ✓
#   （发送那条早已挪出主线程 ✓ 见 `gui/player_panel.py::_PadSender`）。
# ⇒ 现在由下面这条**独立线程**按固定节拍读系统指针、算位移 ⇒ 节拍只受"线程能不能被调度"
#   影响，与 GUI 卡不卡无关 ✓。

#: 轮询节拍（秒）：250 Hz（比触控板本身的事件率还高 ⇒ 不丢节拍 ✓）。
#: ⚠ 能不能真的到 4 ms 取决于**系统定时器精度**：Windows 默认 ~15.6 ms ⇒ 实际约 64 Hz，
#:   但**仍然均匀** ✓ —— 而"均匀"正是要害（用户要的不是"更快"，是"别一顿一顿" ✓）。
#:   本仓库只在 `core/winperf.py` **一处**声明精度（工作台启动时按「性能保活」设置应用 ✓，
#:   实测见 `tools/selftest_winperf.py` ✓）⇒ **这里绝不自己再调 `timeBeginPeriod`** ✗
#:   （同一件事只该有一处实现 ✓）。实际节奏看打点 **`pad_poll_gap_ms`** ✓。
TRACK_INTERVAL_S = 0.004


def pointer_pos():
    """当前指针位置（**物理像素**，Win32 屏幕坐标 ✓）；读不到返回 None ✓。"""
    try:
        pt = wintypes.POINT()
        if _user32.GetCursorPos(ctypes.byref(pt)):
            return (int(pt.x), int(pt.y))
    except Exception:                   # noqa: BLE001 —— 读不到就当"这拍没数据"✓ 不炸 ✓
        pass
    return None


def pointer_warp(x, y):
    """把指针摆到 (x, y)（**物理像素** ✓）。返回是否成功 ✓。

    ⚠ **会失败，而且是"无声失败"**：本机实测（2026-10-05）在**锁屏 / 非交互桌面**下
      `SetCursorPos` 返回 0、指针纹丝不动，而 `GetLastError` **还是 0** ✗（查不出原因 ✓）
      —— 这不是缺陷，是桌面的规矩 ✓。所以调用方必须**按返回值行事**、不能想当然 ✓
      （`PointerTracker` 因此用 `pos - 上一次` 算位移 ✓：拨回成不成功都不影响位移正确 ✓）。
    """
    try:
        return bool(_user32.SetCursorPos(int(x), int(y)))
    except Exception:                   # noqa: BLE001
        return False


class PointerTracker:
    """⭐ 按**固定节拍**轮询系统指针、算位移的线程（用户 2026-10-05 ✓ "直接2"）。

    它干的活（每拍）：读位置 → 减掉**上一次读到的位置**得位移 → 位移非零就
      `on_delta(dx, dy)` → 把指针 `SetCursorPos` 拨回钉点（钉点 = 进触控模式时的板心 ✓）。

    ⚠⚠ 三条纪律（违反了就是另一个 bug ✗）：
      · **只用 Win32**（`GetCursorPos` / `SetCursorPos`）—— **绝不碰 Qt** ✗：
        `QCursor.setPos()` 只许在 GUI 线程调，而这条线程不是 ✓；
      · `on_delta` **在这条线程里被调** ⇒ 回调只许做线程安全的事（加法 / 入队 / 置事件 ✓），
        **绝不许碰界面** ✗ —— 要通知界面就走 Qt 信号（那是排队的 ✓ 见 `gui/touchpad.py`）；
      · 位移用 **`pos - 上一次`**，**不是** `pos - 钉点` ✗ —— 后者在"拨回失败 / 指针被别的
        东西挪走"那一下会**算出一大跳**（而且之后一路算 0 ✗）；前者不管拨回成不成功，
        都只算"这一拍真的动了多少" ✓。

    ⚠ 钉点是**物理像素**（`GetCursorPos` 的原生坐标 ✓）⇒ 高分屏下与 Qt 的逻辑坐标差一个
      比例 ⇒ 位移由调用方除回去（`scale` ✓ = `devicePixelRatio` ✓）；钉点本身不用换算 ✓
      （它来自 `GetCursorPos` 自己 ✓）。
    """

    def __init__(self, ref, on_delta, scale=1.0, interval=TRACK_INTERVAL_S):
        self.ref = (int(ref[0]), int(ref[1]))
        self.on_delta = on_delta
        self.scale = max(1e-6, float(scale))
        self.interval = max(0.001, float(interval))
        self.n_poll = 0          # 轮询拍数（自检/诊断用 ✓）
        self.n_delta = 0         # 其中有位移的拍数 ✓
        self._stop = False
        self._th = None

    def start(self):
        """起线程。返回 True/False：`GetCursorPos` 读不到 ⇒ False（调用方自己给提示 ✓）。"""
        if pointer_pos() is None:
            return False
        self._th = threading.Thread(target=self._run, daemon=True, name="pad-track")
        self._th.start()
        return True

    def stop(self, timeout=0.3):
        """收工：置停止 + **等线程真的退出**（别留一条守护线程在后台摸指针 ✗）。"""
        self._stop = True
        th = self._th
        self._th = None
        if th is not None:
            th.join(timeout)             # 一拍之内必退（interval 只有几 ms ✓）

    def _run(self):
        last = None                      # 上一次读到的位置（`pos - last` ✓ 见纪律第三条）
        last_t = None
        while not self._stop:
            pos = pointer_pos()
            if pos is not None:
                now = time.perf_counter()
                if last_t is not None:
                    perf.sample("pad_poll_gap_ms", (now - last_t) * 1000.0)
                last_t = now
                self.n_poll += 1
                if last is not None:
                    dx = pos[0] - last[0]
                    dy = pos[1] - last[1]
                    if dx or dy:
                        self.n_delta += 1
                        perf.sample("pad_poll_px",
                                    (abs(dx) + abs(dy)) / self.scale)
                        try:
                            self.on_delta(dx / self.scale, dy / self.scale)
                        except Exception:       # noqa: BLE001 —— 回调坏了别把跟踪线程弄死 ✗
                            pass
                        # ⚠⚠ 拨回成功 ⇒ **指针此刻就在钉点** ⇒ `last` 必须跟着改成钉点 ✗：
                        #   不改，下一拍就会把"**我们自己拨回去**"读成一次**反向位移** ✗
                        #   （(-dx,-dy) ⇒ 远程鼠标每拍抖一下 ✓ 实测就是这么抓出来的 ✓
                        #    用例里假指针会立刻跟到钉点 ✓ 所以它是**确定的红**、不是碰运气 ✓）。
                        # ⚠ 拨回可能失败（锁屏 / 非交互桌面：返回 0 且 `GetLastError` 也是 0 ✓
                        #   见 `pointer_warp` 的说明 ✓）—— **失败不算错**（位移照旧正确 ✓ 因为
                        #   用的是 `pos - 上一次` ✓），但要**留痕**：否则"指针跑掉了"查不出来 ✗
                        if pointer_warp(self.ref[0], self.ref[1]):
                            pos = self.ref
                        else:
                            perf.count("pad_warp_fail")
                last = pos
            time.sleep(self.interval)


def _drop_remote():
    """关闭当前远程/串口后端（socket / 串口不泄漏），切到别的后端前调用。"""
    global _remote
    if _remote is not None:
        try:
            _remote.close()
        except Exception:
            pass
    _remote = None


def use_network(host, port, cafile):
    """切到网络后端（agent 在控制机时，启动阶段调用一次）。

    拿锁期间会做完整的 TLS 握手（可能要几秒）—— 这是故意的：握手和「关旧连接」
    必须互斥，否则会互相打断（见 _link_rlock 的说明）。

    ⚠⭐ 闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开 ✓ 见 `net_allowed`）⇒ **一个 socket
      都不建**、直接转"什么都不发"（`use_blocked` ✓）—— 这条是**兜底**：正常 GUI 走
      `connect_async`（它自己先问闸 ✓，连线程都不起 ✓）。
    """
    global _remote, _cfg, _blocked, _connect_err
    if not net_allowed():
        use_blocked("remote")
        _connect_err = "闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开）⇒ 没去连被控机"
        return
    with _link_rlock:
        _drop_remote()
        from remote_kbd.kbd_client import KbdClient
        _remote = KbdClient(host, port, cafile)
        _cfg = ("remote", host, port, cafile)   # 记下来给 reconnect_remote 用
        _blocked = False
        _connect_err = ""                       # 连上了 ⇒ 清掉上次的失败原因 ✓


def _serial_fail_text(port, exc, found=True):
    """把「串口打不开」翻译成**能照着做的人话**（见 `_connect_err` 的说明 ✓）。

    ⚠ 为什么要分这么细（2026-10-10 ✓ 用户现场「本地ProMicro连接失败」✗）：
      **设备在不在**和**能不能打开**是两件事 ✗ —— 当时设备好好的（`COM8 Arduino Leonardo`
      ✓ 自动找口也找得到 ✓），可 `serial.Serial(...)` 抛
      `PermissionError(13, 拒绝访问)` ✗ = **串口被别的进程占着** ✓
      （Windows 串口是**独占**的 ✓ —— 当时有**两个工作台**在跑 ✓）。
      这两种病要给**完全相反**的建议（一个"拔插 USB"、一个"关掉另一个工作台" ✓）
      ⇒ 混成一句"连不上"等于没说 ✓。
    """
    s = ("%s: %s" % (type(exc).__name__, exc)) if exc is not None else ""
    low = str(exc).lower() if exc is not None else ""
    if found is None:
        return ("找不到 Pro Micro 串口（USB 插着吗？也可以看看 `config/link.yaml` 的 "
                "`serial_local` ✓ 换 USB 口后串口号会变、这儿会自动找 ✓）")
    busy = (isinstance(exc, PermissionError) or "拒绝访问" in s or "access is denied" in low
            or "permissionerror" in low or "busy" in low or "占用" in s)
    if busy:
        # ⚠ 只说**这一种病**该怎么做 ✓ —— 别把"拔插 USB"混进来 ✗：那是"插都没插"那一种的
        #   建议 ✓ 两种病建议**相反** ✓（混着说是把人往反方向推 ✓ 见上面那段说明 ✓）。
        return ("串口 %s 被**别的进程占用**了（%s）⇒ 一个串口同时只能被一个进程打开 ✓："
                "**多半是另一个工作台 / relay 还开着** ✓ 关掉多余的那个、再点连接 ✓"
                % (port, s))
    return ("串口 %s 打不开（%s）⇒ 拔插一次 USB、或换个 USB 口试试 ✓" % (port, s))


def use_serial(port):
    """切到本地直连 Pro Micro（USB CDC 串口），无需 relay。

    port 为空或打开失败时，自动扫描 Arduino/SparkFun 串口（换 USB 口不用改配置）。

    ⚠⭐ 闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开 ✓ 见 `net_allowed`）⇒ **串口也不开**、
      直接转"什么都不发"（`use_blocked` ✓）—— 同 `use_network`：谁直接调都绕不过去 ✓。

    ⚠⚠ 连不上时**抛出的那句话就是给人看的**（见 `_serial_fail_text` ✓）—— 别把它简化成
      "连接失败" ✗（那正是 2026-10-10 现场"查不出原因"的由来 ✓）。
    """
    global _remote, _cfg, _blocked, _connect_err
    if not net_allowed():
        use_blocked("serial")
        _connect_err = "闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开）⇒ 没去连串口"
        return
    with _link_rlock:
        _drop_remote()
        from remote_kbd.serial_kbd import SerialKbd, find_pro_micro_port
        if port:
            try:
                _remote = SerialKbd(port)
                _cfg = ("serial", port)
                _blocked = False
                _connect_err = ""
                return
            except Exception as e:              # noqa: BLE001 —— 记下**第一手**原因 ✓（下面要说清 ✓）
                _first = e
        else:
            _first = None
        found = find_pro_micro_port()
        if found is None:
            _connect_err = _serial_fail_text(port, _first, found=None)
            raise RuntimeError(_connect_err)
        try:
            _remote = SerialKbd(found)
        except Exception as e:                  # noqa: BLE001 —— 找到了却打不开（多半是被占用 ✓）
            _connect_err = _serial_fail_text(found, e)
            raise RuntimeError(_connect_err)
        _cfg = ("serial", found)
        _blocked = False
        _connect_err = ""


def use_local():
    """切回本地 SendInput，并关闭远程连接（socket / 串口不泄漏）。

    ⚠⚠ **不清 `_connect_err`** ✗（2026-10-10 ✓ 特意留的）：连不上时 `PlayerPanel` 会调
      它**回退本地** ✓，接着就要把"为什么没连上"显示出来 ✓ ⇒ 这里清掉的话
      界面又只剩"连接失败"四个字 ✗（那正是要治的病 ✓ 见 `_connect_err` ✓）。
      真正该清的地方是**连接成功**那两条路（`use_serial` / `use_network` ✓）。
    """
    global _blocked
    with _link_rlock:
        _drop_remote()
        _blocked = False


def shutdown():
    """关闭远程键盘连接（如果存在），停止指令传输。GUI 关闭时调用。"""
    use_local()
