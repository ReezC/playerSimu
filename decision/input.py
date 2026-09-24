"""键盘模拟：键名映射 + SendInput 底层发送。

游戏通常读 GetAsyncKeyState / DirectInput。SendInput 的虚拟键码对
GetAsyncKeyState 有效（大多数 ARPG）；个别用 DirectInput 读扫描码的
游戏才需要 KEYEVENTF_SCANCODE，这里先走虚拟键码，遇到不生效再换。

键名用可读字符串（"left"/"ctrl"/"f10"），UI 层只管键名，不碰 VK 码。
"""

import ctypes
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

    def set(self, keys):
        """keys：本次应该按下的键名集合。"""
        keys = set(k for k in keys if k)
        for k in self._pressed - keys:
            key_up(k)
        for k in keys - self._pressed:
            key_down(k)
        self._pressed = keys

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

# 换后端（连 / 断 / 重连）**必须串行**：这几件事都是「先关旧的、再建新的」。
# 两个线程同时做就会出现「一边正在 TLS 握手，另一边把那个 socket 关掉」——
# 表现是 relay 侧刷「客户端在 TLS 握手阶段断开」（实测踩过：手动的「重置指令通道」
# 和界面上新加的自动重连撞在一起）。用 RLock：reconnect_remote 内部要调
# use_network（同一线程重入），普通 Lock 会把自己锁死。
_link_rlock = threading.RLock()

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
        return {"backend": "blocked" if _blocked else "local",
                "ok": not _blocked, "err": "", "fails": 0, "silent": 0.0}
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
    """
    global _remote, _blocked
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
def mouse_move(dx, dy):
    """相对移动鼠标：dx/dy 像素（正数向右/向下）。"""
    if _remote is not None:
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
    """
    global _remote, _cfg, _blocked
    with _link_rlock:
        _drop_remote()
        from remote_kbd.kbd_client import KbdClient
        _remote = KbdClient(host, port, cafile)
        _cfg = ("remote", host, port, cafile)   # 记下来给 reconnect_remote 用
        _blocked = False


def use_serial(port):
    """切到本地直连 Pro Micro（USB CDC 串口），无需 relay。

    port 为空或打开失败时，自动扫描 Arduino/SparkFun 串口（换 USB 口不用改配置）。
    """
    global _remote, _cfg, _blocked
    with _link_rlock:
        _drop_remote()
        from remote_kbd.serial_kbd import SerialKbd, find_pro_micro_port
        if port:
            try:
                _remote = SerialKbd(port)
                _cfg = ("serial", port)
                _blocked = False
                return
            except Exception:
                pass   # 配置的串口打不开，走自动发现
        found = find_pro_micro_port()
        if found is None:
            raise RuntimeError("找不到 Pro Micro 串口（确认已插入，或检查 config/link.yaml 的 serial_local）")
        _remote = SerialKbd(found)
        _cfg = ("serial", found)
        _blocked = False


def use_local():
    """切回本地 SendInput，并关闭远程连接（socket / 串口不泄漏）。"""
    global _blocked
    with _link_rlock:
        _drop_remote()
        _blocked = False


def shutdown():
    """关闭远程键盘连接（如果存在），停止指令传输。GUI 关闭时调用。"""
    use_local()
