"""键盘模拟：键名映射 + SendInput 底层发送。

游戏通常读 GetAsyncKeyState / DirectInput。SendInput 的虚拟键码对
GetAsyncKeyState 有效（大多数 ARPG）；个别用 DirectInput 读扫描码的
游戏才需要 KEYEVENTF_SCANCODE，这里先走虚拟键码，遇到不生效再换。

键名用可读字符串（"left"/"ctrl"/"f10"），UI 层只管键名，不碰 VK 码。
"""

import ctypes
import time
from ctypes import wintypes

# ---- 键名 → 虚拟键码 ----
_VK = {
    "left": 0x25, "right": 0x27, "up": 0x26, "down": 0x28,
    "ctrl": 0x11, "alt": 0x12, "shift": 0x10,
    "enter": 0x0D, "space": 0x20, "esc": 0x1B, "tab": 0x09,
    "del": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22,
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
}
DISPLAY.update({"f%d" % i: "F%d" % i for i in range(1, 13)})
DISPLAY.update({chr(c): chr(c).upper() for c in range(ord("A"), ord("Z") + 1)})
DISPLAY.update({str(i): str(i) for i in range(10)})

# 键盘映射里可以选的键（按常用度排序）
CHOICES = ["left", "right", "up", "down", "ctrl", "alt", "shift", "space"] + \
          ["f%d" % i for i in range(1, 13)] + \
          [chr(c) for c in range(ord("A"), ord("Z") + 1)] + \
          [str(i) for i in range(10)]

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
    return _VK.get(str(key).lower())


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
    if _remote is not None:
        _send_remote("PRESS " + _remote_key(name))
        return
    vk = resolve_vk(name)
    if vk is not None:
        _send(vk, False)


def key_up(name):
    if _remote is not None:
        _send_remote("RELEASE " + _remote_key(name))
        return
    vk = resolve_vk(name)
    if vk is not None:
        _send(vk, True)


def tap(name, duration=0.06):
    """点按一下（按下 → 松开）。给「开关自动」这类瞬发键用。"""
    if _remote is not None:
        _send_remote("TAP %s %d" % (_remote_key(name), int(duration * 1000)))
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


# ---- 远程后端（agent 跑在控制机，通过 TLS 把按键发到游戏机的 Pro Micro）----
# 默认 _remote=None 走本地 SendInput；调用 use_network() 后切到远程硬件键盘。
_remote = None

_CMD_KEY = {
    "left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN",
    "ctrl": "CTRL", "alt": "ALT", "shift": "SHIFT",
    "enter": "ENTER", "space": "SPACE", "esc": "ESC", "tab": "TAB",
    "del": "DEL", "insert": "INSERT", "home": "HOME", "end": "END",
    "pageup": "PAGEUP", "pagedown": "PAGEDOWN",
}
_CMD_KEY.update({"f%d" % i: "F%d" % i for i in range(1, 13)})


def _remote_key(name):
    """input.py 键名 → 固件键名。"""
    n = str(name).lower()
    return _CMD_KEY.get(n, n)


def _send_remote(line):
    if _remote is not None:
        try:
            _remote.send(line)
        except Exception:
            pass


def use_network(host, port, cafile):
    """切到网络后端（agent 在控制机时，启动阶段调用一次）。"""
    global _remote
    from remote_kbd.kbd_client import KbdClient
    _remote = KbdClient(host, port, cafile)


def use_local():
    """切回本地 SendInput，并关闭远程连接（socket 不泄漏）。"""
    global _remote
    if _remote is not None:
        try:
            _remote.close()
        except Exception:
            pass
    _remote = None


def shutdown():
    """关闭远程键盘连接（如果存在），停止指令传输。GUI 关闭时调用。"""
    use_local()
