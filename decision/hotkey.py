"""系统级全局热键（RegisterHotKey + WM_HOTKEY）。

让「开启自动」的 F11 在**任何窗口聚焦时**都能触发 —— 普通 QShortcut 只在
工作台 GUI 有焦点时有效，而自动打怪时焦点在游戏窗口（MapleNecrocer 的
Playmode），必须用系统级热键 RegisterHotKey 才能跨窗口捕获。
"""

import ctypes
from ctypes import wintypes

MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

VK_F11 = 0x7A


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", ctypes.c_long * 2),
    ]


_user32 = ctypes.windll.user32


def register(hwnd, hotkey_id, vk, modifiers=MOD_NOREPEAT):
    """注册全局热键。成功返回 True，失败（被占用）返回 False。"""
    try:
        return bool(_user32.RegisterHotKey(int(hwnd), int(hotkey_id),
                                           int(modifiers), int(vk)))
    except Exception:
        return False


def unregister(hwnd, hotkey_id):
    try:
        _user32.UnregisterHotKey(int(hwnd), int(hotkey_id))
    except Exception:
        pass


def is_hotkey(message_ptr, hotkey_id):
    """nativeEvent 的 message 指针 → 是不是指定的 WM_HOTKEY。"""
    try:
        msg = MSG.from_address(int(message_ptr))
        return msg.message == WM_HOTKEY and msg.wParam == int(hotkey_id)
    except Exception:
        return False
