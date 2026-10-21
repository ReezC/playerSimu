# -*- coding: utf-8 -*-
"""Win32 胶水：ctypes 结构体 + `user32` 句柄（**全仓只此一份** ✓）。

为什么独立成文件：`SendInput`（发键）与 `GetCursorPos/SetCursorPos`（指针）
都要这套结构体 ⇒ 放在 `input.py` 里就成了"指针模块反向依赖传输模块"✗
（那是分层反了 ✓）。现在两边都往下依赖它 ✓ 方向正确 ✓。
"""
import ctypes
from ctypes import wintypes  # noqa: F401  —— 结构体字段类型要用 ✓




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




#: 公开名（不加下划线 ✓）：`execution.input`（发键）和 `execution.pointer`（指针）
#: 都要用它 ⇒ 它是**这一层的公共件** ✓ 不是谁的私有物 ✓
user32 = ctypes.windll.user32
