"""手动输入：捕获本机键盘，实时转发到游戏。

开启后，本机按下的每个「游戏支持」的键，都会通过 decision.input 的
后端（本地 SendInput 或远程 Pro Micro）转发到游戏，实现玩家手动接管控制。

**可以和自动打怪同时开着**（不需要先关自动）。代价是两边可能抢同一个键：

    固件侧按住的键是个**集合**（`addHeld` 里已经有就返回 false），不是引用计数 ——
    谁先发 RELEASE 谁就真把那个键松开。所以「自动正按着左、你手动松一下左」之后，
    固件那边左已经松了，而决策层的 KeyState 还以为按着、之后不再补发 PRESS。

对策是每次手动按键都请求一次「按键状态重同步」（见 `_request_resync`）：
决策层清掉本地按键状态，下一帧重新声明它需要的键。清状态**不发** RELEASE，
所以不会让自动的按键闪断。
"""

import threading

from decision import input as dinput
from pynput import keyboard

# pynput 特殊键 → input.py 键名
_KEY_MAP = {
    keyboard.Key.left: "left",
    keyboard.Key.right: "right",
    keyboard.Key.up: "up",
    keyboard.Key.down: "down",
    keyboard.Key.ctrl: "ctrl",
    keyboard.Key.ctrl_l: "ctrl",
    keyboard.Key.ctrl_r: "ctrl",
    keyboard.Key.alt: "alt",
    keyboard.Key.alt_l: "alt",
    keyboard.Key.alt_r: "alt",
    keyboard.Key.shift: "shift",
    keyboard.Key.shift_l: "shift",
    keyboard.Key.shift_r: "shift",
    keyboard.Key.enter: "enter",
    keyboard.Key.space: "space",
    keyboard.Key.esc: "esc",
    keyboard.Key.tab: "tab",
    keyboard.Key.backspace: "backspace",
    keyboard.Key.delete: "del",
    keyboard.Key.insert: "insert",
    keyboard.Key.home: "home",
    keyboard.Key.end: "end",
    keyboard.Key.page_up: "pageup",
    keyboard.Key.page_down: "pagedown",
}
for _i in range(1, 13):
    _KEY_MAP[getattr(keyboard.Key, "f%d" % _i)] = "f%d" % _i


def _key_name(key):
    """pynput key → input.py 键名；只返回 input.py 认识（有 VK 码）的键。"""
    name = _KEY_MAP.get(key)
    if name is None:
        ch = getattr(key, "char", None)
        if ch:
            name = ch.lower()
            if name in ("`", "~"):
                name = "grave"
    # 本机操作键（F10 触控板开关 / F11 开关自动 / F12）不转发。真正拦住它们的是
    # decision/input.py 的 LOCAL_ONLY，这里先挡一道是为了少一次无谓的
    # 「按键状态重同步」—— 那会让决策层白清一次按键状态。
    if name and not dinput.is_local_only(name) and dinput.resolve_vk(name) is not None:
        return name
    return None


_listener = None
_pressed = set()
_lock = threading.Lock()


def _request_resync():
    """让决策层重同步按键状态（下一帧重新声明它需要的键）。

    背景见模块文档：固件侧的按键记账是个集合，手动输入松开的键如果正好是自动
    也按着的，固件那边就真松了，而决策层还以为按着、之后不再补发 PRESS。
    清一次本地状态（**不发** RELEASE）就能让它下一帧重新补上，不会被看出闪断。
    """
    try:
        from decision.agent import settings
        settings.input_resync = True
    except Exception:
        pass


def _on_press(key):
    name = _key_name(key)
    if not name:
        return
    with _lock:
        if name in _pressed:
            return   # 按住自动重复，去重
        _pressed.add(name)
    _request_resync()
    dinput.key_down(name)


def _on_release(key):
    name = _key_name(key)
    if not name:
        return
    with _lock:
        _pressed.discard(name)
    _request_resync()
    dinput.key_up(name)


def start():
    """开启手动输入：启动全局键盘监听。"""
    global _listener
    stop()
    _listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    _listener.start()


def stop():
    """关闭手动输入：停止监听并释放所有已转发的键（防卡键）。"""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None
    with _lock:
        held = list(_pressed)
        _pressed.clear()
    if held:
        # 这里发的 RELEASE 可能把自动也正按着的键一并松开（固件侧是个集合），
        # 所以同样要请决策层重同步一次，让它下一帧把需要的键补回来。
        _request_resync()
    for name in held:
        try:
            dinput.key_up(name)
        except Exception:
            pass


def active():
    return _listener is not None
