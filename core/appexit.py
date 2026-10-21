"""**退出必须真的走**：清完该清的 ⇒ `os._exit(code)`，不回解释器收尾 ✓。

⭐⭐⭐⭐ 2026-10-21（用户：「**窗口关掉但是实例清不掉**」）

**现场**（同一台机器上抓到的）：窗口早就关了、`pythonw -m gui.app` 还挂着（72 条线程），
`data/locks/udp_5000.lock` 里还写着它的 pid ⇒ 下次启动只会说「UDP 端口已被另一个工作台实例占用」
⇒ 用户看到的就是**实例清不掉** ✓。

**卡在哪**（`py-spy dump --native` 抓的原生栈，不是猜的）：
```
<module> (gui/app.py:339)                 ← `sys.exit(main())`，main 早就返回了
Py_Main → Py_RunMain → Py_PreInitialize   ← **解释器收尾**（不是我们自己的清理 ✗）
QWidget::~QWidget (Qt5Widgets)            ← 收尾时销毁最后一个 QWidget
QWindowPrivate::destroy / qwindows.dll
NtUserDestroyWindow → PostMessageW → MSCTF/TSF（输入法）CtfImeDestroyThreadMgr
 → CtfImmCoUninitialize → CoreMessaging.dll → ZwAlpcSendWaitReceivePort   ← **永远等下去**
```
Windows 上 TSF/IME 收尾卡死是出了名的（`pythonw` 无控制台、远程桌面断开时更容易撞 ✓），
而且它发生在**我们的清理之后**（那段有 `gui/app.py` 的 `WATCHDOG_EXIT_S` 看门狗兜着 ✓）
⇒ 看门狗救不到 ✗。

⇒ 和那个看门狗同一条口径：**收尾卡住就是元凶 ⇒ 绕开它** ✓

⚠ `os._exit` 之后这些事都做不了，所以走之前替它们做掉：
  · `stdout/stderr` 刷一下（`_ensure_std_streams` 可能给了缓冲 ✓）；
  · **放掉本进程占的 UDP 端口锁**（`instance.release_my_ports` ✓）—— 这正是
    「窗口关了进程还活着 ⇒ 实例被挡」那把锁 ✓；
  · 全仓库**唯一**一个 `atexit` 依赖：`perception.map_client` 收小地图定位的 TCP ✓。
  行为日志不用管：`core/behavior._write` 是**逐行 open/close**，没有跨行缓冲 ✓。
"""

from __future__ import annotations

import os
import sys

#: 硬退前要跑的动作（**由下层自己登记** ✓ 别在这里 import 上层 ✗ —— `core → perception`
#: 那条是"函数内向上依赖"，分层扫描会当场打红 ✓ 2026-10-21 试过）。
_HOOKS = []


def at_exit_hook(fn):
    """登记一个"硬退之前跑一下"的动作 ⇒ `True` = 登记上了 ✓（重复登记同一个函数只算一次 ✓）。"""
    try:
        if not callable(fn) or any(h is fn for h in _HOOKS):
            return False
        _HOOKS.append(fn)
        return True
    except Exception:                      # noqa: BLE001
        return False


def hard_exit(code=0):
    """清完该清的 ⇒ `os._exit(code)`（**不返回** ✓）。两个入口都走它（工作台 / 部署台 ✓）。"""
    try:
        if sys.stdout is not None:
            sys.stdout.flush()
        if sys.stderr is not None:
            sys.stderr.flush()
    except Exception:                      # noqa: BLE001
        pass
    try:
        from core import instance as _inst
        _inst.release_my_ports()
    except Exception:                      # noqa: BLE001
        pass
    # 替 atexit 做（`os._exit` 不跑 atexit ✓）：谁要收尾谁自己 `at_exit_hook(...)` 登记 ✓
    for fn in list(_HOOKS):
        try:
            fn()
        except Exception:                  # noqa: BLE001 —— 一个收尾动作崩了别耽误退出 ✗
            pass
    os._exit(int(code or 0))
