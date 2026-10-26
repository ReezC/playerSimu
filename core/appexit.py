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
import threading
import time

#: 硬退前要跑的动作（**由下层自己登记** ✓ 别在这里 import 上层 ✗ —— `core → perception`
#: 那条是"函数内向上依赖"，分层扫描会当场打红 ✓ 2026-10-21 试过）。
_HOOKS = []

#: ⭐⭐⭐⭐ **"自尽兜底"等多久**（秒）。`hard_exit` 一进来就武装它：
#: 这么久还没退干净 ⇒ `TerminateProcess(自己)`（内核级，绝不回解释器 ✓）。
#:
#: **为什么连 `os._exit` 都不能信**（2026-10-23 现场，用户："**进程又没杀掉 请根治**"）：
#:   `crash.log` 里抓到的是这么一段（`pythonw` 无控制台也照样写进去了 ✓）：
#:   ```
#:   Current thread 0x00008de0 (most recent call first):
#:     File "core/appexit.py", line 73 in hard_exit      ← 就是 `os._exit(code)` 那一行
#:     File "gui/app.py", line 228 in _hard_exit
#:     File "gui/app.py", line 383 in main
#:   Windows fatal exception: code 0xc000000d            ← STATUS_INVALID_PARAMETER
#:   ```
#:   ⇒ `os._exit()` 在**收尾时自己炸了**、而且**没退成**（函数返回了）⇒ 又掉回
#:     `Py_Main → Py_RunMain` 那条解释器收尾 ⇒ 撞 TSF/IME 死等 ⇒ **窗口没了、进程还挂着** ✓
#:     （实测：`pid 64052` `pythonw -m gui.app`，89 条线程、2.8 GB、还攥着 `udp_5000.lock`）。
#:   ⇒ 结论：**"能退"这件事不能只靠一条路** —— 见 `_arm_suicide` / `_terminate_self` ✓。
_SELF_KILL_S = 3.0


def _crash_note(text):
    """往崩溃日志里留一句（尽力而为：进程正在死，写不进去就算了 ✓ 绝不抛 ✓）。"""
    try:
        from core import instance as _inst
        p = _inst.crash_path()
        with open(p, "a", encoding="utf-8") as fh:
            fh.write("\n=== 退出兜底 %s ===\n%s\n"
                     % (time.strftime("%Y-%m-%d %H:%M:%S"), text))
    except Exception:                      # noqa: BLE001 —— 兜底代码，绝不抛 ✓
        pass


def _terminate_self(code=0):
    """**内核级自尽**：`TerminateProcess(GetCurrentProcess(), code)` —— 不跑解释器收尾 ✓。

    为什么不是 `os._exit`（它是"软"的：走 CRT/`ExitProcess`，会跑 DLL 卸载）：
      2026-10-23 现场 `os._exit` 自己抛了 `Windows fatal exception: code 0xc000000d`
      并且**没退成**（见 `_SELF_KILL_S` 那段）⇒ 然后掉回解释器收尾 ⇒ 撞 TSF/IME 死等 ✗。
      `TerminateProcess` 是**内核直接处死**：不跑 `atexit`、不析构 Qt、不等别的线程 ✓。
    """
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k.TerminateProcess.restype = ctypes.c_int
        # `GetCurrentProcess()` 返回伪句柄 `(HANDLE)-1` ✓（不用 OpenProcess，一定拿得到 ✓）
        k.TerminateProcess(ctypes.c_void_p(-1), ctypes.c_uint(int(code or 0)))
    except Exception:                      # noqa: BLE001
        pass
    # Windows 上 `os.kill(pid, 9)` 就是 TerminateProcess ✓（上一步真成了就根本走不到这儿 ✓）
    try:
        os.kill(os.getpid(), 9)
    except Exception:                      # noqa: BLE001
        pass
    time.sleep(0.2)                        # 还没死？那也没有别的招了（返回值不会被用到 ✓）


def _arm_suicide(code=0):
    """武装"自尽兜底"：`_SELF_KILL_S` 秒后还没退干净 ⇒ 内核级强杀 ✓（返回线程，不 join ✓）。

    ⚠ 只在这个函数里武装 ⇒ **正常运行绝不会被它误杀** ✓（进程退出了，守护线程跟着没 ✓）。
    """
    def _fire():
        time.sleep(max(0.1, float(_SELF_KILL_S)))
        _crash_note("超过 %.1f 秒还没退干净 ⇒ 走 TerminateProcess（`os._exit` 没退成，"
                    "见本文件 `_SELF_KILL_S` 那段现场 ✓）" % _SELF_KILL_S)
        _terminate_self(code)

    th = threading.Thread(target=_fire, name="exit-suicide", daemon=True)
    try:
        th.start()
    except Exception:                      # noqa: BLE001 —— 线程都起不来也别耽误退出 ✓
        pass
    return th


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
    """清完该清的 ⇒ `os._exit(code)`（**不返回** ✓）。两个入口都走它（工作台 / 部署台 ✓）。

    ⭐ 2026-10-23 加固（用户："**进程又没杀掉 请根治**"）：**三层**，任何一层成了都算成功 ✓
      ① 先武装 `_arm_suicide` —— 后面无论卡在哪一步，`_SELF_KILL_S` 秒后内核强杀 ✓；
      ② 照旧做清理（刷缓冲 / 放端口锁 / `at_exit_hook` 登记的动作 ✓）；
      ③ `os._exit` 放 `try` 里、`finally` 兜 `_terminate_self` —— **它要是没退成**
        （2026-10-23 现场：`os._exit` 自己抛 `0xc000000d` 然后返回了 ✗），
        `finally` 立刻换成 `TerminateProcess` ✓，**绝不再掉回解释器收尾**（TSF/IME 死等 ✓）。
    """
    _code = int(code or 0)
    _arm_suicide(_code)                    # ① 先武装（**必须在所有清理之前** ✗）
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
    try:
        os._exit(_code)                    # ③ 正常路径（走到这儿就再也不返回 ✓）
    finally:
        # 走到这里 = **`os._exit` 没退成**（崩了 / 被拦了）⇒ 绝不许掉回解释器收尾 ✗
        _crash_note("`os._exit(%d)` 没有退出（它自己抛异常了）⇒ 改用 TerminateProcess"
                    % _code)
        _terminate_self(_code)
