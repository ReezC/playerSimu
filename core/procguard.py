# -*- coding: utf-8 -*-
"""**把子进程整棵树绑在"父进程一死就一起走"的 Job Object 上**（用户 2026-10-05 ✓）。

⚠⚠ **病根（实测 ✓）**：用户问"数据工作台 / 测谎演示有没有**杀不掉的死进程**" ✓ —— 查下来：
  · `tools/lie_demo.py`（测谎演示）✓：关窗 `closeEvent → DetsWorker.close()` ✓，而且就算不收尸，
    子进程也会在**管道 EOF** 时自己退 ✓（实测 ✓）⇒ **干净** ✓；
  · `gui/yolo_workbench.py`（YOLO 工作台）✗✗：**训练 / 验证**是 `QProcess` ✓，只有按「停止」
    才 `kill()` ✗、关窗时**谁也没管它** ✗ —— 实测（同款手法见 `tools/selftest_yolo_wb.py` ✓）：
    **父进程关窗退出后，子进程照样在干活** ✗✗；而且 `ultralytics` 自己还会叉 **dataloader 子进程**
    ✗ ⇒ 留下的是一棵**看不见的进程树** ✓（没窗口、任务管理器里只有一堆 `python.exe` ✗、
    显存还被占着 ✗、单杀一个还不一定干净 ✓）= 用户说的"杀不掉" ✓。

⇒ 处置（用户 2026-10-05 ✓ 原话："**把训练/验证子进程塞进一个 Job Object（KILL_ON_JOB_CLOSE ✓）
  ⇒ 父进程一死（崩溃、强杀都算 ✓）整棵树被系统收走**" ✓✓）：
  · `guard(pid)`：把这个进程塞进**本进程专属**的那个 job ✓（`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` ✓）
    ⇒ **它之后叉出来的子孙进程也自动落在同一个 job 里** ✓（Windows 的语义 ✓）；
  · 本进程一死（**正常退出、崩溃、被强杀都算** ✓）⇒ 句柄被系统收掉 ⇒ **job 关闭 ⇒ 内核把这棵树
    全干掉** ✓✓ —— 这是"看不见的进程树"这一类问题的标准解 ✓；
  · `kill_tree(pid)`：`taskkill /T /F` ✓ —— 给「停止」按钮 / 正常关窗用 ✓
    （⚠ **必须连树** ✗：只 `kill()` 直接孩子的话，dataloader 那些孙进程会留下来 ✗）；
  · `release()`：主动把 job 关掉（= 把里面**整棵树**清场 ✓），关窗最彻底的那一手 ✓。

⚠ **四条纪律**（都踩过 / 都想过 ✓）：
  · 非 Windows 一律**空转**（返回 `False` ✓ 不抛 ✓）：本仓库只跑 Windows ✓，别处保持原行为 ✓；
  · **一切都不抛异常** ✗：这是兜底代码 ✓ —— 失败最多退回老行为（有死进程 ✗），
    绝不能让界面起不来 ✓；
  · `guard` 要**尽早**叫 ✗（赶在子进程叉孙进程之前 ✓）：接在 `QProcess.started` 上 ✓
    （训练那行 `from ultralytics import YOLO` 要好几秒 ✓、dataloader 更晚 ✓ ⇒ 实测够早 ✓）；
  · ⚠ **绑错了也救不回已经叉出去的孙进程** ✗（job 只管"绑定之后"的新生进程 ✓）——
    所以"尽早 + 事后 `kill_tree` "两条一起上 ✓。
"""
import ctypes
import os
import subprocess
import threading
from ctypes import wintypes

#: `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`（winnt.h ✓）：job 的**最后一个句柄**一关 ⇒ 内核杀整棵树 ✓
_KILL_ON_JOB_CLOSE = 0x2000
#: `JobObjectExtendedLimitInformation`（winnt.h 的枚举值 ✓）
_EXTENDED_LIMIT = 9
#: `AssignProcessToJobObject` 要的权限：`PROCESS_SET_QUOTA | PROCESS_TERMINATE`（MSDN ✓）
_PROC_ACCESS = 0x0100 | 0x0001
#: 不弹黑框（与 `deploy/runner.py` / `core/wzexport.py` **同一处口径** ✓）
_NO_WINDOW = 0x08000000

#: `None` = 还没建过 ✓；`_MISS` = **试过了、建不起来** ✓（别每拍都重试 ✗）
_MISS = object()
_JOB = None
_LOCK = threading.Lock()


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


class _BASIC_LIMIT(ctypes.Structure):
    """⚠ `Affinity` **必须按指针宽度**（`ULONG_PTR` ✓）✗✗ —— 写成 `DWORD` 在 64 位上整个结构会
    错位 ⇒ `SetInformationJobObject` 失败 ✓（网上抄来的版本常踩这个 ✓）。"""
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _EXT_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _job():
    """本进程专属的 job 句柄（懒建 ✓ 只建一次 ✓；建不起来回 `None` ✓ 不抛 ✓）。"""
    global _JOB
    if os.name != "nt":
        return None
    with _LOCK:
        if _JOB is None:
            _h = 0
            try:
                _k = ctypes.windll.kernel32
                _k.CreateJobObjectW.restype = wintypes.HANDLE
                _k.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
                _h = _k.CreateJobObjectW(None, None)
                if not _h:
                    _JOB = _MISS
                else:
                    _info = _EXT_LIMIT()
                    _info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
                    _k.SetInformationJobObject.restype = wintypes.BOOL
                    _k.SetInformationJobObject.argtypes = [
                        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
                    _ok = _k.SetInformationJobObject(
                        wintypes.HANDLE(_h), _EXTENDED_LIMIT,
                        ctypes.byref(_info), ctypes.sizeof(_info))
                    if _ok:
                        _JOB = _h
                    else:
                        _k.CloseHandle(wintypes.HANDLE(_h))
                        _JOB = _MISS
            except Exception:                  # noqa: BLE001 —— 兜底代码，绝不抛 ✓
                _JOB = _MISS
        return None if _JOB is _MISS else _JOB


def guard(pid):
    """把 `pid`（**及它之后叉出来的子孙** ✓）塞进"父死即全灭"的 job ✓。回是否成功 ✓，不抛 ✓。"""
    try:
        _pid = int(pid or 0)
    except Exception:                          # noqa: BLE001
        return False
    if os.name != "nt" or _pid <= 0:
        return False
    try:
        _h = _job()
        if not _h:
            return False
        _k = ctypes.windll.kernel32
        _k.OpenProcess.restype = wintypes.HANDLE
        _k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        _hp = _k.OpenProcess(_PROC_ACCESS, False, _pid)
        if not _hp:
            return False
        try:
            _k.AssignProcessToJobObject.restype = wintypes.BOOL
            _k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            return bool(_k.AssignProcessToJobObject(wintypes.HANDLE(_h),
                                                    wintypes.HANDLE(_hp)))
        finally:
            _k.CloseHandle(wintypes.HANDLE(_hp))
    except Exception:                          # noqa: BLE001
        return False


def kill_tree(pid):
    """**连树**杀（`taskkill /T /F` ✓）——「停止」按钮 / 正常关窗走这个 ✓、不抛 ✓。

    ⚠ 为什么不用 `Popen.kill()` / `QProcess.kill()` ✗：那**只杀直接孩子** ✗ ⇒
      `ultralytics` 的 dataloader 孙进程会留下来 ✓（= 用户看到的那种"杀不掉"✗）。
    """
    try:
        _pid = int(pid or 0)
    except Exception:                          # noqa: BLE001
        return False
    if os.name != "nt" or _pid <= 0:
        return False
    try:
        _r = subprocess.run(["taskkill", "/T", "/F", "/PID", str(_pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=_NO_WINDOW)
        return int(_r.returncode) == 0
    except Exception:                          # noqa: BLE001
        return False


def release():
    """主动把 job 关掉 ⇒ 里面**整棵树立刻被内核干掉** ✓（关窗最彻底的那一手 ✓）。

    ⚠ 关掉之后本进程还会**重新建一个** job ✓（后续 `guard` 照旧能用 ✓）⇒ 可安全重复调 ✓。
    """
    global _JOB
    if os.name != "nt":
        return False
    with _LOCK:
        _h, _JOB = _JOB, None
    if not _h or _h is _MISS:
        return False
    try:
        return bool(ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(_h)))
    except Exception:                          # noqa: BLE001
        return False
