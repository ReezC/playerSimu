"""当前进程跑在**哪个 Windows 会话** —— 判断「**硬件键盘（ProMicro 的HID）能不能打进这个会话**」。

为什么要有这个模块（用户 2026-10-19 现场 ✓ 原话：
「**C机远程控制B机环境下 B机的本地ProMicro操作失效了**」）：

  ProMicro 在游戏眼里是一块**真·USB 键盘**（HID）。Windows 把**物理**键盘 / HID 的输入
  只喂给**物理控制台会话**（插着显示器键盘的那一个 ✓）。而 C 用**远程桌面（RDP）**连上 B 时，
  会**新建另一个会话** —— 游戏（`Maplestory_Classic.exe`）和本工作台都在那个会话里
  ⇒ ProMicro 打出来的键**一个都进不去**（进的是没人看的控制台会话）✗
  ⇒ 现象就是「**ProMicro 操作全都失效**」，而串口、固件、链路**全都正常** ✓
  （所以从链路上查一辈子也查不出来 ✗）。

实测（2026-10-19，B 机 `LOCKERSHOME`，C 正远程控着它）：
```text
 qwinsta →
   >rdp-tcp#0  Locker  1  Active   ← C 的远程桌面会话（工作台 + 游戏都在这）
    console             2  Conn     ← 物理控制台：HID 只会进这里
 本进程 session = 1          ← ProcessIdToSessionId(自己)
 COM8 Arduino Leonardo       ← 串口好好的（不是占用 / 不是驱动问题 ✓）
```

判据只有一条（**不猜** ✓）：
  `ProcessIdToSessionId(自己) == WTSGetActiveConsoleSessionId()` ⇒ 硬件 HID 能到这里 ✓；
  不等 ⇒ 软/硬件键盘打不到本会话（本机 ProMicro 输出必然"看起来没用"✓）。

⚠ 只管**硬件 HID** 那条路（ProMicro 直连 / 本地 relay）✗ ——
  「本机 SendInput」不受影响（`SendInput` 发的是**本会话**的输入 ✓ 所以远程桌面里
  用 SendInput 是能打的 ✓）；「ProMicro(远程)」也不受影响（HID 打在**被控机**的会话上 ✓
  只有那台机器自己也在远程桌面会话里才会踩同一个坑 ✓）。
"""

import ctypes
import os

#: `WTSGetActiveConsoleSessionId()` 说"没有控制台会话"时的返回值
_NO_CONSOLE = 0xFFFFFFFF

#: 缓存（会话不会中途变；测试里用 `refresh()` 清 ✓）
_INFO = {}


def _proc_sid():
    """本进程所在会话号；取不到 ⇒ None（**判不了就不报警** ✓）。"""
    try:
        sid = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.ProcessIdToSessionId(
            int(os.getpid()), ctypes.byref(sid))
        return int(sid.value) if ok else None
    except Exception:                                   # noqa: BLE001
        return None


def _console_sid():
    """物理控制台会话号；没有 / 取不到 ⇒ None。"""
    try:
        v = int(ctypes.windll.kernel32.WTSGetActiveConsoleSessionId())
        return None if v == _NO_CONSOLE or v < 0 else v
    except Exception:                                   # noqa: BLE001
        return None


def info(refresh=False):
    """`{"sid", "console_sid", "hid_ok", "why"}` —— 一次算清，之后走缓存 ✓。

    · `hid_ok` 为 **True** 的三种情形：两台会话相同 ✓ / 判不出来（拿不到会话号 ✓）/
      机器上根本没有控制台会话（纯服务器机 ✓）—— **一律放行**（没证据就不吓人 ✓
      同 `map_locator.crop_view_trusted` / `remote.panel_mismatch` 那条纪律 ✓）。
    """
    if _INFO and not refresh:
        return dict(_INFO)
    sid = _proc_sid()
    con = _console_sid()
    if sid is None or con is None:
        hid_ok = True
        why = ""
    elif sid == con:
        hid_ok = True
        why = ""
    else:
        hid_ok = False
        why = ("本进程在**会话 %d**、物理控制台是**会话 %d** ⇒ "
               "ProMicro 的硬件按键只会打进控制台那个会话，本会话收不到"
               % (sid, con))
    _INFO.update({"sid": sid, "console_sid": con, "hid_ok": hid_ok, "why": why})
    return dict(_INFO)


def hid_ok():
    """硬件键盘（ProMicro HID）能不能打进本会话。"""
    return bool(info().get("hid_ok"))


def warning_text():
    """给人看的一句话（能打进 ⇒ 空串 ✓ 界面据此决定要不要提示 ✓）。"""
    d = info()
    if d.get("hid_ok"):
        return ""
    return ("⚠ 本工作台跑在**远程桌面会话**里（会话 %s / 控制台 %s）——"
            "ProMicro 是硬件键盘，按键只会打进**物理控制台**那个会话 ⇒ "
            "游戏在本会话里一个键都收不到。三种解法：\n"
            " · ① C 改用**镜像类**远程工具连 B 的控制台会话"
            "（ToDesk / RustDesk「控制台/物理桌面」等，别用 RDP 新建会话）；\n"
            " · ② 把**当前会话搬到控制台**：管理员 CMD 里 `tscon %s /dest:console`"
            "（游戏/工作台都不用重启；RDP 窗口会当场断开，之后用镜像工具或 `mstsc /admin` 再看）"
            "—— 之后 `qwinsta` 里本进程那一行的会话号要和 console 那行**相同**才算成 ✓；\n"
            " · ③ 让 C 用 `mstsc /admin` 连（老式管理连接，有时能直接落到控制台会话上；"
            "连上后用 `qwinsta` 验会话号 ✓）。"
            % (d.get("sid"), d.get("console_sid"), d.get("sid")))


def refresh():
    """重算（自检 / 用户点了「重连」时用 ✓）。"""
    _INFO.clear()
    return info()
