"""查"这台机器是谁 + 现在前台是谁 + 游戏在不在这台机上"（2026-10-09 ✓）。

现场症状：面板绿字「已连接 ProMicro(本地)」✓、开启自动后「Agent 当前按键」**有** ✓，
但**角色不动** ✗。三条已知事实摆在一起，最可能是"**按键打到了另一台机器 / 另一个窗口**"✗
—— 所以这个脚本只读三样东西：
  ① 这台机器的 IP（对照 `config/link.yaml` 的 a_host / b_host ✓ 看自己是 A 还是 B ✓）；
  ② **当前前台窗口**（HID 按键只会送到前台窗口 ✗ —— 游戏不是前台就一个键都收不到 ✓）；
  ③ 游戏进程在不在这台机上（`冒险岛` / `Maplestory` ✓ 路径 + 启动时间 + 有没有主窗口 ✓）。
"""

import ctypes
import time

import psutil

u = ctypes.windll.user32


def fg_pid():
    """当前前台窗口属于哪个进程（0 = 判不出来）。"""
    pid = ctypes.c_ulong(0)
    u.GetWindowThreadProcessId(u.GetForegroundWindow(), ctypes.byref(pid))
    return int(pid.value)


def win_title(hwnd):
    buf = ctypes.create_unicode_buffer(512)
    u.GetWindowTextW(hwnd, buf, 512)
    return buf.value


print("=== ① 这台机器是谁 ===")
try:
    from core.config import get
    print("  link.yaml: a_host(A机/游戏机) = %r   b_host(B机/工作台机) = %r"
          % (get("", "a_host"), get("", "b_host")))
except Exception as e:
    print("  （读不到 link.yaml：%s: %s）" % (type(e).__name__, e))
try:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("192.168.1.9", 9))
    print("  本机在 192.168.1.x 网段的地址 = %s" % s.getsockname()[0])
    s.close()
except Exception as e:
    print("  （探不到本机地址：%s）" % (e,))

print("=== ② 当前前台窗口（HID 按键只会送到这里 ✗）===")
h = u.GetForegroundWindow()
print("  标题：%r   进程 PID=%s" % (win_title(h), fg_pid()))

print("=== ③ 游戏在不在这台机上 ===")
here = fg_pid()
found_game = False
for p in psutil.process_iter(["pid", "name", "exe", "create_time"]):
    nm = (p.info["name"] or "")
    exe = (p.info["exe"] or "")
    blob = (nm + " " + exe).lower()
    if not any(k in blob for k in ("maplestory", "冒险岛", "nexon", "blackcipher")):
        continue
    found_game = True
    try:
        t = time.strftime("%H:%M:%S", time.localtime(p.info["create_time"] or 0))
    except Exception:
        t = "?"
    is_fg = (p.info["pid"] == here)
    print("  PID %-6s %-24s 起于 %s  前台=%s" %
          (p.info["pid"], nm[:24], t, "是 ✓" if is_fg else "不是 ✗"))
    print("        exe = %s" % (exe or "?",))
if not found_game:
    print("  ⇒ **这台机器上没有冒险岛 / 它的启动器 / 反作弊进程** ✗"
          "（那『本地板』打的这台机器上根本没有游戏 ✓）")
elif here != 0:
    print("  ⇒ 前台那个窗口（PID %s）是不是游戏进程，自己看上面那一列 ✓" % here)
