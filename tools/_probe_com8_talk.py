"""跟本地 COM8 板子**说两句话**，判定"串口开了但没回执"的根因（2026-10-09 ✓）。

背景（现场症状）：工作台面板那行红字写着 **「指令发不出去：…」** —— 那句话是面板自己打印的
（`gui/player_panel.py::_poll_link_health` ✓），含义是 `link_health()` 为假 ✓ —— 而它**两种都算**
假：① 发送报错；② **发出去了、5 秒内等不到固件回执**（`silent_for` ✗）。
`perf.log` 那边 `send` 有、`send_fail` **零** ⇒ 落在第 ② 种 ✓ ⇒ **我们在发，板子不说话** ✗。

这个脚本就查"板子为什么不说话"，**全程不敲任何键** ✓（`PING` 是固件不认识的指令 ⇒ 固件回一条
`ERR` ✓ 什么 HID 都不发 ✓；`MOVE 0 0` 位移为零 ⇒ 光标不动 ✓）：

  A. 一打开口就冒 `READY` ⇒ **口一开板子就被复位** ✓✓（`READY` 是 `setup()` 里打印的 ✓）
     ⇒ 那么"面板每 5 秒自动重连一次"就是元凶 ✗：重连 ⇒ 重开 ⇒ 复位 ⇒ 8 秒 bootloader ⇒
     这 8 秒里一条回执都没有 ⇒ 判"静默" ⇒ 又重连 ⇒ **永远在引导里转** ✗
     （症状正是"串口连着、一条都发不出去" ✓）。
  B. `PING` ⇒ 回 `ERR` ⇒ **是这一版固件、而且它在跑** ✓（那就查 B 机侧读回执那条 ✓）。
  C. `PING` / `MOVE 0 0` 都**毫无反应** ⇒ 刷的不是这一版固件（或固件卡死 ✗）⇒ 重新烧录
     `remote_kbd/pro_micro/pro_micro.ino` ✓。

⚠ 前提：**工作台必须先把口让出来**（面板把输入设备切到「本地键盘」✓）否则本脚本只会拿到
  `PermissionError` ✓（那不是板子的毛病 ✓）。
"""

import sys
import time

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM8"

import serial  # noqa: E402

print("=== 跟 %s 对话（不敲任何键 ✓）===" % PORT)
t0 = time.time()
try:
    ser = serial.Serial(PORT, 115200, timeout=0.3, write_timeout=1.0)
except Exception as e:
    print("  ✗ 打不开：%s: %s" % (type(e).__name__, e))
    if "denied" in str(e).lower():
        print("  ⇒ **口还被占着** ✗ —— 先把面板的输入设备切到「本地键盘」（那会把口让出来 ✓）")
    sys.exit(1)
try:
    ser.dtr = False                 # 同 SerialKbd（但**复位已经发生了** ✓ 见下面 A ✓）
    ser.rts = False
except Exception:
    pass

print("  已打开（%.2fs）⇒ 开始读 3.5 秒，看有没有开机横幅" % (time.time() - t0))
buf = b""
ready_at = None
while time.time() - t0 < 3.5:
    try:
        d = ser.read(256)
    except Exception as e:
        print("  读异常：%s: %s" % (type(e).__name__, e))
        break
    if d:
        buf += d
        if b"READY" in d and ready_at is None:
            ready_at = time.time() - t0
        print("  t=%.2fs 收到 %r" % (time.time() - t0, d[:100]))

if b"READY" in buf:
    print("  ⭐⭐ **看到 READY**（t=%.2fs）⇒ 这块板子**在打开口的那一刻被复位了** ✓" % ready_at)
    print("     ⇒ 「面板每 5 秒自动重连」会不停把它复位 ⇒ 一直卡在 8 秒 bootloader 里 ⇒"
          " 一条回执都收不到 ✗（症状完全对得上 ✓）")
else:
    print("  （3.5 秒里没有 READY）⇒ 打开口**没有**让它重新走一遍 setup ✓")

print("=== 发 PING（固件不认识 ⇒ 应当回 ERR ✓ 不发任何 HID ✓）===")
try:
    ser.write(b"PING\n")
    ser.flush()
except Exception as e:
    print("  ✗ 写失败：%s: %s" % (type(e).__name__, e))
got = b""
t1 = time.time()
while time.time() - t1 < 2.0:
    try:
        d = ser.read(256)
    except Exception:
        break
    if d:
        got += d
print("  2 秒内收到：%r" % (got[:200],))
if b"ERR" in got:
    print("  ✓ **固件在跑，而且是这一版**（未知指令回 ERR ✓）⇒ 板子没问题 ✓"
          " ⇒ 问题在 B 机侧（读回执 / 连接时序 ✓）")
elif got:
    print("  ⚠ 收到东西但不是 ERR —— 可能是**别的 sketch**（它把自己的输出打出来了 ✗）")
else:
    print("  ✗ **一点反应都没有** ⇒ 刷的不是这个固件（或固件卡死 ✗）"
          " ⇒ 重新烧 `remote_kbd/pro_micro/pro_micro.ino` ✓")

print("=== 发 MOVE 0 0（位移为零 ⇒ 光标不动 ✓ 应当回 DONE ✓）===")
try:
    ser.write(b"MOVE 0 0\n")
    ser.flush()
except Exception as e:
    print("  ✗ 写失败：%s: %s" % (type(e).__name__, e))
got2 = b""
t2 = time.time()
while time.time() - t2 < 2.0:
    try:
        d = ser.read(256)
    except Exception:
        break
    if d:
        got2 += d
print("  2 秒内收到：%r" % (got2[:200],))
if b"DONE" in got2:
    print("  ✓ 双向通 ✓（发得出去、回执也收得到 ✓）")
else:
    print("  ✗ 没有 DONE ⇒ 板子侧确实没在处理指令 ✗")

try:
    ser.close()
except Exception:
    pass
print("=== 收工（口已释放 ✓）===")
