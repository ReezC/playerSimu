"""本地 ProMicro（COM8）**只读**体检 —— 不写一个字节给板子 ✓。

用户报"本地 COM8 指令发不出去"（2026-10-09 ✓）。这个脚本按 `SerialKbd` **完全一样**的方式
去开那个口，但**只读不写**（不敲任何键 ⇒ 不会打到本机窗口里 ✓），把下面这些查清楚：

  1. pyserial 装没装（`use_serial` 全靠它 ✓ 没装 ⇒ `SerialKbd.__init__` 里 ImportError
     ⇒ `connect_async` 判 "fail" ⇒ **静默回退成本地 SendInput** ✗ 正是"发不出去"的典型表象 ✓）；
  2. 系统里到底有哪些串口 + 各自的 **VID:PID**（决定自动发现找不找得到 ✓）；
  3. **COM8 能不能打开** —— 打不开时区分两种：
       · `PermissionError`（Access is denied）⇒ **被别的程序占着** ✗（另一个工作台实例 / relay /
         串口助手 ✓ —— 用户这两天在开第二个工作台 ✓ 这是头号嫌疑 ✓）；
       · `FileNotFoundError` ⇒ 口不存在（板子没插 / 换了号 ✓）；
  4. 能打开的话：**读 2.5 秒**看固件有没有开机横幅 / READY 之类 ✓（有 = 固件活着 ✓；
     一片安静也可能是固件只在收到指令才回话 ✓ 见固件那边 ✓）。
"""

import sys
import time

print("=== 1. pyserial ===")
try:
    import serial
    print("  装了 ✓ 版本", getattr(serial, "__version__", "?"))
except Exception as e:
    print("  ✗ 没装 / 导不进来：%s：%s" % (type(e).__name__, e))
    print("  ⇒ `SerialKbd` 一开场就 ImportError ⇒ 上层判 \"fail\" ⇒ 静默回退本地键盘 ✗")
    print("  ⇒ 装一下：pip install pyserial")
    sys.exit(0)

print("=== 2. 系统里的串口 ===")
ports = []
try:
    import serial.tools.list_ports as lp
    for p in lp.comports():
        ports.append(p.device)
        print("  %-8s hwid=%s" % (p.device, p.hwid))
except Exception as e:
    print("  列不出来：%s：%s" % (type(e).__name__, e))
if not ports:
    print("  （一个串口都没有 ⇒ 板子没插上 / 驱动没装 ✓）")

print("=== 3. 自动发现会挑哪一个 ===")
try:
    sys.path.insert(0, ".")
    from remote_kbd.serial_kbd import find_pro_micro_port, _PRO_MICRO_VIDPIDS
    print("  认的 VID:PID：", _PRO_MICRO_VIDPIDS)
    print("  find_pro_micro_port() ⇒", find_pro_micro_port())
except Exception as e:
    print("  查不了：%s：%s" % (type(e).__name__, e))

print("=== 4. 按 SerialKbd 的方式开 COM8（只读 ✓）===")
PORT = sys.argv[1] if len(sys.argv) > 1 else "COM8"
print("  目标口：%s" % PORT)
try:
    ser = serial.Serial(PORT, 115200, timeout=0.3, write_timeout=1.0)
except Exception as e:
    print("  ✗ 打不开：%s: %s" % (type(e).__name__, e))
    kind = type(e).__name__
    if kind == "PermissionError" or "denied" in str(e).lower():
        print("  ⇒ **口被占着** ✗ —— 谁在占？按可能性：")
        print("     ① 另一个工作台实例（`input_device` 是项目级的，两边都用本地就是两口板 ·"
              " 但同一个 COM8 只能开一次 ✓ 2026-10-09 你在开第二个工作台 ✓）")
        print("     ② 串口监视器 / Arduino IDE 开着 ✓")
        print("     ③ 那个 relay（只在游戏机上跑 ✓ 本机不该有 ✓）")
        print("     查法（任选）：关掉另一个工作台再试 / 任务管理器看还有没有 python 在跑 ✓")
    elif "could not open" in str(e).lower() or kind == "SerialException":
        print("  ⇒ 口存在但开不了：多半也是被占 ✓（或驱动异常，试试拔插 + 换 USB 口 ✓）")
    else:
        print("  ⇒ 口不存在 ⇒ 板子没插 / 换了号 ⇒ 把 `config/link.yaml` 的 `serial_local`"
              " 改成实际号 ✓（或插着让它自动发现 ✓）")
    sys.exit(1)

print("  打开了 ✓")
try:
    ser.dtr = False      # 同 SerialKbd：别让 32U4 复位 ✓
    ser.rts = False
except Exception:
    pass
time.sleep(0.5)
print("=== 5. 读 2.5 秒（固件开机横幅 / READY ✓ 不写一个字节 ✓）===")
t0 = time.time()
got = b""
while time.time() - t0 < 2.5:
    try:
        d = ser.read(256)
    except Exception as e:
        print("  读异常：%s：%s" % (type(e).__name__, e))
        break
    if d:
        got += d
        print("  +%5.2fs 收到 %d 字节：%r" % (time.time() - t0, len(d), d[:80]))
try:
    ser.close()
except Exception:
    pass
if not got:
    print("  （2.5 秒一片安静）")
    print("  ⚠ 这不一定是坏事：固件可能只在**收到指令**时才回 DONE/READY（见固件那条 ✓）")
    print("     ⇒ 下一步要做的是\"**发一条**按键看有没有 DONE\" ✓ 但那会真的打到本机窗口 ✗"
          " —— 所以先让用户点头 / 先把焦点放好 ✓")
else:
    print("  合计 %d 字节 ✓ ⇒ 固件在说话（板子活着 ✓）" % len(got))
