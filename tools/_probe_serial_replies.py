"""验"工作台侧收不全回执"这个假设（2026-10-09 ✓）。

## 现在已知（现场取到的三条硬证据 ✓）

刚跑的 `data/_probe_com8_talk.py`（**板子刚让出来时**）：

| 探针做的事 | 结果 | 结论 |
|---|---|---|
| 打开 COM8 后只读 3.5 秒 | **没有** READY | 打开口**不复位**板子 ✓（"自锁复位环"那个假设**不成立** ✗） |
| 发 `PING`（固件不认识 ✓） | 回 `ERR` ✓ | **固件是这一版、而且在跑** ✓ |
| 发 `MOVE 0 0`（位移为零 ✓） | 回 `DONE` ✓ | **双向通**（发得出去、回执收得到 ✓） |

⇒ 板子、固件、串口、双向通道**全都没问题** ✓ ⇒ 问题在**工作台那一侧** ✗
（而面板红字「指令发不出去」= `link_health()` 假 ✓，`perf.log` 里 `send_fail` 为 0 ✓
 ⇒ 只能是「发出去了、等不到回执」那一路 ✓）。

## 这个脚本验的假设

`SerialKbd._drain` 收回执是这么数的：

```python
data = self._ser.read(256)
n = data.count(b"DONE") + data.count(b"ERR")
```

⚠⚠ `read(256)` **不保证按行返回** ✗ —— USB CDC 一次给你 `b"DO"`、下一次给 `b"NE\r\n"` 是常事 ✓。
那样 `count(b"DONE")` 两次都数不到 ✗ ⇒ `_pending` **永远减不下去** ✗ ⇒
`silent_for()` 恒 ≥5 秒 ⇒ `link_health()` **永久为假** ✗ ⇒ 面板一直红着写「指令发不出去」、
每 5 秒自动重连一次 …… **而板子其实一直好好地在回 DONE** ✓✓
（这正是"串口连着、板子也没坏、就是发不出去"的完美解释 ✓）。

⇒ 本脚本按工作台**一模一样**的方式发一串指令（`MOVE 0 0` ✓ 位移为零 ⇒ 光标不动 ✓
不会敲任何键 ✓），然后对账：`sent` / `replies` / 还没对上的 `_pending` / `silent_for()`。
**若 replies < sent 或 silent ≥5 秒 ⇒ 假设成立** ✓ ⇒ 修 `_drain`（按行缓冲 ✓），
顺带 `kbd_client`（TLS 那条）也照抄同一份 ✓。
"""

import sys
import time

sys.path.insert(0, ".")
from remote_kbd.serial_kbd import SerialKbd  # noqa: E402

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM8"
print("=== 按工作台的方式发指令，看回执能不能对上（%s ✓）===" % PORT)
try:
    k = SerialKbd(PORT)
except Exception as e:
    print("  ✗ 打不开：%s: %s" % (type(e).__name__, e))
    print("  （口被占着？先把面板的输入设备切到「本地键盘」✓）")
    sys.exit(1)
print("  已打开 ✓（不等 READY，跟 `use_serial` 一样 ✓）")
time.sleep(0.3)

k.send("RELEASEALL")          # 无害：松开所有键 ✓
time.sleep(0.3)
base_sent, base_replies = k.sent, k.replies
print("  暖场后：sent=%d replies=%d（差值 %d）" % (base_sent, base_replies,
                                                base_sent - base_replies))

for phase, gap in (("慢一点（50ms 一条）", 0.05), ("连着发（不等 ✓）", 0.0)):
    s0, r0 = k.sent, k.replies
    for _ in range(20):
        k.send("MOVE 0 0")     # 位移为零 ⇒ 光标一动不动 ✓ 只为了让固件回一条 DONE ✓
        if gap:
            time.sleep(gap)
    time.sleep(1.5)
    d_sent, d_rep = k.sent - s0, k.replies - r0
    print("  %-20s 发了 %d 条 ⇒ 对上 %d 条%s"
          % (phase, d_sent, d_rep, "" if d_rep >= d_sent else "   ← ⚠ 少了对不上 ✗"))

print()
print("=== 总账 ===")
print("  sent=%d  replies=%d  对不上=%d" % (k.sent, k.replies, k._pending))
print("  silent_for() = %.2f 秒   （≥5 就会被判「链路卡死」✗）" % k.silent_for())
print("  ok=%s  fails=%d  last_err=%r" % (k.ok, k.fails, k.last_err))

try:
    from execution import input as dinput
    from execution import remote as _remote_mod
    _remote_mod._remote, _remote_mod._cfg = k, ("serial", PORT)
    print("  link_health() =", _remote_mod.link_health())
except Exception as e:
    print("  （link_health 没跑成：%s: %s）" % (type(e).__name__, e))

if k._pending > 0 or k.replies < k.sent:
    print()
    print("  ⭐⭐ **假设成立**：板子在回 DONE（上面探针验过 ✓），但**工作台数不上**✗")
    print("     ⇒ 根因 = `SerialKbd._drain` 按字节数 `DONE` ✗ —— `read(256)` 会把一条回执"
          "**切成两半** ⇒ 数不到 ⇒ `_pending` 减不下去 ⇒ `silent_for()` 恒 ≥5 秒"
          " ⇒ `link_health()` 永久为假 ⇒ 面板红着「指令发不出去」+ 每 5 秒自动重连 ✓")
    print("     ⇒ 修法：`_drain` 改成**按行缓冲**（跨读取拼行 ✓），`kbd_client` 那条同源一份 ✓")
else:
    print()
    print("  ⇒ 假设**不成立**（回执全对上了 ✓）⇒ 再往上查工作台那侧（连接时序 / 重连 ✓）")

try:
    k.close()
except Exception:
    pass
print("=== 收工（口已释放 ✓）===")
