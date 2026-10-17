"""快慢对照：同一段位移，**一次发** vs **拆成很多小条慢慢发**（一次性排查件）。

为什么要问（2026-10-07）：标定里 60/120/800 三档**线性度都对**（0.78 / 0.80 / 0.80 ✓），
但 **1600 以上直接量不到**（差图里只剩旧位一块 ⇒ 光标跑到画面外/被夹住 ✗），
而且「撞屏幕四角」时光标**在画面里看不见** ✗。这三件事指向同一个嫌疑：
**「提高指针精确度」（指针加速）没关干净** ✗ —— 加速是**按速度**生效的 ✓
（Windows 按每个 HID 报文的速度算 ✓），所以：
  · 一条 800 单位的指令 = 固件拆成 7 个 120 的报文（**快** ✗）；
  · 拆成 80 条 10 单位的指令（**慢** ✓）⇒ 加速不生效 ⇒ 位移应当**比快的那次小** ✓。
⇒ 两次位移明显不等 ⇒ 加速确实还在 ✗（那 `mouse_aim` 的绝对定位**必偏** ✓，必须先关掉 ✓）。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

from decision import input as dinput                                # noqa: E402
from tools.mouse_aim_calib import (_connect_input, _grab, _open_stream,   # noqa: E402
                                   park_below_band, probe_band_rect)
from tools.mouse_gain_calib import _cursor_delta                    # noqa: E402


def main():
    ok, why = _connect_input()
    print("[1] %s" % why)
    if not ok:
        return 2
    src, url = _open_stream()
    print("[2] 开流 %s" % url)
    band = [None]

    def grab():
        f = _grab(src)
        if f is None:
            return None
        if band[0] is None:
            band[0] = probe_band_rect(f.shape[1], f.shape[0]) or ()
        if band[0]:
            x, y, w, h = band[0]
            f[y:y + h, x:x + w] = 0
        return f

    def run(name, moves, gap):
        park_below_band()
        time.sleep(0.4)
        b = grab()
        for dx in moves:
            dinput.mouse_move(dx, 0)
            time.sleep(gap)
        time.sleep(0.6)
        a = grab()
        # 回程（逐条反向 ✓ 每步也小 ✓）
        for dx in moves:
            dinput.mouse_move(-dx, 0)
            time.sleep(gap)
        time.sleep(0.4)
        d = _cursor_delta(b, a, direction=1) if (b is not None and a is not None) else None
        total = sum(moves)
        print("    %-22s 共 %4d 单位（%d 条 / 间隔 %.3fs）⇒ 量到 %s"
              % (name, total, len(moves), gap, d))
        return (abs(d[0]) if d else None), total

    print("[3] A 慢发：80 条 × 10 单位（每条 10 counts ⇒ 每个报文都慢 ✓ 加速不该生效）")
    a_px, a_u = run("慢（80×10）", [10] * 80, 0.02)
    print("[4] B 快发：1 条 × 800 单位（固件拆成 7 个 120 报文 ⇒ 快 ✗ 加速若开着就会放大）")
    b_px, b_u = run("快（1×800）", [800], 0.02)

    if a_px is None or b_px is None:
        print("[X] 有一次量不到 ⇒ 结果不可比（先把现场帧存下来看 ✓）")
        return 2
    ra, rb = a_px / a_u, b_px / b_u
    print("[5] 慢发 gain %.4f px/单位 ｜ 快发 gain %.4f px/单位 ｜ 相差 %.1f%%"
          % (ra, rb, abs(ra - rb) / max(ra, rb) * 100.0))
    if abs(ra - rb) / max(ra, rb) > 0.10:
        print("    ⇒ **两次明显不等**：快的那次 %s ✗ ⇒ 「提高指针精确度」还在生效 ✗"
              % ("更远" if rb > ra else "更近"))
        print("      修：A 机 → 鼠标设置 → 指针选项 → **取消勾选「提高指针精确度」**，"
              "指针速度停在正中（6/11）✓ 然后重跑标定 ✓")
    else:
        print("    ⇒ 两次相当 ⇒ **加速已关** ✓（那 1600 量不到另有原因：光标离开了"
              "画面可见区 ✓ 见 `data/_probe_corner_dump` ✓）")

    # 收尾：光标放回画面左侧中部
    park_below_band()
    try:
        src.close()
    except Exception:                                  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
