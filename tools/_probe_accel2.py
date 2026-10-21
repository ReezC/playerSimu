"""慢发对照的**自证**：换个"不重复"的小步序列再量一次（一次性排查件）。

背景（2026-10-07）：工具内置的慢发对照（80 条**一模一样**的 `MOVE 10`）量出 0.47，
而一条 `MOVE 800` 量出 0.79 ⇒ 判"加速还在" ✗ —— 但用户已确认取消勾选 ✓。
⇒ 先怀疑**测试本身** ✗：中继 / 固件 / HID 层若把**重复的相同报文**去重或合并 ✗，
80 条 `MOVE 10` 就不会全部落地 ⇒ "慢发"量出来的位移自然偏小 ✓（与加速无关 ✗）。

三种发法，同一个总位移 800 单位：
  A 重复小步：80 × `MOVE 10`        （现在的慢发对照 ✓）
  B 交替小步：40 × (`MOVE 9`,`MOVE 11`)（**相邻不相等** ⇒ 去重就吃不掉它 ✓）
  C 一条大指令：1 × `MOVE 800`       （现用"快"档 ✓）
判读：
  B ≈ C ⇒ **加速其实已经关了** ✓，是 A 这条测试被去重坑了 ✗ ⇒ 工具里改用 B ✓
  B ≈ A（都远小于 C）⇒ 加速确实还在 ✗ ⇒ 继续查设置 ✓
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

from execution import input as dinput                                       # noqa: E402
from tools.mouse_aim_calib import (_connect_input, _grab, _open_stream,     # noqa: E402
                                   park_below_band, probe_band_rect)
from tools.mouse_gain_calib import _cursor_delta                           # noqa: E402

TOTAL = 800


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

    def run(name, steps, gap):
        park_below_band()
        time.sleep(0.4)
        b = grab()
        for s in steps:
            dinput.mouse_move(s, 0)
            time.sleep(gap)
        time.sleep(0.6)
        a = grab()
        for s in steps:                       # 回程：逐条反向（同样慢 ✓）
            dinput.mouse_move(-s, 0)
            time.sleep(gap)
        time.sleep(0.4)
        d = _cursor_delta(b, a, direction=1) if (b is not None and a is not None) else None
        tot = sum(steps)
        px = abs(d[0]) if d else None
        g = (px / tot) if px else 0.0
        print("    %-16s %4d 单位 / %3d 条 ⇒ %s ⇒ gain %.4f"
              % (name, tot, len(steps), ("%.1f px" % px) if px else "量不到 ✗", g))
        return g

    a = run("A 重复小步", [10] * 80, 0.02)
    b = run("B 交替小步", [9, 11] * 40, 0.02)
    c = run("C 一条大指令", [800], 0.02)
    print("[3] A(重复小步)=%.4f  B(交替小步)=%.4f  C(一条大)=%.4f" % (a, b, c))
    if c <= 0 or b <= 0:
        print("    ⇒ 有量不到的 ⇒ 先看图（--dump / 现场帧）再说 ✓")
    elif abs(b - c) / max(b, c) <= 0.15:
        print("    ⇒ **B 与 C 相当** ⇒ 加速其实**已经关了** ✓ —— 是 A 那条（重复报文）"
              "被去重坑了 ✗ ⇒ 工具里的慢发对照改用**交替小步** ✓")
    else:
        print("    ⇒ **B 也远小于 C** ⇒ 加速**确实还在** ✗ ⇒ 继续查设置 ✓"
              "（另一个常见来源：鼠标厂商驱动里的\"指针加速/游戏模式\" ✗ 会盖掉系统设置 ✓）")
    park_below_band()
    try:
        src.close()
    except Exception:                          # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
