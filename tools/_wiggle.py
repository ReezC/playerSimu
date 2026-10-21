# -*- coding: utf-8 -*-
"""临时：让指针**来回扫 30 秒**，人盯 A 机屏幕看光标动不动（用完删）。

要区分的两件事（2026-10-07 ✓ 已经排掉其它所有可能）：
    · 光标**动** ⇒ ProMicro 那条鼠标线是好的 ⇒ 问题在**推流没画指针**（改 push 的 draw_mouse ✓）
    · 光标**不动** ⇒ ProMicro 的鼠标接口没工作（键盘那条是好的 ✓）⇒ 重烧固件 ✓
⛔ 只发 MOVE（不点任何键 ✓ 不碰键盘 ✓）。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

from tools.mouse_aim_calib import _connect_input

SECONDS = 240       # ⭐ 拉长：用户常常没赶上第一轮 ⇒ 让它在后台一直扫，随时抬头都能看到 ✓
STEP = 120          # ≤127 ⇒ 老固件也走得对 ✓（每条单独走，不依赖分块）


def main():
    ok, why = _connect_input()
    print("[1] %s" % why)
    if not ok:
        return 2
    from execution import input as dinput
    print("[2] 开始来回扫 %d 秒 —— **请看 A 机的屏幕**：光标动不动？" % SECONDS)
    t0 = time.time()
    n = 0
    while time.time() - t0 < SECONDS:
        for _ in range(6):                       # 往右扫（6×120 ≈ 一段 ✓）
            dinput.mouse_move(STEP, 0)
            time.sleep(0.05)
        time.sleep(0.6)
        for _ in range(6):                       # 再扫回来
            dinput.mouse_move(-STEP, 0)
            time.sleep(0.05)
        time.sleep(0.6)
        n += 1
        print("   第 %d 轮（%.0fs）" % (n, time.time() - t0))
    print("[3] 结束 ✓ 光标动过吗？（A 机屏幕上看到的）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
