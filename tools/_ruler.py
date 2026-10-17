# -*- coding: utf-8 -*-
"""临时：**用"看一眼指针落在哪"量 gain**，绕开帧差那套噪声（用完删）。

为什么要换法（2026-10-07 ✓ 帧差那套在这块屏上量不到）：
    游戏界面上有东西在动（负载条/动画 ✗）+ 探针带在画面顶部 ✗ ⇒ "恰好两个块"的判据随机抓到
    别的东西 ⇒ 每轮数字乱跳 ⇒ 工具只能老实报"量不到" ✓。
    ⭐ 但**撞角归零是好的**（实测：归零后指针就停在画面左上角 ✓ 见 `_look_3_top_strip.png` ✓）
    ⇒ 那就**不用帧差**：归零后发一条**已知大位移**、**把画面裁下来看指针落在哪** ✓
    ⇒ gain = 指针的新 x ÷ 发出的单位数 ✓✓（不受画面里别的东西动的影响 ✓）。

⛔ 只发 MOVE（不点任何键 ✓）。产出一张对比图（归零后 / 走了 X 单位后）⇒ 我读图中指针的 x ✓。
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

from tools.mouse_aim_calib import _connect_input, _grab, _open_stream

UNITS = 1000          # 已知大位移（>127 ⇒ 顺带验固件分块发送 ✓）
CROP = (0, 0, 900, 46)  # 画面顶部那一条（归零后指针就在这附近 ✓）


def main():
    ok, why = _connect_input()
    print("[1] %s" % why)
    if not ok:
        return 2
    src, url = _open_stream()
    print("[2] 已开流：%s" % url)
    from decision import input as dinput
    try:
        dinput.mouse_move(-4000, -4000)          # 撞角归零（已验证会落到 (0,0) ✓）
        time.sleep(0.8)
        a = _grab(src)
        dinput.mouse_move(UNITS, 0)              # 已知位移
        time.sleep(0.8)
        b = _grab(src)
    finally:
        try:
            src.close()
        except Exception:                        # noqa: BLE001
            pass
    if a is None or b is None:
        print("[X] 抓不到帧")
        return 2
    x, y, w, h = CROP
    ca, cb = a[y:y + h, x:x + w], b[y:y + h, x:x + w]
    big = np.vstack([ca, np.full((6, ca.shape[1], 3), 255, np.uint8), cb])
    big = cv2.resize(big, None, fx=1.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
    for i in range(0, w, 100):                   # 每 100 流像素画一条刻度（便于读 x ✓）
        cv2.line(big, (int(i * 1.5), 0), (int(i * 1.5), big.shape[0] - 1),
                 (0, 0, 255), 1)
        cv2.putText(big, str(i), (int(i * 1.5) + 2, big.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(big, "after corner-zero      |      after MOVE %d 0" % UNITS,
                (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(big, "after corner-zero      |      after MOVE %d 0" % UNITS,
                (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    out = ROOT / "data" / "_ruler.png"
    cv2.imwrite(str(out), big)
    print("[3] 上：归零后；下：发 MOVE %d 0 后 ⇒ 存 %s" % (UNITS, out))
    print("    读法：两张图里**白色指针箭头的 x 差** ÷ %d = gain（px/单位）✓" % UNITS)
    print("    ⚠ 刻度是流像素 ✓（每 100 一格）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
