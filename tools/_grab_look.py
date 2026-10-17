# -*- coding: utf-8 -*-
"""临时：抓帧存图，直接看"画面里到底有没有鼠标指针"（用完删）。

判据不是"帧差有没有块"（那只能说明'有没有东西动' ✗），而是**人眼/图上看有没有指针**：
    撞角归零后一定有指针停在画面左上角 ✓ ⇒ 抓一张、放大左上角看 ✓；
    再发一条 MOVE 600 ⇒ 抓第二张 + 放大的差分图 ⇒ 指针去哪了一目了然 ✓。
⛔ 只发 MOVE（不点任何键 ✓）。
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

from tools.mouse_aim_calib import (_connect_input, _grab, _open_stream,
                                  probe_band_rect)

OUT = ROOT / "data"


def main():
    ok, why = _connect_input()
    print("[1] %s" % why)
    if not ok:
        return 2
    src, url = _open_stream()
    print("[2] 已开流：%s" % url)
    from decision import input as dinput
    try:
        from decision import mouse_aim
        print("[3] load_gain() =", mouse_aim.load_gain())
        dinput.mouse_move(-4000, -4000)
        import time as _t
        _t.sleep(0.6)
        f1 = _grab(src)
        dinput.mouse_move(600, 0)
        _t.sleep(0.6)
        f2 = _grab(src)
    finally:
        try:
            src.close()
        except Exception:                                  # noqa: BLE001
            pass
    if f1 is None or f2 is None:
        print("[X] 抓不到帧")
        return 2
    band = probe_band_rect(f1.shape[1], f1.shape[0])
    a, b = f1.copy(), f2.copy()
    if band:
        x, y, w, h = band
        for im in (a, b):
            im[y:y + h, x:x + w] = 0
    # 整帧（缩小便于人看）+ 左上角放大 4 倍（指针就在那附近 ✓）
    cv2.imwrite(str(OUT / "_look_1_corner.png"), cv2.resize(a, (960, 540)))
    cv2.imwrite(str(OUT / "_look_2_moved.png"), cv2.resize(b, (960, 540)))
    crop1 = a[0:80, 0:600]
    crop2 = b[0:80, 0:600]
    big = np.vstack([cv2.resize(crop1, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST),
                     cv2.resize(crop2, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)])
    cv2.imwrite(str(OUT / "_look_3_top_strip.png"), big)
    d = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    print("[4] 整帧差分：最大 %.0f，>25 的像素 %d 个（%s）"
          % (float(d.max()), int((d > 25).sum()),
             "有东西动了 ✓" if (d > 25).sum() else "**整帧没有一处变化** ✗"))
    # ⭐ 光看"有没有动"不够 —— 要**数字**：动了多少、在哪（= 指针挪了多远 ✓）
    from tools._diag_aim_cursor import blobs, desc
    bs = blobs(a, b)
    print("[4b] 变化块：%s" % desc(bs))
    if len(bs) >= 2:
        (_a1, x1, y1), (_a2, x2, y2) = bs[0], bs[1]
        print("     最大两块：A(%.0f,%.0f) B(%.0f,%.0f) ⇒ dx=%.1f dy=%.1f"
              % (x1, y1, x2, y2, x2 - x1, y2 - y1))
        print("     （这次发的是 MOVE 600 ⇒ 若 dx ≈ 600 × gain，把 gain 反推出来 ✓"
              " 期望量级 0.1~0.6 ⇒ dx 60~360 流像素 ✓）")
    print("     ℹ 上面这些块都在**探针带外**（带已遮 %s ✓）" % (band,))
    vis = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=3.0), cv2.COLORMAP_JET)
    cv2.imwrite(str(OUT / "_look_4_diff.png"), cv2.resize(vis, (960, 540)))
    print("[5] 存了 4 张：_look_1_corner / _look_2_moved / _look_3_top_strip / _look_4_diff")
    print("    画面尺寸 %dx%d" % (f1.shape[1], f1.shape[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
