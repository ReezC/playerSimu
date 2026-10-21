# -*- coding: utf-8 -*-
"""临时：**判定板子上的固件是不是老版本**（没用完删）。

判据（一眼定案 ✓）：
    先往右挪开一点脱离屏幕边缘 ⇒ 抓帧 A；
    发 `MOVE 200 0` ⇒ 抓帧 B ——
      · **新固件（会分块）** ⇒ 指针**再往右**走约 200×gain ✓
      · **老固件（signed char 截断）** ⇒ 200 被截成 **−56** ⇒ 指针**往左**走 ✗
    再发 `MOVE 400 0` ⇒ 抓帧 C（400 → 截断成 −112 ⇒ 更明显往左 ✓）
⛔ 只发 MOVE（不点任何键 ✓）。产出三行对比图（带 100px 刻度 ✓）⇒ 读指针的 x 差 ✓。
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

AWAY = 4          # 先发几条 120 脱离左边缘
AWAY_STEP = 120
CROP = (0, 0, 700, 40)


def main():
    ok, why = _connect_input()
    print("[1] %s" % why)
    if not ok:
        return 2
    src, url = _open_stream()
    print("[2] 已开流：%s" % url)
    from execution import input as dinput
    frames = []
    try:
        dinput.mouse_move(-4000, -4000)          # 撞角（老固件会被截断 ⇒ 不指望它 ✓）
        time.sleep(0.5)
        for _ in range(AWAY):                    # 先往右挪开（每条都 ≤127 ⇒ 老固件也走对 ✓）
            dinput.mouse_move(AWAY_STEP, 0)
            time.sleep(0.08)
        time.sleep(0.6)
        frames.append(("A: away from edge", _grab(src)))
        dinput.mouse_move(200, 0)                # ← 关键一条（>127 ⇒ 老固件截成 -56 ✓）
        time.sleep(0.7)
        frames.append(("B: after MOVE 200", _grab(src)))
        dinput.mouse_move(400, 0)
        time.sleep(0.7)
        frames.append(("C: after MOVE 400", _grab(src)))
    finally:
        try:
            src.close()
        except Exception:                        # noqa: BLE001
            pass
    if any(f is None for _t, f in frames):
        print("[X] 有帧没抓到")
        return 2
    x, y, w, h = CROP
    rows = []
    for tag, im in frames:
        c = im[y:y + h, x:x + w].copy()
        cv2.putText(c, tag, (255, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(c, tag, (255, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(c)
        rows.append(np.full((4, w, 3), 255, np.uint8))
    big = np.vstack(rows[:-1])
    big = cv2.resize(big, None, fx=1.6, fy=2.6, interpolation=cv2.INTER_NEAREST)
    for i in range(0, w, 100):
        cv2.line(big, (int(i * 1.6), 0), (int(i * 1.6), big.shape[0] - 1),
                 (0, 0, 255), 1)
        cv2.putText(big, str(i), (int(i * 1.6) + 2, big.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    out = ROOT / "data" / "_ruler2.png"
    cv2.imwrite(str(out), big)
    print("[3] A/B/C 三行对比 ⇒ %s" % out)
    print("    读法：看白色指针箭头的 x 在 A→B、B→C 是**变大**（新固件 ✓）还是**变小**"
          "（老固件截断 ✗）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
