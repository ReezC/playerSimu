"""量**纵向**增益：标定一直只量横向、纵向是"假设一样"（一次性件，用完删）。

为什么现在要量（2026-10-07 ✓ 用户回报）：真点"蓝蜗牛"那次落点**偏下了一点** ✗ ——
而 `mouse_aim` 的 `gain_y` 是**照搬 gain_x** 的（`mouse_aim_calib` 明确写着"只沿 x 量" ✓）。
若真实纵增益 ≠ 横增益 ⇒ 纵向就会**走过头**（偏下）或走不到 ✓。

量法：撞角归零停靠好 ⇒ 连拍两帧、中间发一条**纯纵向**指令 ⇒ 差图里那对小块（同一列上下 ✓）
的纵向距离 / 指令数 = 纵增益 ✓；再发一条**横向**的做对照 ✓。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import cv2                                            # noqa: E402
import numpy as np                                    # noqa: E402

from execution import input as dinput                  # noqa: E402
from tools.mouse_aim_calib import (_connect_input, _grab, _open_stream,     # noqa: E402
                                   park_below_band, probe_band_rect)

STEP = 150


def pairs(before, after, thr=60.0):
    """差图里所有"面积相当"的小块（不设方向先验 ✓，横竖都能认 ✓）。"""
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(cv2.absdiff(g1, g2), thr, 255, cv2.THRESH_BINARY)[1]
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    res = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = res[0] if isinstance(res, tuple) else res
    pts = []
    for c in cs:
        a = cv2.contourArea(np.asarray(c, np.float32).reshape(-1, 1, 2))
        if a < 8 or a > 2500:
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            pts.append((m["m10"] / m["m00"], m["m01"] / m["m00"], a))
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1, a1 = pts[i]
            x2, y2, a2 = pts[j]
            if max(a1, a2) / max(1.0, min(a1, a2)) > 5.0:
                continue
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if max(dx, dy) < 5:
                continue
            score = a1 + a2
            if best is None or score > best[0]:
                best = (score, dx, dy, x1, y1, x2, y2)
    return best


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

    def run(name, dx, dy):
        park_below_band()
        time.sleep(0.5)
        b = grab()
        dinput.mouse_move(dx, dy)
        time.sleep(0.6)
        a = grab()
        dinput.mouse_move(-dx, -dy)
        time.sleep(0.4)
        if b is None or a is None:
            print("    %-10s 抓不到帧" % name)
            return
        r = pairs(b, a)
        if not r:
            print("    %-10s 量不到（画面里没有成对小块）✗ 现场：请人眼看一下光标在不在画面里 ✓"
                  % name)
            return
        _, pdx, pdy, x1, y1, x2, y2 = r
        print("    %-10s 发 (%4d,%4d) ⇒ 实走 (%.1f, %.1f) px ⇒ 横 %.4f / 纵 %.4f px/单位"
              "  [块 (%.0f,%.0f)→(%.0f,%.0f)]"
              % (name, dx, dy, pdx, pdy, pdx / abs(dx or 1), pdy / abs(dy or 1),
                 x1, y1, x2, y2))

    print("[3] 两次纯轴移动（各 %d 单位 ✓ 中间撞角归零 ✓）" % STEP)
    run("横向", STEP, 0)
    run("纵向", 0, STEP)
    print("[4] 若横纵两个数**明显不等** ⇒ `gain_y` 不能照搬 `gain_x` ✗"
          "（那正是「偏下」的来源 ✓）⇒ 工具要改成两轴各量 ✓")
    park_below_band()
    try:
        src.close()
    except Exception:                                  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
