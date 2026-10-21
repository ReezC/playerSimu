"""斜走 vs 分轴走：真点击要斜着走（dx、dy 一起发 ✓），这条准不准（一次性排查件）。

背景（2026-10-07）：注册表 `MouseSpeed/Threshold1/2` 全 0、`MouseSensitivity=10` ⇒
**Windows 加速是关的** ✓；但同 800 单位"快发 vs 慢发"仍差 40% ✗ ⇒ 不是 Windows 的锅
（很可能是**游戏自己画的光标**走它自己的曲线 ✓ —— 这也解释了"撞屏幕四角看不见光标" ✓）。

本探针只问一件要紧事：**真点击那条路（一条指令、dx+dy 一起）准不准** ✓。
三种发法都走到同一个相对位置 (400, 400)：
  A 一条斜的：`MOVE 400 400`                       （= 真点击用的那条 ✓）
  B 分两条走：`MOVE 400 0` + `MOVE 0 400`
  C 慢斜走  ：40 × (`MOVE 10 10`)
各量"落点"（用差图里两块质心，不设方向先验 ✓），看三者是否落到同一处 ✓。
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


def blobs(before, after, thr=60.0):
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(cv2.absdiff(g1, g2), thr, 255, cv2.THRESH_BINARY)[1]
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    res = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = res[0] if isinstance(res, tuple) else res
    out = []
    for c in cs:
        a = cv2.contourArea(np.asarray(c, np.float32).reshape(-1, 1, 2))
        if a < 5:
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            out.append((a, m["m10"] / m["m00"], m["m01"] / m["m00"]))
    out.sort(key=lambda p: -p[0])
    return out


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
        time.sleep(0.5)
        b = grab()
        for dx, dy in steps:
            dinput.mouse_move(dx, dy)
            time.sleep(gap)
        time.sleep(1.2)
        a = grab()
        for dx, dy in steps:
            dinput.mouse_move(-dx, -dy)
            time.sleep(gap)
        time.sleep(0.5)
        if b is None or a is None:
            print("    %-14s 抓不到帧" % name)
            return
        bs = blobs(b, a)
        # 只留"人差不多的两块"（光标旧位 / 新位 ✓）：面积相近、且离得开 ✓
        print("    %-14s 前几块：%s" % (name, "  ".join("[a=%.0f(%.0f,%.0f)]" % t for t in bs[:4])))

    print("[3] 三种发法各走 (400, 400)（从撞角归零后的落点出发 ✓）")
    run("A 一条斜走", [(400, 400)], 0.02)
    run("B 分轴两条", [(400, 0), (0, 400)], 0.02)
    run("C 慢斜走", [(10, 10)] * 40, 0.02)
    park_below_band()
    try:
        src.close()
    except Exception:                                  # noqa: BLE001
        pass
    print("[4] 若三者落点相近 ⇒ 斜走没问题 ✓（真点击可以一条指令发）；"
          "差得多 ⇒ 得改成分轴 / 小步 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
