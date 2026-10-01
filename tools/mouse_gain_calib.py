# -*- coding: utf-8 -*-
"""M2c 鼠标增益标定：发一条**已知位移**的鼠标指令 ⇒ 从画面里量光标实际走了多少 ⇒
存 `config/mouse_gain.json`（`{"gain_x": px/单位, "gain_y": ..., "probe": {...}}` ✓）。

为什么要标定：鼠标通道只有**相对移动**（`decision/input.py::mouse_move(dx,dy)` ✓），
控制律（`perception/lie_controller.py`）要把"目标误差（游戏像素）"换算成"发多少指令"
—— 换算比例就是增益。标定错得离谱（>2×）时闭环要靠反馈慢慢纠，误差大、命中率掉 ✗。

**怎么量**（不依赖游戏里画不画光标模板 ✓）：连拍两帧、中间发一条指令 ——
  · 前后帧差 ⇒ 只有光标动过（游戏画面若无其它运动 ✓）；
  · 两个差块（旧位/新位）的**质心位移** = 光标的实际位移（游戏像素 ✓）；
  · 增益 = 位移 / 指令单位。

用法：
    python -m tools.mouse_gain_calib                # 用 config/live.yaml 的窗口
    python -m tools.mouse_gain_calib --dx 60 --dy 0
    python -m tools.mouse_gain_calib --dry-run      # 只打印，不写文件

⚠ 需要：① ProMicro 后端在线（`dinput.mouse_available()` ✓，本地 SendInput 模式没有
硬件鼠标 ✗）；② 画面能抓（窗口模式 `core.wincap.grab_rect` ✓）。缺任何一个就明确报错
并给替代做法，**不许瞎写一个增益** ✗（错的标定比不标定更糟 —— 见 docs §M2c ✓）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decision import input as dinput  # noqa: E402

OUT = ROOT / "config" / "mouse_gain.json"


def _grab():
    """抓一帧窗口客户区画面（BGR ✓）—— 用 `config/live.yaml` 的 rect ✓。"""
    from core import config as cfg
    from core import wincap
    live = cfg.load_live()
    rect = live.get("rect")
    if not rect:
        raise RuntimeError("config/live.yaml 里没有 rect（窗口模式抓图要用它 ✓）")
    return wincap.grab_rect(tuple(rect))


def _cursor_delta(before, after, min_area=3):
    """两帧差 ⇒ 光标位移（游戏像素 `(dx, dy)` ✓，量不到 ⇒ `None`）。

    光标是**唯一在动**的东西 ⇒ 差图里恰好两个小连通域（旧位 + 新位）：
    取它们的质心之差 = 位移 ✓；少于两个域（糊在一起 / 没动 / 画面在动）⇒ None。
    """
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    _, bw = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    res = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = res[0] if isinstance(res, tuple) else res
    pts = []
    for c in cs:
        a = cv2.contourArea(np.asarray(c, np.float32).reshape(-1, 1, 2))
        if a < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            pts.append((m["m10"] / m["m00"], m["m01"] / m["m00"], a))
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: -p[2])
    a, b = pts[0], pts[1]
    return (b[0] - a[0], b[1] - a[1])


def measure(dx, dy, settle=0.12, retries=3):
    """发一条指令 ⇒ 量实际位移（多次取中位 ✓ 抗单次噪声）。"""
    vals = []
    for _ in range(retries):
        before = _grab()
        time.sleep(settle)
        dinput.mouse_move(dx, dy)
        time.sleep(settle)
        after = _grab()
        d = _cursor_delta(before, after)
        if d:
            vals.append(d)
        time.sleep(0.15)
    if not vals:
        return None
    xs = sorted(v[0] for v in vals)
    ys = sorted(v[1] for v in vals)
    return (xs[len(xs) // 2], ys[len(ys) // 2], len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=int, default=60)
    ap.add_argument("--dy", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not dinput.mouse_available():
        print("[X] 鼠标通道不可用：本地 SendInput 模式没有硬件鼠标 ✗")
        print("    先把「被控机部署台」的 ProMicro 后端连上（docs §M2c ✓），再跑本工具。")
        return 2
    try:
        d = measure(args.dx, args.dy)
    except Exception as e:                    # noqa: BLE001
        print("[X] 抓图失败：%s" % e)
        print("    窗口模式需要 config/live.yaml 的 rect 且窗口可见 ✓")
        return 2
    if d is None:
        print("[X] 量不到光标位移（画面里找不到两个差块 ✗）")
        print("    检查：目标区域有画面在动吗？光标显示了吗？settle 够吗？")
        return 2
    mx, my, n = d
    gx = mx / args.dx if args.dx else 0.0
    gy = my / args.dy if args.dy else 0.0
    print("指令 (%d, %d) ⇒ 实测位移 (%.1f, %.1f) px（%d 次取中位 ✓）" % (args.dx, args.dy, mx, my, n))
    print("增益：gx=%.4f  gy=%.4f px/单位" % (gx, gy))
    if gx <= 0 or gy <= 0:
        print("[X] 增益非正 ⇒ 方向对不上（指令正 dx 光标却往左？）—— 不写文件 ✗")
        return 2
    if args.dry_run:
        print("（--dry-run ⇒ 不写 %s）" % OUT)
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"gain_x": gx, "gain_y": gy,
                               "probe": {"dx": args.dx, "dy": args.dy,
                                         "measured": [mx, my], "n": n}},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写 %s ✓（控制律 `load_gain()` 会自动吃它 ✓）" % OUT)
    print("⚠ 换个「指针加速」状态（游戏内/桌面）要**重新标定** ✓ —— 标定只在同状态成立。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
