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


def _cursor_delta(before, after, min_area=3, thr=60.0, max_area=2500.0,
                  y_tol=6.0, area_ratio=5.0, min_dx=3.0,
                  direction=0, anchor=None, anchor_tol=50.0, left_edge=60.0):
    """两帧差 ⇒ 光标位移（游戏像素 `(dx, dy)` ✓，量不到 ⇒ `None`）。

    光标是"**唯一会沿指令方向成对出现**"的东西 ⇒ 差图里两个小连通域（旧位 + 新位）：

    ⚠⚠⚠ **v2（2026-10-07 现场重写 ✗ 旧版在这儿栽了整整一晚）**：旧版只做
    「按面积排序、取**最大的两块**」✗ —— 真实推流画面里**满是低幅噪声**（H.264 压缩
    块效应 + 界面自己动的动画 ✓），阈值 25 下差图能有 **62 个团** ✗，前两名是
    `148x8 的横条`之类 ✗ ⇒ 量出来的数乱跳 ⇒ 工具报"量不到位移"⇒ 人只能瞎猜
    （"指针速度不对""推流不画指针""固件坏了" ✗ 全是冤枉路 ✓）。
    现场帧实测（`data/_probe_blob_stats.py` ✓ 400 单位 → 真值 322px）：
      阈值 25：前两名 = 噪声 ✗；阈值 50~90：前两名**恰好**是
      `(8,394) 15x18`（旧位）+ `(331,395) 17x21`（新位）✓✓ 且 **y 完全相同** ✓。
    ⇒ v2 三条一起上：
      ① **抬高阈值**（25 ✗ → 60 ✓）：光标边缘是**高对比**、噪声是低幅 ✓；
      ② **只用配得上对的那两块**（不是"最大的两块" ✗）：面积相当（≤`area_ratio` 倍 ✓
         —— ⚠ 这条**别收紧** ✗：2026-10-07 实测**同一支光标**旧位 `a=102` / 新位 `a=34`
         就差 **3.0 倍** ✓（新位落在对比度低的背景上 ⇒ 白箭头只剩黑描边 ⇒ 面积小 ✓）⇒
         阈值卡 3.0 会把**真位移**误拒 ⇒ 有些轮次"量不到" ✗（`u120_r01` 现场 ✓）；
         放宽到 5.0 就稳 ✓ —— 敢放宽是因为下面还有"同水平线 + 旧位位置"两条**硬先验** ✓）、
         **同一条水平线上**（`|dy| ≤ y_tol` ✓ —— 我们的指令只沿一个轴走 ✓）、
         有实际位移（`|dx| ≥ min_dx` ✓）、尺寸是**光标量级**（`max_area` ✓ 挡掉大片动画 ✓）；
      ③ 合规的对**可能不止一对**（画面里恰好还有别的成对移动 ✗）⇒ 取**两团都更实**的那对
         （面积和最大 ✓）—— 实测干净帧里就一对 ✓，脏帧里也仍能挑对 ✓。

    ⚠ 本函数只**保证位移的大小**：`dx` 的正负与"哪个是旧位"无关（抗锯齿 / 贴边裁切会
    让两块面积随机大小 ✓）⇒ 调用方要用 `abs`、或**用自己发出去的指令符号**定方向 ✓
    （现场踩过：+60 量回 -17.0px ⇒ 报"增益非正 ⇒ 方向对不上" ⇒ 又一轮冤枉路 ✓）。
    ⚠ 画面里**每帧都在变**的已知叠层（时间码探针 ✓）要先遮掉 ⇒ 见
    `mouse_aim_calib.probe_band_rect` ✓（它同样每帧都变，但**不满足"同一条水平线"** ✓
    ⇒ 只影响速度、不再骗结果 ✓）。
    """
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    _, bw = cv2.threshold(diff, float(thr), 255, cv2.THRESH_BINARY)
    # 去毛刺：压缩噪声常是一两个像素的碎点（开运算 2x2 ✓ 便宜且够）
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    res = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = res[0] if isinstance(res, tuple) else res
    pts = []
    for c in cs:
        a = cv2.contourArea(np.asarray(c, np.float32).reshape(-1, 1, 2))
        if a < float(min_area) or a > float(max_area):
            continue                        # 太小 = 噪声碎点；太大 = 一片动画 ✗
        m = cv2.moments(c)
        if m["m00"] > 0:
            pts.append((m["m10"] / m["m00"], m["m01"] / m["m00"], a))
    if len(pts) < 2:
        return None
    # 选哪一对（三档严格度，按"我们知不知道自己发了什么"来 ✓）：
    #   · `direction=0`（不知道方向 ⇒ 老工具那条路 ✓）：只按"两团都最实"挑 ✓（旧行为 ✓）；
    #   · `direction=±1`（**标定这条路**：明明只往一个方向发 ✓）⇒ 再加两条**物理先验**：
    #       ① 两端必在**同一水平线**（横向指令 ⇒ 实测 |dy| ≤ 1 ✓）；
    #       ② 旧位那一头：给了 `anchor`（上一轮认出来的旧位 ✓ 每轮都会走回来 ⇒ 基本不动 ✓）
    #          就认它 ±`anchor_tol`；没给 ⇒ 认**左边缘**（`left_edge` 内 ✓ —— 标定每轮
    #          都先撞角归零 ✓）。⚠ **这条是 2026-10-07 大位移那档换来的** ✓：
    #          `u800_r00` 里界面动画也凑出一对「同水平线、面积相当」
    #          `(563,606)&(371,604)` ✗ ⇒ 光看面积会挑到它 ⇒ 两轮对不上 ⇒ 判"量不到" ✗；
    #          而光标那对里**必有一头在左边缘** `(8,394)` ✓ ⇒ 加上这条就唯一了 ✓。
    best, best_score = None, -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1, a1 = pts[i]
            x2, y2, a2 = pts[j]
            if max(a1, a2) / max(1.0, min(a1, a2)) > float(area_ratio):
                continue                    # 一大一小 ⇒ 多半一动一静（不是同一支光标）✗
            if abs(x1 - x2) < float(min_dx):
                continue                    # 没动 / 糊在一起 ✗
            if direction:
                if abs(y1 - y2) > float(y_tol):
                    continue                # 横向指令 ⇒ 两端必在同一行 ✓
                # 谁在"指令方向的反面"那一侧，谁就是**旧位**（我们往 +x 发 ⇒ 旧位在左 ✓）
                if (x1 - x2) * direction < 0:
                    old, new = (x1, y1, a1), (x2, y2, a2)
                else:
                    old, new = (x2, y2, a2), (x1, y1, a1)
                if (new[0] - old[0]) * direction <= 0:
                    continue                # 位移方向与发出去的指令相反 ⇒ 不是它 ✗
                if anchor is not None:
                    if (abs(old[0] - anchor[0]) > float(anchor_tol)
                            or abs(old[1] - anchor[1]) > float(anchor_tol)):
                        continue            # 旧位不在上一轮那个落点附近 ⇒ 不是它 ✗
                elif old[0] > float(left_edge):
                    continue                # 没锚点 ⇒ 只认左边缘那一头（撞角归零 ✓）✗
                score = a1 + a2
                cand = (new[0] - old[0], new[1] - old[1], (old[0], old[1]))
            else:
                score = a1 + a2             # 不知道方向 ⇒ 旧的"最实的一对"行为 ✓
                cand = (x2 - x1, y2 - y1, None)
            if score > best_score:
                best_score, best = score, cand
    return best


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
