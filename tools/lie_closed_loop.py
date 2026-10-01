# -*- coding: utf-8 -*-
"""M2c 离线闭环仿真：把 M2a 的追踪输出当"目标流"，喂给 **M2c 控制律 + 模拟光标**，
比对「闭环命中率」与「离线命中率」——就是设计文档 M2c 判据的**离线预演** ✓。

设计文档 M2c 判据：*对回放画面接真鼠标跑，命中率与离线一致（±10%）*。
真鼠标要先有硬件与实机画面 ⇒ 这里用**模拟光标**（吃指令、带量化、带一拍延迟 ✓）
先把控制律这一段验掉：**感知不变**的前提下，闭环（控制+延迟）相对离线（理想
"上一拍输出即光标"）掉多少 ✓。

用法：
    python -m tools.lie_closed_loop --sample 40
    python -m tools.lie_closed_loop --all --gain-true 1.6      # 标定错 60% 的鲁棒性
    python -m tools.lie_closed_loop --all --no-feedback        # 反向验证：掐反馈该崩
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perception.lie_controller import LieMouseController  # noqa: E402
from perception.lie_tracker import LieTracker  # noqa: E402
from tools.lie_replay import _MIN_RADIUS, load_cycles  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _radius(area):
    return max(_MIN_RADIUS, (area or 0.0) ** 0.5 / 1.7725)


def closed_loop_cycle(frames, dets, gain_cfg, gain_true, feedback=True,
                      assume=(375.0, 250.0)):
    """跑一个 cycle ⇒ (离线命中率, 闭环命中率, 帧数)。

    **目标流 = 追踪器输出**（离线口径的唯一"真值源" ✓ 感知两端不变 ⇒ 差异只来自
    控制律 ✓）；光标 = 模拟（吃指令 × `gain_true`、整数量化、一拍延迟 ✓）。
    """
    tr = LieTracker()
    c = LieMouseController(gain=gain_cfg, assume=assume)
    cur = assume                                   # 模拟光标（初始 = 弹窗中心假定 ✓）
    prev_cmd_due = []
    off_hit = off_n = cl_hit = cl_n = 0
    prev_off = None                                # 离线合成鼠标（= 上一拍输出 ✓）
    prev_cur = None                                # 闭环合成鼠标（= 上一拍光标 ✓）
    t_prev = None
    for _idx, ts, path in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        d = None
        if dets is not None:
            raw = dets.get(path.stem)
            d = [b[1:] for b in raw] if raw else None
        out = tr.process(img, ts=ts, dets=d)
        # 指令延迟一拍生效（真实的传输+处理延迟 ✓ 也算控制律要扛的账 ✓）
        due, prev_cmd_due = prev_cmd_due, []
        for cmd in due:
            cur = (cur[0] + cmd[0] * gain_true[0], cur[1] + cmd[1] * gain_true[1])
        cur = (round(cur[0]), round(cur[1]))       # 游戏里光标就是整像素 ✓
        pos = out["pos"] if out["state"] == "track" else None
        r = _radius(out["area"])
        # 离线口径：上一拍追踪输出（就是 lie_replay 的算法 ✓）
        if pos is not None and prev_off is not None:
            off_n += 1
            if ((prev_off[0] - pos[0]) ** 2 + (prev_off[1] - pos[1]) ** 2) ** 0.5 <= r:
                off_hit += 1
        # 闭环口径：**本帧已知的光标**（= 上一拍指令落地后 ✓）
        # ⚠⚠ 别用"上一拍光标"（`prev_cur`）—— 那比离线口径**多滞后一拍** ✗，
        #   白送闭环 2 帧滞后 ⇒ 判据永远差 ~0.13（实测踩过 ✗ 2026-09-30 ✓）。
        if pos is not None and cur is not None:
            cl_n += 1
            if ((cur[0] - pos[0]) ** 2 + (cur[1] - pos[1]) ** 2) ** 0.5 <= r:
                cl_hit += 1
        prev_off = pos
        prev_cur = cur
        cmd = c.step(pos, ts=ts, cursor=cur if feedback else None)
        if cmd[0] or cmd[1]:
            prev_cmd_due.append(cmd)               # 下一拍生效 ✓
        t_prev = ts
    return (off_hit / off_n if off_n else 0.0,
            cl_hit / cl_n if cl_n else 0.0, off_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--cycles", type=str, default="")
    ap.add_argument("--dets", type=str, default="datasets/lie/dets_v2.json")
    ap.add_argument("--gain-true", type=float, default=1.0,
                    help="模拟世界的真增益（标定错的鲁棒性测试 ✓ 默认 1.0）")
    ap.add_argument("--no-feedback", action="store_true",
                    help="反向验证：不给光标反馈（纯航位推算 ⇒ 该崩 ✓）")
    args = ap.parse_args()

    cyc = load_cycles()
    dets = None
    if args.dets and Path(args.dets).is_file():
        dets = json.loads(Path(args.dets).read_text(encoding="utf-8"))
    if args.cycles:
        ids = [c.strip().zfill(5) for c in args.cycles.split(",") if c.strip()]
    elif args.all:
        ids = sorted(cyc)
    else:
        n = max(1, args.sample)
        ids = sorted(cyc)[::max(1, len(cyc) // n)][:n]

    gain_cfg = (1.0, 1.0)                          # 配置增益（标定文件缺省 1:1 ✓）
    rows = []
    for cid in ids:
        if cid not in cyc:
            continue
        off, cl, n = closed_loop_cycle(cyc[cid], dets, gain_cfg,
                                      (args.gain_true, args.gain_true),
                                      feedback=not args.no_feedback)
        if n:
            rows.append((cid, off, cl, n))
    if not rows:
        print("没有可用 cycle")
        return 1
    offs = [r[1] for r in rows]
    cls = [r[2] for r in rows]
    print("cycles %d ｜ 帧 %d" % (len(rows), sum(r[3] for r in rows)))
    print("离线命中（上一拍追踪输出）: mean=%.3f med=%.3f"
          % (statistics.mean(offs), statistics.median(offs)))
    print("闭环命中（本帧光标 ✓）: mean=%.3f med=%.3f"
          % (statistics.mean(cls), statistics.median(cls)))
    print("闭环 − 离线: mean %+.3f ｜ med %+.3f（判据 ±0.10 ✓）"
          % (statistics.mean(cls) - statistics.mean(offs),
             statistics.median(cls) - statistics.median(offs)))
    worst = sorted(rows, key=lambda r: r[2] - r[1])[:8]
    for cid, off, cl, n in worst:
        print("  掉最多 %s 离线 %.2f → 闭环 %.2f（帧 %d）" % (cid, off, cl, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
