"""测谎追踪器的**新判据**（用户 2026-09-30 认可 ✓ 换掉"自洽命中率" ✗）。

两条判据：
 ① **位置误差（需要真值）**：在**合成台**（`tools/synth_lie.py` ✓ 用户 7 条事实同构 ✓）上
    量 `|报告位置 − 真值|` 的 中位/p90/最大 ✓ + **未建档率**（`state != track`，即 `init` ✓
    丢失不再是独立状态，已并进预测态 ✓）。
 ② **物理判据（真素材可用、不需要标注）**：在报告位置处量**局部运动**（小窗口模板匹配 ✓）
    与群体平移 `T` 的差 ⇒ `局部偏离`。真目标是全场**唯一不随群体走**的东西 ✓（事实 2/3）
    ⇒ 追对了 `局部偏离` 应该**显著 > 0** ✓；追到群体/背景上则 ≈ 0 ✗。

用法：
    python -m tools.eval_lie --synth 3            # 合成台（有真值 ✓）
    python -m tools.eval_lie --real 5             # 真素材（物理判据 ✓）
    python -m tools.eval_lie --synth 3 --no-dets  # 不给检测框（纯残差路线 ✓ 对照）
"""
from __future__ import annotations

import argparse
import math
import pathlib
import statistics as st
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perception.lie_group import GroupMotion  # noqa: E402
from perception.lie_tracker import LieTracker  # noqa: E402
from tools.synth_lie import make_case  # noqa: E402


def local_dev(prev_g, cur_g, pos, t, half=40, search=9):
    """报告位置处的**局部运动 − T**（px ✓ 物理判据）。取不到 ⇒ None。"""
    x0 = int(round(pos[0] - half))
    y0 = int(round(pos[1] - half))
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    if x0 < 0 or y0 < 0 or x1 > prev_g.shape[1] or y1 > prev_g.shape[0]:
        return None
    tpl = prev_g[y0:y1, x0:x1]
    ex, ey = x0 + float(t[0]), y0 + float(t[1])
    sx0 = int(max(0, round(ex - search)))
    sy0 = int(max(0, round(ey - search)))
    sx1 = int(min(cur_g.shape[1], round(ex + tpl.shape[1] + search)))
    sy1 = int(min(cur_g.shape[0], round(ey + tpl.shape[0] + search)))
    if sx1 - sx0 < tpl.shape[1] or sy1 - sy0 < tpl.shape[0]:
        return None
    res = cv2.matchTemplate(cur_g[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    dx = sx0 + loc[0] - x0 - float(t[0])
    dy = sy0 + loc[1] - y0 - float(t[1])
    return math.hypot(dx, dy)


def run(items, use_dets=True, tag="", verbose=True):
    """`items` = [(name, frames, boxes_or_None, truth_or_None, dt)] ⇒ 汇总打印 ✓。"""
    errs, local, lost, scored, tot = [], [], 0, 0, 0
    for name, frames, boxes, truth, dt in items:
        tr = LieTracker()
        gm = GroupMotion()
        prev_g = None
        for f, im in enumerate(frames):
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            dets = (boxes[f] if (use_dets and boxes) else None)
            typ = st.median([b[2] * b[3] for b in dets]) if dets else 0.0
            r = gm.estimate(im, dets or [], typ)
            o = tr.process(im, ts=(f + 1) * dt, dets=dets)
            tot += 1
            if o["state"] != "track" or o["pos"] is None:
                lost += 1
            else:
                scored += 1
                if truth is not None:
                    errs.append(math.hypot(o["pos"][0] - truth[f][0],
                                           o["pos"][1] - truth[f][1]))
                if prev_g is not None and r["t"] is not None:
                    d = local_dev(prev_g, g, o["pos"], r["t"])
                    if d is not None:
                        local.append(d)
            prev_g = g
    print("\n===== %s =====" % (tag or "结果"))
    print("帧 %d ｜ track %d（%.0f%%）｜ init %d" % (tot, scored,
                                                    100.0 * scored / max(1, tot), lost))
    if errs:
        s = sorted(errs)
        print("① **位置误差**（px）：中位 %.1f ｜ p90 %.1f ｜ 最大 %.1f ｜ ≤30px 占比 %.0f%%"
              % (s[len(s) // 2], s[int(len(s) * 0.9)], s[-1],
                 100.0 * sum(1 for x in s if x <= 30) / len(s)))
    if local:
        s = sorted(local)
        print("② **局部偏离 T**（px）：中位 %.1f ｜ p90 %.1f ｜ ＞5px 占比 %.0f%%"
              "（追对 ⇒ 应显著 >0 ✓）"
              % (s[len(s) // 2], s[int(len(s) * 0.9)],
                 100.0 * sum(1 for x in s if x > 5) / len(s)))
    return errs, local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=int, default=0)
    ap.add_argument("--real", type=int, default=0)
    ap.add_argument("--no-dets", action="store_true")
    ap.add_argument("--gif", default="")
    a = ap.parse_args()
    if a.synth:
        items = []
        for s in range(a.synth):
            c = make_case(seed=s + 1)
            items.append(("synth#%d" % (s + 1), c["frames"], c["boxes"], c["truth"], c["dt"]))
        run(items, use_dets=not a.no_dets,
            tag="合成台（%d 例，%s）" % (a.synth, "纯残差（无检测框）" if a.no_dets else "带检测框"))
    if a.real:
        from tools.lie_demo import DetsWorker, load_gif
        w = DetsWorker()
        items = []
        root = pathlib.Path("datasets/photo1")
        for d in sorted(x for x in root.iterdir() if x.is_dir())[:a.real]:
            fs = sorted(p for p in d.iterdir() if p.suffix.lower() == ".bmp")
            fr, prev = [], None
            for p in fs:
                im = cv2.imread(str(p))
                if im is None:
                    continue
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if prev is not None and float(cv2.absdiff(g, prev).mean()) < 0.5:
                    continue
                prev = g
                fr.append(im)
            items.append((d.name, fr, [None] * len(fr), None, 0.18))
        if a.gif and pathlib.Path(a.gif).exists():
            raw, _, _, _, fps = load_gif(a.gif)
            items.append((pathlib.Path(a.gif).name, [x[0] for x in raw], None, None, 1.0 / fps))
        # 真素材没有预计算框（键对不上 ✗）⇒ 现场跑检测器 ✓
        items = [(n, f, [w.detect(im) for im in f] if b is None else b, t, dt)
                 for n, f, b, t, dt in items]
        run(items, use_dets=True, tag="真素材（%d 例，物理判据）" % len(items))
        w.close()


if __name__ == "__main__":
    main()
