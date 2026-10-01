# -*- coding: utf-8 -*-
"""M2a 离线回放器：把 `datasets/liedetectorphotos` 按 cycle 按时间戳喂给追踪器，
输出"假如鼠标在上一帧追踪位置"的命中率报告（判据见 docs/测谎设计.md §5）。

用法：
    python -m tools.lie_replay --sample 8            # 抽 8 个 cycle 试跑
    python -m tools.lie_replay --all                 # 全量
    python -m tools.lie_replay --cycles 211,250 --viz-out datasets/lie/viz

评分口径：
  · **合成鼠标** = 追踪器**上一处理帧**的输出位置（滞后一步 = 真实系统延迟的下界 ✓）；
  · 命中 = 鼠标点落在目标块内（半径 = sqrt(area/π)，下限 18px ✓）；
  · 追踪器报 lost ⇒ 该帧记 miss（诚实评分 ✗ 不藏）。
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perception.lie_tracker import LieTracker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAP = ROOT / "datasets" / "liedetectorphotos"
PAT = re.compile(r"cycle_(\d+)_(\d+)_([\d.]+)s\.bmp")
_MIN_RADIUS = 18.0


def load_cycles():
    """cycle 内的帧序 = **文件创建时间序**（用户 2026-09-29 ✓）—— 录制器会**复用帧序号**
    （同一 cycle 里 `0000001` 有 7 个文件 ✗）⇒ 按文件名排序会把真时序打乱 ✗。"""
    cyc = defaultdict(list)
    for p in CAP.glob("*.bmp"):
        m = PAT.match(p.name)
        if m:
            cyc[m.group(1)].append((os.path.getctime(str(p)), int(m.group(2)),
                                    float(m.group(3)), p))
    out = {}
    for k, v in cyc.items():
        v.sort(key=lambda r: r[0])
        out[k] = [(idx, ts, p) for _ct, idx, ts, p in v]
    return out


def run_cycle(cid, frames, viz_dir=None, dets=None):
    """跑一个 cycle ⇒ 摘要 dict（顺带按需导出标注帧 ✓）。

    `dets`（M2b ✓）：{帧 stem: [[cls, cx, cy, w, h], ...]} —— 按 stem 查到就传给
    追踪器（复活/吸附通道 ✓）；没查到 = 纯残差路线 ✓。
    """
    tr = LieTracker()
    rows = []
    for _idx, ts, path in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        d = None
        if dets is not None:
            raw = dets.get(path.stem)
            if raw:
                d = [b[1:] for b in raw]      # 去掉 cls ⇒ [cx, cy, w, h] ✓
        rows.append((ts, tr.process(img, ts=ts, dets=d), path))
    scored = hits = 0
    first_lost = None            # ⭐ `lost` 已废除（用户 2026-10-01 ✓ 三态：融合/锁定/预测 ✓）
    prev_pos = None              #   丢失 = 预测态（绿圈照画）⇒ 不再有独立"丢失"统计 ✗
    snap = {}
    for ts, out, path in rows:
        if out["state"] == "track" and out["pos"] is not None:
            if prev_pos is not None:
                scored += 1
                r = max(_MIN_RADIUS, (out["area"] or 0.0) ** 0.5 / 1.7725)
                d = ((prev_pos[0] - out["pos"][0]) ** 2
                     + (prev_pos[1] - out["pos"][1]) ** 2) ** 0.5
                if d <= r:
                    hits += 1
            prev_pos = out["pos"]
        else:
            prev_pos = None
        # 可视化抽样：建档帧 / 每 15 处理帧一张 ✓
        if viz_dir is not None and out["state"] == "track":
            n = sum(1 for _, o, _ in rows if o["state"] == "track")
            if n == 1 or n % 15 == 0:
                im = cv2.imread(str(path))
                cv2.circle(im, (int(out["pos"][0]), int(out["pos"][1])),
                           max(6, int((out["area"] or 0) ** 0.5 / 1.7725)),
                           (0, 255, 0), 2)
                if prev_pos is not None:
                    cv2.drawMarker(im, (int(prev_pos[0]), int(prev_pos[1])),
                                   (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(im, "%s %5.2fs %s" % (cid, ts, out["state"]),
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2)
                snap["%s_%04d" % (cid, len(snap))] = im
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        for name, im in snap.items():
            cv2.imwrite(str(viz_dir / ("%s.png" % name)), im)
    dur = rows[-1][0] - rows[0][0] if rows else 0.0
    return {"cid": cid, "frames": len(rows), "dur": dur,
            "init_ok": any(o["state"] == "track" and o["init"] is not None
                           for _, o, _ in rows),
            "lost_end": False,      # ⭐ `lost` 已废除 ✓ 丢失 = 预测态（绿圈照画）⇒ 恒"未丢" ✓
            "first_lost": first_lost,
            "scored": scored, "hits": hits,
            "hit": (hits / scored) if scored else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--cycles", type=str, default="")
    ap.add_argument("--viz-out", type=str, default="")
    ap.add_argument("--dets", type=str, default="",
                    help="v2 检测 json（_tmp_lie_dets.py 产物）⇒ 启用 M2b 复活通道 ✓")
    args = ap.parse_args()
    cyc = load_cycles()
    dets = None
    if args.dets:
        import json
        dets = json.loads(Path(args.dets).read_text(encoding="utf-8"))
        print("检测通道：启用（%d 帧 ✓）" % len(dets))
    if args.cycles:
        ids = [c.strip().zfill(5) for c in args.cycles.split(",") if c.strip()]
    elif args.all:
        ids = sorted(cyc)
    else:
        ids = sorted(cyc)[::max(1, len(cyc) // max(1, args.sample))][:args.sample]
    viz = Path(args.viz_out) if args.viz_out else None
    rep = [run_cycle(c, cyc[c], viz_dir=viz, dets=dets) for c in ids if c in cyc]
    ok = [r for r in rep if r["init_ok"]]
    full = [r for r in ok if not r["lost_end"]]
    print("cycles: %d 跑通 / init 成功 %d / 全程未丢 %d"
          % (len(rep), len(ok), len(full)))
    if ok:
        import statistics
        hs = [r["hit"] for r in ok]
        print("命中率（init 成功的）: mean=%.2f med=%.2f min=%.2f"
              % (statistics.mean(hs), statistics.median(hs), min(hs)))
    worst = sorted(ok, key=lambda r: r["hit"])[:10]
    for r in worst:
        print("  差 %s hit=%.2f 帧=%d 时长=%.1fs 首丢=%s"
              % (r["cid"], r["hit"], r["frames"], r["dur"],
                 ("%.1fs" % r["first_lost"]) if r["first_lost"] else "未丢"))
    bad = [r for r in rep if not r["init_ok"]]
    for r in bad[:5]:
        print("  建档失败 %s 帧=%d 时长=%.1fs" % (r["cid"], r["frames"], r["dur"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
