"""A 层验收（重构第 1 步）：在**真素材**上量群体一致性 + 与旧追踪器交叉核对。

⚠ 第一版这里拿 `|ΔT|` 当"抖动"是**错的口径** ✗ —— 它含**真实位移**（~20~25px/帧 ✓），而且
photo1 采集间隔不规律（capture.txt: fps=1.75 ✗）⇒ 每帧位移本来就变 ✗，拿它判断好坏没意义 ✗。
真正该看的三件：
  ① **帧内共识紧不紧**（`spread` = 框残差中位 ✓ 越小越好 ✓）；
  ② **群体一致性**的分布（随群体平移的框 ≈ 1.0 ✓，真目标/合并块明显低 ✓）；
  ③ **交叉核对**：一致性最差那只的位置，是否落在**旧追踪器报出的位置**附近 ✓
     （旧版离线 281 cycle 命中 mean 0.98 ✓ ⇒ 拿它当"真值代理" ✓）。

用法：python -m tools.probe_group_motion --cycles 6
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics as st
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perception.lie_group import GroupMotion  # noqa: E402
from perception.lie_tracker import LieTracker  # noqa: E402
from tools.lie_demo import DetsWorker  # noqa: E402


def load_cycle(dirp, dets, worker):
    """⇒ `[(bgr, boxes)]`（重复帧去重 ✓；缺预计算检测就现场跑 ✓）。"""
    fs = sorted(p for p in pathlib.Path(dirp).iterdir()
                if p.suffix.lower() in (".bmp", ".png", ".jpg"))
    out, prev = [], None
    for p in fs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if prev is not None and float(cv2.absdiff(g, prev).mean()) < 0.5:
            continue
        prev = g
        b = dets.get(p.stem)
        if b is None:
            b = worker.detect(im)
        out.append((im, [(float(x[1]), float(x[2]), float(x[3]), float(x[4]))
                         for x in b]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--dets", default="datasets/lie/dets_v2.json")
    ap.add_argument("--gif", default="")
    a = ap.parse_args()
    dets = json.loads(pathlib.Path(a.dets).read_text(encoding="utf-8")) \
        if pathlib.Path(a.dets).exists() else {}
    worker = DetsWorker()
    srcs = []
    root = pathlib.Path("datasets/photo1")
    for d in sorted(x for x in root.iterdir() if x.is_dir())[a.skip:a.skip + a.cycles]:
        srcs.append((d.name, load_cycle(d, dets, worker)))
    if a.gif and pathlib.Path(a.gif).exists():
        from tools.lie_demo import load_gif
        raw, _, _, _, _ = load_gif(a.gif)
        fr = [(f[0], [(float(x[1]), float(x[2]), float(x[3]), float(x[4]))
                      for x in worker.detect(f[0])]) for f in raw]
        srcs.append((pathlib.Path(a.gif).name, fr))

    gm = GroupMotion()
    spread, c_med, c_min, nin, frames, near, far, nreg = [], [], [], [], 0, 0, 0, 0
    for name, fr in srcs:
        gm.reset()
        tr = LieTracker()
        prev_t = None
        for im, boxes in fr:
            typ = st.median([w * h for _, _, w, h in boxes]) if boxes else 0.0
            r = gm.estimate(im, boxes, typ)
            o = tr.process(im, ts=(frames + 1) * 0.18, dets=boxes)
            frames += 1
            if r["t"] is None or not r.get("consist"):
                continue
            spread.append(r["spread"])
            c_med.append(r["consist_med"])
            c_min.append(r["consist_min"])
            nin.append(r["used"])
            # ③ 交叉核对：一致性最差那只的中心 vs 旧追踪器位置
            if o.get("pos") is not None and r.get("dev_i") is not None:
                j = r["dev_i"]
                if 0 <= j < len(boxes):
                    d = math.hypot(boxes[j][0] - o["pos"][0], boxes[j][1] - o["pos"][1])
                    nreg += 1
                    if d <= 90:
                        near += 1
                    else:
                        far += 1
            prev_t = r["t"]
    worker.close()
    print("有效帧 %d ｜ 参与共识框数 中位 %.0f" % (frames, st.median(nin) if nin else 0))
    if spread:
        s = sorted(spread)
        print("① 帧内共识紧度（框残差中位）：中位 %.2f px ｜ p90 %.2f" %
              (s[len(s) // 2], s[int(len(s) * 0.9)]))
        cs = sorted(c_med)
        cm = sorted(c_min)
        print("② 群体一致性：**中位框** 中位 %.3f ｜ 一致性**最差框** 中位 %.3f ｜ p10 %.3f"
              % (cs[len(cs) // 2], cm[len(cm) // 2], cm[int(len(cm) * 0.1)]))
    if nreg:
        print("③ 交叉核对：一致性最差那只落在旧追踪器位置 ≤90px 的占比 **%.0f%%**（%d/%d ✓）"
              % (100.0 * near / nreg, near, nreg))


if __name__ == "__main__":
    main()
