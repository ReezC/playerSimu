# -*- coding: utf-8 -*-
"""一次性验证（用完即删）：**粘连降权**生效了吗 ✓（用户 2026-10-03 ✓ "别太急" ✓）。

要看三件事（帧 13/14 = 实测粘连那两拍 ✓）：
  ① `stuck` 有没有认出来 ✓；
  ② 目标 `score` **不再被那个假 `dev` 顶上去** ✓（旧：帧 13 → 198.6 ✗）；
  ③ 报出的 `pos` **不再被拽到"两个目标的中点"** ✓（它应该贴着**预测/旧位置**走 ✓）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.lie_demo as D                                          # noqa: E402
from perception.lie_motion import MotionTracker                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "liedetectorVideo" / "10月1日.mp4"

frames, ts, proc, scale, fps = D.load_source(str(SRC))
_w = D.DetsWorker(None, conf=0.25)
_dets = [_w.detect(f[1]) for f in frames]
_w.close()

_tr = MotionTracker()
print("帧 ｜ sel ｜ 面积/基准 ｜ stuck ｜ dev ｜ score ｜ 报出的 pos ｜ 它这拍的 obs(框中心)")
for i in range(len(frames)):
    _tr.process(frames[i][1], ts=ts[i], dets=_dets[i])
    if not (10 <= i + 1 <= 20):
        continue
    _s = _tr._by_id(_tr.tid)
    if _s is None:
        continue
    print("%3d ｜ #%-3d ｜ %5.2fx %s ｜ %5d ｜ %5.1f ｜ %6.1f ｜ (%5.1f,%5.1f) ｜ (%5.1f,%5.1f)"
          % (i + 1, _s.id, (float(_s.w) * float(_s.h)) / max(1.0, _s.area_ema),
             "⚠粘住" if _s.stuck else "     ", _s.stuck, _s.dev, _s.score,
             _tr.pos[0], _tr.pos[1], _s.obs[0], _s.obs[1]))
