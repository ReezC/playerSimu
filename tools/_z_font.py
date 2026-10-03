# -*- coding: utf-8 -*-
"""一次性验证（用完即删）：**字号调大后画布上长什么样** ✓（用户 2026-10-04 ✓ "要瞎了" ✓）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2                                                           # noqa: E402
import tools.lie_demo as D                                           # noqa: E402
from perception.lie_motion import MotionRunner                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "liedetectorVideo" / "10月1日.mp4"

frames, ts, proc, scale, fps = D.load_source(str(SRC))
_w = D.DetsWorker(None, conf=0.25)
_dets = [_w.detect(f[1]) for f in frames]
_w.close()
_h, _wd = frames[0][1].shape[:2]
_rn = MotionRunner(dets=_dets, gain=None, assume=(_wd / 2.0, _h / 2.0), follow_gain=1.0)

for i in range(20):
    _o = _rn.step(frames[i][1], i, ts[i])
    if i + 1 != 16:
        continue
    _big = frames[i][0]
    _mo = dict(_o[5] or {})
    _mo["show"] = {"cands": True, "rel": True, "boxes": True}
    _vis = D.draw(_big.copy(), (_o[0], _o[1], _o[2], _o[3], _o[4]),
                  scale[0], scale[1], "Frame 16/70 | track",
                  cursor=_rn.cursor, motion=_mo, names=["shape"], mask=0.30)
    cv2.imwrite(str(ROOT / "tools" / "_z_font.png"), _vis)
    print("已出图：%s（%d×%d）" % (ROOT / "tools" / "_z_font.png",
                                  _vis.shape[1], _vis.shape[0]))
