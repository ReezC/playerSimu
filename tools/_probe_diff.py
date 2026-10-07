# -*- coding: utf-8 -*-
"""临时取证（**只读**）：差分层为什么不给候选（用户 2026-10-06："报 (380,238) 没站在平台上"）。

把 `dot_candidates_diff` 内部的每一个数都打出来：
  floor(背景档) / raw_peak(最亮) / 对比度 / 两道闸 / 归一化掩码上的候选数与"被淹"判定。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np                                             # noqa: E402

from core import mapdata                                       # noqa: E402
from perception import minimap as mm                           # noqa: E402
from tools import mmap_dot_probe as P                          # noqa: E402

MID = "110040000"
t = mapdata.load(MID, with_canvas=True)
cal = mapdata.load_calib(MID, "stream") or {}
print("门槛：DIFF_FLOOR_WARN=%s(只提示) DOT_DIFF_MIN_CONTRAST=%s DOT_DIFF_THR=%s"
      % (mm.DIFF_FLOOR_WARN, mm.DOT_DIFF_MIN_CONTRAST, mm.DOT_DIFF_THR))

panels, note = P.grab_stream(2, 8.0)
print("取帧:", note)
for i, panel in enumerate(panels):
    roi = mm._calib_roi(panel.shape[:2], cal, t)
    vis, di = mm.diff_panel_vs_canvas(panel, t.canvas, cal)
    print("\n帧%d 面板=%dx%d roi=%s" % (i, panel.shape[1], panel.shape[0], roi))
    if vis is None:
        print("   diff_panel_vs_canvas 没算出图 ✗：%s" % (di,))
        continue
    floor = float(di.get("floor") or 0.0)
    peak = float(di.get("raw_peak") or 0.0)
    print("   背景档 floor=%.1f   最亮 raw_peak=%.1f   对比度=%.1f"
          % (floor, peak, peak - floor))
    print("   闸①不同源 floor>=%s ? %s（只提示 ✓）" % (mm.DIFF_FLOOR_WARN, floor >= mm.DIFF_FLOOR_WARN))
    print("   闸②对比度 < %s ? %s" % (mm.DOT_DIFF_MIN_CONTRAST, (peak - floor) < mm.DOT_DIFF_MIN_CONTRAST))
    m = (vis.max(axis=2) >= mm.DOT_DIFF_THR).astype(np.uint8)
    m2 = mm._apply_roi(m, roi)
    cands, dense = mm._dot_candidates(m2, panel.shape[1])
    area = 1
    if roi:
        rx0, ry0, rx1, ry1 = [int(v) for v in roi]
        area = max(1, (rx1 - rx0) * (ry1 - ry0))
    print("   归一化掩码(>=%d) 命中 %d 像素；候选 %d 个、成片 %d 像素 ⇒ 被淹? %s"
          % (mm.DOT_DIFF_THR, int(m2.sum()), len(cands), dense,
             mm._is_flood(len(cands), dense, area)))
    dc, dd, dinfo = mm.dot_candidates_diff(panel, calib=cal, terrain=t,
                                           ref_w=panel.shape[1], roi=roi)
    print("   dot_candidates_diff ⇒ 候选 %d 个 / 成片 %d ；info: ok=%s why=%s"
          % (len(dc), dd, dinfo.get("ok"), (dinfo.get("why") or "")[:110]))
    if dc:
        for c in dc[:6]:
            try:
                px, py = (c.get("x"), c.get("y")) if isinstance(c, dict) else (c[0], c[1])
            except Exception:                                  # noqa: BLE001
                px = py = None
            print("      候选 (%.1f,%.1f)  %r" % (px, py, c if not isinstance(c, dict) else ""))
