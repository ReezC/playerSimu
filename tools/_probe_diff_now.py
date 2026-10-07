"""临时取证（**只读**）：把**差分图**（面板 − 底图 ⇒ 黑底 + 亮点）原样存下来看。

用户 2026-10-06 ✓ 原话："**为什么还要担心被沙吞掉？从黑色背景里找黄点应该都没有沙了**"✓
—— 完全对 ✓：差分图里沙已经被减掉 ✓，玩家点应该是**黑底上一个亮团** ✓。
本脚本存三张图（面板 / 差分 / 差分二值）＋ 差分层的候选，回答"真点在不在差分图上" ✓。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402

from core import mapdata                                # noqa: E402
from core.imgio import imwrite                          # noqa: E402
from perception import minimap as mm                    # noqa: E402
from tools import mmap_dot_probe as P                   # noqa: E402

MID = "110040000"
OUT = ROOT / "data" / "dot_probe"
t = mapdata.load(MID, with_canvas=True)
cal = mapdata.load_calib(MID, "stream") or {}

panels, note = P.grab_stream(1, 8.0)
print("取帧:", note)
panel = panels[0]
imwrite(str(OUT / "now_panel.png"), panel)

vis, info = mm.diff_panel_vs_canvas(panel, t.canvas, cal)
print("差分 info:", {k: info.get(k) for k in ("ok", "why", "floor", "peak",
                                             "raw_mean", "raw_peak", "at", "rect")})
if vis is not None:
    imwrite(str(OUT / "now_diff.png"), vis)
    m = (vis.max(axis=2) >= mm.DOT_DIFF_THR).astype(np.uint8)
    imwrite(str(OUT / "now_diff_mask.png"), np.dstack([m * 255] * 3))
    print("差分二值掩码: %d 像素（%.2f%% 面板）" % (int(m.sum()),
                                                   100.0 * m.sum() / m.size))
    for tag, near in (("首次捕获 / 弃锁重捕（near=None）", None),
                      ("跟踪态（喂 App 那个假锁 856,501）", (856.5, 501.8))):
        r = mm.find_player_dot(panel, calib=cal, terrain=t, near=near)
        if r.get("x") is not None:
            wx, wy = mm.panel_to_world(float(r["x"]), float(r["y"]), cal, t)
        else:
            wx = wy = None
        print("  [%s] ok=%s layer=%s px=(%s,%s) 世界≈(%s,%s)"
              % (tag, r.get("ok"), r.get("layer"), r.get("x"), r.get("y"), wx, wy))
        print("      reason=%s" % str(r.get("reason"))[:140])
    dc, dd, di = mm.dot_candidates_diff(panel, cal, t)
    print("差分层候选 %d 个（成片 %d）:" % (len(dc), dd))
    for c in dc:
        wx, wy = mm.panel_to_world(c["x"], c["y"], cal, t)
        fh = t.foothold_below(wx, wy)
        print("   (%.1f,%.1f) %sx%s 面积%d  世界≈(%.0f,%.0f) 脚下=%s"
              % (c["x"], c["y"], c["w"], c["h"], c["area"], wx, wy,
                 getattr(fh, "fid", None)))
else:
    print("差分算不出来:", info.get("why"))
