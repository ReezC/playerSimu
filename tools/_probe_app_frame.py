"""临时取证（**只读**）：拿 **App 自己存的那块面板** 喂给**现在的代码**，看会给出什么。

为什么（用户 2026-10-06 ✓ 现场"卡住了"）：App 的 `pos` 还是 `x=198.3 y=194.6` ✗
—— 和**几小时前**一模一样 ⇒ 先判"是代码没生效（旧进程 ✓）"还是"新代码在这个画面上也错 ✗"。
判据：**同一块面板**（`data/dot_probe/app_panel.png` ✓ App 每 10 秒写一次 ✓）在现在的代码下
应当给出 世界 x ≈ **1176** ✓（用户从小地图下方信息栏读到的数 ✓）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402

from core import mapdata                                # noqa: E402
from perception import minimap as mm                    # noqa: E402

MID = "110040000"
OUT = ROOT / "data" / "dot_probe"
t = mapdata.load(MID, with_canvas=True)
cal = mapdata.load_calib(MID, "stream") or {}
p = OUT / "app_panel.png"
print("面板:", p, p.stat().st_size, "B")
panel = cv2.imread(str(p))
print("形状:", None if panel is None else panel.shape)

if panel is not None:
    vis, info = mm.diff_panel_vs_canvas(panel, t.canvas, cal)
    print("差分 at =", info.get("at"), " floor/peak =",
          info.get("floor"), info.get("peak"))
    for tag, near in (("near=None（首次/重捕）", None),
                      ("near=(955,372)（App 现在的锁附近）", (955.0, 371.6)),
                      ("near=(993,488)", (993.2, 487.8))):
        r = mm.find_player_dot(panel, calib=cal, terrain=t, near=near)
        wx = wy = None
        if r.get("x") is not None:
            wx, wy = mm.panel_to_world(float(r["x"]), float(r["y"]), cal, t)
        print("  [%s] ok=%s layer=%s px=(%s,%s) 世界≈(%s,%s)"
              % (tag, r.get("ok"), r.get("layer"), r.get("x"), r.get("y"),
                 None if wx is None else round(wx, 1),
                 None if wy is None else round(wy, 1)))
        print("      reason=%s" % str(r.get("reason"))[:130])
    # 差分层候选原样摆出来（人眼可判 ✓）
    dc, dd, di = mm.dot_candidates_diff(panel, cal, t)
    print("差分层候选 %d 个（成片 %d）at=%s:" % (len(dc), dd, di.get("at")))
    for c in dc:
        wx, wy = mm.panel_to_world(c["x"], c["y"], cal, t)
        print("   (%.1f,%.1f) %sx%s 面积%d  世界≈(%.0f,%.0f)"
              % (c["x"], c["y"], c["w"], c["h"], c["area"], wx, wy))
    print("参考：App 现在的 pos = (198.3, 194.6) ✗；信息栏真值 ≈ (1176, 178) ✓")
    print("     核心色掩码像素 = %d" % int(mm.dot_core_mask(panel).sum()))
