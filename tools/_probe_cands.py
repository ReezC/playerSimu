# -*- coding: utf-8 -*-
"""临时取证（**只读**）：把黄族的**每一个候选**都掏出来，逐个问"你到底是谁"。

用户 2026-10-06：新代码（底图相减优先）已经把落点从"锁死的假点"换成**段 9（底层）**✓，
但 x 还是对不上（−32…−70 vs 界面那行的 1176 ✗）⇒ 说明**两个候选还在互相抢** ✓。
这一枪要回答：每个候选的
  · 面板位置 / 大小 / 面积 / 实际颜色（BGR ✓）；
  · 像不像"玩家点核心色"（`_blob_looks_core` ✓）；
  · ⭐ **把它换算到底图上的同一点，底图那一块是不是本来就有东西**（有 ⇒ 它是**地图元素** ✗；
    没有 ⇒ 它才是"**多出来的**玩家点" ✓）—— 这条就是"假点"的铁证 ✓。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2                                                     # noqa: E402
import numpy as np                                             # noqa: E402

from core import mapdata                                       # noqa: E402
from perception import minimap as mm                           # noqa: E402
from tools import mmap_dot_probe as P                          # noqa: E402

MID = "110040000"
t = mapdata.load(MID, with_canvas=True)
cal = mapdata.load_calib(MID, "stream") or {}
assert t is not None and t.canvas is not None, "没有地形/底图"
cw, ch = t.canvas.shape[1], t.canvas.shape[0]
sx, sy = mm.scales_of(cal)
ox, oy = (list(cal.get("offset") or (0, 0)) + [0, 0])[:2]
ppw = float(t.px_per_world)
cx0 = float(t.mini.get("centerX") or 0)
cy0 = float(t.mini.get("centerY") or 0)
gray_canvas = cv2.cvtColor(t.canvas, cv2.COLOR_BGR2GRAY)
print("标定 scale=(%.4f,%.4f) offset=(%.1f,%.1f)  底图 %dx%d  px_per_world=%.3f"
      % (sx, sy, ox, oy, cw, ch, ppw))


def cand_xy(c):
    """候选的面板坐标（形状按 `_dot_candidates` 给的对象试几种取法 ✓）。"""
    for key in ("x", "cx"):
        if isinstance(c, dict) and c.get(key) is not None:
            return float(c[key]), float(c.get("y") or c.get("cy") or 0)
    try:
        return float(c["x"]), float(c["y"])
    except Exception:                                          # noqa: BLE001
        pass
    try:
        return float(c[0]), float(c[1])
    except Exception:                                          # noqa: BLE001
        return None, None


panels, note = P.grab_stream(3, 8.0)
print("取帧:", note)
for i, panel in enumerate(panels):
    m0 = mm.dot_mask(panel, "yellow")
    m_bm = mm._basemap_extra_mask(panel, cal, t, m0, "yellow")
    print("\n帧%d 面板=%dx%d  黄族像素: 颜色层 %d / 相减层 %d"
          % (i, panel.shape[1], panel.shape[0], int(m0.sum()), int(m_bm.sum())))
    for tag, m in (("color ", m0), ("basemap", m_bm)):
        cands, dense = mm._dot_candidates(m, panel.shape[1])
        print("  [%s] 候选 %d 个，成片 %d 像素" % (tag, len(cands), dense))
        for c in cands:
            px, py = cand_xy(c)
            if px is None:
                print("      ? %r" % (c,))
                continue
            core = bool(mm._blob_looks_core(panel, m, c))
            # 面板 → 底图 → 世界（口径同 `map_terrain_view.image_xy` 的反向 ✓）
            gx = (px - ox) / sx
            gy = (py - oy) / sy
            gxi, gyi = int(round(gx)), int(round(gy))
            canvas_gray = int(gray_canvas[gyi, gxi]) if (
                0 <= gxi < cw and 0 <= gyi < ch) else -1
            canvas_bgr = (tuple(int(v) for v in t.canvas[gyi, gxi])
                          if 0 <= gxi < cw and 0 <= gyi < ch else None)
            y0, y1 = max(0, gyi - 2), min(ch, gyi + 3)
            x0, x1 = max(0, gxi - 2), min(cw, gxi + 3)
            cg_mean = float(gray_canvas[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else -1
            wx = gx * ppw - cx0
            wy = gy * ppw - cy0
            print("      面板(%.1f,%.1f) 核心色=%s  底图同点=%s(灰%s/邻域均%.0f) "
                  "世界≈(%.0f,%.0f)"
                  % (px, py, "是" if core else "**否**", canvas_bgr, canvas_gray, cg_mean, wx, wy))
    print("  真值参考：「底层」= 段 9（x −2002..1202），你说人贴在**最右下角** ≈1176；"
          "h007=(855,150)")
