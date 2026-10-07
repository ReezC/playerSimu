# -*- coding: utf-8 -*-
"""临时取证（**只读**）：实时在跑时，把"决策那条链"每一步的真实输入/输出打出来。

用户 2026-10-06："现在实时在跑你直接查" ✓。要回答的是：
  · 决策那条链喂的是哪块像素（尺寸 ✓）、用的哪份标定（src/mode ✓）；
  · **黄点认在哪一层**（`color` / `basemap` / `diff` ✓ —— 现场签名是"颜色层从没赢过"✗）；
  · 算出来的世界坐标反查地形落在哪条段 / 哪个集合（对照现场真值：段 9「底层」右端附近 ✓）。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from core import mapdata                                       # noqa: E402
from core import zones as Z                                    # noqa: E402
from perception import minimap as mm                           # noqa: E402
import mmap_dot_probe as P                                     # noqa: E402

MID = "110040000"
cal = mapdata.load_calib(MID, "stream") or {}
t = mapdata.load(MID, with_canvas=True)
imp = mm.implied_panel(cal, (t.canvas.shape[1], t.canvas.shape[0])) if t.canvas is not None else None
print("标定[stream] mode=%s scale=%s src=%s 记的panel=%s" % (
    cal.get("mode"), cal.get("scale"), cal.get("src"), mm.panel_of(cal)))
print("底图 PNG = %s   反推「整张底图要多大面板」= %s" % (
    None if t.canvas is None else (t.canvas.shape[1], t.canvas.shape[0]),
    tuple(round(v) for v in imp) if imp else None))
z = Z.Zones(MID)
print("真值参考：段 9 = 底层一带，右端 x≈1202；h007=(855,150)")

panels, note = P.grab_stream(5, 8.0)
print("取帧:", note)
if not panels:
    raise SystemExit("没取到推流帧")
loc = mm.PlayerLocator(MID)
for i, f in enumerate(panels):
    r = loc.update(f, src="stream", terrain=t, fh_xtol=12)
    wx, wy = r.get("world_x"), r.get("world_y")
    fh = t.foothold_below(wx, wy) if wx is not None else None
    seg = t.segment_of(wx, wy) if wx is not None else None
    try:
        sets = list(z.set_of(str(getattr(fh, "fid", ""))))
    except Exception:                                          # noqa: BLE001
        sets = []
    print("帧%d 面板=%dx%d 点=(%.1f,%.1f) 层=%-8s 世界=(%.1f,%.1f) 面=%s 段=%s 集合=%s | %s"
          % (i, f.shape[1], f.shape[0], r.get("px") or -1, r.get("py") or -1,
             r.get("dot_layer") or "?", wx if wx is not None else float("nan"),
             wy if wy is not None else float("nan"), getattr(fh, "fid", None),
             getattr(seg, "index", None), sets, (r.get("dot") or "")[:44]))
