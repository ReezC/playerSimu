"""临时验证（**只读**）：跟踪器那条「局部搜索」现在会不会被假锁困住。

现场（用户 2026-10-06 ✓）：App 锁死在一个**假点**上（`px≈993/955/910` ✗），真点在
`(1135,478) ⇒ 世界 (1176,145)+偏移33 = (1176,178)` ✓ —— 而老写法把面板裁成 37×37、
又不传 calib/terrain ⇒ 真点永远在窗口外、几何层全跳过 ⇒ 钉死在 198 ✓。
这里：**同一块 App 面板** + **摆那个假锁** ⇒ 期望**甩掉假锁、落到真点** ✓。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                              # noqa: E402

from core import mapdata                                # noqa: E402
from perception import minimap as mm                    # noqa: E402

MID = "110040000"
t = mapdata.load(MID, with_canvas=True)
cal = mapdata.load_calib(MID, "stream") or {}
_src = ROOT / "data" / "dot_probe" / "app_panel.png"
_snap = ROOT / "data" / "dot_probe" / "_snap_app_panel.png"
import shutil                                           # noqa: E402
shutil.copyfile(str(_src), str(_snap))                  # ⭐ 先快照：App 每 10 秒覆盖它 ✓
panel = cv2.imread(str(_snap))
print("面板(快照):", None if panel is None else panel.shape,
      " 源文件改于", _src.stat().st_mtime)
if panel is not None:
    _vis, _info = mm.diff_panel_vs_canvas(panel, t.canvas, cal)
    print("差分 at =", _info.get("at"), " floor/peak =",
          _info.get("floor"), _info.get("peak"))
    print("核心色掩码像素 =", int(mm.dot_core_mask(panel).sum()))
    _dc, _dd, _di = mm.dot_candidates_diff(panel, cal, t)
    print("差分层候选 %d 个:" % len(_dc))
    for _c in _dc:
        _wx, _wy = mm.panel_to_world(_c["x"], _c["y"], cal, t)
        print("   (%.1f,%.1f) %sx%s 面积%d 世界≈(%.0f,%.0f)"
              % (_c["x"], _c["y"], _c["w"], _c["h"], _c["area"], _wx, _wy))

for lock in ((993.2, 487.8), (955.0, 371.6), (910.3, 311.8), (856.5, 501.8)):
    print("--- 假锁 (%.1f, %.1f) ---" % lock)
    # ① 直接问跟踪器（改的就是它 ✓）
    try:
        tr = mm.PlayerDotTracker()
        try:
            tr._last = {"x": lock[0], "y": lock[1]}
        except Exception:                               # noqa: BLE001
            tr._last = None
        tr.x, tr.y = lock
        _t0 = time.perf_counter()
        r = tr._search(panel, cal, t, None)
        _ms = (time.perf_counter() - _t0) * 1000.0
        wx, wy = (None, None)
        if r.get("x") is not None:
            wx, wy = mm.panel_to_world(float(r["x"]), float(r["y"]), cal, t)
        print("   跟踪器: ok=%s layer=%s px=(%.1f,%.1f) 世界≈(%s,%s)  **耗时 %.2f ms**"
              % (r.get("ok"), r.get("layer"), r.get("x") or 0, r.get("y") or 0,
                 None if wx is None else round(wx, 1),
                 None if wy is None else round(wy, 1), _ms))
    except Exception as e:                              # noqa: BLE001
        print("   跟踪器: 调用失败 %s: %s" % (type(e).__name__, e))
    # ② 对照：不带 roi 的全画面（我前几轮验过的那条 ✓）
    r2 = mm.find_player_dot(panel, calib=cal, terrain=t, near=lock)
    wx2, wy2 = (None, None)
    if r2.get("x") is not None:
        wx2, wy2 = mm.panel_to_world(float(r2["x"]), float(r2["y"]), cal, t)
    print("   全画面: ok=%s layer=%s px=(%.1f,%.1f) 世界≈(%s,%s)"
          % (r2.get("ok"), r2.get("layer"), r2.get("x") or 0, r2.get("y") or 0,
             None if wx2 is None else round(wx2, 1),
             None if wy2 is None else round(wy2, 1)))
print("期望：两条都给 世界 x ≈ 1176 ✓（y=145 再加标定偏移 33 = 178 ✓）")
