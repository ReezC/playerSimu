# -*- coding: utf-8 -*-
"""临时取证：**模拟 App 那种"带着旧锁定"的状态**，看最终挑哪个（用户 2026-10-06 ✓）。

我的探针每次新建 locator（没历史）⇒ 挑对 ✓；App 带着上一次的位置（`near` ✓）⇒ 挑错 ✗
（现场 (198,195) ✗ vs 真值 (1176,178) ✓）。这一枪就把 `near` 塞进去，直接看结果。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from core import mapdata                                       # noqa: E402
from perception import minimap as mm                           # noqa: E402
from tools import mmap_dot_probe as P                          # noqa: E402

MID = "110040000"
t = mapdata.load(MID, with_canvas=True)
cal = mapdata.load_calib(MID, "stream") or {}


def _xy(c):
    if isinstance(c, dict):
        return c.get("x"), c.get("y")
    return c[0], c[1]


panels, note = P.grab_stream(3, 8.0)
print("取帧:", note)
for i, f in enumerate(panels):
    roi = mm._calib_roi(f.shape[:2], cal, t)
    r0 = mm.find_player_dot(f, calib=cal, terrain=t, roi=roi)
    dc, _dd, _di = mm.dot_candidates_diff(f, calib=cal, terrain=t,
                                          ref_w=f.shape[1], roi=roi)
    fakes = [_xy(c) for c in dc] if dc else []
    print("\n帧%d 差分层候选 %d 个: %s" % (i, len(dc), [(round(a or 0, 1), round(b or 0, 1))
                                                      for a, b in fakes]))
    print("   ① 无历史        → (%.1f,%.1f) 层=%s"
          % (r0.get("x") or -1, r0.get("y") or -1, r0.get("layer")))
    for fx, fy in fakes:
        r1 = mm.find_player_dot(f, calib=cal, terrain=t, roi=roi, near=(fx, fy))
        print("   ② 假锁 near=(%.0f,%.0f) → (%.1f,%.1f) 层=%s   %s"
              % (fx, fy, r1.get("x") or -1, r1.get("y") or -1, r1.get("layer"),
                 (r1.get("reason") or "")[:60]))
