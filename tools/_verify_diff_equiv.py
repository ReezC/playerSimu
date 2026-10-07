"""临时验证（只读）：差分层这次"省算力"的改写**逐像素等价**吗。

用户 2026-10-06 ✓ 帧率现场（`py-spy`：`_amax` **9.4%** ✗）⇒ 把 `diff_panel_vs_canvas`
里两次"整幅浮点数组"与 `dot_candidates_diff` 里那次 `vis.max(axis=2)` 省掉 ✓。
**等价性论证**（不靠眼看 ✓）：
  ① `vis` 是 `cvtColor(gray, GRAY2BGR)` ⇒ 三通道相等 ⇒ `vis.max(axis=2) == vis[:,:,0]` ✓；
  ② `info["gray"]` 就是那份本色（= `vis[:,:,0]`）⇒ 用它算的门限与用 `max` 算的**逐像素同值** ✓；
  ③ `at` = `argmax`（`clip` 单调 ⇒ 与老写法找的是同一个点 ✓）、`peak = max(0, g[at] − floor)` ✓。
用**真实那块 App 面板**跑一遍，把三条都断言掉 ✓。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                              # noqa: E402

from core import mapdata                                # noqa: E402
from perception import minimap as mm                    # noqa: E402

MID = "110040000"
t = mapdata.load(MID, with_canvas=True)
cal = mapdata.load_calib(MID, "stream") or {}
panel = cv2.imread(str(ROOT / "data" / "dot_probe" / "app_panel.png"))
print("面板:", None if panel is None else panel.shape)
assert panel is not None

ok = True
for tag, src in (("整幅", panel), ("半幅（模拟 ROI 路径）", panel[:300, :600])):
    vis, info = mm.diff_panel_vs_canvas(src, t.canvas, cal)
    if vis is None:
        print("[%s] 差分算不出来：%s" % (tag, info.get("why")))
        continue
    g = info["gray"]
    a = np.array_equal(vis[:, :, 0], vis[:, :, 1]) and \
        np.array_equal(vis[:, :, 0], vis[:, :, 2])
    b = np.array_equal(g, vis[:, :, 0])
    m_old = (vis.max(axis=2) >= mm.DOT_DIFF_THR)
    m_new = (g >= mm.DOT_DIFF_THR)
    c = np.array_equal(m_old, m_new)
    ay, ax = np.unravel_index(int(g.argmax()), g.shape)
    d = (info["at"] == (int(ax), int(ay)))
    e = abs(info["peak"] - max(0.0, float(g[ay, ax]) - info["floor"])) < 0.05
    print("[%s] 三通道相等=%s  本色==第0通道=%s  掩码逐像素同值=%s  "
          "at=argmax=%s  peak 同值=%s"
          % (tag, a, b, c, d, e))
    print("       floor=%s peak=%s at=%s 掩码像素=%d（老写法 %d）"
          % (info["floor"], info["peak"], info["at"],
             int(m_new.sum()), int(m_old.sum())))
    ok = ok and a and b and c and d and e
print("==> 等价性:", "全部成立 ✓" if ok else "**有不等价的地方 ✗**")
