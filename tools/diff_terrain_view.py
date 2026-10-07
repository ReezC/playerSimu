# -*- coding: utf-8 -*-
"""⭐ **底图相减 + 地形线 + 全部命名，合成一张图**（用户 2026-10-06 ✓ 原话：

    "2.底图相减后，把地形线条绘制上去（包含所有的地图元素命名）"

**做什么**：拿一帧小地图面板（默认 A 机推流 ✓）减去**底图** ⇒ 只剩"底图上没有的东西"
（玩家点 / 怪 / 特效 ✓）；把这份差分按标定**摆回底图坐标**、放大到和地形图同一尺度，
再用**红点**画在地形图上 ⇒ 一张图同时看得到：**地形线 ✓ 元素名 ✓ 差分（多出来的东西）✓**。

**为什么用红点叠、不直接显示差分**：差分图是黑白的、看不到"这是哪块平台"✗；叠在地形图上
才能一眼回答"这个多出来的点落在哪条段 / 哪个集合" ✓（查黄点认错那种事就靠它 ✓）。

**只读**：不按键、不写配置、不碰标定文件 ✓；输出落 `data/diff_terrain/` ✓。

跑法：
    python tools/diff_terrain_view.py 110040000              # 收流取一帧
    python tools/diff_terrain_view.py 110040000 --image x.png # 离线量一张
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2                                                     # noqa: E402
import numpy as np                                             # noqa: E402

from core import mapdata                                       # noqa: E402
from core import zones as Z                                    # noqa: E402
from perception import minimap as mm                           # noqa: E402
from tools import map_terrain_view as MTV                      # noqa: E402

OUT_DIR = ROOT / "data" / "diff_terrain"
TARGET_W = 1600
#: 差分阈值（灰阶）：超过它才算"底图上没有的东西"（同 `mm.DIFF_FLOOR_WARN` 的量级 ✓）
DIFF_MIN = 40


def _panel_from_stream(seconds=8.0):
    from tools import mmap_dot_probe as P
    frames, note = P.grab_stream(1, seconds)
    print("取帧:", note)
    return frames[0] if frames else None


def main() -> int:
    ap = argparse.ArgumentParser(description="底图相减 + 地形线 + 命名（合成一张 ✓）")
    ap.add_argument("map_id", nargs="?", default="110040000")
    ap.add_argument("--image", help="离线：直接量一张面板图")
    ap.add_argument("--src", default="stream", help="标定按哪条来源取（默认 stream ✓）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t = mapdata.load(a.map_id, with_canvas=True)
    if t is None or t.canvas is None:
        print("没有 %s 的地形/底图（先导出地形并生成地形图）" % a.map_id)
        return 2
    panel = (cv2.imread(a.image) if a.image else _panel_from_stream())
    if panel is None:
        print("没拿到面板画面")
        return 2
    cal = mapdata.load_calib(a.map_id, a.src) or {}
    if not mm.has_geometry(cal):
        print("这份标定没有几何（先在「路线识别 → 小地图定位」标定一次）")
        return 2
    ok, why = mm.panel_mismatch(cal, panel, (t.canvas.shape[1], t.canvas.shape[0]))
    if not ok:
        print(why)
        return 3

    cw, ch = t.canvas.shape[1], t.canvas.shape[0]
    sx, sy = mm.scales_of(cal)
    ox, oy = (list(cal.get("offset") or (0, 0)) + [0, 0])[:2]
    z = max(1, int(round(float(TARGET_W) / max(1, cw))))

    # ---- 面板 → 底图尺度的一块（`canvas = (panel − offset) / scale` ✓）----
    pw, ph = panel.shape[1], panel.shape[0]
    sw, sh = max(1, int(round(pw / sx))), max(1, int(round(ph / sy)))
    small = cv2.resize(cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY), (sw, sh),
                       interpolation=cv2.INTER_AREA)
    px, py = int(round(-float(ox) / sx)), int(round(-float(oy) / sy))
    gray_canvas = cv2.cvtColor(t.canvas, cv2.COLOR_BGR2GRAY)
    ref = np.zeros((sh, sw), np.uint8)
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(cw, px + sw), min(ch, py + sh)
    if x1 > x0 and y1 > y0:
        ref[y0 - py:y1 - py, x0 - px:x1 - px] = gray_canvas[y0:y1, x0:x1]
    # 只在"面板盖住底图"的那块里比 ⇒ 面板里露出的游戏 UI 不算差分 ✓
    valid = np.zeros_like(ref, np.uint8)
    valid[y0 - py:y1 - py, x0 - px:x1 - px] = 255
    diff = cv2.absdiff(small, ref)
    mask = ((diff >= DIFF_MIN) & (valid > 0)).astype(np.uint8) * 255
    n_px = int((mask > 0).sum())
    print("面板 %dx%d → 底图尺度 %dx%d（摆到底图 (%d,%d) 起，%dx 放大）"
          % (pw, ph, sw, sh, px, py, z))
    print("差分像素: %d（阈值 %d）" % (n_px, DIFF_MIN))

    # ---- 底图：地形线 + 名字（复用「生成地形图」那一份 ✓）----
    base_path = OUT_DIR / ("%s_terrain.png" % a.map_id)
    zones = None
    try:
        zones = Z.load(a.map_id)
    except Exception:                                          # noqa: BLE001
        zones = None
    MTV.render(t, base_path, zones=zones, target_w=TARGET_W, k=t.px_per_world)
    base = cv2.imread(str(base_path), cv2.IMREAD_UNCHANGED)
    if base is None:
        print("底图没画出来")
        return 2
    if base.ndim == 3 and base.shape[2] == 4:
        vis, alpha = base[:, :, :3].copy(), base[:, :, 3].copy()
    else:
        vis, alpha = base.copy(), np.full(base.shape[:2], 255, np.uint8)

    # ---- 把差分**红点**叠上去（放大 z 倍 ✓ 与底图同一尺度）----
    big = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    hit = big > 0
    vis[hit] = (0, 0, 255)                    # BGR 红 ✓：这就是"底图上没有的东西"
    alpha[hit] = 255
    out = a.out or str(OUT_DIR / ("%s_diff_terrain.png" % a.map_id))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, cv2.merge([vis, alpha]))
    print("已写出: %s  (%dx%d, 红=%d 像素)"
          % (out, vis.shape[1], vis.shape[0], int(hit.sum())))
    print("看图要点：地形线+名字=地图；**红点**=这一帧里底图上没有的东西（玩家点就该是唯一的大红团 ✓）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
