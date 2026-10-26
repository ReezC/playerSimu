# -*- coding: utf-8 -*-
"""小地图「定位不上」体检：一条命令说清是**框错了**、**底图不对**、还是**标定过期**。

为什么要有它（2026-10-22 现场 ✓ 用户："**森林迷宫珠缨 无法去二楼**"）：
  症状是"人站着不动、任务不往下走"，日志里只有一句
  `mmap_pick n=0 … 没和底图对上 ⇒ 停车` —— 但"为什么对不上"要看三个东西：
    · **框**（`mmap_crop`）有没有把小地图的**标题栏**一起框进来（标题栏上是文字和图标，
      和地图内容毫无关系 ⇒ 匹配分被白拉低一截 ✗）；
    · **底图**（`datasets/map/<id>.png`）和游戏里那张小地图**是不是同一张**（长宽比先对一眼）；
    · **标定**（`.mapcalib.json`）的几何还对不对。
  这三件事原来只能靠现场一条条试，现在一次报出来 ✓（**只读，不改任何文件** ✓）。

用法：
    python -m tools.mmap_diag                     # 用最近打开的项目 + 抓一帧实时画面
    python -m tools.mmap_diag --map 105040303 --src live
    python -m tools.mmap_diag --image 面板图.png  # 离线：直接喂一张面板图

⚠ 判据一律用仓库里那份实现（`overlap_pearson` / `CropViewTracker` / `locate_crop`），
  不在这里另写一套"像不像" ✗ —— 两套分迟早互相打架（见 `minimap.overlap_pearson` 的说明）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2                                    # noqa: E402
import numpy as np                            # noqa: E402

from core import mapdata                      # noqa: E402
from perception import map_const, minimap as mm   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _recent_project():
    """最近打开的项目目录（`config/session.json` 里那个）→ `gui.project.Project` | None。"""
    try:
        from gui.project import Project
        p = ROOT / "config" / "session.json"
        if not p.exists():
            return None
        name = json.loads(p.read_text(encoding="utf-8")).get("last_project")
        return Project.open(Path(name)) if name else None
    except Exception:                          # noqa: BLE001
        return None


def _crop_info(proj, cfg):
    """框选到底从哪来（项目 / 全局兜底 / 标定里的 panel）—— 三处口径一次说清。"""
    out = {}
    try:
        out["project"] = (proj.get("mmap_crop") if proj is not None else None)
    except Exception:                          # noqa: BLE001
        out["project"] = None
    out["live.yaml"] = (cfg or {}).get("mmap_crop")
    return out


def _title_band_ratio(panel, thr=150.0, rows=12):
    """面板顶端那几行有多亮 —— 小地图**标题栏**（白字 + 图标）会把均值顶得很高 ✓。

    只用来提醒"你这框把标题栏框进来了"，不参与任何判据 ✓
    """
    if panel is None or panel.size == 0:
        return None
    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    k = max(1, min(rows, g.shape[0] // 4))
    return float(g[:k].mean())


def _grab_panel(map_id, src, crop, image=None, project=None):
    """拿到"面板那一块"：给了 `--image` 就用它，否则抓一帧实时画面按 `crop` 裁。"""
    if image:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        return img, "离线图片 %s" % image
    from core import wincap
    win = None
    try:
        rel_win = (project or {}).get("rect_win")
        if rel_win and len(rel_win) == 2:
            win = int(rel_win[0])
    except Exception:                          # noqa: BLE001
        win = None
    if win is None:
        win = wincap.window_by_title("冒险岛")
    if not win:
        return None, "没找到游戏窗口（用 --image 喂一张面板图也行）"
    geom = wincap.window_geom(win)
    img = wincap.grab_rect(tuple(geom)) if geom else None
    if img is None:
        return None, "抓屏失败：%s" % wincap.last_note()
    if not crop:
        return img, "整帧（还没框过小地图）"
    x, y, w, h = [int(v) for v in crop]
    part = img[y:y + h, x:x + w]
    return (part if part.size else None), "实时画面 框=%r" % (crop,)


def main() -> int:
    ap = argparse.ArgumentParser(description="小地图定位体检（只读）")
    ap.add_argument("--map", dest="map_id", help="地图 id（不给就用最近打开的项目）")
    ap.add_argument("--src", default=None, choices=["live", "stream"],
                    help="哪条来源的标定（不给就按这张图默认那条）")
    ap.add_argument("--image", help="离线：直接给一张**面板**图（不是整帧）")
    ap.add_argument("--crop", help="离线时先按 x,y,w,h 从 --image 上裁一块")
    args = ap.parse_args()

    proj = _recent_project()
    cfg = {}
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config" / "live.yaml").read_text(encoding="utf-8")) or {}
    except Exception:                          # noqa: BLE001
        cfg = {}
    map_id = str(args.map_id or (proj.get("map_id") if proj is not None else "") or "")
    if not map_id:
        print("✗ 不知道是哪张图：给 --map，或先在工作台里打开一个项目")
        return 2
    src = args.src or str(mm.source_for_map(map_id, cfg))

    print("=" * 66)
    print("地图 %s   来源 %s" % (map_id, src))

    terr = mapdata.load(map_id, with_canvas=True)
    if terr is None or getattr(terr, "canvas", None) is None:
        print("✗ 这张图没有底图（`datasets/map/%s.png`）—— 先「生成地形图」" % map_id)
        return 1
    canvas = terr.canvas
    ch, cw = canvas.shape[:2]
    calib = dict(mapdata.load_calib(map_id, src=src) or {})
    print("底图 canvas = %dx%d   长宽比 %.2f" % (cw, ch, cw / float(ch)))

    crops = _crop_info(proj, cfg)
    print("框选：项目=%r   live.yaml=%r" % (crops["project"], crops["live.yaml"]))
    crop = crops["project"] or crops["live.yaml"]
    if not crop and calib.get("panel"):
        crop = [int(calib.get("offset", [0, 0])[0]), 0] + list(calib["panel"])[:2]
    print("标定：mode=%s scale=%r offset=%r view=%r panel=%r legacy=%s"
          % (calib.get("mode"), calib.get("scale"), calib.get("offset"),
             calib.get("view"), calib.get("panel"), calib.get("legacy")))

    panel, how = _grab_panel(map_id, src, [int(v) for v in crop] if crop else None,
                             image=args.image, project=proj)
    print("面板：%s → %s" % (how, None if panel is None else panel.shape))
    if panel is None or panel.size == 0:
        return 1
    ph, pw = panel.shape[:2]

    band = _title_band_ratio(panel)
    if band is not None and band > 90:
        print("⚠ 面板**顶端很亮**（前 12 行均值 %.0f）—— 很可能把「小地图标题栏/图例」"
              "一起框进来了：\n   标题栏上是文字和图标，和地图内容无关 ⇒ 匹配分会被白拉低一截 ✗"
              % band)

    if pw / float(ph) > 1.6 * (cw / float(ch)):
        print("⚠ 面板长宽比 %.2f 和底图 %.2f 差得远 —— 先怀疑**底图不是这张图的**（或框到了一半）"
              % (pw / float(ph), cw / float(ch)))

    sc, ov, why = mm.overlap_pearson(panel, canvas, calib) if calib else (None, 0.0, "没有标定")
    print("\n① 按**现有标定**量：pearson=%s  重叠=%.2f  %s"
          % ("—" if sc is None else "%.3f" % sc, ov, str(why)[:90]))

    tr = mm.CropViewTracker()
    o = tr.update(panel, canvas, calib)
    print("② 跟踪器第一拍：ok=%s score=%s where=%s view=%s"
          % (o.get("ok"), "%.3f" % (o.get("score") or 0), o.get("where"), o.get("view")))
    if not o.get("ok"):
        print("   why=%s" % str(o.get("why"))[:140])

    best = None
    try:
        best = mm.locate_crop(panel, canvas, min_score=0.0)
    except Exception as e:                     # noqa: BLE001
        print("③ 自动重搜索失败：%s" % e)
    if best:
        print("③ 自动重搜索：score=%.3f scale=%.3f view=%s"
              % (float(best.get("score") or 0), float(best.get("scale") or 0), best.get("view")))
        if float(best.get("score") or 0) < map_const.TRUST_SCORE:
            print("   ⚠ 连「自动找」都过不了可信线 %.2f ⇒ 多半是**底图/框**的问题，"
                  "不是标定误差" % map_const.TRUST_SCORE)

    print("\n结论（按上面三条自己会对上一条）：")
    print("  · 顶端很亮 + 有标题栏字样 ⇒ 重框，只框小地图**内容区**")
    print("  · 长宽比差得远 / 重搜索也上不去 ⇒ 底图 `datasets/map/%s.png` 换一张对的" % map_id)
    print("  · 以上都正常、只有 ① 低 ⇒ 重标一次（双点标定）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
