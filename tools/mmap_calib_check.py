"""小地图标定**精度核对**（只读）：量一遍「当前这份标定差多少**世界像素**」。

**为什么要它**（用户 2026-09-27："先把全局小地图弄准"）：在这之前，"这份标定准不准"
**到处都问不出来** ✗ ——
  · `score`（匹配分）是无量纲的"像不像"；
  · `resid_px` / `axis_gap_pct` 只长在**两点法**里，而且是**面板像素**；
  · `tools/mmap_dot_probe.py` 量的是**黄点认不认得出来**（识别率）。
用户决策④：容差按像素定值、**表述一律世界坐标** ⇒ 这里就把它变成一个**世界像素**的数 ✓。

做法（`perception.minimap.check_calib`，一处实现 ✓）：模板匹配当一把**独立的尺子** ——
同一张真帧，把底图压进去另解一份几何（`locate` + 细化 ✓），再看"同一个面板像素，两套
几何映射到世界差多少"，四个角都算、报最坏那个 ✓。**只报数、不改任何东西**。

⚠ **先看"尺子可信吗"**：命中的那块要是混着别的游戏 UI、或"显示方式"选错，匹配分会掉到
0.6~0.7 一带 —— 那种分数下报出来的偏差**只能当参考**（结论文里会写出来 ✓）。

跑法（和取证工具同风格）：

    python -m tools.mmap_calib_check --src live        # 实时画面（按项目的框选区域裁）
    python -m tools.mmap_calib_check --src stream      # A 机独立小地图推流
    python -m tools.mmap_calib_check --image 现场.png [--box x,y,w,h]

    --map 105040303     用哪张图的地形/标定（不给就用「最近打开的项目」那张）
    --project 森林迷宫III  取哪个项目的框选区域（不给就用最近打开的那个）
    --frames 8 --seconds 20  实时来源收多少帧（默认 8 帧 / 最多等 20 秒）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2                                                 # noqa: E402
import numpy as np                                         # noqa: E402

from core import mapdata                                   # noqa: E402
from gui.project import last_opened                        # noqa: E402
from perception import minimap as mm                       # noqa: E402
from core.config import load_live                         # noqa: E402
from tools.mmap_dot_probe import grab_live, grab_stream     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _map_id(a, project):
    """用哪张图的地形/标定：命令行 > 项目 > 都不给就报错（不许猜 ✗）。"""
    if a.map_id:
        return a.map_id
    if project is not None and project.get("map_id"):
        return str(project.get("map_id"))
    return None


def _report(tag, frame, terrain, calib, mode=None):
    """量一帧 → 打印一行结论；返回这次的"最坏世界偏差"（给中位数用）。"""
    r = mm.check_calib(frame, terrain, calib, mode=mode)
    if r["err_world"] is None:
        print("  [%s] 量不出来：%s" % (tag, r["why"]))
        return None
    print("  [%s] %s" % (tag, r["verdict"]))
    print("         偏差 x/y = %+.0f / %+.0f 世界像素（最坏在面板 %s；"
          "1 实时像素 = %.2f 世界像素）"
          % (r["err_world_x"], r["err_world_y"],
             tuple(int(v) for v in r["worst_panel"]), r["px_per_world"]))
    print("         标定 scale=%.4f  vs 尺子 scale=%.4f（差 %+.2f%%）；"
          "offset %s vs %s"
          % (r["scale_cur"], r["scale_ref"], r["scale_delta_pct"] or 0.0,
             [round(float(v), 2) for v in r["offset_cur"]],
             [round(float(v), 2) for v in r["offset_ref"]]))
    if not r["trust"]:
        print("         ⚠ 尺子匹配分只有 %.2f（< %.2f）⇒ 上面这个数**只能当参考**："
              "画面里小地图没框全 / 被游戏 UI 挡住 / 显示方式选错"
              % (r["score"], mm.TRUST_SCORE))
    return float(r["err_world"])


def main():
    ap = argparse.ArgumentParser(description="小地图标定精度核对（只读）")
    ap.add_argument("--src", default="live", choices=["live", "stream"],
                    help="量哪条来源（默认 live = 实时画面里裁；stream = A 机独立推流）")
    ap.add_argument("--image", help="离线模式：直接量一张图（面板图，不是整帧）")
    ap.add_argument("--box", help="离线模式：先从这张图上裁一块 x,y,w,h（给整帧截图时才要）")
    ap.add_argument("--map", dest="map_id", help="用哪张图的地形/标定（不给就用最近打开的项目）")
    ap.add_argument("--project", help="取哪个项目的框选区域（不给就用最近打开的那个）")
    ap.add_argument("--frames", type=int, default=8, help="实时来源收多少帧（默认 8）")
    ap.add_argument("--seconds", type=float, default=20.0, help="实时来源最多等多久（默认 20s）")
    a = ap.parse_args()

    from gui.project import Project
    project = None
    if a.project:
        root = ROOT / "projects" / a.project
        if not (root / "project.yaml").exists():
            print("没有这个项目：%s（%s 下没有 project.yaml）" % (a.project, root))
            return 2
        project = Project.open(root)
    if project is None:
        project = last_opened()        # 和取证工具同一条口径 ✓
    live = load_live()

    mid = _map_id(a, project)
    if not mid:
        print("不知道用哪张图的标定：加 --map <地图id>，或先在工作台里打开一个项目")
        return 2
    t = mapdata.load(mid, with_canvas=True)
    if t is None or t.canvas is None:
        print("地图 %s 没有底图（先在「路线识别」里生成地形图）" % mid)
        return 2

    src = mm.SRC_STREAM if a.src == "stream" else mm.SRC_LIVE
    calib = mapdata.load_calib(mid, src)
    print("地图 %s（底图 %s，1 底图像素 = %.3f 世界像素）"
          % (mid, t.canvas.shape[:2], t.px_per_world))
    print("来源 %s　标定 %s" % (mm.SRC_LABEL.get(src, src),
                              "有" if mm.has_geometry(calib) else "**没有**（先标一次）"))
    if not mm.has_geometry(calib):
        print("  标定还没有几何 —— 「路线识别 → 寻路配置 → 标定…」或双点标定一次再来量 ✓")
        return 1

    print("标定几何：mode=%s scale=%.4f/%s offset=%s view=%s"
          % (calib.get("mode"), calib.get("scale"),
             round(float(mm.scales_of(calib)[1]), 4), calib.get("offset"),
             calib.get("view")))
    errs = []
    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            print("读不出这张图：%s" % a.image)
            return 2
        if a.box:
            try:
                x, y, w, h = (int(v) for v in str(a.box).split(","))
            except Exception:
                print("--box 要写成 x,y,w,h")
                return 2
            frame = frame[y:y + h, x:x + w]
        errs.append(_report("离线图", frame, t, calib))
    elif a.src == "stream":
        print("收 A 机小地图推流…")
        frames, note = grab_stream(a.frames, a.seconds)
        if note:
            print("  （%s）" % note)
        for i, f in enumerate(frames[:2]):
            errs.append(_report("stream#%d" % (i + 1), f, t, calib))
    else:
        crop = mm.crop_of(project, live)
        if not crop:
            print("来源 live 需要**小地图框选区域**：工作台「路线识别 → 寻路配置 → 框选小地图」"
                  "框一次（**按项目存**）")
            return 2
        print("按框选区域 %s 从实时画面裁…" % (crop,))
        frames, note = grab_live(crop, a.frames, a.seconds)
        if note:
            print("  （%s）" % note)
        for i, f in enumerate(frames[:2]):
            errs.append(_report("live#%d" % (i + 1), f, t, calib))

    good = [e for e in errs if e is not None]
    if good:
        print("── 中位偏差 %.0f 世界像素（≈ %.1f 实时像素；判据 %.0f = 和「坐标对齐误差范围」"
              "同一把尺）" % (float(np.median(good)), float(np.median(good))
                            / float(t.px_per_world), 10.0))
    else:
        print("── 一帧都没量出来：先把实时/推流跑起来（画面里要能看见小地图）")
        return 1
    # 只读工具：不写任何文件、不改标定 ✓
    print("（只读核对：没有改任何东西 —— 要改标定走「双点标定」/「标定…」✓）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
