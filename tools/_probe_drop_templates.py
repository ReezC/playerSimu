"""只读探针：**掉落物模板匹配的误标归因**（用户 2026-10-06 问"图小、误标太多怎么办"）。

为什么要有它：掉落物图标只有十几二十像素（图库里全是 `icon_0.png` 一张），
`TM_CCORR_NORMED` 在这么小的模板上**分数分布很平** ⇒ 阈值/区分度稍松就满屏假框。
光看"标了多少框"说不出**是谁在乱标、标在哪** ⇒ 这里把那两件事量出来 ✓。

## 判据（为什么按"跨帧同位置聚类"排序）

  · **真掉落物**：掉在地上**位置不固定**、只出现在**少数帧**（捡走就没了）✓
  · **静态假框**：同一位置在**很多帧**里反复出现 —— 血条 / 快捷栏 / 地形花纹 / 文字 / 别的 UI ✗
  ⇒ 按"框中心聚类 → 跨帧出现次数"降序排 ⇒ **排在最前那几个位置就是误标的来源** ✓

## 三种用法

    python -m tools._probe_drop_templates                    # 只列图库（裁后尺寸 / 不透明像素 / 平坦度）
    python -m tools._probe_drop_templates --project 石人寺院III --n 40
    python -m tools._probe_drop_templates --frames <画面目录> --ids 04030000,04030001 --n 40

## 纪律

  · **只读** ✓：不改项目、不写配置、不动 `labels_auto`（临时标注写 `data/_probe_drop/` ✓）；
  · 匹配**复用真实现**（`tools.detect_mobs.run_detect` ✓ 一个 id 跑一趟）——
    ⛔ 探针里**不另写一套匹配** ✗（那样量出来的就不是线上那套 ✓）；
  · 每个 id **各写一个临时目录** ✓：`run_detect` 有"这一帧 0 检出就保留旧框"的保命逻辑 ✓，
    共用一个目录会让上一项的框串到下一项 ✗。
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from core.imgio import imread

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROBE_DIR = ROOT / "data" / "_probe_drop"


def _lib_stats(ids, mirror=False):
    """图库里这些 id 的模板长什么样（裁后尺寸 / 不透明像素 / **平坦度**）。

    「平坦度」= 灰度里与中位数差 ≤ 12 的像素占比 ✓ —— **只是弱参考** ✗⚠：
      它高**不一定**说明"没形状"（圆圈 / 圆盘类图标本身就有大片同色，轮廓却很清晰 ✓
      —— 用户 2026-10-06 当场纠正过这点 ✓）。**要判"该不该踢"得看图**（`--vis` ✓）
      或者看"同一位置跨帧反复中"（那才是实锤 ✓ 见文件头的判据 ✓）。
    """
    from core import wzexport
    from tools.detect_mobs import load_templates

    root = wzexport.drop_sprite_dir()
    # 复用线上那份加载（裁 alpha 包围盒 ✓ 与匹配时**同一份**模板 ✓）
    tpl = load_templates(Path(root), list(ids), 0,
                         min_side=5, min_alpha=60, frames_sel=None, mirror=mirror)
    if not tpl:
        print("这些 id 在图库里一个可用模板都没有（检查 datasets/sprites/drop/<id>/ ✓）")
        return
    print("模板 %d 个（每种 icon %s；见 load_templates ✓）"
          % (len(tpl), "还会镜像一份" if mirror else "**不镜像** —— 掉落物那条的默认"))
    print("%-10s %-14s %6s %6s %8s %8s" % ("id", "帧", "宽", "高", "不透明", "平坦度"))
    for mid, frame, b, al in tpl:
        g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        med = float(np.median(g))
        flat = float((np.abs(g - med) <= 12).mean())
        print("%-10s %-14s %6d %6d %8d %7.0f%%"
              % (mid, frame, b.shape[1], b.shape[0], int((al > 128).sum()), flat * 100))


def _read_boxes(out_dir, W, H):
    """读一个临时标注目录 ⇒ `[(cx, cy, w, h, 帧 stem)]`（像素坐标 ✓）。"""
    out = []
    for p in sorted(Path(out_dir).glob("*.txt")):
        for ln in p.read_text(encoding="utf-8").splitlines():
            v = ln.split()
            if len(v) != 5:
                continue
            _c, cx, cy, bw, bh = v
            out.append((float(cx) * W, float(cy) * H,
                        float(bw) * W, float(bh) * H, p.stem))
    return out


def _one_run(frames_dir, did, sample, out_root, thresh, dist, peaks, scale,
             min_side, min_alpha, sel=None, vis=False, energy=0.55, mirror=False):
    """跑一个 id 一趟真匹配 ⇒ 返回 `[(cx, cy, w, h, 帧), …]`（像素坐标 ✓）。

    `sel` 给了就只让这一帧当模板（`frames_sel` ✓ —— 这是线上那套"用户手工挑模板帧"的同一个口 ✓
    ⇒ **按帧归因**靠它 ✓，不用另写一套匹配 ✗）。
    """
    from core import wzexport
    from perception.classes import CLASS_DROP
    from tools.detect_mobs import run_detect

    key = did if not sel else "%s__%s" % (did, sel)
    out = Path(out_root) / ("out_" + key)
    try:
        run_detect({
            "mobs": [did], "sprites": str(wzexport.drop_sprite_dir()),
            "frames": str(frames_dir), "out": str(out),
            "vis_dir": str(Path(out_root) / ("vis_" + key)),
            "only": sample, "scale": float(scale),
            "min_side": int(min_side), "min_alpha": int(min_alpha),
            "thresh": float(thresh), "min_distinct": float(dist),
            "max_peaks": int(peaks), "downscale": 1,
            # ⭐ 掉落物链上那两处（见 `tools/label_drops.py` 头部 ✓）—— 探针**必须跟线上一致** ✗
            "min_energy_ratio": float(energy), "mirror": bool(mirror),
            "frames_sel": ({"drop:%s" % did: [sel]} if sel else None),
            "cls": int(CLASS_DROP), "target": "drop",
            "vis": bool(vis), "coarse": False,
        })
    except Exception as e:                           # noqa: BLE001 —— 单项失败别打断别的 ✓
        print("   （%s 跑不动：%s）" % (key, e))
        return []
    img = imread(sorted(Path(frames_dir).glob("*.png"))[0])
    H, W = img.shape[:2]
    return _read_boxes(out, W, H)


def _per_frame(frames_dir, ids, n, out_root, **kw):
    """**按模板帧归因**：一个 id 有 4 张 icon 时，到底是哪几张在乱标 ✓。

    做法 = 每次只让**一张**帧当模板（走线上的 `frames_sel` ✓）⇒ 谁的框多谁就是元凶 ✓
    ⇒ 直接告诉用户"④ 里那一条的**哪几帧该取消勾**"✓。
    """
    from core import wzexport
    from tools.detect_mobs import load_templates

    all_frames = sorted(Path(frames_dir).glob("*.png"))
    step = max(1, len(all_frames) // max(1, n))
    sample = [f.stem for f in all_frames[::step]][:n]
    _vis = bool(kw.pop("vis", False))       # ⚠ 先取出来（循环里 `pop` 只有第一次拿得到 ✗）
    for did in ids:
        tpl = load_templates(Path(wzexport.drop_sprite_dir()), [did], 0,
                             min_side=5, min_alpha=60)
        stems = []
        for _mid, frame, _b, _a in tpl:
            if frame not in stems:
                stems.append(frame)
        print("\n== %s ==（按模板帧归因，样本 %d 帧）" % (did, len(sample)))
        for st in stems:
            bx = _one_run(frames_dir, did, sample, out_root, sel=st,
                          vis=_vis, **kw)
            hit = len({b[4] for b in bx})
            print("   帧 %-8s 框 %4d 个 / 命中 %2d 帧（%.0f%%）%s"
                  % (st, len(bx), hit, 100.0 * hit / max(1, len(sample)),
                     "   ⚠ 这个模板就是误标来源" if len(bx) >= 20 else ""))


def _run(frames_dir, ids, n, thresh, dist, peaks, scale, min_side, min_alpha,
         vis=False, energy=0.55, mirror=False):
    """每个 id 各自跑一趟**真匹配**（只取 n 帧样本 ✓）⇒ 每项的框 + 静态簇统计。"""
    all_frames = sorted(Path(frames_dir).glob("*.png"))
    if not all_frames:
        print("画面目录里没有 png：%s" % frames_dir)
        return
    step = max(1, len(all_frames) // max(1, n))
    sample = [f.stem for f in all_frames[::step]][:n]
    # ⚠ 必须用 `core.imgio.imread`（**不是 `cv2.imread`** ✗）：本项目采下来的帧它读得开、
    #   `cv2.imread` 会直接返回 None（实测第一版就这么崩的 ✓）—— 全仓同一份读图口径 ✓。
    img = imread(all_frames[0])
    H, W = img.shape[:2]
    print("画面 %d 张 ⇒ 抽 %d 张样本（%s…）  尺寸 %dx%d"
          % (len(all_frames), len(sample), sample[0], W, H))
    print("参数：阈值 %.2f / 区分度 %.3f / 每帧最多 %d 框 / 尺度 %.3f / 小图闸 %dpx·%d"
          " / 能量比 %.2f / %s"
          % (thresh, dist, peaks, scale, min_side, min_alpha, energy,
             "镜像" if mirror else "**不镜像**"))
    if PROBE_DIR.exists():
        shutil.rmtree(PROBE_DIR, ignore_errors=True)

    for did in ids:
        boxes = _one_run(frames_dir, did, sample, PROBE_DIR,
                         thresh, dist, peaks, scale, min_side, min_alpha,
                         vis=vis, energy=energy, mirror=mirror)
        frames_hit = len({b[4] for b in boxes})
        print("\n== %s ==  框 %d 个 / 命中 %d 帧（样本 %d 帧，%.0f%% 的帧有框）"
              % (did, len(boxes), frames_hit, len(sample),
                 100.0 * frames_hit / max(1, len(sample))))
        if not boxes:
            continue
        # 静态簇：框中心按 8px 归并 ⇒ 同一位置跨帧出现几次（见文件头"判据" ✓）
        cluster = defaultdict(lambda: {"n": 0, "w": 0.0, "h": 0.0, "pos": None})
        for cx, cy, bw, bh, _st in boxes:
            k = (int(cx // 8), int(cy // 8))
            c = cluster[k]
            c["n"] += 1
            c["w"] += bw
            c["h"] += bh
            c["pos"] = (cx, cy)
        top = sorted(cluster.items(), key=lambda kv: -kv[1]["n"])[:6]
        for k, c in top:
            px, py = c["pos"]
            print("   %5d 帧@同一处  中心(%4.0f,%4.0f) 框 %2.0fx%2.0f  %s"
                  % (c["n"], px, py, c["w"] / c["n"], c["h"] / c["n"],
                     "⚠ 静态 ⇒ 多半是背景/UI 误标" if c["n"] >= max(3, len(sample) // 4)
                     else ""))
        tail = len(cluster) - len(top)
        if tail > 0:
            print("   （另有 %d 个位置各只出现 1~%d 次 —— 那些更像真掉落物/零星噪声 ✓）"
                  % (tail, top[-1][1]["n"] if top else 0))
    print("\n临时标注/可视化在 %s（只读探针留下的唯一产物 ✓ 可直接删）" % PROBE_DIR)


def _project_params(name):
    """按**项目配置**解析掉落物那条链的参数（与 ④ 卡片 `LabelCard.make_task` **同一套键** ✓）。

    ⚠⚠ **必须用项目里的数**（不是代码默认值 ✗）：这正是"量出来跟线上不一样"的老坑 ✓
      —— 实测（石人寺院III）：默认 `scale=1.0` 时 30 帧**一个框都没有** ✓，
      而线上用的是 `drop_scale`/`scale`（≈1.4）⇒ 拿默认值量等于白量 ✓。
    兜底链照抄卡片（`drop_*` → 总那个 → 常量 ✓）：老项目没存过 `drop_*` ⇒ 取到的是原来那个数 ✓。
    """
    from gui.project import Project

    p = Project.open(ROOT / "projects" / name)
    sec = p.sec("label") or {}
    _sc = p.get("drop_scale") or p.get("scale") or 1.12
    return {
        "scale": float(_sc),
        "thresh": float(sec.get("drop_thresh", sec.get("thresh", 0.90))),
        "dist": float(sec.get("drop_min_distinct", sec.get("min_distinct", 0.06))),
        "peaks": int(sec.get("drop_max_peaks", sec.get("max_peaks", 4))),
        "min_side": 5, "min_alpha": 60,          # 掉落物那条的默认（见 `label_drops` ✓）
        "drops": [str(d.get("id") if isinstance(d, dict) else d)
                  for d in (sec.get("drops") or [])],
        "visible": bool(sec.get("visible", False)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="", help="项目名（用它的 frames/ + **它自己的参数** ✓）")
    ap.add_argument("--frames", default="", help="画面目录（与 --project 二选一 ✓）")
    ap.add_argument("--ids", default="", help="逗号分隔的掉落物 id（默认 = 项目里配的 / 图库默认 ✓）")
    ap.add_argument("--n", type=int, default=30, help="抽多少帧当样本（默认 30 ✓）")
    ap.add_argument("--per-frame", action="store_true",
                    help="改成**按模板帧归因**：每次只让一张 icon 当模板 ⇒ 谁在乱标一目了然 ✓")
    ap.add_argument("--vis", action="store_true",
                    help="把框画到图上（`data/_probe_drop/vis_*/` ✓ 用眼睛确认哪些是真掉落物 ✓）")
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--min-distinct", type=float, default=None)
    ap.add_argument("--max-peaks", type=int, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--min-side", type=int, default=5)
    ap.add_argument("--min-alpha", type=int, default=60)
    ap.add_argument("--energy", type=float, default=None,
                    help="窗口能量比下限（默认 = 掉落物那条的 0.55 ✓ 见 label_drops ✓）")
    ap.add_argument("--mirror", action="store_true",
                    help="给掉落物**开镜像**（默认关 ✓ 想看「镜像多出多少假框」才加）")
    a = ap.parse_args()

    from core import wzexport

    proj = {}
    if a.project:
        try:
            proj = _project_params(a.project)
        except Exception as e:                       # noqa: BLE001 —— 读不到就说清 ✓
            print("读项目配置失败（%s）⇒ 这一趟用**默认值**量 ✗（结论不可当准 ✓）" % e)

    def _pick(k, cli, dflt):
        if cli is not None:
            return cli
        return proj.get(k, dflt)

    ids = [s.strip() for s in a.ids.split(",") if s.strip()] or \
        [i for i in (proj.get("drops") or []) if i] or \
        list(wzexport.default_drop_ids() or [])
    thr = _pick("thresh", a.thresh, 0.90)
    dist = _pick("dist", a.min_distinct, 0.06)
    peaks = _pick("peaks", a.max_peaks, 4)
    scale = _pick("scale", a.scale, 1.0)
    print("掉落物 id：%s" % "、".join(ids))
    if a.project and proj:
        print("参数来自项目「%s」的配置 ✓（与 ④ 卡片跑的是同一套 ✓）" % a.project)
    elif a.project:
        print("⚠ 项目配置没读到 ⇒ 下面是**默认值** ✗ 结论只能当参考 ✓")
    else:
        print("⚠ 没给 --project ⇒ 下面是**默认值** ✗（想量线上的数就加 `--project <项目名>` ✓）")

    frames = a.frames
    if not frames and a.project:
        frames = str(ROOT / "projects" / a.project / "frames")
    if not frames:
        _lib_stats(ids)
        print("\n（想量「标在哪」就加 `--project <项目名>` 或 `--frames <画面目录>` ✓）")
        return 0
    ap_vis = bool(a.vis)
    # ⭐ 掉落物链上那两处：能量比 + 不镜像（**缺省就按线上那份** ✓ 见 label_drops ✓）
    from tools.label_drops import DROP_MIN_ENERGY_RATIO

    energy = float(a.energy) if a.energy is not None else float(DROP_MIN_ENERGY_RATIO)
    mirror = bool(a.mirror)
    _lib_stats(ids, mirror=mirror)
    if a.per_frame:
        # 按模板帧归因（哪一个 icon 在乱标 ✓）⇒ 直接告诉用户"选帧里该取消勾哪几帧" ✓
        if PROBE_DIR.exists():
            shutil.rmtree(PROBE_DIR, ignore_errors=True)
        _per_frame(frames, ids, a.n, PROBE_DIR, thresh=thr, dist=dist,
                   peaks=peaks, scale=scale, min_side=a.min_side,
                   min_alpha=a.min_alpha, vis=ap_vis, energy=energy, mirror=mirror)
        print("\n临时标注/可视化在 %s（只读探针留下的唯一产物 ✓ 可直接删）" % PROBE_DIR)
        return 0
    _run(frames, ids, a.n, thr, dist, peaks, scale, a.min_side, a.min_alpha,
         vis=ap_vis, energy=energy, mirror=mirror)
    return 0


if __name__ == "__main__":
    sys.exit(main())
