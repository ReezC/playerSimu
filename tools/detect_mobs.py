"""模板匹配自动标注器（多进程）。

用 WZ 导出的精灵作模板，在真实画面上做归一化互相关匹配，
高分命中即为伪标注，再 NMS 去重。

CLI:
    python -m tools.detect_mobs --mobs 0130100,0130101 --frames data/plain2 ^
        --sprites datasets/sprites/mob --out datasets/labels_plain2 --downscale 2

GUI:
    调 run_detect(params, ctx)，见 core/context.py
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from core.context import ConsoleContext, TaskContext
from core.imgio import imread, imwrite

CLASS_MOB = 1
_W = {}


# ══════════════════════════════════════════════════════════════
# 模板
# ══════════════════════════════════════════════════════════════
# 死亡动画不做模板。
#
# 怪物临死时的样子和活着时差异极大（倒下、碎裂、淡出、变半透明），
# 拿它当模板会带来两个坏处：
#   1. 把"正在消失的怪"也框出来 —— 而 bot 不该攻击一只马上就没的目标，
#      这等于主动教模型学错
#   2. 死亡帧透明度高、边缘糊，匹配位置也不稳
# 排除之后这些帧自然不会被检出，正是我们想要的结果。
SKIP_ACTIONS = ("die",)


def _pick_frames(files, max_per_mob=0):
    """挑出用作模板的帧：**排除死亡动画**，其余**全部**用。

    **为什么默认全用**
        怪物的动作帧本来就不多（本项目实测：绿水灵 12 帧、木妖 6 帧、
        猪猪 8 帧），全用上代价很小，却能完整覆盖每个动画相位。

        曾经用 per_mob=6 做取样，结果把绿水灵 stand 的第三个相位
        （最窄的 stand_2）挤掉了 —— 画面里处于该相位的怪**永远匹配不上**，
        于是自动标注从来不标它，训练集里一个样本都没有，模型也就不认识。
        这种"看不见的洞"比零星误检更难发现。

    max_per_mob 是**上限，不是目标数**：
        · 非死亡帧数 ≤ 上限  → 全部使用，不截断（常规情况）
        · 非死亡帧数 > 上限  → 才按动作轮询取样，保证各动作都有代表帧
        · 0 或负数           → 完全不限制

    **为什么仍然保留上限**
        少数怪（尤其是 Boss）可能有上百个动作帧，全用会让匹配耗时失控，
        所以留一道保险，而不是放任自流。
    """
    groups = {}
    for f in files:
        act = f.name.rsplit("_", 1)[0]
        if act.startswith(SKIP_ACTIONS):
            continue
        groups.setdefault(act, []).append(f)

    # 保持原始文件顺序，避免下游对顺序有依赖时行为突变
    flat = [f for f in files if f.name.rsplit("_", 1)[0] in groups]

    if not flat:
        return []
    if max_per_mob <= 0 or len(flat) <= max_per_mob:
        return flat          # ← 不超上限就全用，这是默认路径

    # 战斗里最常见的姿态优先排队；未列入的动作按名字排在后面
    order = ("stand", "move", "hit1", "hit2",
             "attack1", "attack2", "skill1", "jump")
    acts = [a for a in order if a in groups]
    acts += [a for a in sorted(groups) if a not in acts]

    picked = []
    round_ = 0
    while len(picked) < per_mob:
        added = False
        for a in acts:
            if round_ < len(groups[a]):
                picked.append(groups[a][round_])
                added = True
                if len(picked) >= per_mob:
                    break
        if not added:
            break
        round_ += 1

    return picked


def load_templates(root, mob_ids, max_per_mob=0):
    """载入模板并裁到 alpha 包围盒。

    max_per_mob 是**上限**（默认 20）：每只怪取"除死亡动画外的所有帧"，
    帧数超过这个上限时才按动作轮询取样。详见 _pick_frames 的说明。
    """
    out = []
    root = Path(root)

    for mid in mob_ids:
        d = root / mid
        if not d.is_dir():
            continue

        files = _pick_frames(sorted(d.glob("*.png")), max_per_mob)

        for f in files:
            t = imread(f, cv2.IMREAD_UNCHANGED)
            if t is None or t.ndim != 3 or t.shape[2] != 4:
                continue

            b, al = t[:, :, :3], t[:, :, 3]

            # 裁掉透明边框 —— 大片透明区会把整幅响应图抬高，制造假匹配
            ys, xs = np.nonzero(al > 128)
            if len(xs) < 200:
                continue

            b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

            if b.shape[0] < 20 or b.shape[1] < 20:
                continue

            out.append((mid, b, al))
            # 怪会朝左右两个方向，镜像多一份模板，召回明显更好
            out.append((mid, cv2.flip(b, 1), cv2.flip(al, 1)))

    return out


# ══════════════════════════════════════════════════════════════
# NMS
# ══════════════════════════════════════════════════════════════
def nms(dets):
    """dets: [(score, x, y, w, h, tag)]，按分数贪心抑制重叠框。"""
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []

    for d in dets:
        _, x, y, w, h, _ = d
        for k in keep:
            _, kx, ky, kw, kh, _ = k
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            u = w * h + kw * kh - ix * iy
            if u > 0 and ix * iy / u > 0.35:
                break
        else:
            keep.append(d)

    return keep


# ══════════════════════════════════════════════════════════════
# 进程池 worker（不能碰 ctx —— 它跑在子进程里）
# ══════════════════════════════════════════════════════════════
def init_worker(cfg):
    cv2.setNumThreads(1)
    _W["cfg"] = cfg
    _W["tpl"] = load_templates(Path(cfg["sprites"]), cfg["mobs"], cfg["per_mob"])


def work(fp_str):
    cfg = _W["cfg"]
    fp = Path(fp_str)

    full = imread(fp)
    if full is None:
        return None

    H, W = full.shape[:2]
    img = full
    ox = oy = 0

    if cfg["region"]:
        ox, oy, rw, rh = cfg["region"]
        img = full[oy:oy + rh, ox:ox + rw]

    ds = cfg["ds"]
    if ds > 1:
        img = cv2.resize(img, None, fx=1.0 / ds, fy=1.0 / ds,
                         interpolation=cv2.INTER_AREA)

    dets = []

    for mid, b, al in _W["tpl"]:
        for sc in cfg["scales"]:
            h = int(round(b.shape[0] * sc / ds))
            w = int(round(b.shape[1] * sc / ds))
            if h < 8 or w < 8 or h >= img.shape[0] or w >= img.shape[1]:
                continue

            t = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
            m = (cv2.resize(al, (w, h), interpolation=cv2.INTER_AREA) > 128)
            m = m.astype(np.uint8) * 255

            try:
                r = cv2.matchTemplate(img, t, cv2.TM_CCORR_NORMED, mask=m)
            except Exception:
                continue

            r = np.nan_to_num(r)
            if r.max() < cfg["thr"]:
                continue

            # 区分度判据：峰值必须显著高于该响应图的背景水平。
            # 只看绝对分数会在画面各处产生海量假匹配。
            p = float(np.percentile(r, 99.5))

            for _ in range(cfg["peaks"]):
                _, mx, _, ml = cv2.minMaxLoc(r)
                if mx < cfg["thr"] or mx - p < cfg["dist"]:
                    break

                dets.append((float(mx), ml[0] * ds + ox, ml[1] * ds + oy,
                             w * ds, h * ds, mid))

                # 抹掉该峰邻域，避免同一只怪被反复计入
                r[max(0, ml[1] - h // 2):ml[1] + h // 2 + 1,
                  max(0, ml[0] - w // 2):ml[0] + w // 2 + 1] = 0.0

    kept = nms(dets)

    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    txt = "\n".join("%d %.6f %.6f %.6f %.6f" % (
        CLASS_MOB, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
        for _, x, y, w, h, _ in kept)
    (out_dir / (fp.stem + ".txt")).write_text(txt, encoding="utf-8")

    if cfg["vis"]:
        for _, x, y, w, h, _ in kept:
            cv2.rectangle(full, (x, y), (x + w, y + h), (0, 255, 0), 2)
        imwrite(Path(cfg["vis_dir"]) / (fp.stem + ".jpg"), full, quality=82)

    return len(dets), len(kept)


# ══════════════════════════════════════════════════════════════
# 统计（质检台用它判断标注质量）
# ══════════════════════════════════════════════════════════════
def collect_stats(labels_dir, frame_dir=None):
    """扫描标注文件算诊断统计。

    界面靠这两组数字判断标注能不能用：
      · 每帧框数 —— 全 0 说明模板根本不匹配；个别帧暴涨说明误检
      · 框尺寸   —— 明显偏大/偏小说明 scale 标定错了，所有标注都得重来
    """
    counts = []
    sides = []          # 框的等效边长（像素）
    W = H = 0

    if frame_dir:
        first = next(iter(sorted(Path(frame_dir).glob("*.png"))), None)
        if first is not None:
            img = imread(first)
            if img is not None:
                H, W = img.shape[:2]

    for f in sorted(Path(labels_dir).glob("*.txt")):
        try:
            lines = [x for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        except Exception:
            continue

        counts.append(len(lines))
        for ln in lines:
            p = ln.split()
            if len(p) >= 5:
                try:
                    bw, bh = float(p[3]), float(p[4])
                except ValueError:
                    continue
                if W and H:
                    sides.append((bw * W * bh * H) ** 0.5)
                else:
                    sides.append((bw * bh) ** 0.5)

    def m(arr, q):
        if not arr:
            return 0
        return float(np.percentile(arr, q))

    return {
        "frames": len(counts),
        "boxes": int(sum(counts)),
        "zero_frames": int(sum(1 for c in counts if c == 0)),
        "max_boxes": int(max(counts)) if counts else 0,
        "median_boxes": m(counts, 50),
        "size_median": m(sides, 50),
        "size_p10": m(sides, 10),
        "size_p90": m(sides, 90),
        "img_size": (W, H),
    }


# ══════════════════════════════════════════════════════════════
# 主入口（GUI / CLI 共用）
# ══════════════════════════════════════════════════════════════
def run_detect(params, ctx=None):
    """自动标注。

    params:
        mobs         怪种 id 列表（用这些做模板）
        sprites      精灵库根目录
        frames       画面目录
        out          标注输出目录
        vis_dir      可视化图目录（vis=True 时用）
        scale        精灵缩放比例
        downscale    降采样倍数
        thresh       匹配阈值
        min_distinct 区分度
        per_mob      每种怪用几帧做模板，0=全部
        max_peaks    每个模板最多取几个峰
        region       "x,y,w,h" 或 None
        vis          是否输出可视化图
        workers      进程数，0=自动
    """
    ctx = ctx or TaskContext()

    frames_dir = Path(params["frames"])
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError("画面目录里没有 png：%s\n请先完成 ② 采集" % frames_dir)

    limit = int(params.get("limit", 0) or 0)
    if limit:
        frames = frames[:limit]

    mobs = [str(m).strip() for m in (params.get("mobs") or []) if str(m).strip()]
    if not mobs:
        raise ValueError("没有指定怪种 —— 请先在 ① 地图 里选一张有怪的地图")

    sprites = params.get("sprites") or ""
    if not Path(sprites).is_dir():
        raise FileNotFoundError("精灵库不存在：%s" % sprites)

    out = Path(params["out"])
    out.mkdir(parents=True, exist_ok=True)

    ds = max(1, int(params.get("downscale", 1)))
    # 键名沿用 per_mob 以兼容已有项目配置，语义是「上限」：
    # 非死亡帧数不超过它就不截断，全量使用。默认 20 足够覆盖绝大多数怪。
    per_mob = int(params.get("per_mob", 20))
    thr = float(params.get("thresh", 0.90))
    dist = float(params.get("min_distinct", 0.06))
    peaks = int(params.get("max_peaks", 4))

    region = params.get("region")
    if isinstance(region, str) and region.strip():
        try:
            region = tuple(int(v) for v in region.split(","))
        except Exception:
            raise ValueError('搜索区域格式应为 "x,y,w,h"，收到: %s' % region)
    else:
        region = None

    vis = bool(params.get("vis", True))
    vis_dir = params.get("vis_dir") or str(out.parent / "vis_map")

    cfg = {
        "sprites": sprites,
        "mobs": mobs,
        "ds": ds,
        "scales": [float(params.get("scale", 1.12))],
        "thr": thr,
        "dist": dist,
        "peaks": peaks,
        "per_mob": per_mob,
        "out": str(out),
        "vis": vis,
        "vis_dir": vis_dir,
        "region": region,
    }

    n_tpl = len(load_templates(Path(sprites), mobs, per_mob))
    if n_tpl == 0:
        raise RuntimeError(
            "模板为空 —— 精灵库里找不到这些怪：\n  %s\n"
            "（检查 config/wz.yaml 的 sprite_dir，以及怪种 ID 是否补足 7 位）"
            % ", ".join(mobs[:8]))

    ctx.log("模板 %d 个（%d 种怪，含镜像）" % (n_tpl, len(mobs)))
    ctx.log("画面 %d 张   尺度 %.3f   降采样 %d" % (len(frames), cfg["scales"][0], ds))
    ctx.log("阈值 %.2f   区分度 %.3f   每模板峰数 %d" % (thr, dist, peaks))
    if region:
        ctx.log("搜索区域 %s" % (region,))
    ctx.log("每帧 %d 次匹配" % n_tpl)

    workers = int(params.get("workers", 0)) or max(1, (os.cpu_count() or 4) - 1)
    ctx.log("并行进程 %d" % workers)
    ctx.log("")

    t0 = time.perf_counter()
    total = 0
    frames_with = 0
    n = len(frames)

    def consume(iterator):
        nonlocal total, frames_with
        for i, r in enumerate(iterator, 1):
            if ctx.canceled():
                return False
            if r:
                _, kept = r
                total += kept
                if kept:
                    frames_with += 1
            if i % 20 == 0 or i == n:
                ctx.progress(i, n, "已检出 %d 框" % total)
                ctx.log("  [%d/%d]  检出 %d  用时 %.0fs"
                        % (i, n, total, time.perf_counter() - t0))
        return True

    try:
        if workers <= 1:
            init_worker(cfg)
            ok = consume(work(str(f)) for f in frames)
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                     initargs=(cfg,)) as ex:
                ok = consume(ex.map(work, [str(f) for f in frames], chunksize=2))
    except Exception as e:
        raise RuntimeError("标注失败: %s: %s" % (type(e).__name__, e))

    dt = time.perf_counter() - t0

    if not ok:
        ctx.log("已取消", "warn")
        return {"summary": "已取消", "frames": n, "detections": total}

    stats = collect_stats(out, frames_dir)

    # 落盘一份，界面的卡片摘要直接读它，不用每次重扫几百个 txt
    try:
        with open(out / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    ctx.log("")
    ctx.log("── 标注完成 ──", "ok")
    ctx.log("  画面 %d 张 / 检出 %d 框 / 有检出 %d 帧（%.0f%%）"
            % (n, total, frames_with, 100.0 * frames_with / max(1, n)))
    ctx.log("  每帧框数: 中位 %.0f，最多 %d，空帧 %d"
            % (stats["median_boxes"], stats["max_boxes"], stats["zero_frames"]))
    if stats["size_median"]:
        ctx.log("  框尺寸:   中位 %.0f px（%.0f ~ %.0f）"
                % (stats["size_median"], stats["size_p10"], stats["size_p90"]))
    ctx.log("  用时 %.1fs（%.0f 帧/秒）" % (dt, n / max(dt, 0.001)))
    ctx.log("")

    # 诊断提示 —— 让用户不必自己看数字找问题
    if frames_with == 0:
        ctx.log("没有任何帧检出目标。可能是：模板与画面姿态差异过大、"
                "scale 不对、或搜索区域把目标排除了", "warn")
    elif stats["zero_frames"] > n * 0.5:
        ctx.log("过半帧没有检出（%d/%d）—— 检查 scale 是否标定正确"
                % (stats["zero_frames"], n), "warn")

    summary = "%d 帧 / %d 框 / 有检出 %.0f%%" % (n, total, 100.0 * frames_with / max(1, n))

    return {
        "frames": n,
        "detections": total,
        "frames_with_det": frames_with,
        "stats": stats,
        "seconds": dt,
        "out_dir": str(out),
        "vis_dir": vis_dir if vis else "",
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mobs", required=True)
    ap.add_argument("--sprites", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="datasets/labels_map")
    ap.add_argument("--vis-dir", default=None)
    ap.add_argument("--scale", type=float, default=1.12)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--thresh", type=float, default=0.90)
    ap.add_argument("--min-distinct", type=float, default=0.06)
    ap.add_argument("--per-mob", type=int, default=20,
                    help="每只怪的模板帧**上限**（默认 20）。"
                         "非死亡帧数不超过它时全部使用，不截断")
    ap.add_argument("--max-peaks", type=int, default=4)
    ap.add_argument("--region", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--vis", action="store_true")
    a = ap.parse_args()

    params = {
        "mobs": [m.strip() for m in a.mobs.split(",") if m.strip()],
        "sprites": a.sprites,
        "frames": a.frames,
        "out": a.out,
        "vis_dir": a.vis_dir,
        "scale": a.scale,
        "downscale": a.downscale,
        "thresh": a.thresh,
        "min_distinct": a.min_distinct,
        "per_mob": a.per_mob,
        "max_peaks": a.max_peaks,
        "region": a.region,
        "vis": a.vis,
        "workers": a.workers,
    }

    try:
        run_detect(params, ConsoleContext())
    except Exception as e:
        print("[detect] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
