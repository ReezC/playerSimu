"""用 sprite 模板匹配自动标注怪物，输出 YOLO 格式标注。

原理：
    尺度已标定为 1.12（1920x1080 推流画面坐标系），搜索范围可收窄到 ±0.02。
    对每张画面，用 sprite 库逐帧匹配，高分命中即为伪标注，再做 NMS 去重。

局限（必须知道）：
    - 只能匹配「与模板姿态一致」的怪；怪物被大幅遮挡、有发光/半透明特效时会漏
    - 因此产出的是「伪标注」，需要用 YOLO 迭代自训练来补全，并抽样人工校验

用法：
    python -m tools.auto_label --frames data/frames --sprites datasets/sprites/mob \
        --out datasets/labels --limit-mobs 60 --per-mob 1 --thresh 0.90
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CLASS_MOB = 1     # 与 DatasetExporter 的类目顺序保持一致: player=0 mob=1 drop=2 npc=3


def load_templates(root: Path, limit_mobs: int, per_mob: int, verbose=True):
    """载入模板，裁到 alpha 包围盒。优先 stand/move 帧（最常见可见姿态）。"""
    out = []
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]

    for d in dirs:
        frames = sorted(d.glob("*.png"))
        if not frames:
            continue

        # fly 也要在优先队列里：飞的怪常常只有 fly，没有 stand
        pref = [f for f in frames
                if f.name.startswith(("stand_", "fly_", "move_"))] or frames
        for f in pref[:per_mob]:
            t = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if t is None or t.ndim != 3 or t.shape[2] != 4:
                continue

            bgr, alpha = t[:, :, :3], t[:, :, 3]
            ys, xs = np.nonzero(alpha > 128)
            if len(xs) < 200:
                continue

            bgr = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

            if bgr.shape[0] < 20 or bgr.shape[1] < 20:
                continue

            out.append((d.name, f.name, bgr, alpha))

        if limit_mobs and len(out) >= limit_mobs:
            break

    if verbose:
        print(f"[label] 模板 {len(out)} 个（来自 {min(len(dirs), limit_mobs or len(dirs))} 只怪）")
    return out


def nms(dets, iou_thresh=0.35):
    """dets: [(score, x, y, w, h, tag)]，按分数贪心抑制重叠框。"""
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: -d[0])
    keep = []

    for d in dets:
        _, x, y, w, h, _ = d
        ok = True
        for k in keep:
            _, kx, ky, kw, kh, _ = k
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            inter = ix * iy
            union = w * h + kw * kh - inter
            if union > 0 and inter / union > iou_thresh:
                ok = False
                break
        if ok:
            keep.append(d)

    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=str, required=True)
    ap.add_argument("--sprites", type=str, required=True)
    ap.add_argument("--out", type=str, default="datasets/labels")
    ap.add_argument("--limit-mobs", type=int, default=0, help="用多少只怪作模板，0=全部")
    ap.add_argument("--per-mob", type=int, default=1, help="每只怪用几帧作模板")
    ap.add_argument("--scale", type=float, default=1.12)
    ap.add_argument("--scale-span", type=float, default=0.02)
    ap.add_argument("--thresh", type=float, default=0.90)
    ap.add_argument("--min-distinct", type=float, default=0.06,
                    help="峰值须高于响应图 99.5 分位这么多。实测真匹配≈0.15、噪声≈0.03")
    ap.add_argument("--max-peaks", type=int, default=3,
                    help="每个 模板x尺度 组合最多取几个峰")
    ap.add_argument("--region", type=str, default=None, help="x,y,w,h 搜索区域")
    ap.add_argument("--vis", action="store_true", help="同时输出可视化图")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    args = ap.parse_args()

    frame_paths = sorted(Path(args.frames).glob("*.png"))
    if args.limit:
        frame_paths = frame_paths[:args.limit]
    if not frame_paths:
        print(f"[label] {args.frames} 里没有 png")
        return 1

    templates = load_templates(Path(args.sprites), args.limit_mobs, args.per_mob)
    if not templates:
        print("[label] 没有可用模板")
        return 1

    scales = [round(args.scale - args.scale_span, 3),
              round(args.scale, 3),
              round(args.scale + args.scale_span, 3)]

    ox, oy = 0, 0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[label] 尺度 {scales}  阈值 {args.thresh}  "
          f"每图 {len(templates) * len(scales)} 次匹配")

    total_det = 0
    t_all = time.perf_counter()

    for fi, fp in enumerate(frame_paths):
        full = cv2.imread(str(fp))
        if full is None:
            continue

        H, W = full.shape[:2]
        img = full
        if args.region:
            try:
                rx, ry, rw, rh = (int(v) for v in args.region.split(","))
                img = full[ry:ry + rh, rx:rx + rw]
                ox, oy = rx, ry
            except Exception:
                pass

        dets = []
        for mob_id, fname, bgr, alpha in templates:
            for sc in scales:
                h = int(round(bgr.shape[0] * sc))
                w = int(round(bgr.shape[1] * sc))
                if h >= img.shape[0] or w >= img.shape[1]:
                    continue

                interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR
                t = cv2.resize(bgr, (w, h), interpolation=interp)
                m = (cv2.resize(alpha, (w, h), interpolation=interp) > 128)
                m = m.astype(np.uint8) * 255

                try:
                    res = cv2.matchTemplate(img, t, cv2.TM_CCORR_NORMED, mask=m)
                except Exception:
                    continue

                # 同 detect_mobs：分母趋 0 时 OpenCV 会给 FLT_MAX（有限值，
                # nan_to_num 不管），不掐掉的话黑边会给出"满分"假框。
                res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
                res[res > 1.0] = 0.0
                if res.max() < args.thresh:
                    continue

                # 区分度判据：峰值必须显著高于该响应图的背景水平。
                # 只用绝对分数会在画面各处产生海量假匹配（实测单图两万个候选）。
                p995 = float(np.percentile(res, 99.5))

                work = res.copy()
                for _ in range(args.max_peaks):
                    _, mx, _, ml = cv2.minMaxLoc(work)
                    if mx < args.thresh or (mx - p995) < args.min_distinct:
                        break

                    dets.append((float(mx), int(ml[0]) + ox, int(ml[1]) + oy,
                                 w, h, mob_id))

                    # 抹掉该峰邻域，避免同一只怪被反复计入
                    x0 = max(0, ml[0] - w // 2)
                    y0 = max(0, ml[1] - h // 2)
                    x1 = min(work.shape[1], ml[0] + w // 2 + 1)
                    y1 = min(work.shape[0], ml[1] + h // 2 + 1)
                    work[y0:y1, x0:x1] = 0.0

        kept = nms(dets)
        total_det += len(kept)

        # YOLO 格式：cls cx cy w h（归一化到整幅图）
        lines = []
        for score, x, y, w, h, _ in kept:
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            lines.append(f"{CLASS_MOB} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")

        stem = fp.stem
        (out_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        print(f"  [{fi + 1}/{len(frame_paths)}] {fp.name}: "
              f"候选 {len(dets)} -> NMS 后 {len(kept)}", flush=True)

        if args.vis:
            vis = full.copy()
            for score, x, y, w, h, mob_id in kept:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis, f"{mob_id}", (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            vis_dir = out_dir.parent / "vis"
            vis_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_dir / f"{stem}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])

    dt = time.perf_counter() - t_all
    print(f"[label] 完成: {len(frame_paths)} 张 / {total_det} 个标注 / {dt:.1f}s")
    print(f"[label] 标注 -> {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
