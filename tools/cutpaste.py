"""Cut-Paste 合成训练集：把 WZ 提取的怪物 sprite 贴到真实推流背景上，自动产出 YOLO 标注。

为什么用这个方案：
    - 前景（sprite）来自 WZ，与官方客户端同源，外观一致
    - 背景来自真实推流画面，域差距极小
    - bbox 由贴图位置直接算出，标注零成本、零误差
    - 全自动、可无人值守生成几十万张

尺度依据：
    实测（tools/calibrate_scale.py）在 1920x1080 推流画面坐标系下
    游戏内 sprite 显示尺寸 = WZ 原生像素 x 1.12。

用法：
    python -m tools.cutpaste --bg data/screens --sprites datasets/sprites/mob \
        --out datasets/synth --count 500 --per 2,5
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

CLASS_MOB = 1


def load_sprites(root: Path, limit_frames: int, verbose=True):
    """载入 sprite，裁到 alpha 包围盒，并把 origin 换算到裁剪后的坐标系。"""
    out = []
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]

    for d in dirs:
        meta = {}
        mt = d / "meta.tsv"
        if mt.exists():
            for line in mt.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        meta[parts[0]] = (int(parts[1]), int(parts[2]))
                    except ValueError:
                        pass

        for f in sorted(d.glob("*.png")):
            if limit_frames and len(out) >= limit_frames:
                break

            t = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if t is None or t.ndim != 3 or t.shape[2] != 4:
                continue

            alpha = t[:, :, 3]
            ys, xs = np.nonzero(alpha > 128)
            if len(xs) < 200:
                continue

            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            crop = t[y0:y1, x0:x1]

            if crop.shape[0] < 16 or crop.shape[1] < 16:
                continue

            ox, oy = meta.get(f.name, (crop.shape[1] // 2, crop.shape[0]))
            out.append((d.name,
                        crop[:, :, :3].copy(),
                        crop[:, :, 3].copy(),
                        ox - x0, oy - y0))

        if limit_frames and len(out) >= limit_frames:
            break

    if verbose:
        print(f"[synth] 载入 sprite {len(out)} 帧")
    return out


def paste_one(bg, sprite, target_x, target_y, scale, flip):
    """把 sprite 以 origin 为锚点贴到 (target_x, target_y)，返回 bbox 或 None。"""
    _, bgr, alpha, ox, oy = sprite
    h, w = bgr.shape[:2]

    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    b = cv2.resize(bgr, (nw, nh), interpolation=interp)
    a = cv2.resize(alpha, (nw, nh), interpolation=interp)
    if flip:
        b = cv2.flip(b, 1)
        a = cv2.flip(a, 1)
        ox = w - ox

    # origin 是锚点（一般接近脚底），把它对齐到 (target_x, target_y)
    px = int(round(target_x - ox * scale))
    py = int(round(target_y - oy * scale))

    H, W = bg.shape[:2]
    # 计算与画布的交集
    sx0, sy0 = max(0, -px), max(0, -py)
    dx0, dy0 = max(0, px), max(0, py)
    cw = min(nw - sx0, W - dx0)
    ch = min(nh - sy0, H - dy0)
    if cw <= 4 or ch <= 4:
        return None

    b_crop = b[sy0:sy0 + ch, sx0:sx0 + cw]
    a_crop = (a[sy0:sy0 + ch, sx0:sx0 + cw].astype(np.float32) / 255.0)[:, :, None]
    roi = bg[dy0:dy0 + ch, dx0:dx0 + cw].astype(np.float32)

    bg[dy0:dy0 + ch, dx0:dx0 + cw] = (b_crop.astype(np.float32) * a_crop +
                                      roi * (1.0 - a_crop)).astype(np.uint8)
    return dx0, dy0, cw, ch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", type=str, required=True, help="背景图目录")
    ap.add_argument("--sprites", type=str, required=True)
    ap.add_argument("--out", type=str, default="datasets/synth")
    ap.add_argument("--count", type=int, default=200, help="生成多少张")
    ap.add_argument("--per", type=str, default="2,5", help="每张图贴几只怪 min,max")
    ap.add_argument("--scale", type=float, default=1.12, help="sprite 缩放（实测值）")
    ap.add_argument("--scale-jitter", type=float, default=0.02)
    ap.add_argument("--bg-crop", type=str, default=None,
                    help="背景裁剪 x,y,w,h —— 用来去掉桌面/OBS 等非游戏区域")
    ap.add_argument("--max-frames", type=int, default=0, help="最多载入多少 sprite 帧")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    bg_paths = sorted(Path(args.bg).glob("*.png")) + sorted(Path(args.bg).glob("*.jpg"))
    if not bg_paths:
        print(f"[synth] {args.bg} 里没有图片")
        return 1

    sprites = load_sprites(Path(args.sprites), args.max_frames)
    if not sprites:
        print("[synth] 没有可用 sprite")
        return 1

    lo, hi = (int(v) for v in args.per.split(","))

    crop = None
    if args.bg_crop:
        crop = tuple(int(v) for v in args.bg_crop.split(","))

    img_dir = Path(args.out) / "images"
    lbl_dir = Path(args.out) / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for i in range(args.count):
        bg = cv2.imread(str(random.choice(bg_paths)))
        if bg is None:
            continue

        if crop:
            x, y, w, h = crop
            bg = bg[y:y + h, x:x + w]
            if bg.size == 0:
                continue

        H, W = bg.shape[:2]
        lines = []

        for _ in range(random.randint(lo, hi)):
            sp = random.choice(sprites)
            sc = args.scale * random.uniform(1 - args.scale_jitter,
                                             1 + args.scale_jitter)

            # 落点限制在画面下方的地面带 —— 怪物不会浮在天上
            tx = random.uniform(W * 0.05, W * 0.95)
            ty = random.uniform(H * 0.55, H * 0.98)

            box = paste_one(bg, sp, tx, ty, sc, random.random() < 0.5)
            if box is None:
                continue

            bx, by, bw, bh = box
            lines.append(f"{CLASS_MOB} {(bx + bw / 2) / W:.6f} "
                         f"{(by + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f}")

        if not lines:
            continue

        stem = f"synth_{i:06d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), bg,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        ok += 1

        if (ok % 50 == 0) or ok == args.count:
            print(f"[synth] {ok}/{args.count}", flush=True)

    print(f"[synth] 完成 {ok} 张 -> {img_dir}")
    print(f"[synth] 标注   -> {lbl_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
