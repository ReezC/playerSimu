"""裁剪探针条区域并放大，红点标出 B 机当前的采样位置。

用法:
    python -m tools.probe_crop
"""

import argparse

import cv2

from link import PyAVSource
from tools.config import get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--y1", type=int, default=50)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--x1", type=int, default=1000)
    ap.add_argument("--out", default="data/probe_crop.png")
    args = ap.parse_args()

    src = PyAVSource(get("stream", "url"))
    src.open()
    print("recv size:", src.size)

    f = None
    for _ in range(args.skip):
        f = src.read()
    src.close()
    if f is None:
        print("no frame")
        return 1

    img = f.image
    h, w = img.shape[:2]
    y0, y1 = max(0, args.y0), min(h, args.y1)
    x0, x1 = max(0, args.x0), min(w, args.x1)
    crop = img[y0:y1, x0:x1]

    big = cv2.resize(crop, None, fx=2.0, fy=6.0, interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)

    px = get("probe", "x", 100)
    py = get("probe", "y", 8)
    cell = get("probe", "cell", 16)
    gap = get("probe", "gap", 2)
    bits = get("probe", "bits", 40)
    n = 2 + bits

    for i in range(n):
        cx = px + i * (cell + gap) + cell // 2
        cy = py + cell // 2
        X = int((cx - x0) * 2.0)
        Y = int((cy - y0) * 6.0)
        if 0 <= X < bgr.shape[1] and 0 <= Y < bgr.shape[0]:
            cv2.circle(bgr, (X, Y), 4, (0, 0, 255), -1)

    cv2.imwrite(args.out, bgr)
    print("saved", args.out, bgr.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
