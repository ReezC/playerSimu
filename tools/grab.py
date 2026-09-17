"""抓一帧存成 PNG（B 机），便于肉眼核对探针区域。

用法:
    python -m tools.grab
"""

import argparse

import cv2

from link import PyAVSource
from tools.config import get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/frame.png")
    ap.add_argument("--url", default=None)
    ap.add_argument("--skip", type=int, default=120, help="跳过前 N 帧等流稳定")
    args = ap.parse_args()

    src = PyAVSource(args.url or get("stream", "url"))
    src.open()
    print("size:", src.size, "fps:", src.fps)

    f = None
    for _ in range(args.skip):
        f = src.read()
        if f is None:
            break
    src.close()

    if f is None:
        print("未收到帧")
        return 1

    bgr = cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR)
    px = get("probe", "x", 100)
    py = get("probe", "y", 8)
    cell = get("probe", "cell", 16)
    gap = get("probe", "gap", 2)
    bits = get("probe", "bits", 40)

    # 取整：gap 是 2.25（浮点），不取整 cv2 不接受浮点坐标
    bx1 = int(round(px + (2 + bits) * (cell + gap) + 6))
    by2 = int(round(py + cell + 10))
    cv2.rectangle(bgr, (int(round(px)) - 6, int(round(py)) - 6),
                  (bx1, by2), (0, 255, 0), 2)
    cv2.imwrite(args.out, bgr)
    print("已保存", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
