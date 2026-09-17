"""探针区域诊断 v2：扫描行找出方块真实边界，推断是否发生缩放错位。

用法:
    python -m tools.probe_dump
"""

import argparse

import cv2

from link import PyAVSource
from tools.config import get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--url", default=None)
    args = ap.parse_args()

    src = PyAVSource(args.url or get("stream", "url"))
    src.open()
    print("frame size:", src.size)

    f = None
    for _ in range(args.skip):
        f = src.read()
        if f is None:
            break
    src.close()
    if f is None:
        print("no frame")
        return 1

    gray = cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    px = get("probe", "x", 100)
    py = get("probe", "y", 8)
    cell = get("probe", "cell", 16)
    gap = get("probe", "gap", 2)
    bits = get("probe", "bits", 40)
    n = 2 + bits

    cy = py + cell // 2
    row = gray[cy, :]
    b = (row > 128).astype(int)

    edges = [x for x in range(1, w) if b[x] != b[x - 1]]
    print("\n[scan row y=%d]" % cy)
    print("edges:", edges[:40])
    if len(edges) > 3:
        diffs = [edges[i + 1] - edges[i] for i in range(min(len(edges) - 1, 30))]
        print("edge gaps:", diffs)

    # 用配置的坐标采样
    half = max(1, cell // 4)
    vals, bits_out = [], []
    for i in range(n):
        cx = px + i * (cell + gap) + cell // 2
        patch = gray[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        v = int(patch.mean())
        vals.append(v)
        bits_out.append("1" if v > 128 else "0")
    print("\n[configured sampling] x=%d y=%d cell=%d gap=%d" % (px, py, cell, gap))
    print("vals:", vals)
    print("bits:", "".join(bits_out))
    print("expect start marker '10', got '%s'" % "".join(bits_out[:2]))

    # 试算：若起点其实是 px0、周期是 p，看哪个组合能让第 2 个 bit 为 0
    print("\n[试算缩放]")
    for p in (18, 12, 9, 24, 36):
        for x0 in range(0, 200, 2):
            pat = []
            for i in range(6):
                cx = x0 + i * p + p // 4
                if cx < w:
                    pat.append("1" if gray[cy, cx] > 128 else "0")
            s = "".join(pat)
            if s.startswith("10") and s != "10" + "0" * (len(s) - 2):
                print("  period=%2d x0=%3d -> %s" % (p, x0, s))
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
