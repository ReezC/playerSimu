"""【B 机运行】生成带时间码探针的测试片段，用于离线自测收流/解码/探针链路。

在 A 机还没推流时，用它验证 FileSource + probe_codec + probe_recv 统计逻辑是否自洽。

用法：
    python -m tools.make_testclip
    python -m tools.probe_recv --file data/recordings/testclip.mp4 --offset 0
"""

import argparse
import sys
import time

import av
import cv2
import numpy as np

from tools.config import get, record_dir
from tools.probe_codec import encode_bits, now_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="testclip.mp4")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    w = args.w or get("stream", "width", 1280)
    h = args.h or get("stream", "height", 720)
    px = get("probe", "x", 100)
    py = get("probe", "y", 8)
    cell = get("probe", "cell", 16)
    gap = get("probe", "gap", 2)
    bits = get("probe", "bits", 40)

    out = record_dir() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(out), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "veryfast"}

    print(f"[make_testclip] -> {out}  {w}x{h}@{args.fps}  frames={args.frames}")

    for i in range(args.frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (24, 24, 24)  # 深灰背景，便于观察压缩伪影

        # 模拟"怪物"：从左向右移动的绿色方块
        cx = int(50 + (i / max(1, args.frames - 1)) * (w - 200))
        cv2.rectangle(img, (cx, h // 2), (cx + 80, h // 2 + 80), (0, 220, 0), -1)

        # 模拟"角色"：底部固定蓝色方块
        cv2.rectangle(img, (w // 2 - 40, h - 140), (w // 2 + 40, h - 60), (220, 120, 0), -1)

        # 屏幕时间码探针（编码当前毫秒时间戳）
        ts_ms = now_ms()
        for k, v in enumerate(encode_bits(ts_ms, bits)):
            rx = px + k * (cell + gap)
            cv2.rectangle(img, (rx, py), (rx + cell, py + cell),
                          (255, 255, 255) if v else (0, 0, 0), -1)

        vf = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"[make_testclip] 完成，接下来跑：")
    print(f"  python -m tools.probe_recv --file {out.relative_to(out.parents[2])} --offset 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
