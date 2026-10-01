# -*- coding: utf-8 -*-
"""测谎演示/实机用的**常驻检测子进程**：stdin 收 JPEG 帧、stdout 吐检测 JSON。

为什么必须是独立进程（2026-09-30 ✓ 既有结论）：**"先 Qt 后 torch"的进程里
c10.dll 初始化必炸**（WinError 1114 ✗）—— 工作台/演示窗这类 Qt 进程里**不能**
进程内 import ultralytics。常驻（不是每帧起一个 ✗ 那每次要 ~2s 启动）：
    · 帧协议：`4 字节大端长度 + JPEG 字节` 循环；`长度 == 0` 表示收工 ✓；
    · 输出协议：每帧一行 JSON `[[cls, cx, cy, w, h, conf], ...]`（**输入帧像素坐标** ✓
      置信度 2026-09-30 加 ✓ 演示窗要写"类别名 + 置信度" ✓）。

用法（一般由 `tools/lie_demo.py` 拉起，不直接跑）：
    python -m tools.lie_dets_worker --weights <best.pt> [--conf 0.25] [--imgsz 960]
    python -m tools.lie_dets_worker --names        # 打印类别名 JSON 后退出（一次性 ✓）
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str,
                    default=str(ROOT / "datasets" / "runs" / "detect" / "v2"
                                / "weights" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--names", action="store_true",
                    help="只打印模型的类别名（JSON 数组 ✓）后退出 ✓")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    if args.names:
        nm = getattr(model, "names", None) or {}
        if isinstance(nm, dict):
            out = [str(nm[k]) for k in sorted(nm, key=lambda x: int(x))]
        else:
            out = [str(x) for x in nm]
        print(json.dumps(out, ensure_ascii=False))
        return 0
    inp = sys.stdin.buffer
    out = sys.stdout
    while True:
        head = inp.read(4)
        if len(head) < 4:
            break
        n = struct.unpack(">I", head)[0]
        if n == 0:
            break
        buf = b""
        while len(buf) < n:                      # 读满一帧（管道会分片 ✓）
            chunk = inp.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            out.write("[]\n")
            out.flush()
            continue
        res = model.predict(img, conf=args.conf, imgsz=args.imgsz,
                            verbose=False)[0]
        boxes = [[int(b.cls[0])] + [float(v) for v in b.xywh[0]]
                 + [float(b.conf[0])] for b in (res.boxes or [])]
        out.write(json.dumps(boxes) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
