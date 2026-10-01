# -*- coding: utf-8 -*-
"""M2b：用检测器对全部 cycle 帧离线批量推理 ⇒ `datasets/lie/dets_v2.json`。

用法：
    python -m tools.lie_detect_batch [--weights PATH] [--conf 0.25]

输出：{帧名stem: [[cls, cx, cy, w, h], ...]}（全分辨率像素、中心点口径 ✓）。
独立 CLI 进程（无 Qt ✓ 无 DLL 冲突 —— "先 Qt 后 torch"进程内推理必炸 ✗）；
stream 逐帧出，内存不爆 ✓。产物给 `tools/lie_replay --dets` 用 ✓。
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "lie" / "dets_v2.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str,
                    default=str(ROOT / "datasets" / "runs" / "detect" / "v2"
                                / "weights" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    m = YOLO(args.weights)
    out = {}
    n = 0
    for r in m.predict(source=str(ROOT / "datasets" / "liedetectorphotos"),
                       conf=args.conf, imgsz=960, stream=True, verbose=False):
        boxes = [[int(b.cls[0])] + [float(v) for v in b.xywh[0]]
                 for b in (r.boxes or [])]
        out[Path(r.path).stem] = boxes
        n += 1
        if n % 2000 == 0:
            print("processed", n, flush=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print("DETS_DONE", n, "->", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
