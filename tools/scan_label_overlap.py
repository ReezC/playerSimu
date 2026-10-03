#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 YOLO 标注文件，找出「同帧同类多个高重叠框」—— 可能是多框标注的大怪。

背景：一个目标被小目标挡住时，如果用多个矩形标大目标（排除小目标干扰），
训练时检测器会把它当多个目标 ⇒ 推理时一个大怪输出多个框。
标准做法是「一个目标一个框」（含被遮挡部分）。

用法：
    python -m tools.scan_label_overlap projects/龙族打猎场/labels
    python -m tools.scan_label_overlap projects/龙族打猎场/labels --iou 0.3
    python -m tools.scan_label_overlap projects/龙族打猎场/labels --verbose

输出：
    统计：总帧数、有高重叠的帧数、高重叠组数
    详细：每帧的类别、IoU（--verbose）
"""
import argparse
import sys
from pathlib import Path


def iou(a, b):
    """两个归一化矩形 (x,y,w,h) 的 IoU。"""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def scan_file(path):
    """解析一个 txt → [(class, x, y, w, h), ...]。"""
    out = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    c = int(parts[0])
                    x, y, w, h = (float(v) for v in parts[1:5])
                    out.append((c, x, y, w, h))
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")        # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="扫 YOLO 标注找「同帧同类高重叠框」")
    ap.add_argument("dir", help="labels 目录")
    ap.add_argument("--iou", type=float, default=0.3,
                    help="IoU 阈值（默认 0.3）—— 高于这个算「高重叠」")
    ap.add_argument("--verbose", action="store_true", help="列出每个高重叠组的详情")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print("目录不存在：%s" % root)
        return 1

    txts = sorted(root.glob("*.txt"))
    total = len(txts)
    overlap_frames = 0
    overlap_groups = 0
    details = []

    for txt in txts:
        boxes = scan_file(txt)
        by_class = {}
        for c, x, y, w, h in boxes:
            by_class.setdefault(c, []).append((x, y, w, h))
        frame_hit = False
        for c, cls_boxes in by_class.items():
            if len(cls_boxes) < 2:
                continue
            for i in range(len(cls_boxes)):
                for j in range(i + 1, len(cls_boxes)):
                    v = iou(cls_boxes[i], cls_boxes[j])
                    if v >= args.iou:
                        overlap_groups += 1
                        frame_hit = True
                        details.append((txt.stem, c, v, cls_boxes[i], cls_boxes[j]))
        if frame_hit:
            overlap_frames += 1

    print("=" * 60)
    print("扫描目录：%s" % root)
    print("IoU 阈值：%.2f" % args.iou)
    print("总帧数：%d" % total)
    print("有高重叠的帧数：%d（%.1f%%）" % (
        overlap_frames, 100 * overlap_frames / total if total else 0))
    print("高重叠组数：%d" % overlap_groups)
    print("=" * 60)
    if args.verbose and details:
        print("详情（帧 / 类别 / IoU / 框1 / 框2）：")
        for stem, c, v, b1, b2 in details:
            print("  %-16s cls=%d  IoU=%.2f  框1=(%.3f,%.3f,%.3f,%.3f)  框2=(%.3f,%.3f,%.3f,%.3f)"
                  % (stem, c, v, *b1, *b2))
    elif details:
        print("（加 --verbose 看详情）")
    if not details:
        print("✓ 没发现同帧同类高重叠框——标注看起来是「一个目标一个框」✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
