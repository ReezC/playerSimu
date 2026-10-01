# -*- coding: utf-8 -*-
"""自动粗标提案器（M2b 第一步）：经典 CV 在增强图上提形状候选框 → YOLO 格式。

流程（用户 2026-09-29 定的管线 ✓）：自动粗标 → 人工精标 → 训练 → 稳定检测器 → 追踪。
本脚本只做"粗标提案"：宁多勿漏（框可以删，漏了要人手补 ✗）。

用法：
    python -m tools.lie_label_propose                # 40 个 cycle × 3 帧 起步
    python -m tools.lie_label_propose --per-cycle 5  # 每 cycle 抽 5 帧
提案输出：datasets/lie/label_round1/{images/*.png, labels/*.txt}（YOLO 格式 ✓）
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.lie_replay import load_cycles  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "lie" / "label_round1"


def propose(img):
    """增强 → 阈值 → 连通域 ⇒ [(cx, cy, w, h)]（像素坐标 ✓）。宁多勿漏 ✓。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hp = cv2.subtract(g, cv2.GaussianBlur(g, (21, 21), 0))   # 高通：淡轮廓凸显 ✓
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)
    _, bw = cv2.threshold(hp, 42, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          np.ones((9, 9), np.uint8), iterations=2)
    out = []
    res = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = res[0] if isinstance(res, tuple) else res
    for c in cs:
        arr = np.asarray(c, dtype=np.float32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(arr)
        a = cv2.contourArea(arr)
        if not (350 <= a <= 30000):            # 面积过滤（太大 = 全图噪声 ✗）
            continue
        if max(w, h) / max(1, min(w, h)) > 4.5:  # 细长条多为纹理伪影 ✗
            continue
        out.append((x + w / 2.0, y + h / 2.0, w, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cycle", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=40)
    args = ap.parse_args()
    cyc = load_cycles()
    ids = sorted(cyc)[::max(1, len(cyc) // args.cycles)][:args.cycles]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    total = 0
    for cid in ids:
        frames = cyc[cid]
        picks = [frames[int(len(frames) * f)]
                 for f in np.linspace(0.3, 0.9, args.per_cycle)]
        for idx, ts, p in picks:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes = propose(img)
            stem = "c%s_f%03d" % (cid, idx)
            cv2.imwrite(str(OUT / "images" / ("%s.png" % stem)), img)
            with open(OUT / "labels" / ("%s.txt" % stem), "w",
                      encoding="utf-8") as f:
                for cx, cy, bw_, bh in boxes:
                    f.write("0 %.6f %.6f %.6f %.6f\n"
                            % (cx / w, cy / h, bw_ / w, bh / h))
            total += len(boxes)
    print("cycles=%d frames=%d boxes=%d → %s"
          % (len(ids), len(ids) * args.per_cycle, total, OUT))


if __name__ == "__main__":
    main()
