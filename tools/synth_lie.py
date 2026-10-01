"""**有真值的合成测谎序列**（用户 2026-09-30 决策 ✓ 换掉"自洽命中率"当判据 ✓）。

为什么需要它：`tools/lie_replay.py` 的 `hit` = "相邻两帧输出距离 ≤ 目标半径" ✗ = **自洽指标**
（站着不动也能 1.0 ✗✗）⇒ 它证明不了追得对 ✓。合成台给**逐帧真值**，才能算**位置误差** ✓。

按用户给的 7 条事实造场景（与真素材同构 ✓）：
  · 背景 = **周期性纹理**（同真素材 ✓ 会让模板匹配跳格 ⇒ 逼出好的实现 ✓）；
  · **假目标群**：贴住背景、只随群体**平移** ✓（事实 2/3）；
  · **真目标**：随群体平移 + **自身匀速** + **每帧旋转** ✓（事实 2/3）；
  · **合并/分离**：让真目标在若干帧里**压住**一个假目标 ✓（事实 5/7）—— 框由"亮块的连通域"
    现算 ✓ ⇒ 重叠时**自然**并成一个大框 ✓（正是素材里的现象 ✓）；
  · **漏检**：可指定帧把某块的框丢掉 ✓（事实 4）。

对外：`make_case(seed=.., n=..) -> dict(frames, boxes, truth, tvec)`。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

FIG = 96                  # 图形边长（px ✓ 与素材同量级 ✓）
BG_LO, BG_HI = 30, 70     # 背景灰度
FAKE_LO, FAKE_HI = 85, 135     # **假目标**：中灰纹理（检测器看得见 ✓ 但**不够白**）
TGT_LO, TGT_HI = 190, 235      # **真目标**：亮白纹理 ⇒ 开局"唯一的白块" ✓（真游戏前提 ✓
                               #   —— 第一版 13 个图形一样亮 ✗ ⇒ 建档根本认不出真目标 ✗
                               #   实测：那等于用一个"目标不可辨识"的场景去量误差 ✗ 不公平 ✓）
BIN_THR = 100             # "亮块"阈值（算框用 ✓ = 模拟检测器的连通域 ✓ 两者都过 ✓）
TILE = 60                 # 背景 tile 边长（周期性 ✓ 跳格陷阱 ✓）


def _shape(size=FIG, seed=0):
    """图形模板（亮的实心块 ✓ 带点自转可辨的结构 ✓）。"""
    rng = np.random.RandomState(seed)
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), int(size * 0.40), 255, -1)
    for k in range(5):
        a = 2 * math.pi * k / 5 + rng.rand() * 0.6
        cv2.circle(m, (int(size / 2 + math.cos(a) * size * 0.26),
                       int(size / 2 + math.sin(a) * size * 0.26)),
                   int(size * 0.17), 255, -1)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    return m


def _bg(w, h, seed=3):
    rng = np.random.RandomState(seed)
    t = (rng.rand(TILE, TILE) * (BG_HI - BG_LO) + BG_LO).astype(np.float32)
    t = cv2.GaussianBlur(t, (3, 3), 0)
    g = np.tile(t, (h // TILE + 2, w // TILE + 2))[:h + TILE, :w + TILE]
    return g


def _fill(mask, lo, hi, seed):
    """形状掩码 ⇒ **纹理填充**（灰阶在 `[lo, hi]` ✓ 有纹理才做得了模板匹配 ✓）。"""
    rng = np.random.RandomState(seed)
    n = cv2.GaussianBlur(rng.rand(*mask.shape).astype(np.float32), (9, 9), 0)
    n = (n - n.min()) / max(1e-6, n.max() - n.min())
    return (mask.astype(np.float32) / 255.0) * (lo + n * (hi - lo))


def _paste(canvas, mask, cx, cy, ang, fill):
    """把图形（可旋转、带纹理）贴到画布上（**取亮者**：重叠处亮的那只胜 ✓ 与素材同构 ✓）。"""
    h, w = canvas.shape
    size = mask.shape[0]
    rot = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
    m = mask if ang == 0 else cv2.warpAffine(mask, rot, (size, size))
    f = fill if ang == 0 else cv2.warpAffine(fill, rot, (size, size))
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    x1, y1 = x0 + size, y0 + size
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return None
    sub = f[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    msk = (m[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)] > 0)
    cur = canvas[y0c:y1c, x0c:x1c]
    canvas[y0c:y1c, x0c:x1c] = np.where(msk, np.maximum(cur, sub), cur)
    return (x0c, y0c, x1c - x0c, y1c - y0c)


def make_case(seed=1, n=40, w=750, h=500, group_amp=(110.0, 70.0), group_per=(28.0, 23.0),
              tgt_v=(9.0, 6.0), spin=7.0, n_fake=12, miss_at=(), dt=0.18):
    """⇒ `{"frames": [bgr...], "boxes": [[(cx,cy,w,h)...]...], "truth": [(x,y)...],
            "tvec": [(dx,dy)...], "dt": dt}`"""
    rng = np.random.RandomState(seed)
    shape = _shape(seed=seed)
    amp = np.asarray(group_amp, np.float32)
    per = np.asarray(group_per, np.float32)
    tv = np.asarray(tgt_v, np.float32)
    # 群体 = **振荡平移**（平滑 ✓ 每帧 ~20px 与素材同量级 ✓，且**零均值** ⇒ 画面里始终有
    #   内容 ✓、真目标不会跑出视口 ✗ 第一版用单调平移 ⇒ 真值跑到 x=1190 出画 ✗ 实测踩过）
    pad = int(amp.max() * 2.2) + 2 * TILE
    base = pad // 2
    bg = _bg(w + pad, h + pad, seed=seed + 100)
    # 假目标：铺在背景上（**跟着背景一起走** ✓ 用户事实 2/3 ✓）；填充纹理**固定**（好匹配 ✓）
    fakes, fills = [], []
    for i in range(n_fake):
        fx = 70 + (i % 4) * (w - 140) / 3.0 + rng.rand() * 20
        fy = 70 + (i // 4) * (h - 140) / max(1, (n_fake // 4)) + rng.rand() * 20
        fakes.append(np.array([fx, fy], np.float32))
        fills.append(_fill(shape, FAKE_LO, FAKE_HI, seed * 100 + i))
    tgt0 = np.array([w * 0.30, h * 0.62], np.float32)
    # 在真目标**自身路径**上放一个假目标 ⇒ 途中会**压住它**（合并 ✓ 事实 5/7 ✓）。
    # ⚠ 放**远一点**（11 → 22 帧）：靠近的话目标刚建档就被压住 ⇒ 还没学会它自己的速度就
    #   进融合期 ✗（实测：前 5 帧就合并 ⇒ 速度从未建立 ⇒ 融合期只能干等 ✗ 那是场景不公平 ✗）。
    fakes.append(tgt0 + tv * 22.0)
    fills.append(_fill(shape, FAKE_LO, FAKE_HI, seed * 100 + 99))
    tgt_fill = _fill(shape, TGT_LO, TGT_HI, seed * 100 + 7)
    frames, boxes, truth, tvec = [], [], [], []
    for f in range(n):
        off = amp * np.sin(2.0 * np.pi * f / per)      # 群体平移（平滑 ✓）
        vel = amp * 2.0 * np.pi / per * np.cos(2.0 * np.pi * f / per)
        tvec.append((float(vel[0]), float(vel[1])))
        ox = int(np.clip(base + round(off[0]), 0, pad))
        oy = int(np.clip(base + round(off[1]), 0, pad))
        canvas = bg[oy:oy + h, ox:ox + w].copy()
        for i, fp in enumerate(fakes):
            p = fp + off
            _paste(canvas, shape, p[0], p[1], 0.0, fills[i])
        # 真目标（随群体 + **自身匀速** + **每帧旋转** ✓ 事实 2/3 ✓）
        tp = tgt0 + off + tv * f
        _paste(canvas, shape, tp[0], tp[1], spin * f, tgt_fill)
        frame = np.clip(canvas, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # 框 = **亮块连通域**（重叠自然并成一个大框 ✓ = 素材现象 ✓）
        bright = (canvas > BIN_THR).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        ncc, lab, stats, cent = cv2.connectedComponentsWithStats(bright, 8)
        bs = []
        for i in range(1, ncc):
            x, y, bw, bh, area = stats[i]
            if area < 400:
                continue
            bs.append((float(cent[i][0]), float(cent[i][1]), float(bw), float(bh)))
        if f in miss_at:                              # 事实 4：漏检（丢掉某个框）
            bs = [b for b in bs if math.hypot(b[0] - tp[0], b[1] - tp[1]) > 60]
        frames.append(frame)
        boxes.append(bs)
        truth.append((float(tp[0]), float(tp[1])))
    return {"frames": frames, "boxes": boxes, "truth": truth, "tvec": tvec, "dt": dt}
