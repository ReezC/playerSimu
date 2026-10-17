"""第五问：把"地形掩码"调到能用 —— 挑一个**两种来源覆盖率接近、且对齐处明显更高**的版本
（2026-10-09 ✓ 现场 `data/mmap_diag/check_*.png` ✓）

第四问的结论 ✓：
  · 「亮 + 去饱和」✗ 不行（面板整体更亮 ⇒ 覆盖率 33.8% vs 2.7% ✗ 掩码完全不是一回事）；
  · ⭐ 「**纹理**（局部标准差）」✓ 可用：覆盖率 7.6% vs 4.6% ✓、IoU 在**对齐处最高**
    （0.253）并且**往外单调下降**（dx=1→0.204、2→0.162、4→0.125 ✓）⇒ **能当尺子** ✓。
这一轮把阈值/尺度/容差扫一遍，挑一个"对齐处尽量高、两边覆盖率尽量接近"的版本 ✓：
  · `纹理` 的窗口 k 与阈值 thr；· `自适应阈值`（局部对比 ⇒ 天生对整体亮度差免疫 ✓）；
  · **容差**：算 IoU 前把掩码各膨胀 1~3 px ✓ —— 人眼对齐本来就允许 1~2 px 的差 ✓
    （第三问量到真值只差 ~2px ✓）⇒ 有容差的 IoU 才反映"人看得出来的偏差" ✓。
"""

import sys

import cv2
import numpy as np

panel = cv2.imread("data/mmap_diag/check_panel.png")
canvas = cv2.imread("data/mmap_diag/check_canvas.png")
if panel is None or canvas is None:
    print("缺图 ✓")
    sys.exit(1)
K2 = np.ones((3, 3), np.uint8)


def _tex(img, k, thr):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sd = np.sqrt(np.clip(cv2.blur(g * g, (k, k)) - mean * mean, 0, None))
    return ((sd > thr) & (g > 40)).astype(np.uint8)


def _adaptive(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 11, -6)
    return b.astype(np.uint8)


def _grad(img, thr=45):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return (cv2.magnitude(gx, gy) > thr).astype(np.uint8)


def shift(img, dx, dy=0):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def iou_tol(a, b, tol=2):
    """带容差的 IoU：两边各膨胀 `tol` px 后再交并 ✓（对"差一两个像素"宽容 ✓）。"""
    if tol > 0:
        a = cv2.dilate(a, K2, iterations=tol)
        b = cv2.dilate(b, K2, iterations=tol)
    inter = int(((a > 0) & (b > 0)).sum())
    uni = int(((a > 0) | (b > 0)).sum())
    return inter / float(uni) if uni else 0.0


FUNCS = [
    ("纹理 k7 thr18", lambda im: _tex(im, 7, 18)),
    ("纹理 k9 thr12", lambda im: _tex(im, 9, 12)),
    ("纹理 k5 thr25", lambda im: _tex(im, 5, 25)),
    ("自适应阈值", lambda im: _adaptive(im)),
    ("梯度>45", lambda im: _grad(im)),
]

print("%-16s %-10s %-10s %s" % ("掩码", "面板覆盖", "底图覆盖", "IoU(容差) @dx=0 / +2 / +8"))
for name, fn in FUNCS:
    mp, mc = fn(panel), fn(canvas)
    cov = "%.1f%%/%.1f%%" % (100.0 * mp.mean(), 100.0 * mc.mean())
    row = []
    for tol in (1, 2):
        v0 = iou_tol(mp, mc, tol)
        v2 = iou_tol(shift(mp, 2), mc, tol)
        v8 = iou_tol(shift(mp, 8), mc, tol)
        row.append("tol%d: %.3f/%.3f/%.3f" % (tol, v0, v2, v8))
    print("%-16s %-10s %s" % (name, cov, "   ".join(row)))

print()
print("挑法：**对齐处（dx=0）尽量高** ✓ ＋ **+8px 明显更低** ✓ ＋ 两边覆盖率别差太多 ✓")
