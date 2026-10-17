"""第四问（决定性）：**"地形重合率"能不能当判据** —— 对 UI 覆盖层/渲染差异不敏感、
又能真的区分"对齐准 / 不准"？（2026-10-09 ✓ 现场 `data/mmap_diag/check_*.png` ✓）

要挑一个"掩码函数"：把一张小地图变成"**地形在哪**"的二值图 ✓，要求
  ① 游戏小地图上的**图标/按钮**（纯色块 ✓）**不该**被算成地形 ✗；
  ② 地形（亮绿/褐色、**有纹理** ✓）要算 ✓；
  ③ 对"客户端渲染 vs 项目重绘"的亮度/色偏**不敏感** ✓。

候选：
  (a) 亮度阈值 + 排除高饱和（图标 = 高饱和纯色 ✓）；
  (b) ⭐ **局部方差**（有纹理 ⇒ 地形 ✓；纯色块/平背景 ⇒ 方差低 ✓）—— 理论上最对路 ✓。
做法：两边各算掩码 ⇒ 算 **IoU** ✓；再把面板整体平移 +2/+4/+8 px ⇒ 看 IoU 是否**单调下降** ✓
（能单调 ⇒ 它就能当"偏了多少"的尺子 ✓，而不是像匹配分那样一上来就趴地上 ✗）。
"""

import sys

import cv2
import numpy as np

panel = cv2.imread("data/mmap_diag/check_panel.png")
canvas = cv2.imread("data/mmap_diag/check_canvas.png")
if panel is None or canvas is None:
    print("缺图：先各点一次把现场存下来 ✓")
    sys.exit(1)


def mask_bright(img):
    """(a) 亮 + 不极端饱和 ⇒ 地形。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    icon = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 140)).astype(np.uint8)
    m = ((g > 55).astype(np.uint8)) * (1 - icon)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return (m > 0).astype(np.uint8)


def mask_texture(img, k=7, thr=18.0):
    """(b) ⭐ 局部标准差（纹理）⇒ 地形；纯色块/平背景被排掉。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sq = cv2.blur(g * g, (k, k))
    var = np.clip(sq - mean * mean, 0, None)
    sd = np.sqrt(var)
    m = ((sd > thr) & (g > 45)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return (m > 0).astype(np.uint8)


def iou(a, b):
    inter = int(((a > 0) & (b > 0)).sum())
    uni = int(((a > 0) | (b > 0)).sum())
    return (inter / float(uni)) if uni else 0.0


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


print("掩码覆盖率：")
for name, fn in (("(a) 亮+去饱和", mask_bright), ("(b) 纹理", mask_texture)):
    mp, mc = fn(panel), fn(canvas)
    print("  %-14s 面板 %.1f%%   底图 %.1f%%"
          % (name, 100.0 * mp.mean(), 100.0 * mc.mean()))

print()
print("=== IoU 随「面板整体平移」的变化（真值应当就在 0 附近 ✓）===")
for name, fn in (("(a) 亮+去饱和", mask_bright), ("(b) 纹理", mask_texture)):
    mp, mc = fn(panel), fn(canvas)
    row = []
    for d in (0, 1, 2, 4, 8, 16):
        v = iou(shift(mp, d, 0), mc)
        row.append("dx=%2d: %.3f" % (d, v))
    print("  %-14s %s" % (name, "  ".join(row)))
    row = []
    for d in (0, 2, 4, 8):
        v = iou(shift(mp, 0, d), mc)
        row.append("dy=%2d: %.3f" % (d, v))
    print("  %-14s %s" % (name, "  ".join(row)))

print()
print("=== 两种来源各算一遍（面板=小地图 ✓ / 底图=重绘 ✓）===")
for name, fn in (("(a) 亮+去饱和", mask_bright), ("(b) 纹理", mask_texture)):
    print("  %-14s IoU(对齐 0,0) = %.3f" % (name, iou(fn(panel), fn(canvas))))
print()
print("读法：IoU 在 0 处最高、往外**单调下降** ⇒ 它就能当尺子 ✓（而且绝对值高 ⇒ 说明「重合好」✓）")
