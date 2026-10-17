"""第二问：分数 0.478（边缘域 ✓ 见上一条探针）离 0.55 还差一口气 —— 差在**哪儿**？
（2026-10-09 ✓ 现场 `data/mmap_diag/check_*.png` ✓）

两种可能，量一下就能分开 ✓：
  A. **几何差几个像素**（手工对齐"看上去"完美，但真值落在 1~3 px 外 ✓）⇒ 允许小范围平移后
     分数应当**明显回升** ✓；
  B. **渲染风格不同**（底图是项目按 wz 重绘的 ✓，游戏小地图是客户端原版渲染 + 一层透明叠加 ✓）
     ⇒ 就算对齐到最好，分数也上不去（**天花板**）✓ —— 这是"逐像素相关"这类判据的固有上限 ✗。

做法：把底图**四周补边** ⇒ `matchTemplate` 就能扫过小幅平移 ✓，再把**最大分**和**最好那个偏移**
打出来（灰度 / 边缘两种口径都量 ✓）。
"""

import sys

import cv2
import numpy as np

panel = cv2.imread("data/mmap_diag/check_panel.png")
canvas = cv2.imread("data/mmap_diag/check_canvas.png")
if panel is None or canvas is None:
    print("缺图：先各点一次把现场存下来 ✓")
    sys.exit(1)

MARGIN = 16          # 允许扫 ±16 px（远大于"手工对齐差一点"的量级 ✓）


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def edges(img):
    g = cv2.GaussianBlur(gray(img), (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def scan(tpl_img, big_img, name, scales=(1.0,)):
    best = (-2.0, None, None)
    tpl0 = tpl_img if tpl_img.ndim == 2 else gray(tpl_img)
    big0 = big_img if big_img.ndim == 2 else gray(big_img)
    for s in scales:
        tpl = (tpl0 if abs(s - 1.0) < 1e-6
               else cv2.resize(tpl0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        if tpl.shape[0] >= big0.shape[0] or tpl.shape[1] >= big0.shape[1]:
            continue
        r = np.nan_to_num(cv2.matchTemplate(big0, tpl, cv2.TM_CCOEFF_NORMED))
        _mn, mx, _ml, ml = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (float(mx), s, (int(ml[0]), int(ml[1])))
    print("  %-26s 最高分 = %.3f（scale=%s 偏移=%s）%s"
          % (name, best[0], best[1], best[2], "✓ 过门槛" if best[0] >= 0.55 else ""))


big_gray = cv2.copyMakeBorder(gray(canvas), MARGIN, MARGIN, MARGIN, MARGIN,
                              cv2.BORDER_REPLICATE)
big_edge = cv2.copyMakeBorder(edges(canvas), MARGIN, MARGIN, MARGIN, MARGIN,
                              cv2.BORDER_REPLICATE)

print("=== 允许 ±%d px 平移后（扫最大分）===" % MARGIN)
scan(panel, big_gray, "灰度（不缩放）")
scan(panel, big_edge, "边缘 Sobel（不缩放）")
print("=== 再放开一点点缩放（±2%，模拟渲染尺寸差）===")
scales = [0.96, 0.98, 1.0, 1.02, 1.04]
scan(panel, big_gray, "灰度（±4% 缩放）", scales)
scan(panel, big_edge, "边缘 Sobel（±4% 缩放）", scales)

print()
print("=== 判定 ===")
print("  · 平移后**明显回升**（比如 0.13 → 0.6+）⇒ 真因是**几何差几个像素** ✓")
print("      ⇒ 那就要让 `refine_fit`/`locate_fit` 的搜索范围/细化把这一小段吃进去 ✓")
print("  · 平移后**还是很低**（< 0.5 左右就封顶）⇒ 是**渲染风格不同的天花板** ✗")
print("      ⇒ 逐像素相关这类判据在这儿天生不够用 ⇒ 要么换**更结构化的判据**（边缘/骨架/角点 ✓），")
print("        要么把「手工对齐」这条路的**可信度**与这个分数**分开看待** ✓（不许拿它否定人 ✓）")
