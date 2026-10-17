"""量一件事：**面板上那些游戏 UI 覆盖层，是不是把匹配分压下去的原因**（2026-10-09 ✓）。

现场（`data/mmap_diag/check_*.png` ✓ 用户刚点「实测精度」存的 ✓）：
  · `check_panel.png` = 喂进去的那块小地图（240×79 ✓）；
  · `check_canvas.png` = 这张图的底图（240×79 ✓）。
同尺寸、1:1 ⇒ 看一眼就知道：**布局完全一致** ✓（框选是对的 ✓），但面板上**多了**：
  · 两个亮蓝的圆（小地图的按钮 ✓）；· 红点 / 绿点 / 黄菱形（玩家、NPC、传送门标记 ✓）；
  · 整体更亮、对比更强（游戏小地图是压过一层透明的 ✓）。
⇒ 假设：**是这些覆盖层把 `TM_CCOEFF_NORMED` 拖低了** ✗（原始分 0.232），
   而人眼看"手工对齐"时**自动忽略**它们 ⇒ 于是"人对得很准、工具却说量不出来" ✓✓。

这个脚本把几种"更抗覆盖层"的口径都量一遍，看谁能过 0.55 ✓ —— **先量再改** ✓（别猜 ✗）。
"""

import sys

import cv2
import numpy as np

P = "data/mmap_diag/check_panel.png"
C = "data/mmap_diag/check_canvas.png"

panel = cv2.imread(P)
canvas = cv2.imread(C)
if panel is None or canvas is None:
    print("缺图：先各点一次「实测精度」/「自动定位」把现场存下来 ✓")
    sys.exit(1)
print("面板 %s  底图 %s" % (panel.shape, canvas.shape))


def score(a, b, name, mask=None):
    """a 在 b 里找（同尺寸 ⇒ 只有一个位置 ✓）。返回分并打印。"""
    if a.shape[:2] != b.shape[:2]:
        print("  %-22s 尺寸不同，跳过" % name)
        return None
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a = np.ascontiguousarray(a.astype(np.float32))
    b = np.ascontiguousarray(b.astype(np.float32))
    try:
        if mask is None:
            r = cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)[0][0]
        else:
            # 掩码匹配只支持 SQDIFF / CCORR_NORMED ⇒ 用后者（先各自去均值 ⇒ 抗亮度差 ✓）
            m = np.ascontiguousarray(mask.astype(np.uint8))
            aa = a - float(a[m > 0].mean())
            r = cv2.matchTemplate(b, aa, cv2.TM_CCORR_NORMED, mask=m)[0][0]
            r = float(r)
    except Exception as e:                            # noqa: BLE001
        print("  %-22s 算不了：%s" % (name, e))
        return None
    print("  %-22s 分 = %.3f %s" % (name, float(r), "✓ 过门槛" if float(r) >= 0.55 else ""))
    return float(r)


print("=== ① 现在这套（基准，应当复现 0.232）===")
score(panel, canvas, "原灰度 CCOEFF")

print("=== ② 把「高饱和色块」掩掉（红/绿/黄标记 + 亮蓝按钮 ✓）===")
hsv_p = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
sat = hsv_p[:, :, 1]
for thr in (60, 90, 120):
    keep = (sat < thr).astype(np.uint8) | (cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) < 40).astype(np.uint8)
    score(panel, canvas, "掩饱和>S%d CCORR" % thr, mask=keep)

print("=== ③ 只在「边缘/结构」上比（图标是大色块、边少 ✓ 地形边多 ✓）===")


def edges(img, mode="sobel"):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    if mode == "sobel":
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.convertScaleAbs(mag)
    return cv2.Canny(g, 40, 120)


for mode in ("sobel", "canny"):
    ep, ec = edges(panel, mode), edges(canvas, mode)
    score(ep, ec, "边缘(%s) CCOEFF" % mode)

print("=== ④ 边缘 + 掩掉高饱和（两条一起）===")
ep, ec = edges(panel, "sobel"), edges(canvas, "sobel")
keep = (sat < 90).astype(np.uint8)
score(ep, ec, "边缘Sobel 掩饱和", mask=keep)

print("=== ⑤ 只比「暗底上的亮结构」（小地图地形是亮绿/亮褐，覆盖层是纯色）===")
g_p = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
g_c = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
score(cv2.threshold(g_p, 60, 255, cv2.THRESH_BINARY)[1],
      cv2.threshold(g_c, 60, 255, cv2.THRESH_BINARY)[1], "二值>60 CCOEFF")
