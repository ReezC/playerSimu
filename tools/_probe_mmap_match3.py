"""第三问：把「游戏 UI 覆盖层」从**面板**上抹掉（底图上本来就没有 ✓）之后，分数能到多少？
（2026-10-09 ✓ 现场 `data/mmap_diag/check_*.png` ✓）

前两次量的（原始分 0.135 / 实测精度那份 0.232 ✓、边缘域 0.478 ✓）说明"结构像、像素色不像" ✓。
这一次把两种**已知的干扰**逐个排除，看天花板到底在哪 ✓：
  · 面板上那些**高饱和色块**（红点/绿点/黄菱形/亮蓝按钮 ✓）⇒ 用掩码 + 邻域填充抹掉 ✓；
  · 面板**整体更亮/更糊**（小地图上压了一层透明 ✓）⇒ 只看**结构**（去均值 + 提对比 / 边缘 ✓）。
抹的手法：掩码处用 `cv2.inpaint` 填 ✓（比"置 0"强 ✗ 置 0 会造出假边缘 ✗），再算 CCOEFF ✓。
"""

import sys

import cv2
import numpy as np

panel = cv2.imread("data/mmap_diag/check_panel.png")
canvas = cv2.imread("data/mmap_diag/check_canvas.png")
if panel is None or canvas is None:
    print("缺图：先各点一次把现场存下来 ✓")
    sys.exit(1)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def edges(img):
    g = cv2.GaussianBlur(gray(img), (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def cc(a, b, name, allow_shift=0):
    a = a if a.ndim == 2 else gray(a)
    b = b if b.ndim == 2 else gray(b)
    if allow_shift:
        b = cv2.copyMakeBorder(b, allow_shift, allow_shift, allow_shift,
                               allow_shift, cv2.BORDER_REPLICATE)
    if a.shape[0] > b.shape[0] or a.shape[1] > b.shape[1]:
        print("  %-30s 尺寸放不下" % name)
        return None
    r = np.nan_to_num(cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED))
    _mn, mx, _ml, ml = cv2.minMaxLoc(r)
    off = (ml[0] - allow_shift, ml[1] - allow_shift)
    print("  %-30s 分 = %.3f  偏移=%s %s"
          % (name, mx, off, "✓ 过门槛" if mx >= 0.55 else ("（过 0.45）" if mx >= 0.45 else "")))
    return float(mx)


hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
sat, val = hsv[:, :, 1], hsv[:, :, 2]
# 覆盖层 = 高饱和（彩点/按钮 ✓）或 极亮（白框 ✓）
mask = (((sat > 80) & (val > 60)) | (val > 235)).astype(np.uint8) * 255
mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
print("  覆盖层掩码：%d / %d 像素（%.1f%%）"
      % (int((mask > 0).sum()), mask.size, 100.0 * (mask > 0).sum() / mask.size))

print("=== ① 抹掉覆盖层（inpaint）后 ===")
p_clean = cv2.inpaint(panel, mask, 3, cv2.INPAINT_TELEA)
cc(p_clean, canvas, "inpaint 后 灰度 CCOEFF")
cc(edges(p_clean), edges(canvas), "inpaint 后 边缘 CCOEFF")

print("=== ② 只留「暗底 + 亮结构」（两边都二值化 ⇒ 抹掉亮度/色偏差异）===")
for thr in (40, 60, 80):
    cc(cv2.threshold(gray(p_clean), thr, 255, cv2.THRESH_BINARY)[1],
       cv2.threshold(gray(canvas), thr, 255, cv2.THRESH_BINARY)[1],
       "二值 >%d（inpaint 后）" % thr)

print("=== ③ 结构版：先 inpaint 再只比边缘，并允许 ±6px 平移 ===")
cc(edges(p_clean), edges(canvas), "边缘 + 允许 ±6px", allow_shift=6)

print()
print("=== 结论怎么读 ===")
print("  · 抹掉覆盖层后**明显上一个台阶**（比如 →0.6+）⇒ 真凶是**那些图标/按钮** ✓")
print("     ⇒ 修法：匹配前先掩掉高饱和覆盖层（本项目已有「按饱和色找黄点」的同款做法 ✓）")
print("  · 抹掉之后**还是 0.5 上下** ⇒ 剩下的差值来自**底图渲染 vs 客户端渲染** ✓")
print("     ⇒ 那就不该拿逐像素相关当「标定对不对」的判据 ✗（换结构判据 / 或者人眼叠图 ✓）")
