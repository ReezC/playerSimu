"""验证：珠缨真实位置的匹配分数 vs 全图最高分（临时脚本）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.imgio import imread  # noqa: E402

t = imread(pathlib.Path("datasets/sprites/player/珠缨/珠缨-stand1-0.png"),
           cv2.IMREAD_UNCHANGED)
b, al = t[:, :, :3].copy(), t[:, :, 3]
ys, xs = np.nonzero(al > 128)
b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
m = ((al > 128) * 255).astype(np.uint8)
print(f"模板 {b.shape[1]}x{b.shape[0]}", flush=True)

img0 = imread(pathlib.Path("projects/彩虹村东郊平原/frames/frame_00251.png"))
img = cv2.resize(img0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# 珠缨真实位置：原图身体中心约 (798, 430) → 降采样 (399, 215)
TRUE = (399, 215)

vis = img.copy()
cv2.circle(vis, TRUE, 6, (255, 0, 255), 2)

for s in (1.0, 1.15, 1.2, 1.3):
    w = max(8, int(round(b.shape[1] * 0.5 * s)))
    h = max(8, int(round(b.shape[0] * 0.5 * s)))
    tb = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    tm = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    r = cv2.matchTemplate(img, tb, cv2.TM_CCORR_NORMED, mask=tm)
    r = np.nan_to_num(r)
    # 真实位置对应匹配图上的左上角
    px, py = TRUE[0] - w // 2, TRUE[1] - h // 2
    ts = float(r[py, px]) if 0 <= py < r.shape[0] and 0 <= px < r.shape[1] else float('nan')
    # 真实位置附近 3px 内的最大值
    region = r[max(0,py-3):py+4, max(0,px-3):px+4]
    near = float(region.max()) if region.size else float('nan')
    _, mx, _, ml = cv2.minMaxLoc(r)
    print(f"s={s}: 模板{w}x{h}  真实位置分数={ts:.3f} 附近max={near:.3f}  "
          f"全图max={mx:.3f}@{ml}", flush=True)
    cv2.rectangle(vis, ml, (ml[0] + w, ml[1] + h), (0, 0, 255), 1)

cv2.imwrite("data/_true_score.png", vis)
print("saved data/_true_score.png（紫圈=珠缨真实位置，红框=全图最高分位置）", flush=True)
