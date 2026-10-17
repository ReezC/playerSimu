"""扫 stand1 模板的最佳 scale（极简，前台快速）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.imgio import imread  # noqa: E402

t = imread(pathlib.Path("datasets/sprites/player/珠缨-stand1/珠缨-stand1-0.png"),
           cv2.IMREAD_UNCHANGED)
b, al = t[:, :, :3].copy(), t[:, :, 3]
ys, xs = np.nonzero(al > 128)
b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
m = ((al > 128) * 255).astype(np.uint8)

fp = sorted(pathlib.Path("projects/彩虹村东郊平原/frames").glob("*.png"))[0]
img0 = imread(fp)
img = cv2.resize(img0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
print(f"模板 {b.shape[1]}x{b.shape[0]}  帧 {img0.shape}->{img.shape}", flush=True)

best = []
for s in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
    w = max(8, int(b.shape[1] * s))
    h = max(8, int(b.shape[0] * s))
    tb = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    tm = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    r = cv2.matchTemplate(img, tb, cv2.TM_CCORR_NORMED, mask=tm)
    _, mx, _, ml = cv2.minMaxLoc(r)
    best.append((mx, s, ml))
    print(f"scale={s:.1f} score={mx:.3f} 降采样pos={ml} 原坐标=({int(ml[0]/0.5)},{int(ml[1]/0.5)})", flush=True)
