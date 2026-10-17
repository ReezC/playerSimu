"""扫多帧找角色：stand1 模板对采样帧匹配，找分数最高的帧（临时脚本）。"""
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

files = sorted(pathlib.Path("projects/彩虹村东郊平原/frames").glob("*.png"))
samples = files[::50][:8]

for f in samples:
    img0 = imread(f)
    img = cv2.resize(img0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    w = int(b.shape[1] * 1.4)
    h = int(b.shape[0] * 1.4)
    tb = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    tm = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    r = cv2.matchTemplate(img, tb, cv2.TM_CCORR_NORMED, mask=tm)
    r = np.nan_to_num(r)
    _, mx, _, ml = cv2.minMaxLoc(r)
    print(f"{f.name}  score={mx:.3f}  pos={ml}", flush=True)
