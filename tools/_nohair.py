"""验证：抹掉模板头发区域后，匹配分数是否恢复（临时脚本）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.imgio import imread  # noqa: E402


def load_tpl(path, cut_head=0):
    im = imread(path, cv2.IMREAD_UNCHANGED)
    b, al = im[:, :, :3].copy(), im[:, :, 3].copy()
    if cut_head:
        al[:cut_head, :] = 0          # 抹掉头部（发色不同的区域）
    ys, xs = np.nonzero(al > 128)
    if len(xs) < 100:
        return None
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return b, ((al > 128) * 255).astype(np.uint8)


tpl_path = pathlib.Path("datasets/sprites/player/珠缨-stand1/珠缨-stand1-0.png")
img0 = imread(pathlib.Path("projects/彩虹村东郊平原/frames/frame_00251.png"))
img = cv2.resize(img0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

for cut in (0, 28, 34, 40):
    t = load_tpl(tpl_path, cut_head=cut)
    if not t:
        continue
    b, m = t
    row = []
    for s in (1.0, 1.2, 1.4, 1.6):
        w = max(8, int(b.shape[1] * s))
        h = max(8, int(b.shape[0] * s))
        tb = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
        tm = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        r = cv2.matchTemplate(img, tb, cv2.TM_CCORR_NORMED, mask=tm)
        r = np.nan_to_num(r)
        _, mx, _, ml = cv2.minMaxLoc(r)
        row.append(f"s={s}:{mx:.3f}@{ml}")
    print(f"抹掉头部{cut}px: " + "  ".join(row), flush=True)
