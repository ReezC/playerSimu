"""角色模板匹配可视化（快速版）：降采样 + 核心动作，让用户判断匹配对不对。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.imgio import imread  # noqa: E402


def p(*a):
    print(*a, flush=True)


def load_tpl(path):
    im = imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    b, al = im[:, :, :3].copy(), im[:, :, 3]
    ys, xs = np.nonzero(al > 128)
    if len(xs) < 100:
        return None
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return b, ((al > 128) * 255).astype(np.uint8)


def nms(dets, iou_thr=0.4):
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []
    for d in dets:
        _, x, y, w, h, _ = d
        for k in keep:
            _, kx, ky, kw, kh, _ = k
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            u = w * h + kw * kh - ix * iy
            if u > 0 and ix * iy / u > iou_thr:
                break
        else:
            keep.append(d)
    return keep


# 核心动作各取 1 帧
root = pathlib.Path("datasets/sprites/player")
tpls = []
for act in ("stand1", "walk1", "jump", "alert", "swingO1"):
    files = sorted((root / f"珠缨-{act}").glob("*.png"))
    if not files:
        continue
    t = load_tpl(files[0])
    if t:
        b, m = t
        tpls.append((act, b, m))
        tpls.append((act + "@L", cv2.flip(b, 1), cv2.flip(m, 1)))
p(f"模板 {len(tpls)} 个")

# 读训练帧（1920x1080），降采样到 960 宽加速
fpng = sorted(pathlib.Path("projects/彩虹村东郊平原/frames").glob("*.png"))
img0 = imread(fpng[0])
DS = 0.5
img = cv2.resize(img0, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)
p(f"帧 {fpng[0].name} {img0.shape} → 降采样 {img.shape}")

SCALES = [1.0, 1.2, 1.4, 1.6]   # 相对降采样后模板的缩放
THR = 0.70
cands = []
for name, tb, tm in tpls:
    tb = cv2.resize(tb, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)
    tm = cv2.resize(tm, None, fx=DS, fy=DS, interpolation=cv2.INTER_NEAREST)
    for sc in SCALES:
        w = max(8, int(round(tb.shape[1] * sc)))
        h = max(8, int(round(tb.shape[0] * sc)))
        if w >= img.shape[1] or h >= img.shape[0]:
            continue
        tbb = cv2.resize(tb, (w, h), interpolation=cv2.INTER_AREA)
        tmm = cv2.resize(tm, (w, h), interpolation=cv2.INTER_NEAREST)
        r = cv2.matchTemplate(img, tbb, cv2.TM_CCORR_NORMED, mask=tmm)
        r = np.nan_to_num(r)
        ys, xs = np.where(r > THR)
        for y, x in zip(ys, xs):
            cands.append((float(r[y, x]), x, y, w, h, name))

kept = nms(cands, 0.4)
p(f"候选 {len(cands)} → NMS 后 {len(kept)}")

vis = img.copy()
for i, (sc, x, y, w, h, name) in enumerate(sorted(kept, reverse=True)[:12]):
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(vis, f"{sc:.2f}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    p(f"  #{i} score={sc:.3f} 降采样坐标=({x},{y}) {w}x{h}  原坐标=({int(x/DS)},{int(y/DS)})  {name}")

cv2.imwrite("data/_match_result.png", vis)
p("saved data/_match_result.png（坐标是降采样后的，原图要 ×2）")
