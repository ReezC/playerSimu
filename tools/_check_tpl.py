"""检查角色模板 PNG：通道数、alpha、尺寸、朝向（临时脚本）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.imgio import imread  # noqa: E402


def p(*a):
    print(*a, flush=True)


root = pathlib.Path("datasets/sprites/player")
pngs = sorted(root.rglob("*.png"))
p(f"模板总数: {len(pngs)}")

for f in pngs[:6] + pngs[-3:]:
    im = imread(f, cv2.IMREAD_UNCHANGED)
    if im is None:
        p(f"  读不到 {f.name}")
        continue
    ch = im.shape[2] if im.ndim == 3 else 1
    info = f"  {f.parent.name}/{f.name}  {im.shape[1]}x{im.shape[0]} 通道={ch}"
    if ch == 4:
        a = im[:, :, 3]
        info += f"  alpha: min={a.min()} max={a.max()} 透明占比={(a < 128).mean() * 100:.0f}%"
    else:
        # 无 alpha：看角落像素（背景色）
        corner = im[0, 0].tolist()
        info += f"  无alpha! 角落色={corner}"
    p(info)
