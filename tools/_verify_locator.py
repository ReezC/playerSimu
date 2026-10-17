"""验证 PlayerLocator 模块（临时脚本）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from core.imgio import imread  # noqa: E402
from perception.player_locator import PlayerLocator  # noqa: E402


def p(*a):
    print(*a, flush=True)


loc = PlayerLocator("珠缨", scale=1.25, threshold=0.78)
p(f"加载模板 {loc.template_count} 个")

img = imread(pathlib.Path("projects/彩虹村东郊平原/frames/frame_00251.png"))

# 珠缨真实位置（原图中心 ≈ (798, 430)）
r = loc.locate(img)
if r:
    cx, cy, w, h, score, name = r
    p(f"命中: 中心=({cx:.0f},{cy:.0f}) {w}x{h} score={score:.3f} 模板={name}")
    p(f"期望中心≈(798,430)，偏差=({cx-798:.0f},{cy-430:.0f})")
    p("成功!" if abs(cx - 798) < 60 and abs(cy - 430) < 60 else "位置偏差较大")
else:
    p("未命中（分数低于阈值）")
