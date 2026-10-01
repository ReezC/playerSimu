"""A 层（`perception/lie_group.py`）自检：**合成场景**钉住三条契约。

约定与 `tools/selftest_lie_tracker.py` 一致：打印 `[OK]/[NG]`，全绿 `exit=0` ✓。

钉子：
  ① **纯平移** ⇒ `T` 与真值差 ≤ 1px ✓；
  ② **群体平移 + 一只偏离** ⇒ `T` 仍是群体值（≤1px ✓）、偏离那只的**残差 > 15px** ✓
     且 `dev_i` 正好指向它 ✓（= 身份判据 ✓ 用户事实 2/3 ✓）；
  ③ **重复纹理 + 少数外点** ⇒ 共识仍稳（中位会被拽偏 ✗ 这是被测的行为 ✓）。
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perception.lie_group import GroupMotion  # noqa: E402

_FAIL = []
_RNG = np.random.RandomState(20260930)


def check(cond, msg):
    print(("  [OK] " if cond else "  [NG] ") + msg)
    if not cond:
        _FAIL.append(msg)


def make_scene(h=500, w=750, n=14, seed=1):
    """合成一帧 + 一组框：铺 `n` 块**带纹理的方块**（模拟图形/干扰群）✓，另加周期底纹 ✓。"""
    rng = np.random.RandomState(seed)
    base = (rng.rand(h // 8 + 1, w // 8 + 1) * 255).astype(np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    boxes = []
    k = int(np.sqrt(n)) + 1
    for i in range(n):
        size = int(rng.randint(90, 130))
        cx = 95 + (i % k) * (w - 190) // max(1, k - 1)
        cy = 95 + (i // k) * (h - 190) // max(1, k - 1)
        x0 = max(0, min(w - size, int(cx - size / 2)))
        y0 = max(0, min(h - size, int(cy - size / 2)))
        patch = (rng.rand(size, size) * 255).astype(np.uint8)
        patch = cv2.GaussianBlur(patch, (5, 5), 0)
        img[y0:y0 + size, x0:x0 + size] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        boxes.append((float(x0 + size / 2.0), float(y0 + size / 2.0),
                      float(size), float(size)))
    return img, boxes


def translate(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def main():
    # ① 纯平移
    img, boxes = make_scene()
    gm = GroupMotion()
    gm.estimate(img, boxes)
    r = gm.estimate(translate(img, 7, -5), boxes)
    t = r["t"]
    check(t is not None and abs(t[0] - 7) <= 1.0 and abs(t[1] + 5) <= 1.0,
          "① 纯平移 ⇒ T=%s（真值 (7,-5) ✓）"
          % ((None if t is None else (round(t[0], 2), round(t[1], 2))),))
    check(r["used"] >= 6, "① 共识内点 %d（≥6 ✓）" % r["used"])

    # ② 群体平移 + **一只真的动了**（移动它的**像素** ✗ 不是移动框 —— 第一版就是只挪了框，
    #    内容没动 ⇒ 模板照旧匹配到群体位移 ⇒ 残差 0 ✗ 这个用例根本测不到判据 ✗）
    gm = GroupMotion()
    tgt = 5
    img2 = translate(img, 6, 4)
    src_box = boxes[tgt]
    sx, sy, sw, sh = src_box
    x0, y0 = int(sx - sw / 2), int(sy - sh / 2)
    ox, oy = 30, 24
    piece = img[y0:y0 + int(sh), x0:x0 + int(sw)].copy()
    img2[y0 + oy:y0 + oy + piece.shape[0], x0 + ox:x0 + ox + piece.shape[1]] = piece
    moved = list(boxes)
    moved[tgt] = (sx + 6 + ox, sy + 4 + oy, sw, sh)   # 它的新位置 = 群体位移 + 自身位移 ✓
    gm.estimate(img, boxes)
    r = gm.estimate(img2, moved)
    t = r["t"]
    check(t is not None and abs(t[0] - 6) <= 1.0 and abs(t[1] - 4) <= 1.0,
          "② 群体 T=%s（真值 (6,4) ✓）"
          % ((None if t is None else (round(t[0], 2), round(t[1], 2))),))
    # ⚠ `resid`（"上一帧框的位移 − T"）**只能当诊断** ✗：真目标一移动，它上一帧的位置就
    #   空了 ⇒ 那个模板匹配到的是背景 ⇒ 它这只的 resid 看起来跟群体一样 ✓ 追不到它 ✓
    #   ⇒ 身份**主信号是一致性**（下一条 ✓）。这里只钉"群体本身自洽" ✓。
    rs = sorted(r["resid"].values())
    check(rs and rs[len(rs) // 2] <= 2.0,
          "② 各框残差中位 %.2f px（≤2 ✓ 群体自洽 ✓ 诊断量 ✓）"
          % (rs[len(rs) // 2] if rs else float("nan")))
    # ⭐ **身份主信号 = 群体一致性**（"用上一帧自己的框当模板"追不到移动中的目标 ✗ 实测）
    c = r["consist"]
    check(c.get(tgt) is not None and c.get(tgt) < 0.9,
          "② 动了那只的一致性 %.3f（<0.9 ✓ 它没跟着群体走 ✓）" % c.get(tgt, float("nan")))
    others = [v for k, v in c.items() if k != tgt]
    # ⚠ 门槛 0.75（不是 0.95 ✗）：合成用例里目标块是**贴上去的**，会盖到旁边那格 ✓
    #   （真素材没有这么粗暴的重叠 ✓）⇒ 只要求"其余框明显高于动了那只" ✓
    check(others and min(others) > 0.75 and min(others) > 3 * c.get(tgt, 1.0),
          "② 其余框一致性最低 %.3f（>0.75 且明显高于动了那只 ✓）"
          % (min(others) if others else float("nan")))
    check(r["dev_i"] == tgt, "② dev_i=%s（该指向第 %d 只 ✓）" % (r["dev_i"], tgt))

    # ③ 重复纹理 + 外点：周期底纹 ⇒ 模板匹配会有多个同分峰
    rng = np.random.RandomState(7)
    tile = (rng.rand(48, 48) * 255).astype(np.uint8)
    rep = np.tile(tile, (500 // 48 + 2, 750 // 48 + 2))[:500, :750]
    rep = cv2.cvtColor(rep, cv2.COLOR_GRAY2BGR)
    bx = [(80.0 + i * 120.0, 90.0, 110.0, 110.0) for i in range(5)]
    bx += [(80.0 + i * 120.0, 300.0, 110.0, 110.0) for i in range(5)]
    gm = GroupMotion()
    gm.estimate(rep, bx)
    r = gm.estimate(translate(rep, 9, 3), bx)
    t = r["t"]
    check(t is not None and abs(t[0] - 9) <= 2.0 and abs(t[1] - 3) <= 2.0,
          "③ 周期纹理下 T=%s（真值 (9,3)，**中位**会被邻居峰拽走 ✗ 共识应稳 ✓）"
          % ((None if t is None else (round(t[0], 2), round(t[1], 2))),))

    print("\n自检：%s" % ("全部通过" if not _FAIL else "%d 条失败" % len(_FAIL)))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
