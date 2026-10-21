# -*- coding: utf-8 -*-
"""离线量「**候选点像不像玩家那个点**」（用户 2026-10-11 ✓ 现场："**队友橙色点上方有个很小的
黄点标记，经常认错**"✗）—— 给 `DOT_PLAYER_SIDE_HARD_REL` / `DOT_LIKE_MIN` / σ 定阈值用 ✓。

**为什么要有它**：那几个阈值原来只有"真点 26×25 @ 面板宽 753"这一条量测 ✓（见
`DOT_CORE_*` 那段 ✓）—— 而现场是**两个点一起出现**（真点 + 队友头上那个小的 ✓）：
不把它们各自的"短边占比 / 分数"量出来，阈值就是拍的 ✗（本仓库老规矩 ✓）。

跑法（**只用现成样本 ✓ 不开游戏 ✓ 不连推流 ✓**）：
    python -m tools.mmap_dot_like_probe                    # 默认扫 data/dot_probe/*.png
    python -m tools.mmap_dot_like_probe 某个.png 另一个.png  # 指定文件
    python -m tools.mmap_dot_like_probe --dir 某目录

看什么：
  · **短边占比** = 短边 ÷ 面板宽 —— 真点应当集中在 2%~4%（≈0.032 ✓）；
    比它低一个量级的（< 1.2% ✓）就是"队友那个小标记"那一档 ✓ ⇒ 硬尺寸闸照它定 ✓；
  · **like 分**（`dot_player_like` ✓）—— 真点该高、小标记该低 ✓ ⇒ 滞后门槛照它定 ✓；
  · ⚠ 只报**事实**：不做任何挑选、不猜哪个是玩家 ✓（那是 `find_player_dot` 的事 ✓）。
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2                                                       # noqa: E402
import numpy as np                                              # noqa: E402

from perception import minimap as mm                            # noqa: E402
from perception import map_dot

DEFAULT_DIR = "data/dot_probe"


def _cands(panel):
    """这一块面板上的候选（**颜色层** ✓ 黄色族 ✓）—— 与实时那条链同一入口 ✓。"""
    from perception import map_dot
    pw = int(panel.shape[1])
    out = {}
    m = map_dot.dot_mask(panel, "yellow")
    out["yellow"], _ = map_dot._dot_candidates(m, pw)
    m2 = map_dot.dot_core_mask(panel)
    out["core"], _ = map_dot._dot_candidates(m2, pw)
    return pw, out


def _row(tag, pw, c):
    from perception import map_dot
    side = 0.5 * (float(c["w"]) + float(c["h"]))
    rel = side / max(1.0, float(pw))
    like = map_dot.dot_player_like(c, float(pw))
    return ("   %-7s %4d×%-4d  面积 %5d  短边占比 %6.3f%%  like %5.3f  @(%.1f, %.1f)"
            % (tag, int(c["w"]), int(c["h"]), int(c["area"]), rel * 100.0, like,
               float(c["x"]), float(c["y"])))


def main(argv):
    from perception import map_dot
    paths = []
    if len(argv) >= 2 and argv[0] == "--dir":
        paths = sorted(glob.glob(os.path.join(argv[1], "*.png")))
    elif argv:
        paths = list(argv)
    else:
        paths = sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.png")))
    if not paths:
        print("没有样本文件（找 %s 或直接给文件名）" % DEFAULT_DIR)
        return 1
    print("=== 量 %d 个面板（`dot_player_like` 的输入口径 ✓ 只报事实 ✗ 不挑点）===" % len(paths))
    rels, likes = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print("· %s：读不出来（跳过）" % os.path.basename(p))
            continue
        pw, groups = _cands(img)
        print("· %s  面板 %d×%d" % (os.path.basename(p), img.shape[1], img.shape[0]))
        for tag, cands in groups.items():
            if not cands:
                print("   %-7s （无候选）" % tag)
                continue
            for c in cands[:6]:
                print(_row(tag, pw, c))
                rels.append(0.5 * (float(c["w"]) + float(c["h"])) / max(1.0, float(pw)))
                likes.append(map_dot.dot_player_like(c, float(pw)))
        print("")
    if rels:
        rels = sorted(rels)
        likes = sorted(likes)
        def _q(a, f):
            return a[min(len(a) - 1, max(0, int(round(f * (len(a) - 1)))))]
        print("=== 汇总（%d 个候选）===" % len(rels))
        print("  短边占比：min %.3f%% / p25 %.3f%% / 中位 %.3f%% / p75 %.3f%% / max %.3f%%"
              % (rels[0] * 100, _q(rels, .25) * 100, _q(rels, .5) * 100,
                 _q(rels, .75) * 100, rels[-1] * 100))
        print("  like 分：min %.3f / p25 %.3f / 中位 %.3f / p75 %.3f / max %.3f"
              % (likes[0], _q(likes, .25), _q(likes, .5), _q(likes, .75), likes[-1]))
        print("  ⇒ 当前常量：DOT_PLAYER_SIDE_REL=%.3f  DOT_PLAYER_SIDE_HARD_REL=%.3f  "
              "DOT_LIKE_MIN=%.2f" % (map_dot.DOT_PLAYER_SIDE_REL,
                                     getattr(map_dot, "DOT_PLAYER_SIDE_HARD_REL", float("nan")),
                                     getattr(map_dot, "DOT_LIKE_MIN", float("nan"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
