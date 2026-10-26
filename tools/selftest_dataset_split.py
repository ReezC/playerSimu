# -*- coding: utf-8 -*-
"""自检：train/val 划分方式（`perception/prepare_dataset._assign_split`）。

**为什么单独钉这一条**（2026-10-24 现场量的）：

`寺院通道_珠缨` 的 `split.json` 是**按帧随机抽**的（val 里是 frame_00002 / 00004 /
00005 / 00009 …）—— 同一段录像里相邻帧几乎一模一样 ⇒ val 里全是"训练集见过的画面"。
同一个模型实测：

    训练集里训过的帧      召回 99.0%
    val（随机抽的）       召回 79.7%
    建完数据集后才拍的段   召回 30.1%   ← 这才是"实战"，用户看到的漏检

⇒ 默认改成"**每段录像留尾做 val**"（`SPLIT_SEGMENT`），老口径 `random` 只留作复现。
本自检钉五件：
  ① 默认就是按段留尾（不传 `mode` 时）；
  ② 同一段里 **val 的帧号全都大于 train 的帧号**（不再交错 ✓ 这才是"不泄漏"）；
  ③ val 张数≈ `val_ratio`，且至少各留 1 张（空 val 会让 ultralytics 报难懂的错）；
  ④ 多个前缀（本项目 + `p01__` 外来项目）各自留尾 ✓；
  ⑤ `random` 老口径仍能跑（复现旧数据集时用 ✓）。

跑：`python -m tools.selftest_dataset_split`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception import prepare_dataset as P            # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def t_segment_default():
    stems = ["frame_%05d" % i for i in range(100)]
    val = P._assign_split(stems, {}, 0.2, 42)          # 不传 mode ⇒ 默认
    check(P.SPLIT_SEGMENT == "segment",
          "默认划分方式不是 `segment`：%r" % (P.SPLIT_SEGMENT,))
    check(len(val) == 20,
          "val 张数该是 20（100×0.2），实得 %d ⇒ 比例没按 val_ratio 走 ✗" % len(val))

    va = sorted(P._stem_key(s)[1] for s in val)
    tr = sorted(P._stem_key(s)[1] for s in stems if s not in val)
    check(min(va) > max(tr),
          "val 和 train **交错**了（val 从 %d 开始，train 到 %d 才结束）⇒ 相邻帧几乎一样，"
          "val 分数会虚高 ✗" % (min(va), max(tr)))
    check(va == list(range(80, 100)),
          "val 该是**这段录像的尾巴**（80~99），实得 %s ✗" % (va[:3],))


def t_multi_prefix():
    stems = (["frame_%05d" % i for i in range(50)]
             + ["p01__frame_%05d" % i for i in range(50)])
    val = P._assign_split(stems, {}, 0.2, 42)
    own = [s for s in val if not s.startswith("p01__")]
    p01 = [s for s in val if s.startswith("p01__")]
    check(len(own) == 10 and len(p01) == 10,
          "每个前缀都要各自留 10 张尾（实得 本项目 %d、p01 %d）⇒ 外来项目的画面全进 train 了 ✗"
          % (len(own), len(p01)))
    check(min(P._stem_key(s)[1] for s in own) == 40,
          "本项目的 val 该从 frame_00040 起（尾巴），实得 %s ✗" % (sorted(own)[:2],))
    check(min(P._stem_key(s)[1] for s in p01) == 40,
          "p01 的 val 该从 frame_00040 起，实得 %s ✗" % (sorted(p01)[:2],))


def t_tiny_block():
    # 只有 1 帧的段 ⇒ 不能整段进 val（train 会空）
    check(P._assign_split(["frame_00000"], {}, 0.2, 42) == set(),
          "只有 1 帧时不该把它划进 val（train 就空了）✗")
    val = P._assign_split(["frame_00000", "frame_00001"], {}, 0.2, 42)
    check(len(val) == 1, "2 帧时该留 1 张 val、1 张 train，实得 val=%r ✗" % (val,))


def t_random_still_works():
    stems = ["frame_%05d" % i for i in range(100)]
    val = P._assign_split(stems, {}, 0.2, 42, mode=P.SPLIT_RANDOM)
    check(len(val) == 20, "`random` 老口径也要给 20 张 val，实得 %d ✗" % len(val))
    # 老口径本来就是交错抽 —— 这里**不该**要求连续（那是新口径的规矩）
    va = sorted(P._stem_key(s)[1] for s in val)
    check(min(va) < 40,
          "`random` 老口径居然只抽了尾巴（%s）⇒ 复现旧划分的能力没了 ✗" % (va[:3],))


TESTS = [
    ("⭐⭐⭐⭐ train/val 默认按**采集段留尾**（不交错 ⇒ 不再自欺）", t_segment_default),
    ("⭐⭐⭐ 多个前缀各自留尾（外来项目的画面不许全进 train）", t_multi_prefix),
    ("⭐⭐ 极小段：1 帧不划走、2 帧各留一", t_tiny_block),
    ("⭐⭐ 老口径 `random` 仍可用（复现旧数据集）", t_random_still_works),
]


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:                 # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
