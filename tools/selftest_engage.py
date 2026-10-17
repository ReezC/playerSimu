# -*- coding: utf-8 -*-
"""「输出收益口径」（`decision/engage.py` ✓ 标飞第一块）的用例 —— **纯函数，毫秒级** ✓。

用户 2026-10-11 ✓：要做「标飞」职业 AI，三条特性（**贴脸挥拳≈0 输出** / **跳能舍近求远** /
**频繁评估最佳输出位置**）都指向同一个缺失概念 ⇒ 先把它立成**能单测的口径** ✓ 再谈选点 ✓。

⚠ 这些用例**不开游戏、不碰 Qt、不读配置** ✓ —— 这是它全部的价值：让"一次动作值多少"
  这件事可以被**断言** ✓（塞进 9881 行的 `agent.py` 就没这个福气了 ✗）。
⚠ 表里的数都是"口径验证"✗ 不是"标定结果" ✓（跳攻击带那两个数要现场量 ✓）。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decision.engage import DEFAULT_PROFILE, engage_score        # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


#: 一个"像标飞"的档案：盲区 30、攻击到 120、竖直够 40、跳带 到 +60、跳收益 0.6
P = {"min_dist": 30.0, "max_dist": 120.0, "up": 40.0, "down": 30.0,
     "jump_band": (0.0, 60.0), "jump_benefit": 0.6,
     "back_penalty": 0.5, "cd_penalty": 0.5, "big_bonus": 2.0}


def t_close_hit_is_zero():
    """⭐ **贴脸（盲区内）必须 0** —— 标飞特性①（挥拳几乎没输出 ✓）。"""
    check(engage_score(P, 10.0) == 0.0, "盲区内（距 10 < 30）该判 0：%r"
          % (engage_score(P, 10.0),))
    check(engage_score(P, 29.9) == 0.0, "盲区边界内该判 0：%r"
          % (engage_score(P, 29.9),))
    check(engage_score(P, 30.0) == 1.0, "刚好到盲区下界 ⇒ 进框 ⇒ 1：%r"
          % (engage_score(P, 30.0),))
    # ⚠ 关键：**0 就是 0**（不打折 ✓ 别给 0.1 这种"假希望"✗）
    check(engage_score(P, 10.0, big=True, can_jump=True) == 0.0,
          "盲区内不该因为「大怪 / 能跳」就变成正数（决策会来回改道 ✗）：%r"
          % (engage_score(P, 10.0, big=True, can_jump=True),))


def t_in_box_is_one():
    """框内 ⇒ 1.0 ✓；超出 ⇒ 0 ✓；竖直够不着 ⇒ 0 ✓。"""
    check(engage_score(P, 100.0) == 1.0, "框内该判 1：%r" % (engage_score(P, 100.0),))
    check(engage_score(P, 120.0) == 1.0, "框的上界也算框内：%r" % (engage_score(P, 120.0),))
    check(engage_score(P, 121.0) == 0.0, "超出上界且不能跳 ⇒ 0：%r"
          % (engage_score(P, 121.0),))
    check(engage_score(P, 100.0, dy=41.0) == 0.0, "竖直够不着（41 > up 40）⇒ 0：%r"
          % (engage_score(P, 100.0, dy=41.0),))
    check(engage_score(P, 100.0, dy=-30.0) == 1.0, "竖直刚好够（-30 = down）⇒ 1：%r"
          % (engage_score(P, 100.0, dy=-30.0),))
    check(engage_score(P, 100.0, dy=999.0, ) == 0.0, "太高 ⇒ 0 ✓")
    # 负数 = 不限（同 `attack_up/down_dist` 的口径 ✓）
    free = dict(P, up=-1.0, down=-1.0)
    check(engage_score(free, 100.0, dy=999.0) == 1.0,
          "`up` 为负 = 不限 ⇒ 再高也算够得着：%r" % (engage_score(free, 100.0, dy=999.0),))


def t_jump_band():
    """⭐⭐ 标飞特性②：**跳攻击带**内 ⇒ 按 `jump_benefit` 打折（且**必须先标定** ✓）。"""
    check(engage_score(P, 150.0, can_jump=True) == 0.6,
          "跳带内（max 120 + 0..60）⇒ = jump_benefit：%r"
          % (engage_score(P, 150.0, can_jump=True),))
    check(engage_score(P, 180.0, can_jump=True) == 0.6, "跳带上界内 ⇒ 同一个收益：%r"
          % (engage_score(P, 180.0, can_jump=True),))
    check(engage_score(P, 181.0, can_jump=True) == 0.0, "出跳带 ⇒ 0：%r"
          % (engage_score(P, 181.0, can_jump=True),))
    # ⚠⚠ **没标定（`can_jump=False`）⇒ 一律 0** ✗（不许自己假设"跳得过去" ✓）
    check(engage_score(P, 150.0, can_jump=False) == 0.0,
          "没量出跳攻击带时不该把「跳」算进收益（老行为等价 ✓）：%r"
          % (engage_score(P, 150.0, can_jump=False),))
    # 默认档案**还没有**跳攻击带 ⇒ 绝不能凭空多出收益
    check(engage_score(None, 150.0, can_jump=True) == 0.0,
          "默认档案（`jump_band=(0,0)`）⇒ 跳带为空 ⇒ 0：%r"
          % (engage_score(None, 150.0, can_jump=True),))
    check(engage_score(P, 150.0, dy=999.0, can_jump=True) == 0.0,
          "跳过去**竖直也够不着** ⇒ 0（跳不是万能的 ✓）：%r"
          % (engage_score(P, 150.0, dy=999.0, can_jump=True),))


def t_penalties_and_clamp():
    """折扣：背后 / 受击 CD / 大怪 ✓；结果 **clamp 到 0..1** ✓（系数配歪也不许越界 ✗）。"""
    check(engage_score(P, 100.0, in_back=True) == 0.5,
          "背后（要转身）⇒ × back_penalty：%r" % (engage_score(P, 100.0, in_back=True),))
    check(engage_score(P, 100.0, on_cd=True) == 0.5,
          "目标在受击 CD 里 ⇒ × cd_penalty：%r" % (engage_score(P, 100.0, on_cd=True),))
    check(abs(engage_score(P, 100.0, in_back=True, on_cd=True) - 0.25) < 1e-9,
          "两个折扣该相乘：%r"
          % (engage_score(P, 100.0, in_back=True, on_cd=True),))
    check(engage_score(P, 100.0, big=True) == 1.0,
          "大怪 ×2 但 **clamp 到 1**：%r" % (engage_score(P, 100.0, big=True),))
    check(engage_score(P, 150.0, big=True, can_jump=True) == 1.0,
          "跳带内 ×2 也不许超过 1：%r"
          % (engage_score(P, 150.0, big=True, can_jump=True),))
    # 系数配歪（> 1 的 jump_benefit）⇒ 仍然不许越界
    wonky = dict(P, jump_benefit=5.0)
    check(engage_score(wonky, 150.0, can_jump=True) == 1.0,
          "`jump_benefit` 配成 5 也该被 clamp 到 1：%r"
          % (engage_score(wonky, 150.0, can_jump=True),))


def t_bad_input_never_raises():
    """参数坏 / 档案缺键 ⇒ **0 或默认值** ✓ **绝不抛** ✗（这函数会在每拍被调用 ✓）。"""
    for bad in (None, "x", {}, [1], float("nan")):
        try:
            engage_score(bad, 100.0)
        except Exception as e:                                   # noqa: BLE001
            raise AssertionError("档案=%r 抛了 %s: %s" % (bad, type(e).__name__, e))
    for bad_d in (None, "x", [], {"a": 1}):
        check(engage_score(P, bad_d) == 0.0, "距离=%r 该判 0（不猜 ✗）：%r"
              % (bad_d, engage_score(P, bad_d)))
    check(engage_score(P, -100.0) == 1.0,
          "负距离按**绝对值**看（同 `abs` ✓）：%r" % (engage_score(P, -100.0),))
    # 缺键 ⇒ 用默认（不许 KeyError ✗）；但默认 `max_dist=0` ⇒ 打不着 ⇒ 0 ✓
    check(engage_score({"min_dist": 10.0}, 100.0) == 0.0,
          "档案只有一半键 ⇒ 用默认补（`max_dist=0` ⇒ 0 ✓ 不炸 ✓）：%r"
          % (engage_score({"min_dist": 10.0}, 100.0),))
    check(engage_score(dict(P, up="x"), 100.0, dy=10.0) == 0.0,
          "`up` 是坏值 ⇒ 当够不着（宁缺勿错 ✓）：%r"
          % (engage_score(dict(P, up="x"), 100.0, dy=10.0),))
    check(DEFAULT_PROFILE["jump_band"] == (0.0, 0.0),
          "默认档案的跳攻击带必须是**空**（没量过 ⇒ 不许凭空多出收益 ✗）：%r"
          % (DEFAULT_PROFILE["jump_band"],))


TESTS = (
    ("⭐⭐ 贴脸（盲区内）= 0（标飞特性①：挥拳≈没输出）且不打折",
     t_close_hit_is_zero),
    ("框内 = 1 / 超出 = 0 / 竖直够不着 = 0（负数 = 不限 ✓）", t_in_box_is_one),
    ("⭐⭐ 跳攻击带：带内 = jump_benefit（未标定 ⇒ 一律 0，不替用户假设 ✓）",
     t_jump_band),
    ("折扣（背后 / 受击CD / 大怪）与 clamp 到 0..1", t_penalties_and_clamp),
    ("参数坏 / 档案缺键 ⇒ 0 或默认，绝不抛", t_bad_input_never_raises),
)


def main():
    ok = 0
    for name, fn in TESTS:
        try:
            fn()
            ok += 1
            print("  [ OK ] %s" % name)
        except AssertionError as e:
            print("  [FAIL] %s\n         %s" % (name, e))
    print("\n输出收益口径自检：%d/%d 通过" % (ok, len(TESTS)))
    return 0 if ok == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
