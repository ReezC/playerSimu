# -*- coding: utf-8 -*-
"""「**这个位置打这一只怪，一次动作值多少输出**」的唯一口径 —— 标飞那三条需求的第一块 ✓。

来源（用户 2026-10-11 ✓）：要给「标飞」职业做 AI，它的三条特性是 ——
  ① **近距离挥拳几乎 0 输出**（所以必须保持距离 ✓ 现有 `min_attack_dist` 就是这个"盲区" ✓）；
  ② **跳**可以"舍近求远打怪"（跳到远处那只 ⇒ 一跳带进攻击框 ✓）；
  ③ **需要频繁评估最佳输出位置并前往**（不是定时/规则，而是**按输出效率选点** ✓）。
⇒ 三条都指向同一个缺失概念：**一次攻击动作值多少输出** ✓（现在的代码里没有它 ✗）。

**为什么单独一个模块，而不是塞进 `decision/agent.py`** ✗（9881 行 ✓ 策略/输出序列/路由胶水
全搅在一起 ⇒ 写进去就没人敢改 ✓）：
  · 它必须是**纯函数** ✓ —— 合成怪框 + 几何参数 ⇒ 断言分数 ✓ **不用开游戏** ✓（见
    `tools/selftest_engage.py` 那张表 ✓）；
  · 它是"最佳输出位置"决策器（③）的**输入** ✓ ⇒ 先把它钉住，再谈选点 ✓。

⚠⚠ **本模块这一轮不被 agent import** ✗（**只立口径 + 用例** ✓ 零行为改动 ✓）：
  接线要等「**跳跃标定**」出来（`jump_band` / `jump_benefit` 那两个数得**量**出来 ✓，
  见 `docs/职业档案-标飞.md` 的步骤 ✓）。在此之前它是"可单测的落点" ✓ 不是"已生效的策略" ✗。
⚠ 所有系数都**从职业档案来** ✗（`DEFAULT_PROFILE` 只是"先给个能跑的"✗ 不是量出来的 ✓）——
  分层：**角色能力**（本档案 ✓ 距离带 / 跳攻击带 / 权重）／**地图事实**（`zones.json` ✓）
  ／**这块平台允不允许打**（`battle_zones` 的 `can_fight` / `fight_dst` ✓）⇒ 别混 ✗。
"""
#: 默认档案：**标飞的起点值** ✓（全部可被 `profiles/<职业>.yaml` 覆盖 ✓）。
#: ⚠ 这些数字是"先给个能跑的"✗ 不是标定结果 ✓（`jump_band` 默认 (0, 0) = **还没有跳攻击带** ✓
#:   ⇒ 现在跑起来它等价于"只能正常攻击"✓ 与老行为一致 ✓ 不会偷偷改输出 ✗）。
DEFAULT_PROFILE = {
    "min_dist": 0.0,        # 盲区下界（贴得比它更近 ⇒ **打不到** ✓ 标飞特性①）
    "max_dist": 0.0,        # 正常攻击的上界
    "up": 0.0,              # 竖直：往上够多少（**负数 = 不限** ✓ 同现有键口径）
    "down": 0.0,            # 竖直：往下够多少（负数 = 不限 ✓）
    "jump_band": (0.0, 0.0),   # 跳攻击带：`(相对 max_dist 的外扩起, 止)` ✓ 标定后填 ✓
    "jump_benefit": 0.6,    # 跳过去打的收益（打折：要花跳跃时间 ✓）
    "back_penalty": 0.5,    # 怪在**背后**（得先转身 ✓）
    "cd_penalty": 0.35,     # 目标正在**受击 CD** 里（打了白打 ✓）
    "big_bonus": 1.0,       # 大怪加成（1.0 = 不加成 ✓）
}


def _v_ok(dy, up, down):
    """竖直上够不够（`dy` = 怪比脚底高多少 ✓ **负数 = 不限** ✓ 同 `attack_up/down_dist` ✓）。"""
    try:
        _dy, _up, _dn = float(dy), float(up), float(down)
    except (TypeError, ValueError):
        return False
    if _up >= 0 and _dy > _up:
        return False
    if _dn >= 0 and (-_dy) > _dn:
        return False
    return True


def engage_score(profile, dist, dy=0.0, in_back=False, on_cd=False,
                 big=False, can_jump=False):
    """这个距离/高度上打一下值多少：`0.0 .. 1.0` ✓（**纯函数** ✓ 拿不到就 0 ✓ 不抛 ✓）。

    口径（与「攻击范围框」**同一条几何** ✓ —— 别在这儿另发明一套 ✗）：
      ① **盲区内 / 超出攻击框 / 竖直够不着** ⇒ **0** ✓（标飞特性①：贴脸挥拳≈没输出 ✓）；
         跳到带外打不着 ⇒ **一律 0** ✗（**不打折** ✓ —— 0.1 这种"假希望"会让决策来回改道 ✓）；
      ② **框内** ⇒ **1.0** ✓ 再乘折扣（背后 `back_penalty` ✓ 受击 CD `cd_penalty` ✓
         大怪 `big_bonus` ✓）；
      ③ **跳攻击带内**（`can_jump=True` ✓ 且落在 `(max+j0, max+j1)` ✓）⇒ 从
         `jump_benefit` 起再乘折扣 ✓ —— 标飞特性② ✓；
      ④ 结果 **clamp 到 0..1** ✓（系数配歪了也不许越界 ✗）。
    ⚠ `can_jump` 默认 **False** ✓ ⇒ 没量出跳攻击带之前，这个函数**不会**把"跳"算进收益 ✓
      （老行为等价 ✓ 见模块头 ✓）。
    """
    try:
        p = dict(DEFAULT_PROFILE)
        p.update(profile or {})
        d = abs(float(dist))
        lo, hi = float(p["min_dist"]), float(p["max_dist"])
    except (TypeError, ValueError, KeyError):
        return 0.0                               # 参数本身坏了 ⇒ 0（宁缺勿错 ✓）
    if hi <= 0.0:
        return 0.0                               # 还没配攻击距离 ⇒ 打不着 ⇒ 0 ✓（不猜 ✗）
    base = 0.0
    if lo <= d <= hi and _v_ok(dy, p.get("up", 0.0), p.get("down", 0.0)):
        base = 1.0
    elif can_jump:
        _band = p.get("jump_band") or (0.0, 0.0)
        try:
            j0, j1 = float(_band[0]), float(_band[1])
        except (TypeError, ValueError, IndexError):
            j0, j1 = 0.0, 0.0
        if j1 > 0.0 and (hi + j0) <= d <= (hi + j1) \
                and _v_ok(dy, p.get("up", 0.0), p.get("down", 0.0)):
            base = float(p.get("jump_benefit", 0.6) or 0.0)
    if base <= 0.0:
        return 0.0
    try:
        if in_back:
            base *= float(p.get("back_penalty", 1.0))
        if on_cd:
            base *= float(p.get("cd_penalty", 1.0))
        if big:
            base *= float(p.get("big_bonus", 1.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, base))
