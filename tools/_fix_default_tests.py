"""把三条"默认值"用例从**读全局 `ag.settings`** 改成**读新建的 `DecisionSettings()`**（一次性件）。

为什么（2026-10-08 ✓ 三条红查清）：
  `t_align_params` / `t_climb_align_near_px` / `t_climb_align_gap` 断的是"**默认值** ≠ `route` 常量"，
  可它们读的是**全局** `ag.settings` ✗ —— 而整套跑时，前面有谁把**项目配置**灌进了全局
  （工作树里 `project.yaml` 也动过 ✓）⇒ 读回来的是**项目里的值**（1 / 200 ✓）⇒ 假红 ✗。
  ⇒ "默认值"就该读**新建实例**：`DecisionSettings()` ✓ 与顺序无关 ✓（口径也更对 ✓）。
"""
from pathlib import Path

P = Path("tools/selftest_decision.py")
src = P.read_text(encoding="utf-8")

REPL = [
    # ① t_align_params（第 3119 行附近 ✓）
    ('check(int(getattr(settings, "climb_align_gap_ms", -1))',
     'check(int(getattr(ag.DecisionSettings(), "climb_align_gap_ms", -1))'),
    # ② t_climb_align_near_px
    ('check(int(getattr(ag.settings, "climb_align_near_px", -1)) == int(route.NEAR_PX),',
     'check(int(getattr(ag.DecisionSettings(), "climb_align_near_px", -1)) == int(route.NEAR_PX),'),
    # ③ t_climb_align_gap
    ('check(int(getattr(ag.settings, "climb_align_gap_ms", -1))\n'
     '          == int(route.TAP_PERIOD_S * 1000),',
     'check(int(getattr(ag.DecisionSettings(), "climb_align_gap_ms", -1))\n'
     '          == int(route.TAP_PERIOD_S * 1000),'),
]

for i, (old, new) in enumerate(REPL, 1):
    n = src.count(old)
    print("① ② ③"[i - 1], "命中", n, "处")
    if n == 1:
        src = src.replace(old, new, 1)
    elif n == 0:
        print("   ⚠ 锚点没找到（源码变了？）")

P.write_text(src, encoding="utf-8")
print("写回 ✓")
