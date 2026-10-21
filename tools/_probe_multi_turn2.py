# -*- coding: utf-8 -*-
"""探针 v2（只读）：**多点巡逻"正走着"（有走位任务）+ 背后有怪 + 输出CD** 那几拍。

v1 只摆了"没任务"的静态场景，没复现出现场；现场"被**寻路**朝向拽回去"里的"寻路"=
**走位任务在跑**（`_climb` 非空 ✓ 多点巡逻去下一个点位就是它 ✓）⇒ 这一版把任务摆上。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import decision.agent as ag          # noqa: E402

ag.DecisionSettings.save = lambda self: None
import tools.selftest_decision as S   # noqa: E402
from decision import jobs as jobmod

STEP = 1.0 / 15.0


def run(in_cd=True, ticks=16, mob_x=460.0, job_x=(300.0, 700.0), hold_ms=300,
        leave_after=None):
    from decision import jobs as jobmod
    s = S.fresh_settings(attack_dist=120.0)
    s.attack_cd = 500
    s.min_turn_hold_ms = hold_ms
    s.strategy = "multi"
    s.multi_spots = [{"kind": "set", "name": "A"}, {"kind": "set", "name": "B"}]
    s.station_spot = {"kind": "set", "name": "A"}
    h = S.Harness(s)
    # `leave_after`（秒）之后那只怪**走出视野** ⇒ 看"它一走，任务是不是立刻把人拽回去" ✓
    h.mobs_fn = lambda t: ([] if (leave_after is not None and t >= leave_after)
                           else [S.Mob(id=7, x=mob_x, y=500.0, w=40.0, h=40.0, conf=0.9)])
    with h._patched():
        a = h.agent
        h.clock0 = h.clock.t
        a.set_facing(1)
        a._last_output = (h.clock.t if in_cd else h.clock.t - 10.0)
        # 摆一条**走位任务**（多点巡逻去下一个点位就是它 ✓）：往右走 ⇒ 和"朝背后怪转"相反 ✓
        a.start_climb(jobmod.WalkJob("乙平台", [(job_x[0], 510.0, job_x[1], -208.0, "1")]))
        print("   （任务起跑：climb=%s）" % ("有" if a._climb is not None else "-"))
        for i in range(ticks):
            a.tick(h.ws())
            print("  t=%7.3f state=%-6s tgt=%-4s face=%+d keys=%-10s cd=%.2f turn_at=%s until=%s climb=%s"
                  % (h.clock.t, a.state, a._target_id, a.facing,
                     ",".join(sorted(k for k in a.keys.pressed()
                                     if k in ("left", "right", "up", "down"))) or "-",
                     a.out_cd_left_s(now=h.clock.t),
                     (round(a._turn_at, 3) if a._turn_at else None),
                     (round(a._turn_until, 3) if a._turn_until else None),
                     "有" if a._climb is not None else "-"))
            h.clock.t += STEP


print("=== ① 走位任务在跑 + 背后 40px 有怪 + 正在输出CD（min_turn_hold=300ms）===")
run(True)
print("=== ② 对照：同上但早就出了 CD ===")
run(False)
print("=== ③ 同上（在 CD 里）+ 第 0.3 秒起那只怪**走掉** ===")
run(True, leave_after=0.3)
