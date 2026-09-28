# -*- coding: utf-8 -*-
"""**寻路探针** —— 拿**真实地图**把一条路线解析一遍，把人眼要看的东西摆出来。

用户 2026-09-27 要求："可以拿**石人寺院**地图测试寻路" ✓。

为什么要有它：`selftest_*` 钉的是"**某一处判据**对不对" ✓；而"**这张图到底能不能走过去**"要把
地形 + 集合 + 边 + **执行器参数**串起来看才知道 ✓ —— 这个串法**只有一处**：`route.plan_jobs`
（面板「命令前往」与 agent 的路径解析器都走它 ✓）。这里就把它的结果摆出来 + 做几项体检 ✓。

用法：
    python -m tools.route_probe                       # 默认 石人寺院III：一楼 → 顶层
    python -m tools.route_probe 106010105 底层 顶层
    python -m tools.route_probe --all                 # 该图**所有集合两两**可达性体检

退出码：0 = 体检全过；1 = 有项没过（可直接进 CI ✓）。
配套看图：`python -m tools.map_terrain_view <map_id>`（把地形画在底图上 ✓）。
"""
from __future__ import annotations

import argparse
import sys

#: 默认那张图 —— **石人寺院III**（`projects/石人寺院III/project.yaml` ✓）：地形 + 集合都齐 ✓
DEFAULT_MAP = "106010105"
DEFAULT_SRC, DEFAULT_DST = "一楼", "顶层"


def probe(map_id, src, dst, out=print):
    """解析 `src → dst` 并体检 → `[(ok, 说明), …]`（**纯函数**：用例直接调它 ✓）。"""
    from core import mapdata
    from core import zones as zmod
    from decision import route

    checks = []
    t = mapdata.load(map_id)
    z = zmod.load(map_id)
    if t is None or z is None:
        return [(False, "地形/集合读不到（%s）：terrain=%s zones=%s"
                 % (map_id, t is not None, z is not None))]
    names = sorted(z.sets)
    out("地图 %s：集合 %d 个（%s）　边 %d 条"
        % (map_id, len(names), "、".join(names), len(z.edges)))
    for name in (src, dst):
        checks.append((name in z.sets,
                       "集合「%s」在这张图里圈过" % name))
    if not all(ok for ok, _ in checks):
        return checks

    plan = route.plan_jobs(t, z, src, dst, tol_px=route.ALIGN_TOL_PX, hold_ms=250)
    path = list(plan.get("path") or [])
    jobs = list(plan.get("jobs") or [])
    checks.append((bool(jobs), "解析出 %d 步：%s"
                   % (len(jobs), " → ".join(path) or "（空）")))
    if not jobs:
        why = str(plan.get("why") or "").strip()
        checks.append((False, "解析不出来的原因：%s" % (why or "（没说）")))
        return checks

    ladders = set(str(v) for v in zmod.ladder_ids(t).values())
    out("步骤：")
    for i, j in enumerate(jobs):
        kind = route.JOB_KINDS.get(type(j).__name__, type(j).__name__)
        label = zmod.kind_label(kind) if kind in zmod.EDGE_KINDS else type(j).__name__
        bits = ["第 %d 步 %s" % (i + 1, label)]
        lid = getattr(j, "ladder_id", None)
        if lid is not None:
            bits.append("绳=%s" % lid)
            checks.append((str(lid) in ladders,
                           "第 %d 步的绳号「%s」在这张图的绳梯编号里" % (i + 1, lid)))
            # ⚠ 执行器存的属性名是 **`dir`**（构造参数叫 `direction` ✓）—— 探针第一版读错了，
            #   当场把两条能走的爬绳报成"没说清上下" ✗（体检项自己也要对 ✓）。
            d = int(getattr(j, "dir", getattr(j, "direction", 0)) or 0)
            bits.append("方向=%s" % ("上爬" if d > 0 else ("下爬" if d < 0 else "?")))
            checks.append((d != 0, "第 %d 步说清了往上还是往下爬" % (i + 1)))
        for attr, cn in (("src_set", "起点集合"), ("dst_set", "目标集合")):
            v = str(getattr(j, attr, "") or "")
            if v:
                bits.append("%s=%s" % (cn, v))
                checks.append((v in z.sets, "第 %d 步的%s「%s」存在" % (i + 1, cn, v)))
        for attr, cn in (("dst_x", "目标 x"), ("dst_y", "目标面 y")):
            v = getattr(j, attr, None)
            if v is not None:
                bits.append("%s=%s" % (cn, ("%.0f" % v) if isinstance(v, (int, float)) else v))
        out("  " + "　".join(bits))
    return checks


def probe_all(map_id, out=print):
    """该图**所有集合两两**可达性体检 → `[(ok, 说明), …]`（看"缺哪条边"最直观 ✓）。"""
    from core import mapdata
    from core import zones as zmod
    from decision import route

    t, z = mapdata.load(map_id), zmod.load(map_id)
    if t is None or z is None:
        return [(False, "地形/集合读不到：%s" % map_id)]
    names = sorted(z.sets)
    bad = []
    for a in names:
        for b in names:
            if a == b:
                continue
            plan = route.plan_jobs(t, z, a, b, tol_px=route.ALIGN_TOL_PX, hold_ms=250)
            if not plan.get("jobs"):
                bad.append((a, b, str(plan.get("why") or "").strip()))
    total = len(names) * (len(names) - 1)
    ok = total - len(bad)
    out("两两可达性：%d/%d" % (ok, total))
    for a, b, why in bad:
        out("  ✗ %s → %s：%s" % (a, b, why))
    return [(not bad, "所有集合两两可解析（%d 对）" % total)]


def main() -> int:
    ap = argparse.ArgumentParser(description="寻路探针（真实地图上解析一条路线 + 体检）")
    ap.add_argument("map_id", nargs="?", default=DEFAULT_MAP)
    ap.add_argument("src", nargs="?", default=DEFAULT_SRC)
    ap.add_argument("dst", nargs="?", default=DEFAULT_DST)
    ap.add_argument("--all", action="store_true", help="所有集合两两可达性体检")
    a = ap.parse_args()

    checks = probe_all(a.map_id) if a.all else probe(a.map_id, a.src, a.dst)
    failed = [d for ok, d in checks if not ok]
    for ok, d in checks:
        print(("  [OK] " if ok else "  [!!] ") + d)
    print("\n%d/%d 项通过" % (len(checks) - len(failed), len(checks)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
