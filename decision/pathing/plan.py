"""把「起点集合 → 终点集合」变成一串执行器任务（从 `decision/route.py` 切出）。

**一处实现，两边共用**：界面「命令前往」与实时线程的路径解析器都调 `plan_jobs` ✓
—— 各写一份迟早分叉（出现"面板能下发、休息却走不过去"这种最难查的怪事 ✗）。

⚠ 依赖方向：`pathing` → `jobs`（造任务要执行器类）；`jobs` **不许**反过来 import
`pathing`（否则成环 ✓）。
"""

from decision.jobs.climb import ClimbJob
from decision.jobs.drop import DropJob
from decision.jobs.jump import JumpJob
from decision.jobs.params import (ALIGN_TOL_PX, JUMP_START_PX, JOB_KINDS, STALL_S)
from decision.jobs.portal import PortalJob
from decision.jobs.walk import WalkJob
from decision.pathing.cost import hop_cost
from decision.pathing.spans import _set_spans


def job_label(job):
    """当前**执行器**的一行文本（界面用）：`下跳(drop):{详细内容}`；没有 job 给空串 ✓。

    用户 2026-09-27 的要求（原话）："你能把当前的执行器写一下吗？例如格式：`下跳(drop):{详细内容}`"
    —— 卡住的时候，第一件想知道的事就是"**现在跑的是哪个执行器、它卡在哪一步**" ✓；
    原来画面上只有一句自由文本（`agent.current_goto_note` ✓），**看不出是哪一类执行器** ✗。

    「详细内容」= 这个 job 的 `note`（各执行器自己维护的那句话：对齐差几像素 / 谁不在哪块面上 /
    爬不动了 / 到达了 …✓）；note 空的时候退回 `phase`（至少告诉你在哪个相位 ✓）。
    中文名走 `zones.kind_label`（**一处口径** ✓）；类名认不出来时**退回类名**（不猜中文 ✗）。
    """
    if job is None:
        return ""
    from core import zones as zones_mod

    kind = JOB_KINDS.get(type(job).__name__, "")
    name = zones_mod.kind_label(kind) if kind else type(job).__name__
    note = str(getattr(job, "note", "") or "").strip()
    if not note:
        note = str(getattr(job, "phase", "") or "").strip()
    return ("%s:%s" % (name, note)) if note else name


def pick_edge(terrain, z, a, b, at=None):
    """`a → b` **这一步用哪条边** —— 同一对集合有多条时的**唯一判据**。

    ⚠ **路径本身不在这儿定**：集合序列由 `core.zones.find_path`（无权 BFS =
      **最少段数**）决定 ✓ —— 这里只决定"**同一段内走哪条**"，**距离只在同一段内比** ✓
      （用户 2026-09-27 明确："路径仍按最少段数、距离只在同一段内比"）。

    判据（按顺序）：
      1. **只留能用的边**：不在 `zones.EDGE_KINDS` 里的（老数据残留的 `unconfirmed` ✓；
         ⚠ `portal` **2026-10-06 起是正经通行方式了** ✓ 见 `EDGE_KINDS` ✓）、以及
         **建不出任务**的（爬：绳找不到 / 上下说不清；走·跳·下跳：目标集合没有非墙 foothold；
         传送点：门找不到 / **跨图** ✓）⇒ **不参选** ✓（不然会挑出一条根本执行不了的边 ✗）；
      2. 有玩家位置 ⇒ 按 **`hop_cost`**（纯距离 ✓）取**最小**的那条；
         代价**算不出来**的边排到最后 ✓；
      3. 并列 / 没有位置 / 全算不出 ⇒ 退回**老口径**：类型优先级
         （`zones.KIND_PRIORITY`：走 > 爬 > 下跳 > 跳）→ **文件顺序** ✓
         （`edges_between` 的排序就是它 ⇒ 取第一个 ✓ —— 稳定、可预期 ✓）。

    「入绳点」= 真正开始爬的那个点：**上爬在下端**（`L.bottom`）、**下爬在上端**（`L.top`）
    —— 那才是"要走到哪儿才能上绳"，比到绳中点的距离贴近实际 ✓（它是 `hop_cost` 里
    爬绳那一项的一半 ✓）。
    """
    from core import zones as zones_mod

    hit = zones_mod.edges_between(z, a, b)
    if not hit:
        return None
    usable = []
    for e in hit:
        if str(e.get("kind") or "") not in zones_mod.EDGE_KINDS:
            continue
        try:
            job_for_edge(terrain, z, e)      # 只"试造"一下（不要它的结果 ✓）
        except Exception:                    # noqa: BLE001
            continue                         # 建不出来 ⇒ 这条不参选 ✓
        usable.append(e)
    if not usable:
        return None
    if at is None:
        return usable[0]                     # 老口径：类型优先级 → 文件顺序 ✓
    best, best_cost = usable[0], None
    for e in usable:
        c = hop_cost(terrain, z, e, at)
        if c is None:
            continue                         # 算不出 ⇒ 排到最后 ✓
        if best_cost is None or c < best_cost:
            best, best_cost = e, c           # 严格小于 ⇒ 并列时保留**先出现**的那条 ✓
    return best if best_cost is not None else usable[0]


def path_cost(terrain, z, src_sets, dst_sets, at=None):
    """「我站的集合 → 目标的集合」的**寻路距离**（世界像素）；算不出来给 `None`。

    用户 2026-09-27 的需求（原话）："优化：锁定目标优先级要按「**寻路距离**」最近，而不是旧的
    应该是**绝对距离**" —— 挑要打哪只怪时，"哪只更值得先打"该看**走过去多贵** ✓，而不是画面上
    的直线距离（隔着平台 / 要绕一根绳的怪，直线可能很近、实际要走很远 ✗）。

    口径（**完全沿用择路那一套，一格都不新造** ✓）：
      · **路径本身** = `core.zones.find_path`（无权 BFS = **最少段数** ✓ —— 用户 2026-09-27 定的
        "路径仍按最少段数、距离只在同一段内比" ✓）；
      · **每一段的代价** = `pick_edge` 选中的那条边的 `hop_cost`（纯距离 ✓，四类边的表见它那儿 ✓）；
      · **总代价** = 逐段相加（世界像素 ⇒ 天然可比 ✓）。

    ⚠ 这**不是**"加权最短路"：先最少段数、再在段内比距离 —— 与 `plan_jobs` / `pick_edge`
      同一套口径 ✓（约定 39：**别把 `hop_cost` 塞进 `find_path`** ✗）。
    ⚠ **任何一段算不出来（`None`）⇒ 整条给 `None`**（**不猜** ✓，调用方退回老口径 ✓）；
      没有玩家位置（`at is None`）时必然算不出 ⇒ 也是 `None` ✓。
    ⚠ 来源/目标都可能**不止一个集合**（脚下一般 1 个、怪那边 1~3 个 ✓）⇒ 去重后试组合，
      取**能算出来的那些里最小的** ✓（同一集合记 0 ✓）。
    ⚠ **开销**：一次 = 一次 BFS + 每段一次 `pick_edge`（里面还要"试造 job" ✗ 偏重）⇒
      **别挂进每拍、更别对每只怪都算** ✗ —— 同 `route_plan` 的纪律
      （`docs/寻路设计.md` 明说 BFS 不在帧循环里 ✓）。现在只在"**要挑目标**"那一刻、
      对当时候选的怪算一次 ✓（`agent._nearest`）。
    """
    from core import zones as zones_mod

    srcs = [str(s) for s in (src_sets or []) if str(s)]
    dsts = [str(d) for d in (dst_sets or []) if str(d)]
    if z is None or terrain is None or not srcs or not dsts:
        return None
    best = None
    seen = set()
    for a in srcs:
        for b in dsts:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            if a == b:
                return 0.0                     # 同一个集合 ⇒ 一步都不用走 ✓
            try:
                path, _why = zones_mod.find_path(z, a, b)
            except Exception:                  # noqa: BLE001
                continue
            if not path or len(path) < 2:
                continue
            total, ok = 0.0, True
            for i in range(len(path) - 1):
                e = pick_edge(terrain, z, path[i], path[i + 1], at)
                if e is None:
                    ok = False
                    break
                c = hop_cost(terrain, z, e, at)
                if c is None:
                    ok = False
                    break
                total += float(c)
            if ok and (best is None or total < best):
                best = total
    return best


def plan_jobs(terrain, z, src, dst, tol_px=ALIGN_TOL_PX, hold_ms=250, stall_s=None,
              jump_start_px=None, at=None):
    """「从 `src` 走到 `dst`」→ `{"path", "jobs", "why", "here"}`（**一处实现，两边共用**）。

    谁在用：
      · `gui/route_panel._command_first_step`（「命令前往」下整条路径）；
      · `gui/live_thread` 给 agent 注入的**路径解析器**（「定点休息」要 agent 自己走过去）。
    ⚠ 两边**必须同一套判据** —— 各写一份迟早分叉（出现"面板能下发、休息却走不过去"
    这种最难查的怪事 ✗），所以判据只放这里。

    判据（就是「命令前往」原来那套 ✓）：
      · `zones.find_path` 找不到路 ⇒ `jobs=[]`，`why` 说清（它会把"从起点能走到的全部
        集合"列出来 —— 那就是缺边的位置，比一句"没有路径"有用得多）；
      · 每一步**走哪条边**：`pick_edge`（同一对集合有多条绳时**挑最近的** ✓）；
      · 路径里出现**还没做执行器的通行方式**（现在只剩「传送门」；「跳」2026-09-27 已接 ✓）
       ⇒ **整条都不造**并说清是第几步、靠什么（半途停半空比不下更糟 ✗）；
      · 任何一段造任务抛 `ValueError`（绳找不到 / 说不清上下 / 落点找不到）⇒ **抛给
        上层**，让它如实报警或跳过，**不许猜**。
    `src == dst` ⇒ `here=True`（人已经在目标上，不用走 ✓）。
    `at` = **玩家世界坐标** `(x, y)`：给 `pick_edge` 用来挑最近的绳 ✓（None ⇒ 按文件顺序，
    老行为；两个调用方都传得起：面板有定位读数、实时线程有 WorldState ✓）。
    返回的 `jobs` 直接喂 `agent.start_route`；`edges` = **这一步实际选中的那些边**
    （给"命令前往"那行显示用 —— 显示和执行必须是同一条边 ✓）。
    """
    from core import zones as zones_mod

    src, dst = str(src or ""), str(dst or "")
    path, why = zones_mod.find_path(z, src, dst)
    if not path:
        return {"path": [], "jobs": [], "why": why, "here": False, "edges": []}
    if len(path) < 2:
        return {"path": path, "jobs": [],
                "why": "已经在「%s」上了，不用走" % dst, "here": True, "edges": []}
    kinds = []
    #: 每段**选中的那条边**的代价（世界像素；`None` = 算不出来 ⇒ 那一段是按老口径
    #: 的类型优先级取的 ✓）—— 给界面解释"为什么走这条"用 ✓（`docs/寻路设计.md`「择路」✓）。
    costs = []
    for a, b in zip(path, path[1:]):
        e = pick_edge(terrain, z, a, b, at=at)
        if e is None:
            return {"path": path, "jobs": [], "edges": kinds, "costs": costs,
                    "why": ("%s → %s 之间**没有一条能走的边**（已经不存在的通行方式 / "
                            "建不出任务 / 绳或落点找不到）—— 回「寻路编辑器」看看" % (a, b)),
                    "here": False}
        kinds.append(e)
        costs.append(hop_cost(terrain, z, e, at))
    # 「卡住判定时长(s)」从这里灌进**攀爬 + 跳跃**这三类任务（None ⇒ 模块默认 `STALL_S`=3 ✓）：
    # 它管**上爬"爬不动了"**、**下爬"等落地"**、**跳跃"按跳后没落到目标集合"** 三个判据 ✓
    #（用户 2026-09-27 明确同意把它的范围从"只管攀爬"扩成"攀爬 + 跳跃" ✓）。
    # ⚠ **走 / 下跳都不吃它**（`job_for_edge` 会把这一格摘掉 ✓）：走照旧用它自己的
    #   `STALL_S`，卡住由「寻路超时时间」兜底 ✓。
    _stall = STALL_S if stall_s is None else float(stall_s)
    # 「起跳距离(px)」（**临时参数**，见 `JUMP_START_PX`）也一起传 —— 只有「跳」用得上，
    # 别的边由 `job_for_edge` 摘掉 ✓（同一个写法：谁不认这个参数谁 pop ✓）。
    _jump_start = JUMP_START_PX if jump_start_px is None else max(
        0.0, float(jump_start_px))
    jobs = [job_for_edge(terrain, z, e, tol_px=int(tol_px), hold_ms=int(hold_ms),
                         stall_s=_stall, jump_start_px=_jump_start)
            for e in kinds]
    return {"path": path, "jobs": jobs, "why": "", "here": False, "edges": kinds,
            "costs": costs}


def _mid_land(terrain, L, mid_y, mid_dir, tol=90.0):
    """「中途跳下」那个高度、朝 `mid_dir` 那一侧，**有没有能落脚的平台** → `(ok, 说明)`。

    为什么要验（2026-10-19 用户现场 ✓）：那两条边的「高度」是**手写的数**，不保证那儿真有平台。
    实测 200010302 的 `2→3` / `3→2` 配 `mid_y=-1024`，而 y 在 **-1100~-950 之间一条 foothold
    都没有**（最近的平台在 -697 和 -1237，各差 327 / 213）⇒ 一跳到那儿就**直接掉回底层** ✗
    （用户："从底层去 3，在爬 L1 去入口时莫名其妙会跳下来"✓）。

    ⚠ 只用现成几何（`terrain.footholds` 的 x1/x2/y1/y2，与 `foothold_below` 同一套数据 ✓），
      不新造尺子 ✓；`tol` = 允许的竖直误差（跳过之后会往下掉一段 ⇒ 给 90px 余量 ✓）。
    """
    try:
        x = float(L.x)
        ys = float(mid_y)
    except (TypeError, ValueError):
        return True, ""                      # 参数不全 ⇒ 不归这条管（别误杀 ✓）
    d = str(mid_dir or "")
    for f in getattr(terrain, "footholds", ()) or ():
        if getattr(f, "is_wall", False):
            continue
        try:
            y1, y2 = float(f.y1), float(f.y2)
            x1, x2 = float(f.x1), float(f.x2)
        except (TypeError, ValueError):
            continue
        if min(abs(y1 - ys), abs(y2 - ys)) > float(tol):
            continue
        if d == "right" and max(x1, x2) <= x + 4.0:
            continue
        if d == "left" and min(x1, x2) >= x - 4.0:
            continue
        return True, "#%s 在 y=%.0f（离配置的高度 %+.0f px）" % (f.fid, y1, y1 - ys)
    return False, ("配置的高度 %.0f 上下 %.0f px、朝%s那一侧**一条平台都没有**"
                   % (ys, tol, {"right": "右", "left": "左"}.get(d, "目标中心")))


def job_for_edge(terrain, z, edge, **kw):
    """从一条「爬」边造出上绳任务 —— **上下由目标那边在绳的上端还是下端判**。

    这条把三样东西接起来（都走现成设施，不自己发明）：
        · `zones.ladder_ids` 找绳（与编辑器画在绳上的编号同一套）；
        · `zones.climb_direction` 判上下（用户 2026-09-26 的规则）；
        · `ClimbJob` 执行。
    判不出来（绳找不到 / 目标不在绳的两端）就抛 ValueError —— 调用方要么报警要么跳过，
    **不许猜**（猜错就是"往反方向爬"，比不做更糟）。

    ⚠ **坐标口径**（2026-09-26 用户澄清，这里曾经改错过一次）：`L.x / L.y1 / L.y2 /
    dst_y` 全来自地形数据，而**地形数据本来就在游戏的世界坐标系里**（例：105090600 的
    fh44 地形 y=-208，它的世界 y 就是 -208）⇒ 玩家读数（= 定位 + 那个"坐标系偏移" ✓）
    与它们**直接比大小**就是对的 ✓。**不要**给这些数再加偏移 —— 加了会把"到达目标平台面"
    从 -208 放宽成 -175（差 33px），看起来像"提前算到了" ✗。
    """
    from core import zones

    kind = edge.get("kind") or ""
    if kind == "drop":
        # 下跳**没有**"卡住"那套判据（它只管"落地了没有"✓）⇒ 把「卡住判定时长」摘掉再传，
        # 不然 `DropJob` 会收到一个不认识的参数 ✗（`plan_jobs` 是给四种边一起灌的 ✓）。
        kw.pop("stall_s", None)
        kw.pop("jump_start_px", None)      # 下跳不吃「起跳距离」（那是「跳」的参数 ✓）
        return drop_job_for_edge(terrain, z, edge, **kw)
    if kind == "walk":
        # ⚠ **走不吃「卡住判定时长(s)」**（用户 2026-09-27 明确："移除对走的相关判定，走用
        # 超时配置兜底"）：那格设置是给**攀爬 / 跳跃**用的 ✓；
        # 走照旧用它自己的常量 `STALL_S`（默认 3s ✓ 老行为不变），另外「寻路超时时间」
        # （`goto_timeout_s`，agent 那一层 ✓）本来就是走的兜底 ✓。
        kw.pop("stall_s", None)
        kw.pop("jump_start_px", None)
        return walk_job_for_edge(terrain, z, edge, **kw)
    if kind == "portal":
        # ⭐ 传送点（用户 2026-10-06 ✓）：**走到门 ⇒ 点按一下 ↑** ✓ 见 `PortalJob` ✓。
        #   ⚠ 它**不吃**「卡住判定时长」/「起跳距离」（那两个是攀爬 / 跳的参数 ✓）；
        #   ⚠ 也**不接**「走不动按跳」那个开关 ✗（那是走路执行器的事 ✓ 见 `PortalJob` 说明 ✓）。
        return portal_job_for_edge(terrain, z, edge, **kw)
    if kind == "jump":
        # 「跳」**吃**「卡住判定时长」（按跳后多久没落到目标集合就回入口 ✓）
        # 和「起跳距离」（临时参数 ✓）—— 两个都留着 ✓。
        return jump_job_for_edge(terrain, z, edge, **kw)
    if kind != "climb":
        raise ValueError("这个模块只认「走」「爬」「下跳」「跳」，收到：%s" % kind)
    # ⚠ 「起跳距离」**爬现在也吃**（2026-09-27 用户要求）：它是「**斜跳上绳**」的区间上界
    #   （`ClimbJob.jump_start_px` ✓，`0` = 关 ⇒ 与老版本**完全一致** ✓）—— 所以原来那行
    #   `kw.pop("jump_start_px", None)` **删掉** ✗，让它顺着下面的 `**kw` 传进 `ClimbJob`
    #   （`plan_jobs` 的两个调用方本来就都在传这个参数 ✓，不用改调用方 ✓）。
    lids = zones.ladder_ids(terrain)
    want = str(edge.get("ladder") or "")
    L = next((x for x in terrain.ladders if lids.get(id(x)) == want), None)
    if L is None:
        raise ValueError("这张图上找不到绳 %s" % (want or "(空)"))
    # ⭐ **方向只看起点在哪端**（用户 2026-09-28）；配了「中途跳下」时，终点**允许不在
    #   绳端**（爬到 `mid_y` 再往 `mid_dir` 跳出去 ✓）—— 见 `zones.climb_direction` ✓。
    d = zones.climb_direction(terrain, z, L, edge.get("from"), edge.get("to"),
                              mid_jump=bool(edge.get("mid_dir")),
                              mid_y=edge.get("mid_y"))
    if d is None:
        raise ValueError(
            "说不清 %s →(爬 %s)→ %s 是向上还是向下"
            "（起点不在绳的两端；或终点不在绳端、而这条边又没配「中途跳下」；"
            "或一头圈进了两个集合）—— 回编辑器看一下。"
            % (edge.get("from"), want, edge.get("to")))
    dst_y = dst_surface_y(terrain, z, edge.get("to"), L, d)
    # ⚠ 这里原来还造一个 `entry_span`（入绳端那块 foothold）给「按跳许可」用 ——
    #   2026-09-27 用户明确否掉那道门（"**对齐了就按跳开始爬啊，不需要判定别的**" ✓）
    #   ⇒ 一起删了 ✓（`zones.ladder_ends` 别处还在用 ✓）。
    # ⭐ 「中途跳下」（用户 2026-09-28 ✓）：**逐边**配置，原样转给执行器 ✓
    #   ⚠ `mid_y` 只有配了方向才有意义（老边两个都没有 ⇒ `None`/`""` ⇒ 执行器那条路整条不走 ✓）
    kw.setdefault("mid_dir", zones.mid_jump_dir(edge))
    kw.setdefault("mid_y", edge.get("mid_y"))
    # ⭐⭐⭐ **「中途跳下」要验落点**（2026-10-19 用户现场 ✓："从底层去 3，在爬 L1 去入口时
    #   **莫名其妙会跳下来**"✗）：那些边配的 `mid_y` 是**凭手写的数**，不保证那个高度上
    #   真有平台 —— 实测 200010302 的 `2→3` / `3→2` 两条边配 `mid_y=-1024`，
    #   而 y 在 **-1100~-950 之间一条 foothold 都没有** ⇒ 一跳到那儿就**直接掉回底层** ✗。
    #   ⇒ 到时候按数据核一遍（`_mid_land` ✓ 同 `foothold_below` 那套几何，不新造尺子）：
    #     落点不存在 ⇒ **不启用中途跳**（照常爬完整根绳 ✓ 宁可多爬，也不掉下去 ✗），
    #     并把原因挂在任务的 `mid_warn` 上（执行器的 note 会显示出来 ✓ 人不至于蒙在鼓里 ✓）。
    _md = str(kw.get("mid_dir") or "")
    _my = kw.get("mid_y")
    if _md and _my is not None:
        _ok, _why = _mid_land(terrain, L, float(_my), _md)
        if not _ok:
            kw["mid_dir"] = ""
            kw["mid_y"] = None
            kw["mid_warn"] = (
                "⚠ 这条边配的「中途跳下」**没启用**：%s ⇒ 跳下去只会掉回底层 ✗"
                "（把「高度」改成要落的那块平台的 y 再试 ✓）" % _why)
        else:
            # ⭐⭐ **方向改由「高度」在运行期定**（用户 2026-10-19 ✓ 原话：
            #   「3→2 **不管方向朝哪里**，只要是需要先上绳梯，那**就要跳上绳梯**；
            #    上了之后再根据 `mid_y` 看是按 ↑ 还是 ↓」✓）。
            #   为什么必须在运行期定：这种边**起点常常挂在绳的中段**（既不是绳顶也不是绳底 ✓），
            #   规划期只能靠"目标在绳哪一头"猜 ⇒ 猜成 +1 就是"越爬越远"✗（3→2 现场 ✓）。
            #   ⇒ 上绳阶段一律 ↑（抓绳只认 ↑），上绳后拿**当时的 y** 与 `mid_y` 比 ✓。
            kw["dir_by_mid"] = True
    # ⭐⭐ **「斜跳上绳」默认判定**（用户 2026-10-06 ✓ 原话："这个比较特殊，在 x 范围不包含绳梯 x
    #   的情况下默认走斜跳上绳梯，攀爬执行器需要默认判定出来并支持" ✓）：
    #   `ClimbJob` 里那条「斜跳上绳」（①' ✓ 2026-09-27 用户的设计 ✓）**默认是关的** ——
    #   要人在设置里配「起跳距离」才开 ✗。这里替它**从数据判**：
    #     · 判据就一条（用户给的 ✓）：**起点那一层的 x 范围不含绳的 x** ⇒ 人不在绳底下 ⇒
    #       **斜着跳**过去抓绳 ✓（`zones.climb_via_jump` ✓）；含 ⇒ 原地跳就够 ✓ 不启用 ✓；
    #     · 跳得够不够得着 ⇒ 上界 = **这一层离绳最近的那条边到绳的距离** ✓
    #       （`zones.climb_jump_reach` ✓ 现场 7/L6 = 44px ✓ —— **从数据算，不拍数字** ✓）。
    #   ⚠⚠ **2026-10-19 改**（用户现场："从 3 到 2 上绳梯时**没有按跳**"✓）：
    #     原来只在设置里没配（`jump_start_px <= 0`）时才用数据算的 reach ✗ ——
    #     可**设置里那个值可能比"绳到平台边的距离"还小** ⇒ 斜跳窗口
    #     `near_px < |dx| <= jump_start_px` 永远**含不住**人的站位（站在平台边就是
    #     `|dx| = 绳到边的距离`）⇒ 只会一直"走"、**永远不按跳** ✗（实测：集合 3 的边到
    #     绳 103px、集合 2 是 87px，而设置里 `起跳距离` 只有 65 ⇒ 两个都进不去 ✓）。
    #     ⇒ 只要判据成立（绳**不在**起始层的 x 范围里 ⇒ 走不过去、只能斜跳），
    #       就把 `jump_start_px` 抬到 **数据算出来的 reach**（+6px 读数余量）✓；
    #       设置里给得更大就照用 ✓（不往下压 —— 那是人有意调的 ✓）。
    if zones.climb_via_jump(terrain, z, L, edge.get("from")):
        _reach = zones.climb_jump_reach(terrain, z, L, edge.get("from"))
        if _reach and float(_reach) > 0:
            _cur = float(kw.get("jump_start_px") or 0.0)
            kw["jump_start_px"] = max(_cur, float(_reach) + 6.0)
    return ClimbJob(want, L.x, L.y1, L.y2, d, dst_set=edge.get("to"),
                    # src_set：没上绳时用来分"跳没生效（接着按跳）"和"真掉下来了"（要求 ②）
                    src_set=edge.get("from"),
                    dst_y=dst_y, dst_x=dst_center_x(terrain, z, edge.get("to")),
                    **kw)


def drop_job_for_edge(terrain, z, edge, **kw):
    """从一条「下跳」边造出任务：可下跳 foothold 取边上的那一格（不填 = 起点集合的全部）。

    抛 ValueError 的两种情况：一个可下跳的都没有、或者那些 id 在本图里找不到
    （换了地图/地形重导过）—— **不许猜**，让上层报警或跳过。

    坐标口径同 `job_for_edge`：地形数据本来就在世界坐标系里，玩家读数与它**直接比** ✓。
    """
    from core import zones

    if (edge.get("kind") or "") != "drop":
        raise ValueError("不是「下跳」边：%s" % edge.get("kind"))
    ids = zones.drop_footholds(z, edge)
    if not ids:
        raise ValueError("这条下跳边一个可下跳 foothold 都没有：%s → %s"
                         % (edge.get("from"), edge.get("to")))
    known = {str(f.fid): f for f in terrain.footholds}
    spots = []
    for fid in ids:
        f = known.get(str(fid))
        if f is None or f.is_wall:
            continue
        # ⭐ **五元组**（中心 x, 左, 右, **面 y**, id ✓ 同 `WalkJob` 的落点形状 ✓）：
        #   面 y = 该 foothold **中点 x** 处的面 y（与 `_set_spans` 同一口径 ✓）——
        #   ALIGN 用它判"**人是不是站在这一块面上**"（用户 2026-10-02 ✓ 见 `DropJob.update`
        #   的 ALIGN ✓，就是这次"掉到别的层还按 ↓+跳"那个 bug 的闸 ✓）。
        _mid = (f.left + f.right) / 2.0
        spots.append((_mid, f.left, f.right, float(f.y_at(_mid)), str(fid)))
    if not spots:
        raise ValueError("可下跳 foothold 在本图里都找不到（或都是墙）：%s" % (ids,))
    # ⭐ **目标平台的"面 y"**（用户 2026-09-28：下跳的到达判据换成"脚下这块面 == 目标面"✓）——
    #   取**目标集合里所有非墙 foothold 的面 y**（`_set_spans` 给的是"该 foothold 中点 x 处的
    #   面 y" ✓，与择路同一口径 ✓）；运行时落点是哪一条取决于从哪个下跳点跳 ⇒ 这里**不预先
    #   绑定**那一条，`DropJob` 命中任意一条即算 ✓。
    dst_ys = [float(s[2]) for s in _set_spans(terrain, z, edge.get("to"))]
    return DropJob(edge.get("from"), edge.get("to"), spots, dst_ys=dst_ys, **kw)


def walk_job_for_edge(terrain, z, edge, **kw):
    """从一条「走」边造出任务：落点 = 目标集合里的非墙 foothold（**不许猜**）。

    抛 ValueError 的两种情况：目标集合里一条 foothold 都没有、或者那些 id 在本图里
    找不到（换了地图/地形重导过）—— 让上层如实报警，别下一条"不知道往哪走"的命令 ✗。

    ⚠ **这条边的走法（`dir` = 仅向左 / 仅向右）在这里灌进任务**（2026-09-27 起生效；
    它原来是空占位）。逐边配的东西只有这一条路能传进去 —— 全局设置里不许有它
    （2026-09-26 用户明确否掉，见 `WalkJob` 注释）✓。
    """
    from core import zones

    if (edge.get("kind") or "") != "walk":
        raise ValueError("不是「走」边：%s" % edge.get("kind"))
    ids = set(str(v) for v in
              ((z.zones.get(edge.get("to")) or {}).get("footholds") or []))
    if not ids:
        raise ValueError("目标集合「%s」里一条 foothold 都没有：%s → %s"
                         % (edge.get("to"), edge.get("from"), edge.get("to")))
    spots = []
    for f in terrain.footholds:
        if f.is_wall or str(f.fid) not in ids:
            continue
        spots.append(((f.left + f.right) / 2.0, float(f.left), float(f.right),
                      float(f.y_at((f.left + f.right) / 2.0)), str(f.fid)))
    if not spots:
        raise ValueError("目标集合「%s」的 foothold 在本图里都找不到（或都是墙）：%s"
                         % (edge.get("to"), sorted(ids)))
    # 逐边的走法（`dir`）灌进来；调用方显式给了就以调用方为准（写用例时用得上 ✓）
    kw.setdefault("walk_dir", zones.walk_dir(edge))
    return WalkJob(edge.get("to"), spots, **kw)


def portal_job_for_edge(terrain, z, edge, **kw):
    """从一条「传送点」边造出任务（用户 2026-10-06 ✓）：**必须指名门** ✓，且落点要能解析出来 ✓。

    抛 ValueError 的三种（都**如实说清** ✓ 绝不静默 ✗；上层会照实报给界面 ✓）：
      · 边里没写 `portal`（哪扇门 ✗）—— 编辑器里选一个 ✓；
      · 本图没有这扇门（换了地图 / 地形重导过 ✓）；
      · 这扇门**跨图**、或 `tn` 指的门本图里找不到 ⇒ **本功能不支持跨图** ✓（用户明确 ✓）。
    """
    # ⚠ `core.zones` **在 route 里没有顶层 import** ✗（别处用的是函数内懒加载 ✓）——
    #   照同款在这里懒加载 ✓（实测：写成 `zones.portal_by_pn(...)` 而不 import ⇒
    #   `NameError: name 'zones' is not defined` ✓ 而且**只有走到这条边才炸** ✓）。
    from core import zones

    pn = str(edge.get("portal") or "").strip()
    if not pn:
        raise ValueError("「传送点」边没指名是哪扇门：%s → %s（编辑器里选一个 ✓）"
                         % (edge.get("from"), edge.get("to")))
    p = zones.portal_by_pn(terrain, pn)
    if p is None:
        raise ValueError("「传送点」边引用的门本图里没有：%r（换一扇，或重导地形 ✓）" % pn)
    ex = zones.portal_exit(terrain, pn)
    if ex is None:
        # ⚠ 判据（2026-10-06 修正后 ✓）：`portal_exit` 只看"**`tn` 指的那扇门在不在本图**"✓
        #   ⇒ 走到这儿就是"目标门不在本图"（真跨图 ✓ 如 `west00 → east00` ✓）或 `tn` 是空的 ✓。
        raise ValueError("「传送点」%r 的目标门**不在本图**（跨图 / 或 WZ 里没配去向）⇒ "
                         "不支持跨图 ✓（换一扇本图内的门，或改成别的通行方式）" % pn)
    # ⚠⚠ **吃什么 / 不吃什么**（各自有主 ✓ —— `plan_jobs` 是给四种边**一起灌**参数的 ✓，
    #   漏摘一个就是 `PortalJob.__init__() got an unexpected keyword argument` ✗ 实测漏过 ✓）：
    #   · `stall_s`「卡住判定时长」⇒ **摘** ✗（那是攀爬 / 跳跃的 ✓ 走也摘 ✓）；
    #   · `jump_start_px`「起跳距离」⇒ **摘** ✗（那是跳 / 斜跳的 ✓）；
    #   · ⭐ `hold_ms`「坐标对齐误差**时间**」⇒ **留着** ✓（用户 2026-10-06："整套照搬
    #     对齐式爬绳梯的逻辑"✓ —— 爬绳就是"进容差 ⇒ **站住稳 hold_ms** ⇒ 再动作"✓，
    #     传送点现在同一套 ✓ 点按 ↑ 那一下仍然只按 `TAP_ON_S`(30ms) ✓ 两件事不混 ✗）；
    #   · ⭐ `tol_px`「坐标对齐误差**范围**」⇒ **留着** ✓（用户 2026-10-06 ✓："在「坐标误差
    #     范围」内就可以按 ↑" ✓ —— 与攀爬**同一把尺** ✓；**原来这里把它摘掉了** ✗，
    #     于是传送点用的是自己那个"门的框"当对齐线 ✓）。
    #   ⚠⚠ **只摘这两个** ✗（`tol_px` / `hold_ms` 都留着 ✓ —— 见上面两条 ✓ 实测漏改过一次：
    #     注释写着"留着 `hold_ms`"、而这行还摘着它 ⇒ 任务里永远是最默认的 250 ✗
    #     被自检 `t_portal_align_like_climb` ⑦ 当场抓住 ✓ 教训：**注释和代码要对得上** ✗）。
    for _k in ("stall_s", "jump_start_px"):
        kw.pop(_k, None)
    return PortalJob(edge.get("to"), p, ex, **kw)


def jump_job_for_edge(terrain, z, edge, **kw):
    """从一条「跳」边造出任务：候选落点 = **目标集合**里的非墙 foothold（**不许猜** ✗）。

    抛 ValueError 的两种情况：目标集合里一条 foothold 都没有、或者那些 id 在本图里
    找不到（换了地图 / 地形重导过）—— 让上层如实报警，别下一条"不知道往哪跳"的命令 ✗。

    ⚠ **不读**边上的 `footholds` 格（「下跳」有那一格）：用户 2026-09-27 确认先用
      "**目标集合的全部 foothold、就近挑一条**"（要逐边指定是以后的事 ✓）。
    ⚠ 坐标口径同 `walk_job_for_edge`：地形数据本来就在**世界坐标系**里，玩家读数
      与它直接比 ✓（别再加减偏移 ✗）。
    """
    if (edge.get("kind") or "") != "jump":
        raise ValueError("不是「跳」边：%s" % edge.get("kind"))
    ids = set(str(v) for v in
              ((z.zones.get(edge.get("to")) or {}).get("footholds") or []))
    if not ids:
        raise ValueError("目标集合「%s」里一条 foothold 都没有：%s → %s"
                         % (edge.get("to"), edge.get("from"), edge.get("to")))
    spots = [(float(f.left), float(f.right), str(f.fid))
             for f in terrain.footholds if not f.is_wall and str(f.fid) in ids]
    if not spots:
        raise ValueError("目标集合「%s」的 foothold 在本图里都找不到（或都是墙）：%s"
                         % (edge.get("to"), sorted(ids)))
    return JumpJob(edge.get("to"), spots, src_set=edge.get("from"), **kw)


def dst_center_x(terrain, z, dst_set):
    """目标集合里所有**非墙** foothold 的 x 中点（世界坐标）—— 只有"下爬"要用它。

    给 `ClimbJob.dst_x`：下爬最后那一步「跳下绳梯」要朝**目标集合的中心**按住方向键
    （用户 2026-09-26："向量运算就能解决"）。口径**照抄「走」那条边**（`walk_job_for_edge`
    算 `target_x` 就是"各条 foothold 的中心 x 的 (min+max)/2"✓）—— 两边必须同一套，
    否则"走"和"爬"对"集合中心"的理解会分叉 ✗。

    算不出来（集合没圈 foothold、或圈的全是墙）⇒ 返回 None：让执行器**如实说"没有方向
    可依"**，别自己猜一个方向 ✗（同 README「不许拍脑袋补数」）。
    """
    ids = set(str(v) for v in
              ((z.zones.get(str(dst_set)) or {}).get("footholds") or []))
    xs = [((f.left + f.right) / 2.0) for f in terrain.footholds
          if not f.is_wall and str(f.fid) in ids]
    if not xs:
        return None
    return (min(xs) + max(xs)) / 2.0


def dst_surface_y(terrain, z, dst_set, L, direction=1):
    """目标集合里**爬上去会踩上的那一层**的面（y）。

    ⚠ **向上爬已经不用它当到达判据了**（2026-09-26 用户改口径：上爬按**绳梯上端**的 y，
    见 `ClimbJob._arrived`）—— 现在它的用处是**向下爬的落地面**（`ClimbJob._landed`
    的判据①）。这条函数本身留着（`dst_y` 还在灌），但上爬不再读它 ✓。
    挑选（**按方向**，2026-09-26 用户实测定的口径）：
      · 向上爬 ⇒ 只要**在绳上端之上（或齐平）**的那些面（`y <= y_top`），取**最靠下**
        的那个（也就是最先踩上的那层）；
      · 向下爬 ⇒ 对称：只要 `y >= y_bot` 的，取最靠上的那个。
      两组都空（绳顶悬在平台面**下面**这种事很常见）⇒ 退回"离绳端最近的那个面"。

    ⚠ 为什么不能只取"离绳端最近"（第一版就是这么写的，当场挑错了）：同一个集合里
    可能有**只差几像素的两层**（实测「左上平台」里既有一条 -204、又有一条 -208），
    "最近"会挑中绳端**下方**那条 ⇒ 判据比用户要的松 ✗。

    找不到（集合里没有 foothold / 本图没有这些 id）⇒ None ⇒ 调用方退回几何兜底
    （下爬那边退回"绳梯下端"，见 `_target_y`）。
    """
    ids = set(str(v) for v in ((z.zones.get(dst_set) or {}).get("footholds") or []))
    if not ids:
        return None
    y_top, y_bot = min(L.y1, L.y2), max(L.y1, L.y2)
    cand = []
    for f in terrain.footholds:
        if f.is_wall or str(f.fid) not in ids:
            continue
        cand.append(float(f.y_at((f.x1 + f.x2) / 2.0)))
    if not cand:
        return None
    if direction >= 0:                            # 向上：绳端之上的那些面，取最靠下的
        up = [y for y in cand if y <= y_top]
        return max(up) if up else min(cand, key=lambda y: abs(y - y_top))
    dn = [y for y in cand if y >= y_bot]          # 向下：对称
    return min(dn) if dn else min(cand, key=lambda y: abs(y - y_bot))
