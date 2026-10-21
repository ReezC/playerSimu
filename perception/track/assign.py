# -*- coding: utf-8 -*-
"""**全局最优分配**（匈牙利算法）—— "谁认领谁"那一步（2026-10-11 ✓ 借 OC-SORT 的思路 ✓）。

来源与理由（用户 2026-10-11 ✓ 原话："**我们是否可以参考 visual_tracking 的思路**"✓）：
  `visual_tracking_sdk_20260920` 的跟踪内核是 vendored 的 **OC-SORT** ✓，它的关联那一步走的是
  **全局最优分配**（`_vendor/ocsort/association.py:214` ✓ 优先 `lap.lapjv` ✓ 退
  `scipy.optimize.linear_sum_assignment` ✓）＋ **速度方向代价**（同文件 `:251-284` ✓ 的 ORU ✓）
  —— 而我们的 `MobTracker` 原来是**贪心**（按轨迹顺序逐条认领最近的 ✓ 见它的模块说明 ✓）。

**贪心 vs 全局最优差在哪**（这才是要换的原因 ✓）：
    两只怪靠得近、又有两只框时，贪心会"**先来的轨迹先挑走**"✓ ⇒ 排在后面的那条轨迹只能捡
    剩下的 ✓ ⇒ 偶尔会出现"**两条轨迹互换了对像**"✗（表现为 id 跳一下 ✓ 画面上一闪 ✓）。
    全局最优是**整张代价表一起看**：总代价最小的那种配对 ✓ 天生抗"互相抢" ✓。

⚠ **为什么不直接用 `scipy` / 抄 `lap`**：本模块只有几十行、输入规模是"一屏几十个框"✓
  ⇒ 自己实现**没有新依赖**、可离线对拍（`tools/selftest_tracker.py` 里拿**暴力全排列**验 ✓）。
⚠ 本模块**只做数学**：不看 IoU、不懂怪 ✓ —— 代价表由调用方给（`MobTracker._pair_costs` ✓）。
"""
#: 大于它的代价一律视为"**配不上**"（禁止配对 ✓）。用一个大而不至于溢出的数：
#: 代价表本身在 0..1 附近 ✓（`1 - score` ✓），所以 1e6 足够当"无穷"✓ 又不会把浮点算糊 ✓。
BIG = 1e6


def hungarian(cost):
    """**先"配对最多"、再"总代价最小"**的全局分配（矩形 ✓ 行数可以多于列数 ✓）。

    `cost[i][j]` = 把第 `i` 行配给第 `j` 列的代价（`>= BIG` = 禁止 ✓；行可以不等长 ✓）。
    返回 `[(i, j), ...]`：每行每列**最多一次** ✓，且禁止对**永不出现** ✓。

    ⚠ 语义要说死（本文件第一版没写清、结果验算脚本跟着写歪了 ✗ 见下）：
      · **不是**"空配免费"（那样最优永远是"一对都不配"✗）；
      · **是**"能配上的尽量都配上 ✓，在配对数一样多的方案里挑总代价最小的"✓ —— 这正是
        跟踪要的："每条轨迹尽量认领一个框 ✓，但别为了凑数把框硬塞给远处的轨迹"✓。

    ⚠⚠ 第一版是手写的 **JV（势 + 最短增广路）**，**它有一个真 bug** ✗（2026-10-11 实测逮到 ✓）：
      为了表达"这一列是占位的（配不上 ✓）"，我往矩阵里填了 `BIG`（1e6 ✓）当**有限**代价 ✓
      ⇒ 势被抬到 1e6 量级 ⇒ 真代价之间那点差值（0.25 vs 0.46 ✓）在增广路上被淹没 ✓
      ⇒ **选错行**（实测：`[[0.713],[0.25],[0.455]]` 它挑了 0.455 ✗ 应为 0.25 ✓）。
      ⇒ 换成下面这个**最短路增广**（每加一条配对就找一次"代价增量最小的交替路" ✓）：
      它**不需要占位列** ✗（不合法的边**根本不在图里** ✓）⇒ 没有那个量级问题 ✓；
      而且它算的就是"**先最多、再最小**"（每轮成功增广一次 ⇒ 配对数 +1 ✓
      且每轮拿的是当前残量图上的最短路 ⇒ 总代价最小 ✓ 这是最小费用流的经典结论 ✓）。
      规模是我们这种"一屏几十个框"⇒ 每轮 SPFA 几微秒 ✓ 够快 ✓。
    """
    n = len(cost)
    if n == 0:
        return []
    m = max(len(r) for r in cost)
    if m == 0:
        return []
    # 只留合法边（`cost[i][j] < BIG` ✓ 别的**不进图** ✓）
    edges = []
    for i in range(n):
        row = []
        for j in range(len(cost[i])):
            c = float(cost[i][j])
            if c < BIG:
                row.append((j, c))
        edges.append(row)
    match_r = [-1] * n                  # 行 → 列
    match_c = [-1] * m                  # 列 → 行
    INF = float("inf")
    for _ in range(min(n, m)):          # 最多增广这么多次（每次 +1 条配对 ✓）
        # 残量图：行节点 0..n-1 ／ 列节点 n..n+m-1
        #   行 i → 列 j（合法边 ✓ 代价 c） ・ 列 j → 行 i（**已配对**时 ✓ 代价 −c）
        #   源 = **所有空闲行**（距离 0 ✓ 多源 ⇒ 每轮挑"最容易配上"的那条路 ✓）
        elist = []
        for i in range(n):
            for j, c in edges[i]:
                elist.append((i, n + j, c))                     # 前向 ✓
        for j in range(m):
            i2 = match_c[j]
            if i2 >= 0:
                elist.append((n + j, i2, -_cost_of(cost, i2, j)))   # 反向（拆配对 ✓）
        dist = [INF] * (n + m)
        prev = [None] * (n + m)

        def _has_free(i2):
            return match_r[i2] < 0
        for i in range(n):
            if _has_free(i):
                dist[i] = 0.0
        # ⚠⚠ **有界** Bellman-Ford（最多 V−1 轮 ✓ 早退 ✓）—— **不许用 SPFA** ✗：
        #   残量图里有**负权边** ✓，队列式（SPFA）一旦遇上负环就**转不停** ✓
        #   —— 2026-10-11 实测：第一版就是 SPFA，跑这个 3×1 的例直接**卡死**（超时 ✓）。
        #   有界轮数从结构上保证收工 ✓（本图是"行↔列"分层的 ✓ 不会真有负环 ✓）。
        for _round in range(n + m):
            changed = False
            for u, w, c in elist:
                if dist[u] == INF:
                    continue
                if dist[u] + c < dist[w] - 1e-12:
                    dist[w] = dist[u] + c
                    prev[w] = (u, w - n if w >= n else w)
                    changed = True
            if not changed:
                break
        # 找一个**空闲列**里距离最小的 ⇒ 沿 prev 翻出那条交替路（配对 +1 ✓）
        best_j, best_d = -1, INF
        for j in range(m):
            if match_c[j] < 0 and dist[n + j] < best_d:
                best_d, best_j = dist[n + j], j
        if best_j < 0 or best_d == INF:
            break                       # 没有可增广的路 ⇒ 已经"最多配对"了 ✓
        # 交替路 = 前向边（行→列 ✓ 要配上）+ 反向边（列→行 ✓ 只是回退）
        fwd = []
        v = n + best_j
        ok = False
        while prev[v] is not None:
            u, _j = prev[v]
            if u < n:                   # 行 → 列：一条前向边 ✓ 记下（行, 列）
                fwd.append((u, _j))
                v = u
            else:                       # 列 → 行：只是回退到那一行 ✓（本步不入表 ✓）
                v = u
        if v < n and match_r[v] < 0:    # 走到了一个**空闲行** ⇒ 这条交替路合法 ✓
            ok = True
        if not ok or not fwd:
            break                       # 防御：路不合法就收工 ✓（绝不硬翻 ✗ 也绝不循环 ✗）
        # ⚠⚠ **只要把前向边配上就行** ✗ —— **别再去"清旧列"**：
        #   路是 `自由行 r1 → c1 → r2 → c2 → … → rk → ck（自由列）` ✓
        #   ⇒ c_i 原来的主人正是 **r_{i+1}**（它也在路里 ✓ 马上会被重新指派到 c_{i+1} ✓）
        #   ⇒ 额外的"清"反而会把**同一轮里已经配好的行**打回空闲 ✗
        #     （2026-10-11 实测：就因为这个，600 个随机例里有 5 例出现"**同一列配给两行**"✗）。
        for (r, c) in fwd:
            match_r[r] = c
            match_c[c] = r
    out = [(i, match_r[i]) for i in range(n) if match_r[i] >= 0]
    out.sort()
    return out


def _cost_of(cost, i, j):
    """取代价（**只给已确认合法的边**用 ✓ 越界给 `BIG` ✓ 不抛 ✗）。"""
    try:
        return float(cost[i][j])
    except (IndexError, TypeError, ValueError):
        return BIG
