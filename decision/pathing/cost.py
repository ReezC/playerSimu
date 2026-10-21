"""一条边"从我现在站的地方过去要多久"的世界像素代价（从 `decision/route.py` 切出）。

⚠ **坐标系只有一个：世界坐标**。面板像素只用于显示和落点误差，不许拿来做代价计算
（换分辨率 / 换窗口就会算出两条不同的路 ✗）。
"""

from decision.pathing.spans import _set_spans, _x_span_gap


#: ⭐ **「自然下落垂直折扣系数」**（用户 2026-09-28 定）：**往下掉的垂直距离要打折** ——
#: 掉下去比走过去 / 爬上去快得多，不能按 1:1 算进"要花多远" ✓。
#: 取值来历（用户："先内置为目标值，后续开发物理参数时再细化"）：
#:   MapleNecrocer（`E:\MyPrograms\RippleRogue\MapleNecrocer`）里**玩家**的下落终速
#:   `Player.MaxFallSpeed = 8`（`Client/MapleCharacter.cs:138` ✓）；同一份代码里"标准实体
#:   移速"取 `MoveSpeed = 2.5`（`Client/Pet.cs:75` / `Client/Familiar.cs:79` 都用它 ✓；
#:   玩家自己的移速是键盘直接驱动的、没写成常量 ✗）⇒ 比值 = 8 / 2.5 = **3.2** ✓。
#: ⇒ 语义："**同样一段垂直距离，掉下去只相当于走 1/3.2**" ✓。
#: ⚠ 这是个**内置目标值、不是量出来的**：等物理参数做起来（走速 / 跳 / 重力可配）再换成
#:   算出来的 ✓（用户明确这么安排 ✓）。
FALL_DISCOUNT = 3.2


def _vert_cost(from_y, to_y):
    """**垂直段**的代价：**往下（`to_y` 更低）按「自然下落折扣」折算；往上按原值**（不折 ✓）。

    用户 2026-09-28 要的是「**自然下落**垂直距离折扣系数」—— 只有"掉下去"才算自然下落 ✓，
    跳上去 / 走上去照样得花时间 ⇒ 不打折 ✓。
    ⚠ 世界坐标 **y 向下增大** ⇒ `to_y > from_y` 就是"目标更低" ✓。
    ⛔ **只给"真的会下落"的通行方式用**（2026-09-28 用户第二次明确："**所有的垂直下落端都要
      打折**（**下攀爬不是垂直下落**）"✓）：
      · `jump` 往下 ✓、`drop` ✓ —— 那是下落 ✓；
      · **`walk` 不用它** ✗（走路往下走一层不是自然下落 ⇒ 直接 `abs(fy - py)` ✓）；
      · **`climb` 也不用** ✗（爬上去 / **往下爬**都不是"自然下落"⇒ 不打折 ✓，现在是原值 ✓）。
    """
    d = float(to_y) - float(from_y)
    return (d / FALL_DISCOUNT) if d > 0 else -d


def hop_cost(terrain, z, edge, at):
    """这条边"**从我现在站的地方过去**"要花多少**距离**（世界像素）；算不出来给 `None`。

    用户 2026-09-27 定的口径（**纯距离，不加任何动作惩罚** ✓）；
    **2026-09-28 用户又要求把"垂直段"算进来、并且按「自然下落折扣」折算** ✓：

    | 通行方式 | 代价 |
    |---|---|
    | **走 `walk`** | **横向** = max（玩家 → 目标集合 foothold 端点的距离, **玩家 → 当前集合最近端点的距离**）✓（后半句是 2026-09-28 用户加的："**最小也是**当前 foothold 集合的最边缘端点距离"✓ —— 多层平台 x 重叠时，光看"我的 x 落在目标面的范围里"会算成 0 ✗）+ **垂直段**：玩家脚下 → 那条 foothold 的面 y，**原值不折** ✓（走路不是自然下落 ✓）|
    | **跳 `jump`** | **横向**：玩家 → 目标集合 foothold 端点的距离 + **垂直段**：按 `_vert_cost`（**往下按 `FALL_DISCOUNT` 折算**✓、往上原值 ✓）—— 跳**可以原地起跳**，所以**不加**"源集合边缘"下限 ✓ |
    | **爬 `climb`** | 玩家 → **入绳点**的距离（上爬下端 `L.bottom` / 下爬上端 `L.top` ✓）+ **绳长**（`abs(y2-y1)` ✓ —— 爬得越高越贵 ✓）⚠ **不折扣**（爬上去不是"自然下落" ✓）|
    | **下跳 `drop`** | 玩家 → **最近那条「允许下跳的 foothold」**的**横向**距离 + **该 foothold → 落点的落差**（要落多深 ✓；**往下 ⇒ 按 `FALL_DISCOUNT` 折算** ✓；落点 = 目标集合里离它最近的那片面 ✓） |
    | **传送点 `portal`** ⭐ | 用户 2026-10-06 定："**角色当前到传送起点的距离 ＋ 传送终点距离到目标点的距离**" ✓ ⇒ 玩家 → **起点门**（那扇门 ✓）＋ **出口门** → 目标集合（横向间隙 + 垂直，**不打折** ✓ 落点之后是走过去、不是自然下落 ✓）|

    ⚠ **为什么垂直段要折扣**（用户 2026-09-28）：**掉下去是"自然下落"，比走过去 / 跳上去快得多**
    ⇒ 不折算的话"从高处绕一格"会被算得和"平地走一格"一样贵 ⇒ 择路会挑错 ✓
    （系数的取值来历见 `FALL_DISCOUNT` ✓：先内置目标值，等物理参数做起来再换成算出来的 ✓）。
    ⛔ **但只有"下落"才折**（同一天第二次明确："**所有的垂直下落端都要打折**（**下攀爬不是
      垂直下落**）"✓）⇒ 收窄成：**`jump` 往下** + **`drop`** 折 ✓；**`walk` 不折** ✗
      （走路往下不是下落 ✓）、**`climb` 不折** ✗（上下爬都不是 ✓）。

    ⚠ **算不了就不猜**：没有玩家位置（`at is None`）/ 几何缺失（绳找不到、上下说不清、
      目标集合没有非墙 foothold）⇒ `None` ⇒ 由 `pick_edge` 退回老口径
      （类型优先级 → 文件顺序 ✓）。
    ⚠ 坐标**不做换算**：地形数据本来就在世界坐标系里，和玩家的 `world_x/world_y` 直接比 ✓
      （同 `job_for_edge` 的 ⚠）。
    """
    from core import zones as zones_mod

    if at is None:
        return None
    kind = str((edge or {}).get("kind") or "")
    x = float(at[0])
    if kind == "walk":
        spans = _set_spans(terrain, z, edge.get("to"))
        if not spans:
            return None
        # ⭐⭐ **2026-09-28 用户第二次修这条**（原话："walk 横向我说过了，**最小也是当前
        #   foothold 集合的最边缘端点距离**"✓ + "**所有的垂直下落端都要打折**（**下攀爬不是
        #   垂直下落**）"✓）：
        #   · **横向有下限**：走到"目标集合那条面"其实**不一定够** —— 人得先走到**当前集合的
        #     边缘端点**那一带（多层平台 x 大量重叠，"我 x 落在目标面的范围里"**不等于**
        #     "我这一层就能过去" ✗）。实测现场（图 `106010105`、三楼 → 二楼）：玩家 x=836、
        #     目标二楼 fh=22 的 x 范围 `[738,902]` **罩住了他** ⇒ 老算法横向算成 **0** ✗，
        #     整条 walk 只剩 29.4（垂直打折后）⇒ 险胜 drop 的 29.8 ✗ ⇒ 用户看到的
        #     "**三楼 #91 到二楼**竟然选了 walk"✓。
        #     ⇒ 横向取「目标集合的 x 间隙」与「**到当前集合最近端点的距离**」的**较大者** ✓
        #     （用户口径是"**最小也是**……"⇒ 就是取大 ✓）。
        #   · **垂直不打折**：走路往下走一层**不是自然下落** ✓ ⇒ 按原值算 ✓
        #     （⛔ 原来 walk / jump 共用一个 `_vert_cost` ✗ ⇒ walk 也被打了折 ✗ —— 那正是
        #      "95px 高差只算 29.4"的来源 ✓）。
        _src = _set_spans(terrain, z, edge.get("from"))
        _edge_min = None
        if _src:
            _lo = min(s[0] for s in _src)
            _hi = max(s[1] for s in _src)
            # 到**最近那个端点**的距离（"最边缘端点"✓；人往哪边走近就往哪边走 ✓）
            _edge_min = min(abs(x - _lo), abs(x - _hi))
        best = None
        for a_, b_, fy, _i in spans:
            _h = _x_span_gap(x, a_, b_)
            if _edge_min is not None:
                _h = max(_h, _edge_min)          # "**最小也是**当前集合边缘端点的距离" ✓
            c = _h + abs(float(fy) - float(at[1]))   # 垂直**不打折**（走路不是下落 ✓）
            if best is None or c < best:
                best = c
        return best
    if kind == "jump":
        spans = _set_spans(terrain, z, edge.get("to"))
        if not spans:
            return None
        # 「跳」保持**垂直打折**（往下跳就是自然下落 ✓、往上按原值 ✓ 见 `_vert_cost`）；
        # ⚠ **不加**"源集合边缘"下限（跳可以从原地起跳，不必先走到边缘 ✓ —— 那是 walk 的事〕。
        best = None
        for a_, b_, fy, _i in spans:
            c = _x_span_gap(x, a_, b_) + _vert_cost(at[1], fy)
            if best is None or c < best:
                best = c
        return best
    if kind == "climb":
        lids = zones_mod.ladder_ids(terrain)
        want = str(edge.get("ladder") or "")
        L = next((q for q in terrain.ladders if lids.get(id(q)) == want), None)
        if L is None:
            return None
        # ⚠ `mid_jump` 一定要传（用户 2026-09-28 ✓）：配了「中途跳下」的边，终点**本来
        #   就不在绳端**（爬到 mid_y 侧面跳出去 ✓）⇒ 不传的话这里算不出代价 ⇒ 这条边
        #   在择路时会**排到最后**（哪怕它是唯一一条）✗ —— 与 `job_for_edge` 的口径必须
        #   一致（同一件事两处判 ⇒ 必然漂，§12 的老教训 ✓）。
        # ⚠ `mid_y` 也要传（2026-10-18 用户定稿 ✓）：起点不在绳端时，方向**全靠它**判 ✓
        #   （不再要求"终点贴绳端"✗ 见 `zones.climb_direction` ✓）
        d = zones_mod.climb_direction(terrain, z, L, edge.get("from"), edge.get("to"),
                                      mid_jump=bool(edge.get("mid_dir")),
                                      mid_y=edge.get("mid_y"))
        if d is None:
            return None
        pt = L.bottom if d > 0 else L.top
        # ⭐ 「绳只是**路过**起点这一层」（2026-10-06 用户口径 ✓ 见 `zones.climb_via_jump`）：
        #   这种情况**入绳点不是绳端** ✗ —— 是"**人跳到绳上的那一点**" = `(L.x, 这一层的面 y)` ✓；
        #   **爬升也不是整根绳** ✗ —— 只有"这一层 → **绳的那一端**"那一段 ✓（见下面那段 ⚠⚠）。
        #   ⚠ 照绳端算入绳点会**离谱地大** ⇒ 择路永远不选它 ⇒ "建得出来却从不被用" ✗
        #     （与上一轮传送点那个坑**同一个形状** ✓ 所以必须一起接 ✓）。
        #   ⚠ 判据 = **只看"起点不在绳两端"** ✓（老情形"起点在绳端"一个字不变 ✓）——
        #     ⛔ 原来这里还 `and climb_via_jump(...)` ✗：那只是"要不要**斜**跳"（x 含不含绳 x ✓
        #       `job_for_edge` 灌斜跳区间时才用它 ✓），**与"从哪上绳"无关** ✗ ——
        #       起点不在绳端 + **x 罩住绳 x**（人正站在绳**底下** ⇒ 原地跳就够 ✓ 不用斜跳 ✓）时，
        #       老条件会把代价打回老分支 ⇒ 按**绳端**算入绳点 ⇒ 距离虚大 ⇒ **能建却几乎不被选** ✗
        #       （与上一轮传送点那个坑**同一个形状** ✓ 这正是用户要我"检查"出来的第二处 ✓）。
        #     ⇒ 这类边一律：入绳点 = **(L.x, 这一层的面 y)** ✓（人从这一层跳到绳上 ✓
        #       斜跳还是原地跳，是 `climb_via_jump` 的事 ✗ 与代价无关 ✓）。
        _up, _dn = zones_mod.ladder_ends(terrain, L)
        _su = {n for f in (_up,) if f is not None for n in z.set_of(str(f.fid))}
        _sd = {n for f in (_dn,) if f is not None for n in z.set_of(str(f.fid))}
        if str(edge.get("from")) not in (_su | _sd):
            a = zones_mod.set_span(
                terrain, (z.zones.get(str(edge.get("from"))) or {}).get("footholds") or [])
            if a is None:
                return None
            near = ((x - float(L.x)) ** 2 + (float(at[1]) - float(a[2])) ** 2) ** 0.5
            # ⚠⚠ **爬升那一段必须跟执行器的"停止点"同一套**（用户 2026-10-06 ✓ 原话：
            #   "注意这种寻路代价算法与斜跳上升算法一样，检查一下" ✓）——
            #   执行器爬到哪儿由 `ClimbJob._arrived` / `_target_y` 说了算 ✓：
            #     · **上爬（d>0）** ⇒ 停在**绳梯上端** ✓（用户 2026-09-26 改的口径 ✓
            #       "改为绳子上端的 y" ✓ —— `dst_y`（目标平台面）**不参与上爬判定** ✗）；
            #     · **下爬（d<0）** ⇒ 停在**绳梯下端**（广播 `at_bottom` ✓；落到目标平台面
            #       `dst_y` 也算 ✓ 而本支的放行条件保证"目标层贴着某一端" ⇒ 两者同一处 ✓）。
            #   ⇒ 一律取**绳的那一端**到**起点这一层的面 y** ✓（**不是**目标层的 y ✗：
            #     那是我第一版写错的地方 ✓ 差虽小（本图 2px）但口径必须一处 ✓）。
            #   ⚠ 老分支（起点在绳端）那行 `abs(L.y2 - L.y1)`（整根绳长）**不用改** ✓ ——
            #     起点在绳端 ⇒ 起点层就是那一端 ⇒ 整根绳长**恰好等于**"从起点层爬到另一端" ✓
            #     （两套写法在这里是同一个数 ✓ 这也是它当年能对的原因 ✓）。
            _end_y = min(float(L.y1), float(L.y2)) if d > 0 else max(float(L.y1), float(L.y2))
            return near + abs(_end_y - float(a[2]))
        near = ((x - float(pt[0])) ** 2
                + (float(at[1]) - float(pt[1])) ** 2) ** 0.5
        return near + abs(float(L.y2) - float(L.y1))
    if kind == "drop":
        ids = set(zones_mod.drop_footholds(z, edge))
        spans = [s for s in _set_spans(terrain, z, edge.get("from")) if s[3] in ids]
        if not spans:
            return None
        left, right, fy, _fid = min(spans,
                                    key=lambda s: _x_span_gap(x, s[0], s[1]))
        lands = _set_spans(terrain, z, edge.get("to"))
        if not lands:
            return None
        # 落点 = 目标集合里**x 方向离这条 foothold 最近**的那片面（从这儿掉下去落到的就是它 ✓）
        lf = min(lands, key=lambda s: _x_span_gap((left + right) / 2.0, s[0], s[1]))
        # ⭐ 横向 + **落差**（用户 2026-09-28：**往下 ⇒ 按「自然下落折扣」折算** ✓）——
        #   原来是 `abs(lf[2] - fy)`（1:1 全算）✗ ⇒ 下跳会被算得比实际贵得多 ✓。
        return _x_span_gap(x, left, right) + _vert_cost(fy, lf[2])
    if kind == "portal":
        # ⭐⭐ **传送点的代价**（用户 2026-10-06 ✓ **原话**）："至于寻路代价：我们应该能知道这个
        #   传送点通向哪里，**寻路代价就等于角色当前到传送起点的距离 ＋ 传送终点距离到目标点的
        #   距离**" ✓ ⇒ 拆成两段，跟用户说的一字对应 ✓：
        #     · 前半 `near` = 玩家 → **起点门**（那扇门在哪儿很清楚 ✓ 见 `portal_by_pn` ✓）；
        #     · 后半 `back` = **出口门** → 目标集合（出口 = `tn` 指的那扇门 ✓ 见 `portal_exit` ✓）。
        #   ⚠ 拿不到门 / 出口（跨图 ✓ 用户明确不支持 ✓）⇒ **`None`：不猜** ✓ ——
        #     `pick_edge` 会把这条排到最后，`job_for_edge` 那边也会如实抛 ✓ 两处口径一致 ✓。
        #   ⚠ 后半**不打折**（同 `climb` 的取舍 ✓）：传送落点之后是**走过去**，不是自然下落 ✓
        #     ⇒ 只有 `jump` 往下 / `drop` 才用 `_vert_cost` ✓。
        _pn = str((edge or {}).get("portal") or "")
        p = zones_mod.portal_by_pn(terrain, _pn)
        if p is None:
            return None
        ex = zones_mod.portal_exit(terrain, _pn)
        if ex is None:
            return None
        spans = _set_spans(terrain, z, edge.get("to"))
        if not spans:
            return None
        back = None
        for a_, b_, fy, _i in spans:
            c = _x_span_gap(float(ex.x), a_, b_) + abs(float(ex.y) - float(fy))
            if back is None or c < back:
                back = c
        near = ((x - float(p.x)) ** 2 + (float(at[1]) - float(p.y)) ** 2) ** 0.5
        return near + back
    return None
