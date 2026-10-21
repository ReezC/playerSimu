"""集合的 x 跨度 / 包围盒与"两段之间有多远"（从 `decision/route.py` 切出）。

⚠ 这里只做**纯几何**：给 foothold 列表或集合名，算出 `(lo, hi)` 这样的区间 ——
不查玩家位置、不算代价（那是 `pathing.cost` 的事 ✓）。
"""


def _x_span_gap(x, left, right):
    """`x` 到区间 `[left, right]` 的距离（**1D** x，世界像素）—— 落在区间里算 0 ✓。

    口径同 `JumpJob._span_dist`（那边是同一个意思，只是写在类里 ✓）。
    """
    x = float(x)
    if x < float(left):
        return float(left) - x
    if x > float(right):
        return x - float(right)
    return 0.0


def _set_spans(terrain, z, name):
    """集合 `name` 里所有**非墙** foothold → `[(左, 右, 面y, id)…]`（没有就 `[]` ✓）。"""
    ids = set(str(v) for v in ((z.zones.get(name) or {}).get("footholds") or []))
    out = []
    for f in terrain.footholds:
        if f.is_wall or str(f.fid) not in ids:
            continue
        out.append((float(f.left), float(f.right),
                    float(f.y_at((f.left + f.right) / 2.0)), str(f.fid)))
    return out


def set_span(terrain, z, name):
    """集合 `name` 在世界上占的 **x 范围** `(左, 右)`（所有非墙 foothold 的并集）；没有给 `None`。

    用户 2026-09-27 要求 2 用到它（原话）："…且其框底边与角色玩家当前 foothold 集合的
    **(x 范围 + 最大攻击距离)** 有交集…" —— "我这块平台横着占多宽"就是这里 ✓
    （外扩量 `attack_dist` 由调用方 `agent._reachable_without_path` 加，这里只管"范围"本身 ✓）。

    ⚠ **复用 `_set_spans`**（一处实现 ✓，它才是"集合里有哪些非墙 foothold"的口径）：
      这里只把它的 `(左, 右)` 取并集 ✓，不另算一遍 ✗。
    ⚠ 坐标是**世界**坐标（和 foothold 数据同系 ✓）；画面坐标要先用 `screen_to_world` 那套
      平移过去才能比 ✓（画面 1 像素 = 世界 1 像素 ✓）。
    """
    if terrain is None or z is None:
        return None
    spans = _set_spans(terrain, z, str(name))
    if not spans:
        return None
    return (min(s[0] for s in spans), max(s[1] for s in spans))


def set_box(terrain, z, name):
    """集合 `name` 在世界上的**包围盒** `(左, 右, 上y, 下y)`；没有给 `None`（用户 2026-09-28 ✓）。

    ⛔ **为什么单开一个**（2026-09-28 现场修 ✗）：`live_thread._make_toward_set_resolver`
    （"**朝『玩家 → 怪』方向、最近的那个集合**"那条降级 ✓）原来是这样写的：
        `sp = route_mod.set_span(t, z, nm)`  ⇒ 拿到的只是 **(左, 右)** ✗
        `x0, x1, y0, y1 = sp[0], sp[1], sp[2], sp[3]`  ⇒ **`sp[2]` 直接 IndexError** ✗
    而调用方 `agent._mob_goto_towards` 里那句 `except Exception: return False` **把它吞了** ✗
    ⇒ **这条降级从上线起一次都没成功过**，而且**一条痕迹都不留** ✗ —— 现场看到的就是
    "怪判不出集合时**原地卡住**、也不下任务"（用户原话："**没有重新查询、没找到也没根据
    向量下达寻路任务**" ✓）。
    ⇒ 口径**复用 `_set_spans`**（一处实现 ✓ 别另算一遍 ✗）：
      · `左 / 右` = 所有非墙 foothold 的 x 包络（和 `set_span` **同一个值** ✓）；
      · `上y / 下y` = 那些面 y 的 **min / max**（世界坐标**越小越靠上** ✓
        ⇒ 上界 = min、下界 = max ✓ —— 调用方那两行 `if y0 < py` / `if y1 > py` 的语义就是它 ✓）。
    ⚠ 别把它和 `core.zones.set_span` 搞混：那个是**编辑器**用的、收 foothold id 列表、返回四元组 ✓；
      这个是**决策层**用的、收集合名、返回四元组 ✓（两个不同的函数，名字像而已 ✗）。
    """
    if terrain is None or z is None:
        return None
    spans = _set_spans(terrain, z, str(name))
    if not spans:
        return None
    ys = [float(s[2]) for s in spans]
    return (min(s[0] for s in spans), max(s[1] for s in spans),
            min(ys), max(ys))
