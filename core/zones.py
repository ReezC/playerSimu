"""foothold 集合（人工分组）+ 集合之间的边 + 路径点 —— 寻路模块的数据层。

详见 `docs/寻路设计.md` §12。三条设计决定（都是踩过之后定的，别改）：

1. **不做自动分段** ❌
   自动串段会把"要跳/攀才能互通"的 foothold 并成同一段 —— 实测 `105090600` 的第 0 段
   把一面 **388 像素高**的悬崖当成了平台边缘（43 条 foothold、其中 12 条长竖条）。
   所以集合由**人工在编辑器里分组**，判定一律用「**脚下的 foothold id ∈ 集合**」
   （`Terrain.foothold_below()` + 本模块的 `Zones.set_of()`）。段只留作显示/辅助选择。

2. **存 foothold id，不存段号** ✅
   id 来自 WZ，**重导地形、改串段算法都不失效**；段号会变，存段号 = 战斗区指到别处。

3. **边是有向的、默认不可用** ✅
   "B 沿绳 L3 上去到 C" 与 "C 掉到 B" 是**两条**边（下落单向）。
   类型 `walk` / `climb` / `drop` / `unconfirmed` / `jump`；**默认 `unconfirmed`**，
   `walk` 只由 `walk_suggestions()` **建议**、人工确认后才写进 edges —— 与 README
   「标定前不把视觉预测变成按键」一致；本阶段**根本不生成 `jump` 边**。

文件：`datasets/map/<map_id>.zones.json`（一条 foothold 一行，便于 review 与 diff）
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZONES_DIR = ROOT / "datasets" / "map"

#: 「同层可走」的阈值 —— **0/0 不是拍脑袋**：105090600 实测 85 条横向 foothold
#: 399 处候选接缝里，Δy=0 且 gap=0 的正好 70 处；放宽到 Δy≤2 / gap≤8 **结果一样**
#: （数据双峰：要么严丝合缝、要么差得明显）⇒ 阈值不敏感。见 §12.2。
SEAM_DY = 0
SEAM_GAP = 0
#: "近同层"：只作**建议**（编辑器里高亮并显示 Δx/Δy，人工确认）。实测多出 14 处。
NEAR_DY = 10
NEAR_GAP = 20
#: 接缝附近若有墙，判定半径（一个身位的量级）
WALL_PAD = 24
#: 「归属」判定半径：绳梯/传送门算不算某个集合的（世界像素）
ATTACH_PAD = 24

#: 边的类型（界面上叫「**通行方式**」，见 EDGE_LABELS 的说明）。
#: `portal` 是**传送门**：走法既不是走/爬/掉，需要记走的是哪个门（`portal` 字段），
#: 而门的落点**导出数据里没有**（WZ 里是 tm 指向别的图/别的门），只能人来指认。
#: ⚠ `jump` / `drop` 都是**跳跃键**的动作，执行前要先做**跳跃标定**（§6），
#: 但**记录**不受影响 —— 先标好"这里要跳/要下跳"，以后标定完就能跑。
EDGE_KINDS = ("walk", "climb", "drop", "portal", "unconfirmed", "jump")

#: 边类型 → 界面上的说法 + 画布箭头颜色（绿实线/蓝虚线/橙虚线…，见 §12.3 B5）
#:
#: 界面上这一组叫「**通行方式**」（2026-09-26 用户定的术语，已写进 §12.3）——
#: 问的是"从 A 到 B **怎么过去**"。五种（外加一个占位的"待确认"）：
#:
#:     walk  走         —— 走过去（同层、无缝）
#:     climb 爬（绳梯） —— 爬绳/梯子（要指名哪根绳）
#:     jump  跳         —— **在 foothold 边缘按跳键**（平着/斜着蹦过去）
#:     drop  下跳       —— **按住 ↓ 再按跳**（从平台上穿下去，落到下一层）
#:     portal 传送门    —— 走哪个门（要指名哪个门）
#:
#: ⚠ 跳 vs 下跳是**两个不同类型**，别混（2026-09-26 用户专门澄清过）。它们的
#: 按键不一样、落点也不一样：跳是"往前蹦"，下跳是"往下穿"。
EDGE_LABELS = {
    "walk": ("走(walk)", "#188038"),
    "climb": ("爬（绳梯）", "#1a73e8"),
    "jump": ("跳(jump)", "#00897b"),
    "drop": ("下跳(drop)", "#e8710a"),
    "portal": ("传送门", "#a142f4"),
    "unconfirmed": ("待确认", "#9aa0a6"),
}

#: 集合默认配色（编辑器按注册顺序取，可在界面上改）
PALETTE = ("#4a8f4a", "#4a6f8f", "#8f4a6f", "#8f7a4a", "#6f4a8f",
           "#4a8f8a", "#8f4a4a", "#5a5a8f")


# ══════════════════════════════════════════
# 几何：接缝判据
# ══════════════════════════════════════════

def _interval(f):
    """foothold 的 (x_left, y_left, x_right, y_right)（按 x 排序，便于算接缝）。"""
    if f.x1 <= f.x2:
        return f.x1, f.y1, f.x2, f.y2
    return f.x2, f.y2, f.x1, f.y1


def seam(f1, f2):
    """两条 foothold 的接缝 → (gap, dy)；x 完全分离（gap > NEAR_GAP）时返回 None。

    gap = 两者 x 区间之间的水平空隙（重叠算 0）；dy = 接缝处的**高度差**。
    **同一个口径只写这一份** —— 编辑器显示、建议生成、自动化检查都调它。
    """
    a, b = _interval(f1), _interval(f2)
    if a[2] < b[0]:
        gap, xa, xb = b[0] - a[2], a[2], b[0]
    elif b[2] < a[0]:
        gap, xa, xb = a[0] - b[2], b[2], a[0]
    else:                                   # x 重叠
        gap, xa, xb = 0, max(a[0], b[0]), min(a[2], b[2])
    if gap > NEAR_GAP:
        return None
    return gap, abs(f1.y_at(xa) - f2.y_at(xb))


def seam_kind(f1, f2):
    """接缝的性质 → "walk" / "near" / None。

    `walk` = 严丝合缝（Δy=0 且 gap=0）⇒ 可以**自动建议**成可走边；
    `near` = 差一点点 ⇒ 只作提示，**必须人工确认**（可能是台阶、也可能是需要跳）；
    None   = 差得明显 ⇒ 不是同层（跳/掉/攀，属于别的边类型）。
    """
    s = seam(f1, f2)
    if s is None:
        return None
    gap, dy = s
    if dy <= SEAM_DY and gap <= SEAM_GAP:
        return "walk"
    if dy <= NEAR_DY and gap <= NEAR_GAP:
        return "near"
    return None


def _wall_blocks(terrain, x, y):
    """接缝处 (x, y) 是否被**竖向 foothold（墙）**挡着 → bool。

    实测数据里墙就是长竖条（最短 20、中位 60 世界像素），`find_below` 早就跳过它们
    （站不上去）；走不走得过去同样要看有没有墙横在那里 —— 否则会把"隔着崖壁的两块地"
    判成可走。
    """
    for f in terrain.footholds:
        if not f.is_wall:
            continue
        if abs(f.x1 - x) <= WALL_PAD and \
                min(f.y1, f.y2) - WALL_PAD <= y <= max(f.y1, f.y2) + WALL_PAD:
            return True
    return False


def flush_pairs(terrain):
    """整张图里所有「严丝合缝」的 foothold 对 → [(f1, f2)]（按 Δy/gap 排序）。

    这就是编辑器里那批**现成的 walk 建议**（105090600 上是 70 处）。
    """
    fh = [f for f in terrain.footholds if not f.is_wall]
    out = []
    for i in range(len(fh)):
        for j in range(i + 1, len(fh)):
            if seam_kind(fh[i], fh[j]) == "walk":
                out.append((fh[i], fh[j], seam(fh[i], fh[j])))
    out.sort(key=lambda t: (t[2][1], t[2][0]))
    return [(a, b) for a, b, _s in out]


# ══════════════════════════════════════════
# 归属：一个集合包含哪些绳梯 / 传送门
# ══════════════════════════════════════════

def set_span(terrain, ids):
    """一组 foothold id 的世界跨度 → (x0, x1, y0, y1) / None。

    **一份实现**：编辑器的"聚焦到集合"、绳梯/传送门归属、将来的边判定都取它 ——
    各写一份必然差几像素，而"线有没有压在地形上"正是靠这几像素判断的。
    """
    want = {str(i) for i in ids}
    xs, ys = [], []
    for f in terrain.footholds:
        if str(f.fid) in want:
            xs += [f.x1, f.x2]
            ys += [f.y1, f.y2]
    if not xs:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def ladders_of(terrain, ids):
    """**这块地形上**的绳梯 → [Ladder…]（空间归属，判据要**严格**）。

    判据：绳的 **y 区间与集合的 y 跨度重叠**（一根绳常常横跨好几层 ⇒ 多归属，
    这是对的：从下层爬上来的那根绳，上层平台也算"我的绳"），
    且绳的 x 落在集合 x 跨度内（±ATTACH_PAD）。

    **为什么必须严格**（2026-09-26 实测踩过）：为了"绳端离平台 45px 也算连着"，
    曾把这里 y 的容差放到 60px —— 结果 `105090600` 里选「右下」（y=280 那块大平台）
    时，**下面两根绳全亮了**（L2 顶端 231、L3 顶端 237，都在平台下方 43~49px，
    y 区间根本碰不到它）。用户一眼就看出来不对：那两个是「右下休息平台」的绳。
    ⇒ 归属问的是"**这块地形上有没有绳**"，只能严格；
      而"**爬上这根绳能不能到上面那块平台**"是另一个问题（连通性，绳端差几十像素
      也算），走 `ladder_ends` / `climb_suggestions`，**两者别混用容差**。
    """
    sp = set_span(terrain, ids)
    if sp is None:
        return []
    x0, x1, y0, y1 = sp
    out = []
    for L in terrain.ladders:
        if not (x0 - ATTACH_PAD <= L.x <= x1 + ATTACH_PAD):
            continue
        ly0, ly1 = min(L.y1, L.y2), max(L.y1, L.y2)
        if ly1 < y0 - ATTACH_PAD or ly0 > y1 + ATTACH_PAD:
            continue
        out.append(L)
    return out


def ladders_touching(terrain, ids):
    """「和这个集合有关」的绳 → [Ladder…] = **本集合范围内的绳** ∪ **绳端落在本集合的绳**。

    给"加爬升边时选哪根绳"用（§12.3 B5：只列与 from 集合有关的绳）。比 `ladders_of`
    多一类：绳段在下面、**顶端却通到这块平台**（绳端离平台面几十像素是常态）——
    那正是"能爬上来"的那种绳，选边时当然要能选到。
    **高亮不用这个**：那会把"下面平台的绳"也画到本集合身上（见 ladders_of 的说明）。
    """
    out = list(ladders_of(terrain, ids))
    seen = {id(L) for L in out}
    want = {str(i) for i in ids}
    for L in terrain.ladders:
        if id(L) in seen:
            continue
        for f in ladder_ends(terrain, L):
            if f is not None and str(f.fid) in want:
                out.append(L)
                break
    return out


def ladder_ends(terrain, ladder, span=260):
    """绳的两端各通向哪条**横向平台** → (**上端平台, 下端平台**)，取不到给 None。

    顺序按绳端的 y 从小到大（**上端在前**）—— 调用方（`climb_suggestions` 的提示
    文字、将来的执行器"往上爬/往下走"）都按这个顺序读，混了就会写出"上端通向下面
    那块平台"这种自相矛盾的话。

    为什么是"找最近的平台"而不是"看 y 区间重叠"：绳端离平台差几十像素是常态
    （实测 43~49px）—— 区间重叠那套在绳的顶端必然落空，于是"这根绳爬上去到哪儿"
    永远算不出来（这正是 `climb` 边一直为 0 条的原因）。
    只看非墙的 foothold（墙不是落脚点），且绳的 x 要落在平台 x 范围内（±8px）。
    """
    pairs = []
    for yy in (min(ladder.y1, ladder.y2), max(ladder.y1, ladder.y2)):
        best = None
        for f in terrain.footholds:
            if f.is_wall:
                continue
            if not (f.left - 8 <= ladder.x <= f.right + 8):
                continue
            d = abs(f.y_at(min(max(ladder.x, f.left), f.right)) - yy)
            if d <= span and (best is None or d < best[0]):
                best = (d, f)
        pairs.append((yy, best[1] if best else None))
    pairs.sort(key=lambda q: q[0])          # 上端在前
    return (pairs[0][1], pairs[1][1])


def portals_of(terrain, ids):
    """集合里**包含**的传送门 → [Portal…]（判据：落在集合包围盒 ±ATTACH_PAD 内）。"""
    sp = set_span(terrain, ids)
    if sp is None:
        return []
    x0, x1, y0, y1 = sp
    return [p for p in terrain.portals
            if x0 - ATTACH_PAD <= p.x <= x1 + ATTACH_PAD
            and y0 - ATTACH_PAD <= p.y <= y1 + ATTACH_PAD]


# ══════════════════════════════════════════
# 集合 / 边 / 点
# ══════════════════════════════════════════

def zones_path(map_id):
    return ZONES_DIR / ("%s.zones.json" % map_id)


class Zones:
    """一张图的集合 + 边 + 路径点。

    `sets`   : {集合名: {"color": "#xxx", "footholds": ["2", "3", …]}}（**id 是字符串**，
               与导出的 JSON 一致 —— 混用 int 会静默匹配不上）
    `edges`  : [{from, to, kind, ladder?, why?}]（**有向**）
    `points` : [{name, x, y, of}]（编辑器标注的路径点，如"绳 L3 入口"）
    """

    def __init__(self, map_id):
        self.map_id = str(map_id)
        self.sets = {}
        self.edges = []
        self.points = []

    # ---------------- 集合 ----------------

    def add_set(self, name, foothold_ids, color=None):
        """注册一个集合。名字重复时**报错**（不覆盖 —— 免得手滑毁掉已分好的组）。"""
        name = str(name).strip()
        if not name:
            raise ValueError("集合名不能为空")
        if name in self.sets:
            raise ValueError("集合名已存在：%s" % name)
        ids = [str(i) for i in foothold_ids]
        if not ids:
            raise ValueError("集合 %s 没有任何 foothold" % name)
        if color is None:
            color = PALETTE[len(self.sets) % len(PALETTE)]
        self.sets[name] = {"color": color, "footholds": ids}
        return name

    def remove_set(self, name):
        """删集合 → 同时删掉**所有涉及它的边**（返回被删的边，便于撤销/提示）。"""
        self.sets.pop(name, None)
        gone = [e for e in self.edges if e.get("from") == name or e.get("to") == name]
        self.edges = [e for e in self.edges if e not in gone]
        self.points = [p for p in self.points if p.get("of") != name]
        return gone

    def rename_set(self, old, new):
        """改名 → **级联改边与路径点**（边按名字指向，不改就成悬空边 ✗）。"""
        old, new = str(old).strip(), str(new).strip()
        if old not in self.sets:
            raise ValueError("没有这个集合：%s" % old)
        if not new or (new in self.sets and new != old):
            raise ValueError("新名字不可用：%s" % new)
        self.sets[new] = self.sets.pop(old)
        for e in self.edges:
            if e.get("from") == old:
                e["from"] = new
            if e.get("to") == old:
                e["to"] = new
        for p in self.points:
            if p.get("of") == old:
                p["of"] = new
        return new

    def set_of(self, foothold_id):
        """这条 foothold 属于哪些集合 → [名字…]（**允许多归属**，顺序按注册顺序）。"""
        fid = str(foothold_id)
        return [n for n, s in self.sets.items() if fid in s["footholds"]]

    def member_ids(self):
        out = set()
        for s in self.sets.values():
            out.update(s["footholds"])
        return out

    # ---------------- 边 ----------------

    def add_edge(self, src, dst, kind, ladder=None, portal=None, footholds=None,
                 walk_dir=None, why=""):
        """加一条**有向**边。自动去重（同 from/to/kind[/portal] 只留一条）。

        `ladder`：climb 边必须给（爬哪根绳，id 形如 "L2"，见 `ladder_ids`）。
        `portal`：portal 边必须给（走哪个门，门名或 "L 坐标" 都行 —— 数据里
        `Portal.tn` 只有跨图才填，所以这一格是给人看的备注，不参与判据）。
        """
        kind = str(kind)
        if kind not in EDGE_KINDS:
            raise ValueError("未知的边类型：%s（可用：%s）" % (kind, "/".join(EDGE_KINDS)))
        for n in (src, dst):
            if n not in self.sets:
                raise ValueError("集合不存在：%s" % n)
        # ⚠ 校验要放在**去重之前**：已经有一条同 from/to/kind 的边时，下面会直接
        # 返回那一条 —— 乱填的方向类型就悄悄过去了（踩过）。
        if walk_dir and walk_dir not in WALK_DIRS:
            raise ValueError("「走」的方向类型只能是 %s，收到：%r"
                             % ("/".join(x or "(默认)" for x in WALK_DIRS),
                                walk_dir))
        for e in self.edges:
            if (e.get("from") == src and e.get("to") == dst
                    and e.get("kind") == kind
                    and str(e.get("portal") or "") == str(portal or "")):
                return e
        e = {"from": src, "to": dst, "kind": kind}
        if ladder:
            e["ladder"] = str(ladder)
        if portal:
            e["portal"] = str(portal)
        if footholds:
            # 「可下跳 foothold」（2026-09-26 用户要求）：**下跳**从哪些 foothold 起跳。
            # 不填 = 起点集合的全部 foothold（见 `drop_footholds`）—— 所以**只在
            # 人工增删过之后**才落这一格，老文件里没有它照样能用。
            e["footholds"] = [str(x) for x in footholds]
        if walk_dir:
            # 「走」的方向类型（默认方向 = **不写这一格**，老文件不用补；
            # 合法性在上面统一校验过了）
            e["dir"] = str(walk_dir)
        if why:
            e["why"] = str(why)
        self.edges.append(e)
        return e

    def edges_of(self, name):
        return [e for e in self.edges if e.get("from") == name]

    # ---------------- 校验 ----------------

    def validate(self, terrain=None):
        """检查一遍 → [问题文本…]（空 = 没问题）。编辑器保存前与打开时都跑它。

        三类必须报出来（都是"看着没事、跑起来才发现"的）：
          · 引用了本图**不存在**的 foothold id（WZ 版本差异、或换了张图）；
          · **悬空边**（from/to 指向已删除或改名的集合）；
          · **climb 边没指定绳**（攀爬必须知道爬哪根）。
        """
        out = []
        if terrain is not None:
            known = {str(f.fid) for f in terrain.footholds}
            for name, s in self.sets.items():
                miss = [i for i in s["footholds"] if i not in known]
                if miss:
                    out.append("集合「%s」引用了本图不存在的 foothold id：%s"
                               % (name, ", ".join(miss[:8]) + ("…" if len(miss) > 8 else "")))
        for e in self.edges:
            for k in ("from", "to"):
                if e.get(k) not in self.sets:
                    out.append("悬空边：%s → %s（%s 不存在）"
                               % (e.get("from"), e.get("to"), e.get(k)))
                    break
            if e.get("kind") == "climb" and not e.get("ladder"):
                out.append("攀爬边没指定绳：%s → %s" % (e.get("from"), e.get("to")))
            if e.get("kind") == "walk" and e.get("dir") \
                    and e.get("dir") not in WALK_DIRS:
                out.append("走边的方向类型不认识：%s → %s：%r（可用：%s）"
                           % (e.get("from"), e.get("to"), e.get("dir"),
                              " / ".join(x or "(默认)" for x in WALK_DIRS)))
            if e.get("kind") == "drop" and "footholds" in e:
                # 只有**人工改过**这一格才查（没这一格 = 用起点集合的全部，见 drop_footholds）
                got = [str(x) for x in (e.get("footholds") or [])]
                if not got:
                    out.append("下跳边没留下任何可下跳 foothold：%s → %s"
                               "（空列表 = 没法下跳）" % (e.get("from"), e.get("to")))
                elif terrain is not None:
                    known = {str(f.fid) for f in terrain.footholds}
                    miss = [x for x in got if x not in known]
                    if miss:
                        out.append("下跳边引用了本图不存在的 foothold：%s → %s：%s"
                                   % (e.get("from"), e.get("to"), ", ".join(miss[:8])))
            if e.get("kind") == "portal" and not e.get("portal"):
                out.append("传送门边没指定是哪个门：%s → %s（本图内传送门的落点"
                           "导出数据里没有，只能人工指认）"
                           % (e.get("from"), e.get("to")))
        for p in self.points:
            if p.get("of") and p["of"] not in self.sets:
                out.append("路径点「%s」指向不存在的集合：%s" % (p.get("name"), p["of"]))
        return out

    # ---------------- 存 / 读 ----------------

    def to_dict(self):
        return {"map_id": self.map_id, "sets": self.sets,
                "edges": self.edges, "points": self.points}

    def save(self, path=None):
        p = Path(path) if path else zones_path(self.map_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d):
        z = cls(d.get("map_id") or "")
        for name, s in (d.get("sets") or {}).items():
            z.sets[str(name)] = {"color": s.get("color") or PALETTE[0],
                                 # 一律转成字符串：导出的 id 是字符串，混 int 会匹配不上
                                 "footholds": [str(i) for i in (s.get("footholds") or [])]}
        z.edges = [dict(e) for e in (d.get("edges") or [])]
        z.points = [dict(p) for p in (d.get("points") or [])]
        return z


def load(map_id):
    """读某张图的集合文件；没有就返回**空的 Zones**（不是 None —— 编辑器要能直接往上写）。"""
    p = zones_path(map_id)
    if not p.exists():
        return Zones(map_id)
    try:
        return Zones.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:                    # noqa: BLE001
        raise ValueError("集合文件读不出来：%s（%s）" % (p, e))


# ══════════════════════════════════════════
# 建议：集合之间哪些"同层可走"
# ══════════════════════════════════════════

def edge_between(zones, a, b):
    """a → b 之间已有的边（多条时取"最可靠"的那条：走 > 爬 > 跳 > 传送门）。"""
    order = {"walk": 0, "climb": 1, "drop": 2, "portal": 3, "unconfirmed": 8, "jump": 9}
    hit = [e for e in zones.edges if e.get("from") == a and e.get("to") == b]
    if not hit:
        return None
    return sorted(hit, key=lambda e: order.get(e.get("kind"), 5))[0]


def edge_text(zones, a, b):
    """a → b 这一步怎么写给人看（如「走」「爬 绳 L2」）。没有边时是「（没有边）」."""
    e = edge_between(zones, a, b)
    if e is None:
        return "（没有边）"
    zh = EDGE_LABELS.get(e.get("kind"), (str(e.get("kind")), ""))[0]
    if e.get("ladder"):
        return "%s %s" % (zh, e["ladder"])
    if e.get("portal"):
        return "%s %s" % (zh, e["portal"])
    return zh


def find_path(zones, src, dst):
    """集合之间的**有向**路径 → (路径列表或 None, 说明)。

    BFS（边不带权，"最少转几次"就是它）。**找不到时要说清边界** —— 路线测试最常问的
    就是"我还缺哪条边"，只回一句"没有路径"帮不上忙：这里会把"从起点出发能走到的
    全部集合"列出来（那就是缺边的位置）。
    """
    if not src or not dst:
        return None, "还没选平台"
    if src not in zones.sets or dst not in zones.sets:
        return None, "集合不存在：%s" % (src if src not in zones.sets else dst)
    if src == dst:
        return [src], "已经在目标平台上"
    nbr = {}
    for e in zones.edges:
        a, b = e.get("from"), e.get("to")
        if a in zones.sets and b in zones.sets:
            nbr.setdefault(a, []).append(b)
    prev, seen, q = {}, {src}, [src]
    while q:
        cur = q.pop(0)
        for nxt in nbr.get(cur, []):
            if nxt in seen:
                continue
            prev[nxt] = cur
            if nxt == dst:
                p, x = [], dst
                while x != src:
                    p.append(x)
                    x = prev[x]
                p.append(src)
                return list(reversed(p)), "经过 %d 段" % (len(p) - 1)
            seen.add(nxt)
            q.append(nxt)
    reach = sorted(seen - {src})
    if not reach:
        return None, ("走不到：从「%s」出发**一条出边都没有**（它现在是孤岛）—— "
                      "先把它和别的集合连起来" % src)
    return None, ("走不到：从「%s」出发只能到 %s —— 缺边、或者边的方向不对"
                  % (src, "、".join(reach)))


def _has_key(zh):
    """名字里是不是已经带了 `(…)` 那一截 —— 如「走(walk)」「跳(jump)」。

    判据只看**半角括号**：中文注解在这个项目里一律用全角「（绳梯）」（那不算键，
    要补上英文键），而人自己定的键一律用半角（`(jump)`、`(未标定…)` 这种）。
    """
    i = zh.find("(")
    return i >= 0 and zh.find(")", i + 1) > i


def kind_label(kind):
    """边类型在界面上怎么写：名字（已带英文键的**不再拼一遍**）。

    `EDGE_LABELS` 里的名字是**人定的**（「走(walk)」「跳(jump)」是 2026-09-26 用户
    逐条改的），有的带了英文键、有的没带（「爬（绳梯）」）—— 界面上要显示 key 的
    那几处统一走这里，别自己拼：拼出来就是「跳(jump)（drop）」那种自相矛盾
    （显示说 jump、数据里其实是 drop —— 键只该出现一次，而且以名字为准）。
    下拉里用英文键当 data，不受影响。
    """
    zh = EDGE_LABELS.get(kind, (str(kind), ""))[0]
    if _has_key(zh):
        return zh
    return "%s（%s）" % (zh, kind)


def ladder_ids(terrain):
    """{id(ladder): "L1"…} —— 绳梯的**稳定 id**（按 (x, page) 排序编号）。

    与导出约定同一口径（§12.5：ladderRope 补稳定 id = 按 (x,page) 排序编号），
    所以这个编号在"编辑器、边、日志、将来的执行器"之间是对得上的。
    为什么现算而不是写进文件：导出那边保证同一份数据排序稳定，两边用同一口径即可。
    """
    out = {}
    for i, L in enumerate(sorted(terrain.ladders,
                                 key=lambda q: (q.x, getattr(q, "page", 0))), 1):
        out[id(L)] = "L%d" % i
    return out


#: 「走(walk)」的方向类型（2026-09-26 用户要求）：
#:   ""      默认方向 —— 朝目标走就行（现在的行为）
#:   "left"  仅向左 —— 只能按左走过去
#:   "right" 仅向右
#:
#: ⚠ **暂时只是配置占位**：执行器还没有"走到 x"那一步，所以这三个值现在**不影响任何
#: 行为** —— 先在编辑器里能填、能存，逻辑等后面做。界面上必须说清楚这一点，
#: 不然人会以为"设了仅向左它就会只往左走"。
WALK_DIRS = ("", "left", "right")
WALK_DIR_LABELS = {"": "默认方向（朝目标走）",
                   "left": "仅向左",
                   "right": "仅向右"}


def walk_dir(edge):
    """这条「走」边配的方向类型 → "" / "left" / "right"（认不出的当默认）。"""
    v = str((edge or {}).get("dir") or "")
    return v if v in WALK_DIRS else ""


def drop_footholds(z, edge):
    """这条**下跳**边能从哪些 foothold 起跳 → [id…]。

    **没填过就是起点集合的全部**（用户 2026-09-26 定的默认）—— 所以老文件里
    没有这一格也照样能用，不用批量补数据。
    """
    if "footholds" in edge:
        return [str(x) for x in (edge.get("footholds") or [])]
    s = z.sets.get(edge.get("from")) or {}
    return [str(x) for x in (s.get("footholds") or [])]


def climb_direction(terrain, z, ladder, src, dst):
    """「src →(爬这根绳)→ dst」是**向上**还是**向下** → +1 / -1 / None（说不清）。

    用户 2026-09-26 定的规则：**按目标那边在绳的上端还是下端判**。
    为什么不直接比 y 值：绳可能是斜的、两端的 y 也可能挨得近；而 `ladder_ends` 给的
    是"绳的两头各自压在**哪条 foothold** 上"，再查那条 foothold 属于哪个集合 ——
    用的是同一份数据，不引入第二套判据（§12 的老教训：两套判据必然漂）。

    返回 None 的三种情况（**宁可不做也别猜**）：找不到绳；目标集合不在任何一端；
    两头都算得上（比如集合把整根绳都圈进去了 ⇒ 那是圈错了，不是"上下都行"）。
    """
    if ladder is None:
        return None
    up, dn = ladder_ends(terrain, ladder)

    def sets_of(f):
        return set(z.set_of(str(f.fid))) if f is not None else set()

    su, sd = sets_of(up), sets_of(dn)
    if not src or not dst:
        return None
    if dst not in (su | sd) or src not in (su | sd):
        return None                    # 起点或终点压根不在这根绳的两头
    if dst in su and dst not in sd and src not in su:
        return 1                       # 目标在上端 ⇒ 往上爬
    if dst in sd and dst not in su and src not in sd:
        return -1                      # 目标在下端 ⇒ 往下爬
    return None                        # 两头都算得上 ⇒ 说不清


def climb_suggestions(zones, terrain):
    """由**绳**连起来的集合对 → [{from, to, kind:"climb", ladder, why, x, y}…]。

    `to` 可能是 **None**：那根绳的另一端还没圈成集合 —— 这不是"没建议"，而是最有用的
    一条（"这里还差一块地形，坐标在 (x,y) 附近"），编辑器要把坐标显示出来。
    判据全走 `ladder_ends`（为什么不能用 y 区间重叠，见那里的说明）。
    """
    lids = ladder_ids(terrain)
    out = []
    for L in terrain.ladders:
        ends = ladder_ends(terrain, L)      # (上端平台, 下端平台)
        names = [zones.set_of(str(f.fid)) if f is not None else [] for f in ends]
        lid = lids.get(id(L), "L?")
        for i, j in ((0, 1), (1, 0)):
            for a in names[i]:
                for b in names[j]:
                    if a == b:
                        continue
                    out.append({
                        "from": a, "to": b, "kind": "climb", "ladder": lid,
                        "why": "绳 %s(x=%d) 两端连着「%s」和「%s」" % (lid, L.x, a, b),
                        "x": L.x, "y": (min(L.y1, L.y2) + max(L.y1, L.y2)) / 2.0})
        # 有一端没圈 ⇒ 给出"该圈哪儿"（编辑器把它当"还差一块地形"显示）
        for idx, f in enumerate(ends):
            if f is None or zones.set_of(str(f.fid)):
                continue
            cx = (f.left + f.right) / 2.0
            cy = f.y_at(min(max(cx, f.left), f.right))
            for a in names[1 - idx]:
                out.append({
                    "from": a, "to": None, "kind": "climb", "ladder": lid,
                    "why": ("绳 %s(x=%d) 的%s端通向 foothold %s —— **还没圈进任何"
                            "集合**，所以这条爬升现在断在这儿"
                            % (lid, L.x, "上" if idx == 0 else "下", f.fid)),
                    "x": cx, "y": cy})
    return out


def portal_suggestions(zones, terrain):
    """**本图内**的传送门 → [{from, to, kind:"portal", portal, why, x, y}…]。

    数据的实情（实测 105090600）：`Portal.tm` 是目标地图号，`999999999` = 本图内；
    `tn` 是目标门名。**本图内**的能直接连（从这个门进去 → 从 `tn` 那个门出来）；
    跨图的（`tm` 是别的图号）连不了 —— 那张图的地形不在这儿，也不是"回位"要走的路。
    另外 `pt=0` 是**出生点**、不是门（Spine 里那几个 `sp` 就是），要排除掉。
    """
    inmap = {}
    for p in terrain.portals:
        if p.pt != 0 and p.pn:
            inmap.setdefault(p.pn, []).append(p)
    out = []
    for p in terrain.portals:
        if p.pt == 0 or not p.tn or p.tm != 999999999:
            continue                       # 出生点 / 没目标 / 跨图 —— 都不是本图的路
        for q in inmap.get(p.tn, []):
            # 用 foothold_below（不是"正好落在平台上"）：门通常画在平台上方一点点，
            # 拿点当判据会漏。
            fp = terrain.foothold_below(p.x, p.y)
            fq = terrain.foothold_below(q.x, q.y)
            if fp is None or fq is None:
                continue
            for a in zones.set_of(str(fp.fid)):
                for b in zones.set_of(str(fq.fid)):
                    if a == b:
                        continue
                    out.append({
                        "from": a, "to": b, "kind": "portal", "portal": p.pn,
                        "why": "传送门 %s → %s（tn=%s）" % (p.pn, q.pn, p.tn),
                        "x": p.x, "y": p.y})
    return out


def walk_suggestions(zones, terrain, include_near=False):
    """集合两两之间的**可走建议** → [{from, to, kind, why, x, y}…]（有向、双向各一条）。

    **只是建议**：`kind` 是 `walk`（严丝合缝，可直接采纳）或 `near`（差一点，需人工看）。
    判据全走 `seam_kind` + `_wall_blocks`，理由写进 `why`（编辑器把它画出来给人核）。
    默认不产出 `near`（噪声大）；编辑器上"显示边缘带"时传 include_near=True。
    """
    by_id = {}
    for f in terrain.footholds:
        by_id.setdefault(str(f.fid), f)
    names = list(zones.sets)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            best = None
            for fa_id in zones.sets[a]["footholds"]:
                fa = by_id.get(fa_id)
                if fa is None or fa.is_wall:
                    continue
                for fb_id in zones.sets[b]["footholds"]:
                    fb = by_id.get(fb_id)
                    if fb is None or fb.is_wall:
                        continue
                    k = seam_kind(fa, fb)
                    if k is None or (k == "near" and not include_near):
                        continue
                    gap, dy = seam(fa, fb)
                    fx = min(max(fa.left, fb.left), fa.right, fb.right)
                    fy = fa.y_at(fx)
                    if _wall_blocks(terrain, fx, fy):
                        continue
                    cand = (0 if k == "walk" else 1, dy, gap)
                    if best is None or cand < best[0]:
                        best = (cand, fa, fb, k, fx, fy, gap, dy)
            if best is None:
                continue
            _c, fa, fb, k, fx, fy, gap, dy = best
            why = ("foothold %s ↔ %s：Δy=%d、gap=%d%s"
                   % (fa.fid, fb.fid, dy, gap, "" if k == "walk" else "，差一点需确认"))
            for (src, dst) in ((a, b), (b, a)):
                out.append({"from": src, "to": dst, "kind": k, "why": why,
                            "x": fx, "y": fy})
    return out
