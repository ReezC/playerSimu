"""寻路数据层自检：`Terrain.foothold_below` + `core/zones.py`（集合/边/路径点/同层判据）。

**为什么先钉这一层**：它是"我在不在战斗区"的唯一判据 —— 判错了，回位逻辑再对也没用
（会以为自己在 A 平台，实际站在悬崖另一侧）。而且这一层**纯几何、不碰硬件**，
一台机器就能全覆盖（同 `selftest_push_presets` 的思路）。

跑法：
    python -m tools.selftest_zones       # 全过返回 0，有失败返回 1
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import mapdata, zones                              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: 正在做寻路的那张图（见 docs/寻路设计.md §12.1）
MAP_ID = "105090600"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _terrain(footholds, ladders=(), portals=()):
    """造一份小地形（字段与导出的 JSON 一致：**全是字符串**）。"""
    # ⚠ 键名照导出的 JSON：`ladderRope`（绳梯）、**`portals`**（不是 `portal`）
    data = {"footholds": [dict(d, layer="1", group="0") for d in footholds],
            "ladderRope": list(ladders), "portals": list(portals)}
    return mapdata.Terrain("T", data)


def _fh(fid, x1, y1, x2, y2, **kw):
    d = {"id": str(fid), "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2)}
    d.update({k: str(v) for k, v in kw.items()})
    return d


# ---------------------------------------------------------------- find_below / foothold_below

def t_foothold_below_same_rule_as_find_below():
    """`foothold_below` 与 `find_below` **必须同一套口径**，只是多告诉你是哪一条。

    为什么要单钉：判定"在不在战斗区"用前者、显示/旧代码用后者 —— 两边容差或取整
    一旦不同，就会出现"工具说在这条上、界面说在那条上"这种最难查的分歧。
    """
    t = _terrain([_fh(1, 0, 100, 100, 100),          # 地面 y=100
                  _fh(2, 100, 100, 100, 120),        # 竖向：墙（与 1 首尾相接）
                  _fh(3, 0, 300, 100, 300)])         # 更下面一层
    f = t.foothold_below(50, 80)                     # 玩家中心在 y=80，脚底在 100
    check(f is not None and f.fid == 1, "脚下的 foothold 判错了：%s" % f)
    check(t.find_below(50, 80) == (50, 100.0),
          "find_below 与 foothold_below 口径不一致：%s" % (t.find_below(50, 80),))
    check(t.foothold_below(100, 50) is not None
          and not t.foothold_below(100, 50).is_wall,
          "墙被当成了可站立面（find_below 早就不认它，这里也必须跳过）")
    # 头顶上方的地面不算（容差 40）：站在 y=200 问，y=100 那块地应被跳过
    check(t.foothold_below(50, 200) is None
          or t.foothold_below(50, 200).fid == 3,
          "把头顶上方的地面当成了脚下：%s" % t.foothold_below(50, 200))
    # 逐条 foothold 与逐段都要能给出来（段只作显示，语义判定走集合）
    check(t.segment_of(50, 80) is not None, "segment_of 找不到脚下那条段")


def t_seam_kinds():
    """接缝判据：严丝合缝 = walk；差一点 = near（只作建议）；差得多 = 不是同层。"""
    a = _terrain([_fh(1, 0, 100, 100, 100)]).footholds[0]
    same = _terrain([_fh(2, 100, 100, 200, 100)]).footholds[0]
    off5 = _terrain([_fh(3, 100, 105, 200, 105)]).footholds[0]
    gap10 = _terrain([_fh(4, 110, 100, 200, 100)]).footholds[0]
    far = _terrain([_fh(5, 100, 280, 200, 280)]).footholds[0]
    check(zones.seam_kind(a, same) == "walk", "严丝合缝没判成 walk")
    check(zones.seam(a, same) == (0, 0), "接缝算错了：%s" % (zones.seam(a, same),))
    check(zones.seam_kind(a, off5) == "near", "差 5 像素没判成 near")
    check(zones.seam_kind(a, gap10) == "near", "间隙 10 没判成 near")
    check(zones.seam_kind(a, far) is None, "差 180 像素被当成同层了")


def t_flush_pairs_on_real_map():
    """真实图上「严丝合缝」的对数 —— 把 §12.2 的实测数字钉住（70 处）。

    数据是**双峰**的：要么 Δy=0 且 gap=0，要么差得明显 ⇒ 阈值不敏感（放宽到
    Δy≤2/gap≤8 还是 70）。这条测试真正的价值是：万一哪天串段/接缝口径被改动，
    数字会立刻变，而不是等到"战斗区判定莫名其妙不对"才发现。
    """
    p = ROOT / "datasets" / "map" / (MAP_ID + ".json")
    check(p.exists(), "缺少 %s —— 先导出这张图的地形（见 docs/寻路设计.md §12.5）" % p.name)
    t = mapdata.load(MAP_ID)
    pairs = zones.flush_pairs(t)
    check(len(pairs) == 70,
          "严丝合缝的接缝数变了：%d（文档记的是 70 —— 对不上就先看接缝口径）" % len(pairs))
    for a, b in pairs:
        s = zones.seam(a, b)
        check(s == (0, 0), "flush_pairs 里混进了非严丝合缝的：%s vs %s = %s"
              % (a.fid, b.fid, s))


# ---------------------------------------------------------------- 集合 / 边 / 校验

def t_zones_roundtrip_and_validate():
    """集合存读一致、改名级联、删集合连带删边、悬空边/不存在的 id 要报出来。"""
    t = _terrain([_fh(1, 0, 100, 100, 100), _fh(2, 100, 100, 200, 100)])
    z = zones.Zones(MAP_ID)
    z.add_set("A平台", ["1"])
    z.add_set("B平台", ["2"])
    z.add_edge("A平台", "B平台", "walk", why="严丝合缝")
    check(z.set_of("1") == ["A平台"], "set_of 反查错了：%s" % z.set_of("1"))
    check(z.validate(t) == [], "正常文件不该报警：%s" % z.validate(t))

    # 存 / 读（id 必须仍是字符串 —— 混 int 会静默匹配不上）
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.zones.json"
        z.save(p)
        z2 = zones.Zones.from_dict(json.loads(p.read_text(encoding="utf-8")))
        check(z2.sets == z.sets and z2.edges == z.edges, "存读不一致：%s" % z2.sets)
        check(all(isinstance(i, str) for i in z2.sets["A平台"]["footholds"]),
              "id 存成了非字符串")

    # 改名 → 边与路径点级联
    z.points.append({"name": "绳入口", "x": 100, "y": 100, "of": "A平台"})
    z.rename_set("A平台", "战斗区")
    check(z.edges[0]["from"] == "战斗区" and z.points[0]["of"] == "战斗区",
          "改名没级联：%s / %s" % (z.edges[0], z.points[0]))
    # 删集合 → 涉及它的边一起走
    gone = z.remove_set("战斗区")
    check(len(gone) == 1 and not z.edges, "删集合没连带删边：%s" % z.edges)

    # 悬空边 + 引用了不存在的 foothold ⇒ 都要报
    bad = zones.Zones.from_dict({
        "map_id": MAP_ID,
        "sets": {"A": {"footholds": ["1", "999"]}},
        "edges": [{"from": "A", "to": "幽灵", "kind": "walk"}],
    })
    probs = bad.validate(t)
    check(any("999" in s for s in probs), "引用了不存在的 id 没报警：%s" % probs)
    check(any("悬空边" in s for s in probs), "悬空边没报警：%s" % probs)
    check(bad.validate(t) and zones.Zones(MAP_ID).validate(t) == [],
          "空文件不该报警")


def t_walk_suggestions_and_wall_guard():
    """可走建议：严丝合缝的两组之间出**双向**建议；中间有墙 ⇒ 不提。"""
    t = _terrain([_fh(1, 0, 100, 100, 100), _fh(2, 100, 100, 200, 100)])
    z = zones.Zones(MAP_ID)
    z.add_set("A", ["1"])
    z.add_set("B", ["2"])
    s = zones.walk_suggestions(z, t)
    check(len(s) == 2 and {x["kind"] for x in s} == {"walk"},
          "双向 walk 建议不对：%s" % s)
    check(all("Δy=0" in x["why"] for x in s), "建议里没写理由：%s" % s)

    # 同一对，但在接缝处竖一堵墙 —— 隔着崖壁不算可走
    t2 = _terrain([_fh(1, 0, 100, 100, 100), _fh(2, 100, 100, 200, 100),
                   _fh(9, 100, 60, 100, 140)])
    check(zones.walk_suggestions(z, t2) == [],
          "隔着墙还被建议成可走：%s" % zones.walk_suggestions(z, t2))

    # 差 5 像素：默认不提（噪声大），显式打开 include_near 才提
    t3 = _terrain([_fh(1, 0, 100, 100, 100), _fh(2, 100, 105, 200, 105)])
    check(zones.walk_suggestions(z, t3) == [], "near 默认不该出现")
    get = zones.walk_suggestions(z, t3, include_near=True)
    check(len(get) == 2 and get[0]["kind"] == "near",
          "include_near 没生效：%s" % get)


def t_rope_ends_and_climb_suggestions():
    """**两个问题两套判据**：绳"属于谁"要严格，"爬上它能不能到上面"要宽松（差 45px 也算）。

    实测踩过（2026-09-26，用户报的）：为了后者把前者的 y 容差从 24 放到 60px ✗ ⇒
    选「右下」（y=280 那块大平台）时**下面两根绳全亮了**（L2 顶端 231、L3 顶端 237，
    都在平台下方 43~49px，y 区间根本碰不到它）—— 那两根其实是「右下休息平台」的绳。
    ⇒ 归属 `ladders_of` 严格；连通性 `ladder_ends` / `climb_suggestions` 宽松；
      "加边选绳"用 `ladders_touching`（两者并集）。这条用例把三者钉在一起，
      以后谁再把容差混用就会红。
    """
    # 上平台 y=100、下平台 y=300；绳 x=50，y 从 300 到 145（**顶端离上平台还差 45px**）
    t = _terrain([_fh(1, 0, 100, 100, 100), _fh(2, 0, 300, 100, 300)],
                 ladders=[{"x": "50", "y1": "300", "y2": "145",
                           "l": "1", "page": "2"}])
    z = zones.Zones(MAP_ID)
    z.add_set("上", ["1"])
    z.add_set("下", ["2"])
    # ① 归属**严格**：绳段只在下平台那一带 ⇒ 只有「下」认它
    check(len(zones.ladders_of(t, ["2"])) == 1, "绳段就在下平台旁边，却没算成它的")
    check(zones.ladders_of(t, ["1"]) == [],
          "绳端离上平台 45px，却被算成「上」的绳了 —— 归属必须严格"
          "（用户实测报过：选「右下」时下面两根绳全亮）")
    # ② 连通/选绳**宽松**：绳端通到上平台 ⇒ 加爬升边时得能选到它
    check(len(zones.ladders_touching(t, ["1"])) == 1,
          "上平台加攀爬边时选不到这根绳（它的顶端通到这份地形上）")
    # ③ 建议：双向 climb，并带上**稳定的绳 id**
    s = zones.climb_suggestions(z, t)
    lids = zones.ladder_ids(t)
    check(len(s) == 2 and {x["kind"] for x in s} == {"climb"},
          "没给出双向攀爬建议：%s" % s)
    check(all(x["ladder"] in lids.values() for x in s),
          "建议里没有稳定的绳 id（要和导出/日志对得上）：%s" % s)
    check(all(x["to"] for x in s), "两端都有集合，不该出现「没圈」的建议：%s" % s)

    # ③ 另一端没圈 ⇒ 必须报出来：那是"还差一块地形"，不是"没建议"
    z2 = zones.Zones(MAP_ID)
    z2.add_set("上", ["1"])
    s2 = zones.climb_suggestions(z2, t)
    check(len(s2) == 1 and s2[0]["from"] == "上" and s2[0]["to"] is None,
          "另一端没圈时没报出来：%s" % s2)
    check("还没圈" in s2[0]["why"], "理由没说清是「没圈」：%s" % s2[0]["why"])
    # ④ **用户报的那个 case，用真实图钉住**：fh=41 是「右下」那块 y=280 的大平台上的
    #    一条 ⇒ 下面那两根绳（x=701 顶端 231、x=1245 顶端 237）**不属于**它
    #    （y 区间碰不到），但**通到**它（加边时要能选到）。
    #    ⚠ 那两块平台各由好几条 foothold 拼成（fh 19 / 41 都在 y=280、fh 72 在 y=6），
    #      所以这里把「右下」那一段的两条都给上 —— 只给一条的话另一根绳的绳端就在
    #      集合外了（那不是判据错，是集合真的没含它）。
    tr = mapdata.load(MAP_ID)
    check(tr is not None, "缺少 %s 的地形数据" % MAP_ID)
    check(zones.ladders_of(tr, ["19", "41"]) == [],
          "y=280 那块平台上冒出了绳 —— 就是用户看到亮起来的那两根")
    xs = sorted(L.x for L in zones.ladders_touching(tr, ["19", "41"]))
    check(701 in xs and 1245 in xs,
          "「顶端通到这块平台」的两根绳没被找出来（加边选绳要能选到）：%s" % xs)
    # 没圈的是**下端**那块（y=300）：坐标要指到它上面，而且说法不能上下颠倒
    check(abs(s2[0]["x"] - 50) < 1 and abs(s2[0]["y"] - 300) < 1,
          "没给出「该圈哪儿」的坐标：%s" % s2)
    check("的下端" in s2[0]["why"], "上/下端说反了：%s" % s2[0]["why"])


def t_portal_edge_and_validate():
    """`portal` 边：本图内的门能自动连出来；**出生点/跨图门不连**；缺门名要报警。

    实测 105090600：8 个门里 6 个是 `pt=0`（出生点，不是门），2 个是跨图
    （`tm=105090500/105090700`）—— 本图回位用不上，所以 `portal_suggestions` 一条都不出，
    这是**对的**（不是坏了）。
    """
    t = _terrain(
        [_fh(1, 0, 100, 100, 100), _fh(2, 400, 100, 500, 100)],
        portals=[
            {"pn": "left", "pt": "1", "x": "50", "y": "90",
             "tm": "999999999", "tn": "right"},
            {"pn": "right", "pt": "1", "x": "450", "y": "90",
             "tm": "999999999", "tn": "left"},
            {"pn": "sp", "pt": "0", "x": "20", "y": "90",
             "tm": "999999999", "tn": ""},                      # 出生点
            {"pn": "out", "pt": "2", "x": "30", "y": "90",
             "tm": "105090700", "tn": "west00"},                # 跨图
        ])
    z = zones.Zones(MAP_ID)
    z.add_set("左", ["1"])
    z.add_set("右", ["2"])
    check("portal" in zones.EDGE_KINDS, "边类型里没有「传送门」这一类")
    s = zones.portal_suggestions(z, t)
    # 两个门互为落点 ⇒ **双向各一条**（不是一条）
    check(len(s) == 2 and {(x["from"], x["to"]) for x in s}
          == {("左", "右"), ("右", "左")},
          "本图内的传送门没连出来（出生点/跨图的不该连）：%s" % s)
    check({x.get("portal") for x in s} == {"left", "right"},
          "没记下走的是哪个门：%s" % s)

    # 边本身：portal 边要能存，缺门名要报警（validate 会把问题说出来）
    z.add_edge("左", "右", "portal", portal="left")
    check(z.validate(t) == [], "正常的传送门边不该报警：%s" % z.validate(t))
    bad = zones.Zones.from_dict({
        "map_id": MAP_ID,
        "sets": {"A": {"footholds": ["1"]}, "B": {"footholds": ["2"]}},
        "edges": [{"from": "A", "to": "B", "kind": "portal"}]})
    check(any("传送门边" in x for x in bad.validate(t)),
          "缺门名的传送门边没报警：%s" % bad.validate(t))


def t_find_path_over_edges():
    """边图上的路径：走得通要给**逐步**路线，走不通要说清**边界**（缺边的位置）。

    路线测试最常问的就是这两件事 —— 光回一句"没有路径"帮不上忙。
    """
    t = _terrain([_fh(1, 0, 100, 100, 100), _fh(2, 100, 100, 200, 100),
                  _fh(3, 200, 100, 300, 100), _fh(4, 400, 100, 500, 100)])
    z = zones.Zones(MAP_ID)
    for n, fid in (("A", "1"), ("B", "2"), ("C", "3"), ("D", "4")):
        z.add_set(n, [fid])
    z.add_edge("A", "B", "walk")
    z.add_edge("B", "C", "climb", ladder="L1")

    path, why = zones.find_path(z, "A", "C")
    check(path == ["A", "B", "C"], "路径不对：%s" % (path,))
    check("2 段" in why, "没说清几段：%s" % why)
    check(zones.edge_text(z, "B", "C") == "爬（绳梯） L1",
          "每一步的说法不对：%s" % zones.edge_text(z, "B", "C"))

    # **有向**：反过来走不通，而且要说清"从 B 只能到 C"（缺的就是 C→B 那条边）
    path2, why2 = zones.find_path(z, "B", "A")
    check(path2 is None and "只能到" in why2 and "C" in why2,
          "走不到时没说出边界：%s" % why2)

    # 待在原地 / 集合不存在 / 孤岛
    check(zones.find_path(z, "A", "A")[0] == ["A"], "同一个集合没返回原地")
    check(zones.find_path(z, "A", "不存在")[0] is None, "不存在的集合没拦住")
    check("一条出边都没有" in (zones.find_path(z, "D", "A")[1] or ""),
          "孤岛没说出「一条边都没有」：%s" % (zones.find_path(z, "D", "A")[1],))


def t_kind_label_text():
    """通行方式在界面上怎么写：**名字里带了英文键就不再拼一遍**；跳 / 下跳别混。

    踩过两次（都记在这儿）：
      · 自动补键那套遇上有键的名字会拼成「跳(jump)（drop）」—— 显示说 jump、
        数据里是 drop，两句话打架；
      · 用户澄清后语义才定下来：`jump` = **在 foothold 边缘按跳键**（往前/斜着蹦），
        `drop` = **按住 ↓ 再按跳**（往下穿）。按键不同、落点不同 ⇒ 必须是**两个能分辨**的
        类型（数据里一直就是两个，只是显示名一度撞在一起）。
    """
    check(zones.kind_label("jump") == "跳(jump)",
          "「跳」的显示名不对：%s" % zones.kind_label("jump"))
    check(zones.kind_label("drop") == "下跳(drop)",
          "「下跳」的显示名不对：%s" % zones.kind_label("drop"))
    check("（jump）" not in zones.kind_label("jump")
          and "（drop）" not in zones.kind_label("drop"),
          "又把数据里的键拼了一遍：%s / %s"
          % (zones.kind_label("jump"), zones.kind_label("drop")))
    # 名字里没带键的照旧补上（不然界面上看不出它对应哪个 kind）
    check(zones.kind_label("climb") == "爬（绳梯）（climb）",
          "没带键的类型该补上键：%s" % zones.kind_label("climb"))
    check(zones.kind_label("portal") == "传送门（portal）",
          "没带键的类型该补上键：%s" % zones.kind_label("portal"))
    # **两个"跳"必须分得开**（一个往前蹦、一个往下穿）
    check(zones.kind_label("drop") != zones.kind_label("jump"),
          "两个「跳」的显示名一样了，分不出是哪个：%s / %s"
          % (zones.kind_label("jump"), zones.kind_label("drop")))
    # 数据里的 kind 不许被改名影响（改了已有边文件就失效）
    check("drop" in zones.EDGE_KINDS and "jump" in zones.EDGE_KINDS,
          "边类型表被动过了：%s" % (zones.EDGE_KINDS,))
    # 每种类型都得有显示名（下拉是照 EDGE_LABELS 列的，缺了就会少一项）
    miss = [k for k in zones.EDGE_KINDS if k not in zones.EDGE_LABELS]
    check(not miss, "这些类型没有显示名（界面上会少一项）：%s" % miss)


def t_climb_direction():
    """「爬这根绳」是**向上还是向下** —— 按目标在绳的**上端还是下端**判（2026-09-26 规则）。

    真实数据（105090600 + 用户圈的集合）：L1 上端压着 foothold#44（属于「左上平台」）、
    下端压着 #6（属于「左上」）⇒ `左上 →(爬 L1)→ 左上平台` = **向上**，反过来 = 向下。
    为什么钉它：方向判错就是**往反方向爬**，比不做更糟；说不清时一律 None（宁可不做）。
    """
    t = mapdata.load(MAP_ID)
    z = zones.load(MAP_ID)
    lids = zones.ladder_ids(t)
    L1 = next((x for x in t.ladders if lids.get(id(x)) == "L1"), None)
    check(L1 is not None, "这张图没有 L1，这条测不了")
    check(zones.climb_direction(t, z, L1, "左上", "左上平台") == 1,
          "左上 → 左上平台 该判成向上")
    check(zones.climb_direction(t, z, L1, "左上平台", "左上") == -1,
          "左上平台 → 左上 该判成向下")
    for a, b, why in (("左上", "左上", "起点终点同一个集合（自环）"),
                      ("左上", "右下", "终点压根不在这根绳的两端"),
                      ("幽灵", "左上平台", "起点不在绳的两端")):
        check(zones.climb_direction(t, z, L1, a, b) is None,
              "%s：该判不出来（None），不许猜" % why)
    check(zones.climb_direction(t, z, None, "左上", "左上平台") is None,
          "连绳都没有的时候该给 None")


def t_walk_dir():
    """「走」的方向类型（2026-09-26 用户要求）：**只是配置占位** —— 能填、能存、能校验。

    现在执行器还用不上这三个值（"走到 x"那一步还没做），所以更要把"存得住 / 读得回 /
    认不出的拦住"钉住：不然等逻辑做上去时，会发现存进来的根本不是那三样东西。
    """
    z = zones.Zones(MAP_ID)
    z.add_set("甲", ["1"])
    z.add_set("乙", ["2"])
    e = z.add_edge("甲", "乙", "walk", walk_dir="left", why="只往左")
    check(e.get("dir") == "left" and zones.walk_dir(e) == "left",
          "方向类型没存住：%r" % e)
    # 默认方向 ⇒ **不写这一格**（老文件不用补数据）
    e2 = z.add_edge("乙", "甲", "walk")
    check("dir" not in e2 and zones.walk_dir(e2) == "",
          "默认方向不该写进文件：%r" % e2)
    # 存读一圈还在
    z2 = zones.Zones.from_dict(z.to_dict())
    check(zones.walk_dir(z2.edges[0]) == "left", "存读之后丢了：%r" % z2.edges[0])
    # 乱填的值：读的时候当默认，**校验要报出来**（手改文件最容易出这种）
    z.edges.append({"from": "甲", "to": "乙", "kind": "walk", "dir": "up"})
    check(zones.walk_dir(z.edges[-1]) == "",
          "认不出的值该当默认：%r" % zones.walk_dir(z.edges[-1]))
    check(any("方向类型不认识" in p for p in z.validate()),
          "认不出的方向类型没在校验里报出来：%s" % z.validate())
    # 写进去的时候就要拦住（别等保存才发现）
    try:
        z.add_edge("甲", "乙", "walk", walk_dir="斜着")
        raise AssertionError("乱填的方向类型该当场拦住")
    except ValueError:
        pass


# ---------------------------------------------------------------- 跑

TESTS = (
    ("foothold_below 与 find_below 同一口径（且跳过墙）",
     t_foothold_below_same_rule_as_find_below),
    ("接缝判据：严丝合缝=walk / 差一点=near / 差得多=不是同层", t_seam_kinds),
    ("真图上严丝合缝 70 处（§12.2 的实测数字）", t_flush_pairs_on_real_map),
    ("集合存读 / 改名级联 / 删集合连带删边 / 校验报警",
     t_zones_roundtrip_and_validate),
    ("可走建议（双向 + 理由）与「隔着墙不可走」", t_walk_suggestions_and_wall_guard),
    ("绳端差 45px 也要接上：攀爬建议（双向 + 绳 id + 没圈的一端）",
     t_rope_ends_and_climb_suggestions),
    ("传送门边：本图内自动连、出生点/跨图不连、缺门名报警", t_portal_edge_and_validate),
    ("边图上的路径：走得通给逐步路线，走不通说清边界", t_find_path_over_edges),
    ("通行方式的显示名：带了键就不再拼一遍；跳 / 下跳 分得开", t_kind_label_text),
    ("爬绳的上下：按目标在绳的上端/下端判（说不清就给 None）", t_climb_direction),
    ("走路的方向类型：存得住/默认不写文件/乱填当场拦住+校验报出", t_walk_dir),
)


def main():
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print("[ OK ] %s" % name)
        except Exception as e:                 # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
