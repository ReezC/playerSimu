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


def t_foothold_below_x_tolerance():
    """站在平台**边上**几像素也要认（`xtol` = 设置里的「坐标对齐误差范围」）。

    为什么要单钉（用户 2026-09-26 实测 105090600 的 (650,283)）：人明明站在 id=41 那块
    平台上（面 y=280，差 3px ✓），可 x=650 在它左边界 **657 的外面 7px** ✗ ——
    老口径 x **一点容差都没有**（y 却有 40px）⇒ `foothold_id=None` ⇒「脚下没有平台 /
    未分组 / 起点不知道」⇒ 起点判定、集合判定、到达判定①**一起瘫** ✗。
    三条判据，缺一条都会出问题：
      · **只兜底**：原本判得出来的结果**一个都不许变**（放宽主判据 = 会误认隔壁平台 ✗）；
      · 放宽后能认出**那一条**（不是随便一条 ✗），超容差仍然找不到（不许无限放宽 ✗）；
      · 放宽也**不许**把"头顶上方的地面"当成脚下（y 判据还得出力 ✓）。
    """
    t = _terrain([_fh(1, 0, 100, 100, 100),        # 地面 [0,100]，面 y=100
                  _fh(2, 200, 100, 300, 100)])     # 隔壁那块 [200,300]，同高
    # ① 站在平台**里面**：容差不许改变结论（一个字节都不该变 ✓）
    check(t.foothold_below(50, 80).fid == 1
          and t.foothold_below(50, 80, xtol=10).fid == 1,
          "xtol 改变了原本判得出来的结果 ✗")
    # ② 边界外 5px：0 容差找不到 → 10px 要认出来（并给出**第 1 条**）
    check(t.foothold_below(105, 80) is None, "用例前提不成立：105 本来不该找得到")
    _got = t.foothold_below(105, 80, xtol=10)
    check(_got is not None and _got.fid == 1,
          "站在平台边外 5px 没认出来（用户 (650,283) 就是这个形状 ✗）：%s" % _got)
    # ③ 超出容差 ⇒ 还是找不到（容差不是"总能找到"✗）
    check(t.foothold_below(120, 80, xtol=10) is None,
          "超出容差也认了（容差形同虚设）✗")
    # ④ 两块平台之间：要认**离得最近的那一条**，不许跨到对面去
    check(t.foothold_below(195, 80, xtol=10).fid == 2,
          "没认最近那条平台：%s" % t.foothold_below(195, 80, xtol=10))
    # ⑤ 头顶上方的地面：y 判据仍要出力（站在 y=200 问，不能把 y=100 当脚下）
    check(t.foothold_below(105, 200, xtol=10) is None,
          "放宽 x 之后把头顶上方的地面当成了脚下 ✗")
    # ⑥ 段号（显示用）要跟同一条判据 —— 否则会出现"集合认得出、段号说没有"这种
    #    最容易被当成程序坏了的分歧 ✗
    check(t.segment_of(105, 80) is None and t.segment_of(105, 80, xtol=10) is not None,
          "segment_of 没跟 xtol 对齐（集合认得出、段号说没有 ✗）")


def t_foothold_below_nearest_y():
    """「脚下」要认**离玩家最近**的那条面（用户 2026-09-27 报「位置状态判错了」）。

    现场（图 `106010105` 的 (577,193)，用户截图那张）：x 命中的只有两条 ——
    `#12 小平台` 面 y=155（离中心 **−38**）、`#47 底层` 面 y=185（离中心 **−8**）。
    老写法在"容差 40 以内"**一律取 fy 最小** ⇒ 挑中了 `#12 小平台` ✗（等于偏好"玩家**胸口
    以上**那块"）；用户要的是**更近的** `#47`，也就是「**底层**」✓ —— 判错的不只是名字：
    "我在哪块平台上"错 ⇒ 归属集合 / 限制战斗区域 / 起点判定**全跟着错** ✗。

    钉四件：
      ① **真实数据照抄现场**：`106010105` 的 (577,193) ⇒ `#47`，且它的集合就是「底层」✓；
      ② 容差内"比中心高"的那几条 ⇒ 取**离得最近**的（fy 最大 ✓）；
      ③ **"身下"不加分**：一律比 |Δy| —— 更近的那条哪怕在中心**上方**也选它 ✓
         （用户第二次现场 (459,154) 就是这么报的 ✓）；
      ④ 一条都不在容差内 ⇒ `None`（不猜 ✓），而且**同一个 x 上的其它层**不许被顺手认走 ✓。
    """
    from core import mapdata, zones

    # ② 容差内、都比中心高的两条 ⇒ 取离得最近的那条（fy 最大）
    t = _terrain([_fh(1, 0, 100, 200, 100),          # 面 y=100（比中心高 38）
                  _fh(2, 0, 130, 200, 130)])         # 面 y=130（比中心高 8）→ 该选它
    got = t.foothold_below(50, 138)                  # 中心 138；tol=40 ⇒ 两条都在容差内
    check(got is not None and got.fid == 2,
          "容差内「比中心高」的那几条里没取**最近**的（用户报的就是这个 ✗）：%s" % got)

    # ③ **"身下"不是加分项**：一律比 |Δy| —— 更近的那条哪怕在中心**上方**也选它 ✓
    #    ⚠ 这件是用户**第二次**现场逼出来的：他报 (459,154) 被判成「底层」，
    #      而更近的那条恰好在中心**上方**几像素 ⇒ 我中途加的"优先认身下那条"就判错了 ✗
    #      （现在去掉了 ✓）。
    t2 = _terrain([_fh(1, 0, 100, 200, 100),         # Δ=−20（更近 ✓ 但在中心上方）
                   _fh(2, 0, 160, 200, 160)])        # Δ=+40（身下，但更远）
    got2 = t2.foothold_below(50, 120)
    check(got2 is not None and got2.fid == 1,
          "「身下」那条更远、却压过了更近的那条（用户 (459,154) 报的就是这个 ✗）：%s" % got2)

    # ④ 全超出容差 ⇒ None（"头顶上方的地面"不许当成脚下）
    check(t.foothold_below(50, 300) is None,
          "全超出容差却还是给了一条（那是硬给 ✗）：%s" % t.foothold_below(50, 300))

    # ① 真实数据：照抄用户现场
    tt, zz = mapdata.load("106010105"), zones.load("106010105")
    f = tt.foothold_below(577.0, 193.0)
    check(f is not None and str(f.fid) == "47",
          "用户 2026-09-27 报的那一处又判错了（该是 #47「底层」✗）：%s" % f)
    check("底层" in zz.set_of(str(f.fid)),
          "判出的那条不是「底层」（用户说的就是这块 ✗）：%s" % zz.set_of(str(f.fid)))
    # 顺带：同一处**上层**的 (577,153) 仍该认「小平台」（别把上面那层也改坏 ✗）
    f2 = tt.foothold_below(577.0, 153.0)
    check(f2 is not None and "小平台" in zz.set_of(str(f2.fid)),
          "上层 (577,153) 判错了（该仍是「小平台」）：%s / %s"
          % (f2, None if f2 is None else zz.set_of(str(f2.fid))))


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


def t_removed_kinds():
    """**「待确认」不是通行方式；「传送点」2026-10-06 又加回来了**（口径变过两次，这条把终点钉住）。

    2026-09-27 用户要求移除「传送门」与「待确认」—— 理由：传送门**没有执行器**、门的落点
    导出数据里也没有（WZ 只给跨图的 `tm` + 目标门名 `tn`，本图内的落点得人指认）；
    "待确认"更不是通行方式（它是编辑器**建议边**的中转标记 ✗）。
    ⭐ **2026-10-06 用户把「传送点」加回来了**（原话："操作方式是走到该点位按↑"✓）—— 那两条
      理由**都堵上了** ✓：
        · **执行器** = `route.PortalJob` ✓（走到位 ⇒ **点按一下 ↑** ✓ 用户定 ✓）；
        · **落点** = **同图**门的 `tn` 就是**目标门名** ✓ ⇒ `zones.portal_exit` 一解析就拿到
          出口坐标 ✓（跨图**明确不支持** ✓ 用户定 ✓）。
    ⚠ 纪律不变：**每种通行方式都必须有执行器** ✓ ⇒ 加 kind 与加执行器**必须同一轮落** ✗
      （只加 kind ⇒ 规划出一条传送点边会当场炸 ✓）。

    钉六件：
      ① 通行方式**正好是** 走/爬/下跳/跳/**传送点**；
      ② 界面名字表 / 优先级表都要有传送点 ✓，而 `unconfirmed` **一个都不许留** ✗；
      ③ "把传送门连成边"那个建议函数仍然**不许有**（`portal_suggestions` ✗ —— 建议只产
         "走 / 爬"两种可信候选 ✓ 自动产不出可信的门 ✓）；
      ④ 老文件残留 `unconfirmed` ⇒ `validate` **必须报** ✓；残留 `portal` **但不指名门** ⇒
         也要报（"没指定是哪扇门" ✓ 同"爬不指名绳" ✓）；
      ⑤ `add_edge` 收哪种 kind：`unconfirmed` **拦住** ✓、`portal` **收下** ✓（身份要含门 ✓）；
      ⑥ 去重身份含门 ⇒ **两扇门是两条边** ✓、同一扇门重复加**幂等** ✓。
    """
    check(set(zones.EDGE_KINDS) == {"walk", "climb", "drop", "jump", "portal"},
          "通行方式该是走/爬/下跳/跳/传送点 五种：%s" % (zones.EDGE_KINDS,))
    for k in zones.EDGE_KINDS:
        check(k in zones.EDGE_LABELS, "「%s」没有界面名字" % k)
    check("portal" in zones.KIND_PRIORITY,
          "传送点没进优先级表 ⇒ 并列时的兜底变成「看运气」✗")
    check("unconfirmed" not in zones.EDGE_LABELS
          and "unconfirmed" not in zones.KIND_PRIORITY,
          "「待确认」不是通行方式，不该留在标签 / 优先级表里 ✗")
    check(not hasattr(zones, "portal_suggestions"),
          "`portal_suggestions` 不该回来 —— 建议只产「走 / 爬」两种可信候选 ✓")
    # 老文件残留「待确认」⇒ validate 要报（portal 现在是正经 kind，见下一条 ✓）
    bad = zones.Zones.from_dict({
        "map_id": MAP_ID,
        "sets": {"A": {"footholds": ["1"]}, "B": {"footholds": ["2"]}},
        "edges": [{"from": "A", "to": "B", "kind": "unconfirmed"}]})
    check(any("已经移除的通行方式" in x for x in bad.validate()),
          "老文件里残留的「待确认」边没报警：%s" % bad.validate())
    # ⭐ 传送点**不指名门** ⇒ 必须报（同"爬不指名绳"✓ 一句话说清怎么办 ✓）
    nop = zones.Zones.from_dict({
        "map_id": MAP_ID,
        "sets": {"A": {"footholds": ["1"]}, "B": {"footholds": ["2"]}},
        "edges": [{"from": "A", "to": "B", "kind": "portal"}]})
    _v = nop.validate()
    check(any("没指定是哪扇门" in x for x in _v),
          "传送点边不指名门居然没报警（§校验该拦它 ✓）：%s" % _v)
    # add_edge：unconfirmed 拦、portal 收（且身份含门 ✓）
    z = zones.Zones(MAP_ID)
    z.add_set("A", ["1"])
    z.add_set("B", ["2"])
    try:
        z.add_edge("A", "B", "unconfirmed")
    except ValueError:
        pass
    else:
        check(False, "`add_edge` 居然收下了 'unconfirmed'（那种走法不存在 ✗）")
    e1 = z.add_edge("A", "B", "portal", portal="h009")
    e2 = z.add_edge("A", "B", "portal", portal="h010")
    check(len(z.edges) == 2 and e1 is not e2,
          "两扇门该是**两条边**（去重身份必须含门 ✗ 否则第二条被静默吞掉 ✓）：%r" % (z.edges,))
    check(z.add_edge("A", "B", "portal", portal="h009") is e1,
          "同一扇门重复加该**幂等**（返回老那条 ✓）")
    check(zones.edge_cond(e1) == "门 h009",
          "列表行里没写出哪扇门（同一对集合两扇门就分不清了 ✗）：%r" % zones.edge_cond(e1))


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
    check(zones.kind_label("climb") == "爬（绳梯）（climb）",
          "没带键的类型该补上键：%s" % zones.kind_label("climb"))
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
    """「爬这根绳」是**向上还是向下**。

    ⭐ **2026-09-28 口径变了**（用户现场指出 ✗）：原来是"按**目标**在绳的上端还是下端判"，
      现在改成"**按起点在哪端判**"（起点在下端 ⇒ 上爬）—— 用户原话：
      "虽然 L2 没有连接左上，但是可以通过在 L2 上跳（即我配的方式）到达左上。因此我们的
       寻路编辑器编的只是**寻路意图**，Agent 只要照着我配的准则去做就行了，至于连不连通、
       是否能成功都不用管，位置状态、任务、执行器会自洽。"

    ⚠⚠ **为什么老口径必须改**：它把"**终点必须是绳的一个端点**"当成了前提，于是和
      「**中途跳下**」（`mid_dir`/`mid_y`，用户 2026-09-28 加的 ✓）**自相矛盾** ——
      中途跳下**本来就意味着终点不在端点**（爬到 `mid_y` 再往侧面跳出去 ✓）。用户的实例
      `右下 --爬 L2（中途跳下·仅向左·高度 50）--> 左上`：L2 的两头是「右下 / 上下过渡平台」，
      `左上` **不在任何一头**，可这条边**完全成立** ⇒ 老口径判"说不清" ⇒ 建不出任务 ⇒
      `pick_edge` 把它剔掉 ⇒ 界面报"**没有一条能走的边**"✗（用户报的就是这个）。
      ⇒ 现在：**方向只看 `src`**；终点不在绳端时，**配了「中途跳下」才放行** ✓。

    钉五件（前四件是**老行为**，必须原样保住 ✓）：
      ① 终点在绳端 ⇒ 上/下照判（真实数据：`左上 →(爬 L1)→ 左上平台` = 向上）；
      ② 终点在**同一端**（自环）/ 起点不在绳端 / 没绳 ⇒ `None`（宁可不做也不猜 ✓）；
      ③ ⭐ **终点不在绳端 + 配了中途跳下 ⇒ 放行**，方向按起点算；
      ④ ⚠ **终点不在绳端 + 没配 ⇒ 仍然 `None`**（那种情况是真到不了，不许猜 ✓）；
      ⑤ ⭐ **两处口径一致**：`job_for_edge`（建任务）和 `hop_cost`（算代价）都要传
         `mid_jump` —— 只改一处的话，边能建出来却在择路时**排到最后**（很隐蔽 ✗）。
    """
    t = mapdata.load(MAP_ID)
    z = zones.load(MAP_ID)
    lids = zones.ladder_ids(t)
    L1 = next((x for x in t.ladders if lids.get(id(x)) == "L1"), None)
    L2 = next((x for x in t.ladders if lids.get(id(x)) == "L2"), None)
    check(L1 is not None, "这张图没有 L1，这条测不了")

    # ---- ①② 老行为：终点在绳端 ----
    check(zones.climb_direction(t, z, L1, "左上", "左上平台") == 1,
          "左上 → 左上平台 该判成向上")
    check(zones.climb_direction(t, z, L1, "左上平台", "左上") == -1,
          "左上平台 → 左上 该判成向下")
    for a, b, why in (("左上", "左上", "起点终点同一个集合（自环）"),
                      ("幽灵", "左上平台", "起点不在绳的两端")):
        check(zones.climb_direction(t, z, L1, a, b) is None,
              "%s：该判不出来（None），不许猜" % why)
    check(zones.climb_direction(t, z, None, "左上", "左上平台") is None,
          "连绳都没有的时候该给 None")

    # ---- ③④ 「中途跳下」：终点不在绳端 ----
    #   ⚠ 这条边是**现造**的（只借真实的绳 L2 ✓）—— 用户的数据随时会改，
    #     用例不该把它当契约 ✗（同 `t_walk_dir` 的做法 ✓）。
    check(L2 is not None, "这张图没有 L2，这条测不了")
    check(zones.climb_direction(t, z, L2, "右下", "左上") is None,
          "**没配**「中途跳下」时，终点不在绳端的边该判不出来（None）—— "
          "那种情况是真到不了，不许猜 ✗")
    check(zones.climb_direction(t, z, L2, "右下", "左上", mid_jump=True) == 1,
          "⭐ **配了**「中途跳下」时，终点不在绳端的边该放行，且方向按**起点**算"
          "（右下是 L2 的**下端** ⇒ 往上爬 +1）—— 这正是用户 2026-09-28 报的那条边 ✗")
    check(zones.climb_direction(t, z, L2, "左上", "右下", mid_jump=True) is None,
          "起点不在绳的两端 ⇒ 就算配了中途跳下也判不出来（不知道从哪上绳 ✗）")
    check(zones.climb_direction(t, z, L2, "右下", "右下", mid_jump=True) is None,
          "自环（起终点同一块集合）⇒ 就算配了中途跳下也判不出来："
          "终点落在绳端、却跟起点**同一端** ⇒ 那是圈错了，不是「上下都行」✗")

    # ---- ⑤ 两处调用点都要传 `mid_jump`（源码级：这是"两套判据漂了"的典型 ✗）----
    from pathlib import Path

    _rt = (Path(__file__).resolve().parents[1] / "decision" / "route.py"
           ).read_text(encoding="utf-8")
    check(_rt.count("mid_jump=bool(edge.get(\"mid_dir\"))") == 2,
          "`mid_jump` 该在 **两处** 都传（`job_for_edge` 建任务 + `hop_cost` 算代价），"
          "现在传了 %d 处 ⇒ 只改一处的话边能建出来、却在择路时排到最后（很隐蔽 ✗）"
          % _rt.count("mid_jump=bool(edge.get(\"mid_dir\"))"))


def t_climb_over_layer():
    """⭐⭐⭐ **「绳只是路过这一层」也能爬**（用户 2026-10-06 ✓ 原话："这个比较特殊，在 **x 范围不
    包含绳梯 x** 的情况下默认走**斜跳上绳梯**，攀爬执行器需要**默认判定出来并支持**"）。

    现场（图 `110040000`，集合 `7` →(爬 L6)→ `6` —— 用户报的"从7去6 发呆"✓）：
      · 集合 7 那一层 x[467..883] y=**-113**；集合 6 在 y=**-293**（在它上面 180px ✓）；
      · L6 x=**423**、两端 y=-291（≈集合 6 那层）和 +32（更下面那层）✓
      ⇒ 起点（7）**不在绳的任何一端** ✗ ⇒ 老判据 `climb_direction` 给 `None` ⇒ 建不出任务 ⇒
        规划报"没有一条能走的边" ⇒ **发呆** ✓。而真相是：**绳从 7 这层穿过去了**（-113 落在
        [-291, +32] 里 ✓）⇒ 人**够得着** ✔ 只是够不到它**正下方**（绳在 423、7 层从 467 起 ⇒
        差 **44px** ✗）⇒ 得**斜跳** ✓。
    ⇒ 两条口径一起落（都从**数据**算 ✓ 不拍数字 ✓）：
      ① 方向：绳路过这一层 ⇒ 看"**目标层贴着绳的哪一端**"✓（`climb_direction` 新分支 ✓）；
      ② 斜跳：**x 范围不含绳 x** ⇒ 默认斜跳 ✓（`climb_via_jump` ✓）、跳得够不够得着
         = **这一层最近的边到绳的距离** ✓（`climb_jump_reach` ✓ 现场 = 44 ✓）。

    合成地形（手算 ✓ 绳下端故意离起点层 300px > `ladder_ends` 的 span 260 ⇒ 起点**查不到**
    任何一端 ⇒ 保证走**新分支** ✗ 不然测的还是老路 ✓）：
      集合「下」fh1 x[100..200] y=**300**；集合「上」fh2 x[100..200] y=**100**；
      集合「底」fh4 x[300..400] y=**600**；集合「中」fh3 x[300..400] y=**250**；
      **L1** x=**60** y[**110**..600]（x 在「下」的 x 范围**外** ⇒ 要斜跳 ✓；y 跨过 300 ⇒ 路过 ✓；
      ⚠ 上端特意取 **110**（不是目标层的 100）⇒ 让"代价用绳端"与"用目标层"差出 10px 分辨得出 ✓）。
    钉七件：① 路过/不含绳 x/跳多远（手算 40 ✓）；② 方向 +1（目标贴上端 ✓）、-1（目标贴下端 ✓）、
    None（目标悬在绳中间 ✗）；③ 绳**不路过**时不放行（仍然 None ✓ 不许乱猜 ✗）；
    ④ `job_for_edge` **默认**把斜跳区间灌进去（不用人配「起跳距离」✓）；
    ⑤ `hop_cost` 算得出且**手算对** —— ⚠ 爬升那一段取的是「**起点层 → 绳的那一端**」✓
       （**与执行器的停止点同一套** ✓ 用户 2026-10-06 特地提醒"代价算法与斜跳上升算法一样" ✓：
        上爬停在绳上端 ✓ ⇒ 用 |绳上端−起点层面| = 190 ✓，**不是**目标层的 y = 200 ✗）；
    ⑥ `plan_jobs` 出任务（对照：改之前这里 **jobs=0** ⇒ 发呆 ✓）；
    ⑦ **执行器真跑**：离绳 40px（≤ 上界 ✓、> 对齐距离 20 ✓）⇒ 那一拍 `jump=True` + `dir` 朝上 ✓。
    """
    from decision import route as route_mod

    t = _terrain([_fh(1, 100, 300, 200, 300),        # 「下」（起点那一层，y=300）
                  _fh(2, 100, 100, 200, 100),        # 「上」（目标层，y=100 —— 贴着绳上端 ✓）
                  _fh(3, 300, 250, 400, 250),        # 「中」（悬在绳中间 ⇒ 该判不出 ✓）
                  _fh(4, 300, 600, 400, 600)],       # 「底」（y=600 —— 贴着绳下端 ✓）
                 ladders=[{"x": "60", "y1": "110", "y2": "600", "l": "1", "page": "0"}])
    z = zones.Zones(MAP_ID)
    for n, fid in (("下", "1"), ("上", "2"), ("中", "3"), ("底", "4")):
        z.add_set(n, [fid])
    L1 = t.ladders[0]

    # ---- ① 三个几何判据（手算）----
    check(zones.ladder_over_set(t, z, L1, "下"),
          "绳 y[100..600] 罩住了「下」那一层（y=300）⇒ 该算「路过」✓")
    check(zones.climb_via_jump(t, z, L1, "下"),
          "绳 x=60 不在「下」的 x[100..200] 里 ⇒ **该斜跳** ✓（用户 2026-10-06 的唯一判据 ✓）")
    check(abs(zones.climb_jump_reach(t, z, L1, "下") - 40.0) < 1e-6,
          "跳多远该是「这一层最近的边到绳」= 100-60 = 40 ✓：%r"
          % zones.climb_jump_reach(t, z, L1, "下"))

    # ---- ② 方向（新分支）----
    check(zones.climb_direction(t, z, L1, "下", "上") == 1,
          "起点不在绳端、但绳路过这一层、目标层贴着**上端** ⇒ 该判**往上爬 +1** ✓：%r"
          % zones.climb_direction(t, z, L1, "下", "上"))
    check(zones.climb_direction(t, z, L1, "下", "底") == -1,
          "目标层贴着**下端** ⇒ 该判**往下爬 -1** ✓：%r"
          % zones.climb_direction(t, z, L1, "下", "底"))
    check(zones.climb_direction(t, z, L1, "下", "中") is None,
          "目标层**悬在绳中间** ⇒ 爬到绳头会冲过头 ⇒ 必须仍然 None（不许猜 ✗）：%r"
          % zones.climb_direction(t, z, L1, "下", "中"))

    # ---- ③ 绳不路过 ⇒ 仍然 None ----
    t2 = _terrain([_fh(1, 100, 300, 200, 300), _fh(2, 100, 100, 200, 100)],
                  ladders=[{"x": "60", "y1": "500", "y2": "700", "l": "1", "page": "0"}])
    z2 = zones.Zones(MAP_ID)
    z2.add_set("下", ["1"])
    z2.add_set("上", ["2"])
    check(zones.climb_direction(t2, z2, t2.ladders[0], "下", "上") is None,
          "绳在这层的**下面**（压根够不着）⇒ 必须 None ✗（放宽口径 ≠ 乱放行 ✗）")
    check(not zones.ladder_over_set(t2, z2, t2.ladders[0], "下"),
          "「路过」判据把够不着的绳也算进来了 ✗")

    # ---- ④⑤⑥ 任务 / 代价 / 规划（合成地形上跑真函数）----
    e = z.add_edge("下", "上", "climb", ladder="L1")
    at = (150.0, 300.0)                     # 人站在「下」的中间
    j = route_mod.job_for_edge(t, z, e, tol_px=6, hold_ms=250, stall_s=1.0)
    check(type(j).__name__ == "ClimbJob" and j.dir == 1,
          "任务没建对：%r dir=%r" % (type(j).__name__, getattr(j, "dir", None)))
    check(abs(float(j.jump_start_px) - 40.0) < 1e-6,
          "⭐ **默认**没把「斜跳上绳」的区间灌进去（用户要的正是「默认判定出来」✗）：%r"
          % j.jump_start_px)
    # 代价 = 走到绳那一段（90px）+ 爬到**绳上端**那一段（|110-300| = 190）⇒ 280 ✓
    # ⚠⚠ **爬升那一段必须跟执行器的停止点同一套**（用户 2026-10-06："注意这种寻路代价算法与
    #   斜跳上升算法一样，检查一下" ✓）：执行器上爬停在**绳上端**（`ClimbJob._arrived` ✓
    #   `dst_y` 不参与上爬判定 ✗）⇒ 该用 |绳上端 − 起点层面| = |110−300| = **190** ✓，
    #   **不是** |目标层 − 起点层面| = |100−300| = 200 ✗（差 10px 正是专门留出来分辨这两者的 ✓）。
    c = route_mod.hop_cost(t, z, e, at)
    check(c is not None and abs(c - 280.0) < 1e-6,
          "代价该 = 90（走到绳）+ 190（爬到**绳上端** ✓ 与执行器同一停止点）= 280 ✓"
          "（用目标层的 y 会得 290 ✗ 照整根绳算更离谱 ✗）：%r" % c)
    p = route_mod.plan_jobs(t, z, "下", "上", at=at)
    check(len(p["jobs"]) == 1 and not p["why"],
          "⭐ 规划该给出 **1 个任务**（改之前这里 jobs=0 ⇒ 界面报「没有一条能走的边」⇒ 发呆 ✗）："
          "%d / %r" % (len(p["jobs"]), p["why"]))

    # ---- ⑦ 执行器真跑：离绳 40px ⇒ **先按住朝绳走，下一拍按跳**（斜跳 ✓）----
    o1 = j.update(0.0, px=100.0, py=300.0, ladder_id=None, here_sets=("下",))
    check(o1["move"] == -1 and not o1["jump"],
          "斜跳前一拍该**先按住朝绳的方向**（不是当场按跳 ✓ 那是它自己的两步节奏 ✓）：%r" % (o1,))
    o2 = j.update(0.05, px=100.0, py=300.0, ladder_id=None, here_sets=("下",))
    check(o2["jump"] and o2["dir"] == 1,
          "离绳 40px（≤ 上界 40 ✓、> 对齐距离 20 ✓）该**斜跳 + 按 ↑**：%r" % (o2,))

    # ---- ⑧ 候选不再说反话：这类绳该被算进「相关」（界面别标「走不到它」✗）----
    check(L1 in zones.ladders_touching(t, z.sets["下"]["footholds"]),
          "「路过这一层」的绳没进候选 ⇒ 界面上还是那句「⚠ 从「下」走不到它」✗（说反话 ✓）")

    # ---- ⑨ ⭐ **人正站在绳底下**（起点不在绳端 + x **罩住**绳 x）—— 不斜跳，但代价同样要对 ----
    #   ⚠ 这是"检查代价与执行器一不一样"逼出来的**第二处**（用户 2026-10-06 ✓）：老条件把
    #     这一类打回老分支 ⇒ 拿**绳端**当入绳点 ⇒ 距离虚大 ⇒ 能建出来却几乎不被选 ✗。
    t3 = _terrain([_fh(1, 100, 300, 200, 300), _fh(2, 100, 100, 200, 100),
                   _fh(3, 300, 250, 400, 250), _fh(4, 300, 600, 400, 600)],
                  ladders=[{"x": "150", "y1": "110", "y2": "600", "l": "1", "page": "0"}])
    z3 = zones.Zones(MAP_ID)
    for n, fid in (("下", "1"), ("上", "2"), ("中", "3"), ("底", "4")):
        z3.add_set(n, [fid])
    L2 = t3.ladders[0]
    check(zones.climb_direction(t3, z3, L2, "下", "上") == 1,
          "起点不在绳端、目标贴上端 ⇒ 照样该判 +1 ✓：%r"
          % zones.climb_direction(t3, z3, L2, "下", "上"))
    check(not zones.climb_via_jump(t3, z3, L2, "下"),
          "绳 x=150 落在「下」的 x[100..200] 里 ⇒ 人正站在绳**底下** ⇒ **不该斜跳** ✓")
    e3 = z3.add_edge("下", "上", "climb", ladder="L1")
    j3 = route_mod.job_for_edge(t3, z3, e3, tol_px=6, hold_ms=250, stall_s=1.0)
    check(float(j3.jump_start_px or 0) == 0.0,
          "x 罩住绳 x 时**不该**启用斜跳（用户口径只看「x 不含绳 x」✓）：%r" % j3.jump_start_px)
    c3 = route_mod.hop_cost(t3, z3, e3, (150.0, 300.0))
    check(c3 is not None and abs(c3 - 190.0) < 1e-6,
          "人站在绳底下 ⇒ 代价该 = 0（横向）+ 190（从这一层爬到绳上端）= 190 ✓"
          "（拿绳端当入绳点会虚大 ✗ 那就是「能建却几乎不被选」✗）：%r" % c3)


def t_walk_dir():
    """「走」的方向类型（2026-09-26 用户要求；**2026-09-27 起真的生效**）：能填、能存、能校验。

    它原来只是**配置占位**（"执行器还没做只按那一边"）—— 用户 2026-09-27 就是踩在这个上
    （配了「仅向左」，人却被按着 → 往集合中点走 ✗）。现在 `route.WalkJob.walk_dir` 会按它
    发键 ⇒ 更要钉住"存得住 / 读得回 / 认不出的拦住"，不然逻辑跑起来才发现存进来的根本不是
    那三样东西 ✗。顺带钉 `edge_text`：**「命令前往」那行显示的就是它**，配了方向必须看得出来 ✓。
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
    # 「命令前往」那行显示的就是 `edge_text` ⇒ 配了方向**必须看得出来**（用户 2026-09-27
    # 卡住的原因正是分不清"到底选的哪条边、它要往哪边走"）
    check("仅" in zones.edge_text(z2, "甲", "乙"),
          "「仅向左」的边没在步骤说明里写出来：%r" % zones.edge_text(z2, "甲", "乙"))
    check("仅" not in zones.edge_text(z2, "乙", "甲"),
          "默认方向的边不该带方向后缀：%r" % zones.edge_text(z2, "乙", "甲"))
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

def t_edge_identity_includes_ladder():
    """边的**身份**里有**绳号**：同一对集合、同一类型、**不同绳 = 两条边**（2026-09-27 修）。

    用户 2026-09-27 报的 bug（地图 105040303）：里面已经有 `二楼 →(爬 L3)→ 三楼`，
    再加 `二楼 →(爬 L7)→ 三楼` 被当成**重复** ⇒ `add_edge` **静默返回老那条** ⇒
    界面上"点击增加后无添加项目" ✗（一个字的提示都没有，看着就像程序坏了）。
    判据原来是 `(from, to, kind, portal)`：**门**当初就进去了（走哪个门是另一条边 ✓），
    **绳**却漏了 —— 这两件事是对称的 ✓。
    """
    z = zones.Zones(MAP_ID)
    z.add_set("二楼", ["1"])
    z.add_set("三楼", ["2"])
    e1 = z.add_edge("二楼", "三楼", "climb", ladder="L3", why="第一条")
    check(len(z.edges) == 1, "第一条就没加进去")
    # **完全一样** ⇒ 幂等（还是那一条，返回值也是它 —— 调用方可能是"采纳建议"那种重放 ✓）
    again = z.add_edge("二楼", "三楼", "climb", ladder="L3", why="重放")
    check(len(z.edges) == 1 and again == e1,
          "完全一样的边该幂等（返回老那条），实际 %d 条" % len(z.edges))
    # ⚠ **换一根绳 = 另一条边** —— 用户要的就是这一下
    e2 = z.add_edge("二楼", "三楼", "climb", ladder="L7", why="另一根绳")
    check(len(z.edges) == 2,
          "换一根绳被当成重复了（用户报的「点增加没反应」）：%s" % z.edges)
    check(e2.get("ladder") == "L7" and e2 is not e1, "第二条没建对：%r" % e2)
    # ⭐ 身份里**还有 `portal`**（走哪个门）那一格 —— 2026-09-27 随「传送门」移除过 ✓、
    #   **2026-10-06 加回** ✓。它与绳号**同理**：同一对集合、同是传送点、**不同门**是
    #   两条不同的边 ✓（落点不同、代价也不同 ✓ 用户 2026-10-06 定的口径 ✓）。
    p1 = z.add_edge("二楼", "三楼", "portal", portal="h009", why="第一扇门")
    p2 = z.add_edge("二楼", "三楼", "portal", portal="h010", why="另一扇门")
    check(len(z.edges) == 4 and p1 is not p2,
          "换一扇门被当成重复了（身份里少一格 ⇒ 第二条被**静默吞掉** ✗）：%s" % (z.edges,))
    check(p2.get("portal") == "h010", "第二条门没存对：%r" % p2)
    check(len(z.edges) == 4, "加了不该加的东西：%s" % z.edges)
    # 同一对、同一类型、多条时的**取法**：按类型优先级，同类型不同绳取**先出现**的那条。
    # 把这条钉住，免得以后有人"顺手"换成别的口径（寻路用哪根绳靠它 ✓）
    got = zones.edge_between(z, "二楼", "三楼")
    check(got.get("ladder") == "L3",
          "同类型多条时的取法变了（该取文件里先出现的）：%r" % got)


def t_portal_align_like_climb():
    """⭐⭐ 传送点的**对齐 / 补按**与爬绳执行器**同一套**（用户 2026-10-06 ✓ 两句话：
    "去传送门的对齐需要调用攀爬执行器对齐那个方法：在「坐标误差范围」内就可以按 ↑
      ＋ 位置状态没回执补按 ↑"✓；看完日志又补："看 log 会在从5到4的 h010 传送门
     **左右来回晃但是没有按 ↑** —— 我们应该**整套照搬对齐式爬绳梯的逻辑**，**还要补按方向**"✓）。

    钉七件（①~⑥ 都在 `PortalJob` 上真跑 ✓）：
      ① **对齐判据 = 「坐标对齐误差范围」**（`tol_px` ✓ 与 `ClimbJob.tol_px` 同一把尺 ✓）
         ＋ **进容差先站住稳 `hold_ms`**（「坐标对齐误差时间」✓ 照搬爬绳"站住等 ⇒ 再动作"✓
         —— 不做这道就是"路过一下也按 ↑"✗）；`hold_ms=0` ⇒ **同一拍就按**（不白拖 ✓）；
      ② **还没对齐 ⇒ 按住朝门走**（远超 `near_px` 一口气按住 ✓ 近了才点按 ✓ 同爬绳分档 ✓）；
      ③ ⭐ **不会在门口来回晃**（用户报的病 ✗）：**过冲之后往回按（换了方向）也要等
         「对齐绳梯移动延迟」** ✓ —— 没过间隔那一拍必须**一个键都不按** ✓；
      ④ **近处点按**：按够 `TAP_ON_S`(30ms) 就**松开**（别一路蹭过去 = 冲过头 ✗）；
      ⑤ ⭐ **补按方向**（用户："**还要补按方向**"✓）：按着某个方向却**一直没进步**、
         过了「移动操作尝试间隔」⇒ `reassert=True` ＋ 还是那个方向 ✓（对面把键丢了也能救 ✓）；
         没到间隔 ⇒ **不补**（否则就是键盘风暴 ✗）；`retry_ms=0` ⇒ 一个都不补 ✓（老行为 ✓）；
      ⑥ ⭐ **按 ↑ 只看 x 对齐**（照搬爬绳那条："对齐了就按跳开始爬啊，**不需要判定别的**"✓
         见 `ClimbJob._align_step` ✓）：y 差 50px（`> PORTAL_TOL_Y`）**照样按 ↑** ✓，
         但**差多少必须写进 note** ✓（人就靠日志判断"为什么按了没反应"✓）；
      ⑦ 真规划那条路：`plan_jobs` 一起灌的 `tol_px` / `hold_ms` **真落进** `PortalJob` ✓
         （`tol_px` 原来在 `portal_job_for_edge` 里被摘掉 ✗ ⇒ 用的是它自己那个"门的框"✓）。
    """
    from decision import route as route_mod

    t = _terrain([_fh(1, 0, 300, 100, 300),          # 「下」：x[0..100] y=300
                  _fh(2, 600, 300, 700, 300)],       # 「上」：x[600..700] y=300
                 portals=[{"pn": "h009", "pt": "1", "x": "50", "y": "280",
                           "tm": "999999999", "tn": "h010"},
                          {"pn": "h010", "pt": "1", "x": "650", "y": "280",
                           "tm": "999999999", "tn": ""}])
    z = zones.Zones(MAP_ID)
    z.add_set("下", ["1"])
    z.add_set("上", ["2"])
    pe = z.add_edge("下", "上", "portal", portal="h009", why="用例：本图内的门")
    p = zones.portal_by_pn(t, "h009")
    ex = zones.portal_exit(t, "h009")
    check(p is not None and ex is not None, "用例地形没造好（门 / 出口解析不出来）")

    # ---- ① 对齐判据（＋ 先稳住 ＋ ⭐**进容差就点按 ↑**，用户 2026-10-09 修正口径 ✓）----
    j = route_mod.PortalJob("上", p, ex, tol_px=50, hold_ms=200, retry_ms=0)
    o = j.update(0.0, px=10.0, py=280.0)            # 差 40px ≤ 容许 50 ⇒ 对齐好了 ✓
    check(o["move"] == 0 and "对齐好了" in j.note,
          "差 40px、容许 ±50 ⇒ 判「对齐好了」、站住（照搬爬绳：稳 hold_ms ✓）：%s / %r"
          % (o, j.note))
    check(o["dir"] == 1 and j._up_taps == 1,
          "⭐ 这一拍**已经在「坐标对齐误差范围」里**（差 40 ≤ 容许 50）⇒ 就该点按 ↑"
          "（2026-10-09 口径：起拍点 = **首次进容差** ✓ 这一拍正是首次 ✓）：%s" % (o,))
    o = j.update(0.07, px=10.0, py=280.0)           # 按够 `TAP_ON_S`(30ms) ⇒ 松手 ✓
    check(o["dir"] == 0,
          "点按要**按够 `TAP_ON_S` 就松**（一路按着对面只算一次长按 ✗）：%s" % (o,))
    o = j.update(0.25, px=10.0, py=280.0)           # 稳够 200ms ⇒ 那一下「正式的 ↑」
    check(o["dir"] == 1 and o["move"] == 0,
          "稳够「坐标对齐误差时间」⇒ 该**只按 ↑**（`dir=+1` 才是 ↑ ✓ 按消费点口径 ✓）：%s" % (o,))
    j0 = route_mod.PortalJob("上", p, ex, tol_px=50, hold_ms=0, retry_ms=0)
    o0 = j0.update(0.0, px=10.0, py=280.0)
    check(o0["dir"] == 1,
          "`hold_ms=0`（不等）⇒ **同一拍就该按 ↑**（别白拖一拍 ✓ 同爬绳 ✓）：%s" % (o0,))

    # ---- ② 还没进容差 ⇒ 横向朝门走，⭐**但远处一个 ↑ 都不许按**（用户 2026-10-09 ✓）----
    j2 = route_mod.PortalJob("上", p, ex, tol_px=10, hold_ms=200, retry_ms=0)
    o2 = j2.update(0.0, px=10.0, py=280.0)          # 差 40px > 容许 10（远）
    check(o2["move"] == 1, "差 40px、容许 ±10 ⇒ 该**朝门走**：%s" % (o2,))
    check(o2["dir"] == 0 and j2._up_taps == 0,
          "⭐⭐ **还差 40px（没进「坐标对齐误差范围」±10）就点按 ↑** ✗ —— 用户 2026-10-09 "
          "原话：「对齐期间搁很远就按 ↑ 是不对的」✓：%s" % (o2,))

    # ---- ③④ 近处点按 + **过冲往回按也得等「移动延迟」**（用户报的"左右晃"就是这个 ✗）----
    j3 = route_mod.PortalJob("上", p, ex, tol_px=2, hold_ms=200, retry_ms=0,
                             align_gap_ms=180, near_px=20)
    o3 = j3.update(0.0, px=45.0, py=280.0)          # dx=+5（超容许、在 near_px 内）⇒ 点按"→"
    check(o3["move"] == 1, "差 +5px ⇒ 该朝右点一下：%s" % (o3,))
    o3 = j3.update(0.05, px=55.0, py=280.0)         # 冲过头 ⇒ dx=-5 ⇒ **这一拍不许反按**
    check(o3["move"] == 0 and "等移动延迟" in j3.note,
          "过冲之后**立刻反按** ⇒ 人就在门口左右晃 ✗（用户 2026-10-06 现场报的正是这个 ✗）："
          "%s / %r" % (o3, j3.note))
    o3 = j3.update(0.20, px=55.0, py=280.0)         # 过了 180ms ⇒ 才允许反按
    check(o3["move"] == -1, "过了「对齐绳梯移动延迟」该反按回来了：%s" % (o3,))
    o3 = j3.update(0.20 + route_mod.TAP_ON_S + 0.01, px=55.0, py=280.0)
    check(o3["move"] == 0 and "松开" in j3.note,
          "近处点按：按够 `TAP_ON_S`(30ms) 就该**松开**（一路蹭过去 = 冲过头 ✗）：%s / %r"
          % (o3, j3.note))

    # ---- ⑤ 补按方向（用户："还要补按方向"✓）----
    j5 = route_mod.PortalJob("上", p, ex, tol_px=10, hold_ms=200, retry_ms=500,
                             align_gap_ms=180, near_px=20)
    j5.update(0.0, px=10.0, py=280.0)               # 远 ⇒ 一口气按住"→"
    o5 = j5.update(0.3, px=10.0, py=280.0)          # 位置**没动**、但还没到间隔
    check(o5["move"] == 1 and not o5.get("reassert"),
          "还没到「移动操作尝试间隔」就补按了（那是键盘风暴 ✗）：%s" % (o5,))
    o5 = j5.update(0.6, px=10.0, py=280.0)          # 过了 500ms 还是没动 ⇒ 补按一次
    check(o5.get("reassert") is True and o5["move"] == 1,
          "按着某个方向却**一直没进步**、过了「移动操作尝试间隔」⇒ 该**补按一次方向键**"
          "（`reassert=True` ✓ 用户 2026-10-06：「还要补按方向」✗）：%s" % (o5,))
    j5b = route_mod.PortalJob("上", p, ex, tol_px=10, hold_ms=200, retry_ms=0,
                              align_gap_ms=180, near_px=20)
    j5b.update(0.0, px=10.0, py=280.0)
    o5b = j5b.update(9.0, px=10.0, py=280.0)
    check(not o5b.get("reassert"),
          "`retry_ms=0`（= 不重试，老行为 ✓）却补按了 ✗：%s" % (o5b,))

    # ---- ⑥ 按 ↑ 只看 x 对齐（照搬爬绳那条 ✓）＋ y 差写进 note ✓ ----
    j6 = route_mod.PortalJob("上", p, ex, tol_px=50, hold_ms=0, retry_ms=0)
    o6 = j6.update(0.0, px=10.0, py=330.0)          # y 差 50px（> PORTAL_TOL_Y=12）
    check(o6["dir"] == 1,
          "按 ↑ 只看 x 对齐（照搬爬绳「对齐了就按，不需要判定别的」✓）——"
          " y 差 50px 就不按了 ✗：%s" % (o6,))
    check("y" in j6.note and "50" in j6.note,
          "y 差多少该**写进 note** ✓（人正靠日志判断「为什么按了没反应」✓）：%r" % (j6.note,))

    # ---- ⑥′ ⭐⭐ 本轮口径本体（用户 **2026-10-09** 修正"起拍点"）：**首次到达「坐标对齐
    #      误差范围」内才开始连按 ↑** —— 原话："传送点执行器，**对齐期间搁很远就按 ↑ 是不对
    #      的**，需要是要在**首次到达「坐标对齐误差范围」内才开始连按 ↑**"✓。
    #      它替掉了 2026-10-06 那条"进对齐相就开打"（那条实测在**还差 40px** 时就已经一路 ↑ ✗）。
    #      ⚠ 时间**不手算**：按 0.1 秒一拍往前喂，只数"按了几下" ✓（手算节拍踩过坑 ✗）。
    #      ⚠⚠ **`hold_ms` 必须给大**（这里 2000ms）+ **`stall_s` 给足** ✗：这两条不是随手写的 ✓
    #        —— `hold_ms=0` 时人一进容差就**离开对齐相**、改由 `TAP/WAIT` 相按 ↑（那半条口径
    #        没变 ✓）⇒ 那样就**数不到**"对齐期间一直连按"了 ✗（第一版就是这么红的 ✓）；
    #        而"原地不动"喂久了会被"走不动"兜底判死 ⇒ 也得给足 `stall_s` ✓。
    j8 = route_mod.PortalJob("上", p, ex, tol_px=2, hold_ms=2000, retry_ms=500,
                             stall_s=10.0)
    # ⚠⚠ **时钟变量别叫 `t`** ✗（2026-10-09 当场踩过 ✓）：本用例的**地形**就叫 `t`
    #   （见 771 行 `t = _terrain(...)` ✓）—— 我第一版写成 `t = 0.0` ⇒ **把地形覆盖成
    #   float** ⇒ 走到 ⑦ 那句 `job_for_edge(t, ...)` 就炸成
    #   `ValueError: 「传送点」边引用的门本图里没有：'h009'`（探针打出来 `type=float` ✓）
    #   —— 看着像"产品的门查找坏了"，其实是**用例自己的名字撞车** ✗✓。
    _t8 = 0.0
    _far = []
    for _ in range(15):                       # 远处喂 1.5 秒（远超一个点按周期 180ms ✓）
        _t8 += 0.1
        _far.append(j8.update(_t8, px=10.0, py=280.0))
    check(all(r["dir"] == 0 for r in _far) and j8._up_taps == 0,
          "⭐⭐ 远处（差 40px、容许 ±2）2 秒里居然按了 **%d** 下 ↑ ✗ "
          "（用户 2026-10-09 要修的就是这个）：%r" % (j8._up_taps, [r["dir"] for r in _far][:8]))
    # ⭐ **首次进容差那一拍 ⇒ 立刻起拍**（不空等 ✓ —— 人一站到门口就开始试门 ✓）
    _t8 += 0.1
    o = j8.update(_t8, px=49.0, py=280.0)     # 差 1px ≤ 容许 2 ⇒ 首次进容差
    check(o["dir"] == 1 and j8._up_taps == 1,
          "⭐ **首次进「坐标对齐误差范围」那一拍**就该点按 ↑（要的就是「到了门口就开始试」✓）："
          "%s" % (o,))
    # 进了容差 ⇒ **连着打**（节拍 = `TAP_PERIOD_S` ✓ 老口径那半条没变 ✓）
    _n0 = j8._up_taps
    for _ in range(10):                       # 容差内再喂 1 秒 ⇒ 该又多几下（≈5）
        _t8 += 0.1
        j8.update(_t8, px=49.0, py=280.0)
    check(j8._up_taps >= _n0 + 4,
          "进容差之后没连着打：1 秒只多了 %d 下 ✗（该 ≈5 下 ✓）" % (j8._up_taps - _n0))
    # ⭐ **过冲又出容差 ⇒ ↑ 照旧连着打**（"武装"是只认第一次的闩 ✓ 别在门口打一半停下 ✗）
    _n1 = j8._up_taps
    for _ in range(10):                       # 又回到"差 40px"（出容差）
        _t8 += 0.1
        j8.update(_t8, px=10.0, py=280.0)
    check(j8._up_taps >= _n1 + 4,
          "进过容差之后走开了，↑ 就**断了** ✗（口径是「首次到达之后就一直连按」✓）：多了 %d 下"
          % (j8._up_taps - _n1))
    # ⭐ `retry()` ⇒ **复位成"没武装"**：新一轮要重新走到容差里才起拍 ✗（不然远处又一路 ↑ ✗）
    j8.retry()
    check(j8._up_armed is False and j8._up_taps == 0,
          "`retry()` 没复位 `_up_armed` ⇒ 新一轮在远处又会一路按 ↑ ✗")
    _r = []
    for _ in range(10):
        _t8 += 0.1
        _r.append(j8.update(_t8, px=10.0, py=280.0))
    check(all(r["dir"] == 0 for r in _r) and j8._up_taps == 0,
          "`retry()` 之后在远处又按 ↑ 了 ✗：%r" % ([r["dir"] for r in _r][:8],))
    # ⭐ **与「移动操作尝试间隔」无关**（用户 2026-10-06 第二轮口径 ✓ **没变**）：
    #   **进了容差之后** `retry_ms=0` 也照样按「点按周期」连续发 ✓
    #   （`retry_ms` 只再管 `WAIT` 相的"补按 ↑" ✓）。
    j9 = route_mod.PortalJob("上", p, ex, tol_px=50, hold_ms=2000, retry_ms=0,
                             stall_s=10.0)
    t9 = 0.0
    o = j9.update(t9, px=10.0, py=280.0)       # 差 40px ≤ 容许 50 ⇒ 一上来就在容差里 ✓
    check(o["dir"] == 1 and j9._up_taps == 1, "容差内第一拍没按 ↑：%s" % (o,))
    for _ in range(10):
        t9 += 0.1
        j9.update(t9, px=10.0, py=280.0)
    check(j9._up_taps >= 5,
          "`retry_ms=0` 在容差内也没有连续点按 ✗（只 %d 下）" % j9._up_taps)

    # ---- ⑦ 真规划那条路：`tol_px` / `hold_ms` 要真落进任务 ----
    j7 = route_mod.job_for_edge(t, z, pe, tol_px=42, hold_ms=333, stall_s=1.0,
                                jump_start_px=0)
    check(type(j7).__name__ == "PortalJob" and int(j7.tol_px) == 42
          and int(j7.hold_ms) == 333,
          "`plan_jobs` 灌的「坐标对齐误差范围 / 时间」没进传送点任务：tol=%r hold=%r"
          % (getattr(j7, "tol_px", None), getattr(j7, "hold_ms", None)))


def t_portal_kind_and_cost():
    """⭐⭐ **传送点**：出口解析 / **代价按用户那条公式** / 择路里真的参与比 / 执行器真跑一遍
    （用户 2026-10-06 ✓ 口径与落地见 `docs/开发日志.md` **220** ＋ SKILL ✓）。

    用户在问答里把口径一条条定死了 ✓：
      · "**跨图传送点不支持**" ✓（跨图的门解析不出落点 ⇒ `portal_exit` 给 None ✓）；
      · "**点一下 ↑**" ✓（`PortalJob` 按 `TAP_ON_S` 点按、不按住 ✓）；
      · "至于**寻路代价**：我们应该能知道这个传送点通向哪里，寻路代价就等于**角色当前到传送
        起点的距离** ＋ **传送终点距离到目标点的距离**" ✓ —— 这条**必须手算对**（本用例的核心 ✓）。

    钉六件：
      ① **出口解析**：同图门（`tm == 999999999`）按 `tn` 找到出口 ✓；**跨图 / tn 找不到
         ⇒ None** ✓（不猜 ✓）；
      ② **代价 = near + back**（手算对：人在 (100,300)、门在 (50,280)、出口在 (650,280)、
         目标集合面在 x[600..700] y=300 ⇒ `sqrt(50²+20²) + (0 + 20)` ✓）；
      ③ **择路里真参与比**：同一个 `at`，走很贵时挑传送点 ✓、人已经快到目标时挑走 ✓
         （不是"传送点永远优先/永远靠后" ✗）；
      ④ **建任务**：同图门 ⇒ `PortalJob`（且带上出口 ✓）；跨图门 ⇒ **抛**（如实说 ✗ 别静默）；
      ⑤ **执行器真跑**：远处 ⇒ 朝门走；进门 ⇒ `dir=-1`（**点按 ↑** ✓）；按完 ⇒ 松手等；
         到出口 / 踏上目标集合 ⇒ `done` ✓；
      ⑥ `edge_text` 里**写得出是哪扇门**（同一对集合两扇门时靠它分辨 ✓）。
    """
    from decision import route as route_mod

    t = _terrain([_fh(1, 0, 300, 100, 300),          # 「下」：x[0..100] y=300
                  _fh(2, 600, 300, 700, 300)],       # 「上」：x[600..700] y=300
                 portals=[{"pn": "h009", "pt": "1", "x": "50", "y": "280",
                           "tm": "999999999", "tn": "h010"},
                          {"pn": "h010", "pt": "1", "x": "650", "y": "280",
                           "tm": "999999999", "tn": ""},
                          {"pn": "zz", "pt": "1", "x": "900", "y": "280",
                           "tm": "105090600", "tn": "yy"}])
    z = zones.Zones(MAP_ID)
    z.add_set("下", ["1"])
    z.add_set("上", ["2"])

    # ---- ① 出口解析：**先把"同图"的两种写法都钉住**（用户 2026-10-06 报的 bug ✓）----
    #   ⚠⚠ 用户原话："**阳光沙滩的 h00x 这些传送点的地图 id 不就是阳光沙滩本身吗？为什么写着
    #     跨图？**" ✓ —— 老判据只认 `tm == 999999999`（当成"本图内"✗），而**实数据里同图门的
    #     `tm` 就是本图的地图 id**（`110040000` ✓）⇒ 整张图的同图门**全被误判成跨图** ✗
    #     （`h009 → h010` 就在本图，也照样被挡 ✓）。
    #   ⇒ 判据改成"**`tn` 指的那扇门在不在本图**"✓（`tm` 只用于话术 ✓）；这里真图回归。
    t_real = mapdata.load("110040000")
    if t_real is not None:                    # 老环境没导这张图 ⇒ 跳过（别红 ✗）
        _e8 = zones.portal_exit(t_real, "h008")
        check(_e8 is not None and _e8.pn == "h009",
              "同图门（`tm` = 本图 id ✓ 实数据就是这样）解析不出出口 ⇒ 又回到「跨图」误判 ✗：%r"
              % (_e8,))
        check(zones.portal_exit(t_real, "h009") is not None
              and zones.portal_exit(t_real, "h009").pn == "h010",
              "用户点名的 `h009 → h010` 解析不出来 ✗：%r"
              % (zones.portal_exit(t_real, "h009"),))
        check(zones.portal_exit(t_real, "west00") is None,
              "**真跨图**的门（`tn=east00` 不在本图）该给 None ✓：%r"
              % (zones.portal_exit(t_real, "west00"),))
        check(zones.portal_exit(t_real, "h006") is None,
              "`tn` 是空的门（没配去向）该给 None ✓：%r"
              % (zones.portal_exit(t_real, "h006"),))

    # ---- ① 出口解析 ----
    check(zones.portal_exit(t, "h009") is not None
          and zones.portal_exit(t, "h009").pn == "h010",
          "同图门没解析出出口：%r" % (zones.portal_exit(t, "h009"),))
    check(zones.portal_exit(t, "zz") is None,
          "**跨图**门居然解析出了出口（用户明确不支持跨图 ✓ 必须 None ✗）")
    check(zones.portal_exit(t, "没有这扇门") is None, "本图没有的门该给 None")

    pe = z.add_edge("下", "上", "portal", portal="h009", why="用例：本图内的门")
    zw = z.add_edge("下", "上", "walk", why="用例：走")
    # ---- ⑥ 行里写得出是哪扇门（⚠ `edge_cond` 那条在 t_removed_kinds 里钉 ✓）----
    check("h009" in zones.edge_text(z, "下", "上", edge=pe),
          "「命令前往」那行没写出哪扇门（两扇门就分不清了 ✗）：%r"
          % zones.edge_text(z, "下", "上", edge=pe))

    # ---- ② 代价 = near + back（手算）----
    at = (100.0, 300.0)
    near = ((100 - 50) ** 2 + (300 - 280) ** 2) ** 0.5        # 人 → 起点门 = 53.85
    # 出口门 (650,280) → 目标面 x[600..700] y=300：横向落在面里 ⇒ 0 ✓ ＋ 垂直 20 ✓
    back = 0.0 + abs(280 - 300)
    c = route_mod.hop_cost(t, z, pe, at)
    check(c is not None and abs(c - (near + back)) < 0.01,
          "传送点代价不是「当前→起点门 ＋ 出口门→目标」（用户 2026-10-06 定的口径 ✗）："
          "算得 %r，手算 %r" % (c, near + back))
    # 跨图那条算不出来 ⇒ None（不猜 ✓；择路会把它排到最后 ✓）
    zc = zones.Zones(MAP_ID)
    zc.add_set("下", ["1"])
    zc.add_set("上", ["2"])
    zc.add_edge("下", "上", "portal", portal="zz")
    check(route_mod.hop_cost(t, zc, zc.edges[0], at) is None,
          "跨图门的代价该算不出来（None ⇒ 不猜 ✓）")

    # ---- ③ 择路：走贵 ⇒ 挑门；快到目标 ⇒ 挑走（两个方向都要成立 ✗ 别只钉一边）----
    p1 = route_mod.pick_edge(t, z, "下", "上", at=at)
    check(p1 is not None and p1.get("kind") == "portal",
          "人在起点这一头（离门 53.9、离目标面 500）⇒ 该挑传送点 ✓：%r" % (p1,))
    p2 = route_mod.pick_edge(t, z, "下", "上", at=(650.0, 300.0))
    check(p2 is not None and p2.get("kind") == "walk",
          "人已经贴在目标那一头（离目标面 0、离门 600）⇒ 该挑走 ✓"
          "（传送点**不是**永远优先 ✗）：%r" % (p2,))

    # ---- ③b ⚠⚠ **按 `plan_jobs` 的传参方式**也得建得出 ------------------------------------
    #   `plan_jobs` 给四种边**一起灌** `tol_px` / `hold_ms` / `stall_s` / `jump_start_px` ✓
    #   ⇒ 传送点分支漏摘一个就 `TypeError: unexpected keyword argument` ✗ ——
    #   **实测漏过 `tol_px`** ✓：直接 `job_for_edge(...)` 的用例看不出来（它没带那些 kw ✗），
    #   而**真规划路径**全炸 ✗（"传送点边在实机上从来没生效过"✓ 这种最难查 ✓）。
    j4 = route_mod.job_for_edge(t, z, pe, tol_px=6, hold_ms=250, stall_s=1.0,
                                jump_start_px=0)
    check(type(j4).__name__ == "PortalJob",
          "带上 `plan_jobs` 那套参数就建不出来了（漏摘 kw ✗ 真机上一次都跑不起来）：%r"
          % (type(j4).__name__,))

    # ---- ④ 建任务：同图门 ⇒ PortalJob（带出口）；跨图 ⇒ 抛 ----
    job = route_mod.job_for_edge(t, z, pe)
    check(type(job).__name__ == "PortalJob" and getattr(job, "exit_portal", None) is not None
          and job.exit_portal.pn == "h010",
          "同图门该建出 PortalJob 且带上出口：%r" % (job,))
    try:
        route_mod.job_for_edge(t, zc, zc.edges[0])
    except ValueError as ex:
        check("跨图" in str(ex), "抛了，但没说清是「跨图」（用户口径要如实说 ✓）：%r" % (ex,))
    else:
        check(False, "**跨图**门居然也建出了任务（用户明确不支持 ✓ 该抛 ✗）")

    # ---- ⑤ 执行器真跑：走 → 对齐 + **稳住** → 点按 ↑ → 等 → 到 ----
    o = job.update(0.0, px=0.0, py=300.0)
    check(o["move"] == 1, "门在右边（x=50、人在 0）⇒ 该往右走：%r" % (o,))
    # ⭐ 进容差 ⇒ **横向站住**＋**立刻起拍点按 ↑**（两轮口径合并后的样子 ✓）：
    #   · 2026-10-06："**对齐期间一直在点按 ↑**"（稳住那几拍也照按 ✓ —— 产品自己那句 note
    #     "对齐期间一直在点按 ↑，已打 N 下"就是这么写的 ✓）；
    #   · 2026-10-09：**起拍点 = 首次进容差**（在那之前"很远就按"是错的 ✗ 见 `route.PortalJob` ✓）
    #     —— 这一拍正是首次 ✓。
    #   ⚠ 这条断言原来写的是 `dir == 0`（"先站住、先别按 ↑"）⇒ 与上面那两轮口径**正面矛盾** ✗
    #     （10-06 改口径时漏改的一条 ✓ 2026-10-09 一起对齐 ✓）。
    o = job.update(0.1, px=50.0, py=280.0)
    check(o["dir"] == 1 and o["move"] == 0 and "对齐好了" in job.note,
          "进「坐标对齐误差范围」的第一拍该**站住（横向不动）＋ 点按 ↑**：%r / %r"
          % (o, job.note))
    _t = 0.1 + job.hold_ms / 1000.0 + 0.01        # 稳够 ⇒ 该点按 ↑ 了 ✓
    o = job.update(_t, px=50.0, py=280.0)
    check(o["dir"] == 1 and o["move"] == 0,
          "稳够之后该**只按 ↑**（⚠ 按**消费点**的口径：`dir=+1` 才是 ↑ ✗ "
          "—— `agent._climb_tick` 里是 `km.get(\"up\") if out[\"dir\"] > 0 else km.get(\"down\")` ✓；"
          "这里**不能**照实现写期望 ✗ 那正是漏过这个 bug 的原因 ✓）：%r" % (o,))
    # ⭐ **契约钉**（用户 2026-10-06 现场："站在传送点门口不按↑"✗ —— 根因就是 `dir` 正负写反 ✓）：
    #   断言**消费点那行**还在（谁改了它的语义、或把 `dir` 的约定换掉，这条当场红 ✓）。
    _ag = (ROOT / "decision" / "agent.py").read_text(encoding="utf-8")
    check('km.get("up") if out["dir"] > 0 else km.get("down")' in _ag,
          "`agent` 里「`dir > 0` ⇒ ↑」这条约定变了 ⇒ 传送点（以及爬绳/下跳）的竖直键会反向 ✗")
    o = job.update(_t + route_mod.TAP_ON_S + 0.01, px=50.0, py=280.0)
    check(o["dir"] == 0 and not o["done"],
          "按够 `TAP_ON_S` 就该**松手**（用户定：点一下 ✓ 不是按住 ✗）：%r" % (o,))
    o = job.update(_t + 0.5, px=650.0, py=280.0, here_sets=("上",))
    check(o["done"], "已经到了目标集合 / 出口门上，还没判完成：%r" % (o,))
    check(not o["failed"], "正常走到出口却判失败了：%r" % (o,))
    check("到达" in job.note, "完成时那句说明不对：%r" % (job.note,))


def t_pick_nearest_rope():
    """到目标集合有**多条绳**时挑**最近的**（2026-09-27 用户要求）。

    `zones.edge_between` 只看类型优先级、**手里没有玩家位置** ⇒ 同类型的多条只能取
    文件顺序 ✗。择路（`decision.route.pick_edge`）有位置：按玩家到**入绳点**的距离
    取最近的那根 ✓（一张图两根绳接同一对平台很常见，挑近的省一段走位）。

    钉四件：
      ① **挑最近**：人在右边 ⇒ 右边那根绳；人在左边 ⇒ 左边那根；
      ② ⚠ 2026-09-27 起**类型优先级不再是硬性在前**：真正的判据是 `hop_cost`
         （纯距离：走/跳 = 到目标 fh 端点距离；爬 = 到入绳点距离 + 绳长；下跳 = 到可下跳
         fh 距离 + 落差 ✓）—— **跨类型也一起比** ✓（本条用例里两根绳等长，所以"挑最近"
         仍然成立 ✓；跨类型的比法见 `t_hop_cost_and_pick_edge` ✓）；
      ③ 没有位置（`at=None`）⇒ **退回老口径**：类型优先级（走 > 爬 > 下跳 > 跳）→
         **文件顺序**（算不了就不猜 ✗）；
      ④ **端到端**：`plan_jobs(at=…)` 造出来的 `ClimbJob` 就是那根最近的绳
         （`ladder_id` / `x`），而且"命令前往"那行显示的也是**它**（说的和做的一致 ✓）。
    """
    from decision import route as route_mod

    # 下平台 y=300、上平台 y=100（各一条 foothold，横向都盖住两根绳的位置）
    t = _terrain([_fh(1, 0, 300, 600, 300),
                  _fh(2, 0, 100, 600, 100)],
                 ladders=[{"x": "50", "y1": "300", "y2": "145",
                           "l": "1", "page": "0"},
                          {"x": "550", "y1": "300", "y2": "145",
                           "l": "1", "page": "0"}])
    z = zones.Zones(MAP_ID)
    z.add_set("下", ["1"])
    z.add_set("上", ["2"])
    z.add_edge("下", "上", "climb", ladder="L1", why="左边那根")
    z.add_edge("下", "上", "climb", ladder="L2", why="右边那根")
    check(zones.edge_between(z, "下", "上").get("ladder") == "L1",
          "前提不成立：没有位置时该按文件顺序取先加的那条")

    # ① 挑最近
    check(route_mod.pick_edge(t, z, "下", "上", at=(560, 300)).get("ladder") == "L2",
          "人站在右边，却没挑右边那根绳：%s"
          % route_mod.pick_edge(t, z, "下", "上", at=(560, 300)))
    check(route_mod.pick_edge(t, z, "下", "上", at=(40, 300)).get("ladder") == "L1",
          "人站在左边，却没挑左边那根绳：%s"
          % route_mod.pick_edge(t, z, "下", "上", at=(40, 300)))
    # ③ 没位置 ⇒ 文件顺序（不猜）
    check(route_mod.pick_edge(t, z, "下", "上", at=None).get("ladder") == "L1",
          "没给位置时该退回文件顺序（稳定、可预期）")

    # ④ 端到端：任务用的是那根最近的绳，显示也是它
    plan = route_mod.plan_jobs(t, z, "下", "上", at=(560, 300))
    check(len(plan["jobs"]) == 1, "该造出 1 个任务：%s" % plan["jobs"])
    job = plan["jobs"][0]
    check(job.ladder_id == "L2" and abs(job.x - 550) < 1e-6,
          "造出来的任务不是那根最近的绳（ladder_id=%r x=%r）"
          % (job.ladder_id, job.x))
    check(plan["edges"] and plan["edges"][0].get("ladder") == "L2",
          "规划结果里没带上「选中的那条边」（显示要用它）：%s" % plan.get("edges"))
    check(zones.edge_text(z, "下", "上", edge=plan["edges"][0]) == "爬（绳梯） L2",
          "那行显示的不是选中的那条边（说的和做的不一样 ✗）：%r"
          % zones.edge_text(z, "下", "上", edge=plan["edges"][0]))


def t_hop_cost_and_pick_edge():
    """**同一段内按「距离」挑边**（2026-09-27 用户口径："路径仍按最少段数、距离只在同一段内比"）。

    代价函数 `decision.route.hop_cost`（**纯距离，不加动作惩罚** ✓）：
      · 走 / 跳 = 玩家 → **目标集合 foothold 端点**距离；
      · 爬     = 玩家 → **入绳点**（上爬下端 / 下爬上端）距离 + **绳长**；
      · 下跳   = 玩家 → **最近那条允许下跳的 foothold** 距离 + 它到**落点**的垂直距离。

    钉六件：
      ① **跨类型也一起比**：走要 460px、爬只要 10+20px ⇒ **爬赢**（老口径"走永远优先" ✗）；
      ② **同类多绳按"距离 + 绳长"**：近的那根**特别长** ⇒ 输给稍远但很短的那根 ✓；
      ③ **下跳把落差算进去**（单位价也钉一下数值 ✓）；
      ④ **没位置 ⇒ 退回老口径**（类型优先级：走 > 爬，不猜 ✓）；
      ⑤ **建不出任务的边不参选**（绳号在本图不存在 ⇒ 跳过它 ✓）；
      ⑥ ⚠ **集合序列仍是 `find_path` 的最少段数**（距离**只**在同一段内比 ✓）——
         直连 1 段那条永远赢，哪怕绕路的两段都更近 ✓。
    """
    from decision import route as route_mod

    # 下平台 y=300、上平台 y=100，两条 fh 的 x 都盖住 0..600（这样 x=45 和 x=550 的两根绳
    # 两头都能落在平台上 ⇒ `zones.climb_direction` 才判得出"向上" ✓）
    t = _terrain([_fh(1, 0, 300, 600, 300),
                  _fh(2, 0, 100, 600, 100)],
                 ladders=[{"x": "45", "y1": "300", "y2": "100", "l": "1", "page": "0"},
                          {"x": "550", "y1": "300", "y2": "100", "l": "1", "page": "0"}])
    z = zones.Zones(MAP_ID)
    z.add_set("下", ["1"])
    z.add_set("上", ["2"])
    z.add_edge("下", "上", "climb", ladder="L1", why="左边那根")
    z.add_edge("下", "上", "drop", why="直接跳下去")

    # ① **跨类型也一起比**：人在 (50,300) ⇒ 爬 5+200=205、下跳 0+200=200 ⇒ 该选**下跳**
    #    （老口径按类型优先级会选「爬」✗ —— 这就是新口径要改掉的那件事 ✓）
    got = route_mod.pick_edge(t, z, "下", "上", at=(50, 300))
    check(got.get("kind") == "drop",
          "下跳 200 < 爬绳 205，却没按距离选（老口径「爬优先于下跳」已经改了 ✗）：%r" % got)
    # ② 爬绳代价 = **到入绳点距离 + 绳长**（入绳点 = 下端 (45,300) ⇒ 5；绳长 |100-300|=200）
    c = route_mod.hop_cost(t, z, {"from": "下", "to": "上", "kind": "climb",
                                  "ladder": "L1"}, at=(50, 300))
    check(c is not None and abs(c - 205.0) < 1e-6,
          "爬绳代价该是「到入绳点 5 + 绳长 200 = 205」：%r" % c)
    # ③ 下跳代价 = **到最近那条可下跳 foothold 的距离 + 落差**（0 + |100-300| = 200）
    c = route_mod.hop_cost(t, z, {"from": "下", "to": "上", "kind": "drop"},
                           at=(50, 300))
    check(c is not None and abs(c - 200.0) < 1e-6,
          "下跳代价该是「距离 0 + 落差 200 = 200」：%r" % c)
    # ④ 没位置 ⇒ **老口径**：类型优先级（爬 > 下跳）→ 文件顺序
    got = route_mod.pick_edge(t, z, "下", "上", at=None)
    check(got.get("kind") == "climb", "没给位置时该退回类型优先级（爬优先于下跳）：%r" % got)
    # ⑤ 建不出来的边不参选：绳号在本图不存在 ⇒ 那条 climb 跳过
    zz = zones.Zones(MAP_ID)
    zz.add_set("下", ["1"])
    zz.add_set("上", ["2"])
    zz.add_edge("下", "上", "walk", why="走得通")
    zz.add_edge("下", "上", "climb", ladder="L9", why="绳号不存在")
    got = route_mod.pick_edge(t, zz, "下", "上", at=(4, 300))
    check(got is not None and got.get("ladder") in (None, ""),
          "绳号不存在的 climb 还参选（会挑出一条根本建不出任务的边 ✗）：%r" % got)
    # ⑥ 距离**不改**集合序列：直连 1 段永远赢（`find_path` = 无权 BFS ✓）
    t2 = _terrain([_fh(1, 0, 0, 50, 0), _fh(2, 400, 0, 450, 0),
                   _fh(3, 800, 0, 850, 0)])
    z2 = zones.Zones(MAP_ID)
    z2.add_set("甲", ["1"])
    z2.add_set("乙", ["2"])
    z2.add_set("丙", ["3"])
    z2.add_edge("甲", "丙", "walk", why="直连（很远）")
    z2.add_edge("甲", "乙", "walk", why="近")
    z2.add_edge("乙", "丙", "walk", why="近")
    plan = route_mod.plan_jobs(t2, z2, "甲", "丙", at=(0, 0))
    check(plan["path"] == ["甲", "丙"] and len(plan["jobs"]) == 1,
          "集合序列该还是**最少段数**（距离只在同一段内比）：%r / %r"
          % (plan["path"], plan["jobs"]))

    # ② ⭐ 「走」**真的更近**时它要赢（跨类型也一起比距离 ✓，不是"走永远优先"✗）。
    #    ⚠ **2026-09-28 换了场景**：walk 的横向现在**有下限**（"最小也是当前 foothold 集合的
    #      最边缘端点距离"✓）⇒ 原来那个"人站在集合中部（x=560）、横向算 0"的场景**不再成立** ✗
    #      （现在会算 40+200=240 ✗）。⇒ 改成**人站在集合端点**（下限 = 0 ✓）且目标面同 x：
    #      · walk  = 0（横向）+ 200（往上）                = **200** ✓
    #      · climb = 到入绳点 550 + 绳长 200               = 750 ✗
    #    ⇒ walk 赢 ✓（语义保持："真的更近就赢" ✓）。
    t7 = _terrain([_fh(1, 0, 300, 600, 300), _fh(2, 0, 100, 600, 100)],
                  ladders=[{"x": "550", "y1": "300", "y2": "100", "l": "1", "page": "0"}])
    z7 = zones.Zones(MAP_ID)
    z7.add_set("下", ["1"])
    z7.add_set("上", ["2"])
    z7.add_edge("下", "上", "climb", ladder="L1", why="右边那根（很远）")
    z7.add_edge("下", "上", "walk", why="同层走得过去")
    got = route_mod.pick_edge(t7, z7, "下", "上", at=(0, 300))   # x=0 = 「下」集合的左端点
    check(got.get("kind") == "walk",
          "走真的更近却没选它（跨类型要比距离，不是「走永远优先」✗）：%s" % got)

    # ⑦ ⭐ **垂直段也要算进代价**（用户 2026-09-28 **两次**口径，第二次改过 ✗）：
    #    · **`walk` 的垂直段不打折**（2026-09-28 第二次修，原话："**所有的垂直下落端都要
    #      打折**（**下攀爬不是垂直下落**）"✓ ⇒ 走路往下走一层**不是自然下落** ⇒ 按原值 ✓）；
    #    · **`jump` 往下**按 `FALL_DISCOUNT` 折算 ✓（跳下去就是下落 ✓）、往上原值 ✓；
    #    · **`drop` / `climb`** 各自的口径在下面几条 ✓；
    #    · **逐条 foothold 比"横向 + 垂直"的总代价** ⇒ "横向脚下但低 1000" 会**输给**
    #      "横向差 50 但同一高度" ✓（⛔ 不能"先按横向挑一条、再加它的高度" ✗）。
    t3 = _terrain([_fh(1, 0, 0, 600, 0),             # 我站的面：y=0
                   _fh(2, 100, 100, 200, 100),       # 乙：更低（y=100）
                   _fh(3, 100, -100, 200, -100)])    # 丙：更高（y=-100）
    z3 = zones.Zones(MAP_ID)
    z3.add_set("甲", ["1"])
    z3.add_set("乙", ["2"])
    z3.add_set("丙", ["3"])
    z3.add_edge("甲", "乙", "walk", why="往下走一格")
    z3.add_edge("甲", "丙", "walk", why="往上走一格")
    z3.add_edge("甲", "乙", "jump", why="跳下去（对照：jump 要打折 ✓）")
    FD = route_mod.FALL_DISCOUNT
    c_down = route_mod.hop_cost(t3, z3, {"from": "甲", "to": "乙", "kind": "walk"},
                                at=(0, 0))
    check(c_down is not None and abs(c_down - 200.0) < 1e-6,
          "**走（walk）往下**的垂直段**不该打折** —— 走路往下不是「自然下落」✗"
          "（该 100+100=200）：%r" % c_down)
    c_jump = route_mod.hop_cost(t3, z3, {"from": "甲", "to": "乙", "kind": "jump"},
                                at=(0, 0))
    check(c_jump is not None and abs(c_jump - (100 + 100 / FD)) < 1e-6,
          "**跳（jump）往下**该按 `FALL_DISCOUNT`(%s) 折算（跳下去就是下落 ✓）：%r"
          % (FD, c_jump))
    c_up = route_mod.hop_cost(t3, z3, {"from": "甲", "to": "丙", "kind": "walk"},
                              at=(0, 0))
    check(c_up is not None and abs(c_up - 200.0) < 1e-6,
          "**往上**的垂直段**不打折**（该 100+100=200）：%r" % c_up)

    # ⑦' ⭐ **`walk` 的横向有"下限"**（用户 2026-09-28 第二次修，原话："walk 横向我说过了，
    #    **最小也是当前 foothold 集合的最边缘端点距离**"✓）：多层平台的 x 大量重叠时，
    #    "我的 x 落在目标面的范围里" ⇒ 老算法把横向算成 **0** ✗ ⇒ "楼上楼下"的 walk 假便宜 ✗
    #    （实测现场：三楼 #91 到二楼因此选了 walk ✗，而 drop 明显更近 ✓）。
    t5 = _terrain([_fh(1, 1000, 1000, 1100, 1000),   # 甲（我站）：x 1000~1100（很宽）
                   _fh(2, 1000, 1050, 1100, 1050)])   # 乙：x 1050~1100、就在正下方
    z5 = zones.Zones(MAP_ID)
    z5.add_set("甲", ["1"])
    z5.add_set("乙", ["2"])
    z5.add_edge("甲", "乙", "walk", why="楼上楼下、x 重叠")
    # 玩家 x=1060：**同时**落在甲的 [1000,1100] 和乙的 [1050,1100] 里 ⇒ 老算法横向 = 0 ✗
    c5 = route_mod.hop_cost(t5, z5, {"from": "甲", "to": "乙", "kind": "walk"},
                            at=(1060, 1000))
    # ⇒ 横向下限 = 到甲的最近端点（1100）的距离 = 40 ✓ + 垂直 50（walk 不打折 ✓）
    check(c5 is not None and abs(c5 - 90.0) < 1e-6,
          "**walk 横向没有下限**（x 落在目标面范围里就算 0 ✗）⇒ 楼上楼下会假便宜"
          "（该 40+50=90）：%r" % c5)
    # 对照：**跳（jump）不加这个下限**（可以原地起跳 ✓）
    c5j = route_mod.hop_cost(t5, z5, {"from": "甲", "to": "乙", "kind": "jump"},
                             at=(1060, 1000))
    check(c5j is not None and abs(c5j - 50.0 / FD) < 1e-6,
          "**跳**不该被加上「源集合边缘」下限（跳可以原地起跳 ✓）：%r" % c5j)
    t4 = _terrain([_fh(1, 0, 0, 600, 0),
                   _fh(2, 0, 1000, 10, 1000),        # 横向上就在脚下，但低 1000
                   _fh(3, 50, 0, 60, 0)])            # 横向差 50，同一高度
    z4 = zones.Zones(MAP_ID)
    z4.add_set("甲", ["1"])
    z4.add_set("乙", ["2", "3"])
    z4.add_edge("甲", "乙", "walk", why="同一段内的两条落脚面")
    c4 = route_mod.hop_cost(t4, z4, {"from": "甲", "to": "乙", "kind": "walk"},
                            at=(0, 0))
    check(c4 is not None and abs(c4 - 50.0) < 1e-6,
          "该挑「横向 50 + 同高」那条（50），而不是「横向 0 但低 1000」（0+1000/%s=%.1f）"
          " —— 代价必须**逐条按总距离**比 ✗：%r" % (FD, 1000 / FD, c4))


def t_ladder_at_margin():
    """「在不在绳上」的 x 容差**不许比上绳执行器的对齐容差还紧**（用户 2026-09-27 报：
    "这里**上绳子了**，但是位置状态说**没上**，为什么"）。

    现场（`106010105`）：绳 `x=1189`（y −83..68）、人 `(1179, -17)` ⇒ **Δx 正好 = 10.0**。
    老 `LADDER_DX = 10` 判的是 `|dx| < 10`（**不含** ✗），而执行器自己的对齐容差
    （`route.ALIGN_TOL_PX` 默认 10「坐标对齐误差范围」）判的是 `<=`（**含** ✓）⇒ 同样 10px：
    执行器说"对齐好了 ✓"、位置状态说"**没在绳上** ✗" ⇒ 上绳那一支永远等不到"吸上绳"
    （`ladder_id` 恒 `None`）⇒ 来回重跳 / 报"还没上绳" ✗。

    钉四件：
      ① 真实数据：`(1179, -17)` ⇒ **找得到**那根 x=1189 的绳 ✓（回归用户现场）；
      ② x 容差**不比方 1 个世界像素还紧、也不比执行器对齐容差紧**：`|dx| = 10` 必须算在绳上 ✓；
      ③ 明显不在绳上（`|dx| > LADDER_DX`）⇒ `None`（别把整条街都算成绳上 ✗）；
      ④ y 也得在绳段 ± `LADDER_PAD` 之内；出到绳段下方很远 ⇒ `None` ✓（"掉下来了"要判得出来 ✓）。
    """
    from core import mapdata
    from decision import route

    t = mapdata.load("106010105")
    # ① 用户现场：Δx 正好 10
    L = t.ladder_at(1179.0, -17.0)
    check(L is not None and abs(L.x - 1189.0) < 0.5,
          "用户 2026-09-27 的现场又判成「没在绳上」了（绳 x=1189、人 x=1179 ✗）：%r" % (L,))
    # ② 容差口径：不许比执行器的对齐容差更紧（否则两者会互相打架 ✗）
    check(t.LADDER_DX >= route.ALIGN_TOL_PX,
          "绳梯 x 容差(%s)比上绳对齐容差(%s)还紧 ⇒ 会出现「执行器说对齐好了、位置状态说没在绳上」✗"
          % (t.LADDER_DX, route.ALIGN_TOL_PX))
    check(t.ladder_at(1189.0 + t.LADDER_DX - 1.0, -17.0) is not None,
          "刚好在容差里面却判不在绳上（边界口径 ✗）")
    # ③ 明显在外面 ⇒ None
    check(t.ladder_at(1189.0 + t.LADDER_DX + 5.0, -17.0) is None,
          "离绳 %.0f px 还判在绳上（容差太宽 ✗）" % (t.LADDER_DX + 5.0,))
    # ④ y 出绳段 ± PAD ⇒ None
    check(t.ladder_at(1189.0, float(max(L.y1, L.y2)) + t.LADDER_PAD + 5.0) is None,
          "人已经掉到绳段下方很远，却还判在绳上（「掉下来」该判得出来 ✓）")


def t_canvas_alpha_kept():
    """底图/叠加图的**透明区必须留住**（用户 2026-09-27："**小地图底图**、地形叠加图的背景
    应该透明吧？**你自己加的黑色**？"）。

    真相：**不是谁画的黑色** ✗ —— WZ 那张底图 PNG 有近一半像素本来就是**透明**的（小地图面板
    之外的圆角 ✓，四角 alpha=0 ✓），而读图时 `IMREAD_COLOR` 把 alpha 丢了 ⇒ 透明区变**纯黑** ✗
    ⇒ 编辑器底图、以及叠在实时画面上的「地形叠加图」都带一块黑底 ✗。

    钉四件：
      ① `Terrain.canvas` **仍然是 3 通道 BGR**（全仓的匹配/拼接都按它写 ✓，不许破坏 ✗）；
      ② 透明区从 **`Terrain.canvas_alpha`** 拿得到，且**四角是 0**（那正是用户看到的"黑" ✗）；
      ③ 生成的叠加图 **带 alpha**（4 通道 ✓）、**面板外是透明的**、**画出来的线是不透明的** ✓；
      ④ 底图 PNG 本来没有 alpha 的图 ⇒ `canvas_alpha is None`（别硬造一个全 255 的 ✗）。
    """
    import shutil
    import tempfile
    from pathlib import Path

    import cv2

    from core import mapdata
    from core.imgio import imread

    # ④ 没有 alpha 的底图（拿它对照）——手边可能一张都没有 ⇒ 跳过这一件而不是误报 ✗
    plain = None
    mid = None
    for q in sorted(mapdata.map_dir().glob("*.png")):
        if q.stem.endswith(("_overlay", "_zones")):
            continue
        tt = mapdata.load(q.stem, with_canvas=True)
        if tt is None or tt.canvas is None:
            continue
        if getattr(tt, "canvas_alpha", None) is None:
            plain = q.stem
        elif mid is None:
            mid = q.stem
    check(mid is not None, "手边没有带 alpha 的底图，这条测不了")

    t = mapdata.load(mid, with_canvas=True)
    # ① canvas 仍旧 3 通道（全仓按 BGR 写 ✓）
    check(t.canvas.shape[2] == 3,
          "底图读成了 %d 通道 —— 全仓的匹配/拼接都按 BGR 写，会被打崩 ✗" % t.canvas.shape[2])
    # ② alpha 单独拿得到、四角是透明的
    a = t.canvas_alpha
    check(a is not None and a.shape == t.canvas.shape[:2],
          "底图 alpha 没拿到 / 尺寸对不上：%r vs %r" % (getattr(a, "shape", None), t.canvas.shape))
    check(a[0, 0] == 0 and a[0, -1] == 0 and a[-1, -1] == 0,
          "底图四角本该是全透明的（**那正是用户看到的「黑」** ✗）：%r"
          % ((a[0, 0], a[0, -1], a[-1, -1]),))
    check((a < 255).any(), "整张底图都不透明 ⇒ 这条测不出东西（换张带透明区的图 ✓）")
    # ④ 没有 alpha 的底图 ⇒ 不许硬造
    if plain is not None:
        tp = mapdata.load(plain, with_canvas=True)
        check(getattr(tp, "canvas_alpha", None) is None,
              "底图本来没有透明区，却造了一份 alpha 出来（%s ✗）" % plain)

    # ③ 真生成一张叠加图：带 alpha、面板外透明、线不透明
    from tools.map_terrain_view import render

    tmp = Path(tempfile.mkdtemp(prefix="ov_alpha_"))
    try:
        out = tmp / ("%s_overlay.png" % mid)
        render(t, out)
        got = imread(str(out), cv2.IMREAD_UNCHANGED)
        check(got is not None and got.shape[2] == 4,
              "生成的叠加图没有 alpha 通道（贴到实时画面上就是一块黑底 ✗）：%r"
              % (getattr(got, "shape", None),))
        al = got[:, :, 3]
        check((al == 0).any(),
              "叠加图整张不透明（透明区被填黑了 ✗）—— 面板外那片本该透出实时画面 ✓")
        check(int(al.max()) == 255,
              "画出来的地形线也得是不透明的（否则叠图上什么都看不见 ✗）")
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


TESTS = (
    ("foothold_below 与 find_below 同一口径（且跳过墙）",
     t_foothold_below_same_rule_as_find_below),
    ("脚下 foothold 的 x 容差：站平台边上要认、超容差不认、不改原有结果",
     t_foothold_below_x_tolerance),
    ("「脚下」按 y 认最近的那条：容差内不许偏好「胸口以上」那块（用户报的位置状态判错）",
     t_foothold_below_nearest_y),
    ("「在不在绳上」的 x 容差不许比上绳对齐容差更紧（用户报「上绳子了却说没上」）",
     t_ladder_at_margin),
    ("底图/叠加图的透明区要留住（canvas 仍 BGR，alpha 单独给 canvas_alpha）",
     t_canvas_alpha_kept),
    ("接缝判据：严丝合缝=walk / 差一点=near / 差得多=不是同层", t_seam_kinds),
    ("真图上严丝合缝 70 处（§12.2 的实测数字）", t_flush_pairs_on_real_map),
    ("集合存读 / 改名级联 / 删集合连带删边 / 校验报警",
     t_zones_roundtrip_and_validate),
    ("可走建议（双向 + 理由）与「隔着墙不可走」", t_walk_suggestions_and_wall_guard),
    ("绳端差 45px 也要接上：攀爬建议（双向 + 绳 id + 没圈的一端）",
     t_rope_ends_and_climb_suggestions),
    ("通行方式清单：**传送点 2026-10-06 加回**（五种、都有执行器）；「待确认」仍不是通行方式、"
     "老数据残留要报警、传送点不指名门也要报警", t_removed_kinds),
    ("⭐⭐ 传送点：出口按 `tn` 解析（跨图 ⇒ None）/ **代价 = 当前→起点门 ＋ 出口门→目标**（手算对）/ "
     "择路里真参与比（走贵挑门、快到目标挑走）/ 跨图建任务要抛 / `PortalJob` 真跑（走到门 ⇒ "
     "点按 ↑ ⇒ 到出口 done）", t_portal_kind_and_cost),
    ("⭐⭐ 传送点的对齐与补按**与攀爬执行器同一套**（用户 2026-10-06）：对齐判据 = "
     "「坐标对齐误差范围」（`tol_px` ✓ 不再用门的框）；**位置状态没回执 ⇒ 补按 ↑**"
     "（过了「移动操作尝试间隔」才补 ✓）；不再自己判「按了多久还没到」（交给寻路超时 ✓）",
     t_portal_align_like_climb),
    ("边图上的路径：走得通给逐步路线，走不通说清边界", t_find_path_over_edges),
    ("通行方式的显示名：带了键就不再拼一遍；跳 / 下跳 分得开", t_kind_label_text),
    ("爬绳的上下：按**起点**在哪端判；终点不在绳端时只有配了「中途跳下」才放行"
     "（说不清一律 None）", t_climb_direction),
    ("⭐⭐⭐ 「绳只是**路过**这一层」也能爬（用户 2026-10-06）：方向看目标层贴哪端 / x 不含绳 x ⇒ "
     "**默认斜跳上绳**（跳多远从数据算）/ 绳不路过仍 None / 任务·代价·规划都通 / 执行器真跑", t_climb_over_layer),
    ("走路的方向类型：存得住/默认不写文件/乱填当场拦住+校验报出", t_walk_dir),
    ("边的身份含**绳号 / 门**：换一根绳、换一扇门都是另一条边（原来静默当重复）；"
     "同类型多条取先出现的", t_edge_identity_includes_ladder),
    ("择路：到目标集合有多条绳时挑**最近的**（没位置则退回类型优先级 → 文件顺序）",
     t_pick_nearest_rope),
    ("同一段内按**距离**挑边：跨类型也比 / 爬绳算「距离+绳长」/ 下跳算落差 / 没位置退回"
     "老口径 / 建不出的边不参选 / **集合序列仍是最少段数**", t_hop_cost_and_pick_edge),
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
