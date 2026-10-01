"""地图地形数据（寻路用）：读 `datasets/map/<id>.json` + 小地图底图。

数据由 WzProbe 的 `dump-terrain` 导出（见 `docs/寻路设计.md` §11）：

    datasets/map/<id>.json     footholds / portals / ladderRope / miniMap / info / life
    datasets/map/<id>.png      小地图**底图**（miniMap/canvas，1 像素 = mag 世界像素）
    datasets/map/_physics.json 全地图共用的物理参数（真实客户端那套）

**三套坐标**（详见 `docs/寻路设计.md` §3）：

    世界坐标 (X, Y)      foothold / portal / ladderRope 用的就是它（帧原点在地图原点，
                          向右 X+、向下 Y+）
    底图像素 (cx, cy)    cx = (X + centerX) / mag，cy = (Y + centerY) / mag
                        **必须用 mag，不能用 16** —— 实测 105090700 的 mag=4，
                        MapleNecrocer 硬编码 /16，对这张图会偏 4 倍。
    屏幕坐标 (sx, sy)    小地图**面板**里的位置 —— 面板是底图的缩放/裁剪（S3 验证）

这一层只做**几何与查询**，不碰图像匹配、不发按键。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def map_dir() -> Path:
    return ROOT / "datasets" / "map"


def physics():
    """全地图共用的物理参数（真实客户端那套）；没有返回 {}。"""
    p = map_dir() / "_physics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

def _i(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class Foothold:
    """一段直的平台（或墙）。x1==x2 就是墙（竖直，不可站立）。"""

    __slots__ = ("layer", "group", "fid", "x1", "y1", "x2", "y2",
                 "nxt", "prv", "forbid_down", "force_down")

    def __init__(self, d):
        self.layer = _i(d.get("layer"))
        self.group = _i(d.get("group"))
        self.fid = _i(d.get("id"))
        self.x1, self.y1 = _i(d.get("x1")), _i(d.get("y1"))
        self.x2, self.y2 = _i(d.get("x2")), _i(d.get("y2"))
        self.nxt, self.prv = _i(d.get("next")), _i(d.get("prev"))
        self.forbid_down = _i(d.get("forbidDown"))      # 1 = 不能从上面穿下去
        self.force_down = _i(d.get("forceDown"))        # 1 = 强制下落

    @property
    def is_wall(self):
        return self.x1 == self.x2

    @property
    def left(self):
        return min(self.x1, self.x2)

    @property
    def right(self):
        return max(self.x1, self.x2)

    def y_at(self, x):
        """这段在 x 处的 y（直线插值）—— 和 MapleNecrocer 的 FindBelow 同一套口径。"""
        if self.x1 == self.x2:
            return float(min(self.y1, self.y2))
        t = (x - self.x1) / float(self.x2 - self.x1)
        return self.y1 + t * (self.y2 - self.y1)

    def __repr__(self):
        return "Foothold#%d((%d,%d)-(%d,%d) next=%d)" % (
            self.fid, self.x1, self.y1, self.x2, self.y2, self.nxt)


class Segment:
    """一条由 prev/next 串起来的平台（可能是折线：上坡/台阶）。"""

    #: `index` = 它在 `Terrain.segments` 里的序号（由 `_chain` 填）。
    #: **为什么要它**：段本身没有 id，而"我在哪块平台上"必须能报给人和日志 ——
    #: 序号在同一次导出的地形数据里是稳定的（重新导出地形后可能变，那时也该重看）。
    __slots__ = ("footholds", "group", "index")

    def __init__(self):
        self.footholds = []
        self.group = None
        self.index = -1

    @property
    def points(self):
        """串成折线的点列（首尾相接，去掉重复点）。"""
        pts = []
        for f in self.footholds:
            a, b = (f.x1, f.y1), (f.x2, f.y2)
            if pts and pts[-1] == a:
                pts.append(b)
            elif pts and pts[-1] == b:
                pts.append(a)
            else:
                pts.extend([a, b])
        # 去掉连续重复
        out = []
        for p in pts:
            if not out or out[-1] != p:
                out.append(p)
        return out

    @property
    def left(self):
        xs = [p[0] for p in self.points]
        return min(xs) if xs else 0

    @property
    def right(self):
        xs = [p[0] for p in self.points]
        return max(xs) if xs else 0

    def __len__(self):
        return len(self.footholds)

    def __repr__(self):
        return "Segment(%d 段, x[%d..%d])" % (len(self.footholds),
                                              self.left, self.right)


class Portal:
    __slots__ = ("pn", "pt", "x", "y", "tm", "tn")

    def __init__(self, d):
        self.pn = str(d.get("pn") or "")
        self.pt = _i(d.get("pt"))            # 0=起点(出生点) 1=普通 2/7=小地图上画的
        self.x, self.y = _i(d.get("x")), _i(d.get("y"))
        self.tm = _i(d.get("tm"))            # 目标地图（999999999 = 本图内）
        self.tn = str(d.get("tn") or "")     # 目标 portal 名

    def __repr__(self):
        return "Portal(%s pt=%d (%d,%d) -> %s/%s)" % (
            self.pn, self.pt, self.x, self.y, self.tm, self.tn or "-")


class Ladder:
    """绳梯 / 绳子：竖直的一条，x 固定，y1..y2 是范围。"""

    __slots__ = ("x", "y1", "y2", "l", "uf", "page")

    def __init__(self, d):
        self.x = _i(d.get("x"))
        self.y1, self.y2 = _i(d.get("y1")), _i(d.get("y2"))
        self.l = _i(d.get("l"))              # 1 = 绳子（可穿越），0 = 梯子
        self.uf = _i(d.get("uf"))            # 1 = 从上面才能用
        self.page = _i(d.get("page"))

    @property
    def top(self):
        return (self.x, min(self.y1, self.y2))

    @property
    def bottom(self):
        return (self.x, max(self.y1, self.y2))

    def __repr__(self):
        return "Ladder(x=%d y[%d..%d] %s uf=%d)" % (
            self.x, self.y1, self.y2, "绳子" if self.l else "梯子", self.uf)


class Terrain:
    """一张图的地形：所有 foothold 段 + 绳梯 + 传送点 + 小地图换算。"""

    #: 传送点/绳梯的判定容差（口径抄 MapleNecrocer 的两个 Find：
    #  portal |dx|<15 且 |dy|<12；ladder |dx|<10 且 y 在 [y1-12, y2+12]）
    PORTAL_DX, PORTAL_DY = 15, 12
    #: ⚠ 绳梯的 **x 容差从 10 放宽到 24**（2026-09-27 用户报"这里上绳子了，但是位置状态说没上"）：
    #  原值 10 是抄 MapleNecrocer 的 `|dx| < 10`，但它跟**上绳执行器自己的对齐容差**
    #  （「坐标对齐误差范围」= `decision.route.ALIGN_TOL_PX` 默认 **10**）**卡在同一个数上、
    #  边界却相反** ✗ —— 对齐判的是 `|dx| <= 10`（**含** ✓），这里判的是 `|dx| < 10`（**不含** ✗）。
    #  实测用户那张图（`106010105`：绳 `x=1189`、人 `(1179, -17)` ⇒ **Δx 正好 = 10.0**）：
    #  执行器认为"对齐好了 ✓"、位置状态却报"**没在绳上** ✗" ⇒ 上绳那一支永远等不到"吸上绳"
    #  （`ladder_id` 一直 `None`）⇒ 来回重跳 / 报"还没上绳" ✗。
    #  24 = 那个默认容差(10) + 绳子的视觉半宽与读数误差（≈14）⇒ **保证不比对齐容差更紧** ✓
    #  ⚠ 放宽它是安全的：真正"算不算在绳上"还有一道**按键许可**
    #    （`agent.holding_vertical()`：没按 ↑/↓ 一律不算 ✓，见 `gui/live_thread._fill_route_ctx`）
    #    ⇒ 走路路过绳口不会被粘住 ✓（那才是当初把容差收紧的顾虑 ✓）。
    LADDER_DX, LADDER_PAD = 24, 12

    def __init__(self, map_id, data, canvas=None, canvas_alpha=None):
        self.id = str(map_id)
        self.info = dict(data.get("info") or {})
        self.mini = dict(data.get("miniMap") or {})
        #: 小地图底图（**numpy BGR，永远是 3 通道** ✓ —— 全仓的匹配/拼接都按它写 ✓）。
        self.canvas = canvas
        #: ⭐ 底图自己的 **alpha**（`uint8` 单通道；底图 PNG 没有透明区 = `None` ✓）——
        #: 2026-09-27 加：底图 PNG 有近一半像素是全透明的（面板之外的圆角 ✓），
        #: 而 `cv2.imread(IMREAD_COLOR)` 会把它们读成**纯黑** ✗ ⇒ 编辑器底图与「地形叠加图」
        #: 都带一块黑底 ✗（用户："小地图底图、地形叠加图的背景应该透明吧？**你自己加的黑色**？"）。
        #: ⇒ 现在**分开存**：`canvas` 仍旧 3 通道（不动任何现有代码 ✓），透明区要用的人自己
        #: 拿这份 alpha 合回去（`tools/map_terrain_view.render` / `gui/zone_editor._add_background` /
        #: `gui/minimap_calib` 的底图参照 ✓）。
        self.canvas_alpha = canvas_alpha

        self.footholds = [Foothold(d) for d in (data.get("footholds") or [])]
        self.portals = [Portal(d) for d in (data.get("portals") or [])]
        self.ladders = [Ladder(d) for d in (data.get("ladderRope") or [])]
        self.life = list(data.get("life") or [])
        self._by_id = {}
        for f in self.footholds:
            self._by_id.setdefault(f.fid, f)
        self.segments = self._chain()

    # ---------------- 串段 ----------------

    def _chain(self):
        """把 foothold 按 prev/next 串成 Segment。

        next/prev 是**平台链表**（不是空间索引），墙（x1==x2）也在这条链里 ——
        所以一条"完整的平台"通常两端各带一堵墙。

        **先沿 prev 退到链头再往前走**：文件顺序不保证从链头开始，
        只沿 next 走的话，同一条平台会被从中间切成两段（实测踩过）。
        """
        seen = set()
        segs = []

        def walk(start, step):
            out, cur, guard = [], start, 0
            while cur is not None and cur.fid not in seen and guard < 100000:
                seen.add(cur.fid)
                out.append(cur)
                guard += 1
                cur = self._by_id.get(getattr(cur, step))
            return out

        for f in self.footholds:
            if f.fid in seen:
                continue
            head = f
            guard = 0
            while True:                     # 退到链头（prev 一路回溯）
                p = self._by_id.get(head.prv)
                if p is None or p.fid in seen or p.fid == head.fid:
                    break
                head = p
                guard += 1
                if guard > 100000:
                    break
            seg = Segment()
            seg.footholds = walk(head, "nxt")
            if seg.footholds:
                seg.group = seg.footholds[0].group
                seg.index = len(segs)      # 「我在第几段」靠它说清楚（见 __slots__）
                segs.append(seg)
        return segs

    # ---------------- 查询（Agent 用这几个） ----------------

    def foothold_below(self, x, y, tol=40, xtol=0, above_tol=None, band=None):
        """脚下那条**可站立**的 foothold（跳过墙）→ Foothold 或 None。

        **与 `find_below` 同一套口径**（同一容差、同样跳过竖向的墙、同样取最近那条），
        区别只是**顺便告诉你是哪一条**。这是"我在不在某个 foothold 集合里"的唯一判据
        （`core/zones.py`）：先问脚下是哪条 foothold，再看它属于哪个集合。

        为什么不复用"段"：自动串段会把**要跳/攀才能互通**的 foothold 并成同一段
        （实测 105090600 的第 0 段把一面 388 像素高的悬崖当成了平台边缘 ✗）——
        集合改由人工分组，判定只认 foothold，见 docs/寻路设计.md §12。

        `xtol` = **x 方向容差**（0 = 老行为；调用方传设置里的「坐标对齐误差范围」✓）：
        人站在平台**边上**时，小地图黄点（画的是玩家中心）会读到边界外几个像素 ——
        实测 105090600 的 (650, 283)：人明明站在 id=41 那块平台上（面 y=280，差 3px ✓），
        可 x=650 在它左边界 **657 的外面 7px** ✗ ⇒ 老口径一个字都不认 ⇒
        `foothold_id=None` ⇒「脚下没有平台 / 未分组 / 起点不知道」⇒ 起点与到达判定①全瘫 ✗
        （用户 2026-09-26 报的正是这个坐标）。y 有 40px 容差、x 却**一点都没有**，
        这个不对称本来就没道理。

        ⚠ **`xtol` 怎么用（2026-09-28 改过一次，别走回去 ✗）**：
          · 先按 `dx=0` 比一遍（老口径 ✓）；
          · 有 `xtol` ⇒ 再按 `dx=xtol` 比一遍，**只有放宽后能找到更近的那条时才改判** ✓
            （严格 `<`）。
          ⛔ 原来写的是"**只在 dx=0 一条都找不到时**才按 `xtol` 找一遍 ⇒ 不可能改变任何
            原本判得出来的结果（风险为 0）"✗ —— 那条口径**保护不了真现场**：人站在
            「小平台」右边界**外 3px**、而 23px 之下的「底层」x 范围正好罩住他 ⇒ 老口径
            **判得出**（判成「底层」✗）⇒ 放宽那一遍**永远不会跑** ⇒ `foothold_id` 错 ⇒
            `here_sets` 错 ⇒ 归属集合 / 限制战斗区域 / 起点判定全错 ✗
            （用户 2026-09-28 报的 "(594,162) 应该属于小平台、却被解析成了底层" ✓）。
            ⇒ "风险为 0"让位给"判对" ✓；**老结果里本来就对的那些仍然一点没动** ✓
            （放宽只在"更近"时生效 ✓）。

        ⚠ **y 判据的取向**（用户 2026-09-27 连着两次报"位置状态判错了"）：**取 |Δy| 最小的那条**
          ✓（容差之内一律平等比较 —— "身下"不再是加分项 ✓）。
          · (577,193)：`#12 小平台` Δ=−38 ／ `#47 底层` Δ=−8 ⇒ 判 **#47「底层」** ✓；
          · (459,154)：更近的那条在中心**上方**几像素 ⇒ 就判它 ✓。
        ⭐ **`above_tol`**（2026-09-28 加，给**怪**用 ✓）：`None` ⇒ 沿用 `tol`（**老行为一字
          不变** ✓）；传了值 ⇒ 上面那句"头顶上方超过容差就跳过"改用**它** ✓。
          ⚠ 为什么必须有它（现场）：给怪算的是"**怪框底**换算的世界 y"（`gui/live_thread.py`
            的 `_resolve` ✓），而**大怪的检测框并不包住脚** —— 实测那只二楼石人，框底比它
            真正站的二楼面**高 84 像素** ✗ ⇒ 老口径（上方只给 40）**直接跳过那条面** ⇒
            只剩"下方最近"的**一楼** ⇒ 怪被判成在下一层 ✗ ⇒ 追击下"前往一楼"、
            到了又判"底层"⇒ **来回走 + 卡死** ✗
            （用户现场："**查#171 一个二楼的怪显示在底层，现在已经卡住了**"✓）。
          ⚠ 放宽**不会**把原本判对的弄错：比较始终是"**|Δy| 最小**"✓ ⇒ 怪真站在一楼
            （框底就是脚）时一楼 Δ≈0 照样赢 ✓。
        ⭐⭐ **`band=(lo, hi)`**（2026-09-28 再加，**给怪用的最终口径** ✓）：直接给一个**绝对
          y 区间**，面的 y 落进去才算候选 ✓；给了它，上面那条"上方容差"**就不看了** ✓。
          ⚠ 为什么"固定容差"这条路走不通（用户当天又报"**又把顶层怪判成底层怪了**"✓）：
            **层距会到 540** —— 实测图 105040303：顶层 375 / 三楼 915 / 二楼 1455 /
            底层 1995 ⇒ 相邻层就差 **540** ✗ ⇒ `above_tol=200` 那种固定值**连半层都盖不住**
            ⇒ 照样跨层 ✗。
          ⇒ 口径改成"用**怪框自己的高度**当区间"（调用方 `gui/live_thread.py` 给 ✓）：
            区间 = `[框底世界 y − 框高, 框底世界 y + 框高]` ✓ ——
            · "框底 ≠ 脚底"**两个方向都会偏**（框没包住脚 ⇒ 面在框底**上方**；框包过头 ⇒
              面在**下方** ✓）⇒ 所以要**对称** ✓；
            · 身体高是**检测框直接给的** ⇒ **不新造魔数** ✓，而且**自适应**：小怪 `h≈40`
              ⇒ ±40（比老的 ±200 **更严** ✓）；大怪 `h≈300` ⇒ ±300（盖得住 540 ✓）。
          ⚠ 区间里一条面都没有 ⇒ 调用方会**退回**"只看 `above_tol`"那一版（不瞎判 ✓）。
          两版历史（都别再走回去 ✗）：① 最早"容差内一律取 fy 最小"= 偏好**玩家胸口以上**那块 ✗；
          ② 我中途加的"优先认身下那条"✗ —— 被第二次现场推翻 ✓。
          （"我在哪块平台上"一错，归属集合 / 限制战斗区域 / 起点判定全跟着错 ✗。）
        """
        def _pick(dx):
            best_d, best_f = None, None
            for f in self.footholds:
                if f.is_wall or not (f.left - dx <= x <= f.right + dx):
                    continue
                # x 在区间外时用**最近的端点**求高度：`y_at` 是按直线外推的，
                # 斜平台会一路跑偏（那样 y 判据就会把整条误杀掉 ✗）。
                fx = min(max(x, f.left), f.right)
                fy = f.y_at(fx)
                # ⭐ 面比给的 y 高多少算"还在考虑范围内"：默认 `tol`（40 ✓ **老行为一字不变** ✓）；
                #   传了 `above_tol` 就用它（见下面那段与 `above_tol` 的说明 ✓）。
                # ⭐⭐ `band`（2026-09-28 再加，**给怪用的最终口径** ✓）：直接给一个**绝对
                #   y 区间** `(lo, hi)`，面的 y 落进去才算候选 ✓ —— 见下面 `band` 的说明
                #   （为什么"固定容差"这条路走不通 ✓）。
                if band is not None:
                    if not (float(band[0]) <= fy <= float(band[1])):
                        continue
                elif fy < y - (tol if above_tol is None else float(above_tol)):
                    continue
                # **离玩家最近的那条**（|Δy| 最小 ✓，容差内一律平等比较）——
                # 用户 2026-09-27 连着两次报的都是这条口径：
                #   · (577,193)：`#12 小平台` Δ=−38 ／ `#47 底层` Δ=−8 ⇒ 该判 **#47** ✓；
                #   · (459,154)：`#12` 更近 ／ 某条「底层」更远 ⇒ 该判 **#12** ✓。
                # ⚠ 我中途自己加过一版"**优先认身下那条**"（fy ≥ y 优先）✗ —— 被第二次现场
                #   推翻 ✗（那次更近的那条恰好在中心**上方**几像素）⇒ **已去掉**，别再自作聪明 ✗。
                _d = abs(fy - float(y))
                if best_d is None or _d < best_d:
                    best_d, best_f = _d, f
            return best_f, best_d

        # ⭐ **`xtol` 内的候选也参与比较**（2026-09-28 修，用户报 "(594,162) 应该属于小平台、
        #   却被解析成了底层"✗）。实测（图 106010105，真数据）：
        #     · `_pick(0)`  ⇒ **fh=46「底层」**（x 范围 585~675 把它罩住了）Δy=**+23** ✗
        #     · `_pick(10)` ⇒ **fh=13「小平台」**（面 y=155，x 范围 579~591）Δy=**−7** ✓
        #   人的世界 x=594 只在「小平台」右边界 **外 3px**（定位的抖动就这个量级 ⇒ 这正是
        #   `xtol` 存在的理由 ✓），而原来那句是"**只在 dx=0 一条都找不到时**才放宽"✗
        #   ⇒ **小平台压根没参与比较** ⇒ 判给了 23px 之下的「底层」✗ —— 一笔错到底：
        #   `foothold_id` 错 ⇒ `here_sets` 错 ⇒ 归属集合 / 限制战斗区域 / 起点判定全跟着错 ✗。
        # ⇒ 现在：**先按老口径比一遍（dx=0），再按放宽口径比一遍（dx=xtol）**，
        #   **只有在放宽后能找到"更近"的那条时才改判** ✓（严格 `<`，不是 `<=`）。
        #   ⚠ 这样**只多认"原来认不出或明显更差"的情形**，不动"原本就判得对"的结果 ✓：
        #     · 人真站在 `46` 上（y≈185）：13 的 Δ=30 > 46 的 Δ=0 ⇒ **照样判 46** ✓；
        #     · 人站在「小平台」上（y≈155）：dx=0 时 594 落空 ⇒ 老口径会判 46（Δ=30）✗，
        #       放宽后 13（Δ≈0）更近 ⇒ 判 13 ✓（这就是本条的修法 ✓）。
        #   ⚠ 与上面 `xtol` 那段注释里"**不可能改变**任何原本判得出来的结果"（2026-09-26 口径）
        #     有冲突 ✗ —— 用户 2026-09-28 看到的正是"原本判出来了、但判错了"，所以那条
        #     "风险为 0"的口径**让位给"判对"** ✓（`t_foothold_below_prefers_nearer_within_xtol`
        #     两边都钉着：`xtol=0` 的老结果一字不变 ✓、`xtol>0` 时更近的赢 ✓）。
        got, bd = _pick(0.0)
        if xtol > 0:
            got2, d2 = _pick(float(xtol))     # 放宽那一次：候选更多，比法照旧（|Δy| 最小 ✓）
            if got2 is not None and (got is None or d2 < bd):
                got, bd = got2, d2
        return got

    def find_below(self, x, y, tol=40, xtol=0):
        """脚下最近的可站立平台 → (x, y_on_line) 或 None。

        口径抄 MapleNecrocer 的 `Footholds.FindBelow`：在 x 覆盖范围内的**非墙**
        foothold 里，取 y 不小于给定 y（允许 tol 容差）中**最小**的那个，再按直线插值。
        ⚠ 2026-09-27 起 y 判据更细一层（用户报"位置状态判错"）：**先认"在身下"的**
          （`fy >= y` 取最小 fy ✓）；**一条身下的都没有**时才退到"容差内比中心高"的那几条、
          取**离玩家最近的**（fy 最大 ✓）。理由与实测见 `foothold_below` 的说明 ✓ ——
          两边**仍是同一套口径**（这里只是转调它 ✓）。

        **tol 默认 40 而不是 2**：小地图上的黄点画的是玩家**中心**
        （`MiniMap.cs` 用的是 `Game.Player.X/Y`），而地面在脚底 —— 实测 105090700
        的出生点 sp 离脚下地面 **8 世界像素**（玩家越高/站着不动时会有小幅波动）。
        所以「我站在哪块平台上」要按"下方几十像素内的最近地面"来判，不能用 2px。

        实现在 `foothold_below`（**一份口径，别各写一份** —— 两边容差一旦不同，
        就会出现"工具说在这条上、界面说在那条上"这种最难查的分歧）。
        """
        f = self.foothold_below(x, y, tol, xtol)
        return None if f is None else (x, f.y_at(x))

    def segment_of(self, x, y, tol=40, xtol=0):
        """点 (x, y) 落在哪条段上（判定"我现在站在哪块平台"）→ Segment 或 None。

        **按 foothold_below 的口径实现**：先问脚下是哪条 foothold，再反查它在哪条段里。
        （原先靠"y 差 ≤0.51"反查，是为没有 foothold 返回值时打的补丁；现在直接认对象。）
        注意：**语义判定一律走集合**（`core/zones.py`），段只留作显示/辅助选择。
        """
        f = self.foothold_below(x, y, tol, xtol)
        if f is None:
            return None
        for seg in self.segments:
            if f in seg.footholds:      # Foothold 没有 __eq__ ⇒ 按身份比较（同一批对象）
                return seg
        return None

    def ladder_at(self, x, y):
        """(x, y) 附近有没有绳梯/绳子 → Ladder 或 None（口径同 MapleNecrocer）。"""
        for L in self.ladders:
            if abs(x - L.x) < self.LADDER_DX and \
               (min(L.y1, L.y2) - self.LADDER_PAD <= y
                    <= max(L.y1, L.y2) + self.LADDER_PAD):
                return L
        return None

    def portal_at(self, x, y):
        """(x, y) 附近有没有传送点 → Portal 或 None（口径同 MapleNecrocer）。"""
        for p in self.portals:
            if abs(x - p.x) < self.PORTAL_DX and abs(y - p.y) < self.PORTAL_DY:
                return p
        return None

    @property
    def spawn(self):
        """出生点（pt == 0 的 portal，取第一个）。"""
        for p in self.portals:
            if p.pt == 0:
                return p
        return self.portals[0] if self.portals else None

    # ---------------- 小地图换算 ----------------

    @property
    def mag(self):
        """WZ 里的 `miniMap/mag`（**不是**换算尺度，见 `px_per_world`）。"""
        return float(self.mini.get("mag") or 0)

    @property
    def world_span(self):
        """小地图覆盖的**世界跨度** (宽, 高)。

        ⚠ 实测：`miniMap/width,height` 不是底图像素尺寸，而是**世界跨度**
        （105090700：width=2630，而底图 PNG 只有 164px 宽）。
        底图是把整个世界跨度画进了 width/底图宽 ≈ 16 个世界像素/底图像素的图里。
        """
        w, h = self.mini.get("width"), self.mini.get("height")
        if w is None or h is None:
            return None
        return (_i(w), _i(h))

    @property
    def canvas_size(self):
        """底图的**实际像素**尺寸（读 PNG 得到）；没读图时按 16 反推。"""
        if self.canvas is not None:
            h, w = self.canvas.shape[:2]
            return (w, h)
        span = self.world_span
        if not span:
            return None
        return (max(1, int(span[0] / 16)), max(1, int(span[1] / 16)))

    @property
    def px_per_world(self):
        """底图 1 像素 = 多少世界像素（= 世界跨度 / 底图宽）。

        **不要用 mag**：实测 105090700 的 mag=4，而真正能对上的是 ~16
        （MapleNecrocer 硬编码 16 恰好是对的）。用 width/底图宽 反推最稳 ——
        它由数据本身决定，换图也不会错。
        """
        size, span = self.canvas_size, self.world_span
        if not size or not span or size[0] <= 0:
            return 16.0
        return float(span[0]) / float(size[0])

    def world_to_canvas(self, x, y):
        """世界坐标 → 底图像素。"""
        cx = float(self.mini.get("centerX") or 0)
        cy = float(self.mini.get("centerY") or 0)
        k = self.px_per_world
        return ((x + cx) / k, (y + cy) / k)

    def canvas_to_world(self, cx, cy):
        k = self.px_per_world
        return (cx * k - float(self.mini.get("centerX") or 0),
                cy * k - float(self.mini.get("centerY") or 0))

    @property
    def bounds(self):
        """世界坐标范围 (xmin, ymin, xmax, ymax)。

        由 width/height（世界跨度）与 center 推出：北西角 = (−centerX, −centerY)。
        """
        span = self.world_span
        if not span:
            return None
        cx = float(self.mini.get("centerX") or 0)
        cy = float(self.mini.get("centerY") or 0)
        return (-cx, -cy, span[0] - cx, span[1] - cy)

    @property
    def minimap_hidden(self):
        return str(self.info.get("hideMinimap", "0")) not in ("0", "")


# ══════════════════════════════════════════════════════════════
# 载入
# ══════════════════════════════════════════════════════════════

def load(map_id, with_canvas=False):
    """读一张图的地形；文件不存在返回 None。

    with_canvas=True 时顺便把小地图底图读进 `Terrain.canvas`（**numpy BGR；底图 PNG 带 alpha
    时是 4 通道 BGRA** ✓ —— 见下面 `IMREAD_UNCHANGED` 那段的说明）。
    """
    d = map_dir()
    try:
        data = json.loads((d / ("%s.json" % map_id)).read_text(encoding="utf-8"))
    except Exception:
        return None

    canvas = None
    canvas_alpha = None
    if with_canvas:
        name = (data.get("miniMap") or {}).get("canvas")
        if name:
            import cv2
            import numpy as np
            # ⭐ **alpha 必须留住**（2026-09-27 用户："小地图底图、**地形叠加图**的背景应该透明
            #   吧？**你自己加的黑色**？"）：WZ 那张底图 PNG 有**近一半像素是全透明的**（小地图
            #   面板之外的圆角区域，四角 alpha=0 ✓，实测 106010105：透明 49.2% / 不透明 46.5%），
            #   而 `IMREAD_COLOR` 会把它们**读成纯黑** ✗ ⇒ 编辑器底图、以及叠在实时画面上的
            #   「地形叠加图」都会带一块黑底 ✗。
            #   ⚠ 所以"黑底"**不是谁画上去的**，是**读图时丢掉的 alpha** ✗（别去绘制代码里找 ✗）。
            # ⇒ 读**原样**：带 alpha 就是 4 通道 BGRA ✓。按 `shape[2] == 4` 判、各走各的：
            #   `tools/map_terrain_view.render`（画线在 BGR 上、alpha 合并回去 ✓）、
            #   `gui/minimap_calib.np_to_pixmap`（BGRA ⇒ Format_ARGB32 ✓）、
            #   `gui/zone_editor._add_background`（同上 ✓）。
            img = cv2.imread(str(d / name), cv2.IMREAD_UNCHANGED)
            if img is not None and getattr(img, "ndim", 0) == 3 and img.shape[2] == 4:
                # 带 alpha ⇒ **分开存**：`canvas` 保持 3 通道（全仓都按 BGR 写 ✓），
                # alpha 单独给 `canvas_alpha` ✓（谁要透明谁自己合 ✓）
                canvas = np.ascontiguousarray(img[:, :, :3])
                canvas_alpha = np.ascontiguousarray(img[:, :, 3])
            else:
                canvas = img
    return Terrain(map_id, data, canvas, canvas_alpha)


def calib_path(map_id):
    """小地图标定文件：`datasets/map/<id>.mapcalib.json`（每张图一份）。

    存的是「面板 → 底图 → 世界」的换算参数（见 `perception/minimap.py`）：
        {"mode": "fit" | "crop", "scale": …, "scale_y": …（可选，缺省 = 同 scale）,
         "offset": [x, y], "view": [x, y], "score": …, "src": …, "note": "…"}
    `scale_y` = y 轴自己的缩放（2026-09-27「双点标定」加的）：读两轴一律走
    `perception.minimap.scales_of()`，别各处自己 `cal["scale"]` —— 那是"换算用 x、
    画图用 y"这种最难查的分歧的来源。
    **为什么每张图一份**：不同的图客户端可能用不同显示方式（装得下就整张缩放、
    装不下就 1:1 裁剪滚动），而且缩放/偏移也各不相同。

    **还要按「小地图来源」分开存**（实测）：同一张图，独立推流那条面板
    753×612、从实时画面那条 134×109 —— 差 5.6 倍。一份标定只对一条来源成立，
    共用就等于「拿 A 量出来的几何去算 B」：算出来的位置整体错，而错的位置会被
    当成「我在哪块平台上」，比没标定糟得多。

    文件形状（v2）：
        {"v": 2, "sources": {"stream": {...}, "live": {...}}}
    每条来源里的字段和老格式**完全一样**，所以消费单个标定的代码
    （locate / panel_to_canvas / has_geometry）一行都不用改。
    老格式（整份平铺、没记来源）仍读得出来，但会标上 `legacy`：界面据此提醒
    「这份是换来源之前量的，请重量一次」。
    """
    return map_dir() / ("%s.mapcalib.json" % map_id)


#: 标定文件格式版本（1 = 老格式：整份平铺；2 = 按来源分）
CALIB_V2 = 2


def _read_calib_file(map_id):
    import json as _json
    try:
        d = _json.loads(calib_path(map_id).read_text(encoding="utf-8"))
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def load_calibs(map_id):
    """→ {来源: 标定 dict}。老格式（整份平铺）归到 `""` 这个键下。"""
    d = _read_calib_file(map_id)
    if not d:
        return {}
    srcs = d.get("sources")
    if isinstance(srcs, dict):
        return {k: dict(v) for k, v in srcs.items() if isinstance(v, dict)}
    if d.get("mode") or d.get("scale"):
        return {"": d}
    return {}


def zoom_of(calib, default=1):
    """这份标定是在 A 机 zoom 为多少的时候标出来的（**没记** ⇒ 当 `default`）。

    ⭐ 为什么标定要记 zoom（2026-10-01 修的那个坑）：A 机推来的一帧是
      「小地图面板 × zoom」（`tools/minimap_push`），标定里的 `scale` / `offset`
      是**对着那个像素尺寸**量出来的。zoom 一变、帧的尺寸就整倍数变 ⇒ 旧标定照套
      会让 `perception.minimap.panel_to_world` 出来的坐标**整倍数错**，
      而**全程没有任何报错**（`gui/route_panel._calib_panel_wh` 里那个"读了没写"的
      `panel` 字段正是当年没堵住的口子）。记下来，读的时候才可能**拒不发货** ✓。
    """
    try:
        z = int((calib or {}).get("zoom"))
    except (TypeError, ValueError):
        return int(default)
    return z if z >= 1 else int(default)


def _zoom_ok(calib, zoom):
    """`zoom` 给了 ⇒ 与标定里记的那个**相符**才肯用。没给 ⇒ 不管（保持老行为 ✓）。"""
    if zoom is None:
        return True
    try:
        return int(zoom) == zoom_of(calib)
    except (TypeError, ValueError):
        return False


def load_calib(map_id, src=None, zoom=None):
    """→ 某条来源的标定（形状同老格式）；没有 / **zoom 不相符** → None。

    src=None：老格式直接给；新格式**只有一份**时给那一份，多份则返回 None ——
    「哪一份」必须由调用方说清，猜错就是拿另一条来源的几何去算世界坐标。

    ⭐ `zoom`（**可选**，不传 = 与从前完全一样）：标定是**按某个 zoom 标出来的**
      （A 机推流时的放大倍数）。传了 zoom 而文件里那份记的不是这个 zoom ⇒
      **返回 None**（而不是把错的几何发回去）—— zoom 变了就该重标，这是**事实**，
      "凑合着用"才会让人拿着错了的坐标去查寻路 ✗。
      ⚠ 只**新**写下来的标定才带 zoom 字段；老文件没有 ⇒ 一律当 zoom=1
      （`zoom_of`），于是"现在是 zoom=3"的机器上它们会判成不适用 ⇒ 界面会说
      「这份标定不是在当前 zoom 下标的」，人来决定重标还是改 zoom —— 这就是要它说的那句话。
    """
    srcs = load_calibs(map_id)
    if src:
        if src in srcs:
            return srcs[src] if _zoom_ok(srcs[src], zoom) else None
        if "" in srcs:
            if not _zoom_ok(srcs[""], zoom):
                return None
            out = dict(srcs[""])
            out["legacy"] = True        # 老格式：先当它可用，但让界面提醒重量
            return out
        return None
    if "" in srcs:
        return srcs[""] if _zoom_ok(srcs[""], zoom) else None
    if len(srcs) == 1:
        only = next(iter(srcs.values()))
        return only if _zoom_ok(only, zoom) else None
    return None


def save_calib(map_id, calib, src=None, zoom=None):
    """写标定。**src 给了就只覆盖那一条来源**，其它来源原样保留。

    src=None 走老格式（整份平铺）—— 只有诊断/迁移用；界面那条路一律传 src，
    不传的话写一次就把另一条来源的标定抹掉了（而人看不出来）。

    ⭐ `zoom`（**可选**）：写进这一份，标明**这份几何是对着哪个 A 机 zoom 量的**
      （对上 `zoom_of` / `load_calib` 的 zoom 参数）。**保存标定那几条路都要传** ——
      不写的话，将来 zoom 一变就没人知道这份几何其实已经不成立了 ✗。
    """
    import json as _json
    p = calib_path(map_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(calib or {})
    body.pop("legacy", None)            # 读的时候临时加的标记，别写进文件
    if zoom is not None:
        body["zoom"] = max(1, int(zoom))
    if src is None:
        out = body
    else:
        srcs = load_calibs(map_id)
        srcs.pop("", None)              # 老格式那份已经由这次保存的来源接管了
        srcs[src] = body
        out = {"v": CALIB_V2, "sources": srcs}
    p.write_text(_json.dumps(out, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def available():
    """已导出的地图 id 列表（排序）。"""
    d = map_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if not p.stem.startswith("_"))
