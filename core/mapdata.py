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
    LADDER_DX, LADDER_PAD = 10, 12

    def __init__(self, map_id, data, canvas=None):
        self.id = str(map_id)
        self.info = dict(data.get("info") or {})
        self.mini = dict(data.get("miniMap") or {})
        self.canvas = canvas

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

    def find_below(self, x, y, tol=40):
        """脚下最近的可站立平台 → (x, y_on_line) 或 None。

        口径抄 MapleNecrocer 的 `Footholds.FindBelow`：在 x 覆盖范围内的**非墙**
        foothold 里，取 y 不小于给定 y（允许 tol 容差）中**最小**的那个，再按直线插值。

        **tol 默认 40 而不是 2**：小地图上的黄点画的是玩家**中心**
        （`MiniMap.cs` 用的是 `Game.Player.X/Y`），而地面在脚底 —— 实测 105090700
        的出生点 sp 离脚下地面 **8 世界像素**（玩家越高/站着不动时会有小幅波动）。
        所以「我站在哪块平台上」要按"下方几十像素内的最近地面"来判，不能用 2px。
        """
        best = None
        for f in self.footholds:
            if f.is_wall or not (f.left <= x <= f.right):
                continue
            fy = f.y_at(x)
            if fy < y - tol:          # 在头顶上方（容差内不算）
                continue
            if best is None or fy < best:
                best = fy
        return None if best is None else (x, best)

    def segment_of(self, x, y, tol=40):
        """点 (x, y) 落在哪条段上（判定"我现在站在哪块平台"）→ Segment 或 None。

        **按 find_below 的口径实现**（不是"离得足够近"）：黄点在玩家中心、
        离地面 ~26px，用固定距离判定会一直判不到。所以先找脚下地面，
        再反查它属于哪条段。
        """
        pos = self.find_below(x, y, tol)
        if pos is None:
            return None
        for seg in self.segments:
            for f in seg.footholds:
                if f.is_wall or not (f.left - 1 <= x <= f.right + 1):
                    continue
                if abs(f.y_at(x) - pos[1]) <= 0.51:
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

    with_canvas=True 时顺便把小地图底图读进 `Terrain.canvas`（numpy BGR，可能要 cv2）。
    """
    d = map_dir()
    try:
        data = json.loads((d / ("%s.json" % map_id)).read_text(encoding="utf-8"))
    except Exception:
        return None

    canvas = None
    if with_canvas:
        name = (data.get("miniMap") or {}).get("canvas")
        if name:
            import cv2
            canvas = cv2.imread(str(d / name), cv2.IMREAD_COLOR)
    return Terrain(map_id, data, canvas)


def calib_path(map_id):
    """小地图标定文件：`datasets/map/<id>.mapcalib.json`（每张图一份）。

    存的是「面板 → 底图 → 世界」的换算参数（见 `perception/minimap.py`）：
        {"mode": "fit" | "crop", "scale": …, "offset": [x, y],
         "view": [x, y], "score": …, "note": "…"}
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


def load_calib(map_id, src=None):
    """→ 某条来源的标定（形状同老格式）；没有 → None。

    src=None：老格式直接给；新格式**只有一份**时给那一份，多份则返回 None ——
    「哪一份」必须由调用方说清，猜错就是拿另一条来源的几何去算世界坐标。
    """
    srcs = load_calibs(map_id)
    if src:
        if src in srcs:
            return srcs[src]
        if "" in srcs:
            out = dict(srcs[""])
            out["legacy"] = True        # 老格式：先当它可用，但让界面提醒重量
            return out
        return None
    if "" in srcs:
        return srcs[""]
    if len(srcs) == 1:
        return next(iter(srcs.values()))
    return None


def save_calib(map_id, calib, src=None):
    """写标定。**src 给了就只覆盖那一条来源**，其它来源原样保留。

    src=None 走老格式（整份平铺）—— 只有诊断/迁移用；界面那条路一律传 src，
    不传的话写一次就把另一条来源的标定抹掉了（而人看不出来）。
    """
    import json as _json
    p = calib_path(map_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(calib or {})
    body.pop("legacy", None)            # 读的时候临时加的标记，别写进文件
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
