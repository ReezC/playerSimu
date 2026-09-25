"""把地形画在小地图底图上，供**人工验证**（S2 的验收手段）。

    python -m tools.map_terrain_view                # 用已导出的第一张图
    python -m tools.map_terrain_view 105090700
    python -m tools.map_terrain_view 105090700 --out 别的路径.png

画什么（都在**底图像素**坐标上，换算用 WZ 的 mag）：
    平台段     每段一色（按 next/prev 串出来的整条平台 —— 看它是不是连贯的一条）
    墙         红色（竖直 foothold，不可站立）
    绳梯/绳子  青色竖线
    传送点     黄色圆点 + 名字（sp / st00 …，都是 ASCII）
    刷怪点     品红小点（life，带巡逻范围 —— 以后寻路"去哪打"的目标来源）
    世界边界   白色框（由 miniMap 的 mag/centerX/centerY/width/height 推出）

打印的摘要同样是验收材料：段数、最长段、越界的 foothold（应该为 0）、
以及几个 find_below 抽样点。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core import mapdata

#: 相邻段的配色（循环用）—— 只为"看得出这是两条不同的平台"。
_PALETTE = [(60, 220, 60), (60, 200, 255), (255, 160, 60), (200, 80, 255),
            (255, 255, 60), (120, 255, 200), (200, 200, 200), (90, 130, 255)]


def _label(vis, text, org, color, scale=0.9):
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def render(t, out_path, use_canvas=True, target_w=1280, k=None, dx=0, dy=0):
    """把地形画在底图上。

    **必须放大**：底图只有 164x94 这种尺寸（世界跨度 2630 → 16 世界像素/底图像素），
    1:1 画出来根本看不清；这里统一放大到约 target_w 宽（最近邻，不引入模糊）。
    """
    size = t.canvas_size or (1600, 900)
    w, h = size
    z = max(1, int(round(float(target_w) / max(1, w))))
    W, H = w * z, h * z
    if use_canvas and t.canvas is not None:
        vis = t.canvas.copy()
        if vis.shape[1] != w or vis.shape[0] != h:
            vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_NEAREST)
        vis = cv2.resize(vis, (W, H), interpolation=cv2.INTER_NEAREST)
        # 底图偏亮时把地形压在暗化层上，线才看得清
        vis = cv2.addWeighted(vis, 0.55, np.zeros_like(vis), 0, 0)
    else:
        vis = np.zeros((H, W, 3), np.uint8)

    # k/dx/dy 是**人工标定**用的微调：默认用数据推出来的 px_per_world（实测 ≈16），
    # 差几像素就拧这几个数（`--k 16.0 --dx 2 --dy -1`），看图对准为止。
    kk = float(k or t.px_per_world)
    ox = float(t.mini.get("centerX") or 0)
    oy = float(t.mini.get("centerY") or 0)

    def P(x, y):
        return (int(round((x + ox) / kk * z + dx * z)),
                int(round((y + oy) / kk * z + dy * z)))

    # ---- 平台段（每段一色）----
    n_fh = n_wall = 0
    for i, seg in enumerate(t.segments):
        col = _PALETTE[i % len(_PALETTE)]
        pts = [P(*p) for p in seg.points]
        for a, b in zip(pts, pts[1:]):
            cv2.line(vis, a, b, col, 2)
        # 段号：状态行里那句「第 N 段」就是它 —— 不标出来的话，那个数字在图上
        # 对不上号（颜色是循环用的，段数一多就重色了）。
        # 标在**最左边那个点**上，多段之间不会挤在一起。
        if pts:
            lft = min(pts, key=lambda q: q[0])
            _label(vis, "%d" % i, (lft[0] + 6, max(22, lft[1] - 6)), col, 0.7)
        for f in seg.footholds:
            n_fh += 1
            if f.is_wall:
                n_wall += 1

    # ---- 绳梯 / 绳子 ----
    for L in t.ladders:
        a, b = P(L.x, min(L.y1, L.y2)), P(L.x, max(L.y1, L.y2))
        cv2.line(vis, a, b, (255, 255, 0), 2)
        cv2.circle(vis, a, 4, (255, 255, 0), -1)
        cv2.circle(vis, b, 4, (255, 255, 0), -1)

    # ---- 传送点 ----
    for p in t.portals:
        c = P(p.x, p.y)
        cv2.circle(vis, c, 7, (0, 215, 255), -1)
        cv2.circle(vis, c, 7, (0, 0, 0), 1)
        if p.pn:
            _label(vis, p.pn, (c[0] + 10, c[1] + 5), (0, 215, 255))

    # ---- 刷怪点（life）----
    for e in t.life:
        if str(e.get("type")) not in ("m", "1", "mob"):
            continue
        x, y = mapdata._i(e.get("x")), mapdata._i(e.get("cy"))
        c = P(x, y)
        cv2.circle(vis, c, 3, (255, 0, 255), -1)

    # ---- 世界边界 ----
    b = t.bounds
    if b:
        cv2.rectangle(vis, P(b[0], b[1]), P(b[2], b[3]), (255, 255, 255), 2)

    # ---- 图例 ----
    lines = [
        "%s   canvas %dx%d  (x%d)" % (t.id, w, h, z),
        "green/blue/...: foothold segments (%d) — 数字 = 段号"
        "（状态行里的「第 N 段」）" % len(t.segments),
        "red: walls   cyan: ladderRope (%d)" % len(t.ladders),
        "yellow: portals (%d)   magenta: mob spawns" % len(t.portals),
        # 尺度用 px_per_world（= 世界跨度/底图宽），**不是 mag**
        "1 px = %.2f world px   (span %s, mag=%s 仅记录)"
        % (t.px_per_world, t.world_span, t.mag or "-"),
    ]
    for i, s in enumerate(lines):
        _label(vis, s, (12, 34 + i * 34), (255, 255, 255), 0.8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return W, H, n_fh, n_wall          # 返回**实际输出**尺寸（已放大）


def main() -> int:
    ap = argparse.ArgumentParser(description="把地形画在小地图底图上（人工验证用）")
    ap.add_argument("map_id", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=float, default=None,
                    help="世界像素/底图像素（默认按 世界跨度/底图宽 推，实测≈16）")
    ap.add_argument("--dx", type=int, default=0, help="底图像素级的水平微调")
    ap.add_argument("--dy", type=int, default=0, help="底图像素级的垂直微调")
    args = ap.parse_args()

    avail = mapdata.available()
    mid = args.map_id or (avail[0] if avail else None)
    if not mid:
        print("datasets/map 里还没有地形数据。先导出：")
        print("  WzProbe.exe dump-terrain <WZ目录> datasets/map --only <地图id>")
        return 2

    t = mapdata.load(mid, with_canvas=True)
    if t is None:
        print("读不到 %s 的地形 JSON（datasets/map/%s.json）" % (mid, mid))
        return 2

    print("地图 %s" % t.id)
    print("  info: hideMinimap=%s fieldLimit=%s" % (t.minimap_hidden,
                                                    t.info.get("fieldLimit")))
    print("  小地图底图: %s" % ("有" if t.canvas is not None else "没有"))
    print("  mag=%s   center=(%s, %s)   底图 %s" % (
        t.mag, t.mini.get("centerX"), t.mini.get("centerY"), t.canvas_size))
    print("  世界范围: %s" % ((tuple(round(v, 1) for v in t.bounds)
                               if t.bounds else None),))
    print("  foothold %d（其中墙 %d）→ 串成 %d 条段"
          % (len(t.footholds), sum(1 for f in t.footholds if f.is_wall),
             len(t.segments)))
    print("  ladderRope %d   portal %d   life %d"
          % (len(t.ladders), len(t.portals), len(t.life)))

    if t.segments:
        longest = max(t.segments, key=len)
        print("  最长的段: %d 个 foothold，x[%d..%d]"
              % (len(longest), longest.left, longest.right))

    # ---- 一致性检查：地形坐标必须落在小地图推出的世界范围里 ----
    b = t.bounds
    if b:
        bad = [f.fid for f in t.footholds
               if not (b[0] - 2 <= f.left and f.right <= b[2] + 2
                       and b[1] - 2 <= min(f.y1, f.y2)
                       and max(f.y1, f.y2) <= b[3] + 2)]
        print("  越界的 foothold: %d %s" % (len(bad), bad[:5] if bad else ""))
        lb = [L.x for L in t.ladders if not (b[0] <= L.x <= b[2])]
        pb = [p.pn for p in t.portals if not (b[0] <= p.x <= b[2]
                                             and b[1] <= p.y <= b[3])]
        print("  越界的 ladder x: %d   越界的 portal: %d %s"
              % (len(lb), len(pb), pb[:5] if pb else ""))

    # ---- find_below 抽样（拿几个 portal/life 的 x 试）----
    print("  find_below 抽样:")
    for p in (t.portals[:2] + t.portals[-1:] if len(t.portals) > 2 else t.portals):
        r = t.find_below(p.x, p.y - 50)
        print("    x=%d（portal %s 上方 50px 处）→ %s" % (p.x, p.pn, r))

    out = args.out or (mapdata.map_dir() / ("%s_overlay.png" % t.id))
    w, h, n_fh, n_wall = render(t, out, k=args.k, dx=args.dx, dy=args.dy)
    print()
    print("已写出: %s  (%dx%d)" % (out, w, h))
    print("打开它对照游戏里的小地图：每色一条 = 一条平台，黄色 = 传送点，青色 = 绳梯。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
