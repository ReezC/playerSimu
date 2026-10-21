"""叠图核对：把「面板现在显示底图哪一块」**画出来 / 量出来**（2026-10-18 从 minimap 切出）。

`perception/minimap.py` 只负责"量"（定位、标定几何在 `map_calib`）；本模块负责把结果
摆到图上给人看：叠图、差异、相似度打分 —— 工作台的「叠图核对」与标定窗用它。

⚠ 依赖单向：本模块 import `minimap`（只借 `_gray`），`minimap` **不** import 本模块。
"""

from pathlib import Path

import cv2
import numpy as np

from core import mapdata
from core.imgio import imread, imwrite
from perception import map_calib
from perception import map_const
from perception.minimap import _gray, overlap_pearson


# ══════════════════════════════════════════════════════════════
# 「现在显示的是底图哪一块」（crop / 局部小地图专用）
# ══════════════════════════════════════════════════════════════

#: tools/map_terrain_view.py 把底图放大到大约这个宽度（更宽的底图不缩小）
_OVERLAY_W = 1280


def overlay_zoom(canvas_w):
    """底图 → 叠加图 的放大倍数（和 tools/map_terrain_view.render 同一套算法）。"""
    return max(1, int(round(float(_OVERLAY_W) / max(1, int(canvas_w)))))


def view_rects(loc, panel_wh, canvas_wh):
    """crop 模式：算出「面板现在显示的是底图哪一块」→ (底图矩形, 叠加图矩形)。

    面板是**放大过**的（scale = 面板像素 / 底图像素），所以它在底图坐标里覆盖
    的大小是 `面板 / scale`，起点是 `view - inset/scale`。

    注意 inset 只能减一次：模板是从面板里裁掉 inset 再去匹配的，匹配位置本身
    就是底图坐标（见 locate_crop / panel_to_canvas 的注释）；scale 也是同理 ——
    匹配时面板已经按 1/s 缩回去了，所以 view 就是底图坐标，**不要再折算一次**。
    """
    pw, ph = panel_wh
    sx, sy = map_calib.scales_of(loc)
    inset = float((loc.get("offset") or [4, 4])[0])
    vx, vy = loc.get("view") or (0, 0)
    x0, y0 = vx - inset / sx, vy - inset / sy
    x1, y1 = x0 + pw / sx, y0 + ph / sy
    z = overlay_zoom(canvas_wh[0])
    return ((int(x0), int(y0), int(x1), int(y1)),
            (int(x0 * z), int(y0 * z), int(x1 * z), int(y1 * z)))


def overlay_draw_rects(loc, panel_wh, canvas_wh):
    """→（叠加图上的**源矩形**（None = 整张）, 面板里的**目标矩形**）。

    把标定好的叠加画到别处时就靠这一对：源矩形从叠加图上取一块，
    贴到目标矩形（面板坐标）。

      fit  ：面板里放的是**整张**底图 → 源 = 整张叠加图，目标 = 面板里从 `offset`
             起、`底图 × scale` 那么大的那一块（面板比它大的地方本来就该留白）。
      crop ：面板里放的是底图的**一块** → 源 = 那一块（`view_rects` 算好的），
             目标 = 整个面板。

    **为什么抽成一份**：标定弹窗的叠加层、以及「把叠加画到实时画面上」都要用
    同一个映射。各写一份迟早对不上，而这种偏最阴 —— 两边看着都"差不多"，
    只有拿尺子量（或者进游戏走两步）才发现差一截。
    """
    pw, ph = panel_wh
    cw, ch = canvas_wh
    if loc.get("mode") == map_const.MODE_CROP:
        _canvas_rect, img_rect = view_rects(loc, panel_wh, canvas_wh)
        return img_rect, (0, 0, int(pw), int(ph))
    sx, sy = map_calib.scales_of(loc)
    ox, oy = loc.get("offset") or (0, 0)
    return None, (int(ox), int(oy),
                  int(round(cw * sx)), int(round(ch * sy)))


def diff_panel_vs_canvas(panel, canvas, calib, gain=None, color=False, local=None):
    """面板图 **减** 底图上它盖住的那一块 ⇒ 差分图 `(vis, info)`（用户 2026-10-06 ✓）。

    ⭐ 「像素差分地图」那一路（路线识别 → 地形图 → **显示类型**）**唯一的算法**：
    面板被按标定摆在底图的某一块上（`panel_to_canvas` 是**唯一口径** ✓）⇒ 两者
    "本该长得一样" ✓ ⇒ 差出来的就是**底图上没有的东西** —— 最典型的就是**玩家点**
    （黄点）以及其它实时元素 ✓（这正是用户 2026-10-06 问"是不是颜色相减"时想看的 ✓）。

    ⚠⚠ **先减掉"背景"再放大** ✗ 不许直接放大：底图与游戏里那张小地图**常常不同源**
      （底色 / 透明度 / 线条都不一样 ✗）⇒ 直接放大整幅糊成白的、什么也看不出来 ✓；
      减掉背景之后，"**比周围多差了那么一截**"的那一小块才会跳出来 ✓（那才是要看的 ✓）。

    ⭐⭐⭐ **背景是"逐块"的**（用户 2026-10-12 ✓ 现场：石人寺院Ⅳ 差分"处理很差"✓）：
      · 老口径减的是**整幅一个中位**（`floor` ✓）⇒ 只能消掉**均匀**那一部分 ✗；
      · 实测现场那一帧：游戏里的小地图**半透明**（背景透出画面 ⇒ 底图纯黑处、面板
        灰度**中位 53 / 标准差 39.5**✗）、地形又是"放大 5.6 倍 + 抗锯齿"渲染 ⇒
        整幅到处碎白点、真玩家点被淹 ✓；
      · 现在减的是**每个像素周围 `k×k` 的均值**（`DIFF_LOCAL_PX` ✓ `cv2.blur` ✓）⇒
        实测残差中位 **67 → 5.1 灰阶**、地形区 95% 位 **86 → 46** ✓✓。
      ⚠ `floor`（整幅中位）**照旧如实报** ✓ —— 界面上那行还要让人看见"这张图整幅有多偏"
        ✓；判"是不是真不同源"改用 `res_med`（**局部归一之后**的残差中位 ✓）。
      ⚠ `local=0`（或负）⇒ **老口径**（减整幅中位 ✓ 零回归 ✓；用例/取证工具要复现老图时用 ✓）。

    ⚠ 它**只是"看"的**：不参与定位、不写盘、不缓存 ✓（真要拿它去找黄点，那是下一步 ✓）。

    返回 `(vis, info)`：
      · `vis` = 与 `panel` **同尺寸**的 3 通道 BGR（= `clip((原始差 − 局部背景) × gain)` ✓）；
        算不出来 ⇒ `None`（**不给假图** ✗）；
      · `info` = `{ok, why, color, gain, local, floor, res_med, peak, at, raw_mean,
        raw_peak, rect, gray}` ——
        `at` = "**比周围多差得最厉害**"那个点的**面板像素** `(x, y)` ✓；`rect` = 底图上
        盖住的那一块 `(x0, y0, x1, y1)` ✓（调用方可以直接报给人看 ✓）。
    """
    if panel is None or canvas is None:
        return None, {"ok": False, "why": "没有面板图或底图"}
    try:
        ph, pw = int(panel.shape[0]), int(panel.shape[1])
        ch, cw = int(canvas.shape[0]), int(canvas.shape[1])
    except Exception:                       # noqa: BLE001 —— "看"的东西，坏形状别抛 ✗
        return None, {"ok": False, "why": "图的样子不对"}
    if pw < 8 or ph < 8:
        return None, {"ok": False, "why": "面板太小（%d×%d）" % (pw, ph)}
    try:
        bx, by = map_calib.panel_to_canvas(0.0, 0.0, calib)
        ex, ey = map_calib.panel_to_canvas(float(pw), float(ph), calib)
    except Exception:                       # noqa: BLE001
        return None, {"ok": False, "why": "标定几何读不出来"}
    x0, x1 = (bx, ex) if ex >= bx else (ex, bx)
    y0, y1 = (by, ey) if ey >= by else (ey, by)
    ix0, iy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    ix1, iy1 = min(cw, int(np.ceil(x1))), min(ch, int(np.ceil(y1)))
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None, {"ok": False,
                      "why": "面板落在底图外面（底图 %d×%d，要 (%d,%d)-(%d,%d)）"
                             % (cw, ch, ix0, iy0, ix1, iy1)}

    def _bgr(a):
        if a.ndim == 2:
            return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        return a[:, :, :3] if a.shape[2] > 3 else a

    try:
        want = cv2.resize(canvas[iy0:iy1, ix0:ix1], (pw, ph),
                          interpolation=cv2.INTER_LINEAR)
    except Exception as ex:                 # noqa: BLE001
        return None, {"ok": False, "why": "缩放底图那一块失败（%s）" % ex}
    d = cv2.absdiff(_bgr(panel), _bgr(want))
    # ⭐⭐ **别转 float32**（用户 2026-10-06 ✓ 帧率现场 ✓ `py-spy`：`_amax` 占 **9.4%** ✗）：
    #   老写法 `.astype(np.float32)` 凭空建一个 **2.7 MB** 浮点数组（1920×1080 那档更大 ✗），
    #   下面又建一个（`extra` ✓）⇒ 每帧几次整幅分配＋遍历 ✗。而**中位数与后面的运算
    #   对 uint8 与 float32 完全同值** ✓（差值是整数 ✓ `clip`/`*gain` 在同一个表达式里做 ✓）
    #   ⇒ 去掉那次转换、并把两个 `clip` 合成一遍 ✓ —— **算出来的数一个都没变** ✓。
    g = d.max(axis=2)                        # 逐像素取"差得最多的那个通道" ✓（uint8 ✓）
    floor = float(np.median(g))              # 整幅背景档（**照旧如实报** ✓ 界面那行用它 ✓）
    # ⭐⭐⭐ **减掉的是「局部背景」**（用户 2026-10-12 ✓ 见 `DIFF_LOCAL_PX` 的说明 ✓）：
    #   每个像素减掉**它周围 `k×k` 的均值** ⇒ 整片/成片的亮度差（半透明小地图、底色偏 ✓）
    #   一次性消掉 ✓，只留"**比周围多出来的那一截**"（点 / 实时元素 ✓）。
    #   `cv2.blur` 是 O(1) 的箱式均值 ✓（只多一个浮点数组 ⇒ 与老写法同量级 ✓）。
    #   ⚠ `local=0/负` ⇒ **老口径**（`base` = 整幅一个常数 ⇒ 与老写法逐像素同值 ✓ 零回归 ✓）。
    gf = g.astype(np.float32)
    _k = int(map_const.DIFF_LOCAL_PX if local is None else local)
    if _k >= 3:
        _k = _k + 1 if (_k % 2 == 0) else _k         # 奇数窗口 ✓（居中 ✓）
        base = cv2.blur(gf, (_k, _k))
    else:
        _k = 0
        base = np.float32(floor)
    res = gf - base                          # ⭐ "比周围多差了这么多"（正 = 更多 ✓）
    gv = np.clip(res * float(gain if gain is not None else map_const.DIFF_GAIN),
                 0.0, 255.0).astype(np.uint8)
    # ⭐ `at` / `peak` 取的是**残差** `res`（不是 `g` ✗）—— 老写法里 `base` 是整幅常数 ⇒
    #   `argmax(g)` 与 `argmax(res)` 同一个点 ✓（`clip` 单调 ✓）；现在 `base` **逐像素不同**
    #   ⇒ 必须直接在 `res` 上取 ✓（不然会挑到"整幅最偏"的那一片、而不是"比周围最突出"的点 ✗）。
    #   ⚠ 全幅一样时 `argmax` 取第一个 ✓（同老口径 ✓）。
    #   `peak = max(0, res[at])` = "比周围多出来的那一截" ✓（**不乘 gain** ✓ 与老口径同量纲 ✓）。
    vis = cv2.cvtColor(gv, cv2.COLOR_GRAY2BGR)
    # ⭐⭐ `color=True` ⇒ **把点的颜色显示出来**（用户 2026-10-06 ✓ 原话："像素差分地图能把点的
    #   颜色显示出来而不是纯白吗？"✓）：只把"**差得够亮**"的那些像素（`gv ≥ DOT_DIFF_THR` ✓
    #   与定位那一层**同一根门限** ✓）**按面板本来的颜色**画出来、其余留黑 ✓ —— 于是玩家点
    #   还是**黄色**（而不是一团白 ✓），底图上本来就有的东西依旧是黑的（差 ≈ 0 ✓）。
    #   ⚠ **默认仍旧是灰阶**（`color=False` ✓）：**定位那一层**（`dot_candidates_diff`）吃的
    #     就是这份归一化灰阶 ✓（它的门限是"峰值的比例"✓ ⇒ 换成颜色就变味了 ✗ 别改它 ✓）。
    if color:
        _m = gv >= map_const.DOT_DIFF_THR
        vis = np.zeros_like(vis)
        vis[_m] = _bgr(panel)[_m]
    ay, ax = np.unravel_index(int(res.argmax()), res.shape)
    return vis, {"ok": True, "why": "", "color": bool(color),
                 "gain": float(gain if gain is not None else map_const.DIFF_GAIN),
                 "local": int(_k),
                 "floor": round(floor, 1),
                 #: ⭐ "局部归一**之后**还剩多少" —— **判"真不同源"用这一个** ✓
                 #:   （`floor` 只管"整幅偏多少"：偏得再多，只要**均匀**，局部一减就没了 ✓）。
                 "res_med": round(float(np.median(np.abs(res))), 1),
                 "peak": round(max(0.0, float(res[ay, ax])), 1),
                 "at": (int(ax), int(ay)),
                 "raw_mean": round(float(g.mean()), 1),
                 "raw_peak": round(float(g.max()), 1),
                 "rect": (ix0, iy0, ix1, iy1),
                 #: ⭐ 归一化灰阶本体（`vis` 就是它 `cvtColor` 出来的 ✓）—— 定位那一层
                 #: 直接用这份算门限 ✓，省掉"再取一次三通道 max"（见 `dot_candidates_diff` ✓）。
                 "gray": gv}


def frame_overlay_rects(loc, frame_rect, canvas_wh, calib_panel=None):
    """→（源矩形, **实时画面坐标**下的目标矩形）。

    面板在实时画面里的位置由 `frame_rect=(fx, fy, fw, fh)` 给出（工作台
    「小地图定位 → 框选小地图」框出来的那一块，存 config/live.yaml 的 `mmap_crop`），
    标定说面板里该怎么放 —— 两者拼一下就是"画在画面的哪儿"。

    **收流来源同样适用**：那一路小地图虽然另有独立推流（画质好、黄点不糊），
    但小地图面板本身**也在主画面里**（只是被 H.264 压过），所以"它在画面的哪儿"
    照样是框出来的那一块，和来源无关。

    `calib_panel` = **标定当时那块面板的尺寸** `(w, h)`（不传 = 不折算，老行为）。
    ⚠ 为什么必须传（2026-09-26 实测踩过）：标定记的是"**那块**面板像素 ↔ 底图像素"的
    换算，而这里要往"**现在**框出来的那一块"上画 —— 两者尺寸可能差好几倍（典型：A 机
    推流带 zoom，标定面板 754px 宽，而主画面里的小地图只有 251px）⇒ 不折算的话，
    叠图会被整块放大：`底图 134 × scale 5.63 = 754` 的方块糊在 251px 的小地图上。
    传了它，就按 `标定面板 ÷ 当前那块画面` 的比例把几何先换算过去再画 ——
    于是 A 机改 zoom、或面板尺寸变了，叠图都不会再被放大。

    ⚠⚠ **折算哪些量**（2026-09-29 修，用户报"**刚标定完叠加图显示就不对**"✓）：
      标定那份说的是 `底图 = (p标定 − offset) / scale + view`，而框选那块面板与它
      **看的是同一块地图**、只是像素尺寸差 z 倍（`p标定 = z · p框选`）⇒ 代进去：
      `底图 = (p框选 − offset/z) / (scale/z) + view` ⇒
      **`scale` 与 `offset` 要除以 z，`view` 不动** ✓（`view` 是**底图坐标**，跟面板
      放大几倍无关）。
      ⛔ 以前把 `view` 也除了 z ✗ ⇒ 叠图整块平移 `view × (1 − 1/z)` 个底图像素
      （实测 105040303：偏 (7.3, 69.8) 个底图像素，而那块面板一共才装得下
      (58.1, 60.8) 个 ⇒ **叠图跑到地图别处去了** ✗）。
      **为什么这个错一直没露**：`fit` 的 `view` 恒为 `[0, 0]`（整张底图不动）⇒ 除不除
      都一样 ✗；只有 `crop` 的 `view` 是"现在显示在地图哪儿"、必然非 0 ⇒ 一除就错 ✓
      —— 于是现象恰好是"**只有局部小地图**、刚标定完叠图就不对"。
    """
    fx, fy, fw, fh = (int(v) for v in frame_rect)
    loc2 = loc
    if calib_panel:
        try:
            pw = float(calib_panel[0] or 0)
        except (TypeError, ValueError, IndexError):
            pw = 0.0
        if pw > 0 and fw > 0 and abs(pw / float(fw) - 1.0) > 1e-6:
            z = pw / float(fw)                  # 标定面板 → 当前那块画面 的比例
            sx, sy = map_calib.scales_of(loc)             # 两轴都要折算（比例 z 本身是等比的 ✓）
            loc2 = map_calib.with_scales(loc, sx / z, sy / z)
            loc2["offset"] = [float(v) / z for v in (loc.get("offset") or (0, 0))]
            # ⛔ **`view` 不折算**（`底图 = (p − offset/z)/(scale/z) + view` ⇒ view 原样 ✓）
            #    —— 这里原来写的是 `loc2["view"] = [v / z ...]`，crop 图上会让叠图整体
            #    平移几个到几十个底图像素（见上面那段 ⚠⚠ ✓）。
    src, (dx, dy, dw, dh) = overlay_draw_rects(loc2, (fw, fh), canvas_wh)
    return src, (fx + dx, fy + dy, dw, dh)



def align_score(panel, terrain, calib, max_samples=6000, min_samples=64,
                near=False, near_px=3, near_max=9):
    """**当前这套几何贴不贴** → `{"score", "overlap", "ok", "why", "near"}`（**不做搜索** ✓）。

    和 `locate*`（找位置）是两件事：这里固定用你**手上这套几何**，只问"面板和底图对得上吗"。
    为什么要它（用户 2026-09-29 第 ⑤ 条 ✓ 原话："对于局部地图，评判分应该按局部来，
    必须制定剪裁范围才能让评分合理"）：
      · `locate` 给的是"**整张面板**在底图里最像哪儿"——crop（局部小地图）下，面板只是
        底图的一小块，那个分跟"你对齐了没有"没关系 ✗（手工对齐的更干脆：没有分 ✓）；
      · 这里改成**只算重叠区**：把面板像素按当前几何映到底图上，**落在底图里**的那些
        才参与打分 ⇒ "剪裁范围"由**这套几何自己定出来**，不需要人再去框一块 ✓
        （要靠人剪裁才能算分，等于把这件客观的事变主观 ✗）。
    分数 = 两串像素的**归一化相关**（和 `TM_CCOEFF_NORMED` 的 0~1 同一个口径 ✓）：
      1.0 = 完全重合；≥ `TRUST_SCORE`(0.8) 基本就是"对上了" ✓。
    `overlap` = 面板里落在底图上的像素占比 —— **这个数小就说明"框多了"**（把游戏 UI /
      面板外那圈框进了 crop 区域）⇒ 分再高也别全信 ✓（这也是 `_refresh_judge` 要一起
      报它的原因 ✓）。

    ⭐⭐ `near=True` ⇒ 多算一圈**邻域**，返回 `near = {"score", "dx", "dy"}`（**相对现在这套
      几何**平移 `(dx, dy)` 底图像素之后最好的那一档；`(0, 0)` 就是现在 ✓）。

      **为什么非有不可**（用户 2026-09-29 下一句就是："森林迷宫III项目，我已经手工对得很
      好了**匹配分还是很低**！"✗）：**绝对分不是一把通用的尺** —— 有的图客户端小地图
      的画法与底图**不同源**（自己另画一套线/底色），那类图**对得再准也只有 0.3~0.5** ✗
      ⇒ 光看"0.42 低于 0.55"会让人以为"我没对齐"，于是把对的几何改坏 ✗✗（判断贴合分
      **低不低**，只能和**它自己**比：附近有没有明显更高的一档 ✓）。
      `near["score"] - score <= 0.02` ⇒ **已经在这一档最好** ✓（这时分数低是"这张图
      天生如此"，不是你没对齐 ✓）；差得多 ⇒ `near` 里那对 `(dx, dy)` 就是**该往哪挪** ✓。

    ⚠ **只读**：不搜位置、不改标定、不写文件；`near=False` 时每拍 ~0.3 ms，
      开邻域约 9~13 次 ⇒ ~4 ms（标定弹窗每拍算一次，够用 ✓）。
    """
    out = {"score": None, "overlap": 0.0, "ok": False, "why": "", "near": None}
    canvas = getattr(terrain, "canvas", None) if terrain is not None else None
    sc, ov, why = overlap_pearson(panel, canvas, calib, max_samples, min_samples)
    out["overlap"] = ov
    if sc is None:
        out["why"] = why
        return out
    out["score"] = sc
    out["ok"] = True
    if near:
        out["near"] = _near_best(panel, canvas, calib, sc,
                                 near_px, near_max, max_samples, min_samples)
    return out


def _near_best(panel, canvas, calib, base, near_px, near_max, max_samples, min_samples):
    """邻域里最好的一档（**爬坡**：先 3×3，最好的落在边上就再往外走一档）→
    `{"score", "dx", "dy"}`（`(0,0)` = 现在这套 ✓）。

    ⚠ 只平移 `view`（crop）/ `offset`（fit）—— 也就是**只回答"往哪挪"**；缩放不在里面 ✓
    （缩放的微调是另一件事，见 `refine_calib` ✓；两者混在一起搜，人读不懂那个数）。
    """
    if not map_calib.has_geometry(calib):
        return None
    # 邻域要算十几次 ⇒ 灰度**只转一次**（不然每转一次 675×600 就是 ~1 ms ✗）
    panel = panel if getattr(panel, "ndim", 3) == 2 else _gray(panel)
    canvas = canvas if getattr(canvas, "ndim", 3) == 2 else _gray(canvas)

    def at(dx, dy):
        c = dict(calib)
        if calib.get("mode") == map_const.MODE_CROP:
            v = list(calib.get("view") or (0, 0))
            c["view"] = [float(v[0]) + dx, float(v[1]) + dy]
        else:
            o = list(calib.get("offset") or (0, 0))
            c["offset"] = [float(o[0]) - dx * float(map_calib.scales_of(calib)[0]),
                           float(o[1]) - dy * float(map_calib.scales_of(calib)[1])]
        s, _ov, _w = overlap_pearson(panel, canvas, c, max_samples, min_samples)
        return s

    best = {"score": float(base), "dx": 0, "dy": 0}
    for k in range(1, max(1, int(near_max / max(1, near_px))) + 1):
        r = k * near_px
        found = False
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                if dx == 0 and dy == 0:
                    continue
                s = at(dx, dy)
                if s is not None and s > best["score"]:
                    best = {"score": float(s), "dx": dx, "dy": dy}
                    found = True
        if not found:
            break                      # 这一圈没有更好的 ⇒ 不再往外走（省时间 ✓）
    return best


def crop_compare(frame, canvas, rect):
    """左 = 实时小地图面板，右 = 底图上匹配到的那一块（并排看对不对）。"""
    x0, y0, x1, y1 = rect
    h, w = frame.shape[:2]
    right = np.full((h, w, 3), 40, np.uint8)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if cx1 > cx0 and cy1 > cy0:
        piece = canvas[cy0:cy1, cx0:cx1]
        right[:] = cv2.resize(piece, (w, h), interpolation=cv2.INTER_NEAREST)
    pad = np.full((h, 6, 3), 30, np.uint8)
    return np.hstack([frame, pad, right])


def draw_on_overlay(map_id, rect, out=None):
    """在 `<id>_overlay.png` 上框出「当前是这一块」→ 返回写出的路径（没有叠加图返回 None）。

    这就是"当前小地图是叠加图的哪一块"最直观的答案：直接把框画在那张图上。
    （用了 `--dx/--dy` 微调过叠加图的话，框会差那几个像素。）
    """
    src = mapdata.map_dir() / ("%s_overlay.png" % map_id)
    if not src.exists():
        return None
    img = imread(str(src))
    if img is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in rect)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 3)
    org = (max(4, x0 + 6), max(26, y0 - 10))
    cv2.putText(img, "live view", org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(img, "live view", org, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2)
    dst = Path(out) if out else (mapdata.map_dir() / ("%s_overlay_live.png" % map_id))
    imwrite(dst, img)
    return dst
