"""面板 ↔ 底图 ↔ 世界的**换算与标定几何**（2026-10-18 从 `perception/minimap.py` 拆的第二块 ✓）。

一句话：这里只回答"**这个像素是世界的哪一点**"，**不认黄点、不碰推流、不画图** ✓：

| 组 | 干什么 | 函数 |
|---|---|---|
| ① | 面板几何（这份标定说的是哪块面板） | `panel_of` / `implied_panel` / `panel_mismatch` / `crop_rel_of` / `crop_from_rel` |
| ② | 标定几何（缩放 / 偏移 / 齐不齐） | `offset_of` / `world_offset` / `scales_of` / `with_scales` / `has_geometry` |
| ③ | **三套坐标的换算**（唯一口径） | `panel_to_canvas` / `panel_to_world` / `world_to_panel` |
| ④ | 与地形图对齐（量"标得准不准"） | `terrain_mask` / `canvas_to_panel_matrix` / `warp_canvas_to_panel` / `calib_overlap` / `overlap_verdict` |
| ⑤ | 双点标定（填两个坐标解几何） | `solve_two_point` / `calib_from_two_point` / `drop_stale_world_offset` / `world_offset_advice` |

**依赖方向是单向的** ✓：`map_const`（纯字面量）+ `core.mapdata`（读 `mapcalib.json`）
→ **本模块** → `minimap.py` / `map_dot.py` / 界面 / 工具。
⚠ 本模块**不许** import `minimap` / `map_dot` ✗ —— 否则"量一下几何"又要把收流线程、
模板匹配、差分那一坨拖起来 ✗；也正好挡住成环（判据 `selftest_arch::t_map_calib_split` ✓）。
⚠ **坐标口径只有这一处**（约定 10）：面板像素 → 底图像素 → 世界，别在界面/工具里再写一遍 ✗。
"""

import cv2
import numpy as np

from core import mapdata
from perception import map_const

#: 标定 dict 的形状（存 datasets/map/<id>.mapcalib.json）
#:   mode/scale/offset  —— 面板像素 → 底图像素：canvas = (panel - offset) / scale
#:   view               —— crop 模式专用：当前显示的是底图从 (view) 开始的那一块
#:   score              —— 标定时匹配得有多好（>0.8 才可信）
#:   scale_y            —— **可选**：y 轴自己的缩放（2026-09-27「双点标定」加的）。
#:                         ⚠ 为什么是"另开一个键"而不是把 `scale` 改成 `[sx, sy]`：
#:                         改类型会让**所有老读法**（`float(cal["scale"])`、模板匹配
#:                         存下来的那些、命令行工具）在读到新文件时当场炸 ✗；而
#:                         "`scale` = x 轴、`scale_y` 缺省 = 跟 x 一样"这套写法，
#:                         两个方向都兼容 —— 老代码读新文件只是把 y 也当成 x（两轴
#:                         本来就该几乎相等），新代码读老文件**精确**（缺省即同轴）✓。
#:                         取两轴一律走 `scales_of()`，别各处自己读（少一处就是
#:                         "换算用 x、画图用 y"这种最难查的分歧 ✗）。

# ══════════════════════════════════════════════════════════════
# ① 面板几何：这份标定说的是哪块面板
# ══════════════════════════════════════════════════════════════

def panel_of(calib):
    """标定里记的「**量它的时候那块面板多大**」→ `(w, h)`；没记（老标定）→ `None`。"""
    v = (calib or {}).get("panel")
    try:
        if v is not None and len(v) == 2:
            w, h = int(v[0]), int(v[1])
            if w > 0 and h > 0:
                return (w, h)
    except (TypeError, ValueError, IndexError):
        pass
    return None

def implied_panel(calib, canvas_wh):
    """**从几何反推**：「面板要塞下整张底图，得多大」→ `(w, h)`；推不出 → `None`。

    ⚠⚠ **方向别搞反**（我第一版就写反了 ✗，当场被现场数据打脸 ✓）：标定的换算公式是
      `canvas = (panel − offset) / scale`（见本文件上面的形状说明 ✓）
      ⇒ **面板 = scale × 底图 + offset** ✓（**乘**，不是除 ✗）。
      实测图 `110040000`：`scale=5.618`、底图 PNG **206×103**（⚠ 不是 `miniMap.width`
      那个 3305 ✗ —— 那是**地图**尺度，底图 PNG 是缩过的 ✗）⇒ 反推 **≈1157×584**
      ✓ 正好是 A 机推来的 **1164×585** ✓ ⇒ 那份标定**是自洽的** ✓ —— 所以**今天这一场
      不是"面板尺寸错"** ✗（我先前那个"整倍数错一半"的推断**作废** ✓，如实记 ✓）。
    `mode=crop`（局部小地图）底下这个数**没有"整张底图"的意思**（面板是滚动着看的一块 ✓）
      ⇒ 不参与判断（`panel_mismatch` 只认记下来的 `panel` ✓）。
    """
    if not has_geometry(calib) or (calib or {}).get("mode") == map_const.MODE_CROP:
        return None
    try:
        cw, ch = float(canvas_wh[0]), float(canvas_wh[1])
        sx, sy = scales_of(calib)
        if cw <= 0 or ch <= 0 or sx <= 0 or sy <= 0:
            return None
        ox, oy = (list(calib.get("offset") or (0, 0)) + [0, 0])[:2]
        return (sx * cw + float(ox), sy * ch + float(oy))
    except (TypeError, ValueError, IndexError):
        return None

def panel_mismatch(calib, panel, canvas_wh, tol=map_const.PANEL_TOL):
    """**这块面板 / 这份标定对得上吗** → `(ok, why)`。`ok=False` ⇒ 调用方**拒用**这份标定 ✓。

    用途（**为什么必须有一道"用之前"的闸**）：`perf.mmap_ok` 这类计数只看"**算出来了没有**"
    ✗ —— 标定与面板对不上时，它照样每拍 1.00 ✓，而世界坐标是**整倍数错**的 ✗
    ⇒ 决策层"**自信地错**"：2026-10-06 现场就是拿错坐标判"门 h007 还在右边"⇒ **一路怼墙** ✓。
    判据**优先**用标定里记的 `panel` ✓（精确 ✓；`gui/minimap_calib` 保存时写 ✓）；
    老标定没记 ⇒ 退回 `implied_panel` 反推 ✓（"面板要塞下整张底图得多大" ✓）；
    两样都没有（crop 的老标定 ✓）⇒ **不拦** ✓（没证据就别判死 ✗）。
    ⚠ 这正是 `core/mapdata.zoom_of` 说明里点名过的缺口：`panel` 字段**一直被读、全项目没人写**
      ✗ ⇒ 这里把闸补上 ✓。
    """
    if panel is None or not has_geometry(calib):
        return True, ""          # 没画面 / 没几何 ⇒ 不归这条管（各自有别的报错 ✓）
    try:
        pw, ph = int(panel.shape[1]), int(panel.shape[0])
    except Exception:            # noqa: BLE001 —— 拿不到尺寸就别拦 ✓
        return True, ""
    rec = panel_of(calib)
    ref = rec or implied_panel(calib, canvas_wh or (0, 0))
    if not ref:
        return True, ""
    rw, rh = float(ref[0]), float(ref[1])
    if abs(pw - rw) <= max(2.0, rw * tol) and abs(ph - rh) <= max(2.0, rh * tol):
        return True, ""
    return False, (
        "⚠ **这份标定对不上现在这块面板 ⇒ 已拒绝使用**（%s：标定那块是 **%.0f×%.0f**，"
        "现在推来的是 **%d×%d**，差 **%.2f 倍**）—— 照用会让世界坐标**整倍数错**、"
        "而且一路都不报错 ✗。请**重新标定**一次（路线识别 → 小地图定位 → 标定…），"
        "或者把 A 机推流/框选那块调回标定时的大小。"
        % ("记着" if rec else "从几何反推", rw, rh, pw, ph,
           (float(pw) / rw) if rw else 0.0))

def panel_rel_of(calib):
    """标定里记的「**面板占当初那一帧的多少**」→ `(nx, ny)`；没记 → `None`。

    ⭐⭐⭐ 为什么要记这个（用户 2026-10-20 ✓ 原话："为什么经常出现这个问题" ⇒ 选了方案 A）：
      · `mmap_crop` 有**比例副本** ✓（帧尺寸一变就按比例跟着缩，不用重框 ✓ 2026-10-17）；
      · 可标定钉的是**像素**尺寸 `panel` ✗ ⇒ 帧尺寸一变，面板像素跟着变 ⇒
        `panel_mismatch` 拒用整份标定 ⇒ 界面看起来像"认不出黄点"（2026-10-20 天空 ✓）。
    ⇒ 记下 `panel_rel` 之后，就能反推"**如果框选区域没变，这块面板现在该多大**"：
      对得上 ⇒ 这是**同一块区域被重采样** ⇒ 几何按比例缩放即可继续用 ✓（不用重标 ✓）；
      对不上 ⇒ 人**真重框了** ⇒ 照旧拒用 ✓（绝不猜 ✗）。
    """
    v = (calib or {}).get("panel_rel")
    try:
        if v is not None and len(v) == 2:
            nx, ny = float(v[0]), float(v[1])
            if nx > 0.0 and ny > 0.0:
                return (nx, ny)
    except (TypeError, ValueError, IndexError):
        pass
    return None


def scale_geometry(calib, kx, ky=None):
    """把一份几何**按 (kx, ky) 等比缩放**（面板像素变了、框选区域没变时用 ✓）。

    换算式是 `canvas = (panel − offset) / scale`（见文件头 ✓）⇒ 要让**同一个 panel 像素
    仍然指到同一个 canvas**，面板像素放大 k 倍时就得 `scale × k`、`offset × k` ✓
    （crop 模式的 `view` 是**底图像素**、不随面板变 ⇒ 不动 ✓）。

    ⚠ 缩放坏了（没有 scale / 值不是数）⇒ `None`（调用方退回"拒用"✓ 别拿半份几何去算 ✗）。
    """
    out = dict(calib or {})
    ky = kx if ky is None else ky
    try:
        sx = float(out.get("scale") or 0.0)
    except (TypeError, ValueError):
        return None
    if sx <= 0.0:
        return None
    out["scale"] = sx * float(kx)
    if out.get("scale_y") is not None:
        try:
            sy = float(out["scale_y"])
        except (TypeError, ValueError):
            sy = 0.0
        if sy > 0.0:
            out["scale_y"] = sy * float(ky)
        else:
            out.pop("scale_y", None)          # 坏的 y 缩放 ⇒ 让它跟 x（`scales_of` 的缺省 ✓）
    # ⚠ 这里读的是**几何偏移** `calib["offset"]`（`canvas = (panel − offset) / scale` ✓），
    #   **不是** `offset_of()` —— 那个读的是 `world_offset`（黄点脚底 ↔ 玩家原点那层 ✓
    #   见它的说明），两码事，混了就是把世界偏移当缩放中心 ✗（本用例当场抓到过一次 ✓）。
    try:
        _o = (list(out.get("offset") or (0.0, 0.0)) + [0.0, 0.0])[:2]
        out["offset"] = [float(_o[0]) * float(kx), float(_o[1]) * float(ky)]
    except (TypeError, ValueError):
        return None
    return out


def adapt_panel(calib, panel, frame_wh=None, tol=map_const.PANEL_TOL):
    """标定与面板尺寸不符时，**能不能按比例救回来** ⇒ `(calib_out, info)`。

    · `calib_out is not None` ⇒ 这份几何**已经换算到当前面板**，直接用 ✓（不用重标 ✓）；
    · `calib_out is None` ⇒ 救不回来 ⇒ 调用方**照旧拒用** ✓（老行为一字不变 ✓）。

    `info` = `{"ok", "why", "kx", "ky"}`（`why` 是给人看的一句话 ✓）。

    判据（**只有一条**，不猜 ✗）：
      `panel_rel`（标定里记的比例）＋ `frame_wh`（这一帧多大）⇒ 反推
      「如果框选区域没变，这块面板现在该是 **panel_rel × frame_wh**」
      ⇒ 与**实际**面板差在 `tol`（±5%，至少 ±2px）以内 ⇒ **等比缩放几何** ✓；
      否则 = 人真重框了 ⇒ `None` ✓。

    ⚠ 老标定没有 `panel_rel` ⇒ **一律救不了** ✓（宁可让重标一次，也别拿像素错配的
      几何算出"看着像真的"世界坐标 ✗ —— 那正是这道闸存在的理由 ✓）。
    """
    info = {"ok": False, "why": "", "kx": 1.0, "ky": 1.0}
    if panel is None or not has_geometry(calib or {}):
        info["why"] = "没有画面 / 没有几何 ⇒ 不归这条管"
        return None, info
    rec = panel_of(calib)
    if not rec:
        info["why"] = "这份标定没记面板尺寸（老标定）⇒ 不敢按比例套"
        return None, info
    try:
        pw, ph = int(panel.shape[1]), int(panel.shape[0])
    except Exception:                    # noqa: BLE001
        info["why"] = "拿不到面板尺寸"
        return None, info
    rel = panel_rel_of(calib)
    if not rel:
        info["why"] = "这份标定没记「面板 / 帧」的比例 ⇒ 分不清是\"重采样\"还是\"真重框\""
        return None, info
    try:
        fw, fh = float(frame_wh[0]), float(frame_wh[1])
    except (TypeError, ValueError, IndexError):
        info["why"] = "调用方没给这一帧的尺寸 ⇒ 没法反推"
        return None, info
    if fw <= 0.0 or fh <= 0.0:
        info["why"] = "帧尺寸不可用"
        return None, info
    rw, rh = float(rec[0]), float(rec[1])
    exp_w, exp_h = rel[0] * fw, rel[1] * fh
    if (abs(pw - exp_w) > max(2.0, exp_w * tol)
            or abs(ph - exp_h) > max(2.0, exp_h * tol)):
        info["why"] = ("框选区域变了（按标定的比例该是 **%.0f×%.0f**，实际推来 **%d×%d**）"
                       % (exp_w, exp_h, pw, ph))
        return None, info
    kx, ky = pw / rw, ph / rh
    if kx <= 0.0 or ky <= 0.0 or abs(kx - ky) > max(0.02, 0.05 * kx):
        info["why"] = ("面板两方向缩放不一致（%.2f / %.2f 倍）⇒ 不是简单重采样"
                       % (kx, ky))
        return None, info
    out = scale_geometry(calib, kx, ky)
    if out is None:
        info["why"] = "几何里没有可用的 scale ⇒ 缩不了"
        return None, info
    out["panel"] = [pw, ph]                 # 记成"现在的尺寸"⇒ 后面那道闸就过了 ✓
    info.update(ok=True, kx=kx, ky=ky,
                why=("面板 %d×%d → %d×%d（%.3f 倍）⇒ 已按比例换算几何，不用重标"
                     % (int(rw), int(rh), pw, ph, kx)))
    return out, info


def crop_rel_of(crop, shape):
    """**帧内像素** crop `[x, y, w, h]` ⇒ **比例** `[nx, ny, nw, nh]`（相对**这一帧** ✓）。

    ⭐⭐⭐ 为什么要有一份比例（用户 2026-10-17 ✓ 原话：
      「**为什么我移动本地游戏窗口后，小地图要重新框选？它不是相对游戏画面的坐标偏移
      记录的吗？？**」✗）：
      · `mmap_crop` 本来就是**相对帧**的 ✓（不是屏幕坐标 ✓）—— 但它是**像素** ✗
        ⇒ 只要**帧的尺寸**变了（窗口最大化 / 拖到缩放比例不同的显示器 / 换来源 ✓），
        它就错位或越界 ✗ ⇒ 只能重框 ✓；
      · 而"帧"本身是跟着窗口走的 ✓（`rect_rel` + `rect_win` ✓ 见 `core/wincap` ✓）
        ⇒ 那么**框也应该跟着比例走** ✓ —— 和 `rect_rel` 完全同一个道理 ✓。
    ⚠ 纯函数、无副作用 ✓；`shape` 用 `(H, W)` 或 `(H, W, C)` 都行 ✓
      （内部只取前两位 ✓）；w/h ≤ 0 或 shape 不可用 ⇒ `None` ✓（不猜 ✗）。
    """
    try:
        x, y, w, h = (float(v) for v in crop)
        H, W = float(shape[0]), float(shape[1])
    except (TypeError, ValueError, IndexError):
        return None
    if W <= 0 or H <= 0 or w <= 0 or h <= 0:
        return None
    return [round(x / W, 6), round(y / H, 6), round(w / W, 6), round(h / H, 6)]

def crop_from_rel(rel, shape):
    """**比例** ⇒ 当前这帧的像素 crop `(x, y, w, h)` ✓（`crop_rel_of` 的逆 ✓）。

    ⚠ 折算完**必须还在画面里** ✓：越界就夹回边界 ✓ —— 宁可裁小一点，也不要越界后
      "整块取不到 ⇒ 定位全停" ✗（`_mmap_panel_from_frame` 那边越界会直接返回 None ✓）。
    ⚠ 算不出来 ⇒ `None` ✓（调用方退回绝对 `mmap_crop` ✓ 老行为一字不变 ✓）。
    """
    try:
        nx, ny, nw, nh = (float(v) for v in rel)
        H, W = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError):
        return None
    if W <= 0 or H <= 0 or nw <= 0 or nh <= 0:
        return None
    x = max(0, min(W - 1, int(round(nx * W))))
    y = max(0, min(H - 1, int(round(ny * H))))
    w = max(1, min(W - x, int(round(nw * W))))
    h = max(1, min(H - y, int(round(nh * H))))
    return (x, y, w, h)

# ══════════════════════════════════════════════════════════════
# ② 标定几何：缩放 / 偏移 / 齐不齐
# ══════════════════════════════════════════════════════════════

def offset_of(calib):
    """从**标定字典**里读「坐标系偏移」→ `(x, y)`（没有 / 坏了给 `(0, 0)`）。

    ⚠ **口径**（2026-09-26 用户澄清，**别搞反**）：`定位结果 + 这个偏移` 才是玩家的
    **真实世界坐标** —— 那正是**游戏自己的世界坐标系**。地形数据（`core/mapdata` 的
    foothold / ladder 的 x、y）**本来就在这个系里**（例：105090600 的 fh44 地形 y=-208，
    它的世界 y 就是 -208）。⇒ 判据里拿玩家读数与地形坐标比大小是**直接比**，
    **不要**给地形坐标再加偏移（我犯过这个错：把"到达目标平台面"从 -208 放宽成 -175 ✗）。
    这个偏移修的是"黄点上的那个锚点 ↔ 玩家原点"那层对应，不是坐标系换算。

    ⚠ **2026-09-27 起，锚点是黄点的「下沿（脚底）」**（`dot_feet`，见那儿的口径说明）：
    双点标定采样的也是下沿 ⇒ 这一项从"重心 ↔ 原点"（≈ 半个点高，现场是 (7,33)）
    变成"下沿 ↔ 原点"的**残差**，正常应该接近 **(0, 0)** ✓。
    ⇒ 老的 (7,33) 那份是**重心口径**的，别搬进新标定（那是双重补偿，会凭白偏 33 世界像素 ✗）。
    """
    v = (calib or {}).get("world_offset") or []
    if len(v) == 2:
        try:
            return (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            pass
    return (0.0, 0.0)

def world_offset(map_id, src=None):
    """某张图 + 某个来源的「坐标系偏移」（见 `offset_of` 的口径说明）。

    ⚠ `src` 的默认值**不要在签名里写成 `map_const.SRC_STREAM`**：默认值在 `def` 那一刻
    就要取值 ⇒ 改成"定义在 `def` 之后"或改 import 顺序时会直接 `NameError`
    （2026-10-18 拆分前就是这么炸过一次 ✗）。所以这里 `src=None`，调用时再兜底。
    """
    if not map_id:
        return (0.0, 0.0)
    try:
        return offset_of(mapdata.load_calib(map_id, src or map_const.SRC_STREAM))
    except Exception:                                # noqa: BLE001
        return (0.0, 0.0)

def scales_of(calib):
    """标定里的**两轴缩放** → `(sx, sy)`。

    `scale` = x 轴；`scale_y` **缺省 = 跟 x 一样**（老文件就是这么写的，见
    `perception/minimap.py` 顶部那个形状说明）。读两轴一律走这里 —— 各处自己读
    迟早出现"换算用 x、画图用 y"的分歧，而那种偏只有拿尺子量才看得出来 ✗。
    """
    cal = calib or {}
    try:
        sx = float(cal.get("scale") or 1.0)
    except (TypeError, ValueError):
        sx = 1.0
    sy = cal.get("scale_y")
    if sy is None:
        return sx, sx
    try:
        sy = float(sy)
    except (TypeError, ValueError):
        return sx, sx
    return (sx, sy) if sy > 0 else (sx, sx)

def with_scales(calib, sx, sy=None):
    """把两轴缩放写进一份标定（copy）—— **两轴一样时不留 `scale_y`**。

    为什么不留：老文件里就没有这个键，而"缺省 = 同 x"这条口径已经够表达"等比" ✓；
    留着 `scale_y == scale` 只会让 diff 看起来像改过、也让人以为这份标定是量过两轴的。
    """
    out = dict(calib or {})
    out["scale"] = float(sx)
    if sy is None or abs(float(sy) - float(sx)) < 1e-9:
        out.pop("scale_y", None)
    else:
        out["scale_y"] = float(sy)
    return out

def has_geometry(calib):
    """这份标定里有没有**可用的几何** —— 「标定过没有」都该用它判断。

    **为什么不看 `score`**（现场踩到）：`score` 是「模板匹配有多像」，而
    **人工对齐没有匹配分**（人拿眼睛对出来的，分数就是 0）。以前拿 score 当
    「标过没有」，于是手工对齐存下去之后仍被判成「没标过」：

      · 标定弹窗重开时把刚存好的几何**覆盖**成粗略猜测 —— 现象就是
        「我明明点了保存，重开却变回去了」；
      · 工作台那行一直显示「未标定」。

    两条都能在 `tools/selftest_minimap.py` 的 `t_save_then_reopen` 里复现。
    """
    if not isinstance(calib, dict) or not calib.get("mode"):
        return False
    if not calib.get("scale"):
        return False
    need = "view" if calib.get("mode") == map_const.MODE_CROP else "offset"
    return calib.get(need) is not None

# ══════════════════════════════════════════════════════════════
# ③ 三套坐标的换算（唯一口径）
# ══════════════════════════════════════════════════════════════

def panel_to_canvas(px, py, calib):
    """面板像素 → 底图像素（calib 来自 locate / 标定文件）。"""
    sx, sy = scales_of(calib)
    ox, oy = calib.get("offset") or (0, 0)
    cx, cy = (px - ox) / sx, (py - oy) / sy
    if calib.get("mode") == map_const.MODE_CROP:
        vx, vy = calib.get("view") or (0, 0)
        cx, cy = cx + vx, cy + vy
    return cx, cy

def panel_to_world(px, py, calib, terrain):
    """面板像素 → **世界坐标**。"""
    return terrain.canvas_to_world(*panel_to_canvas(px, py, calib))

def world_to_panel(wx, wy, calib, terrain):
    """世界坐标 → **面板像素**（`panel_to_world` 的逆；**一处实现**）。

    为什么要单开一个（2026-09-27）：反向换算原来**散在三处** —— 双点标定窗的 `_uv`、
    地形视图 `image_xy`、编辑器的 `_add_background` ⇒ 谁改一处就分叉（约定 10 ✗）。
    而"把已知地标（portal / 绳梯 / 集合）投到面板像素上做核对"要用它（用户 2026-09-27
    决策③：投影产物按项目持久存 ✓）。
    ⚠ 与 `panel_to_world` **必须互逆**（crop 下 `view` 的加减号最容易写反）——
      用例 `t_world_to_panel_roundtrip` 钉着这条 ✓。
    """
    cx, cy = terrain.world_to_canvas(wx, wy)
    sx, sy = scales_of(calib)
    ox, oy = calib.get("offset") or (0, 0)
    if calib.get("mode") == map_const.MODE_CROP:
        vx, vy = calib.get("view") or (0, 0)
        cx, cy = cx - vx, cy - vy
    return cx * sx + ox, cy * sy + oy

# ══════════════════════════════════════════════════════════════
# ④ 与地形图对齐：量「标得准不准」
# ══════════════════════════════════════════════════════════════

def terrain_mask(img, thr=map_const.TERRAIN_THR):
    """一张小地图 / 底图 → 「**地形在哪**」的二值图（`uint8` 0/1 ✓）。

    ⭐⭐ 为什么用**梯度**（2026-10-09 ✓ 现场把三种口径都量了 ✓）：
      · 「亮 + 去饱和」✗：游戏小地图整体更亮 ⇒ 覆盖率 33.8% vs 底图 2.7% ✗ 掩码根本不是一回事；
      · 「局部纹理（标准差）」✓ 能用（0.253 → 单调 ✓）但覆盖率 7.6%/4.6% 偏小；
      · ⭐ 「**梯度幅值**」✓：覆盖率 29.3%/35.0% ✓ 两边接近、对齐处 IoU 最高（0.667 ✓）——
        而且它天然躲开"逐像素相关"栽的那个坑 ✗：**小地图上的图标/按钮是纯色块**（梯度少 ✓），
        地形（平台 / 藤蔓 / 台阶的边）梯度丰富 ✓。
    """
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return (cv2.magnitude(gx, gy) > float(thr)).astype(np.uint8)

def canvas_to_panel_matrix(calib):
    """「底图 → 面板」的 2×3 仿射（**口径照抄 `world_to_panel` ✓ 一处不差**）。

    `panel_to_canvas` 是"面板 → 底图" ✓（`canvas = (面板 − offset)/scale + view·crop` ✓），
    这里就是它的逆：`面板 = 底图 × scale + offset − view·scale`（crop ✓；fit 没有 view ✓）。
    ⚠ 别自己另拍一套（写反一个符号 ⇒ 叠图/重合率整块错位 ⇒ 又变成"工具说人错了" ✗）。
    """
    sx, sy = scales_of(calib)
    ox, oy = (calib.get("offset") or (0, 0))[:2]
    ox, oy = float(ox), float(oy)
    if calib.get("mode") == map_const.MODE_CROP:
        vx, vy = (calib.get("view") or (0, 0))[:2]
        ox, oy = ox - float(vx) * sx, oy - float(vy) * sy
    return np.float32([[sx, 0.0, ox], [0.0, sy, oy]])

def warp_canvas_to_panel(canvas, calib, size):
    """把底图整幅投到**面板坐标系**（`size=(w, h)` ✓）→ 图 / `None`（几何不合法 ✓）。

    用在两处：`calib_overlap`（算地形重合 ✓）和 `dump_diag`（存叠图/差分 ✓）。
    """
    if canvas is None or not isinstance(calib, dict) or not has_geometry(calib):
        return None
    try:
        M = canvas_to_panel_matrix(calib)
        return cv2.warpAffine(canvas, M, (int(size[0]), int(size[1])),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    except Exception:                                # noqa: BLE001 —— 诊断用，失败当没有 ✓
        return None

def calib_overlap(panel, terrain, calib, search=8, tol=1, thr=map_const.TERRAIN_THR):
    """这份几何下「面板地形」与「底图地形」的**重合度** ＋「最好要挪几像素」→ dict / None。

    ⭐⭐ 为什么必须有它（用户 2026-10-09 ✓ 原话："**这些提示是不正确的**，因为我的手动标定对齐
      地图已经接近完美了"）：**逐像素相关**（`locate_fit`/`locate_crop` 那条）在这张图上
      **天花板只有 0.45 左右** ✗（底图是项目按 wz **重绘**的、面板是客户端渲染 + 图标叠加 ✓）
      ⇒ 拿它当"标定对不对"的判据是**错的** ✗。现场实测（`data/mmap_diag/check_*.png` ✓）：
        灰度相关 0.135 / 边缘相关 0.478 / 抹掉覆盖层 0.45 —— 而人眼手工那份**只差 ~2px** ✓。
      这个判据换成**地形掩码重合**（`terrain_mask` ✓ 对覆盖层与渲染差异不敏感 ✓）：
        · 对齐处最高、往外单调下降（0.667 → 0.603 @+8px ✓）；
        · **最好那个平移量**直接告诉你"偏了几像素、往哪偏" ✓✓ —— 这才答得了"我标得准不准" ✓。

    返回（`None` = 面板/底图/几何缺 ✓）：
      · `iou`   最好那个平移下的 IoU ✓（**判"重合好不好"用这个** ✓）
      · `shift` 最好那个平移 `(dx, dy)` ✓（**判"要不要挪"用这个** ✓ 均在 ±`search` 内 ✓）
      · `iou0`  不挪时（= 当前这份几何）的 IoU ✓
      · `cover_panel` / `cover_canvas` 两边掩码覆盖率 ✓（差太多说明掩码本身不可比 ✓）

    ⚠ 容差 `tol`：两边掩码各膨胀 `tol` px 再算 ✓ —— 人眼对齐本来就允许 1~2px ✓（真值差 ~2px ✓）。
    """
    if panel is None or calib is None:
        return None
    # `terrain` 既可以是地形对象（`.canvas` ✓ 惯用形 ✓），也可以**直接是底图那张图** ✓
    #   —— 标定弹窗手里只有 `self.canvas` 那张图（没有地形对象 ✓），别为它再造一个 ✗。
    cv_img = getattr(terrain, "canvas", terrain)
    if cv_img is None or not hasattr(cv_img, "shape"):
        return None
    if not isinstance(calib, dict) or not has_geometry(calib):
        return None
    mp = terrain_mask(panel, thr)
    mc = terrain_mask(cv_img, thr)
    warped = warp_canvas_to_panel(mc, calib, (panel.shape[1], panel.shape[0]))
    if warped is None:
        return None
    k = np.ones((3, 3), np.uint8)
    a = cv2.dilate(mp, k, iterations=int(tol)) if tol > 0 else mp
    b = cv2.dilate(warped, k, iterations=int(tol)) if tol > 0 else warped
    na, nb = int((a > 0).sum()), int((b > 0).sum())
    if na == 0 or nb == 0:
        return None
    # ⚠ 一次算完**所有偏移**（别在 Python 里 one-by-one 做 warp ✗ 大面板上会卡）：
    #   二进制掩码的 `TM_CCORR` 就是"交集像素数" ✓ ⇒ 每对偏移都拿得到 inter ✓
    #   （上面把 b 采成 0/1 ✓）union = na + nb − inter ✓ ⇒ IoU 直接出 ✓。
    pad = int(search)
    big = cv2.copyMakeBorder(a.astype(np.float32), pad, pad, pad, pad,
                             cv2.BORDER_CONSTANT, value=0.0)
    corr = np.nan_to_num(cv2.matchTemplate(big, b.astype(np.float32),
                                           cv2.TM_CCORR))
    inter = corr
    union = float(na + nb) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        ious = np.where(union > 0, inter / union, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(ious)), ious.shape)
    shift = (int(ix) - pad, int(iy) - pad)           # (dx, dy) ✓
    iou0 = float(ious[pad, pad])
    return {"iou": float(ious[iy, ix]), "shift": shift, "iou0": iou0,
            "cover_panel": float(mp.mean()), "cover_canvas": float(mc.mean())}

def overlap_verdict(ov):
    """`calib_overlap` 的结果 → **一句人话**（界面/命令行直接用 ✓；`None` ⇒ 空串 ✓）。

    ⚠ 这句话要回答的正是用户那个问题："**我这份标定，到底准不准？**" ✓
      ⇒ 不许再拿"匹配分低"当结论 ✗（那在本图上是"尺子不适用"，不是"你错了" ✓）。
    """
    if not ov:
        return ""
    iou, (dx, dy) = float(ov.get("iou") or 0.0), (ov.get("shift") or (0, 0))
    mag = max(abs(int(dx)), abs(int(dy)))
    if iou >= map_const.OVERLAP_OK:
        if mag <= map_const.OVERLAP_SHIFT_OK:
            return ("地形重合 **%.2f**、最优点就在原地（偏 %+d,%+d px）⇒ "
                    "**与实际重合良好** ✓" % (iou, dx, dy))
        return ("地形重合 %.2f，但**最好再挪 %+d,%+d px** ⇒ 这份几何偏了约 %d px ✓"
                % (iou, dx, dy, mag))
    return ("地形重合只有 %.2f（%+d,%+d px 处最好）⇒ 这份几何与底图**对不太上**，"
            "建议重标或按提示挪一下 ✓" % (iou, dx, dy))

# ══════════════════════════════════════════════════════════════
# ④b 显示区跟不住时：**按地形掩码**把 `view` 找回来
# ══════════════════════════════════════════════════════════════

#: 找回来的搜索半径（底图像素 ±）—— 底图也就一两百像素高 ⇒ 90 已经能覆盖大半张图 ✓
CROP_RELOCK_SEARCH = 90
#: 掩码重合要**明显更好**才挪（不然就在原地抖 ✓ 平地上差 0.02 说明不了任何事 ✗）
CROP_RELOCK_GAIN = 0.05
#: 两边掩码覆盖率下限（太稀 = 这把尺子没内容 ⇒ 不敢按它挪 ✗）
CROP_RELOCK_MIN_COVER = 0.05


def relock_view(panel, terrain, calib, search=CROP_RELOCK_SEARCH,
                gain=CROP_RELOCK_GAIN, min_cover=CROP_RELOCK_MIN_COVER):
    """显示区跟不住时，**用「地形掩码」把 `view` 重新找回来** → `(view, info)`。

    ⭐⭐⭐ 为什么非它不可（用户 2026-10-20 ✓ 原话："森林迷宫_珠缨项目，**L9 完全找错了**"✗）：
      · 这张图（105040303）底图是**重绘的小地图**、面板是客户端渲染 ⇒ **逐像素相关几乎是
        平的**（天花板 ~0.55、峰间距 12~13px）✗ ⇒ `CropViewTracker` 一旦丢了锚，
        按像素找回来的那个位置**等于随机**（实测：像素分 0.547 的那份 view，
        真实位置差 **60 个底图像素 ≈ 一整层** ✗）；
      · 而**地形掩码**分得开（同一帧：错的 0.157 / 标定那份 0.251 / 真值 0.376 ✓）✓
        —— 这就是 `crop_view_trusted` 用来"复核"的那把尺子，这里改成"**用它找**" ✓。
      · 找不到（掩码太少 / 没有明显更好的位置）⇒ `view=None` ⇒ 调用方**照旧用标定那份**
        ✓（宁可承认"没跟住"，也别拿一个猜的位置算世界坐标 ✗）。

    ⚠ **符号**：`calib_overlap` 的 `shift` 是「底图 warp 到面板后**还要挪多少**」，
      要让 `view` 对上去得**反着改**：`view -= shift` ✓
      （2026-10-20 实测：view.y=107 + shift=(-2,-10) ⇒ 107-(-10)=117 时重合 0.410、
       真值 118.95 ✓；写成 `view += shift` 会掉到 0.211 ✗ —— 这一步错了比不做还坏 ✓）。
    """
    info = {"why": "", "iou": 0.0, "iou0": 0.0, "shift": None, "gain": 0.0}
    if panel is None or calib is None:
        info["why"] = "没有画面 / 没有几何"
        return None, info
    ov = calib_overlap(panel, terrain, calib, search=search)
    if not ov:
        info["why"] = "算不出地形重合（面板 / 底图 / 几何缺一样）"
        return None, info
    a, b = float(ov.get("iou") or 0.0), float(ov.get("iou0") or 0.0)
    cp, cc = float(ov.get("cover_panel") or 0.0), float(ov.get("cover_canvas") or 0.0)
    info.update(iou=a, iou0=b, shift=ov.get("shift"), gain=a - b)
    if min(cp, cc) < float(min_cover):
        info["why"] = ("地形掩码太少（面板 %.2f / 底图 %.2f）⇒ 这个尺子没内容，不敢按它挪"
                       % (cp, cc))
        return None, info
    if (a - b) < float(gain):
        info["why"] = ("按掩码没有明显更好的位置（+%.3f，门槛 %.2f）⇒ 就在原地不动"
                       % (a - b, gain))
        return None, info
    v = calib.get("view")
    try:
        vx, vy = float(v[0]), float(v[1])
    except (TypeError, ValueError, IndexError):
        info["why"] = "这份几何没有 `view`（不是局部小地图）⇒ 这条不适用"
        return None, info
    dx, dy = ov.get("shift") or (0, 0)
    info["why"] = ("按地形掩码挪 %+d,%+d px（重合 %.2f → %.2f）" % (-dx, -dy, b, a))
    return (vx - float(dx), vy - float(dy)), info


# ══════════════════════════════════════════════════════════════
# ⑤ 双点标定：填两个坐标解几何
# ══════════════════════════════════════════════════════════════

def solve_two_point(c1, p1, c2, p2):
    """两点标定：两对「底图像素 ↔ 面板像素」→ **两轴缩放 + 偏移 + 自查**。

    这是标定文件里 `scale` / `scale_y` / `offset` 的算法（用户 2026-09-27 要的
    「双点标定」；也就是 `docs/寻路设计.md` §5「地标法」那条的落地）。

    公式就是 `panel_to_canvas` 的反函数：`面板 = 底图 × scale + offset`，两轴各一套 ⇒
    每轴两个点给两个方程、解两个未知量（缩放、偏移）—— **恰好定解**。

    ⚠ **所以这里不能拿残差当自查**（两轴各自解的话它恒为 0，等于没查）—— 这条是
      写这个函数时才想透的，记在这儿：真正的自查是
        · `axis_gap_pct` = **两轴一致性**：游戏把整张小地图**等比**缩放 ⇒ `sx` 与
          `sy` 本该几乎相等；差得多就说明**两点没点在同一个地标上**（或画面被非等比
          拉过）⇒ 这就是用户要的"x/y 缩放不一致"那个提示 ✓；
        · `resid_px` = **按等比再拟合一次**（三个未知量、四个方程）的最大残差 ——
          这个数才有诊断价值（"如果两轴相同，最合适的几何差几个像素"）✓。

    **退化要拦住**（不能给个假数）：两点的底图 x 相同 ⇒ x 轴解不出来（分母 0）；
    y 相同 ⇒ y 轴解不出来；两点完全相同 ⇒ 两条都不成立。

    返回 `{"ok", "why", "scale", "scale_y", "offset", "iso_scale", "iso_offset",
    "resid_px", "axis_gap_pct"}`；`ok=False` 时 `why` 是人话（直接可以显示给人看）。
    """
    try:
        c1, p1 = [float(v) for v in c1], [float(v) for v in p1]
        c2, p2 = [float(v) for v in c2], [float(v) for v in p2]
        if not (len(c1) == len(p1) == len(c2) == len(p2) == 2):
            raise ValueError("不是两个坐标")
    except (TypeError, ValueError, IndexError):
        return {"ok": False, "why": "坐标要是四个数（x、y）：%r %r %r %r"
                                    % (c1, p1, c2, p2)}
    dcx, dcy = c2[0] - c1[0], c2[1] - c1[1]
    if abs(dcx) < 1e-6 and abs(dcy) < 1e-6:
        return {"ok": False, "why": "两点的底图坐标完全一样 —— 这是同一个点，量不出缩放"}
    if abs(dcx) < 1e-6:
        return {"ok": False, "why": "两点的底图 x 相同（同一列）⇒ x 轴的缩放解不出来，"
                                    "请把两点取成既不同列、也不同行"}
    if abs(dcy) < 1e-6:
        return {"ok": False, "why": "两点的底图 y 相同（同一行）⇒ y 轴的缩放解不出来，"
                                    "请把两点取成既不同列、也不同行"}
    sx = (p2[0] - p1[0]) / dcx
    sy = (p2[1] - p1[1]) / dcy
    if sx <= 0 or sy <= 0:
        return {"ok": False, "why": "算出来的缩放是负的（x %.3f / y %.3f）—— 两点多半"
                                    "左右（上下）对调了：实时图上那个地标应当由底图上"
                                    "**同一个**地标放大而来" % (sx, sy)}
    ox, oy = p1[0] - c1[0] * sx, p1[1] - c1[1] * sy
    # 按「等比」再拟合一次（3 个未知量、4 个方程 ⇒ 有残差）——抵消共用的那一个 s，
    # 偏移取两轴各自的均值（给定 s 时这才是最小二乘解）。
    mcx, mcy = (c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0
    mpx, mpy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    num = ((c1[0] - mcx) * (p1[0] - mpx) + (c2[0] - mcx) * (p2[0] - mpx)
           + (c1[1] - mcy) * (p1[1] - mpy) + (c2[1] - mcy) * (p2[1] - mpy))
    den = ((c1[0] - mcx) ** 2 + (c2[0] - mcx) ** 2
           + (c1[1] - mcy) ** 2 + (c2[1] - mcy) ** 2)
    s_iso = (num / den) if den > 0 else sx
    iox, ioy = mpx - s_iso * mcx, mpy - s_iso * mcy
    resid = max(float(np.hypot(p1[0] - (c1[0] * s_iso + iox),
                               p1[1] - (c1[1] * s_iso + ioy))),
                float(np.hypot(p2[0] - (c2[0] * s_iso + iox),
                               p2[1] - (c2[1] * s_iso + ioy))))
    return {"ok": True, "why": "",
            "scale": float(sx), "scale_y": float(sy), "offset": [float(ox), float(oy)],
            "iso_scale": float(s_iso), "iso_offset": [float(iox), float(ioy)],
            "resid_px": resid,
            "axis_gap_pct": float(abs(sy - sx) / sx * 100.0)}

def calib_from_two_point(calib, r, inset=4):
    """把 `solve_two_point` 的结果写进一份标定 dict（**保留** mode/alpha/ref/…）。

    ⚠ 两种显示方式的写法**不一样**，这是这个函数唯一存在的理由：
      · fit ：`canvas = (面板 - offset) / scale` ⇒ 解出来的 `offset` 就是它，直接用 ✓；
      · crop：同一层关系，但 `offset` 在这套约定里**必须是 inset**（它是"模板从面板
        边缘往里缩了多少"，见 `locate_crop`/`panel_to_canvas`）⇒ 得把解出来的"面板系
        截距"换算成 `view`：`(面板 - inset)/scale + view = (面板 - o)/scale`
        ⇒ `view = (inset - o) / scale`（逐轴）✓。
    写错这一处的现象很隐蔽：读数**整体平移** inset 个底图像素（≈ 几十世界像素），
    看着"差不多对"（`t_two_point_save_roundtrip` 钉着这条）。

    顺带：`src` 标成 `"two_point"`（这份几何是两点解出来的，不是模板匹配也不是目测），
    **不带 `score`**（它没有匹配分 —— 留着上一次的分数只会让人以为这份是自动量出来的）。

    ⚠ 走 `drop_stale_world_offset`：**重新量几何时把遗留的「坐标系偏移」清掉**（锚点换成
      下沿之后它该是 (0, 0)；老的 (7, 33) 是重心口径的，带过去就是凭空偏 33 像素 ✗）。
    """
    out = drop_stale_world_offset(calib)
    sx, sy = float(r["scale"]), float(r.get("scale_y", r["scale"]))
    ox, oy = (float(v) for v in r["offset"])
    out = with_scales(out, sx, sy)
    if out.get("mode") == map_const.MODE_CROP:
        out["offset"] = [int(inset), int(inset)]
        out["view"] = [int(round((inset - ox) / sx)), int(round((inset - oy) / sy))]
    else:
        out["mode"] = out.get("mode") or map_const.MODE_FIT
        # ⚠ **别取整**（2026-09-27 改）：这张图上 1 个面板像素 = 7.9~8.6 世界像素 ⇒
        #   取整就是系统性偏最多半个像素。实测：解出来的 -0.452 / -10.507 被存成
        #   0 / -11 ⇒ 读数的 x 永远差 4 个世界像素、y 永远差 4 个，**看着像点没点准**
        #   （用户 2026-09-27 那个 (48,-222) 里就有这 4 个像素 ✗）。存 3 位小数就够
        #   （0.001 面板像素 = 0.009 世界像素，远小于任何读数误差）。
        out["offset"] = [round(ox, 3), round(oy, 3)]
        out["view"] = [0, 0]
    out["src"] = "two_point"
    out.pop("score", None)
    return out

def drop_stale_world_offset(calib):
    """**重新量几何**之前：把遗留的「坐标系偏移」清掉（下沿锚点之后它该是 (0, 0) ✓）。

    为什么必须清（2026-09-27 查出来的一个真 bug，33 个世界像素就这么来的）：
      · 2026-09-27 之前锚点是黄点**重心** ⇒ 那时量出来的 `world_offset` 是"重心 ↔ 原点"
        的补偿（105040303 里现存 `[7, 33]` ✓）；
      · 现在锚点是**下沿（脚底）**（见 `dot_feet`）⇒ 同一项的正常值就是 (0, 0)；
      · 而"保存新几何"那条路是 `out = dict(老的 calib)` 再改几个键（`calib_from_two_point` /
        `minimap_calib._on_save` 都是）⇒ **老值会被原样带过去** ✗ ⇒ 读完标定、算世界坐标时
        凭空偏 33 像素 —— 而 33 世界像素 ≈ 4 个面板像素，正好落在"看着像标定没量准"的量级 ✗。
    ⇒ 结论：任何一次**重新量几何**（双点标定 / 对着底图手工对齐 / 自动匹配）都把这一项归零；
      真要单独量它得走专门的采样办法（`world_offset_advice`，目前不在界面上 ✓）。
    """
    out = dict(calib or {})
    if out.get("world_offset"):
        out.pop("world_offset", None)
    return out

def world_offset_advice(pairs, calib, cur_offset, terrain):
    """「坐标系偏移」核对：**黄点采样**（面板坐标 + 那块像素站着的世界坐标）→ 建议值。

    ⚠ **必须是"面板坐标 + 世界坐标"，而且面板坐标要是黄点（玩家标记）的位置** ——
      这条是写第一版时实算过的，记在这儿免得以后有人再走一遍：
        · 「坐标系偏移」加在**世界坐标**那一层（`panel_to_canvas` 之后）；
        · 而"底图像素 ↔ 世界坐标"那层由**地图数据**自己说了算（`world_to_canvas` 的逆）；
        · 拿**平台角那种地标**（底图坐标）+ 它的世界坐标去算 ⇒ 两者都由地图数据定死
          ⇒ 结果**恒等于 0**，是个死结论 ✗（除非你是要查"点歪了没"，那是另一件事）。
      黄点不一样：它是"玩家原点 + 偏移"画出来的 ⇒ 才带得出偏移 ✓。

    算式（就是「坐标系偏移」那个 tooltip 里让人手动做的那件事）：

        偏移_i = 世界_i - panel_to_world(黄点面板_i, 标定, 地图数据)

    两次采样给两个**独立**估计：互相一致 ⇒ 就是真实偏移，和现在存的一比就知道
    "该改成多少"（`suggest` / `delta`）✓；互相差很多 ⇒ 采样那一下人不在你说的那个
    地方（或标定不对）⇒ 把 `spread`（差多少世界像素）摆出来让人自己判 ✓。

    ⚠ 本函数**不设**"差多少算不一致"的阈值 —— 那是拍脑袋的常数（用户规矩：加常数要
      先说明来历）。所以它把数全摆出来、由界面决定怎么说。
    ⚠ **只给建议，不写配置**（用户 2026-09-27 定的口径：核对但不自动改）。

    `pairs` = `[((面板 x, 面板 y), (世界 x, 世界 y)), …]`。
    返回 `{"ok", "why", "per_point": [(ox, oy), …], "suggest": (ox, oy) | None,
    "cur": (cx, cy) | None, "delta": (dx, dy) | None, "spread": float}`。
    """
    if terrain is None or getattr(terrain, "px_per_world", None) in (None, 0):
        return {"ok": False, "why": "这张图没有底图/换算数据（先「生成地形图」）",
                "per_point": [], "suggest": None, "cur": None, "delta": None,
                "spread": 0.0}
    pts = []
    for item in pairs or []:
        try:
            p, w = item
            p = [float(v) for v in p]
            w = [float(v) for v in w]
            if len(p) != 2 or len(w) != 2:
                raise ValueError("不是两个坐标")
        except (TypeError, ValueError, IndexError):
            continue
        wx, wy = panel_to_world(p[0], p[1], calib, terrain)
        pts.append((w[0] - wx, w[1] - wy))
    if not pts:
        return {"ok": False, "why": "没填黄点的位置和它站着的世界坐标（或格式不对）"
                                    "⇒ 没有可核对的", "per_point": [], "suggest": None,
                "cur": None, "delta": None, "spread": 0.0}
    spread = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            spread = max(spread, float(np.hypot(pts[i][0] - pts[j][0],
                                                pts[i][1] - pts[j][1])))
    suggest = (int(round(sum(p[0] for p in pts) / len(pts))),
               int(round(sum(p[1] for p in pts) / len(pts))))   # 设置里存整数 ✓
    cur = delta = None
    try:
        if cur_offset is not None and len(list(cur_offset)) == 2:
            cur = (int(cur_offset[0]), int(cur_offset[1]))
            delta = (suggest[0] - cur[0], suggest[1] - cur[1])
    except (TypeError, ValueError):
        cur = delta = None
    return {"ok": True, "why": "", "per_point": pts, "suggest": suggest,
            "cur": cur, "delta": delta, "spread": spread}
