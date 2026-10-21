"""小地图定位（寻路用）：收 A 机单独推来的小地图流 → 换算成世界坐标。

数据链路（见 `docs/寻路设计.md`）：

    A 机 tools/minimap_push.py   小地图矩形截图 → JPEG → TCP 长度帧
    ↓
    MiniMapClient（本模块）      收到 → 只留**最新一帧**（和主画面同一个原则：
                                 宁可丢帧、绝不排队）
    ↓
    locate_canvas()              在流画面里找 `datasets/map/<id>.png`（底图），
                                 定出「面板怎么显示底图」：缩放 s + 偏移 (ox, oy)
    ↓
    panel_to_canvas()            流像素 → 底图像素 → （mapdata）世界坐标

**为什么单独一路**：主画面是 H.264 压过的，小地图上的黄点只有 2~5 像素，
压完就糊了；这一路是原始像素（JPEG q=100），黄点位置几乎无损。

**把 s/偏移量出来**（标定）在工作台里做：「路线识别 → 小地图定位 → 标定…」
（gui/minimap_calib.py）—— 那边看得见面板和底图重合不重合，这是唯一的判据。
本模块的命令行是同一条路的等价做法（留给排查/脚本用）：
    python -m perception.minimap --map 105090600

**显示方式（全局 / 局部）由用户人工选**（工作台「路线识别 → 小地图定位」，
每张图一份，存在 `datasets/map/<id>.mapcalib.json` 的 `mode`）——
**不按匹配分自动挑**：那两种画法在不同图上确实不一样（要进游戏走两步看地形
动不动才知道），分数高低说明不了该用哪种。这里的工具一律**沿用你选的那个**
（`--mode saved`，默认），分数只用来判断"这一次定位可不可信"。
"""

import socket
import struct
import sys
import threading
import time
import weakref
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core import mapdata, zones
from core.imgio import imread, imwrite
from perception import map_calib
from perception import map_const
from perception.map_client import (MiniMapClient,  # noqa: F401
                                   stop_all_clients)   # 支持中文路径（cv2.imread 遇中文静默失败）

ROOT = Path(__file__).resolve().parent.parent


# ⭐ 注册成**解释器退出前**的动作（2026-10-17 ✓ 实测量到过这个崩 ✓）：
#   自检套件（`selftest_main_window` 等）建了面板就会惰性起收流线程 ✓，而它们
#   **不走 `player_panel.shutdown`** ✗ ⇒ 解释器直接拆 socket/缓冲 ⇒
#   `0xC0000005`（access violation）✓ —— 表现是"**16/16 全过，进程却非 0 退出**"✓
#   （现场排查时很容易被当成"套件挂了" ✗）。
#   ⇒ 出口统一收一遍 ✓ 生产路径（关窗）多收一次也无害（幂等 ✓）。


def is_blackout(frame, ratio=0.98, max_val=24):
    """这一帧是不是**几乎全黑**（「进传送门 → 黑屏」那一瞬）→ bool。

    ⭐ **切图的判据**（用户 2026-10-01 定）：「进传送门 → 黑屏 → 亮起」算进了新图，
      由 **B 机**认这个瞬间，再把新图的 id 发给 A 机（`MiniMapClient.switch_map`）
      ⇒ 推流内容自动换到新图那块区域。

      ⚠ 这里**只判"黑不黑"**。什么时候算"亮起"、新图是哪张，得由**编排层**决定：
      「黑 → 亮」是个**过程**，两次采样之间到底算不算换完了，只有知道目标图的
      那一层说得清 —— 在这一层猜，就会把"半黑"判成"已经进图"✗。

    `ratio`：多大比例的像素算黑，就算黑屏（默认 98% —— 面板边框残留几个亮点不算
      "已经亮起"，别把"还剩 1% 有内容"当成已经进图 ✗）。
    `max_val`：多暗算黑（0~255，默认 24 —— 给 JPEG 噪点和暗部 UI 留点余量）。

    ⚠ **空帧 / 取不到帧 ⇒ False**：没有证据就说"进黑屏了"是猜 —— 那会把
      "还没收到帧"（断流的症状 ✗）误判成"正在换图"，正好把人往错方向带。
    """
    try:
        arr = np.asarray(frame)
    except Exception:                                      # noqa: BLE001
        return False
    if getattr(arr, "ndim", 0) < 2 or getattr(arr, "size", 0) == 0:
        return False
    # ⭐⭐⭐ **抽样判，别整幅算**（用户 2026-10-06 ✓ 帧率现场 ✓ `py-spy` 采样：小地图那条收帧
    #   线程里整幅浮点运算占了大头 ✓）：老写法先 `arr.mean(axis=2)`（**整幅 float 数组** ✗
    #   1164×585×3 ⇒ 分配 2 MB、再逐元素比较 ✓）⇒ 每帧几毫秒 ✗。
    #   ⚠ **判据本身一个字没变**（还是"亮度 ≤ `max_val` 的像素占比 ≥ `ratio`"✓）：
    #     黑屏/亮起是**整幅统计**（默认 98% 几乎全黑 ✓）⇒ 每隔 `_BLACKOUT_STEP` 个像素取一个 ✓
    #     得到的占比与全画面在统计上等价（样本仍有几万个 ✓ 比例误差 ≪ 那 2% 的余量 ✓）
    #     —— 这不是"改判据"，是**换一种等价的采样算法** ✓。
    #   ⚠ 步长取 4：样本数仍 ≥ (1164/4)×(585/4) ≈ 4.3 万 ✓（面板更小也够 ✓）。
    if arr.ndim == 3:
        arr = arr[::_BLACKOUT_STEP, ::_BLACKOUT_STEP].mean(axis=2)
    else:
        arr = arr[::_BLACKOUT_STEP, ::_BLACKOUT_STEP]
    try:
        return bool((arr <= float(max_val)).mean() >= float(ratio))
    except Exception:                                      # noqa: BLE001
        return False


#: `is_blackout` 的**抽样步长**（见它说明 ✓）：整幅逐像素 → 每 4 个取一个 ⇒ 省 ~16 倍算力 ✓。
_BLACKOUT_STEP = 4


#: `(map_id, src)` → `{"tries", "at", "why", "score"}`；**只活在本进程** ✓
#: （重启就重来 —— 它只用来"别在坏面板上死磕" ✓ 不是要记账给谁看 ✓）。
_AUTOFIT_STATE = {}


def autofit_reset(map_id=None, src=None):
    """清掉自动补标定的尝试计数（用例隔离用 ✓；`map_id/src` 都不给 ⇒ 全清 ✓）。"""
    if map_id is None and src is None:
        _AUTOFIT_STATE.clear()
        return
    _AUTOFIT_STATE.pop((str(map_id or ""), str(src or "")), None)


def any_mode_of(map_id):
    """这张图**任何一条来源**记过的显示方式（`fit` / `crop`）；一条都没有 → `None` ✓。

    ⚠⚠ **为什么"自动补标定"必须先问它** ✗：显示方式（全局 / 局部）是**人工选的** ——
      模块头那段写着"**不按匹配分自动挑**"✓（两种画法在不同图上确实不一样，分数高低
      说明不了该用哪种 ✓）。⇒ 自动这条路**只许沿用**已选过的那个 ✓，**不许**自己挑 ✗。
    """
    try:
        for cal in (mapdata.load_calibs(map_id) or {}).values():
            m = (cal or {}).get("mode")
            if m in (map_const.MODE_FIT, map_const.MODE_CROP):
                return m
    except Exception:                                # noqa: BLE001 —— 读不出来当没有 ✓
        return None
    return None


def try_autofit(panel, canvas, map_id, src, min_score=map_const.LOCATE_MIN_SCORE, seed=None):
    """给「**这张图 + 这条来源**」自动补一份标定 —— 成功 ⇒ **已经写盘** ✓。

    用户 2026-10-11 ✓ 原话："**为什么要我补标定？标定的数据不是都存好了？**"✓
      —— 存好的那份是**另一条来源**的 ✗：标定记的是「面板像素 ↔ 底图像素」的换算 ✓，
      而两条来源的面板**几何完全不同**（实测同一张图：独立推流 1164×585、从实时画面
      240×79 ✓ 差 5.6 倍 ✓，`panel_mismatch` 会直接判"对不上"✓）⇒ **不能互换** ✓。
      而"自动量一遍"这条路**是通的** ✓（实测黄金海滩：`locate_fit` 一次就给
      `score 0.9388 / scale 5.625` ✓ 与手工那份基本一致 ✓）⇒ 缺哪条就补哪条 ✓。

    返回 `{"ok", "why", "score", "tries", "already"?}`（**绝不抛** ✓ 异常一律变成 `ok=False` ✓）。
    ⚠ 调用方应**先**确认这条来源还没有几何 ✓（这里**不覆盖**已有的 ✗ —— 有的话直接回
      `already=True`，也不消耗尝试次数 ✓）。
    ⚠ **只沿用**已选过的显示方式 ✓（见 `any_mode_of` ✓）；一条都没有 ⇒ 拒绝并说清原因 ✓。
    """
    key = (str(map_id or ""), str(src or ""))
    try:
        if map_calib.has_geometry(mapdata.load_calib(map_id, src) or {}):
            return {"ok": True, "already": True, "tries": 0, "score": None,
                    "why": "这条来源已经有标定了 ✓（不用补）"}
    except Exception:                                # noqa: BLE001
        pass
    st = _AUTOFIT_STATE.setdefault(key, {"tries": 0, "at": 0.0, "why": "", "score": None})
    now = time.monotonic()
    if st["tries"] >= map_const.AUTOFIT_MAX_TRIES:
        return {"ok": False, "tries": st["tries"], "score": st.get("score"),
                "why": "已经自动试过 %d 次都量不出来 ⇒ 得人工标一次（或换来源）" % st["tries"]}
    if st["tries"] and (now - float(st["at"])) < map_const.AUTOFIT_COOLDOWN_S:
        return {"ok": False, "tries": st["tries"], "score": st.get("score"),
                "why": st["why"] or "刚试过，等一会儿再试"}
    st["tries"] += 1
    st["at"] = now
    try:
        mode = any_mode_of(map_id)
        if not mode:
            st["why"] = ("这张图还没选过「显示方式」（全局 / 局部）⇒ 自动**不敢替你挑** ✗"
                         "（那两种画法得分说明不了该用哪种 ✓）")
            return {"ok": False, "tries": st["tries"], "why": st["why"], "score": None}
        if panel is None or canvas is None or getattr(canvas, "size", 0) == 0:
            st["why"] = "没有面板画面 / 这张图还没有底图"
            return {"ok": False, "tries": st["tries"], "why": st["why"], "score": None}
        if mode == map_const.MODE_CROP:
            got = locate_crop(panel, canvas, min_score=min_score, seed=seed)
        else:
            got = locate_fit(panel, canvas, min_score=min_score)
        if not got:
            st["why"] = ("自动量了一次：这块面板和底图对不上（低于 %.2f）⇒ 得人工标一次"
                         % float(min_score))
            return {"ok": False, "tries": st["tries"], "why": st["why"], "score": None}
        cal = dict(got)
        cal.update({"mode": mode, "ref": "canvas",
                    "panel": [int(panel.shape[1]), int(panel.shape[0])],
                    "src": "auto", "picked_by": "auto",
                    "note": "自动补标定（未人工确认 ✓ 用户 2026-10-11）"})
        mapdata.save_calib(map_id, cal, src=src)
        st["why"] = ""
        st["score"] = float(cal.get("score") or 0.0)
        return {"ok": True, "tries": st["tries"], "score": st["score"],
                "why": "自动补标定成功（分 %.3f）" % st["score"]}
    except Exception as e:                           # noqa: BLE001 —— 自动这条路出事绝不许炸回路 ✗
        st["why"] = "自动补标定出错：%s: %s" % (type(e).__name__, e)
        return {"ok": False, "tries": st["tries"], "why": st["why"], "score": None}


def _pad_gray(c, frame, ratio=0.25, least=12):
    """底图四周补上"面板底色" → `(补边后的底图, (padx, pady))`。

    **为什么需要**（2026-10-18 实测 ✓）：局部小地图的面板常常**比底图宽**
    （105040303：底图 82×218、面板 110×112、缩放 1.0）⇒ 面板里"底图外面那圈没有地图的
    底色"（实测灰 131）在**原始底图**上**没有任何对应物** ⇒ 拿它当模板/拿它算贴合分，
    分数只能是噪声（现场 0.24、`mmap_pick` 7997 次全失败 ✗）。补上同色边之后同一帧能到
    **0.6~0.7** ✓。

    ⚠ 口径只有这一处：`CropViewTracker._pad_canvas` 的**运行时**跟踪和 `locate_crop` 的
      **一键标定**都走它 ✓（两处各写一份迟早分叉 ✗）。
    ⚠ 回到原坐标系的减法由调用方自己做（`view - pad`）—— 别在这里偷偷改 `view` ✗。
    """
    ph, pw = frame.shape[:2]
    ch, cw = c.shape[:2]
    m = max(int(least), int(round(max(pw, ph) * float(ratio))))
    if m <= 0:
        return c, (0, 0)
    try:
        bg = int(np.median([frame[1, 1], frame[1, -2], frame[-2, 1], frame[-2, -2]]))
    except (IndexError, ValueError):
        bg = 0
    out = np.full((ch + 2 * m, cw + 2 * m), bg, np.uint8)
    out[m:m + ch, m:m + cw] = c
    return out, (int(m), int(m))


def _blank_outside_ratio(frame, canvas_wh, calib, tol=18):
    """面板里那圈**底色**有没有落在底图**之外** → 0~1（正确的几何下必然如此 ✓）。

    **为什么需要它**（2026-10-18 实测 ✓）：`overlap_pearson`（判据那行那把尺）在这张图上会
    **被骗** —— 面板 88% 是暗底，把"放大 4.09 倍、只看得到 27×27 底图标"的错几何也算到
    **0.770**（比真几何 1.0 的 0.65 还高 ✗）。而几何上 4.09 是不可能的：面板里明明画着
    3 条平台，4.09 倍只对应 27 底图像素 ≈ 0.6 条 ✗。

    判据很朴素、也很硬：局部小地图的面板里，**"底图外面那圈底色"是真的在底图外面** ✓
    ⇒ 正确几何下这些像素映到 canvas 坐标应当**越界**；错的放大倍数会把它们映进底图内部 ✗。
    """
    f = _gray(frame)
    try:
        sx, sy = map_calib.scales_of(calib)
    except (TypeError, ValueError):
        return 1.0
    if sx <= 0 or sy <= 0:
        return 1.0
    try:
        bg = int(np.median([f[1, 1], f[1, -2], f[-2, 1], f[-2, -2]]))
    except (IndexError, ValueError):
        return 1.0
    blank = np.abs(f.astype(np.int16) - bg) <= int(tol)
    if float(blank.mean()) < 0.05:
        return 1.0          # 面板里没有明显底色边圈 ⇒ 这条不适用（别误杀 ✓）
    ch, cw = canvas_wh
    ox, oy = (calib.get("offset") or (0, 0))
    vx, vy = (calib.get("view") or (0, 0))
    h, w = f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (xx - float(ox)) / sx + float(vx)
    cy = (yy - float(oy)) / sy + float(vy)
    outside = (cx < 0) | (cx >= cw) | (cy < 0) | (cy >= ch)
    return float(outside[blank].mean())


def _crop_scales(canvas_wh, panel_wh):
    """crop 的候选放大倍数：整数为主（像素画放大就是复制像素），再补半档。"""
    cw, ch = canvas_wh
    pw, ph = panel_wh
    cand = set(float(v) for v in range(1, 13))
    cand |= {x * 0.5 for x in range(2, 25)}
    if cw > 0 and ch > 0:
        # 「面板刚好装下底图」的那个比例：客户端常在它附近取值
        cand.add(round(min(pw / cw, ph / ch), 3))
    return sorted(c for c in cand if 0.5 <= c <= 12.0)


def overlay_alpha_pct(calib):
    """标定里的叠图浓淡（0~100）。**0 就是 0** ✓。

    ⚠ 2026-09-26 修：原来是 `int(cal.get("alpha") or 55)` ⇒ **0 是 falsy** ⇒ 滑块拉到
    最左端（0 = 完全看不见）会被 `or` 换成 55 ✗（用户报"拖动条在最左端时透明度是
    55% 而不是 0%"）。判"有没有这个键"只能用 `is None`，不能用 `or` —— 这个值**特意
    要 0**，而 `or` 恰恰把 0 当"没填"。
    """
    v = (calib or {}).get("alpha")
    if v is None:
        return map_const.DEFAULT_ALPHA_PCT
    try:
        return max(0, min(100, int(round(float(v)))))
    except (TypeError, ValueError):
        return map_const.DEFAULT_ALPHA_PCT


def overlay_alpha(calib):
    """同上，但给 Qt / 绘制用的 0.0~1.0 透明度（`setOpacity` 那种）。"""
    return overlay_alpha_pct(calib) / 100.0


def live_src(cfg=None):
    """`config/live.yaml` 里的 `mmap_src`（非法或缺省 ⇒ 收流）。

    **一份口径**：路线识别面板和设置窗口（浓淡那一项按来源存 ✓）都从这儿取，
    不然两边各判一次"哪个来源合法"迟早分叉。
    """
    v = (cfg or {}).get("mmap_src")
    return v if v in (map_const.SRC_STREAM, map_const.SRC_LIVE) else map_const.SRC_STREAM


def source_for_map(map_id, live_cfg=None):
    """**这一张图**现在用哪条小地图来源 —— 全仓库唯一口径（体检 / 路线识别页都用它 ✓）。

    ⚠⚠ 为什么单独立一个（2026-10-10 ✓ 用户现场："**为什么我标定过了还显示这个**"✗）：
      · 来源 **2026-10-03 起按地图 id 存**（用户要求"路线识别页签所有配置按地图 id 存"✓
        见 `core/route_cfg.py` ✓），只在**没灌过**这张图时才回退全局 `config/live.yaml` ✓；
      · 而"开自动前的检查"（`gui/player_panel._auto_precheck_problems`）当时读的是
        **全局** `mm.live_src()` ✗ ⇒ 与路线识别页**不是同一处** ✗：
        现场这张图按图存的是 `live`（「从实时画面」✓ 标定就在这条下 ✓），
        全局那份是 `stream`（「独立推流」✗）⇒ 体检去问 `stream` ⇒ 报"还没标定" ✗，
        而人明明刚标过 ✓（**标定是按来源分开存的** ✓ 见 `core/mapdata.calib_path` ✓）。
      ⇒ 口径收在这一处：谁要问"这张图用哪条来源"，都走它 ✓。

    ⚠ `live_cfg` 不给 ⇒ 自己读 `config/live.yaml` ✓（调用方已经读过就传进来，省一次 IO ✓）。
    """
    try:
        from core import route_cfg
        v = (route_cfg.load(str(map_id or "")) or {}).get("mmap_src")
    except Exception:                            # noqa: BLE001 —— 读不到就回退全局 ✓
        v = None
    if v in (map_const.SRC_STREAM, map_const.SRC_LIVE):
        return v
    if live_cfg is None:
        try:
            from core.config import load_live
            live_cfg = load_live()
        except Exception:                        # noqa: BLE001
            live_cfg = {}
    return live_src(live_cfg)


def crop_of(project, cfg=None):
    """小地图面板在**实时画面**里的位置 `[x, y, w, h]`；取不到给 None。

    **按项目（地图）存** —— `projects/<名>/project.yaml` 的 `mmap_crop`。用户 2026-09-27
    现场：不同地图的小地图面板**尺寸/位置完全不同**，而这个值原来只有全局一份
    （`config/live.yaml`）⇒ 换个项目还是上一张图的框 ⇒ 叠图/定位全错 ✗。所以它和
    「寻路编辑器 / 集合 / 地形图」是同一类东西（跟项目走 ✓），入口也在那一页
    （「路线识别 → 寻路配置 → 框选小地图」，就在「寻路编辑器」下面 ✓）。

    ⚠ **兜底**：本项目**还没框过**时读老的那份全局值（`cfg` 里的 `mmap_crop`，那是
      2026-09-27 之前唯一的一份）—— 升级后不用重框就能接着用；**按本项目框过一次之后
      它就不再起作用** ✓。这是"读"的便利，**不是第二处存储**：谁都不许再往
      `live.yaml` 写它 ✗（那正是"换项目就错"的来源）。
    ⚠ `project` 传 `gui.project.Project`（或任何有 `.get()` 的映射）；没打开项目就传 None
      ⇒ 只剩兜底那条路 ✓。
    ⚠ **一条口径只有这一处**：路线识别面板（`_mmap_crop`）、命令行取证工具都用它，
      别再各写一份"项目优先还是全局优先" ✗。
    """
    for src in (project, cfg):
        if src is None:
            continue
        r = src.get("mmap_crop")
        if isinstance(r, (list, tuple)) and len(r) == 4:
            return [int(v) for v in r]
    return None


#: ⭐⭐⭐ 「差这么点分算**同一档**」的容差（`_nearest_peak` 用 ✓）—— 2026-10-20 现场定的。
#:   实测（同一张图、同一份数据，两处独立记过）：
#:     · 平台等间距重复 ⇒ 相邻峰间距 **12~13 个底图像素**、**高差只有 0.005** ✓；
#:     · 换成"平台重复"那一档更宽的局面，峰间高差能到 **0.04**（0.84~0.88 ✓ 见
#:       `_match` 的说明）。
#:   ⇒ 原来的 **0.02 比 0.04 还窄** ✗：真峰只要低了 0.03 就被踢出"同一档"，
#:     `_nearest_peak` 于是拿**全局最大那个**（隔壁那一层）⇒ `view` 在两层之间来回跳 ✓
#:     —— 用户 2026-10-20 原话："**地形图里的小地图匹配不断从三楼往底层闪烁**"✗
#:     （这张图 1 底图像素 ≈ 15 世界像素 ⇒ 13 px ≈ 190 世界像素 = 一层 ✓）。
#:   ⚠ 但**不能只把 tie 调大**（我第一版就是想这么改 ✗）：老的实现是"阈值 + 连通域质心"，
#:     阈值一放宽整片区域连成一块 ⇒ 质心成了**整张图的中心** ⇒ 位置更离谱 ✗✗。
#:     所以同一版里把峰提取换成**局部极大**（见 `_nearest_peak` ✓），两个一起改才对 ✓。
PEAK_TIE = 0.05


def _nearest_peak(res, best, prior, tie=None):
    """响应图里所有"接近最高分"的**峰**里，取**离先验最近**的那个 → `(x, y) | None`。

    **为什么需要**（2026-10-18 真机实测 ✓）：小地图上的平台是**等间距重复**的 ⇒ 响应图上
    每根平台一个峰、分数几乎一样（0.84~0.88）⇒ 直接取全局最大会在平台之间**跳**
    （人没动，`view_y` 却在 18 / 48 / 77 之间跳，正好差**一根平台** ✗）。
    先验 = 上一拍那个位置 / 按当前几何该在的位置 ⇒ 连续性天然把"跳一根"挡掉 ✓。

    ⚠⚠ **2026-10-20 改了两件事**（"三楼↔底层闪烁"那条 ✓ 见 `PEAK_TIE` 的说明）：
      ① **先取局部极大、再按分差筛**（老的是"阈值 + 连通域质心" ✗ —— 阈值一放宽，
         `res >= best - tie` 这一片会**连成一块**，质心跑到整张图中间 ⇒ 位置更离谱 ✗）；
      ② `tie` 的默认值走 `PEAK_TIE`（0.05 ✓ 盖得住实测 0.04 的峰间高差）。
    这样"同一档的峰"永远是**真的峰**（局部极大）✓，连续性才挑得对 ✓。
    """
    if res is None or getattr(res, "size", 0) == 0:
        return None
    _tie = float(PEAK_TIE if tie is None else tie)
    try:
        # 局部极大：等于"3×3 邻域最大值"且不低于 best - tie
        _k = np.ones((3, 3), np.uint8)
        _mx = cv2.dilate(res, _k)
        _loc = (res >= (_mx - 1e-9)) & (res >= float(best) - _tie)
        ys, xs = np.nonzero(_loc)
    except Exception:                            # noqa: BLE001 —— 算不出来就退回全局最大 ✓
        return None
    if len(xs) == 0:
        return None
    _px, _py = float(prior[0]), float(prior[1])
    _d = (xs.astype(np.float64) - _px) ** 2 + (ys.astype(np.float64) - _py) ** 2
    _i = int(np.argmin(_d))
    return (float(xs[_i]), float(ys[_i]))


def _subpixel_peak(res, ml):
    """`matchTemplate` 结果图上的峰值 → **亚像素**位置 `(fx, fy)`。

    **为什么不能只用 `minMaxLoc`**（2026-09-27）：它给的是**整数**像素，而这张图上
    1 个面板像素 = 8.5 世界像素 ⇒ 取整等于白送最多半个像素的系统偏差（4 个世界像素
    —— 现场"看着像点没点准"那个量级 ✗）。做法是标准的三点抛物线插值（x / y 各一次），
    只在峰值邻域 ±1 里做；邻域越界（贴边）就退回整数 ✓。
    插出来的偏移**夹在 ±0.5 内**：噪声大时抛物线会外推到离谱的位置 ✗。
    """
    mx, my = int(ml[0]), int(ml[1])
    h, w = res.shape[:2]
    fx, fy = float(mx), float(my)

    def _one(a, b, c):
        """三点 (a, b, c) 中间那个是最大 ⇒ 峰相对中间偏多少（±0.5 内）。"""
        den = a - 2.0 * b + c
        if abs(den) < 1e-12:
            return 0.0
        return max(-0.5, min(0.5, 0.5 * (a - c) / den))

    if 0 < mx < w - 1:
        fx = mx + _one(float(res[my, mx - 1]), float(res[my, mx]),
                       float(res[my, mx + 1]))
    if 0 < my < h - 1:
        fy = my + _one(float(res[my - 1, mx]), float(res[my, mx]),
                       float(res[my + 1, mx]))
    return fx, fy


def _fit_at(fg, cg, s):
    """按缩放 `s` 把底图缩到面板大小匹配一次 → `(分数, 峰值, 结果图)`；放不下 ⇒ None。"""
    if not s or s <= 0:
        return None
    tw, th = int(round(cg.shape[1] * s)), int(round(cg.shape[0] * s))
    if tw < 8 or th < 8 or tw > fg.shape[1] or th > fg.shape[0]:
        return None
    t = cv2.resize(cg, (tw, th), interpolation=cv2.INTER_AREA)
    r = np.nan_to_num(cv2.matchTemplate(fg, t, cv2.TM_CCOEFF_NORMED))
    _, mx, _, ml = cv2.minMaxLoc(r)
    return float(mx), ml, r


def refine_fit(fg, cg, best, rounds=6, span0=0.25, steps=7):
    """把 `locate_fit` 的**粗**结果细化：连续尺度 + 亚像素偏移（2026-09-27 新增）。

    **为什么非有不可**（"把全局小地图弄准"的第一步就是先把尺子弄准）：
      · 粗搜的尺度是**离散**的（`1.00~6.00` 每 0.25 一档，往上每 0.5 一档）；
      · 峰值是**整数**像素（`minMaxLoc`）。
      合成面板实测：真值 `scale=1.874` 时粗搜给 1.933（差 3.1%）⇒ **最远角差 11 个面板
      像素 ≈ 93 世界像素**。拿这种尺子去说"标定偏了多少"，尺子自己就是噪声 ✗。
    做法：先把偏移换成同一档的**亚像素**结果；再逐轮以**上一轮最优为中心**均匀取
    `steps` 档（`span` 每轮减半），取分最高的那档 —— **分数是单调判据，改进不了就停**
    （不然细化可能越弄越差 ✗）。**代价**：约 `rounds × steps` 次匹配（默认 6×7 ≈ 40 次，
    几十毫秒，一次性按钮操作 ✓）。
    ⚠ **不许离开粗搜那一档的窗口**（`±span0/2` = 粗搜格子的一半）：分数面很平时
      （面板里混着游戏 UI / 底图压过 / 分辨率对不上）"只挑更高的分"会**一路漂走** ——
      实测踩过：从 1.25 漂到 0.84，两套几何最远角差 6600 世界像素，全是噪声 ✗。
      窗口 = "真值确实在粗搜那一档附近"这个前提的边界 ✓。
    ⚠ **粗搜分数低也要细化**（`locate_fit` 里细化在门槛之前判 ✓）：真值落在两档中间时，
      粗搜每一档都差 6% 以上 ⇒ 分数能低到 0.27，但细化到 0.03 的格子就接近满分 ✓。
    """
    if best is None:
        return best
    steps = max(3, int(steps))
    base_s = float(best["scale"])          # 粗搜那一档（窗口的中心）
    lo, hi = base_s - span0 / 2.0, base_s + span0 / 2.0
    b_score = float(best["score"])
    b_off = [float(v) for v in best["offset"]]
    s0 = base_s

    def _eval(s):
        got = _fit_at(fg, cg, s)
        if got is None:
            return None
        sc, ml, r = got
        return sc, list(_subpixel_peak(r, ml))

    # ① 同一档，偏移换成亚像素（尺度不动也可能立刻改善 ✓）
    cur = _eval(s0)
    if cur is not None and cur[0] >= b_score:
        b_score, b_off = cur[0], cur[1]
    # ② 逐轮细化尺度：以上一轮最优为中心，但不许出窗口 ✓
    for rd in range(int(rounds)):
        span = span0 / (2.0 ** rd)
        center = min(max(s0, lo), hi)
        cand = None
        for i in range(steps):
            s = min(max(center + span * (2.0 * i / (steps - 1) - 1.0), lo), hi)
            got = _eval(s)
            if got is None:
                continue
            if cand is None or got[0] > cand[0]:
                cand = (got[0], s, got[1])
        if cand is None or cand[0] <= b_score + 1e-9:
            break                      # 这一轮没有改进 ⇒ 停（别越弄越差 ✗）
        b_score, s0, b_off = cand
    return {"mode": best.get("mode", map_const.MODE_FIT), "scale": float(s0),
            "offset": [float(v) for v in b_off], "score": float(b_score),
            "view": list(best.get("view") or [0, 0]), "refined": True}


def locate_fit(frame, canvas, scales=None, min_score=map_const.LOCATE_MIN_SCORE, refine=True):
    """方式 1：整张底图缩放到面板 → 标定 dict 或 None。

    ⚠ 默认**细化**（`refine_fit`，2026-09-27 加）：粗搜是离散尺度 + 整数峰值，不够当
      一把"量精度的尺子"（见 `refine_fit` 的说明）。要老行为（纯粗搜、整数偏移）
      就传 `refine=False` ✓（`locate_crop` 那份暂时仍是粗搜 —— crop 的自动标定
      用户明确暂缓 ✓）。
    """
    fg = _gray(frame)
    cg = _gray(canvas)
    if scales is None:
        fh, fw = fg.shape[:2]
        ch, cw = cg.shape[:2]
        base = min(fw / float(cw), fh / float(ch))
        # 候选尺度要**够密**：客户端把小地图放大几倍是不确定的（2x / 3x / 正好嵌进面板）。
        # 上限放到 12：小底图（几十像素）被客户端放大 6~10 倍是实测见过的，
        # 原来只到 5，那种图上的自动定位会一路失败（人手动拖也不够，见 locate_crop）。
        cand = [round(1.0 + 0.25 * i, 3) for i in range(21)]      # 1.00 ~ 6.00
        cand += [round(6.0 + 0.5 * i, 3) for i in range(13)]      # 6.0 ~ 12.0
        cand += [round(base, 3), round(base * 0.5, 3), round(base * 2.0, 3)]
        scales = sorted({c for c in cand if 0.25 <= c <= 12.0})

    best = None
    for s in scales:
        got = _fit_at(fg, cg, s)
        if got is None:
            continue
        mx, ml = got[0], got[1]
        if best is None or mx > best["score"]:
            best = {"mode": map_const.MODE_FIT, "scale": float(s),
                    "offset": [int(ml[0]), int(ml[1])], "score": float(mx),
                    "view": [0, 0]}
    if best is None:
        return None
    if refine:
        # ⚠ **先细化再判门槛**（2026-09-27）：粗搜的格子是 0.25 ⇒ 真值落在两档中间
        #   时**每一档都差 6% 以上**，粗搜的分数可以低到 0.27（合成面板实测：真值
        #   1.874、粗搜 1.75/2.0 都只有 0.2~0.3）⇒ 拿粗分数去卡 0.55 会直接判"量不出来" ✗。
        #   先细化到 0.0625 的格子 ⇒ 真值那一档分数接近 1.0 ✓，再拿它和市场门槛比才有意义。
        best = refine_fit(fg, cg, best)
    if best["score"] < min_score:
        return None
    return best


def _crop_blocks(f, inset, s, cw, ch):
    """按倍数 `s` 从面板上取**几块**当模板 → `[(面板上的左上角, 模板, 真正实现的倍数), ...]`。

    每块都缩回**底图尺度**（`cv2.resize(INTER_AREA)` ✓），且缩回后 ≤
    `CROP_TPL_FRAC`×底图（大块才有约束力、且要留得出可放的位置 ✓）。

    ⚠⚠ **为什么要好几块**（用户 2026-09-29："自动定位按钮 好像对于局部地图来说不好用"✗）：
    **到底哪一块落在底图里，事先不知道** —— 那正是要求的东西 ✗（实测用户那份：
    面板映到底图 x ∈ [−12.2, 76.7]，而底图只有 82 宽 ⇒ 面板右侧一大截在底图之外，
    只有**靠左**的那块才落得进去 ⇒ 只取"正中间那块"会**整块放不进底图**、
    匹配位置贴着边界 ✗ ⇒ 量出来的倍数是错的 8.4（真 5.645）✗✗）。
    ⇒ 面板在某个轴上比底图宽（或差不多）时，那个轴给**左 / 中 / 右**三个位置 ✓，
    轴向上装得下就只给中间一个 ✓（省时间：多数情况只有 1~3 块 ✓）。
    """
    h, w = f.shape[:2]
    ax0, ay0 = inset, inset
    ax1, ay1 = w - inset, h - inset
    if ax1 - ax0 < 8 or ay1 - ay0 < 8:
        ax0, ay0, ax1, ay1 = 0, 0, w, h
    if ax1 - ax0 < 8 or ay1 - ay0 < 8:
        return []
    foot_w = (ax1 - ax0) / float(s)          # 这个倍数下面板映到底图有多大
    foot_h = (ay1 - ay0) / float(s)
    # 那个轴上**装不下整个面板**（crop 的常态）⇒ 只能挪着试（哪块在底图里事先不知道 ✓）；
    # 装得下就只取中间一块（省时间 ✓）
    out = []
    for frac in ((map_const.CROP_TPL_FRAC, map_const.CROP_TPL_FRAC_SMALL)
                 if (foot_w >= cw * 0.98 or foot_h >= ch * 0.98)
                 else (map_const.CROP_TPL_FRAC,)):
        tw = int(min(foot_w, cw * frac))
        th = int(min(foot_h, ch * frac))
        if tw < map_const.CROP_TPL_MIN or th < map_const.CROP_TPL_MIN:
            continue
        if tw >= cw or th >= ch:        # 必须**严格小于**底图，否则只有一个位置可放
            continue
        wq = max(map_const.CROP_TPL_MIN, min(int(round(tw * s)), ax1 - ax0))
        hq = max(map_const.CROP_TPL_MIN, min(int(round(th * s)), ay1 - ay0))
        xs = [ax0 + ((ax1 - ax0) - wq) // 2]
        ys = [ay0 + ((ay1 - ay0) - hq) // 2]
        # ⚠ **面板**装不下那个轴 ⇒ 左 / 中 / 右都试：面板贴到底图外沿时，只有靠边那块
        #   落得进去（实测用户那份面板映到底图 x ∈ [−12.2, 76.7] ⇒ 只有靠左那块 ✓）
        if foot_w >= cw * 0.98:
            xs = sorted({ax0, ax0 + ((ax1 - ax0) - wq) // 2, ax1 - wq})
        if foot_h >= ch * 0.98:
            ys = sorted({ay0, ay0 + ((ay1 - ay0) - hq) // 2, ay1 - hq})
        for px0 in xs:
            for py0 in ys:
                if any(v[0] == (px0, py0) for v in out):
                    continue                     # 两档 frac 撞在一起 ⇒ 别重复算 ✓
                sub = f[py0:py0 + hq, px0:px0 + wq]
                if getattr(sub, "size", 0) == 0:
                    continue
                tpl = cv2.resize(sub, (max(1, int(round(wq / s))),
                                       max(1, int(round(hq / s)))),
                                 interpolation=cv2.INTER_AREA)
                if tpl.shape[0] < map_const.CROP_TPL_MIN or tpl.shape[1] < map_const.CROP_TPL_MIN:
                    continue
                if tpl.shape[0] >= ch or tpl.shape[1] >= cw:
                    continue
                # ⚠ `eff` 一律取**真正实现出来的倍数**（按实际输出尺寸算），不写"想要的那个 s" ——
                #   否则 view 是对的、scale 是错的（`t_crop_scale_not_faked` 钉着 ✓）
                out.append(((px0, py0), tpl, wq / float(tpl.shape[1])))
    return out


def _peak_subpix(r, x, y):
    """相关图峰值的**亚像素**位置（抛物线拟合）→ `(fx, fy)` 浮点。

    面板放大 5~10 倍很常见 ⇒ **1 个底图像素的量化误差 = 5~10 个面板像素** ✗
    （实测 105040303：一个底图像素 ≈ 7.4 个世界像素 ⇒ 那点误差直接进世界坐标 ✓）。
    峰值贴着边界（拟合不了）就返回整数 ✓。
    """
    out = [float(x), float(y)]
    h, w = r.shape[:2]
    if 0 < x < w - 1:
        l, m, rr = float(r[y, x - 1]), float(r[y, x]), float(r[y, x + 1])
        d = l - 2.0 * m + rr
        if abs(d) > 1e-9:
            out[0] = x + max(-0.5, min(0.5, 0.5 * (l - rr) / d))
    if 0 < y < h - 1:
        l, m, rr = float(r[y - 1, x]), float(r[y, x]), float(r[y + 1, x])
        d = l - 2.0 * m + rr
        if abs(d) > 1e-9:
            out[1] = y + max(-0.5, min(0.5, 0.5 * (l - rr) / d))
    return out[0], out[1]


def scroll_of(calib):
    """标定里写的滚动方向 → `"x" | "y" | "xy"`（没写 / 写错 ⇒ 双轴 = 老行为 ✓）。"""
    v = str((calib or {}).get("scroll") or map_const.SCROLL_XY).strip().lower()
    if v in ("x", "h", "horizontal", "仅x"):
        return map_const.SCROLL_X
    if v in ("y", "v", "vertical", "仅y"):
        return map_const.SCROLL_Y
    return map_const.SCROLL_XY


def scroll_axes(calib):
    """→ `(能动的轴 X, 能动的轴 Y)`（`(True, True)` = 双轴 ✓）。

    调用方一律用这一对布尔判"这根轴要不要搜 / 要不要动" ✓（别在别处再解析一遍字符串 ✗）。
    """
    s = scroll_of(calib)
    return (s in (map_const.SCROLL_XY, map_const.SCROLL_X), s in (map_const.SCROLL_XY, map_const.SCROLL_Y))


def _crop_calib_from(tpl_pos, sub_xy, eff, inset):
    """"模板落在底图哪里" → 标定 dict（`view` = 面板 (inset, inset) 对应的底图坐标 ✓）。

    ⚠ 口径照抄 `panel_to_canvas`：`canvas = (panel − offset)/scale + view`
      ⇒ 面板上 `sub_xy` 那个点映到 `tpl_pos` ⇒ `view = tpl_pos − (sub_xy − offset)/scale` ✓
      （整块模板时 `sub_xy == offset` ⇒ `view == tpl_pos`，和老行为**一字不差** ✓）。
    """
    fx, fy = tpl_pos
    vx = fx - (sub_xy[0] - inset) / eff
    vy = fy - (sub_xy[1] - inset) / eff
    return {"mode": map_const.MODE_CROP, "scale": float(eff), "offset": [inset, inset],
            "view": [float(vx), float(vy)]}


def _crop_score(frame, canvas, calib, max_samples=2000):
    """一套 crop 几何的贴合分（**只在重叠够多时才算数**）→ `(score|None, overlap)`。

    ⚠ 搜索时采样取 **2000**（不是显示那个 6000）：拟合要算几百次，2000 点足够分辨
      "哪套几何更好" ✓，反而快 ~2.5 倍 ✓；**给人看的那个分**仍走 `align_score` 的 6000 ✓
      （两者同一个函数、同一个几何族，只是采样密度不同 ⇒ 差别在小数点后第三位 ✓）。
    """
    sc, ov, _why = overlap_pearson(frame, canvas, calib, max_samples=max_samples)
    if sc is None or ov < 0.08:
        # 面板几乎全在底图外面 ⇒ 那几个采样点说明不了什么 ✗（容易碰巧高分）
        return None, ov
    return sc, ov


def _crop_pos_fit(frame, canvas, calib, steps=(8.0, 4.0, 2.0, 1.0, 0.5, 0.25),
                  axes=None):
    """**固定缩放**，只挪位移（每一步试 8 个方向，逐级细化到 0.25 底图像素）→ 最好的那份。

    ⚠ 为什么要"固定缩放"分开做（实测踩到）：位移和缩放**不是可分的** —— 缩放差 2%
      时，把它单独往真值挪是**下坡**（位移能补偿一部分）⇒ 一起坐标下降会**卡在这条山脊上**
      （实测：9.0/7.0 那份真几何贴合分 1.0000，可从 9.54/7.41 出发一起爬只能到
      9.25/7.41 的 0.9298 就动不了了 ✗）。固定缩放时位移那一维是**单峰**的 ⇒ 稳 ✓。

    ⚠⚠ **细档（≤1 像素）要用高采样**（6000），粗档（≥2）才用 2000：采样稀了分数面会
       "阶梯化"（实测：1:1 的合成面板在 2000 采样下，8.00 / 8.25 / 7.75 都是 1.0000、
       8.50 掉到 0.8606 ⇒ 爬坡最多只能保证 ±0.25，再叠加"只许严格变好"就**停在偏
       0.5 像素的地方** ✗ —— 那 0.5 底图像素在这张图上是 ~4 个世界像素 ✓）。
      细档的eval 只有几十次 ⇒ 多花十几毫秒，换回来的是"跟出来的显示区真的准" ✓。
    """
    ms = 6000 if max(steps) <= 1.0 else 2000
    # ⭐ **单轴的图只试那根轴**（"仅 X" ⇒ 不试 dy ✓）：那些 dy ≠ 0 的探针要么被算出来是
    #    下坡（白花），要么会把"根本不会动的那根轴"跟歪 ✗ ⇒ 2/3 的探针直接省掉 ✓
    ax_on, ay_on = (True, True) if axes is None else axes
    best_sc, _ov = _crop_score(frame, canvas, calib, max_samples=ms)
    if best_sc is None:
        return None
    best = {"calib": dict(calib), "score": float(best_sc)}
    for step in steps:
        moved = True
        while moved:
            moved = False
            base = best["calib"]
            v = list(base.get("view") or (0, 0))
            for dx in ((step, 0.0, -step) if ax_on else (0.0,)):
                for dy in ((step, 0.0, -step) if ay_on else (0.0,)):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    probe = dict(base, view=[v[0] + dx, v[1] + dy])
                    sc, _ov = _crop_score(frame, canvas, probe, max_samples=ms)
                    if sc is not None and sc > best["score"] + 1e-4:
                        best = {"calib": probe, "score": float(sc)}
                        moved = True
                        break
                if moved:
                    break
    return best


def _fit_crop(frame, canvas, calib, axes=None):
    """位移 + 缩放一起定：**缩放扫网格、每个缩放都把位移归位**，再收窄重来一遍 ✓。

    为什么这么绕（见 `_crop_pos_fit` 的说明）：两者不可分 ⇒ 只能"固定一个、优化另一个"
    交替 ✓。两轮足够（第一轮 ±8% 步 1%，第二轮在最好的附近 ±1% 步 0.12% ✓）。

    ⚠ 为什么不用模板匹配把这件事做完：它只能给**整像素**峰值，而面板放大 5 倍时
      **1 底图像素 = 5~10 个面板像素** ✗；贴合分是连续的 ⇒ 能定到 0.25 底图像素 ✓
      （而且它**就是判据那行显示的那个数** —— 把它做到最好，就是"自动定位"该干的事 ✓）。
    """
    fit = _crop_pos_fit(frame, canvas, calib, axes=axes)
    if fit is None:
        return None
    best = fit
    # 三段：先 ±20%（粗，位置只粗定）→ ±4% → ±0.8%（细，位置抠到 0.25 ✓）
    # ⚠ 第一段要**宽**：种子可能离真值 10%~20%（模板匹配在小块上给不准 ✓），
    #   而收窄之后就只能在本山头上找 ⇒ 太窄会把对的那份关在外面 ✗
    for span, step, steps in ((0.20, 0.04, (4.0, 2.0)),
                              (0.04, 0.008, (2.0, 1.0)),
                              (0.008, 0.0016, (1.0, 0.5, 0.25))):
        s0 = float(map_calib.scales_of(best["calib"])[0])
        if s0 <= 0:
            break
        n = max(2, int(round(span / step)))
        for k in range(-n, n + 1):
            if k == 0:
                continue
            cand = map_calib.with_scales(best["calib"], s0 * (1.0 + step * k),
                               s0 * (1.0 + step * k))
            got = _crop_pos_fit(frame, canvas, cand, steps=steps, axes=axes)
            if got is not None and got["score"] > best["score"] + 1e-4:
                best = got
    return best


def refine_crop(panel, canvas, calib, move_scale=True):
    """在**现有几何**附近抠一抠 ⇒ 更好的那一份（`{"calib", "score", "before", "gain"}`）。

    用途：手工对齐之后想"再抠一点"（用户 2026-09-29："我已经手工对的很好了"✓ —— 那就
    量一量**还能不能更好**：好多少、往哪边 ✓）。**只接受更好的** ⟹ 绝不会把对的改坏 ✓
    （所以它可以放心地做成一个按钮 ✓）。算不出来 ⇒ `None`。

    `move_scale=False` ⇒ **只挪位移、缩放一个字不动** —— 界面上那个「不改缩放」就是它 ✓
    （"我已经量好缩放、只想知道位置"那条路 ✓）。
    """
    cal = dict(calib or {})
    if cal.get("mode") != map_const.MODE_CROP or not map_calib.has_geometry(cal):
        return None
    if float(map_calib.scales_of(cal)[0]) <= 0:
        return None
    axes = scroll_axes(cal)          # ⭐ 单轴 ⇒ 只跟会动的那根轴（见 `scroll_axes` ✓）
    got = _fit_crop(panel, canvas, cal, axes=axes) if move_scale \
        else _crop_pos_fit(panel, canvas, cal, axes=axes)
    if not move_scale:
        # ⚠ 只挪位移时**光靠"手上那套"当起点不够**：它可能整块是错的（首次标定时就是）
        #   ⇒ 再把"就按这个缩放、用模板匹配粗定一遍"那份当第二号起点 ✓
        #   ⛔ 但**缩放必须锁死在调用方给的那个**：粗定那份的 `eff` 是"真正实现出来的
        #     倍数"（可能 5.652 ≠ 输入的 5.645）⇒ 直接用它就等于偷偷改了缩放 ✗✗
        #     （`t_crop_locate_real_shape` 里钉着这一条 ✓）。
        s_lock = float(map_calib.scales_of(cal)[0])
        alt = crop_seed_at(panel, canvas, s_lock,
                           int(cal.get("offset", [4, 4])[0]))
        if alt is not None:
            alt2 = _crop_pos_fit(panel, canvas,
                                 map_calib.with_scales(alt["calib"], s_lock, s_lock),
                                 axes=axes)
            if alt2 is not None and (got is None or alt2["score"] > got["score"]):
                got = {"calib": map_calib.with_scales(alt2["calib"], s_lock, s_lock),
                       "score": alt2["score"]}
    if got is None:
        return None
    before, _ov = _crop_score(panel, canvas, cal)
    before = 0.0 if before is None else float(before)
    return {"calib": got["calib"], "score": float(got["score"]), "before": before,
            "gain": float(got["score"] - before)}


def crop_seed_at(frame, canvas, s, inset=4):
    """**某个倍数**下用模板匹配粗定一份几何 → `{"calib", "score"}`（都不行 ⇒ `None`）。

    ⚠ 块取自**面板**（`_crop_blocks` ✓）、分数一律是**整张面板的贴合分**（`_crop_score`
    ✓）—— 于是"取哪块模板"只影响**能不能搜到**，不影响"谁胜出" ✓（全世界同一把尺 ✓）。
    `locate_crop`（撒种子）与 `refine_crop`（"不改缩放"那条路也用它）共用这一处 ✓。
    """
    f = _gray(frame) if getattr(frame, "ndim", 3) == 3 else frame
    c = _gray(canvas) if getattr(canvas, "ndim", 3) == 3 else canvas
    ch, cw = c.shape[:2]
    h, w = f.shape[:2]
    if not (s > 0):
        return None
    # 面板缩回底图后盖住多大：远超底图 ⇒ 几何上讲不通，不是这个倍数
    if w / float(s) > cw * map_const.CROP_FOOT_MAX or h / float(s) > ch * map_const.CROP_FOOT_MAX:
        return None
    out = None
    for sub_xy, tpl, eff in _crop_blocks(f, inset, s, cw, ch):
        r = np.nan_to_num(cv2.matchTemplate(c, tpl, cv2.TM_CCOEFF_NORMED))
        _mx, _y, _z, ml = cv2.minMaxLoc(r)
        cal = _crop_calib_from(_peak_subpix(r, ml[0], ml[1]), sub_xy, eff, inset)
        sc, _ov = _crop_score(frame, canvas, cal)
        if sc is None:
            continue
        if out is None or sc > out["score"]:
            out = {"calib": cal, "score": float(sc)}
    return out


def locate_crop(frame, canvas, inset=4, min_score=map_const.LOCATE_MIN_SCORE, scales=None,
                refine=True, seed=None, scroll=None):
    """方式 2：面板是底图的**一块**（可能先放大了若干倍）→ 标定 dict 或 None。

    **为什么不能只按 1:1 找**（实测踩到）：客户端把小地图放大后再切块很常见。
    猴林迷宫I 就是：底图 78×203，而面板 753×612 —— 光宽度就差 9.7 倍。这种图上
    「整张缩放进面板」（fit）和「1:1 取一块」**两种都量不出来**，人手动把叠加层
    拖到 4 倍也不够（缩放上限），表现就是「匹配分 0.7 上下、怎么看都不对」。

    **怎么做**（用户 2026-09-29 报"自动定位对局部地图不好用"✗ 之后重写的一版）：
      ① **撒种子**（`_crop_blocks` + `matchTemplate`）：对每个候选倍数，从面板上取
         **中间 / 左 / 右**几块缩回底图尺度、在底图里找它 ⇒ 一批**粗略**的 (倍数, 位置)；
         有 `seed`（**手上现有那套几何**）时它就是第一号种子 ✓ —— 手工对齐过的那份
         离答案最近，从它爬最省事 ✓；
      ② **按贴合分拟合**（`_crop_pos_fit` + `_fit_crop`）：沿着**判据那行显示的那个分**
         （位移 → ±0.25 底图像素、缩放 → ±0.2% ✓）。⚠ 模板匹配只能给**整像素**峰值，
         而面板放大 5 倍时 **1 底图像素 = 5~10 个面板像素** ⇒ 只靠它，几何天生差那么
         一截 ✗；贴合分是连续的 ⇒ 能爬到那一截里面 ✓；
      ③ 谁胜出**一律由贴合分说了算**（`overlap_pearson` = 判据那行**同一个函数** ✓）。

    ⚠⚠ 老版（2026-09-29 之前）**三条毛病凑成一件事**：自动定位对 crop 图**永远给不出
    正确的那份几何**（实测森林迷宫III / 105040303：给的是 scale=12.000，真值 5.645 ✗）：
      ① **缩放只有整数 + 半档**（`abs(s − round(s)) > 0.3 ⇒ 跳过`）⇒ 5.645 这种值
         **一个候选都够不着** ✗（而差 3% 就能让分数从 0.95 掉到 0.2 一带 ✓）；
      ② **整块模板放不进底图就跳过**（`tt.shape[1] > c.shape[1]: continue`）——
         而那份真倍数缩回来是 **87.5 宽 > 底图 82** ✗ ⇒ **唯一正确的倍数被扔掉** ✗✗；
      ③ **候选之间比分**：`TM_CCOEFF_NORMED` 在不同大小的模板之间**不可比**（实测
         42×34 的小模板碰巧 0.479 > 真倍数那块 0.2 一带）⇒ 一比就选那个离谱的 12 ✗。
      ⇒ 现在：**连续缩放 + 多块模板 + 统一口径排序 + 爬坡细化** ✓。

    于是标定里同时有 `scale`（面板像素 / 底图像素）和 `view`（面板左上角对应的
    底图坐标），换算仍是 `canvas = (panel - offset) / scale + view`。

    ⭐ `scroll`（`"x"|"y"|"xy"`，默认从 `seed` 里读 ✓）—— **单轴的图只跟会动的那根轴**
    （用户 2026-09-29：滚动类型"仅X/仅Y/双轴"，单轴图上实时匹配要快 ✓）：拟合少 2/3 的
    探针 ✓，而且**不会动的那根轴一个字都不改**（由标定定死 ✓）。第一次标这张图（没种子）
    时不带它 ⇒ 两轴都搜 ✓。
    """
    f = _gray(frame)
    c = _gray(canvas)
    h, w = f.shape[:2]
    ch, cw = c.shape[:2]
    if scales is None:
        scales = _crop_scales((cw, ch), (w, h))
    # ⭐⭐ **底图补边**（2026-10-18 ✓ 与运行时跟踪同一套，见 `_pad_gray`）：
    #   面板比底图宽时（本图 110 > 82），面板里"底图外面那圈底色"在原始底图上没有对应物 ⇒
    #   粗搜/贴合分全被它拖成噪声（现场：一键标定给 6.9/0.58、真值 1.0 ✗）。
    #   ⚠ 进去之后**所有坐标（含 `seed` 的 view、`ref`、返回值）都是补边空间的** ⇒
    #     只在**最后**减一次（一处进一处出 ✓）。
    c, (padx, pady) = _pad_gray(c, f)
    if seed is not None and seed.get("view"):
        _sv = seed.get("view") or (0.0, 0.0)
        seed = dict(seed, view=[float(_sv[0]) + padx, float(_sv[1]) + pady])
    # ⭐ **滚动方向**：手上这套几何（`seed`）或调用方显式给的都算数（用户 2026-09-29 ✓）。
    #   单轴 ⇒ 拟合只在会动的那根轴上走（少 2/3 的探针 ✓），而且**不会动的那根轴一个
    #   字都不改** ✓（它由标定定死；seed 给的 y 就是对的 ✓）。
    #   ⚠ 没有种子（第一次标这张图）时不带 `scroll` ⇒ 两根轴都得搜 ✓（那时还不知道该定死哪轴 ✗）。
    axes = scroll_axes(seed if seed is not None else {"scroll": scroll})

    # ⭐ **单轴图：不动的那根轴"一个字都不改"** —— 种子阶段就要钉住（`_fit_crop` 只保证
    #   "拟合时不往那根轴动"，可粗搜出来的那些种子**那根轴是随便的** ✗ ⇒ 它们一旦比分
    #   胜出，y（比如）就被悄悄改了 ✗✗。参考值 = 调用方给的 `seed` 里那个 ✓
    #   （没有 seed 就没有参考 —— 第一次标这张图时两轴都得搜 ✓）。
    ref = None
    if (seed and seed.get("mode") == map_const.MODE_CROP and map_calib.has_geometry(seed)
            and not (axes[0] and axes[1])):
        _v = seed.get("view") or [0.0, 0.0]
        ref = (float(_v[0]), float(_v[1]))

    def _pin(cal):
        """把"不动的那根轴"钉回 `ref` ✓（双轴 / 没参考 ⇒ 原样返回 ✓）。"""
        if ref is None:
            return cal
        v = list(cal.get("view") or (0.0, 0.0))
        if not axes[0]:
            v[0] = ref[0]
        if not axes[1]:
            v[1] = ref[1]
        return dict(cal, view=[float(v[0]), float(v[1])])

    seeds = []
    # ① 手上这套几何（有的话）**最先**：手工对着的那份离答案最近 ✓
    if seed and seed.get("mode") == map_const.MODE_CROP and float(map_calib.scales_of(seed)[0]) > 0:
        seeds.append(dict(seed))
        # 它附近也撒两个（缩放 ±3%、±6%）：手工缩放差一点点很常见 ✓
        s0 = float(map_calib.scales_of(seed)[0])
        for fct in (0.94, 0.97, 1.03, 1.06):
            seeds.append(map_calib.with_scales(seed, s0 * fct, s0 * fct))
    # ② 再靠粗搜：多块模板 + 贴合分排序 ✓（第一次标定这张图时只走这一路 ✓）
    got = []
    for s in scales:
        if s <= 0:
            continue
        ev = crop_seed_at(f, c, s, inset)
        if ev is not None:
            got.append(ev)
    got.sort(key=lambda v: -v["score"])
    seeds.extend(_pin(v["calib"]) for v in got[:6])
    if not seeds:
        return None
    # ③ 排名靠前的几号种子做**完整拟合**（位移 + 缩放分开定，见 `_fit_crop` ✓），
    #   其余的只按原样比分 —— 拟合一次 ~200 ms ⇒ 只给前两号，免得点一下等两秒 ✗
    ranked = []
    for cal in seeds:
        # ⚠ 用**补边后**的底图 `c`（种子里的 view 已在补边空间 ✓ 两者必须同一坐标系）
        sc, _ov = _crop_score(f, c, cal)
        if sc is None:
            continue
        # ⭐ 「面板底色该落在底图外面」这条闸（2026-10-18 ✓ 见 `_blank_outside_ratio`）：
        #   它要用**原始**底图尺寸 + **原始**坐标系 ⇒ 这里把 view 减回原空间再算
        _v = cal.get("view") or (0.0, 0.0)
        bl = _blank_outside_ratio(f, (ch, cw),
                                  dict(cal, view=[float(_v[0]) - padx,
                                                  float(_v[1]) - pady]))
        ranked.append({"calib": dict(cal), "score": float(sc), "blank": float(bl)})
    if not ranked:
        return None
    # ⭐⭐ **先过闸、再比分**：这套贴合分（`overlap_pearson`）在"面板大半是暗底"的图上会被骗
    #   —— 实测把"放大 4.09 倍、只看得到 27×27 底图"的错几何算到 **0.770**（比真值 1.0 的
    #   0.65 还高 ✗）。而"底色落在底图外面"是**几何事实**，错的倍数过不去 ✓
    pool = [v for v in ranked if v["blank"] >= 0.6] or ranked
    pool.sort(key=lambda v: -v["score"])
    best = pool[0]
    for seed_cal in ([v["calib"] for v in pool[:2]] if refine else []):
        got = _fit_crop(f, c, seed_cal, axes=axes)
        if got is not None and got["score"] > best["score"]:
            best = got
    if best is None or best["score"] < min_score:
        return None
    out = dict(best["calib"])
    # ⭐ **出补边空间**：`view` 减掉那圈边（进去时加、出来时减 —— 一处进一处出 ✓）
    _v = out.get("view") or (0.0, 0.0)
    out["view"] = [float(_v[0]) - padx, float(_v[1]) - pady]
    out["score"] = float(best["score"])       # ⚠ 报**贴合分**（口径同判据那行 ✓）
    return out


def locate(frame, canvas, mode):
    """定出「面板 → 底图」的换算。**mode 必填**（MODE_FIT / MODE_CROP）。

    **故意不给默认值**：以前默认是"两种都试、取分高的" —— 那等于程序替你选了。
    可是这两种画法在不同图上确实不一样（要进游戏走两步看地形动不动才知道），
    分数高低说明不了该用哪种（项目规范，见本模块开头）。所以这里强制调用方
    说清是哪个方式，值就是用户在工作台里选的那个。
    """
    if mode == map_const.MODE_FIT:
        return locate_fit(frame, canvas)
    if mode == map_const.MODE_CROP:
        return locate_crop(frame, canvas)
    return None


class CropViewTracker:
    """crop（局部小地图）**运行时跟踪**「现在显示的是底图哪一块」→ `view`（底图像素）。

    **为什么非有不可**（用户 2026-09-29 指定的下一步："开发「**局部小地图**」的世界坐标
    相关功能" ✓）：
      · `fit`（全局小地图）**没这个问题** —— 整张底图永远都在面板里、**地形不随人动**，
        标定文件里那份几何量一次就永久有效 ✓；
      · `crop`（局部小地图）**面板只是底图的一块、而且随玩家滚动** ✓（本模块顶部与
        `docs/寻路设计.md` §5 都写着"随玩家滚动 / 随人滚动"）⇒ 标定文件里存的 `view`
        只是**标定那一刻**的位置 ⇒ 人一走，`panel_to_world` 就整体偏"滚了多少 ×
        底图刻度"（量级：这张图 **1 底图像素 ≈ 7.4 世界像素** ⇒ 滚 50 像素就偏 ~370
        世界像素 ✗）⇒ 世界坐标 / `here_sets` /"怪在哪一层"**全错**，
        而现场看着像"标定没量准"或"怪在天上" ✗。
      ⇒ 所以每拍得重新问一遍"现在显示的是哪一块"，这就是本类 ✓。

    **怎么跟**：拿**面板正中间那一块**当模板（⚠ 为什么是正中间、而不是"起点算出来的那一段"，
    见 `_patch` 的说明 —— 那是最容易写错、且**一错就再也回不来**的一处 ✗），缩回
    **底图尺度**后在底图上找它落在哪 ⇒ 那就是新的 `view` ✓ ——
    换算与 `locate_crop` 同一套：`view = 匹配到的位置 − 模板在面板里的偏移 / scale` ✓
    （`offset` 在 crop 下就是模板从面板边缘往里缩了多少，见 `locate_crop` 的说明 ✓）。

    **在哪找**：
      · 有历史 ⇒ **先在上一拍附近找**（`±pad`，`pad = max(PAD_MIN, PAD_GAIN × 上一拍滚了多少)`）：
        滚动是连续的 ⇒ 附近找又快、又不容易认错到另一处相似的地形上 ✓；
        `PAD_GAIN` 是留给"突然加速"的余量 —— 真追不上时匹配分会掉 ⇒ **这一拍就自动
        全图重找** ✓（自愈，不需要额外的状态机 ✓）；
      · 没有历史（第一拍）/ 附近没找到 ⇒ **全图重找**，而且**门槛分两档**（见 `update` ②）：
        第一拍用"量得出来"那条线（`LOCATE_MIN_SCORE` = 标定时同一把尺 ✓），
        已经在跟的时候重找要 `TRUST_SCORE`（0.8）—— 因为此刻**错一次就会被当成"跟住了"、
        之后都在那一带附近找 ⇒ 再也不会自我纠正** ✗（实测：拿另一张图的底图能到 0.555、
        面板被挡一半也有 0.6 上下 ⇒ 0.55 挡不住"像但不是"）。
      实测（本机基准）：82×218 的底图 + 41×36 的模板，全图找 **~1 ms**、附近找 **~0.5 ms**
      ⇒ 30 拍/秒的实时回路扛得住 ✓（比一帧推理便宜几十倍 ✓）。

    **跟不住时绝不猜** ✗（本项目反复吃过的亏）：保留上一拍跟住的那个（没有就用标定里
    那个），`ok=False` + `why` 说清 —— `PlayerLocator.update` 会把这条写进 `note`/`short`，
    界面那一行看得见 ✓（"我这一拍的世界坐标可不可信"必须当场说得出口 ✓）。
    """

    #: 「附近找」最少给多少余量（底图像素）。2 ≈ 黄点本身的分辨率（亚像素 ±0.5 面板像素
    #: ÷ 缩放）—— 再小就成了"要求零位移"，人一走就掉进全图重找 ✓。
    PAD_MIN = 2.0
    #: 速度余量：上一拍滚了 d ⇒ 这一拍允许 3d（滚动是连续的，允许它加速到 3 倍）
    PAD_GAIN = 3.0
    #: 模板最小边长（底图像素）：太小的模板"配什么都能得高分" ⇒ 等于没量 ✗
    MIN_TPL = 6
    #: ⭐⭐ **连续多少拍"没和底图对上"之后，就不再让帧间位移推进 `view`**（2026-10-18 现场 ✓）。
    #:
    #: **为什么必须有上限**：帧间位移量的是**相对**位移，它自己**没有任何锚** ——
    #: 一旦推流框错/画面不是小地图（现场：推过来的是一块米色墙面 ✓），它照样"跟得住"
    #: （墙面也有纹理！），`view` 就一拍一拍往前爬，越走越远 ○ 现场表现就是
    #: **人明明在底层，位置却算到三楼**（偏了 1127 世界像素 ✗）。
    #: ⇒ 连续 30 拍（≈1 秒 @30fps）都没跨图成功 ⇒ 停手、报"没跟住"，**宁可不动也不漂** ✓
    FLOW_MAX_ABS_MISS = 30
    #: ⭐ **帧间位移**（`_flow` ✓ 2026-10-18）那条路的最低分：它比的是**同一块内容**
    #: 在相邻两帧之间（不是"面板 ↔ 底图"跨图比 ✓）⇒ 噪声小得多，门槛可以订得高
    #: （0.72）✓ 够不着就不动 —— 宁可停一拍，也别把"滚了多少"猜歪 ✗
    FLOW_MIN_SCORE = 0.72
    #: ⭐ **帧间位移单拍最多认多少底图像素**（2026-10-20 第二轮 ✓）：真滚动一拍也就几个像素
    #:   （`VIEW_DEADBAND` 那档），**上百像素的单拍位移只可能是换层 / 面板突变 / 匹配错** ✗
    #:   —— 直接采纳就是世界坐标一次跳一整层（用户："一直认不准 导致一直下跳"✓）。
    FLOW_MAX_STEP = 12.0
    #: 帧间模板取内容区的这一比例（剩下的是"这一拍滚了多少"的余量 ✓）
    FLOW_BAND = 0.40
    #: ⭐⭐⭐ 「**放宽档**」的下限（2026-10-19 ✓ 见 `_update` 里那段）：
    #: 正常 `bar`（≈0.8 那档）在**重绘底图**上够不着（105040303 实测天花板 0.54~0.73 ✓）⇒
    #: 分数不到 bar 也允许报上去，但**不许什么都报**：低于这条线的一律当"没跟住" ✓
    #: （阈值取自现场两档：真跟住 0.54 ✓ / 认错画面 0.16 ✗ —— 0.35 在中间且离两边都远 ✓）。
    #: ⚠ 放宽档报上去的值**仍然是 `trust=False`**，最终要不要用由调用方的
    #:   「地形掩码复核」（`map_locator.crop_view_trusted` ✓）说了算 —— 两道闸串联 ✓。
    RELAX_MIN_SCORE = 0.35

    def __init__(self, min_score=None):
        self.min_score = float(map_const.LOCATE_MIN_SCORE if min_score is None else min_score)
        #: 最近一次**跟住**的 view（底图像素，float 二元组）；**没跟住过 = None**
        #: （⚠ 只由"跟住"那一拍写 —— 没跟住时给的"标定里那个"只进返回值，不进这里 ✗，
        #:   否则下一拍会以为"已经在跟"，全图重找的门槛就抬不起来了 ✓）
        self.view = None
        #: 这一轮**跟住过**没有（`reset` 清掉）—— 决定全图重找用哪条门槛（见 `update` ② ✓）
        self.tracked = False
        #: 最近一次的匹配分（0 = 还没跟住过）
        self.score = 0.0
        #: 连续没跟住几拍（跟住就归零）—— 界面/打点用它说"已经晃了多久"
        self.missed = 0
        #: 这一拍为什么没跟住（跟住时是空串）
        self.why = ""
        #: 这一拍在哪找到的（near / full；排查用）—— 老在 full 上说明"附近找"失灵了
        self.where = ""
        #: 上一拍 view 滚了多少（底图像素）—— 给"附近找"定余量
        self._step = None
        #: 底图灰度缓存（每拍转一次纯属浪费；底图对象在一次会话里不变 ✓）
        self._cg = None
        self._cg_key = None
        #: ⭐ 死区（底图像素 ✓）：`view` 的小变化**不跟**（防叠图每帧抖 ✓ 见 `VIEW_DEADBAND`）
        self.deadband = float(map_const.VIEW_DEADBAND)
        #: 上一拍的面板灰度 + 当时的 inset（`_flow` 的模板 ✓ 没跑过 = None）
        self._prev = None
        #: 这一拍底图**补了多少边**（`(padx, pady)` 底图像素 ✓ 见 `_pad_canvas`）
        self._pad = (0, 0)
        #: 连续**没和底图对上**几拍（跨图成功就归零）—— 帧间位移的"刹车"用它 ✓
        self.abs_miss = 0
        #: 这一拍 `update` **改之前**的 `view`（`reject()` 撤回去用 ✓ 见它的说明）
        self._pre_view = None
        #: ⭐⭐⭐ **刹车之后「停在原地等」**（2026-10-20 ✓ 用户："现在会**一直认不准** 导致一直下跳"✗）：
        #:   刹车之后再"解锁重找"会带来一个更难看的副作用 —— 这张图（105040303）**逐像素相关是
        #:   平的**（天花板 ~0.55）⇒「放宽档」锁回来的位置**每拍都不一样**（等于随机），
        #:   而每次锁上都会把 `abs_miss` 归零 ⇒ **帧间位移又重新放行 30 拍** ⇒ 位置一路漂
        #:   （现场实测：y 1337 → 1438 → 962 → **2290**，两秒跨了整张图 ✗）⇒ 决策层认不准、
        #:   一路"下跳"✓。
        #:   ⇒ 刹车 = **停车**：`parked=True` 时**不走帧间位移、也不走放宽档**（谁都不许动它 ✓），
        #:     只有两件事能解开它：
        #:       ① **真的跨图匹配上**（分数过 `min_score` 的那一路 ✓ 正常图压根到不了这里 ✓）；
        #:       ② 调用方的**地形掩码找回**（`map_calib.relock_view` ✓ 绝对定位、每拍可复算 ✓）。
        self.parked = False

    def reset(self, view=None):
        """换图 / 重标定：忘掉跟过的位置（给了 `view` 就以它当起点 ✓）。"""
        self.view = (float(view[0]), float(view[1])) if view else None
        self.tracked = bool(view)
        self.score = 0.0
        self.missed = 0
        self.why = ""
        self.where = ""
        self._step = None
        self._prev = None         # 换图 / 重标定 ⇒ 上一帧不可比，帧间位移从头攒 ✓
        self._pad = (0, 0)
        self.abs_miss = 0
        self.parked = False       # 换图 / 重标定 ⇒ "停车"也解开 ✓（见 `parked` 的说明）
        return self

    def _gray_canvas(self, canvas):
        """底图灰度（**按对象缓存** —— 底图对象在一次会话里不变 ✓）。"""
        key = (id(canvas), canvas.shape[0], canvas.shape[1])
        if self._cg is None or self._cg_key != key:
            self._cg = _gray(canvas)
            self._cg_key = key
        return self._cg

    @staticmethod
    def _inset(calib):
        """crop 下的面板内缩量 `(x, y)`（= `offset`，见 `locate_crop` 的口径 ✓）。"""
        try:
            ox, oy = calib.get("offset") or (0, 0)
            return float(ox), float(oy)
        except (TypeError, ValueError):
            return 0.0, 0.0

    def _pad_canvas(self, cg, panel):
        """底图两侧/上下补上**面板底色**，让"面板比底图宽"也能匹配 ✓（2026-10-18）。

        局部小地图里，底图外面那圈是"没有地图"的底色（实测 105040303 = 灰 131）——
        在**原始底图**上它不存在 ⇒ 模板里那圈像素没有对应物、分数永远上不去（0.24 ✗）。
        补边之后同一帧能到 **0.69** ✓（这是"缩放 1.0 明明是对的却一直跟不住"的真凶）。

        `self._pad = (padx, pady)`：记账用 —— 进了补边空间的坐标，出去时**必须减回来**
        （`_update` 里一处进、一处出 ✓ 别在别处再补 ✗）。
        """
        # 算法在 `_pad_gray`（**一键标定**那条路也用它 ⇒ 一处实现 ✓）
        out, self._pad = _pad_gray(cg, panel)
        return out

    def _patch(self, panel, canvas, calib, seed, clip=True, csize=None):
        """取模板 → `(模板灰度, 面板左上角, 底图上的期望位置)`；取不出 ⇒ `None`。

        两种取法（`clip`），**都必须有**（各有各挡不住的情况 ✓，见 `update`）：
          · `clip=True` —— **按起点算"面板哪一段落在底图上"**，再往里收掉"起点可能已经
            错掉"的那点余量、并**不超过底图的一半**，在剩下那段里居中取。它最干净
            （一点面板边/底色都不含 ✓），但**依赖起点**：起点旧了就会把"面板边 / 底图外的
            底色"取进来 ✗（那种像素在底图上没有对应 ⇒ 分数掉下去 ⇒ 全图重找也救不回来 ✗）；
          · `clip=False` —— **面板正中间那一半**，完全不看起点 ✓（起点很旧时靠它救回来 ✓），
            代价是：面板边/底色落在正中间时（人贴着地图上/下边缘走）它会含到 ✗。

        ⚠ **上限：不超过底图的一半**（两种取法都要）—— 再大，能放它的位置就只剩几像素 ⇒
        位置根本量不出来（实测：面板比底图**宽**时，77×77 的模板在 82×218 的底图上只有
        **5 个水平位置** ⇒ 匹配永远贴着边界、x 一动不动 ✗ —— 而"面板比底图宽"正是局部
        小地图的常态：实测 105040303 是 82×218 的底图 + 502×408 的面板 ✓）。
        """
        sx, sy = map_calib.scales_of(calib)
        if sx <= 0 or sy <= 0:
            return None
        ph, pw = panel.shape[:2]
        # ⚠ **补边之后**的尺寸要由调用方给（`csize`）—— `canvas` 还是那张没补边的底图，
        #   拿它的尺寸夹模板会把"面板里那圈底色能对上补边"这条路又堵死 ✗（2026-10-18）
        ch, cw = (tuple(csize) if csize else canvas.shape[:2])
        vx, vy = float(seed[0]), float(seed[1])
        ox, oy = self._inset(calib)
        if clip:
            # 面板四边换到底图坐标、和底图求交（超出底图的那一圈在底图上没有对应 ✓）
            bx0 = max(0.0, vx - ox / sx)
            bx1 = min(float(cw), vx + (pw - ox) / sx)
            by0 = max(0.0, vy - oy / sy)
            by1 = min(float(ch), vy + (ph - oy) / sy)
            # **收掉"起点可能已经错掉"的那点余量**：上一拍滚过多少就是它最可能错多少
            # （附近找的窗口也是按它开的 ✓ 两处同一个量 ✓）。
            mpx = (max(self.PAD_MIN, self.PAD_GAIN * abs(self._step[0]))
                   if self._step else self.PAD_MIN)
            mpy = (max(self.PAD_MIN, self.PAD_GAIN * abs(self._step[1]))
                   if self._step else self.PAD_MIN)
            bx0, bx1 = bx0 + mpx, bx1 - mpx
            by0, by1 = by0 + mpy, by1 - mpy
            if bx1 - bx0 < self.MIN_TPL or by1 - by0 < self.MIN_TPL:
                return None
        else:
            bx0, bx1 = vx, vx + pw / sx
            by0, by1 = vy, vy + ph / sy
        # 上限 + 居中
        w = min(bx1 - bx0, max(float(self.MIN_TPL), cw / 2.0))
        h = min(by1 - by0, max(float(self.MIN_TPL), ch / 2.0))
        tw = int(round(w))
        th = int(round(h))
        if tw < self.MIN_TPL or th < self.MIN_TPL:
            return None
        if tw >= cw or th >= ch:      # 必须**严格小于**底图：否则只有一个位置可放 ⇒ 量不出来
            return None
        cx0 = bx0 + (bx1 - bx0 - w) / 2.0
        cy0 = by0 + (by1 - by0 - h) / 2.0
        px0 = int(max(0, min(pw - 1, int(np.floor((cx0 - vx) * sx + ox)))))
        px1 = int(max(px0 + 1, min(pw, int(np.ceil((cx0 + w - vx) * sx + ox)))))
        py0 = int(max(0, min(ph - 1, int(np.floor((cy0 - vy) * sy + oy)))))
        py1 = int(max(py0 + 1, min(ph, int(np.ceil((cy0 + h - vy) * sy + oy)))))
        sub = panel[py0:py1, px0:px1]
        if getattr(sub, "size", 0) == 0:
            return None
        tpl = cv2.resize(_gray(sub), (tw, th), interpolation=cv2.INTER_AREA)
        # 这个模板在**底图上的期望位置**（按起点算）：附近找的窗口围着它 ✓
        ax = (px0 - ox) / sx + vx
        ay = (py0 - oy) / sy + vy
        return tpl, (px0, py0), (ax, ay)

    def _match(self, region, tpl, ox, oy, where, prior=None):
        """在 `region`（底图灰度的一段）里找 `tpl` → `(分数, 绝对位置 x, y, 在哪找的)`。

        ⭐ `prior`（**先验位置**，底图坐标）给了就**取"离先验最近的那个高峰"**，而不是
        全局最大 —— 小地图平台**等间距重复**，响应图上每根平台一个峰、分数几乎一样
        （实测 0.84~0.88）⇒ 取全局最大会在平台之间**跳**（人没动，`view_y` 在 18/48/77
        之间跳，正好差一根平台 ✗ 2026-10-18 真机实测）。
        """
        rh, rw = region.shape[:2]
        th, tw = tpl.shape[:2]
        if tw > rw or th > rh or tw < 1 or th < 1:
            return None
        res = np.nan_to_num(cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED))
        _, mx, _, ml = cv2.minMaxLoc(res)
        if prior is not None:
            pick = _nearest_peak(res, mx, (float(prior[0]) - float(ox),
                                           float(prior[1]) - float(oy)))
            if pick is not None:
                mx = float(res[int(round(pick[1])), int(round(pick[0]))])
                fx, fy = _subpixel_peak(res, (int(round(pick[0])), int(round(pick[1]))))
                return float(mx), float(fx) + float(ox), float(fy) + float(oy), where
        fx, fy = _subpixel_peak(res, ml)
        return float(mx), float(fx) + float(ox), float(fy) + float(oy), where

    def _search(self, cg, patch, cw, ch, axes=(True, True)):
        """在一个模板上跑「**附近找 → 全图重找**」→ `((分数, x, y, where)|None, 见过的最好分)`。

        `axes` = `(能动 X, 能动 Y)`（`scroll_axes(calib)` ✓ **单轴的图只搜那根轴** ——
        用户 2026-09-29："仅X/仅Y"的图上实时匹配要快 ✓）：
          · **附近找**：不动的那根轴 `pad = 0` ⇒ 窗口在那一轴上**只有模板那么宽** ✓
            （面板在那一轴上的小抖动不会把结果带偏 ✓），而且那根轴的搜索面积直接没了 ✓；
          · **全图重找**：不动的那根轴收成一条**带**（横带 / 竖带，高/宽 = 模板 + 一点余量 ✓）
            ⇒ matchTemplate 的面积少掉一大截（实测：底图 82×218、模板 41×36 时，
            仅 X 那条横带的面积是整图的 ~20% ✓ ~5 倍 ✓）。
          · ⚠ 只在**有先验**时才能收成带（`ax/ay` = 模板按当前 view 该落在哪儿 ✓）；
            先验不可信（还没跟住过 / 落在底图外）⇒ 老实全图找 ✓（宁慢不猜 ✗）。

        门槛**分两档**（用户 2026-09-29 ✓ 实测定的）：
          · **附近找**（有历史才走）⇒ 用「量得出来」那条线（`min_score` = 标定时同一把尺 ✓）。
            滚动是连续的 ⇒ 附近找**认错也只能错在 pad 之内**（几个底图像素 ✓）；
          · **全图重找** ⇒ 第一拍（还没跟住过）仍是 `min_score`（标定那一刻人刚好站在
            那儿 ⇒ 标定里那个 `view` 就是先验 ✓）；**已经在跟**的时候要 `TRUST_SCORE`
            （0.8，"这份几何可信"那档）。为什么必须抬：此刻错一次就会被当成"跟住了"、
            之后每拍都围着那一带找 ⇒ **再也不会自我纠正** ✗。
            实测：拿**另一张图**的底图去匹配能到 **0.555**、面板被挡掉一半（重复花纹那种
            图）也有 **0.6** 上下 ⇒ 0.55 这条线挡不住"像但不是" ✗。

        ⚠⚠ **附近找的结果"不太像"时要再全图找一次**（2026-09-29 实测踩到）：窗口只有
          `pad`（冷启动是 `PAD_MIN` = 2 个底图像素）⇒ 人**站着不动之后猛一走**
          （或传送/跳一下）真位置落在窗口外 ⇒ 匹配就贴着窗口边给个 0.6~0.7 的结果
          （过得了 0.55 ✓）⇒ 那一拍的世界坐标偏 pad 那么多，而且**看着像跟住了** ✗。
          ⇒ 采纳条件加一条：**已经在跟**的时候，附近找的分要到 `TRUST_SCORE` 才直接算数；
          不够就**再全图找一次**，谁分高用谁 ✓（全图找 ~1 ms，只在"不太像"那拍多花 ✓）。
        """
        ax_on, ay_on = axes
        tpl, _pq, (ax, ay) = patch
        th, tw = tpl.shape[:2]
        best, seen = None, 0.0
        # 门槛：附近找要够像才算数（不够 ⇒ 走下面那次全图找 ✓）
        bar = (self.min_score if not self.tracked
               else max(self.min_score, map_const.TRUST_SCORE))
        if self._step is not None and self.view is not None:
            # ⭐ 不动的那根轴**余量为 0**（单轴的图：面板在那一轴上不会滚 ✓）
            padx = (max(self.PAD_MIN, self.PAD_GAIN * abs(self._step[0]))
                    if ax_on else 0.0)
            pady = (max(self.PAD_MIN, self.PAD_GAIN * abs(self._step[1]))
                    if ay_on else 0.0)
            wx0 = int(max(0, np.floor(ax - padx)))
            wy0 = int(max(0, np.floor(ay - pady)))
            wx1 = int(min(cw, np.ceil(ax + padx) + tw))
            wy1 = int(min(ch, np.ceil(ay + pady) + th))
            if wx1 - wx0 >= tw and wy1 - wy0 >= th:
                near = self._match(cg[wy0:wy1, wx0:wx1], tpl, wx0, wy0, "near",
                                   prior=(ax, ay))
                if near is not None:
                    seen = max(seen, near[0])
                    if near[0] >= bar:
                        best = near
        if best is None:
            # ⭐ 全图重找：不动的那根轴收成**一条带**（先验 = 模板按当前 view 该落在哪儿 ✓，
            #    由 `_patch` 给的那个 `ax/ay` ✓）。带子比整图小一大截 ⇒ 单轴图上快好几倍 ✓。
            #    ⚠ **带里没找到像的 ⇒ 再老实全图找一次**（多花 ~1 ms，但"先验恰好是错的"
            #    —— 标定里的位置不准 / 刚换图 —— 那一拍就不会白丢 ✓ 宁慢不猜 ✗）。
            for _try_band in (True, False):
                got, sc = self._search_full(cg, tpl, cw, ch, bar,
                                            (ax, ay) if _try_band else None, axes)
                seen = max(seen, sc)
                if got is not None:
                    best = got
                    break
                if not _try_band:
                    break
        return best, seen

    def _search_full(self, cg, tpl, cw, ch, bar, prior, axes):
        """全图（或单轴时的**一条带**）里找 → `(分数, x, y, where)|None`（不够好 ⇒ None ✓）。

        `prior = (ax, ay)`（模板按当前 view 该落在哪儿 ✓）：给了它、而且是**单轴**的图
        ⇒ 只扫不动的那根轴上的一条带 ✓；`prior=None` ⇒ 整图 ✓（宁慢不猜 ✗）。
        """
        ax_on, ay_on = axes
        th, tw = tpl.shape[:2]
        reg, rx0, ry0 = cg, 0, 0
        if prior is not None and not (ax_on and ay_on):
            ax, ay = prior
            band = int(self.PAD_MIN) + 2          # 容忍先验自己差几个像素 ✓
            if not ay_on:
                ry0 = int(max(0, np.floor(ay) - band))
                ry1 = int(min(ch, np.ceil(ay) + th + band))
                if ry1 - ry0 >= th:
                    reg = cg[ry0:ry1, :]
            if not ax_on:
                rx0 = int(max(0, np.floor(ax) - band))
                rx1 = int(min(cw, np.ceil(ax) + tw + band))
                if rx1 - rx0 >= tw:
                    reg = reg[:, rx0:rx1]
            if (not ay_on and reg.shape[0] < th) or (not ax_on and reg.shape[1] < tw):
                reg, rx0, ry0 = cg, 0, 0            # 带子装不下 ⇒ 退回整图 ✓
        # ⚠ 带先验：**高峰里取离先验最近的**（平台等间距重复 ⇒ 不这样会在平台间跳 ✓）
        full = self._match(reg, tpl, rx0, ry0, "full", prior=prior)
        if full is None:
            return None, 0.0
        return (full if full[0] >= bar else None), float(full[0])

    def _shift_rows(self, old, new, r0, r1, c0, c1, lim=None):
        """`old` 中间那条带 → 在 `new` 的同一块里找它挪了几行 ⇒ `(位移, 分数) | None`。

        **只量一个方向**（调用方已经把"要量的那根轴"转成行了 ✓）。模板取**中间一条带**
        （`FLOW_BAND` 40% ✓），留出的余量正好够"这一拍滚了多少" ✓；
        够不着 `FLOW_MIN_SCORE` ⇒ `None`（宁可停一拍，不猜 ✗）。

        ⚠⚠ **`lim`（最多允许挪多少行）必须给**（2026-10-18 现场 ✓）：小地图上的平台是
        **一根一根等间距重复**的 ⇒ 整块搜索会锁到**隔壁那根平台**上（实测：人没动，
        `view_y` 却在 42→70→41 之间来回跳 ✗ 一拍就差一根平台的距离）。限死窗口之后，
        它只能在"上一拍的速度"附近找 ⇒ 跳平台这条路直接堵掉 ✓。
        """
        h, w = r1 - r0, c1 - c0
        if h < 12 or w < 12:
            return None
        th = int(h * self.FLOW_BAND)
        if th < 8:
            return None
        ty0 = r0 + (h - th) // 2
        tpl = old[ty0:ty0 + th, c0:c1]
        reg = new[r0:r1, c0:c1]
        top = r0
        if lim is not None:
            lo = max(r0, ty0 - int(round(lim)))
            hi = min(r1, ty0 + th + int(round(lim)))
            if hi - lo < th + 2:
                return None
            reg = new[lo:hi, c0:c1]
            top = lo
        if tpl.shape[0] >= reg.shape[0] or tpl.shape[1] != reg.shape[1]:
            return None
        try:
            res = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return None
        if res is None or getattr(res, "size", 0) == 0:
            return None
        _, score, _, ml = cv2.minMaxLoc(res)
        if not np.isfinite(score) or score < self.FLOW_MIN_SCORE:
            return None
        return (int(ml[1]) + top - ty0, float(score))

    def _flow(self, panel, calib, axes):
        """**帧间位移**：拿上一帧的面板当模板，量这一拍滚了多少 ⇒ 新的 `view`。

        为什么需要（2026-10-18 现场 ✓）：`scroll="y"` 的图（105040303：底图 82×218、
        面板 110×112）**每拍面板内容都在上下滚**，而世界 y = `view_y` + 面板行/缩放 ⇒
        `view_y` 跟不上，人就"上下永远不动"（现场 **7997 次**「没跟住显示区 ⇒ 用标定里
        那个位置」✓ 用户报的「实时小地图没有上下移动」正是它）。

        它跟"面板 ↔ 底图"那条的根本区别：**不跨图**，只比**相邻两帧的同一块内容**
        ⇒ 底图缩放 / 配色 / 局部被挡**都不影响**（量的是相对量 ✓）。代价是误差会**累计**
        （所以这条路给的 `trust=False` ✓ 界面该说"跟住但不太稳"）。

        返回 `(view, 分数) | None`（`None` = 这一拍量不出来 ⇒ 调用方走原来的兜底 ✓）。
        """
        if self._prev is None:
            return None
        ax_on, ay_on = axes
        if ax_on and ay_on:       # 两轴都会滚 ⇒ 一条带定不住（不猜 ✗ 交给全图那条 ✓）
            return None
        try:
            sx, sy = map_calib.scales_of(calib)
        except (TypeError, ValueError):
            return None
        if sx <= 0 or sy <= 0:
            return None
        base = self.view
        if base is None:          # 没有绝对锚点 ⇒ 相对位移没处加（第一拍交给标定 ✓）
            return None
        pg, (pox, poy) = self._prev
        cur = _gray(panel)
        ox, oy = self._inset(calib)
        # 面板尺寸 / 内缩变了（换了框选、改了显示方式）⇒ 上一帧根本不可比 ✗
        if pg.shape != cur.shape or abs(pox - ox) > 1.5 or abs(poy - oy) > 1.5:
            return None
        ph, pw = cur.shape[:2]
        axis = 1 if ay_on else 0
        scale = float(sy) if ay_on else float(sx)
        # 这一拍**最多允许挪多少行**：按"上一拍真挪了多少"的三倍 + 一点余量（同 `PAD_GAIN` ✓）。
        # 没历史 ⇒ 只给一点点（人不可能一上来就瞬移 ✓）—— 这条就是防"锁到隔壁那根平台"的。
        # ⚠ `self._step` 是**底图像素**，这里的位移是**面板像素** ⇒ 要乘缩放 ✓
        #   （`PAD_MIN` 2 底图像素在 5.8 倍的图上就是 11 面板像素，不折算会窄得离谱 ✗）
        if self._step is not None:
            lim = max(float(self.PAD_MIN) * 2.0,
                      float(self.PAD_GAIN) * abs(float(self._step[axis])))
        else:
            lim = float(self.PAD_MIN) * 2.0
        lim = max(4.0, lim * max(1.0, scale))
        # ⚠ **上限**：一拍真滚不了多少（30 拍/秒下通常 <2 px）⇒ 万一上一拍被"跳一根平台"
        #   带歪了，`3×` 会把窗口撑到 30~90 px，下一拍继续跳 ⇒ 恶性循环 ✗（2026-10-18 实测）
        lim = min(lim, max(6.0, 0.25 * float(ph - 2 * int(oy))))
        if ay_on:
            # 上下滚：模板取"中间一条横带"，在整块内容里找 ✓
            got = self._shift_rows(pg, cur, int(oy), ph - int(oy), int(ox), pw - int(ox),
                                   lim=lim)
        else:
            # 左右滚：**转置**后"上下滚"就是原来的"左右滚"（同一套代码，不写两份 ✓）；
            # ⚠ 转置后行数受限的是**原来的列数**，所以限窗也按列来 ✓
            got = self._shift_rows(np.ascontiguousarray(pg.T), np.ascontiguousarray(cur.T),
                                   int(ox), pw - int(ox), int(oy), ph - int(oy),
                                   lim=lim)
        if got is None:
            return None
        shift, score = got
        # 同一块内容从第 `ty0` 行挪到了 `ty0+shift` 行 ⇒ 面板行 +shift
        # ⇒ `view` 沿该轴**减** shift/缩放（`p = (by - view)*s + offset` ✓ 见 `_patch`）
        nv = [float(base[0]), float(base[1])]
        nv[axis] += -float(shift) / scale
        return ((nv[0], nv[1]), score)

    def update(self, panel, canvas, calib=None):
        """这一拍的面板 → `view` 跟踪结论 dict（**薄壳**）。

        真正干活的是 `_update`；这里只负责"收尾时记住这一帧的面板灰度" ——
        帧间位移（`_flow`）要拿**上一帧**当模板 ✓，而 `_update` 里提前 return 的路子
        很多（没标定 / 没画面 / 跟丢）⇒ 必须放在 `finally` 里，不能只写在末尾 ✗
        （2026-10-18 加「A：帧间位移跟踪」时踩的点）。

        ⭐ 同时**记下这一拍改之前的位置**（`_pre_view`），给 `reject()` 用 ✓ ——
        「掩码复核没过」时调用方会把它撤回去 ✓（用户 2026-10-20"乱飞乱抖"✓）。
        """
        self._pre_view = self.view
        try:
            return self._update(panel, canvas, calib)
        finally:
            self._remember(panel, calib)

    def reject(self, why=""):
        """**撤回这一拍的 `view`**（调用方复核没过 ⇒ `map_locator.crop_view_trusted` ✗）。

        为什么要有（用户 2026-10-20 ✓ 原话："现在会**乱飞乱抖**，匹配的完全不对"✗）：
          复核没过却把错位置留在跟踪器里 ⇒ 下一拍"附近找"以它为起点（窗口也是它算的 ✓）
          ⇒ 接着错、错上 30 拍再刹车，下一拍又挑一个错峰 ⇒ 看着就是"隔一阵跳一下"✓。
        ⇒ 撤回到**这一拍改之前**那个位置，并把 `tracked` 清掉（0.8 那道门槛别把我们锁死 ✓）
          —— 下一拍重新找，找不对就继续被拒，**输出始终停在最后一个可信位置** ✓。
        """
        self.view = getattr(self, "_pre_view", None) or self.view
        self.tracked = False
        self.score = 0.0
        self.why = str(why or "复核没过（地形掩码）")
        self.where = "rejected"
        return self

    def _remember(self, panel, calib):
        """存下这一帧的面板灰度 + 当时的 inset，给下一拍 `_flow` 当模板 ✓。"""
        try:
            if panel is None or getattr(panel, "size", 0) == 0:
                self._prev = None
                return
            self._prev = (_gray(panel), self._inset(calib or {}))
        except (AttributeError, TypeError, ValueError):
            self._prev = None

    def _update(self, panel, canvas, calib=None):
        """这一拍的面板 → `view` 跟踪结论 dict。

        返回 `{"ok", "view": [x, y]|None, "score", "trust", "why", "missed", "where"}`；
        `view` 是**可以用的那个**：跟住 ⇒ 新跟出来的；没跟住 ⇒ 上一拍跟住的那个
        （都没有 ⇒ 标定里那个）⇒ 调用方只在 `ok` 时才该改几何 ✓（没跟住时
        标定里那份就是今天的行为，一字不变 ✓）。
        `trust` = 这一拍的匹配分到没到**可信**那一档（`TRUST_SCORE`）—— 面板被挡掉一块时
        分会落在中间那带（`trust=False`），**照样能用**，但界面该说一句"不太稳" ✓。
        """
        out = {"ok": False, "view": None, "score": 0.0, "why": "", "where": ""}
        if panel is None or canvas is None or getattr(canvas, "size", 0) == 0:
            out["why"] = "没有面板画面 / 没有底图"
            return self._done(out, calib)
        if (calib or {}).get("mode") != map_const.MODE_CROP:
            out["why"] = "这份标定不是「局部小地图」"
            return self._done(out, calib)
        if not map_calib.has_geometry(calib):
            out["why"] = "标定里还没有几何（缩放 / 显示区起点）"
            return self._done(out, calib)
        cg = self._gray_canvas(canvas)
        # ⭐⭐⭐ **补边**（2026-10-18 实测 ✓ 这是"局面板一直跟不住"的真凶之一）：
        #   局部小地图的**面板常常比底图宽**（105040303：底图 82×218、面板 110×112，
        #   缩放 1.0 ✓）⇒ 底图左右两侧在面板里显示的是"没有地图"的底色 ✓。
        #   不补边时：模板被 `_patch` 夹成"底图的一半"、而面板那圈底色在底图上**没有任何
        #   对应** ⇒ 分数只能是噪声（现场 0.24 ✗）；补上同色边之后**同一帧能到 0.69** ✓。
        cg = self._pad_canvas(cg, panel)
        ch, cw = cg.shape[:2]
        padx, pady = self._pad
        sx, sy = map_calib.scales_of(calib)
        ox, oy = self._inset(calib)
        # 起点候选：优先"上一拍跟住的那个"（连续 ✓），其次**标定里那个**（第一拍 / 人刚
        # 重标过 ✓）。它决定两件事：模板取哪一段（`clip=True` 那路 ✓）与"附近找"的窗口 ✓。
        # ⚠ 起点是**底图坐标** ⇒ 进补边空间要 +pad（最后出结果时再减回去 ✓ 一处进一处出 ✓）。
        seeds = []
        for cand in (self.view, (calib.get("view") or (0, 0))):
            if cand is None:
                continue
            cc = (float(cand[0]) + padx, float(cand[1]) + pady)
            if cc not in seeds:
                seeds.append(cc)
        # 模板候选（顺序 = 优先尝试的顺序）：先按起点裁出来的（最干净），再**面板正中间**
        # 那块（不看起点 ⇒ 起点很旧 / 跟丢一阵时靠它救回来 ✓）。两者可能一样 ⇒ 去重 ✓
        # （按"面板上哪个矩形"去重 —— 同样的方块跑两遍纯属浪费 ✗）。
        patches = []
        for seed in seeds:
            pt = self._patch(panel, canvas, calib, seed, clip=True, csize=(cw, ch))
            if pt is not None and all(p[1] != pt[1] for p in patches):
                patches.append(pt)
        pt2 = self._patch(panel, canvas, calib, seeds[0] if seeds else (0, 0),
                          clip=False, csize=(cw, ch))
        if pt2 is not None and all(p[1] != pt2[1] for p in patches):
            patches.append(pt2)
        if not patches:
            out["why"] = ("面板里能对到底图的那一段太小 —— 面板没框全 / 标定的缩放或"
                          "显示区起点不对")
            return self._done(out, calib)
        seen = 0.0               # 这一拍见过的最好分数（只为把"差多少"说清楚 ✓）
        best = None
        pick = None
        # ⭐ **滚动方向**（用户 2026-09-29 ✓）：单轴的图只搜会动的那根轴 ⇒ 又快、又不会
        #   把"根本不动的轴"跟歪 ✓（面板在那一轴上的抖动 / 匹配噪声都进不来 ✓）。
        axes = scroll_axes(calib)
        for pt in patches:
            got, sc = self._search(cg, pt, cw, ch, axes=axes)
            seen = max(seen, sc)
            if got is not None:
                best, pick = got, pt
                break
        if best is None:
            # ⭐⭐⭐ **放宽一档再试一次**（2026-10-19 ✓ 105040303「森林迷宫_珠缨」现场）：
            #   这张图的底图是**重绘的小地图**、面板是客户端渲染 ⇒ 逐像素相关**天花板只有
            #   ~0.55**（实测沿 view.y 从 80 扫到 175 全是 0.70~0.73 的平顶，见
            #   `perception/map_locator.crop_view_trusted` 那段），按正常 `bar` **永远**"跟不住"
            #   ⇒ 面板一滚，`view` 就冻在标定那份上（人一换层，读数就整体偏一整层 ✗
            #   用户看到的就是"局部小地图的匹配完全错误"✓）。
            #   ⇒ 分数不到 bar 也**先把最好的那个报上去**，由调用方拿「地形掩码复核」
            #   （`map_locator.crop_view_trusted` ✓ 它才是这张图上靠谱的尺子）决定要不要：
            #   复核不过 ⇒ 照样退回标定 ✓（闭环与"没跟住"一字不差 ✓）。
            #   ⚠ `trust=False` 必须给（这一拍终究是"低分档"⇒ 界面该说不太稳 ✓）。
            _relax = None
            _relax_sc = 0.0
            for pt in patches:
                # ⚠ `patches` 里装的是 `(模板图, 面板左上, 底图锚点)` ⇒ 模板是 `pt[0]` ✓
                got_r, sc_r = self._search_full(cg, pt[0], cw, ch, 0.0, None, axes)
                if got_r is not None and float(sc_r) > _relax_sc:
                    _relax, _relax_sc = (got_r, pt), float(sc_r)
            # ⚠ **只在"还没锁定过"时放宽**（`self.tracked` 为假 ✓）：
            #   老用例 `t_crop_view_track` 钉着「**已经在跟**的时候，全图重找的门槛必须抬高」
            #   （0.65 那种半吊子分不许把已经跟住的位置拽走 ✗ 错一次就再也纠不回来）——
            #   放宽档只负责**第一次锁上**；锁上之后继续跟靠"附近搜 + 帧间位移" ✓
            #   而且**锁上之后每次报上来的值仍然过掩码复核** ⇒ 漂了会当场被打回标定 ✓。
            # ⚠ **不够这条线也不放宽**（"跟不住不许猜"那条契约不许丢 ✓ 见
            #   `RELAX_MIN_SCORE` 的说明：真跟住 0.54 / 认错画面 0.16，阈值摆中间 ✓）
            #   ⚠⚠ **`parked` 时连放宽档也不许开**（2026-10-20 第二轮 ✓ 用户："现在会一直
            #     认不准 导致一直下跳"✗）：这张图的分数是**平的** ⇒ 放宽档锁回来的位置**每拍
            #     都不一样**（等于随机）⇒ 位置乱跳 ⇒ 决策层一路"下跳"✓（实测 y 两秒跨整张图）。
            #     停车之后只认**真的跨图匹配** 与 调用方的**地形掩码找回** ✓（见 `parked`）。
            if (not self.tracked and not self.parked and _relax is not None
                    and _relax_sc >= self.RELAX_MIN_SCORE):
                best, pick = _relax
                out["relaxed"] = True
                seen = max(seen, _relax_sc)
        if best is None:
            self.abs_miss += 1
            # ⭐ **再看要不要让帧间位移接**（2026-10-18 加 ✓ 用户选的「A」）：跨图匹配失败
            #   ≠ 人没动 —— 面板每拍都在滚，只是"面板 ↔ 底图"对不上 ⇒ 拿上一帧当模板量
            #   **相对位移**，至少让 `view` 跟着走 ✓。绝对位置仍由标定锚定、误差**累计**
            #   ⇒ 这条路给 `trust=False`（界面该说"跟住但不太稳"✓，别当成干净那一拍 ✗）。
            # ⚠⚠ 但**必须有刹车**（`FLOW_MAX_ABS_MISS`）：推流框错 / 画面根本不是小地图时，
            #   帧间位移照样"跟得住"（墙面也有纹理 ✓ 现场就是这么漂到三楼的 ✗）⇒
            #   连续 30 拍没跨图成功就停手，宁可不动 ✓
            if self.abs_miss > self.FLOW_MAX_ABS_MISS:
                # ⭐⭐⭐ **放弃帧间位移的同时，必须把"已锁定"解掉**（2026-10-20 ✓ 用户原话：
                #   "森林迷宫_珠缨项目，L9 **完全找错了**"✗）
                #   病根：`tracked=True` 时"附近找"的门槛是 `max(min_score, TRUST_SCORE)`
                #   = **0.8** ✓，而这张图（105040303 重绘底图）逐像素相关的**天花板只有
                #   ~0.55** ✗ ⇒ 一旦丢了锚就**再也锁不回来** ⇒ `view` 永远冻在标定那份 ⇒
                #   人一换层，世界坐标整体偏（实测差 12 底图像素 ≈ **135 世界像素**，
                #   正好差一层）⇒ 绳梯/集合全认错 ⇒ 现场看到的就是"L9 找错了"✓
                #   （日志实证：连续 **18397 拍**都没锚，一路用标定里那个位置 ✓）。
                #   ⇒ 刹车只管"别再让帧间位移往前爬" ✓；它**不是**"以后都不许重找"✗ ——
                #     解锁之后下一拍就走「放宽档 + 地形掩码复核」那两道闸重新锁 ✓
                #     （0.35 下限 + `crop_view_trusted` 复核，猜不出来 ✓）。
                self.tracked = False
                self.parked = True
                out["score"] = float(seen)
                out["why"] = ("连续 %d 拍没和底图对上（只有帧间位移，没有锚）⇒ **不再跟**"
                              "（怕一路漂到别的楼层），这一拍不用它，并**停车**"
                              "（等真的跨图匹配上 / 等掩码找回）—— 顺手看一眼推流框"
                              "是不是框到小地图外面去了"
                              % self.abs_miss)
                return self._done(out, calib)
            flow = None if self.parked else self._flow(panel, calib, axes)
            if flow is not None:
                nv, fsc = flow
                prev = self.view
                if prev is not None and self.deadband > 0.0:
                    if (abs(nv[0] - prev[0]) < self.deadband
                            and abs(nv[1] - prev[1]) < self.deadband):
                        nv = (float(prev[0]), float(prev[1]))
                # ⚠⚠ **单拍位移离谱 ⇒ 这帧的帧间位移不可信**（2026-10-20 第二轮 ✓）：
                #   实测这张图有过"一拍挪 100+ 底图像素"的（换层 / 面板突变）⇒ 直接采纳就是
                #   世界坐标一次性跳一整层 ✗（用户看到的"认不准、一直下跳"✓）。
                #   ⇒ 超过 `FLOW_MAX_STEP` 就当**这一拍没接住**（下面那段失败分支收拾 ✓）。
                if (prev is not None
                        and max(abs(nv[0] - prev[0]), abs(nv[1] - prev[1])) > self.FLOW_MAX_STEP):
                    flow = None
            if flow is not None:
                self.view = (float(nv[0]), float(nv[1]))
                self.tracked = True
                self.score = float(fsc)
                self.missed = 0
                self.why = ""
                self.where = "flow"
                self._step = ((self.view[0] - prev[0], self.view[1] - prev[1])
                              if prev is not None else None)
                out.update(ok=True, view=[self.view[0], self.view[1]],
                           score=float(fsc), where="flow", trust=False)
                return out
            out["score"] = float(seen)
            out["why"] = ("没在底图上找到这块面板（最好的一次匹配分只有 %.2f）—— 面板被挡住 / "
                          "换了图 / 「显示方式」选错，或标定的缩放不对"
                          % seen)
            return self._done(out, calib)
        tpl, (px0, py0), (ax, ay) = pick
        score, mx, my, where = best
        # 匹配到的位置 → view：**模板左上角对应面板的 (px0, py0)**（同一套换算，
        # ⚠ 别再加一次 inset —— `locate_crop` 那条"view 里已经含 inset"的约定在这边
        #   由 `(px0 - ox)` 抵消掉 ✓，`t_crop_roundtrip` 钉着同一个口径 ✓）。
        # ⚠ 匹配是在**补边空间**里做的（`cg` ✓）⇒ 出结果时要减掉那圈边，
        #   否则 `view` 每拍都会整体偏 pad（一处进一处出 ✓）
        nvx = float(mx) - padx - (float(px0) - ox) / sx
        nvy = float(my) - pady - (float(py0) - oy) / sy
        # ⭐ 单轴：不动的那根轴**一个字都不改**（照上一拍/标定里那个 ✓）—— 带子搜索给回来的
        #   那个坐标本来就带着几个像素的余量 ⇒ 直接采纳就会让显示区在那一轴上慢慢漂 ✗。
        ax_on, ay_on = axes
        if seeds and not (ax_on and ay_on):
            # ⚠ `seeds` 是**补边空间**里的坐标 ⇒ 锁回不动的那根轴时也要减掉那圈边 ✗
            #   （不然 `view` 会整体偏 pad —— 2026-10-18 实测：x 从 -9.7 变 18.3 ✗）
            if not ay_on:
                nvy = float(seeds[0][1]) - pady
            if not ax_on:
                nvx = float(seeds[0][0]) - padx
        prev = self.view
        # ⭐⭐ **死区**：面板其实没动时，模板匹配也会在相邻像素间来回跳（±1 底图像素 ✓）
        #   ⇒ `view` 每拍变一点 ⇒ 叠图每帧挪一下 ⇒ 用户说的「**疯狂抖动**」✗（2026-10-17 ✓）。
        #   ⇒ 变化小于 `self.deadband` ⇒ **保持上一拍**（不写新值）✓：
        #      · 真滚动每拍都超过 deadband ⇒ **一点延迟都没有** ✓（不是限速 ✗）；
        #      · 慢滚（每拍不到 1.5 底图像素）⇒ 攒够再跟一步 ✓ 肉眼是"一格一格走"，
        #        而不再是每帧抖 1 像素 × 放大倍数 ✓；
        #      · 第一拍 `prev is None` ⇒ 照常写 ✓（必须有值 ✓）。
        if prev is not None and self.deadband > 0.0:
            if (abs(float(nvx) - float(prev[0])) < self.deadband
                    and abs(float(nvy) - float(prev[1])) < self.deadband):
                nvx, nvy = float(prev[0]), float(prev[1])
        self.view = (nvx, nvy)
        self.tracked = True
        self.abs_miss = 0          # 跨图成功了 ⇒ 帧间位移的刹车归零 ✓
        self.parked = False        # ⭐ **真的跨图匹配上 ⇒ 解开"停车"** ✓（见 `parked` 的说明）
        self.score = float(score)
        self.missed = 0
        self.why = ""
        self.where = where
        # 下一拍的"附近"余量按这一拍真滚了多少来定（自己标定自己 ⇒ 不用拍一个速度常数 ✓）
        self._step = ((nvx - prev[0], nvy - prev[1]) if prev is not None else None)
        # 「可信吗」也一起报（口径同 `TRUST_SCORE`：0.55 那档只说明"量得出来"，0.8 才是
        # "这份几何可信"✓）。面板被挡掉一块时分数就落在中间那一带，而**平掉的那半不贡献
        # 方差** ⇒ 分数会虚高（实测：上半被挡 ⇒ 0.6~0.7，位置还差几个底图像素）
        # ⇒ 界面那行要能说"跟住但不太稳"，别让人以为这拍的位置和干净那拍一样准 ✓。
        # ⚠ 「放宽档」跟到的**一律 `trust=False`**（它本来就是分数不够那档 ✓ 见上面那段）——
        #   界面照旧提示"跟住了，但匹配分只有 X" ✓，而要不要用由调用方的掩码复核说了算 ✓。
        out.update(ok=True, view=[nvx, nvy], score=float(score),
                   where=where if not out.get("relaxed") else "relaxed",
                   trust=bool(float(score) >= map_const.TRUST_SCORE)
                         and not out.get("relaxed"))
        return out

    def _done(self, out, calib):
        """没跟住这一拍：**保留**上一拍跟住的那个（没有就用标定里那个），只记一笔 ✓。

        ⚠ `self.view` **只由"跟住"那一拍写** —— 这里只是把"这一拍该用哪个 view"
        填进返回值（标定里那个是兜底，不是"跟住了"✗，见 `tracked` 的说明 ✓）。
        """
        if out.get("ok"):
            return out
        self.missed += 1
        self.why = out.get("why") or ""
        self.where = out.get("where") or ""
        if out.get("view") is None:
            v = self.view
            if v is None:
                v = (calib or {}).get("view")
                try:
                    v = (float(v[0]), float(v[1])) if v else None
                except (TypeError, ValueError, IndexError):
                    v = None
            if v is not None:
                out["view"] = [v[0], v[1]]
        out["missed"] = self.missed
        return out


def region_match_score(frame, rect, terrain, mode=None, ok_score=map_const.TRUST_SCORE):
    """框出来的那一块画面 ↔ 底图，对一次分（**框完当场验证**的唯一一份实现）。

    **为什么抽到这里**（2026-09-27）：「框选小地图」按钮按用户要求从「路线识别」页搬进
    设置的「实时画面 · 地形叠加」组，而这个"当场验证"原来长在
    `route_panel._verify_mmap_region` 里 —— 不抽出来就是两处各写一份（README
    「口径只有一处」那条 ✗）。现在设置窗（和任何要验证的调用方）都走它。

    验证什么：框大了（把血条/聊天栏框进去）、面板被游戏 UI 挡住、「显示方式」选错 ——
    这三种的现象都是"寻路看起来坏了"，而在这里（匹配分）一眼能看出来。

    `ok_score` 沿用的就是 `locate_fit` / `locate_crop` 内部那条 `min_score` 0.8，
    **不是这里新拍的一个数** ✓。
    返回 `{"ok", "score": float|None, "why": str, "mode": str}`；`why` 是人话。
    """
    mode = mode or map_const.MODE_FIT
    if terrain is None or getattr(terrain, "canvas", None) is None:
        return {"ok": False, "score": None, "mode": mode,
                "why": "这张图还没有底图（先「生成地形图」）"}
    if not rect or len(list(rect)) != 4:
        return {"ok": False, "score": None, "mode": mode, "why": "还没框选区域"}
    x, y, w, h = (int(v) for v in rect)
    if w < 1 or h < 1:
        return {"ok": False, "score": None, "mode": mode, "why": "框选区域是空的"}
    fh_, fw_ = frame.shape[:2]
    if x < 0 or y < 0 or x + w > fw_ or y + h > fh_:
        # 画面尺寸变过（换分辨率 / 改推流参数）⇒ 旧框选区越界，说清并让它重框 ✓
        return {"ok": False, "score": None, "mode": mode,
                "why": "框选区域超出当前画面（%d×%d）—— 画面尺寸变过，重框一次" % (fw_, fh_)}
    loc = locate(frame[y:y + h, x:x + w], terrain.canvas, mode)
    if loc is None:
        return {"ok": False, "score": None, "mode": mode,
                "why": "匹配不上 —— 可能框大了（含血条/聊天栏）、面板被游戏 UI 挡住，"
                       "或「显示方式」选错了"}
    sc = float(loc["score"])
    return {"ok": sc >= ok_score, "score": sc, "mode": mode,
            "why": "" if sc >= ok_score else "偏低（%.1f 以上才算对上）" % ok_score}


def _fmt_score(v):
    """分数 → 人话（`None` = 一次都没量出来 ✓）。"量出来"是 0~1 的小数 ✓。"""
    return "没量出" if v is None else "%.3f" % float(v)


def dump_diag(panel, canvas, calib=None, tag="x", extra=None, dir=None):
    """把"这次量不出来的**现场**"存一份 ⇒ 返回目录（失败返回 `""` ✓ 绝不影响调用方 ✓）。

    ⚠⚠ 为什么非要它（2026-10-09 ✓ 用户原话："这些提示是不正确的，因为我的手动标定对齐地图
      已经接近完美了"）：失败时界面只给一句**结论 + 一串"可能"** ✗ ⇒ 人看不了、我也查不了 ✗，
      只能猜（"框多了？压糊了？底图不对？"✓）—— 而这三件事**看一眼图就分得清** ✓
      （和鼠标标定那一晚的教训同一条：**先看图，别猜** ✗ 见 SKILL ✓）。
    存：`<tag>_panel.png`（喂进去的那块面板 ✓）/ `<tag>_canvas.png`（底图 ✓）/
        `<tag>_meta.json`（面板与底图尺寸、当时的几何、附带信息 ✓）。
    """
    try:
        import json
        import time
        from pathlib import Path

        import cv2

        # ⚠ `dir` 是给**自检**用的（临时目录 ✓）：不传就写 `data/mmap_diag` ✓ ——
        #   自检**不许**去删用户的现场 ✗（2026-10-09 踩过：用例 `rmtree` 把现场删了 ✗）。
        d = Path(dir) if dir else (Path("data") / "mmap_diag")
        d.mkdir(parents=True, exist_ok=True)
        if panel is not None:
            cv2.imwrite(str(d / ("%s_panel.png" % tag)), panel)
        if canvas is not None:
            cv2.imwrite(str(d / ("%s_canvas.png" % tag)), canvas)
        meta = {
            "tag": tag,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "panel_wh": (None if panel is None
                         else [int(panel.shape[1]), int(panel.shape[0])]),
            "canvas_wh": (None if canvas is None
                          else [int(canvas.shape[1]), int(canvas.shape[0])]),
            "calib": (dict(calib) if isinstance(calib, dict) else None),
            "extra": (extra or {}),
        }
        # ⭐ 叠图 / 差分也存一份（用户 2026-10-09 ✓）：**人眼判"偏了多少"这条路要留着** ✓
        #   —— 逐像素匹配分在这张图上天生够不着门槛 ✗（见 `calib_overlap` 的说明 ✓），
        #   而"把底图按这份几何投到面板上叠起来看"对**人**来说是最好用的判据 ✓。
        if panel is not None and canvas is not None and isinstance(calib, dict):
            try:
                warped = map_calib.warp_canvas_to_panel(canvas, calib,
                                              (panel.shape[1], panel.shape[0]))
                if warped is not None:
                    cv2.imwrite(str(d / ("%s_overlay.png" % tag)),
                                cv2.addWeighted(panel, 0.55, warped, 0.45, 0))
                    dg = cv2.cvtColor(
                        cv2.absdiff(cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)),
                        cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(str(d / ("%s_diff.png" % tag)),
                                cv2.applyColorMap(cv2.convertScaleAbs(dg, alpha=3.0),
                                                  cv2.COLORMAP_JET))
                    meta["overlay"] = True
            except Exception:                        # noqa: BLE001 —— 叠图失败不影响取证 ✓
                pass
        (d / ("%s_meta.json" % tag)).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        return str(d)
    except Exception:                                # noqa: BLE001 —— 取证坏了别影响判断 ✓
        return ""


def check_calib(panel, terrain, calib, mode=None, min_score=map_const.LOCATE_MIN_SCORE,
                ok_world_px=10.0):
    """量一遍「**当前这份标定到底差多少**」—— 拿真帧，一律报**世界像素**。

    用户 2026-09-27 要求"先把全局小地图弄准"，而**在此之前根本问不出这个数** ✗：
      · `score` 是无量纲的"像不像"（模板匹配才有，两点/手工标定一律没有）；
      · `resid_px` / `axis_gap_pct` 只长在两点法里、而且是**面板像素**；
      · `tools/mmap_dot_probe.py` 量的是**黄点认不认得出来**（识别率），不是标定误差。
    ⇒ 这里补上那把尺子。用户决策④：容差按像素定值、**表述一律世界坐标** ✓。

    **怎么量**：模板匹配当**独立的**尺子 —— 同一张真帧，把底图压进去另解一份几何
    （`locate` → `refine_fit` ✓），再看"**同一个面板像素，两套几何映射到世界差多少**"，
    四个角都算、报最坏的那个 ✓。尺子自己的可信度一并报出来（`score` / `px_per_world` /
    `refined`）：不报的话，人会把"尺子抖"当成"标定偏" ✗。

    ⚠ **只报数、不改任何东西**（核对不是标定 —— 要改走「双点标定」✓）。
    ⚠ `mode` 缺省取标定里那份；**crop 的尺子仍是粗搜**（`locate_crop` 没细化，用户明确
      把 crop 的自动标定暂缓 ✓）⇒ 那种情况下 `refined=False`，读数时要心里有数 ✓。
    ⚠ `ok_world_px` 默认 **10** = 和「坐标对齐误差范围」（`jobs.ALIGN_TOL_PX`）**同一把
      尺**：寻路判据能容忍的偏差，标定就不该更差 ✓（不是这里新拍的数 ✓）。

    返回（除带 `_px` 的，其余都是**世界像素**）：
      ok / why / verdict（人话，界面/命令行直接用）/ score / refined
      err_world（四角最大偏差）/ err_world_x / err_world_y（那个角的分量）
      err_px（换成**实时面板像素** —— 现场最直观 ✓）/ worst_panel（那个角的面板像素）
      px_per_world（1 面板像素 = 多少世界像素）
      scale_cur / scale_ref / scale_delta_pct、offset_cur / offset_ref
    """
    #: 四角的名字（只为了把"最坏在哪个角"说成人话 ✓）
    _corner_name = {(0, 0): "左上", (1, 0): "右上", (0, 1): "左下", (1, 1): "右下"}
    base = {"ok": False, "why": "", "verdict": "", "score": None,
            "refined": False, "err_world": None, "err_world_x": None,
            "err_world_y": None, "err_px": None, "worst_panel": None,
            "px_per_world": None, "scale_cur": None, "scale_ref": None,
            "scale_delta_pct": None, "offset_cur": None, "offset_ref": None,
            # ⭐ 量不出来时给"能查的东西"（2026-10-09 ✓ 见下面那段说明 ✓）：
            #   `raw_scores` = 两种方式**真实**的匹配分（`min_score=0` 量的 ✓ 不许提前判死 ✗）；
            #   `hint` = 一段人话（界面把它放进详情行 ✓）；`panel_wh` / `canvas_wh` = 尺寸对账 ✓。
            "raw_scores": None, "hint": "", "panel_wh": None, "canvas_wh": None,
            # ⭐ 「匹配尺子不适用」时的**兜底判据**（地形重合 ✓ 见 `calib_overlap` / `overlap_verdict` ✓）
            "overlap": None, "overlap_text": ""}
    if panel is None or terrain is None or getattr(terrain, "canvas", None) is None:
        return dict(base, why="没有画面 / 这张图还没有底图（先「生成地形图」）")
    if not calib or not map_calib.has_geometry(calib):
        return dict(base, why="这份标定还没有几何 —— 先标一次（双点标定 / 自动定位）")
    mode = mode or calib.get("mode") or map_const.MODE_FIT
    ref = locate(panel, terrain.canvas, mode)
    if ref is None:
        # ⭐⭐ **"量不出来"要给能查的东西，不是一串"可能"** ✗（用户 2026-10-09 ✓ 原话：
        #   "**这些提示是不正确的**，因为我的手动标定对齐地图已经接近完美了"）。
        #   旧文案把原因一律推给"没框全 / 被挡住 / 显示方式选错" ✗ ⇒ 三件事全对的人也被告知
        #   是他的错 ✗。事实是：
        #     · 这里量的是「**面板像素 vs 底图像素长得像不像**」（模板匹配 ✓）；
        #     · 手工对齐量的是「**人眼看叠图重合**」✓ —— 两者**可以不一致** ✓
        #       （手工那份连 `score` 都没有：`src=manual` ⇒ 0 ✓ 见 `save_calib` ✓）；
        #   ⇒ 所以先把**事实**说清（两种方式各自的真实分数 + 门槛 ✓），再列可查的项 ✓，
        #     并把现场交给调用方落盘（`dump_diag` ✓）。
        raw = {}
        try:
            _r = locate_fit(panel, terrain.canvas, min_score=0.0)
            raw["fit"] = (None if _r is None else round(float(_r["score"]), 3))
        except Exception:                            # noqa: BLE001 —— 诊断不许把核对弄崩 ✓
            raw["fit"] = None
        try:
            _r = locate_crop(panel, terrain.canvas, min_score=0.0,
                             seed=(calib if mode == map_const.MODE_CROP else None))
            raw["crop"] = (None if _r is None else round(float(_r["score"]), 3))
        except Exception:                            # noqa: BLE001
            raw["crop"] = None
        # ⭐⭐ **兜底：改用"地形重合率"**（见 `calib_overlap` 的说明 ✓）—— 这是回答
        #   "标定准不准"的那条路 ✓；匹配分低只说明"这把尺子在本图不适用"✗，不许当结论 ✗。
        try:
            ov = map_calib.calib_overlap(panel, terrain, calib)
        except Exception:                            # noqa: BLE001 —— 诊断不许把核对弄崩 ✓
            ov = None
        _ow = map_calib.overlap_verdict(ov)
        return dict(
            base,
            overlap=ov,
            overlap_text=_ow,
            why=("模板匹配没对上面板（全局 %s / 局部 %s，门槛 %.2f）—— 这条尺子在本图不适用，"
                 "**改看重合率**：%s"
                 % (_fmt_score(raw["fit"]), _fmt_score(raw["crop"]), min_score,
                    _ow or "重合率也算不出来（面板/底图缺 ✓）")),
            raw_scores=raw,
            panel_wh=[int(panel.shape[1]), int(panel.shape[0])],
            canvas_wh=[int(terrain.canvas.shape[1]), int(terrain.canvas.shape[0])],
            hint=("⚠ **这不代表你的手工对齐错了** ✗ —— 手工对齐量的是「人眼看叠图重合」✓，"
                  "这里量的是「面板像素和底图像素像不像」✓（模板匹配），**两件事可以不一致** ✓"
                  "（手工那份本来就没有匹配分 ✓）。\n"
                  "按分数往下看：两种都很低 ⇒ 多半是「像素不像」：来源画质（H.264 压糊 ✗）/ "
                  "面板里混进了非小地图的像素（框多了、被 UI 挡了一角 ✗）/ 底图不是这张图的；\n"
                  "只有**当前这种**低、另一种挺高 ⇒ 是「显示方式」（全局/局部小地图）选错了 ✓。\n"
                  "⇒ 现场已存 `data/mmap_diag/`（panel.png ＋ canvas.png ＋ 那份几何 ✓）："
                  "**先看图再定** ✓（别再猜 ✗）。"))

    fh, fw = panel.shape[:2]
    ppw = float(terrain.px_per_world) or 1.0
    worst = None                                  # (世界误差, 角名, dx, dy, (px,py))
    for kx in (0, 1):
        for ky in (0, 1):
            px, py = float(kx * fw), float(ky * fh)
            ax, ay = map_calib.panel_to_world(px, py, calib, terrain)
            bx, by = map_calib.panel_to_world(px, py, ref, terrain)
            dx, dy = float(bx - ax), float(by - ay)
            d = float(np.hypot(dx, dy))
            if worst is None or d > worst[0]:
                worst = (d, _corner_name[(kx, ky)], dx, dy, (px, py))
    err, cname, dx, dy, wpt = worst
    sx_c, _sy_c = map_calib.scales_of(calib)
    sx_r, _sy_r = map_calib.scales_of(ref)
    err_px = err / ppw
    dlt = None if not sx_c else (sx_r - sx_c) / sx_c * 100.0
    # 「主要差在哪个方向」：分量差值说话（只报数不说话，人还得自己算一遍 ✗）
    axis = "x" if abs(dx) >= abs(dy) else "y"
    # ⚠ **尺子自己也可能是错的**：匹配分低（画面里混着游戏 UI / 面板被挡 / 显示方式选错）
    #   时上面那套搜索只是"矮子里拔高个" ⇒ 必须把可信度说出来，否则人会把"尺子抖"
    #   当成"标定偏"（`TRUST_SCORE` 与 `region_match_score` 的 `ok_score` 同一个口径 ✓）。
    score = float(ref["score"])
    trust = score >= map_const.TRUST_SCORE
    verdict = ("最大偏差 **%.0f 世界像素**（≈ %.1f 实时像素，%s角，主要差在 %s）；"
               "尺子匹配分 %.2f%s"
               % (err, err_px, cname, axis, score,
                  "" if trust else "（**偏低，这个数只能当参考**）"))
    ok = trust and err <= float(ok_world_px)
    # ⭐ **这一份也要给"重合率"读数**（2026-10-09 ✓）：两个按钮任何时候都该能回答
    #   "跟底图重合得怎么样" ✓ —— 尤其"尺子分低"时，重合率是**唯一还能看**的那个数 ✓
    #   （匹配分低只说明"这把尺子在本图不适用"✗，**不许**翻译成"你的画面不对" ✗）。
    try:
        ov = map_calib.calib_overlap(panel, terrain, calib)
    except Exception:                                # noqa: BLE001 —— 诊断不许把核对弄崩 ✓
        ov = None
    _ow = map_calib.overlap_verdict(ov)
    if not trust:
        why = ("尺子匹配分偏低（%.2f < %.2f）⇒ 上面那个偏差**只能当参考**。%s"
               % (score, map_const.TRUST_SCORE, _ow or "（重合率也算不出来：面板/底图缺 ✓）"))
    elif not ok:
        why = "偏差 %.0f 世界像素 > %.0f（≈1 个实时像素）" % (err, ok_world_px)
    else:
        why = ""
    return {"ok": ok,
            "why": why,
            "verdict": verdict,
            "trust": trust,
            "score": float(ref["score"]),
            "refined": bool(ref.get("refined")),
            "err_world": err, "err_world_x": dx, "err_world_y": dy,
            "err_px": err_px, "worst_panel": [wpt[0], wpt[1]],
            "px_per_world": ppw,
            "scale_cur": float(sx_c), "scale_ref": float(sx_r),
            "scale_delta_pct": dlt,
            "offset_cur": [float(v) for v in (calib.get("offset") or (0, 0))],
            "offset_ref": [float(v) for v in (ref.get("offset") or (0, 0))],
            "overlap": ov, "overlap_text": _ow}


def stream_panel(timeout=5.0):
    """从 A 机的「小地图推流」取**一帧**面板（BGR ndarray）→ `(帧, 原因)`。

    **一处实现**：工作台的「实测精度」按钮（来源选「独立推流」时要拿一帧来核对 ✓）和
    命令行工具（`tools/mmap_dot_probe.grab_stream` / `tools/mmap_calib_check`）都要它 ——
    各写一遍"连哪口 / 最多等多久 / 清了没有"迟早分叉 ✗（和 `crop_of` 同一个理由 ✓）。
    取不到时 `原因` 是人话（没配 a_host / 超时没收到 ✓），调用方直接显示 ✓。
    """
    from core.config import get                    # 本模块只在用到处局部 import ✓
    host = get("a_host")                            # link.yaml 顶层
    port = get("minimap", "port", 5003)
    if not host:
        return None, "link.yaml 里没读到 a_host（A 机地址）"
    cli = MiniMapClient(host, port=port, timeout=float(timeout)).start()
    deadline = time.time() + max(0.5, float(timeout))
    try:
        while time.time() < deadline:
            f, _t = cli.latest(clear=True)
            if f is not None:
                return f, ""
            time.sleep(0.02)
    finally:
        try:
            cli.stop()
        except Exception:                           # noqa: BLE001
            pass
    return None, ("%.0f 秒内没收到小地图推流 —— A 机的「小地图推流」开了吗？"
                  % float(timeout))


def _gray(img):
    """→ 灰度。**已经是单通道就原样返回**（2026-10-18：补边后的底图是灰度的，
    再喂给 `cvtColor(BGR2GRAY)` 会当场 `Bad number of channels` ✗）。"""
    return img if getattr(img, "ndim", 3) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




def screen_to_world(player, sx, sy):
    """**游戏画面坐标 → 世界坐标**（用**玩家做锚**：`画面 = 世界 − Camera` ✓）。

    为什么要有它（用户 2026-09-27："chase状态需要判定一下怪物位于的foothold集合，若不与
    玩家处于同一个，需要先下达前往任务"）：**怪只有画面坐标**（`perception.world.world_state.Mob`
    的 `x/y` = 框中心、画面像素 ✓），而查"怪站哪块 foothold / 属于哪些集合"（
    `core.mapdata.Terrain.foothold_below` + `core.zones.Zones.set_of` ✓）**要世界坐标**
    ⇒ 中间这一跳就是本函数 ✓（全仓**唯一**一处"画面 → 世界" ✓）。

    原理（`docs/寻路设计.md`）：世界 → 屏幕是 `屏幕 = 世界 − Camera`；
    ⚠ 那条公式**没有比例尺** ⇒ 它的前提就是「**画面 1 像素 = 世界 1 像素**」（推流按原始
      分辨率抓、没有缩放 ✓）。这条前提不成立时本函数只是**刻度不对** ⇒ 调用方要把推出来的
      世界坐标**留痕**（写进 note / 打点 ✓），别当精密测量用 ✗。
    Camera 不用另外找：玩家**同时**有画面坐标（`player.x` / `player.bottom`）和世界坐标
    （`player.world_x/world_y`，来自小地图黄点 ✓）⇒ `Camera = 世界 − 画面`，直接相减 ✓。
    ⚠ 口径与「黄点脚底」对齐（约定 10）：`sy` 要给**脚底**的画面 y —— 怪是
      `mob.y + mob.h / 2` ✓、玩家那头用 `player.bottom`（框底 ✓，`world_y` 也是脚底 ✓）；
      x 两边都用中心 ✓（`player.x` ↔ `world_x` ✓）。
    ⚠ 拿不到玩家的世界坐标（没定位）⇒ 返回 `None`（**不猜** ✓，调用方退回老行为 ✓）。

    ⭐⭐ **优先用「可信相机」**（`player.cam_x/cam_y`，用户 2026-09-29 任务 2 ✓）：
    上面那个"现算"的相机**每一拍都在动**，而它的两个来源一个是权威的（黄点世界坐标 ✓）、
    一个不稳（YOLO 玩家框 ⇒ 框抖 Δ ⇒ 每只怪的世界坐标同量平移 Δ ✗）。
    `PlayerTracker` 已经把"跟黄点对得上的那几拍"挑出来存在 `player.cam_x/cam_y` 上
    （口径与这里**完全一致** ✓ 见 `perception/tracker._cam_of`）⇒ 这里照用，全仓
    "画面 → 世界"因此走**同一份相机** ✓。
    ⚠ 没有它（`None` ⇒ 还没建立）⇒ **老口径现算**，行为一字不变 ✓。
    """
    wx0 = getattr(player, "world_x", None)
    wy0 = getattr(player, "world_y", None)
    if wx0 is None or wy0 is None:
        return None
    # ⭐ **脚底偏移**（用户 2026-09-28 ✓）：框底不一定正好压在脚底（鞋底阴影 / 披风 /
    #   特效会让框多出一截 ✗）⇒ 用镜像值补正 ✓（`live_thread` 每帧灌进 `world_state` ✓
    #   —— 这边**不能**反过来 import decision，会绕成循环依赖 ✗）。
    from perception.world import world_state as _ws
    cam_x = getattr(player, "cam_x", None)
    cam_y = getattr(player, "cam_y", None)
    if cam_x is None or cam_y is None:
        cam_x = float(wx0) - float(getattr(player, "x", 0.0) or 0.0)
        cam_y = (float(wy0) - (float(getattr(player, "bottom", 0.0) or 0.0)
                               + float(_ws.FOOT_OFFSET_PX)))
    return (float(sx) + float(cam_x), float(sy) + float(cam_y))


#: ⭐⭐ **「位置还在滚/刚跳」的判据**（用户 2026-10-20 ✓ 原话："在局部地图滚动时，寻路任务的
#:   执行需要推迟直至其稳定"）。
#:
#: 为什么用**速度**（世界像素/秒）而不是"一拍挪多少底图像素" ✗：
#:   · 各图**比例尺差得远**（这张图 1 底图像素 ≈ 15 世界像素，而 1920 宽的小地图基本 1:1）
#:     ⇒ 拿固定底图像素当门槛，在 1:1 的图上**正常行走**（一拍 30~50 px）就会被判成"在滚"
#:     ⇒ **人永远不动**（死锁 ✗，比乱跳还糟）；
#:   · 而"世界坐标一拍跳了多远 / 用了多久"是**跟比例尺无关**的 —— 人再怎么跑也就 200~300
#:     世界像素/秒（5 fps 下一拍 40~60 px ✓），一波跳变轻松 1000+ ✓ 分得开 ✓。
VIEW_JUMP_PX_S = 800.0
#: 显示区**一拍挪这么多底图像素**也算"在滚"（辅助判据：换层 / 大跳 / 掩码找回 ✓）。
#: ⚠ 门槛必须**很大**（80）：这是"像素"单位，取小了会在 1:1 的小地图上误伤正常跟随 ✗
VIEW_JUMP_PX = 80.0
#: 跳完之后要**静这么多秒**才算稳（5 fps 下约 3 拍 ✓ —— 够让坐标重新对齐 ✓）
VIEW_SETTLE_S = 0.6


def apply_to_player(player, loc, now=None):
    """把定位结论写进 `WorldState` 的 `Player` 那一页（**一处口径**）。

    三个字段的含义见 `perception/world/world_state.py`；`segment_id` 是**地形**的段号
    （`core/mapdata.Terrain.segment_of` 给的）—— 寻路要的就是它。

    ⭐⭐ **顺手算「局部地图稳没稳」**（用户 2026-10-20 ✓）：`world_stable` / `world_why` ——
    执行器看到 `world_stable=False` 就**先别按**（判据见 `VIEW_JUMP_PX_S` / `VIEW_SETTLE_S` ✓）。
    为什么判据落在**这一层**：只有这里同时看得到"上一拍的位置/显示区"和"这一拍的" ✓；
    交给执行器自己记，等于让它也维护一份视图状态（两处迟早不一致 ✗ 见 `PosSnapshot` 的教训 ✓）。

    ⚠ `now` 可传（测试要控时间 ✓）；缺省取 `time.monotonic()` ✓。
    """
    player.world_x = loc.get("world_x")
    player.world_y = loc.get("world_y")
    player.segment_id = loc.get("segment_id")
    # held = 这一拍没认出黄点、位置沿用上一帧（防抖窗口内）。决策层可以照用，
    # 但要能分辨出来 —— 将来做"精细落点"时这类位置不该当新鲜观测。
    player.world_held = bool(loc.get("held"))
    player.world_note = loc.get("note") or ""
    # ---- ⭐ 局部地图稳没稳 ----
    _t = float(time.monotonic() if now is None else now)
    _jump_at = float(getattr(player, "_stab_jump_at", 0.0) or 0.0)
    _why = ""
    _wx, _wy = player.world_x, player.world_y
    if _wx is None or _wy is None:
        # 这一拍根本没有世界坐标 ⇒ 这条判据**没得判** ✓（"拿不到坐标"有它自己那道闸 ✓）
        object.__setattr__(player, "_stab_world", None)
        object.__setattr__(player, "_stab_at", 0.0)
        object.__setattr__(player, "_stab_view", None)
        object.__setattr__(player, "_stab_jump_at", 0.0)
        player.world_stable = True
        player.world_why = ""
        return player
    # ① **主判据**：世界坐标一拍跳了多远 / 用了多久（跟比例尺无关 ✓）
    _pw = getattr(player, "_stab_world", None)
    _pt = float(getattr(player, "_stab_at", 0.0) or 0.0)
    if _pw is not None and _pt > 0.0:
        _dt = max(0.0, _t - _pt)
        _step = float(np.hypot(float(_wx) - _pw[0], float(_wy) - _pw[1]))
        _speed = _step / max(_dt, 1e-3)
        if _speed > VIEW_JUMP_PX_S:
            _jump_at = _t
            _why = ("位置一拍跳了 %.0f 像素（≈%.0f 像素/秒）⇒ 局部地图还在滚"
                    % (_step, _speed))
    object.__setattr__(player, "_stab_world", (float(_wx), float(_wy)))
    object.__setattr__(player, "_stab_at", _t)
    # ② **辅助判据**：显示区自己挪了一大截（换层 / 大跳 / 掩码找回 ✓；门槛很大见上面说明 ✓）
    _v = loc.get("view")
    try:
        _v = (float(_v[0]), float(_v[1])) if _v is not None else None
    except (TypeError, ValueError, IndexError):
        _v = None
    if _v is not None:
        _pv = getattr(player, "_stab_view", None)
        if _pv is not None:
            _vstep = max(abs(_v[0] - _pv[0]), abs(_v[1] - _pv[1]))
            if _vstep > VIEW_JUMP_PX:
                _jump_at = _t
                if not _why:
                    _why = ("小地图显示区一拍挪了 %.0f 像素（还在滚）" % _vstep)
        object.__setattr__(player, "_stab_view", _v)
    else:
        object.__setattr__(player, "_stab_view", None)
    object.__setattr__(player, "_stab_jump_at", _jump_at)
    _quiet = (_t - _jump_at) if _jump_at > 0.0 else 1e9
    player.world_stable = bool(_quiet >= VIEW_SETTLE_S)
    if player.world_stable:
        player.world_why = ""
    else:
        player.world_why = (_why or "位置刚跳变，等它稳住（%.1f 秒内还会再判 ✓）"
                            % VIEW_SETTLE_S)
    return player


def overlap_pearson(panel, canvas, calib, max_samples=6000, min_samples=64):
    """**按重叠区**算面板↔底图的相关 → `(score|None, overlap, why)`。

    **一处口径**：`align_score`（给人看的那行）与 `locate_crop`（自动定位挑候选）都用它 ✓
    —— 两处各写一个"像不像"的算法，迟早变成"自动定位说 0.9、判据那行说 0.4" ✗。
    采样点**落在底图里**的才参与 ⇒ 面板边上的游戏 UI / 底图外那圈自然被排除 ✓。

    ⚠ **传灰度图（2 维）也认**：自动定位要算几百次，每次现转一次灰度就是几百毫秒 ✗
      （实测 675×600 的面板转一次 ~1 ms）⇒ 调用方**自己转一次、反复用** ✓。
    """
    if panel is None or getattr(panel, "size", 0) == 0:
        return None, 0.0, "没有面板画面"
    if canvas is None or getattr(canvas, "size", 0) == 0:
        return None, 0.0, "这张图还没有底图（先「生成地形图」）"
    if not map_calib.has_geometry(calib):
        return None, 0.0, "这份标定还没有几何（缩放 / 偏移）"
    panel = panel if getattr(panel, "ndim", 3) == 2 else _gray(panel)
    canvas = canvas if getattr(canvas, "ndim", 3) == 2 else _gray(canvas)
    ch, cw = canvas.shape[:2]
    sx, sy = map_calib.scales_of(calib)
    if sx <= 0 or sy <= 0:
        return None, 0.0, "标定里的缩放不是正数"
    # ⚠ 字段口径照抄 `panel_to_canvas`（**唯一那处** ✓）：crop 才加 `view`
    ox, oy = calib.get("offset") or (0, 0)
    vx, vy = ((calib.get("view") or (0, 0)) if calib.get("mode") == map_const.MODE_CROP
              else (0, 0))
    ph, pw = panel.shape[:2]
    # 采样步长：面板可能 750×800 ⇒ 别一次算几十万像素（1 底图像素就够 ✓）
    step = max(1, int(round((pw * ph / float(max(1, max_samples))) ** 0.5)))
    xg, yg = np.meshgrid(np.arange(0, pw, step), np.arange(0, ph, step))
    bx = np.rint((xg - ox) / sx + vx).astype(np.int64)
    by = np.rint((yg - oy) / sy + vy).astype(np.int64)
    inside = (bx >= 0) & (bx < cw) & (by >= 0) & (by < ch)
    n = int(inside.sum())
    ov = float(n) / float(max(1, xg.size))
    if n < max(16, int(min_samples)):
        return None, ov, ("面板几乎没有落在底图上（只有 %.1f%% 的采样点在里面）—— "
                          "缩放 / 偏移 / 显示区起点不对？" % (ov * 100.0))
    try:
        a = panel[yg[inside], xg[inside]].astype(np.float32)
        b = canvas[by[inside], bx[inside]].astype(np.float32)
    except Exception as e:                              # noqa: BLE001
        return None, ov, "采样失败：%s" % e
    if float(a.std()) < 1e-3 or float(b.std()) < 1e-3:
        return None, ov, "面板或底图那一块几乎是纯色（没有可比的纹理）"
    sc = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(sc):
        return None, ov, "算不出相关（那一块是平的）"
    return max(0.0, min(1.0, sc)), ov, ""
