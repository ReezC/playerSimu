"""⭐「**框到可见部分**」—— 模板匹配的框自适应到目标**实际露出来的那一块**。

⭐⭐ **为什么住在这儿**（2026-10-06 从 `tools/detect_mobs.py` 搬过来 ✓）：
  这条判据要被**两条内核**共用 —— 怪物/宠物/掉落走 `tools/detect_mobs.work` ✓、
  玩家走 `perception/player_locator.PlayerLocator` ✓（用户："**所有的匹配都需要
  「框到可见部分」参数**"✓）。
  ⛔ 不能反过来让 `perception/player_locator` 去 import `tools/detect_mobs` ✗ ——
  那个模块头上就 `from gui.theme import class_color`（画可视化框用 ✓），而 `perception`
  是**运行时**（实时回路 / 决策）要 import 的东西 ⇒ 会把 gui 拖进运行时的依赖里 ✗。
  ⇒ 放到中性层 `perception/` ✓：`tools/*` 和 `perception/*` 两边都 import 得起 ✓。

# ══════════════════════════════════════════════════════════════
# ⭐⭐ 框到**可见部分**（用户 2026-10-05 ✓）
# ══════════════════════════════════════════════════════════════
# 用户原话："现在模板匹配的自动标注框不管实际漏出目标的部分有多少，都用完整贴图大小的框，
#   这个能自适应识别出的框大小吗？"
#
# **现状**（改之前 ✓）：模板 = 精灵按 alpha 裁到内容包围盒 ✓、匹配是带掩码的 ✓，可写出去的框
#   一直是**模板的完整尺寸** ✗（`work` 里 `w * ds, h * ds` ✓）⇒ 目标被挡掉一半时，框照旧把
#   **看不见的那半**也包进去 ✓（画面里只有半只怪，框却是整只 ✓）。
#
# **口径先说清（这是改"标注语义"，不是修 bug ✗）**：框住**整只**（含被挡部分）是 COCO/YOLO 的
#   通行约定 ✓（人手工也这么标 ✓）⇒ 老行为**不算错** ✓；下面这套是另一种同样正当的约定 =
#   **只框看得见的部分**（visible box ✓）⇒ **默认关** ✓（`cfg["visible"]` 缺省 False ✓
#   ⇒ 老项目 / 老调用方一字不变 ✓）；开了之后同一帧的框会变小 ⇒ 要重跑这一趟标注 ✓。
#
# **判据**（在峰值那个窗口里，逐像素问"这里**真的**是这个精灵吗" ✓）：
#   · 灰度差 ≤ `VISIBLE_TOL` ✓ **并且** · 边缘（Sobel 方向 + 强度）对得上 ✓ ⇒ 算"确实出现了" ✓；
#   ⇒ 取**最大连通块**的包围盒 = 可见部分 ✓。
#   ⚠ **不能取"全部相符像素"的包围盒** ✗ —— 零散几点会把框撑回全尺寸 ✓（用例钉住这条 ✓）。
#   ⚠ 只看灰度差不够 ✗：被挡住的区域若与精灵**亮度相近**（深色怪被深色地形挡 ✓）灰度差很小 ✓
#     ⇒ 会误判成"精灵还在" ✓；叠上**边缘**判据才稳 ✓（挡它的是别的东西 ⇒ 边缘方向/强度对不上 ✓）。
#
# **天花板（别指望它完美 ✓）**：判据靠纹理/边缘 ✓ ⇒
#   · 遮挡边界落在**有纹理的地方** ⇒ 判得准 ✓；
#   · 落在**大片纯色**上（纯色身体被挡住一角 ✓）⇒ 那里没有边缘可用 ⇒ **一律回退全尺寸** ✓
#     （这是**故意**的保守 ✓ 不是漏做 ✓）。
"""
import cv2
import numpy as np

#: 灰度差容差（0~255）：抗抗锯齿 / 半透明 / 轻微光照差 ✓
VISIBLE_TOL = 26.0
#: 模板边缘多强才算"这里有边缘可用"（Sobel 8 位尺度）；低于它的像素**不参与判断** ✓
VISIBLE_GMIN = 24.0
#: 边缘方向一致的下限（余弦 ≥ 0.5 = 夹角 ≤ 60° ✓）
VISIBLE_COS = 0.5
#: 边缘强度比的可接受区间（画面 0.5~3 倍以内的缩放/抗锯齿都算"像" ✓）
VISIBLE_RATIO_LO, VISIBLE_RATIO_HI = 0.35, 3.0
#: 缩得不足这个比例（15%）⇒ **不缩**（防像素级抖动把框忽大忽小 ✗）
VISIBLE_MIN_SHRINK = 0.15
#: 框任一边小于它 ⇒ 不缩（小框没有意义 ✓）
VISIBLE_MIN_SIDE = 6
#: 相符面积占**模板边缘像素**的比例下限 ⇒ 低于它回退全尺寸 ✓（宁可不缩 ✗）
VISIBLE_KEEP = 0.2


def visible_box(imgf, tg, tgx, tgy, mask_bool, x, y,
                tol=VISIBLE_TOL, keep=VISIBLE_KEEP):
    """这一框的**可见部分**（模板坐标系 ✓ 与传进来的 `x, y` 同一套 ✓）。

    `imgf` = 该帧的**浮点灰度图**（与 `x, y` **同一个尺度** ✓ —— 匹配若在缩略图上做，
    这里就得给那张缩略图的灰度 ✓）、`tg` = 模板灰度（**同一尺度** ✓）、
    `tgx`/`tgy` = 模板的 Sobel（每个模板算一次 ✓ 别在框里重算 ✗）、
    `mask_bool` = 模板的不透明掩码 ✓、`(x, y)` = 峰值位置（模板左上角 ✓）。

    返回 `(vx, vy, vw, vh)`（**绝对坐标**：已经加上 `x, y` ✓ —— 与传进来的同一套坐标系 ✓）；
    **判不出来 / 证据不足 / 出任何岔子 ⇒ 原样回全尺寸** ✓
    （这只是把框收小一点 ✓ 坏了就别收 ✓ —— 绝不因为它漏掉一整只 ✓）。

    ⚠ 调用方注意：**缩略图上算出来的可见框是缩略图尺度** ✗ ⇒ 要自己乘回去再写标注 ✓
    （见 `tools/detect_mobs.work` 与 `perception/player_locator.PlayerLocator.locate` ✓）。
    """
    h, w = mask_bool.shape
    full = (x, y, w, h)
    try:
        if (x < 0 or y < 0 or x + w > imgf.shape[1] or y + h > imgf.shape[0]
                or tgx is None or tgy is None or w < VISIBLE_MIN_SIDE or h < VISIBLE_MIN_SIDE):
            return full

        # 模板这边"有边缘可用"的像素（纯色部位不参与判断 ✓ 见上面那段天花板的说明 ✓）
        mg = np.sqrt(tgx * tgx + tgy * tgy)
        strong = (mg >= VISIBLE_GMIN) & mask_bool
        n_strong = int(strong.sum())
        # 边缘像素太少（纯色精灵 ✓）⇒ 判不了 ⇒ 保守回全尺寸 ✓
        if n_strong < max(16, int(int(mask_bool.sum()) * 0.02)):
            return full

        patch = imgf[y:y + h, x:x + w]
        px = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        py = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mpx = np.sqrt(px * px + py * py)
        cos = (tgx * px + tgy * py) / (mg * mpx + 1e-6)
        ratio = mpx / (mg + 1e-6)
        edge_ok = strong & (cos >= VISIBLE_COS) & (ratio >= VISIBLE_RATIO_LO) \
                         & (ratio <= VISIBLE_RATIO_HI)
        gray_ok = np.abs(patch - tg) <= tol
        ev = edge_ok & gray_ok                      # "这里确实是这个精灵" ✓
        n_ev = int(ev.sum())
        if n_ev < max(8.0, float(keep) * n_strong):  # 证据太少 ⇒ 保守 ✓
            return full

        # ⚠⚠ **先去孤立点（开运算），再取连通块** ✗➡✓：
        #   被挡那一片里总有**少数**像素碰巧"灰度也像、边缘也像"（实测 8 个 ✓）——
        #   用闭运算会把它们和可见区**连成一块** ⇒ 包围盒又变回全尺寸 ✗（实测踩过 ✓：
        #   `ev` 有 97% 落在可见区 ✓ 却因为那几个散点拿到了整幅 bbox ✓）。
        #   ⇒ 开运算（3×3 先腐蚀再膨胀）把孤立点抹掉 ✓，剩下的连通块才是"目标主体" ✓。
        ev_u8 = cv2.morphologyEx(ev.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8)) & mask_bool.astype(np.uint8)
        n_lab, lab, stats, _cent = cv2.connectedComponentsWithStats(ev_u8, 8)
        if n_lab <= 1:
            return full
        # 取**证据像素最多**的那一块 ✓（= 目标主体 ✓）。
        # ⚠ 不能按面积挑 ✗ —— 散点连起来的那一大块面积更大 ✓（实测会挑错 ✓）。
        best, best_n = -1, -1
        for i in range(1, n_lab):
            cnt = int(ev_u8[lab == i].sum())
            if cnt > best_n:
                best, best_n = i, cnt
        if best < 0 or best_n < n_ev * 0.5:         # 主体都不占一半证据 ⇒ 不稳 ⇒ 保守 ✓
            return full

        vx = int(stats[best, cv2.CC_STAT_LEFT])
        vy = int(stats[best, cv2.CC_STAT_TOP])
        vw = int(stats[best, cv2.CC_STAT_WIDTH])
        vh = int(stats[best, cv2.CC_STAT_HEIGHT])
        vx = max(0, min(vx, w - 1))                 # 夹进模板框内 ✓
        vy = max(0, min(vy, h - 1))
        vw = max(1, min(vw, w - vx))
        vh = max(1, min(vh, h - vy))
        # ⚠ 缩得不够 ⇒ 就当没缩（不然"没被挡"的框会被抗锯齿啃掉一两像素 ⇒ 忽大忽小 ✗）
        if vw > w * (1.0 - VISIBLE_MIN_SHRINK) and vh > h * (1.0 - VISIBLE_MIN_SHRINK):
            return full
        if vw < VISIBLE_MIN_SIDE or vh < VISIBLE_MIN_SIDE or vw * vh < float(keep) * (w * h):
            return full
        return (x + vx, y + vy, vw, vh)
    except Exception:                               # noqa: BLE001 —— 见上：坏了就别收 ✓
        return full
