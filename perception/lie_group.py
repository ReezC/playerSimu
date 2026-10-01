"""A 层：**群体刚体平移 `T` + 逐框残差**（重构第 1 步；用户 2026-09-30 实测选定）。

为什么是这一层（实测依据 `tools/probe_lie_rotation.py`，4 个素材）：

· 用户事实 2/3："假目标相对静止、假群只平移；只有真目标动" ⇒ 群体平移 `T` = 背景/干扰群
  的共同运动，而**真目标 = 唯一偏离 `T` 的那个** ✓；
· 实测：用 `位移 − T` 的**残差**，中位 **1.2~1.7px**、p90 4.3~6.8px、**最大 70~105px**
  ⇒ 真目标突出 **40~60 倍** ✓✓（当前最干净的身份/位置信号）；
· ⚠ 两条实测修正（都踩过 ✓）：
  **① 模板必须从"上一帧自己的框"裁** ✗ —— 拿当前帧框位置去上一帧裁（错位一个位移 ~20px）
  会让峰中位掉到 **0.17**、只有 17% 能过门槛 ✗（改成用上一帧框 ⇒ 见下 ✓）；
  **② 别用中位/均值汇总** ✗ —— 背景是**同形状纹理** ⇒ 有个别框会跳到隔壁同形状上（70px 级
  外点 ✗）⇒ 中位被拽走；改成**二维直方图取最密簇（众数/共识）** ✓。

对外：`GroupMotion.estimate(cur_bgr, boxes, typ_area)`（内部记住上一帧 ✓ 逐帧调用即可 ✓）。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

_SEARCH = 16.0            # 先验搜索窗半径（px）：先验 = 上一帧 T ⇒ 再留 ±16px ✓（**必须小**：
                          # 大了隔壁同形状的"邻居峰"就进来了 ✗ 实测）
_FALLBACK_SEARCH = 26.0   # 没有先验（首帧/刚重置）时的搜索半径。⚠ **必须小**：素材背景是
                          #   **周期性纹理** ⇒ 窗口一大就跳到隔壁一"格"（实测：真值 (9,3)
                          #   跳到 (9−48,3−48) ✗✗ 因为格子周期 48px ✓）⇒ 26px 既够实际
                          #   位移（~20~25px/帧 ✓）又把跳格挡在外面 ✓；帧率/间隔大时由调用方
                          #   用 `search` 参数显式放大 ✓（别默认放大 ✗）
_PAD = 4                  # 裁模板时外扩（多点上下文 ⇒ 峰更尖 ✓）
_MIN_PEAK = 0.45          # 相关峰下限（实测：用对模板后中位 ~0.7+ ✓；0.45 留足余量 ✓）
_BIN = 2.0                # 共识直方图格宽（px ✓）
_MIN_KEEP = 4             # 至少这么多框同簇才认 T ✓（少了没有"群体"可言 ✗）
_MERGE_RATIO = 1.4        # 面积 > 标准 × 它 ⇒ 并集块 ⇒ 不进群体统计 ✗（事实 5）
_PAIR_GATE = 34.0         # 「本帧框 ↔ 上一帧框预测位置」配对门（px ✓）
_CONSIST_SEARCH = 6.0     # 群体一致性检验时允许的**局部对齐**半径（px ✓）：检测框中心有
                          #   ±几 px 抖动 ✗ ⇒ 硬对齐没用（实测 0.63 ✗ 真假不分 ✗）
_CONSIST_LOW = 0.90       # 群体一致性低于它 ⇒ 这个框"没跟着群体走" ⇒ 可疑 ✓（实测见
                          #   `tools/probe_group_motion.py` ② 的数字，按实测定门槛 ✓）


def _patch(im, cx, cy, w, h, pad=_PAD):
    x0 = max(0, int(round(cx - w / 2)) - pad)
    y0 = max(0, int(round(cy - h / 2)) - pad)
    x1 = min(im.shape[1], int(round(cx + w / 2)) + pad)
    y1 = min(im.shape[0], int(round(cy + h / 2)) + pad)
    return im[y0:y1, x0:x1], (x0, y0)


def box_shift(prev_g, cur_g, box, prior=None, search=None):
    """**上一帧第 i 个框** ⇒ 它在当前帧的位移 `(dx, dy, peak)`；不可信 ⇒ None。

    模板取自 `prev_g` 的**该框位置** ✓（不是当前框位置 ✗ 实测踩过）。
    `prior`（上一帧 T）把搜索窗压在它附近 ⇒ 同形状邻居进不来 ✓。
    """
    cx, cy, w, h = box
    tpl, (tx, ty) = _patch(prev_g, cx, cy, w, h)
    if tpl.size == 0 or min(tpl.shape[:2]) < 6:
        return None
    r = float(search if search is not None else (_SEARCH if prior else _FALLBACK_SEARCH))
    px = 0.0 if prior is None else float(prior[0])
    py = 0.0 if prior is None else float(prior[1])
    x0 = int(max(0, round(tx + px - r)))
    y0 = int(max(0, round(ty + py - r)))
    x1 = int(min(cur_g.shape[1], round(tx + tpl.shape[1] + px + r)))
    y1 = int(min(cur_g.shape[0], round(ty + tpl.shape[0] + py + r)))
    if x1 - x0 < tpl.shape[1] or y1 - y0 < tpl.shape[0]:
        return None                                   # 贴边 ⇒ 窗装不下，跳过 ✓
    res = cv2.matchTemplate(cur_g[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    if peak < _MIN_PEAK:
        return None
    ix, iy = loc
    dx, dy = float(ix), float(iy)
    if 0 < ix < res.shape[1] - 1:                     # 抛物线亚像素 ✓
        a, b, c = float(res[iy, ix - 1]), float(res[iy, ix]), float(res[iy, ix + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-6:
            dx += 0.5 * (a - c) / den
    if 0 < iy < res.shape[0] - 1:
        a, b, c = float(res[iy - 1, ix]), float(res[iy, ix]), float(res[iy + 1, ix])
        den = a - 2 * b + c
        if abs(den) > 1e-6:
            dy += 0.5 * (a - c) / den
    return (x0 + dx - tx, y0 + dy - ty, float(peak))


def consensus(shifts, bin_size=_BIN, min_keep=_MIN_KEEP):
    """位移集合 ⇒ **最密那一簇**（众数）⇒ `(T, 内点索引)`；太散 ⇒ `(None, [])`。

    个别框跳到隔壁同形状（70px 级外点 ✗）时，中位/均值会垮 ✗；二维直方图取峰值 =
    只有**真在一起动**的那一簇才算群体 ✓。
    """
    if len(shifts) < min_keep:
        return None, []
    pts = np.asarray([(s[0], s[1]) for s in shifts], dtype=np.float64)
    wts = np.asarray([max(0.1, s[2]) for s in shifts], dtype=np.float64)
    lo = pts.min(axis=0) - bin_size
    nx = max(2, int(math.ceil((pts[:, 0].max() - lo[0]) / bin_size)) + 1)
    ny = max(2, int(math.ceil((pts[:, 1].max() - lo[1]) / bin_size)) + 1)
    hist = np.zeros((ny, nx), dtype=np.float64)
    ix = np.clip(((pts[:, 0] - lo[0]) / bin_size).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - lo[1]) / bin_size).astype(int), 0, ny - 1)
    np.add.at(hist, (iy, ix), wts)
    sm = cv2.GaussianBlur(hist, (3, 3), 0.8)
    my, mx = np.unravel_index(int(np.argmax(sm)), sm.shape)
    c0 = (lo[0] + (mx + 0.5) * bin_size, lo[1] + (my + 0.5) * bin_size)
    d = np.hypot(pts[:, 0] - c0[0], pts[:, 1] - c0[1])
    keep = np.nonzero(d <= 1.5 * bin_size)[0]
    if len(keep) < min_keep:
        return None, []
    t = (float(np.average(pts[keep, 0], weights=wts[keep])),
         float(np.average(pts[keep, 1], weights=wts[keep])))
    return t, [int(i) for i in keep]


def box_consist(prev_g, cur_g, box, t, search=_CONSIST_SEARCH):
    """**群体一致性**：本帧框的像素块 ↔ 上一帧"同位置**减去 T**"的块，最佳归一化相关 ✓。

    · 随群体平移的框（假目标 ✓）：内容就是平移过来的 ⇒ 相关高 ✓；
    · **真目标**：它这一帧的位置，上一帧（减去 T 后）是**背景** ✗ ⇒ 相关低 ✓✓
      —— ⭐ 比"用上一帧自己的框当模板去追"强得多 ✗（真目标一移动，它上一帧的位置就空了
      ⇒ 那个模板匹配到的是**背景**，位移看起来跟群体一样 ⇒ 残差 0、追不到它 ✗ 实测）；
    · **合并块**（事实 5）：多进来一块内容 ⇒ 相关低 ⇒ 一并被标出来 ✓。

    ⚠⚠ **必须允许一个小范围局部对齐**（`search`）：检测框中心本身有 **±几 px 抖动** ✗
    ⇒ 硬按 `c − T` 对（第一版 ✓）会连"随群体平移的框"都只有 **0.63** ✗✗ 完全不分真假 ✗。
    用一次 `matchTemplate` 在 ±`search` 内取**最佳**相关 ✓（顺带给出该框的**自身残差位移** ✓）。
    ⇒ 返回 `(最相关, 该框相对群体的位移 (dx,dy))`；不可用 ⇒ None。
    """
    cx, cy, w, h = box
    tpl, (tx, ty) = _patch(prev_g, cx - float(t[0]), cy - float(t[1]), w, h, pad=0)
    if tpl.size == 0 or min(tpl.shape[:2]) < 8:
        return None
    # ⚠⚠ 搜索窗要**以"预期当前位置"为中心**（= 模板位置 + T ✓），不是以模板位置为中心 ✗
    #   —— 第一版居中了模板位置 ⇒ 一直在"上一帧那块"附近找 ⇒ 连随群体平移的框都只有
    #   **0.11** ✗✗（第二版 0.63 ✗ 同因）⇒ 完全不分真假 ✗。
    ex, ey = tx + float(t[0]), ty + float(t[1])
    x0 = int(max(0, round(ex - search)))
    y0 = int(max(0, round(ey - search)))
    x1 = int(min(cur_g.shape[1], round(ex + tpl.shape[1] + search)))
    y1 = int(min(cur_g.shape[0], round(ey + tpl.shape[0] + search)))
    if x1 - x0 < tpl.shape[1] or y1 - y0 < tpl.shape[0]:
        return None
    res = cv2.matchTemplate(cur_g[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    # 该框**自身位移** = 匹配到的原点 − 模板原点；再减去 T ⇒ 相对群体的残差 ✓
    return (float(peak), (x0 + loc[0] - tx - float(t[0]),
                          y0 + loc[1] - ty - float(t[1])))


class GroupMotion:
    """群体平移估计器（**有状态**：记住上一帧灰度与框 ✓ 只为了"先验 + 正确裁模板"）。"""

    def __init__(self):
        self.t = None
        self.prev_g = None
        self.prev_boxes = None
        self.last = None          # 上一帧的返回（诊断/下游兜底 ✓）

    def reset(self):
        self.t = None
        self.prev_g = None
        self.prev_boxes = None
        self.last = None

    def estimate(self, cur_bgr, boxes, typ_area=0.0, cur_gray=None):
        """逐帧调用。⇒ `dict`：

        · `t`     群体平移 `(dx,dy)`（共识 ✓）；数据不够 ⇒ None
        · `med`   朴素中位（诊断对照 ✓）
        · `used`  参与共识的框数；`n_matched` 匹配成功的框数
        · `resid` `{本帧框序号: 残差 px}`（= 该框相对群体的偏离 ✓ **真目标在此突出** ✓）
        · `dev_i` 残差最大的那个**本帧框序号**（= 最可疑的真目标 ✓）；`dev_pos` 它的预测位置 ✓
        """
        g = cur_gray if cur_gray is not None else cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
        out = {"t": None, "med": None, "used": 0, "n_matched": 0,
               "resid": {}, "self_resid": {},
               "consist": {}, "consist_min": None, "consist_med": None,
               "dev_i": None, "dev_pos": None, "spread": 0.0}
        boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                 for b in (boxes or [])]
        if self.prev_g is None or self.prev_g.shape != g.shape or not self.prev_boxes:
            self.prev_g, self.prev_boxes, self.last = g, boxes, out
            return out
        typ = float(typ_area or 0.0)
        # ---- 每个"上一帧框"求位移（并集块不参与群体统计 ✗）----
        shifts, base = [], []
        for b in self.prev_boxes:
            if typ > 0 and b[2] * b[3] > _MERGE_RATIO * typ:
                continue
            mv = box_shift(self.prev_g, g, b, prior=self.t)
            if mv is not None:
                shifts.append(mv)
                base.append(b)
        out["n_matched"] = len(shifts)
        if shifts:
            t, keep = consensus(shifts)
            out["med"] = (float(np.median([s[0] for s in shifts])),
                          float(np.median([s[1] for s in shifts])))
            if t is None:                              # 共识失败 ⇒ 退回中位（有比没有强 ✓）
                t = out["med"]
                keep = list(range(len(shifts)))
            out["t"] = t
            out["used"] = len(keep)
            # ---- 把"上一帧框"的位移配到"本帧框"（预测位置最近 ✓）⇒ 给出本帧的残差 ----
            preds = [(base[k][0] + shifts[k][0], base[k][1] + shifts[k][1],
                      math.hypot(shifts[k][0] - t[0], shifts[k][1] - t[1]))
                     for k in range(len(shifts))]
            best_dev, best_i, best_pos = -1.0, None, None
            for j, cb in enumerate(boxes):
                bi, bd = None, None
                for k, (px, py, dev) in enumerate(preds):
                    d = math.hypot(cb[0] - px, cb[1] - py)
                    if d <= _PAIR_GATE and (bd is None or d < bd):
                        bi, bd = k, d
                if bi is None:
                    continue
                dev = preds[bi][2]
                out["resid"][j] = dev
                if dev > best_dev:
                    best_dev, best_i, best_pos = dev, j, (preds[bi][0], preds[bi][1])
            out["dev_i"], out["dev_pos"] = best_i, best_pos
            if out["resid"]:
                out["spread"] = float(np.median(list(out["resid"].values())))
            # ---- **群体一致性**（身份/事件的主信号 ✓）：谁没跟着群体走 = 谁可疑 ✓ ----
            consist = {}
            for j, cb in enumerate(boxes):
                if typ > 0 and cb[2] * cb[3] > _MERGE_RATIO * typ:
                    continue                               # 并集块另行处理（事实 5）✓
                s = box_consist(self.prev_g, g, cb, t)
                if s is not None:
                    consist[j] = s[0]                     # 最佳相关 ✓
                    out["self_resid"][j] = s[1]           # 它相对群体的**自身位移** ✓
            out["consist"] = consist
            if consist:
                worst = min(consist, key=consist.get)
                out["consist_min"] = consist[worst]
                out["consist_med"] = float(np.median(list(consist.values())))
                # `dev_i`：优先给"一致性最差"的那个（= 真目标/合并/新出现 ✓）；
                # 没有一致性数据时退回"残差最大"（诊断兜底 ✓）
                if consist[worst] < _CONSIST_LOW:
                    out["dev_i"] = worst
                else:
                    out["dev_i"] = None
        self.prev_g, self.prev_boxes, self.last = g, boxes, out
        return out
