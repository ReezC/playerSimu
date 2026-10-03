# -*- coding: utf-8 -*-
"""**运动不合群式**测谎跟踪（用户 2026-10-03 ✓ 三次方案转向的终点 ✓ 都记在下面 ✗ 免得再走一遍）。

用户给的**确定规律**（2026-10-03 原话，逐条）：
  ① 出现"3 秒后…"那个窗口 ⇒ 数秒后开始（本模块没用它 ✓ 留给触发层 ✓）；
  ② 出现正片 ⇒ 开始实时追踪；**一开始依旧用白色图形展示真目标**；
  ③ **假目标钉死在背景上**（相对静止）+ **相机纯平移**缓速平滑随机移动 +
     **"波浪式滤镜"= 画面几何被扭曲**（像隔着水波看 ✓ 用户后来澄清，**不是亮度噪声** ✓）；
  ④ 真目标**相对假目标群体**运动（非匀速、会随机变向）⇒ 与假目标**重叠再分离后**仍要靠
     **运动方式**辨别出来继续跟踪；
  ⑤ **"还是需要依赖 yolo，无论真假至少把目标定位出来"** ✓（用户 2026-10-03 追加）；
  ⑥ **"只有背景有有色，目标都是透明的，但可以看出类似玻璃的轮廓"** ✓（同一天追加 ✓ 关键 ✗）。

────────────────────────────────────────────────────────────────────────
**五次方案（前四条都作废 ✗ 只有最后一条立住 ✓）**
  1. "减掉全局运动 ⇒ 真目标露出"✗ —— 相位相关对齐 + 帧间差分。实测补偿后残差只从 38 降到
     **17.8**（一半都补不掉 ✗）⇒ 前景遍地亮块 ✗。用户确认病根：**波浪是几何扭曲** ⇒
     **"整幅图对齐"这个前提本身不成立** ✗。
  2. "观测用颜色、预测用运动"✗ —— 找最白块（对亮度免疫 ✓ 白星期实测准到 **0~1.4px** ✓）。
     ⚠ 但它**只覆盖开局**：目标**逐渐透明** ⇒ 白度消失 ⇒ 掩膜只剩零碎噪点 ⇒ 无观测 ✗
     （用户第 ⑥ 条坐实了这点：**透明到只剩"玻璃轮廓"** ✗）。
  3. "全图找最强环（玻璃轮廓）"✗ —— 环形显著性 = 圆周梯度 − 内外侧梯度。实测逐帧跳
     **29~305px** ✗ 完全不连续：**背景纹理里的环比目标强**，把目标淹了 ✗。
  4. "框内外不一致度"✗ —— 猜"玻璃 ⇒ 框内纹理被抹平" ⇒ 比框内 vs 周边环带的梯度/对比度。
     实测所有框都只有 **0.05 / 0.02 / 0.00** ✗ 区分度≈0（开局目标是**实的白星**不是玻璃 ✗）。
  ⇒ **5.（现行）yolo 给候选 + "运动不合群"分辨** ✓✓：
     · **yolo 全程检得到**（实测透明期各帧 **15~18 个 cls0 @ conf 0.98** ✓ 用户第 ⑤ 条成立 ✓）
       —— 但它**分不出真假** ✗（尺寸形状都一样 ✓ 用户自己也说"无论真假" ✓）⇒ **它只负责定位** ✓；
     · **真目标 = "运动不合群"的那个** ✓（用户第 ④ 条 ✓）：假目标群体的位移 = **相机平移**
       （同一份 ✓）⇒ 真目标**相对群体**在动 ⇒ 它的位移**偏离群体中位** ✓；
     · ⭐⭐ **为什么波浪毁不掉它** ✓✓：波浪对**全场一视同仁**（同一场波搅所有框 ✓）⇒
       **减掉"同一拍所有框的中位位移"就抵消了** ✓✓ —— 这跟方案 1 的死因**恰好相反**：
       方案 1 对齐的是**整幅图**（波浪一扭曲就对不上 ✗），这里对齐的是**同拍所有框的中位** ✓。

**决定性实测（`10月1日.mp4` 全片 ✓ 探针一次跑完 ✓）**：
  · **噪声底**（假目标那批的"相对群体偏离"中位）= 只有 **3.3~6.5 px** ✓✓
    —— 相机平移 + **波浪抖动** + 配对误差**全在里面** ⇒ 波浪的贡献被抵消得干干净净 ✓；
  · **信号**（每帧 top1 的偏离）= **8~26 px** ⇒ **信噪比 1.6~5.1** ✓；
  · **校准**（唯一有真值的时刻 ✓）：帧 1 白星真值 `(373,251)` ⇒ 帧 6 判据 top1 = **`(373,250)`**
    ✓✓ 完全吻合（信噪比 **5.12** ✓）；帧 6→12→18 的 top1 轨迹 `(373,250)→(354,269)→(317,263)`
    **连续** ✓；
  · ⚠ 探针里"累积偏离"按**框索引**记 ⇒ 索引每帧会变 ⇒ 后面串位（跳到画面另一侧 ✗）——
    **是实现 bug 不是判据问题** ✓ ⇒ 本模块改成**按位置关联的轻量轨迹表** ✓（见下 ✓）。

**每拍流程**（本模块 ✓）：
  ① **配对**：yolo 的框 ↔ 已有轨迹（用轨迹的**预测位置** ✓ 最近邻 + `pair_gate` ✓ **一对一** ✓）；
  ② **群体中位位移**：所有配上的轨迹的"本拍观测量 − 上一拍观测量"取中位 ✓（**≥ `_MIN_PAIRS` 个才算** ✓）；
  ③ **偏离**：`dev = |该框位移 − 群体中位位移|` ✓；
  ④ **累积**：`score = score × score_decay + dev`（**带衰减 ⇒ 追"持续不合群" ✗ 不追"偶然跳一下"** ✓）；
     另加**白度辅助分**（用户第 ② 条 ✓）：`pick_white` 找到的白块附近的轨迹 `+ white_w × _WHITE_HIT`
     ⇒ **开局几条拍就有明确答案** ✓（透明后白度没了 ⇒ 该分自然归零 ✓ 不干扰 ✓）；
  ⑤ **淘汰**：`_TRACK_LOST` 拍没配上的轨迹删掉 ✓；没配上的框**新建**轨迹（score=0 ✓）；
  ⑥ **选目标**：`score` 最高、且 `hits ≥ min_hits` ⇒ 当选 ✓；**切换有迟滞** ✓
     （新候选要超过当前目标 `+ switch_margin` 才换 ⇒ 不抖 ✗）；
  ⑦ **报告**：位置 = 目标轨迹的位置（带平滑 ✓）；速度 = 轨迹速度 ✓；半径由框尺寸学 ✓
     （之后冻结 ✓）；⚠ 目标轨迹**本拍没配上**时 ⇒ 用"上一拍位置 + 速度"外推照给 ✓
     （用户第 ④ 条要的"重叠再分离后**继续跟踪**" ✓）。

⚠ **本模块不做**：相位相关 ✗ / 帧间差分 ✗ / 全图环检测 ✗ / 框内纹理一致性 ✗（前四条都实测作废 ✓）。
⚠ `pick_white` **保留** ✓ 但**只当"开局辅助证据"** ✗ 不再是唯一观测 ✓。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from perception.lie_controller import LieMouseController

# ---------------- 跟踪参数（前 5 个在界面上可调 ✓ 见 `lie_demo` 第 4 行 ✓）----------------
#: **配对门限**（px ✓）：本拍框与轨迹预测位置超过它就配不上 ✓（相机一帧能走几十 px ⇒ 别太小 ✗）。
_PAIR_GATE = 70.0
#: **偏离累积的每拍衰减**（0.90 ✓）：追"**持续**不合群" ✗ 不追"偶然跳一下" ✓。
_SCORE_DECAY = 0.90
#: **切换目标的迟滞余量**（分 ✓）：新候选要超过当前目标这么多分才换 ⇒ 不来回跳 ✓。
_SWITCH_MARGIN = 8.0
#: 轨迹至少被关联上这么多拍，才有资格当目标（防"刚建的空轨迹"抢位 ✓）。
_MIN_HITS = 4
#: **白度辅助权重**（用户第 ② 条 ✓ 0 = 关掉 ✓）：`pick_white` 找到的白块附近的轨迹加分 ✓。
_WHITE_W = 1.0
#: 白度命中的一次加分量（比一次典型 `dev`（~10）略小 ⇒ **辅助而非主导** ✓）。
_WHITE_HIT = 6.0
#: 白度与轨迹"算同一处"的距离门（px ✓）。
_WHITE_MATCH = 45.0
#: **轨迹表上限**（防长片失控 ✓ 实测 70 帧涨到 34 条 ✗）：超了就丢分数最低、且**当前没用**的 ✓。
_MAX_TRACKS = 40
#: **"相对群体"轨迹线保留多少个点**（用户 2026-10-03 ✓ 要看"历史结果" ✓）：~60 拍 ≈ 6 秒 ✓。
_REL_HIST = 60
#: 算群体中位至少要有几个配对（少了中位不稳 ✗）。
_MIN_PAIRS = 3
#: 轨迹多少拍没配上就删（✓ 之后若又出现 ⇒ 当**新轨迹** ✓）。
_TRACK_LOST = 30
# ---------------- 报告位置那一档（内部常量 ✓ 不上界面 ✗）----------------
#: 观测平滑（`新 = a×旧 + (1−a)×观测` ✓）：实测目标**每帧能跑 ~37px** ⇒ 0.70 会滞后到 87px ✗
#:   ⇒ 0.55 ✓ 仍能防抖 ✓。
_SMOOTH = 0.55
#: 外推速度的每拍衰减（防一次挑歪就一路飞出去 ✓）。
_VEL_DECAY = 0.6
#: 连续多少拍没有目标轨迹（被淘汰）⇒ 判 `lost`。
_LOST_AFTER = 24
# ---------------- 白度辅助（`pick_white` 那一套 ✓ 保留 ✓ 见模块头 ⚠）----------------
_WHITE_Q = 95.0
_AREA_LO = 300.0
_AREA_HI = 30000.0
_FILL_LO = 0.35
#: ⚠ 上限取 **1.0**（= 不设上限 ✓）—— 原来写 0.95 ✗ 会把"**实心方块**"（fill 恰好 1.00 ✗）挡掉
#:   ⇒ 自检里一块纯白方块就挑不出来了 ✗（实测踩到 ✓）。真目标那个白星是**接近实心**的 ✓
#:   ⇒ 这条上限本来就不是用来筛它的 ✗（筛文字靠**下限** ✓ 文字 fill 只有 0.1~0.3 ✓）。
_FILL_HI = 1.0
_AREA_REF = 8000.0
_AREA_W = 0.5
#: **"这块得真的比周围亮"的下限**（灰度差 ✓）：⚠⚠ 不写它会踩大坑 ✗ —— **画面一均匀**（透明透到
#:   只剩背景 / 纯色 / 过曝 ✓）时 `percentile(全同一值, 95)` = 那个值 ⇒ `g >= 阈值` **恒真** ✗
#:   ⇒ 整个门限圈连成一块 ⇒ 重心≈圈心 ⇒ "完美观测"（位置纹丝不动 ✗ 看着稳、其实没看画面 ✗）。
_MIN_GAIN = 12.0
#: 目标候选的半径估计（由框面积换算 ⇒ 画绿圈用 ✓）。
_RAD_MIN = 12.0


def _gray(img):
    """BGR ⇒ float32 灰度（白度辅助用**原图** ✓ 不做模糊 ✗ —— 模糊会把"白"的边界抹开 ✓）。"""
    if img is None:
        return None
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3
            else img).astype(np.float32)


def _blobs(mask, area_lo, area_hi, fill_lo=_FILL_LO, fill_hi=_FILL_HI):
    """连通域 ⇒ `[(cx, cy, area, fill, label)]`（⚠ **填充度**在这里就筛掉文字笔画 ✓）。"""
    _n, _lab, _stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    _out = []
    for _k in range(1, _n):
        _area = float(_stats[_k, cv2.CC_STAT_AREA])
        if _area < area_lo or _area > area_hi:
            continue
        _bw = float(_stats[_k, cv2.CC_STAT_WIDTH])
        _bh = float(_stats[_k, cv2.CC_STAT_HEIGHT])
        _fill = _area / max(1.0, _bw * _bh)
        if _fill < fill_lo or _fill > fill_hi:
            continue
        _out.append((float(_cent[_k][0]), float(_cent[_k][1]), _area, _fill, _k))
    return _out, _lab


def pick_white(g, pos=None, gate=None, area_lo=_AREA_LO, area_hi=_AREA_HI,
               q=_WHITE_Q, want_mask=False, min_gain=_MIN_GAIN):
    """**白度观测**（⚠ 现在只当"**开局辅助证据**" ✗ 不再是唯一观测 ✓ 见模块头 ⚠）。

    用户第 ② 条："一开始依旧用白色图形展示真目标" ✓ ⇒ 开局几条拍用它能**立刻定下来** ✓
    （⚠ 目标逐渐透明后它就没用了 ✗ 那时靠"运动不合群" ✓ 两者互补 ✓）。

    · 对**波浪式几何扭曲**免疫 ✓（扭曲改的是位置 ✗ 不是亮度 ✓）；
    · `pos` / `gate` 给了 ⇒ 只在门限内找 ✓；没给 ⇒ 全图找 ✓；
    · ⚠ 判据是**相对**的（门限内取前 `100-q` % 最亮的 ✓）✗ 不是绝对白度阈值 ✓
      —— 真目标**逐渐透明** ⇒ 绝对白度一直掉 ✓ 设死阈值到后面必然失效 ✗；
    · ⚠ **"这块得真的比周围亮"**（`min_gain` ✓）：防**均匀画面**上"整圈连成一块、重心恰好
      = 门限中心"的假观测 ✓（实测抓到过 ✓ 见 `_MIN_GAIN` ✓）。
    """
    if g is None:
        return (None, None) if want_mask else None
    _roi = np.zeros(g.shape, np.uint8)
    if pos is not None and gate is not None and float(gate) > 1.0:
        cv2.circle(_roi, (int(round(float(pos[0]))), int(round(float(pos[1])))),
                   int(round(float(gate))), 255, -1)
    else:
        _roi[:] = 255
    _vals = g[_roi > 0]
    if _vals.size < 50:
        return (None, None) if want_mask else None
    _lo = float(np.percentile(_vals, float(q)))
    _med = float(np.median(_vals))
    _m = ((g >= _lo) & (_roi > 0)).astype(np.uint8)
    _m = cv2.morphologyEx(_m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _bs, _lab = _blobs(_m, area_lo, area_hi)
    _best, _bv = None, -1e9
    for (_cx, _cy, _area, _fill, _k) in _bs:
        _mean = float(g[_lab == _k].mean())
        _score = _mean / 255.0 + min(_area, _AREA_REF) / _AREA_REF * _AREA_W
        if _score > _bv:
            _best, _bv = (_cx, _cy, _area, _mean), _score
    if _best is None or float(_best[3]) < _med + float(min_gain):
        return (None, None) if want_mask else None
    if want_mask:
        return _best, _m
    return _best


class _Track:
    """一条**候选轨迹**（= 一个"疑似目标" ✓ 真假不分 ✓ 靠 `score` 见分晓 ✓）。

    · `obs`：本拍**观测到**的位置（配不上时保留上一拍 ✓）；
    · `prev`：**上一拍的观测位置**（算位移必须用"观测对观测" ✗ 用平滑位置会把速度带进来 ✓）；
    · `score`：**偏离群体中位的累积**（带衰减 ✓）—— 这就是"运动不合群"的分数 ✓；
    · `hits`：被关联上的累计拍数 ✓（防"刚建的空轨迹" ✓）。
    """

    __slots__ = ("id", "obs", "prev", "vel", "w", "h", "score", "hits", "miss",
                 "mv", "dev", "rel", "rel_hist")

    def __init__(self, tid, cx, cy, w, h):
        self.id = tid
        self.obs = (float(cx), float(cy))
        self.prev = (float(cx), float(cy))
        self.vel = None
        self.w, self.h = float(w), float(h)
        self.score = 0.0
        self.hits = 1
        self.miss = 0
        # ⭐⭐ **诊断用**（可视化 / 排错必须 ✓ 用户 2026-10-03："**我希望：能看到你判定的
        #   可视化依据，不然我无法汇报问题**" ✓）：
        #   · `mv` = **本拍的位移**（`本拍观测 − 上一拍观测` ✓ 与"群体中位位移"同一量纲 ✓）；
        #   · `dev` = **本拍相对群体中位的偏离**（= `|mv − 群体中位|` ✓ **它就是"不合群"的瞬间值** ✓）。
        #   ⚠⚠ 这两个**必须与 `score` 分开记** ✗✗ —— 之前我把 `score` 塞进了叫 `"dev"` 的键里
        #      ⇒ 用户看到的"依据"是错的（分数冒充偏离 ✗）⇒ 排错时完全误导 ✗（踩过 ✓）。
        self.mv = (0.0, 0.0)
        self.dev = 0.0
        # ⭐⭐⭐ **"相对群体"的位置与历史**（用户 2026-10-03 ✓ 原话："我希望：像经典模式一样，
        #   也把**相对于群体的轨迹线**画出来，这样就能看出**历史结果**" ✓）：
        #   · `rel` = 它的位置**减掉"群体累计位移"**后的残余 ✓ —— 起点 = 它出生那一刻（`rel` 初始
        #     就是 `obs` ✓）；
        #   · 每拍 `rel += (自身位移 − 群体中位位移)` ✓ ⇒ **它"相对大家"净走了多少** ✓；
        #   · `rel_hist` = 这条线的最近若干点 ✓（画到画面上就是**"相对群体的轨迹线"** ✓✓）。
        #   ⭐ **为什么这条线一眼能分辨真假** ✓：把"大家一起动的那一份"减掉之后 ——
        #     **假目标** ⇒ 残余只是噪声 ⇒ 线**绕着原地打转** ✓；
        #     **真目标** ⇒ 残余是它自己的运动 ⇒ 线**朝一个方向持续延伸** ✓✓。
        self.rel = (float(cx), float(cy))
        self.rel_hist = [(float(cx), float(cy))]

    @property
    def pos(self):
        return self.obs

    @property
    def rad(self):
        return max(_RAD_MIN, 0.25 * (self.w + self.h))


class MotionTracker:
    """**运动不合群**跟踪器（用户 2026-10-03 规律 ✓ 见模块头 ✓）。

    ⚠ 纯算法（不依赖 Qt ✓ 不依赖 `lie_tracker` ✓）⇒ 可以离线在自检里逐段量 ✓。
    ⚠ **观测来自 yolo 的检出框**（用户第 ⑤ 条 ✓）⇒ 没有检测就没有观测 ✗
      （⚠ 但**不是**"只信检测" ✗ —— 轨迹断了照样外推 ✓ 见 `process` ✓）。
    """

    def __init__(self, pair_gate=_PAIR_GATE, score_decay=_SCORE_DECAY,
                 switch_margin=_SWITCH_MARGIN, min_hits=_MIN_HITS,
                 white_w=_WHITE_W, white_hit=_WHITE_HIT,
                 smooth=_SMOOTH, vel_decay=_VEL_DECAY, lost_after=_LOST_AFTER):
        self.pair_gate = float(pair_gate)
        self.score_decay = min(0.999, max(0.0, float(score_decay)))
        self.switch_margin = float(switch_margin)
        self.min_hits = max(1, int(min_hits))
        self.white_w = max(0.0, float(white_w))
        self.white_hit = float(white_hit)
        self.smooth = min(0.95, max(0.0, float(smooth)))
        self.vel_decay = min(1.0, max(0.0, float(vel_decay)))
        self.lost_after = max(1, int(lost_after))
        self.tracks = []                     # [`_Track`]（真假不分 ✓）
        self.tid = None                      # 当前认定的目标轨迹 id ✓
        self.pos = None
        self.vel = None
        self.state = "init"
        self._next_id = 1
        self._miss = 0                       # 连续多少拍"目标没配上" ✓
        self._dev_med = 0.0                  # 本拍群体中位位移的模（诊断 ✓）
        self.last_white = None               # 本拍白度辅助找到的位置（诊断 ✓）
        self.last_dets = 0                   # 本拍框数（诊断 ✓）

    # ---------------- 内部 ----------------
    def _by_id(self, tid):
        for _t in self.tracks:
            if _t.id == tid:
                return _t
        return None

    def _pair(self, dets):
        """本拍框 ↔ 已有轨迹 **一对一**配对（贪心：距离从小到大取 ✓）。

        ⚠ 用轨迹的**观测位置**（不是预测位置 ✗）当基准 —— 相机一帧走几十 px、目标非匀速 ✓
          ⇒ 预测反而可能更远 ✗；**门限 70px** 已经足够宽 ✓（实测配对门限 70 时中位偏离只 4px ✓）。
        """
        _cand = []
        for _j, _b in enumerate(dets):
            _cx, _cy = float(_b[1]), float(_b[2])
            for _t in self.tracks:
                _d = math.hypot(_cx - _t.obs[0], _cy - _t.obs[1])
                if _d <= self.pair_gate:
                    _cand.append((_d, _t.id, _j))
        _cand.sort()
        _pairs, _used_t, _used_d = [], set(), set()
        for _d, _tid, _j in _cand:
            if _tid in _used_t or _j in _used_d:
                continue
            _used_t.add(_tid)
            _used_d.add(_j)
            _pairs.append((_tid, _j))
        return _pairs, _used_d

    # ---------------- 主入口 ----------------
    def process(self, img, ts=None, dets=None):
        """喂一帧（+ 本拍 yolo 的框 ✓）⇒ `dict(pos, state, area, rad, vel, tracks, dev…)`。

        `dets` = `[(cls, cx, cy, w, h[, conf]), ...]`（**加工域** ✓ 与画面同一把尺 ✓）。
        """
        _dets = [b for b in (dets or []) if b is not None and len(b) >= 5]
        self.last_dets = len(_dets)
        # ⚠ 没框 ⇒ 不观测（**照样走外推** ✓ 不是判丢 ✗ —— 用户第 ④ 条要"重合再分离后继续跟踪" ✓）
        _pairs = []
        if _dets and self.tracks:
            _pairs, _used_d = self._pair(_dets)
        else:
            _used_d = set()
        # ---- ① 位移 + ② 群体中位（⚠ 用"本拍观测 − 上一拍观测" ⇒ 与波浪无关的**相对**量 ✓）----
        _moves = {}
        for _tid, _j in _pairs:
            _t = self._by_id(_tid)
            _b = _dets[_j]
            _moves[_tid] = (float(_b[1]) - _t.prev[0], float(_b[2]) - _t.prev[1])
        _mx = _my = 0.0
        self._dev_med = 0.0
        if len(_moves) >= _MIN_PAIRS:
            _mx = float(np.median([v[0] for v in _moves.values()]))
            _my = float(np.median([v[1] for v in _moves.values()]))
            self._dev_med = float(np.hypot(_mx, _my))
            # ⭐⭐ **存下"群体中位位移"本身**（诊断 / 可视化要用 ✓ 用户："能看到判定的可视化依据" ✓）
            #   —— ⚠ 之前 `motion_viz` 里这个键填的是 `(0.0, 0.0)` 空值 ✗ ⇒ 用户看不到"相机这一拍
            #     走了多少" ⇒ **没法判断谁"不合群"** ✗（实测被用户一眼看出 ✓）。
            self._median_mv = (_mx, _my)
        # ---- ③④ 更新轨迹：偏离 ⇒ 累积（带衰减 ✓）----
        _updated = set()
        for _tid, _j in _pairs:
            _t = self._by_id(_tid)
            if _t is None:
                continue
            _b = _dets[_j]
            _dev = (math.hypot(_moves[_tid][0] - _mx, _moves[_tid][1] - _my)
                    if len(_moves) >= _MIN_PAIRS else 0.0)
            _t.mv = _moves[_tid]                   # ⭐ 本拍位移（诊断 ✓）
            _t.dev = _dev                          # ⭐ 本拍偏离（诊断 ✓ 与 score 分开 ✓）
            # ⭐⭐ **"相对群体"的累计**（用户要的那条轨迹线 ✓ 见 `_Track.rel` 说明 ✓）：
            #   本拍它净走了 `mv − median` ⇒ 累到 `rel` 上 ⇒ 得到"它相对大家走到了哪" ✓。
            _t.rel = (_t.rel[0] + _moves[_tid][0] - _mx,
                      _t.rel[1] + _moves[_tid][1] - _my)
            _t.rel_hist.append(_t.rel)
            if len(_t.rel_hist) > _REL_HIST:       # 只留最近 `_REL_HIST` 点（省内存 + 画面清爽 ✓）
                _t.rel_hist.pop(0)
            _t.score = _t.score * self.score_decay + _dev
            _t.prev = _t.obs                       # ⚠ 位移要用"观测对观测" ✓
            _t.obs = (float(_b[1]), float(_b[2]))
            _t.w, _t.h = float(_b[3]), float(_b[4])
            _t.hits += 1
            _t.miss = 0
            if _t.vel is None:
                _t.vel = (0.0, 0.0)
            _t.vel = (0.7 * _t.vel[0] + 0.3 * (_t.obs[0] - _t.prev[0]),
                      0.7 * _t.vel[1] + 0.3 * (_t.obs[1] - _t.prev[1]))
            _updated.add(_tid)
        # ---- ④' 白度辅助分（用户第 ② 条 ✓ 只在看得见白块时有效 ✓ 透明后自然归零 ✓）----
        self.last_white = None
        if self.white_w > 0.0 and img is not None:
            _w = pick_white(_gray(img))
            if _w is not None:
                self.last_white = (_w[0], _w[1])
                for _t in self.tracks:
                    if (_t.id in _updated
                            and math.hypot(_t.obs[0] - _w[0], _t.obs[1] - _w[1])
                            <= _WHITE_MATCH):
                        _t.score += self.white_w * self.white_hit
        # ---- ⑤ 没配上的轨迹：衰减 + 淘汰；没配上的框：**新建**（score 从 0 起 ✓）----
        for _t in self.tracks:
            if _t.id not in _updated:
                _t.miss += 1
                _t.score *= self.score_decay
                # ⭐⭐⭐ **配不上时，按"群体中位位移"推它一把** ✗✗ —— 不然会**死锁** ✓：
                #   配不上 ⇒ 位置不更新 ⇒ 下一拍差得更远 ⇒ 更配不上（相机每拍走 ~27px ⇒
                #   十拍就差 ~270px ✗ 远超 `pair_gate` 70px ⇒ **永远回不来** ✗）。
                #   实测（用户 2026-10-03）：目标轨迹从帧 12 起就再没配上过 ⇒ `dev` 恒定、
                #   位置**卡死不动** ⇒ 用户看到"**绿圈/箭头乱来**" ✓。
                #   ⚠ 为什么用"群体中位"推 ✗：轨迹配不上的**最常见原因就是被相机带走了** ✓ ——
                #     全场一起动的那一份就是 `median` ✓ ⇒ 推它一把 ⇒ **下一拍多半就能配上** ✓
                #     （自愈 ✓）。⚠ 不是"瞎猜" ✗：这是**全场共识的运动** ✓ 不是它的个性 ✓
                #     ⇒ 因此 `dev` 记 0（"它这一拍没表现出不合群 ✓" 不奖不罚 ✓）。
                _t.obs = (_t.obs[0] + _mx, _t.obs[1] + _my)
                _t.mv = (_mx, _my)
                _t.dev = 0.0
                # ⚠ **`rel` 这里不动** ✗ —— 因为观测刚才是**按 `median` 推的** ✓ ⇒ 它相对群体的
                #   净位移就是 **0** ✓（"这一拍它没有任何个性运动" ✓）⇒ 不进 `rel` ✓
                #   ⚠ 但**历史点要补一个**（否则线会断 ✓ 看不出"那几拍它没动" ✓）。
                _t.rel_hist.append(_t.rel)
                if len(_t.rel_hist) > _REL_HIST:
                    _t.rel_hist.pop(0)
        self.tracks = [_t for _t in self.tracks if _t.miss <= _TRACK_LOST]
        for _j, _b in enumerate(_dets):
            if _j in _used_d:
                continue
            self.tracks.append(_Track(self._next_id, _b[1], _b[2], _b[3], _b[4]))
            self._next_id += 1
        # ---- ⑥ 选目标（`score` 最高 + `hits` 够 ✓ + **切换迟滞** ✓）----
        _elig = [_t for _t in self.tracks if _t.hits >= self.min_hits]
        if _elig:
            _best = max(_elig, key=lambda t: t.score)
            _cur = self._by_id(self.tid) if self.tid is not None else None
            # ⭐⭐⭐ **别"一丢就换"** ✗✗ —— 这是"**绿圈乱飞**"的真凶 ✓（用户 2026-10-03 实测：
            #   "**黄色箭头带着绿圆圈乱飞**" ✓）。原来写 `_cur.miss > 0` ⇒ **只要当前目标本拍
            #   没配上就立刻换人** ✗ ⇒ 而相机一直在动 + 波浪抖 ⇒ **偶尔**配不上是常态 ✓
            #   ⇒ 于是目标在两条轨迹间**来回横跳**（实测帧 21 换到 hits=21 那条、帧 22 又换回来 ✓
            #     hits 12→21→11 ✓ 一眼可见 ✓）。
            #   ⇒ 现在：只有**连续丢够 `lost_after` 拍**才允许换 ✓；偶尔没配上 ⇒ **保留它 +
            #     继续外推** ✓（这正是用户第 ④ 条要的"重叠再分离后继续跟踪" ✓）。
            _old_tid = self.tid
            self._switch_why = ""
            if _cur is None or _cur not in _elig:
                self.tid = _best.id
                self._switch_why = "旧目标还没资格（配得少）"
            elif _cur.miss >= self.lost_after:
                self.tid = _best.id
                self._switch_why = "旧目标连续 %d 拍没配上" % _cur.miss
            elif _best.id != _cur.id and _best.score > _cur.score + self.switch_margin:
                self.tid = _best.id
                self._switch_why = ("新目标分数 %.0f 超过旧目标 %.0f + 余量 %.0f"
                                    % (_best.score, _cur.score, self.switch_margin))
            # ⭐ 只有**真换了**才留痕 ✓（给日志用 ✓ 用户 2026-10-03："我希望：你能把你大概的
            #   计算、必要的我可以跟你沟通的信息在日志里写出来（最好通俗易懂）" ✓）
            self._switch_note = ("" if self.tid == _old_tid
                                 else "#%s → #%s（%s）" % (_old_tid, self.tid,
                                                          self._switch_why))
        # ⚠ **轨迹总数封顶**（防长片失控 ✓ 实测 70 帧涨到 34 条 ✗）：超了就丢分数最低、
        #   且**当前没用**的那些 ✓ 不碰目标那条 ✗。
        if len(self.tracks) > _MAX_TRACKS:
            _keep = sorted(self.tracks, key=lambda t: (t.id == self.tid, t.score),
                           reverse=True)[:_MAX_TRACKS]
            _kids = {t.id for t in _keep}
            self.tracks = [t for t in self.tracks if t.id in _kids]
        _tgt = self._by_id(self.tid) if self.tid is not None else None
        # ---- ⑦ 报告（⚠ 目标本拍没配上 ⇒ 外推照给 ✓ 不判丢 ✗）----
        if _tgt is not None:
            if _tgt.miss == 0:
                _o = _tgt.obs
                self.pos = (_o if self.pos is None else
                            (self.smooth * self.pos[0] + (1.0 - self.smooth) * _o[0],
                             self.smooth * self.pos[1] + (1.0 - self.smooth) * _o[1]))
                self._miss = 0
            else:
                self._miss += 1
                _v = _tgt.vel or (0.0, 0.0)
                _v = (_v[0] * self.vel_decay, _v[1] * self.vel_decay)
                _tgt.vel = _v
                if self.pos is not None and ts is not None:
                    self.pos = (self.pos[0] + _v[0], self.pos[1] + _v[1])
        else:
            self._miss += 1
        self.vel = (_tgt.vel if _tgt is not None else None)
        if self.pos is None:
            self.state = "init"
        elif self._miss > self.lost_after:
            self.state = "lost"
        else:
            self.state = "track"
        return {"pos": self.pos, "state": self.state,
                "area": (None if (self.pos is None or _tgt is None)
                         else math.pi * _tgt.rad * _tgt.rad),
                "rad": (None if _tgt is None else _tgt.rad),
                "vel": self.vel, "dev_med": self._dev_med,
                "n_tracks": len(self.tracks), "score": (None if _tgt is None else _tgt.score)}

    # ---------------- 诊断（演示窗画 / 看 ✓）----------------
    def motion_viz(self, followed_pos=None):
        """给演示窗的 `motion` 字典 —— ⚠⚠ **键必须给全** ✗✗（`Runner` 会给 20 多个 ✓）。

        血泪说明（实测踩到 ✓）：演示窗各处**大多是 `.get(...)`** ⇒ 少几个键*通常*不崩 ✗，
        但只要**任何一处**硬取就是 `KeyError` ⇒ PyQt5 未捕获异常 = **`abort()`** = 用户眼里的
        "**闪退**" ✗✗（还看不出哪一行 ✗）。⇒ 规避办法**不是**猜哪处硬取 ✗，而是**把键补齐** ✓。

        ⭐ `tracks` 里放**每条候选轨迹**（位置 / 速度 / `dev` / `must` ✓）⇒ 演示窗会把它画成
        **绿点 + 箭头** ✓ ⇒ **用户能直接看到算法在给谁加分** ✓✓（这是本模式最好的可验证性 ✓）。
        """
        _viz = []
        _med0 = tuple(getattr(self, "_median_mv", (0.0, 0.0)))   # ⭐ 相对速度要用它 ✓
        for _t in self.tracks:
            _v = _t.vel or (0.0, 0.0)
            _rel0 = _t.rel                       # ⭐ 以"它当前所在"为原点 ⇒ 历史点都写成**相对偏移** ✓
            _viz.append({
                "p": _t.obs, "v": _v, "tid": _t.id, "a": _t.rad,
                # ⭐⭐⭐ **判定依据（用户要的就是这几个 ✓）**：
                #   · `mv`   = **本拍位移**（它这一拍走了多少 ✗ 含相机那份 ✓）；
                #   · `dev`  = **本拍相对群体中位的偏离** = "**不合群的瞬间值**" ✓；
                #   · `score`= **累积分**（带衰减 ✓ "**持续不合群**"的那个 ✓ 选目标就看它 ✓）；
                #   · `hits` = 被关联上的拍数 ✓；`sel` = **本拍是不是它** ✓；
                #   · `ok`   = 够不够资格（`hits ≥ min_hits` ✓）。
                "mv": tuple(_t.mv), "dev": float(_t.dev), "score": float(_t.score),
                "hits": _t.hits,
                "sel": bool(_t.id == self.tid),
                "ok": bool(_t.hits >= self.min_hits),
                "must": bool(_t.id == self.tid),
                # ⭐⭐⭐ **"相对群体"的历史轨迹线**（用户 2026-10-03 ✓ 原话："我希望：像经典模式
                #   一样，也把**相对于群体的轨迹线**画出来，这样就能看出**历史结果**" ✓）——
                #   点坐标全是**相对"它当前所在"的偏移** ✓（原点 = 它自己 ✓）
                #   ⇒ 画的时候直接"以这条候选当前位置为锚点"铺出去 ✓ 不用管绝对坐标 ✓。
                #   ⚠ 读法：**绕原地打转 = 跟大家一起动（假目标）** ✓；
                #          **朝一个方向延伸出去 = 在相对群体移动（真目标）** ✓✓。
                "rel_hist": [(float(_h[0] - _rel0[0]), float(_h[1] - _rel0[1]))
                             for _h in _t.rel_hist],
                # ⭐⭐⭐ **本拍的"绝对速度"与"相对群体速度"**（用户 2026-10-03 ✓ 原话：
                #   "把所有目标的**绝对速度**像经典模式一样用**小蓝箭头**标一下，把真目标的
                #   **相对群体速度**用**小白箭头**标一下" ✓）：
                #   · `v_abs` = 它这一拍**实际走了多少**（= `mv` ✓ 屏幕坐标系 ✓ **蓝箭头** ✓）；
                #   · `v_rel` = `mv − 群体中位位移` ⇒ **"相对大家多走了多少"** ✓（**白箭头** ✓）
                #     ⚠ 它就是 `dev` 的**向量版** ✓ —— 标量 `dev` 只说"偏了多少"，
                #     向量还说"**往哪偏**" ✓ 排错时这个才是关键 ✓（用户说"朝运动方向右偏" ✓
                #     就是拿它看出来的 ✓）。
                "v_abs": tuple(_t.mv),
                "v_rel": (float(_t.mv[0] - _med0[0]), float(_t.mv[1] - _med0[1])),
            })
        _tgt = self._by_id(self.tid) if self.tid is not None else None
        _box = (None if (self.pos is None or _tgt is None)
                else (float(self.pos[0]), float(self.pos[1]),
                      2.0 * _tgt.rad, 2.0 * _tgt.rad))
        # ⭐⭐⭐ **给底部日志区的一行"通俗说明"**（用户 2026-10-03 ✓ 原话："我希望：你能把你大概的
        #   计算、必要的我可以跟你沟通的信息在日志里写出来（**最好通俗易懂**）" ✓）——
        #   一句话讲清**这一拍发生了什么**：相机走了多少 / 目标分数与位移 / 它有没有"最不合群" /
        #   **有没有换目标及为什么** ✓ ⇒ 用户截一行日志就能跟我对话 ✓✓。
        _selv = next((t for t in _viz if t["sel"]), None)
        _worst = max(_viz, key=lambda t: t["dev"]) if _viz else None
        _log = ("候选 %d 条 ｜ 相机这一拍走了 (%+.0f,%+.0f) ｜ 全场典型偏离 %.1f px"
                % (len(_viz), _med0[0], _med0[1], self._dev_med))
        if _selv is not None:
            _log += ("\n　目标 #%s：分数 %.0f ｜ 它这一拍走了 (%+.0f,%+.0f) ｜ 相对大家偏了 %.1f px"
                     % (_selv["tid"], _selv["score"],
                        _selv["v_abs"][0], _selv["v_abs"][1], _selv["dev"]))
        if (_worst is not None and _selv is not None
                and _worst["tid"] != _selv["tid"] and _worst["dev"] > _selv["dev"] * 1.5):
            _log += (" ｜ ⚠ 但**最偏的是 #%s（%.1f px）**，比目标还偏 ⇒ 目标可能选错了"
                     % (_worst["tid"], _worst["dev"]))
        if getattr(self, "_switch_note", ""):
            _log += "\n　⚠ **本拍换目标**：" + self._switch_note
        return {
            # ---- 本模块真有的（画绿圈 / 白线 / 目标框 / 候选点用 ✓）----
            "tgt_radius": (None if _tgt is None else _tgt.rad),
            "tbox": _box,
            "followed_pos": (None if followed_pos is None
                             else (float(followed_pos[0]), float(followed_pos[1]))),
            "miss": self._miss, "state": self.state,
            "tracks": _viz,
            # ⭐ **这两个是"所有判断的基准"** ✓ 必须给真值 ✗ 给空值就完全没法核对 ✗（踩过 ✓）：
            #   · `median` = **群体中位位移**（= 相机这一拍走了多少 ✓）
            #   · `dev`    = 它自己的模长（= 全场"典型偏离" ✓ 当噪声底看 ✓）
            "median": tuple(getattr(self, "_median_mv", (0.0, 0.0))),
            "dev": self._dev_med,
            "tid": self.tid,
            "followed": self.tid,
            "typ_area": (0.0 if _tgt is None else float(_tgt.w * _tgt.h)),
            "n_cands": len(_viz),
            "sel_score": (None if _tgt is None else float(_tgt.score)),
            "sel_dev": (None if _tgt is None else float(_tgt.dev)),
            # ⭐ **日志区那一行**（演示窗把它追加到底部日志 ✓ 通俗说明 ✓ 见上面那段的拼装 ✓）
            "log_text": _log,
            # ⭐ 标记"这是运动分离模式" ⇒ 演示窗据此决定**要不要画那套依据** ✓（经典模式不画 ✓）
            "mode": "motion",
            "track_v": (self.vel or (0.0, 0.0)), "path_pts": [],
            "box_moves": [], "box_roles": [], "box_pred": [],
            "rad_frozen": True, "red_i": None, "raw_pos": None,
            "tbox_rad": None, "tbox_wh": None,
            "vel_abs": self.vel, "vel_rel": self.vel, "reg": [], "merged": False,
        }


class MotionRunner:
    """`Runner` 的**同签名替身**（`step` 回同样的 6 元组 ✓）⇒ `lie_demo` **只换构造** ✓。

    这样演示窗那套（`warm_tick` / `_show_precomputed` / `draw` / 拖动 / 播放 ✓）**一行都不用改** ✓
    —— 包括"白线 = 报告位置历史"、"绿圈 = 目标"、"红点 = 光标"、**"绿点 = 每条候选轨迹"**
    全都是现成的 ✓。

    ⚠ 本类的 `dets` 与 `Runner` **同一个用法** ✗ 别自作主张：`warm_tick` 塞的是**整段列表**
      （`self.runner.dets = self.dets` ✓）⇒ 所以这里必须 `self.dets[i]` 取本拍 ✓；
      `live_lie.py` 塞的是**单元素列表**（`[_raw]` ✓）+ `i=0` ✓ ⇒ 同一套写法两边都对 ✓。
    """

    def __init__(self, dets=None, gain=None, assume=(375.0, 250.0), follow_gain=None,
                 **tracker_kw):
        # ⚠⚠ `gain` 要的是**每像素的二元组** ✗✗ —— 千万别把"跟随效率倍率"（**标量** ✓ 那是
        #   `follow_gain` ✓）填进来 ⇒ `tuple(标量)` 会 `TypeError` ⇒ PyQt5 里 = **闪退** ✓（踩过 ✓）。
        self.ctl = LieMouseController(gain=gain, assume=assume, follow_gain=follow_gain)
        self.cursor = list(assume)
        self.pend = []
        self.hits = 0
        self.n = 0
        self.dets = dets                     # ⚠ `warm_tick` 每拍会覆盖它 ✓
        self.tr = MotionTracker(**tracker_kw)

    def step(self, small, i, ts):
        """与 `Runner.step` **同签名同返回**（`(out, pos, rad, hit, boxes, motion)` ✓）。"""
        for cmd in self.pend:                 # 先让上一拍指令落地（时序同实机 ✓）
            self.cursor[0] += cmd[0] * self.ctl.gain[0]
            self.cursor[1] += cmd[1] * self.ctl.gain[1]
        self.cursor = [round(self.cursor[0]), round(self.cursor[1])]
        self.pend = []
        _d = None
        if self.dets is not None:
            _raw = self.dets[i] if i < len(self.dets) else None
            _d = _raw if _raw else None
        out = self.tr.process(small, ts=ts, dets=_d)
        pos = out["pos"] if out["state"] == "track" else None
        r = max(18.0, (out["rad"] or 0.0))
        hit = None
        if pos is not None:
            self.n += 1
            dist = ((self.cursor[0] - pos[0]) ** 2 + (self.cursor[1] - pos[1]) ** 2) ** 0.5
            hit = dist <= r
            self.hits += 1 if hit else 0
        cmd = self.ctl.step(pos, ts=ts, vel=out.get("vel"),
                            cursor=(self.cursor[0], self.cursor[1]))
        if cmd[0] or cmd[1]:
            self.pend.append(cmd)
        motion = self.tr.motion_viz(followed_pos=pos)
        motion["followed_v"] = out.get("vel")
        motion["boxes"] = list(_d or [])
        motion["dt"] = 1.0 / 60.0
        # ⚠⚠ **把本拍 YOLO 检出框传出去** ✗✗（我第一版这里硬写 `[]` ⇒ **画面上一个检出框都
        #   看不到** ✗ 用户一眼看出来："**没有看到 yolo 检出框**" ✓）。
        #   `draw()` 的 `boxes` 要的就是 `Runner` 那份**检测原框**（`(cls, cx, cy, w, h, conf)` ✓
        #   带 cls/conf ⇒ 类别标签才画得出 ✓）⇒ 原样回传 ✓。
        return out, pos, r, hit, list(_d or []), motion
