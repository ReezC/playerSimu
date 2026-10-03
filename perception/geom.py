"""**几何重叠尺**（用户 2026-10-03 ✓ 原话："我才学到 IoU 这个算法，我认为我们很多判定都可以换成
这个算法（例如『噪声容差』改成『噪声最小 IoU』）" ✓）。

三种"两个框像不像"的量，**全部收在这里**（之前 tracker 与登记表各自实现了一份 ✗ 口径两处 ✓）：

| 函数 | 算法 | 语义 | 现在用在哪 |
|---|---|---|---|
| `iou(a, b)` | 交集 ÷ **并集** | **对称**："这两个框是不是**同一格**" | 融合**续任格**（`lie_tracker._box_iou` ✓）|
| `overlap_min(a, b)` | 交集 ÷ **较小面积** | "小框叠在大框上"时接近 1 | 登记表**上板防抖**（`lie_registry._ov_ratio` ✓）|
| `cover_ratio(a, b)` | 交集 ÷ **b 的面积** | **单向**："a **盖住** b 多少" | **覆盖率**族（判定为砖 / 身份 / 包住参照砖 ✓）|

⚠⚠ **为什么不是"所有判定都换成 IoU"** ✗（三条理由 ✓ 都是实测/既有口径）：

· **"覆盖"类换不得 IoU** ✗ —— 它们问的是"**大框盖住小砖多少**"，**框多大不进分母** ✓；
  换 IoU 会把"框比砖大得多"本身当成不合规 ⇒ 真并集框**永远盖不满**砖（IoU 天然小 ✓）
  ⇒ 融合/身份/保持全线误判 ✗（例：并集框 1.8× 砖面积时，覆盖率可 0.9+、IoU 只有 0.5 左右 ✓）。
· **"小框叠大砖"换不得 IoU** ✗ —— 那正是"**旧砖被重复往上叠加砖**"的形态 ✓（交集÷较小面积
  ≈ 1.0 ✓ 而 IoU ≈ 小/大 ⇒ **拦不住** ✗）⇒ 用户 2026-10-02 已按此选定 ✓（见 `_ov_ratio` 注释 ✓）。
· **"圆心 ± 噪声容差 与框相交"根本不是面积重叠** ✗（点/边 对 区域 ✓）⇒ IoU **无对应物** ✓；
  它量的是"离框边还有几像素"（绝对值 ✓ 尺度相关 ✓）。

⇒ 结论：**"同一格 / 同一块"这类对称问题 ⇒ 用 IoU** ✓；"覆盖"⇒ `cover_ratio` ✓；
  上板防抖 ⇒ `overlap_min` ✓。三把尺各司其职，但**共用这一个文件**（口径一处 ✓ 便于换/调 ✓）。

⚠ 所有函数都**只吃"框"**（`(cx, cy, w, h)` ✓）与 `Entry` 的兼容靠调用方自己拆 ✓（保持无依赖 ✓）；
  任一项算不出（面积 ≤ 0 / 没交集）⇒ 回 `0.0` ✓ 不抛 ✗。
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

Box = Sequence[float]

__all__ = ["iou", "overlap_min", "cover_ratio", "inter_area"]


def inter_area(a: Optional[Box], b: Optional[Box]) -> float:
    """两框的**交集面积**（`(cx, cy, w, h)` 布局 ✓ 任一为空 / 无交集 ⇒ `0.0` ✓）。"""
    if a is None or b is None:
        return 0.0
    _ax0, _ay0 = float(a[0]) - float(a[2]) / 2.0, float(a[1]) - float(a[3]) / 2.0
    _ax1, _ay1 = float(a[0]) + float(a[2]) / 2.0, float(a[1]) + float(a[3]) / 2.0
    _bx0, _by0 = float(b[0]) - float(b[2]) / 2.0, float(b[1]) - float(b[3]) / 2.0
    _bx1, _by1 = float(b[0]) + float(b[2]) / 2.0, float(b[1]) + float(b[3]) / 2.0
    _ix = min(_ax1, _bx1) - max(_ax0, _bx0)
    _iy = min(_ay1, _by1) - max(_ay0, _by0)
    if _ix <= 0.0 or _iy <= 0.0:
        return 0.0
    return float(_ix * _iy)


def iou(a: Optional[Box], b: Optional[Box]) -> float:
    """**IoU = 交集 ÷ 并集**（`[0, 1]` ✓ 对称 ✓）—— "这两个框是不是**同一格**" ✓。

    用户 2026-10-03 ✓：`10月1日.mp4` 帧 23 的"续任格"就是靠它从"conf 0.80 的小框"
    （旧口径 0.901 ✗ 虚高）纠正回"conf 0.94 那格本身"（IoU 0.790 vs 0.503 ✓）。
    """
    if a is None or b is None:
        return 0.0
    _ai = float(a[2]) * float(a[3])
    _bi = float(b[2]) * float(b[3])
    if _ai <= 0.0 or _bi <= 0.0:
        return 0.0
    _inter = inter_area(a, b)
    _union = _ai + _bi - _inter
    return float(_inter / _union) if _union > 0.0 else 0.0


def overlap_min(a: Optional[Box], b: Optional[Box]) -> float:
    """**重叠率 = 交集 ÷ 两者中较小的面积**（用户 2026-10-02 选定 ✓）：**小框叠在大框上 ⇒ ≈ 1** ✓
    —— 正是"旧砖被重复往上叠加砖"的形态 ✓（用 IoU 会偏小、**拦不住** ✗ 见模块头 ✓）。"""
    if a is None or b is None:
        return 0.0
    _ai = float(a[2]) * float(a[3])
    _bi = float(b[2]) * float(b[3])
    _mn = min(_ai, _bi)
    if _mn <= 0.0:
        return 0.0
    return float(inter_area(a, b) / _mn)


def cover_ratio(a: Optional[Box], b: Optional[Box]) -> float:
    """**覆盖率 = 交集 ÷ b 的面积**（**单向** ✓）："**a 盖住 b 多少**" ✓ —— `a` 多大**不进分母** ✓
    （框比砖大得多也能到 1.0 ✓；这正是"并集框盖住砖"的情形 ✓ 换 IoU 反而会误判 ✗ 见模块头 ✓）。"""
    if a is None or b is None:
        return 0.0
    _bi = float(b[2]) * float(b[3])
    if _bi <= 0.0:
        return 0.0
    return float(inter_area(a, b) / _bi)


def _as_box(e) -> Optional[Tuple[float, float, float, float]]:
    """把登记表的 `Entry` 拆成 `(cx, cy, w, h)`（给本模块的调用方用 ✓ 拿不到 ⇒ `None` ✓）。"""
    try:
        return (float(e.x), float(e.y), float(e.w), float(e.h))
    except Exception:                       # noqa: BLE001 —— 不是框对象 ⇒ 判不出 ✓
        return None
