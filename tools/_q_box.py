# -*- coding: utf-8 -*-
"""一次性探针（用完即删）：**重叠 ⇒ 框变大 ⇒ 观测被拉偏** 这条链成立吗 ✓。

用户 2026-10-03（两条关键线索 ✓）：
  ① 帧 13 的日志（`v_rel = 40.5` = 噪声底 4.2 的 **9.6 倍**）⇒ 他说："这一帧其实**已经跟假目标
     重叠**了，我认为应该稍微**沿着轨迹预测减点速**，而不是急于**寻找跟踪信号**（现在看起来
     就很急）" ✓；
  ② 帧 16 有个**大检出框**：中心 (316.2,249.5)、**尺寸 183×209**、面积 38291（常态那种约
     150×150 ⇒ 面积 ~22500 ✓）⇒ 他说："真目标**有一部分跟假目标重叠** ⇒ 检出框就**变大了**" ✓。

⇒ 本探针要证的三件事（逐帧列出 ✓）：
  · **框面积**随时间有没有**突增**（= 两个目标粘成一个 ✓）；
  · 突增的**是不是当前目标那条**（`#13` ✓）；
  · 突增那几拍，它的 `v_rel` / `dev` / **观测位置漂移**是不是同时异常 ✓
    ⇒ 若是 ⇒ **那个 40.5 不是"它在剧烈机动"✗ 而是"观测被粘住了"** ✓✓
    ⇒ 那就**不该拿它当"不合群信号"去加分** ✗（用户说得对 ✓）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.lie_demo as D                                          # noqa: E402
from perception.lie_motion import MotionTracker                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "liedetectorVideo" / "10月1日.mp4"

frames, ts, proc, scale, fps = D.load_source(str(SRC))
_w = D.DetsWorker(None, conf=0.25)
_dets = [_w.detect(f[1]) for f in frames]
_w.close()

_tr = MotionTracker()
_hist = {}          # tid -> [面积, …]（**只记真配上那几拍** ✓ 没配上 `w/h` 是不更新的 → 重复记没意义 ✗）
print("帧 ｜ sel ｜ 观测位置 ｜ 它这一拍的框 ｜ 面积/历史中位 ｜ mv ｜ v_rel ｜ dev ｜ score ｜ 全场最大框")
for i in range(len(frames)):
    _tr.process(frames[i][1], ts=ts[i], dets=_dets[i])
    _sel = _tr._by_id(_tr.tid)
    _maxdet = max((b[3] * b[4] for b in _dets[i]), default=0.0)
    if _sel is not None and _sel.miss == 0:
        _hist.setdefault(_sel.id, []).append(_sel.w * _sel.h)
    if not (9 <= i + 1 <= 22):
        continue
    _a = None if _sel is None else _sel.w * _sel.h
    _h = _hist.get(None if _sel is None else _sel.id, [])
    _ref = float(np.median(_h)) if _h else float("nan")
    _med = tuple(getattr(_tr, "_median_mv", (0.0, 0.0)))
    _vrel = None if _sel is None else (_sel.mv[0] - _med[0], _sel.mv[1] - _med[1])
    print("%3d ｜ %4s ｜ (%5.0f,%5.0f) ｜ %3.0fx%-3.0f ｜ %6.0f = **%.2fx** %s ｜ (%+5.1f,%+5.1f) ｜ "
          "%6s ｜ %5.1f ｜ %5.1f ｜ %6.0f"
          % (i + 1, "-" if _sel is None else "#%d" % _sel.id,
             _sel.obs[0], _sel.obs[1], _sel.w, _sel.h, _a, _a / max(1.0, _ref),
             "⚠**粘住了**" if _a / max(1.0, _ref) > 1.25 else "",
             _sel.mv[0], _sel.mv[1],
             "-" if _vrel is None else "%.1f" % ((_vrel[0] ** 2 + _vrel[1] ** 2) ** 0.5),
             _sel.dev, _sel.score, _maxdet))
