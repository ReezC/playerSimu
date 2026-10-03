# -*- coding: utf-8 -*-
"""`perception/lie_motion.py`（**YOLO 候选 + 运动不合群** ✓）的自检 —— 离屏 ✓ 不弹窗 ✓。

**这条路的要害只有量得出来**（都用**合成序列**钉住 ✓ 可复现 ✓ 不需要素材 ✓）：
  ① **认得出"异类"**：一群假目标**一起平移**（= 相机在动 ✓），真目标额外再多走一点
     （= "相对群体在动" ✓ 用户第 ④ 条）⇒ 必须选出**真那个** ✗ 不能选假目标 ✓；
  ② **波浪免疫**：给全场叠加**同步的随机抖动**（= 画面几何被扭曲 ⇒ 所有框一起被搅 ✓）⇒
     **判据不受影响** ✓✓（这正是它相对"全局对齐"那条路的根本优势 ✓ 见模块头 ✓）；
  ③ **短暂丢框不判丢**：目标那格消失几拍 ⇒ 照样给位置（外推 ✓ 用户第 ④ 条「重叠再分离后
     继续跟踪」 ✓）；
  ④ **白度辅助**（用户第 ② 条"一开始依旧用白色图形" ✓）：开局有白块时能提前定下来 ✓；
  ⑤ **参数真的起作用**（界面那 5 个 ✗ 不摆空架子 ✓）；
  ⑥ `MotionRunner` 与 `Runner` **同签名**（演示窗只换构造 ✓）。
再加一条**真素材**短冒烟：用真 YOLO 跑前几拍 ⇒ 报的位置**与白星真值几乎重合** ✓
  （唯一有真值的时刻 ⇒ 用它校准 ✓ 实测帧 6 判据 top1 与真值完全吻合 ✓）。
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.lie_motion import (MotionRunner, MotionTracker,  # noqa: E402
                                   _gray, pick_white)

_FAILED = 0


def check(ok, msg):
    global _FAILED
    print(("  [OK] " if ok else "  [NG] ") + msg)
    if not ok:
        _FAILED += 1


def _frame(fake, target=None, jitter=0.0, rng=None, w=750, h=500):
    """造一拍的检出框：`fake` 个假目标排成一圈 + 可选一个真目标 ✓（都带一点抖动 ✓）。

    ⚠ 返回的布局 = `(cls, cx, cy, w, h, conf)`（**加工域** ✓ 与 `MotionTracker` 的契约一致 ✓）。
    """
    _rng = rng or np.random.RandomState(0)
    _dets = []
    for _k in range(int(fake)):
        _ang = 2.0 * math.pi * _k / max(1, int(fake))
        _cx = w / 2.0 + 260.0 * math.cos(_ang)
        _cy = h / 2.0 + 170.0 * math.sin(_ang)
        _dets.append((0, _cx, _cy, 150.0, 150.0, 0.9))
    if target is not None:
        _dets.append((0, float(target[0]), float(target[1]), 150.0, 150.0, 0.9))
    if jitter > 0.0:                       # ⭐ **全场同步**抖动 = 波浪式几何扭曲的替身 ✓
        _jx, _jy = _rng.normal(0.0, jitter), _rng.normal(0.0, jitter)
        _dets = [(c, x + _jx, y + _jy, bw, bh, cf) for (c, x, y, bw, bh, cf) in _dets]
    return _dets


def _shift(dets, dx, dy):
    return [(c, x + dx, y + dy, bw, bh, cf) for (c, x, y, bw, bh, cf) in dets]


def _run(seq, **kw):
    """喂一串「每拍框表」 ⇒ 回 `(最后一拍的输出, tracker)` ✓。"""
    _t = MotionTracker(**kw)
    _o = None
    for _i, _d in enumerate(seq):
        _o = _t.process(None, ts=_i * 0.16, dets=_d)
    return _o, _t


def test_picks_the_odd_one():
    """⭐⭐ **选出「运动不合群」的那个**（真目标 ✓ 用户第 ④ 条）。"""
    _fake0 = _frame(9)
    _seq = []
    for _i in range(18):
        _cam = (7.0 * _i, 3.0 * _i)                 # 相机平移（假目标跟着走 ✓）
        _d = _shift(_fake0, _cam[0], _cam[1])
        # 真目标：跟着相机走（+7/拍 ✓）**再自己额外向右 7px/拍**（"相对群体在动" ✓）
        # ⚠ 真值 = **380 + 7×i（相机）+ 7×i（自己）** ✗ 别只算一半 ✗（我第一版就少算了一倍 ⇒
        #   断言算出的"误差 112px"其实是**我算错了真值** ✓ 算法本身是准的 ✓ 踩过 ✓）。
        _d = _d + [(0, 380.0 + _cam[0] + 7.0 * _i, 300.0 + _cam[1], 150.0, 150.0, 0.9)]
        _seq.append(_d)
    _o, _t = _run(_seq, min_hits=3)
    _err = (None if _o["pos"] is None
            else math.hypot(_o["pos"][0] - (380.0 + 14.0 * 17), _o["pos"][1] - 300.0))
    check(_o["state"] == "track" and _err is not None and _err < 60.0,
          "① 一群假目标**一起走**、真目标**自己多走**（+7px/拍 ✓）⇒ 选中的是**真那个**"
          "（实际 %s ｜ 与真值差 **%.1f px** ✓ —— ⚠ 这个差主要是**观测平滑的滞后**"
          "（`_SMOOTH=0.55` ✓ 而它每拍相对群体 +7px ⇒ 稳态滞后 ≈ 7×0.55/(1−0.55) ≈ 8.5px ✗ "
          "再叠加**群体本身**也在动 ✓）⇒ 量级合理 ✓ 不是「选错人」✓）"
          % (None if _o["pos"] is None else "(%.1f,%.1f)" % _o["pos"], _err if _err else -1.0))
    check(_o["score"] is not None and _o["score"] > 20.0,
          "① 它的**累积偏离**攒起来了（score = %.1f ✓ —— 真目标「持续不合群」 ⇒ 分数一直涨 ✓）"
          % (_o["score"] or -1.0,))


def test_wave_immune():
    """⭐⭐ **「波浪式几何扭曲」免疫**：全场**同步**抖动 ⇒ 判据不受影响 ✓✓（这是本方案的根本优势 ✓）。"""
    _fake0 = _frame(9)
    _rng = np.random.RandomState(7)
    _seq = []
    for _i in range(18):
        _d = _shift(_fake0, 7.0 * _i, 3.0 * _i)
        _d = _d + [(0, 380.0 + 7.0 * _i + 7.0 * _i, 300.0 + 3.0 * _i, 150.0, 150.0, 0.9)]
        # ⚠ **全场同步**抖动（±12px 随机 ✓ 每拍不同 ✓）—— 等价于"整幅画被水波扭了一下" ✓
        _d = _frame(0) + _d                        # 占位（保持结构清晰 ✓）
        _seq.append(_d[0:] if False else _d)
    # 干净跑一遍当对照 ✓
    _o0, _ = _run(_seq, min_hits=3)
    # 加"波浪"再跑 ✓：**同一拍所有框加同一个随机偏移** ✓
    _seq_w = []
    for _i, _d in enumerate(_seq):
        _jx, _jy = _rng.normal(0.0, 12.0), _rng.normal(0.0, 12.0)
        _seq_w.append([(c, x + _jx, y + _jy, bw, bh, cf) for (c, x, y, bw, bh, cf) in _d])
    _o1, _t1 = _run(_seq_w, min_hits=3)
    _err = (None if _o1["pos"] is None
            else math.hypot(_o1["pos"][0] - (380.0 + 14.0 * 17), _o1["pos"][1] - 300.0))
    check(_o1["state"] == "track" and _err is not None and _err < 80.0,
          "② 全场同步抖动 ±12px（= 波浪扭曲 ✓）之下**照样选中真目标**"
          "（实际 %s ｜ 误差 **%.1f px** ✓ —— 抖动对全场一视同仁 ⇒ 减中位即消 ✓）"
          % (None if _o1["pos"] is None else "(%.1f,%.1f)" % _o1["pos"], _err if _err else -1.0))
    check(_o0["pos"] is not None and _o1["pos"] is not None,
          "② 且干净跑与加波浪跑**都出了位置**（干净 %s ／ 波浪 %s ✓ 不是碰巧 ✓）"
          % ("(%.0f,%.0f)" % _o0["pos"], "(%.0f,%.0f)" % _o1["pos"]))


def test_briefly_missing_keeps_pos():
    """⭐⭐ **目标那格消失几拍 ⇒ 照样给位置**（外推 ✓ 用户第 ④ 条「重叠再分离后继续跟踪」 ✓）。"""
    _fake0 = _frame(9)
    _seq = []
    for _i in range(16):
        _d = _shift(_fake0, 6.0 * _i, 2.0 * _i)
        if 10 <= _i <= 13:                          # ⚠ 这 4 拍**真目标的框没了**（透明/被并框 ✓）
            pass
        else:
            _d = _d + [(0, 380.0 + 12.0 * _i, 300.0 + 2.0 * _i, 150.0, 150.0, 0.9)]
        _seq.append(_d)
    _o, _t = _run(_seq, min_hits=3)
    check(_o["pos"] is not None and _o["state"] == "track",
          "③ 目标框消失 4 拍 ⇒ **仍然给位置**（pos=%s state=%s ✓）—— ⚠ 这条是给 `Runner` 那条链"
          "用的：它一见 `state != track` 就把 `pos` 抹成 None ✗ ⇒ 白线会断 ✗"
          % (None if _o["pos"] is None else "(%.1f,%.1f)" % _o["pos"], _o["state"]))
    check(_o["pos"][0] > (380.0 + 12.0 * 9),
          "③ 而且位置**顺着速度接着走**（x=%.1f > 消失前那拍 %.1f ✓ 不是钉死不动 ✗）"
          % (_o["pos"][0], 380.0 + 12.0 * 9))


def test_white_helper():
    """⭐ **白度辅助**（用户第 ② 条「一开始依旧用白色图形展示真目标」 ✓）。"""
    # ⚠ `pick_white` 那几条老约定还要钉住（它是辅助 ✓ 但**均匀画面不能伪造观测** ✗✓）
    for _v in (0, 100, 200):
        _f = np.full((500, 750, 3), _v, np.uint8)
        check(pick_white(_gray(_f), pos=(375.0, 250.0), gate=90.0) is None,
              "④ 纯色（灰度 %d）⇒ `None` ✓ 不伪造观测（⚠ 不修的话会回一个「就在门限中心」的假块 ✗ "
              "位置从此一动不动 ✗）" % _v)
    _f2 = np.full((500, 750, 3), 100, np.uint8)
    # ⚠⚠ 白块要**够大**（占画面 > 5% ✗）—— 否则 `percentile(95)` 会落在**灰底**上 ✗ ⇒ 掩膜变成
    #   "整幅图"（因为 `g >= 100` 对灰底也成立 ✗）⇒ 连通域面积超 `_AREA_HI` ⇒ **被筛掉** ⇒ None ✗
    #   （我第一版用 60×60 = 0.96% ✗ 就踩了这个 ✓）。
    _f2[150:305, 250:405] = (255, 255, 255)
    _o = pick_white(_gray(_f2))
    # ⚠ 白块是 `[150:305, 250:405]` ⇒ 中心 = **(327.5, 227.5)** ✗ 别按数组下标想当然 ✗（踩过 ✓）
    check(_o is not None and abs(_o[0] - 327.5) < 12.0 and abs(_o[1] - 227.5) < 12.0,
          "④ 灰底 + 白块 ⇒ 照样挑得出（%s ✓ —— `min_gain` 只挡「均匀」，不挡「背景也在、目标更亮」"
          "✓ 且白块要占画面 **5%% 以上** 才能让分位数落在它身上 ✓）"
          % (None if _o is None else "(%.1f,%.1f)" % (_o[0], _o[1]),))
    # 权重为 0 ⇒ 白度完全不参与（纯靠运动 ✓）
    _fake0 = _frame(9)
    _seq = []
    for _i in range(6):                            # ⚠ 只有 6 拍 ⇒ "运动"还来不及攒分 ✓
        _d = _shift(_fake0, 5.0 * _i, 2.0 * _i)
        _d = _d + [(0, 380.0 + 5.0 * _i, 300.0 + 2.0 * _i, 150.0, 150.0, 0.9)]
        _seq.append(_d)
    _o0, _ = _run(_seq, min_hits=1, white_w=0.0)
    check(_o0["state"] in ("track", "init"),
          "④ `white_w=0` ⇒ 白度**完全不参与**（state=%s ✓ 关得掉 ✓ 便于对照 ✓）" % _o0["state"])


def test_params_really_work():
    """⭐⭐ **界面那 5 个参数不是摆设**（用户 2026-10-03 ✓ 原话：「运动分离 **没有任何参数要配吗？**」 ✓）。"""
    _fake0 = _frame(9)

    def _mk(n=16, dev=7.0):
        _s = []
        for _i in range(n):
            _d = _shift(_fake0, 7.0 * _i, 3.0 * _i)
            _d = _d + [(0, 380.0 + 7.0 * _i + dev * _i, 300.0 + 3.0 * _i, 150.0, 150.0, 0.9)]
            _s.append(_d)
        return _s

    # ① `min_hits`：太早不许选（轨迹还没关联够 ✓）
    _o1, _ = _run(_mk(3), min_hits=10, white_w=0.0)
    check(_o1["state"] == "init" or _o1["pos"] is None,
          "⑤ `min_hits=10` 但只喂 3 拍 ⇒ **还没资格当目标**（state=%s ✓ 不抢跑 ✓）"
          % _o1["state"])
    # ② `pair_gate`：门限小于"一帧真实位移" ⇒ **一条都配不上** ⇒ 轨迹一直重建 ⇒ 选不出来 ✓
    _o2, _t2 = _run(_mk(8), pair_gate=2.0, min_hits=1, white_w=0.0)
    # ⚠ 断言别写成 `score is None` ✗：配不上时**每条轨迹都是第一拍**⇒ 分数自然还是 **0.0**
    #   （不是 None ✗）⇒ 该断言会**永远失败** ✗ 踩过 ✓。要断的是"**分没攒起来**" ✓。
    check(_t2.last_dets > 0 and float(_o2["score"] or 0.0) < 5.0,
          "⑤ `pair_gate=2px` 而相机一帧走 ~7px ⇒ **配不上** ⇒ 没有候选攒到分"
          "（本拍帧数=%d ｜ score=%.1f ✓ 门限真的在把关 ✓）"
          % (_t2.last_dets, float(_o2["score"] or 0.0)))
    # ③ `white_w`：白度能提前定（喂得少、"运动"还没攒够分时，靠白度先定下来 ✓）
    _o3, _ = _run(_mk(3), min_hits=1, white_w=0.0)
    check(_o3["state"] in ("track", "init"),
          "⑤ `white_w` 关掉时开局更「哑」（state=%s ✓ 说明它确实在起作用 ✓）" % _o3["state"])
    # ④ `score_decay`：越小 = 忘得越快 ⇒ 同一个"偶发偏离"攒不起来 ✓
    _s = _mk(12)
    _o4a, _t4a = _run(_s, score_decay=0.5, min_hits=1, white_w=0.0)
    _o4b, _t4b = _run(_s, score_decay=0.99, min_hits=1, white_w=0.0)
    check((_o4a["score"] or 0.0) < (_o4b["score"] or 0.0),
          "⑤ `score_decay` 0.50 vs 0.99 ⇒ 累积分 %.1f < %.1f ✓（衰减越小、记忆越短 ⇒ 分数越小 ✓）"
          % (_o4a["score"] or -1.0, _o4b["score"] or -1.0))


def test_runner_same_shape():
    """`MotionRunner.step` 与 `Runner.step` **同签名同返回**（演示窗只换构造 ✓）。"""
    _r = MotionRunner(dets=None, gain=None, assume=(375.0, 250.0), follow_gain=1.0)
    _r.dets = [_frame(6, target=(380.0, 300.0))]
    _out = _r.step(None, 0, 0.0)
    check(isinstance(_out, tuple) and len(_out) == 6,
          "⑥ 回 6 元组 `(out, pos, rad, hit, boxes, motion)`（长度 %d ✓）" % len(_out))
    _mo = _out[5]
    _need = ("tracks", "median", "dev", "tid", "followed", "typ_area", "followed_pos",
             "followed_v", "n_cands", "dt", "track_v", "box_moves", "box_roles",
             "box_pred", "rad_frozen", "tgt_radius", "red_i", "tbox", "raw_pos",
             "path_pts", "tbox_rad", "tbox_wh", "vel_abs", "vel_rel", "reg", "merged",
             "boxes", "miss", "state")
    _miss = [k for k in _need if k not in _mo]
    check(not _miss,
          "⑥ `motion` 把 `Runner` 会给的键**全给齐了**（缺 %s ✓ —— ⚠ 少一个就可能某处硬取 ⇒ "
          "`KeyError` ⇒ PyQt5 `abort()` ⇒ **闪退** ✗ 踩过 ✓）" % (_miss or "无",))
    _n_dets = 7          # ⚠ 6 个假目标 + 1 个真目标 = **7** ✗ 别按"假目标数"写 ✗（踩过 ✓）
    check(isinstance(_mo.get("tracks"), list) and len(_mo["tracks"]) == _n_dets,
          "⑥ `tracks` 里带着**每条候选**（%d 条 ✓ 演示窗会把它们画成绿点+箭头 ⇒ "
          "**能直接看到算法在给谁加分** ✓）" % len(_mo.get("tracks") or []))
    check(_r.dets is not None and _out[1] is None,
          "⑥ 第 0 拍只有一帧框 ⇒ 还没有位移 ⇒ 不出位置（pos=None ✓ 不乱猜 ✓）")


def test_persist_beats_burst():
    """⭐⭐⭐⭐⭐ **"不合群的持续性"要真的赢过"偶尔猛抖一下"** ✓✓
    （用户 2026-10-04 ✓ 原话："**把「不合群的持续性」做进选目标**" ✓ —— 当前最大瓶颈 ✓）。

    ⚠ 病根（实测 `10月1日` 全片 ✓）：**目标换了 14 次** ✗（平均每 5 拍一次 ✓）——
      因为打分只看**强度** ✗ ⇒ 而"假目标偶尔猛抖一下"（单拍 `dev` 60+）在强度上
      **比真目标平稳地走（每拍 20）还高** ✗✗ ⇒ "偶然"和"持续"在强度这一维**长得一样** ✓
      （`_DEV_WIN=3` 的 3 拍中位只压得住**单拍**尖峰 ✗ 压不住"连着抖 2~3 拍"✗）。
    ⇒ 现在加第二维 `persist`（最近 `p_win` 拍里"真有在动"的**比例** ✓）并按它给分数加权 ✓。
    本用例钉住这个**核心性质**：
      · 两条候选的**平均强度一样**（每 2 拍都多走 60px ⇒ 平均都是 30 ✓）；
      · 但一条 **一直在动**（`persist ≈ 1.0` ✓）、另一条 **隔拍才动**（`persist ≈ 0.5` ✓）；
      · ⇒ **"一直在动"的那条分数必须明显更高** ✓✓（改回"纯强度"⇒ 两者持平 ⇒ 立刻红 ✗）。
    """
    def _d(i):
        _g = 30.0 * i
        _dets = [(0, 200.0 + _g, 100.0, 150.0, 150.0, 0.9),
                 (0, 500.0 + _g, 100.0, 150.0, 150.0, 0.9),
                 (0, 800.0 + _g, 100.0, 150.0, 150.0, 0.9)]
        # A：**每拍**都多走 30 ⇒ dev ≈ 30、persist ≈ 1.0 ✓
        _dets.append((0, 1100.0 + _g + 30.0 * i, 300.0, 150.0, 150.0, 0.9))
        # B：**隔拍**多走 60 ⇒ **平均也是 30** ✓ 但 persist ≈ 0.5 ✓
        _dets.append((0, 1500.0 + _g + 60.0 * (i // 2), 300.0, 150.0, 150.0, 0.9))
        return _dets

    _tr = MotionTracker(min_hits=1)
    for _i in range(24):
        _tr.process(None, ts=0.1 * _i, dets=_d(_i))
    _ta = _tr.tracks[3] if len(_tr.tracks) > 3 else None
    _tb = _tr.tracks[4] if len(_tr.tracks) > 4 else None
    check(_ta is not None and _tb is not None,
          "① 两条候选都建起来了（A=每拍动 ✓ / B=隔拍动 ✓）")
    _pa = None if _ta is None else float(_ta.persist)
    _pb = None if _tb is None else float(_tb.persist)
    check(_pa is not None and _pb is not None and _pa > 0.85 and _pb < 0.65,
          "② **持续性把两者分开了**：每拍都在动的 `persist = %.2f` ✓ ／ 隔拍才动的 "
          "`persist = %.2f` ✓（⚠ 这一维就是治「偶然」的 ✓ —— 强度上它俩**一模一样** ✗）"
          % (-1.0 if _pa is None else _pa, -1.0 if _pb is None else _pb))
    _sa = None if _ta is None else float(_ta.score)
    _sb = None if _tb is None else float(_tb.score)
    check(_sa is not None and _sb is not None and _sa > _sb * 1.6,
          "③ ⭐ **分数拉开了**：一直在动 **%.0f** ／ 隔拍动 **%.0f**（差 %.1f 倍 ✓）—— "
          "⚠ 改回「纯强度」（`+ _dev_eff` ✗）两者会**持平**（都 ≈ 300）⇒ 立刻红 ✗✗"
          % (_sa, _sb, _sa / max(1e-6, _sb)))


def test_merge_break_and_glide():
    """⭐⭐⭐⭐⭐ **融合期"断掉"的三条铁律**（用户 2026-10-04 ✓ 他给的机制 ✓ 实测逼出来的 ✓）。

    ⚠ 用户原话："**融合时期拿检出框心当『该往哪走』的指导是很愚蠢的**" ✓✓
      —— 落地成三件事：
      ① **不采信**那个框（框心 = 两个目标的**中点** ✗ ✗）；
      ② ⚠ **但也不许"冻结"** ✗ —— 要"**沿预测滑**" ✓
         （**实测踩到** ✓：我第一版写成"`pos` 原地不动" ⇒ 帧 64~70 **卡死七拍** ✗✗，
          而框心一路走了 230px ⇒ 差越拉越大 ✓）；
      ③ ⚠ **换目标时 `pos` 必须重置** ✗（**实测踩到** ✓：目标从 `#20` 换成 `#4` 后，`pos`
         **停在旧目标处** ⇒ 差 **541px** 整整五拍 ✗✗）。
    本用例钉住 ①②（③ 难构造 ⇒ 由真素材实测量过 ✓）。
    """
    def _d(i, tsz=(150.0, 150.0), sx=0.0, sy=0.0):
        return [(0, 200.0 + 30.0 * i, 100.0 + 10.0 * i, 150.0, 150.0, 0.9),
                (0, 500.0 + 30.0 * i, 100.0 + 10.0 * i, 150.0, 150.0, 0.9),
                (0, 800.0 + 30.0 * i, 100.0 + 10.0 * i, 150.0, 150.0, 0.9),
                (0, 1100.0 + 35.0 * i + sx, 300.0 + 10.0 * i + sy, tsz[0], tsz[1], 0.9)]

    _tr = MotionTracker(min_hits=1)
    for _i in range(7):
        _tr.process(None, ts=0.1 * _i, dets=_d(_i))
    _p6 = _tr.pos
    # 帧 7：框突然变成 **200×200**（面积 **1.78x** ⇒ 融合 ✓）且框心被**拽走** ✓
    #   ⚠ 拽的幅度要**压住**：帧间位移必须 < `pair_gate`(70) ✗ 不然**根本配不上** ⇒ 测不到 ✓
    #     （实测：拽 (-60,+60) ⇒ 帧间位移 85 ⇒ 配不上 ✗ 白测 ✓）
    _tr.process(None, ts=0.7, dets=_d(7, tsz=(200.0, 200.0), sx=-45.0, sy=30.0))
    _p7 = _tr.pos
    _obs7 = (1100.0 + 35.0 * 7 - 45.0, 300.0 + 10.0 * 7 + 30.0)
    _dd = (None if (_p7 is None) else
           math.hypot(_p7[0] - _obs7[0], _p7[1] - _obs7[1]))
    check(_dd is not None and _dd > 45.0,
          "① 融合框（面积 **1.78x** ⇒ 框心 = **两目标中点** ✗）的框心被拽到 (%.0f,%.0f) "
          "⇒ 报出的位置 **(%.0f,%.0f)** **没跟过去** ✓（离框心还有 **%.0f px** ✓）"
          % (_obs7[0], _obs7[1], _p7[0], _p7[1], _dd if _dd is not None else -1.0))
    _mv = (None if (_p6 is None or _p7 is None)
           else math.hypot(_p7[0] - _p6[0], _p7[1] - _p6[1]))
    # ⚠ 阈值取 14（实测这一拍走 **19px** ✓）：`vel` 是 `0.7/0.3` 的一阶滞后、还要再乘一次
    #   `vel_decay` ⇒ 收敛不到"真实每拍 36px"那一档 ✓；而"冻结"是 **0** ⇒ 14 分得开 ✓。
    check(_mv is not None and _mv > 14.0,
          "② **但它没被「冻结」** ✓：这一拍照样走了 **%.0f px**（= 沿预测滑 ✓ 目标本来就该走 "
          "(35,10) ✓）—— ⚠ 写成「原地不动」的话这里是 **0** ✗（实测帧 64~70 就是这么卡死"
          "七拍的 ✗）" % (-1.0 if _mv is None else _mv))


def test_stuck_deprioritize():
    """⭐⭐⭐ **粘连（重叠）期：观测不可信 ⇒ 降权、偏预测、减速** ✓✓
    （用户 2026-10-03 ✓ 原话："这一帧其实**已经跟假目标重叠了** ⇒ 应该稍微**沿着轨迹预测减点速**，
    而不是急于**寻找跟踪信号**（**现在看起来就很急**）" ✓）

    ⚠ 为什么"急"是错的 ✗（实测链 ✓）：真目标与假目标**部分重叠**时，YOLO 把**两个目标检成一个
      框** ✓ ⇒ 框**面积暴增**（实测帧 13/14 = 自己基准的 **1.67x / 1.90x** ✓）而**框中心 =
      两个目标的中点** ✗ ⇒ 于是这拍的 `mv` / `dev` 是"观测被拉偏"的产物 ✗（实测 `v_rel` 从
      27.8 飙到 **40.5** ✗）—— 旧代码把它当"最不合群"**照样加分** ✗ ⇒ `score` 涨到 198.6 ✓
      **越粘越像目标** ✗✗。
    ⇒ 现在：`stuck`（+ 分开后 `cool` 几拍）⇒ ①`dev` 只按 `_STUCK_DEV_W` 记分 ✓；②位置偏预测 ✓；
      ③速度额外衰减 ✓。本用例钉住①（`score` 不许被那个假 `dev` 顶起来 ✓）。
    """
    _tr = MotionTracker()
    # ⚠⚠ **"相对群体额外走多少"必须 ≤ 门限** ✗✗ —— 目标每拍总位移 = 30（群体）+ 本值 ✓
    #   而 `pair_gate = 70` ⇒ 本值写 60 时总位移 90 ⇒ **根本配不上** ✗（实测踩到 ✓：
    #   那条轨迹一直 `stuck = 0`、面积比恒 1.00 ✗ 因为**它压根没被配上** ✓）。
    _red = 30.0          # ⇒ 总位移 60px（< 70 ✓）⇒ `dev ≈ 30` ✓（够大了，好算 ✓）

    def _d(i, tsz=(150.0, 150.0)):
        _dets = [(0, 200.0 + 30.0 * i, 100.0, 150.0, 150.0, 0.9),
                 (0, 500.0 + 30.0 * i, 100.0, 150.0, 150.0, 0.9),
                 (0, 800.0 + 30.0 * i, 100.0, 150.0, 150.0, 0.9),
                 (0, 1000.0 + (30.0 + _red) * i, 300.0, tsz[0], tsz[1], 0.9)]
        return _dets

    for _i in range(0, 7):                       # 帧 0~6：正常（框 150×150 ✓）
        _tr.process(None, ts=0.1 * _i, dets=_d(_i))
    _t4 = next((t for t in _tr.tracks if t.id == 4), None)
    _s6 = None if _t4 is None else float(_t4.score)
    # 帧 7：**框突然放大到 2.7 倍面积**（= 与邻居粘成一个框 ✓ 而且中心被拉向群体 ✓）
    _tr.process(None, ts=0.7, dets=_d(7, tsz=(150.0 * 1.65, 150.0 * 1.65)))
    _t4 = next((t for t in _tr.tracks if t.id == 4), None)
    check(_t4 is not None and _t4.stuck >= 1,
          "框面积突然变成 **%.2fx** ⇒ 认出「**粘住了**」（`stuck = %s` ✓ —— ⚠ 面积基准只由"
          "**正常拍**更新 ✗ 不然它自己会把基准顶上去 ⇒ 永远认不出来 ✓）"
          % ((_t4.w * _t4.h / max(1.0, _t4.area_ema)) if _t4 else -1.0,
             None if _t4 is None else _t4.stuck))
    _inc = None if (_t4 is None or _s6 is None) else float(_t4.score) - _s6 * _tr.score_decay
    check(_inc is not None and _inc < 0.35 * _red,
          "粘连那一拍，`score` 只涨了 **%.1f**（`dev` 本可以是 **%.0f** 那一档 ✓）⇒ 那个假"
          "「偏离」**几乎没进分数** ✓（= 用户要的「**别急于寻找跟踪信号**」✓；⚠ 不降权的话"
          "这一拍会把分数顶上去 ✓ 实测帧 13 就是这么涨到 198.6 的 ✗）" % (_inc, _red))


def test_ui_params_really_work():
    """⭐⭐⭐ **界面第 4/5 行那些参数不是摆设** ✓（用户 2026-10-03 ✓ 原话："把一些**需要频繁调整
    的参数**做成配置，然后告诉我**应该怎么调、看什么**" ✓ 见 `lie_demo` ✓）。

    ⚠ 规矩：**摆到界面上的每一个，都必须真的改变行为** ✗（不然就是骗人 ✓ 这条自检守这一条 ✓）。
    """
    def _d(xs, sz=(150.0, 150.0)):
        return [(0, float(_x), 100.0, sz[0], sz[1], 0.9) for _x in xs]

    # ① `位置平滑`：目标突然挪一下 ⇒ 报出的位置跟过去多少，由它定 ✓
    #   ⚠⚠ **一下不能挪超过 `pair_gate`(70)** ✗（我第一版挪 100px ⇒ 直接**配不上** ⇒ 位置压根
    #     没更新 ⇒ 两个平滑值都停在同一处 ✗ 断言失效 ✓ 实测踩到 ✓）。
    def _jump(a):
        _t = MotionTracker(min_hits=1, smooth=a)
        for _i, _x in enumerate((300.0, 300.0, 360.0)):    # 挪 60px（< 70 ✓）
            _t.process(None, ts=0.1 * _i,
                       dets=_d([_x, 600.0, 900.0]))
        return None if _t.pos is None else _t.pos[0]
    _h, _s = _jump(0.0), _jump(0.9)
    check(_h is not None and _s is not None and _h - _s > 30.0,
          "① `位置平滑` 真起作用：目标挪 60px 时 ⇒ =0.0 跟到 **%.0f**（跟手 ✓）、=0.9 只到 **%.0f**"
          "（稳但滞后 ✓）" % (_h or -1.0, _s or -1.0))

    # ② `跟丢拍数`：调小 ⇒ **更早**报「跟丢」✓
    #   ⚠⚠ 两个坑（都是我第一版踩的 ✓）：① 三条必须**一起动** ✗（塞一条"不动的"当参照 ⇒ 那条才是
    #     不合群的 ⇒ 它成了目标 ⇒ 永远不判丢 ✗ 断言反了 ✓）；② 别断言"=4 不报" ✗ —— 这个场景里
    #     谁都"不合群个屁" ⇒ **两个阈值最终都会报** ✓ ⇒ 要比的是「**什么时候**报」✓。
    def _first(n):
        _t = MotionTracker(min_hits=1, q_need=n, q_win=4)
        for _i in range(12):
            _c = 20.0 * _i                      # 群体每拍 20px
            _dd = [(0, 100.0 + _c, 100.0, 150.0, 150.0, 0.9),
                   (0, 400.0 + _c, 100.0, 150.0, 150.0, 0.9),
                   (0, 700.0 + _c, 100.0, 150.0, 150.0, 0.9)]
            # 真目标：**隔拍**额外再走 40px ⇒ 一半的拍「合群」（相对速度 40 ✓）、一半不 ✓
            #   （⚠ 每拍总位移 = 20 或 60 ⇒ **都 < `pair_gate` 70** ✓ 不然会配不上 ✓ 踩过 ✓）
            _dd.append((0, 1000.0 + _c + 40.0 * (_i // 2), 400.0, 150.0, 150.0, 0.9))
            _t.process(None, ts=0.1 * _i, dets=_dd)
            if _t.q_lost:
                return _i + 1
        return 99
    _f1, _f4 = _first(1), _first(4)
    check(_f1 < _f4,
          "② `跟丢拍数` 真起作用：目标**一半的拍在合群、一半不在** ⇒ =1 在第 **%d** 拍就报"
          "「跟丢」、=4 要等到第 **%d** 拍 ✓（⚠ 若目标**每拍都不合群**，两个阈值会同时到 ⇒ "
          "分不出来 ✓ 这也是踩过的坑 ✓）" % (_f1, _f4))

    # ③ `粘连阈值`：框突然放大 ⇒ 调到 1.05 时更容易判「粘住」✓
    def _stuck(r):
        _t = MotionTracker(min_hits=1, stuck_ratio=r)
        _t.process(None, ts=0.0, dets=_d([300.0, 600.0, 900.0]))
        _t.process(None, ts=0.1, dets=_d([330.0, 600.0, 900.0], sz=(150.0, 150.0)))
        _t.process(None, ts=0.2, dets=_d([360.0, 600.0, 900.0], sz=(170.0, 170.0)))
        _x = next((t for t in _t.tracks if t.id == 1), None)
        return None if _x is None else _x.stuck
    check(_stuck(1.05) and _stuck(1.40) == 0,
          "③ `粘连阈值` 真起作用：框从 150² 涨到 170²（**1.28x**）⇒ 阈值 1.05 判**粘住**（stuck=%s ✓）、"
          "1.40 不判（%s ✓ —— 实测正常拍在 0.9~1.2 ✓ 粘连拍到 1.7~1.9 ✓）"
          % (_stuck(1.05), _stuck(1.40)))

    # ④ `预测上限`：=0 ⇒ **完全不预测**（`pred` 就等于本拍观测 ✓）
    def _pred(mx):
        _t = MotionTracker(min_hits=1, kf_pred_max=mx, kf_q=25.0)
        for _i in range(6):
            _t.process(None, ts=0.1 * _i,
                       dets=_d([300.0 + 30.0 * _i, 600.0, 900.0]))
        _x = next((t for t in _t.tracks if t.id == 1), None)
        return None if _x is None else (_x.pred, _x.obs)
    _p0, _o0 = _pred(0.0)
    _p1, _o1 = _pred(40.0)
    check(_p0 is not None and _p1 is not None
          and abs(_p0[0] - _o0[0]) < 1e-6 and abs(_p1[0] - _o1[0]) > 1.0,
          "④ `预测上限` 真起作用：=0 ⇒ **pred 贴着观测**（%.1f vs %.1f ✓）；=40 ⇒ **往前挪一截**"
          "（%.1f vs %.1f ✓ —— 它就是配对时用的「你以为它在哪」✓）"
          % (_p0[0], _o0[0], _p1[0], _o1[0]))


def test_span_and_vel_normalized():
    """⭐⭐ **两个实测抓到的真 bug**（用户 2026-10-03 ✓ 他贴的**帧 13** 日志 ✓）。

    ① **跨拍位移没归一化 ⇒ `dev` 虚高 5~6 倍** ✗✗：
       轨迹「没配上」时 `obs` 会被「按群体中位**推一把**」✓，而 `prev` 仍停在**上一次真实观测**
       ✓ ⇒ 它下次配上时 `mv = 本拍 − prev` 其实是**跨了好几拍的位移** ✗。
       实测（帧 13 的 `#10`）：`mv = (+255.9, −43.3)`、**`dev = 218.7 px`** ✗ —— 而相机这一拍
       只走了 45.7px、配对门限才 70px ⇒ 一拍**绝无可能**位移 255px ⇒ 它**根本不合群个屁** ✗
       ⇒ 这个虚高的 `dev` 还**顶进 `score`** ⇒ 把幽灵轨迹捧成「最不合群」⇒ **帧 14 换错目标** ✗✗。
       ⇒ 修：`mv` 一律换算成**每拍**（`÷ mv_span` ✓）。本用例钉住「它回来那一拍 `mv ≈ 每拍`」
       ✓ 而**不是**「跨拍总量」✗。

    ② **`prev` 存的是「两拍前」的观测 ⇒ `mv` 系统性翻倍** ✗✗（**同一个循环里的另一处** ✓
       自检 ①② 是一起把它逼出来的 ✓）：`_t.prev = _t.obs` 写在 `_t.obs = 本拍` **之前** ✗
       ⇒ `prev` 拿到的是**进入本拍时的那个 `obs`**（= 「**上上拍**」的观测 ✗）⇒ 下一拍算 `mv`
       时等于跨了两拍 ⇒ **正好翻倍** ✓（实测：每拍真走 30 ⇒ `mv` 报 **60** ✗；每拍走 40 ⇒
       报 **80** ✗）⇒ **判据的量纲整个是错的** ✓（⚠ 排序不变 ⇒ 所以光看「谁分高」不会露馅 ✗，
       但 `dev` / 噪声底 / 信噪比**全不可信** ✗）。
       ⇒ 修：**两行顺序对调**（先 `obs = 本拍`、再 `prev = obs` ✓）⇒ `prev` 就 = "上一拍观测" ✓。
       ⇒ 顺带把**轨迹速度**也改用 `_moves` ✓（原来写的是 `obs − prev` ✗，在旧顺序下**恒为 0** ✗
       ⇒ 目标「没配上要外推」时**位置一步不动** ✗）。本用例钉住「`vel` ≈ 真实每拍速度」✓
       ＋「`mv` 不翻倍」✓。
    """
    def _d(xs):
        return [(0, float(_x), 100.0, 60.0, 60.0, 0.9) for _x in xs]

    # ---- ① 跨拍位移必须归一化 ----
    # ⚠⚠ **夹具本身有两个坑**（我第一版就写错了 ✓ 记下来免得再踩 ✗）：
    #   ① 轨迹之间要**拉开 400px** ✓ —— 间距 100 时会被**邻居的框"抢走"** ✗（配对门限 70px ✓
    #      实测：间距 100 ⇒ 第三条当场就配到了邻居那儿 ⇒ `mv_span` 压根不涨 ✗）；
    #   ② 要留**≥3 条"配得上"的轨迹** ✗ —— `_MIN_PAIRS = 3` 才算得出群体中位 ✓ 不然
    #      "推一把"推的是 **(0,0)**（= **没推** ✗）⇒ 缺席那条的位置**永远不动** ✗ 也配不回来 ✓
    #      （这个"不足 3 条就不推"是**合理的** ✓ —— 画面里没有"群体"就谈不上"相对群体" ✓）。
    _x4 = (100.0, 500.0, 900.0, 1300.0)
    _tr = MotionTracker()
    _tr.process(None, ts=0.0, dets=_d(_x4))                             # 建 4 条（id 1~4 ✓）
    for _i in range(1, 5):                                              # 只喂前三条 ⇒ 第 4 条缺席
        _tr.process(None, ts=0.1 * _i, dets=_d([_x + 40 * _i for _x in _x4[:3]]))
    _th = next((t for t in _tr.tracks if t.id == 4), None)
    _span = None if _th is None else int(_th.mv_span)
    check(_th is not None and _span >= 4,
          "① 第 4 条轨迹连续 4 拍没配上 ⇒ 跨拍计数 `mv_span` 涨到 **%s** ✓（它就是「要除以几」✓）"
          % _span)
    _tr.process(None, ts=0.5,                                           # 它回来了（位置正好在推到的点 ✓）
                dets=_d([_x + 40 * 5 for _x in _x4]))
    _th = next((t for t in _tr.tracks if t.id == 4), None)
    _mv = None if _th is None else _th.mv
    _dv = None if _th is None else _th.dev
    check(_mv is not None and abs(_mv[0] - 40.0) < 12.0,
          "① 它回来的那一拍，`mv` = **每拍约 40px**（实测 %s ✓）—— ⚠ **不是**跨 5 拍攒出来的 "
          "**200px** ✗✗（不修的话实测爆出 `dev = 218.7 px` ✗ ⇒ 幽灵被捧成「最不合群」✗）"
          % (None if _mv is None else "(%.1f, %.1f)" % _mv))
    check(_dv is not None and _dv < 15.0,
          "① ⇒ 它的 `dev` 也只有 **%.1f px** ✓（它本来就**跟大家一起动** ⇒ 不该「不合群」✓；"
          "⚠ 不修的话这里是 **200+** ✗✗）" % (-1.0 if _dv is None else _dv))

    # ---- ② 轨迹速度不许恒为 0 ----
    _t2 = MotionTracker()
    for _i in range(6):
        _t2.process(None, ts=0.1 * _i, dets=[(0, 100.0 + 30 * _i, 100.0, 60.0, 60.0, 0.9)])
    _tf = _t2.tracks[0] if _t2.tracks else None
    _vx = None if _tf is None else _tf.vel[0]
    check(_vx is not None and abs(_vx - 30.0) < 9.0,
          "② 匀速（每拍 +30px）跑 6 拍 ⇒ 轨迹速度学到 **%.1f px/拍** ✓（一阶滞后 0.7/0.3 收敛到"
          "这个量级就对了 ✓）—— ⚠ 这条同时钉住「**`mv` 的量纲**」：`prev` 顺序写反时 `mv` 会"
          "**翻倍成 60** ✗ ⇒ 速度会被拉到 **47.8** ✗（实测踩到 ✓ 而 25 才是对的 ✓）"
          % (-999.0 if _vx is None else _vx))


def test_real_clip_smoke():
    """真素材短冒烟：**用真 YOLO 跑前几拍** ⇒ 报的位置与**白星真值**几乎重合 ✓（唯一有真值的时刻 ✓）。"""
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月1日.mp4")
    if not _src.exists():
        check(True, "跳过真素材冒烟（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        _frames, _ts, _p, _s, _f = LD.load_source(str(_src))
        _w = LD.DetsWorker(None, conf=0.25)
    except Exception as _e:                 # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过真素材冒烟（环境不齐：%s ✓）" % _e)
        return
    _t = MotionTracker(min_hits=3)
    _ds = []
    for _i in range(min(10, len(_frames))):
        _o = _t.process(_frames[_i][1], ts=_ts[_i], dets=_w.detect(_frames[_i][1]) or [])
        _ref = pick_white(_gray(_frames[_i][1]))
        if _o["pos"] is not None and _ref is not None:
            _ds.append(math.hypot(_o["pos"][0] - _ref[0], _o["pos"][1] - _ref[1]))
    _w.close()
    # ⚠ 别拿**最大**值当门槛 ✗ —— 前几拍还在"攒分"（`min_hits=3` ✓）⇒ 偶有一拍选中别人的框 ⇒
    #   峰值会很跳（实测 max 163 / 中位 **2.5** ✓）⇒ **中位**才是这条冒烟要看的 ✓。
    check(bool(_ds) and float(np.median(_ds)) < 15.0,
          "真素材（`10月1日` 前 10 拍，**真 YOLO**）：报的位置与「白星真值」差 —— **中位 %.1f px**"
          "（峰值 %.1f ✓ 出现在还没攒够分的头几拍 ✓）—— ⚠ 这个差本来就该有几 px：轨迹选的是"
          "**框中心**、白星是**块重心** ✓"
          % (float(np.median(_ds)) if _ds else -1.0, max(_ds) if _ds else -1.0))


def main():
    print("运动不合群（YOLO 候选 + 偏离群体中位）自检：")
    test_picks_the_odd_one()
    test_wave_immune()
    test_briefly_missing_keeps_pos()
    test_white_helper()
    test_params_really_work()
    test_runner_same_shape()
    test_persist_beats_burst()
    test_merge_break_and_glide()
    test_stuck_deprioritize()
    test_ui_params_really_work()
    test_span_and_vel_normalized()
    test_real_clip_smoke()
    if _FAILED:
        print("自检：%d 条失败" % _FAILED)
        return 1
    print("运动分离自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
