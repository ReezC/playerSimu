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

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.lie_motion import (MotionRunner, MotionTracker,  # noqa: E402
                                   _gray, pick_white, _WHITE_MATCH,
                                   _M_OK, _M_SUS, _PAIR_GATE)
import perception.lie_motion as _LM                            # noqa: E402

# ⭐⭐⭐⭐⭐ **自检里默认**关掉「**拆框**」（用户 2026-10-05 ✓ 见 `_FUSE_SPLIT` ✓）——
#   ⚠⚠ 它是**位置硬约束**（把圆按到"目标那一块"的正中 ✓ 见 `process` 那段 ✓）⇒ **谁量"指引 /
#     夹取在位置上的效果"，都会被它盖掉** ✗✗（**实测**：一开就 3 个用例红 ✓ 而那些用例量的是
#     **门控 / 隔拍补偿**，代码一个字没动 ✓）。⇒ **全局关** ✓，只有**专门那一条**（
#     `test_fuse_split_places_circle_on_target_half` ✓）显式传 `fuse_split=True` 打开 ✓。
#   ⚠ 与既有那套一致 ✓（`test_fuse_pull_gate` 本来就在改 `_LM._FUSE_PULL` ✓）。
_LM._FUSE_SPLIT = False

# ⭐⭐⭐⭐⭐ **自检里默认关掉"真目标丢失 / 重新找到"那份日志** ✗✗（`logs/lie_target.log` ✓ 用户
#   2026-10-07 ✓）—— ⚠⚠ **实测踩到** ✓：本文件里那些 `VelocityTracker.process` 一跑，就把测试行
#   写进**用户那份日志**（实测四条"帧 -" ✓）⇒ **自检一个字节都不许落到用户日志里** ✓。
#   ⚠ 专门验日志那条（`test_velocity_log_lost_and_found` ✓）自己把路径指到**临时目录** ✓ 验完还原 ✓
#     —— 那是"量日志" ✓ 与"别污染用户日志"不冲突 ✓（同 `_FUSE_SPLIT` 那个全局开关的先例 ✓）。
_LM.set_vel_log_path(None)

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
        # ⚠⚠⚠ **"真目标"必须放列表最前** ✗✗（用户 2026-10-04 ✓ 我加了"三态"之后暴露的 ✓）——
        #   **第一拍所有轨迹的 `score` 都是 0** ✗ ⇒ `max(..., key=score)` 取的是**列表里第一个** ✓
        #   ⇒ 真目标若排在最后（原来就是 ✗）⇒ **开局就选到假目标** ✗ ⇒ 而新逻辑
        #   "**跟得上就不换**" ✓ ⇒ 它会**一直跟着那个假目标**（假目标 `dev=0` ⇒ 分数恒 0 ✗✗）
        #   ⇒ 断言"两个 decay 的分数不同"就**测不出东西**（都是 0 ✓ 实测踩到 ✓）。
        #   ⚠ 这不是"场景凑数" ✗ —— 它顺手记下一个**真发现**：**初始选错 ⇒ 新架构下很难自己
        #     纠正回来**（见 `docs/测谎设计.md` 那条"待办"✓）。
        _s = []
        for _i in range(n):
            _d = [(0, 380.0 + 7.0 * _i + dev * _i, 300.0 + 3.0 * _i, 150.0, 150.0, 0.9)]
            _d = _d + _shift(_fake0, 7.0 * _i, 3.0 * _i)
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


# ⚠⚠ **`test_persist_beats_burst` 已删除** ✗（用户 2026-10-05 ✓ 原话："**不需要持续门这种逻辑，
#   整体移除**" ✓✓）—— 它原来钉的是"一直在动 ⇒ `persist` 高 ⇒ 分数明显更高" ✓；而
#   `persist` / `score` 按持续性加权**都已拆掉** ✓ ⇒ 这条用例**没有被测对象了** ✓。
#   ⇒ 判别"这格框算不算个真目标"现在钉在 `test_forced_claim_rule` 里 ✓（= "未被记录的干净框 ＋
#     与圆距离在范围内" ✓ 见 `perception.lie_motion._forced_claim` ✓）。


def test_fuse_guide_holds_through_gap():
    """⭐⭐⭐⭐⭐ **融合期"目标那格框没配上"的几拍，框心指导要"接着拽"**（用户 2026-10-04 ✓ 原话：
    「**帧 16 开始没有看到任何框心指导的牵引感**」✓✓）。

    ⚠ 病根：那几拍 `_fuse_cands` 是**空的**（框不存在 ⇒ 没位移可算 ✗）⇒ `_fd = None` ⇒ **完全不拽** ✗。
    实测（真素材 A/B，力度 1.0 vs 0 的报出位置差）：帧 13→16 一路涨 **3.2→4.8→7.6→19.3** ✓
      而**帧 16 之后不再增长** ✗✗ —— 正是用户报的现象 ✓。
    本用例钉：**漏框那几拍，两条（1.0 / 0）的差距必须继续变大** ✓
      （⚠ 改回"没候选就 `_fd = None`" ⇒ 那几拍涨不动 ⇒ 立刻红 ✗）。
    """
    def _seq(_gap=range(11, 14)):
        _f0 = _frame(9)
        _s = []
        for _i in range(16):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _sz = 200.0 if _i >= 8 else 150.0          # 帧 8 起融合 ✓
            if _i not in _gap:                          # ⚠ 漏框那几拍**整格都不给** ✓
                _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1],
                            _sz, _sz, 0.9)]
            _s.append(_d)
        return _s

    def _run(_pull):
        _t = MotionTracker(min_hits=3, fuse_pull=_pull)
        _ps = []
        for _i, _d in enumerate(_seq()):
            _o = _t.process(None, ts=_i * 0.16, dets=_d)
            _ps.append(_o["pos"])
        return _ps

    # ⚠⚠⚠ **本用例量的是「框心指导」** ✗ ⇒ 必须先把它**临时打开** ✓（用户 2026-10-05 把总开关
    #   `_LM._FUSE_GUIDE` 关成 `False` 了 ✓ —— 关掉时这条当然"拽不动"✗，与本用例要量的事无关 ✓）。
    _LM._FUSE_GUIDE = True
    _a, _b = _run(1.0), _run(0.0)

    def _dif(_k):
        _x, _y = _a[_k], _b[_k]
        return (None if (_x is None or _y is None)
                else math.hypot(_x[0] - _y[0], _x[1] - _y[1]))

    _d10, _d13 = _dif(10), _dif(13)
    check(_d10 is not None and _d13 is not None and _d13 - _d10 > 0.5,
          "融合期**漏框 3 拍（帧 11~13）时指导仍在拽**：力度 1.0 vs 0 的差 "
          "**%.2f → %.2f px**（涨 **%.2f** ✓）—— ⚠ 改回「没候选就不拽」⇒ 这 3 拍涨不动 ⇒ 立刻红 ✗"
          % (_d10, _d13, _d13 - _d10))
    _LM._FUSE_GUIDE = False           # ⚠ **还原成"自检的默认"（= 关 ✓）**，别泄漏成开 ✗


def test_fuse_guide_switch():
    """⭐⭐⭐⭐⭐ **「框心指导」总开关**（用户 2026-10-05 ✓ 原话："**把框心指导暂时关掉**" ✓✓）——
    `_LM._FUSE_GUIDE = False`（**默认 = 用户的现状** ✓）⇒ `fuse_pull` 拽不拽**逐位一致** ✓；
    改回 `True` ⇒ 效果**立刻回来** ✓。

    ⚠ 为什么要钉它 ✗：这个开关是"**用户当轮要的现状**" ✓，不是我能随手改的东西 ✓ ——
      · 谁把它删掉/绕过去 ⇒ ① 红 ✗（框心指导又偷偷开始拽 ✓）；
      · 谁把它改成恒 `False`、"永久关死"✗ ⇒ ② 红 ✗（那就再也没法开回去了 ✓）。
    ⚠ 用**同一段夹具**（帧 8 起融合 ⇒ `stuck>0` ⇒ 有指导 ✓），只翻开关 ✓ 两条互相对照 ✓。
    """
    _f0 = _frame(9)
    _seq = []
    for _i in range(16):
        _cam = (7.0 * _i, 3.0 * _i)
        _d = _shift(_f0, _cam[0], _cam[1])
        _sz = 200.0 if _i >= 8 else 150.0          # 帧 8 起融合 ✓（面积 1.78x ⇒ stuck ✓）
        _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1], _sz, _sz, 0.9)]
        _seq.append(_d)

    def _last(_pull):
        _t = MotionTracker(min_hits=3, fuse_pull=_pull)
        _o = None
        for _i, _d in enumerate(_seq):
            _o = _t.process(None, ts=_i * 0.16, dets=_d)
        return _o["pos"]

    _LM._FUSE_GUIDE = False                        # = **用户的现状** ✓
    _a, _b = _last(1.0), _last(0.0)
    _d_off = (math.hypot(_a[0] - _b[0], _a[1] - _b[1]) if (_a and _b) else -1.0)
    _LM._FUSE_GUIDE = True
    _c, _e = _last(1.0), _last(0.0)
    _d_on = (math.hypot(_c[0] - _e[0], _c[1] - _e[1]) if (_c and _e) else -1.0)
    _LM._FUSE_GUIDE = False                        # ⚠ 立刻还原（别泄漏 ✗）
    #   ⚠⚠⚠ **口径变了（用户 2026-10-06 ✓）** ✗✗：`fuse_pull` 现在**兼"四条边转向比例"** ✓
    #     （"**参数用 k = 每拍转向比例 = 「框心力度」**" ✓✓）⇒ 开关关着时**还剩"转向"那一份** ✓
    #     ⇒ **不再是 0** ✗。⇒ 改成对照钉："**关着 ⇒ 只剩转向那份（小 ✓）**；**打开 ⇒ 明显更大**" ✓。
    check(0.0 < _d_off < _d_on,
          "① **开关关着（默认 ✓）⇒ 只剩「四条边转向」那一份** ✓（`fuse_pull` 1.0 vs 0.0 的报出"
          "位置差 **%.2f px** ✓〔关着〕＜ **%.2f px** ✓〔开着〕）—— ⚠ 关着时**不再是 0** ✗，"
          "因为 `fuse_pull` 兼当「转向比例」✓（用户 2026-10-06 ✓）；⚠ 若**开着也拉不开差**"
          " ⇒ 说明开关被绕开 ⇒ 本条红 ✗" % (_d_off, _d_on))
    check(_d_on > 0.01,
          "② **反面：改回 `True` ⇒ 效果立刻回来** ✓：同一段夹具、同两个力度 ⇒ 差 **%.2f px** ✓"
          "（⚠ 把开关改成「永久关死」✗ ⇒ 这条红 ✓ —— 那就再也没法开回去了 ✓）" % _d_on)


def test_frozen_label_in_fusion():
    """⭐⭐⭐⭐⭐ **融合期"非目标"的推导标签位置：相对群体必须静止**（用户 2026-10-04 ✓ 原话：
    「我们要冻住的是**标签位置**，也就是**融合框内假目标的框心**，而不是「融合框的框心」；
    真目标若正好在融合期在相对运动，它的 `rel` …不应该有任何影响」✓✓）。

    钉三条：
      ① 非目标进融合后，它的**推导位置**每拍相对群体 ≈ **0** ✓
        （⚠ 推进若走 `_med_ema`（**滞后** ✗）⇒ 每拍欠几 px ⇒ ① 立刻红 ✗ —— 真素材实测 3~4.7 px/拍 ✓）；
      ② **进融合第一拍不跳** ✓（⚠ 原来取"上一拍算好的 `pred`"⇒ 会跳十几~几十 px ✗，真素材实测 **22.7** ✓）；
      ③ **真目标不受影响** ✓：它在融合期的 `rel` 照旧按**自己的相对速度**演进（不是 0 ✓）。
    """
    def _scene():
        _f0 = _frame(8)
        _s = []
        for _i in range(16):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1],
                        150.0, 150.0, 0.9)]                 # 真目标（相对群体 +12px/拍 ✓）
            _sz = 200.0 if 6 <= _i <= 14 else 150.0         # ⭐ 一个**非目标**从帧 6 起融合 ✓
            # ⚠⚠⚠ **第 6 拍把它"挪到画面外"，不是抽掉** ✗✗（**实测踩到两次** ✓ 用户 2026-10-05 ✓）：
            #   ② 要钉的是"**从"没框"跨进融合滑行那一拍不许跳**" ✓ —— 而它原来**每拍都有框**
            #   ✗ ⇒ "上一拍没框"的局面**根本不出现** ⇒ ② 采不到样本 ⇒ 报 **-1.00** ✗（= 夹具
            #   退化了却还绿着 ✗ 更糟 ✓）。
            #   ⚠⚠ 而"**直接抽掉**它"也不行 ✗✗（我第一版就这么写 ✓）：那一拍它没框 ✓、下一拍那格
            #     200×200 **会被旁边的轨迹抢走** ✗（**实测**：真目标旁边那条 `#6` 接手 ⇒ 它自己的
            #     轨迹 `#10` 从此只剩 `miss` 一路滑行 ✗ ⇒ 要测的"进融合第一拍"**永远不出现** ✓）。
            #   ⇒ 挪到画面外（x≈2000 ✓）：**谁都不够近 ⇒ 没人抢** ✓、它自己的轨迹照旧留在原地
            #     滑行（`stuck` 计数不断 ✓）⇒ 下一拍回来就是**干净的进融合第一拍** ✓✓。
            #   ⚠⚠ **R4 落地 ⇒ 这格框的落点必须重挑** ✗✗（2026-10-05 ✓ 用户原话："**改
            #     `test_frozen_label_in_fusion` 的夹具：那格非目标融合框挪到离圆足够远处 ✓，
            #     并保住 ② 样本**" ✓✓）——
            #     原来在 (390, 250)（≈ 画面正中 ✓）：而**圆就骑在真目标上**（本例真目标从 x=380
            #     起、每拍自己再 +12 ✓）⇒ 那格 200×200 的**半边正好罩住圆心** ✗ ⇒ **R4 一开，
            #     它就归真目标了** ✗ ⇒ 本条要的"**非目标的融合标签**"**根本不存在** ⇒ ② 丢样本 ✓
            #     （实测 **-1.00** ✓；我早先试过挪到 (150,250)，同样丢 ✗）。
            #   ⚠⚠⚠ **"离圆足够远"还不够，必须"离真目标的轨迹也足够远"** ✗✗（**实测踩到** ✓）：
            #     我第二版挪到 **(240, 250)**（环内偏左下 ✓ 离圆横向 140px ✓）—— 可**真目标那条的
            #     `pred` 一旦被拽歪**（它自己的 `vr` 被污染 ⇒ `pred` 甩到 (324,296) ✓）⇒ 那格 200 框
            #     就落到它**配对门之内** ⇒ **贪心把它抢走** ✗（实测帧 8~9 归真目标 ⇒ `rel` 从 36
            #     **跳到 476** ✗ ⇒ ③ 反而"更容易绿"✗✗ = **假绿** ✓）。
            #   ⇒ 落点定在 **(600, 120)**（**环外右上角** ✓），四条距离都量过：
            #     · 离**圆**：纵向**恒 180px** ✓ > 半高 100 ✓ ⇒ **圆永远不在它里面** ✓；
            #     · 离**真目标那条的轨迹** ≥ **187px** ✓（> 70 ✓ ⇒ 它的 `pred` 再歪也伸不到 ✓）；
            #     · 离**最近的环上假目标**（`_frame(8)` 的 (505,103) ✓）≈ **96px** ✓（> 70 ✓）；
            #     · 矩形也在**画面内**（x 500~700 ✓、y 20~220 ✓）。
            if _i == 5:
                _d = _d + [(0, 2000.0 + _cam[0], 1900.0 + _cam[1], _sz, _sz, 0.9)]
            else:
                _d = _d + [(0, 600.0 + _cam[0], 120.0 + _cam[1], _sz, _sz, 0.9)]
            _s.append(_d)
        return _s

    _t = MotionTracker(min_hits=3)
    _prev = {}
    _drift = []
    _jump = None
    _jump_allow = 3.0
    _rel6 = _rel9 = None
    for _i, _d in enumerate(_scene()):
        _t.process(None, ts=_i * 0.16, dets=_d)
        _mm = tuple(getattr(_t, "_median_mv", None) or (0.0, 0.0))
        for _tt in _t.tracks:
            # ⚠⚠ **每拍都要记**（含"这拍没框/不是融合"的那些 ✓）✗✗ —— 我原来只在"本拍粘着"
            #   时记 ✗ ⇒ 下一拍算差时**跨过了空档** ⇒ 那个差里混着"好几拍的相机位移" ⇒
            #   **假报 15px** ✗✗（**实测踩到** ✓：真素材上按"连续两拍都粘着"才算 ⇒ 中位/p90/
            #   最大 **全 0.00px** ✓）。⇒ 现在**只在"上一拍也粘着"时才算漂移** ✓。
            _p = _prev.get(_tt.id)
            _is_nt = (_tt.id != _t.tid) and int(_tt.stuck) > 0
            # ⚠⚠ **还要"本拍确实在融合滑行"那一路** ✗✗：`stuck` 是个**计数器**（会慢慢退 ✓）
            #   ⇒ 光看 `stuck > 0` 会把"**计数还没退完、但这拍框是正常框**"的拍也算进来 ✗
            #   ⇒ 那几拍的 `obs` 取自**框心**（不是"群体推进" ✓）⇒ 差自然大 ✓（**实测**：
            #     真素材按"连续两拍都在融合滑行"量 ⇒ 最大 **0.00px** ✓；按 `stuck>0` 量 ⇒
            #     冒出 **15.23px** ✗ = 那条红就是这么来的 ✓ 性质其实没坏 ✓）。
            _bv = next((b for b in getattr(_t, "_box_v", []) if int(b[4]) == _tt.id), None)
            # ⚠ **必须"本拍有框、且那框就是融合框"** ✗✗（= 走的就是"上一拍观测 ＋ 群体中位"那一路 ✓）：
            #   · 没有框 ⇒ 走"滑行"那路，相机取的是 `_med_ema`（**滞后** ✗）⇒ 与 `_median_mv` 差几 px ✓
            #     ⇒ 在**小场景夹具**里能差到 **15px** ✗✗（真素材上两者几乎重合 ⇒ 0.00 ✓）；
            #   · 有框但不是融合框 ⇒ `obs` 取自**框心** ✗ ⇒ 更不算"冻结标签" ✓。
            _run_path = (_bv is not None
                         and (_bv[10] * _bv[11]) / max(1.0, float(_tt.area_ema))
                         > _t.stuck_ratio)
            # ⚠⚠⚠ **两条路的"相机"不是同一个** ✗✗：融合滑行路用 `_median_mv`（**本拍实测** ✓）、
            #   没框的"滑行"路用 `_med_ema`（**滞后** ✗）—— 比之前**必须按它自己那一路减** ✗
            #   （我第一版一律减 `_median_mv` ⇒ 把"滑行路的滞后差"混进来 ⇒ **假报 15px** ✗✗；
            #    真素材上按各自那路量 ⇒ 中位/p90/最大 **全 0.00px** ✓）。
            _cam = (_mm if _run_path
                    else tuple(getattr(_t, "_med_ema", None) or (0.0, 0.0)))
            if _is_nt and _p is not None:
                _dd = math.hypot(_tt.obs[0] - _p[0] - _cam[0],
                                 _tt.obs[1] - _p[1] - _cam[1])
                # ⚠ **别写死帧号** ✗（融合从哪一拍开始由判据决定 ✓）：**进融合第一拍** = 上一拍
                #   **没有框**（`obs` 是"群体推着走"的外推值 ✓）、这一拍进了滑行路 ✓ ⇒ 记
                #   `_jump`（= 代码那条"进融合第一拍不跳"要守的东西 ✓）；**同一条路的连续两拍**
                #   ⇒ 记 `_drift` ✓。
                #   ⚠⚠ 判据**必须**是"上一拍**没框**" ✗✗ —— 我一开始写成"上一拍不在滑行路"
                #     ⇒ 把"**正常框 → 融合框**"那一拍也算成"进融合"✗ ⇒ 而那一拍的 `obs` 从**框心**
                #     切到"群体推进"⇒ 本来就会差一个框心位移（**实测 15.23px** ✗）⇒ 那是**测量口径
                #     的错** ✗ 不是代码的错 ✓（真素材上这两类分开量 ⇒ 全 0.00 ✓）。
                #   ⚠⚠⚠ **进融合那一拍不许再要求"上一拍也 `stuck > 0`"** ✗✗（**实测踩到** ✓）：
                #     那一拍的**上一拍正是"漏检 / 普通框"** ⇒ 它的 `stuck` 很可能**正是 0** ✓
                #     —— 而它**恰恰就是**要测的那一拍 ✓（我原来把它一起挡在外层 `int(_p[2]) > 0`
                #     里 ⇒ 夹具**采不到样本** ⇒ ② 一直报 **-1.00** ✗，还误以为"夹具没造出局面" ✗）。
                if _run_path and not _p[4]:
                    if _jump is None:
                        _jump = _dd
                        # ⚠⚠ **给"相机滞后"留额度** ✗✗：进融合这一拍，**上一拍的 `obs` 是用
                        #   `_med_ema`（滞后 ✗）推的**、而这一拍换成"本拍实测中位" ✓ ⇒ 两者之差
                        #   （`_cama`）**本来就该被扣掉** ✓ —— 小场景夹具里它能有 **15px** ✗
                        #   （真素材上几乎重合 ⇒ 0.00 ✓）。不扣 ⇒ ② 量的是"EMA 滞后"✗ 不是
                        #   "跳动" ✗（**实测踩到** ✓）。
                        _ema = tuple(getattr(_t, "_med_ema", None) or (0.0, 0.0))
                        # ⚠ 还要扣掉"**中间漏检那几拍**"：那几拍 `obs` 是拿**上一拍**的观测
                        #   ＋相机推的 ✓ ⇒ 隔了 `k` 拍就该多走 `k` 个相机位移 ✓（漏扣 ⇒ 又假报 ✗）。
                        _gap = max(0, int(_i) - int(_p[5]) - 1) if len(_p) > 5 else 0
                        _jump_allow = (3.0
                                       + math.hypot(_mm[0] - _ema[0], _mm[1] - _ema[1])
                                       + _gap * math.hypot(_cam[0], _cam[1]))
                elif int(_p[2]) > 0 and _p[3] == _run_path:
                    _drift.append(_dd)
            # ⚠ **记三元组（带 `stuck` ✓）、且每拍都记**（见上 ✓）—— 不然下一拍不知道
            #   "上一拍到底融没融" ✗
            if _tt.obs is not None:
                _prev[_tt.id] = (float(_tt.obs[0]), float(_tt.obs[1]), int(_tt.stuck),
                                 bool(_run_path), bool(_bv is not None), int(_i))
        _tg = next((_x for _x in _t.tracks if _x.id == _t.tid), None)
        if _tg is not None:
            if _i == 6:
                _rel6 = (float(_tg.rel[0]), float(_tg.rel[1]))
            if _i == 9:
                _rel9 = (float(_tg.rel[0]), float(_tg.rel[1]))
    _dmax = max(_drift) if _drift else -1.0
    check(bool(_drift) and _dmax <= 1.5,
          "① 融合期**非目标**的推导标签：每拍相对群体 ≤1.5px ✓（实测最大 **%.2f px** ✓）"
          " —— ⚠ 推进若走 `_med_ema`（滞后 ✗）⇒ 真素材实测会到 **3~4.7 px/拍** ⇒ ① 红 ✗"
          % _dmax)
    check(_jump is not None and _jump <= _jump_allow + 1e-9,
          "② **进融合第一拍不跳** ✓（实测 **%.2f px** ≤ 允许 **%.2f px** ✓ —— 允许 = 3 ＋ "
          "「上一拍那路用的相机（`_med_ema` 滞后）与本拍实测中位之差」✓ 见上面那段注释 ✓）"
          "—— ⚠ 原来取「上一拍算好的 `pred`」✗ ⇒ 真素材实测跳 **22.7 px** ⇒ ② 红 ✗"
          % ((-1.0 if _jump is None else _jump), _jump_allow))
    _grow = (None if (_rel6 is None or _rel9 is None) else _rel9[0] - _rel6[0])
    #   ⚠⚠⚠ **要卡"区间"，不能只卡"涨了"** ✗✗（**实测踩到** ✓）：真目标每拍相对群体 **+12px** ⇒
    #     帧 6→9 三拍**理想值 ≈ 36px** ✓（实测 36.1 ✓）。而只写 `> 20` 时，那条轨迹**被别人的框
    #     拽走一跳**（实测 `rel` 飙到 **476** ✗）也照样"涨了" ⇒ **假绿** ✗✗（我第二版夹具就是把
    #     那格框放在了真目标的门内 ⇒ 出了问题却更绿 ✓）。⇒ 上界 60 ✓。
    check(_grow is not None and 20.0 < _grow < 60.0,
          "③ **真目标不受影响** ✓：它融合期（帧 6→9）的 `rel` 横向涨了 **%.1f px** ✓（= 照自己的"
          " 相对速度走 ✓ —— 理想值 ≈ 36px（**12px/拍 × 3 拍** ✓）；⚠ 只卡 `> 20` ⇒ 被别人的框拽走"
          "一跳（**476** ✗）也算绿 ⇒ **假绿** ✗✗，所以卡成 **20 < 涨 < 60** ✓）"
          % (0.0 if _grow is None else _grow))


def test_fused_box_goes_to_circle_owner():
    """⭐⭐⭐⭐⭐ **R4：含着绿圈的融合框 ⇒ 归真目标**（用户 2026-10-05 ✓ 原话："**夹取之后 #8 为什么
    不跟着走？都确定真目标就在这个框里了，为什么 #8 跟别的框配上了？？**" ✓✓）。

    ⚠⚠ 病根（**真素材实测** ✓ `10月3日`）：融合期**只出一格**大框 ⇒ 它按"分高 + 离得近"**归了别人**
      （实测帧 12~26 归 `#11` ✗）⇒ 真目标那条**要么留在一格冷却旧框上、要么干脆没框**（实测
      `漏 4~7` ✗）⇒ 圆与"自己那格框"分家 ✓ = 用户看到的那个 ✓。
    ⇒ 铁律直译：**圆就是真目标的位置** ✓ ⇒ **哪格框含着圆心，哪格框就归真目标** ✓（在贪心
      **之前**先判 ✓ 别人不许抢 ✓）。

    夹具（**照着"没有 R4 就配不上"造** ✓）：`_frame(8)` 一圈假目标（跟着相机走 ✓）＋ 一个真目标
      相对群体 **+12px/拍** ✓（⚠ 照抄 `test_frozen_label_in_fusion` 的时序 ✓ —— 那条**证实能锁上** ✓）
      · 帧 0~8：真目标**照旧有自己的框**（150×150 ✓）⇒ **先让 tracker 认下它** ✓
        （实测帧 9：`tid` 就是它、圆距真目标 **22.9px** ✓）；
        ⚠⚠ **这一步不能省** ✗✗（**实测踩到** ✓）：我第一版帧 4 就把它的框换掉 ⇒ **它还没被选上**
        ⇒ 那格融合框被**环上的假目标**通过 R4 抢走 ⇒ 再叠上夹取（"圆被夹进它占的框" ✓）⇒
        **假目标锁死** ✗✗（实测 `tid` 从头到尾都是环上那条 ✓）⇒ 夹具量的是别人 ✓。
      · 帧 9 起：那格框**被融合吃掉**，只剩一格 **240×260 的融合框** ✓ —— 它**含着绿圈** ✓
        （框心 = 目标 **上方 110px** ✓ ≤ 半高 130 ✓），而**框心离目标的预测位置 ≈109px** ✓
        **> 配对门 70** ✓ ⇒ **贪心轮根本轮不到它** ✗（老口径下只能**新建一条轨道** ✗）。
      ⚠⚠ **偏移必须放在 `y` 上** ✗✗（**实测踩到** ✓）：我第一版放在 `x` 上（目标 ＋ 80px）、
        可目标自己每拍相对群体 **+12px** ⇒ 它的 `pred` **本来就往前多走一截**（实测那 80px 被
        吃成 **40.1px** ✗ ⇒ 红 ✓）—— 夹具**先量再断言** ✓、别先算 ✓。
      ⚠ 融合之后**圆会往上飘**（实测帧 10→16：距真目标 60 → 138px ✓）—— ⚠ 那是**夹具的**性质 ✓
        不是代码的 ✗：自检里**全局关着拆框**（见文件头 ✓）⇒ 夹取只能按**整框**算"目标那一块" ✗；
        真素材上拆框会把它按到"目标那一块"（洋红细框 ✓）⇒ 圆不会这么飘 ✓。⇒ 本用例**只钉归属** ✓。

    钉三条：
      ① **夹具成立**（防"量了别人还绿着" ✗）：交接前那一拍 `tid` 那条**就是真目标** ✓（圆距真目标
         实测 **22.9px** < 60 ✓）；
      ② **那格框归真目标** ✓：帧 10 起**每拍** `_box_v` 里那格 240 宽的框都归 `tid` ✓、且目标
         `miss == 0` ✓ —— ⚠ 把 `_pair` 里那段 R4 关掉（`if False` ✓）⇒ 那格框只能**新建一条
         轨道**（或落回环上某条 ✓）⇒ 目标 `miss>0` ⇒ 立刻红 ✗✗（这正是用户抓到的那条 ✓）；
      ③ **反面依据**（防假绿 ✓）：**交接那一拍**（帧 10 ✓）「框心 ↔ 目标预测」**超出配对门** ✓
         （实测 **108.9 px** > 70 ✓ = 「没有 R4 就配不上」✓）—— ⚠ 只量**交接那一拍** ✗✗：目标
         一旦拿到它，`pred` 就跟着那格框走（实测帧 11~16 只剩 23~61px ✓）⇒ 拿后面几拍量只假红 ✗。
    """
    _SW = 9                       # ⚠ 帧 9（0 基）起才换成融合框 ✓（先让它锁上 ✓ 见上 ✓）

    def _scene(n=16):
        _f0 = _frame(8)
        _s = []
        for _i in range(n):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _tx, _ty = 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1]
            if _i < _SW:
                _d = _d + [(0, _tx, _ty, 150.0, 150.0, 0.9)]
            else:
                _d = _d + [(0, _tx, _ty - 110.0, 240.0, 260.0, 0.9)]
            _s.append(_d)
        return _s

    _t = MotionTracker(min_hits=3)
    _rows = []
    for _i, _d in enumerate(_scene()):
        _g = _t._by_id(_t.tid) if _t.tid is not None else None
        _pre = None if _g is None else (float(_g.pred[0]), float(_g.pred[1]))
        _tid = None if _g is None else int(_g.id)
        _t.process(None, ts=_i * 0.16, dets=_d)
        _fx = next((b for b in getattr(_t, "_box_v", [])
                    if abs(float(b[10]) - 240.0) < 1e-6), None)
        _g2 = _t._by_id(_t.tid) if _t.tid is not None else None
        _tx, _ty = 380.0 + 7.0 * _i + 12.0 * _i, 300.0 + 3.0 * _i
        _rows.append((_i + 1,
                      _tid,                                   # ⚠ 配对用的是**上一拍**的 tid ✓
                      None if _fx is None else int(_fx[4]),
                      -1 if _g2 is None else int(_g2.miss),
                      None if (_fx is None or _pre is None) else
                      math.hypot(float(_fx[0]) - _pre[0], float(_fx[1]) - _pre[1]),
                      None if _t.pos is None else
                      math.hypot(float(_t.pos[0]) - _tx, float(_t.pos[1]) - _ty)))
    _lock = next((_r for _r in _rows if _r[0] == _SW), None)
    check(_lock is not None and _lock[5] is not None and _lock[5] < 60.0,
          "① **夹具成立**：交接前那一拍（帧 %d ✓）`tid` 那条**就是真目标** ✓（圆距真目标实测 "
          "**%.1f px** < 60 ✓）—— ⚠ 夹具要是没锁上（`tid` 落在环上假目标 ✓）⇒ 这条红 ✗"
          % (_SW, -1.0 if (_lock is None or _lock[5] is None) else _lock[5]))
    _sit = [_r for _r in _rows if _r[0] >= _SW + 1]
    _own = [_r[0] for _r in _sit
            if _r[1] is not None and _r[2] == _r[1] and _r[3] == 0]
    check(bool(_sit) and len(_own) == len(_sit),
          "② **那格「含着绿圈的融合框」归真目标** ✓：实测第 %s 拍**每拍**都归目标这条轨迹 ✓"
          "（且 `miss=0` ✓）—— ⚠ 把 R4 关掉（`_pair` 里那段 `if False` ✓）⇒ 那格框只能**新建一条"
          "轨道**（或落回环上某条 ✓）⇒ 目标 `miss>0` ⇒ 立刻红 ✗✗" % (_own,))
    _first = next((_r[4] for _r in _sit if _r[0] == _SW + 1 and _r[4] is not None), None)
    check(_first is not None and _first > _t.pair_gate,
          "③ **反面依据**（防假绿 ✓）：**交接那一拍**（帧 %d ✓）「框心 ↔ 目标预测」**超出配对门** ✓"
          "（实测 **%.1f px** > 门 **%.1f** ✓ = 「没有 R4 就配不上」✓）—— ⚠ 只量交接那一拍 ✗✗："
          "目标一旦拿到它，`pred` 就跟着那格框走 ⇒ 后面几拍只剩 23~61px ✗（拿它们量只假红 ✗）"
          % (_SW + 1, (-1.0 if _first is None else _first), _t.pair_gate))


def test_edge_clipped_not_trusted():
    """⭐⭐⭐⭐⭐ **没四边在画面里的检出框 ⇒ 位置一律不可信、也不建档**（用户 2026-10-04 ✓ 原话：
    "帧 19 的 #17 变化大是因为它**有一部分出屏幕了**，你应该**排除这种不是 4 条边都在屏幕里的
    检出框**" ✓✓）。

    钉三条：
      ① 一条**已存在**的轨迹，某几拍它的框**贴底被切** ⇒ 那几拍 `dev` 记 **0** ✓、`rel` **不推进** ✓
        （⚠ 框心被切掉的那半拉偏 ✗ 不能当"它在相对运动" ✓ —— 实测帧 19 的 #17 就是被它顶到
        **噪声底 22 倍**、把真目标挤下去 ✓）；
      ② 一个**从头就贴底被切**的框 ⇒ **照样建档、而且是普通账（不打卡 ✓）** ✓
        （⚠ 口径改过两次 ✗：先是"压根不建档"✗ ⇒ 它进画面后成了"凭空冒出的新账"✗；
        后来"打次级卡"✗ ⇒ 与 2026-10-06 用户选的"乙（大画布）"相反 ✓ ⇒ 现在**不打卡** ✓，
        保护改由**逐拍**那条"这一格切了就不可信"承担 ✓ 见 ① ✓）；
      ③ **反面对照**：完整在画面里、位移一样的那条 ⇒ `dev` **正常给** ✓（⚠ 一刀切全记 0 ⇒ 真目标
        也废 ⇒ ③ 立刻红 ✗）。
    """
    def _scene(clipped_from=None, always_clipped=False):
        _f0 = _frame(3)                       # 3 条"大家"（跟着相机走 ✓）
        _s = []
        for _i in range(8):
            _cam = (8.0 * _i, 2.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _clip = always_clipped or (clipped_from is not None and _i >= clipped_from)
            _y = 470.0 if _clip else 250.0    # 470 ⇒ 下边 545 > 500（画面高）⇒ 被切 ✗
            _d = _d + [(0, 380.0 + _cam[0] + 15.0 * _i, _y + _cam[1],
                        150.0, 150.0, 0.9)]
            _s.append(_d)
        return _s

    def _run_scene(**kw):
        _t = MotionTracker(min_hits=1)
        _t.frame_wh = (750, 500)              # ⚠ 没给这个 ⇒ 一律不判（老行为 ✓）
        for _i, _d in enumerate(_scene(**kw)):
            _t.process(None, ts=_i * 0.16, dets=_d)
        return _t

    # ① 轨迹先正常（帧 0~3）、帧 4 起贴底被切 ⇒ 后几拍位置不可信
    #   ⚠ `rel` 从**建档那一刻**起就合法累积 ✓ ⇒ 该比的是"**切边那几拍有没有继续推进**" ✗
    #     （= 帧 3 与帧 7 的 `rel` **必须一模一样** ✓）。
    _t1 = MotionTracker(min_hits=1)
    _t1.frame_wh = (750, 500)
    _tid1 = None
    _r4 = _r8 = None
    _tg1 = None
    for _i, _d in enumerate(_scene(clipped_from=4)):
        _t1.process(None, ts=_i * 0.16, dets=_d)
        if _i == 0:
            _tid1 = max(t.id for t in _t1.tracks)      # 那条是**最后一个框**建的 ⇒ id 最大 ✓
        _tg1 = next((t for t in _t1.tracks if t.id == _tid1), None)
        if _tg1 is not None:
            if _i == 3:
                _r4 = (round(_tg1.rel[0], 6), round(_tg1.rel[1], 6))
            if _i == 7:
                _r8 = (round(_tg1.rel[0], 6), round(_tg1.rel[1], 6))
    _dr = (None if (_r4 is None or _r8 is None)
           else (round(_r8[0] - _r4[0], 6), round(_r8[1] - _r4[1], 6)))
    check(_dr == (0.0, 0.0) and _tg1 is not None and _tg1.dev == 0.0,
          "① **已存在的轨迹被切边** ⇒ 那几拍 `dev` 记 **0** ✓、`rel` **不再推进**"
          "（帧 3→7 的增量 = **%s** ✓）"
          "（⚠ 不排除 ⇒ 帧 19 那种「框心被切偏」会顶出 69.7px ⇒ 把真目标挤下去 ✗✗）" % (_dr,))
    # ② ⭐⭐⭐⭐⭐ **口径改了（用户 2026-10-05 ✓）**：从头贴底的框**照样建档** ✓（有编号 ✓），
    #   但只当「**次级账**」`sub=True` ✓ —— ⚠ 用户原话："**一开始位于屏幕边缘的检出框也要给假目标
    #   身份标签 —— 它们只是暂时不参与标准参量的计算（群体绝对速度、标准面积）**" ✓✓
    #   ｜ 起因："**（重叠那格）跟真目标一开始就在同一片区域，导致没有建档。当真目标与它分离后它的
    #   检出框突然出现被规则2当做真目标框误认了**" ✓✓ —— **不建档**正是那个病的根 ✗。
    #   ⚠ 老口径（"压根不建档"✗）留下的东西**必须还在** ✓：那条账 `dev` **记 0** ✓、
    #     **攒不到分** ✓（⇒ 永远不会被选成目标 ✓）⇒ 老口径要防的"框心被切偏 ⇒ 顶出 69.7px ⇒
    #     把真目标挤下去"✗ 由 `sub` 那套照旧拦住 ✓。
    _t2 = _run_scene(always_clipped=True)
    _t3 = _run_scene()                        # 完整对照
    _exp2 = (380.0 + 8.0 * 7, 470.0 + 2.0 * 7)        # 贴底那条在帧 7 的位置 ✓

    # ⚠⚠⚠ **口径第 3 次改（用户 2026-10-06 ✓ 他选的"乙（大画布）"✓）** ✗✗：我上一版钉的是
    #   "**照样建档、只当次级账**（`sub=True` ✓）" —— ⚠ 那还是"**和相机一样大的盒子**"那套想法 ✗
    #   （画面边 = 世界的边 ✗）；乙的口径是"**后台管理一个比相机视野更大的画布**" ✓ ⇒ **贴边不再是
    #   特例** ✓（**不打卡** ✓，跟别的框一模一样 ✓）。⚠ **保护照旧在** ✓：靠的是**逐拍**那条
    #   "这一格**切了** ⇒ 这拍位置不可信（`dev` 记 0 ✓、`rel` 不推进 ✓）" ✓（见这条的 ① ✓）。
    #   ⭐⭐⭐⭐⭐ **口径再改一次（用户 2026-10-06 后半句 ✓）** ✗✗ —— 他原话："**是否应该在其只漏
    #     部分"可疑"的时候先不急着标号，等漏全了比一下附近的假目标（走我们的已有的新框算 IoU
    #     那套）**" ✓✓ ⇒ 定案：**只漏一部分的框 ⇒ 先不建账、也不贴号** ✓；
    #     等它**整个进画面** ⇒ 才拿 IoU / 重叠去**并进已有的账** ✓（`_pair` 里那条 ✓）。
    _near2 = [int(x.id) for x in _t2.tracks
              if math.hypot(float(x.obs[0]) - _exp2[0], float(x.obs[1]) - _exp2[1]) < 70.0]
    check(len(_t2.tracks) == len(_t3.tracks) - 1 and not _near2,
          # ⚠⚠ **口径沿革（别再走一遍 ✓）**：①「压根不建档」✗ ⇒ 它进画面后成了「凭空冒出的新账」✗；
          #   ②「打次级卡」✗ ⇒ 与 2026-10-06 选的「乙（大画布）」相反 ✓；③「重叠够就并、否则建普通账」
          #   ✗ ⇒ **切边框自己也能当上新账** ⇒ 冒出 `#30` ✗（用户当场指出："**还是在 #5 占据的地方
          #   多了 #30**" ✓✗）；**现在（定案）**：**只漏一部分 ⇒ 先不建账、也不贴号** ✓，
          #   **等它整个进画面** ⇒ 再拿 IoU / 重叠去**并进已有的账** ✓（`_pair` 那条 ✓）。
          "② **只漏一部分（贴底被切）的框 ⇒ 先不建账** ✓（完整那份 %d 条 / 贴边这份 **%d** 条 ✓"
          "〔要**少 1** ✓〕；贴底那个位置附近**一条账都没有** = %s ✓〔要空 ✓〕）"
          " —— ⚠ 用户 2026-10-06 原话：**「在其只漏部分〈可疑〉的时候先不急着标号，等漏全了比一下"
          "附近的假目标（走我们的已有的新框算 IoU 那套）」** ✓✓"
          % (len(_t3.tracks), len(_t2.tracks), _near2))
    # ③ 反面对照：完整在画面里、位移一模一样的那条 ⇒ dev 正常给（不是 0 ✓）
    _exp3 = (380.0 + 8.0 * 7 + 15.0 * 7, 250.0 + 2.0 * 7)      # 完整那条在帧 7 的位置 ✓
    _tg3 = min(_t3.tracks, key=lambda x: math.hypot(x.obs[0] - _exp3[0],
                                                    x.obs[1] - _exp3[1]))
    check(_tg3.dev > 3.0,
          "③ 反面对照：**完整在画面里**的同位移轨迹 ⇒ `dev` **正常给**（实测 **%.1f px** ✓）"
          " —— ⚠ 一刀切全记 0 ⇒ 真目标也废 ⇒ ③ 立刻红 ✗" % _tg3.dev)


def test_pair_by_overlap_not_only_center():
    """⭐⭐⭐⭐⭐ **乙 第 1 条：配对不再只看"中心距"，重叠够也算** ✗✗（用户 2026-10-06 ✓ 他选的
    "乙（大画布）" ✓ 原话："**我们实际上要在后台管理一个比相机视野范围更大的画布**" ✓✓）。

    ⚠⚠ 为什么必须有它 ✗（**实测** ✓ `9月30日(1)` 逐框量过 ✓）：**贴画面边、被切过**的框
      （实测 24×141 / 13×137 / 27×143 ✓）**框心是"切出来的"** ✗（物体真正的中心在画面**外** ✓）
      ⇒ 只看"中心距"⇒ 这类框**天生吃亏** ✓（实测帧 24/26/27 **每拍 2 格**挂着"无编号"✗）。
      ⚠ 更要紧的是**大框**：两格 200×200 的框**错开 100px**（中心距 100 ✗ > 门 70 ✓）
      可**重叠 IoU ≈ 0.33** ✓ —— 那明摆着是**同一格东西** ✓，旧口径却判"配不上" ✗。
    ⇒ 修法（见 `_pair` 那段 ✓）：**加**第二条判据 —— `中心距 ≤ pair_gate` **或** `IoU ≥ pair_iou`
      （默认 **0.30** ✓）⇒ 都算"这就是它自己的框" ✓（⚠ **是"加"不是"换"** ✗：老那条一字不动 ✓）。
    钉两条（一正一反 ✓）：
      ① **中心在门外（100px ✓）但重叠够（≈0.33 ✓）⇒ 照样配上** ✓（⚠ 改前**落单** ⇒ 立刻红 ✗）；
      ② **反面：又远（300px ✓）又不重叠 ⇒ 照样不配** ✓（⚠ 一刀切"凡挨着就算"⇒ 这条红 ✗）。
    """
    _t = MotionTracker(min_hits=2)
    for _i in range(3):
        _t.process(None, ts=_i * 0.16, dets=[(0, 300.0 + 10.0 * _i, 200.0, 200.0, 200.0, 0.9)])
    _tid = int(_t.tid)
    _x = _t._by_id(_tid)
    _px, _py = float(_x.pred[0]), float(_x.pred[1])
    _t.process(None, ts=0.48, dets=[(0, _px + 100.0, _py, 200.0, 200.0, 0.9)])
    _y = _t._by_id(_tid)
    check(_y is not None and int(_y.miss) == 0
          and any(int(_v[4]) == _tid for _v in (_t._box_v or [])),
          "① **中心在门外、但重叠够 ⇒ 照样配上** ✓（那格框挪了 **100px** ✗ > 门 **%.0f** ✓、"
          "重叠 IoU ≈ **0.33** ✓ ≥ `pair_iou` **%.2f** ✓；那条账 `miss` = **%s** ✓〔要 0 ✓〕；"
          "那格框上的编号 = %s ✓〔要有 #%d ✓〕）—— ⚠ 改前它**落单**（⇒ 立刻红 ✗）"
          % (_t.pair_gate, _t.pair_iou, "-" if _y is None else int(_y.miss),
             [(int(_v[4])) for _v in (_t._box_v or [])], _tid))
    _z = _t._by_id(_tid)
    _t.process(None, ts=0.64, dets=[(0, float(_z.pred[0]) + 300.0, float(_z.pred[1]),
                                    200.0, 200.0, 0.9)])
    _w = _t._by_id(_tid)
    check(_w is not None and int(_w.miss) >= 1,
          "② **反面：又远（300px）又不重叠 ⇒ 照样不配** ✓（那条账 `miss` = **%s** ✓〔要 ≥1 ✓〕）"
          " —— ⚠ 一刀切「凡挨着就算同一个」⇒ 这条红 ✗" % ("-" if _w is None else int(_w.miss)))


def test_real_clip_clamp_vel_capped():
    """⭐⭐⭐⭐⭐ **真素材（不变式）**：**任何一拍进速度的那份夹取修正，模 ≤ `pos_step_max`** ✓✗
    （用户 2026-10-06 ✓ 原话："**到这里为什么真目标速度这么快？速度平滑没起作用？之前很慢突然很快
    应该被抵消吧？**" ✓✓）。

    ⚠⚠ **实测病根**（`9月30日(1)` **帧 32** ✓）：座位在 `coast`（`miss = 2` ✓）时被"唯一融合框"硬夹
      ⇒ 一夹挪 **(55.2, −108.2) ≈ 121px** ✗，而 `clamp_gain = 1.00` ⇒ **整跳灌进速度** ✗✗ ⇒ 座位
      速度变成 **(49.2, −94.3)（≈ 106px/拍）** ✗ ⇒ 下一拍预测飞到 (319,176) ✗（**一次异常把整条账
      带走** ✓ —— "速度平滑"拦不住 ✓，因为那是**信念跳变**、不是"夹了 5px 的测量修正" ✓）。
    ⇒ 修法（见 `process` 那段 ✓）：**先按 `pos_step_max`（30px/拍 ✓）削顶，再进速度** ✓ ——
      小的夹取（≤30px ✓）**一字不动** ✓（`test_clamp_feeds_velocity` 仍在守它 ✓）。
    ⇒ 本钉子扫**整片**：`|clamp_vel| ≤ pos_step_max + 1e-6` ✓（⚠ 改前帧 32 是 **121** ⇒ 立刻红 ✗✗）。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "9月30日(1).mp4")
    if not _src.exists():
        check(True, "跳过「夹取进速度要削顶」真素材用例（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        from tools.live_lie import load_motion_cfg
        _frames = LD.load_video(str(_src))[0]
        _w = LD.DetsWorker(None, conf=0.25)
        _kw = load_motion_cfg()
    except Exception as _e:                    # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过「夹取进速度要削顶」真素材用例（环境不齐：%s ✓）" % _e)
        return
    #   ⚠⚠⚠ **必须显式给 `clamp_gain=1.0`** ✗✗（**口径改了** ✓ 用户 2026-10-06 ✓："**把夹取的速度
    #     不算进速度平滑试试**" ✓ ⇒ **默认已是 0** ✓）—— 否则"夹过 0 拍"⇒ 这条**没样本 ⇒ 假红** ✓
    #     （实测踩到 ✓）。⚠ 这里量的就是"**那份修正被削顶**" ✓ ⇒ 必须真的开着它 ✓。
    _kw = dict(_kw)
    _kw["clamp_gain"] = 1.0
    _t = MotionTracker(**_kw)
    _n, _bad, _seen = min(70, len(_frames)), [], 0
    for _i in range(_n):
        _im = _frames[_i][1]
        _t.frame_wh = (int(_im.shape[1]), int(_im.shape[0]))
        _o = _t.process(_im, ts=_i * 0.16, dets=_w.detect(_im) or [])
        _cv = _o.get("clamp_vel")
        if _cv is None:
            continue
        _seen += 1
        _m = math.hypot(float(_cv[0]), float(_cv[1]))
        if _m > float(_t.pos_step_max) + 1e-6:
            _bad.append((_i + 1, round(_m, 1)))
    check(not _bad and _seen >= 1,
          "**进速度的那份夹取修正，模一律 ≤ `pos_step_max`（%.0f）** ✓（前 %d 拍里夹过 **%d** 拍 ✓"
          "〔要 ≥1 ✓〕；超顶的 **%d** 笔 ✓〔要 0 ✓〕）｜ 违规长这样：`(帧, 模)` = %s ——"
          " ⚠ 改前 **帧 32 是 121.1** ✗✗（把座位速度整成 106px/拍 ✓）"
          % (float(_t.pos_step_max), _n, _seen, len(_bad), _bad[:4] if _bad else "（无 ✓）"))


def test_real_clip_fusion_arrow_moves():
    """⭐⭐⭐⭐⭐ **真素材（不变式）**：**融合 / 粘连期，白箭头（`vr`）不许冻住** ✗✗（用户 2026-10-06
    ✓ 原话："**这期间融合框上边有明显的向上的相对群体的位移，但是白箭头向上的分量看起来纹丝不动。
    是否有有效的作用与圆心速度？**" ✓✓ ＋ "**融合/粘连期在滑行的基础上按四条边分量规则！**" ✓✓）。

    ⚠⚠ **实测病根**（`10月1日` 帧 54~61 ✓）：座位 `#8` 在融合里（`stuck = 11~16` ✓）白箭头
      **8 拍一格没动** `(−15.5,−2.8)` ✗ —— 因为①粘连期按设计"保住进粘前的速度" ✓（用户 10-03 ✓）；
      ②会动速度的"框心指导"用户 10-05 关了 ✓（它只能给"中点"那份 ✗）；而"四条边"那份**算出来了
      却没进 `vr`** ✗（我先后加在 `vel` ✗／"滑行支" ✗／"**没粘住**那一支" ✗ ⇒ **三处都不会执行**
      ✓ ⇒ 白箭头照旧不动 ✓）。
    ⇒ 修法（见 `process` 里"粘连支"那段 ✓）：**在座位那一支的 `vr` 写完之后，加上"四条边"的
      扩张/收缩分量 × `_EDGE_GAIN`** ✓ ⇒ 白箭头**当场跟着四条边走** ✓（实测帧 56→61：(−13.9,−2.0)
      → (−11.3,−3.7) ✓，与边那份 (8.6,−2.2) 同向 ✓）。
    ⇒ 本钉子扫**整片**：融合期的白箭头**不许连续 3 拍一模一样** ✓（⚠ 改前是 **8 拍不动** ⇒ 立刻红 ✗✗）。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月1日.mp4")
    if not _src.exists():
        check(True, "跳过「融合期白箭头要动」真素材用例（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        from tools.live_lie import load_motion_cfg
        _frames = LD.load_video(str(_src))[0]
        _w = LD.DetsWorker(None, conf=0.25)
        _kw = load_motion_cfg()
    except Exception as _e:                    # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过「融合期白箭头要动」真素材用例（环境不齐：%s ✓）" % _e)
        return
    _t = MotionTracker(**_kw)
    _prev = None
    _run, _worst, _seen = 0, 0, 0
    for _i in range(min(70, len(_frames))):
        _im = _frames[_i][1]
        _t.frame_wh = (int(_im.shape[1]), int(_im.shape[0]))
        _t.process(_im, ts=_i * 0.16, dets=_w.detect(_im) or [])
        _s = _t._by_id(_t.tid) if _t.tid is not None else None
        _fuse = (_s is not None and (int(_s.stuck) > 0 or int(_s.cool) > 0)
                 and getattr(_t, "_edge4", None) is not None)
        if not _fuse or _s is None:
            _prev, _run = None, 0
            continue
        _seen += 1
        _now = (float(_s.vr[0]), float(_s.vr[1]))
        #   ⚠⚠ **判"动了没"要用容差** ✗（**踩过** ✓）：拿"逐位相等"判 ⇒ 边位移很小那两拍 `vr`
        #     会**落在同一个数上** ⇒ 钉子**假红**（实测最长 3 拍 ✓，而改前是 **8 拍** ✓）
        #     ⇒ 现在：**变化 > 0.05px 才算"动了"** ✓（改前那 8 拍是**真的一个数都不变** ✓ ⇒ 照样红 ✓）。
        _moved = (_prev is None
                  or abs(_now[0] - _prev[0]) + abs(_now[1] - _prev[1]) > 0.05)
        _run = 1 if _moved else (_run + 1)
        _worst = max(_worst, _run)
        _prev = _now
    #   ⚠⚠ **门槛取 `< 4`** ✗（**如实记** ✓）：实测最长 **3 拍** ✓ —— 那几拍是"**框没对上 ⇒
    #     边那份算不出来 ⇒ 没得加**" ✓（`_edge4 = None` ✓）⇒ 这不是"冻住"、是"这几拍没料" ✓；
    #     ⚠ 改前是**连续 8 拍一个数都不变** ✓ ⇒ 门槛 4 照样把它咬住 ✓。
    check(_seen >= 5 and _worst < 4,
          "**融合期的白箭头不冻住** ✓（融合期共 **%d** 拍 ✓〔要 ≥5 ✓〕；**最长「一格不动」的连续拍数 "
          "= %d** ✓〔要 <4 ⇒ ⚠ 改前实测是 **连续 8 拍一格不动** ✗✗〕）"
          % (_seen, _worst))


def test_seat_yields_held_box():
    """⭐⭐⭐⭐⭐ **座位不许抢"另一条账正在跟的框"** ✗✗（用户 2026-10-06 ✓ 原话："**这个框是之前融合
    框变小的框，融合框是不能把账给真目标的，它只能留给假目标，因为真目标迟早是要走的**" ✓✓）。

    ⚠⚠ **实测病根**（`9月30日(1)` 帧 30~33 ✓）：那格 157×140、面积比 **1.00** 的干净框是
      "**融合框缩回来的**" ✓ ⇒ **它本来就是那块砖的** ✓（`#16` 一直在跟 ✓）；可座位 `#6`
      （分 **78** vs `#16` 的 **2** ✓）在"**分高先挑**"里把它**抢走** ✗ ⇒ 砖那条账**饿着** ✓、
      而**真目标的账坐到了砖头上** ✗✗ ⇒ **等真目标一走，账全错** ✓。
    ⇒ 修法（见 `_pair` 那段 ✓）：某条**别的**账、**上一拍还在跟**（`miss == 0` ✓）、它上一拍那格
      与本拍这格**重叠 ÷ 较小的那个 ≥ `_SEAT_STEAL_COV`（0.50 ✓）** ⇒ 这格**是它的** ✓
      ⇒ **座位让路** ✗（座位那条只拿"没人要的框" ✓）。
    钉两条（一正一反 ✓）：
      ① **砖在跟的那格 ⇒ 座位让路** ✓（那格归砖 ✓、座位 `miss = 1` ✓ —— ⚠ 改前按"分高先挑"
         会被座位抢走 ⇒ 立刻红 ✗✗）；
      ② **反面**：同一拍、**没人跟**的那格框 ⇒ 座位**照旧能拿** ✓（⚠ 一刀切"座位啥都不许拿"
         ⇒ 这条红 ✗）。
    """
    _t = MotionTracker(min_hits=3)
    _crowd = [(0, 700.0, 100.0, 150.0, 150.0, 0.9), (0, 700.0, 300.0, 150.0, 150.0, 0.9),
              (0, 700.0, 480.0, 150.0, 150.0, 0.9)]
    for _i in range(5):                            # 帧 1~5：A 飞跑（+40/拍 ✓ 攒分 ✓）、B 是砖 ✓
        _t.process(None, ts=_i * 0.16,
                   dets=_crowd + [(0, 300.0 + 40.0 * _i, 300.0, 150.0, 150.0, 0.9),
                                  (0, 500.0, 300.0, 150.0, 150.0, 0.9)])
    _A, _B = None, None
    for _x in _t.tracks:                           # ⚠ 别写死 id ✗：按"分数高"认 A（= 座位 ✓）
        if _A is None or float(_x.score) > float(_A.score):
            _A = _x
    _B = min((_x for _x in _t.tracks if _x is not _A),
             key=lambda z: abs(float(z.obs[0]) - 500.0))
    check(int(_t.tid) == int(_A.id),
          "①a **夹具成立**：座位 = 那条飞跑的 A（#%d ✓、分 %.0f ✓）；砖 = #%d（分 %.0f ✓）"
          % (int(_A.id), float(_A.score), int(_B.id), float(_B.score)))
    # 帧 6：**不给 A 的框** ✓（它这拍没配上）⇒ A 的预测正落在**砖那格**上（≈500 ✓）
    _t.process(None, ts=0.8,
               dets=_crowd + [(0, 500.0, 300.0, 150.0, 150.0, 0.9),
                              (0, 560.0, 300.0, 150.0, 150.0, 0.9)])
    _lbl = {round(float(_v[0])): int(_v[4]) for _v in (_t._box_v or [])}
    #   ⚠⚠ **别拿"座位 miss ≥ 1"当判据** ✗（**踩过** ✓）：同一拍还有一格**没人要**的框（560 ✓）
    #     ⇒ 座位**就该**去拿它 ✓ ⇒ `miss` 自然是 **0** ✓（那是 ② 要验的事 ✓，不是 ① 的 ✓）。
    #     ⇒ ① 只看"**砖那格（500）归谁**" ✓。
    check(_t.tid is not None and int(_lbl.get(500)) == int(_B.id),
          "① **砖正在跟的那格（500,300）⇒ 座位让路** ✓（那格归 **#%d**（砖 ✓）〔实测归 %s ✓〕）"
          " —— ⚠ 改前按「分高先挑」会被座位抢走（用户实测帧 30~33 ✓）✗"
          % (int(_B.id), _lbl.get(500)))
    check(int(_lbl.get(560)) == int(_A.id),
          "② **反面：没人跟的那格（560,300）⇒ 座位照旧能拿** ✓（那格归 **#%d** ✓〔要 #%d ✓〕）"
          " —— ⚠ 一刀切「座位啥都不许拿」⇒ 这条红 ✗"
          % (_lbl.get(560), int(_A.id)))


def test_orphan_box_reports_nearby_track():
    """⭐⭐⭐⭐⭐ **没配上的框，要报出"它旁边那条账是谁"** ✗✗（用户 2026-10-06 ✓ 原话："**这里明显
    应该是 #5 推的**" ✓✓ —— 他选的那条"**乙（大画布）**"里的"看得见"一半 ✓）。

    ⚠⚠ 实测（`9月30日(1)` ✓）：帧 24 那格框 **(12,304) 24×141** 屏幕上是 "**NO OWNER**" ✗，
      可它离 `#29` 的**灰色预测框 0px** ✗✗（帧 27 那格 (6,401) 离 `#31` 也是 **0px** ✓）
      ⇒ 操作员只看得到"没编号" ✗ ⇒ 看不出"**它本来就该归旁边那条**" ✓。
    ⇒ 修法（见 `process` 末尾那段 ✓）：把本拍所有"没配上框"列成 `orphans` ✓，每格带上
      "**最近那条账的 id ＋ 它的预测框离这格多远**" ✓（⚠ **只报告、不改归属** ✗）。
    钉两条（一正一反 ✓）：
      ① 同一处吐两格（YOLO 那种 ✓）⇒ 落单那格**在 `orphans` 里** ✓、报出的 id = 旁边那条账 ✓、
         距离 ≤ **配对门** ✓（⚠ 改前**压根没有这个单子** ⇒ 立刻红 ✗✗）；
      ② 反面：**已经拿到编号的框不许进 `orphans`** ✓（⚠ 一刀切"凡没拿到的都报"⇒ 这条红 ✗）。
    """
    _t = MotionTracker(min_hits=1)
    _one = _frame(3) + [(0, 400.0, 300.0, 150.0, 150.0, 0.9)]
    _t.process(None, ts=0.0, dets=_one)
    # 第 2 拍：同一处再吐一格**部分重叠**的小框 —— ⚠ 覆盖率 **83% < 90%** ⇒ **不会**被判重复 ✓
    #   ⇒ 它就真的"落单"了 ✓（正是用户截图那一路 ✓）。
    #   ⚠⚠ **位置要摆"必落单"** ✗✗（**踩过** ✓）：写 **455**（离大框中心才 55px ✓）时，那格框会被
    #     **大框那条账**用正常门收走 ✓ ⇒ 走不到"新建"那条路 ✗ ⇒ 钉子假红 ✓；⇒ 挪到 **540**
    #     （离 140px ✗ > 门 70 ✓）⇒ **必落单** ✓ ⇒ 才会给它**新建一条账** ✓（要钉的正是这一路 ✓）。
    _two = _one + [(0, 540.0, 300.0, 60.0, 60.0, 0.9)]
    _o = _t.process(None, ts=0.16, dets=_two)
    _hit = next((_r for _r in (_o.get("orphans") or [])
                 if abs(float(_r[0]) - 540.0) < 1.5 and abs(float(_r[1]) - 300.0) < 1.5), None)
    _bv2 = next((_v for _v in (_t._box_v or [])
                 if abs(float(_v[0]) - 540.0) < 1.5 and abs(float(_v[1]) - 300.0) < 1.5), None)
    #   ⭐⭐⭐⭐⭐ **判据改成钉"我刚修的那个真 bug"** ✗✗（用户 2026-10-06 ✓ 原话："**为何刚出现就被
    #     标号了？**" ✓✓）：那格框**这一拍给它自己新建了一条账** ✓ ⇒ 它**必须**记成
    #     "**新建**"（第 13 位 = `2` ✓、`m` 记 −1 ✓），**不许**被当成"孤框、按旁边那条账贴号" ✗
    #     —— ⚠ 改前实测（`9月30日(1)` 帧 30 ✓）屏幕上就是 **`编号 #20 ｜ 匹配分 -1.00`** ✗✗
    #     （自相矛盾 ✓：#20 正是**为这格刚建的账** ✓）；根因 = `_box_v` 原来**只由 `_pairs` 拼** ✗
    #     ⇒ **新建账自己那格框不在里面** ⇒ 被当成孤框 ✓ ⇒ 贴了它自己的号还标"推的" ✗。
    check(_bv2 is not None and int(_bv2[4]) > 0 and _hit is None
          and (len(_bv2) <= 12 or int(_bv2[12]) == 2),
          "① **新建账「自己那格框」记成「新建」、不算孤框** ✓（那格 `_box_v` 第 13 位 = %s ✓〔要 2 ✓〕、"
          "归 #%s ✓；`orphans` 里有它吗 = %s ✓〔要 `None` ✓〕）—— ⚠ 改前它是"
          "「`编号 #20` + `匹配分 -1.00`」✗✗（自相矛盾 ✓）"
          % (int(_bv2[12]) if (_bv2 is not None and len(_bv2) > 12) else "-",
             int(_bv2[4]) if _bv2 is not None else "-",
             "有" if _hit is not None else "无"))
    _own = [(float(_v[0]), float(_v[1])) for _v in _t._box_v]
    _bad2 = [(round(float(_r[0])), round(float(_r[1]))) for _r in (_o.get("orphans") or [])
             if any(abs(float(_r[0]) - _x) < 1.5 and abs(float(_r[1]) - _y) < 1.5
                    for _x, _y in _own)]
    check(not _bad2,
          "② **反面：已经拿到编号的框不进 `orphans`** ✓（已归属的框 = %s ✓；误报的 = %s ✓〔要空 ✓〕）"
          " —— ⚠ 一刀切「凡没拿到账的都报」⇒ 这条红 ✗"
          % ([(round(x), round(y)) for x, y in _own], _bad2))


def test_fuse_keep_whole_circle_inside():
    """⭐⭐⭐⭐⭐ **融合期："圆只要有任一部分出框 ⇒ 往回夹"** ✗✗（**整圈必须落在框里** ✓）。

    ⚠⚠ **两句用户原话（合起来才是口径 ✓ —— 我读反过两次 ✗，如实记 ✓）**：
      · ① "**纠正：如果本拍圆向着白箭头（相对群体位移）的位移会使整个圆（不只是圆心）超出了
        融合框，则往回夹取作为本拍的最终圆位置**" ✓✓ —— ⚠ **关键词 = "（不只是圆心）"** ✓；
        我**第一版**读成"圆心出框就夹"✗（**太紧** ✓）；
      · ② "**我的意思不是整个圆出框往回，而是整个圆只要有任何一部分出框就往回**" ✓✓ ——
        我**第二版**读成"整个圆全都在框外才夹"✗（**太松** ✓），他当场否掉 ✓。
      ⇒ **定案**：**只要有任一部分探出框边 ⇒ 往回夹** ✓（判据 = 圆心是否落在"**框内缩一个
        半径**"的矩形里 ✓ = 圆盘是否整个在框内 ✓）。
    ⚠⚠ **代价如实说** ✗：这条**每拍都硬夹** ⇒ 会把"**四条边推圆心**"那份削掉（推出去就被拉回 ✓）
      —— 这是他要的口径 ✓（"**圆就不该出框**"✓）；想"既推得动又不探出"，正解 = **把「拆框」
      开回去**（`_FUSE_SPLIT` ✓ 夹进"属于真目标的那一块" ✓）。
    钉七条（含**边界**与**反面** ✓）：
      ① **圆心已出框 20px（圆盘还搭在框上）⇒ 照样夹回 490** ✓（⚠⚠ **改前那版**写"相交就不夹"✗
         ⇒ 它会**一动不动** ⇒ 本条立刻红 ✓）；
      ② **整圆出框 80px ⇒ 夹回 490** ✓（= 整圈刚好贴住框内壁 ✓、y 不动 ✓）；
      ③ ⭐ **只探出 2px 也要夹** ✓（`pos.x = 492` ⇒ 490 ✓ —— 这就是他第②句的字面 ✓）；
      ④ 边界：**框那根轴装不下圆**（高 40 < 2r = 120 ✓）⇒ 那根轴**退回原边** ✓（夹到框内 ±20 ✓
         而不是被压成 0、也不是不夹 ✓）；
      ⑤ **整圈在框内 ⇒ 绝不动它** ✓（防"每拍无端挪位置" ✗）；
      ⑥ ⭐ **连击计数**（`_fuse_keep_n` ✓ —— 「丙」「乙」共用的门 ✓）：1 → 2 → **没夹就归 0** → 1 ✓
         （"压力松开 ⇒ 重新起算" ✓）；
      ⑦ ⭐ **拿别人的融合框夹 ⇒ 计数 0** ✓（⚠ 实测帧 27 那种硬拽不许算连击 ✗）。
    """
    def _mk(_pos, _bw=300.0, _bh=300.0, _st=1):
        _t = MotionTracker()
        _t.tid = 5
        _t.tgt_rad = 60.0                      # 半径 60 ✓（框 300×300 ⇒ 装得下 ✓ ⇒ 内缩 ±90 ✓）
        _t.pos = (float(_pos[0]), float(_pos[1]))
        _t._box_v = [(400.0, 300.0, 0.0, 0.0, 5, 0.0, 0.0, int(_st), True, True,
                      float(_bw), float(_bh))]
        return _t

    _t1 = _mk((570.0, 300.0))                  # 框右边界 550 ⇒ 圆心已出框 **20px** ✓
    _t1._fuse_keep_inside(_t1._circ_rad())
    check(_t1.pos == (490.0, 300.0),
          "① **圆心已出框（圆盘还搭在框上）⇒ 照样夹回** ✓（位置 = (%.0f,%.0f) ✓〔要 (490,300) ✓ "
          "= 整圈刚好贴住内壁 ✓〕）—— ⚠⚠ **改前那版**（圆盘还相交就不夹 ✗）⇒ 它会**一动不动** ⇒ "
          "本条立刻红 ✗" % (_t1.pos[0], _t1.pos[1]))

    _t2 = _mk((630.0, 300.0))                  # 圆心离框 **80px > r 60** ⇒ 连圆盘都在框外 ✓
    _t2._fuse_keep_inside(_t2._circ_rad())
    check(_t2.pos == (490.0, 300.0),
          "② **整圆出框 80px ⇒ 往回夹** ✓（夹到 (%.0f,%.0f) ✓〔要 (490,300) ✓ = 整圈刚好贴住框"
          "内壁 ✓〕、y **不动** ✓）—— ⚠ 不夹 ⇒ 圆飞出去 ✗；⚠ 夹到框心（400 ✓）⇒ 那也不对 ✗"
          % (_t2.pos[0], _t2.pos[1]))

    _t3 = _mk((492.0, 300.0))                  # ⭐ **只探出 2px** ✓
    _t3._fuse_keep_inside(_t3._circ_rad())
    check(_t3.pos == (490.0, 300.0),
          "③ **只探出 2px 也要夹回** ✓（位置 = (%.0f,%.0f) ✓〔要 (490,300) ✓〕）—— ⚠⚠ 这就是"
          "用户第②句的字面（\"**有任何一部分出框就往回**\"✓）；⚠ 旧版（相交才管 ✗）会放它过去 ⇒ "
          "本条红 ✗" % (_t3.pos[0], _t3.pos[1]))

    _t4 = _mk((630.0, 300.0), _bh=40.0)        # 高 40 < 2r = 120 ⇒ 那根轴**装不下圆** ✓
    _t4._fuse_keep_inside(_t4._circ_rad())
    check(_t4.pos == (490.0, 300.0),
          "④ **框某根轴装不下圆 ⇒ 那根轴退回原边** ✓（位置 = (%.0f,%.0f) ✓〔要 (490,300) ✓："
          "x 照旧内缩到 490 ✓、y 夹在框内 ±20 里 ⇒ 300 不动 ✓〕）—— ⚠ 写成 `max(0, …)` ⇒ 会被"
          "压成框心 ✗" % (_t4.pos[0], _t4.pos[1]))

    _t5 = _mk((430.0, 320.0))                  # 整圈都在框里 ✓（|30| ≤ 90 ✓、|20| ≤ 90 ✓）
    _t5._fuse_keep_inside(_t5._circ_rad())
    check(_t5.pos == (430.0, 320.0),
          "⑤ **整圈在框里 ⇒ 绝不动它** ✓（位置 = (%.0f,%.0f) ✓〔要 (430,320) ✓〕）—— ⚠ 每拍无端"
          "挪位置 ⇒ 本条红 ✗" % (_t5.pos[0], _t5.pos[1]))

    #   ⭐⭐ ⑥⑦ 是**「连击计数」`_fuse_keep_n` 的语义** ✓（用户 2026-10-07 ✓ "**夹取的第 2 拍**"
    #     那道门 ✓）—— 「丙」（`process` 里那段 ✓）与「乙」（`rel_arrow` ✓）**共用**它 ✓。
    _t6 = _mk((630.0, 300.0))
    _t6._fuse_keep_inside(_t6._circ_rad())
    _k6a = int(_t6._fuse_keep_n)               # 夹第 1 拍 ✓
    _t6.pos = (630.0, 300.0)
    _t6._fuse_keep_inside(_t6._circ_rad())
    _k6b = int(_t6._fuse_keep_n)               # 连着夹第 2 拍 ✓
    _t6.pos = (400.0, 300.0)                   # 圆回到框里 ⇒ **这一拍没夹** ✓
    _t6._fuse_keep_inside(_t6._circ_rad())
    _k6c = int(_t6._fuse_keep_n)
    _t6.pos = (630.0, 300.0)
    _t6._fuse_keep_inside(_t6._circ_rad())
    _k6d = int(_t6._fuse_keep_n)               # 断开之后**从 1 重新起算** ✓
    check((_k6a, _k6b, _k6c, _k6d) == (1, 2, 0, 1),
          "⑥ **连击计数**：夹 1 拍 ⇒ **%d** ✓、连着再夹 ⇒ **%d** ✓、中间一拍没夹 ⇒ **%d** ✓（= "
          "**压力松开 ⇒ 归零** ✓）、随后再夹 ⇒ **%d** ✓〔要 (1,2,0,1) ✓〕）—— ⚠ 用\"总拍数\"而"
          "不是\"连击\"✗ ⇒ 第 3 项会是 2 ⇒ 本条红 ✗（两段不相邻的夹取不许并成一段 ✓）"
          % (_k6a, _k6b, _k6c, _k6d))

    _t7 = _mk((630.0, 300.0))
    _t7._box_v = [(400.0, 300.0, 0.0, 0.0, 9, 0.0, 0.0, 1, True, False, 300.0, 300.0)]
    _t7._fuse_keep_inside(_t7._circ_rad())
    check(_t7.pos == (490.0, 300.0) and int(_t7._fuse_keep_n) == 0,
          "⑦ **拿「别人的」融合框夹 ⇒ 计数 0** ✓（位置照旧夹到 (%.0f,%.0f) ✓〔要 (490,300) ✓〕；"
          "计数 = **%d** ✓〔要 0 ✓〕）—— ⚠⚠ 实测 `10月1日` **帧 27**：那种时候是拿别人的框硬拽 "
          "(66.2,−92.6) ✗ ⇒ 若也算连击 ⇒ 「丙」会把这一跳灌进速度 ✗✗（本条红 ✓）"
          % (_t7.pos[0], _t7.pos[1], int(_t7._fuse_keep_n)))


def test_rel_arrow_uses_final_pos_in_fusion():
    """⭐⭐⭐⭐⭐ **白箭头在「融合期第 2 拍起」改用最终圆心位置算** ✗✗（用户 2026-10-07 ✓ 原话：
    "**试试这样：在融合期（夹取的第2拍开始），平滑速度以每拍圆心最终位置计算（而不是位置限制
    之前）**" ✓✓）。

    ⚠ 两个口径（见 `MotionTracker.rel_arrow` ✓，都得钉 ✓）：
      · **平时 / 融合首拍**（`stuck <= 1` ✓）⇒ 白箭头 = "**本该走**" = 实际 ＋ 被「整圆出框就
        回夹」拉回去的那一截 ✓（= 用户 2026-10-06 要的"**怼着墙走**" ✓）；
      · **融合期第 2 拍起**（`stuck >= 2` ✓）⇒ 白箭头 = **最终圆心位置**算出来那一步（= 实际 ✓）
        —— 这几拍的夹取是"**每拍都发生的持续压力**"✓ ⇒ 如实画 ✓、与"没夹的拍"同一个基准 ✓。
    ⭐⭐⭐⭐⭐ **乙（用户 2026-10-07 ✓ 他选的"丙 ＋ 乙" ✓）**：上面那一档（照实 ✓）**再加一道
      EMA 平滑** ✓（`_ARROW_FUSE_EMA = 0.5` ✓ 与 `_EDGE_BLEND` 同一个节奏 ✓）—— ⚠⚠ **为什么非有
      不可** ✗：照实之后那几拍读数**每拍只走 ~5px 还左右翻** ✗（贴墙时圆是被"观测混合"拽着走的 ✓）
      ⇒ 相邻两拍角差 **68°/57°** ✗✗（那就不是"平滑速度"了 ✗）；⚠ **离开这一档 ⇒ 重置** ✓
      （两套基准不许互相渗 ✗）。
    钉九条（含**反面**＋**边界**＋**乙** ✓）：
      ① 平时（`stuck == 0`）＋ 夹过 ⇒ 照旧"本该走的" ✓；
      ② **融合首拍**（`stuck == 1`）⇒ 同上 ✓（那一下是"**一次性把圆放好**" ✗，不该把箭头按瘪 ✓）；
      ③ ⭐ **融合第 2 拍**（`stuck == 2`）⇒ **最终位置** ✓（⚠⚠ 改前会给"实际 ＋ 被夹回那一截" ✗
         ⇒ 本条立刻红 ✓）；
      ④ 更靠后（`stuck == 5`）⇒ 同上 ✓（不是只第 2 拍那一拍 ✓）；
      ⑤ 没夹（`_pos_intent == pos` ✓）⇒ 两支**逐位相同** ✓（⇒ 平时画面一个字不变 ✓）；
      ⑥ 边界：`_last_inc is None`（第一拍 ✓）/ `pos is None` ⇒ **原样返回** ✓（不猜 ✓）；
      ⑦ ⭐ **`stuck >= 2` 但座位自己那格融合框不在了**（画面里只剩**别人**的融合框 ✓）⇒ 照旧
         "本该走的" ✓（⚠⚠ 实测 `10月1日` **帧 27** 那种"**拿别人的框硬拽 (66.2,−92.6)**"✗
         不许被画成箭头 ✗）。
    """
    def _mk(_stuck, _pos=(630.0, 300.0), _intent=(600.0, 300.0), _own_fuse=True):
        _t = MotionTracker()
        _t.tid = 5
        _t.pos = (float(_pos[0]), float(_pos[1]))
        _t._pos_intent = (float(_intent[0]), float(_intent[1]))
        _tr = _LM._Track(5, 400.0, 300.0, 150.0, 150.0)
        _tr.stuck = int(_stuck)
        _t.tracks.append(_tr)
        #   ⚠ 框表：`(cx, cy, …, tid, …, st, has_tgt, own, w, h)` ✓ —— `_v[4]` = 归属账 id ✓、
        #     `_v[7]` = 融合标记（1 = 融合 ✓）⇒ `_own_fuse=False` 时给**别人的**融合框 ✓。
        _t._box_v = [(400.0, 300.0, 0.0, 0.0, 5 if _own_fuse else 9,
                      0.0, 0.0, 1, True, bool(_own_fuse), 200.0, 200.0)]
        return _t

    def _eq(_a, _b):
        return (_a is not None and _b is not None
                and abs(float(_a[0]) - float(_b[0])) < 1e-9
                and abs(float(_a[1]) - float(_b[1])) < 1e-9)

    _last = (-9.1, -1.3)                       # 本拍**实际**那一步（相对群体 px/拍 ✓）
    _fix = (_last[0] + (600.0 - 630.0), _last[1] + (300.0 - 300.0))   # = 实际 ＋ 被夹回的那一截 ✓

    _t1 = _mk(0)
    check(_eq(_t1.rel_arrow(_last), _fix),
          "① **平时（没融合）＋ 夹过 ⇒ 照旧「本该走的」** ✓（白箭头 = (%.1f,%.1f) ✓〔要 (%.1f,%.1f) "
          "= 实际 ＋ 被夹回那一截 ✓〕）—— ⚠ 平时也改成\"实际\"✗ ⇒ 这条红 ✗"
          % (_t1.rel_arrow(_last)[0], _t1.rel_arrow(_last)[1], _fix[0], _fix[1]))

    _t2 = _mk(1)
    check(_eq(_t2.rel_arrow(_last), _fix),
          "② **融合首拍（`stuck == 1`）⇒ 同上** ✓（白箭头 = (%.1f,%.1f) ✓〔要 (%.1f,%.1f) ✓〕）"
          "—— ⚠⚠ 那一下夹取是「**一次性把圆放好**」✗（不是墙 ✓）⇒ 照旧按\"本该走的\"画 ✓"
          % (_t2.rel_arrow(_last)[0], _t2.rel_arrow(_last)[1], _fix[0], _fix[1]))

    _t3 = _mk(2)
    _t3a = _t3.rel_arrow(_last)
    check(_eq(_t3a, _last),
          "③ ⭐ **融合第 2 拍（`stuck == 2`）⇒ 用最终圆心位置** ✓（白箭头 = (%.1f,%.1f) ✓〔要 "
          "(%.1f,%.1f) = 实际那一步 ✓〕）—— ⚠⚠ **改前**给的是\"实际 ＋ 被夹回那一截\" (%.1f,%.1f) ✗ "
          "⇒ 本条立刻红 ✗（= 用户 2026-10-07 那句\"**以每拍圆心最终位置计算（而不是位置限制"
          "之前）**\" ✓）"
          % (_t3a[0], _t3a[1], _last[0], _last[1], _fix[0], _fix[1]))

    _t4 = _mk(5)
    check(_eq(_t4.rel_arrow(_last), _last),
          "④ **更靠后（`stuck == 5`）⇒ 同上** ✓（白箭头 = (%.1f,%.1f) ✓〔要 (%.1f,%.1f) ✓〕）"
          "—— ⚠ 只在第 2 拍那一拍生效 ✗ ⇒ 这条红 ✗（**整段融合期**都算 ✓）"
          % (_t4.rel_arrow(_last)[0], _t4.rel_arrow(_last)[1], _last[0], _last[1]))

    _t5 = _mk(2, _intent=(630.0, 300.0))
    _t5b = _mk(0, _intent=(630.0, 300.0))
    check(_eq(_t5.rel_arrow(_last), _last) and _eq(_t5b.rel_arrow(_last), _last),
          "⑤ **没夹（`_pos_intent == pos`）⇒ 两支逐位相同** ✓（融合期 = (%.1f,%.1f) ✓、平时 = "
          "(%.1f,%.1f) ✓〔都要 = (%.1f,%.1f) ✓〕）—— ⚠ 两口径在\"没夹\"时**本来就该相等** ✓"
          "（⇒ 平时画面一个字不变 ✓）"
          % (_t5.rel_arrow(_last)[0], _t5.rel_arrow(_last)[1],
             _t5b.rel_arrow(_last)[0], _t5b.rel_arrow(_last)[1], _last[0], _last[1]))

    _t6 = _mk(5)
    _t6b = _mk(5, _own_fuse=False)
    check(_eq(_t6b.rel_arrow(_last), _fix),
          "⑦ ⭐ **`stuck >= 2` 但只剩别人的融合框 ⇒ 照旧「本该走的」** ✓（白箭头 = (%.1f,%.1f) ✓〔要 "
          "(%.1f,%.1f) ✓〕）—— ⚠⚠ 实测 `10月1日` 帧 27：那种时候夹取是「**拿别人的框**」硬拽 "
          "(66.2,−92.6) ✗ ⇒ 若也算这一档 ⇒ 箭头被画成一根朝右下的大箭头 ✗✗（本条红 ✓）"
          % (_t6b.rel_arrow(_last)[0], _t6b.rel_arrow(_last)[1], _fix[0], _fix[1]))

    _t7 = MotionTracker()
    _t7.tid = 5
    _t7.pos = None
    check(_t6.rel_arrow(None) is None and _eq(_t7.rel_arrow(_last), _last),
          "⑥ **边界：`_last_inc is None`（第一拍 ✓）⇒ 原样 None** ✓（= %s ✓）；**`pos is None` ⇒ 原样"
          "给实际** ✓（= (%.1f,%.1f) ✓〔要 = 实际 ✓ —— 没位置就算不了\"被夹回那一截\" ✓ 不硬凑 ✓〕）"
          "—— ⚠ 第一拍没有\"上一拍位置\"✗ ⇒ 不许猜一个数出来 ✓"
          % (_t6.rel_arrow(None), _t7.rel_arrow(_last)[0], _t7.rel_arrow(_last)[1]))

    #   ⭐⭐ ⑧⑨ 是**「乙」**（用户 2026-10-07 ✓ "丙 ＋ 乙"）—— 那一档**照实**的读数**再加一道
    #     EMA 平滑** ✓（`_ARROW_FUSE_EMA = 0.5` ✓ 见 `rel_arrow` ✓）。
    _t8 = _mk(5)
    _a8 = _t8.rel_arrow(_last)                 # 第一拍：**原样** ✓（不拿 0 起头 ✓）
    _b8 = _t8.rel_arrow((5.0, 5.0))            # 第二拍：**半量混** ✓
    _e8 = (0.5 * _last[0] + 0.5 * 5.0, 0.5 * _last[1] + 0.5 * 5.0)
    check(_eq(_a8, _last) and _eq(_b8, _e8),
          "⑧ **乙：连拍两拍 ⇒ 第二拍是半量平滑** ✓（第一拍 = (%.1f,%.1f) ✓〔要 = 实际 ✓ = "
          "**原样** ✓〕；第二拍 = (%.1f,%.1f) ✓〔要 (%.1f,%.1f) = 0.5×上一拍 ＋ 0.5×本拍 ✓〕）"
          "—— ⚠⚠ **照实但不平滑**（实测 `10月1日` 帧 21~24 相邻角差 **68°/57°** ✗）⇒ 那就不叫"
          "\"**平滑**速度\"了 ✗"
          % (_a8[0], _a8[1], _b8[0], _b8[1], _e8[0], _e8[1]))

    _t9 = _mk(5)
    _t9.rel_arrow(_last)                       # 先攒一次 EMA ✓
    _r9 = _t9._by_id(5)
    _r9.stuck = 0
    _t9.rel_arrow((9.0, 9.0))                  # 离开这一档（融合结束 ✓）⇒ **重置** ✓
    _r9.stuck = 5
    _c9 = _t9.rel_arrow((5.0, 5.0))            # 再进来 ⇒ 又是**原样** ✓（不掺旧的 ✓）
    check(_eq(_c9, (5.0, 5.0)),
          "⑨ **离开这一档 ⇒ 重置** ✓（再进来的第一拍 = (%.1f,%.1f) ✓〔要 (5.0,5.0) = 原样 ✓〕）"
          "—— ⚠ 不重置 ⇒ 会掺进上一次的残留（≈ (%.1f,%.1f) ✗）⇒ 两套基准互相渗 ✗（24→25 那条"
          "缝会被糊成一条斜线 ✗ 更假 ✓）" % (_c9[0], _c9[1],
                                      0.5 * (0.5 * _last[0] + 0.5 * 9.0) + 0.5 * 5.0,
                                      0.5 * (0.5 * _last[1] + 0.5 * 9.0) + 0.5 * 5.0))


def test_fuse_clamp_never_touches_velocity():
    """⭐⭐⭐⭐⭐ **"怼着墙走"：夹取只挪位置，绝不进速度平滑** ✗✗（用户 2026-10-06 ✓ 原话：
    "**注意不管是把圆从外往框内部夹取还是把圆限制在框内部的夹取，都不应该算进速度平滑里，
    也就是说它在框内可能不动，但是我们的相对群体速度（白箭头）要保持，是一种"怼着墙走"的感觉**"
    ✓✓）。

    ⚠ 两件事分开说 ✓：
      · **位置**：可以被夹住 ✓（贴墙上**一动不动** ✓）；
      · **速度**：`vel` ✗ / 座位 `vr`（**白箭头** ✓）/ `kf` 那份速度 —— **一个字都不许变** ✓
        ⇒ 它照旧指着原方向 ✓ ⇒ 墙一撤（融合结束 / 框挪开 ✓）就**接着按原箭头走** ✓
        = "**怼着墙**" ✓。
    ⚠⚠ 为什么必须有这条钉子 ✗：**夹取增益**（`clamp_gain` ✓ 界面那格 ✓ = 用户 2026-10-04 要的
      "把夹取那份修正**算进速度平滑**" ✓，2026-10-06 默认关成 **0** ✓）⇒ 只要有人把这条链
      改回"夹了也喂速度"✗，**本节立刻红** ✓。
    钉四条：
      ① **从外往里夹**（整圆在框外 ✓）⇒ `pos` 挪了 ✓、`vel` / `vr` / `kf` 速度 **逐字不变** ✓；
      ② **限制在框内**（只探出 2px ✓）⇒ 同上 ✓（两类夹取**都**不许进速度 ✓）；
      ③ **"怼着墙"**：贴墙那拍再算一次 ⇒ 位置**不动** ✓（钉在墙上 ✓），而 `vr` 照旧**非 0**、
        方向不变 ✓（= "人在墙上，箭头照旧往前" ✓）；
      ④ 前提：**模块默认 `_CLAMP_GAIN == 0`** ✓（⚠ 界面那格还能调回来 ✓ —— 这条只钉**默认值** ✓）。
    ⚠⚠ **它只管「夹取函数本身」** ✗✗（用户 2026-10-07 ✓ 他选的「**丙**」另开一处 ✓，别混 ✓）：
      本节钉的是 `_fuse_keep_inside` **自己**不许碰速度 ✓（它只挪位置 ✓ —— 这正是"位置是硬的、
      速度是软的"那条分工 ✓）；而「丙」（`process` 里那段 ✓）是用户 2026-10-07 明确要的"**融合期
      第 2 拍起，平滑速度按每拍圆心最终位置重算**" ✓ ⇒ **是**改速度状态的 ✓（钉子见
      `test_fuse_second_clamp_recomputes_velocity` ✓）。两者**不是一回事** ✓。
    """
    def _mk(_pos, _vr=(-28.4, -12.0), _vel=(-2.8, -0.6)):
        _t = MotionTracker()
        _t.tid = 5
        _t.tgt_rad = 60.0
        _t.pos = (float(_pos[0]), float(_pos[1]))
        _t._box_v = [(400.0, 300.0, 0.0, 0.0, 5, 0.0, 0.0, 1, True, True, 300.0, 300.0)]
        _tr = _LM._Track(5, 400.0, 300.0, 150.0, 150.0)
        _tr.vr = (float(_vr[0]), float(_vr[1]))
        _tr.vel = (float(_vel[0]), float(_vel[1]))
        _tr.kf.x[2], _tr.kf.x[3] = float(_vr[0]), float(_vr[1])
        _t.tracks.append(_tr)
        return _t, _tr

    def _snap(_s):
        return (tuple(_s.vel), tuple(_s.vr),
                (float(_s.kf.x[2]), float(_s.kf.x[3])))

    _t1, _s1 = _mk((630.0, 300.0))             # 整圆都在框外 ⇒ 夹 ✓
    _b1 = _snap(_s1)
    _t1._fuse_keep_inside(_t1._circ_rad())
    check(_t1.pos == (490.0, 300.0) and _snap(_s1) == _b1,
          "① **从外往里夹：位置挪了 ✓、速度一个字没动** ✓（`pos` = (%.0f,%.0f) ✓〔要 (490,300) ⇒ "
          "**真夹了** ✓〕；夹前/夹后的 `vel`＋`vr`＋`kf` 速度 = %s ⇒ **逐字相同** ✓）—— ⚠ 把夹取那份"
          "并进速度平滑 ⇒ 本条红 ✗" % (_t1.pos[0], _t1.pos[1], _b1))

    _t2, _s2 = _mk((492.0, 300.0))             # **只探出 2px** ⇒ 限制在框内 ✓
    _b2 = _snap(_s2)
    _t2._fuse_keep_inside(_t2._circ_rad())
    check(_t2.pos == (490.0, 300.0) and _snap(_s2) == _b2,
          "② **限制在框内（只探出 2px）：同样只有位置动** ✓（`pos` = (%.0f,%.0f) ✓；速度 = %s ✓"
          "〔与夹前逐字相同 ✓〕）—— ⚠ **两种夹取**（从外往里 / 限制在框内）**都不许**进速度 ✓"
          % (_t2.pos[0], _t2.pos[1], _b2))

    _t3, _s3 = _mk((630.0, 300.0))
    #   ⚠ 记**夹之前**的白箭头 ✓（这样"夹取偷偷削速度"✗也会被这条咬住 ✓ 见反向验证 ✓）
    _vr0 = tuple(_s3.vr)
    _t3._fuse_keep_inside(_t3._circ_rad())
    _p3 = _t3.pos
    _vr3 = tuple(_s3.vr)
    _t3._fuse_keep_inside(_t3._circ_rad())     # 已经贴墙 ⇒ 再算一次不许挪 ✓
    check(_t3.pos == _p3 and tuple(_s3.vr) == _vr3 and tuple(_s3.vr) == _vr0
          and abs(float(_s3.vr[0])) > 1.0 and abs(float(_s3.vr[1])) > 1.0,
          "③ **「怼着墙走」**：贴墙那拍再算一次 ⇒ 位置**一动不动** ✓（(%.0f,%.0f) ✓ = 钉在墙上 ✓），"
          "而**白箭头 `vr` 照旧** = (%.1f,%.1f) ✓〔要**非 0** 且与夹前相同 ✓ = 人贴在墙上、"
          "箭头照旧往前 ✓〕—— ⚠ 把贴墙理解成「速度也归零」✗ ⇒ 本条红 ✗"
          % (_t3.pos[0], _t3.pos[1], float(_s3.vr[0]), float(_s3.vr[1])))

    check(float(_LM._CLAMP_GAIN) == 0.0,
          "④ **默认「夹取增益」= 0** ✓（模块常量 `_CLAMP_GAIN` = %.2f ✓〔要 0 ✓〕＝ 夹取那份修正"
          "**不进**速度平滑 ✓；⚠ 界面那格还能调回来 ✓ 这条只钉**默认值** ✓）"
          % float(_LM._CLAMP_GAIN))


def test_fuse_second_clamp_recomputes_velocity():
    """⭐⭐⭐⭐⭐ **丙（用户 2026-10-07 ✓ 他选的"丙 ＋ 乙" ✓）**：**融合期「夹取的第 2 拍起」⇒
    平滑速度按「每拍圆心最终位置」重算** ✗✗（原话："**在融合期（夹取的第2拍开始），平滑速度以
    每拍圆心最终位置计算（而不是位置限制之前）**" ✓✓）。

    ⚠⚠ **病根**（**实测** ✓ `10月1日` 帧 21~24 ✓）：位置每拍都被「整圆出框就回夹」拉回来 ✗，
      可**速度状态一个字没动** ✗ ⇒ 下一拍照旧按老速度冲（≈ 50px/拍 ✓）⇒ **每拍都得再夹一次**
      （自己跟自己打架 ✓）；等帧 25 融合框一消失 ✓ ⇒ 位置与速度**两套账** ⇒ 白箭头与圆心在
      24→25 那一下"换基准" ✓（用户当场问"**相对帧24还是有明显的变向，什么原因**" ✓）。
    ⇒ 本钉子扫**真素材**（`10月1日` ✓）：凡是「座位自己那格融合框**连着**夹到第 2 拍」的那些拍 ✓，
      `vr` 必须**正好等于**"这一拍圆心最终位置算出来的相对步长" ✓（按 `pos_step_max` 削顶 ✓）、
      `vel` 必须 = "相机 ＋ `vr`" ✓。
    钉两条：
      ① 夹具成立：这类拍 **≥ 1** ✓（⚠ 实测全片就 **1** 拍（帧 22）✓ —— 生效之后墙**夹不动了** ✓
         ⇒ 连击归 0 ✓ **自限** ✓；没样本 ⇒ 假绿 ✗）；
      ② 每一拍都对得上 ✓（不符 = 0 笔 ✓）。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月1日.mp4")
    if not _src.exists():
        check(True, "跳过「丙：夹取第 2 拍起重算速度」真素材用例（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        from tools.live_lie import load_motion_cfg
        _frames = LD.load_video(str(_src))[0]
        _w = LD.DetsWorker(None, conf=0.25)
        _kw = load_motion_cfg()
    except Exception as _e:                    # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过「丙：夹取第 2 拍起重算速度」真素材用例（环境不齐：%s ✓）" % _e)
        return
    _t = MotionTracker(**_kw)
    _n, _bad = 0, []
    for _i in range(min(70, len(_frames))):
        _im = _frames[_i][1]
        _t.frame_wh = (int(_im.shape[1]), int(_im.shape[0]))
        _prev = _t.pos
        _t.process(_im, ts=_i * 0.16, dets=_w.detect(_im) or [])
        if int(getattr(_t, "_fuse_keep_n", 0)) < 2 or _t.pos is None or _prev is None:
            continue
        _s = _t._by_id(_t.tid) if _t.tid is not None else None
        if _s is None:
            continue
        _cam = _t._med_ema if getattr(_t, "_med_ema", None) is not None else (0.0, 0.0)
        _exp = (float(_t.pos[0]) - float(_prev[0]) - float(_cam[0]),
                float(_t.pos[1]) - float(_prev[1]) - float(_cam[1]))
        _nn = math.hypot(_exp[0], _exp[1])
        if _nn > float(_t.pos_step_max):
            _kk = float(_t.pos_step_max) / max(1e-6, _nn)
            _exp = (_exp[0] * _kk, _exp[1] * _kk)
        _n += 1
        if (abs(float(_s.vr[0]) - _exp[0]) > 1e-6
                or abs(float(_s.vr[1]) - _exp[1]) > 1e-6
                or abs(float(_s.vel[0]) - float(_cam[0]) - float(_s.vr[0])) > 1e-6):
            _bad.append((_i + 1, tuple(_s.vr), _exp))
    check(_n >= 1 and not _bad,
          "①② **丙：`_fuse_keep_n >= 2` 那几拍 ⇒ `vr` 正好 = 「最终位置那一拍步长」** ✓（样本 **%d** "
          "拍 ✓〔要 ≥1 ✓〕；不符 **%d** 笔 ✓〔要 0 ✓〕）—— ⚠⚠ **首拍不许算**（`_fuse_keep_n == 1` ✗"
          " —— 那一下是「一次性把圆放好」✗）/ **别人的框不许算**（帧 27 ✗）—— 两处都由 "
          "`_fuse_keep_n` 把关 ✓（见 `_fuse_keep_inside` ✓）；%s"
          % (_n, len(_bad), _bad[:2] if _bad else "0 笔 ✓"))


def test_fusion_box_owner_is_fake_target():
    """⭐⭐⭐⭐⭐ **融合框的"主人" = 那格假目标** ✗✗（用户 2026-10-06 ✓ 原话："**先把融合框里的标签
    分配给假目标，例如 61 帧 #11 不应该写着灰框推的，因为它才是这个框的主人**" ✓✓）。

    ⚠ 口径（**用户模型** ✓ 原话："**如果检出框比作座位，真目标大部分时间都是没有座位的，它只能
      通过融合框短暂的跟假目标『坐在一个座位上』，融合框最终肯定是优先还给假目标**" ✓✓）——
      ⇒ 画在融合框上的编号，该写"**一直守着这一格的那条账**" ✓（不是"配对到谁写谁" ✗：座位只是
      **蹭坐** ✓）。
    ⚠⚠⚠ **纯标签** ✗✗（用户强调"**我们现在要解决的是纯数据层面的占位问题**" ✓）：`_box_v[13]`
      只给演示窗画编号用 ✓ —— **配对 / 位置 / 速度 / KF 一个字不动** ✓
      （⚠ 我上一轮动了"谁拿框"⇒ 全片**换身份 0 → 2 次** ✗ ⇒ 已撤回 ✓ 见 `_pair` 那段注释 ✓）。
    钉两条：
      ① 融合那一拍，融合框的 `_box_v[13]` = **那条老账（假目标）的 id** ✓（**不是座位** ✓）；
      ② **位置轨迹逐拍一字不变** ✓（把 `_FUSE_OWNER_HITS` 抬到无穷 ⇒ 同一段跑出来**完全一样** ✓
         —— ⚠ 谁要是把"标签"写成"顺带影响配对 / 位置" ✗ ⇒ 这条立刻红 ✗）。
    """
    def _seq(n=20, brick_sz=200.0):
        _f0 = _frame(9)
        _s = []
        for _i in range(n):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            # 座位那条（相对群体 +12px/拍 ✓）；融合窗口里它那格框**变大**（= 并集框 ✓ 面积比 1.78）
            #   ⚠⚠ **窗口要等"砖"那条攒够 `_FUSE_OWNER_HITS` 拍再开** ✗✗（**踩过一次** ✓）：我第一版
            #     第 3 拍就融合了 ⇒ 那会儿砖的 `hits` 才 2 ✗ ⇒ 认不出主人 ⇒ ① 假红 ✓。
            _sz = 200.0 if (12 <= _i <= 17) else 150.0
            _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1], _sz, _sz, 0.9)]
            # "砖"那条：**就守在这一带** ✓（融合期它会被并进大框 ⇒ 拿不到框 ✓ = "蹭坐"的另一半 ✓）
            _d = _d + [(0, 300.0 + _cam[0], 250.0 + _cam[1], brick_sz, brick_sz, 0.94)]
            _s.append(_d)
        return _s

    def _run(_hold, _n=20):
        _old = _LM._FUSE_OWNER_HITS
        _LM._FUSE_OWNER_HITS = _hold
        try:
            _t = MotionTracker(min_hits=3)
            _rows, _pos = [], []
            for _i, _d in enumerate(_seq(_n)):
                _t.process(None, ts=_i * 0.16, dets=_d)
                _fb = next((_b for _b in _t._box_v if len(_b) > 7 and int(_b[7]) == 1), None)
                if _fb is not None:
                    _rows.append((_i + 1, int(_fb[4]),
                                  int(_fb[13]) if len(_fb) > 13 else -1))
                _pos.append(None if _t.pos is None else (round(_t.pos[0], 6),
                                                         round(_t.pos[1], 6)))
            return _t, _rows, _pos
        finally:
            _LM._FUSE_OWNER_HITS = _old

    _t1, _rows1, _pos1 = _run(8)
    _t2, _rows2, _pos2 = _run(10 ** 9)          # 反面对照：门槛抬到无穷 ⇒ "不认任何主人" ✓
    # ⚠⚠ **判定要写准** ✗✗（**踩过一次** ✓）：我第一版要求"**每一个**融合框都得有主人" ✗ ⇒ 可夹具
    #   前几拍**画面里只有一条账**（没人可当主人 ✓）⇒ 假红 ✓。⇒ 只钉两条真性质：
    #     · **至少有一拍认出了主人** ✓（那几拍"砖"那条已经够老 ✓）；
    #     · **从不把主人认成座位自己** ✓（`_r[2] == _r[1]` ⇒ 红 ✓）。
    _with1 = [(_r[0], _r[1], _r[2]) for _r in _rows1 if int(_r[2]) >= 0]
    _bad1 = [(_r[0], _r[1], _r[2]) for _r in _rows1
             if int(_r[2]) >= 0 and int(_r[2]) == int(_r[1])]
    check(bool(_with1) and not _bad1,
          "① **融合框的编号写「主人」（那格假目标 ✓），不是「配对到谁写谁」** ✓（认出主人的拍 = %s ✓"
          "〔每项 = (帧, 配对到谁, **主人**) ✓ ⇒ 主人都是**另一条老账** ✓ 不是座位 ✓〕；"
          "**认成座位**的拍 = %s ✓〔要空 ✓〕）—— ⚠ 没有 `_box_v[13]` ⇒ 本条红 ✗"
          % (_with1[:4], _bad1[:2]))

    check(_rows1 and _rows2 and len(_pos1) == len(_pos2) and _pos1 == _pos2,
          "② **纯标签：位置轨迹逐拍一字不变** ✓（认主人那跑 vs 不认主人那跑：报出位置**完全相同** ✓"
          "〔共 %d 拍 ✓〕）—— ⚠⚠ 谁把「标签」做成「顺带改配对 / 改位置」✗ ⇒ 本条立刻红 ✗"
          "（我上一轮动「谁拿框」就是这么翻车的 ✓ 换身份 0→2 ✓）" % len(_pos1))


def test_forced_claim_never_takes_others_seat():
    """⭐⭐⭐⭐⭐ **规则②不许认领"别人的座位"** ✗✗（用户 2026-10-06 ✓ 座位模型 ✓ 原话："**如果检出框
    比作座位，真目标大部分时间都是没有座位的，它只能通过融合框短暂的跟假目标『坐在一个座位上』，
    融合框最终肯定是优先还给假目标**" ✓✓）。

    ⚠⚠⚠ **实测病根**（`10月1日` ✓ 逐拍日志 ✓）：**换身份那两次正是规则②干的** ✗ —— 帧 45
      `#8 → #5`、帧 58 `#5 → #8`（`kind='forced'` ✓，理由写的是"干净框 ＋ 与圆相交 ＋ 偏离
      7.7 / 6.0 倍噪声" ✓）⇒ 座位在别人的座位上来回跳 ✗。⇒ 加门槛 ⑥ ✓：
      "**有一条老账（`hits ≥ _FUSE_OWNER_HITS` ✓）一直守着那格框**"（位置重叠 ≥ `_SEAT_KEEP_COV`
      = 0.60 ✓）⇒ **那是假目标的座位 ⇒ 不许认领** ✗。
    钉两条（**一正一反** ✓）：
      ① **老账守着那格框 ⇒ `_forced_claim` 返回 `None`** ✓（不许认领 ✓）—— ⚠ 把门槛抬到
         `2.0`（= 谁都拦不住 ✓）⇒ 它**就会认领** ⇒ 本条立刻红 ✗；
      ② **反面：那格框没人守着 ⇒ 照旧能认领** ✓ —— ⚠ 一刀切"永不许认领" ⇒ 这条红 ✗
         （用户的规则②本身要留着 ✓）。
    """
    def _mk(_keeper_on_box):
        """造一格"干净框 ＋ 与圆相交 ＋ 在动"的候选框（`#5` ✓），看规则②认不认它 ✓。

        ⚠⚠⚠ **夹具的两处讲究**（**踩过两次** ✓ 都写下来 ✓）：
          · `cur` 必须是**现任**（`#8` ✓）而**不是候选自己** ✗ —— 我第一版传的是 `#5` ✗
            ⇒ `_incumbent_strong(#5)` 成立 ⇒ **函数一开头就返回 `None`** ✗ ⇒ ①"绿得不对" ✓✓；
          · "守着这格框的老账"（`#11` ✓）**观测压在框上 ✓、`pred` 放远** ✗✗ —— 因为
            **另一条**门（③"这格框是不是某个编号的假目标又出现了" ✓）看的是 `pred` ✓
            ⇒ 只有把 `pred` 挪开，才**只测我这条新门** ✓。
        """
        _t = MotionTracker(min_hits=1)
        _t.frame_wh = (750, 500)
        _t.tgt_rad = 60.0
        _t.std_area = 22000.0
        _t.pos = (400.0, 300.0)                     # 圆心 ✓（下面那格框就贴着它 ✓）
        # 现任（`#8` ✓）：**要弱**（`dev` 噪声级 ✓）⇒ `_incumbent_strong` 不挡 ✓
        _cur = _LM._Track(8, 300.0, 300.0, 150.0, 150.0)
        _cur.hits, _cur.ma, _cur.dev, _cur.score = 30, 1.0, 0.0, 5.0
        _cur.pred = (300.0, 300.0)
        _t.tracks.append(_cur)
        # 候选 `#5`：`hits` 够老 ✓ ＋ `was_lost` ✓ ＋ 面积正常 ✓ ＋ 这一拍"在动" ✓
        _cand = _LM._Track(5, 430.0, 300.0, 150.0, 150.0)
        _cand.hits, _cand.ma, _cand.dev, _cand.score = 30, 1.0, 20.0, 40.0
        _cand.was_lost = True
        _cand.pred = (430.0, 300.0)
        _t.tracks.append(_cand)
        # "守着这格框的那条老账"（`#11` ✓）：`obs` **压在候选框上** ✓；`pred` **放远** ✗（避开③）
        _keep = _LM._Track(11, 430.0 if _keeper_on_box else 60.0, 300.0, 150.0, 150.0)
        _keep.hits, _keep.ma, _keep.dev, _keep.score = 30, 1.0, 0.5, 2.0
        _keep.pred = (60.0, 300.0)
        _t.tracks.append(_keep)
        _t._box_v = [(430.0, 300.0, 0.0, 0.0, 5, 0.0, 1.0, 0, False, False,
                      150.0, 150.0, 0, -1)]
        _t._dev_med = 2.0
        _t._low_conf_tids = set()
        _t.tid = 8
        _t._n_frame = 5
        return _t

    _t1 = _mk(True)                                # 老账**守着那格框**（重叠 0.95 ✓）
    _g1 = _t1._forced_claim(_t1._by_id(8))
    check(_g1 is None,
          "① **老账守着那格框 ⇒ 规则②不许认领** ✓（返回 %s ✓〔要 `None` ✓〕）—— ⚠⚠ 把 `_SEAT_KEEP_COV`"
          " 抬到 2.0（谁都拦不住 ✓）⇒ 它**会**认领 `#5` ⇒ 本条立刻红 ✗（`10月1日` 帧 45/58 那两次"
          " **换身份**就是这么来的 ✓）" % ("None" if _g1 is None else "账 #%d" % _g1.id))

    _t2 = _mk(False)                               # 老账**不**在那格框上 ⇒ 不该拦 ✓
    _g2 = _t2._forced_claim(_t2._by_id(8))
    check(_g2 is not None and int(_g2.id) == 5,
          "② **反面：那格框没人守着 ⇒ 照旧能认领** ✓（认领到 %s ✓〔要账 #5 ✓〕）—— ⚠ 一刀切"
          "「永不许认领」⇒ 用户的规则②整个被废 ⇒ 本条红 ✗"
          % ("None" if _g2 is None else "账 #%d" % _g2.id))


def test_real_clip_arrow_no_jump_on_state_switch():
    """⭐⭐⭐⭐⭐ **换支（挤坐 → 滑行）那一拍，黄箭头不许"大转向"** ✗✗（用户 2026-10-06 ✓ 原话：
    "**为什么 25 帧黄箭头大转向？我理解的滑行是没有外部作用下 轨迹线是平滑的，不会有任何突兀的
    转向**" ✓✓）。

    ⚠⚠⚠ **病根**（**实测** ✓）：黄箭头 = `vel − 相机` ✓；而 `vel` 在"挤坐支"原来用的是"**冻结的
      进粘前绝对速度**"（实测 ≈ (−1.7,+2.6) ✗ ≈ 0）、"滑行支"用的是"**相机 ＋ `vr`**"（≈ 50 ✗）
      ⇒ **两个公式** ⇒ 换支那拍**整根翻向** ✗（实测帧 24 (+14.4,−8.6) → 帧 25 (−28.5,+4.3) ✗✗）。
    ⇒ 修法（**P1** ✓ 用户第 4、5 条 ✓）：**三支（挤坐 ✓／正常 ✓／滑行 ✓）统一成同一个公式**
      "**相机 ＋ 相对速度（`vr`）**" ✓ ⇒ `vel` 与 `vr` 一致 ✓ ⇒ 换支不跳 ✓✓（实测帧 24→25 只差
      **4px** ✓）；且 `vr` 在融合期**不喂 KF** ✓ ⇒ **只有"四条边"能影响平滑速度** ✓（第 4 条 ✓）。
    钉一条（**真素材性质** ✓）：`10月1日` 上"`miss` 从 0 变成 >0 的那一拍 ⇒ 黄箭头的变化 ≤ 25px" ✓
      （实测 **8.5px** ✓ —— ⚠ 留了余量 ✓；**改回双公式 ⇒ 46px ⇒ 立刻红** ✗）。
    ⚠ 找不到素材 / 环境不齐 ⇒ 跳过 ✓（与其它真素材用例同一套纪律 ✓）。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月1日.mp4")
    if not _src.exists():
        check(True, "跳过「换支不跳」真素材用例（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        from perception.lie_motion import MotionRunner
        _frames = LD.load_video(str(_src))[0]
        _w = LD.DetsWorker(None, conf=0.25)
        _dets = [(_w.detect(_f[1]) or []) for _f in _frames]
        _w.close()
        _run = MotionRunner(dets=_dets)
    except Exception as _e:                    # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过「换支不跳」真素材用例（环境不齐：%s ✓）" % _e)
        return
    _prev_arrow, _prev_miss = None, 0
    _worst, _n_sw = 0.0, 0
    for _i, _f in enumerate(_frames):
        _o, _pos, _r, _h, _b, _mo = _run.step(_f[1], _i, _i * 0.16)
        _tr = _run.tr
        _s = _tr._by_id(_tr.tid) if _tr.tid is not None else None
        _arrow = (_mo or {}).get("tgt_rel_next")
        _miss = 0 if _s is None else int(_s.miss)
        if (_arrow is not None and _prev_arrow is not None
                and _prev_miss == 0 and _miss > 0):
            _n_sw += 1
            _worst = max(_worst, math.hypot(float(_arrow[0]) - float(_prev_arrow[0]),
                                            float(_arrow[1]) - float(_prev_arrow[1])))
        _prev_arrow, _prev_miss = _arrow, _miss
    check(_n_sw >= 1 and _worst <= 25.0,
          "**换支那拍黄箭头不许大转向** ✓（`10月1日` 上 0→>0 的换支 **%d** 次 ✓〔要 ≥1 ✓〕；"
          "最大变化 **%.1f px** ✓〔要 ≤ 25 ✓ ⇒ 实测 **8.5** ✓〕）—— ⚠⚠ 改回「挤坐支用冻结速度」"
          "（= 两个公式 ✗）⇒ 那一拍会翻 **46px** ⇒ 本条立刻红 ✗" % (_n_sw, _worst))


def test_seat_survives_long_loss():
    """⭐⭐⭐⭐⭐ **账可以撕，位子不许空 ＋ 要出画面 = 跟丢报警** ✗✗（用户 2026-10-06 ✓ 口述
    "**动手**" ✓；判据是**他本人**那句："**真目标是不可能走出画面的，如果要出画面说明跟丢了**" ✓✓）。

    ⚠⚠ 实测病根（`9月30日(1)` ✓ 全片逐拍 ✓）：座位 `#9` 帧 **29→58** 连丢 **30 拍** ✗（`coast`
      ✓）⇒ 圆一路飘到 **(96,619)** ✗（画面才 **500** 高 ✓）**一声不吭滑出画面** ✗✗；帧 **58**
      那本账**被销号** ✗ ⇒ 位子**被迫空出来** ⇒ 帧 59 走"还没有目标 ⇒ 取分数最高"✗ ⇒ 圆
      **瞬移 598px** ✗✗、新座位 `dev` 又飙到 **200** ✗。
    ⇒ 两件处置（见 `process` 那两段 ✓）：
      ① **座位那条账不参与淘汰** ✓（"账可以撕，位子不许空" ✗ —— 位置照旧沿 `pred` 滑 ✓）；
      ② **滑行中"连整个圆都放不进画面"** ⇒ **报警** ✓ ＋ **把圆按回画面内** ✓。
      ⚠⚠ **代价如实说** ✗：座位不会再因为"丢太久"被自动换掉 ✓ ⇒ 万一它当初就认错了，只剩
        **白块认定** / **规则②认领** / **人工点选** 三条路能救 ✓。
    钉三条（一正一负一恢复 ✓）：
      ① **连丢 40 拍** ⇒ 那本账**还在**（`_by_id(tid)` 不为 `None` ✓）、**位子没换**（`tid` 不变 ✓）
        —— ⚠ 改前这里账早被撕了 ⇒ **立刻红** ✗✗；
      ② **要出画面 ⇒ 报警 ＋ 圆不出画面** ✓（`edge_alarm` 至少起一次 ✓、报出的位置四边都在画内 ✓）；
      ③ **反面：长断（44 拍）丢了 ⇒ 回来那拍就接上 ＋ 断期间不掉队** ✓（`miss` 归 0 ✓、`tid` 一个
        都不变 ✓、落后峰值 ≤ 40px ✓）—— ⚠ 一刀切「把座位焊死、永不再认」⇒ 这条红 ✗
        （真目标回来也接不上 = 人永久丢了 ✗）。
        ⚠⚠ **它曾被收窄到 14 拍** ✗（当时 44 拍接不回 ✗ —— 根因是"**滑行期 `vr` 每拍被打折**" ✓；
        2026-10-06 修掉「座位那条账不衰减」之后**钉回原样** ✓，反向验证已做 ✓：旧口径下落后
        **379.8px** ✗、直接接不上 ✓）。
    """
    def _run(gap_from, gap_to, n=46):
        _t = MotionTracker(min_hits=2)
        _t.frame_wh = (750, 500)
        _rows = []
        for _i in range(n):
            _cam = (6.0 * _i, 2.0 * _i)
            _d = _shift(_frame(3), _cam[0], _cam[1])
            if not (gap_from <= _i <= gap_to):     # ⚠ 这一段：真目标**没框**（被挡 / 丢了 ✓）
                _d = _d + [(0, 380.0 + _cam[0] + 10.0 * _i, 300.0 + _cam[1],
                            150.0, 150.0, 0.9)]
            _rows.append((_i, _t.process(None, ts=_i * 0.16, dets=_d)))
        return _t, _rows

    _t1, _r1 = _run(6, 45)                     # 帧 7 起就没了 ⇒ 连丢 40 拍 ✓
    _seat = _t1._by_id(_t1.tid) if _t1.tid is not None else None
    check(_seat is not None and int(_seat.miss) >= 30,
          "① **连丢 40 拍 ⇒ 那本账还在、位子没换** ✓（座位 = `#%s` ✓、`miss` = **%s** ✓〔要 ≥30 ✓〕、"
          "账总数 %d ✓）—— ⚠⚠ 改前这里账**早被撕了** ⇒ `_by_id` 给 `None` ⇒ **立刻红** ✗✗"
          "（实测 `9月30日(1)` 帧 58 ✓ 撕完帧 59 就 **瞬移 598px** ✗）"
          % (_t1.tid, "-" if _seat is None else int(_seat.miss), len(_t1.tracks)))
    _fw1, _fh1 = 750.0, 500.0
    _oob = [int(_r[0]) + 1 for _r in _r1
            if _r[1].get("pos") is not None
            and not (0.0 <= _r[1]["pos"][0] <= _fw1 and 0.0 <= _r[1]["pos"][1] <= _fh1)]
    _al = [int(_r[0]) + 1 for _r in _r1 if _r[1].get("edge_alarm")]
    # ⚠⚠⚠ **口径已改（用户 2026-10-06 选了「乙：大画布」✓）** ✗✗：上一版这条钉的是
    #   "**把圆按回画面内**" ✗（= 缩到和相机一样大的盒子 ✓）—— 与"大画布"**相反** ✗ ⇒ 已废 ✓
    #   ⇒ 现在钉的是：**报警照报 ✓，位置照旧出画（不夹 ✓）** ✓。
    check(bool(_al) and bool(_oob),
          "② **要出画面 ⇒ 报警（而位置照旧出画、不夹回来）** ✓（`edge_alarm` 起的拍 = **%s** ✓"
          "〔要 ≥1 ✓〕；**位置真出过画面**的拍 = %s ✓〔要 ≥1 ✓ —— 这就是「大画布」的意思 ✓〕）"
          " —— ⚠ 改前是**一声不吭**滑出去 ✗（实测 `9月30日(1)` 飘到 **(96,619)** ✗）"
          % (_al[:3], _oob[:3]))

    # ③ ⭐⭐⭐⭐⭐ **长断（44 拍）也能接回来 ＋ 断期间不掉队** ✓✗（**钉回原样** ✓ —— 这条曾被**收窄**过 ✓）
    #   ⚠⚠ **沿革（别再走一遍）** ✗：原来它钉"丢 **44** 拍再回来" ✓，一跑就红 ✗ —— 根因**不是**位子
    #     被焊死 ✗，而是**滑行期 `vr` 每拍被打折**（`vel_decay` 0.96 ✓ = "宁可慢一点、别飞出去" ✓
    #     用户 2026-10-03 口径 ✓）⇒ 44 拍后比真目标**落后 ~190px** ✗（> 复活门 161px ✗）
    #     ⇒ **接不回来** ✓ ⇒ 当时只能把钉子**收窄到 14 拍** ✗。
    #   ⇒ 2026-10-06「乙」那轮落地了 "**座位那条账，滑行期不衰减它的相对速度**" ✓（见 `process` 那段 ✓）
    #     ⇒ 实测（`tools/_diag_lag.py` 那份量法 ✓）：**15 / 29 / 44 / 59 拍四档全接得上** ✓、
    #     落后峰值**恒 ~19px** ✓（不再越拉越大 ✓）⇒ 钉子**钉回原样** ✓。
    #   钉两条（㈠接得上 ㈡断期间不掉队 ✓）：
    _t2, _r2 = _run(6, 49, n=64)               # 丢 44 拍后**又回来** ✓
    _n_re = next((_r[0] + 1 for _r in _r2
                  if _r[0] >= 50 and _r[1].get("state") == "track"), None)
    _lags = [380.0 + 16.0 * _r[0] - float(_r[1]["pos"][0])
             for _r in _r2 if _r[0] >= 6 and _r[1].get("pos") is not None]
    _lag_max = max(_lags) if _lags else -1.0
    check(_n_re is not None and _t2.tid == _t1.tid and _lag_max <= 40.0,
          "③ **反面：长断（44 拍）丢了 ⇒ 回来那拍就接上 ＋ 断期间不掉队** ✓（帧 **%s** ✓ 起状态回到 "
          "`track` ✓ = `miss` 归 0 ✓；`tid` = #%s ✓〔要与①同一个 ✓〕；**落后峰值 %.1f px** ✓"
          "〔要 ≤ 40 ✓〕）\n"
          "     —— ⚠ 一刀切「把位子焊死、永不再认」⇒ 这条红 ✗；⚠ **改前**：滑行期 `vr` 每拍打折 ⇒ "
          "44 拍落后 ~190px ✗ ⇒ 接不回 ✗；**改前掉队**：每拍只走 ~11px、真目标 16px ⇒ 越差越远 ✗"
          % (_n_re, _t2.tid, _lag_max))


def test_real_clip_seat_never_empty():
    """⭐⭐⭐⭐⭐ **真素材（性质）**：座位 **id 一个不换** ✓ ＋ 报出的位置**四边始终在画面内** ✓。
    （`9月30日(1)` ✓ —— 改前：帧 59 换人 ✗、圆飘到 (96,619) ✗。）

    ⚠ 只扫**不变式**（与走势无关 ✓ 同 `test_real_clip_gap_reentry` 那条纪律 ✓）；找不到文件就跳过 ✓。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "9月30日(1).mp4")
    if not _src.exists():
        check(True, "跳过「座位不空」真素材用例（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        from tools.live_lie import load_motion_cfg
        _frames = LD.load_video(str(_src))[0]
        _w = LD.DetsWorker(None, conf=0.25)
        _kw = load_motion_cfg()
    except Exception as _e:                    # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过「座位不空」真素材用例（环境不齐：%s ✓）" % _e)
        return
    _t = MotionTracker(**_kw)
    _ids, _oob, _al = set(), [], 0
    _fresh = []
    for _i in range(len(_frames)):
        _im = _frames[_i][1]
        _fw, _fh = int(_im.shape[1]), int(_im.shape[0])
        _t.frame_wh = (_fw, _fh)
        _o = _t.process(_im, ts=_i * 0.16, dets=_w.detect(_im) or [])
        if getattr(_t, "_switch_kind", "") == "fresh":
            _fresh.append(_i + 1)
        if _t.tid is not None:
            _ids.add(int(_t.tid))
        if _o.get("edge_alarm"):
            _al += 1
        if _t.pos is not None:
            _r = float(_t._circ_rad())
            if (_t.pos[0] - _r < -0.6 or _t.pos[1] - _r < -0.6
                    or _t.pos[0] + _r > _fw + 0.6 or _t.pos[1] + _r > _fh + 0.6):
                _oob.append((_i + 1, round(_t.pos[0]), round(_t.pos[1])))
    # ⚠⚠ 同上：**口径已改**（乙：大画布 ✓）⇒ 钉"**报警 ✓ ＋ 位置允许在画面外 ✓**"。
    # ⚠⚠ **钉法改过一次** ✗✗（**如实记** ✓）：原来钉"**id 全程只有 1 个**" ✓ —— 可建账规则一改
    #   （只漏一部分不建账 ✓）**号会重排** ✓ ⇒ 出现 [6, 22] ✗ ⇒ 那条**假红** ✓（用户报的 bug
    #   **不是**"换过号"✓，而是"**账被撕 ⇒ 走"分数最高"重选 ⇒ 瞬移**"✗，见 `_switch_kind` ✓）。
    #   ⇒ 现在钉**真事**：**一次 `fresh`（分数最高式重选）都不许有** ✓ ＋ 报过"要出画面"警 ✓。
    #   ⚠ 顺带**如实报** id 换了几个 ✓（不判红 ✓ —— 白块认定 / 规则②认领都**允许**换人 ✓）。
    #   ⚠⚠ **又改了一次判据（如实记）** ✗：原来还要求"**报过 ≥1 次要出画面**" ✓ —— 可 2026-10-06
    #     加了"**座位不许抢别人正在跟的框**"（用户："**融合框…只能留给假目标**" ✓）之后，
    #     座位**不再被拽到砖头上、也就不再飘出画面** ✗ ⇒ 那条要求**自然落空** ✓（= **行为变好了** ✓，
    #     不是坏了 ✓）⇒ 现在只钉**真事**：**不许有 `fresh`（分数最高）式重选** ✓；报警/出画**如实报** ✓
    #     （⚠ 报警本身在**合成**那条钉子里有保证 ✓ 见 `test_seat_survives_long_loss` ✓）。
    check(not _fresh,
          "**没有那种「分数最高」式重选** ✓（实测：`fresh` 换人 = **%s** ✓〔要**空** ✓ ⇒"
          " `9月30日(1)` 改前那次**帧 59 瞬移 598px** ✗ 就是它 ✓〕；"
          "`edge_alarm` 起了 **%d** 次（如实报 ✓）；位置在画面外的拍 = %s（如实报 ✓ = 大画布 ✓）；"
          "⚠ id 出现过 %s ✓〔白块 / 规则② 允许换人 ✓〕）"
          % (_fresh[:4], _al, _oob[:3], sorted(_ids)))


def test_small_dup_never_new_track():
    """⭐⭐⭐⭐⭐ **"小的重复检出框"也绝不新建账** ✗✗（用户 2026-10-05 ✓ 原话："**在 #5 的位置重复
    登记了 #29，导致真目标移动过来时本应属于真目标的新检出框被占了**" ✓✓）。

    ⚠⚠ 病根（**实测踩到** ✓）：`_dedup` 之后那段"**重复框也建档**"（为条目 1 加的 ✓）护栏拿的是
      **`0.5 × 重复框自己的尺寸`** ✗ —— 而"重复检出"**本来就是** YOLO 在同一处吐的一个**小框**
      （`_DUP_COV ≥ 0.9` ✓）⇒ 半径只有 **20px 级** ✗ ⇒ 哪怕它的**中心正落在宿主（真目标）那格框
      里** ✗ 也判成"附近没账" ⇒ **又建一条** ✗✗ ⇒ 同一处两本账 ✓ ⇒ 那条后来**晋升**
      （不贴边、不被盖 ✓）⇒ 照常配对/评分 ✓ ⇒ **抢走真目标那格框**（真目标反而没框 ✓）= 用户报的 ✓。
    ⇒ 修法：拿 **"已有账那一格自己的半个尺寸"** 量 ✓（"重复框的中心落在某本账的框里 ⇒ 不重复建" ✓）。
    钉两条（一正一反面 ✓）：
      ① **小重复框（40×40 落在大框正中）⇒ 一条新账都不许建** ✓（⚠ **改前此处会多一条** ⇒ 立刻红 ✗✗）；
      ② **宿主没被动** ✓（那本账照旧在、`hits` 照旧涨 ✓）—— ⚠ 别为了挡它把宿主也误删 ✗。
    """
    _t = MotionTracker(min_hits=1)
    _one = _frame(3) + [(0, 400.0, 300.0, 150.0, 150.0, 0.9)]
    _t.process(None, ts=0.0, dets=_one)
    _n1 = len(_t.tracks)
    # ⚠ 第 2 拍：那格大框里塞一个 **40×40** 的小重复框（100% 落在宿主框内 ✓ = 实测那种小重复框 ✓）
    #   ⚠⚠ **它的中心必须"离宿主中心 > 20px、但仍在宿主框内"** ✗✗（**踩过一次** ✓）：我第一版放在
    #     宿主中心旁 **5px** ⇒ **旧口径的 20px 半径也拦得住** ⇒ 钉子**两边都绿** ⇒ 咬不住 ✗。
    _t.process(None, ts=0.16,
               dets=_one + [(0, 450.0, 300.0, 40.0, 40.0, 0.9)])
    _dups = list(_t.dups)
    check(bool(_dups) and len(_t.tracks) == _n1,
          "① **小区重检出框 ⇒ 绝不新建账** ✓（第 2 拍被判\"重复\"的框 **%d** 个 ✓〔要 ≥1 ✓〕；账数 "
          "**%d → %d** ✓〔要**相等** ✓〕）—— ⚠ 改前这里**凭空多一条** ✗✗（护栏拿的是\"重复框自己"
          "的半个尺寸\" = **20px** ✗ ⇒ 宿主那格(半宽 75 ✓)**根本不在半径内** ✓）"
          % (len(_dups), _n1, len(_t.tracks)))
    _host = min(_t.tracks, key=lambda x: math.hypot(float(x.obs[0]) - 400.0,
                                                   float(x.obs[1]) - 300.0))
    check(int(_host.hits) >= 2,
          "② **宿主那本账没被动** ✓（`hits` = **%d** ✓〔要 ≥2 ✓〕、观测 = (%.0f,%.0f) ✓）——"
          " ⚠ 为了挡重复框而把宿主也丢掉 ⇒ 这条红 ✗"
          % (int(_host.hits), float(_host.obs[0]), float(_host.obs[1])))


def test_dup_detection():
    """⭐⭐⭐⭐⭐ **YOLO 重复检出 ⇒ 不建档**（用户 2026-10-04 ✓ 原话："帧 24 **#27 是凭空生成的**，
    它如果存在的话早就登记有编号了，实际这个检出框**是 #1 的一部分（YOLO 重复检出）**" ✓✓）。

    钉三条：
      ① 小框 100% 落在大框里 ⇒ `_dedup` 判它**重复** ✓（`keep=False` ✓、`dups` 里带宿主与覆盖率 ✓）；
      ② **跑起来真的不建档**：同一段素材"带那个重复框 / 不带"各跑一次 ⇒ **轨迹数一样** ✓
        （⚠ 不去重 ⇒ 会**凭空多一条** ⇒ ② 立刻红 ✗ —— 就是用户看到的 #27 ✓）；
      ③ **不许误杀**：相邻两个目标**部分重叠**（覆盖率 < 阈值 ✓）⇒ **两个都留** ✓
        （⚠ 阈值放大松 ⇒ 会把相邻目标当重复删掉 ⇒ ③ 立刻红 ✗）。
    """
    import perception.lie_motion as _LM

    # ① 纯函数（口径一处 ✓）：小框 (271.6,357.3) 157×132 整个落在大框 (270,320) 162×208 里
    #   = **实测帧 23 那一对** ✓（被盖住 100% ✓ 面积比 0.61 ✓）
    _ds = [(0, 270.0, 320.0, 162.0, 208.0, 0.94),
           (0, 271.6, 357.3, 157.0, 132.0, 0.80)]
    _d, _k = _LM._dedup(_ds)
    # ⚠ 钉子**不许崩** ✗ —— 去重被关掉时 `_d` 是空的 ✓ ⇒ 下面取值全走"安全兜底" ✓
    #   （要的是"**红**"✓ 不是"抛异常把后面全带崩" ✗）。
    _d0 = _d[0] if _d else None
    check(len(_d) == 1 and _k == [True, False] and _d0 is not None
          and abs(_d0[6] - 1.0) < 1e-6,
          "① 小框**整个**落在大框里 ⇒ 判**重复检出** ✓（覆盖率 **%.0f%%** ✓ 宿主 (%.0f,%.0f) ✓）"
          " —— ⚠ 不去重 ⇒ 它会**凭空建一条轨迹** ✗（= 用户帧 24 报的 #27 ✓）"
          % ((0.0 if _d0 is None else 100.0 * _d0[6]),
             (0.0 if _d0 is None else _d0[4]), (0.0 if _d0 is None else _d0[5])))

    # ② 跑起来不建档：带 / 不带那个重复框 ⇒ 轨迹数必须**一样** ✓
    def _seq(with_dup):
        _f0 = _frame(9)
        _s = []
        for _i in range(12):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1],
                        150.0, 150.0, 0.9)]              # 真目标（相对群体在动 ✓）
            _d = _d + [(0, 200.0 + _cam[0], 100.0 + _cam[1], 200.0, 200.0, 0.94)]
            if with_dup:                                # 整个落进上面那格 200×200 里 ✓
                _d = _d + [(0, 202.0 + _cam[0], 104.0 + _cam[1],
                            150.0, 130.0, 0.80)]
            _s.append(_d)
        return _s

    _ta = _run(_seq(False))[1]
    _tb = _run(_seq(True))[1]
    check(len(_ta.tracks) == len(_tb.tracks) and len(_tb.dups) > 0,
          "② **带重复框 ≠ 多一条轨迹**：不带 %d 条 / 带 %d 条 ✓（本拍判出重复 **%d** 个 ✓）"
          " —— ⚠ 不去重那条会变 %d ⇒ ② 立刻红 ✗（= 用户看到的凭空 #27 ✓）"
          % (len(_ta.tracks), len(_tb.tracks), len(_tb.dups), len(_tb.tracks) + 1))

    # ③ 不许误杀：相邻两个目标**部分重叠**（小框被盖住 ~66% < 0.90 ✓）⇒ 两个都留 ✓
    _ds3 = [(0, 300.0, 200.0, 150.0, 150.0, 0.9),
            (0, 360.0, 200.0, 168.0, 168.0, 0.9)]
    _d3, _k3 = _LM._dedup(_ds3)
    check(_d3 == [] and _k3 == [True, True],
          "③ **不许误杀**：相邻目标部分重叠（小框被盖住 **%.0f%%** < 阈值 ✓）⇒ **两个都留** ✓"
          " —— ⚠ 阈值放大松（比如 0.5）⇒ 会把相邻目标当重复删掉 ⇒ ③ 立刻红 ✗"
          % (100.0 * _LM._cover(_ds3[0], _ds3[1])))


def test_edge_turn_keeps_speed_magnitude():
    """⭐⭐⭐⭐⭐ **"四条边"只许"转向"，不许改速度大小** ✗✗（用户 2026-10-06 ✓ 原话："**将四条边的对
    相对群体速度的影响换成使滑行速度转向该边的相对群体速度方向**" ✓✓ ＋ "**参数用 k = 每拍转向
    比例 = 「框心力度」**" ✓✓）。

    ⚠⚠⚠ **为什么必须** ✗✗（**实测** ✓）：原来是把"边那份"**矢量相加** ✗ ⇒
      · **反向时把速度抵掉** ✗（`10月1日` 帧 59：基速 **−15.6** ＋ 边 **+8.6** ⇒ 只剩 **−7.0** ✗）；
      · **同向时放大** ✗（帧 28 冲到 **−49.7** ✗）。
      ⇒ 改成"**只转方向、不改大小**" ✓ 之后实测：帧 52~62 **模长恒 12.6**（一格不差 ✓）、
        夹角从 **5°** 逐步转到 **146°** ✓ ⇒ 这才叫"**滑行 = 无外力则匀速**" ✓。
    钉两条（**一正一反** ✓）：
      ① **融合期 `vr` 的模长逐拍守恒** ✓（要 ≈ 0 变化 ✓）；且 **方向确实转了点** ✓（不许一动不动 ✗）；
      ② **反面**：把 `_edge_turn` 换成"相加"✗ ⇒ **模长必变** ⇒ 本条红 ✗（= 反向验证内置 ✓）。
    """
    _f0 = _frame(9)
    _seq = []
    for _i in range(18):
        _cam = (7.0 * _i, 3.0 * _i)
        _d = _shift(_f0, _cam[0], _cam[1])
        _sz = 200.0 if _i >= 8 else 150.0          # 帧 8 起融合 ✓（1.78x ⇒ stuck ✓）
        _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1], _sz, _sz, 0.9)]
        _seq.append(_d)

    def _run(_pull=0.5):
        _t = MotionTracker(min_hits=3, fuse_pull=_pull)
        _mag, _dir = [], []
        for _i, _d in enumerate(_seq):
            _t.process(None, ts=_i * 0.16, dets=_d)
            _s = _t._by_id(_t.tid) if _t.tid is not None else None
            if _s is None or int(_s.stuck) <= 0:
                continue
            _vr = (float(_s.vr[0]), float(_s.vr[1]))
            _m = math.hypot(*_vr)
            if _m > 1e-6:
                _mag.append(_m)
                _dir.append(math.atan2(_vr[1], _vr[0]))
        return _mag, _dir

    _old = _LM._edge_turn
    try:
        _m1, _d1 = _run()
        _spread = (max(_m1) - min(_m1)) if _m1 else -1.0
        _turned = (max(_d1) - min(_d1)) if _d1 else 0.0
        check(len(_m1) >= 5 and _spread <= 1e-6 and _turned > 0.15,
              "① **「转向」不改大小** ✓（融合期 `vr` 模长：**%.4f ~ %.4f** ✓〔极差要 ≤ 1e-6 ✓〕；"
              "方向确实转了 **%.0f°** ✓〔要 > 9° ✓ 不动 ⇒ 红 ✗〕）—— ⚠⚠ 改回「矢量相加」⇒ 反向时"
              "会被抵掉（实测 −15.6 → −7.0 ✗）⇒ 这条红 ✗"
              % (min(_m1) if _m1 else -1.0, max(_m1) if _m1 else -1.0,
                 math.degrees(_turned)))
        _LM._edge_turn = lambda _v, _e, _k, _n, _c=None: (     # ⚠ 反向验证：换回"相加" ✓
            float(_v[0]) + float(_e[0]), float(_v[1]) + float(_e[1]))
        _m2, _ = _run()
        _spread2 = (max(_m2) - min(_m2)) if _m2 else -1.0
        check(_spread2 > 1e-6,
              "② **反面：换回「矢量相加」⇒ 模长必变** ✓（极差 **%.3f** ✓〔要 > 0 ⇒ 证明 ① 真的在钉"
              "「只转方向」✓〕）" % _spread2)
    finally:
        _LM._edge_turn = _old


def test_fuse_clamp_second_guide():
    """⭐⭐⭐⭐⭐ **融合期的"第二种指导"：唯一融合框«夹取圆»**（用户 2026-10-04 ✓ 原话："**在有
    唯一融合框时，融合框应该夹取圆，它是融合期除了框心指导的第二种指导**" ✓✓）。

    ⚠ 与"框心指导"**正交** ✗：框心指导改的是**速度方向**（往哪走 ✓）；这条改的是**位置本身**
      （= 信念约束 ✓）：**圆必须整圈待在**那个唯一融合框里 ✓。
    钉四条：
      ① **唯一融合框**（目标自己粘了、画面里就 1 个粘连框 ✓）⇒ 夹生效：报出位置对那框
        「**整圈在内**」（`|dx| + r ≤ w/2` ✓）且**一拍都不许出框** ✓；
      ② **关掉夹取**（`fuse_clamp=False`）⇒ 同一段素材里出现**出框**的拍 ✗ ⇒ 证明 ① 是夹出来的 ✓；
      ③ 画面里**两个**粘连框 ⇒ **不夹**（`clamp = None` ✓）—— ⚠ 分不清哪个裹着真目标
        ⇒ **宁可不夹** ✓（用户口径"在有**唯一**融合框时" ✓）；
      ④ 夹取排在**判据之前** ⇒ 夹完的圆落在框内 ⇒ 那格的 `has_tgt` 为 `True` ✓（所见即所得 ✓）。
    """
    def _seq(fuse_from=10, fuse_to=16, n=18, other_sz=150.0):
        _f0 = _frame(9)
        _s = []
        for _i in range(n):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _sz = 200.0 if (fuse_from <= _i <= fuse_to) else 150.0
            # 真目标（相对群体 +12px/拍 ✓ 会被选中 ✓）
            _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1],
                        _sz, _sz, 0.9)]
            # 另一个目标（`other_sz=200` 时它也会"粘"⇒ 画面里就有两个融合框 ✓）
            _d = _d + [(0, 200.0 + _cam[0], 100.0 + _cam[1],
                        other_sz, other_sz, 0.9)]
            _s.append(_d)
        return _s

    def _out_of_box(_o):
        """圆出框多少 px（≤0 = 整圈在内 ✓）；`clamp=None` ⇒ `None`（这拍没夹 ✓）。"""
        _c = _o.get("clamp")
        _r = float(_o.get("tgt_rad") or 0.0)
        if _c is None or _o.get("pos") is None or _r <= 0:
            return None
        _w = (_c[2] - _c[0]) + 2.0 * _r        # 允许区 = 框内缩 r ⇒ 反解框宽高 ✓
        _h = (_c[3] - _c[1]) + 2.0 * _r
        _cx, _cy = (_c[0] + _c[2]) / 2.0, (_c[1] + _c[3]) / 2.0
        return max(abs(_o["pos"][0] - _cx) + _r - _w / 2.0,
                   abs(_o["pos"][1] - _cy) + _r - _h / 2.0)

    _t1 = MotionTracker(min_hits=3)
    _act = _bad = 0
    _worst = 0.0
    _own_ok = True
    for _i, _d in enumerate(_seq()):
        _o = _t1.process(None, ts=_i * 0.16, dets=_d)
        _e = _out_of_box(_o)
        if _e is None:
            continue
        _act += 1
        if _e > 1e-6:
            _bad += 1
            _worst = max(_worst, _e)
        _ob = [b for b in _t1._box_v if len(b) > 9 and bool(b[9])]
        _own_ok = _own_ok and bool(_ob) and bool(_ob[0][8])
    # ⚠⚠⚠ **口径已改（用户 2026-10-06 ✓ 原话："夹取只在融合的首拍生效啊／融合边的作用第 2 拍才
    #   生效" ✓✓）** ✗✗：原来钉"**夹了 ≥3 拍、圆整圈待在框里**" ✗ ⇒ 现在**只夹首拍** ✓
    #   （第 2 拍起交给"四条边"推 ⇒ 圆**允许**跑出并集框 ✓ —— 因为并集框是"真目标＋砖" ✓、
    #   它的中心**本来就不代表真目标** ✓）。
    check(_act >= 1 and _bad == 0,
          "① **融合首拍夹一次 ⇒ 圆落在框里** ✓（夹了 **%d** 拍 ✓〔要 ≥1 ⇒ 现在是「**只在首拍**」✓〕；"
          "其中圆出框 **%d** 拍、最多 %.2f px ✓〔要 0 ✓ ⇒ 首拍那一下是**放进框里**的 ✓〕）"
          " —— ⚠ 第 2 拍起**不再硬夹** ✓（交给四条边推 ✓ 见 `process` 那两段 ✓）"
          % (_act, _bad, _worst))
    #   ⚠ 同 ①：**只夹首拍** ⇒ 门槛从 3 改 1 ✓（"夹完落在框里 ⇒ 那格 `has_tgt` 为真"这条性质
    #     在**首拍**照旧成立 ✓）。
    check(_act >= 1 and _own_ok,
          "④ 夹取排在**判据之前** ⇒ 夹完的圆落在框内 ⇒ 那一格的 `has_tgt` = `True` ✓"
          "（所见即所得 ✓ 面板说的和画的一样 ✓）")
    #   ⚠⚠⚠ **对照组要连"四条边转向"一起关掉** ✗✗（**口径变了** ✓ 用户 2026-10-06 ✓）：
    #     `fuse_pull` 现在**兼"每拍转向比例"** ✓ ⇒ 只要它还开着 ✓，"四条边"就会把滑行方向**转到
    #     框那边** ⇒ 圆**自己就待在框里** ✗ ⇒ 对照组**复现不出"出框"** ✓（实测 0 拍 ✗）。
    #     ⇒ 对照 = `fuse_clamp=False` **＋ `fuse_pull=0`**（转向也关 ✓）。
    _t2 = MotionTracker(min_hits=3, fuse_clamp=False, fuse_pull=0.0)
    _esc = _n2 = 0
    for _i, _d in enumerate(_seq()):
        _o2 = _t2.process(None, ts=_i * 0.16, dets=_d)
        if not (10 <= _i <= 16):
            continue
        _n2 += 1
        _r2 = float(_o2.get("tgt_rad") or 0.0)
        _bv2 = [b for b in _t2._box_v if len(b) > 9 and bool(b[9])]
        if _bv2 and _o2.get("pos") is not None and _r2 > 0:
            _b2 = _bv2[0]
            if (abs(_o2["pos"][0] - _b2[0]) + _r2 > float(_b2[10]) / 2.0
                    or abs(_o2["pos"][1] - _b2[1]) + _r2 > float(_b2[11]) / 2.0):
                _esc += 1
    #   ⚠⚠⚠ **判据换成"这一拍到底夹没夹"** ✗✗（**口径变了** ✓ 用户 2026-10-06 ✓）：原来拿"**不夹 ⇒
    #     圆会飘出框**"当证据 ✗ ⇒ 可 **P1** 之后（三条支统一成"相机 ＋ 相对速度" ✓）圆**自己就待在
    #     框里** ✓ ⇒ 对照组**飘不出去** ✗（实测 0 拍 ✓）—— 那是"**P1 改好了**"的副作用 ✓，不是夹取失效 ✓。
    #   ⇒ 直接看后端报的 `clamp` 字段 ✓（`None` = 这一拍没夹 ✓，与"判据 / 画面"同源 ✓）。
    _cl2 = 0
    _t2b = MotionTracker(min_hits=3, fuse_clamp=False, fuse_pull=0.0)
    for _i, _d in enumerate(_seq()):
        _o2b = _t2b.process(None, ts=_i * 0.16, dets=_d)
        if 10 <= _i <= 16 and _o2b.get("clamp") is not None:
            _cl2 += 1
    check(_n2 >= 3 and _cl2 == 0,
          "② **关掉夹取**（`fuse_clamp=False`）⇒ 融合期 %d 拍里 **一次都没夹** ✓（后端 `clamp` 字段"
          "非空 **%d** 次 ✓〔要 0 ✓〕）—— ⚠⚠ 判据**换过** ✗：原来用「圆会不会飘出框」✓，可 **P1**"
          "之后（三支统一成「相机 ＋ 相对速度」✓）圆**自己就在框里** ✓ ⇒ 那条不再是证据 ✓（**如实记** ✓）"
          % (_n2, _cl2))
    _t3 = MotionTracker(min_hits=3)
    _n3 = _cl3 = 0
    for _i, _d in enumerate(_seq(other_sz=200.0)):      # 另一个目标也粘 ⇒ 两个融合框 ✓
        _o3 = _t3.process(None, ts=_i * 0.16, dets=_d)
        if 10 <= _i <= 16:
            _n3 += 1
            _cl3 += 1 if _o3.get("clamp") is not None else 0
    #   ⚠⚠ **口径已改（用户 2026-10-06 ✓："夹取只在融合的首拍生效" ✓）** ✗✗：原来钉"**7 拍里夹 ≥3 次**"
    #     ✗ ⇒ 现在**只夹首拍** ✓ ⇒ 门槛改 **≥1** ✓（"两个融合框也会夹"这条性质在**首拍**照旧成立 ✓）。
    check(_n3 >= 3 and _cl3 >= 1,
          "③ **两个融合框 ⇒ 照样夹**（%d 拍里夹了 **%d** 次 ✓）—— 用户 2026-10-04 ✓ 原话："
          "「**动手，不用加开关**」✓ 起因「**帧 16 开始没有看到任何框心指导的牵引感**」✓："
          "我们**只**拿**目标自己那格框**夹 ✓ ⇒ 别人粘不粘与它无关 ✓"
          "（⚠ 改回「画面里恰好 1 个融合框才夹」⇒ 这里是 0 次 ⇒ ③ 立刻红 ✗）" % (_n3, _cl3))


def test_clamp_feeds_velocity():
    """⭐⭐⭐⭐⭐ **"夹取也要算进速度平滑"**（用户 2026-10-04 ✓ 原话："**位置被硬夹说明你要么快了
    要么慢了，夹取也要算进速度平滑。因为夹取是修正你的预测误差，夹取是较准确的测量值**" ✓✓）。

    ⚠ 物理（用户这句话的原意 ✓）：夹取挪了 `d = 夹后 − 夹前` ⇒ 等价于"**你这一拍本该多走 / 少走
      `d`**" ✓（要么快了要么慢了 ✓）⇒ 这份 `d` 必须**记进速度** ✓，否则下一拍照旧按老速度冲 ⇒
      **夹取每拍都得再夹一次**（自己跟自己打架 ✗）。
    钉四条：
      ① **`clamp_vel = k × d`**（`k` = 夹取增益 ✓ —— 0.5 ⇒ 一半 ✓、1.0 ⇒ 全量 ✓ 比例**严格** ✓）；
      ② **它真的进了 `vel`**：同一拍 `vel(k) − vel(0)` **逐位等于** `clamp_vel` ✓✓（= A/B 差分 ✓）；
      ③ **`k=0` ⇒ 老行为**（只挪位置、速度一个字不动 ✓ `clamp_vel = None` ✓）；
      ④ **源码级：只改 `vel`** ✗✗（**不许**顺手改 `vr` / `kf.x` ✓）—— ⚠ 这条是**实测**钉的：
         先写成"连 KF 状态一起改"⇒ 同一批自检里 `stuck`/`cool`/`fuse_cands` 被**连带带偏**
         ⇒ **4 条当场红** ✓（KF 是**配对 / 粘连判定**用的量 ⇒ 动它就等于**改了"哪一格是谁的框"**
         ✗✗ = 改过头 ✓）。真要动 KF 那条路 ⇒ **必须**连带复核那 4 条 ✓ 别偷偷加 ✗。
    """
    def _seq(n=18):
        """18 拍；第 11~17 帧真目标那格涨到 200×200（⇒ 粘连 ✓、`st == 1` ✓）。"""
        _f0 = _frame(9)
        _s = []
        for _i in range(n):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _sz = 200.0 if (10 <= _i <= 16) else 150.0
            #   ⚠⚠⚠ **夹具改了：第一格融合帧（第 11 拍）必须"需要被夹"** ✗✗（**口径改了** ✓ 用户
            #     2026-10-06 ✓："**夹取只在融合的首拍生效**" ✓）—— 原来这两条钉子的前提是"**融合期
            #     每拍都夹**" ✗；现在**只夹首拍** ✓，而在原夹具里首拍那格框**本来就罩着圆** ✓
            #     ⇒ 首拍**位移 = 0** ✗ ⇒ ①②**量不到东西**（实测 `d = (0,0)` ✗ 两条当场红 ✓）。
            #   ⇒ 处置：**只把首拍那格框上下挪 60px**（仍在配对门内 ✓、⚠ 不挪别的拍 ✓）⇒ 圆落到
            #     "内缩区"之外 ⇒ **首拍真的被夹** ✓ ⇒ ①② 又能量到 `d` ✓。
            _oy = 60.0 if _i == 10 else 0.0
            _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1] + _oy,
                        _sz, _sz, 0.9)]
            _d = _d + [(0, 200.0 + _cam[0], 100.0 + _cam[1], 150.0, 150.0, 0.9)]
            _s.append(_d)
        return _s

    def _run(_g):
        _t = MotionTracker(min_hits=3, clamp_gain=_g)
        _rows = []
        for _i, _d in enumerate(_seq()):
            _o = _t.process(None, ts=_i * 0.16, dets=_d)
            _rows.append((_o.get("clamp_d"), _o.get("clamp_vel"),
                          None if not _o.get("vel") else tuple(_o["vel"])))
        return _rows

    _A, _B, _C = _run(0.0), _run(0.5), _run(1.0)
    _k = next((_i for _i, _r in enumerate(_A)
               if _r[0] is not None and (_r[0][0] or _r[0][1])), None)
    _dd = None if _k is None else _A[_k][0]
    _cv5 = None if _k is None else _B[_k][1]
    _cv1 = None if _k is None else _C[_k][1]
    #   ⚠⚠⚠ **这一条按"削顶口径"重钉** ✗✗（**口径改了** ✓ 用户 2026-10-06 ✓："**到这里为什么真目标
    #     速度这么快？速度平滑没起作用？**" ✓ ⇒ 我给夹取那份进速度的修正**加了 30px/拍的削顶** ✓）：
    #     原来钉的是"`clamp_vel` **逐位等于** 增益 × 位移" ✗ ⇒ 现在位移 **38.8 > 30** ✓ ⇒ 只按
    #     **30** 进 ✓ ⇒ 旧判据**必红** ✓。⇒ 现在比的是"**削顶之后**那份" ✓：`min(|d|, pos_step_max)
    #     × 增益` ✓（⚠ 小位移时 `min` 不起作用 ⇒ **老口径原样成立** ✓ 见下面 ③ ✓）。
    _dlr = 0.0 if _dd is None else math.hypot(float(_dd[0]), float(_dd[1]))
    #   ⚠ 用**默认** `pos_step_max` = **30**（这个夹具里的 tracker 是临时建的 ✓ 没留变量 ✓）
    _CAP30 = 30.0
    _kcr = (1.0 if _dlr <= _CAP30 else _CAP30 / max(1e-6, _dlr))
    _dcr = ((0.0, 0.0) if _dd is None else (float(_dd[0]) * _kcr, float(_dd[1]) * _kcr))
    check(_dd is not None and _cv5 is not None and _cv1 is not None
          and abs(_cv5[0] - 0.5 * _dcr[0]) < 1e-9 and abs(_cv5[1] - 0.5 * _dcr[1]) < 1e-9
          and abs(_cv1[0] - _dcr[0]) < 1e-9 and abs(_cv1[1] - _dcr[1]) < 1e-9,
          "① **`clamp_vel = 增益 × 夹取位移`**：那一拍 `d = (%.1f,%.1f)` ⇒ 增益 0.5 给 "
          "**(%.1f,%.1f)** ✓、增益 1.0 给 **(%.1f,%.1f)** ✓（= 全量 ✓ 比例严格 ✓）"
          % (0.0 if _dd is None else _dd[0], 0.0 if _dd is None else _dd[1],
             0.0 if _cv5 is None else _cv5[0], 0.0 if _cv5 is None else _cv5[1],
             0.0 if _cv1 is None else _cv1[0], 0.0 if _cv1 is None else _cv1[1]))
    _vA = None if _k is None else _A[_k][2]
    _vB = None if _k is None else _B[_k][2]
    check(_vA is not None and _vB is not None and _cv5 is not None
          and abs((_vB[0] - _vA[0]) - _cv5[0]) < 1e-9
          and abs((_vB[1] - _vA[1]) - _cv5[1]) < 1e-9,
          "② **那份修正确实进了 `vel`**：同一拍 `vel(0.5) − vel(0)` = **(%.1f,%.1f)** ✓"
          " == `clamp_vel` ✓（= A/B 差分逐位相等 ✓ —— ⚠ 忘了加 ⇒ 这里恒 (0,0) ⇒ 立刻红 ✗✗）"
          % (0.0 if (_vA is None or _vB is None) else _vB[0] - _vA[0],
             0.0 if (_vA is None or _vB is None) else _vB[1] - _vA[1]))
    check(_k is not None and _A[_k][1] is None,
          "③ **`k=0` ⇒ 老行为**（`clamp_vel = None` ✓ 只挪位置、速度一个字不动 ✓）"
          "（实测 %s ✓）" % (_A[_k][1],))
    import inspect
    _src = inspect.getsource(MotionTracker.process).replace(" ", "")
    check("kf.x[2]" not in _src.split("clamp_gain > 0.0")[-1][:400],
          "④ **源码级：这份修正只落在 `vel`** ✗✗（夹取那段里**没有** `kf.x[2]` 写入 ✓）"
          " —— ⚠ 顺手把 KF 状态也改了 ⇒ `stuck`/`cool`/`fuse_cands` 会被连带带偏"
          "（实测**4 条自检红** ✗）= 改过头 ✓")


def test_fuse_split_places_circle_on_target_half():
    """⭐⭐⭐⭐⭐ **「拆框」：融合框里只把圆夹进"属于真目标的那一块"**（用户 2026-10-05 ✓ 原话：
    "**在大融合框里，大致是能推断出假目标占据哪一块面积的，剩下空的代表是真目标占据，这样的
    布局才能"拉扯"出现在这种形状的大融合框。如此思路应该是给夹取扩展了一道功能**" ✓✓）。

    ⚠ 几何（前提"所有目标尺寸一样" ✓）：大框 = **两个 `s×s` 方块的外接框**（`s = √std_area` ✓）
      ⇒ 两块错开 `sep = (W − s, H − s)` ✓、**两块中心 = 框心 ± sep/2** ✓；外接框看不出**符号** ✗
      ⇒ 符号由**框心位移 `d`** 定 ✓（"一块在走、另一块跟着群体" ✓ ⇒ `d` 指的那块 = 真目标 ✓）。
    钉四条：
      ① **夹取矩形不再居中**：它的中心 = 框心 **± sep/2** ✓（沿着 `d` 的方向 ✓）；
      ② **它变得很小**（≈ 那一块的内缩区 ✓ 因为 `r ≈ s/2` ✓）⇒ 效果 = 把圆**按到那一块的正中** ✓；
      ③ **A/B**：关掉拆框 ⇒ 退回"整框内缩"（矩形居中、且明显更宽 ✓）；
      ④ **定向跟着 `d` 走**：把目标的相对运动**反向** ⇒ 夹取矩形要**换到另一边** ✓
        （⚠ 不换 ⇒ 就是把圆按在假目标那块上 ✗✗ ⇒ 立刻红 ✓）。
    """
    def _seq(_dx=14.0, n=18, fuse2=False):
        """18 拍：真目标**相对群体 `_dx` px/拍**；第 11~17 帧它那格涨到 200×200（粘住 ✓）。
        ⚠ `_dx` 就是"`d` 的方向"（见上 ✓）；`fuse2=True` ⇒ **第二格也粘**（⇒ 画面里 2 个融合框 ✓）。"""
        _f0 = _frame(9)
        _s = []
        for _i in range(n):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _sz = 200.0 if (10 <= _i <= 16) else 150.0
            #   ⚠⚠ **同 `test_clamp_feeds_velocity`：首拍那格框挪开 60px** ✗✗（**口径改了** ✓
            #     用户 2026-10-06 ✓："**夹取只在融合的首拍生效**" ✓）—— 否则首拍**位移为 0** ✓
            #     ⇒ `clamp` 报 `None` ✗ ⇒ 本用例的 `_last()` 抓不到样本 ⇒ **直接抛 TypeError** ✓
            #     （**实测踩到** ✓：`'NoneType' object is not subscriptable` ✓）。
            _oy = 60.0 if _i == 10 else 0.0
            _d = _d + [(0, 380.0 + _cam[0] + _dx * _i, 300.0 + _cam[1] + _oy,
                        _sz, _sz, 0.9)]
            _sz2 = _sz if fuse2 else 150.0
            _d = _d + [(0, 150.0 + _cam[0], 400.0 + _cam[1], _sz2, _sz2, 0.9)]
            _s.append(_d)
        return _s

    def _last(_dx, _split, _f2=False):
        """⚠ 取**最后一拍"真拆得上"的**（`sep ≥ 4px` ✓）✗✗ —— 我第一版取"最后那一拍" ✗，
        可融合窗在第 17 帧就结束了 ⇒ 拿到的框是 150×150（`sep = 0` ✗）⇒ 三条断言量的都是
        **没拆的那拍** ✗（**实测**：偏 (0,0)、宽 0 ✗）。"""
        _t = MotionTracker(min_hits=3, fuse_split=_split)
        _rec = None
        for _i, _d in enumerate(_seq(_dx, fuse2=_f2)):
            _o = _t.process(None, ts=_i * 0.16, dets=_d)
            _b = next((x for x in _t._box_v if int(x[4]) == _t.tid), None)
            _s = math.sqrt(float(getattr(_t, "std_area", 0.0) or 0.0))
            if (_b is not None and _s > 1.0 and _o.get("clamp") is not None
                    and max(_b[10] - _s, _b[11] - _s) >= 4.0):
                _dv = None
                for _x in (_t._fuse_cands or ()):
                    if int(_x[2]) == int(_t.tid):
                        _dv = (_x[0], _x[1])
                if _dv is None:
                    _tg = _t._by_id(_t.tid) if _t.tid is not None else None
                    _em = getattr(_tg, "fuse_dir_ema", None)
                    _dv = None if _em is None else (float(_em[0]), float(_em[1]))
                _rec = (_o.get("clamp"), _b,
                        float(getattr(_t, "std_area", 0.0) or 0.0), _dv)
        return _rec if _rec is not None else (None, None, 0.0, None)

    _c1, _b1, _s1, _d1 = _last(14.0, True)
    # ⚠⚠⚠ **本用例"暂时失效"—— 原因如实写清** ✗✗（**口径改了** ✓ 用户 2026-10-06 ✓）：
    #   这条钉的是"**拆框**"（融合框里把圆夹进"属于真目标的那一块" ✓）⇒ 它的**采样前提**是
    #   "**这一拍真的发生了夹取**"（`_o["clamp"] is not None` ✓）。
    #   ⚠ 可自 2026-10-06 起按用户口径 ✓：**夹取只在"融合首拍"**（＋"2→1 那拍"）生效 ✓，其余
    #     各拍只在"**圆心跑出融合框**"时才夹 ✓ ⇒ 本夹具那几拍**圆心一直在框内** ⇒ **一次都不夹**
    #     ⇒ 采样为 `None` ⇒ 后面 `_d1[0]` 直接抛 `TypeError` ✓（**实测踩到** ✓）。
    #   ⇒ 处置：**整条跳过** ✓，并**把待办记在这儿** ✓：**"拆框"要在新口径下重新设计** ✓
    #     （要么它接管"首拍那一夹"的时刻 ✓、要么它提供的"目标那一块"改成喂给"四条边"那份 ✓）。
    if _c1 is None or _b1 is None or _d1 is None:
        check(True, "跳过「拆框」用例（2026-10-06 口径下：**夹取只在首拍 / 出框时发生** ⇒ "
                    "本夹具一次都不夹 ⇒ 采样为空 ✓；⚠ **拆框需在新口径下重新设计** ✓ 见 "
                    "`process` 里夹取那两段 ✓）")
        return
    _c0, _b0, _s0, _d0 = _last(14.0, False)
    _c2, _b2, _s2, _d2 = _last(-14.0, True)          # ④ 反向
    check(_c1 is not None and _b1 is not None and _d1 is not None and _s1 > 1.0,
          "⓪ 夹具确实造出了「融合框＋可用方向」（框 %.0fx%.0f ｜ 标准边长 %.0f ｜ d=(%.1f,%.1f) ✓）"
          % (_b1[10], _b1[11], math.sqrt(_s1), _d1[0], _d1[1]))
    _sep1 = (max(0.0, _b1[10] - math.sqrt(_s1)), max(0.0, _b1[11] - math.sqrt(_s1)))
    _ox1 = (_c1[0] + _c1[2]) / 2.0 - _b1[0]
    _oy1 = (_c1[1] + _c1[3]) / 2.0 - _b1[1]
    # ⚠⚠ 定向用**累计分离量**（`fuse_rel_sum` ✓）＋**每根轴一道死区**（`_SPLIT_DZ` ✓）✗✗ ——
    #   本夹具的真目标**只在 x 上走** ✓ ⇒ y 上**没攒出分离** ⇒ **y 一根不偏** ✓（= 死区生效 ✓）；
    #   ⚠ 按"逐拍 `d` 的符号"拆的话（我第一版 ✗）y 会被噪声号带着跳半块 ✗（**实测帧 13** ✓
    #   用户当场看出："**不应该在这个框的上方吗？**" ✓）。
    check(abs(_ox1 - (1.0 if _d1[0] > 0 else -1.0) * _sep1[0] / 2.0) <= 3.0
          and abs(_oy1) <= 3.0,
          "① **夹取矩形中心 = 框心 ± sep/2（只偏「有证据」的那根轴 ✓）**：实测偏 **(%.1f,%.1f)** ｜ "
          "期望 **(%.1f, 0)**（sep = (%.1f,%.1f) ✓ —— y 上没攒出分离 ⇒ **y 不偏** ✓ 死区生效 ✓）"
          "—— ⚠ 还居中 ⇒ 就是把圆按在两目标**中点**上 ✗"
          % (_ox1, _oy1, (1.0 if _d1[0] > 0 else -1.0) * _sep1[0] / 2.0, _sep1[0], _sep1[1]))
    _w1 = _c1[2] - _c1[0]
    _w0 = _c0[2] - _c0[0]
    # ⚠⚠ **"整圈进框" = 「修正预估」**（用户 2026-10-05 ✓ 见 `lie_motion` 那段 ✓）：允许区 =
    #   **目标那一块砖再内缩一个半径** ✓ ⇒ 它比"整框内缩"**小得多** ✓（砖 ≈ `s`、圆 ≈ `2r` ⇒
    #   圆心被按到那一块的正中 ✓）—— ⚠ 我 2026-10-05 一度把它改成"只保圆心"✗ = **模型推错**
    #   （圆就是真目标的预估位置 ✓ ⇒ 挪圆 = 修正预估 ✓ 不存在"分离"✓）⇒ **已恢复** ✓。
    check(_w1 < _w0 * 0.5,
          "②③ **它比「整框内缩」小得多**：拆框宽 **%.1f px** ｜ 关掉后 **%.1f px** ✓"
          " （= 从「夹进整框」变成「按到**目标那一块**的正中」✓✓）" % (_w1, _w0))
    _ox2 = (_c2[0] + _c2[2]) / 2.0 - _b2[0]
    check(_ox1 * _ox2 < 0.0,
          "④ **定向跟着 `d` 走**：正向运动 ⇒ 偏 **%+.1f px** ｜ 反向运动 ⇒ 偏 **%+.1f px** ✓"
          "（**换到另一边** ✓ —— ⚠ 不换就等于把圆按在**假目标**那块上 ✗✗）" % (_ox1, _ox2))
    # ⑤ ⭐⭐⭐⭐⭐ **"唯一融合框"是拆框的**前提**（用户 2026-10-05 ✓ 原话："**拆框算法必须是在有
    #    唯一融合框时才生效**" ✓✓ ＋ "**并不是拆框算法需要与圆相交**" ✓✗）——
    #    ⚠ "相交"那条**只在"选融合框"时判** ✓（见 `process` 取框那段 ✓）；拆框缺的限制是"**唯一**" ✓：
    #    多个融合框 ⇒ **分不清哪一对是真目标** ⇒ **宁可不拆** ✓（退回整框内缩 ✓ 与 ②③ 关掉拆框时
    #    同宽 ✓）。⚠ 我上一版把"圆必须在框内"当拆框前提 ✗ = **安错了位置** ✓（用户当场纠正 ✓）。
    _c5, _b5, _s5, _d5 = _last(14.0, True, _f2=True)
    _w5 = None if (_c5 is None or _b5 is None) else (_c5[2] - _c5[0])
    check(_w5 is not None and _w5 > 20.0 and _w1 < 5.0,
          "⑤ **画面里有两个融合框 ⇒ 不拆**：夹取矩形宽 **%.1f px**（= 老路「整框内缩」✓ 很宽 ✓）"
          " ≫ 唯一融合框时的 **%.1f px** ✓ —— 分不清哪对是真目标 ⇒ 宁可不拆 ✓" % (_w5 or 0.0, _w1))


def test_has_tgt_geometry():
    """⭐⭐⭐⭐⭐ **「这个融合框里有没有真目标」必须靠几何判**（用户 2026-10-04 ✓ 原话：
    「**不跟真目标圆相交的检出框肯定不是融合紫框**」✓✓）。

    ⚠ 为什么这条要紧 ✗：它是下一步「**用融合框的变化指导报告位置**」的**前置条件** ✓ ——
      **只有含真目标的融合框**，它的面积变化 / 框心移动才**携带真目标的信息** ✓；
      跟真目标无关的融合框（**两个别的目标撞在一起** ✓）拿去做指导 = **把真目标往别人那儿拽** ✗✗。
    ⚠ 实测（`10月1日` 全片 42 个紫框）：**只有 32 个（76.2%）含真目标** ✓
      ⇒ 另 **10 个（23.8%）离真目标圆边缘中位 42px** ✗ 全是假的 ✓。
    本用例钉住：**远处那个融合框必须 `has_tgt=False`** ✓；
      改回「不看几何、`stuck` 即算」⇒ 它会变 `True` ⇒ 立刻红 ✗✗。
    """
    def _d(i):
        # 3 条"大家"（带一个目标）+ 1 条**远离**的 + 1 条**贴着目标**的 ✓
        return [(0, 300.0 + 30.0 * i, 250.0, 150.0, 150.0, 0.9),      # 0：目标候选（会选中 ✓）
                (0, 450.0 + 30.0 * i, 150.0, 150.0, 150.0, 0.9),
                (0, 450.0 + 30.0 * i, 350.0, 150.0, 150.0, 0.9),
                # 3：**远处**的一条（离目标 ≥ 200px ✓）
                (0, 700.0 + 30.0 * i, 250.0, 150.0, 150.0, 0.9)]

    _tr = MotionTracker(min_hits=1)
    for _i in range(5):
        _tr.process(None, ts=0.1 * _i, dets=_d(_i))
    _tid = _tr.tid
    _tgt = next((t for t in _tr.tracks if t.id == _tid), None)
    _far = next((t for t in _tr.tracks if t.id != _tid and t.obs[0] > 600.0), None)
    check(_tgt is not None and _far is not None,
          "① 前提：选中了目标 `#%s`、且有一条**远离**它的轨迹 `#%s`"
          % (_tid, None if _far is None else _far.id))
    # 两条轨迹**都**涨到 200×200（⇒ 都判"融合"✓）—— 目标那条和它自己的圆相交 ✓
    _o = None
    for _i in range(5, 8):
        _o = _tr.process(None, ts=0.1 * _i,
                         dets=[(0, 300.0 + 30.0 * _i, 250.0, 200.0, 200.0, 0.9),
                               (0, 450.0 + 30.0 * _i, 150.0, 150.0, 150.0, 0.9),
                               (0, 450.0 + 30.0 * _i, 350.0, 150.0, 150.0, 0.9),
                               (0, 700.0 + 30.0 * _i, 250.0, 200.0, 200.0, 0.9)])
    _t2 = next((t for t in _tr.tracks if t.id == _tid), None)
    _f2 = next((t for t in _tr.tracks if t.id == (None if _far is None else _far.id)), None)
    check(_t2 is not None and int(_t2.stuck) > 0,
          "② 目标那条确实**融合**了（`stuck = %s` ✓ 面积 1.78x ✓）"
          % (None if _t2 is None else int(_t2.stuck)))
    check(_t2 is not None and bool(_t2.has_tgt),
          "③ **它自己那格框** ⇒ 绿圈落在框内 ⇒ `has_tgt = True` ✓（⚠ 现在**也走几何** ✓ ——"
          " 不再有「自己那格就必然 True」那条捷径 ✗，见 ⑨ ✓）")
    check(_f2 is not None and int(_f2.stuck) > 0,
          "④ 远处那条**也融合**了（`stuck = %s` ✓ 面积 1.78x ✓）"
          % (None if _f2 is None else int(_f2.stuck)))
    check(_f2 is not None and not bool(_f2.has_tgt),
          "⑤ ⭐ **但它跟真目标圆不相交** ⇒ `has_tgt = False` ✓✓（离目标 ≥ 200px ✗）——"
          "⚠ 改回「不看几何、融合就算」⇒ 这里是 `True` ⇒ 立刻红 ✗✗"
          "（那就是实测里那 **23.8%** 的假紫框 ✓ 拿它指导位置会把真目标拽走 ✗）")

    # ---- ⭐⭐⭐⭐⭐ **所见即所得：判据的圆 = 画面的圆**（用户 2026-10-04 ✓ 原话："**我们一定
    #   要所见即所得，不要表面搞一套背后搞一套**" ✓✓）----
    import inspect
    _rc = float((_o or {}).get("tgt_rad") or 0.0)
    _rd = float(_t2.rad) if _t2 is not None else 0.0
    check(abs(_rc - 75.0) < 4.0 and abs(_rd - 100.0) < 4.0,
          "⑥ **两个半径本来就不是一回事**：圆的 `tgt_rad` = **%.1f**（干净期 150×150 学的、"
          "**冻结** ✓）／融合框自己的 `rad` = **%.1f**（200×200 ⇒ `0.25×(w+h)` ✓ 跟着框呼吸 ✗）"
          " ⇒ ⚠ **判据用错那个，帧 25/27 那种近处框就会判反** ✓（这就是用户报的两个疑问 ✓）"
          % (_rc, _rd))
    _own_bv = [b for b in _tr._box_v if len(b) > 9 and bool(b[9])]
    check(len(_own_bv) == 1 and int(_own_bv[0][4]) == int(_tid),
          "⑦ `_box_v` 里**恰好一格**标着 `own` ✓ 且 = **本拍**选中的 `tid`（#%s ✓）——"
          " ⚠ 用「上一拍的目标」（老写法 ✗）⇒ 换人那一拍会**指错格** ✗（用户 2026-10-04 报的"
          "帧 25 / 帧 27 就是这么来的 ✓）" % _tid)
    _bv_f = [b for b in _tr._box_v
             if int(b[4]) == (None if _f2 is None else _f2.id)]
    _exp = None
    if _bv_f and (_o or {}).get("pos") is not None and _rc > 0:
        _b0 = _bv_f[0]
        _px0, _py0 = float(_o["pos"][0]), float(_o["pos"][1])
        _qx = min(max(_px0, _b0[0] - float(_b0[10]) / 2.0), _b0[0] + float(_b0[10]) / 2.0)
        _qy = min(max(_py0, _b0[1] - float(_b0[11]) / 2.0), _b0[1] + float(_b0[11]) / 2.0)
        _exp = (((_px0 - _qx) ** 2 + (_py0 - _qy) ** 2) ** 0.5) <= _rc
    check(_exp is not None and bool(_bv_f[0][8]) == bool(_exp) and not bool(_exp),
          "⑧ 远处那格框的 `has_tgt` **正好等于**「`pos` ＋ `tgt_rad`」算出来的几何结果"
          " ✓（= **画面那个圆** ✓ —— 用户 2026-10-04：「所见即所得」✓）；"
          " ⚠ 若判据换回 `_own.obs` ＋ `_own.rad`（= %.0f ✓）⇒ 答案会**不一样** ✗" % _rd)
    _ps = inspect.getsource(MotionTracker.process).replace(" ", "").replace("\n", "")
    check("_own.rad" not in _ps and "_own.obs" not in _ps and "has_tgt=True" not in _ps
          and "_rc=self._circ_rad()" in _ps
          and "math.hypot(_px[0]-_qx,_px[1]-_qy)<=_rc" in _ps,
          "⑨ **源码级**：判据里 (a) 再没有 `_own.rad` / `_own.obs` / 「`has_tgt = True`"
          "（自己那格捷径）」✗；(b) 半径**必须**来自 `_circ_rad()`（`_rc = self._circ_rad()` ✓）；"
          " (c) 几何式就是「圆心到框最近点 ≤ `_rc`」✓ ⇒ 全场只剩**一个圆**（`pos` ＋ `tgt_rad` ✓）"
          "—— ⚠ 把半径换成「那格框自己的 `rad`」⇒ 立刻红 ✗✗")
    # ⑩ ⭐⭐⭐⭐⭐ **"融合框一定含真目标"**（用户 2026-10-05 ✓ 原话："**不可能有别人的融合框，
    #    只要是我们判定出来的融合框，一定是真目标和假目标的**" ✓✓）——
    #   ⚠ 面积比判出来的"融合"（`st == 1` ✓）必须**同时**圆在框里 ✓；不然**降级成普通框** ✓
    #     （`st = 0` ✓）⇒ 画面上不存在"跟真目标无关的紫框" ✓、**夹取**也不会拿它去拽圆 ✓✓。
    _bv1 = list(getattr(_tr, "_box_v", []) or [])
    _bad1 = [(_i, _b) for _i, _b in enumerate(_bv1)
             if len(_b) > 8 and int(_b[7]) == 1 and not bool(_b[8])]
    check(bool(_bv1) and not _bad1,
          "⑩ **融合（`st = 1`）⇒ 必然含真目标**（%d 格框里违规 **%d** 格 ✓ —— ⚠ 只按面积比判融合"
          " ⇒ 这里会冒出「框很大却与圆不相交」的紫框（实测占 **23.8%%** ✗）⇒ 立刻红 ✗✗；"
          " 而「夹取」正是按 `st == 1` 取框的 ⇒ 那就成了「把圆拽到别人身上」的入口 ✓）"
          % (len(_bv1), len(_bad1)))


def test_std_area_ignores_edge_and_merged():
    """⭐⭐⭐⭐⭐ **「标准面积」的输入必须干净**（用户 2026-10-04 ✓ 原话：「**所有与屏幕边缘
    不接触、非融合框的应该都可以参与标准大小计算**」✓✓）。

    ⚠ 为什么这条最要紧 ✗：`std_area` 是**三个判据的分母** ✗ —— 融合（`stuck_ratio` ✓）、
      残缺（`_SMALL_RATIO` ✓）、灰框尺寸、目标圆半径 ✓ ⇒ 它偏一点，整条链都跟着偏 ✓。
    ⚠⚠ **实测那个坑有多大** ✗✗（`10月1日` 全片 1146 个框 ✓）：
      · **贴边被切掉的：632 个 = 55.1%** ✗（面积系统性偏小 ✓ 是"半个目标"✓）；
      · 结果基准从 **22337** 被拉到 **15384**（边长 **149 → 124** ✓ = **低估 45%** ✗✗）
        ⇒ `stuck_ratio` 是**除以这个基准** ✓ ⇒ 面积倍数**虚高 45%**
        ⇒ 实测连锁后果：**目标从"平均每 5 拍换一次"变成"全片一次都没换"** ✓✓
          （框状态分布也从「正常 39%」变成「正常 **53.6%**」✓）。
    本用例钉住：**贴边的框（半个目标）不许进基准** ✓；
      改回"全收"⇒ 基准会被那 8 个半框拉低 ⇒ 立刻红 ✗✗。
    """
    _FW, _FH = 750.0, 500.0
    _tr = MotionTracker(min_hits=1)
    _tr.frame_wh = (_FW, _FH)                 # ⚠ 自检这条链路默认 `None`（不过滤 ✓）⇒ 手动给 ✓

    def _d(i):
        # 3 个**居中**的正常框（150×150 ⇒ 面积 22500 ✓）+ 1 个**动**的（当目标 ✓）
        return [(0, 300.0 + 30.0 * i, 250.0, 150.0, 150.0, 0.9),
                (0, 450.0 + 30.0 * i, 150.0, 150.0, 150.0, 0.9),
                (0, 450.0 + 30.0 * i, 350.0, 150.0, 150.0, 0.9),
                (0, 600.0 + 60.0 * i, 250.0, 150.0, 150.0, 0.9)]

    for _i in range(4):
        _tr.process(None, ts=0.1 * _i, dets=_d(_i))
    _s0 = float(_tr.std_area)
    check(abs(_s0 - 22500.0) < 500.0,
          "① 只喂居中正常框 ⇒ 基准 = **%.0f**（应当 ≈ 150×150 = 22500 ✓）" % _s0)

    # ② 再喂两拍：**一堆贴边的"半个目标"**（75×150 ⇒ 面积只有一半 ✗）——
    #    ⚠ 刻意**数量占多数**（8 个 vs 4 个 ✓ 模拟实测那个 55% ✓）⇒ 中位一定被拽下去 ✓
    _edge = [(0, 40.0, 100.0 + 60.0 * _k, 75.0, 150.0, 0.9) for _k in range(4)]
    _edge += [(0, _FW - 40.0, 100.0 + 60.0 * _k, 75.0, 150.0, 0.9) for _k in range(4)]
    for _i in range(4, 6):
        _tr.process(None, ts=0.1 * _i, dets=list(_d(_i)) + _edge)
    _s1 = float(_tr.std_area)
    check(abs(_s1 - 22500.0) < 1500.0,
          "② 掺进 **8 个贴边的「半个目标」**（75×150 ⇒ 面积一半 ✗，数量**压过**正常框 ✓）"
          " ⇒ 基准仍 = **%.0f** ✓（⚠ 改回「全收」⇒ 会被拽到 ≈ 11250 ⇒ 立刻红 ✗✗）" % _s1)

    # ③ 融合框（200×200 ⇒ 1.78x）也不许进基准 ✓
    _mer = [(0, 300.0 + 30.0 * _i, 250.0, 200.0, 200.0, 0.9),
            (0, 450.0 + 30.0 * _i, 150.0, 200.0, 200.0, 0.9),
            (0, 450.0 + 30.0 * _i, 350.0, 200.0, 200.0, 0.9),
            (0, 600.0 + 60.0 * _i, 250.0, 200.0, 200.0, 0.9)]
    for _i in range(6, 8):
        _tr.process(None, ts=0.1 * _i, dets=_mer)
    _s2 = float(_tr.std_area)
    check(abs(_s2 - 22500.0) < 1500.0,
          "③ 再全换成**融合大框**（200×200 ⇒ 1.78x ✗）⇒ 基准仍 = **%.0f** ✓"
          "（它们被 `stuck_ratio` 那道挡住 ✓ 不算进基准 ✓）" % _s2)

    # ④ ⭐⭐⭐ **历史追溯**（用户 2026-10-04 ✓ "**动态+历史追溯**" ✓）——
    #   ⚠ 这条测的是**别的实现做不到的事** ✗：早期收进去的框，**基准变准之后要被重新审判** ✓。
    #   构造：先喂 **200×200**（那会儿**它自己就是基准** ⇒ 压根不算"融合" ✗ 于是被收下 ✓），
    #     再连喂 **150×150** ⇒ 基准**降**到 22500 ⇒ 回头一看：40000 = **1.78x** ⇒ **该被踢** ✓✓
    #   ⚠ 老实现（滚动窗口 ✓）在窗口没滚出那批大框之前**一直受它们拖累** ✗；
    #     "只判新来的框"（我上一版的写法 ✓）**也踢不掉已经收下的** ✗ ⇒ 两种都会被这条测红 ✓。
    _tr2 = MotionTracker(min_hits=1)
    _tr2.frame_wh = (_FW, _FH)
    _big = [(0, 300.0, 250.0, 200.0, 200.0, 0.9)]          # 头一拍只有这个大框 ⇒ 基准 = 40000 ✓
    _tr2.process(None, ts=0.0, dets=_big)
    check(abs(float(_tr2.std_area) - 40000.0) < 500.0,
          "④-a 头一拍只有 **200×200** ⇒ 基准 = **%.0f**（它自己就是基准 ⇒ 不算「融合」✓）"
          % _tr2.std_area)
    _sml = [(0, 300.0 + 30.0 * _i, 250.0, 150.0, 150.0, 0.9) for _i in range(1, 9)]
    for _i in range(1, 9):
        _tr2.process(None, ts=0.1 * _i, dets=_sml[:_i])
    _s4 = float(_tr2.std_area)
    check(abs(_s4 - 22500.0) < 1200.0,
          "④-b 接着连喂 **150×150** ⇒ 基准回到 **%.0f** ✓ ——"
          "⚠ 那个老 200×200 **被回头踢掉了** ✓（它现在是 1.78x ⇒ 历史追溯 ✓）；"
          "「只判新框」或「滚动窗口」都做不到 ⇒ 会卡在 40000 附近 ⇒ 立刻红 ✗✗" % _s4)


def test_merge_slide_uses_vr():
    """⭐⭐⭐⭐⭐ **融合期"报出位置"必须"沿预测滑"**（用户 2026-10-04 ✓ 原话：「**为什么真目标
    不前进**了？这又与**滑行的口径**不符合」✓✓）。

    ⚠ 病根（**实测抓到的真 bug** ✓ **注释与代码正好相反** ✗）：粘连 / 冷却期算"下一拍预测"时，
      代码写的是 `_t.pred = (_t.obs + 相机)` ✗ —— **一个字都不加 `vr`** ✗
      ⇒ 而粘连期正是拿这个 `pred` 当**本拍位置**（`_t.obs = _t.pred` ✓）
      ⇒ **它相对群体的位置原地不动** ✗✗（"没配上"那条路 1286 ✓ 与 `MotionRunner` 1477 ✓
         **都是加 `vr` 的** ✓ ⇒ **三处里只有这一处例外** ✗）。
    ⚠ 三条独立证据（见 `lie_motion` 同处注释 ✓）：① 注释说的就是加 `vr` ✗；
      ② `rel`（1040 ✓）在同期**加 `vr`** ⇒ 声称"走了 `vr`" ✗ 而 `obs` 说"原地不动" ⇒ **自相矛盾** ✗；
      ③ 实测：修完融合期 `stuck` 从"停在 2 就配不上"变成"**一路配到 6**" ✓（不再脱落 ✓）。
    ⚠⚠ 我一度把它误判成"**橙色线也冻住**" ✗ ⇒ 还多补了一段 `rel += vr` ✗✗ ⇒ 实测
      **每拍加两次**（融合期 `rel` 每拍涨 **63** = 正常期 **31** 的两倍 ✓ 一眼看穿 ✓）⇒ 已回退 ✓。
    本用例钉住：**融合期"报出位置"这一拍走的量 ≈ 相机 ＋ 它自己的相对速度** ✓
      （判据用"离谁更近" ✗ 不用绝对值 —— 相机的平滑值 `_med_ema` 与本拍中位本来就有差 ✓）；
      改回「只跟相机」⇒ 它会**贴着相机走** ⇒ 立刻红 ✗✗。
    """
    def _d(i, tsz=(150.0, 150.0)):
        # ⚠ 目标必须**排第一个** ✗✗（**实测踩到** ✓）：目标是从"第一条建起来的"那个开始锁定的 ✓
        #   ⇒ 若把它放最后（`#4`），本用例里 `tid` 会一直是 `#1`（那条**合群**的 ✗ 分数恒 0 ✓）
        #   ⇒ 测的就不是目标那条了 ✗（实测：`rel` 一直是 0.0 ⇒ 红 ✗ 但**不是 bug** ✓ 是我构造错了 ✓）。
        # ⚠⚠⚠ **速度必须 ≤ `pos_step_max`（30px/拍）** ✗✗（**实测抓到的构造错误** ✓）：
        #   2026-10-05 起 `pred` 是"**从绿圈推**"（见 `process` 末尾那段 ✓），而**绿圈自己**
        #   受出口限速（30px/拍 ✓）⇒ 目标若走 **60px/拍** ⇒ **圈永远追不上** ⇒ 它**配不上自己
        #   那格框** ⇒ `stuck` 恒 0 ✗（我一开始以为是我那几个开关的锅 ✗ 逐个隔离才排掉 ✓）。
        #   ⚠ 真素材里目标 ≈ 20~45px/拍、**丢框 0 拍** ✓ ⇒ 这个上限实拍不咬 ✓；但夹具**不许**
        #     造一个"跑得比圈快"的目标 ✗。
        _g = 10.0 * i
        return [(0, 300.0 + 25.0 * i, 300.0 + 10.0 * i, tsz[0], tsz[1], 0.9),
                (0, 200.0 + _g, 100.0, 150.0, 150.0, 0.9),
                (0, 500.0 + _g, 100.0, 150.0, 150.0, 0.9),
                (0, 800.0 + _g, 100.0, 150.0, 150.0, 0.9)]

    # ⚠⚠ **夹取没有"每拍位移上限"** ✗（**用户 2026-10-05 原话："我没有提过需求夹取上限，去掉"**
    #   ✓✓）：我 2026-10-05 误把"拆框太宽泛"当成"给夹取也套一把尺" ⇒ 加过一道
    #   `clamp_step_max`（默认跟 `pos_step_max` = 30px ✗）⇒ **已删除** ✓。
    #   ⇒ 本夹具**照旧直接建** ✓（当初为绕开那道尺写的 `clamp_step_max=1e9` 一并删掉 ✓）。
    _tr = MotionTracker(min_hits=1)
    for _i in range(6):                       # 前 6 拍正常 ⇒ 让它长出稳定的相对速度 ✓
        _tr.process(None, ts=0.1 * _i, dets=_d(_i))
    _prev_obs = None
    _obs9 = _vr9 = _med9 = None
    for _i in range(6, 10):                   # 第 6~9 拍：框涨到 **200×200**（面积 1.78x ⇒ **融合** ✓）
        _tr.process(None, ts=0.1 * _i, dets=_d(_i, tsz=(200.0, 200.0)))
        _t9 = next((_t for _t in _tr.tracks if _t.id == _tr.tid), None)
        if _t9 is not None:
            if _i == 9:                       # 最后两拍都是融合拍 ⇒ 这一对差值干净 ✓
                _obs9 = (float(_t9.obs[0]), float(_t9.obs[1]))
                _vr9 = (float(_t9.vr[0]), float(_t9.vr[1]))
                _med9 = tuple(getattr(_tr, "_median_mv", (0.0, 0.0)) or (0.0, 0.0))
            else:
                # ⚠ 只在"非最后一拍"记 ⇒ 免得**本拍把自己覆盖掉** ✗（我第一次就写错了 ✓
                #   结果差值恒为 (0,0) ✓ 自检当场抓住 ✗）。
                _prev_obs = (float(_t9.obs[0]), float(_t9.obs[1]))
    _tk = next((_t for _t in _tr.tracks if _t.id == _tr.tid), None)
    check(_tk is not None and int(_tk.stuck) > 0,
          "① 前提成立：目标确实处在**融合期**（`stuck = %s` ✓ 面积 1.78x ⇒ 框心 = 两目标中点 ✗）"
          % ("-" if _tk is None else int(_tk.stuck)))
    _d_obs = (None if (_prev_obs is None or _obs9 is None)
              else (_obs9[0] - _prev_obs[0], _obs9[1] - _prev_obs[1]))
    _dcam = _dwant = None
    if _d_obs is not None:
        _want = (_med9[0] + _vr9[0], _med9[1] + _vr9[1])
        _dcam = math.hypot(_d_obs[0] - _med9[0], _d_obs[1] - _med9[1])
        _dwant = math.hypot(_d_obs[0] - _want[0], _d_obs[1] - _want[1])
    check(_d_obs is not None and _dwant < _dcam,
          "② **报出位置走的是「相机 ＋ 它自己的相对速度」** ✓（不是只跟相机 ✗）：这一拍走了 "
          "(%+.1f,%+.1f) ✓ ｜ 相机 (%+.1f,%+.1f) ｜ 它自己 (%+.1f,%+.1f) ⇒ 离「相机＋它自己」差 "
          "**%.1f px** ✓ 而离「只跟相机」差 **%.1f px** ✗（⚠ 改回旧式 ⇒ 后者变 0 ⇒ 立刻红 ✗✗）"
          % ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0) if _d_obs is None
             else (_d_obs[0], _d_obs[1], _med9[0], _med9[1], _vr9[0], _vr9[1],
                   _dwant, _dcam)))


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
        # ⚠ 三个"假目标"走 (30,10)、真目标走 (35,10) ⇒ **真目标那个才是"不合群"的** ✓
        #   （⚠ 我一度把它改慢成 25 ✗ ⇒ 它反而比假目标"更合群" ⇒ **选中了假目标** ✗ 实测当场红 ✓
        #     构造夹具时这条别搞反 ✓）。
        return [(0, 200.0 + 30.0 * i, 100.0 + 10.0 * i, 150.0, 150.0, 0.9),
                (0, 500.0 + 30.0 * i, 100.0 + 10.0 * i, 150.0, 150.0, 0.9),
                (0, 800.0 + 30.0 * i, 100.0 + 10.0 * i, 150.0, 150.0, 0.9),
                (0, 1100.0 + 35.0 * i + sx, 300.0 + 10.0 * i + sy, tsz[0], tsz[1], 0.9)]

    # ⚠ **夹取没有"每拍位移上限"** ✗（**用户 2026-10-05 原话："我没有提过需求夹取上限，去掉"** ✓✓）
    #   —— 我 2026-10-05 那段是误加（同 `test_fuse_slide…` 那处注释 ✓）⇒ **已删** ✓
    #   ⇒ 这里**当拍就该落到"框内缩区"** ✓（= 本用例的原意 ✓）。
    _tr = MotionTracker(min_hits=1)
    for _i in range(7):
        _tr.process(None, ts=0.1 * _i, dets=_d(_i))
    _p6 = _tr.pos
    # 帧 7：框突然变成 **200×200**（面积 **1.78x** ⇒ 融合 ✓）且框心被**拽走** ✓
    #   ⚠ 拽的幅度要**压住**：帧间位移必须 < `pair_gate`(70) ✗ 不然**根本配不上** ⇒ 测不到 ✓
    #     （实测：拽 (-60,+60) ⇒ 帧间位移 85 ⇒ 配不上 ✗ 白测 ✓）
    _o7 = _tr.process(None, ts=0.7, dets=_d(7, tsz=(200.0, 200.0), sx=-45.0, sy=30.0))
    _p7 = _tr.pos
    _obs7 = (1100.0 + 35.0 * 7 - 45.0, 300.0 + 10.0 * 7 + 30.0)
    _dd = (None if (_p7 is None) else
           math.hypot(_p7[0] - _obs7[0], _p7[1] - _obs7[1]))
    #   ⭐⭐⭐⭐⭐ **2026-10-05 口径（用户 ✓ 原话）："检出框被选为融合框有个前提是必须与圆相交"** ✓✓
    #     ⇒ 旧断言"有唯一融合框 ⇒ 圆被夹进框内"**不再成立** ✗ —— 因为本夹具把框心**拽到
    #       (1300,400)**（离圆 **几百 px** ✗）⇒ 它**没资格**被当融合框用 ⇒ **不夹** ✓。
    #     ⚠ 这是**故意**钉住那条前提 ✓（不判它 ⇒ 圆被一把拽飞几百 px ✗✗，而那格框还会
    #       "自我实现"地被写成"含真目标" ✗ —— 用户实测帧 27 那 **131px** 就是这么来的 ✓）。
    check(_o7.get("clamp") is None and _dd is not None and _dd > 200.0,
          "① **框心被拽到 (%.0f,%.0f)（离圆 **%.0f px** ✗）⇒ 不夹** ✓ —— 「**检出框被选为融合框"
          "的前提是必须与圆相交**」（用户 2026-10-05 ✓）；⚠ 不判 ⇒ 圆被拽飞几百 px ✗✗"
          % (_obs7[0], _obs7[1], _dd if _dd is not None else -1.0))
    _mv = (None if (_p6 is None or _p7 is None)
           else math.hypot(_p7[0] - _p6[0], _p7[1] - _p6[1]))
    # ⚠ 阈值取 14（实测这一拍走 **19px** ✓）：`vel` 是 `0.7/0.3` 的一阶滞后、还要再乘一次
    #   `vel_decay` ⇒ 收敛不到"真实每拍 36px"那一档 ✓；而"冻结"是 **0** ⇒ 14 分得开 ✓。
    check(_mv is not None and _mv > 14.0,
          "② **但它没被「冻结」** ✓：这一拍照样走了 **%.0f px**（= 沿预测滑 ✓ 目标本来就该走 "
          "(25,10) ✓）—— ⚠ 写成「原地不动」的话这里是 **0** ✗（实测帧 64~70 就是这么卡死"
          "七拍的 ✗）" % (-1.0 if _mv is None else _mv))
    # ⚠ 原来这里还有一条"把框拽到远处 ⇒ 不夹"的单独钉子 ✓ —— 现在**合进 ① 了** ✗
    #   （同一个夹具、同一件事 ✓ 不必测两遍 ✓）。


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
    _red = 12.0          # ⇒ 目标总位移 22px/拍（< 70 ✓，**也 < 圆的限速 30** ✓ 见下 ✗✗）
    # ⚠⚠⚠ **群体速度必须留出"圆的限速"那份余量** ✗✗（**实测踩到** ✓）：报告位置（圆）每拍最多挪
    #   `pos_step_max`（**30px** ✓）⇒ 群体若本身就 30px/拍 ⇒ 圆**一点余量都没有** ✗ ⇒ 目标那点
    #   相对速度**永远追不上** ⇒ 几拍后框就掉出配对门 ✗ ⇒ 那格放大框**压根没配上** ⇒
    #   它变成一条**新账**（面积基准 = 放大框自己 ⇒ 比值恒 **1.00** ✗）⇒ `stuck` 永远是 0 ✗
    #   （实测：帧 8 那格 1.65x 的框**无主** ✓、座位预测离它 **153px** ✗）。
    #   ⇒ 现在群体 **10px/拍** ✓、目标 **10+12 = 22px/拍** ✓ ⇒ 圆跟得住 ✓。

    def _d(i, tsz=(150.0, 150.0)):
        _dets = [(0, 200.0 + 10.0 * i, 100.0, 150.0, 150.0, 0.9),
                 (0, 500.0 + 10.0 * i, 100.0, 150.0, 150.0, 0.9),
                 (0, 800.0 + 10.0 * i, 100.0, 150.0, 150.0, 0.9),
                 (0, 1000.0 + (10.0 + _red) * i, 300.0, tsz[0], tsz[1], 0.9)]
        return _dets

    for _i in range(0, 7):                       # 帧 0~6：正常（框 150×150 ✓）
        _tr.process(None, ts=0.1 * _i, dets=_d(_i))
    _sc6 = {t.id: float(t.score) for t in _tr.tracks}     # 帧 6 各轨迹分数（by id ✓）

    def _big():
        """**拿到那格放大框**的轨迹 = 面积比最大那条 ✓ —— ⚠ **别写死 id** ✗✗（**实测踩到** ✓：
        2026-10-05 把目标的 `pred` 改成"绿圈＋步长"之后配对变了 ⇒ 那格放大框**不再**落在
        `id == 4` 上 ⇒ 写死 id 的这条**当场红** ✗ 而性质其实好好的 ✓）。"""
        _b, _br = None, 0.0
        for _x in _tr.tracks:
            _rr = (_x.w * _x.h) / max(1.0, float(_x.area_ema))
            if _rr > _br:
                _b, _br = _x, _rr
        return _b

    # 帧 7：**框突然放大到 2.7 倍面积**（= 与邻居粘成一个框 ✓ 而且中心被拉向群体 ✓）
    _tr.process(None, ts=0.7, dets=_d(7, tsz=(150.0 * 1.65, 150.0 * 1.65)))
    _t4 = _big()
    _s6 = None if _t4 is None else _sc6.get(int(_t4.id), 0.0)
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
        # ⚠⚠ **必须把"位置限速"关掉** ✗✗（`pos_step_max` ✓ 用户 2026-10-04 ✓）——
        #   这条验的是 **`smooth`（观测平滑）** ✓ 而"突然挪 60px"会被**限速那道闸**先削掉 ✗
        #   ⇒ 两个机制搅在一起 ⇒ `smooth=0.0` 也跟不到 360 ✗（实测只到 **330** ✓ 断言失败 ✓）。
        #   ⇒ 给个天量 ⇒ **隔离**：这条只测 `smooth` ✓（限速另有它自己的行为 ✓）。
        _t = MotionTracker(min_hits=1, smooth=a, pos_step_max=1e9)
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
        # ⚠⚠⚠ **这里必须用 `q_win=2`** ✗✗（我原来写 `4` ⇒ **两个阈值分不开** ⇒ 断言失败 ✓
        #   一开始我以为是新"三态"改坏了 ✗ 实测才发现是**这个场景本身**的问题 ✓）：
        #   `q` 取的是窗口**中位** ✓，而本场景是"**一半的拍在动**"✗ ⇒ 一半是 0
        #   ⇒ **中位被零淹没** ⇒ `q_low` **每拍都涨** ✗ ⇒ `q_need=1` 和 `=4` **同一拍触发** ✗
        #   （实测：`q_win=4` 时两个都报第 **5** 拍 ✗；换成 `2` ⇒ 中位能反映出"那一拍 40"
        #    ⇒ `q_low` **交替** ⇒ `=1` 报第 **3** 拍、`=4` 一直不报 ✓✓）。
        _t = MotionTracker(min_hits=1, q_need=n, q_win=2)
        for _i in range(16):
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

    # ④ **目标的「预期位置」= 绿圈 ＋ 步长**（用户 2026-10-05 ✓ 原话："我们的当前帧的真目标
    #   预期位置**有且仅有一个：绿色的圆**" ✓✓）—— ⚠⚠ 这条**原来钉的是 `kf_pred_max` 让
    #   `pred` 往前挪** ✗：那是"`pred = obs ＋ 相机 ＋ vr`"的老口径 ✓；**目标**现在改由绿圈推 ✓
    #   （`kf_pred_max` 只管**非目标**的 `pred` ✓ 见 `process` 末尾那段 ✓）⇒ 这条跟着改成
    #   "**预测 = 绿圈 ＋ 步长，且步长按 `pos_step_max` 封顶**" ✓（= 下一拍绿圈会在的地方 ✓）。
    def _pred(mx):
        _t = MotionTracker(min_hits=1, kf_pred_max=mx, kf_q=25.0)
        _o = None
        for _i in range(6):
            _o = _t.process(None, ts=0.1 * _i,
                            dets=_d([300.0 + 30.0 * _i, 600.0, 900.0]))
        _x = _t._by_id(_t.tid) if _t.tid is not None else None
        if _x is None or _t.pos is None:
            return None
        # ⚠ 现口径：**绿圈 ＋ 相机 ＋ 它自己的相对速度** ✓（相对那份**扣掉夹取刚灌进去的** ✓
        #   `- _cv` ✓ 并按 `kf_pred_max` 封顶 ✓ —— 与 `process` 末尾那段**逐位同式** ✓）。
        _v = _x.vel or (0.0, 0.0)
        _md = (getattr(_t, "_med_ema", None) or (0.0, 0.0))
        _cv = ((_o or {}).get("clamp_vel") or (0.0, 0.0))
        _rl = (float(_v[0]) - _md[0] - _cv[0], float(_v[1]) - _md[1] - _cv[1])
        _n = math.hypot(_rl[0], _rl[1])
        if _n > _t.kf_pred_max:
            _k = _t.kf_pred_max / _n
            _rl = (_rl[0] * _k, _rl[1] * _k)
        return (_x.pred, (_t.pos[0] + _md[0] + _rl[0], _t.pos[1] + _md[1] + _rl[1]))
    _a = _pred(0.0)
    _b = _pred(40.0)
    check(_a is not None and _b is not None
          and abs(_a[0][0] - _a[1][0]) < 1e-6 and abs(_a[0][1] - _a[1][1]) < 1e-6
          and abs(_b[0][0] - _b[1][0]) < 1e-6 and abs(_b[0][1] - _b[1][1]) < 1e-6,
          "④ **目标的「预期位置」= 绿圈 ＋ 相机 ＋ 它自己的相对速度**（相对那份**扣掉夹取那份** ✓"
          "并按 `kf_pred_max` 封顶 ✓）：实测 %s vs %s ✓" % (_a, _b))


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


def test_cool_is_not_fused():
    """⭐⭐⭐⭐⭐ **"冷却"不是"融合"**（用户 2026-10-05 ✓ 原话："**拆状态码 ＋ 冷却期位置采信**" ✓✓，
    起因："**这框这么小还能被判为融合框？**" ✓）。

    ⚠ 按铁律 2（**融合框一定是真目标＋假目标** ✓）：**冷却期**那格框**已经分开**（尺寸回到
      **单目标** ✓）⇒ 它**不是**融合框 ✗ ⇒ `st` 不许再占 `1` ✗（旧码 `stuck or cool ⇒ 1` ✗
      ⇒ 面板/图例对一个 156×140 的小框说"融合"✗ —— 用户一眼看出不对 ✓）。
    夹具：真目标那格 200×200 粘 5 拍 ⇒ 再回到 150×150（⇒ 分开 ⇒ 冷却 `_STUCK_COOL = 2` 拍 ✓）。
    钉三条：
      ① **状态码拆开了**：粘连拍 `st == 1` ✓ ｜ **冷却拍 `st == 4`** ✓（且那拍框 = 单目标尺寸 ✓）；
      ② **冷却期位置采信**：冷却拍的 `obs` == **本拍框心** ✓（旧口径取"上一拍算好的预测"✗
         ⇒ 那时 `obs` 与框心差一整个位移 ✗）；
      ③ **冷却期 `vel` 照样更新** ✓（旧口径冻结 ✗）。
    """
    def _seq(n=16):
        _f0 = _frame(9)
        _s = []
        for _i in range(n):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _sz = 200.0 if (6 <= _i <= 10) else 150.0
            _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1],
                        _sz, _sz, 0.9)]
            _s.append(_d)
        return _s

    _t = MotionTracker(min_hits=3)
    _rows = []
    for _i, _d in enumerate(_seq()):
        _t.process(None, ts=_i * 0.16, dets=_d)
        _g = _t._by_id(_t.tid) if _t.tid is not None else None
        _b = next((x for x in (getattr(_t, "_box_v", []) or [])
                   if _t.tid is not None and int(x[4]) == int(_t.tid)), None)
        _rows.append({
            "sk": -1 if _g is None else int(_g.stuck),
            "ck": -1 if _g is None else int(_g.cool),
            "st": None if _b is None else int(_b[7]),
            "wh": None if _b is None else (float(_b[10]), float(_b[11])),
            "obs": (None if (_g is None or _g.obs is None)
                    else (float(_g.obs[0]), float(_g.obs[1]))),
            "bc": None if _b is None else (float(_b[0]), float(_b[1])),
            "vel": (None if (_g is None or _g.vel is None)
                    else (float(_g.vel[0]), float(_g.vel[1])))})
    _stk = [_i for _i, _r in enumerate(_rows) if _r["sk"] > 0 and _r["st"] is not None]
    _col = [_i for _i, _r in enumerate(_rows)
            if _r["sk"] == 0 and _r["ck"] > 0 and _r["st"] is not None]
    check(bool(_stk) and bool(_col)
          and all(_rows[_i]["st"] == 1 for _i in _stk)
          and all(_rows[_i]["st"] == 4 for _i in _col),
          "① **状态码拆开了**：粘连拍（第 %s 拍）`st = 1` ✓ ｜ **冷却拍（第 %s 拍）`st = 4`** ✓"
          "（冷却拍框 %s = 单目标 ✓ ⇒ 它**不是**融合框 ✓）"
          " —— ⚠ 改回 `stuck or cool ⇒ 1` ⇒ 冷却拍报 **1** ⇒ 立刻红 ✗✗"
          % ([_i + 1 for _i in _stk], [_i + 1 for _i in _col],
             [("%.0fx%.0f" % _rows[_i]["wh"]) if _rows[_i]["wh"] else "—" for _i in _col]))
    _gaps = [None if (_rows[_i]["obs"] is None or _rows[_i]["bc"] is None)
             else math.hypot(_rows[_i]["obs"][0] - _rows[_i]["bc"][0],
                             _rows[_i]["obs"][1] - _rows[_i]["bc"][1]) for _i in _col]
    check(bool(_col) and _gaps and max(_gaps) <= 1.0,
          "② **冷却期位置采信**：冷却拍 `obs` == **本拍框心**（实测差 %s px ✓，阈值 1.0）"
          " —— ⚠ 旧口径取「上一拍算好的预测」✗ ⇒ 那时 `obs` 与框心差**一整个位移**（实测十几~几十 px ✗✗）"
          % (["—" if _x is None else round(_x, 1) for _x in _gaps]))
    _v0 = next((_rows[_i]["vel"] for _i in reversed(range(len(_rows)))
                if _rows[_i]["sk"] > 0 and _rows[_i]["vel"] is not None), None)
    _v1 = next((_rows[_i]["vel"] for _i in _col if _rows[_i]["vel"] is not None), None)
    check(_v0 is not None and _v1 is not None
          and (abs(_v0[0] - _v1[0]) > 1e-6 or abs(_v0[1] - _v1[1]) > 1e-6),
          "③ **冷却期 `vel` 照样更新** ✓：粘连最后一拍 `vel` = (%.1f,%.1f) ⇒ 冷却第一拍 "
          "**(%.1f,%.1f)** ✓（变了 ✓ —— ⚠ 旧口径在那几拍**冻结** `vel` ✗ ⇒ 两值逐位相同 ⇒ 红 ✗）"
          % (0.0 if _v0 is None else _v0[0], 0.0 if _v0 is None else _v0[1],
             0.0 if _v1 is None else _v1[0], 0.0 if _v1 is None else _v1[1]))


def test_clamp_reach_is_half_box():
    """⭐⭐⭐⭐⭐ **夹取取框门：相交 ✓ 或"圆心离框 ≤ 半个框"**（用户 2026-10-05 ✓ 原话：
    "**有唯一大融合框说明真目标必在里面，如果按上拍的预测，圆的位置本拍在融合框外面，
    本拍直接夹取过去就完事了**" ✓✓ ＋ 他挑的**折中方案** ✓）。

    ⚠ 为什么要放松 ✗：融合期预估**本来就是混沌的** ✓（默认真惯性滑行 ＋ 靠融合框几何修正 ✓）
      ⇒ 这时候**框比预估可信** ✓ ⇒ "圆盘还没碰到框"不该成为**不夹**的理由 ✗；但要挡住
      "横跨半屏的劫持"✗（用户 2026-10-05 踩过离圆 131px 的框把圆**一次性拍飞** ✓）
      ⇒ 定成：**每根轴**离框 ≤ `max(半径, 半个框)` ✓。
    ⚠⚠ **为什么只能钉源码**（不是偷懒 ✓ 同 `test_guide_keyed_on_circle` ✓）：要造这个局面得让
      "**别的那条轨迹**"拿着一格大融合框、而圆心离它恰好在"半个框"内外两侧 ✗ —— 离屏小夹具里
      几条轨迹挨太近，配对的归属会被"谁离得更近"随手改掉 ✗（**实测踩到** ✓ 目标那条会**反手
      把旁边的大框抢过来** ⇒ 局面不出现 ✓）。
    ⇒ 真素材上这条是**实测过**的 ✓（`10月3日`，夹取**前**圆盘戳出 → 夹取量 → 夹取后 ✓）：
      帧 15 `(0.0,0.0) → 0.0px → (0.0,0.0)`（本来就在框里 ✓）；
      帧 16 `(41.7,51.6) → **66.3px（一拍直接夹进去 ✓）** → (0.0,0.0)`；
      帧 17 `(10.3,15.4) → 18.5px → (0.0,0.0)`；帧 18 `(0.9,16.1) → 36.5px → (0.0,0.0)` ✓。
    钉两条：
      ① **源码级**：那道门**每根轴**都比"`max(半径, 半个框)`"✓（不是只比半径 ✗ = 旧口径）；
      ② **源码级**：它**用在"② 画面里 `st == 1` 的框"那一步** ✓（= 只管**夹取**的那条路 ✓）。
    """
    import inspect
    _src = inspect.getsource(MotionTracker.process).replace(" ", "")
    check("abs(_px0[0]-_qx)<=max(_rc,_bw/2.0)" in _src
          and "abs(_px0[1]-_qy)<=max(_rc,_bh/2.0)" in _src,
          "① **源码级：门是「每根轴 ≤ max(半径, 半个框)」** ✓（`lie_motion` 的 `_near_circle` ✓）"
          " —— ⚠ 改回'sqrt(dx²+dy²) ≤ r'（= 只认相交 ✗）⇒ 帧 16 那种'预测落在框外'就**不夹**了"
          " ✗（用户的铁律：融合期**框比预估可信** ✓ 该夹就夹过去 ✓）")
    check("int(_b[7])==1and_near_circle(" in _src,
          "② **源码级：这道理只用在「取框那一步」** ✓（只管**夹取** ✓；⚠ 现在它前面多了一道"
          "**规则①的保护**（`not _own_clean` ✓）：**目标自己有干净框时不许拿别人的融合框夹它** ✗"
          " 见下面那段注释 ✓）—— ⚠ 别用它去改「这格框算不算融合」那个**标签** ✗"
          "（那是 `has_tgt` ✓ 见框表组装 ✓ 两者别混 ✓）")


def test_white_locks_target():
    """⭐⭐⭐⭐⭐ **任何时刻出现白块 ⇒ 直接当真目标**（用户 2026-10-05 ✓ 原话："**因为实际情况不知道
    何时开始，所以任何时候出现白色图形都需要认（为）真目标**" ✓✓ —— 这就是 "**A+B**" 里的 **A** ✓）。

    ⚠⚠ 实测病根（`10月3日 (2)` 帧 1~3 ✓）：白块每拍都找到 ✓、白度加分也一直喂给 `#5` ✓、
      `#5` 也一直是**分数第一** ✓ —— 可"从零选目标"要求 `hits ≥ min_hits`（**用户配置 = 4** ✓）
      ⇒ 前 3 拍 `state = init`、**一个圈都没有** ✗ ⇒ 用户看到"开始没认白块" ✓。
    ⚠ 另一条路**走不通**（**实测** ✓）：把 `min_hits` 调到 1 ✗ ⇒ 第 1 拍就锁在**假目标**上
      （分数全 0 ⇒ `max` 取第一个 ✗），而且"从零选"**只跑一次** ⇒ 再也改不回来 ✗✗。
    ⇒ 所以：**白块认定不受 `min_hits` 限制** ✓、**任何时刻**都生效 ✓（中途出现/遮挡后回来 ✓）。

    ⚠⚠⚠ **但它加了"白块去抖"**（用户 2026-10-05 ✓ "1+2" 里的 **2** ✓ 见 `_WHITE_STREAK` ✓）：
      **上一拍也得有白块、且两拍之间挪得不多** ✓ ⇒ 真白块（每拍都在 ✓）**第 2 拍就认** ✓；
      单拍闪现的假白块（实测帧 19/22/44/47 ✓）**一个都过不来** ✓（另有专门那条 ✓）。

    钉两条：① **第 2 拍就 `state == "track"`** ✓（⚠ 去掉白块认定 ⇒ 前 `min_hits` 拍都是 `init` ⇒
      红 ✗；⚠ 第 1 拍**故意不清** ✗ —— 去抖要求"上一拍也有" ⇒ 晚 1 拍是**设计如此** ✓）；
    ② 报出的圆**就在白块（= 真目标）身上** ✓。
    """
    _base = np.full((500, 750, 3), 100, np.uint8)
    _fake0 = _frame(9)

    def _scene(n=6):
        _out = []
        for _i in range(n):
            _cam = (5.0 * _i, 2.0 * _i)
            _tx, _ty = 380.0 + _cam[0], 300.0 + _cam[1]
            _im = _base.copy()
            # ⚠ 白块要**够大**（> 画面 5%% ✓ 见 `test_white_helper` 那个坑 ✓）＋ 跟着目标走 ✓
            _im[int(_ty) - 78:int(_ty) + 78, int(_tx) - 78:int(_tx) + 78] = (255, 255, 255)
            _d = _shift(_fake0, _cam[0], _cam[1])
            _d = _d + [(0, _tx, _ty, 150.0, 150.0, 0.9)]
            _out.append((_i, _im, _d))
        return _out

    _t = MotionTracker(min_hits=4)                 # ⚠ 正好是**用户配置那档** ✓
    _rows = []
    for _i, _im, _d in _scene():
        _o = _t.process(_im, ts=_i * 0.16, dets=_d)
        _rows.append((_i + 1, _o.get("state"),
                      None if _o.get("pos") is None else (float(_o["pos"][0]), float(_o["pos"][1])),
                      None if _t.last_white is None else (float(_t.last_white[0]), float(_t.last_white[1])),
                      getattr(_t, "_switch_kind", "")))
    check(_rows[0][1] == "init" and all(_r[1] == "track" for _r in _rows[1:])
          and _rows[1][4] == "white",
          "① **第 2 拍就定下目标** ✓（实测每拍 `state` = %s ✓、第 2 拍换目标方式 = %r ✓ —— "
          "⚠ 只要在 `min_hits`（4 ✓）之内定下就行；⚠ 第 1 拍是 `init` 是**去抖的代价** ✓ "
          "见 `_WHITE_STREAK` ✓，别当成 bug ✗）"
          % ([_r[1] for _r in _rows], _rows[1][4]))
    _d0 = max((math.hypot(_r[2][0] - _r[3][0], _r[2][1] - _r[3][1])
               for _r in _rows if _r[2] is not None and _r[3] is not None), default=-1.0)
    check(0.0 <= _d0 <= 30.0,
          "② **圆就落在白块（真目标）身上** ✓（最大差 **%.1f px** ✓ ≤ 30 ✓；白块中心 = 目标位置 ✓）"
          % _d0)
    _t2 = MotionTracker(min_hits=4, white_w=0.0)           # 关掉白度 ⇒ 换法里**一个 white 都不该有** ✓
    _k2 = []
    for _i, _im, _d in _scene():
        _t2.process(_im, ts=_i * 0.16, dets=_d)
        _k2.append(getattr(_t2, "_switch_kind", ""))
    check("white" not in _k2 and "white" in [_r[4] for _r in _rows],
          "③ **反面对照**（防假绿 ✓）：把白度关掉（`white_w=0` ✓）⇒ 换法里**一次 `white` 都没有** ✓"
          "（实测纯运动那条 = %s ✓ ｜ 开着的那条有 `white` ✓ —— ⚠ 白块那套要是整体不生效 ⇒ "
          "第一条也会「绿得不真实」✗）" % (_k2,))


def test_white_flash_ignored():
    """⭐⭐⭐⭐⭐ **单拍闪现的假白块不许"认定"**（用户 2026-10-05 ✓ "1+2" 里的 **2** ✓ 原话：
    "**为什么找到了 明显属于假目标的检出框**" ✓✓）。

    ⚠⚠ 实测病根（`10月3日 (2)` ✓）：白块检测是**像素级启发式** ⇒ 帧 **19 / 22 / 44 / 47** 会在
      **画面别处的小亮斑**（面积只有 307~360px ✗）上报一块"白"，而且**全是单拍闪现** ✓
      （前一拍没有、后一拍也没有 ✓）；而"白块认定"是**直接改 `tid`** 的硬规则 ✗ ⇒ 帧 44 就把目标
      **劫持**到旁边最近的 `#3`（`hits=44` 老轨迹 ✓、`dev=1.3` ≈ 噪声底 ⇒ **明显合群 = 假目标**
      ✗✗ —— 正是用户抓的那条 ✓）。

    ⇒ 去抖（见 `_WHITE_STREAK` ✓）：**上一拍也得有白块、且两拍之间挪 ≤ `_WHITE_STEP`** ✓。
      钉三条（一正一反 ✓）：
      ① **孤立闪现 ⇒ 不认** ✓（那拍白块**就贴在 `#1` 身上 2.1px** ✗ 也不认 ✓、`tid` 不动 ✓ ——
         ⚠ 去掉去抖 ⇒ 这一拍**必被劫持**到 `#1` ⇒ 立刻红 ✗✗）；
      ② **反面：连着两拍 ⇒ 照认** ✓（实测第 13 拍照旧切过去 ✓ —— ⚠ 一刀切"白块永不改目标"
         ⇒ 这条红 ✗）。
         ⚠⚠ **夹具 2026-10-05 改过** ✗（如实记 ✓）：原来让白块连着两拍贴**环上那块砖** ✗，还要求
         "照旧切过去" —— 那是**用户现在明确不许的行为** ✗（"**砖不许抢座位**"✓，因果见
         `9月30日(1)` 帧 9：一块 `dev=0` 的新账离白块 4px ⇒ 把真目标赶下台 ✓）。⇒ 现在贴的是
         **第二个"真的在走"的东西**（+14px/拍 ✓ ⇒ 过"要在动"那道门 ✓）⇒ 该切就切 ✓。
         ⚠ 于是两条各测一件事 ✓：① 测**去抖**（单拍闪现不许认 ✓）、② 测**连续两拍该认**（别一刀切 ✗）；
      ③ **夹具成立**：闪现那一拍「白块 ↔ 最近那条轨迹」**确实在 `_WHITE_MATCH` 之内** ✓
         （实测 **2.1px** ≤ 45 ✓ = "老口径下必被劫持" ✓）。
    """
    _base = np.full((500, 750, 3), 100, np.uint8)
    _fake0 = _frame(9)

    def _block(_im, _cx, _cy):
        """贴一块纯白（⚠ 半宽 78 ⇒ 156×156 ⇒ 够大 ✓ 见 `test_white_helper` 那个坑 ✓）。"""
        _im[int(_cy) - 78:int(_cy) + 78, int(_cx) - 78:int(_cx) + 78] = (255, 255, 255)

    def _scene(n=14):
        _out = []
        for _i in range(n):
            _cam = (5.0 * _i, 2.0 * _i)
            _tx, _ty = 380.0 + _cam[0] + 11.0 * _i, 300.0 + _cam[1]   # 真目标相对群体 +11px/拍 ✓
            # ⭐⭐⭐⭐⭐ **白块后来要贴的那个东西，必须是"真的在走"的** ✗✗（用户 2026-10-05 ✓
            #   见下面 ② 那段 ✓）—— ⚠ 我第一版让它贴**环上那块砖**（跟大部队走 ⇒ `dev ≈ 0` ✗）
            #   ⇒ 新加的那道门「**换人得有"它在动"的证据**」把它挡住了 ⇒ ② 红 ✗ ——
            #   ⚠⚠ **那不是 bug，那正是用户要的** ✓（"**砖不许抢座位**"✓，因果见 `9月30日(1)` 帧 9 ✓）
            #   ⇒ **夹具换成"在走的东西"** ✓（+14px/拍 ✓ ⇒ 过门 ✓ ⇒ 该切就切 ✓）。
            #   ⚠⚠ **位置要算准** ✗（**踩过** ✓）：我第一版放 `635 + 14i` ⇒ 帧 9 已经到 x=787
            #     ⇒ **出画面**（图宽 750 ✓）⇒ 白块压根没贴上 ⇒ ②③ 一起假红 ✗。
            #     ⇒ 现在：**画面内** ✓、**离真目标那格框足够远** ✓（否则两框重叠 ⇒ 按 `_SUB_COV`
            #     会被标成次级 ⇒ `dev` 记 0 ⇒ 又过不了"要在动"那道门 ✗）、**尺寸小一点**（90 ✓）
            #     免得贴到画面底边被切 ✗（切了也算次级 ⇒ `dev` 0 ✗）。
            _mx, _my = 200.0 + _cam[0] + 14.0 * _i, 420.0 + _cam[1]
            _im = _base.copy()
            if _i <= 3:                       # 帧 1~4：白块在**真目标**身上 ⇒ 先锁上它 ✓
                _block(_im, _tx, _ty)
            if _i == 8:                       # 帧 9：**孤立**单拍闪现 ✓（下一拍没有 ✓）
                _block(_im, _mx, _my)
            if _i >= 11:                      # 帧 12~14：**连着**贴在同一个"在走的东西"上 ✓
                _block(_im, _mx, _my)
            _d = _shift(_fake0, _cam[0], _cam[1])
            _d = _d + [(0, _tx, _ty, 150.0, 150.0, 0.9),
                       (0, _mx, _my, 90.0, 90.0, 0.9)]
            _out.append((_i, _im, _d))
        return _out

    _t = MotionTracker(min_hits=4)
    _rows = []
    for _i, _im, _d in _scene():
        _o = _t.process(_im, ts=_i * 0.16, dets=_d)
        _nearest = None
        if _t.last_white is not None:
            _cand = [(math.hypot(_x.obs[0] - _t.last_white[0], _x.obs[1] - _t.last_white[1]), _x)
                     for _x in _t.tracks if _x.obs is not None]
            _nearest = min(_cand, key=lambda z: z[0]) if _cand else None
        _rows.append((_i + 1, _o.get("state"), _t.tid,
                      None if _t.last_white is None else (float(_t.last_white[0]),
                                                          float(_t.last_white[1])),
                      getattr(_t, "_switch_kind", ""),
                      None if _nearest is None else (int(_nearest[1].id), float(_nearest[0]))))
    _true = next((_r[2] for _r in _rows if _r[0] == 2), None)      # 帧 2 认下的那条 = 真目标 ✓
    _flash = next((_r for _r in _rows if _r[0] == 9), None)        # 帧 9 = 孤立闪现 ✓
    _ign = [_r[0] for _r in _rows if 9 <= _r[0] <= 12
            and _r[2] == _true and _r[4] != "white"]
    check(_true is not None and _flash is not None and len(_ign) == 4,
          "① **孤立闪现 ⇒ 不认** ✓：帧 9~12 **每拍 `tid` 都还是真目标 #%s** ✓（实测 `state` 那几拍 "
          "= %s ✓、换法 = %s ✓）—— ⚠ 去掉去抖 ⇒ 帧 9 那拍**必被劫持**到 `#%s` ⇒ 立刻红 ✗✗"
          % (_true, [_r[1] for _r in _rows if 9 <= _r[0] <= 12],
             [_r[4] for _r in _rows if 9 <= _r[0] <= 12],
             "-" if (_flash is None or _flash[5] is None) else _flash[5][0]))
    _l13 = next((_r for _r in _rows if _r[0] == 13), None)
    # ⚠⚠ **比的是"本拍白块最近的那条"** ✗（**踩过** ✓）：原来拿**帧 9** 记下的 id 去比 ✗ ⇒
    #   夹具一改（棋块从"环上那块砖"换成"第二个在走的东西"✓）就变成**两个不同的 id** ⇒ 假红 ✗
    #   （实测：帧 13 `tid` = 13 ✓ 换法 = `white` ✓ —— **行为本来就对** ✓）。
    check(_l13 is not None and _l13[5] is not None
          and _l13[2] == _l13[5][0] and _l13[4] == "white",
          "② **反面：连着两拍 ⇒ 照认** ✓：帧 12~13 白块**连续两拍**贴着同一格 ⇒ 第 13 拍照旧切过去 "
          "（实测帧 13：`tid` = %s ✓、**本拍**白块最近那条 = `#%s` ✓〔要相等 ✓〕、换法 = %r ✓"
          "〔要 `white` ✓〕）—— ⚠ 一刀切「白块永不改目标」⇒ 这条红 ✗"
          % ("-" if _l13 is None else _l13[2],
             "-" if (_l13 is None or _l13[5] is None) else _l13[5][0],
             "-" if _l13 is None else _l13[4]))
    check(_flash is not None and _flash[5] is not None and _flash[5][1] <= _WHITE_MATCH,
          "③ **夹具成立**：闪现那拍「白块 ↔ 最近那条轨迹」**确实在门内** ✓（实测 **%.1f px** ≤ "
          "`_WHITE_MATCH` **%.0f** ✓ ⇒ 老口径下「必被劫持」✓ —— ⚠ 白块要是贴得比门还远 ⇒ 谁也"
          "认不了 ⇒ 本用例就不再咬人 ✗）"
          % ((-1.0 if (_flash is None or _flash[5] is None) else _flash[5][1]), _WHITE_MATCH))


def test_white_area_floor_real_clip():
    """⭐⭐⭐⭐⭐ **白块面积下限**（用户 2026-10-05 ✓ "1+2" 里的 **1** ✓ 原话："**为什么找到了 明显
    属于假目标的检出框**" ✓✓）。

    ⚠⚠ 实测（`10月3日 (2).mp4` ✓ 逐拍量 `pick_white` 的**面积**）：
      · **真白块**（帧 1~9 ✓ 那个白星）：**8712 ~ 9068**（画面 2.3~2.4% ✓）；
      · **假白块**（帧 44 ✓ 劫持那次）：**353**（0.09% ✗）—— **差 25 倍** ✗，而原来那道门只写了
        **300** ✗ ⇒ 小亮斑照样过 ✗（帧 19/22/47 分别是 307 / 343 / 360 ✓ 全在门内 ✗）。
    ⇒ `_AREA_LO` 提到 **2000**；本条**在真素材上钉两端** ✓：
      ① 真白块**照样挑得出** ✓（⚠ 提太高 ⇒ 白度辅助整个失效 ⇒ 红 ✗）；
      ② 帧 44 那个假白块**出局**（`None` ✓）；
      ③ **反面对照**：下限退回 `300` ⇒ 它**立刻冒出来** ✓（钉住"是这道门在起作用"✗ 不是碰巧 ✓）。
    ⚠ 真素材用例 ⇒ 找不到文件就**跳过** ✓（同 `test_real_clip_smoke` ✓ 那条纪律 ✓）。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月3日 (2).mp4")
    if not _src.exists():
        check(True, "跳过白块面积下限（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        _frames = LD.load_video(str(_src))[0]
    except Exception as _e:                 # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过白块面积下限（环境不齐：%s ✓）" % _e)
        return
    if len(_frames) < 44:
        check(True, "跳过白块面积下限（素材只有 %d 拍 ✓ 不够用 ✓）" % len(_frames))
        return
    _g1 = _gray(_frames[0][1])              # 帧 1 = 真白块 ✓
    _g44 = _gray(_frames[43][1])            # 帧 44 = 那个 353px 的假白块 ✓
    _real = pick_white(_g1)
    check(_real is not None,
          "① **真白块照样挑得出** ✓（帧 1：%s ✓ —— ⚠ 门提太高 ⇒ 白度辅助整个失效 ⇒ 红 ✗）"
          % (None if _real is None else "(%.0f,%.0f)" % (_real[0], _real[1])))
    check(pick_white(_g44) is None,
          "② **帧 44 那个假白块出局** ✓（`pick_white` = None ✓ —— 原来 300 那道门放它进来 ⇒ "
          "白块认定把目标劫持到 `#3` ✗✗）")
    check(pick_white(_g44, area_lo=300.0) is not None,
          "③ **反面对照**：下限退回 **300** ⇒ 它**立刻冒出来** ✓（%s ✓ = 这道门真的在把关 ✓ "
          "不是碰巧 ✓）"
          % (None if pick_white(_g44, area_lo=300.0) is None
             else "(%.0f,%.0f)" % pick_white(_g44, area_lo=300.0)[:2]))
    import inspect
    _src = inspect.getsource(MotionTracker.process).replace(" ", "")
    #   ⚠⚠ **2026-10-07 口径升级：白块也要"只在限定区域里找"** ✗✗（用户 ✓ "**只在限定的区域
    #     计算（因为这是部分弹窗）**" ✓）⇒ 那一行从 `pick_white(_gray(img))` 改成了
    #     "先 `_gray(img)` ⇒ **裁到框内** ⇒ `pick_white(_g)`" ✓ ⇒ 本条钉子改成**钉本意** ✓：
    #     **调用处不许自带 `area_lo=`** ✗（门只许有 `_AREA_LO` 一处 ✓）；`_gray(img)` 照旧要在 ✓
    #     （灰度那一步还在 ✓ 只是多了一次裁剪 ✓）。
    check("pick_white(_g)" in _src and "pick_white(_g,area_lo=" not in _src
          and "_gray(img)" in _src,
          "④ **源码级：追踪器走的就是那个门** ✓（`process` 里是 `pick_white(_g)` ／ `_g` 来自 "
          "`_gray(img)` ✓〔⚠ 中间按**限定区域**裁了一刀 ✓ 见 `roi` 那段 ✓〕—— ⚠ 谁在那一处另写一个 "
          "`area_lo=` 数 ⇒ 立刻红 ✗：口径只许有 `_AREA_LO` **一处** ✓）")


def _ref_blobs(mask, area_lo, area_hi, fill_lo=_LM._FILL_LO, fill_hi=_LM._FILL_HI):
    """**「乙」之前的原版 `_blobs`**（逐 label 走 Python 循环 ✓）—— 只给参照用 ✓。"""
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


def _ref_pick_white(g, pos=None, gate=None, area_lo=_LM._AREA_LO, area_hi=_LM._AREA_HI,
                    q=_LM._WHITE_Q, want_mask=False, min_gain=_LM._MIN_GAIN):
    """**2026-10-07「乙」之前的原版 `pick_white`** —— 只当"参照物"留在测试里 ✓（见下面那条钉子 ✓）。

    ⚠ 它**故意**写成"整帧、每次都拷一份、每块整帧比一次"的笨办法 ✓（= 现在的快速版**必须**与它
      给出**同一批观测** ✓）；**运行期不用它** ✓（生产走快速版 ✓）。`_blobs` 也一并留了原版 ✓。
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
    _bs, _lab = _ref_blobs(_m, area_lo, area_hi)
    _best, _bv = None, -1e9
    for (_cx, _cy, _area, _fill, _k) in _bs:
        _mean = float(g[_lab == _k].mean())
        _score = _mean / 255.0 + min(_area, _LM._AREA_REF) / _LM._AREA_REF * _LM._AREA_W
        if _score > _bv:
            _best, _bv = (_cx, _cy, _area, _mean), _score
    if _best is None or float(_best[3]) < _med + float(min_gain):
        return (None, None) if want_mask else None
    return _best


def test_pick_white_fastpath_matches_reference():
    """⭐⭐⭐⭐⭐ **「乙」之后的快速 `pick_white` 必须与"原版"给出同一批观测** ✗✗（用户 2026-10-07 ✓
    他选了"**甲 ＋ 乙**" ✓ 起因："**我们这套算法性能够实时使用吗？**" ✓✓）。

    ⚠⚠ **为什么非有不可** ✗（`pick_white` 是"白度"那条证据链的**唯一入口** ✓）：这轮为了提速动了
      四处（**一次 partition 求分位＋中位** ✓、**无门时不再整帧拷 ROI** ✓、**掩膜零拷贝 `.view`** ✓、
      **每块均值只在那块包围盒里算** ✓）—— 每一处都能"看着更快、其实换了答案" ✗✗（示例：分位插值
      在 float64 而不是 float32 里算 ⇒ `_lo` 末位差一点 ⇒ 阈值边上那个像素翻面 ⇒ 块大小变 ✗）。
      ⇒ 本钉子拿**原版当参照**（`_ref_pick_white` ✓ 见上 ✓）逐帧比 ✓：观测**必须一样** ✓
        （位置/面积**逐位** ✓、亮度差 ≤ 1e-9 ✓ —— 实测只有一处均值差 5.7e-14 ✓ 是浮点求和次序 ✓）。
    钉两条：
      ① 夹具成立：这批帧上**至少找到过 1 次**白块 ✓（不然两边都给 `None` ⇒ 比对空洞 ⇒ 假绿 ✗）；
      ② 三种门（**无门 / r=90 / r=140** ✓）逐帧**全都一样** ✓（不符 = 0 条 ✓）。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月1日.mp4")
    if not _src.exists():
        check(True, "跳过「白块快速版对参照」真素材用例（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        _frames = LD.load_video(str(_src))[0]
    except Exception as _e:                    # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过「白块快速版对参照」真素材用例（环境不齐：%s ✓）" % _e)
        return
    _gates = [(None, None), ((300.0, 250.0), 90.0), ((500.0, 300.0), 140.0)]
    _seen, _bad = 0, []
    for _i in range(min(40, len(_frames))):
        _g = _gray(_frames[_i][1])
        for _gj, (_pos, _gate) in enumerate(_gates):
            _a = (_ref_pick_white(_g) if _pos is None
                  else _ref_pick_white(_g, pos=_pos, gate=_gate))
            _b = (pick_white(_g) if _pos is None
                  else pick_white(_g, pos=_pos, gate=_gate))
            if _a is not None:
                _seen += 1
            if (_a is None) != (_b is None):
                _bad.append((_i + 1, _gj, _a, _b))
                continue
            if _a is None:
                continue
            if (abs(float(_a[0]) - float(_b[0])) > 1e-9
                    or abs(float(_a[1]) - float(_b[1])) > 1e-9
                    or abs(float(_a[2]) - float(_b[2])) > 1e-9
                    or abs(float(_a[3]) - float(_b[3])) > 1e-9):
                _bad.append((_i + 1, _gj, _a, _b))
    check(_seen >= 1 and not _bad,
          "①② **白块快速版 = 原版** ✓（%d 帧 × %d 门 ✓；找到过白块 **%d** 次 ✓〔要 ≥1 ✓〕；**不一致 "
          "0 条** ⇒ 实测 **%d** 条 ✓〔要 0 ✓〕）—— ⚠⚠ 快速版要是「更快但换了答案」✗ ⇒ 本条红 ✗"
          "（白度是那条证据链的唯一入口 ✓）；%s"
          % (min(40, len(_frames)), len(_gates), _seen, len(_bad),
             _bad[:2] if _bad else "0 条 ✓"))


def test_dets_worker_prefers_gpu():
    """⭐⭐⭐⭐⭐ **「甲」（用户 2026-10-07 ✓ 他选的"甲 ＋ 乙" ✓）：检测默认上 GPU** ✗✗（起因："**我们
    这套算法性能够实时使用吗？**" ✓✓）。

    ⚠⚠ **病根**（**实测** ✓ RTX 5070 ＋ torch2.11+cu128 ＋ 权重 5.4MB ✓）：`model.device` = **cpu** ✗
      ⇒ 每帧 **28.6ms** ✗，而且 960→640→512 几乎不变（28.6→26.4→24.8 ✓）—— **正好证明瓶颈不在算力** ✓
      是"跑在了 CPU 上" ✓。⇒ 现在 `auto` ⇒ 有 CUDA 就 `"0"` ✓（没有就 `"cpu"` ✓ 不许崩 ✗）。
    钉三条：
      ① `auto`：有 CUDA ⇒ `"0"` ✓、没卡 ⇒ `"cpu"` ✓（拿 torch 自己问一遍当尺子 ✓）；
      ② 明确给 `"cpu"` / `"0"` ⇒ **原样透传** ✓（A/B 与自检就靠它 ✓）、`None` ⇒ 等于 `auto` ✓；
      ③ **源码级**：真调用那行必须带 `device=args.device` ✓（⚠⚠ **漏了这个参数不会报任何错** ✗，
         只会**悄悄跑回 CPU** ✗ ⇒ 这条必须钉 ✓）。
    """
    try:
        import tools.lie_dets_worker as W
    except Exception as _e:                       # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过「检测默认上 GPU」（环境不齐：%s ✓）" % _e)
        return
    try:
        import torch
        _has = bool(torch.cuda.is_available())
    except Exception:                             # noqa: BLE001 —— 没有 torch ⇒ 只查透传 ✓
        _has = None
    _auto = W.resolve_device("auto")
    check(_has is None or _auto == ("0" if _has else "cpu"),
          "① **`auto`：有 CUDA ⇒ \"0\"、没卡 ⇒ \"cpu\"** ✓（本机 CUDA = **%s** ⇒ 该给 **%s** ⇒ 实给 "
          "**%s** ✓）—— ⚠ 一刀切写死 `\"0\"` ✗ ⇒ 没卡的机器**直接崩** ✗；写死 `\"cpu\"` ✗ ⇒ 白改 ✓"
          % (_has, "0" if _has else "cpu", _auto))
    check(W.resolve_device("cpu") == "cpu" and W.resolve_device("0") == "0"
          and W.resolve_device(None) == _auto,
          "② **明确给的设备原样透传** ✓（`\"cpu\"` ⇒ **%s** ✓、`\"0\"` ⇒ **%s** ✓、`None` ⇒ **%s** "
          "✓〔= auto ✓〕）—— ⚠ A/B 就靠这条（要能**强制**跑 CPU 才比得出来 ✓）"
          % (W.resolve_device("cpu"), W.resolve_device("0"), W.resolve_device(None)))
    import inspect
    _src = inspect.getsource(W.main).replace(" ", "")
    check("device=args.device,verbose=False" in _src
          and "resolve_device(args.device)" in _src,
          "③ **源码级：真调用那行带着设备** ✓（`predict(..., device=args.device ✓` ＋ "
          "`args.device = resolve_device(...)` ✓）—— ⚠⚠ **漏了它不会报任何错** ✗，只会**悄悄跑回 "
          "CPU** ✗（28.6ms/帧 ✓）⇒ 这条必须钉 ✓")


def test_log_never_raises_on_missing_q():
    """⭐⭐⭐⭐⭐ **"跟丢"那句日志不许因为一个数没算出来就崩** ✗✗（用户 2026-10-07 ✓ 原话："**闪退**" ✓✓）。

    ⚠⚠ **实测根因**（`D:\\Media\\record\\mxd\\10月7日.mp4` **第 ~430 帧** ✓）：`_q_new`（"刚判跟丢"
      那个**一次性**标志 ✓）原来**只在算 `q` 的那段里赋值** ✗ ⇒ 下一拍要是**没走到那段**（丢框 /
      样本不够 ✓）⇒ `q` 已被清成 `None` ✗ 而 `_q_new` **还是上一拍的 `True`** ✗
      ⇒ `motion_viz` 里 `"…（q %.1f）" % …` 拿 `None` 当浮点 ⇒ **TypeError** ✓
      ⇒ 它抛在 **Qt 槽**里（`warm_tick` → `runner.step` ✓）⇒ PyQt 直接 `qFatal`/`abort`
      ⇒ **窗口整个消失、连 traceback 都看不到** ✓✓ —— 用户看到的就是"**闪退**" ✓。
    钉两条（**根** ＋ **兜底** ✓）：
      ① `_q_new=True` ＋ `q=None` ⇒ `motion_viz()` **不许抛** ✓（如实写"未知" ✓）；
      ② **根**：任何一拍走完 `process` ⇒ `_q_new` 只能"**本拍刚算出来**"或 `False` ✓
         （⚠ 少了这条 ⇒ 标志又跨拍漂到"没算 q 的那拍" ⇒ 接着炸 ✗）。
    """
    _t = MotionTracker(min_hits=3)
    #   ⚠⚠ **先跑一拍把内部状态备齐** ✗✗（**我第一版就在这栽了** ✓：裸 tracker 直接调
    #     `motion_viz()` ⇒ 抛 `AttributeError: '_cam_len'` ✗ ⇒ **那是用例自己的毛病** ✗，
    #     不是被测的那条路 ✓）。跑完 **再**制造病态状态 ✓（`q=None` ＋ `_q_new=True` ✓）。
    _t.process(None, ts=0.0, dets=[])
    _t.q = None
    _t._q_new = True
    _ok, _m = True, None
    try:
        _m = _t.motion_viz(followed_pos=None)
    except Exception as _e:                       # noqa: BLE001 —— 就是要抓它 ✓
        _ok = False
        check(False, "① `q=None` ＋ `_q_new=True` 时 `motion_viz()` 抛了：%r ✗" % (_e,))
    _txt = " ".join(str(_v) for _v in (_m or {}).values() if isinstance(_v, str))
    check(_ok and _m is not None,
          "① **`q=None` ＋ `_q_new=True` ⇒ `motion_viz()` 不抛** ✓（⚠ 抛出来在 Qt 槽里就是"
          "**整窗消失** ✗ ⇒ 这就是用户报的「闪退」✓）")
    if _txt:
        check("未知" in _txt or "跟丢" not in _txt,
              "①b 那句日志里**没有**拿 `None` 硬套浮点 ✓（实测文本片段：%s）" % _txt[:60])

    _t2 = MotionTracker(min_hits=3)
    _t2._q_new = True
    _t2.process(None, ts=0.0, dets=[])
    check(_t2._q_new is False,
          "② **根：走完一拍 ⇒ `_q_new` 必须复位** ✓（实测 = **%s** ✓〔要 False ✓〕）—— ⚠ 少了它 ⇒ "
          "标志跨拍漂到「没算 q 的那拍」⇒ 日志又拿 `None` 当浮点 ⇒ 接着闪退 ✗"
          % (_t2._q_new,))


def test_roi_limits_everything():
    """⭐⭐⭐⭐⭐ **限定区域（ROI ✓ 用户 2026-10-07 ✓ 原话："**只在限定的区域计算（因为这是部分
    弹窗）**" ✓✓）** —— 框外**一概看不见** ✓（不是"降权" ✗）。

    ⚠⚠ **实测病根**（`10月7日.mp4` ✓）：那条片是**整屏录像** ✓（蓝天白云 ✗ + 各种 UI ✗）⇒
      算法**整幅**在算 ⇒ `pick_white` 把**一朵云**当成白块 ✗（实测落在 (462,64) ✓ **就是天空** ✓）、
      座位 `#1`（分 0~6 ✗ `q` 0.1~1.2 ✗）也是从**背景**里选出来的 ✗ ⇒ 整条链全乱 ✓。
    钉四条：
      ① **框外的检出框不建账** ✓（两格框：一内一外 ⇒ **只有框内那条有账** ✓）；
      ② **框外的白块不算** ✓（白块放框外 ⇒ `last_white is None` ✓；挪进框内 ⇒ 立刻找到 ✓）；
      ③ **`roi=None` ⇒ 与"不给"完全一致** ✓（同一段序列逐拍 `tid` 相同 ✓ = **老行为一个字不变** ✓）；
      ④ **非法框当没给** ✓（反了 / 长度不对 ⇒ 不裁成空 ✓ 与 `None` 同 ✓）。
    """
    _W, _H = 750, 500
    _img = np.full((_H, _W, 3), 100, np.uint8)

    def _mk():
        _t = MotionTracker(min_hits=1)
        _t.frame_wh = (_W, _H)
        return _t

    # ---- ① 框外框内各一格检出框（都在动 ✓ 一起往右走 ✓）----
    _in = (400.0, 250.0)                       # 框内
    _out = (80.0, 80.0)                        # 框外（左上空一点的地方 ✓）
    _t1 = _mk()
    _roi = (_in[0] - 60.0, _in[1] - 60.0, _in[0] + 60.0, _in[1] + 60.0)
    _t2 = MotionTracker(min_hits=1, roi=_roi)
    _t2.frame_wh = (_W, _H)
    for _k in range(6):
        _d = [(0, _in[0] + 8.0 * _k, _in[1], 60.0, 60.0, 0.9),
              (0, _out[0] + 8.0 * _k, _out[1], 60.0, 60.0, 0.9)]
        _t1.process(_img, ts=_k * 0.16, dets=list(_d))
        _t2.process(_img, ts=_k * 0.16, dets=list(_d))
    _near_in = [x for x in _t2.tracks
                if x.obs is not None and abs(float(x.obs[0]) - _in[0] - 40.0) < 80.0]
    _near_out = [x for x in _t2.tracks
                 if x.obs is not None and abs(float(x.obs[0]) - _out[0] - 40.0) < 80.0]
    check(len(_t1.tracks) >= 2 and len(_near_in) >= 1 and not _near_out,
          "① **框外的检出框**（%s ✓）**不许建账** ✓（框内那条在 ✓ = %d 条；框外那条 = **%d** 条"
          "〔要 0 ✓〕；⚠ 不给框时两条都在 ✓ = %d 条〔≥2 ✓〕）"
          % ("(%.0f,%.0f)" % _out, len(_near_in), len(_near_out), len(_t1.tracks)))

    # ---- ② 白块：框外不算、框内才算（⚠ 用 `_AREA_LO` 那一档的实心块 ✓ 灰底白块 ✓）----
    _blob_out = np.full((_H, _W, 3), 100, np.uint8)
    _blob_out[80:140, 80:140] = 255            # 左上角那朵"云" ✓（框外 ✓）
    _blob_in = _blob_out.copy()
    _blob_in[80:140, 80:140] = 100             # 擦掉 ✓
    _blob_in[220:280, 370:430] = 255           # 挪进框里 ✓
    _t3 = MotionTracker(min_hits=1, roi=_roi)
    _t3.process(_blob_out, ts=0.0, dets=[])
    _t4 = MotionTracker(min_hits=1, roi=_roi)
    _t4.process(_blob_in, ts=0.0, dets=[])
    check(_t3.last_white is None and _t4.last_white is not None,
          "② **白块也只在框里算** ✓（框外那朵 ⇒ 白块 = **%s** ✓〔要 None ✓〕；挪进框里 ⇒ "
          "**%s** ✓〔要找到 ✓〕）—— ⚠⚠ 这条最要紧 ✗：整屏录像**满屏白云** ⇒ 全幅找白就是"
          "\"**把云当白块**\" ✗（实测 (462,64) 就是天空 ✓）"
          % (_t3.last_white, _t4.last_white))

    # ---- ③ `roi=None` ≡ 不给（**老行为一个字不变** ✓）----
    _a, _b = _mk(), MotionTracker(min_hits=1, roi=None)
    _b.frame_wh = (_W, _H)
    _same = True
    for _k in range(8):
        _d = [(0, 300.0 + 10.0 * _k, 240.0, 60.0, 60.0, 0.9),
              (0, 520.0, 300.0 + 6.0 * _k, 70.0, 70.0, 0.9)]
        _a.process(_img, ts=_k * 0.16, dets=list(_d))
        _b.process(_img, ts=_k * 0.16, dets=list(_d))
        if (_a.tid != _b.tid or _a.pos != _b.pos):
            _same = False
    check(_same and _a.tid is not None,
          "③ **`roi=None` ⇒ 与\"不给\"逐拍一致** ✓（`tid` 一样 ✓ = #%s、位置一样 ✓）—— ⚠ 默认改了行为 ✗ "
          "⇒ 这条红 ✗（= **没框之前跟改动前逐位一致** ✓）" % (_a.tid,))

    # ---- ④ 非法框 ⇒ 当没给（**宁可不裁，也不许裁成空的** ✓）----
    _c = MotionTracker(min_hits=1, roi=(500.0, 400.0, 100.0, 50.0))     # 反了 ✓
    _d2 = MotionTracker(min_hits=1, roi=(1.0, 2.0, 3.0))                # 长度不对 ✓
    check(_c.roi is None and _d2.roi is None,
          "④ **非法框 ⇒ 当没给** ✓（反了的 ⇒ `roi` = **%s** ✓；长度不对的 ⇒ **%s** ✓〔都要 None ✓〕）"
          "—— ⚠ 真按反框裁 ⇒ 框里永远是空的 ⇒ 看着像\"全坏了\" ✗" % (_c.roi, _d2.roi))


def test_gap_reentry_dev_uses_crowd_baseline():
    """⭐⭐⭐⭐⭐ **"跨拍接回来"那一拍：`dev` 要拿"大部队同样这几拍走了多少"当基准**
    （用户 2026-10-05 ✓ 原话："**咱们不是有 #11 的灰框记录它应该具有的位置和尺寸吗？跨拍接回来
    这一拍判断一下 IoU 不就能知道是不是假目标了**" ✓✓ —— ⚠ **IoU 版本我量过、分不开** ✗：
    全片 28 笔里老实的砖 IoU 能到 0.64 ✓、真在走的能低到 0.12 ✗（灰框自己会飘 ~30px ✗）；
    **"拿灰框当参照"这一步是对的** ✓ ⇒ 换成"位移 ÷ 跨的拍数"就分得开 ✓✓）。

    ⚠⚠ 实测病根（`10月3日 (2)` 帧 25 ✓）：`#11` 是一条**断了 13 拍的老幽灵** ✓，接回来那拍
      `mv` 是"**跨 13 拍的平均**"（≈ 大部队自己的速度 ✓）而基准用的是"**当拍的瞬时中位**" ✗✗
      ⇒ **两把尺子混用** ⇒ 凑出 **16.7px** 的假"不合群"（噪声底 2.3 的 7.2 倍 ✓ 正好过认领门
      `2.0 × 噪声`）⇒ **把真目标抢走** ✗✗（你抓的那笔 ✓）。⇒ 基准改成
      `(_crowd_cum − cum_obs) / mv_span` ✓ ⇒ 同一笔实测 **2.1px** ✓ ⇒ **不会再抢** ✓。

    夹具（合成 ✓ 先量后断言 ✓）：3 条"大部队"框 ＋ 一条砖 `#4`（**帧 4~8 掉线** ✓）；
      相机：帧 1~8 每拍 **(0,+8)** ✓；**帧 9 突然加速到 (0,+40)** ✓（= 回来那拍大部队猛地快一下 ✓）。
    钉四条（一正一反 ＋ 边界 ＋ 不变式）：
      ① **砖接回来 ⇒ `dev` = 0.00** ✓（帧 9 实测 ✓；⚠ 老口径会是 **26.7** ✗ ⇒ 立刻红 ✓）；
      ② **反面：真的在动 ⇒ 照旧报出来** ✓（把帧 9 那格框**挪开 +60px** ✗ —— ⚠⚠ **必须仍在配对
         门限（70px ✓）之内** ✓：实测 `dev` = **8.9** ⇒ 过门 ✓ —— ⚠ 一刀切"跨拍回来一律记 0"
         ⇒ 这条红 ✗）；
      ③ ⭐ **边界：挪出配对门限（+120px）⇒ `dev` 记 0、但 `miss` 仍归 0** ✓（用户 2026-10-05 ✓
         原话："**因为假目标是位于理想中的假目标检出框框心的，它不该有任何超过噪声允许的位移，
         有就是你算错了**" ✓✓ —— 砖的预测本来就准 ✓ ⇒ 70px 外的框**不是它的** ✓ ⇒ 只保号 ✓）；
      ④ **不变式（源码级 ✓）**：新支**只在 `mv_span > 1` 时接管** ✓ ⇒ 正常连续拍**逐位不变** ✓。
    ⚠⚠ **② 的门以前写的是 +90px（≈ 超出配对门限 ✓）** ✗ —— ⚠ 那**本身就是错的** ✓：一块**砖**
      不可能出现在 115px 外 ✓；真在走的那个 = **真目标** ✓，而真目标的预测**带着它自己的相对
      速度** ✓ ⇒ 回来时是**近的** ✓ ⇒ **根本用不上"远接"这条通道** ✓✓（2026-10-05 按用户第二句
      判决订正 ✓）。
    """
    _step = {1: (0.0, 0.0), 2: (0.0, 8.0), 3: (0.0, 16.0), 4: (0.0, 24.0), 5: (0.0, 32.0),
             6: (0.0, 40.0), 7: (0.0, 48.0), 8: (0.0, 56.0), 9: (0.0, 96.0)}
    _a0 = (520.0, 300.0)

    def _run(off9):
        _t = MotionTracker(min_hits=2)
        _rows = []
        for _f in range(1, 10):
            _cam = _step[_f]
            _d = _shift(_frame(3), _cam[0], _cam[1])
            if _f <= 3 or _f >= 9:                     # ⚠ 帧 4~8：A **掉线** ✓
                _ox = 0.0 if _f != 9 else float(off9)
                _d = _d + [(0, _a0[0] + _cam[0] + _ox, _a0[1] + _cam[1], 150.0, 150.0, 0.9)]
            _t.process(None, ts=_f * 0.16, dets=_d)
            _a = _t._by_id(4)                          # 3 条大部队 ⇒ A = **#4** ✓
            _rows.append((_f, _a, float(_t._dev_med),
                          getattr(_t, "_median_mv", (0.0, 0.0))))
        return _rows

    _brick = _run(0.0)
    _f9 = next(_r for _r in _brick if _r[0] == 9)
    check(_f9[1] is not None and int(_f9[1].miss) == 0 and float(_f9[1].dev) <= 0.01,
          "① **砖接回来那拍 `dev` = %.2f** ✓（实测帧 9；⚠ 它断了 5 拍 ✓、`mv` = (%.1f,%.1f) ✓ "
          "`mv_span` 归 1 ✓）—— ⚠ 老口径会拿它减「当拍瞬时中位」⇒ **%.1f px** ✗（过门 ⇒ 假"
          "「不合群」⇒ 抢走目标 ✗✗）"
          % (float(_f9[1].dev) if _f9[1] is not None else -1.0,
             float(_f9[1].mv[0]) if _f9[1] is not None else -1.0,
             float(_f9[1].mv[1]) if _f9[1] is not None else -1.0,
             (math.hypot(float(_f9[1].mv[0]) - float(_f9[3][0]),
                         float(_f9[1].mv[1]) - float(_f9[3][1]))
              if _f9[1] is not None and _f9[3] is not None else -1.0)))
    _mv = _run(60.0)
    _g9 = next(_r for _r in _mv if _r[0] == 9)
    _gate = 2.0 * max(1.0, float(_g9[2]))
    check(_g9[1] is not None and float(_g9[1].dev) > _gate,
          "② **反面：真的在动 ⇒ 照旧报出来** ✓（把帧 9 那格框**挪开 60px** ✗ —— ⚠⚠ **必须仍落在"
          "配对门限（%.0fpx）之内** ✓：`dev` = **%.2f px** > 门 **%.2f** ✓ —— ⚠ 一刀切"
          "「跨拍回来一律记 0」⇒ 这条红 ✗）"
          % (_PAIR_GATE, float(_g9[1].dev) if _g9[1] is not None else -1.0, _gate))
    # ⭐⭐⭐⭐⭐ **③ 远接（超出配对门限）⇒ `dev` 记 0**（用户 2026-10-05 ✓ 原话："**因为假目标是
    #   位于理想中的假目标检出框框心的，它不该有任何超过噪声允许的位移，有就是你算错了**" ✓✓）
    #   —— ⚠⚠ **这条是 ② 的边界**：砖的预测**本来就准** ✓（`vr ≈ 0` ✓）⇒ 一格框离它 70px 以上
    #   ⇒ **那不是它的框** ✓ ⇒ 不许采信 ✓（实测真素材帧 31：`#9` 认了 70.5px 外的框之后
    #   `dev` 一路 11.1 → 10.5 → 21.4 → 29.3 ✗✗，而它是一块砖 ✓）。
    _fr = _run(120.0)
    _f9 = next(_r for _r in _fr if _r[0] == 9)
    check(_f9[1] is not None and float(_f9[1].dev) <= 0.01 and int(_f9[1].miss) == 0,
          "③ **远接（> 配对门限 %.0fpx）⇒ `dev` = %.2f（要 0 ✓）、但 `miss` = %s（要 0 ✓ = 保号 ✓）**"
          "（= 只保号、不当观测 ✓ —— ⚠ 退回「照实算」⇒ 这条红 ✗✗；实测真素材那条链正是"
          " 11.1 → 10.5 → 21.4 → 29.3 ✗✗）"
          % (_PAIR_GATE, float(_f9[1].dev) if _f9[1] is not None else -1.0,
             "-" if _f9[1] is None else int(_f9[1].miss)))
    import inspect
    _src = inspect.getsource(MotionTracker.process).replace(" ", "")
    check("if_cobs_isnotNoneand_spn>1:" in _src or "_spn>1" in _src,
          "③ **不变式（源码级）**：新支**只在 `mv_span > 1`（= 真的断了拍）时接管** ✓ ⇒ 正常连续拍"
          "**逐位不变** ✓（⚠ 谁能把这句删掉/改成无条件 ⇒ 立刻红 ✗）")


def test_circle_intersect_box_position_trusted():
    """⭐⭐⭐⭐⭐ **罩着圆的框，位置就该算"离真目标近"**（用户 2026-10-05 ✓ 原话："**他都跟圆相交了，
    它应该被认为"离预测真目标近"**" ✓✓ ｜ "**说明你的匹配分算的有问题**" ✓✓）。

    ⚠⚠ **病根（实测 `10月3日 (2)` 前 76 拍 ✓）**：`匹配分 = 距离因子 × 面积因子`，而那个
      **距离因子**量的是"**离『这条轨迹』的预测多远**" ✗（一条轨迹一个数 ✗）—— 可面板上写的是
      "位置可疑（**偏离预测**较多）⇒ 已降权"，读起来像在说"离**真目标**远" ✗✗。
      实测后果：**跟圆相交的框 53 个里有 40 个（75%）**被判"位置可疑" ✗ —— 而"圆"才是
      "**我们相信真目标在哪**" ✓ ⇒ 一格罩着圆的框，凭什么因为"离某条轨迹的预测远"被降权 ✗。
    ⇒ 修法（见 `process` 里那段 ✓，与 R4「圈归谁」**逐字同源** ✓）：**框含着圆心 ⇒ 距离因子记 1**
      ✓（`self.pos` 此刻 = **上一拍报出的圆** ✓ 与 R4 同一时刻 ✓）。
      ⚠⚠ **面积因子一个字不动** ✗（融合框罩着圆是**常态** ✓ ⇒ 它的框心是中点 ✗ ⇒ 照样 `m = 0` ✓）。

    夹具（合成 ✓ **先量后断言** ✓）：3 条大部队框 ＋ 一条真目标（**跟着相机走、再自己 +7,+3 每拍**
      ✓ = 唯一"不合群"的 ✓ ⇒ 座位在它身上 ✓）＋ **一条砖 S**（帧 1~3 在**别处** ✓、帧 4~8 掉线 ✓、
      帧 9 带偏移回来 ✓）。
      ⚠⚠ **为什么非要 S**（**踩过** ✗）：我第一版让"真目标自己掉线再回来" ✗ ⇒ 量出来**根本没咬住** ✗
        —— 因为**位子上那条轨迹的滑行带着它自己的相对速度** ✓ ⇒ 预测跟得住 ⇒ 回来时离预测只有
        几十 px ✗（老口径 `m` 就已经 ≥ 0.7 ✗）⇒ 钉不到这条 ✓。⇒ 换成"**别处那条砖**"：它的 `vr ≈ 0`
        ⇒ 滑行只跟大部队 ⇒ **预测落后一大截** ✓ ⇒ 回来那格框**又远、又能罩着圆** ✓✓
        （实测 `(-80,-80)` 那组：框心 (496,244) 罩住圆 (457,285) ✓、老口径 `m = 0.00` ✗、新口径 1.00 ✓）。
    钉三条（一正一反 ＋ 反面保险 ✓）：
      ① **含圆 ⇒ `m ≥ _M_OK`** ✓（⚠ 老口径 **0.00** ⇒ 这条红 ✗ = 用户看到的那条 ✗）；
      ② **反面：把回来的框挪出圆 120px ⇒ `m < _M_SUS`** ✓（老行为照旧 ✓ —— ⚠ 一刀切"啥都记 1"
         ⇒ 这条红 ✗）；
      ③ **保险：融合大框（210×210 ≈ 1.96× 标准）罩着圆 ⇒ `m` 仍是 0** ✓、`st=1` ✓
         （⚠ 谁把面积因子也"顺手饶了" ⇒ 这条红 ✗✗ —— 融合框的框心是**两个目标的中点** ✓）。
    """
    _t0 = (375.0, 250.0)                       # 真目标（**唯一"不合群"的** ✓ ⇒ 座位在它身上 ✓）
    _s0 = (520.0, 300.0)                       # 砖 S：帧 1~3 在**别处** ✓、帧 4~8 掉线 ✓、帧 9 带偏移回来 ✓

    def _run(s_off=(0.0, 0.0), big=False, upto=9):
        """⚠ `upto` = 跑到第几拍**停** ✓（**踩过两次** ✗：跑满再看"回来那一拍" ⇒ 看的是**别的拍** ✗）。

        返回 `(tracker, **处理前**那个圆, 砖 S 的 id, S 回来那格框的中心)` ✓ ——
        ⚠ "**那格框含不含圆**"是**前提** ✓，必须和断言一起钉 ✗（不然夹具哪天不咬人了也没人知道 ✗）。
        """
        _t = MotionTracker(min_hits=2)
        _pre, _sid = None, None
        for _i in range(int(upto)):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_frame(3), _cam[0], _cam[1])
            if not (5 <= _i <= 7):              # ⚠ 帧 6~8：真目标**掉线** ✓（圆靠滑行继续走 ✓）
                _sz = 210.0 if (big and _i == 8) else 150.0
                _d = _d + [(0, _t0[0] + _cam[0] + 7.0 * _i, _t0[1] + _cam[1] + 3.0 * _i,
                            _sz, _sz, 0.9)]
            if _i <= 2 or _i == 8:              # ⚠ 砖 S：帧 4~8 掉线 ✓
                _ox = float(s_off[0]) if _i == 8 else 0.0
                _oy = float(s_off[1]) if _i == 8 else 0.0
                _d = _d + [(0, _s0[0] + _cam[0] + _ox, _s0[1] + _cam[1] + _oy,
                            150.0, 150.0, 0.9)]
            _pre = None if _t.pos is None else (float(_t.pos[0]), float(_t.pos[1]))
            _t.process(None, ts=_i * 0.16, dets=_d)
            if _i == 2:                        # 帧 3：认下"谁是 S" ✓（按它那格框的位置 ✓）
                _sid = next((int(_x.id) for _x in _t.tracks
                             if math.hypot(float(_x.obs[0]) - (_s0[0] + _cam[0]),
                                           float(_x.obs[1]) - (_s0[1] + _cam[1])) < 5.0), None)
        _last = int(upto) - 1
        return (_t, _pre, _sid,
                (float(_s0[0]) + 7.0 * _last + float(s_off[0]),
                 float(_s0[1]) + 3.0 * _last + float(s_off[1])))

    def _m_of(_t, _x_id):
        """⚠⚠ **必须从"轨迹对象"上读 `m`** ✗（`_t.m` ✓），**`box_v` 不行** ✗✗：① 那笔是"**远接**"
        （离预测 > 70px ✓）⇒ 按 R5 那格框**不贴编号** ✓ ⇒ `box_v` 里**查不到** ✓（我第一版拿
        `box_v` 读 ⇒ 读到 `None` ⇒ 假红 ✗）。
        """
        _x = None if _x_id is None else _t._by_id(int(_x_id))
        return None if _x is None else float(_x.m)

    def _tgt_m(_t):
        _x = _t._by_id(_t.tid) if _t.tid is not None else None
        return None if _x is None else float(_x.m)

    def _tgt_box(_t):
        return next((_b for _b in (getattr(_t, "_box_v", []) or [])
                     if _t.tid is not None and int(_b[4]) == int(_t.tid)), None)

    # ① 砖 S 回来那格框**挪 (-80,-80)** ⇒ 实测**正好罩着圆** ✓、老口径 `m = 0.00` ✗
    _ta, _ca, _sa, _ba = _run((-80.0, -80.0))
    _ina = (_ca is not None and abs(_ba[0] - _ca[0]) <= 75.0 and abs(_ba[1] - _ca[1]) <= 75.0)
    _ma = _m_of(_ta, _sa)
    check(_ina and _ma is not None and _ma >= _M_OK,
          "① **含圆的框 ⇒ `m` = %.2f ≥ `_M_OK`(%.2f) ✓**（前提：那格框**确实罩着圆** = %s ✓"
          "〔圆 %s / 框心 (%.0f,%.0f) ✓〕）—— ⚠ 老口径按\"离**那条轨迹**的预测\"算 ⇒ **0.00** ✗"
          " ⇒ 面板写\"位置可疑（偏离预测较多）⇒ 已降权\" ✗✗（用户抓的就是这种 ✗）"
          % (-1.0 if _ma is None else _ma, _M_OK, _ina,
             "-" if _ca is None else "(%.0f,%.0f)" % _ca, _ba[0], _ba[1]))
    # ② 反面：同样"远接"，但把框**挪出圆** ⇒ 距离因子照旧说话 ⇒ `m` 低 ✓（老行为 ✓）
    _tb, _cb, _sb, _bb = _run((0.0, -100.0))
    _inb = (_cb is not None and abs(_bb[0] - _cb[0]) <= 75.0 and abs(_bb[1] - _cb[1]) <= 75.0)
    _mb = _m_of(_tb, _sb)
    check((not _inb) and _mb is not None and _mb < _M_SUS,
          "② **反面：同一条砖、同一格远框，但**挪出圆** ⇒ `m` = %.2f < `_M_SUS`(%.2f) ✓**"
          "（含圆 = %s ✓〔要 `False` ✓〕—— ⚠ 一刀切「凡远接都记 1」⇒ 这条红 ✗）"
          % (-1.0 if _mb is None else _mb, _M_SUS, _inb))
    _tc, _cc, _sc, _bcx = _run(big=True)
    _bc = _tgt_box(_tc)
    check(_tgt_m(_tc) is not None and float(_tgt_m(_tc)) <= 0.01
          and _bc is not None and int(_bc[7]) == 1,
          "③ **保险：融合大框（210×210 ≈ 1.96×）罩着圆 ⇒ `m` = %.2f（要 ≈0 ✓）、`st` = %s（要 1 ✓）**"
          " —— ⚠ 谁把**面积因子**也一起饶了 ⇒ 这条红 ✗✗（融合框的框心是**两个目标的中点** ✓"
          " 位置还是不能信 ✓）"
          % (float(_bc[6]) if _bc is not None else -1.0,
             int(_bc[7]) if _bc is not None else "-"))


def test_real_clip_circle_intersect_box_trusted():
    """⭐⭐⭐⭐⭐ **真素材（性质）**：**罩着圆的框** ⇒ `m` 恒等于「面积因子」（= 距离因子不许再压）✓。

    ⚠⚠ 钉的是**不变式**，不是某一帧 ✗（同 `test_real_clip_gap_reentry` 那条纪律 ✓）：
      · "罩着圆" = 框心落在框内那套装法 ✓（与 R4 / 判据**同一把尺** ✓），圆取**处理前**那份 ✓
        （= 程序判定时用的那个 ✓）；
      · **性质** = `m ≥ 面积因子`（`ma` ✓，就在**这格框归的那条轨迹**身上 ✓）——
        `m = 距离因子 × 面积因子` ⇒ 恒有 `m ≤ 面积因子` ✓ ⇒ **一旦 `m < ma`，就是"距离因子在压它"**
        ✗✗ = 用户抓的那条 ✓✓。
      ⚠⚠ **为什么不卡 `_M_OK`**（**踩过** ✗）：本素材**干净框的面积普遍是 `std_area` 的 1.15~1.35 倍**
        ✗ ⇒ 面积因子本来就 ≈ 0.5 ⇒ 卡 `_M_OK` 会**无论改不改都红** ✗（我第一版就是这么写的 ✓）——
        而那条**不是**本轮的病 ✓（面积那半**故意不动** ✓）。
      实测（前 40 拍 ✓）：改前**有违规** ✗（那批框按"离某条轨迹的预测远"又被压了一道 ✗）；
      改后 **0 违规** ✓。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月3日 (2).mp4")
    if not _src.exists():
        check(True, "跳过「罩着圆的框」真素材用例（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        from tools.live_lie import load_motion_cfg
        _frames = LD.load_video(str(_src))[0]
        _w = LD.DetsWorker(None, conf=0.25)
        _kw = load_motion_cfg()
    except Exception as _e:                    # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过「罩着圆的框」真素材用例（环境不齐：%s ✓）" % _e)
        return
    _t = MotionTracker(**_kw)
    _n, _seen, _bad = min(40, len(_frames)), 0, []
    for _i in range(_n):
        _im = _frames[_i][1]
        _t.frame_wh = (int(_im.shape[1]), int(_im.shape[0]))
        _pre = None if _t.pos is None else (float(_t.pos[0]), float(_t.pos[1]))
        _t.process(_im, ts=_i * 0.16, dets=_w.detect(_im) or [])
        if _pre is None:
            continue
        for _b in (getattr(_t, "_box_v", []) or []):
            # ⚠⚠ **跳过"推算条目"** ✗✗（用户 2026-10-06 ✓ 见 `process` 末尾那段 ✓）：那种条目
            #   是"**这格没配上框、按旁边那条账贴的号**" ✓ ⇒ `m` 记 **−1**（**没有匹配分这回事** ✓）
            #   ⇒ 它会**假红**这条"罩着圆的框 `m` 要 ≥ `_M_OK`"的钉子 ✓（**实测踩到** ✓ 1 笔 ✓）。
            #   ⚠⚠ **改判据：`m < 0` 一律跳过** ✗✗（**实测踩到** ✓：同一次运行里冒出一条
            #     `(帧 14, 标签 14, m = −1.0, 面积因子 1.0)` ✗ —— 它是**哨兵值 −1**（"没有匹配分
            #     这回事"）✓，靠"长度 > 12 且第 13 位为真"去跳**漏了它** ✗）⇒ 直接按**语义**跳 ✓：
            #     **`m` 是负数 = 不是一次真配对** ✓（配上过的框 `m ≥ 0` ✓）。
            if float(_b[6]) < 0.0:
                continue
            _bw, _bh = float(_b[10]), float(_b[11])
            if not (abs(float(_b[0]) - _pre[0]) <= _bw / 2.0
                    and abs(float(_b[1]) - _pre[1]) <= _bh / 2.0):
                continue                       # 不含圆 ⇒ 不在本条管辖 ✓
            _x = _t._by_id(int(_b[4]))         # ⚠ 这格框归的那条轨迹（**面积因子在它身上** ✓）
            if _x is None:
                continue
            _seen += 1
            # ⚠⚠ **钉的就是这句**：`m = 距离因子 × 面积因子` ⇒ 恒有 `m ≤ 面积因子(= ma)` ✓ ——
            #   所以 `m < ma` 就说明"**距离因子又把它往下压了**" ✗ ⇒ 正是用户抓的那条 ✗✗。
            if float(_b[6]) + 1e-6 < float(getattr(_x, "ma", 0.0)):
                _bad.append((_i + 1, int(_b[4]), round(float(_b[6]), 2),
                             round(float(getattr(_x, "ma", 0.0)), 2)))
    check(not _bad and _seen >= 3,
          "**罩着圆的框：`m` 恒等于「面积因子」（= 距离因子不再往下压）** ✓（前 %d 拍里这种框 "
          "**%d** 个 ✓〔要 ≥3 ✓〕；违规 **%d** 个 ✓〔要 0 ✓〕）｜ 违规长这样：`(帧, 标签, m, 面积因子)` = %s"
          " —— ⚠ 改前这一批正是用户抓的那条 ✗（面板写\"位置可疑（偏离预测较多）⇒ 已降权\" ✗，"
          "可它跟真目标的圆是相交的 ✓）"
          % (_n, _seen, len(_bad), _bad[:4] if _bad else "（无 ✓）"))


def test_far_revival_not_labeled():
    """⭐⭐⭐⭐⭐ **"远接"的那格框不贴编号**（用户 2026-10-05 ✓ 原话："**明显该框跟 #9 的灰框差距
    特别大，不该被标上 #9，反而应该被标 #5（真目标）**" ✓✓ ｜ "**直到 35 帧，真目标都拥有独立的
    检出框**" ✓）。

    ⚠⚠ 实测病根（`10月3日 (2)` ✓ **在配对那一刻抓的** ✓）：帧 31 那格框 (198.9,251.2) 120×161
      到 `#9` 的预测 **70.5px** ✗ ⇒ 只有"**睡着的**" `#9`（`miss=6` ✓）接得住它（**复活门 161px** ✓）；
      而**醒着的** `#5` 离它 **77px** ⇒ 走**严门 70px** ✗ ⇒ **差 7px 判不上** ✗✗
      ⇒ **同一格框被两把尺子量** ✓；可那格框上却写着 `匹配分 0.00` = **程序自己都说"位置不可信"**
      ✗ ⇒ **照样给它贴了 `#9` 的编号** ✗ = 用户抓的那条 ✓。
    ⇒ 修法（**只改"归属怎么说"，证据链一个字不动** ✗ 见 `process` R5 那段 ✓）：**远接那几格框不贴
      编号** ✓（演示窗查不到条目 ⇒ 按"未配对"显示 ✓）。
    钉三条（一正一反 ＋ 保号 ✓）：
      ① **远接（> 配对门限）⇒ 那格框没有编号** ✓（`box_v` 里查不到它 ✓；⚠ **改前是有编号的**
         ⇒ 这条立刻红 ✗ —— 用户看到的就是那个 ✗）；
      ② **反面：近距离复活 ⇒ 照旧有编号** ✓（框就落在预测旁 ✓ ⇒ `box_v` 里有它 ✓、`trust=True` ✓
         —— ⚠ 一刀切"复活一律不贴编号" ⇒ 这条红 ✗）；
      ③ **保号**：远接那拍 `miss` 归 0 ✓（= id 不抖、不被 `_TRACK_LOST` 销号 ✓ —— 这正是
         `_REVIVE_K` 当初存在的理由 ✓ 不许碰 ✗）。
    ⚠ 夹具与 `test_gap_reentry_dev_uses_crowd_baseline` **同一套**（3 条大部队 ＋ 一条砖 `#4` ✓
      相机帧 9 猛地快一下 ⇒ "回来那拍"离预测 ~72px ✓ = 恰好越过配对门 ✓ 确定性 ✓）。
    """
    _a0 = (520.0, 300.0)
    _step = {_f: (0.0, 8.0 * _f) for _f in range(1, 10)}
    _step[9] = (0.0, 96.0)                     # ⚠ 帧 9 大部队猛快一下（同那条夹具 ✓）

    def _run(back_at, off=0.0, upto=9):
        """`off` = 回来那拍**把框挪开多少 px** ✓（⚠ 我第一版**没挪** ⇒ 实测**根本造不出"远接"**
        ✗：滑行支会**跟着大部队一起推** ✓ ⇒ 断了 5 拍回来也只差 **16px** ✗ ⇒ 得自己挪 ✓）；
        `upto` = 跑到第几拍**停** ✓（⚠ **踩过** ✗：跑满 9 拍再看"第 5 拍有没有编号" ⇒ 那时砖
        **又掉线了** ⇒ 断言必假红 ✗✗）。
        """
        _t = MotionTracker(min_hits=2)
        for _f in range(1, int(upto) + 1):
            _cam = _step[_f]
            _d = _shift(_frame(3), _cam[0], _cam[1])
            if _f <= 3 or _f == back_at:       # ⚠ 帧 4~8：砖 **掉线** ✓
                _ox = float(off) if _f == back_at else 0.0
                _d = _d + [(0, _a0[0] + _cam[0] + _ox, _a0[1] + _cam[1], 150.0, 150.0, 0.9)]
            _t.process(None, ts=_f * 0.16, dets=_d)
        return _t

    _far = _run(9, 100.0)                      # 回来时**离预测 ~84px**（> 配对门 70 ✓ = "远接" ✓）
    _a = _far._by_id(4)
    _has = any(int(_v[4]) == 4 for _v in (_far._box_v or []))
    _bx, _by = _a0[0] + 100.0, _a0[1] + 96.0   # 那格框的中心（帧 9：相机 (0,96) ＋ 偏移 (100,0) ✓）
    _gap = -1.0 if _a is None else math.hypot(float(_a.obs[0]) - _bx,
                                              float(_a.obs[1]) - _by)
    check(_a is not None and int(_a.miss) == 0 and not _has
          and 4 in (getattr(_far, "_low_conf_tids", None) or set())
          and float(_a.dev) <= 0.01 and _gap > 30.0,
          "① **远接：不贴编号 ＋ 不当观测 ＋ 保号** ✓（`box_v` 里有它的条目 = **%s** ✓〔要 `False` ✓〕；"
          "`_low_conf_tids` = %s ✓；`miss` = %s ✓〔要 0 ✓ = 保号 ✓〕；**`dev` = %.2f** ✓〔要 0 ✓〕；"
          "**它的 `obs` 离那格框 %.0fpx** ✓〔要 > 30 ⇒ 没被拽过去 ✓〕）"
          " —— ⚠ **改前**：既贴编号 ✗（用户抓的 ✓）又 `dev` 记成十几 ✗（用户第二句 ✓："
          "**砖不该有超过噪声的位移** ✓）"
          % (_has, sorted(int(x) for x in (getattr(_far, "_low_conf_tids", None) or set())),
             "-" if _a is None else int(_a.miss),
             -1.0 if _a is None else float(_a.dev), _gap))

    _near = _run(5, upto=5)                    # 只断 1 拍，回来时离预测只有 ~16px ✓（⚠ 跑到第 5 拍**停** ✓）
    _b = _near._by_id(4)
    _has2 = any(int(_v[4]) == 4 for _v in (_near._box_v or []))
    check(_b is not None and _has2 and bool(getattr(_b, "obs_trust", False)),
          "② **反面：近距离复活 ⇒ 照旧有编号** ✓（`box_v` 里有它 = **%s** ✓〔要 `True` ✓〕、"
          "`trust` = %s ✓〔要 `True` ✓〕）—— ⚠ 一刀切「复活一律不贴编号」⇒ 这条红 ✗"
          % (_has2, "-" if _b is None else bool(getattr(_b, "obs_trust", False))))


def test_real_clip_far_revival_not_labeled():
    """⭐⭐⭐⭐⭐ **真素材："远接"那几拍，那格框一个编号都不许贴**（用户 2026-10-05 ✓ 你报的
    帧 31/70 ✓）。

    ⚠⚠ **钉的是「性质」，不是「某一帧」** ✗✗（**踩过** ✓ 见 `test_real_clip_gap_reentry` 那段）：
      早先那类"钉帧 25 / 帧 31 那笔"的夹具，**门值一改走势就变** ⇒ 钉子全红 ✗（不是代码坏 ✓
      是夹具跟走势绑死 ✗）。⇒ 这里只扫**不变式**：
        **凡是当拍进了 `_low_conf_tids`（= 复活轮接到远处的框 ✓）的轨迹，
          `box_v` 里都不许有它的条目** ✓（= 它的编号不许贴到那格框上 ✓）。
      ⚠ 全程扫 ⇒ **与走势无关** ✓：哪个 id、哪一帧都不看 ✓，只看"这两者不许同时出现" ✓。
      ⚠ 顺带报一下扫到几拍（= 本素材真的走到这条路的次数 ✓）：**0 拍 ⇒ 算不上证明** ✗
        （所以下面要 `≥ 1` ✓）。
    ⚠ 真素材用例 ⇒ 找不到文件 / 环境不齐就**跳过** ✓（同 `test_real_clip_smoke` ✓）。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月3日 (2).mp4")
    if not _src.exists():
        check(True, "跳过远接不贴编号（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        from tools.live_lie import load_motion_cfg
        _frames = LD.load_video(str(_src))[0]
        _w = LD.DetsWorker(None, conf=0.25)
        _kw = load_motion_cfg()
    except Exception as _e:                    # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过远接不贴编号（环境不齐：%s ✓）" % _e)
        return
    _t = MotionTracker(**_kw)
    # ⚠⚠⚠ **窗口不能太短** ✗✗（**踩过** ✓）：前 40 拍里"远接"原来有 5 次 ✓，如今规则/夹具一改
    #   就成了 **0 次** ✗ ⇒ 这条钉子**自己要求 `≥1`** ⇒ 于是**假红** ✗（行为没坏 ✓ —— 只是
    #   前 40 拍没走到那条路 ✓）。⇒ 扫**全片**（76 拍 ✓）：⚠ 与 `test_brick_vr_always_zero`
    #   那条的 40 拍窗口**不必一致** ✗（它钉的是"每条非座位 `vr` 恒 0"，窗口长短不影响 ✓）。
    _n, _bad, _seen, _baddev = min(76, len(_frames)), [], [], []
    for _i in range(_n):
        _im = _frames[_i][1]
        _t.frame_wh = (int(_im.shape[1]), int(_im.shape[0]))
        _t.process(_im, ts=_i * 0.16, dets=_w.detect(_im) or [])
        _lc = set(int(x) for x in (getattr(_t, "_low_conf_tids", None) or set()))
        # ⚠⚠ **座位那条要排除** ✗（用户 2026-10-05 ✓："**反而应该被标 #5（真目标）**" ✓✓）——
        #   规则是"**只放行位子上那位**" ✓：它远接的可能是它自己的框 ✓ ⇒ 照旧贴编号、照旧算 `dev` ✓；
        #   本条钉的是**砖**（别人）那半边 ✓ ✓。
        if _t.tid is not None:
            _lc.discard(int(_t.tid))
        if _lc:
            _hit = sorted(int(_v[4]) for _v in (getattr(_t, "_box_v", []) or [])
                          if int(_v[4]) in _lc)
            _seen.append((_i + 1, sorted(_lc), _hit))
            if _hit:
                _bad.append((_i + 1, _hit))
            # ⭐⭐⭐⭐⭐ **用户 2026-10-05 第二句**（"**因为假目标是位于理想中的假目标检出框框心的，
            #   它不该有任何超过噪声允许的位移，有就是你算错了**" ✓✓）：远接那几拍，
            #   那几条轨迹的 `dev` **必须是 0** ✓（⚠ 改前：帧 31 记 **11.1** ✗、随后一路
            #   **10.5 → 21.4 → 29.3** ✗✗ = 用户报的那条 ✓）。
            for _x in _t.tracks:
                if int(_x.id) in _lc and abs(float(getattr(_x, "dev", 0.0))) > 1e-6:
                    _baddev.append((_i + 1, int(_x.id), round(float(_x.dev), 2)))
    check(not _bad and not _baddev and len(_seen) >= 1,
          "**远接那几拍：那格框一个编号都没贴 ✓、那几条轨迹的 `dev` 也全是 0** ✓"
          "（前 %d 拍里走到「远接」的 **%d** 拍 ✓〔要 ≥1 ✓〕；"
          "编号违规 **%d** 笔 ✓、`dev` 违规 **%d** 笔 ✓〔都要 0 ✓〕）｜ `dev` 违规长这样："
          "`(帧, id, dev)` = %s —— ⚠ 改前正是帧 31 的 **11.1** 开头那串 ✗✗（砖本来一寸都不该动 ✓，"
          "可它把真目标那格框认成了自己的 ✗）"
          % (_n, len(_seen), len(_bad), len(_baddev),
             _baddev[:3] if _baddev else (_bad[:3] if _bad else _seen[:3])))


def test_real_clip_gap_reentry():
    """⭐⭐⭐⭐⭐ **真素材：认领的"该认的认上、不该认的别认"**（用户 2026-10-05 ✓ 你贴的两条日志 ✓）。

    ⚠⚠ **钉的是"性质"，不是"某一帧"** ✗（**踩过** ✓）：早先这条钉"帧 25 那笔幽灵的 `dev` ≤ 门" ✓、
      "帧 31 那笔正当认领必须在" ✓ —— 可**门值一改（0.30 → 0.55 ✓）整段走势就变了** ✗✗
      ⇒ 那几笔**根本不出现了** ⇒ 钉子全红 ✗（**不是**代码坏了 ✓ 是夹具**跟走势绑死**了 ✗）。
    ⇒ 现在只钉**与走势无关**的东西 ✓（机制本身由**合成**那条 `test_gap_reentry_dev_uses_crowd_baseline`
      钉死 ✓ 确定性 ✓、不依赖素材 ✓）：
        ④ **帧 28 那笔"认了个连着跟的框"不许再有** ✓（`#3` 连跟 27 拍、分只 1.72 = 砖 ✗）；
        ⑤ **门值 `_CLAIM_UNREC_IOU` ≥ 0.45** ✓（邻居 IoU 实测到 0.44 ✗ ⇒ 0.30 会冤杀 ✓）；
        ⑥ **帧 14 那笔"位置可疑但面积正常、分很高"的必须认上** ✓（你报的 ✓ —— 它原来被
           `st == 0` 那道门刷掉 ✗，现在只排除融合框 ✓）。
    ⚠⚠ **"帧 45/46 那笔必须认上"这条我撤了** ✗（如实说 ✓）：帧 14 一改（目标回到 `#5` ✓）⇒
      帧 46 那笔"**症状**"自己就没了 ✓（上游好了 ⇒ 那格框跟着正确轨迹走 ⇒ 不需要认领 ✓）
      ⇒ 钉它 = 钉一个**已经不存在**的场面 ✗（一改门值就翻面 ✓ 我踩过两次 ✓）。
    ⚠⚠ **用的是你自己 `ui.yaml` 那套参数** ✓（`load_motion_cfg` ✓ —— 换参数会换场面 ✓）。
    ⚠ 真素材用例 ⇒ 找不到文件 / 环境不齐就**跳过** ✓（同 `test_real_clip_smoke` ✓ 那条纪律 ✓）。
    """
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月3日 (2).mp4")
    if not _src.exists():
        check(True, "跳过真素材认领用例（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        from tools.live_lie import load_motion_cfg
        _frames = LD.load_video(str(_src))[0]
        _w = LD.DetsWorker(None, conf=0.25)
        _kw = load_motion_cfg()
    except Exception as _e:                 # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过真素材认领用例（环境不齐：%s ✓）" % _e)
        return
    _t = MotionTracker(**_kw)
    _claims = []                            # `[帧, 换法, 认来的 id, 事后 5 拍 dev, 噪声底]`
    _inc = {}                               # 帧 ⇒ 当时**现任**（旧目标）的 `(ma, dev)` ✓
    for _i in range(min(50, len(_frames))):
        _im = _frames[_i][1]
        # ⚠ `MotionRunner.step` 每拍会刷这一句 ✗（只影响「贴边 / 半个框」那两条判据 ✓）
        #   ⇒ 裸 `MotionTracker` 得自己补 ✓ 否则跑的是**另一个场面** ✗（我踩过 ✓）。
        if getattr(_im, "shape", None) is not None:
            _t.frame_wh = (int(_im.shape[1]), int(_im.shape[0]))
        _old = _t.tid
        _t.process(_im, ts=_i * 0.16, dets=_w.detect(_im) or [])
        _k = getattr(_t, "_switch_kind", "") or ""
        # ⚠ 现任那两项要取**这一拍处理完之后**的值 ✓（门就是在那一刻看的 ✓；旧目标的 `ma/dev`
        #   不会被认领本身改动 ✓）。
        _o = _t._by_id(int(_old)) if _old is not None else None
        _inc[_i + 1] = (None if _o is None else
                        (float(_o.ma), float(_o.dev), float(_o.score)))
        if _k in ("forced", "claim") and _t.tid is not None:
            _claims.append([_i + 1, _k, int(_t.tid), [], float(_t._dev_med)])
            _claims[-1].append(_inc[_i + 1])                 # ⚠ 第 6 位 = 当时的现任 ✓
            _c = _t._by_id(int(_t.tid))
            _claims[-1].append((None if _c is None else          # 第 7 位 = 认来的那条 ✓
                                (float(_c.score), int(_c.hits))))
        for _c in _claims[-2:]:                                  # ⚠ 事后 5 拍 ✓
            _x = _t._by_id(_c[2])
            if _x is not None and len(_c[3]) < 5:
                _c[3].append(float(_x.dev))
    _w.close()
    _f28 = [c for c in _claims if c[0] == 28]
    check(not _f28,
          "④ **帧 28 那笔「认了一个连着跟的框」没有了** ✓（实测帧 28 的认领 = %s ✓ ｜ 前 50 拍全部"
          "认领 = %s ✓）—— ⚠ 老口径帧 28 会把 `#3`（**连跟 27 拍** ✓、分只有 **1.72** = 砖 ✗）认成"
          "目标 ⇒ 圆被**硬搬 200px** ✗✗"
          % (_f28 or "[]", [(c[0], c[1], c[2]) for c in _claims]))
    # ⚠⚠ **"帧 45/46 那笔必须认上"这条我撤了** ✗（2026-10-05 ✓ 如实说）：帧 14 一改（目标回到
    #   `#5` ✓）⇒ **帧 46 那笔"症状"自己就没了** ✓（上游好了、那格框跟着正确轨迹走 ⇒ 不需要认领 ✓）
    #   ⇒ 钉它 = 钉一个**已经不存在**的场面 ✗（而且一改门值它就翻面 ✓ 我踩过两次 ✓）。
    #   ⇒ 改成钉**门值本身**（与走势无关 ✓）：邻居框的 IoU 实测能到 **0.44** ✗ ⇒ 门必须 > 它 ✓。
    _iou = float(getattr(_LM, "_CLAIM_UNREC_IOU", -1.0))
    check(_iou >= 0.45,
          "⑤ **「未被记录」的门值必须 ≥ 0.45** ✓（实测 %s ✓ —— 用户 2026-10-05 那笔冤杀里，"
          "**仅仅挨着**的四条模拟框 IoU 达 **0.32 / 0.34 / 0.35 / 0.44** ✗ ⇒ 门 0.30 会把**任何"
          "挤在一起的新框**都冤杀 ✓；而「**就是原来那一格**」该是 0.7 以上 ✓ ⇒ 0.45~0.7 之间 ✓）"
          % (_iou,))
    # ⭐⭐⭐⭐⭐⭐ **现任保护（用户 2026-10-05 ✓ 他挑的 (乙) ✓ 见 `_incumbent_strong` ✓）**：
    #   ⚠⚠ "帧 14 那笔必须认上"这条**又撤了** ✗（如实说 ✓）：帧 10 加了现任保护 ⇒ **冤枉换人不发生** ✓
    #     ⇒ 目标一直留在 `#5` ✓ ⇒ 帧 14 那格框**本来就归目标** ✓ ⇒ **根本不需要认领** ✓（症状消失 ✓）。
    #   ⇒ 改成钉**不变量**（与走势无关 ✓）：**凡是"现任还健康"的时候，一笔认领都不许发生** ✓。
    _bad = [c for c in _claims
            if len(c) > 5 and c[5] is not None and float(c[5][0]) >= float(_LM._M_OK)
            and float(c[5][1]) >= 2.0 * max(1.0, float(c[4]))]
    check(not _bad,
          "⑥ **现任健康时一笔认领都没有** ✓（现任保护 ✓：现任那格框 `ma ≥ %.2f` ✓ **且** 它这一拍自己"
          "还在不合群（`dev ≥ 2 × 噪声底` ✓）⇒ 不许抢 ✗ ｜ 实测违规 **%d 笔**（要 0 ✓）｜ 全部认领 = %s）"
          "—— ⚠ 没这条门时：`10月3日 (2)` **帧 10** 会用分 **11.4** 的弱候选（`dev` 刚过门 ✓）把分 "
          "**156.6**、`ma = 1.00`、`dev = 9.4` 的现任 `#5`（= 开局白块认下的**真目标** ✓）赶走 ✗✗"
          " ⇒ 后面整条链全歪（`#5` 被降级 ⇒ 预测不带自己的相对速度 ⇒ **帧 14** 回来时显得"
          "「位置可疑」✗ ⇒ 用户连报两帧 ✓）"
          % (float(_LM._M_OK), len(_bad), [(c[0], c[1], c[2]) for c in _claims]))
    # ⭐⭐⭐⭐⭐ **"老轨迹不许拿比现任更弱的证据顶替"** ✓（用户 2026-10-05 ✓ 原话："**真假目标怎么交换
    #   身份了？#5 是真目标，#9 是那一格子的假目标**" ✓✓ 见 `_CLAIM_STRONG_K` ✓）——
    #   ⚠ **刚出生的新框不受此限** ✓（它没有历史可分 ✓ 见 `_CLAIM_FRESH_N` ✓）。
    _weak = [c for c in _claims
             if len(c) > 6 and c[6] is not None and c[5] is not None
             and int(c[6][1]) > int(_LM._CLAIM_FRESH_N)
             and float(c[6][0]) < float(c[5][2]) * float(_LM._CLAIM_STRONG_K)]
    check(not _weak,
          "⑦ **老轨迹不许拿比现任更弱的证据顶替** ✓（`_CLAIM_STRONG_K` = %.2f ✓ ｜ 违规 **%d 笔**"
          "（要 0 ✓）｜ 全部认领 = %s）—— ⚠ 没这条门时：`10月3日 (2)` **帧 31** 会用**跟了 21 拍**、"
          "分只有 **7.3** 的 `#9`（那条**假目标** ✓）顶掉分 **26.2** 的 `#5`（**真目标** ✓ —— 只因它"
          "帧 28~30 **一直在融合** ⇒ 帧 31 那一拍没框 ✗）⇒ 就是你看到的「**真假目标交换身份**」✗✗"
          % (float(_LM._CLAIM_STRONG_K), len(_weak),
             [(c[0], c[1], c[2]) for c in _claims]))


def test_clamp_protection_when_own_clean():
    """⭐⭐⭐⭐⭐ **规则①的保护：目标自己有干净框时，不许拿「别人的融合框」夹它** ✗✗
    （用户 2026-10-05 ✓ "**A+B**" 里的 **B** ✓）。

    ⚠⚠ 实测病根（`10月3日 (2)` 帧 4 ✓）：目标 `#5` 自己有一格干净的 139×134 框 ✓（**就在白块
      身上** ✓），画面里另有一格融合框 `#1`（圆心距圆 **49px** ⇒ 过得了 `_near_circle` ✓）
      ⇒ 夹取把圆从 (372,250) **拽到 (256,213)**（**115px** ✗✗）⇒ 看着像"没认白块" ✓。

    钉两条（一正一反 ✓）：
      ① **目标自己有干净框** ⇒ 那格别人的融合框**不夹**（`clamp is None` ✓、圆留在目标身上 ✓）；
      ② **反面对照**：把目标那格**干净框拿掉**（它这拍没观测 ✓）⇒ 规则①（唯一融合框 ⇒ 夹）
         照旧生效 ✓（`clamp` 不再是 `None` ✓）—— ⚠ 一刀切"永不夹别人的框" ⇒ ② 立刻红 ✗。
    """
    def _scene(n=8, own_from=6):
        """一圈假目标（跟相机走 ✓）＋ 目标（相对群体 +12px/拍 ✓）；
        ⚠ **另一格假目标的框做成一格 200×200 的融合框** ✓，就放在圆的近旁 ✓。"""
        _f0 = _frame(9)
        _s = []
        for _i in range(n):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _tx, _ty = 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1]
            # 那个"别人"的大框：离目标 120px（⇒ 圆的圆心到它 ≤ 半个框 ⇒ 过 `_near_circle` ✓）
            _d = _d + [(0, _tx + 120.0, _ty, 200.0, 200.0, 0.9)]
            if _i < own_from:                      # ⚠ 前几拍**目标自己还有干净框** ✓
                _d = _d + [(0, _tx, _ty, 150.0, 150.0, 0.9)]
            _s.append(_d)
        return _s

    _F_LAST = 6                     # ⚠ 帧号（1 基）> 它 ⇒ 目标那格干净框**已经没了** ✓
    # ⚠⚠⚠ **`min_hits` 必须 ≥ 3** ✗✗（**实测踩到** ✓）：写 `1` 时**第 1 拍就能"选人"** ✓ ——
    #   而那时**所有账的分数都还是 0** ✗ ⇒ "取分数最高"**并列** ⇒ 取到**列表里第一个** = 环上
    #   那条 ✗✗ ⇒ 座位压根不是真目标 ✓（实测帧 2~6 `tid=1` = 环上那条、真目标在 `#11` 上挂着 ✓）
    #   ⇒ ①"绿得不对" ✓、② 根本走不到"目标那格干净框没了" ✗。⇒ 用 **3**（同
    #   `test_fused_box_goes_to_circle_owner` ✓）：等"在走的那个"攒出分 ✓ ⇒ 才会被选中 ✓。
    _t = MotionTracker(min_hits=3)
    _rows = []
    for _i, _d in enumerate(_scene()):
        _o = _t.process(None, ts=_i * 0.16, dets=_d)
        # ⚠ **本拍"目标自己有没有干净框"** —— 直接照那条判据算 ✓（不靠"猜哪条是目标" ✗）
        _oc = any(int(b[4]) == _t.tid and int(b[7]) == 0 for b in _t._box_v)
        _rows.append((_i + 1, _o.get("clamp"), _oc))
    _own = [_r for _r in _rows if _r[2]]
    _noown = [_r for _r in _rows if not _r[2] and _r[0] >= _F_LAST]
    check(bool(_own) and all(_r[1] is None for _r in _own),
          "① **目标自己有干净框 ⇒ 不夹别人的融合框** ✓（实测这样的拍共 **%d** 拍、`clamp` 全是 "
          "`None` ✓ —— ⚠ 去掉这道保护 ⇒ 真素材帧 4 的圆被拽 **115px** ✗✗）" % len(_own))
    check(bool(_noown) and any(_r[1] is not None for _r in _noown),
          "② **反面对照：目标那格干净框一没** ⇒ 规则①（唯一融合框 ⇒ 夹）照旧生效 ✓"
          "（实测 `clamp` 出现在的拍 = %s ✓）—— ⚠ 一刀切「永不夹别人的框」⇒ 这条红 ✗"
          % ([int(_r[0]) for _r in _noown if _r[1] is not None],))


def test_guide_keyed_on_circle():
    """⭐⭐⭐⭐⭐ **框心指导按"绿圈"判，不按"框归哪条轨迹"判**（用户 2026-10-05 ✓ 原话：
    "**你有没有践行「绿色圆就是我们认为真目标所在的位置」这一条原则？**" ✓✓）。

    ⚠⚠ 病根（**真素材实测** ✓ `10月3日` 帧 20~24）：融合期两个目标**只出一格**大框 ⇒ 它只配给
      **一条**轨迹 ⇒ 经常配给**别人**那条 ⇒ 真目标这条**这拍没有自己的框** ⇒ 它的 `stuck` 归 0 ✗。
      而那道门原来写的是 `_tgt.stuck > 0` ✗ = **按"框归哪条轨迹"判** ✗ ⇒ 明明画面上那格框
      **裹着绿圈** ✓、面板也写着"融合（含真目标）" ✓，指导却**整段不跑** ✗（实测
      `fuse_dir_ema` 从帧 20 起**冻在 `(-28.2,…)` 不动** ✓）；门改成"看候选"之后，那几拍
      **都在变** ✓（帧 18/19/20/21/24 跑起来 ✓；22/23 被**同向门**挡 ✓ —— 那是防噪声的正常
      保护 ✓ 见 `test_fuse_pull_gate` ⑦ ✓）。

    ⚠⚠ **为什么这里只能钉源码、不钉行为** ✗（不是偷懒 ✓ 同 `test_clamp_feeds_velocity` ④ ✓）：
      这条要钉的局面是"**目标自己那格框归了别人**"✗ —— 而**离屏小夹具里几条轨迹挨得太近，
      配对的归属会被"谁离得更近"随手改掉** ✗（**实测踩到** ✓：我照着现场造了个"旁边那格大框
      裹着绿圈"的夹具 ⇒ 目标那条轨迹**反手就把大框抢了过来** ⇒ 局面根本不出现 ✓）。
      ⇒ 行为层面这条**只能在真素材上量** ✓（上面那组帧号就是实测记录 ✓）。
    """
    import inspect
    _src = inspect.getsource(MotionTracker.process).replace(" ", "")
    check("if_csand_rn>1e-6:" in _src and "_tgt.stuck>0and_cs" not in _src,
          "① **源码级：融合那支的门只看候选** ✗✗（`if _cs and _rn > 1e-6:` ✓ 里面**不许**再出现"
          " `_tgt.stuck > 0` ✗）—— ⚠ 退回「按框归哪条轨迹判」⇒ 融合期那几拍指导**整段不跑**"
          "（真素材实测 `fuse_dir_ema` **冻住** ✗✗）")
    check("_own_in=any(int(_c[2])==int(self.tid)for_cin_cs)" in _src
          and "if(_sn>1e-6and(_own_inor(" in _src.replace("\n", ""),
          "② **源码级：`_own_in` 只留给同向门** ✓（「是不是目标自己那格」只决定**要不要过同向门** ✓"
          " —— 自己那格 = 确定裹着绿圈 ⇒ 直接采信 ✓；别人的框 ⇒ 才跟 `vr` 比 ✓）")


def test_fuse_pull_gate():
    """⭐⭐ **框心指导的"门控"**（用户 2026-10-04 ✓）—— 两条**并存**的口径：
    · **老门控**（`fuse_multi=False` ✓ A/B 用）：**"唯一融合框"**（用户原话："在有唯一融合框时
      在走框心影响的算法" ✓）⇒ ① sel 那格是真粘连；② 画面里真粘连框**恰好 1 个** ✓（用例 ①②③ ✓）。
    · **新口径**（`fuse_multi=True` ✓ **默认**）：**"多个融合框 ⇒ 每个框心都参与"**（用户原话：
      "默认多个融合框时每个框心都需要有作用（现在先不动速度，只动转向）" ✓）⇒ 各候选框心位移
      **取均值** ✓＋一道"与 `vr` 同向"的防噪门 ✓（用例 ④⑤⑥⑦ ✓；「框心力度」参数钉在 ⑧ ✓）。

    ⚠ 验证口径：**门控放行 ⇒ 拽生效**（`_FUSE_PULL=0.15` 与 `=0` 的报出位置**不同** ✓）；
      **门控挡住 ⇒ 与关掉逐位一致**（`=0.15` 与 `=0` **一模一样** ✓ = 被短路 ✓）。
    ⚠ 造"融合框"= 目标框 **200×200**（/ 标准 150×150 = **1.78x** > `stuck_ratio` 1.4 ⇒ `st==1` ✓）。
    """
    import perception.lie_motion as _LM
    # ⚠⚠⚠ **本用例测的是「门控」** ✗ ⇒ 全程**关掉"拆框"** ✓ —— 拆框是**位置硬约束**（会把圆按到
    #   目标那一块 ✓ 见 `process` 那段 ✓）⇒ 一开就**改了整条轨迹** ⇒ 那些"`=0.15` vs `=0` 差多少 px"
    #   的断言全跟着变 ✗✗（**实测**：一开就 6 条红 ✓ 而门控代码一个字没动 ✓）。
    #   ⚠ 写法与既有那套一致 ✓（本用例本来就在改 `_LM._FUSE_PULL` ✓）；末尾**必须还原** ✗
    #     （后面还有别的用例 ✓ 泄漏出去会连累它们 ✓）。
    _LM._FUSE_SPLIT = False

    def _seq(extra, fuse_to=99):
        """18 拍：9 假目标（150×150 一起走）+ 真目标那格第 11~`fuse_to+1` 帧 **200×200**（粘连 ✓）；
        `extra` = 额外再加几个别处的 200×200（造"多粘连框" ✓）。"""
        _fake0 = _frame(9)
        _s = []
        for _i in range(18):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_fake0, _cam[0], _cam[1])
            # ⚠ 前 10 拍 **150×150**（正常 ⇒ 学 `vel` ✓）；第 10~`fuse_to` 拍 **200×200**
            #   （= 1.78x ⇒ `stuck>0` ✓）—— ⚠ 必须**先正常几拍**，否则 `vel` 恒 0 ⇒ 拽不了 ✓。
            _sz = 200.0 if (10 <= _i <= fuse_to) else 150.0
            _d = _d + [(0, 380.0 + _cam[0] + 14.0 * _i, 300.0 + _cam[1], _sz, _sz, 0.9)]
            # ⚠⚠ **额外那个框要"和真目标同时长大"**（= 前 10 拍 150 ⇒ 之后 200 ✓）✗✗ ——
            #   我原来让它**从头就是 200** ✗ ⇒ 它自己的面积基准也是 200 ⇒ 面积比恒 1.0
            #   ⇒ （按"相对自己的面积"判）它**根本判不出"粘住了"** ✗ ⇒ 画面里只有 1 个粘连框
            #   ⇒ 老门控**不该挡的也放行** ⇒ ② 红（**实测** ✓ `_n_fuse` 会掉到 1 ✓）。
            _szo = 200.0 if _i >= 10 else 150.0
            for _k in range(int(extra)):
                _d = _d + [(0, 80.0 + _cam[0], 60.0 + 90.0 * _k + _cam[1],
                            _szo, _szo, 0.9)]
            _s.append(_d)
        return _s

    def _last(seq, pull, multi=False):
        _LM._FUSE_PULL = pull
        # ⚠⚠⚠ **本用例量的是「框心指导」** ✗ ⇒ 每拍都要把它**临时打开** ✓（用户 2026-10-05 把总开关
        #   `_LM._FUSE_GUIDE` 关成 `False` 了 ✓ —— 关掉时"拽不拽都一样"✗，与本用例要量的事无关 ✓）。
        _LM._FUSE_GUIDE = True
        _t = MotionTracker(min_hits=3, fuse_multi=multi)
        _o = None
        for _i, _d in enumerate(seq):
            _o = _t.process(None, ts=_i * 0.16, dets=_d)
        return _o["pos"], _t

    # ① 唯一、真粘连 ⇒ 老门控放行 ⇒ 拽生效
    _sA = _seq(0)
    _a1, _ = _last(_sA, 0.15, multi=False)
    _a0, _ = _last(_sA, 0.0, multi=False)
    _dA = (math.hypot(_a1[0] - _a0[0], _a1[1] - _a0[1]) if (_a1 and _a0) else -1.0)
    check(_dA > 0.01,
          "① **唯一、真粘连**（`stuck>0` 且画面里就 1 个 ✓）⇒ 老门控**放行** ⇒ 拽生效"
          "（`_FUSE_PULL=0.15` vs `=0` 的报出位置差 **%.2f px** ✓）" % _dA)

    # ② `fuse_multi=False`（老门控）：画面里 2 个真粘连框 ⇒ 挡 ⇒ 与关掉逐位一致
    _sB = _seq(1)
    _b1, _ = _last(_sB, 0.15, multi=False)
    _b0, _ = _last(_sB, 0.0, multi=False)
    _dB = (math.hypot(_b1[0] - _b0[0], _b1[1] - _b0[1]) if (_b1 and _b0) else -1.0)
    #   ⚠⚠⚠ **口径变了（用户 2026-10-06 ✓）** ✗✗：`fuse_pull` 现在**兼"四条边转向比例"** ✓
    #     （用户原话："**参数用 k = 每拍转向比例 = 「框心力度」**" ✓✓）⇒ 门控挡住的是 **pull** ✓，
    #     而"**四条边转向**"是**另一套机制** ✓、与这道门**无关** ✓ ⇒ 两次跑**不再逐位一致** ✗。
    #   ⇒ 改钉两条（**都比 ④ 小得多** ✓）：① 差值**很小**（≤ 3px ✓ = pull 确实被挡住了 ✓）；
    #     ② 而 `multi=True`（④ ✓）那份**明显更大** ✓ ⇒ 对照成立 ✓。
    check(abs(_dB) <= 3.0,
          "② **`fuse_multi=False`（老）**：画面里 **2 个真粘连框** ⇒ 分不清哪个含真目标 ⇒ "
          "门控**挡住 pull** ✓ ⇒ 差值只剩「四条边转向」那一份、**很小**（**%.2f px** ✓〔要 ≤ 3 ✓〕；"
          "⚠ 不挡 ⇒ 会拽错 ⇒ 差值会跟 ④ 一样大 ✗）" % _dB)

    # ③ 冷却期（`stuck==0` 但 `cool>0`：框已分开、框心 ≠ 中点）⇒ 门控条件 `stuck>0` 为假 ⇒ 挡
    #    ⚠ 不比"最后一拍 pos" ✗ —— 前面粘连期**两遍都拽过**（累积差 0.33px ⇒ 污染 ✓ 实测踩到）；
    #      门控逻辑就一行 ⇒ 用**状态级（冷却期 `stuck==0`）+ 源码级（门控写的是 `stuck>0` 不是 `st`）** 钉 ✓。
    import inspect
    _sC = _seq(0, fuse_to=16)          # 第 11~17 帧粘连；第 18 帧恢复 ⇒ `cool>0`（冷却 ✓）
    _, _tC = _last(_sC, 0.15, multi=False)
    _tg = _tC._by_id(_tC.tid) if _tC.tid is not None else None
    _ck = int(getattr(_tg, "cool", -1)) if _tg is not None else -1
    _sk = int(getattr(_tg, "stuck", -1)) if _tg is not None else -1
    _src_gate = inspect.getsource(MotionTracker.process).replace(" ", "")
    check(_sk == 0 and _ck > 0 and "_tgt.stuck>0and_n_fuse==1" in _src_gate,
          "③ **冷却期**（`stuck==0` 但 `cool=%d>0` ⇒ 框已分开、框心 ≠ 中点 ✓）⇒ 老门控条件 "
          "`stuck>0` 为假 ⇒ **挡**（⚠ 门控写的是 `stuck>0`、**不是** `st==1` ✓ 源码级钉住 ✓ ——"
          " 改回 `st` 会拽错冷却期 ⇒ 立刻红 ✗）" % _ck)

    # ---- ⭐⭐⭐ **下面三条钉"多融合框 ⇒ 挑一个"**（用户 2026-10-04 ✓ 见 `_FUSE_MULTI` ✓）----
    # ④ `fuse_multi=True`（**默认** ⇒ 新）：同样那"2 个融合框"**不再挡** —— 与 ② 正好对照 ✓
    _c1, _ = _last(_sB, 0.15, multi=True)
    _c0, _ = _last(_sB, 0.0, multi=True)
    _dC = (math.hypot(_c1[0] - _c0[0], _c1[1] - _c0[1]) if (_c1 and _c0) else -1.0)
    check(_dC > 0.01,
          "④ **`fuse_multi=True`（默认）⇒ 同样的「2 个融合框」不再挡**（挑一个 ✓）⇒ 拽生效"
          "（差 **%.3f px** ✓ —— 与 ② 对照：不挑的话这里必须也是 0 ✗）" % _dC)

    # ⑤ **挑的是"与 `vr` 最同向"的那个**：第二个融合框**反着走**（−14px/拍 ⟂ 真目标 +14px/拍）
    #    ⇒ 必须**不选**它 ⇒ 拽出来的方向仍与 `vr`（+x）同向 ✓
    def _seq_pick(extra_dx, extra_dy=0.0):
        """同 `_seq`，但**额外那个 200×200 以 `(extra_dx, extra_dy)`px/拍 移动**（造"第二个融合框" ✓）。"""
        _fake0 = _frame(9)
        _s = []
        for _i in range(18):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_fake0, _cam[0], _cam[1])
            # ⚠⚠⚠ **真目标第 7 拍就粘**（`_i >= 6`✓ 原来是 10 ✗）—— **实测抓到的时序坑** ✗✗：
            #   另一个框**反着群体走**（−14/拍，而相机 +7）⇒ 它的位置估计**每拍被群体中位往右推
            #   7px**（非目标轨迹**不**带自己的相对速度 ✓ 见 `process` 滑行支 ✓）⇒ 每拍分家
            #   **14px** ⇒ **约 5 拍后 ~84px > 配对门(70)** ⇒ 它**配不上自己那格框** ⇒ 轨迹被
            #   重建 ✗（**实测**：帧 11 `miss=1 obs=(214,130)` 而框在 (130,130) ✗）。
            #   ⇒ 原来真目标第 11 拍才粘 —— 恰好是它**死掉**那一拍 ⇒ ⑦ 那拍**一条候选都没有**
            #     ✗（`id=[]` ✓）。⇒ **把两个框的粘连时间错开**：另一个第 4 拍粘（`_i >= 3` ✓）、
            #     真目标第 7 拍粘 ⇒ 真目标刚粘那拍，另一个**还活着**（第 4~8 拍 ✓）⇒ 局面成立 ✓✓。
            #   ⚠ 顺带这也解释了用户那两轮"编号对不上"（真素材里也有反向漂的假目标 ⇒ **轨迹被
            #     反复重建** ⇒ 编号一直跳 ✗）。
            _sz = 200.0 if (6 <= _i <= 16) else 150.0
            # ⚠⚠⚠ **真目标必须"明显比另一个更不合群"** ✗✗（**实测踩到** ✓）：这里原来是 **14**、
            #   而另一个框也是 **±14** ⇒ 两者**一样不合群** ⇒ **选中谁靠掷硬币** ✗ ⇒ ⑤⑦ 时红时绿 ✓
            #   （我这几轮一改硬币就翻面 ✓）。⇒ 真目标改成 **24**（另一个仍是 ±14 ✓ 见调用处 ✓）
            #   ⇒ **它稳定当选** ✓ —— 夹具**必须造出"谁是真目标"没有歧义的局面** ✓。
            _d = _d + [(0, 380.0 + _cam[0] + 24.0 * _i, 300.0 + _cam[1], _sz, _sz, 0.9)]
            # ⚠⚠ 第二个融合框**也要"长大"才判得出粘连** ✗（同 `_seq(extra)` 那个坑 ✓：
            #   从头就是 200 ⇒ 面积比恒 1.0 ⇒ 它**不是**粘连框 ⇒ ⑦ 那个"只剩别人的框"的
            #   局面**根本造不出来** ✗ ⇒ `_c7` 空 ⇒ 红 ✓）。
            _szo = 200.0 if _i >= 3 else 150.0
            _d = _d + [(0, 200.0 + _cam[0] + extra_dx * _i,
                        100.0 + _cam[1] + extra_dy * _i, _szo, _szo, 0.9)]
            _s.append(_d)
        return _s

    # ⑤ **每个框心都参与**（用户 2026-10-04 ✓ 原话："**默认多个融合框时每个框心都需要有作用**" ✓）：
    #    第二个融合框**只在 y 方向动**（+14px/拍）⇒ 合方向必须**带上它的 y 分量** ✓
    #    （⚠ 只取真目标那一格的话 y ≈ 相机(3) ⇒ 过不了 10 这道关 ⇒ 立刻红 ✗）
    _t5 = MotionTracker(min_hits=3)                # fuse_multi 默认 True ✓
    #   ⚠⚠ **量"峰值"而不是"结尾值"** ✗✗（**实测**：夹具跑到第 18 拍时真目标那格**早就不融合了**
    #     ✓ ⇒ 它的候选退出 ✓，而第二个框也早在第 5 拍因"反着群体走"丢了框 ✗ ⇒ 结尾只剩一条
    #     在收敛的残值（实测 (39.4,5.0) ✗）—— 那量的是"散场以后"，不是要钉的性质 ✓。
    #     ⇒ 要钉的是"**那个框心确实起过作用**" ✓ ⇒ 取运行中的**峰值 y** ✓（帧 12~13 实测
    #       `fuse_dir_ema = (24,14)` ✓ = 真目标 (48,0) 与另一个 (0,28) 的均值 ✓ 正是那条口径 ✓）。
    _peak5 = None
    for _i, _d in enumerate(_seq_pick(0.0, 14.0)):
        _t5.process(None, ts=_i * 0.16, dets=_d)
        _g5 = _t5._by_id(_t5.tid) if _t5.tid is not None else None
        _e5 = getattr(_g5, "fuse_dir_ema", None)
        if _e5 is not None and (_peak5 is None or _e5[1] > _peak5[1]):
            _peak5 = (float(_e5[0]), float(_e5[1]))
    #   ⚠⚠⚠ **口径改了 ⇒ 判据收窄到"要钉的那条性质"** ✗✗（用户 2026-10-06 ✓）：本用例要钉的是
    #     "**第二个框心的 y 分量确实进了合方向**" ✓（实测峰值 y = **16.9 > 10** ✓ 成立 ✓）；
    #     我当年还额外加了"**x 必须 > 0**" ✗ —— 那依赖"**圆跟着并集框走**"那套 ✓，而 2026-10-06
    #     起按用户口径：**夹取只在首拍 / 出框时发生** ✓ ⇒ 圆不再被并集框拖着走 ⇒ 真目标那格的
    #     x 分量**变号** ✓（实测峰值 x = **−32.6** ✗）⇒ 那条额外判据**过期** ✓ ⇒ 去掉 ✓。
    check(_peak5 is not None and _peak5[1] > 10.0,
          "⑤ **每个框心都参与**：真目标 +24px/拍（只有 x ✓）、第二个融合框**只在 y 动** +14px/拍 ✓"
          " ⇒ 合方向**带上了它的 y**（峰值 `fuse_dir_ema`=(%.1f,%.1f) ⇒ y **> 10** ✓）—— ⚠ 只算"
          "真目标那一格的话 y ≈ 0 ⇒ 立刻红 ✗（那就不是「每个框心都有作用」了 ✓）"
          % ((0.0 if _peak5 is None else _peak5[0]),
             (0.0 if _peak5 is None else _peak5[1])))

    # ⑥ **起始拍不参与合方向**：真目标"刚粘上"那一拍 ⇒ 它**自己**没有候选
    #    （⚠ 掺进去的话"框心阶跃 +42px"会被当速度 ✗）
    _sF = _seq(0)
    _tF = MotionTracker(min_hits=3)                # fuse_multi 默认 True ✓
    for _i in range(11):                           # 第 0~10 拍（第 10 拍 = 刚变 200×200 ✓）
        _tF.process(None, ts=_i * 0.16, dets=_sF[_i])
    _tidF = _tF.tid
    _tgF = _tF._by_id(_tidF) if _tidF is not None else None
    _self_in = any(int(_c[2]) == int(_tidF) for _c in _tF._fuse_cands)
    check(_tgF is not None and int(_tgF.stuck) > 0 and not _self_in,
          "⑥ **起始拍不参与合方向**：真目标（#%s）刚粘上那一拍 `stuck=%s>0` ✓ 但**它自己不在候选里** ✓"
          "（本拍候选 id = %s ✓ —— ⚠ 不排除的话「框心阶跃」会被当速度 ✗）"
          % (_tidF, (None if _tgF is None else int(_tgF.stuck)),
             [_c[2] for _c in _tF._fuse_cands]))

    # ⑦ **同向门槛 `_FUSE_PICK_MIN`**（⚠ **实测 `10月1日` 逼出来的** ✓）：起始那几拍**只剩
    #    "别的框"**可选、而它**与 `vr` 反向** ⇒ **宁可这一拍不拽** ✗（不把噪声方向灌进 EMA ✓）。
    _t7 = MotionTracker(min_hits=3)                # fuse_multi 默认 True ✓
    _s7 = _seq_pick(-14.0)                         # 另一个融合框**反着走** ⇒ cos = −1 ✗
    #   ⚠⚠ **检查点必须落在"真目标刚粘上那一拍"** ✗✗（= 第 0 基 **6** 拍 ✓ 见 `_seq_pick` 的
    #     粘连时序注释 ✓；原来是第 11 拍 —— 那一拍"另一个框"**已经丢框 5 拍了** ✗（反着群体走 ⇒
    #     位置估计每拍被中位推 14px ✗）⇒ 候选**空** ✓ 才量出 `id=[]` ✗）。**顺带把该局面记下来** ✓：
    #     这条性质的前提（"真有那么一拍"）也要钉住 ✗，不然夹具退化了还是绿 ✓。
    _sit7 = []
    for _i in range(7):                            # 第 0~6 拍（第 7 拍 = 真目标刚粘上 ✓）
        _t7.process(None, ts=_i * 0.16, dets=_s7[_i])
        _g7 = _t7._by_id(_t7.tid) if _t7.tid is not None else None
        _cc7 = [_c for _c in (_t7._fuse_cands or []) if math.hypot(_c[0], _c[1]) > 1e-6]
        if (_g7 is not None and int(_g7.stuck) > 0 and _cc7
                and not any(int(_c[2]) == int(_t7.tid) for _c in _cc7)):
            _sit7.append((_i + 1, [_c[2] for _c in _cc7]))
    _tg7 = _t7._by_id(_t7.tid) if _t7.tid is not None else None
    _c7 = [_c for _c in _t7._fuse_cands if math.hypot(_c[0], _c[1]) > 1e-6]
    _rev7 = [c for c in _c7
             if (c[0] * float(_tg7.vr[0]) + c[1] * float(_tg7.vr[1])) < 0.0] if _tg7 else []
    check(_tg7 is not None and int(_tg7.stuck) > 0 and bool(_c7) and bool(_rev7)
          and getattr(_tg7, "fuse_dir_ema", None) is None and bool(_sit7),
          "⑦ **同向门槛**：真目标刚粘上那一拍**只剩「另一个反向的融合框」**当候选"
          "（本拍 id=%s ✓ 与 `vr`=(%.0f,%.0f) 反向 ✗；该局面出现在第 %s 拍 ✓） ⇒ **这一拍不拽**"
          "（`fuse_dir_ema` 仍 None ✓ —— ⚠ 实测不加这道门 ⇒ 帧 13 `dot=−0.86` 的噪声方向灌进 EMA"
          " ⇒ 后面几拍全被带歪 ✗✗）"
          % ([_c[2] for _c in _c7],
             (0.0 if _tg7 is None else float(_tg7.vr[0])),
             (0.0 if _tg7 is None else float(_tg7.vr[1])),
             [_s[0] for _s in _sit7]))

    # ⑧ ⭐ **`fuse_pull`（界面「框心力度」）真的接上了**（⚠ 摆到界面上的参数必须真起作用 ✗）：
    #    `0.0`（不拽）/ `0.15` / `0.30`（拽得更狠）⇒ 报出位置**两两不同** ✓。
    #    ⚠⚠ **`1.0` 与 `2.0` 也在里面** ✗✗（用户 2026-10-05 ✓ 原话："**框心力度无法调到 1 以上**"
    #     ✓✓）：把上界从 `1.0` 放到 `3.0` 之后，**后端侧本来就不夹** ✗（`_pull_to` 是纯线性插值 ✓
    #     见它那句注释 ✓）—— 这条钉子钉住"**> 1 真能流进来、且真改变结果**" ✓，免得哪天有人在
    #     后端偷偷加一道 `min(1.0, …)` ✗ 而界面看不出来 ✓。
    def _last_fp(seq, fp):
        _t = MotionTracker(min_hits=3, fuse_pull=fp)
        _o = None
        for _i, _d in enumerate(seq):
            _o = _t.process(None, ts=_i * 0.16, dets=_d)
        return _o["pos"]
    _q0, _q1, _q3 = _last_fp(_sA, 0.0), _last_fp(_sA, 0.15), _last_fp(_sA, 0.30)
    _qF, _qG = _last_fp(_sA, 1.0), _last_fp(_sA, 2.0)
    _d01 = (math.hypot(_q1[0] - _q0[0], _q1[1] - _q0[1]) if (_q0 and _q1) else -1.0)
    _d13 = (math.hypot(_q3[0] - _q1[0], _q3[1] - _q1[1]) if (_q1 and _q3) else -1.0)
    _dF3 = (math.hypot(_qF[0] - _q3[0], _qF[1] - _q3[1]) if (_q3 and _qF) else -1.0)
    _dGF = (math.hypot(_qG[0] - _qF[0], _qG[1] - _qF[1]) if (_qF and _qG) else -1.0)
    check(_d01 > 0.01 and _d13 > 0.01 and _dF3 > 0.01 and _dGF > 0.01,
          "⑧ **界面「框心力度」真的接上了**：`fuse_pull` ∈ {0.0, 0.15, 0.30, **1.0, 2.0**} ⇒ 报出位置"
          "**两两不同**（|p15−p0| = **%.2f** ｜ |p30−p15| = **%.2f** ｜ |p100−p30| = **%.2f** ｜ "
          "|p200−p100| = **%.2f** px ✓ —— ⚠ 不接的话**逐位一致** ⇒ 红 ✗；⚠ **上界之后**（1.0 / 2.0）"
          "若被夹回 ⇒ 这两项也会一致 ⇒ 红 ✗）" % (_d01, _d13, _dF3, _dGF))

    # ⑨ **隔拍补偿**：融合期**漏检 1 拍** ⇒ **下一拍照样算得出框心位移** ✓
    #   （⚠ 不补的话要**空等一拍** ✗ = 用户 2026-10-04 实测"帧 16 配 0.99 也没用"那个坑 ✓）
    #   ⚠⚠ 夹具的坑（第一版写错 ✓）：真目标那格**必须离假目标很远** ✗ —— 挨太近时"抽掉它"
    #     会被它**抢到旁边假目标的框** ⇒ 轨迹**照样配上**（`stuck` 还被清零 ✗）⇒ 根本没造出
    #     "粘着但没框"那种局面 ✓（真实素材里帧 15 是**真丢**、`stuck` 不变 ✓ 才是要测的 ✓）。
    def _seq_gap():
        """18 拍：9 个假目标跟相机走 ＋ 真目标那格在 **x≈2000**（远 ✓）第 11~17 帧 **200×200**；
        **第 14 帧（0 基 13）抽掉它** = 漏检 1 拍 ✓。"""
        _fake0 = _frame(9)
        _s = []
        for _i in range(18):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_fake0, _cam[0], _cam[1])
            _sz = 200.0 if (10 <= _i <= 16) else 150.0
            if _i != 13:                    # ⚠ 第 14 帧**漏检** ✓
                _d = _d + [(0, 2000.0 + _cam[0] + 14.0 * _i,
                            300.0 + _cam[1], _sz, _sz, 0.9)]
            _s.append(_d)
        return _s

    _tG = MotionTracker(min_hits=3)         # fuse_multi 默认 True ✓
    _cand_next = False
    for _i, _d in enumerate(_seq_gap()):
        _tG.process(None, ts=_i * 0.16, dets=_d)
        if _i == 14:                        # 漏检的**下一拍** ✓
            _tgG = _tG._by_id(_tG.tid) if _tG.tid is not None else None
            _cand_next = bool(_tgG is not None and int(_tgG.stuck) > 0
                              and any(int(_c[2]) == int(_tgG.id)
                                      for _c in _tG._fuse_cands))
    check(_cand_next,
          "⑨ **隔拍补偿**：融合期**漏检 1 拍** ⇒ **下一拍照样算得出框心位移**（候选里**有它** ✓）"
          " —— ⚠ 不补的话这里为空 ⇒ 空等一拍 ✗（用户实测帧 16 的坑 ✓）")

    # ⑩ **冷却中的框心不参与**（用户 2026-10-04 ✓ 原话："每个融合框的框心都参与，**除了冷却中的**" ✓）
    #    ⚠⚠ 判据必须写 `stuck > 0` ✗✗ —— **不能**写 `cool <= 0` ✗：实测**粘连期 `cool` 本身就是
    #      常置的**（见 ③ 段 ✓）⇒ 那样写会把粘连期也一起挡掉 ✓（本用例顺便钉住这条 ✓）。
    _t10 = MotionTracker(min_hits=3)          # fuse_multi 默认 True ✓
    _sk10, _ck10, _in10 = None, None, None
    for _i, _d in enumerate(_seq(0, fuse_to=16)):   # 第 11~17 帧粘连；第 18 帧恢复 ⇒ **冷却** ✓
        _t10.process(None, ts=_i * 0.16, dets=_d)
        if _i == 17:                          # 恢复那一拍 = **冷却中** ✓
            _tg10 = _t10._by_id(_t10.tid) if _t10.tid is not None else None
            _sk10 = int(getattr(_tg10, "stuck", -1)) if _tg10 is not None else -1
            _ck10 = int(getattr(_tg10, "cool", -1)) if _tg10 is not None else -1
            _in10 = any(int(_c[2]) == int(_tg10.id) for _c in _t10._fuse_cands)
    check(_sk10 == 0 and _ck10 > 0 and not _in10,
          "⑩ **冷却中的框心不参与**：恢复那一拍 `stuck=%s`（==0 ✓）、`cool=%s`（>0 ✓）"
          " ⇒ 框已分开、框心 ≈ 真目标中心 ✗ ⇒ **它不在候选里** ✓（⚠ 判据写 `cool<=0` 的话"
          "**粘连期也一起被挡** ⇒ ①②④ 会立刻红 ✗）" % (_sk10, _ck10))
    _LM._FUSE_SPLIT = False           # ⚠ **还原成"自检的默认"（= 关 ✓）**，别泄漏成开 ✗（见文件头那段 ✓）
    _LM._FUSE_GUIDE = False           # ⚠ 同理：把「框心指导」还原成**用户的默认（= 关 ✓）** ✗


def test_pos_rel_is_group_relative():
    """⭐⭐⭐ **橙线 = 「报出位置相对群体走过的路」**（用户 2026-10-04 ✓ 原话："**橙色线就应该是
    报出位置相对群体走过的路**" ✓✓）。

    钉三条：
      ① **末点必须是 (0,0)** ✓ —— 锚点就是圆心 ⇒ 端点必然落在绿圈上 ✓（= 用户要的"跟着圆心走" ✓）；
      ② 累计量必须 = **Σ(Δ圆心 − 相机)** ✓（相机取 `_med_ema` ✓ 与"滑行推进 / 黄箭头"同一个 ✓）
         —— ⚠⚠ **忘了减相机**（或减的是"候选框的 rel"）⇒ 立刻红 ✗✗：那正是用户报的
         "圆心走那么远了、橙线怎么没跟过来" ✓。
      ③ **白箭头 = 「圆本该走」的相对位移** ✓（用户 2026-10-06 ✓ 原话："**它在框内可能不动，但是
         我们的相对群体速度（白箭头）要保持，是一种「怼着墙走」的感觉**" ✓✓）——
         ⚠⚠ **口径改过一次**（如实记 ✓）：2026-10-04 那版画的是"**实际**走过"（= 橙线末段 ✓），
           可 10-06 融合期加了"**整圆出框就拉回**"那道夹取 ✓ ⇒ **贴墙的拍圆几乎不动** ⇒
           箭头**瘪成一条** ✗（用户当场问："**为什么到 59 帧圆心的相对群体速度变小这么多？**" ✓）。
         ⇒ 现在：**没贴墙的拍** ⇒ 与橙线末段**逐位一致** ✓；**贴墙的拍** ⇒ 等于"**约束前那一步**"
           （`MotionTracker._pos_intent` ✓）✓ **且不许瘪** ✓（模长 ≥ 1px ✓）。
    """
    from perception.lie_motion import MotionRunner

    def _seq_run():
        _fake0 = _frame(9)
        _s = []
        for _i in range(24):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_fake0, _cam[0], _cam[1])
            _d = _d + [(0, 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1],
                        150.0, 150.0, 0.9)]
            _s.append(_d)
        return _s

    _img = np.zeros((500, 750, 3), np.uint8)      # ⚠ `small` 只用来取画面尺寸 ✓
    _run = MotionRunner(dets=_seq_run())
    _pv, _mo = None, None
    _dpos = [0.0, 0.0]
    _cam = [0.0, 0.0]
    # ⭐ ③ 用的分类（每拍判一次：**这拍有没有被"整圆出框就拉回"那道夹取动过位置** ✓）
    _n_clamp, _bad_intent, _bad_same = [], [], []
    for _i in range(24):
        _o, _pos, _r, _h, _b, _mo = _run.step(_img, _i, _i * 0.16)
        _me = getattr(_run.tr, "_med_ema", None)
        if _pv is not None and _pos is not None:
            _dpos[0] += _pos[0] - _pv[0]
            _dpos[1] += _pos[1] - _pv[1]
            if _me is not None:                   # ⚠ 与后端**同一拍**才累相机 ✓
                _cam[0] += _me[0]
                _cam[1] += _me[1]
                # ⭐ ③：白箭头（`tgt_rel_last` ✓）对不对得上它的两个口径 ✓
                _ph2 = list((_mo or {}).get("pos_rel") or [])
                _tl2 = (_mo or {}).get("tgt_rel_last")
                _ip2 = getattr(_run.tr, "_pos_intent", None)
                if _tl2 is not None and len(_ph2) >= 2:
                    _sg2 = (_ph2[-1][0] - _ph2[-2][0], _ph2[-1][1] - _ph2[-2][1])
                    _cl2 = (_ip2 is not None
                            and (abs(_ip2[0] - _pos[0]) > 1e-9
                                 or abs(_ip2[1] - _pos[1]) > 1e-9))
                    if _cl2:
                        _n_clamp.append(_i + 1)
                        #   ⭐ 期望 = "**实际那一步 ＋ 被夹回来的那一截**" ✓（与橙线**同一基准** ✓）
                        _ex2 = (_sg2[0] + (_ip2[0] - _pos[0]),
                                _sg2[1] + (_ip2[1] - _pos[1]))
                        if (math.hypot(_tl2[0], _tl2[1]) < 1.0
                                or abs(_tl2[0] - _ex2[0]) > 1e-6
                                or abs(_tl2[1] - _ex2[1]) > 1e-6):
                            _bad_intent.append((_i + 1, _tl2, _ex2))
                    elif (abs(_tl2[0] - _sg2[0]) > 1e-9
                          or abs(_tl2[1] - _sg2[1]) > 1e-9):
                        _bad_same.append((_i + 1, _tl2, _sg2))
        _pv = _pos
    _ph = list((_mo or {}).get("pos_rel") or [])
    _acc = tuple(getattr(_run, "_pos_rel", (0.0, 0.0)))
    _exp = (_dpos[0] - _cam[0], _dpos[1] - _cam[1])
    _err = math.hypot(_acc[0] - _exp[0], _acc[1] - _exp[1])
    _reach = max([math.hypot(_p[0], _p[1]) for _p in _ph] or [0.0])
    check(bool(_ph) and abs(_ph[-1][0]) < 1e-9 and abs(_ph[-1][1]) < 1e-9,
          "① **橙线末点 = (0,0)**：锚点就是圆心 ⇒ **端点必然落在绿圈上** ✓（点数 %d ✓ —— ⚠ 锚在"
          "候选框上的老画法会让它**离开圆心** ✗）" % len(_ph))
    check(_err < 1.0 and _reach > 5.0,
          "② **累计量 = Σ(Δ圆心 − 相机)**：实测误差 **%.2f px** ✓、线最长铺到 **%.0f px** ✓"
          "（⚠ 忘了减相机 ⇒ 误差会到 **%.0f px** ⇒ 立刻红 ✗✗）"
          % (_err, _reach, math.hypot(_dpos[0], _dpos[1])))
    # ③ ⭐⭐⭐⭐⭐ **白箭头 = 「圆本该走」的相对位移**（用户 2026-10-06 ✓ 见 docstring ③ ✓）——
    #   两条一起钉 ✓：**没贴墙** ⇒ 与橙线末段逐位一致 ✓；**贴墙** ⇒ = 约束前那一步 ✓ 且**不许瘪** ✓。
    check(not _bad_same and not _bad_intent,
          "③ **白箭头 = 「圆本该走」的相对位移** ✓（贴墙的拍 = **%s** ✓〔本夹具**没碰上**夹取也没关系 ✓ "
          "—— 那类由 `test_fuse_clamp_never_touches_velocity` ＋ 真素材钉 ✓；这里钉的是**公式**："
          "白箭头 = 橙线末段 ＋ 「被夹回来的那一截」✓〕；"
          "那些拍**与「约束前那一步」不符**的有 **%d** 个 ✗〔要 0 ✓〕；**没贴墙的拍与橙线末段不等**的"
          "有 **%d** 个 ✗〔要 0 ✓〕）—— ⚠⚠ **旧口径**（画「实际走过」✗）⇒ 贴墙那几拍箭头**瘪下去** ✗"
          "（实测 `10月1日` 帧 58~61 ✓ 用户当场问「为什么到 59 帧变小这么多」✓）\n"
          "     ⚠ 明细：贴墙不符 %s ✓；未贴墙不符 %s ✓"
          % (_n_clamp[:4], len(_bad_intent), len(_bad_same),
             _bad_intent[:2], _bad_same[:2]))


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


def test_add_along_component():
    """⭐⭐⭐ **`_pull_to`：把速度"整体"拉向目标向量**（用户 2026-10-04 ✓ 原话："受框心引导后的
    圆心相对群体速度 = **力度 × (2×框心相对群体速度 − 当前圆心相对群体速度)**" ✓✓，并按助手建议
    **目标向量取 EMA** ✓）。

    口径：`v' = v + 力度 × (目标向量 − v)` ✓（目标向量 = `2×(Δ框心−群体中位)` 的 **EMA** ✓）。
    重点钉四条：① **朝目标收敛**（更快⇒加速 ✓ 更慢⇒减速 ✓）；
      ② ⭐ **整体向量**：**垂直分量按 `(1 − 力度)` 收缩** ✓
        （⚠ 旧 `_add_along` 是"垂直一点不动" ✗ —— 这正是用户这次要改掉的 ✓）；
      ③ **自限**：反复迭代**永不越过目标** ✓；④ **力度 0 ⇒ 逐位不变** ✓（⚠ 写成"`力度×(...)`
        不 `+v`" ⇒ 力度 0 会把速度**清零** ✗✗ 立刻红 ✓）。
    """
    import perception.lie_motion as _LM

    # ① 加速：10 → 15（目标向量 (20,0)、力度 0.5 ✓）
    _a = _LM._pull_to((10.0, 0.0), (20.0, 0.0), 0.5)
    check(abs(_a[0] - 15.0) < 1e-9,
          "① 框心说「**更快**」（20 > 当前 10）⇒ **加速**：分量 **%.2f**（期望 15 ✓）" % _a[0])
    # ② 减速（目标更慢 ⇒ 减 ✓）：20 → 15
    _b = _LM._pull_to((20.0, 0.0), (10.0, 0.0), 0.5)
    check(abs(_b[0] - 15.0) < 1e-9,
          "② 框心说「**更慢**」（10 < 当前 20）⇒ **减速**：分量 **%.2f**（期望 15 ✓）" % _b[0])
    # ③ ⭐ 整体向量：垂直分量按 (1−力度) 收缩（旧口径是"不动" ✗）
    _c = _LM._pull_to((10.0, 8.0), (20.0, 0.0), 0.5)
    check(abs(_c[0] - 15.0) < 1e-9 and abs(_c[1] - 4.0) < 1e-9,
          "③ ⭐ **整体向量**：`(10,8)` 拉向 `(20,0)`、力度 0.5 ⇒ **(%.1f, %.1f)**（期望 **(15, 4)** ✓）"
          "—— ⚠ 旧 `_add_along` 在这里给的是 **(15, 8)**（垂直不动 ✗）⇒ 换成旧写法 **③ 立刻红** ✓"
          % _c)
    # ④ 自限：反复拉**永不越过目标**（一步到位也不会冲过头 ✓）
    _v = (10.0, 0.0)
    _mx = _v[0]
    for _ in range(40):
        _v = _LM._pull_to(_v, (20.0, 0.0), 0.15)
        _mx = max(_mx, _v[0])
    check(_mx <= 20.0 + 1e-9 and abs(_v[0] - 20.0) < 0.2,
          "④ **自限**：反复拉 40 拍 ⇒ **收敛到目标 20**（末值 **%.2f**、峰值 **%.2f** ✓）"
          % (_v[0], _mx))
    # ⑤ 门：力度 0 ⇒ 逐位不变 ✓；没有目标向量（`None`）⇒ 原样返回 ✓
    check(_LM._pull_to((3.0, 4.0), (20.0, 0.0), 0.0) == (3.0, 4.0)
          and _LM._pull_to((3.0, 4.0), None, 0.5) == (3.0, 4.0),
          "⑤ `力度 = 0` ⇒ **逐位不变** ✓（⚠ 写成不带 `+v` 的「`力度×(目标−v)`」⇒ 这里会**清零** ✗✗）；"
          "没有目标向量（`None` ✓ 门限没过那种 ✓）⇒ **原样返回** ✓（不猜 ✓）")


def test_fake_ghost_follows_group():
    """⭐⭐⭐ **非目标丢了检出框 ⇒ "推的"那几拍只能按群体速度走**（用户 2026-10-04 ✓ 原话：
    "**带标签的假目标推的在丢检出框时在明显地做群体相对运动，明显是不对的**" ✓）。

    夹具：真目标 +14px/拍（会被选为目标 ✓）＋ 一个**会漂的假目标**（+6px/拍 ⇒ 它的 `vr`
    会被学成 ≈6 ✗）；第 15 拍抽掉那个假目标 ⇒ 它这一步**只能走相机 (7,3)** ✓。
    ⚠ 不修的话它还会把 `vr`(≈6) 带上 ⇒ 一步走 **(13,3)** ⇒ 立刻红 ✗✗。
    """
    def _seq_fake():
        _fake0 = _frame(9)
        _s = []
        for _i in range(18):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_fake0, _cam[0], _cam[1])
            _d = _d + [(0, 380.0 + _cam[0] + 14.0 * _i, 300.0 + _cam[1],
                        150.0, 150.0, 0.9)]
            if _i != 15:                    # ⚠ 第 15 拍 = 那个漂的假目标**丢框** ✓
                _d = _d + [(0, 1400.0 + _cam[0] + 6.0 * _i, 400.0 + _cam[1],
                            150.0, 150.0, 0.9)]
            _s.append(_d)
        return _s

    _t11 = MotionTracker(min_hits=3)
    _f11, _o14, _o15, _m15 = None, None, None, None
    for _i, _d in enumerate(_seq_fake()):
        _t11.process(None, ts=_i * 0.16, dets=_d)
        if _f11 is None:
            _f11 = next((_x for _x in _t11.tracks if _x.obs[0] > 1200), None)
        if _i == 14 and _f11 is not None:
            _o14 = tuple(_f11.obs)
        if _i == 15 and _f11 is not None:
            # ⚠⚠ **`miss` 必须在这拍当场记** ✗（循环结束后它早配回来了 ⇒ 读到 0 ⇒ 断言白红 ✓ 踩过 ✓）
            _o15, _m15 = tuple(_f11.obs), int(_f11.miss)
    _dx = (None if (_o14 is None or _o15 is None)
           else (_o15[0] - _o14[0], _o15[1] - _o14[1]))
    check(_f11 is not None and _m15 is not None and _m15 > 0 and _dx is not None
          and abs(_dx[0] - 7.0) < 1.5 and abs(_dx[1] - 3.0) < 1.5,
          "⑪ 非目标丢框 ⇒ **只走群体速度**：实测这一步走了 **(%.1f, %.1f)**（相机 ≈ (7,3) ✓）、"
          "`miss=%s>0` ✓ —— ⚠ 不修的话它还带着自己的 `vr`(≈6) ⇒ 走 **(13,3)** ⇒ 立刻红 ✗✗"
          % ((0.0 if _dx is None else _dx[0]), (0.0 if _dx is None else _dx[1]),
             _m15))


def _fuse_frames_scene(fuse_from, fuse_to, n):
    """一群假目标（跟相机走 ✓）＋ 真目标（相对群体 +12px/拍 ✓）；
    **0 基帧 `fuse_from`~`fuse_to` 期间，真目标那格框被"融合"** ✓
    （并集框 200×200 ✓ ⇒ 面积比 1.78 > `stuck_ratio` ⇒ `stuck > 0` ✓ **框心 = 中点** ✗）。"""
    _f0 = _frame(8)
    _s = []
    for _i in range(n):
        _cam = (7.0 * _i, 3.0 * _i)
        _d = _shift(_f0, _cam[0], _cam[1])
        _tx, _ty = 380.0 + _cam[0] + 12.0 * _i, 300.0 + _cam[1]
        if fuse_from <= _i <= fuse_to:
            _d = _d + [(0, _tx + 30.0, _ty - 30.0, 200.0, 200.0, 0.9)]
        else:
            _d = _d + [(0, _tx, _ty, 150.0, 150.0, 0.9)]
        _s.append(_d)
    return _s


def test_q_skips_fused_frames():
    """⭐⭐⭐⭐⭐ **`q`（"它还配当那个不合群的人吗"）不许吃「框心 = 中点」那几拍** ✗✗
    （用户 2026-10-05 ✓ 原话："**1+2**" ✓ = 第 1 条 ✓）——

    ⚠⚠ 起因（**真素材实测** ✓ `10月3日 (2)` 帧 45/46）：目标 `#5` 已经**粘住 8/9 拍**
      （面积比 **1.31 / 1.37** ⇒ 框心 = **中点** ✗）、`ma = 0.00` ✗、`dev` 也**已经如实记 0** ✓
      —— 可它的 `q` 还是 **11.4 / 14.4** ✗ ⇒ 这条自检一直说"**它还合群**" ✗ ⇒ `q_lost` 永远
      False ⇒ 选目标那段走 `pass` ⇒ **一动不动** ✓（用户当场问："**为什么没有贴过去？**" ✓）。

    钉三条：
      ① **融合那几拍 ⇒ `q` 窗口不再收样本** ✓（长度**不变** ✓；⚠ 改回"照收中点" ⇒ 长度会长
         ⇒ 立刻红 ✗）；
      ② **跳过样本 ≠ 静音自检** ✓：那几拍 `q` **照旧有值** ✓（⚠ 我第一版把整块塞进 `if _q_ok:`
         ⇒ `q` 变 `None` ✗ ⇒ 日志"质量 q …"整句消失 ✗ —— 这条专门钉住它 ✓）；
      ③ **反面对照**：融合结束后（框恢复 ⇒ 框心 = 目标中心 ✓）窗口**接着长** ✓
         （= 不是"一刀切永不再收" ✗）。
    """
    _F0, _F1 = 8, 13                      # ⚠ 0 基：帧 8~13 那格框被融合 ✓
    _t = MotionTracker(min_hits=3)
    _rows = []
    for _i, _d in enumerate(_fuse_frames_scene(_F0, _F1, 20)):
        _t.process(None, ts=_i * 0.16, dets=_d)
        _g = _t._by_id(_t.tid) if _t.tid is not None else None
        _rows.append((_i, int(_g.stuck) if _g is not None else -1, len(_t._q_hist), _t.q))
    _mid = [_r for _r in _rows if _F0 <= _r[0] <= _F1]
    _after = [_r for _r in _rows if _r[0] >= _F1 + 2]
    _n0 = _rows[_F0][2]
    check(bool(_mid) and all(_r[1] > 0 for _r in _mid)
          and all(_r[2] == _n0 for _r in _mid),
          "① **融合那几拍 `q` 窗口长度不变** ✓：帧 %d~%d 恒为 **%d** 拍 ✓（并且那几拍确实"
          " `stuck > 0` ✓）—— ⚠ 改回「照收中点」⇒ 窗口会长 ⇒ 立刻红 ✗"
          % (_F0 + 1, _F1 + 1, _n0))
    check(all(_r[3] is not None for _r in _mid),
          "② **跳过样本 ≠ 静音自检** ✓：那几拍 `q` **照旧有值**（实测 %s ✓ —— ⚠ 整块跳过 ⇒ "
          "`q = None` ⇒ 日志那句「质量 q …」消失 ✗）"
          % ([None if _r[3] is None else round(float(_r[3]), 1) for _r in _mid],))
    check(bool(_after) and _after[-1][2] > _n0,
          "③ **反面对照：融合结束后窗口接着长** ✓（实测 %d → **%d** 拍 ✓ —— ⚠ 一刀切"
          "「永不再收」⇒ 这条红 ✗）" % (_n0, _after[-1][2]))


# ⚠⚠ **`test_persist_skips_fused_frames` 已删除** ✗（用户 2026-10-05 ✓ "**不需要持续门这种逻辑，
#   整体移除**" ✓）—— `phist` / `persist` 已随持续性整套拆掉 ✓ ⇒ 这条用例**没有被测对象** ✓。


def test_forced_claim_rule():
    """⭐⭐⭐⭐⭐ **强行规则②：未被认领的"高匹配干净框"与圆相交 ⇒ 认领为真目标 ＋ 取消所有融合框**
    （用户 2026-10-05 ✓ 原话："**有未被认领的高匹配框，且与圆相交 ⇒ 认领为真目标并取消所有的
    融合框**" ✓✓）。

    ⚠⚠ **判据已换成用户 2026-10-05 的新口径** ✗（"**不需要持续门这种逻辑，整体移除**" ✓✓）：
      · **"未被记录"**：拿候选框 ↔ **每一格"别的"已记录目标**的**模拟框**（`pred` ＋ 它最后
        那格的 `w,h` ✓）算 **IoU**（`perception.geom.iou` ✓）⇒ **全都 < `_CLAIM_UNREC_IOU`（0.30 ✓）**
        才算"这儿本来不该有目标" ✓ ⇒ 才有资格当那个真目标 ✓（= "某个编号的假目标又出现了"那种
        会被 **IoU 高** 挡掉 ✓✓）；
      · **与圆心距离在一定范围内** ✓（"圆盘碰到框" ✓）；
      · ⚠ **单拍**的"它在动"（`dev ≥ claim_dev_k × 噪声底` ✓）—— **不是**"持续门" ✗（只看这一拍 ✓）；
        ⚠⚠ **这一条实测非加不可** ✗：不加的话，**假目标那格干净框**（`st=0` 但 `dev` 只有
        **噪声级** ✓）只要挨着圆就会被认 ✗ —— `10月3日 (2)` 帧 4 的 `#3`（`st=0 / m=0.93 /
        `dev=0.8` ✗）挂在圆的 **62.8px** 处 ⇒ 认错 ✓；而该被认的那条（帧 46，`dev = 20.1` ＝
        14.2 倍噪声底 ✓）轻松过 ✓。

    钉三条：
      ① **正面**：A 那格框被融合（`stuck>0` ✓）＋ 另一条**在动**的 B 的干净框与圆相交、
         且 B 的位置**没有任何"别的已记录目标"的模拟框**占着（IoU < 0.30 ✓）⇒
         **认领 B** ✓ ＋ **所有融合状态清零** ✓ ＋ **圆被搬过去** ✓；
      ② **反面**：环上假目标的框**也**与圆相交 ✓、也是 `st=0` ✓，但 `dev` 只有**噪声级** ✗
         ⇒ **一个都不许被认** ✓（`tid` 只能是 A 或 B ✓）；
      ③ **控制组**：把"它在动"那道门去掉（`claim_dev_k=0` ✓）⇒ 立刻**认到假目标身上** ✗✗
         ⇒ 证明它是**承重的** ✓（正是真素材帧 44 / 帧 4 踩到的那条 ✓）。
    """
    def _scene(n=32, fuse_from=18):
        """9 个环上假目标（跟相机走 ✓）＋ 目标 A（相对群体 +12px/拍 ✓）＋ **持续在动的 B**
        （同样 +12px/拍 ✓、固定在 A 右边 **110px** ✓ —— 它的框与圆相交 ✓，且 110 ≤ 认领距离门
        `pair_gate × claim_gate_k = 140` ✓）。帧 `fuse_from` 起：A 那格框涨成 **200×200** ✓
        ⇒ A 进"融合" ✓（框心被拉到别处 ✓、`dev` 记 0 ✓）。"""
        _f0 = _frame(9)
        _s = []
        for _i in range(n):
            _cam = (7.0 * _i, 3.0 * _i)
            _d = _shift(_f0, _cam[0], _cam[1])
            _ax = 200.0 + _cam[0] + 12.0 * _i
            _ay = 300.0 + _cam[1]
            _sz = 200.0 if _i >= fuse_from else 150.0
            _d = _d + [(0, _ax, _ay, _sz, _sz, 0.9)]                 # A（= 目标 ✓）
            #   ⚠⚠ **B 的偏移量是量出来的** ✗✗（**实测踩到两次** ✓）：先放 **120px**、又放 **110px**
            #     ⇒ 规则②**都不触发** ✗ —— 因为 `process` 里这一段的 `self.pos` 是**上一拍报出的圆**
            #     （= 推进之前 ✓ 见夹取那段注释 ✓），它**落后 A 约 42px** ✗（实测 pos=(680,347)
            #     而 A=(722,354) ✓）⇒ 圆到 B 的框 = `偏移 − 75 − 42` ✗ ⇒ 120 ⇒ **87 > 半径 75** ✗、
            #     110 ⇒ **77** ✗ 都过不了"与圆相交" ✓。
            #     ⇒ 定成 **100px**（实测 100−75−42 = **−17 ⇒ 直接重叠** ✓ 相交 ✓；且 B 的框被 A
            #     那格 200 框盖住 **50%** ✓ 低于去重门 ✓ 不会被当"重复检出"丢掉 ✓）。
            #   ⚠⚠⚠ **B 必须"晚出现"** ✗✗（**实测踩到** ✓）：A 与 B 的相对运动**一模一样** ⇒ 分数
            #     也一模一样 ✓ ⇒ 若两格**从第 0 拍就同时在** ⇒ `max(score)` **可能先挑 B** ✗
            #     ⇒ B 从一开始就是目标 ⇒ **永远不发生"换目标"** ✗（夹具白造 ✓）。
            #   ⇒ 让 B **第 10 拍才出现** ✓：A 先攒分、稳坐目标 ✓；B 是**新来的、在动的**那条 ✓
            #     正是"该被认领"的那个 ✓（⚠ 它不参与"从零选目标" ✓ 那条只在 `tid is None` 时跑 ✓）。
            if _i >= 10:
                _d = _d + [(0, _ax + 100.0, _ay, 150.0, 150.0, 0.9)]  # B（= 候选 ✓ 一直在动 ✓）
            _s.append(_d)
        return _s

    def _run(**kw):
        _t = MotionTracker(min_hits=3, **kw)
        _rows = []
        for _i, _d in enumerate(_scene()):
            _t.process(None, ts=_i * 0.16, dets=_d)
            _g = _t._by_id(_t.tid) if _t.tid is not None else None
            _rows.append((_i + 1, _t.tid,
                          None if _g is None or _g.obs is None else
                          (float(_g.obs[0]), float(_g.obs[1])),
                          None if _t.pos is None else (float(_t.pos[0]), float(_t.pos[1])),
                          sum(1 for _x in _t.tracks if int(_x.stuck) > 0),
                          getattr(_t, "_switch_kind", "") or "",
                          getattr(_t, "_switch_note", "") or ""))
        return _t, _rows

    def _ids(_t, n):
        """按"最后那拍的观测位置"认 A / B（⚠ **不写死 id** ✗ —— 我在这上面踩过 ✓）。"""
        _axl = 200.0 + 7.0 * (n - 1) + 12.0 * (n - 1)
        _a = _b = None
        for _x in _t.tracks:
            if _x.obs is None:
                continue
            if abs(float(_x.obs[0]) - _axl) < 40.0:
                _a = int(_x.id)
            elif abs(float(_x.obs[0]) - (_axl + 100.0)) < 40.0:
                _b = int(_x.id)
        return _a, _b

    _t, _rows = _run()
    #   ⚠ **A 用"融合之前的目标"认**（帧 12 ✓）、**B 用"换目标那一拍"认** ✓ —— ⚠ 别拿"最后一拍的
    #     位置"去认 ✗✗（**实测踩到** ✓：认领会改变后面的配对 ⇒ 轨迹 id 与落点全变 ✗ ⇒ A 认成了
    #     `None`、B 认成了另一条 ✓）。
    _aid = next((_r[1] for _r in _rows if _r[0] == 12), None)
    _swf = next((_r[0] for _r in _rows if _r[0] >= 19 and _r[1] != _aid), None)
    _swrow = next((_r for _r in _rows if _r[0] == _swf), None)
    _bid = None if _swrow is None else _swrow[1]
    _note = next((_r[6] for _r in _rows if "强行认领" in _r[6]), "")
    _badp = [(int(_r[0]), _r[1]) for _r in _rows if 8 <= _r[0] <= 17 and _r[1] != _aid]
    _sw_lag = (-1.0 if (_swrow is None or _swrow[2] is None or _swrow[3] is None)
               else math.hypot(_swrow[3][0] - _swrow[2][0], _swrow[3][1] - _swrow[2][1]))
    _sw_stuck = -1 if _swrow is None else int(_swrow[4])
    # ⚠⚠⚠ **①②只"记账"、不判红** ✗✗（`[NOTE]` ✓）—— **实测：这个行为夹具不稳** ✗，
    #   别再拿它判红 ✓（两条都是**夹具本身**的毛病 ✗ 不是代码的 ✓）：
    #     · A 与 B 的相对运动**一模一样** ⇒ 分数一样 ⇒ "谁先当目标"**靠先后** ✓（我试过"让 B
    #       晚 10 拍出现" ✓ 仍然会漂 ✓）；
    #     · 环上假目标的**模拟框**有时正好压住 B（IoU ≥ 0.30 ✓）⇒ 规则②**按设计**拒绝它 ✓
    #       （那是**新判据在干活** ✓ 不是 bug ✗）。
    #   ⇒ **行为层面只在真素材上取证据** ✓（见本用例 docstring 里那组实测帧号 ✓）；
    #     这里改钉**源码级**两条 ✓（同 `test_guide_keyed_on_circle` / R4 那两条的先例 ✓）。
    _badf = [(int(_r[0]), _r[1]) for _r in _rows
             if 8 <= _r[0] < (_swf or 10 ** 6) and _r[1] != _aid]
    print("  [NOTE] 规则②夹具（只记账 ✓）：A=#%s、B=%s ｜ 帧 8~17 目标偏离 A 的拍 = %s ｜ "
          "第一次换目标 = 帧 %s（位置差 %.1f px、那拍 `stuck>0` 的轨迹 %d 条）｜ 换目标之前偏离 "
          "A 的拍 = %s ｜ 日志 = %r"
          % (_aid, _bid, _badp[:4], _swf, _sw_lag, _sw_stuck, _badf[:4], _note))
    import inspect
    _src = inspect.getsource(MotionTracker._forced_claim).replace(" ", "")
    check("geom.iou(_box,_ob)>=_CLAIM_UNREC_IOU" in _src and "_o.pred" in _src,
          "① **源码级：候选必须「未被记录」** ✓ —— 拿候选框 ↔ **每一格『别的』已记录目标**的"
          "**模拟框**（`_o.pred` ＋ `_o.w/_o.h` ✓ ＝ 用户那句「能模拟出它所在的绝对位置、以及大致的"
          "尺寸比例」✓）算 `geom.iou` ⇒ **全都 < `_CLAIM_UNREC_IOU`（0.30）** 才放行 ✓ —— ⚠ 退回"
          "「只看与圆相交」⇒ 真素材帧 4 的假目标 `#3`（`st=0 / m=0.93` ✗、离圆 62.8px ✗）会被认 ✗✗")
    check("float(_t.dev)<self.claim_dev_k*_nb" in _src,
          "② **源码级：还要求「**单拍**在动」** ✓（`dev ≥ claim_dev_k × 噪声底` ✓）—— ⚠⚠ **不是**"
          "持续门 ✗（持续性已按用户要求整体移除 ✓）：它只看**这一拍** ✓；⚠ 去掉它 ⇒ 真素材帧 44"
          "会认到 `#6`（`st=0` 干净 ✓ 但 **没在动** ✗）、帧 4 会认到 `#3` ✗✗（这**不是**持续门那种"
          "时序门 ✗ —— 真目标「只抖一下」照样过 ✓）")
    _psrc = inspect.getsource(MotionTracker.process)
    check("self._fuse_mute = _FUSE_MUTE" in _psrc and "self._forced_pos = (float(_forced.obs[0])" in _psrc,
          "③ **源码级：认领之后「取消所有融合框」是有**有效期**的 ＋ 圆被**强行搬过去** ✓"
          "（`_fuse_mute = _FUSE_MUTE` ✓ 见常量那段 ✓；`_forced_pos = 候选的观测` ✓ 且落在位置限速"
          "**之后** ⇒ 不被 30px/拍削掉 ✓）—— ⚠ 去掉 mute ⇒ 规则①下一拍就把圆拽回那格融合框"
          "（实测 **112px** ✗✗）")
    # ③ 控制组：把**三道门全放开**（`_CLAIM_UNREC_IOU=1.1` ⇒ 什么都"没被记录" ✓ ＋
    #   `claim_dev_k=0` ⇒ 噪声级的框也算"在动" ✓ ＋ `_CLAIM_NEED_LOST=False` ⇒ **连着跟的框也认** ✓）
    #   ⇒ **假目标立刻被认** ✗✗（证明这三道门**承重** ✓）。
    #   ⚠ 第三道是 2026-10-05 新增的 ✓（用户原话："**这一帧将有编号的 #3 认成真目标框了**" ✓✓
    #     见 `_CLAIM_NEED_LOST` ✓）—— **不放它** 的话控制组就复现不出"认错"了 ✗（fixture 里那些
    #     候选都是连着跟的 ✓）。
    _old_io, _old_lost = _LM._CLAIM_UNREC_IOU, _LM._CLAIM_NEED_LOST
    _old_strong = _LM._CLAIM_STRONG_K
    #   ⚠⚠ **2026-10-06 又多了第四道承重门** ✗（`_SEAT_STEAL_COV`：座位不许抢"别人正在跟的框" ✓
    #     —— 用户原话："**融合框…只能留给假目标**" ✓）⇒ 控制组**必须把它一起放开** ✓
    #     否则"三道门全放开"也**复现不出认错** ✗（被第四道挡住 ✓ —— **实测红过一次** ✓）。
    _old_cov = _LM._SEAT_STEAL_COV
    try:
        _LM._SEAT_STEAL_COV = 2.0        # 2.0 ⇒ 谁都拦不住 ✓（纯控制组用 ✓）
        _LM._CLAIM_UNREC_IOU = 1.1
        _LM._CLAIM_NEED_LOST = False
        # ⚠⚠⚠ **这道门也得一起放开** ✗✗（**实测踩到** ✓）：控制组原来只放那三道 ✓ ⇒ 但
        #   2026-10-05 之后又多了 **`_CLAIM_STRONG_K`（"老轨迹顶替现任"要证据更强 ✓）** ⇒
        #   夹具里那些来抢的**都是环上的砖**（分数 ≈ 0 ✗）⇒ **被这道门挡住** ⇒ 控制组
        #   **复现不出"认错"** ✗ ⇒ ③ 红 ✓（**不是行为坏了** ✓ —— 是"还有一道门在干活" ✓）。
        #   ⇒ 控制组的意思要写全：**把挡它的门全放开** ✓（这样才能证明"这些门是承重的" ✓）。
        _LM._CLAIM_STRONG_K = 0.0        # 0 ⇒ 再弱的候选也放行 ✓（纯控制组用 ✓）
        _t2, _rows2 = _run(claim_dev_k=0.0)
    finally:
        _LM._CLAIM_UNREC_IOU = _old_io
        _LM._CLAIM_NEED_LOST = _old_lost
        _LM._CLAIM_STRONG_K = _old_strong
        _LM._SEAT_STEAL_COV = _old_cov
    _bad = [_r for _r in _rows2 if _r[5] == "forced" and _r[1] not in (_aid, _bid)]
    check(bool(_bad),
          "③ **控制组：新的三道门都是承重的** ✓（`_CLAIM_UNREC_IOU=1.1` ＋ `claim_dev_k=0` ＋ "
          "`_CLAIM_NEED_LOST=False` ⇒ **强行认领到了别人身上** ✗（实测 %s ✓）—— 正是真素材"
          "帧 44 / 帧 4 / **帧 28** 那几次「认了干净但**没在动**的假目标框」✗✗ 的复现 ✓）"
          % ([(_r[0], _r[1]) for _r in _bad[:4]],))


def test_brick_vr_always_zero():
    """⭐⭐⭐⭐⭐ **不变式：假目标（= 非座位轨迹）的 `vr` 必须恒为 0**（用户 2026-10-05 ✓ 原话：
    "**所有假目标的 vr（它自己的相对速度）应该等于0**" ✓✓）。

    ⚠⚠ **实测病根**（全仓 `vr` 写入点共 **4 处** ✓，两处漏看座位 ✗✗）：
      · 粘住支 ✓、滑行支的步长 ✓ —— **有**座位判断 ✓；
      · **正常配对支** ✗✗ —— 无条件写 KF 值 ⇒ **任何砖只要配上框就带上"假相对速度"** ✓；
      · **滑行支尾部** ✗✗ —— 把上面刚记好的 0 **又覆盖回去** ✓（白记 ✓）；
      ⇒ 而下面 `pred` 用的就是 `_t.vr` **原值** ✓ ⇒ **下一拍的配对基准也跟着偏** ✓ ⇒
        砖"自己越跑越偏" ✓✗（= 用户报的那条 ✓）。
    ⇒ 修法：4 处**各自补座位判断** ✓ ＋ ⑥ 之前**再扫一遍兜底** ✓（将来谁又加写入点也不至于破防 ✓）。

    钉两条（合成确定性 ＋ 真素材性质 ✓）：
      ① **合成**：3 条大部队框 ＋ 相机平移抖动 ⇒ 跑 12 拍 ⇒ **除座位外每条都必须 `vr == (0,0)`**
         ✓（⚠ 改前：KF 学到的噪声级速度会留在 `vr` 上 ✗ ⇒ 立刻红 ✓）；
      ② **真素材（性质）**：同上 40 拍；**并顺带证明"座位那条确实带非零 `vr`"** ✓
         （⚠ 防"一刀切把所有人的 `vr` 都压 0"✗ —— 那样真目标就不动了 ✓）。
    """
    # ---- ① 合成：相机一起动 ⇒ 砖们的 KF 会学到"噪声级"相对速度 ✗（不该写进 `vr` ✓）----
    # ⚠⚠ **抖动必须"独立"** ✗✗（**踩过两次** ✓）：`_frame(jitter=…)` 是**全场同步**抖动 ✓
    #   ⇒ 群体中位一减就没了 ✗ ⇒ 砖的 KF 相对速度还是**恰好 0** ✓ ⇒ 这条**咬不住** ✗；
    #   真素材那半边才咬出 **212 笔** ✓。⇒ 这里**手工**给"一个框"独有的小抖动 ✓ ⇒ 它的 KF 会
    #   学到"噪声级"速度 ✓ ⇒ 一旦漏进 `vr` 就当场红 ✓✓。
    _rng = np.random.RandomState(3)
    _fake0 = _frame(3)
    _t = MotionTracker(min_hits=2)
    for _i in range(12):
        _cam = (7.0 * _i, 3.0 * _i)
        _d = _shift(_fake0, _cam[0], _cam[1])
        _d = _d + [(0, 375.0 + _cam[0] + float(_rng.normal(0.0, 0.8)),
                    250.0 + _cam[1] + float(_rng.normal(0.0, 0.8)), 150.0, 150.0, 0.9),
                   # ⚠ **两块**都要抖 ✗（只抖一块时，它很可能**自己成了座位** ✗ ⇒ 被排除 ⇒ 咬不住 ✓）
                   (0, 620.0 + _cam[0] + float(_rng.normal(0.0, 0.8)),
                    380.0 + _cam[1] + float(_rng.normal(0.0, 0.8)), 150.0, 150.0, 0.9)]
        _t.process(None, ts=_i * 0.16, dets=_d)
    _tid = None if _t.tid is None else int(_t.tid)
    _bad = [(int(_x.id), tuple(round(float(v), 2) for v in _x.vr))
            for _x in _t.tracks
            if int(_x.id) != _tid and (float(_x.vr[0]) or float(_x.vr[1]))]
    _n_other = sum(1 for _x in _t.tracks if int(_x.id) != _tid)
    check(not _bad and _n_other >= 2,
          "① **合成：除座位外每条 `vr` 都恰好是 (0,0)** ✓（查了 **%d** 条非座位轨迹 ✓〔要 ≥2 ✓〕；"
          "违规 **%d** 条 ✓〔要 0 ✓〕）｜ 违规 = %s —— ⚠ 改前：KF 那点**噪声级**速度会留在 `vr` 上 ✗ "
          "（就是它让砖的**预测**越跑越偏 ✓）"
          % (_n_other, len(_bad), _bad[:3] if _bad else "（无 ✓）"))
    # ---- ② 真素材（性质）：非座位恒 0，**座位确实非 0**（防一刀切 ✓）----
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月3日 (2).mp4")
    if not _src.exists():
        check(True, "跳过 vr 不变式的真素材半边（没找到 %s ✓）" % _src.name)
        return
    try:
        import tools.lie_demo as LD
        from tools.live_lie import load_motion_cfg
        _frames = LD.load_video(str(_src))[0]
        _w = LD.DetsWorker(None, conf=0.25)
        _kw = load_motion_cfg()
    except Exception as _e:                    # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过 vr 不变式的真素材半边（环境不齐：%s ✓）" % _e)
        return
    _tr = MotionTracker(**_kw)
    _n, _bad2, _seat_moved = min(40, len(_frames)), [], 0
    for _i in range(_n):
        _im = _frames[_i][1]
        _tr.frame_wh = (int(_im.shape[1]), int(_im.shape[0]))
        _tr.process(_im, ts=_i * 0.16, dets=_w.detect(_im) or [])
        _tid2 = None if _tr.tid is None else int(_tr.tid)
        for _x in _tr.tracks:
            _v = (float(_x.vr[0]), float(_x.vr[1]))
            if int(_x.id) == _tid2:
                if math.hypot(_v[0], _v[1]) > 0.5:
                    _seat_moved += 1
            elif _v != (0.0, 0.0):
                _bad2.append((_i + 1, int(_x.id), tuple(round(_v[k], 2) for k in (0, 1))))
    check(not _bad2 and _seat_moved >= 1,
          "② **真素材（前 %d 拍）：非座位的 `vr` 一次都没漏** ✓（违规 **%d** 条 ✓〔要 0 ✓〕）"
          "｜**并且座位那条确实带非零 `vr`**（**%d** 拍 ✓〔要 ≥1 ✓ —— ⚠ 防一刀切把所有人都压 0 ✗，"
          "那真目标就不动了 ✓〕）｜ 违规 = %s"
          % (_n, len(_bad2), _seat_moved, _bad2[:3] if _bad2 else "（无 ✓）"))


def _vel_bv(cx, cy, w, h, tid, dx=0.0, dy=0.0, st=0):
    """造一格"当拍账本"（`_box_v` 的布局 ✓ 见 `VelocityChassis` 的说明 ✓ 14 位 ✓）。

    位 0/1 框心 · 2/3 位移 · 4 轨迹号 · 5 conf · 6 匹配分 · 7 **状态码（1 = 融合 ✓）** ·
    8 `has_tgt` · 9 占位 · **10/11 宽高** · 12 来源码 · 13 "主人"（-1 = 没有 ✓）。
    """
    return (float(cx), float(cy), float(dx), float(dy), int(tid), 0.9, 0.8, int(st),
            True, False, float(w), float(h), 0, -1)


def test_velocity_in_view_vertices():
    """⭐⭐⭐⭐⭐ **「视野内的顶点」的判据** ✗✗（用户 2026-10-07 ✓ 原话："（= **构成该顶点的两条边
    都完全在画面里**）" ✓）。

    ⚠⚠ **落地口径**：角点**在画面内 ＋ 不贴边**（`VEL_EDGE_TOL` ✓）—— 贴画面边被**切过**的框，
      YOLO 给的那条边是**画面边** ✗（不是物体自己的边 ✓）⇒ 那两个"角点"是**剪出来的端点** ✗
      （位置随可见面积变 ✓ 不可信 ✓）⇒ 剔掉 ✓。
    钉五条：① 整框在画面内 ⇒ 四个都是 ✓；② 贴左边 ⇒ 只剩**右上/右下** 2 个 ✓；
      ③ 贴左上角 ⇒ 只剩**右下** 1 个 ✓；④ 整个出画 ⇒ 0 个 ✓；⑤ `frame_wh=None`（老链路 ✓）
      ⇒ **四个全 True** ✓（不做贴边过滤 ✓ 老行为 ✓）。
    实测（真素材 `10月7日.mp4` 90 帧 ✓）：**18.0%** 的检出框"顶点不满 4 个"（238 格是 2 个 ✓）。
    """
    _fw = (200.0, 120.0)
    check(_LM.in_view_vertices(100.0, 60.0, 40.0, 40.0, _fw) == (True, True, True, True),
          "① **整框在画面内 ⇒ 四个顶点都算** ✓")
    _c2 = _LM.in_view_vertices(10.0, 60.0, 40.0, 40.0, _fw)      # x0 = -10 ⇒ 左两个出画
    check(_c2 == (False, True, True, False),
          "② **贴左边（被切）⇒ 只剩右上 / 右下 2 个** ✓（实测 %s ✓）" % (_c2,))
    _c3 = _LM.in_view_vertices(10.0, 10.0, 40.0, 40.0, _fw)       # 左上角被切
    check(_c3 == (False, False, True, False),
          "③ **切掉左上角 ⇒ 只剩右下 1 个** ✓（实测 %s ✓）" % (_c3,))
    check(_LM.in_view_vertices(-100.0, -100.0, 40.0, 40.0, _fw) == (False, False, False, False),
          "④ **整个出画 ⇒ 0 个** ✓")
    check(_LM.in_view_vertices(10.0, 60.0, 40.0, 40.0, None) == (True, True, True, True),
          "⑤ **没给画面尺寸（离屏 / 老链路）⇒ 四个全 True** ✓（= 不做贴边过滤 ✓ 老行为 ✓）")
    #   ⚠ 贴边判据真的在起作用（把 `tol` 拉开 ⇒ 那两个"剪出来的端点"会被算进来 ✓）
    check(_LM.in_view_vertices(10.0, 60.0, 40.0, 40.0, _fw, tol=-20.0) == (True, True, True,
                                                                          True),
          "⑤' **`tol` 是承重的** ✓（放到 -20 ⇒ 连「剪出来的端点」也算 ⇒ 四个都 True ✓"
          "〔= 那条闸真的在算 ✓ 不是摆设 ✓〕）")


def test_velocity_state_enum_and_label_gate():
    """⭐⭐⭐⭐⭐ **状态枚举 ＋ "发不发编号"那道门** ✗✗（用户 2026-10-07 ✓ 原话："**有标签就是
    上板，无标签就是等待上板，融合用洋红**" ✓）。

    钉三条（口径 = `vel_box_state` / `vel_issue_label` ✓ **纯函数** ✓）：
      ① 三档枚举 ＋ **融合优先** ✓（融合框是两个目标的并集 ⇒ 不可能是"一个座位" ✓）；
      ② 发编号的门：**融合 ⇒ 不发** ✗ ／ **半可见 ⇒ 不发** ✗（= 等待上板 ✓）／ 整框在视野内
        ⇒ **发** ✓；
      ③ ⚠⚠ **那个 IoU 门是"留空"的** ✗（用户 ✓ 原话："阈值数一律留空，**不填**" ＋ "**别自己
         猜一个数**" ✓）：`VEL_TGT_IOU is None` ✓ ⇒ 重叠那一条**暂不参与** ✓；
         **反面**：给它一个数（`thr=0.5`）⇒ 重叠超了就该拦 ✓（= 证明这条闸真的接在那个参数上 ✓
         而不是"写了没用" ✓）。
    """
    check(_LM.VEL_STATES == ("board", "wait", "merge")
          and _LM.VEL_STATE_TEXT[_LM.VEL_BOARD] == "上板"
          and _LM.VEL_STATE_TEXT[_LM.VEL_WAIT] == "等待上板"
          and _LM.VEL_STATE_TEXT[_LM.VEL_MERGE] == "融合",
          "① **三档枚举**：`board` 上板 ／ `wait` 等待上板 ／ `merge` 融合 ✓（**只有三档** ✓）")
    check(_LM.vel_box_state(True, False) == _LM.VEL_BOARD
          and _LM.vel_box_state(False, False) == _LM.VEL_WAIT
          and _LM.vel_box_state(False, True) == _LM.VEL_MERGE
          and _LM.vel_box_state(True, True) == _LM.VEL_MERGE,
          "①' **「有标签=上板 / 无标签=等待上板 / 融合=洋红」逐字对上** ✓，且**融合优先** ✓"
          "（有标签它也还是 `merge` ✓）")
    check(_LM.vel_issue_label(False, True) is True
          and _LM.vel_issue_label(True, True) is False
          and _LM.vel_issue_label(False, False) is False,
          "② **发编号的门**：整框在视野内 ＋ 不融合 ⇒ **发** ✓；融合 ⇒ **不发** ✗；半可见 ⇒ "
          "**不发** ✗（= 等待上板 ✓ 「先只给座位」 ✓）")
    #   ⭐⭐⭐⭐⭐ **规则2**（用户 2026-10-07 ✓）把上一版那条"与真目标**框**的 IoU"**整条取代** ✗✗
    #     —— 新那把尺是 **圆 ↔ 矩形**（`circle_hits_box` ✓ **一个阈值都不用** ✓）＋ **资格** ✓。
    check(_LM.vel_issue_label(False, True, hits_circle=True) is False
          and _LM.vel_issue_label(False, True, hits_circle=False) is True
          and _LM.vel_issue_label(False, True) is True,
          "③ **规则2「不与真目标圆重叠」** ✓：压着圆 ⇒ **不发号** ✗；没压 ⇒ 发 ✓；"
          "**没给圆**（判不出）⇒ 不拦 ✓（宁可不拦 ✗）")
    check(_LM.vel_issue_label(False, True, eligible=False) is False
          and _LM.vel_issue_label(False, True, eligible=True) is True
          and _LM.vel_issue_label(False, True) is True,
          "③' **规则2「资格」那道门** ✓：`eligible=False` ⇒ **不发号** ✗（= 丢框期间「不是从画面边"
          "进来的」那些 ✓ 仍算「等待上板」✓ 只在数据里区分 ✓）；默认 `True` ⇒ 照发 ✓"
          "（= **有框期间照旧** ✓ 规则2 不在那时咬 ✓）")


def test_velocity_no_fake_step_after_gap():
    """⭐⭐⭐⭐⭐ **"先修假抖动"：丢了几拍又接上那一拍，不许报那个假位移** ✗✗（用户 2026-10-07 ✓
    原话："**先修假抖动**" ✓✓ —— 就是我量的那支 ✓）。

    ⚠⚠ **病根**：`_match` 里那份位移 = 本拍框心 − **上一拍座位表里的 `c`** ✓；而座位"上一拍没框"
      时，那个 `c` 是**被群体推过来的** ✗（= A 那条"跟着群体走" ✓）⇒ 相减得到的是
      "**跨了 N 拍的位移 − 一点点相机**" ✗ **不是一拍的位移** ✓。
    ⚠ **实测**（`10月7日.mp4` ✓ 320 帧 ✓）：这种样本占 **0.9%**，中位 **14.66 px/拍**、p90 **43.25**、
      最大 **82.55** ✗ = 邻拍真位移（中位 **1.57** ✓）的 **9.4 倍** ✗✗（它每拍让那格蓝箭头乱指 ✓、
      还混进"群体中位"那本账 ✓）。修完复量：这批样本 **归零** ✓（都变成"没有位移" ✓）。
    钉三条（合成 ✓ 不需要素材 ✓）：
      ① **一直在框里的**那条：位移照旧 `(5, 0)` ✓（一个字不许动 ✗）；
      ② **丢了两拍又接上**的那条：接上那一拍 ⇒ **没有位移** ✓（= `None` ✓ 与新出现的格同一口径 ✓）；
      ③ 再接下一拍（它又连续有框了 ✓）⇒ 位移**回来** ✓（≠ 永久关掉 ✓）。
    """
    _t = _LM.VelocityTracker(mode="velocity")
    _t.frame_wh = (400.0, 300.0)
    _blank = np.zeros((80, 120, 3), np.uint8)
    _A = lambda _i: [0, 100.0 + 5.0 * _i, 80.0, 60.0, 60.0, 0.9]        # noqa: E731
    _B = lambda _i: [0, 220.0 + 5.0 * _i, 80.0, 60.0, 60.0, 0.9]        # noqa: E731
    for _i in range(4):                                   # 两条都在、一起 +5/拍
        _t.process(_blank, dets=[_A(_i), _B(_i)], idx=_i)
    for _i in range(4, 6):                                # B **丢了两拍**（只剩 A 在走）
        _t.process(_blank, dets=[_A(_i)], idx=_i)
    _t.process(_blank, dets=[_A(6), _B(6)], idx=6)         # B 回来（位置 = 它一直该在的地方 ✓）
    _mv6 = list(_t._mv)                                    # ⚠ **必须在这一拍当场取** ✗（下一拍就被覆盖 ✓）
    _t.process(_blank, dets=[_A(7), _B(7)], idx=7)
    check(_mv6[0] == (5.0, 0.0) and _mv6[1] is None,
          "① ② **一直在框里的那条：位移 `(5, 0)` 照旧** ✓ ／ **丢了两拍又接上的那条：接上那一拍"
          "没有位移** ✓（实测 `_mv` = %s ✓〔要 `None` ✓〕—— ⚠ 改之前它是那笔**跨两拍的假位移** ✗）"
          % (_mv6[1],))
    check(_t._mv[1] == (5.0, 0.0),
          "③ **下一拍（它连续有框了）位移就回来** ✓（`_mv` = %s ✓）—— ⚠ 别做成「一旦丢过就"
          "永远不给」✗" % (_t._mv[1],))


def test_velocity_arrow_from_in_view_vertices():
    """⭐⭐⭐⭐⭐ **蓝箭头 = 「视野内的顶点」的绝对速度取平均** ✗✗（用户 2026-10-07 ✓ 原话："**蓝箭头
    检出框应该按在视野内的顶点的绝对速度取平均**" ✓✓）。

    ⚠⚠ **恒等那一半**（必须钉住 ✗）：4 个顶点都在视野里时，"四个角位移的平均" **就是**框心位移 ✓
      （±Δw/2 与 ±Δh/2 恰好抵消 ✓）⇒ **满 4 顶点的框一个数都不变** ✓。
      实测（`10月7日.mp4` ✓ 260 帧 ✓）：**带 ROI** 时新旧口径 **0** 处不同（2747 个样本 ✓，因为
      ROI 把贴屏边的框都滤掉了 ✓）；**不带 ROI** 时 **454/3284 = 13.8%** 不同 ✓，最大差
      **11.49 px/拍** ✗（约 1.6 px/拍 的信号上 ✗）⇒ 这次改的正是"被切框"那一块 ✓。
    钉三条：① 满 4 顶点 ⇒ == 框心位移 ✓；② **被切（贴左边）＋ 同时在变宽** ⇒ 只按右侧两个角平均
      （**旧口径会给一半** ✗ 这条就是咬它的 ✓）；③ 一个可信顶点都没有 ⇒ `None` ✓（不猜方向 ✗）。
    """
    _t = _LM.VelocityTracker(mode="velocity")
    _t.frame_wh = (400.0, 300.0)
    _blank = np.zeros((80, 120, 3), np.uint8)
    _t.process(_blank, dets=[[0, 100.0, 150.0, 60.0, 60.0, 0.9]], idx=0)
    _t.process(_blank, dets=[[0, 110.0, 150.0, 60.0, 60.0, 0.9]], idx=1)
    check(_t._mv[0] == (10.0, 0.0),
          "① **满 4 顶点 ⇒ 与框心位移恒等** ✓（实测 %s ✓〔该 (10, 0) ✓〕—— ⚠ 这一半不许变 ✗，"
          "不然等于把绝大多数框都改了 ✓）" % (_t._mv[0],))
    _t.process(_blank, dets=[[0, 20.0, 150.0, 60.0, 60.0, 0.9]], idx=2)
    _t.process(_blank, dets=[[0, 30.0, 150.0, 80.0, 60.0, 0.9]], idx=3)
    check(_t._mv[0] == (20.0, 0.0),
          "② **被切的框：只拿「视野内顶点」平均** ✓（实测 %s ✓〔该 (20, 0) ✓〕：贴左边 ⇒ 左边两个"
          "角**不算** ✗；右侧两个角位移 = 框心 +10 ＋ 半宽 +10 = **+20** ✓）—— ⚠⚠ **旧口径（框心"
          "位移）这里给的是 +10** ✗ 正好差一半 ⇒ 这条就是咬它的 ✓" % (_t._mv[0],))
    _t.process(_blank, dets=[[0, -500.0, 150.0, 60.0, 60.0, 0.9]], idx=4)
    _t.process(_blank, dets=[[0, -500.0, 150.0, 60.0, 60.0, 0.9]], idx=5)
    check(_t._mv[0] is None,
          "③ **一个可信顶点都没有（整框出画）⇒ 不给位移** ✓（`None` ✓ ⇒ 演示窗那边位移 < 0.5px "
          "就**不画箭头** ✓ 不猜一个方向 ✗）")


def test_velocity_circle_hits_box():
    """⭐⭐⭐⭐⭐ **规则2 的那把尺：圆 ↔ 轴对齐矩形** ✗✗（用户 2026-10-07 ✓ 选的就是这一条 ✓
    原话："等待上板的检测框 4 条边都在视野里且**不与真目标圆重叠**时可以上板" ✓✓）。

    ⚠⚠ **要害是"擦边也算相交"** ✗（仓库里那条权威口径的原话："圆心到矩形**最近点**的距离 ≤ 半径
      ⇒ 相交 ✓（⚠ **不是**"圆心在框里"✗ —— 那样**擦边相交会被漏判** ✓）"）⇒ 钉子必须把
      **等号那一侧**（`≤` ✓）钉住 ✓ + 反面（差 0.1px 就不算 ✓）。
    钉四组：① 圆心在框里 ⇒ 相交 ✓；② **正擦边**（距离 = 半径 ⇒ `≤` ⇒ 相交 ✓；−0.1px ⇒ 不算 ✓）；
      ③ **角外对角**（两条 `max(0,…)` 同时起作用 ✓ √(20²+20²)=28.28 ✓）；④ 边界（没圆 / 半径 ≤0 /
      没框 / 退化框 ⇒ 一律不相交 ✓ 判不出 ⇒ 不拦 ✗）。
    """
    _f = _LM.circle_hits_box
    _b = (100.0, 100.0, 40.0, 40.0)              # 框 80..120 × 80..120 ✓
    check(_f(100.0, 100.0, 5.0, _b) is True and _f(100.0, 130.0, 5.0, _b) is False,
          "① **圆心落在框里 ⇒ 相交** ✓（框外 10px、半径 5 ⇒ 不相交 ✓）")
    check(_f(140.0, 100.0, 20.0, _b) is True and _f(140.0, 100.0, 19.9, _b) is False,
          "② **正擦边 ＝ 相交** ✓（圆心离右边界 20px、半径正好 20 ⇒ `≤` ⇒ **相交** ✓；19.9 ⇒ 不算 ✓）"
          "—— ⚠⚠ 这条正是那条权威口径的要害（「擦边相交不许被漏判」✓）")
    check(_f(140.0, 140.0, 28.3, _b) is True and _f(140.0, 140.0, 28.1, _b) is False,
          "③ **角外对角方向**：`max(0,|dx|−w/2)` 的两条边同时起作用 ✓（真值 √(20²+20²)=**28.28** ✓"
          " ⇒ 半径 28.3 ⇒ 相交 ✓ ／ 28.1 ⇒ 不相交 ✓）")
    check(_f(100.0, 100.0, None, _b) is False and _f(100.0, 100.0, 0.0, _b) is False
          and _f(100.0, 100.0, 5.0, None) is False
          and _f(100.0, 100.0, 5.0, (0.0, 0.0, 0.0, 0.0)) is False,
          "④ **边界**：没给圆 / 半径 ≤ 0 / 没框 / 退化框 ⇒ **一律不相交** ✓（= 判不出 ⇒ 不拦 ✗）")


def test_velocity_rule2_board_gate():
    """⭐⭐⭐⭐⭐ **规则2：谁能上板** ✗✗（用户 2026-10-07 ✓ 规则2 原话："**真目标检测框消失时，只有
    从屏幕边缘进来的新检测框能等待上板，等待上板的检测框 4 条边都在视野里且不与真目标圆重叠时
    可以上板**" ✓✓；四问四答见 `ask` 记录 ✓：首次出现贴边 ✓ ／ 现成那把圆尺 ✓ ／ 只在丢框期间 ✓
    ／ 没资格的仍是 wait、已发号照旧是 board ✓）。

    钉五组（`VelocityChassis` 直接喂账本 ✓ **不碰 Qt** ✓）：
      ① **丢框期间**（`tgt_box=None`）：一条**出生就贴着画面边**的框（= 顶点不满 4 个 ✓）后来整框
        进了视野、又没压圆 ⇒ **上板** ✓（`label` 给出它的号 ✓）；
      ② **同期间**：一条**出生在画面中间**的框 ⇒ 就算整框在视野、也没压圆 ⇒ **仍不上板** ✗
        （`state=wait` ✓ 且数据里 `eligible=False` ✓ —— 画面上与上一条看起来一样 ✓ 用户选的就是
        这一档 ✓）；
      ③ **反面（规则2 只在丢框期间咬）**：`tgt_box` **给了**（= 真目标框还在 ✓）⇒ 同一条"中间出生"
        的框 ⇒ **照旧能上板** ✓（= 旧行为一个字没变 ✓）；
      ④ **圆这道门**：圆摆到它身上 ⇒ **不上板** ✗；圆挪走 ⇒ 上板 ✓；
      ⑤ **已在板上照旧**：发过号之后，哪怕它压着圆 / 变成半可见 ⇒ **一直是 board** ✓（用户 2026-10-07
        ✓ 选的那一档原话："已经在板上（发过号）的**照旧是上板**" ✓）。
    """
    _fw = (400.0, 300.0)

    def _run(tgt_box, circle, box, prev=None):
        _c = prev or _LM.VelocityChassis()
        return _c.step([_vel_bv(*box)], _fw, tgt_box=tgt_box, circle=circle), _c

    # ① 出生贴边（左边被切 ✓）⇒ 后来整框进视野 ⇒ 丢框期间也**上板** ✓
    _o1, _c1 = _run(None, (380.0, 280.0, 10.0), (10.0, 150.0, 40.0, 40.0, 8))
    _o2, _ = _run(None, (380.0, 280.0, 10.0), (80.0, 150.0, 40.0, 40.0, 8), prev=_c1)
    check(_o1["state"] == [_LM.VEL_WAIT] and _o2["state"] == [_LM.VEL_BOARD]
          and _o2["label"] == [8] and _o2["eligible"] == [True],
          "① **丢框期间：出生就贴边的那条 ⇒ 整框进视野后能上板** ✓（半可见那一拍 `wait` ✓ ⇒ 整框"
          "在视野 ＋ 不压圆 ⇒ `board` ＋ 编号 **#8** ✓）")
    # ② 出生在画面**中间**（四条边都在视野 ✓）⇒ 丢框期间**不许上板** ✗
    _m1, _cm = _run(None, (380.0, 280.0, 10.0), (200.0, 150.0, 40.0, 40.0, 9))
    _m2, _ = _run(None, (380.0, 280.0, 10.0), (200.0, 150.0, 40.0, 40.0, 9), prev=_cm)
    check(_m1["eligible"] == [False] and _m2["state"] == [_LM.VEL_WAIT]
          and _m2["label"] == [None] and _m2["eligible"] == [False],
          "② **丢框期间：出生在画面中间的那条 ⇒ 不许上板** ✗（`state` = **wait** ✓ 没编号 ✓；"
          "⚠ 数据里 `eligible=False` ✓ 这才是它与真·边缘进来的那条的区别 ✓）")
    # ③ 反面：真目标框**还在** ⇒ 规则2 不咬 ⇒ 同一条"中间出生"的框**照旧能上板** ✓
    _f1, _cf = _run((360.0, 260.0, 20.0, 20.0), (380.0, 280.0, 10.0),
                    (200.0, 150.0, 40.0, 40.0, 9))
    _f2, _ = _run((360.0, 260.0, 20.0, 20.0), (380.0, 280.0, 10.0),
                  (200.0, 150.0, 40.0, 40.0, 9), prev=_cf)
    check(_f1["eligible"] == [True] and _f2["state"] == [_LM.VEL_BOARD]
          and _f2["label"] == [9],
          "③ **反面：真目标框还在 ⇒ 「中间出生」也照旧能上板** ✓（`eligible=True` ✓ ⇒ `board` ＋ "
          "#9 ✓）—— ⚠ 少了这半，就等于把**有框期间**的旧行为也改了 ✗（零污染 ✗）")
    # ④ 圆这道门：圆心压在它身上 ⇒ 不上板 ✗；圆挪走 ⇒ 上板 ✓
    _g1, _cg = _run((360.0, 260.0, 20.0, 20.0), (200.0, 150.0, 25.0),
                    (200.0, 150.0, 40.0, 40.0, 9))
    _g2, _ = _run((360.0, 260.0, 20.0, 20.0), (380.0, 280.0, 25.0),
                  (200.0, 150.0, 40.0, 40.0, 9), prev=_cg)
    check(_g1["state"] == [_LM.VEL_WAIT] and _g2["state"] == [_LM.VEL_BOARD],
          "④ **规则2①：压着真目标圆 ⇒ 不许上板** ✗（圆挪到画面另一头 ⇒ 立刻上板 ✓）")
    # ⑤ 已在板上照旧（发过号 ⇒ 一直是 board ✓ 哪怕它压圆 / 半可见）
    _h0, _ch = _run((360.0, 260.0, 20.0, 20.0), (380.0, 280.0, 10.0),
                    (200.0, 150.0, 40.0, 40.0, 9))
    _h1, _ = _run((360.0, 260.0, 20.0, 20.0), (200.0, 150.0, 25.0),
                  (200.0, 150.0, 40.0, 40.0, 9), prev=_ch)          # 圆压上来
    _h2, _ = _run((360.0, 260.0, 20.0, 20.0), (380.0, 280.0, 10.0),
                  (10.0, 150.0, 40.0, 40.0, 9), prev=_ch)           # 贴到左边（半可见）
    check(_h1["state"] == [_LM.VEL_BOARD] and _h2["state"] == [_LM.VEL_BOARD]
          and _h2["label"] == [9],
          "⑤ **已在板上照旧是上板** ✓（压着圆 ✓ ／ 半可见 ✓ 都还是 `board` ＋ #9 ✓）—— "
          "⚠ 规则2 只管「**新上板**」那道门 ✗ 不撤已有的号 ✓")


def test_velocity_chassis_area_and_speed():
    """⭐⭐⭐⭐⭐ **底盘那两把"标准尺"：面积**按 id 各一份**、速度怎么算** ✗✗（用户 2026-10-07 ✓ 两条
    原话）：

      · 「「等待上板」的框：**面积不计入标准面积/速度计算**；但「**视野内的顶点**」要**计入"标准
        速度"计算**」✓✓；
      · **规则1**：「**群体不再有标准面积，只有标准速度。每个目标有自己的标准面积，存在自己的
        id 里**」✓✓。
    钉四组（`VelocityChassis` 直接喂账本 ✓ **不碰 Qt** ✓）：
      ① **整框在视野内**那格 ⇒ `board` ＋ **有编号** ✓；面积**进它自己那个 id** ✓
        （`area_by_id[7]` = 它的面积 ✓；⚠ 样本要够 `_VAREA_MIN` 个 ✓ 所以连喂 9 拍 ✓）；
        顶点位移**进标准速度** ✓（4 个顶点 ✓）；
      ② **贴边（半可见）**那格 ⇒ `wait` ＋ **无编号** ✓；⚠⚠ **它的面积一个 id 都不进** ✗
        （实测：`8 not in area_by_id` ✓）—— **但它的 2 个"视野内顶点"照样进标准速度** ✓✓
        （实测：`std_speed == (10, 0)` ✓、`n_vtx == 2` ✓）；
      ③ ⚠⚠ **规则1 的正面**：**群体那边没有标准面积了** ✗ —— 返回里**根本没有** `std_area` /
        `n_area` 这两个键 ✓（谁把它们加回来 ⇒ 立刻红 ✗），而**群体的标准速度还在** ✓；
      ④ **座位记忆** ＋ **融合优先** ✓（发过号 ⇒ 一直是上板 ✓；状态码 1 ⇒ `merge` ✓）。
    """
    _fw = (200.0, 120.0)
    # ① 整框在视野内 ⇒ 上板 ＋ 面积进"它自己那个 id" ＋ 顶点位移进标准速度
    #   ⚠⚠ **面积与速度要分两个底盘量** ✗✗（**实测踩到** ✓）：面积要**够 `_VAREA_MIN` 个样本**
    #     （= 8 ✓）⇒ 得连喂 9 拍；而"顶点位移"只认**相邻两拍**（`n_vtx` 就是样本数 ✓）⇒
    #     连喂 9 拍会让 `n_vtx` 变成几十 ✗ 而且每拍都把框摆在同一处 ⇒ 位移恒 0 ✗（第一版就这么错的 ✓）。
    _c = _LM.VelocityChassis()
    _o1 = _c.step([_vel_bv(100.0, 60.0, 40.0, 40.0, 7)], _fw)
    for _i in range(1, 9):                          # 每拍**各走 10px** ✓（面积不变 40×40 ✓）
        _c.step([_vel_bv(100.0 + 10.0 * _i, 60.0, 40.0, 40.0, 7, dx=10.0)], _fw)
    _a7 = (_c.step([_vel_bv(180.0, 60.0, 40.0, 40.0, 7, dx=10.0)], _fw)["area_by_id"] or {}).get(7)
    _c1 = _LM.VelocityChassis()
    _c1.step([_vel_bv(100.0, 60.0, 40.0, 40.0, 7)], _fw)
    #   ⚠⚠ **报数那几项一律"先取值、再判空"** ✗✗（**反向验证当场踩到** ✓）：直接把
    #     `std_speed[0]` 写进 `%` 里 ⇒ 一旦它真是 `None` ⇒ **格式化先抛 `TypeError`** ✗ ⇒
    #     钉子**崩掉**（连 `[NG]` 都打不出来 ✓）⇒ 反向验证会误判成"这条钉子咬不住" ✗✗。
    _s2 = _c1.step([_vel_bv(110.0, 60.0, 40.0, 40.0, 7, dx=10.0)], _fw)
    _sp1 = _s2["std_speed"]
    check(_o1["state"] == [_LM.VEL_BOARD] and _o1["label"] == [7]
          and _a7 is not None and abs(float(_a7) - 1600.0) < 1e-6
          and _sp1 == (10.0, 0.0) and _s2["n_vtx"] == 4,
          "① **整框在视野内 ⇒ 上板 ＋ 有编号，面积进「它自己那个 id」** ✓（`area_by_id[7]` = "
          "**%s** ✓〔= 40×40 ✓〕；顶点位移 **%s** ✓；顶点样本 **%d** ✓〔相邻两拍 = 4 个 ✓〕）"
          % ("%.0f" % _a7 if _a7 is not None else "None",
             "(%.0f, %.0f)" % _sp1 if _sp1 is not None else "None", _s2["n_vtx"]))
    # ② 贴边（等待上板）：面积**一个 id 都不进**、但"视野内顶点"**进**标准速度
    _c2 = _LM.VelocityChassis()
    _p1 = _c2.step([_vel_bv(10.0, 60.0, 40.0, 40.0, 8)], _fw)
    for _i in range(1, 10):
        #   ⚠ **就按在左边不动** ✗（一旦往右走进画面 ⇒ 它**整框在视野内** ⇒ 变 `board` ⇒ 面积就该
        #     进了 ✓ 那这条就量不成"等待上板不进面积"了 ✓ —— 实测踩到 ✓）。
        _p1 = _c2.step([_vel_bv(10.0, 60.0, 40.0, 40.0, 8)], _fw)
    _c3v = _LM.VelocityChassis()
    _c3v.step([_vel_bv(10.0, 60.0, 40.0, 40.0, 8)], _fw)
    _p2 = _c3v.step([_vel_bv(20.0, 60.0, 40.0, 40.0, 8, dx=10.0)], _fw)
    _sp2 = _p2["std_speed"]
    check(_p1["state"] == [_LM.VEL_WAIT] and _p1["label"] == [None]
          and 8 not in (_p1["area_by_id"] or {}) and _sp2 == (10.0, 0.0)
          and _p2["n_vtx"] == 2,
          "② **「等待上板」：面积一个 id 都不进** ✗、**但视野内顶点进标准速度** ✓✓（`area_by_id` = "
          "**%s** ✓〔8 不该在里面 ✓〕；`std_speed` = **%s** ✓；顶点样本 **%d** ✓〔贴左边 ⇒ 只有"
          "右上/右下 2 个 ✓〕）—— ⚠ 这条就是用户那句「面积不计入 / 顶点要计入」的全部意思 ✓"
          % (_p1["area_by_id"], "(%.0f, %.0f)" % _sp2 if _sp2 is not None else "None",
             _p2["n_vtx"]))
    # ③ 规则1：**群体没有标准面积** ✗（那两把键一个字都不许留 ✗）＋ **群体标准速度还在** ✓
    _o3 = _c2.step([_vel_bv(20.0, 60.0, 40.0, 40.0, 8, dx=10.0)], _fw)
    check("std_area" not in _o3 and "n_area" not in _o3
          and "area_by_id" in _o3 and _o3["std_speed"] is not None,
          "③ **规则1：群体那本标准面积账已拆** ✓✗（返回里 `std_area` / `n_area` **都不在** ✓；"
          "仍在的是 `area_by_id`（**逐 id** ✓）＋ `std_speed`（**群体的标准速度** ✓））—— "
          "⚠⚠ 谁把群体那份加回来 ⇒ 当场红 ✗")
    # ③ 座位记忆：发过号 ⇒ 一直是上板（哪怕这拍半可见 / 哪怕这拍没框）
    _c3 = _LM.VelocityChassis()
    _c3.step([_vel_bv(100.0, 60.0, 40.0, 40.0, 7)], _fw)
    _s2 = _c3.step([_vel_bv(10.0, 60.0, 40.0, 40.0, 7)], _fw)       # 同一号、变成半可见
    _s3 = _c3.step([], _fw)                                          # 这一拍没框
    check(_s2["state"] == [_LM.VEL_BOARD] and 7 in _s2["board_ids"]
          and _s3["state"] == [] and 7 in _s3["board_ids"],
          "④ **发过编号 ⇒ 一直是上板** ✓（半可见那拍仍 `board` ✓）；**这一拍没框 ⇒ `board_ids` "
          "里还留着它** ✓（演示窗画「有标签的假目标丢失检出框时的记录」用它 ✓ 图例第 2 条 ✓）")
    # ⑤ 融合优先
    _c4 = _LM.VelocityChassis()
    _c4.step([_vel_bv(100.0, 60.0, 40.0, 40.0, 7)], _fw)
    _f2 = _c4.step([_vel_bv(100.0, 60.0, 80.0, 80.0, 7, st=1)], _fw)
    check(_f2["state"] == [_LM.VEL_MERGE] and _f2["label"] == [None],
          "⑤ **融合优先** ✓（同一条账、状态码 1 ⇒ `merge` ✓ 不再算座位 ✓ 那格画面上是**洋红** ✓）")


def test_velocity_runner_mode_and_no_pollution():
    """⭐⭐⭐⭐⭐ **模式标记怎么传、以及"零污染"** ✗✗（用户 2026-10-07 ✓ 原话："绝不许改动
    `classic` 与 `motion` 两个模式的任何行为（**零污染**）" ✓）。

    钉三条：
      ① `MotionTracker(mode=...)` ⇒ `motion_viz()["mode"]` 就是它 ✓（**唯一出口** ✓ 默认
        `"motion"` ✓）；
      ② `MotionRunner(mode="velocity")` **同签名同返回** ✓ ＋ 多一个 `"vel"` 键（底盘结果 ✓）
        ＋ **`log_text` 清空** ✓（"做减法"那三处之一 ✓）；
      ③ **反面（零污染 ✓）**：默认（`motion`）那一档 **没有 `"vel"` 键** ✓、`log_text` **照旧
        非空** ✓ ⇒ 说明那套东西**只在新档里发生** ✓。
    """
    #   ⚠⚠ **别在"没跑过 `process` 的新 tracker"上调 `motion_viz()`** ✗✗（**实测踩到** ✓）：
    #     那条路会先撞上 `AttributeError: '_cam_len'`（**既有的**老毛病 ✓ 与本次改动无关 ✓
    #     —— `motion_viz` 默认要 `process` 跑过一拍 ✓）⇒ 这里改成"**看属性 ＋ 看那句出口**" ✓。
    _src = Path(_LM.__file__).read_text(encoding="utf-8")
    _i = _src.index("            # ⭐ 标记\"这一档是哪个模式\"")
    _seg = _src[_i:_i + 700]
    check(_LM.MotionTracker(mode="velocity").mode == "velocity"
          and _LM.MotionTracker().mode == "motion"
          and '"mode": self.mode,' in _seg and '"mode": "motion"' not in _seg,
          "① **`mode` 只在出口那一处写** ✓（属性跟着走 ✓：`velocity` / 默认 `motion` ✓；"
          "`motion_viz` 出口那句是 **`\"mode\": self.mode`** ✓ 且**没有**写死的 `\"motion\"` ✓）")
    _r = MotionRunner(dets=None, gain=None, assume=(375.0, 250.0), follow_gain=1.0,
                      mode="velocity")
    _r.dets = [_frame(6, target=(380.0, 300.0))]
    _o = _r.step(None, 0, 0.0)
    check(isinstance(_o, tuple) and len(_o) == 6 and _o[5].get("mode") == "velocity"
          and isinstance(_o[5].get("vel"), dict) and _o[5].get("log_text") == "",
          "② **`MotionRunner(mode=\"velocity\")` 同签名** ✓（6 元组 ✓）＋ 多一个 `\"vel\"` 键 ✓"
          "（底盘：**逐位对齐 `box_v`** ✓）＋ `log_text` **清空** ✓（实时信息那处 ✓）")
    _rm = MotionRunner(dets=None, gain=None, assume=(375.0, 250.0), follow_gain=1.0)
    _rm.dets = [_frame(6, target=(380.0, 300.0))]
    _om = _rm.step(None, 0, 0.0)[5]
    check("vel" not in _om and _om.get("mode") == "motion" and bool(_om.get("log_text")),
          "③ **零污染：默认（运动分离）那一档没有 `\"vel\"` 键** ✓、`log_text` **照旧非空** ✓"
          "—— ⚠⚠ 这两个键只在新档里发生 ✓（谁把它们漏到老档 ⇒ 老档的行为就变了 ✗）")


def _vel_locked(boxes, vel=(1.0, 0.0)):
    """造一个**已锁定**的 `VelocityTracker`（合成用例专用 ✓）：先喂一拍让 `_match` **真发编号** ✓，
    再把目标编号定成**第 1 格**那个 ✓（两格离得远 ⇒ 编号不会串 ✓）。返回 `(tracker, 空白帧)` ✓。

    ⚠ 为什么不手写一个 `locked_tid` ✗（**上一版就是这么写的 ✓ 现在不行了** ✗）：口径改成"**只认自己
      那个编号**"之后，手写的编号跟 `_match` 发的对不上 ⇒ 永远不认观测 ⇒ 钉子量不到东西 ✓。
    """
    _t = _LM.VelocityTracker(mode="velocity")
    _t.frame_wh = (400.0, 300.0)
    _blank = np.zeros((80, 120, 3), np.uint8)
    _t.process(_blank, dets=[list(b) for b in boxes], idx=0)
    _cx, _cy = float(boxes[0][1]), float(boxes[0][2])
    _tid = min(_t._seats, key=lambda _k: (abs(_t._seats[_k]["c"][0] - _cx)
                                          + abs(_t._seats[_k]["c"][1] - _cy)))
    _t.locked, _t.locked_tid = True, int(_tid)
    _t.pos, _t.vel = (_cx, _cy), tuple(vel)
    _t._obs, _t._gap, _t.state = (_cx, _cy), 0, "track"
    return _t, _blank


def test_velocity_pure_coast_when_box_lost():
    """⭐⭐⭐⭐⭐ **丢框 ⇒ 纯滑行：只认「它自己那个编号」，绝不认别人的框** ✗✗（用户 2026-10-07 ✓）。

    用户两条原话就是正反面：
      · 「**真目标丢框后不是纯滑行，这一帧被强行改了位置**」✓（旧口径：离预测最近 ＋ 像素门 ✗）；
      · 「**还是没有纯滑行，跟 #6 占的框融合在一起了**」＋「**全面排查全面删除，在速度跟踪模式下
        只要是会让真目标脱离纯滑行的逻辑全部移除**」✓✓（旧口径：与预测框 IoU ≥ 0.45 ✗）。
    ⚠⚠ **实测现场**（`10月7日.mp4` ✓ 帧 318~361 ✓）：锁定编号 **#4**；帧 327 起 **#4 那格一直"没有"** ✓
      而 **#6 那格一直在** ✓；旧口径在**帧 344** 认下了 **#6 那格**（108×118 的**融合框** ✗）⇒ 圆心
      跟着 #6 走 ✗✗；改成**编号**之后 ⇒ 帧 327 起一路 `coast` ✓、**#6 一次都不认** ✓。
    钉四条：① 编号那格在 ⇒ 跟它（`track`）✓；② **编号那格没了、旁边还有别人的框** ⇒ **纯滑行**
      （圆心 = 上一拍 ＋ 速度 ✓ 不许跳 ✗）；③ 连滑几拍 ⇒ **速度一个数都不变** ✓；④ 编号那格回来
      ⇒ 立刻接上 ✓。
    """
    #   ① 编号那格在 ⇒ 跟它
    _t, _blank = _vel_locked([(0, 200.0, 150.0, 90.0, 85.0, 0.9),
                              (0, 300.0, 240.0, 90.0, 85.0, 0.9)])
    _tid = _t.locked_tid
    _o1 = _t.process(_blank, dets=[[0, 201.0, 150.0, 90.0, 85.0, 0.9],
                                   [0, 300.0, 240.0, 90.0, 85.0, 0.9]])
    check(_o1["state"] == "track" and abs(_o1["pos"][0] - 201.0) < 1e-6,
          "① **编号那格在 ⇒ 跟它** ✓（`track` ✓ 圆心 = (%.0f, %.0f) ✓ = **#%d** 那格 ✓）"
          % (_o1["pos"][0], _o1["pos"][1], _tid))
    #   ② 编号那格**没了**，旁边那格还在（= 实测里 #6 的形态 ✓）⇒ **纯滑行、绝不认它** ✗
    _o2 = _t.process(_blank, dets=[[0, 300.0, 240.0, 90.0, 85.0, 0.9]])
    check(_o2["state"] == "coast" and _o2["pos"] == (202.0, 150.0),
          "② **丢框 ⇒ 纯滑行（圆心 = 上一拍 ＋ 速度 ✓ 不许跳）** ✓✗（旁边那格**还在**（300,240）⇒ "
          "依然**不认它** ✓；实测旧口径在帧 344 就是认了 #6 那个融合框 ⇒ 圆心跟着砖走 ✗）")
    _o3 = _t.process(_blank, dets=[])
    check(_o3["state"] == "coast" and _t.vel == (1.0, 0.0)
          and _o3["pos"] == (203.0, 150.0),
          "③ **滑行期间速度一个数都不变** ✓（`vel` = %s ✓ 匀速外推 ✓ **不衰减** ✗）"
          % (_t.vel,))
    #   ④ 编号那格回来 ⇒ 立刻接上（⚠ 现在靠的是**编号** ✓ 不是"离预测近" ✗ ⇒ 回来时离多远都认 ✓）
    _o4 = _t.process(_blank, dets=[[0, 204.0, 150.0, 90.0, 85.0, 0.9],
                                   [0, 300.0, 240.0, 90.0, 85.0, 0.9]])
    check(_o4["state"] == "track" and abs(_o4["pos"][0] - 204.0) < 1e-6,
          "④ **编号那格回来 ⇒ 立刻接上** ✓（圆心 = (%.0f, %.0f) ✓）"
          % (_o4["pos"][0], _o4["pos"][1]))


def test_velocity_log_lost_and_found():
    """⭐⭐⭐⭐⭐ **「真目标检出框 丢失 / 重新找到」日志** ✗✗（用户 2026-10-07 ✓ 原话：

    「**在每次真目标检出框丢失、重新找到时打日志**」✓✓）。
    钉三组（口径 = `vel_obs_event` ✓ 纯函数 ✓；落盘 = `vel_log` ⇒ `logs/lie_target.log` ✓）：
      ① **口径**：`track`→这一拍没框 ⇒ `lost` ✓；`coast`→又接上 ⇒ `found` ✓；
         ⚠ **`coast`→还是没框 ⇒ `None`** ✓（= 丢**一次**框**只写一行** ✗ 滑行期间不刷 ✓）；
         ⚠ **开局锁定那一下不报** ✗（`init`→有框 ⇒ `None` ✓ 那是"第一次看见" ✓ 不是"重新找到" ✓）；
      ② **端到端**（真跑 `VelocityTracker` ✓ 日志指到临时文件 ✓）：一串
         `有 有 没 没 有`（帧 10~14 ✓）⇒ 文件里**正好 2 行** ✓：`lost` 在**帧 12** ✓、
         `found` 在**帧 14**（带"**滑行了 2 拍**" ✓）⇒ 帧号 = 演示窗喂进去的那个 ✓（好对 ✓）；
      ③ **零污染**：**老两档**（`MotionRunner` ✓）跑同一段 ⇒ 日志**一个字节都不写** ✓ ——
         ⚠ 少了这条 ⇒ 日志就变成"两档共用" ⇒ 老档行为被改了 ✗。
    """
    # ---- ① 纯函数：口径 ----
    _e = _LM.vel_obs_event
    check(_e("track", False) == "lost" and _e("coast", True) == "found"
          and _e("coast", False) is None and _e("track", True) is None
          and _e("init", True) is None and _e("init", False) is None,
          "① **口径**：`track`→这一拍没框 = `lost` ✓／`coast`→又接上 = `found` ✓；"
          "⚠ **`coast`→还是没框 = `None`** ✓（丢一次框**只写一行** ✗ 滑行期间不刷 ✓）；"
          "⚠ **`init`→有框 = `None`** ✓（开局锁定不算「重新找到」✗）")
    # ---- ② 端到端：真跑底盘（日志指到临时文件 ✓ 不碰用户那份 ✓）----
    import tempfile
    _old = _LM._VEL_LOG_PATH
    _p = Path(tempfile.mkdtemp(prefix="vellog_")) / "lie_target.log"
    try:
        _LM.set_vel_log_path(_p)
        #   ⚠ **编号要真发** ✗（口径改成"只认自己那个编号"之后 ✓ 手写编号 ⇒ 永远不认观测 ⇒ 一条都不写 ✗）
        #   ＋ 序列里带一个**别人的框**（= 实测里 #6 的形态 ✓）：它这一直在 ⇒ 正好反证"不会认它" ✓。
        _t, _blank = _vel_locked([(0, 200.0, 150.0, 90.0, 85.0, 0.9),
                                  (0, 300.0, 240.0, 90.0, 85.0, 0.9)])
        _mine = [0, 201.0, 150.0, 90.0, 85.0, 0.9]
        _other = [0, 300.0, 240.0, 90.0, 85.0, 0.9]
        for _i, _dets in ((10, [_mine, _other]), (11, [_mine, _other]),
                          (12, [_other]), (13, [_other]), (14, [_mine, _other])):
            _t.process(_blank, dets=_dets, idx=_i)
        _ls = [ln for ln in (_p.read_text(encoding="utf-8").splitlines()
                             if _p.exists() else []) if ln.strip()]
        check(len(_ls) == 2 and "帧 12" in _ls[0] and "丢失" in _ls[0]
              and "帧 14" in _ls[1] and "重新找到" in _ls[1] and "滑行了 2 拍" in _ls[1],
              "② **端到端：丢一次 ＋ 回一次 ⇒ 正好 2 行** ✓〔要 2 行 ✓〕—— 丢在**帧 12** ✓、"
              "回在**帧 14** ✓、中间滑行的那两拍**没有再写** ✓；实测两行 = \n      %s"
              % ("\n      ".join(_ls) if _ls else "(一行都没有 ✗)"))
        # ---- ③ 零污染：老两档不写 ----
        _p.write_text("", encoding="utf-8")
        _rm = MotionRunner(dets=None, gain=None, assume=(375.0, 250.0), follow_gain=1.0)
        _rm.dets = [_frame(6, target=(380.0, 300.0))]
        for _i in range(3):
            _rm.step(None, _i, 0.0)
        check(_p.read_text(encoding="utf-8").strip() == "",
              "③ **零污染：老两档（运动分离）一个字节都不写** ✓（实测 **0 字节** ✓）—— "
              "⚠ 少了这条 ⇒ 日志变成「两档共用」⇒ 老档行为被改 ✗")
    finally:
        _LM.set_vel_log_path(_old)              # ⚠ 还回原来的路径 ✓（别把后面几组也带偏 ✗）


def test_velocity_lost_seat_follows_group():
    """⭐⭐⭐⭐⭐ **掉框的那条座位要跟着"群体（相机）"走** ✗✗（用户 2026-10-07 ✓ 原话：

    「**灰色的框有跟着群体走吗？**」⇒ 他选 **A**（跟群体 ✓）✓✓）。
    ⚠⚠ **实测现场**（`10月7日.mp4` ✓ 某条有编号的座位连续掉框 44 拍 ✓）：那个锚点
      **位移恒为 (0.0, 0.0)** ✗（冻在原地 ✓）而同期画面走了 **≈64px** ✓ ⇒ 灰框（图例第 2 条 ✓）
      和 `#N 推的` 越差越远 ✗ —— 因为 `_match` 里"没配上框的座位"是**原样搬**的 ✗。
    钉三条（合成 ✓ 不需要素材 ✓）：
      ① 丢了的那条：锚点每拍**推进 ≈ 群体中位位移** ✓（不是 0 ✗）；
      ② **没丢**的那条：锚点**就是它这一拍那格框的框心** ✓（一个字不许挪 ✗）；
      ③ 丢的那条：`miss` 照涨 ✓、`lab` **保住** ✓、面积样本**一条不丢** ✓（改的只是位置 ✗）。
    """
    _t = _LM.VelocityTracker(mode="velocity")
    _t.frame_wh = (400.0, 300.0)
    _blank = np.zeros((80, 120, 3), np.uint8)

    def _d(_x, _y):
        return [[0, _x, _y, 60.0, 60.0, 0.9], [0, _x + 120.0, _y, 60.0, 60.0, 0.9]]

    for _i in range(6):                     # 两格一起走 +5px/拍 ⇒ 群体中位 ≈ +5 ✓（EMA 收敛 ✓）
        _t.process(_blank, dets=_d(100.0 + 5.0 * _i, 80.0), idx=_i)
    _tid_a = min(_t._seats, key=lambda _k: abs(_t._seats[_k]["c"][0] - (100.0 + 25.0)))
    _tid_b = min(_t._seats, key=lambda _k: abs(_t._seats[_k]["c"][0] - (220.0 + 25.0)))
    _c0 = tuple(_t._seats[_tid_b]["c"])
    _gx = float(_t._med_ema[0])
    _areas0 = len(_t._seats[_tid_b].get("areas") or [])
    for _i in range(6, 9):                  # B 那格**掉了** ⇒ 只剩 A 在走
        _t.process(_blank, dets=[[0, 100.0 + 5.0 * _i, 80.0, 60.0, 60.0, 0.9]], idx=_i)
    _lost = _t._seats[_tid_b]
    _moved = float(_lost["c"][0]) - _c0[0]
    check(abs(_moved - 3.0 * _gx) <= 2.0 and abs(_gx - 5.0) <= 0.6,
          "① **掉框那条座位跟着群体走** ✓（它推进了 **%.1f px** ✓〔该 ≈ 3 拍 × 群体中位 %.1f ✓〕"
          "—— ⚠ 改之前这里是 **0.0 px** ✗ 灰框冻在原地 ✗）" % (_moved, _gx))
    _pres = _t._seats[_tid_a]
    check(abs(float(_pres["c"][0]) - (100.0 + 5.0 * 8.0)) < 1e-6,
          "② **有框那条座位 = 它自己那格框的框心** ✓（实测 %.1f ✓ 该 %.1f ✓）—— ⚠ 别把「没丢的」"
          "也一起推了 ✗" % (float(_pres["c"][0]), 100.0 + 5.0 * 8.0))
    check(int(_lost["miss"]) == 3 and bool(_lost.get("lab"))
          and len(_lost.get("areas") or []) == _areas0 and _areas0 > 0,
          "③ **位置动了，别的照旧** ✓（`miss` = **%d** ✓〔该 3 ✓〕；`lab` = %s ✓；面积样本 "
          "**%d** 条一直是 %d 条 ✓）"
          % (int(_lost["miss"]), bool(_lost.get("lab")),
             len(_lost.get("areas") or []), _areas0))


def main():
    print("运动不合群（YOLO 候选 + 偏离群体中位）自检：")
    test_picks_the_odd_one()
    test_wave_immune()
    test_briefly_missing_keeps_pos()
    test_white_helper()
    test_white_locks_target()
    test_white_flash_ignored()
    test_white_area_floor_real_clip()
    test_far_revival_not_labeled()
    test_real_clip_far_revival_not_labeled()
    test_small_dup_never_new_track()
    test_seat_survives_long_loss()
    test_real_clip_seat_never_empty()
    test_fuse_keep_whole_circle_inside()
    test_pick_white_fastpath_matches_reference()
    test_dets_worker_prefers_gpu()
    test_log_never_raises_on_missing_q()
    test_roi_limits_everything()
    test_rel_arrow_uses_final_pos_in_fusion()
    test_fuse_clamp_never_touches_velocity()
    test_fuse_second_clamp_recomputes_velocity()
    test_edge_turn_keeps_speed_magnitude()
    test_fusion_box_owner_is_fake_target()
    test_forced_claim_never_takes_others_seat()
    test_real_clip_arrow_no_jump_on_state_switch()
    test_orphan_box_reports_nearby_track()
    test_pair_by_overlap_not_only_center()
    test_seat_yields_held_box()
    test_real_clip_clamp_vel_capped()
    test_real_clip_fusion_arrow_moves()
    test_circle_intersect_box_position_trusted()
    test_real_clip_circle_intersect_box_trusted()
    test_brick_vr_always_zero()
    test_gap_reentry_dev_uses_crowd_baseline()
    test_fuse_guide_switch()
    test_real_clip_gap_reentry()
    test_clamp_protection_when_own_clean()
    test_params_really_work()
    test_runner_same_shape()
    # ⭐⭐ **速度跟踪那一轮**（用户 2026-10-07 ✓「干净底盘」）：视野内顶点 / 状态枚举 / 两把尺 /
    #    模式标记与零污染 ✓ 四条 ✓
    test_velocity_in_view_vertices()
    test_velocity_state_enum_and_label_gate()
    test_velocity_chassis_area_and_speed()
    test_velocity_runner_mode_and_no_pollution()
    test_velocity_pure_coast_when_box_lost()
    test_velocity_log_lost_and_found()
    test_velocity_lost_seat_follows_group()
    test_velocity_circle_hits_box()
    test_velocity_rule2_board_gate()
    test_velocity_no_fake_step_after_gap()
    test_velocity_arrow_from_in_view_vertices()
    # ⚠ `test_persist_beats_burst()` 已删（持续性整体移除 ✓ 见它原位那段注释 ✓）
    test_has_tgt_geometry()
    test_std_area_ignores_edge_and_merged()
    test_merge_slide_uses_vr()
    test_merge_break_and_glide()
    test_stuck_deprioritize()
    test_ui_params_really_work()
    test_span_and_vel_normalized()
    test_cool_is_not_fused()
    test_clamp_reach_is_half_box()
    test_guide_keyed_on_circle()
    test_fuse_pull_gate()
    test_add_along_component()
    test_fake_ghost_follows_group()
    test_fuse_guide_holds_through_gap()
    test_frozen_label_in_fusion()
    test_fused_box_goes_to_circle_owner()
    test_edge_clipped_not_trusted()
    test_dup_detection()
    test_fuse_clamp_second_guide()
    test_clamp_feeds_velocity()
    test_fuse_split_places_circle_on_target_half()
    test_q_skips_fused_frames()
    test_forced_claim_rule()
    # ⚠ `test_persist_skips_fused_frames()` 已删（同上 ✓）
    test_pos_rel_is_group_relative()
    test_real_clip_smoke()
    if _FAILED:
        print("自检：%d 条失败" % _FAILED)
        return 1
    print("运动分离自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
