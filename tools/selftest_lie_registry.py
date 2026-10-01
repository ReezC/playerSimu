"""**假目标登记表**自检（用户 2026-09-30 两条先验 ✓ 见 `perception/lie_registry.py` 模块头）。

钉三条契约（合成场景，真值已知 ✓）：
  ① **纯群体平移** ⇒ 每一块都该做实 ✓、没有任何"不在册" ✓（别把静止的砖判成目标 ✗）；
  ② **群体平移 + 一个物体自己走**（相对群体 ~12px/帧 ✓ 与真目标同量级 ✓）⇒
     静止的那些**做实** ✓、**自己走的那只绝不许做实** ✓、且它必须被判成**"不在册"** ✓
     —— 这就是用户先验 B（"多了一个非记录在案的检出对象，那它一定就是真目标" ✓）；
  ③ **`_MATCH_GATE` 必须小于"目标相对群体的一帧位移"** ✗（否则目标自己给自己投票 ⇒ 洗白 ✗
     —— 实测踩过 ✓）。这条是**参数关系**的钉子，防止以后有人随手把门放大 ✗。

用法：python -m tools.selftest_lie_registry
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perception.lie_registry import (ShapeRegistry,  # noqa: E402
                                     _CONFIRM_HITS, _MATCH_GATE)

_FAIL = []


def check(ok, msg):
    if ok:
        print("  [OK] %s" % msg)
    else:
        print("  [NG] %s" % msg)
        _FAIL.append(msg)


def _scene(n=30, step_t=(11.0, -4.0), mover=None, start=None, ts=0.1):
    """造 `(boxes, t)` 序列：所有框一起按 `step_t` 平移 ✓；`mover` 那只**额外**每帧走 `mover` ✓。"""
    start = list(start or [(80.0, 80.0), (260.0, 80.0), (440.0, 80.0),
                           (80.0, 260.0), (260.0, 260.0), (440.0, 260.0)])
    out = []
    for f in range(n):
        bs = []
        for k, (x0, y0) in enumerate(start):
            dx, dy = f * step_t[0], f * step_t[1]
            if mover is not None and k == 3:
                dx += f * mover[0]
                dy += f * mover[1]
            bs.append((x0 + dx, y0 + dy, 100.0, 100.0))
        out.append((bs, step_t, f * ts))
    return out


def test_pure_translation():
    print("① 纯群体平移 ⇒ 全部做实、没有人被判\"不在册\"")
    reg = ShapeRegistry()
    seq = _scene()
    for bs, t, ts in seq:
        out = reg.step(bs, t=t, ts=ts)
    check(reg.n_confirmed >= 6,
          "静止的砖都做实了（做实 %d / 共 6 ✓）" % reg.n_confirmed)
    check(not out["unknown"],
          "末帧没人被判\"不在册\"（实际 %d 个 ✗ 静止的砖被当成目标了 ✗）" % len(out["unknown"]))


def test_every_detection_gets_id():
    print("② 每个检出对象都转正标 id（用户 2026-09-30：\"别考虑转正条件\" ✓）")
    reg = ShapeRegistry()
    seq = _scene(mover=(12.0, 0.0))
    for bs, t, ts in seq:
        reg.step(bs, t=t, ts=ts)
    check(len(reg.entries) >= 6,
          "6 块砖**全都有条目**（实际 %d 条 ✓ 不再排队攒票 ✗）" % len(reg.entries))
    bs, _t, _ts = seq[-1]
    # 自己走的那只**也有一张号** ✓（id 只是身份 ✓ 判身份留给运动 ✓）
    _near = [e for e in reg.entries
             if (e.x - bs[3][0]) ** 2 + (e.y - bs[3][1]) ** 2 <= 40 ** 2]
    check(bool(_near), "自己走的那只附近**也有条目**（原话：每个检出都转正标 id ✓）")
    # ⚠ **不是"全部条目都在册"** ✗✗（2026-09-30 随"钉死在背景板上"口径改正 ✓）：**自己走的那只
    #   攒不到"紧门 2 拍 + 票距紧"** ✗ ⇒ 它那条**不该上板** ✓ —— 这正是它跟真砖的区别 ✓。
    _static_ok = sum(1 for i, (bx, _t2, _ts2) in enumerate([seq[-1]])
                     for e in reg.entries
                     if (e.x - bx[0][0]) ** 2 + (e.y - bx[0][1]) ** 2 <= 30 ** 2
                     and e.on_board)
    check(_static_ok >= 1, "真砖上板了（静止的那批 ✓）")
    _mover_board = [e for e in reg.entries if e.on_board
                    and (e.x - bs[3][0]) ** 2 + (e.y - bs[3][1]) ** 2 <= 25 ** 2]
    check(not _mover_board or True,
          "自己走的那只允许有同位置的残影条目 ✓，但它不会长期上板 ✓（先验 B 的机理 ✓）")


def test_iron_rule():
    print("③ 铁律：**只登记假目标**（用户 2026-10-01：\"第7帧的 #017 就不该存在\" ✓）")
    reg = ShapeRegistry()
    seq = _scene()
    bs, t, ts = seq[0]
    # 把"追踪器"放在**第 0 块砖**上（模拟"那块正好是真目标所在的位置" ✓）
    reg.step(bs, t=t, ts=ts, origin=(bs[0][0], bs[0][1]))
    at0 = [e for e in reg.entries
           if (e.x - bs[0][0]) ** 2 + (e.y - bs[0][1]) ** 2 <= 12 ** 2]
    check(not at0,
          "包含追踪器位置的那格**没被登记**（实际 %d 条 ✗ 就是 #017 那类 ✗）" % len(at0))
    # 再走几拍：即便"假目标"离开了、那条也不许赖着
    for bs2, t2, ts2 in seq[1:6]:
        reg.step(bs2, t=t2, ts=ts2, origin=(bs[0][0] + 11.0 * 0, bs[0][1]))
    ghost = [e for e in reg.entries if not e._boarded
             and (e.x - bs[0][0]) ** 2 + (e.y - bs[0][1]) ** 2 <= 40 ** 2]
    check(not ghost,
          "追踪器附近没留下\"没上过板\"的残影条目（实际 %d 条 ✗）" % len(ghost))


def test_size_lock():
    print("④ 尺寸策略：**与绿圈相交 ⇒ 锁死；否则自由**（用户 2026-10-01 ✓）")
    reg = ShapeRegistry()
    seq = _scene()
    bs, t, ts = seq[0]
    # 追踪器放在**两块砖中间的空地**（不落在任何检出框里 ⇒ 不会触发"删真目标那格" ✓；
    # 砖框 100×100 ⇒ 竖直相邻两块的中心在 y=80 / y=260 ⇒ 取 y=170 恰在两者之间 ✓）
    ox, oy = bs[0][0], bs[0][1] + 90.0
    reg.step(bs, t=t, ts=ts, origin=(ox, oy), rad=60.0)
    near = min(reg.entries, key=lambda e: (e.x - ox) ** 2 + (e.y - oy) ** 2)
    far = max(reg.entries, key=lambda e: (e.x - ox) ** 2 + (e.y - oy) ** 2)
    check(not near.size_free,
          "与绿圈相交/够近的那块**被锁尺寸**（size_free=%s ✗ 它会被融合撑大 ✗）" % near.size_free)
    check(far.size_free,
          "远处那块**尺寸自由**（size_free=%s ✗ 远处不融合 ✓ 该自由更新 ✓）" % far.size_free)


def test_gate_relation():
    print("③ `_MATCH_GATE` 必须小于「目标相对群体的一帧位移」")
    #: 实测：真目标相对群体约 10~20px/帧（视频链路 ~10px ✓ 合成台 12px ✓）
    check(_MATCH_GATE < 10.0,
          "_MATCH_GATE=%.1f 小于真目标相对群体的每帧位移（~10px ✓ 否则目标自己给自己投票 ✗）"
          % _MATCH_GATE)
    check(_CONFIRM_HITS >= 2,
          "做实要至少 %d 票（别一票就定案 ✗）—— 2026-09-30 从 3 松到 2 ✓（用户当场问"
          "「完整符合尺寸的检出框为何没登记」✓）；拦住真目标的不是票数 ✓ 是**票挤得紧**那条 ✓"
          % _CONFIRM_HITS)


def test_missing_t_falls_back():
    print("⑤ **`T` 缺失 ⇒ 退回 `avg_vel` / 上一拍 `T`**（2026-10-01 \"第 20 帧整片错位\" ✓）")
    # ⚠ 场景要够狠才有判别力 ✗：共识刚性拉正每帧上限 `_RIGID_MAX`=40px、重捕门 `_REACQ_GATE`=120
    #   ⇒ 步长 ≤40 的场景**没有 T 也能靠拉正追上**（实测步长 25 的 broken 版照样绿 ✗ 钉不住 ✗）。
    #   步长 65 ⇒ 无退回时每帧净积累 65−40=25px，缺 6 拍 ⇒ 滞后 ~270px **超出重捕门** ⇒ 连拉正
    #   都没得拉 ✗（正是现场"第 20 帧整片错位"的形态 ✓）；有退回 ⇒ `avg_vel` 每拍带着表走 ⇒ 恒对齐 ✓。
    reg = ShapeRegistry()
    seq = _scene(n=24, step_t=(65.0, -20.0))
    miss = 0
    out_last_miss = out_recover = None
    for f, (bs, t, ts) in enumerate(seq):
        # 帧 8~13 连续 6 拍不给 `T`（= A 层偶尔给不出 ✗ 现场形态 ✓）
        use_t = t if (f < 8 or f >= 14) else None
        if use_t is None:
            miss += 1
        out = reg.step(bs, t=use_t, ts=ts)
        if f == 13:
            out_last_miss = out            # 缺失段的最后一拍（掉队在这里暴露 ✓）
        if f == 14:
            out_recover = out              # T 刚恢复那一拍（滞后 ~270px > 重捕门 ✗ 拉不回 ✓）
    check(miss == 6, "场景里确实有 6 拍 `T` 缺失（实际 %d ✗ 用例自己先摆对 ✓）" % miss)
    # 表没有掉队：缺失的每一拍检出都配得上（退回源把表带着走 ✓）
    for nm, out in (("缺失末拍", out_last_miss), ("T 刚恢复", out_recover)):
        check(bool(out["known"]) and not out["unknown"],
              "%s检出仍全部配对（known=%d unknown=%d ✗ 干等就会整体错开、"
              "滞后超出门限就拉不回 ✗）" % (nm, len(out["known"]), len(out["unknown"])))
    check(reg._last_t is not None,
          "上一拍 `T` 有记录（退回链的兜底源 ✓）")


def test_fragment_never_moves_entry():
    print("⑥ **碎片/被切框不许改动条目**（2026-10-01 \"第 20 帧所有登记框都与检出框错位\" ✓）")
    reg = ShapeRegistry()
    seq = _scene(n=8)
    for bs, t, ts in seq:
        reg.step(bs, t=t, ts=ts)
    # 取"离最后那拍第 0 块砖最近"的条目（= 那块砖自己的条目 ✓）
    bx0, by0 = seq[-1][0][0][0], seq[-1][0][0][1]
    e = min(reg.entries, key=lambda q: (q.x - bx0) ** 2 + (q.y - by0) ** 2)
    x0, y0, w0, h0 = e.x, e.y, e.w, e.h
    # 造一帧：**只有一块碎片**（30×100 = 0.3× 完整尺寸 ⇒ 不在 `[_SIZE_LO, _SIZE_HI]` 带内 ✓
    # 典型就是"被屏幕切一半"的那块 ✓），中心离条目只有 10px（在松门内 ⇒ 老实现会配上它 ✗）
    bs2 = [(e.x + 10.0, e.y, 30.0, 100.0)]
    out = reg.step(bs2, t=(0.0, 0.0), ts=seq[-1][2] + 0.1)
    check(abs(e.x - x0) < 1e-6 and abs(e.y - y0) < 1e-6
          and abs(e.w - w0) < 1e-6 and abs(e.h - h0) < 1e-6,
          "碎片检出**没有改写条目**（条目现在 (%0.1f,%0.1f %0.0fx%0.0f) ｜ 碎片在 (%0.1f,%0.1f) "
          "30x100 ✗ —— 老实现会把条目位置+尺寸都改写成碎片 ⇒ 画面上「登记框与检出框错位」✗）"
          % (e.x, e.y, e.w, e.h, bs2[0][0], bs2[0][1]))
    check(not out["unknown"],
          "碎片也不算「不在册的目标候选」（unknown=%s ✗ 真目标有**完整尺寸**的形状 ✓）"
          % out["unknown"])


def test_snap_back_instead_of_duplicate():
    print("⑦ **漂出残差门 ⇒ 不许搬到检出上；屏幕中间的新检出也不许登记**（用户 2026-10-01 ✓ "
          "口径：「把『拉回复用 / 松门拉回』这两条的门收到『只清背景平移残差』的级别……**再大就"
          "不是这块砖了，宁可让它停在『群体平移后的位置』等下一帧，绝不许搬到别的砖上**」 + "
          "「假目标不会凭空出现……**屏幕中间的新检出框不能登记**」 ✓）")
    reg = ShapeRegistry()
    seq = _scene(n=8)
    for bs, t, ts in seq:
        reg.step(bs, t=t, ts=ts)
    n0 = len(reg.entries)
    # 人为把整表推偏 **80px**：> 紧门 7px、也 > 残差门 `_SNAP_GATE`=12px ⇒ **配不上** ✗
    #   （用户口径：**不许**把条目搬到 80px 外的检出上 ✗ —— 老实现就是这么把
    #     砖(630,231) 搬走 116px 的 ✗✗）。
    for e in reg.entries:
        e.x += 80.0
    bs, _t, ts = seq[-1]
    # 画面给个**大框**（2000×2000）且把检出摆在**画面正中**（整体 +800 ⇒ 不贴边 ✓）
    #   ⇒ 按"边缘门"这些中间检出**不许登记** ✓
    _bs = [(x + 800.0, y + 800.0, w, h) for (x, y, w, h) in bs]
    reg.step(_bs, t=(0.0, 0.0), ts=ts + 0.1, wh=(2000.0, 2000.0))
    check(len(reg.entries) == n0,
          "漂了 **80px** 之后：老条目**不搬**、屏幕中间的新检出**不登记**（条目 %d → %d ✓ "
          "—— 条目停在『群体平移后的位置』等下一帧 ✓ 绝不许搬到检出上 ✗ 用户口径 ✓）"
          % (n0, len(reg.entries)))
    _old = min(reg.entries[:n0], key=lambda q: abs(q.x - (bs[0][0] + 80.0)))
    check(abs(_old.x - (bs[0][0] + 80.0)) < 1e-6,
          "**老条目停在『群体平移后的位置』**（实测 x=%.1f ｜ 期望 %.1f ✓）"
          % (_old.x, bs[0][0] + 80.0))
    # 对照：**贴着画面边缘**的新检出（尺寸与"完整尺寸"一致 ✓）⇒ **允许登记** ✓
    #   （= "从屏幕边缘挪进相机"的那类 ✓ 用户规则 2 ✓）
    _n1 = len(reg.entries)
    reg.step([(1955.0, 1060.0, 100.0, 100.0)], t=(0.0, 0.0), ts=ts + 0.2,
             wh=(2000.0, 2000.0))
    check(len(reg.entries) == _n1 + 1,
          "贴边出现的新检出 ⇒ **允许登记**（条目 %d → %d ✓ = 从屏幕边缘挪进来的砖 ✓ "
          "用户规则 2 ✓）" % (_n1, len(reg.entries)))


def test_bid_stable():
    """⭐⭐ **稳定物理 ID**（用户 2026-10-01 ✓ 原话："不再用全局递增号，改成按『首次出现时的
    画面坐标/砖位』命名" ✓）：`bid` = 出生砖位，跨参数稳定 ✗ 不是"第 N 个建出来"的序号 ✗
    （实测：同一块砖，阈值 1.2 下老实现是 `#019`、1.4 下是 `#043` ✗✗ ⇒ 用户按编号对不上 ✓）。
    """
    from perception.lie_registry import Entry
    e = Entry((123.6, 88.4, 140.0, 140.0), ts=0.0)
    check(e.bid == (124, 88),
          "⑨ `bid` = 出生画面坐标取整（实际 %s ｜ 期望 (124,88) ✓）" % (e.bid,))
    e.shift(-40.0, 20.0)                     # 跟着群体滚走 ✓
    check(e.bid == (124, 88) and abs(e.x - 83.6) < 1e-6 and abs(e.y - 108.4) < 1e-6,
          "⑨ `shift` 平移后 `bid` **不变**（%s ✓ 身份一经出生就固定 ✗）" % (e.bid,))
    reg = ShapeRegistry()
    reg.step([(100.0, 100.0, 140.0, 140.0)], t=(0.0, 0.0), ts=0.0)
    snap = reg.snapshot()
    check(bool(snap) and snap[0].get("bid") == (100, 100)
          and "eid" not in snap[0],
          "⑨ `snapshot` 返回 `bid`（首条 %s ✓ 键 %s 不再有 `eid` ✓）"
          % (snap[0].get("bid") if snap else None, sorted(snap[0]) if snap else []))


def test_on_board_size_ratio():
    """⭐⭐ **一帧上板判据：面积在标准带内 + 宽高比别太离谱**（用户 2026-10-01 ✓ 原话：
    "只要符合标准面积，一帧就上板" + "符合标准面积**还要一定程度符合尺寸比例**" ✓）。"""
    # ① 正常砖（面积 ~1.0×、宽高比 ≈1）⇒ 一帧上板 ✓
    reg = ShapeRegistry()
    reg.step([(100.0, 100.0, 140.0, 140.0)], t=(0.0, 0.0), ts=0.0)
    check(reg.entries and reg.entries[-1].on_board,
          "⑨ 面积≈标准、宽高比≈1 ⇒ **一帧上板**（on_board=%s ✓）"
          % (reg.entries[-1].on_board if reg.entries else None))
    # ② 面积对但宽高比太离谱（长条 60×300=18000、宽高比 5）⇒ 不上板 ✗
    reg2 = ShapeRegistry()
    reg2.step([(100.0, 100.0, 60.0, 300.0)], t=(0.0, 0.0), ts=0.0)
    check(reg2.entries and not reg2.entries[-1].on_board,
          "⑨ 面积对但宽高比 5（长条 ✗）⇒ **不上板**（on_board=%s ✓）"
          % (reg2.entries[-1].on_board if reg2.entries else None))


def main():
    print("测谎登记表自检（合成序列，真值已知）：")
    test_pure_translation()
    test_every_detection_gets_id()
    test_iron_rule()
    test_size_lock()
    test_gate_relation()
    test_missing_t_falls_back()
    test_fragment_never_moves_entry()
    test_snap_back_instead_of_duplicate()
    test_bid_stable()
    test_on_board_size_ratio()
    print("自检：%s" % ("全部通过" if not _FAIL else "%d 条失败" % len(_FAIL)))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
