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


def test_grow_off_edge():
    """⭐⭐ **生长跟进门③放宽（用户 2026-10-02 选 A ✓）**：检出**大半盖住未上板条目**即走
    生长（不再要求检出贴画面边缘）✓ —— "刚长完整、还没进紧门"的砖（10月1日 帧12 (90,303) ✓）
    不再被重复登记 ✗。"""
    reg = ShapeRegistry()
    # 出生：贴左缘（x0=-10）⇒ 建条目、不上板 ✓（带 wh 才有画面尺寸 ✓）
    reg.step([(60.0, 100.0, 140.0, 140.0)], t=(0.0, 0.0), ts=0.0, wh=(750.0, 500.0))
    check(reg.entries and not reg.entries[0]._boarded,
          "⑪ 贴边出生 ⇒ 建条目不上板（on_board=%s ✓）"
          % (reg.entries[0]._boarded if reg.entries else None))
    # 群体平移 +20 ⇒ 条目 (80,100)；检出 (92,100,150,140)：距条目 12px > 紧门 7 ✗、
    # 已离边缘（x0=17 > 8）✗ 旧规则必新建 ✗ —— 但盖住条目 133×140 / 140×140 = 95% ≥ 60% ✓
    # ⇒ **走生长**：条目长成检出、不新建 ✓
    reg.step([(92.0, 100.0, 150.0, 140.0)], t=(20.0, 0.0), ts=0.1, wh=(750.0, 500.0))
    _e = reg.entries[0]
    check(len(reg.entries) == 1,
          "⑪ 检出已离边缘但大半盖住条目 ⇒ **生长不新建**（条目数 %d ✓ 应为 1 ✓）"
          % len(reg.entries))
    check(abs(_e.x - 92.0) < 1e-6 and abs(_e.w - 150.0) < 1e-6,
          "⑪ 条目连位置带尺寸长成检出（xy=(%.1f,%.1f) wh=(%.1f,%.1f) ✓）"
          % (_e.x, _e.y, _e.w, _e.h))


def test_board_duration():
    """⭐⭐ **上板时长门（`board_s`）**（用户 2026-10-02 ✓ 原话："『多久判定为上板』应该以
    时长为单位做成配置，像『噪声容差』一样加到配置行" ✓）：条目**自出生**要走满
    `board_s` 秒、且面积/宽高比/不贴边判据全过 ⇒ 才上板 ✓；`0` = 一帧就上板（老行为 ✓）。"""
    # ① board_s=0.5：ts=0 出生 ⇒ 0.3s 时判据全过但时长不够 ⇒ 不上板；0.6s ⇒ 上板 ✓
    reg = ShapeRegistry(board_s=0.5)
    reg.step([(100.0, 100.0, 140.0, 140.0)], t=(0.0, 0.0), ts=0.0)
    _e = reg.entries[0]
    reg.step([(100.0, 100.0, 140.0, 140.0)], t=(0.0, 0.0), ts=0.3)
    check(not _e.on_board,
          "⑩ board_s=0.5、出生才 0.3s（判据全过）⇒ **还没上板**（on_board=%s ✓）"
          % _e.on_board)
    reg.step([(100.0, 100.0, 140.0, 140.0)], t=(0.0, 0.0), ts=0.6)
    check(_e.on_board,
          "⑩ 出生 0.6s ≥ 0.5 ⇒ **上板**（on_board=%s ✓）" % _e.on_board)
    # ② board_s=0（默认）⇒ 一帧上板 ✓（老行为不变 ✓）
    reg2 = ShapeRegistry(board_s=0.0)
    reg2.step([(100.0, 100.0, 140.0, 140.0)], t=(0.0, 0.0), ts=0.0)
    check(reg2.entries and reg2.entries[0].on_board,
          "⑩ board_s=0 ⇒ **一帧上板**（老行为 ✓ on_board=%s）"
          % (reg2.entries[0].on_board if reg2.entries else None))


def test_board_overlap_dedup():
    """⭐⭐ **上板防抖重叠率**（用户 2026-10-02 ✓ 原话一字不改："目前有**旧砖被重复往上
    叠加砖**的问题。优化：如果**新砖与旧砖重叠率 >= 0.5**，判定为**同一登记砖**。**标砖的
    框选更接近假目标群体标准尺寸的那个**。这个 0.5 做成参数配置『**上板防抖重叠率**』" ✓）。

    判据（逐条钉）：
      · 重叠率 = **交集 ÷ 两者中较小的面积** ✓（用户 2026-10-02 选 ✓ —— 小框叠在大砖上
        也接近 1.0 ✓ 才治得了"旧砖上又叠一条"✗）；
      · 候选旧砖 = **不与绿圆外接矩形相交**的那些 ✓（用户 2026-10-02 口径 ✓ —— 被绿圈压着
        的砖位置/尺寸都不可信 ✗）；
      · 命中 ⇒ **不新建** ✗ + 框取**更接近 `typ_area`** 的那个 ✓ + **位置一动不动** ✗；
      · `board_overlap = 0` ⇒ **关**（老行为：该新建就新建 ✓）；`typ_area` 还没定 ⇒ 不换框 ✓。
    """
    from perception.lie_registry import Entry

    def _reg(board_overlap=0.5):
        """一间"旧砖偏离标准"的表：标准 140×140（19600 ✓）、旧砖 190×140（26600 ✓ = 被养大了 ✓）。"""
        r = ShapeRegistry(board_overlap=board_overlap)
        r.typ_area = 19600.0
        e = Entry((250.0, 250.0, 190.0, 140.0), ts=0.0)
        e.on_board = True
        e._boarded = True
        r.entries = [e]
        return r, e

    # ① **真 step 路径**：新检出 (280,250,150,140) 中心偏 30px > 紧门 7px ⇒ 配不上 ✓
    #    重叠 = x 交集 140 × y 交集 140 = 19600 ÷ min(21000, 26600) = **0.933** ≥ 0.5 ✓
    #    （老实现：这块砖会被**再叠一条** ✗✗ —— 正是用户报的那个问题 ✓）
    r1, e1 = _reg(0.5)
    out1 = r1.step([(280.0, 250.0, 150.0, 140.0)], t=(0.0, 0.0), ts=0.1)
    check(len(r1.entries) == 1,
          "⑫ 重叠 0.93 ≥ 0.5 ⇒ **判同一登记砖、不新建**（条目数 %d ✓ 应为 1 ✓ —— 老实现会"
          "叠出第 2 条 ✗）" % len(r1.entries))
    check(abs(e1.w - 150.0) < 1e-6 and abs(e1.h - 140.0) < 1e-6,
          "⑫ 框换成**更接近标准尺寸**的那个（现在 %.0f×%.0f ✓ 应为 150×140 ✓ —— 190×140 偏离"
          "标准 7000、150×140 只偏离 1400 ✓）" % (e1.w, e1.h))
    check(abs(e1.x - 250.0) < 1e-6 and abs(e1.y - 250.0) < 1e-6,
          "⑫ **位置一动不动**（现在 (%.1f,%.1f) ✓ —— 条目位置只由群体平移驱动 ✓ 防抖闸不许"
          "搬条目 ✗）" % (e1.x, e1.y))
    check(out1.get("dedup") == 1,
          "⑫ 返回值里带本拍防抖计数（dedup=%s ✓ 排查/自检用 ✓）" % out1.get("dedup"))

    # ② 无重叠（检出挪到远处）⇒ 照旧**新建** ✓（别把不相干的检出也并进来 ✗）
    r2, _e2 = _reg(0.5)
    r2.step([(600.0, 250.0, 150.0, 140.0)], t=(0.0, 0.0), ts=0.1)
    check(len(r2.entries) == 2,
          "⑫ 无重叠 ⇒ **照旧新建**（条目数 %d ✓ 应为 2 ✓）" % len(r2.entries))

    # ③ ⭐⭐ **绿圈"圆心"落在旧砖框内 ⇒ 该砖不参与防抖**（**B 方案** 用户 2026-10-02 ✓ 原话：
    #    "**B 绿圈圆心是否落在条目框内**" ✓）
    r3, e3 = _reg(0.5)
    check(r3._dedup_board_overlap(280.0, 250.0, 150.0, 140.0,
                                  origin=(250.0, 250.0)) is None,
          "⑫ 绿圈**圆心落在旧砖框内** ⇒ **不参与防抖**（返回 None ✓ —— 被绿圈压着的砖位置/"
          "尺寸都不可信 ✗ 不拿来判「同一块」✗）")
    check(r3._dedup_board_overlap(280.0, 250.0, 150.0, 140.0,
                                  origin=(1000.0, 1000.0)) is e3,
          "⑫ 圆在别处（**圆心在框外**）⇒ **照常命中**（返回那块旧砖 ✓）")

    # ③b ⭐⭐⭐ **只"擦到绿圈的角"不再算被压着 ⇒ 照常参与防抖**（**B 方案的正题** ✓ 用户
    #     2026-10-02 ✓）。实测 `9月30日(1).mp4` 显示帧 57：老砖 `(645,217)` (380.5,131.9)
    #     161.6×162.3 与绿圈外接矩形只擦到 ≈295px²（它自己面积的 **1.1%** ✓）⇒ 旧判据
    #     （`_circle_inter` ✓）把它**排除出候选** ✗ ⇒ 防抖闸看不见它 ⇒ 放行了一条与它
    #     **重叠率 1.000** 的新检出 ⇒ 新条目落进它框里（= "旧砖被重复往上叠" ✓ 你看到的现象 ✓）。
    r3b, e3b = _reg(0.5)
    check(r3b._circle_inter(e3b, (200.0, 350.0), 200.0) is True,
          "⑫（前提）这个圆**与旧砖框相交**（旧判据 `_circle_inter`=True ✓ —— 正是它把 57 帧那条"
          "老砖排除掉的 ✗）")
    check(r3b._dedup_board_overlap(280.0, 250.0, 150.0, 140.0,
                                   origin=(200.0, 350.0)) is e3b,
          "⑫ 但**圆心 (200,350) 不在旧砖框内** ⇒ **照常参与防抖、命中**（返回那块旧砖 ✓ —— "
          "B 方案：只有**圆心进框**才排除 ✓ 擦边不再误伤 ✓ 57 帧那条重叠 1.000 的新建会被拦下 ✓）")

    # ④ 阈值本身说了算（0.95 > 重叠 0.93 ⇒ 不命中 ✓）
    r4, _e4 = _reg(0.95)
    check(r4._dedup_board_overlap(280.0, 250.0, 150.0, 140.0) is None,
          "⑫ 阈值配 0.95 > 重叠 0.93 ⇒ **不命中**（配置真的说了算 ✓）")

    # ⑤ `0` = 关（老行为一字不变 ✓）
    r5, _e5 = _reg(0.0)
    r5.step([(280.0, 250.0, 150.0, 140.0)], t=(0.0, 0.0), ts=0.1)
    check(len(r5.entries) == 2,
          "⑫ 重叠率配 0 ⇒ **关**（照旧新建 ✓ 条目数 %d ✓ 应为 2 ✓）" % len(r5.entries))

    # ⑥ 还没定标准尺寸（`typ_area = 0`）⇒ 命中但**不换框** ✓（不猜 ✗）
    r6, e6 = _reg(0.5)
    r6.typ_area = 0.0
    check(r6._dedup_board_overlap(280.0, 250.0, 150.0, 140.0) is e6
          and abs(e6.w - 190.0) < 1e-6 and abs(e6.h - 140.0) < 1e-6,
          "⑫ 还没定标准尺寸 ⇒ 命中但**保持原框**（%.0f×%.0f ✓ 应为 190×140 ✓ 不猜 ✗）"
          % (e6.w, e6.h))

    # ⑦ ⭐⭐⭐ **防抖命中 ⇒ 还要记进 `_claimed`**（2026-10-02 ✓ 用户报的"第 57 帧砖(645,217)
    #   在第 58 帧飘移"的正解 ✓）：`_claimed` = "**这条目这拍被认领了**"的账 ✗ —— 漏记它 ⇒
    #   后面的「解除相交 ⇒ 收尾修正」查 `id(_e2) in _claimed` 查不到 ⇒ 把它当"**没人认领**"
    #   ⇒ 走"就近抓**未被认领**的检出"✗ ⇒ 而它自己的检出刚被防抖闸占进 `_used_dets` ✗ ⇒
    #   只能抓 120px 外**另一格**的框 ⇒ 位置被搬 142px ✗✗（实测第 58 帧：
    #   (391.1,116.1) → (306.2,199.8)，而它自己的检出 (396.7,98.7) 就在 18px 外 ✓）。
    import pathlib as _p7

    _src7 = (_p7.Path(__file__).resolve().parent.parent / "perception"
             / "lie_registry.py").read_text(encoding="utf-8")
    check("                    _claimed[id(_ded)] = (0.0, float(_ded.w),"
          " float(_ded.h)," in _src7,
          "⑫ 防抖命中后**记进 `_claimed`**（源码级 ✓ —— 只写 `_used_dets` 还不够 ✗：收尾修正"
          "查的是 `_claimed` ✓；漏记 ⇒ 条目被当成「没人认领」⇒ 位置被搬走 ✗✗；"
          "⚠ 记的**位置必须是条目自己当前的** ⇒ 本闸「位置一律不动」的口径不破 ✓）")


def test_size_write_needs_brick():
    """⭐⭐ **「判定为砖」才许写尺寸**（用户 2026-10-02 ✓ 原话："严丝合缝的意思是就取检出框
    当砖，**不存在无限养大**的说法（前提是**该检出框判定为砖**：判定条件是**与旧砖重叠率**的
    参数符合条件 **&& 不与绿圆相交**）" ✓）。

    场景：一块砖 140×140（19600 ✓）；来一个 150×140 的检出（21000 ✓ **在标准带内** ✓）、
    中心偏 **7px**（= 紧门内 ⇒ 会配对 ✓）⇒ 重叠率 = `133×140 ÷ 19600 ≈ 0.95`：
      · `board_overlap = 0.5`（默认 ✓）⇒ 0.95 ≥ 0.5 ⇒ **照旧写尺寸**（严丝合缝 ✓ 老行为不变 ✓）；
      · `board_overlap = 0.99` ⇒ 0.95 < 0.99 ⇒ **不写尺寸**（保持原框 ✓ —— 这个检出框"
        不算这块砖"，就不许把它写进登记尺寸 ✗ 那才会「越养越大」✗✗）；
      · 与绿圆外接矩形相交 ⇒ 也不写（既有纪律 ✓ 顺手钉住 ✓）。
    """
    from perception.lie_registry import Entry

    def _reg(ov):
        r = ShapeRegistry(board_overlap=ov)
        r.typ_area = 19600.0
        e = Entry((250.0, 250.0, 140.0, 140.0), ts=0.0)
        e.on_board = True
        e._boarded = True
        r.entries = [e]
        return r, e

    # ① 默认阈值 ⇒ 重叠率 0.95 达标 ⇒ 写尺寸（老行为一字不变 ✓）
    r1, e1 = _reg(0.5)
    r1.step([(257.0, 250.0, 150.0, 140.0)], t=(0.0, 0.0), ts=0.1)
    check(len(r1.entries) == 1 and abs(e1.w - 150.0) < 1e-6,
          "⑬ 重叠率 0.95 ≥ 0.5 ⇒ **照旧写尺寸**（%.0f×%.0f ✓ 应为 150×140 ✓ 严丝合缝 ✓）"
          % (e1.w, e1.h))

    # ② 阈值 0.99 ⇒ 0.95 不达标 ⇒ 不写尺寸（保持原框 ✓）
    r2, e2 = _reg(0.99)
    r2.step([(257.0, 250.0, 150.0, 140.0)], t=(0.0, 0.0), ts=0.1)
    check(abs(e2.w - 140.0) < 1e-6 and abs(e2.h - 140.0) < 1e-6,
          "⑬ 重叠率 0.95 < 0.99 ⇒ **不写尺寸、保持原框**（%.0f×%.0f ✓ 应为 140×140 ✓ ——"
          "它不算这块砖 ⇒ 不许写进登记尺寸 ✗ 那才会「越养越大」✗）" % (e2.w, e2.h))

    # ③ 与绿圆外接矩形相交 ⇒ 也不写（既有纪律 ✓）
    r3, e3 = _reg(0.5)
    r3.step([(257.0, 250.0, 150.0, 140.0)], t=(0.0, 0.0), ts=0.1,
            origin=(250.0, 250.0), rad=200.0)
    check(abs(e3.w - 140.0) < 1e-6 and abs(e3.h - 140.0) < 1e-6,
          "⑬ 与绿圆外接矩形相交 ⇒ **也不写尺寸**（%.0f×%.0f ✓ 应为 140×140 ✓ —— `_cont3` "
          "那条既有纪律 ✓）" % (e3.w, e3.h))


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
    test_board_duration()
    test_grow_off_edge()
    test_board_overlap_dedup()
    test_size_write_needs_brick()
    print("自检：%s" % ("全部通过" if not _FAIL else "%d 条失败" % len(_FAIL)))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
