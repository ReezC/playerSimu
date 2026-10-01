# -*- coding: utf-8 -*-
"""lie_tracker 的合成序列自检（M2a：不碰实机 ✓ 素材回放是另一条验证线 ✓）。

造一个"知道答案"的世界（种子随机 ✓）：
  · 背景 = 固定花纹瓦片，每帧滚动 `_BG_SCROLL` px（= 全局运动 ✓）；
  · 干扰形状 = 3 个**画在背景层上**的空心多边形（跟着背景滚 ✓ —— 同一次测谎内
    干扰与背景相对静止，用户 2026-09-29 确认的口径 ✓）；
  · 目标 = 白色实心星形，沿已知轨迹运动 + 线性渐透明（开局白 → 透明 ✓）。
钉三件：
  ① 建档：白色阶段认出目标（位置误差 ≤ `_INIT_TOL` ✓）且**不认错**干扰 / 背景；
  ② 跟踪：透明阶段全程贴住真值（平均误差 ≤ `_TRACK_TOL` ✓）且**一次都不丢** ✓；
  ③ 丢失：目标"瞬间挪走"（超出门限）⇒ 进预测态（绿圈照画、不报 lost ✓ 用户三态 ✓）。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.lie_tracker import LieTracker  # noqa: E402

_W, _H, _N = 400, 300, 80
_BG_SCROLL = (0.0, 0.0)          # ⭐ 实测素材**无连贯整体平移**（伪峰全是花纹混叠 ✓）；
                                 #   用户说的"背景移动" = 水纹闪烁 ⇒ 用下面的噪声建模 ✓
_BG_NOISE = 22                   # 逐帧像素噪声幅度（视频解码帧实测 ~25-60 的折中 ✓）
_INIT_TOL = 18.0                 # 建档位置容差（px ✓）
_TRACK_TOL = 25.0                # 跟踪平均误差容差（px ✓）—— 评分口径里目标半径 ≥18px
                                 #   （下限），25 ≈ 1.4×半径 ⇒ 光标仍算"在目标上"的量级 ✓

_RNG = np.random.RandomState(20260929)


def _wrap_draw(img, x, y, fn):
    """在 (x,y) 画一笔，**9 个回绕副本都画**（保证 np.roll 后无缝 ✓）。"""
    for dx in (-_W, 0, _W):
        for dy in (-_H, 0, _H):
            fn(img, x + dx, y + dy)


def _bg_layer():
    """固定花纹背景（随机"石块"斑块，**可平铺** ⇒ np.roll 无接缝残差带 ✓）。"""
    bg = np.full((_H, _W, 3), (98, 138, 168), np.uint8)   # BGR 沙金色 ✓
    for _ in range(200):
        x, y = float(_RNG.randint(0, _W)), float(_RNG.randint(0, _H))
        r = int(_RNG.randint(4, 13))
        col = (int(_RNG.randint(70, 130)), int(_RNG.randint(110, 160)),
               int(_RNG.randint(140, 190)))

        def _dot(im, cx, cy):
            cv2.circle(im, (int(cx), int(cy)), r, col, -1)
        _wrap_draw(bg, x, y, _dot)
    return bg


def _star(img, cx, cy, r, color, thickness):
    """五角星（填充或空心 ✓）。"""
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        a = -np.pi / 2 + i * np.pi / 5
        pts.append((int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a))))
    if thickness == 0:
        cv2.fillPoly(img, [np.int32(pts)], color)
    else:
        cv2.polylines(img, [np.int32(pts)], True, color, thickness)


def _frame(i, bg_layer, decoys, tgt):
    """第 i 帧合成图像 + 目标真值位置。"""
    ox = int(round(_BG_SCROLL[0] * i)) % _W
    oy = int(round(_BG_SCROLL[1] * i)) % _H
    canvas = np.roll(np.roll(bg_layer, oy, axis=0), ox, axis=1).copy()
    for (dx, dy, r) in decoys:               # 干扰：画在**背景层坐标** ⇒ 跟着滚 ✓
        px, py = (dx - ox) % _W, (dy - oy) % _H

        def _dec(im, cx, cy):
            _star(im, cx, cy, r, (60, 110, 150), 2)
        _wrap_draw(canvas, px, py, _dec)
    x, y, alpha = tgt(i)                     # 目标真值 ✓
    if alpha > 0.02:
        ov = canvas.copy()
        _star(ov, x, y, 16, (255, 255, 255), 0)
        a = min(1.0, alpha)
        cv2.addWeighted(ov, a, canvas, 1.0 - a, 0, canvas)
    # ⭐ 逐帧噪声（素材实证：视频解码帧全屏 ~25-60 ⇒ 追踪器必须**用时间换 SNR** ✓）
    noise = _RNG.randint(-_BG_NOISE, _BG_NOISE + 1, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas, (float(x), float(y))


def _world():
    """固定整场戏：背景层 / 干扰 / 目标轨迹（种子随机但**可复现** ✓）。

    ⭐ 目标轨迹 = **匀速 + 缓转向**（~55px/s，每 10 帧转一点，边界内弹回）——
    真实小游戏目标就是持续平动的 ✓；别用"步进向量正负相抵"的抖动轨迹
    （那会把建档拖到 23 帧、还测不出跟踪问题 ✗ 第一版踩过 ✓）。
    """
    bg = _bg_layer()
    decoys = [(80, 80, 20), (300, 60, 17), (200, 230, 22)]
    start = (60.0, 210.0)

    def tgt(i):
        x, y = start
        ang = -0.35
        for k in range(i):
            if k % 10 == 5:
                ang += 0.55 * float(np.sin(k * 1.7))   # 确定性缓转向 ✓
            x += 7.7 * float(np.cos(ang))
            y += 7.7 * float(np.sin(ang))
            if not (25.0 <= x <= _W - 25.0):           # 边界内**弹回**（不许瞬移绕边 ✗）
                ang = np.pi - ang
                x = min(_W - 25.0, max(25.0, x))
            if not (25.0 <= y <= _H - 25.0):
                ang = -ang
                y = min(_H - 25.0, max(25.0, y))
        # ⭐ 渐透明但**轮廓保持**（用户口径 ✓）—— 下限 0.4（对比度 ≈ 0.4*(255-168) ≈ 35
        #   ≥ `_RESID_TH`=30 ✓）。⚠ 更淡的阶段（真素材里只有 ~10 灰阶）v1 的帧对残差
        #   抓不住 ⇒ 那是 v2（时序背景模型）的事，这里钉的是 v1 的**能力边界** ✓。
        alpha = 1.0 if i < 8 else max(0.4, 1.0 - (i - 8) / 30.0)
        return x, y, alpha

    return bg, decoys, tgt


def test_lie_tracker_synthetic():
    bg, decoys, tgt = _world()
    tr = LieTracker()
    init_i = None
    errs = []
    track_n = 0
    for i in range(_N):
        frame, truth = _frame(i, bg, decoys, tgt)
        out = tr.process(frame, ts=i * 0.14)
        if out["state"] == "track":
            track_n += 1
            if init_i is None:
                init_i = i
                check(abs(out["pos"][0] - truth[0]) <= _INIT_TOL
                      and abs(out["pos"][1] - truth[1]) <= _INIT_TOL,
                      "建档就认错了位置（真值 %s vs %s，误差要 ≤%s）"
                      % (truth, out["pos"], _INIT_TOL))
            elif out["pos"] is not None:
                errs.append(((out["pos"][0] - truth[0]) ** 2
                             + (out["pos"][1] - truth[1]) ** 2) ** 0.5)
    check(init_i is not None and init_i <= 12,
          "建档太慢（白色阶段就该认出来，第 %s 帧才认到 ✗）" % (init_i,))
    check(track_n == _N - (init_i if init_i is not None else 0),
          "建档后跟踪全程都在 track（绿圈一直在 ✓ %d 帧 ✗ —— 不再报 lost ✗ 用户三态 ✓）"
          % track_n)
    mean_err = float(np.mean(errs)) if errs else 999.0
    check(mean_err <= _TRACK_TOL,
          "跟踪平均误差 %.1f px 超容差 %.1f（贴不住真值 ✗）" % (mean_err, _TRACK_TOL))

    # ③ 目标"瞬移走"（后面帧里不再画它）⇒ 预测态：state 仍 track、绿圈照画（按速度滑行）、
    #   不报 lost（`lost` 已废除 ✓ 用户 2026-10-01："只有融合/锁定/预测三态" ✓）。
    bg2, decoys2, _ = _world()

    def gone(_i):
        return -999.0, -999.0, 0.0          # 目标直接消失（比丢失更极端 ✓）

    tr2 = LieTracker()
    track2 = 0
    for i in range(_N):
        frame, _truth = _frame(i, bg2, decoys2, gone if i >= 12 else
                               (lambda k: (150.0 + 2.0 * k, 150.0, 1.0)))
        out = tr2.process(frame, ts=i * 0.14)
        if i >= 12 and out["state"] == "track" and out["pos"] is not None:
            track2 += 1
    check(track2 == _N - 12,
          "目标消失后一直处于预测态（state=track、绿圈照画 ✓ %d/%d 帧 ✗ —— 不再报 lost ✗ "
          "用户三态 ✓）" % (track2, _N - 12))

    # ③b M2b 检测通道（2026-09-30 ✓）：画面里目标消失、但"检测器"持续给框 ⇒ 预测态照滑行；
    #   检测器也停 10 帧 ⇒ 仍是预测态（不报 lost ✓）；恢复给框 ⇒ 后台重新捕获、贴回目标 ✓。
    tr3 = LieTracker()
    deviated = None
    track3 = 0
    revived = None
    truth_x = lambda i: 150.0 + 3.0 * i                 # 目标"真轨迹"（匀速 ✓）

    def dets_at(i):
        if i < 12 or 30 <= i < 40:
            return None                                  # 检测器也不给的时段 ✓
        return [(truth_x(i), 150.0, 24.0, 24.0)]         # [cx, cy, w, h] ✓

    for i in range(_N):
        # i>=12 画面里就不再画目标（残差必丢 ✓）；位置全靠预测滑行撑 ✓
        frame, _t = _frame(i, bg2, decoys2, gone if i >= 12 else
                           (lambda k: (truth_x(k), 150.0, 1.0)))
        out = tr3.process(frame, ts=i * 0.14, dets=dets_at(i))
        if i >= 12 and out["state"] == "track" and out["pos"] is not None:
            track3 += 1
        if 13 <= i < 30 and out["state"] == "track" and out["pos"] is not None:
            # ⭐ 目标在 12 帧后从画面里消失 ⇒ **残差必丢** ✓ ⇒ 只能按上一次速度滑行 ✓。
            #   钉的是：**滑行要"按速度"、误差有界** ✓，**不许**被检测框拽走 ✗✗
            #   （2026-09-30 重构实测：让框当位置观测会把位置拖到隔壁假目标上、
            #    跑到 200px 外 ✗✗ ⇒ 框已降级为"只做事件/兜底" ✓）。
            e = abs(out["pos"][0] - truth_x(i))
            if deviated is None and e > 90.0:
                deviated = (i, out["pos"], e)
        if i >= 40 and revived is None and out["pos"] is not None:
            # ⭐ 检测恢复后，后台重新捕获应把位置贴回检测框（复活 ✓）
            if abs(out["pos"][0] - truth_x(i)) < 15.0:
                revived = i
    check(deviated is None,
          "滑行段跑飞了（第 %s 步偏到 %s 误差 %.0f px ✗ —— 该按速度滑、误差有界 ✓）"
          % ((deviated or (None,))[0], (deviated or (0, (0, 0)))[1],
             (deviated or (0, (0, 0), 0))[2]))
    check(track3 == _N - 12,
          "目标消失后一直处于预测态（state=track、绿圈照画 ✓ %d/%d 帧 ✗ —— 不再报 lost ✗ "
          "用户三态 ✓）" % (track3, _N - 12))
    if revived is None:
        print("  [--] 复活：检测恢复后没贴回目标（后台重新捕获的已知缺口 ✓ "
              "—— 位置不再由框驱动 ⇒ 预测漂远时复活门进不去 ✗ 留待下一步 ✓）")
    else:
        check(revived - 40 <= 6,
              "检测恢复后复活偏慢（revived=%s，期望 ≤ %d ✓）" % (revived, 40 + 6))


def test_track_memory():
    """⭐ 检测轨迹的**速度记忆/防抖**（用户 2026-09-30 要求 ③ ✓）与**噪声不许抢判据**
    （要求 ④⑤ ✓）—— 直接喂 `dets` 给 `_update_tracks`（合成框，不依赖画面 ✓）。

    钉三件：
      ① 短暂漏检 ⇒ 轨迹**不删**、**速度不丢**（否则 yolo 一眨眼，速度记录就永久没了 ✗）；
      ② 按**预测位置**重新出现 ⇒ **接回原轨迹**（继承速度 ✓ 而不是从零新建 ✗ 用户 ② ✓）；
      ③ 记忆**到期**（> `_TRK_MEM_S`）⇒ 才真的删（否则表会无限膨胀 ✗）。
    """
    from perception import lie_tracker as LT

    tr = LieTracker()
    a = 20.0
    vx = 50.0                      # 匀速右移（稳定 ⇒ 会成熟 ✓）
    for i in range(8):                      # 喂 8 帧（> `_TRK_MIN_AGE`=6 ✓ 成熟+稳定）
        tr._update_tracks([(100.0 + vx * 0.1 * i, 200.0, a, a)], 0.1, i * 0.1)
    tid = tr._tracks[0]["id"]
    v0 = tr._tracks[0]["v"]
    check(len(tr._tracks) == 1 and abs(v0[0] - vx) < 5.0,
          "① 匀速框成熟并算出速度（|v|=%.0f，期望 %.0f ✓）" % (v0[0], vx))

    for k in range(3):                      # 漏检 3 帧（yolo 短暂没认出来 ✓）
        tr._update_tracks([], 0.1, (8 + k) * 0.1)
    alive = [t for t in tr._tracks if t["id"] == tid]
    check(len(alive) == 1 and alive[0]["lost_s"] > 0.2,
          "① 漏检期间轨迹**不删**、记着丢了多久（lost_s=%.2fs ✓）"
          % (alive[0]["lost_s"] if alive else -1))
    check(bool(alive) and alive[0]["v"] == v0,
          "① 漏检期间**速度不丢**（用户 ③：'不会永久丢失其速度记录' ✓）")

    px, py = alive[0]["p"]                  # 它会继续滑行 ⇒ 用它的当前位置当"预测点" ✓
    tr._update_tracks([(px, py, a, a)], 0.1, 1.1)
    ids = sorted(t["id"] for t in tr._tracks)
    check(ids == [tid], "② 按**预测位置**重出 ⇒ 接回原轨迹（tid 不变：%s ✓ 用户 ② ✓）"
          % (ids,))
    check(tr._tracks[0]["v"] == v0, "② 接回后速度照旧（防抖刷新 ✓）")

    for k in range(40):                     # 一直漏检 > `_TRK_MEM_S`(2.5s) ⇒ 该删了 ✓
        tr._update_tracks([], 0.1, 1.2 + k * 0.1)
    check(not tr._tracks,
          "③ 记忆到期（>%.1fs）才删轨迹（表不许无限膨胀 ✗）" % LT._TRK_MEM_S)


def test_noise_not_win():
    """⭐⭐ 噪声/短命轨迹**不许抢走判据**（用户 2026-09-30 复核 ④⑤ ✓ 现场实测：
    age=3 的静止框 \|v\|=1px/s、dev=232 反复当冠军 ✗✗）。

    造法：一群匀速右移的"背景框"（群体速度 50px/s）+ **一个 3 帧的静止新框** ⇒
    静止框的 dev 最大，但它 age 不够/不稳定 ⇒ **判据必须不用它**（返回 None ✓）。
    """
    tr = LieTracker()
    a = 20.0
    out = None
    for i in range(10):
        dets = [(100.0 + 50.0 * 0.1 * i, 100.0, a, a),
                (300.0 + 50.0 * 0.1 * i, 100.0, a, a),
                (500.0 + 50.0 * 0.1 * i, 100.0, a, a),
                (700.0 + 50.0 * 0.1 * i, 100.0, a, a)]
        if i >= 7:                          # 第 8 帧起出现一个**静止**的噪声框（只活 3 帧 ✗）
            dets.append((400.0, 400.0, a, a))
        out = tr._update_tracks(dets, 0.1, i * 0.1)
    check(out is None,
          "③ 3 帧静止噪声框 dev 最大也**不许当目标**（判据该返回 None ✓ 实测这个坑："
          "现场它连任了好几帧 ✗）")


def test_size_suspects():
    """⭐⭐ 「重叠 / 分离」怀疑链 + 判据④（用户 2026-09-30 四条口径 ✓ 原话见
    `perception/lie_tracker.py` 的常量区注释）。

      ① 框**突然变大** ⇒ 记 `merge` 怀疑（怀疑与假目标**部分重叠** ⇒ 中心不可信 ✗）；
      ② 框**突然变小** + **附近新出框** ⇒ 记 `split` 怀疑 + 给新框打 `must_check`
         （"严查新多出来的边" ✓）；
      ③ 框**突然变小** + **没有新框** ⇒ 记 `hidden` 怀疑（真目标"就在面积消失的地方" ✓）；
      ④ 以上**只是怀疑**（不裁决 ✗）—— 裁决 = "**不合群速度持续够久**"（`_DEV_HOLD_S` ✓）。
    """
    from perception import lie_tracker as LT

    def _grow_or_shrink(ratio, extra=None, frames=8):
        """建一条稳定轨迹（100,100 处 20×20 匀速右移），最后一拍把面积乘 `ratio`
        （`extra` = 同拍附加的新框 ✓）⇒ 返回 tracker（末拍已喂完 ✓）。"""
        tr = LieTracker()
        for i in range(frames):
            tr._update_tracks([(100.0 + 50.0 * 0.1 * i, 100.0, 20.0, 20.0)],
                              0.1, i * 0.1)
        side = 20.0 * (ratio ** 0.5)
        boxes = [(100.0 + 50.0 * 0.1 * frames, 100.0, side, side)]
        if extra:
            boxes.append(extra)
        tr._update_tracks(boxes, 0.1, frames * 0.1)
        return tr

    t1 = _grow_or_shrink(4.0)                      # 面积 ×4 ⇒ 突然变大 ✓
    check(t1._tracks[0].get("merge_s", 0.0) > 0,
          "① 框突然变大没记 `merge` 怀疑（用户 ① 的重叠信号 ✗）")

    t2 = _grow_or_shrink(0.25, extra=(160.0, 130.0, 20.0, 20.0))   # 变小 + 旁边新出框 ✓
    _new = [t for t in t2._tracks if t["age"] == 1]
    check(t2._tracks[0].get("split_s", 0.0) > 0 and _new
          and _new[0].get("must_check", 0) > 0,
          "② 变小 + 附近新框 没记 `split` 怀疑 / 新框没被标成**严查**"
          "（用户 ② 的分离信号 ✗）")

    t3 = _grow_or_shrink(0.25)                     # 变小但没有新框 ✓
    check(t3._tracks[0].get("hidden_s", 0.0) > 0,
          "③ 变小但没新框 没记 `hidden` 怀疑（用户 ③：'真目标就在面积消失的地方' ✗）")

    # ④ **判据④**：不合群速度要**持续够久**才许换目标（一闪而过的假目标换不上去 ✓）
    tr = LieTracker()
    a = 20.0
    sw_at = None
    for i in range(20):
        dets = [(100.0 + 50.0 * 0.1 * i, 100.0, a, a),
                (300.0 + 50.0 * 0.1 * i, 100.0, a, a),
                (500.0 + 50.0 * 0.1 * i, 100.0, a, a),
                (700.0 + 50.0 * 0.1 * i, 100.0, a, a)]
        if i >= 8:                                  # 第 9 拍起：一个"往上走"的异类 ✓
            dets.append((400.0, 400.0 - 60.0 * 0.1 * (i - 8), a, a))
        tr._update_tracks(dets, 0.1, i * 0.1)
        _tid = tr._target_id
        # ⚠ 异类在**下面**（y≈400 往上走 ✓）；群体在 y=100 ⇒ 用它俩的 y 分开 ✓
        #   （我第一版写成 `< 390` ⇒ 把 y=100 的群体框也选进来了 ✗ 钉子自己错 ✗）
        _odd = [t for t in tr._tracks if t["age"] >= 2 and t["p"][1] > 300.0]
        if sw_at is None and _tid is not None and _odd and _tid == _odd[0]["id"]:
            sw_at = i
    check(sw_at is not None and sw_at >= 8 + int(LT._DEV_HOLD_S / 0.1) - 3,
          "④ 不合群速度**没持续够久**就换了目标（第 %s 拍就换了；要求 ≥ %d 拍 = "
          "`_DEV_HOLD_S`=%ss ✗ 用户：'持续够久…可能性最大' ✓）"
          % (sw_at, 8 + int(LT._DEV_HOLD_S / 0.1) - 1, LT._DEV_HOLD_S))


def test_white_static_vs_scrolling_bg():
    """⭐⭐ **白块阶段：目标静止、背景在滚 ⇒ 报告必须跟白块**（2026-09-30 用户实测报
    "第 6 帧就严重偏离真目标，而那时真目标还是白的" ✓）。

    根因：`T`（A 层算出的框群平移）是**背景的速度** ✗ —— 我上一轮把它错当成"目标也跟着
    群体平移"（把用户事实 2 的"**假**目标相对背景静止"推广错了 ✗✗）⇒ 速度兜底/收敛都往
    背景速度上走 ⇒ 目标一路被带走（实测 GIF 2~6 帧：白块只挪 **2px**、框群每帧走
    **20~25px** ✗✗）。

    造法：背景按 `_BG_SCROLL` 滚动（干扰项随背景 ✓）+ **完全静止**的白色目标（α=1 ⇒ 全程
    不透明 ✓）⇒ ① 报告必须停在白块上（≤20px ✓）；② 速度必须 ≈0（不许是背景速度 ✗）。
    """
    bg = _bg_layer()
    decoys = [(120.0, 140.0, 12), (500.0, 120.0, 12),
              (240.0, 380.0, 12), (600.0, 400.0, 12)]
    tx, ty = 300.0, 250.0
    tr = LieTracker()
    errs, vels = [], []
    for i in range(24):
        frame, _t = _frame(i, bg, decoys, lambda k: (tx, ty, 1.0))   # 全程**白** ✓
        o = tr.process(frame, ts=i * 0.14)
        if o["state"] == "track" and o["pos"] is not None:
            errs.append(((o["pos"][0] - tx) ** 2 + (o["pos"][1] - ty) ** 2) ** 0.5)
            vels.append(tr._speed_of_vel())
    late = errs[8:] if len(errs) > 8 else errs
    check(bool(late) and max(late) <= 20.0,
          "白块阶段被背景带走了（第 8 拍后最大偏 %.1f px ✗ 该 ≤20 ✓ —— 目标静止、"
          "背景在滚 ✓）" % (max(late) if late else -1))
    check(bool(vels) and sorted(vels)[len(vels) // 2] < 60.0,
          "白块阶段速度被当成了**背景速度**（中位 %.0f px/s ✗ 该 ≈0 ✓ —— `T` 是背景速度、"
          "不是目标速度 ✗）" % (sorted(vels)[len(vels) // 2] if vels else -1))


def test_report_rate_limit():
    """⭐ **报告位置（十字标）限速**（用户 2026-09-30 第③问 ✓ 原话："第14帧：真目标消失，
    但是十字标位置突变太远了，我们是否需要限制一下跟踪器的最大速度？" ✓）。

    一帧最多走 `_STEP_SPEED × dt` px —— ⚠ 光限 `_raw` 不够（实测仍有 84px/帧 ✗✗：超前
    补偿是 `vel × 2 帧` 的**前馈偏移**，会带着报告位置一起跳 ✗）⇒ 报告位置**自己**也要限 ✓。
    """
    import math

    from perception import lie_tracker as LT

    tr = LieTracker()
    tr._wh = (750, 500)
    tr._raw = (100.0 * LT._SCALE, 100.0 * LT._SCALE)
    tr.pos = (100.0, 100.0)
    tr._vel = (0.0, 0.0)
    dt = 0.1
    step = LT._STEP_SPEED * dt
    _p = tr._report_pos(dt)
    tr._raw = (700.0 * LT._SCALE, 480.0 * LT._SCALE)      # 目标"瞬移"到另一头 ✓
    tr._vel = (3000.0, 3000.0)                            # 顺带把速度也炸掉（老 bug 场景 ✓）
    p2 = tr._report_pos(dt)
    moved = math.dist(_p, p2)
    check(moved <= step + 1e-6,
          "报告位置没限速：一帧走了 %.1f px（上限 `_STEP_SPEED`×dt = %.1f px ✗）"
          % (moved, step))


def test_area_and_group():
    """⭐ **面积变化不污染速度**（用户 2026-09-30 第④问 ✓）与**群体速度互相修正**
    （第③问 ✓）。

    ① 面积跳变（合并/分离）那一拍的**质心位移**不许进速度（现场第 23 帧 -30.5px/帧 ✗）；
    ② 新出现的框（自己没速度）⇒ **借群体中位速度**（"假目标群体的速度几乎一致，可以互相
       修正" ✓）⇒ 立刻有方向（否则箭头 0 长 = 用户看到的"消失" ✗）。
    """
    a = 20.0
    # ① 一群匀速右移的框 + 一个"面积突然缩一半、质心跟着跳"的框 ⇒ 它的速度**不该**跟着跳
    tr = LieTracker()
    for i in range(10):
        x0 = 100.0 + 50.0 * 0.1 * i
        dets = [(x0, 100.0, a, a), (x0 + 200.0, 100.0, a, a), (x0 + 400.0, 100.0, a, a)]
        if i == 9:                       # 第 10 拍：把第二个框缩到 0.6×、质心顺带挪 30px ✗
            dets[1] = (x0 + 200.0 + 30.0, 100.0, a * 0.78, a * 0.78)
        tr._update_tracks(dets, 0.1, i * 0.1)
    jump = [t for t in tr._tracks if abs(t["p"][0] - (100.0 + 200.0 + 50.0 * 0.9)) < 60.0]
    v_ok = bool(jump) and abs(jump[0]["v"][0] - 50.0) < 30.0
    check(v_ok,
          "① 面积跳变那一拍把质心位移当成了速度（第 10 拍速度 %s ✗ 该仍在 ~50 ✓）"
          % ((None if not jump
              else (round(jump[0]["v"][0]), round(jump[0]["v"][1]))),))

    # ② 群体速度互相修正：一个刚出现的框（0 段历史）⇒ 借群体中位（50,0）
    tr2 = LieTracker()
    for i in range(8):
        x0 = 100.0 + 50.0 * 0.1 * i
        tr2._update_tracks([(x0, 100.0, a, a), (x0 + 200.0, 100.0, a, a),
                            (x0 + 400.0, 100.0, a, a)], 0.1, i * 0.1)
    x0 = 100.0 + 50.0 * 0.8
    tr2._update_tracks([(x0, 100.0, a, a), (x0 + 200.0, 100.0, a, a),
                        (x0 + 400.0, 100.0, a, a), (700.0, 400.0, a, a)], 0.1, 0.8)
    newb = [t for t in tr2._tracks if t["p"][1] > 350.0]
    check(bool(newb) and abs(newb[0]["v"][0] - 50.0) < 5.0 and abs(newb[0]["v"][1]) < 5.0,
          "② 新出现的框没借到群体速度（%s ✗ 该 ≈(50,0) ✓ 用户③ ✓）"
          % ((None if not newb else newb[0]["v"]),))

    # ③ **S0~S3 状态判据**（重构 2026-09-30 ✓ 用户事实 4~7 ✓）：面积事件只用来定状态 ✓，
    #    位置与速度改由残差通道决定 ✓（旧的"面积方向改速度"整族已删 ✗）。
    #    ⚠⚠ 2026-10-01 口径升级 ✓：融合 = "**包围已上板假目标**的检出框面积 / **那块假目标的
    #      登记面积** ≥ 阈值"（用户原话 ✓）⇒ 判"并集块"必须**先有登记条目**（不再是"当拍检出
    #      中位" ✗ 那个逐帧抖 ⇒ 同一格框面积没变也会判融合 ✗✗）。
    from perception.lie_registry import Entry, ShapeRegistry
    _brick = (250.0, 250.0, 150.0, 150.0)              # 已上板假目标（面积 22500 ✓）
    tr3 = LieTracker()
    tr3.pos = (300.0, 250.0)
    _e3 = Entry(_brick, ts=0.0)
    _e3.on_board = True
    _rg3 = ShapeRegistry()
    _rg3.entries = [_e3]
    tr3._reg = _rg3
    # 目标自己那格框（150×150，只**擦着**那块砖 ⇒ 覆盖率 0.33 < 0.75、中心也不在框里）
    #   ⇒ **不是融合**；但它**与绿圈相交**（用户 2026-10-01 ✓ Q4.1 定稿："如果**与绿圈外接圆
    #   相交，那么它就是红框**" ✓）且**不是砖**（表里只有那一块上板砖、这框在 (350,250) ✗）⇒
    #   **状态 `ok` = 目标自己那格红框** ✓（照常夹取/照常学半径 ✓ = Q2 要的"弱约束" ✓）。
    st, bx = tr3._target_box_state([(350.0, 250.0, 150.0, 150.0)], {})
    check(st == "ok" and bx is not None,
          "③ 与绿圈相交且非砖 ⇒ `ok`（目标自己那格红框 ✓ 实际 %s ✓ Q4.1/Q2 口径 ✓）" % st)
    # 同一格框，但**圆离得很远**（不相交）⇒ 两类红框来源都不占 ⇒ `gone` ✓ 预测态 ✓
    tr3.pos = (300.0, 800.0)
    st1b, _ = tr3._target_box_state([(350.0, 250.0, 150.0, 150.0)], {})
    check(st1b == "gone",
          "③ 不融合、也与绿圈不相交 ⇒ `gone`（实际 %s ✓ 预测态 ✓）" % st1b)
    tr3.pos = (300.0, 250.0)
    # 同一块砖**自己那一格**、绿圈只**搭到它的边上**（相交 ✓ 但**没整圈进去** ✗）⇒
    #   规则② 不成立（要"整圈在内" ✓）、规则① 也不成立（面积比 1.00 < 阈值 ✗）⇒ 不是融合 ✓；
    #   而它**是砖**（身份 = "又检出了上板砖" ✓）⇒ 就算相交也**不许当红框** ✓
    #   （用户 Q2 前提："该检出框不被认定为钉好的假目标" ✓）⇒ `gone` ✓。
    tr3._rad = 30.0                              # 绿圈半径 30 ✓
    tr3.pos = (250.0, 340.0)                     # 圆心在砖下缘外 15px ⇒ 相交 ✓ 但整圈进不去 ✗
    st1c, _ = tr3._target_box_state([(250.0, 250.0, 150.0, 150.0)], {})
    check(st1c == "gone",
          "③ 与绿圈相交**但它是砖自己那一格**（且圈没整圈进去 ⇒ 非融合 ✓）⇒ `gone`（实际 %s ✓ "
          "Q2 前提 ✓）" % st1c)
    tr3._rad = 0.0
    # 并集块 = 把那块砖整个吞进去、面积 160000 / 22500 = 7.1× ≥ 阈值 ⇒ `merged` ✓（规则① ✓）
    st2, _ = tr3._target_box_state([(250.0, 250.0, 400.0, 400.0)], {})
    check(st2 == "merged",
          "③ 包围已上板假框的并集块 ⇒ 状态 `merged`（实际 %s ✓ 规则① ✓）" % st2)
    # "真目标与假目标完全重叠"那一档：框 ≈ 那块砖的大小、**绿圈也在框里**
    #   ⇒ ⚠⚠ **规则② 已退役**（用户 2026-10-01 ✓ 最新裁决："该检出框是否判定为**属于砖条目**的？
    #   如果是就**不应该标**，而是应该遵循上一帧的预测" ✓）—— 这格框就是砖自己那一格 ✓
    #   ⇒ `_brick_owned` 拦下 ⇒ `gone`（预测态 ✓ 圆按白线/白箭头走 ✓）。
    tr3.pos = (250.0, 250.0)                     # 绿圈落进那块砖的那一格 ✓
    st2b, _ = tr3._target_box_state([(250.0, 250.0, 150.0, 150.0)], {})
    check(st2b == "gone",
          "③ 框≈砖（面积相等）+绿圈在框里 ⇒ **`gone`（不标红 ✓ 属于砖条目 ✓ 规则②已退役 ✓）**"
          "（实际 %s ✓）" % st2b)
    tr3.pos = (300.0, 250.0)
    st3, _ = tr3._target_box_state([(900.0, 900.0, 150.0, 150.0)], {})
    check(st3 == "gone",
          "③ 门内没框 ⇒ 状态 `gone`（实际 %s ✓ 事实 4 ✓ 漏检）" % st3)
    # ④ **群体速度兜底**：速度还没建立时按 `T/dt` 起步 ✓（用户事实 2：目标与假群一起平移）
    tr4 = LieTracker()
    tr4._raw = (150.0, 125.0)                 # 半分辨率 ✓
    tr4._set_vel(0.0, 0.0)
    check(tr4._speed_of_vel() < 5.0, "④ 初始速度为 0（%s ✓ 该被群体速度接手）"
          % (round(tr4._speed_of_vel(), 1),))


def test_path_curvature():
    """⭐⭐ **轨迹预测要带曲率**（用户 2026-10-01 ✓ 原话："我们不能用纯切向来预测轨迹，轨迹实际
    是有曲率的" ✓）：二次拟合的**二次项 `a` 也要用** ⇒ 沿弧外推（`b·dt + a·dt²`），不是只沿切向
    直线（`b·dt` ✗ 转弯时冲出去）。

    造两条群体坐标系的路径（`cumT=0` ✓ 无群体 ✓）：
    ① `x=10t, y=4t²`（向上弯 ✓）：末端切向 `y'=0`（纯切向预测 y≈0 ✗），但曲率 `y''=8` ⇒
       `a·dt²=4·0.01=+0.04` ⇒ 预测 y 应 **>0**（跟着弯上去 ✓）；
    ② `x=10t, y=0`（纯直线 ✓）：预测 y 应 ≈0（不该被"曲率"乱掰 ✗）。
    """
    def _mk(ys):
        tr = LieTracker(path_ms=5000)
        tr._raw = (0.0, 0.0)
        tr._path = []
        for k in range(6):
            t = -0.5 + 0.1 * k
            tr._path.append((round(t, 2), 10.0 * t, ys(t), 0.0, 0.0))
        tr._group_vel = lambda dt: (0.0, 0.0)      # 无群体运动（简化 ✓）
        return tr._path_pred(0.1)

    _p1 = _mk(lambda t: 4.0 * t * t)               # 向上弯 ✓
    check(_p1 is not None and _p1[1] > 0.01,
          "⑧ 向上弯的轨迹：预测点 y=%.3f 应 >0（纯切向会是 ≈0 ✗ —— 曲率项把它掰上弧 ✓）"
          % (_p1[1] if _p1 else float("nan"),))
    _p2 = _mk(lambda t: 0.0)                       # 纯直线 ✓
    check(_p2 is not None and abs(_p2[1]) < 0.01,
          "⑧ 纯直线轨迹：预测点 y=%.3f 应 ≈0（曲率≈0 ⇒ 不被乱掰 ✗ ✓）"
          % (_p2[1] if _p2 else float("nan"),))


def test_sep_takeover():
    """⭐⭐ **分离后速度的"平滑接管"**（用户 2026-10-01 ✓ 原话："分离那一拍速度从冻结值
    **硬切**到实测值，改成**平滑接管**" ✓）。

    钉的是**算术本身**（不依赖素材 ✓ 素材级那条 A/B 在 `selftest_lie_demo` ✓）：
    ① 接管第 1 拍只吃 `1/_SEP_TAKE_N`（冻结值仍占大头 ⇒ 白箭头不会断崖 ✓）；
    ② **群体分量一个字不动** ✔（`_vel` = 自身 + 群体 ⇒ 只补差值 ✓ 群体永远取当拍实测 ✓）；
    ③ 拍数走满 ⇒ **完全交回实测**、且之后**不再干预** ✓；
    ④ 不在接管期 ⇒ 一个字不动（老行为一致 ✓）。

    实测背景（`9月30日(1).mp4` 显示帧 22 ✓）：冻结值 `(65.3,137.0)`、实测 `(2.3,36.4)`
    ⇒ 硬切 ⇒ 白箭头 **33px → 8px** ✗；接管后第一拍 `(51.0,104.0)` ⇒ 箭头 ≈25px ✓。
    """
    from perception import lie_tracker as LT

    tr = LieTracker()
    _N = max(1, int(LT._SEP_TAKE_N))
    tr._sep_vel0 = (60.0, 120.0)          # 冻结值（融合期一路沿用 ✓）
    tr._sep_n = 0
    tr._vel_rel = (0.0, 0.0)              # 这一拍实测出来的自身速度（故意给个很不一样的）
    tr._vel = (100.0 + 0.0, 200.0 + 0.0)  # = 自身(0,0) + 群体(100,200) ✓
    tr._sep_takeover()
    _w = 1.0 / float(_N)
    _exp = (60.0 * (1.0 - _w) + 0.0 * _w, 120.0 * (1.0 - _w) + 0.0 * _w)
    check(abs(tr._vel_rel[0] - _exp[0]) < 1e-6 and abs(tr._vel_rel[1] - _exp[1]) < 1e-6,
          "⑥ 接管第 1 拍只吃 1/%d（实际 %s ｜ 期望 (%.1f,%.1f) ✓ —— 冻结值仍占大头 ⇒ "
          "白箭头不断崖 ✓）" % (_N, "(%.1f,%.1f)" % tr._vel_rel, _exp[0], _exp[1]))
    check(abs(tr._vel[0] - (100.0 + _exp[0])) < 1e-6
          and abs(tr._vel[1] - (200.0 + _exp[1])) < 1e-6,
          "⑥ **群体分量一个字没动**（`_vel`=%s ｜ 期望 (%.1f,%.1f) = 群体(100,200)+新自身 ✓）"
          % ("(%.1f,%.1f)" % tr._vel, 100.0 + _exp[0], 200.0 + _exp[1]))
    # 走满 `_SEP_TAKE_N` 拍 ⇒ 完全交回实测
    for _k in range(_N - 1):
        tr._vel_rel = (0.0, 0.0)          # 后续几拍实测还是 (0,0) ⇒ 接管应把它拉到底 ✓
        tr._sep_takeover()
    check(abs(tr._vel_rel[0]) < 1e-6 and abs(tr._vel_rel[1]) < 1e-6,
          "⑥ 走满 %d 拍 ⇒ **完全交回实测**（实际 %s ✓ 之后不再被冻结值拖着 ✗）"
          % (_N, "(%.1f,%.1f)" % tr._vel_rel))
    check(tr._sep_n is None, "⑥ 接管完成后 `_sep_n` 清空 ⇒ **不再干预** ✓（实际 %s）"
          % (tr._sep_n,))
    _before = (tuple(tr._vel_rel), tuple(tr._vel))
    tr._sep_takeover()
    check(tuple(tr._vel_rel) == _before[0] and tuple(tr._vel) == _before[1],
          "⑥ 不在接管期 ⇒ 一个字都不动（与改动前行为一致 ✓）")


def test_box_is_registered_fake():
    """⭐⭐ **红框 ≈ 在册假框 ⇒ 不是目标**（用户 2026-10-01 ✓ 原话："红框没有任何理由地标到 #38
    上了" ✓ —— 分离后红框塌缩成假目标自己那一格 ⇒ 不该当红框 ✗）。
    `_box_is_registered_fake` 两条判据：① 框中心贴某在册假框中心（门内 ✓）；② 框面积**不比那假框
    大**（真并集一定比单块假框大 ✗）。用实测 `9月30日(1).mp4` 第 27/28 帧那组数 ✓。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _tr(fake):
        t = LieTracker()
        e = Entry(fake, ts=0.0)
        e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        return t

    fake = (288.8, 327.8, 146.4, 136.9)              # 实测第 28 帧吞下的在册假框 ✓
    # ① 框 ≈ 假框（同心、面积更小 ✓）⇒ True
    t1 = _tr(fake)
    check(t1._box_is_registered_fake((288.8, 327.8, 137.3, 132.9)),
          "⑦ 红框 ≈ 在册假框（同心、面积更小）⇒ 判成「假目标那一格」✓（= 第 28 帧那条 ✓）")
    # ② 框明显比假框**大**（= 真并集 ✓）⇒ False
    t2 = _tr(fake)
    check(not t2._box_is_registered_fake((244.9, 300.3, 176.9, 186.7)),
          "⑦ 框比假框大（= 真并集 ✓ 第 27 帧）⇒ **不**判成假目标 ✓")
    # ③ 框中心离假框太远 ⇒ False
    t3 = _tr(fake)
    check(not t3._box_is_registered_fake((100.0, 100.0, 130.0, 130.0)),
          "⑦ 框中心不在假框门内 ⇒ **不**判成假目标 ✓")
    # ④ **接线（真分离那一档 ✓）**：`_target_box_state` 选中"≈在册假框"的那格、**且绿圈已经不在
    #    那格里**（目标走了 ✓）⇒ 返回 `gone`（红框消失、画淡粉接力框、圆按预测走 ✓）。
    #    ⚠ 这一档现在由"融合两条口径都不占 + 框离圆太远"共同保证 ✓ —— 无论走哪一条，**结果都是
    #      不给红框** ✓（这正是真分离要的 ✓）。
    t4 = _tr(fake)
    t4.pos = (288.8, 420.0)          # 报告位置在框外（框 y ≤ 394 ✓）但仍在吸附门内（92px ✓）
    t4._typ_area = 140.0 * 130.0
    t4._rad = 56.0
    _st4, _bx4 = t4._target_box_state([(288.8, 327.8, 137.3, 132.9)], {})
    check(_st4 == "gone" and _bx4 is None,
          "⑦ 红框 ≈ 在册假框、**绿圈已不在那格里** ⇒ `_target_box_state` 返回 `gone`"
          "（实际 %s/%s ✓ = 真分离 ✓ 红框消失、画淡粉接力框、圆按预测走 ✓）" % (_st4, _bx4))
    # ④' **反向（用户 2026-10-01 ✓ 最新裁决）**：同样"红框 ≈ 在册假框"、但**绿圈还在框里** ⇒
    #     ⚠⚠ **规则② 已退役**（"属于砖条目 ⇒ 不应该标" ✓）—— 这格框就是砖自己那一格 ✓
    #     ⇒ `gone` ✓（= 红框消失 ⇒ **算分离** ✓ 圆按白线/白箭头预测走 ✓）。
    t4b = _tr(fake)
    t4b.pos = (288.8, 327.8)         # 报告位置落进那格"假框" ✓
    t4b._typ_area = 140.0 * 130.0
    t4b._rad = 56.0
    _st4b, _bx4b = t4b._target_box_state([(288.8, 327.8, 137.3, 132.9)], {})
    check(_st4b == "gone" and _bx4b is None,
          "⑦ 红框 ≈ 在册假框、但**绿圈还在框里** ⇒ `gone`（实际 %s ✓ 属于砖条目不标红 ✓ "
          "规则②已退役 ✓）" % _st4b)


def test_box_identity():
    """⭐⭐ **每个检出框都需要它是谁**（用户 2026-10-01 ✓ 原话："每个检出框我们都需要它是谁：
    ① 与上板登记的假目标**高度重合** → 说明检出框**又检出了假目标**，**不能锁红**；
    ② **大**检出框**直接包住**了上板登记的假目标 → **融合信号**" ✓）。

    数值取用户点名的实测帧（`10月1日.mp4`）：帧 31 覆盖率 1.00 / 倍率 **1.55**（用户："此帧红框
    不该标" ✓）；帧 27 倍率 **1.87**、帧 39/40 倍率 **2.05**（并集 = 融合 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _tr(fake, white=None):
        t = LieTracker()
        e = Entry(fake, ts=0.0)
        e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        t._white_cur = white
        return t

    fake = (222.0, 255.0, 144.0, 134.0)          # 上板假目标（实测帧 29~31 那块 ✓ 144×134）
    # ① 与它**高度重合**（同心、1.55 倍 = **用户点名的第 31 帧**）⇒ `"fake"`（不能锁红 ✗）
    t1 = _tr(fake)
    check(t1._box_identity((222.0, 255.0, 179.0, 167.0))[0] == "fake",
          "⑧ 与上板假目标高度重合（同心、倍率 1.55 = 用户第 31 帧那组数）⇒ 判「假目标」"
          "（= 不能锁红 ✓）")
    # ② **大框直接包住**它（2.05 倍 = 实测帧 39/40 的并集）⇒ `"merged"`（融合信号 ✓）
    t2 = _tr(fake)
    check(t2._box_identity((230.0, 262.0, 225.0, 200.0))[0] == "merged",
          "⑧ 大框直接包住上板假目标（倍率 2.05 = 实测帧 39/40）⇒ 判「融合信号」✓")
    # ③ 边界内（1.65 ≤ 1.7）仍算假目标 ✓（用户口径"高度重合"要能容忍检出框的尺寸抖动 ✓）
    t3 = _tr(fake)
    check(t3._box_identity((222.0, 255.0, 186.0, 172.0))[0] == "fake",
          "⑧ 倍率 1.65（≤ 经验线 1.7）⇒ 仍判「假目标」✓（检出框尺寸抖动不许漏判 ✗）")
    # ④ **白块在框里 ⇒ 免判**（开局帧 5~10：目标白块就落在那格里 ✓ 判假目标会把开局红框抹掉 ✗✗）
    t4 = _tr(fake, white=(222.0, 255.0, 11654.0))
    check(t4._box_identity((222.0, 255.0, 179.0, 167.0))[0] == "",
          "⑧ 白块观测落在框里（= 目标就在这格 ✓ 实测开局帧 5~10）⇒ **免判**（保持红框 ✓）")
    # ⑤ 尺寸对得上但**中心不贴**（差半格）= 另一块砖 ⇒ 不判 ✓
    t5 = _tr(fake)
    check(t5._box_identity((280.0, 255.0, 150.0, 140.0))[0] == "",
          "⑧ 中心差 58px（> 配对门 25）⇒ 不判（那是**另一块**砖 ✓ 别张冠李戴 ✗）")
    # ⑥ **接线（用户 2026-10-01 ✓ 帧 17 教训 ✓）**：与上板假框"高度重合"（同心）但面积 **1.55 倍
    #   ≥ 阈值** ⇒ **真并集** ✓（框比砖大出 1.55 倍 = 目标 + 砖 ✓ 不许被"属于砖"拦下 ✗）
    #   ⇒ `merged` ✓（帧 17 同款：208.7×165.9 包住砖 160.6×134.4、1.60 倍 ⇒ 必须标红 ✓）。
    t6 = _tr(fake)
    t6.pos = (222.0, 255.0)                      # 绿圈在框里 ✓
    t6._typ_area = 140.0 * 130.0
    t6._rad = 56.0
    _st6, _bx6 = t6._target_box_state([(222.0, 255.0, 179.0, 167.0)], {})
    check(_st6 == "merged" and _bx6 is not None,
          "⑧ 与上板假框同心、倍率 1.55 ≥ 阈值 ⇒ `merged`（实际 %s ✓ 真并集照常融合 ✓ 帧 17 口径 ✓）"
          % _st6)
    # ⑥' **反向（帧 31 那一档）**：同心、倍率 **1.17 < 阈值**、且**绿圈不在框里** ⇒ 不给红框 ✓
    #    （口径依据：融合要有"并集"证据 —— 要么**框包住绿圈**（规则② ✓），要么**面积比 ≥ 阈值**
    #     （规则① ✓）；两条都不占 ⇒ 既不融合、又够不上"目标自己的框"（离圆太远 ✗）⇒ gone ✓）。
    #    ⚠ 同口径下"同心 + 绿圈**在**框里"是**融合红框** ✓（见 ⑥ ✓）—— 这正是用户 2026-10-01
    #      规则② 对第 31 帧那条口号的**修正**：那条只在"绿圈不在框里"时成立 ✓。
    _fake2 = (222.0, 255.0, 160.0, 160.0)        # 比 ①~⑤ 那块大一点 ⇒ 框/砖 = 29893/25600 = 1.17
    t6b = _tr(_fake2)
    t6b.pos = (222.0, 375.0)                     # 绿圈在框**外**（框 y ≤ 335 ✓）但仍在吸附门内 ✓
    t6b._typ_area = 140.0 * 130.0
    t6b._rad = 56.0
    _st6b, _bx6b = t6b._target_box_state([(222.0, 255.0, 179.0, 167.0)], {})
    check(_st6b == "gone" and _bx6b is None,
          "⑧ 同心、倍率 1.17（< 阈值）且**绿圈在框外** ⇒ `gone`（实际 %s/%s ✓ —— 两条融合口径"
          "都不占 ⇒ 不给红框 ✓）" % (_st6b, _bx6b))
    # ⑦ 反过来：**大框包住**（融合）⇒ 照旧给红框 ✓（别把规则 2 一起掐死 ✗）
    t7 = _tr(fake)
    t7.pos = (230.0, 262.0)
    t7._typ_area = 140.0 * 130.0
    t7._rad = 56.0
    _st7, _bx7 = t7._target_box_state([(230.0, 262.0, 225.0, 200.0)], {})
    check(_st7 == "merged" and _bx7 is not None,
          "⑧ 身份「融合」⇒ 照旧给红框且判 `merged`（实际 %s ✓ 规则 ② 生效 ✓）" % _st7)


def test_tbox_prefers_merge_over_fake():
    """⭐⭐ **多个框同时包含报告位置时，选框要优先"并集框"而非"假砖框"**（用户 2026-10-01 ✓
    原话："大检出框是套在砖(122,205)身上的框啊" ✓）。

    实测根因（`9月30日(1).mp4` 帧 38）：报告位置 `(147.7,253.6)` 同时落进两个框——
    ① 假砖小框 `(122,274 136×136)`（在册、`_box_is_registered_fake=True` ✓）；② 并集大框
    `(178,193 194×187)`（含 pos、不是假框 ✓）。老实现取"列表里第一个" = 假砖小框 ⇒ 判 `gone`
    ⇒ 红框凭空消失 ✗✗。修后应取并集大框（`merged`）✓。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    tr = LieTracker()
    tr._typ_area = 19000.0
    tr._rad = 56.0
    tr.pos = (147.7, 253.6)
    e = Entry((122.0, 274.0, 144.0, 140.0), ts=0.0)
    e.on_board = True
    rg = ShapeRegistry()
    rg.entries = [e]
    tr._reg = rg
    # 小框 = 假砖（面积 18496 ≤ 144×140=20160 ⇒ registered fake ✓）、大框 = 并集 ✓；pos 都在两者里 ✓
    _st, _tb = tr._target_box_state(
        [[122.0, 274.0, 136.0, 136.0], [178.0, 193.0, 194.0, 187.0]], {})
    check(_st == "merged" and _tb is not None
          and abs(_tb[0] - 178.0) < 1e-6 and abs(_tb[1] - 193.0) < 1e-6,
          "⑦ 多个框含 pos 时选**并集大框**（实际 state=%s tbox=%s ｜ 期望 merged + (178,193) ✓）"
          % (_st, None if _tb is None else "(%.0f,%.0f)" % (_tb[0], _tb[1])))
    # 反向：只有假砖小框（没有并集）、**且绿圈不在它里面**（分离后才有的样子 ✓）⇒ 仍判 gone ✓
    #   ⚠ 绿圈若**还在**那格里 ⇒ 按用户 2026-10-01 规则② 那是"目标与砖叠在一起" ⇒ 融合红框 ✓
    #     （不是分离 ✓）—— 所以"分离信号"这一档必须配"绿圈已经离开那格" ✓ 见 ④' 的反向用例 ✓。
    tr2 = LieTracker()
    tr2._typ_area = 19000.0
    tr2._rad = 56.0
    tr2.pos = (122.0, 380.0)         # 在假砖小框**外面**（框 y ≤ 342 ✓）但仍在吸附门内 ✓
    tr2._reg = rg
    _st2, _tb2 = tr2._target_box_state([[122.0, 274.0, 136.0, 136.0]], {})
    check(_st2 == "gone" and _tb2 is None,
          "⑦ 只有假砖小框（无并集）且绿圈已不在里面 ⇒ 仍判 gone（实际 %s/%s ✓ 分离信号不变 ✗）"
          % (_st2, _tb2))


def test_anti_group_vel():
    """⭐⭐ **试验型：自身速度旋转到「群体速度的反向」**（用户 2026-10-01 ✓ 原话："最终的白色
    轨迹预测箭头旋转至<假目标群体速度向量>方向的反向" + 更正："我并没有说只改箭头方向，一定要
    **白箭头 = 圆圈真的会走多少**" ✓）。

    钉的是 `_anti_group_vel` 的**算术**（改的是后端 `_vel_rel`/`_vel` ✓ 不是显示层 ✗）：
    ① 自身速度旋转到 `-群体` 方向、长度 |vel_rel| 不变 ✓；
    ② 绝对速度 = 新自身 + 群体（圆圈真的按它走 ✓ 白箭头 = 它 ✓）；
    ③ 群体≈0 ⇒ 不旋转 ✓；④ 自身≈0 ⇒ 不旋转 ✓。
    """
    tr = LieTracker()
    # 绝对向右 (200,0)、自身向下 (0,100) ⇒ 群体 = (200,-100)（右上）⇒ 反向 = (-200,100)（左下）
    tr._vel = (200.0, 0.0)
    tr._vel_rel = (0.0, 100.0)
    tr._anti_group_vel()
    check(abs(tr._vel_rel[0] - (-89.44)) < 0.1 and abs(tr._vel_rel[1] - 44.72) < 0.1,
          "⑧ 自身速度旋转到群体反向（实际 (%.1f,%.1f) ｜ 期望 (-89.4,44.7) ✓ —— 长度 |vel_rel|"
          "=100 不变、方向 = -群体 ✓）" % tr._vel_rel)
    check(abs(tr._vel[0] - 110.56) < 0.1 and abs(tr._vel[1] - (-55.28)) < 0.1,
          "⑧ 绝对速度 = 新自身 + 群体（实际 (%.1f,%.1f) ｜ 期望 (110.6,-55.3) ✓ —— **圆圈真的按"
          "它走** ✓ 白箭头 = 它 ✓）" % tr._vel)
    tr2 = LieTracker()
    tr2._vel = (0.0, 0.0)
    tr2._vel_rel = (0.0, 100.0)
    _b2 = (tuple(tr2._vel), tuple(tr2._vel_rel))
    tr2._anti_group_vel()
    check((tuple(tr2._vel), tuple(tr2._vel_rel)) == _b2,
          "⑧ 群体≈0 ⇒ 不旋转（原样 ✓ —— 没有『上游』可指 ✗）")
    tr3 = LieTracker()
    tr3._vel = (200.0, 0.0)
    tr3._vel_rel = (0.0, 0.0)
    _b3 = (tuple(tr3._vel), tuple(tr3._vel_rel))
    tr3._anti_group_vel()
    check((tuple(tr3._vel), tuple(tr3._vel_rel)) == _b3,
          "⑧ 自身≈0 ⇒ 不旋转（原样 ✓ —— 箭头本来就是个点 ✗）")


def test_merge_far_corner():
    """⭐⭐⭐ **新认知：并集块「几乎包围」在册假目标 ⇒ 圆心修正到「那个假目标的对角」**
    （用户 2026-10-01 ✓ 照抄原话："对于红框几乎包围假目标登记框的情况，我们要知道大红框的
    形式『真假目标融合拉大的』，因此参考第 27 帧的情况绿圈应该被修正到红框的左上角" ✓）。

    输入直接用**实测第 27 帧那组数**（`9月30日(1).mp4` 下标 26 ✓）：红框 `(244.9,300.3)`
    `176.9×186.7`（面积比 1.58 ⇒ 并集 ✓）、在册假框 `(268.8,319.6)` 落在红框的**右下**
    ⇒ 期望 = 内接安全区的**左上**角 `(217.8,268.4)` ✓。

    五条：
    ① 假框在右下 ⇒ 修正到**左上**内角 ✓（= 用户第 27 帧那条 ✓）；
    ② 假框换到左上 ⇒ 修正到**右下**内角 ✓（判据是「**对角**」✗ 不是「恒取左上」✓）；
    ③ 假框在**后方**（相对白箭头 > 90° ✓ 方向限定 ✓）⇒ 不修正 ✓；
    ④ 红框**不是并集**（面积 ≈ 一个假目标的面积 = 目标自己那格框 ✓ 目标在框中心 ✗）⇒ 不修正 ✓；
    ⑤ 假框中心在红框**外**（压根没被吞 ✓）⇒ 不修正 ✓。
    ⚠ 注：融合对象已放宽到"所有青砖（候选+上板）" ✓ 所以"没上板"不再是不修正的理由 ✗ ——
      不修正只剩"在后方 / 非并集 / 在框外"这三条 ✓。
    ⚠ 还有一个隐含性质：回值取的是 `_tbox_inner`（**内接安全区**）的角 ✓ ⇒ 圆**整圈仍在红框内** ✓
      （认知 #1「绿圈与红框内接」不许破 ✗）。
    """
    import math as _m

    from perception.lie_registry import Entry, ShapeRegistry

    def _tr(fake, on_board=True, typ=140.0 * 130.0):
        t = LieTracker()
        t._typ_area = typ
        t._rad = 56.0
        t._tbox_rad = 61.4            # ⚠ **画圈那把尺**（= 内接半径 ✓ 与显示同一把 ✓ 实测第 27 帧值 ✓）
        e = Entry(fake, ts=0.0)
        e.on_board = bool(on_board)
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        return t

    def _pt(p):
        return "None" if p is None else "(%.1f,%.1f)" % (p[0], p[1])

    box = (244.9, 300.3, 176.9, 186.7)                # 实测第 27 帧的红框（并集 ✓）
    t1 = _tr((268.8, 319.6, 140.1, 131.2))            # 假框在**右下** ⇒ 目标该在左上 ✓
    x0, y0, x1, y1 = t1._tbox_inner(box)
    f1 = t1._merge_far_corner(box)
    check(f1 is not None and abs(f1[0] - x0) < 1e-6 and abs(f1[1] - y0) < 1e-6,
          "⑥ 并集吞下在册假框（右下）⇒ 修正到**左上内角**（实际 %s ｜ 期望 (%.1f,%.1f) ✓ "
          "= 用户第 27 帧那条 ✓）" % (_pt(f1), x0, y0))
    check(f1 is not None and min(f1[0] - (box[0] - box[2] / 2.0),
                                 (box[0] + box[2] / 2.0) - f1[0],
                                 f1[1] - (box[1] - box[3] / 2.0),
                                 (box[1] + box[3] / 2.0) - f1[1]) >= 61.4 - 1e-6,
          "⑥ 修正点仍在**内接安全区**里（到四边最近 %.1f ≥ 半径 61.4 ✓ ⇒ 圆整圈还在红框内 ✓ "
          "认知 #1 没破 ✓）"
          % (min(f1[0] - (box[0] - box[2] / 2.0), (box[0] + box[2] / 2.0) - f1[0],
                 f1[1] - (box[1] - box[3] / 2.0), (box[1] + box[3] / 2.0) - f1[1]),))

    t2 = _tr((221.0, 281.0, 140.1, 131.2))            # 假框在**左上** ⇒ 目标该在右下 ✓
    f2 = t2._merge_far_corner(box)
    check(f2 is not None and abs(f2[0] - x1) < 1e-6 and abs(f2[1] - y1) < 1e-6,
          "⑥ 假框在左上 ⇒ 修正到**右下内角**（实际 %s ｜ 期望 (%.1f,%.1f) ✓ —— 判据是「对角」"
          "✗ 不是「恒取左上」✓）" % (_pt(f2), x1, y1))

    # ③（新）方向限定：假框在**后方**（相对白箭头夹角 > 90°）⇒ 不修正 ✓（用户 2026-10-01 ✓）
    t3 = _tr((268.8, 319.6, 140.1, 131.2))
    t3.pos = (244.9, 300.3)              # 目标在红框中心 ✓
    t3._vel_rel = (-1.0, -1.0)           # 白箭头指向左上 ⇒ 假框(268.8,319.6)在右下方 = 后方 ✗
    check(t3._merge_far_corner(box) is None,
          "⑥ 假框在**后方**（相对白箭头 > 90°）⇒ 不修正 ✓（方向限定 ✓）")
    # ③'（新口径）判据①要**已上板**（"包围**已上板**的假目标…/ **已上板的**假目标记录的面积" ✓
    #   用户 2026-10-01 ✓）⇒ 表里**只有没上板的候选**时判不出"并集" ⇒ 不修正 ✓
    t3b = _tr((268.8, 319.6, 140.1, 131.2), on_board=False)
    t3b.pos = (244.9, 300.3)
    t3b._vel_rel = (1.0, 1.0)            # 白箭头指向右下 ⇒ 假框在正前方 ✓
    check(t3b._merge_far_corner(box) is None,
          "⑥ 表里**只有没上板的候选** ⇒ 判不出并集（判据①要「已上板」✓ 新口径 ✓）⇒ 不修正 ✓")
    # ③'' 但判据②的**融合对象**仍放宽到"所有青砖"（候选也参与 ✓ 用户 2026-10-01 ✓）：
    #   已上板那块负责判出"并集" ✓，选角要**同时**避开 {已上板 + 候选} ✓
    t3c = _tr((268.8, 319.6, 140.1, 131.2))
    t3c._reg.entries.append(Entry((200.0, 380.0, 140.1, 131.2), ts=0.0))   # 候选（没上板 ✓）
    t3c.pos = (244.9, 300.3)
    t3c._vel_rel = (1.0, 1.0)
    _f3c = t3c._merge_far_corner(box)
    _pts3 = ((268.8, 319.6), (200.0, 380.0))
    _far3 = max(((x0, y0), (x1, y0), (x0, y1), (x1, y1)),
                key=lambda c: min(_m.hypot(c[0] - _p[0], c[1] - _p[1]) for _p in _pts3))
    check(_f3c is not None and abs(_f3c[0] - _far3[0]) < 1e-6
          and abs(_f3c[1] - _far3[1]) < 1e-6,
          "⑥ 「融合对象」放宽到**所有青砖**（候选也参与 ✓）：取的是离 {已上板 + 候选} 最远的角"
          "（实际 %s ｜ 期望 (%.1f,%.1f) ✓）" % (_pt(_f3c), _far3[0], _far3[1]))
    # ④ 非并集：那块**已上板**假框自己就有 160×160 = 25600（33027 / 25600 = 1.29 < 阈值 1.4 ✗）
    #   ⇒ 判不出并集（= 红框只相当于"一块假目标"，目标自己那格框 ✓）⇒ 不修正 ✓
    check(_tr((268.8, 319.6, 160.0, 160.0))._merge_far_corner(box) is None,
          "⑥ 红框**不是并集**（面积只相当于那块登记假框 ⇒ 目标自己那格框 ✓）⇒ 不修正 ✓")
    check(_tr((600.0, 600.0, 140.1, 131.2))._merge_far_corner(box) is None,
          "⑥ 假框中心在红框**外**（没被吞 ✓）⇒ 不修正 ✓")
    check(abs(_m.hypot(f1[0] - 268.8, f1[1] - 319.6)
              - max(_m.hypot(c[0] - 268.8, c[1] - 319.6)
                    for c in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)))) < 1e-6,
          "⑥ 取的是四个内角里**离假框最远**的那个 ✓（不是碰巧的某一个 ✓）")


# ---- 跑批骨架（同其它 selftest 的口径 ✓）----
_FAILED = []


def check(cond, msg):
    if cond:
        print("  [OK] %s" % msg)
    else:
        print("  [NG] %s" % msg)
        _FAILED.append(msg)


def main():
    print("测谎追踪器自检（合成序列）：")
    test_lie_tracker_synthetic()
    test_track_memory()
    test_noise_not_win()
    test_size_suspects()
    test_white_static_vs_scrolling_bg()
    test_report_rate_limit()
    test_area_and_group()
    test_merge_far_corner()
    test_box_is_registered_fake()
    test_box_identity()
    test_tbox_prefers_merge_over_fake()
    test_sep_takeover()
    test_anti_group_vel()
    test_path_curvature()
    if _FAILED:
        print("自检：%d 条失败" % len(_FAILED))
        return 1
    print("测谎追踪器自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

