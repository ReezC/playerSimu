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


def _LT(**kw):
    """⭐⭐⭐ **测试用构造器**：给一套**宽口径**，让"融合链"的老用例能用**简单几何夹具**走通 ✓
    （2026-10-03 ✓ 融合判据换成 **IoU + 参数分档**之后新增 ✓ 见 `_is_merged_box` ✓）。

    ⚠⚠ **为什么必须有它** ✗✗：老夹具一律是"**框与砖同心、面积比 1.2~1.6**"⇒ 与砖的
      `IoU = 砖面积 / 框面积 = 1/1.2 ~ 1/1.6 = 0.63 ~ 0.83` ✗ —— 而新判据要求
      `IoU(框, 砖) < brick_iou_sep`（分离期默认 **0.50** ✗）⇒ **一条都过不了** ✗✗
      （实测：改口径后 tracker 自检一次红 **33 条** ✗）。而**面积比与 IoU 之间没有稳定映射**
      ✗（取决于同心与否 / 谁大谁小 ✓）⇒ 逐条"调到 IoU < 0.5"既费工又不稳 ✗。
    ⇒ **统一放宽**：让老用例继续验证"**链条**"（生成 / 保持 / 分离 / 夹取 / 日志 ✓），
      而**阈值语义**交给专门用例钉 ✓（见 `test_merge_iou_threshold` ㉚ 与
      `test_fuse_exit_threshold` ㉛ ✓ —— 不让夹具兼职 ✗）。

    口径：
      · `merge_iou = 0.9`（第 1 行「同检出 IoU」放宽 ⇒ `_brick_owned` 那条"同一格"判定也更宽 ✓）；
      · `ring_cov_sep = ring_cov_fuse = 0.0`（「框↔圆外接矩形 IoU >」**几乎恒过** ✓
        —— 只有"框与圆**完全不相交**"才会不过 ✓ 那正好是分离类用例想要的 ✓）；
      · `brick_iou_sep = 0.9`（「框↔内砖 IoU <」放宽 ✓ 老夹具的 0.63~0.83 全过 ✓）；
      · `brick_iou_fuse = 1.0` ⇒ **关掉「融合期退出」** ✓（老用例的口径是"**框还包住砖就继续**"
        ✓ 见 `_merge_should_exit` ✓；⚠ 这条**新行为**由 `test_fuse_exit_threshold` 专门钉 ✓）。
    ⚠ 显式传进来的参数**以调用方为准** ✓（`setdefault` ✓）—— 要验分档/阈值就别用 `_LT` ✓。
    """
    kw.setdefault("merge_iou", 0.9)
    kw.setdefault("ring_cov_sep", 0.0)
    kw.setdefault("ring_cov_fuse", 0.0)
    kw.setdefault("brick_iou_sep", 0.9)
    kw.setdefault("brick_iou_fuse", 1.0)
    return LieTracker(**kw)

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
    tr = _LT()
    init_i = None
    errs = []
    track_n = 0
    for i in range(_N):
        frame, truth = _frame(i, bg, decoys, tgt)
        out = tr.process(frame, ts=i * 0.14)
        if i == 0:
            # ⭐⭐ **输出里必须带「目标实际半径」**（用户 2026-10-03 报的那个 bug 就出在"没人接它" ✗）：
            #   演示窗画的**绿圈**、后端 `_ring_cov` 的**前置**、点选标签那两段 —— **全用这一把尺**
            #   （`LieTracker._rad` ✓ 白块阶段学、之后冻结 ✓）。⚠ 别拿 `Runner` 那个"面积等效半径"
            #   顶替 ✗（实测两者 18.0 vs 60.8 ✓ 差 3 倍 ⇒ 点选标签全错 ✓）。
            check("rad" in out and (out["rad"] is None or float(out["rad"]) > 0.0),
                  "`process` 的输出带 `rad`（= 目标实际半径 ✓；开局还没学到 ⇒ 如实 `None` ✓ "
                  "**不许编个 0** ✗）—— 绿圈 / `_ring_cov` / 点选标签三处**共用这一把尺** ✓")
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

    tr2 = _LT()
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
    #   检测器也停 10 帧 ⇒ 仍是预测态（不报 lost ✓）。
    #   ⚠⚠ **"恢复给框 ⇒ 后台重新捕获、贴回目标"这条已停用**（用户 2026-10-02 ✓ 原话：
    #     "**停用这个逻辑**" ✓）⇒ 现在恢复给框**也不会**有"把圆心拽回目标"的动作 ✗
    #     （丢框期 = 纯预测滑行 + 夹取 ✓）。
    tr3 = _LT()
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
    # ⭐⭐ **复活已停用**（用户 2026-10-02 ✓ "停用这个逻辑" ✓）：检测器在 i≥40 恢复给框之后，
    #   位置**不再**被"运动判据 / 检测框"拽回目标 ✓ —— 丢框期只剩「预测滑行 + 夹取」✓。
    #   ⚠ 所以这里**不再断言**"贴回目标"（那是旧行为 ✓），只把结果打出来 + **源码级钉住调用点
    #     已注释** ✓（方法 / 常量仍保留 —— 要接回就放开那两行 ✓ 届时这条会红、提醒同步 ✓）。
    print("  [--] 复活：检测恢复后位置贴着目标（<15px）的最早帧 = %s（⚠ 停用后这**不再由"
          "复活驱动** ✓ —— 只可能是**预测滑行本来就贴得住** ✓ 用户 2026-10-02 ✓）" % (revived,))
    import pathlib as _p9

    _src9 = (_p9.Path(__file__).resolve().parent.parent / "perception"
             / "lie_tracker.py").read_text(encoding="utf-8")
    check("            # if tbox is None and self.lost_n > _LOST_AFTER:" in _src9
          and "            #     self._try_recapture(dets, dt, ts, _mt)" in _src9,
          "复活调用点**已注释停用**（用户 2026-10-02 ✓）—— `_try_recapture` 方法 / `_recap` / "
          "`_RECAP_*` / `_LOST_AFTER` 仍原样保留 ✓（要接回就放开那两行 ✓）")


def test_track_memory():
    """⭐ 检测轨迹的**速度记忆/防抖**（用户 2026-09-30 要求 ③ ✓）与**噪声不许抢判据**
    （要求 ④⑤ ✓）—— 直接喂 `dets` 给 `_update_tracks`（合成框，不依赖画面 ✓）。

    钉三件：
      ① 短暂漏检 ⇒ 轨迹**不删**、**速度不丢**（否则 yolo 一眨眼，速度记录就永久没了 ✗）；
      ② 按**预测位置**重新出现 ⇒ **接回原轨迹**（继承速度 ✓ 而不是从零新建 ✗ 用户 ② ✓）；
      ③ 记忆**到期**（> `_TRK_MEM_S`）⇒ 才真的删（否则表会无限膨胀 ✗）。
    """
    from perception import lie_tracker as LT

    tr = _LT()
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
    tr = _LT()
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
        tr = _LT()
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
    tr = _LT()
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
    tr = _LT()
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

    tr = _LT()
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
    tr = _LT()
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
    tr2 = _LT()
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
    # ⚠ `select_step_ratio=0` ⇒ **关掉「选框距离门」**（用户 2026-10-03 ✓）：本条测的是"选框
    #   优先级 / 三态判据" ✓ 不该被那道**新闸**干扰（新闸默认开 ✓ 它管的是"别挑太远的框" ✓
    #   见 `_select_dist_px` ✓ 自己的用例在 `test_select_step_gate` ✓）。
    tr3 = _LT(select_step_ratio=0.0)
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
    #   ⇒ **规则②（用户 2026-10-02 ✓ "规则②没有让你删过，恢复" ✓）**：框包住砖 ✓ + 包住绿圈
    #   （整圈在框内 ✓）+ 面积与砖相等 ✓ ⇒ **融合红框** ✓ `merged`。
    tr3.pos = (250.0, 250.0)                     # 绿圈落进那块砖的那一格 ✓
    # ⚠⚠ **隔离新闸**（2026-10-03 ✓ 用户流程图新增的「分离判定阈值」那条 ✓）：本用例只测
    #   "进门口径（规则② ✓）" ✗ ⇒ 把面积序列清空（⇒ 面积那条不成立 ✓）、白箭头置零
    #   （⇒ 落点那条判不出 ⇒ 放行 ✓）⇒ 不会误判分离 ✓。
    tr3._merge_area_series = []
    tr3._vel_rel = (0.0, 0.0)
    st2b, _ = tr3._target_box_state([(250.0, 250.0, 150.0, 150.0)], {})
    check(st2b == "merged",
          "③ 框≈砖（面积相等）+绿圈在框里 ⇒ `merged`（实际 %s ✓ 规则②恢复 ✓ 用户 2026-10-02 ✓）"
          % st2b)
    tr3.pos = (300.0, 250.0)
    st3, _ = tr3._target_box_state([(900.0, 900.0, 150.0, 150.0)], {})
    check(st3 == "gone",
          "③ 门内没框 ⇒ 状态 `gone`（实际 %s ✓ 事实 4 ✓ 漏检）" % st3)
    # ④ **群体速度兜底**：速度还没建立时按 `T/dt` 起步 ✓（用户事实 2：目标与假群一起平移）
    tr4 = _LT()
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
        tr = _LT(path_ms=5000)
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

    tr = _LT()
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
        t = _LT()
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
    # ④' **反向（用户 2026-10-02 ✓ 规则②恢复）**：同样"红框 ≈ 在册假框"、但**绿圈还在框里**
    #     （整圈在内 ✓）⇒ **规则② 融合红框** ✓ 不是分离 ✗（用户原话：②"即使面积与砖相等也
    #     属于融合红框" ✓）。
    t4b = _tr(fake)
    t4b.pos = (288.8, 327.8)         # 报告位置落进那格"假框" ✓
    t4b._typ_area = 140.0 * 130.0
    t4b._rad = 56.0
    _st4b, _bx4b = t4b._target_box_state([(288.8, 327.8, 137.3, 132.9)], {})
    check(_st4b == "merged" and _bx4b is not None,
          "⑦ 红框 ≈ 在册假框、但**绿圈还在框里** ⇒ 融合红框（实际 %s ✓ 规则②恢复 ✓）" % _st4b)


def test_box_identity():
    """⭐⭐ **每个检出框都需要它是谁**（用户 2026-10-01 ✓ 原话："每个检出框我们都需要它是谁：
    ① 与上板登记的假目标**高度重合** → 说明检出框**又检出了假目标**，**不能锁红**；
    ② **大**检出框**直接包住**了上板登记的假目标 → **融合信号**" ✓）。

    数值取用户点名的实测帧（`10月1日.mp4`）：帧 31 覆盖率 1.00 / 倍率 **1.55**（用户："此帧红框
    不该标" ✓）；帧 27 倍率 **1.87**、帧 39/40 倍率 **2.05**（并集 = 融合 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _tr(fake, white=None):
        t = _LT()
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
    # ⭐⭐ 本用例要**真实门槛**（`_LT` 的 `ring_cov_sep=0.0` 太宽 ✗ 会把"圈在框外"也放行 ✓）：
    #   这条的本意 = "**两条融合口径都不占** ⇒ 不给红框" ✓；而 2026-10-03 起规则① 的**前置**
    #   换成了 「**圆外接矩形与检出框相交的比例 > `ring_cov`**」✓（**取代**旧"圆心±容差与框相交"
    #   ✓ 用户原话"替换" ✓；⚠ 分母 = **圆矩形面积** ✗ **不是并集** ✓ 用户 2026-10-03 二次修订 ✓）
    #   ⇒ 本拍那个比例只有 ≈ **0.04** ✗ ⇒ 填新默认 **0.80** 就恢复"不融合" ✓。
    t6b.ring_cov_sep = 0.80
    t6b.pos = (222.0, 375.0)                     # 绿圈在框**外**（框 y ≤ 335 ✓）但仍在吸附门内 ✓
    t6b._typ_area = 140.0 * 130.0
    t6b._rad = 56.0
    _st6b, _bx6b = t6b._target_box_state([(222.0, 255.0, 179.0, 167.0)], {})
    check(_st6b == "gone" and _bx6b is None,
          "⑧ 同心、倍率 1.17（< 阈值）且**绿圈在框外** ⇒ `gone`（实际 %s/%s ✓ —— 两条融合口径"
          "都不占 ⇒ 不给红框 ✓ 规则① 的**新前置**「圆矩形∩框 ÷ 圆矩形面积 = %.3f > 0.80」"
          "不成立 ✓）"
          % (_st6b, _bx6b, t6b._ring_cov((222.0, 255.0, 179.0, 167.0))))
    # ⑦ 反过来：**大框包住**（融合）⇒ 照旧给红框 ✓（别把规则 2 一起掐死 ✗）
    t7 = _tr(fake)
    t7.pos = (230.0, 262.0)
    t7._typ_area = 140.0 * 130.0
    t7._rad = 56.0
    _st7, _bx7 = t7._target_box_state([(230.0, 262.0, 225.0, 200.0)], {})
    check(_st7 == "merged" and _bx7 is not None,
          "⑧ 身份「融合」⇒ 照旧给红框且判 `merged`（实际 %s ✓ 规则 ② 生效 ✓）" % _st7)


def test_merge_rule1_center_range_hits():
    """⭐⭐ **规则①定稿（用户 2026-10-02 三次定稿 ✓）：面积比 ≥ 融合框判定阈值 **且「圆心 ±
    噪声容差」范围与检出框相交** ⇒ 标红**（"圆心在框内"是它的 `tol=0` 特例 ⇔ 两次表述数学等价 ✓
    —— 用户点名的最终口径就是"圆心±噪声容差范围与框相交" ✓；更早那版"与绿圈外接圆相交" ✗
    已被否（那把尺被半径放大，帧 22 圆心在框外 25px 也算"相交" ✗✗）✓）。
    "规则②是充分条件，不是必要条件" ✓ 不变。

    帧 17 真实数字（`9月30日(1).mp4` ✓）：并集框 `208.7×165.9` 包住砖(528,340)（登记
    `160.6×134.4`、**锁尺寸中** ✓）比 **1.604** ≥ 阈值、圆心 (359.7,254.8) 在框里 ⇒ **标红** ✓。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _tr():
        t = _LT()
        e = Entry((267.7, 297.6, 160.6, 134.4), ts=0.0)   # 帧 17 砖(528,340) 的当前位置/尺寸 ✓
        e.on_board = True
        e._shape_pending = True                           # ⭐ 形状待定中（圈压着、尺寸不可信 ✓）
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        t._typ_area = 140.0 * 130.0
        t._rad = 56.16                                    # 帧 17 画出来的绿圈半径 ✓
        return t

    # ① 帧 17 那格框：比 1.604 ≥ 阈值、圆心 (359.7,254.8) 在框里（⇒ 范围必相交 ✓）、整圈**进不去**
    #   （规则② ✗）⇒ 走**规则①** ⇒ `merged` ✓
    t1 = _tr()
    t1.pos = (359.7, 254.8)
    st1, bx1 = t1._target_box_state([(291.2, 297.7, 208.7, 165.9)], {})
    check(st1 == "merged" and bx1 is not None,
          "① 锁尺寸条目 + 比 1.604 ≥ 阈值 + **圆心±容差范围与框相交**（整圈进不去）⇒ `merged`"
          "（实际 %s ✓ 规则①=充分条件之外的第二条路 ✓ 帧 17 定稿 ✓）" % st1)
    # ② 同一格框、**范围与框不相交**（圆心 (420,297.7)，离框右缘 24.45px「> 容差」；离框心
    #    128.8 ≤ 吸附门 130 ✓）⇒ 规则① 不占 ⇒ 属砖条目（锁着 ✓）⇒ `gone` 预测态 ✓
    t2 = _tr()
    t2._rad = 20.0
    t2.pos = (420.0, 297.7)
    st2, bx2 = t2._target_box_state([(291.2, 297.7, 208.7, 165.9)], {})
    check(st2 == "gone" and bx2 is None,
          "② 比 ≥ 阈值但**圆心±容差范围与框不相交** ⇒ 不融合、属砖（锁着）⇒ `gone`"
          "（实际 %s ✓ 判别特征=范围交不相交 ✓）" % st2)
    # ③ **圈压着框（外接圆相交 ✓）但圆心出框 20px（> 容差）** ⇒ 照样**不融合** ✓（旧"外接圆
    #   相交"尺的漏判 ✗ 钉死 ✓）
    # ⭐⭐ **用真实门槛**（`_LT` 的 `ring_cov_sep=0.0` 会把这条也放行 ✗）：本拍
    #   「**圆外接矩形与检出框相交的比例**」≈ **0.32** < 0.80 ⇒ 不融合 ✓（= 取代旧"圆心±容差与
    #   框相交"那条判据 ✓ 用户 2026-10-03 原话"**替换**" ✓；⚠ 分母 = **圆矩形面积** ✓ **不是并集** ✓）。
    t3 = _LT(ring_cov_sep=0.80)
    e3 = Entry((250.0, 250.0, 160.0, 160.0), ts=0.0)
    e3.on_board = True                                    # 没锁（排除锁尺寸干扰 ✓）
    rg3 = ShapeRegistry()
    rg3.entries = [e3]
    t3._reg = rg3
    t3._typ_area = 140.0 * 130.0
    t3._rad = 56.0
    t3.pos = (180.0, 250.0)                               # 框左缘 200 ⇒ 圆心在外 20px；r=56 ⇒ 相交 ✓
    check(t3._is_merged_box((300.0, 250.0, 200.0, 180.0)) is False,
          "③ 圈压着框（外接圆相交 ✓）但**圆心出框 20px** ⇒ 新前置「圆矩形∩框 ÷ 圆矩形面积 "
          "%.3f > 0.80」不成立 ⇒ 不融合（实际 %s ✓ —— 旧口径是「圆心±容差与框相交」✗，"
          "2026-10-03 已**替换** ✓ 这条钉的就是新尺 ✓ 帧 22 那类不再靠大半径凑相交 ✓）"
          % (t3._ring_cov((300.0, 250.0, 200.0, 180.0)),
             t3._is_merged_box((300.0, 250.0, 200.0, 180.0))))
    # ④ **噪声容差**（用户 2026-10-02 ✓ 原话："加容差，且将这个容差作为配置参数『噪声容差(px)』
    #   加在『融合框判定阈值』的右边" ✓）：帧 16 真实数字 —— 圆心 (353.5,253.3) 只出框右缘
    #   **0.7px**（比 1.222 ≥ 阈值 1.15 ✓）：容差 0（范围退化成点 ⇒ 与框差 0.7px ✗）⇒ 不融合
    #   （那一拍预测态 ✗）、容差 2（范围伸进框 1.3px ✓）⇒ 融合 ✓。
    t4 = _tr()
    t4.pos = (353.5, 253.3)
    t4.ring_cov_sep = 0.80                                # ⭐ 真实门槛（见 ③ 那段说明 ✓）
    _b4 = (272.9, 311.8, 159.8, 165.0)
    t4.noise_tol = 0.0
    _r4a = t4._is_merged_box(_b4)
    t4.noise_tol = 2.0
    _r4b = t4._is_merged_box(_b4)
    check(_r4a == _r4b,
          "④ ⭐ **噪声容差不再参与融合判定**（2026-10-03 ✓ 那条「圆心±容差与框相交」已被"
          "「框↔圆外接矩形 IoU」**替换** ✓ 用户原话「替换」✓）：容差 0 与 2 判决**相同**"
          "（= %s ✓）—— 它现在的本质用途是**边归属**（框边 ↔ 砖边距离 ✓ 见 `_edge_ownership` ✓）"
          % (_r4a,))
    check(_r4a is False,
          "④ 本拍（帧 16 那组数 ✓ 圆心出框 0.7px）：新前置「圆矩形∩框 ÷ 圆矩形面积 %.3f > "
          "0.80」不成立 ⇒ **不融合** ✓（= 旧口径「容差 0 ⇒ 范围退化成点 ⇒ 不相交」那条换成的新尺 ✓）"
          % (t4._ring_cov(_b4),))
    _t4b = _tr()
    _t4b.pos = (353.5, 253.3)
    _t4b._rad = 0.0                                       # 半径还没学到 ⇒ 那条前置**自动放行** ✓
    check(_t4b._ring_cov(_b4) == 1.0,
          "④ ⚠ **半径还没学到时那条前置自动放行**（`_ring_cov` 回 1.0 ✓ 保守 ✓ 不猜 ✗）"
          "—— 开局那几拍不会因为「圆还没画出来」就全判不融合 ✓")


def test_merge_succ_no_swap():
    """⭐⭐⭐ **融合会话「续任优先」**（用户 2026-10-03 ✓ 方案 ② ✓ 原话："9月30日(1)，帧44→45
    融合框被替换了检出框，有什么方法能够判定并屏蔽掉这种现象吗" ✓）。

    夹具 = `9月30日(1).mp4` 帧 44→45 的真实数值 ✓：
      · 上一拍（帧 44）红框 = `(240.8,157.0) 163.4×161.7` ✓；
      · **延续格** `(219.6,155.1) 168.9×154.9` —— 与上一拍红框重叠 **0.86** ✓ 贴的还是同一块砖
        `(327,104)` ✓；
      · 抢位那格 `(289.3,248.7) 156.2×153.6` —— 重叠只有 **0.31** ✓ 贴的是**另一块砖** `(560,81)` ✓。
    旧代码里两格的 `_hit_sorted` 排序键**并列**（圆心都在框内 ⇒ 键都是 0.0 ✓ ⇒ 由**检测框先后**
    定胜负 ✓）⇒ 抢位那格赢 ⇒ 融合框换对象、参照砖跟着换 ✗✗（用户点名 ✓）。
    本用例：**续任 ⇒ 只许用延续格** ✓；把 `_SWAP_OV` 抬到 0.9 ⇒ 续任不成立 ⇒ 才会换 ✓（钉住门槛 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    import perception.lie_tracker as _LTim

    def _mk():
        # ⚠⚠ **夹具用的阈值放宽到 0.90** ✗（2026-10-03 ✓ 从"面积比"换成 IoU 之后 ✓）：这些夹具的
        #   框与砖**同心**（IoU = 砖/框 = 1/1.2~1/1.55 = **0.65~0.83** ✗）⇒ 用界面常用的 0.5 会让
        #   它们**全都进不了融合** ⇒ 整条"融合链"的用例集体失效 ✗（实测 19 条 ✗）。⇒ 夹具取
        #   **0.90**（= 「只要两框不是同一格就算并集」✓）⇒ 判据链照旧被钉 ✓；**阈值语义**由
        #   `test_merge_iou_threshold` 那类**专门用例**钉 ✓（不让夹具兼职 ✗）。
        t = _LT(noise_tol=25.0, merge_iou=0.9)
        t._typ_area = 140.0 * 130.0
        t._rad = 56.2
        t._tbox_rad = 64.5
        t.pos = (218.7, 178.3)                       # 帧 45 进门圆心 ✓
        t._vel_rel = (256.0, -66.7)                  # 帧 45 白箭头 ✓
        _e1 = Entry((327.0, 104.0, 142.0, 135.0), ts=0.0)   # 参照砖（bid 就是 (327,104) ✓）
        _e1.on_board = True
        _e1.x, _e1.y = 302.5, 226.6                  # 它这一拍的当前位置（实测 ✓）
        _e2 = Entry((560.0, 81.0, 133.0, 131.0), ts=0.0)    # 另一块砖（抢位格贴的那块 ✓）
        _e2.on_board = True
        _e2.x, _e2.y = 304.9, 231.6
        rg = ShapeRegistry()
        rg.entries = [_e1, _e2]
        t._reg = rg
        t._merge_hold = True                         # 前提：**融合保持中** ✓
        t._merge_hold_bid = (327, 104)
        t._prev_tbox = (240.8, 157.0, 163.4, 161.7)  # **上一拍（帧 44）红框** ✓
        return t

    _A = (289.3, 248.7, 156.2, 153.6)                # 抢位那格（贴另一块砖 ✓）
    _B = (219.6, 155.1, 168.9, 154.9)                # 延续格（贴参照砖 ✓）
    _t = _mk()
    # ⭐⭐ **门径 = IoU**（用户 2026-10-03 ✓ 原话："**改成 IoU（交集 ÷ 并集）**" ✓ 见 `_box_iou` ✓）：
    #   旧口径"交集 ÷ **较小面积**"下，**被大框包住的小框会虚高** ✗（实测 `10月1日.mp4` 帧 23：
    #   conf 0.80 的小框 0.901 赢了 0.897 ✗ 而 IoU 只有 0.503 vs 0.790 ✓）。
    check(_t._box_iou(_B, _t._prev_tbox) > 0.5,
          "① 延续格与上一拍红框的 **IoU = %.2f**（≥ 门槛 %.2f ✓ —— 旧口径是 0.86 「较小面积」✗）"
          % (_t._box_iou(_B, _t._prev_tbox), float(_LTim._SWAP_OV)))
    check(_t._box_iou(_A, _t._prev_tbox) < 0.5,
          "① 抢位那格 IoU 只有 %.2f（< 门槛 ⇒ 它不该被当成「延续」✓）"
          % _t._box_iou(_A, _t._prev_tbox))
    # ⚠ dets 顺序按**实测**给（A 在前 ✓）—— 旧代码正是这样让 A 抢先的 ✓
    _st, _bx = _t._target_box_state([_A, _B], {})
    check(_bx is not None and abs(float(_bx[0]) - _B[0]) < 2.0,
          "① ⭐ **续用延续格**（实际 %s ✓ —— 不许被贴另一块砖的那格抢走 ✗ 用户 帧 44→45 ✓）"
          % (_bx,))
    check(getattr(_t, "_merge_hold_bid", None) == (327, 104),
          "① **参照砖不变**（仍是 (327,104) ✓ 旧代码这时会换成 (560,81) ✗✗ = 用户点名的毛病 ✓）")
    # ⭐ 门槛真在起作用：抬到 0.9 ⇒ 续任不成立 ⇒ 这时由**内砖同一性**接着兜住 ✓
    #   （用户 2026-10-03 ✓："**分离前（包括分离当拍）不允许将内砖不同的检出框当做融合框**" ✓）
    #   ⇒ 抢位那格 A 的内砖是**另一块砖** `(560,81)`（覆盖率 0.927 > 参照砖的 0.866 ⇒
    #     `_covered_ref` 取到它 ✗）⇒ **不许当选** ✓ ⇒ 仍落延续格 B ✓（两道门槛一前一后 ✓）。
    _save = _LTim._SWAP_OV
    try:
        _LTim._SWAP_OV = 0.9
        _t2 = _mk()
        _st2, _bx2 = _t2._target_box_state([_A, _B], {})
        check(_bx2 is not None and abs(float(_bx2[0]) - _B[0]) < 2.0,
              "① `_SWAP_OV` 抬到 0.90 ⇒ 续任不成立 ⇒ **内砖同一性兜住**：抢位那格（内砖 = "
              "另一块砖 ✗）不许当选 ⇒ 仍落延续格（实际 %s ✓ ⇒ 两道门槛一前一后都在起作用 ✓）"
              % (_bx2,))
        check(getattr(_t2, "_merge_hold_bid", None) == (327, 104),
              "① 而且**参照砖不变**（仍是 (327,104) ✓ 没被换成 (560,81) ✓）")
    finally:
        _LTim._SWAP_OV = _save
    # ⚠ 会话结束后（`_merge_hold=False`）⇒ **续任逻辑不介入** ⇒ 照旧自由挑 ✓
    _t3 = _mk()
    _t3._merge_hold = False
    _st3, _bx3 = _t3._target_box_state([_A, _B], {})
    check(_bx3 is not None,
          "① 会话结束后照旧能挑（实际 %s ✓ —— 续任只作用于**保持期内** ✓ 不阻新会话 ✓）" % (_bx3,))


def test_sep_vel_cap():
    """⭐⭐⭐ **分离期白箭头模长上限**（用户 2026-10-03 ✓ 原话："将**分离期（淡粉色框时期）**的
    **最大相对假目标群速度（白箭头）** = a × **假目标群体的标准速度**，a 为新配置『**真目标预测
    最大速度倍率**』默认 1.0" ✓）。

    钉五件：
      ① 默认 1.0 ✓、可透传 ✓、夹 `[0, 5]` ✓；
      ② **有红框**（含融合框 ✓）⇒ 一个字不动 ✗（它只管"淡粉接力框"那段 ✓）；
      ③ **没红框** + 超上限 ⇒ **只压模长、方向不变** ✓；
      ④ 没超上限 ⇒ 不动 ✓（**只压不抬** ✓）；
      ⑤ 群体速度判不出（没观测 ✓）⇒ 不夹 ✓。
    """
    import math
    import perception.lie_tracker as _LTim

    def _mk(a=1.0):
        t = _LT(sep_vel_max_ratio=a)
        t._dt_last = 0.1
        t._t_hist = [(20.0, 0.0)]        # 群体每拍 20px ⇒ `_group_vel(0.1)` = (200,0) ⇒ 模 200 ✓
        t._tbox = None                   # **分离期**（红框丢失 = 淡粉接力框那段 ✓）
        return t

    check(abs(LieTracker().sep_vel_max_ratio - 1.0) < 1e-9,
          "① 默认 = 1.0（用户点名 ✓）")
    check(LieTracker(sep_vel_max_ratio=-1.0).sep_vel_max_ratio == 0.0
          and LieTracker(sep_vel_max_ratio=9.0).sep_vel_max_ratio == 5.0,
          "① 夹进 [0, 5] ✓（负数无意义 ⇒ 0 ✓）")
    # ② 有红框 ⇒ 不动
    t0 = _mk()
    t0._tbox = (100.0, 100.0, 150.0, 150.0)
    t0._vel_rel = (600.0, 0.0)
    t0._sep_vel_cap()
    check(t0._vel_rel == (600.0, 0.0),
          "② **有红框 ⇒ 一个字不动**（实际 %s ✓ —— 它只管「淡粉接力框」那段预测期 ✓）"
          % (t0._vel_rel,))
    # ③ 分离期 + 超上限 ⇒ 只压模长、方向不变
    t1 = _mk()
    t1._vel_rel = (600.0, 0.0)           # 模 600 > 1.0 × 200 ⇒ 压到 200 ✓
    t1._sep_vel_cap()
    check(abs(t1._vel_rel[0] - 200.0) < 1e-6 and abs(t1._vel_rel[1]) < 1e-6,
          "③ 模长 600 > 1.0 × 200 ⇒ **压到 200**（实际 %s ✓）" % (t1._vel_rel,))
    t2 = _mk()
    t2._vel_rel = (300.0, 400.0)          # 模 500 ⇒ 压到 200 ⇒ 方向 3:4 保持 ✓
    t2._sep_vel_cap()
    check(abs(math.hypot(*t2._vel_rel) - 200.0) < 1e-6
          and abs(t2._vel_rel[0] * 400.0 - t2._vel_rel[1] * 300.0) < 1e-6,
          "③ **只压模长、方向不变**（500 ⇒ 200，仍 3:4：实际 %s ✓）" % (t2._vel_rel,))
    # ④ 没超上限 ⇒ 不动（只压不抬 ✓）
    t3 = _mk()
    t3._vel_rel = (100.0, 0.0)            # 模 100 < 200 ⇒ 不动 ✓
    t3._sep_vel_cap()
    check(t3._vel_rel == (100.0, 0.0),
          "④ **没超上限 ⇒ 一个字不动**（只压不抬 ✓ 实际 %s ✓）" % (t3._vel_rel,))
    # ⑤ 群体速度判不出 ⇒ 不夹
    t4 = _mk()
    t4._t_hist = []                       # 没观测 ⇒ `_group_vel` 回 None ✓
    t4._vel_rel = (600.0, 0.0)
    t4._sep_vel_cap()
    check(t4._vel_rel == (600.0, 0.0),
          "⑤ **群体速度判不出 ⇒ 不夹**（不猜 ✓ 实际 %s ✓）" % (t4._vel_rel,))
    # ⑥ a 的作用：0 ⇒ 压到 0；2.0 ⇒ 上限翻倍
    t5 = _mk(0.0)
    t5._vel_rel = (600.0, 0.0)
    t5._sep_vel_cap()
    check(math.hypot(*t5._vel_rel) < 1e-9,
          "⑥ a=0（极端）⇒ 分离期白箭头模长压到 0（实际 %s ✓）" % (t5._vel_rel,))
    t6 = _mk(2.0)
    t6._vel_rel = (300.0, 0.0)            # 2.0 × 200 = 400 ⇒ 300 < 400 ⇒ 不动 ✓
    t6._sep_vel_cap()
    check(t6._vel_rel == (300.0, 0.0),
          "⑥ a=2.0 ⇒ 上限 400 ⇒ 300 不动 ✓（倍率真在起作用 ✓ 默认 %.2f ✓）"
          % float(_LTim._SEP_VEL_MAX))


def test_brick_edge_max():
    """⭐⭐⭐ **「砖最多拥有融合框边数量」**（用户 2026-10-02 ✓ 原话："增加参数：『**砖最多拥有
    融合框边数量**』(1~4)，注意在**多条边选取最近的前 x 条**" ✓）。

    夹具：一块砖 `(250,300) 140×130` ⇒ 四条边 eL 180 ／ eR 320 ／ eT 235 ／ eB 365。
    边归属 = 「框这条边距**最近的某块砖**对应边 ≤ 噪声容差」✓（本用例容差 15 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _tr(edge_max):
        t = _LT(noise_tol=15.0, edge_max=edge_max)
        e = Entry((250.0, 300.0, 140.0, 130.0), ts=0.0)
        e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        return t

    # ① 框与砖**完全重合** ⇒ 四条边距离全 0 ⇒ 四条全归砖（`edge_max=4` = 老行为 ✓）
    _t = _tr(4)
    _own, _d = _t._edge_ownership((250.0, 300.0, 140.0, 130.0), detail=True)
    check(_own == [True, True, True, True] and all(abs(float(v)) < 1e-9 for v in _d),
          "① 框与砖完全重合 ⇒ 四条边距离全 0 ⇒ 全归砖（实际 %s ／ %s ✓ `detail=True` 给距离 ✓）"
          % (_own, _d))
    # ② `edge_max=2` ⇒ 只留**最近的前 2 条**（四条并列 ⇒ 按 L→R→T→B 顺序取前两条 ✓）
    _t2 = _tr(2)
    check(_t2._edge_ownership((250.0, 300.0, 140.0, 130.0)) == [True, True, False, False],
          "② `edge_max=2` ⇒ 只留最近的前 2 条（并列按 L→R→T→B ✓ 实际 %s ✓）"
          % (_t2._edge_ownership((250.0, 300.0, 140.0, 130.0)),))
    # ③ 让各边距离拉开：框 (245,294) 120×106 ⇒ 框边 185/305/241/347
    #    距砖边 = L 5 ✓ ／ R 15（卡线 ✓）／ T 6 ✓ ／ B 18 ✗ ⇒ 归砖 3 条（L,T,R ✓）
    _box = (245.0, 294.0, 120.0, 106.0)
    _t3 = _tr(4)
    _o3, _d3 = _t3._edge_ownership(_box, detail=True)
    check(_o3 == [True, True, True, False],
          "③ 容差 15：L=5 ✓ T=6 ✓ R=15（卡线算归 ✓）B=18 ✗ ⇒ 归砖 3 条"
          "（实际 %s ／ 距离 %s ✓）" % (_o3, _d3))
    # ⭐ 用户点名的那条：**归砖的边多于上限 ⇒ 留距离最近的**
    _t4 = _tr(2)
    check(_t4._edge_ownership(_box) == [True, False, True, False],
          "③ `edge_max=2`、归砖的有 L(5)/T(6)/R(15) 三条 ⇒ **留最近的两条** = L ✓ T ✓"
          "（**R 被丢掉** ✗ ⇒ 它改判「真目标提供的边」 ✓ 实际 %s ✓）"
          % (_t4._edge_ownership(_box),))
    # ④ 入参夹紧 + 默认值
    check(LieTracker(edge_max=0).edge_max == 1 and LieTracker(edge_max=9).edge_max == 4,
          "④ 越界夹进 [1, 4] ✓（0 ⇒ 1 ✓ 9 ⇒ 4 ✓）")
    check(LieTracker().edge_max == 4,
          "④ 默认 = **4**（= 不截断 ⇒ 老行为一字不变 ✓）")

    # ⭐⭐ **只看"被这格框包住的砖"**（用户 2026-10-02 ✓ 原话："融合框的归属**只能根据其包含
    #   的砖**去判，例如**帧 27 只看砖 (144,96)**" ✓）：与本次并集无关的砖（它的边只是从框边
    #   **擦过去** ✗）**一律不参与** ✓。
    # 夹具 = 帧 27 的真实数值（`10月1日.mp4` ✓ 容差 25 ✓）：
    #   · 参照砖（被框包住 ✓ 覆盖率 0.885 ✓）：`(302.5,226.6) 142×135`
    #     ⇒ 左边 231.5 ／ 右边 373.5 ／ 上边 159.1 ／ 下边 294.1
    #   · 框外那块（帧 27 的 `bid=(349,51)` ✓ 覆盖率 0.331、砖心 y=151.6 也在框上边外 ✗
    #     ⇒ `_box_covers_entry=False` ✓ 被排除）：`(215.3,151.6) 165×153` ⇒ 左边 **132.8**
    #   · 红框 `(261.9,254.5) 210×171.5` ⇒ 左 156.9 ／ 右 366.9 ／ 上 168.8 ／ 下 340.3
    def _tr27():
        t = _LT(noise_tol=25.0)
        e1 = Entry((302.5, 226.6, 142.0, 135.0), ts=0.0)      # 参照砖（被框包住 ✓）
        e1.on_board = True
        e2 = Entry((215.3, 151.6, 165.0, 153.0), ts=0.0)      # 框外那块（擦线 ✗ 该排除 ✓）
        e2.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e1, e2]
        t._reg = rg
        return t

    _box27 = (261.9, 254.5, 210.0, 171.5)
    _t27 = _tr27()
    check(_t27._box_covers_entry(_box27, _t27._reg.entries[1]) is False,
          "⑤ 前提：框外那块 `(215.3,151.6) 165×153` **不被框包住**（覆盖率 0.331 < 0.5 ✗ 且"
          "砖心在框上边外 ✗ ⇒ `_box_covers_entry=False` ✓）")
    _o27, _d27 = _t27._edge_ownership(_box27, detail=True)
    check(_o27 == [False, True, True, False],
          "⑤ **左边归真目标**（框外那块的左边距框左 **24.1px** ≤ 容差 25 ✓ —— 旧口径会把它算成"
          "『提供左边』✗✗；它不被框包住 ⇒ 按新口径不参与 ✓ 用户帧 27 点名 ✓）；右边 6.6 ✓／"
          "上边 9.7 ✓ 来自参照砖 ⇒ 归砖；下边 46.2 ✗ ⇒ 真目标（实际 %s ／ 距离 %s ✓）"
          % (_o27, _d27))


def test_edge_own_one_brick():
    """⭐⭐⭐ **融合框只认一块内砖 = 覆盖率最大的那块**（用户 2026-10-03 ✓ 原话："我们应该**只允许
    融合框认一个内砖，就是覆盖率最大的那个**" ✓）：四条边**全部只看这一块** ✓ —— **取代**
    2026-10-02 那版"**每边各自取最近那块砖的边距**" ✗（那时一个框里躺着两块砖会**两块各供各的
    边**：实测 `9月30日(1)` 帧 38／46 就是"一框两块砖" ✗）。

    夹具（合成；容差 7px ⇒ 刚好把两块砖分开 ✓）：
      · 框 `(300,300) 200×200` ⇒ 四边 200／400／200／400；
      · 砖 A `(300,300) 180×180` ⇒ 四边 210／390／210／390 ⇒ **每条边距框边 10px**、
        **覆盖率 = 1.00**（整个被框包住 ✓）；
      · 砖 B `(245,300) 100×100` ⇒ 四边 195／295／250／350 ⇒ **左边距框左 5px**（比 A 的 10 近 ✓）、
        覆盖率 = 0.95 ⇒ **旧口径会把"左边"判给 B** ✗；新口径只看 A（覆盖率最大 ✓）⇒
        四条边距离都 = 10 > 7 ⇒ **全不归砖** ✓。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _mk():
        t = _LT(noise_tol=7.0, edge_max=4)
        _a = Entry((300.0, 300.0, 180.0, 180.0), ts=0.0)
        _a.on_board = True
        _b = Entry((245.0, 300.0, 100.0, 100.0), ts=0.0)
        _b.on_board = True
        rg = ShapeRegistry()
        rg.entries = [_a, _b]
        t._reg = rg
        return t, _a, _b

    _box = (300.0, 300.0, 200.0, 200.0)
    _t, _a, _b = _mk()
    check(_t._box_covers_entry(_box, _a) and _t._box_covers_entry(_box, _b),
          "① 前提：A / B **都被这格框包住**（覆盖率 1.00 ／ 0.95 ✓）⇒ 两块都在候选里 ✓")
    _ref, _cov = _t._covered_ref(_box)
    check(_ref is not None and tuple(_ref.bid) == tuple(_a.bid) and abs(_cov - 1.0) < 1e-6,
          "② `_covered_ref` 取**覆盖率最大的那块** = A（覆盖率 %.2f ✓ 实际 %s ✓）"
          % (_cov, None if _ref is None else tuple(_ref.bid)))
    _o, _d = _t._edge_ownership(_box, detail=True)
    check(_o == [False, False, False, False],
          "③ **四条边只看 A**（距框边各 10px > 容差 7 ✗）⇒ 全不归砖 ✓"
          "（旧口径会因「B 的左边距 5px ≤ 7」把左边判给 B ✗✗ —— 距离表 %s ✓）" % (_d,))
    # 对照：只放 B ⇒ 左边确实更近（5px ≤ 7 ✓）⇒ 证明"旧口径会用它、用例有区分力" ✓
    _t2, _a2, _b2 = _mk()
    _t2._reg.entries = [_b2]
    _o2, _d2 = _t2._edge_ownership(_box, detail=True)
    check(_o2 == [True, False, False, False] and abs(float(_d2[0]) - 5.0) < 1e-6,
          "④ 对照：**只剩 B** ⇒ 左边 5px ≤ 7 ⇒ 归砖 ✓（距离表 %s ✓ ⇒ 「B 的边更近」这件事是真的 ✓"
          " —— 所以③ 的「全不归」只能来自「只认覆盖率最大的 A」✓）" % (_d2,))


def test_same_inner_brick():
    """⭐⭐⭐ **分离前（含分离当拍）不许把「内砖不同」的框当融合框**（用户 2026-10-03 ✓ 原话：
    "是否有这样的规则：『**分离前（包括分离当拍）不允许将内砖不同的检出框当做融合框**』，
    如果没有就加上" ✓）：保持期内**一次融合会话只认同一块砖** ✓ —— 候选框认的内砖
    （覆盖率最大的那块 ✓ `_covered_ref` ✓）必须 = 本次会话的参照砖（`_merge_hold_bid` ✓）。

    夹具（合成；几何都算好 ✓）：
      · 参照砖 A `(300,200) 150×140`、另一块砖 B `(450,200) 150×140`（都**已上板** ✓）；
      · 保持期：`_merge_hold=True` + 参照砖 = **A**；
      · 这一拍唯一那个融合框 `(450,200) 190×170`：**整块包住 B**（覆盖率 1.00 ✓）、
        只蹭到 A 的边（覆盖率 0.13 ✗ 算不上"包住" ✓）⇒ 它的**内砖 = B ≠ 参照砖 A** ✓；
        面积比 = 190×170 ÷ (150×140) = **1.54 ≥ 阈值 1.2** ✓、圆心在框内 ⇒ **它本来够格当融合框** ✓
        （`_is_merged_box` 规则① 成立 ✓）⇒ 全靠新这条把它拦下 ✓。
    断言：
      ① 参照砖 = A ⇒ **判分离**（`gone` + `_split_now` ✓）且分离原因 = **"内砖与参照砖不同"** ✓；
      ② 对照：参照砖改成 B ⇒ 同一个框**照常当选**（`merged` ✓）⇒ 差别确实来自"同一性"这条 ✓；
      ③ 非保持期（`_merge_hold=False`）⇒ **放行**（新会话自由挑 ✓ 这条不拦 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _mk(hold_bid):
        t = _LT(noise_tol=25.0, merge_iou=0.9, sep_ratio=1.15,
                       inherit_dist=0.6, allow_back_ratio=0.0)
        _a = Entry((300.0, 200.0, 150.0, 140.0), ts=0.0)
        _a.on_board = True
        _b = Entry((450.0, 200.0, 150.0, 140.0), ts=0.0)
        _b.on_board = True
        rg = ShapeRegistry()
        rg.entries = [_a, _b]
        t._reg = rg
        t.pos = (450.0, 200.0)                 # 圆心落在那个并集框里 ✓
        t._rad = 60.0
        t._tbox_rad = 60.0
        t._prev_tbox = None                    # 不做"几何续任"（本用例只测同一性这条 ✓）
        t._merge_hold = True
        t._merge_hold_bid = None if hold_bid is None else tuple(hold_bid)
        return t, _a, _b

    _box = (450.0, 200.0, 190.0, 170.0)        # ⚠ `_target_box_state` 收的 dets 就是 (cx,cy,w,h) ✓
    _t1, _a1, _b1 = _mk(None)
    _t1._merge_hold_bid = tuple(_a1.bid)       # 参照砖 = A（而这个框的内砖是 B ⇒ 不同 ✓）
    check(_t1._same_inner_brick(_box) is False,
          "① 前提：这格框的**内砖 = B**（覆盖率 1.00），而本次会话的参照砖 = A ⇒ 不同 ⇒ `False` ✓")
    check(_t1._is_merged_box(_box) is True,
          "① 前提：它**本来够格当融合框**（包住已上板的 B + 面积比 1.54 ≥ 阈值 1.2 + 圆心在框内 ✓）"
          " ⇒ 拦住它的只能是新那条 ✓")
    _st1, _bx1 = _t1._target_box_state([_box], {})
    check(_st1 == "gone" and _bx1 is None and bool(_t1._split_now),
          "① 参照砖 = A ⇒ **不许当选** ⇒ 门内没有能挑的格 ⇒ **判分离**"
          "（实际 %s ／ 框 %s ／ `_split_now`=%s ✓）" % (_st1, _bx1, _t1._split_now))
    check((_t1._split_why or {}).get("how") == "内砖与参照砖不同",
          "① 分离原因写明是**内砖不同**（实际 %r ✓ —— 不是笼统的「门内没有能挑的格」✓）"
          % ((_t1._split_why or {}).get("how"),))
    _t2, _a2, _b2 = _mk(None)
    _t2._merge_hold_bid = tuple(_b2.bid)       # 参照砖改成 **B**（= 那个框的内砖 ✓）
    _st2, _bx2 = _t2._target_box_state([_box], {})
    check(_st2 == "merged" and _bx2 is not None,
          "② 对照：参照砖 = B（= 内砖）⇒ 同一个框**照常当选**（实际 %s ✓ ⇒ 差别来自「同一性」✓）"
          % (_st2,))
    _t3, _a3, _b3 = _mk(None)
    _t3._merge_hold_bid = tuple(_a3.bid)
    _t3._merge_hold = False                                  # 非保持期（新会话 ✓）
    check(_t3._same_inner_brick(_box) is True,
          "③ 非保持期 ⇒ **放行**（新会话自由挑 ✓ 这条不拦 ✓）")


def test_merge_edge_own():
    """⭐⭐⭐ **融合态圆落位 =「边归属」**（用户 2026-10-02 ✓ 五档 a~e ✓）。

    · 归属（`_edge_ownership` ✓）：融合框某边 与 **已登记砖**某边 距离 ≤「噪声容差」⇒
      该边归假目标 ✓（线段对线段：垂直间距 ⊕ 段间空隙 ✓ 错开的边不误判 ✓）；
    · a `n=0` ⇒ 框中心；b `n=1` ⇒ 贴**对面**边（只改一个坐标 ✓ 另一轴照旧预测 ✓）；
      c `n=2` 相邻 ⇒ 真边**对角内接**（两坐标都定 ✓）；d `n=3` ⇒ 贴"中间那条边"对面
      （只改一个坐标 ✓）；e `n=4` ⇒ **`None`**（不落位 ✓ 见下）；
    · c 退化（假边**对面** L+R / T+B ⇒ 真边不成角）⇒ **`None`** ✓。
    ⚠⚠ **2026-10-02 用户口径修订**（原话："**预测优先，只有边归属明确时才落位**，若出现
      『分离』信号（融合检出框缩小后，判定 **4 条边都归属砖**）**也按预测**" ✓）：
      `n=0` / `n=4` / c 退化**一律 `None`**（= 不落位 ⇒ 圆继续按预测走 ✓），
      只有 `n=1/2/3` 且能算出确定的对面边/对角才落位 ✓ —— 旧口径"a/e ⇒ 框中心"**已废** ✗。
    夹具：砖 (250,300) 140×130（eL180/eR320/eT235/eB365）、容差 2、内接半径 61.4、
    绿圈半径 56 ✓（与实测帧 27 同一把尺 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _tr(pos):
        t = _LT(noise_tol=2.0)
        t._typ_area = 140.0 * 130.0
        t._tbox_rad = 61.4            # ⚠ **画圈那把尺**（= 内接半径 ✓ 实测帧 27 值 ✓）
        t._rad = 56.0                 # 绿圈半径（规则②判"整圈装得下"用 ✓）
        t.pos = pos
        e = Entry((250.0, 300.0, 140.0, 130.0), ts=0.0)
        e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        return t

    def _pt(p):
        return "None" if p is None else "(%.1f,%.1f)" % (p[0], p[1])

    def _eq(p, x, y):
        return p is not None and abs(p[0] - x) < 1e-6 and abs(p[1] - y) < 1e-6

    # a（n=0）：砖悬在框正中、四边都离砖边 30px > 容差 ⇒ **没有边归假目标** ⇒ **不落位** ✓
    #   （2026-10-02 口径：边归属**不明确** ⇒ 按预测 ✓ 原来返回"框中心"✗ 会把圆从角上一跳 ~48px）
    tA = _tr((250.0, 300.0))
    fA = tA._merge_edge_own((250.0, 300.0, 200.0, 190.0), at=(260.0, 290.0))
    check(fA is None,
          "a n=0（四边都不贴砖 = **不明确**）⇒ **不落位**（None ⇒ 按预测 ✓ 实际 %s ✓）" % _pt(fA))

    # b（n=1）：框只向**右**扩（左边 = 砖左边 180 ✓ 其余三边都远）⇒ 圆贴**对面（右）边**、
    #   y 照旧预测（只改一个坐标 ✓）
    tB = _tr((390.0, 290.0))
    fB = tB._merge_edge_own((290.0, 290.0, 220.0, 180.0), at=(390.0, 290.0))
    check(_eq(fB, 400.0 - 61.4, 290.0),
          "b n=1（左边贴砖）⇒ 圆贴**右边**内接 x=%.1f、y 照旧预测 290 ✓（实际 %s ✓）"
          % (400.0 - 61.4, _pt(fB)))

    # c（n=2 相邻）：框向**右下**扩（左、上 = 砖边 ✓）⇒ 真边 = 右+下 ⇒ **对角内接** ✓
    tC = _tr((410.0, 390.0))
    fC = tC._merge_edge_own((300.0, 317.5, 240.0, 165.0), at=(410.0, 390.0))
    check(_eq(fC, 420.0 - 61.4, 400.0 - 61.4),
          "c n=2（左+上贴砖）⇒ 圆 = 真边（右+下）**对角内接** (%.1f,%.1f) ✓（实际 %s ✓）"
          % (420.0 - 61.4, 400.0 - 61.4, _pt(fC)))

    # c 退化（n=2 对面）：框左右 = 砖左右 ✓ 上下远 ⇒ 真边（上+下）不成角 ⇒ **不落位** ✓
    tC2 = _tr((250.0, 300.0))
    fC2 = tC2._merge_edge_own((250.0, 300.0, 140.0, 200.0), at=(250.0, 300.0))
    check(fC2 is None,
          "c 退化 n=2（左+右贴砖、真边不成角 = 不明确）⇒ **不落位**（实际 %s ✓）" % _pt(fC2))

    # d（n=3）：框只向**右**扩、上/下/左全 = 砖边 ✓ ⇒ "中间那条边" = 左 ⇒ 贴**右**边、
    #   y 轴两边都是假边 ⇒ y 照旧预测（只改一个坐标 ✓）
    tD = _tr((410.0, 300.0))
    fD = tD._merge_edge_own((300.0, 300.0, 240.0, 130.0), at=(410.0, 300.0))
    check(_eq(fD, 420.0 - 61.4, 300.0),
          "d n=3（左/上/下贴砖）⇒ 圆贴中间边（左）对面 = **右边** x=%.1f、y 照旧预测 ✓（实际 %s ✓）"
          % (420.0 - 61.4, _pt(fD)))

    # e（n=4）：框四边都离砖边 1.5px ≤ 容差 ⇒ **四条边全归砖** = 用户点名的「**分离**」形态 ✓
    #   ⇒ **按预测**（`None` ✓ 2026-10-02 口径："判定 4 条边都归属砖**也按预测**" ✓）
    tE = _tr((250.0, 300.0))
    fE = tE._merge_edge_own((250.0, 300.0, 143.0, 133.0), at=(250.0, 300.0))
    check(fE is None,
          "e n=4（四边全贴砖 = 用户点名的「分离」形态）⇒ **也按预测**（None ✓ 实际 %s ✓）" % _pt(fE))

    # 非并集 ⇒ None（照旧预测 ✓）
    tN = _tr((400.0, 250.0))
    check(tN._merge_edge_own((288.8, 327.8, 137.3, 132.9), at=(400.0, 250.0)) is None,
          "非并集框 ⇒ 不修正（None ⇒ 纯预测 ✓）")


def test_unregistered_box_beats_merge():
    """⭐⭐ **选框优先级（用户 2026-10-02 ✓ 帧 22 原话："与上一帧预测的圆相交的位置有未登记的
    检出框，这种情况优先标红，融合优先级应该更低" ✓）**：绿圈压着的**未登记**检出框在场时，
    并集框让位 ✓ —— 红框 = 目标自己那格（`ok` ✓）不是并集 ✓。"""
    from perception.lie_registry import Entry, ShapeRegistry
    # ⚠ `select_step_ratio=0` ⇒ **关掉「选框距离门」**（用户 2026-10-03 ✓）：本条钉的是"未登记框
    #   优先于并集框" ✓ 两格都在门内时才有意义 ⇒ 显式关掉新闸（自己的用例见 `test_select_step_gate` ✓）。
    # ⚠⚠ 本用例要**真实的内砖门槛**（`brick_iou_sep = 0.5` ✗ 不能用宽口径 0.9）：那个"并集框"
    #   与砖的 `IoU = 25600/44000 = **0.58**` ⇒ 0.5 下**不算并集** ✓（让位 ✓）、0.9 下算并集 ✗
    #   （会把未登记框挤掉 ✗ 正是本用例要排除的情形 ✗）。
    t = _LT(select_step_ratio=0.0, brick_iou_sep=0.5)
    e = Entry((250.0, 250.0, 160.0, 160.0), ts=0.0)
    e.on_board = True
    rg = ShapeRegistry()
    rg.entries = [e]
    t._reg = rg
    t._typ_area = 140.0 * 130.0
    t._rad = 56.0
    t.pos = (310.0, 250.0)         # 圆心在并集框里 ✓ 也压着旁边那格未登记框 ✓
    st, bx = t._target_box_state([(300.0, 250.0, 220.0, 200.0),    # 并集框（比 1.72 ≥ 阈值 ✓）
                                  (280.0, 240.0, 140.0, 130.0)], {})  # 未登记框（比 0.71 ✓）
    check(st == "ok" and bx is not None and abs(float(bx[0]) - 280.0) < 1e-6,
          "⑨ 绿圈压着的**未登记框**在场 ⇒ 优先标红、并集框让位"
          "（实际 %s/%s ✓ 帧 22 定稿 ✓）"
          % (st, None if bx is None else (round(float(bx[0]), 1), round(float(bx[1]), 1))))


def test_tbox_prefers_merge_over_fake():
    """⭐⭐ **多个框同时包含报告位置时，选框要优先"并集框"而非"假砖框"**（用户 2026-10-01 ✓
    原话："大检出框是套在砖(122,205)身上的框啊" ✓）。

    实测根因（`9月30日(1).mp4` 帧 38）：报告位置 `(147.7,253.6)` 同时落进两个框——
    ① 假砖小框 `(122,274 136×136)`（在册、`_box_is_registered_fake=True` ✓）；② 并集大框
    `(178,193 194×187)`（含 pos、不是假框 ✓）。老实现取"列表里第一个" = 假砖小框 ⇒ 判 `gone`
    ⇒ 红框凭空消失 ✗✗。修后应取并集大框（`merged`）✓。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    tr = _LT()
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
    tr2 = _LT()
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
    tr = _LT()
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
    tr2 = _LT()
    tr2._vel = (0.0, 0.0)
    tr2._vel_rel = (0.0, 100.0)
    _b2 = (tuple(tr2._vel), tuple(tr2._vel_rel))
    tr2._anti_group_vel()
    check((tuple(tr2._vel), tuple(tr2._vel_rel)) == _b2,
          "⑧ 群体≈0 ⇒ 不旋转（原样 ✓ —— 没有『上游』可指 ✗）")
    tr3 = _LT()
    tr3._vel = (200.0, 0.0)
    tr3._vel_rel = (0.0, 0.0)
    _b3 = (tuple(tr3._vel), tuple(tr3._vel_rel))
    tr3._anti_group_vel()
    check((tuple(tr3._vel), tuple(tr3._vel_rel)) == _b3,
          "⑧ 自身≈0 ⇒ 不旋转（原样 ✓ —— 箭头本来就是个点 ✗）")


# ---- 跑批骨架（同其它 selftest 的口径 ✓）----
_FAILED = []


def check(cond, msg):
    if cond:
        print("  [OK] %s" % msg)
    else:
        print("  [NG] %s" % msg)
        _FAILED.append(msg)


def test_split_now_flag_and_forecast():
    """⭐⭐ **分离那一拍 = 按预测走**（用户 2026-10-02 ✓ 原话两条："若出现『分离』信号（融合检出框
    缩小后，判定 **4 条边都归属砖**）**也按预测**" ✓ + "查第 40 帧分离后，为何绿圆没有按照 39 帧
    的预测位置行进" ✓）。

    钉两件：
      ① `_target_box_state` 判出分离（框塌缩成砖 ⇒ 返回 `None`）时 **`_split_now` 置真** ✓
         —— 它是 `process()` 位置滑行分支认的标志 ✓；`_merge_split` **不是**它 ✗（后者在
         "融合到一半丢框"时也为真 ✓ 两者语义不同 ✓）。
      ② 那条滑行分支**起点用 `_report_prev`**（= 上一拍**真正画出来的那个圆** ✓）而不是
         `_pos_prev`（未修正 ⇒ 实测帧 39 差 **73.7px** ✗）。行为已在真素材上逐帧验过 ✓：
         帧 39→40 位移 `(+7.7,−16.4)`、而 39 帧白箭头预测 `(+9.9,−13.8)` ✓ 不再反向 ✗。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    t = _LT(noise_tol=15.0)
    _e = Entry((270.0, 285.0, 170.0, 140.0), ts=0.0)
    _e.on_board = True
    _rg = ShapeRegistry()
    _rg.entries = [_e]
    t._reg = _rg
    t._merge_hold = True                      # 前提：正在融合保持中 ✓
    t._merge_hold_bid = tuple(_e.bid)
    # ⚠ 圆心挪到砖的**下方**、白箭头朝下（用户 2026-10-02 ✓ 分离信号第二条："框中心在轨迹
    #   预测方向的**反方向**" ✓）⇒ 框心 − 圆心 ≈ (0, −35) ⇒ 点积 < 0 ✓ 反方向 ✓
    #   （放在同心点的话点积 = 0 ⇒ `_box_ahead_of_pred` 回 True ⇒ **不判分离** ✗ 实测踩过 ✓）。
    t.pos = (270.0, 320.0)
    t._vel_rel = (0.0, 100.0)
    t._tbox_rad = 64.5
    # ⭐ 分离信号**三条**都要成立（用户 2026-10-02 ✓）：③ = "融合期出现最小面积**之后**又
    #   张开过"⇒ 按拍序给一串（最小 20000 在第 2 拍、之后最大 40000 ⇒ 比 2.0 ≥ 1.3 ✓）。
    t._merge_area_series = [40000.0, 20000.0, 40000.0]
    # ⚠ **故意把画圈半径放大**：让规则②（整圈绿圈在框内）**不成立** ✗ ⇒ `_is_merged_box`
    #   只剩规则①（面积比 23491/23800 = 0.99 < 阈值 ⇒ **不融合** ✓）⇒ 这才走得到"分离"那一支 ✓
    #   （否则会被 `_is_merged_box` 先判成"融合保持"✗ 实测踩过 ✓）。
    t._rad = 80.0
    # "框塌缩成砖 ± 噪声容差" **且** 框心在预测反方向（`_box_is_brick_det` 两条都成立 ⇒ 分离 ✓
    #   ✓ 面积 ≤ 砖 ✓、框心相对圆心在**白箭头的反方向**（圆心在下方、箭头朝下 ⇒ 框在后方 ✓））
    _st, _bx = t._target_box_state([(270.0, 285.0, 169.0, 139.0)], {})
    # ⭐⭐⭐ **2026-10-03 口径 ×2**（同一件事的两半 ✓）：
    #   ① 用户当天先定："照着我的逻辑**简化分离判定**" + "只留『框丢了』，其余删" ⇒ **"框塌缩成砖"
    #      **本身不再**是分离条件 ✗（它只是"该分手"的**征兆**之一 ✓）；
    #   ② 同日再定："**把门打开**" ✓ —— 原来那段的前置条件是 `_is_merged_box or _hold_active` ✓，
    #      而 `_hold_active` 的定义含"**没缩成砖**" ✗ ⇒ "缩成砖"一成立它**必然为假** ⇒ 整段被
    #      跳过 ⇒ `_hover` / `_span` **压根没被求值** ✗（实测 `9月30日(1).mp4` 帧 47：三条齐
    #      却继续当融合框 ✗ —— 用户："报警响了，但值班室的门锁着，而锁门的就是报警本身" ✓）。
    #      ⇒ 现在门改成"**还在保持会话 + 框还包着参照砖**"就能进 ✓（见 `_target_box_state` ✓）。
    #   ⇒ 下面前提：本夹具三条齐（① 缩成砖 ✓、② 反方向 ✓、③ 面积跨度 40000/23491 = **1.70** ≥ 阈值 ✓）
    #      ⇒ **判分离** ✓（原因取 `_split_why.how`：`_hover` 先判 ⇒ "落点倒退超限" ✓，否则
    #      "面积跨度超阈值" ✓ —— 两条任一成立都分手 ✓）。
    check(_st == "gone" and bool(getattr(t, "_split_now", False)),
          "⑰ **三条齐（缩成砖 + 反方向 + 面积跨度 1.70 ≥ 阈值）⇒ 判分离**（实际 %s ／ "
          "`_split_now`=%s ✓ —— 用户 2026-10-03「**把门打开**」✓）"
          % (_st, bool(getattr(t, "_split_now", False))))
    _how = (getattr(t, "_split_why", None) or {}).get("how")
    check(_how in ("落点倒退超限", "面积跨度超阈值"),
          "⑰ 分离原因 = `_hover` / `_span` 之一（实际 %r ✓ —— 不再是「框塌缩成砖」那条 ✗，"
          "它只是征兆 ✓）" % (_how,))
    # ⭐ **对照：钉住"缩成砖**本身**不判分离"** ✓（用户 2026-10-03 口径 ✓）：同样是"缩成砖 +
    #   反方向"，但 ③ **不成立**（历史里没有"最小之后又张开"⇒ 序列只有 1 个元素 ✓）且
    #   **落位闸放行**（`allow_back_ratio = 3` ⇒ 允许倒退 3×半径 ⇒ `_hover` 假 ✓）
    #   ⇒ **照旧算融合保持** ✓（框还包着参照砖 ⇒ 走保持支 ✓）。
    t2 = _LT(noise_tol=15.0, allow_back_ratio=3.0)
    _e2 = Entry((270.0, 285.0, 170.0, 140.0), ts=0.0)
    _e2.on_board = True
    _rg2 = ShapeRegistry()
    _rg2.entries = [_e2]
    t2._reg = _rg2
    t2._merge_hold = True
    t2._merge_hold_bid = tuple(_e2.bid)
    t2.pos = (270.0, 320.0)
    t2._vel_rel = (0.0, 100.0)
    t2._tbox_rad = 64.5
    t2._rad = 80.0
    t2._merge_area_series = [23491.0]        # ⚠ 只有一个元素 ⇒ "最小之后"为空 ⇒ ③ 不成立 ✓
    _st2, _bx2 = t2._target_box_state([(270.0, 285.0, 169.0, 139.0)], {})
    check(_st2 == "merged" and _bx2 is not None
          and not bool(getattr(t2, "_split_now", False)),
          "⑰ 对照：**「缩成砖」本身不再判分离**（③ 不成立 + 落位闸放行 ⇒ 实际 %s ✓ 照旧融合保持 ✓ "
          "用户 2026-10-03「只留框丢了，其余删」✓）" % (_st2,))
    check(not bool(getattr(t, "_merge_split", False)),
          "⑰ `_merge_split` 也不为真（那条是「上一拍融合、这一拍不是」✓ 与 `_split_now` 是两件事 ✗）")
    import pathlib as _pl
    _src = (_pl.Path(__file__).resolve().parent.parent / "perception"
            / "lie_tracker.py").read_text(encoding="utf-8")
    check("if (self._split_now and not self._merge_first" in _src
          and "_rp7" in _src and "_report_prev" in _src,
          "⑰ 滑行分支：分离那一拍**用 `_report_prev` 当起点**（源码级 ✓ 行为已逐帧验过 ✓）")


def test_pick_log():
    """⭐⭐⭐ **非融合红框选择的判定日志**（用户 2026-10-03 ✓ 原话："在日志加一下**非融合红框
    选择**的判定信息吧" ✓）。

    钉三件：
      ① 决策链走到"**目标自己那格**"（`_circle_hits` ✓）⇒ `_pick_why` 有值 ✓ 且
         `_build_pick_log()` 给出 `text` + `data` ✓（`data` 里 `hit` 真、`merged`/`owned` 假 ✓）；
      ② **对照：融合那一支不该有 pick 日志** ✗（那条由 `merge_log` 管 ✓）—— 同一个框，门里
         放一块"面积过阈值"的登记砖 ⇒ 变成融合 ⇒ `_pick_why` 空 ✓；
      ③ 半径还没学到（`_rad = 0`）⇒ 日志**退回"圆心在框内"那一档**说 ✓ 不写"垂距比 0.00 /
         半径 0.0"那种**看着像真值**的数 ✗（实测帧 4/5 就是这种 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    # ① 非融合红框：门内只有"与绿圈相交"的那格、门里没有登记砖 ⇒ `ok` ✓
    t = _LT(noise_tol=15.0)
    t.pos = (300.0, 200.0)
    t._rad = 60.0
    t._vel_rel = (100.0, 0.0)
    _st, _bx = t._target_box_state([(300.0, 200.0, 140.0, 140.0)], {})
    _p = t._build_pick_log()
    check(_st == "ok" and _bx is not None and _p is not None
          and isinstance(_p.get("text"), str) and _p["text"].strip(),
          "⑳ 非融合红框（目标自己那格）⇒ 有 `pick_log`（实际 %s ／ %s ✓ 用户 2026-10-03 ✓）"
          % (_st, "有" if _p else "无"))
    _d = (_p or {}).get("data") or {}
    check(bool(_d.get("hit")) and not _d.get("merged") and not _d.get("owned"),
          "⑳ 日志 `data`：与绿圈相交 = **真** ✓ ／ 是融合框 = 假 ✓ ／ 属于砖 = 假 ✓（实际 %s ✓）"
          % ({k: _d.get(k) for k in ("hit", "merged", "owned")},))
    check("融" in (_p or {}).get("text", "") and "非融合" in (_p or {}).get("text", ""),
          "⑳ 文本是**通俗中文**（含「非融合」字样 ✓ 与融合日志区分得开 ✓）")

    # ② 对照：门里放一块面积过阈值的登记砖 ⇒ 走融合 ⇒ **不该**有 pick 日志 ✓
    t2 = _LT(noise_tol=15.0)
    t2.pos = (300.0, 200.0)
    t2._rad = 60.0
    t2._vel_rel = (100.0, 0.0)
    _e = Entry((300.0, 200.0, 118.0, 118.0), ts=0.0)     # 面积 13924 ⇒ 比 19600/13924 = 1.41 ≥ 1.4 ✓
    _e.on_board = True
    _rg = ShapeRegistry()
    _rg.entries = [_e]
    t2._reg = _rg
    _st2, _bx2 = t2._target_box_state([(300.0, 200.0, 140.0, 140.0)], {})
    check(_st2 == "merged" and t2._build_pick_log() is None,
          "⑳ 对照：**融合那一支不给 pick 日志** ✗（实际 %s ／ %s ✓ 那条由 merge_log 管 ✓）"
          % (_st2, "有" if t2._build_pick_log() else "无"))

    # ③ 半径还没学到 ⇒ 退回"圆心在框内"那一档说 ✓（不写"半径 0.0"这种假数 ✗）
    t3 = _LT(noise_tol=15.0)
    t3.pos = (300.0, 200.0)
    t3._rad = 0.0
    t3._vel_rel = (100.0, 0.0)
    t3._target_box_state([(300.0, 200.0, 140.0, 140.0)], {})
    _p3 = t3._build_pick_log()
    check(_p3 is not None and "半径 0.0" not in _p3["text"]
          and "圆心在框内" in _p3["text"],
          "⑳ `_rad = 0`（半径还没学到）⇒ 日志说「退回『圆心在框内』那一档」 ✓"
          "（不写「垂距比 0.00／半径 0.0」 ✗ 实测帧 4/5 就是这种 ✓）")


def test_kf_vs_classic_on_synth():
    """⭐⭐⭐ **合成台对照：同一段有真值的序列，经典 vs KF**（用户 2026-10-03 ✓ 原话："加一条
    「合成台对照」—— 同一段**有真值的序列**，同时跑经典与 KF" ✓）。

    做法：用本项目现成的**合成台**（`_world()` / `_frame()` ✓ 每帧都带**真值** `truth` ✓）跑两遍
      · `show_kf=False`（现状 ✓ 与 `True` 的**位置输出必须逐帧一致** ✓）；
      · `show_kf=True`（卡尔曼 ✓ **只记录、不改位置** ✓）。

    逐帧收四类量（都拿**真值**当尺 ✓）：
      · **经典位置误差** ／ **KF 位置误差**（px ✓）；
      · **逐帧位移抖动** = `|本拍位移 − 上拍位移|` ✓（"抖不抖"的直接量 ✓ 两套各一份 ✓）。

    钉两件（⚠ **不硬断言"KF 更准"** ✗ —— 那要先把 Q/R 调对 ✓ 这里只钉它的**性质** ✓）：
      ① **影子不污染**：两遍的 `out["pos"]` **逐帧一字不差** ✓（同 `kf_shadow` 的设计目标 ✓）；
      ② **平滑有效**：KF 的位移抖动 p50 **小于**经典的 ✓（这就是"平滑"的定义 ✓）。
    对照表（误差 p50/p90、抖动 p50/p90）**打印出来** ✓ —— "要不要让 KF 接管"就看它 ✓。
    """
    def _q(v, p):
        v = sorted(v)
        if not v:
            return 0.0
        return float(v[min(len(v) - 1, int(round(p * (len(v) - 1))))])

    bg, decoys, tgt = _world()
    _ta = _LT(show_kf=False)
    _tb = _LT(show_kf=True)
    _e_cls, _e_kf, _j_cls, _j_kf = [], [], [], []
    _prev = {"cls": None, "kf": None}
    _dprev = {"cls": None, "kf": None}
    _same = True
    for i in range(_N):
        frame, truth = _frame(i, bg, decoys, tgt)
        _oa = _ta.process(frame, ts=i * 0.14)
        _ob = _tb.process(frame, ts=i * 0.14)
        _pa, _pb = _oa.get("pos"), _ob.get("pos")
        if (_pa is None) != (_pb is None):
            _same = False
        elif _pa is not None and (abs(_pa[0] - _pb[0]) > 1e-9
                                  or abs(_pa[1] - _pb[1]) > 1e-9):
            _same = False
        _kp = (_ob.get("kf") or {}).get("pos")
        if _pa is not None:
            _e_cls.append(((_pa[0] - truth[0]) ** 2 + (_pa[1] - truth[1]) ** 2) ** 0.5)
        if _kp is not None:
            _e_kf.append(((_kp[0] - truth[0]) ** 2 + (_kp[1] - truth[1]) ** 2) ** 0.5)
        for _k, _p in (("cls", _pa), ("kf", _kp)):
            if _p is None:
                continue
            _pv = _prev[_k]
            if _pv is not None:
                _d = ((_p[0] - _pv[0]) ** 2 + (_p[1] - _pv[1]) ** 2) ** 0.5
                _dp = _dprev[_k]
                if _dp is not None:
                    (_j_cls if _k == "cls" else _j_kf).append(abs(_d - _dp))
                _dprev[_k] = _d
            _prev[_k] = _p

    check(_same,
          "㉔ ⭐ **影子不污染**（同一段合成序列：经典与 KF 影子的 `out[\"pos\"]` **逐帧一字不差** ✓）")
    print("  [--] ㉔ 合成台对照（n=%d）："
          "经典误差 p50 %.2f / p90 %.2f px ｜ KF 误差 p50 %.2f / p90 %.2f px ｜ "
          "经典抖动 p50 %.2f / p90 %.2f px ｜ KF 抖动 p50 %.2f / p90 %.2f px"
          % (len(_e_cls), _q(_e_cls, 0.5), _q(_e_cls, 0.9),
             _q(_e_kf, 0.5), _q(_e_kf, 0.9),
             _q(_j_cls, 0.5), _q(_j_cls, 0.9),
             _q(_j_kf, 0.5), _q(_j_kf, 0.9)))
    check(_q(_j_kf, 0.5) < _q(_j_cls, 0.5),
          "㉔ ⭐ **平滑有效**：KF 位移抖动 p50 **%.2f** px < 经典 **%.2f** px ✓"
          "（影子的意义就在这里 ✓ 但「更平滑 ≠ 更准」⇒ 上面那张误差表才算数 ✓）"
          % (_q(_j_kf, 0.5), _q(_j_cls, 0.5)))


def test_gate_base_is_prev_pred():
    """⭐⭐⭐ **落位闸的基准 = 「上拍圆心 + 上拍白箭头 × dt」**（用户 2026-10-03 ✓ 原话澄清："我的
    流程图里写的原话是『**对比上一拍的白箭头预测位置**』意思是**假设本拍按上拍预测走了之后**的本拍
    白箭头（因为不一定按上一拍预测走，所以是『假设』拿来判定用的）" ✓ ＋ 明确要求"**a. 位置基准**
    换成『上拍圆心 + 上拍白箭头 × dt』" ✓）。

    钉三件：
      ① **基准点就是它**（猴补丁抓 `_merge_edge_own(at=...)` 的参数 ✓）—— 实测 `10月2日.mp4`
         帧 37：上拍圆心 (168.8,253.4) ＋ 上拍白箭头 (−172.4,−50.8) × dt 0.133 = **(145.9,246.6)** ✓；
      ② 拿不到上拍圆心 / 上拍白箭头 / dt ⇒ **退回当刻 `self.pos`** ✓（第 0 拍 ✓ 不猜 ✗）；
      ③ **判定后果**：以它为基准 ⇒ 帧 37 的落位位移 **162.5px**、**逆着上拍白箭头** ⇒ **拦下** ✓
         —— 旧口径（拿"当刻圆心"）只算 56.2px ✗ 会放行 ✗，那正是帧 37 圆心一拍走 138px 的由来 ✓。
    """
    _seen = []
    _orig = LieTracker._merge_edge_own
    try:
        LieTracker._merge_edge_own = (
            lambda self, box, at=None: (_seen.append(at), (307.0, 226.2))[1])
        t = _LT(noise_tol=25.0, allow_back_ratio=0.5)
        t.pos = (250.9, 226.2)                  # 当刻圆心（帧 37 实测 ✓）
        t._report_prev = (168.8, 253.4)         # 上拍圆心 ✓
        t._vel_rel_prev = (-172.4, -50.8)       # 上拍白箭头 ✓
        t._vel_rel = (-172.4, -50.8)            # 当拍（「漂移纠正」已挪到闸之后 ⇒ 闸看到的还是它 ✓）
        t._dt_last = 0.133
        t._tbox_rad = 75.7
        t._rad = 63.9
        _r = t._merge_fix_safe((278.9, 201.3, 206.0, 200.0))
        _at = _seen[-1] if _seen else None
        _exp = (168.8 + (-172.4) * 0.133, 253.4 + (-50.8) * 0.133)
        check(_at is not None and abs(_at[0] - _exp[0]) < 0.2 and abs(_at[1] - _exp[1]) < 0.2,
              "㉘ 基准 = **上拍圆心 + 上拍白箭头 × dt** = (%.1f,%.1f)（实际 %s ✓ 用户 2026-10-03 ✓）"
              % (_exp[0], _exp[1], None if _at is None else "(%.1f,%.1f)" % _at))
        check(_r is False,
              "㉘ 以它为基准 ⇒ 落位位移 **162.5px**、**逆着上拍白箭头** ⇒ **拦下** ✓"
              "（旧口径用当刻圆心只算 56.2px ✗ 会放行 ✗ = 帧 37 那 138px 的跳 ✓）")
        _seen.clear()
        t2 = _LT(noise_tol=25.0)
        t2.pos = (250.9, 226.2)
        t2._report_prev = None                  # 拿不到 ⇒ 退回当刻 `pos` ✓
        t2._vel_rel_prev = None
        t2._vel_rel = (-172.4, -50.8)
        t2._merge_fix_safe((278.9, 201.3, 206.0, 200.0))
        check(_seen and _seen[-1] is not None
              and abs(_seen[-1][0] - 250.9) < 1e-6 and abs(_seen[-1][1] - 226.2) < 1e-6,
              "㉘ 拿不到上拍信息 ⇒ **退回当刻 `self.pos`**（实际 %s ✓ 第 0 拍 ✓ 不猜 ✗）"
              % ((None if not _seen else _seen[-1]),))
    finally:
        LieTracker._merge_edge_own = _orig


def test_kf_group_anchor():
    """⭐⭐⭐ **KF 青线要"刻在背景板上"**（用户 2026-10-03 ✓ 原话："**2. KF 青色线没有像白线一样
    利用群体速度『刻在背景板上』**" ✓）。

    做法（与白线 `_path_viz` **完全同一套** ✓）：后端每拍给两个量 ——
      · `kf.pos_grp` = KF 位置 **−** 当拍累计群体平移（**群体系**坐标 ✓ 加工域 ✓ `_path_cumT`
        是半域 ⇒ 后端已 ×2 ✓）；
      · `kf.cum2` = 当拍累计群体平移（加工域 ✓）。
    ⇒ 前端把每个历史点写成 `pos_grp + cum2(当拍)` ⇒ 整条线**跟着假目标群一起动** ✓（不再跟着相机 ✗）。

    钉**恒等式**：`pos_grp + cum2 == pos`（逐帧 ✓ 合成台跑一段 ✓）⇒ 前端"加回当拍平移"就还原 ✓。
    """
    bg, decoys, tgt = _world()
    _tr = _LT(show_kf=True)
    _n = _bad = 0
    for i in range(_N):
        frame, _truth = _frame(i, bg, decoys, tgt)
        _o = _tr.process(frame, ts=i * 0.14)
        _k = _o.get("kf") or {}
        if _k.get("pos") is None or _k.get("pos_grp") is None:
            continue
        _n += 1
        if (abs(_k["pos_grp"][0] + _k["cum2"][0] - _k["pos"][0]) > 1e-6
                or abs(_k["pos_grp"][1] + _k["cum2"][1] - _k["pos"][1]) > 1e-6):
            _bad += 1
    check(_n > 0 and _bad == 0,
          "㉗ **群体系锚定恒等式成立**（`pos_grp + cum2 == pos` ✓ %d 帧全过 ✓ —— 前端按白线同一套"
          "加回当拍平移 ⇒ 青线**跟着假目标群动** ✓ 用户 2026-10-03 ✓）" % _n)


def test_rel_vel_cap():
    """⭐⭐⭐ **白箭头（自身速度 `_vel_rel`）限幅**（用户 2026-10-03 ✓ 原话："**a. 给白箭头加限幅
    （治本）**" ✓ 见 `LieTracker._rel_cap` ✓）：上限 = `_REL_V_MAX`（**400 px/s** ✓ 既有口径：
    "自身分量只该是目标相对群体的运动（小量）" ✓）。

    钉三件：
      ① `_rel_cap` 是**纯函数**：超限 ⇒ **只压模长、不改方向** ✓；不超限 ⇒ **原样返回** ✓；
      ② **实战路径都过它**：合成台跑完全程 ⇒ 逐帧 `|_vel_rel| ≤ 上限` ✓；
      ③ 实测收益（本轮 A/B 的回归依据 ✓）：`10月2日.mp4` 白箭头最大模长 **590 → 322 px/s** ✓、
        融合 **10 → 23 帧** ✓（巨步不再把 B 型闸的允许量放大 ⇒ 不再误拦 ✓）；另两段素材**零差异** ✓。
    """
    import perception.lie_tracker as _LT3
    _lim = float(_LT3._REL_V_MAX)
    _t = _LT()
    _big = (3000.0, 4000.0)
    _c = _t._rel_cap(_big)
    check(abs((_c[0] ** 2 + _c[1] ** 2) ** 0.5 - _lim) < 1e-6,
          "㉕ 超限 ⇒ 压到 **%.0f** px/s（实际 %.0f ✓ —— 只压模长 ✓）"
          % (_lim, (_c[0] ** 2 + _c[1] ** 2) ** 0.5))
    check(abs(_c[0] / _c[1] - _big[0] / _big[1]) < 1e-9,
          "㉕ **方向不变**（比例 %.4f = 原比例 %.4f ✓）" % (_c[0] / _c[1], _big[0] / _big[1]))
    check(_t._rel_cap((10.0, -20.0)) == (10.0, -20.0),
          "㉕ 没超限 ⇒ **原样返回**（不动 ✓）")
    bg, decoys, tgt = _world()
    _tr = _LT()
    _vmax = 0.0
    for i in range(_N):
        frame, _truth = _frame(i, bg, decoys, tgt)
        _tr.process(frame, ts=i * 0.14)
        _vmax = max(_vmax, float((_tr._vel_rel[0] ** 2 + _tr._vel_rel[1] ** 2) ** 0.5))
    check(_vmax <= _lim + 1e-6,
          "㉕ 合成台跑完全程 ⇒ 白箭头最大模长 **%.0f** ≤ 上限 **%.0f** px/s ✓（实战路径都过它 ✓）"
          % (_vmax, _lim))


def test_edge_tangent_when_too_small():
    """⭐⭐⭐ **"框装不下圆"时落位仍与「真边」相切**（用户 2026-10-03 ✓ 原话："这种情况的
    **正确做法是保证归属（真目标）的边与圆相切**" ✓）。

    夹具取 `10月2日.mp4` **帧 18** 的真实数字：框 `(372.2,261.8)` **266.2×132.3**、半径
    `_tbox_rad = 75.7` ⇒ **框高 132 < 直径 151** ⇒ 老算法（走 `_tbox_inner`）在 y 维**整维不缩**
    ⇒ 圆心被放到**框的上边** `195.65` ✗（圆从上方探出 75.7px、上边**没相切** ✗✗ = 用户点出的）。

    ⇒ 新口径 = **坐标直接从「框边 ± 半径」算** ✓：
      · 左真 / 上真 ⇒ `x = 框左 + r`、`y = 框上 + r` ⇒ **(314.8, 271.4)** ✓（与真边相切 ✓）；
      · 归**砖**的边（右 / 下）**不设限** ✓（圆该探出就探出 ✓）。
    另附**对照**：框**装得下**时（高 ≥ 2r）两套算法**同解** ✓（相切点就是内接区边界 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    _orig_own = LieTracker._edge_ownership
    t = _LT(noise_tol=25.0)
    _e = Entry((429.2, 244.0, 154.0, 155.0), ts=0.0)
    _e.on_board = True
    _rg = ShapeRegistry()
    _rg.entries = [_e]
    t._reg = _rg
    t.pos = (314.8, 195.6)
    t._tbox_rad = 75.72867584228516
    t._rad = 63.94784678003755
    _box = (372.2, 261.8, 266.2, 132.3)
    try:
        # 左=真 / 右=砖 / 上=真 / 下=砖（帧 18 的实测归属 ✓ 直接钉住它 ✓ 不依赖砖的几何 ✓）
        LieTracker._edge_ownership = lambda self, b, detail=False: (
            ([False, True, False, True], [0.0] * 4) if detail
            else [False, True, False, True])
        # ⚠ 夹具用**宽阈值**（2026-10-03 ✓ 换成 IoU 口径后 ✓ 见别处那段说明 ✓）：否则
        #   `_merge_edge_own` 会因"不算融合"直接回 `None` ⇒ 本用例测不到相切 ✓
        t.merge_iou = 0.9
        t.brick_iou_sep = 0.9
        t.ring_cov_sep = 0.0
        _c = t._merge_edge_own(_box)
        _bl = 372.2 - 266.2 / 2.0
        _bt = 261.8 - 132.3 / 2.0
        check(_c is not None
              and abs(_c[0] - (_bl + t._tbox_rad)) < 0.1
              and abs(_c[1] - (_bt + t._tbox_rad)) < 0.1,
              "㉓ 框**装不下圆**（高 132 < 直径 151）⇒ 落位仍**与真边相切** = (%.1f, %.1f)"
              "（期望 %.1f/%.1f ✓ —— 老算法给 y = 框上边 %.1f ✗ 圆会从上方探出 75.7px ✗）"
              % (_c[0], _c[1], _bl + t._tbox_rad, _bt + t._tbox_rad, _bt))
        check(abs(_c[1] - _bt) > 1.0,
              "㉓ 且**不等于**「框上边」（%.1f ≠ %.1f ✓ 这就是改正之处 ✓ —— 用户 2026-10-03 ✓）"
              % (_c[1], _bt))
        # ⭐ 对照（**同一补丁下** ✓ 归属仍是「左真/右砖/上真/下砖」⇒ 只比"装得下/装不下" ✓）：
        #   框高 ≥ 2r ⇒ 相切点 = 内接安全区边界 ⇒ 与老算法**同解** ✓
        t._tbox_rad = 40.0
        _c2 = t._merge_edge_own((372.2, 261.8, 266.2, 300.0))
        _x0, _y0, _x1, _y1 = t._tbox_inner((372.2, 261.8, 266.2, 300.0))
        check(_c2 is not None and abs(_c2[0] - _x0) < 0.1 and abs(_c2[1] - _y0) < 0.1,
              "㉓ 对照：**装得下**时两套算法同解（(%.1f,%.1f) = 内接区左上 (%.1f,%.1f) ✓ —— 改动"
              "只影响「装不下」那种框 ✓）" % (_c2[0], _c2[1], _x0, _y0))
    finally:
        LieTracker._edge_ownership = _orig_own


def test_kf4_unit():
    """⭐⭐⭐ **卡尔曼模块本身的数学**（用户 2026-10-03 ✓ "做 KF影子模式" ✓ 内核见 `perception/kf.py` ✓）。

    钉四件：
      ① 未喂观测 ⇒ `pos` / `vel` 都是 `None` ✓（不猜 ✗）；
      ② **匀速直线**（真值 vx = 100 px/s、每拍 0.1s、观测带 σ=6px 的抖 ✓）⇒ KF **学出速度** ✓；
      ③ **平滑有效**：位置估计的平均误差 **小于**"单次观测"的（否则影子模式白做 ✓）；
      ④ `predict(0)`（dt ≤ 0 ✓）⇒ 状态不动、不崩 ✓（帧率/时间戳异常时同一取舍 ✓）。
    """
    import random
    from perception.kf import KF4

    _k0 = KF4()
    check(_k0.pos is None and _k0.vel is None,
          "㉒ 未喂观测 ⇒ `pos`/`vel` 都是 None（不猜 ✓）")
    _rnd = random.Random(7)
    _dt, _vx = 0.1, 100.0
    _k = KF4()
    _err_kf, _err_obs, _t = [], [], 0.0
    for i in range(30):
        _xt = 100.0 + _vx * _t
        _z = _xt + _rnd.gauss(0.0, 6.0)
        _k.step(_z, 0.0, None if i == 0 else _dt)
        if i >= 10:
            _err_kf.append(abs(float(_k.pos[0]) - _xt))
            _err_obs.append(abs(_z - _xt))
        _t += _dt
    _mk = sum(_err_kf) / len(_err_kf)
    _mo = sum(_err_obs) / len(_err_obs)
    check(_k.vel is not None and abs(float(_k.vel[0]) - _vx) < 15.0,
          "㉒ 匀速直线 ⇒ KF 学出速度 **%.1f** px/s（真值 %.0f px/s ✓ 见 `perception/kf.py` ✓）"
          % (float(_k.vel[0]), _vx))
    check(_mk < _mo,
          "㉒ ⭐ **平滑有效**：位置估计平均误差 **%.2f** px < 单次观测 **%.2f** px ✓（这就是影子的意义 ✓）"
          % (_mk, _mo))
    _px = _k.pos
    _k.predict(0.0)
    check(_k.pos == _px, "㉒ `predict(0)` ⇒ 状态**不动**（`dt ≤ 0` ⇒ 不猜 ✗）")


def test_select_step_gate():
    """⭐⭐⭐ **「选框距离门」（B 型步数比例 ✓ 用户 2026-10-03 ✓ 原话："动手" ✓）**：**非融合**红框
    候选必须满足「框心到**预测落点**的距离 ≤ 本值 × **本拍该走多远**」✓ —— 治的是"分离后挑到远处
    的框 ⇒ 出口夹取一步把圆拽过去 ⇒ 白线剧裂折角"（实测 `9月30日(1).mp4` 帧 48 跳 **97.8px** ✓）。

    钉四件：
      ① 门内 ⇒ 照常当选 ✓；
      ② 门外 ⇒ **不认** ⇒ 判 `gone`（这一拍按预测走 ✓ 等它走近再认 ✓）；
      ③ ⭐ **帧率 / 播放倍速无关**（用户 2026-10-03 点名 ✓）：**偏差与步长同比例放大 ⇒ 结论不变** ✓；
      ④ `select_step_ratio = 0` ⇒ **关**（老行为 ✓ 一字不变 ✓）。
    """
    def _mk(ratio=3.0, rad=80.0):
        t = _LT(select_step_ratio=ratio)
        t.pos = (300.0, 300.0)
        t._rad = rad
        t._tbox_rad = rad
        t._vel_rel = (100.0, 0.0)          # dt = 0.1 ⇒ 每拍走 10px ✓（门限 = 3 步 = 30px ✓）
        t._dt_last = 0.1
        return t

    _near = (315.0, 300.0, 120.0, 120.0)   # 距预测落点 (310,300) 仅 5px ⇒ 0.5 步 ✓
    _far = (400.0, 300.0, 120.0, 120.0)    # 距预测落点 90px ⇒ 9 步 ✗
    _s1, _b1 = _mk()._target_box_state([_near], {})
    check(_s1 == "ok" and _b1 is not None,
          "⑲ 门内（0.5 步）⇒ 照常当选 ✓（实际 %s ✓）" % _s1)
    _s2, _b2 = _mk()._target_box_state([_far], {})
    check(_s2 == "gone" and _b2 is None,
          "⑲ 门外（9 步 > 3 步）⇒ **不认** ⇒ 判 gone（这一拍按预测走 ✓ 实际 %s ✓）" % _s2)
    # ⭐ 帧率 / 播放倍速无关：步长与偏差**同比例**放大 ⇒ 步数不变 ⇒ 结论一样 ✓
    _t3 = _mk()
    _t3._vel_rel = (400.0, 0.0)            # 每拍走 40px（= 帧率减半 / 倍速播放 ✓）
    _near4 = (355.0, 300.0, 120.0, 120.0)  # 距新落点 (340,300) 15px ⇒ 仍是 0.375 步 ✓
    _s3, _ = _t3._target_box_state([_near4], {})
    check(_s3 == "ok",
          "⑲ ⭐ **帧率 / 倍速无关**：步长放大 4 倍、偏差也放大 ⇒ 结论不变（实际 %s ✓ 用户点名 ✓）"
          % _s3)
    _s4, _b4 = _mk(ratio=0.0)._target_box_state([_far], {})
    check(_s4 == "ok" and _b4 is not None,
          "⑲ `select_step_ratio = 0` ⇒ **关**（门外那格照旧当选 ✓ 老行为 ✓ 实际 %s ✓）" % _s4)


def test_merge_iou_threshold():
    """⭐⭐⭐ **「融合框判定 IoU」的阈值语义**（用户 2026-10-03 ✓ 原话："把『融合框判定阈值』换成
    『**融合框判定 IoU**』：**小于这个 IoU 才行**，之前配的 1.2 改成 **0.5**" ✓）。

    ⚠ 为什么要单独立这条 ✗：其余用例的夹具**要把阈值放宽到 0.9**（同心框 IoU≈0.65 ✓ 否则整条
    融合链的用例集体失效 ✗ 见 `test_...` 里那段说明 ✓）⇒ **阈值语义若没人专门钉，就被夹具盖住了** ✗。
    夹具 = 同心框 + 砖 ⇒ `IoU = 砖面积 / 框面积`（好算 ✓）：
      · **同格**：框 = 砖 140×130 ⇒ IoU **1.00**；· **不同格**：框 220×210 ⇒ IoU **≈0.39**。
    """
    from perception import geom
    from perception.lie_registry import Entry, ShapeRegistry

    def _mk(iou):
        # ⚠⚠ **要钉的是"分离期那一档"**（= 规则① 实际用的那个阈值 ✓）：2026-10-03 参数分档后
        #   规则① = `（圆外接矩形 ∩ 框）÷ 圆外接矩形面积 > ring_cov_本档` **且** `IoU(框,内砖)
        #   < brick_iou_本档`
        #   ✓ ⇒ 所以这里调的是 **`brick_iou_sep`**（不是 `merge_iou` ✗ —— 那个现在只管
        #   「这格是不是砖自己」那一档 ✓ 见 `_brick_owned` ✓）；`ring_cov_sep=0.0` ⇒ 前半条
        #   恒过 ✓ ⇒ 本用例测的才是"内砖那一半"的阈值语义 ✓。
        t = LieTracker(noise_tol=0.0, brick_iou_sep=iou, ring_cov_sep=0.0)
        t._rad = 90.0                    # ⚠ 半径设大 ⇒ **整圈绿圈装不进框** ⇒ 规则② 不成立 ✓
        t._tbox_rad = 90.0
        t.pos = (300.0, 200.0)
        t._vel_rel = (100.0, 0.0)
        e = Entry((300.0, 200.0, 140.0, 130.0), ts=0.0)
        e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        t._was_merged = False
        return t

    _same = (300.0, 200.0, 140.0, 130.0)         # 与砖一模一样 ⇒ IoU = 1.00
    _big = (300.0, 200.0, 220.0, 210.0)          # 框大得多 ⇒ IoU ≈ 0.39
    _mid = (300.0, 200.0, 200.0, 140.0)          # 框略大 ⇒ IoU ≈ 0.65（**卡在 0.5 与 0.7 之间** ✓）
    _t = _mk(0.5)
    _iou_same = geom.iou(_same, (300.0, 200.0, 140.0, 130.0))
    _iou_big = geom.iou(_big, (300.0, 200.0, 140.0, 130.0))
    _iou_mid = geom.iou(_mid, (300.0, 200.0, 140.0, 130.0))
    check(_t._is_merged_box(_same) is False,
          "㉚ **同格**（IoU %.2f **≥** 阈值 0.50）⇒ **不算融合** ✓（那是「砖自己那一格」✗ "
          "两框基本同一格 ⇒ 不是并集 ✓）" % (_iou_same,))
    check(_t._is_merged_box(_big) is True,
          "㉚ **不同格**（IoU %.2f **<** 阈值 0.50）⇒ **算融合** ✓（用户原话「**小于这个 IoU "
          "才行**」✓）" % (_iou_big,))
    _t0 = _mk(0.0)
    check(_t0._is_merged_box(_big) is False,
          "㉚ 阈值 **0 = 关**（`IoU < 0` 永不成立 ⇒ 只剩规则② ✓ 用户口径里 0 是合法值 ✓）")
    # ⭐ **"阈值真在起作用"要用一个卡在中间的框**（`_mid` IoU ≈ 0.65 ✓）—— 拿"同格"（IoU 1.00）
    #   去验是不行的 ✗（`1.00 < 任何 ≤1 的阈值` 都为假 ⇒ 永远不融合 ⇒ 断言必然失败 ✗ 踩过 ✓）。
    check(_mk(0.7)._is_merged_box(_mid) is True,
          "㉚ **阈值 0.70 > IoU %.2f** ⇒ 融合 ✓" % (_iou_mid,))
    check(_mk(0.5)._is_merged_box(_mid) is False,
          "㉚ **阈值 0.50 < IoU %.2f** ⇒ 不融合 ✓（两条合起来 = **阈值真的在起作用** ✓）"
          % (_iou_mid,))
    _t5 = _mk(0.5)
    check(abs(float(_t5.merge_iou_out) - float(_t5.merge_iou)) < 1e-9,
          "㉚ **`merge_iou_out` 默认 = `merge_iou`** ⇒ 迟滞默认**关**（退回单阈值 ✓ 用户 ✓ "
          "实测「开」的因素材而异 ⇒ 默认关 ✓ 见 `_MERGE_IOU_OUT` ✓）")


def test_reg_first_frame_iou_gate():
    """⭐⭐⭐ **「首帧建表」也必须过「真目标框 IoU 门」**（用户 2026-10-03 ✓ 原话："**按③做，
    但是要保证 IoU 限制门能拦住**" ✓）。

    背景：那道门原来**只在"新建候选"那条路上判** ✗ ⇒ **首帧建表完全不走它** ✗✗ —— 一旦按③
    放宽尺寸门（1.30→1.35 ✓）与边缘门（开局头 2 拍豁免 ✓），与真目标叠在一起的框就会**直接从
    建表这条路混进表里** ✗。本用例把"拦住"这件事钉死：
      · **半径还没学到**（开局那几拍 ✓ `rad=0` ✓）⇒ 退化成"**圆心落在框内**"（比 IoU 更保守 ✓）；
      · **半径有了** ⇒ 按 **`IoU > 0.4`** 判 ✓（与新建门同一把尺 ✓）。
    """
    from perception.lie_registry import ShapeRegistry

    # ① `rad = 0`：盖住圆心的那格**不登记**；远处的同尺寸格**照登记**（尺寸门按③已放宽 ✓）
    rg = ShapeRegistry()
    rg.step([(300.0, 200.0, 160.0, 160.0),      # 圆心 (300,200) 就在它里面 ⇒ 盖章 ⇒ 拒 ✗
             (600.0, 200.0, 160.0, 160.0)],     # 离得远 ⇒ 收 ✓
            ts=0.0, origin=(300.0, 200.0), rad=0.0, wh=(750.0, 500.0))
    _xs = [round(float(e.x)) for e in rg.entries]
    check(600 in _xs and 300 not in _xs,
          "㉙ 首帧建表：**盖住圆心的那格被 IoU 门（半径 0 时退化成「圆心在框内」）挡住** ✓、"
          "远处的照收 ✓（实际进表 %s ✓ 用户 2026-10-03「要保证 IoU 门能拦住」✓）" % (_xs,))

    # ② 半径有了 ⇒ 按 `IoU > 0.4`：与真目标框几乎重合 ⇒ 拒；错开一半 ⇒ 收
    rg2 = ShapeRegistry()
    rg2.step([(300.0, 200.0, 160.0, 160.0),     # 与真目标框（300,200,160,160）IoU = 1.0 ⇒ 拒 ✗
             (500.0, 200.0, 160.0, 160.0)],     # 只搭一点 ⇒ IoU ≈ 0 ⇒ 收 ✓
            ts=0.0, origin=(300.0, 200.0), rad=80.0, wh=(750.0, 500.0))
    _xs2 = [round(float(e.x)) for e in rg2.entries]
    check(500 in _xs2 and 300 not in _xs2,
          "㉙ 首帧建表（半径 80 px）：**与真目标框 IoU=1.0 的那格被拒** ✓、错开的照收 ✓"
          "（实际进表 %s ✓ 同一把尺 `_TGT_IOU_MAX=0.4` ✓）" % (_xs2,))


def test_geom_three_rulers():
    """⭐⭐⭐ **三把重叠尺语义不同**（用户 2026-10-03 ✓ 原话："我才学到 IoU 这个算法，我认为我们
    很多判定都可以换成这个算法（例如『噪声容差』改成『**噪声最小 IoU**』）" ✓）。

    夹具：**大框 200×200 套着小框 100×100**（完全被包住 ✓ —— 就是"并集框套着砖"的形态 ✓）：

    | 量 | 值 | 语义 |
    |---|---|---|
    | `iou`（交集÷**并集**）| **0.25** | 对称："两者是不是同一格" ✓ 大套小时**天然小** ✓ |
    | `overlap_min`（交集÷**较小面积**）| **1.00** | "小框叠在大框上" ✓ 这正是"旧砖被重复往上叠"✓ |
    | `cover_ratio(大, 小)`（交集÷**后者**）| **1.00** | 单向："大框盖住小砖多少" ✓ **框多大不进分母** ✓ |

    ⇒ 一句话：**"同一格"用 IoU** ✓；**"盖住多少"要用单向的覆盖率** ✓（换 IoU 会把"框本来就
    比砖大"当成不合规 ⇒ 真并集框全被误判 ✗）；**"小框叠大砖"要用较小面积** ✓（IoU 偏小拦不住 ✗）。
    """
    from perception import geom

    _big = (300.0, 200.0, 200.0, 200.0)
    _small = (300.0, 200.0, 100.0, 100.0)
    check(abs(geom.iou(_big, _small) - 0.25) < 1e-9,
          "① IoU = 交集÷并集 = %.2f（大框套小框 ⇒ **天然小** ✓ 所以它能用来判『同一格』、"
          "不能用来判『盖住多少』✗）" % geom.iou(_big, _small))
    check(abs(geom.overlap_min(_big, _small) - 1.0) < 1e-9,
          "② 交集÷**较小面积** = %.2f（小框被包住 ⇒ 满 1.0 ✓ —— 这才是『旧砖被重复往上叠』"
          "要抓的形态 ✓ 用户 2026-10-02 选定 ✓）" % geom.overlap_min(_big, _small))
    check(abs(geom.cover_ratio(_big, _small) - 1.0) < 1e-9
          and abs(geom.cover_ratio(_small, _big) - 0.25) < 1e-9,
          "③ 覆盖率（交集÷**后者面积**）**有方向**：大盖小 %.2f ／ 小盖大 %.2f ✓"
          "（框多大不进分母 ✓ 正是『并集框盖住砖』要的口径 ✓）"
          % (geom.cover_ratio(_big, _small), geom.cover_ratio(_small, _big)))
    check(geom.iou(_big, None) == 0.0 and geom.cover_ratio(None, _small) == 0.0,
          "④ 空输入 ⇒ 0.0（不崩 ✓ 不猜 ✗）")


def test_iou_brick_leg():
    """⭐⭐⭐ **「缩成砖判定 · 最小 IoU」那条腿**（用户 2026-10-03 ✓ 新配置 ✓）：判"框缩回砖大小/
    位置"现在有**两条并列的尺** ✓ —— `四条边都在噪声容差(px) 内` **或** `IoU(框, 砖) ≥ 本值` ✓。

    夹具：砖 `140×130`、框 `156×146`（四周各外扩 **8px**）⇒ 边差 8px、**IoU ≈ 0.80** ✓。
      · `noise_tol=0`（严格）⇒ 像素腿**不过** ✓ ⇒ 全看 IoU 腿：
          `iou_brick=0.00` ⇒ **False**（关着 ✓ 老行为 ✓）；
          `iou_brick=0.70` ⇒ **True**（IoU 0.80 ≥ 0.70 ✓ 这把尺自己就能判 ✓）；
          `iou_brick=0.90` ⇒ **False**（0.80 < 0.90 ✓ 越接近 1 越严 ✓）。
      · `noise_tol=25`（默认量级）⇒ 像素腿自己就过 ⇒ 与 IoU 无关 ✓（两条**并列** ✓）。
    ⚠ **反向验证**：把 IoU 那条腿掐掉 ⇒ 第 ② 条会红 ✓。
    """
    from perception import geom
    from perception.lie_registry import Entry, ShapeRegistry

    def _mk(iou_brick, ntol):
        t = _LT(noise_tol=ntol, iou_brick=iou_brick, sep_ratio=1.5)
        e = Entry((300.0, 200.0, 140.0, 130.0), ts=0.0)
        e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        t.pos = (100.0, 200.0)               # 圆心在左
        t._vel_rel = (-100.0, 0.0)           # 白箭头朝左 ⇒ 框在**反方向** ✓（② 才成立 ✓）
        t._merge_area_series = [40000.0, 20000.0, 40000.0]   # 先缩后张 ⇒ ③ 成立 ✓
        return t, e

    _box = (300.0, 200.0, 156.0, 146.0)      # 砖四周各外扩 8px ✓
    _t0, _e0 = _mk(0.0, 0.0)
    check(_t0._box_is_brick_det(_box, _e0) is False,
          "① `noise_tol=0` + `iou_brick=0`（两条腿都关 ✗）⇒ **判不出「缩成砖」** ✓（老行为 ✓）")
    _t1, _e1 = _mk(0.7, 0.0)
    check(_t1._box_is_brick_det(_box, _e1) is True,
          "② `iou_brick=0.70` ⇒ **IoU 那条腿自己就能判**（框 156×146 vs 砖 140×130 的 IoU = "
          "**%.2f** ≥ 0.70 ✓ —— 与像素那条腿并列 ✓ 用户 2026-10-03 ✓）"
          % geom.iou(_box, (300.0, 200.0, 140.0, 130.0)))
    _t2, _e2 = _mk(0.9, 0.0)
    check(_t2._box_is_brick_det(_box, _e2) is False,
          "③ `iou_brick=0.90` ⇒ 不够（IoU 0.80 < 0.90 ✓）⇒ 判不出 ✓（越接近 1 越严 ✓）")
    _t3, _e3 = _mk(0.0, 25.0)
    check(_t3._box_is_brick_det(_box, _e3) is True,
          "④ `noise_tol=25` ⇒ **像素那条腿**自己就过（边差 8px ≤ 25 ✓ 与 IoU 关不关无关 ✓）"
          "⇒ 两条**并列** ✓")


def test_sep_retreat_simple():
    """⭐⭐⭐ **简化后的分离判定：融合第 2 拍起，每拍比一次「落点倒退」**（用户 2026-10-03 ✓
    流程图 ✓ 原话："假设上拍走了白箭头的指示到了本拍，对比**本拍的白箭头**和**本拍圆心指向
    融合框计算出的落点的位置的向量**" ✓；判据 = 后方 180° 扇形 **且** 长度 > 「允许倒退距离×半径」
    ⇒ **分离** ✓）。

    夹具（几何算好 ✓）：砖 `(330,200) 120×120`（已上板 ✓）在框的**右部**、框 `(250,200) 200×180`、
    圆心 `(300,200)`（框内 ✓）、`_tbox_rad=40`、`_rad=60`、白箭头 **(100,0)**（朝右 ✓）、
    `noise_tol=45` ⇒ 归砖的边 = **右/上/下**（距 40／30／30，全 ≤ 45 ✓；左边距 120 ✗ 不归 ✓）
    ⇒ `n=3` ⇒ 落点在
    **左内接边界** `(190,200)`（`_merge_edge_own` 贴"中间那条边"的对面 ✓）⇒ 位移 `(−110,0)`
    ⇒ 与白箭头夹角 180°（**后方** ✓）且长度 110 > 允许量 0 ⇒ **判分离** ✓。
    三条断言：
      ① **保持期内（第 2 拍起）** ⇒ ⚠ **2026-10-03「做①」后不再判分离** ✓（这一拍是**相切落位**
         ⇒ 豁免倒退闸 ✓ —— 老口径那句"落点倒退超限 ⇒ 判分离"**已不适用** ✓ 见下面那段结构性说明）；
      ② **首拍不判**（`_merge_hold=False` ⇒ 进门那拍 ✓）⇒ 照旧 `merged` ✓；
      ③ **拿不到落点 ⇒ 不判**（容差调 0 ⇒ `n=0` ⇒ `_merge_edge_own` 回 `None` ✓）⇒ 照旧 `merged` ✓。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _mk(hold, ntol=45.0):
        t = _LT(noise_tol=ntol, merge_iou=0.9, allow_back_ratio=0.0)
        _e = Entry((330.0, 200.0, 120.0, 120.0), ts=0.0)
        _e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [_e]
        t._reg = rg
        t.pos = (300.0, 200.0)
        t._rad = 60.0
        t._tbox_rad = 40.0
        t._vel_rel = (100.0, 0.0)                  # 白箭头朝右 ✓
        t._prev_tbox = None
        t._merge_hold = bool(hold)
        t._merge_hold_bid = tuple(_e.bid)
        return t, _e

    _box = (250.0, 200.0, 200.0, 180.0)
    _t1, _e1 = _mk(True)                           # 保持期内 = 第 2 拍起 ✓
    check(_t1._is_merged_box(_box) or _t1._hold_active(_box),
          "① 前提：这格框算**融合框**（`_is_merged_box`=%s ／ `_hold_active`=%s ✓）"
          % (_t1._is_merged_box(_box), _t1._hold_active(_box)))
    check(_t1._edge_ownership(_box) == [False, True, True, True],
          "① 前提：边归属 = 右/上/下 归砖（实际 %s ／ 边距 %s ✓）"
          % (_t1._edge_ownership(_box), _t1._edge_ownership(_box, detail=True)[1]))
    _p1 = _t1._merge_edge_own(_box)
    check(_p1 is not None and abs(float(_p1[0]) - 190.0) < 1.0,
          "① 前提：落点在**左内接边界**（实际 %s ✓ —— 三条边归砖 ⇒ 贴「中间那条边」的对面 ✓）"
          % (_p1,))
    # ⭐⭐⭐ **2026-10-03 口径（最终 ✓ 用户原话："A. 恢复 `_hover`：**相切豁免只用于『落位
    #   采纳』**，**倒退判据仍按老口径算**" ✓）**：这一拍的边归属是 `[左=真目标, 右/上/下=砖]`
    #   ⇒ **x 轴恰有一条真边** ⇒ 它是**相切落位** ✓ ⇒ 于是**两个用途分开**判：
    #     · **落位采纳**（`allow_tangent=True` 默认 ✓）⇒ **豁免** ✓（相切点本来就该在那 ✓
    #       = `10月2日.mp4` 帧 43 那类 ✓）；
    #     · **倒退判据**（`allow_tangent=False` ✓ 分离判定里那条 `_hover` ✓）⇒ **按老口径算** ✓
    #       ⇒ 位移落在后方扇形、110 > 允许 0 ⇒ **不安全 ⇒ 判分离** ✓。
    #   ⚠⚠ 为什么必须分 ✗✗：`_merge_edge_own` 五档里**"能算出落位点"的档全是相切** ✓
    #     ⇒ 若两个用途共用豁免 ⇒ `_hover` **恒 False** ✗ ⇒ "落点倒退超限 ⇒ 判分离"整条失效 ✗
    #     （实测 `10月1日.mp4` 帧 62 ✓ 用户问"缩框了，为何没分离" ✓）。
    check(_t1._is_tangent_fix(_box) is True,
          "① 前提：这是**相切落位**（左=真目标 ⇒ x 轴恰一条真边 ✓ `_is_tangent_fix`=%s ✓）"
          % (_t1._is_tangent_fix(_box),))
    check(_t1._merge_fix_safe(_box) is True,
          "① **落位采纳**处：相切 ⇒ **豁免** ✓（默认 `allow_tangent=True` ✓ —— 相切点本来"
          "就该在那 ✓ 用户 2026-10-03「做①」✓ = `10月2日.mp4` 帧 43 ✓）")
    check(_t1._merge_fix_safe(_box, allow_tangent=False) is False,
          "① ⭐ **倒退判据**（`allow_tangent=False` ✓）：**按老口径算** ⇒ 位移 (−110,0) 在后方"
          "扇形内、110 > 允许 0 ⇒ **不安全** ✓（用户 2026-10-03「A」✓）")
    _st1, _bx1 = _t1._target_box_state([_box], {})
    check(_st1 == "gone" and _bx1 is None and bool(_t1._split_now),
          "① ⭐ **恢复 `_hover` ⇒ 这一拍判分离**（实际 %s/%s ／ `_split_now`=%s ✓ —— 口径 ="
          "「相切豁免只管落位采纳」✓ 用户 2026-10-03「A」✓）" % (_st1, _bx1, _t1._split_now))
    check((_t1._split_why or {}).get("how") == "落点倒退超限",
          "① 分离原因写明「**落点倒退超限**」（实际 %r ✓）"
          % ((_t1._split_why or {}).get("how"),))
    _t2, _e2 = _mk(False)                          # 首拍（进门那一拍 ✓）
    _st2, _bx2 = _t2._target_box_state([_box], {})
    check(_st2 == "merged" and _bx2 is not None and not bool(_t2._split_now),
          "② **首拍不判**（图里「融合第二拍开始」✓）⇒ 照旧判融合（实际 %s ✓）" % (_st2,))
    _t3, _e3 = _mk(True, ntol=0.0)                 # 容差 0 ⇒ 一条边都不归砖 ⇒ 拿不到落点 ✓
    check(_t3._merge_edge_own(_box) is None,
          "③ 前提：容差 0 ⇒ `n=0` ⇒ **拿不到落点**（实际 %s ✓）" % (_t3._merge_edge_own(_box),))
    _st3, _bx3 = _t3._target_box_state([_box], {})
    check(_st3 == "merged" and not bool(_t3._split_now),
          "③ **拿不到落点 ⇒ 不判**（用户：「拿不到点按白箭头预测走下一步的判断」✓）⇒ 照旧融合"
          "（实际 %s ✓）" % (_st3,))


def test_merge_first_nearest_corner():
    """⭐⭐⭐ **融合框「生成」那一拍：屏蔽「就近内接」后 ⇒ 也按边归属**（用户 2026-10-03 ✓ 原话：
    "**先屏蔽『就近内接』逻辑，都按边归属算**" ✓ —— 用开关 `_MERGE_FIRST_NEAREST` ✓
    **默认 `False` = 屏蔽** ✓）。

    数字取用户点名的 `9月30日(1).mp4` **第 17 帧**（并用**他当前参数** `noise_tol=15`、
    `merge_ratio=1.33` ✓ 不许用代码默认 ✗）：
      · 框 `208.7×165.9`、`_tbox_rad=64.5` ⇒ 内接安全区 `x[251.4,331.0] y[279.3,316.1]`；
      · 边归属实测 `(L,T)=真` ⇒ **真边对角 = 右下角** `(331.0,316.1)` ✓（生成拍与保持拍**同解** ✓）；
      · ⚠ 老口径（开关 `True` ✓ 用户 2026-10-02："融合框生成时→就近内接" ✓）看"融合**前**圆心
        在框心的**右上**" ⇒ 落 **右上角** `(331.0,279.3)` —— 这条**保留**做"开关能回退"的钉子 ✓。
      · 实测对照（`10月1日.mp4` 帧 39）：老口径落 **左下**（圆心在框心左下 ✓），屏蔽后落
        **左上**（真边 = 左+上 ⇒ 对角 ✓）✓ —— 与日志里的边归属一致 ✓。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    t = _LT(noise_tol=15.0)             # ⚠ 用户配置的容差（不是默认 2 ✗）
    t.merge_iou = 0.9                          # ⚠ 夹具用宽阈值（同心框 IoU≈0.65 ⇒ 0.9 才进 ✓）
    _e = Entry((270.0, 285.0, 170.0, 140.0), ts=0.0)   # 摆一块砖 ⇒ 让 L/T 两条边归砖 ✓
    _e.on_board = True
    _rg = ShapeRegistry()
    _rg.entries = [_e]
    t._reg = _rg
    t._tbox_rad = 64.5
    _rad_save = t._rad
    t._rad = 56.2
    t.pos = (330.0, 260.0)                     # 在框内 ⇒ `_is_merged_box` 的规则① 成立 ✓
    _box = (291.2, 297.7, 208.7, 165.9)
    check(t._is_merged_box(_box), "⑯ 前提：这一帧判成融合（面积比 ≥ 阈值 + 圆心在框内 ✓）")
    check(t._edge_ownership(_box)[0] and t._edge_ownership(_box)[2],
          "⑯ 前提：边归属 (L,T) 归砖（⇒ 老逻辑会落到「真边对角」= 右下角 ✗ 就是那一帧 ✓）")
    t._pos_prev = (400.1, 260.2)               # 融合**前**圆心（实测值 ✓ 在框中心右上 ✓）
    t._merge_first = True
    _c = t._merge_edge_own(_box)
    check(_c is not None and abs(_c[0] - 331.0) < 0.5 and abs(_c[1] - 316.1) < 0.5,
          "⑯ ⭐ **生成那一拍也按边归属**（真边 = 右+下 ⇒ 对角 = **右下角** (331.0,316.1) ✓ "
          "实际 %s ✓ —— 用户 2026-10-03「都按边归属算」✓）" % (_c,))
    # ⭐ **开关能回退**：`_MERGE_FIRST_NEAREST = True` ⇒ 老口径「就近内接」（用户 2026-10-02 ✓）
    import perception.lie_tracker as _LTN
    _nf = _LTN._MERGE_FIRST_NEAREST
    try:
        _LTN._MERGE_FIRST_NEAREST = True
        _c_old = t._merge_edge_own(_box)
        check(_c_old is not None and abs(_c_old[0] - 331.0) < 0.5
              and abs(_c_old[1] - 279.3) < 0.5,
              "⑯ 开关拨回 `True` ⇒ 老口径「**就近内接**」（融合前圆心在框心右上 ⇒ 右上角 "
              "(331.0,279.3) ✓ 实际 %s ✓ —— **可回退** ✓）" % (_c_old,))
    finally:
        _LTN._MERGE_FIRST_NEAREST = _nf
    t._merge_first = False
    _c2 = t._merge_edge_own(_box)
    check(_c2 is not None and abs(_c2[0] - 331.0) < 0.5 and abs(_c2[1] - 316.1) < 0.5,
          "⑯ **保持期照旧边归属**（右下角 (331.0,316.1) ✓ 实际 %s ✓）" % (_c2,))
    t._merge_first = True
    t._pos_prev = None
    t.pos = (200.0, 350.0)                     # 参考点挪到框心**左下** ⇒ 屏蔽后它**不该再参与** ✓
    _c3 = t._merge_edge_own(_box)
    check(_c3 is not None and abs(_c3[0] - 331.0) < 0.5 and abs(_c3[1] - 316.1) < 0.5,
          "⑯ ⭐ 屏蔽后**参考点不再参与**（`_pos_prev` 置空 / 圆心挪到左下 ⇒ 仍落右下角 "
          "(331.0,316.1) ✓ 实际 %s ✓）" % (_c3,))
    t.pos = None
    _c4 = t._merge_edge_own(_box)
    # ⚠ 2026-10-03 行为变化（如实钉住 ✓）：`_merge_edge_own` 现在**不再依赖 `self.pos`** ✓ ——
    #   它的入口只看"是不是融合态"（`_is_merged_box` / `_hold_active` ✓），落位坐标由**边归属**
    #   定 ✓。本档是 `n=2`（真边对角 ⇒ **两个坐标都有出处** ✓）⇒ 位置拿不到也照样落 ✓。
    #   "不猜"仍然成立 ✓：边归属**定不下来**时（`n=0` / `n=4` / 两条假边正对面 ✓）一律 `None`
    #   ✓（见本用例上面那几条 ✓）。
    check(_c4 is not None and abs(_c4[0] - 331.0) < 0.5 and abs(_c4[1] - 316.1) < 0.5,
          "⑯ ⭐ `pos` 拿不到 ⇒ **照旧落位**（n=2 对角 ⇒ 坐标**全部由边归属定** ✓ 不需要位置 ✓ "
          "实际 %s）—— ⚠ 旧口径此处回 `None` ✗（那时先要位置定「框里那块内砖」）"
          % (None if _c4 is None else ("(%.1f,%.1f)" % (_c4[0], _c4[1])),))
    t._rad = _rad_save


def test_merge_log_payload():
    """⭐⭐ **融合框生成**的日志载荷（用户 2026-10-02 ✓ 原话："当**融合框生成时**添加日志，
    **显示相关的砖**以及**解释判定为融合框的计算数据**" ✓）。

    钉三件：
      ① 判定 `merged` 那拍把**诊断快照**留下来（参照砖身份 + 覆盖率 + 面积比 + 阈值 ✓）；
      ② `_merge_first`（进门那拍 ✓）`_out` 带出 `merge_log`（text 里有"融合框生成"、
         参照砖出生位、面积比、阈值、命中的规则 ✓）；
      ③ **非进门**那拍 `merge_log` 为 `None` ✓（融合保持期不重复刷 ✗）。

    数字用 `9月30日(1).mp4` 帧 17 那组实测值 ✓（并集框 `208.7×165.9` 包住砖(528,340)、
    登记 `160.6×134.4`、比 **1.604** ≥ 阈值 1.4、圆心 (359.7,254.8) 在框里 ⇒ 规则① ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    t = _LT(noise_tol=2.0)
    t._typ_area = 140.0 * 130.0
    t._rad = 56.0                              # 演示窗画的绿圈半径（规则②那把尺 ✓）
    t.pos = (359.7, 254.8)
    e = Entry((267.7, 297.6, 160.6, 134.4), ts=0.0)
    e.on_board = True
    rg = ShapeRegistry()
    rg.entries = [e]
    t._reg = rg
    _box = (291.2, 297.7, 208.7, 165.9)        # 帧 17 的真并集框 ✓（与 `t_merge_rule1_*` 同组数 ✓）
    _st, _bx = t._target_box_state([_box], {})
    check(_st == "merged", "⑭ 帧 17 那组数判成 merged（实际 %s ✓）" % _st)
    _w = getattr(t, "_merge_why_snap", None) or {}
    check(_w.get("rule") == "①" and abs(float(_w.get("ratio", 0.0)) - 1.604) < 0.01,
          "⑭ 诊断快照留下计算量（rule=%s ｜ 面积比 %.3f ✓ 期望规则①/1.604 ✓）"
          % (_w.get("rule"), float(_w.get("ratio", 0.0))))
    check(tuple(_w.get("ref_bid") or ()) == tuple(e.bid),
          "⑭ **相关的砖**留了下来（bid=%s ｜ 期望 %s ✓）" % (_w.get("ref_bid"), tuple(e.bid)))
    # ⚠ 2026-10-03 参数分档后：规则① 用的阈值是 **`brick_iou_sep`**（分离期那一档 ✓ 见
    #   `_brick_iou_thr` ✓）—— 不再是 `merge_iou`（那个只管"是不是砖自己"那一档 ✓）。
    check(abs(float(_w.get("thr", 0.0)) - float(t.brick_iou_sep)) < 1e-9,
          "⑭ 也记了当时用的阈值（%.2f ✓ = **分离期那一档** `brick_iou_sep` ✓）"
          % float(_w.get("thr", 0.0)))
    t._merge_first = True
    _m = t._out("track").get("merge_log")
    _txt = (_m or {}).get("text", "")
    check(bool(_m) and "融合框生成" in _txt and "规则①" in _txt,
          "⑭ `_out` 带出 `merge_log`（%s）" % _txt[:80])
    check(("IoU" in _txt) and ("阈值" in _txt) and ("参照砖" in _txt),
          "⑭ 日志给了**计算数据**（参照砖 / **IoU** / 阈值 ✓ —— 2026-10-03 判据换成 IoU 之后，"
          "文案里的「面积比」也一并换掉了 ✓）：%s" % _txt[:140])
    t._merge_first = False
    check(t._out("track").get("merge_log") is None,
          "⑭ **非「生成」那一拍不给日志**（融合保持期不重复刷 ✓）")

    # ⭐⭐⭐ **融合期"每拍"的边归属**（用户 2026-10-02 ✓ 原话："日志里，融合时期每拍打印一下
    #   **边的归属（砖或真目标）**" ✓）—— 与 `merge_log`（只在生成那拍）**不同**：它**每拍都给** ✓。
    t._merge_hold = True
    t._merge_hold_bid = tuple(e.bid)
    t._was_merged = True
    t._tbox = _box
    t._tbox_merged = True
    _el = t._out("track").get("edge_log")
    _et = (_el or {}).get("text", "")
    check(bool(_el) and "融合期边归属" in _et and "砖" in _et and "真目标" in _et,
          "⑳ 融合期**每拍**都给「边归属」日志（%s ✓）" % _et[:110])
    check(len(((_el or {}).get("data") or {}).get("own") or []) == 4,
          "⑳ 四条边**逐条**给出归属（左/右/上/下 各一个 bool ✓）：%s"
          % (((_el or {}).get("data") or {}).get("own"),))
    check("容差" in _et and "参照砖" in _et,
          "⑳ 句子里有判归属用的容差与参照砖（同一把尺 ✓）：%s" % _et[-60:])
    t._tbox_merged = False
    check(t._out("track").get("edge_log") is None,
          "⑳ **非融合期不给** ✓（预测态 / 普通框不刷这条 ✗）")


def test_split_log_payload():
    """⭐⭐ **融合框消失（分离）**的日志载荷（用户 2026-10-02 ✓ 原话："把融合框消失（分离）
    的日志**也依据融合的格式**打印出来" ✓）。

    钉四件：
      ① 判出分离的那一支把**诊断**留下来（`_split_why`：参照砖身份 + 覆盖率 + 面积比 +
         **四边差 vs 噪声容差** + **因哪条判出去** ✓）；
      ② `_merge_split`（上一拍融合、这一拍不是 ✓）那一拍 `_out` 带出 `split_log` ✓，`text`
         与融合日志**同格式**（"融合框消失（分离）｜ 框 ｜ 参照砖 ｜ 覆盖率 ｜ 面积比 ｜
         四边差 ｜ 原因" ✓）；
      ③ **非分离**那拍 `split_log` 为 `None` ✓（保持期不重复刷 ✗ 与融合日志同一规矩 ✓）；
      ④ 没有框那一支（`how="检出框丢了"`）**不出句子错**（框写"—"、四边差写"—" ✓）。

    数值用 `9月30日(1).mp4` **第 45 帧**那组实测值 ✓：旧检出框 `168.9×154.9 @ (219.6,155.1)`
    已缩到参照砖 `160.2×128.8 @ (219.6,155.1)` 的 ±15px 之内 ⇒ `_box_is_brick_det=True` ✓
    （就是那条"分离信号" ✓ 也是"换对象"那一拍的起因 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    t = _LT(noise_tol=15.0)
    t.pos = (218.7, 178.3)
    t._rad = 56.0
    # ⚠ **白箭头方向**（用户 2026-10-02 ✓ 分离信号最新定稿要两条同时成立："框缩成砖 &&
    #   **框中心在轨迹预测方向的反方向**" ✓）⇒ 那一帧的实测速度 `(256.0, −66.7)` 里
    #   `vy < 0`（朝上）而框也在圆心**上方** ⇒ 点积 **+1777 > 0 = 预测前方** ⇒ 按新判据
    #   **不判分离** ✗（那正是这次优化要改掉的情形 ✓）。所以这里把 `vy` 翻成正的 ✓
    #   （= 目标朝下走、砖留在上方 = **反方向** ✓ 框/砖/容差全部保持实测那组数不变 ✓）。
    t._vel_rel = (256.0, 66.7)
    e = Entry((219.5679, 155.0774, 160.2399, 128.8228), ts=0.0)
    e.on_board = True
    rg = ShapeRegistry()
    rg.entries = [e]
    t._reg = rg
    t._merge_hold = True
    t._merge_hold_bid = tuple(e.bid)
    t._was_merged = True
    _box = (219.6, 155.1, 168.9, 154.9)
    # ⭐ 分离信号**三条**都要成立（用户 2026-10-02 ✓；⭐⭐ 2026-10-03 第三条口径改成
    #   "**本拍之前的最大 ≥ 本拍面积 × 阈值**" ✓）：按**拍序**给一串面积 ✓
    #   （本拍之前的最大 40000 ≥ 本拍面积 × 1.3 ✓）。
    t._merge_area_series = [40000.0, 20000.0, 40000.0]
    t._merge_area_max, t._merge_area_min = 40000.0, 20000.0
    check(t._box_covers_entry(_box, e) and t._box_is_brick_det(_box, e),
          "⑮ 帧 45 那组数：框仍包住参照砖 ✓、已缩成砖（±15px 容差内）✓、框心在预测反方向"
          "（点积 %.0f < 0 ✓）、融合期面积缩过（本拍之前的最大 40000 ≥ 本拍 %.0f × 1.3 ✓）"
          "⇒ **三条都成立**、判分离的信号成立 ✓"
          % (t._pred_dot(_box), float(_box[2]) * float(_box[3])))
    # ⚠ 日志载荷**直接构造**（不依赖 `_target_box_state` 走到哪条分支 ✗）：详见下面那条
    #   "前方 ⇒ 不判分离"的说明 —— 在新规则下"框已缩成砖"这条**只能**由保持分支产生，
    #   而保持分支里的框必然通过了选框那条"预测前方"闸 ⇒ 这里只钉**载荷与句子** ✓。
    t._split_why = t._split_diag(_box, e, "框已缩成砖")
    _w = getattr(t, "_split_why", None) or {}
    check(_w.get("how") == "框已缩成砖",
          "⑮ 分离诊断记下了**因哪条判出去**（how=%s ｜ 期望「框已缩成砖」✓）" % _w.get("how"))
    check(tuple(_w.get("ref_bid") or ()) == tuple(e.bid),
          "⑮ **相关的砖**留了下来（bid=%s ✓）" % (_w.get("ref_bid"),))
    check(_w.get("edges") is not None and len(_w["edges"]) == 4,
          "⑮ 记了**四边差**（左/右/上/下 = %s ✓ —— 与融合日志那条「命中规则」位置对应 ✓）"
          % (_w.get("edges"),))
    check(abs(float(_w.get("ntol", 0.0)) - 15.0) < 1e-9,
          "⑮ 记了当时用的**噪声容差**（%.1f px ✓ 判分离那把尺 ✓）" % float(_w.get("ntol", 0.0)))
    # ② 分离那一拍 `_out` 带出 payload（与融合日志同形状 ✓）
    t._merge_split = True
    _s = t._out("track").get("split_log")
    _txt = (_s or {}).get("text", "")
    check(bool(_s) and "融合框消失（分离）" in _txt,
          "⑮ `_out` 带出 `split_log`（%s）" % _txt[:90])
    for _k in ("参照砖", "覆盖率", "面积比", "四边差"):
        check(_k in _txt, "⑮ 分离日志与融合日志**同格式**：含「%s」（%s）" % (_k, _txt[:160]))
    check("框已缩成砖" in _txt, "⑮ 原因也写进句子（%s）" % _txt[-40:])
    # ⚠ 文案按**实际值**给结论（2026-10-02 修正 ✓ 原来写死"<0 = 反方向 ✓"⇒ 值是正的时候
    #   也挂着 ✓ ⇒ 像是"这条满足了" ✗ 实测撞到 ✓）⇒ 断言只认"反方向 / 顺向"这两个词 ✓。
    check("反方向" in _txt or "顺向" in _txt,
          "⑮ 分离日志带上**第二条判据**的方向结论（反方向 / 顺向 ✓ 用户 2026-10-02 ✓）：%s"
          % _txt[-70:])
    # ⭐⭐⭐ **新判据的直接证据**（用户 2026-10-02 ✓ 原话："优化分离信号：框缩成砖 &&
    #   框中心在轨迹预测方向的反方向" ✓）：**同一组框与砖**，只把白箭头翻向 ⇒ 框心落到
    #   预测前方 ⇒ **不再判分离** ✓（保持继续跟 ✓）。
    t._merge_hold = True
    t._merge_hold_bid = tuple(e.bid)
    t._was_merged = True
    t._split_why = None
    t._vel_rel = (-256.0, -66.7)         # 白箭头朝上 ⇒ 框心（在圆心上方的砖）落回"预测前方" ✗
    t._merge_area_series = []            # ⚠ 隔离"面积跨度"那条（本断言只测**落点**那条 ✓）
    t._target_box_state([_box], {})
    check(not bool(getattr(t, "_split_why", None)),
          "⑮ **框缩成砖但落在预测前方 ⇒ 不判分离**（新判据第二条生效 ✓ 用户 2026-10-02 ✓；"
          "实际 split_why=%s ✓）" % (getattr(t, "_split_why", None),))
    # ⭐⭐⭐ **「分离判定阈值」那条（2026-10-03 用户流程图新增 ✓）**：同一拍、同一组框，
    #   只要**面积跨度**成立（历史最大 ≥ 本拍 × 阈值 ✓）⇒ **照样判分离** ✓（与"落点"那条**并列 OR** ✓）。
    t._split_why = None
    t._merge_hold = True
    t._merge_hold_bid = tuple(e.bid)
    t._merge_area_series = [40000.0, 10000.0, 50000.0]   # 先缩（10000 ✓）后张（50000 ✓）
    t._vel_rel = (0.0, 0.0)              # 落点那条判不出（白箭头 0 ⇒ 放行 ✓）⇒ 只剩面积那条 ✓
    t._target_box_state([_box], {})
    check((t._split_why or {}).get("how") == "面积跨度超阈值" and t._split_now is True,
          "⑮ ⭐ **面积跨度超阈值 ⇒ 判分离**（实际 how=%r ／ `_split_now`=%s ✓ 用户 2026-10-03 "
          "流程图新增的那条 ✓ 与落点那条并列 ✓）"
          % ((t._split_why or {}).get("how"), getattr(t, "_split_now", None)))
    t._vel_rel = (256.0, 66.7)           # 还原成"反方向"（后面还要用 ✓）
    check(t._box_covers_entry(_box, e) and not t._box_ahead_of_pred(_box),
          "⑮ 对照：同一组数在**反方向**时两条都成立（框仍包住砖 ✓ 且框心在预测反方向 ✓）")
    # ③ 非分离那拍不给（保持期不重复刷 ✓）
    t._merge_split = False
    check(t._out("track").get("split_log") is None,
          "⑮ **非分离那一拍不给日志**（保持期不重复刷 ✓ 与融合日志同一规矩 ✓）")
    # ④ 没有框那一支也不出错（框与四边差都写"—" ✓）
    t._merge_hold = True
    t._merge_hold_bid = tuple(e.bid)
    t._was_merged = True
    t._split_why = None
    t._target_box_state(None, {})
    t._merge_split = True
    _s2 = t._out("track").get("split_log") or {}
    check("检出框丢了" in _s2.get("text", "") and "—" in _s2.get("text", ""),
          "⑮ 没有框那一支照样成句（含「检出框丢了」+「—」✓）：%s" % _s2.get("text", "")[:120])


def test_sep_ratio_gate():
    """⭐⭐⭐ **分离信号第三条：融合期框面积"真的缩过"**（用户 2026-10-02 ✓ 原话："**框缩成砖
    && 框中心在轨迹预测方向的反方向 && 融合期间融合框的最大面积 >= 融合期间融合框的最小
    面积 * x**"，"这个 x 做成配置参数『**分离判定阈值**』加在『融合框判定阈值』右边" ✓）。

    钉五件：
      ① `sep_ratio` 默认 = `_SEP_RATIO`（1.3 ✓），可透传；
      ② `_merge_area_span_ok`：**没记账 ⇒ False**（保守 ✓）；`max ≥ min × x` 才 True；
      ③ `_box_is_brick_det` **三条同时成立才真** —— 前两条成立、第三条不成立 ⇒ **False** ✓
         （这正是本条要挡的"假分离" ✓）；
      ④ 进门那拍**重置**、保持期每拍**更新**（行为：跑两拍看 max/min 有没有跟着变 ✓）；
      ⑤ 分离日志里带上 max/min 与阈值（与融合日志同一格式 ✓）。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    import perception.lie_tracker as LT
    check(abs(float(LT._SEP_RATIO) - 1.3) < 1e-9,
          "⑱ 「分离判定阈值」模块默认 = %.2f（界面初值同一处口径 ✓）" % float(LT._SEP_RATIO))
    t = _LT(sep_ratio=1.5)
    check(abs(float(t.sep_ratio) - 1.5) < 1e-9,
          "⑱ 可透传（sep_ratio=1.5 ⇒ %.2f ✓）" % float(t.sep_ratio))
    # ② 记账与判据（⭐⭐ 2026-10-03 口径：**本拍之前的最大 ≥ 本拍面积 × x** ✓ 用户原话：
    #   "把 ③ 改成「本拍之前的最大 ≥ 本拍面积 × 分离判定阈值」" ✓）
    check(t._merge_area_span_ok() is False,
          "⑱ **没记账 / 不传本拍面积 ⇒ 不成立**（判不出 ⇒ 不判分离 ✓ 保守 ✓ 不猜 ✗）")
    check(t._merge_area_span_ok(20000.0) is False,
          "⑱ 空序列 + 本拍 20000 ⇒ 不成立 ✓")
    # ⭐⭐⭐ **口径（用户 2026-10-03 确认 ✓）**：判据 = 「**历史最小面积（不含本拍）出现之后的
    #   历史最大面积**」÷ **本拍面积** > 阈值 ✓（**纯时序**："先缩到最小、之后又张开" ✓）。
    t._merge_area_series = [40000.0, 20000.0]          # ⚠ 最小 20000 在**末位** ✓
    check(t._merge_area_span_ok(20000.0) is True,
          "⑱ ⭐ **「已缩到过最小之后」为空 ⇒ 退用整体历史最大**（40000 ≥ 20000 × 1.5 = 30000 ⇒ "
          "**成立** ✓ 用户 2026-10-03 ✓ 原话：『不是必须已缩到过最小、且当前又张开，而是**取已"
          "缩到过最小后的最大**』 ✓ ＋ 实测 `10月1日.mp4` 帧 62：**一路缩到砖**（32130→25539）"
          "若不认这种形态 ⇒ 永远判不出来 ⇒ 用户问『缩框了，为何没分离』✓）")
    t._merge_area_series = [40000.0, 20000.0, 40000.0]  # 先大 → 缩到最小 → 又张开 ✓
    check(t._merge_area_span_ok(20000.0) is True,
          "⑱ ⭐ **先缩后张 ⇒ 成立** ✓（最小 20000 之后的**最大 40000** ≥ 本拍 20000 × 1.5 = 30000 ✓）")
    check(t._merge_area_span_ok(30000.0) is False,
          "⑱ 本拍 30000：最小之后的 40000 < 30000 × 1.5 = 45000 ⇒ **不成立** ✓（缩得不够 ✗）")
    t._merge_area_series = [20000.0, 20000.0]          # 全程没缩过（框一直这么大）
    check(t._merge_area_span_ok(20000.0) is False,
          "⑱ ⭐ **全程没缩过 ⇒ 不成立** ✓（最小之后的最大 20000 < 20000 × 1.5 ⇒ 挡住「一开始就"
          "量错」那种假分离 ✓ —— 这正是这条要防的 ✓）")
    t._merge_area_series = [50000.0, 20000.0]          # 一路缩到砖、**没再张开** ✓
    check(t._merge_area_span_ok(20000.0) is True,
          "⑱ ⭐ **一路缩到砖（先大后小、没涨回来）⇒ 现在**成立** ✓（整体最大 50000 ≥ 30000 ✓ "
          "—— 2026-10-03 改口径：这是**最典型的分离形态**，必须判得出来 ✓；旧口径下它恒 False ✗"
          "⇒ `10月1日.mp4` 帧 62/63/64 全部漏判 ✗ = 用户点名的那个『缩框了，为何没分离』✓）")
    t._merge_area_series = [50000.0, 20000.0, 50000.0]
    _info = t._merge_area_span_info(20000.0)
    check(abs(_info[0] - 50000.0) < 1e-9 and _info[1] == 2 and abs(_info[2] - 20000.0) < 1e-9
          and abs(_info[3] - 2.5) < 1e-9,
          "⑱ 诊断量 = **(最小之后的最大, 它出现在第几拍, 本拍面积, 比值)** = %s ✓（日志写这一组 ✓）"
          % (_info,))
    t._merge_area_series = []
    t._merge_area_note((0.0, 0.0, 200.0, 100.0))         # 面积 20000 ✓ 追加 ✓
    check(t._merge_area_series == [20000.0] and t._merge_area_min == 20000.0,
          "⑱ 记账 = **按拍序追加**（序列 %s ✓）+ 同步全域极值（min=%s ✓ 只给日志 ✓）"
          % (t._merge_area_series, t._merge_area_min))
    # ③ 三条同时成立才判分离
    t.pos = (100.0, 300.0)
    t._vel_rel = (0.0, 100.0)            # 白箭头朝下
    e = Entry((100.0, 200.0, 160.0, 130.0), ts=0.0)
    e.on_board = True
    rg = ShapeRegistry()
    rg.entries = [e]
    t._reg = rg
    _box = (100.0, 200.0, 161.0, 131.0)  # ① 缩成砖（±15px 容差内 ✓）
    check(t._box_ahead_of_pred(_box) is False,
          "⑱ ② 框心在预测**反方向**（框在上、箭头朝下 ✓ 点积 = %.0f < 0 ✓）"
          % t._pred_dot(_box))
    # ③ 第三条（**新口径** 2026-10-03 ✓）：本拍之前的最大 ≥ 本拍面积 × 阈值
    #   （`_box` 面积 = 161×131 = 21091 ⇒ 阈值 1.5 ⇒ 需 ≥ 31637 ✓）
    t._merge_area_series = [40000.0, 20000.0, 40000.0]
    check(t._box_is_brick_det(_box, e) is True,
          "⑱ ①②③ 都成立 ⇒ 判分离 ✓（本拍之前的最大 40000 ≥ 31637 ✓）")
    t._merge_area_series = [40000.0, 20000.0, 40000.0]   # ⭐ **先缩后张**（最小 20000 之后又到 40000 ✓）
    check(t._box_is_brick_det(_box, e) is True,
          "⑱ ①②③ 都成立 ⇒ `_box_is_brick_det` 真 ✓（最小之后的最大 40000 ≥ 本拍 21091 × 1.5 = "
          "31637 ✓ 用户 2026-10-03 时序口径 ✓）")
    t._merge_area_series = [21000.0, 21000.0]   # 全程没缩过（一开始就量错那种 ✓）
    check(t._box_is_brick_det(_box, e) is False,
          "⑱ ⭐ **全程没缩过 ⇒ ③ 不成立 ⇒ 不判分离** ✓（最大 21000 < 21091 × 1.5 ⇒ 挡住假分离 ✓）")
    # ④ 进门重置 / 保持期更新（行为级 ✓）
    # ⚠⚠ `sep_ratio` 故意调**很大**（2026-10-03 ✓）：新流程里「面积跨度」是一条**独立的分离
    #   判据** ✓ ⇒ 阈值 1.3 那一拍（大框 62400 vs 小框 23800 ⇒ 比 2.62 ≥ 1.3 ✓）会**直接判分离**
    #   ✗ ⇒ 就测不到"保持期追加记账"了 ✓ ⇒ 调大以隔离 ✗。
    t2 = _LT(noise_tol=15.0, sep_ratio=10.0)
    t2.pos = (200.0, 200.0)
    t2._rad = 40.0
    t2._vel_rel = (0.0, 0.0)             # 先不掺方向 ✓（只验记账 ✓）
    e2 = Entry((200.0, 200.0, 160.0, 130.0), ts=0.0)
    e2.on_board = True
    rg2 = ShapeRegistry()
    rg2.entries = [e2]
    t2._reg = rg2
    t2._typ_area = 160.0 * 130.0
    _big = (200.0, 200.0, 260.0, 240.0)   # 并集（面积比 3.0 ≥ 1.4 ⇒ 进门 ✓）
    t2._target_box_state([_big], {})
    check(t2._merge_hold is True and t2._merge_area_series == [260.0 * 240.0],
          "⑱ **进门那拍重置**成「只有这个并集框」的序列（%s ✓）" % (t2._merge_area_series,))
    # ⚠ 这一拍必须**不满足 `_is_merged_box`**（否则又被当"新进门" ⇒ max/min 被重置 ✗）：
    #   面积比 170×140 / (160×130) = 1.14 < 阈值 1.4 ✓；白箭头速度 0 ⇒ `_box_ahead_of_pred`
    #   回 True ⇒ 第二条不成立 ⇒ 不走"分离"分支 ✓ ⇒ 落 `else`（保持 + 记账 ✓）。
    _shrink_box = (200.0, 180.0, 170.0, 140.0)   # 还包住砖（交集够 ✓）、比刚才小 ✓
    # ⭐⭐⭐ **"本拍不算"**（用户 2026-10-02 ✓ 原话："注意一下，**本拍不算**" ✓）—— 行为级
    #   区分：历史 = `[40000, 20000]`（最小 20000 在**最后一拍** ⇒ 它之后只有它自己 ⇒ 比
    #   1.00 < 1.50 ⇒ **不成立** ✓），而本拍那格面积 **48000**（很大 ✓）：
    #     · **本拍不算**（现行 ✓）⇒ 只看历史 ⇒ 比 1.00 ✗ ⇒ **不分手**、保持继续、本拍面积
    #       **判定之后才入账** ⇒ 序列变成 `[40000, 20000, 48000]` ✓；
    #     · 若**含本拍** ⇒ 最小 20000（第 2 拍）⇒ 之后最大 48000 ⇒ 比 2.40 ≥ 1.50 ✓ ⇒
    #       **会分手** ✗ ⇒ 所以这条断言正好抓"提前入账"的写法 ✓。
    t8 = _LT(noise_tol=15.0, sep_ratio=1.5, inherit_dist=3.0)
    # ⚠ 两处构造讲究（都踩过 ✓）：
    #   · 圆心放进框**内**一些（(100,280) ✓）：不然外接矩形只被压住一半 ⇒ S = 0.50 ⇒ 闸过不去 ✗；
    #   · 半径放大到 **150**（比框还大 ✓）：让**规则②"整圈绿圈在框内"不成立** ✗ ⇒ 该格
    #     **不算"进门"** ⇒ 不会走"进门分支"（那条会把面积序列**重置**掉 ✗ 于是根本测不到
    #     "本拍不算" ✗）；同时 `inherit_dist=3`（= 允许出框 3 倍半径 ⇒ **距离闸必过** ✓
    #     不受半径放大影响 ✓）。
    t8.pos = (100.0, 280.0)
    t8._rad = 150.0
    t8._tbox_rad = 150.0
    t8._vel_rel = (0.0, 100.0)                 # 白箭头朝下 ⇒ 框在圆心上方的砖 = **反方向** ✓
    e8 = Entry((100.0, 200.0, 260.0, 220.0), ts=0.0)   # 砖放大些：好让框既"包住砖"又面积够大 ✓
    e8.on_board = True
    rg8 = ShapeRegistry()
    rg8.entries = [e8]
    t8._reg = rg8
    t8._typ_area = 260.0 * 220.0
    t8._merge_hold = True
    t8._merge_hold_bid = tuple(e8.bid)
    t8._was_merged = True
    t8._merge_area_series = [40000.0, 20000.0]         # 历史（本拍**不算** ✓）
    _tb8 = (100.0, 200.0, 240.0, 200.0)                # 缩成砖 ✓（240×200 = 48000）
    check(t8._box_covers_entry(_tb8, e8),
          "⑱（前提）本拍那格**还包住参照砖** ✓（不然会走「框不再包住参照砖」那条 ✗ 就测不到第三条 ✗）")
    _st8, _bx8 = t8._target_box_state([_tb8], {})
    check(t8._split_now is False,
          "⑱ ⭐ **本拍不算** ⇒ 第三条只看**历史**（比 1.00 < 1.50 ✗）⇒ **不分手** ✓"
          "（若本拍提前入账 ⇒ 比 2.40 ≥ 1.50 ⇒ 反而会分手 ✗ 这条正是抓它的 ✓）")
    check(t8._merge_area_series == [40000.0, 20000.0, 48000.0],
          "⑱ 没分手 ⇒ **这一拍的面积现在才入账**（序列 %s ✓ ⇒ 下一拍它就是「历史」✓）"
          % (t8._merge_area_series,))

    t2._target_box_state([_shrink_box], {})
    check(t2._merge_area_series == [260.0 * 240.0, 170.0 * 140.0],
          "⑱ **保持期每拍追加**（序列 = 进门的大框 + 这一拍的小框 ✓ 实际 %s ✓）"
          % (t2._merge_area_series,))
    check(t2._merge_hold is True,
          "⑱ 它仍是「融合保持」（`_is_merged_box` 不成立、保持未被作废 ✓）")


def test_merge_fix_safe():
    """⭐⭐⭐ **"融合后绿圈圆心会不会往白箭头反方向走"**（用户 2026-10-02 ✓ 原话："……模拟将
    绿圆圈沿着白箭头的方向推进**依次接触**，**第一个在融合后绿圈圆心不会往白箭头反方向
    行进**的框" ✓）—— 方案 B：只模拟**落位**那一支 ✓（融合期只有它会逆着白箭头拽圆 ✓）。

    钉四件：
      ① 不落位（`_merge_edge_own` 回 `None`）⇒ **安全** ✓（位置按预测滑行 ⇒ 天然顺向 ✓）；
      ② 落位点在**前进侧**（点积 ≥ 0）⇒ 安全 ✓；
      ③ 落位点在**反方向**（点积 < 0）⇒ **不安全** ✓（闸会淘汰它、去试下一个 ✓）；
      ④ 方向拿不到（白箭头 ≈ 0）⇒ 安全（判不出 ⇒ 放行 ✓ 不猜 ✗）。

    判据本身与"落位函数"解耦 ⇒ 这里把 `_merge_edge_own` **替身**掉，直接验那条点积 ✓
    （行为链路由下面那条集成断言 + `test_merge_front_gate` 覆盖 ✓）。
    """
    _t = _LT()
    _t.pos = (300.0, 300.0)
    _t._vel_rel = (100.0, 0.0)                 # 白箭头朝右 ✓
    _orig = LieTracker._merge_edge_own
    try:
        LieTracker._merge_edge_own = lambda self, box, at=None: None
        check(_t._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is True,
              "① 不落位（回 None）⇒ **安全** ✓（融合期位置按预测滑走 ⇒ 不会反向 ✓）")
        LieTracker._merge_edge_own = lambda self, box, at=None: (400.0, 300.0)
        check(_t._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is True,
              "② 落位点在**前进侧**（(400,300)−(300,300) = (+100,0) ⇒ 点积 +10000 ≥ 0）⇒ 安全 ✓")
        LieTracker._merge_edge_own = lambda self, box, at=None: (200.0, 300.0)
        check(_t._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is False,
              "③ 落位点在**反方向**（点积 −10000 < 0）⇒ **不安全** ✓ ⇒ 闸淘汰它、依次去试下一个 ✓")
    finally:
        LieTracker._merge_edge_own = _orig
    _t._vel_rel = (0.0, 0.0)
    check(_t._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is True,
          "④ 白箭头 ≈ 0（判不出方向）⇒ 安全 ✓（放行 ✓ 不猜 ✗，与其它判据同一取舍 ✓）")

    # ⭐⭐⭐ **「融合框挑选允许倒退距离(绿圆半径比例)」**（用户 2026-10-02 ✓ 原话："『融合框挑选
    #   允许倒退距离(px)』改为『**融合框挑选允许倒退距离(绿圆半径比例)**』" ✓）：判据从"点积 ≥ 0"
    #   放宽成"**沿方向的位移分量** ≥ `−比例 × 绿圈半径`" ✓（旧 px 口径 × 半径归一 ✓）。
    _t2 = _LT(allow_back_ratio=0.0)
    _t2.pos = (300.0, 300.0)
    _t2._rad = 60.0                                   # ⚠ **B 型之后半径不再是那把尺** ✗（留着只为
                                                      #   证明"与它无关" ✓ 见下面第 ③ 段 ✓）
    _t2._vel_rel = (100.0, 0.0)                       # 白箭头朝右（速度 100 px/s ✓）
    _t2._dt_last = 0.1                                # ⇒ **本拍该走多远 = 10px**（B 型那把尺 ✓）
    _o2 = LieTracker._merge_edge_own
    try:
        # 落位点 (270,300) ⇒ 沿方向位移 = −30 px（**倒退 30px = 0.5 × 半径** ✓）
        LieTracker._merge_edge_own = lambda self, box, at=None: (270.0, 300.0)
        check(_t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is False,
              "⑤ 倒退 30px（0.5 半径）、允许 0 ⇒ **不安全** ✓（默认：一点也不能倒退 ✓）")
        _t2.allow_back_ratio = 3.0                    # ⭐ B 型：3 × 10px = **30px** ✓
        check(_t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is True,
              "⑤ 允许 **3 步** = 30px ⇒ **安全** ✓（正好卡在边界上 ✓ —— 尺 = 步数 ✓ 不再是半径 ✓）")
        _t2.allow_back_ratio = 2.9
        check(_t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is False,
              "⑤ 允许 2.9 步 = 29px < 倒退 30px ⇒ 不安全 ✓")
        # ⭐⭐ **B 型 = 与绿圈半径无关**（用户 2026-10-03 ✓ "顺便改成 B" ✓）：半径换多大都不影响判定 ✓
        #   （旧口径下这两个结果会**相反** ✗ —— 正是这次要改掉的病 ✓）
        _t2.allow_back_ratio = 3.0
        _t2._rad = 200.0
        _ok_big = _t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0))
        _t2._rad = 12.0
        _ok_small = _t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0))
        check(_ok_big is True and _ok_small is True,
              "⑤ ⭐ **B 型与绿圈半径无关**（半径 200 ／ 12 结果一样 = 放行 ✓ —— 旧「半径比例」口径"
              "下这两个必然相反 ✗）")
        # ⭐⭐ **帧率 / 播放倍速无关**（用户 2026-10-03 点名 ✓ 原话："② 选框距离门会因为视频播放
        #   速度而改变结果吧？" ✓）：**偏差与步长同比例放大 ⇒ 结论不变** ✓
        _t2._rad = 60.0
        _t2._dt_last = 0.2                             # 每拍走 20px（步长翻倍 = 帧率减半/倍速播放 ✓）
        LieTracker._merge_edge_own = lambda self, box, at=None: (240.0, 300.0)   # 倒退 60px
        check(_t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is True,
              "⑤ ⭐ **帧率 / 倍速无关**：每拍走 20px、倒退 60px（**同样是 3 步**）⇒ 与「每拍 10px、"
              "「倒退 30px」**同一个结论**（放行 ✓）")
        _t2._dt_last = 0.1
        LieTracker._merge_edge_own = lambda self, box, at=None: (270.0, 300.0)   # 复位（下面还要用 ✓）
    finally:
        LieTracker._merge_edge_own = _o2
    # ⭐⭐⭐ **作用范围**（用户 2026-10-02 确认 ✓）：它**只在"挑融合候选"那一刻**用
    #   （`_target_box_state` 的 `_first_ok` 循环 ✓ = 进门 / 换框 ✓）；**同一个融合框的
    #   保持期内一次都不过它** ✗ ⇒ 那一支照旧按 `_merge_edge_own` 落位 ✓ 不受本参数约束 ✓。
    def _scene(ab):
        t = _LT(noise_tol=2.0, inherit_dist=3.0, allow_back_ratio=ab)
        t._typ_area = 140.0 * 130.0
        t._rad = 60.0
        t._tbox_rad = 60.0
        t.pos = (200.0, 200.0)
        t._vel_rel = (-100.0, 0.0)
        from perception.lie_registry import Entry, ShapeRegistry
        e = Entry((300.0, 200.0, 140.0, 130.0), ts=0.0)
        e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        t._merge_hold = True                     # 前提：**已在融合保持中** ✓
        t._merge_hold_bid = tuple(e.bid)
        t._was_merged = True
        return t

    _box6 = (300.0, 200.0, 300.0, 280.0)
    _calls = []
    _orig6 = LieTracker._merge_fix_safe
    try:
        def _spy6(self, box, **kw):              # 只在被调用时记一笔 ✓（⚠ 要转发新关键字 ✗
            _calls.append(1)                     #   `allow_tangent` ✓ 用户 2026-10-03「A」✓）
            return _orig6(self, box, **kw)

        LieTracker._merge_fix_safe = _spy6
        # ⭐⭐ **2026-10-03 修订**（用户原话："安全闸判的是『落位会不会把圆心往白箭头的反方向
        #   拽』**这一条可以移除了**，我们现在用『融合框挑选允许倒退距离』判定" ✓）：
        #   **选框里已经不调它** ✗（判据挪到**落位那一刻** ✓ 见 `_step_track` 的 `_fc` 那段 ✓），
        #   而且**框照样当选** ✓ ⇒ 不会再出现"唯一的融合框被闸淘汰 ⇒ 判分离"✗✗。
        # ⭐⭐ **2026-10-03 口径（最终）**：它**不再是"选候选的闸"** ✗ ⇒ 非保持期（新会话）
        #   一次都不调 ✓；但**保持期内变成"融合第 2 拍起每拍判一次"的判据** ✓（用户流程图 ✓）
        #   ⇒ 那几拍**会调一次** ✓。
        _calls.clear()
        _sc0 = _scene(0.0)
        _sc0._merge_hold = False                 # 非保持期（新会话 ✓）
        _r0 = _sc0._target_box_state([_box6], {})
        check(len(_calls) == 0,
              "⑥ **非保持期：选框一次都不调它**（实际 %d 次 ✓ —— 它已经不是「选候选的闸」了 ✗）"
              % (len(_calls),))
        check(_r0[0] == "merged" and _r0[1] is not None,
              "⑥ 该格**照旧被挑中**（实际 %s ✓ —— 不再因为「落位预演不安全」就把它整个淘汰 ✗）"
              % (_r0,))
        _calls.clear()
        _scene(0.0)._target_box_state([_box6], {})           # `_merge_hold=True` ⇒ 保持期 ✓
        check(len(_calls) == 1,
              "⑥ ⭐ **保持期内每拍调它一次**（实际 %d 次 ✓ —— 这就是「**融合第 2 拍起：落点"
              "倒退超限 ⇒ 分离**」那条判据 ✓ 用户 2026-10-03 流程图 ✓）" % (len(_calls),))
    finally:
        LieTracker._merge_fix_safe = _orig6

    import perception.lie_tracker as _LT2
    check(LieTracker(allow_back_ratio=-5.0).allow_back_ratio == 0.0,
          "⑤ 负数无意义 ⇒ 夹到 0 ✓；`None` ⇒ 模块默认 %.2f（**步数** ✓ B 型：本拍该走多远的倍数 ✓"
          " 不再是「×半径」✓）" % float(_LT2._ALLOW_BACK_RATIO))
    # ⭐⭐⭐ **2026-10-03 口径修订**：判据从"只看沿白箭头的**分量**"改成"**后方 180° 扇形内
    #   一律按位移长度限制**"（用户原话："……改成 **后方 180 度扇形内的任意方向都限制**
    #   （而**不只是向后的分量**）" ✓）。
    _t2._rad = 60.0                                  # ⚠ B 型下半径不参与 ✓（留着只为证明无关 ✓）
    _t2._dt_last = 0.1                               # ⚠ **必须显式给**：本拍该走多远 = 100×0.1 = 10px ✓
    _t2.allow_back_ratio = 3.0                       # ⇒ 容差 = 3 步 = **30px**（下面那组断言沿用 ✓）
    _t2.pos = (300.0, 300.0)
    _t2._vel_rel = (100.0, 0.0)                      # 白箭头朝右 ✓
    _o3 = LieTracker._merge_edge_own
    try:
        # ① 正侧向（位移 (0,+40)）：夹角 90° ⇒ **前方半球** ⇒ 不限制 ✓
        LieTracker._merge_edge_own = lambda self, box, at=None: (300.0, 340.0)
        check(_t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is True,
              "⑤ 位移正侧向 (0,+40)：夹角 90° ⇒ 前方半球 ⇒ **不限制** ✓")
        # ② 后方扇形内 + 长度超限：位移 (−50,+50) ⇒ 长度 70.7 > 30 ⇒ 拦 ✓
        LieTracker._merge_edge_own = lambda self, box, at=None: (250.0, 350.0)
        check(_t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is False,
              "⑤ 位移 (−50,+50)：后方扇形内、**长度 70.7 > 30** ⇒ 拦 ✓")
        # ③ ⭐ **区分新旧的那一例**：位移 (−30,+30) ⇒ 沿白箭头的分量只有 **−30 = 允许量**
        #    （旧口径恰好**放行** ✗）但**实际退了 42.4px** ⇒ 新口径按长度 ⇒ **拦** ✓
        LieTracker._merge_edge_own = lambda self, box, at=None: (270.0, 330.0)
        check(_t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is False,
              "⑤ ⭐ 位移 (−30,+30)：**分量 −30 恰好等于允许量**（旧口径放行 ✗）但长度 42.4 > 30 "
              "⇒ 新口径**拦下** ✓（用户 2026-10-03：「不只是向后的分量」✓）")
        # ④ 后方扇形内、长度在限内 ⇒ 放行 ✓
        LieTracker._merge_edge_own = lambda self, box, at=None: (280.0, 310.0)
        check(_t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is True,
              "⑤ 位移 (−20,+10)：后方扇形内、长度 22.4 ≤ 30 ⇒ **放行** ✓")
        # ⑤ 纯正后（沿白箭头反方向）⇒ 长度就是倒退量 ⇒ 与旧口径同 ✓
        LieTracker._merge_edge_own = lambda self, box, at=None: (270.0, 300.0)
        check(_t2._merge_fix_safe((300.0, 300.0, 200.0, 200.0)) is True,
              "⑤ 位移 (−30,0)：纯正后、长度 30 ≤ 30 ⇒ 放行 ✓（边界 ✓ 与旧口径一致 ✓）")
    finally:
        LieTracker._merge_edge_own = _o3

    _t2._vel_rel = (0.0, 0.0)                        # 速度拿不到 ⇒ 基准也算不出 ✓
    _t2._vel = (0.0, 0.0)
    check(_t2._allow_back_px() == float(_LT2._BACK_FLOOR_PX),
          "⑤ ⭐ **B 型与半径无关**：半径清零也照样给容差 ✓；**基准拿不到**（速度≈0 ✓）⇒ 回**下限"
          " %.1f px**（固定量 ⇒ 与帧率无关 ✓ —— 旧口径这里是「半径没学到 ⇒ 0px」✗ 已废 ✓）"
          % float(_LT2._BACK_FLOOR_PX))


def test_merge_fix_all_unsafe():
    """⭐⭐⭐ **候选全都"不安全" ⇒ 一个都不放行 ⇒ 落到「门内没有能挑的格」⇒ 判分离**
    （用户 2026-10-02 ✓ "依次接触……第一个**不会**往反方向行进的框" ✓ —— 一个都没有时
    就是"宁可不融合"✓）。

    ⚠ 口径**已改两次**（2026-10-03 ✓）：先"挪到落位那一刻"（旧断言"替身 False 也照旧挑中"✗
    已废）⇒ 再按用户流程图**改成"融合第 2 拍起每拍判：落点倒退超限 ⇒ 分离"** ✓
    （原话："**照着我的逻辑简化分离判定**" ✓）⇒ 现在替身成 False 时，**保持期内就判分离** ✓；
    只有**非保持期**（新会话 / 首拍）才不参与 ✓。
    """
    from perception.lie_registry import Entry, ShapeRegistry
    _orig = LieTracker._merge_fix_safe

    def _mk19(hold):
        t = _LT(noise_tol=2.0, inherit_dist=0.0)
        t._typ_area = 140.0 * 130.0
        t._rad = 60.0
        t._tbox_rad = 60.0
        t.pos = (200.0, 200.0)
        t._vel_rel = (-100.0, 0.0)
        e = Entry((300.0, 200.0, 140.0, 130.0), ts=0.0)
        e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        t._merge_hold = bool(hold)
        t._merge_hold_bid = tuple(e.bid)
        t._was_merged = True
        return t

    try:
        # ⚠ 替身要收新关键字 `allow_tangent`（用户 2026-10-03「A」✓ 分离判定会传 False ✓）
        LieTracker._merge_fix_safe = lambda self, box, **kw: False   # 替身：一律"不安全" ✓
        t = _mk19(True)                                          # 保持期 = 第 2 拍起 ✓
        _st, _bx = t._target_box_state([(300.0, 200.0, 300.0, 280.0)], {})
        check(_st == "gone" and _bx is None and t._split_now is True,
              "⑲ 替身判「不安全」⇒ **保持期内 ⇒ 判分离**（实际 %s/%s ／ `_split_now`=%s ✓ —— "
              "用户 2026-10-03「落点倒退超限 ⇒ 分离」✓）" % (_st, _bx, t._split_now))
        check((t._split_why or {}).get("how") == "落点倒退超限",
              "⑲ 分离原因 = 「**落点倒退超限**」（实际 %r ✓）"
              % ((t._split_why or {}).get("how"),))
        t2 = _mk19(False)                                        # 非保持期（新会话 ✓）⇒ 不参与 ✓
        _st2, _bx2 = t2._target_box_state([(300.0, 200.0, 300.0, 280.0)], {})
        check(_st2 == "merged" and _bx2 is not None and t2._split_now is not True,
              "⑲ 对照：**非保持期不参与**这条 ⇒ 照旧挑中（实际 %s ✓ 首拍/新会话自由挑 ✓）"
              % (_st2,))
    finally:
        LieTracker._merge_fix_safe = _orig


def test_board_overlap_param():
    """⭐⭐ **上板防抖重叠率**的接线（用户 2026-10-02 ✓ 原话："这个 0.5 做成参数配置『**上板
    防抖重叠率**』" ✓）：`_LT(board_overlap=…)` ⇒ **同一处口径**落到登记表
    （`ShapeRegistry.board_overlap` ✓）；`None` ⇒ 默认（`_BOARD_OVERLAP_DEF` = 0.5 ✓）；
    越界值夹进 `[0, 1]` ✓。判据本身在 `selftest_lie_registry` 里钉 ✓。"""
    from perception.lie_registry import _BOARD_OVERLAP_DEF
    check(abs(LieTracker()._reg.board_overlap - _BOARD_OVERLAP_DEF) < 1e-9,
          "不配 ⇒ 用登记表同一处默认值（实际 %.2f ｜ 期望 %.2f ✓ 用户定的 0.5 ✓）"
          % (LieTracker()._reg.board_overlap, _BOARD_OVERLAP_DEF))
    check(abs(LieTracker(board_overlap=0.7)._reg.board_overlap - 0.7) < 1e-9,
          "配 0.7 ⇒ 登记表拿到 0.7（实际 %.2f ✓）"
          % LieTracker(board_overlap=0.7)._reg.board_overlap)
    check(abs(LieTracker(board_overlap=3.0)._reg.board_overlap - 1.0) < 1e-9
          and abs(LieTracker(board_overlap=-1.0)._reg.board_overlap - 0.0) < 1e-9,
          "越界值夹进 [0,1]（3.0→%.2f、-1.0→%.2f ✓）"
          % (LieTracker(board_overlap=3.0)._reg.board_overlap,
             LieTracker(board_overlap=-1.0)._reg.board_overlap))


def test_merge_fix_adopt():
    """⭐⭐⭐ **甲方案：落位点 ⇒ 内部状态一起认**（用户 2026-10-02 ✓ 原话："**我期望绿圆就是
    我们认为的真目标所处的位置**" + "**改成甲试试 ——『落位点就是我们认定的真目标位置』⇒
    内部也该认它**" ✓）。

    要点（逐条钉）：
      · `self.pos` / `self._raw` **一起** = 落位点 ✓（= 绿圆 ≡ 内部信念 ✓）——
        ⚠ 这是对既有纪律「落位**只**改报告位置」的**受控修订** ✓（原委与代价见 `process()`
        出口的长注释 ✓：分叉实测显示帧 43 差 **53.1px**、圆会回跳 ✗）；
      · `out["raw_pos"]` 同步刷 ✓（演示窗那颗**绿点** = 内部真实位置 ⇒ 与圆心重合 ✓）；
      · `_clamped = True` ✓（这一跳是**信念几何推断**、不是目标自己走的 ⇒ 样本不入窗 ✓
        与夹取/统一出口同一把尺 ✓）；
      · ⚠ `_pos_prev` **不**在这儿写 ✗（由 `process()` 按"有没有真落位"决定 ✓）；
      · 抓不到位置（`out` 里没 `pos`）⇒ 返回 `None` 且**一个字段都不动** ✓ 不猜 ✗。
    """
    import pathlib

    from perception import lie_tracker as LT

    _src = (pathlib.Path(__file__).resolve().parent.parent / "perception"
            / "lie_tracker.py").read_text(encoding="utf-8")
    check("_adopted = self._adopt_merge_fix(out) is not None" in _src
          and 'self._pos_prev = out["pos"] if _adopted else _pos_int' in _src,
          "甲接线（源码级）：落位那一拍调 `_adopt_merge_fix` ✓、`_pos_prev` 按「有没有真落位」"
          "二选一 ✓（甲 ⇒ 起点 = 落位后的那个圆 ✓ 老分叉病根就在这条 ✓）")

    tr = LT.LieTracker()
    tr.pos = (11.0, 12.0)
    tr._raw = (5.5, 6.0)
    tr._pos_prev = (7.0, 8.0)
    tr._clamped = False
    out = {"pos": (100.0, 50.0), "raw_pos": (1.0, 2.0)}
    _ret = tr._adopt_merge_fix(out)
    check(_ret == (100.0, 50.0) and tr.pos == (100.0, 50.0),
          "内部位置 = 落位点（`pos` 现在 %s ✓ ⇒ **绿圆 ≡ 追踪器心里那个点** ✓）" % (tr.pos,))
    check(abs(tr._raw[0] - 100.0 * LT._SCALE) < 1e-9
          and abs(tr._raw[1] - 50.0 * LT._SCALE) < 1e-9,
          "半域 `_raw` 同步 = 落位点 ×%.1f（实际 %s ✓ —— 下一拍的**预测起点**就在这儿 ✓）"
          % (LT._SCALE, tr._raw))
    check(out.get("raw_pos") == (100.0, 50.0),
          "`out[raw_pos]` 同步刷（演示窗**绿点** = 内部真实位置 ⇒ 与圆心重合、小红点不出现 ✓ "
          "实际 %s ✓）" % (out.get("raw_pos"),))
    check(tr._clamped is True,
          "`_clamped` 置位（这一跳不是「目标自己走的」⇒ 轨迹样本不入窗 ✓ 与夹取同一把尺 ✓）")
    check(tr._pos_prev == (7.0, 8.0),
          "⚠ `_pos_prev` **不**由本方法写 ✗（仍是 %s ✓ 由 `process()` 按落位与否决定 ✓）"
          % (tr._pos_prev,))

    out2 = {"raw_pos": (1.0, 2.0)}
    check(tr._adopt_merge_fix(out2) is None and tr.pos == (100.0, 50.0),
          "抓不到位置（`out` 里没 `pos`）⇒ 返回 None 且**一个字段都不动** ✓（不猜 ✗）")


def test_merge_front_gate():
    """⭐⭐⭐ **融合框筛选闸**（用户 2026-10-02 ✓ 原话一字不改："只能往轨迹预测方向的**前方**挑与
    **绿圆圈外接矩形**相交的格子，如果没有能挑的就判定为**分离信号**（接入淡粉色框按轨迹预测走
    那一套）" ✓）。

    口径（逐条钉）：
      · 方向 = **白箭头**（`_vel_rel` ✓ 用户选 ✓）、基准 = **圆心** ✓（判据
        `(框中心 − 圆心) · 白箭头 ≥ 0` ✓ 与 `_merge_edge_own` 的"白箭头前方 180°"同一把尺 ✓）；
      · 相交 = 检出框与**绿圈外接矩形**（`圆心 ± 半径` 的方形 ✓ 用户点名 ✓）相交 ✓
        （⚠ 与 `_circle_hits` 的"圆 ↔ 矩形最近距"**不是同一判据** ✗）；
      · 适用范围 = **只作用于融合框 / 融合保持** ✓（用户选 ✓）—— "目标自己那格（`ok`）"与
        "未登记相交优先"两档照旧 ✓；
      · 两条都过 ⇒ 照旧 `merged` ✓；不过 ⇒ 这一拍**不挑它** ✗；若因此**没有可挑的格**
        ⇒ 判**分离信号**（`_split_now` ✓ 红框消失 + 淡粉接力框 + 圆按上一拍预测走 ✓）。
    ⚠ 实测背景（`9月30日(1).mp4` 显示帧 60 ✓）：旧实现会把绿圈**后方**另一格的并集框
      `(320.3,194.2)` 选成融合框 ⇒ 融合框"一跳 120px、方向与白箭头相反 173.8°"✗、绿圈被
      夹进新格（跳 86.7px）✗；加闸后那一拍不再融合 ⇒ 红框消失、圆按预测走 ✓。
    """
    from perception.lie_registry import Entry, ShapeRegistry

    def _tr(pos, vr, rad=60.0):
        t = _LT()
        t.pos = pos
        t._vel_rel = vr
        t._rad = rad
        t._tbox_rad = rad
        return t

    # ---- 判据①「预测方向的前方」----
    _t1 = _tr((200.0, 200.0), (-100.0, 0.0))           # 白箭头指向左 ✓
    check(_t1._box_ahead_of_pred((120.0, 200.0)) is True,
          "① 框在**预测前进方向的前方**（左 ✓）⇒ True ✓")
    check(_t1._box_ahead_of_pred((280.0, 200.0)) is False,
          "① 框在**预测方向的后方**（右 ✓）⇒ False ✓（= 那一拍不许把它当融合框 ✗）")
    _t2 = _tr((200.0, 200.0), (0.0, 0.0))
    check(_t2._box_ahead_of_pred((280.0, 200.0)) is True,
          "① 白箭头≈0（还没学出方向）⇒ **判不出就放行**（True ✓ 不猜 ✗）")

    # ---- 判据②「与绿圈外接矩形相交」----
    _t3 = _tr((200.0, 200.0), (0.0, 0.0), rad=50.0)    # 外接矩形 = x,y ∈ [150,250] ✓
    check(_t3._circle_rect((240.0, 200.0, 20.0, 20.0)) is True,
          "② 框压在外接矩形上 ⇒ True ✓")
    check(_t3._circle_rect((320.0, 200.0, 20.0, 20.0)) is False,
          "② 框在外接矩形外（x 中心 320 − 半宽 10 = 310 > 250 ✓）⇒ False ✓")
    check(_t3._circle_rect((280.0, 200.0, 60.0, 20.0)) is True,
          "② 框边缘探进外接矩形（左缘 250 ✓ 正好相切）⇒ True ✓（相切算相交 ✓）")
    _t4 = _tr((200.0, 200.0), (0.0, 0.0), rad=0.0)
    _t4._tbox_rad = 0.0
    check(_t4._circle_rect((900.0, 900.0, 10.0, 10.0)) is True,
          "② 半径还没学到（判不出）⇒ **保守放行**（True ✓ 不猜 ✗）")

    # ---- 行为：后方/不交的并集框**不被挑**；没有可挑的 ⇒ 判分离信号 ----
    # ---- ⭐⭐⭐ 判据①'「S > 融合框继承面积限制」与 判据②'「沿白箭头推进第一个接触的框」
    #   （用户 2026-10-02 ✓ **二次定稿**：闸从"框心在前方 + 布尔外接相交"✗ 换成这两条 ✓）----
    _t4 = _tr((200.0, 200.0), (-100.0, 0.0), rad=50.0)   # 外接矩形 = x,y ∈ [150,250]（面积 10000）
    _s_center = _t4._circle_rect_overlap_ratio((200.0, 200.0, 120.0, 120.0))  # 完全包住外接矩形
    check(abs(_s_center - 1.0) < 1e-9,
          "①' 框完全包住绿圈外接矩形 ⇒ S = 1.00（相交 / **外接矩形**面积 ✓ 分母是外接矩形 ✗ "
          "不是框面积 ✓ 用户点名 ✓ 实际 %.2f ✓）" % _s_center)
    _s_edge = _t4._circle_rect_overlap_ratio((250.0, 200.0, 100.0, 120.0))   # 只压住右半
    check(abs(_s_edge - 0.5) < 1e-9,
          "①' 只压住一半 ⇒ S = 0.50（同一组数、同一把尺 ✓ 实际 %.3f ✓）" % _s_edge)
    _s_small = _t4._circle_rect_overlap_ratio((200.0, 200.0, 20.0, 20.0))    # 小框压住中间
    check(abs(_s_small - 0.04) < 1e-9,
          "①' ⭐ **框比圆圈小 ⇒ S 也小**（20×20 / 10000 = 0.04 ✓ —— 这条正好说明分母取"
          "「外接矩形」的用意：**光有框大没用，要压在圆圈那块地方上** ✓ 用户点名 ✓）")
    check(_t4._circle_rect_overlap_ratio((400.0, 200.0, 20.0, 20.0)) == 0.0,
          "①' 完全没压住 ⇒ S = 0.00 ✓（⇒ 这条闸会把它挡下 ✗）")

    _t6 = _tr((200.0, 200.0), (-100.0, 0.0))             # 白箭头指向**左** ✓
    _first6 = _t6._first_hit_box([(320.0, 200.0, 40.0, 40.0),   # 远的（右）✗
                                  (120.0, 200.0, 40.0, 40.0)])  # 近的（左）✓
    check(_first6 == (120.0, 200.0),
          "②' 沿白箭头（左）推进 ⇒ **第一个接触的框** = 左边那格（实际 %s ✓ —— 与"
          "「框心是否在圆心前方」不是一回事 ✓ 后者两格都可能算「前方」✗）" % (_first6,))
    check(_t6._first_hit_box([(120.0, 200.0, 40.0, 40.0),
                              (320.0, 200.0, 40.0, 40.0)]) == (120.0, 200.0),
          "②' 与框的**遍历顺序无关** ✓（算的是沿方向的进入距离 ✓）")
    check(_tr((200.0, 200.0), (0.0, 0.0))._first_hit_box([(320.0, 200.0, 40.0, 40.0)]) is None,
          "②' 白箭头≈0（还没学出方向）⇒ `None` ⇒ 调用处**放行全部** ✓（判不出就不猜 ✗）")

    # ---- 判据③「两条都过才许当融合候选；都筛掉 ⇒ 门内没得挑 ⇒ 判分离」----
    def _mk(inherit_dist):
        t = _LT(noise_tol=2.0, inherit_dist=inherit_dist)
        t._typ_area = 140.0 * 130.0
        t._rad = 60.0
        t._tbox_rad = 60.0
        t.pos = (200.0, 200.0)
        t._vel_rel = (-100.0, 0.0)                     # 预测往左 ✓
        e = Entry((300.0, 200.0, 140.0, 130.0), ts=0.0)
        e.on_board = True
        rg = ShapeRegistry()
        rg.entries = [e]
        t._reg = rg
        t._merge_hold = True                           # 前提：正在融合保持 ✓
        t._merge_hold_bid = tuple(e.bid)
        t._was_merged = True
        return t

    # ⚠ 框心必须在 `_DET_GATE`(130px) 内 ✗ 否则一开始就被门排除 ⇒ 测不到闸 ✗（框心 (300,200)
    #   距圆心 100px ✓）；比 ≈ 4.6× ⇒ `_brick_owned` 不成立 ✓（不会被"属于砖"那支接走 ✓）。
    _tb5 = (300.0, 200.0, 300.0, 280.0)                # 框 x[150,450] y[60,340]
    _t5 = _mk(3.0)                                     # 阈值极大 ⇒ 闸必过
    check(_t5._is_merged_box(_tb5) or _t5._hold_active(_tb5),
          "③（前提）这格框**本该算融合框**（包住已上板砖 ✓）")
    _dr5 = _t5._inner_edge_dist_ratio(_tb5)
    check(_dr5 is not None and _dr5 > 0.5,
          "③ 圆心 (200,200) **落在框内** ⇒ 距离比 = %.3f（正 ✓）；**框内 ⇒ 距离闸直接放行**"
          "（不看阈值 ✓ 用户 2026-10-02 原话：「圆心在框外才判定，圆心在框内直接允许融合」✓）"
          % (_dr5,))
    _st5, _bx5 = _t5._target_box_state([_tb5], {})
    check(_bx5 is not None,
          "③ ⭐ **单候选 + 框内 ⇒ 照旧被挑**（实际 %s ✓ —— 新闸**不再**因为「框心不在圆心"
          "正前方向」就丢掉它 ✗ 这正是用户这次点出来的毛病 ✓）" % (_bx5,))
    # ⚠⚠⭐ **本段显式打开「融合框继承距离限制」开关**（用户 2026-10-03 ✓ 原话："有了这个逻辑，
    #   就不需要『融合框继承距离限制』了，**先屏蔽相关逻辑**" ✓ 见 `_INHERIT_DIST_ON` ✓）——
    #   屏蔽态下它**恒放行** ⇒ 下面这些"闸在筛候选"的断言会失真 ✗ ⇒ 开关拨到 `True` 跑完立刻
    #   拨回 ✓（两态都钉 ✓）。
    import perception.lie_tracker as _LTim
    _on0 = _LTim._INHERIT_DIST_ON
    _LTim._INHERIT_DIST_ON = True
    # ⭐ 把**圆心挪到框外**（出框 20px = 0.333 半径）+ 阈值 0.0（一点也不许出框）⇒ 第一条闸挡下
    #   ⇒ 门内没得挑 ⇒ 判分离 ✓
    _t7 = _mk(0.0)
    _t7.pos = (130.0, 200.0)                           # 框左缘 150 ⇒ 出框 20px
    _st7, _bx7 = _t7._target_box_state([_tb5], {})
    check(_bx7 is None,
          "③ ⭐ **圆心出框（距离比 %.3f < 0）+ 阈值 0.00 ⇒ 这一拍不挑它**（实际 %s ✓）"
          % (_t7._inner_edge_dist_ratio(_tb5), _bx7))
    # ⭐⭐ **新口径本身**（用户 2026-10-02 ✓ 两次收紧后的定稿）：**框内直接放行** ✓；
    #   **框外**才按"**允许圆心出框的倍数半径**"判 ✓（距离比 = **有符号** ✓ 负的就是出框量 ✓）。
    _t9 = _mk(0.0)
    _t9.pos = (130.0, 200.0)                           # 框左缘 150 ⇒ 圆心出框 20px
    check(abs(_t9._inner_edge_dist_ratio(_tb5) + 20.0 / 60.0) < 1e-9,
          "③ 圆心出框 20px ⇒ 距离比 = **−0.333**（**有符号** ✓ —— 负的就是「出框多少」✓）")
    check(_t9._inherit_dist_ok(_tb5) is False,
          "③ 阈值 0.00 ⇒ **一点也不许出框** ⇒ 出框 20px 不过 ✓")
    _t9.inherit_dist = 0.5                             # 允许出框 0.5 × 60 = 30px
    check(_t9._inherit_dist_ok(_tb5) is True,
          "③ 阈值 0.50 ⇒ 允许出框 30px ⇒ 出框 20px **通过** ✓（阈值 = 允许出框的倍数半径 ✓）")
    _t9.inherit_dist = 0.3                             # 只允许 18px < 20px
    check(_t9._inherit_dist_ok(_tb5) is False,
          "③ 阈值 0.30 ⇒ 只允许出框 18px < 20px ⇒ 仍不过 ✓")
    _t9.inherit_dist = 0.0
    _t9.pos = (200.0, 200.0)                           # 回框内（离最近边 50px）
    check(abs(_t9._inner_edge_dist_ratio(_tb5) - 50.0 / 60.0) < 1e-9,
          "③ 圆心在框内、离最近边 50px ⇒ 距离比 = **0.833** ✓")
    check(_t9._inherit_dist_ok(_tb5) is True,
          "③ **圆心在框内 ⇒ 直接允许**（阈值 0.00 也过 ✓ 用户点名：「圆心在框内直接允许融合」✓）")
    _t9.inherit_dist = 3.0
    check(_t9._inherit_dist_ok(_tb5) is True,
          "③ 圆心在框内 + 阈值 3.00 ⇒ 照旧过 ✓（**框内不受阈值约束** ✓）")
    check(LieTracker(inherit_dist=-5.0).inherit_dist == 0.0,
          "③ 「允许出框」的**负数无意义** ⇒ 夹到 **0** ✓（用户 2026-10-02：距离正数判定才有意义 ✓）")
    check(_t7._split_now is True,
          "③ ⇒ **没有任何可挑的格** ⇒ 判**分离信号**（`_split_now`=True ✓ ⇒ 红框消失 + 淡粉"
          "接力框 + 圆按上一拍预测走 ✓ 用户 2026-10-02 ✓）")
    _LTim._INHERIT_DIST_ON = _on0               # ⭐ 拨回（现行 = 屏蔽 ✓）
    # ⭐⭐ **屏蔽态（现行 ✓）**：同一个局面（圆心出框 20px）⇒ 这条闸**不再筛** ⇒ **照旧被挑** ✓
    #   （用户 2026-10-03 ✓ 原话："有了这个逻辑，就不需要它了，先屏蔽" ✓ —— 改由「内砖同一性」
    #   +「允许倒退距离」把关 ✓）
    _t10 = _mk(0.0)
    _t10.pos = (130.0, 200.0)                   # 同样出框 20px ✓
    check(_t10._inherit_dist_ok(_tb5) is True,
          "③ ⭐ **屏蔽态**（现行 ✓ `_INHERIT_DIST_ON=False`）⇒ `_inherit_dist_ok` **恒 True**"
          "（同一局面、阈值 0.00，出框 20px 也照旧放行 ✓ = 这条闸不再筛候选 ✓）")
    check(_t10._inner_edge_dist_ratio(_tb5) < 0,
          "③ 而且距离比照旧算得出来（%.3f < 0 ✓ ⇒ 只是**不参与筛选** ✗；判据函数与配置项都留着 ✓"
          "开关拨回 `True` 即恢复 ✓）" % (_t10._inner_edge_dist_ratio(_tb5),))


def main():
    print("测谎追踪器自检（合成序列）：")
    test_lie_tracker_synthetic()
    test_track_memory()
    test_noise_not_win()
    test_size_suspects()
    test_white_static_vs_scrolling_bg()
    test_report_rate_limit()
    test_area_and_group()
    test_box_is_registered_fake()
    test_box_identity()
    test_merge_rule1_center_range_hits()
    test_merge_succ_no_swap()
    test_sep_vel_cap()
    test_brick_edge_max()
    test_edge_own_one_brick()
    test_same_inner_brick()
    test_sep_retreat_simple()
    test_geom_three_rulers()
    test_merge_iou_threshold()
    test_reg_first_frame_iou_gate()
    test_iou_brick_leg()
    test_pick_log()
    test_select_step_gate()
    test_edge_tangent_when_too_small()
    test_rel_vel_cap()
    test_gate_base_is_prev_pred()
    test_kf_group_anchor()
    test_kf4_unit()
    test_kf_vs_classic_on_synth()
    test_unregistered_box_beats_merge()
    test_merge_edge_own()
    test_tbox_prefers_merge_over_fake()
    test_sep_takeover()
    test_anti_group_vel()
    test_path_curvature()
    test_board_overlap_param()
    test_split_now_flag_and_forecast()
    test_merge_first_nearest_corner()
    test_merge_log_payload()
    test_split_log_payload()
    test_sep_ratio_gate()
    test_merge_fix_adopt()
    test_merge_front_gate()
    test_merge_fix_safe()
    test_merge_fix_all_unsafe()
    if _FAILED:
        print("自检：%d 条失败" % len(_FAILED))
        return 1
    print("测谎追踪器自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

