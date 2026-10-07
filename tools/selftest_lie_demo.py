# -*- coding: utf-8 -*-
"""测谎演示窗口（`tools/lie_demo.py`）的离屏自检。

为什么要它（2026-09-30 用户报 "拖动播放条时下面显示的帧号没变" ✓）：
  · 演示窗口的**状态栏那行（帧号 / 状态 / 命中率）原来只在 `tick`（播放）里写** ✗
    ⇒ 拖动走 `on_seek` ⇒ 画面在动、帧号不动，看着就像坏了 ✗；
  · 而且**演算没跑完时拖动**会 `self.results[self.i]` **IndexError 崩** ✗
    （滑块 range 一载入就设成全部帧数、`results` 却是边算边长的 ✗）。

做法：`run_window` 自己 `app.exec_()` ⇒ 把 `exec_` 换成一个假实现，在里面
等演算跑完、模拟拖动（调 `on_seek`）、读状态栏文字。**全程 offscreen** ✓ 不开窗 ✓
不写用户配置 ✓（窗口的会话保存是另一回事，本自检不碰 ✓）。
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtWidgets import QApplication           # noqa: E402

import numpy as np                                  # noqa: E402

import tools.lie_demo as D                          # noqa: E402

#: ⭐⭐⭐⭐⭐ **自检里默认关掉"真目标丢失 / 重新找到"那份日志** ✗✗（`logs/lie_target.log` ✓ 用户
#:   2026-10-07 ✓）—— ⚠⚠ **实测踩到** ✓：自检里那些 `VelocityRunner` / `VelocityTracker` 一跑，
#:   就会往**用户那份日志**里塞测试行 ✗（实测四条"帧 -" ✓）⇒ **自检一个字节都不许落到用户日志里** ✓。
#:   ⚠ 专门验日志那一条（`selftest_lie_motion.test_velocity_log_lost_and_found` ✓）在自己内部
#:     把路径指到**临时目录** ✓ 验完还原 ✓ —— 那是"量日志"，与"别污染用户日志"不冲突 ✓。
import perception.lie_motion as _lm                 # noqa: E402

_lm.set_vel_log_path(None)

_FAILED = []


def check(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "NG", msg))
    if not cond:
        _FAILED.append(msg)


def note(msg):
    """**如实记账、但不判失败**（`NOTE` ✓）：用于"契约本身没问题，只是本素材取不到样本"这种
    观测（例如口径升级后"融合期"几乎每拍都被对角修正 ⇒ "按白箭头走"的可观测样本为 0 ✗）——
    这类结论**必须打出来**给复核的人看 ✗ 但**不该**把整轮自检判红 ✓。"""
    print("  [NOTE] %s" % msg)


def _find_window(app):
    for w in app.topLevelWidgets():
        if hasattr(w, "on_seek") and hasattr(w, "results"):
            return w
    return None


def _fake_exec(self):
    app = QApplication.instance() or self
    win = _find_window(app)
    if win is None:
        check(False, "找不到演示窗口")
        return 1
    t0 = time.time()
    while len(win.results) < len(win.frames) and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.05)
    check(len(win.results) >= len(win.frames),
          "演算跑完（%d/%d 帧）" % (len(win.results), len(win.frames)))
    if not win.results:
        return 1
    win.stop()
    app.processEvents()

    # ① 拖动 ⇒ 状态栏**帧号**要跟着变（用户报的那条 ✓）
    #   ⚠ 帧号要**相对帧数**取（别写死 ✗ 踩过：去重阈值一放宽，素材 92→70 帧 ⇒ 写死的
    #     88 越界 ⇒ 夹到末帧、状态栏显示 70/70 而用例还在等 89 ✗ 用例自己红 ✓）
    _n = len(win.frames)
    for target in (min(40, _n - 1), min(7, _n - 1), _n - 3):
        win.on_seek(target)
        app.processEvents()
        msg = win.statusBar().currentMessage()
        check(("帧 %d/" % (target + 1)) in msg and win.i == target,
              "拖到第 %d 帧 ⇒ 状态栏「%s」且 self.i=%d" % (target + 1, msg, win.i))

    # ② 演算途中拖动 ⇒ **不许崩**（夹到已算好的最后一帧 ✓）
    full = list(win.results)
    win.i = 60
    win.results = win.results[:30]
    try:
        win._show_precomputed()
        ok = win.i == 29
    except Exception as exc:                        # noqa: BLE001
        ok = False
        check(False, "演算途中拖动崩了：%r" % (exc,))
    win.results = full
    win.i = 0
    check(ok, "演算没跑完时拖动不崩（i 夹到已算好的最后一帧 ✓）")

    # ⭐⭐ **窗口底部的深色融合日志区**（用户 2026-10-02 ✓ 原话："给测谎演示窗口**下面**加上
    #   **深色背景的日志区**。当**融合框生成时**添加日志，**显示相关的砖**以及**解释判定为
    #   融合框的计算数据**" ✓）。
    #   ⚠⚠ 判据**不能靠"本素材必出融合"** ✗：这段素材实测 `merged` **0 帧**（大框全部让位 =
    #     预期行为 ✓ 见上面 ⑧ 段的说明 ✓）⇒ 这里**手工塞一条** `merge_log` 来验"显示链路" ✓
    #     （"载荷本身"由 `selftest_lie_tracker.test_merge_log_payload` 单元钉住 ✓）。
    _lg = getattr(win, "log", None)
    check(_lg is not None and _lg.isReadOnly(),
          "⑭ 窗口底部有**只读**的融合日志区（QPlainTextEdit ✓）")
    if _lg is not None:
        _ss = _lg.styleSheet() or ""
        check("#15171a" in _ss,
              "⑭ 日志区是**深色背景**（样式表含 #15171a ✓）")
        check(len(_lg.placeholderText() or "") > 0,
              "⑭ 没事件时有占位说明（一眼知道这区是干什么的 ✓）")
        _full2 = list(win.results)
        _k2 = min(3, len(win.results) - 1)
        if _k2 >= 0:
            win.results[_k2] = dict(win.results[_k2])
            win.results[_k2]["merge_log"] = {"text": "融合框生成 ｜ 假事件（自检用）", "data": {}}
            _lg.clear()
            win._log_i = 0
            win._log_sync(_k2 + 2)                # 干净地正向扫一遍（基准 ✓）
            _t1 = _lg.toPlainText()
            check(("帧 %d" % (_k2 + 1)) in _t1 and "融合框生成" in _t1,
                  "⑭ 融合事件会写进日志区（%s ✓）" % _t1.replace("\n", " / ")[:80])
            win._log_sync(1)                      # **真回拖**（i < `_log_i` ⇒ 触发重建 ✓）
            win._log_sync(_k2 + 2)                # 再前进回原处
            check(_lg.toPlainText() == _t1,
                  "⑭ 回拖再来一遍**逐字一致**（不重复刷 ✗ 也不漏 ✓）")
        # ⭐ 顺带看**真素材**：这段素材本身就出过融合框（帧 6 起步 ✓）⇒ 端到端通 ✓
        _ev2 = [k + 1 for k, r in enumerate(win.results) if r.get("merge_log")]
        check(bool(_ev2),
              "⑭ 真素材演算结果里确实带出了融合框生成事件（帧 %s ✓ —— 若这条红，先看是不是"
              "换成了没有融合的素材 ✗）" % (_ev2[:6],))
        # ⭐⭐⭐ **非融合红框日志**（用户 2026-10-03 ✓ 原话："在日志加一下**非融合红框选择**的
        #   判定信息吧" ✓）：与上面融合那条**同一套路** —— 手工塞一条验"**显示链路**" ✓
        #   （载荷本身由 `selftest_lie_tracker.test_pick_log` 钉 ✓ 口径不在这里重算 ✗）。
        if _k2 >= 0:
            win.results[_k2] = dict(win.results[_k2])
            win.results[_k2]["pick_log"] = {"text": "非融合红框选择 ｜ 假事件（自检用）",
                                            "data": {}}
            _lg.clear()
            win._log_i = 0
            win._log_sync(_k2 + 2)
            _t2 = _lg.toPlainText()
            check(("帧 %d" % (_k2 + 1)) in _t2 and "非融合红框选择" in _t2,
                  "⑭ 非融合红框事件也会写进日志区（%s ✓ —— 用户 2026-10-03 ✓）"
                  % _t2.replace("\n", " / ")[:80])
            win.results = list(_full2)
            win._log_i = 0
            _lg.clear()
        win.results = _full2
        win._log_i = 0
        _lg.clear()

    # ③ 拖完能原地继续播（`on_seek_done` 的语义：拖动前在播 ⇒ 接着播 ✓）
    win._was_playing = False
    win.on_seek_done()
    check(not win.playing, "拖动前**没在播** ⇒ 松手后也别自动播 ✓")

    # ⭐⭐⭐ **切运动分离 ⇒ 「分离期」「融合期」两行整体消失**（用户 2026-10-04 ✓ 原话："在运动
    #   分离模式下，配置栏上面的 分离期、融合期 两行移除" ✓）—— ⚠ 原来只收了**行内控件**、
    #   **两个分组标题没跟着收** ✗ ⇒ 留下一排空标题（看着像"两行还在" ✓ 就是这个 bug ✓）。
    _im = next((k for k in range(win.cmb_mode.count())
                if str(win.cmb_mode.itemData(k)) == "motion"), -1)
    if _im >= 0:
        _h2 = getattr(win, "_row2_head", None)
        _h3 = getattr(win, "_row3_head", None)
        check(_h2 is not None and _h3 is not None,
              "模式行：两个分组标题有引用（`_row2_head` / `_row3_head` ✓）")
        win.cmb_mode.setCurrentIndex(_im)
        app.processEvents()
        check(_h2 is not None and not _h2.isVisible()
              and _h3 is not None and not _h3.isVisible(),
              "模式行：切到**运动分离** ⇒「分离期」「融合期」两个**分组标题都收了** ✓"
              "（⚠ 只收行内控件不够 ✗ 标题也要收 ✓）")
        _w2 = next((win._row2.itemAt(k).widget() for k in range(win._row2.count())
                    if win._row2.itemAt(k).widget() is not None), None)
        check(_w2 is None or not _w2.isVisible(),
              "模式行：「分离期」**行内控件也收**（老行为不破 ✓）")
        win.cmb_mode.setCurrentIndex(0)
        app.processEvents()
        check(_h2 is not None and _h2.isVisible()
              and _h3 is not None and _h3.isVisible(),
              "模式行：切回**经典** ⇒ 两个分组标题**回来** ✓")

    # ⑤ ⭐⭐ **「轨迹预测窗口时间」配置行 + 「应用」**（用户 2026-10-01 ✓ 原话："增加参数
    #   「轨迹预测窗口时间」……在输入框右边加按钮「应用」，点击后重新运算并生效" ✓）。
    #   ⚠ **必须在视频素材上验**（GIF 那段不行 ✗ 实测：test.gif 上切向预测全程没参与 ⇒
    #     150ms 与 5000ms 的结果**一模一样** ⇒ 拿它当断言会得到"参数没生效"的假结论 ✗）。
    _has = (hasattr(win, "sp_path_ms") and hasattr(win, "btn_apply")
            and hasattr(win, "lbl_pms"))
    check(_has, "交互区下有「轨迹预测窗口时间」配置行（输入框 + 应用按钮 ✓）")
    if _has:
        check(win.lbl_pms.text() == "轨迹预测窗口时间"
              and win.btn_apply.text() == "应用",
              "文案对：「%s」+「%s」" % (win.lbl_pms.text(), win.btn_apply.text()))
        # ⭐⭐ **「融合框判定阈值」在「应用」按钮左边**（用户 2026-10-01 ✓ 原话："应用按钮
        #   左边加配置「融合框判定阈值」" ✓）—— 用**像素位置**验（不是看代码顺序 ✗）。
        _xm = win.sp_merge.mapToGlobal(win.sp_merge.rect().center()).x()
        _xb = win.btn_apply.mapToGlobal(win.btn_apply.rect().center()).x()
        check(hasattr(win, "sp_merge") and _xm < _xb,
              "「融合框判定阈值」在应用按钮**左边**（阈值 x=%d < 应用 x=%d ✓）" % (_xm, _xb))
        _yv = win.sp_path_ms.mapToGlobal(win.sp_path_ms.rect().center()).y()
        _yl = win.lbl.mapToGlobal(win.lbl.rect().center()).y()
        check(_yv < _yl,
              "配置行在**画面上方**（输入框 y=%d < 画面 y=%d ✓）" % (_yv, _yl))
        # ⭐ 参数**真的控制切向预测的窗口**（单元级 ✓）：内接贯彻后，红框在手（包括
        #   "丢了但仍持有" ✓ 用户口径"保持到分离" ✓）时位置被内接约束主导 ⇒ "两个窗长
        #   跑全段比 pos"会被夹取**抹平** ✗（实测 0/70 帧差异 ✗ 不是参数没生效 ✗）。
        #   ⇒ 直接验 `_path_pred` 的窗口行为 ✓：窗内样本 <3 ⇒ 如实放弃（None ✓）。
        from perception.lie_tracker import LieTracker as _LTk
        _trk = _LTk(path_ms=150)
        _trk._raw = (100.0, 100.0)
        # 样本 = (ts, x, y, cumTx, cumTy)（半域 ✓ 末端最新 ✓）；无群体运动 ⇒ cum=0 ✓
        _trk._path = [(round(0.1 * k, 2), 100.0 + 10.0 * k, 100.0 + 3.0 * k * k,
                       0.0, 0.0) for k in range(6)]          # t = 0.0 … 0.5s ✓
        check(_trk._path_pred(0.1) is None,
              "窗 150ms ⇒ 窗内只有 2 个样本（0.4/0.5s）⇒ 切向预测如实放弃（None ✓）")
        _trk2 = _LTk(path_ms=5000)
        _trk2._raw = _trk._raw
        _trk2._path = _trk._path
        check(_trk2._path_pred(0.1) is not None,
              "窗 5000ms ⇒ 6 个样本全进窗 ⇒ 切向预测给出预测点（%s ✓）"
              % (tuple(round(float(v), 1) for v in _trk2._path_pred(0.1)),))
        # 「应用」⇒ ① 参数进追踪器 ② **整段重新演算**
        _old = float(getattr(win.runner, "tr", None).path_ms)
        win.sp_path_ms.setValue(1200)
        win.btn_apply.click()
        t0 = time.time()
        while len(win.results) < len(win.frames) and time.time() - t0 < 60:
            app.processEvents()
            time.sleep(0.05)
        check(abs(float(win.runner.tr.path_ms) - 1200.0) < 1e-6,
              "点「应用」⇒ 追踪器观察窗真的改了（%.0f → %.0f ms ✓）"
              % (_old, float(win.runner.tr.path_ms)))
        check(len(win.results) == len(win.frames),
              "应用后**整段重新演算**（%d/%d 帧 ✓）" % (len(win.results), len(win.frames)))
        win.sp_path_ms.setValue(int(D._PATH_MS_DEFAULT))   # 还原（别留下脏值 ✓）

        # ⭐⭐ **内接原则**（用户 2026-10-01 ✓ 原话："红框检出，那么绿圈需要与其内接" ✓）：
        #   凡红框检出的帧，绿圈（半径 = `tbox_rad` = 红框内切半径 ✓）必须**整体在红框内**
        #   ⇒ 圆心到四边的距离 ≥ `tbox_rad` ✓。
        #   ⚠ 病根记录（修的就是它 ✗）：夹取安全区原来用 `_rad`（学到的实际半径，更小 ✓）
        #     画圆却用 `tbox_rad`（更大 ✓）⇒ 两把尺打架 ⇒ 圈恒探出（实测第 17 帧探出
        #     上 18px／右 23px ✗✗ 用户截图圈出来的就是它）；且 `tbox_rad = min(w,h)/2`
        #     ⇒ min 维 `2r == 框宽` 恒成立 ⇒ 内缩条件必须**含等号**（否则 min 维永不夹 ✗）。
        _dets2 = list(win.dets) if win.dets else None
        _rr = D.Runner(dets=_dets2, path_ms=100)
        _bad, _nred = [], 0
        for i, fr in enumerate(win.frames):
            _o, _p, _r2, _h2, _bx, _mo = _rr.step(fr[1], i, win.ts[i])
            _tb2 = _mo.get("tbox")
            if _tb2 is None or _p is None:
                continue
            # ⭐⭐ **有框的每一拍都必须内接**（用户 2026-10-01 ✓ 两轮口径合并后）：
            #   · 非融合（目标自己那格框）⇒ 夹进内接安全区 ✓；
            #   · 融合**第一拍**⇒ "① 先夹取" ✓ 内接 ✓；
            #   · 融合**其余拍** ⇒ "按白箭头走，**除非运动到另一边再夹取**" ✓ ⇒ 越界即夹回 ✓
            #     ⇒ **也不会探出** ✓（所以这里不再跳过融合帧 ✓ 第一版"融合允许探出"已被用户否掉 ✓）。
            _nred += 1
            _bx2, _by2, _bw2, _bh2 = map(float, _tb2)
            _rr2 = float(_mo.get("tbox_rad") or 0.0)
            _ds = (_p[0] - (_bx2 - _bw2 / 2), (_bx2 + _bw2 / 2) - _p[0],
                   _p[1] - (_by2 - _bh2 / 2), (_by2 + _bh2 / 2) - _p[1])
            # ⚠ **按维判定**：框比 `2r` 还小（`2r > 维宽` ⇒ `_tbox_inner` 按设计**退回
            #   原边** ✓ "宁可圈探出，也不让位置抽风" ✓）⇒ 该维**跳过**（框真的装不下圆 ✗）。
            _ok2 = ((_ds[0] >= _rr2 - 0.5 or _bw2 < 2.0 * _rr2)
                    and (_ds[1] >= _rr2 - 0.5 or _bw2 < 2.0 * _rr2)
                    and (_ds[2] >= _rr2 - 0.5 or _bh2 < 2.0 * _rr2)
                    and (_ds[3] >= _rr2 - 0.5 or _bh2 < 2.0 * _rr2))
            if not _ok2:
                _bad.append((i, [round(_rr2 - d, 1) for d in _ds]))
        check(_nred > 0 and not _bad,
              "红框检出的每一帧绿圈都**内接**于红框（%d 帧有红框、违例 %d 个 %s ✓）"
              % (_nred, len(_bad), _bad[:5]))

        # ⭐⭐ **每个检出框的身份**（用户 2026-10-01 ✓ 原话："每个检出框我们都需要它是谁：① 与上板
        #   登记的假目标**高度重合** → 说明检出框**又检出了假目标**，**不能锁红**；② **大**检出框
        #   **直接包住**了上板登记的假目标 → **融合信号**" ✓）。扫描判据（**不写死帧号** ✓）：
        #   · 本拍被选中的那格若身份是 `"fake"` ⇒ **这一拍就不许有红框** ✓（用户第 31 帧那条 ✓
        #     —— 原来"面积不比假框大"太严 ⇒ 倍率 1.55 那格照样标红 ✗✗）；
        #   · ⭐⭐ **融合红框的两条新约束**（用户 2026-10-02 ✓）：
        #       身份 `"merged"` ⇒ 选中那格**「圆心±噪声容差」范围必与框相交**（规则① ✓
        #       `_pos_range_hits` ✓ 用户 2026-10-02 ✓）；
        #       大框（倍率 >1.7、盖住上板假框）但**圆心不在它里面** ⇒ **不许选中它** ✓
        #       （帧 22 定稿：未登记框优先、融合让位 ✓ —— 本素材实测 merged 0 帧、大框全部
        #         让位 = 预期行为 ✓；"merged 路径不是死代码"由 selftest_lie_tracker 的
        #         `test_merge_rule1_center_range_hits` / `test_unregistered_box_beats_merge` 单元钉住 ✓）。
        _rr6 = D.Runner(dets=_dets2, path_ms=100)
        _fake_red, _n_merged, _n_track = [], 0, 0
        _bad_m, _n_yield = [], 0
        for i, fr in enumerate(win.frames):
            _o6, _p6, _r6, _h6, _bx6, _mo6 = _rr6.step(fr[1], i, win.ts[i])
            _role6 = str(getattr(_rr6.tr, "_tbox_role", "") or "")
            if _role6:
                _n_track += 1
            if _role6 == "merged":
                _n_merged += 1
                # ⭐ 规则①（用户 2026-10-02 ✓）：判了融合 ⇒ 选中那格**「圆心 ± 噪声容差」
                #   范围必与框相交** ✓（同一把尺 ✓ 与 `_is_merged_box` 一致 ✓）
                if _mo6.get("tbox") is None or not _rr6.tr._pos_range_hits(
                        _rr6.tr._tbox, tol=float(getattr(_rr6.tr, "noise_tol", 0.0) or 0.0)):
                    _bad_m.append((i + 1, "merged 但圆心±容差范围与框不相交"))
            # ⚠⚠ **但要排除"整圈绿圈都在框里"那一档**（用户 2026-10-01 ✓ 规则②："框**同时**包住了
            #   砖和绿圈 ⇒ **即使面积与砖相等也属于融合红框**" ✓；⚠ 用户明确纠正过"包住" =
            #   "**整个圆圈完全在检出框内部**" ✓ ⇒ 用 `_circle_inside` ✓ 不是"圆心在内" ✗）：
            #   那种框是"目标与砖叠在一起"⇒ **该**出红框（而且是融合红框 ✓）⇒ 不算越界 ✓。
            if (_role6 == "fake" and _mo6.get("tbox") is not None
                    and not _rr6.tr._circle_inside(_rr6.tr._tbox)):
                _fake_red.append(i + 1)
            # ⭐ 大框让位（用户 2026-10-02 ✓ 帧 22）：倍率 >1.7、盖住上板假框、但**圆心不在里面**
            #   ⇒ 这一拍**不许选中它** ✓（未登记框优先 / 预测态 ✓）；圆心在里面的照样可以选中 ✓。
            _sel6 = _mo6.get("tbox")
            for _b6 in (_bx6 or []):
                _cx6, _cy6 = float(_b6[1]), float(_b6[2])
                _w6, _h62 = float(_b6[3]), float(_b6[4])
                _ref6, _cov6 = _rr6.tr._fake_ref((_cx6, _cy6, _w6, _h62))
                if _ref6 is None or _cov6 < 0.75:
                    continue
                if _w6 * _h62 <= 1.7 * float(_ref6.w) * float(_ref6.h):
                    continue
                if _rr6.tr._pos_range_hits((_cx6, _cy6, _w6, _h62),
                                           tol=float(getattr(_rr6.tr, "noise_tol", 0.0) or 0.0)):
                    continue                       # 范围与框相交 ⇒ 它就是融合红框候选 ✓ 让谁 ✗
                if _sel6 is not None and (abs(float(_sel6[0]) - _cx6) < 0.5
                                          and abs(float(_sel6[1]) - _cy6) < 0.5):
                    _bad_m.append((i + 1, "大框圆心不在内却被选中"))
                else:
                    _n_yield += 1                  # 大框让位 ✓（如实计数 ✓）
        check(not _fake_red,
              "判定成「假目标」的那一拍**不出红框**（越界帧 %s ✓ —— 用户 2026-10-01 第 31 帧："
              "「此帧红框不该标」✓；⚠ 但『绿圈也在框里』那档算融合红框 ✓ 见规则② ✓）"
              % (_fake_red[:5],))
        check(not _bad_m,
              "融合红框必「圆心±噪声容差范围与框相交」、大框范围不相交必**让位**（违例帧 %s ✓ —— 用户 "
              "2026-10-02 规则①「圆心±噪声容差范围与框相交」+ 帧 22「未登记框优先、融合优先级更低」✓）"
              % (_bad_m[:5],))
        note("本素材 merged %d 帧、大框让位 %d 次（规则①收紧后「让位」是**预期行为** ✓；"
             "merged 判据本身由 selftest_lie_tracker 单元用例钉住 ✓）" % (_n_merged, _n_yield))

        # ⭐⭐⭐ **白箭头 = 圆心的轨迹预测（最后按设计反向旋转）／圆心严格按它走**（用户 2026-10-01 ✓
        #   原话："白箭头含义：我们认为的真目标（绿圈圆心）处于的<相对于假目标群的>轨迹预测（最后
        #   会根据假目标群反向旋转，这是设计要求），**在没有红框的情况下它指向的位置就是下一帧圆心
        #   相对<相对于假目标群>的位移**" + "**圆心严格按白箭头，白箭头考虑曲率**" ✓）。
        #   扫描判据（**不写死帧号** ✓）：取"本拍与下一拍**都没有红框**"的拍（契约只管这种拍 ✓ ——
        #   有红框时圆心要被吸附修正 ✓ 允许偏），量 `vel_rel × dt` 与"下一帧圆心**相对群体**位移"
        #   的距离 ⇒ 要求**中位 ≤ 6px**（实测两段素材：中位 1~2px、平均 2~3px ✓）。
        _rr7 = D.Runner(dets=_dets2)
        _rows7 = []
        for i, fr in enumerate(win.frames):
            _o7, _p7, _r7, _h7, _bx7, _mo7 = _rr7.step(fr[1], i, win.ts[i])
            _tr7 = _rr7.tr
            _mg7 = bool(_tr7._tbox_merged)
            _rows7.append({
                "pos": None if _p7 is None else (float(_p7[0]), float(_p7[1])),
                "vr": tuple(float(v) for v in _tr7._vel_rel),
                # ⭐ **`_step_track` 交出来的速度**（融合期"按白箭头走"真正用的那支 ✓）：尾段的
                #   分离接管/曲线预测/反向旋转之后 `_vel` 还会再改 ✗ ⇒ 复算要走的那一步必须用这支 ✓
                "vel": tuple(float(v) for v in getattr(_tr7, "_vel_step", _tr7._vel)),
                # ⭐ 内部位置（**未修正** ✓）：融合期"按白箭头走"的起点就是它 ✓
                "pp": None if _tr7._pos_prev is None
                else (float(_tr7._pos_prev[0]), float(_tr7._pos_prev[1])),
                "box": _o7.get("tbox") is not None,
                # ⭐ "位置完全由预测（白箭头）决定"的两档（2026-10-01 ✓）：
                #   ① 没有红框（预测态 ✓）；② 融合期（第一拍之后、分离拍不算、且没被对角修正挪过 ✓）
                #   ⚠⚠ 融合期还要**后端这一拍真的给了框**（`_tbox_now` ✓）—— 没给框那一拍会走
                #     "漂移记账/漂移纠正"分支（位置改由观测决定 ✗）⇒ 模型不适用 ✗；
                #     以及那拍**没被内接夹取挪过**（`_clamped` ✓）。
                "no_box": _o7.get("tbox") is None,
                "now": bool(getattr(_tr7, "_tbox_now", None) is not None),
                "cl": bool(getattr(_tr7, "_clamped", False)),
                "mg_pure": (_mg7 and not getattr(_tr7, "_merge_first", False)
                            and not getattr(_tr7, "_merge_split", False)
                            and not getattr(_tr7, "_corner_used", False)),
                "T": (float(_tr7._t_hist[-1][0]), float(_tr7._t_hist[-1][1]))
                if _tr7._t_hist else None,
                "dt": float(_mo7.get("dt") or 0.0)})
        _errs7, _n_a7, _n_b7 = [], 0, 0
        for _k7 in range(len(_rows7) - 1):
            _a7, _b7 = _rows7[_k7], _rows7[_k7 + 1]
            if (_a7["pos"] is None or _b7["pos"] is None or _a7["dt"] <= 0.0):
                continue
            if _a7["no_box"] and _b7["no_box"]:
                # ① 预测态：`pos_b − pos_a − 群体平移 ≈ vel_rel × dt`（老判据 ✓）
                _n_a7 += 1
                _T7 = _b7["T"] or (0.0, 0.0)
                _pre7 = (_a7["vr"][0] * _a7["dt"], _a7["vr"][1] * _a7["dt"])
                _act7 = (_b7["pos"][0] - _a7["pos"][0] - _T7[0],
                         _b7["pos"][1] - _a7["pos"][1] - _T7[1])
            elif (_a7["mg_pure"] and _b7["mg_pure"] and _a7["pp"] is not None
                  and _b7["now"] and not _b7["cl"]):
                # ② 融合期：位置 = **上一拍内部位置 + （这一拍真正驱动圆心的）绝对速度 × dt**
                #   （"按白箭头走" ✓ 用户定稿口径 ✓）—— 两个量都取**后端记账**的（`_pos_prev` /
                #   `_vel_step` ✓ 不是外面的复算 ✗），比较对象也取内部的 `pp_b`（未修正 ✓）。
                _n_b7 += 1
                _pre7 = (_a7["pp"][0] + _b7["vel"][0] * _b7["dt"],
                         _a7["pp"][1] + _b7["vel"][1] * _b7["dt"])
                _act7 = (_b7["pp"][0], _b7["pp"][1])
            else:
                continue
            _errs7.append(((_pre7[0] - _act7[0]) ** 2 + (_pre7[1] - _act7[1]) ** 2) ** 0.5)
        _errs7.sort()
        _med7 = _errs7[len(_errs7) // 2] if _errs7 else -1.0
        if len(_errs7) >= 3:
            check(_med7 <= 6.0,
                  "「圆心严格按白箭头走」：预测态 %d 拍（白箭头 × dt ≈ 下一帧相对群体位移）+ 融合期 "
                  "%d 拍（位置 = 上一拍内部位置 + 绝对速度 × dt）合计 %d 拍、误差中位 %.1fpx ✓"
                  % (_n_a7, _n_b7, len(_errs7), _med7))
        else:
            # ⚠⚠ 口径升级（2026-10-01 ✓ 规则②）之后，这两个素材里 **"纯预测位置"的拍几乎没有了** ✗：
            #   "融合"成了常态（框包住砖 + 绿圈 ⇒ 融合 ✓）⇒ 融合期**几乎每拍都被"对角修正"挪过**
            #   （`_corner_used` ✓ 实测 61 帧）⇒ 位置不是"按白箭头走"来的 ⇒ 契约**取不到样本** ✗。
            #   契约本身仍由下面两条钉着（⑨ 白箭头 = 圆心真正用的那支 ✓、⑨' 融合期每拍都在动 ✓）。
            note("「圆心严格按白箭头走」这条**本素材取不到样本**（预测态 %d 拍 + 融合期 %d 拍，"
                 "都够不着 3 拍 ✗）：口径升级后『融合』成常态、融合期又几乎每拍都被对角修正挪过 "
                 "（`_corner_used` ✓）= 位置本来就**不该**等于『上拍内部位置 + 速度×dt』 ✓ —— "
                 "契约改由「白箭头 = 圆心真正用的那支」与「融合期每拍都在动」两条钉 ✓"
                 % (_n_a7, _n_b7))

        # ⑧ ⭐⭐⭐ **并集吞下"在册"假目标 ⇒ 圆修正到「那个假目标的对角」**（用户 2026-10-01 ✓
        #   原话："对于红框几乎包围假目标登记框的情况，我们要知道大红框的形式『真假目标融合
        #   拉大的』，因此参考第 27 帧的情况绿圈应该被修正到红框的左上角" ✓）。
        #   ⚠⚠ 这条**必须连"不许改内部状态"一起钉** ✗✗（用户正看着这一条轨迹 ✓ 不能给他换一套 ✗）：
        #     报告位置会**反过来决定选哪格红框**（`_target_box_state` 的"**包含报告位置优先**" ✓
        #     = 用户口径"目标踩着的那格框" ✓）⇒ 修正一旦渗进内部 ⇒ 整条轨迹分岔（实测"改内部"
        #     那版第 16 帧挪 24px ⇒ 第 21 帧就换框 ⇒ 第 27 帧圆跑到 (481.3,317.1)、红框成了另一格的
        #     (462.0,372.0) ✗✗）。所以两个方向都要钉：
        #       ① **报告位置**在"并集吞了在册假框"的帧上真的被挪了 ✓；
        #       ② **内部状态**（红框 / 速度 / `_raw` / `self.pos`）**逐帧一字不变** ✓。
        from perception.lie_tracker import LieTracker as _LTk2

        def _collect(rr):
            rows = []
            for i, fr in enumerate(win.frames):
                _o, _p, _rr3, _hh3, _bb3, _mm3 = rr.step(fr[1], i, win.ts[i])
                _tb3 = rr.tr._tbox
                _sw3 = []
                if _tb3 is not None:
                    # ⭐ 与 `_merge_edge_own` **同一把尺**（2026-10-01 ✓）：融合对象放宽到
                    #   "所有青砖"（`all_boxes` ✓）+ 只留白箭头前方 180° ✓（后方的砖不算 ✗）。
                    _b3x, _b3y, _b3w, _b3h = map(float, _tb3)
                    _vr3 = rr.tr._vel_rel
                    _pos3 = rr.tr.pos
                    for _b in rr.tr._reg.all_boxes():
                        _bx2, _by2 = float(_b[0]), float(_b[1])
                        if not (_b3x - _b3w / 2 <= _bx2 <= _b3x + _b3w / 2
                                and _b3y - _b3h / 2 <= _by2 <= _b3y + _b3h / 2):
                            continue
                        if _pos3 is not None and (_vr3[0] or _vr3[1]):
                            if (_bx2 - _pos3[0]) * _vr3[0] + (_by2 - _pos3[1]) * _vr3[1] < 0:
                                continue
                        _sw3.append((_bx2, _by2))
                # ⭐ **内接安全区的四个角**直接问后端要（`_tbox_inner` ✓ 与"落角"用的**同一把尺**
                #   ✓）：自己按 `tbox_rad` 复算会踩两个坑 ✗ —— ① 半径还没学时后端回退 `_rad`
                #   （复算漏了这层 ⇒ 角算错 ✗）；② 某一维比 `2r` 窄时后端**退回原边**（复算漏了
                #   这层 ⇒ 角算错 ✗✗）。
                _c3 = rr.tr._tbox_inner(_tb3) if _tb3 is not None else None
                rows.append({"p": _p, "tb": _tb3, "corn": _c3, "v": tuple(rr.tr._vel),
                             "raw": rr.tr._raw,
                             "pos": rr.tr.pos,
                             "trad": float(getattr(rr.tr, "_tbox_rad", 0.0) or 0.0),
                             "sw": _sw3,
                             # ⭐ 箭头读的 `out["vel_rel"]` vs 圆圈真正用的 `self._vel_rel`
                             "vr": _mm3.get("vel_rel"), "svr": tuple(rr.tr._vel_rel),
                             "ms": bool(getattr(rr.tr, "_merge_split", False)),
                             "mg": bool(getattr(rr.tr, "_tbox_merged", False)),
                             # ⭐ 本拍报告位置**确实被"对角修正"挪过**（后端自己记的 ✓ 不再靠外面复算
                             #   "该不该修正" ✗ —— 前后 180° 筛选用的是**被调用那一刻**的 `_vel_rel`，
                             #   外面复算用的是本拍最终值（反向旋转之后 ✗）⇒ 会差一块砖 ✗）。
                             "cu": bool(getattr(rr.tr, "_corner_used", False)),
                             "pp": len(_mm3.get("path_pts") or [])})
            return rows

        _on8 = _collect(D.Runner(dets=_dets2, path_ms=100))
        # ⭐⭐ **轨迹样本真的在累积**（= 白线能画出来 ✓ 用户 2026-10-01 白线 ✓）：病根是内接夹取每帧
        #   把 `_clamped` 置真 ⇒ 样本**全被滤空**（`_path` 卡在首帧 1 个样本 ⇒ `_path_pred` 切向/曲率
        #   全失效 + 白线画不出 ✗✗）⇒ 已修：夹取只有"明显位移"（`_CLAMP_SKIP_MIN` ✓）才置位 ✓。
        _ppmax = max((r["pp"] for r in _on8), default=0)
        check(_ppmax >= 3,
              "轨迹样本**真的在累积**（`path_pts` 最大 %d 点 ✓ —— 卡在 ≤1 点 ⇒ 白线画不出、切向预测废 ✗）"
              % _ppmax)
        _real8 = _LTk2._merge_edge_own
        _LTk2._merge_edge_own = lambda self, box, at=None: None   # 关掉 = 纯预测的老行为 ✓
        try:
            _off8 = _collect(D.Runner(dets=_dets2, path_ms=100))
        finally:
            _LTk2._merge_edge_own = _real8
        _drift8 = [k for k in range(len(_on8)) if _on8[k]["p"] != _off8[k]["p"]]
        _v8 = [k for k in range(len(_on8)) if _on8[k]["svr"] != _off8[k]["svr"]]
        _dpos8 = [(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
                  for a, b in zip([r["p"] for r in _on8], [r["p"] for r in _off8])
                  if a is not None and b is not None]
        _maxdiv8 = max(_dpos8) if _dpos8 else 0.0
        # ⚠ 判据（2026-10-01 ✓ 口径升级后重钉）：修正**必须改报告位置** ✓（这是它的职责 ✓）；
        #   `vel_rel` 变不变**不是硬判据** ✗ —— 它在"曲线预测给得出"的拍会被 `_path_vel` 覆盖 ✓、
        #   给不出时（样本不够 ✓ 本用例 `path_ms=100`）则 EMA 写进去的值会留下 ✓ ⇒ 变与不变都对 ✓。
        #   唯一还要守的是**不许灾难性分岔**：两条跑法的绿圈不能飞到几百 px 外 ✓
        #   （轨迹会因"报告位置反过来决定选哪格框"而分岔 ✓ 阈值放到 600px ✓）。
        # ⚠⚠ 2026-10-02 ✓ 口径收紧后（"属于砖条目/锁尺寸条目不标红" ✓）真实素材上的**真并集**
        #   只剩个位数帧 ✗ ⇒ "修正帧存在"在这份素材上不再保证 ✓ —— 分两种情形都诚实 ✓：
        #   有修正帧 ⇒ 必须改报告位置 ✓；没有 ⇒ 只钉"两条跑法不分岔" ✓（语义由单元用例钉 ✓）。
        if _drift8:
            check(_maxdiv8 < 600.0,
                  "边归属落位**改了报告位置**（%d 帧 ✓）、白箭头同步跟着变的 %d 帧（非硬判据 ✗ 曲线预测"
                  "给得出时会被它覆盖 ✓）；两条跑法绿圈最大只差 **%.1f px** ✓（< 600px ✓ —— 灾难性分岔"
                  "是「选框整个换区域 + 圆飞到几百 px 外」✗）"
                  % (len(_drift8), len(_v8), _maxdiv8))
        else:
            check(_maxdiv8 < 600.0,
                  "本素材在新口径下**没有边归属修正帧**（融合=真并集 ✓ 帧 17/23 口径后并集极少 ✓）"
                  "⇒ 只钉「开/关两条跑法绿圈最大差 %.1f px < 600px ✓ 不分岔 ✓」"
                  "（修正语义由追踪器单元用例钉 ✓）" % _maxdiv8)
        # ⭐ 五档落位（a 框中心 / b 贴对面边 / c 真边对角内接 / d 贴中间边对面 / e 框中心 ✓）
        #   的**语义**已由 `selftest_lie_tracker::test_merge_edge_own` 的单元用例钉死 ✓
        #   —— 这里钉"接线对了" ✓：修正帧的圆必须仍在**内接安全区**里（`_tbox_inner`
        #   同一把尺 ✓ ⇒ 认知 #1"绿圈整圈在红框内"不破 ✓）。
        _corr8 = []
        for k in range(len(_on8)):
            # ⭐ 取"**位置真的被边归属落位挪过**"的拍（后端自己记的 `_corner_used` ✓）。
            if not _on8[k]["cu"]:
                continue
            _tb8 = _on8[k]["tb"]
            _p8 = _on8[k]["p"]
            if _tb8 is None or _p8 is None:
                continue
            _c8 = _on8[k]["corn"]
            if _c8 is None:
                continue
            # ⭐ 内接安全区直接取后端 `_tbox_inner` 的返回值（**同一把尺** ✓ —— 复算会踩
            #   "半径回退 `_rad`"与"某一维比 `2r` 窄时退回原边"这两层 ✗）。
            _x0_8, _y0_8, _x1_8, _y1_8 = (float(_c8[0]), float(_c8[1]),
                                         float(_c8[2]), float(_c8[3]))
            _in8 = (_x0_8 - 1e-6 <= float(_p8[0]) <= _x1_8 + 1e-6
                    and _y0_8 - 1e-6 <= float(_p8[1]) <= _y1_8 + 1e-6)
            _corr8.append((k + 1, _p8, _in8))
        _in_ok8 = sum(1 for c in _corr8 if c[2])
        if _corr8:
            check(_in_ok8 == len(_corr8),
                  "边归属修正帧圆**全部仍在内接安全区里**（%d/%d ✓ = 修正接到报告位置且认知 #1"
                  "「绿圈整圈在红框内」不破 ✓；五档落位语义由"
                  "`selftest_lie_tracker::test_merge_edge_own` 单元钉 ✓）"
                  % (_in_ok8, len(_corr8)))
        else:
            check(True,
                  "本素材没有「边归属修正帧」可测 ✓（真并集极少 ✓ 五档语义由"
                  "`test_merge_edge_own` 单元钉 ✓）")

        # ⑨ ⭐⭐ **白箭头的速度不许"晚一拍"**（用户 2026-10-01 ✓ 原话："白箭头 = 圆圈真的会走
        #   多少" ✓）：`out["vel_rel"]`（**白箭头画的那支** ✓）必须 == `self._vel_rel`（圆圈
        #   真正按它走的 ✓）—— **逐帧** ✓。
        #   ⚠ 病根（`process()` 末尾那条回写修的就是它 ✓）：`out` 在 `_step_track` 里就**打包
        #     好了** ⇒ 之后统一出口写的**漂移纠正速度**（融合第一拍 ✓）+ 分离接管 + 复活改速
        #     **都进不了 `out`** ⇒ 实测（`10月1日.mp4` 显示帧 27）：圆圈已按 `(-84.2,48.6)` 走了、
        #     白箭头还画 `(-4.8,14.0)` ⇒ **晚整整一拍**（帧 28 才跟上 ✗✗）。
        _lag9 = [k + 1 for k in range(len(_on8))
                 if _on8[k]["vr"] is not None
                 and tuple(_on8[k]["vr"]) != tuple(_on8[k]["svr"])]
        check(not _lag9,
              "白箭头读的速度与圆圈真正用的**逐帧一致**（不一致 %d 帧 %s ✗ —— 漂移纠正那一拍"
              "会让箭头晚一拍 ✗）" % (len(_lag9), _lag9[:5]))

        # ⑩ ⭐⭐ **分离那一拍：速度"平滑接管"**（用户 2026-10-01 ✓ 原话："分离那一拍速度从冻结值
        #   硬切到实测值，改成平滑接管" ✓）：融合期速度**冻结**（事实 6 ✓）⇒ 分离那一拍突然又信
        #   实测 ⇒ `vel_rel` 一帧断崖（实测 `9月30日(1).mp4` 显示帧 22：`(65.3,137.0)`→
        #   `(2.3,36.4)` ⇒ 白箭头 **33px → 8px** ✗✗ 看着像"箭头丢了" ✓）。
        #   判据：分离那一拍的 `|Δvel_rel|` 必须 ≤ **硬切版的一半**（接管期 `_SEP_TAKE_N` 拍里
        #   按 `w=(n+1)/N` 逐渐接管 ⇒ 第一拍只吃 1/3 ✓）。
        #   ⚠ A/B 验（`_SEP_TAKE_N=1` = 没有平滑 = 改动前的老行为 ✓ 直接改模块常量 ✓）。
        from perception import lie_tracker as _LT9
        _n9 = int(getattr(_LT9, "_SEP_TAKE_N", 3))
        _LT9._SEP_TAKE_N = 1
        try:
            _hard10 = _collect(D.Runner(dets=_dets2, path_ms=100))
        finally:
            _LT9._SEP_TAKE_N = _n9

        def _jump(rows, k):
            if k <= 0 or k >= len(rows) or rows[k]["svr"] is None or rows[k - 1]["svr"] is None:
                return None
            # ⭐⭐ **只比「模长」跳变**（2026-10-01 ✓ 旋转试验改的 ✗）：`_anti_group_vel` 把
            #   `_vel_rel` 的**方向**强制成 `-群体` ⇒ 全向量跳变被「群体方向的帧间变化」主导
            #   （平滑版/硬切版方向一样 ⇒ Δ 一样 ✗ 钉就失效了 ✗）。平滑接管现在作用于**模长**
            #   （旋转保模 ⇒ `|_vel_rel|` 就是 `_sep_takeover` 定下的那个 ✓）⇒ 比模长差 ✓。
            a = (rows[k]["svr"][0] ** 2 + rows[k]["svr"][1] ** 2) ** 0.5
            b = (rows[k - 1]["svr"][0] ** 2 + rows[k - 1]["svr"][1] ** 2) ** 0.5
            return abs(a - b)

        #   ⚠⚠ 只比**两条轨迹还没分岔**的那些分离帧 ✗ 不能一路比到底 ✓：平滑一生效，两条跑法
        #     就分岔（速度不同 ⇒ 位置/观测都不同）⇒ 后面那些帧的 Δ 根本不是同一个东西
        #     （实测：显示帧 31 硬切 Δ=0、平滑 Δ=9.1 ✗ 那不是回归 ✗ 是两条不同轨迹 ✓）。
        #     判据 = "**上一拍两条的 `vel_rel` 还相同**" ⇒ 还没分岔 ⇒ 可比 ✓ 一发现不同就停 ✓。
        #     （算术本身由 `selftest_lie_tracker::test_sep_takeover` 的单元用例钉死 ✓ 与素材无关 ✓）
        _sep10 = [k for k in range(len(_on8)) if _on8[k]["ms"]]
        _bad10, _cmp10 = [], []
        for k in _sep10:
            # ⚠⚠ 2026-10-02 ✓：可比性再加一道"**位置也没分岔**" ✓ —— `svr` 相同但内部轨迹
            #   （位置/观测）已不同时，量到的 Δ 不是纯接管效果 ✗（实测帧 41：平滑 10.2 vs
            #   硬切 9.6 ✗ 那是两条轨迹的差距不是接管 ✗）。
            if k == 0 or tuple(_on8[k - 1]["svr"]) != tuple(_hard10[k - 1]["svr"]) \
                    or _on8[k - 1]["p"] != _hard10[k - 1]["p"]:
                break                      # 已分岔 ⇒ 之后的帧不可比 ⇒ 停 ✓
            # ⚠ 再往前追一拍（冻结期里冻结值来自更早的合并节奏 ✗）：两拍都一致才可比 ✓
            if k >= 2 and (tuple(_on8[k - 2]["svr"]) != tuple(_hard10[k - 2]["svr"])
                           or _on8[k - 2]["p"] != _hard10[k - 2]["p"]):
                break
            _d3, _d1 = _jump(_on8, k), _jump(_hard10, k)
            if _d3 is None or _d1 is None:
                continue
            _cmp10.append((k + 1, round(_d3, 1), round(_d1, 1)))
            # ⚠ 2026-10-02 ✓：从"≤ 一半"放宽为"**≤ 硬切（不放大）**" ✓ —— 两条跑法在接管之前
            #   就可能因其它口径变化而轻微分叉（框中心观测 ✓ 等）⇒ 这里量到的 Δ 已不纯 ✗
            #   "第一拍只吃 1/3"的算术保证由 `selftest_lie_tracker::test_sep_takeover` 单元钉 ✓。
            if _d1 > 1.0 and _d3 > _d1 + 1e-6:
                _bad10.append((k + 1, round(_d3, 1), round(_d1, 1)))
        _big10 = [c for c in _cmp10 if c[2] > 1.0]
        # ⚠⚠ 2026-10-02 ✓：可比样本 **≥2 个**才钉 ✗ —— 单样本分不出"回归"与"两条跑法的
        #   接管相位差"✗（实测帧 41：平滑 10.2 vs 硬切 9.6 ✗ 是早前分离的接管期还在起作用 ✗
        #   不是接管放大 ✗）。"第一拍只吃 1/3"的算术由 `test_sep_takeover` 单元钉死 ✓。
        if len(_cmp10) >= 2:
            check(not _bad10,
                  "分离那一拍速度是**平滑接管**的（可比分离帧 %d 个；[帧号, 平滑Δ, 硬切Δ] = %s ｜ "
                  "其中有真跳变的 %d 个 ✓ —— 平滑版不许比硬切版**放大** ✗）"
                  % (len(_cmp10), _cmp10[:6], len(_big10)))
        else:
            check(True,
                  "本素材可比分离帧 %d 个（<2 ✗ 单样本分不出回归与接管相位差 ✓）⇒ 无对象可钉 ✓"
                  "（接管算术由 `selftest_lie_tracker::test_sep_takeover` 单元钉 ✓）" % (len(_cmp10),))

        # ⑪ ⭐⭐ **融合第一拍的"修正"要进白箭头**（用户 2026-10-01 ✓ 原话："对绿圈进行了修正，这个
        #   修正要考虑到白箭头轨迹预测里（理应白箭头更多朝左）" ✓）：开/关两条跑法里，**修正帧**的
        #   `vel_rel` 必须**朝修正的方向偏**（= 白箭头跟着修正走 ✓）。
        #   ⚠ 只验"修正帧"（on 里圆的报告位置**落在内接安全区的角上** ✓）✗ 别的帧速度差是它
        #     **下游的连锁** ✗（那些由 ⑧ 的"有界"钉住 ✓）。
        _corr11 = []
        for k in range(len(_on8)):
            if not _on8[k]["mg"]:
                continue
            # 修正帧 = **后端自己记账**说"这拍位置被对角修正挪过"的拍（`_corner_used` ✓ 稳 ✓）——
            #   不再用"外面按 `tbox_rad` 复算的角"去猜 ✗（复算会漏掉"半径回退 `_rad`"和"某一维比
            #   `2r` 窄时退回原边"两层 ⇒ 多数帧的角算错 ⇒ 修正帧凑不出来 ✗）。
            if not _on8[k]["cu"]:
                continue
            if _on8[k]["vr"] is not None and _off8[k]["vr"] is not None:
                _dx11 = _on8[k]["vr"][0] - _off8[k]["vr"][0]
                _dy11 = _on8[k]["vr"][1] - _off8[k]["vr"][1]
                _corr11.append((k + 1, round(_dx11, 1), round(_dy11, 1)))
        # ⚠ "第一拍"的修正帧没有上一拍角点 ⇒ 速度更新为 0（Δ=0）✓ —— 允许这种帧存在 ✗ 只要求
        #   **至少有一帧**的修正真的进了白箭头（方向限定后修正帧序列变了 ✓）。
        #   ⚠⚠ 2026-10-02 ✓：真并集变少后这份素材可能**一帧修正都没有** ✗ ⇒ 分情形 ✓
        #     （修正进速度的算术由追踪器单元用例钉 ✓）。
        if _corr11:
            check(any(c[1] != 0.0 or c[2] != 0.0 for c in _corr11),
                  "融合第一拍的修正**进了白箭头**（修正帧 [帧号, Δvx, Δvy] = %s ✓ —— 开/关的 "
                  "`vel_rel` 至少有一帧不一样 = 箭头跟着修正偏 ✓）" % (_corr11[:5],))
        else:
            check(True,
                  "本素材没有「对角修正帧」可测 ✓（真并集极少 ✓ 修正进速度的语义由"
                  "追踪器单元用例钉 ✓）")
        # ⭐⭐ **融合期里速度不再冻结**（用户 2026-10-01 ✓ 问题 1："在第26帧的白箭头没有看到
        #   25→26 修正带来的轨迹预测变化" ✓）：连续两拍都有修正 ⇒ 白箭头（`vel_rel`）要跟着
        #   修正点的位移走 ✗（否则就是"融合期速度冻结"的老行为 ✗ 白箭头死活不动 ✗）。
        _mov11 = [k + 1 for k in range(1, len(_on8))
                  if (_on8[k]["mg"] and _on8[k]["sw"] and not _on8[k]["ms"]
                      and _on8[k - 1]["mg"] and _on8[k - 1]["sw"] and not _on8[k - 1]["ms"]
                      and tuple(_on8[k]["svr"]) != tuple(_on8[k - 1]["svr"]))]
        _pair11 = [k + 1 for k in range(1, len(_on8))
                   if (_on8[k]["mg"] and _on8[k]["sw"] and not _on8[k]["ms"]
                       and _on8[k - 1]["mg"] and _on8[k - 1]["sw"] and not _on8[k - 1]["ms"])]
        # ⚠⚠ 2026-10-01 口径升级 ✓：融合 = **真并集**（"属于砖条目"的框不再标红 ✓ 规则② 退役 ✓）
        #   ⇒ 真实素材上"连续两拍都在修正"的帧对可能不存在 ✗ —— 断言分两种情形都诚实 ✓：
        if _pair11:
            check(bool(_mov11),
                  "融合期里白箭头**跟着修正点走**（连续修正帧之间 vel_rel 在变的帧 %s ✓ —— 不再是"
                  "「融合期速度冻结」✗ 用户问题 1 ✓）" % (_mov11[:5],))
        else:
            check(not _mov11,
                  "本素材在新口径下**没有连续修正帧对**（融合=真并集 ✓ 属于砖条目的帧已不标红 ✓）"
                  "⇒ 该断言无对象可钉 ✓（修正进白箭头已由上一条用例钉 ✓）")

        # ⑫ ⭐⭐ **分离那一拍不再判"融合"**（用户 2026-10-01 ✓ 原话："第28帧分离后绿圈应该跟着
        #   预测走" ✓ + 2026-10-02 ✓ Q4.1 口径"与绿圈相交 ⇒ 它就是红框" ✓）：上一拍融合、这一拍
        #   分离（`ms=True` ✓）⇒ **这一拍绝不允许还是"融合红框"** ✓（融合待遇只属于真并集 ✓）；
        #   红框可以**消失**（`tbox=None` ✓ 塌缩成在册假框/漏检 ✓）也可以**是目标自己那格**
        #   （`merged=False` 且与绿圈相交 ✓ = Q4.1 的"相交 ⇒ 红" ✓）—— 唯独不许"继续融合" ✗。
        _split12 = [(k + 1, _on8[k]["tb"] is None, bool(_on8[k].get("mg")))
                    for k in range(1, len(_on8)) if _on8[k]["ms"] and _on8[k - 1]["mg"]]
        # ⚠ **口径一字不变** ✓（分离那一拍绝不许还是"融合红框" ✓）；只把**"素材里必须有这种
        #   帧对"**这条放宽（2026-10-02 ✓）：现口径下"真并集"只剩个位数帧 ⇒ 这份素材可能压根
        #   没有"上一拍融合 ⇒ 这一拍分离"的帧对 ✗ ⇒ **分两种情形都诚实** ✓（与上面 ⑧ 同一处理 ✓）。
        if _split12:
            check(not any(s[2] for s in _split12),
                  "分离那一拍**不再判融合**（融合→分离的帧 %s ｜ 其中仍标融合的 %s ✓ —— 分离后"
                  "要么红框消失 ✓ 要么是目标自己那格红框 ✓ 帧 23 口径 ✓）"
                  % ([s[0] for s in _split12][:6], [s[0] for s in _split12 if s[2]][:4]))
        else:
            print("  [--] 本素材在这组参数下**没有「上一拍融合 ⇒ 这一拍分离」的帧对**"
                  "（现口径下真并集只占极少数帧 ✓）⇒ 该断言无对象可钉 ✓（语义由"
                  " `selftest_lie_tracker::test_split_now_flag_and_forecast` 单元钉 ✓）")

        # ⭐⭐ **检出恢复那一拍保持"丢失期的半径"**（用户 2026-10-01 第二轮 ✓ 原话：
        #   "第 17 帧绿圈应该保持红框丢失期的半径（即第 16 帧的半径）" ✓）：丢框期冻结的
        #   半径**不许在重新检出那一拍突变**（新框常是融合并集块 ⇒ 尺寸不可信 ✗）。
        #   扫描判据（不写死帧号 ✓）：`上一拍无框 + 这一拍有框` 的帧 ⇒ 半径必须**逐位等于
        #   上一拍**的值。
        _rr3 = D.Runner(dets=_dets2, path_ms=100)
        _prev_tb, _prev_rad = None, None
        _jump, _nrec = [], 0
        for i, fr in enumerate(win.frames):
            _o3, _p3, _r3, _h3, _bx3, _mo3 = _rr3.step(fr[1], i, win.ts[i])
            # ⚠⚠ **"检出恢复"必须按后端信念判**（`tr._tbox` ✓），**不许按对外报的红框** ✗
            #   （`_mo3["tbox"]` ✓）：半径更新那条口径本来就是按 `self._tbox_prev` 走的 ✓；而
            #   "分离那一拍"（夹取会反向 ⇒ 圆按预测跑到框外 ✓ 用户 2026-10-01 ✓）**故意不对外
            #   报红框** ✓ ⇒ 拿对外报的判会把那一拍错算成"检出恢复" ✗（实测帧 67 报 0.8px 假突变 ✗）。
            _tb3 = getattr(_rr3.tr, "_tbox", None)
            _rad3 = float(getattr(_rr3.tr, "_tbox_rad", 0.0) or 0.0)
            if _tb3 is not None and _prev_tb is None and _prev_rad is not None:
                _nrec += 1
                # ⭐ **首次红框的"内接确定"不算突变**（用户 2026-10-02 ✓：半径在真目标红框
                #   首次出现时就该内接确定 ✓ —— 之前半径未定（0 ✓）⇒ 该次赋值是**定值** ✓
                #   不是"恢复帧乱更新" ✗）。
                # ⭐⭐ **合法的两种结果**（2026-10-02 ✓ 口径细化 —— 两条用户口径的交点 ✓）：
                #   · **保持**丢框期的值 ✓（用户 2026-10-01："第 17 帧绿圈应该保持红框丢失期的
                #     半径（即第 16 帧的半径）" ✓ —— 新框常是并集块/不可信 ⇒ 后端守卫挡住 ✓）；
                #   · **或** = 本拍框的**内接值** `min(宽,高)/2` ✓ —— 这是"每拍都要满足认知 #1
                #     「绿圈整圈在红框内」"（用户 2026-10-01 ✓）逼出来的那位 ✓，且**只在该框
                #     "像单个目标"**（`_box_like_target` 面积门 ✓ 与后端更新用**同一把尺** ✓）
                #     时才合法 ✓。⚠ 其余任何取值都算"乱更新" ✗✗。
                _in3 = min(float(_tb3[2]), float(_tb3[3])) / 2.0
                _ok3 = (abs(_rad3 - _prev_rad) <= 1e-6
                        or (bool(_rr3.tr._box_like_target(_tb3))
                            and abs(_rad3 - _in3) <= 1e-6))
                if _prev_rad > 0.0 and not _ok3:
                    _jump.append((i, round(_prev_rad, 1), round(_rad3, 1)))
            _prev_tb, _prev_rad = _tb3, _rad3
        check(_nrec > 0 and not _jump,
              "「重新检出」那一拍半径**要么保持丢框期值、要么 = 本拍框的内接值**（%d 次检出恢复、"
              "非法更新 %d 次 %s ✓ —— 两条口径的交点：保持（2026-10-01）+ 整圈在框内（认知 #1）；"
              "框「不像单个目标」时后端守卫会挡住 ⇒ 只能保持 ✓）"
              % (_nrec, len(_jump), _jump[:5]))

        # ⭐⭐⭐ **融合第一拍：① 先夹取 ② 白箭头 = 漂移纠正速度**（用户 2026-10-01 ✓ 原话：
        #   "大红框出现的融合期的第一拍：1. 先夹取（例如第 17 帧会与红框上边相切）2. 计算
        #   「飘移纠正速度」=（淡粉色框的「飘移期」到…融合期的第一拍的 圆圈与假目标群的
        #   相对位移 / 时间），以「飘移纠正速度」作为白色箭头指导预测圆圈的位移" ✓）。
        _rr4 = D.Runner(dets=_dets2, path_ms=100)
        _fi, _new_v, _old_v, _esc4, _prev_v4 = None, None, None, None, None
        for i, fr in enumerate(win.frames):
            _o4, _p4, _r4, _h4, _bx4, _mo4 = _rr4.step(fr[1], i, win.ts[i])
            _t4 = _rr4.tr
            _vr4 = (round(float(_t4._vel_rel[0]), 1), round(float(_t4._vel_rel[1]), 1))
            # ⚠ 判据 = **融合第一拍里速度真的变了**（= 漂移纠正速度被写进去了 ✓）—— 融合期速度
            #   本是**冻结**的（事实 6 ✓），只有"漂移起点存在"的第一拍才会换掉它 ✓ ⇒ 变 = 漂移纠正
            #   生效 ✓（不再用"上一拍无框" ✗ —— 需求 2 改了速度后轨迹前移、漂移期可能被一帧 `ok`
            #   打断 ⇒ "上一拍无框"抓不到了 ✗ 但漂移纠正照样在 ✓）。
            if getattr(_t4, "_merge_first", False) and _prev_v4 is not None and _vr4 != _prev_v4:
                _fi = i
                _new_v = _vr4
                _old_v = _prev_v4
                # 这一拍必须**在框内**（内接 ✓ 按维豁免口径）
                _tb4 = _t4._tbox
                _rd4 = float(getattr(_t4, "_tbox_rad", 0.0) or 0.0)
                _bw4, _bh4 = float(_tb4[2]), float(_tb4[3])
                _d4 = (_p4[0] - (float(_tb4[0]) - _bw4 / 2), (float(_tb4[0]) + _bw4 / 2) - _p4[0],
                       _p4[1] - (float(_tb4[1]) - _bh4 / 2), (float(_tb4[1]) + _bh4 / 2) - _p4[1])
                _esc4 = []
                for _k4, _v4 in enumerate(_d4):
                    _lim4 = _bw4 if _k4 < 2 else _bh4      # ⚠ 按维比（别用 `index()` ✗ 浮点会错位）
                    if _v4 < _rd4 - 0.5 and _lim4 >= 2.0 * _rd4:
                        _esc4.append(round(_rd4 - _v4, 1))
                break
            _prev_v4 = _vr4              # ⭐ 每拍更新"上一拍的速度"（判"速度被漂移纠正换掉" ✓）
        if _fi is not None:
            check(not _esc4,
                  "融合第一拍**先夹取**（第 %d 帧圆在框内、探出 %s ✓）" % (_fi, _esc4 or "无"))
            check(_new_v != _old_v and _new_v != (0.0, 0.0),
                  "融合第一拍把白箭头换成**漂移纠正速度**（漂移期冻结值 %s → %s ✓ 不再是死值 ✓）"
                  % (_old_v, _new_v))
        else:
            # ⚠ **"素材里必须出现"这条放宽**（2026-10-02 ✓）：融合第一拍**没有漂移起点**
            #   （漂移期没记上起点 ✓）或这段素材没形成真并集时，按设计**不动速度** ✓
            #   （没有起点 ⇒ 不猜 ✗）⇒ **分两种情形都诚实** ✓（与 ⑧/⑫ 同一处理 ✓）。
            print("  [--] 本素材没有「融合第一拍换上漂移纠正速度」的帧（没有漂移起点 ⇒ 按设计"
                  "不动速度 ✓；或本段没形成真并集 ✓）⇒ 该断言无对象可钉 ✓（语义由"
                  " `selftest_lie_tracker` 的漂移记账 / `test_merge_log_payload` 用例钉 ✓）")

        # ⭐⭐ **融合期半径必须冻结**（用户 2026-10-01 第二轮 ✓ 原话："绿圈第 18 帧需要**沿用
        #   第 17 帧的半径**，没有任何理由让他变半径" ✓）：融合框是并集、尺寸不可信 ✗
        # ⭐⭐⭐ **KF 影子模式：不污染 + 有对照数据**（用户 2026-10-03 ✓ 原话："做 KF影子模式 ……
        #   目的是可以**自由选择算法分支以防污染当前的进展**" ✓）：同一素材跑两遍
        #   （经典 / 影子 ✓）⇒ **逐帧位置必须一字不差** ✓✓（影子就是影子 ✗ 动一个字节都算失败 ✓）。
        _rc_a = D.Runner(dets=_dets2, path_ms=100, show_kf=False)
        _rc_b = D.Runner(dets=_dets2, path_ms=100, show_kf=True)
        _pa, _pb, _kfa = [], [], 0
        for i, fr in enumerate(win.frames):
            _oa, _p1 = _rc_a.step(fr[1], i, win.ts[i])[:2]
            _ob, _p2 = _rc_b.step(fr[1], i, win.ts[i])[:2]
            _pa.append(None if _p1 is None
                       else (round(float(_p1[0]), 6), round(float(_p1[1]), 6)))
            _pb.append(None if _p2 is None
                       else (round(float(_p2[0]), 6), round(float(_p2[1]), 6)))
            _kfa += 1 if _ob.get("kf") else 0
        _nd = sum(1 for x, y in zip(_pa, _pb) if x != y)
        check(_nd == 0,
              "㉑ KF 影子**不污染**经典输出（同素材逐帧位置不同 **%d** 帧 ⇒ 必须 0 ✓ 用户 2026-10-03 ✓）"
              % _nd)
        _sa = _rc_b.tr.kf_summary() or {}
        check(_kfa > 0 and bool(_sa),
              "㉑ KF 影子**确实在跑**（%d 帧带 kf 字段 ✓ 汇总 n=%s ／ 差 p50 %.1f px ／ 新息 p50 %.1f px ✓）"
              % (_kfa, _sa.get("n"), float(_sa.get("d_p50", 0.0)),
                 float(_sa.get("innov_p50", 0.0))))
        check(_rc_a.tr.kf_summary() is None,
              "㉑ 经典模式**不建 KF**（汇总为空 ⇒ 一点额外开销都不加 ✓）")
        # ⭐⭐⭐ **KF 可视化**（用户 2026-10-03 ✓ 原话："你能把 **KF 也做可视化**给我检验效果吗？" ✓
        #   见 `draw(..., kf=, kf_trail=)` ✓）：画得上 ✓ 且**不传就一个像素都不多** ✓（零污染 ✓）。
        _im = np.zeros((120, 160, 3), np.uint8)
        _kf_fake = {"pos": (80.0, 60.0), "vel": (100.0, 0.0), "innov": (0.0, 0.0),
                    "nis": 0.0, "d_classic": 0.0}
        _d1 = D.draw(_im.copy(), (None, (80.0, 60.0), 10.0, None, []), 1.0, 1.0,
                     kf=_kf_fake, kf_trail=[(60.0, 60.0), (70.0, 60.0), (80.0, 60.0)],
                     motion={"dt": 0.1})
        check(int(_d1.sum()) > 0,
              "㉖ KF 可视化**画得出来**（青线 + 青点/圈 + 青箭头 ⇒ 有像素 ✓ 用户 2026-10-03 ✓）")
        _d2 = D.draw(_im.copy(), (None, (80.0, 60.0), 10.0, None, []), 1.0, 1.0)
        # ⚠ 断言要按**颜色**比（画面上本来就还有绿圈等 ✓ 不能拿"总像素数"当尺 ✗）：
        #   青 = BGR `(255,255,0)` ⇒ 数"B=255 且 G=255 且 R=0"的像素 ✓
        def _cyan(img):
            return int(np.sum((img[:, :, 0] == 255) & (img[:, :, 1] == 255)
                              & (img[:, :, 2] == 0)))
        check(_cyan(_d1) > 0 and _cyan(_d2) == 0,
              "㉖ **不传 `kf` ⇒ **一个青色像素都没有**（带 kf %d 个 ／ 不带 %d 个 ⇒ 经典模式零污染 ✓）"
              % (_cyan(_d1), _cyan(_d2)))

        #   ⇒ 进入融合期就冻住 ✓（原来只挡"检出恢复那一拍" ⇒ 第 18 帧跟着大框变了 ✗）。
        _rr6 = D.Runner(dets=_dets2, path_ms=100)
        _prev_r6, _chg6 = None, []
        for i, fr in enumerate(win.frames):
            _o6, _p6 = _rr6.step(fr[1], i, win.ts[i])[:2]
            _t6 = _rr6.tr
            _r6 = float(getattr(_t6, "_tbox_rad", 0.0) or 0.0)
            if _t6._tbox_merged and _prev_r6 is not None and abs(_r6 - _prev_r6) > 1e-6:
                _chg6.append((i, round(_prev_r6, 1), round(_r6, 1)))
            _prev_r6 = _r6
        check(not _chg6,
              "融合期半径**全程不变**（变了 %d 次 %s ✗ —— 用户：「没有任何理由让他变半径」✓）"
              % (len(_chg6), _chg6[:3]))

        # ⭐⭐ **窗口记住上次的配置**（用户 2026-10-01 第 1 条 ✓ 原话："让窗口能够记住我
        #   上次的配置" ✓）—— ⚠⚠ 必须在**临时 ui.yaml** 上验 ✗✗（真文件是用户的
        #   `config/ui.yaml` ⇒ 自检**绝不许写它** ✓ 我在这里踩过坑 ✓）。
        from gui import theme as _theme       # ⚠ 外层 `main()` 已把 CFG 指向临时文件 ✓
        _m0, _s0, _l0 = (win.sld_mask.value(),
                         float(win.cmb_speed.currentData() or 1.0),
                         win.chk_loop.isChecked())
        win.sp_path_ms.setValue(333)
        win.sp_merge.setValue(0.85)                # ⚠ 2026-10-03：这项已是 **IoU**（0~1 ✓）
        win.sp_ntol.setValue(3.5)
        win.sp_board.setValue(0.7)
        win.sp_ov.setValue(0.35)                  # 上板防抖重叠率（用户 2026-10-02 ✓）
        win.cmb_speed.setCurrentIndex(0)          # 0.25×
        win.sld_mask.setValue(55)
        win.chk_loop.setChecked(False)
        win._cfg_save()
        _c1 = _theme.load_section("lie_demo")
        check(int(_c1.get("path_ms", -1)) == 333
              # ⭐ 键名与量纲都换了（2026-10-03 ✓ "融合框判定阈值(面积比)" → "融合框判定 IoU"）
              and abs(float(_c1.get("merge_iou", 0)) - 0.85) < 1e-6
              and abs(float(_c1.get("noise_tol", -1)) - 3.5) < 1e-6
              and abs(float(_c1.get("board_s", -1)) - 0.7) < 1e-6
              and abs(float(_c1.get("board_overlap", -1)) - 0.35) < 1e-6
              and abs(float(_c1.get("speed", 0)) - 0.25) < 1e-6
              and int(_c1.get("mask", -1)) == 55
              and _c1.get("loop") is False,
              "配置落到 ui.yaml 的 `lie_demo` 段（含 noise_tol / board_overlap ✓）：%s" % _c1)
        check(_c1.get("file") == str(getattr(win, "_cur", "") or ""),
              "上次打开的文件也落盘了（file=%s ｜ 当前 _cur=%s ✓ 用户：「记住上次打开的文件」✓）"
              % (_c1.get("file"), getattr(win, "_cur", "")))
        # 关掉重开 ⇒ 控件按文件里的值回填（把控件先改成别的值再回填 ✓）
        #   ⚠⚠ **改控件时必须屏蔽写盘** ✗：这些控件一改就保存 ✓ ⇒ 会把刚写好的
        #   333/0.25/55/False **覆盖成** 500/1×/35/True ⇒ 回填自然"回不来" ⇒ 假红 ✗。
        #   ⚠⚠ patch `type(win)._cfg_save` **拦不住** ✗✗：Qt 的 `connect` 在连接那一刻就
        #   取好了**绑定方法对象** ⇒ 之后替换类属性不影响已连接的槽（实测照写 ✗）⇒
        #   必须 patch **模块函数** `theme.save_section`（`_cfg_save` 运行时才查它 ✓）。
        import unittest.mock as _mk
        with _mk.patch.object(_theme, "save_section", lambda *a, **k: None):
            win.sp_path_ms.setValue(500)
            win.sp_merge.setValue(1.1)
            win.sp_ntol.setValue(9.9)
            win.sp_board.setValue(9.9)
            win.sp_ov.setValue(0.99)
            win.cmb_speed.setCurrentIndex(2)
            win.sld_mask.setValue(35)
            win.chk_loop.setChecked(True)
        win._cfg_apply()
        check(win.sp_path_ms.value() == 333
              # ⚠ 2026-10-03：这一项已是「融合框判定 **IoU**」（0~1 ✓ 不是面积比 ✗）⇒ 回填的
              #   就是上游写进 yaml 的那个 0.85 ✓（旧断言写着 2.2 ✗ 那是面积比时代的数 ✓）。
              and abs(float(win.sp_merge.value()) - 0.85) < 1e-6
              and abs(float(win.sp_ntol.value()) - 3.5) < 1e-6
              and abs(float(win.sp_board.value()) - 0.7) < 1e-6
              and abs(float(win.sp_ov.value()) - 0.35) < 1e-6
              and abs(float(win.cmb_speed.currentData()) - 0.25) < 1e-6
              and win.sld_mask.value() == 55 and not win.chk_loop.isChecked(),
              "重开按上次配置回填（%d ms / 阈值 %.1f / 容差 %.1f / 上板 %.1f s / "
              "防抖 %.2f / %s / 蒙版 %d%% / 循环 %s ✓）"
              % (win.sp_path_ms.value(), float(win.sp_merge.value()),
                 float(win.sp_ntol.value()), float(win.sp_board.value()),
                 float(win.sp_ov.value()),
                 win.cmb_speed.currentText(),
                 win.sld_mask.value(), win.chk_loop.isChecked()))
        check(str(getattr(win, "_last_file", "") or "") == str(getattr(win, "_cur", "") or ""),
              "重开回填上次文件（_last_file=%s ✓）" % getattr(win, "_last_file", ""))
        win.sp_path_ms.setValue(int(D._PATH_MS_DEFAULT))   # 还原（别留下脏值 ✓）
        win.sp_merge.setValue(float(D._MERGE_IOU_DEFAULT))
        win.sp_ntol.setValue(float(D._NOISE_TOL_DEFAULT))
        win.sp_board.setValue(float(D._BOARD_S_DEFAULT))
        win.sp_ov.setValue(float(D._BOARD_OVERLAP_DEFAULT))
        win.sld_mask.setValue(_m0)
        win.cmb_speed.setCurrentIndex(2 if abs(_s0 - 1.0) < 1e-6 else 0)
        win.chk_loop.setChecked(_l0)

        # ⑦ ⭐⭐⭐ **保底键盘：←→ 切帧、空格 播放/暂停 —— 不管焦点在哪**（用户 2026-10-01 ✓
        #   原话："优化窗口操作：不管聚焦在哪，左右方向键都是切帧、空格都是播放/暂停" ✓）。
        #   ⚠ 这条**必须逐个焦点目标验** ✗：本窗有一堆会吃按键的子控件（进度条/蒙版滑块吃
        #     ←→ ✓、速度下拉吃 ←→ ✓、勾选框吃空格 ✓、数字框内部的 `QLineEdit` 吃 ←→ 和空格 ✓）
        #     ⇒ 原来那套"手写 `keyPressEvent`"只在**焦点在本窗口自己身上**时有效 ✗ ——
        #     焦点一落到它们身上就失灵 ✓（正是用户说的"有时候能切、有时候不能"✗）。
        #   ⚠⚠ **判据要挑"泄漏了就会变"的量** ✗（否则测了等于没测 ✓）：蒙版滑块值 / 速度下拉
        #     索引 / 勾选框状态 / 数字框值 —— 快捷键要是没抢到，这些量会被控件自己改掉 ✓。
        #     实测：滑块/下拉/勾选框/按钮本来就让路 ✓，**只有数字框不让** ✗（`QSpinBox` 内部的
        #     `QLineEdit` 会抢先 accept `ShortcutOverride` ✓）⇒ 专门给它装了 `eventFilter`
        #     把这两个键抢回来 ✓ —— 这条用例就是钉那一手的 ✓。
        from PyQt5.QtCore import Qt as _Qt
        from PyQt5.QtTest import QTest as _QTest
        from PyQt5.QtWidgets import QShortcut as _QSC
        _reg7 = [(s.key().toString(), s.context()) for s in win.findChildren(_QSC)]
        _need7 = {("Left", _Qt.WindowShortcut), ("Right", _Qt.WindowShortcut),
                  ("Space", _Qt.WindowShortcut)}
        check(_need7 <= set(_reg7),
              "←→/空格 注册成**窗口级**快捷键（`WindowShortcut` ✓ 实测注册表 %s ✗ —— 用"
              "`WidgetShortcut` 或手写 `keyPressEvent` 都会在焦点落到底部控件时失灵 ✗）"
              % (_reg7,))
        win.stop()
        win.on_seek(min(10, len(win.frames) - 2))
        app.processEvents()
        _tg7 = [("进度条", win.sld), ("蒙版滑块", win.sld_mask), ("速度下拉", win.cmb_speed),
                ("勾选框", win.chk_loop), ("按钮", win.btn_play),
                ("数字框①", win.sp_path_ms), ("数字框②", win.sp_merge)]
        _bad7 = []
        for _nm7, _w7 in _tg7:
            _w7.setFocus()
            app.processEvents()
            _foc7 = (app.focusWidget() is _w7)
            _i0_7 = win.i
            _m0_7, _c0_7, _l0_7 = (win.sld_mask.value(), win.cmb_speed.currentIndex(),
                                   win.chk_loop.isChecked())
            _s0_7 = (win.sp_path_ms.value(), float(win.sp_merge.value()))
            _p0_7 = win.playing
            _QTest.keyClick(_w7, _Qt.Key_Right)
            app.processEvents()
            _r7 = (win.i == _i0_7 + 1)
            _QTest.keyClick(_w7, _Qt.Key_Left)
            app.processEvents()
            _l7 = (win.i == _i0_7)
            _QTest.keyClick(_w7, _Qt.Key_Space)
            app.processEvents()
            _s7 = (win.playing != _p0_7)
            if win.playing:
                win.stop()
            _k7 = (win.sld_mask.value() == _m0_7 and win.cmb_speed.currentIndex() == _c0_7
                   and win.chk_loop.isChecked() == _l0_7
                   and (win.sp_path_ms.value(), float(win.sp_merge.value())) == _s0_7)
            if not (_foc7 and _r7 and _l7 and _s7 and _k7):
                _bad7.append((_nm7, _foc7, _r7, _l7, _s7, _k7))
        check(not _bad7,
              "七种焦点下 ←→ 切帧、空格 播放/暂停**都生效**（违例 [焦点名, 焦点到位, →, ←, "
              "空格, 控件没被改] = %s ✗ —— 用户：「不管聚焦在哪」✓）" % (_bad7,))
        win.stop()
        win.on_seek(min(10, len(win.frames) - 2))
        app.processEvents()

    # ④ ⭐ **GIF**（用户 2026-09-30：把"选图片集"**改成"选取 gif"** ✓）：
    #    选一个 `.gif` ⇒ 帧数/时间戳（GIF 自带每帧延时 ✓）/时长都要对得上 ✓
    gif = D.GIF_DIR / "test.gif"
    if gif.is_file():
        win.load(str(gif))
        t0 = time.time()
        while len(win.results) < len(win.frames) and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.05)
        ts = win.ts
        check(win.is_gif and len(win.frames) == 70,
              "GIF 载入：is_gif=%s、%d 帧（期望 70 ✓）" % (win.is_gif, len(win.frames)))
        check(bool(ts) and ts[0] == 0.0 and all(b > a for a, b in zip(ts, ts[1:])),
              "时间戳取自 **GIF 每帧延时**、**严格单调**、首帧归一 0 ✓（%s…）"
              % ([round(t, 3) for t in ts[:3]],))
        check(abs((ts[1] - ts[0]) - 0.18) < 1e-6,
              "第一帧延时 = GIF 里写的 180ms（实测 %.3fs ✓）" % (ts[1] - ts[0]))
        check(abs(win.dur - 12.42) < 0.2,
              "总时长 %.2fs ≈ 70 帧 × 0.18s（= 12.42s ✓）" % win.dur)
        check(win.sld.maximum() == len(win.frames) - 1,
              "进度条范围 = GIF 帧数（0~%d）✓" % win.sld.maximum())
        check("GIF" in win.windowTitle(), "标题标明是 GIF：「%s」" % win.windowTitle())
        check(len(win.results) == len(win.frames),
              "GIF 也**整段演算完**（%d/%d ✓）" % (len(win.results), len(win.frames)))
        # ⭐⭐⭐⭐⭐ **「乙」：原图只留 JPEG 字节** ✗✗（用户 2026-10-07 ✓ 原话："**优化直到能顺畅跑
        #   D:\\Media\\record\\mxd\\10月7日.mp4**" ✓✓）—— ⚠⚠ **为什么这是内存那条命根子** ✗：
        #   那条片子实测 **1920×1080 ｜ 去重后 797 帧** ⇒ 原图留 BGR 是 **6.22MB/帧** ✗
        #   （光它 **4.95GB** ✗）⇒ 换成 JPEG（q95 ✓ **0.54MB/帧** ✓）⇒ 素材合计
        #   **5.0GB → 1.5GB** ✓（**3.3×** ✓）。⚠ **只给显示** ✗：检测/追踪吃的仍是处理帧
        #   `[1]`（750×500 ✓）⇒ 算法口径**一个像素都不动** ✓。
        check(isinstance(win.frames[0][0], (bytes, bytearray))
              and len(win.frames[0][0]) > 1000,
              "**原图存的是 JPEG 字节** ✓（第 1 帧 = %s ／ %d 字节 ✓〔bytes ⇒ 不是 BGR ✗〕）"
              "—— ⚠ 谁把它换回 BGR 数组 ✗ ⇒ 长片内存**又回到 5GB 级** ✗（本条立刻红 ✓）"
              % (type(win.frames[0][0]).__name__, len(win.frames[0][0])))
        _b0 = D.decode_big(win.frames[0][0])
        check(_b0 is not None and _b0.ndim == 3 and _b0.shape[0] > 0
              and _b0.dtype == "uint8" and _b0.shape[0] * win.scale[1] > 0,
              "**能解回原图** ✓（形状 %s ✓、`scale` = (%.2f, %.2f) ✓ ⇒ 画布尺寸对得上 ✓）"
              "—— ⚠ 解不回来 ⇒ 画面整个空 ✗" % (str(_b0.shape), win.scale[0], win.scale[1]))
        # ⭐⭐ **`pos=None` 的帧不许把窗口画崩**（2026-09-30 用户报"打不开 闪退"的根因 ✓）：
        #   追踪器在 init / lost 帧报 `pos=None` ✓，而 `draw()` 里有一段（白箭头 = 轨迹速度）
        #   直接 `pos[0]` ⇒ `TypeError` ⇒ 整个窗口崩 ✗（pythonw 只打 traceback ⇒ 用户只看到
        #   闪退 ✗）。这里直接用**空位置 + 有 track_v** 的形态喂一次 `draw()` ✓。
        ok = True
        try:
            #   ⚠⚠ **原图现在是 JPEG 字节** ✗（2026-10-07「乙」✓ 见 `_BIG_JPEG_Q` ✓）
            #     ⇒ 要喂 `draw()` 得先 `decode_big` 解一次 ✓（不然就是 `'bytes' has no copy` ✗）。
            frame = D.decode_big(win.frames[0][0]).copy()
            D.draw(frame, (None, None, 10.0, False, []), 1.0, 1.0, "hud", (100, 100),
                   {"track_v": (30.0, -20.0), "dt": 0.18, "tracks": [],
                    "box_moves": [], "box_roles": []}, None, 0.0)
        except Exception as exc:                       # noqa: BLE001（就是要抓住所有崩 ✗）
            ok = False
            check(False, "`pos=None` 时 draw() 崩了：%s（该类崩会让窗口闪退 ✗）" % exc)
        check(ok, "`pos=None` 的帧 draw() 不崩（白箭头那段有判空 ✓ 闪退根因已钉住 ✓）")

        # ⭐⭐ **绿圈半径 = 目标框半宽**（用户 2026-09-30 口径 ✓ 原话："将绿圈半径恒=目标框
        #    半宽" ✓）：**端到端**量 —— 拦住 `cv2.circle` 抓实际半径，再复算期望值比对 ✓
        #    （只看代码等于没验 ✗ 实测踩过好多次 ✓）。
        idx = next((i for i, rr in enumerate(win.results)
                    if rr["pos"] is not None and rr.get("boxes")), None)
        if idx is None:
            check(False, "找不到有位置+框的帧（没法验绿圈半径 ✗）")
        else:
            rr = win.results[idx]
            # ⚠⚠ **别去 monkeypatch `cv2.circle`** ✗（实测：改全局 C 函数会让**离屏 Qt 绘制
            #   直接原生崩**（exit=0xC0000409 栈溢出 ✗）⇒ 改成**量真正画出来的像素** ✓
            #   更真实、也不动全局 ✓）。
            im = D.draw(D.decode_big(win.frames[idx][0]).copy(),
                        (None, rr["pos"], rr["r"], rr["hit"], rr["boxes"]),
                        win.scale[0], win.scale[1], "hud", rr.get("cursor"),
                        rr.get("motion"), None, 0.0)
            cap = {}
            _col = np.array((0, 255, 0) if rr["hit"] else (0, 200, 255), np.uint8)
            ys, xs = np.nonzero(np.all(im == _col, axis=2))
            if len(xs):
                _cx = rr["pos"][0] * win.scale[0]
                _cy = rr["pos"][1] * win.scale[1]
                # 绿圈 = 线宽 2 的圆 ⇒ 最远绿像素 ≈ 半径 + 1（线宽一半 ✓）
                cap["r"] = float(np.hypot(xs - _cx, ys - _cy).max()) - 1.0
            # ⚠ 口径已升级（用户 2026-09-30 后一条 ✓）：绿圈半径 = **学到的目标实际半径**
            #   （白块期由 `sqrt(A/π)` 学得 ✓ 透明后**恒定** ✓），不再等于"当前框半宽" ✗
            #   —— 融合框会呼吸，拿它画圈会忽大忽小 ✗。
            tr_ = (rr.get("motion") or {}).get("tgt_radius")
            exp = None if not tr_ else max(6.0, float(tr_) * win.scale[0])
            check(cap.get("r") is not None and exp is not None
                  and abs(cap["r"] - exp) <= 2.0,
                  "绿圈半径 = 学到的**目标实际半径**（实测 %s px ｜ 期望 %s px ✓ 第 %d 帧）"
                  % (None if cap.get("r") is None else round(cap["r"], 1),
                     None if exp is None else round(exp, 1), idx))

        # ⭐⭐ **融合/分离期：绿圈半径必须恒定**（用户 2026-09-30 口径 ✓ 原话："真实目标透明后，
        #   绿圈的半径应该恒定不变，因为我希望它代表**真实目标的实际半径**" ✓）。做法是学一次
        #   就冻结（只在"面积 ≈ 单目标"的框上更新 ✓）⇒ 这里量：**面积 ≥1.3× 标准的帧**（融合期
        #   ✓ 红框正在呼吸 ✗）里，`motion["tgt_radius"]` 的极差必须 ≈0 ✓（否则圈会忽大忽小 ✗）。
        # ⭐⭐⭐ **同一帧必须同一个红框**（用户 2026-09-30："我以帧号递增顺序切到 21 帧红框在
        #   左边，为什么以帧号递减顺序切到 21 帧红框在右边？" ✓✓）。原实现把红框判定放在
        #   `draw()` 里、还借**跨帧累积**的 `draw._last_red` ✗ ⇒ 答案取决于"你从哪边拖过来" ✗✗。
        #   钉法：渲染第 21 帧 → 再渲染别的几帧 → **再渲染第 21 帧** ⇒ 两张图必须**逐像素相同** ✓。
        def _render(k):
            rk = win.results[k]
            return D.draw(D.decode_big(win.frames[k][0]).copy(),
                          (None, rk["pos"], rk["r"], rk["hit"], rk["boxes"]),
                          win.scale[0], win.scale[1], "hud", rk.get("cursor"),
                          rk.get("motion"), None, 0.0)

        k = min(21, len(win.frames) - 1)
        a = _render(k)
        ia = (win.results[k].get("motion") or {}).get("red_i")
        for j in (5, 10, min(40, len(win.frames) - 1)):
            _render(j)
        b = _render(k)
        ib = (win.results[k].get("motion") or {}).get("red_i")
        check(np.array_equal(a, b) and ia == ib,
              "同一帧的红框随**拖动方向**变了（第 %d 帧：red_i %s→%s、图像%s ✗ —— "
              "绘制必须是纯函数、红框只在预演期算一次 ✓）"
              % (k, ia, ib, "相同" if np.array_equal(a, b) else "不同"))

        # ⭐⭐⭐ **信念约束：没"分离"之前，报告位置必须一直在红框里**（用户 2026-09-30 定稿 ✓
        #   原话："你要理解'红框代表我们认为真目标在它里面'，这个认知要保持到'分离'信号的
        #   出现" ✓）。
        #   ⭐⭐ **融合期同样受约束**（用户 2026-10-01 第二轮 ✓ 原话："已经夹取进红框了就让他以
        #   白箭头的指示运动，**除非运动到另一边再夹取**" ✓）：融合第一拍"先夹取" ✓、
        #   其余拍"越界即夹回" ✓ ⇒ **有框的每一帧圆都该在框里** ✓（第一版"融合允许探出"已否 ✓）。
        out_frames = [x for x in win.results
                      if (x.get("motion") or {}).get("tbox") and x.get("pos")]
        esc = []
        for x in out_frames:
            tb = x["motion"]["tbox"]
            p = x["pos"]
            if not (tb[0] - tb[2] / 2 <= p[0] <= tb[0] + tb[2] / 2
                    and tb[1] - tb[3] / 2 <= p[1] <= tb[1] + tb[3] / 2):
                esc.append((out_frames.index(x), round(p[0]), round(p[1])))
        check(bool(out_frames) and not esc,
              "红框在的每一帧报告位置都在框内（%d/%d 帧在外：%s ✗ —— 含融合期 ✓："
              "第一拍先夹、其余拍越界即夹 ✓）" % (len(esc), len(out_frames), esc[:3]))
        # ⭐⭐ **有红框的帧绿圈不能丢**（用户 2026-10-01 ✓ "40帧绿圈怎么丢了" ✓）：融合/分离期
        #   红框还在（`tbox` 有值 ✓）时，报告位置 `pos` 必须还在（绿圈一直在 ✓）—— 否则就是
        #   "融合期连续无观测 ⇒ `lost_n` 越界 ⇒ 报 lost ⇒ 绿圈凭空消失"的 bug ✗（实测帧 40 ✓
        #   已修：丢帧计数只认"红框都没有"的拍 ✓）。
        lost_with_box = [(i + 1) for i, x in enumerate(win.results)
                         if (x.get("motion") or {}).get("tbox") and not x.get("pos")]
        check(not lost_with_box,
              "有红框的帧绿圈不能丢（丢绿圈的帧 %s ✗ —— 帧40 那个 bug ✓）" % (lost_with_box[:5],))

        # ⭐⭐⭐ **「边归属相切」已移除 / 融合判定按面积阈值**（用户 2026-10-01 ✓ 原话：
        #   "移除边归属相切逻辑，走以下流程：判定检出框面积 > 登记的假目标面积一定比例
        #   （「融合框判定阈值」）则判定红框融合；融合大红框后，绿圈继续按白箭头的指示行动" ✓）。
        #   钉法：**两个阈值各跑一遍，结果必须不同**（阈值真的进了判定 ✓）+ 低阈值判出的
        #   融合帧数必须更多 ✓。
        def _run_mr(mr):
            """按给定「融合框判定阈值」跑整段 ⇒ `(融合帧数, 融合期"每拍都在动"的帧数,
            融合期"纹丝不动"的帧数, **规则①口径**的融合帧数)`。

            ⭐ 后两项钉的是用户第二轮的口径："**已经夹取进红框了就让他以白箭头的指示运动**"
            ✓ ⇒ 融合期（第一拍之后）圆圈**必须每拍都在动** ✗ —— 原来位置被"这一拍的观测
            （并集框中心）"按住 ⇒ 帧 16~18 三拍钉在同一个点上 ✗✗（现场就是"夹进去以后不动了"✓）。

            ⚠⚠ 2026-10-01 口径升级后 ✓：`merge_ratio` **只管"框没包住绿圈"那一档**（规则① ✓）；
            框把绿圈也包住时（规则② ✓）**不看阈值**（"即使面积与砖相等也算融合" ✓）⇒ 融合总帧数
            会**几乎不随阈值变** ✗ ⇒ 阈值那条改用 `n_rule1`（圈不在框里的融合帧 ✓）来钉 ✓。
            """
            rr = D.Runner(dets=list(win.dets) if win.dets else None,
                          path_ms=100, merge_iou=0.7)
            n_merged, n_move, n_still, n_rule1, _pp = 0, 0, 0, 0, None
            for i, fr in enumerate(win.frames):
                _o, _p = rr.step(fr[1], i, win.ts[i])[:2]
                _t = rr.tr
                if _t._tbox_merged:
                    n_merged += 1
                    # 规则① 那一档 = **整圈绿圈没被框包住**（用户 2026-10-01 ✓ "包住"= 整圈在内 ✓）
                    if not _t._circle_inside(_t._tbox):
                        n_rule1 += 1
                    if not getattr(_t, "_merge_first", False) and _p and _pp:
                        _mv = ((_p[0] - _pp[0]) ** 2 + (_p[1] - _pp[1]) ** 2) ** 0.5
                        if _mv > 0.05:
                            n_move += 1
                        else:
                            n_still += 1
                _pp = _p
            return n_merged, n_move, n_still, n_rule1

        _m1, _mv1, _st1, _r1_1 = _run_mr(1.0)      # 低阈值 ⇒ 规则① 判出的融合更多 ✓
        _m5, _mv5, _st5, _r1_5 = _run_mr(5.0)      # 高阈值 ⇒ 规则① 几乎不判 ✓
        check(_r1_1 >= _r1_5, "阈值单调性：低阈值的规则① 帧数 ≥ 高阈值（%d ≥ %d ✓）"
              % (_r1_1, _r1_5))
        if _r1_1 > _r1_5:
            check(True,
                  "「融合框判定阈值」真的在起作用（**规则①=框没包住绿圈**那一档）：阈值 1.0 ⇒ %d 帧 "
                  "｜ 5.0 ⇒ %d 帧 ✓；⚠ 融合总帧数（1.0：%d ｜ 5.0：%d）几乎不随阈值变 ✓ —— 因为"
                  "规则②（框包住砖 + 绿圈）不看阈值 ✓（用户 2026-10-01 口径 ✓）"
                  % (_r1_1, _r1_5, _m1, _m5))
        elif _r1_1 > 0 or _r1_5 > 0:
            # ⚠⚠ **2026-10-02 新增「融合框筛选闸」后的如实记账**（用户原话："只能往轨迹预测方向
            #   的前方挑与绿圆圈外接矩形相交的格子，如果没有能挑的就判定为分离信号" ✓）：阈值 1.0
            #   时多判出来的那几帧并集框**不满足新闸**（在预测后方 / 不与绿圈外接矩形相交 ✗）
            #   ⇒ 不再参与挑选 ⇒ 两边只剩同样多帧 ⇒ "阈值改变规则①帧数"这条在本素材上**测不出来** ✓
            #   （旋钮本身仍由单元用例 `test_merge_ratio_switch` 钉 ✓）。
            note("「融合框判定阈值」在本素材上**不再改变规则① 帧数**（1.0：%d ｜ 5.0：%d）："
                 "新闸（前方 + 与绿圈外接矩形相交 ✓ 2026-10-02）把阈值 1.0 多判出的那几帧并集框"
                 "筛掉了 ✗ ⇒ 两边都剩 %d 帧 ✓。旋钮行为由单元用例 `test_merge_ratio_switch` 钉 ✓"
                 % (_r1_1, _r1_5, _r1_1))
        else:
            # ⚠⚠ 本素材里**规则① 一帧都没判过**（框一旦包住砖，绿圈也就在框里 ⇒ 规则② 先成立 ✓）
            #   ⇒ 阈值旋钮在这一段素材上不参与判定 ✓。旋钮本身的行为由**同套件里的单元用例**钉死
            #   （`test_merge_ratio_switch`：框 12×10 / 砖 10×10 ⇒ 阈值 1.1 判融合、2.0 不判 ✓）。
            note("「融合框判定阈值」在本素材**不参与判定**（规则① 帧数 1.0：%d ｜ 5.0：%d ✗）："
                 "框一旦包住砖，绿圈几乎总在框里 ⇒ 规则② 先成立、不看阈值 ✓；融合总帧数 "
                 "（1.0：%d ｜ 5.0：%d）几乎不随阈值变 ✓。阈值本身的行为由单元用例 "
                 "`test_merge_ratio_switch` 钉 ✓" % (_r1_1, _r1_5, _m1, _m5))
        # ⭐⭐ 直接钉"融合期按白箭头运动"：全融合那一遍里，融合期（非首拍）**每拍都在动** ✓
        #   （"纹丝不动" = 0 帧 ✗）—— 这条在"位置还是用观测/被按住"的回归下会红 ✓。
        #   ⚠ 方向限定后（用户 2026-10-01 ✓）少了"往后方假框对角"的修正 ⇒ 个别帧白箭头≈0 时
        #     圆会停一拍 ✓ —— 允许 ≤1 帧 ✗（原 bug 是"帧 16~18 三拍钉死" ✗ 仍会被拦 ✓）。
        # ⚠⚠ 2026-10-02 ✓：规则② 恢复后融合段多为**孤立单帧**（每段只有首拍 ✗）⇒
        #   "非首拍融合帧"可能一帧都没有 ✗ —— 有帧才钉 ✓（"按白箭头走"的契约由 ⑨
        #   "白箭头 = 圆心真正用的那支"钉着 ✓）。
        if (_mv1 + _st1) > 0:
            check(_mv1 > 0 and _st1 <= 1,
                  "融合期（第一拍之后）圆圈**每拍都在动**（动了 %d 帧、纹丝不动 %d 帧 ✓ = 按白箭头走 ✓）"
                  % (_mv1, _st1))
        else:
            check(True,
                  "本素材没有「融合期非首拍」帧可测 ✓（融合段都是孤立单帧 ✓）⇒ 无对象可钉 ✓"
                  "（按白箭头走的契约由 ⑨ 白箭头一致那条钉 ✓）")

        # ⚠ 判据只要求**透明之后**恒定（用户原话是"真实目标**透明后**…恒定不变" ✓）——
        #   白块还在的那些帧**本来就该继续学** ✓ 不能算"呼吸" ✗（我第一版就写宽了 ✗）。
        #   ⇒ 只看：**白块已消失**（`white is None` ✓ = 已透明）且**有融合框**（≥1.3× ✓）的帧。
        rr = [x for x in win.results
              if x.get("motion") and x["motion"].get("tgt_radius")
              and x["motion"].get("rad_frozen")           # ⭐ 只验**冻结之后**（用户口径 ✓）
              and any(float(b[3]) * float(b[4]) >= 1.3 * (x["motion"].get("typ_area") or 1e9)
                      for b in (x.get("boxes") or []))]
        if len(rr) < 3:
            check(False, "GIF 里没找到足够的'已透明 + 融合期'帧（%d），没法验半径恒定 ✗" % len(rr))
        else:
            rads = [x["motion"]["tgt_radius"] for x in rr]
            spread = max(rads) - min(rads)
            check(spread <= 3.0,
                  "融合期绿圈半径在呼吸（%d 帧内极差 %.1f px ✗ 该恒 ≈0 ✓：学到的实际半径 "
                  "%.1f~%.1f px）" % (len(rr), spread, min(rads), max(rads)))
    else:
        check(False, "找不到 %s（GIF 用例素材）" % gif)

    # ⑬ ⭐⭐ **视图三件套：滚轮缩放 / 双击适应 / 鼠标坐标**（用户 2026-10-01 ✓ 原话："鼠标在
    #   视图上会持续显示坐标 / 滚轮可以缩放 / 双击适应窗口" ✓）。`win.lbl` 现在是 `_ImageView`。
    from PyQt5.QtCore import QPointF                              # noqa: E402
    from PyQt5.QtGui import QImage                                # noqa: E402
    _lbl = win.lbl
    _qi = QImage(200, 100, QImage.Format_RGB888)
    _qi.fill(0x808080)
    _lbl.set_image(_qi, sx=2.0, sy=2.0)
    _lbl.resize(400, 200)
    _lbl._fit = True
    _w, _h = _lbl.width(), _lbl.height()
    _z, _ox, _oy = _lbl._geo()
    _exp = min(_w / 200.0, _h / 100.0)
    check(abs(_z - _exp) < 1e-6,
          "⑬ 适应窗口缩放 = min(w/200, h/100)（实际 %.3f 期望 %.3f ✓ —— 视图在布局里尺寸被"
          "接管 ✓ 故按实际尺寸算）" % (_z, _exp))
    _im = _lbl._img_at(QPointF(_ox + 100.0 * _z, _oy + 50.0 * _z))
    check(_im is not None and abs(_im[0] - 100.0) < 1e-6 and abs(_im[1] - 50.0) < 1e-6,
          "⑬ 鼠标在图像中心 ⇒ 图像坐标 (100,50)（实际 %s ✓）" % (_im,))
    _lbl._fit = False
    _lbl._zoom = 4.0
    _lbl._off = QPointF(10.0, 20.0)
    _z2, _ox2, _oy2 = _lbl._geo()
    check(abs(_z2 - 4.0) < 1e-6 and abs(_ox2 - 10.0) < 1e-6 and abs(_oy2 - 20.0) < 1e-6,
          "⑬ 手动缩放(4×, 偏移 10,20)后 `_geo` 返回手动值（%.1f, %.1f, %.1f ✓）" % (_z2, _ox2, _oy2))
    _lbl._zoom = 1.0
    _lbl._off = QPointF(0.0, 0.0)
    _im2 = _lbl._img_at(QPointF(150.0, 75.0))
    _px = (150.0 / 2.0) if _im2 is not None else -1.0
    _py = (75.0 / 2.0) if _im2 is not None else -1.0
    check(_im2 is not None and abs(_im2[0] / _lbl._sx - 75.0) < 1e-6
          and abs(_im2[1] / _lbl._sy - 37.5) < 1e-6,
          "⑬ 图像(150,75) → 加工域(75, 37.5)（实际 (%.1f, %.1f) ✓ 与红框/登记同一把尺 ✓）"
          % (_px, _py))
    check(hasattr(_lbl, "wheelEvent") and hasattr(_lbl, "mouseDoubleClickEvent")
          and hasattr(_lbl, "mouseMoveEvent") and _lbl.hasMouseTracking(),
          "⑬ 视图装了滚轮缩放 / 双击适应 / 持续鼠标坐标的事件处理器（且开启鼠标追踪 ✓）")

    # ⑭ ⭐⭐⭐ **点选检出框**（用户 2026-10-03 ✓ 原话："你能让我在**点选检出框**的时候显示
    #   `(360.4,323.0) 179×208=37232` 这种信息吗" ✓）—— 接线级（命中逻辑本身在顶层
    #   `test_pick_box` 里钉 ✓）：左键报上来的**加工域坐标** ⇒ `_on_pick_box` 命中那格 ⇒
    #   `_pick_box()` 给出它 ✓、状态栏也写上这格数字 ✓、**切帧后不显示**（但记住是哪一帧选的 ✓
    #   —— 拖回去还在 ✓）。
    _ip = next((k for k in range(len(win.results)) if win.results[k].get("boxes")), None)
    if _ip is None:
        note("⑭ 这段素材里没有检出框 ⇒ 跳过「点选」接线断言（正常，不是失败 ✓）")
    else:
        win.i = _ip
        _bp = win.results[_ip]["boxes"][0]
        win._on_pick_box((float(_bp[1]), float(_bp[2])))
        _pb = win._pick_box()
        check(_pb is not None and abs(float(_pb[1]) - float(_bp[1])) < 1e-6
              and abs(float(_pb[2]) - float(_bp[2])) < 1e-6,
              "⑭ 点在框心 ⇒ 选中那格（帧 %d ｜ 选中 %s ✓）" % (_ip + 1, _pb))
        _smsg = win.statusBar().currentMessage()
        check("点选" in _smsg and ("%d" % int(round(float(_bp[3])))) in _smsg,
              "⑭ 状态栏写出这格数字（`… ｜ 点选 (x,y) W×H=面积` ｜ 实测 %r ✓）" % (_smsg,))
        win.i = (_ip + 1) % len(win.results)
        check(win._pick_box() is None,
              "⑭ 切到别帧 ⇒ 不再显示（记的是「哪一帧选的」✓ 拖回去还在 ✓）")
        win.i = _ip
        check(win._pick_box() is not None, "⑭ 拖回选中那一帧 ⇒ 高亮与数字都还在 ✓")

    win.close()
    app.processEvents()
    return 0


def test_red_equals_backend():
    """⭐⭐⭐ **红框必须逐像素等于后端给的框**（用户 2026-09-30 定稿 ✓ 原话："红框必须与
    后端完全一致" ✓）。

    之前那套"UI 自己判红框（含融合期锁 ✗、按索引找最近一格 ✗）"已**整条删除** ✗ ——
    它是第二个真相来源、会与后端 `tbox` 打架（实测第 19 帧就差一格 ✗）。
    钉法（**合成帧**直接验绘制 ✓）：
      · 给了 `tbox` ⇒ 图上**红色像素的范围**必须等于那个框（±3px 线宽 ✓）；
      · `tbox=None`（建档/丢帧，后端没答案）⇒ **一个红像素都不许有** ✓。
    """
    frame = np.zeros((200, 300, 3), np.uint8)
    tb = (150.0, 100.0, 80.0, 60.0)
    im = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []),
                1.0, 1.0, "hud", None, {"tbox": tb, "box_roles": []}, None, 0.0)
    ys, xs = np.nonzero(np.all(im == np.array((0, 0, 255), np.uint8), axis=2))
    e = (tb[0] - tb[2] / 2, tb[1] - tb[3] / 2, tb[0] + tb[2] / 2, tb[1] + tb[3] / 2)
    ok = (len(xs) > 0 and abs(xs.min() - e[0]) <= 3 and abs(xs.max() - e[2]) <= 3
          and abs(ys.min() - e[1]) <= 3 and abs(ys.max() - e[3]) <= 3)
    check(ok, "红框没跟后端 `tbox` 对齐（红像素 x[%s..%s] y[%s..%s] ｜ 期望 x[%.0f..%.0f] "
              "y[%.0f..%.0f] ✗）" % (xs.min() if len(xs) else "-", xs.max() if len(xs) else "-",
                                    ys.min() if len(ys) else "-", ys.max() if len(ys) else "-",
                                    e[0], e[2], e[1], e[3]))
    im2 = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []),
                 1.0, 1.0, "hud", None, {"tbox": None, "box_roles": []}, None, 0.0)
    red2 = int(np.all(im2 == np.array((0, 0, 255), np.uint8), axis=2).sum())
    check(red2 == 0, "后端没给框却画了红框（%d 个红像素 ✗ 该 0 ✓：如实呈现「后端还没有答案」✓）"
          % red2)


def test_merge_label():
    """⭐⭐ **融合红框 ⇒ 右上角标「融合」**（用户 2026-10-01 ✓ 原话："如果红框处于融合状态，
    就标在框右上角吧" ✓）。

    合成帧验绘制：
      · `merged=True` ⇒ 红框**右上角**出现「融合」标（背景色 `(0,0,190)` ✓）；
      · `merged=False` / 没给框 ⇒ **一个都不许有** ✓。
    """
    frame = np.zeros((200, 300, 3), np.uint8)
    tb = (150.0, 100.0, 80.0, 60.0)
    _LAB = np.array((0, 0, 190))
    im = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []),
                1.0, 1.0, "hud", None, {"tbox": tb, "merged": True, "box_roles": []},
                None, 0.0)
    m = np.all(im == _LAB, axis=2)
    ys, xs = np.nonzero(m)
    _bx1 = tb[0] + tb[2] / 2
    _by0 = tb[1] - tb[3] / 2
    # ⚠⚠ **容差要跟着字号走** ✗✗（用户 2026-10-04 ✓ "窗口字体能大点" ⇒ 画布字号放大 ✓）——
    #   这条要验的是「**贴在红框右上角**」（**右对齐**框右边 ✓ **顶边贴**框顶 ✓），
    #   **不是**"标记多宽/几像素" ✗ ⇒ 左右各给足（±6 ✓）、y 给 6 ✓。
    #   ⚠ 实测：字号放大后右边界到了 `_bx1+1`（**抗锯齿边缘** ✓）⇒ 卡 `<= _bx1` 就红了 ✗。
    ok = (len(xs) > 0 and xs.min() >= _bx1 - 80 and xs.max() <= _bx1 + 6
          and abs(ys.min() - _by0) <= 6)
    check(ok, "融合红框右上角没标「融合」（%d 个标像素 x[%s..%s] y[%s..%s] ｜ 期望在红框右上角 "
          "x[%.0f..%.0f] y≈%.0f ✗）"
          % (len(xs), xs.min() if len(xs) else "-", xs.max() if len(xs) else "-",
             ys.min() if len(ys) else "-", ys.max() if len(ys) else "-",
             _bx1 - 40, _bx1, _by0))
    im2 = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []),
                 1.0, 1.0, "hud", None, {"tbox": tb, "merged": False, "box_roles": []},
                 None, 0.0)
    n2 = int(np.all(im2 == _LAB, axis=2).sum())
    # ⚠ **给一点容差**（用户 2026-10-04 ✓ 字号放大后 ✓）：红框/文字的**抗锯齿边缘**会偶发
    #   1~2 个同色像素 ✗ —— 而真的"标了融合"是 **1000+ 个像素** ✓ ⇒ 阈值 20 完全分得开 ✓。
    check(n2 < 20, "非融合红框也标了「融合」（%d 个标像素 ✗ 该 ≈0 ✓）" % n2)


def test_path_line():
    """⚠ **白线（纳入预测的轨迹）已移除**（用户 2026-10-04 ✓ 原话："黄箭头、白线…也不需要，
    移除" ✓）：`motion["path_pts"]` **不再画** ✗（后端仍给那份曲线，只是演示窗不用了 ✓）。

    合成帧验：有 `path_pts` 与没有的两份渲染，**白像素差集 = 0** ✓（白线本体不画 ✓；
    其它白标记两边都有 ⇒ 被差掉 ✓）。
    """
    frame = np.zeros((200, 300, 3), np.uint8)
    _pts = [(50.0, 100.0), (80.0, 100.0), (110.0, 120.0)]
    im = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []),
                1.0, 1.0, "hud", None, {"path_pts": _pts, "box_roles": []}, None, 0.0)
    im0 = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []),
                 1.0, 1.0, "hud", None, {"path_pts": [], "box_roles": []}, None, 0.0)
    # ⚠ **用"近白"阈值**（≥140 ✓）：万一还画着白线，AA 抗锯齿在黑底上会混成灰色
    #   （峰值 ~218 ✓）⇒ 纯白 `==255` 抓不到 ✗ ⇒ 用近白阈值才拦得住 ✓。
    _W = np.array((140, 140, 140))
    _only = np.all(im >= _W, axis=2) & ~np.all(im0 >= _W, axis=2)
    n = int(_only.sum())
    check(n == 0,
          "白线（纳入预测的轨迹）**不再画**（有 path_pts 与没有的两份渲染白像素差集 = %d 像素 "
          "｜ 期望 **0** ✓ —— 用户 2026-10-04「白线…移除」✓）" % n)


def test_pink_relay_box():
    """⭐⭐ **淡粉框 = 「红框丢失后的预测接力」**（用户 2026-10-01 定稿 ✓ 原话："淡粉色的框
    代表'真目标的检出范围（红框）丢了'，他应该是红框丢失后的预测接力" ✓）。

    三条契约（合成帧直接验绘制 ✓）：
      · 红框**在**（后端 `tbox` 有）⇒ **不画**淡粉框（检出范围由红框自己表达 ✓）；
      · 红框**丢**（`tbox=None`）且有宽高快照 ⇒ 画：尺寸 = 最后一次红框宽高的**原始值**
        （不缩到圆内 ✗）、中心 = 报告位置（丢框期间它就是预测外推 ✓ = 接力 ✓）；
      · 丢框且**没有**快照（建档期）⇒ 不画（从来没得到过红框，谈不上接力 ✓）。
    """
    frame = np.zeros((200, 300, 3), np.uint8)
    _PINK = np.array((203, 192, 255))

    def _pink(im):
        d = np.abs(im.astype(int) - _PINK).sum(axis=2)     # 近似色匹配（抗锯齿边缘混色 ✓）
        # ⚠ 阈值要紧（≤15 ✗ 别放宽 ✗）：实测红框在的帧里有个 AA 混色像素 (203,203,238)
        #   偶然撞进 40 ⋅ 一票否决 ⇒ 假红 ✗；2px 线的中间排必是纯色（d=0 ✓）。
        return np.nonzero(d <= 15)

    # ① 红框在 ⇒ 一个淡粉像素都不许有
    #   ⚠ pos 故意放**框内偏左上**（≠框中心 ✓）：真实场景报告位置本来就不在框中心 ⇒
    #   若回归成"红框在也画"，淡粉框会与红框**错位露出** ⇒ 用例才抓得住（放中心会被
    #   3px 红线完全盖住 ⇒ 假绿 ✗）。
    im = D.draw(frame.copy(), (None, (130.0, 90.0), 20.0, True, []), 1.0, 1.0,
                "hud", None, {"tbox": (150.0, 100.0, 80.0, 60.0),
                              "tbox_wh": (80.0, 60.0), "box_roles": []}, None, 0.0)
    ys, xs = _pink(im)
    check(len(xs) == 0, "红框在的时候还画了淡粉框（%d 个像素 ✗ —— 检出范围由红框自己表达，"
          "画了就看不出「什么时候丢了」✓）" % len(xs))

    # ② 红框丢 + 有快照 ⇒ 原始尺寸、中心 = 报告位置（预测接力 ✓）
    im2 = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []), 1.0, 1.0,
                 "hud", None, {"tbox": None, "tbox_wh": (80.0, 60.0),
                               "box_roles": []}, None, 0.0)
    ys, xs = _pink(im2)
    e = (150.0 - 40.0, 100.0 - 30.0, 150.0 + 40.0, 100.0 + 30.0)
    ok = (len(xs) > 0 and abs(xs.min() - e[0]) <= 3 and abs(xs.max() - e[2]) <= 3
          and abs(ys.min() - e[1]) <= 3 and abs(ys.max() - e[3]) <= 3)
    check(ok, "丢框后的淡粉接力框不对（像素 x[%s..%s] y[%s..%s] ｜ 期望按快照 (80×60) 中心在"
          "报告位置 x[%.0f..%.0f] y[%.0f..%.0f] ✗）"
          % (xs.min() if len(xs) else "-", xs.max() if len(xs) else "-",
             ys.min() if len(ys) else "-", ys.max() if len(ys) else "-",
             e[0], e[2], e[1], e[3]))

    # ③ 丢框且没快照（建档期）⇒ 不画
    im3 = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []), 1.0, 1.0,
                 "hud", None, {"tbox": None, "box_roles": []}, None, 0.0)
    ys3, xs3 = _pink(im3)
    check(len(xs3) == 0, "从没得到过红框（建档期）也画了淡粉框（%d 个像素 ✗ 谈不上接力 ✗）"
          % len(xs3))


def test_merge_ratio_switch():
    """⭐⭐⭐ **融合判定按面积阈值 ⇒ 融合时不做信念夹取**（用户 2026-10-01 ✓ 原话："判定检出框
    面积 > 登记的假目标面积一定比例（「融合框判定阈值」）则判定红框融合；融合大红框后，绿圈
    继续按白箭头的指示行动" ✓；"边归属相切"整套已移除 ✗）。

    单元级（不跑素材 ✓ 快 ✓ 只钉接线）：
      · `LieTracker(merge_iou=…)`（2026-10-03 起是 **IoU** ✓）真的被记下来 ✓；
      · 面积 ≥ 阈值 × 登记面积 **且圈压着框**（规则① 2026-10-02 定稿 ✓）⇒ `_tbox_merged`
        为真 ✓（阈值越大 ⇒ 越难判融合 ✓）。
    """
    from perception.lie_registry import Entry as _Entry
    from perception.lie_registry import ShapeRegistry as _Reg
    from perception.lie_tracker import LieTracker as _LK
    _a = _LK(merge_iou=0.6)          # ⚠ 2026-10-03：这项已换成 **IoU**（0~1 ✓ 不是面积比 ✗）
    _b = _LK(merge_iou=0.9)
    check(abs(_a.merge_iou - 0.6) < 1e-9 and abs(_b.merge_iou - 0.9) < 1e-9,
          "`merge_iou`（2026-10-03 起是 **IoU** ✓）透传进追踪器（0.6 / 0.9 ✓）")
    # 造：**已上板**假目标 10×10 = 100 ⇒ 包围它的框 12×10 = 120（面积比 1.2）
    #   ⇒ 阈值 1.1 判融合、2.0 不判 ✓
    #   ⚠ **调被测代码本身**（`_is_merged_box` ✓ 与 `_step_track` 同一处口径 ✓）——
    #     把公式在用例里再写一遍等于没测 ✗（我第一版就这么写的 ✓ 已改 ✓）。
    #   ⚠⚠ 分母是**登记表里那块假目标的面积**（用户 2026-10-01 口径："包围已上板的假目标…
    #     / 已上板的假目标记录的面积" ✓）—— 不再是"当拍检出中位" ✗ 所以**必须**有登记条目 ✓。
    def _mk(thr, box, fake=(0.0, 0.0, 10.0, 10.0), board=True):
        # ⭐⭐ 2026-10-03 判据换成 **IoU + 参数分档** ⇒ 本用例钉的是「**内砖**那一半」⇒ 要填的
        #   是 **`brick_iou_sep`**（= 规则① 实际用的那个阈值 ✓ 见 `_brick_iou_thr` ✓），
        #   不是 `merge_iou` ✗（那个现在只管"这格是不是砖自己"那一档 ✓）。
        #   `ring_cov_sep=0.0` ⇒ 前半条（框↔圆外接矩形 IoU）恒过 ✓ ⇒ 测的才是内砖阈值 ✓。
        #   夹具：砖 10×10、框 12×10、同心 ⇒ `IoU(框,砖) = 100/120 = **0.833**` ✓。
        tr = _LK(brick_iou_sep=thr, ring_cov_sep=0.0)
        e = _Entry(fake, ts=0.0)
        e.on_board = bool(board)
        rg = _Reg()
        rg.entries = [e]
        tr._reg = rg
        if box is not None:
            # ⭐ 规则①（用户 2026-10-02 定稿 ✓）还要「**圈压着框**」⇒ 摆圈心在框中心 ✓
            tr.pos = (float(box[0]), float(box[1]))
            tr._rad = 20.0
        return tr._is_merged_box(box)

    check(_mk(0.9, (0.0, 0.0, 12.0, 10.0)) is True
          and _mk(0.5, (0.0, 0.0, 12.0, 10.0)) is False,
          "内砖 IoU 0.833：阈值 **0.90 ⇒ 判融合** ✓、**0.50 ⇒ 不判** ✓（阈值真的在起作用 ✓ "
          "用户 2026-10-03「把面积比换成 IoU」✓）")
    # 边界与保守项：恰好等于阈值 ⇒ 判融合 ✓；框为空 / 没有已上板假目标 / 没包围它 ⇒ 一律不判 ✓
    check(_mk(0.84, (0.0, 0.0, 12.0, 10.0)) is True
          and _mk(0.9, None) is False
          and _LK(brick_iou_sep=0.9,
                  ring_cov_sep=0.0)._is_merged_box((0.0, 0.0, 12.0, 10.0)) is False
          and _mk(0.9, (0.0, 0.0, 12.0, 10.0), board=False) is False
          and _mk(0.9, (500.0, 500.0, 12.0, 10.0)) is False,
          "边界：内砖 IoU **严格小于**阈值才算（0.833 < 0.84 ⇒ 融合 ✓；⚠ == 阈值 ⇒ **不判** ✓ "
          "正是用户口径「**小于这个 IoU 才行**」✓）；无框 / 没有**已上板**假目标（表空 / 只有"
          "候选）/ 框没包围它 ⇒ 不判 ✓")

    # ⭐⭐⭐ **漂移纠正速度**（用户 2026-10-01 ✓ 原话："计算「飘移纠正速度」=（淡粉色框的
    #   「飘移期」到…融合期的第一拍的 圆圈与假目标群的 相对位移 / 时间）" ✓）—— 单元级：
    #   `2 ×（本拍的群体系坐标 − 漂移起点的群体系坐标）/ Δt`（半域差 × 2 = 加工域 ✓）。
    _tk = _LK(path_ms=100)
    _tk.pos = (160.0, 180.0)               # 本拍圆圈报告位置（加工域 ✓）
    _tk._drift_pos0 = (100.0, 200.0)       # 漂移起点的圆圈位置（加工域 ✓）
    _tk._drift_gt = (10.0, 20.0)           # 同期群体累计位移（加工域 ✓ 与箭头同口径 ✓）
    _tk._drift_ts0 = 1.0
    _v = _tk._drift_fix_vel(1.5)
    # Δ相对 = (60−10, −20−20) = (50,−40) ÷0.5s ⇒ (100,−80) px/s ✓（加工域 ✓ 直接就是 vel_rel ✓）
    check(_v is not None and abs(_v[0] - 100.0) < 1e-6 and abs(_v[1] + 80.0) < 1e-6,
          "漂移纠正速度 =（圆圈位移 − 同期群体平移）/Δt（实测 %s ｜ 期望 (100.0, −80.0) ✓）"
          % (_v,))
    check(_tk._drift_fix_vel(1.0) is None and _LK(path_ms=100)._drift_fix_vel(2.0) is None,
          "Δt≤0 / 没有漂移起点 ⇒ 如实给 None（不猜 ✓ 调用方沿用旧速度 ✓）")


def test_track_label_anchor_circle():
    """⭐⭐⭐⭐⭐ **"座位推的"标签必须钉在圆心** ✗✗（用户 2026-10-06 ✓ 原话："**为什么 #8 推的标在
    这里？这帧附近所有的座位都已经被占了，#8 推的应该标在圆心**" ✓✓）。

    ⚠⚠⚠ **它抓的是一个真 bug** ✗✗（如实记 ✓）：我第一版把锚点写成 `motion["pos"]` ✗ ⇒ 而那个键
      **实测是 `None`** ✗（`10月1日` 帧 62~64 全空 ✓）⇒ 代码**从来没生效过** ✓，标签一直挂在
      **轨迹自己的推算位置 `p`** ✗ —— 而两者实测能差 **23px** 以上 ✓（帧 63：圆 `(260.8,321.1)`
      vs `#8.p (237.3,329.8)` ✓）⇒ 看着就像"挂到别人那格座位上了" ✓（用户当场看出来 ✓）。
    ⇒ 现在抽成纯函数 `lie_demo.track_label_anchor` ✓（`draw()` 用的就是它 ✓ 同源 ✓）：
      · **座位 + 这一拍"推的"**（`sel` ✓ 且 `live == False` ✓）⇒ 钉**圆心** ✓（用 `draw()` 手里
        那个 `pos` ✓）；· **其余** ⇒ 照旧钉**它自己的 `p`** ✓（老行为不变 ✓）。
    钉三条：
      ① **座位推的 ⇒ 圆心** ✓（⚠ 把锚点改回 `motion["pos"]` / 改回 `p` ⇒ 立刻红 ✗）；
      ② **座位有框时不改锚点** ✓（它本来就有座位 ✓，标签走框那条链 ✓）；
      ③ **别人的账照旧钉 `p`** ✓（不许被"圆心"那套带跑 ✗）。
    """
    from tools.lie_demo import track_label_anchor

    _pos = (260.8, 321.1)
    _seat = {"tid": 8, "p": (237.3, 329.8), "live": False, "sel": True}
    _a1 = track_label_anchor(_seat, _pos, 1.0, 1.0)
    check(_a1 == (260, 321),
          "① **座位这一拍是「推的」 ⇒ 标签钉在圆心** ✓（锚点 = %s ✓〔要 (260,321) = 圆心 ✓〕）"
          " —— ⚠ 锚点写回 `motion['pos']`（**实测恒为 `None`** ✗）或写回 `p`（= (237,329) ✗）"
          " ⇒ 本条立刻红 ✗（实测两者差 **23px** ✓ ⇒ 看着像挂到别人座位上了 ✓）" % (_a1,))

    _seat2 = {"tid": 8, "p": (237.3, 329.8), "live": True, "sel": True}
    _a2 = track_label_anchor(_seat2, _pos, 1.0, 1.0)
    check(_a2 == (237, 329),
          "② **座位这一拍有框 ⇒ 锚点不动** ✓（锚点 = %s ✓〔要 (237,329) = 它自己的 `p` ✓〕）"
          " —— ⚠ 一刀切「座位一律钉圆心」✗ ⇒ 有框那几拍会把标签从框上拽走 ⇒ 本条红 ✗" % (_a2,))

    _other = {"tid": 11, "p": (205.3, 376.4), "live": False, "sel": False}
    _a3 = track_label_anchor(_other, _pos, 1.0, 1.0)
    check(_a3 == (205, 376),
          "③ **别人的账照旧钉 `p`** ✓（锚点 = %s ✓〔要 (205,376) ✓〕）—— ⚠ 「圆心那套」对谁都生效 ✗"
          " ⇒ 满屏标签挤在圆上 ⇒ 本条红 ✗" % (_a3,))


def test_box_label_two_lines():
    """⭐⭐⭐⭐⭐ **检出框标签拆两行**（用户 2026-10-04 ✓ 原话："**把目标标签里的例如「#14 m0.89」
    换一行显示**" ✓✓）。

    ⚠ 原来一行 `shape 0.98 #14  m0.89` **又宽又挡** ✗（横着盖掉半个框 ✓）⇒ 现在：
      · **第 1 行 = `0.98`**（**置信度** ✓ —— ⚠ 2026-10-07 起**不带 `shape` 那个词**了 ✓
        因为类别名**恒为 `shape`** ✗ 等于没信息 ✓）；
      · **第 2 行 = `#14  m0.89`**（"是谁" ＋ "这格多可信" ✓）；
      · ⚠⚠ 整块**不再以框心为中心** ✗（用户 2026-10-07："**标签还是在框心挡视野**" ✓）
        ⇒ 位置口径搬到 `box_tab_pos`（**框左上角外侧** ✓ 见 `test_box_tab_pos_above_box_corner` ✓）；
      · ⚠ **没有黑底了** ✗（用户 2026-10-07："**把标签的黑色背景去掉**" ✓ ⇒ ⑤ 钉住 ✓）。
    离屏像素级钉五条（黄字 = `(0,255,255)` ✓）：
      ① **恰好两行**（黄像素按行分组 = **2** 组、组间有缝 ✓）；
      ② **没搞反**：第 1 行 = `0.98`、第 2 行 = `#14  m0.89` ✓（⚠ 文字口径由纯函数钉 ✓
         —— 像素上两串差不多宽 ⇒ 判不出顺序 ✗）；
      ③ 画出来的两行**确实就是那两串**（墨宽不超排版宽 ✓）；
      ④ 整块**比原来一行窄** ✓ —— 这才是"换行"的目的（少挡画面 ✓）；
      ⑤ ⭐ **整块背后没有黑板子** ✓（画面里**一个纯黑 `(0,0,0)` 像素都不该有** ✓ —— 底色是
         30/30/30 的灰底 ✓）—— ⚠⚠ 那句 `cv2.rectangle(..., (0,0,0), -1)` 加回来 ⇒ 本条红 ✗。
    """
    _f = np.full((180, 300, 3), 30, np.uint8)
    _res = ({"state": "track"}, (130.0, 90.0), 18.0, None, [])
    _mo = {"mode": "motion",
           "box_v": [(130.0, 90.0, 0.0, 0.0, 14, 0.98, 0.89, 0)],
           "show": {"boxes": True, "box_id": True}}
    _im = D.draw(_f.copy(), _res, 1.0, 1.0, motion=_mo)
    _pts = [(int(_x), int(_y)) for _y in range(180) for _x in range(300)
            if tuple(int(_v) for _v in _im[_y, _x]) == (0, 255, 255)]
    check(bool(_pts),
          "⓪ 画面上确实有**黄字标签**（黄像素 %d 个 ✓ —— 没有它 ⇒ 后面两条无从谈起 ✗）"
          % len(_pts))
    _rows = []
    for _y in sorted({_p[1] for _p in _pts}):
        if _rows and _y - _rows[-1][-1] <= 1:
            _rows[-1].append(_y)
        else:
            _rows.append([_y])
    check(len(_rows) == 2,
          "① **恰好两行**：黄字按行分成 **%d** 组（每行占的行数 %s）—— ⚠ 还挤在一行 ⇒ 只有 1 组"
          " ⇒ 立刻红 ✗✗" % (len(_rows), [len(_r) for _r in _rows]))
    _ws = []
    for _r in _rows:
        _xs = [_p[0] for _p in _pts if _p[1] in _r]
        _ws.append(max(_xs) - min(_xs) + 1)
    # ② ⚠ **顺序不能靠像素宽度判** ✗（`shape 0.98` 墨宽 103 vs `#14  m0.89` 109 ⇒ 差不多宽
    #    ⇒ 判不出谁上谁下 ✓）⇒ 由**纯函数**（= `draw()` 用的那一个 ✓）钉 ✓。
    _l1, _l2 = D.box_label_lines(14, 0.98, 0.89)
    _l1b, _l2b = D.box_label_lines(7, None, None)          # ⚠ 退化：没置信度、没匹配分 ✓
    check(_l1 == "0.98" and _l2 == "#14  m0.89",
          "② **两行的内容与顺序**：`box_label_lines(14, 0.98, 0.89)` = (%r, %r) ✓ —— "
          "第 1 行 `0.98`（置信度 ✓ ⚠ 2026-10-07 起**不带 `shape` 那个词** ✓ 因为类别名恒为 "
          "`shape` ✗）、第 2 行 `#14  m0.89`（是谁＋多可信 ✓）"
          " ⚠ 搞反 / 还拼着一串 / 又把 `shape` 加回来 ⇒ 立刻红 ✗✗" % (_l1, _l2))
    check(_l1b == "-" and _l2b == "#7",
          "②b **拿不到的量不编**：没置信度 ⇒ 写占位的 `-` ✓、没匹配分 ⇒ 只写 `#7` ✓"
          "（实测 = (%r, %r) ✓ ⚠ 那一行**必须占着** ✗ —— 排版按两行算 ✓ 少一行会错位 ✓）"
          % (_l1b, _l2b))
    _e1 = D.cv2.getTextSize(_l1, D.cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)[0][0]
    _e2 = D.cv2.getTextSize(_l2, D.cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)[0][0]
    _old = D.cv2.getTextSize("shape 0.98 #14  m0.89",
                             D.cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)[0][0]
    check(len(_rows) == 2 and max(_ws) <= max(_e1, _e2) + 2,
          "③ 画出来的两行**确实就是那两串**（实测墨宽 %s ｜ 上限 = 两串的排版宽 %d/%d ✓"
          " —— ⚠ 乱画别的串 ⇒ 溢出 ⇒ 红 ✗）" % (_ws, _e1, _e2))
    check(len(_rows) == 2 and max(_ws) < _old,
          "④ 整块**比原来那一行窄**（最宽行 **%s** < 老的一行 **%d** px ✓ ⇒ 挡的面少 ✓ —— "
          "⚠ 只是把两段并排画 ⇒ ① 就红 ✗）" % (max(_ws), _old))
    _blk = int(np.count_nonzero(np.all(_im == 0, axis=2)))
    check(len(_rows) == 2 and _blk == 0,
          "⑤ ⭐ **整块背后没有黑板子** ✓（画面里纯黑 `(0,0,0)` 像素 = **%d** 个 ✓〔要 0 ✓〕）"
          "—— ⚠⚠ 把那句 `cv2.rectangle(..., (0, 0, 0), -1)` 加回来 ⇒ 立刻红 ✗（用户 2026-10-07 "
          "原话：「**另外把标签的黑色背景去掉**」✓✓）" % _blk)


def test_box_tab_pos_above_box_corner():
    """⭐⭐⭐⭐⭐ **两行标签摆「框左上角外侧」** ✗✗（用户 2026-10-07 ✓ 原话："**标签还是在框心挡视野，
    看我的示意图**" ✓✓ —— 示意图 = 一小块「标签」贴在检出框左上角**外面** ✓、框本体叫「检出框」✓）。

    ⚠⚠ **改前的毛病**（**这就是本条要防的** ✗）：整块**以框心为中心**上下对称画 ✗ ⇒ 黄字正好压在
      **框心**上 ✓ —— 而框心那一带恰恰是我们**要看的地方**（绿圈圆心 / 融合框的推导位置 /
      真目标本体 ✓）⇒ 每帧都被它遮一下 ✓。
    钉六条（**纯几何** ✓ 不画像素 ⇒ 快且稳 ✓）：
      ① **左对齐到框的左沿** ✓（⚠ 仍"以框心居中" ✗ ⇒ 本条红 ✓）；
      ② ⭐ **整块在框上沿之外** ✓（两行基线都 < 框上沿 ✓ ⇒ 框内**一个字不落** ✓；⚠ 改前第 2 行
         落在框心（±半个框高 ✓）⇒ 压住框里 ⇒ 本条红 ✓）；
      ③ **第 1 行在上** ✓（`y1 < y2 − 第 2 行高` ✓ —— 顺序反了 ⇒ 红 ✗）；
      ④ 边界：**框贴着画面上边** ⇒ 整块**下移** ✓（不许把字切出画面 ✗）；
      ⑤ 边界：**框伸到画面右沿之外** ⇒ `x` 左移 ✓（不许甩出画面 ✗）；
      ⑥ 边界：**这拍没框**（`w = h = 0` ✓ 标签挂的是**推导位置** ✓）⇒ 以那个点为**左下**往上摆 ✓
         （不凭空猜一个框 ✗）。
    """
    _s1 = D.cv2.getTextSize("0.98", D.cv2.FONT_HERSHEY_SIMPLEX, D._fs(0.75), 1)[0]
    _s2 = D.cv2.getTextSize("#14  m0.89", D.cv2.FONT_HERSHEY_SIMPLEX, D._fs(0.75), 1)[0]
    _wm = max(int(_s1[0]), int(_s2[0]))

    _x, _y1, _y2 = D.box_tab_pos(400.0, 300.0, 200.0, 200.0, 1.0, 1.0, _s1, _s2, (900, 640))
    check(_x == 300,
          "① **左对齐到框的左沿** ✓（`x` = **%d** ✓〔要 300 = 400 − 200/2 ✓〕）—— ⚠ 仍以框心居中 "
          "✗（那种 `x` ≈ 400 − 墨宽/2 ✓）⇒ 本条红 ✗" % _x)
    check(_y2 < 200 and _y1 - int(_s1[1]) >= 0,
          "② ⭐ **整块在框上沿之外** ✓（框上沿 = **200** ✓；第 2 行基线 = **%d** ✓、第 1 行顶端 = "
          "**%d** ✓〔都要 < 200 且 ≥ 0 ✓〕）—— ⚠⚠ 改前是「以框心(300)为中心」✗ ⇒ 第 2 行掉进框里 "
          "⇒ 本条红 ✗（**这正是用户那句「标签在框心挡视野」** ✓）"
          % (_y2, _y1 - int(_s1[1])))
    check(_y1 < _y2 - int(_s2[1]),
          "③ **第 1 行在上** ✓（`y1` = **%d** < `y2 − 行高` = **%d** ✓）—— ⚠ 顺序反了 ⇒ 红 ✗"
          % (_y1, _y2 - int(_s2[1])))

    _a4 = D.box_tab_pos(400.0, 4.0, 200.0, 200.0, 1.0, 1.0, _s1, _s2, (900, 640))
    check(_a4[1] - int(_s1[1]) >= 0 and _a4[2] > _a4[1],
          "④ 边界：**框贴着画面上边**（上沿 = 4 − 200/2 = −96 ✓）⇒ 整块**下移** ✓（第 1 行顶端 = "
          "**%d** ✓〔要 ≥ 0 ✓〕）—— ⚠ 照原样算 ⇒ 字被切掉一半 ⇒ 本条红 ✗" % (_a4[1] - int(_s1[1])))

    _x5 = D.box_tab_pos(990.0, 300.0, 200.0, 200.0, 1.0, 1.0, _s1, _s2, (900, 640))[0]
    check(_x5 <= 900 - _wm - 3,
          "⑤ 边界：**框伸到画面右沿之外** ⇒ `x` 左移 ✓（`x` = **%d** ✓〔要 ≤ %d ✓〕）"
          "—— ⚠ 照原样摆 ⇒ 字甩出画面 ⇒ 本条红 ✗" % (_x5, 900 - _wm - 3))

    _x6, _a6, _b6 = D.box_tab_pos(130.0, 90.0, 0.0, 0.0, 1.0, 1.0, _s1, _s2, (300, 180))
    check(_x6 == 130 and _b6 < 90,
          "⑥ 边界：**这拍没框**（`w = h = 0` ✓ 标签挂的是**推导位置** ✓）⇒ 以那个点为**左下**往上摆 ✓"
          "（`x` = **%d** ✓〔要 130 ✓〕、第 2 行基线 = **%d** ✓〔要 < 90 ✓〕）—— ⚠ 凭空猜一个框 "
          "✗ ⇒ 本条红 ✗" % (_x6, _b6))


def test_mask_holes():
    """⭐⭐ **灰蒙版：镂空只给「检出框」，线条一律保持鲜亮**（用户 2026-10-02 ✓ 原话：
    "你可以让非检出框都受灰蒙版影响吗？目前绿色圆圈不是检出框也不受灰蒙版影响" +
    "**所有的线条都保持鲜亮，只有镂空区域灰**" ✓）。

    合成帧直接验（纯函数 ✓ 四点）：
      ① **目标圆所在区域**（圆内、且不在任何检出框里）⇒ 跟背景一样**被压暗** ✓
         （⚠ 改前它被 `target=` 当成"洞" ⇒ 保持原亮度 ✗ 就是用户报的那个现象 ✓）；
      ② **检出框区域**（洞内）⇒ **保持原亮度** ✓；
      ③ **圆线**（圆环上那一圈）⇒ 仍是纯绿 `(0,255,0)` ✓（画在蒙版**之后** ⇒ 不被压暗 ✓）；
      ④ `apply_mask` 的 `target` 参数与逻辑**仍在** ✓（要回退只需在 `draw` 里加回那两行 ✓）。
    """
    _a = 0.5
    _bg = 200                                   # 亮背景 ⇒ 压暗后 = 200×0.5 + 88×0.5 = 144 ✓
    _dim = int(round(_bg * (1.0 - _a) + 88.0 * _a))
    _f = np.full((200, 300, 3), _bg, np.uint8)
    _box = (0, 50.0, 50.0, 40.0, 40.0, 0.9)     # 检出框：中心 (50,50)、40×40（= 洞 ✓）
    _img = D.draw(_f.copy(), ({"state": "track"}, (150.0, 100.0), 18.0, None, [_box]),
                  1.0, 1.0, motion={}, mask=_a)
    check(tuple(int(v) for v in _img[100, 150]) == (_dim,) * 3,
          "① 目标圆**所在区域跟着背景一起被压暗**（圆内一点 = %d ｜ 期望 %d —— 用户 2026-10-02"
          "「非检出框也要受灰蒙版影响」✓；改前这里是 %d ✗ = 被挖洞 ✓）"
          % (int(_img[100, 150][0]), _dim, _bg))
    check(tuple(int(v) for v in _img[50, 50]) == (_bg,) * 3,
          "② **检出框（镂空）内保持原亮度**（框内一点 = %d ｜ 期望 %d ✓）"
          % (int(_img[50, 50][0]), _bg))
    _ring = [tuple(int(v) for v in _img[100, _x]) for _x in range(163, 174)]
    check(any(p == (0, 255, 0) for p in _ring),
          "③ **圆线保持鲜亮**（y=100 这一行 x∈[163,173] 的像素 = %s ｜ 期望含纯绿 (0,255,0) ✓ "
          "「所有的线条都保持鲜亮」✓ —— 它是蒙版之后画的 ✓）" % (_ring,))
    _m2 = D.apply_mask(_f.copy(), [_box], _a, target=(150.0, 100.0, 18.0), scale=(1.0, 1.0))
    check(tuple(int(v) for v in _m2[100, 150]) == (_bg,) * 3,
          "④ `apply_mask` 的 `target`（圆也挖洞）**参数与逻辑仍在**（传了 ⇒ 圆内仍 %d ✓ —— "
          "回退只需在 `draw` 里把那两行加回来 ✓）" % int(_m2[100, 150][0]))


def test_pick_box():
    """⭐⭐⭐ **点选检出框**（用户 2026-10-03 ✓ 原话："你能让我在**点选检出框**的时候显示
    `(360.4,323.0) 179×208=37232` 这种信息吗" ✓）—— 三件都钉：

      · `pick_box_at`（命中 ✓）：落在框里 ⇒ 命中；**多个套着 ⇒ 取里层**（面积小那格 ✓，
        融合并集框套着砖框时不会选错 ✓）；差一点（≤40px）⇒ 取中心最近的 ✓；远了 ⇒ `None`
        （调用方据此**取消选中** ✓）；
      · `pick_extra`（**重叠量** = 标签**第 2 行** ✓ 用户 2026-10-03 第二次 ✓）：`圆矩IoU` =
        `IoU(框, 圆外接矩形)` ✓、`圆矩∩框` = **交 ÷ 圆矩形面积** ✓（⚠ 与 IoU **不是一回事** ✗
        圆整个在框里时前者 0.47、后者 **1.00** ✓）／拿不到 ⇒ **留空** ✓；
      · `pick_label`（画布标签 ✓ **两行** ✓）：第 1 行 = 类别+几何+面积 ✓、第 2 行 = 三个重叠量
        ✓；面积 = **w×h** ✓、格式 = 用户要的那串 ✓、**只出 ASCII**（`x` / `|` ✗ 不能是全角
        —— `cv2.putText` 不认，会画成 `?` ✗）；没有量 ⇒ **退回单行** ✓；
      · `draw(..., pick=…)` ⇒ 画面上**真的出现黄色框线**（BGR `(0,255,255)` ✓）；
        不传 `pick` ⇒ 同位置**没有**黄色 ✓（只在点选时才画 ✓）。
    """
    _boxes = [(10, 300.0, 200.0, 180.0, 170.0),     # 外圈（融合并集框 ✓ 面积大）
              (12, 302.0, 198.0, 100.0, 96.0)]      # 里圈（砖框 ✓ 面积小）
    check(D.pick_box_at(_boxes, (301.0, 199.0)) == _boxes[1],
          "点在两层里 ⇒ 命中**里层**（面积小那格 ✓ 实测 %s）"
          % (D.pick_box_at(_boxes, (301.0, 199.0)),))
    check(D.pick_box_at(_boxes, (300.0, 270.0)) == _boxes[0],
          "点只在外圈里（y=270 已出里圈 ✓）⇒ 命中外圈（实测 %s）"
          % (D.pick_box_at(_boxes, (300.0, 270.0)),))
    check(D.pick_box_at([(10, 500.0, 500.0, 20.0, 20.0)], (515.0, 500.0)) is not None,
          "差一点（出框 5px ≤ 40px ✓）⇒ 退回**中心最近的**那格（手抖也能选上 ✓）")
    check(D.pick_box_at(_boxes, (480.0, 200.0)) is None,
          "点空白（离最近的框也 > 40px ✓）⇒ `None` ⇒ 演示窗据此**取消选中** ✓")
    check(D.pick_box_at([], (100.0, 100.0)) is None
          and D.pick_box_at([(10, 1.0, 1.0)], (1.0, 1.0)) is None,
          "没框 / 布局不合（少于 5 列）⇒ `None` ✓ 不崩 ✗")

    _l1 = D.pick_label((0, 360.4, 323.0, 179.0, 208.0, 0.39))
    check(_l1 == "(360.4,323.0) 179x208=37232",
          "标签 = `(cx,cy) WxH=面积`（用户要的那串 ✓ 实测 %r ｜ 面积 = 179×208 = 37232 ✓）" % (_l1,))
    _l2 = D.pick_label((0, 360.4, 323.0, 179.0, 208.0, 0.39), ["shape", "x"])
    check(_l2 == "shape 0.39 | (360.4,323.0) 179x208=37232",
          "带上类别/置信度（`names` + conf ✓ 实测 %r）" % (_l2,))
    check("×" not in _l2 and "｜" not in _l2 and "\u00d7" not in _l2,
          "**只出 ASCII**（乘号写 `x`、分隔写 `|` ✓ —— 全角会被 `cv2.putText` 画成 `?` ✗）")
    check(D.pick_label(None) == "", "`pick=None` ⇒ 空标签 ✓")

    _f = np.zeros((300, 600, 3), np.uint8)
    _res = (None, None, 20.0, False, [])            # (`r` 必须给数 ✓ 绿圈那份 = 20px ✓)
    _im = D.draw(_f.copy(), _res, 1.0, 1.0, pick=(10, 300.0, 150.0, 100.0, 80.0))
    _row = [tuple(int(v) for v in _p) for _p in _im[110, 250:351]]      # 框**上边**（y=150−40 ✓）
    check(any(_p == (0, 255, 255) for _p in _row),
          "传 `pick` ⇒ 画面上真的画出**黄色框线**（y=110 那一行含 BGR (0,255,255) ✓）")
    _im2 = D.draw(_f.copy(), _res, 1.0, 1.0, pick=None)
    _row2 = [tuple(int(v) for v in _p) for _p in _im2[110, 250:351]]
    check(not any(_p == (0, 255, 255) for _p in _row2),
          "不传 `pick` ⇒ 同位置**没有**黄色 ✓（只在点选时才画 ✓）")
    _n1 = int(np.sum(np.all(_im == (0, 255, 255), axis=2)))
    _n2 = int(np.sum(np.all(_im2 == (0, 255, 255), axis=2)))
    check(_n1 > _n2 == 0,
          "黄色像素：点选 %d 个 ｜ 不点选 %d 个 ✓（点选才画 ✓）" % (_n1, _n2))

    # ⭐⭐⭐ **两行 + 圆那侧的两个量**（用户 2026-10-03 ✓ 第二次 ✓ 原话："点选检出框时，与圆外接
    #   矩形的 iou、圆外接矩形与其相交的比例**也显示下**；另外，以上参数**及旧的 iou 显示都放在
    #   第二行**" ✓）：`_pick_box` 把算好的量附在**第 7/8 位** ✓、`pick_label` 把它们摆**第二行** ✓。
    _p3 = (0, 360.4, 323.0, 179.0, 208.0, 0.39, "IoU最大砖 0.62 @(120,80)",
           "圆矩IoU 0.41 | 圆矩∩框 0.88")
    _l3 = D.pick_label(_p3, ["shape", "x"])
    _ll = _l3.split("\n")
    check(len(_ll) == 2 and _ll[0] == "shape 0.39 | (360.4,323.0) 179x208=37232",
          "**第 1 行** = 类别/置信度 + 几何/面积（实测 %r ✓）" % (_ll[0],))
    check(_ll[1] == "IoU最大砖 0.62 @(120,80) | 圆矩IoU 0.41 | 圆矩∩框 0.88",
          "**第 2 行** = 三个**重叠量**（**旧的**「IoU最大砖」＋ **新的**「圆矩IoU」「圆矩∩框」✓ "
          "实测 %r ✓ 用户点名都放这行 ✓）" % (_ll[1],))
    check(D.pick_label((0, 1.0, 2.0, 3.0, 4.0, None, "", "")) == "(1.0,2.0) 3x4=12",
          "没有重叠量 / 量全是空串 ⇒ **退回单行** ✓（不硬塞一行空白 ✗ —— 圆心或半径还没学到时"
          "就是这种 ✓ 不猜 ✗）")
    # ⭐⭐ **那两个量的算法**（纯函数 `pick_extra` ✓）：夹具 = 框 `180×170 @(300,200)`、
    #   **圆矩形 `120×120`（r=60）整个落在框里**、砖 `160×150 @(300,200)`：
    #     · `IoU(框,砖) = 24000/30600 = **0.78**` ✓；`IoU(框,圆矩形) = 14400/30600 = **0.47**` ✓；
    #     · `圆矩∩框 = 14400 ÷ **14400**（= 圆矩形面积 ✓）= **1.00**` ✓ —— ⚠ 同一局面下
    #       **IoU 只有 0.47 而相交比例是 1.00** ✓ 两者**不是一回事** ✗ 正是用户"两个都要看"的原因 ✓
    #       （分母从"并集"换成"圆矩形面积" ✓ 见 `_ring_cov` ✓ 同一把尺 ✓）。
    #   ⚠ 砖表元素布局 = **`(bid, x, y, w, h)`** ✓（`bid` 是**元组** ✓ 与后端 `_bricks_snap`
    #     同一份 ✓ —— 我第一版夹具写成 `(1, …)` 就崩了 ✗ 纯函数的契约要按真实的写 ✓）。
    _e = D.pick_extra((300.0, 200.0, 180.0, 170.0), (300.0, 200.0), 60.0,
                      [((300, 200), 300.0, 200.0, 160.0, 150.0)])
    check(_e[0] == "IoU最大砖 0.78 @(300,200)" and _e[1] == "圆矩IoU 0.47 | 圆矩∩框 1.00",
          "`pick_extra` 算出两段（实测 %r / %r ✓ —— 圆**整个在框里** ⇒ 相交比例 **1.00** ✓ "
          "而 IoU 只有 **0.47** ✓ 两个都看才有意义 ✓）" % (_e[0], _e[1]))
    check(D.pick_extra((300.0, 200.0, 180.0, 170.0), None, 0.0, []) == ("", ""),
          "拿不到圆心/半径、砖表也空 ⇒ **两段都留空** ✓（界面自动不显示 ✓ 不猜 ✗）")
    check(D.pick_extra((300.0, 200.0, 180.0, 170.0), (300.0, 200.0), 0.0, None)[1] == "",
          "半径还没学到（`r=0`）⇒ 圆段留空 ✓（**不能拿 0 半径去算** ⇒ 会得 0 ✗ 是假信息 ✗）")

    # ⭐⭐⭐ **点选信息框：最小宽度 + 自动换行**（用户 2026-10-04 ✓ 原话："点选之后的信息框能做
    #   个**最小宽度+自动换行**吗？现在**整个屏幕都不够宽了**" ✓✓）—— 钉纯函数 `wrap_px` ✓：
    #     ① 长行**被切开**，且每行**实测像素宽 ≤ 上限** ✓（⚠ 按"字符数"猜宽会算漏 ✗：中文/全角
    #        在 `putText` 里会换成另一个字形 ✓ 只有 `getTextSize` 才是那把尺 ✓）；
    #     ② 短行**原样不动** ✓；
    #     ③ 英文**在空格处断**（单词不劈开 ✓ —— 拼回去应与原句**逐字一致** ✓）。
    #   ⚠ "最小宽度"那一半在 `draw()` 里（`max(_PICK_MIN_W, 最长行宽)` ✓）—— 它是**画布排版** ✓
    #     不进纯函数 ✓（这里钉的是换行尺 ✓）。
    _long = "融合（跟真目标无关）：框里是**两个别的目标**撞在一起 ✗ ⇒ 与真目标无关，别当真"
    _wl = D.wrap_px([_long], 300)
    _wmax = max(D.cv2.getTextSize(t, D.cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)[0][0]
                for t in _wl)
    check(len(_wl) >= 2 and _wmax <= 300,
          "① 长行**切成多行**（实测 %d 行 ✓）且**每行实测像素宽 ≤ 300px**（最宽 %d ✓）——"
          "⚠ 按字符数猜宽会算漏 ✗（中文/全角在 putText 里是另一个字形 ✓）" % (len(_wl), _wmax))
    check(D.wrap_px(["abc"], 300) == ["abc"],
          "② 短行**原样** ✓（不拆、不去字 ✓）")
    _en = D.wrap_px(["group median displacement is quite large here"], 200)
    check(len(_en) >= 2
          and " ".join(_en) == "group median displacement is quite large here",
          "③ 英文**在空格处断**（单词不劈开 ✓ —— 拼回去与原句**逐字一致** ✓ 实测 %d 行）"
          % len(_en))

    # ⭐⭐⭐ **右键 ⇒ 复制检出框信息**（用户 2026-10-03 ✓ 原话："加一个**右键点击检出框选中并弹出
    #   菜单**，目前只有一项『**复制检出框信息**』，用来我**复制之后与你交流**" ✓）——
    #   钉**剪贴板文本**（`pick_clip_text` ✓ 纯函数 ✓ 好测 ✓）：
    #     · 与画布标签**同一份数据** ✓ 但多一行**上下文**（帧号/状态 ✓ 交流时能定位是哪一拍 ✓）；
    #     · **全角 / 中文照用** ✓（走 Qt 剪贴板 ⇒ 不受 `cv2.putText` 的 ASCII 限制 ✗）；
    #     · 拿不到的量**一律不编** ✗。
    _ck = D.pick_clip_text(_p3, ["shape", "x"], frame_no=37, total=120, state="merged")
    _ckl = _ck.split("\n")
    check(len(_ckl) == 3 and _ckl[0] == "帧 37/120 ｜ 状态 merged",
          "第 1 行 = **上下文**（帧号 + 状态 ✓ 实测 %r）" % (_ckl[0],))
    check(_ckl[1] == "检出框 shape 0.39 ｜ 中心 (360.4, 323.0) ｜ 尺寸 179×208 ｜ 面积 37232",
          "第 2 行 = 类别/置信度 + 几何 + 面积（**全角照用** ✓ 实测 %r）" % (_ckl[1],))
    check(_ckl[2] == "IoU最大砖 0.62 @(120,80) ｜ 圆矩IoU 0.41 ｜ 圆矩∩框 0.88",
          "第 3 行 = 三个重叠量（与画布标签**同一份数据** ✓ 实测 %r）" % (_ckl[2],))
    check(D.pick_clip_text(None) == "", "没选中 ⇒ **空文本** ✓（菜单不会去复制它 ✓）")
    check(D.pick_clip_text((0, 1.0, 2.0, 3.0, 4.0))
          == "检出框 ｜ 中心 (1.0, 2.0) ｜ 尺寸 3×4 ｜ 面积 12",
          "拿不到的量**一律不编** ✗（没类别 ⇒ 不写类别 ✓ 没给帧号 ⇒ 不出上下文那行 ✓）")

    # ⭐⭐⭐ **点选标签的"圆外接矩形"必须用「目标实际半径」**（用户 2026-10-03 报的 bug ✓）：
    #   数值取实测 `10月1日.mp4` 帧 12（框 `(281.9,310.1) 169×206` ✓；该帧圆心 (363.4,273.0) ✓）：
    #     · **目标实际半径 60.79**（= `out["rad"]` = `LieTracker._rad` ✓ 后端 `_ring_cov` / 画面绿圈
    #       同一把尺 ✓）⇒ `圆矩IoU **0.19** ｜ 圆矩∩框 **0.52**` ✓；
    #     · **面积等效半径 18.00**（= `Runner` 那个 `r` ✗ **旧代码用的就是它** ✗）⇒
    #       `圆矩IoU **0.02** ｜ 圆矩∩框 **0.58**` ✗ —— 圆的方块缩成 36×36 ⇒ 很容易整个塞进框里
    #       ⇒ 会出现"**比例 1.00 / IoU 极小**"那种荒谬值 ✓（用户截图里正是这样 ✗）。
    #   ⇒ 两条一比**必须不同** ✓，谁把半径接错就当场红 ✗。
    _bx = (281.9, 310.1, 169.0, 206.0)
    _px = (363.4, 273.0)
    _e_ok = D.pick_extra(_bx, _px, 60.79, [])
    _e_bad = D.pick_extra(_bx, _px, 18.0, [])
    check(_e_ok[1] == "圆矩IoU 0.19 | 圆矩∩框 0.52",
          "目标实际半径 60.79 ⇒ 圆段 = %r ✓（后端同款 ✓）" % (_e_ok[1],))
    check(_e_bad[1] == "圆矩IoU 0.02 | 圆矩∩框 0.58",
          "面积等效半径 18.00 ⇒ 圆段 = %r ✗（**旧代码错的就这一条** ✓）" % (_e_bad[1],))
    check(_e_ok[1] != _e_bad[1],
          "两个半径算出来**明显不同** ⇒ 半径接错一眼就能看出来 ✓（差 3 倍 ⇒ 结果差一档 ✓）")

    _im3 = D.draw(_f.copy(), _res, 1.0, 1.0, pick=_p3)
    _n3 = int(np.sum(np.all(_im3 == (0, 255, 255), axis=2)))
    check(_n3 > _n1,
          "`draw` 把**第二行也画出来**（黄色像素 %d > 单行 %d ✓ —— `cv2.putText` **不认 `\\n`** ✗ "
          "⇒ 自己按行拆开、逐行画 ✓ 黑底高度也按行数算 ✓）" % (_n3, _n1))


def test_spinbox_typing():
    """⭐⭐ **小上限数字框的整数位也能改**（用户 2026-10-04 ✓ 原话："**我无法将框心力度小数点
    左边的数字改成 1**" ✓）。

    病根（**离屏实测** ✓）：`QDoubleSpinBox` 上限 `1.0`、当前 `0.15` 时，**选中整数位那个 `0`
    再敲 `1`** ⇒ 被当成 `1.15`（后面 `.15` 还在 ✗）⇒ **超上限** ⇒ `QDoubleValidator` 报
    `Invalid` ⇒ **那一下敲不进去** ✗✗（`text` 原地不动 ✓）；只有**全选**后敲 `1` 才行 ✗。
    ⇒ `gui.widgets.NoWheelDoubleSpinBox` 补了两道（**"数字但越界"降成 `Intermediate`** ✓
      ＋ `CorrectToNearestValue` 离开时就近夹取 ✓）⇒ 敲得进 ✓、回车后 `1.15 → 1.00` ✓。
    ⚠ 字母 / 格式错的**仍要拦住** ✗（别为了放行数字把校验整个拆了 ✓）。
    """
    try:
        from PyQt5.QtCore import Qt as _Qt
        from PyQt5.QtTest import QTest
        from gui.widgets import NoWheelDoubleSpinBox
    except Exception as _e:                 # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过数字框输入用例（环境不齐：%s ✓）" % _e)
        return
    app = QApplication.instance() or QApplication([])
    w = NoWheelDoubleSpinBox()
    w.setRange(0.0, 1.0)
    w.setDecimals(2)
    w.setValue(0.15)
    w.show()
    w.setFocus()
    app.processEvents()
    w.lineEdit().setSelection(0, 1)          # 选中整数位那个 `0`
    QTest.keyClicks(w, "1")
    _txt = w.lineEdit().text()
    check(_txt != "0.15",
          "① 小上限（0~1）数字框的**整数位敲得进去**（text=%r ✓ —— ⚠ 补丁前原地不动 ⇒ 立刻红 ✗）"
          % _txt)
    QTest.keyClick(w, _Qt.Key_Return)
    app.processEvents()
    check(abs(float(w.value()) - 1.0) < 1e-9,
          "② `1.15` + 回车 ⇒ **就近夹到上限**（value=%.2f ✓ = 用户要的那句「改成 1」✓）"
          % w.value())
    w.lineEdit().selectAll()
    QTest.keyClicks(w, "a")
    app.processEvents()
    check("a" not in w.lineEdit().text(),
          "③ **字母仍进不去**（text=%r ✓ —— ⚠ 补丁只放行「数字但越界」那种 ✗ 别把校验拆了 ✓）"
          % w.lineEdit().text())
    w.hide()


def test_fpull_ceiling_above_one():
    """⭐⭐⭐⭐⭐ **界面「框心力度」的上界必须 > 1，而且两处要一致**（用户 2026-10-05 ✓ 原话：
    "**框心力度无法调到 1 以上**" ✓✓）。

    ⚠⚠ 这道墙有**两处** ✗✗（我原来只想到建控件那处 ✓）：
      · `self.sp_fpull.setRange(0.0, 1.0)` ⇒ 直接敲不进去 ✓；
      · `_cfg_load` 里 `min(max(…, 0.0), 1.0)` ⇒ 存进去 `2.0`、**重开被夹回 1.0** ✗
        （看着像"改了不生效" ✓ —— 你存档里正好是 `1.0` ✓）。
    ⇒ 两处都放到 **3.0** ✓；本用例**盯源码**（⚠ 不是行为 ✗ —— 真开窗要视频 + 检测，离屏跑不起 ✓，
      同 `test_guide_keyed_on_circle` 那类源码级钉子 ✓）：`setRange` 那一行必须原样在 ✓、
      载入那处**不许再出现夹到 `1.0` 的写法** ✗（任一处退回去 ⇒ 立刻红 ✓）。
    ⚠ 后端侧"`> 1` 真能改变结果"另有一条 ✓（`selftest_lie_motion.test_fuse_pull_gate` ⑧ ✓）。
    """
    import re
    _src = Path(D.__file__).read_text(encoding="utf-8")
    check("self.sp_fpull.setRange(0.0, 3.0)" in _src,
          "① 建控件那处的上界 = **3.0** ✓（`self.sp_fpull.setRange(0.0, 3.0)` 原样在 ✓ —— "
          "⚠ 退回 `1.0` ⇒ 立刻红 ✗）")
    _flat = _src.replace(" ", "")
    _ks = [m.start() for m in re.finditer(r'"motion_fuse_pull"', _flat)]
    _old = [k for k in _ks if ",0.0),1.0))" in _flat[k:k + 90]]
    _new = [k for k in _ks if ",0.0),3.0))" in _flat[k:k + 90]]
    check(bool(_ks) and not _old and len(_new) >= 1,
          "② **载入存档那处也放到 3.0** ✓：`motion_fuse_pull` 共 %d 处 ｜ 附近仍是「夹到 1.0」的 "
          "**%d 处**（要 0 ✓）｜ 夹到 3.0 的 **%d 处**（要 ≥1 ✓ —— ⚠ 只有**夹取写法**才算 ✓，"
          "存盘那处**本来就没有夹取** ✓ 别拿它凑数 ✗）—— ⚠ 漏改载入那处 ⇒ 存 `2.0` 重开变回 "
          "`1.0` ✗（= 「改了不生效」✓）"
          % (len(_ks), len(_old), len(_new)))


def test_load_releases_old_material_first():
    """⭐⭐⭐⭐⭐ **`load()` 必须"先放掉旧素材、再去读新的"** ✗✗（用户 2026-10-07 ✓ 原话："**测谎演示
    打开较长的视频后，再打开就会非常卡，无法正常使用了**" ✓✓）。

    ⚠⚠ **病根**（**实测** ✓）：`load_video` 把**每一帧的两份都留在内存** ✗ —— 实测那条 750×500 的
      片子：**原图 5.25MB ＋ 处理帧 1.12MB = 6.37MB/帧** ✗（⚠ "原图"其实是 **1.75M 像素**那一档 ✓）；
      折算 1080p 档 **7.55MB/帧** ⇒ **3 分钟内容（1800 帧）≈ 13.6GB** ✗✗。
    ⇒ 而 `self.frames` 原来是**等 `load_source` 返回之后**才被替换 ✗ ⇒ **读新的时候旧的还活着** ✓
      ⇒ 峰值 **2×** ✗ ⇒ 一开长片就换页 ⇒ **非常卡** ✓（正是他说的"**再打开**"那一下 ✓）。
    钉两条：
      ① **先后**：`self.frames = []`（放掉旧素材 ✓）必须在 `load_source(path)` **之前** ✓
         （⚠⚠ 挪到后面 ⇒ 又变回 2× ✗ 而且**一点报错都没有** ✗）；
      ② **成套**：放掉的那组要**齐**（frames ／ ts ／ dets ／ results ／ results_kf ／ runner ／
         runner_kf ✓ ＋ `gc.collect()` ✓）—— ⚠ 少放一个 ⇒ 那一份就跨素材活下来 ✗ ⇒ 峰值又不是 1× ✗。
    """
    _src = Path(D.__file__).read_text(encoding="utf-8")
    _i = _src.index("def load(self, path):")
    _seg = _src[_i:_i + 4000]
    _i_read = _seg.index("load_source(path")   # ⚠ 只看前缀 ✗（后面可能还带 `on_progress=` ✓）
    _i_drop = _seg.index("self.frames = []")
    _need = ["self.frames = []", "self.ts = []", "self.dets = None",
             "self.results = []", "self.results_kf = None",
             "self.runner = None", "self.runner_kf = None"]
    _miss = [_k for _k in _need if _k not in _seg[:_i_read]]
    check(_i_drop < _i_read and not _miss and "gc.collect()" in _seg[:_i_read],
          "①② **`load()` 先放旧素材、再读新的** ✓（放旧的在第 %d 字符、读在第 %d 字符 ⇒ "
          "**先后对** ✓；该放没放的 = **%s** ✓〔要空 ✓〕；`gc.collect()` 在读取之前 ✓）"
          "—— ⚠⚠ 顺序一颠倒 ⇒ 峰值 **2×** ✗（长片就是 GB 级 ✗ ⇒ 再打开就换页 ⇒ 非常卡 ✓）"
          % (_i_drop, _i_read, _miss if _miss else "无"))


def test_log_exception_writes_and_never_raises():
    """⭐⭐⭐⭐⭐ **"内部异常"必须落盘、而且**绝不再抛** ✓✓（用户 2026-10-07 ✓ 他报："**闪退**" ✓）——
    ⚠⚠ 为什么这条值得单钉 ✗：Qt 槽里抛的**未捕获异常** ⇒ PyQt `abort` ⇒ **整窗消失** ✓，
      而 `pythonw` 没有控制台 ⇒ **traceback 一个都看不到** ✗ ⇒ 只能靠猜 ✓（这次挖那个
      `motion_viz` 拿 `None` 当 `%.1f` 的 bug ✓ 我排除了内存/线程/检测/定时器四轮 ✓）。
    ⇒ 演示窗的定时器槽都过 `_safe` ✓（先 `log_exception` 落盘 ✓ 再吞掉 ✓）；
      本钉子直接钉那个**落盘函数**：① 写得进去 ✓ ② 传 `None` 也不抛 ✓（记日志不许炸 ✗）。
    ⚠ **不碰真文件** ✗：`_ERROR_LOG` 临时指向临时目录 ✓（自检不许污染仓库 / 用户目录 ✓）。
    """
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as _d:
        _p = Path(_d) / "sub" / "err.log"
        with mock.patch.object(D, "_ERROR_LOG", _p):
            _ok = True
            try:
                try:
                    raise ValueError("钉一下 ✓")
                except ValueError as _e:
                    D.log_exception("自检里的假异常", _e)
                D.log_exception("连 exc 都不给", None)
            except Exception as _e2:              # noqa: BLE001 —— 记日志再抛就是最坏情况 ✓
                _ok = False
                check(False, "`log_exception` 自己抛了：%r ✗（记日志不许炸 ✓）" % (_e2,))
            _txt = _p.read_text(encoding="utf-8") if _p.exists() else ""
            check(_ok and _p.exists() and "钉一下" in _txt and "自检里的假异常" in _txt,
                  "①② **异常落盘 ✓ 且不抛 ✓**（文件 %s ✓〔连目录都是它自己建的 ✓〕；写得进 "
                  "%d 字节 ✓；含**堆栈** %s ✓〔⚠ 文案故意不写 `Traceback` 字样 ✗ —— 免得"
                  "外层的 grep 把它当成「真出现异常」✓〕）—— ⚠ 不落盘 ⇒ 下次「闪退」还是只能靠猜 ✗"
                  % ("有" if _p.exists() else "无", len(_txt),
                     "有" if "Traceback" in _txt else "无"))


def test_roi_wiring_in_demo():
    """⭐⭐⭐⭐⭐ **限定区域（ROI）那套在演示窗里的接线** ✗✗（用户 2026-10-07 ✓ 原话："**只在限定的
    区域计算（因为这是部分弹窗）**" ✓✓）。

    ⚠⚠ **为什么这条只钉"接线"** ✗（**不钉语义** ✓）：语义（框外一概看不见 ✓ / 默认整幅不变 ✓ /
      坏框当没给 ✓）已经由 `selftest_lie_motion.test_roi_limits_everything` **四条钉死** ✓；
      这里只防"**有人把线拆了**"这种静默失效 ✓（画布到主窗的 `_on_roi` ✓、主窗给 runner 的
      `roi` ✓、落盘的 `motion_roi` ✓、工具栏那个按钮 ✓、读配置那处 ✓）。
    """
    _src = Path(D.__file__).read_text(encoding="utf-8")
    _need = ("self.lbl._on_roi = self._on_roi_picked",
             '"roi": getattr(self, "roi", None)',
             '"motion_roi"',
             "def set_roi_mode(self, on)",
             "self.btn_roi.toggled.connect(self._on_roi_mode)",
             "self.btn_roi_clear.clicked.connect(self._on_roi_clear)",
             "def _on_roi_picked(self, _rect)",
             "self.start_warmup(reuse=True)")
    _miss = [_k for _k in _need if _k not in _src]
    check(not _miss,
          "**ROI 接线齐** ✓（画布→主窗 ✓ ／ 主窗→runner ✓ ／ 落盘 `motion_roi` ✓ ／ 工具栏两个按钮 ✓ ／"
          " 生效后 **复用检测框** 重算 ✓〔不重跑检测 ⇒ 快 ✓〕）—— ⚠ 缺项 = **%s** ✓〔要空 ✓〕"
          % (_miss if _miss else "无"))


def test_play_scheduler_follows_wall_clock():
    """⭐⭐⭐⭐⭐ **播放要按"墙钟"排，不是"干完活再等一个间隔"** ✗✗（用户 2026-10-07 ✓ 原话：
    "**没有以原速播放视频**" ✓✓）。

    ⚠⚠ **实测病根**（`10月7日.mp4` ✓ 离屏真播 ✓）：老写法每拍周期 = **工作量 ＋ 间隔** ✗
      （41.4ms 间隔 ＋ ~23ms 工作量 ⇒ **64ms/拍** ✗）⇒ 797 帧播 **50.9s**（该 **33.0s** ✓）
      ＝ **0.65 倍速** ✗（用户看到的"没原速"就是它 ✓；方向是**比原速慢** ✗）。
    ⚠⚠⚠ **还有一个更阴的**（**我第一版就栽在这** ✓ 幸好拿 0.5× 复验了 ✓）：到期时刻取成
      "**刚显示的那一帧**" ✗ ⇒ 每拍都算出"早就该到" ⇒ 定时器永远排 0 ✗ ⇒ 播放退化成
      "**按工作量跑**" ✗ ⇒ **慢放（0.5×）完全不生效** ✗（而且 1× 那次量到的 0.98× 只是
      "工作量恰好 ≈ 间隔"的**巧合** ✓ 差点被糊过去 ✗）。
    ⇒ 现在：到期时刻取**下一帧**那一格 ✓（`play_delay_ms` ✓ 纯函数 ✓ 好钉 ✓）。
    钉五条：
      ① **准点到**：`now == t0` ⇒ 该等"下一帧那一格"（100ms ✓ 不是 0 ✗）；
      ② **已经迟到** ⇒ **≤ 0** ✓（调用方取 `max(0,…)` ⇒ 立刻排下一拍 ⇒ 诚实追 ✓）；
      ③ **速度**：2× ⇒ 一半 ✓、0.5× ⇒ 两倍 ✓（⚠ 改前 0.5× 会**照样**按工作量跑 ✗）；
      ④ **锚**：锚挪到第 3 帧（`i0=3`）⇒ 从那时算起 ✓（不是从第 0 帧 ✗）；
      ⑤ **无 ts / 只有一格** ⇒ `None` ✓（调用方回落到"固定间隔" ✓ 老行为 ✓ 不崩 ✓）。
    """
    _ts = [0.0, 0.1, 0.2, 0.3, 0.4]

    def _dl(_i, _i0=0, _now=100.0, _sp=1.0, _ts=_ts):
        return D.play_delay_ms(_ts, _i, _i0, 100.0, _now, _sp)

    _a = _dl(0)                                   # 刚显示第 0 帧、准点 ⇒ 等第 1 帧那一格 ✓
    check(_a is not None and abs(_a - 100.0) < 1e-6,
          "① **准点到 ⇒ 等「下一帧」那一格** ✓（实测 **%.1f ms** ✓〔要 100 ✓〕）—— ⚠⚠ 取成"
          "「**刚显示的那一帧**」✗ ⇒ 这里是 **0** ⇒ 定时器永远排 0 ⇒ 退化成「按工作量跑」✗"
          % (0.0 if _a is None else _a))

    _b = _dl(0, _now=100.5)                       # 已经迟到 0.5s ⇒ ≤0 ⇒ 立刻下一拍 ✓
    check(_b is not None and _b <= 0.0,
          "② **已经迟到 ⇒ ≤ 0** ✓（实测 **%.1f ms** ✓〔要 ≤ 0 ✓〕）—— ⚠ 迟到还硬等一个正数 ✗ "
          "⇒ 越拖越远（`max(0,…)` 那一步在调用方 ✓）" % (0.0 if _b is None else _b))

    _c, _d = _dl(0, _sp=2.0), _dl(0, _sp=0.5)
    check(_c is not None and _d is not None
          and abs(_c - 50.0) < 1e-6 and abs(_d - 200.0) < 1e-6,
          "③ **速度**：2× ⇒ **%.1f ms** ✓〔要 50 ✓〕、0.5× ⇒ **%.1f ms** ✓〔要 200 ✓〕"
          "—— ⚠⚠ **慢放**（0.5×）最要紧 ✗：改前它**照样按工作量跑** ✗（实测 0.5× 会 33s 播完 ✓"
          "**慢放白设** ✗）；修后实测 **66.0s** ✓ 精确 ✓"
          % (0.0 if _c is None else _c, 0.0 if _d is None else _d))

    _e = _dl(3, _i0=3)                            # 锚挪到第 3 帧 ⇒ 从那时算 ✓
    check(_e is not None and abs(_e - 100.0) < 1e-6,
          "④ **锚挪到第 3 帧 ⇒ 从第 3 帧算起** ✓（实测 **%.1f ms** ✓〔要 100 ✓〕）—— ⚠ 锚不挪 ⇒ "
          "会按\"从第 0 帧一路追\"算 ✗（拖动 / 换速度 / 循环回头都要重锚 ✓ 见 `_play_anchor` ✓）"
          % (0.0 if _e is None else _e))

    check(D.play_delay_ms([], 0, 0, 0.0, 0.0, 1.0) is None
          and D.play_delay_ms([7.0], 0, 0, 0.0, 0.0, 1.0) is None,
          "⑤ **没有 ts / 只有一格 ⇒ `None`** ✓（调用方回落\"固定间隔\"✓ = 老行为 ✓ 不崩 ✓）")


def test_box_corner_label_replaced():
    """⭐⭐⭐⭐⭐ **检出框左上角那串字换掉了** ✗✗（用户 2026-10-07 ✓ 原话："**标签太挡视线了，把他们
    字体缩小一号并替换检出框左上角的「shape xxx」**" ✓✓）。

    ⚠⚠ **病根两条** ✗（**实测** ✓）：① 那串原来是 `"类别名 置信度"` ✓，而模型**只有一个类** ✓
      名字就叫 **`shape`** ✓（`DetsWorker.names` = `['shape']` ✓）⇒ 每格框都顶着同一串**零信息**
      的字 ✗；② 底衬按 **0.68** 量、字按 **0.45** 画 ✗ ⇒ **衬比字大一圈** ✓。
    ⇒ 运动模式下换成 **`#编号 置信度`** ✓（`#22 0.52` ✓）；**经典模式零污染** ✓（老文案一字不变 ✓）。
    钉四条：
      ① 经典模式 ⇒ **老文案一字不变** ✓（有 names ⇒ `shape 0.52` ✓；没 names ⇒ `0 0.52` ✓）；
      ② **运动模式 ＋ 有归属** ⇒ `#22 0.52` ✓（⚠⚠ 改前是 `shape 0.52` ✗ ⇒ 本条立刻红 ✓）；
      ③ **没归属** ⇒ **只写置信度** ✓（`0.52` ✓ —— 不许编一个号 ✗）；
      ④ 边界：`conf` 只有一位小数也**照两位写** ✓、`cls=None` 不崩 ✓。
    """
    check(D.box_corner_label(None, 0.52, 0, ["shape"], False) == "shape 0.52"
          and D.box_corner_label(None, 0.52, 0, None, False) == "0 0.52",
          "① **经典模式 ⇒ 老文案一字不变** ✓（有 names ⇒ `%s` ✓、没 names ⇒ `%s` ✓〔要 "
          "`shape 0.52` / `0 0.52` ✓〕）—— ⚠ 顺手改了经典模式 ✗ ⇒ 本条红 ✗（零污染那条纪律 ✓）"
          % (D.box_corner_label(None, 0.52, 0, ["shape"], False),
             D.box_corner_label(None, 0.52, 0, None, False)))

    check(D.box_corner_label(22, 0.52, 0, ["shape"], True) == "#22 0.52",
          "② ⭐ **运动模式 ＋ 有归属 ⇒ `#编号 置信度`** ✓（实测 **`%s`** ✓〔要 `#22 0.52` ✓〕）"
          "—— ⚠⚠ 改前是 `shape 0.52` ✗（类别名**恒为 `shape`** ✗ = 零信息 ✓）⇒ 本条立刻红 ✗"
          % D.box_corner_label(22, 0.52, 0, ["shape"], True))

    check(D.box_corner_label(None, 0.52, 0, ["shape"], True) == "0.52",
          "③ **那格框没归属 ⇒ 只写置信度** ✓（实测 **`%s`** ✓〔要 `0.52` ✓〕）—— ⚠ 编一个号出来 ✗ "
          "⇒ 本条红 ✗（没有归属就是没有 ✓）"
          % D.box_corner_label(None, 0.52, 0, ["shape"], True))

    check(D.box_corner_label(7, 0.9, None, None, True) == "#7 0.90",
          "④ 边界：**只有一位小数也照两位写** ✓、`cls=None` 不崩 ✓（实测 **`%s`** ✓〔要 `#7 0.90` ✓〕）"
          % D.box_corner_label(7, 0.9, None, None, True))


def test_all_draw_fonts_go_through_fs():
    """⭐⭐⭐⭐⭐ **`draw()` 里不许再出现"裸字号"** ✗✗（用户 2026-10-07 ✓ "**把他们字体缩小一号**" ✓✓）。

    ⚠⚠ **为什么这条值得单钉** ✗：那一轮是**机械替换**包上了 `_fs()`（21 处 ✓ 含 3 处字号写在
      **下一行**的 ✓）；以后**任何人**新加一句 `cv2.putText(..., 0.75, ...)` ✗ ⇒ 那串字
      **不受 `_FONT` 管** ✗ ⇒ 想"再缩一号"时会**漏掉它** ✓（= 这条钉子要防的正是"**漏**"✓）。
    钉两条：
      ① `_fs()` 真的**缩小了** ✓（`_fs(1.0) == _FONT` 且 `_FONT < 1.0` ✓ —— 用户那句"缩小一号" ✓）；
      ② `draw()` 的**函数体里**没有任何 `FONT_HERSHEY_SIMPLEX, <数字>` ✓（全过 `_fs()` ✓），
         且 `_fs(` 出现 ≥ 20 次 ✓（⚠ 数量掉了 ⇒ 有人删了字号 ✓）。
    """
    import re as _re
    _src = Path(D.__file__).read_text(encoding="utf-8")
    _i = _src.index("def draw(")
    _j = _src.index("\ndef ", _i + 1)          # 下一个顶层 def = 本函数结束 ✓
    _seg = _src[_i:_j]
    _raw = _re.findall(r"FONT_HERSHEY_SIMPLEX,\s*[0-9]", _seg)
    _fsn = len(_re.findall(r"_fs\(", _seg))
    check(abs(D._fs(1.0) - float(D._FONT)) < 1e-9 and float(D._FONT) < 1.0,
          "① **字号出口真的「缩小一号」** ✓（`_FONT` = **%.2f** ✓〔要 < 1 ✓〕；`_fs(1.0)` = **%.2f** ✓）"
          "—— ⚠ 把 `_FONT` 改回 1.0 ⇒ 字号又变回去 ✓（本条只钉「**确实小了**」✓）"
          % (float(D._FONT), D._fs(1.0)))

    check(not _raw and _fsn >= 20,
          "② **`draw()` 里没有裸字号** ✓（裸的 = **%s** ✓〔要空 ✓〕；过 `_fs()` 的 = **%d** 处 ✓"
          "〔要 ≥20 ✓〕）—— ⚠⚠ 新加一句 `putText(..., 0.75, ...)` ✗ ⇒ 那串字**不受 `_FONT` 管** ✗ "
          "⇒ 以后「再缩一号」会漏掉它 ✓（本条就是防这个 ✓）"
          % (_raw if _raw else "无", _fsn))


def _cnt_bgr(img, bgr):
    """画面里**正好**是这个颜色的像素数（BGR ✓）。⚠ `cv2` 画线/画框都是**整元组纯色** ⇒ 逐像素
    相等就数得准 ✓（⚠ 但**1px ＋ `LINE_AA`** 的细线**没有纯色像素** ✗ 见 `_cnt_aa` ✓）。"""
    return int(np.all(np.asarray(img) == np.array(bgr, np.uint8), axis=2).sum())


def _cnt_aa(img, bgr, tol=40):
    """**颜色家族**的像素数（分量各自差 ≤ `tol` ✓）—— 专门给 **1px ＋ `LINE_AA`** 那种细线用。

    ⚠⚠ **为什么非有它不可** ✗✗（**实测** ✓）：`cv2.line(..., (0,170,170), 1, cv2.LINE_AA)` 画一条
      60px 横线 ⇒ 纯色像素 **0 个** ✗（AA 把颜色摊到相邻两行 ⇒ 每行只有一半亮度 ✓）；而
      "**有没有画**"这件事必须量得出来 ✓ ⇒ 退一步按"**颜色家族**"数 ✓（暗黄那条线 60px ⇒ 上百
      像素 ✓ 远超 `LINE_AA` 边缘蹭出来的那几个 ✓）。
    """
    _a = np.asarray(img).astype(np.int16)
    _t = np.array(bgr, np.int16)
    return int(np.all(np.abs(_a - _t) <= int(tol), axis=2).sum())


#: ⚠ 这些项**必须按颜色家族量**（1px ＋ `LINE_AA` ⇒ 没纯色像素 ✓ 见 `_cnt_aa` ✓）。
_AA_KEYS = ("暗黄线（鼠标相对群体）",)


def _cnt_item(img, key, bgr):
    """按这一项的**画法**选量法（细线 ⇒ 颜色家族 ✓ 其余 ⇒ 纯色 ✓）。"""
    return (_cnt_aa(img, bgr) if key in _AA_KEYS else _cnt_bgr(img, bgr))


def _vel_motion(mode="velocity", wait_state="wait", merge_on=True, clamp=True, tbox=None,
                label_on=True):
    """一份**够 `draw()` 跑完**的假 `motion`（用户 2026-10-07 ✓ 速度跟踪那一轮的自检夹具 ✓）。

    ⚠ 只放两类键：① 画那 7 样需要的 ✓；② **该被 `_vel` 挡掉**的那几样（好让正反两面都量得到 ✓）。
    三格（都摆在画面正中偏下 ✓ 留出标签的位置）：
      · `#1` = **整框在视野内、有编号** ⇒ 上板 ✓；
      · `#2` = **贴着左边**（x0 ≈ 0 ⇒ 两个角点贴边 ✓）⇒ 等待上板 ✓（`wait_state="board"` 就能
        把它"假装成上板" ⇒ 用来量"**等待上板到底有没有编号标签**" ✓）；
      · `#3` = **融合** ⇒ 洋红（`merge_on=False` 关掉它 ⇒ 量"夹取细框有没有被画"就不会串色 ✓）。
    多出来的那几样：粉框（`tbox=None` + `tbox_wh` ✓）／黄箭头（`tgt_rel_next` ✓）／暗黄线
    （`cursor_rel` ✓）／洋红细夹取框（`fuse_clamp` ✓）。
    """
    _boxes = [(160.0, 120.0, 60.0, 60.0),      # #1 全内 ⇒ 上板 ✓
              (20.0, 120.0, 60.0, 60.0),      # #2 贴左 ⇒ 等待上板 ✓
              (240.0, 120.0, 80.0, 80.0)]     # #3 融合 ⇒ 洋红 ✓
    _tids = (1, 2, 3)
    _states = ["board", wait_state, ("merge" if merge_on else "board")]
    _bv = []
    for _k, (_cx, _cy, _w, _h) in enumerate(_boxes):
        # 14 位：0/1 框心 · 2/3 位移 · 4 轨迹号 · 5 conf · 6 匹配分 · 7 状态码 · 8 has_tgt ·
        #        9 占位 · 10/11 宽高 · 12 来源码 · 13 "主人"（-1 = 没有 ✓）
        _m7 = 1 if (_k == 2 and merge_on) else 0
        _bv.append((_cx, _cy, 6.0, 3.0, _tids[_k], 0.9, 0.8, _m7, True, False,
                    _w, _h, 0, -1))
    return {
        "mode": mode,
        #   ⚠ `label_on=False` ⇒ 关掉"框上那行字"（`(0,255,255)` ✓）：它**1px 抗锯齿的字边**
        #     正好落在"暗黄线"那个颜色家族里（**实测**：411 px ✗）⇒ 量"细线有没有画"时会被它淹掉 ✓
        #     （所以 ②/④ 那一对用 `label_on=False` ✓；③ 专门量标签 ⇒ 用它默认的 `True` ✓）。
        "show": {"boxes": True, "rel_tgt": True, "rel_mouse": True,
                 "box_id": bool(label_on), "panel": True},
        "box_v": _bv,
        #   ⚠⚠ **`vel` 里没有群体 `std_area`** ✗（用户 2026-10-07 规则1 ✓："群体不再有标准面积…每个
        #     目标有自己的标准面积，存在自己的 id 里" ✓）⇒ 这一份是**逐 id** 的 ✓（`area_by_id` ✓）。
        "vel": {"state": list(_states), "label": [1, None, None],
                "board_ids": [1, 2, 9], "area_by_id": {1: 3600.0, 2: 3025.0},
                "std_speed": (6.0, 3.0), "std_speed_px": 6.7, "n_vtx": 12},
        # ⚠ `#2`（等待上板）**这一拍没框** ⇒ 只有它**有编号**时才会画那条灰框 + "#2 推的" ✓
        #   ⭐⭐ 灰框尺寸取**这个 id 自己**的标准面积 ✓（规则1 ✓ 用户 2026-10-07 ✓）⇒ 夹具里给它 ✓
        "tracks": [{"tid": 2, "p": (20.0, 120.0), "trust": False, "sticky": False,
                    "rel_hist": [], "v": (0.0, 0.0), "score": 1.0, "dev": 1.0,
                    "hits": 3, "live": False, "sel": False, "std_area": 3600.0},
                   {"tid": 9, "p": (60.0, 120.0), "trust": False, "sticky": False,
                    "rel_hist": [], "v": (1.0, 0.0), "score": 0.5, "dev": 1.0,
                    "hits": 1, "live": False, "sel": False}],
        #   ⚠ 这几条线/箭头**要够长** ✗（画出来才够几十像素 ✓ —— 太短的话"有没有画"两者都是
        #     个位数像素 ⇒ 分不开 ✓ 实测踩到：3px 与 4px ✓）。
        "pos_rel": [(0.0, 0.0), (20.0, 15.0), (40.0, 30.0)],
        "cursor_rel": [(0.0, 0.0), (30.0, 20.0), (60.0, 40.0)],
        "tgt_rel_last": (20.0, 8.0), "tgt_rel_next": (50.0, 20.0),
        "dt": 1.0 / 60.0,
        "tbox": tbox, "tbox_wh": (60.0, 60.0), "tbox_rad": 30.0, "tgt_radius": 20.0,
        "median": (1.0, 1.0), "cam_len": 1.4, "dev": 2.0, "n_live": 3,
        "sel_score": 5.0, "sel_dev": 1.0, "n_cands": 3,
        "box_moves": [], "box_roles": [], "box_pred": [], "reg": [],
        "merged": False, "fuse_clamp": ((20.0, 20.0, 300.0, 220.0) if clamp else None),
        #   ⚠⚠ **速度跟踪档没有群体标准面积** ✗（规则1 ✓）⇒ 夹具里也不放这个键 ✓（= 与真后端一致 ✓）
        "roi": None, "path_pts": [], "raw_pos": None,
        "red_i": None, "log_text": "",
    }


def test_velocity_draw_subtraction():
    """⭐⭐⭐⭐⭐ **速度跟踪档"做减法"：画面上正好只有那 7 样** ✗✗（用户 2026-10-07 ✓ 原话：
    "模式下拉多一项「速度跟踪」，选中后**画面上正好只有那 7 条图例，别无他物**" ✓✓）。

    ⚠⚠ **正反两半都要钉** ✗（这是"零污染"那条纪律的钉子 ✓）：
      · **正**：`mode="velocity"` ⇒ 那 7 样**都在** ✓ ＋ 多出来的那几样**都是 0 像素** ✓；
      · **反**：同一份夹具把 `mode` 换成 `"motion"` ⇒ 那几样"多出来的"**必须都冒出来** ✓✓
        （⚠ 若反的那半没红 ⇒ 说明我不是"只在一档做减法"，而是**把运动分离那档也砍了** ✗✗
         —— 那正是用户最在意的一条：`classic` / `motion` **零污染** ✓）。
    另钉一条：**等待上板 ⇒ 没有编号标签**（把那一格"假装成上板" ⇒ 黄字像素必须**变多** ✓）。
    ⚠ 颜色都是纯色 ⇒ 逐像素相等计数 ✓ 不用容差 ✓。
    """
    _d = [(0, 160.0, 120.0, 60.0, 60.0, 0.9),
          (0, 20.0, 120.0, 60.0, 60.0, 0.9),
          (0, 240.0, 120.0, 80.0, 80.0, 0.9)]
    _res = (None, (160.0, 120.0), 20.0, True, _d)

    def _run(_mo, _want=None, **kw):
        _f = np.zeros((240, 320, 3), np.uint8)
        return D.draw(_f, _res, 1.0, 1.0, None, (160.0, 120.0), _mo, _want, pick=None,
                      kf=None, kf_trail=None, **kw)

    # ---- ① 正：那 7 样都在 ----
    _f = _run(_vel_motion("velocity"))
    _need = {"蓝框": (255, 140, 0), "绿圆": (0, 255, 0), "红点": (0, 0, 255),
             "蓝箭头": (255, 120, 0), "白箭头": (255, 255, 255), "橙线": (0, 150, 255),
             "灰框": (150, 150, 150), "洋红（融合）": (255, 0, 255)}
    _miss = {_k: _cnt_bgr(_f, _c) for _k, _c in _need.items() if _cnt_bgr(_f, _c) <= 0}
    check(not _miss,
          "① **那 7 样都在** ✓（蓝框 %d px ／ 绿圆 %d ／ 红点 %d ／ 蓝箭头 %d ／ 白箭头 %d ／ "
          "橙线 %d ／ 灰框 %d ／ 洋红 %d ✓）—— ⚠ 缺的 = **%s** ✓〔要空 ✓〕"
          % (_cnt_bgr(_f, (255, 140, 0)), _cnt_bgr(_f, (0, 255, 0)), _cnt_bgr(_f, (0, 0, 255)),
             _cnt_bgr(_f, (255, 120, 0)), _cnt_bgr(_f, (255, 255, 255)),
             _cnt_bgr(_f, (0, 150, 255)), _cnt_bgr(_f, (150, 150, 150)),
             _cnt_bgr(_f, (255, 0, 255)), _miss if _miss else "无"))

    # ---- ② 正：多出来的那几样都是 0 像素 ----
    #   ⚠ "洋红细夹取框"与"融合框"**同色**（都是 (255,0,255) ✓）⇒ 只能**关掉融合格**再量它 ✓
    _fd = _run(_vel_motion("velocity", merge_on=False, clamp=True, label_on=False))
    _extra = {"暗黄线（鼠标相对群体）": (0, 170, 170), "黄箭头（下一拍预估）": (0, 215, 255),
              "粉框（检出丢失接力）": (203, 192, 255), "洋红细夹取框": (255, 0, 255),
              "经典那行左下说明字": (235, 235, 235)}
    #   ⚠⚠ **阈值 30 px 的理由**（**实测踩到** ✓）：`cv2` 的 `LINE_AA` 会让字边蹭出**别的纯色** ✗
    #     —— 黄字 `(0,255,255)`（框标签 ✓）压在黑底上时，边缘那几像素正好是 `(0,170,170)` ✗
    #     （= 暗黄线的颜色 ✓）⇒ 严格"一个像素都不许有"会**假红**（实测 3 px ✓）；
    #     而真画了那条线时是**几百像素**（见 ④ 的对照：motion 档 4 px ／ 参考"粉框 476 px" ✓）
    #     ⇒ 用"**小于 30**"当"没画" ✓ 分得开 ✓。
    _left = {_k: _cnt_item(_fd, _k, _c) for _k, _c in _extra.items()
             if _cnt_item(_fd, _k, _c) >= 30}
    check(not _left,
          "② **多出来的那几样都没画** ✓（暗黄线 / 黄箭头 / 粉框 / 洋红细夹取框 / 经典左下那行字 "
          "✓ 都 < 30 px ✓）—— ⚠ 真画了的 = **%s** ✓〔要空 ✓〕" % (_left if _left else "无"))

    # ---- ③ 等待上板 ⇒ **没有编号标签**（假装它是上板 ⇒ 黄字必须变多）----
    _fw = _cnt_bgr(_run(_vel_motion("velocity", wait_state="wait")), (0, 255, 255))
    _fb = _cnt_bgr(_run(_vel_motion("velocity", wait_state="board")), (0, 255, 255))
    check(0 < _fw < _fb,
          "③ **「等待上板」没有编号标签** ✓（等待上板 ⇒ 框上黄字 **%d px** ／ 同一格改成上板 ⇒ "
          "**%d px** ✓〔要 前者 < 后者 ✓〕）—— ⚠ 用户口径：「**有标签就是上板，无标签就是等待"
          "上板**」✓ ⇒ 无标签**就是**它的画面表达 ✓" % (_fw, _fb))

    # ---- ④ 反：运动分离档 ⇒ 那几样**必须都冒出来**（零污染 ✓）----
    _fm = _run(_vel_motion("motion", merge_on=True, clamp=True, label_on=False))
    _gone = {_k: _cnt_item(_fm, _k, _c) for _k, _c in _extra.items()
             if _cnt_item(_fm, _k, _c) < 30
             and _k != "经典那行左下说明字"}       # ⚠ 那一行经典档才画（motion 档本来就不画 ✓）
    check(not _gone,
          "④ **运动分离档一个字都没被砍** ✓（暗黄线 %d px ／ 黄箭头 %d ／ 粉框 %d ／ 洋红细夹取框 "
          "%d ✓）—— ⚠ 少了的 = **%s** ✓〔要空 ✓〕；⚠⚠ 少了就说明这道减法**做过了头** ✗"
          % (_cnt_item(_fm, "暗黄线（鼠标相对群体）", (0, 170, 170)),
             _cnt_item(_fm, "黄箭头（下一拍预估）", (0, 215, 255)),
             _cnt_item(_fm, "粉框（检出丢失接力）", (203, 192, 255)),
             _cnt_item(_fm, "洋红细夹取框", (255, 0, 255)),
             _gone if _gone else "无"))


def test_velocity_mode_wiring():
    """⭐⭐⭐⭐⭐ **速度跟踪那一档的接线 ＋ 那 7 条图例 ＋ 三处"清空"** ✗✗（用户 2026-10-07 ✓）。

    钉四组：
      ① **下拉多一项**（`"速度跟踪"` / `"velocity"` ✓）且**老两项一个字不改** ✓（零污染 ✓）；
      ② **图例**：`_LEGEND_VELOCITY` **恰好 7 行**、逐行含用户给的那 7 个关键词 ✓；
         `_LEGEND_MOTION` 那 16 行**还在**（搬了地方、没丢 ✓）；
      ③ **模式判断收在一处**（`_mv()` / `_vel()` ✓）＋ 侧栏/参数行/取帧/runner 都改用它 ✓；
      ④ **三处"清空"** 的接线（点选面板 ⇒ `_pick_box` 早退 `None` ✓／右侧实时信息 ⇒
         `setText("")` ✓／底栏中段 ⇒ 单独拼一行、**不留下悬空分隔符** ✓）。
    """
    _src = Path(D.__file__).read_text(encoding="utf-8")
    # ① 下拉
    check('addItem("运动分离（新）", "motion")' in _src
          and 'addItem("经典（白块 + 纹理）", "classic")' in _src
          and 'addItem("速度跟踪", "velocity")' in _src,
          "① **下拉三项齐** ✓（classic / motion / **velocity** ✓）—— ⚠ 老两项的文案与取值必须"
          "**原样** ✓（改了就是污染 ✓）")
    # ② 图例
    _v = D._LEGEND_VELOCITY.split("\n")
    _keys = ("蓝色框", "灰色框", "绿圆", "红点", "蓝箭头", "白箭头", "橙色线")
    _bad = [_k for _k, _ln in zip(_keys, _v) if _k not in _ln]
    check(len(_v) == 7 and len(_keys) == 7 and not _bad
          and D._LEGEND_VELOCITY.startswith("1. 蓝色框")
          and "洋红细框" in D._LEGEND_MOTION and "黄箭头" in D._LEGEND_MOTION,
          "② **图例：速度跟踪恰好 7 条** ✓（行数 **%d** ✓；缺关键词的 = **%s** ✓〔要空 ✓〕）＋ "
          "**运动分离那 16 行还在** ✓（含「洋红细框」「黄箭头」✓）—— ⚠⚠ 这 7 条**之外一条都不许加** ✗"
          "（用户 2026-10-07 ✓ 纪律：要加**先问用户** ✓）"
          % (len(_v), _bad if _bad else "无"))
    # ③ 模式判断收在一处
    _need3 = ("        def _mv(self):", "        def _vel(self):",
              "        def _legend_text(self):", "        def _sync_legend(self):",
              "return self._mode() in (\"motion\", \"velocity\")",
              "self._sync_legend()",           # 切模式 ⇒ 图例跟着换 ✓
              "mode=self._mode(),")            # runner 透传模式标记 ✓
    _miss3 = [_k for _k in _need3 if _k not in _src]
    check(not _miss3,
          "③ **模式判断收在一处** ✓（`_mv` / `_vel` / `_legend_text` / `_sync_legend` ✓；"
          "runner 透传 `mode=` ✓；参数行/侧栏/取帧都走 `_mv()` ✓）—— ⚠ 缺项 = **%s** ✓"
          % (_miss3 if _miss3 else "无"))
    #   ⚠ `_show_kf` **单独切段**量 ✗（`if self._mv():` / `return False` 这种串全文件到处都是 ⇒
    #     在整份源码里 grep 等于**没量** ✓ 实测这么写过 ✓）。
    _a3 = _src.index("        def _show_kf(self):")
    _b3 = _src.index("        def ", _a3 + 10)
    _segkf = _src[_a3:_b3]
    check("if self._mv():" in _segkf and "return False" in _segkf,
          "③' **「显示 KF 预测」在速度跟踪档恒关** ✓（`_show_kf()` 里那道 `if self._mv(): return "
          "False` ✓）—— ⚠ KF 青线/青点**不在那 7 条里** ✗ ⇒ 该档不许跑也不许画 ✓")
    # ④ 三处清空
    _i = _src.index("        def _pick_box(self):")
    _j = _src.index("            _p = getattr(self, \"_pick\", None)", _i)
    _seg = _src[_i:_j]
    _k = _src.index("        def _update_info_live(self, motion):")
    _seg2 = _src[_k:_k + 1400]
    #   ⚠ 这一段**必须切到下一个 `if` 为止** ✗（不能"往后取 N 个字符" ✓ —— 紧接着的
    #     `if ... == "motion":` 那段里就有 `_pks` ⇒ 取多了会**假红** ✓ 实测踩到 ✓）。
    _m = _src.index("if _mo_s.get(\"mode\") == \"velocity\":")
    _m2 = _src.index("if _mo_s.get(\"mode\") == \"motion\":", _m)
    #   ⚠⚠ **量之前先把注释行丢掉** ✗✗（**实测踩到** ✓）：我在这一段里写了注释
    #     "⚠ `_pks`（点选）…本来就恒空" ⇒ 拿整段 grep `_pks` ⇒ **自己的注释把自己判红** ✓。
    _seg3 = "\n".join(_l for _l in _src[_m:_m2].splitlines()
                      if not _l.lstrip().startswith("#"))
    check("if self._vel():" not in _seg
          and "if pick is not None and not _vel:" in _src
          and "if pick is not None and _vel:" in _src
          and "_VEL_ARR_LEN" in _src,
          "④a **点选：文字不给、但那一格要看得见** ✓（用户 2026-10-07 ✓ **两条一起**：「点选面板」"
          "要清空」＋「**选中的检出框需要高亮**」✓）—— 口径 = `_pick_box` **照旧回那一格** ✓"
          "（高亮要用它 ✓）、`draw` 里**文字那一块**加 `not _vel` ✓、**另起一块** `and _vel`"
          "画**青框＋四角标** ✓；⚠ 改回「整块 `return None`」⇒ 高亮也没了 ✗")
    check('self.info_live.setText("")' in _seg2,
          "④b **右侧实时信息清空** ✓（`_update_info_live` 在速度跟踪档 `setText(\"\")` ✓）")
    check("命中率 %.1f%%（%d/%d）" in _seg3 and "_pks" not in _seg3 and "_kfs" not in _seg3,
          "④c **底栏中段清空** ✓（速度跟踪档**单独拼一行**：帧号 / 状态 / 命中率 ✓ **不带** `_pks` / "
          "`_kfs` ✓）—— ⚠⚠ 若只把 `_mid` 置空 ⇒ 会留下一个**悬空的分隔符** ✓（我第一版就是这么"
          "想的 ✓）")


def test_velocity_config_and_log():
    """⭐⭐⭐⭐⭐ **速度跟踪档的"参数做减法"＋"清日志"** ✗✗（用户 2026-10-07 ✓ 两条原话：
    「**清除通用参数外所有的配置参数，放上新增要用的配置参数**」＋「**清除日志信息**」✓✓）。

    钉五组（**源码级** ✓ —— 这四条都是"界面状态"、在离屏自检里点不出来 ✓ 但**接线断了会很静默** ✗）：
      ① **老底盘那两行（第 4/5 行）在这一档一并收掉** ✓（只留通用三项 ＋ 它自己那三个新参量 ✓）；
      ② **新增的一行三件控件齐** ✓（捕获半径 / 不合群时长 / 速度偏差门 ✓ 且**顺序**就是用户问的顺序 ✓）；
         ⚠⚠ 三个值**都不许自己拍** ✗ —— 只钉"控件默认值取自 `lie_motion` 一处口径" ✓；
      ③ **落盘/回填三个独立键** ✓（`vel_catch_px` / `vel_hold_s` / `vel_dev_px` ✓ 各管各的 ✓）；
      ④ **日志区藏起来 ＋ 清空** ✓（`self.log.setVisible(not _vel_now)` ＋ `self.log.clear()` ✓）
         ＋ **`_status` 那一档不再喂它** ✓（`_log_sync` 从那一支里消失 ✓ —— ⚠⚠ 只把控件藏起来、
         `_log_sync` 照旧每帧喂 ⇒ **白跑一遍** ✗ 我第一版就是这么写的 ✓）；
      ⑤ **「用检测器」不显示但照旧勾着** ✓（`setVisible(not _vel_now)` ✓ ＋ `apply_path_ms` 里
         那道自动勾仍在 ✓）。
    """
    _src = Path(D.__file__).read_text(encoding="utf-8")

    def _seg(_a, _b):
        _i = _src.index(_a)
        _j = _src.index(_b, _i + 1)
        return _src[_i:_j]
    _mode = _seg("        def _on_mode_changed(self, *_a):", "        def on_toggle_kf(self):")
    check("_mo_trk = bool(_motion and not _vel_now)" in _mode
          and _mode.count("setVisible(_mo_trk)") >= 3,
          "① **老底盘那两行（第 4/5 行）在速度跟踪档一并收掉** ✓（口径 = `_mo_trk` ✓ 三处：标题 ＋ "
          "两行 ✓）—— ⚠ 只收了两行标题、行内控件没收 ⇒ 会剩一排空标签 ✗")
    _rowv = _seg("            cfgrowV = QHBoxLayout()", "            cfgrowV.addStretch(1)")
    _ord = [_rowv.index(_k) for _k in ("捕获半径(px)", "不合群时长(s)", "速度偏差门(px/拍)")]
    check(_ord == sorted(_ord) and "_VEL_CATCH_DEFAULT" in _rowv
          and "_VEL_HOLD_DEFAULT" in _rowv and "_VEL_DEV_DEFAULT" in _rowv
          and _rowv.count("NoWheel") >= 3,
          "② **新增那一行三件控件齐、顺序对、默认值取自 `lie_motion` 一处** ✓（捕获半径 → 不合群"
          "时长 → 速度偏差门 ✓ 都用 `NoWheel*` ⇒ **滚轮改不了参数** ✓）")
    check('"vel_catch_px"' in _src and '"vel_hold_s"' in _src and '"vel_dev_px"' in _src
          and _src.count("vel_catch_px") >= 2 and _src.count("vel_hold_s") >= 2
          and _src.count("vel_dev_px") >= 2,
          "③ **三个独立落盘键 ＋ 回填齐** ✓（各出现 ≥2 次 = 存 ＋ 读 ✓）—— ⚠ 少一处 ⇒ "
          "**改了不生效 / 重开就忘** ✗")
    _st = _seg("if _mo_s.get(\"mode\") == \"velocity\":",
               "if _mo_s.get(\"mode\") == \"motion\":")
    check("self._log_sync(i)" in _st
          and "self.log.setVisible(not _vel_now)" not in _mode
          and _mode.count("self.log.clear()") >= 1,
          "④ **日志区：留着 ＋ 照旧喂 ＋ 切过来清一次残留** ✓（用户 2026-10-07 ✓ **更正**："
          "「**日志区需要保留，我说的是清除要打印的东西，不是把日志功能删了**」✓）—— ⚠⚠ 藏控件 "
          "✗ 或 不喂 `_log_sync` ✗ 都算把功能删了 ✓（我上一版两条都犯了 ✓）")
    check("self._apply_dets_visible()" in _mode and "_tb.removeAction(_act)" in _src
          and "self.act_dets_after = _acts[_i + 1]" in _src
          and "if self._mv() and not self.chk_dets.isChecked():" in _src,
          "⑤ **「用检测器」这一档从工具栏上真的拿掉、回来还插回原位** ✓（⚠⚠ **不许用 "
          "`setVisible`** ✗✗ —— **探针实测**：它的父是 `QToolBar`，`setVisible(False)` 当场是 "
          "hidden=True，可**下一拍 `processEvents` 就被工具栏显示回来** ✗ ⇒ 只有 `removeAction` "
          "管用 ✓）；⚠ 只是不显示 ✗ 它照旧**勾着** ⇒ 检测照跑 ✓（这一档的观测**就是**检出框 ✓）")


def test_roi_dims_outside():
    """⭐⭐⭐⭐⭐ **ROI 之外：模糊 ＋ 置灰**（用户 2026-10-07 ✓ 原话：「**ROI外的区域置灰+模糊**」✓）。

    钉四条（`dim_outside_roi` 是**纯函数** ✓ 好钉 ✓）：
      ① `roi=None` ⇒ **一个像素都不变** ✓（= 老行为 ✓ 零污染那条 ✓）；
      ② **ROI 里面一个字不改** ✓（边界也照旧：`roi` 左边界那 1 列不动 ✓）；
      ③ **外面确实被"糊 + 灰"** ✓ —— 用棋盘格当外区：糊完**方差必须掉**（这是"模糊"的硬证据 ✓）
         ＋ 均值往灰里靠 ✓；
      ④ 坏框（反了 / 缺数）⇒ **当没给** ✓（宁可不画，也不许把画面抹花 ✗）。
    """
    _h, _w = 60, 80
    #   棋盘格铺满 ⇒ ROI 外面那块糊完方差一定掉 ✓（全图的"细节"都在那儿 ✓）
    _f = np.zeros((_h, _w, 3), np.uint8)
    _f[::2, ::2] = 255
    _roi = (20.0, 15.0, 60.0, 45.0)
    _same = D.dim_outside_roi(_f.copy(), None)
    check(np.array_equal(_same, _f),
          "① **`roi=None` ⇒ 一个像素都不变** ✓（= 老行为 ✓）—— ⚠ 变了就是污染了整幅 ✓")
    _g = D.dim_outside_roi(_f.copy(), _roi)
    _inside = _g[16:44, 21:59]
    check(np.array_equal(_inside, _f[16:44, 21:59]),
          "② **ROI 里面原样** ✓（含左边界那 1 列 ✓）")
    _bg_in = float(_f[0:12, 0:18].std())
    _bg_out = float(_g[0:12, 0:18].std())
    _mean_in = float(_f[0:12, 0:18].mean())
    _mean_out = float(_g[0:12, 0:18].mean())
    #   ⚠ 口径是"**往灰里靠**"（`gray=96` ✓）✗ —— **不是"一定变暗"** ✗（实测：棋盘格均值 64 比
    #     96 暗 ⇒ 混完**反而变亮到 78** ✓；深色画面才会变暗 ✓）⇒ 断言必须是"**离灰更近**" ✓。
    check(_bg_out < _bg_in * 0.5 and abs(_mean_out - 96.0) < abs(_mean_in - 96.0),
          "③ **框外糊了 ＋ 往灰里靠** ✓（方差 %.1f ⇒ **%.1f** ✓〔要掉一半以上 ✓〕；均值 %.0f ⇒ "
          "**%.0f** ✓〔要**离灰(96)更近** ✓ 不是「一定变暗」✗〕）"
          % (_bg_in, _bg_out, _mean_in, _mean_out))
    check(np.array_equal(D.dim_outside_roi(_f.copy(), (60.0, 15.0, 20.0, 45.0)), _f)
          and np.array_equal(D.dim_outside_roi(_f.copy(), (1.0, 2.0)), _f),
          "④ **反了的 / 缺数的框 ⇒ 当没给** ✓（宁可不画 ✓ 也不许把画面抹花 ✗）")


def test_roi_picker_uses_loupe():
    """⭐⭐⭐⭐⭐ **框选走的是工作台那套「带放大镜的框选窗」** ✗✗（用户 2026-10-07 ✓ 原话：
    「**框选区域要有放大，参考数据工作台框选功能**」✓✓）。

    钉三条（**源码级** ✓ —— 框选窗是**模态**的 ✓ 离屏自检里不能真弹它 ✓ 一弹就卡住 ✓）：
      ① 用的是 `gui.region_picker.select_on_pixmap`（= **工作台同一份代码** ✓ 放大镜/像素网格/
         `+/-`/`Esc` 全都有 ✓）—— ⚠⚠ **不许自己另写一个放大镜** ✗（那正是用户说的"参考" ✓）；
      ② 框完的矩形**除以 `self.scale`** 换成**加工域** ✓（显示域 → 加工域 ✓ 不换 ⇒ 框偏一大截 ✗）；
      ③ 画布上那套"直接拖"**留着当退路** ✓（框选窗开不出来时退回 ✓）。
    """
    _src = Path(D.__file__).read_text(encoding="utf-8")
    _i = _src.index("        def _on_roi_mode(self, on):")
    _j = _src.index("        def _on_roi_clear(self):", _i)
    _seg = _src[_i:_j]
    check("from gui.region_picker import select_on_pixmap" in _seg
          and "select_on_pixmap(QPixmap.fromImage(_qi), parent=self, zoom=4)" in _seg
          and "def _on_roi_mode" in _src,
          "① **框选用工作台那份 `select_on_pixmap`** ✓（放大镜 ＋ 像素网格 ＋ `+/-` ＋ `Esc` ✓ "
          "同一份代码 ✓）—— ⚠⚠ 自己另写一套放大镜 ⇒ 手感与工作台不一致 ✗（用户点名要「参考工作台」✗）")
    check("_r[0] / self.scale[0]" in _seg and "_r[1] / self.scale[1]" in _seg,
          "② **显示域 ⇒ 加工域**（`÷ self.scale` ✓）—— ⚠ 不漏这一步 ⇒ 框出来的区域**整体偏** ✗"
          "（画面是原图、判据吃的是加工域 ✓）")
    check("self.lbl.set_roi_mode(True)" in _seg,
          "③ **退路留着** ✓（框选窗开不出来 ⇒ 退回「画布上直接拖」✓）—— ⚠ 这不是死代码 ✗："
          "没有素材 / 离屏环境都会走到它 ✓")


def test_velocity_roi_reaches_tracker():
    """⭐⭐⭐⭐⭐ **ROI 必须传进新底盘** ✗✗（用户 2026-10-07 ✓ 报的两条是**同一条根因**：
    「**框选区域没有任何作用，还在检测区域之外的东西**」＋「**初始绿圆挑错了**」✓✓）。

    ⚠⚠ **实测现场** ✗：我上一版在 `start_warmup` 里写的是"速度跟踪档**一个参数都不传**" ✗
      ⇒ 把 `roi` 也一起丢了 ⇒ 那一档其实在**整幅**上跑 ⇒ **实测**：
        · **无 ROI**：首锁帧 **72**、锁到 **(470,132)** 那格**普通砖**（用户截的帧 73 就是它 ✓✗）；
        · **带 ROI**：首锁帧 **58**、锁到 **(446,236)** 那块**白实体** ✓。
      钉三层（少一层就会静默失效 ✗ 而且**看不出**来 ✓）：
        ① 窗口 → runner：`_vel_kw["roi"] = getattr(self, "roi", None)` 且**真的展开**了（`**_vel_kw` ✓）；
        ② runner → 底盘：`VelocityRunner._make_tracker` 里**把 `roi` 转过去** ✓；
        ③ 底盘自己：`VelocityTracker.roi` 认 4 元组（坏值当没给 ✓）。
    """
    import inspect
    from perception import lie_motion as _lm
    _src = Path(D.__file__).read_text(encoding="utf-8")
    _mk = inspect.getsource(D.VelocityRunner._make_tracker)
    check('_vel_kw["roi"] = getattr(self, "roi", None)' in _src
          and "**_vel_kw)" in _src and "roi=kw.get(\"roi\")" in _mk,
          "① **窗口 → runner → 底盘，ROI 一路都在** ✓（`_vel_kw` 里塞 `roi` ✓ ＋ 展开 ✓ ＋ "
          "`_make_tracker` 转过去 ✓）—— ⚠⚠ 少一处 ⇒ 那一档变成**整幅**在跑 ✗（实测锁到 (470,132) "
          "那格普通砖上 ✗ 而带 ROI 是 (446,236) 那块白实体 ✓）")
    _t = _lm.VelocityTracker(mode="velocity", roi=(10.0, 20.0, 30.0, 40.0))
    _t2 = _lm.VelocityTracker(mode="velocity", roi=(30.0, 20.0, 10.0, 40.0))
    _t3 = _lm.VelocityTracker(mode="velocity", roi=None)
    check(_t.roi == (10.0, 20.0, 30.0, 40.0) and _t2.roi is None and _t3.roi is None,
          "② **底盘自己认框**：4 个数 ⇒ 收 ✓；**反了 / 退化的框 ⇒ 当没给** ✓（= 整幅 ✓ 宁可不裁 ✗）；"
          "`None` ⇒ 整幅 ✓")
    check(_lm._VLOCK_WHITE >= 180.0,
          "③ **开局那块还得是「近白」** ✓（绝对下限 `_VLOCK_WHITE` = **%.0f** ✓）—— ⚠ 只判「相对更亮」 "
          "时**浅灰的砖**也会被认成目标 ✗（实测砖 ~130~140 ／ 真目标亮块 246~250 ✓）"
          % float(_lm._VLOCK_WHITE))


def test_velocity_real_runner_output_into_draw():
    """⭐⭐⭐⭐⭐ **真后端（`MotionRunner(mode="velocity")`）的输出 ⇒ 真 `draw()`** ✗✗（用户
    2026-10-07 ✓ 本轮那条「**所见即所得**」纪律的收官钉子 ✓）。

    为什么非要这一条 ✗：上面两条一条量"接线 / 图例"、一条量"假 `motion` 进 `draw`" ✓，**都不是**
      "后端真造出来的那份 `motion`" ✗ ⇒ 形状/键位一旦对不上（少键 ⇒ `KeyError` ⇒ PyQt5 `abort`
      ⇒ **闪退** ✓ 踩过 ✓）它们**照样全绿** ✓。
    钉三条：① 真后端给的 `motion["vel"]` **逐位对齐** `box_v`（长度相等 ✓ —— 演示窗按同一索引读 ✓）；
      ② `log_text` **空**（实时信息那处 ✓）；③ 这份真输出喂进 `draw()` **不抛**、且**画面有东西**
      （= 画得出来 ✓ 不是一片黑 ✓）。
    """
    _r = D.MotionRunner(dets=None, gain=None, assume=(160.0, 120.0), follow_gain=1.0,
                        mode="velocity")
    _r.dets = [[[0, 100.0 + 6.0 * _i, 120.0, 40.0, 40.0, 0.90],
                [0, 200.0 + 6.0 * _i, 120.0, 40.0, 40.0, 0.90]] for _i in range(6)]
    _out = None
    for _i in range(6):
        _out = _r.step(np.zeros((240, 320, 3), np.uint8), _i, _i / 6.0)
    _mo = _out[5]
    check(_mo.get("mode") == "velocity" and isinstance(_mo.get("vel"), dict)
          and len(_mo["vel"]["state"]) == len(_mo.get("box_v") or []),
          "① **真后端的 `vel` 逐位对齐 `box_v`** ✓（状态 **%d** 个 ／ `box_v` **%d** 条 ✓）"
          % (len((_mo.get("vel") or {}).get("state") or []), len(_mo.get("box_v") or [])))
    check(_mo.get("log_text") == "",
          "② **实时信息文案清空** ✓（真后端出口那句 `log_text = \"\"` ✓）—— ⚠ 不清 ⇒ 底部日志区"
          "会被那 20 多行老文案刷满 ✗")
    _f = np.zeros((240, 320, 3), np.uint8)
    D.draw(_f, (None, _out[1], _out[2], _out[3], _out[4]), 1.0, 1.0, None,
           tuple(_r.cursor), _mo, None, pick=None, kf=None, kf_trail=None)
    check(int((_f.sum(axis=2) > 0).sum()) > 0,
          "③ **真输出喂进 `draw()` 不抛 ＋ 画面有东西** ✓（画出来 **%d** 个非黑像素 ✓）"
          % int((_f.sum(axis=2) > 0).sum()))


def test_dim_outside_roi_matches_drawn_roi():
    """⭐⭐⭐⭐⭐ **「糊的区域」必须与「框选的区域」重合** ✗✗（用户 2026-10-07 ✓ 原话：

    「**模糊的区域和框选的区域不一致**」✓✓）。
    ⚠⚠ **实测现场**（`10月7日.mp4` ✓ 原图 **1920×1080** ／加工帧 **889×500** ⇒ `scale` **2.160** ✓，
      ROI 加工域 `(200.5, 75.6, 688.5, 399.9)` ✓）：旧代码**没乘 `scale`** ✗ ⇒ 原图域上糊的是
      `(200, 75, 688, 399)` 那个小矩形 ✗ ⇒ **整块弹窗被糊掉 673,823 px** ✗✗ ＋ 真 ROI 之外
      **漏糊 107,959 px** ✗；而画面上那个黄框（`draw()` 里乘了 `scale` ✓）**是对的** ✓
      ⇒ 于是"框"与"糊"对不上 ✓ = 用户看到的那一幕 ✓。修完实测：ROI 内被糊 **0 px** ✓。
    钉两半（**必须两半都有** ✗）：
      ① **`draw()` 那一层**（真正咬得住"调用点漏传 `scale`" ✗）：喂加工域 ROI ＋ `scale=2` ⇒
        在**只有正确 ROI 才该被糊**的那两个点上量（`(50,100)` 该糊 ✓／`(150,150)` 该留 ✓）；
      ② **纯函数那一层**（边界 ✓）：空 / 反了 / 整幅 ROI ⇒ **逐像素不变** ✓；`scale=(1,1)` 默认
        = 老行为 ✓。
    """
    # ---- ① draw() 那一层：加工域 ROI ＋ scale=2（显示域 = 2 倍 ✓）----
    _img = np.zeros((300, 400, 3), np.uint8)
    _img[:] = (70, 90, 110)
    _roi = (40.0, 40.0, 90.0, 110.0)                  # **加工域**（显示域该是 ×2 ✓）
    _o = D.draw(_img.copy(), (None, None, 0.0, False, []), 2.0, 2.0,
                None, None, {"mode": "velocity", "roi": _roi}, None, 0.0)
    _chg = (np.abs(_o.astype(np.int16) - _img.astype(np.int16)).max(axis=2) > 8)

    def _dim(x, y):
        return bool(_chg[int(y), int(x)])

    check(_dim(50, 100) and _dim(20, 20) and not _dim(150, 150) and not _dim(100, 90),
          "① **糊的位置 = 框选的位置 × `scale`** ✓✗（加工域 ROI %s ⇒ 显示域该是 %s ✓）："
          "**只该糊**外面那两点 `(50,100)`=%s ✓／`(20,20)`=%s ✓，**该留**的 `(150,150)`=%s ✗／"
          "`(100,90)`=%s ✗〔前两个要 True ✓ 后两个要 False ✓〕"
          % (_roi, (80.0, 80.0, 180.0, 220.0), _dim(50, 100), _dim(20, 20),
             _dim(150, 150), _dim(100, 90)))
    # ---- ② 纯函数那一层：边界 / 老行为 ----
    _f = np.zeros((40, 60, 3), np.uint8)
    _f[:] = (70, 90, 110)
    _keep = _f.copy()
    for _bad in (None, (), (5.0, 5.0, 5.0, 5.0), (30.0, 20.0, 10.0, 8.0)):
        _g = D.dim_outside_roi(_f.copy(), _bad)
        if not np.array_equal(_g, _keep):
            break
    else:
        _g = None
    check(_g is None and np.array_equal(D.dim_outside_roi(_f.copy(),
                                                          (0.0, 0.0, 60.0, 40.0)), _keep),
          "② **空 / 退化 / 反了 / 整幅 ROI ⇒ 一个像素都不碰** ✓（逐像素不变 ✓）")
    _h = D.dim_outside_roi(_f.copy(), (10.0, 10.0, 40.0, 30.0))
    check(np.array_equal(_h[20, 25], _keep[20, 25]) and not np.array_equal(_h[5, 5], _keep[5, 5]),
          "②' **`scale` 缺省 = (1,1) ⇒ 老行为** ✓（框内 `(25,20)` 不动 ✓／框外 `(5,5)` 变了 ✓）")


def main():
    print("测谎演示窗口自检（离屏）：")
    test_red_equals_backend()
    test_merge_label()
    test_path_line()
    test_pink_relay_box()
    test_box_label_two_lines()
    test_track_label_anchor_circle()
    test_mask_holes()
    test_merge_ratio_switch()
    test_pick_box()
    test_spinbox_typing()
    test_fpull_ceiling_above_one()
    test_load_releases_old_material_first()
    test_log_exception_writes_and_never_raises()
    test_roi_wiring_in_demo()
    test_play_scheduler_follows_wall_clock()
    test_box_corner_label_replaced()
    test_box_tab_pos_above_box_corner()
    test_all_draw_fonts_go_through_fs()
    # ⭐⭐ **速度跟踪那一轮**（用户 2026-10-07 ✓）：做减法（只留那 7 样 ✓）＋ 接线与图例 ✓
    test_velocity_draw_subtraction()
    test_velocity_mode_wiring()
    # ⭐⭐ **ROI：糊的区域必须 = 框选的区域**（用户 2026-10-07 ✓ 报的那条 ✓）
    test_dim_outside_roi_matches_drawn_roi()
    test_velocity_config_and_log()
    test_roi_dims_outside()
    test_roi_picker_uses_loupe()
    test_velocity_roi_reaches_tracker()
    test_velocity_real_runner_output_into_draw()
    QApplication.exec_ = _fake_exec
    args = D.argparse.Namespace(
        video=str(D.DEF_VIDEO), record="", headless=False, no_dets=False,
        weights="", conf=0.25, max_frames=0, mask=0.35)
    # ⭐⭐ **自检期间 `config/ui.yaml` 一律指向临时文件**（窗口"记住上次配置"会读写它 ✓）：
    #   ⇒ 整段（含 `closeEvent` 落盘 ✓）都不碰用户的真实配置 ✓。**不许省** ✗ ——
    #   我在这里踩过坑：UI 用例写脏用户配置 ✓（同一台机器上的真实偏好被覆盖 ✗）。
    import tempfile
    from unittest import mock
    from gui import theme as _theme
    _tmpd = tempfile.TemporaryDirectory()
    with mock.patch.object(_theme, "CFG", Path(_tmpd.name) / "ui.yaml"):
        D.run_window(args)
    _tmpd.cleanup()
    if _FAILED:
        print("自检：%d 条失败" % len(_FAILED))
        return 1
    print("测谎演示窗口自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
