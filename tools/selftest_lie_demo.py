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
        # ⭐⭐ **`pos=None` 的帧不许把窗口画崩**（2026-09-30 用户报"打不开 闪退"的根因 ✓）：
        #   追踪器在 init / lost 帧报 `pos=None` ✓，而 `draw()` 里有一段（白箭头 = 轨迹速度）
        #   直接 `pos[0]` ⇒ `TypeError` ⇒ 整个窗口崩 ✗（pythonw 只打 traceback ⇒ 用户只看到
        #   闪退 ✗）。这里直接用**空位置 + 有 track_v** 的形态喂一次 `draw()` ✓。
        ok = True
        try:
            frame = win.frames[0][0].copy()
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
            im = D.draw(win.frames[idx][0].copy(),
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
            return D.draw(win.frames[k][0].copy(),
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
    ok = (len(xs) > 0 and xs.min() >= _bx1 - 40 and xs.max() <= _bx1
          and abs(ys.min() - _by0) <= 3)
    check(ok, "融合红框右上角没标「融合」（%d 个标像素 x[%s..%s] y[%s..%s] ｜ 期望在红框右上角 "
          "x[%.0f..%.0f] y≈%.0f ✗）"
          % (len(xs), xs.min() if len(xs) else "-", xs.max() if len(xs) else "-",
             ys.min() if len(ys) else "-", ys.max() if len(ys) else "-",
             _bx1 - 40, _bx1, _by0))
    im2 = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []),
                 1.0, 1.0, "hud", None, {"tbox": tb, "merged": False, "box_roles": []},
                 None, 0.0)
    n2 = int(np.all(im2 == _LAB, axis=2).sum())
    check(n2 == 0, "非融合红框也标了「融合」（%d 个标像素 ✗ 该 0 ✓）" % n2)


def test_path_line():
    """⭐⭐ **纳入预测的轨迹画成白线**（用户 2026-10-01 ✓ 原话："用白色的线把纳入预测的轨迹
    画出来" ✓）：`motion["path_pts"]`（加工域 ✓ 后端给的群体相对曲线 ✓）⇒ 连成白折线 ✓。

    合成帧验：有 `path_pts` 与没有的两份渲染，**白像素差集** = 白线本体 ✓（其它白标记
    （圆心/白箭头）两边都有 ⇒ 被差掉 ✓）。
    """
    frame = np.zeros((200, 300, 3), np.uint8)
    _pts = [(50.0, 100.0), (80.0, 100.0), (110.0, 120.0)]
    im = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []),
                1.0, 1.0, "hud", None, {"path_pts": _pts, "box_roles": []}, None, 0.0)
    im0 = D.draw(frame.copy(), (None, (150.0, 100.0), 20.0, True, []),
                 1.0, 1.0, "hud", None, {"path_pts": [], "box_roles": []}, None, 0.0)
    # ⚠ **用"近白"阈值**（≥140 ✓）而不是纯白 ✗：`cv2.LINE_AA` 抗锯齿在**黑底**上会把 1px
    #   白线混成灰色（实测峰值 ~218、端点 ~143 ✓）⇒ 纯白 `==255` 一个都抓不到 ✗✗。
    _W = np.array((140, 140, 140))
    _only = np.all(im >= _W, axis=2) & ~np.all(im0 >= _W, axis=2)
    n = int(_only.sum())
    ys, xs = np.nonzero(_only)
    # ⚠ 容差 ±4（端点画了个半径 3 的空心圆 ✓ + AA 边缘 ✓）
    ok = (n > 0 and xs.min() >= 50 - 4 and xs.max() <= 110 + 4
          and ys.min() >= 100 - 4 and ys.max() <= 120 + 4)
    check(ok, "纳入预测的轨迹画成白线（%d 像素 x[%s..%s] y[%s..%s] ｜ 期望覆盖折线 "
          "(50,100)→(110,120) ✓）"
          % (n, xs.min() if n else "-", xs.max() if n else "-",
             ys.min() if n else "-", ys.max() if n else "-"))


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


def main():
    print("测谎演示窗口自检（离屏）：")
    test_red_equals_backend()
    test_merge_label()
    test_path_line()
    test_pink_relay_box()
    test_mask_holes()
    test_merge_ratio_switch()
    test_pick_box()
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
