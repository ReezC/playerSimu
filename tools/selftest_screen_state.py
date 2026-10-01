# -*- coding: utf-8 -*-
"""M1「测谎报警」自检：模板检测 / 阈值 / 缺模板 / agent 接管 / 接线钉子。

跑法：`python -u -m tools.selftest_screen_state`（offscreen 环境变量照旧 ✓）。
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from perception import screen_state

ROOT = Path(__file__).resolve().parent.parent


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _frame_with(name):
    """合成一帧：把模板**原尺寸**贴进 1366×768 的噪声画布（同源缩放 ⇒ 匹配分≈1.0 ✓）。"""
    t = cv2.imread(str(screen_state.TEMPLATE_DIR / (name + ".png")),
                   cv2.IMREAD_COLOR)
    assert t is not None, "模板缺失：%s" % name
    frame = np.full((768, 1366, 3), 60, np.uint8)
    rng = np.random.default_rng(7)
    frame += rng.integers(0, 12, frame.shape, dtype=np.uint8)   # 加点噪声防退化 ✓
    h, w = t.shape[:2]
    frame[80:80 + h, 300:300 + w] = t
    return frame


def t_detect_synthetic():
    """三张模板各自贴进画布 ⇒ 判出对应状态；空画布 ⇒ combat。"""
    for name in screen_state.TEMPLATES:
        st, det = screen_state.check_frame(_frame_with(name))
        check(st == name and det.get("name") == name,
              "贴了「%s」却判成 %s（detail=%r）" % (name, st, det))
        check(det.get("score", 0) >= screen_state.MATCH_THRESHOLD,
              "命中分低于阈值（模板/缩放口径漂了 ✗）：%r" % det)
    frame = np.full((768, 1366, 3), 60, np.uint8)
    st, det = screen_state.check_frame(frame)
    check(st == "combat", "空画布判成了 %s（误报 ✗）：%r" % (st, det))


def t_threshold_reverse():
    """反向钉：阈值拉到不可能命中 ⇒ 一律 combat（没有阈值就没有「没弹窗」可言 ✗）。"""
    st, _ = screen_state.check_frame(_frame_with("lie_warn"), threshold=1.1)
    check(st == "combat", "阈值 1.1 还能命中（阈值没生效 ✗）：%s" % st)


def t_missing_templates():
    """模板目录不存在 ⇒ 如实回 combat + why（**绝不猜** ✗）。"""
    frame = _frame_with("lie_warn")        # ⚠ 先在**真目录**把帧造好，再 patch 目录 ✓
    with tempfile.TemporaryDirectory(prefix="lie_no_tmpl_") as td:
        with mock.patch.object(screen_state, "TEMPLATE_DIR", Path(td)), \
             mock.patch.object(screen_state, "_cache", None):
            st, det = screen_state.check_frame(frame)
    check(st == "combat" and det.get("why") == "no templates",
          "没模板时没如实报告 ✗：%s / %r" % (st, det))


def t_agent_takeover():
    """agent 接管：测谎弹窗期间**停发一切键**、任务保留并打打断点、结束恢复。"""
    import tempfile as _tf

    from core import behavior

    from tools.selftest_decision import Harness, fresh_settings

    logf = Path(_tf.mkdtemp(prefix="lie_m1_")) / "behavior.log"
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        behavior.configure(True, log=logf)
        s = fresh_settings()
        s.enabled = True
        h = Harness(s)
        from decision import route
        job = route.ClimbJob("L2", x=700.0, y1=100.0, y2=300.0, direction=1,
                             dst_set="上", src_set="下", tol_px=10, hold_ms=0,
                             timeout_s=600.0)
        h.agent.start_climb(job)
        h.clock0 = h.clock.t                   # ⚠ 直接调 tick 必须自设（_rec 靠它 ✓ 踩过）
        with h._patched():
            ws = h.ws(with_mob=False)
            ws.screen = "lie_game"
            ctx = h.agent.tick(ws)
            check(isinstance(ctx, dict) and ctx.get("keys") == []
                  and "界面接管" in str(ctx.get("reason", "")),
                  "测谎弹窗期间没停手（角色被锁操作，按键全无效 ✗）：%r" % ctx)
            check(h.agent._climb is job,
                  "接管把任务撤了（该保留 —— 报警结束从原地继续 ✗）")
            check(getattr(job, "_paused_at", None) is not None,
                  "进沿没给任务调 interrupted()（弹窗那几十秒会被当成"
                  "「卡住/超时」✗）")
            # ⭐ 断线侧（用户 2026-09-29："把断线的判断也做一下"）：ui_state 的界面 id
            #    同样接管 ✓ —— login_err = 断线提示框，重连动作归 reconnect.py，
            #    agent 只负责"别跟它抢着发战斗键" ✓
            ws1 = h.ws(with_mob=False)
            ws1.screen = "login_err"
            ctx1 = h.agent.tick(ws1)
            check(isinstance(ctx1, dict) and ctx1.get("keys") == []
                  and "login_err" in str(ctx1.get("reason", "")),
                  "断线提示画面没接管（该停手别跟重连抢键 ✗）：%r" % ctx1)
            ws2 = h.ws(with_mob=False)
            ws2.screen = "combat"
            ctx2 = h.agent.tick(ws2)
            check("界面接管" not in str(ctx2.get("reason", "")),
                  "画面恢复没恢复正常决策 ✗：%r" % ctx2)
            check(h.agent._climb is job, "恢复后任务丢了 ✗")
    finally:
        behavior.configure(old_en, log=old_log)
    txt = logf.read_text(encoding="utf-8") if logf.exists() else ""
    check("screen_takeover" in txt and "screen_resume" in txt,
          "screen_takeover / screen_resume 没落 behavior.log ✗：%r" % txt[:200])


def t_wiring_source_pins():
    """接线钉子（源码级）：检测拍 / 两路 ws / 状态行载荷 / 打点报警存帧 / agent 接管。"""
    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("self._screen_beat(raw)" in lt,
          "没把**原生帧**喂进界面状态检测拍（喂显示帧会被画框污染 ✗）")
    check(lt.count("self._screen_last") >= 3,
          "检测没限流（整帧匹配 ~0.1s，每帧都跑会吃掉推理预算 ✗）")
    check(lt.count("ws.screen = self._screen_state") >= 2,
          "两路 ws（推理 / 纯画面）都得带上界面状态 ✗")
    check('"screen": self._screen_state' in lt,
          "状态行载荷没带 screen ✗")
    for pin in ('screen_state.check_frame', 'behavior.event("screen_state"',
                "screen_state.capture_frame", "ui_state.detect(raw)"):
        check(pin in lt, "live_thread 缺接线：%s ✗" % pin)
    # ⭐ 报警音在 **live_panel**（GUI 线程 ✓ QMediaPlayer 不能进收流线程 ✗）
    lp2 = (ROOT / "gui" / "widgets.py").read_text(encoding="utf-8")
    check("def play_sound(" in lp2, "widgets 缺 play_sound（报警音一处实现 ✗）")
    lp = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("play_sound(getattr(settings" in lp,
          "live_panel 没接报警音（combat → 非combat 过渡 ✗）")
    check("【测谎弹窗" in lp, "状态行没有测谎报警显示 ✗")
    mw = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
    for pin in ('"挂机保护"', "build_protection_page"):
        check(pin in mw, "main_window 缺「挂机保护」页接线：%s ✗" % pin)
    # ⭐ 决策状态搬进信息栏（2026-09-29）：route_panel 有行、live_thread 左上角白字删除 ✓
    rp = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    check('"决策：%s"' in rp, "route_panel 缺决策状态行（信息栏 ✗）")
    check("决策:%s" not in lt, "左上角那行 cv2 白字没删（已搬进信息栏 ✗）")
    check("def stop_sound(" in lp2 and "def sound_player(" in lp2,
          "widgets 缺播放器状态化（试听 ⇄ 停止 ✗）")
    pp_src = (ROOT / "gui" / "player_panel.py").read_text(encoding="utf-8")
    check("_toggle_preview" in pp_src and "stateChanged.connect" in pp_src,
          "试听 ⇄ 停止的切换没接（播完要弹回「试听」✗）")
    ag = (ROOT / "decision" / "agent.py").read_text(encoding="utf-8")
    for pin in ("界面接管", "screen_takeover", "screen_resume"):
        check(pin in ag, "agent 缺接线：%s ✗" % pin)


def t_screen_beat_bridge():
    """桥接钉子（**行为级**）：测谎模板没命中 ⇒ 问 ui_state；断线界面 ⇒ 状态切换
    + 打点 + 报警 + 存帧。（源码钉子挡不住 `and False` 这种死接线 ✗ —— 就栽过 ✓）"""
    import types
    import tempfile as _tf

    from core import behavior

    from gui import live_thread as _lt
    from perception import ui_state

    logf = Path(_tf.mkdtemp(prefix="lie_bridge_")) / "behavior.log"
    old_log, old_en = behavior.LOG, behavior.ENABLED
    th = types.SimpleNamespace(_screen_state="combat", _screen_last=0.0)
    frame = np.full((768, 1366, 3), 60, np.uint8)
    caps = []
    try:
        behavior.configure(True, log=logf)
        with mock.patch.object(ui_state, "detect", lambda f: "login_err"), \
             mock.patch.object(screen_state, "capture_frame",
                               lambda f, st: caps.append(st)):
            _lt.LiveThread._screen_beat(th, frame)
        check(th._screen_state == "login_err",
              "ui_state 判出断线界面却没进状态 ✗：%r" % th._screen_state)
        check(caps == ["login_err"], "进入断线界面没存帧 ✗：%r" % caps)
    finally:
        behavior.configure(old_en, log=old_log)
    txt = logf.read_text(encoding="utf-8") if logf.exists() else ""
    # ⚠ behavior.log 的字段按 key 排序（st/prev 顺序不保证）⇒ 逐个判 ✓
    check("screen_state" in txt and "st=login_err" in txt and "prev=combat" in txt,
          "screen_state 事件没落 behavior.log ✗：%r" % txt[:200])


def t_protection_page():
    """「挂机保护」页（行为级）：防掉线组 **re-parent** 过来 + 「防挂机」可配 ✓。"""
    from PyQt5.QtWidgets import QApplication

    from gui.player_panel import PlayerPanel

    from decision.agent import settings
    from tools.selftest_decision import _kill_qt

    _app = QApplication.instance() or QApplication([])   # ⚠ 不建它 = 裸建 QWidget ⇒ 原生崩 ✓

    p = PlayerPanel()
    old_sound = getattr(settings, "lie_alarm_sound", "")
    try:
        page = p.build_protection_page()
        titles = [g.title() for g in page.findChildren(type(p.afk))]
        check("防掉线" in titles, "防掉线组没搬进挂机保护页 ✗：%r" % titles)
        check("防挂机" in titles, "「防挂机」新组不在页里 ✗：%r" % titles)
        # ⭐ re-parent 而不是重建：决策参数页那边**不再有**它（同一份控件 ✓）
        check(p.afk.parent() is not p,
              "防掉线组还在决策参数页的布局里（没真搬 ✗）：%r" % p.afk.parent())
        # ⚠ 别用 findChild(QLineEdit)：面板里第一个 QLineEdit 不是它（实测拿到 '1.00' ✗）
        check(p.ed_alarm_sound.text().strip() != "",
              "触发音效没回填默认值 ✗：%r" % p.ed_alarm_sound.text())
        check(p.ed_alarm_sound.parentWidget() is not p,
              "防挂机组没真搬进挂机保护页 ✗")
        # 配置回写（**不许写用户的 project.yaml** ✓ patch 掉 save ✓）
        with mock.patch.object(settings, "save", lambda: None):
            p.ed_alarm_sound.setText("datasets/sound/x.mp3")
            p._on_alarm_sound()
            check(settings.lie_alarm_sound == "datasets/sound/x.mp3",
                  "音效路径没写回 settings ✗：%r" % settings.lie_alarm_sound)
    finally:
        settings.lie_alarm_sound = old_sound
        _kill_qt(p)
        _kill_qt(page)


TESTS = [
    ("三张弹窗模板贴帧 ⇒ 判对；空画布 ⇒ combat", t_detect_synthetic),
    ("阈值反向：拉满必不命中", t_threshold_reverse),
    ("模板缺失 ⇒ combat + why（绝不猜）", t_missing_templates),
    ("agent 接管：停手 / 任务保留 + interrupted / 恢复 / 事件落盘", t_agent_takeover),
    ("断线桥接（行为级）：ui_state 判出 login_err ⇒ 状态切换 + 报警 + 存帧 + 打点",
     t_screen_beat_bridge),
    ("挂机保护页：防掉线组 re-parent + 防挂机组可配", t_protection_page),
    ("接线钉子（live_thread / live_panel / agent）", t_wiring_source_pins),
]


def main():
    npass = 0
    fails = []
    for desc, fn in TESTS:
        try:
            fn()
            npass += 1
            print("  [OK] %s" % desc)
        except Exception as e:                 # noqa: BLE001 —— 用例失败要显示不中断 ✓
            fails.append((desc, e))
            print("  [NG] %s\n         %s" % (desc, e))
    print("界面状态自检：%d/%d 通过%s" % (npass, len(TESTS),
                                          "，%d 条失败" % len(fails) if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
