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


def t_scale_invariant_by_normalizing():
    """⭐⭐⭐ **换分辨率也要认得出**（用户 2026-10-10 ✓ 原话："为什么测谎开始后 Agent 还在操作
    角色按方向键？" ✓）。

    病（真帧实测 ✓ 见开发日志 286 ✓）：三块模板是从 **1366×768** 那一路流裁的 ✓，而
    `cv2.matchTemplate` 只对**平移**宽容、对**缩放**不宽容 ✗ ⇒ 本机「本地窗口」那块是
    **1920×1080** ⇒ `lie_warn` 从 **0.857** 掉到 **0.659** ✗ ⇒ 判成 `combat` ⇒
    `_takeover_st` 不动 ⇒ **Agent 照旧按方向键** ✓✓（用户两轮看到的就是这个 ✓）。
    ⇒ 匹配前**先把帧归一到 `BASE_W/BASE_H`** ✓（`_prepare_gray` ✓）。

    钉三件：
      ① **基准尺寸**照旧判得出（老行为一字不变 ✓）；
      ② **放大 / 缩小**到 1920×1080 / 1600×900 / 1024×576 / 2560×1440 ⇒ 仍然判得出 ✓
         （⚠ 这条就是"用户那台机器"的情形 ✓ 少一个尺寸就等于没修 ✓）；
      ③ 反过来：**空画布**在任何尺寸下都不许判成 lie（归一化不许把误报带回来 ✗）。
    """
    for w, h in ((1366, 768), (1920, 1080), (1600, 900), (1024, 576), (2560, 1440)):
        f = _frame_with("lie_warn")
        if (w, h) != (1366, 768):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        st, det = screen_state.check_frame(f)
        check(st == "lie_warn",
              "%dx%d 上判成了 %s ⇒ 换分辨率就认不出测谎 ⇒ Agent 在测谎里继续按方向键 ✗：%r"
              % (w, h, st, det))
        blank = np.full((h, w, 3), 60, np.uint8)
        st2, det2 = screen_state.check_frame(blank)
        check(st2 == "combat",
              "%dx%d 的空画布判成了 %s（归一化把误报带回来了 ✗）：%r" % (w, h, st2, det2))


def t_ui_variants_score_best():
    """⭐⭐⭐ 一块锚点可以挂**多份模板**（变体），判分取最高 —— 治「本地窗口那版认不出登录界面」。

    病（用户 2026-10-10 现场 ✓ 原话："**我现在就在本地实时的登陆界面，鳄鱼潭1项目。没有任何
    警报**"）：模板是照**旧客户端**裁的 ✓，而用户这台**本地窗口**那一版渲染不一样 ✗ ⇒
    「连接」按钮只 0.781（< 0.80 ✗）、底部健康忠告那条只 0.51（新版换行带标点 ✗）
    ⇒ `login` 判不出 ⇒ **没警报** ✓。按真帧重裁后 0.977/0.932 ✓ —— 但同一套重裁模板拿去判
    **收流那版**（1366×768）又判不出 ✗（两边不是同一版渲染 ✓ ⇒ **一套服务不了两边** ✗）。

    ⇒ 改法：`ui_state` 支持 `<name>.png`（主）+ `<name>.<n>.png`（变体 ✓），
      **判分取最高**（`score_best` ✓）、门限照旧 0.80 ✓（不放松标准 ✓）。

    钉四件：
      ① 变体路径命名（`template_path(name, 1)` ⇒ `<name>.1.png` ✓）；
      ② `login` 的两块锚点**真的有变体**（本地窗口那版 ✓ 否则这条就白写了 ✗）；
      ③ **行为级**：把**变体**贴进一张 1920×1080 画布的锚点位置 ⇒ `detect` 必须判出 `login` ✓
         （证明变体**真的被用上了** ✓ 不是摆着看 ✓）；
      ④ 空画布 ⇒ `None` ✓（加了变体不许把误报带来 ✗）。
    """
    import shutil
    import tempfile

    from core.imgio import imread

    from perception import ui_state

    _old_dir = ui_state.TEMPLATE_DIR
    _old_anchors = dict(ui_state.ANCHORS)

    check(ui_state.template_path("x", 1).name == "x.1.png",
          "变体命名不是 `<name>.<n>.png`（那 `load_variants` 就找不到它 ✗）")

    has = [n for n, _r, _t, _d in ui_state.ANCHORS["login"]
           if ui_state.template_path(n, 1).exists()]
    check(len(has) == 2,
          "`login` 的两块锚点都得有「本地窗口那版」变体（现在只有 %r ✗ —— "
          "少了它这台机器上登录界面就判不出、警报不响 ✓）" % (has,))

    # ③ 行为级：把**变体**贴到锚点位置 ⇒ 必须判出 login ✓
    base = np.full((ui_state.BASE_H, ui_state.BASE_W, 3), 60, np.uint8)
    for name, rect, _ts, _desc in ui_state.ANCHORS["login"]:
        t = imread(str(ui_state.template_path(name, 1)))
        h, w = t.shape[:2]
        x, y = int(rect[0]), int(rect[1])
        base[y:y + h, x:x + w] = t
    ui_state.clear_cache()
    try:
        check(ui_state.detect(base) == ui_state.UI_LOGIN,
              "把**变体**贴到锚点位置却判不出 login ✗（那变体等于没接上 ✓）：%r"
              % (ui_state.scores(base)[0],))
        # ④ 空画布不许误判 ✗
        check(ui_state.detect(np.full((ui_state.BASE_H, ui_state.BASE_W, 3), 60,
                                      np.uint8)) is None,
              "空画布被判成了某个界面（变体把误报带回来了 ✗）")

        # ②′ 只留主模板（删掉变体）⇒ 那一版就**应该**判不出 ✓（反向验证：证明确实是变体在起作用）
        _d = Path(tempfile.mkdtemp(prefix="uitpl_"))
        try:
            for nm in [n for n, _r, _t, _dsc in ui_state.ANCHORS["login"]]:
                shutil.copy2(str(ui_state.template_path(nm)), str(_d / (nm + ".png")))
            for ui, anchors in ui_state.ANCHORS.items():        # 别的锚点照旧（否则缺模板跳过 ✓）
                for nm, _r, _t, _dsc in anchors:
                    p = ui_state.template_path(nm)
                    if p.exists() and not (_d / (nm + ".png")).exists():
                        shutil.copy2(str(p), str(_d / (nm + ".png")))
            ui_state.TEMPLATE_DIR = _d
            ui_state.clear_cache()
            check(ui_state.detect(base) is None,
                  "**删掉变体后**居然还判得出 login ✗ ⇒ 说明起作用的不是变体"
                  "（那这条用例等于没验到东西 ✓ 要重看 ✓）")
        finally:
            ui_state.TEMPLATE_DIR = _old_dir
            ui_state.clear_cache()
            shutil.rmtree(str(_d), ignore_errors=True)
    finally:
        ui_state.TEMPLATE_DIR = _old_dir
        ui_state.ANCHORS.clear()
        ui_state.ANCHORS.update(_old_anchors)
        ui_state.clear_cache()


def t_login_banner_is_optional():
    """⭐⭐⭐ 登录界面底部的**健康忠告横幅**只是"可选锚点"，不参与"必须全过"（用户 2026-10-10 ✓）。

    病（实测 ✓）：那条横幅（`login_agree` ✓）**同一台机器上两帧之间就 0.93 → 0.61** ✗
    （旧版单行无标点 / 新版两行带标点 + 多一行「我已详细阅读并同意《隐私政策》…」✓）
    ⇒ 拿它当"必须过"的门 = 登录识别**看运气** ✗（21:19:55 恰好判出了 `login` ✓、
    紧接着另一帧又判不出 ✗ —— 用户看到的就是"时有时无"✓）。

    钉三件：
      ① `login_agree` 在 `OPTIONAL_ANCHORS` 里 ✓（口径写在一处 ✓）；
      ② **行为级**：画布上**只贴「连接」按钮**（登录界面独有的那块 ✓）⇒ 必须判出 `login` ✓
         （横幅缺失/变了都不该挡住 ✓）；
      ③ 反向：把「连接」按钮也去掉（空画布）⇒ **必须判不出** ✗（可选锚点不等于"什么都放行"✓）。
    """
    from core.imgio import imread

    from perception import ui_state

    check("login_agree" in ui_state.OPTIONAL_ANCHORS,
          "`login_agree`（底部通用健康忠告横幅）该是**可选**锚点 —— 它内容会变、别处也有 ✗ "
          "（不放可选 ⇒ 登录识别时灵时不灵 ✓ 用户报过 ✓）")

    only = np.full((ui_state.BASE_H, ui_state.BASE_W, 3), 60, np.uint8)
    for name, rect, _ts, _desc in ui_state.ANCHORS["login"]:
        if name in ui_state.OPTIONAL_ANCHORS:
            continue                                  # ② **故意不贴横幅** ✓
        t = imread(str(ui_state.template_path(name, 1)))
        h, w = t.shape[:2]
        only[int(rect[1]):int(rect[1]) + h, int(rect[0]):int(rect[0]) + w] = t
    ui_state.clear_cache()
    check(ui_state.detect(only) == ui_state.UI_LOGIN,
          "只贴「连接」按钮（登录界面独有的那块）却判不出 `login` ✗ —— "
          "横幅那条可选锚点没生效 / 或者又把必须过的门加回去了 ✗：%r"
          % (ui_state.scores(only)[0],))

    # ③ 空画布（连按钮都没有）⇒ 必须判不出 ✓
    check(ui_state.detect(np.full((ui_state.BASE_H, ui_state.BASE_W, 3), 60,
                                  np.uint8)) is None,
          "空画布也判成 `login` ✗ ⇒ 可选锚点被当成「什么都放行」了 ✓（那就不是判据了 ✗）")


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
    # ⭐⭐⭐⭐⭐ **两路 ws 都要带"接管版"界面状态**（用户 2026-10-10 ✓ 本行 2026-10-10 改口径 ✓）：
    #   原来钉的是 `ws.screen = self._screen_state`（**原生识别结果** ✓）——那条被现场否掉 ✗：
    #   测谎**小游戏那段画面匹配不上任何模板**（`live_thread` 自己的实测写着"是常态"✓）
    #   ⇒ 原生结果回落 `combat` ⇒ Agent 当场把方向盘拿回去、**又开始按方向键** ✗✗
    #   （用户原话："**为什么测谎开始后 Agent 还在操作角色按方向键？**"✓）。
    #   ⇒ 现在两路都报 `_takeover_st`（**留住接管**后的那份 ✓ 见 `_screen_beat` 末段 ✓）；
    #     ⚠ **原生 `_screen_state` 照旧单独给录屏/样本/状态行**（下面那条 `"screen":` 钉着 ✓）
    #       —— 两件事**故意分开** ✗ 别合并 ✓。
    check(lt.count('ws.screen = (getattr(self, "_takeover_st", None)') >= 2,
          "两路 ws（推理 / 纯画面）都得带上**接管版**界面状态（`_takeover_st`）✗ —— "
          "用原生 `_screen_state` 就是「测谎中小游戏画面匹配不上 ⇒ 回落 combat ⇒ "
          "Agent 恢复按键」那个老毛病 ✗")
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
    # ⚠ 这个替身是**绕过 `__init__`** 造的（只摆被测路径要用的字段 ✓）—— 而
    #   `_screen_beat` 里后来多了**断线录屏**那条分支（2026-10-02 加 ✓：判到
    #   `ui_state.UI_LOGIN_ERR` ⇒ `_lie_rec_start("disc")`）⇒ 替身得给它个空实现 ✓。
    #   本用例管的是"状态切换 + 存帧 + 打点"（✓ 那三件照旧真跑）；**录屏是另一件
    #   单独测的事** ✗ ⇒ 这里空实现是对的，不是把断言放水 ✓。
    th._lie_rec_start = lambda kind="lie": False
    th._lie_rec_stop = lambda why="": None
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


def t_lie_game_threshold_widened():
    """⭐⭐⭐⭐ **`lie_game` 单独放宽门限**（用户 2026-10-10 第二轮 ✓ 原话："**测谎的时候 Agent
    还在操作移动、跳跃**"✗）。

    **实测**（真录像 `data/recordings/lie_20261010_015650.mp4` ✓ `check_frame(fr, threshold=0)`
    拿"最像哪块 + 多少分"✓ 见开发日志 **271** ✓）：
      · `lie_warn` **0.867~0.869** ✓（干净 ✓）；**`lie_game` 0.783~0.785** ✗ —— **低于 0.80**
        ⇒ 小游戏那 18 秒**根本认不出来** ⇒ 状态回落 `combat` ⇒ **Agent 恢复按键** ✗✗
        （同一场：`lie_rec` 录满 300 秒上限、而 `perf.log` 每段 `agent_why=-` ✓ 对得上 ✓）。
      · ⚠⚠ 反面：普通战斗帧对这块模板能到 **0.805~0.836** ✗（比真的还高 ✓ 10-08 那 4 次误判 ✓）
        ⇒ **只降门限挡不住假阳性** ⇒ 真正的判据是 `_screen_beat` 那条 **A 规则**（必须先见
        `lie_warn` ✓）⇒ **两条合起来**才完整 ✓（本用例只钉"放宽"这一半 ✓ 另一半由
        `t_lie_needs_warn` 钉着 ✓）。

    钉三件：① 这块的门限必须**低于基础门限**、且**低于实测真小游戏分**（0.783 ✓ 留出余量）；
    ② `check_frame` 的**显式** `threshold=` 语义不许被改（自检/调试用 ✓）；
    ③ 走默认门限时，那块模板确实吃自己的门限（源码级：`MATCH_THRESHOLD_BY` 被查过 ✓）。
    """
    base = float(screen_state.MATCH_THRESHOLD)
    by = dict(getattr(screen_state, "MATCH_THRESHOLD_BY", {}) or {})
    check("lie_game" in by, "`lie_game` 没有自己的门限 ⇒ 真小游戏 0.783 还是认不出来 ✗")
    thr = float(by["lie_game"])
    check(thr < base, "`lie_game` 的门限没放宽（%g 不低于基础 %g）⇒ 小游戏照旧认不出来 ✗"
          % (thr, base))
    check(thr <= 0.78 and thr >= 0.70,
          "`lie_game` 门限取 %g：要**低于实测真分 0.783**（否则还是认不出 ✗）又不该低到没边"
          "（滥用会淹掉 A 规则那道闸 ✓）" % thr)
    src = (ROOT / "perception" / "screen_state.py").read_text(encoding="utf-8")
    check("MATCH_THRESHOLD_BY.get(best_name, threshold)" in src,
          "`check_frame` 没查每模板门限（白设了 ✗）")
    # ② 显式 threshold 必须照旧是"字面值"（用例/调试靠它 ✓）
    import numpy as np
    fr = np.full((768, 1366, 3), 60, np.uint8)
    st, det = screen_state.check_frame(fr, threshold=0.0)   # 阈值 0 ⇒ 一定"命中" ✓
    check(st != "combat",
          "显式 `threshold=0.0` 时应当给出最像的那块（显式门限的语义被改了 ✗）：%r" % st)


def t_lie_needs_warn():
    """⭐⭐⭐⭐⭐ **A ＋ C：测谎状态的两道门** ✗✗（用户 2026-10-08 ✓ 两条都是他选的 ✓）。

      · **A** 原话："**加一条「必须先见过 lie_warn 才算测谎」—— 真测谎一定有 warn 在前**" ✓
        ⇒ `lie_game` / `lie_success` 只有"这一场开着"（`LIE_SESSION_MAX_S` = **60 秒**内见过
        `lie_warn` ✓）才认 ✓；否则**当 combat**（不起录、不报警 ✓）＋ 记一条 `lie_state_denied` ✓；
      · **C** 原话："**进 lie_* 后若一直没见到 lie_warn/lie_success、且 lie_game 分数贴着门槛 ⇒
        到点就停，别录满 5 分钟**" ✓ ⇒ **没见到成功弹窗**、且这一场 `lie_game`/`lie_success` 的
        **最高分 < 门限＋0.05 = 0.85** ⇒ 录到 **90 秒**收工 ✓（老逻辑 5 分钟 ✗）；
        ⚠ **每拍都评** ✗（不是在"状态变了"那一拍才评 ✓ —— 画面卡住时那样永远评不到 ✓ 见源码注释 ✓）。

    实测依据（`10月7日.mp4` ＋ `behavior.log` ✓）：今天那 4 次误判全是 `combat → lie_game`
      （**0.805~0.836** ✓）**没有一次 `lie_warn`** ⇒ A 当场全灭 ✓；真 `lie_warn`/`lie_success` =
      **0.857 / 0.879** ⇒ C 那条 0.85 两边都留了余量 ✓。
    """
    import types

    from core import behavior

    from gui import live_thread as _lt
    from perception import ui_state

    _NOW = [1000.0]
    frame = np.full((768, 1366, 3), 60, np.uint8)

    def _mk():
        th = types.SimpleNamespace(
            _screen_state="combat", _screen_last=0.0, _lie_rec=None, _lie_rec_pending=0.0,
            _lie_rec_stop_at=0.0, _lie_rec_saw_success=False, _lie_rec_last_lie=0.0,
            _rec_kind="", _lie_warn_at=0.0, _lie_sess_t0=0.0, _lie_rec_top_game=0.0)
        calls = {"start": [], "stop": []}

        def _start(kind="lie"):
            calls["start"].append(kind)
            if kind == "lie":                       # 真起录 ⇒ 置上"在录"那两个字段 ✓（C 靠它判 ✓）
                th._rec_kind, th._lie_rec_pending = "lie", 1.0
            return True

        def _stop(why=""):
            calls["stop"].append(why)
            th._rec_kind, th._lie_rec_pending = "", 0.0
        th._lie_rec_start, th._lie_rec_stop = _start, _stop
        #   ⚠ 这个替身**绕过 `__init__`** ⇒ 类属性也得搬过来 ✗（`_screen_beat` 里会读它们 ✓：
        #     `LIE_REC_GAP_S` / `LIE_REC_HOLD_S` ✓ —— 不搬 ⇒ 走到那条兜底时 `AttributeError` ✗，
        #     实测踩到 ✓）。
        for _k in ("LIE_REC_HOLD_S", "LIE_REC_GAP_S"):
            setattr(th, _k, getattr(_lt.LiveThread, _k))
        return th, calls

    def _beat(th, st, score):
        with mock.patch.object(screen_state, "check_frame",
                               lambda f: (st, {"score": score})), \
             mock.patch.object(ui_state, "detect", lambda f: None), \
             mock.patch.object(screen_state, "capture_frame", lambda f, s: None), \
             mock.patch.object(_lt.time, "monotonic", lambda: _NOW[0]):
            _lt.LiveThread._screen_beat(th, frame)

    logf = Path(tempfile.mkdtemp(prefix="lie_warn_gate_")) / "behavior.log"
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        behavior.configure(True, log=logf)
        th, calls = _mk()
        # ① **A 反向**：像今天那 4 次一样 —— `combat` 直接跳 `lie_game`（0.83 ✓）⇒ 不认 ✗
        _NOW[0] = 1000.0
        _beat(th, "lie_game", 0.83)
        check(th._screen_state == "combat" and not calls["start"],
              "① **没 warn 打头的 lie_game ⇒ 不认**（实测状态 %r ／ 起录 %d 次〔该 0 ✓〕）"
              % (th._screen_state, len(calls["start"])))
        # ② **A 正向**：`lie_warn` ⇒ 进状态 ＋ 起录
        _NOW[0] = 1002.0
        _beat(th, "lie_warn", 0.86)
        check(th._screen_state == "lie_warn" and calls["start"] == ["lie"]
              and th._rec_kind == "lie",
              "② **lie_warn ⇒ 进状态 ＋ 起录**（实测 %r ／ start %r ✓）"
              % (th._screen_state, calls["start"]))
        # ③ 同一场里的 `lie_game` ⇒ 认（**不重开录屏** ✓）
        _NOW[0] = 1006.0
        _beat(th, "lie_game", 0.81)
        check(th._screen_state == "lie_game" and calls["start"] == ["lie"],
              "③ **同一场里的 lie_game ⇒ 认**（实测 %r ／ 起录还是 %d 次 ✓〔该 1 ✓〕）"
              % (th._screen_state, len(calls["start"])))
        # ④ **C**：离开 `lie_*` 不急着停；到 90 秒才收 ＋ 关掉这一场
        _NOW[0] = 1010.0
        _beat(th, "combat", 0.55)
        check(not calls["stop"],
              "④ **刚离开 lie_* 不急着停**（还没见到成功弹窗 ⇒ 照旧一路录 ✓）")
        _NOW[0] = 1002.0 + _lt.LIE_REC_SOFT_S + 1.0
        _beat(th, "combat", 0.55)
        check(len(calls["stop"]) == 1 and "贴着门槛" in calls["stop"][0]
              and not th._lie_warn_at,
              "④ **C：到 %g 秒收工 ＋ 把这一场关掉**（实测 stop=%r ／ `_lie_warn_at`=%r ✓）"
              % (_lt.LIE_REC_SOFT_S, calls["stop"], th._lie_warn_at))
        # ⑤ 关场之后再来的 `lie_game` ⇒ 又被 A 拦（不来回切 ✓）
        _NOW[0] += 5.0
        _beat(th, "lie_game", 0.82)
        check(th._screen_state == "combat" and len(calls["start"]) == 1,
              "⑤ **关场之后再来的 lie_game 不认**（实测 %r ✓ ／ 起录仍 %d 次 ✓）"
              % (th._screen_state, len(calls["start"])))
        # ⑥ **反例**：这场分数够高（0.88 ≥ 0.85 ✓）⇒ C **不许**收（收了就是把真场面掐掉 ✗）
        #   ⚠⚠ **沿途必须喂一拍 `lie_game`** ✗（2026-10-10 修 ✗ 这条原来写漏了 ✓）：
        #      C 的门槛（`LIE_REC_SOFT_S` = 90 秒 ✓）**比 GAP 兜底（`LIE_REC_GAP_S` = 60 秒 ✓）
        #      长** ⇒ 中间一次 `lie_*` 都不喂的话，**先触发的是 GAP** ✗ —— 那是**另一条规则**
        #      （"整场失败 / 界面卡住 ⇒ 不能无限录" ✓ 见它自己的用例 ⑧ ✓），不是 C ✗。
        #      原来这条一口气跳到 91 秒 ⇒ 拿到的 `stop` 是 GAP 的 ✓ ⇒ 期望写错了（实测 stop=
        #      `['长时间没再见到测谎（60 秒）⇒ 收工']` ✗）。
        #      ⚠ 而且**只喂 `lie_*` 不算数** ✗：`_lie_rec_last_lie` 只在**状态变化**那一拍更新
        #        （它排在 `_screen_beat` 的早退**之后** ✓）⇒ 中间那拍必须是一次**真变化**
        #        （`combat → lie_game` ✓ 现场小游戏期间"偶尔匹得上几拍"就是这个样子 ✓）。
        th2, calls2 = _mk()
        _NOW[0] = 2000.0
        _beat(th2, "lie_warn", 0.86)
        _NOW[0] = 2002.0
        _beat(th2, "lie_game", 0.88)
        _NOW[0] = 2004.0
        _beat(th2, "combat", 0.5)                # 中间几拍匹不上（**常态** ✓）
        _NOW[0] = 2000.0 + _lt.LIE_REC_SOFT_S / 2.0
        _beat(th2, "lie_game", 0.88)             # ← 又匹上了（把 GAP 按住 ✓ 分数依旧够高 ✓）
        _NOW[0] = 2000.0 + _lt.LIE_REC_SOFT_S + 1.0
        _beat(th2, "combat", 0.5)
        check(not calls2["stop"],
              "⑥ **分数够高（0.88 ≥ %.2f）⇒ C 不收工**（实测 stop=%r ✓ —— ⚠ 收了就是掐真场面 ✗）"
              % (_lt.LIE_REC_STRONG, calls2["stop"]))
        # ⑦ `lie_warn` 早过 60 秒 ⇒ 后面的 `lie_game` 照样不认
        th3, calls3 = _mk()
        _NOW[0] = 3000.0
        _beat(th3, "lie_warn", 0.86)
        _NOW[0] = 3000.0 + _lt.LIE_SESSION_MAX_S + 1.0
        _beat(th3, "combat", 0.5)
        _beat(th3, "lie_game", 0.9)
        check(th3._screen_state == "combat",
              "⑦ **warn 早过 %g 秒 ⇒ lie_game 不认**（实测 %r ✓）"
              % (_lt.LIE_SESSION_MAX_S, th3._screen_state))
        # ⑧ ⭐ **GAP 兜底**（既有规则 ✓ **本轮才补上它的用例** ✗）：整场失败 / 界面卡住 ⇒
        #    `LIE_REC_GAP_S` 秒没再见到**任何** `lie_*` 就收工 ✓（不许无限录 ✗）。
        #    ⚠ 它必须**远大于**中间小游戏那一截（现场实测 **18 秒**没有任何 `lie_*` ✓
        #      这就是它取 60 的原因 ✓）—— 与 C（90 秒、看**分数贴不贴门槛** ✓）是**两条**规则 ✓：
        #      这条只看"还见不见得到 `lie_*`" ✓。
        th4, calls4 = _mk()
        _NOW[0] = 4000.0
        _beat(th4, "lie_warn", 0.86)
        _NOW[0] = 4002.0
        _beat(th4, "combat", 0.55)
        check(not calls4["stop"],
              "⑧ 刚离开 `lie_*` 不该停（宽限还没到 ✓）")
        _gap = float(_lt.LiveThread.LIE_REC_GAP_S)     # ⚠ **类属性** ✗（不是模块级 ✓）
        _NOW[0] = 4002.0 + _gap + 1.0
        _beat(th4, "combat", 0.55)
        check(len(calls4["stop"]) == 1 and "长时间" in calls4["stop"][0],
              "⑧ **`LIE_REC_GAP_S` = %g 秒没再见到任何 `lie_*` ⇒ 收工**"
              "（实测 stop=%r ✗ —— 不收就会无限录 ✗）"
              % (_gap, calls4["stop"]))
    finally:
        behavior.configure(old_en, log=old_log)
    txt = logf.read_text(encoding="utf-8") if logf.exists() else ""
    check("lie_state_denied" in txt,
          "**被拦下来的那几拍如实打点**（`lie_state_denied` 落 behavior.log ✓ 事后能核 ✓）：%r"
          % txt[-200:])


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
    ("测谎两道门（A 必须先见 warn ＋ C 贴门槛 90 秒收工）", t_lie_needs_warn),
    ("⭐⭐⭐⭐ 「`lie_game` 单独放宽门限」（用户 2026-10-10）：真录像实测 0.783~0.785 低于基础"
     "门限 ⇒ 小游戏认不出来 ⇒ Agent 在测谎里还在按移动/跳跃；放宽 + 保留 A 规则才是完整口径",
     t_lie_game_threshold_widened),
    ("⭐⭐⭐ 换分辨率也要认得出（归一化到模板基准尺寸）——治「测谎里 Agent 还在按方向键」",
     t_scale_invariant_by_normalizing),
    ("⭐⭐⭐ `ui_state` 一块锚点可挂多份模板（变体取最高分）——治「本地窗口那版认不出登录界面」",
     t_ui_variants_score_best),
    ("⭐⭐⭐ 登录底部那条通用健康横幅只是**可选**锚点（治「登录识别时灵时不灵」）",
     t_login_banner_is_optional),
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
