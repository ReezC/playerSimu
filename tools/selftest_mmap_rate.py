"""自检：**小地图「尽可能高帧率更新」这条链**（2026-10-03 新增 ✓）。

用户的需求（原话）："**小地图作为权威世界坐标，应该尽可能用较高的帧率去更新**" ✓
（补答："A机 fps 已经 60 了，quality 允许降；游戏（小地图）**应该有 200+** [刷新率]"✓
⇒ 游戏不是上限，两个卡点都在我们自己这边 ✓）。

查出来的两个卡点（都有实测 ✓）：
  ① **B 机把定位绑在"推理拍"上**（`gui/live_thread.py` 主回路每帧调一次 ✓）——
     而实测 `locate_ms` 中位只有 **~0.5ms**（`infer_ms` 才是 ~20ms 的大头 ✓）⇒
     世界坐标只能 30 次/秒更新，A 机推 60fps 有一半**被丢掉** ✗（`MiniMapClient`
     只留最新帧 ✓ `n_drop` 就是它 ✓）。⇒ 现在起一条**高频定位回路**：每收到一帧就定位
     一次（60fps 只占约 3% CPU ✓），主回路每帧**取最新一份** ✓。
  ② **A 机推流默认 30fps / quality 100**，而且发帧用**阻塞 `sendall`** ✗（哪一路慢就把
     整个帧循环拖住 ⇒ 所有客户端一起变慢 ✗）。⇒ 默认提到 **60 / 80**（quality 降一半体积
     ≈ 省一半编码时间 ✓ 正好抵掉翻倍的开销 ✓），并且**某一路跟不上就跳它的帧** ✓
     （跳帧在这里**无损**：B 机本来就只留最新帧 ✓）。

⚠ 本用例只钉"接线与口径"（真实测帧率要 A 机 + 游戏，摆不动 ✗）：
   · 源码级钉：回路存在、只由收流来源起、主回路取槽、`_gate_closed` 语义没被改、
     `locate_ms`/`mmap_ok` 只记一处（不然 perf 里的数会骗人 ✗）；
   · 行为级钉：`_loc_slot_take` 的**新鲜度**与**丢帧增量**算得对（替身对象 ✓）；
   · 部署侧钉：默认值 60/80 在 **DEFAULTS / 卡片 / 命令行 fallback** 三处一致 ✓。

跑法：
    python -m tools.selftest_mmap_rate
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK：用例名里带 ⇒ / ✓ 这类字符时 `print` 会抛 UnicodeEncodeError ✓
try:
    sys.stdout.reconfigure(encoding="utf-8")            # type: ignore[attr-defined]
except Exception:                                       # noqa: BLE001
    pass


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _src(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def t_high_rate_loop_is_wired():
    """① 源码级：B 机那条**高频定位回路**接线对（含"只由收流来源起"和"只一处记账"）。"""
    lt = _src("gui/live_thread.py")
    check("def _mmap_loc_loop(self)" in lt,
          "没有那条第高频定位回路（世界坐标还是绑在推理拍上 ✗）")
    check("def _loc_slot_take(self)" in lt and "def _locate_latest(self)" in lt,
          "回路没有「放结果/取结果」的那两个口 ✗")
    check("threading.Thread(\n                            target=self._mmap_loc_loop"
          in lt.replace("\r\n", "\n"),
          "那条回路没有被真正起起来（写了不用 = 没改 ✗）")
    # 只在**收流**来源起：live 来源（从实时画面裁小地图）没有独立帧源 ✓
    check('if (self._mmap_src != "live" and self._mmap_mid' in lt,
          "回路的启动条件没排除 `live` 来源（那种来源没有独立帧源 ⇒ 会跟主回路抢 locator ✗）")
    # `PlayerLocator` 有跨帧状态 ⇒ 同一时刻只能一个线程调它 ✗
    check('if self._mmap_src == "live" or not self._mmap_mid:' in lt,
          "回路里没判「这条来源归不归我」（`live` 时它该歇着 ✓）")
    # 主回路：gate 时不取用（老行为）✓；live 或没回路 ⇒ 现算 ✓；否则取槽 ✓
    check("if _gate_closed:\n                        _loc = None" in lt.replace("\r\n", "\n"),
          "`_gate_closed`（框面积闸）被顺手改掉了 ✗ —— 那条闸的语义是"
          "「这一拍检测无效」，不是「小地图不可信」，本轮**不该动它** ✓")
    check("_loc = self._loc_slot_take()" in lt,
          "主回路没有改成「取回路给的最新一份」（那还是每拍现算 ⇒ 白改 ✗）")
    check('_loc = self._locate_latest()      # 没有独立帧源 ⇒ 照旧每拍现算 ✓' in lt,
          "`live` 来源的兜底（照旧每拍现算）没了 ⇒ 那种来源会拿不到坐标 ✗")


def t_locate_accounting_is_single_entry():
    """② 源码级：`locate_ms` / `mmap_ok|miss` **只在一处**记账（否则 perf 里的数会骗人 ✗）。

    为什么要钉：这两条是"定位到底跑了几次、多快"的唯一证据 —— 如果主回路与回路**各记一次**，
    `locate_ms` 的样本数会翻倍、中位数会被"没算的 0ms"污染 ⇒ 以后没人信它 ✗。
    """
    lt = _src("gui/live_thread.py").replace("\r\n", "\n")
    check(lt.count('perf.ms("locate_ms"') == 1,
          "`locate_ms` 有 %d 处记账（应该只有 `_locate_latest` 一处 ✗）"
          % lt.count('perf.ms("locate_ms"'))
    check(lt.count('perf.count("mmap_ok" if loc.get("ok")') == 1,
          "`mmap_ok/miss` 记了不止一处（样本会翻倍 ✗）")


def t_loc_slot_take_reports_age_and_drop():
    """③ 行为级：`_loc_slot_take` 的**新鲜度**（`mmap_age_ms`）与**丢帧增量**（`mmap_drop`）算得对。

    这两个数就是"提高帧率有没有用"的**量化** ✓（没有它们，改完只能靠感觉 ✗）：
      · `mmap_age_ms` = 这份坐标背后的那一帧**是多久以前收到的**；
      · `mmap_drop` = 这段时间 `MiniMapClient` 又丢了几帧（>0 持续 ⇒ 推流快过消费 ✓）。
    用替身对象（`LiveThread.__new__`）摆字段 ⇒ 不起 Qt、不起线程 ✓。
    """
    from gui.live_thread import LiveThread
    from core import perf

    got = {}                                        # perf 指标名 → 值列表（`setdefault` 收 ✓）
    old_sample = perf.sample
    perf.sample = lambda name, value, **kw: got.setdefault(name, []).append(value)
    try:
        import threading as _th
        import time as _time
        t = LiveThread.__new__(LiveThread)          # 替身：只摆这段用到的字段 ✓
        t._loc_lock = _th.Lock()
        t._loc_age_ms = None
        t._loc_drop_last = None

        class _Cli:
            n_drop = 7

        t._mmap_cli = _Cli()
        # 槽里空 ⇒ 返回 None（"这一拍没定位" ⇒ 沿用上一拍 ✓ 老行为 ✓）
        t._loc_slot = None
        check(t._loc_slot_take() is None, "槽空时该返回 None（沿用上一拍 ✓），不该编一个坐标 ✗")

        # 放一份"0.05 秒前收到的帧" ⇒ age 应该 ≈ 50ms ✓
        _t0 = _time.perf_counter() - 0.05
        t._loc_slot = (_t0, {"ok": True, "world_x": 1.0, "world_y": 2.0})
        loc = t._loc_slot_take()
        check(loc is not None and loc.get("world_x") == 1.0,
              "取槽拿到的不是那份结论 ✗：%r" % (loc,))
        _ages = got.get("mmap_age_ms") or []
        check(_ages and 40.0 <= float(_ages[-1]) <= 200.0,
              "`mmap_age_ms` 算得不对（摆的是 50ms 前收到的帧）：%r" % (_ages,))

        # 第一次记 drop 只做基线（不该报一个"平台期"的巨值 ✗）；涨了才报**增量** ✓
        got.pop("mmap_drop", None)
        t._loc_slot_take()
        check(not got.get("mmap_drop"),
              "第一次记 `n_drop` 就报了数（那是历史累计值 ⇒ 假信号 ✗）：%r"
              % (got.get("mmap_drop"),))
        _Cli.n_drop = 10                            # 又丢了 3 帧
        t._loc_slot_take()
        check(got.get("mmap_drop") == [3.0],
              "丢帧要报**增量**（3），不是累计值：%r" % (got.get("mmap_drop"),))
    finally:
        perf.sample = old_sample


def t_push_backpressure_and_defaults():
    """④ A 机：默认 **60fps / quality 80** + **跟不上就跳帧**（不阻塞整条推流 ✗）。"""
    mp = _src("tools/minimap_push.py")
    check('"fps": 60' in mp and '"quality": 80' in mp,
          "推流默认还是 30fps / quality 100（用户已批「quality 允许降」、A 机已经 60 ✓）")
    check("_budget_ms = 500.0 / max(1, fps)" in mp,
          "没有「一帧预算」这个判据（跳帧背压就无从谈起 ✗）")
    check("if _last_ms > _budget_ms:" in mp and "_skip[name] = " in mp,
          "某一路跟不上时没有**跳过它这一帧** ✗（那就还是阻塞 `sendall` 拖慢所有客户端 ✗）")
    check("clients.append([c, name, 0, time.perf_counter(), 0.0])" in mp,
          "客户端记录里没留「上一帧耗时」那一格（跳帧判据没地方放 ✗）")


def t_deploy_defaults_agree():
    """⑤ 部署台三处默认值必须一致：`DEFAULTS` / 卡片 spec / 命令行 fallback（都是 60 / 80）✓。"""
    from deploy import config as dcfg
    from deploy import services
    check(int(dcfg.DEFAULTS["mmap"]["fps"]) == 60,
          "`deploy/config.py` 的 mmap.fps 不是 60：%r" % dcfg.DEFAULTS["mmap"]["fps"])
    check(int(dcfg.DEFAULTS["mmap"]["quality"]) == 80,
          "`deploy/config.py` 的 mmap.quality 不是 80：%r" % dcfg.DEFAULTS["mmap"]["quality"])
    cmd = " ".join(services.build_cmd("mmap", {}))
    check("--fps 60" in cmd and "--quality 80" in cmd,
          "命令行 fallback 不是 60/80（`config/deploy.json` 里没这两个键时用的就是它 ✗）：%s"
          % cmd)
    fps_spec = next(s for s in services.PARAMS["mmap"] if s["key"] == "fps")
    check("60" in fps_spec.get("choices", ()),
          "帧率下拉里没有 60 这一档（用户会把 A 机也放到 60 ✓）：%r" % (fps_spec.get("choices"),))


def t_route_panel_shows_link_rate():
    """⑥ 路线识别页：「小地图链路」那一行（用户 2026-10-03 ✓ 他问"我能在路线识别页签→地形图
    看到验证结果吗？"）—— 行为级 + 源码级都钉。

    口径：这一行回答「**快不快/新不新**」（收帧 / 丢帧 / 定位耗时 ✓），
    与上面那行「玩家世界坐标」（"**对不对**"✓ 几何/标定）分工不同 ✓。
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from types import SimpleNamespace
    from gui.route_panel import RoutePanel

    class _Lbl:
        def __init__(self):
            self.text = ""
            self.tip = ""

        def setText(self, t):                    # noqa: N802 —— 仿 QLabel 的接口
            self.text = str(t)

        def setToolTip(self, t):                 # noqa: N802
            self.tip = str(t)

    class _Cli:
        connected = True
        fps = 58.24
        n_drop = 12
        n_recv = 723

    # 行为级：替身对象 + **未绑定**方法 ⇒ 不起 Qt 也能测 ✓
    fake = SimpleNamespace(lbl_mmap_rate=_Lbl(), _mmap_cli=_Cli(), _loc_ms_last=0.46)
    RoutePanel._refresh_mmap_rate(fake, "stream")
    txt = fake.lbl_mmap_rate.text
    check("58.2" in txt, "没显示 A 机**真正推出来**的收帧率（`cli.fps`）：%r" % (txt,))
    #   ⚠ 占比要**钉数值**（12/735 = 1.6% ✓）—— 光看有没有 `%` 字符抓不住"占比算错/写死 0"
    #     （反向验证真踩了这个盲区 ✓）。
    check("12/735" in txt and "1.6%" in txt,
          "没显示丢帧与占比（`n_drop`/`n_recv` ⇒ 「消费侧够不够快」就看它 ✗）：%r" % (txt,))
    check("0.46" in txt, "没显示这一拍的定位耗时：%r" % (txt,))

    # 没连上 / 来源=实时画面 ⇒ 明说一句（别留上一份数骗人 ✗）
    fake2 = SimpleNamespace(lbl_mmap_rate=_Lbl(), _mmap_cli=None, _loc_ms_last=0.5)
    RoutePanel._refresh_mmap_rate(fake2, "stream")
    check("没连上" in fake2.lbl_mmap_rate.text,
          "链路没连上时该说清，不是留空/留旧数 ✗：%r" % (fake2.lbl_mmap_rate.text,))
    fake3 = SimpleNamespace(lbl_mmap_rate=_Lbl(), _mmap_cli=None, _loc_ms_last=0.5)
    RoutePanel._refresh_mmap_rate(fake3, "live")
    check("实时画面" in fake3.lbl_mmap_rate.text,
          "来源=实时画面时该说明「没有这条推流链」✗：%r" % (fake3.lbl_mmap_rate.text,))

    # 源码级：那一行**每拍都刷**（在早退之前 ✓ —— 连不上时正是它最该说话的时候 ✓）
    rp = _src("gui/route_panel.py")
    check("self._refresh_mmap_rate(src)" in rp and "self.lbl_mmap_rate" in rp,
          "路线识别页没有那行链路读数（用户要在那儿看验证结果 ✗）")
    _i_call = rp.find("self._refresh_mmap_rate(src)")
    _i_ret = rp.find("return _say(\"玩家世界坐标：还没收到小地图推流")
    check(0 < _i_call < _i_ret,
          "那行是在早退**之后**刷的 ⇒ 连不上/没框选时它永远是上一句话 ✗：call@%s ret@%s"
          % (_i_call, _i_ret))


def t_live_panel_shows_coord_age():
    """⑦ 实时面板状态行要显示**坐标新鲜度**（`mmap_age_ms` ✓ 用户要的 b）。

    为什么这个数最关键：它是"权威坐标有多新"的**唯一直接量化** ✓ —— 小地图帧率从 30 提到
    60 有没有用，看它（应该稳定在半帧~一帧 ≈ 8~17ms ✓）。
    """
    lt = _src("gui/live_thread.py")
    check('"mmap_age_ms": self._loc_age_ms' in lt,
          "stats 里没把坐标新鲜度发给面板（那界面上就看不到 ✗）")
    check("def _mmap_fps_now(self)" in lt and "self._mmap_fps_now()" in lt,
          "没有安全读收帧率的那一处（读数不许把实时线程带崩 ✗）")
    check('"mmap_drop": int(getattr(self._mmap_cli, "n_drop"' in lt,
          "stats 里没带累计丢帧（界面上就说不清「消费侧跟不跟得上」✗）")
    lp = _src("gui/live_panel.py")
    check("坐标 %s" in lp and "_age_txt" in lp,
          "状态行里没有「坐标 xx ms」那一格 ✗")
    #   ⚠ 钉子要钉到**赋值/守卫那一行**，不能只钉"这个词出现过" ✗ —— 反向验证真踩了：
    #     把 `_age_txt` 写死成 `—` 照样绿 ✗。
    check('_age_txt = ("%5.1f ms" % float(_age)) if isinstance(_age, (int, float))'
          in lp,
          "那一格显示了，但不是**实际值**（写死成 `—` 就白显示了 ✗）")
    check('if _age is not None or s.get("mmap_rate") is not None:' in lp,
          "tooltip 那一段的守卫被摘了（等于不显示明细 ✗）")
    check("小地图（世界坐标的来源）：" in lp,
          "tooltip 里没有小地图那一路的明细（收帧/丢帧放哪儿 ✗）")


TESTS = (
    ("B 机高频定位回路：起得来、只由收流来源起、主回路取槽、`_gate_closed` 语义没被蹭坏",
     t_high_rate_loop_is_wired),
    ("定位记账只有一处（`locate_ms` / `mmap_ok|miss`）—— 不然 perf 的数会骗人",
     t_locate_accounting_is_single_entry),
    ("新鲜度与丢帧增量算得对（`mmap_age_ms` / `mmap_drop`：提高帧率有没有用就看它）",
     t_loc_slot_take_reports_age_and_drop),
    ("A 机推流：默认 60fps / quality 80 + 跟不上就跳帧（不阻塞整条推流）",
     t_push_backpressure_and_defaults),
    ("部署台三处默认值一致（DEFAULTS / 卡片 spec / 命令行 fallback = 60 / 80）",
     t_deploy_defaults_agree),
    ("路线识别页那行「链路体检」：收帧 / 丢帧 / 定位耗时，且每拍都刷（早退之前）",
     t_route_panel_shows_link_rate),
    ("实时面板状态行显示坐标新鲜度（`mmap_age_ms`）+ tooltip 明细（收帧 / 丢帧）",
     t_live_panel_shows_coord_age),
)


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
