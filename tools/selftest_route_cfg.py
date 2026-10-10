"""自检：**「路线识别」配置按地图 id 存**（2026-10-03 新增 ✓ 用户要求）。

用户口径（逐条问过 ✓）：
  · 2026-10-02："**路线识别页签所有配置按地图 id 存**"；
  · 2026-10-03 审计后定稿：
    - 那 9 个键（攀爬参数 / 移动重试 / 起跳距离 / **小地图框选区域** / **来源** /
      **黄点跟踪参数**）从"按项目、全局"改成**按地图 id** ⇒ `datasets/map/<id>.route.json`；
    - `project.yaml`（6 个参数 + `mmap_crop`）与 `config/live.yaml`（`mmap_src` /
      `mmap_track`）**留着当"新图第一次打开时的播种值"** ✓；
    - 「地形图」叠加层的**透明度保持现状**（全局本机 `live.yaml` ✓ 纯显示偏好 ✓）。

为什么必须有它：这批键**错了不报错**，只在"换图之后"表现为**参数串了** ✗
（A 图的起跳距离 / 框选区域拿去 B 图用 ⇒ 画面看着像寻路坏了 ✓）。钉四件：
  ① `route_cfg` 存读往返 + **A/B 两图互不影响**（"没配过"与"配过又清空"也要分得开 ✓）；
  ② `DecisionSettings.apply_route_cfg`：**只认那 6 个参数键** ✗（别的键一个都不许动）、
     类型跟着现值走、坏值保留原值；
  ③ **播种**取自"老家"（`project.yaml` 的 `decision` / `mmap_crop`、`live.yaml` 的两键 ✓），
     ⚠ **不是** settings 当前值（换图那一刻那是**上一张图**的值 ⇒ 会串 ✗）；
  ④ 面板/窗口的接线：切图与换项目两处都灌、三个写口都存、保存钩子一处带写回 ✓。

⚠ 还有一条**防漏**（最有价值的一条）：扫 `gui/route_panel.py` 里所有 `getattr(settings, "…")`
  的键 ⇒ 断言它们都在 `route_cfg.KEYS` 里（或者在白名单里 ✓）—— 以后**加参数忘登记**就红 ✗
  （漏一个 = 那条参数悄悄退回"按项目"，而没人会发现 ✓）。
"""

import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")            # type: ignore[attr-defined]
except Exception:                                       # noqa: BLE001
    pass

from core import mapdata, route_cfg                     # noqa: E402

_TMP = None


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _tmp():
    global _TMP
    if _TMP is None:
        _TMP = Path(tempfile.mkdtemp(prefix="selftest-routecfg-"))
    return _TMP


def _src(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _with_map_dir(fn):
    """把 `mapdata.map_dir()` 指到临时目录（**绝不碰真实配置** ✓ 同别的自检口径 ✓）。"""
    old = mapdata.map_dir
    d = _tmp() / "map"
    d.mkdir(parents=True, exist_ok=True)
    mapdata.map_dir = lambda: d
    try:
        return fn(d)
    finally:
        mapdata.map_dir = old


def t_roundtrip_and_two_maps_do_not_mix():
    """① 存读往返 + **两张图互不影响**；"没配过"与"配过又清空"分得开 ✓。"""
    def run(_d):
        check(route_cfg.load("100") is None, "没配过的图该给 None（= 要播种 ✓）")
        route_cfg.save("100", {"jump_start_px": 40, "mmap_src": "stream"})
        route_cfg.save("200", {"jump_start_px": 99})
        check(route_cfg.load("100")["jump_start_px"] == 40,
              "A 图的值被 B 图改了（这正是要治的「串」✗）：%r" % (route_cfg.load("100"),))
        check(route_cfg.load("200")["jump_start_px"] == 99, "B 图没读回来：%r"
              % (route_cfg.load("200"),))
        # 「配过又清空」≠「没配过」✗（一个给 {}、一个给 None ✓ 同 core.battle 的口径 ✓）
        route_cfg.save("300", {})
        check(route_cfg.load("300") == {},
              "空文件该给 {}（= 明确空，不该再播种 ✗）：%r" % (route_cfg.load("300"),))
        check(route_cfg.path("300").name == "300.route.json",
              "文件名不是 <id>.route.json（与 zones/battle/mapcalib 并列 ✓）：%r"
              % (route_cfg.path("300"),))
        # 手改坏的文件 ⇒ 当"没这份"（别崩 ✗）
        route_cfg.path("400").write_text("{这不是 json", encoding="utf-8")
        check(route_cfg.load("400") is None, "坏文件该当「没这份」（而不是抛异常 ✗）")
    _with_map_dir(run)


def t_apply_only_touches_param_keys():
    """② `apply_route_cfg` **只认那 6 个参数键**；类型跟着现值；坏值保留原值 ✓。"""
    from decision.agent import DecisionSettings
    s = DecisionSettings()
    s.goto_timeout_s = 123.0                     # 不在清单里 ⇒ 一个字节都不许动 ✗
    s.facing_timeout_min = 7.0
    n = s.apply_route_cfg({
        "jump_start_px": 55,
        "move_retry_ms": 2500,
        "climb_align_gap_ms": "180",             # 字符串数字 ⇒ 该转成 int ✓
        "climb_retry_delay_s": 2.5,              # 现值是 float ⇒ 保持 float ✓
        "goto_timeout_s": 999.0,                 # ⛔ 不在清单里 ⇒ **不许**被改 ✗
        "mmap_crop": [1, 2, 3, 4],               # ⛔ 面板的键 ⇒ settings 不认 ✗
        "mmap_src": "live",                      # ⛔ 同上 ✗
    })
    check(s.jump_start_px == 55 and isinstance(s.jump_start_px, int),
          "`jump_start_px` 没灌进去 / 类型不对：%r" % (s.jump_start_px,))
    check(s.move_retry_ms == 2500 and isinstance(s.move_retry_ms, int),
          "`move_retry_ms` 没灌进去：%r" % (s.move_retry_ms,))
    check(s.climb_align_gap_ms == 180 and isinstance(s.climb_align_gap_ms, int),
          "字符串数字没转成 int（老配置里手写的 \"180\" ✓）：%r" % (s.climb_align_gap_ms,))
    check(abs(s.climb_retry_delay_s - 2.5) < 1e-9 and isinstance(s.climb_retry_delay_s, float),
          "float 类型没保持：%r" % (s.climb_retry_delay_s,))
    check(s.goto_timeout_s == 123.0 and s.facing_timeout_min == 7.0,
          "动了**不在清单里**的参数（那会让「按图」污染到别的键 ✗）：%r / %r"
          % (s.goto_timeout_s, s.facing_timeout_min))
    check(not hasattr(s, "mmap_crop") or getattr(s, "mmap_crop", None) in (None, []),
          "把面板的键灌到 settings 上了（那三样不归它 ✗）")
    # 坏值 ⇒ 保留原值（不猜 ✗）
    s.jump_start_px = 33
    s.apply_route_cfg({"jump_start_px": "abc"})
    check(s.jump_start_px == 33, "坏值该保留原值（不猜一个默认 ✗）：%r" % (s.jump_start_px,))
    check(n >= 3, "返回值该报「真的灌了几项」：%r" % (n,))


def t_no_chase_path_is_per_map():
    """⭐ 「**禁用杀怪寻路**」**按地图 id 存**（用户 2026-10-06 ✓ 原话："**从此 禁用杀怪寻路就是
    按地图id存的数据，而不是在设置里全局一份**"✓）。

    为什么该按图：它决定的是"**这张图**要不要用寻路追怪"（寻路成不成立是**图**的属性 ✗），
    跟"这个项目"不是一回事 —— 换图不该把它带过去，两张图各配各的 ✓。

    钉四件：
      ① 它在 `route_cfg.PARAM_KEYS` 里 ✓（登记了就自动"读 / 写 / 播种"三处齐 ✓）；
      ② **按图隔离**：A 图开了 ⇒ B 图那份**读不到它** ✓（这正是这一套要治的"串图" ✗）；
      ③ `apply_route_cfg` 能灌**真布尔**（开 ✓ 关 ✓ 都行）；
      ④ ⚠ 手改坏的文件里写成 `"false"` **字符串** ⇒ **保留原值** ✗ ——
         `bool("false")` 是 **True**（意思正好反了 ✗ 最坑 ✓）⇒ 那种值一律不收 ✓。
    """
    from core import route_cfg
    from decision.agent import DecisionSettings

    check("disable_chase_pathfinding" in route_cfg.PARAM_KEYS,
          "没登记进 `PARAM_KEYS` ⇒ 它还是**按项目/全局**那份（用户 2026-10-06 要按图 ✗）：%r"
          % (route_cfg.PARAM_KEYS,))

    def run(_tmp=None):                         # ⚠ `_with_map_dir` 会传一个临时目录进来 ✓
        route_cfg.save("700", {"disable_chase_pathfinding": True})
        route_cfg.save("701", {"jump_start_px": 5})
        check(route_cfg.load("700")["disable_chase_pathfinding"] is True,
              "A 图没存住：%r" % (route_cfg.load("700"),))
        check("disable_chase_pathfinding" not in (route_cfg.load("701") or {}),
              "B 图读到了 A 图的开关 ⇒ 串图 ✗（按图存的意义就在这儿）：%r"
              % (route_cfg.load("701"),))

    _with_map_dir(run)

    s = DecisionSettings()
    s.disable_chase_pathfinding = False
    s.apply_route_cfg({"disable_chase_pathfinding": True})
    check(s.disable_chase_pathfinding is True,
          "按图那份灌不进 settings（那按图存就白存了 ✗）：%r" % (s.disable_chase_pathfinding,))
    s.apply_route_cfg({"disable_chase_pathfinding": False})
    check(s.disable_chase_pathfinding is False, "关也灌不回来：%r" % (s.disable_chase_pathfinding,))
    # ⚠ 原值必须取 **False** ✗ —— 拿 True 当原值时，坏值被错误收成 True 也**看不出来** ✗
    #   （这条我自己先踩过一次：反向验证"居然过了"⇒ 说明用例没钉住 ✓ 现在这样才钉得住 ✓）。
    s.disable_chase_pathfinding = False
    s.apply_route_cfg({"disable_chase_pathfinding": "false"})
    check(s.disable_chase_pathfinding is False,
          "手改坏的 `\"false\"` 字符串被收下了 ⇒ `bool(\"false\")` 是 **True** ✗"
          "（意思正好反了，最坑 ✓ 该保留原值 ✗）：%r" % (s.disable_chase_pathfinding,))


def t_seed_comes_from_homes_not_from_memory():
    """③ **播种取自老家**（project.yaml / live.yaml ✓），**不是** settings 当前值 ✗。"""
    import gui.route_panel as rpmod
    from gui.route_panel import RoutePanel
    fake_proj = SimpleNamespace(get=lambda k, d=None: {
        "decision": {"jump_start_px": 7, "climb_align_gap_ms": 250},
        "mmap_crop": [10, 20, 200, 150],
    }.get(k, d))
    fake = SimpleNamespace(project=fake_proj)
    # ⚠ 要 patch **面板里那个名字**（它是 `from core.config import load_live` 进来的 ✓
    #    改 `core.config.load_live` 对它没用 ✗ —— 这条我第一版就踩了 ✓）
    old_live = rpmod.load_live
    rpmod.load_live = lambda: {"mmap_src": "live", "mmap_track": {"mmap_max_jump": 90}}
    try:
        got = RoutePanel._seed_route_cfg(fake)
    finally:
        rpmod.load_live = old_live
    check(got.get("jump_start_px") == 7,
          "播种没取 `project.yaml` 的 `decision` 那份（取成内存里的值会**串上一张图** ✗）：%r"
          % (got,))
    check(got.get("climb_align_gap_ms") == 250, "同上：%r" % (got,))
    check(got.get("mmap_crop") == [10, 20, 200, 150],
          "`mmap_crop` 的播种没取 project.yaml 顶层（它的老家 ✓）：%r" % (got,))
    check(got.get("mmap_src") == "live" and got.get("mmap_track", {}).get("mmap_max_jump") == 90,
          "`mmap_src`/`mmap_track` 的播种没取 `config/live.yaml`（它们的老家 ✓）：%r" % (got,))
    # `decision` 段为空（新项目还没存过）⇒ 用 settings 当前值兜底（那就是默认值 ✓）
    p2 = SimpleNamespace(get=lambda k, d=None: {} if k == "decision" else d)
    fake2 = SimpleNamespace(project=p2)
    got2 = RoutePanel._seed_route_cfg(fake2)
    from decision.agent import settings as _s
    check(got2.get("move_retry_ms") == getattr(_s, "move_retry_ms", None),
          "项目还没存过参数时该用 settings 当前值兜底（否则新图全是硬编码默认 ✗）：%r" % (got2,))


def t_apply_is_read_only():
    """⭐⭐ **切图 / 换项目是只读动作：一个字都不许写**（2026-10-03 现场踩了 ✓）。

    病（实测）：`_apply_route_cfg` 播种时顺手把 `<id>.route.json` 写了出来 ✗ ⇒
      · 任何"跑一遍自检 / 探针"都会把**当时的假值**（用例 mock 出来的 `live.yaml` /
        项目值 ✓）写成**真实配置** ✗✗；
      · 于是**下一次**跑用例时读到的是上一次的脏值 ⇒ 实测 3 个文件被写出来、其中一个
        `climb_retry_delay_s=1.0` 把"换项目刚绑好的 3.0"顶掉 ⇒ `selftest_minimap`
        当场红（"bind 里的 setValue 又写回配置了"✓）。
    ⇒ 只读化的理由（不是"省一次 IO" ✓）：**播种源就是老家**（`project.yaml` / `live.yaml` ✓），
      而它们会被用户的每次改动同步更新 ⇒ 下次读不到文件时**再播一次结果一样** ✓；
      真正该落盘的时刻只有一个 —— **用户改了参数**（`_save_route_cfg` ✓ 经保存钩子 / 三个写口 ✓）。
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from decision.agent import settings as _s
    from gui.route_panel import RoutePanel

    class _Combo:
        def __init__(self):
            self.i = 0

        def findData(self, v):                   # noqa: N802
            return {"stream": 0, "live": 1}.get(v, -1)

        def currentIndex(self):                  # noqa: N802
            return self.i

        def blockSignals(self, _b):              # noqa: N802
            pass

        def setCurrentIndex(self, i):            # noqa: N802
            self.i = i

    wrote = []

    def run(_d):
        import core.route_cfg as rc
        _old = rc.save
        rc.save = lambda mid, vals: wrote.append(str(mid))
        _saved = {}
        for k in route_cfg.PARAM_KEYS:
            _saved[k] = getattr(_s, k, None)
        try:
            fake = SimpleNamespace(
                _map_id=lambda: "999", _mmap_cli=None, _mmap_track=None,
                cmb_mmap_src=_Combo(), _sp_track={}, _crop_override=None,
                _src_override=None, project=None,
                _seed_route_cfg=lambda: {"jump_start_px": 12,
                                         "climb_align_gap_ms": 180,
                                         "mmap_src": "live"},
                _refresh_crop_status=lambda: None)
            n = RoutePanel._apply_route_cfg(fake, "999")
            # ⚠ 快照要在**恢复之前**取（第一版把断言写在 `finally` 后面 ⇒ 读到的是恢复后的
            #    原值 ⇒ 假红 ✗ 自己踩的 ✓）
            _got = {k: getattr(_s, k, None)
                    for k in ("jump_start_px", "climb_align_gap_ms")}
        finally:
            rc.save = _old
            for k, v in _saved.items():
                setattr(_s, k, v)
        check(wrote == [], "切图/换项目竟然写盘了（会把用例的假值写成真配置 ✗）：%r" % (wrote,))
        check(n >= 2, "只读也不该白跑：那 6 个参数该被灌进 settings（灌了 %r 项）" % (n,))
        check(_got.get("jump_start_px") == 12,
              "播种值没灌进 settings（那\"新图继承老家值\"就落空了 ✗）：%r" % (_got,))
        check(fake._src_override == "live",
              "来源的覆盖值没记上（`_mmap_src()` 还得靠它 ✓）：%r" % (fake._src_override,))
        # 落了盘的文件也一个都不该有 ✓
        check(list(_d.glob("*.route.json")) == [],
              "目录里出现了 `<id>.route.json`（只读动作不该留痕 ✗）：%r"
              % ([p.name for p in _d.glob("*.route.json")],))

    # ⚠ 用**自己一块干净目录**（别的用例会正常写文件 ⇒ 共用目录时会假红 ✗ 踩了 ✓）
    _old_dir = mapdata.map_dir
    _d = _tmp() / "map_readonly"
    _d.mkdir(parents=True, exist_ok=True)
    mapdata.map_dir = lambda: _d
    try:
        run(_d)
    finally:
        mapdata.map_dir = _old_dir

    rp = _src("gui/route_panel.py")
    _body = rp.split("def _apply_route_cfg", 1)[-1].split("def _refresh_crop_status", 1)[0]
    # ⚠⚠ **扫之前先去注释**（2026-10-10 ✓ 修一次假红）：那段说明里写着
    #   "真正落盘的时刻只有一个 —— 用户改了参数（走 `_save_route_cfg` ✓ 见 `core.route_cfg.save`）"
    #   ⇒ 裸子串 `route_cfg.save` **在注释里也会命中** ✗ ⇒ 报"又写盘了" ✗（其实一个写都没有 ✓）。
    #   ⇒ 只钉**真的那一下调用**（`route_cfg.save(` ✓ 前后不留白：`core.route_cfg.save` 这种
    #     引用**不会**命中 ✓），并且先把注释切掉 ⇒ 以后注释里怎么写都不会假红 ✓。
    _code = "\n".join(ln.split("#", 1)[0] for ln in _body.splitlines())
    check("route_cfg.save(" not in _code,
          "`_apply_route_cfg` 里又写盘了（切图是只读动作 ✗）")


def t_panel_registers_every_key():
    """④ **防漏**：面板上读的每个 settings 键，都必须在 `route_cfg.KEYS` 里（或白名单 ✓）。

    这条是本次改造里最有价值的一条 ✓：以后**加一个参数忘了登记** ⇒ 它就悄悄退回"按项目" ✗
    （换图串参数），而**没有任何现象**提示你 ✗ —— 这条用例会让它当场红 ✓。
    """
    import re
    src = _src("gui/route_panel.py")
    keys = set(re.findall(r'getattr\(settings,\s*"([a-z0-9_]+)"', src))
    #: 白名单：**不归本次改造**的（不在本页控件上、或是运行时状态 / 别的面板推过来的 ✓）
    allow = {
        "enabled",              # 自动开关（运行时 ✓）
        "rest_state", "rest_pending", "next_afk_monotonic",   # 防掉线运行时状态 ✓
        "custom_timers", "custom_timer_next",                 # 定时行为（在别的页签配 ✓）
        "resetall_interval",                                  # 全局按键维护 ✓
        "route_goto_set",                                     # 前往预览（运行时 ✓）
        "battle_zones",                                       # 已按图 ✓（<id>.battle.json）
        "align_tol_px",                                       # 控件在「设置 → 判定参数」✗ 不在这页
    }
    missing = sorted(k for k in keys
                     if k not in allow and k not in route_cfg.PARAM_KEYS
                     and k not in route_cfg.MAP_KEYS)
    check(not missing,
          "面板上有这些键**没登记**到 `core/route_cfg.py` ⇒ 它们会悄悄退回「按项目」（换图串 ✗）：%r"
          % (missing,))


def t_wiring_is_in_place():
    """⑤ 接线（源码级）：切图/换项目两处都灌；三个写口都存；保存钩子一处带写回 ✓。"""
    rp = _src("gui/route_panel.py")
    check('self._apply_route_cfg(self._map_id())' in rp,
          "切项目（`bind`）时没灌「这张图」的配置 ✗")
    check("self._apply_route_cfg(mid)" in rp,
          "「手动更换」换图时没灌「这张图」的配置 ✗（那换图还是用上一张图的参数 ✗）")
    check(rp.count("self._save_route_cfg()") >= 3,
          "三个写口（框选 / 来源 / 黄点跟踪）没都写回这张图 ✗：%d 处"
          % rp.count("self._save_route_cfg()"))
    mw = _src("gui/main_window.py")
    check("rp._save_route_cfg()" in mw,
          "保存钩子里没有「顺手写回这张图」✗（那是「以后加控件不会漏」的唯一保证 ✓）")
    check("set_save_hook(_save_everywhere)" in mw,
          "钩子没挂上（`settings.save()` 之后不会写回图文件 ✗）")
    # ⚠ 面板那三样**不许**塞进 settings（它们不是 settings 属性 ✓）
    check('"mmap_crop"' not in _src("core/route_cfg.py").split("PARAM_KEYS = (")[1].split(")")[0],
          "`mmap_crop` 被登记进 PARAM_KEYS 了（那三样归面板 ✗ 会被灌到 settings 上 ✗）")


TESTS = (
    ("按图存读：往返 + **两图互不影响** +「没配过」与「清空」分得开",
     t_roundtrip_and_two_maps_do_not_mix),
    ("`apply_route_cfg` 只认 `PARAM_KEYS` 里的键（别的键一个都不许动）+ 类型/坏值口径",
     t_apply_only_touches_param_keys),
    ("⭐ 「禁用杀怪寻路」**按地图 id 存**（用户 2026-10-06）：两图隔离 / 真布尔灌得进 / "
     "`\"false\"` 字符串不收（`bool(\"false\")` 是 True ✗ 意思正好反了）",
     t_no_chase_path_is_per_map),
    ("播种取自「老家」（project.yaml / live.yaml）——**不是** settings 内存值（否则串图 ✗）",
     t_seed_comes_from_homes_not_from_memory),
    ("切图/换项目是**只读**：一个字都不许写（现场踩过：播种写盘 ⇒ 假值污染真配置）",
     t_apply_is_read_only),
    ("防漏：面板上每个 settings 键都在 `route_cfg.KEYS` 里（加参数忘登记 ⇒ 红）",
     t_panel_registers_every_key),
    ("接线：切图/换项目两处都灌 + 三个写口都存 + 保存钩子一处带写回",
     t_wiring_is_in_place),
)


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:                          # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    if _TMP is not None:
        shutil.rmtree(_TMP, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
