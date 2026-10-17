"""AI 结构自检：把「架构边界」变成**会变红的钉子**（用户 2026-10-10「动手」✓）。

**为什么要这一套**（不是洁癖 ✗，是**用现场事故换来的** ✓）：2026-10-10 那一轮连报 6 个
bug，其中 **5 个**是同一个病：

  ① **同一事实两条链**：地形图自己用差分算玩家坐标（和信息栏两个数 ✗）；
     `ws.screen`（原生识别）vs `_takeover_st`（留住接管 ✗）；界面各自 `new PlayerLocator`
     （2026-10-06 那次 ✗）；
  ② **每拍职责被「早退」吃掉**：`if st == self._screen_state: return` 把「按时间到点的判据」
     挡在后面 ✗（同一处踩过**三次** ✗）；
  ③ **隐式契约靠注释维护**：`climbing_vertical()` 天然滞后一拍、限速跳过的拍必须补喂
     `tick_timing` ✗。

⇒ 这三条**一条静态检查就能拦住** ✓，而它们原先只活在注释里 ✗。本套件把它们钉死：

    A. **一个事实一个写者**（`t_single_writer_*` ✓）—— AST 找**赋值点**（注释/字符串里
       写什么都不算 ✓），对不上名单就红 ✓；
    B. **没有 `setattr` 后门**（`t_no_setattr_backdoor` ✓）—— 挡「绕过 A 偷偷写」✗；
    C. **给 Agent 的 `ws` 必须在 `agent.tick` 之前装好**（`t_pipeline_order_*` ✓）——
       否则 Agent 读到**上一拍**的界面状态 ✗；
    D. **每拍职责必须在早退之前**（`t_screen_beat_*` ✓）；
    E. **分层/环/巨型文件**（`t_layering_*` / `t_giant_*` ✓）—— 用的是 `tools/arch_scan.py`
       的**同一套判据**（人看报告、机器拦截 ✓ 不是两套 ✗），基线**冻结**：只管「不许更差」✓
       （全清不现实 ✗，但不许悄悄多出来 ✓）。

跑法：

    python -m tools.selftest_arch              # 全过 0 / 有失败 1
    python -m tools.selftest_arch --list       # 只列出各事实的**写点**（复核 / 写文档用 ✓）

⚠ 本套件**纯静态**（只读源码 + 架构图）⇒ 不 import Qt/torch，跑起来毫秒级 ✓；
  「真跑起来」的那几条在 `tools/selftest_live_panel.py` / `selftest_screen_state.py` 里 ✓。
"""
import ast
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.arch_scan import scan, skip                                 # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


#: 运行期包（**不含 `tools/`** ✓ —— 探针/用例里写这些字段是当 fixture 用 ✓ 不算实现 ✗）
#: ⭐ `runtime`（2026-10-15 阶段 1「编排收口」新建 ✓）：一帧的**调度**（阶段顺序 / 限速闸 /
#:   接管状态 ✓）—— 它**必须**在这份名单里 ✗，否则 R1「一个事实一个写者」会扫不到它 ⇒
#:   真源搬进去之后基线当场变成"期望 2 处、实得 0 处"✗（判据空转的最隐蔽形态 ✓）。
RUNTIME_PKGS = ("core", "perception", "decision", "runtime", "gui", "link",
                "remote_kbd", "deploy")

# ══════════════════════════════════════════════════════════════════════════════
#  A. 一个事实一个写者：**这份清单 = 那些「多一个写者就出事」的事实**
#     每项：
#       attr   —— 赋值目标的**最后一段**（`_agent_mod.LAST_POS` / `ws.screen` 都命中 ✓）
#       writes —— 允许的**实现写点**：{相对路径: 处数}（多一处 = 又长出一条链 ✗）
#       decl   —— 允许的**声明**处（模块级 `X = None` 这种 ✓ 与「写」分开算 ✓）
#       why    —— 为什么要它唯一（写清代价 ⇒ 后人不敢随手加 ✗）
# ══════════════════════════════════════════════════════════════════════════════
FACTS = (
    {
        "attr": "LAST_POS",
        "writes": {"gui/live_thread.py": 2},
        "decl": {"decision/agent.py": 1},
        "why": "**玩家坐标的唯一来源**（界面那行状态字 + 地形图上的圈 ✓）—— 它一旦有第二个"
               "写者，就会出现「两处坐标对不上」✗（2026-10-06 实锤：地形图自己用差分算一份 ✗；"
               "2026-10-10 又差点在界面里重算 ✗）。写者只能是实时回路 ✓ 别处**只读** ✓。",
    },
    {
        "attr": "screen",                      # `ws.screen`（喂给 Agent 的那份 ✓）
        "writes": {"gui/live_thread.py": 2},
        # 默认值在感知层的容器上（`WorldState.screen = "combat"` ✓）—— 那是**声明**不是实现 ✓
        "decl": {"perception/world_state.py": 1},
        "why": "**Agent 的停手判据就是它**（`ws.screen != combat` ✓）—— 两个写点 = 两条路径"
               "（推理路 / 只收流路 ✓ 这两处**都该有** ✓）且都必须取自 `_takeover_st` ✓"
               "（取值本身由 `selftest_screen_state` 钉着 ✓）。⚠ 别把默认值也算成写者 ✗"
               "（2026-10-10 实测就是这样：`world_state.py` 只声明 ✓ 写点全在实时回路 ✓）。",
    },
    {
        "attr": "_screen_state",
        "writes": {"gui/live_thread.py": 2},
        "decl": {},
        "why": "**原生识别结果**（录屏 / 样本采集 / 状态行 / 报警都读它 ✓）。只许 `_screen_beat`"
               "里那**一个**写点（+ `__init__` 的初值 ✓）—— 用户报的「测谎中还按方向键」就是"
               "这条事实被拿来当 Agent 的判据用出来的 ✗（Agent 那份必须另走 `_takeover_st` ✓）。",
    },
    {
        "attr": "_takeover_st",
        "writes": {"gui/live_thread.py": 2},
        "decl": {},
        "why": "**该报给 Agent 的界面状态**（留住接管后的那份 ✓）—— 唯一写点在 `_screen_beat`"
               "的留住逻辑里 ✓（+ 初值 ✓）。多一个写者 = 「谁说了算」又说不清 ✗。",
    },
    {
        "attr": "_lie_warn_at",
        "writes": {"gui/live_thread.py": 5},
        "decl": {},
        "why": "**「这一场测谎」的账本**（开一场 / 响 warn / 成功清零 / C 保险收工 ✓ **四处都在"
               "`_screen_beat` 同一个函数里** ✓ 2026-10-10 逐处核过 ✓）—— 录屏的停录判据与"
               "Agent 的留住判据**共用**它 ✓ ⇒ 两个写者就会出现「录屏停了、Agent 还在动」✗"
               "（或反过来 ✗）。"
               "⚠⚠ **第 5 个写点**（2026-10-17 ✓ 用户：「**测谎触发后录屏最多30秒，超过后需要"
               "停止相关逻辑，然后判定是测谎成功还是失败**」✓）：`_lie_window_over` 里那一句"
               "清零 ✓ —— 它和上面四处**同一本账、同一个语义**（\"这一场没了\"✓），只是"
               "**触发条件**多了\"30 秒到点\"那一条 ✓ ⇒ 仍是**一条链** ✓（不是新写的第二条链 ✗）。"
               "⚠ 它和停录**在同一个函数里、紧挨着**（先停录、再清账 ✓）⇒ 不会出现"
               "「录屏还在录、账已经清了」那种错位 ✓。",
    },
    {
        "attr": "_lie_seen_game",
        "writes": {"gui/live_thread.py": 5},
        "decl": {},
        "why": "同一本账的「这场进过小游戏吗」。⚠ 2026-10-10 晚**口径变过**：现在它**不**决定"
               "接管时长（接管只看「这一场还在不在」✓ 见 `_screen_beat` ✓），而是打点里分辨"
               "「真小游戏 / warn 误判」的**唯一线索** ✓（将来修好 `lie_game` 模板要靠它收紧"
               "策略 ✓ 别删 ✗）—— 但**写者仍必须唯一** ✓：它与录屏那本账同源 ✓。"
               "⚠⚠ **第 5 个写点**（2026-10-17 ✓ 用户：「**测谎触发后录屏最多30秒…然后判定**」✓）："
               "`_lie_window_over` 里那句清零 ✓ —— 与 `_lie_warn_at` 的第 5 个写点**在同一个函数里、"
               "紧挨着**（先停录、再清这本账 ✓），语义仍是「这一场没了」✓ ⇒ **同一条链** ✓。",
    },
    {
        "attr": "_lie_sess_t0",
        "writes": {"gui/live_thread.py": 4},
        "decl": {},
        "why": "同一场测谎的**起点时刻**（`LIE_SESSION_MAX_S` 靠它判一场有没有超时 ✓）—— "
               "与录屏同一个口径 ✓ 不许第二个人写 ✗。",
    },
    {
        "attr": "_pipe_fps",
        "writes": {"gui/live_thread.py": 3},
        "decl": {},
        "why": "**重活闸**的档位（配置热读 + 初值 + 解析失败兜底 ✓ 三处都在这一个文件里 ✓）"
               "—— 它是「每秒跑多少拍重活」的**唯一权威** ✓ 别处不许改 ✗。",
    },
    {
        "attr": "_heavy_at",
        "writes": {},
        "decl": {},
        "why": "⛔ **已搬走**（2026-10-15 阶段 1「编排收口」✓）：放行时刻的真源 = "
               "`runtime.pipeline.HeavyGate.last_pass_at` ✓ —— `gui/live_thread.py` 里"
               "**一次都不许再出现**它 ✗（留个副本 = 两个时钟 ⇒ 闸的节奏看人 ✓）。"
               "基线从「live_thread 里 2 处」改成**全运行期 0 处** ✓。",
    },
    {
        "attr": "fps_limit",
        "writes": {"runtime/pipeline.py": 2, "gui/live_thread.py": 1},
        "decl": {},
        "why": "重活闸档位的**真源**（`HeavyGate.__init__` 里两处：初值 + 参数 ✓）；"
               "`gui/live_thread.py` 那一处是**委托口**的 setter（`self._gate.fps_limit = …` ✓）"
               "—— 2026-10-15 从 `gui/live_thread.py` 搬过来 ✓ 见上面 `_pipe_fps` 那条 ✓。",
    },
    {
        "attr": "last_pass_at",
        "writes": {"runtime/pipeline.py": 2},
        "decl": {},
        "why": "闸的**上一次放行时刻**（初值 + 放行那一次 ✓ 都在 `HeavyGate` 自己身上 ✓）"
               "—— 真源 ✓ 见上面 `_heavy_at` 那条 ✓。",
    },
)


def _runtime_files():
    """运行期包里的 `.py`（跳过 venv / 备份 / 缓存 ✓，scope 见 `RUNTIME_PKGS` ✓）。"""
    out = []
    for pkg in RUNTIME_PKGS:
        d = ROOT / pkg
        if not d.is_dir():
            continue
        out += [p for p in sorted(d.rglob("*.py")) if not skip(p)]
    return out


def _dots(node):
    """把赋值目标折成 (点号名字, 类别)：`a.b.c` -> (`a.b.c`, `attr`)；`x` -> (`x`, `name`)。"""
    if isinstance(node, ast.Name):
        return node.id, "name"
    if isinstance(node, ast.Attribute):
        base, _k = _dots(node.value)
        return ((base + "." + node.attr) if base else node.attr), "attr"
    if isinstance(node, ast.Subscript):
        return _dots(node.value)
    return "", "other"


def _sites(attr):
    """全运行期源码里对 `attr` 的**赋值点**：[(相对路径, 行号, 类别, 完整点号名), ...]。

    ⚠ 用 AST ✗ 不用正则 —— 注释和字符串里写 `ws.screen = ...` 到处都是（本仓库注释极多 ✓），
      正则会把它们全算成「写者」⇒ 判据当场失效 ✗（2026-10-10 实测过 ✓）。
    """
    hits = []
    for p in _runtime_files():
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:                       # 语法坏掉的文件不算（另有编译检查 ✓）
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.NamedExpr)):
                targets = [node.target]
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            else:
                continue
            for t in targets:
                full, kind = _dots(t)
                if not full:
                    continue
                if ((kind == "attr" and full.rsplit(".", 1)[-1] == attr)
                        or (kind == "name" and full == attr)):
                    hits.append((str(p.relative_to(ROOT)).replace("\\", "/"),
                                 node.lineno, kind, full))
    return hits


def _dump():
    """`--list`：只列写点（复核 / 写文档用 ✓）。"""
    for f in FACTS:
        print("-- %s" % f["attr"])
        for rel, ln, kind, full in _sites(f["attr"]):
            print("     %-24s L%-6d %-5s %s" % (rel, ln, kind, full))
    return 0


def _one_fact(name):
    """钉一个事实：实现写点 / 声明处**都必须正好等于名单** ✓。"""
    f = [x for x in FACTS if x["attr"] == name][0]
    hits = _sites(name)
    got_w = Counter(rel for rel, _ln, kind, _full in hits if kind == "attr")
    got_d = Counter(rel for rel, _ln, kind, _full in hits if kind == "name")
    want_w, want_d = Counter(f["writes"]), Counter(f["decl"])
    if got_w != want_w or got_d != want_d:
        detail = "\n".join("       %s L%d %s %s" % (rel, ln, kind, full)
                           for rel, ln, kind, full in hits)
        raise AssertionError(
            "「%s」的写者名单对不上（**两个写者 = 两条链** ✗）：\n"
            "     期望 实现写点 %s / 声明 %s\n"
            "     实际 实现写点 %s / 声明 %s\n"
            "     ⚠ 这是**有意的门槛**：如果这次确实要新增/挪动写点 ⇒ 改本文件里 %s 的名单，"
            "并在 `why` 里写清「为什么它可以有第二个写者」✓；否则就是又长出一条链 ✗\n"
            "     实际写点：\n%s\n     原来为什么唯一：%s"
            % (name, dict(want_w), dict(want_d), dict(got_w), dict(got_d),
               name, detail, f["why"]))
    # 名单里写的文件必须真在（防止「名单过时 ⇒ 判据空转」✗）
    for rel in list(f["writes"]) + list(f["decl"]):
        check((ROOT / rel).is_file(),
              "名单里写着 %s ⇒ 文件不存在（判据在**空转** ✗ 赶紧改名单 ✓）" % rel)


def t_single_writer_coords():
    """A-① **玩家坐标只有一个写者**（`LAST_POS` ✓）：实时回路写、界面/地形图**只读** ✓。

    为什么：2026-10-06 实锤过「界面自己算一份」✗（两个数对不上）；2026-10-10 又差点在
    地形图里重算一遍 ✗（同一条链的第二个写者 ✓）。
    """
    _one_fact("LAST_POS")


def t_single_writer_screen_takeover():
    """A-② **界面状态是两个事实、各自唯一写者**（原生 `_screen_state` / 接管 `_takeover_st` ✓）。

    用户 2026-10-10 报的「测谎开始后 Agent 还在按方向键」✗ 就是这两条事实被当成一条用 ✗：
    Agent 读的那份必须是 `_takeover_st`（留住接管 ✓），而录屏 / 状态行读 `_screen_state`
    （原生 ✓）—— **故意分开** ✓ 谁把它们合回去，这条就红 ✓。
    """
    for name in ("screen", "_screen_state", "_takeover_st", "_lie_warn_at",
                 "_lie_seen_game", "_lie_sess_t0"):
        _one_fact(name)


def t_single_writer_heavy_gate():
    """A-③ **重活闸的账只有一个写者**（`_pipe_fps` / `_heavy_at` / 两个真源 ✓）。

    闸的档位来自配置**热读** ✓（现场能调 ✓）；要是别处也能写它 ⇒「现在到底限没限速」就
    说不清了 ✗（那正是排查性能时最怕的 ✓）。

    ⭐ 2026-10-15 阶段 1「编排收口」：**实现搬进 `runtime/pipeline.py::HeavyGate`** ✓ ——
    这条顺手多钉一件**搬家之后才可能出的错**：`runtime` 是「调度」，**不许 import
    `gui` / `tools`** ✗（一旦沾上 Qt / 屏幕 / Ultralytics，它就再也拆不出来了 ✓）。

    ⚠ 这条**故意不靠** E-① 那条分层棘轮 ✗：E-① 是"不许比基线更差"（基线可以被人加一条 ✓），
      而这里要的是**硬规矩**（谁来都不许 ✗）—— 两条一起在，才既挡住"顺手加"，也挡住
      "改基线放行" ✓。用的是 `ast` 直读 import ✗ 不靠人肉维护名单 ✓。
    """
    for name in ("_pipe_fps", "_heavy_at", "fps_limit", "last_pass_at"):
        _one_fact(name)
    bad = []
    for p in sorted((ROOT / "runtime").glob("*.py")):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.ImportFrom) and n.module:
                root = n.module.split(".")[0]
            elif isinstance(n, ast.Import):
                root = n.names[0].name.split(".")[0]
            else:
                continue
            if root in ("gui", "tools"):
                bad.append("runtime/%s:%d  %s" % (p.name, n.lineno, root))
    check(not bad,
          "`runtime` 里出现了指向 `gui` / `tools` 的 import ✗：\n      %s\n"
          "     （它是「一帧的调度」✓ 只许依赖 core / perception / decision ✓ —— "
          "沾上 Qt 或探针，阶段 2/3 就没法做了 ✗）" % "\n      ".join(sorted(set(bad))))


def t_no_setattr_backdoor():
    """B. **不许用 `setattr` 绕过去**（挡「绕过 A 偷偷写」✗）。

    为什么单列：A 那条是**静态读 AST** ✓，`setattr(self, "_screen_state", st)` 这种它看不见 ✗
    ⇒ 等于后门 ✓。现在全运行期** 0 处** ✓（2026-10-10 实测 ✓）⇒ 这条现在能立 ✓。
    """
    names = [f["attr"] for f in FACTS]
    bad = []
    for p in _runtime_files():
        txt = p.read_text(encoding="utf-8")
        if "setattr(" not in txt:
            continue
        for i, ln in enumerate(txt.splitlines(), 1):
            code = ln.split("#")[0]               # ⚠ 砍掉注释再判（注释里也会写它 ✓）
            if "setattr(" not in code:
                continue
            for n in names:
                if ('"%s"' % n) in code or ("'%s'" % n) in code:
                    bad.append("%s L%d: %s" % (p.relative_to(ROOT), i, ln.strip()[:90]))
    check(not bad,
          "有人用 `setattr` 写这些**唯一写者**的字段（静态判据看不见 ⇒ 后门 ✗）：\n      %s"
          % "\n      ".join(bad))


def _lt_tree():
    """`gui/live_thread.py` 的 (源码, AST)。

    ⚠⚠ **判据走 AST ✗ 绝不用文本查找** —— 本仓库注释密度极高 ✓，而注释里**照着抄出**被钉
      的那句代码是**常态**（"这块为什么这么写"的教材 ✓）⇒ 文本判据会被注释骗 ✗。
      **2026-10-10 实测踩到**：D 那条第一版就是文本查找，当场把注释里的
      `if st == self._screen_state: return`（正是三次踩坑那段教材 ✓）当成正文 ⇒ 误报 ✗。
    """
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    return src, ast.parse(src)


def _fn(tree, name):
    """按名字找函数节点（找不到返回 `None` ✓）。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _attr_assigns(scope, attr):
    """`<任意>.<attr> = ...` 的**行号**列表（AST ✓ 注释/字符串都不算 ✓）。"""
    out = []
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Attribute) and t.attr == attr:
                out.append(node.lineno)
    return sorted(out)


def t_pipeline_order_screen_before_tick():
    """C. **喂给 Agent 的 `ws` 必须在 `agent.tick` 之前装好**（同一拍 ✓）。

    为什么：`ws.screen` 是 Agent 那拍**停手 / 接管**的判据 ✓ ⇒ 若先 `agent.tick(ws)` 再填
    `ws.screen`，Agent 读到的就是**上一拍**的界面状态 ✗ —— 症状是「弹窗都出来了它还在打」✗
    或「弹窗关了它还停着」✗，而且**只在切换那一拍对不上** ⇒ 极难查 ✓。
    """
    _src, tree = _lt_tree()
    ticks = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "tick" and n.args and isinstance(n.args[0], ast.Name)
             and n.args[0].id == "ws"]
    check(ticks, "主回路里找不到 `agent.tick(ws)`（接线变了 ⇒ 这条判据要跟着改 ✗）")
    check(len(ticks) == 1,
          "`agent.tick(ws)` 找到 %d 处（行 %s）⇒ 一帧跑了几次决策？✗（重活只该出一条路 ✓）"
          % (len(ticks), ticks))
    screens = _attr_assigns(tree, "screen")
    check(len(screens) == 2,
          "`ws.screen` 的赋值点是 %d 个（行 %s）—— 两路（推理 / 只收流）都该有 ✓，"
          "多了就是又长出一条链 ✗" % (len(screens), screens))
    check(min(screens) < ticks[0],
          "喂界面状态（L%d）排在 `agent.tick(ws)`（L%d）**之后** ⇒ Agent 那拍读的是**上一拍**"
          "的界面状态 ✗" % (min(screens), ticks[0]))
    timing = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "tick_timing"]
    check(timing, "找不到时间拍（跳帧的拍必须靠它保 CD/delay 精度 ✓ 见 "
                   "`selftest_live_panel` ✓）")


def t_screen_beat_transition_vs_everytick():
    """D. **每拍职责必须在「短路」之前**／**两个口的调用序不许反**（踩过三次 ✗ ⇒ 钉住 ✓）。

    规律（血的教训 ✓）：`_screen_beat` 里那句早退是「状态没变就不往下走」的短路 ✓ ——
    而**按时间到点的判据**（留住接管「这一场还在不在」✓、录屏"待停"到点没到 ✓）每秒都必须
    被评估 ✗ ⇒ 一旦写在早退**后面**就**永远不被评估** ✗（状态不变时最容易发生 ✓：
    实测 `lie_success` 之后状态**再也不变** ⇒ 待停时刻永远没人查 ⇒ 录满 300 秒才被切 ✗）。

    ⭐ 2026-10-15（阶段 1「编排收口」）这一拍**拆成了两个口**：`_screen_every_tick`（每拍都必须
    跑 ✓）/ `_screen_on_transition`（只在"变"那一拍 ✓）⇒ 那条"顺序"从**函数内的行序**变成了
    **两个口之间的调用序** ✗ ⇒ 这条钉跟着改（**判据口径一字不变**，只是换了看的地方 ✓）：
    留住判据在短路之前 ✓、两者之间**没有 `return`** ✓、每拍口里写 `_takeover_st` 一处 ✓、
    **调用序**（每拍口先于过渡口 ✓ 且过渡口挂在它的返回值那个 `if` 里 ✓）。
    """
    _src, tree = _lt_tree()
    every = _fn(tree, "_screen_every_tick")
    on_tr = _fn(tree, "_screen_on_transition")
    check(every is not None and on_tr is not None,
          "找不到 `_screen_every_tick` / `_screen_on_transition` 这两个口 ✗（2026-10-15 阶段 1 "
          "拆出来的结构变了 ⇒ 这条钉子要跟着改 ✓ —— 但**别把两半又粘回一个函数** ✗ "
          "那正是踩过三次那个坑的温床 ✓）")

    def _is_short_circuit(node):
        v = node.value
        return (isinstance(v, ast.Compare) and isinstance(v.left, ast.Name)
                and v.left.id == "st"
                and any(isinstance(c, ast.Attribute) and c.attr == "_screen_state"
                        for c in v.comparators))

    # ① 每拍口里：留住判据必须在「状态没变就短路」之前
    early = sorted(n.lineno for n in ast.walk(every)
                   if isinstance(n, ast.Return) and n.value is not None
                   and _is_short_circuit(n))
    check(early, "`_screen_every_tick` 里找不到那句短路"
                 "（`return st != self._screen_state` ✓）")
    holds = sorted(n.lineno for n in ast.walk(every)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_hold" for t in n.targets))
    check(holds, "找不到留住判据那句（`_hold = bool(_open_sess) ...` ✓）")
    i_ret, i_hold = early[0], holds[0]
    check(i_hold < i_ret,
          "**留住接管的判据（L%d）排在短路（L%d）之后** ⇒ 状态一直不变时它**永远不被评估** ✗✗"
          "（2026-10-10 同类坑踩过三次 ✓ 用户报的「测谎中还按方向键」就是这条 ✗）"
          % (i_hold, i_ret))
    mid = sorted(n.lineno for n in ast.walk(every)
                 if isinstance(n, ast.Return) and i_hold < n.lineno < i_ret)
    check(not mid,
          "留住判据与短路之间夹着 `return`（L%s）⇒ 它后面那些**每拍职责**会被跳过 ✗"
          % mid)
    takes = _attr_assigns(every, "_takeover_st")
    check(len(takes) == 1,
          "`_screen_every_tick` 里写 `_takeover_st` 有 %d 处（行 %s）—— 决策只该有一个出处 ✗"
          % (len(takes), takes))

    # ② 两个口的调用序 + 过渡口必须挂在「每拍口」的返回值那个 if 里
    beat = _fn(tree, "_screen_beat")
    check(beat is not None, "找不到 `_screen_beat`（判据要跟着改 ✗）")
    i_every = sorted(n.lineno for n in ast.walk(beat)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "_screen_every_tick")
    i_on = sorted(n.lineno for n in ast.walk(beat)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "_screen_on_transition")
    check(len(i_every) == 1 and len(i_on) == 1,
          "`_screen_beat` 里两个口的调用各该**正好一处** ✗（实得 every=%d on=%d ✓）"
          % (len(i_every), len(i_on)))
    check(i_every[0] < i_on[0],
          "先调了 `_screen_on_transition`（L%d）、后调 `_screen_every_tick`（L%d）✗ ⇒ "
          "每拍职责晚一拍才评估（同那个踩过三次的坑 ✓）" % (i_on[0], i_every[0]))
    guarded = [n for n in ast.walk(beat)
               if isinstance(n, ast.If)
               and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                       and c.func.attr == "_screen_every_tick" for c in ast.walk(n.test))]
    check(guarded and any(
        any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == "_screen_on_transition" for c in ast.walk(g))
        for g in guarded),
        "`_screen_on_transition` **没挂在 `_screen_every_tick` 的返回值上** ✗ ⇒ 它每拍都会跑"
        "（补事件 / 抓样本 / 起停录屏变成每拍一次 ✓ 那就不是「只在变那一拍」了 ✗）")


#: 分层违反的**冻结基线**（2026-10-10 实测 3 条 ✓）—— 只许更少，不许更多 ✓
#: ⭐⭐⭐ 2026-10-16 **清空了**（用户："继续完成代码重构"✓）：三条**全部**处理完 ✓ ——
#:   · `gui.live_thread(3) -> tools.probe_codec(4)`：`tools/probe_codec.py`（568 行 ✓）
#:     **只 import 标准库** ⇒ 天生低层 ⇒ `git mv` 到 `core/probe_codec.py` ✓（14 个文件 ✓）；
#:   · `gui.live_thread(3) -> tools(4)`：**同一件事的另一半** ✓ —— 它那个"模块级 `tools` 依赖"
#:     就是 `probe_codec` ⇒ 跟着一起消失 ✓（重扫实测确认 ✓）；
#:   · `deploy.push_sweep(3) -> tools(4)`：`tools/push_presets.py`（481 行 ✓）与
#:     `tools/sweep_link.py`（398 行 ✓）**也都只 import 标准库** ✓ ⇒ 同法 `git mv` 到 `core/` ✓
#:     （6 个文件的导入一起改 ✓；`deploy/push_sweep.py` 那行是"逗号连着两个模块"✓
#:       直接换包名就对了 ✓ 因为两个都搬 ✓）。
#:   ⇒ 实际分层违反 **6 处（2026-10-10 之前）→ 3 → 0** ✓。
#: ⚠⚠ **空集合 = 从今往后一条分层违反都不许有** ✗（"只许更少"的终点 ✓）——
#:   要往这里加条目，必须在开发日志里写清理由 ✗ 别顺手塞 ✗。
LAYERING_BASELINE = set()

#: ≥1500 行的运行期文件**冻结名单**（2026-10-10 实测 ✓）—— 只许更少 ✓
#: ⚠ 这里**故意**不含 `tools/`（用例文件本来就长 ✓ 那是另一件事 ✗）
GIANT_BASELINE = {
    # ⭐⭐ 2026-10-16 **根目录 `agent.py` 已删** ✓（那行原文是：
    #   `"agent.py",  # ← 根目录 stray 副本（见 docs/架构-AI三层.md 待办 ✓）`）
    #   —— 用户："开动"✓ ⇒ `git rm agent.py` ✓（**受 git 跟踪 ⇒ 可回滚** ✓）。
    #   删前已核实（证据在开发日志 372 ✓）：**9293 行**、**98.7% 的非空行与 `decision/agent.py`
    #   逐字重复** ✓、头部 docstring 是**更老的一版** ✓、**全仓库无一处 import 它** ✓、
    #   `.bat` / 脚本也不跑它 ✓ ⇒ 它确实只是"搬家时留下的旧副本" ✓。
    #   ⚠ 名字改成 `decision/agent.py` 那条**照旧在下面** ✓（那是真身 ✗ 别跟着删 ✗）。
    "decision/agent.py",
    "decision/route.py",
    "deploy/app.py",
    "gui/live_panel.py",
    "gui/live_thread.py",
    "gui/minimap_calib.py",
    "gui/player_panel.py",
    "gui/route_panel.py",
    "gui/steps/cards.py",
    "gui/yolo_workbench.py",
    "gui/zone_editor.py",
    "perception/anchor_track.py",
    "perception/lie_motion.py",
    #: ⭐⭐ 2026-10-13 **新增这一条，理由必须写在这儿**（棘轮规矩 ✓）：
    #:   `perception/tracker.py` 本轮（321 那条）为治"检出框疯狂闪烁 / Agent 丢目标"补了
    #:   **GMC 兜底 + 跨漏帧补桥 + 同 ID 跳变隔离**三件 ✓，加上"每一处都写清为什么"的注释
    #:   （本仓库纪律 ✓），从 ~1400 行涨到 ≥1500 行 ✗。
    #:   ⚠ 它是**单一职责**的（"给检测框稳定的 id" ✓ 模块文档第一句就是它 ✓），没有混两件事 ✓
    #:     ⇒ 按棘轮的口径写进名单 ✓；真要变短该做的是把这些**判据抽成独立小模块**
    #:     （如 `perception/gmc.py` / `bridge.py` ✓）而不是删注释 ✗ —— 那是另一轮的事 ✓。
    "perception/tracker.py",
    "perception/lie_tracker.py",
    "perception/minimap.py",
}


def t_layering_ratchet():
    """E-① **分层与环不许更差**（用 `tools/arch_scan.scan()` 的**同一套判据** ✓）。

    为什么只「冻结」不「全清」✗：`gui.live_thread -> tools` 那三条是**既有**的 ✓（拆它要动
    大结构 ✗ 另排计划 ✓）—— 但**不许再冒新的** ✓；**环必须保持 0** ✓（一有环就说明
    「谁都能 import 谁」⇒ 层的意义就没了 ✗）。
    """
    res = scan(verbose=False)
    new = sorted(set(res["layering"]) - LAYERING_BASELINE)
    check(not new,
          "**新冒出来的分层违反**（低层 import 高层 ✗）：\n      %s\n"
          "     （既有的 %d 条已冻结 ✓ 见 `docs/架构-AI三层.md`；新的必须当场拆掉 ✗）"
          % ("\n      ".join(new), len(LAYERING_BASELINE)))
    check(not res["cycles"],
          "**出现模块级循环依赖**：%s ✗（层这时候已经没有意义了 ✓ ⇒ 要么拆，要么把它写进 "
          "`LAYERING_BASELINE` 并在文档里说明 ✓）" % (res["cycles"],))


def t_giant_files_ratchet():
    """E-② **不许再冒「巨型运行期文件」**（≥1500 行 ✓）。

    为什么钉这条：文件大到一定程度，**结构就是它自己** ✗ —— `decision/agent.py`（近万行 ✓）
    里策略 / 输出序列 / idle / 路由胶水全混着 ✓，改一处得先读半万行 ✗（本次排查就是这么耗的 ✓）。
    这条不要求现在拆 ✓，只要求「别再多一个」✓（每个新巨物都该先问：它是不是又混了两件事 ✗）。
    """
    res = scan(verbose=False)
    runtime = {rel for _n, rel in res["big"]
               if rel.split("/")[0] in RUNTIME_PKGS or "/" not in rel}
    new = sorted(runtime - GIANT_BASELINE)
    check(not new,
          "**新的巨型运行期文件**（≥1500 行 ✗）：\n      %s\n"
          "     （要么拆成两件事 ✓，要么它确实是单一职责 ⇒ 连「为什么」一起写进 "
          "`GIANT_BASELINE` ✓）" % "\n      ".join(new))


TESTS = (
    ("A-① 玩家坐标唯一写者（`LAST_POS`：实时回路写，界面/地形图只读）",
     t_single_writer_coords),
    ("A-② 界面状态两个事实各自唯一写者（原生 `_screen_state` / 接管 `_takeover_st` + "
     "「这一场测谎」的三笔账）", t_single_writer_screen_takeover),
    ("A-③ 重活闸的账唯一写者（`_pipe_fps` / `_heavy_at`）", t_single_writer_heavy_gate),
    ("B 不许用 `setattr` 绕过唯一写者（后门）", t_no_setattr_backdoor),
    ("C 喂给 Agent 的 `ws` 必须在 `agent.tick` 之前装好 + 一帧只喂一次",
     t_pipeline_order_screen_before_tick),
    ("D `_screen_beat` 的每拍职责必须在早退之前（踩过三次的老坑）",
     t_screen_beat_transition_vs_everytick),
    ("E-① 分层违反 / 循环依赖不许更差（冻结基线，环必须 0）", t_layering_ratchet),
    ("E-② 不许再冒巨型运行期文件（≥1500 行，冻结名单）", t_giant_files_ratchet),
)


def t_climb_arrival_yields_to_route_shortcut():
    """⭐⭐⭐⭐ 「到顶再按住 ↑」不许被那条捷径抢掉（用户 2026-10-13 ✓ 原话："**向上攀爬判断
    到顶后还是没有额外按住 ↑「坐标对齐误差时间」**"✓）。

    病根（现场数据说话 ✓ `behavior.log` 13:27~13:46）：`climb_hold_up`（上轮刚加的打点 ✓）
      **35** 条 ✓，而"人已经站在「X」里了 ⇒ 收工"（`route_goal_reached` ✓）**132** 条 ✓
      ⇒ 到顶那一拍**常常先撞上这条捷径** ✗ ⇒ `_task_finished` ⇒ 执行器那段"到顶后再按住 ↑
      `base_hold_ms`"（`route.ClimbJob._arrive_step` ✓）**根本没机会跑** ✓✓。

    判据（**静态** ✓ 不跑实时 ✓）：`CombatAgent._climb_tick` 里那句收工条件必须**同时**带
      ① `not _hold_up` —— 执行器**已经**判到到达那几拍（2026-09-28 那条 ✓）；
      ② `not _climb_going` —— 执行器**正在爬**（`ClimbJob` 的 CLIMB 相 ✓ 本轮加的 ✓）。
      少 ② ⇒ 回到"爬到一半、脚下刚进目标集合就被提前收工" ✓ 而且这条链是**静默**失效 ✗
      （不报错、只是"再按住 ↑"永远看不到 ✓）⇒ 必须钉死 ✓。
    ⚠ 按**行**判 ✗ 不按整份源码的子串判（注释里也写着这两个词 ✓ 见 SKILL 311 那条教训 ✓）。
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "decision" / "agent.py").read_text(
        encoding="utf-8")
    lines = src.splitlines()
    idx = next((i for i, l in enumerate(lines)
                if "goal in here" in l and not l.strip().startswith("#")), None)
    check(idx is not None,
          "找不到「人已经站在目的地里 ⇒ 收工」那条判据了 ✗（源码结构变了 ⇒ 这条钉子要跟着改 ✓）")
    blk = " ".join(lines[idx:idx + 3])
    check("not _hold_up" in blk,
          "收工捷径少了「执行器已判到到达」那道让路闸 ✗（用户 2026-09-28 那条 ✓）：%r"
          % (blk[:140],))
    check("not _climb_going" in blk,
          "收工捷径少了「执行器**正在爬**」那道让路闸 ✗ —— 到顶那一拍会先把任务收掉 ⇒ "
          "「再按住 ↑ 坐标对齐误差时间」永远跑不到 ✓（用户 2026-10-13 ✓）：%r" % (blk[:140],))


TESTS = TESTS + (
    ("⭐⭐⭐⭐ 到顶「再按住 ↑」不许被收工捷径抢掉（`not _hold_up` + `not _climb_going`）",
     t_climb_arrival_yields_to_route_shortcut),
)


def t_climb_hold_allows_attack():
    """⭐⭐⭐⭐ 「到顶再按住 ↑」那几拍**照常输出**（用户 2026-10-13 ✓ 原话："**只按 ↑+朝锁定目标**
    ＋ **照常输出**"✓）。

    口径（与"爬绳途中"同一条 ✓ 用户 2026-09-28："**爬绳途中正常触发 attack，但不会松开 ↑ 键**"✓）：
      · `_on_rope`（"不许进 attack"那道闸 ✓）**必须排除到达保持那几拍** ✗ —— 否则到顶那一刻
        人已出绳的 y 范围（`on_rope` 假 ✓）而 `holds_player` 仍真（按过跳 ✓）⇒ 闸为真 ⇒
        **不打** ✗（用户要的正是"照常输出"✓）；
      · tick 里的 climb 分支**必须让路**（`and not _hold_up_now` ✓）⇒ 状态/输出落回策略链 ✓，
        而 `_climb_tick` 照旧跑（↑ 与"朝锁定目标"那个方向键**一直按着** ✓）。
     ⚠ 只钉这两处 ✗：另外三条判据（`_jumped_once` / `_in_rope` / `_align_phase`）**原样** ✓
       —— "按下跳 → 吸上绳"那一段仍不许打架 ✓（用户 2026-09-27 ✓）。
    ⚠ 若哪天把"到顶额外按住 ↑"整个删掉 ⇒ 这两条钉子也失去对象 ✓ 那时连这条用例一起删 ✓。
    """
    from pathlib import Path
    _ls = (Path(__file__).resolve().parents[1] / "decision" / "agent.py").read_text(
        encoding="utf-8").splitlines()

    def _blk(mark, n=3):
        k = next((i for i, l in enumerate(_ls)
                  if mark in l and not l.strip().startswith("#")), None)
        return None if k is None else " ".join(_ls[k:k + n])

    # ⚠⚠ 判据用「**数出现次数**」这种最笨但最稳的写法 ✗ 别再用"按标记取块" —— 本轮连撞
    #   两次同名（`_on_rope = bool(...)` 与 `not self._climb_tick(` 都各有**第二处** ✓
    #   见 SKILL 311 那条教训 ✓）⇒ 钉子自己先红了两次 ✓。
    _src = "\n".join(_ls)
    _n = _src.count("not _hold_up_now")
    check(_n >= 2,
          "让路那两处缺了 ✗：`_on_rope`（那道不许打怪的闸 ✓）与 tick 的 climb 分支"
          "**各要一处** `not _hold_up_now` ✓，实得 %d 处 ⇒ 到顶那几拍会被挡住/被 climb 接管 ⇒ "
          "**不打怪** ✓（用户 2026-10-13 ✓）" % _n)
    check("_hold_up_now = bool(" in _src and "holding_up()" in _src,
          "`_hold_up_now` 的定义没了 ✗（该读 `ClimbJob.holding_up()` ✓）")


TESTS = TESTS + (
    ("⭐⭐⭐⭐ 到顶「再按住 ↑」那几拍照常输出（`_on_rope` 与 climb 分支都要让路）",
     t_climb_hold_allows_attack),
)


def t_ws_frozen_before_tick():
    """⭐ **快照冻结在喂 Agent 之前**（2026-10-13 阶段 1「冻结」✓）。

    为什么：`ws._freeze()` 之后写 `ws` / `ws.player` 字段会抛异常（「写者唯一」由**类型**
    硬保证 ✓ 见 `perception/world_state.py` 的 `_frozen` ✓）。它必须排在 `agent.tick(ws)`
    **之前** —— 否则喂给 Agent 的还是**没锁**的快照，「构造完不可变」这道保险等于没上 ✗
    （Agent 内部那个反向信号 `want_dot_reacquire` 已经挪回它自己身上 ✓ 见那边 ✓）。
    """
    _src, tree = _lt_tree()
    freezes = [n.lineno for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "_freeze"
               and isinstance(n.func.value, ast.Name) and n.func.value.id == "ws"]
    check(freezes, "找不到 `ws._freeze()`（快照冻结接线断了 ✗）")
    ticks = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "tick" and n.args and isinstance(n.args[0], ast.Name)
             and n.args[0].id == "ws"]
    check(ticks, "找不到 `agent.tick(ws)`")
    check(min(freezes) < ticks[0],
          "`ws._freeze()`（L%d）排在 `agent.tick(ws)`（L%d）**之后** ⇒ 喂给 Agent 的快照没锁 ✗"
          % (min(freezes), ticks[0]))


TESTS = TESTS + (
    ("⭐⭐ 快照冻结 `ws._freeze()` 必须在喂 Agent 之前（写者唯一由类型保证）",
     t_ws_frozen_before_tick),
)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv:
        return _dump()
    bad = 0
    for name, fn in TESTS:
        try:
            fn()
            print("[ OK ] %s" % name)
        except Exception as e:                       # noqa: BLE001
            bad += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
    print("\n%d/%d 通过" % (len(TESTS) - bad, len(TESTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
