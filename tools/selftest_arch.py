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
        "decl": {"perception/world/world_state.py": 1},
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
        "writes": {"gui/live_thread.py": 2},
        "decl": {},
        "why": "**重活闸**的档位（初值 + 配置热读 ✓ 两处都在这一个文件里 ✓）"
               "—— 它是「每秒跑多少拍重活」的**唯一权威** ✓ 别处不许改 ✗。"
               "⚠ 2026-10-17 从 3 处变 2 处：那条「解析失败兜底」不再直接写 `_pipe_fps`，"
               "而是写局部 `_base`（最终仍由热读那一处统一赋值 ✓）⇒ 写点少了一个、"
               "权威点没变 ✓（顺带加了「测谎让路」，也走这唯一一处 ✓）。",
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
    # ⭐ 2026-10-18 **`decision/route.py`（4995 行）已拆** ⇒ 这一条换成拆出来的那个大的：
    #   `decision/jobs/climb.py`（1947 行，其中 `ClimbJob` 一个类 1914 行 ✓）。
    #   理由（棘轮规矩：写清为什么 ✓）：它**是单一职责**的（"定点上绳"这一个执行器 ✓，
    #   文件第一句就是它 ✓），没有混两件事 ⇒ 允许登记；
    #   ⚠ 但它正是下一轮要拆的对象（用户 2026-10-18 的功能规划：绳梯当目的地 +
    #     重做角色操作）⇒ **下轮动它时，拆完记得把这条从名单里删掉** ✗ 别让它变成永久豁免。
    "decision/jobs/climb.py",
    "deploy/app.py",
    "gui/live_panel.py",
    "gui/live_thread.py",
    "gui/minimap_calib.py",
    "gui/player_panel.py",
    "gui/route_panel.py",
    "gui/steps/cards.py",
    "gui/yolo_workbench.py",
    "gui/zone_editor.py",
    "perception/world/anchor_track.py",
    "liedetect/lie_motion.py",
    #: ⭐⭐ 2026-10-13 **新增这一条，理由必须写在这儿**（棘轮规矩 ✓）：
    #:   `perception/track/tracker.py` 本轮（321 那条）为治"检出框疯狂闪烁 / Agent 丢目标"补了
    #:   **GMC 兜底 + 跨漏帧补桥 + 同 ID 跳变隔离**三件 ✓，加上"每一处都写清为什么"的注释
    #:   （本仓库纪律 ✓），从 ~1400 行涨到 ≥1500 行 ✗。
    #:   ⚠ 它是**单一职责**的（"给检测框稳定的 id" ✓ 模块文档第一句就是它 ✓），没有混两件事 ✓
    #:     ⇒ 按棘轮的口径写进名单 ✓；真要变短该做的是把这些**判据抽成独立小模块**
    #:     （如 `perception/gmc.py` / `bridge.py` ✓）而不是删注释 ✗ —— 那是另一轮的事 ✓。
    "perception/track/tracker.py",
    "liedetect/lie_tracker.py",
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
      `base_hold_ms`"（`jobs.ClimbJob._arrive_step` ✓）**根本没机会跑** ✓✓。

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
    硬保证 ✓ 见 `perception/world/world_state.py` 的 `_frozen` ✓）。它必须排在 `agent.tick(ws)`
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


#: 函数内**向上**依赖的**冻结基线**（2026-10-17 实测 ✓ 只许更少 ✗ 不许新增）
#: 为什么这条必须单独有：模块级分层违反已经 0 ✓，但上面这 29 处是**写在函数体里**绕过去的 ✗
#: —— 拆块期间最容易复发的就是这一手（`gui` 里 `from tools.xxx import …` 一行就跨了两层 ✗）。
LAZY_UP_BASELINE = {
    "gui -> tools": 17,
    "deploy -> tools": 10,
    "perception -> tools": 1,
    "perception -> gui": 1,
}

#: 单函数 ≥600 行的**冻结名单**（2026-10-17 实测 12 个 ✓）
#: 为什么用"名单 + 条数"双保险：只比条数 ⇒ 拆了一个又新写一个照样过 ✗；
#: 只比名单 ⇒ 某个函数从 600 涨到 2000 也看不出来 ✗（条数那条挡不了，得靠人看报告 ✓）。
LONG_FN_BASELINE = {
    ("tools/lie_demo.py", "run_window"),
    ("liedetect/lie_motion.py", "process"),
    ("gui/live_thread.py", "_run"),
    ("gui/player_panel.py", "_build"),
    ("tools/lie_demo.py", "__init__"),
    ("decision/agent.py", "tick"),
    ("gui/route_panel.py", "__init__"),
    ("tools/lie_demo.py", "draw"),
    ("liedetect/lie_registry.py", "step"),
    ("liedetect/lie_tracker.py", "_step_track"),
    ("decision/agent.py", "__init__"),
    ("decision/agent.py", "_climb_tick"),
}


def t_lazy_upward_ratchet():
    """E-③ 函数内**向上**依赖不许新增（拆块期间的护栏 ✓）。

    模块级分层违反已经 0 ✓，但实测还有 **29 处**藏在函数体里：
        `gui -> tools` 17 · `deploy -> tools` 10 · `perception -> tools` 1 · `perception -> gui` 1
    这些"改一行 import 就跨两层"的做法，正是分层判据失效的地方 ✗ ⇒ 冻结、只许更少 ✓。
    """
    up = scan(verbose=False)["lazy_up"]
    for k, c in up.items():
        base = LAZY_UP_BASELINE.get(k, 0)
        check(c <= base,
              "函数内向上依赖 `%s` 变成 %d 处（基线 %d）⇒ 边界被绕过 ✗ "
              "（要就明着写在模块级、然后按分层规矩办 ✓）" % (k, c, base))
    for k in LAZY_UP_BASELINE:
        check(k in up, "基线里的 `%s` 不见了 ✓（好的方向 ✓ 请顺手把基线这条删掉）" % k)


def t_long_fn_ratchet():
    """E-④ 单函数 ≥600 行不许新增（拆块期间的"别再造一个 2000 行函数"✓）。

    实测 12 个（最狠 `tools/lie_demo.run_window` 3904 行、`liedetect/lie_motion.process` 2511 行、
    `gui/live_thread._run` 1868 行、`decision/agent.tick` 1160 行）—— 拆块的目标就是让这张表变短 ✓，
    但**拆的过程里最容易写出来的**恰好是又一个超长函数 ✗ ⇒ 冻结名单 + 只许更少 ✓。
    """
    fn = scan(verbose=False)["long_fn"]
    got = {(path, name) for _ln, path, name in fn}
    new = sorted(got - LONG_FN_BASELINE)
    check(not new,
          "新长出 ≥600 行的函数 %s ⇒ 拆块不许再造巨型函数 ✗（拆成几个 <300 行的 ✓）"
          % (new,))
    check(len(got) <= len(LONG_FN_BASELINE),
          "≥600 行函数 %d 个（基线 %d）⇒ 只许更少 ✗" % (len(got), len(LONG_FN_BASELINE)))


#: `perception/` 的分组（2026-10-17 第 2 波 ✓）：组名 → 组内模块
PERCEPTION_GROUPS = {
    "detect": ("predict", "classes", "visible_box"),
    "track": ("tracker", "bytetrack", "assign"),
    "world": ("world_state", "pos_state", "player_state", "player_locator",
              "anchor_track", "region"),
    "screen": ("screen_state", "ui_state"),
}


def t_perception_split():
    """⭐⭐⭐ `perception/` 四个子包不许再黏回顶层（2026-10-17 第 2 波 ✓）。

    拆分前：`perception/` 23 个文件平铺 —— "检测 / 跟踪 / 世界 / 界面" 四件事混在一层 ✗
    ⇒ 想找"框是谁给的"要翻 23 个文件名 ✗。

    钉五条：
      ① 四个子包 + `__init__.py` 都在（防止"拆了又合"✗）；
      ② 组内模块**不许**再出现在 `perception/` 顶层（两处同名 = 两条路 ✗）；
      ③ `perception/__init__.py` **保持空壳**（不许 re-export ⇒ 否则旧路径永远活着 ✗）；
      ④ 组内模块之间**不许**绕回去 import 顶层旧路径（`from perception import tracker` ✗）；
      ⑤ 依赖方向：`track/world/detect/screen` 之间只许 → `perception.{geom,kf}` 这种同层通用件 ✓
         （`perception.screen` **不许** import `track`/`world` ✗ —— 界面状态机不该认识怪物追踪 ✓）。
    """
    perc = ROOT / "perception"
    for g, mods in PERCEPTION_GROUPS.items():
        d = perc / g
        check(d.is_dir(), "`perception/%s/` 不见了（拆出去的组被合回去了 ✗）" % g)
        check((d / "__init__.py").is_file(), "`perception/%s/__init__.py` 没了 ✗" % g)
        for m in mods:
            check((d / ("%s.py" % m)).is_file(),
                  "`perception/%s/%s.py` 不见了 ✗" % (g, m))
            check(not (perc / ("%s.py" % m)).exists(),
                  "`perception/%s.py` 又出现在顶层 ⇒ 同一模块两条路径 ✗" % m)

    # 只看"有没有 import"：docstring / 注释随便写 ✓ re-export 一句都不许 ✗
    init_mod = ast.parse((perc / "__init__.py").read_text(encoding="utf-8"))
    init_imp = [n for n in init_mod.body if isinstance(n, (ast.ImportFrom, ast.Import))]
    check(not init_imp,
          "`perception/__init__.py` 开始 import/re-export 了（%d 处）⇒ "
          "旧路径会一直活着，拆了等于没拆 ✗" % len(init_imp))

    for g, mods in PERCEPTION_GROUPS.items():
        for m in mods:
            src = (perc / g / ("%s.py" % m)).read_text(encoding="utf-8")
            for other in ("predict", "classes", "visible_box", "tracker", "bytetrack",
                          "assign", "world_state", "pos_state", "player_state",
                          "player_locator", "anchor_track", "region", "screen_state",
                          "ui_state"):
                check("from perception import %s" % other not in src,
                      "`perception/%s/%s.py` 还在用旧路径 `from perception import %s` ✗"
                      % (g, m, other))

    # 界面那组不许认识追踪/世界（依赖方向 ✓）
    for m in PERCEPTION_GROUPS["screen"]:
        src = (perc / "screen" / ("%s.py" % m)).read_text(encoding="utf-8")
        for bad in ("perception.track", "perception.world"):
            check(bad not in src,
                  "`perception/screen/%s.py` import 了 `%s` ⇒ 界面状态机反向依赖追踪/世界 ✗"
                  % (m, bad))

    # 收流客户端独立成 `perception/map_client.py`（2026-10-18 ✓）
    mc, mm = perc / "map_client.py", perc / "minimap.py"
    check(mc.is_file(), "`perception/map_client.py` 不见了（收流客户端被塞回去 ✗）")
    mcs = mc.read_text(encoding="utf-8")
    check("perception.minimap" not in mcs,
          "`map_client.py` 反向 import 了 `minimap` ⇒ 成环 ✗（方向只能是 minimap → map_client ✓）")
    check("class MiniMapClient" not in mm.read_text(encoding="utf-8"),
          "`minimap.py` 里又出现 `class MiniMapClient` ⇒ 同一件事两份实现 ✗")

    # 数学通用件下沉 `core/`（2026-10-18 ✓）
    for m in ("geom", "kf"):
        check((ROOT / "core" / ("%s.py" % m)).is_file(),
              "`core/%s.py` 不见了（数学通用件又被塞回 perception ✗）" % m)
        check(not (perc / ("%s.py" % m)).exists(),
              "`perception/%s.py` 又出现了 ⇒ 同一件尺子两个位置 ✗" % m)


def _map_const_names():
    """`perception/map_const.py` 里定义的所有常量名（**判据从文件本身推导** ✓）。

    ⚠ 别在这里再抄一份名单 ✗：本波还要继续往里搬（`map_dot` / `map_calib` 那两步 ✓）——
      抄一份就等于"搬了新的、判据还盯着旧的" ⇒ 判据空转 ✗。
    """
    t = ast.parse((ROOT / "perception" / "map_const.py").read_text(encoding="utf-8"))
    out = []
    for n in t.body:
        if isinstance(n, ast.Assign):
            out += [tg.id for tg in n.targets if isinstance(tg, ast.Name)]
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.append(n.target.id)
    return out


def _dotted(node):
    """`perception.minimap` / `mm` 这种（属性链 → 点号串 ✓；还原不出来给 None）。"""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def t_map_const_split():
    """⭐⭐⭐ 小地图的**常量**跟它的实现分家（2026-10-18 第 3 波第 1 步 ✓）。

    人话：界面只想问"`crop` 这个模式叫啥"、工具只想拿一条"差分门槛"，
    以前都得 `import perception.minimap`（5000 行的定位实现 ✓）—— 于是
    "拿个字符串"顺手把收流/模板匹配/差分全拖进进程 ✗；而**同一根线两个数**也正是
    从这种顺手复制里长出来的（`DIFF_FLOOR_WARN` 当年就复制过一份 ✓）。

    ⇒ 常量收进 `perception/map_const.py`，钉四条：
      ① 这个文件**一句 import 都没有**（"拿常量"不许牵扯任何依赖 ✓，也不可能成环 ✓）；
      ② `minimap.py` 里**不许再定义**这些名字（再写一份 = 两个数 ✗）；
      ③ 别处**不许再从 minimap 模块上摸**它们（`mm.MODE_CROP` / `from perception.minimap
         import MODE_CROP` ✗ —— 那样等于没拆 ✓）；
      ④ `minimap.py` 必须真的从 `map_const` 拿（否则它自己会 NameError ✓ 这条是给
         "只搬了常量、忘了改引用"留的哨兵 ✓）。
    """
    mc = ROOT / "perception" / "map_const.py"
    check(mc.is_file(), "`perception/map_const.py` 不见了（常量被塞回 minimap 了 ✗）")
    tree = ast.parse(mc.read_text(encoding="utf-8"))
    imps = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    check(not imps,
          "`map_const.py` 里出现了 %d 处 import ⇒ 「拿一个常量」又会被拖去别的依赖 / 可能成环 ✗"
          % len(imps))
    names = set(_map_const_names())
    check(len(names) >= 20,
          "`map_const.py` 只剩 %d 个常量 ⇒ 判据空转了（文件被清空/搬到别处？✗）" % len(names))

    mm = ROOT / "perception" / "minimap.py"
    msrc = mm.read_text(encoding="utf-8")
    mtree = ast.parse(msrc)
    dup = []
    for n in mtree.body:
        targets = []
        if isinstance(n, ast.Assign):
            targets = [tg for tg in n.targets if isinstance(tg, ast.Name)]
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            targets = [n.target]
        for tg in targets:
            if tg.id in names:
                dup.append("%s  L%d" % (tg.id, n.lineno))
    check(not dup,
          "`minimap.py` 里又定义了 map_const 的常量（两处各写一个数迟早分叉 ✗）：\n     "
          + "\n     ".join(dup))
    check("from perception import map_const" in msrc,
          "`minimap.py` 没从 `perception.map_const` 拿常量 ⇒ 它自己会 NameError ✗")

    bad = []
    for p in Path(ROOT).rglob("*.py"):
        if skip(p):
            continue
        try:
            t = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        aliases = set()
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name == "perception.minimap":
                        aliases.add(a.asname or "perception")
            elif isinstance(n, ast.ImportFrom) and n.module == "perception":
                for a in n.names:
                    if a.name == "minimap":
                        aliases.add(a.asname or a.name)
            elif isinstance(n, ast.ImportFrom) and n.module == "perception.minimap":
                for a in n.names:
                    if a.name in names:
                        bad.append("%s:%d  from perception.minimap import %s"
                                   % (p.relative_to(ROOT), n.lineno, a.name))
        for n in ast.walk(t):
            if not isinstance(n, ast.Attribute) or n.attr not in names:
                continue
            base = _dotted(n.value)
            if base == "perception.minimap" or (base and base in aliases):
                bad.append("%s:%d  %s.%s" % (p.relative_to(ROOT), n.lineno, base, n.attr))
    check(not bad,
          "这些地方还在从**小地图实现**上摸常量（该 `from perception import map_const` ✗）：\n     "
          + "\n     ".join(sorted(set(bad))[:8]))


#: 第 3 波第 2 步搬走的函数（`perception.minimap` → `perception.map_calib` ✓ 2026-10-18）
MAP_CALIB_FNS = (
    "panel_of", "implied_panel", "panel_mismatch", "crop_rel_of", "crop_from_rel",
    "offset_of", "world_offset", "scales_of", "with_scales", "has_geometry",
    "panel_to_canvas", "panel_to_world", "world_to_panel",
    "terrain_mask", "canvas_to_panel_matrix", "warp_canvas_to_panel",
    "calib_overlap", "overlap_verdict",
    "solve_two_point", "calib_from_two_point", "drop_stale_world_offset",
    "world_offset_advice",
)

#: `map_calib` 只许用这四个依赖（单向、纯算术 ✓）
MAP_CALIB_DEPS = ("cv2", "numpy", "core.mapdata", "perception.map_const")

#: 第 3 波第 3 步搬走的东西（`perception.minimap` → `map_dot` / `map_locator` ✓ 2026-10-18）
MAP_DOT_NAMES = (
    "dot_feet", "track_params", "tracker_kwargs", "dot_core_mask", "_blob_looks_core",
    "dot_color", "dot_mask", "dot_side_max", "dot_player_like", "_dot_candidates",
    "roi_around", "_is_flood", "_calib_roi", "_apply_roi", "_basemap_extra_mask",
    "dot_candidates_diff", "_drop_on_canvas_marks", "_prefer_on_foothold",
    "find_player_dot", "_shift_calib", "_local_result_trustworthy",
    "DOT_YELLOW", "DOT_CYAN", "DOT_FAMILIES", "DOT_V_MIN", "DOT_SIDE_MIN",
    "DOT_SIDE_MAX_DIV", "DOT_FILL_MIN", "DOT_ASPECT_MAX", "DOT_INSET",
    "DOT_MAX_CANDS", "DOT_FLOOD_RATIO", "DOT_FLOOD_PX", "DOT_MAX_JUMP_PX",
    "DOT_CONFIRM_FRAMES", "DOT_HOLD_MS", "DOT_ROI_PAD", "DOT_GHOST_DECAY",
    "DOT_GHOST_MAX_SHIFT", "DOT_PLAYER_SIDE_REL", "DOT_EXP_MIN_SAMPLES",
    "DOT_EXP_HISTORY", "DOT_LIKE_MIN", "DOT_REACH_REL", "DOT_DIST_W",
    "DOT_PLAYER_SIDE_HARD_REL", "DOT_SIDE_HARD_ABS", "DOT_SIDE_HARD_ABS_MIN",
    "DOT_SIDE_HARD_ABS_REL", "DOT_SIDE_HARD_ABS_MAX", "DOT_LOCK_MARGIN",
    "DOT_SMOOTH_ALPHA", "PLAYER_FB_EMA_ALPHA", "TRACK_KEYS",
    "DOT_CORE_G_MIN", "DOT_CORE_R_MIN", "DOT_CORE_B_MAX", "DOT_CORE_MIN_SIDE",
    "DOT_JUMP_GIVEUP_N", "DOT_JUMP_WINDOW_S",
)
MAP_LOCATOR_NAMES = ("PlayerDotTracker", "PlayerLocator")

#: 第 3 波第 4 步搬走的（`perception.minimap` → `map_overlay` / `tools.mmap_diag` ✓ 2026-10-18）
MAP_OVERLAY_NAMES = (
    "_OVERLAY_W", "overlay_zoom", "view_rects", "overlay_draw_rects",
    "diff_panel_vs_canvas", "frame_overlay_rects", "align_score", "_near_best",
    "crop_compare", "draw_on_overlay",
)
MAP_CLI_NAMES = ("diagnose", "watch", "_desc", "_mode_zh")


def t_map_calib_split():
    """⭐⭐⭐ 「这个像素是世界的哪一点」跟定位实现分家（2026-10-18 第 3 波第 2 步 ✓）。

    人话：标定窗/双点标定/叠图核对/工具脚本都只想问"**面板上这点是世界的哪里**"，
    以前一律 `import perception.minimap`（定位实现 4000+ 行 ✓）—— 于是"换算一个坐标"
    顺手把收流线程、模板匹配、差分全拖进进程 ✗。

    ⇒ 换算与标定几何收进 `perception/map_calib.py`，钉五条：
      ① `map_calib` 只依赖 `cv2/numpy/core.mapdata/perception.map_const`（单向 ✓）；
      ② 它**不许** import `minimap` / `map_dot`（否则依赖成环 + 拖进程 ✗）；
      ③ 那 22 个函数**只在 map_calib 里定义**（minimap 里再写一份 = 两个口径 ✗）；
      ④ `minimap.py` 必须真的从 `map_calib` 拿（否则它自己 NameError —— 哨兵 ✓）；
      ⑤ 别处**不许再从 minimap 模块上摸**它们（`mm.panel_to_world` ✗）。
    """
    mc = ROOT / "perception" / "map_calib.py"
    check(mc.is_file(), "`perception/map_calib.py` 不见了（换算被塞回 minimap 了 ✗）")
    ctree = ast.parse(mc.read_text(encoding="utf-8"))

    deps, bad_dep = set(), []
    for n in ast.walk(ctree):
        if isinstance(n, ast.Import):
            for a in n.names:
                deps.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if mod in ("core", "perception"):      # from core import mapdata ✓
                for a in n.names:
                    deps.add(mod + "." + a.name)
            else:
                deps.add(mod)
    for d in sorted(deps):
        if d not in MAP_CALIB_DEPS:
            bad_dep.append(d)
    check(not bad_dep,
          "`map_calib.py` 引入了不该有的依赖 %s ⇒ 它只许 %s（单向、纯算术 ✗）"
          % (bad_dep, list(MAP_CALIB_DEPS)))
    check(not [d for d in deps if d in ("perception.minimap", "perception.map_dot",
                                        "perception.minimap.minimap")],
          "`map_calib.py` import 了 minimap/map_dot ⇒ 依赖成环、还会把定位实现拖起来 ✗")

    defined = {n.name for n in ctree.body if isinstance(n, ast.FunctionDef)}
    miss = [f for f in MAP_CALIB_FNS if f not in defined]
    check(not miss, "`map_calib.py` 少了这些函数（搬漏了 / 改名了 ✗）：%s" % miss)

    mm = ROOT / "perception" / "minimap.py"
    msrc = mm.read_text(encoding="utf-8")
    mtree = ast.parse(msrc)
    dup = ["%s  L%d" % (n.name, n.lineno) for n in mtree.body
           if isinstance(n, ast.FunctionDef) and n.name in MAP_CALIB_FNS]
    check(not dup,
          "`minimap.py` 里又定义了搬走的函数（两处各写一个口径迟早分叉 ✗）：\n     "
          + "\n     ".join(dup))
    check("from perception import map_calib" in msrc,
          "`minimap.py` 没从 `perception.map_calib` 拿换算 ⇒ 它自己会 NameError ✗")

    bad = []
    for p in Path(ROOT).rglob("*.py"):
        if skip(p):
            continue
        try:
            t = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        aliases = set()
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name == "perception.minimap":
                        aliases.add(a.asname or "perception")
            elif isinstance(n, ast.ImportFrom) and n.module == "perception":
                for a in n.names:
                    if a.name == "minimap":
                        aliases.add(a.asname or a.name)
            elif isinstance(n, ast.ImportFrom) and n.module == "perception.minimap":
                for a in n.names:
                    if a.name in MAP_CALIB_FNS:
                        bad.append("%s:%d  from perception.minimap import %s"
                                   % (p.relative_to(ROOT), n.lineno, a.name))
        for n in ast.walk(t):
            if not isinstance(n, ast.Attribute) or n.attr not in MAP_CALIB_FNS:
                continue
            base = _dotted(n.value)
            if base == "perception.minimap" or (base and base in aliases):
                bad.append("%s:%d  %s.%s" % (p.relative_to(ROOT), n.lineno, base, n.attr))
    check(not bad,
          "这些地方还在从**定位实现**上摸换算（该 `from perception import map_calib` ✗）：\n     "
          + "\n     ".join(sorted(set(bad))[:8]))


#: 拆包/搬文件时**新引入的别名**（谁用谁必须在自己作用域里 import ✓ 2026-10-18）
ALIAS_IMPORTS = {
    "mouse_aim_c": "from execution import aim_convert as mouse_aim_c",
    "ma_c": "from execution import aim_convert as ma_c",
    "_remote_mod": "from execution import remote as _remote_mod",
    "_link": "from execution import remote as _link",
    "conv": "from execution import aim_convert as conv",
    "dptr": "from execution import pointer as dptr",
    "mac": "from execution import aim_convert as mac",
}


def t_refactor_aliases_imported():
    """⭐⭐⭐⭐ 搬文件引进的**别名**，谁用谁必须在**自己那个作用域**里 import。

    为什么单列这一条（2026-10-18 用户现场 ✓）：`gui/live_thread.py` 是**一律局部导入**
    （模块级名会被静态自检抓 ✗），拆 `aim_convert` 时把 `mouse_aim.xxx` 机械改成了
    `mouse_aim_c.xxx` ✓ 但**只补了一部分作用域**的 import ✗ ⇒ 现场点「测谎确定」直接
    `NameError: name 'mouse_aim_c' is not defined` ✗（编译期完全看不出来 ✓ 只有走到那行才炸 ✓）。

    判据：全仓扫一遍 —— 每个用到别名的函数，它的**自身或模块级**必须有那句 import。
    （`ast.walk(fn)` 会把内层函数的 import 也算上 ✓ 那是宽松方向、可接受 ✓）
    """
    bad = []
    for p in Path(ROOT).rglob("*.py"):
        if skip(p):
            continue
        try:
            s = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not any(a in s for a in ALIAS_IMPORTS):
            continue
        try:
            t = ast.parse(s)
        except SyntaxError:
            continue
        mod_ok = set()
        for n in t.body:
            if isinstance(n, ast.ImportFrom) and n.module == "execution":
                for a in n.names:
                    if a.asname:
                        mod_ok.add(a.asname)
        for fn in [x for x in ast.walk(t)
                   if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            used = {c.id for c in ast.walk(fn)
                    if isinstance(c, ast.Name) and c.id in ALIAS_IMPORTS}
            if not used:
                continue
            local = set()
            for n in ast.walk(fn):
                if isinstance(n, ast.ImportFrom) and n.module == "execution":
                    for a in n.names:
                        if a.asname:
                            local.add(a.asname)
            # ⚠ 同名**局部变量**不算（如 `conv = ROOT / "aim_convert.py"` 那条路径常量 ✓）
            shadow = {c.id for c in ast.walk(fn)
                      if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store)}
            miss = used - local - mod_ok - shadow
            if miss:
                bad.append("%s:%d %s() 缺 %s"
                           % (p.relative_to(ROOT), fn.lineno, fn.name, sorted(miss)))
    check(not bad,
          "这些地方用了拆包后的别名却没 import（走到就 NameError ✗）：\n     "
          + "\n     ".join(bad[:6]))


#: 拆过家之后**旧模块上已经没有了**的名字（2026-10-18 ✓ 现场 `_ma.load_gain` 那条）
MOVED_ATTRS = {
    "execution.mouse_aim": (
        "load_gain", "save_gain", "gain_sizes", "gain_ref_frame", "gain_frame_key",
        "gain_mismatch", "gain_mismatch_block", "gain_record", "gain_warn",
        "auto_measure", "auto_measure_async", "auto_measure_enabled",
        "set_auto_measure", "calib_status", "note_frame_shape", "last_frame_shape",
        "screen_gain", "target_px", "roi_px", "frame_rect", "to_screen",
        "counts_for", "aim_available", "corner_zero", "GAIN_PATH", "GAIN_V2",
    ),
    "execution.input": (
        "link_health", "connect_async", "reconnect_remote", "release_all_remote",
        "net_allowed", "use_network", "use_serial", "use_local",
        "cursor_probe", "window_probe", "last_connect_err", "shutdown",
    ),
    "perception.minimap": (("MiniMapClient", "stop_all_clients") + MAP_CALIB_FNS
                           + MAP_DOT_NAMES + MAP_LOCATOR_NAMES
                           + MAP_OVERLAY_NAMES + MAP_CLI_NAMES),
}


def t_moved_attrs_called_on_new_module():
    """⭐⭐⭐⭐⭐ 拆出去的名字**不许再从旧模块上摸**（`_ma.load_gain` 这类 ✗）。

    为什么必须单独钉（2026-10-18 用户现场 ✓）：拆 `aim_convert` 时把**函数体内**的
    `import execution.mouse_aim as _ma` 一并留下了 ✓ 而 `_ma.load_gain(...)` 没人改 ✗
    ⇒ 编译绿、自检绿（自检不走那条）✗ ⇒ **真跑起来点一下就炸**：
        `AttributeError: module 'execution.mouse_aim' has no attribute 'load_gain'`
    这类错"看得见却查不着"，只能靠全仓扫属性访问 ✓。

    判据：凡是从旧模块导入的**别名**（`_ma` / `ma` / `mouse_aim` / `dinput` …），
    其后**不许**出现 `.已被搬走的名字` ✓。
    """
    bad = []
    for p in Path(ROOT).rglob("*.py"):
        if skip(p):
            continue
        try:
            s = p.read_text(encoding="utf-8", errors="ignore")
            t = ast.parse(s)
        except (OSError, SyntaxError):
            continue
        aliases = {}
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module in MOVED_ATTRS:
                for a in n.names:
                    aliases[a.asname or a.name] = n.module
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name in MOVED_ATTRS:
                        aliases[a.asname or a.name.split(".")[0]] = a.name
        if not aliases:
            continue
        for n in ast.walk(t):
            if not isinstance(n, ast.Attribute) or not isinstance(n.value, ast.Name):
                continue
            mod = aliases.get(n.value.id)
            if mod and n.attr in MOVED_ATTRS[mod]:
                bad.append("%s:%d  %s.%s（它已经不在 %s 里了 ✗）"
                           % (p.relative_to(ROOT), n.lineno, n.value.id, n.attr, mod))
    check(not bad,
          "这些地方还在从**旧模块**上摸搬走的名字（运行到就 AttributeError ✗）：\n     "
          + "\n     ".join(bad[:8]))


def t_map_dot_split():
    """⭐⭐⭐ 「认哪个点是玩家」与「玩家现在在哪」跟定位实现分家（2026-10-18 第 3 波第 3 步）。

    人话：以前想让"这一帧玩家那个黄点在面板哪儿"或者"他现在站哪块平台"，都得
    `import perception.minimap`（4000+ 行，收流线程 / 模板匹配 / 叠图诊断全拖起来）。
    ⇒ 拆成两层：
      · `perception/map_dot.py`：**单帧**认点（颜色族 / 候选 / 挑点 / 颜色复核）；
      · `perception/map_locator.py`：**跨帧**定位（`PlayerDotTracker` 平滑防跳、
        `PlayerLocator` 换算世界坐标 + 各种兜底）。

    钉五条：
      ① 两个文件在，且搬走的名字**只在**各自模块里定义（minimap 再写一份 = 两个口径 ✗）；
      ② `minimap` **不许** import 它们（否则成环；它只该被它们 import）；
      ③ `map_dot` **不许** import `map_locator`（认点不该知道定位 ✓ 单向）；
      ④ `MOVED_ATTRS` 必须覆盖这批名字（这样别处 `mm.find_player_dot` 才有人管 ✗）；
      ⑤ 两个新文件都不许再长回 ≥1500 行（那是"又混了两件事"的信号 ✓）。
    """
    dot = ROOT / "perception" / "map_dot.py"
    loc = ROOT / "perception" / "map_locator.py"
    check(dot.is_file() and loc.is_file(),
          "`perception/map_dot.py` / `map_locator.py` 不见了（又被塞回 minimap 了 ✗）")

    mm = ROOT / "perception" / "minimap.py"
    msrc = mm.read_text(encoding="utf-8")
    mtree = ast.parse(msrc)
    gone = set(MAP_DOT_NAMES) | set(MAP_LOCATOR_NAMES)
    dup = ["%s  L%d" % (n.name, n.lineno) for n in mtree.body
           if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in gone]
    dup += ["%s  L%d" % (tg.id, n.lineno) for n in mtree.body
            if isinstance(n, ast.Assign) for tg in n.targets
            if isinstance(tg, ast.Name) and tg.id in gone]
    check(not dup,
          "`minimap.py` 里又定义了搬走的名字（两处各一份口径迟早分叉 ✗）：\n     "
          + "\n     ".join(dup))

    dtree = ast.parse(dot.read_text(encoding="utf-8"))
    dmods = set()
    for n in ast.walk(dtree):
        if isinstance(n, ast.ImportFrom):
            dmods.add(n.module or "")
        elif isinstance(n, ast.Import):
            for a in n.names:
                dmods.add(a.name)
    check("perception.map_locator" not in dmods,
          "`map_dot.py` import 了 `map_locator` ⇒ 认点反向依赖定位（单向坏掉了 ✗）")

    # ⚠ 必须按 AST 判：`from perception import map_dot` 里并没有 "perception.map_dot"
    #   这个字面串（2026-10-18 反向验证踩到：文本判据放过了一条真反向 import ✗）。
    bad_imp = []
    for n in ast.walk(mtree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in ("perception.map_dot", "perception.map_locator"):
                    bad_imp.append("L%d import %s" % (n.lineno, a.name))
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if mod in ("perception.map_dot", "perception.map_locator"):
                bad_imp.append("L%d from %s" % (n.lineno, mod))
            elif mod == "perception":
                for a in n.names:
                    if a.name in ("map_dot", "map_locator"):
                        bad_imp.append("L%d from perception import %s" % (n.lineno, a.name))
    check(not bad_imp,
          "`minimap.py` import 了 map_dot / map_locator ⇒ 依赖成环 ✗：%s" % bad_imp)

    covered = set(MOVED_ATTRS["perception.minimap"])
    miss = sorted(gone - covered)
    check(not miss,
          "`MOVED_ATTRS['perception.minimap']` 少了这批名字 %s ⇒ 别处再从 minimap 摸"
          "就没人管了（`AttributeError` 只会在跑起来那一刻炸 ✗）" % miss[:8])

    for p, nm in ((dot, "map_dot.py"), (loc, "map_locator.py")):
        n = len(p.read_text(encoding="utf-8").splitlines())
        check(n < 1500, "%s 长到 %d 行 ⇒ 又混了两件事（该再拆一刀 ✓）" % (nm, n))


def t_map_overlay_split():
    """⭐⭐⭐ 「叠图核对」与「诊断命令行」也从 minimap 搬走（2026-10-18 第 3 波第 4 步）。

    人话：`minimap.py` 从 4264 一路瘦到 ~1680，现在只留"量"（定位 / 标定调用 / 来源）。
    搬走的两截：
      · `perception/map_overlay.py`：把"面板显示底图哪一块"**画出来 / 量出来**（叠图、
        差异、相似度打分）—— 工作台标定窗与「叠图核对」在用；
      · `tools/mmap_diag.py`：命令行诊断（`python -m tools.mmap_diag --map <id>`）。

    钉五条：
      ① 两个文件在，且搬走的名字**只在**各自模块定义（minimap 再写一份 = 两处口径 ✗）；
      ② `minimap` 不许 import 它们（`map_overlay` 反过来要 import minimap 借 `_gray` /
         `overlap_pearson` ⇒ 反向立刻成环 ✗）；
      ③ **`gui/` 不许 import `tools.mmap_diag`**（分层：工具层在界面层之后，界面只能把
         命令**当提示文字**显示给用户 ✓ —— 这是当初不肯把训练工具直接丢进 tools/ 的同一条理由）；
      ④ `MOVED_ATTRS["perception.minimap"]` 必须覆盖这批名字（别处再摸就没人管 ✗）；
      ⑤ 两个新文件都不许 ≥1500 行（那是"又混了两件事"的信号 ✓）。
    """
    ov = ROOT / "perception" / "map_overlay.py"
    cli = ROOT / "tools" / "mmap_diag.py"
    check(ov.is_file() and cli.is_file(),
          "`perception/map_overlay.py` / `tools/mmap_diag.py` 不见了（又被塞回 minimap ✗）")

    mm = ROOT / "perception" / "minimap.py"
    msrc = mm.read_text(encoding="utf-8")
    mtree = ast.parse(msrc)
    gone = set(MAP_OVERLAY_NAMES) | set(MAP_CLI_NAMES)
    dup = ["%s  L%d" % (n.name, n.lineno) for n in mtree.body
           if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in gone]
    dup += ["%s  L%d" % (tg.id, n.lineno) for n in mtree.body
            if isinstance(n, ast.Assign) for tg in n.targets
            if isinstance(tg, ast.Name) and tg.id in gone]
    check(not dup,
          "`minimap.py` 里又定义了搬走的名字（两处各一份口径迟早分叉 ✗）：\n     "
          + "\n     ".join(dup))

    bad_imp = []
    for n in ast.walk(mtree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in ("perception.map_overlay", "tools.mmap_diag"):
                    bad_imp.append("L%d import %s" % (n.lineno, a.name))
        elif isinstance(n, ast.ImportFrom):
            m = n.module or ""
            if m in ("perception.map_overlay", "tools.mmap_diag"):
                bad_imp.append("L%d from %s" % (n.lineno, m))
            elif m == "perception":
                for a in n.names:
                    if a.name == "map_overlay":
                        bad_imp.append("L%d from perception import map_overlay" % n.lineno)
    check(not bad_imp,
          "`minimap.py` import 了 map_overlay / mmap_diag ⇒ 依赖成环 ✗：%s" % bad_imp)

    # ⚠ 只盯**新搬进去的这个**：`gui→tools` 另有 17 处历史遗留，由
    #   `t_lazy_upward_ratchet` 冻结着（只许减不许增）；这里不重复管、也不许它涨。
    gui_cli = []
    for p in (ROOT / "gui").rglob("*.py"):
        if skip(p):
            continue
        try:
            gtree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for n in ast.walk(gtree):
            if isinstance(n, ast.ImportFrom) and n.module == "tools.mmap_diag":
                gui_cli.append("%s L%d" % (p.name, n.lineno))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name == "tools.mmap_diag":
                        gui_cli.append("%s L%d" % (p.name, n.lineno))
    check(not gui_cli,
          "`gui/` import 了 `tools.mmap_diag`（界面只该把这条命令当**提示文字** ✓）：%s"
          % gui_cli[:4])

    covered = set(MOVED_ATTRS["perception.minimap"])
    miss = sorted(gone - covered)
    check(not miss,
          "`MOVED_ATTRS['perception.minimap']` 少了这批名字 %s ⇒ 别处再从 minimap 摸"
          "就没人管了（`AttributeError` 只在跑起来那一刻炸 ✗）" % miss[:8])

    for p, nm in ((ov, "map_overlay.py"), (cli, "mmap_diag.py")):
        n = len(p.read_text(encoding="utf-8").splitlines())
        check(n < 1500, "%s 长到 %d 行 ⇒ 又混了两件事（该再拆一刀 ✓）" % (nm, n))


#: 执行器（jobs）与规划的固定成员（2026-10-18 从 `decision/route.py` 拆出）
ROUTE_JOB_FILES = ("params.py", "base.py", "align.py", "climb.py", "drop.py",
                   "walk.py", "portal.py", "jump.py")
ROUTE_PATHING_FILES = ("spans.py", "cost.py", "plan.py")
ROUTE_JOB_CLASSES = {
    "ClimbJob": "climb.py", "DropJob": "drop.py", "WalkJob": "walk.py",
    "PortalJob": "portal.py", "JumpJob": "jump.py",
}


def t_route_split():
    """⭐⭐⭐ 寻路拆成「执行器 jobs」＋「规划 pathing」（2026-10-18，用户拍板"先做 A"）。

    人话：以前 `decision/route.py`（4995 行）里，**五个动作执行器**（爬 1914 / 下跳 942 /
    传送门 482 / 走 311 / 跳 214）和**规划**（代价 → 选边 → 造任务）挤在一个文件里 ——
    改「绳梯能不能当目的地」得先读五千行。

    ⇒ 现在的分工（也是下一轮功能的落点）：
      · `decision/jobs/*`：**怎么走**（每个执行器只回答"往哪挪 / 按不按 / 到没到"）；
      · `decision/pathing/*`：**走哪条**（世界坐标里的代价与选边，`plan_jobs` 一处实现）。

    钉五条：
      ① `decision/route.py` **不许回来**（一回来结构就退化 ✓）；
      ② 五个执行器类**各只在对应文件定义一次**（两处各一份 = 迟早分叉 ✗）；
      ③ **依赖方向单向**：`pathing → jobs` ✓；`jobs` **不许** import `pathing`（成环 ✗）；
      ④ `pathing.plan` 必须真的从 `jobs` 拿执行器（哨兵：不然它自己建不出任务 ✓）；
      ⑤ 全仓**不许**再 `import decision.route`（旧路径已删 ✓ 谁写谁炸）。
    """
    check(not (ROOT / "decision" / "route.py").exists(),
          "`decision/route.py` 又回来了 ⇒ 又变成「什么都装」（该拆 ✓）")

    jd = ROOT / "decision" / "jobs"
    pd = ROOT / "decision" / "pathing"
    miss = [f for f in ROUTE_JOB_FILES if not (jd / f).is_file()]
    miss += ["pathing/" + f for f in ROUTE_PATHING_FILES if not (pd / f).is_file()]
    check(not miss, "`decision/jobs/` 或 `decision/pathing/` 少了这些文件：%s" % miss)

    # ② 一处定义
    seen = {}
    for p in sorted(jd.glob("*.py")):
        for n in ast.parse(p.read_text(encoding="utf-8")).body:
            if isinstance(n, ast.ClassDef) and n.name in ROUTE_JOB_CLASSES:
                seen.setdefault(n.name, []).append(p.name)
    dup = ["%s: %s" % (k, v) for k, v in seen.items() if len(v) > 1]
    check(not dup, "执行器类被定义了不止一处（两个口径 ✗）：%s" % dup)
    wrong = ["%s 应在 %s，实际在 %s" % (k, v, seen.get(k))
             for k, v in ROUTE_JOB_CLASSES.items() if seen.get(k) != [v]]
    check(not wrong, "执行器类放错文件了：%s" % wrong)

    # ③ 单向依赖
    bad = []
    for p in sorted(jd.glob("*.py")):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("decision.pathing"):
                bad.append("%s L%d from %s" % (p.name, n.lineno, n.module))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith("decision.pathing"):
                        bad.append("%s L%d import %s" % (p.name, n.lineno, a.name))
    check(not bad,
          "`decision/jobs/` 反向 import 了 `decision/pathing` ⇒ 成环 ✗：%s" % bad[:4])

    # ④ 哨兵：规划必须真的从执行器拿
    plan_src = (pd / "plan.py").read_text(encoding="utf-8")
    check("from decision.jobs" in plan_src,
          "`pathing/plan.py` 没从 `decision.jobs` 拿执行器 ⇒ 它自己造不出任务（NameError ✗）")

    # ⑤ 旧路径不许再被 import
    old_imp = []
    for p in Path(ROOT).rglob("*.py"):
        if skip(p):
            continue
        try:
            t = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module == "decision.route":
                old_imp.append("%s L%d" % (p.relative_to(ROOT), n.lineno))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name == "decision.route":
                        old_imp.append("%s L%d" % (p.relative_to(ROOT), n.lineno))
    check(not old_imp,
          "`decision.route` 已经删了，还有地方 import 它（跑起来就 ImportError ✗）：%s"
          % old_imp[:6])


TESTS = TESTS + (
    ("⭐⭐ 快照冻结 `ws._freeze()` 必须在喂 Agent 之前（写者唯一由类型保证）",
     t_ws_frozen_before_tick),
    ("⭐⭐⭐⭐⭐ 搬走的名字不许再从旧模块上摸（治 `_ma.load_gain` ✓ 只跑才炸那种）",
     t_moved_attrs_called_on_new_module),
    ("⭐⭐⭐⭐ 拆包别名必须在使用处 import（治 `NameError: mouse_aim_c` ✓）",
     t_refactor_aliases_imported),
    ("⭐⭐⭐ `perception/` 四组（detect/track/world/screen）不许黏回顶层",
     t_perception_split),
    ("⭐⭐⭐ 小地图常量与实现分家（`map_const.py` 零依赖 / 一处定义 / 别从 minimap 摸）",
     t_map_const_split),
    ("⭐⭐⭐ 面板↔底图↔世界换算分家（`map_calib.py` 单向 / 一处定义 / 别从 minimap 摸）",
     t_map_calib_split),
    ("⭐⭐⭐ 认黄点 / 玩家定位分家（`map_dot` ← `map_locator` 单向 / 一处定义 / 别从 minimap 摸）",
     t_map_dot_split),
    ("⭐⭐⭐ 叠图核对 / 诊断命令行分家（`map_overlay` ← minimap 单向 / 诊断进 tools 且 gui 不许 import）",
     t_map_overlay_split),
    ("⭐⭐⭐ 寻路拆「执行器 jobs」+「规划 pathing」（单向 / 一处定义 / 旧 route.py 不许回来）",
     t_route_split),
    ("⭐⭐⭐ 函数内**向上**依赖冻结（模块级 0 也挡不住的那些：gui→tools 17 处…）",
     t_lazy_upward_ratchet),
    ("⭐⭐⭐ 单函数 ≥600 行冻结（拆块期间不许再造巨型函数：12 个基线）",
     t_long_fn_ratchet),
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
