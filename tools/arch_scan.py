# -*- coding: utf-8 -*-
"""架构风险扫描（只读）：依赖分层 / 循环 import / 跨层引用 / 私有符号外借 / 巨型文件。

**两个入口、一套判据**（用户 2026-10-10 ✓「架构的边界要有牙齿」✓）：

    · 人看报告：`python -m tools.arch_scan`
    · 机器拦截：`tools/selftest_arch.py` 直接调本文件的 `scan()` ⇒ 超基线就**变红** ✓
      （为什么非要这样：报告没人看就白搭 ✗ —— 本仓库 2026-10-10 那轮 6 个 bug 里
       **5 个**是"同一事实两条链 / 每拍职责被早退吃掉 / 隐式契约靠注释维护"✗，
       都是**一条 lint 就能拦住**的 ✓）

判据的层级（`PKGS` 里的数越小越底层）：

    core(0) < link/remote_kbd/perception(1) < decision/runtime(2) < gui/deploy(3) < tools(4)

⚠ `runtime`（2026-10-15 阶段 1「编排收口」新建 ✓）与 `decision` **同层**：它**编排**
  `perception`(1) + `decision`(2)，自己被 `gui`(3) 调用 ✓。另有一条更严的钉子单独管它
  （**不许 import `gui` / `tools`** ✓ 见 `tools/selftest_arch.py` ✓）—— 因为它是"调度"，
  一旦沾上 Qt / 屏幕 / Ultralytics 就再也拆不出来了 ✗。

⚠ 扫描范围**必须**跳开 `.venv` / `site-packages`（2026-10-10 修：原来用 `"venv" not in
  p.parts` 这种**精确相等**比较 ⇒ `.venv` 全漏进来 ✗ ⇒ 1951 行输出里绝大多数是
  `torch`/`pandas` 自己的文件，**真违例被噪音淹掉** ✗）。
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKGS = {"core": 0, "perception": 1, "decision": 2, "gui": 3,
        "link": 1, "remote_kbd": 1, "deploy": 3, "tools": 4,
        # ⭐⭐ **执行层（发键/动鼠标）独立成包**（2026-10-17 ✓ 阶段 2 ✓）：
        #   只许 import `core` / `remote_kbd`（都是 1 层及以下 ✓）——
        #   ⚠ **不许** import `decision` / `gui` / `perception` ✗（那是它的上层 ✓）
        #   判据：`execution/input.py`、`execution/mouse_aim.py` ✓
        "execution": 1,
        # ⭐⭐ **测谎独立成包**（2026-10-17 ✓ 阶段 3 ✓）：`perception/lie_*` 五个文件搬出。
        #   ⚠ 允许依赖 `core.geom` / `core.kf`（同一层 ✓ 通用件 ✓）；
        #   **不许** import `decision` / `gui` / `tools` ✗
        "liedetect": 1}

#: 扫描时要跳过的目录名**片段**（**片段**匹配 ✓ 不是精确相等 ✗ 见文件头那段）
SKIP_PARTS = (".venv", "venv", "site-packages", "__pycache__", ".git", ".codebuddy",
              "dist", "build", "node_modules", "_backup", "_backup_myfiles",
              "visual_tracking_sdk")
#: 自己不算（本文件自己就写着这些字面量 ✓ 否则报告里全是它自己 ✓）
SKIP_NAMES = ("arch_scan.py",)


def skip(p):
    """该文件要不要跳过（venv / 缓存 / 备份 / 第三方 SDK / 扫描器自己 ✓）。

    ⚠ 用**前缀**匹配 ✗ 不是精确相等 —— 2026-10-10 实测踩到：目录名是
      `visual_tracking_sdk_20260920`（带日期后缀 ✓）⇒ `seg in SKIP_PARTS` 这种精确比较
      **跳不掉它** ✗ ⇒ 报告里混进一堆第三方文件 ✓。
    """
    try:
        parts = p.relative_to(ROOT).parts
    except ValueError:
        return True
    return (any(seg == t or seg.startswith(t) for seg in parts for t in SKIP_PARTS)
            or p.name in SKIP_NAMES)


def py_files():
    """要被扫的 `.py`（跳过上面那些目录 ✓）。"""
    # ⚠⚠ **不许用 `ROOT.rglob("*.py")` + 事后 `skip()` 过滤** ✗（2026-10-15 实测踩到 ✓）：
    #   `rglob` 会**先一头钻进 `.venv`** ✗ —— 而这里是**目录联接**（junction ✓）⇒ Windows 当场
    #   `OSError: [WinError 448] 无法遍历该路径，因为它包含不受信任的装入点` ✗ ⇒ `scan()` 整个
    #   抛出去 ⇒ **E-①（分层/环）与 E-②（巨型文件）两条判据一起红** ✗✗。
    #   ⚠ 这种红最坏：它**看着像**"分层更差了 / 又冒巨型文件了"✗，其实**两条判据一个字都没判** ✓
    #     —— 判据被中断 ≠ 判据通过（本仓库的"判据空转"家族又一员 ✓）。
    #   ⚠ `skip()` 是**事后过滤**：文件/目录**已经被遍历过了** ✗ ⇒ 拦不住遍历本身 ✓
    #     （所以 E-①/E-② 之前一直挂着这两条红 ✓ —— 不是环境天生如此，是这里该改 ✓）。
    #   ⇒ `os.walk(topdown=True)` + **就地改 `dirs[:]`**（只有就地改才真的不进去 ✓）；
    #     再挂 `onerror` 兜住别的读不了的目录 ✓（一个坏目录不该中断整次扫描 ✓）。
    import os
    out = []
    for base, dirs, names in os.walk(str(ROOT), topdown=True, onerror=lambda e: None):
        b = Path(base)
        # ⚠ 片段匹配（同 `SKIP_PARTS` 的口径 ✓ 见 `skip()` 那段为什么用片段 ✗）
        dirs[:] = [d for d in dirs
                   if not any(frag in d.lower() for frag in SKIP_PARTS)]
        for n in names:
            if n.endswith(".py"):
                p = b / n
                if not skip(p):
                    out.append(p)
    return sorted(out)


def mod_of(path):
    rel = path.relative_to(ROOT)
    parts = rel.with_suffix("").parts
    if len(parts) == 1:
        return parts[0]
    return ".".join(parts)


def imports_of(path):
    """返回 (顶层 import 列表, 函数内 import 列表)。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        print("SYNTAXERR %s: %s" % (path, e))
        return [], []
    top, lazy = [], []

    def walk(node, in_func):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(ch, True)
                continue
            if isinstance(ch, ast.Import):
                for a in ch.names:
                    (lazy if in_func else top).append(a.name)
            elif isinstance(ch, ast.ImportFrom):
                if ch.level == 0 and ch.module:
                    (lazy if in_func else top).append(ch.module)
                elif ch.level > 0:
                    base = ".".join(path.relative_to(ROOT).with_suffix("").parts[:ch.level])
                    mod = base + (("." + ch.module) if ch.module else "")
                    (lazy if in_func else top).append(mod)
            else:
                walk(ch, in_func)
    walk(tree, False)
    return top, lazy


def local_mod(m):
    """m 是否本项目模块（前缀命中）。"""
    return m.split(".")[0] in PKGS


def _cycles(g):
    """模块级强连通分量（长度≥2 = 循环依赖）✓ Tarjan 迭代版（避免深递归 ✗）。"""
    index, low, on, stack, sccs = {}, {}, set(), [], []
    cnt = [0]
    sys.setrecursionlimit(10000)

    def dfs(v):
        index[v] = low[v] = cnt[0]
        cnt[0] += 1
        stack.append(v)
        on.add(v)
        for w in g.get(v, ()):
            if w not in index:
                dfs(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                sccs.append(sorted(comp))

    for v in list(g):
        if v not in index:
            dfs(v)
    return sccs


def _own_body(node):
    """遍历一个函数的**自身**语句（**不钻进嵌套函数/类** ✗）。

    为什么需要：`ast.walk(fn)` 会把**内层函数**里的 import 也算进外层 ⇒
    同一处向上依赖被数两次 ✗（拆块期间这类计数必须准，否则棘轮天天假红 ✗）。
    """
    stack = list(getattr(node, "body", []) or [])
    while stack:
        x = stack.pop()
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield x
        stack.extend(ast.iter_child_nodes(x))


def scan(verbose=True):
    """跑一遍全部判据，返回**结构化结果**（钉子和人看的是同一份 ✓）。

    返回：
        {"big":     [(行数, 相对路径), ...]                  巨型文件（≥1500 行 ✓）
         "layering":["gui.live_thread(3) -> tools(4)", ...]  低层 import 高层 ✓
         "lazy":    ["gui.live_thread  lazy->  decision", ...] 函数内 import（常用来藏环 ✓）
         "lazy_up": {"gui -> tools": 17, ...}                 **函数内向上依赖**的次数（2026-10-17 ✓）
         "long_fn": [(行数, 相对路径, 函数名), ...]           单函数 ≥600 行（2026-10-17 ✓）
         "priv":    ["tools.x -> from y import _z", ...]     跨文件借私有符号 ✓
         "cycles":  [[a, b], ...]                            模块级循环依赖 ✓}
    """
    edges = {}          # mod -> set(mod)
    lazy_edges = {}     # 函数内 import（常用于藏循环）
    priv_borrows = []   # from x import _y（跨文件的私有符号）
    big = []
    lazy_up = {}        # "src -> dst" -> 次数（**函数内向上依赖** ✓ 2026-10-17）
    long_fn = []        # 单函数 ≥600 行（拆块期间最该盯的指标 ✓）
    for p in py_files():
        m = mod_of(p)
        top, lazy = imports_of(p)
        for imp in top + lazy:
            if not local_mod(imp):
                continue
            d = (lazy_edges if imp in lazy else edges).setdefault(m, set())
            d.add(imp)
        src = p.read_text(encoding="utf-8")
        t = ast.parse(src)
        for node in ast.walk(t):
            if isinstance(node, ast.ImportFrom) and node.module and local_mod(node.module):
                for a in node.names:
                    if a.name.startswith("_") and not a.name.startswith("__"):
                        priv_borrows.append("%s -> from %s import %s" % (m, node.module, a.name))
        # ⭐ 函数内**向上**依赖 + 超长函数（2026-10-17 加 ✓ 拆块期间的两条棘轮）
        #   口径：只数"**写在函数体里**、且目标层号更大"的 import ✓
        #   （模块级的向上依赖由 `layering` 那条管 ✓ 这里专抓"藏在函数里绕过去"的 ✗）
        _sp = m.split(".")[0]
        for fn in [x for x in ast.walk(t)
                   if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for sub in _own_body(fn):
                dst = None
                if isinstance(sub, ast.ImportFrom) and sub.level == 0 and sub.module:
                    dst = sub.module.split(".")[0]
                elif isinstance(sub, ast.Import) and sub.names:
                    dst = sub.names[0].name.split(".")[0]
                if dst and _sp in PKGS and dst in PKGS and PKGS[dst] > PKGS[_sp]:
                    _k = "%s -> %s" % (_sp, dst)
                    lazy_up[_k] = lazy_up.get(_k, 0) + 1
            _ln = (fn.end_lineno or fn.lineno) - fn.lineno
            if _ln >= 600:
                long_fn.append((_ln, str(p.relative_to(ROOT)).replace("\\", "/"),
                                fn.name))

        n = len(src.splitlines())
        if n >= 1500:
            # ⚠ 路径一律折成 POSIX 风格（`/` ✓）—— 否则 Windows 上返回 `decision\agent.py`，
            #   而钉子的基线名单写的是 `decision/agent.py` ⇒ **一比较全都不相等** ✗（2026-10-10
            #   实测踩到：E-② 当场把 26 个文件全报成「新巨型」✗）。
            big.append((n, str(p.relative_to(ROOT)).replace("\\", "/")))

    layering = []
    for m, ts in sorted(edges.items()):
        lm = m.split(".")[0]
        for t in sorted(ts):
            lt = t.split(".")[0]
            if lm in PKGS and lt in PKGS and PKGS[lt] > PKGS[lm]:
                layering.append("%s(%s) -> %s(%s)" % (m, PKGS[lm], t, PKGS[lt]))

    seen = set()
    lazy = []
    for m, ts in sorted(lazy_edges.items()):
        lm = m.split(".")[0]
        for t in sorted(ts):
            lt = t.split(".")[0]
            if lm != lt and (lm, t) not in seen:
                seen.add((lm, t))
                lazy.append("%s  lazy->  %s" % (m, t))

    g = {m: set() for m in edges}
    for m, ts in edges.items():
        for t in ts:
            if t in g:
                g[m].add(t)
    cycles = _cycles(g)

    res = {"big": sorted(big, reverse=True), "layering": layering,
           "lazy": lazy, "priv": priv_borrows, "cycles": cycles,
           "lazy_up": dict(sorted(lazy_up.items(), key=lambda kv: -kv[1])),
           "long_fn": sorted(long_fn, reverse=True)}

    if verbose:
        print("=== 巨型文件（≥1500 行） ===")
        for n, path in res["big"]:
            print("%6d  %s" % (n, path))

        print("\n=== 分层违反（低层 import 高层；层数越大越上层） ===")
        for s in res["layering"]:
            print("  " + s)

        print("\n=== 函数内**向上**依赖（模块级 0 也挡不住的那些 ✗ 按处数） ===")
        for k, c in res["lazy_up"].items():
            print("  %-26s %3d 处" % (k, c))

        print("\n=== 单函数 ≥600 行（拆块最该盯的指标 ✗） ===")
        for ln, path, name in res["long_fn"]:
            print("%5d  %-44s %s()" % (ln, path, name))

        print("\n=== 函数内 import（懒加载，常用于藏循环；列出跨包的） ===")
        for s in res["lazy"]:
            print("  " + s)

        print("\n=== 跨文件私有符号外借（from x import _y） ===")
        for s in res["priv"]:
            print("  " + s)

        print("\n=== 模块级循环依赖（强连通，长度≥2） ===")
        for c in res["cycles"]:
            print("  " + " <-> ".join(c))
        print("（%d 个环）" % len(res["cycles"]))
    return res


def main():
    scan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
