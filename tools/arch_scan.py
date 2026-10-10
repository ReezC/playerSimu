# -*- coding: utf-8 -*-
"""架构风险扫描（只读）：依赖分层 / 循环 import / 跨层引用 / 私有符号外借 / 巨型文件。

**两个入口、一套判据**（用户 2026-10-10 ✓「架构的边界要有牙齿」✓）：

    · 人看报告：`python -m tools.arch_scan`
    · 机器拦截：`tools/selftest_arch.py` 直接调本文件的 `scan()` ⇒ 超基线就**变红** ✓
      （为什么非要这样：报告没人看就白搭 ✗ —— 本仓库 2026-10-10 那轮 6 个 bug 里
       **5 个**是"同一事实两条链 / 每拍职责被早退吃掉 / 隐式契约靠注释维护"✗，
       都是**一条 lint 就能拦住**的 ✓）

判据的层级（`PKGS` 里的数越小越底层）：

    core(0) < link/remote_kbd/perception(1) < decision(2) < gui/deploy(3) < tools(4)

⚠ 扫描范围**必须**跳开 `.venv` / `site-packages`（2026-10-10 修：原来用 `"venv" not in
  p.parts` 这种**精确相等**比较 ⇒ `.venv` 全漏进来 ✗ ⇒ 1951 行输出里绝大多数是
  `torch`/`pandas` 自己的文件，**真违例被噪音淹掉** ✗）。
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKGS = {"core": 0, "perception": 1, "decision": 2, "gui": 3,
        "link": 1, "remote_kbd": 1, "deploy": 3, "tools": 4}

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
    return [p for p in sorted(ROOT.rglob("*.py")) if not skip(p)]


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


def scan(verbose=True):
    """跑一遍全部判据，返回**结构化结果**（钉子和人看的是同一份 ✓）。

    返回：
        {"big":     [(行数, 相对路径), ...]                  巨型文件（≥1500 行 ✓）
         "layering":["gui.live_thread(3) -> tools(4)", ...]  低层 import 高层 ✓
         "lazy":    ["gui.live_thread  lazy->  decision", ...] 函数内 import（常用来藏环 ✓）
         "priv":    ["tools.x -> from y import _z", ...]     跨文件借私有符号 ✓
         "cycles":  [[a, b], ...]                            模块级循环依赖 ✓}
    """
    edges = {}          # mod -> set(mod)
    lazy_edges = {}     # 函数内 import（常用于藏循环）
    priv_borrows = []   # from x import _y（跨文件的私有符号）
    big = []
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
           "lazy": lazy, "priv": priv_borrows, "cycles": cycles}

    if verbose:
        print("=== 巨型文件（≥1500 行） ===")
        for n, path in res["big"]:
            print("%6d  %s" % (n, path))

        print("\n=== 分层违反（低层 import 高层；层数越大越上层） ===")
        for s in res["layering"]:
            print("  " + s)

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
