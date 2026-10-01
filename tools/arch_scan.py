# -*- coding: utf-8 -*-
"""架构风险扫描（只读）：依赖分层 / 循环 import / 跨层引用 / 私有符号外借 / 巨型文件。"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKGS = {"core": 0, "perception": 1, "decision": 2, "gui": 3,
        "link": 1, "remote_kbd": 1, "deploy": 3, "tools": 4}

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

files = [p for p in ROOT.rglob("*.py")
         if ".git" not in p.parts and "__pycache__" not in p.parts
         and ".codebuddy" not in p.parts and "dist" not in p.parts
         and "venv" not in p.parts]

edges = {}          # mod -> set(mod)
lazy_edges = {}     # 函数内 import（常用于藏循环）
priv_borrows = []   # from x import _y（跨文件的私有符号）
big = []
for p in files:
    m = mod_of(p)
    top, lazy = imports_of(p)
    for imp in top + lazy:
        if not local_mod(imp):
            continue
        tgt = imp
        # from 包.模块 import x → 依赖到 模块 这一级
        d = (lazy_edges if imp in lazy else edges).setdefault(m, set())
        d.add(tgt)
    src = p.read_text(encoding="utf-8")
    t = ast.parse(src)
    for node in ast.walk(t):
        if isinstance(node, ast.ImportFrom) and node.module and local_mod(node.module):
            for a in node.names:
                if a.name.startswith("_") and not a.name.startswith("__"):
                    priv_borrows.append("%s -> from %s import %s" % (m, node.module, a.name))
    n = len(src.splitlines())
    if n >= 1500:
        big.append((n, str(p.relative_to(ROOT))))

print("=== 巨型文件（≥1500 行） ===")
for n, p in sorted(big, reverse=True):
    print("%6d  %s" % (n, p))

print("\n=== 分层违反（低层 import 高层；层数越大越上层） ===")
allowed_down = True
for m, ts in sorted(edges.items()):
    lm = m.split(".")[0]
    for t in ts:
        lt = t.split(".")[0]
        if lm in PKGS and lt in PKGS and PKGS[lt] > PKGS[lm]:
            print("  %s(%s) -> %s(%s)" % (m, PKGS[lm], t, PKGS[lt]))

print("\n=== 函数内 import（懒加载，常用于藏循环；列出跨包的） ===")
seen = set()
for m, ts in sorted(lazy_edges.items()):
    lm = m.split(".")[0]
    for t in sorted(ts):
        lt = t.split(".")[0]
        if lm != lt and (lm, t) not in seen:
            seen.add((lm, t))
            # 只报"本包内应该可以在顶层 import"的跨包引用
            print("  %s  lazy->  %s" % (m, t))

print("\n=== 跨文件私有符号外借（from x import _y） ===")
for s in priv_borrows:
    print("  " + s)

# 循环 import 检测（模块级，取包.模块两级；环长度 ≥2）
print("\n=== 模块级循环依赖（强连通，长度≥2） ===")
g = {m: set() for m in edges}
for m, ts in edges.items():
    for t in ts:
        if t in g:
            g[m].add(t)
index, low, on, stack, sccs, cnt = {}, {}, set(), [], [], [0]
def dfs(v):
    index[v] = low[v] = cnt[0]; cnt[0] += 1
    stack.append(v); on.add(v)
    for w in g.get(v, ()):
        if w not in index:
            dfs(w); low[v] = min(low[v], low[w])
        elif w in on:
            low[v] = min(low[v], index[w])
    if low[v] == index[v]:
        comp = []
        while True:
            w = stack.pop(); on.discard(w); comp.append(w)
            if w == v:
                break
        if len(comp) > 1:
            sccs.append(sorted(comp))
import sys as _s
_s.setrecursionlimit(10000)
for v in list(g):
    if v not in index:
        dfs(v)
for c in sccs:
    print("  " + " <-> ".join(c))
print("（%d 个环）" % len(sccs))
