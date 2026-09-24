"""UI 规范自检：把 docs/UI规范.md 里能自动查的条目查掉。

用法：
    python -m tools.check_ui          # 只报错
    python -m tools.check_ui --all    # 连提示一起报

**为什么要有这个脚本**：规范写在文档里没人会每次翻，而这几条都是踩过坑的 ——
尤其「滚轮改参数」和「导入写在方法里」，肉眼评审极难发现，脚本一秒就能查。

退出码：有 ERROR 返回 1，否则 0（可以直接挂到提交前检查）。
"""
import argparse
import ast
import io
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 参数控件：这些一律要用 gui/widgets.py 的 NoWheel* 版本
_VALUE_TYPES = ("QSpinBox", "QDoubleSpinBox", "QComboBox", "QSlider")
_RAW_RE = re.compile(r"\b(%s)\s*\(" % "|".join(_VALUE_TYPES))


def iter_files(subdir, suffix=".py"):
    base = ROOT / subdir
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for fn in sorted(files):
            if fn.endswith(suffix):
                yield Path(root) / fn


def iter_scanned(suffix=".py"):
    """需要过界面规范的全部文件（gui + deploy）。"""
    for sub in SCAN_DIRS:
        for p in iter_files(sub, suffix):
            yield p


def read(path):
    return io.open(path, encoding="utf-8").read()


def rel(path):
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


# ---------------------------------------------------------------- 检查项

# 扫哪些目录。deploy/ 是 A 机部署台（另一个界面），同一套界面规范照管。
SCAN_DIRS = ("gui", "deploy")


def check_raw_widgets(errors):
    """不能裸用参数控件 —— 必须走 NoWheel*（否则滚轮会改参数）。"""
    for p in iter_scanned():
        if p.name == "widgets.py":      # 定义处例外
            continue
        for i, ln in enumerate(read(p).splitlines(), 1):
            m = _RAW_RE.search(ln)
            # 排除 import 行、继承声明（class Foo(_NoWheel, QComboBox)）
            if not m or ln.lstrip().startswith(("import ", "from ", "class ")):
                continue
            # QSpinBox -> NoWheelSpinBox（去掉 Q 前缀再拼）
            suggested = "NoWheel" + m.group(1)[1:]
            errors.append("%s:%d  裸用 %s —— 请改用 %s（滚轮会误改参数）\n"
                          "        %s" % (rel(p), i, m.group(1), suggested, ln.strip()))


def _imports_with_scope(node, inside=False):
    """递归收集 import 节点，并标明它是否落在**函数/类体**里。

    `try:` / `if:` 不算 —— 它们不产生局部名字，只是模块级的普通语句。
    这条区分很关键：deploy/app.py 用模块级 `try:` 包住依赖导入（装不上时
    要先弹个错误框，而不是让 pythonw 静默退出），不该被当成违规。
    """
    for child in ast.iter_child_nodes(node):
        deeper = inside or isinstance(child, (ast.FunctionDef,
                                              ast.AsyncFunctionDef,
                                              ast.ClassDef))
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            yield child, inside
        yield from _imports_with_scope(child, deeper)


def _imports_widgets(node):
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").startswith("gui.widgets")
    return any(a.name.startswith("gui.widgets") for a in node.names)


def check_widgets_import_scope(errors):
    """gui.widgets 的导入必须在模块级。

    写在方法/类体里会变成「局部名字」：构造函数在它之前用到 NoWheel* 就会
    UnboundLocalError（本仓库已经这样炸过一次）。

    判据取 AST 上的**祖先里有没有 函数/类**，而不是「这一行有没有缩进」——
    后者连模块级 `try:` 里的 import 也会报（那是误报，见 _imports_with_scope）。
    """
    for p in iter_scanned():
        src = read(p)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        for node, inside in _imports_with_scope(tree):
            if inside and _imports_widgets(node):
                errors.append("%s:%d  gui.widgets 的导入被写进了函数/类体，"
                              "请挪到模块顶部\n        %s"
                              % (rel(p), node.lineno,
                                 lines[node.lineno - 1].strip()))


def check_settings_sync(warnings):
    """DecisionSettings 的字段 / to_dict / from_dict 要三边同步。

    新增参数最容易漏其中一边：漏 to_dict 就存不下去，漏 from_dict 就重启即丢，
    而且都不会报错 —— 表现是「我明明配了，重启就没了」。
    """
    path = ROOT / "decision" / "agent.py"
    text = read(path)
    m = re.search(r"class DecisionSettings:(.*?)\nclass ", text, re.S)
    if not m:
        warnings.append("decision/agent.py 里没找到 DecisionSettings，跳过同步检查")
        return
    body = m.group(1)

    # 字段：__init__ 里 self.xxx = ... 到「运行时状态，不持久化」标记为止
    init = body.split("def __init__", 1)[-1]
    marker = init.find("运行时状态，不持久化")
    persisted = init[:marker] if marker > 0 else init
    fields = []
    lines = init.splitlines()
    for i, line in enumerate(lines):
        if "以下是运行时状态" in line:     # 显式分节标记：这行之后全是运行时状态
            break
        fm = re.match(r"\s*self\.([A-Za-z_]\w*)\s*=", line)
        if not fm:
            continue
        # 「不持久化」写在同行或上面两行注释里都算
        ctx = "\n".join(lines[max(0, i - 2):i + 1])
        if "不持久化" not in ctx:
            fields.append(fm.group(1))

    dto = body.split("def to_dict", 1)[-1].split("def ", 1)[0]
    dto_keys = set(re.findall(r'"([a-z_][a-z0-9_]*)"\s*:', dto))

    frm = body.split("def from_dict", 1)[-1]
    frm_keys = set(re.findall(r'data\.get\(\s*"([a-z_][a-z0-9_]*)"', frm))

    for f in fields:
        if f not in dto_keys:
            warnings.append("DecisionSettings.%s 没写进 to_dict（改完存不下去）" % f)
        if f not in frm_keys:
            warnings.append("DecisionSettings.%s 没在 from_dict 里读（重启就丢）" % f)


def check_theme_colors(warnings):
    """（占位）样式里的硬编码颜色。

    **故意留空**：本仓库的约定就是用内联样式表写颜色，真扫起来上百条，
    全是不该改的。一个「每次跑都报一堆、但都合理」的检查等于没有检查 ——
    检查项必须可信，否则大家会习惯性忽略它的输出。
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="连提示一起报")
    args = ap.parse_args()

    errors, warnings = [], []
    check_raw_widgets(errors)
    check_widgets_import_scope(errors)
    check_settings_sync(warnings)
    check_theme_colors(warnings)

    if warnings and args.all:
        print("提示（%d 条）：" % len(warnings))
        for w in warnings:
            print("  - " + w)
        print()

    if errors:
        print("错误（%d 条）：" % len(errors))
        for e in errors:
            print("  [x] " + e)
        print("\n规范见 docs/UI规范.md")
        return 1

    n_warn = len(warnings)
    print("UI 规范自检通过（错误 0 条；提示 %d 条，加 --all 看明细）" % n_warn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
