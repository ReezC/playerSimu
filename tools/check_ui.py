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


#: 毫秒类参数的名字特征：变量名里带这些词 ⇒ 必须是**整数**控件（不许小数）。
_MS_NAME_RE = re.compile(r"ms|millis|毫秒", re.I)

def check_unit_in_label(errors):
    """单位**不许写进编辑框**（`setSuffix`）—— 一律写在框外（2026-09-26 用户要求）。

    为什么：`setSuffix` 把单位塞进框内 ⇒ 1) 数字和单位挤在一起，长数字会被挤窄；
    2) 单位混进 `value()` 的显示文本里，读起来像"值的一部分"；
    3) 同一个单位在不同页面写法不一（" ms" / "ms" / " 毫秒"），查配置时对不上。
    做法：单位放进**表单行标签**（`addRow("对齐保持时间(ms)", sp)`），
    或紧挨着控件放一个 QLabel（`row.addWidget(QLabel("×"))`）。

    ⚠ 这条**一开始是"提示"级**（存量有 15 处，一上来就红一片的检查等于没有检查）；
    2026-09-26 存量清零后升级为**错误** —— 从此新写的代码不许再犯。
    """
    for p in iter_scanned():
        if p.name == "widgets.py":      # 定义处例外
            continue
        for i, ln in enumerate(read(p).splitlines(), 1):
            if "setSuffix(" not in ln:
                continue
            errors.append("%s:%d  单位不许写进编辑框（请挪到框外的标签里）：%s"
                          % (rel(p), i, ln.strip()))


def check_ms_is_integer(errors):
    """**毫秒类参数必须是整数控件**（2026-09-26 用户要求：ms 都用整数）。

    判据：变量名里带 ms / millis / 毫秒，却建成了 `NoWheelDoubleSpinBox`
    （或调了 `setDecimals(>0)`）⇒ 报错。
    为什么：毫秒没有小数意义，而小数位一旦写错（例如把"步进"写进了小数位）
    就会出现 `0.000…0`（50 个 0）这种框 —— 实测用户就被这个坑到过 ✗。
    """
    for p in iter_scanned():
        if p.name == "widgets.py":
            continue
        lines = read(p).splitlines()
        for i, ln in enumerate(lines, 1):
            m = re.match(r"\s*self\.([A-Za-z_]\w*)\s*=\s*(NoWheelDoubleSpinBox|QDoubleSpinBox)\s*\(", ln)
            if not m:
                continue
            if not _MS_NAME_RE.search(m.group(1)):
                continue
            errors.append("%s:%d  「%s」是毫秒参数，却建成了小数控件 —— "
                          "请改用 NoWheelSpinBox（毫秒一律整数）\n        %s"
                          % (rel(p), i, m.group(1), ln.strip()))
        # 同一个文件里跟手调的 setDecimals(>0) 也查一遍（名字带 ms 的那种）
        for i, ln in enumerate(lines, 1):
            m = re.match(r"\s*self\.([A-Za-z_]\w*)\.setDecimals\((\d+)\)", ln)
            if not m or not _MS_NAME_RE.search(m.group(1)):
                continue
            if int(m.group(2)) > 0:
                errors.append("%s:%d  「%s」是毫秒参数，却被设成了 %s 位小数 —— "
                              "毫秒一律整数（decimals=0）\n        %s"
                              % (rel(p), i, m.group(1), m.group(2), ln.strip()))


def _decision_fields():
    """`DecisionSettings` 里**会持久化**的字段名（解析口径同 `check_settings_sync`）。

    为什么单独抽出来：它现在有两个用处 —— 三边同步（字段/to_dict/from_dict）和
    "界面上有没有入口"。
    """
    text = read(ROOT / "decision" / "agent.py")
    m = re.search(r"class DecisionSettings:(.*?)\nclass ", text, re.S)
    if not m:
        return []
    init = m.group(1).split("def __init__", 1)[-1]
    marker = init.find("运行时状态，不持久化")
    persisted = init[:marker] if marker > 0 else init
    out = []
    lines = init.splitlines()
    for i, line in enumerate(lines):
        if "以下是运行时状态" in line:      # 显式分节标记：这行之后全是运行时状态
            break
        fm = re.match(r"\s*self\.([A-Za-z_]\w*)\s*=", line)
        if not fm:
            continue
        ctx = "\n".join(lines[max(0, i - 2):i + 1])
        if "不持久化" not in ctx:
            out.append(fm.group(1))
    _ = persisted
    return out


def check_settings_have_ui(warnings):
    """每个**持久化**的决策参数，界面上都要有能改它的地方（2026-09-26 用户要求）。

    为什么：参数只写进 `DecisionSettings` 而不给界面入口 ⇒ 就成了"**只能在配置文件里
    改**"的隐藏参数 —— 实测就这么栽过：`goto_timeout_s` 先只有 `to_dict`，
    界面上找不到，只能改 yaml ✗（断线重连那几个子参数也一样）。用户的说法是
    "把工作台所有的配置项都检查统一一遍"。

    判据：字段名**出现在 gui/ 或 deploy/ 的源码里**（读它或写它都算）。这是**提示**级：
    有些参数确实故意不做界面（脚本/工具读的），列出来只是为了让人**逐个确认**，
    不是逼着给每个都摆一个控件。
    """
    names = set()
    for p in iter_scanned():
        names.add(read(p))
    blob = "\n".join(names)
    for f in _decision_fields():
        if f not in blob:
            warnings.append("DecisionSettings.%s 在界面上没有入口"
                            "（只能在配置文件里改；确认是有意为之就忽略）" % f)


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
    check_ms_is_integer(errors)          # 毫秒一律整数（2026-09-26 新增）
    check_unit_in_label(errors)          # 单位写在框外（存量清零后已升级为错误）
    check_settings_sync(warnings)
    check_settings_have_ui(warnings)     # 每个参数都要有界面入口（2026-09-26 新增）
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
