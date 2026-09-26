"""行为序列**预设**的存取（纯逻辑，不碰 Qt）—— 给「行为编辑器」的「保存 / 加载」用。

用户 2026-09-26 定的做法（T1，见 docs/开发计划.md）：
输出行为 / 回身输出 / 进入·退出隐身 / 自定义定时行为都要写序列，以前只能手敲，
**换号、换项目就得重来** ✗ ⇒ 存成文件、随时读回来；编辑器**上方两个按钮**（保存 / 加载）。

文件形状（**人能读**、能直接手改再加载回来）::

    {
      "format": "playerSimu.sequence",
      "version": 1,
      "name": "回身输出",
      "saved_at": "2026-09-26 15:04:22",
      "_note": "「行为编辑器 → 保存」写出来的预设；改完文本可以直接加载回来。",
      "seq": [
        {"type": "down", "key": "back", "prob": 80, "then": [
            {"type": "down", "key": "attack"}, {"type": "up", "key": "attack"}]},
        {"type": "up", "key": "back"},
        {"type": "delay", "ms": 200}
      ]
    }

判据（写死在这，别放松）:
  · **加载必须校验**：坏文件要报出**哪一段坏**，**绝不许静默清空** ✗ ——
    静默清空 = 用户以为加载成功了，其实序列被清掉 ⇒ 按键行为**直接变了** ✗；
  · 存 → 读 → 一致（含空序列、含 `then` 嵌套、含中文自定义键名）；
  · 元素里出现**没见过的字段**一律报错（打错一个字段名 = 那条动作**静默失效** ✗）。

**键名本身不校验**：序列里允许 `back` / `forward`（执行时按朝向解析）、固定键，
以及用户自己加的自定义键 —— 那是 `seq_editor` 的活。这里只管"结构对不对"
（不然换个 `custom_keys`、或者从别人机器上拷来的预设就读不进来 ✗）。
"""

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 预设默认放的目录（相对仓库根）。`config/` 里已经有别的预设清单（`push_presets.json`）。
DEFAULT_DIR = "config/sequences"

#: 「额外延迟」的上限（毫秒）= 24 小时。
#: 原来卡在 100000（100 秒），想配分钟级的等待根本填不进去。
MAX_DELAY_MS = 86400000

#: 文件标记 + 版本（认不出来的格式直接报错，别猜）。
FORMAT = "playerSimu.sequence"
VERSION = 1

#: 嵌套深度上限、元素总数上限 —— 防手改坏的文件把编辑器拖死。
MAX_DEPTH = 16
MAX_ELEMS = 5000

#: 元素形状：类型 → 允许出现的字段（`then` 三种都有：任何元素都能挂"触发后执行"）。
_TYPES = ("down", "up", "delay")
_FIELDS = {"down": ("key", "prob", "then"),
           "up": ("key", "prob", "then"),
           "delay": ("ms", "then")}

_NOTE = "「行为编辑器 → 保存」写出来的预设；改完文本可以直接加载回来。"


def _kind(v):
    """给报错用的人话类型名（"应该是整数，实际是字符串"比"类型错误"有用得多）。"""
    if v is None:
        return "空"
    if isinstance(v, bool):
        return "布尔"
    if isinstance(v, (int, float)):
        return "数字 %r" % (v,)
    if isinstance(v, str):
        return "字符串 %r" % (v,)
    if isinstance(v, dict):
        return "对象"
    if isinstance(v, (list, tuple)):
        return "列表"
    return type(v).__name__


def default_dir(root=None):
    """预设目录（**不保证存在**；`save_preset` 自己会建）。"""
    return Path(root) if root else (ROOT / DEFAULT_DIR)


def safe_name(name, fallback="行为预设"):
    """把名字变成能当文件名用的（行为名里可能有 `/` `:` 这些 Windows 不许的字符）。"""
    bad = '<>:"/\\|?*'
    out = "".join(("_" if c in bad else c) for c in str(name or "").strip())
    return out.strip(" .") or fallback


def count(seq, _depth=0):
    """元素总数（含嵌套）—— 给界面回话用（"存了 N 个元素"）。

    带深度上限：手改坏的文件（自引用）不会把这里递归到爆。

    ⚠ **数不出来就不数（算 0）**，别在这里报"超限"：
      · 元素大多没有子序列 ⇒ 缺席的 `then` 是 `None`，拿它当超限会让**任何一条正常
        序列**都报"元素太多"（写第一版时就是这么炸的 ✗）；
      · `then` 写成了对象 / 字符串那种**结构错**，要留给 `normalize` 逐段报
        （"seq[0].then：应该是列表"比"元素太多"有用一百倍 ✗）。
    """
    if seq is None or _depth > MAX_DEPTH or not isinstance(seq, (list, tuple)):
        return 0
    n = 0
    for e in seq:
        if isinstance(e, dict):
            n += 1 + count(e.get("then"), _depth + 1)
    return n


def normalize(seq, where="seq", _depth=0):
    """校验 + 规范化 ⇒ 一份**深拷贝**（原对象不动，可以安全地存/交给执行层）。

    坏结构抛 `ValueError`，消息里带**路径**（例：`seq[2].then[0].ms`）——
    "哪一段坏"比"文件坏了"有用得多。

    两处和编辑器保持一致的归一化（省得同一份序列在文件里/内存里两个样子）：
      · `prob` = 100 ⇒ **不写出来**（等于默认）；
      · `then` = 空列表 ⇒ **不写出来**。
    """
    if _depth > MAX_DEPTH:
        raise ValueError("%s：嵌套太深（超过 %d 层）" % (where, MAX_DEPTH))
    if isinstance(seq, tuple):
        seq = list(seq)
    if not isinstance(seq, list):
        raise ValueError("%s：应该是列表，实际是 %s" % (where, _kind(seq)))
    if count(seq) > MAX_ELEMS:
        raise ValueError("%s：元素太多（超过 %d 个）" % (where, MAX_ELEMS))
    out = []
    for i, elem in enumerate(seq):
        w = "%s[%d]" % (where, i)
        if not isinstance(elem, dict):
            raise ValueError("%s：应该是对象，实际是 %s" % (w, _kind(elem)))
        typ = elem.get("type")
        if typ not in _TYPES:
            raise ValueError("%s.type：只能是 %s，实际是 %s"
                             % (w, " / ".join(_TYPES), _kind(typ)))
        extra = sorted(k for k in elem if k not in ("type",) + _FIELDS[typ])
        if extra:
            raise ValueError("%s：不认识的字段 %s（这一档允许：type / %s）"
                             % (w, " / ".join(extra), " / ".join(_FIELDS[typ])))
        if typ == "delay":
            ms = elem.get("ms")
            if isinstance(ms, bool) or not isinstance(ms, int):
                raise ValueError("%s.ms：应该是整数毫秒，实际是 %s" % (w, _kind(ms)))
            if not 0 <= ms <= MAX_DELAY_MS:
                raise ValueError("%s.ms：要落在 0 ~ %d，实际是 %d"
                                 % (w, MAX_DELAY_MS, ms))
            new = {"type": "delay", "ms": int(ms)}
        else:
            key = elem.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("%s.key：应该有键名（字符串），实际是 %s"
                                 % (w, _kind(key)))
            new = {"type": typ, "key": key.strip()}
            if "prob" in elem:
                p = elem["prob"]
                if isinstance(p, bool) or not isinstance(p, int) or not 0 <= p <= 100:
                    raise ValueError("%s.prob：应该是 0~100 的整数，实际是 %s"
                                     % (w, _kind(p)))
                if p < 100:
                    new["prob"] = p
        if "then" in elem:
            sub = normalize(elem["then"], "%s.then" % w, _depth + 1)
            if sub:
                new["then"] = sub
        out.append(new)
    return out


def is_valid(seq):
    """能不能当序列用（只给是/否，不抛异常）。"""
    try:
        normalize(seq)
        return True
    except ValueError:
        return False


def save_preset(path, seq, name="", note=None, now=None):
    """把序列写成预设文件 ⇒ 写进去的那个 dict（含 `name` / `saved_at` / `seq`）。

    **原子写**：先写 `<名字>.json.tmp` 再 `os.replace` —— 中途失败（磁盘满、被杀）
    不会留半个文件把原来的预设毁掉 ✗。
    """
    p = Path(path)
    good = normalize(seq, "seq")                 # 先校验：坏序列**根本不落盘**
    doc = {"format": FORMAT,
           "version": VERSION,
           "name": str(name or p.stem or "行为预设"),
           "saved_at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
           "_note": note or _NOTE,
           "seq": good}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    # `ensure_ascii=False`：中文键名 / 行为名要**原样**能读（这是"人能读"的一半）
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(str(tmp), str(p))
    return doc


def load_preset(path):
    """读预设文件 ⇒ `{"name", "seq", "saved_at", "note"}`。

    坏文件抛 `ValueError`（消息里说清**哪一段坏**）；文件不存在 / 读不了抛 `OSError`。
    ⚠ **调用方只能报错，绝不许把当前序列清掉**（见模块头部的判据）。
    """
    p = Path(path)
    try:
        txt = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as ex:
        raise ValueError("%s：不是 UTF-8 文本（%s）" % (p.name, ex)) from None
    try:
        doc = json.loads(txt)
    except ValueError as ex:
        raise ValueError("%s：不是合法的 JSON（%s）" % (p.name, ex)) from None
    if not isinstance(doc, dict):
        raise ValueError("%s：顶层应该是对象，实际是 %s" % (p.name, _kind(doc)))
    if doc.get("format") not in (None, FORMAT):
        raise ValueError("%s：这不是行为预设（format=%s）"
                         % (p.name, _kind(doc.get("format"))))
    if "seq" not in doc:
        raise ValueError("%s：里面没有 `seq` 字段（没有序列可加载）" % p.name)
    ver = doc.get("version", VERSION)
    if isinstance(ver, int) and ver > VERSION:
        raise ValueError("%s：预设版本 %d 比本程序新（只认到 %d）—— 别拿旧程序读新文件"
                         % (p.name, ver, VERSION))
    return {"name": str(doc.get("name") or p.stem or "行为预设"),
            "seq": normalize(doc["seq"], "seq"),
            "saved_at": str(doc.get("saved_at") or ""),
            "note": str(doc.get("_note") or "")}


def list_presets(folder=None):
    """目录里的预设 ⇒ `[(名字, Path), ...]`（按名字排序）。

    读不了的文件**也列出来**（名字后面标"读不了"）—— 静默跳过就等于"我明明存过，
    怎么没了"，那种坑比报错难查得多。
    """
    d = Path(folder) if folder else default_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.name.lower()):
        if p.name.endswith(".json.tmp"):
            continue
        try:
            out.append((load_preset(p)["name"], p))
        except (OSError, ValueError):
            out.append(("%s（读不了）" % p.stem, p))
    return sorted(out, key=lambda t: t[0])
