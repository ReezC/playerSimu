"""把一次训练的指标翻译成**人话 + 下一步**（训练报告弹窗 / 训练日志用）。

**为什么单独一个模块**：卡片的 `summarize()` 只该拼字符串，而"这些数该怎么读"是
纯逻辑 —— 放这儿能脱离 PyQt 自检（`tools/selftest_metrics.py`、`selftest_train_report.py`），
命令行也能复用。

⚠ **刻意不设"多少分算好"的阈值**：那属于拍脑袋，见 README「工程约定：不许拍脑袋补数」。
这里只用**能算出来的东西**说话：
  · 指标之间比 —— P 和 R 谁在拖后腿（这是两者的定义，不是阈值）；
  · `mAP50` 与 `mAP50-95` 的**差** —— 差越大说明框的位置越不准；
  · 与**上一版**比 —— 涨了多少、跌了多少。
"够不够用"的最终判据只有一条：**实机跑一段看漏检 / 误检**（训练集数字只说明
"在这批图上"）。

指标键沿用 `run.json` 的写法（`perception/train.py` 写的）：
`{"precision", "recall", "map50", "map", "epochs", "base", "name"}`；缺的显示 `—`。

**两种出口**（2026-09-26）：
  · `advise_items()` → 结构化条目 `[{"kind", "text", "hl"}]` —— 给弹窗按 kind 上色；
  · `advise()`       → 把条目拼成一段纯文本 —— 给训练日志（`train.py`）和旧调用方。
`kind` 的**颜色由界面决定**（颜色是显示的事，纯逻辑层不认识颜色 —— 它要能脱离 PyQt 自检）。
"""

# ---- 条目种类（弹窗按它选颜色；这里只给语义，不给颜色）----
METRICS = "metrics"      # 指标行（P / R / mAP50 / mAP50-95 + 轮数）
MISSED = "missed"        # 漏检偏多（R < P）
FALSE = "false"          # 误检偏多（P < R）
BALANCED = "balanced"    # P / R 持平
BOX = "box"              # mAP50 与 mAP50-95 之差 → 框的位置准不准
COMPARE = "compare"      # 与上一版的涨跌
NOBASE = "nobase"        # 有上一版，但它没留下指标
FIRST = "first"          # 第一版（没有可比的历史）
ACCEPT = "accept"        # 验收口径（永远在最后一条）


def _f(v):
    """指标 → 三位小数；取不到给 `—`（和 `gui/steps/cards._fmt3` 同口径）。"""
    try:
        return "%.3f" % float(v)
    except (TypeError, ValueError):
        return "—"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def advise_items(run_info, prev_info=None):
    """→ `[{"kind", "text", "hl"}]`。

    `text` 是给人看的那句话（和以前 `advise()` 拼的一模一样）；
    `hl` 是这句话里**该高亮的关键词**（弹窗上色用）—— 关键词必须**真的出现在 text 里**，
    否则弹窗会白高亮一场（用例钉了这条）。
    """
    info = run_info or {}
    prev = prev_info or {}
    m = info.get("metrics") or {}
    ep = ("　（%s 轮）" % info["epochs"]) if info.get("epochs") else ""
    # ⚠ `hl` 里**别放单字母**（`P` / `R`）：它们出现在别的词里（`mAP50` 里的 P），
    # 高亮时会把 `<b>` 插进已经生成的 HTML 标签里，把整段渲染弄坏 ✗。
    # 指标行本身整行就是数据，不需要再挑关键词。
    out = [{"kind": METRICS, "hl": [],
            "text": "指标：P %s · R %s · mAP50 %s · mAP50-95 %s%s"
                    % (_f(m.get("precision")), _f(m.get("recall")),
                       _f(m.get("map50")), _f(m.get("map")), ep)}]

    p, r = _num(m.get("precision")), _num(m.get("recall"))
    if p is not None and r is not None:
        # ⚠ 按**显示精度**（三位小数）比：差不到 0.001 的两者，文字里会写成同一个数，
        # 那就绝不能说"R < P" —— 否则出现「R 0.879 < P 0.879」这种自相矛盾的话
        #（2026-09-26 拿真实产物试出来的，正是这条最容易让人不信这段文字）。
        p3, r3 = round(p, 3), round(r, 3)
        if r3 < p3:
            out.append({
                "kind": MISSED, "hl": ["漏检偏多", "更多样的怪"],
                "text": "→ 漏检偏多（R %s < P %s）：认到的都是对的，但有一批怪没认出来。"
                        "先让它见到更多样的怪（多录几段：不同位置、遮挡/重叠、远近、"
                        "怪多的时候），比加轮数管用。" % (_f(r), _f(p))})
        elif p3 < r3:
            out.append({
                "kind": FALSE, "hl": ["误检偏多", "负样本"],
                "text": "→ 误检偏多（P %s < R %s）：该认的都认到了，也认了些不该认的。"
                        "补一批「画面里没有怪 / 有别的东西」的负样本最直接。"
                        % (_f(p), _f(r))})
        else:
            out.append({
                "kind": BALANCED, "hl": ["持平"],
                "text": "→ P 与 R 持平（都是 %s）：两端没有明显偏科。" % _f(p)})

    m50, m95 = _num(m.get("map50")), _num(m.get("map"))
    if m50 is not None and m95 is not None:
        out.append({
            "kind": BOX, "hl": ["框的位置", "越不准", "输入尺寸"],
            "text": "→ mAP50 %s 与 mAP50-95 %s 差了 %+.3f：这个差越大，说明「框的位置」"
                    "越不准（mAP50-95 对框的松紧更敏感）—— 常见原因就是小目标 / "
                    "贴边目标；把标注框贴紧一点、把「输入尺寸」提上去，通常比加轮数有用。"
                    % (_f(m50), _f(m95), m50 - m95)})

    pm = prev.get("metrics") or {}
    if pm and m:
        bits = []
        for key, zh in (("map50", "mAP50"), ("map", "mAP50-95"),
                        ("precision", "P"), ("recall", "R")):
            a, b = _num(m.get(key)), _num(pm.get(key))
            if a is not None and b is not None:
                bits.append("%s %+.3f" % (zh, a - b))
        if bits:
            out.append({"kind": COMPARE, "hl": ["与上一版"],
                        "text": "→ 与上一版「%s」相比：%s"
                                % (prev.get("name") or "?", "　".join(bits))})
    elif prev:
        # ⚠ 有上一版、但它**没留下指标** —— 不能说成"这是第一版"（那是假话）。
        out.append({"kind": NOBASE, "hl": ["没留下指标记录"],
                    "text": "→ 上一版「%s」没留下指标记录（早期版本可能没写 run.json），"
                            "所以这次没法比涨跌 —— 以后每一版都会有。"
                            % (prev.get("name") or "?")})
    else:
        out.append({"kind": FIRST, "hl": ["第一版"],
                    "text": "→ 这是本项目第一版（没有上一版可比）：以后的每一版都会在这里"
                            "列出「比上一版涨跌多少」，方便判断改动有没有用。"})

    out.append({"kind": ACCEPT, "hl": ["验证", "实机走一段看漏检/误检"],
                "text": "→ 验收：拿这一版去「验证」步骤跑一批，再实机走一段看漏检/误检 —— "
                        "训练集上的数字只说明「在这批图上」。"})
    return out


def advise(run_info, prev_info=None):
    """→ 一段**多行**纯文本（通俗评估 + 建议）。没有任何指标时也返回"怎么验收"那行。

    就是 `advise_items()` 拼起来 —— 保留这个入口是为了训练日志（`perception/train.py`）
    和旧调用方不用改一行。要上色的界面请用 `advise_items()`。
    """
    return "\n".join(it["text"] for it in advise_items(run_info, prev_info))
