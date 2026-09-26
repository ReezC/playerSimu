"""训练建议（`perception/metrics.advise`）自检：数字必须来自 run.json，不许编。

为什么值得单列一条：这段文字会被当成"看训练结果的依据"。它一旦出现**凭空阈值**
（"mAP50 大于 0.9 就算好"这种谁也说不出来源的话），人就会照着一个拍脑袋的标准
调参 —— 所以这条用例钉两件事：
  ① 提到的数**必须是**传进去的那几个（P/R/mAP50/mAP50-95、与上一版的差值）；
  ② 指标缺失时**不许崩、也不许瞎说**（该显示 `—`，并仍然给出"怎么验收"那句）。

（口径见 README「工程约定：不许拍脑袋补数」。）
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.metrics import advise                            # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _info(**kw):
    m = {k: kw.pop(k) for k in ("precision", "recall", "map50", "map")
         if k in kw}
    out = {"name": "detect_v1", "epochs": 120, "metrics": m}
    out.update(kw)
    return out


def t_numbers_come_from_the_run():
    """① 文字里的数**就是**传进去的那几个；与上一版的差值算得对。"""
    cur = _info(precision=0.72, recall=0.41, map50=0.780, map=0.463)
    prev = dict(_info(precision=0.70, recall=0.30, map50=0.749, map=0.467),
                name="detect_v0")
    txt = advise(cur, prev)
    for want in ("0.720", "0.410", "0.780", "0.463", "120"):
        check(want in txt, "文字里没写出 %s：\n%s" % (want, txt))
    # 差值：0.780-0.749=+0.031、0.463-0.467=-0.004、P +0.020、R +0.110
    for want in ("+0.031", "-0.004", "+0.020", "+0.110"):
        check(want in txt, "没写出与上一版的差值 %s：\n%s" % (want, txt))
    check("detect_v0" in txt, "没写清是跟哪一版比：\n%s" % txt)
    check("实机" in txt,
          "没给出最终验收口径（训练集数字只说明「在这批图上」）：\n%s" % txt)


def t_r_lower_says_missed():
    """② R < P ⇒ 说"漏检偏多"；P < R ⇒ 说"误检偏多"（这是两者的定义，不是阈值）。"""
    a = advise(_info(precision=0.80, recall=0.30))
    check("漏检" in a, "R 明显低于 P 却没说漏检：\n%s" % a)
    b = advise(_info(precision=0.30, recall=0.80))
    check("误检" in b, "P 明显低于 R 却没说误检：\n%s" % b)


def t_missing_metrics_are_honest():
    """③ 指标缺失：不崩、显示 `—`、仍然给验收口径（绝不编数）。"""
    txt = advise({"name": "detect_v1", "metrics": {}})
    check("—" in txt, "没有指标时该显示 — ：\n%s" % txt)
    check("0." not in txt.replace("0.000", ""),
          "没有指标时却冒出了假的小数：\n%s" % txt)
    txt2 = advise(None, None)
    check("实机" in txt2, "空输入也要给出验收口径：\n%s" % txt2)


def t_first_version_says_so():
    """④ 第一版没得比时**说出来**（别留空、也别假装有对比）。"""
    txt = advise(_info(precision=0.5, recall=0.5, map50=0.5, map=0.3))
    check("第一版" in txt, "第一版该说明没有可比的历史：\n%s" % txt)


TESTS = (
    ("建议里的数字必须来自 run.json（含与上一版的差值）", t_numbers_come_from_the_run),
    ("R<P 说漏检 / P<R 说误检", t_r_lower_says_missed),
    ("指标缺失时不许崩、不许编数", t_missing_metrics_are_honest),
    ("第一版明说没有可比的历史", t_first_version_says_so),
)


def main():
    bad = 0
    for name, fn in TESTS:
        try:
            fn()
            print("[ OK ] %s" % name)
        except Exception as e:                   # noqa: BLE001
            bad += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
    print("\n%d/%d 通过" % (len(TESTS) - bad, len(TESTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
