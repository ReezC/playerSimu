"""「查看报告」弹窗自检（2026-09-26 用户要求：建议从卡片结果区**移位**到按钮弹窗）。

钉四件事（错了都会让人拿着假信息调参）：
  ① 左栏列出**本项目所有权重**（归档的 models/*.pt + 还在 runs 里的 best.pt），新的在前；
  ② 选中哪一版，右栏就显示**那一版**的数字（不是最新那版的）；
  ③ 「与上一版比」用的是**紧挨着的上一版**，且标签用**目录名**；
  ④ 上了色（关键词高亮 + 按种类整行颜色都在）—— 用户要的就是"不同颜色区分关键词"。

另外反向钉一条：**卡片结果区不许再贴那段建议**（它已经移走了，再贴回来就是两处重复、
且会跟弹窗对不上）。
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.steps import cards as cards_mod                       # noqa: E402
from gui.train_report import TrainReportDialog                 # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class _P:
    """最小项目桩：弹窗只用到 `dir_of`（和 `gui.Project` 同语义）。"""

    def __init__(self, root, name="测试项目"):
        self.root = Path(root)
        self.name = name

    def dir_of(self, what):
        d = self.root / what
        d.mkdir(exist_ok=True)
        return d


def _mk_run(proj, ver, m, epochs=100, base="yolo26n.pt"):
    """造一个训练产物目录：runs/detect_v<ver>/{weights/best.pt, run.json}。"""
    d = proj.dir_of("runs") / ("detect_v%d" % ver)
    (d / "weights").mkdir(parents=True, exist_ok=True)
    (d / "weights" / "best.pt").write_bytes(b"stub")      # 内容无所谓，只要存在
    (d / "run.json").write_text(json.dumps(
        {"name": "detect_v%d" % ver, "base": base, "epochs": epochs,
         "imgsz": 960, "batch": 8, "device": "0", "seconds": 600.0,
         "finished_at": "2026-09-26 12:00:00", "metrics": m},
        ensure_ascii=False), encoding="utf-8")
    return d


def t_lists_all_weights():
    """① 左栏 = 本项目所有权重，新的在前（归档那份优先，不重复列）。"""
    tmp = Path(tempfile.mkdtemp(prefix="tr_"))
    try:
        p = _P(tmp)
        _mk_run(p, 1, {"precision": 0.50, "recall": 0.40,
                       "map50": 0.600, "map": 0.300})
        _mk_run(p, 2, {"precision": 0.879, "recall": 0.870,
                       "map50": 0.922, "map": 0.650})
        # v2 归档成 models/detect_v2.pt（正式产物；同名以归档为准，不许列两遍）
        (p.dir_of("models") / "detect_v2.pt").write_bytes(b"stub")
        dlg = TrainReportDialog(p)
        names = [it[0] for it in dlg._items]
        check(names == ["detect_v2", "detect_v1"],
              "左栏没列出本项目所有权重（该新的在前、不重复）：%s" % names)
        check(dlg.lst.count() == 2, "左栏条目数不对：%d" % dlg.lst.count())
        dlg.close()
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_shows_selected_version():
    """② 选哪版显示哪版；③ 与上一版比用紧挨着的那一版 + 目录名标签。"""
    tmp = Path(tempfile.mkdtemp(prefix="tr_"))
    try:
        p = _P(tmp)
        _mk_run(p, 1, {"precision": 0.50, "recall": 0.40,
                       "map50": 0.600, "map": 0.300})
        _mk_run(p, 2, {"precision": 0.879, "recall": 0.870,
                       "map50": 0.922, "map": 0.650})
        (p.dir_of("models") / "detect_v2.pt").write_bytes(b"stub")
        dlg = TrainReportDialog(p)
        # 默认选最新（detect_v2）
        check(dlg.lst.currentRow() == 0, "默认没选最新那版")
        txt = dlg.txt.toPlainText()
        for want in ("detect_v2", "0.922", "0.650", "0.879", "0.870"):
            check(want in txt, "最新那版没显示 %s：\n%s" % (want, txt))
        check("与上一版「detect_v1」相比" in txt,
              "没写清是跟哪一版比（标签该用目录名）：\n%s" % txt)
        check("+0.322" in txt, "与上一版的差值没算对（0.922-0.600=+0.322）：\n%s" % txt)

        # 改选 detect_v1 ⇒ 右栏换成**那一版**的数字，且是"第一版"
        dlg.lst.setCurrentRow(1)
        t1 = dlg.txt.toPlainText()
        check("0.600" in t1 and "0.922" not in t1,
              "换了一版右栏还是旧的数字：\n%s" % t1)
        check("第一版" in t1, "第一版该明说没有可比的历史：\n%s" % t1)
        dlg.close()
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_colors_and_keywords():
    """④ 上了色：按种类整行颜色 + 关键词加粗高亮（用户要的"区分关键词与重要信息"）。"""
    tmp = Path(tempfile.mkdtemp(prefix="tr_"))
    try:
        p = _P(tmp)
        _mk_run(p, 1, {"precision": 0.90, "recall": 0.30,     # R < P ⇒ 漏检
                       "map50": 0.900, "map": 0.500})
        dlg = TrainReportDialog(p)
        html = dlg.rendered_html()
        check("color:" in html, "整行没上色：\n%s" % html[:400])
        check('style="color:#f28b82"' in html,
              "『漏检偏多』那条没用红色标出来：\n%s" % html[:600])
        check('<b style="color:#ffd54f">' in html,
              "关键词没加粗高亮（该是 %s）：\n%s" % ("#ffd54f", html[:600]))
        for kw in ("漏检偏多", "更多样的怪", "框的位置"):
            check((">%s</b>" % kw) in html,
                  "关键词「%s」没被高亮（要么文案改了，要么 hl 对不上）：\n%s"
                  % (kw, html[:800]))
        dlg.close()
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_missing_and_empty():
    """没有 run.json 的权重 / 空项目：都要**如实说**，不许编。"""
    tmp = Path(tempfile.mkdtemp(prefix="tr_"))
    try:
        p = _P(tmp)
        (p.dir_of("models") / "handmade.pt").write_bytes(b"stub")   # 手工塞的权重
        dlg = TrainReportDialog(p)
        check("没留下指标记录" in dlg.txt.toPlainText(),
              "没有 run.json 的权重该明说：\n%s" % dlg.txt.toPlainText())
        dlg.close()

        empty = _P(Path(tempfile.mkdtemp(prefix="tr0_")))
        dlg2 = TrainReportDialog(empty)
        check("还没有训练过" in dlg2.txt.toPlainText(),
              "空项目该明说：\n%s" % dlg2.txt.toPlainText())
        dlg2.close()
        shutil.rmtree(str(empty.root), ignore_errors=True)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_card_moved_it_out():
    """反向钉：卡片结果区**不许再贴**那段建议，但有「查看报告」按钮。"""
    tmp = Path(tempfile.mkdtemp(prefix="tr_"))
    try:
        p = _P(tmp)
        _mk_run(p, 1, {"precision": 0.90, "recall": 0.30,
                       "map50": 0.900, "map": 0.500})
        c = cards_mod.TrainCard()
        txt = c.summarize(p)
        check("历次 mAP50" in txt, "卡片摘要本身没了：%s" % txt)
        for bad in ("验收", "漏检偏多", "第一版", "误检偏多"):
            check(bad not in txt,
                  "建议又贴回卡片结果区了（该只在「查看报告」弹窗里）：%s" % txt)
        check(hasattr(c, "btn_report") and c.btn_report.text() == "查看报告",
              "卡片上没有「查看报告」按钮")
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


TESTS = (
    ("弹窗左栏列出本项目所有权重（新的在前、归档优先、不重复）", t_lists_all_weights),
    ("选中哪版显示哪版 + 与上一版比（目录名标签、差值算对）", t_shows_selected_version),
    ("不同颜色区分关键词与重要信息（整行按 kind 上色 + 关键词高亮）", t_colors_and_keywords),
    ("没有 run.json 的权重 / 空项目：如实说，不编", t_missing_and_empty),
    ("卡片结果区不再贴建议 + 有「查看报告」按钮", t_card_moved_it_out),
)


def main():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])      # noqa: F841
    bad = 0
    for name, fn in TESTS:
        try:
            fn()
            print("[ OK ] %s" % name)
        except Exception as e:                             # noqa: BLE001
            bad += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
    print("\n%d/%d 通过" % (len(TESTS) - bad, len(TESTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
