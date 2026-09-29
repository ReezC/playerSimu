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

from gui import theme as _theme                                # noqa: E402
from gui.steps import cards as cards_mod                       # noqa: E402
from gui.train_report import TrainReportDialog                 # noqa: E402

# --------------------------------------------------------------- 自检不许改用户文件
# ⛔ `config/ui.yaml` 是**用户的**文件（窗口几何 / 字号 / 颜色）⇒ 自检**绝不许**写它 ✗
#    （全仓库规矩 ✓，见 `selftest_main_window._fake_store` 的说明）：本套件会开训练报告
#    弹窗（接了 `theme.bind_window_state` ⇒ 一 show/hide 就写 `windows.train_report` ✗ ——
#    2026-09-29 逐个套件量出来的）⇒ 把 theme 的**落点**指到临时文件 ✓。
_theme.CFG = Path(tempfile.mkdtemp(prefix="psimu_ui_")) / "ui.yaml"


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


def _mk_run(proj, ver, m, epochs=100, base="yolo26n.pt", frames=None):
    """造一个训练产物目录：runs/detect_v<ver>/{weights/best.pt, run.json}。

    `frames`（可选）= `{"train": 3, "val": 2, "total": 5}` ⇒ 写进 run.json ✓；
    不传就是**老产物**（没有这一项 ✓ ⇒ 报告该显示 "—" ✓）。
    """
    d = proj.dir_of("runs") / ("detect_v%d" % ver)
    (d / "weights").mkdir(parents=True, exist_ok=True)
    (d / "weights" / "best.pt").write_bytes(b"stub")      # 内容无所谓，只要存在
    info = {"name": "detect_v%d" % ver, "base": base, "epochs": epochs,
            "imgsz": 960, "batch": 8, "device": "0", "seconds": 600.0,
            "finished_at": "2026-09-26 12:00:00", "metrics": m}
    if frames is not None:
        info["frames"] = dict(frames)
    (d / "run.json").write_text(json.dumps(info, ensure_ascii=False),
                                encoding="utf-8")
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


def t_train_card_model_combo():
    """⑦「基础权重」是**下拉**（2026-09-27 用户要求"整理一下"）：官方预训练 + 各项目权重 + 自定义。

    为什么要它（用户原话："我希望基础权重这里能用下拉列表选择"）：新项目和旧项目
    **背景、怪都很像**时，直接挑旧项目那一版微调最省事；而以前这格是**文本框** ——
    人不知道有哪些能选、该填什么 ✗。

    钉四件：
      ① 候选里有**官方预训练**（仓库根的 `*.pt`；至少 `yolo26n.pt`，默认值永远选得上 ✓）；
      ② 有**各项目练好的权重**（`tools.yolo_augment.list_weights()`，与 ④ 那张卡的
         「YOLO 权重」**同一处来源** ✓）—— 本机还没训过就跳过这条（不硬报 ✗）；
      ③ **当前值不在候选里也要补一项**（老项目里手打过的怪路径 ⇒ 不许被吞掉 ✗：
         `set_value` 按 userData 找项，找不到就**静默不选中** ⇒ 界面显示的会和你实际
         拿去训练的**不是同一个** ✗）；
      ④ 最后一项是**自定义 / 浏览…**（逃生口：任意 .pt 仍填得进去 ✓）。
    """
    from tools.yolo_augment import list_weights

    c = cards_mod.TrainCard()
    cmb = c.widgets["model"][0]
    c._fill_model_options(None, cur="yolo26n.pt")
    datas = [cmb.itemData(i) for i in range(cmb.count())]
    texts = [cmb.itemText(i) for i in range(cmb.count())]
    check("yolo26n.pt" in datas,
          "候选里没有官方预训练（默认值就是它）：%s" % datas)
    check(cmb.currentData() == "yolo26n.pt",
          "默认没选中 yolo26n.pt：%r" % cmb.currentText())
    check(c.values(["model"])["model"] == "yolo26n.pt",
          "取值没走 userData（存进 project.yaml 的会是别的）：%s"
          % c.values(["model"])["model"])
    check(datas[-1] == cards_mod.TrainCard.MODEL_CUSTOM,
          "最后一项不是「自定义 / 浏览…」逃生口：%s" % datas[-1])
    check(texts[-1].startswith("（自定义"),
          "自定义那一项没写清楚：%r" % texts[-1])
    others = [p for _l, p, _pid in list_weights()]
    if others:
        miss = [p for p in others if p not in datas]
        check(not miss,
              "有些项目已训好的权重没进下拉（跨项目微调就选不到）：%s" % miss[:3])
    else:
        print("      （本机还没有任何项目训过的权重，跳过第 ② 条）")
    # ③ 当前值不在候选里 ⇒ 补一项，而且**真的选中**（显示 = 实际）
    c._fill_model_options(None, cur="D:/some/where/weird_best.pt")
    check(cmb.currentData() == "D:/some/where/weird_best.pt",
          "存着的怪路径没被选中（界面会显示别的权重、训练却用存的 ✗）：%r"
          % cmb.currentText())
    check("当前填写" in cmb.currentText(),
          "补进来的那一项没标明是「当前填写」：%r" % cmb.currentText())


def t_train_resume():
    """⭐⭐ **「接着上次跑」（resume）**（用户 2026-09-29 ✓ 原话："自动找最近一个 run 的
    last.pt、崩了能一键续、并且在续之前先报一句当前显存够不够"）。

    钉四件：
      ① `find_last_ckpt` 按 **`last.pt` 的 mtime** 找最近 —— ⚠ **不是按目录名**：
         `detect_v10` 的字典序在 `detect_v2` **前面** ⇒ 按名字排会**续错 run** ✗；
      ② `_run_args` 能从 run 目录的 `args.yaml` 读回**真实参数**（续训时界面上填的
         一律不生效 ⇒ 日志 / `run.json` / 摘要必须说**真话** ✓）；
      ③ `_vram_note`：**够就不吵、不够就 warn**（本次训练就是显存不够崩的 ⇒ 先报一句 ✓）；
      ④ 源码级：`model.train(resume=True)` 真的传下去了、CLI 有 `--resume`、
         GUI 训练卡片上有这一项且会传给 `run_train` ✓。
    """
    import os
    import tempfile
    import time
    from pathlib import Path

    from perception import train as T

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "runs"

        def _mk(name, when, imgsz=1280, batch=8, epochs=200):
            """造一个 run：`weights/last.pt` + `args.yaml`，mtime 设成 when ✓"""
            d = root / name
            (d / "weights").mkdir(parents=True)
            ck = d / "weights" / "last.pt"
            ck.write_bytes(b"x")
            (d / "args.yaml").write_text(
                "epochs: %d\nimgsz: %d\nbatch: %d\nname: %s\n"
                % (epochs, imgsz, batch, name), encoding="utf-8")
            for f in (ck, d / "args.yaml"):
                os.utime(str(f), (when, when))
            return ck

        # ⚠ 名字**小**的那个是**旧**的、名字**大**的是**新**的 —— 专门用来抓"按名字排序"这个错 ✗
        #   （字典序里 `detect_v10` < `detect_v2` ⇒ 按名字排会挑中**旧**的那个 ✓
        #     这样"反向验证"才咬得住：把按 mtime 改成按名字排 ⇒ 当场红 ✓）。
        _mk("detect_v2", time.time() - 600, imgsz=960, batch=4, epochs=120)
        _mk("detect_v10", time.time() - 10)

        _got = T.find_last_ckpt(str(root))
        check(_got is not None and _got.parent.parent.name == "detect_v10",
              "「接着上次跑」找错了 run —— 必须按 **last.pt 的修改时间**找最近那个，"
              "不能按目录名排序（字典序 `detect_v10` 在 `detect_v2` 前面 ⇒ 会续错 run ✗）：%r"
              % (_got,))

        _a = T._run_args(_got)
        check(int(_a.get("imgsz") or 0) == 1280 and int(_a.get("batch") or 0) == 8
              and int(_a.get("epochs") or 0) == 200,
              "没从 `args.yaml` 读回**真实参数**（续训时界面填的不生效 ⇒ 日志/摘要会说假话 ✗）：%r"
              % (_a,))

        class _Ctx(object):
            def __init__(self):
                self.lines = []

            def log(self, msg, level="info"):
                self.lines.append((level, msg))

        # ⚠ 「够就不吵」这条**只有在真能读到显存时**才有意义 —— 没有 GPU / 拿不到
        #   `mem_get_info` 时 `_vram_note` 会**静默跳过**（那是**设计**：查不到就不瞎报 ✗），
        #   用例不该因此判红 ✗（否则同一份代码在 CI / 没卡的机器上必红 ✓）。
        # ⚠ `import torch` **本身**也可能失败（DLL / 驱动问题 —— 见过
        #   `OSError: [WinError 1114] ... c10.dll` ✓），那是**环境**问题、不是这次改动
        #   的问题 ⇒ 不许让用例因此判红 ✗。
        # ⭐ **判据三档**（纯函数 ⇒ 能造场景 ✓ —— 2026-09-29 就是这里定错了阈值：
        #   原来按"八成富余"判 ⇒ 空闲 10.8GB / 需 9.2GB 被误报成"可能不够" ⇒ 把能跑的人
        #   劝退了 ✗。真实判据是"**要不要得下**"✓）。
        check(T._vram_verdict(10.8, 9.2) == "ok",
              "空间明明够（空闲 10.8 / 需 9.2）却报「偏紧 / 不够」⇒ 误报会让提醒变噪音 ✗")
        check(T._vram_verdict(2.4, 9.2) == "short",
              "空间明显不够却没报 `short` ✗（这正是 2026-09-28 训练崩掉的现场）")
        check(T._vram_verdict(10.0, 9.2) == "tight",
              "余量只剩 0.8GB 却没报「偏紧」✗")
        check(T._vram_verdict(20.0, 9.2) == "ok" and T._vram_verdict(None, 1) == "ok",
              "余量充足 / 传了坏值 ⇒ 都该安静返回 `ok`（不许崩 ✗）")

        try:
            import torch as _torch
            _has_gpu = bool(_torch.cuda.is_available())
        except Exception:
            _has_gpu = False
        if _has_gpu:
            _good = _Ctx()
            T._vram_note(320, 1, _good)      # 320²×1 ⇒ 约 0.07GB ⇒ 肯定够 ✓
            check(len(_good.lines) == 1,
                  "显存**明明够**却还在报警（提醒变噪音 ⇒ 以后没人看 ✗）：%r"
                  % (_good.lines,))
            _bad = _Ctx()
            T._vram_note(100000, 9999, _bad)  # 天文数字 ⇒ 必定不够 ✓
            check(any(lv == "warn" for lv, _ in _bad.lines),
                  "显存**明显不够**却不提醒（用户 2026-09-29 要的就是「续之前先报一句」✗）")
        else:
            _n = _Ctx()
            T._vram_note(1280, 8, _n)
            check(_n.lines == [],
                  "没有可用 GPU 时不该凭空报显存（查不到就静默跳过 ✓）：%r" % (_n.lines,))

        _root = Path(__file__).resolve().parent.parent
        _src = (_root / "perception" / "train.py").read_text(encoding="utf-8")
        # ⭐ 早停时"轮数"必须说**实话**（2026-09-29 ✓ 用户看到 `epoch 95/200` 之后直接
        #   "训练完成"、界面上却写「200 轮」⇒ 来问"怎么回事" ✓）。⚠ 用**源码级钉**
        #   （真跑一次训练来测不现实 ✗ —— 要十几分钟 + 显存 ✓）。
        check('"epochs": done_epochs' in _src,
              "`run.json` 的 `epochs` 又写回**计划值**了 ⇒ 早停时「历次版本」会说假话 ✗")
        check('"epochs_planned": epochs' in _src
              and '"early_stopped": early_stopped' in _src,
              "没把「计划轮数 / 是否早停」另存 ⇒ 信息丢了（改回计划值就没法分辨 ✗）")
        check("done_epochs < epochs" in _src and "早停" in _src,
              "没算实际轮数 / 没标早停 ⇒ 界面会从 `epoch 95` 直接跳到「完成」，"
              "看着像 bug ✗（用户 2026-09-29 就是被这个弄糊涂的）")
        check('"（早停）" if early_stopped else ""' in _src,
              "摘要没标出早停 ⇒ 卡片上只看得到「200 轮」✗")

        check("model.train(resume=True)" in _src,
              "续训分支没把 `resume=True` 真传给 ultralytics（勾了也没用 ✗）")
        check('ap.add_argument("--resume"' in _src,
              "CLI 没有 `--resume`（命令行续不了 ✗）")
        check("find_last_ckpt" in _src and "_vram_note" in _src,
              "`perception/train.py` 少了续训要用的两个件 ✗")
        _cards = (_root / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
        check('"resume", "接着上次跑", "bool"' in _cards
              and '"resume": bool(sec.get(' in _cards,
              "训练卡片上没有「接着上次跑」这一项 / 没把它传给 `run_train` ✗")


def t_train_frames_used_and_guard():
    """⭐⭐「补了帧但没重跑 ⑥」要**看得见**（用户 2026-09-29 ✓ 原话："我后补的帧数据训练，
    它如果早退了那是不是白补了？"）。

    代码事实（这是这条用例要钉的东西）：`perception/train.py` 全文**不扫 `frames/`、不读
    `split.json`**，只吃 `dataset/images/{train,val}` 那份**物理拷贝** ⇒ 补了帧却没跑 ⑥ 时，
    那批帧**一张都用不上** ✗；而原来全仓**没有任何地方**报"这次训练用了多少帧" ✗。

    钉四件：
      ① `count_dataset_images`：按 `data.yaml` 的 `path` + 相对目录数**磁盘上真有几张** ✓
         （训练的**唯一**输入就是它 ✓）；坏 yaml / 目录不在 ⇒ `None`（**绝不抛** ✓）；
      ② 训练开始**打一行**"本次用到 N 张" + `run.json` **记一份**（口径同一处 ✓）；
      ③ 报告里显示那一份（**老 `run.json` 没有这个键 ⇒ 显示 "—"** ✓ 不编数 ✗）；
      ④ ⑦ 卡片**开跑前挡住**："frames/ 里有 N 张比 data.yaml 新 ⇒ 先跑 ⑥" ✓
         （⑥ 会重写 data.yaml ⇒ 挡的条件自动消失 ✓）。
    """
    import inspect
    import os

    from perception import train as train_mod

    # ① 数磁盘（data.yaml 的 path + 相对目录；没有 path 就相对 yaml 自己）
    tmp = Path(tempfile.mkdtemp(prefix="tr_"))
    try:
        ds = tmp / "dataset"
        for sub, n in (("train", 3), ("val", 2)):
            d = ds / "images" / sub
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (d / ("frame_%05d.jpg" % i)).write_bytes(b"x")
        # ⚠ yaml **故意放在 dataset 外面**（`path` 指进去）—— 这样"必须按 `path` 解析"
        #   才是被测到的（把 yaml 放在 dataset 里的话，两种写法结果一样 ⇒ 钉不住 ✗）
        y = tmp / "data.yaml"
        y.write_text("path: %s\ntrain: images/train\nval: images/val\n"
                     % ds.as_posix(), encoding="utf-8")
        check(train_mod.count_dataset_images(y) == (3, 2, 5),
              "数出来的张数不对（该是 train 3 / val 2）：%r"
              % (train_mod.count_dataset_images(y),))
        # 没有 `path` 键 ⇒ 相对 **yaml 自己**那层解析（老 data.yaml 就是这个样子的可能 ✓）
        y2 = ds / "data2.yaml"
        y2.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
        check(train_mod.count_dataset_images(y2) == (3, 2, 5),
              "没有 `path` 键时没按 yaml 自己那层解析：%r"
              % (train_mod.count_dataset_images(y2),))
        # 数不出来（文件不在 / 内容根本不是配置）⇒ 一律 None，**绝不抛** ✓
        check(train_mod.count_dataset_images(tmp / "根本没有这个.yaml") is None,
              "文件不在时该返回 None（**绝不抛**：这是给人看的一行字 ✗）")
        bad = ds / "bad.yaml"
        bad.write_text("这不是一份配置（就是一个字符串）\n", encoding="utf-8")
        check(train_mod.count_dataset_images(bad) is None,
              "yaml 不是配置时该返回 None（**绝不抛**）")

        # ② 源码级：那一行日志 + run.json 里的 frames（口径同一处 ✓）
        _tsrc = inspect.getsource(train_mod)
        check("本次用到 %d 张" in _tsrc,
              "训练开始没报「本次用到 N 张」（那「补了帧没用上」就还是看不见 ✗）")
        check("count_dataset_images(data)" in _tsrc,
              "那行数字不是走 `count_dataset_images`（口径该只有一处 ✗）")
        check('"total": _n[2]' in _tsrc,
              "`run.json` 没记这一版用了多少张（事后没法核对 ✗）")

        # ③ 报告里显示（有 frames 的显示数字；老产物显示 "—"）
        p = _P(tmp)
        _mk_run(p, 1, {"precision": 0.90, "recall": 0.30,
                       "map50": 0.900, "map": 0.500},
                frames={"train": 3, "val": 2, "total": 5})
        dlg = TrainReportDialog(p)
        try:
            html = dlg.rendered_html()
            # ⚠ 判据带上那个全角冒号（标签是 `训练帧数：` ✓）：只查"训练帧数"的话，
            #   标签被改坏（比如 "训练帧数x"）也照样通过 ✗（反向验证当场抓过这一次 ✓）
            check("训练帧数：" in html,
                  "报告里没有「训练帧数」这一行（补帧有没有用上，翻报告也看不出来 ✗）")
            check("5 张（train 3 / val 2）" in html,
                  "报告没把帧数写全：\n%s" % html[:600])
        finally:
            dlg.close()
        tmp2 = Path(tempfile.mkdtemp(prefix="tr_"))
        try:
            p2 = _P(tmp2)
            _mk_run(p2, 1, {"precision": 0.90, "recall": 0.30,
                            "map50": 0.900, "map": 0.500})     # 老产物：没有 frames
            d2 = TrainReportDialog(p2)
            try:
                h2 = d2.rendered_html()
                check("训练帧数" in h2 and "—" in h2,
                      "老 run.json（没这一项）该显示「—」，不许编数 ✗：\n%s" % h2[:600])
            finally:
                d2.close()
        finally:
            shutil.rmtree(str(tmp2), ignore_errors=True)

        # ④ ⑦ 卡片：frames/ 里有比 `dataset/data.yaml` 新的帧 ⇒ **开跑前挡住** ✓
        #   （卡片看的是**数据集里那一份** data.yaml ✓ 与项目真实布局一致 ✓）
        p.dataset = ds
        y_card = ds / "data.yaml"
        y_card.write_text("path: %s\ntrain: images/train\nval: images/val\n"
                         % ds.as_posix(), encoding="utf-8")
        (tmp / "frames").mkdir(exist_ok=True)
        old = os.stat(y_card).st_mtime
        (tmp / "frames" / "frame_99999.jpg").write_bytes(b"x")
        os.utime(tmp / "frames" / "frame_99999.jpg", (old + 10, old + 10))
        card = cards_mod.TrainCard()
        try:
            ok, why = card.check_deps(p)
            check(not ok, "有 1 张新帧没进数据集，却放行训练了（那批帧会白补 ✗）")
            check("1 张" in why and "⑥" in why,
                  "挡住时没说清「有几张 / 该怎么办」：%r" % (why,))
            # ⑥ 跑过（data.yaml 被重写、比帧新）⇒ 自动放行 ✓
            os.utime(y_card, (old + 999, old + 999))
            ok2, why2 = card.check_deps(p)
            check(ok2 and not why2,
                  "重跑 ⑥ 之后还被挡着（人会被锁在门外 ✗）：%r / %r" % (ok2, why2))
        finally:
            card.deleteLater()
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


TESTS = (
    ("⭐⭐ 「补了帧没跑 ⑥」：训练报帧数 + run.json 记一份 + 报告显示 + ⑦ 卡片开跑前挡住",
     t_train_frames_used_and_guard),
    ("⭐⭐ ⑦「接着上次跑」：按 last.pt 的 **mtime** 找最近 run（不是按目录名）+ 读回真实"
     "参数 + 续前报显存（用户 2026-09-29）", t_train_resume),
    ("弹窗左栏列出本项目所有权重（新的在前、归档优先、不重复）", t_lists_all_weights),
    ("选中哪版显示哪版 + 与上一版比（目录名标签、差值算对）", t_shows_selected_version),
    ("不同颜色区分关键词与重要信息（整行按 kind 上色 + 关键词高亮）", t_colors_and_keywords),
    ("没有 run.json 的权重 / 空项目：如实说，不编", t_missing_and_empty),
    ("卡片结果区不再贴建议 + 有「查看报告」按钮", t_card_moved_it_out),
    ("⑦「基础权重」下拉：官方预训练 + 各项目权重 + 自定义（当前值不许被吞掉）",
     t_train_card_model_combo),
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
