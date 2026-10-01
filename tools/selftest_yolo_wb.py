# -*- coding: utf-8 -*-
"""yolo工作台自检：**纯逻辑** + 离屏开窗（不跑训练 ✗ 那要权重/数据 ✓）。

纯逻辑（脱离 Qt ✓）：
  ① 标签读写回环（像素 ⇄ 归一化 ✓ 坏行跳过 ✓ 原子写不留 .tmp ✓）；
  ② data.yaml 生成（类别顺序 = id ✓；**选帧清单**模式 ✓ 2026-09-30）；
  ③ 数据集扫描（扩展名过滤 ✓ 排序 ✓）；
  ④ 自动提案（合成图上有已知形状 ⇒ 框能套上 ✓）；
  ⑤ 提案去重 / 选帧台账 / 框数统计（2026-09-30 新功能四件的纯逻辑 ✓）。
离屏开窗：
  ⑥ 三页签能建出来、切页不炸 ✓（离屏环境 ✓——本套件**没有** QGraphicsView
     进不去的问题：质检台画布在别的套件里建过 ✓）；
  ⑦ 标注页**筛选/排序** + **复制/粘贴/撤销** + 选帧弹窗 + 训练页按清单出 yaml
     （2026-09-30 用户要的四件 ✓）。
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui import yolo_workbench as W  # noqa: E402

_FAILED = []


def check(cond, msg):
    if cond:
        print("  [OK] %s" % msg)
    else:
        print("  [NG] %s" % msg)
        _FAILED.append(msg)


def _mk_dataset(tmp):
    """合成一个迷你数据集：3 张图（画三个白色斑块）+ 标签目录 ✓。"""
    root = Path(tmp) / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    rng = np.random.RandomState(7)
    for i in range(3):
        img = rng.randint(80, 160, (240, 320, 3), dtype=np.uint8)
        cx, cy = 60 + 90 * i, 120
        cv2.rectangle(img, (cx - 25, cy - 25), (cx + 25, cy + 25),
                      (255, 255, 255), -1)
        p = root / "images" / ("img%02d.png" % i)
        cv2.imwrite(str(p), img)
        # 已有标签：一个框套住那个白块（像素 → 存成归一化 ✓）
        W.save_yolo_labels(W.label_path_for(p, root / "labels"),
                           [(0, cx, cy, 50.0, 50.0)], 320, 240)
    return root


def test_pure():
    import tempfile as _t
    with _t.TemporaryDirectory(prefix="yolowb_") as tmp:
        root = _mk_dataset(tmp)
        # ③ 扫描
        imgs = W.scan_images(root / "images")
        check(len(imgs) == 3 and imgs[0].name == "img00.png",
              "数据集扫描（3 张、按名排序 ✗）：%r" % ([p.name for p in imgs],))
        # ① 读写回环
        got = W.load_yolo_labels(W.label_path_for(imgs[0], root / "labels"),
                                 320, 240)
        check(len(got) == 1 and abs(got[0][1] - 60.0) < 1e-6
              and abs(got[0][3] - 50.0) < 1e-6,
              "标签读写回环（像素坐标还原 ✗）：%r" % (got,))
        # 坏行跳过 ✓
        lp = W.label_path_for(imgs[0], root / "labels")
        lp.write_text("0 0.1 0.1 0.2 0.2\n垃圾行\n0 0.3 0.3 0.1 0.1\n",
                      encoding="utf-8")
        check(len(W.load_yolo_labels(lp, 320, 240)) == 2,
              "坏行该跳过（一行坏数据炸整帧 ✗）")
        check(not list(lp.parent.glob("*.tmp")),
              "原子写不该留 .tmp 残片 ✗")
        # ② data.yaml
        y = W.make_data_yaml(root, ["apple", "star"])
        check("path:" in y and "  0: apple" in y and "  1: star" in y,
              "data.yaml 生成（类别顺序 = id ✗）：%r" % (y,))
        # ④ 提案：白块应该被提案套住（中心接近真值 ✓）
        img = cv2.imread(str(imgs[1]))
        props = W.propose_boxes(img)
        hit = any(abs(px - 150.0) < 40 and abs(py - 120.0) < 40
                  for px, py, _w, _h in props)
        check(hit, "自动提案没套住图上的白块（真值 (150,120)）：%r" % (props[:4],))

        # ---- ⑤ 新功能四件的纯逻辑（2026-09-30）----
        # IoU / 提案去重（YOLO 提案只补漏 ✓ 同 yolo_augment 口径 0.30 ✓）
        a = (0, 0.0, 0.0, 50.0, 50.0)             # (cls, x, y, w, h) 左上角 ✓
        near = (0, 5.0, 5.0, 50.0, 50.0)          # IoU ≈ 0.68 > 0.30 ⇒ 丢 ✓
        far = (0, 200.0, 200.0, 50.0, 50.0)       # 无重叠 ⇒ 留 ✓
        kept = W.dedup_proposals([a], [near, far])
        check(kept == [far],
              "提案去重（与已有框重叠大的该丢、不重叠的该留 ✗）：%r" % (kept,))
        check(abs(W.iou_xywh((0, 0, 10.0, 10.0), (0, 0, 10.0, 10.0)) - 1.0) < 1e-9
              and W.iou_xywh((0, 0, 10.0, 10.0), (100, 100, 10.0, 10.0)) == 0.0,
              "IoU 基本量（同框=1、无重叠=0 ✗）")
        # 框数统计（有效行 ✓ 坏行不算 ✓ 没文件=0 ✓）
        check(W.count_label_lines(lp) == 2
              and W.count_label_lines(lp.parent / "nope.txt") == 0,
              "框数统计（有效行数 / 缺文件=0 ✗）")
        # 选帧台账 roundtrip + 过滤
        sel_p = root / "train_frames.json"
        W.save_selection(sel_p, ["img00", "img02"])
        check(W.load_selection(sel_p) == ["img00", "img02"],
              "选帧台账 roundtrip（json 原子写 ✗）")
        check([p.name for p in W.apply_selection(imgs, ["img00"])] == ["img00.png"]
              and [p.name for p in W.apply_selection(imgs, None)]
              == [p.name for p in imgs],
              "按台账过滤（保持扫描序 ✓ None=全部 ✗）")
        # data.yaml 清单模式（train/val 指向清单 ✓）
        y2 = W.make_data_yaml(root, ["apple"], subset="train_frames.txt")
        check("train: train_frames.txt" in y2 and "val: train_frames.txt" in y2,
              "data.yaml 清单模式（train/val 指向清单 ✗）：%r" % (y2,))


def test_window():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    # ⭐ 空临时目录压过会话恢复（别把用户真数据集 26893 张载进后台 scan ✗）
    win = W.YoloWorkbench(dataset_root=tempfile.mkdtemp(prefix="ywb_empty_"))
    win._save_session_on_close = False   # §11：自检不许写用户的 config ✓
    win.tab_label.ed_images.setText("")       # 空数据集 ✓ 不读盘
    win.tabs.setCurrentIndex(1)               # 切到训练页 ✓
    win.tabs.setCurrentIndex(2)               # 切到验证页 ✓
    check(win.tabs.count() == 3 and win._names() == ["shape"],
          "三页签 + 默认类别名（开窗即用 ✗）：%r" % (win._names(),))
    win.close()
    del win
    app.processEvents()


def test_settings_dialog_builds():
    """⭐ **设置弹窗能建 + 蒙版滑条在位**（2026-09-29 用户报"设置打不开"的钉子 ✓）。

    第一版把局部函数 `_on_mask` 误写成 `self._on_mask` ⇒ 打开即 AttributeError ✗。
    ⚠ 放本套件：这里的离屏 Qt 环境**健康**（review 套件建 SettingsDialog 会
    0xC0000409 ✗ —— 同 selftest_decision 的已知问题 ✓）。
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    from gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    check(dlg is not None and dlg.sld_mask is not None,
          "设置弹窗 / 蒙版滑条建不出来（用户报的『设置打不开』✗）")
    dlg.close()
    dlg.deleteLater()
    app.processEvents()


def test_window_features():
    """⑦ **2026-09-30 用户要的四件**（离屏开窗 ✓ 真迷你数据集 ✓）：

    ① YOLO 提案按钮在位（真推理要 ultralytics+权重 ⇒ 只钉 UI 在位 + 纯逻辑在 ⑤ ✓）；
    ② 训练页「选帧」：台账 → `train_frames.txt` 清单 → data.yaml 指向它 ✓；
    ③ 标注页筛选/排序：行文本带框数、筛"空帧"该把有框的藏掉 ✓；
    ④ 画布复制/粘贴/撤销：**发真信号**走全链路（复制→粘贴→框变多→撤销→还原 ✓）。
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory(prefix="yolowb_win_") as tmp:
        root = _mk_dataset(tmp)
        # ⭐ **传 dataset_root**（2026-09-30 ✓，注意语义是**数据集根** ⇒ 内部拼
        #   /images ✓）：会话恢复（150.14）会把用户上次的真实数据集（现场 26893 张 ✓）
        #   灌回来，后台 scan 占着 _task ⇒ 测试的 reload 被"上一个任务还在跑"顶掉 ✗。
        #   ⚠ 会话还会带 labels_dir ⇒ **必须清空**（否则测试写用户的真实标签 ✗✗）。
        win = W.YoloWorkbench(dataset_root=str(root))
        win._save_session_on_close = False   # §11：自检不许写用户的 config ✓
        win.tab_label.ed_labels.setText("")  # 会话灌回的用户标签目录 ✗ 清掉
        win.tab_label.ed_images.setText(str(root / "images"))
        win.tab_label.reload()
        # ⭐ 载入已后台化 ⇒ 测试要**泵事件**等到 scan 收尾（_task 归 None ✓）
        from time import sleep as _sleep
        for _ in range(600):
            app.processEvents()
            if win.tab_label._task is None:
                break
            _sleep(0.02)
        check(win.tab_label._task is None and win.tab_label.lst.count() == 3,
              "后台载入该完成（3 张 ✗）：count=%d state=%r log_tail=%r"
              % (win.tab_label.lst.count(), win.tab_label.lbl_state.text(),
                 win.tab_label.log.toPlainText()[-300:]))
        tab = win.tab_label

        # ---- ③ 筛选 / 排序 ----
        check(tab.lst.count() == 3 and "1框" in tab.lst.item(0).text(),
              "列表行该带框数（3 张、各 1 框 ✗）：%r" % (tab.lst.item(0).text(),))
        tab.cmb_filter.setCurrentIndex(1)     # 空帧 ⇒ 全藏（都有 1 框 ✓）
        check(tab.lst.count() == 0, "筛「空帧」该把有框的全藏掉 ✗")
        tab.cmb_filter.setCurrentIndex(0)
        check(tab.lst.count() == 3, "筛回「全部」该都回来 ✗")
        tab.cmb_sort.setCurrentIndex(2)       # 按框数（并列 ⇒ 名字序 ✓）
        check([tab.lst.item(i).data(0x0100) for i in range(3)]
              == ["img00.png", "img01.png", "img02.png"],
              "按框数排序（并列回退名字序 ✗）：%r"
              % ([str(tab.lst.item(i).data(0x0100)) for i in range(3)],))
        tab.cmb_sort.setCurrentIndex(0)

        # ---- ③b 排序：按创建时间（2026-09-30 用户要的 Windows 同款 ✓）----
        idx_ct = tab.cmb_sort.findData("ctime")
        check(idx_ct >= 0, "排序下拉该有「按创建时间」（key=ctime ✗）")
        tab.cmb_sort.setCurrentIndex(idx_ct)
        check(tab.lst.count() == 3, "按创建时间排序不该丢行 ✗")
        tab.cmb_sort.setCurrentIndex(0)

        # ---- ③c 框上类名 = 顶部配的（2026-09-30 用户报"YOLO 提案显示玩家"✓）----
        tab.lst.setCurrentRow(0)
        check(bool(tab.canvas.boxes)
              and tab.canvas.boxes[0].label.text() == "shape",
              "框标签该显示顶部配的类别名（shape ✗ 不是画布写死的『玩家』）：%r"
              % (tab.canvas.boxes[0].label.text() if tab.canvas.boxes else None,))

        # ---- ④ 复制 / 粘贴 / 撤销（发真信号走全链路 ✓）----
        tab.lst.setCurrentRow(0)              # 载入第一帧（1 框 ✓）
        check(len(tab.canvas.get_boxes()) == 1, "第一帧该载入 1 个框 ✗")
        tab.canvas.boxes[0].setSelected(True)
        tab.canvas.copy_requested.emit()
        check("已复制" in tab.lbl_state.text(), "复制后该有状态回执 ✗")
        tab.canvas.paste_requested.emit()
        check(len(tab.canvas.get_boxes()) == 2, "粘贴后该是 2 个框（偏移 12px ✓）✗")
        tab.canvas.undo_requested.emit()
        check(len(tab.canvas.get_boxes()) == 1,
              "撤销后该回到 1 个框（粘贴前快照 ✗）：%d" % len(tab.canvas.get_boxes()))
        check(tab._undo == [], "撤销栈该被弹空（无残留 ✗）")

        # ---- ④b 切帧往返不漂移（2026-09-30 用户报"切一帧再切回来全部错位"✓）----
        #   根因 = `load_yolo_labels` 给中心坐标、`canvas.load` 当左上角画 ⇒ 每次往返
        #   向右下漂 (半宽, 半高) 且**累积** ✗。修复 = 载入时中心→左上角 ✓。
        before = [(b[0], round(b[1], 4), round(b[2], 4), round(b[3], 4), round(b[4], 4))
                  for b in tab.canvas.get_boxes()]
        tab._save()
        tab.lst.setCurrentRow(1)              # 切走（改过 ⇒ 自动保存 ✓）
        tab.lst.setCurrentRow(0)              # 切回来
        after = [(b[0], round(b[1], 4), round(b[2], 4), round(b[3], 4), round(b[4], 4))
                 for b in tab.canvas.get_boxes()]
        check(before == after,
              "切帧往返框位置该分毫不差（中心/左上角换算 ✗ 会漂半宽半高）：%r vs %r"
              % (before, after))

        # ---- ④b2 「人工改过」记账（2026-09-30 用户要求 ✓）----
        _man_p = W.manual_ledger_path(root / "labels")
        check(_man_p.is_file() and "img00" in (W.load_selection(_man_p) or []),
              "画布保存过 ⇒ 该记进「人工改过」台账 ✗：%r" % (W.load_selection(_man_p),))
        check("人工改过" in tab.lst.item(0).text(),
              "列表行该带「人工改过」标记 ✗：%r" % (tab.lst.item(0).text(),))
        tab.cmb_filter.setCurrentIndex(tab.cmb_filter.findData("manual"))
        check(tab.lst.count() == 1, "筛「人工改过」该只剩保存过的 img00 ✗：%d"
              % tab.lst.count())
        tab.cmb_filter.setCurrentIndex(0)
        # ---- ④c 画新框的类别 = 下拉选的（2026-09-30 用户报"我标出来的是怪物"✓）----
        check(tab.canvas.current_cls == 0
              and tab.cmb_cls.currentData() == 0,
              "画新框的类别该跟下拉（0=shape ✗ 原来一直是 CLASS_MOB=1）：%r/%r"
              % (tab.canvas.current_cls, tab.cmb_cls.currentData()))

        # ---- ③d 画布聚焦时 ↑/↓ 切帧（2026-09-30 用户要求 ✓）----
        from PyQt5.QtWidgets import QShortcut
        scs = tab.canvas.findChildren(QShortcut)
        _skeys = sorted(str(sc.key().toString()) for sc in scs)
        check("Up" in _skeys and "Down" in _skeys,
              "画布上该挂 ↑/↓ 切帧快捷键（WidgetShortcut ✓）：%r" % (_skeys,))
        tab.lst.setCurrentRow(1)
        tab._step_frame(+1)
        check(tab.lst.currentRow() == 2, "↓ 该切到下一帧 ✗")
        tab._step_frame(-1)
        tab._step_frame(-1)
        check(tab.lst.currentRow() == 0, "↑ ↑ 该切回第一帧 ✗")
        tab._step_frame(-1)
        check(tab.lst.currentRow() == 0, "首帧再 ↑ 该原地不动（越界钳制 ✗）")

        # ---- ②b 多选 + 一键批量提案（2026-09-30 用户要求 ✓ 推理替身 ✓）----
        from PyQt5.QtWidgets import QListWidget
        check(tab.lst.selectionMode() == QListWidget.ExtendedSelection,
              "列表该是 ExtendedSelection（多选 ✓）")
        import time as _t
        fake_w = root / "fake.pt"
        fake_w.write_bytes(b"")
        tab._yolo_weights = str(fake_w)
        tab._refresh_model_label()
        check("fake.pt" in tab.lbl_model.text(),
              "右边模型显示该跟手选走（2026-09-30 用户要求 ✓）：%r"
              % (tab.lbl_model.text(),))
        tab.lst.clearSelection()              # ⚠ setCurrentRow 会带出隐选中 ✗（当前行）⇒ 先清 ✓
        tab.lst.item(1).setSelected(True)
        tab.lst.item(2).setSelected(True)
        sel = tab.selected_frames()
        check(len(sel) == 2 and sel[0].name == "img01.png",
              "selected_frames 该按列表顺序给多选帧 ✗：%r" % ([p.name for p in sel],))
        before_boxes = [(b[1], b[2]) for b in tab.canvas.get_boxes()]
        real_many = W.yolo_propose_many
        W.yolo_propose_many = lambda w, paths, conf=0.40: {
            str(p): [(0, 10.0, 10.0, 40.0, 40.0)] for p in paths}
        try:
            tab._batch_propose(sel)
            for _ in range(300):              # 轮询泵事件（后台线程 + QTimer ✓）
                app.processEvents()
                if tab.btn_yolo.isEnabled():
                    break
                _t.sleep(0.02)
        finally:
            W.yolo_propose_many = real_many
        check(tab.btn_yolo.isEnabled(), "批量提案跑完该恢复按钮 ✗")
        got1 = W.load_yolo_labels(W.label_path_for(sel[0], root / "labels"), 320, 240)
        check(len(got1) == 2,
              "批量提案该把框**直接写进各帧标签**（1 旧 + 1 新 ✗）：%r" % (got1,))
        check(len(tab.canvas.get_boxes()) == len(before_boxes),
              "批量提案不该动画布（画面只保留上次单选那帧 ✗）")
        check("批量提案" in tab.lbl_state.text(),
              "状态行该汇报批量结果 ✗：%r" % (tab.lbl_state.text(),))
        _man = W.load_selection(_man_p) or []
        check("img00" in _man and "img01" not in _man and "img02" not in _man,
              "批量提案**不算人工** —— 台账不该变 ✗：%r" % (_man,))

        # ---- ②c 设置弹窗（桩回调 ⇒ 绝不碰 config/ui.yaml ✓ §11）----
        calls = []
        dlg = W._WbSettingsDialog(on_mask=lambda a: calls.append(("m", a)),
                                  on_font=lambda p: calls.append(("f", p)),
                                  parent=win)
        check(dlg.sld_mask is not None and dlg.sp_font is not None,
              "设置弹窗该有蒙版滑条与字号 ✗")
        dlg.sp_font.setValue(12)
        check(("f", 12) in calls, "字号改动该走回调（存+应用都归外面 ✓）：%r" % (calls,))
        dlg.close()
        win._apply_font(12)
        check(win.font().pointSize() == 12, "字号该应用到本窗口（子控件继承 ✓）✗")
        win._apply_font(9)

        # ---- ② 训练页「选帧」→ 清单 yaml ----
        tt = win.tab_train
        tt._sel = ["img00"]
        tt._make_yaml()
        lst = (root / "train_frames.txt").read_text(encoding="utf-8").split()
        check(lst == ["./images/img00.png"],
              "选帧清单该只有选中那帧（⭐ 带 ./ 前缀 ✗ —— ultralytics 只对 ./ 开头的行"
              "做相对清单位置的换算，否则按 cwd 解析全 404 ✗）：%r" % (lst,))
        ytxt = (root / "data.yaml").read_text(encoding="utf-8")
        check("train: train_frames.txt" in ytxt,
              "选了帧 ⇒ data.yaml 该指向清单 ✗：%r" % (ytxt[:80],))
        tt._sel = None
        tt._make_yaml()
        ytxt = (root / "data.yaml").read_text(encoding="utf-8")
        check("train: images" in ytxt, "清掉选帧 ⇒ data.yaml 回到整个 images ✗")
        # ---- ②b 清单相对路径（2026-09-30 现场翻车钉 ✓）：图片目录**不叫 images** 时，
        #      条目必须是真实相对路径 —— 原来写死 "images/" 前缀，现场目录叫
        #      liedetectorphotos ⇒ 清单指向不存在的 datasets/images/ ⇒ ultralytics
        #      每张读不到 ⇒ 逐帧 pip 重试 ⇒ 扫描 173s/帧 ✗（38 小时 ✗）。
        (root / "rawphotos").mkdir()
        (root / "rawphotos" / "img00.png").write_bytes(
            (root / "images" / "img00.png").read_bytes())
        win.tab_label.ed_images.setText(str(root / "rawphotos"))
        win.tab_label.reload()
        tt._sel = ["img00"]
        tt._make_yaml()
        lst = (root / "train_frames.txt").read_text(encoding="utf-8").split()
        check(lst == ["./rawphotos/img00.png"],
              "非 images 目录 ⇒ 清单条目该是真实相对路径（./rawphotos/… ✗ 不是 "
              "images/… ✗）：%r" % (lst,))
        win.tab_label.ed_images.setText(str(root / "images"))
        win.tab_label.reload()
        tt._sel = ["img00"]
        tt._make_yaml()

        # 弹窗本体（不 exec_ ✓ 直接造）
        dlg = W._SelFramesDialog(win, root / "images", root / "labels", None)
        check(dlg.total == 3 and len(dlg.selected_stems()) == 3,
              "选帧弹窗默认全选（3 张 ✗）")
        check("人工改过" in dlg.lst.item(0).text()
              and "人工改过" not in dlg.lst.item(1).text(),
              "选帧弹窗行该带「人工改过」标记（只 img00 ✓）✗")
        dlg._uncheck_all()
        check(dlg.selected_stems() == [], "全不选 ⇒ 空 ✗")
        dlg._invert()
        check(len(dlg.selected_stems()) == 3, "反选 ⇒ 又全选 ✗")
        dlg.close()

        # ---- ① YOLO 提案按钮在位（真推理不跑 ✓）----
        check(hasattr(tab, "btn_yolo"), "YOLO 提案按钮该建出来 ✗")
        weird = [p.name for p in (root / "labels").iterdir() if p.is_dir()]
        check(not weird, "close 前：labels 里出现了**目录** ✗：%r" % (weird,))
        win.close()
        weird2 = [p.name for p in (root / "labels").iterdir() if p.is_dir()]
        check(not weird2, "close 后：labels 里出现了**目录** ✗（谁建的？）：%r"
              % (weird2,))
        del win, dlg, tab, tt
        app.processEvents()


def main():
    print("yolo工作台自检：")
    test_pure()
    test_window()
    test_window_features()
    test_settings_dialog_builds()
    if _FAILED:
        print("自检：%d 条失败" % len(_FAILED))
        return 1
    print("yolo工作台自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
