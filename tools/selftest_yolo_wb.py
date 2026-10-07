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
import os
import subprocess
import sys
import tempfile
import time
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


# ══════════════════════════════════════════════════════════
# ⭐⭐⭐⭐⭐ **Job Object：父进程一死（崩溃/强杀）⇒ 整棵树被系统收走**（用户 2026-10-05 ✓）
# ══════════════════════════════════════════════════════════
_NOWIN = 0x08000000          # CREATE_NO_WINDOW（与 `deploy/runner` / `core/procguard` 同一处口径 ✓）
_PJ = {}                     # 记号文件路径（每次用例开头重建 ✓）


def _pj_paths(d):
    return {"father": d / "father_pid.txt", "train": d / "train_pid.txt",
            "gc_up": d / "gc_up.txt", "gc_late": d / "gc_late.txt"}


def _pj_scripts(d, guard):
    """把**三级**脚本写进临时目录 ✓（⚠ 写成文件、**不用** `-c` 一锅串 ✗：三层嵌套的引号+换行
    没人看得懂 ✓）。

    形状**照真实那棵树**搭 ✓（这很重要 ✗ —— 只测一级的话，恰好绕过"孙进程才是最麻烦的"那点 ✓）：
      父亲（= 工作台 ✓）→ 训练（= `QProcess` 那个 ✓）→ **孙进程**（= `ultralytics` 的 dataloader ✓
      —— 它才是"没窗口、还在干活、单杀还不一定干净"的那个 ✓）。
    """
    _gc = d / "gc.py"                                  # 孙进程：报到 ✓ → 睡 6 秒 → 再报到 ✓
    _gc.write_text(
        "import os, time\n"
        "open(r'%s', 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "time.sleep(6)\n"
        "open(r'%s', 'w', encoding='utf-8').write('late')\n"
        % (_PJ["gc_up"], _PJ["gc_late"]), encoding="utf-8")
    _tr = d / "train.py"                               # 训练：报到 ✓ → 叉一个孙进程 ✓ → 一直睡 ✓
    _tr.write_text(
        "import os, subprocess, sys, time\n"
        "open(r'%s', 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "subprocess.Popen([sys.executable, r'%s'])\n"
        "time.sleep(300)\n" % (_PJ["train"], _gc), encoding="utf-8")
    _fa = d / "father.py"                              # 父亲：可选 `guard` ✓ → 起训练 ✓ → 一直睡 ✓
    _fa.write_text(
        "import os, subprocess, sys, time\n"
        "sys.path.insert(0, r'%s')\n"
        "%s"
        "open(r'%s', 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "p = subprocess.Popen([sys.executable, r'%s'])\n"
        "%s"
        "time.sleep(300)\n"
        % (str(Path(__file__).resolve().parent.parent),
           "from core import procguard\n" if guard else "",
           _PJ["father"], _tr,
           "procguard.guard(p.pid)\n" if guard else ""), encoding="utf-8")
    return _fa


def _pj_wait(path, timeout):
    _t0 = time.time()
    while time.time() - _t0 < timeout:
        if path.exists():
            return True
        time.sleep(0.2)
    return False


def _pj_int(path):
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:                                  # noqa: BLE001
        return 0


def _pj_kill(pid, tree=False):
    """杀一个 pid ✓。`tree=False` ⇒ **只杀它自己**（⚠ 这一档才是"**父进程自己死了**"那条路 ✓✓——
    崩溃 / 任务管理器"结束进程"都是这样 ✓）；`tree=True` ⇒ 连树杀（**收尾**用 ✓）。

    ⚠⚠ **踩过的坑**（很值 ✓）：我第一版"强杀父亲"也用了 `/T` ✗ ⇒ **两轮都被我自己连树杀了** ✗
    ⇒ 反面那条立刻红 ✓（它红得**完全正确** ✓）—— 也就是说：如果不带这个反面，
    这条用例会**假绿**（还以为 job 在起作用 ✗，其实是 `taskkill /T` 干的 ✓）。
    """
    if pid:
        _cmd = ["taskkill", "/F", "/PID", str(int(pid))]
        if tree:
            _cmd.insert(1, "/T")
        subprocess.run(_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=_NOWIN)


def test_procguard_kills_tree():
    """⭐⭐⭐⭐⭐ **父进程一死（崩溃 / 强杀）⇒ 整棵树被系统收走**（用户 2026-10-05 ✓ 原话：
    "**把训练/验证子进程塞进一个 Job Object（KILL_ON_JOB_CLOSE ✓）⇒ 父进程一死（崩溃、强杀
    都算 ✓）整棵树被系统收走**" ✓✓ 见 `core/procguard.py` ✓）。

    ⚠⚠ **实测病根**（用户问"有没有杀不掉的死进程" ✓）：`gui/yolo_workbench.py` 原来**关窗根本
      不管**训练子进程 ✗ ⇒ 父进程退了它**照样在干活** ✗✗；而 `ultralytics` 自己还会叉
      **dataloader 孙进程** ✗ ⇒ 留下一棵**看不见的进程树** ✓（没窗口、显存占着 ✓）= 用户说的
      "杀不掉" ✓。

    钉两条（一正一反 ✓）：
      ① **父亲 `guard` 过** ⇒ `taskkill /F` **强杀父亲**（= 崩溃那条路 ✓ 走不到 `closeEvent` ✓）
         ⇒ **孙进程永远不写"还在干活"** ✓（= 内核把整棵树收走了 ✓；
         ⚠ 没这套 ⇒ 它会写 ⇒ 这条红 ✗✗）；
      ② **反面：父亲没 `guard`** ⇒ 同一个强杀 ⇒ 孙进程**照写** ✓（= 证明"**是 job 在起作用**" ✓
         —— ⚠ 一刀切以为"`taskkill` 会连树杀"就错了 ✗：`/T` 只在**按父 pid 杀**时才连树，
         而"**父进程自己死掉**"这件事**没人去杀** ✓）。
    ⚠ 非 Windows / 本机 `AssignProcessToJobObject` 不可用 ⇒ **跳过** ✓（退回老行为，不假绿 ✓）。
    ⚠ 夹具**自己不留尾巴** ✓：两轮跑完都把"父亲/训练/孙进程"三个 pid 连根清一遍 ✓。
    """
    if os.name != "nt":
        check(True, "跳过 Job Object 用例（非 Windows ⇒ `procguard` 空转 ✓）")
        return
    import core.procguard as PG
    global _PJ
    # ⚠ 先**用一个扔掉的子进程**试这套 API 在不在（**绝不能拿自己试** ✗：那会把自己也塞进 job ✓）
    _pr = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=_NOWIN)
    _usable = PG.guard(_pr.pid)
    _pr.wait(timeout=10)
    if not _usable:
        check(True, "跳过 Job Object 用例（本机 `AssignProcessToJobObject` 不可用 ⇒ 退回老行为 ✓）")
        return

    def _run(guard):
        global _PJ                                  # ⚠ 少了这句 ⇒ `_pj_scripts` 读到的还是空字典 ✗（踩过 ✓）
        _d = Path(tempfile.mkdtemp(prefix="pj_"))
        _PJ = _pj_paths(_d)
        _fa = _pj_scripts(_d, guard)
        _f = subprocess.Popen([sys.executable, str(_fa)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              creationflags=_NOWIN)
        if not _pj_wait(_PJ["gc_up"], 25.0):            # 孙进程报到（首次 import 可能慢 ✓）
            for _k in ("father", "train", "gc_up"):
                _pj_kill(_pj_int(_PJ[_k]))
            _pj_kill(_f.pid)
            return None
        # ⚠ **强杀父亲**（= 崩溃 / 任务管理器"结束进程" ✓ —— 走不到 `closeEvent` ✓）——
        #   ⚠⚠ **必须只杀它自己**（`tree=False` ✓ 不带 `/T` ✗✗）：真实那条路**没有人**连树杀 ✓
        _pj_kill(_f.pid, tree=False)
        time.sleep(8.0)                                 # 孙进程 6 秒后才写「还在干活」✓
        _late = _PJ["gc_late"].exists()
        _pids = [_pj_int(_PJ[_k]) for _k in ("father", "train", "gc_up")]
        for _p in _pids + [_f.pid]:                     # 收尾：**连根清** ✓（这里才用 `/T` ✓）
            _pj_kill(_p, tree=True)
        return _late, _pids

    _g = _run(True)
    check(_g is not None and not _g[0],
          "① **父亲 `guard` 过 ⇒ 强杀父亲 ⇒ 孙进程不再干活** ✓（实测「还在干活」= %s ✓〔要 False ✓〕；"
          "涉及 pid %s ✓）—— ⚠ 没这套 ⇒ 它会写 ⇒ 这条红 ✗✗（= 用户报的那棵看不见的进程树 ✓）"
          % ("-" if _g is None else _g[0], "-" if _g is None else _g[1]))
    _n = _run(False)
    check(_n is not None and _n[0],
          "② **反面：父亲没 `guard` ⇒ 孙进程照旧在干活** ✓（实测「还在干活」= %s ✓〔要 True ✓〕；"
          "pid %s ✓ —— ⚠ 这正是修前的行为 ✓ = 证明**是 job 在起作用** ✓，不是 `taskkill` 自己连树"
          " ✗：`/T` 只在按父 pid 杀时才连树，而「父进程自己死掉」这件事没人去杀 ✓）"
          % ("-" if _n is None else _n[0], "-" if _n is None else _n[1]))


def test_procguard_wired():
    """⭐⭐⭐⭐⭐ **"栽进 job"与"连树杀"在界面上真的接上了**（用户 2026-10-05 ✓）。

    ⚠ 为什么盯**源码**✗（不是行为 ✗）：真跑一遍训练要权重 + 数据集 + 几分钟 ✓（本套件
      `test_window` 开头就写明"不跑训练"✓）⇒ 走**源码级**那一套 ✓（同 `selftest_lie_demo.
      test_white_area_wired` ✓ 的手法 ✓）—— 钉的是"**接口还在不在**"✓，
      行为那一半由上面 `test_procguard_kills_tree` 量 ✓。
    ⚠ 四条缺一不可 ✓：
      ① 训练页 `QProcess` 起来时**栽进 job** ✓；
      ② 验证页（**另一个类** ✗）也得栽 ✓（所以 `guard_qprocess` 抽成模块级函数 ✓）；
      ③ 「停止」按钮要**连树**杀 ✓（⚠ 退回 `self._proc.kill()` ⇒ 只杀直接孩子 ⇒ dataloader
         孙进程留下 ✗✗ = 用户报的那条 ✓）；
      ④ 关窗要**问一句 + 连树杀 + `release()` 清场** ✓（⚠ 只做 ① 不做 ④ ⇒ **正常关窗**时
         子进程还在跑 ✗（job 只在**本进程死掉**时才兜底 ✓ —— 而工作台关窗后本进程就退 ✓，
         所以 ④ 主要防"窗口关了但进程还在"那种情形 ✓，也算把话说清楚 ✓）。
    """
    _src = Path(W.__file__).read_text(encoding="utf-8")
    check("self._proc.started.connect(lambda: guard_qprocess(self._proc))" in _src,
          "① **训练页起来就栽进 job** ✓（`_proc.started → guard_qprocess` ✓ —— ⚠ 删掉这句 ⇒ 立刻红 ✗）")
    check("self._vproc.started.connect(lambda: guard_qprocess(self._vproc))" in _src,
          "② **验证页也栽了** ✓（⚠ 它是**另一个类** ✗ ⇒ `guard_qprocess` 必须是模块级的 ✓）")
    check("procguard.kill_tree(int(self._proc.processId()))" in _src,
          "③ **「停止」连树杀** ✓（`kill_tree` ＋ `kill()` 兜底 ✓ —— ⚠ 退回只 `kill()` ⇒ "
          "dataloader 孙进程留下 ✗✗）")
    check("procguard.kill_tree(int(_p.processId()))" in _src and "procguard.release()" in _src
          and "还在跑" in _src and "e.ignore()" in _src,
          "④ **关窗：问一句 ＋ 连树杀 ＋ `release()` 清场** ✓（缺一样 ⇒ 红 ✗ —— "
          "`release()` 是把 job 里剩下的一网打尽 ✓）")


def main():
    print("yolo工作台自检：")
    test_pure()
    test_window()
    test_window_features()
    test_settings_dialog_builds()
    test_procguard_kills_tree()
    test_procguard_wired()
    if _FAILED:
        print("自检：%d 条失败" % len(_FAILED))
        return 1
    print("yolo工作台自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
