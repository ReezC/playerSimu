# -*- coding: utf-8 -*-
"""轻量版通用 **YOLO 工作台**（标注 → 训练 → 验证；**不绑定本项目** ✓）。

仿照工作台「模型训练」页签的工作流与 GUI 结构（用户 2026-09-29 ✓），但**独立、
轻量、通用**：任何"一组图 + YOLO 格式标签"都能用 —— 不依赖项目的地形/集合/标定。

    python -m gui.yolo_workbench [数据集根目录]

三个页签（= 一次完整的工作流）：
    ① **标注**：浏览图片 → 在画布上改框（复用质检台同款 `ImageCanvas` ✓）→
       存 YOLO 格式标签；「自动提案」经典 CV、「YOLO 提案」拿训好的权重推理（补漏 ✓）；
       列表带**筛选 / 排序** ✓、画布带**复制 / 粘贴 / 撤销** ✓（同质检台的交互 ✓）。
    ② **训练**：自动生成 `data.yaml` →「选帧」指定哪些帧参与 ✓ → 挑基础权重 /
       轮数 / 尺寸 / 批大小 → 跑 ultralytics（QProcess，日志实时流进来 ✓）。
    ③ **验证**：对练好的权重跑 val + 用**检测预览**把结果画在图上人眼看 ✓。

分工纪律（同工作台 ✓）：**界面只编排**，纯逻辑（yaml 生成 / 标签读写 / 数据集扫描 /
提案）都是模块级函数 ⇒ 能脱离 Qt 自检 ✓（`tools/selftest_yolo_wb.py` ✓）。
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QProcess, Qt, QTimer
from PyQt5.QtGui import QFont, QKeySequence, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPlainTextEdit,
    QMessageBox, QProgressBar, QPushButton, QShortcut, QSplitter, QTabWidget,
    QVBoxLayout, QWidget)

from gui import theme
from gui.canvas import ImageCanvas
from gui.widgets import NoWheelComboBox, NoWheelSlider, NoWheelSpinBox

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ══════════════════════════════════════════════════════════
# 纯逻辑（能脱离 Qt 自检 ✓）
# ══════════════════════════════════════════════════════════
def scan_images(img_dir):
    """数据集图片目录 → 排序后的图片路径列表（只认常见扩展 ✓）。"""
    d = Path(img_dir)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def label_path_for(img_path, lbl_dir):
    """图片路径 → 对应的 YOLO 标签路径（同词干 .txt ✓）。"""
    return Path(lbl_dir) / (Path(img_path).stem + ".txt")


def load_yolo_labels(txt_path, w, h):
    """读 YOLO 标签 → [(cls, cx, cy, bw, bh)]（**像素**坐标 ✓ 坏行跳过 ✓）。"""
    out = []
    p = Path(txt_path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 5:
            continue
        try:
            cls, nx, ny, nw, nh = (int(parts[0]), float(parts[1]), float(parts[2]),
                                   float(parts[3]), float(parts[4]))
        except ValueError:
            continue                          # 坏行跳过（别让一行坏数据炸掉整帧 ✗）
        out.append((cls, nx * w, ny * h, nw * w, nh * h))
    return out


def save_yolo_labels(txt_path, boxes, w, h):
    """[(cls, cx, cy, bw, bh)]（像素）→ YOLO 归一化 txt（**原子写** ✓）。"""
    p = Path(txt_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["%d %.6f %.6f %.6f %.6f"
             % (int(cls), cx / w, cy / h, bw / w, bh / h)
             for cls, cx, cy, bw, bh in boxes]
    tmp = p.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp, p)                        # 原子替换（半截文件不会出现 ✓）


def make_data_yaml(root, names, subset=None):
    """数据集根 + 类别名 ⇒ `data.yaml` 文本（YOLO 标准结构：images/ labels/ ✓）。

    ⭐ `subset`（2026-09-30 加 ✓）：**选帧清单**文件名（如 `"train_frames.txt"`，
    每行一条相对 `path` 的图片路径 ✓）—— 给了它 train/val 就**只吃清单里那几帧** ✓
    （用户要的「分数据集」步骤：指定哪些帧参与训练 ✓）；不给 = 老行为
    （train/val 都吃整个 images 目录 ✓）。
    """
    names_lines = "\n".join("  %d: %s" % (i, n) for i, n in enumerate(names))
    tv = subset if subset else "images"
    return ("path: %s\ntrain: %s\nval: %s\n\nnames:\n%s\n"
            % (Path(root).resolve().as_posix(), tv, tv, names_lines))


def propose_boxes(img):
    """经典 CV 自动提案：高通凸显淡轮廓 → 连通域 ⇒ [(cx, cy, w, h)] 像素框。

    ⭐ **宁多勿漏**（提案只求召回 ✓ 精度由人工删改 ✓）；换个数据集形态差得远时，
    调这里的阈值（或以后换成"训练过的模型低 conf 提案" ✓ 接口不变 ✓）。
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hp = cv2.subtract(g, cv2.GaussianBlur(g, (21, 21), 0))
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)
    _, bw = cv2.threshold(hp, 65, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          np.ones((3, 3), np.uint8), iterations=1)
    out = []
    res = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cs = res[0] if isinstance(res, tuple) else res
    H, W = g.shape
    for c in cs:
        arr = np.asarray(c, dtype=np.float32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(arr)
        if not (24 <= w <= W * 0.4 and 24 <= h <= H * 0.4):
            continue                          # 太小 = 纹理碎边；太大 = 整片噪声 ✗
        if max(w, h) / max(1.0, min(w, h)) > 4.5:
            continue                          # 细长条多为伪影 ✗
        out.append((x + w / 2.0, y + h / 2.0, float(w), float(h)))
    return out


def find_weights(root):
    """数据集根下的训练产物：runs/detect/*/weights/best.pt（新→旧 ✓）。"""
    d = Path(root) / "runs" / "detect"
    if not d.is_dir():
        return []
    return sorted(d.glob("*/weights/best.pt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def iou_xywh(a, b):
    """两个左上角框 `(x, y, w, h)` 的 IoU（无重叠 = 0 ✓）。"""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0.0 else 0.0


def dedup_proposals(existing, proposals, iou_thr=0.30):
    """提案去重：与**已有框** IoU > `iou_thr` 的提案丢掉（同 `yolo_augment` 口径 ✓）。

    两边都是 `[(cls, x, y, w, h)]`（像素、左上角 ✓）—— YOLO 提案只**补漏**，
    跟人工框重叠大的没意义 ✓；提案之间的重复不管（很少见、人工删一下就行 ✓）。
    """
    return [p for p in proposals
            if not any(iou_xywh(p[1:], e[1:]) > iou_thr for e in existing)]


def count_label_lines(txt_path):
    """标注 txt 的**有效框行数**（文件不存在 = 0 ✓ 坏行不算 ✓）。"""
    p = Path(txt_path)
    if not p.is_file():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 5:
            continue
        try:
            int(parts[0])
            for v in parts[1:5]:
                float(v)
        except ValueError:
            continue
        n += 1
    return n


def load_selection(path):
    """读「参与训练的帧」台账（json `{"stems": [...]}`）。读不到 ⇒ `None`（全部 ✓）。"""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        stems = [str(s) for s in d.get("stems") or []]
    except Exception:                     # noqa: BLE001 —— 坏台账当没存过 ✓
        return None
    return stems or None


def save_selection(path, stems):
    """存「参与训练的帧」台账（**原子写** ✓ 同 `save_yolo_labels` 的手法 ✓）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"stems": list(stems)}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, p)


def manual_ledger_path(labels_dir):
    """「人工改过」台账的位置：`labels/_manual.json`（下划线开头 ⇒ 不会被当成
    任何一帧的 YOLO 标签 ✓，跟着标签目录走 ✓）。结构复用 `load_selection`
    （`{"stems": [...]}` ✓ 一份读写代码两处用 ✓）。"""
    return Path(labels_dir) / "_manual.json"


def apply_selection(images, stems):
    """按台账过滤图片列表（**保持扫描序** ✓；`stems` 为 None/空 = 全部 ✓）。"""
    if not stems:
        return list(images)
    want = set(stems)
    return [p for p in images if p.stem in want]


#: 推理**子进程**的执行体（2026-09-30 ✓）：torch 的 c10.dll 在「**先 Qt 后 torch**」的
#: 进程里初始化失败（实测 WinError 1114 ✗）—— 工作台进程 import PyQt5 在前 ⇒ 进程内
#: `from ultralytics import YOLO` 必炸 ✗。跟训练页 QProcess 同款办法：**单独进程跑** ✓
#: （参数 = 权重 / conf / 图片…，stdout 最后一行 = JSON `[[图, [[cls,cx,cy,w,h]…]]…]` ✓）。
_YOLO_CHILD = (
    "import sys, json\n"
    "from ultralytics import YOLO\n"
    "m = YOLO(sys.argv[1])\n"
    "conf = float(sys.argv[2])\n"
    "out = []\n"
    "for p in sys.argv[3:]:\n"
    "    r = m.predict(p, conf=conf, verbose=False)[0]\n"
    "    boxes = [[int(b.cls[0])] + [float(v) for v in b.xywh[0]]\n"
    "             for b in (r.boxes or [])]\n"
    "    out.append([p, boxes])\n"
    "print(json.dumps(out))\n"
)


def yolo_propose_many(weights, img_paths, conf=0.40, chunk=200):
    """**多帧**提案：一次子进程跑一批（模型只加载一次 ✓ —— 2026-09-30 用户
    要的效率 ✓ 原话："YOLO 提案每帧都要重选权重吗？这样效率也太低了"）。

    返回 `{路径字符串: [(cls, x, y, w, h)]}`（像素、左上角 ✓）。`chunk` 分批防
    Windows 命令行超长（每批 200 帧 ✓ 批间模型重载 —— 上千帧略慢 ✓）。
    ⭐ 推理仍在**子进程**（`_YOLO_CHILD` ✓ Qt→torch DLL 那条 ✓）；没装 / 权重坏 ⇒
    异常往上抛 ✓。
    """
    import subprocess
    paths = [str(p) for p in img_paths]
    out = {}
    step = max(1, int(chunk))
    for i in range(0, len(paths), step):
        part = paths[i:i + step]
        proc = subprocess.run(
            [sys.executable, "-c", _YOLO_CHILD, str(weights), str(conf)] + part,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT))
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "").strip()[-400:]
                               or "推理子进程失败（rc=%d）" % proc.returncode)
        lines = (proc.stdout or "").strip().splitlines()
        if not lines:
            raise RuntimeError("推理子进程没有输出")
        for p, boxes in json.loads(lines[-1]):
            out[p] = [(int(c), cx - w / 2.0, cy - h / 2.0, w, h)
                      for c, cx, cy, w, h in boxes]
    return out


def yolo_propose_boxes(weights, img_path, conf=0.40):
    """拿训好的权重对**一张图**推理 ⇒ `[(cls, x, y, w, h)]`（像素、左上角 ✓）。

    （单帧便利封装 = `yolo_propose_many` 取那一帧 ✓ 子进程口径同 ✓ —— Qt 先于
    torch 的进程里 c10.dll 会炸 ✗ 所以真推理永远在子进程 ✓。）
    """
    return yolo_propose_many(weights, [img_path], conf)[str(img_path)]

_YOLO_PLOT_CHILD = (
    "import sys, json, base64, cv2\n"
    "from ultralytics import YOLO\n"
    "m = YOLO(sys.argv[1])\n"
    "out = []\n"
    "for p in sys.argv[2:]:\n"
    "    r = m.predict(p, conf=0.25, verbose=False)[0]\n"
    "    ok, buf = cv2.imencode('.jpg', r.plot())\n"
    "    out.append([p, base64.b64encode(buf).decode()])\n"
    "print(json.dumps(out))\n"
)


def yolo_plot_many(weights, img_paths):
    """检测预览：子进程跑推理、把**画好框的图**带回来 → `[(路径, BGR)]` ✓。

    ⭐ 必须在**子进程**跑（进程内 import ultralytics 会撞 Qt→torch 的 c10.dll ✗ ——
    150.3 那条；旧 `_predict` 就是进程内 ✗ 一并收编 ✓）。"""
    import base64
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-c", _YOLO_PLOT_CHILD, str(weights)]
        + [str(p) for p in img_paths],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "").strip()[-400:]
                           or "推理子进程失败（rc=%d）" % proc.returncode)
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        raise RuntimeError("推理子进程没有输出")
    import cv2
    import numpy as np
    out = []
    for p, b64 in json.loads(lines[-1]):
        buf = np.frombuffer(base64.b64decode(b64), np.uint8)
        out.append((Path(p), cv2.imdecode(buf, cv2.IMREAD_COLOR)))
    return out


# ══════════════════════════════════════════════════════════
# ① 标注
# ══════════════════════════════════════════════════════════
class _LabelTab(QWidget):
    """标注：图片列表 + 画布改框（复用质检台同款 `ImageCanvas` ✓）+ 自动提案。"""

    def __init__(self, names_getter, parent=None):
        super().__init__(parent)
        self._names_getter = names_getter     # 类别名从窗口顶上那格取（训练页共用 ✓）
        self._dirty = False
        self._cur = None                      # 当前图片路径（Path）
        self._frames = []                     # [(Path, 框数, 标注mtime)]（筛选/排序 ✓）
        self._clipboard = []                  # 复制的框（进程内 ✓ 同质检台 ✓）
        self._yolo_weights = None             # 手选的提案权重（本会话 ✓）
        self._pending_frame = ""              # 会话恢复：scan 收尾要选回的帧 ✓
        self._manual = set()                  # 人工改过的帧（台账 ✓ 见 _save ✓）
        self._task = None                     # 后台任务（载入 / 批量提案 ✓ §5：活在子线程 ✓）
        self._task_timer = QTimer(self)
        self._task_timer.setInterval(150)
        self._task_timer.timeout.connect(self._poll_task)
        self._undo = []                       # [(stem, 快照)] 上限 100 ✓ 同质检台 ✓
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ---- 数据集（一行一件事 ✓ §4）----
        form = QFormLayout()
        form.setSpacing(6)
        self.ed_images = QLineEdit()
        self.ed_images.setPlaceholderText(r"例如 D:\data\xmx\images")
        btn_img = QPushButton("浏览…")
        btn_img.clicked.connect(self._pick_images)
        row_i = QHBoxLayout()
        row_i.addWidget(self.ed_images, 1)
        row_i.addWidget(btn_img)
        form.addRow("图片目录", row_i)
        self.ed_labels = QLineEdit()
        self.ed_labels.setPlaceholderText("留空 = 图片目录旁的 labels（YOLO 惯例 ✓）")
        form.addRow("标签目录", self.ed_labels)
        root.addLayout(form)
        tip_ds = ("**图片目录**：要标注的图片（png/jpg/bmp ✓）；\n"
                  "**标签目录**：YOLO 格式 txt（每图一份、同词干 ✓）—— 留空就用\n"
                  "图片目录**旁边**的 labels ✓。改完目录点「刷新列表」✓。")
        for w in (self.ed_images, self.ed_labels):
            w.setToolTip(tip_ds)

        brow = QHBoxLayout()
        self.btn_reload = QPushButton("刷新列表")
        self.btn_reload.clicked.connect(self.reload)
        brow.addWidget(self.btn_reload)
        self.btn_propose = QPushButton("自动提案")
        self.btn_propose.setToolTip(
            "用经典 CV 在**当前这张图**上铺一遍候选框（**宁多勿漏** ✓）——\n"
            "框会直接加进画布，人工删改后再保存 ✓。")
        self.btn_propose.clicked.connect(self._on_propose)
        brow.addWidget(self.btn_propose)
        self.btn_yolo = QPushButton("YOLO 提案")
        self.btn_yolo.setToolTip(
            "拿**训好的权重**（runs/detect 里新→旧取第一个 ✓）对**当前这张图**推理，\n"
            "预测框直接铺上画布（conf=0.40；与已有框重叠 IoU>0.30 的丢掉 ✓）——\n"
            "人工删改后再保存 ✓。没训过会弹窗让你挑权重（或先去「训练」页 ✓）。")
        self.btn_yolo.clicked.connect(self._on_yolo_propose)
        brow.addWidget(self.btn_yolo)
        # ⭐⭐ **模型显示 + 选择在右边**（2026-09-30 用户要求 ✓ 原话："我想改权重
        #   怎么办呢？在右边加个模型选择当前模型显示与模型选择按钮，提案按钮只能
        #   管提案"）—— **换权重**在右边这颗 ✓；提案只管提案 ✓（没选 ⇒ 状态行
        #   提示 ✓ 不再从提案按钮里弹文件框 ✗）。
        brow.addStretch(1)
        self.lbl_model = QLabel("")
        self.lbl_model.setStyleSheet("color: #5f6368;")
        self.lbl_model.setMinimumWidth(140)
        brow.addWidget(self.lbl_model)
        self.btn_model = QPushButton("选择模型…")
        self.btn_model.setToolTip(
            "选 YOLO 提案用的权重（*.pt ✓）—— 选一次记整场 ✓；\n"
            "没手选时自动用 数据集根/runs/detect 里**最新**的 ✓。")
        self.btn_model.clicked.connect(self._pick_model)
        brow.addWidget(self.btn_model)
        root.addLayout(brow)

        # ---- 筛选 + 排序（仿质检台的筛选 ✓ 用户 2026-09-30 要的 ③）----
        frow = QHBoxLayout()
        frow.setSpacing(6)
        frow.addWidget(QLabel("只看"))
        self.cmb_filter = NoWheelComboBox()
        for text, key in (("全部", "all"), ("空帧（没标注）", "empty"),
                          ("有框", "has"), ("多框（≥8）", "many"),
                          ("人工改过", "manual")):
            self.cmb_filter.addItem(text, key)
        self.cmb_filter.setToolTip("按**当前标注**筛（存一帧就跟着变 ✓）—— 只动显示 ✓。")
        self.cmb_filter.currentIndexChanged.connect(self._rebuild_list)
        frow.addWidget(self.cmb_filter)
        frow.addWidget(QLabel("排序"))
        self.cmb_sort = NoWheelComboBox()
        for text, key in (("按名称", "name"), ("按标注时间（新→旧）", "mtime"),
                          ("按框数（少→多）", "boxes"),
                          ("按创建时间（早→晚）", "ctime")):
            self.cmb_sort.addItem(text, key)
        self.cmb_sort.setToolTip(
            "按名称 = 字典序 ✓；按标注时间 = 最近存过的在前（接着标最顺手 ✓）；\n"
            "按框数 = 空帧最前（补标最急的先看到 ✓）。")
        self.cmb_sort.currentIndexChanged.connect(self._rebuild_list)
        frow.addWidget(self.cmb_sort)
        frow.addStretch(1)
        root.addLayout(frow)

        # ---- 列表 + 画布（左右分栏 ✓）----
        split = QSplitter(Qt.Horizontal)
        self.lst = QListWidget()
        self.lst.setSelectionMode(QListWidget.ExtendedSelection)
        self.lst.setToolTip(
            "点选单帧查看 ✓；**Ctrl/Shift 多选**后点「YOLO 提案」＝\n"
            "**一键多帧提案**（结果直接写进各帧标签 ✓ —— 画布只保留上次单选那帧 ✓）。\n"
            "**人工改过** = 在画布上保存过（区别于纯批量提案的帧 ✓）——\n"
            "「只看」里能单独筛出它们 ✓。")
        self.lst.currentItemChanged.connect(self._on_pick_frame)
        split.addWidget(self.lst)
        self.canvas = ImageCanvas()
        self.canvas.boxes_changed.connect(self._mark_dirty)
        # ⭐⭐ **画布聚焦时 ↑/↓ 也切帧**（2026-09-30 用户要求 ✓ 原话："聚焦标注编辑区时
        #   快捷键 ↑↓ 也要能切换选中帧"）—— 与列表的原生 ↑/↓ 同语义（↓ = 下一帧、
        #   ↑ = 上一帧 ✓，按**当前列表顺序**＝筛选/排序后的所见顺序 ✓）。
        #   ⚠ `WidgetShortcut` 只在**画布聚焦**时生效 ✓（§10：焦点范围只在那一处 ✓，
        #     不抢列表自己的 ↑/↓ ✓，更不抢全局 ✓）；切帧走 `setCurrentRow` ⇒
        #     `_on_pick_frame` 的自动保存照旧 ✓（旧工作不白费 ✓）。
        #   ⚠ 画布自己没有别的 ↑/↓ 用途（缩放是滚轮/中键 ✓ Del/Backspace 是删框 ✓）
        #     ⇒ 无冲突 ✓。
        for _key, _delta in ((Qt.Key_Up, -1), (Qt.Key_Down, +1)):
            _sc = QShortcut(QKeySequence(_key), self.canvas)
            _sc.setContext(Qt.WidgetShortcut)
            _sc.activated.connect(lambda _d=_delta: self._step_frame(_d))
        # ⭐ 复制/粘贴/撤销 + 撤销快照（仿质检台 ✓ 用户 2026-09-30 要的 ④）——
        #   画布的信号三件套一直在发（`canvas.py` keyPressEvent ✓），这边一直没接 ✗。
        self.canvas.before_change.connect(self._on_before_change)
        self.canvas.copy_requested.connect(self._copy)
        self.canvas.paste_requested.connect(self._paste)
        self.canvas.undo_requested.connect(self._undo_edit)
        split.addWidget(self.canvas)
        split.setSizes([220, 620])
        root.addWidget(split, 1)

        # ---- 保存行 ----
        srow = QHBoxLayout()
        self.cmb_cls = NoWheelComboBox()
        self.cmb_cls.setToolTip("新拉框的类别（名字在窗口顶上那格配 ✓）")
        srow.addWidget(QLabel("类别"))
        srow.addWidget(self.cmb_cls)
        # ⭐⭐ **画新框的类别 = 下拉选的**（2026-09-30 用户报 ✓ 原话："为什么 YOLO
        #   提案标的框是默认的 shape 而我标出来的是怪物"）—— `canvas.current_cls`
        #   原来一直是质检台的默认 `CLASS_MOB`(1) ✗ **没人同步过** ⇒ 手拉的框全是
        #   类别 1（存成 cls 1、画布显示"怪物"）✗。切下拉立即生效 ✓。
        def _cls_changed(_i=0):
            d = self.cmb_cls.currentData()
            if d is not None:
                self.canvas.current_cls = d
        self.cmb_cls.currentIndexChanged.connect(_cls_changed)
        self.btn_save = QPushButton("保存本帧")
        self.btn_save.setShortcut(QKeySequence.Save)
        self.btn_save.clicked.connect(self._save)
        srow.addWidget(self.btn_save)
        self.lbl_state = QLabel("")
        self.lbl_state.setStyleSheet("color: #5f6368;")
        srow.addWidget(self.lbl_state, 1)
        root.addLayout(srow)
        # ---- 过程条 + 日志（2026-09-30 用户要求 ✓："只要是有处理过程的行为都需要
        #      进度条"、"需要日志区"）—— 进度条只在干活时露面 ✓；日志**只追加**、
        #      自动滚到最新 ✓、超长截头（几百行封顶 ✗ 别无限吃内存 ✓）。
        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        self.pbar.setFormat("%v / %m")
        self.pbar.setToolTip("后台任务（载入 / 批量提案 ✓）的进度 —— 干完自己收起来 ✓。")
        root.addWidget(self.pbar)
        self.log = QPlainTextEdit()           # Log style (data-wb same)
        self.log.setObjectName("Log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setTextInteractionFlags(Qt.TextSelectableByMouse
                                         | Qt.TextSelectableByKeyboard)
        self.log.setMaximumHeight(120)
        self.log.setToolTip(
            "处理历史：载入 / 提案（单帧与批量）/ 保存 / 失败，全在这 ✓ ——\n"
            "带时间戳、只追加、自动滚到最新 ✓；训练页/验证页各有自己的日志 ✓。")
        root.addWidget(self.log)

    def _pick_images(self):
        d = QFileDialog.getExistingDirectory(self, "选图片目录",
                                             self.ed_images.text() or "")
        if d:
            self.ed_images.setText(d)
            self.reload()

    def labels_dir(self):
        if self.ed_labels.text().strip():
            return Path(self.ed_labels.text().strip())
        img = self.ed_images.text().strip()
        return Path(img).parent / "labels" if img else None

    def reload(self):
        """载入数据集 —— **后台线程 + 进度条**（2026-09-30 用户要求 ✓："只要是有处理
        过程的行为都需要进度条"）—— 2.6 万张的目录全量 stat/读标签要几秒 ✗ 同步跑就是
        "点了没反应" ✗。跑完**恢复上次的当前帧** ✓（旧工作不白费 ✓）。"""
        if self._task is not None:
            self._log("上一个任务还在跑 —— 等它结束再刷新 ✓")
            return
        self._save()                          # 换数据集前先把手上的存了 ✓（旧工作不白费 ✓）
        self._cur = None
        self._undo.clear()                    # 撤销栈跨数据集没意义（stem 会撞 ✗）
        self._frames = []
        self.lst.clear()
        cur_it = self.lst.currentItem()
        prev = str(cur_it.data(Qt.UserRole) or "") if cur_it is not None else None
        paths = scan_images(self.ed_images.text().strip())
        lbl_dir = self.labels_dir()
        self._task = {"kind": "scan", "paths": paths, "i": 0, "rows": [],
                      "done": False, "err": None, "lbl_dir": lbl_dir, "prev": prev}
        self.pbar.setVisible(True)
        self.pbar.setRange(0, max(1, len(paths)))
        self.pbar.setValue(0)
        self.btn_reload.setEnabled(False)
        self.lbl_state.setText("载入中… 0 / %d" % len(paths))
        self._log("载入数据集：%s（%d 张）"
                  % (self.ed_images.text().strip() or "（空）", len(paths)))

        def _work():
            t = self._task
            rows = []
            try:
                for i, p in enumerate(t["paths"]):
                    lp = label_path_for(p, t["lbl_dir"]) if t["lbl_dir"] else None
                    try:
                        mt = lp.stat().st_mtime if (lp is not None and lp.is_file()) \
                            else 0.0
                    except OSError:
                        mt = 0.0
                    try:
                        ct = p.stat().st_ctime   # Windows = 创建时间 ✓（同资源管理器 ✓）
                    except OSError:
                        ct = 0.0
                    rows.append([p, count_label_lines(lp) if lp else 0, mt, ct])
                    t["i"] = i + 1               # 进度（只写数字 ✓ §5：UI 只读 ✓）
                t["manual"] = (sorted(load_selection(
                    manual_ledger_path(t["lbl_dir"])) or [])
                    if t["lbl_dir"] else [])
                t["rows"] = rows
            except Exception as e:               # noqa: BLE001 —— 收集给主线程弹 ✓
                t["err"] = str(e)
            t["done"] = True

        threading.Thread(target=_work, daemon=True).start()
        self._task_timer.start()

    def _rebuild_list(self):
        """按「只看」筛 + 「排序」重画列表（只动**显示** ✓ 仿质检台/选帧 ✓）。

        ⭐ 真名存在 `Qt.UserRole`、行文本只是展示（带框数 ✓）—— `_on_pick_frame`
        从 UserRole 拿名，别拿 text 拼（会被后缀骗 ✗）。
        """
        if not hasattr(self, "cmb_filter"):
            return
        mode = self.cmb_filter.currentData() or "all"
        key = self.cmb_sort.currentData() or "name"
        rows = list(self._frames)
        if mode == "empty":
            rows = [f for f in rows if f[1] == 0]
        elif mode == "has":
            rows = [f for f in rows if f[1] > 0]
        elif mode == "many":
            rows = [f for f in rows if f[1] >= 8]
        elif mode == "manual":
            rows = [f for f in rows if f[0].stem in self._manual]
        if key == "mtime":
            rows.sort(key=lambda f: f[2], reverse=True)   # 最近标注在前 ✓
        elif key == "boxes":
            rows.sort(key=lambda f: (f[1], f[0].name))    # 空帧最前（补标急 ✓）
        elif key == "ctime":
            rows.sort(key=lambda f: (f[3], f[0].name))    # 早→晚（资源管理器默认 ✓）
        else:
            rows.sort(key=lambda f: f[0].name)
        self.lst.blockSignals(True)
        self.lst.clear()
        for p, n, _mt, _ct in rows:
            it = QListWidgetItem(self._row_text(p.name, n))
            it.setData(Qt.UserRole, p.name)
            self.lst.addItem(it)
        self.lst.blockSignals(False)
        self.lbl_state.setText("共 %d 张（显示 %d）" % (len(self._frames), len(rows)))

    def _row_text(self, name, n):
        """列表行文案：`名字　N框`＋**人工改过**标记 ✓（2026-09-30 用户要求 ✓
        —— 台账记的是"这帧在画布上保存过"，与纯批量提案区分开 ✓）。"""
        mark = ("　人工改过" if Path(name).stem in getattr(self, "_manual", ())
                else "")
        return "%s　%d框%s" % (name, n, mark)

    def _step_frame(self, delta):
        """↑/↓ 切换选中帧（±1 行 ✓）：越界原地不动 ✓；走 `setCurrentRow` ⇒
        `_on_pick_frame` 照常触发（翻帧自动保存 ✓ 撤销快照跨帧 ✓）。"""
        row = self.lst.currentRow() + int(delta)
        if 0 <= row < self.lst.count():
            self.lst.setCurrentRow(row)

    def _on_pick_frame(self, cur, _prev):
        if self._dirty:
            self._save()                      # 翻帧自动保存（只在改过时写 ✓）
        if cur is None:
            self._cur = None
            self.canvas.load(QPixmap(), [])
            return
        name = str(cur.data(Qt.UserRole) or cur.text())
        p = Path(self.ed_images.text().strip()) / name
        self._cur = p
        pm = QPixmap(str(p))
        w, h = pm.width(), pm.height()
        boxes = load_yolo_labels(label_path_for(p, self.labels_dir()), w, h)
        # ⚠⚠ **`load_yolo_labels` 返回的是中心 (cx, cy)，`canvas.load` 吃的是左上角
        #   (x, y)** —— 不转的话每次切帧往返框整体向右下漂移 (半宽, 半高)，再存再载
        #   **继续累积**（2026-09-30 用户报 "切一帧别的再切回来就全部错位了" ✗ 实锤；
        #   `_save` 那侧的 反向转换（左上→中心）一直是对的 ✓ 这头漏了 ✓）。
        boxes = [(b[0], b[1] - b[3] / 2.0, b[2] - b[4] / 2.0, b[3], b[4])
                 for b in boxes]
        self.canvas.load(pm, boxes, editable=True, fit=True)
        self._dirty = False
        self._sync_cls_combo()

    def _sync_cls_combo(self):
        names = self._names_getter()
        # ⭐ **框上的类名 = 顶部配的**（2026-09-30 用户报 ✓：YOLO 提案的框显示
        #   "玩家"= 画布写死的质检台名 ✗）—— 画布的标签按这张表取 ✓。
        self.canvas.set_class_names(names)
        if names and self.canvas.current_cls >= len(names):
            self.canvas.current_cls = 0       # 名表变短 ⇒ 别停在越界类别上 ✓
        if [self.cmb_cls.itemText(i) for i in range(self.cmb_cls.count())] == names:
            return
        self.cmb_cls.clear()
        for i, n in enumerate(names):
            self.cmb_cls.addItem("[%d] %s" % (i, n), i)

    def _mark_dirty(self):
        self._dirty = True

    def _save(self):
        if self._cur is None:
            return
        w, h = self.canvas.img_size()
        boxes = [(b[0], b[1] + b[3] / 2.0, b[2] + b[4] / 2.0, b[3], b[4])
                 for b in self.canvas.get_boxes()]   # 左上角 ⇒ 中心 ✓
        save_yolo_labels(label_path_for(self._cur, self.labels_dir()), boxes, w, h)
        self._dirty = False
        # ⭐⭐ **人工改过记账**（2026-09-30 用户要求 ✓）：画布保存 = 人碰过这帧 ✓
        #   —— 与**批量提案**（直接落盘 ✗ 不算人工 ✓）区分开；台账跟着标签目录 ✓
        #   （`labels/_manual.json` ✓）；重复保存幂等（集合 ✓）。
        if self._cur.stem not in self._manual:
            self._manual.add(self._cur.stem)   # ⭐ **stem 口径** ✓（与选帧弹窗/
            #   train_frames.json 一致 ✓ —— 带扩展名会让两处对不上 ✗ 实测 ✓）
            _lp = self.labels_dir()
            if _lp is not None:
                save_selection(manual_ledger_path(_lp), sorted(self._manual))
        self.lbl_state.setText("已存 %s（%d 框）" % (self._cur.name, len(boxes)))
        # 列表行上的框数 / 标注时间跟着刷新（筛"空帧/有框"才不会骗人 ✓）
        for f in self._frames:
            if f[0] == self._cur:
                f[1] = len(boxes)
                f[2] = time.time()
                break
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if str(it.data(Qt.UserRole) or "") == self._cur.name:
                it.setText(self._row_text(self._cur.name, len(boxes)))
                break

    def _on_propose(self):
        if self._cur is None:
            return
        img = cv2.imread(str(self._cur))
        if img is None:
            return
        self.canvas.before_change.emit()      # 撤销快照（铺错 Ctrl+Z 一步回来 ✓）
        for cx, cy, w, h in propose_boxes(img):
            self.canvas.add_box(cx - w / 2.0, cy - h / 2.0, w, h,
                                self.cmb_cls.currentData() or 0, False)
        self.canvas.boxes_changed.emit()
        self.lbl_state.setText("提案 %d 框已铺上 —— 删掉误报、补上漏的再保存 ✓"
                               % len(self.canvas.boxes))

    def _log(self, msg):
        """标注页**处理日志**：带时间戳、只追加、自动滚到最新、超长截头 ✓
        （2026-09-30 用户要求"需要日志区"✓ —— 状态行只说**现在**，日志留**历史** ✓）。"""
        self.log.appendPlainText(time.strftime("%H:%M:%S  ") + msg)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())             # always scroll to newest

    # ---------------- 权重 / 多帧批量提案（2026-09-30 ✓）----------------
    def _effective_weights(self):
        """当前生效的提案权重：**手选优先** ✓ → runs/detect 最新 ✓ → `None` ✓。"""
        if getattr(self, "_yolo_weights", None) \
                and Path(self._yolo_weights).is_file():
            return Path(self._yolo_weights)
        root = Path(self.ed_images.text().strip()).parent
        ws = find_weights(root)
        return ws[0] if ws else None

    def _refresh_model_label(self):
        """右边那格**当前模型**显示（手选 = 记整场 ✓；没手选 = 自动的那个 ✓）。"""
        w = self._effective_weights()
        if w is None:
            self.lbl_model.setText("模型：（未选）")
            self.lbl_model.setToolTip(
                "还没选权重 —— 点「选择模型…」✓。")
        else:
            self.lbl_model.setText("模型：%s" % w.name)
            self.lbl_model.setToolTip(
                "当前提案用的权重：\n%s\n\n「选择模型…」可换（选一次记整场 ✓）；\n"
                "没手选时自动用 runs/detect 里最新的 ✓。" % w)

    def _pick_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "选 YOLO 权重",
                                           str(ROOT), "YOLO 权重 (*.pt)")
        if not f:
            return
        self._yolo_weights = f
        self._refresh_model_label()

    def selected_frames(self):
        """多选的帧（按**列表顺序** ✓）—— 单选时就是那一帧 ✓。"""
        base = Path(self.ed_images.text().strip())
        rows = sorted(self.lst.row(it) for it in self.lst.selectedItems())
        return [base / str(self.lst.item(r).data(Qt.UserRole)
                           or self.lst.item(r).text()) for r in rows]

    def _batch_propose(self, paths):
        """**一键多帧提案**（2026-09-30 用户要求 ✓）：后台线程分块跑推理（每批 25 帧 ✓
        进度条按帧推进 ✓），完成后把提案**直接写进各帧标签**（宁多勿漏、与已有框
        IoU>0.30 去重 ✓ —— **旧工作不白费** ✓ 人工框全保留 ✓）。画布只保留**上次
        单选**那帧 ✓。按钮禁用 + 进度条 + 日志全程留痕 ✓。"""
        weights = self._effective_weights()
        if weights is None:
            self.lbl_state.setText("还没选模型 —— 点右边「选择模型…」✓")
            self._log("批量提案跳过：还没选模型 ✓")
            return
        if self._dirty:
            self._save()                  # 当前帧手上的改动先落盘（旧工作不白费 ✓）
        chunk = 25
        chunks = [paths[i:i + chunk] for i in range(0, len(paths), chunk)]
        self._task = {"kind": "batch", "paths": list(paths), "chunks": chunks,
                      "ci": 0, "out": {}, "done": False, "err": None,
                      "weights": weights, "labels_dir": self.labels_dir()}
        self.btn_yolo.setEnabled(False)
        self.btn_model.setEnabled(False)
        self.pbar.setVisible(True)
        self.pbar.setRange(0, max(1, len(paths)))
        self.pbar.setValue(0)
        self.lbl_state.setText("YOLO 提案中… 0 / %d 帧" % len(paths))
        self._log("批量提案开始：%d 帧（权重 %s，每批 %d 帧）"
                  % (len(paths), weights.name, chunk))

        def _work():
            t = self._task
            try:
                for ch in t["chunks"]:
                    t["out"].update(yolo_propose_many(t["weights"], ch))
                    t["ci"] += 1                  # 进度（只写数字 ✓ §5 ✓）
            except Exception as e:                # noqa: BLE001 —— 收集给主线程弹 ✓
                t["err"] = str(e)
            t["done"] = True

        threading.Thread(target=_work, daemon=True).start()
        self._task_timer.start()

    def _poll_task(self):
        """后台任务轮询（150ms ✓）：scan / batch 各自收尾（§5：UI 只读结果 ✓）。"""
        t = self._task
        if t is None or not t["done"]:
            if t is not None and t["kind"] == "scan":
                self.pbar.setValue(t["i"])
                self.lbl_state.setText("载入中… %d / %d" % (t["i"], len(t["paths"])))
            elif t is not None and t["kind"] == "batch":
                self.pbar.setValue(len(t["out"]))
                self.lbl_state.setText("YOLO 提案中… %d / %d 帧"
                                       % (len(t["out"]), len(t["paths"])))
            return
        self._task_timer.stop()
        self.pbar.setVisible(False)
        self.btn_reload.setEnabled(True)
        err = t.get("err")
        if t["kind"] == "scan":
            self._frames = [] if err else t["rows"]
            self._manual = set(t["manual"] or []) if not err else self._manual
            self._rebuild_list()
            self._sync_cls_combo()
            if err:
                self._log("载入失败：%s" % err)
                self.lbl_state.setText("载入失败 ✗（见日志）")
                QMessageBox.warning(self, "yolo工作台", "载入数据集失败：%s" % err)
            else:
                # ⭐ **旧工作不白费** ✓：上次的当前帧（或会话里记的那帧 ✓ 首次载入
                #   prev 为空 ⇒ 用会话的 ✓）还在 ⇒ 原地选回去 ✓
                prev = t["prev"] or getattr(self, "_pending_frame", "")
                if prev:
                    for i in range(self.lst.count()):
                        if str(self.lst.item(i).data(Qt.UserRole) or "") == prev:
                            self.lst.setCurrentRow(i)     # 触发 _on_pick_frame ✓
                            break
                self._log("载入完成：%d 张（筛选后显示 %d）"
                          % (len(self._frames), self.lst.count()))
                self.lbl_state.setText("共 %d 张（显示 %d）"
                                       % (len(self._frames), self.lst.count()))
            self._task = None
            return
        # ---- batch 收尾 ----
        self.btn_yolo.setEnabled(True)
        self.btn_model.setEnabled(True)
        if err:
            self._log("批量提案失败：%s" % err)
            QMessageBox.warning(self, "yolo工作台", "YOLO 推理失败：%s" % err)
            self._task = None
            return
        out = t["out"] or {}
        lbl_dir = t["labels_dir"]
        total = 0
        for p, props in out.items():
            p = Path(p)
            w, h = self._img_size_of(p)
            existing = load_yolo_labels(label_path_for(p, lbl_dir), w, h)
            # ⚠ 中心→左上（同 `_on_pick_frame` 载入口径 ✓ 别再犯 150.7 那个错 ✗）
            existing = [(c, x - bw / 2.0, y - bh / 2.0, bw, bh)
                        for c, x, y, bw, bh in existing]
            fresh = dedup_proposals(existing, props)
            if not fresh:
                continue
            merged = existing + fresh
            save_yolo_labels(label_path_for(p, lbl_dir),
                             [(c, x + bw / 2.0, y + bh / 2.0, bw, bh)
                              for c, x, y, bw, bh in merged], w, h)
            total += len(fresh)
            for f in self._frames:
                if f[0] == p:
                    f[1] = len(merged)
                    break
            for i in range(self.lst.count()):
                it = self.lst.item(i)
                if str(it.data(Qt.UserRole) or "") == p.name:
                    it.setText(self._row_text(p.name, len(merged)))
                    break
        cur = Path(str(self._cur)) if self._cur is not None else None
        if cur is not None and cur in {Path(p) for p in out}:
            self._on_pick_frame(self.lst.currentItem(), None)   # 在批里 ⇒ 重载 ✓
        self._log("批量提案完成：%d 帧、新铺 %d 框（权重 %s）—— 逐帧删误报补漏 ✓"
                  % (len(out), total, t["weights"].name))
        self.lbl_state.setText(
            "YOLO 批量提案：%d 帧、新铺 %d 框 —— 逐帧删误报补漏 ✓"
            % (len(out), total))
        self._task = None

    @staticmethod
    @staticmethod
    def _img_size_of(p):
        pm = QPixmap(str(p))
        return pm.width(), pm.height()

    def _on_yolo_propose(self):
        """YOLO 辅助标注（用户 2026-09-30 要的 ① ✓ 仿数据集工作台「YOLO 标注」✓）。
        ⭐ 多选（>1 帧）⇒ 转**批量提案**（`_batch_propose` ✓）；单帧 ⇒ 铺上画布 ✓。"""
        sel = self.selected_frames()
        if len(sel) > 1:
            self._batch_propose(sel)          # **多选 ⇒ 一键多帧提案**（落盘 ✓）
            return
        if self._cur is None:
            return
        weights = self._effective_weights()
        if weights is None:
            # ⭐ **提案按钮只管提案**（用户 2026-09-30 ✓）—— 换模型在右边 ✓
            self.lbl_state.setText("还没选模型 —— 点右边「选择模型…」✓"
                                   "（选一次记整场 ✓）")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            props = yolo_propose_boxes(weights, self._cur)
        except Exception as e:                # noqa: BLE001 —— 没装/权重坏 ⇒ 弹窗 ✓
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "yolo工作台",
                                "YOLO 推理失败（权重：%s）：%s" % (weights.name, e))
            return
        QApplication.restoreOverrideCursor()
        fresh = dedup_proposals(self.canvas.get_boxes(), props)
        n_cls = len(self._names_getter())
        self.canvas.before_change.emit()      # 撤销快照 ✓
        for cls, x, y, w, h in fresh:
            # 模型类别 id 直接用（训的时候同 data.yaml ⇒ 顺序一致 ✓）；
            # 越界（权重类别对不上）⇒ 退回当前下拉选的类别 ✓
            self.canvas.add_box(x, y, w, h,
                                cls if 0 <= cls < n_cls
                                else (self.cmb_cls.currentData() or 0), False)
        self.canvas.boxes_changed.emit()
        self.lbl_state.setText(
            "YOLO 提案：铺上 %d 框（与已有框重叠丢掉 %d 个 · 权重 %s）——"
            "删误报补漏再保存 ✓" % (len(fresh), len(props) - len(fresh), weights.name))
        self._log("YOLO 提案 %s：+%d 框（重叠去重 %d · 权重 %s）"
                  % (self._cur.name, len(fresh), len(props) - len(fresh),
                     weights.name))

    # ------------- 复制 / 粘贴 / 撤销（仿质检台 ✓ 用户 2026-09-30 要的 ④）-------------
    def _on_before_change(self):
        """破坏性操作（加框/删框/拖动）前记录撤销快照（上限 100 层 ✓ 同质检台 ✓）。"""
        if self._cur is None:
            return
        self._undo.append((self._cur.name, self.canvas.get_boxes()))
        if len(self._undo) > 100:
            del self._undo[0]

    def _copy(self):
        sel = self.canvas.get_selected_boxes()
        if not sel:
            self.lbl_state.setText("没有选中的框 —— 先点一下框再复制")
            return
        self._clipboard = list(sel)
        self.lbl_state.setText("已复制 %d 个框" % len(sel))

    def _paste(self):
        if not self._clipboard or self._cur is None:
            return
        self.canvas.before_change.emit()      # 撤销快照 ✓
        for cls, x, y, w, h, _manual in self._clipboard:   # 6 元组（带 manual ✓）
            # 偏移 12px，避免和原框完全重叠看不清（同质检台 ✓）
            self.canvas.add_box(x + 12, y + 12, w, h, cls, True)
        self.canvas.boxes_changed.emit()
        self.lbl_state.setText("已粘贴 %d 个框" % len(self._clipboard))

    def _undo_edit(self):
        if not self._undo:
            self.lbl_state.setText("没有可撤销的操作")
            return
        stem, snap = self._undo.pop()
        if self._cur is None or self._cur.name != stem:
            self._jump_to(stem)               # 撤销的可能不是当前帧（跨帧 ✓ 同质检台 ✓）
        self.canvas.replace_boxes(snap)
        self._dirty = True
        self.lbl_state.setText("已撤销（%s）" % stem)

    def _jump_to(self, stem):
        for i in range(self.lst.count()):
            if str(self.lst.item(i).data(Qt.UserRole) or "") == stem:
                self.lst.setCurrentRow(i)     # 触发 `_on_pick_frame`（旧帧自动保存 ✓）
                return


# ══════════════════════════════════════════════════════════
# 选帧弹窗（② 训练用 ✓ 仿数据集工作台的 FramePickDialog ✓）
# ══════════════════════════════════════════════════════════
class _SelFramesDialog(QDialog):
    """「哪些帧参与训练」选帧弹窗（用户 2026-09-30 要的 ② ✓）。

    交互同 `FramePickDialog`：每行复选框、Shift/Ctrl 多选、**空格批量切换**所选行、
    全选/全不选/反选、顶部筛选（只动**显示**不动勾选 ✓）、帧号段「只勾这些」✓。
    """

    FILTERS = (("全部", "all"), ("已勾选", "checked"), ("未勾选", "unchecked"),
               ("空帧（没标注）", "empty"), ("有框", "has"), ("人工改过", "manual"))

    def __init__(self, parent, img_dir, labels_dir, checked=None):
        super().__init__(parent)
        self.setWindowTitle("选帧 —— 哪些参与训练")
        self._items = []                      # [(stem, QListWidgetItem)]
        self._boxes = {}                      # stem -> 框数（显示/筛选用 ✓）
        self.total = 0
        paths = scan_images(img_dir)
        self.total = len(paths)
        for p in paths:
            lp = label_path_for(p, labels_dir) if labels_dir else None
            self._boxes[p.stem] = count_label_lines(lp) if lp else 0
        self._manual = (set(load_selection(manual_ledger_path(labels_dir))
                            or []) if labels_dir else set())

        head = QLabel("要哪些帧参与训练？（默认全选；Shift 选一段后按空格可批量勾 / 取消）")
        head.setStyleSheet("color: #202124;")
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("只看"))
        self.cmb_filter = NoWheelComboBox()
        for text, key in self.FILTERS:
            self.cmb_filter.addItem(text, key)
        self.cmb_filter.currentIndexChanged.connect(self._apply_filter)
        bar.addWidget(self.cmb_filter)
        bar.addWidget(QLabel("帧号"))
        self.ed_range = QLineEdit()
        self.ed_range.setPlaceholderText("如 0-499,1200（按名字末尾的数字 ✓）")
        self.ed_range.returnPressed.connect(self._check_range_only)
        bar.addWidget(self.ed_range, 1)
        b = QPushButton("只勾这些")
        b.setToolTip("按上面的帧号段勾选，其余全部取消")
        b.clicked.connect(self._check_range_only)
        bar.addWidget(b)
        root.addLayout(bar)

        btns = QHBoxLayout()
        btns.setSpacing(6)
        for text, slot in (("全选", self._check_all), ("全不选", self._uncheck_all),
                           ("反选", self._invert)):
            btns.addWidget(QPushButton(text, clicked=slot))
        for text, force in (("勾选所选", True), ("取消勾选所选", False)):
            btns.addWidget(QPushButton(text, clicked=lambda _=False, f=force:
                                       self._toggle_selected(f)))
        btns.addStretch(1)
        root.addLayout(btns)

        self.lst = QListWidget()
        self.lst.setSelectionMode(QListWidget.ExtendedSelection)
        self.lst.setUniformItemSizes(True)
        root.addWidget(self.lst, 1)
        for key in (Qt.Key_Space, Qt.Key_Return):
            sc = QShortcut(key, self.lst)
            sc.setContext(Qt.WidgetShortcut)
            sc.activated.connect(lambda: self._toggle_selected(None))

        foot = QHBoxLayout()
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color: #5f6368;")
        foot.addWidget(self.lbl_count, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("确定")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        theme.unify_ok_cancel(bb.button(QDialogButtonBox.Ok),
                              bb.button(QDialogButtonBox.Cancel))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        foot.addWidget(bb)
        root.addLayout(foot)

        self.resize(680, 540)
        self._populate(checked)
        theme.bind_window_state(self, "yolo_wb_frames")

    # ---------------- 列表 ----------------
    def _populate(self, checked):
        want = set(checked) if checked is not None else None
        self.lst.blockSignals(True)
        for stem in sorted(self._boxes):
            _mk = "　人工改过" if stem in self._manual else ""
            it = QListWidgetItem("%s　%d框%s" % (stem, self._boxes[stem], _mk))
            it.setData(Qt.UserRole, stem)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if want is None or stem in want
                             else Qt.Unchecked)
            self.lst.addItem(it)
            self._items.append((stem, it))
        self.lst.blockSignals(False)
        self._apply_filter()

    def _apply_filter(self):
        mode = self.cmb_filter.currentData() or "all"
        for stem, it in self._items:
            if mode == "checked":
                it.setHidden(it.checkState() != Qt.Checked)
            elif mode == "unchecked":
                it.setHidden(it.checkState() == Qt.Checked)
            elif mode == "empty":
                it.setHidden(self._boxes.get(stem, 0) > 0)
            elif mode == "has":
                it.setHidden(self._boxes.get(stem, 0) == 0)
            elif mode == "manual":
                it.setHidden(stem not in self._manual)
            else:
                it.setHidden(False)
        self._update_count()

    def _update_count(self):
        n = sum(1 for _s, it in self._items if it.checkState() == Qt.Checked)
        vis = sum(1 for _s, it in self._items if not it.isHidden())
        self.lbl_count.setText("已勾选 %d / 共 %d%s"
                               % (n, len(self._items),
                                  "" if vis == len(self._items)
                                  else "（当前显示 %d）" % vis))

    def _toggle_selected(self, force=None):
        """空格 / 「所选」按钮：切换选中行的勾选（None = 有没勾上的就全勾上 ✓）。"""
        sel = self.lst.selectedItems()
        if not sel:
            return
        if force is None:
            force = any(it.checkState() != Qt.Checked for it in sel)
        self.lst.blockSignals(True)
        for it in sel:
            it.setCheckState(Qt.Checked if force else Qt.Unchecked)
        self.lst.blockSignals(False)
        self._apply_filter()

    def _check_all(self):
        self._toggle_all(True)

    def _uncheck_all(self):
        self._toggle_all(False)

    def _invert(self):
        self.lst.blockSignals(True)
        for _s, it in self._items:
            it.setCheckState(Qt.Unchecked if it.checkState() == Qt.Checked
                             else Qt.Checked)
        self.lst.blockSignals(False)
        self._apply_filter()

    def _toggle_all(self, want):
        self.lst.blockSignals(True)
        for _s, it in self._items:
            it.setCheckState(Qt.Checked if want else Qt.Unchecked)
        self.lst.blockSignals(False)
        self._apply_filter()

    def _parse_range(self):
        """帧号段文本 → 数字集合（`0-499,1200` ✓；取名字**末尾**的数字 ✓）。"""
        out = set()
        for part in (self.ed_range.text() or "").replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, _, b = part.partition("-")
                if a.strip().isdigit() and b.strip().isdigit():
                    lo, hi = int(a), int(b)
                    out.update(range(min(lo, hi), max(lo, hi) + 1))
            elif part.isdigit():
                out.add(int(part))
        return out

    def _check_range_only(self):
        rng = self._parse_range()
        if not rng:
            return
        self.lst.blockSignals(True)
        for stem, it in self._items:
            tail = stem.rsplit("_", 1)[-1]
            it.setCheckState(Qt.Checked if (tail.isdigit() and int(tail) in rng)
                             else Qt.Unchecked)
        self.lst.blockSignals(False)
        self._apply_filter()

    # ---------------- 结果 ----------------
    def selected_stems(self):
        """返回勾选的帧名列表（**全勾 = 调用方当"不限"** ✓）。"""
        return [stem for stem, it in self._items
                if it.checkState() == Qt.Checked]


# ══════════════════════════════════════════════════════════
# ② 训练
# ══════════════════════════════════════════════════════════
class _TrainTab(QWidget):
    """训练：生成 data.yaml → 挑参 → QProcess 跑 ultralytics（日志实时流 ✓）。"""

    def __init__(self, names_getter, img_dir_getter, labels_dir_getter=None,
                 parent=None):
        super().__init__(parent)
        self._names_getter = names_getter
        self._img_dir_getter = img_dir_getter
        self._labels_dir_getter = labels_dir_getter
        self._proc = None
        self._sel = None                      # 参与训练的帧（None = 全部 ✓）
        root = QVBoxLayout(self)
        root.setSpacing(6)

        form = QFormLayout()
        form.setSpacing(6)
        self.cmb_model = NoWheelComboBox()
        self._fill_models()
        btn_m = QPushButton("浏览…")
        btn_m.clicked.connect(self._pick_model)
        row_m = QHBoxLayout()
        row_m.addWidget(self.cmb_model, 1)
        row_m.addWidget(btn_m)
        form.addRow("基础权重", row_m)
        self.sp_epochs = NoWheelSpinBox()
        self.sp_epochs.setRange(1, 2000)
        self.sp_epochs.setValue(120)
        self.sp_epochs.setToolTip("训练多少轮。数据几百张时 100~200 轮通常够；\n"
                                  "连珠 40 轮不涨 ultralytics 会自动早停 ✓。")
        form.addRow("轮数", self.sp_epochs)
        self.sp_imgsz = NoWheelSpinBox()
        self.sp_imgsz.setRange(320, 2048)
        self.sp_imgsz.setValue(960)
        self.sp_imgsz.setToolTip("训练时把画面缩到多大再喂网络；越大越吃显存 ✓。")
        form.addRow("输入尺寸", self.sp_imgsz)
        self.sp_batch = NoWheelSpinBox()
        self.sp_batch.setRange(1, 128)
        self.sp_batch.setValue(8)
        self.sp_batch.setToolTip("每步喂几张。报显存不足（OOM）就调小 ✗。")
        form.addRow("批大小", self.sp_batch)
        self.ed_name = QLineEdit("v1")
        self.ed_name.setToolTip("本次训练的名字 ⇒ 产物落在 数据集根/runs/detect/<名字> ✓")
        form.addRow("本次名字", self.ed_name)
        srow = QHBoxLayout()
        srow.setSpacing(6)
        self.btn_frames = QPushButton("选帧…")
        self.btn_frames.setToolTip(
            "指定**哪些帧参与训练**（仿数据集工作台「选帧」✓ 用户 2026-09-30 要的 ②）——\n"
            "弹窗里勾选，确定后立即重写 data.yaml（train/val 只吃选中的帧 ✓）；\n"
            "**全选 = 不限**（台账清掉 ✓）。台账存在 数据集根/train_frames.json ✓。")
        self.btn_frames.clicked.connect(self._pick_frames)
        srow.addWidget(self.btn_frames)
        self.lbl_sel = QLabel("")
        self.lbl_sel.setStyleSheet("color: #5f6368;")
        srow.addWidget(self.lbl_sel, 1)
        form.addRow("参与训练的帧", srow)
        root.addLayout(form)

        brow = QHBoxLayout()
        self.btn_yaml = QPushButton("生成 data.yaml")
        self.btn_yaml.clicked.connect(self._make_yaml)
        brow.addWidget(self.btn_yaml)
        self.btn_run = QPushButton("开始训练")
        self.btn_run.clicked.connect(self._toggle)
        brow.addWidget(self.btn_run)
        brow.addStretch(1)
        root.addLayout(brow)

        self.log = QPlainTextEdit()           # Log style (data-wb same)
        self.log.setObjectName("Log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setTextInteractionFlags(Qt.TextSelectableByMouse
                                         | Qt.TextSelectableByKeyboard)
        root.addWidget(self.log, 1)

    def _fill_models(self):
        self.cmb_model.clear()
        for p in sorted(ROOT.glob("*.pt")):
            self.cmb_model.addItem(p.name, str(p))
        self.cmb_model.addItem("yolov8n.pt（联网下载）", "yolov8n.pt")

    def _pick_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "选基础权重",
                                           str(ROOT), "YOLO 权重 (*.pt)")
        if f:
            self.cmb_model.addItem(Path(f).name, f)
            self.cmb_model.setCurrentIndex(self.cmb_model.count() - 1)

    def _root(self):
        d = self._img_dir_getter()
        return Path(d).parent if d else None

    def _sel_path(self):
        """选帧台账位置（数据集根/train_frames.json ✓）。"""
        root = self._root()
        return root / "train_frames.json" if root else None

    def _current_selection(self):
        """当前生效的选帧（**懒读**台账 ✓ —— 工作台重开后 json 还在 ⇒ 仍生效 ✓）。"""
        if self._sel is None:
            p = self._sel_path()
            if p is not None and p.is_file():
                self._sel = load_selection(p)
        return self._sel

    def _refresh_sel_label(self, stems):
        total = len(scan_images(self._img_dir_getter() or ""))
        if stems is None:
            self.lbl_sel.setText("全部 %d 张（未挑选）" % total)
        else:
            self.lbl_sel.setText("已挑 %d / %d 张（data.yaml 已重写 ✓）"
                                 % (len(stems), total))

    def _pick_frames(self):
        """弹选帧窗；确定 ⇒ 存台账 + **立即重写 data.yaml**（少一步心记 ✓）。"""
        d = self._img_dir_getter()
        if not d:
            QMessageBox.warning(self, "yolo工作台", "先在「标注」页填图片目录 ✓")
            return
        sel_path = self._sel_path()
        dlg = _SelFramesDialog(self, d,
                               self._labels_dir_getter() if self._labels_dir_getter else None,
                               self._current_selection())
        if dlg.exec_() != QDialog.Accepted or dlg.total == 0:
            return
        stems = dlg.selected_stems()
        if len(stems) == dlg.total:
            stems = None                      # 全选 = 不限 ✓（台账清掉 ✓）
        if sel_path is not None:
            if stems is None:
                if sel_path.exists():
                    sel_path.unlink()
            else:
                save_selection(sel_path, stems)
        self._sel = stems
        self._refresh_sel_label(stems)
        self._make_yaml()

    def _make_yaml(self):
        root = self._root()
        if root is None:
            QMessageBox.warning(self, "yolo工作台", "先在「标注」页填图片目录 ✓")
            return
        subset = None
        stems = self._current_selection()
        if stems:
            imgs = scan_images(self._img_dir_getter() or "")
            want = set(stems)
            # ⭐ 条目 = 相对 `root` 的**真实相对路径**，且**必须带 `./` 前缀**（2026-09-30
            #   修 ✓ 两次翻车）：① 原来写死 "images/" 前缀，现场目录叫 liedetectorphotos
            #   ⇒ 指向不存在的路径 ⇒ 每张读不到 ⇒ 逐帧 pip 重试 ✗；② ultralytics 读清单
            #   （`get_img_files`）**只对 "./" 开头的行**换算成相对清单文件的位置，其余行
            #   按进程 cwd 解析 ⇒ 不带 "./" 全部 404 ⇒ "No labels found" ✗。
            lines = ["./" + p.relative_to(root).as_posix()
                     for p in imgs if p.stem in want]
            (root / "train_frames.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            subset = "train_frames.txt"       # train/val 都指向清单 ✓
            self.log.appendPlainText("选帧：%d / %d 张参与（清单 %s）"
                            % (len(lines), len(imgs), root / "train_frames.txt"))
            # ⚠ **YOLO 找标签靠目录名**：ultralytics 把图片路径里的 `images` 段替换成
            #   `labels`（img2label_paths ✓）—— 图片目录不叫 `images` ⇒ 标签永远配不上
            #   ⇒ **全部当背景图**（训练照跑但啥也学不到 ✗）。提醒一句，别让用户白训。
            if Path(self._img_dir_getter() or "").name.lower() != "images":
                self.log.appendPlainText(
                    "⚠ 图片目录名不是「images」（现在是 %s）—— ultralytics 靠 "
                    "images/labels 目录名配对找标签，配不上会全当背景图 ✗。\n"
                    "  建议把图片放进 <数据集根>/images/、标签放 <数据集根>/labels/ "
                    "再训练。" % Path(self._img_dir_getter()).name)
        text = make_data_yaml(root, self._names_getter(), subset)
        (root / "data.yaml").write_text(text, encoding="utf-8")
        self.log.appendPlainText("已生成 %s\n%s" % (root / "data.yaml", "-" * 40))
        self.log.appendPlainText(text)

    def _toggle(self):
        if self._proc is not None:
            self._proc.kill()                 # 停止 = 杀掉训练进程 ✓（早停交给 ultralytics ✓）
            self._proc = None
            self.btn_run.setText("开始训练")
            return
        root = self._root()
        yaml_path = root / "data.yaml" if root else None
        if yaml_path is None or not yaml_path.is_file():
            QMessageBox.warning(self, "yolo工作台", "先点「生成 data.yaml」✓")
            return
        code = ("from ultralytics import YOLO; YOLO(r'%s').train("
                "data=r'%s', epochs=%d, imgsz=%d, batch=%d, name=r'%s')"
                % (self.cmb_model.currentData(), yaml_path,
                   self.sp_epochs.value(), self.sp_imgsz.value(),
                   self.sp_batch.value(), self.ed_name.text().strip() or "v1"))
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(root))
        self._proc.readyReadStandardOutput.connect(self._pump)
        self._proc.readyReadStandardError.connect(self._pump_err)
        self._proc.finished.connect(self._done)
        self._proc.start(sys.executable, ["-c", code])
        self.btn_run.setText("停止")
        self.log.appendPlainText("$ 训练已启动（ultralytics，日志实时刷新 ↓）")

    def _pump(self):
        if self._proc is not None:
            self.log.appendPlainText(str(self._proc.readAllStandardOutput(), "utf-8",
                                errors="replace").rstrip())

    def _pump_err(self):
        if self._proc is not None:
            self.log.appendPlainText(str(self._proc.readAllStandardError(), "utf-8",
                                errors="replace").rstrip())

    def _done(self, _code, _st):
        root = self._root()
        bests = find_weights(root) if root else []
        self.log.appendPlainText("训练结束 ✓ 最终权重：%s"
                        % (bests[0] if bests else "（没找到 best.pt —— 看上面的日志 ✗）"))
        self.btn_run.setText("开始训练")
        self._proc = None


# ══════════════════════════════════════════════════════════
# ③ 验证
# ══════════════════════════════════════════════════════════
class _ValTab(QWidget):
    """验证：跑 val 看指标 + 检测预览把结果画在图上人眼看 ✓。"""

    def __init__(self, img_dir_getter, parent=None):
        super().__init__(parent)
        self._img_dir_getter = img_dir_getter
        self._results = []                    # [(图路径, 画好的 BGR)]（预览翻页 ✓）
        self._ri = 0
        self._vproc = None                    # val 的 QProcess（流式日志 ✓ 不冻界面 ✓）
        self._ptask = None                    # 检测预览的后台任务 ✓
        self._ptimer = QTimer(self)
        self._ptimer.setInterval(150)
        self._ptimer.timeout.connect(self._poll_predict)
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setSpacing(6)
        lv.addWidget(QLabel("权重"))
        self.ed_weights = QLineEdit()
        self.ed_weights.setPlaceholderText("runs/detect/…/best.pt（「扫描权重」自动填 ✓）")
        lv.addWidget(self.ed_weights)
        self.btn_scan = QPushButton("扫描权重")
        self.btn_scan.clicked.connect(self._scan)
        lv.addWidget(self.btn_scan)
        self.btn_val = QPushButton("跑验证 (val)")
        self.btn_val.setToolTip(
            "对**整个数据集**跑一遍评估，日志里给出 mAP50 / mAP50-95 等指标 ✓。")
        self.btn_val.clicked.connect(self._run_val)
        lv.addWidget(self.btn_val)
        self.btn_prev = QPushButton("检测预览")
        self.btn_prev.setToolTip(
            "拿当前权重对数据集的**前 8 张图**跑检测，把结果画出来人眼看 ✓。")
        self.btn_prev.clicked.connect(self._predict)
        lv.addWidget(self.btn_prev)
        self.btn_next = QPushButton("下一张预览 →")
        self.btn_next.clicked.connect(self._next)
        lv.addWidget(self.btn_next)
        self.log = QPlainTextEdit()           # Log style (data-wb same)
        self.log.setObjectName("Log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setTextInteractionFlags(Qt.TextSelectableByMouse
                                         | Qt.TextSelectableByKeyboard)
        lv.addWidget(self.log, 1)
        # ⭐ 进度条（2026-09-30 用户要求 ✓）：val / 检测预览都是"有处理过程"的行为 ✓
        #    —— 时长未知 ⇒ **忙式**（range(0,0) 来回滚 ✓）干完自己收起来 ✓。
        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        self.pbar.setRange(0, 0)              # 忙式 ✓
        lv.addWidget(self.pbar)
        split.addWidget(left)
        self.canvas = ImageCanvas()
        split.addWidget(self.canvas)
        split.setSizes([300, 620])
        root.addWidget(split)

    def _scan(self):
        root = Path(self._img_dir_getter() or ".").parent
        ws = find_weights(root)
        self.log.appendPlainText("找到 %d 版权重\n%s" % (len(ws), "\n".join(map(str, ws))))
        if ws:
            self.ed_weights.setText(str(ws[0]))

    def _run_val(self):
        """跑 val —— **QProcess 流式日志**（2026-09-30 交互模式复审 ✓）：原来
        `subprocess.run` 把界面**冻住几分钟** ✗（日志一点不出来 ✗）。"""
        if getattr(self, "_vproc", None) is not None:
            self.log.appendPlainText("上一轮验证还在跑 ✗ —— 先等它结束 ✓")
            return
        w = self.ed_weights.text().strip()
        root = Path(self._img_dir_getter() or ".").parent
        yaml_path = root / "data.yaml"
        if not w or not yaml_path.is_file():
            QMessageBox.warning(self, "yolo工作台", "先填权重、并确认 data.yaml 在 ✓")
            return
        self.pbar.setVisible(True)            # 忙式 ✓（时长未知 ✓）
        self.btn_val.setEnabled(False)
        self.log.appendPlainText("$ 验证已启动（ultralytics val，日志实时刷新 ↓）")
        self._vproc = QProcess(self)
        self._vproc.readyReadStandardOutput.connect(self._pump_val)
        self._vproc.readyReadStandardError.connect(self._pump_val_err)
        self._vproc.finished.connect(self._val_done)
        self._vproc.start(sys.executable, [
            "-c", "from ultralytics import YOLO; YOLO(r'%s').val(data=r'%s')"
            % (w, yaml_path)])

    def _pump_val(self):
        if self._vproc is not None:
            self.log.appendPlainText(str(self._vproc.readAllStandardOutput(), "utf-8",
                                errors="replace").rstrip())

    def _pump_val_err(self):
        if self._vproc is not None:
            self.log.appendPlainText(str(self._vproc.readAllStandardError(), "utf-8",
                                errors="replace").rstrip())

    def _val_done(self, code, _st):
        self.log.appendPlainText("验证结束（rc=%d）—— mAP 等指标看上面的日志 ✓" % int(code))
        self.pbar.setVisible(False)
        self.btn_val.setEnabled(True)
        self._vproc = None

    def _predict(self):
        """检测预览 —— **子进程**推理 + 后台线程 + 忙式进度 ✓（旧实现进程内 import
        ultralytics：Qt→torch 的 c10.dll 直接炸 ✗ 150.3 那条；还把界面冻住 ✗）。"""
        if getattr(self, "_ptask", None) is not None \
                and not self._ptask.get("done"):
            self.log.appendPlainText("上一轮预览还在跑 ✗ —— 先等它结束 ✓")
            return
        w = self.ed_weights.text().strip()
        imgs = scan_images(self._img_dir_getter() or "")
        if not w or not imgs:
            QMessageBox.warning(self, "yolo工作台", "先填权重、并确认图片目录 ✓")
            return
        self._ptask = {"done": False, "err": None, "results": None}
        self.pbar.setVisible(True)            # 忙式 ✓
        self.btn_prev.setEnabled(False)
        self.log.appendPlainText("检测预览：前 %d 张（权重 %s）…"
                        % (min(8, len(imgs)), Path(w).name))

        def _work():
            t = self._ptask
            try:
                t["results"] = yolo_plot_many(w, imgs[:8])
            except Exception as e:            # noqa: BLE001 —— 收集给主线程弹 ✓
                t["err"] = str(e)
            t["done"] = True

        threading.Thread(target=_work, daemon=True).start()
        self._ptimer.start()

    def _poll_predict(self):
        """预览后台任务收尾：展示结果 / 报错（§5：UI 只读结果 ✓）。"""
        t = self._ptask
        if t is None or not t["done"]:
            return
        self._ptimer.stop()
        self.pbar.setVisible(False)
        self.btn_prev.setEnabled(True)
        if t["err"]:
            self.log.appendPlainText("检测预览失败：%s" % t["err"])
            QMessageBox.warning(self, "yolo工作台",
                                "推理失败：%s（权重/环境见日志 ✓）" % t["err"])
            self._ptask = None
            return
        self._results = t["results"] or []
        self._ri = 0
        self._ptask = None
        self._show()
        self.log.appendPlainText("预览就绪：%d 张 —— 「下一张预览 →」翻页 ✓"
                        % len(self._results))

    def _show(self):
        if not self._results:
            return
        p, bgr = self._results[self._ri % len(self._results)]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, bgr.shape[1], bgr.shape[0],
                      rgb.strides[0], QImage.Format_RGB888)
        self.canvas.load(QPixmap.fromImage(qimg), [], editable=False, fit=True)
        self.log.appendPlainText("预览 %d/%d：%s"
                        % (self._ri + 1, len(self._results), p.name))

    def _next(self):
        self._ri += 1
        self._show()


# ══════════════════════════════════════════════════════════
# 设置弹窗（蒙版 / 界面字号 —— 2026-09-30 用户要求 ✓）
# ══════════════════════════════════════════════════════════
class _WbSettingsDialog(QDialog):
    """yolo 工作台的「设置」弹窗（用户 2026-09-30 ✓）：

    · **蒙版**：看帧时罩一层灰、**框内挖洞不受影响** ✓（同数据工作台那套 ✓）——
      浓淡 0~100%（0 = 不罩 ✓），拖动**即时生效 + 即时保存** ✓；
    · **界面字号**（pt ✓ §9：单位在表单行标签里 ✓）—— 只作用于**本窗口** ✓
      （应用级的那套不归这儿管 ✓），改动即时生效 ✓。
    ⚠⚠ 本类**不写配置、不碰画布**：改动经 `on_mask` / `on_font` 回调交给外面 ✓
      （本类不认得主窗 ✓ 好测 ✓ —— 自检里用桩回调就**绝不碰** config/ui.yaml ✓
      §11「自检绝不许改用户的 config/ui.yaml」✓）。
    """

    def __init__(self, on_mask=None, on_font=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("yolo 工作台 · 设置")
        self.resize(440, 170)
        root = QVBoxLayout(self)
        root.setSpacing(10)
        form = QFormLayout()
        form.setSpacing(6)

        # ---- 蒙版（同 settings_dialog 的质检台滑条口径 ✓）----
        self.sld_mask = NoWheelSlider(Qt.Horizontal)
        self.sld_mask.setRange(0, 100)
        self.sld_mask.setValue(int(round(theme.yolo_mask_alpha() * 100)))
        self.sld_mask.setToolTip(
            "标注页看帧时罩一层灰蒙版的**浓淡**（% = 不透明度 ✓）。\n\n"
            "· **0 = 不罩**（原始画面 ✓）；\n"
            "· 越大画面越暗 —— **框里挖了洞**（框内的目标保持原亮度 ✓）。\n\n"
            "拖动**即时生效并跟着客户端存** ✓。")
        self._mask_live = QLabel()
        self._mask_live.setStyleSheet("color: #5f6368;")
        self._mask_live.setMinimumWidth(40)
        mrow = QHBoxLayout()
        mrow.setSpacing(8)
        mrow.addWidget(self.sld_mask, 1)
        mrow.addWidget(self._mask_live)
        form.addRow("蒙版", mrow)

        # ---- 界面字号（pt ✓）----
        self.sp_font = NoWheelSpinBox()
        self.sp_font.setRange(7, 16)
        self.sp_font.setValue(theme.ui_font_pt())
        self.sp_font.setToolTip(
            "本窗口的**界面字号**（pt ✓）—— 改动**即时生效** ✓（跟着客户端存 ✓）。\n"
            "⚠ 只作用于 yolo 工作台这个窗口 ✓。")
        form.addRow("界面字号(pt)", self.sp_font)
        root.addLayout(form)

        hint = QLabel("改动**即时生效并保存** ✓ —— 没有确定/取消语义 ✓。")
        hint.setStyleSheet("color: #80868b;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.button(QDialogButtonBox.Close).setText("关闭")
        theme.unify_ok_cancel(bb.button(QDialogButtonBox.Close))
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        def _on_mask_v(v):
            self._mask_live.setText("%d%%" % v)
            if callable(on_mask):
                on_mask(v / 100.0)      # 存配置 + 画布都归外面 ✓（本类零副作用 ✓）

        self.sld_mask.valueChanged.connect(_on_mask_v)
        self._mask_live.setText("%d%%" % self.sld_mask.value())   # 只对齐文案 ✓
        self.sp_font.valueChanged.connect(
            lambda v: callable(on_font) and on_font(int(v)))
        theme.bind_window_state(self, "yolo_wb_settings")   # §11：几何按客户端记 ✓


# ══════════════════════════════════════════════════════════
# 主窗
# ══════════════════════════════════════════════════════════
class YoloWorkbench(QMainWindow):
    """轻量版通用 YOLO 工作台主窗（标注 / 训练 / 验证 三页签 ✓）。"""

    def __init__(self, dataset_root=None):
        super().__init__()
        self.setWindowTitle("YOLO 工作台（轻量版 · 标注 → 训练 → 验证）")
        # Log colors = same as the data workbench (QPlainTextEdit#Log in
        # main_window) -- standalone window does not inherit the main QSS,
        # so re-declare this one rule at window level.
        self.setStyleSheet(
            "QPlainTextEdit#Log { background: #202124; color: #e8eaed;\n"
            "  border: 1px solid #3c4043; border-radius: 6px;\n"
            "  font-family: Consolas, monospace; }")
        self.resize(1080, 720)

        top = QWidget()
        root = QVBoxLayout(top)
        root.setSpacing(6)

        # ---- 全局：类别名（一行 ✓ 三页共用 ✓）----
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("类别名（逗号分隔，顺序 = 类别 id）"))
        self.ed_names = QLineEdit("shape")
        self.ed_names.setToolTip(
            "**类别名**，逗号分隔、顺序 = 类别 id（0 起 ✓）—— 例：`apple,star`。\n"
            "「标注」页拉框时按它选类别 ✓；「训练」页生成 data.yaml 用它 ✓。")
        nrow.addWidget(self.ed_names, 1)
        self.btn_settings = QPushButton("设置…")
        self.btn_settings.setToolTip(
            "蒙版浓淡（框里挖洞 ✓）/ 界面字号 —— 即时生效 ✓。")
        self.btn_settings.clicked.connect(self._open_settings)
        nrow.addWidget(self.btn_settings)
        root.addLayout(nrow)

        self.tabs = QTabWidget()
        self.tab_label = _LabelTab(self._names)
        self.tab_train = _TrainTab(self._names, self.tab_label.ed_images.text,
                                   self.tab_label.labels_dir)
        self.tab_val = _ValTab(self.tab_label.ed_images.text)
        self.tabs.addTab(self.tab_label, "① 标注")
        self.tabs.addTab(self.tab_train, "② 训练")
        self.tabs.addTab(self.tab_val, "③ 验证")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(top)
        # ⭐ 关窗时保存数据状态 ✓（§11 本机偏好 → config/ui.yaml ✓）；
        #   ⚠ 自检里**必须关掉**（§11「自检绝不许改用户的 config/ui.yaml」✓）。
        self._save_session_on_close = True

        # ⭐⭐ **恢复上次关闭前的数据状态**（2026-09-30 用户要求 ✓）—— 路径/类别名/
        #   手选模型/页签/当前帧/筛选排序/训练参数 全回来 ✓（CLI 传了 dataset_root
        #   则图片目录以它为准 ✓）。恢复在 reload **之前**（类别名先回下拉 ✓），
        #   当前帧交给 `_pending_frame`（scan 收尾按它选回 ✓ 见 _LabelTab ✓）。
        self._apply_session(theme.yolo_session(), dataset_root)
        # ⭐ 界面字号 / 蒙版（设置弹窗那套 ✓）开机即生效 ✓（蒙版画布自己记浓淡 ✓）
        self._apply_font(theme.ui_font_pt())
        self.tab_label.canvas.set_mask_alpha(theme.yolo_mask_alpha())
        theme.bind_window_state(self, "yolo_wb")   # 拉过的大小/位置按客户端记住 ✓

    def _open_settings(self):
        _WbSettingsDialog(on_mask=self._on_setting_mask,
                          on_font=self._on_setting_font,
                          parent=self).exec_()

    def _on_setting_mask(self, a):
        theme.set_yolo_mask_alpha(a)          # 跟着客户端存 ✓（§11：本机偏好 ✓）
        self.tab_label.canvas.set_mask_alpha(a)   # 正在看的帧也跟着变 ✓

    def _on_setting_font(self, pt):
        theme.set_ui_font_pt(int(pt))
        self._apply_font(int(pt))

    def _apply_font(self, pt):
        """把**本窗口**的界面字号设成 `pt`（子控件全继承 ✓ —— 样式表没写死字号的
        都跟着变 ✓）。开机从偏好读一次 ✓（`theme.ui_font_pt` ✓）。"""
        f = QFont()
        f.setPointSize(int(pt))
        self.setFont(f)

    # ---------------- 会话保存/恢复（2026-09-30 用户要求 ✓）----------------
    def _apply_session(self, sess, dataset_root=None):
        """把**上次关闭前的数据状态**灌回界面 ✓（不写配置 ✓ 好测 ✓）。
        ⚠ 恢复顺序：类别名/路径先回控件 ⇒ 再 reload（scan 收尾按 `_pending_frame`
        选回当前帧 ✓）；CLI 的 `dataset_root` **优先于**会话里的图片目录 ✓。"""
        sess = dict(sess or {})
        if sess.get("names"):
            self.ed_names.setText(str(sess["names"]))
        if sess.get("images_dir"):
            self.tab_label.ed_images.setText(str(sess["images_dir"]))
        if sess.get("labels_dir"):
            self.tab_label.ed_labels.setText(str(sess["labels_dir"]))
        _wm = str(sess.get("weights_manual") or "")
        if _wm and Path(_wm).is_file():
            self.tab_label._yolo_weights = _wm    # 文件还在才恢复 ✓
            self.tab_label._refresh_model_label()
        if sess.get("weights_val"):
            self.tab_val.ed_weights.setText(str(sess["weights_val"]))
        _fi = self.tab_label.cmb_filter.findData(str(sess.get("filter") or ""))
        if _fi >= 0:
            self.tab_label.cmb_filter.setCurrentIndex(_fi)
        _si = self.tab_label.cmb_sort.findData(str(sess.get("sort") or ""))
        if _si >= 0:
            self.tab_label.cmb_sort.setCurrentIndex(_si)
        try:
            _ti = int(sess.get("tab") or 0)
        except (TypeError, ValueError):
            _ti = 0
        self.tabs.setCurrentIndex(max(0, min(_ti, self.tabs.count() - 1)))
        _t = sess.get("train") or {}
        if isinstance(_t, dict):
            for _k, _sp in (("epochs", self.tab_train.sp_epochs),
                            ("imgsz", self.tab_train.sp_imgsz),
                            ("batch", self.tab_train.sp_batch)):
                try:
                    _sp.setValue(int(_t.get(_k, _sp.value())))
                except (TypeError, ValueError):
                    pass
            if _t.get("name"):
                self.tab_train.ed_name.setText(str(_t["name"]))
        if sess.get("cur_frame"):
            self.tab_label._pending_frame = str(sess["cur_frame"])
        if dataset_root:
            self.tab_label.ed_images.setText(str(Path(dataset_root) / "images"))
            self.tab_label.reload()
        elif self.tab_label.ed_images.text().strip():
            self.tab_label.reload()               # 有目录（会话的 ✓）就载入 ✓

    def _collect_session(self):
        """收集当前数据状态 → dict（**纯读** ✓ 好测 ✓ 落盘归 closeEvent ✓）。"""
        it = self.tab_label.lst.currentItem()
        return {
            "images_dir": self.tab_label.ed_images.text().strip(),
            "labels_dir": self.tab_label.ed_labels.text().strip(),
            "names": self.ed_names.text().strip(),
            "weights_manual": self.tab_label._yolo_weights or "",
            "weights_val": self.tab_val.ed_weights.text().strip(),
            "tab": self.tabs.currentIndex(),
            "cur_frame": (str(it.data(Qt.UserRole) or "")
                          if it is not None else ""),
            "filter": self.tab_label.cmb_filter.currentData() or "",
            "sort": self.tab_label.cmb_sort.currentData() or "",
            "train": {"epochs": self.tab_train.sp_epochs.value(),
                      "imgsz": self.tab_train.sp_imgsz.value(),
                      "batch": self.tab_train.sp_batch.value(),
                      "name": self.tab_train.ed_name.text().strip()},
        }

    def closeEvent(self, e):
        """关窗 = **保存数据状态**（本机偏好 → config/ui.yaml ✓ 同 §11 ✓）。
        ⚠ 存不上也别拦关窗 ✓；自检里 `_save_session_on_close = False` ✗
        「自检绝不许改用户的 config/ui.yaml」（§11 ✓）。"""
        if getattr(self, "_save_session_on_close", True):
            try:
                theme.set_yolo_session(self._collect_session())
            except Exception:                 # noqa: BLE001
                pass
        super().closeEvent(e)

    def _names(self):
        """类别名列表（去空 ✓）。"""
        return [t.strip() for t in self.ed_names.text().split(",") if t.strip()]


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    win = YoloWorkbench(root)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
