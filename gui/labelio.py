"""标注文件的读写。

**数据分三层**（这是刻意的，不是多余的目录）：

    labels_auto/     自动标注框（每次自动标注生成/更新，质检台删自动框时同步更新）
    labels/          人工框（质检台新增/拖动过的框，重标时保留、不被覆盖）
    labels_backup/   每次保存人工框前的旧版本

读取时**按框合并**：人工框优先，自动框里和人工框 IoU 过高的丢弃。

**类别编号**（和 dataset/data.yaml 的 names 对齐）：

    0 = player   角色
    1 = mob      怪物
    2 = drop     掉落物
    3 = npc      NPC

标注文件是 YOLO 格式：`cls cx cy w h`（归一化坐标）。
框在内存里统一是 `(cls, x, y, w, h, manual)`（像素坐标），manual 标记是否人工。
"""
import json
import time
from pathlib import Path

# 类别表的唯一定义处是 perception/classes.py（id / 英文名 / 中文名 / 颜色都在那），
# 这里只转发，保持既有的名字 —— 别处（质检台、实时预览、验证、数据集导出）都从
# 这里或那里取，加类别时只用改那一个文件。
from perception.classes import (CLASS_DROP, CLASS_MOB, CLASS_NPC,
                                CLASS_OTHER_PLAYER, CLASS_PLAYER,
                                EN_NAMES as CLASS_NAMES, ORDER, ZH_NAMES, label)


def _files(project, stem):
    return (project.dir_of("labels_auto") / (stem + ".txt"),
            project.dir_of("labels") / (stem + ".txt"))


def _xywh_iou(a, b):
    """两个 (x, y, w, h) 框的 IoU。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merged_raw(project, stem):
    """合并 labels_auto 与 labels 的原始行，返回 [(cls, x, y, w, h, manual)]（归一化左上角坐标）。

    自动框若和某个人工框 IoU 过高则丢弃 —— 人工框已覆盖该区域。
    """
    auto, manual_path = _files(project, stem)

    def _read(path, manual):
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return []
        out = []
        for ln in text.splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            out.append((cls, cx - bw / 2.0, cy - bh / 2.0, bw, bh, manual))
        return out

    manual_boxes = _read(manual_path, True)
    auto_boxes = _read(auto, False)

    kept_auto = [b for b in auto_boxes
                 if not any(_xywh_iou(b[1:5], m[1:5]) > 0.5
                            for m in manual_boxes)]
    return kept_auto + manual_boxes


def load_boxes(project, stem, img_w, img_h):
    """读一帧的标注，返回 [(cls, x, y, w, h, manual)]，像素坐标。

    合并 labels_auto（自动框，manual=False）与 labels（人工框，manual=True）。
    """
    out = []
    for cls, x, y, w, h, manual in _merged_raw(project, stem):
        out.append((cls, x * img_w, y * img_h, w * img_w, h * img_h, manual))
    return out


def save_boxes(project, stem, boxes, img_w, img_h):
    """写编辑结果。boxes: [(cls, x, y, w, h, manual)]，像素坐标。

    人工框（manual=True）写 labels/，自动框（manual=False）写 labels_auto/。
    写 labels/ 前先把旧版本挪进 labels_backup/。返回人工框数。
    """
    _auto, manual = _files(project, stem)
    manual.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(box):
        cls, x, y, w, h, _m = box
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        return "%d %.6f %.6f %.6f %.6f" % (cls, cx, cy, w / img_w, h / img_h)

    manual_boxes = [b for b in boxes if len(b) > 5 and b[5]]
    auto_boxes = [b for b in boxes if not (len(b) > 5 and b[5])]

    # 自动框写 labels_auto（含删除的自动框 —— 编辑后自动部分以此为准）
    auto_text = "\n".join(_fmt(b) for b in auto_boxes)
    _auto.parent.mkdir(parents=True, exist_ok=True)
    _auto.write_text((auto_text + "\n") if auto_text else "\n", encoding="utf-8")

    # 人工框写 labels/（先备份旧的）
    if manual.exists():
        bak = project.dir_of("labels_backup") / (stem + ".txt")
        try:
            bak.write_text(manual.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    if manual_boxes:
        manual.write_text("\n".join(_fmt(b) for b in manual_boxes) + "\n",
                          encoding="utf-8")
    else:
        if manual.exists():
            try:
                manual.unlink()
            except Exception:
                pass

    return len(manual_boxes)


def revert_frame(project, stem):
    """丢掉人工修正，回到自动标注的结果。"""
    _auto, manual = _files(project, stem)
    if not manual.exists():
        return False
    try:
        manual.unlink()
        return True
    except Exception:
        return False


def is_manual(project, stem):
    """这帧被人工改过吗？"""
    _auto, manual = _files(project, stem)
    return manual.exists()


def delete_frame(project, stem):
    """剔除一帧：四处文件一起删。

    只删一处会让"帧"和"标注"对不上号 —— 那种错误很难从数字上看出来。
    """
    targets = [
        project.frames / (stem + ".png"),
        project.dir_of("labels_auto") / (stem + ".txt"),
        project.dir_of("labels") / (stem + ".txt"),
        project.vis / (stem + ".jpg"),
    ]

    n = 0
    for t in targets:
        if t.exists():
            try:
                t.unlink()
                n += 1
            except Exception:
                pass
    return n


def count_boxes(project, stem):
    """只要框数，返回 (总框数, 是否含人工)。合并统计（自动 + 人工，去重）。"""
    raw = _merged_raw(project, stem)
    return len(raw), any(b[5] for b in raw)


def frame_summary(project, stem):
    """一帧的概况：(按类的框数, 是否含人工框)。

    选帧弹窗要给每帧列一行（「怪 3 · 玩 1   已处理」），逐项调 count_boxes /
    count_by_class 会把同一帧读好几遍盘 —— 这里一次读完给全。
    """
    counts = {}
    manual = False
    for b in _merged_raw(project, stem):
        counts[b[0]] = counts.get(b[0], 0) + 1
        manual = manual or bool(b[5])
    return counts, manual


def count_by_class(project, stem):
    """按类别统计框数，返回 {cls: n}。

    给质检台的「玩家框丢失或重复」这类筛选用 —— 那边关心的是**数量**（恰好 1 个
    才算正常），所以这里返回计数而不是布尔。labels/ 与 labels_auto/ 合并统计。
    """
    out = {}
    for b in _merged_raw(project, stem):
        out[b[0]] = out.get(b[0], 0) + 1
    return out


# ---------------------------------------------------------------- 处理台账
#
# 「这一帧的某类标注跑过没有」光看标注文件在不在是判断不了的：
#   1. 怪物 pass 和玩家 pass 写的是**同一个 txt**（各覆盖自己那一类、保留另一类）；
#   2. 零框的帧也会写一个空文件（见 tools/detect_mobs.py 的落盘处）。
# 所以另存一份台账：labels_auto/_state.json
#
#     {"mob":    {"frames": ["frame_00000", ...], "last_run": "2026-09-23 21:40"},
#      "player": {"frames": [...], "last_run": "..."}}
#
# **只记「跑过哪些帧」，不记框数** —— 框数随时能从标注文件读（count_by_class），
# 台账便不跟着编辑动作变，少一处会不同步的地方。
STATE_NAME = "_state.json"


def state_path(labels_dir):
    return Path(labels_dir) / STATE_NAME


def read_state(labels_dir):
    """读处理台账；文件不存在或坏了都当空的。"""
    try:
        with open(state_path(labels_dir), "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def processed_set(labels_dir, target):
    """某一类标注「跑过」的帧名集合。"""
    return set((read_state(labels_dir).get(target) or {}).get("frames") or [])


def mark_processed(labels_dir, target, stems):
    """把一批帧记成「target 这类标注跑过了」（与已有台账取并集后落盘）。"""
    d = read_state(labels_dir)
    cur = set((d.get(target) or {}).get("frames") or [])
    cur |= {str(s) for s in stems}
    d[target] = {"frames": sorted(cur),
                 "last_run": time.strftime("%Y-%m-%d %H:%M")}
    _write_state(labels_dir, d)


def unmark_processed(labels_dir, stems):
    """帧被剔除（图都删了）时，把它从**所有**类的台账里删掉。"""
    gone = {str(s) for s in stems}
    d = read_state(labels_dir)
    for v in d.values():
        if not isinstance(v, dict):
            continue
        v["frames"] = [s for s in (v.get("frames") or []) if s not in gone]
    _write_state(labels_dir, d)


def _write_state(labels_dir, d):
    try:
        p = state_path(labels_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
    except Exception:
        pass


def label_progress(project):
    """统计各类标注进度，返回 {mob, player, total}。

    total = frames 目录里的 png 数；mob / player = 有对应类框的帧数
    （labels/ 人工优先，labels_auto/ 回退）。给④自动标注卡片的进度摘要用。
    """
    total = mob = player = 0
    for f in sorted(project.frames.glob("*.png")):
        total += 1
        by_cls = count_by_class(project, f.stem)
        if by_cls.get(CLASS_MOB, 0) > 0:
            mob += 1
        if by_cls.get(CLASS_PLAYER, 0) > 0:
            player += 1
    return {"mob": mob, "player": player, "total": total}
