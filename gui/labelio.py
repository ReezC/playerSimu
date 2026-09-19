"""标注文件的读写。

**数据分三层**（这是刻意的，不是多余的目录）：

    labels_auto/     自动标注的原始结果，**只读**，任何情况下都不改它
    labels/          人工修正后的结果
    labels_backup/   每次保存人工修正前的旧版本

读取时优先用 `labels/`，没有才回退 `labels_auto/`。

**类别编号**（和 dataset/data.yaml 的 names 对齐）：

    0 = player   角色
    1 = mob      怪物
    2 = drop     掉落物
    3 = npc      NPC

标注文件是 YOLO 格式：`cls cx cy w h`（归一化坐标）。
框在内存里统一是 `(cls, x, y, w, h)`（像素坐标），cls 在前面。
"""
from pathlib import Path

CLASS_PLAYER = 0
CLASS_MOB = 1
CLASS_DROP = 2
CLASS_NPC = 3
CLASS_NAMES = {0: "player", 1: "mob", 2: "drop", 3: "npc"}


def _files(project, stem):
    return (project.dir_of("labels_auto") / (stem + ".txt"),
            project.dir_of("labels") / (stem + ".txt"))


def load_boxes(project, stem, img_w, img_h):
    """读一帧的标注，返回 [(cls, x, y, w, h)]，像素坐标。

    有人工修正就用人工的，否则用自动的。
    """
    auto, manual = _files(project, stem)
    f = manual if manual.exists() else auto

    if not f.exists():
        return []

    try:
        text = f.read_text(encoding="utf-8")
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

        w = bw * img_w
        h = bh * img_h
        out.append((cls, cx * img_w - w / 2.0, cy * img_h - h / 2.0, w, h))

    return out


def save_boxes(project, stem, boxes, img_w, img_h):
    """写人工修正结果。boxes: [(cls, x, y, w, h)]，像素坐标。

    写之前先把旧版本挪进 labels_backup/。
    """
    _auto, manual = _files(project, stem)
    manual.parent.mkdir(parents=True, exist_ok=True)

    if manual.exists():
        bak = project.dir_of("labels_backup") / (stem + ".txt")
        try:
            bak.write_text(manual.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    lines = []
    for box in boxes:
        cls, x, y, w, h = box
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        lines.append("%d %.6f %.6f %.6f %.6f"
                     % (cls, cx, cy, w / img_w, h / img_h))

    manual.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


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
    """只要框数，不做坐标换算（列表筛选时批量调用，要快）。

    返回 (总框数, 是否人工)。
    """
    auto, manual = _files(project, stem)
    f = manual if manual.exists() else auto
    if not f.exists():
        return 0, False

    try:
        text = f.read_text(encoding="utf-8")
    except Exception:
        return 0, manual.exists()

    n = sum(1 for x in text.splitlines() if x.strip())
    return n, manual.exists()


def count_by_class(project, stem):
    """按类别统计框数，返回 {cls: n}。

    给「玩家框丢失」这类筛选用 —— 只关心某一类有没有，不用换算坐标。
    """
    auto, manual = _files(project, stem)
    f = manual if manual.exists() else auto
    if not f.exists():
        return {}

    try:
        text = f.read_text(encoding="utf-8")
    except Exception:
        return {}

    out = {}
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        out[cls] = out.get(cls, 0) + 1
    return out


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


# ══════════════════════════════════════════════════════════════
# 相邻帧对比：新框出现 / 旧框消失
# ══════════════════════════════════════════════════════════════
def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match_group(prev_grp, cur_grp, iou_thr):
    """同类别的一组框做 IoU 配对。入参是 [(原下标, (x,y,w,h))]。

    返回 (new_ids, lost_ids)，下标是**原框在 prev/cur 里的下标**。
    """
    pairs = []
    for pi, a in prev_grp:
        for pj, b in cur_grp:
            v = _iou(a, b)
            if v >= iou_thr:
                pairs.append((v, pi, pj))

    pairs.sort(key=lambda t: -t[0])     # 相似度最高的先配，贪心即可

    used_p, used_c = set(), set()
    for _v, pi, pj in pairs:
        if pi in used_p or pj in used_c:
            continue                    # 一对一配对，避免一个框被抢两次
        used_p.add(pi)
        used_c.add(pj)

    new_ids = [pj for _pi, pj in cur_grp if pj not in used_c]
    lost_ids = [pi for pi, _a in prev_grp if pi not in used_p]
    return new_ids, lost_ids


def match_boxes(prev, cur, iou_thr=0.30):
    """把相邻两帧的框按 IoU 配对，返回 (新增的当前框下标, 消失的上一帧框下标)。

    prev/cur 是 [(cls, x, y, w, h)]。**只在同类别之间配对** ——
    玩家框和怪框不互配，否则配出来的"新增/消失"是错的。

    **为什么不直接比框数**：
        上一帧 4 个框、这一帧也是 4 个，数量没变 —— 但可能是
        「旧的消失 1 个 + 新的冒出 1 个」。比数量会把这帧整帧漏掉。

    **为什么用 IoU 而不是位置相等**：
        怪在走，同一只怪相邻两帧能差十几个像素。
    """
    if not prev:
        return list(range(len(cur))), []
    if not cur:
        return [], list(range(len(prev)))

    p_by_cls = {}
    c_by_cls = {}
    for i, box in enumerate(prev):
        p_by_cls.setdefault(box[0], []).append((i, box[1:]))
    for j, box in enumerate(cur):
        c_by_cls.setdefault(box[0], []).append((j, box[1:]))

    new_all, lost_all = [], []
    for cls in set(p_by_cls) | set(c_by_cls):
        new_ids, lost_ids = _match_group(p_by_cls.get(cls, []),
                                         c_by_cls.get(cls, []), iou_thr)
        new_all.extend(new_ids)
        lost_all.extend(lost_ids)

    return sorted(new_all), sorted(lost_all)
