"""标注文件的读写。

**数据分三层**（这是刻意的，不是多余的目录）：

    labels_auto/     自动标注的原始结果，**只读**，任何情况下都不改它
    labels/          人工修正后的结果
    labels_backup/   每次保存人工修正前的旧版本

读取时优先用 `labels/`，没有才回退 `labels_auto/`。

**为什么不在 labels_auto 上直接改**：
  改掉之后就没法回答"人工修过之后模型好了多少" ——
  而这个问题恰恰决定了后面该往哪个方向使劲。
  分层之后，随时能拿两套标注各训一个模型做对比。
"""

from pathlib import Path

CLASS_MOB = 1


def _files(project, stem):
    return (project.dir_of("labels_auto") / (stem + ".txt"),
            project.dir_of("labels") / (stem + ".txt"))


def load_boxes(project, stem, img_w, img_h):
    """读一帧的标注，返回 [(x, y, w, h)]，像素坐标。

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
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue

        w = bw * img_w
        h = bh * img_h
        out.append((cx * img_w - w / 2.0, cy * img_h - h / 2.0, w, h))

    return out


def save_boxes(project, stem, boxes, img_w, img_h):
    """写人工修正结果。写之前先把旧版本挪进 labels_backup/。"""
    _auto, manual = _files(project, stem)
    manual.parent.mkdir(parents=True, exist_ok=True)

    if manual.exists():
        bak = project.dir_of("labels_backup") / (stem + ".txt")
        try:
            bak.write_text(manual.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    lines = []
    for x, y, w, h in boxes:
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        lines.append("%d %.6f %.6f %.6f %.6f"
                     % (CLASS_MOB, cx, cy, w / img_w, h / img_h))

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
    """只要框数，不做坐标换算（列表筛选时批量调用，要快）。"""
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


def match_boxes(prev, cur, iou_thr=0.30):
    """把相邻两帧的框按 IoU 配对，返回 (新增的当前框下标, 消失的上一帧框下标)。

    **为什么不直接比框数**：
        上一帧 4 个框、这一帧也是 4 个，数量没变 —— 但可能是
        「旧的消失 1 个 + 新的冒出 1 个」。比数量会把这帧整帧漏掉，
        而它恰恰最该看：漏检和误检同时发生。

    **为什么用 IoU 而不是位置相等**：
        怪在走，同一只怪相邻两帧能差十几个像素。要求位置一致，
        会把"移动中的怪"全部误判成"旧的消失 + 新的出现"。
        阈值 0.3 是权衡：抽帧去重后相邻画面差异已经不小，
        阈值再高会把正常位移也算成异常。
    """
    if not prev:
        return list(range(len(cur))), []
    if not cur:
        return [], list(range(len(prev)))

    pairs = []
    for i, a in enumerate(prev):
        for j, b in enumerate(cur):
            v = _iou(a, b)
            if v >= iou_thr:
                pairs.append((v, i, j))

    pairs.sort(key=lambda t: -t[0])     # 相似度最高的先配，贪心即可

    used_p, used_c = set(), set()
    for _v, i, j in pairs:
        if i in used_p or j in used_c:
            continue                        # 一对一配对，避免一个框被抢两次
        used_p.add(i)
        used_c.add(j)

    new_ids = [j for j in range(len(cur)) if j not in used_c]
    lost_ids = [i for i in range(len(prev)) if i not in used_p]
    return new_ids, lost_ids
