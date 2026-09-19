"""把标注整理成 YOLO 训练集结构。

输入: frames(png) + labels(txt，可多层)
输出: <out>/images/{train,val}/*.jpg
      <out>/labels/{train,val}/*.txt
      <out>/data.yaml

CLI:
    python -m perception.prepare_dataset --frames data/plain2 ^
        --labels projects/x/labels projects/x/labels_auto --out datasets/yolo

GUI:
    调 run_dataset(params, ctx)，见 core/context.py

**为什么要支持多层标注目录**
    标注分两层：labels/ 是人工修正过的，labels_auto/ 是原始自动结果。
    人工通常只改一部分帧，所以必须「人工优先、其余回退自动」——
    只取一层，要么丢掉人工的心血，要么把 90% 的样本丢掉。
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

from core.context import ConsoleContext, TaskContext
from core.imgio import imread, imwrite

CLASS_NAMES = ["player", "mob", "drop", "npc"]


def _merge_label_lines(stem, label_dirs):
    """对一帧按「类」合并多层标注，返回合并后的 YOLO 行列表（或 None）。

    每类独立找「最高优先级里含该类」的目录，取该目录里该类的**全部**框。
    不同类可以来自不同层：

        怪物框（class 1）: labels → labels_iter → labels_auto
        角色框（class 0）: labels → labels_iter → labels_auto

    这样不会因为某一层缺某一类就整类丢失。典型场景：labels_iter 是旧模型
    迭代产物，只有怪物框没有角色框 —— 若按整帧优先，角色框会被它顶掉，
    只剩怪物的标注把「角色」这一类活活吞了。按类合并后，角色框自动回退到
    labels_auto 补齐。
    """
    per_dir = []  # [(dir, lines)]，保持 label_dirs 传入的优先级顺序
    for d in label_dirs:
        cand = Path(d) / (stem + ".txt")
        if not cand.exists():
            continue
        try:
            lines = [ln for ln in cand.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
        except Exception:
            continue
        if lines:
            per_dir.append((d, lines))

    if not per_dir:
        return None

    all_cls = set()
    for _d, lines in per_dir:
        for ln in lines:
            parts = ln.split()
            if parts:
                try:
                    all_cls.add(int(parts[0]))
                except ValueError:
                    pass

    merged = []
    for cls in sorted(all_cls):
        for _d, lines in per_dir:
            cls_lines = [ln for ln in lines
                         if ln.split() and ln.split()[0] == str(cls)]
            if cls_lines:
                merged.extend(cls_lines)
                break
    return merged


def _collect_pairs(frames_dir, label_dirs, min_boxes):
    """收集 (图, 合并标注文本) 对。label_dirs 按优先级排列，合并按类进行。

    返回 (pairs, 缺标注数, 框数不足数)。
    """
    pairs = []
    no_label = empty = 0

    for f in sorted(Path(frames_dir).glob("*.png")):
        merged = _merge_label_lines(f.stem, label_dirs)
        if merged is None:
            no_label += 1
            continue
        if len(merged) < min_boxes:
            empty += 1
            continue
        pairs.append((f, merged))

    return pairs, no_label, empty


def _count_boxes(label_dirs, split):
    n = 0
    for d in label_dirs:
        sd = Path(d) / split
        if not sd.is_dir():
            continue
        for f in sd.glob("*.txt"):
            try:
                n += sum(1 for x in f.read_text(encoding="utf-8").splitlines() if x.strip())
            except Exception:
                pass
    return n


def run_dataset(params, ctx=None):
    """整理数据集。

    params:
        frames       画面目录
        labels       标注目录，可以是 list（按优先级）或单个路径
        out          输出目录
        val_ratio    验证集比例
        min_boxes    少于这个框数的帧丢弃；0 = 保留空帧（当负样本）
        quality      jpg 质量
        seed         随机种子
        class_names  类别名，默认 player/mob/drop/npc

    返回 dict，界面卡片读它渲染摘要。
    """
    ctx = ctx or TaskContext()

    frames_dir = Path(params["frames"])
    if not frames_dir.is_dir():
        raise FileNotFoundError("画面目录不存在：%s" % frames_dir)

    raw = params.get("labels") or []
    if isinstance(raw, (str, Path)):
        raw = [raw]
    label_dirs = [Path(x) for x in raw if str(x).strip()]
    if not label_dirs:
        raise ValueError("没有指定标注目录")

    existing = [d for d in label_dirs if d.is_dir()]
    if not existing:
        raise FileNotFoundError("标注目录都不存在：%s"
                                % ", ".join(str(d) for d in label_dirs))

    out = Path(params["out"])
    val_ratio = float(params.get("val_ratio", 0.2))
    min_boxes = int(params.get("min_boxes", 1))
    quality = int(params.get("quality", 92))
    seed = int(params.get("seed", 42))
    names = params.get("class_names") or CLASS_NAMES

    ctx.log("画面 %s" % frames_dir)
    for i, d in enumerate(existing):
        ctx.log("标注 %s  （%d 个 · %s）"
                % (d, sum(1 for _ in d.glob("*.txt")),
                   "优先" if i == 0 else "回退"))

    pairs, no_label, empty = _collect_pairs(frames_dir, existing, min_boxes)
    ctx.log("")
    ctx.log("可用 %d 对   缺标注 %d   框数不足 %d" % (len(pairs), no_label, empty))

    if not pairs:
        raise RuntimeError(
            "没有可用的帧。\n"
            "常见原因：标注目录选错、或 min_boxes 比实际框数还大。")

    if empty and min_boxes > 0:
        ctx.log("（框数不足的帧被丢弃。这些多半是漏检，留着等于教模型"
                "「这里没有怪」，所以宁可不要）", "warn")

    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))

    # 清掉上一次的产物。不清的话旧图会混进新数据集，而且是**静默的** ——
    # train 数量看起来变多了，实际混了过期样本，训练结果没法解释。
    #
    # **为什么不原地删除、而是改名**
    #     images/ 是几百张 1920x1080 的 jpg（约 200MB）。实测在某些机器上，
    #     删除这些文件会被杀软实时扫描或索引服务逐个拖住，几百个小文件就能
    #     卡住一分钟以上。而这段时间唯一的日志是"正在清理"，看起来就是死机；
    #     用户一旦中断，就会留下"images 删了、labels 没删"的半残数据集。
    #
    #     改名（rename）是元数据操作，瞬时完成 —— 旧数据完整留在旁边，
    #     确认新数据集没问题后由用户自己删，不需要在这里冒险。
    out_bak = None
    if out.exists():
        if ctx.canceled():
            ctx.log("已在清理前取消，数据集保持原样", "warn")
            raise RuntimeError("已取消")

        out_bak = out.parent / (out.name + "_old")
        shutil.rmtree(out_bak, ignore_errors=True)      # 清掉更早的备份
        try:
            out.rename(out_bak)
            ctx.log("旧数据集已改名为 %s/（确认新数据集无误后可手动删除）"
                    % out_bak.name)
        except Exception as e:
            # 改名失败（比如被占用）才退回原地删除
            ctx.log("旧数据集改名失败（%s），改为原地删除" % e, "warn")
            shutil.rmtree(out, ignore_errors=True)
            out_bak = None

    written = {}
    splits = (("val", pairs[:n_val]), ("train", pairs[n_val:]))

    total_all = sum(len(items) for _n, items in splits)
    done = 0

    for name, items in splits:
        idir = out / "images" / name
        ldir = out / "labels" / name
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)

        ok = 0
        for f, lines in items:
            # 必须走 imgio：cv2.imread 遇到中文路径（项目名就是中文地图名）
            # 会静默返回 None，结果是一堆图凭空消失，还不报错。
            img = imread(f)
            if img is None:
                ctx.log("  读不到 %s，跳过" % f.name, "warn")
                continue

            if ctx.canceled():
                ctx.log("已取消（已写入 %d 张）" % done, "warn")
                return {
                    "train": written.get("train", 0),
                    "val": written.get("val", 0),
                    "boxes": 0,
                    "dropped": no_label + empty,
                    "data_yaml": str(out / "data.yaml"),
                    "summary": "已取消",
                }

            imwrite(idir / (f.stem + ".jpg"), img, quality=quality)
            # 写合并后的标注（按类合并，不再是复制单份原始文件）
            (ldir / (f.stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            ok += 1
            done += 1

            # 张张都报：这里瓶颈是 jpg 编码（1920x1080 每张几十毫秒），
            # 几百张就是一两分钟。不报进度的话界面一动不动，用起来像卡死。
            ctx.progress(done, total_all, "%s %s" % (name, f.stem))

        written[name] = ok
        ctx.log("  %s: %d 张" % (name, ok))

    yaml_text = ("path: %s\n" % out.resolve().as_posix()
                 + "train: images/train\nval: images/val\n"
                 + "names:\n"
                 + "".join("  %d: %s\n" % (i, n) for i, n in enumerate(names)))

    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    boxes = _count_boxes([out / "labels"], "train") + _count_boxes([out / "labels"], "val")

    ctx.log("")
    ctx.log("data.yaml -> %s" % (out / "data.yaml").as_posix())
    ctx.log("── 数据集完成 ──", "ok")
    ctx.log("  train %d / val %d   共 %d 框"
            % (written.get("train", 0), written.get("val", 0), boxes))
    if out_bak is not None and out_bak.exists():
        ctx.log("  旧数据集留在 %s/（确认新数据集无误后可手动删除）" % out_bak.name)
    ctx.log("")

    if not boxes:
        ctx.log("一个框都没有 —— 检查标注内容，或 min_boxes 是否设得过大", "warn")

    return {
        "train": written.get("train", 0),
        "val": written.get("val", 0),
        "boxes": boxes,
        "dropped": no_label + empty,
        "data_yaml": str(out / "data.yaml"),
        "old_dir": str(out_bak) if out_bak is not None else "",
        "summary": "train %d / val %d · %d 框"
                   % (written.get("train", 0), written.get("val", 0), boxes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--labels", required=True, nargs="+",
                    help="一个或多个标注目录，靠前的优先")
    ap.add_argument("--out", default="datasets/yolo")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--min-boxes", type=int, default=1)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    try:
        run_dataset({
            "frames": a.frames,
            "labels": a.labels,
            "out": a.out,
            "val_ratio": a.val_ratio,
            "min_boxes": a.min_boxes,
            "quality": a.quality,
            "seed": a.seed,
        }, ConsoleContext())
    except Exception as e:
        print("[dataset] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
