"""迭代自训练：用当前模型的预测重写标注，再重训一轮。

**原理**
    模板匹配的召回被"姿态必须和模板一致"卡死，本项目实测 4.07 框/帧。
    但训出来的 YOLO 学的是外观特征，同一批画面能给到 6 框/帧 ——
    多出来的正是模板匹配漏掉的目标。用模型的预测当新标注再训一轮，
    就是标准的「伪标注 + 自训练」。

**风险：错误会自我强化**
    模型漏掉的目标，在新标注里同样不存在，下一轮会漏得更多。
    这是伪标注的固有陷阱，不能靠调参绕过，只能靠纪律：

      1. 置信度阈值设高（默认 0.6）—— 只信模型有把握的那部分
      2. labels/ 里人工修正过的帧**整帧跳过**，一个字节都不动
      3. 新标注写到 labels_iter/，labels_auto/ 保持原样 ——
         随时能回退、能把两套标注各训一个模型做对比

**为什么不把原标注和预测合并**
    合并会把模板匹配的误检（背景纹理误匹配出来的框）一起带进新一轮，
    而那正是这次迭代要除掉的东西。

**模型完全没检出的帧**
    默认保留原标注（keep_empty=True）。"模型没检出"更可能是模型的问题，
    不是画面里真没怪 —— 直接清空等于主动丢掉一批样本。
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from core.context import ConsoleContext, TaskContext

CLASS_MOB = 1


def _count_lines(path):
    try:
        return sum(1 for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
    except Exception:
        return 0


def run_relabel(params, ctx=None):
    """用模型预测生成新一轮标注。

    params:
        weights      权重文件
        frames       画面目录
        out          新标注输出目录（建议 labels_iter）
        src_labels   原标注目录，模型没检出时回退用它
        protect      受保护的标注目录列表（人工修正的），这些帧整帧跳过
        conf         置信度阈值，默认 0.60
        iou          NMS 的 IoU 阈值
        imgsz        推理尺寸
        device       设备
        limit        只跑前 N 张，0 = 全部
        keep_empty   模型没检出的帧，是否保留原标注，默认 True

    返回 dict，卡片读它渲染摘要。
    """
    ctx = ctx or TaskContext()

    weights = Path(params.get("weights") or "")
    if not weights.exists():
        raise FileNotFoundError("权重不存在：%s\n请先完成 ⑦ 训练" % weights)

    frames_dir = Path(params.get("frames") or "")
    if not frames_dir.is_dir():
        raise FileNotFoundError("画面目录不存在：%s" % frames_dir)

    frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError("画面目录里没有图：%s" % frames_dir)

    limit = int(params.get("limit", 0) or 0)
    if limit:
        frames = frames[:limit]

    out = Path(params["out"])
    out.mkdir(parents=True, exist_ok=True)

    raw_protect = params.get("protect") or []
    if isinstance(raw_protect, (str, Path)):
        raw_protect = [raw_protect]
    protect_dirs = [Path(x) for x in raw_protect if str(x).strip()]
    protect_dirs = [d for d in protect_dirs if d.is_dir()]

    src_labels = Path(params.get("src_labels") or "")
    conf = float(params.get("conf", 0.60))
    iou = float(params.get("iou", 0.45))
    imgsz = int(params.get("imgsz", 960))
    device = str(params.get("device", "0"))
    keep_empty = bool(params.get("keep_empty", True))

    # 先把受保护的帧登记出来 —— 它们在主循环里整帧跳过
    protected = {}
    for d in protect_dirs:
        for f in d.glob("*.txt"):
            protected.setdefault(f.stem, f)

    ctx.log("权重       %s" % weights)
    ctx.log("画面       %s（%d 张）" % (frames_dir, len(frames)))
    ctx.log("置信度阈值 %.2f" % conf)
    ctx.log("原标注回退 %s" % (src_labels if src_labels.is_dir() else "（无）"))
    if protected:
        ctx.log("保护 %d 帧（人工修正过，整帧跳过）" % len(protected))
    ctx.log("输出       %s" % out)
    ctx.log("")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("没有装 ultralytics。\n请执行：pip install ultralytics")

    ctx.progress(0, 0, "加载模型…")
    model = YOLO(str(weights))
    ctx.progress(0, len(frames), "开始推理")

    t0 = time.perf_counter()
    n_protected = n_from_model = n_kept = n_cleared = 0
    boxes_model = boxes_kept = 0
    counts = []

    stream = model.predict(
        source=[str(f) for f in frames],
        conf=conf, iou=iou, imgsz=imgsz, device=device,
        verbose=False, stream=True,
    )

    for i, r in enumerate(stream, 1):
        if ctx.canceled():
            ctx.log("已取消", "warn")
            break

        path = Path(getattr(r, "path", "") or "")
        stem = path.stem
        if not stem:
            continue

        # ── 防线 2：人工修正过的帧，原样搬过去，一个字节都不动 ──
        src = protected.get(stem)
        if src is not None:
            try:
                shutil.copy2(str(src), str(out / (stem + ".txt")))
                n_protected += 1
            except Exception as e:
                ctx.log("复制 %s 失败：%s" % (stem, e), "warn")
            continue

        # ── 模型预测。xywhn 已经是归一化的 (cx, cy, w, h)，正好是 YOLO 格式 ──
        lines = []
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes):
            for cx, cy, w, h in boxes.xywhn.cpu().numpy():
                lines.append("%d %.6f %.6f %.6f %.6f"
                             % (CLASS_MOB, cx, cy, w, h))

        target = out / (stem + ".txt")

        if lines:
            target.write_text("\n".join(lines), encoding="utf-8")
            n_from_model += 1
            boxes_model += len(lines)
            counts.append(len(lines))
        else:
            # ── 模型没检出：默认保留原标注 ──
            orig = src_labels / (stem + ".txt") if src_labels.is_dir() else None
            if keep_empty and orig is not None and orig.exists():
                n = _count_lines(orig)
                if n:
                    try:
                        shutil.copy2(str(orig), str(target))
                        n_kept += 1
                        boxes_kept += n
                        counts.append(n)
                        continue
                    except Exception:
                        pass
            # 既没预测也没原标注 —— 写成空文件（占位，避免被当成"没处理过"）
            target.write_text("", encoding="utf-8")
            n_cleared += 1
            counts.append(0)

        if i % 20 == 0 or i == len(frames):
            ctx.progress(i, len(frames), "已写入 %d 帧" % (n_from_model + n_kept))
            ctx.log("  [%d/%d]  模型标注 %d  保留原标 %d  空 %d"
                    % (i, len(frames), n_from_model, n_kept, n_cleared))

    dt = time.perf_counter() - t0

    total_boxes = boxes_model + boxes_kept
    stats = {
        "frames": len(counts),
        "boxes": total_boxes,
        "from_model": n_from_model,
        "boxes_from_model": boxes_model,
        "protected": n_protected,
        "kept_orig": n_kept,
        "boxes_kept": boxes_kept,
        "cleared": n_cleared,
        "median_boxes": float(np.percentile(counts, 50)) if counts else 0.0,
        "conf_thr": conf,
        "weights": str(weights),
    }

    try:
        with open(out / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    ctx.log("")
    ctx.log("── 迭代标注完成 ──", "ok")
    ctx.log("  保护未动 %d 帧（人工修正）" % n_protected)
    ctx.log("  模型重写 %d 帧 / %d 框" % (n_from_model, boxes_model))
    if n_kept:
        ctx.log("  保留原标 %d 帧 / %d 框（模型在这几帧什么都没检出）"
                % (n_kept, boxes_kept))
    if n_cleared:
        ctx.log("  清空 %d 帧" % n_cleared)
    if counts:
        ctx.log("  每帧框数中位 %.0f" % stats["median_boxes"])
    ctx.log("  用时 %.1f 分钟" % (dt / 60.0))
    ctx.log("")
    ctx.log("  下一步：⑥ 数据集会自动优先用这批标注，重新整理后 ⑦ 再训一轮。",
            "ok")

    # 别让"框变多了"被当成纯粹的好消息 —— 它同时意味着误检也可能变多
    if n_from_model and counts and stats["median_boxes"] > 0:
        ctx.log("  注意：框数上升说明补回了漏检，但新一轮模型的误检也会被写进来。"
                "建议重训后和旧模型在同一个验证集上比一次。", "warn")
    ctx.log("")

    return {
        "frames": len(counts),
        "boxes": total_boxes,
        "protected": n_protected,
        "from_model": n_from_model,
        "kept_orig": n_kept,
        "stats": stats,
        "out_dir": str(out),
        "seconds": dt,
        "summary": "%d 帧重写 · %d 框 · 保护 %d 帧"
                   % (n_from_model, total_boxes, n_protected),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-labels", default=None)
    ap.add_argument("--protect", nargs="*", default=[])
    ap.add_argument("--conf", type=float, default=0.60)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-keep-empty", dest="keep_empty",
                    action="store_false", default=True)
    a = ap.parse_args()

    try:
        run_relabel({
            "weights": a.weights,
            "frames": a.frames,
            "out": a.out,
            "src_labels": a.src_labels,
            "protect": a.protect,
            "conf": a.conf,
            "iou": a.iou,
            "imgsz": a.imgsz,
            "device": a.device,
            "limit": a.limit,
            "keep_empty": a.keep_empty,
        }, ConsoleContext())
    except Exception as e:
        print("[relabel] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
