"""用训练好的权重推理，输出带框的图和统计。

CLI:
    python -m perception.predict --weights projects/x/models/mob_v1.pt ^
        --source data/new_frames --out data/pred --conf 0.3

GUI:
    调 run_predict(params, ctx)，见 core/context.py

**为什么不用 ultralytics 的 save=True**
    它内部走 cv2.imwrite 落盘，遇到中文路径（项目名就是中文地图名）
    会静默失败 —— 图不生成也不报错，只会让你以为"推理跑出来是空的"。
    所以这里自己画框、用 core.imgio 保存。

**为什么 source 要能指向任意目录**
    验证的意义就在于跳出训练分布。只允许看本项目的 frames，
    得到的结果和训练时的 val 指标是同一批数据，等于白验一次。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core.context import ConsoleContext, TaskContext
from core.imgio import imread, imwrite
from gui import theme
from perception.classes import ZH_NAMES

# 类别名来自 perception/classes.py；颜色来自可视化设置（与实时预览、质检台同一份）
CLASS_NAMES = ZH_NAMES


def _cls_colors():
    """类别 → BGR 颜色，和实时预览/质检台统一（每个类别的颜色都能在设置里改）。"""
    return theme.class_colors()


HISTORY = "history.json"


def _wlabel(path):
    """权重路径 → 好认的版本名：`models/detect_v3.pt` 和
    `runs/detect_v3/weights/best.pt` 都得到 `detect_v3`。"""
    p = Path(str(path or ""))
    if p.stem == "best" and p.parent.name == "weights":
        return p.parent.parent.name
    return p.stem or "?"


def _append_history(out, stats):
    """把这次验证的结果追加到 out/history.json。

    **为什么值得单独存一份**：验证的用途就是拿同一批画面试不同版本的模型，
    但输出目录是公用的、图片每次都被覆盖 —— 只看得到最后一次的数字，
    上一版是多少全靠记忆。存个小历史，卡片就能把几次摆在一起比。
    """
    f = Path(out) / HISTORY
    try:
        hist = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(hist, list):
            hist = []
    except Exception:
        hist = []
    rec = {k: stats.get(k) for k in
           ("frames", "boxes", "frames_with", "zero_frames", "median_boxes",
            "conf_median", "source", "conf_thr")}
    rec["weights"] = _wlabel(stats.get("weights"))
    rec["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    hist.append(rec)
    try:
        # 只留最近 50 次：这是给人看的对照表，不是审计日志
        f.write_text(json.dumps(hist[-50:], ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
    except Exception:
        pass


def _pct(arr, q):
    if not arr:
        return 0.0
    return float(np.percentile(arr, q))


def run_predict(params, ctx=None):
    """推理 + 可视化。

    params:
        weights    权重文件
        source     画面目录（任意目录，不限于本项目的 frames）
        out        输出目录（带框的图写这里）
        conf       置信度阈值
        iou        NMS 的 IoU 阈值
        imgsz      推理尺寸
        device     设备
        limit      只跑前 N 张，0 = 全部

    返回 dict，卡片读它渲染摘要。
    """
    ctx = ctx or TaskContext()

    weights = Path(params.get("weights") or "")
    if not weights.exists():
        raise FileNotFoundError("权重不存在：%s\n请先完成 ⑦ 训练" % weights)

    src = Path(params.get("source") or "")
    if not src.is_dir():
        raise FileNotFoundError("画面目录不存在：%s" % src)

    frames = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError("画面目录里没有 png/jpg：%s" % src)

    limit = int(params.get("limit", 0) or 0)
    if limit:
        frames = frames[:limit]

    out = Path(params["out"])
    out.mkdir(parents=True, exist_ok=True)

    conf = float(params.get("conf", 0.30))
    iou = float(params.get("iou", 0.45))
    imgsz = int(params.get("imgsz", 960))
    device = str(params.get("device", "0"))

    ctx.log("权重     %s" % weights)
    ctx.log("画面     %s（%d 张）" % (src, len(frames)))
    ctx.log("置信度 %.2f   NMS %.2f   尺寸 %d   设备 %s"
            % (conf, iou, imgsz, device))
    ctx.log("输出     %s" % out)
    ctx.log("")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("没有装 ultralytics。\n请执行：pip install ultralytics")

    ctx.progress(0, 0, "加载模型…")
    model = YOLO(str(weights))
    ctx.progress(0, len(frames), "开始推理")

    t0 = time.perf_counter()
    counts = []          # 每帧框数
    confs = []           # 每个框的置信度
    n_boxes = 0
    n_with = 0
    n_read_fail = 0

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
        img = imread(path) if path.name else None
        if img is None:
            n_read_fail += 1
            ctx.log("读不到 %s，跳过" % path.name, "warn")
            continue

        k = 0
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            cfs = boxes.conf.cpu().numpy()
            try:
                clss = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                clss = [1] * len(cfs)
            colors = _cls_colors()
            for (x1, y1, x2, y2), c, cls in zip(xyxy, cfs, clss):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color = colors.get(int(cls), (60, 220, 60))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # 类别名 + 置信度标在框上方，越界就挪到框内
                ty = y1 - 5 if y1 > 14 else y1 + 16
                cv2.putText(img, "%s %.2f" % (CLASS_NAMES.get(int(cls), str(int(cls))), c),
                            (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)
                confs.append(float(c))
                k += 1

        imwrite(out / (path.stem + ".jpg"), img, quality=88)

        counts.append(k)
        n_boxes += k
        if k:
            n_with += 1

        if i % 20 == 0 or i == len(frames):
            ctx.progress(i, len(frames), "已检出 %d 框" % n_boxes)
            ctx.log("  [%d/%d]  检出 %d  用时 %.0fs"
                    % (i, len(frames), n_boxes, time.perf_counter() - t0))

    dt = time.perf_counter() - t0
    n = len(counts)

    stats = {
        "frames": n,
        "boxes": n_boxes,
        "frames_with": n_with,
        "zero_frames": sum(1 for c in counts if c == 0),
        "max_boxes": max(counts) if counts else 0,
        "median_boxes": _pct(counts, 50),
        "conf_p10": _pct(confs, 10),
        "conf_median": _pct(confs, 50),
        "conf_min": min(confs) if confs else 0.0,
        "weights": str(weights),
        "source": str(src),
        "conf_thr": conf,
    }

    try:
        with open(out / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    _append_history(out, stats)

    ctx.log("")
    ctx.log("── 验证完成 ──", "ok")
    ctx.log("  模型 %s" % _wlabel(stats.get("weights")))
    ctx.log("  画面 %d 张 / 检出 %d 框 / 有检出 %d 帧（%.0f%%）"
            % (n, n_boxes, n_with, 100.0 * n_with / max(1, n)))
    if counts:
        ctx.log("  每帧框数: 中位 %.0f，最多 %d，空帧 %d"
                % (stats["median_boxes"], stats["max_boxes"],
                   stats["zero_frames"]))
    if confs:
        ctx.log("  框置信度: 中位 %.3f（最低 %.3f，p10 %.3f）"
                % (stats["conf_median"], stats["conf_min"], stats["conf_p10"]))
    if n_read_fail:
        ctx.log("  %d 张读不到，已跳过" % n_read_fail, "warn")
    ctx.log("  用时 %.1fs（%.0f 帧/秒）" % (dt, n / max(dt, 0.001)))
    ctx.log("")

    # 诊断提示 —— 别让人自己去猜数字
    if not counts:
        ctx.log("一张都没跑成 —— 检查画面目录", "error")
    elif n_with == 0:
        ctx.log("没有任何帧检出目标。可能是：换了地图导致模型不适应、"
                "置信度阈值过高、或画面尺寸与训练差异过大", "warn")
    elif stats["zero_frames"] > n * 0.5:
        ctx.log("过半帧没有检出（%d/%d）—— 和训练时的表现差距很大，"
                "多半是这批画面和训练素材不是同一个分布"
                % (stats["zero_frames"], n), "warn")
    elif confs and stats["conf_p10"] < 0.4:
        ctx.log("有一成以上的框置信度低于 0.4 —— 模型对这些目标不太确定，"
                "打开可视化图看看是哪一类：%s" % out, "warn")

    return {
        "frames": n,
        "boxes": n_boxes,
        "frames_with": n_with,
        "stats": stats,
        "out_dir": str(out),
        "seconds": dt,
        "summary": "%d 帧 / %d 框 / 有检出 %.0f%%"
                   % (n, n_boxes, 100.0 * n_with / max(1, n)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="data/pred")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    try:
        run_predict({
            "weights": a.weights,
            "source": a.source,
            "out": a.out,
            "conf": a.conf,
            "iou": a.iou,
            "imgsz": a.imgsz,
            "device": a.device,
            "limit": a.limit,
        }, ConsoleContext())
    except Exception as e:
        print("[predict] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
