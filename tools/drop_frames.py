"""剔除有问题的帧。

**两种模式**

  dark —— 自动找出被窗口 / UI 遮挡的帧。
        原理：正常游戏画面明亮，暗像素（<60）比例稳定在 13% 左右；
        被深色窗口盖住的帧会飙到 40%~70%，一测就出来。

  list —— 按帧名精确剔除（人工判断后使用）。

**必须三处一起删**
     frames / labels_auto / vis 三个目录是按帧名对应的。
    只删一处，后面的步骤就会读到对不上的帧和标注。

CLI:
    python -m tools.drop_frames --frames projects/xxx/frames --mode dark --threshold 0.25
    python -m tools.drop_frames --frames projects/xxx/frames --mode list --names frame_00000,frame_00393
"""

import argparse
import sys
from pathlib import Path

from core.context import ConsoleContext, TaskContext
from core.imgio import imread


def run_drop(params, ctx=None):
    ctx = ctx or TaskContext()

    frames_dir = Path(params["frames"])
    if not frames_dir.is_dir():
        raise FileNotFoundError("画面目录不存在：%s" % frames_dir)

    project = frames_dir.parent
    labels_dir = Path(params.get("labels") or (project / "labels_auto"))
    vis_dir = Path(params.get("vis") or (project / "vis"))

    mode = params.get("mode", "dark")
    names = []

    # ---------------- 挑出要删的帧 ----------------
    if mode == "list":
        names = [str(x).strip() for x in (params.get("names") or []) if str(x).strip()]
        if not names:
            raise ValueError("mode=list 时必须给 names")

    else:
        threshold = float(params.get("threshold", 0.25))
        ctx.log("扫描画面，找暗像素比例 > %.2f 的帧（疑似被窗口遮挡）" % threshold)

        all_frames = sorted(frames_dir.glob("*.png"))
        ratios = []

        for i, f in enumerate(all_frames, 1):
            if ctx.canceled():
                ctx.log("已取消", "warn")
                return {"summary": "已取消"}
            im = imread(f, 0)
            if im is None:
                continue
            ratios.append((float((im < 60).mean()), f.stem))
            if i % 100 == 0:
                ctx.progress(i, len(all_frames), "已扫描 %d" % i)

        if not ratios:
            raise RuntimeError("一张图都读不了")

        vals = sorted(r for r, _ in ratios)
        median = vals[len(vals) // 2]
        ctx.log("暗像素比例：中位 %.3f，最高 %.3f" % (median, vals[-1]))

        for r, s in sorted(ratios, reverse=True):
            if r > threshold:
                names.append(s)

        if not names:
            ctx.log("没有需要剔除的帧", "ok")
            return {"removed": 0, "summary": "没有异常帧"}

        ctx.log("")
        ctx.log("疑似被遮挡的帧 %d 个：" % len(names))
        for r, s in sorted(ratios, reverse=True):
            if r > threshold:
                ctx.log("   %-18s 暗像素 %.3f" % (s, r))

    # ---------------- 三处一起删 ----------------
    ctx.log("")
    removed = 0
    missing = 0

    for stem in names:
        for d, ext in ((frames_dir, ".png"), (labels_dir, ".txt"), (vis_dir, ".jpg")):
            p = d / (stem + ext)
            if not p.exists():
                missing += 1
                continue
            try:
                p.unlink()
                removed += 1
            except Exception as e:
                ctx.log("删不掉 %s: %s" % (p, e), "warn")

    left = len(list(frames_dir.glob("*.png")))
    left_lbl = len(list(labels_dir.glob("*.txt"))) if labels_dir.is_dir() else 0

    summary = "剔除 %d 帧（%d 个文件）" % (len(names), removed)
    ctx.log("")
    ctx.log(summary, "ok")
    ctx.log("  剩余画面 %d 张 / 剩余标注 %d 个" % (left, left_lbl))

    if left_lbl and left_lbl != left:
        ctx.log("注意：画面和标注数量对不上（%d vs %d）—— 需要重跑 ④ 让它们重新对齐"
                % (left, left_lbl), "warn")

    return {
        "removed": len(names),
        "files": removed,
        "missing": missing,
        "left": left,
        "summary": summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="项目的 frames 目录")
    ap.add_argument("--mode", default="dark", choices=["dark", "list"])
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--names", default="", help="逗号分隔的帧名（不含扩展名）")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--vis", default=None)
    ap.add_argument("--dry-run", action="store_true", help="只看会删哪些，不真删")
    a = ap.parse_args()

    if a.dry_run:
        from core.context import CollectContext
        # 借用同一个逻辑，但把删除部分跳过
        frames_dir = Path(a.frames)
        threshold = a.threshold
        found = []
        for f in sorted(frames_dir.glob("*.png")):
            im = imread(f, 0)
            if im is None:
                continue
            r = float((im < 60).mean())
            if r > threshold:
                found.append((r, f.stem))
        found.sort(reverse=True)
        print("会剔除 %d 帧：" % len(found))
        for r, s in found:
            print("   %-18s 暗像素 %.3f" % (s, r))
        return 0

    params = {
        "frames": a.frames,
        "mode": a.mode,
        "threshold": a.threshold,
        "names": [x.strip() for x in a.names.split(",") if x.strip()],
        "labels": a.labels,
        "vis": a.vis,
    }

    try:
        run_drop(params, ConsoleContext())
    except Exception as e:
        print("[drop] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
