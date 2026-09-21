"""从录制文件里抽帧成 PNG。

CLI:
    python -m tools.extract_frames --file data/recordings/x.mkv ^
        --out data/frames --stride 3 --dedup 0.02 --limit 500

GUI:
    调 run_extract(params, ctx)，见 core/context.py
"""

import argparse
import os
import sys
from pathlib import Path

import av
import cv2
import numpy as np

from core.context import ConsoleContext, TaskContext
from core.imgio import imwrite


def _clean_project_labels(out_dir):
    """重抽帧时，同项目里的旧标注产物全部作废（画面全变了）。

    只清标注相关目录（人工修正 / 自动标注 / 备份 / 可视化 / 迭代），
    标定对比图（calib/）保留。只有 out_dir 的父目录是项目根
    （有 project.yaml）时才清理 —— 避免 CLI 抽帧到任意目录误删文件。
    """
    root = Path(out_dir).parent
    if not (root / "project.yaml").exists():
        return 0

    n = 0
    for name in ("labels", "labels_backup", "labels_auto",
                 "labels_iter", "vis", "vis_player"):
        d = root / name
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.is_file():
                try:
                    os.remove(str(f))
                    n += 1
                except OSError:
                    pass
    return n


def run_extract(params, ctx=None):
    """核心抽帧逻辑。

    params:
        file    录制文件路径
        out     输出目录
        stride  每 N 帧抽一张
        dedup   相似帧去重阈值(0~1)：亮度差超过 8 的像素占比小于它就跳过，0 = 不去重
        limit   最多保存张数，0 = 不限
        prefix  文件名前缀
    """
    ctx = ctx or TaskContext()

    src = Path(params["file"])
    if not src.exists():
        raise FileNotFoundError("找不到录制文件: %s" % src)

    out_dir = Path(params["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(params.get("stride", 10)))
    dedup = float(params.get("dedup", 0.0))
    limit = int(params.get("limit", 0))
    prefix = params.get("prefix", "frame")

    old = list(out_dir.glob(prefix + "_*.png"))

    # 重抽时清空旧帧。编号是从 0 覆盖写的，新视频比旧的短就会留下尾巴，
    # 那些帧会被后续的标定/标注当成「这次的画面」，悄悄污染数据集。
    if old and bool(params.get("clean", False)):
        ctx.log("清空输出目录里 %d 张旧的 %s_*.png" % (len(old), prefix))
        for f in old:
            try:
                f.unlink()
            except Exception:
                pass
        old = []
        # 画面重抽后，同项目里旧的人工/自动标注也全部作废（对应的是旧画面），
        # 否则质检台「人工优先」会读到错位的旧标注，看起来像「框没了」。
        removed = _clean_project_labels(out_dir)
        if removed:
            ctx.log("同时清空旧标注产物 %d 个文件（画面已重抽）" % removed)

    if old:
        ctx.log("输出目录已有 %d 张同名文件，将被覆盖" % len(old), "warn")

    ctx.log("打开 %s（%.1f MB）" % (src.name, src.stat().st_size / 1048576.0))

    container = av.open(str(src), mode="r")
    streams = container.streams.video
    if not streams:
        container.close()
        raise RuntimeError("文件里没有视频轨: %s" % src)

    stream = streams[0]
    total_frames = stream.frames or 0
    if total_frames:
        ctx.log("总帧数 %d，每 %d 帧抽 1 张（约 %d 张）"
                % (total_frames, stride, total_frames // stride + 1))
    else:
        ctx.log("每 %d 帧抽 1 张" % stride)

    saved = 0
    total = 0
    skipped = 0
    last = None
    shape = None
    canceled = False

    try:
        for i, frame in enumerate(container.decode(stream)):
            if ctx.canceled():
                canceled = True
                ctx.log("收到取消请求，已保存 %d 张" % saved, "warn")
                break

            total += 1
            # 进度条放 continue 之前：stride 倍数的帧会被 continue 跳过，
            # 放后面的话触发条件永远落在被跳过的帧上，进度条就完全不更新。
            if total_frames and total % max(1, stride * 10) == 0:
                ctx.progress(total, total_frames, "已保存 %d 张" % saved)

            if i % stride != 0:
                continue
            if limit and saved >= limit:
                break

            bgr = cv2.cvtColor(frame.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
            shape = bgr.shape[:2]

            if dedup > 0:
                small = cv2.cvtColor(cv2.resize(bgr, (80, 45)),
                                     cv2.COLOR_BGR2GRAY).astype(np.float32)
                if last is not None:
                    # 去重用「变化像素比例」而不是「平均像素差」：
                    # 大片不变背景会把小范围变化（待机动画/角色移动）稀释到
                    # 平均差里几乎测不出，导致大量有效帧被当成重复帧误删。
                    diff = np.abs(small - last)
                    changed = float(np.count_nonzero(diff > 8.0)) / float(diff.size)
                    if changed < dedup:
                        skipped += 1
                        continue
                last = small

            imwrite(out_dir / ("%s_%05d.png" % (prefix, saved)), bgr)
            saved += 1

    finally:
        container.close()

    # 编号从 0 开始覆盖写。新视频比旧素材短的话，目录里会残留上次抽的帧 ——
    # 那些帧会被后面的标定/标注当成「这次的画面」，悄悄污染数据集。
    leftover = len(list(out_dir.glob(prefix + "*.png"))) - saved
    if leftover > 0:
        ctx.log("注意：目录里共有 %d 张 png，本次只写了 %d 张，"
                "其余 %d 张是上次残留 —— 建议先清空 frames 再抽"
                % (saved + leftover, saved, leftover), "warn")

    summary = "解码 %d 帧 -> 保存 %d 张" % (total, saved)
    if dedup > 0:
        summary += "（去重跳过 %d）" % skipped
    if canceled:
        summary += "  [已取消]"

    ctx.log(summary, "ok" if not canceled else "warn")
    if shape:
        ctx.log("分辨率 %dx%d" % (shape[1], shape[0]))
    if saved == 0:
        ctx.log("一张都没抽出来 —— 检查抽帧间隔是否过大，或视频是否有有效帧", "warn")

    ctx.progress(saved, saved, summary)
    return {
        "frames": saved,
        "decoded": total,
        "skipped": skipped,
        "size": shape,
        "out_dir": str(out_dir),
        "summary": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--stride", type=int, default=10, help="每 N 帧抽一张")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefix", type=str, default="frame")
    ap.add_argument("--dedup", type=float, default=0.0,
                    help="相似帧去重阈值(0~1)：缩略图里亮度差超过 8 的像素占比小于它就跳过。"
                         "完全静止≈0；待机动画/角色移动会明显高于它，用 0.01 能保留这些帧")
    args = ap.parse_args()

    try:
        run_extract(vars(args), ConsoleContext())
    except Exception as e:
        print("[extract] 失败: %s: %s" % (type(e).__name__, e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
