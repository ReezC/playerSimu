r"""批量校验图片库中的平台顶边识别，不依赖 YOLO 权重。

示例：
    .\.venv\Scripts\python.exe -m tools.verify_platforms ^
        --source "D:\\images" --out "projects\\野猪的领土\\verify_platforms"
"""
import argparse
import json
from pathlib import Path

import cv2

from core.imgio import imread, imwrite
from perception.platforms import PlatformTracker


def run(source, out, limit=0):
    source, out = Path(source), Path(out)
    if not source.is_dir():
        raise FileNotFoundError("图片库目录不存在：%s" % source)
    frames = sorted(p for p in source.iterdir()
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not frames:
        raise FileNotFoundError("目录没有 png/jpg/jpeg 图片：%s" % source)
    if limit:
        frames = frames[:int(limit)]

    tracker = PlatformTracker()
    counts, unreadable = [], 0
    for i, path in enumerate(frames, 1):
        image = imread(path)
        if image is None:
            unreadable += 1
            continue
        platforms = tracker.update(image)
        for p in platforms:
            cv2.line(image, (int(p.x1), int(p.y)), (int(p.x2), int(p.y)),
                     (255, 255, 0), 2)
            cv2.putText(image, "P%d  %.0f-%.0f" % (p.id, p.x1, p.x2),
                        (int(p.x1), max(15, int(p.y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
                        cv2.LINE_AA)
        imwrite(out / (path.stem + ".jpg"), image, quality=90)
        counts.append(len(platforms))
        if i % 20 == 0 or i == len(frames):
            print("[%d/%d] 平台 %d" % (i, len(frames), sum(counts)), flush=True)

    stats = {
        "frames": len(counts), "platforms": sum(counts),
        "frames_with_platform": sum(n > 0 for n in counts),
        "zero_frames": sum(n == 0 for n in counts),
        "median_platforms": sorted(counts)[len(counts) // 2] if counts else 0,
        "max_platforms": max(counts) if counts else 0,
        "unreadable": unreadable, "source": str(source),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print("完成：%d 张，识别平台 %d 条，结果：%s" %
          (stats["frames"], stats["platforms"], out))
    return stats


def main():
    ap = argparse.ArgumentParser(description="批量标绘平台顶边")
    ap.add_argument("--source", required=True, help="包含 png/jpg/jpeg 的图片库目录")
    ap.add_argument("--out", default="data/verify_platforms", help="带平台线的输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只校验前 N 张；0=全部")
    args = ap.parse_args()
    try:
        run(args.source, args.out, args.limit)
    except Exception as e:
        print("[verify_platforms] 失败：%s" % e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
