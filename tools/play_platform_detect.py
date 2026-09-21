"""播放视频并实时标绘平台顶边。按 Q 或 Esc 退出。"""
import argparse
from pathlib import Path

import cv2

from perception.platforms import PlatformDetector, PlatformTracker


def main():
    ap = argparse.ArgumentParser(description="实时播放平台检测结果")
    ap.add_argument("video", help="本地 MP4/视频文件")
    ap.add_argument("--scale", type=float, default=1.0, help="显示缩放比例")
    ap.add_argument("--min-width", type=int, default=400,
                    help="最短平台像素长度；此视频默认 400，用于排除树冠")
    args = ap.parse_args()
    path = Path(args.video)
    if not path.is_file():
        print("视频不存在：%s" % path)
        return 1
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("无法打开视频：%s" % path)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 该视频的可站立地形都是较长平台；提高最小长度能排掉树叶、UI 的短横纹。
    tracker, frames = PlatformTracker(PlatformDetector(min_width=args.min_width)), 0
    delay = max(1, int(1000 / fps))
    title = "平台检测（Q / Esc 退出）"
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frames += 1
            platforms = tracker.update(image)
            for p in platforms:
                cv2.line(image, (int(p.x1), int(p.y)), (int(p.x2), int(p.y)),
                         (255, 255, 0), 2)
                cv2.putText(image, "P%d" % p.id, (int(p.x1), max(15, int(p.y) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
                            cv2.LINE_AA)
            cv2.putText(image, "platforms: %d" % len(platforms), (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
            if args.scale != 1.0:
                image = cv2.resize(image, None, fx=args.scale, fy=args.scale,
                                   interpolation=cv2.INTER_AREA)
            cv2.imshow(title, image)
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("播放完成：%d 帧" % frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
