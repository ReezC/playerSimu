"""从素材视频裁出界面锚点模板图（perception/ui_templates/*.png）。

锚点的裁剪坐标记在 perception/ui_state.py 的 ANCHORS 表里 —— 要加新界面
（选频道 / 选角 …）先在那边登记，再跑本脚本生成图，不用手裁。

用法：
    python -m tools.make_ui_templates            # 生成缺失的模板，已有的跳过
    python -m tools.make_ui_templates --force    # 全部重新生成
    python -m tools.make_ui_templates --show     # 打印每张图裁的是哪一帧哪一块

为什么模板图要进 git：素材视频在 .gitignore 里，不把裁好的 PNG 存下来，
换台机器就重建不出来，判别器直接失效。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception import ui_state


def frame_at(video, ts):
    """读视频里第 ts 秒的那一帧（顺序读，不用 set()+retrieve —— 那样不推进帧号）。"""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(5) or 0.0
    n = int(ts * fps) if fps > 0 else 0
    fr = None
    for _ in range(n + 1):
        ok, fr = cap.read()
        if not ok:
            break
    cap.release()
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已存在的模板也重新生成")
    ap.add_argument("--show", action="store_true", help="只打印锚点清单，不写文件")
    args = ap.parse_args()

    video = ui_state.SOURCE_VIDEO
    rows = ui_state.anchor_rects()
    if args.show:
        for ui, name, rect, ts, desc in rows:
            print("%-12s %-18s %-22s %.1fs  %s" % (ui, name, rect, ts, desc))
        return 0
    if not video.exists():
        print("找不到素材视频：%s" % video)
        print("模板图已经在 perception/ui_templates/ 里，正常使用不需要重新生成。")
        return 1

    ui_state.TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    import cv2

    # 同一帧裁多个锚点时只读一次视频
    by_ts = {}
    for ui, name, rect, ts, desc in rows:
        by_ts.setdefault(ts, []).append((ui, name, rect, desc))

    made = skipped = failed = 0
    for ts in sorted(by_ts):
        todo = [r for r in by_ts[ts]
                if args.force or not ui_state.template_path(r[1]).exists()]
        if not todo:
            skipped += len(by_ts[ts])
            continue
        fr = frame_at(video, ts)
        if fr is None:
            print("读不到 %.1fs 的帧" % ts)
            failed += len(todo)
            continue
        for ui, name, (x, y, w, h), desc in todo:
            crop = fr[y:y + h, x:x + w]
            if crop.size == 0:
                print("裁剪越界：%s %s" % (name, (x, y, w, h)))
                failed += 1
                continue
            ok = cv2.imwrite(str(ui_state.template_path(name)), crop)
            print("%s %-18s %-22s <- %.1fs  %s"
                  % ("  写出" if ok else "  失败", name, (x, y, w, h), ts, desc))
            made += 1 if ok else 0
            failed += 0 if ok else 1

    print("\n生成 %d 张，跳过 %d 张（已存在），失败 %d 张" % (made, skipped, failed))
    miss = ui_state.missing_templates()
    print("模板缺失：" + ("无，判别器可用" if not miss else str(miss)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
