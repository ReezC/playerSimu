"""离线校验界面判别：把整段重连素材喂给 perception.ui_state.detect()，逐段打印判到的界面。

**为什么要有这个脚本**：锚点（`ui_state.ANCHORS` 里的矩形）是照着录像量出来的，
一旦改了矩形、换了模板图、或者游戏 UI 版本更新，都可能悄悄失准 —— 失准的后果是
「判错界面 → 走错步骤 → 乱按键」。跑一遍这个脚本，几秒钟就能看出来还准不准。

用法：
    python -m tools.verify_ui_state            # 每 0.5 秒判一次（快）
    python -m tools.verify_ui_state --step 0.25  # 更密
    python -m tools.verify_ui_state --scores     # 顺带打印每个锚点的分数

需要素材视频（`.gitignore` 里，不在仓库里）。没有视频时它会说明并退出，
不影响正常使用 —— 模板图本身已经存在 `perception/ui_templates/`。

**判定标准**：每个界面在自己那段时间里判得出来、在别人的时间里判不出来（不误报）。
素材里各段的时间轴见 docs/断线重连设计.md。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception import ui_state

# 素材里的真实时间轴：(起, 止, 期望界面)。改锚点后如果这里开始报不符，就是退化了。
EXPECT = [
    (0.00, 2.50, ui_state.UI_LOGIN_ERR),
    (3.25, 5.25, ui_state.UI_LOGIN),
    (11.75, 12.25, ui_state.UI_CHANNEL_LIST),
    (12.50, 15.75, ui_state.UI_CHANNEL_PANEL),
    (16.00, 18.25, ui_state.UI_QUEUE),
    (18.50, 18.50, ui_state.UI_CHANNEL_PANEL),   # 排队结束、面板还在
    (18.75, 20.25, ui_state.UI_CHAR_SELECT),     # 20.0 起是淡出过程，锚点仍能匹配
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.5, help="采样间隔（秒）")
    ap.add_argument("--scores", action="store_true", help="打印每个锚点的分数")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="只跑到该秒数（0 = 整段）")
    args = ap.parse_args()

    video = ui_state.SOURCE_VIDEO
    if not video.exists():
        print("找不到素材视频：%s" % video)
        print("（模板图已经在 perception/ui_templates/，正常使用不需要这个脚本）")
        return 1
    miss = ui_state.missing_templates()
    if miss:
        print("模板缺失：%s\n先跑 python -m tools.make_ui_templates" % miss)
        return 1

    import cv2

    print("素材：%s" % video.name)
    print("阈值 %.2f  匹配缩放 %.2f  搜索余量 %dpx  采样 %.2fs\n"
          % (ui_state.THRESHOLD, ui_state.MATCH_SCALE, ui_state.SEARCH_MARGIN, args.step))

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(5) or 60.0
    dur = args.duration or (cap.get(7) / fps if cap.get(7) else 0.0)
    step = max(0.05, args.step)
    want = [round(i * step, 3) for i in range(0, int(dur / step) + 1)]

    res = []
    cost = []
    i = k = 0
    prev = object()
    while k < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if abs(i / fps - want[k]) < 1.0 / fps / 2:
            t0 = time.perf_counter()
            ui = ui_state.detect(fr)
            cost.append((time.perf_counter() - t0) * 1000.0)
            res.append((want[k], ui))
            if ui != prev:
                print("  %5.2fs   %s" % (want[k],
                                         ui_state.UI_NAMES.get(ui, "（判不出）") if ui
                                         else "（判不出）"))
                prev = ui
            if args.scores:
                per, hit = ui_state.scores(fr)
                print("           " + "  ".join("%s=%.2f" % (n, s)
                                                for n, s in sorted(per.items())))
            k += 1
        i += 1
    cap.release()

    bad = []
    for t, ui in res:
        exp = None
        for a, b, name in EXPECT:
            if a - 1e-6 <= t <= b + 1e-6:
                exp = name
                break
        if ui != exp:
            bad.append((t, ui, exp))

    print()
    if cost:
        print("detect 耗时：平均 %.1f ms，最大 %.1f ms（%d 次）"
              % (sum(cost) / len(cost), max(cost), len(cost)))
    if bad:
        print("与期望不符 %d 处：" % len(bad))
        for t, ui, exp in bad:
            print("   %5.2fs  判到 %-14s 期望 %s"
                  % (t, ui_state.UI_NAMES.get(ui, "None") if ui else "None",
                     ui_state.UI_NAMES.get(exp, "None") if exp else "None"))
        return 1
    print("全部 %d 个采样点与期望一致" % len(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
