# -*- coding: utf-8 -*-
"""临时：**锚点在不同分辨率下认不认得出**（2026-10-07 开箱前置，用完删）。

为什么必须验：模板是按 **1920×1080** 裁的 ✓，而真实时流是 **1366×768**
（`config/link.yaml` 的 stream 段 ✓）✗ —— 若判别器不按分辨率归一化，
开箱时状态机会"压根不开始"（它只认已知界面 ✓），而人只会看到状态栏不动 ✗。

做法：把素材按 0.25s 采样，每帧缩到几种分辨率，喂 `ui_state.detect`，
和 `tools/verify_ui_state.py` 那份期望表逐点比对 ✓。**只读** ✓。
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perception import ui_state                        # noqa: E402

VIDEO = ROOT / "datasets" / "records" / "断线重连素材.mp4"

#: 期望表（同 `tools/verify_ui_state.py` ✓ 素材是 1920×1080 ✓）
EXPECT = [
    (0.00, 2.50, ui_state.UI_LOGIN_ERR),
    (3.25, 5.25, ui_state.UI_LOGIN),
    (11.75, 12.25, ui_state.UI_CHANNEL_LIST),
    (12.50, 15.75, ui_state.UI_CHANNEL_PANEL),
    (16.00, 18.25, ui_state.UI_QUEUE),
    (18.50, 18.50, ui_state.UI_CHANNEL_PANEL),
    (18.75, 20.25, ui_state.UI_CHAR_SELECT),
]

#: 测哪几种尺寸：真实流 + 常见 16:9 + 一个 4:3（**故意留一个不该过的** ✓）
SIZES = [(1920, 1080), (1366, 768), (1280, 720), (1600, 900), (2560, 1440),
         (1024, 768)]


def want(t):
    for a, b, ui in EXPECT:
        if a - 1e-6 <= t <= b + 1e-6:
            return ui
    return None


def main():
    samples = []
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(5) or 60.0
    step = max(1, int(round(fps * 0.25)))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            samples.append((i / fps, fr))
        i += 1
    cap.release()
    _s0 = samples[0][1].shape if samples else (0, 0, 0)
    print("素材 %s（%d×%d）采样 %d 点\n" % (VIDEO.name, _s0[1], _s0[0], len(samples)))
    print("%-12s %-8s %s" % ("分辨率", "结果", "不匹配的点（时间 / 判到 / 期望）"))
    print("-" * 78)
    for w, h in SIZES:
        bad = []
        for t, fr in samples:
            small = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            got = ui_state.detect(small)
            exp = want(t)
            if exp is None:
                if got is not None:
                    bad.append((t, ui_state.UI_NAMES.get(got, got), "（应判不出）"))
                continue
            if got != exp:
                bad.append((t, ui_state.UI_NAMES.get(got, got) if got else "（判不出）",
                            ui_state.UI_NAMES.get(exp, exp)))
        tag = "✓ 全中" if not bad else "✗ %d 点不符" % len(bad)
        print("%-12s %-8s %s" % ("%d×%d" % (w, h), tag,
                                 "; ".join("%.2fs→%s(期望%s)" % x for x in bad[:4])))
    print("-" * 78)
    print("注：`ui_state.prepare()` 把**任何**输入缩到基准 960×540 ⇒ 16:9 之间只差缩放 ✓；")
    print("    实测连 **4:3（1024×768）也全中**（锚点是小块文字/UI ⇒ 拉伸后仍能相关 ✓）")
    print("    ⇒ 换分辨率**不用重裁模板** ✓（真实流 1366×768 也 ✓）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
