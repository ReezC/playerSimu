# -*- coding: utf-8 -*-
"""临时：**离线全流程演练**（2026-10-07 开箱前置，用完删）。

把 `datasets/records/断线重连素材.mp4` 按时间喂给 `ui_state.detect` + `Reconnector`
（替身：玩家框全程"丢"✓），打印**开箱时应该看到的那串动作与状态文字**，
并核对「点服务器 / 点频道」的落点在素材里是不是**格子的正中央**。

⛔ **零输入** ✓：`execution.input.tap/mouse_move/mouse_click` 全换成记录器 ——
   只打印"本来会发什么"，**一个字节都不发** ✓。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO = ROOT / "datasets" / "records" / "断线重连素材.mp4"
OUT_PNG = ROOT / "data" / "_click_check.png"

from execution import input as dinput                       # noqa: E402
from execution import mouse_aim                             # noqa: E402
from execution import aim_convert as mouse_aim_c              # noqa: E402
from decision.agent import DecisionSettings, set_save_hook  # noqa: E402
from decision.reconnect import Reconnector                 # noqa: E402
from perception.screen import ui_state  # noqa: E402

#: 素材里量出来的格子中心（设计文档 §6.1 那张表 ✓）—— 用来判"落点在不在格子里" ✓
SRV_CENTERS = [698, 849, 999, 1149, 1300]
SRV_ROW_Y, SRV_W = 245, 142
CH_CENTERS_X = [788, 919, 1049, 1180]
CH_CENTERS_Y = [550, 593, 636, 680, 724]
CH_W, CH_H = 131, 44


def _which(px, py):
    """落点落在哪个格里（人话 ✓）。"""
    if abs(py - SRV_ROW_Y) <= 30:
        for i, cx in enumerate(SRV_CENTERS):
            if abs(px - cx) <= SRV_W / 2.0:
                return "服务器第 %d 格（中心 %d，格宽 %d）" % (i + 1, cx, SRV_W)
        return "服务器行里，但不在任何一格中心 ±%d 内 ⚠" % (SRV_W // 2)
    for r, cy in enumerate(CH_CENTERS_Y):
        if abs(py - cy) <= CH_H / 2.0 + 4:
            for c, cx in enumerate(CH_CENTERS_X):
                if abs(px - cx) <= CH_W / 2.0 + 4:
                    return "频道第 %d 行第 %d 列（中心 %d,%d；格 %d×%d）" % (
                        r + 1, c + 1, cx, cy, CH_W, CH_H)
    return "不在任何已知格子上 ⚠"


def main():
    sent = []
    orig = (dinput.tap, dinput.mouse_move, dinput.mouse_click,
            mouse_aim_c.load_gain, mouse_aim_c.aim_available)
    dinput.tap = lambda k: sent.append(("tap", k))
    dinput.mouse_move = lambda dx, dy: sent.append(("move", dx, dy))
    dinput.mouse_click = lambda *a, **k: sent.append(("click",))
    # 演练假定"通道已连 + 已标定" ✓（真机那两条待做 ✓ 见 §11）
    mouse_aim_c.load_gain = lambda: (1.0, 1.0)
    mouse_aim_c.aim_available = lambda: (True, "演练（假定已连）")
    set_save_hook(None)

    s = DecisionSettings()
    s.enabled = True
    s.reconnect_enabled = True
    s.reconnect_resume_auto = True
    s.reconnect_probe_after_lost_sec = 0.0
    s.save = lambda *a, **k: None
    r = Reconnector(s)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print("[X] 打不开素材")
        return 1
    fps = cap.get(5) or 60.0
    step = max(1, int(round(fps * 0.25)))                  # 每 0.25 秒喂一帧 ✓
    print("素材 %s（%d×%d, %.1f fps）\n" % (VIDEO.name, cap.get(3), cap.get(4), fps))
    print("%-7s %-26s %s" % ("t(s)", "判到的界面", "状态机做什么"))
    print("-" * 92)
    i = 0
    last_note = None
    clicks = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / fps
            ui = ui_state.detect(fr)
            now = 1.0 + t                                    # 单调时钟 ✓
            act = r.update(fr, False, now)                   # 玩家框全程"丢" ✓
            note = r.note
            what = ""
            if act and act.get("act") == "tap":
                dinput.tap(act["key"])
                what = "→ 按 %s（%s）" % (act["key"], act.get("desc", ""))
            elif act and act.get("act") == "click":
                px = mouse_aim_c.target_px(fr.shape[1], fr.shape[0], act["x"], act["y"])
                # ⚠ 真回路这里会 `mouse_aim.click_ratio(...)`（真发指令 ✗）——
                #   演练只算落点，不动手 ✓
                dinput.mouse_move(-4000, -4000)
                what = "→ 点画面比例 (%.4f, %.4f) ⇒ 像素 %s ⇒ %s" % (
                    act["x"], act["y"], px[0] if px and px[0] else "?", _which(*px[0]))
                clicks.append((t, px[0]))
            elif note != last_note:
                what = "· " + note
            if what or note != last_note:
                print("%-7.2f %-26s %s" % (
                    t, ui_state.UI_NAMES.get(ui, "（判不出）") if ui else "（判不出）", what))
            last_note = note
        i += 1
    cap.release()
    print("-" * 92)
    print("输入记录（演练**不真发** ✓）：%s" % (sent,))
    for t, px in clicks:
        print("  点击落点 t=%.2fs ⇒ 像素 %s ⇒ %s" % (t, px, _which(*px)))

    # ---- 落点可视化（叠在素材那两帧上，一眼看"点没点在格子正中" ✓）----
    _overlay()
    return 0


def _overlay():
    frames = []
    for t, tag in ((12.0, "选频道（服务器列表）"), (13.5, "选频道（频道面板）")):
        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            continue
        px = mouse_aim_c.target_px(fr.shape[1], fr.shape[0],
                                 *((0.3635, 0.2269) if t < 13 else (0.4104, 0.5093)))[0]
        x, y = int(round(px[0])), int(round(px[1]))
        img = fr.copy()
        cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 60, 4)
        cv2.circle(img, (x, y), 26, (0, 255, 255), 3)
        cv2.putText(img, "%s  click(%d,%d)" % (tag, x, y), (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, "%s  click(%d,%d)" % (tag, x, y), (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
        frames.append(img)
    if len(frames) == 2:
        out = np.vstack([cv2.resize(f, (960, 540)) for f in frames])
        cv2.imwrite(str(OUT_PNG), out)
        print("落点示意图 ⇒ %s" % OUT_PNG)


if __name__ == "__main__":
    sys.exit(main())
