"""频道面板那一格的验收（一次性件，验收完删）。

⚠ 为什么必须"连着做"：**频道面板会自己关掉** ✗（2026-10-07 实测：真点开面板几分钟后，
界面又回到 `channel_list` 服务器列表 ✓）⇒ 单跑"走到频道格"没有意义 ✗。

流程（每一步都判界面，不靠感觉 ✓）：
    ① 现在必须是 `channel_list`（服务器列表）✗ 否则不动手；
    ② **点服务器**（`--server-x/--server-y` ✓ 默认 = 真机校正过的 0.3635 / 0.2074 ✓）；
    ③ 判是不是 `channel_panel` ✓ 不是就停（不硬点 ✗）；
    ④ 走到频道格（`--channel-x/--channel-y` ✓ 默认 0.4104 / 0.5093 ✓）——
       **不带 `--click` 时只移动、不点** ✓（给人眼确认落点用 ✓）；
       带 `--click` ⇒ **左键点下去** ✓，然后判界面：`queue`（排队）/ `char_select`（选角）
       都算**点中了** ✓（进频道/排队都说明这一下生效了 ✓）。

用法：
    python -X utf8 data/_accept_click2.py                    # 只移动（人眼校正落点 ✓）
    python -X utf8 data/_accept_click2.py --click            # 真点（验收 ✓）
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import cv2                                            # noqa: E402

from decision import agent as ag                      # noqa: E402
from execution import input as dinput                  # noqa: E402
from execution import mouse_aim as ma                  # noqa: E402
from perception.screen import ui_state as us                 # noqa: E402
from tools.mouse_aim_calib import _connect_input, _grab, _open_stream   # noqa: E402

OUT = Path("data")


def aim(shape, xr, yr, click, double=False):
    from execution import aim_convert as ma_c   # 2026-10-18：拆包后补的 import ✓
    """撞角归零 → 一条指令走到目标（`click=True` 才左键 ✓），返回 `(ok, why, counts)`。"""
    h, w = shape[:2]
    px, why = ma_c.target_px(w, h, xr, yr)
    if px is None:
        return False, why, None
    gain = ma_c.load_gain()
    if not gain:
        return False, "没标定", None
    d, why = ma_c.counts_for(px[0], px[1], *gain)
    if d is None:
        return False, why, None
    if not dinput.mouse_available():
        return False, "鼠标通道不可用", None
    ma_c.corner_zero()
    dinput.mouse_move(d[0], d[1])
    if click:
        dinput.mouse_click("left")
        if double:
            time.sleep(0.06)                        # 双击的间隔（< 系统双击阈值 ✓）
            dinput.mouse_click("left")
    return True, "走 (%d, %d) → 画面 (%.0f, %.0f)%s" % (d[0], d[1], px[0], px[1],
                                                        " + 左键" if click else "（未点）"), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--click", action="store_true", help="真的点下去（默认只移动 ✓）")
    ap.add_argument("--double", action="store_true", help="双击（有的界面要双击才进去 ✓）")
    ap.add_argument("--from-panel", action="store_true",
                    help="允许「已经停在频道面板」时跳过点服务器那步（面板没关时用 ✓）")
    ap.add_argument("--server-x", type=float, default=None)
    ap.add_argument("--server-y", type=float, default=None)
    ap.add_argument("--channel-x", type=float, default=None)
    ap.add_argument("--channel-y", type=float, default=None)
    a = ap.parse_args()
    s = ag.settings
    sx = a.server_x if a.server_x is not None else float(s.reconnect_server_x)
    sy = a.server_y if a.server_y is not None else float(s.reconnect_server_y)
    cx = a.channel_x if a.channel_x is not None else float(s.reconnect_channel_x)
    cy = a.channel_y if a.channel_y is not None else float(s.reconnect_channel_y)
    print("[0] 服务器格 (%.4f, %.4f) ｜ 频道格 (%.4f, %.4f) ｜ click=%s"
          % (sx, sy, cx, cy, a.click))

    ok, why = _connect_input()
    print("[1] %s" % why)
    if not ok:
        return 2
    src, url = _open_stream()
    print("[2] 开流 %s" % url)

    frame = _grab(src)
    if frame is None:
        print("[X] 抓不到帧")
        return 2
    cv2.imwrite(str(OUT / "_accept2_before.png"), frame)
    ui = us.detect(frame)
    print("[3] 起始界面：%s（%s）" % (ui, us.UI_NAMES.get(ui, "?")))
    if ui == us.UI_CHANNEL_PANEL and a.from_panel:
        # ⭐ 面板还开着 ⇒ 不用再点服务器（点它会白点一下 ✓）—— 直接量这一格 ✓
        frame2 = frame
        print("[4] 已经停在频道面板 ⇒ 跳过「点服务器」那步 ✓（--from-panel ✓）")
    elif ui == us.UI_CHANNEL_LIST:
        print("[4] 点服务器（把频道面板叫出来 ✓）")
        ok2, why2, _ = aim(frame.shape, sx, sy, True)
        print("    %s ｜ %s" % (ok2, why2))
        if not ok2:
            return 2
        time.sleep(2.0)
        frame2 = _grab(src)
        cv2.imwrite(str(OUT / "_accept2_panel.png"), frame2)
        ui2 = us.detect(frame2)
        print("[5] 点完判到：%s（%s）" % (ui2, us.UI_NAMES.get(ui2, "?")))
        if ui2 != us.UI_CHANNEL_PANEL:
            print("[X] 面板没出来 ⇒ 停（不硬点频道格 ✗）")
            return 2
    else:
        print("[X] 不在服务器列表、也不在频道面板 ⇒ 不动手 ✓"
              "（先退回列表，或用 --from-panel 从面板接着点 ✓）")
        return 2

    print("[6] 走到频道格：%s" % ("**会点下去** ✓" if a.click else "只移动、不点 ✓"))
    ok3, why3, _ = aim(frame2.shape, cx, cy, a.click, double=a.double)
    print("    %s ｜ %s" % (ok3, why3))
    if not ok3:
        return 2
    if not a.click:
        print("[7] 请看 A 机屏幕：光标在不在**频道面板第 1 格**里、偏多少 ✓")
        print("    （面板可能几秒后自己关 ✓ 那就重跑一次 ✓）")
        return 0

    time.sleep(2.5)
    frame3 = _grab(src)
    cv2.imwrite(str(OUT / "_accept2_after.png"), frame3)
    ui3 = us.detect(frame3)
    print("[7] 点完判到：%s（%s）" % (ui3, us.UI_NAMES.get(ui3, "?")))
    if ui3 in (us.UI_QUEUE, us.UI_CHAR_SELECT):
        print("[8] ✅✅ **频道这一格也点中了**（界面进到 %s ✓）—— 两下点击全通 ✓"
              % us.UI_NAMES.get(ui3, ui3))
    elif ui3 == us.UI_CHANNEL_PANEL:
        print("[8] ❌ 面板还在 ⇒ 没点中 ✗（再看落点、或面板已超时关闭 ⇒ 重跑 ✓）")
    else:
        print("[8] ⚠ 界面变成 %s ⇒ 点到了别处 ✗ 看图确认 ✓"
              % us.UI_NAMES.get(ui3, ui3))
    print("    现场帧：data/_accept2_before/panel/after.png ✓")
    try:
        src.close()
    except Exception:                                  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
