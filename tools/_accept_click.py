"""⭐ 真机验收：用**配置里那套比例**真点一次「选频道」里的第 1 个服务器（一次性件）。

怎么判成功（不看感觉 ✓ 看界面）：
  · 点之前 `ui_state.detect` 必须是**选频道（channel_list）** ✗ 否则**一个字节都不点** ✓
    （这是本仓库的老规矩：判不出界面就不动手 ✓）；
  · 点之后界面变成**频道面板（channel_panel）** ⇒ **点中了** ✓✓；
  · 还是 `channel_list` ⇒ 点偏 / 没生效 ✗（那就把 gain 删掉回滚 ✓）。
两张帧都存下来当证据（`data/_accept_click_before/after.png` ✓）。

⚠ 只发：撞角归零 → **一条指令**走到目标 → **左键**（与生产路径完全同一条 ✓ 不碰键盘 ✓）。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import cv2                                            # noqa: E402

from decision import agent as ag                      # noqa: E402
from decision import mouse_aim as ma                  # noqa: E402
from perception import ui_state as us                 # noqa: E402
from tools.mouse_aim_calib import _connect_input, _grab, _open_stream   # noqa: E402

OUT = Path("data")


def main():
    s = ag.settings
    sx, sy = float(s.reconnect_server_x), float(s.reconnect_server_y)
    # 允许从命令行给一对比例（人眼闭环校正出来的那一对 ✓ 不影响配置文件 ✓）
    if len(sys.argv) >= 3:
        sx, sy = float(sys.argv[1]), float(sys.argv[2])
    print("[0] 配置里的「第 1 个服务器」比例：x=%.4f y=%.4f" % (sx, sy))
    print("    gain = %s（没标定就不点 ✓）" % (ma.load_gain(),))

    ok, why = _connect_input()
    print("[1] %s" % why)
    if not ok:
        return 2
    src, url = _open_stream()
    print("[2] 开流 %s" % url)

    before = _grab(src)
    if before is None:
        print("[X] 抓不到帧")
        return 2
    cv2.imwrite(str(OUT / "_accept_click_before.png"), before)
    ui = us.detect(before)
    print("[3] 点之前判到的界面：%s（%s）" % (ui, us.UI_NAMES.get(ui, "?")))
    if ui != us.UI_CHANNEL_LIST:
        print("[X] **不是选频道界面** ⇒ 一个字节都不点 ✓（判不出/在别的界面就绝不动手 ✓）")
        return 2

    print("[4] 真点一次：撞角归零 → 一条指令走到 (%.1f%%, %.1f%%) → 左键"
          % (sx * 100.0, sy * 100.0))
    ok2, why2 = ma.click_ratio(before.shape, sx, sy)
    print("    click_ratio -> ok=%s ｜ %s" % (ok2, why2))
    if not ok2:
        return 2

    time.sleep(2.5)
    after = _grab(src)
    if after is None:
        print("[X] 抓不到点之后那一帧")
        return 2
    cv2.imwrite(str(OUT / "_accept_click_after.png"), after)
    ui2 = us.detect(after)
    print("[5] 点之后判到的界面：%s（%s）" % (ui2, us.UI_NAMES.get(ui2, "?")))
    if ui2 == us.UI_CHANNEL_PANEL:
        print("[6] ✅✅ **点中了**：频道面板弹出来了 ✓ 标定这条链通了 ✓")
    elif ui2 == us.UI_CHANNEL_LIST:
        print("[6] ❌ **没点中**：还停在服务器列表（点偏 / 没生效）✗")
        print("    ⇒ 处置：删掉 config/mouse_gain.json 回滚 ✓（状态机回到\"不动手\" ✓）")
    else:
        print("[6] ⚠ 界面变成了 %s ⇒ **点到了别的东西** ✗ 立刻回滚 gain ✓ 并看图 ✓"
              % us.UI_NAMES.get(ui2, ui2))
    print("    现场帧：data/_accept_click_before.png / _accept_click_after.png ✓")
    try:
        src.close()
    except Exception:                                  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
