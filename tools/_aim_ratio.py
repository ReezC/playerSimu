"""只移动不点击：把光标走到指定比例，让人眼回一句"偏多少"（一次性件，用完删）。

为什么不用 `mouse_aim.click_ratio`：它**会点** ✗ —— 校准落点阶段不能点 ✓。

用法（在工作机 B 机上跑 ✓，指令经 relay 打到 A 机 ✓）：

    python -X utf8 data/_aim_ratio.py 0.3635 0.2269        # 走到配置里"蓝蜗牛"那一格
    python -X utf8 data/_aim_ratio.py 0.3635 0.24          # 试别的纵向比例

它会：撞角归零（`MOVE -4000,-4000` ✓ 落到屏幕左上角 = 画面原点 ✓）→ **一条指令**走到目标 ✓
（与真点击完全同一条路 ✓ 只少最后那下左键 ✗）。
⚠ 跑的时候人别碰鼠标 ✓（碰了就不准 ✓）；跑完把光标留在原地，请看 A 机屏幕回报偏差 ✓。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decision import input as dinput                    # noqa: E402
from decision import mouse_aim as ma                    # noqa: E402
from tools.mouse_aim_calib import _connect_input, _grab, _open_stream   # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    xr, yr = float(sys.argv[1]), float(sys.argv[2])

    print("[1] gain = %s" % (ma.load_gain(),))
    ok, why = _connect_input()
    print("[2] %s" % why)
    if not ok:
        return 2
    src, url = _open_stream()
    frame = _grab(src)
    if frame is None:
        print("[X] 抓不到帧（A 机在推流吗）")
        return 2
    h, w = frame.shape[:2]
    from perception import ui_state as us
    ui = us.detect(frame)
    print("[3] 当前判到的界面：%s（%s）" % (ui, us.UI_NAMES.get(ui, "?")))
    print("    画面 %dx%d ｜ 目标比例 (%.4f, %.4f) ⇒ 画面 (%.0f, %.0f) 像素"
          % (w, h, xr, yr, xr * w, yr * h))

    px, why = ma.target_px(w, h, xr, yr)
    if px is None:
        print("[X] %s" % why)
        return 2
    gain = ma.load_gain()
    if not gain:
        print("[X] 没标定（config/mouse_gain.json 不在）")
        return 2
    d, why = ma.counts_for(px[0], px[1], *gain)
    if d is None:
        print("[X] %s" % why)
        return 2
    ma.corner_zero()
    dinput.mouse_move(d[0], d[1])
    print("[4] 已走：撞角归零 → (%d, %d) 指令单位 —— **没有点击** ✓" % d)
    print("    请看 A 机屏幕：光标离目标偏多少、往哪偏 ✓（回报给我，我改比例 ✓）")
    try:
        src.close()
    except Exception:                                      # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
