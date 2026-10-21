"""按 **A 机的操作系统光标** 量鼠标 gain —— **不看流里的光标** ✓。

为什么要有它（2026-10-10 ✓ 现场定案）：
  · 老量法（`tools/mouse_aim_calib.py --source stream`）靠"**流里**光标的位移"✓
    —— 而那台机器上跑的是**仿真器** ⇒ 流里那个"光标"**不是操作系统光标** ✗
    （是仿真器自己画 / 自己处理的 ✓）⇒ 它的位移**本来就不是 1:1** ✗
    ⇒ 现场量出来"小位移 0.78 px/单位、大位移 1.0 px/单位"**互相矛盾** ✓
    ⇒ 工具只能拒绝写盘 ✓（它做得对 ✓ 但这条路在这台机器上**根本走不通** ✗）。
  · ⇒ 换尺子：`CURSOR?`（relay 直接读 **A 机操作系统光标** ✓ 见 `remote_kbd/relay.py` ✓）
    ⇒ 量到"一单位走多少**屏幕**像素" ✓ 再乘「帧 ÷ 桌面」折成**帧像素** ✓
    （`push.capture = ddagrab` ⇒ 抓的是**整块桌面** ✓ 见 relay 的 `_capture_region_size` ✓）。

用法（**先关掉工作台** ✓ 它占着流的端口 ✓）：

    python -X utf8 -m tools.mouse_gain_by_cursor                 # 量 + 合格就写盘
    python -X utf8 -m tools.mouse_gain_by_cursor --no-save       # 只量不写
    python -X utf8 -m tools.mouse_gain_by_cursor --frame 1366x768  # 流抓不到帧时手工给尺寸

⚠ 它会**动 A 机的鼠标**（撞角 + 走几步 + 走回来 ✓ **不点** ✓）。
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 量哪几档（单位 ✓）：都在**一屏之内**（不会撞到边界 ✓）、又能看出线性 ✓。
STEPS = (60, 120, 300, 800)
#: 各档"一单位走多少屏幕像素"最大允许差多少（超过 ⇒ 尺子本身不可信 ⇒ **不写** ✗）。
TOL = 0.15


def _probe():
    """读一次 A 机光标 + 桌面 ⇒ `(x, y, vx, vy, vw, vh, cap)`；问不到 ⇒ `None` ✓。"""
    from execution import input as dinput
    from execution import remote as _remote_mod
    return _remote_mod.cursor_probe()


def measure():
    """⇒ `(每档的表, gain_screen, 人话)`；量不了 ⇒ 抛 `RuntimeError(人话)` ✓。"""
    from execution import aim_convert as ma_c
    from execution import mouse_aim as ma
    p = _probe()
    if p is None:
        raise RuntimeError(
            "读不到 A 机光标 ✗ ⇒ A 机 relay 得是**新版**（带 `CURSOR?` ✓）且要**重启过它** ✓；"
            "也可能是 relay 连不上（先看能不能 telnet 到 9000 ✓）")
    vx, vy, vw, vh, cap = p[2], p[3], p[4], p[5], p[6]
    rows = []
    ma_c.corner_zero()                     # 撞到 A 机桌面左上角（已知原点 ✓）
    time.sleep(0.25)
    for u in STEPS:
        a = _probe()
        if a is None:
            raise RuntimeError("走位途中读不到光标了 ⇒ A 机 relay / 链路不稳 ✗")
        ma.dinput.mouse_move(u, u)
        time.sleep(0.25)
        b = _probe()
        ma.dinput.mouse_move(-u, -u)     # ⚠ 立刻挪回来（回到撞角处 ✓）
        time.sleep(0.20)
        if b is None:
            raise RuntimeError("走完那一步读不到光标 ⇒ A 机 relay / 链路不稳 ✗")
        dx, dy = int(b[0]) - int(a[0]), int(b[1]) - int(a[1])
        rows.append((u, dx, dy, (dx / float(u)), (dy / float(u))))
    ok_rows = [r for r in rows if r[3] > 0 and r[4] > 0]
    if len(ok_rows) < len(rows):
        raise RuntimeError("有档位量到「光标没动」（Δ=%r）⇒ 量不成 ✓"
                           % [(r[0], r[1], r[2]) for r in rows])
    gx = sum(r[3] for r in ok_rows) / float(len(ok_rows))
    gy = sum(r[4] for r in ok_rows) / float(len(ok_rows))
    worst = max(abs(r[3] - gx) / gx for r in ok_rows)
    return rows, (gx, gy, vx, vy, vw, vh, cap), worst


def main():
    ap = argparse.ArgumentParser(description="按 A 机**操作系统光标**量 gain（比看流里光标可靠 ✓）")
    ap.add_argument("--no-save", action="store_true", help="只量、不写 config/mouse_gain.json")
    ap.add_argument("--frame", default=None, help="手工给帧尺寸，如 1366x768（流抓不到时用 ✓）")
    a = ap.parse_args()

    from execution import aim_convert as ma_c

    from execution import mouse_aim as ma
    from tools.mouse_aim_calib import _connect_input, _grab, _open_stream

    print("[1/4] 连输入通道…")
    ok, why = _connect_input()
    print("      %s" % why)
    if not ok:
        return 2

    print("[2/4] 抓一帧看帧尺寸（关掉工作台才抓得到 ✓；也可用 --frame 跳过）…")
    frame_wh = None
    if a.frame:
        try:
            w, h = (int(v) for v in str(a.frame).lower().split("x"))
            frame_wh = (h, w)
        except Exception:                        # noqa: BLE001
            print("      --frame 写错了（该像 1366x768 ✓）")
            return 2
    else:
        try:
            s, _url = _open_stream()
            try:
                f = _grab(s, timeout=5.0)
            finally:
                try:
                    s.close()
                except Exception:                # noqa: BLE001
                    pass
            if f is not None:
                img = getattr(f, "image", f)
                frame_wh = (int(img.shape[0]), int(img.shape[1]))
        except Exception as e:                   # noqa: BLE001
            print("      抓帧没成：%s: %s" % (type(e).__name__, e))
    if frame_wh is None:
        print("      ⚠ 拿不到帧尺寸 ⇒ 先关掉工作台再跑 ✓ 或加 `--frame 1366x768` ✓")
        return 2
    print("      帧 = %dx%d" % (frame_wh[1], frame_wh[0]))

    print("[3/4] 撞角 + 分档走一步（读 **A 机操作系统光标** ✓）…")
    rows, geo, worst = measure()
    gx_s, gy_s, vx, vy, vw, vh, cap = geo
    print("      A 机桌面：虚拟屏 (%d,%d) %dx%d" % (vx, vy, vw, vh))
    for u, dx, dy, px, py in rows:
        print("      %5d 单位 ⇒ 走了 (%5d,%5d) 屏幕像素 ⇒ %.4f / %.4f px/单位"
              % (u, dx, dy, px, py))
    print("      各档最大差 %.1f%%（≤ %.0f%% 才算数 ✓）" % (worst * 100.0, TOL * 100.0))
    if worst > TOL:
        print("[X] ⚠ 各档**对不上** ⇒ 尺子本身不可信 ✗ ⇒ **不写盘** ✓")
        print("    （真加速要关；若设置本来就没开 ⇒ 那是**仿真器自己**在改鼠标行为 ✗")
        print("     ⇒ 绝对定位在这台机器上先天不准，只能靠「贴近目标」的保守点法 ✓）")
        return 2

    # 帧 ÷ 桌面：ddagrab 抓整块桌面 ⇒ 缩放 = 帧宽 ÷ 桌面宽 ✓（拿不到抓取区域时按虚拟屏 ✓）
    scr_w, scr_h = (cap[0], cap[1]) if cap else (vw, vh)
    sx = frame_wh[1] / float(scr_w)
    sy = frame_wh[0] / float(scr_h)
    gx, gy = gx_s * sx, gy_s * sy
    print("[4/4] 折算：帧 %dx%d ÷ 桌面 %dx%d ⇒ 缩放 %.4f" % (frame_wh[1], frame_wh[0],
                                                           scr_w, scr_h, sx))
    print("      **帧 gain = %.4f / %.4f px/单位**（屏幕基准 %.4f/%.4f ✓）"
          % (gx, gy, gx_s, gy_s))
    if a.no_save:
        print("（--no-save ⇒ 不写盘 ✓）")
        return 0
    if not (0.01 <= gx <= 50.0 and 0.01 <= gy <= 50.0):
        print("[X] 算出来的数不像话 ⇒ **不写盘** ✓")
        return 2
    ma_c.save_gain(gx, gy, frame_shape=frame_wh,
                 extra={"probe": {"source": "relay-cursorpos", "by": "cursor",
                                  "screen_px_per_unit": [round(gx_s, 6), round(gy_s, 6)],
                                  "scale": [round(sx, 6), round(sy, 6)],
                                  "worst_rel": round(worst, 4)}})
    print("已写 %s 的 **%dx%d** 那一份 ✓（别的尺寸那份不动 ✓）"
          % (ma_c.GAIN_PATH, frame_wh[1], frame_wh[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
