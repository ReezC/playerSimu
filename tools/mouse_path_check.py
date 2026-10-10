"""一条命令分清「鼠标为什么没被操控」（**默认只读、不动鼠标** ✓）。

用法（在**跑工作台这台机器（B 机）**上跑 ✓）：

    python -X utf8 -m tools.mouse_path_check              # 只查，不动鼠标 ✓
    python -X utf8 -m tools.mouse_path_check --move       # 再真走一小步验证通路 ✓

它按顺序回答三件事（**每一句都能直接照做** ✓）：

  ① **这一路画面的帧尺寸是多少 · 这个尺寸量没量过鼠标标定？**
     ⇒ 没量过的话，新代码是「**故意不点**」✓（宁可不点，也别点歪 ✓）
     ⇒ 这**就是**"鼠标没被操控"最常见的那一个原因 ✓ ⇒ 量一次即可 ✓。

  ② **看不看得见光标**（本地后端读本机 / 网络 relay 问 A 机 ✓）＋ 自动量开关状态 ✓。

  ③ （`--move` 才有）真走一小步 ⇒ **光标动没动** ✓
     ⇒ 动了 = 通路好的 ✓（那问题就在 ① ✓）；没动 = 指令到不了 A ✗（去重启 relay / 拔插 ✓）。

⚠ 为什么要有它（用户 2026-10-10 ✓ 同一句"**A 机的鼠标没有被操控**"报了几次 ✓）：
  那句话背后至少有**三种完全不同**的原因（故意不点 / 通路断了 / 忙标记卡住 ✓），
  而它们**修法完全不同** ✗ ⇒ 只靠人描述分不出来 ✓ ⇒ 做成一条命令，**让机器自己说** ✓。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _frame_shape(src):
    """这一路画面现在多少像素 ⇒ `(h, w)`；抓不到 ⇒ `None` ✓（**不下载、只抓一帧** ✓）。"""
    from tools import mouse_aim_calib as mc
    try:
        if src == "stream":
            s, _url = mc._open_stream()
            try:
                f = mc._grab(s, timeout=5.0)
            finally:
                try:
                    s.close()
                except Exception:                # noqa: BLE001
                    pass
        else:
            # 本地窗口那条：用**项目里框选的那块**（口径同实时页 ✓）
            from core import wincap
            from gui import project as proj          # ⚠ `last_opened` 在 `gui/project.py` ✓
            _p = proj.last_opened() or {}
            rect = wincap.resolve_rect(_p.get("rect") or _p.get("rect_rel"))
            live = mc._LiveSrc(rect=rect)         # ⚠ 那个类叫 `_LiveSrc` ✓（见 `mouse_aim_calib` ✓）
            live.open()
            try:
                f = mc._grab(live, timeout=3.0)
            finally:
                try:
                    live.close()
                except Exception:                # noqa: BLE001
                    pass
        if f is None:
            return None
        img = getattr(f, "image", f)
        return (int(img.shape[0]), int(img.shape[1]))
    except Exception as e:                       # noqa: BLE001 —— 抓不到就说抓不到 ✓
        print("  （抓帧没成：%s: %s）" % (type(e).__name__, e))
        return None


def _try_move():
    """真走一小步、看光标动没动（**一定挪回来** ✓：同一组单位反过来再发一次 ✓）。

    ⚠ 走 240 单位（≈ 一两百像素 ✓）—— 够看出动没动 ✓、又不至于跑出屏幕 ✓。
    ⚠ 用**相对**位移对消（不依赖增益 ✓）⇒ 不需要先知道 gain ✓。
    """
    from decision import mouse_aim as ma
    p0 = ma._probe_cursor()
    if p0 is None:
        print("  ❌ **连光标都读不到** ⇒ 通路/relay 那侧的事 ✓（重启 A 机 relay + 拔插 ProMicro ✓）")
        return
    ma.corner_zero()
    p1 = ma._probe_cursor()
    ma.dinput.mouse_move(240, 240)
    p2 = ma._probe_cursor()
    ma.dinput.mouse_move(-240, -240)             # ⚠ 挪回来 ✓
    p3 = ma._probe_cursor()
    print("  撞角后 =", p1, " 走(+240,+240) 后 =", p2, " 挪回后 =", p3)
    if p1 and p2 and (p2[0] != p1[0] or p2[1] != p1[1]):
        print("  ✅ **光标动了** ⇒ 鼠标通路是好的 ✓")
        if p3 and p1 and (abs(p3[0] - p1[0]) > 3 or abs(p3[1] - p1[1]) > 3):
            print("     （挪回来差 %d,%d px ⇒ 指针有加速/非线性，属正常 ✓）"
                  % (p3[0] - p1[0], p3[1] - p1[1]))
    else:
        print("  ❌ **光标没动** ⇒ 指令到不了 A 机 ✗ ⇒ 重启 A 机 relay + 拔插一次 ProMicro ✓")


def main():
    ap = argparse.ArgumentParser(description="鼠标为什么没被操控（默认只读 ✓）")
    ap.add_argument("--move", action="store_true",
                    help="真走一小步验证通路（**会动鼠标** ✓ 走完挪回来 ✓）")
    ap.add_argument("--source", choices=("stream", "live"), default=None,
                    help="按哪一路画面查（默认读 config/live.yaml 的 mmap_src ✓）")
    a = ap.parse_args()

    from core.config import load_live
    from decision import input as dinput
    from decision import mouse_aim as ma
    from perception import minimap as mm

    live = load_live()
    src = a.source or mm.live_src(live)
    print("=" * 68)
    print("① 画面与标定")
    print("  capture_source = %s   小地图来源 = %s" % (live.get("capture_source"), src))
    shape = _frame_shape(src)
    if shape is None:
        print("  帧尺寸 = **抓不到** ✗ ⇒ 先确认这一路画面是活的"
              "（收流：A 机在推 ✓；本地窗口：游戏窗口开着 ✓）")
        print("     ⚠ 若**工作台正开着** ⇒ 流的 5000 端口被它占着（上面那句 `10048` 就是）"
              "⇒ 先关掉工作台再跑本工具 ✓（或改跑 `--source live` ✓）")
    else:
        h, w = shape
        print("  帧尺寸 = %dx%d" % (w, h))
        g = ma.load_gain((h, w, 3))
        print("  这个尺寸的标定 = %s" % (("%.4f / %.4f" % g) if g else "**没有** ✗"))
        print("  已经量过的尺寸 = %s" % (ma.gain_sizes() or "（一个都没有）"))
        if not g:
            print("  ⇒ ★ 新代码在这里是「**故意不点**」✓（宁可不点，也别点歪 ✓）")
            print("     这**就是**「鼠标没被操控」最常见的原因 ✓ ⇒ 量一次就好：")
            print("     python -X utf8 -m tools.mouse_aim_calib --source %s" % src)
    print()
    print("② 看得见光标吗")
    print("  input.cursor_probe() = %s" % (dinput.cursor_probe(),))
    print("  自动量开关 = %s（默认关 ✓ 要开在 config/mouse_gain.json 写 \"auto_measure\": true ✓）"
          % ma.auto_measure_enabled())
    print("  忙标记 _auto_busy() = %s（False = 没有东西挡着鼠标 ✓）" % ma._auto_busy())
    if a.move:
        print()
        print("③ 真走一小步")
        _try_move()
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
