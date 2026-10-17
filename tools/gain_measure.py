# -*- coding: utf-8 -*-
"""**闭环量鼠标增益**：撞角 → 走已知步数 → **问 relay 要回光标真实坐标** ⇒ gain = 像素 ÷ 单位。

和 `tools/mouse_aim_calib.py` 的区别（两条都留着 ✓ 各有各的用处）：
    · `mouse_aim_calib` 用**画面差分**量（要流、要光标在画面里可见、要画面不动）；
    · 本工具用 relay 的 `CURSOR?`（`decision/input.cursor_probe` ✓）**直接读坐标**
      ⇒ 不依赖画面、不依赖光标可见、几秒钟出数 ✓。
    ⚠ 前提：**relay 那一端支持 `CURSOR?`**（`remote_kbd/relay.py:537` ✓）；不支持 ⇒ 直接报错，
      不打哑谜（那时请用 `mouse_aim_calib` 的可视化那条 ✓）。

为什么需要它（用户 2026-10-17 ✓ 现场「**断线重连选区阶段鼠标点歪了**」✗
＋「落点**偏右下，差一整块（越过了目标）**」✗）：
    "点歪"只有两种成因，日志分不出来（送出去的指令数一样 ✓）：
      ① **增益不对**（1 单位到底走多少屏幕像素 —— 文件里写 1.0）；
      ② **起点不对**（撞角没真的到角 / 撞角那串没发完）。
    本工具把两件事**分开量**：先看撞角后是不是 `(vx, vy)`，再走固定步数看比例 ✓。

用法（在 **B 机 / 工作机**上跑；A 机的 relay 与 ProMicro 保持连接 ✓）：
    python -X utf8 -m tools.gain_measure              # 只量、只打印（不动配置 ✓）
    python -X utf8 -m tools.gain_measure --write      # 量完写回 config/mouse_gain.json
    python -X utf8 -m tools.gain_measure --step 800   # 换步长（默认 600 / 400）

⚠ 跑之前：
    1. **把工作台关掉**（relay 同一时刻只服务一个客户端 ✓ 见 `relay.py` 的 `server.accept()`）——
       开着工作台 ⇒ 这里会 **TLS 握手超时** ✓；
    2. **停挂机**（不然状态机也在动鼠标 ✓）；
    3. 光标会真的动几下（**不点左键** ✓），别在要紧的界面上跑 ✓。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get                     # noqa: E402
from decision import input as dinput            # noqa: E402
from decision import mouse_aim as ma            # noqa: E402


def _connect():
    """输入通道：已就绪就用；否则按 `link.yaml [kbd]` 连 relay ✓。"""
    if dinput.mouse_available():
        return True, "输入通道已就绪（%s）" % (dinput.link_health().get("backend"),)
    host = get("kbd", "host")
    port = int(get("kbd", "port", 9000) or 9000)
    cert = get("kbd", "cert", "remote_kbd/certs/cert.pem")
    try:
        dinput.use_network(host, port, cert)
    except Exception as e:                       # noqa: BLE001
        return False, ("连不上被控机 %s:%d（%s: %s）—— relay 在跑吗？"
                       "工作台是不是还占着那条连接（同一时刻只服务一个客户端）？"
                       % (host, port, type(e).__name__, e))
    if not dinput.mouse_available():
        return False, "连上了但鼠标后端不可用"
    return True, "已连上 %s:%d" % (host, port)


def _pos():
    """问一次光标真实坐标（relay 的 `CURSOR?` ✓）。"""
    p = dinput.cursor_probe()
    return None if not p else (int(p[0]), int(p[1]))


def _move(dx, dy, wait=0.45):
    dinput.mouse_move(int(dx), int(dy))
    time.sleep(wait)                             # 命令行可以等 ✓（实时回路里才不许睡 ✗）


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="闭环量鼠标增益（撞角 → 走步 → 读回坐标）")
    ap.add_argument("--step", default="600,400", help="x,y 各走多少单位，默认 600,400")
    ap.add_argument("--write", action="store_true", help="量完写回 config/mouse_gain.json")
    args = ap.parse_args(argv)

    ok, why = _connect()
    print("① 输入通道：%s%s" % ("✓" if ok else "✗", ("　" + why) if why else ""))
    if not ok:
        return 2

    p0 = _pos()
    print("② 读回光标（relay `CURSOR?`）：%r" % (p0,))
    if p0 is None:
        print("   ✗ relay 不支持读光标（`CURSOR?` 没回）⇒ 请改用可视化那条：")
        print("     python -X utf8 -m tools.mouse_aim_calib")
        return 2

    print("③ 撞角归零（走 -6000,-6000）…")
    _move(-6000, -6000, wait=0.8)
    p1 = _pos()
    vx, vy = ma._virtual_origin()
    print("   撞角后 = %r，虚拟屏左上角 = (%d, %d) ⇒ %s"
          % (p1, vx, vy, "✓ 起点对" if p1 == (vx, vy) else "✗ **起点不对**（差 %r）"
             % ((p1[0] - vx, p1[1] - vy) if p1 else None,)))

    try:
        sx, sy = [int(round(float(v))) for v in str(args.step).replace("，", ",").split(",")]
    except Exception:                            # noqa: BLE001
        sx, sy = 600, 400

    print("④ 横向走 %d 单位…" % sx)
    _move(sx, 0)
    p2 = _pos()
    print("④ 纵向走 %d 单位…" % sy)
    _move(0, sy)
    p3 = _pos()
    if not (p1 and p2 and p3):
        print("   ✗ 中途读不到光标，量不下去")
        return 2

    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
    gx = dx1 / float(sx)
    gy = dy2 / float(sy)
    print("\n⑤ 结果：横走 (%d, %d) 像素 / %d 单位 ⇒ **gx = %.4f px/单位**"
          % (dx1, dy1, sx, gx))
    print("        纵走 (%d, %d) 像素 / %d 单位 ⇒ **gy = %.4f px/单位**"
          % (dx2, dy2, sy, gy))
    print("        串轴泄漏：横走时 y 动了 %d、纵走时 x 动了 %d（该是 0 ✓）"
          % (dy1, dx2))

    # 线性度：再来一个大步，看"小步的尺子"能不能预测它 ✓（指针加速就在这一步露馅 ✗）
    _move(0, -sy)
    _move(-sx, 0)
    time.sleep(0.3)
    p4 = _pos()
    big = 2 * sx
    print("⑥ 线性度：再横走 %d 单位（= 2 倍）…" % big)
    _move(big, 0, wait=0.6)
    p5 = _pos()
    if p4 and p5:
        got = p5[0] - p4[0]
        want = big * gx
        err = (got - want) / want * 100.0 if want else 0.0
        print("   走了 %d 像素，按 gx 推算该 %d ⇒ 偏差 **%+.1f%%**%s"
              % (got, int(round(want)), err,
                 "　✓ 线性" if abs(err) <= 8.0 else "　✗ **非线性**（A 机指针加速还开着？）"))
        _move(-big, 0)

    old = ma.load_gain()
    print("\n⑦ 文件里的现值：%r；这次量到：(%.4f, %.4f)" % (old, gx, gy))
    if args.write:
        ma.save_gain(gx, gy, frame_shape=(768, 1366, 3))
        print("   ✅ 已写回 config/mouse_gain.json（重启工作台生效 ✓）")
    else:
        print("   （没写文件 ✓ 想写就加 --write）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
