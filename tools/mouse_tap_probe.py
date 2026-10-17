# -*- coding: utf-8 -*-
"""鼠标绝对定位**实测探针**：撞角归零 → 走固定步数（默认**不点** ✓）。

为什么要有它（用户 2026-10-17 ✓ 原话：「**断线重连选区阶段鼠标点歪了**」✗
＋ 落点「**偏右下，差一整块（越过了目标）**」✗）：
    点歪只有两种成因，**日志分不出来**（指令数一样 ✓）：
      ① **增益不对** —— 1 个指令单位到底走多少屏幕像素（文件里是 `1.0`）；
      ② **起点不对** —— 撞角没真的到角（残留偏移 ✓ 或撞角那串没发完）。
    ⇒ 走一步**固定步数**再看光标实际到哪：两个数一比，①还是②当场分开 ✓。
      · 走 `(500, 500)` 落在 `(500, 500)` 附近 ⇒ 增益对、起点对 ✓（那"点歪"在比例配置 ✗）；
      · 落在 `(700, 700)` 附近 ⇒ **增益小了**（真实 ≈ 1.4 px/单位 ✓）；
      · 落在 `(800, 650)` 这种**两轴不一样多**的 ⇒ **起点有残留偏移** ✓。

跑法（在**控制机 / B 机**跑，鼠标指令会经 relay 发到 A 机）：
    python -m tools.mouse_tap_probe                  # 从角上走 (500, 500)，**不点**
    python -m tools.mouse_tap_probe --counts 1000,0   # 只测横向
    python -m tools.mouse_tap_probe --counts 500,500 --click   # 顺便点一下（有风险，慎用）
    python -m tools.mouse_tap_probe --home-only       # 只撞角，看光标是不是真到 (0,0)

⚠ 它**不在实时回路里**（命令行工具 ✓ 可以 sleep ✓）；
⚠ 跑之前**先把挂机停掉**（不然状态机会同时动鼠标 ✓ 量出来的是错的 ✗）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision import input as dinput          # noqa: E402
from decision import mouse_aim as ma          # noqa: E402


def _numbers(text, default):
    try:
        a, b = [int(round(float(v))) for v in str(text).replace("，", ",").split(",")]
        return a, b
    except Exception:                          # noqa: BLE001 —— 命令行写错就退回默认 ✓
        print("  ⚠ 看不懂 --counts %r ⇒ 用默认 %r" % (text, default))
        return default


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="鼠标绝对定位实测探针（撞角 → 走固定步数）")
    ap.add_argument("--counts", default="500,500", help="走多少指令单位，默认 500,500")
    ap.add_argument("--click", action="store_true", help="走完顺便点一下左键（默认只走不点）")
    ap.add_argument("--home-only", action="store_true", help="只撞角归零，不走路")
    args = ap.parse_args(argv)

    print("=" * 62)
    print(" 鼠标定位探针（走完**请你报**光标实际在屏幕的哪个位置）")
    print("=" * 62)

    ok, why = ma.aim_available()
    print("① 可用性：%s%s" % ("✓" if ok else "✗", ("　" + why) if why else ""))
    if not ok:
        return 2
    gain = ma.load_gain()
    print("② 当前增益（文件）：%r px/单位" % (gain,))
    try:
        vx, vy = ma._virtual_origin()
        print("③ 虚拟屏左上角（撞角落在这里）：(%d, %d)" % (vx, vy))
    except Exception as e:                     # noqa: BLE001
        print("③ 虚拟屏左上角：算不出来（%s: %s）" % (type(e).__name__, e))
    if not dinput.mouse_available():
        print("④ 鼠标通道不可用（本机 SendInput 模式没有硬件鼠标 ⇒ 这条只能走 relay）")
        return 2

    print("④ 撞角归零…（会把光标推到屏幕左上角）")
    _ok2, _why2 = ma.corner_only()
    if not _ok2:
        print("   撞角没发出去：%s" % (_why2,))
        return 2
    time.sleep(0.8)                            # 命令行可以等：让固件把那一串走完 ✓
    if args.home_only:
        print("\n✅ 已只撞角。请看一眼：光标是不是真的在屏幕**左上角**？")
        print("   不在 ⇒ 成因②（起点不对）✓ —— 那与增益无关，别动 `mouse_gain.json` ✗")
        return 0

    dx, dy = _numbers(args.counts, (500, 500))
    print("⑤ 从角上走 (%d, %d) 指令单位…" % (dx, dy))
    dinput.mouse_move(dx, dy)
    time.sleep(0.6)
    if args.click:
        dinput.mouse_click("left")
        print("   已补一下左键 ✓（慎用：可能点到界面上的东西）")

    g = gain or (1.0, 1.0)
    print("\n" + "=" * 62)
    print(" 请你报两个数：光标现在在屏幕的 (x, y)（估个大概也行）")
    print("=" * 62)
    print(" 按当前增益 %.3f / %.3f 推算，应该落在：(%d, %d)"
          % (g[0], g[1], int(round(dx * g[0])), int(round(dy * g[1]))))
    print("  · 实测就是这附近 ⇒ **增益与起点都对** ✓ ⇒ 那「点歪」是**比例配置** ✗（去重框服务器/频道）")
    print("  · 实测偏大（比如 1.4 倍）⇒ **增益小了** ✓ ⇒ 真实值 = 实测 ÷ 走的单位数")
    print("  · 两轴偏得**不一样多** ⇒ **起点有残留** ✓ ⇒ 与增益无关（先查撞角有没有走完）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
