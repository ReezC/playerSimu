# -*- coding: utf-8 -*-
"""临时：判别"标定量到的到底是不是光标"（2026-10-07 用完删）。

标定连跑两轮数字自相矛盾（60 单位 91px、200 单位反而 64.5px，而且两次一模一样 ✗）
⇒ 它量到的**不是光标**。两种可能，必须分开：
  (a) **画面自己在动**（游戏里怪/特效/闪烁 ⇒ 帧差里不止光标那两个块 ✗）
  (b) **指针根本不在这一路画面里**（推流不画鼠标 ⇒ 帧差里**永远**没有光标 ✓）

做法：① **不动**的时候连抓两帧 ⇒ 量"底噪"；② 撞角归零后发一条**大位移** ⇒ 再抓 ⇒
量"这次动的到底是什么、动了多远"。
⛔ 只发 `MOVE`（**不点任何键** ✓）；副作用只有"游戏机光标动一下" ✓。
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.mouse_aim_calib import _connect_input, _grab, _open_stream   # noqa: E402

MOVE_UNITS = 600          # 大、且 >120（顺带验固件的分块发送 ✓）


def blobs(a, b, min_area=12):
    """两帧差里显著的连通块 ⇒ `[(面积, 质心x, 质心y), ...]`（按面积降序 ✓）。"""
    d = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    _t, m = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out.append((area, float(cents[i][0]), float(cents[i][1])))
    out.sort(reverse=True)
    return out


def desc(bs, limit=4):
    if not bs:
        return "没有显著的变化块"
    return "、".join("面积%d@(%.0f,%.0f)" % x for x in bs[:limit]) + \
           ("（另有 %d 块）" % (len(bs) - limit) if len(bs) > limit else "")


def main():
    ok, why = _connect_input()
    print("[1/3] %s" % why)
    if not ok:
        return 2
    src, url = _open_stream()
    print("[2/3] 已开流：%s" % url)
    from decision import input as dinput
    # ⚠ 量之前把「屏幕时间码探针」那条带遮掉（它每帧都在变 ✗ 会把结论带偏 ✓
    #   与标定工具同一份几何 ✓ 见 `mouse_aim_calib.probe_band_rect` ✓）
    from tools.mouse_aim_calib import _blank, probe_band_rect
    band = None
    try:
        # ---- ① 不动时的底噪（判"画面自己动没动" ✓）----
        a = _grab(src)
        time.sleep(0.5)
        b = _grab(src)
        band = probe_band_rect(a.shape[1], a.shape[0])
        if band:
            _blank(a, band)
            _blank(b, band)
            print("（探针带已遮：%s ✓ 下面看到的都是**别的**动块 ✓）" % (band,))
        base = blobs(a, b)
        print("\n① 不动时的底噪：%s" % desc(base))
        if base:
            # ⭐ 底噪**落在探针带里**是最常见的一种（2026-10-07 ✓ 现场就是它 ✓）——
            #   直接点名，省得人去猜"画面里什么东西在动" ✓（标定工具会自动遮掉它 ✓）
            from core.config import get
            _y = float(get("probe", "y", 19))
            _x = float(get("probe", "x", 99))
            _bits = int(get("probe", "bits", 40))
            _cell = float(get("probe", "cell", 16))
            _gap = float(get("probe", "gap", 2.25))
            _in = [p for p in base
                   if abs(p[2] - (_y + _cell / 2.0)) <= _cell
                   and _x - 10 <= p[1] <= _x + _bits * (_cell + _gap) + 10]
            if _in:
                print("   ⭐ 其中有块落在 `link.yaml` 的 probe 段位置上 ⇒ **就是「屏幕时间码"
                      "探针」** ✓（它每帧都在变 ✓）")
                print("      ⇒ 不影响标定：`mouse_aim_calib` 量前会**遮掉**这条带 ✓"
                      "（见 `probe_band_rect` ✓）；实测诊断要看别的动块就先把探针停掉 ✓")
            else:
                print("   ⇒ **画面自己在动** ✗（底噪就有一堆块 ⇒ 标定必定量歪："
                      "把游戏停到**静止界面** ✓）")
        else:
            print("   ⇒ 画面基本静止 ✓（没有干扰块）")

        # ---- ② 撞角归零 + 一条大位移（判"指针在不在画面里 / 动没动" ✓）----
        dinput.mouse_move(-4000, -4000)
        time.sleep(0.6)
        c = _grab(src)
        t0 = time.time()
        dinput.mouse_move(MOVE_UNITS, 0)
        time.sleep(0.6)
        d = _grab(src)
        if band:
            _blank(c, band)
            _blank(d, band)
            print("（探针带已遮 ✓ 下面看到的都是**别的**动块 ✓）")
        moved = blobs(c, d)
        print("\n② 撞角归零后发 MOVE %d 0（%.2fs）：" % (MOVE_UNITS, time.time() - t0))
        print("   变化块：%s" % desc(moved))
        if len(moved) >= 2:
            (_a1, x1, y1), (_a2, x2, y2) = moved[0], moved[1]
            print("   最大两块距离：dx=%.1f dy=%.1f（两侧块 = 光标的「旧位/新位」✓）"
                  % (x2 - x1, y2 - y1))
            print("   ⇒ 若 dx 与 %d 个指令单位的合理位移（约 %d~%d 流像素）差得远 ⇒ "
                  "**这不是光标** ✗" % (MOVE_UNITS, MOVE_UNITS // 4, MOVE_UNITS * 2))
        elif not moved:
            print("   ⇒ **一个块都没有**：光标**没在画面里**（推流不画鼠标？）或"
                  "**它压根没动**（固件 MOVE 没生效？）✗")
        else:
            print("   ⇒ 只有 1 个块（多半是光标出现/消失，或画面里别的东西在动）✗")
        print("\n③ 结论提示：")
        print("   · 底噪干净 + ②里 dx 合理 ⇒ 标定能跑（重跑 `--rounds 6` ✓）")
        print("   · 底噪不干净           ⇒ 把游戏停到**静止界面**（登录 / 选频道）再标 ✓")
        print("   · 底噪干净但②没有块    ⇒ 推流**没画鼠标指针** ⇒ 标定这条路走不通 ✗")
        print("     （要 `-draw_mouse 1`；gdigrab 默认就画 ✓，ddagrab/DXGI 那条待确认 ✓）")
    finally:
        try:
            src.close()
        except Exception:                                  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
