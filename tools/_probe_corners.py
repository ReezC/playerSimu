"""撞四角探针：光标撞到屏幕四个角时，**流画面里看不看得见**、落在哪（一次性排查件）。

为什么要问（2026-10-07）：
  · `mouse_aim` 的绝对定位靠"**撞到屏幕左上角 (0,0)** 当已知原点"（设计文档 §6.1 ✓）；
  · 而"大位移（1600 单位）量不到"时，差图里**只剩旧位一个块** ✗ ⇒ 大位移后光标在画面里
    消失了 ✗。两种可能，且**第二种要命**：
      ① 光标只是走出了**推流画面**（说明推流是**裁过的区域** ✗ ⇒ "画面左上角 = 屏幕左上角"
         这条前提**不成立** ✗ ⇒ 所有点击坐标都会偏 ✗✗）；
      ② 光标到屏幕边被 OS 夹住、但画面没裁（那它就该稳稳出现在画面边上 ✓）。
  这张探针就是把光标分别撞到四角、各存一张图，**看图定论** ✓（不再猜 ✓）。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import cv2                                            # noqa: E402

from execution import input as dinput                  # noqa: E402
from tools.mouse_aim_calib import _connect_input, _grab, _open_stream  # noqa: E402

OUT = Path("data/_probe_corner_dump")


def main():
    ok, why = _connect_input()
    print("[1] %s" % why)
    if not ok:
        return 2
    src, url = _open_stream()
    print("[2] 开流 %s" % url)
    OUT.mkdir(parents=True, exist_ok=True)

    def shot(name, note):
        f = _grab(src)
        if f is None:
            print("    %s: 抓不到帧" % name)
            return
        cv2.imwrite(str(OUT / (name + ".png")), f)
        print("    %s: 存 %s/%s.png  %s" % (name, OUT, name, note))

    # ① 左上角（撞角归零 ✓ —— 这就是绝对原点）
    dinput.mouse_move(-4000, -4000)
    time.sleep(0.4)
    shot("tl", "（= 撞角归零的原点 ⇒ 光标应当在画面左上角 ✓）")

    # ② 右上角（只沿 x 撞）
    dinput.mouse_move(4000, 0)
    time.sleep(0.4)
    shot("tr", "（光标应当在画面**右上角** ✓；看不见 ⇒ 推流是裁过的区域 ✗）")

    # ③ 左下角（先压回左边缘，再往下撞）
    dinput.mouse_move(-4000, 0)
    time.sleep(0.3)
    dinput.mouse_move(0, 4000)
    time.sleep(0.4)
    shot("bl", "（光标应当在画面**左下角** ✓）")

    # ④ 右下角
    dinput.mouse_move(4000, 0)
    time.sleep(0.4)
    shot("br", "（光标应当在画面**右下角** ✓）")

    # 收尾：把光标放回画面中间偏左（别留在角上挡 UI ✓）
    dinput.mouse_move(-4000, -4000)
    time.sleep(0.3)
    for _ in range(3):
        dinput.mouse_move(0, 120)
        time.sleep(0.08)
    print("[3] 收尾：光标已放回画面左侧中部（撞角归零 + 下移 ✓，不点任何键 ✓）")
    try:
        src.close()
    except Exception:                                  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
