"""【A 机运行】在屏幕上绘制时间码探针窗口。

用法（A 机）：
    python -m tools.probe_gen

窗口去边框、置顶、不抢焦点，放在屏幕顶部。
坐标需与 config/link.yaml 的 probe.x / probe.y 一致。
"""

import argparse
import sys
import tkinter as tk

from tools.config import get
from tools.probe_codec import encode_bits, now_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--y", type=int, default=None)
    ap.add_argument("--cell", type=float, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--fps", type=int, default=60, help="重绘频率")
    ap.add_argument("--out-scale", type=float, default=1.0,
                    help="流分辨率 / 屏幕分辨率。A 机 2560x1440 全屏而输出 1920x1080 时填 0.75；"
                         "探针会按 1/scale 放大绘制，缩放后正好与 B 机配置一致")
    args = ap.parse_args()

    x = args.x if args.x is not None else get("probe", "x", 100)
    y = args.y if args.y is not None else get("probe", "y", 8)
    cell = args.cell if args.cell is not None else get("probe", "cell", 16)
    gap = args.gap if args.gap is not None else get("probe", "gap", 2)
    bits = get("probe", "bits", 40)

    # ⚠ 这里有个容易忽略的前提：配置里的 x/y/cell 是**画面坐标**，
    # 而探针窗口画在**屏幕坐标**上。两者只有在"屏幕分辨率 == 输出分辨率"
    # 时才一致。
    #
    # A 机 2560x1440 全屏、采集输出 1920x1080 时，整个画面被缩了 0.75 倍：
    # 配置 x=153 的探针，到了画面里只剩 x=115、方块宽度也从 18 变成 13.5 ——
    # B 机按 153/18 去采样，采到的位置和尺寸全错，表现为
    # 「绿框歪掉 + 解码 100% 失败」。
    #
    # 所以在屏幕上按 1/scale 放大绘制，缩放之后正好等于配置值，
    # B 机一行都不用改。
    s = args.out_scale
    if s and s > 0 and abs(s - 1.0) > 1e-6:
        print(f"[probe_gen] 输出缩放 {s}：配置({x},{y}) cell={cell} "
              f"-> 屏幕需画在 ({x / s:.1f},{y / s:.1f}) cell={cell / s:.2f}")
        x, y, cell, gap = x / s, y / s, cell / s, gap / s

    # gap 在配置里是 2.25（浮点）。tkinter 的 geometry() 接受浮点字符串，
    # 但解码侧拿它算切片下标就必须是整数 —— 两边按同一套取整规则，
    # 保证画出来的方块位置和 B 机采样的位置严格对齐。
    n = 2 + bits
    w = int(round(n * (cell + gap) + gap))
    h = int(round(cell + gap * 2))

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.geometry(f"{w}x{h}+{int(x)}+{int(y)}")

    canvas = tk.Canvas(root, width=w, height=h, bg="black", highlightthickness=0)
    canvas.pack()
    rects = []
    for i in range(n):
        rx = int(round(gap + i * (cell + gap)))
        rects.append(canvas.create_rectangle(rx, int(round(gap)),
                                             rx + cell, int(round(gap)) + cell,
                                             fill="black", outline=""))

    delay = max(1, int(1000 / max(1, args.fps)))

    def tick() -> None:
        seq = encode_bits(now_ms(), bits)
        for r, v in zip(rects, seq):
            canvas.itemconfig(r, fill="white" if v else "black")
        root.after(delay, tick)

    print(f"[probe_gen] 窗口 {w}x{h} @ ({x},{y})，cell={cell} gap={gap} bits={bits}")
    print("[probe_gen] Ctrl+C 或关闭窗口退出")
    root.after(0, tick)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
