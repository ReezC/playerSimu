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
    ap.add_argument("--cell", type=int, default=None)
    ap.add_argument("--gap", type=int, default=None)
    ap.add_argument("--fps", type=int, default=60, help="重绘频率")
    args = ap.parse_args()

    x = args.x if args.x is not None else get("probe", "x", 100)
    y = args.y if args.y is not None else get("probe", "y", 8)
    cell = args.cell if args.cell is not None else get("probe", "cell", 16)
    gap = args.gap if args.gap is not None else get("probe", "gap", 2)
    bits = get("probe", "bits", 40)

    n = 2 + bits
    w = n * (cell + gap) + gap
    h = cell + gap * 2

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.geometry(f"{w}x{h}+{x}+{y}")

    canvas = tk.Canvas(root, width=w, height=h, bg="black", highlightthickness=0)
    canvas.pack()
    rects = []
    for i in range(n):
        rx = gap + i * (cell + gap)
        rects.append(canvas.create_rectangle(rx, gap, rx + cell, gap + cell, fill="black", outline=""))

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
