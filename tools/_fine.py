"""精细扫描 cell/gap，找「解出接近此刻 + 多帧一致」的最优组合（临时脚本）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from link import PyAVSource  # noqa: E402
from core.probe_codec import now_ms, decode_ms  # noqa: E402

BITS = 40

src = PyAVSource("udp://0.0.0.0:5000", container_format="mjpeg")
src.open()
frames = []
for _ in range(60):
    f = src.read()
    if f is None:
        break
    frames.append(cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY))
src.close()

now = now_ms()
print("此刻 = %d，抓到 %d 帧" % (now, len(frames)))
print()

# 候选 x/y 附近的小范围，配合 cell/gap 细扫
best = []
for cell in np.arange(18.0, 20.01, 0.1):
    for gap in np.arange(1.6, 2.6, 0.1):
        for x in (104, 105, 106, 107):
            for y in (13, 14, 15):
                tss = []
                ok_all = True
                for g in frames[:5]:
                    ts = decode_ms(g, x, y, cell, gap, BITS)
                    if ts is None:
                        ok_all = False
                        break
                    tss.append(ts)
                if not ok_all:
                    continue
                # 必须接近此刻
                d = min(abs(min(tss) - now), abs(max(tss) - now))
                if d > 600_000:
                    continue
                # 多帧一致性：标准差越小越好
                std = float(np.std(tss))
                best.append((std, cell, gap, x, y, tss))

best.sort(key=lambda t: t[0])
print("最优组合（按多帧一致性排序）：")
for std, cell, gap, x, y, tss in best[:8]:
    print("  std=%6.1f  cell=%.2f  gap=%.2f  x=%d  y=%d  ts=%s"
          % (std, cell, gap, x, y, [t // 1000 for t in tss]))

if best:
    std, cell, gap, x, y, _ = best[0]
    print()
    print("建议配置：")
    print("  x: %d" % x)
    print("  y: %d" % y)
    print("  cell: %.1f" % cell)
    print("  gap: %.1f" % gap)
