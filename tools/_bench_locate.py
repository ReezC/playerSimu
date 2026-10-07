"""临时测量（只读）：把「定位」每一步的耗时分开量，并看看推理在不在 GPU 上。

背景（用户 2026-10-06 ✓）：GUI 帧率很低、跑几秒就卡 ⇒ `perf.log` 里
`locate_ms` **中位 104.74 ms / p99 149.93** ✗（仓库注释里原本是 **~0.5 ms** ✓）
⇒ 定位这条路慢了约两百倍，正好压在实时回路上 ✓。
本脚本把 `find_player_dot` / 差分 / 掩码 / 连通域分开计时，找那 100 ms 花在哪 ✓。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402

from core import mapdata                                # noqa: E402
from perception import minimap as mm                    # noqa: E402

MID = "110040000"
t = mapdata.load(MID, with_canvas=True)
cal = mapdata.load_calib(MID, "stream") or {}
panel = cv2.imread(str(ROOT / "data" / "dot_probe" / "app_panel.png"))
print("面板:", None if panel is None else panel.shape)
if panel is None:
    raise SystemExit("没有面板图")


def bench(tag, fn, n=5):
    fn()                                    # 预热一次（缓存/分配）
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    print("  %-34s 中位 %7.2f ms  最大 %7.2f ms" % (tag, sorted(ts)[len(ts) // 2], max(ts)))
    return sorted(ts)[len(ts) // 2]


print("=== ① 定位这条路的每一步 ===")
bench("dot_mask(yellow)", lambda: mm.dot_mask(panel, "yellow"))
bench("dot_core_mask", lambda: mm.dot_core_mask(panel))
bench("_dot_candidates(全画面掩码)", lambda: mm._dot_candidates(
    mm.dot_mask(panel, "yellow"), panel.shape[1]))
bench("_basemap_extra_mask(warp+dilate)", lambda: mm._basemap_extra_mask(
    panel, cal, t, mm.dot_mask(panel, "yellow"), "yellow"))
bench("diff_panel_vs_canvas(整幅差分)", lambda: mm.diff_panel_vs_canvas(
    panel, t.canvas, cal))
bench("dot_candidates_diff(整幅差分+连通域)", lambda: mm.dot_candidates_diff(
    panel, cal, t))
bench("find_player_dot(全画面)", lambda: mm.find_player_dot(
    panel, calib=cal, terrain=t))
bench("find_player_dot(带 roi=37x37 窗口)", lambda: mm.find_player_dot(
    panel, calib=cal, terrain=t,
    roi=mm.roi_around((993.2, 487.8), panel.shape, 40), near=(993.2, 487.8)))
bench("panel_to_world", lambda: mm.panel_to_world(1135.0, 478.0, cal, t))

print("=== ② 小面板会不会便宜很多（局部搜的动机）===")
sub = panel[460:520, 960:1030]
print("  子图", sub.shape)
bench("find_player_dot(70x60 子图)", lambda: mm.find_player_dot(sub), n=5)

print("=== ③ 推理/生态在不在 GPU 上 ===")
try:
    import torch                                        # noqa: E402
    print("  torch:", torch.__version__,
          " cuda 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  GPU:", torch.cuda.get_device_name(0),
              " 显存 %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
except Exception as e:                                  # noqa: BLE001
    print("  torch 导入失败:", type(e).__name__, e)
for name in ("onnxruntime", "ultralytics"):
    try:
        m = __import__(name)
        print("  %s: %s" % (name, getattr(m, "__version__", "?")))
        if name == "onnxruntime":
            print("     providers:", m.get_available_providers())
    except Exception as e:                              # noqa: BLE001
        print("  %s 没有: %s" % (name, type(e).__name__))
