"""WGC vs BitBlt 抓帧基准测试。

用法：
    python tools/wgc_bench.py               # 用最大窗口
    python tools/wgc_bench.py --title 关键词  # 用标题含关键词的窗口

各抓 100 帧，对比耗时 / 帧率。用于定位「换 WGC 后变卡」的瓶颈。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def bench(name, fn, n=100):
    # 预热
    for _ in range(10):
        try:
            fn()
        except Exception:
            pass
    t0 = time.perf_counter()
    ok = 0
    worst = 0.0
    for _ in range(n):
        t = time.perf_counter()
        try:
            img = fn()
        except Exception:
            img = None
        dt = time.perf_counter() - t
        worst = max(worst, dt)
        if img is not None:
            ok += 1
    total = time.perf_counter() - t0
    print("%-8s %d 次: %.3fs | %d 成功 | 平均 %.1fms/帧 | %.1f fps | 最慢 %.1fms"
          % (name, n, total, ok, total / n * 1000, ok / total, worst * 1000))


def main():
    from core import wincap

    kw = None
    if "--title" in sys.argv:
        i = sys.argv.index("--title")
        if i + 1 < len(sys.argv):
            kw = sys.argv[i + 1].lower()

    ws = wincap.list_windows()
    if not ws:
        print("没有可用窗口")
        return 1
    if kw:
        hits = [w for w in ws if kw in w["title"].lower()]
        w = hits[0] if hits else None
    else:
        w = ws[0]
    if w is None:
        print("没找到标题含「%s」的窗口，前 10 个窗口：" % kw)
        for x in ws[:10]:
            print("  - %s  %s" % (x["title"][:40], x["rect"]))
        return 1
    rect = tuple(w["rect"])
    print("测试窗口: %s  rect=%s" % (w["title"][:40], rect))

    from core.wincap import grab_rect_fast

    print("\n--- BitBlt ---")
    bench("BitBlt", lambda: grab_rect_fast(rect))

    print("\n--- WGC ---")
    from core import wgc_capture
    if not wgc_capture.available():
        print("wgc-python 未安装")
        return 1
    bench("WGC", lambda: wgc_capture.grab_rect(rect))

    print("\n--- wincap.grab_rect（实际调用路径：WGC 优先） ---")
    bench("grab_rect", lambda: wincap.grab_rect(rect))
    return 0


if __name__ == "__main__":
    sys.exit(main())
