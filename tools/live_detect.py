"""实时流检测：读流 → YOLO 推理 → 画框，并统计真正的端到端延迟。

这是整条链路的最终形态，也是回答"346ms 够不够"这个问题的唯一方法：

    A 机产生画面 ──采集/编码──> 网络 ──> B 机收到帧 ──推理──> 得到目标位置

    端到端 = 探针测出的媒体延迟 + 单帧推理耗时
             ↑ "B 机看到的是多久以前的画面"
                                  ↑ "从看到到算出结果还要多久"

只有这两个加起来才是 bot 真正面对的"反应时间"。

用法：
    python -m tools.live_detect --show
    python -m tools.live_detect --show --conf 0.3 --imgsz 960
    python -m tools.live_detect --seconds 30            # 不显示，只统计
"""

import argparse
import pathlib
import statistics
import sys
import time

import cv2
import numpy as np

from link import PyAVSource
from tools.config import get
from tools.probe_codec import decode_ms, resolve_delay_ms


def find_model(explicit=None):
    """定位要用的权重。显式指定优先，否则取项目 models/ 下最新的一个。"""
    if explicit:
        return explicit
    for d in sorted(pathlib.Path("projects").glob("*")):
        ms = sorted((d / "models").glob("*.pt"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
        if ms:
            return str(ms[0])
        # models/ 空时退回 runs/**/weights/best.pt（和 GUI 的查找逻辑一致）
        runs = sorted((d / "runs").glob("**/weights/best.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            return str(runs[0])
    return None


def load_offset_ms():
    try:
        from tools.clock_sync import sync_offset
        off = sync_offset(save=True)
        if off is not None:
            return off
    except Exception:
        pass
    p = pathlib.Path("config") / "clock_offset.txt"
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return None


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="权重路径，默认用项目 models/ 里最新的")
    ap.add_argument("--url", default=None)
    ap.add_argument("--format", default=None,
                    help="强制输入格式（如 mjpeg）。默认读 config stream.format")
    ap.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seconds", type=float, default=0.0, help="运行时长，0=不限")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--report", type=float, default=3.0, help="统计打印间隔(秒)")
    ap.add_argument("--min-boxes", type=int, default=1,
                    help="框数超过这个值才算'有目标'，用于统计命中率")
    args = ap.parse_args()

    mpath = find_model(args.model)
    if not mpath:
        print("找不到权重文件，请先用 ⑦ 训练或用 --model 指定")
        return 1

    # 探针：用来算媒体延迟
    use_probe = bool(get("probe", "enabled", True))
    px, py = get("probe", "x", 100), get("probe", "y", 8)
    cell, gap = get("probe", "cell", 16), get("probe", "gap", 2)
    bits = get("probe", "bits", 40)
    offset_ms = load_offset_ms()
    if use_probe and offset_ms is None:
        print("[live_detect] 没有 clock_offset.txt，媒体延迟不可用"
              "（先跑 tools.clock_sync --save）")
        use_probe = False

    # 延迟加载：ultralytics 导入要好几秒，且会拖慢 --help
    print(f"[live_detect] 加载模型 {mpath}")
    from ultralytics import YOLO

    model = YOLO(mpath)

    src = PyAVSource(args.url or get("stream", "url"),
                     container_format=args.format or get("stream", "format", None))
    src.open()
    w, h = src.size or (0, 0)

    if w and abs(w - 1366) > 1:
        print(f"[live_detect] ⚠ 流宽 {w} != 预期 1366，探针参数可能不匹配")

    print(f"[live_detect] 流 {w}x{h} fps={src.fps}  conf={args.conf} imgsz={args.imgsz}")
    print(f"[live_detect] 探针 region=({px:.1f},{py:.1f}) cell={cell:.2f} "
          f"offset={offset_ms:.1f}ms")

    inf_ms, med_ms, tot_ms, nbox = [], [], [], []
    miss = 0
    n = 0
    t_start = time.perf_counter()
    t_report = t_start
    last_good = None      # 上一次成功的检测结果，显示用

    try:
        while True:
            f = src.read()
            if f is None:
                break
            n += 1

            # ---- 媒体延迟（画面有多旧）----
            d = None
            if use_probe:
                gray = cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY)
                ts_a = decode_ms(gray, px, py, cell, gap, bits)
                if ts_a is None:
                    miss += 1
                else:
                    d = resolve_delay_ms(
                        (f.t_recv_wall + offset_ms / 1000.0) * 1000.0, ts_a)
                    if d is not None and 0 < d < 5000:
                        med_ms.append(d)
                    else:
                        d = None

            # ---- 推理 ----
            t0 = time.perf_counter()
            res = model.predict(f.image, conf=args.conf, imgsz=args.imgsz,
                                device=args.device, verbose=False)
            t1 = time.perf_counter()
            dt = (t1 - t0) * 1000.0
            inf_ms.append(dt)

            boxes = res[0].boxes
            nbox.append(len(boxes))

            # ---- 端到端 = 画面有多旧 + 算出结果花了多久 ----
            if d is not None:
                tot_ms.append(d + dt)

            if args.show:
                vis = cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR)
                for b in boxes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    cf = float(b.conf[0])
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.putText(vis, "%.2f" % cf, (int(x1), int(y1) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 把两个延迟分开显示 —— 它们的优化手段完全不同：
                #   媒体延迟大 -> 推流链路问题
                #   推理耗时大 -> 模型/显卡问题
                if med_ms:
                    cv2.putText(vis, "media %.0fms" % med_ms[-1], (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis, "infer %.0fms" % dt, (10, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if tot_ms:
                    cv2.putText(vis, "TOTAL %.0fms" % tot_ms[-1], (10, 88),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(vis, "boxes %d" % len(boxes), (10, 116),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                cv2.imshow("live_detect", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.perf_counter()
            if now - t_report >= args.report:
                t_report = now
                el = now - t_start
                line = f"[{el:6.1f}s] fps={n / el:5.1f} boxes p50={pct(nbox,.5):4.0f}"
                if inf_ms:
                    line += (f" | infer p50={pct(inf_ms,.5):5.1f} "
                             f"p95={pct(inf_ms,.95):6.1f}ms")
                if med_ms:
                    line += f" | media p50={pct(med_ms,.5):6.1f}ms"
                if tot_ms:
                    line += (f" | TOTAL p50={pct(tot_ms,.5):6.1f} "
                             f"p95={pct(tot_ms,.95):6.1f}ms")
                print(line, flush=True)

            if args.seconds > 0 and (time.perf_counter() - t_start) >= args.seconds:
                break

    except KeyboardInterrupt:
        print("\n[live_detect] interrupted")
    finally:
        src.close()
        cv2.destroyAllWindows()

    el = time.perf_counter() - t_start
    hit = sum(1 for c in nbox if c >= args.min_boxes)

    print("\n========== 汇总 ==========")
    print(f"时长          : {el:.1f}s")
    print(f"帧数          : {n}  ({n / el:.2f} fps)")
    print(f"探针解码失败  : {miss} / {n}")
    if nbox:
        print(f"每帧框数      : p50={pct(nbox,.5):.0f} p95={pct(nbox,.95):.0f} "
              f"max={max(nbox)}")
        print(f"有目标的帧    : {hit}/{len(nbox)} ({100.0 * hit / max(1,len(nbox)):.1f}%)")
    if inf_ms:
        print(f"推理耗时      : p50={pct(inf_ms,.5):.1f} p95={pct(inf_ms,.95):.1f} "
              f"p99={pct(inf_ms,.99):.1f} max={max(inf_ms):.1f} ms")
    if med_ms:
        print(f"媒体延迟      : p50={pct(med_ms,.5):.1f} p95={pct(med_ms,.95):.1f} ms")
    if tot_ms:
        print(f"端到端总延迟  : p50={pct(tot_ms,.5):.1f} p95={pct(tot_ms,.95):.1f} "
              f"p99={pct(tot_ms,.99):.1f} ms")
        print("                ↑ 从 A 机产生画面，到 B 机算出目标位置")
    elif med_ms:
        print("端到端总延迟  : 无样本")
    return 0


if __name__ == "__main__":
    sys.exit(main())
