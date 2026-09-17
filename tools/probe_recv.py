"""【B 机运行】收流自检：实时显示 + FPS / 帧间隔 / 端到端延迟统计。

用法：
    python -m tools.probe_recv --show
    python -m tools.probe_recv --show --offset 3.42          # 手动指定时钟偏移(ms)
    python -m tools.probe_recv --file data/recordings/s1.mkv # 离线回放
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from link import FileSource, PyAVSource
from tools.config import ROOT, get
from tools.probe_codec import decode_ms, resolve_delay_ms


def load_offset_ms(cli_offset: float | None) -> float:
    if cli_offset is not None:
        return cli_offset
    p = ROOT / "config" / "clock_offset.txt"
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    print("[probe_recv] 未找到时钟偏移，延迟统计不可用。"
          "请先运行 `python -m tools.clock_sync --host <A机IP> --save`")
    return 0.0


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=None)
    ap.add_argument("--file", type=str, default=None, help="离线回放文件（与 --url 二选一）")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--probe", action="store_true", default=None, help="启用屏幕时间码解码")
    ap.add_argument("--offset", type=float, default=None, help="时钟偏移 ms（t_A = t_B + offset）")
    ap.add_argument("--max-delay", type=float, default=5000.0,
                    help="延迟有效上限 ms，超出视为异常丢弃（离线回放自测时调大）")
    ap.add_argument("--report", type=float, default=2.0, help="统计打印间隔(秒)")
    ap.add_argument("--seconds", type=float, default=0.0, help="运行时长(秒)，0=不限（Ctrl+C 停）")
    ap.add_argument("--diag", type=str, default=None,
                    help="诊断：存一张叠加图（绿=配置位置，红=自动定位），值为输出路径")
    args = ap.parse_args()

    cfg_probe = bool(get("probe", "enabled", True))
    use_probe = cfg_probe if args.probe is None else args.probe
    px, py = get("probe", "x", 100), get("probe", "y", 8)
    cell, gap, bits = get("probe", "cell", 16), get("probe", "gap", 2), get("probe", "bits", 40)
    offset_ms = load_offset_ms(args.offset) if use_probe else 0.0

    if args.file:
        src = FileSource(args.file, realtime=True)
    else:
        src = PyAVSource(args.url or get("stream", "url"))

    src.open()
    w, h = src.size or (0, 0)
    print(f"[probe_recv] source={args.file or src.url} {w}x{h} fps={src.fps}")
    print(f"[probe_recv] probe={'on' if use_probe else 'off'} offset={offset_ms:.3f}ms "
          f"region=({px},{py}) cell={cell}")

    gaps: list[float] = []
    delays: list[float] = []
    miss = 0
    n = 0
    last_mono = None
    t_report = time.perf_counter()
    t_start = time.perf_counter()

    try:
        while True:
            f = src.read()
            if f is None:
                break
            n += 1

            if last_mono is not None:
                gaps.append((f.t_recv_mono - last_mono) * 1000.0)
            last_mono = f.t_recv_mono

            if use_probe:
                gray = cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY)
                ts_a = decode_ms(gray, px, py, cell, gap, bits)
                if ts_a is None:
                    miss += 1
                else:
                    d = resolve_delay_ms((f.t_recv_wall + offset_ms / 1000.0) * 1000.0, ts_a)
                    if d is not None and d < args.max_delay:
                        delays.append(d)

            if args.show or (args.diag and n == 1):
                vis = cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR)
                if use_probe:
                    # 坐标必须取整：gap 配的是 2.25（浮点），
                    # 直接参与运算会让坐标变成 float，cv2 不接受
                    # （报 Can't parse 'pt2'. Sequence item ... wrong type）。
                    bx1 = int(round(px + (2 + bits) * (cell + gap) + 2))
                    by2 = int(round(py + cell + 4))
                    cv2.rectangle(vis, (int(round(px)) - 2, int(round(py)) - 2),
                                  (bx1, by2), (0, 255, 0), 1)

                    if args.diag and n == 1:
                        # 绿框 = 配置里写的 probe.x/y；红框 = 自动搜索找到的位置。
                        # 两者一叠就能看出配置偏了多少、往哪偏。光看绿框歪不歪
                        # 说不清是配置错了还是画面整体有位移。
                        from tools.probe_auto import locate

                        r = locate(gray, int(cell), float(gap), int(bits))
                        if r:
                            _c, ax, ay = r
                            cv2.rectangle(vis, (int(ax) - 2, int(ay) - 2),
                                          (int(round(ax + (2 + bits) * (cell + gap) + 2)),
                                           int(round(ay + cell + 4))), (0, 0, 255), 1)
                            print(f"[diag] 自动定位 x={ax} y={ay} | "
                                  f"配置 x={px} y={py} | "
                                  f"偏差 dx={ax - int(px)} dy={ay - int(py)}")
                        else:
                            print("[diag] 自动定位失败：这一帧里没搜到探针")

                        # 画个十字标出配置原点，方便核对
                        ox, oy = int(round(px)), int(round(py))
                        cv2.line(vis, (ox - 12, oy), (ox + 12, oy), (255, 0, 255), 1)
                        cv2.line(vis, (ox, oy - 12), (ox, oy + 12), (255, 0, 255), 1)
                        cv2.putText(vis, f"cfg({px},{py})", (ox + 6, oy + 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    if delays:
                        cv2.putText(vis, f"lat {delays[-1]:.1f}ms", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(vis, "no probe", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if args.diag and n == 1:
                    # 走 imwrite 编码而不是 cv2.imwrite：OpenCV 在中文路径下
                    # 会静默失败（返回 False 不抛异常），存出来的图根本不存在。
                    ok, buf = cv2.imencode(".png", vis)
                    if ok:
                        path = Path(args.diag)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(buf.tobytes())
                        print(f"[diag] 已保存 {path}")
                    else:
                        print("[diag] 编码失败")

                if args.show:
                    cv2.imshow("probe_recv", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if args.seconds > 0 and (time.perf_counter() - t_start) >= args.seconds:
                break

            if time.perf_counter() - t_report >= args.report:
                t_report = time.perf_counter()
                el = time.perf_counter() - t_start
                line = (f"[{el:7.1f}s] frames={n:6d} fps={n / el:6.2f}")
                if gaps:
                    line += (f" | gap p50={pct(gaps,.5):6.2f} p95={pct(gaps,.95):6.2f} "
                             f"max={max(gaps):7.2f}ms")
                if delays:
                    line += (f" | lat p50={pct(delays,.5):6.1f} p95={pct(delays,.95):6.1f} "
                             f"p99={pct(delays,.99):6.1f}ms miss={miss}")
                print(line, flush=True)

    except KeyboardInterrupt:
        print("\n[probe_recv] interrupted")
    finally:
        src.close()
        cv2.destroyAllWindows()

    el = time.perf_counter() - t_start
    print("\n========== 汇总 ==========")
    print(f"时长        : {el:.1f}s")
    print(f"帧数        : {n}  (均值 {n / el:.2f} fps)")
    if gaps:
        print(f"帧间隔      : p50={pct(gaps,.5):.2f} p95={pct(gaps,.95):.2f} "
              f"p99={pct(gaps,.99):.2f} max={max(gaps):.2f} ms "
              f"(jitter={statistics.pstdev(gaps):.2f})")
    print(f"探针解码失败: {miss} / {n}")
    if delays:
        print(f"端到端延迟  : p50={pct(delays,.5):.1f} p95={pct(delays,.95):.1f} "
              f"p99={pct(delays,.99):.1f} max={max(delays):.1f} ms "
              f"(jitter={statistics.pstdev(delays):.2f})")
    else:
        print("端到端延迟  : 无有效样本（探针未解码或超出 --max-delay）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
