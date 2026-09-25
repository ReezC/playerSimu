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
from tools.probe_tune import initial_geo        # 探针几何：与工作台同一套来源


def load_offset_ms(cli_offset: float | None) -> float:
    if cli_offset is not None:
        return cli_offset
    # 自动对时：Windows 墙钟 NTP 漂移会让 offset 几分钟内偏上百毫秒，
    # 用旧值算出的延迟会差一个系统误差。启动时重新对时一次，失败才回退文件。
    try:
        from tools.clock_sync import sync_offset
        off = sync_offset(save=True)
        if off is not None:
            return off
    except Exception:
        pass
    p = ROOT / "config" / "clock_offset.txt"
    if p.exists():
        try:
            return float(p.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    print("[probe_recv] 无法对时也找不到时钟偏移，延迟统计不可用。"
          "请确认 A 机 clock_server 已启动")
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
    ap.add_argument("--format", type=str, default=None,
                    help="强制输入格式。裸流必须指定 —— "
                         "ffmpeg 用 -f mjpeg 推的流没有容器头，自动探测会失败")
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
    bits = int(get("probe", "bits", 40))
    offset_ms = load_offset_ms(args.offset) if use_probe else 0.0

    if args.file:
        src = FileSource(args.file, realtime=True)
    else:
        src = PyAVSource(args.url or get("stream", "url"),
                         container_format=args.format)

    src.open()
    w, h = src.size or (0, 0)

    # 探针参数**不再按流分辨率缩放**。
    #
    # config/link.yaml 里的 x/y/cell/gap 是「流画面里的像素值」，由 A 机
    # probe_gen 的 out_scale 保证缩放后正好等于这个尺寸。所以这里直接用，
    # 不缩放。
    #
    # 之前那套「流宽 != 1920 就 ×比例缩放」的逻辑是给临时实验用的
    # （去掉 A 机 scale 后流变 2560），现在流分辨率固定为 1366x768，
    # 保留缩放反而会误伤：1366 会被 ×0.711，把 18px 方块压成 12.8px，
    # 探针直接失效。
    EXPECT_W = 1366
    if w and abs(w - EXPECT_W) > 1:
        print(f"[probe_recv] ⚠ 流宽 {w} 与配置预期 {EXPECT_W} 不同，"
              f"探针参数可能不匹配（需调整 out_scale 或 cell）")

    # 探针几何：**和工作台同一套优先级**（项目标定 → 全局文件 → link.yaml）。
    #
    # 老写法只读 link.yaml 的 probe.x/y/cell/gap，而工作台走 pick_calib —— 两边在
    # 两套坐标里比大小，对不上时**谁都不报警**。实测（2026-09-25）：link.yaml 那几
    # 个数是旧的（注释写 A 机按 cell=23/gap=3 画、文件里是 cell=16/gap=2，中心距
    # 21 vs 18），每块偏 2.8px、42 块累积 118px → 低位全采错 → 这个工具报出来的
    # "延迟" 是一坨乱数（p95 4.7s，紧贴 --max-delay），被当成真延迟查了半天。
    # 所以现在：用同一份几何 + **把用的是哪份、以及 link.yaml 差多少都打出来**。
    geo, geo_src = initial_geo((h, w, 3), bits) if use_probe else (
        {"x": 0.0, "y": 0.0, "cell": 16.0, "gap": 2.0}, "未启用")
    px, py = geo["x"], geo["y"]
    cell, gap = geo["cell"], geo["gap"]
    cfg_geo = (float(get("probe", "x", 100)), float(get("probe", "y", 8)),
               float(get("probe", "cell", 16)), float(get("probe", "gap", 2)))
    if use_probe and max(abs(cfg_geo[0] - px), abs(cfg_geo[1] - py),
                         abs(cfg_geo[2] - cell), abs(cfg_geo[3] - gap)) > 0.26:
        print("[probe_recv] ⚠ 实际用的几何与 link.yaml 不一致（差 >0.25px，会逐块累积）："
              "link.yaml x=%.1f y=%.1f cell=%.2f gap=%.2f"
              % cfg_geo)

    print(f"[probe_recv] source={args.file or src.url} {w}x{h} fps={src.fps}")
    print(f"[probe_recv] probe={'on' if use_probe else 'off'} offset={offset_ms:.3f}ms "
          f"region=({px:.1f},{py:.1f}) cell={cell:.2f} gap={gap:.2f} 几何来源={geo_src}")

    gaps: list[float] = []
    delays: list[float] = []
    reads: list[float] = []      # src.read() 单帧耗时，用来判断"是否已有积压"
    miss = 0
    n = 0
    last_mono = None
    t_report = time.perf_counter()
    t_start = time.perf_counter()

    try:
        while True:
            # 单独计时 read()：这是判断"延迟在 A 机还是 B 机"的关键。
            #
            #   read() ≈ 16.7ms（一帧周期）-> B 机在实时等帧，数据一到就走，
            #                                延迟全部发生在 A 机侧；
            #   read() ≈ 0~2ms            -> 帧早就堆在 B 机内存里了，
            #                                说明 B 机内部还有一层深缓冲
            #                                （而这一层不在我们改过的参数里）。
            _t0 = time.perf_counter()
            f = src.read()
            _t1 = time.perf_counter()
            if f is None:
                break
            n += 1
            reads.append((_t1 - _t0) * 1000.0)

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
                if reads:
                    line += (f" | read p50={pct(reads,.5):5.1f} "
                             f"p95={pct(reads,.95):6.1f}ms")
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
