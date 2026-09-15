"""【B 机运行】把一个视频文件循环推成 UDP/mpegts 流，模拟 A 机推流。

用于在没有 A 机时验证收流链路（PyAVSource + 低延迟参数 + 探针解码）。

用法：
    python -m tools.push_clip --file data/recordings/testclip.mp4 --loop --duration 20
"""

import argparse
import sys
import time

import av

from tools.config import get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--target", type=str, default=None,
                    help="默认 udp://127.0.0.1:<stream.port>")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--duration", type=float, default=20.0, help="推送时长(秒)")
    args = ap.parse_args()

    # 注意：写模式下 PyAV 不解析 URL query，pkt_size 之类必须走 options，
    # 写成 "udp://host:port?pkt_size=1316" 会在 mux 时报 EINVAL(22)。
    target = args.target or f"udp://127.0.0.1:{get('stream', 'port', 5000)}"
    port = get("stream", "port", 5000)
    fps = get("stream", "fps", 60)
    w = get("stream", "width", 1280)
    h = get("stream", "height", 720)

    inp = av.open(args.file, mode="r")
    istream = inp.streams.video[0]

    out = av.open(target, mode="w", format="mpegts", options={"pkt_size": "1316"})
    ostream = out.add_stream("libx264", rate=fps)
    ostream.width, ostream.height = w, h
    ostream.pix_fmt = "yuv420p"
    ostream.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "20", "g": "30"}

    print(f"[push_clip] {args.file} -> {target} loop={args.loop} duration={args.duration}s", flush=True)

    t0 = time.perf_counter()
    rounds = 0
    try:
        while True:
            for frame in inp.decode(istream):
                if time.perf_counter() - t0 > args.duration:
                    raise StopIteration
                for packet in ostream.encode(frame):
                    out.mux(packet)
            rounds += 1
            if not args.loop:
                break
            inp.seek(0, stream=istream)
    except StopIteration:
        pass
    finally:
        for packet in ostream.encode():
            out.mux(packet)
        out.close()
        inp.close()

    print(f"[push_clip] 结束：{rounds} 轮 / {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
