"""【B 机运行】录制实时流到文件，供 FileSource 离线回放。

用法：
    python -m tools.record --out session01.mp4 --seconds 60
    python -m tools.record --url "udp://@:5000" --out s1.mkv --seconds 0   # 0=不限，Ctrl+C 停
"""

import argparse
import signal
import sys
import time

import av

from link import PyAVSource
from tools.config import get, record_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--stride", type=int, default=1,
                    help="每 N 帧只存 1 帧。背景素材不需要全帧率，--stride 6 可把体积降到 1/6")
    args = ap.parse_args()

    url = args.url or get("stream", "url")
    out_path = record_dir() / (args.out or f"rec_{time.strftime('%Y%m%d_%H%M%S')}.mkv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = PyAVSource(url)
    src.open()
    w, h = src.size or (1280, 720)

    # PyAV 的 add_stream(rate=...) 不接受 float，必须是 int 或 Fraction。
    # 之前直接传 src.fps(float 144.0) 会抛 AttributeError: 'float' object has no attribute 'numerator'。
    fps = int(round(src.fps or 60.0))
    stride = max(1, args.stride)
    eff_fps = max(1, fps // stride)
    print(f"[record] src={url} {w}x{h}@{fps}  stride={stride} -> 输出 {eff_fps}fps")
    print(f"[record] 写入 {out_path}")

    out = av.open(str(out_path), mode="w")
    codec_name = "h264_nvenc" if _has_nvenc(out) else "libx264"
    stream = out.add_stream(codec_name, rate=eff_fps)
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    stream.options = ({"preset": "p1", "cq": str(args.crf)}
                      if codec_name == "h264_nvenc"
                      else {"preset": "veryfast", "crf": str(args.crf)})

    stop = {"v": False}

    def on_sigint(*_):
        stop["v"] = True

    signal.signal(signal.SIGINT, on_sigint)

    t0 = time.perf_counter()
    n = 0
    seen = 0
    try:
        while not stop["v"]:
            if args.seconds > 0 and (time.perf_counter() - t0) > args.seconds:
                break
            f = src.read()
            if f is None:
                break
            seen += 1
            if (seen - 1) % stride != 0:
                continue
            vf = av.VideoFrame.from_ndarray(f.image, format="rgb24")
            for packet in stream.encode(vf):
                out.mux(packet)
            n += 1
    finally:
        for packet in stream.encode():
            out.mux(packet)
        out.close()
        src.close()

    dt = time.perf_counter() - t0
    print(f"[record] 完成：{n} 帧 / {dt:.1f}s = {n / dt:.1f} fps -> {out_path}")
    return 0


def _has_nvenc(container) -> bool:
    try:
        av.codec.Codec("h264_nvenc", "w")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
