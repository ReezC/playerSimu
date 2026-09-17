"""【B 机运行】把一个视频文件循环推成 UDP/mpegts 流，模拟 A 机推流。

注意：PyAV 在 Windows 上直接 av.open("udp://...", mode="w") 会在 mux 时报
EINVAL(22)。这里改用自定义 writable 对象 + Python socket 发送，
绕过 FFmpeg 的 UDP protocol 层。
"""

import argparse
import re
import socket
import sys
import time

import av

from tools.config import get


class UDPWriter:
    def __init__(self, host, port, pkt_size=1316):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, port)
        self.pkt_size = pkt_size

    def write(self, data) -> int:
        mv = memoryview(data)
        for i in range(0, len(mv), self.pkt_size):
            self.sock.sendto(mv[i:i + self.pkt_size], self.addr)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def parse_target(target, default_port):
    m = re.match(r"udp://([^:/?#]*)(?::(\d+))?", target or "")
    host = (m.group(1) if m and m.group(1) else "127.0.0.1")
    port = int(m.group(2)) if m and m.group(2) else default_port
    return host, port


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()

    port = get("stream", "port", 5000)
    fps = get("stream", "fps", 60)
    w = get("stream", "width", 1280)
    h = get("stream", "height", 720)
    host, port = parse_target(args.target, port)

    inp = av.open(args.file, mode="r")
    istream = inp.streams.video[0]

    out = av.open(UDPWriter(host, port), mode="w", format="mpegts")
    ostream = out.add_stream("libx264", rate=fps)
    ostream.width, ostream.height = w, h
    ostream.pix_fmt = "yuv420p"
    ostream.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "20", "g": "30"}

    print(f"[push_clip] {args.file} -> udp://{host}:{port} loop={args.loop} "
          f"duration={args.duration}s", flush=True)

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

    print(f"[push_clip] 结束: {rounds} 轮 / {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
