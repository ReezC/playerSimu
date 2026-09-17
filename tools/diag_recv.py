"""收流诊断：定位 PyAV 为什么收不到/解不了。

用法:
    python -m tools.diag_recv "udp://@:5000" 8
"""

import itertools
import os
import sys
import threading

import av


def probe(url, seconds):
    res = {}

    def work():
        try:
            c = av.open(url, mode="r")
            s = c.streams.video[0]
            res["codec"] = s.codec_context.name
            res["size"] = (s.codec_context.width, s.codec_context.height)
            frames = list(itertools.islice(c.decode(s), 5))
            res["frames"] = len(frames)
            if frames:
                res["shape"] = frames[0].to_ndarray(format="rgb24").shape
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(seconds)
    return res


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "udp://@:5000"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

    print(f"URL  : {url}")
    print(f"超时 : {secs}s")
    r = probe(url, secs)
    print(f"结果 : {r}")
    if not r:
        print("=> 超时：啥都没拿到（open 或 decode 一直阻塞）")
    elif "error" in r:
        print("=> 抛异常")
    elif r.get("frames", 0) > 0:
        print("=> 正常，能收到并解码")
    else:
        print("=> 流信息拿到了，但解不出帧")

    sys.stdout.flush()
    os._exit(0)
