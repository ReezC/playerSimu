"""检测 PyAV 内置 FFmpeg 支持哪些传输协议（临时脚本）。"""
import av

print("PyAV", av.__version__)
print()

for proto in ("udp", "tcp", "rtp", "rtsp", "rtmp", "srt", "rist", "http", "https", "unix", "file"):
    url = "%s://127.0.0.1:1" % proto
    if proto == "file":
        url = "file:///nonexistent_xyz"
    try:
        av.open(url, timeout=0.3)
        print("  %-6s 支持" % proto)
    except Exception as e:
        msg = str(e)
        if "Protocol not found" in msg:
            print("  %-6s ✗ 不支持" % proto)
        else:
            print("  %-6s ✓ 支持   (%s)" % (proto, msg.splitlines()[0][:60]))
