# =====================================================================
# A-machine push script  (run on machine A: the game machine)
# Requires: ffmpeg (winget install --id Gyan.FFmpeg -e), NVIDIA GPU
#
# Measured end-to-end latency (B-machine probe): ~131 ms
#
# Quiet mode:  -hide_banner -loglevel error
#   Hides the banner and the yellow "Non-monotonic DTS" warnings, while
#   KEEPING the frame=...fps=... status line so you can tell it is alive.
#   For full troubleshooting log, use -loglevel warning or -loglevel info.
#
# Parameter notes:
#   -fps_mode passthrough      disable frame-rate conversion.
#                              Omitting it costs ~225 ms of hidden buffering.
#   ddagrab=framerate=144      higher capture rate -> lower latency.
#                              Measured: 60fps -> 316ms, 144fps -> 131ms
#   -muxdelay 0 -muxpreload 0  remove mpegts muxer buffering (~1.2 s)
#   -flush_packets 1           send each packet immediately
#   -loglevel warning          keep warnings only (use this instead of
#                              "error" if you want to see DTS warnings)
# =====================================================================

$B_HOST = "192.168.1.2"
$PORT   = 5000

Write-Host "pushing to udp://${B_HOST}:${PORT}   (Ctrl+C to stop)" -ForegroundColor Cyan

ffmpeg -hide_banner -nostats -loglevel error `
  -init_hw_device d3d11va=dx -filter_hw_device dx `
  -filter_complex "ddagrab=framerate=144,hwdownload,format=bgra,scale=1920:1080,format=yuv420p" `
  -fps_mode passthrough `
  -c:v h264_nvenc -preset p1 -tune ull -zerolatency 1 -rc cbr `
  -b:v 12M -bufsize 1M -g 30 -bf 0 `
  -flush_packets 1 -muxdelay 0 -muxpreload 0 `
  -f mpegts "udp://${B_HOST}:${PORT}?pkt_size=1316"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ffmpeg exited abnormally, code = $LASTEXITCODE" -ForegroundColor Red
}
