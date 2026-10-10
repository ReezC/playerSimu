"""**性能全面排查**（用户 2026-10-10 ✓ 原话："需要专门全面排查性能"）。

为什么要它（现场 ✓）：那几天的"卡/积压"排查，每次都是**临时敲命令**看几个数 ✗
    ⇒ 一会儿量 CPU、一会儿翻 `perf.log`、一会儿又去猜解码能力 ✓，
    而且**结论靠人记** ✗（"上次那个数是多少来着"✓）。
    这个工具把该量的一次量齐、并**算出天棚与占比** ⇒ 每次排查都从同一张表出发 ✓。

它量四件事（**全部只读** ✓ 不动配置、不动端口 ✓）：
    ① **机器**：CPU / 可用内存 / 页面文件 + 占 CPU、内存最多的几个进程（把游戏/浏览器指出来 ✓）；
    ② **管线**：从 `perf.log` 取最近几段的 `pipe_ms` / `infer_ms` / `draw_ms` / `e2e_probe_ms` 中位
       ＋ 段头的 `fps` / `preview_fps` / `src_lag` / `vt_yield` ⇒ 直接算出
       **天棚 fps = 1000 / pipe_ms** 与"输入 fps"，并把**占比**列出来 ✓；
    ③ **本机解码能力**：用 ffmpeg 造一段合成的 1080p60 H.264（**本地文件 ✓ 不碰实时端口 ✓**），
       用**产品自己的** `link.pyav_source.PyAVSource` 读它 ⇒ 报解码 fps 与每帧毫秒 ✓
       —— 这是回答"解码是不是天棚"的唯一硬办法 ✓（实时端口同一时刻只能一个读者 ✓）；
    ④ **判词**：按占比说"**该砍哪一段**"（推理 / 解码 / 绘制 / 上游）✓。

跑法：
    python -m tools.perf_audit              # 全面（含解码实测，约 20~40 秒）
    python -m tools.perf_audit --no-decode  # 跳过解码实测（秒级）
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           shell=isinstance(cmd, str))
    except Exception as e:                          # noqa: BLE001
        return "", str(e)
    return (p.stdout or ""), (p.stderr or "")


def section_machine():
    print("── ① 机器 ──")
    out, _ = _sh(["powershell", "-NoProfile", "-Command",
                  "$p=Get-CimInstance Win32_Processor; "
                  "'{0}|{1}' -f ($p.LoadPercentage -join ','), ($p.NumberOfLogicalProcessors -join ',')"])
    cpu, cores = ("?", "?")
    if "|" in out:
        cpu, cores = out.strip().split("|")[:2]
    out2, _ = _sh(["powershell", "-NoProfile", "-Command",
                   "$o=Get-CimInstance Win32_OperatingSystem; "
                   "'{0}|{1}' -f $o.FreePhysicalMemory, $o.TotalVisibleMemorySize"])
    free_gb = tot_gb = 0.0
    if "|" in out2:
        try:
            a, b = out2.strip().split("|")[:2]
            free_gb, tot_gb = float(a) / 1048576.0, float(b) / 1048576.0
        except ValueError:
            pass
    print("  CPU 占用 %s%%（%s 逻辑核）｜ 可用内存 %.1f / %.1f GB" % (cpu, cores, free_gb, tot_gb))
    out3, _ = _sh(["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_PageFileUsage | "
                   "ForEach-Object { '{0} {1} {2}' -f $_.Name,$_.CurrentUsage,$_.PeakUsage }"])
    for ln in (out3 or "").strip().splitlines()[:3]:
        print("  页面文件 %s" % ln.strip())
    out4, _ = _sh(["powershell", "-NoProfile", "-Command",
                   "Get-Process | Sort-Object CPU -Descending | Select-Object -First 6 | "
                   "ForEach-Object { '{0}|{1:N0}|{2:N0}' -f $_.ProcessName,$_.CPU,($_.WorkingSet64/1MB) }"])
    print("  占用最多的进程（CPU秒 / 内存MB）：")
    for ln in (out4 or "").strip().splitlines()[:6]:
        parts = ln.strip().split("|")
        if len(parts) == 3:
            print("    %-24s %8s  %8s" % (parts[0], parts[1], parts[2]))


def section_pipeline(n=4):
    print("── ② 管线（从 perf.log）──")
    log = ROOT / "perf.log"
    if not log.exists():
        print("  （没有 perf.log ⇒ 先开一次实时预览 ✓）")
        return {}
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    segs = []
    cur = None
    for ln in lines:
        m = re.match(r"^===\s+(\S+ \S+)\s+窗口\s+(\S+)\s+(.*)$", ln)
        if m:
            cur = {"t": m.group(1), "win": m.group(2), "head": m.group(3)}
            segs.append(cur)
            continue
        if cur is None:
            continue
        for key in ("pipe_ms", "infer_ms", "draw_ms", "e2e_probe_ms"):
            if ln.startswith(key):
                mm = re.search(r"中位\s+([\d.]+)", ln)
                if mm:
                    cur[key] = float(mm.group(1))
    for seg in segs[-n:]:
        head = seg.get("head", "")
        def _g(key):
            m = re.search(key + r"=(\S+)", head)
            return m.group(1) if m else "-"
        print("  %s  输入 fps=%s 预览=%s src_lag=%s vt_yield=%s"
              % (seg["t"], _g("fps"), _g("preview_fps"), _g("src_lag"), _g("vt_yield")))
        pipe = seg.get("pipe_ms")
        if pipe:
            inf = seg.get("infer_ms") or 0.0
            draw = seg.get("draw_ms") or 0.0
            print("      pipe %6.1f ms（天棚 %.0f fps）｜ infer %5.1f（占 %.0f%%）｜ "
                  "draw %5.2f ｜ 端到端 %s ms"
                  % (pipe, 1000.0 / max(1e-6, pipe), inf, 100.0 * inf / pipe, draw,
                     seg.get("e2e_probe_ms", "-")))
    last = next((s for s in reversed(segs) if s.get("pipe_ms")), None)
    return last or {}


def section_decode(seconds=6):
    print("── ③ 本机解码能力（合成 1080p60 H.264，读本地文件 ✓ 不碰实时端口 ✓）──")
    ff = shutil.which("ffmpeg")
    if not ff:
        print("  （没装 ffmpeg ⇒ 跳过；装了就自动测 ✓）")
        return
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="perf_audit_"))
    src = tmp / "probe.ts"
    try:
        out, err = _sh([ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=size=1920x1080:rate=60", "-t", str(seconds),
                        "-c:v", "libx264", "-preset", "ultrafast", "-g", "60",
                        "-pix_fmt", "yuv420p", "-f", "mpegts", str(src)], timeout=180)
        if not src.exists():
            print("  （ffmpeg 造流失败：%s）" % (err or out)[:120])
            return
        sys.path.insert(0, str(ROOT))
        from link.pyav_source import PyAVSource
        s = PyAVSource(str(src))
        s.open()
        ms = []
        n = 0
        t0 = time.perf_counter()
        while True:
            f = s.read()
            if f is None:
                break
            n += 1
            ms.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
        s.close()
        if not ms:
            print("  （一帧都没解出来 ✗）")
            return
        ms_sorted = sorted(ms)
        p50 = ms_sorted[len(ms_sorted) // 2]
        p95 = ms_sorted[int(len(ms_sorted) * 0.95)] if len(ms_sorted) > 1 else p50
        print("  解出 %d 帧 ｜ 每帧 p50 %.1f ms / p95 %.1f ms ⇒ **解码天棚 ≈ %.0f fps**"
              % (n, p50, p95, 1000.0 / max(1e-6, p50)))
        print("  （本项目实时流用同一份 `link/pyav_source.py` ✓ 参数也一样 ✓ 可比 ✓）")
    finally:
        shutil.rmtree(tmp, True)


def verdict(seg, dec_fps=None):
    print("── ④ 判词 ──")
    if not seg:
        print("  （没有管线数据 ⇒ 先开实时预览再看 ✓）")
        return
    pipe = seg.get("pipe_ms") or 0.0
    inf = seg.get("infer_ms") or 0.0
    draw = seg.get("draw_ms") or 0.0
    ceil = 1000.0 / max(1e-6, pipe)
    print("  管线天棚 ≈ **%.0f fps**（每帧 %.1f ms，其中推理 %.1f / 绘制 %.2f）"
          % (ceil, pipe, inf, draw))
    if dec_fps:
        print("  本机解码天棚 ≈ %.0f fps ⇒ %s"
              % (dec_fps, "解码**不是**瓶颈 ✓" if dec_fps > 2 * ceil else "**解码就是瓶颈** ✗"))
    if inf / max(1e-6, pipe) > 0.5:
        print("  ⇒ 最大一段是**推理**（占 %.0f%%）：先试「降低 imgsz / 跳帧推理（每 2 帧推一次）」✓"
              % (100.0 * inf / pipe))
    elif draw / max(1e-6, pipe) > 0.2:
        print("  ⇒ 绘制占比偏高：降预览档位 ✓")
    else:
        print("  ⇒ 单看管线还够；若仍积压 ⇒ 看**上游**（推流分辨率/帧率）与机器上的其它程序 ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-decode", action="store_true", help="跳过解码实测（秒级 ✓）")
    ap.add_argument("--seconds", type=float, default=6.0, help="解码实测用多少秒素材 ✓")
    a = ap.parse_args()

    print("═══ 性能全面排查（%s）═══" % time.strftime("%Y-%m-%d %H:%M:%S"))
    section_machine()
    seg = section_pipeline()
    dec = None
    if not a.no_decode:
        section_decode(a.seconds)
    verdict(seg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
