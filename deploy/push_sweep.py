"""【A 机运行】推流自检 · 推流侧：按候选清单逐条试推并采 A 机指标。

配套：B 机跑 `python -m tools.stream_sweep` 量端到端延迟。两边按**同一份清单、
同一个顺序**走，各记起止墙钟，最后把 B 机那份 `perf_push_B.json` 拷过来合并出表
（见 tools/push_presets.py 的 merge / rank）。

两种跑法（同一份实现）：
    部署台 → 工具栏「推流自检」      （界面里点，进度走右下角日志）
    python -m deploy.push_sweep      （想在命令行跑 / 定位问题时）

**采什么、为什么**（详见 tools/push_presets.py 的说明）：
    · `speed`  —— ffmpeg 自己的处理速度，**< 1.00 就是这台机跟不上**（一定丢帧、涨延迟）。
                  这是"A 机扛不扛得住"唯一直接的证据，所以这一路必须带 `-stats`
                  （部署台卡片命令按设计带 `-nostats`，那条状态行会刷屏；这里替换掉）。
    · 实际 fps / 丢帧数 —— 对照目标帧率看有没有偷偷降帧。
    · CPU%（psutil，装了才有）与 NVENC 利用率（nvidia-smi，有 N 卡才有）——
                  余量信息：延迟一样时，占用低的那套更值得选。
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy import config as dcfg, services                      # noqa: E402
from tools import push_presets as pp, sweep_link                 # noqa: E402

#: 发完报告后等 B 机回结论最多多久（B 机还有最后一段要量完 + 合并出表）
RESULT_WAIT_S = 300.0


def make_announcer(cfg):
    """按配置建公告通道（A 机 → B 机）→ Announcer / None。

    **返回 None 的每一种情况都不该影响自检**：配置里没 B 机地址、端口被占、
    网络层出问题 —— 自检照跑，只是两边退回"人掐时间"那套老流程（见 sweep_link 的说明）。
    """
    host = (cfg.get("push") or {}).get("host")
    if not host:
        return None
    try:
        port = int((cfg.get("sweep") or {}).get("port") or sweep_link.DEFAULT_PORT)
        # 绑到同一个端口号：B 机的回执是回给"公告的源地址:源端口"的，
        # 而部署台的日志里那句「B 机已跟上」就是从这张 socket 上读到的。
        return sweep_link.Announcer(str(host), port, bind_port=port)
    except Exception:                            # noqa: BLE001
        return None


def cmd_for(push):
    """推流命令行 —— 用部署台的**同一份**实现，只把 `-nostats` 换成 `-stats`。

    为什么不在 services.py 加参数：卡片那条命令的形态（`-nostats`）是有意的
    （状态行会把日志刷爆），自检是**另一条路**；在这里替换，改动面最小、
    而且"自检看的就是同一套命令、只多了状态行"这件事一眼可见。
    """
    cmd = [str(a) for a in services.build_cmd("push", push)]
    out, done = [], False
    for a in cmd:
        if a == "-nostats" and not done:
            out.append("-stats")
            done = True
        else:
            out.append(a)
    if not done:
        out.insert(1, "-stats")
    return out


def _psutil_cpu_pct(pid, t0, cpu_t0):
    """这个进程从 t0 到现在占了多少 CPU%（best-effort，没有 psutil 就 None）。"""
    try:
        import psutil
        p = psutil.Process(pid)
        cpu = sum(p.cpu_times()[:2])
        el = max(1e-6, time.monotonic() - t0)
        n = psutil.cpu_count() or 1
        return (cpu - cpu_t0) / el / n * 100.0
    except Exception:
        return None


def _psutil_cpu_t0(pid):
    try:
        import psutil
        return sum(psutil.Process(pid).cpu_times()[:2])
    except Exception:
        return 0.0


def _gpu_enc_pct():
    """NVENC 编码器利用率%（best-effort）。取不到返回 None。

    为什么看它：144fps + 高码率下唯一可能先顶到墙的是编码器本身；
    它到 90%+ 就意味着"再加帧率/码率要开始掉帧"，这个余量比 CPU 更难事后补。
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.encoder",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        first = r.stdout.strip().splitlines()[0].split(",")
        if len(first) < 2:
            return None
        return {"gpu_pct": float(first[0].strip()),
                "gpu_enc_pct": float(first[1].strip())}
    except Exception:
        return None


class PushSweep(threading.Thread):
    """逐条试推。可以在 GUI 线程外跑（QThread 里也行）。

    on_line(text)  —— 进度/每段结果（界面里显示到日志）
    on_done(rows, path) —— 收尾（rows 已落盘 + 已按评分排好）
    """

    def __init__(self, cfg, presets_path=None, on_line=None, on_done=None,
                 seconds=None, settle=None, on_result=None):
        super().__init__(daemon=True)
        self.cfg = dict(cfg or {})
        self.presets_path = presets_path
        self.on_line = on_line or (lambda _t: None)
        self.on_done = on_done or (lambda _rows, _p: None)
        #: B 机回传的**结论**（合并后的表 + 建议参数）。跑完 A 机自己只有 A 侧那几个数，
        #: 延迟那一半在 B 机 —— 所以这个回调就是"最后告诉我结果"那条路。
        #: 注意它是在**本线程**里被调的：接 Qt 的话请接信号 emit（见 deploy/app.py）。
        self.on_result = on_result or (lambda _t: None)
        self._seconds = seconds
        self._settle = settle
        self._stop = threading.Event()
        self.rows = []
        self._ann_failed = False      # 公告发不出去只说一次，别刷屏
        self._acks = 0                # B 机回执计数（收尾时一句话交代）

    def stop(self):
        self._stop.set()

    # ---------------- 主流程 ----------------

    def run(self):
        try:
            self._run()
        except Exception as e:                       # noqa: BLE001
            # 出错也要把握手端口放掉：不放掉，同一个部署台里再跑一轮就会"绑不上"
            # 而静默退回老流程（见 make_announcer）。
            link = getattr(self, "_link", None)
            if link is not None:
                link.close()
                self._link = None
            self.on_line("推流自检出错：%s: %s" % (type(e).__name__, e))
            # 出错也要**收尾**：不收尾界面里那两个按钮就永远卡在"运行中"
            # （开始禁用、停止可用），只能关掉部署台重开。
            try:
                self.on_done(self.rows, "")
            except Exception:
                pass

    def _run(self):
        conf = pp.load(self.presets_path)
        seconds = float(self._seconds if self._seconds is not None
                        else conf["seconds"])
        settle = float(self._settle if self._settle is not None
                       else conf["settle_seconds"])
        presets = conf["presets"]
        base = dict(self.cfg.get("push") or {})
        link = make_announcer(self.cfg)
        self._link = link            # 出错路径上要放掉它（见 run()）
        port = int((self.cfg.get("sweep") or {}).get("port") or sweep_link.DEFAULT_PORT)
        self.on_line("候选 %d 条 ×（%.0fs 试推 + %.0fs 丢弃）≈ %.0f 分钟；"
                     "B 机跑 tools.stream_sweep —— **现在跑、或者先跑着等**都行"
                     % (len(presets), seconds, settle,
                        len(presets) * (seconds + settle + 3) / 60.0))
        if link is None:
            self.on_line("        （握手建不起来：push.host 没配、或 UDP %d 被占 —— "
                         "那就还是老规矩：B 机要在点开始后几秒内跑起来）" % port)
        else:
            link.start(len(presets))
            self.on_line("        握手：每段开始会公告到 B 机 UDP %d；B 机一回执这里就"
                         "显示「← B 机已跟上（第 n 段）」" % port)

        for i, p in enumerate(presets):
            if self._stop.is_set():
                self.on_line("已中断（剩余 %d 条没跑）" % (len(presets) - i))
                break
            push = pp.push_block(p, base)
            self.on_line("[%d/%d] %s —— 起推流…" % (i + 1, len(presets), p["name"]))
            t0 = time.time()
            # 公告要**在 ffmpeg 起来之前**发：B 机收到就去等流，而上一段的流刚好停了 ——
            # 这样 B 机等到的第一帧一定属于这一段（对齐是结构性的，不靠时间差凑）。
            if link is not None and not link.seg(i, len(presets), p, seconds, settle):
                if not self._ann_failed:
                    self._ann_failed = True
                    self.on_line("        （公告发不出去 —— 握手失效；B 机会退回老规矩）")
            stats, extra, proc_ok = self._run_once(push, seconds, settle)
            t1 = time.time()
            row = pp.row_a(p["name"], i, p, t0, t1, stats, extra=extra)
            if not proc_ok:
                row["ok"] = False
                row["why"] = "ffmpeg 没起来（参数非法？编码器不可用？）"
                row["speed"] = None
            self.rows.append(row)
            self.on_line("        → speed %s、实际 %s fps、丢帧 %s%s"
                         % (row.get("speed") if row.get("speed") is not None else "-",
                            row.get("out_fps") if row.get("out_fps") is not None else "-",
                            row.get("drop"), 
                            ("" if proc_ok else "（ffmpeg 起不来）")))
            for a in (link.poll_acks() if link is not None else ()):
                self._acks += 1
                self.on_line("        ← B 机已跟上（第 %s 段）"
                             % (int(a.get("seg") or 0) + 1))

        path = dcfg.ROOT / pp.A_REPORT
        try:
            path.write_text(json.dumps(
                {"meta": {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "seconds": seconds, "settle": settle,
                          "presets": [p["name"] for p in presets]},
                 "rows": self.rows}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            self.on_line("已写 %s（%d 条）" % (path, len(self.rows)))
        except Exception as e:                       # noqa: BLE001
            self.on_line("写 %s 失败：%s" % (path, e))
        if link is not None:
            if self._acks:
                self.on_line("握手：B 机跟上 %d/%d 段" % (self._acks, len(self.rows)))
            else:
                self.on_line("握手：**B 机一次回执都没有**（UDP %d）—— 它没在跑 "
                             "tools.stream_sweep、或防火墙挡了这个口。A 机侧这几个数"
                             "不受影响，但延迟那一半这轮拿不到。"
                             % int((self.cfg.get("sweep") or {}).get("port")
                                   or sweep_link.DEFAULT_PORT))
            link.done(len(presets))
            # 把 A 机这份报告**发给 B 机**，并等它回结论 —— 这是"两个开关"收尾的那一半：
            # B 机不用去猜路径、也不用人工拷 perf_push_A.json；结论会由 B 机算好后回传，
            # 部署台弹窗直接显示（合并表 + 建议写回的参数）。
            txt = ""
            try:
                txt = path.read_text(encoding="utf-8")
            except Exception:                    # noqa: BLE001
                pass
            if txt and link.send_blob("rep", txt):
                self.on_line("        已把 %s 发给 B 机；等它测完最后一段并出表…"
                             "（最多 %.0f 秒）" % (Path(path).name, RESULT_WAIT_S))
                res = link.wait_blob("res", RESULT_WAIT_S)
                if res:
                    self.on_line("        ↓ B 机测出的结果 ↓")
                    for ln in res.splitlines():
                        self.on_line("        " + ln)
                    try:
                        self.on_result(res)
                    except Exception as e:        # noqa: BLE001
                        self.on_line("        （结论显示失败：%s）" % e)
                else:
                    self.on_line("        没等到 B 机的结论 —— 它那边的命令是不是没加 "
                                 "`--auto`？或者它提前退出了（看它终端的输出）。")
            elif txt:
                self.on_line("        （报告发给 B 机失败 —— 它可能在旧版本，"
                             "按老办法自己去拷 %s）" % Path(path).name)
            link.close()
        self.on_done(self.rows, path)

    def _run_once(self, push, seconds, settle):
        """起一次 ffmpeg，跑 settle+seconds 秒，收 `-stats` 行。

        返回 (stats 列表, extra, 进程是否起来过)。**只收 stats，不读别的输出** ——
        ffmpeg 的 `-loglevel error` 保证正常时不刷屏，异常时那几行也会进日志。
        """
        cmd = cmd_for(push)
        self.on_line("        " + services.format_cmd(cmd)[:160] + " …")
        stats = []
        extra = {}
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(dcfg.ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
                creationflags=(0x08000000 if sys.platform == "win32" else 0))
        except Exception as e:                       # noqa: BLE001
            self.on_line("        起不来：%s: %s" % (type(e).__name__, e))
            return [], extra, False

        t0 = time.monotonic()
        cpu_t0 = _psutil_cpu_t0(proc.pid)
        gpu_seen = []
        try:
            while True:
                el = time.monotonic() - t0
                if self._stop.is_set() or el >= settle + seconds:
                    break
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.02)
                    continue
                st = pp.parse_ffmpeg_stats(line)
                if st:
                    stats.append(st)
                    if len(gpu_seen) < 3 and el >= settle:
                        g = _gpu_enc_pct()
                        if g:
                            gpu_seen.append(g)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        cpu = _psutil_cpu_pct(proc.pid, t0, cpu_t0)
        if cpu is not None:
            extra["cpu_pct"] = round(cpu, 1)
        if gpu_seen:
            extra["gpu_enc_pct"] = sum(g["gpu_enc_pct"] for g in gpu_seen) / len(gpu_seen)
            extra["gpu_pct"] = sum(g["gpu_pct"] for g in gpu_seen) / len(gpu_seen)
        return stats, extra, True


def main():
    """命令行跑法：python -m deploy.push_sweep [--seconds N] [--presets 路径]"""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--presets", default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--settle", type=float, default=None)
    args = ap.parse_args()

    cfg = dcfg.read()
    rows, path = [], [None]

    def done(r, p):
        rows.extend(r)
        path[0] = p

    sw = PushSweep(cfg, args.presets, on_line=lambda t: print(t, flush=True),
                   on_done=done, seconds=args.seconds, settle=args.settle)
    sw.run()                      # 命令行就直接同步跑
    if rows:
        # A 机侧单独看：只列它负责的那几列。**不套 rank/render** —— 那两样要延迟
        # 数，这里没有；拿 0 当真会把"最优"判成错的（评分口径见 push_presets）。
        print("\n推流自检 · A 机侧（延迟那一半在 B 机）")
        print("-" * 92)
        print("%-16s %5s %6s %6s %8s %8s %8s %7s" %
              ("段", "fps", "码率", "GOP", "speed", "实际fps", "丢帧", "CPU%"))
        for r in rows:
            def f(v, w, p=1):
                return ("%*.1f" % (w, v)) if isinstance(v, (int, float)) else ("%*s" % (w, "-"))
            print("%-16s %5s %6s %6s %8s %s %s %s%s"
                  % (r["name"], r.get("fps"), r.get("bitrate"), r.get("gop"),
                     ("%.2f" % r["speed"]) if r.get("speed") is not None else "-",
                     f(r.get("out_fps"), 8), f(r.get("drop"), 8, 0),
                     f(r.get("cpu_pct"), 7),
                     "" if r.get("ok", True) else "   ← %s" % r.get("why")))
        print("-" * 92)
        print("挑出 speed ≥ 1.00 的那几条，在 B 机那边看它们的延迟；"
              "把 %s 拷到 B 机再跑：\n"
              "    python -m tools.stream_sweep --merge %s"
              % (Path(path[0]).name if path[0] else pp.A_REPORT, pp.A_REPORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
