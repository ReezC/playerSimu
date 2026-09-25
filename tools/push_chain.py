"""采集链快查：**一条命令、几秒钟、不用 B 机**，判断某条推流链在这台机器上起不起得来。

**为什么要有它**（2026-09-26）：一轮「推流自检」= 18 段 × 24s ≈ 9 分钟，还得 B 机
同时跑 `stream_sweep` 才有意义。可"GPU 缩放那条链到底能不能用"这个问题**跟延迟测量
毫无关系** —— 它要么起得来，要么 ffmpeg 立刻退出并打一句报错。为这一句话等 9 分钟、
还占着 B 机，太贵了（实测就是这么卡住的：表上只有"没量到延迟"，原因得另外找）。

**它做的事**（每条候选 3~5 秒）：
    · 用部署台**同一份实现**拼命令（`deploy.services.build_cmd`），只把输出从
      `udp://…` 换成 `-f null -` —— **不碰网络、不打扰 B 机、不需要有人收流**
      （往没人听的 UDP 口推，Windows 上还可能吃到 ICMP 端口不可达而提前报错，
       `-f null` 正好绕开这类假故障）；
    · `-nostats` → `-stats`（要看得见 fps/speed），`-loglevel error` → `warning`
      （配置阶段的线索常常在 warning 级，`error` 会把它吞掉）；
    · 起几秒，把 ffmpeg 的输出**原样打出来** —— 起不来的原因就在那几行里；
    · 先查 ffmpeg 有没有这条链要的滤镜（GPU 那条要 `scale_cuda`）：
      **这是第一个该排除的**，缺了就不必再往下查。

跑法（**在 A 机上跑**：B 机没有采集设备，ddagrab/cuda 都不在那边）：
    python -m tools.push_chain                     # 清单里每条各 3 秒
    python -m tools.push_chain --mode cuda         # 只看 GPU 那条
    python -m tools.push_chain --fps 144 --bitrate 20M --gop 30 --mode cuda
    python -m tools.push_chain --cmd-only          # 只打印命令（拿去手跑 / 贴给别人）

**它不能替代「推流自检」**：这里量不到端到端延迟（那要 B 机 + 探针），它只回答
"这条链在这台机器上跑不跑得起来、跑起来大概多少 fps"。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy import config as dcfg, services                      # noqa: E402
from tools import push_presets as pp                             # noqa: E402

#: 每条候选跑多久。3 秒够 ffmpeg 走到"稳态"或"立刻死"，再长就是浪费。
SECONDS = 3.0

#: 各条链要的滤镜（缺了必然起不来，而且报错很直白：No such filter: 'xxx'）
CHAIN_FILTERS = {"cuda": ("scale_cuda",), "cpu": ()}


def null_output_argv(argv, seconds=SECONDS):
    """部署台那条命令 → 快查用的命令（**纯函数，好测**）。

    改三处，都是为了让"起不起得来"这件事几秒钟内唯一地暴露出来：
      ① 输出端 `… -f mpegts udp://host:port?…` → `-t <seconds> -f null -`
         （不占 UDP 口、不需要接收端；`-t` 让它自己停，不用人去 Ctrl+C）；
      ② `-nostats` → `-stats`（看得见 fps/speed，也就知道"起来了但有多快"）；
      ③ `-loglevel error` → `warning`（配置阶段的线索常在 warning 级）。
    """
    args = list(argv)
    for i, a in enumerate(args):
        if a == "-nostats":
            args[i] = "-stats"
        elif a == "-loglevel":
            args[i + 1] = "warning"
    cut = None
    for i, a in enumerate(args):
        if "udp://" in a or "tcp://" in a:
            cut = i
            # 把紧邻的 `-f mpegts` 一起去掉（保留它再写 `-f null` 会自相矛盾）
            if i >= 2 and args[i - 2] == "-f":
                cut = i - 2
            break
    if cut is None:                          # 命令里没有输出端（不该发生）→ 原样返回
        return args
    return args[:cut] + ["-t", "%.3f" % float(seconds), "-f", "null", "-"]


def have_filters(ff, names):
    """这台机器的 ffmpeg 有没有这几个滤镜 → (查出结果或 None, 缺的)。

    **为什么要第三种状态（None = 没查成）**：查不成时两种做法都是错的 ——
    说"齐"是**假通过**（实测本机 ffmpeg 不在 PATH 上就会这样，屏幕上会写"滤镜齐"，
    而根本没查），说"缺"又会拦掉本来能跑的东西。所以明说"没确认"，让人去看试跑的报错。
    """
    if not names:
        return [], []
    try:
        r = subprocess.run([ff, "-hide_banner", "-filters"],
                           capture_output=True, text=True, timeout=20,
                           encoding="utf-8", errors="replace")
        blob = (r.stdout or "") + (r.stderr or "")
    except Exception as e:                   # noqa: BLE001
        print("[chain] 查不了滤镜（%s）—— **没确认**，跳过这一步，直接试跑" % e)
        return None, []
    return [n for n in names if n in blob], [n for n in names if n not in blob]


#: 报错行里哪几句是**根因**（挑最早出现的那句；别拿 "Terminating thread…" 这种尾巴
#: 或 "Nothing was written into output file" 这种后果当原因）
_REASON_HINTS = ("failed to", "no such filter", "cannot load", "error configuring",
                 "not implemented", "unknown encoder", "invalid argument",
                 "impossible to convert", "device not found", "no capable devices")


def pick_reason(tail):
    """报错行 → 最该看的那一句（纯函数）；挑不到就给最后一行。"""
    for s in tail or []:
        lo = s.lower()
        if any(h in lo for h in _REASON_HINTS):
            return s
    return (tail or [""])[-1]


def verdict(stats, tail, ran_s, seconds):
    """跑完一条之后的判定 → (起得来?, 一句话)。**纯函数 —— 这段决定工具会不会骗人。**

    两条**都**满足才算"起得来"：
      · **输出过帧**（`frame=` > 0）；只看"有没有 stats 行"会被骗 —— ffmpeg 退出前
        照样打一行 `frame=0 fps=0.0 … speed=N/A`。实测 2026-09-26：GPU 那条链根本没
        起来，我的第一版因此报"起得来：实际 -1.0 fps"、结论"采集链没问题" ✗。
        （和上一轮那张错位表是同一类错：**拿一句汇总当结论**。）
      · **真的跑满了时长**（≥90%）—— 立刻退出说明在配置阶段就失败了。
    """
    last = pp.take_last(stats)
    try:
        frames = float(last.get("frame") or 0)
    except (TypeError, ValueError):
        frames = 0.0
    if frames > 0 and ran_s >= float(seconds) * 0.9:
        return True, ("起得来：编码 %.0f 帧、实际 %.1f fps、speed %s"
                      % (frames, last.get("fps") or -1,
                         ("%.3f" % last["speed"])
                         if last.get("speed") is not None else "-"))
    why = pick_reason(tail) or ("只跑了 %.1f 秒、输出 %.0f 帧（一句报错都没有）"
                                % (ran_s, frames))
    return False, "起不来：%s" % why


def run_one(name, cfg, seconds, ff):
    """跑一条候选几秒 → (起得来?, 一句话结论)。ffmpeg 的输出原样转出来。"""
    cmd = null_output_argv(services.build_cmd("push", cfg), seconds)
    print("\n" + "-" * 92)
    print("[chain] %s" % name)
    print("[chain] " + services.format_cmd(cmd)[:220] + " …")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(dcfg.ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
            creationflags=(0x08000000 if sys.platform == "win32" else 0))
    except Exception as e:                   # noqa: BLE001
        print("[chain] 起不来：%s: %s" % (type(e).__name__, e))
        return False, "连进程都没起来（%s）" % e

    stats, tail = [], []
    try:
        while True:
            el = time.time() - t0
            if el > seconds + 8:             # 兜底：绝不在这里卡住等人
                break
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
                continue
            print("      " + line.rstrip()[:200])
            st = pp.parse_ffmpeg_stats(line)
            if st:
                stats.append(st)
            elif line.strip():
                tail.append(line.strip())
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:                    # noqa: BLE001
            try:
                proc.kill()
            except Exception:                # noqa: BLE001
                pass

    return verdict(stats, tail, time.time() - t0, seconds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=SECONDS,
                    help="每条试跑几秒（默认 %.0f）" % SECONDS)
    ap.add_argument("--mode", choices=("cpu", "cuda", "all"), default="all",
                    help="只查某种采集链（默认 all = 清单里每条）")
    ap.add_argument("--presets", default=None, help="候选清单（默认 config/push_presets.json）")
    ap.add_argument("--fps", type=int, default=None, help="临时试一组参数（配合 --cmd-only）")
    ap.add_argument("--bitrate", default=None)
    ap.add_argument("--gop", type=int, default=None)
    ap.add_argument("--cmd-only", action="store_true", dest="cmd_only",
                    help="只打印命令，不真跑（拿去手跑 / 贴给别人）")
    args = ap.parse_args()

    cfg0 = dcfg.read()
    base = dict(cfg0.get("push") or dcfg.DEFAULTS["push"])

    if args.presets:
        # 指定了清单 → 按清单来（`--mode` 只是从里面筛）
        rows = pp.load(args.presets)["presets"]
        if args.mode != "all":
            rows = [p for p in rows if str(p.get("scale_mode") or "cpu") == args.mode]
    elif args.fps or args.bitrate or args.gop or args.mode in ("cpu", "cuda"):
        # 明确点了某种链（或给了参数）→ 用一组临时候选，不必碰清单文件
        modes = ["cpu", "cuda"] if args.mode == "all" else [args.mode]
        rows = []
        for m in modes:
            p = dict(fps=args.fps or base.get("fps") or 60,
                     bitrate=args.bitrate or base.get("bitrate") or "6M",
                     gop=args.gop or base.get("gop") or 30, scale_mode=m)
            p["name"] = pp.name_of(p)
            rows.append(p)
    else:
        rows = pp.load(None)["presets"]

    ff = base.get("ffmpeg") or "ffmpeg"
    print("[chain] 采集链快查：%d 条 × %.0fs（**不用 B 机、不碰网络、不量延迟**）"
          % (len(rows), args.seconds))
    print("[chain] ffmpeg = %s" % ff)
    for m in sorted({str(p.get("scale_mode") or "cpu") for p in rows}):
        need = CHAIN_FILTERS.get(m, ())
        if not need:
            continue
        got, miss = have_filters(ff, need)
        if miss:
            print("[chain] ✗ 这条链要的滤镜这台 ffmpeg **没有**：%s" % "、".join(miss))
            print("        → %s 那档不用试了，起不来；要用就得换带它的构建"
                  "（Gyan 版 ffmpeg 的 full 包才有 scale_cuda）" % m)
        elif got is None:
            print("[chain] %s 链要的滤镜**没能确认**（见上一句）—— 试跑时自己看报错"
                  % m)
        else:
            print("[chain] %s 链要的滤镜齐：%s" % (m, "、".join(need)))

    bad = []
    for p in rows:
        push = pp.push_block(p, base)
        if args.cmd_only:
            print("\n" + "-" * 92)
            print("[chain] %s" % p["name"])
            print(" ".join(null_output_argv(services.build_cmd("push", push),
                                            args.seconds)))
            continue
        ok, why = run_one(p["name"], push, args.seconds, ff)
        print("[chain] → %s" % why)
        if not ok:
            bad.append((p["name"], why))

    print("\n" + "=" * 92)
    if args.cmd_only:
        print("（--cmd-only：上面每条都拿去 A 机手跑即可；带 -f null 不占端口）")
        return 0
    if bad:
        print("起不来的 %d 条：" % len(bad))
        for n, w in bad:
            print("  · %-20s %s" % (n, w))
        print("结论：这几条在这台机器上用不了（原因就在上面那行 ffmpeg 输出里）；"
              "能起来的那些，再用「推流自检」量延迟。")
        return 1
    print("结论：%d 条全部起得来 —— 采集链没问题，延迟/瓶颈再去跑「推流自检」。" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
