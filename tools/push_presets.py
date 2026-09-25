"""推流自检的**共用核心**：候选清单 / 命令行映射 / ffmpeg 指标解析 / 两机对账 / 评分。

**为什么单独一个纯模块**（没有任何 IO、Qt、网络）：
    推流自检要两台机器合作 —— A 机（部署台）负责换参数、起推流、采 A 侧指标；
    B 机负责量端到端延迟。真正容易写错的是**中间那套账**：候选怎么变成命令行、
    ffmpeg 那行 `-stats` 怎么解析、两边的段怎么对齐、谁淘汰谁最优。这些全是纯计算，
    放这里就能在**一台机器上被自检完整覆盖**（见 tools/selftest_push_presets.py），
    而不是等两边都跑起来才发现"表对不上"。

**架构（2026-09-25 定的，方案 A）**：两台机器不新开任何控制通道 ——
    · 候选清单 `config/push_presets.json` 跟着仓库走，**两边是同一份**；
    · A 机按清单**顺序**逐条试推（每条跑 seconds 秒），写 `perf_push_A.json`；
    · B 机按**同一顺序**逐条测量，写 `perf_push_B.json`；
    · 两边都记**起止墙钟**（双机已对时，差 ~1ms），合并时用它核对（见 `merge`）。
    谁也不用去驱动谁，安全面不扩大。

**指标口径**（哪些数能信，都在注释里写死）：
    A 侧：`speed`（<1.00 = A 机跟不上，直接淘汰）、实际 fps、丢帧、实际码率、CPU/GPU。
    B 侧：延迟 p50/p95（**只在探针几何判据通过时才记**，见 probe_codec.Verdict）、
          抖动、recv/目标 fps（丢包代理）、read() 耗时（判延迟在 A 机还是 B 机内部）。
"""

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRESETS_FILE = ROOT / "config" / "push_presets.json"
#: A/B 两侧的落盘文件名（各自机器上，合并时把 A 那份拷过来即可）
A_REPORT = "perf_push_A.json"
B_REPORT = "perf_push_B.json"

#: 每条候选默认跑多久；前 settle 秒丢弃（推流刚起、解码器刚锁、A 机还没进稳态）
SECONDS = 20.0
SETTLE = 4.0
#: 判"这段能不能用"的线（**故意写死在这里**，免得每次跑完临时改口径）
SPEED_MIN = 1.00          # ffmpeg 跟不上就一定会丢帧 / 涨延迟
RECV_RATIO_MIN = 0.98     # 收到帧数 / 目标帧率
#: 两机段对齐：重叠时长占 A 段的比例低于它 → 只警告（B 机启动晚一点很正常）
OVERLAP_WARN = 0.5

#: 候选里允许改的键（其余一律从当前 deploy.json 的 push 段继承 —— 比如
#: host/port/ffmpeg 路径/采集方式/编码器，这些不该被"自检"乱动）
PRESET_KEYS = ("fps", "bitrate", "gop", "passthrough", "low_latency",
               "width", "height", "capture", "encoder", "scale_mode")


# ══════════════════════════════════════════
# 候选清单
# ══════════════════════════════════════════

def default_presets():
    """默认候选：fps{60,120,144} × 码率{6M,12M,20M} × GOP{30,60} = 18 套。

    为什么这几个值：
      · fps 取 60（对齐仿真环境的固定步长）、120、144（屏幕真能出这么多帧时才有意义；
        ddagrab 的 framerate 超过桌面刷新率也拿不到更多帧）；
      · 码率 6M 是"1366x768@50 够用"的老默认，20M 是局域网里的上限档；
        **注意 `-bufsize` 在部署台里硬编码是 1M**，所以码率越高 VBV 缓冲越小：
        6M → 167ms，20M → 50ms（这一项对延迟的影响比码率本身大得多）；
      · GOP 30 / 60：丢一个包要等下一个关键帧，GOP 越短恢复越快（代价是码率略涨）。
    """
    out = []
    for fps in (60, 120, 144):
        for br in ("6M", "12M", "20M"):
            for gop in (30, 60):
                out.append({"fps": fps, "bitrate": br, "gop": gop,
                            "passthrough": True})
    for p in out:
        p["name"] = name_of(p)
    return {"seconds": SECONDS, "settle_seconds": SETTLE, "presets": out}


def name_of(p):
    """候选 → 段名（表里、日志里都用它，**必须稳定**：两机按名字对账）。

    采集链不同（`scale_mode`）也是不同的候选 —— 名字必须能区分，否则两机对账
    会把"CPU 缩放"和"GPU 缩放"混成同一条，表就废了。默认那档不加后缀，
    这样**既有候选的名字一个都不变**（历史表还能对上）。
    """
    base = "%sfps-%s-g%d" % (p.get("fps"), p.get("bitrate"), int(p.get("gop", 0)))
    mode = str(p.get("scale_mode") or "cpu")
    return base if mode == "cpu" else "%s-%s" % (base, mode)


def load(path=None):
    """读候选清单；没有/坏了就返回**默认清单**（并说一声，别静默）。

    返回 `{"seconds":…, "settle_seconds":…, "presets": [ … ]}`，每条候选都补齐
    `name`（缺了就用 name_of 生成）—— 名字是对账用的，不能空着。
    """
    p = Path(path) if path else PRESETS_FILE
    d = None
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print("[push_presets] %s 读不出来（%s），改用默认清单" % (p, e))
    if not isinstance(d, dict) or not d.get("presets"):
        return default_presets()
    out = {"seconds": float(d.get("seconds", SECONDS)),
           "settle_seconds": float(d.get("settle_seconds", SETTLE)),
           "presets": []}
    for i, raw in enumerate(d["presets"]):
        if not isinstance(raw, dict):
            continue
        q = {k: v for k, v in raw.items() if k in PRESET_KEYS}
        q["name"] = str(raw.get("name") or name_of(q) or ("p%d" % (i + 1)))
        out["presets"].append(q)
    return out or default_presets()


def push_block(preset, base):
    """候选 + 当前 push 段 → 新 push 段（**只覆盖候选里写了的键**）。

    `base` 就是 `config/deploy.json` 里那份 push：host/port/ffmpeg/采集方式/编码器
    这些"这台机器的属性"全部继承，自检只动 fps/码率/GOP/passthrough 这几个。
    """
    out = dict(base or {})
    for k in PRESET_KEYS:
        if k in preset and preset[k] is not None:
            out[k] = preset[k]
    return out


# ══════════════════════════════════════════
# ffmpeg 的 -stats 行
# ══════════════════════════════════════════

#: `frame= 1234 fps=143 q=23.0 size= 12345kB time=00:00:08.60
#:  bitrate=11756.5kbits/s speed=0.99x drop=3 dup=0`
_STAT_PATTERNS = {
    "frame": r"frame=\s*(\d+)",
    "fps": r"fps=\s*([\d.]+)",
    "bitrate_kbps": r"bitrate=\s*([\d.]+)\s*kbits/s",
    "speed": r"speed=\s*([\d.]+)\s*x",
    "drop": r"drop=\s*(\d+)",
    "dup": r"dup=\s*(\d+)",
    "size_kb": r"size=\s*(\d+)\s*kB",
}


def parse_ffmpeg_stats(line):
    """ffmpeg 的 `-stats` 行 → dict（缺的键不出现）。认不出这行就返回 {}。

    **为什么必须解析它**：A 机"扛不扛得住"只有 ffmpeg 自己知道 —— `speed` < 1.00
    就是它处理不过来（采集/编码跟不上），此时再怎么调 B 机都没用。部署台原先的
    命令带了 `-nostats`，那一行根本不打；自检会把 `-stats` 加上（见 deploy/app.py）。
    """
    if not line or "=" not in line:
        return {}
    out = {}
    for k, pat in _STAT_PATTERNS.items():
        m = re.search(pat, line)
        if not m:
            continue
        try:
            out[k] = float(m.group(1))
        except ValueError:
            continue
    return out


def take_last(stats_list):
    """一串统计行 → 每一列取**最后一个**（steady state 用尾部，不用平均）。

    为什么不用平均：推流刚起来那几百毫秒必然慢（编码器建会话、解码器锁流），
    把它们平均进来会低估 A 机的能力，也可能反过来掩盖后期的掉速。
    """
    out = {}
    for d in stats_list or []:
        for k, v in d.items():
            out[k] = v
    return out


# ══════════════════════════════════════════
# 两侧的记账（统一行结构，方便合并与渲染）
# ══════════════════════════════════════════

def row_a(name, idx, preset, t0, t1, stats, extra=None):
    """A 机一段的记账 → dict（`t0/t1` 是墙钟秒）。"""
    st = take_last(stats)
    target = float(preset.get("fps") or 0) or None
    r = {"side": "A", "idx": int(idx), "name": str(name),
         "fps": preset.get("fps"), "bitrate": preset.get("bitrate"),
         "gop": preset.get("gop"), "passthrough": preset.get("passthrough"),
         "t_start": float(t0), "t_end": float(t1),
         "wall": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
         "speed": st.get("speed"), "out_fps": st.get("fps"),
         "drop": st.get("drop"), "dup": st.get("dup"),
         "bitrate_kbps": st.get("bitrate_kbps"),
         "frames": st.get("frame"),
         "target_fps": target,
         "out_ratio": (st["fps"] / target) if (st.get("fps") and target) else None,
         "ok": True, "why": ""}
    if extra:
        r.update(extra)
    return r


def row_b(name, idx, preset, t0, t1, meas, extra=None):
    """B 机一段的记账 → dict。`meas` 见 stream_sweep.measure_segment 的返回。"""
    target = float(preset.get("fps") or 0) or None
    r = {"side": "B", "idx": int(idx), "name": str(name),
         "t_start": float(t0), "t_end": float(t1),
         "wall": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
         "recv_fps": meas.get("recv_fps"), "proc_fps": meas.get("proc_fps"),
         "lat_p50": meas.get("lat_p50"), "lat_p95": meas.get("lat_p95"),
         "lat_max": meas.get("lat_max"), "lat_n": meas.get("lat_n"),
         "jitter": meas.get("jitter"), "read_p50": meas.get("read_p50"),
         "gap_p50": meas.get("gap_p50"), "frames": meas.get("frames"),
         "probe_mono": meas.get("probe_mono"),
         "probe_why": meas.get("probe_why"),
         "target_fps": target,
         "recv_ratio": (meas["recv_fps"] / target
                        if (meas.get("recv_fps") and target) else None),
         "ok": True, "why": ""}
    if extra:
        r.update(extra)
    return r


# ══════════════════════════════════════════
# 对账（合并两侧）
# ══════════════════════════════════════════

def overlap_ratio(a, b):
    """两段墙钟窗口的重叠时长 / A 段时长（0~1）。"""
    lo = max(a["t_start"], b["t_start"])
    hi = min(a["t_end"], b["t_end"])
    span = max(1e-6, a["t_end"] - a["t_start"])
    return max(0.0, hi - lo) / span


def merge(rows_a, rows_b, tol_s=2.0):
    """两侧的记账 → 一张表。**按顺序对齐，用墙钟核对**。

    为什么按顺序对齐而不是纯靠墙钟：两边走的是同一份清单、同一个顺序，顺序是最
    强的对齐依据；墙钟拿来做**校验** —— 重叠太少或起止对不上就说明有一边跑歪了
    （比如 B 机中途重开了预览、或 A 机某条没起来），这时要**说出来**，不能闷头
    出一张看着很整齐的表。`tol_s` 是允许的起止偏差。
    """
    rows = []
    n = min(len(rows_a), len(rows_b))
    for i in range(n):
        a, b = rows_a[i], rows_b[i]
        r = {"idx": i, "name": a["name"],
             "fps": a.get("fps"), "bitrate": a.get("bitrate"), "gop": a.get("gop"),
             "passthrough": a.get("passthrough"),
             "speed": a.get("speed"), "out_fps": a.get("out_fps"),
             "out_ratio": a.get("out_ratio"), "drop": a.get("drop"),
             "a_bitrate_kbps": a.get("bitrate_kbps"),
             "cpu_pct": a.get("cpu_pct"), "gpu_enc_pct": a.get("gpu_enc_pct"),
             "recv_fps": b.get("recv_fps"), "recv_ratio": b.get("recv_ratio"),
             "proc_fps": b.get("proc_fps"),
             "lat_p50": b.get("lat_p50"), "lat_p95": b.get("lat_p95"),
             "lat_max": b.get("lat_max"), "jitter": b.get("jitter"),
             "read_p50": b.get("read_p50"),
             "probe_mono": b.get("probe_mono"), "probe_why": b.get("probe_why"),
             "t_start": a.get("t_start"), "t_end": a.get("t_end"),
             "warn": ""}
        ov = overlap_ratio(a, b)
        drift = abs(a["t_start"] - b["t_start"])
        if a["name"] != b["name"]:
            r["warn"] = "两侧顺序不一致（A=%s / B=%s）" % (a["name"], b["name"])
        elif ov < OVERLAP_WARN or drift > tol_s + max(
                a["t_end"] - a["t_start"], 0.0) * 0.5:
            r["warn"] = ("墙钟对不上（重叠 %.0f%%、起点差 %.1fs）—— "
                         "这一行只能按顺序信，别当两机同时测的" % (ov * 100, drift))
        rows.append(r)
    if len(rows_a) != len(rows_b):
        note = ("两侧段数不同（A %d / B %d）—— 多出来的那几条没有对象，"
                "只能各自看" % (len(rows_a), len(rows_b)))
        rows.append({"idx": -1, "name": "(段数不一致)", "warn": note})
    return rows


# ══════════════════════════════════════════
# 评分
# ══════════════════════════════════════════

def classify(row):
    """这一行能不能用 + 为什么 → (可用?, 原因)。**先淘汰再比大小。**

    淘汰规则（顺序即优先级）：
      ① A 机是瓶颈（`speed` < 1.00）—— 这是硬件跟不上，改 B 机没用；
      ② 探针几何判据没过 —— 这一段的延迟数是假的（`probe_mono` False）。
         注意：几何问题**不属于某一条候选**，整轮都该是坏的；所以真跑的时候
         `stream_sweep` 会在一开始先拦（见那边的 precheck），这里只是兜底；
      ③ 丢帧太多（recv/目标 < 0.98）—— 链路或 B 机跟不上，这一段的延迟也不可信。
    """
    sp = row.get("speed")
    if sp is not None and sp < SPEED_MIN:
        return False, "A 机跟不上（speed %.2f < %.2f）" % (sp, SPEED_MIN)
    if row.get("probe_mono") is False:
        return False, "探针几何判据没过，延迟数不可信"
    rr = row.get("recv_ratio")
    if rr is not None and rr < RECV_RATIO_MIN:
        return False, "丢帧太多（收到 %.1f%%，< %.0f%%）" % (rr * 100,
                                                       RECV_RATIO_MIN * 100)
    if row.get("lat_p95") is None:
        return False, "没量到延迟"
    return True, ""


def rank(rows):
    """→ (排好序的 rows, 最优那条或 None, 结论一句话)。

    排序口径（写死，别每次跑完临时改）：
        **先看 p95**（卡顿才是手感杀手），再看 p50，再看码率（省带宽），最后看 CPU。
        并列时**低帧率优先**：同样的 p95，帧率低的那套对 A 机更轻、对 B 机压力更小。
    """
    ok, bad = [], []
    for r in rows:
        good, why = classify(r)
        r["pass"] = good
        r["fail"] = why
        (ok if good else bad).append(r)

    def key(r):
        return (r.get("lat_p95", 1e9), r.get("lat_p50", 1e9),
                r.get("a_bitrate_kbps") or 1e9, r.get("cpu_pct") or 0.0,
                r.get("fps") or 0)

    ok.sort(key=key)
    best = ok[0] if ok else None
    if best is None:
        return ok + bad, None, ("没有一条通过淘汰规则（%d 条全军覆没）—— "
                                "先解决 A 机 speed<1.00 或探针几何" % len(bad))
    why = ("最优：%s（p95 %.0fms / p50 %.0fms / 码率 %.1fMbps）"
           % (best["name"], best["lat_p95"], best.get("lat_p50") or -1,
              (best.get("a_bitrate_kbps") or 0) / 1000.0))
    return ok + bad, best, why


def render(rows, best, title="推流自检"):
    """→ 一张给人看的表（终端/日志都能读）。"""
    out = ["", "=" * 108, "%s   （p95 排序；标 ✗ 的已被淘汰）" % title, "=" * 108]
    out.append("%-16s %5s %6s %5s %7s %7s %6s %7s %9s %9s %9s  %s"
               % ("段", "fps", "码率", "GOP", "speed", "实际fps", "丢帧",
                  "收到fps", "延迟p50", "延迟p95", "抖动", "判定"))
    for r in rows:
        if r.get("idx", -1) < 0:
            out.append("  ⚠ " + (r.get("warn") or ""))
            continue
        mark = "✓" if r.get("pass") else "✗"

        def f(v, w, p=1, dash="-"):
            return ("%*.1f" % (w, v)) if isinstance(v, (int, float)) else \
                ("%*s" % (w, dash))

        out.append("%-16s %5s %6s %5s %7s %s %s %s %9s %9s %9s  %s %s"
                   % (r["name"], r.get("fps"), r.get("bitrate"), r.get("gop"),
                      ("%.2f" % r["speed"]) if r.get("speed") is not None else "-",
                      f(r.get("out_fps"), 7), f(r.get("drop"), 6, 0),
                      f(r.get("recv_fps"), 7),
                      f(r.get("lat_p50"), 9, 0), f(r.get("lat_p95"), 9, 0),
                      f(r.get("jitter"), 9, 0), mark,
                      (("—— " + r["fail"]) if not r.get("pass") else
                       (("⚠ " + r["warn"]) if r.get("warn") else ""))))
    out.append("=" * 108)
    if best:
        out.append("结论：" + render_best_line(best))
    out.append("说明：`speed` 是 ffmpeg 的处理速度（<1.00 = A 机跟不上，一定丢帧）；")
    out.append("      `延迟 p50/p95` 只在**探针几何判据通过**时才记（否则整轮该先修几何）；")
    out.append("      `收到fps/目标fps` < 98% 说明链路或 B 机跟不上，那一段的延迟也别信。")
    return "\n".join(out)


def render_best_line(best):
    return ("%s —— fps=%s 码率=%s GOP=%s passthrough=%s\n"
            "      p95 %.0fms / p50 %.0fms / 抖动 %.0fms；A 机 speed %.2f、实际 %.1ffps、"
            "丢帧 %s；B 机收到 %.1ffps；实测码率 %.1fMbps"
            % (best["name"], best.get("fps"), best.get("bitrate"), best.get("gop"),
               best.get("passthrough"), best.get("lat_p95") or -1,
               best.get("lat_p50") or -1, best.get("jitter") or -1,
               best.get("speed") or -1, best.get("out_fps") or -1,
               best.get("drop"), best.get("recv_fps") or -1,
               (best.get("a_bitrate_kbps") or 0) / 1000.0))


def to_deploy_push(best, base):
    """最优那条 → 可以直接写回 `config/deploy.json` 的 push 段（部署台照它跑）。"""
    return push_block({k: best.get(k) for k in PRESET_KEYS if best.get(k) is not None},
                      base)
