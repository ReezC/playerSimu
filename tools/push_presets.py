"""推流自检的**共用核心**：候选清单 / 命令行映射 / ffmpeg 指标解析 / 两机对账 / 评分。

**为什么单独一个纯模块**（没有任何 IO、Qt、网络）：
    推流自检要两台机器合作 —— A 机（部署台）负责换参数、起推流、采 A 侧指标；
    B 机负责量端到端延迟。真正容易写错的是**中间那套账**：候选怎么变成命令行、
    ffmpeg 那行 `-stats` 怎么解析、两边的段怎么对齐、谁淘汰谁最优。这些全是纯计算，
    放这里就能在**一台机器上被自检完整覆盖**（见 tools/selftest_push_presets.py），
    而不是等两边都跑起来才发现"表对不上"。

**架构（2026-09-25 定的，方案 A；2026-09-25 晚改成握手）**：
    · 候选清单 `config/push_presets.json` 跟着仓库走（两边**不再必须**同一份 —— 参数
      由 A 机的握手公告直接带给 B 机，见 tools/sweep_link.py）；
    · A 机按清单逐条试推（每条跑 seconds 秒），**每段开始前公告**给 B 机，写 `perf_push_A.json`；
    · B 机**按公告对齐**逐段测量（晚起、中途重启都不影响），写 `perf_push_B.json`；
    · 合并**按段号配对**（见 `merge` 的说明：按数组位置配会整表错位一行）；
    · A 机跑完把报告发回 B（同一条通道），B 出表后把结论回传给 A 弹出来。

**指标口径**（哪些数能信，都在注释里写死）：
    A 侧：`speed`（<1.00 = A 机跟不上，直接淘汰）、实际 fps、丢帧、实际码率、CPU/GPU。
    B 侧：延迟 p50/p95（**只在探针几何判据通过时才记**，见 probe_codec.Verdict）、
          抖动、**收到 fps / A 实际发出 fps**（真正的链路丢帧代理）、read() 耗时。
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


def _pair(a, b):
    """一条合并行：A 侧的参数与指标 + B 侧的延迟指标。"""
    return {"idx": int(a.get("idx", -1)), "name": a.get("name"),
            "fps": a.get("fps"), "bitrate": a.get("bitrate"), "gop": a.get("gop"),
            "passthrough": a.get("passthrough"),
            "speed": a.get("speed"), "out_fps": a.get("out_fps"),
            "out_ratio": a.get("out_ratio"), "drop": a.get("drop"),
            "a_bitrate_kbps": a.get("bitrate_kbps"),
            "cpu_pct": a.get("cpu_pct"), "gpu_enc_pct": a.get("gpu_enc_pct"),
            "recv_fps": (b or {}).get("recv_fps"),
            "recv_ratio": (b or {}).get("recv_ratio"),
            "proc_fps": (b or {}).get("proc_fps"),
            "lat_p50": (b or {}).get("lat_p50"), "lat_p95": (b or {}).get("lat_p95"),
            "lat_max": (b or {}).get("lat_max"), "jitter": (b or {}).get("jitter"),
            "read_p50": (b or {}).get("read_p50"),
            "probe_mono": (b or {}).get("probe_mono"),
            "probe_why": (b or {}).get("probe_why"),
            "t_start": a.get("t_start"), "t_end": a.get("t_end"),
            # A 侧自己记的成败（"ffmpeg 起不来"就是靠它说出来的）
            "a_ok": a.get("ok"), "a_why": a.get("why") or "",
            "warn": "", "only": ""}


def merge(rows_a, rows_b, tol_s=2.0):
    """两侧的记账 → 一张表。**按段号（idx）配对**，墙钟只用来核对。

    **为什么不能按数组位置配**（原来的写法，2026-09-26 实测出错）：B 机晚起、
    或某一段没等到流时，两边数组的**长度和起点都不一样** —— 按位置配会把 B 的每一行
    都往前错一格：日志里 `60fps-20M-g60` 显示的是 `120fps-6M-g30` 的数（92.3fps/77ms），
    而表看着毫无破绽 ✗。段号是握手公告直接带过来的（`tools/sweep_link.py` 的 `i`），
    两边一定一致，还能直接看出"谁缺了哪一段"。

    配对不上的行**照样列出来**（不丢）：A 侧多出来的通常是"ffmpeg 起不来"那几条 ——
    那本身就是最有用的信息（比如 GPU 缩放那档在这台机器上支不支持）。
    墙钟仍然核对，但只影响提示文字，不影响配对。
    """
    by_b = {}
    for b in rows_b:
        by_b.setdefault(int(b.get("idx", -1)), b)
    rows, used = [], set()
    for a in rows_a:
        i = int(a.get("idx", -1))
        b = by_b.get(i)
        r = _pair(a, b)
        if b is None:
            r["only"] = "A"
            r["warn"] = "B 机没有这一段（它晚起 / 这一段没等到流）"
        else:
            used.add(i)
            if a.get("name") != b.get("name"):
                r["warn"] = ("同一段号两边名字不同（A=%s / B=%s）—— 公告与清单串了"
                             % (a.get("name"), b.get("name")))
            else:
                ov = overlap_ratio(a, b)
                drift = abs(a["t_start"] - b["t_start"])
                if ov < OVERLAP_WARN or drift > tol_s + max(
                        a["t_end"] - a["t_start"], 0.0) * 0.5:
                    r["warn"] = ("墙钟对不上（重叠 %.0f%%、起点差 %.1fs）—— "
                                 "这一行的延迟别当两机同时测的" % (ov * 100, drift))
        rows.append(r)
    for b in rows_b:                                  # B 侧多出来的（A 没跑这一段）
        i = int(b.get("idx", -1))
        if i in used:
            continue
        r = _pair({"idx": i, "name": b.get("name"), "fps": None, "bitrate": None,
                   "gop": None, "t_start": b.get("t_start"),
                   "t_end": b.get("t_end")}, b)
        r["only"] = "B"
        r["warn"] = "A 机没有这一段（清单里删了？A 机跳过了？）"
        rows.append(r)
    rows.sort(key=lambda r: r.get("idx", -1))
    return rows


# ══════════════════════════════════════════
# 评分
# ══════════════════════════════════════════

def link_loss_ratio(row):
    """**链路**丢帧：B 机收到的 / A 机实际发出的（拿不到给 None）。

    **为什么不能拿"收到 / 目标 fps"当这个判据**（原来的写法，2026-09-26 实测把它
    坑了）：A 机自己发不满目标时（实测 60fps 档只出 54、144 档只出 117），
    那个比值必然 < 0.98 —— 于是**18 条全部被判成"丢帧太多"**，整张表一条建议都给不出 ✗。
    而"没发出那么多"和"发出去丢了"是两件完全不同的事：
      · A 发不满 = A 机的天花板 → 这是**候选之间的差别**，由 speed/实际fps 反映；
      · B 收得比 A 发的少 = 链路/B 机丢帧 → 这一段延迟才真的不可信。
    分母用 A 的**实际输出**才是对的口径。
    """
    out, recv = row.get("out_fps"), row.get("recv_fps")
    if not out or not recv or out <= 0:
        return None
    return max(0.0, float(recv) / float(out))


def classify(row):
    """这一行能不能用 + 为什么 → (可用?, 原因)。**先淘汰再比大小。**

    淘汰规则（顺序即优先级）：
      ① A 机是瓶颈（`speed` < 1.00）—— 这是硬件跟不上，改 B 机没用；
      ② 探针几何判据没过 —— 这一段的延迟数是假的（`probe_mono` False）。
         注意：几何问题**不属于某一条候选**，整轮都该是坏的；所以真跑的时候
         `stream_sweep` 会在一开始先拦（见那边的 precheck），这里只是兜底；
      ③ **链路**丢帧（见 link_loss_ratio）—— 那一段的延迟也不可信；
      ④ 这一段压根没量到延迟。
    """
    if row.get("a_ok") is False:
        # A 机这一段压根没起来（比如 GPU 缩放那档 ffmpeg 起不来）—— 这是最该看见的原因
        return False, "A 机这一段没跑起来（%s）" % (row.get("a_why") or "没说原因")
    if not row.get("out_fps") and (row.get("a_ok") is not False) \
            and row.get("recv_fps") is None and row.get("speed") is None:
        # **一句 stats 都没有** = ffmpeg 起来又立刻死了（实测 2026-09-26：GPU 缩放那两档
        # 就是 `out_fps=0 / frames=0`）。A 侧的 ok 只看"进程起没起来"，这种情况它记的是
        # True，所以判据得放在这里 —— 否则表上只会说"没量到延迟"，真正的原因就丢了。
        return False, ("A 机这一段没起来（ffmpeg 一句输出都没有 —— 多半是这段"
                       "滤镜链不支持，比如 scale_cuda）")
    sp = row.get("speed")
    if sp is not None and sp < SPEED_MIN:
        return False, "A 机跟不上（speed %.3f < %.2f）" % (sp, SPEED_MIN)
    if row.get("probe_mono") is False:
        return False, "探针几何判据没过，延迟数不可信"
    ll = link_loss_ratio(row)
    if ll is not None and ll < RECV_RATIO_MIN:
        return False, ("链路丢帧（B 收到 %.1f / A 发出 %.1f fps，只到 %.0f%%）"
                       % (row.get("recv_fps") or -1, row.get("out_fps") or -1,
                          ll * 100))
    if row.get("lat_p95") is None:
        return False, "没量到延迟"
    return True, ""


def rank(rows):
    """→ (排好序的 rows, 最优那条或 None, 结论一句话)。

    排序口径（写死，别每次跑完临时改）：
        **先看 p95**（卡顿才是手感杀手），再看 p50，再看码率（省带宽），最后看 CPU。
        并列时**低帧率优先**：同样的 p95，帧率低的那套对 A 机更轻、对 B 机压力更小。

    **全军覆没也要给建议**（2026-09-26 修）：原来全部被淘汰时返回 `best=None`，
    表尾只留一句"没有可比的组合" —— 人拿不到任何能用的结论，而那一轮其实**量到了
    很有价值的东西**（比如 144fps 那档 p50 只有 65ms）。现在照样排序、照样给
    **折中建议**，只是把话说明白：它为什么被淘汰、先解决什么。
    """
    ok, bad = [], []
    for r in rows:
        good, why = classify(r)
        r["pass"] = good
        r["fail"] = why
        (ok if good else bad).append(r)

    def key(r):
        # `None` 要当"最差"排（**不能只靠 dict.get 的默认值**：键在、值是 None 时
        # 默认值不生效 —— 实测排序时 `None < float` 直接 TypeError）。
        # 会碰到 None 的行是"只有 A 侧数据"那种（B 机缺段 / ffmpeg 起不来）。
        def n(v, d=1e9):
            return d if v is None else v

        return (n(r.get("lat_p95")), n(r.get("lat_p50")),
                n(r.get("a_bitrate_kbps")), n(r.get("cpu_pct"), 0.0),
                n(r.get("fps"), 0))

    ok.sort(key=key)
    bad.sort(key=key)
    if not ok and not bad:
        return [], None, "没有任何可比的段（两边都没量到？）"
    best = ok[0] if ok else bad[0]
    if ok:
        why = ("最优：%s（p95 %.0fms / p50 %.0fms / 码率 %.1fMbps）"
               % (best["name"], best["lat_p95"], best.get("lat_p50") or -1,
                  (best.get("a_bitrate_kbps") or 0) / 1000.0))
    else:
        why = ("**没有一条满足全部前提**，下面是折中建议：%s（p95 %.0fms / p50 %.0fms）\n"
               "      它被淘汰的原因：%s\n"
               "      先把这个解决掉再重跑一轮 —— 在那之前这张表只能当参考。"
               % (best["name"], best.get("lat_p95") or -1,
                  best.get("lat_p50") or -1, best.get("fail") or "?"))
    return ok + bad, best, why


def render(rows, best, title="推流自检", why=""):
    """→ 一张给人看的表（终端/日志都能读）。

    `why` 是 `rank()` 给的那句结论（**全军覆没时它写着"折中建议 + 为什么"**）——
    传进来优先用它，别自己再拼一句（两处口径会漂）。
    """
    out = ["", "=" * 108, "%s   （p95 排序；标 ✗ 的已被淘汰）" % title, "=" * 108]
    out.append("%-16s %5s %6s %5s %7s %7s %6s %7s %9s %9s %9s  %s"
               % ("段", "fps", "码率", "GOP", "speed", "实际fps", "丢帧",
                  "收到fps", "延迟p50", "延迟p95", "抖动", "判定"))
    for r in rows:
        mark = "✓" if r.get("pass") else "✗"
        if r.get("only"):                     # 只有一侧有数据：照样列，别悄悄丢掉
            mark = "%s侧" % r["only"]

        def f(v, w, p=1, dash="-"):
            return ("%*.1f" % (w, v)) if isinstance(v, (int, float)) else \
                ("%*s" % (w, dash))

        out.append("%-16s %5s %6s %5s %7s %s %s %s %9s %9s %9s  %s %s"
                   % (r["name"], r.get("fps"), r.get("bitrate"), r.get("gop"),
                      # 三位小数：0.996 打印成 "1.00" 会写出"1.00 < 1.00"这种荒唐话
                      ("%.3f" % r["speed"]) if r.get("speed") is not None else "-",
                      f(r.get("out_fps"), 7), f(r.get("drop"), 6, 0),
                      f(r.get("recv_fps"), 7),
                      f(r.get("lat_p50"), 9, 0), f(r.get("lat_p95"), 9, 0),
                      f(r.get("jitter"), 9, 0), mark,
                      ("—— " + "；".join(
                          [t for t in (r.get("a_why"), r.get("warn")) if t])
                       if r.get("only") else
                       (("—— " + r["fail"]) if not r.get("pass") else
                        (("⚠ " + r["warn"]) if r.get("warn") else "")))))
    out.append("=" * 108)
    if best:
        out.append("结论：" + (why or render_best_line(best)))
    elif why:
        out.append(why)
    out.append("说明：`speed` 是 ffmpeg 的处理速度（<1.00 = A 机跟不上，一定丢帧）；")
    out.append("      `实际fps / 目标fps` 低于 100% 是 A 机顶不上去 —— **不一定是坏事**：")
    out.append("      屏幕只有 120Hz 时，144 档最多也就出 120（实测就是 117~120）。")
    out.append("      `延迟 p50/p95` 只在**探针几何判据通过**时才记（否则整轮该先修几何）；")
    out.append("      判定里的「链路丢帧」= B 收到的比 A **实际发出**的少（拿目标帧率当分母"
               "会把「A 发不满」误判成丢帧）。")
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
