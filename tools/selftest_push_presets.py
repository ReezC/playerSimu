"""推流自检的**账**自检：候选 → 命令行、ffmpeg 指标解析、两机对账、评分。

**为什么先钉这套账**：推流自检要两台机器合作（A 机换参数试推、B 机量延迟），
中间那套账（清单顺序、`-stats` 解析、墙钟对齐、谁淘汰谁最优）如果错了，**表会
长得很好看但结论是反的** —— 比如把 `speed=0.94`（A 机跟不上）的那套选成"最优"，
或者用一行的延迟去代表另一条候选。两边都跑起来才发现就太贵了，所以这些纯计算
在这里全部覆盖（一台机器就能跑）。

真正的"尺子"（端到端延迟）另有它的判据：`probe_codec.Verdict`（几何可信度），
见 tools/selftest_probe_tune.py。

跑法：
    python -m tools.selftest_push_presets      # 全过返回 0，有失败返回 1
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import push_presets as pp                     # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


#: 一行真实的 ffmpeg -stats（-nostats 时根本不打，自检会加上 -stats）
FFMPEG_LINE = ("frame= 1234 fps=143 q=23.0 size=  12345kB time=00:00:08.60 "
               "bitrate=11756.5kbits/s speed=0.99x drop=3 dup=0")


def t_presets_file():
    """清单文件本身要**能用、且名字唯一**（名字是两机对账的键）。"""
    d = pp.load()
    ps = d["presets"]
    check(len(ps) >= 1, "清单读出来是空的")
    check(d["seconds"] > 0 and d["settle_seconds"] >= 0, "时长不合法：%s" % d)
    names = [p["name"] for p in ps]
    check(all(names), "有候选没有名字（对账就断了）：%s" % names)
    check(len(set(names)) == len(names), "候选名重复：%s" % names)
    for p in ps:
        check(p.get("fps") and p.get("bitrate"), "候选缺 fps/码率：%s" % p)
    # 默认清单是 fps{60,120,144} × 码率{6,12,20M} × GOP{30,60}
    check(len(pp.default_presets()["presets"]) == 18,
          "默认清单应当 18 套，实际 %d" % len(pp.default_presets()["presets"]))
    # 文件坏了/没了 → 退回默认清单（不能崩）
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.json"
        bad.write_text("{ this is not json", encoding="utf-8")
        check(len(pp.load(bad)["presets"]) == 18, "坏清单没有退回默认")
        check(len(pp.load(Path(td) / "missing.json")["presets"]) == 18,
              "清单缺失没有退回默认")


def t_push_block_inherits_machine():
    """候选只覆盖它写了的键；host/port/ffmpeg/采集/编码器必须**继承**。

    这几项是"这台机器的属性"（B 机 IP、ffmpeg 路径、有没有 N 卡），自检乱动它们
    的后果是推流直接发不出去 —— 而人只会看到"这一套失败了"，方向全错。
    """
    base = {"ffmpeg": "ffmpeg", "capture": "ddagrab", "encoder": "h264_nvenc",
            "host": "192.168.1.2", "port": 5000, "width": 1366, "height": 768,
            "fps": 50, "bitrate": "6M", "gop": 60, "low_latency": True,
            "pkt_size": 1316}
    got = pp.push_block({"fps": 144, "bitrate": "20M", "gop": 30,
                         "passthrough": True}, base)
    check(got["host"] == "192.168.1.2" and got["port"] == 5000, "host/port 被动了")
    check(got["pkt_size"] == 1316 and got["low_latency"] is True,
          "别的键被动了：pkt_size/low_latency")
    check(got["ffmpeg"] == "ffmpeg" and got["encoder"] == "h264_nvenc",
          "ffmpeg/编码器被动了")
    check((got["fps"], got["bitrate"], got["gop"]) == (144, "20M", 30),
          "候选的键没覆盖上去：%s" % got)
    check(got["passthrough"] is True, "passthrough 没覆盖")
    # 候选里没写的键，base 原样保留（比如 width）
    check(got["width"] == 1366, "width 丢了")


def t_parse_ffmpeg_stats():
    """`-stats` 那行要解析出 speed / fps / 丢帧 / 码率 —— A 机"扛不扛得住"全靠它。"""
    st = pp.parse_ffmpeg_stats(FFMPEG_LINE)
    check(abs(st["speed"] - 0.99) < 1e-6, "speed 解析错：%s" % st.get("speed"))
    check(abs(st["fps"] - 143) < 1e-6, "fps 解析错：%s" % st.get("fps"))
    check(int(st["drop"]) == 3, "drop 解析错：%s" % st.get("drop"))
    check(abs(st["bitrate_kbps"] - 11756.5) < 0.1, "码率解析错：%s" % st)
    check(pp.parse_ffmpeg_stats("") == {}, "空行该给 {}")
    check(pp.parse_ffmpeg_stats("[probe_gen] 窗口 800x40 @ (100,20)") == {},
          "不相干的行被解析出东西了（会把日志里别的数字当成指标）")
    # speed 可能带空格（ffmpeg 排版会变），别只认紧挨着的那种
    check(pp.parse_ffmpeg_stats("... speed= 1.02x ...").get("speed") == 1.02,
          "speed= 后面带空格时解析不出来")


def t_take_last():
    """一串统计行取**尾部**（稳态），不是平均 —— 起推那几百毫秒必然慢。"""
    got = pp.take_last([{"speed": 0.30, "fps": 20}, {"speed": 0.98},
                        {"speed": 1.01, "fps": 143}])
    check(got["speed"] == 1.01 and got["fps"] == 143,
          "取尾部的规则错了：%s" % got)
    check(pp.take_last([]) == {}, "空列表该给 {}")


def _row_a(name, speed=1.0, st=0.0, span=20.0, drop=0, fps=60):
    return pp.row_a(name, 0, {"fps": fps, "bitrate": "12M", "gop": 30},
                    st, st + span, [{"speed": speed, "fps": fps, "drop": drop,
                                     "bitrate_kbps": 11000}])


def _row_b(name, p50=120.0, p95=200.0, recv=60.0, mono=True, st=0.0, span=20.0,
           fps=60):
    return pp.row_b(name, 0, {"fps": fps, "bitrate": "12M", "gop": 30},
                    st, st + span,
                    {"recv_fps": recv, "proc_fps": 45.0, "lat_p50": p50,
                     "lat_p95": p95, "lat_max": p95 + 30, "lat_n": 600,
                     "jitter": 12.0, "read_p50": 11.0, "gap_p50": 16.0,
                     "frames": 1200, "probe_mono": mono, "probe_why": "ok"})


def t_merge_alignment():
    """合并：按**顺序**对齐、用**墙钟**核对；对不上要**说出来**（不闷头出表）。"""
    rows = pp.merge([_row_a("a", st=100.0), _row_a("b", st=200.0)],
                    [_row_b("a", st=100.5), _row_b("b", st=200.5)])
    check(len(rows) == 2 and rows[0]["name"] == "a" and rows[1]["name"] == "b",
          "合并顺序不对：%s" % [r["name"] for r in rows])
    check(rows[0]["warn"] == "", "正常对齐却报了警：%r" % rows[0]["warn"])
    check(rows[0]["lat_p95"] == 200.0, "延迟没并进来：%s" % rows[0])

    # 顺序错位（有人在中间插了一条）→ 必须说出来
    rows2 = pp.merge([_row_a("a")], [_row_b("x")])
    check("顺序不一致" in rows2[0]["warn"], "顺序错位没报警：%r" % rows2[0]["warn"])

    # 墙钟差太远（B 机中途重开过预览）→ 必须说出来
    rows3 = pp.merge([_row_a("a", st=100.0)], [_row_b("a", st=160.0)])
    check("墙钟对不上" in rows3[0]["warn"], "墙钟错位没报警：%r" % rows3[0]["warn"])

    # 段数不同 → 明确列一行，别静默丢
    rows4 = pp.merge([_row_a("a"), _row_a("b")], [_row_b("a")])
    check(rows4[-1]["idx"] == -1 and "段数不同" in rows4[-1]["warn"],
          "段数不同没提示：%s" % rows4[-1])


def t_classify_rules():
    """淘汰规则：A 机跟不上 / 几何不可信 / 丢帧太多 —— 一条都不能放过去。"""
    slow = {"speed": 0.94, "probe_mono": True, "recv_ratio": 1.0, "lat_p95": 100.0}
    ok, why = pp.classify(slow)
    check(not ok and "speed" in why, "A 机跟不上没被淘汰：%s" % why)

    jumpy = {"speed": 1.0, "probe_mono": False, "recv_ratio": 1.0, "lat_p95": 280.0}
    ok2, why2 = pp.classify(jumpy)
    check(not ok2 and "几何" in why2, "几何不可信没被淘汰：%s" % why2)

    lossy = {"speed": 1.0, "probe_mono": True, "recv_ratio": 0.90, "lat_p95": 200.0}
    ok3, why3 = pp.classify(lossy)
    check(not ok3 and "丢帧" in why3, "丢帧太多没被淘汰：%s" % why3)

    good = {"speed": 1.02, "probe_mono": True, "recv_ratio": 1.0, "lat_p95": 180.0}
    ok4, why4 = pp.classify(good)
    check(ok4 and not why4, "正常的一条被判不合格：%s" % why4)


def t_rank_picks_p95_first():
    """排序：**先 p95**（卡顿才是手感杀手），并列看 p50、码率、CPU。"""
    a = {**_row_b("慢中位", p50=90.0, p95=400.0)}
    a.update(speed=1.0)
    b = {**_row_b("稳", p50=200.0, p95=210.0)}
    b.update(speed=1.0)
    rows, best, why = pp.rank([a, b])
    check(best is not None and best["name"] == "稳",
          "p95 更小的没被选成最优（选的是 %s）—— 排序口径错了"
          % (best and best["name"]))

    # 两条都淘汰 → best 为空，结论要说清"全军覆没"
    x = {**_row_b("x")}
    x.update(speed=0.5)
    y = {**_row_b("y")}
    y.update(speed=0.6)
    rows2, best2, why2 = pp.rank([x, y])
    check(best2 is None and "全军覆没" in why2, "全灭时的结论不对：%s" % why2)


def t_render_and_apply():
    """出表 + 「最优那条怎么写回部署台」—— 最后一步不能让人手抄错。"""
    a = {**_row_a("best")}
    a.update(speed=1.0)
    b = {**_row_b("best", p50=110.0, p95=150.0)}
    rows, best, _why = pp.rank(pp.merge([a], [b]))
    txt = pp.render(rows, best)
    check("speed" in txt and "p95" in txt, "表头缺少关键列：%s" % txt[:200])
    check("结论：" in txt and "best" in txt, "没给出结论：%s" % txt[-300:])

    base = {"host": "192.168.1.2", "port": 5000, "pkt_size": 1316,
            "fps": 50, "bitrate": "6M", "gop": 60, "width": 1366, "height": 768}
    got = pp.to_deploy_push(best, base)
    check(got["host"] == "192.168.1.2" and got["pkt_size"] == 1316,
          "写回时动了机器属性：%s" % got)
    check(got["fps"] == 60 and got["bitrate"] == "12M" and got["gop"] == 30,
          "写回时没带上最优那条的参数：%s" % got)


def t_a_side_command_has_stats():
    """A 机侧命令：用部署台**同一份**实现 + 把 `-nostats` 换成 `-stats`。

    为什么要单独钉：`speed` 只有 ffmpeg 的状态行才有，而部署台卡片那条命令按设计
    带 `-nostats`。如果自检忘了换，A 机侧会**一条指标都收不到** —— 表里 speed 全空，
    而"哪套 A 机扛得住"这个问题就失去了唯一直接证据。
    """
    from deploy.push_sweep import cmd_for

    base = {"ffmpeg": "ffmpeg", "capture": "ddagrab", "encoder": "h264_nvenc",
            "host": "192.168.1.2", "port": 5000, "width": 1366, "height": 768,
            "fps": 50, "bitrate": "6M", "gop": 60, "low_latency": True,
            "passthrough": False, "pkt_size": 1316}
    p = {"fps": 120, "bitrate": "12M", "gop": 30, "passthrough": True}
    cmd = cmd_for(pp.push_block(p, base))
    check("-stats" in cmd, "A 机侧命令没有 -stats（speed 就收不到了）")
    check("-nostats" not in cmd, "-nostats 还在，状态行不会打出来")
    check(any("ddagrab=framerate=120" in str(a) for a in cmd),
          "候选的帧率没进命令：%s" % cmd)
    check("-fps_mode" in cmd and "passthrough" in cmd, "passthrough 没进命令")
    check("udp://192.168.1.2:5000" in " ".join(str(a) for a in cmd),
          "推流目标被动了（这是机器属性，不该被自检改）")
    check(cmd[0] == "ffmpeg", "命令头不对：%s" % cmd[:2])


CASES = [
    ("清单文件可读、名字唯一、坏了退回默认", t_presets_file),
    ("A 机侧命令带 -stats（speed 的唯一来源）", t_a_side_command_has_stats),
    ("候选只覆盖自己写的键（机器属性继承）", t_push_block_inherits_machine),
    ("ffmpeg -stats 行解析（speed/fps/丢帧/码率）", t_parse_ffmpeg_stats),
    ("统计取尾部（稳态），不是平均", t_take_last),
    ("两机合并：按顺序对齐、墙钟核对、歪了要报", t_merge_alignment),
    ("淘汰规则：跟不上 / 几何不可信 / 丢帧多", t_classify_rules),
    ("排序：先 p95，全灭时结论要说清", t_rank_picks_p95_first),
    ("出表与「写回部署台」不手抄", t_render_and_apply),
]


def main():
    bad = 0
    for name, fn in CASES:
        try:
            fn()
        except Exception as e:
            bad += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("%d/%d 通过" % (len(CASES) - bad, len(CASES)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
