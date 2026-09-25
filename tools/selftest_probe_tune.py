"""探针手动微调（probe_tune）自检：**正几何要解对，偏几何要判出来**。

**为什么必须钉这条**：2026-09-25 查「端到端延迟今天 300~400+」花了半天，最后
发现 `config/link.yaml` 里的探针几何是**旧的**（注释写着 A 机按 cell=23/gap=3
画，文件里却是 cell=16/gap=2 → 中心距 18 vs 21）。每块偏 2.8px，42 块累积
≈118px（5.7 个方块）→ 低位全采错 → 解出的时间戳是假的，而**没有人报警**：

  · `probe_recv` 把假值当延迟报出来（p95 4.7s、max 4.9s，紧贴它自己的
    `--max-delay` 上限）；
  · 工作台那边 `resolve_delay_ms` 只挡"≥1 天"的乱值，挡不住这种"低位采错"。

所以这里的判据是：**几何不对时，必须说「几何不对」并且不给延迟数**。

跑法：
    python -m tools.selftest_probe_tune      # 全过返回 0，有失败返回 1
    python -m tools.probe_tune --selftest    # 同一个入口（转到这里）
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import probe_codec                     # noqa: E402
from tools.probe_tune import (Verdict, apply_key, decode_with,  # noqa: E402
                              initial_geo, nudge_key, project_calib,
                              sample_cells, should_save)

BITS, CELL, GAP = 40, 16.0, 2.0
TS = 70_000_000            # 当天偏移毫秒（约 19:26）—— 只要能表示就行


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def synth(geo, ts=TS, bits=BITS):
    """按给定几何画一条码带（白=1、黑=0），返回灰度图。

    画法必须与 `sample_cells` 的取整规则配套：这里直接按"方块左上角 =
    x + i*(cell+gap)"画，采样点落在方块中心。
    """
    n = 2 + bits
    cell, gap = float(geo["cell"]), float(geo["gap"])
    w = int(n * (cell + gap) + 40)
    h = int(cell + 40)
    img = np.zeros((h, w), np.uint8)
    for i, b in enumerate(probe_codec.encode_bits(ts, bits)):
        cx = int(round(geo["x"] + i * (cell + gap)))
        img[int(geo["y"]):int(geo["y"] + cell), cx:cx + int(cell)] = 255 if b else 0
    return img


# ---------------------------------------------------------------- 用例

def t_roundtrip():
    """几何正确 → 解出原值；且每一块的灰度都清晰地落在黑/白上。"""
    geo = {"x": 20.0, "y": 20.0, "cell": CELL, "gap": GAP}
    img = synth(geo)
    ts, seq, means = decode_with(img, geo, BITS)
    check(ts == TS, "几何正确却解错了：%s（期望 %d）" % (ts, TS))
    check(len(seq) == 2 + BITS, "采样块数不对：%d" % len(seq))
    check(len(means) == 2 + BITS, "灰度数组长度不对：%d" % len(means))


def t_drift_breaks_decode():
    """**核心回归**：偏一点点（0.5px/块）就会解错 —— 这就是今天的病灶。"""
    geo = {"x": 20.0, "y": 20.0, "cell": CELL, "gap": GAP}
    img = synth(geo)
    bad = dict(geo, gap=GAP - 0.5)          # 中心距 18 → 17.5，42 块累积 21px
    ts_bad, _s, _m = decode_with(img, bad, BITS)
    check(ts_bad != TS,
          "偏了 0.5px/块还解出同一个值？那每块的漂移就没被累积 —— 判据失去意义")


def t_verdict_rejects_jumpy_ts():
    """乱跳的时间戳必须被判成「几何不对」，而且**不给延迟**。"""
    vd = Verdict()
    period = 16.7
    for k in range(20):
        vd.add((TS + k * 137) % probe_codec.DAY_MS, period)   # 每帧跳 137ms 且可能回退
    ok, why = vd.summary()
    check(not ok, "乱跳的时间戳被判成可信：%s" % why)
    check(("乱跳" in why) or ("不对" in why), "拒绝理由说不清：%r" % why)


def t_picker_prefers_framed_geometry():
    """框选给的几何**必须赢过 hint**（link.yaml 的旧值），判据是"采样清晰度"。

    实测事故（2026-09-25 晚）：用户把方块带**像素级**框准，回来仍是那份偏小的旧节距
    （cell=16/gap=2 → 周期 18.0，而实际 ≈18.3）—— 因为老逻辑是"谁先解出「接近此刻」
    的值就用谁"，而 hint 排在最前面；错位采样照样能凑出落在 ±10 分钟窗口里的值。
    现象：绿框右边框不全、尾部 6~8 块采样发灰、延迟整块挪 ≈0.1 秒。

    这里造一张**周期 18.3** 的合成码带，框选矩形给对、hint 故意给错的旧值：
    选出来的必须是框选那一份，且**采样模糊 0 块**。
    """
    from tools import probe_codec as pc

    cell_true, gap_true = 17.0, 1.3            # 周期 18.3
    bits = 40
    ts = int(pc.now_ms())
    geo_true = {"x": 30.0, "y": 0.0, "cell": cell_true, "gap": gap_true}
    img = synth(geo_true, ts, bits)
    n = 2 + bits
    # 框选矩形 = 第一条块左缘 → 最后一条块右缘（这就是"像素级框准"）
    rect = (geo_true["x"], 0.0, n * (cell_true + gap_true) - gap_true, cell_true)
    geo = pc.solve_from_rect(img, rect, bits, strict=False,
                             gap_hint=2.0, cell_hint=16.0)      # hint 是旧值
    check(geo is not None, "框选几何解不出来")
    check(abs(geo["cell"] - cell_true) <= 1.01,
          "选出来的仍是 hint 那份（cell=%.2f）而不是框选那份（应≈%.1f）"
          % (geo["cell"], cell_true))
    amb = pc.cell_ambiguity(img, geo["x"], geo["y"], geo["cell"], geo["gap"], bits)
    check(amb == 0, "选出来的几何采样仍发灰（模糊 %d 块）—— 判据没起到作用" % amb)

    # 反面：hint 故意给一个**周期离谱**的（14/0），框选那份仍要赢
    geo2 = pc.solve_from_rect(img, rect, bits, strict=False,
                              gap_hint=0.0, cell_hint=14.0)
    check(geo2 is not None and abs(geo2["cell"] - cell_true) <= 1.01,
          "hint 离谱时没能回到框选那份：%s" % (geo2 or {}).get("cell"))


def t_verdict_tolerates_one_gap():
    """**单次大跳不许误报** —— 那是采样侧掉帧，不是错位。

    实测误报过（2026-09-25 晚）：框选保存前的验证只收到 6 帧、其中一次步长 195ms
    （30fps 的显示层掉了一帧），被判成"几何不对"并弹窗拦下。判据本该分轻重：
      · **倒退**（时间戳往回了）—— 强证据，一次都不许放过（同一帧重复送到是 0）；
      · **大跳**（比平时大）—— 弱证据，容忍 1 次；连着跳才是错位采样。
    """
    vd = Verdict()
    now = probe_codec.now_ms()
    base = now - 400
    # 中间 50→195 是掉帧造成的一跳（+145ms），其余正常
    for st in (0, 16, 33, 50, 195, 212, 229, 246):
        vd.add(base + st, 16.7, delay_raw_ms=300.0)
    ok, why = vd.summary()
    check(ok, "单次掉帧造成的大跳被误判成「几何不对」：%s" % why)

    # 倒退一次 → 必须否
    vd2 = Verdict()
    for st in (0, 16, 33, 20, 50, 66):
        vd2.add(base + st, 16.7, delay_raw_ms=300.0)
    ok2, why2 = vd2.summary()
    check(not ok2, "倒退了一次却判 OK：%s" % why2)
    check("倒退" in why2, "理由里没说清是倒退：%r" % why2)

    # 连着大跳（真错位的形态）→ 必须否
    vd3 = Verdict()
    for k in range(8):
        vd3.add(base + k * 700, 16.7, delay_raw_ms=300.0)
    ok3, why3 = vd3.summary()
    check(not ok3, "连着大跳却判 OK：%s" % why3)


def t_verdict_accepts_smooth_ts():
    """正常（单调、步长≈帧周期、接近此刻）必须放行 —— 否则工具会把好几何也否掉。"""
    vd = Verdict()
    period = 16.7
    now = probe_codec.now_ms()
    for k in range(20):
        vd.add(now - int(20 * period) + int(k * period), period)
    ok, why = vd.summary()
    check(ok, "正常的单调序列被判成不可信：%s" % why)
    check(vd.step_med is not None and 12 <= vd.step_med <= 22,
          "步长中位算错：%s" % vd.step_med)


def t_verdict_blocks_implausible():
    """解出的时刻离"现在"太远 → 也要否掉（那种值算出来的"延迟"毫无意义）。"""
    vd = Verdict()
    for k in range(20):
        vd.add(k * 16, 16.7)        # 从当天 0 点开始，离此刻几小时
    ok, why = vd.summary()
    check(not ok, "离此刻几小时的时间戳被判成可信：%s" % why)


def t_nudge_steps():
    """按键步进：1px / Shift 10px；cell、gap 的 0.25 步进与下限。"""
    geo = {"x": 100.0, "y": 20.0, "cell": 16.0, "gap": 2.0}
    check(nudge_key(geo, "a")["x"] == 99.0, "a 应该是 x-1")
    check(nudge_key(geo, "d", shift=True)["x"] == 110.0, "Shift+d 应该是 x+10")
    check(nudge_key(geo, "w")["y"] == 19.0 and nudge_key(geo, "s")["y"] == 21.0,
          "w/s 应该是 y∓1")
    check(abs(nudge_key(geo, "r")["cell"] - 16.25) < 1e-9, "r 应该是 cell+0.25")
    check(abs(nudge_key(geo, "e")["cell"] - 15.75) < 1e-9, "e 应该是 cell-0.25")
    check(abs(nudge_key(geo, ",")["gap"] - 1.75) < 1e-9, "逗号应该是 gap-0.25")
    check(abs(nudge_key(geo, ".")["gap"] - 2.25) < 1e-9, "点应该是 gap+0.25")
    check(nudge_key(geo, "q") is None, "没定义的键不该改几何")
    # cell 不许小到 3px 以下（太小的话 40 位码根本读不出来）
    tiny = {"x": 0.0, "y": 0.0, "cell": 3.0, "gap": 0.0}
    check(nudge_key(tiny, "e")["cell"] == 3.0, "cell 下限没守住（缩到 3px 以下了）")
    # gap 不许变负（负间距的几何没有意义）
    check(nudge_key({"x": 0.0, "y": 0.0, "cell": 16.0, "gap": 0.0}, ",")["gap"] == 0.0,
          "gap 下限没守住（变成负数了）")


def t_key_dispatch():
    """**按键分派**（`apply_key`）也要钉 —— 不能只测 `nudge_key`。

    踩过的盲区：`main()` 里原本拿字符直接比 "wasder,."，而 Shift 走的是**大写**
    （cv2 给 'D'）→ 全部落到"不改几何"，10px 步进是死的；自检当时只测了
    `nudge_key`（纯函数），所以照样全绿。**测到接线才算测到。**
    """
    geo = {"x": 100.0, "y": 20.0, "cell": 16.0, "gap": 2.0}
    check(apply_key(geo, "d")["x"] == 101.0, "小写 d 应该是 x+1")
    check(apply_key(geo, "D")["x"] == 110.0, "**大写 D（Shift）应该是 x+10**")
    check(apply_key(geo, "A")["x"] == 90.0, "大写 A 应该是 x-10")
    check(apply_key(geo, "W")["y"] == 10.0, "大写 W 应该是 y-10")
    check(apply_key(geo, "S")["y"] == 30.0, "大写 S 应该是 y+10")
    check(apply_key(geo, "R")["cell"] == 16.25, "大写 R 也该改 cell")
    check(apply_key(geo, "p") is None and apply_key(geo, "t") is None
          and apply_key(geo, "q") is None and apply_key(geo, "\n") is None,
          "非微调键不该改几何（否则 p/t/Enter/q 会被吃掉）")
    check(apply_key(geo, "") is None, "空字符不该改几何")


def t_initial_geo_prefers_project():
    """起始几何要**先试项目标定**（工作台优先用它），再退全局 → link.yaml。

    这条防的是"工具和工作台各自取到不同的一份几何"——今天岔开的根源。
    """
    import unittest.mock as mock

    cal = {"x_ratio": 0.1, "y_ratio": 0.02, "cell_ratio": 0.02,
           "gap_ratio": 0.002, "bits": BITS}
    with mock.patch("tools.probe_tune.project_calib", return_value=cal):
        geo, src = initial_geo((768, 1366, 3), BITS)
    check(src == "项目标定", "有项目标定时没用它，来源=%r" % src)
    check(abs(geo["x"] - 0.1 * 1366) < 1e-6, "x 没按画面宽度换算：%s" % geo["x"])
    check(abs(geo["cell"] - 0.02 * 1366) < 1e-6, "cell 没按画面宽度换算：%s" % geo["cell"])


def t_initial_geo_matches_workbench():
    """起始几何要**和工作台同一套来源**（项目标定 → 全局 → link.yaml）。

    老工具的坑：`probe_recv` 只读 link.yaml，工作台走 pick_calib —— 两边在两套
    坐标里比大小，对不上时谁都不报警。
    """
    geo, src = initial_geo((768, 1366, 3), BITS)
    check(int(geo["cell"]) == int(geo["cell"]), "cell 不是数")
    check(src in ("项目标定", "全局标定", "配置值"), "来源标注不认识：%r" % src)
    check(geo["cell"] > 0 and geo["gap"] >= 0, "几何不合法：%s" % geo)
    if src == "配置值":                      # 没标定过时应当等于 link.yaml 的值
        from tools.config import get
        check(abs(geo["x"] - float(get("probe", "x", 100))) < 1e-6,
              "回退到配置值时 x 没对上 link.yaml")


def t_sample_uses_same_rounding():
    """采样取整必须与 `probe_codec.read_bits` 一致（否则对齐了也会解错）。

    用**浮点 cell/gap**（17.0 / 2.25）跑一遍：这正是当年踩过的取整坑
    （`read_bits` 的注释：按整数取整会在 41 块上累计偏一个方块宽）。
    """
    geo = {"x": 33.0, "y": 17.0, "cell": 17.0, "gap": 2.25}
    img = synth(geo)
    seq, _m = sample_cells(img, geo["x"], geo["y"], geo["cell"], geo["gap"], BITS)
    ref = probe_codec.read_bits(img, geo["x"], geo["y"], geo["cell"], geo["gap"], BITS)
    check(seq == ref, "本工具的采样与 probe_codec.read_bits 不一致（两边取整不同）")
    check(probe_codec.bits_to_ms(seq, BITS) == TS,
          "浮点 cell/gap 下解不出原值：%s" % probe_codec.bits_to_ms(seq, BITS))


def t_offline_skips_now_check():
    """离线回放要关掉"接近此刻"那条判据，但**单调性照旧**。

    不然 `--file 昨天的录像` 会把好几何也否掉（全是"对不上本地时间"）。
    """
    old = 3_600_000                     # 一小时前那种
    off = Verdict(need_now=False)
    for k in range(10):
        off.add(old + k * 16, 16.7)
    ok, why = off.summary()
    check(ok, "离线模式（need_now=False）还把「对不上本地时间」当失败：%s" % why)

    live = Verdict(need_now=True)
    for k in range(10):
        live.add(old + k * 16, 16.7)
    ok2, why2 = live.summary()
    check(not ok2, "在线模式却放过了一个离此刻一小时的时间戳：%s" % why2)

    # 关键：离线模式**不能**把"乱跳"一起放行
    off2 = Verdict(need_now=False)
    for k in range(10):
        off2.add((old + k * 811) % probe_codec.DAY_MS, 16.7)
    ok3, why3 = off2.summary()
    check(not ok3, "离线模式把乱跳的时间戳放行了（那判据就白关了）：%s" % why3)


def t_port_busy_says_what_to_do():
    """端口被占（10048）要**说清怎么办** —— 这是实测最常撞的那一下。

    现象：A 机在推流，B 机这边工作台的「实时」页却一直「无流」，工具一跑就丢一个
    `OSError: [Errno 10048]`（连 errno 名字都没有），人只能去猜是网络还是 A 机没推。
    根因几乎总是同一个：**UDP 5000 被工作台的实时预览占着**（一个端口只能有一个
    收流者，而 PyAV/ffmpeg 那个 socket 不让别人复用）。
    """
    import tools.stream_sweep as sw
    from tools.probe_tune import human_open_error

    busy = human_open_error(Exception(
        "[Errno 10048] Error number -10048 occurred: 'udp://0.0.0.0:5000'"))
    check("端口已被占用" in busy, "10048 没被翻译成人话：%s" % busy)
    check("实时" in busy and "停止" in busy,
          "没说清去哪关掉占用者（人要能照做）：%s" % busy)

    other = human_open_error(Exception("timed out"))
    check("端口" not in other and "timed out" in other,
          "别的错被硬套成端口占用了：%s" % other)

    # stream_sweep 侧：认得出"重试没意义"的那类错（别白扫 17 段）
    del sw._LAST_ERR[:]
    for _ in range(3):
        sw._LAST_ERR.append("[Errno 10048] Error number -10048 occurred")
    check(sw.fatal_open_error() is not None,
          "stream_sweep 没认出端口被占（会白等 15 秒 × 18 段）")
    del sw._LAST_ERR[:]
    sw._LAST_ERR.append("timed out")
    check(sw.fatal_open_error() is None, "普通超时被当成了致命错误")


def t_verdict_catches_shifted_value():
    """**整片挪位**也要判出来：时间戳单调、也在一天之内，只是位置不对。

    这是 2026-09-25 实测撞到的第二种错位形态：当时在用的几何
    `x=99 y=21 cell=16.25 gap=2.00`（比实际画的小 2.5px/块）—— 单调性过了、
    ±10 分钟值域也过了，**但延迟算不出来**（解出的时刻比此刻早几十秒~几分钟）。
    只靠前两道判据，界面会把它误报成「时钟对不上 → 去对时」，方向全错；
    真去对时也永远好不了。
    """
    vd = Verdict()
    now = probe_codec.now_ms()
    for k in range(10):
        vd.add(now - 300_000 + k * 16, 16.7, delay_raw_ms=300_000 + k * 16)
    ok, why = vd.summary()
    check(not ok, "整片挪位 5 分钟的解被判成可信（报出来的延迟会是假的）")
    check("挪" in why or "几何" in why, "拒绝理由没说到点上：%r" % why)

    # 正常的一帧（延迟 160ms）不许被误杀
    vd2 = Verdict()
    for k in range(10):
        vd2.add(now - 160 + k * 16, 16.7, delay_raw_ms=160 + k * 16)
    ok2, why2 = vd2.summary()
    check(ok2, "正常的 160ms 延迟被判成不可信：%s" % why2)

    # 没喂 raw 差时（离线回放那条路）不该因此变红 —— 判据是"有数据才判"
    vd3 = Verdict(need_now=False)
    for k in range(10):
        vd3.add(k * 16, 16.7)
    ok3, why3 = vd3.summary()
    check(ok3, "没喂原始差时被值域判据误杀：%s" % why3)


def t_save_guard():
    """**判据没过就不许保存**（要硬存得显式 force）。

    这是今天事故的直接教训：错的几何被存进项目 → 工作台和工具**都**按它采样 →
    延迟读数全是假的，而界面上一切正常（"绿框"只是没框全，没人报警）。
    """
    ok_vd = Verdict()
    now = probe_codec.now_ms()
    for k in range(10):
        ok_vd.add(now - 160 + int(k * 16), 16.7)
    yes, why = should_save(ok_vd)
    check(yes, "判据过了却不给存：%s" % why)

    bad_vd = Verdict()
    for k in range(10):
        bad_vd.add((4000 + k * 733) % probe_codec.DAY_MS, 16.7)   # 乱跳 + 离此刻远
    no, why2 = should_save(bad_vd)
    check(not no, "判据没过却允许保存（错几何会被静默存进项目）：%s" % why2)
    yes2, why3 = should_save(bad_vd, force=True)
    check(yes2, "显式 --force 也不给存：%s" % why3)
    check("force" in why3, "硬存的理由没标出来：%r" % why3)


def t_project_calib_path():
    """`--project` 指定的项目要真的被读到（不能总是退回"最近打开的那个"）。

    写错项目的后果是"工作台按一份错的几何采样"，界面上看不出来 —— 所以存的目标
    必须由人显式指定，且这里钉住"指定了就按指定的读"。
    """
    import unittest.mock as mock

    cal = {"x_ratio": 0.05, "y_ratio": 0.01, "cell_ratio": 0.01,
           "gap_ratio": 0.001, "bits": BITS}

    class _P:
        """替身：`get("decision")` 应当直接返回 decision 那本字典（别再包一层）。"""

        def get(self, k, default=None):
            return {"probe_calib": cal} if k == "decision" else default

    with mock.patch("tools.probe_tune.open_project", return_value=_P()):
        got = project_calib("E:/nonexistent/proj")
        geo, src = initial_geo((768, 1366, 3), BITS, "E:/nonexistent/proj")
    check(got == cal, "指定项目后没读到它的 probe_calib：%s" % got)
    check(src == "项目标定", "指定项目后来源不是项目标定：%r" % src)
    check(abs(geo["cell"] - 0.01 * 1366) < 1e-6, "cell 没按指定项目的比例换算")


def t_probe_recv_shares_geometry_source():
    """`probe_recv` 必须用同一套几何来源（否则它的延迟数和工作台不可比）。

    只钉"接线"：它现在从 `probe_tune.initial_geo` 取几何。哪天有人把它改回
    "只读 link.yaml"，这条就会挂 —— 那正是今天两边岔开的原因。
    """
    import tools.probe_recv as pr
    check(pr.initial_geo is initial_geo,
          "probe_recv 用的不是同一份 initial_geo（又回退成只读 link.yaml 了？）")


def t_verdict_is_shared_with_workbench():
    """判据只能有**一份实现**：工作台（live_thread）与工具都吃 `probe_codec.Verdict`。

    两边各写一份必然漂移，而漂移出来的现象是最难查的那类：工具说"几何 OK"、
    工作台说"几何可疑"（或反过来），于是没人敢信任何一边的数。
    """
    from tools import probe_codec as pc
    from tools import probe_tune as pt
    check(pt.Verdict is pc.Verdict, "probe_tune 的 Verdict 不是 probe_codec 那一份")
    src = (Path(__file__).resolve().parent.parent / "gui" / "live_thread.py"
           ).read_text(encoding="utf-8")
    check("probe_codec.Verdict()" in src,
          "工作台没用共享的 probe_codec.Verdict（各写一份判据会漂移）")
    check("probe_mono" in src and "probe_jumpy" in src,
          "工作台没把单调性结论放进 stats（界面就无从拦假延迟）")
    panel = (Path(__file__).resolve().parent.parent / "gui" / "live_panel.py"
             ).read_text(encoding="utf-8")
    check("probe_mono" in panel and "_verify_geo_live" in panel,
          "界面没接单调性结论（几何可疑时会把假延迟照样摆出来）")


CASES = [
    ("几何正确时解出原值", t_roundtrip),
    ("偏 0.5px/块就解错（今天的病灶）", t_drift_breaks_decode),
    ("乱跳的时间戳被判「几何不对」", t_verdict_rejects_jumpy_ts),
    ("正常单调序列被放行", t_verdict_accepts_smooth_ts),
    ("离此刻太远的解被否掉", t_verdict_blocks_implausible),
    ("按键步进（1px / Shift 10px / 0.25）", t_nudge_steps),
    ("按键分派（Shift 大写不能漏）", t_key_dispatch),
    ("起始几何优先用项目标定", t_initial_geo_prefers_project),
    ("起始几何与工作台同一来源", t_initial_geo_matches_workbench),
    ("采样取整与 read_bits 一致", t_sample_uses_same_rounding),
    ("离线回放关掉「接近此刻」、保留单调性", t_offline_skips_now_check),
    ("框选几何必须赢过 hint（按采样清晰度挑）", t_picker_prefers_framed_geometry),
    ("单次掉帧的大跳不许误报（倒退则必报）", t_verdict_tolerates_one_gap),
    ("整片挪位（单调但位置不对）也要判出来", t_verdict_catches_shifted_value),
    ("端口被占（10048）要说清怎么办", t_port_busy_says_what_to_do),
    ("判据没过不许保存（--force 才硬存）", t_save_guard),
    ("--project 指定项目真的被读到", t_project_calib_path),
    ("probe_recv 与工作台共用几何来源", t_probe_recv_shares_geometry_source),
    ("单调性判据只有一份实现（工作台与工具共用）", t_verdict_is_shared_with_workbench),
]


def main():
    bad = 0
    for name, fn in CASES:
        try:
            fn()
            print("[ OK ] %s" % name)
        except Exception as e:
            bad += 1
            print("[FAIL] %s\n        %s: %s" % (name, type(e).__name__, e))
    print("%d/%d 通过" % (len(CASES) - bad, len(CASES)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
