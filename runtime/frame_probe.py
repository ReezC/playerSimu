"""一帧的第 4 段：**时间戳探针解码**（阶段 1「编排收口」第二刀 ✓ 2026-10-20 ✓）。

屏幕左上角那串黑白方块是 A 机的**时间码**（`core/probe_codec.py` ✓）：解出来就能算
**端到端延迟**（A 机屏幕时间 → 本机解码），那是"控制质量"的真指标 ✓。
本段只干三件事：裁灰度 → 按几何解 40 位 → 记账（成/败/不可能/算不出延迟 ✓）。

搬自 `gui/live_thread.py::_run`（**行为零改动** ✓ —— 连注释里的现场账一起搬过来 ✓）。

⚠⚠ **跨帧状态按 B 挂账本**（用户 2026-10-20 选 B ✓）：`ProbeState` 里的东西**全是跨帧**的 ——
  延迟窗口（最近 120 个）、四类计数、几何键、"上一份几何"、单调性判据（`Verdict` ✓）⇒
  按 `FrameCtx` 的判据"下一帧还会用吗？"⇒ 会 ⇒ **不进盒子** ✗。
  账本由调用方持有（`_run` 里那个 `_probe`），本段只改它 ✓ —— 阶段不需要知道它住哪 ✓。

⚠ 几何**每秒重读**（`geom_fn` 自己带节流 ✓ 见 `_run` 的 `_probe_geom`）：
  框选完不用重开预览就生效 ✓；几何一变就 `Verdict.reset()`（换标定那一刻时间戳必然跳一次，
  不该被算成"乱跳" ✗ —— 否则刚存完就报"几何可疑"，人会被自己刚做的事吓到 ✓）。

判据 `selftest_live_panel::t_frame_probe_stage`：解不出来算 miss 不算 ok · 值不可能（≥一天）
算 invalid（**不是**时钟问题 ✗）· 值合法但算不出延迟算 reject · 延迟进窗口并更新 `e2e_ms` ·
几何一变重置判据 · 窗口上限 120 ✓。
"""

import cv2

from core import perf

#: ⚠ **不许在这里"裸 import"**（复核时补的 ✓ 2026-10-20）：`gui/live_thread.py` 那边原本是
#:   `try: from core import probe_codec / except: probe_codec = None` ⇒ 导入失败时**降级**、
#:   预览照开（探针关掉 ✓）。本模块如果裸 import，一旦那边坏了就是**整个实时线程起不来** ✗
#:   ⇒ 照样 try 一下、坏了置 None，`step()` 里按 None 直接返回 ✓（行为与老代码一致 ✓）。
try:
    from core import probe_codec
except Exception:                                    # pragma: no cover - 正常环境不会走到
    probe_codec = None

#: 延迟窗口上限（超过就丢最老的 ✓ 中位数用最近 120 个 ✓）
DELAY_WINDOW = 120
#: 超过这么多毫秒的"延迟"当**解码错误**（而不是真有 5 秒延迟 ✓）
DELAY_MAX_MS = 5000.0


class _NullVerdict:
    """判据替身：`core.probe_codec` 导入失败时用（**永远是"没问题"** ⇒ 界面不会报假警 ✓）。"""

    __slots__ = ("mono", "value_ok", "value_off_ms", "step_med", "ts_hist")

    def __init__(self):
        self.mono = True
        self.value_ok = True
        self.value_off_ms = None
        self.step_med = 0.0
        self.ts_hist = []

    def reset(self):
        pass

    def add(self, ts, step, delay_raw_ms=None):
        pass


def _new_verdict():
    """建一份判据（codec 可用就用真的 ✓ 不可用就给替身 ✓ 见文件开头那条"不许裸 import"）。"""
    if probe_codec is None:
        return _NullVerdict()
    try:
        return probe_codec.Verdict()
    except Exception:                                # noqa: BLE001 —— 判据建不出来也别挡住预览 ✓
        return _NullVerdict()


class ProbeState:
    """探针的**跨帧账本**（一次运行一本 ✓ 挂在调用方身上 ✓ 见文件开头那条 B ✓）。

    · `vd`       单调性 / 值域判据（`probe_codec.Verdict` ✓ 界面上也直接读它的字段 ✓）
    · `delays`   最近 `DELAY_WINDOW` 个端到端延迟（毫秒 ✓ 状态行取中位数 ✓）
    · `geom_key` 上一次用的几何（变了就重置判据 ✓）
    · `miss` / `invalid` / `ok` / `reject` 四类计数（**分开**才有诊断价值 ✓ 见 `step` ✓）
    · `jumpy`    判据判成"乱跳"的帧数（>0 = 几何可疑 ✓）
    · `e2e_ms`   最近一次算出来的端到端延迟（给世界状态用 ✓ 上绳对齐的保持窗口 ✓）
    """

    __slots__ = ("vd", "delays", "geom_key", "miss", "invalid", "ok",
                 "reject", "jumpy", "e2e_ms")

    def __init__(self, vd=None):
        self.vd = vd if vd is not None else _new_verdict()
        self.delays = []
        self.geom_key = None
        self.miss = 0
        self.invalid = 0
        self.ok = 0
        self.reject = 0
        self.jumpy = 0
        self.e2e_ms = 0.0

    def __repr__(self):
        return "<ProbeState ok=%d miss=%d invalid=%d reject=%d jumpy=%d>" % (
            self.ok, self.miss, self.invalid, self.reject, self.jumpy)


def step(st, on, vis, f, geom_fn, decode_fn, clock_ms, bits, show_fps=30.0):
    """解这一帧的探针 ⇒ 只改 `st`（返回值 = 这一帧算出来的延迟，`None` = 没有 ✓）。

    :param st: `ProbeState`（跨帧账本 ✓）
    :param on: 探针开着吗（`probe_on` ✓ 关掉时**整段不跑** ⇒ 连灰度都不转 ✓ 零开销）
    :param vis: 这一帧（BGR ✓）—— ⚠ 用 `vis` 不是 `raw`：探针在左上角，画框够不着它 ✓
               （老代码就是这么用 `vis` ✓ 别顺手改 ✗）
    :param f: 这一帧（要 `f.t_recv_wall` ✓）
    :param geom_fn: `shape -> (px, py, cell, gap)`（自己带 1 秒重读 ✓ 见 `_run` ✓）
    :param decode_fn: `decode_ms(gray, px, py, cell, gap, bits)`（`core.probe_codec` ✓ 由调用方给，
                     因为它可能是 `None` ⇒ 本段不 import 名字，只收函数 ✓）
    :param clock_ms: `() -> float` 双机时钟偏移（毫秒 ✓）
    :param show_fps: 显示帧率（判据的步长上限用它 ✓）
    """
    if not on or decode_fn is None or probe_codec is None:
        return None
    # 灰度：`f.image` 是 BGR
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    # 几何每帧现算：人工标定是比例，要按当前画面尺寸换算
    # （read_bits 支持浮点 cell，所以不必取整）
    px, py, cell, gap = geom_fn(gray.shape)
    # 几何一变（刚标完 / 换了项目）就重置判据：换标定那一刻时间戳
    # 必然跳一次，不该被算成"乱跳"—— 否则刚存完就报"几何可疑"，人会被自己刚做的事吓到。
    _key = (round(px, 2), round(py, 2), round(cell, 2), round(gap, 2))
    if _key != st.geom_key:
        st.geom_key = _key
        st.vd.reset()
    ts_a = decode_fn(gray, px, py, cell, gap, bits)
    if ts_a is None:
        st.miss += 1
        # 探针解不出来也要留痕：端到端延迟是「控制质量的真指标」，
        # 但它一旦解码失败，日志里只剩少数样本，读出来会**偏小**，
        # 看着像「延迟变好了」—— 那是假象（实测踩过：段内样本
        # 1433→161，中位同步从 35ms 掉到 12ms）。把 miss/ok 记下来，
        # 报告里一比就知道这个数能不能信。
        perf.count("probe_miss")
        return None
    if not (0 <= ts_a < probe_codec.DAY_MS):
        # 解出来了、但值不可能（≥ 一天）＝**几何错了**（采样点没落在
        # 方块上），不是时钟问题。混进「时钟对不上」会把人引错方向
        # （实测：解出 306001920ms ≈ 85 小时，界面却提示去对时）。
        st.invalid += 1
        perf.count("probe_invalid")
        return None
    st.ok += 1
    perf.count("probe_ok")
    # 跨帧单调性（只在这里记，结论交给界面）—— 判据没过时界面
    # **不许显示延迟数**：报一个假延迟比不报糟得多。
    # 步长上限用显示帧周期算（判据自己会乘一个宽松倍数）。
    _t_recv_ms = (f.t_recv_wall + clock_ms() / 1000.0) * 1000.0
    # 原始差也喂进判据：单调但**整片平移**的解（错位采样的另一种
    # 形态）只有它能识破，否则界面会把它误报成"时钟对不上，去对时"。
    st.vd.add(ts_a, 1000.0 / max(1.0, float(show_fps or 30.0)),
              delay_raw_ms=_t_recv_ms - (probe_codec.day_start_ms() + ts_a))
    if not st.vd.mono:
        st.jumpy += 1
        perf.count("probe_jumpy")
    if not st.vd.value_ok:
        perf.count("probe_value_off")
    d = probe_codec.resolve_delay_ms(_t_recv_ms, ts_a)
    # 超过 5 秒视为解码错误（而不是真的有 5 秒延迟）
    if d is None or d >= DELAY_MAX_MS:
        # 解出来了但算不出延迟：多半是双机时钟没对齐（也可能是
        # 解错了一位数字）。**和「解码失败」是两回事**，界面上要
        # 分开显示 —— 以前混成一句「未解出」，框选明明说解出了，
        # 这里却显示未解出，看着像 bug。
        st.reject += 1
        perf.count("probe_reject")
        return None
    st.delays.append(d)
    st.e2e_ms = d                  # 给世界状态用（上绳对齐的保持窗口）
    if len(st.delays) > DELAY_WINDOW:
        del st.delays[0]
    # 端到端延迟（A 机屏幕时间码 → 本机解码）才是最该盯的
    # 性能指标，以前只在界面显示中位数、不进日志。
    perf.sample("e2e_probe_ms", d)
    return d
