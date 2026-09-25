"""屏幕时间码探针：编码 / 解码。

A 机在屏幕固定位置画一串黑白方块，编码"当前毫秒时间戳"；
该画面经 OBS 推流到 B 机后，B 机从图像中解码出时间戳，
与本地接收时刻相减即得端到端媒体延迟（需先做双机时钟对齐）。

方块尺寸取 16px、间隔 2px 是为了扛住 H.264 压缩伪影。
编码格式： [1, 0] 起始标记 + bits 位大端时间戳（毫秒）
"""

import datetime
import json
import statistics
import time
from pathlib import Path

DEFAULT_BITS = 40
DAY_MS = 86_400_000
MASK_40 = (1 << DEFAULT_BITS) - 1


def day_start_ms() -> int:
    """今天本地 00:00:00 的 epoch 毫秒。"""
    lt = time.localtime()
    midnight = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday)
    return int(midnight.timestamp() * 1000)


def now_ms() -> int:
    """当天 00:00 起的毫秒偏移（0 ~ 86_400_000）。

    注意：不能直接用 epoch 毫秒 —— 2^40 ms 仅约 34.9 年，
    从 1970 年起算在 2004 年就已溢出，会得到被截断的错误值。
    """
    return int(time.time() * 1000) - day_start_ms()


def encode_bits(ts_ms: int, bits: int = DEFAULT_BITS) -> list[int]:
    """毫秒时间戳 -> 方块序列（1=白, 0=黑）。

    ts_ms 必须是 now_ms()（当天 0 点起的偏移），不能直接传 epoch 毫秒。
    """
    if not (0 <= ts_ms <= DAY_MS):
        raise ValueError(
            f"ts_ms={ts_ms} 超出当天范围 [0, {DAY_MS}]。"
            "请使用 now_ms()（当天偏移毫秒），不要用 time.time()*1000 这类 epoch 毫秒。"
        )
    seq = [(ts_ms >> i) & 1 for i in range(bits - 1, -1, -1)]
    return [1, 0] + seq


#: 「采样点落在黑白之间」的灰度带。方块只有黑/白两态，采样窗口取在块中心：
#: 对齐时均值贴近 0 或 255；偏了（节距差零点几像素、x/y 差一两像素）就压到边缘上。
#: **这是不依赖时钟的对齐证据** —— 时钟错不会让采样点变灰，几何错才会。
AMBIG_LO = 60.0
AMBIG_HI = 195.0


def read_cells(gray, x, y, cell, gap, bits: int = DEFAULT_BITS):
    """按几何采样 2+bits 个方块 → (bits 列表, 每块灰度均值；取不到给 -1.0)。

    **取整规则必须与 `read_bits` 完全一致**（差半个像素的取整会在 41 块上累积、
    越往后越偏 —— 正是"绿框没框全"那种）。画采样框的地方（工作台实时页、
    tools/probe_tune）也走这一个函数：**"屏幕上画在哪"和"实际采哪儿"永远同一套坐标**，
    否则人看到的框和机器采的点会悄悄错开，越看越糊涂。
    """
    n = 2 + bits
    h, w = gray.shape[:2]
    half = max(1, int(float(cell) // 4))
    cy = int(round(float(y) + float(cell) / 2.0))
    out, means = [], []
    for i in range(n):
        cx = int(round(float(x) + i * (float(cell) + float(gap))
                       + float(cell) / 2.0))
        if cy >= h or cx >= w:
            out.append(0)
            means.append(-1.0)
            continue
        y0, y1 = max(0, cy - half), min(h, cy + half)
        xa, xb = max(0, cx - half), min(w, cx + half)
        patch = gray[y0:y1, xa:xb]
        if patch.size == 0:
            out.append(0)
            means.append(-1.0)
            continue
        m = float(patch.mean())
        means.append(m)
        out.append(1 if m > 128.0 else 0)
    return out, means


def no_decode_hint(gray, x, y, cell, gap, bits: int = DEFAULT_BITS):
    """解不出时间码时，**下一步该查什么** —— 看采样点上的灰度分布。

    只给两种结论，因为**下一步的动作只有两种**：
      · 采样点几乎只有一种灰度（黑一片 / 白一片）→ 那个位置**没有码带**：
        去 A 机确认在画时间码（部署台「探针」卡片），或位置整体挪走了；
      · 黑白都采到了 → **码带是在的**，对不上的是起点/节距/位宽 → 重新 solve 几何
        （另报一句有多少采样点落在黑白之间：那说明采样窗口还压在方块边缘上）。

    **为什么不再细分**：试过按"黑/白/灰各占多少"分三档，但那些数字强烈依赖相位 ——
    实测同样的错位，起点差一点就会量出完全不同的分布（`gap=9` 判成"黑白分明"、
    `gap=6.75` 判成"灰蒙蒙"，而两者要做的事一模一样）。上面这两条的判据是**结构性**的
    （真码带一定黑白都有），所以稳。2026-09-25 现场卡的就是"只有一句没量到延迟"。
    """
    _seq, means = read_cells(gray, x, y, cell, gap, bits)
    good = [m for m in means if m >= 0]
    if len(good) < 2:
        return "采样点几乎都跑到画面外了 —— 位置整个不对（换个几何）"
    lo = sum(1 for m in good if m < 40)          # 黑块
    hi = sum(1 for m in good if m > 215)         # 白块
    mid = len(good) - lo - hi
    if lo == 0 or hi == 0:
        return ("那个位置**几乎只有一种灰度**（黑 %d / 白 %d / 中间 %d，共 %d 个采样点）"
                "—— **画面上大概没有码带**：先确认 A 机在画时间码（部署台「探针」卡片"
                "在跑），再确认位置没整体挪走"
                % (lo, hi, mid, len(good)))
    return ("采样点**黑白都有**（黑 %d / 白 %d，其中 %d 个落在中间）—— **码带是在的**，"
            "对不上的是起点/节距/位宽：用 `python -m tools.probe_auto --solve` 重量。%s"
            % (lo, hi, mid,
               "而且采样点**全落在方块中间**（没有压边的）—— 这种最像「节距差零点几像素」："
               "每块偏一点点，几十块之后正好偏掉一个整块，于是高位是对的、尾部时对时错"
               "（现象就是时间戳偶尔大跳/倒退）"
               if mid == 0 else
               "中间那些偏多 = 采样窗口有一部分压在方块边缘上，同样是重标几何"))


def cell_ambiguity(gray, x, y, cell, gap, bits: int = DEFAULT_BITS):
    """这一组几何有多少个采样点落在黑白之间（含跑到画面外的）→ 越少越准。

    **为什么它是挑选几何的正确标准**："解出一个接近此刻的时间码"骗得过去（错位
    照样能凑出落在 ±10 分钟窗口里的值），而"采样点是不是压在方块中心"骗不过去。
    2026-09-25 实测：用户像素级框选的几何被一份偏小的旧节距顶掉，就是因为在用
    前者当判据（绿框右边框不全、尾部 6~8 块发灰、延迟整片挪 ≈0.1s）。
    """
    _seq, means = read_cells(gray, x, y, cell, gap, bits)
    return sum(1 for m in means if m < 0 or AMBIG_LO <= m <= AMBIG_HI)


def bits_to_ms(seq: list[int], bits: int = DEFAULT_BITS) -> int | None:
    """方块序列 -> 毫秒时间戳；校验失败返回 None"""
    n = 2 + bits
    if len(seq) < n:
        return None
    if seq[0] != 1 or seq[1] != 0:
        return None
    v = 0
    for b in seq[2:n]:
        v = (v << 1) | (1 if b else 0)
    return v


def read_bits(gray, x, y, cell, gap, bits: int = DEFAULT_BITS) -> list[int]:
    """从灰度图中采样方块序列。

    gray : HxW 灰度图 (numpy uint8)
    x, y : 第一个方块左上角坐标

    **cell 支持浮点，且必须支持** —— 用整数取整会累积出致命漂移：
        当屏幕分辨率与推流分辨率不一致（如 2560x1440 推 1920x1080，
        缩放 0.75），画面里的方块只有 13.5px。若按 int(round(cell))=14
        计算，41 个方块累计偏 20px（超过一个方块宽），后半段全部采错。

    **但切片下标仍必须是整数**：浮点直接拿去切会抛
        TypeError: slice indices must be integers or None or have an __index__ method
    gap 在 config 里配的就是 2.25，所以这里统一在算完坐标后再取整。
    """
    n = 2 + bits
    h, w = gray.shape[:2]
    out: list[int] = []

    cellf = float(cell)
    gapf = float(gap)
    xf, yf = float(x), float(y)

    # 采样窗口取方块中心附近的一小块。用 /2.0 而不是 //2：
    # // 对浮点返回浮点，混进切片就崩。
    half = max(1, int(cellf // 4))
    cy = int(round(yf + cellf / 2.0))

    for i in range(n):
        cx = int(round(xf + i * (cellf + gapf) + cellf / 2.0))
        if cy >= h or cx >= w:
            out.append(0)
            continue
        y0, y1 = max(0, cy - half), min(h, cy + half)
        xa, xb = max(0, cx - half), min(w, cx + half)
        patch = gray[y0:y1, xa:xb]
        if patch.size == 0:
            out.append(0)
            continue
        out.append(1 if float(patch.mean()) > 128.0 else 0)
    return out


def decode_ms(gray, x, y, cell, gap, bits: int = DEFAULT_BITS) -> int | None:
    return bits_to_ms(read_bits(gray, x, y, cell, gap, bits), bits)


def ts_plausible(ts, minutes: float = 10.0) -> bool:
    """解出的时间戳是否「接近此刻」（当天毫秒，跨午夜按环形算）。

    **这是「对准了没有」的最强判据**（从 `probe_auto.py` 挪到这里，两边共用一份）。
    错位采样同样能解出一个「合法」的 40 位数 —— 它落在 [0, 86400000) 之内，
    看着完全正常（实测拿到过 14543、1301503、40353791 这类），但那个值本质是
    随机数，落在当前时刻附近的概率极低。

    容差不能放大：曾经用 3 小时，结果凌晨测试时（ts 本身才 219 万毫秒）等于把
    大半个值域都算成「合理」，错位解照样放行。双机对时后差距是毫秒级，
    10 分钟足够覆盖时钟漂移。
    """
    if ts is None:
        return False
    # **先卡范围：必须落在「一天之内」才算合法值。**
    # 40 位能表示到 1.1e12 ms（约 34.9 年），而一天只要 27 位 —— 解出的乱码
    # 大多数都 ≥ 一天。老代码只算环形距离，`DAY_MS - d` 在 d > DAY_MS 时是**负数**，
    # 于是任何高位置 1 的乱码都被判成「接近此刻」＝放行。
    # 后果实测过：错位采样存进一份坏几何，界面说「与本地时刻相符 —— 对准了」，
    # 而实时那边永远算不出延迟（解出的是 306001920ms ≈ 85 小时这种值）。
    if not (0 <= ts < DAY_MS):
        return False
    d = abs(ts - now_ms())
    d = min(d, DAY_MS - d)
    return d < minutes * 60 * 1000


# ---- 人工框选标定（把「位置全靠猜」变成「框一下、当场验证」）----
#
# 存的是**画面比例**而不是像素：换流分辨率、换窗口大小都不用重标
# （read_bits 明确支持浮点 cell，就是为这种缩放准备的）。
# 单独一个文件，不去改 config/link.yaml —— 那份文件注释很多，用 yaml 重写会
# 把注释全冲掉，得不偿失（想固化进 link.yaml 的话，界面会把该抄的值打出来）。
CALIB_FILE = Path(__file__).resolve().parent.parent / "config" / "probe_calib.json"


def geometry_from_rect(rect, bits: int = DEFAULT_BITS, gap_hint=None) -> dict | None:
    """框选矩形 → 解码几何 (x, y, cell, gap)；明显不合理返回 None。

    **约定与 read_bits 一致**：x/y 是**第一个方块**的左上角，不是探针窗口的左上角
    （link.yaml 里也是这么标的：实测「第一个白块左缘」）。

    方块带由 n = 2 + bits 个**等大等距**方块组成，头两个是固定标记（白、黑，
    见 `encode_bits` 的 `[1, 0]`），所以：
        cell = 带子高度
        gap  = (带子宽度 - n*cell) / (n-1)

    gap 用宽度反推：宽度跨 n 个方块，取整误差被 n-1 摊薄；只按高度反推的话
    误差直接落在 cell 上，40 位码会被放大成整个方块宽的漂移。
    """
    if not rect:
        return None
    x, y, w, h = (float(v) for v in rect)
    n = 2 + int(bits)
    cell = h
    if cell < 3 or w <= 0:
        return None
    gap = (w - n * cell) / (n - 1)
    if not (-1.0 <= gap <= 12.0):
        # 宽度推不出合理的 gap：多半是没框住整条带（少框/多框了方块）。
        # 有配置值就先用它 —— 后面的搜索环节还会拿真实解码验证，不会放过去。
        if gap_hint is None:
            return None
        gap = float(gap_hint)
    return {"x": x, "y": y, "cell": cell, "gap": max(0.0, gap), "bits": int(bits)}


def solve_from_rect(gray, rect, bits: int = DEFAULT_BITS, strict: bool = True,
                    gap_hint=None, cell_hint=None) -> dict | None:
    """以框选为起点小范围搜索，返回**真能解出时间码**的几何；解不出返回 None。

    **为什么还要搜**：人眼框的边不等于像素级精确（±2~3px 很常见），而这是 40 位码 ——
    差一个方块宽度就整个错位。以框选算出的几何为中心，试 dx/dy ∈ [-3,3]、
    cell ∈ {h-1,h,h+1}、gap ∈ {推算值, gap_hint}，取**第一个**能解出码的。

    解出来了就说明对准了（前两位标记必须是 1,0）。这就是「不用猜」的依据 ——
    所以界面上的规则是：**解不出就不保存**，宁可不改也不改坏。

    strict=True 时还要求时间戳「接近此刻」（需要双机已对时，见 ts_plausible）；
    还没对时的时候传 False，只要求能解出合法码。

    返回的 dict 里额外带 `ts`（解出的时间戳）、`plausible`（是否对上本地时刻）、
    `snap`（相对框选偏了多少像素 —— 自动对齐的痕迹）。
    """
    base = geometry_from_rect(rect, bits, gap_hint)
    if base is None:
        return None

    gaps = {round(base["gap"], 3)}
    if gap_hint is not None:
        gaps.add(round(float(gap_hint), 3))

    # 优先就近：位移小、cell 偏差小的先试（多半第一下就中 —— 采样窗口本身
    # 就有 ±cell/4 的容差，±3px 的框选误差通常直接过）。
    # 范围给到 ±5：框错一点点不该让人重来一遍；700 多个候选也就几十毫秒。
    # 候选 cell：框选高度 ±1，再**加上调用方给的已知好值**（通常来自 link.yaml
    # 的 probe.cell）。有它，框得整体偏小也能救回来 —— 判据没放宽（照样必须解出
    # 合理时间码），所以多试几个候选不会放坏值进去，只是更宽容。
    cells = [base["cell"], base["cell"] - 1, base["cell"] + 1]
    if cell_hint and float(cell_hint) not in cells:
        cells.append(float(cell_hint))

    cands = []
    # (cell_hint, gap_hint) 也放进候选（通常来自 link.yaml 的旧值）—— 但**只是候选，
    # 不再"排最前面就赢"**。取舍交给下面的「采样清晰度」评分（见那段说明）。
    if cell_hint and gap_hint is not None:
        cands.append((0, 0, 0, 0, float(cell_hint), float(gap_hint)))
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            for cell in cells:
                for g in sorted(gaps, key=lambda v: abs(v - base["gap"])):
                    cands.append((abs(dx) + abs(dy), abs(cell - base["cell"]),
                                  dx, dy, cell, g))
    cands.sort(key=lambda c: (c[0], c[1]))

    # ── 选法（2026-09-25 改：**按「采样清晰度」挑，不按"谁先解出合理值"**）──
    #
    # 老逻辑是"第一个能解出「接近此刻」的候选就用它"，而且**hint（link.yaml 的旧值）
    # 排在最前面**。后果实测：用户把方块带像素级框准了，回来仍是那份**偏小的旧节距**
    # —— 因为错位采样照样能凑出一个落在 ±10 分钟窗口里的值（"看着合法"），hint 永远
    # 抢先。现象：绿框右边框不全、尾部 6~8 块采样发灰、延迟整片挪 ≈0.1 秒。
    #
    # 新判据 `cell_ambiguity`（采样点落在黑白之间的块数）**与时钟无关**：
    # 时钟错不会让采样点变灰，几何错才会。取"模糊块最少"的那份，
    # 同分再比"离框选近"（位移小、cell 偏差小）。
    best_ok = None       # 解出合理值里、模糊最少的
    best_any = None      # 只解出码（对不上此刻）里、模糊最少的 —— 非 strict 时当退路
    for _d, _dc, dx, dy, cell, g in cands:
        if cell < 3:
            continue
        x, y = base["x"] + dx, base["y"] + dy
        try:
            ts = decode_ms(gray, x, y, cell, g, bits)
        except Exception:
            continue
        if ts is None:
            continue
        # 值必须落在一天之内，否则必然是错位采样（见 ts_plausible 的说明）。
        # 这条**不依赖双机是否对时**，所以连 strict=False 的退路也要守。
        if not (0 <= ts < DAY_MS):
            continue
        hit = {"x": x, "y": y, "cell": float(cell), "gap": float(g),
               "bits": int(bits), "ts": int(ts),
               "snap": (dx, dy, int(cell - base["cell"])),
               "plausible": bool(ts_plausible(ts))}
        amb = cell_ambiguity(gray, x, y, cell, g, bits)
        hit["ambiguity"] = int(amb)
        score = (amb, abs(dx) + abs(dy), abs(float(cell) - base["cell"]))
        if hit["plausible"]:
            if best_ok is None or score < best_ok[0]:
                best_ok = (score, hit)
            # 框选本身就对上了（模糊 0、值也合理、没位移）：不用再扫
            if amb == 0 and not dx and not dy:
                break
        elif best_any is None or score < best_any[0]:
            best_any = (score, hit)
    if best_ok is not None:
        return best_ok[1]
    return None if strict else (best_any[1] if best_any else None)


def load_calib() -> dict | None:
    """读人工框选的结果（config/probe_calib.json）；没有或坏了返回 None。"""
    try:
        with open(CALIB_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k in ("x_ratio", "y_ratio", "cell_ratio", "gap_ratio"):
            float(d[k])
        return d
    except Exception:
        return None


def calib_from_geo(geo, frame_shape, bits: int = DEFAULT_BITS) -> dict:
    """框选几何 → 标定 dict（**存画面比例**）。纯函数，不落盘。

    项目标定用的就是它：比例存进项目后，换分辨率/窗口都不用重标
    （read_bits 明确支持浮点 cell）。
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return {
        "x_ratio": geo["x"] / w,
        "y_ratio": geo["y"] / h,
        "cell_ratio": geo["cell"] / w,
        "gap_ratio": geo["gap"] / w,
        "bits": int(bits),
        "frame": [w, h],            # 标定时用的画面尺寸（只为看日志，不参与换算）
        "solved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_calib(geo, frame_shape, bits: int = DEFAULT_BITS) -> dict:
    """把几何存成**全局**标定文件（旧路径，保留兼容）。返回存下的 dict。

    新流程存进**项目**（见 settings.probe_calib）—— 不同项目可能分辨率/客户端
    布局不同；这个文件只作为「项目里没有时的回退」，让老标定不至于白丢。
    """
    d = calib_from_geo(geo, frame_shape, bits)
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    return d


def pick_calib(project_calib=None):
    """该用哪份标定 → (标定 dict, 来源文字)；都没有返回 (None, "配置值")。

    **优先级：项目里的 → 全局文件 → 都不用（调用方回退到 link.yaml 的像素值）。**

    项目优先，和 HP/MP 条同一套思路：不同项目可能分辨率/客户端布局不同，
    标定要跟着项目走；没打开项目时 `project_calib` 来自「最近打开的项目」
    （见 gui/main_window.py 的 _bind_decision_params）。全局文件是旧版存法，
    留着是为了不把已经标好的白丢。
    """
    for d, src in ((project_calib, "项目标定"), (load_calib(), "全局标定")):
        if not d:
            continue
        try:
            vals = [float(d[k]) for k in
                    ("x_ratio", "y_ratio", "cell_ratio", "gap_ratio")]
        except Exception:
            continue                    # 结构不对就跳过，别拿坏值去采样
        if all(v > 0 for v in vals):
            return dict(d), src
    return None, "配置值"


def calib_to_px(cal, shape):
    """标定（比例）→ 当前画面的像素几何 (x, y, cell, gap)。"""
    h, w = shape[:2]
    return (cal["x_ratio"] * w, cal["y_ratio"] * h,
            cal["cell_ratio"] * w, cal["gap_ratio"] * w)


# ---- 「这份几何可不可信」的判据（工作台与 tools/probe_tune 共用一份）----
#
# **为什么光有 ts_plausible 不够**：它只问"解出的时刻接近现在吗"。而 40 位码里
# 低位是**快变化的毫秒**，错位采样只把低位采错 —— 凑出一个仍落在 ±10 分钟窗口里
# 的值完全可能（实测：link.yaml 的几何比实际画的小 2.8px/块，42 块累积偏 118px，
# 低位全错，解出的却"看着合法"，于是延迟被报成一坨乱数 —— p95 4.7s 紧贴上限）。
# 而**单调性是时间轴上的性质**：时间戳每帧只该往前走一小步，错位必然破坏它。
# 所以两者都要：ts_plausible 挡"值域明显不对"，单调性挡"低位被采错"。

#: 判据窗口：最近多少帧的时间戳参与单调性判断
VERDICT_HIST = 24
#: 允许的最大步长 = 帧周期 × 这个倍数（超过就是错位解出的跳变）
VERDICT_STEP_MUL = 4.0
#: 「解出的时刻」与"本机收到这一帧的时刻"之间允许的范围（毫秒）。
#:
#: **这是第三道判据，补的是前两道抓不到的那种错**：错位采样如果只是把整片值**平移**
#: （低位被采错、高位数对了但位置不对），解出来的时间戳会**照旧单调**（平移也是单调的），
#: 也**照旧落在 ±10 分钟窗口里** —— 于是前两道全过，而那个值其实是"另一刻"：
#: 延迟算不出来，界面上还会误报成"时钟对不上，去对时"（实测踩过：几何错的方向全错）。
#:
#: 取值范围怎么定：
#:   · 下界 **−1 秒**：收到的画面**不可能来自未来**。管道只会让内容变旧，所以延迟
#:     必须是正的；留 1 秒是给对时残差/抖动。实测撞到的错位正是"解出的时刻比此刻
#:     晚了十几秒"（落在负区）—— 下界不收紧就抓不住（−60 秒内都算"能接受"）。
#:   · 上界 **60 秒**：真延迟不可能到这个量级（局域网 + 已对时）；超过就是整片挪位了。
VERDICT_VALUE_MIN_MS = -1000
VERDICT_VALUE_MAX_MS = 60_000


class Verdict:
    """几何可不可信（跨帧单调性）+ 延迟能不能报。

    用法：每解出一帧就 `add(ts, 帧周期ms)`，然后 `summary()` 取结论。
    几何变了要 `reset()`（换标定那一刻时间戳会跳一次，不该算作"乱跳"）。
    """

    def __init__(self, need_now=True):
        #: 要不要"解出的时刻接近本机现在"这一条。**离线回放必须关掉它** ——
        #: 拿昨天的录像跑，解出的时刻当然离"现在"几小时；单调性照样有效。
        self.need_now = need_now
        self.ts_hist = []
        self.mono = True
        self.step_med = None
        self.step_max = None
        self.jumped = 0
        #: 分开记：**倒退**（强证据，时间戳本该只增）与**大跳**（弱证据 —— 显示层
        #: 掉一帧、抓帧丢一帧都会造成"这一跳比平时大"，实测误报过：只 1 次跳变、
        #: 最大 195ms，其实是 30fps 显示层的正常抖动）。
        self.back_steps = 0
        self.big_steps = 0
        self.plausible = False
        #: 解出的时刻与"本机收帧时刻"的原始差（毫秒，越接近真实延迟越好）；
        #: None = 还没喂过。见 VERDICT_VALUE_TOL_MS 的说明。
        self.value_off_ms = None
        self.value_ok = True

    def reset(self):
        self.ts_hist = []
        self.mono = True
        self.step_med = None
        self.step_max = None
        self.jumped = 0
        self.back_steps = 0
        self.big_steps = 0
        self.plausible = False
        self.value_off_ms = None
        self.value_ok = True

    def add(self, ts, frame_period_ms, delay_raw_ms=None):
        """喂一帧。`delay_raw_ms` = 本机收帧时刻 − 解出时刻（毫秒）。

        传了它就多一道**值域判据**（见 VERDICT_VALUE_TOL_MS）：单调但整片平移的解
        会被它抓住 —— 那种解"看着一切正常"，只是那个数不能用。
        """
        self.ts_hist.append(ts)
        if len(self.ts_hist) > VERDICT_HIST:
            del self.ts_hist[0]
        ds = [b - a for a, b in zip(self.ts_hist, self.ts_hist[1:])]
        lim = VERDICT_STEP_MUL * frame_period_ms
        self.back_steps = sum(1 for d in ds if d < 0)
        self.big_steps = sum(1 for d in ds if d > lim)
        self.jumped = self.back_steps + self.big_steps
        pos = [d for d in ds if d > 0]
        self.step_med = statistics.median(pos) if pos else None
        self.step_max = max(ds) if ds else None
        # 判据：**倒退一次都不行**（时间戳只该往前走；同一帧重复送到是 0，不算倒退），
        # 而"大跳"最多容忍 1 次 —— 采样侧（工作台的显示帧、工具的取帧）本来就会掉帧，
        # 掉一帧就让步长翻倍；只有连着跳才是错位采样。实测踩过误报：1 次 195ms 的跳变。
        self.mono = (self.back_steps == 0 and self.big_steps <= 1)
        self.plausible = ts_plausible(ts)
        if delay_raw_ms is not None:
            self.value_off_ms = float(delay_raw_ms)
            self.value_ok = (VERDICT_VALUE_MIN_MS <= self.value_off_ms
                             <= VERDICT_VALUE_MAX_MS)

    def summary(self):
        """→ (可信?, 一行话)。不可信时**必须说清原因**，别只说"不行"。"""
        if len(self.ts_hist) < 3:
            return False, "采样中…（时间戳还没攒够）"
        if not self.mono:
            return False, ("几何不对：解出的时间戳在乱跳（倒退 %d 次、大跳 %d 次，"
                           "最大步长 %.0fms）—— 错位采样的典型特征，挪 x/y 或调 "
                           "cell/gap" % (self.back_steps, self.big_steps,
                                         self.step_max or 0))
        if self.need_now and not self.plausible:
            return False, ("解出的时刻对不上本地时间（差超 10 分钟）—— 几何不对，"
                           "或双机时钟没对（先在 B 机跑 tools.clock_sync --save）")
        if not self.value_ok:
            off = self.value_off_ms or 0.0
            if off < 0:
                detail = ("解出的时刻比此刻**晚**了 %.1f 秒 —— 收到的画面不可能"
                          "来自未来" % (-off / 1000.0))
            else:
                detail = ("解出的时刻比此刻**早**了 %.0f 秒 —— 真延迟到不了这个量级"
                          % (off / 1000.0))
            return False, ("几何不对：%s（时间戳单调、也在一天之内，光看单调看不"
                           "出来）—— 挪 x/y 或调 cell/gap" % detail)
        return True, ("几何可用：时间戳单调，步长中位 %s ms"
                      % ("%.1f" % self.step_med if self.step_med else "—"))


def resolve_delay_ms(t_recv_epoch_ms: float, ts_a: int) -> float | None:
    """把"当天偏移毫秒"还原为 epoch 毫秒并求延迟（ms），自动处理跨日。

    t_recv_epoch_ms : B 机接收时刻（B 机墙钟的 epoch 毫秒，已做时钟偏移校正）
    ts_a            : 探针解码出的 A 机"当天偏移毫秒"
    """
    t_content = day_start_ms() + ts_a
    d = t_recv_epoch_ms - t_content
    # 跨日（A 机在 23:59:59.9 生成、B 机在次日 00:00:00.1 收到）时 ±1 天修正
    for k in (0, -1, 1):
        cand = d + k * DAY_MS
        if 0.0 < cand < 60_000.0:
            return cand
    return None
