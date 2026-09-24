"""屏幕时间码探针：编码 / 解码。

A 机在屏幕固定位置画一串黑白方块，编码"当前毫秒时间戳"；
该画面经 OBS 推流到 B 机后，B 机从图像中解码出时间戳，
与本地接收时刻相减即得端到端媒体延迟（需先做双机时钟对齐）。

方块尺寸取 16px、间隔 2px 是为了扛住 H.264 压缩伪影。
编码格式： [1, 0] 起始标记 + bits 位大端时间戳（毫秒）
"""

import datetime
import json
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
                    gap_hint=None) -> dict | None:
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
    cands = []
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            for dcell in (0, -1, 1):
                for g in sorted(gaps, key=lambda v: abs(v - base["gap"])):
                    cands.append((abs(dx) + abs(dy), abs(dcell), dx, dy,
                                  base["cell"] + dcell, g))
    cands.sort(key=lambda c: (c[0], c[1]))

    fallback = None
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
        hit = {"x": x, "y": y, "cell": float(cell), "gap": float(g),
               "bits": int(bits), "ts": int(ts),
               "snap": (dx, dy, int(cell - base["cell"])),
               "plausible": bool(ts_plausible(ts))}
        if hit["plausible"]:
            return hit
        if fallback is None:
            fallback = hit          # 能解出码但对不上此刻：留着当退路
    return None if strict else fallback


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


def save_calib(geo, frame_shape, bits: int = DEFAULT_BITS) -> dict:
    """把几何按**画面比例**存下来（换分辨率不用重标）。返回存下的 dict。"""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    d = {
        "x_ratio": geo["x"] / w,
        "y_ratio": geo["y"] / h,
        "cell_ratio": geo["cell"] / w,
        "gap_ratio": geo["gap"] / w,
        "bits": int(bits),
        "frame": [w, h],            # 标定时用的画面尺寸（只为看日志，不参与换算）
        "solved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    return d


def calib_to_px(cal, shape):
    """标定（比例）→ 当前画面的像素几何 (x, y, cell, gap)。"""
    h, w = shape[:2]
    return (cal["x_ratio"] * w, cal["y_ratio"] * h,
            cal["cell_ratio"] * w, cal["gap_ratio"] * w)


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
