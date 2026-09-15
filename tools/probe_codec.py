"""屏幕时间码探针：编码 / 解码。

A 机在屏幕固定位置画一串黑白方块，编码"当前毫秒时间戳"；
该画面经 OBS 推流到 B 机后，B 机从图像中解码出时间戳，
与本地接收时刻相减即得端到端媒体延迟（需先做双机时钟对齐）。

方块尺寸取 16px、间隔 2px 是为了扛住 H.264 压缩伪影。
编码格式： [1, 0] 起始标记 + bits 位大端时间戳（毫秒）
"""

import datetime
import time

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


def read_bits(gray, x: int, y: int, cell: int, gap: int, bits: int = DEFAULT_BITS) -> list[int]:
    """从灰度图中采样方块序列。

    gray : HxW 灰度图 (numpy uint8)
    x, y : 第一个方块左上角坐标
    """
    import numpy as np

    n = 2 + bits
    h, w = gray.shape[:2]
    out: list[int] = []
    half = max(1, cell // 4)

    for i in range(n):
        cx = x + i * (cell + gap) + cell // 2
        cy = y + cell // 2
        if cy >= h or cx >= w:
            out.append(0)
            continue
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        patch = gray[y0:y1, x0:x1]
        out.append(1 if float(patch.mean()) > 128.0 else 0)
    return out


def decode_ms(gray, x: int, y: int, cell: int, gap: int, bits: int = DEFAULT_BITS) -> int | None:
    return bits_to_ms(read_bits(gray, x, y, cell, gap, bits), bits)


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
