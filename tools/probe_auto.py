"""自动定位时间码探针条（只搜 x0/y，周期取配置值）。

依据：已实测白块间距 == cell+gap，周期正确，仅起点有偏移。

用法:
    python -m tools.probe_auto
"""

import argparse

import cv2
import numpy as np

from link import PyAVSource
from tools.config import get
from tools.probe_codec import DEFAULT_BITS, bits_to_ms, decode_ms


def locate(gray, cell, gap, bits=DEFAULT_BITS, y_search=80, x_search=1400):
    """暴力搜索 (y, x0)，评分 = 采样点的黑白确信度。

    **四条约束缺一不可**
      1. 前两位必须是 1, 0（编码的起始标记）
      2. 必须是黑白相间 —— 只按"偏离灰度中点"打分的话，整片纯黑背景会拿满分
         （|0-128|=128），而真正的码是黑白混合、平均偏离只有 60~90。
      3. 黑白比例要平衡（全 0 或全 1 都不像码）
      4. **同一列上下两行必须给出相同的码**

    第 4 条是踩了坑才加的。前三条都满足却没找到真探针，因为画面左上角的
    游戏 UI（"彩虹岛"/地名那几行白字 + 深灰底）同样是"黑白分明"，
    评分与真探针完全相同，于是搜索停在了 UI 上（实测偏了 45 像素）。

    探针是 cell 像素高的实心横条，跨 y 方向的图案必然一致；
    而文字是逐行不同的笔画，这一条能干净地把两者分开。
    """
    p = cell + gap
    n = 2 + bits
    offsets = (cell // 2 + np.arange(n) * p).astype(np.int32)
    h, w = gray.shape[:2]
    cands = []

    def _bits_at(yy):
        row = gray[yy].astype(np.int32)
        return row, (row > 128)

    for y in range(0, min(y_search, h)):
        y2 = y + cell // 2
        if y2 >= h:
            break
        for x0 in range(0, min(x_search, w - offsets[-1])):
            v, b = _bits_at(y)
            ids = x0 + offsets
            v = v[ids]
            b = b[ids]

            if not (b[0] and not b[1]):
                continue

            white = float(b.mean())
            if white < 0.15 or white > 0.85:
                continue        # 排除全黑/全白：那不是码

            # 上下两行必须一致：探针是实心横条，UI 文字不是
            _v2, b2 = _bits_at(y2)
            if not np.array_equal(b, b2[ids]):
                continue

            cands.append((float(np.mean(np.abs(v - 128))), x0, y))

    if not cands:
        return None

    # 最后必须用**真正的解码路径**验证一遍。
    #
    # 单像素采样和 read_bits 的"小块均值"不是一回事：x0 偏半个方块时，
    # 单像素看仍是清清楚楚的黑或白，但 read_bits 取的 9x9 小块会有一半
    # 吃到背景，均值卡在 128 边界上 —— 结果是"搜到了位置却解不出码"。
    #
    # 而且光判断"解出了数"还不够：图案错位同样能解出个数，只是那是垃圾
    # （实测会得到 1.6e11 这种明显超过一天的数）。时间戳是"当日毫秒"，
    # 必须落在 [0, 86400000) 之内才算数。
    DAY_MS = 86400000

    # 按 conf 降序、只验前 N 个是错的：
    # 探针**左侧**的错位位置 conf 同样接近满分（都是黑白分明的像素），
    # 排序后 x0 小的会占满验证窗口，真正的位置根本轮不到
    # （实测真值 x0=152，而前 60 个候选全落在 x0<152 的错位点上）。
    #
    # 这里按扫描顺序（y 外层、x0 内层）逐个验证，第一个解出合法时间戳的
    # 就是最靠左的有效位置。容差 ±5 像素内都能解出同一个值，
    # 所以落在哪个具体 x0 上不影响结果。
    for conf, x0, y in cands[:5000]:
        ts = decode_ms(gray, x0, y, cell, gap, bits)
        if ts is not None and 0 <= ts < DAY_MS:
            return (conf, x0, y)

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--url", default=None)
    args = ap.parse_args()

    cell = get("probe", "cell", 16)
    gap = get("probe", "gap", 2)
    bits = get("probe", "bits", 40)

    src = PyAVSource(args.url or get("stream", "url"))
    src.open()
    f = None
    for _ in range(args.skip):
        f = src.read()
    src.close()
    if f is None:
        print("no frame")
        return 1

    gray = cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY)
    print("frame:", gray.shape, " cell=%s gap=%s bits=%d" % (cell, gap, bits))

    r = locate(gray, cell, gap, bits)
    if r is None:
        print("未找到探针条（前两位不是 10）")
        return 1

    conf, x0, y = r
    print("找到: x0=%d  y=%d  conf=%.1f   (配置 x=%d y=%d)"
          % (x0, y, conf, get("probe", "x", 100), get("probe", "y", 8)))

    p = cell + gap
    n = 2 + bits
    # 采样下标一律取整：gap 是 2.25（浮点），浮点下标会让 numpy 索引和
    # cv2 绘图同时报错。gap 可以带小数（更贴合实际像素间隔），
    # 但落到"取第几个像素"这一步必须是整数。
    w_img = gray.shape[1]
    idx = [min(w_img - 1, int(round(x0 + cell / 2.0 + i * p)))
           for i in range(n)]

    vals = [int(gray[y, ix]) for ix in idx]
    bstr = "".join("1" if v > 128 else "0" for v in vals)
    print("bits:", bstr)
    print("ts_ms:", bits_to_ms([int(v > 128) for v in vals], bits))

    vis = cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR)
    for i in range(n):
        color = (0, 255, 0) if vals[i] > 128 else (0, 0, 255)
        cv2.circle(vis, (idx[i], int(y)), 3, color, -1)

    x1 = int(round(x0 + n * p + 40))
    cv2.imwrite("data/probe_auto.png", vis[0:120, max(0, x0 - 40):x1])
    print("saved data/probe_auto.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
