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
from tools.probe_codec import (DEFAULT_BITS, bits_to_ms, decode_ms, now_ms,
                               ts_plausible)


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
    p = float(cell) + float(gap)
    n = 2 + bits
    # cell / 2 必须是**真除**：cell // 2 对浮点返回浮点（18.0 // 2 = 9.0），
    # 拿去索引会抛 IndexError: only integers ... are valid indices。
    # 取整放到最后一步做。
    half_cell = float(cell) / 2.0
    offsets = np.round(half_cell + np.arange(n) * p).astype(np.int32)
    h, w = gray.shape[:2]
    cands = []

    def _bits_at(yy):
        row = gray[yy].astype(np.int32)
        return row, (row > 128)

    for y in range(0, min(y_search, h)):
        y2 = y + int(round(half_cell))
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


def measure(gray, bits=DEFAULT_BITS, max_y=140):
    """从画面里量出探针条带的像素几何（不依赖任何配置）。

    返回 (top, bottom, height, cell) 或 None。

    依据：探针是「纯黑底 + 纯黑/纯白方块」的横条，同一行里**同时**出现
    接近 0 和接近 255 的像素。游戏画面虽然花，但同时有大量纯黑和纯白
    的行极少，所以这个判据比"行方差大"干净得多。

    条带高 h = cell + 2*gap（上下各留一个 gap 的边距，probe_gen 里
    create_rectangle 的 y 从 gap 起、到 gap+cell 止）。gap 约为 cell/8，
    所以 cell ≈ h / 1.25。
    """
    h_img, w_img = gray.shape[:2]
    top_lim = min(max_y, h_img)

    black = gray[:top_lim] < 40
    white = gray[:top_lim] > 215
    # 每行要求有足够的纯黑和纯白像素
    rows = [y for y in range(top_lim)
            if black[y].sum() > 20 and white[y].sum() > 20]
    if not rows:
        return None

    # 取连续最长的一段，避免把画面里零散的纯黑/纯白也算进来
    best = cur = [rows[0]]
    for y in rows[1:]:
        if y - cur[-1] <= 2:
            cur.append(y)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [y]
    if len(cur) > len(best):
        best = cur

    top, bottom = best[0], best[-1]
    h = bottom - top + 1

    # 用翻转间距交叉验证 cell：在条带中间行找黑白翻转点
    mid = (top + bottom) // 2
    row = gray[mid].astype(np.int32)
    tr = np.where(np.abs(np.diff(row)) > 100)[0]
    cell_guess = h / 1.25
    if len(tr) > 4:
        d = np.diff(tr)
        cand = sorted(x for x in d if 4 <= x <= h)
        if cand:
            # 翻转间距里**同时含 cell 和 gap**（黑→白是 cell，白→黑是 gap，
            # 或反之）。cell 明显更大且数量占多数，所以取高分位。
            #
            # 这里曾经取低分位（median(cand[:len//3])），结果量出的全是 gap，
            # 推算出的 cell 只有真实值的三分之一，反而把人带偏。
            cell_guess = float(np.percentile(cand, 80))

    return top, bottom, h, cell_guess


def _purity(gray, x, y, cell, gap, bits):
    """采样点小块的灰度标准差均值 —— 判定步长对不对的关键指标。

    真探针的每个采样点都落在方块**内部**，那一小块是纯色，std 接近 0
    （视频压缩会带来一点波动，通常 < 20）。

    步长一旦错位，采样点就压在方块**边界**上，小块里一半黑一半白，
    std 立刻升到 60~120。

    这是唯一能把「步长正确」和「凑巧凑出个合法数」分开的指标：
    单帧看"能否解码"、多帧看"是否递增"都会被错位欺骗 ——
    因为时间戳高位在几帧内不变，错位解出的低位序列照样递增。
    """
    half = max(1, int(round(cell / 6.0)))
    cy = int(round(y + cell / 2.0))
    stds = []
    for i in range(2 + bits):
        cx = int(round(x + i * (cell + gap) + cell / 2.0))
        p = gray[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        if p.size:
            stds.append(float(p.std()))
    return sum(stds) / len(stds) if stds else 999.0


def scan_edges(gray, y, thresh=90):
    """在给定行上找黑白翻转点，返回 (edges, gaps, span)。

    这是**直接测量**，不依赖任何配置：探针是一串黑白方块，行扫描线上的
    灰度会在 0/255 之间反复翻转；翻转点的间距就是 cell 或 gap。
    """
    row = gray[y].astype(np.int32)
    edges = np.where(np.abs(np.diff(row)) > thresh)[0]
    if len(edges) < 6:
        return edges, np.array([], dtype=np.int32), None

    # 只保留"连续交替"的一段：探针内部相邻翻转的间隔必然较小，
    # 而画面其它地方的翻转之间隔着大片同色区域。
    best = cur = [edges[0]]
    for e in edges[1:]:
        if e - cur[-1] <= 60:
            cur.append(e)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [e]
    if len(cur) > len(best):
        best = cur

    if len(best) < 6:
        return edges, np.array([], dtype=np.int32), None

    b = np.asarray(best, dtype=np.int32)
    return b, np.diff(b), (int(b[0]), int(b[-1]))


def solve(gray, bits=DEFAULT_BITS, verbose=True):
    """**直接解出**探针的 cell/gap/x/y，不做参数搜索。

    根据只有两个方程、两个未知数：

        条带高 h = cell + 2*gap        (probe_gen 在上下各留一个 gap 的边距)
        flip    = cell + gap           (行扫描线上相邻翻转的间隔)

    两式相减直接得到 gap = h - flip，cell = flip - gap。

    这比"逐组试 cell/gap、看能不能解码"可靠得多 ——
    后者会被"步长均匀偏小"骗过：每步少一点点时采样点仍落在方块内部，
    解码正常、纯度正常、多帧还递增，所有间接判据全部通过，
    但探针总长算出来是短的（表现为标注框右边够不着）。
    **只有量长度能发现，那就该量长度。**
    """
    m = measure(gray, bits)
    if not m:
        return None
    top, bottom, h, _ = m

    # 在条带中间行量翻转间隔
    mid = (top + bottom) // 2
    row = gray[mid].astype(np.int32)
    edges = np.where(np.abs(np.diff(row)) > 90)[0]
    if len(edges) < 8:
        return None

    # 只保留连续交替的一段（探针内部），排除画面其它地方的零散翻转
    best = cur = [edges[0]]
    for e in edges[1:]:
        if e - cur[-1] <= 50:
            cur.append(e)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [e]
    if len(cur) > len(best):
        best = cur
    if len(best) < 8:
        return None

    b = np.asarray(best)
    gaps = np.diff(b)
    # 排除"相邻同色方块"造成的整数倍大间距（2*(cell+gap) 及以上）
    gaps = gaps[gaps <= np.percentile(gaps, 90)]
    if len(gaps) < 4:
        return None

    med = float(np.median(gaps))

    # 间距本来就是整数像素，直接做直方图。
    #
    # ⚠ 这里不能用"出现次数最多"的值当基准 —— 实测直方图可能是
    #     1×4  2×5  17×2  18×5
    #   `2` 和 `18` 次数一样，argsort 取到 2 之后，所有候选都变成
    #   (2-gap, gap)，cell 接近 0 被全部过滤，最后报"没有组合"。
    #
    # 结构上：小档是 gap（每个方块交界两侧各出现一次，天然偏多），
    # 大档才是 cell 或 cell+gap。所以基准要取**大档**，别看次数。
    gi = np.round(gaps).astype(int)
    vals, counts = np.unique(gi, return_counts=True)
    hist_map = dict(zip(vals.tolist(), counts.tolist()))

    # 大档：取最大的、且出现不止一次的值（单次出现多半是杂散边缘）
    big_v = None
    for v in sorted(vals, reverse=True):
        if hist_map[int(v)] >= 2:
            big_v = int(v)
            break
    if big_v is None:
        big_v = int(round(med))

    # 小档：大档的一半以下里，出现次数最多的那个
    small_v = None
    if big_v:
        cand_small = [(int(v), int(hist_map[int(v)])) for v in vals
                      if v <= big_v * 0.5]
        if cand_small:
            small_v = max(cand_small, key=lambda t: t[1])[0]

    if verbose:
        hist = "  ".join("%d×%d" % (v, c) for v, c in zip(vals, counts))
        print("  条带: y[%d..%d] 高 %d" % (top, bottom, h))
        print("  翻转: %d 个, 间距直方图(值×次数): %s" % (len(b), hist))
        print("  大档=%s  小档=%s  (大档≈cell 或 cell+gap，小档≈gap)"
              % (big_v, small_v))

    # 生成候选 (cell, gap)，逐个用「时间戳是否接近此刻」判定。
    #
    # 两个方程（条带高 h、间距档位）推不出唯一解 —— 实测 16/2、18/1.4、
    # 15/2.5 都能自洽。所以不再纠结怎么推，直接列出所有合理组合，
    # 让 ts_plausible 一票定音：错位解出的时间戳是随机数，
    # 几乎不可能落在"此刻"附近。
    cands = []
    if small_v is not None:
        cands.append((float(big_v), float(small_v)))     # 小档=gap, 大档=cell
    # 用条带高反解：h = cell + 2*gap
    for gap in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5):
        cands.append((h - 2 * gap, gap))                 # 用条带高
        cands.append((big_v - gap, gap))                 # 大档 = cell+gap
        cands.append((float(big_v), gap))                # 大档 = cell

    if verbose:
        print("  候选组合（%d 组），用「时间戳是否接近此刻」逐一定夺:" % len(cands))
        print("  此刻 = %d (当天毫秒)" % now_ms())

    seen = set()
    for cell, gap in cands:
        if not (4 <= cell <= 60) or not (0.3 <= gap < cell):
            continue
        key = (round(cell, 2), round(gap, 2))
        if key in seen:
            continue
        seen.add(key)

        r = locate(gray, cell, gap, bits)
        if r is None:
            if verbose:
                print("    cell=%-6.2f gap=%-6.3f  未定位到" % (cell, gap))
            continue
        conf, x0, y = r
        ts = decode_ms(gray, x0, y, cell, gap, bits)
        ok = ts_plausible(ts)
        if verbose:
            print("    cell=%-6.2f gap=%-6.3f -> x0=%-5d y=%-3d ts=%-10s %s"
                  % (cell, gap, x0, y, ts, "✓ 接近此刻" if ok else ""))
        if ok:
            return {"top": top, "bottom": bottom, "h": h, "cell": cell,
                    "gap": gap, "x": x0, "y": y, "conf": conf, "ts": ts,
                    "span": int(b[-1] - b[0])}

    if verbose:
        print("  没有组合能解出「接近此刻」的时间戳。")
        print("  检查：A 机 probe_gen 在运行？双机时钟是否偏差过大？")
    return None


def calibrate(frames, bits=DEFAULT_BITS, verbose=True):
    """在**多帧**上试若干组 (cell, gap)，用「时间戳单调递增」判定正确解。

    **为什么要试而不是算**：cell/gap 只靠图像量有误差（压缩会让方块
    边缘外扩），而解码对步长极其敏感 —— 步长错 0.5px，41 个方块后就
    偏 20px，后半段全采错。

    **为什么必须多帧**：单帧下「能否解出合法时间戳」几乎无法判定正确性 ——
    图案错位照样能凑出一个落在 [0, 86400000) 里的数。实测同一帧上
    cell=13.0/13.5/14.0/16.0 全都"合法"，但解出的值差了三个数量级，
    显然只有一组是真的。

    可靠判据只有一个：**时间在前进**。同一组参数若在连续多帧上都能
    解出合法时间戳、且数值单调不减，那它几乎必然是对的 —— 错位的采样
    不可能连续多帧都凑出递增序列。

    frames : 灰度图列表（至少 2 帧，越多越可靠）
    """
    DAY_MS = 86400000

    cands = [
        (18.0, 2.25),      # 屏幕 cell=24 gap=3，经 0.75 缩放
        (24.0, 3.0),       # 未缩放，屏幕 cell=24
        (13.5, 1.6875),    # 屏幕 cell=18 gap=2.25，经 0.75 缩放
        (16.0, 2.0),       # 未缩放，屏幕 cell=16
        (20.0, 2.5),
        (13.0, 1.6),
        (14.0, 1.75),
        (12.0, 1.5),
    ]

    hits = []
    for cell, gap in cands:
        rows = []
        for g in frames:
            r = locate(g, cell, gap, bits)
            if r is None:
                rows.append(None)
                continue
            conf, x0, y = r
            ts = decode_ms(g, x0, y, cell, gap, bits)
            rows.append((x0, y, conf, ts))

        got = [r for r in rows if r and r[3] is not None and 0 <= r[3] < DAY_MS]
        all_ok = len(got) == len(frames) and len(frames) >= 2

        rising = False
        if all_ok:
            tss = [r[3] for r in got]
            rising = all(tss[i] <= tss[i + 1] for i in range(len(tss) - 1))

        # 位置稳定度：各帧定位到的 x0/y 应当基本一致
        stable = False
        if all_ok:
            xs = [r[0] for r in got]
            ys = [r[1] for r in got]
            stable = (max(xs) - min(xs) <= 4) and (max(ys) - min(ys) <= 4)

        # 采样点纯度：取各帧里最差的一帧（越稳定越可信）
        pur = 999.0
        if all_ok:
            pur = max(_purity(frames[i], rows[i][0], rows[i][1], cell, gap, bits)
                      for i in range(len(frames)))

        hits.append({"cell": cell, "gap": gap, "rows": rows, "purity": pur,
                     "all_ok": all_ok, "rising": rising, "stable": stable})

        if verbose:
            if all_ok and rising and stable:
                flag = "✓ 递增稳定"
            elif all_ok and rising:
                flag = "递增"
            elif all_ok:
                flag = "合法"
            else:
                flag = "未通过"
            sample = got[0][3] if got else None
            print("  cell=%-6s gap=%-8s %-12s 纯度=%6.1f  (%d/%d 帧, 样本 ts=%s)"
                  % (cell, gap, flag, pur, len(got), len(frames), sample))

    return hits


def _report(hits, bits):
    """打印可直接填进 config/link.yaml 的建议值。"""
    good = [h for h in hits if h["all_ok"] and h["rising"]]
    if not good:
        print()
        print("没有一组候选能连续多帧解出递增时间戳。")
        print("可能原因：画面里没有探针 / 探针被遮挡 / 抓到的帧太少或重复。")
        print("试试加大 --frames，或确认 A 机 probe_gen 正在运行。")
        return

    # 按采样点纯度排序 —— 纯度越低说明采样点越落在方块内部，位置越准。
    # 前面的"能否解码""是否递增"只能筛掉明显错的，真正区分好坏要看这个。
    good.sort(key=lambda h: h["purity"])

    print()
    print("=" * 58)
    print("候选（按采样点纯度排序，越小越可信）：")
    for h in good:
        rows = [r for r in h["rows"] if r]
        x0 = int(round(sum(r[0] for r in rows) / len(rows)))
        y = int(round(sum(r[1] for r in rows) / len(rows)))
        print("  纯度=%6.1f  cell=%-6s gap=%-8s -> x0≈%-5d y≈%-4d"
              % (h["purity"], h["cell"], h["gap"], x0, y))

    h = good[0]
    rows = [r for r in h["rows"] if r]
    x0 = int(round(sum(r[0] for r in rows) / len(rows)))
    y = int(round(sum(r[1] for r in rows) / len(rows)))
    print()
    print("建议写入 config/link.yaml 的 probe 段：")
    print("  probe:")
    print("    x: %d" % x0)
    print("    y: %d" % y)
    print("    cell: %s" % (int(h["cell"]) if float(h["cell"]).is_integer()
                            else h["cell"]))
    print("    gap: %s" % h["gap"])
    print("    bits: %d" % bits)
    print()
    print("（纯度应明显低于其它候选；若几个候选纯度接近，说明量化误差大，")
    print("  可适当加大 --frames 多抓几帧再定）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=30)
    ap.add_argument("--url", default=None)
    ap.add_argument("--file", default=None,
                    help="分析视频或图片文件（图片只用作单帧参考，标定需要视频/实时流）")
    ap.add_argument("--format", default=None,
                    help="强制输入格式。裸流必须指定（如 ffmpeg -f mjpeg 推出来的）")
    ap.add_argument("--calibrate", action="store_true",
                    help="自动标定 cell/gap/x/y 并给出建议配置")
    ap.add_argument("--solve", action="store_true",
                    help="从画面几何直接解出 cell/gap/x/y（推荐，比 --calibrate 可靠）")
    ap.add_argument("--frames", type=int, default=6,
                    help="标定时抓取帧数（越多越可靠，但耗时线性增长）")
    args = ap.parse_args()

    cell = get("probe", "cell", 16)
    gap = get("probe", "gap", 2)
    bits = get("probe", "bits", 40)

    frames = []          # 灰度图列表
    rgb = None

    if args.file:
        low = args.file.lower()
        if low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            buf = np.fromfile(args.file, dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is None:
                print("读不到图片:", args.file)
                return 1
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            frames = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)]
        else:
            from link import FileSource

            src = FileSource(args.file)
            src.open()
            for _ in range(max(1, args.frames)):
                f = src.read()
                if f is None:
                    break
                rgb = f.image
                frames.append(cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY))
            src.close()
    else:
        src = PyAVSource(args.url or get("stream", "url"),
                         container_format=args.format)
        src.open()

        # 探针参数不按流分辨率缩放（和 probe_recv 一致）：配置里的
        # x/y/cell/gap 就是「流画面里的值」，由 A 机 out_scale 保证。
        _w = (src.size or (0, 0))[0]
        if _w and abs(_w - 1366) > 1:
            print("⚠ 流宽 %d != 预期 1366，探针参数可能不匹配" % _w)

        for _ in range(args.skip):
            src.read()
        for _ in range(max(1, args.frames)):
            f = src.read()
            if f is None:
                break
            rgb = f.image
            frames.append(cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY))
        src.close()

    if not frames:
        print("no frame")
        return 1

    gray = frames[0]
    print("frame: %s  取到 %d 帧   配置 cell=%s gap=%s bits=%d"
          % (gray.shape, len(frames), cell, gap, bits))

    # 先量条带，给个参考
    m = measure(gray, bits)
    if m:
        top, bottom, h, cg = m
        print("探针条带: y[%d..%d] 高 %d  -> 推算 cell≈%.1f gap≈%.1f"
              % (top, bottom, h, cg, cg / 8.0))
        print("          （配置里 cell=%s gap=%s）" % (cell, gap))
    else:
        print("未在画面顶部找到探针条带")

    if args.solve:
        print()
        print("直接求解（两个方程两个未知数）...")
        s = solve(gray, bits)
        if s is None:
            print("求解失败：没找到条带，或翻转点不足")
            return 1
        n_all = 2 + int(bits)

        def _num(v):
            return int(round(v)) if abs(v - round(v)) < 0.05 else round(v, 3)

        print()
        print("=" * 58)
        print("建议写入 config/link.yaml 的 probe 段：")
        print("  probe:")
        print("    x: %d" % s["x"])
        print("    y: %d" % s["y"])
        print("    cell: %s" % _num(s["cell"]))
        print("    gap: %s" % _num(s["gap"]))
        print("    bits: %d" % bits)
        print()
        print("探针总长 = %d × (%s + %s) = %.0f px"
              % (n_all, _num(s["cell"]), _num(s["gap"]),
                 n_all * (s["cell"] + s["gap"])))
        return 0

    if args.calibrate:
        if len(frames) < 2:
            print()
            print("⚠ 只有 %d 帧，无法用「时间戳递增」判定正确解。" % len(frames))
            print("  单帧下多组参数都可能凑出合法值，结论不可靠。")
            print("  请改用视频文件或实时流重跑（默认会抓 --frames 帧）。")
            return 1
        print()
        print("自动标定中（%d 帧 × 逐个 cell/gap）..." % len(frames))
        hits = calibrate(frames, bits)
        _report(hits, bits)
        return 0

    # ---- 几何测量：直接量探针的实际像素尺寸，不依赖配置 ----
    n_bits = 2 + int(bits)
    cfg_x, cfg_y = get("probe", "x", 100), get("probe", "y", 8)
    cfg_p = float(cell) + float(gap)
    print()
    print("几何测量（按当前配置的 y 取样行）:")
    yy = int(round(float(cfg_y) + float(cell) / 2.0))
    edges, gaps, span = scan_edges(gray, yy)
    if span:
        x_a, x_b = span
        print("  行 y=%d: 翻转点 %d 个，范围 x[%d..%d]  全长 %d px"
              % (yy, len(edges), x_a, x_b, x_b - x_a))
        print("  翻转间距: 最小 %d  中位 %.1f  最大 %d"
              % (gaps.min(), float(np.median(gaps)), gaps.max()))
        # 探针有 2+bits 个方块，相邻同色方块之间没有翻转，
        # 所以翻转点数量 ≤ 2*n。全长 ≈ (n-1)*(cell+gap) + cell
        print("  按配置算的总长: %d px  (n=%d, cell=%s, gap=%s)"
              % (int(round((n_bits - 1) * cfg_p + float(cell))), n_bits, cell, gap))
        print("  实测/理论 = %.3f" % ((x_b - x_a) / max(1.0, (n_bits - 1) * cfg_p + float(cell))))
        # 反推：假设 gap = cell/8，用实测全长求 cell
        span_len = x_b - x_a
        cell_est = span_len / ((n_bits - 1) * 1.125)
        print("  若 gap=cell/8，由全长反推 cell≈%.2f  (gap≈%.2f)"
              % (cell_est, cell_est / 8.0))
    else:
        print("  行 y=%d 上没找到连续翻转（探针不在这一行？配置 y 需调整）" % yy)

    print()
    r = locate(gray, cell, gap, bits)
    if r is None:
        print("按当前配置未找到探针条 —— 试试 --calibrate")
        return 1

    conf, x0, y = r
    print("找到: x0=%d  y=%d  conf=%.1f   (配置 x=%s y=%s)"
          % (x0, y, conf, cfg_x, cfg_y))

    ts = decode_ms(gray, x0, y, cell, gap, bits)
    print("解码: ts_ms = %s %s"
          % (ts, "OK" if ts is not None else "失败（按当前配置）"))

    p = cell + gap
    n = 2 + bits
    w_img = gray.shape[1]
    idx = [min(w_img - 1, int(round(x0 + cell / 2.0 + i * p)))
           for i in range(n)]
    vals = [int(gray[y, ix]) for ix in idx]

    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for i in range(n):
        color = (0, 255, 0) if vals[i] > 128 else (0, 0, 255)
        cv2.circle(vis, (idx[i], int(y)), 3, color, -1)

    x1 = int(round(x0 + n * p + 40))
    ok, enc = cv2.imencode(".png", vis[0:120, max(0, x0 - 40):x1])
    if ok:
        np.asarray(enc).tofile("data/probe_auto.png")
        print("saved data/probe_auto.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
