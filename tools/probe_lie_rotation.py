"""实测两条新通道的数据可行性（用户 2026-09-30 决策 ①：先出数再重构）。

**关键口径**（第一版实测暴露后修正 ✓）：单帧姿态/位移都是**噪声盖过信号** ✗（角步进中位
14.6°/帧 ✗、框中心中位抖动 16.5px ✗）⇒ 必须用**抗噪估计器**：

· **旋转身份**（用户事实 3："假群只平移、只有真目标不断旋转"）：
  框内分割图形 ⇒ 掩码的 **PCA 主轴角**；判据不是"单帧步进大" ✗ 而是**窗口内的单向性**
  `|净转角| / Σ|角步进|`（→1 = 真在转 ✓；噪声 ≈ 1/√N → 0 ✗）✓。
· **刚体平移** `T_t`（事实 2/3）：**逐框模板匹配**（用像素块求平移 ✓ 不用框中心差 ✗）
  ⇒ 取**中位** = 群体平移 ✓；每框相对中位的**残差** ⇒ 真目标应是唯一的大残差 ✓。
· **互证**：残差最大的框 是否 = 单向转角最大的框 ✓✓。

用法：
    python -m tools.probe_lie_rotation --cycles 4
    python -m tools.probe_lie_rotation --cycles 2 --draw
"""
from __future__ import annotations

import argparse
import math
import pathlib
import statistics as st
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.lie_demo import DetsWorker, load_gif  # noqa: E402

_MIN_AREA_FRAC = 0.12
_MIN_ELONG = 1.03
_SEARCH = 44              # 模板匹配搜索半径（px；素材位移 ~20px/帧 ⇒ 够 ✓）
_WIN = 12                 # 单向性判据的窗口帧数
_UNWRAP_GATE = 40.0       # 逐帧角展开的门（|Δ| > 它 ⇒ 认为跳变、不计入 ✓）


def shape_of(patch):
    """框内图形 ⇒ `(占框比, 主轴角[0,180), 伸长率, 实心度)`；失败 ⇒ None。

    只用**框内部像素**（姿态 ✓），不碰框中心 ✗。掩码取"亮于背景"和"暗于背景"两种
    极性里更实心的那个 ✓（素材极性不统一 ✓ 实测）。
    """
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        return None
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    med = float(np.median(g))
    span = float(g.max()) - float(g.min())
    best = None
    for sign in (+1.0, -1.0):
        m = (g > med + 0.3 * span).astype(np.uint8) * 255 if sign > 0 else \
            (g < med - 0.3 * span).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        found = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = found[0] if len(found) == 2 else found[1]
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        a = float(cv2.contourArea(c))
        if a < _MIN_AREA_FRAC * patch.shape[0] * patch.shape[1] or len(c) < 5:
            continue
        mu = cv2.moments(c)
        if mu["m00"] <= 0:
            continue
        u20, u02, u11 = mu["mu20"] / mu["m00"], mu["mu02"] / mu["m00"], mu["mu11"] / mu["m00"]
        th = 0.5 * math.atan2(2 * u11, (u20 - u02))          # PCA 主轴 ✓
        l1 = 0.5 * (u20 + u02) + 0.5 * math.hypot(u20 - u02, 2 * u11)
        l2 = 0.5 * (u20 + u02) - 0.5 * math.hypot(u20 - u02, 2 * u11)
        elong = math.sqrt(l1 / max(1e-6, l2))
        if elong < _MIN_ELONG:
            continue
        (_, _), (w_, h_), _ = cv2.fitEllipse(c)
        solid = a / max(1.0, w_ * h_ * math.pi / 4.0)
        cand = (a / (patch.shape[0] * patch.shape[1]), math.degrees(th) % 180.0,
                elong, solid)
        if best is None or cand[3] > best[3]:
            best = cand
    return best


def angle_step(a1, a2):
    d = abs(a1 - a2) % 180.0
    return min(d, 180.0 - d)


def signed_step(a1, a2):
    """带符号的无向角步进 ∈ (-90, 90]（用于"单向性"判据 ✓）。"""
    d = (a2 - a1) % 180.0
    if d > 90.0:
        d -= 180.0
    return d


def load_cycle(d):
    """photo1 cycle ⇒ (frames, ts)（去重 + 时间戳单调化 ✓）。"""
    fs = sorted(p for p in d.iterdir()
                if p.suffix.lower() in (".bmp", ".png", ".jpg"))
    out, ts, prev = [], [], None
    for p in fs:
        try:
            t = float(p.stem.split("_")[-1].rstrip("s"))
        except ValueError:
            t = None
        im = cv2.imread(str(p))
        if im is None:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if prev is not None and float(cv2.absdiff(g, prev).mean()) < 0.5:
            continue
        prev = g
        ts.append(t)
        out.append(im)
    if len(out) < 2:
        return out, [0.0]
    mono = all(ts[i] is not None and ts[i] < ts[i + 1] for i in range(len(ts) - 1))
    return out, (ts if mono else [float(i) for i in range(len(out))])


def patch_of(im, cx, cy, w, h, pad=4):
    x0, y0 = max(0, int(cx - w / 2) - pad), max(0, int(cy - h / 2) - pad)
    x1 = min(im.shape[1], int(cx + w / 2) + pad)
    y1 = min(im.shape[0], int(cy + h / 2) + pad)
    return im[y0:y1, x0:x1], (x0, y0)


def match_shift(prev_im, cur_im, cx, cy, w, h):
    """**逐框模板匹配**求该框这一帧的平移 `(dx, dy)`（亚像素 ✓）；失败 ⇒ None。

    比"框中心相减"强得多 ✓：框中心会被**合并/分离/漏检**带偏 ✗，而像素块的相关峰
    只看"这块图案往哪挪了" ✓。
    """
    tpl, (tx, ty) = patch_of(prev_im, cx, cy, w, h)
    if tpl.size == 0 or min(tpl.shape[:2]) < 6:
        return None
    x0 = max(0, tx - _SEARCH)
    y0 = max(0, ty - _SEARCH)
    x1 = min(cur_im.shape[1], tx + tpl.shape[1] + _SEARCH)
    y1 = min(cur_im.shape[0], ty + tpl.shape[0] + _SEARCH)
    if x1 - x0 < tpl.shape[1] or y1 - y0 < tpl.shape[0]:
        return None
    res = cv2.matchTemplate(cur_im[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    if mx < 0.5:                       # 相关太弱 ⇒ 别当位移用 ✗
        return None
    # 亚像素：在峰附近做抛物线拟合 ✓
    px, py = loc
    dx, dy = float(px), float(py)
    if 0 < px < res.shape[1] - 1:
        a, b, c = res[py, px - 1], res[py, px], res[py, px + 1]
        den = (a - 2 * b + c)
        if abs(den) > 1e-6:
            dx += 0.5 * (a - c) / den
    if 0 < py < res.shape[0] - 1:
        a, b, c = res[py - 1, px], res[py, px], res[py + 1, px]
        den = (a - 2 * b + c)
        if abs(den) > 1e-6:
            dy += 0.5 * (a - c) / den
    return (x0 + dx - tx, y0 + dy - ty)


def analyze(name, frames, worker, draw=None):
    n = len(frames)
    if n < 6:
        print("\n===== %s ===== 帧太少（%d）跳过" % (name, n))
        return
    dets = [worker.detect(f) for f in frames]
    typ = st.median([w * h for f in range(n) for b in dets[f]
                     for w, h in [(float(b[3]), float(b[4]))]])
    # ---- 每帧每框：形状 + 相对上一帧的模板匹配位移 ----
    rows = []          # [(fid, cx, cy, w, h, shape)]
    shifts = []        # [(fid, cx, cy, dx, dy, ok)]
    for fid in range(n):
        for b in dets[fid]:
            cx, cy, w, h = (float(b[1]), float(b[2]), float(b[3]), float(b[4]))
            sh = shape_of(patch_of(frames[fid], cx, cy, w, h)[0])
            rows.append((fid, cx, cy, w, h, sh))
            if fid:
                mv = match_shift(frames[fid - 1], frames[fid], cx, cy, w, h)
                shifts.append((fid, cx, cy, 0.0 if mv is None else mv[0],
                               0.0 if mv is None else mv[1], mv is not None))
    # ---- 刚体平移 + 残差（只信"面积正常 + 匹配成功"的框 ✓）----
    tvec, resid = {}, {}
    for fid in range(1, n):
        ds = [(cx, cy, dx, dy) for f, cx, cy, dx, dy, ok in shifts
              if f == fid and ok]
        keep = [(cx, cy, dx, dy) for cx, cy, dx, dy in ds]
        if not keep:
            continue
        tvec[fid] = (st.median([k[2] for k in keep]), st.median([k[3] for k in keep]))
        for cx, cy, dx, dy in keep:
            resid[(fid, round(cx), round(cy))] = math.hypot(
                dx - tvec[fid][0], dy - tvec[fid][1])
    # ---- 旋转：**窗口单向性**（|净转角| / Σ|步进| ✓ 抗噪 ✓）----
    by_id = {}
    for fid, cx, cy, w, h, sh in rows:
        if sh is not None and w * h <= 1.4 * typ:
            by_id.setdefault((round(cx / 40), round(cy / 40)), []).append((fid, sh[1], (cx, cy)))
    rot = {}
    for key, rec in by_id.items():
        rec.sort()
        net = tot = 0.0
        for k in range(1, len(rec)):
            if rec[k][0] - rec[k - 1][0] > 2:
                continue
            d = signed_step(rec[k - 1][1], rec[k][1])
            if abs(d) > _UNWRAP_GATE:
                continue
            net += d
            tot += abs(d)
        if tot >= 30.0 and len(rec) >= _WIN:
            rot[key] = (abs(net) / tot, net, tot, rec[-1][2])
    print("\n===== %s =====" % name)
    print("帧数(去重) %d ｜ 每帧框数中位 %d ｜ 框面积中位 %.0f ｜ 出角 %d/%d"
          % (n, int(st.median(len(d) for d in dets)), typ, len(rows), len(rows)))
    if tvec:
        tm = sorted(math.hypot(*v) for v in tvec.values())
        jj = [math.hypot(tvec[f][0] - tvec[f - 1][0], tvec[f][1] - tvec[f - 1][1])
              for f in sorted(tvec) if f - 1 in tvec]
        rs = sorted(resid.values())
        print("② 刚体平移 T（逐框模板匹配中位）：|T| 中位 %.2f px/帧 ｜ p90 %.2f ｜ "
              "帧间抖动中位 %.2f px" % (tm[len(tm) // 2], tm[int(len(tm) * 0.9)],
                                        st.median(jj) if jj else 0.0))
        print("   各框残差 |位移−T|：中位 %.2f ｜ p90 %.2f ｜ 最大 %.2f px（真目标应突出 ✓）"
              % (rs[len(rs) // 2], rs[int(len(rs) * 0.9)], rs[-1]))
    if rot:
        top = sorted(rot.items(), key=lambda kv: -kv[1][0])[:4]
        print("① 窗口单向性 |净角|/Σ|步进|（→1 才是真在转 ✓ 噪声→0 ✗）：")
        for key, (q, net, tot, pos) in top:
            print("   框(%.0f,%.0f)：单向性 %.2f ｜ 净转角 %+.0f° / 总步进 %.0f°"
                  % (pos[0], pos[1], q, net, tot))
        bi = max(rot.items(), key=lambda kv: kv[1][0])
        print("   最强 = (%.0f,%.0f) 单向性 %.2f 净 %+.0f°"
              % (bi[1][3][0], bi[1][3][1], bi[1][0], bi[1][1]))
    else:
        print("① 旋转：没有框积累到判据门槛（%d° 总步进）⇒ 样本内看不出 ✓/✗" % 30)
    if draw:
        fid = min(n - 1, int(n * 0.6))
        im = frames[fid].copy()
        for f, cx, cy, w, h, sh in rows:
            if f != fid:
                continue
            cv2.rectangle(im, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), (0, 255, 0), 1)
            if sh:
                a = math.radians(sh[1])
                L = max(w, h) * 0.45
                cv2.line(im, (int(cx - math.cos(a) * L), int(cy - math.sin(a) * L)),
                         (int(cx + math.cos(a) * L), int(cy + math.sin(a) * L)),
                         (0, 0, 255), 2)
                cv2.putText(im, "%.0f|%.1f" % (sh[1], sh[2]),
                            (int(cx - w / 2), int(cy - h / 2) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        pathlib.Path("_tmp_probe").mkdir(exist_ok=True)
        outp = pathlib.Path("_tmp_probe") / (draw + ".png")
        cv2.imwrite(str(outp), im)
        print("   调试图：%s" % outp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--gif", default="datasets/liedetectorgifs/test.gif")
    ap.add_argument("--draw", action="store_true")
    a = ap.parse_args()
    w = DetsWorker()
    root = pathlib.Path("datasets/photo1")
    dirs = sorted(d for d in root.iterdir() if d.is_dir())[a.skip:a.skip + a.cycles]
    for d in dirs:
        fr, _ = load_cycle(d)
        analyze(d.name, fr, w, draw=d.name if a.draw else None)
    if a.gif and pathlib.Path(a.gif).exists():
        raw, _, _, _, _ = load_gif(a.gif)
        analyze(pathlib.Path(a.gif).name, [f[0] for f in raw], w,
                draw="gif" if a.draw else None)
    w.close()


if __name__ == "__main__":
    main()
