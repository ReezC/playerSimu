"""**从一张现场截图裁出「本机渲染」的锚点变体** ⇒ `perception/ui_templates/<锚点>.<n>.png`。

为什么需要它（2026-10-17 ✓ 用户原话：「**本机的游戏窗口分辨率是1366x768，按这个走**」✓）：
  · `perception/ui_state.py` 的锚点表是按**基准分辨率 1920×1080** 量的 ✓，而判别前会把整帧
    **硬缩到 960×540**（`prepare` ✓）⇒ 它**假设"游戏 UI 随帧尺寸等比缩放"** ✗ ——
    而这个客户端的木牌元素几乎是**固定像素**的 ✗ ⇒ 换个窗口尺寸，缩到基准之后大小就对不上 ✗
    （实测：1366×768 那帧里「连接」按钮比老模板**大 ~1.33 倍** ⇒ 老模板判 **0.40** ✗，
      门限 0.80 ✓）。
  · ⛔ **运行时补尺度这条路走不通**（试过两版 ⇒ 见 `ui_state` 里那段记录 ✓）：
    `TM_CCOEFF_NORMED` 差 5% 尺度就掉 ~0.1 ✗ 要过门限得精确到 2~3% ✗✗。
  · ⭐ **正解 = 按本机那一帧重裁一份变体**（仓库 2026-10-10 实测：重裁后 **0.977** ✓、
    运行时**零代价** ✓、门限与判据**一个字不用改** ✓）。

用法（在工作台那台机器上跑 ✓ 截图就是"登录窗正在显示"的那一下 ✓）：
```
python -m tools.make_ui_variant --shot 截图.png                     # 自动认窗口尺寸
python -m tools.make_ui_variant --shot 截图.png --win 1366x768      # 也可以显式给
python -m tools.make_ui_variant --shot 截图.png --only login_connect
```

⚠ **它只裁模板、不动锚点坐标** ✗ —— 变体与主模板**共用同一个 `rect`**（登记位置 ±
  `SEARCH_MARGIN` ✓）。⚠ 所以截图里的目标必须落在那个搜索窗里 ✓：偏出去的话
  要么裁之前先调 `ANCHORS` 里的 `rect` ✓，要么把 `SEARCH_MARGIN` 放宽 ✓
  （2026-10-17 本机那颗按钮就是"刚好偏出去 35px"✗ ⇒ 已把 margin 60 → 100 ✓）。

⚠ **存盘要用 `BASE_W / 客户区宽` 这个系数**（不是"窗口/客户区"✗ —— 我第一版就写错了 ⇒
  存盘模板小 1.4 倍 ⇒ 打分 0.786 卡在门限下 ✗ 实测踩到 ✓）。推导：
    客户区渲染 →(×窗口/客户区 到窗口)→(×960/窗口 缩到基准)= ×(960/客户区宽) ✓
    而 `load_variants` 读盘时还会再乘 `MATCH_SCALE`(0.5) ✓
    ⇒ **存盘 = 客户区 × (960/客户区宽) / 0.5 = 客户区 × (1920/客户区宽)** ✓
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception import ui_state                      # noqa: E402

#: 默认要裁哪几块（⚠ 顺序 = 优先裁"必需锚点"✓：`login` 只要求 `login_connect` ✓，
#:   `login_agree` 是可选的两行协议字 ✓ —— 见 `ANCHORS` 里的 required 标记 ✓）
DEFAULT_NAMES = ("login_connect", "login_agree", "login_err_text")

#: 尺度扫描范围（相对**存盘尺度**的主模板 ✓）—— 0.62~0.92 覆盖"本机窗口比基准小"那一带 ✓
KS = [round(0.62 + 0.015 * i, 3) for i in range(21)]

#: 低于这个分就**别裁**（拿垃圾当模板比没有更糟 ✗ —— 会让别的界面误报 ✓）
MIN_CROP_SCORE = 0.75


def _client_area(img, win_w, win_h):
    """截图里的**窗口客户区**（截图常带标题条 ✓ 那张是 1080×632 = 客户区 1080×607 + 标题条 25 ✓）。"""
    h, w = img.shape[:2]
    ty = h - int(round(w * win_h / float(win_w)))
    if ty <= 0 or ty > h // 3:
        return img, 0                      # 认不出来就当没有标题条 ✓（后面靠尺度扫描兜 ✓）
    return img[ty:, :], ty


def make_variant(name, client, win_w, win_h, out_dir, dry=False):
    """裁一块 ⇒ 存成 `.<n>.png`（返回 (存盘路径, 分数, 客户区里的框, 放大系数) ✓）。"""
    tpls = ui_state.load_variants(name)
    if not tpls:
        print("  %-16s ✗ 没有主模板 ⇒ 跳过" % name)
        return None
    # 主模板还原到"存盘尺度"（load_variants 已经乘过 MATCH_SCALE ✓）
    base = cv2.resize(tpls[0], None, fx=1.0 / ui_state.MATCH_SCALE,
                      fy=1.0 / ui_state.MATCH_SCALE, interpolation=cv2.INTER_LINEAR)
    best = (-1.0, None, None)
    for k in KS:
        t = cv2.resize(base, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        if t.shape[0] > client.shape[0] or t.shape[1] > client.shape[1]:
            continue
        res = cv2.matchTemplate(client, t, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, ml = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), ml, (t.shape[1], t.shape[0]))
    score, loc, size = best
    print("  %-16s 客户区里最高分=%.3f @%s size=%s" % (name, score, loc, size))
    if loc is None or score < MIN_CROP_SCORE:
        print("  %-16s ⚠ 分太低 ⇒ **不裁**（宁缺毋滥 ✓）" % name)
        return None
    x, y = loc
    w, h = size
    k = ui_state.BASE_W / float(client.shape[1])            # 客户区 → 存盘尺度 ✓（见文件头推导 ✓）
    store = cv2.resize(client[y:y + h, x:x + w], None, fx=k, fy=k,
                       interpolation=cv2.INTER_LINEAR)
    # 找下一个空的变体编号（别覆盖已有的 ✓）
    n = 1
    while (out_dir / ("%s.%d.png" % (name, n))).exists():
        n += 1
    p = out_dir / ("%s.%d.png" % (name, n))
    if not dry:
        cv2.imwrite(str(p), store)
    print("  %-16s ⇒ %s %s（客户区 %dx%d ⇒ ×%.3f）"
          % (name, p.name, store.shape[:2][::-1], w, h, k))
    return p, score, (x, y, w, h), k


def main(argv=None):
    ap = argparse.ArgumentParser(description="从现场截图裁出本机渲染的锚点变体 ✓")
    ap.add_argument("--shot", required=True, help="现场截图（登录窗正在显示那一张 ✓）")
    ap.add_argument("--win", default=None, help="窗口尺寸，如 1366x768（默认 1366x768 ✓）")
    ap.add_argument("--only", default=None, help="只裁某一块锚点（逗号分隔 ✓）")
    ap.add_argument("--dry", action="store_true", help="只报分，不写盘 ✓")
    args = ap.parse_args(argv)

    win_w, win_h = 1366, 768
    if args.win:
        win_w, win_h = (int(v) for v in args.win.lower().split("x"))

    img = cv2.imdecode(np.fromfile(args.shot, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("读不到截图：%s" % args.shot)
        return 2
    client, ty = _client_area(img, win_w, win_h)
    print("截图 %dx%d ⇒ 客户区 %dx%d（切掉标题条 %d px ✓）"
          % (img.shape[1], img.shape[0], client.shape[1], client.shape[0], ty))

    names = tuple(s.strip() for s in args.only.split(",")) if args.only else DEFAULT_NAMES
    out_dir = ROOT / "perception" / "ui_templates"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        make_variant(name, client, win_w, win_h, out_dir, dry=args.dry)

    # 收尾：把客户区当成本机窗口喂一遍（与生产同口径 ✓）
    print("")
    print("=== 复核（客户区 ⇒ detect ✓ 应与生产一致）===")
    print("  detect() = %r" % (ui_state.detect(client),))
    per, hit = ui_state.scores(client)
    for n, s in sorted(per.items(), key=lambda kv: -kv[1])[:6]:
        print("    %-18s %.3f" % (n, s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
