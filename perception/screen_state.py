"""界面状态（测谎/防挂机弹窗）检测 —— M1「测谎报警」的**唯一判定口**。

输入一帧 **BGR 原生帧**（没画过任何东西的那份 ✓），输出当前界面状态：
    combat       没有弹窗（正常战斗/挂机画面 ✓）
    lie_warn     「测谎探测器」预警弹窗（倒计时，几秒后进小游戏）
    lie_game     测谎小游戏弹窗（图形跟踪 —— M2 的活）
    lie_success  成功弹窗（按 Enter 回战斗）

口径：
  · 模板匹配（TM_CCOEFF_NORMED），模板放 `datasets/lie/templates/<状态名>.png`
    （种子模板来自用户截图 —— 裁剪过的弹窗区域 ✓）；
  · ⚠ 整帧 matchTemplate 太贵（756×590 的模板 × 1366×768 的帧 ≈ 60G 次乘加 ✗）⇒
    **统一按 `MATCH_SCALE=0.25` 缩小后匹配**（弹窗 UI 结构粗大，1/4 分辨率足够 ✓），
    单次 ~0.1s，调用方限流 1s ⇒ 开销可忽略 ✓；
  · 阈值 `MATCH_THRESHOLD=0.80`；多个模板都命中时取**分最高**的那个 ✓；
  · 模板缺失 / 帧为空 ⇒ 如实回 `combat` 并带 `why`（**绝不猜** ✗）；
  · ⚠ 种子模板是用户截图裁的弹窗区域；**第一次真实存帧回来后要用真帧重裁一遍**
    （M1 验证口径：模板与存帧必须同源 ✓）。

报警与攒帧：
  · **报警音**在 live_panel（GUI 线程 ✓，`widgets.play_sound`，音效文件可在
    「挂机保护」页配 ✓ —— 2026-09-29 从这里的 MessageBeep 迁走 ✓）；
  · `capture_frame(frame, st)`：状态切换时把**原生帧**存进 `datasets/lie/captures/`
    （攒真实样本 = 模板自愈的原料 ✓；写失败不抛 —— 收流线程不能停 ✓）。
"""

import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "datasets" / "lie" / "templates"
CAPTURE_DIR = ROOT / "datasets" / "lie" / "captures"
MATCH_SCALE = 0.25          # 匹配前整体缩放（见 docstring 的性能账 ✓）
MATCH_THRESHOLD = 0.80
TEMPLATES = ("lie_warn", "lie_game", "lie_success")

_cache = None               # {name: 缩放后的灰度模板}；None = 还没加载 ✓


def load_templates():
    """懒加载模板（**只在第一次**读盘 ✓）；一个都没有就返回空 dict（调用方如实报 combat ✓）。"""
    global _cache
    if _cache is not None:
        return _cache
    out = {}
    if TEMPLATE_DIR.is_dir():
        for name in TEMPLATES:
            p = TEMPLATE_DIR / (name + ".png")
            if not p.exists():
                continue
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, None, fx=MATCH_SCALE, fy=MATCH_SCALE,
                           interpolation=cv2.INTER_AREA)
            out[name] = g
    _cache = out
    return out


def reset_cache():
    """清模板缓存（自检换目录后用 ✓；运行期模板不变，不用 ✓）。"""
    global _cache
    _cache = None


def check_frame(frame_bgr, threshold=MATCH_THRESHOLD):
    """判一帧的界面状态 → `(state, detail)`。

    `detail` 给打点/排查用：命中带 `name` + `score` ✓；没命中带 `score` ✓；
    判不了带 `why`（没帧 / 没模板 ✓）—— **绝不猜** ✗。
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return "combat", {"why": "no frame"}
    tmpls = load_templates()
    if not tmpls:
        return "combat", {"why": "no templates"}
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=MATCH_SCALE, fy=MATCH_SCALE,
                      interpolation=cv2.INTER_AREA)
    best_name, best_score = "", -1.0
    for name, t in tmpls.items():
        if gray.shape[0] < t.shape[0] or gray.shape[1] < t.shape[1]:
            continue
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _mnl, _mxl = cv2.minMaxLoc(res)
        if mx > best_score:
            best_name, best_score = name, float(mx)
    if best_name and best_score >= threshold:
        return best_name, {"name": best_name, "score": round(best_score, 4)}
    return "combat", {"score": round(best_score, 4) if best_name else -1.0}


def capture_frame(frame_bgr, st):
    """状态切换时存一帧**原生帧**（攒真实样本 ✓）。返回路径；写失败不抛 ✓。"""
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        p = CAPTURE_DIR / ("%s_%s.png" % (st, time.strftime("%Y%m%d_%H%M%S")))
        cv2.imwrite(str(p), frame_bgr)
        return p
    except Exception:                     # noqa: BLE001 —— 收流线程不能停 ✓
        return None
