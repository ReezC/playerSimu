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
#: ⭐⭐⭐ **模板的基准分辨率**（2026-10-10 ✓ 用户报"**测谎开始后 Agent 还在操作角色按方向键**"
#: 查出来的 ✓ —— 与 271 那条"小游戏门限"是**两件事** ✗，两条都得有 ✓）。
#:
#: 这三块模板是从 **1366×768** 那一路流裁下来的 ✓，而匹配是**纯像素**的：
#: `cv2.matchTemplate` 只对**平移**宽容 ✓、对**缩放**不宽容 ✗ ⇒ 给一张别的尺寸的帧
#: （本机"本地窗口"那块是 **1920×1080** ✓）分数会**掉一大截** ⇒ 判成 `combat`
#: ⇒ `_takeover_st` 不动 ⇒ **Agent 照旧按方向键** ✓✓（用户两轮看到的就是这个 ✓）。
#:
#: **实测**（真帧 `datasets/lie/captures/lie_warn_20261010_025017.png` ✓ 见开发日志 272 ✓）：
#:     1366×768（原尺寸）→ **0.857** ✓ 判得出 `lie_warn` ✓
#:     1920×1080（放大）→ **0.659** ✗ 判成 `combat` ✓（差 0.20 分 ✓ 门限 0.80 够不着 ✓）
#:     1600×900 → 0.558 ✗ ／ 1024×576 → 0.501 ✗
#: ⇒ 匹配前**先把帧归一到这个基准尺寸** ✓（见 `_prepare_gray` ✓）；本来就是基准尺寸的帧
#:   一个像素都不动 ✓（老行为一字不变 ✓）。
#: ⚠ 将来**重裁模板**（用 `capture_frame` 存下来的真帧 ✓）时，模板出自哪个尺寸，
#:   就把这里改成那个尺寸 ✓（三块模板必须**同源同尺寸**，否则老问题会回来 ✓）。
BASE_W, BASE_H = 1366, 768
MATCH_THRESHOLD = 0.80
#: ⭐⭐⭐⭐⭐ **每块模板自己的门限**（2026-10-10 ✓ 实测定的 ✓ —— 用户原话：
#:   "**测谎的时候 Agent 还在操作移动、跳跃**"✗，第二轮报的同一件事 ✓）。
#:   **实测**（真录像 `data/recordings/lie_20261010_015650.mp4` ✓ 用 `check_frame(fr, threshold=0)`
#:   拿"最像哪块 + 多少分"✓ 见开发日志 **271** ✓）：
#:     · `lie_warn`    **0.867~0.869** ✓ 很干净（记录早就说过 ✓）
#:     · **`lie_game` 0.783~0.785** ✗ —— **低于 0.80** ⇒ **小游戏那 18 秒根本认不出来** ✗✗
#:       ⇒ 状态回落 `combat` ⇒ Agent 恢复按键 ⇒ 用户两轮看到的就是这个 ✓（录像里录满 300 秒
#:       上限 ✓ 而同期 `perf.log` 每段都是 `agent_why=-` ⇒ **Agent 全程没进接管** ✓ 对得上 ✓）
#:     · `lie_success` 0.578~0.618（那一场没走到成功弹窗 ✓ 不能据此判它坏 ✗）
#:   ⚠⚠ **为什么敢把 `lie_game` 单独放宽到 0.75**：这块模板**裁坏了**（普通战斗帧能到
#:     **0.805~0.836** ✗ 比真的还高 ✓ 2026-10-08 实测 ✓ —— 那次 4 次误判就是它 ✓）
#:     ⇒ **光降门限挡不住假阳性** ✗；**真正的判据是 `_screen_beat` 里那条 A 规则** ✓
#:     （`lie_game`/`lie_success` **必须先见过 `lie_warn`** ✓ 而 warn 那块干净 ✓ 0.867 ✓）
#:     ⇒ 假阳性照样按 `combat` 处理 ✓ ⇒ **两条合起来才是完整口径** ✓ 缺一条都不行 ✗。
#:   ⚠ 改这里要连着看 `LIE_REC_STRONG`（= **基础**门限 + 0.05 ✓ C 那条保险用 ✓ 不跟着动 ✓）。
MATCH_THRESHOLD_BY = {"lie_game": 0.75}
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


def _prepare_gray(frame_bgr):
    """整帧 → 匹配用的灰度图：**先归一到基准分辨率**，再按 `MATCH_SCALE` 缩小 ✓。

    ⚠ 归一那一步是 2026-10-10 补的（见 `BASE_W/BASE_H` 的实测说明 ✓）——
      没有它，换分辨率就认不出测谎 ⇒ Agent 在测谎里照旧按方向键 ✓✓。
    ⚠ 用 `INTER_AREA` 缩小 ✓、放大时才用 `INTER_LINEAR` ✗（`AREA` 放大会块状 ✓）。
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None
    h, w = frame_bgr.shape[:2]
    if w and abs(w / float(BASE_W) - 1.0) > 0.02:      # ±2% 以内当"就是基准尺寸"✓ 不动它 ✓
        k = BASE_W / float(w)
        _nh = max(1, int(round(h * k)))
        frame_bgr = cv2.resize(frame_bgr, (BASE_W, _nh),
                               interpolation=(cv2.INTER_AREA if k < 1.0
                                              else cv2.INTER_LINEAR))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=MATCH_SCALE, fy=MATCH_SCALE,
                      interpolation=cv2.INTER_AREA)


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
    gray = _prepare_gray(frame_bgr)          # ⭐ 先归一到基准分辨率（见 `BASE_W/BASE_H` ✓）
    if gray is None:
        return "combat", {"why": "no frame"}
    best_name, best_score = "", -1.0
    for name, t in tmpls.items():
        if gray.shape[0] < t.shape[0] or gray.shape[1] < t.shape[1]:
            continue
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _mnl, _mxl = cv2.minMaxLoc(res)
        if mx > best_score:
            best_name, best_score = name, float(mx)
    # ⭐ 门限：调用方**显式给了** `threshold` 就按它（自检/调试要能指定 ✓ 语义不变 ✓）；
    #   用默认值时，**每块模板可以有自己的门限**（`MATCH_THRESHOLD_BY` ✓ 见那段实测说明 ✓
    #   —— 这就是"小游戏认不出来 ⇒ Agent 在测谎里还在按移动/跳跃"的修法 ✓）。
    if best_name and best_score >= (threshold if threshold != MATCH_THRESHOLD
                                    else MATCH_THRESHOLD_BY.get(best_name, threshold)):
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
