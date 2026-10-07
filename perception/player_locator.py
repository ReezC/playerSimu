"""玩家定位：模板匹配（每个玩家一套 WZ 合成外观模板）。

架构定位（重要）：
    怪物走 YOLO 训练（地图数据通用），玩家**不走训练** ——
    玩家外观因人而异，训进模型就把模型绑死在某个账号上。

    正确做法是模板匹配：每个玩家导出一套外观模板，放在
        datasets/sprites/player/{角色id}/*.png
    换玩家 = 换模板目录，零训练。这满足「一套地图数据支持不同玩家」。

    模板来自 WZ 合成（用户导出），PNG 带 alpha 透明通道，匹配时 alpha 做掩码。

模板目录结构（用户约定）：
    player/{角色id}/{角色id}-{动作}-{帧号}.png
    例：player/珠缨/珠缨-stand1-0.png
"""
from pathlib import Path

import cv2
import numpy as np

from core.imgio import imread
# ⭐⭐ 「框到可见部分」——**两条内核共用那一份实现** ✓（怪物/宠物/掉落走 `tools/detect_mobs` ✓）
from perception.visible_box import VISIBLE_KEEP, VISIBLE_TOL, visible_box

DEFAULT_ROOT = Path("datasets/sprites/player")


class PlayerLocator:
    """用一套玩家外观模板，在画面里定位玩家自己。

    scale：画面里角色尺寸 / WZ 素材尺寸。随采集/推流分辨率变 ——
        1920x1080 训练帧实测 ≈ 1.25；1366x768 实时流要重标（约 0.89）。
    threshold：TM_CCORR_NORMED 匹配分阈值（实测正确命中 0.81+）。
    """

    def __init__(self, player_id, root=None, scale=1.25, threshold=0.78,
                 filter_prefix=None, frame_scales=None,
                 per_mob=0, min_distinct=0.0, max_peaks=1, downscale=1.0,
                 frames_sel=None, should_stop=None,
                 visible=False, visible_tol=None, visible_keep=None):
        """⭐ 后四个是 2026-10-05 为「④ 里玩家那一段的 5 个参数」补的（用户原话：
        "参数都放，缺功能的就补功能" ✓ 之前玩家这条链**只有阈值 / 尺度** ✓）。

        ⚠⚠ **默认值一律取"老行为"**（库里其它调用方一个字都不用改 ✓）：
          · `per_mob=0` ⇒ 模板帧**不截断**（= 现在这样全用 ✓）；
          · `max_peaks=1` ⇒ 每个模板**只取最好那一个峰**（= 现在这样 ✓）；
          · `min_distinct=0.0` ⇒ **不设**"最佳 / 次佳差距"那道闸 ✓；
          · `downscale=1.0` ⇒ 不缩（`locate()` 那个形参的老默认也是 1.0 ✓）。
        """
        self.player_id = player_id
        self.scale = float(scale)
        self.threshold = float(threshold)
        self.filter_prefix = filter_prefix   # 只加载文件名含此子串的模板（如 "stand"）
        self.frame_scales = dict(frame_scales or {})   # {player_id:帧stem -> scale}
        self.per_mob = max(0, int(per_mob or 0))       # 0 = 不截断 ✓
        #: ⭐ 用户手工挑的**模板帧**（任务 3 ✓ 用户 2026-10-05 ✓）：stem 名单 ✓
        #:   `None` / 空 = 老逻辑（全用 ✓）—— 与掉落物 / 怪物那条**同一份口径** ✓
        #:   （`tools/detect_mobs.load_templates` 的 `frames_sel` ✓）。
        self.frames_sel = ({str(s) for s in frames_sel} if frames_sel else None)
        self.min_distinct = max(0.0, float(min_distinct or 0.0))
        # ⭐ 用户 2026-10-05 ✓：`-1` = 无限 ✓ `0` = 不标注（一个都不取 ✓）。
        #   ⚠ 原来 `max(1, int(max_peaks or 1))` 会把 0 变成 1 ✗（`0 or 1` ✓）⇒ 改掉。
        self.max_peaks = int(max_peaks if max_peaks is not None else 1)
        self.downscale = min(1.0, max(0.25, float(downscale or 1.0)))
        #: ⭐⭐ **"该停了吗"**（用户 2026-10-05 ✓ 原话："能不等他返回吗？"）：加载要逐张
        #:   `imread` ✓，原来是个整块动作 ✗ ⇒ 点了取消只能等它读完 ✓（界面上那句
        #:   "已请求取消：正在收尾…"✓）⇒ 传 `ctx.canceled` 进来就能**边读边收工** ✓
        #:   （与 `tools/detect_mobs.load_templates` 的 `should_stop` **同一份口径** ✓）。
        #: ⚠ 被打断时 `self.stopped = True`（见 `_load` ✓）—— 调用方**必须**看它或
        #:   `ctx.canceled()` 并自己收工 ✗，别把半份模板当完整的用 ✓。
        self._should_stop = should_stop
        self.stopped = False       # 加载过程里被叫停过 ✓
        self._templates = []       # [(name, bgr, mask), ...]
        #: ⭐⭐ **框到可见部分**（用户 2026-10-06 ✓ 原话："**所有的匹配都需要「框到可见部分」
        #:   参数**"）：开了之后 `locate()` 交出去的框会按 `perception/visible_box.py` 收到
        #:   "目标**真的露出来的那一块**"（被挡住的部分不进框 ✓）。
        #:   ⚠⚠ **默认关** ✓（这是改**标注语义**、不是修 bug ✗ —— 老调用方一字不变 ✓；
        #:     实时定位那条路（`gui/live_thread`）**永远不传它** ⇒ 行为与以前完全一致 ✓）。
        #:   ⚠ 两个阈值**从实现那边拿默认**（一处口径 ✓ 别在这儿再写 26.0/0.2 ✗）。
        self.visible = bool(visible)
        self.visible_tol = float(VISIBLE_TOL if visible_tol is None else visible_tol)
        self.visible_keep = float(VISIBLE_KEEP if visible_keep is None else visible_keep)
        #: 名字 → (bgr, mask) 的索引（`_load` 里灌 ✓）：`locate` 拿到胜出的**那一个**名字后
        #:   要拿回它的模板才能算"可见部分" ✓（`cand` 里不塞数组：那些数组在 `visible` 关着
        #:   时**一个都不该建** ✗ —— 实时那条路每帧都要跑，白建就是白花时间 ✓）。
        self._tmap = {}
        self._load(root or DEFAULT_ROOT)

    # ---------------- 模板加载 ----------------

    def _load(self, root):
        d = Path(root) / self.player_id
        if not d.is_dir():
            raise FileNotFoundError("找不到角色模板目录: %s\n"
                                    "请把 WZ 合成的角色动作帧放到这里，"
                                    "结构为 {角色id}/{角色id}-{动作}-{帧号}.png" % d)
        files = []
        for f in sorted(d.glob("*.png")):
            if self.filter_prefix:
                # 支持逗号分隔的多个动作前缀，如 "stand,walk"（匹配含 stand 或 walk 的）
                prefixes = [p.strip() for p in self.filter_prefix.split(",") if p.strip()]
                if prefixes and not any(p in f.stem for p in prefixes):
                    continue
            files.append(f)
        # ⭐ 「最大模板帧」（用户 2026-10-05 ✓）：模板帧**上限** —— 与怪物那条同口径
        #   （`tools/detect_mobs._pick_frames` ✓：不超上限就全用 ✓；超了才截 ✓）。
        #   ⚠ 玩家这边的文件名是 `{id}-{动作}-{帧号}`（不是 `动作_帧号` ✓）⇒ 按"动作"分组
        #   得猜分隔符 ✗ ⇒ 这里改成**在排序好的名单上等距抽** ✓：每个动作都能留到代表帧 ✓
        #   （动作在名单里是连着的 ✓ 等距抽不会整段丢掉某个动作 ✓）。
        #   ⚠ `per_mob <= 0`（库里默认 ✓）⇒ 一个都不截 ✓ = 老行为一字不变 ✓。
        # ⭐⭐ 用户手工挑的**模板帧**（任务 3 ✓ 用户 2026-10-05 ✓）：有名单就**先过滤** ✓
        #   —— 名单是候选集 ✓ 下面那个"上限"照旧在它之上跑 ✓（两根轴正交 ✓
        #   与 `tools/detect_mobs.load_templates` 里那段**同一份口径** ✓）。
        #   ⚠ 一个都没命中 ⇒ 这一条**没有模板** ✓ ⇒ 下面那句 `raise` 会如实报错 ✓
        #     （不静默回落到"全用" ✗ —— 用户全取消勾 = "这条别用" ✓）。
        if self.frames_sel:
            files = [f for f in files if f.stem in self.frames_sel]
        if self.per_mob and len(files) > self.per_mob:
            step = len(files) / float(self.per_mob)
            files = [files[int(i * step)] for i in range(self.per_mob)]
        for f in files:
            # ⭐ 每读一张看一眼"该停了吗"（见 `should_stop` 的说明 ✓）：被叫停 ⇒ 立刻收手 ✓
            #   （**不抛异常** ✗ —— 用户点的是"取消"，不是"这里坏了" ✓）
            if self._should_stop is not None and self._should_stop():
                self.stopped = True
                return
            t = self._load_tpl(f)
            if t:
                b, m = t
                self._templates.append((f.stem, b, m))
                # 镜像出朝左的模板，运行时左右都能匹配
                self._templates.append((f.stem + "@L", cv2.flip(b, 1), cv2.flip(m, 1)))
        # 名字 → 模板的索引（给"框到可见部分"回头看胜出那张用的 ✓ 只存引用 ✓ 不复制像素 ✓）
        self._tmap = {nm: (b, m) for nm, b, m in self._templates}

        # ⚠ 被叫停时**不报错**（那时候空/半份是正常的 ✓ 调用方自己看 `ctx.canceled()` ✓）
        if not self._templates and not self.stopped:
            raise FileNotFoundError("角色模板目录里没有可用 PNG: %s" % d)

    @staticmethod
    def _load_tpl(path):
        """载入单张模板：裁掉透明边框，返回 (bgr, alpha掩码)。"""
        im = imread(path, cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            return None
        b, al = im[:, :, :3].copy(), im[:, :, 3]
        ys, xs = np.nonzero(al > 128)
        if len(xs) < 100:
            return None
        b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return b, ((al > 128) * 255).astype(np.uint8)

    # ---------------- 定位 ----------------

    def locate(self, bgr, search_rect=None, downscale=None):
        """在一帧 BGR 图里定位玩家。

        search_rect: (x, y, w, h) 限定搜索区域（画面坐标），None = 全图。
            横版 ARPG 玩家在画面中央附近，限定区域能把耗时降一个量级。
        downscale: 匹配前把搜索图和模板同比例缩小（如 0.5）。模板匹配耗时
            和面积正相关，0.5 能再降约 4 倍；坐标/尺寸会乘回原尺度。
            玩家模板通常几十个（多动作 × 镜像），这是实时帧率的主要瓶颈。
            ⚠ 不传（None）= 用构造时那个 `downscale` ✓（2026-10-05 加的 ✓ 默认 1.0 ✓
              ⇒ 老调用方不传就是老行为 ✓）。
        返回 (cx, cy, w, h, score, name) —— cx/cy 是框中心（**画面坐标**，
        即使限定区域也会加回偏移），w/h 是框尺寸；找不到返回 None。
        ⭐ `visible=True`（构造时给 ✓ 默认关 ✓）时 w/h 与 cx/cy 是**可见部分**那一块
          （被挡住的部分不进框 ✓ 见 `perception/visible_box.py`）—— 标注那条链要它 ✓，
          实时定位那条链**别开**（看的是"人在哪"、不是"露出来多少" ✓）。
        ⚠ 两种"返回 None"要分清：**真的没找到**（分数不过阈值 ✓）／**判为存疑**
          （最佳与次佳的差距 < `min_distinct` ✓ 见下面那段说明 ✓）—— 后者是本轮
          为用户那个「区分度」参数补的，默认 0 = 不看这项 ✓ 老行为不变 ✓。
        """
        h, w = bgr.shape[:2]
        if search_rect is not None:
            sx, sy, sw, sh = search_rect
            sx = max(0, min(w - 1, int(sx)))
            sy = max(0, min(h - 1, int(sy)))
            sw = max(1, min(w - sx, int(sw)))
            sh = max(1, min(h - sy, int(sh)))
            canvas = bgr[sy:sy + sh, sx:sx + sw]
        else:
            sx = sy = 0
            sw, sh = w, h
            canvas = bgr

        ds = max(0.25, min(1.0, float(self.downscale if downscale is None else downscale)))
        if ds < 1.0:
            canvas = cv2.resize(
                canvas, (max(1, int(sw * ds)), max(1, int(sh * ds))),
                interpolation=cv2.INTER_AREA)

        # ⭐ 候选峰（用户 2026-10-05 ✓ 为了接「每模板峰数」/「区分度」两个参数 ✓）：
        #   老代码每个模板只 `minMaxLoc` 一次、全局留最好那个 ✓ ⇒ 那一份口径
        #   `max_peaks=1` 时**一字不差**地保留下来了 ✓（下面那个 for 只跑一轮 ✓）。
        cand = []
        for name, tb, tm in self._templates:
            # 帧级尺度：去掉 @L 镜像后缀，按 {player_id:帧stem} 查，否则用全局 scale
            stem = name.split("@")[0]
            sc = self.frame_scales.get("%s:%s" % (self.player_id, stem), self.scale)
            tw = max(2, int(round(tb.shape[1] * sc * ds)))
            th = max(2, int(round(tb.shape[0] * sc * ds)))
            if tw >= canvas.shape[1] or th >= canvas.shape[0]:
                continue
            tbb = cv2.resize(tb, (tw, th), interpolation=cv2.INTER_AREA)
            tmm = cv2.resize(tm, (tw, th), interpolation=cv2.INTER_NEAREST)
            try:
                r = cv2.matchTemplate(canvas, tbb, cv2.TM_CCORR_NORMED, mask=tmm)
            except Exception:
                continue
            # 同 tools/detect_mobs：TM_CCORR_NORMED 的分母（窗口能量）趋 0 时，
            # OpenCV 给的是 FLT_MAX 或 NaN —— np.nan_to_num 只把 NaN 变 0，
            # FLT_MAX 会留下，于是黑边/纯色面板上能拿到"满分定位"，
            # minMaxLoc 也会挑中它。归一化相关按定义不超过 1，超了就是退化解。
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            r[r > 1.0] = 0.0
            # ⭐ `-1` = 无限（给安全上限 ✓）；`0` = 不标注 ⇒ 一轮都不跑 ✓
            _rounds = (self.max_peaks if self.max_peaks > 0
                       else (0 if self.max_peaks == 0 else 64))
            for _ in range(_rounds):               # 默认 1 ⇒ 跑一轮 = 老行为 ✓
                _, mx, _, ml = cv2.minMaxLoc(r)
                if mx <= 0.0:
                    break                          # 没峰了（或被上一步抹平）✓
                cand.append((float(mx), ml[0], ml[1], tw, th, name))
                # 抹掉这一峰周围 ⇒ 同一个位置不会被反复取 ✓（同 `detect_mobs` 的手法 ✓）
                x0, y0 = int(ml[0]), int(ml[1])
                r[max(0, y0 - th // 2):y0 + th // 2 + 1,
                  max(0, x0 - tw // 2):x0 + tw // 2 + 1] = 0.0

        if not cand:
            return None
        cand.sort(key=lambda c: -c[0])
        best = cand[0]
        if best[0] < self.threshold:
            return None
        # ⭐ **「区分度」= 最佳与次佳的差距下限**（用户 2026-10-05 ✓ `0` = 不看这项 ✓）。
        #   ⚠ 对玩家这条链，它**不是**"去重闸"而是**可信度闸**：玩家是唯一目标 ✓，而模板是
        #   一堆动作帧（彼此极像 ✓）⇒ "次佳"常常就是同一角色的**另一帧** ✓ ⇒ 分差太小
        #   = 这次定位说不清是哪个位置 / 哪一帧 ⇒ 判**存疑**（返回 None ✓）比硬给一个更诚实 ✓。
        #   ⚠ 正因如此，③ 里**玩家那一段的默认值是 0**（不看 ✓）；调大 = 更严 ✓。
        if self.min_distinct > 0.0 and len(cand) > 1:
            if (best[0] - cand[1][0]) < self.min_distinct:
                return None
        score, x, y, tw, th, name = best
        # ⭐⭐ **框到可见部分**（用户 2026-10-06 ✓ "**所有的匹配都需要「框到可见部分」参数**"
        #   —— 与怪物/宠物/掉落走的是**同一份实现**：`perception/visible_box.py` ✓）。
        #   ⚠ `visible=False`（默认）⇒ 这一段**一点都不跑** ✓：连灰度图都不建 ✓
        #     （实时定位那条路每帧都要 `locate` ⇒ 白建就是白花钱 ✗）。
        #   ⚠ 算出来的框是**canvas 尺度**的（匹配若在缩略图上做 ✓）⇒ 最后还是统一乘回 ✓
        #     （与下面那句老换算**同一个公式** ✓ 只把 `x,y,tw,th` 换成收过的值 ✓）。
        bx, by, bw, bh = x, y, tw, th
        if self.visible:
            hit = self._tmap.get(name)
            if hit is not None:
                tb, tm = hit
                stem = name.split("@")[0]
                sc = self.frame_scales.get("%s:%s" % (self.player_id, stem), self.scale)
                # ⚠ 必须**用同一个尺寸**重算（`matchTemplate` 时就是这么缩的 ✓）——
                #   尺寸对不上就说明上面那条 `continue` 把这张跳过了 ⇒ 别硬算 ✓
                if (max(2, int(round(tb.shape[1] * sc * ds))) == tw
                        and max(2, int(round(tb.shape[0] * sc * ds))) == th):
                    tbb = cv2.resize(tb, (tw, th), interpolation=cv2.INTER_AREA)
                    tmm = cv2.resize(tm, (tw, th), interpolation=cv2.INTER_NEAREST)
                    imgf = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    tg = cv2.cvtColor(tbb, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    tgx = cv2.Sobel(tg, cv2.CV_32F, 1, 0, ksize=3)
                    tgy = cv2.Sobel(tg, cv2.CV_32F, 0, 1, ksize=3)
                    bx, by, bw, bh = visible_box(
                        imgf, tg, tgx, tgy, tmm > 0, x, y,
                        self.visible_tol, self.visible_keep)
        # 坐标与尺寸乘回原尺度（匹配是在 downscale 后的图上做的）✓
        return (bx / ds + sx + bw / ds / 2.0, by / ds + sy + bh / ds / 2.0,
                bw / ds, bh / ds, score, name)

    @property
    def template_count(self):
        return len(self._templates)
