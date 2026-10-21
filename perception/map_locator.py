"""玩家定位：把「单帧认出的黄点」串成稳定轨迹，并换算成世界坐标（2026-10-18 切出）。

分工（一句话）：`perception/map_dot.py` 回答"**这一帧**玩家那个点在面板哪儿"；
本模块回答"**跨帧**他一直在哪、现在到底站在哪块平台上"。

- `PlayerDotTracker`：多候选挑点、跳变丢弃重捕、连续两拍才确认、漏检期外推/保持；
- `PlayerLocator`：整条定位链（认点 → 换算世界坐标 → 结果可信度 → 各种兜底）。

⚠ 依赖方向**单向**：本模块 import `map_dot` / `minimap`，那两个**不** import 本模块。
"""

import time

import numpy as np

from core import mapdata, zones
from perception import map_calib
from perception import map_const
from perception.map_dot import (DOT_CONFIRM_FRAMES, DOT_EXP_HISTORY,
                                DOT_EXP_MIN_SAMPLES, DOT_GHOST_DECAY,
                                DOT_GHOST_MAX_SHIFT, DOT_HOLD_MS,
                                DOT_JUMP_GIVEUP_N, DOT_JUMP_WINDOW_S,
                                DOT_MAX_JUMP_PX, DOT_ROI_PAD,
                                DOT_SMOOTH_ALPHA, PLAYER_FB_EMA_ALPHA,
                                _local_result_trustworthy, _shift_calib,
                                dot_feet, dot_side_max, find_player_dot,
                                roi_around, track_params, tracker_kwargs)
from perception.minimap import CropViewTracker, try_autofit

#: 上次"认不出黄点"时落盘的时刻（节流用 ✓ `_dump_fail_panel` 内部管 ✓）
_LAST_FAIL_DUMP = 0.0

#: ⭐⭐⭐ 「局部小地图跟出来的 `view`」要**比标定那份的地形重合度好多少**才采信
#:   （2026-10-19 现场 ✓ 见 `update` 里那段"地形掩码复核"）。
#:   · 0.06 = 4~6 个百分点的 IoU：小到不会因为噪点误杀**真跟住**的那一拍 ✓，
#:     大到能把"平顶峰里随手挑一个"的假跟住拦住 ✓（现场假跟住 0.095 vs 标定 0.265 ✓）；
#:   · ⚠ 别调成 0：那等于"差不多就信" ⇒ 又回到这次的事故形状 ✗。
CROP_MASK_MARGIN = 0.06


def crop_view_trusted(panel, canvas, calib, view, strict=False):
    """「局部小地图」跟出来的 `view` **值不值得采信** → `(ok, iou_new, iou_ref)`。

    判据只有一条：**地形掩码重合度**（`map_calib.calib_overlap` ✓ 走 `terrain_mask`，
    对"底图是重绘的、面板是客户端渲染 + 图标叠加"这类差异**不敏感** ✓）。

    · `ok=True`  ⇒ 跟到的这份**不比标定那份差**（允许差 `CROP_MASK_MARGIN` 以内）⇒ 用 ✓；
    · `ok=False` ⇒ 明显更差 ⇒ **别用**（退回标定里那份 = "没跟住"那条老路 ✓ 界面会提示 ✓）；
    · 判不了（没画面 / 没几何 / 算不出掩码）⇒ **`ok=True` 不拦** ✓（别把没证据的事判死 ✗，
      与 `panel_mismatch` 同一条纪律 ✓）。

    ⭐⭐⭐ `strict=True`（**没过 `TRUST_SCORE` 那档的结果用**，即放宽档 / 帧间位移）：
      要求掩码**明显更好**（`a >= b + CROP_MASK_MARGIN`）才采信 ✓，而不是"差不多就信" ✗。
      为什么（用户 2026-10-20 ✓ 原话："现在会**乱飞乱抖**，匹配的完全不对"✗）：
        · 这张图的逐像素相关是**平的**（峰间距 12~13 px、高差 0.005）⇒ 放宽档锁回来的位置
          可能**刚好差一个峰**（实测差 40 个底图像素 ≈ 600 世界像素 ✗）；
        · 而"允许差 0.06"的宽判据挡不住它：错峰那份的掩码往往就在 0.06 以内 ⇒ 照样放行
          ⇒ 现场表现就是**每隔一阵跳一下**（`pos.y` 在 810 / 1428 / 1811 之间弹 ✓）；
        · 改成"必须明显更好"之后：真位置（0.376 vs 标定 0.251 = **+0.125**）照样过 ✓，
          错峰那份（0.157 = **−0.09**）当场被拒 ✓。
    """
    if panel is None or canvas is None or not calib or view is None:
        return True, 0.0, 0.0
    try:
        new_cal = dict(calib, view=[float(view[0]), float(view[1])])
    except (TypeError, ValueError, IndexError):
        return True, 0.0, 0.0
    ov_new = map_calib.calib_overlap(panel, canvas, new_cal, search=0)
    ov_ref = map_calib.calib_overlap(panel, canvas, calib, search=0)
    if not ov_new or not ov_ref:
        return True, 0.0, 0.0
    a = float(ov_new.get("iou") or 0.0)
    b = float(ov_ref.get("iou") or 0.0)
    if strict:
        return (a >= b + CROP_MASK_MARGIN), a, b
    return (a + CROP_MASK_MARGIN >= b), a, b


def _dump_fail_panel(panel, r, map_id=None, src_name=None, every=5.0):
    """**认不出黄点时，把程序自己看到的那块面板原样落盘** → `datasets/map/_mmap_fail.png`。

    为什么要有（2026-10-18 现场 ✓）：用户报「黄点永远认不出来」，而我用**同一个框**
    （`[5,71,109,114]`）连抓 20 帧，`find_player_dot` **20/20 都认得出** ⇒ 说明程序拿到的那块
    画面和我抓到的**不是同一块**（来源/坐标/缩放其中之一不同 ✗）。猜不出来，只能看实物 ✓。

    · 节流 `every` 秒一张（固定文件名，后写盖前一张 ✓）⇒ 不会写爆磁盘 ✓；
    · 旁边那份 `.txt` 记下面板尺寸与各层的 diag ✓（拿图 + 数字一起看才判得动 ✓）；
    · **全程 try/except**：这是诊断旁路，绝不能因为它把定位弄挂 ✗。
    """
    global _LAST_FAIL_DUMP
    now = time.monotonic()
    if now - _LAST_FAIL_DUMP < float(every):
        return
    _LAST_FAIL_DUMP = now
    try:
        import cv2
        from pathlib import Path
        # ⚠ **文件名要带地图/来源**（2026-10-18 现场 ✓）：两个工作台同时开着时都写同一个
        #   `_mmap_fail.png` ⇒ **后写的盖前一张** ⇒ 我差点把"天空（收流、画面全黑）"那张
        #   当成"森林迷宫认不出黄点"的证据 ✗（shape 129×71 全零 ⇒ 明显是收流断了的黑帧 ✓）
        mid = str(map_id or "?")
        src = str(src_name or "?")
        tag = "%s_%s" % (mid, src)
        d = Path(__file__).resolve().parents[1] / "datasets" / "map"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / ("_mmap_fail_%s.png" % tag)), panel)
        (d / ("_mmap_fail_%s.txt" % tag)).write_text(
            "shape=%r  all_black=%r\nmask_px=%s dense_px=%s candidates=%s layer=%s roi=%r\nwhy=%s\n"
            % (getattr(panel, "shape", None), _panel_all_black(panel),
               r.get("mask_px"), r.get("dense_px"),
               r.get("candidates"), r.get("layer"), r.get("roi"),
               str(r.get("reason") or "")[:600]),
            encoding="utf-8")
    except Exception:                       # noqa: BLE001 —— 诊断旁路，永不影响定位 ✓
        pass


def _panel_all_black(panel, thr=8):
    """这块面板是不是**整幅全黑**（抓屏断了 / 收流断了 ⇒ 不该说"认不出黄点" ✗）。

    2026-10-18 现场 ✓：失败时落盘的那张 **129×71、灰度均值 0、std 0** ⇒ 那是黑帧，
    不是"小地图上没有黄点" —— 两种病要分开说，否则永远查错方向 ✗。
    """
    try:
        import numpy as _np
        return bool(_np.asarray(panel).size) and int(_np.asarray(panel).max()) <= int(thr)
    except Exception:                       # noqa: BLE001
        return False


class PlayerDotTracker:
    """跨帧跟踪玩家标记：多候选挑最近的、跳变丢掉重捕、连续两拍才算确认。

    **为什么非要跨帧**：单帧里"亮黄的方块"不止玩家一个（NPC 点、特效、地图纹理），
    而玩家点是**连续移动**的：用上一拍位置当下先验，一跳 40px 以上的直接判噪声。
    三条做法都照 `perception/track/tracker.py` 那套（怪物/角色的防抖）：

      ① **有上一帧位置就只在它周围搜一小块**（`DOT_ROI_PAD`）—— 又快又稳：
         收流那条来源的面板 753×612，全图掩码+连通域每拍几毫秒，缩到 37×37 基本
         免费；底图上的杂色也根本进不来；
      ② **漏检就沿用上一帧位置**（`DOT_HOLD_MS` 内），口径同那边的"幽灵框"——
         不然读数一秒里闪好几次"认不出"，寻路也白丢一小段惯性；
      ③ 幽灵期间按**衰减的速度**往前推一点（总位移有上限），同 `_drift`：既不能
         冻在原地（人在跑，位置越差越远），也不能一直推（会飘到地图另一头）。
    """
    def __init__(self, hold_ms=None, roi_pad=None, max_jump=None,
                 ghost_max_shift=None):
        self.x = None
        self.y = None
        self.hits = 0
        self.frames = 0
        # 这四个是**界面参数**（config/live.yaml，见 track_params）：
        #   沿用窗口 / 搜索半径 / 跳变上限 / 外推上限 —— 都能在「路线识别」里调。
        self.hold_ms = float(DOT_HOLD_MS if hold_ms is None else hold_ms)
        self.roi_pad = int(DOT_ROI_PAD if roi_pad is None else roi_pad)
        self.max_jump = float(DOT_MAX_JUMP_PX if max_jump is None else max_jump)
        self.ghost_max_shift = float(DOT_GHOST_MAX_SHIFT if ghost_max_shift is None
                                     else ghost_max_shift)
        self.missed = 0            # 连续漏检拍数（口径同 MobTracker 的 missed）
        self.last_roi = None       # 上一次实际搜的区域（诊断/自检用）
        self._vx = self._vy = 0.0  # 速度（面板像素/拍，平滑过）
        self._shifted = 0.0        # 幽灵外推累计走了多远
        self._last_seen = 0.0
        self._last = None          # 上一次成功的完整结论（补位时原样复用）
        #: ⭐⭐ **自标定的"期望边长"尺子**（2026-10-11 ✓ 见 `DOT_PLAYER_SIDE_REL` 那张量测表 ✓）：
        #:   窗口里放"**最近锁定成功**那些点的短边"，取**中位数**当尺子 ✓。
        #:   为什么必须自标定：那条"短边 = 面板宽 × 3.2%"**跨来源不成立** ✗（1164 宽那块真点
        #:   只有 **2.15%** ⇒ 真点被判成"偏小" ⇒ like 掉到 **0.25** ⇒ 跟杂块同档 ⇒ 谁近谁赢 ✗
        #:   —— 用户报的"经常认错"就是它 ✓）。
        #:   ⚠ 样本不够（`DOT_EXP_MIN_SAMPLES` ✓）⇒ 退回比例兜底 ✓（不猜 ✓ 老行为 ✓）；
        #:   ⚠ 用**中位数**（抗单帧怪块 ✓）；⚠ 硬尺寸闸已经把"几像素的东西"挡在候选外 ✓
        #:     ⇒ 尺子学不到那种尺寸 ✓（不会被锁错的那个带跑偏 ✓）。
        self._exp_hist = []
        self._exp_side = None

    def force_reacquire(self, why=""):
        """⭐ **丢掉"锁定的那个点"，下一拍做全画面重捕**（用户 2026-10-06 ✓ 原话："**那就是位置
        状态把黄点认错了，刚才人在底层最右边角落卡死**"✓）。

        现场证据（`behavior.log` ✓）：`mmap_dot` 一直在 `basemap` ↔ `roi` 之间翻 ✓ ⇒ **颜色那一层
        从来没赢过** ✗ ⇒ 一路靠"底图相减"兜 ⇒ 底图杂色多时**会锁一个不动的假点** ✓ ⇒ 位置状态
        给出"**新鲜但是错**"的坐标 ✓ ⇒ Agent 以为"还差 657px"照直往右顶墙，而人其实**已经卡在
        底层最右边的角落** ✓ ⇒ 卡死 ✓。

        ⚠ **该不该重捕，不在这儿判** ✗：跟踪器**看不见按键** ✓ —— "人真站着不动"（站桩 / 巡逻 ✓）
          和"读数卡住"在它眼里一模一样 ✗ ⇒ 判据只能由**知道按键的那一侧**（agent ✓）给。
        ⚠ 重捕是**无害**的 ✓：真是被墙挡住 ⇒ 重捕之后读数照样不动 ⇒ "走不动"那条判据照旧如实报 ✓
          （所以可以放心地把它当成一道自愈手段 ✓）。
        """
        self.x = self.y = None
        self.hits = 0
        self.missed = 0
        self._vx = self._vy = 0.0
        self._last = None
        self.last_roi = None
        self.reacquires = int(getattr(self, "reacquires", 0)) + 1
        self.reacquire_why = str(why or "")

    def update(self, panel, calib=None, terrain=None, roi=None, now=None):
        now = time.monotonic() if now is None else now
        self.frames += 1
        r = self._search(panel, calib, terrain, roi)

        if not r["ok"]:
            # 漏检：防抖窗口内**沿用上一帧位置**（同 MobTracker 的幽灵框做法）。
            # `hold_ms <= 0` = 关掉这条（口径同项目里其它"0 = 关"的参数）。
            # ⚠ 不判这个的话，"窗口设 0"在同一拍里仍然算没过期 —— Windows 的
            # `time.monotonic()` 粒度约 15.6 ms，两次调用会返回同一个值。
            if (self.hold_ms > 0 and self._last is not None
                    and (now - self._last_seen) * 1000.0 <= self.hold_ms):
                self.missed += 1
                hx, hy = self._drift()
                self.x, self.y = hx, hy
                return dict(self._last, x=hx, y=hy, held=True,
                            missed=self.missed,
                            confirmed=self.hits >= DOT_CONFIRM_FRAMES,
                            reason="上一帧位置（漏检 %d 拍）" % self.missed)
            self.hits = 0
            self.missed = 0
            self._last = None
            return dict(r, hits=0, confirmed=False, held=False, missed=0)

        if self.x is not None:
            d = ((r["x"] - self.x) ** 2 + (r["y"] - self.y) ** 2) ** 0.5
            # ⭐⭐⭐ **真传送 vs 噪声**（用户 2026-10-17 ✓「根本没有防抖…超远 2 个地方跳变认黄点」✗
            #   ＋ 老用例「跳变丢弃后要能重捕，人真传过去就永远追不上了」✓ 两条都要满足 ✓）：
            #   · **同一个远处候选连着两拍** ⇒ 那是**真传送** ✓ ⇒ 接受（否则人传过去永远追不上 ✗）；
            #   · **两个远点交替**（A→B→A→B）⇒ `_jump_rej` 每拍都被"回到原点"那拍清掉 ✓
            #     ⇒ 永远凑不成"连着两拍" ✓ ⇒ **一直拒绝** ✓（这才是用户要的防抖 ✓）。
            #   ⚠ 这就是"不许清锁位"之外的另一半 ✓：只留位置还不够，还得让真传送有出路 ✓。
            _same_jump = False
            if self.max_jump > 0 and d > self.max_jump:
                _jtol = max(6.0, float(self.max_jump) * 0.5)
                _rp = self.__dict__.get("_jump_rej")
                if (_rp is not None
                        and abs(r["x"] - _rp[0]) <= _jtol
                        and abs(r["y"] - _rp[1]) <= _jtol):
                    _same_jump = True
                    self._jump_n = 0
                    self._jump_rej = None
                else:
                    self._jump_rej = (r["x"], r["y"])
            if self.max_jump > 0 and d > self.max_jump and not _same_jump:
                # ⭐⭐ **先数一数"连着被丢了几次"**（用户 2026-10-06 ✓ 见 `DOT_JUMP_GIVEUP_N`
                #   上面那段现场 ✓）：**跳过阈值这么多次 ⇒ 承认旧锁已失效** ✓，不再丢，
                #   **往下走**（= 直接接受这个新候选 ✓，等价于"承认这是传送"✓）。
                #   ⚠ 仍然**照旧丢掉旧位置再报认不出**（老行为 ✓ 零回归 ✓）—— 只是**不再无限重复** ✗。
                # ⚠⚠ **计数必须"按时间窗累计"，不能受"接受"影响** ✗（第二版我用"接受就减一"✗
                #   ⇒ 现场立刻打脸 ✓：`mmap_pick … 第 **1** 次` 一直重复 ✗ —— 因为"丢掉"之后
                #   下一拍是**整幅搜 → 接受**（接受的**还是同一个假点** ⇒ `pos` 才一直不变 ✓）
                #   ⇒ 丢/接受**交替** ⇒ 减一把它永远压回 0 ✓）。语义改成：
                #   **同一个短时间窗里"被丢"了几次** ✓ —— 窗内不衰减 ✓，超过窗口就重新起算 ✓。
                _now_j = time.monotonic()
                _last_j = float(self.__dict__.get("_jump_last") or 0.0)
                if _now_j - _last_j > DOT_JUMP_WINDOW_S:
                    self._jump_n = 1
                else:
                    self._jump_n = int(self.__dict__.get("_jump_n", 0)) + 1
                self._jump_last = _now_j
                if self._jump_n < DOT_JUMP_GIVEUP_N:
                    # ⛔⛔ **不许把锁清掉** ✗（2026-10-17 ✓ 用户原话：
                    #   「**根本就没有防抖（路线识别页签→跳变上限）**，会再超远的 2 个地方跳变
                    #   认黄点**」✗）。病根就是这个 `self.x = self.y = None`：
                    #   清掉之后下一拍 `self.x is None` ⇒ **跳变判据整段不执行** ✗
                    #   （它挂在 `if self.x is not None:` 里面 ✓）⇒ 下一拍"全画面搜"直接
                    #   **无条件接受**远处那个候选 ✓ ⇒ 再下一拍又跳回来 ✓ ⇒
                    #   在两个远点之间**来回认**（画面上就是"跳变上限根本没生效"✓
                    #   —— 其实它每拍都在丢，只是丢完立刻又无条件接了一个 ✓）。
                    #   ⇒ 现在：**位置留着** ✓（`_search` 仍以它为 `near` 附近搜 ✓），
                    #     只把"这一拍没有可信读数"如实报出去 ✓；连续 `DOT_JUMP_GIVEUP_N`
                    #     次仍不过来 ⇒ 下面那条 `force_reacquire`（整幅重捕 ✓ 传送照旧认 ✓）。
                    self.hits = 0
                    self.missed = 0
                    self._last = None
                    return dict(r, ok=False, hits=0, confirmed=False, held=False,
                                missed=0, reason=(
                                    "候选位置跳变 %.0f px（阈值 %.0f）—— 当噪声丢弃，"
                                    "**锁仍留在原位**、下一拍继续在附近找（第 %d 次）"
                                    % (d, self.max_jump, self._jump_n)))
                # ⭐⭐⭐ 到这儿 = 连着 `DOT_JUMP_GIVEUP_N` 次都"跳变" ⇒ **旧锁是假的** ✓
                #   ⇒ **丢掉它、要求下一拍"全画面重捕"** ✓（`force_reacquire` ✓）——
                #   ⚠⚠ **不是"接受眼前这个候选"** ✗（我第一版就是那么写的 ✗，现场立刻证明
                #     不够 ✓）：因为跟踪器只在**锁定点周围一小块**里搜（`_search` 的
                #     `near` ✓ 半径 `roi_pad` ✓）⇒ **真点在框外，压根进不了候选** ✗✗
                #     ⇒ 眼前这些候选（`955/910/824…`）**全是笼子里的东西** ✗，接受哪个都错 ✓。
                #   只有"**下一拍整幅搜**"才可能看到真点 ✓（现场真点在面板 `(1135,478)` ✓，
                #   而锁定点在 `(796,245)` ✓ ⇒ 隔着一个 ROI 半径 ✗）。
                self._jump_n = 0
                self.force_reacquire("连续 %d 次跳变 ⇒ 承认旧锁已失效，要求全画面重捕"
                                     % DOT_JUMP_GIVEUP_N)
                return dict(r, ok=False, hits=0, confirmed=False, held=False, missed=0,
                            reason=("连续 %d 次「候选位置跳变」⇒ 承认旧锁已失效 ⇒ "
                                    "**丢掉锁定、下一拍全画面重捕** ✓" % DOT_JUMP_GIVEUP_N))

        # ⚠ 接受了一个"**贴着锁**"的候选 ⇒ 把"上一个被拒的远点"忘掉 ✓
        #   （否则 A→B→A→B 交替时，第二次 B 会被当成"连着两拍"⇒ 误判成真传送 ✗）
        self._jump_rej = None
        # 速度：够算"漏检期间大概往哪边走"就行，平滑一下免得被单拍抖动带偏
        if self.x is not None:
            self._vx = 0.5 * self._vx + 0.5 * (r["x"] - self.x)
            self._vy = 0.5 * self._vy + 0.5 * (r["y"] - self.y)
        else:
            self._vx = self._vy = 0.0
        # ⭐⭐ **报出去的位置做一阶低通**（2026-10-10 ✓ 用户："根本没做任何平滑"✗）：
        #   ⚠ 只抹抖动、不引入明显滞后 ⇒ α = `DOT_SMOOTH_ALPHA`（0.6 ✓）；
        #   ⚠⚠ **只在"接着上一拍"时滤** ✗：① 刚重捕 / 上一拍没能看见（间隔久了）⇒
        #     **直接到位** ✓（那时平滑会把"我现在其实在这"拖住 ✓）；② 位移本身就很大
        #     （真在跑动 / 传送那一下）⇒ 不滤 ✓（滤了读数就钝 ⇒ 寻路跟着钝 ✓）。
        _rx, _ry = float(r["x"]), float(r["y"])
        _a = float(DOT_SMOOTH_ALPHA)
        if self.x is not None and 0.0 < _a < 1.0 and (now - float(self._last_seen or 0.0)) <= 0.5:
            _d = ((_rx - self.x) ** 2 + (_ry - self.y) ** 2) ** 0.5
            if _d <= max(6.0, 2.0 * dot_side_max(panel.shape[1])):
                _rx = _a * _rx + (1.0 - _a) * self.x
                _ry = _a * _ry + (1.0 - _a) * self.y
        _raw_x, _raw_y = float(r["x"]), float(r["y"])
        self.x, self.y = _rx, _ry
        r = dict(r, x=_rx, y=_ry,
                 smoothed=bool(abs(_rx - _raw_x) > 1e-9 or abs(_ry - _raw_y) > 1e-9))
        # ⚠⚠ **接受时"什么都不做"** ✗（第一版清零 ✗、第二版减一 ✗ 都被现场打脸 ✓）：
        #   计数现在是"**同一个时间窗里被丢了几次**"（见 `DOT_JUMP_WINDOW_S` ✓）——
        #   它只由"丢"推进、由"时间窗过期"回落 ✓ ⇒ **接受不再影响它** ✓。
        #   （原因：丢掉之后下一拍整幅搜往往又接受了**同一个假点** ⇒ 若按"接受"回落 ⇒
        #     计数永远攒不到阈值 ⇒ 出口永远不开 ✓ 现场就是这么打脸的 ✓。）
        self.hits += 1
        self.missed = 0
        self._shifted = 0.0
        self._last_seen = now
        self._last = dict(r)
        # ⭐⭐ **喂一把尺子**（2026-10-11 ✓ 见 `_exp_hist` 那段 ✓）：只有**真检出**才算
        #   （幽灵是外推的 ✗ 拿它量尺寸会把尺子带歪 ✓）；中位数 ⇒ 抗单帧怪块 ✓。
        try:
            _side = 0.5 * (float(r.get("w") or 0.0) + float(r.get("h") or 0.0))
            if _side >= 2.0:
                self._exp_hist.append(_side)
                if len(self._exp_hist) > DOT_EXP_HISTORY:
                    del self._exp_hist[:-DOT_EXP_HISTORY]
                if len(self._exp_hist) >= DOT_EXP_MIN_SAMPLES:
                    self._exp_side = float(np.median(self._exp_hist))
        except (TypeError, ValueError):
            pass
        return dict(r, hits=self.hits, held=False, missed=0,
                    confirmed=self.hits >= DOT_CONFIRM_FRAMES)

    # ---------------- 内部 ----------------

    def _search(self, panel, calib, terrain, roi):
        """这一拍在哪搜：有上一帧位置 → 它周围一小块（`near` 语义）；否则全画面。

        ⭐⭐ **两版合一的最终形**（用户 2026-10-06 ✓ 正确性现场见 SKILL **285** ✓、
          帧率现场见 SKILL 286 ✓）：

          · **老版**（一直用到今天）把面板**裁成一小块**再调 `find_player_dot` ⇒
            **0.24 ms** ✓（比整幅 **62 ms** 便宜 250 倍 ✓）—— 可那一趟**不传标定/地形** ✗
            ⇒ 核心色层 / 底图相减层 / 差分层 / 平台闸**全被跳过** ✗ ⇒ 在被沙淹的画面上
            "窗口里总能挑到个假块" ⇒ `ok=True` ⇒ **永远回不到全画面** ✗ ⇒ 锁死 ✓；
          · **我上一版**改成"整幅算 + 用 `roi` 参数" ✓ 正确 ✓ 但 **62 ms/拍**，而且一帧里
            还算两遍（`_diff_at_global` ＋ 里面再算一次 ✗）⇒ 实测 `locate_ms` **104.74 ms** ✗
            ⇒ `proc_fps` 掉到 **33** ✗（`tools/_bench_locate.py` ✓）。

          ⇒ **最终形 = 便宜那条路 ＋ 把几何带全 ＋ 一道"找到的像不像玩家点"的闸**：
            ① **继续裁块**（便宜 ✓），但标定按裁剪**平移**后一起传进去（`_shift_calib` ✓）
               ⇒ 要几何的层**全都在** ✓（这是上一版买不到的 ✓）；
            ② 找到之后**再核一次**（`_local_result_trustworthy` ✓）：**不像玩家点** ⇒
               **丢掉、走全画面** ✓ —— 那一趟会用整幅的"最亮那团"（`at` ✓）挑 ⇒ **甩掉假锁** ✓
               （66 行处那个 `_diff_at_global` **删掉了** ✗：它的活儿现在由 ② 用**更便宜**
                的判据（核心色 ✓ 0.1 ms 级）承担 ✓，不必每拍烧 22 ms ✗）。
        """
        near = (self.x, self.y) if self._last is not None else None
        if near is not None:
            # 半径要**盖得住玩家点本身**（面板大，点也大：收流那条 753×612 上
            # 点是 28×24），不然点会被裁掉半截、重心也跟着偏。
            pad = max(self.roi_pad, dot_side_max(panel.shape[1]))
            used = roi_around(near, panel.shape, pad)
            self.last_roi = used
            if used is not None:
                x0, y0, x1, y1 = used
                sub = panel[y0:y1, x0:x1]
                _cal_sub = _shift_calib(calib, x0, y0)      # ⭐ 平移后的标定（见它说明 ✓）
                rr = find_player_dot(sub, _cal_sub, terrain,
                                     near=(near[0] - x0, near[1] - y0),
                                     ref_w=panel.shape[1])
                if rr["ok"] and _local_result_trustworthy(sub, rr, _cal_sub, terrain):
                    # **坐标要加回偏移**，候选列表也一样（下游按面板坐标用）
                    return dict(rr, x=rr["x"] + x0, y=rr["y"] + y0,
                                layer="roi",
                                all=[dict(c, x=c["x"] + x0, y=c["y"] + y0)
                                     for c in rr["all"]])
                # 这一小块里没有 / 找到的不像玩家点 ⇒ 退回全画面 ✓
                # （可能真换了地方 / 标定变了 / **旧锁本身就是个假点** ✓）
        self.last_roi = roi
        return find_player_dot(panel, calib, terrain, roi=roi,
                               exp_side=self._exp_side)

    def _drift(self):
        """漏检时按（衰减的）速度往前推一点，总位移有上限（同 MobTracker._drift）。"""
        step = DOT_GHOST_DECAY ** max(0, self.missed - 1)
        dx, dy = self._vx * step, self._vy * step
        d = (dx * dx + dy * dy) ** 0.5
        left = max(0.0, self.ghost_max_shift) - self._shifted
        if d > left and d > 1e-9:                 # 到了上限就不走了
            k = max(0.0, left) / d
            dx, dy = dx * k, dy * k
        self._shifted += (dx * dx + dy * dy) ** 0.5
        return (self.x or 0.0) + dx, (self.y or 0.0) + dy


# ══════════════════════════════════════════════════════════════
# 玩家定位（S3 的成品）：面板画面 → 世界坐标 → 在哪条段
# ══════════════════════════════════════════════════════════════

class PlayerLocator:
    """把三件事串成**一次调用**：黄点识别 → 世界坐标 → 站在哪条段。

        PlayerDotTracker      面板里找玩家标记（黄点，含跨帧确认）
          ↓ panel_to_world    面板像素 → 底图像素 → 世界坐标（用标定 + miniMap 的 mag）
          ↓ segment_of        世界坐标 → 地形里的哪条段（Segment.index）

    **为什么非得有人把它串起来**：这三步以前分别散在「标定弹窗」和「命令行诊断」
    里，各自只做一段 —— 真接进实时回路时又得再拼一遍，而拼错的方式很隐蔽
    （面板↔底图漏一次换算，或者把段号当成别的什么编号）：算出来的位置整体错，
    下游却当成"我在哪块平台上"照用。所以这里把口径固定死，谁都别再拼第二份。

    **算不出来是常态，也是结论**：`world_x/world_y/segment_id` 一律是 None 而不是
    0（0 是合法的世界坐标 —— 地图西北角），`note` 写清为什么。
    """

    def __init__(self, map_id=None, track=None):
        #: 跟踪参数（`PlayerDotTracker` 的关键字：hold_ms/roi_pad/max_jump/
        #: ghost_max_shift）。**必须记在 locator 上**：换图会重建 tracker，
        #: 不记的话"界面里调过一次、一换图又回默认"。
        self.track = {k: v for k, v in (track or {}).items() if v is not None}
        self.tracker = None
        self.map_id = None
        self.terrain = None
        self._calib_cache = {}
        #: 「坐标系偏移」缓存（见 _world_offset）：(时刻, (x, y))
        self._off_cache = None
        #: ⭐ **局部小地图（crop）的运行时 view 跟踪**（用户 2026-09-29 任务 1 ✓）——
        #: crop 的面板**随玩家滚动**，标定文件里那个 `view` 只在标定那一刻成立
        #: ⇒ 每拍得重新问"现在显示的是哪一块"（理由详见 `CropViewTracker` ✓）。
        #: 挂在 locator 上（**有状态**：跨帧记"上一拍滚了多少"给附近找定余量 ✓），
        #: 与 locator 同寿命；`load()`（换图）时清掉 ✓。
        self._view_track = CropViewTracker()
        #: ⭐⭐ **定位保底**（2026-10-13 ✓ 见 `PLAYER_FB_EMA_ALPHA` 那段 ✓）：
        #: `None` = **这张图上还没定过位** ⇒ 认不出时照旧如实说找不到 ✓（不猜 ✓）；
        #: 定过之后就变成一份 `{world_x, world_y, segment_id, foothold_id, at}` ✓
        #: （`at` = **上一次真定位**的时刻 ✓ 算"沿用多少秒"用 ✓ 喂 EMA 只喂真检出 ✓）。
        self._fb = None
        self._new_tracker()
        self.load(map_id)

    def _new_tracker(self):
        self.tracker = PlayerDotTracker(**self.track)

    def set_track(self, **kw):
        """改跟踪参数（界面拨一下立刻生效，并**换图也保留**）。"""
        for k, v in kw.items():
            if v is None:
                self.track.pop(k, None)
            else:
                self.track[k] = v
        self._new_tracker()
        return self

    def use_track_config(self, live_cfg):
        """按 `config/live.yaml` 那份配置更新跟踪参数（界面那排输入框改完就叫它）。

        键名与默认值只在 `tracker_kwargs()` / `track_params()` 里写一份。
        """
        return self.set_track(**tracker_kwargs(track_params(live_cfg)))

    def load(self, map_id):
        """（换）加载某张图的地形。传 None 就是"不定位"。"""
        map_id = map_id or None
        if map_id == self.map_id and self.terrain is not None:
            return self
        self.map_id = map_id
        self.terrain = (mapdata.load(map_id, with_canvas=True)
                        if map_id else None)
        # 换图 → 位置不连续，别拿上一张图的点去跟踪（跳变判据会乱丢）。
        # ⚠ 重建时把界面调的跟踪参数带上（见 self.track）。
        self._new_tracker()
        # ⭐ **保底也跟着换图清掉**（2026-10-13 ✓）：上一张图的世界坐标在另一张图上
        #   毫无意义 ✗ ⇒ 清掉之后新图上"没定过位"照旧如实说找不到 ✓（不猜 ✓）。
        self._fb = None
        self._calib_cache = {}
        # 换图 ⇒ "现在显示的是底图哪一块"整个换了一套（底图都不一样）⇒ 忘掉跟过的位置 ✓
        self._view_track.reset()
        return self

    def calib_for(self, src, ttl=1.0, zoom=None):
        """读某条来源的标定，**带 1 秒缓存**。

        为什么不每次读文件：实时回路 30 拍/秒，每拍读一次 JSON 纯属浪费；
        为什么不只在启动时读一次：标定在弹窗里随时会被改，读死了要重启才生效。
        1 秒的折中足够（弹窗保存后本来就会自己刷新一次）。

        ⭐ `zoom`（**可选**，不传 = 与从前完全一样）：只肯要**在这个 zoom 下标出来的**
           那份（见 `mapdata.zoom_of` / `mapdata.load_calib`）。zoom 变了 ⇒ 返回
           **None**（不是"拿旧的凑合"）⇒ 上层会说"这份标定不适用"，而不是闷头给出
           整倍数错的坐标 ✗。
           ⇒ **换图之后一定要带上 `switch_map` 回回来的那个 zoom**：同一张图换个 zoom
           推，面板像素尺寸就变了，而标定里的 `scale`/`offset` 是对着当时那个尺寸
           量出来的（这正是 2026-10-01 修的那个坑）。
        """
        now = time.monotonic()
        key = (src, int(zoom) if zoom is not None else None)
        hit = self._calib_cache.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
        # ⚠ **不传 zoom 时必须用"老调用形式"**（2026-10-01 修 ✗）：`tools/selftest_minimap`
        #   的「实测精度」用例把 `load_calib` 换成了只认 `(_m, src=None)` 的替身 ⇒
        #   一律多传一个 `zoom=None` 会当场 `TypeError`（那个用例就这么挂的 ✓）。
        #   ⇒ **可选参数不该改变老调用形式** —— 这条对所有可能被替身/被 mock 的函数都成立 ✓。
        if not self.map_id:
            cal = None
        elif zoom is not None:
            cal = mapdata.load_calib(self.map_id, src, zoom=zoom)
        else:
            cal = mapdata.load_calib(self.map_id, src)
        self._calib_cache[key] = (now, cal)
        return cal

    def forget_calib(self):
        """忘掉标定缓存 —— **标定刚被改过**（弹窗保存/重标）时叫它，下一拍立刻用新的。

        为什么要有这个入口：`calib_for` 带 1 秒缓存（实时回路 30 拍/秒，每拍读一次 JSON
        不值当）。那是给"一直跑着的回路"用的；而**人刚点完保存**时，那 1 秒的延迟会让人
        以为"保存没生效 / 读数根本没变"（用户 2026-09-27 报的就是这个形状），所以保存
        那条路上显式忘一次 ✓。别改成"每次都不缓存"：那是把 30 拍/秒的读盘成本还给实时回路。
        """
        self._calib_cache = {}
        return self

    def _world_offset(self, calib):
        """「坐标系偏移」(x, y)：算出来的世界坐标**加上**它（用户 2026-09-26 要求）。

        为什么要它：坐标是按**黄点的下沿（脚底）**算的（见 `dot_feet`），而
        "那个下沿 ↔ 游戏里的玩家原点"这层对应是游戏自己定的 —— 与其猜，不如给一个
        显式偏移让人自己量一次。读数、段号、foothold、绳梯判据都在**加完之后**算
        （见 `update`），所以寻路的"我在哪块平台上"也跟着准。
        ⚠ 锚点是下沿时（2026-09-27 起），这一项的正常值接近 **(0, 0)**（见 `offset_of`）。

        **按地图 id 存**（用户 2026-09-26 定的口径）：它跟标定走 —— 存在
        `datasets/map/<id>.mapcalib.json` 的 `sources.<来源>.world_offset`，
        和 `scale/offset/view/alpha` 同一份（每张图、每个来源各自一套 ✓）。
        直接读 `calib`（调用方已经按来源取好了），所以面板和实时线程天然一致。
        """
        return map_calib.offset_of(calib)

    @staticmethod
    def foothold_gap(f, wx, wy):
        """人在 `(wx, wy)` 时，脚下那条 foothold 的**面**与人差多少像素（>0 = 人在上方 ✓）。

        ⭐ 用户 2026-10-18 ✓ 原话：「**离下方 foothold y 差距如果超过 200 不应该算到达！**
          当前的逻辑会影响各类寻路 / idle 计时的判定，我还在半空中没到点就开始计时了」✓
        世界 y **越小越靠上** ⇒ `面y - 人y` > 0 就是"人还悬在这块平台上方" ✓。
        取 `(wx, wy)` 里 x 对应的面 y（平台可能是斜的 ✓ 与 `foothold_at` 同一套插值 ✓）。
        返回 `None` = 判不了（没有 foothold ✓）。
        """
        if f is None:
            return None
        try:
            fx = min(max(float(wx), float(f.left)), float(f.right))
            return float(f.y_at(fx)) - float(wy)
        except (TypeError, ValueError):
            return None

    def update(self, panel, src=None, calib=None, terrain=None, fh_xtol=0,
               air_gap_px=0, frame_wh=None):
        """面板画面（BGR）→ 定位结论 dict。

        返回：
            ok         这一拍**拿到了世界坐标**（认不出点 / 没标定 → False）
            confirmed  黄点是"连续两拍都在附近"确认过的（决策层该只认这种）
            px, py     黄点在面板里的像素（**重心** —— 画标记用；认不出时是上一次的残值，别用）
            world_x/world_y  世界坐标；没算出来是 None
                       ⚠ 它取的是黄点**下沿（脚底）**换算出来的那个点（`dot_feet`）+
                       「坐标系偏移」，**不是** `px/py` 直接换算 —— 采样（双点标定）
                       用的也是下沿，两边同一口径读数才对得上（见 `dot_feet`）
            segment_id 站在哪条段（None = 没落在平台上）
            foothold_id      踩着哪条 foothold（字符串 id；None = 没落在平台上）。
                       **"我在不在某个 foothold 集合里"用这个，不用 segment_id**：
                       自动串段会把"要跳/攀才能互通"的并成同一段（实测 105090600 的
                       第 0 段把一面 388 像素高的悬崖当成了平台边缘），集合是人工圈的，
                       判据只能是 foothold id ∈ 集合（见 core/zones.py、docs/寻路设计.md §12）
            ladder_id        踩在**哪根绳梯**上（"L2" 这种编号；None = 不在绳上）。
                       编号来自 `zones.ladder_ids` —— 与编辑器画在绳上的、以及「爬」那条
                       边里写的绳号**同一套**。判据是几何的（`mapdata.ladder_at`），
                       和执行器"对齐到绳的 x"用的**判定参数**是两回事（见设置→判定参数）
            dot        黄点那一层给的说明（认不出时就是原因）
            view/view_ok/view_score/view_trust/view_why
                       ⭐ **局部小地图（crop）的运行时跟踪**（用户 2026-09-29 任务 1 ✓）：
                       `view` = **这一拍算坐标时用的**「显示区起点」（底图像素）；`view_ok` =
                       这一拍真的**跟住了**吗（`None` = 不是 crop，没这回事 ✓）；
                       `view_score` = 跟的匹配分；`view_trust` = 到没到 `TRUST_SCORE`
                       那档（`False` = 跟住了不太稳，典型是面板被挡掉一块 ✓）；
                       `view_why` = 没跟住的原因。
                       ⚠ 跟不住时用的是**标定里那份**（= 老行为）⇒ `view_ok=False` 就意味着
                       "这一拍的世界坐标可能整体偏"（crop 会滚动！见 `CropViewTracker` ✓）。
            note       没算出坐标 / 没落平台的原因（**可能很长**，给 tooltip/日志）
            short      `note` 的一句话版本（正常时是空串）——给**贴在画面上**的
                       读数用：那是一行不换行的字，塞下 `note` 会横穿整个画面
        """
        src = src or map_const.SRC_STREAM
        terrain = terrain if terrain is not None else self.terrain
        # `fh_xtol`：「脚下是哪条 foothold / 哪条段」的 **x 方向容差**，由调用方从
        # 设置里的「坐标对齐误差范围」传进来（0 = 老行为）。只兜"原本找不到"的情形，
        # 详见 `core.mapdata.Terrain.foothold_below` 的 xtol 说明 ✓。
        _xt = max(0, int(fh_xtol or 0))
        if calib is None:
            calib = self.calib_for(src)
        # ⭐⭐⭐ **面板尺寸变了 ⇒ 能按比例救就救，救不了才拒**（用户 2026-10-20 ✓ 原话：
        #   "为什么经常出现这个问题" ⇒ 选了方案 A「让标定也跟着比例走」✓）
        #   · 病根：`mmap_crop` 有比例副本（帧尺寸变了自动跟着缩 ✓），可标定钉的是**像素**
        #     尺寸 ✗ ⇒ 改推流分辨率 / 挪游戏窗口之后，**所有图**的标定一起作废，
        #     而症状看起来是"认不出黄点"（2026-10-20 天空 ✓ 见日志 506）。
        #   · 救的条件**必须精确**（`map_calib.adapt_panel` ✓）：标定里记了 `panel_rel` ＋
        #     调用方给了这一帧的尺寸 ⇒ 反推"区域没变的话面板该多大"，对得上才按比例换算 ✓；
        #     对不上（人真重框了）⇒ 不救 ✓，照旧走下面那道 `panel_mismatch` 拒用 ✓。
        #   · ⚠ **必须放在"跟显示区"之前**：那一步（`CropViewTracker`）也吃 `scale`/`offset` ✓，
        #     拿没换算的几何去跟，会把 view 量歪 ✗。
        _panel_adapted = None
        if frame_wh is not None:
            _cw = (terrain.canvas.shape[1], terrain.canvas.shape[0]) \
                if (terrain is not None
                    and getattr(terrain, "canvas", None) is not None) else None
            _pm0, _ = map_calib.panel_mismatch(calib, panel, _cw)
            if not _pm0:
                _ad, _adinfo = map_calib.adapt_panel(calib, panel, frame_wh)
                if _ad is not None:
                    calib, _panel_adapted = _ad, _adinfo
        # ⭐⭐ **局部小地图（crop）：先把"现在显示的是底图哪一块"量出来**（用户 2026-09-29
        #   任务 1 ✓）。crop 的面板**随玩家滚动** ⇒ 标定文件里那个 `view` 只在标定那一刻
        #   成立，人一走后面所有换算都是整体平移的 ✗（理由与量级见 `CropViewTracker` ✓）。
        #   ⚠ **跟住才改几何**：没跟住时用标定里那份 = 今天的行为一字不变 ✓，只把
        #     "这一拍没跟住"写进 note/short（否则人会拿着一个偏了 370 世界像素的读数
        #     当准的用 ✗ —— 那正是"世界坐标整体偏"那类问题的样子 ✓）。
        #   ⚠ 必须在**黄点识别之前**跑：黄点的搜索 ROI / 底图遮罩都吃 `view`
        #     （`_calib_roi` / `_basemap_extra_mask` ✓）⇒ 跟住了连"认黄点"都更准 ✓。
        _vt = {"ok": None, "view": None, "score": None, "why": ""}
        _crop_note = ""
        if (calib or {}).get("mode") == map_const.MODE_CROP \
                and terrain is not None and getattr(terrain, "canvas", None) is not None:
            _vt = self._view_track.update(panel, terrain.canvas, calib)
            # ⭐⭐⭐⭐ **地形掩码复核**（2026-10-19 现场 ✓ "森林迷宫_珠缨 局部小地图的匹配完全错误"）：
            #   跟出来的 `view` **不许直接采信** —— 得先拿「地形重合度」(`calib_overlap`，
            #   走 `terrain_mask` ✓) 跟**标定里那份**比一比，**明显更差就不用**（退回标定 ✓）。
            #
            #   为什么非要复核（这一次的全部证据 ✓）：
            #     · 这张图（105040303）底图是**重绘的小地图**、面板是客户端渲染 ⇒ 逐像素相关
            #       在这张图上**几乎是平的**：现场沿 `view.y` 从 80 扫到 175，分数只有 **0.70~0.73**
            #       （峰间距 12~13 像素、高差 0.005）✗ ⇒ 谁当头都对，跟出来的位置等于**随机**
            #       （用户看到的就是"完全错误"✓）；
            #     · 而同一帧用**地形掩码**量：跟出来的那份 **0.095**、标定那份 0.265、
            #       重拟的那份 **0.438** ⇒ 只有掩码这把尺子分得开 ✓✓（`map_calib` 里 2026-10-09
            #       就为这张图写过这条结论，只是当时只用在标定弹窗，**没进实时回路** ✗）。
            #   ⇒ 判据：`iou(跟到的) + CROP_MASK_MARGIN < iou(标定的)` ⇒ 判"这一拍没跟住"
            #     （`ok=False` ⇒ 用标定那份，和"没跟住"是同一条老路 ✓ 界面也照样提示 ✓）。
            if _vt.get("ok") and _vt.get("view"):
                # ⭐⭐⭐ **没过 `TRUST_SCORE` 那档的结果要"严格版"复核**（2026-10-20 ✓
                #   用户："现在会**乱飞乱抖**，匹配的完全不对"✗）：放宽档 / 帧间位移
                #   在这张**逐像素平掉**的图上可能刚好差一个峰（≈600 世界像素 ✗），
                #   而"允许差 0.06"的宽判据挡不住它 ⇒ 每 30 拍左右跳一下 ✓
                #   ⇒ 这两类必须**明显更好**才采信（判据与实测见 `crop_view_trusted` ✓）。
                _trust, _iou_new, _iou_ref = crop_view_trusted(
                    panel, terrain.canvas, calib, _vt["view"],
                    strict=not _vt.get("trust"))
                if not _trust:
                    _vt = dict(_vt, ok=False, trust=False,
                               why="跟到的显示区和底图对不上（地形重合 %.2f，"
                                   "标定那份 %.2f；%s）"
                                   % (_iou_new, _iou_ref,
                                      "放宽档/帧间位移要明显更好才采信"
                                      if not _vt.get("trust") else "不比标定那份好"))
                    # ⭐ 复核没过 ⇒ 让跟踪器**把这一拍的 view 撤回去**（不然它带着
                    #   这个错位置继续"附近找"，下一拍还会再错一次 ✗）。
                    try:
                        self._view_track.reject(str(_vt.get("why") or ""))
                    except Exception:          # noqa: BLE001 —— 撤不回去不影响这一拍 ✓
                        pass
            if _vt.get("ok") and _vt.get("view"):
                calib = dict(calib, view=[float(v) for v in _vt["view"]])
                if not _vt.get("trust"):
                    # 跟住了但不够可信，两种来源要分开说（2026-10-18）：
                    #  · `where="flow"`：**跨图匹配失败**、靠"相邻两帧面板位移"接住的（用户选的 A）
                    #    ⇒ 位置是**相对**的，误差会一拍拍累计 ⇒ 必须让人看出来 ✗；
                    #  · 其余：跨图匹配到了但分不高（典型：面板被挡掉一块 —— 平掉的那半
                    #    不贡献方差，分数会虚高 ✓）。
                    # 两种都**照样用**（比拿标定里那个旧位置强 ✓），但都要说出来 ✓。
                    if str(_vt.get("where") or "") == "flow":
                        _crop_note = ("局部小地图：底图对不上，这一拍是按**帧间位移**接住的"
                                      "（分 %.2f）—— 位置会累计漂移，别当准的用"
                                      % float(_vt.get("score") or 0.0))
                    else:
                        _crop_note = ("局部小地图：这一拍跟住了，但匹配分只有 %.2f（不太稳，"
                                      "面板被挡住 / 画面糊了？）"
                                      % float(_vt.get("score") or 0.0))
            else:
                # ⭐⭐⭐⭐ **跟不住的时候，改用「地形掩码」把 view 找回来**
                #   （用户 2026-10-20 ✓ 原话："森林迷宫_珠缨项目，**L9 完全找错了**"✗ 见日志 510）
                #   病根摆在一起看才清楚：
                #     · 这张图（105040303 重绘底图）**逐像素相关是平的**（天花板 0.55 ⇒
                #       谁当头都对 ✗）⇒ `CropViewTracker` 一丢锚就只能靠帧间位移，
                #       刹车之后 `view` **冻在标定那份** ⇒ 人一换层，世界坐标整体偏一层 ⇒
                #       绳梯/集合全认错（现场看到的就是"L9 找错了"✓）；
                #     · 而**地形掩码**分得开（同一帧：错的 0.157 / 标定 0.251 / 真值 0.376 ✓）
                #       ⇒ 干脆拿它当"找"的尺子（`crop_view_trusted` 早就拿它当"复核" ✓）。
                #   ⚠ 只在**跟不住**这一支里做 ✓（跟住了就别乱挪 ✗）；挪不回来就照旧用标定那份 ✓。
                # ⚠ 起点用**跟踪器现在这个**（不是标定文件里那个）：停车之后 `view` 只由
                #   这里往前带 ⇒ 人能走多远就跟多远 ✓（每拍 ±90 底图像素足够覆盖一步 ✓）；
                #   拿文件里那个当起点的话，人一走远就再也够不着了 ✗。
                _seed = getattr(self._view_track, "view", None) or calib.get("view")
                _rl_view, _rl_info = map_calib.relock_view(
                    panel, terrain.canvas, dict(calib, view=list(_seed)))
                if _rl_view is not None:
                    calib = dict(calib, view=[float(_rl_view[0]), float(_rl_view[1])])
                    # 跟踪器也拽到这儿，但**别标 `tracked`**：
                    #   这张图 0.8 的门槛它永远够不着 ⇒ 标上等于把自己锁死 ✗
                    try:
                        self._view_track.view = (float(_rl_view[0]), float(_rl_view[1]))
                        self._view_track.tracked = False
                    except Exception:          # noqa: BLE001 —— 拽不动不影响这一拍的结果 ✓
                        pass
                    _vt = {"ok": True, "view": _rl_view, "score": _rl_info.get("iou"),
                           "trust": True, "where": "mask", "relaxed": False}
                    _crop_note = ("局部小地图：这一拍按**地形掩码**找回来的（%s）"
                                  % (_rl_info.get("why") or ""))
                else:
                    _crop_note = ("局部小地图：这一拍没跟住显示区（%s；按掩码也找不回来：%s）"
                                  "⇒ 用的是标定里那个位置，世界坐标可能整体偏"
                                  % (_vt.get("why") or "说不清",
                                     _rl_info.get("why") or "说不清"))
        r = self.tracker.update(panel, calib=calib, terrain=terrain)
        out = {"ok": False, "confirmed": bool(r.get("confirmed")),
               # held：这一拍没认出黄点，位置是**上一帧**的（防抖窗口内沿用）。
               # 决策层可以用（两三拍之内还算得准），但要能分辨出来。
               "held": bool(r.get("held")), "missed": int(r.get("missed") or 0),
               "px": r["x"], "py": r["y"], "world_x": None, "world_y": None,
               "segment_id": None, "foothold_id": None, "src": src,
               #: ⭐ **空中态**（用户 2026-10-18 ✓）：人在脚下那条平台**上方超过 `air_gap_px`**
               #:   ⇒ `foothold_id` / `segment_id` 都给空 ✓、这里为 True ✓（界面「fh：空中」✓；
               #:     到达判定 / idle 计时 / 战斗区域归属 全都不在半空里成立 ✓）。
               "airborne": False,
               # 局部小地图（crop）的运行时跟踪（见上面那段；不是 crop 时一律 None ✓）
               "view": _vt.get("view"), "view_ok": _vt.get("ok"),
               "view_score": _vt.get("score"), "view_why": _vt.get("why") or "",
               "view_trust": _vt.get("trust"),
               "dot": r["reason"], "note": "", "short": "认不出黄点",
               # ⭐ **这一拍是哪一层认出黄点的**（`"color"` / `"basemap"` / `"diff"` ✓
               #   用户 2026-10-06："差分层字样哪里看"✓）—— 原来只塞在 `dot` 那句长理由里 ✗
               #   而那句**只在失败时**才有人显示 ⇒ 成功时"是不是差分层救回来的"**根本看不到** ✗
               #   ⇒ 单独立一个短字段，给日志 / 状态行用 ✓（一处算、多处读 ✓）。
               "dot_layer": str(r.get("layer") or ""),
               #: ⭐ 这份标定与**这块面板**对不上（见 `panel_mismatch` ✓）—— 拒用时为 True ✓
               "panel_mismatch": False,
               #: ⭐ 「面板尺寸变了、但**框选区域没变**」⇒ 几何已按比例换算、**没拒用** ✓
               #:   （用户 2026-10-20 方案 A ✓ 见 `map_calib.adapt_panel` / 日志 507）
               "panel_adapted": False,
               "panel_adapt_note": ""}
        if _panel_adapted:
            out["panel_adapted"] = True
            out["panel_adapt_note"] = str(_panel_adapted.get("why") or "")
        # ⭐⭐⭐ **用之前先问一句「这份标定对得上现在这块面板吗」**（用户 2026-10-06 ✓ 见
        #   `panel_mismatch` 里那段现场 ✓）—— 对不上就**拒用**：这时照用会让**世界坐标整倍数
        #   错**，而 `mmap_ok` 会一直是 1.00 ✗ ⇒ 决策层"**自信地往右怼墙**" ✓（现场那一场 ✓）。
        #   ⚠ 判据全用现成数（标定里记的 `panel` ✓ 或 `mode=fit` 反推 ✓ ＋ 这一帧的 shape ✓），
        #     不新造参数 ✓；推不出就**不拦** ✓（别把没证据的事判死 ✗）。
        _pm_ok, _pm_why = map_calib.panel_mismatch(
            calib, panel,
            (terrain.canvas.shape[1], terrain.canvas.shape[0])
            if (terrain is not None
                and getattr(terrain, "canvas", None) is not None) else None)
        if not _pm_ok:
            out["panel_mismatch"] = True
            out["note"] = _pm_why
            out["short"] = "标定对不上这块面板 ⇒ 已拒用（请重标定）"
            return self._fallback(out, "标定与面板对不上")
        if not r["ok"]:
            # ⭐⭐⭐ **认不出黄点时，把"程序自己看到的这块面板"落一张图**（2026-10-18 ✓ 用户
            #   「黄点永远认不出来」而我这边抓同样的框 **20/20 都认得出** ⇒ 唯一能定案的办法
            #   是看**程序的实际输入**，别再靠猜 ✗）。
            #   · 节流 5 秒一张（`_dump_fail` 自己管 ✓）⇒ 不会把磁盘写爆 ✓；
            #   · 只在失败时写 ✓；写成 `datasets/map/_mmap_fail.png`（**固定名**，后写的盖前一张 ✓）。
            _dump_fail_panel(panel, r, map_id=self.map_id, src_name=src)
            # ⭐⭐ **认不出黄点 ⇒ 走保底**（2026-10-13 ✓ 用户："定位后，不准找不到"✓）：
            #   这就是"石人寺院III 发呆"那条链的**出口** ✓（原来这里直接 return ⇒
            #   `world_x=None` ⇒ 界面说"没定上位" ⇒ Agent 判未定位 ⇒ 发呆 ✓）。
            #   ⚠ 原因**照旧写进 note** ✓（保底没生效时它就是那句"认不出"✓；生效时保底会
            #     把它拼进自己的句子 ✓）—— 我第一版把这行**覆盖掉**了 ✗，用例当场抓住 ✓。
            out["note"] = r["reason"]
            return self._fallback(out, r["reason"])
        if terrain is None:
            out["note"] = "没有这张图的地形/底图（先「生成地形图」）"
            out["short"] = "没有地形数据"
            return self._fallback(out, "没有这张图的地形/底图")
        if not map_calib.has_geometry(calib or {}):
            # ⭐⭐ **先在"这条来源"上自动补一次**（用户 2026-10-11 ✓ 原话："**为什么要我补标定？
            #   标定的数据不是都存好了？**"✓）：标定是**按来源分开存**的 ✓（同一张图两条来源的
            #   面板实测差 5.6 倍 ⇒ 不能互换 ✗），而"自动量一遍"这条路是通的 ✓
            #   （`locate_fit` 实测 0.9388 ✓）⇒ 缺哪条补哪条 ✓，**不用人动手** ✓。
            _af = try_autofit(panel, getattr(terrain, "canvas", None), self.map_id, src,
                              seed=calib)
            self.autofit = _af                     # 给界面/日志看（"自动试过/补上了" ✓）
            if _af.get("ok") and not _af.get("already"):
                self.forget_calib()                # 缓存里那份是"没有" ⇒ 立刻重读 ✓
                calib = self.calib_for(src)
        if not map_calib.has_geometry(calib or {}):
            _af_why = str((getattr(self, "autofit", None) or {}).get("why") or "")
            out["note"] = ("还没量过这张图的「面板 → 底图」换算"
                           "（来源「%s」下点「标定…」）%s"
                           % (map_const.SRC_LABEL.get(src, src),
                              ("；自动补标定：" + _af_why) if _af_why else ""))
            out["short"] = "还没标定"
            return self._fallback(out, "这条来源还没标定")

        # ⚠ **取黄点的下沿（脚底），不是重心**（2026-09-27 改，见 `dot_feet`）：
        #   双点标定采样时填的就是下沿 ⇒ 两边不同口径的话，站回同一个地方读数却
        #   差 (h-1)/2 个面板像素（实测 6 像素高的点 = 12 世界像素，用户报过），
        #   而那个差**看着像标定不准**、最难查 ✗。`px/py`（画标记用）仍是重心 ✓。
        fx, fy = dot_feet(r)
        wx, wy = map_calib.panel_to_world(fx, fy, calib, terrain)
        # 「坐标系偏移」：**在这一步加**（用户自己精确标定用）—— 后面的范围检查、
        # 段号、foothold、绳梯判据全都用加完之后的坐标，口径才是一份。
        # 锚点换成下沿之后，这一项的含义就是"下沿 → 玩家原点"的**残差**（正常接近 0）。
        ox, oy = self._world_offset(calib)
        if ox or oy:
            wx, wy = wx + ox, wy + oy
        out["world_x"], out["world_y"] = wx, wy
        b = terrain.bounds
        if b and not (b[0] - 1 <= wx <= b[2] + 1 and b[1] - 1 <= wy <= b[3] + 1):
            # 落在图外基本只有一个原因：标定（或底图）不对。说出来，别把
            # 一个地图外的坐标交给寻路。
            out["note"] = ("算出来在底图范围外 (%.0f, %.0f) —— 标定或底图不对？"
                           % (wx, wy))
            out["short"] = "算到图外了"
            return self._fallback(out, "算到图外了")
        seg = terrain.segment_of(wx, wy, xtol=_xt)
        out["segment_id"] = seg.index if seg is not None else None
        # 脚下**那条** foothold 的 id（字符串，和 zones 文件里的写法一致）。
        # 与 segment_of 是同一套口径（都走 foothold_below），多算一次可忽略。
        f = terrain.foothold_below(wx, wy, xtol=_xt)
        out["foothold_id"] = str(f.fid) if f is not None else None
        # **在不在绳梯上**（2026-09-26 用户要求：小地图那行要能报「绳梯：L2」）。
        # 判据用 mapdata 现成的 `ladder_at`（绳的 x ± LADDER_DX、y 在绳段 ± LADDER_PAD）；
        # 编号用 `zones.ladder_ids` —— 与编辑器画在绳上的 L1/L2、以及「爬」那条边里写的
        # 绳号**必须是同一套**（不然同一个 "L2" 在两处指两根绳，人会被带沟里）。
        # ⚠⚠ **必须排在"空中判定"之前**（用户 2026-10-18："在绳梯上应该是不能被判属于空中的"✓）：
        #   挂在绳上时脚下那块平台**天然在很远的地方** ⇒ 先判空中就会把绳梯上的人判成"空中" ✗
        #   （现场症状：爬绳全过程 `airborne=True`、`foothold_id`/`segment_id` 被清空、
        #    小地图那行写"空中"而不是"绳梯 L2" ✓）。
        L = terrain.ladder_at(wx, wy)
        _lad = zones.ladder_ids(terrain).get(id(L)) if L is not None else None
        out["ladder_id"] = _lad
        # ⭐⭐⭐⭐ **空中态**（用户 2026-10-18 ✓ 见 `air_gap_px` 那段说明）：
        #   脚下那条平台在**很下面**（人还在半空往下落 / 跳过去没落地）⇒ **不算站在它上面** ✗
        #   ⇒ `foothold_id` 与 `segment_id` 一并清空 ✓（下游 `here_sets` 跟着空 ✓
        #     ⇒ 到达判定 / idle 计时 / 战斗区域归属 都不会在半空里提前成立 ✓）。
        #   ⚠ 判据用**现成几何**（`foothold_gap` ✓ 与 `foothold_below` 同一套插值 ✓），
        #     不新造第二把尺 ✗；`air_gap_px <= 0` ⇒ 整段不判（老行为一字不变 ✓）。
        #   ⚠⚠ **在绳梯上 ⇒ 一概不算空中**（`L is None` 才进这条）：人挂在绳上是"正常攀爬"，
        #     不是"半空掉下来" ✓ ⇒ `airborne` 保持 False、报「绳梯：Lx」✓；
        #     但**照样不算站在平台上**（平台还在脚下很远）⇒ `foothold_id`/`segment_id` 照旧清空 ✓
        #     （到达判定/区域归属不会被绳上的人提前触发 ✓）。
        _air_gap = max(0.0, float(air_gap_px or 0.0))
        if f is not None and _air_gap > 0:
            _gap = self.foothold_gap(f, wx, wy)
            if _gap is not None and _gap > _air_gap:
                out["foothold_id"] = None
                out["segment_id"] = None
                if _lad:
                    out["note"] = ("绳梯：%s（人在下面那块平台上方 %.0f px，正挂在绳上 "
                                   "⇒ **不算空中**）" % (_lad, _gap))
                    out["short"] = "绳梯 %s" % _lad
                else:
                    out["airborne"] = True
                    out["note"] = ("空中：人在下面那块平台上方 %.0f px（> 空中判定 %.0f）"
                                   "⇒ 不算站上去" % (_gap, _air_gap))
                    out["short"] = "空中"
        out["ok"] = True
        if out["airborne"] or _lad:
            # ⚠ `seg is None` 在这两种情况下**必然成立**（脚下没平台 ✓）——
            #   原来那条 `if seg is None` 会把上面写好的「空中 / 绳梯 Lx」**覆盖成**
            #   「脚下没有平台（半空 / 墙里？）」✗（现场：绳梯上显示的就是这句 ✓）⇒ 先让开 ✓
            pass
        elif seg is None:
            out["note"] = "坐标算出来了，但脚下没有平台（半空 / 墙里？）"
            out["short"] = "脚下没有平台"
        elif out["held"]:
            # 位置是上一帧的（这一拍没认出黄点，防抖窗口内沿用）—— 说出来，
            # 别让人以为这是这一拍量出来的
            out["note"] = ("这一拍没认出黄点，用的是上一帧的位置（漏检 %d 拍）"
                           % out["missed"])
            out["short"] = "上一帧位置"
        else:
            out["short"] = ""
        if _crop_note:
            # 局部小地图没跟住 ⇒ **必须说出来**（这一拍的世界坐标可能整体偏了"滚了多少"✗）：
            # 拼在已有 note 后面（原来那句可能是"脚下没有平台"，两条都要看得见 ✓）；
            # short 只在还没有话说的时候顶上（画面上那行越短越好，见它的说明 ✓）。
            out["note"] = (out["note"] + "；" if out["note"] else "") + _crop_note
            if not out["short"]:
                # ⚠ 话说准（用户 2026-10-20 ✓）：按掩码**找回来了**就别说"没跟住" ✗
                #   （那是两码事：`where="mask"` 这一拍的位置是**绝对**的 ✓ 可以用 ✓）
                if _vt.get("ok") and str(_vt.get("where") or "") == "mask":
                    out["short"] = "局部小地图（掩码找回）"
                elif _vt.get("ok"):
                    out["short"] = "局部小地图不太稳"
                else:
                    out["short"] = "局部小地图没跟住"
        # ⭐⭐ **真定位 ⇒ 喂保底一把**（2026-10-13 ✓ 见 `PLAYER_FB_EMA_ALPHA` ✓）：
        #   ⚠ **只喂真检出**（`held` 那几拍不喂 ✗）—— 否则保底值会被它自己的旧值反复平均
        #     ⇒ 越拖越钝 ✓（"沿用值当观测"是这一层的经典自欺 ✓）。
        if not out["held"]:
            self._remember_pos(float(wx), float(wy), out)
        return out

    def _remember_pos(self, wx, wy, out):
        """把这一拍的**真定位**喂给保底（EMA ✓ + 带上段号/foothold ✓）。

        EMA 只在"已经有上一份"时才平均 ✓（首帧直接落值 ✓ 不带偏 ✓）。
        ⚠ 段号 / foothold 取**这一拍**的（它们本来就是"够用就行"的粗粒度 ✓：人是走过去的，
          不会瞬移 ✓ 而**坐标**才是要紧的那个 ✓）。
        """
        a = float(PLAYER_FB_EMA_ALPHA)
        if self._fb is not None and self._fb.get("world_x") is not None:
            wx = a * wx + (1.0 - a) * float(self._fb["world_x"])
            wy = a * wy + (1.0 - a) * float(self._fb["world_y"])
        self._fb = {"world_x": wx, "world_y": wy,
                    "segment_id": out.get("segment_id"),
                    "foothold_id": out.get("foothold_id"),
                    "at": time.monotonic()}

    def _fallback(self, out, why=""):
        """**认不出黄点时的保底**：用"最后一次定位的 EMA 值"顶上（用户 2026-10-13 ✓）。

        用户原话："**做保底：定位后，不准找不到，丢了就用上次EMA平滑的**"✓
        （配套现场：小面板上认不出黄点 ⇒ 界面"这一拍还没定上位" ⇒ Agent 未定位 ⇒ **发呆**✗）。

        ⚠ **从没定过位**（`self._fb is None` = 刚开 / 没标定 / 刚换图 ✓）⇒ **原样返回** ✓
          —— 照旧如实说"找不到" ✓（保底的前提就是"**定位后**" ✓ 不猜 ✗）。
        ⚠ 填的字段与真定位**一模一样**（`world_x/world_y/segment_id/foothold_id` ✓）⇒
          下游（`apply_to_player` / `LAST_POS` / 决策层 ✓）**一个字都不用改** ✓；
          多出来的只有如实标志：`held=True`（沿用 ✓）`stale=True`（保底值 ✓）
          `stale_s`（多少秒前那次真定位 ✓）+ 说得清的 `note/short` ✓。
        ⚠ `ok=True` 是**故意**的 ✓：口径同跟踪器自己的"幽灵帧"（`PlayerDotTracker` 沿用上一帧
          也报 `ok=True` + `held=True` ✓ 见那边 ✓）—— "有可用位置"与"这一拍量出来的"是两件事 ✓。
        """
        fb = self._fb
        if not fb:
            return out
        _age = max(0.0, time.monotonic() - float(fb.get("at") or 0.0))
        out.update(ok=True, held=True, stale=True, stale_s=round(_age, 1),
                   world_x=fb.get("world_x"), world_y=fb.get("world_y"),
                   segment_id=fb.get("segment_id"),
                   foothold_id=fb.get("foothold_id"))
        out["note"] = ("认不出黄点 ⇒ 用上次定位的 EMA 保底（%.1f 秒前那次真定位；原因：%s）"
                       % (_age, str(why or out.get("note") or "—")[:80]))
        # 画面那行越短越好（见 `note/short` 的说明 ✓）：给一句带秒数的 ✓
        out["short"] = "沿用上次定位(%.0fs)" % _age
        return out
