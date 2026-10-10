"""把外部 `visual_tracking` SDK 接成工作台**进程内**的一个薄封装（自动测谎用）。

为什么是"进程内薄封装"而不是旁路进程（2026-10-09 实测定的 ✓）：
  · `visual_tracking_sdk_20260920\\.venv` 是指向 `playerSimu\\.venv` 的 **Junction**
    （`Get-Item .LinkType = Junction` ✓）⇒ 两个工程**共用同一个 venv**：python 3.11.9、
    ultralytics 8.4.114、av 18.1.0、torch 2.11.0+cu128 **逐条一致** ✓，工作台里
    `import vision_sdk` 直接成功 ✓ ⇒ 不必旁路进程（省掉一整套 IPC ✗）。
  · ⛔ **绝不改 SDK 目录里的文件**：它是校验过的交付包（`verify_package.py` 逐文件比对
    sha256 ✓）。这里只把 SDK 根目录**塞进 `sys.path`** ✓，且**延迟 import**
    （没开这个功能时零成本 ✓ 少一个"启动就炸"的入口 ✗）。
  · 本模块**只做"喂帧 → 拿状态"**：不发鼠标、不碰 Qt、不打点 —— 那是调用方的事 ✓
    （同 SDK 自己的纪律：成功时 `reason='Host owns confirmation click'` ✓，
    `docs/INTEGRATION.md:105` 也明写"本接口不自动点击按钮" ✓）。

口径（都是拿真实素材量出来的，别随手改 ✗）：
  · **喂原生帧**（画框之前的那份 ✓ —— SDK 要求"输入图不能包含人为绘制的检测框"✓）；
  · `screen_origin=(origin_x, origin_y)` 交**截取区域在屏幕上的原点** ⇒ 回来的
    `Observation.aim_screen` 就是**屏幕物理坐标**，可直接喂鼠标后端 ✓（`INTEGRATION.md:34`）；
  · 时间戳必须**严格递增**⇒ 这里兜一层（调用方给重复值会 `ValueError` ✗）；
  · **截图尺寸/原点一变，SDK 自己会 `finish('capture_geometry_changed')`** ⇒ 这里对齐着
    `reset()` 一次，免得整个会话悄悄停摆 ✓。

实测（2026-10-09，素材 `D:\\Media\\record\\mxd\\10月7日.mp4`，RTX 5070）：
  · `prepare()`（YOLO 权重加载）**11.4s** ✓；`process` 中位 **37~39ms**、p90 172ms ✓；
  · 原始 1920×1080 帧：第 138 帧 LOCKED（ROI=(434,165,1055,703)，瞄点 (963,518)）✓；
  · 加工域 889×500 帧：第 58 帧 LOCKED（ROI=(201,77,488,325)）✓ —— 两边 ROI 按比例自洽 ✓。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

#: SDK 交付包默认落点（相对**仓库根**：本文件在 `perception/` 下 ⇒ 上跳一层 ✓）
DEFAULT_SDK_DIR = Path(__file__).resolve().parent.parent / "visual_tracking_sdk_20260920"

#: 成功面板匹配阈值：与 SDK 自己那份**同一个数**（`vision_sdk/success.py:36` 的 `.86` ✓）——
#: 换一个数就会出现"SDK 说成功、我们找不到按钮"（或反过来）那种互相打脸 ✗。
CONFIRM_MATCH_THR = 0.86


def frame_rect_of(obs):
    """把 SDK 的目标框（**追踪图坐标**）换算到**喂进去那张图**的坐标 ⇒ `(x0,y0,x1,y1)`。

    这条换算与 `docs/INTEGRATION.md:33` 给 `aim_frame` 的那条**是同一条**（同一份 roi 与
    追踪图尺寸 ✓）—— 只是它给的是点、这里给的是矩形 ✓。自己另写一套必然"框和点对不上" ✗。
    ⚠ 无框 / 无 roi / 无追踪图 ⇒ 返回 None（**不猜** ✓ 宁可不画 ✗）。
    """
    try:
        roi = getattr(obs, "roi", None)
        res = getattr(obs, "result", None)
        bbox = getattr(res, "target_bbox", None)
        tf = getattr(obs, "tracking_frame", None)
        if not roi or bbox is None or tf is None:
            return None
        rx, ry, rw, rh = (float(v) for v in roi)
        th, tw = int(tf.shape[0]), int(tf.shape[1])
        if tw <= 0 or th <= 0:
            return None
        sx, sy = rw / float(tw), rh / float(th)
        return (rx + float(bbox.x0) * sx, ry + float(bbox.y0) * sy,
                rx + float(bbox.x1) * sx, ry + float(bbox.y1) * sy)
    except Exception:                    # noqa: BLE001 —— 画不出来不能拖垮实时回路 ✓
        return None


class VtSdkError(RuntimeError):
    """SDK 不可用（目录不对 / 缺依赖 / 权重缺失）—— 调用方据此**停用**这项功能，别硬撑。"""


class VtLieSdk:
    """一帧进、一个 `Observation` 出的同步封装（**单一所有者**：只在实时线程里喂 ✓）。

    生命周期：`load()`（可后台）→ `feed(...)` 若干拍 → `reset()`（换任务）/ `close()`。
    """

    def __init__(self, sdk_dir=None, *, confidence=0.15, region="CN",
                 success_detection=True, max_duration_s=120.0):
        self.sdk_dir = Path(sdk_dir) if sdk_dir else DEFAULT_SDK_DIR
        self.confidence = float(confidence)
        self.region = str(region or "CN").upper()
        self.success_detection = bool(success_detection)
        self.max_duration_s = float(max_duration_s)
        # 运行时状态（只读给界面/日志用 ✓）
        self.ready = False
        self.loading = False
        self.error = ""                     # 加载失败的原因（人话 ✓）
        self.load_ms = 0.0
        self.frames = 0                     # 喂进去多少帧（诊断用 ✓）
        self.last_phase = "-"
        self.last_aim_screen = None
        self.last_roi = None
        self.shapec = None                  # 上一拍的 (shape, origin)：一变就 reset ✓（见模块头 ✓）
        self._sess = None
        self._obs_cls = None
        self._lock = threading.Lock()        # 只保护"加载"与"喂"不并发 ✓（喂永远在实时线程 ✓）
        self._thr = None
        # 成功面板/确认按钮模板（懒加载 ✓）
        self._cg = None
        self._qr = None
        self._btn_ratio = None

    # ---------------- 加载 ----------------

    def _import_sdk(self):
        """把 SDK 根目录塞进 `sys.path` 并 import（**幂等** ✓）。"""
        root = str(self.sdk_dir)
        if not Path(root).is_dir():
            raise VtSdkError("SDK 目录不存在：%s" % root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from vision_sdk import Session, Observation      # noqa: F401
        except Exception as e:                               # noqa: BLE001
            raise VtSdkError("import vision_sdk 失败（%s: %s）" % (type(e).__name__, e))
        return Session, Observation

    def load(self, *, blocking=True):
        """加载模型。`blocking=False` ⇒ 起后台线程（界面别被 11 秒卡住 ✓）。"""
        if self.ready or self.loading:
            return self.ready
        self.error = ""
        if not blocking:
            self.loading = True
            self._thr = threading.Thread(target=self._load_worker, name="vt_sdk_load",
                                         daemon=True)
            self._thr.start()
            return False
        self._load_worker()
        return self.ready

    def _load_worker(self):
        try:
            Session, Observation = self._import_sdk()
            _t0 = time.perf_counter()
            sess = Session(confidence=self.confidence, region=self.region,
                           success_detection=self.success_detection,
                           max_duration_s=self.max_duration_s)
            sess.prepare()                    # 阻塞加载 YOLO 权重（实测 ~11s ✓）
            with self._lock:
                self._sess, self._obs_cls = sess, Observation
            self.load_ms = (time.perf_counter() - _t0) * 1000.0
            self.shapec = None
            self.ready = True
        except Exception as e:                # noqa: BLE001 —— 加载失败只记原因，不抛到实时线程 ✓
            self.error = "%s: %s" % (type(e).__name__, e)
            self.ready = False
        finally:
            self.loading = False

    # ---------------- 会话 ----------------

    def reset(self, roi=None):
        """开一轮新任务（清身份/平滑/轨迹/成功计数 ✓）。

        ⚠ **必须把 `_last_ts` 也清掉**（2026-10-09 自检当场逮到 ✗）：SDK 的 `reset()` 会把
          `_previous_time` 清空（`session.py:68` ✓）⇒ **新会话允许重新从任意时间开始** ✓；
          而本模块那道"重复/倒退就顶一下"的兜底若留着旧值 ⇒ 新会话的时间会被顶到上一轮的
          尾巴上 ✗ ⇒ SDK 内部 `_roi_time` 跟着偏 ⇒ 「锁定 15s 后才扫成功」那道闸悄悄推迟，
          表现是"成功面板明明在、却一直 WAITING" ✗（自检第 ③ 条就是这么红的 ✓）。
        """
        if self._sess is not None:
            self._sess.reset(roi=roi)
            self.shapec = None
            self._last_ts = None
            self.last_phase = "-"
            self.last_aim_screen = None
            self.last_roi = None

    def finish(self, reason="host_finished"):
        if self._sess is not None:
            try:
                self._sess.finish(reason)
            except Exception:                 # noqa: BLE001 —— 收尾失败无所谓 ✓
                pass

    def close(self):
        if self._sess is not None:
            try:
                self._sess.close()
            except Exception:                 # noqa: BLE001
                pass
            self._sess = None
        self.ready = False

    def feed(self, frame_bgr, timestamp_s, screen_origin=(0, 0)):
        """喂一帧 ⇒ `Observation`（没就绪 / 尺寸变了就返回 None，调用方按 None 跳过 ✓）。"""
        if not self.ready or self._sess is None:
            return None
        ts = float(timestamp_s)
        if not (ts == ts) or ts in (float("inf"), float("-inf")):    # NaN/Inf ⇒ 挡住 ✓
            return None
        key = (getattr(frame_bgr, "shape", None),
               (round(float(screen_origin[0]), 3), round(float(screen_origin[1]), 3)))
        if self.shapec is not None and key != self.shapec:
            # SDK 自己会 `finish('capture_geometry_changed')`（`session.py:97-98` ✓）
            # ⇒ 对齐着重开一轮，别让整个会话悄悄停摆 ✗
            self.reset()
        self.shapec = key
        # SDK 要求**严格递增**（`session.py:85-86` 会抛 ValueError ✗）⇒ 重复/倒退就顶一下 ✓
        last = getattr(self, "_last_ts", None)
        if last is not None and ts <= last:
            ts = last + 1e-3
        self._last_ts = ts
        obs = self._sess.process(frame_bgr, ts, screen_origin=screen_origin)
        self.frames += 1
        self.last_phase = str(getattr(obs, "phase", "-"))
        self.last_aim_screen = getattr(obs, "aim_screen", None)
        self.last_roi = getattr(obs, "roi", None)
        return obs

    # ---------------- 「成功后点确定」用的确认按钮定位 ----------------

    def _templates(self):
        """懒加载 `cg.png`（成功面板）/ `qr.png`（确认按钮），并**量出**按钮在面板里的比例。

        ⚠ 比例是**量出来**的，不是写死的（模板换了自动跟上 ✓）：实测 `qr.png` 在 `cg.png`
          里 score=**1.000**（就是面板右下那个按钮 ✓）⇒ 位置 (0.9201, 0.9232)。
        """
        if self._cg is not None and self._btn_ratio is not None:
            return True
        try:
            import cv2
            import numpy as np
            root = self.sdk_dir / "liescc"
            self._cg = cv2.imread(str(root / "cg.png"), cv2.IMREAD_GRAYSCALE)
            self._qr = cv2.imread(str(root / "qr.png"), cv2.IMREAD_GRAYSCALE)
            if self._cg is None or self._qr is None:
                return False
            _mn, _mx, _ml, _loc = cv2.minMaxLoc(
                cv2.matchTemplate(self._cg, self._qr, cv2.TM_CCOEFF_NORMED))
            if _mx < 0.9:                    # 量不准就不猜（返回 None，调用方留痕 ✓）
                return False
            self._btn_ratio = ((_loc[0] + self._qr.shape[1] / 2.0) / self._cg.shape[1],
                               (_loc[1] + self._qr.shape[0] / 2.0) / self._cg.shape[0])
            del np
            return True
        except Exception:                    # noqa: BLE001 —— 缺 cv2/模板 ⇒ 当"找不到按钮" ✓
            return False

    def find_confirm_point(self, frame_bgr, roi=None, thr=CONFIRM_MATCH_THR):
        """在成功面板**外框**里找确认按钮 ⇒ 返回**帧内像素** (x, y)；找不到返回 None。

        口径与 SDK 自己那套**同一处**（`vision_sdk/success.py:23-27` ✓）：先由内容区 ROI
        反推整窗外框（`liescc/locate.py` 的 `HOLE_R*` ✓），再在那个框 ±12% 的范围内匹配。
        ⚠ 匹配尺度表用 **SDK 自己的那份**（`vision_sdk.success.SCALES` ✓）——
          自己另写一套就会"SDK 判成功、我找不到按钮" ✗（2026-10-09 实测：把面板放大到
          4.5 倍贴上去，超尺度表 ⇒ 直接误报到 0.804 的假位置 ✓ 这条就是这么踩出来的）。
        """
        if not self._templates():
            return None
        if not isinstance(roi, (tuple, list)) or len(roi) != 4:
            return None
        try:
            import cv2
            from liescc import locate
            from vision_sdk.success import SCALES
        except Exception:                    # noqa: BLE001
            return None
        try:
            x, y, w, h = (float(v) for v in roi)
            ow, oh = w / locate.HOLE_RW, h / locate.HOLE_RH
            ox, oy = x - ow * locate.HOLE_RX, y - oh * locate.HOLE_RY
            x0 = max(0, int(ox - ow * 0.12))
            y0 = max(0, int(oy - oh * 0.12))
            x1 = min(frame_bgr.shape[1], int(ox + ow * 1.12))
            y1 = min(frame_bgr.shape[0], int(oy + oh * 1.12))
            if x1 - x0 < 32 or y1 - y0 < 32:
                return None
            gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            best = (-1.0, 0, 0, 0, 0)        # score, x, y, w, h
            for sc in SCALES:
                tw, th = int(self._cg.shape[1] * sc), int(self._cg.shape[0] * sc)
                if tw < 16 or th < 12 or tw > gray.shape[1] or th > gray.shape[0]:
                    continue
                templ = cv2.resize(self._cg, (tw, th), interpolation=cv2.INTER_AREA)
                _mn, mx, _ml, loc = cv2.minMaxLoc(
                    cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED))
                if mx > best[0]:
                    best = (mx, loc[0], loc[1], tw, th)
            if best[0] < float(thr):
                return None                  # 不够像 ⇒ 宁可不点（点错比不点糟 ✗）
            px = x0 + best[1] + self._btn_ratio[0] * best[3]
            py = y0 + best[2] + self._btn_ratio[1] * best[4]
            return (float(px), float(py))
        except Exception:                    # noqa: BLE001 —— 定位失败不能拖垮实时回路 ✓
            return None
