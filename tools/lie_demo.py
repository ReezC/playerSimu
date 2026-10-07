# -*- coding: utf-8 -*-
"""测谎**效果演示窗**（2026-09-30 用户要求 ✓ 原话：「做一个窗口程序，可以选择视频文件
播放，播放的时候实时显示效果"；前身是"红点代表鼠标"的定视频演示 ✓）。

窗口能做什么：
  · **选择视频文件**（工具栏"选择视频…"✓ 默认开在 `datasets/liedetectorVideo`）；
  · **播放 / 暂停**（空格也行 ✓）、**速度** 0.25~2×、**循环**；
  · **检测器开关**（关掉 = 纯残差路线 M2a；开着 = 加检测通道 M2b：建档种子 / 校验 / 吸附 ✓
    ⚠ **复活已于 2026-10-02 停用** ✓ 好对比）；
  · 打开后**先预热检测**（进度显示在状态栏 ✓ 别让人对着黑屏等），然后自动开播 ✓；
  · 画面上：🔴 **大红点（带白圈）= 鼠标**（控制器输出的光标估计 ✓）、🟢 **绿圈 = 我们认为的
    真目标位置**（圆心 = 后端报告位置 ✓）、🟢 **实心绿点 = 追踪器真实位置**（未平滑 ✓）、
    🔴 **小红点 = 绿圈圆心**（只在绿点不在圆心时画 ⇒ 两种可能：**拟人平滑正在生效** ✓ 用户
    2026-09-30 认知 #3/#4 ✓，或**信念修正正在生效**（"并集吞下在册假目标 ⇒ 圆修正到对角" ✓
    用户 2026-10-01 ✓ 见 `LieTracker._merge_far_corner` ✓））、
    **粗红框 = 我们认为真目标被包含在里面**（后端 `tbox` ✓，绿圈**整圈**在它
    里面 ✓）、细蓝框 = 检测器看到的东西、白线 = 还差多少 ✓；命中 ⇒ 鼠标点转绿 ✓；
    ⭐ **左键点选检出框**（用户 2026-10-03 ✓）：黄框高亮 + **两行标签**（第 1 行 = `(cx,cy)
    WxH=面积` 例如 `(360.4,323.0) 179x208=37232` ✓；第 2 行 = 三个**重叠量** `IoU最大砖 … |
    圆矩IoU … | 圆矩∩框 …` ✓ 用户第二次点名"都放在第二行" ✓ 见 `pick_extra` ✓），状态栏同步
    写一行；点空白取消 ✓；⭐ **右键**（用户 2026-10-03 ✓）= 选中 + 弹菜单 ⇒"**复制检出框
    信息**"（剪贴板文本 ✓ 见 `pick_clip_text` ✓ 用来"复制之后与人交流" ✓），点空白不开菜单 ✓
    （纯显示 ✓ 不动任何判定 ✓ 数字口径 = **加工域** ✓ 与红框/登记同一把尺 ✓）；
  · 状态栏：`帧 i/N ｜ 状态 ｜ 命中率 x%（h/n）` 实时刷新 ✓。

管线（与实机同构 ✓）：帧 →（缩到 500 高、素材域 ✓）→ 检测**常驻子进程**给框 →
`LieTracker.process(dets=…)` → `LieMouseController.step(cursor=模拟光标)` → 光标按指令
+ 一拍延迟移动（固件鼠标也是这样动 ✓）。

用法：
    python -m tools.lie_demo                                 # 开窗（自动载入默认视频 ✓）
    python -m tools.lie_demo --video <路径>                   # 指定视频开窗
    python -m tools.lie_demo --headless --record out.mp4      # 无窗录制（自验标准姿势 ✓）
    python -m tools.lie_demo --headless --no-dets             # 纯残差对比 ✓
"""
import argparse
import gc
import json
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.lie_controller import LieMouseController  # noqa: E402
#: ⭐⭐⭐ **鼠标跟随效率倍率**默认值（用户 2026-10-03 ✓ 原话："能否开放一个系数配置，调追踪器
#: 跟上圆心的效率倍率？" + "**我不想影响位置计算**，只调追踪器（较大的白描边圈绿圆）" ✓
#: 与控制器**同一处口径** ✓）：控制器每拍把"光标 → 目标点"的误差消掉这么多倍 ✓（1.0 = 全量 ✓）。
from perception.lie_controller import _FOLLOW_GAIN as _FOLLOW_GAIN_DEFAULT  # noqa: E402
#: ⭐⭐⭐ **新模式"运动分离"**（用户 2026-10-03 ✓ 见 `perception/lie_motion.py` 的模块头 ✓）：
#:   它和 `Runner` **同签名**（`step` 回同样的 6 元组 ✓）⇒ 演示窗只换构造 ✓ 别的零改动 ✓。
from perception.lie_motion import MotionRunner                          # noqa: E402
#: ⭐⭐⭐ **速度跟踪档的壳**（用户 2026-10-07 ✓「**我们要重头开始**」✓ 见 `VelocityRenderer` 那一族 ✓）
#:   —— ⚠ 它与 `MotionRunner` **同签名同返回** ✓（底盘换成 `VelocityTracker` ✓）⇒ 演示窗**只换构造** ✓（`warm_tick` / `draw` /
#:   拖动 / 播放**一行都不用改** ✓）。
from perception.lie_motion import VelocityRunner                        # noqa: E402
#: ⭐⭐⭐ **速度跟踪那三个参量的默认值**（用户 2026-10-07 ✓ 他问的那三个 ✓）——
#:   ⚠ 口径**只在 `perception/lie_motion.py` 一处** ✓（界面别抄数 ✗ 会漂 ✓）；
#:   ⚠⚠ **判据本轮还没接** ⇒ 这三个值**先只落盘** ✓（见 `_rowV` 的 tooltip ✓）。
from perception.lie_motion import (VEL_CATCH_DEF as _VEL_CATCH_DEFAULT,   # noqa: E402
                                   VEL_DEV_DEF as _VEL_DEV_DEFAULT,
                                   VEL_HOLD_DEF as _VEL_HOLD_DEFAULT)
# ⭐⭐⭐ **速度跟踪（velocity）那三档状态码 + 那两个"等待上板"有关的常量**（用户 2026-10-07 ✓）——
#   ⚠ 一律从 `perception/lie_motion.py` **同一处口径取** ✗ 别在界面里抄字符串 ✗（"merge" 之类
#     写错一个字 ⇒ 画面上就不洋红 ✓ 而且不留痕 ✓）；本文件只会用到这几个 ✓。
from perception.lie_motion import (VEL_WAIT as _VEL_WAIT,            # noqa: E402
                                   VEL_MERGE as _VEL_MERGE)
# ⭐⭐⭐ **运动分离那几个参数的默认值**（用户 2026-10-03 ✓ 原话："运动分离 **没有任何参数要配吗？**"
#   ✓）—— 从 `perception/lie_motion.py` **同一处口径**取 ✓（界面别抄一份数 ✗ 会漂 ✓）。
from perception.lie_motion import _DIR_N as _MOTION_DIRN_DEFAULT          # noqa: E402
from perception.lie_motion import _KF_PRED_MAX as _MOTION_PREDMAX_DEFAULT  # noqa: E402
from perception.lie_motion import _MIN_HITS as _MOTION_MINHITS_DEFAULT   # noqa: E402
from perception.lie_motion import _PAIR_GATE as _MOTION_PAIRED_DEFAULT   # noqa: E402
from perception.lie_motion import _Q_MIN as _MOTION_QMIN_DEFAULT         # noqa: E402
from perception.lie_motion import _Q_NEED as _MOTION_QNEED_DEFAULT       # noqa: E402
from perception.lie_motion import _SCORE_DECAY as _MOTION_SDECAY_DEFAULT  # noqa: E402
from perception.lie_motion import _SMOOTH as _MOTION_SMOOTH_DEFAULT      # noqa: E402
from perception.lie_motion import _STUCK_RATIO as _MOTION_STUCK_DEFAULT  # noqa: E402
from perception.lie_motion import _SWITCH_MARGIN as _MOTION_MARGIN_DEFAULT  # noqa: E402
from perception.lie_motion import _WHITE_W as _MOTION_WHITEW_DEFAULT     # noqa: E402
# ⭐⭐ **框心指导的力度**默认值（用户 2026-10-04 ✓ 原话："我在哪里调控框心指导的力度？" ✓
#    ⇒ 见 `_FUSE_PULL` ✓ —— 界面上就是「框心力度」那一项 ✓）
from perception.lie_motion import _FUSE_PULL as _MOTION_FPULL_DEFAULT  # noqa: E402
#: ⭐⭐⭐⭐⭐ **「夹取增益」的默认值**（用户 2026-10-04 ✓ 原话："**位置被硬夹说明你要么快了要么
#:   慢了，夹取也要算进速度平滑。因为夹取是修正你的预测误差，夹取是较准确的测量值**" ✓✓）
#:   —— 见 `_CLAMP_GAIN` ✓；⚠ **不许界面另抄一份数** ✗（同「框心力度」那条纪律 ✓）。
from perception.lie_motion import _CLAMP_GAIN as _MOTION_CGAIN_DEFAULT  # noqa: E402
from perception.lie_tracker import LieTracker  # noqa: E402
# ⭐ 轨迹预测观察窗 / 融合框判定阈值的**默认值**（界面初值用它 ✓ 与追踪器同一处口径 ✓）
# ⭐⭐ 重叠尺统一在 `perception/geom.py` ✓（用户 2026-10-03 ✓）：点选检出框时要按
#   "**与哪块砖 IoU 最大**"显示 ⇒ 用**同一把尺**算 ✓（口径一处 ✓ 不自己写一份 ✗）。
from perception import geom                                     # noqa: E402
from perception.lie_tracker import _MERGE_IOU as _MERGE_IOU_DEFAULT  # noqa: E402
from perception.lie_tracker import _MERGE_IOU_OUT as _MERGE_IOU_OUT_DEFAULT  # noqa: E402
# ⭐⭐⭐ **参数分档默认值**（用户 2026-10-03 ✓ 第 2/3 行 ✓ 与追踪器同一处口径 ✓）
from perception.lie_tracker import _RING_COV_SEP as _RING_COV_SEP_DEFAULT  # noqa: E402
from perception.lie_tracker import _BRICK_IOU_SEP as _BRICK_IOU_SEP_DEFAULT  # noqa: E402
from perception.lie_tracker import _RING_COV_FUSE as _RING_COV_FUSE_DEFAULT  # noqa: E402
from perception.lie_tracker import _BRICK_IOU_FUSE as _BRICK_IOU_FUSE_DEFAULT  # noqa: E402
#: ⭐⭐⭐ **分离判定阈值**默认值（用户 2026-10-02 ✓ 与追踪器**同一处口径** ✓ 见 `lie_tracker`）：
#: 分离信号第三条的 x —— 融合期框**最大面积 ≥ 最小面积 × 它** ✓。
from perception.lie_tracker import _SEP_RATIO as _SEP_RATIO_DEFAULT  # noqa: E402
#: ⭐⭐⭐ **融合框继承距离限制**默认值（用户 2026-10-02 ✓ 口径修订 ✓ 与追踪器**同一处口径** ✓）：
#: 融合框候选第一条闸 —— `圆心到框最近边的有符号垂距 ÷ 绿圈半径 > 它` ✓（0 = 圆心须在框内 ✓）。
from perception.lie_tracker import _INHERIT_DIST as _INHERIT_DIST_DEFAULT  # noqa: E402
#: ⭐⭐⭐ **融合框挑选允许倒退距离(绿圆半径比例)** 默认值（用户 2026-10-02 ✓ 与追踪器同一处口径 ✓）：
#: 第二条闸的容差 —— 落位位移在白箭头方向上的分量 ≥ `−它 × 绿圈半径` 才算"安全候选" ✓（0 = 一点不许 ✓）。
from perception.lie_tracker import _ALLOW_BACK_RATIO as _ALLOW_BACK_RATIO_DEFAULT  # noqa: E402
#: ⭐⭐⭐ **砖最多拥有融合框边数量**默认值（用户 2026-10-02 ✓ 与追踪器**同一处口径** ✓）：
#: 融合框四边的"边归属"里归砖的边**只留距离最近的这么条**（4 = 不截断 ✓）。
from perception.lie_tracker import _EDGE_MAX as _EDGE_MAX_DEFAULT  # noqa: E402
#: ⭐⭐⭐ **真目标预测最大速度倍率**默认值（用户 2026-10-03 ✓ 与追踪器**同一处口径** ✓）：
#: 分离期（红框丢失 = 淡粉接力框那段）白箭头模长上限 = `a × 当拍群体速度模长` ✓。
from perception.lie_tracker import _SEP_VEL_MAX as _SEP_VEL_MAX_DEFAULT  # noqa: E402
from perception.lie_tracker import _NOISE_TOL as _NOISE_TOL_DEFAULT      # noqa: E402
#: ⭐⭐⭐ **"缩成砖判定 · 最小 IoU"默认值**（用户 2026-10-03 ✓ 原话："我才学到 IoU 这个算法，
#: 我认为我们很多判定都可以换成这个算法（例如『噪声容差』改成『**噪声最小 IoU**』）" ✓
#: 与追踪器**同一处口径** ✓）：给"框缩回砖大小/位置"补一把**尺度无关**的尺 ✓ ——
#: 与"四条边都在噪声容差(px) 内"**并列 OR** ✓（0 = 关 ✓ 只用像素那条腿 ✓）。
from perception.lie_tracker import _IOU_BRICK as _IOU_BRICK_DEFAULT      # noqa: E402
from perception.lie_tracker import _PATH_MS as _PATH_MS_DEFAULT  # noqa: E402
from perception.lie_registry import _BOARD_S_DEF as _BOARD_S_DEFAULT     # noqa: E402
#: ⭐⭐ **上板防抖重叠率**默认值（用户 2026-10-02 ✓ 与登记表**同一处口径** ✓ 见 `lie_registry`）
from perception.lie_registry import _BOARD_OVERLAP_DEF as _BOARD_OVERLAP_DEFAULT  # noqa: E402

VIDEO_DIR = ROOT / "datasets" / "liedetectorVideo"
GIF_DIR = ROOT / "datasets" / "liedetectorgifs"   # GIF 素材默认根（用户 2026-09-30 ✓）


def _latest_video():
    """默认载入**目录里最新的那段录像**（别写死文件名 ✗ —— 用户会重录/改名 ✓
    实测：`9月30日.mp4` 被换成 `9月30日(1).mp4` 后写死的路径直接读不到 ✗）。"""
    if VIDEO_DIR.is_dir():
        vs = sorted(VIDEO_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime,
                    reverse=True)
        if vs:
            return vs[0]
    return VIDEO_DIR / "demo.mp4"


DEF_VIDEO = _latest_video()
PROC_H = 500                       # 处理域高度（素材/模型是按 750×500 调的 ✓ 宽度按比例）
MIN_PROC_W = 320
VIDEO_EXT = "视频 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv)"
GIF_EXT = "GIF 动图 (*.gif)"

# ⚠ 原来这里还有"图片集帧名时间戳"的正则（`_PHOTO_TS` / `_IMG_EXT` ✓）—— 用户
#   2026-09-30 要求**移除图片集播放方式、改成"选取 gif"** ✗ ⇒ 一并删掉（不留死代码 ✗）。


class DetsWorker:
    """常驻检测子进程（JPEG 进 / JSON 出 ✓ 见 tools/lie_dets_worker.py）。

    ⚠ 为什么不是进程内推理：**"先 Qt 后 torch"的进程里 c10.dll 必炸** ✗
    （WinError 1114 ✓ 既有结论，工作台/YOLO 提案同款处置 ✓）。
    """

    def __init__(self, weights=None, conf=0.25, imgsz=960, device=None):
        #   ⚠⚠ **2026-10-07「甲」**：worker 默认 **auto = 有 CUDA 就上 GPU** ✓（见
        #     `lie_dets_worker` 那段 ✓ —— 实测原来跑在 CPU 上 ⇒ 每帧 28.6ms ✗）；
        #     这里留一个口子只为 **A/B 与自检**（`device="cpu"` ✓）—— 生产**不要传** ✓。
        cmd = [sys.executable, "-m", "tools.lie_dets_worker",
               "--conf", str(conf), "--imgsz", str(imgsz)]
        if device:
            cmd += ["--device", str(device)]
        if weights:
            cmd += ["--weights", str(weights)]
        self.p = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)

    def detect(self, bgr):
        ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return []
        self.p.stdin.write(struct.pack(">I", len(jpg)) + jpg.tobytes())
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        try:
            return json.loads(line) if line else []
        except Exception:                     # noqa: BLE001
            return []

    @staticmethod
    def names(weights=None):
        """问一次模型的**类别名**（另起一个短命子进程 ✓ 打印 JSON 后退出 ✓）。"""
        cmd = [sys.executable, "-m", "tools.lie_dets_worker", "--names"]
        if weights:
            cmd += ["--weights", str(weights)]
        try:
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
            line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
            return json.loads(line) if line else []
        except Exception:                     # noqa: BLE001 —— 拿不到就退回显示编号 ✓
            return []

    def close(self):
        try:
            self.p.stdin.write(struct.pack(">I", 0))
            self.p.stdin.flush()
        except Exception:                     # noqa: BLE001
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:                     # noqa: BLE001
            self.p.kill()


#: ⭐⭐⭐⭐⭐ **"原图"存成 JPEG 字节**（用户 2026-10-07 ✓ 原话："**优化直到能顺畅跑
#:   D:\Media\record\mxd\10月7日.mp4**" ✓✓）—— ⚠⚠ **实测**（那条片子 ✓）：**1920×1080 ｜ 30fps ｜
#:   991 帧 ｜ 33 秒** ⇒ 去重后约 **667 帧**；按"原图留 BGR"的口径是 **7.55MB/帧 ⇒ 5.0GB** ✗✗
#:   （再打开一次翻倍 ⇒ **10GB** ✗ = 他说的"卡到没法用" ✓）。
#:   ⇒ 原图**只留 JPEG**（1080p q95 ≈ **0.35MB/帧** ✓ ⇒ 每帧 7.55 → **1.68MB** ✓，**4.5×** ✓），
#:     显示那一拍**解码一次** ✓（~5~8ms ✓ 有缓存 ✓ 见 `DemoWindow._big` ✓）。
#:   ⚠ **只给显示用** ✗✗：**检测 / 追踪吃的是处理帧 `[1]`**（750×500 那档 ✓ 见 `warm_tick` ✓）⇒
#:     算法**一个像素都不碰** ✓（"所见即所得"那条不受影响 ✓）。
#:   ⚠ 质量取 **95** ✗（不是检测传输那个 92 ✓）：这是**给人看的** ✓，图上还要画浮层 ✓。
_BIG_JPEG_Q = 95


def decode_big(blob):
    """把"原图"那一份（JPEG 字节 ✓ 见 `_BIG_JPEG_Q`）解回 BGR ✓；已经解好的数组**原样返回** ✓。"""
    if blob is None:
        return None
    if isinstance(blob, np.ndarray):          # ⚠ 兼容老口径（已经是数组 ✓）
        return blob
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


def _enc_big(_f):
    """BGR ⇒ JPEG 字节（⚠ 编码失败就**退回原数组** ✓ 不许因此读不进素材 ✗）。"""
    try:
        _ok, _buf = cv2.imencode(".jpg", _f, [int(cv2.IMWRITE_JPEG_QUALITY), _BIG_JPEG_Q])
        return _buf.tobytes() if _ok else _f
    except Exception:                         # noqa: BLE001
        return _f


#: ⭐⭐⭐⭐⭐ **演示窗"内部异常"的落盘文件**（用户 2026-10-07 ✓ 他报"**闪退**" ✓✓）。
#:   ⚠⚠⚠ **为什么非有不可** ✗✗：Qt 槽里抛出的**未捕获 Python 异常** ⇒ PyQt 直接 `qFatal`/`abort`
#:     ⇒ **窗口整个消失** ✓（退出码 `0xC0000409` ✓），而用户跑的是 `pythonw`（**没有控制台** ✗）
#:     ⇒ **traceback 一个字都看不到** ✗✗ ⇒ 只能靠猜 ✓ —— 这次这个 bug（`motion_viz` 一句日志
#:     拿 `None` 当 `%.1f` ✓、崩在他片子的第 **~430** 帧 ✓）我先后排除了内存 / 线程 / 检测 /
#:     QTimer 才挖到 ✓。⇒ 现在**任何定时器槽异常都先落盘** ✓（见 `DemoWindow._safe` ✓），
#:     而且**不让它把窗口带走** ✓（下面 `_ERROR_LOG` 就是给下次"一眼看见"用的 ✓）。
_ERROR_LOG = ROOT / "logs" / "lie_demo_error.log"


def log_exception(where, exc=None):
    """异常（含 traceback）追加到 `logs/lie_demo_error.log` ✓ —— ⚠ **绝不再抛** ✗（记日志不许炸 ✓）。"""
    try:
        _ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_ERROR_LOG, "a", encoding="utf-8") as _f:
            _f.write("\n[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), where))
            traceback.print_exc(file=_f)
            if exc is not None:
                _f.write("  %r\n" % (exc,))
    except Exception:                             # noqa: BLE001 —— 记不下来也只能算了 ✓
        pass


def play_delay_ms(ts, i, i0, t0, now, speed):
    """播放调度（**纯函数** ✓ 好测 ✓）：**下一拍该等多少毫秒**（≤ 0 也行 ⇒ 立刻下一拍 ✓）。

    ⚠⚠⚠ **2026-10-07：为什么非有不可** ✗✗（用户原话："**没有以原速播放视频**" ✓✓）——
      老写法是"**这一拍先干活、干完再 `timer.start(这一帧的真实间隔)`**" ✗✗ ⇒ 每拍的**实际周期**
      = **工作量 ＋ 间隔** ✗（**离屏实测** ✓ `10月7日.mp4`：间隔 41.4ms ＋ 工作量 ~23ms
      ⇒ **64ms/拍** ✗）⇒ 797 帧播了 **50.9s**（该 **33.0s** ✓）＝ **0.65 倍速** ✗✗
      —— **用户看到的"没有原速"就是它** ✓（而且方向是**比原速慢** ✗，不是快 ✓）。
      ⚠ 顺手量到的另一件：他那台的速度下拉停在 **2.00×** ✗（会存进配置 ✓ 下次照旧 ✓）
      ⇒ 间隔被算成 20.7ms ✗ < 工作量 ⇒ 更追不上 ✓（**2× 需要"每拍 ≤ 20ms"才行** ✗）。
    ⇒ 现在按**墙钟**排 ✓：从播放起点出发，第 `i` 帧**本该在**
      `t0 + (ts[i] − ts[i0]) / speed` 这一刻出现 ✓ ⇒ **差额**才是该等的 ✓
      （`≤ 0` ⇒ **立刻**排下一拍 ✓ —— 追不上就**诚实落后** ✗，但**绝不"额外多等一个间隔"** ✗）。
    ⚠ `ts` 空 / 太短（自检里那种裸窗口 ✓）⇒ 回**固定 `间隔`**（老行为 ✓ 不崩 ✓）。
    """
    try:
        _n = len(ts)
        _last = float(ts[-1]) if _n else 0.0
        _v = max(0.05, float(speed))
        _i = max(0, min(int(i), _n - 1))
        _j = max(0, min(int(i0), _n - 1))
        if _n >= 2 and _last > 0.0:
            #   ⚠⚠⚠ **到期时刻要取"下一帧"那一格** ✗✗（**我第一版取成了"刚显示的那一帧"** ✓
            #     **当场把这条量出来** ✓）：取成"刚显示的那帧" ⇒ 每一拍算出来都是"早就该到" ✗
            #     ⇒ 定时器**永远排 0** ⇒ 播放退化成"**按工作量跑**" ✗✗ ⇒ **慢放（0.5×）完全不生效** ✗、
            #     而且 1× 那次量到的 0.98× 只是"工作量恰好 ≈ 间隔"的**巧合** ✓✓（差点被它糊过去 ✗）。
            _nxt = min(_i + 1, _n - 1)
            _due = float(t0) + (float(ts[_nxt]) - float(ts[_j])) / _v
            if _i + 1 >= _n:                     # 已经到末帧 ⇒ 再等一格的间隔就收尾 ✓
                _due += (float(ts[-1]) - float(ts[-2])) / _v
            return (_due - float(now)) * 1000.0
    except Exception:                            # noqa: BLE001 —— 算不出来就退老路 ✓
        pass
    return None                                  # ⇒ 调用方回落到 `间隔 = 1/fps ÷ speed` ✓


def load_video(path, on_progress=None):
    """读整段视频 ⇒ `([(原图, 处理帧)], ts, (proc_w, proc_h), (sx, sy), fps)`。

    原图给显示 ✓；处理帧缩到**高 500**（素材域 ✓ 宽按比例，3:2 输入正好 750×500 ✓）。
    ⭐ `ts` = **每帧时间戳（秒，从 0 起）**（视频是等间隔 ⇒ 就按帧率算 ✓）——
    GIF 那条路（`load_gif`）用的是**GIF 自带的每帧延时**（不等间隔 ✓）⇒ 两条路
    同形返回，界面/追踪器都按 `ts[i]` 走 ✓（别再用 `i / fps` 硬算 ✗ 那样 GIF 的
    真实节奏就丢了 ✗）。

    ⭐⭐ **重复帧去重 + 按内容帧率播放**（2026-09-30 用户："箭头消失了，一个都看不到"）：
    现场录像标称 **60fps**，但游戏实际只 ~**10fps** 刷新 ⇒ 录像里 **6 帧有 5 帧像素级
    全同**（相邻帧均值差 0.00 ✗ 实测；每 6 帧才真变一次 ~19px ✓）。后果：逐框位移在
    重复帧上**恒为 0** ⇒ 箭头一根都不显示 ✗✗（我先前还误判成"框是静止的" ✗）。
    ⇒ 保留**内容变化**的帧、丢掉重复帧，`fps` 按保留比例折算（60 × 70/420 = **10** ✓）
    ⇒ 每个显示帧之间都有**真实位移** ✓ 且**墙钟时长不变**（70 帧 @10fps = 7s ✓ = 420
    帧 @60fps ✓）—— 所有下游（ts = i/fps、定时器、录像 writer、进度条）自动跟着对 ✓。
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not cap.isOpened():
        return [], [], (0, 0), (1.0, 1.0), fps
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    pw = max(MIN_PROC_W, int(round(w * PROC_H / h))) if h else 750
    ph = PROC_H
    frames = []
    n_all = 0
    prev_g = None
    _f0 = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n_all += 1
        small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_g is not None:
            # 与上一**保留**帧几乎逐像素相同 ⇒ 重复帧（录像补帧 ✗）⇒ 丢掉 ✓
            if float(cv2.absdiff(g, prev_g).mean()) < _DUP_EPS:
                continue
        #   ⭐ **原图只留 JPEG 字节**（见 `_BIG_JPEG_Q` ✓）：BGR 那份**这一拍用完就丢** ✗
        #     ⇒ 读入过程也不占内存 ✓（否则**读的时候**就要 5GB ✗）。
        if _f0 is None:
            _f0 = (int(f.shape[1]), int(f.shape[0]))       # ⚠ 缩放比用它算 ✓（别去解 JPEG ✓）
        frames.append((_enc_big(f), small))
        prev_g = g
        #   ⭐⭐⭐ **报进度**（用户 2026-10-07 ✓ "**优化直到能顺畅跑**" ✓）：⚠⚠ 长片读入要
        #     **十几秒** ✗（那条片子实测 **13.1s / 991 帧** ✓）⇒ 不报进度就是"点开就冻住" ✓
        #     ⇒ 调用方（演示窗 ✓）能据此**刷 label ＋ 让事件循环喘气** ✓（见 `load` 那段 ✓）。
        if on_progress is not None and (n_all % 20 == 0):
            on_progress(len(frames), n_all)
    cap.release()
    if not frames:
        return [], [], (pw, ph), (1.0, 1.0), fps
    fps_eff = fps * len(frames) / float(max(1, n_all))     # 折算成**内容帧率** ✓
    sx = (_f0[0] if _f0 else pw) / float(pw)
    sy = (_f0[1] if _f0 else ph) / float(ph)
    ts = [i / max(1.0, fps_eff) for i in range(len(frames))]   # 等间隔 ⇒ 就按帧率算 ✓
    return frames, ts, (pw, ph), (sx, sy), max(1.0, fps_eff)


def load_gif(path, on_progress=None):
    """读 **GIF 动图** ⇒ 与 `load_video` **同形**（`(frames, ts, proc, scale, fps)` ✓）。

    用户 2026-09-30：**把"选图片集"换成"选取 gif"** ✓（素材放
    `datasets/liedetectorgifs/*.gif` ✓ 实测 `test.gif`：**750×500、70 帧、每帧 180ms**
    ⇒ 5.6fps ✓ —— 就是去重后那版素材导成的动图 ✓）。

    ⭐ **时间戳 = GIF 自带的每帧延时**（`im.info["duration"]` **毫秒** ✓ 累加成 ts ✓）——
    这是 GIF 里唯一的真实时间信息 ✓（正好对上"轨迹追踪最小精度 = 一帧"那条口径 ✓）；
    首帧归一到 0 ✓。
    ⚠⚠ **OpenCV 读不了 GIF**（`VideoCapture` 不支持 ⇒ 直接返回空 ✗）⇒ 必须走 **PIL**
      （随 ultralytics 已经装着 ✓ 不用新依赖 ✓）。
    ⚠ 重复帧（逐像素几乎相同 ✓）照样按 `_DUP_EPS` 丢掉，但**把它的延时累进时间线**
      （`t_acc += dur` ✓）—— 否则整段会**变快** ✗（丢掉的那帧的时间不该消失 ✓）。
    """
    from PIL import Image
    im = Image.open(str(path))
    n = int(getattr(im, "n_frames", 1) or 1)
    pw = ph = 0
    frames, ts, prev_g = [], [], None
    _f0 = None
    t_acc = 0.0
    for i in range(n):
        im.seek(i)
        dur = max(1.0, float(im.info.get("duration") or 100)) / 1000.0
        f = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)  # P → RGB ✓
        if not ph:
            ph = PROC_H
            pw = max(MIN_PROC_W, int(round(f.shape[1] * PROC_H / f.shape[0])))
        small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_g is not None and float(cv2.absdiff(g, prev_g).mean()) < _DUP_EPS:
            t_acc += dur                  # 重复帧：丢掉画面，但**延时累进时间线** ✓
            continue
        if _f0 is None:
            _f0 = (int(f.shape[1]), int(f.shape[0]))
        frames.append((_enc_big(f), small))            # ⭐ 原图只留 JPEG ✓（见 `_BIG_JPEG_Q` ✓）
        ts.append(t_acc)
        t_acc += dur
        prev_g = g
        if on_progress is not None and (i % 20 == 0):  # ⭐ 报进度 ✓（同 `load_video` ✓）
            on_progress(len(frames), i + 1)
    if not frames:
        return [], [], (0, 0), (1.0, 1.0), 10.0
    span = ts[-1] if ts else 0.0
    fps = (len(ts) - 1) / span if (len(ts) > 1 and span > 1e-3) else 10.0
    sx = (_f0[0] if _f0 else pw) / float(pw)
    sy = (_f0[1] if _f0 else ph) / float(ph)
    return frames, ts, (pw, ph), (sx, sy), max(0.5, fps)


def load_source(path, on_progress=None):
    """统一入口：**`.gif` ⇒ GIF 动图** ✓（用户 2026-09-30：「改为『选取 gif』" ✓）；
    其余 ⇒ **视频文件** ✓。同形返回 `(frames, ts, (proc_w, proc_h), (sx, sy), fps)` ✓。

    ⚠ 原"选图片集（帧目录）"那条**已按用户要求移除** ✗ —— 目录不再是合法输入 ✓
      （传目录会被当视频读 ⇒ 读不到 ⇒ 界面提示"读不到视频" ✓ 不会静默 ✗）。
    ⭐ `on_progress(kept, seen)` ✓（可选 ✓）：长片读入要**十几秒** ✗（那条片子实测 13.1s ✓）
      ⇒ 演示窗拿它**刷 label ＋ 让事件循环喘气** ✓（见 `load` 那段 ✓）。
    """
    p = Path(str(path))
    if p.suffix.lower() == ".gif":
        return load_gif(p, on_progress=on_progress)
    return load_video(p, on_progress=on_progress)


_DUP_EPS = 2.0             # 相邻帧像素均值差 < 它 ⇒ 判为**重复帧**（录像补帧 ✓ 丢掉 ✓）
                           #   ⚠ 原来 0.5 ⇒ 现场有**一批帧差 0.52** 的"看着一模一样"的帧
                           #     被**漏放进来** ✗（用户 2026-09-30："为什么12、13帧长的一样，
                           #     导致速度向量全丢了？" ✓）—— 那两帧里逐框位移中位 **0.09px**
                           #     ⇒ 箭头全变点 ✗。真运动帧的帧差是 **40+**（实测 ✓）⇒ 阈值放到
                           #     2.0 既能把 0.52 那批丢掉、又碰不到真运动 ✓（余量 20 倍 ✓）
_BOX_MATCH_GATE = 40.0     # 逐框配对门（处理域 px ✓）。⚠ 标定依据（实测两轮）：
                           #   · 视频 6 帧里 5 帧重复 ⇒ **内容帧间**真位移 ~**19px**
                           #     （相位相关：帧 6 dx=-8.1 dy=17.2 ✓ 帧 12 dx=-11.6 dy=15.0 ✓）；
                           #   · 图形间距 ~150px ⇒ 40px 门**不会串到隔壁图形** ✓；
                           #   · 我先按"每帧 ~3px"把门收到 12 ⇒ **把真运动当错配拒掉** ✗✗
                           #     （箭头全消失 ✗ 用户实测）—— 别按标称 60fps 估位移 ✗
_MOVE_MIN_PX = 1.2         # ⭐ 位移小于它 ⇒ **只画点不画箭头**（两侧都没真动 ⇒ 箭头方向
                           #   会是亚像素量化噪声 ✗ 不是"实测方向" ✗）
# ⚠ 原来这里有"箭头基准长度 / 最小 10% / 最大 300% / 绝对 300px"一组常量 —— 用户
#   2026-09-30 第 A2 条**取消长度配置** ✗：箭头长度 = **这一拍的位移**（箭尖 = 预计下一帧
#   的位置 ✓）⇒ 这些常量连同滑块/CLI 参数一并删掉（不留死配置 ✗）。


def _track_velocity(hist, n=3):
    """**追踪器自己的轨迹速度**（用户 2026-09-30 第 A1 条 ✓ 原话："白箭头改成展示当前
    追踪器的**轨迹速度向量**（也就是说从追踪器出发指向运动方向）" ✓）。

    `hist` = 最近若干帧的**报告位置** `[(ts, x, y), …]`；用**整窗端点差 / 用时**算速度
    （⚠ 别用逐帧差分 ✗ 一帧抖动就把它带飞 ✓ K 线级噪声；这条口径和轨迹层一致 ✓）。
    返回 `(vx, vy)` px/s（样本不够 ⇒ None ✓）。
    """
    hh = [h for h in hist if h[1] is not None][-n:]
    if len(hh) < 2:
        return None
    (t0, x0, y0), (t1, x1, y1) = hh[0], hh[-1]
    dt = t1 - t0
    if dt <= 1e-6:
        return None
    return ((x1 - x0) / dt, (y1 - y0) / dt)


_RED_MATCH_GATE = 25.0    # ⭐ "追踪器认定的目标框"↔"本帧检测框"的配对门（px ✓ 它就是
                          #   从那批框里选的 ⇒ 基本重合 ✓ 25px 足够容下检测抖动 ✓）
_RAD_LO, _RAD_HI = 0.85, 1.25   # ⭐ **学习"目标实际半径"时只信这个面积带**的框（× 标准
                                #   面积 ✓）：≈1.0× = 单个目标 ✓；跑出去（融合/分离 ✗）
                                #   ⇒ **冻结**不学 ⇒ 透明阶段圆半径恒定 ✓（用户口径 ✓）
_DOT_GAP = 2.0            # ⭐⭐ **绿点（追踪器真实位置 = `raw_pos`）与绿圈圆心（报告位置 = `pos`）
                          #   差多少 px 才算"两者不一致"**（显示域 px ✓ 用户 2026-09-30
                          #   认知 #4 ✓：差超过它就**用红点标出圆心** ✓；重合 ⇒ 只画绿点 ✓）。
                          #   ⚠ 不一致的**来源**：① **拟人平滑**（`_ease_pos` ✓ 老来源 ✓）；
                          #     ② **统一出口的信念夹取**那一支（`raw_pos` 是本拍**夹取前**的快照
                          #     ⇒ 可能滞后一拍 ✓ 见 `LieTracker.process()` 出口 ✓）；
                          #     ~~③ 落位修正~~ —— ⭐⭐ **甲方案之后它不在了** ✓（用户 2026-10-02
                          #     ✓ 原话："**我期望绿圆就是我们认为的真目标所处的位置**" ✓ ——
                          #     落位那一拍**内部一起认**（`LieTracker._adopt_merge_fix` ✓ 且
                          #     `out["raw_pos"]` 同拍同步刷 ✓）⇒ **绿点与圆心重合** ✓、小红点不再
                          #     出现 ✓；老口径"只改报告位置那一层"见 tracker 出口的留档注释 ✓）。
                          #   2px 是显示上的"看得出"下限（1px 会被整数取整的抖动误触发 ✗）。
_RAD_STEP = 2.5           # ⭐ 半径估计的**每帧最大变化**（px ✓ 防单帧噪声跳变 ✗）
_RAD_FREEZE_S = 0.6       # ⭐ 白块连续消失这么久 ⇒ 判定"已透明" ⇒ **永久冻结**实际半径 ✓
_RAD_LEARN_S = 2.0        # ⭐ **最长学习窗口**（从第一次见到白块算起 ✓）：白块阶段就那么长
                          #   （GIF ~1.8s ✓ 素材 ~5s 也够 ✓），而透明阶段亮背景块会偶尔
                          #   "冒充"白块 ✗（实测让它一直不冻结 ⇒ 半径跟着框呼吸 ✗✗）⇒ 到点
                          #   强制冻结 ✓ 用户口径："**透明后**半径恒定不变" ✓
def _choose_target_box(boxes, motion, pos, last_center, typ):
    """**全屏有且只有一个红框**（用户 2026-09-30 第 B1 条 ✓ 原话："全屏有且只能有，且必须
    有 1 个红框，就是当前认为最可能真目标属于的红框" ✓）。

    选法（依次退让，**一定选出 1 个** ✓ 只要画面里有框）：
      ① **报告位置落在哪个框里** ⇒ 就是它 ✓（最可信 ✓）；
      ② 没有落点（在框缝里 / 追踪器丢了）⇒ 取**离目标位置最近**的框 ✓；
      ③ 目标位置也没有（还在建档、没锁定 ✓）⇒ 用**上一拍红框的中心** ✓；
      ④ 连它都没有（刚打开）⇒ 用**画面中心**（测谎弹窗的目标就在中间刷出来 ✓ 实测 ✓）。
    ⭐ 选的时候给"**面积不像单个目标**"的框打折（用户第 B2 条："所有目标的面积（无论真假）
    都是基本相等的" ✓ ⇒ ≈2× 的框是两个图形并成一块 ✗）：距离 ×1.5 的罚分 ✓。
    """
    if not boxes:
        return None
    # ⚠⚠ **UI 端的"融合期锁红框"已整条删除** ✗（2026-09-30 用户："红框必须与后端完全一致" ✓）：
    #   那套锁（面积 ≥1.3× 标准就锁住上一拍红框 ✓）是**第二个真相来源** ✗ —— 它和后端
    #   `_target_box_state`（包含报告位置优先 ✓）会给出不同答案 ⇒ 红框与"后端认为的目标框"
    #   对不上 ✗。**行为已由后端承担** ✓（融合块包含报告位置 ⇒ 后端自然会锁它 ✓ 实测
    #   GIF 19~21 帧后端给的正是那格融合框 ✓）。
    #   ⇒ 本函数退回**纯 UI 兜底**用途（只在"后端没有框"时给个就近的框当参考 ✓，
    #     以及给绿圈半径的兜底 ✓）；**红框本身一律画后端那个** ✓（见 `draw` ✓）。
    tgt = pos if pos is not None else last_center
    if tgt is None:
        tgt = (boxes[0][1], boxes[0][2])          # 兜底（下面一律按"最近"选 ✓）
    best, bs = None, None
    for i, b in enumerate(boxes):
        cx, cy, w, h = b[1], b[2], b[3], b[4]
        # ① 落在框里 ⇒ 直接定（多个命中取面积最接近标准的 ✓）
        inside = (cx - w / 2 <= tgt[0] <= cx + w / 2
                  and cy - h / 2 <= tgt[1] <= cy + h / 2)
        a = w * h
        odd = bool(typ) and not (0.7 * typ <= a <= 1.4 * typ)
        d = ((cx - tgt[0]) ** 2 + (cy - tgt[1]) ** 2) ** 0.5
        score = (0.0 if inside else d + 1e6) + (0.0 if not odd else 5e5)
        if bs is None or score < bs:
            best, bs = i, score
    return best


def _sus_line(motion):
    """图例那行「怀疑链"（用户 2026-09-30 ①②③④ ✓）：目标轨迹 id / 怀疑类型 / 不合群秒数。

    · `sus`：`merge` = 框突然变大（疑似与假目标**重叠** ⇒ 中心不可信、按预测走 ✓）；
      `split` = 框突然变小 + 旁边**新出框**（疑似**分离** ⇒ 新框已标"严查" ✓）；
      `hidden` = 框突然变小但**没检出新的**（真目标"就在面积消失的地方" ⇒ 位置锚住 ✓）。
    · `dev_s` = 这条轨迹"速度不合群"已经持续多少秒；满 `_DEV_HOLD_S`(0.6s) 判据④才敢
      把它当目标 ✓。
    """
    tid = motion.get("tid")
    row = None
    for t in (motion.get("tracks") or []):
        if t.get("tid") == tid:
            row = t
            break
    if row is None:
        return "suspect: (no target track yet)   dev_s=-"
    return ("suspect: target=#%s  sus=%s  dev_s=%.1fs  mustcheck=%d"
            % (tid, row.get("sus") or "-", float(row.get("dev_s") or 0.0),
               int(row.get("must") or 0)))


def _box_moves(prev, cur, motion):
    """逐框位移 + 角色（用户 2026-09-30 口径 ✓）。

    · 位移：本帧每个框 → 上一帧**最近**的框（门内 ✓），位移 = 本帧中心 − 上帧中心；
      配不上（新出现的框 / 跑太快）⇒ `None`（**不画箭头** ✗ 别瞎指方向 ✓）。
    · 角色（只影响颜色 ✓）：
      `"target"` = 追踪器认定的**真目标**那条 ⇒ **红框加粗** ✓（用户第⑤问 ✓）；
      `"suspect"` = 被标 `must_check` 的**严查框**（怀疑真目标在此框内 ✓ 也画红 ✓）；
      `"followed"` = 实际在跟的那个框 ✓。
    """
    moves, roles, preds = [], [], []
    for b in (cur or []):
        cx, cy = float(b[1]), float(b[2])
        mv = None
        if prev:
            bd = None
            for pb in prev:
                d = ((cx - pb[1]) ** 2 + (cy - pb[2]) ** 2) ** 0.5
                if bd is None or d < bd[0]:
                    bd = (d, float(pb[1]), float(pb[2]))
            if bd is not None and bd[0] <= _BOX_MATCH_GATE:
                mv = (cx - bd[1], cy - bd[2])       # 处理域位移 ✓ 画时再放大 ✓
        # ⭐⭐ **上一帧配不上 ⇒ 用"轨迹速度"补一根**（用户 2026-09-30 第②问 ✓ 原话：
        #   "从第1帧到第12帧都确定其为1个目标……它就应该跟着**轨迹预测**显示，且程序后台
        #   需要认为其还存在（只是暂时没被检测出来）" ✓）—— 新出现的框 / 上一帧漏检后重出
        #   的框本来**没有逐框位移**（画不出箭头 ⇒ 看着像"向量消失了" ✗）；轨迹侧有速度
        #   （自己的 ✓ 或**借群体中位**的 ✓ 见 `lie_tracker._borrow_vel` 用户③ ✓）⇒
        #   `速度 × dt` 换算成"这一拍走了多少 px"（与逐框位移**同量纲** ✓）补上 ✓。
        #   ⚠ 只在这一格**没有实测位移**时补 ✗（不然一根框上会出现两根箭头 ✗ 乱 ✓）。
        _pred = False
        if mv is None:
            _dt = float((motion or {}).get("dt") or 0.0)
            for t in (motion.get("tracks") or []):
                if _dt <= 0:
                    break
                if ((cx - t["p"][0]) ** 2 + (cy - t["p"][1]) ** 2) ** 0.5 \
                        <= _BOX_MATCH_GATE:
                    vx, vy = t.get("v") or (0.0, 0.0)
                    if vx or vy:
                        mv = (vx * _dt, vy * _dt)   # 轨迹速度 ⇒ 每拍位移 ✓
                        _pred = True
                    break
        moves.append(mv)
        preds.append(_pred)
        role = ""
        for t in (motion.get("tracks") or []):
            if ((cx - t["p"][0]) ** 2 + (cy - t["p"][1]) ** 2) ** 0.5 <= _BOX_MATCH_GATE:
                # ⚠⚠ **判据认定的那条不再自动画红**（用户 2026-09-30："第24帧：右上的红框
                #   没必要，明显绿圈都被当前的红框包围了，可信度**碾压**" ✓）：判据轨迹可能
                #   是个**远处的干扰**（实测第 24-35 帧判据 #12 一直在右上 (649,37) ✗），
                #   而"报告位置所在的那个框"才是一眼可信的 ✓ ⇒ 红框**只认后者**（见下面兜底 ✓）；
                #   判据那条保留**绿点/绿箭头**（图例已注明"most-different ✓"）就够 ✓。
                if t.get("must") and not role:
                    role = "suspect"                # ⭐ 严查框（怀疑真目标在内 ✓ 红）
                elif t["tid"] == motion.get("followed") and not role:
                    role = "followed"
        roles.append(role)
    # ⭐⭐ **"追踪器所在的框" = 真目标框**（用户 2026-09-30 第①问 ✓ 原话："既然一开始白色
    #   图像所在的框确定为真目标，那么它就应该是红框" ✓）：判据认定的轨迹在建档期还没
    #   建立、或那一拍暂时没配上 ✗ ⇒ 用**报告位置落在哪个框里**兜底 ✓ ——
    #   这条不依赖轨迹，一眼就能对上"十字标现在压在哪个框上" ✓。
    fp = (motion or {}).get("followed_pos")
    if fp is not None:
        for _i, b in enumerate(cur or []):
            x1, y1 = b[1] - b[3] / 2.0, b[2] - b[4] / 2.0
            x2, y2 = b[1] + b[3] / 2.0, b[2] + b[4] / 2.0
            if x1 <= fp[0] <= x2 and y1 <= fp[1] <= y2:
                roles[_i] = "target"
                break
    return moves, roles, preds


class Runner:
    """帧流 → 追踪 → 控制 → 光标（窗口与 headless 共用 ✓ 好测 ✓）。"""

    def __init__(self, dets=None, gain=None, assume=(375.0, 250.0), path_ms=None,
                 merge_iou=None, merge_iou_out=None, sep_ratio=None,
                 ring_cov_sep=None, brick_iou_sep=None, ring_cov_fuse=None,
                 brick_iou_fuse=None,
                 inherit_dist=None, allow_back_ratio=None,
                 edge_max=None, sep_vel_max_ratio=None, noise_tol=None, board_s=None,
                 board_overlap=None, follow_gain=None, iou_brick=None, show_kf=None,
                 kf_pos=None):
        # ⭐ `path_ms` = **轨迹预测窗口时间**（毫秒 ✓ 演示窗那一行配 ✓）⇒ 透传给追踪器 ✓
        self.path_ms = path_ms
        # ⭐⭐⭐ `merge_iou` = **融合框判定 IoU**（用户 2026-10-03 ✓ 原话："把『融合框判定阈值』
        #   换成『**融合框判定 IoU**』：**小于这个 IoU 才行**，之前配的 1.2 改成 **0.5**" ✓）
        #   ⇒ 透传 ✓（判据 = `IoU(检出框, 该砖的登记框) < 它` ⇒ 判红框融合 ✓ 见 `_MERGE_IOU` ✓）。
        self.merge_iou = merge_iou
        # ⭐⭐ `merge_iou_out` = **融合判定 IoU · 退出**（用户 2026-10-03 ✓ 第 1 步"迟滞双阈值" ✓）：
        #   进用 `merge_iou`（严 ✓）、守用 `merge_iou_out`（松 ✓）⇒ 边界不再横跳 ✓ 见 `_merge_iou_thr` ✓。
        self.merge_iou_out = merge_iou_out
        # ⭐⭐⭐ **参数分档（用户 2026-10-03 ✓ 第 2/3 行）**：每档一对 —— "框↔圆外接矩形 IoU >"
        #   与 "框↔内砖 IoU <" ✓ 两条**同时**满足才算融合 ✓（分离期=进入 ✓ 融合期=保持 ✓
        #   融合期不满足 ⇒ **判分离** ✓）；两档之差 = **迟滞** ✓（进严守松 ✓）。
        self.ring_cov_sep = ring_cov_sep
        self.brick_iou_sep = brick_iou_sep
        self.ring_cov_fuse = ring_cov_fuse
        self.brick_iou_fuse = brick_iou_fuse
        # ⭐⭐⭐ `sep_ratio` = **分离判定阈值**（用户 2026-10-02 ✓ 界面上就在"融合框判定阈值"
        #   **右边** ✓）⇒ 透传给追踪器 ✓（分离信号第三条的 x ✓ 见 `_merge_area_span_ok` ✓）。
        self.sep_ratio = sep_ratio
        # ⭐⭐⭐ `inherit_dist` = **融合框继承距离限制**（用户 2026-10-02 ✓ 界面上在"分离判定
        #   阈值"右边 ✓ 口径已从"面积比"改为"距离比" ✓）⇒ 透传给追踪器 ✓
        #   （融合框候选第一条闸 ✓ 见 `LieTracker._inherit_dist_ok` ✓）。
        self.inherit_dist = inherit_dist
        # ⭐⭐⭐ `allow_back_ratio` = **融合框挑选允许倒退距离(绿圆半径比例)**（用户 2026-10-02 ✓
        #   口径从 px 改为"半径比例" ✓）⇒ 透传 ✓（第二条闸的容差 ✓ 见 `_merge_fix_safe` ✓）。
        self.allow_back_ratio = allow_back_ratio
        # ⭐⭐⭐ `edge_max` = **砖最多拥有融合框边数量**（用户 2026-10-02 ✓ 原话："增加参数：
        #   『**砖最多拥有融合框边数量**』(1~4)，注意在多条边选取最近的**前 x 条**" ✓）⇒ 透传 ✓
        #   （融合框四边的"边归属"只留距离最近的这么条 ✓ 见 `LieTracker._edge_ownership` ✓）。
        self.edge_max = edge_max
        # ⭐⭐⭐ `sep_vel_max_ratio` = **真目标预测最大速度倍率**（用户 2026-10-03 ✓ 原话："将分离期
        #   （淡粉色框时期）的**最大相对假目标群速度（白箭头）** = a × 假目标群体的标准速度" ✓）
        #   ⇒ 透传 ✓（分离期白箭头模长上限 ✓ 见 `LieTracker._sep_vel_cap` ✓）。
        self.sep_vel_max_ratio = sep_vel_max_ratio
        # ⭐ `noise_tol` = **噪声容差(px)**（融合规则①：**"圆心 ± 它"的范围与框相交** ✓
        #   用户 2026-10-02 ✓ ⇒ 见 `LieTracker._pos_range_hits` ✓）⇒ 透传 ✓
        self.noise_tol = noise_tol
        # ⭐ `board_s` = **上板时长(秒)**（登记条目要走满这么久才上板 ✓ 用户 2026-10-02 ✓）
        self.board_s = board_s
        # ⭐ `board_overlap` = **上板防抖重叠率**（新检出与旧砖重叠 ≥ 它 ⇒ 判同一块砖、不新建 ✓
        #   用户 2026-10-02 ✓ 原话："如果新砖与旧砖重叠率 >= 0.5，判定为同一登记砖……这个 0.5
        #   做成参数配置『上板防抖重叠率』" ✓）⇒ 透传 ✓
        self.board_overlap = board_overlap
        # ⭐⭐⭐ `iou_brick` = **"缩成砖判定 · 最小 IoU"**（用户 2026-10-03 ✓ 原话："我才学到 IoU
        #   这个算法，我认为我们很多判定都可以换成这个算法（例如『噪声容差』改成『**噪声最小
        #   IoU**』）" ✓）⇒ 透传 ✓（给"框缩回砖大小/位置"补一把**尺度无关**的尺 ✓ 与"四条边都在
        #   噪声容差(px) 内"并列 OR ✓；0 = 关 ✓ 见 `LieTracker._IOU_BRICK` ✓）。
        self.iou_brick = iou_brick
        # ⭐⭐⭐ `show_kf` = **"显示 KF 预测"开关**（用户 2026-10-03 ✓ 原话："**1. 移除"位置估计"
        #   参数**（非可视化的配置用户界面不需要），新增"**显示 KF 预测**"的开关" ✓）：
        #   **它不是"位置估计算法分支"** ✗（那名字会让人以为会改位置 ✗）—— 它是**可视化开关** ✓：
        #   `False`（默认 ✓）= 不跑 KF、画面不画 ⇒ 零开销零污染 ✓；`True` = 跑 KF（只记录 ✓）＋
        #   画青线/青点/青箭头 ✓ **位置仍走经典** ✓ 见 `LieTracker.show_kf` ✓。
        self.show_kf = show_kf
        # ⭐⭐⭐ `kf_pos` = **"KF 位置驱动"**（用户 2026-10-03 ✓ 原话："因为**预测结果不同，实际
        #   走向也会不同**，所以在**实时演算上叠加 KF 显示意义不大**。能不能：当『KF预测』开启时，
        #   将**窗口分为两列视图**，左边跑经典右边跑 KF，这样对照才有意义" ✓）：`True` ⇒ 这一路
        #   **真的用 KF 位置当地点** ✓（见 `LieTracker.kf_pos` ✓）⇒ 演示窗拿它当**右列** ✓。
        #   ⚠ **两列各跑一份 `Runner`**（各自一份 `LieTracker` ⇒ 状态完全隔离 ✓ 互不污染 ✓）。
        self.kf_pos = kf_pos
        self.tr = LieTracker(path_ms=path_ms, merge_iou=merge_iou,
                             merge_iou_out=merge_iou_out, sep_ratio=sep_ratio,
                             ring_cov_sep=ring_cov_sep, brick_iou_sep=brick_iou_sep,
                             ring_cov_fuse=ring_cov_fuse, brick_iou_fuse=brick_iou_fuse,
                             inherit_dist=inherit_dist, allow_back_ratio=allow_back_ratio,
                             edge_max=edge_max, sep_vel_max_ratio=sep_vel_max_ratio,
                             noise_tol=noise_tol,
                             board_s=board_s, board_overlap=board_overlap,
                             iou_brick=iou_brick, show_kf=show_kf, kf_pos=kf_pos)
        # ⭐⭐⭐ `follow_gain` = **鼠标跟随效率倍率**（用户 2026-10-03 ✓ 原话："能否开放一个系数
        #   配置，调**追踪器**跟上圆心的效率倍率？" ✓ + "**我不想影响位置计算**，只调追踪器
        #   （较大的白描边圈绿圆）" ✓）⇒ 透传给**控制器** ✓ —— 它只决定"**模拟鼠标朝圆心走多快**"
        #   ✗ 一个字都不动 `LieTracker` 的位置计算 ✓（1.0 = 全量，一拍尽量贴上去 ✓）。
        self.follow_gain = follow_gain
        self.ctl = LieMouseController(gain=gain, assume=assume, follow_gain=follow_gain)
        self.dets = dets
        self.prev_boxes = None                 # 上一帧的检测框（算逐框位移用 ✓）
        self.tgt_rad = None                    # ⭐ 真实目标的**实际半径**（学一次、之后冻结 ✓）
        self.rad_frozen = False                # 白块消失够久 ⇒ 冻结（透明后恒定 ✓）
        self.rad_seen = None                   # 最后一次"看见白块"的时刻 ✓
        self.rad_t0 = None                     # 第一次看见白块的时刻（学习窗口起点 ✓）
        self.rad_samples = []                  # 白块半径样本（学"实际半径"用 ✓ 取中位 ✓）
        self.prev_red = None                   # ⭐ 上一拍红框（**只给预演期的顺序兜底** ✓
                                               #   绘制阶段一律读预演结果 ⇒ 与拖动方向无关 ✓）
        self.prev_ts = None                    # 上一帧时间戳（算 dt ⇒ 粘滞箭头 ✓）
        self.pos_hist = []                     # 报告位置历史（算"追踪器轨迹速度" ✓ 用户 A1 ✓）
        self.cursor = list(assume)             # 模拟光标（处理域 ✓）
        self.pend = []                         # 待生效指令（一拍延迟 ✓ = 固件鼠标 ✓）
        self.hits = 0
        self.n = 0

    def step(self, small, i, ts):
        for cmd in self.pend:                  # 先让上一拍指令**落地**（时序同实机 ✓）
            self.cursor[0] += cmd[0] * self.ctl.gain[0]
            self.cursor[1] += cmd[1] * self.ctl.gain[1]
        self.cursor = [round(self.cursor[0]), round(self.cursor[1])]
        self.pend = []
        d = None
        raw = None
        if self.dets is not None:
            raw = self.dets[i] if i < len(self.dets) else None
            # ⚠ tracker 只吃 `[cx, cy, w, h]`（2026-09-30 起检测输出多了 conf ✗
            #   整条 `b[1:]` 会多一个字段 ⇒ 必须切 `b[1:5]` ✓）；
            #   **画图要用 `raw`**（带 cls/conf ✓ 标签/挖洞都要它 ✓ 别拿 tracker 那份 ✗）
            d = [list(b[1:5]) for b in raw] if raw else None
        out = self.tr.process(small, ts=ts, dets=d)
        pos = out["pos"] if out["state"] == "track" else None
        r = max(18.0, (out["area"] or 0.0) ** 0.5 / 1.7725)
        hit = None
        if pos is not None:
            self.n += 1
            dist = ((self.cursor[0] - pos[0]) ** 2
                    + (self.cursor[1] - pos[1]) ** 2) ** 0.5
            hit = dist <= r
            self.hits += 1 if hit else 0
        # ⭐ 每步都把**实测光标**喂给控制器（实机里就是"画面里量到的光标" ✓ 增益误差
        #   靠反馈自校正 ✓）
        cmd = self.ctl.step(pos, ts=ts, vel=out.get("vel"),
                            cursor=(self.cursor[0], self.cursor[1]))
        if cmd[0] or cmd[1]:
            self.pend.append(cmd)
        # ⭐ 运动向量快照（画每个候选的箭头 ✓ `followed_pos=pos` ⇒ 认出"真在跟哪个" ✓）
        motion = self.tr.motion_viz(followed_pos=pos)
        motion["followed_pos"] = pos                 # 实际跟随目标的位置（残差路线给的 ✓）
        motion["followed_v"] = out.get("vel")        # ⭐ 它自己的运动向量（**最该看的箭头** ✓）
        motion["n_cands"] = len(motion.get("tracks") or [])
        # ⭐⭐ **逐框位移**（用户 2026-09-30 定口径 ✓ 原话："正确的箭头至少应该是从本框上
        #   一帧的位置指向本帧的位置"）：每个检测框去上一帧的框里找**最近的那个**，
        #   位移 = 本帧中心 − 上一帧中心 ⇒ **方向就是真实运动方向** ✓。
        #   ⚠ 别拿追踪轨迹的速度当箭头（那是 8 帧窗口的中位差，静止候选上是噪声 ⇒
        #     方向乱指 ✗ 实测踩过 ✓）。
        # ⭐ 本帧间隔（秒 ✓）：画"粘滞"轨迹（`miss>0` 那条 ✓ 用户第④问）的箭头要用
        #   `轨迹速度(px/s) × dt` 换算成"这一拍走了多少 px" ✓ 与逐框位移同一量纲 ✓。
        #   ⚠ **必须在 `_box_moves` 之前算**（它要用 dt 把轨迹速度换成位移 ✓ 补漏配的框 ✓）。
        motion["dt"] = max(1e-3, ts - self.prev_ts) if self.prev_ts else 1.0 / 60.0
        self.prev_ts = ts
        # ⭐ **追踪器自己的轨迹速度**（用户 2026-09-30 A1 条 ✓ 白箭头就画它 ✓）
        self.pos_hist.append((ts, None if pos is None else float(pos[0]),
                              None if pos is None else float(pos[1])))
        if len(self.pos_hist) > 13:
            self.pos_hist.pop(0)
        motion["track_v"] = _track_velocity(self.pos_hist)
        moves, roles, preds = _box_moves(self.prev_boxes, raw, motion)
        motion["box_moves"] = moves
        motion["box_roles"] = roles
        motion["box_pred"] = preds      # ⭐ 这一格的位移是**轨迹预测**补的 ✓
        self.prev_boxes = raw
        # ⭐⭐⭐ **真实目标的"实际半径"**（用户 2026-09-30 口径 ✓ 原话："真实目标透明后，绿圈的
        #   半径应该恒定不变，因为我希望它代表**真实目标的实际半径**" ✓）。
        #   做法：**只在"面积像单个目标"的框上学习**（`_RAD_LO`~`_RAD_HI` × 标准 ✓ = 没融合
        #   也没分离 ✓），一旦融合/分离（面积跑偏 ✗）就**冻结** ⇒ 透明阶段圆半径恒定 ✓
        #   （不再跟着融合框呼吸 ✗ 用户实测就是这个现象 ✓）。学习本身也限速+EMA ⇒ 不跳变 ✓。
        tb = _choose_target_box(raw or [], motion, pos, None, motion.get("typ_area") or 0.0)
        if tb is not None and self.tgt_rad is None:
            # 白块还没出现前的兜底：先用**目标框半宽**（用户最早那条口径 ✓）
            self.tgt_rad = float(raw[tb][3]) / 2.0
        # ⭐⭐⭐ **学一次、之后永久冻结**（用户 2026-09-30 ✓ 原话："真实目标**透明后**，绿圈的
        #   半径应该恒定不变，因为我希望它代表**真实目标的实际半径**" ✓）：
        #   · **只在白块还在时学**（`out["white"]` = 白块面积 ✓ 有 ⇒ 还没透明 ✓ 那时目标 =
        #     唯一大白块 ⇒ `sqrt(A/π)` 就是**目标自己的实际半径** ✓ 比"框半宽"更贴 ✓）；
        #   · 面积落在"单个目标"带里才学（`_RAD_LO/_HI` × 标准 ✓ 排除"白块与倒计时数字并成
        #     一块"那种 ✗）；
        #   · 白块**连续消失 > `_RAD_FREEZE_S`** ⇒ 判定已透明 ⇒ **永久冻结** ✓（之后哪怕
        #     融合框撑到 1.9× 也不动它 ✗ 用户实测就是这个现象 ✓）。
        wa = out.get("white")
        if wa and not self.rad_frozen:
            # ⚠⚠ **别拿"框面积带"筛白块** ✗（实测踩到）：图形只填满外接框约 **60%** ⇒ 白块面积
            #   （~10162）永远落在"框面积带"（0.85~1.25 × 18500 = 15700+）之外 ✗ ⇒ 学习**一次都
            #   不触发**（量出来半径一直卡在兜底的 76.2 ✗）。⇒ 改成**对白块样本取中位** ✓：
            #   干净帧（~57 ✓）占多数、与倒计时数字并块的少数帧（~73 ✗）被中位压掉 ✓。
            self.rad_samples.append((wa / 3.141592653589793) ** 0.5)
            if len(self.rad_samples) > 240:
                self.rad_samples.pop(0)
            self.tgt_rad = sorted(self.rad_samples)[len(self.rad_samples) // 2]
            self.rad_seen = ts
            if self.rad_t0 is None:
                self.rad_t0 = ts
            if ts - self.rad_t0 > _RAD_LEARN_S:
                self.rad_frozen = True                    # 学习窗口到点 ⇒ 强制冻死 ✓
        elif wa:
            self.rad_seen = ts                            # 还看得见 ⇒ 刷新"最后见到"的时刻 ✓
        elif self.rad_seen is not None and ts - self.rad_seen > _RAD_FREEZE_S:
            self.rad_frozen = True                        # 白块没了够久 ⇒ 冻死 ✓
        motion["rad_frozen"] = self.rad_frozen
        # ⭐⭐ **优先用追踪器学到的"目标实际半径"**（同源 ✓ 别两边各学一套 ✗ 会不一致：
        #   追踪器那份用在"融合块缩小时的**相切落位**" ✓ 图形半径必须跟它一致才对得上 ✓）
        # ⭐⭐⭐⭐⭐ **圆半径 = 后端"唯一圆"那个数**（用户 2026-10-04 ✓ 原话："**我们一定要
        #   所见即所得，不要表面搞一套背后搞一套**" ✓✓）—— ⚠⚠ `out["tgt_rad"]` **就是后端
        #   判据 `has_tgt` 用的那个** ✗（见 `MotionTracker._circ_rad` ✓）⇒ **判据 / 画面 /
        #   标签 三者同源同拍** ✓✓（这才是"所见即所得" ✓）。
        #   ⚠ 原来这里演示窗**自己学一套** ✗（白块面积中位 ＋ `out["rad"]` —— 而后者是**那一格
        #     框**的 `0.25×(w+h)` ✓ 融合/被切边时会胀/塌 ✗ ⇒ 实测帧 25~28：画面 73.1~73.6、
        #     判据口径 64.3→45.9 ✗✗ ⇒ 同一个框两边结论相反 ✓）。
        #   ⚠ 回落顺序：**运动模式**（后端会给 `tgt_rad` ✓）⇒ 用它；**经典模式**没有这个键 ⇒
        #     才走原来那套（`out["rad"]` / 白块学习值 ✓ 经典模式行为**不变** ✓）。
        _tr2 = out.get("tgt_rad")
        if _tr2:
            self.tgt_rad = float(_tr2)
        elif out.get("rad"):
            self.tgt_rad = float(out["rad"])
        motion["tgt_radius"] = self.tgt_rad        # 全分辨率半径 px（画时 × 缩放 ✓）
        # ⭐⭐⭐ **唯一红框 = 后端认定的目标框**（用户 2026-09-30："我以帧号递增顺序切到 21 帧
        #   红框在左边，为什么以帧号递减顺序切到 21 帧红框在右边？后台数据到底认为哪个是真目标
        #   所属的框？" ✓✓）。
        #   原实现把判定放在 `draw()` 里、还用一个**跨帧累积**的 `draw._last_red` ✗ ⇒ 红框
        #   取决于**你从哪边拖过来** ✗✗（同一帧两个答案 = 真 bug ✓）。
        #   ⇒ 现在：**预演期（顺序、一次）**算好索引存进本帧结果 ✓，绘制只读 ✓（纯函数 ✓）。
        #   权威来源 = 追踪器输出的 `tbox`（它选"包含报告位置优先、否则门内最近"那格 ✓）。
        ri = None
        _tb = out.get("tbox")
        if _tb is not None and raw:
            bi, bd = None, None
            for j, b in enumerate(raw):
                d = ((b[1] - _tb[0]) ** 2 + (b[2] - _tb[1]) ** 2) ** 0.5
                if d <= _RED_MATCH_GATE and (bd is None or d < bd):
                    bi, bd = j, d
            ri = bi
        if ri is None:                             # 建档/丢帧（没有目标框）⇒ 顺序兜底 ✓
            ri = _choose_target_box(raw or [], motion, pos, self.prev_red,
                                    motion.get("typ_area") or 0.0)
        if ri is not None and raw:
            _b = raw[ri]
            self.prev_red = (_b[1], _b[2], float(_b[3]) * float(_b[4]))
        motion["red_i"] = ri
        # ⭐ 把"后端答案"也放进 `motion`（演示窗/探针都能直接看 ✓ 用户问"后台认为哪个" ✓）
        motion["tbox"] = out.get("tbox")
        # ⭐⭐⭐⭐⭐ **融合框«夹取圆»这一拍生效的允许区域**（用户 2026-10-04 ✓ 见 `lie_motion` 的
        #   "第二种指导"那段 ✓）—— `(x0, y0, x1, y1)`（加工域 ✓）／ `None` = 这一拍没夹 ✓。
        #   ⚠ `draw()` 会把它**画出来** ✓（所见即所得 ✓ 否则"圆为什么被按住"看不出来 ✓）。
        motion["fuse_clamp"] = out.get("clamp")
        # ⭐⭐⭐⭐⭐ **夹取这一拍"记进速度"的那一份**（用户 2026-10-04 ✓ 原话："**位置被硬夹说明你
        #   要么快了要么慢了，夹取也要算进速度平滑。因为夹取是修正你的预测误差，夹取是较准确的
        #   测量值**" ✓✓）—— 见 `lie_motion` 那段 ✓：`(kx, ky)` px/拍 ✓／`None` = 没夹 ✓。
        #   ⚠ 面板会**写出来** ✓（否则"黄箭头为什么突然变了"看不见原因 ✗ 见下面 ✓）。
        motion["clamp_vel"] = out.get("clamp_vel")
        # ⭐⭐ **未平滑的追踪器真实位置**（用户 2026-09-30 认知 #3 ✓）：演示窗画**绿点**用它 ✓，
        #   **绿圈**画 `pos`（后端报告位置 ✓）⇒ 两点不一致 = **拟人平滑正在生效** ✓（#4 ✓）。
        motion["raw_pos"] = out.get("raw_pos")
        # ⭐⭐ **纳入预测的轨迹**（用户 2026-10-01 ✓ 原话："用白色的线把纳入预测的轨迹画出来" ✓）
        motion["path_pts"] = out.get("path_pts") or []
        # 用户 2026-10-01：圆圈半径 = 红框内切半径（丢框后冻结 ✓）+ 红框宽高快照（画淡粉矩形 ✓）。
        motion["tbox_rad"] = out.get("tbox_rad")
        motion["tbox_wh"] = out.get("tbox_wh")
        # ⭐⭐⭐ **两个速度分量 → 两支箭头**（用户 2026-10-01 定稿 ✓ 原话："用**黄色箭头**标圆圈的
        #   **绝对速度**，用**白色箭头**标圆圈**相对于假目标群的速度预测**" ✓）：
        #   · 黄 = `vel`（绝对/屏幕速度 ✓ = 自身 + 群体 ✓）；
        #   · 白 = `vel_rel`（**相对假目标群**的自身速度 ✓ 丢框后沿用的就是它 ✓）。
        motion["vel_abs"] = out.get("vel")
        motion["vel_rel"] = out.get("vel_rel")
        # ⭐⭐⭐ **假目标登记表快照**（用户 2026-09-30："将登记的框做个标记显示出来我看看你登记的
        #   对不对，例如 `shape 0.98 #001`" ✓）—— 演示窗直接画它 ✓（只读 ✓ 不影响任何判定 ✓）。
        motion["reg"] = out.get("reg")
        # ⭐⭐ **本拍红框是不是"融合框"**（用户 2026-10-01 新流程 ✓）也转发给演示窗/自检 ✓：
        #   融合 ⇒ 不做信念夹取、绿圈按白箭头（预测）走 ✓（"边归属相切"整套已移除 ✓）。
        motion["merged"] = bool(out.get("merged"))
        return out, pos, r, hit, raw, motion      # `raw` = 检测原框（带 cls/conf ✓ 画图用 ✓）

    def reset(self, assume=(375.0, 250.0)):
        # ⚠ 这里只回传了 `path_ms` / `merge_iou`（既有口径 ✓ 我按同一处加 `sep_ratio` ✓
        #   —— `noise_tol` / `board_s` / `board_overlap` 三个**本来就没回传** ✗，不在本轮范围内 ✓）。
        self.__init__(dets=self.dets, gain=self.ctl.gain, assume=assume,
                      path_ms=self.path_ms, merge_iou=self.merge_iou,
                      merge_iou_out=self.merge_iou_out,
                      ring_cov_sep=self.ring_cov_sep, brick_iou_sep=self.brick_iou_sep,
                      ring_cov_fuse=self.ring_cov_fuse,
                      brick_iou_fuse=self.brick_iou_fuse,
                      sep_ratio=self.sep_ratio, inherit_dist=self.inherit_dist,
                      allow_back_ratio=self.allow_back_ratio, edge_max=self.edge_max,
                      sep_vel_max_ratio=self.sep_vel_max_ratio,
                      follow_gain=self.follow_gain, iou_brick=self.iou_brick,
                      show_kf=self.show_kf, kf_pos=self.kf_pos)


def analyze(frames, dets, gain=None, assume=(375.0, 250.0), on_progress=None,
            follow_gain=None, iou_brick=None, show_kf=None, kf_pos=None,
            merge_iou=None, merge_iou_out=None):
    """**整段一次算完**（用户 2026-09-30 要求 ② ✓ 原话："加载视频的时候就演算完，
    这样我拖帧就不用重算了"）⇒ 返回每帧结果列表，播放/拖动只是**查表** ✓。

    每项：`{"pos","r","hit","state","cursor","boxes"}`（都是**处理域**坐标 ✓ 画时再放大）。
    `on_progress(k, n)` 每帧回调（界面报进度 ✓）。
    `follow_gain` ⇒ 透传给控制器（**鼠标跟随效率倍率** ✓ 用户 2026-10-03 ✓ 见 `Runner` ✓）。
    """
    r = Runner(dets=dets, gain=gain, assume=assume, follow_gain=follow_gain,
               iou_brick=iou_brick, show_kf=show_kf, kf_pos=kf_pos,
               merge_iou=merge_iou, merge_iou_out=merge_iou_out)
    out = []
    for i, (_big, small) in enumerate(frames):
        o, pos, rad, hit, d, motion = r.step(small, i, i / 60.0)
        out.append({"pos": pos, "r": rad, "hit": hit, "state": o["state"],
                    "cursor": tuple(r.cursor), "boxes": d or [],
                    "motion": motion,
                    # ⭐ **本拍砖表快照**（用户 2026-10-03 ✓ 与演示窗同口径 ✓ 点选时算
                    #   "IoU 最大的砖"用它 ✓）
                    "bricks": o.get("bricks")})
        if on_progress and (i % 20 == 0 or i == len(frames) - 1):
            on_progress(i + 1, len(frames))
    return out


def dim_outside_roi(frame, roi, blur=4.0, gray=96, mix=0.45, scale=(1.0, 1.0)):
    """**ROI 之外**的像素 一律**模糊 ＋ 置灰**（用户 2026-10-07 ✓ 原话：「**ROI外的区域置灰+模糊**」✓）。

    · `roi` 为空 / 非法 ⇒ **一个像素都不碰** ✓（= 老行为 ✓ 也零拷贝 ✓）；
    · 做法 = **框外那四条带**各做一次"高斯模糊 ＋ 往灰里混" ✓ —— ⚠ 不做"整幅模糊再贴回"
      ✗（效果一样，但要多一次整幅拷贝 ✓ 白花 ✓）；
    · ⚠⚠ **纯显示** ✗：一个判据都不碰 ✓ —— 框外那些检出框**后端本来就没算** ✓
      （见 `VelocityTracker.process` 开头那道 ROI 过滤 ✓），这里只是把"不算"**画出来** ✓。

    ⚠⚠⚠ **`scale` 与 `apply_mask` 同口径** ✗✗（用户 2026-10-07 ✓ 报："**模糊的区域和框选的区域
      不一致**" ✓✓ —— **实测现场**：ROI 加工域 `(200.5, 75.6, 688.5, 399.9)` ✓、原图 **1920×1080**
      ／加工帧 **889×500** ⇒ `scale` **(2.160, 2.160)** ✓）：
      · **`frame` 是原图域** ✓（`draw()` 的调用方一律喂 `decode_big(...)` ✓ 见 `apply_mask` 同一个
        约定 ✓），而 **`roi` 是加工域** ✓（与检出框 / 判据同一把尺 ✓）；
      · ⇒ 我上一版**没乘** ✗ ⇒ 于是把"加工域那个小矩形"当成原图域 ⇒ **整块弹窗被糊掉**
        （**实测 673,823 px 糊进了 ROI 里** ✗✗）＋ 真 ROI 之外**漏糊 107,959 px** ✗；
      · ⚠ **画面上那个黄框是对的** ✗✗（`draw()` 里画框时乘了 `scale` ✓）⇒ **框和糊对不上** ✓
        = 用户看到的那一幕 ✓。
    """
    if not roi or frame is None:
        return frame
    h, w = int(frame.shape[0]), int(frame.shape[1])
    try:
        x0, y0, x1, y1 = (int(round(float(roi[0]) * float(scale[0]))),
                          int(round(float(roi[1]) * float(scale[1]))),
                          int(round(float(roi[2]) * float(scale[0]))),
                          int(round(float(roi[3]) * float(scale[1]))))
    except Exception:                       # noqa: BLE001 —— 缺数 ⇒ 当没给 ✓（宁可不画 ✗）
        return frame
    if x1 <= x0 or y1 <= y0:                # ⚠ **反了 / 退化** ⇒ 同样当没给 ✓（否则四条带会重叠 ✗）
        return frame
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    for (ax, ay, bx, by) in ((0, 0, x0, h), (x1, 0, w, h), (x0, 0, x1, y0), (x0, y1, x1, h)):
        if (bx - ax) < 1 or (by - ay) < 1:
            continue
        _sub = frame[ay:by, ax:bx]
        _b = cv2.GaussianBlur(_sub, (0, 0), float(blur))
        frame[ay:by, ax:bx] = cv2.addWeighted(_b, 1.0 - float(mix),
                                              np.full_like(_b, int(gray)), float(mix), 0)
    return frame


def apply_mask(frame, boxes, alpha, target=None, scale=(1.0, 1.0)):
    """⭐ **灰蒙版 + 框挖洞**（用户 2026-09-30 要求 ① 原话："把 yolo 工作台的背景蒙版 +
    框挖洞那一套用到测谎演示里" ✓ 同款语义：**框外压暗、框里保持原亮度** ✓）。

    `alpha` = 蒙版不透明度（0 = 关 ✓）；洞 = **每个检测框** ✓（+ 可选 `target` 目标圆 ✓）。
    ⚠ **当前调用方（`draw`）只传检测框** ✗（用户 2026-10-02 ✓ 原话："你可以让非检出框都受灰蒙版
      影响吗？目前绿色圆圈不是检出框也不受灰蒙版影响" ✓）—— 目标圆**不再挖洞** ⇒ 那块区域跟
      背景一起压暗 ✓；而圆线照旧**鲜亮** ✓（所有线条都是蒙版之后画的 ✓）。
      `target` 参数与逻辑**保留不删** ✓（要回退就把 `draw` 里那两行加回来 ✓）。
    现场帧 1620×1080 的整幅 alpha 运算 ~10ms ✓ 可接受（不透明为 0 时直接跳过 ✓）。
    """
    if alpha <= 0.005:
        return frame
    h, w = frame.shape[:2]
    # ⚠ **别用整幅 alpha 浮点蒙版**（1620×1080 浮点运算 + 两次大数组分配 ⇒ ~60ms/帧 ✗
    #   60fps 播放直接掉到十几帧 ✗ 实测踩过）。**便宜做法**：
    #   ① 整幅 `addWeighted` 压暗（C++ 快 ✓）→ ② 把**洞里**的原图像素**拷回来** ✓
    #   （洞的总面积只有几万像素 ⇒ 拷贝开销可忽略 ✓）
    gray = np.full_like(frame, 88)
    out = cv2.addWeighted(frame, 1.0 - float(alpha), gray, float(alpha), 0)
    rects = []
    for b in boxes:
        cx, cy, bw, bh = (b[1] * scale[0], b[2] * scale[1],
                          b[3] * scale[0], b[4] * scale[1])
        rects.append((cx, cy, bw / 2.0, bh / 2.0))
    if target is not None:                    # 目标圈也挖（看目标才不费眼 ✓）
        rects.append((target[0], target[1], target[2], target[2]))
    for cx, cy, hw, hh in rects:
        x1, y1 = max(0, int(cx - hw) - 2), max(0, int(cy - hh) - 2)
        x2, y2 = min(w, int(cx + hw) + 2), min(h, int(cy + hh) + 2)
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return out


def pick_box_at(boxes, pt, tol=40.0):
    """**点（加工域坐标）命中哪一格检出框**（用户 2026-10-03 ✓ 原话："你能让我在**点选检出框**
    的时候显示 `(360.4,323.0) 179×208=37232` 这种信息吗" ✓）—— 纯函数（好测 ✓ 窗口与自检共用 ✓）。

    规则：
      · 点**落在框里** ⇒ 命中 ✓；**多个框套着** ⇒ 取**面积最小**的那个（里层优先 ✓
        一眼点在"最里面那格" ✓ —— 融合并集框套着砖框时不会选错 ✓）；
      · 都不落在里面 ⇒ 退回**中心最近**且 ≤ `tol` px 的一格 ✓（手抖差一点也能选上 ✓）；
      · 连近的也没有 ⇒ **`None`**（调用方据此**取消选中** ✓）。
    """
    best, bs = None, None
    for b in (boxes or []):
        try:
            cx, cy = float(b[1]), float(b[2])
            w, h = float(b[3]), float(b[4])
        except Exception:                       # noqa: BLE001 —— 布局不合（少于 5 列）就跳过 ✓
            continue
        inside = (cx - w / 2.0 <= pt[0] <= cx + w / 2.0
                  and cy - h / 2.0 <= pt[1] <= cy + h / 2.0)
        d = ((cx - pt[0]) ** 2 + (cy - pt[1]) ** 2) ** 0.5
        if inside:
            sc = (w * h) ** 0.5                 # 套着的 ⇒ 里层（面积小）优先 ✓
        elif d <= float(tol):
            sc = 1e6 + d                        # 差一点 ⇒ 按中心近的选 ✓
        else:
            continue
        if bs is None or sc < bs:
            best, bs = tuple(b), sc
    return best


def pick_extra(box, pos=None, rad=0.0, bricks=None):
    """点选框的**重叠量**（= 画布标签**第 2 行**的内容 ✓ **纯函数** ✓ 好测 ✓ 窗口与自检共用 ✓）。

    用户 2026-10-03 ✓ **两次**（都在同一天 ✓）：
      · 第一次："**点选检出框的时候，希望他能显示 IoU 最大的砖的 IoU**" ✓；
      · 第二次（本次）："点选检出框时，**与圆外接矩形的 iou**、**圆外接矩形与其相交的比例**也
        显示下；另外，以上参数**及旧的 iou 显示都放在第二行**" ✓。
    ⇒ 回 `(砖段, 圆段)` 两串（**空串 = 不显示那一段** ✓ 不猜 ✗）：
      · 砖段 = `IoU最大砖 %.2f @(x,y)` —— 与**哪块砖** IoU 最大 ✓（同一把尺 `geom.iou` ✓）；
      · 圆段 = `圆矩IoU %.2f | 圆矩∩框 %.2f`：
          - `圆矩IoU` = **`IoU(检出框, 圆外接矩形)`** ✓（= **旧口径**那个量 ✓ 留作对照 ✓）；
          - `圆矩∩框` = **圆外接矩形与检出框相交的比例** ✓ = 交 ÷ **圆矩形面积**（**单向** ✓
            ⇒ 与后端 `LieTracker._ring_cov` **同一把尺** ✓ 不自己写一份 ✗）。
    ⚠ `box` = `(cx, cy, w, h)`（**加工域** ✓ 与红框/登记/状态栏同一把尺 ✓）；
      `pos` / `rad` 拿不到（**预测态**没圆心 / 半径还没学到 ✓）⇒ **圆段留空** ✓；
      `bricks` 空（未演算 / 表为空 ✓）⇒ **砖段留空** ✓。
    ⚠ `bricks` 的元素布局 = **`(bid, x, y, w, h)`** ✓ —— 与后端的本拍快照**同一份**
      （`LieTracker._bricks_snap` ✓ `bid` 是**元组** ✓ 见那里 ✓）；`bid` 用来标"是哪一块砖" ✓。
    """
    _btxt = ""
    _best, _bv = None, -1.0
    for _t in (bricks or []):
        _v = geom.iou(box, (float(_t[1]), float(_t[2]), float(_t[3]), float(_t[4])))
        if _v > _bv:
            _best, _bv = _t, _v
    if _best is not None:
        _btxt = "IoU最大砖 %.2f @(%d,%d)" % (_bv, int(_best[0][0]), int(_best[0][1]))
    _rtxt = ""
    _rr = float(rad or 0.0)
    if pos is not None and _rr > 0.0:
        _ring = (float(pos[0]), float(pos[1]), 2.0 * _rr, 2.0 * _rr)
        _rtxt = ("圆矩IoU %.2f | 圆矩∩框 %.2f"
                 % (geom.iou(box, _ring), geom.cover_ratio(box, _ring)))
    return _btxt, _rtxt


#: ⭐⭐⭐ **点选信息框的排版**（用户 2026-10-04 ✓ 原话："点选之后的信息框能做个**最小宽度+自动换行**
#:   吗？现在**整个屏幕都不够宽了**" ✓✓）——
#:   · `_PICK_MIN_W` = **最小宽度**（px ✓）：短信息也撑成一块像样的面板 ✓ 不缩成一条 ✗；
#:   · `_PICK_MAX_W` = **最大宽度**（px ✓）：超过就**自动换行** ✓；
#:   · `_PICK_MARGIN` = 距画面左右边的最小留白 ✓（实际换行宽 = `min(最大宽, 画面宽−2×它)` ✓
#:     ⇒ **绝不顶出屏幕** ✓）；
#:   · `_PICK_TOP` = 面板最高不许顶到的 y（⚠ HUD 画在 `y≈38` ✓ 会让开它 ✓）。
_PICK_MIN_W = 320
_PICK_MAX_W = 640
_PICK_MARGIN = 12
_PICK_TOP = 54

#: ⭐⭐⭐⭐⭐ **两档模式各自的「图例」文字**（用户 2026-10-07 ✓「速度跟踪」那一轮）——
#:   · `_LEGEND_MOTION` = **运动分离**那 16 行（**原样搬过来** ✗ 一个字不改 ✓ 零污染 ✓）；
#:   · `_LEGEND_VELOCITY` = **速度跟踪**的**初始图例**（用户 2026-10-07 ✓ **原话给的、只这 7 条** ✓）。
#:   ⚠⚠⚠ **项目纪律**（用户 2026-10-07 ✓ 原话："以后**不得私自新增**任何观测参数 / 图例条目 /
#:     面板文字 / 底栏提示。要新增**必须先问用户、得到同意**才做"✓）⇒ **这 7 条之外一条都不许加** ✗
#:     （自检脚本内部的输出不算 ✓ 调试日志不算 ✓）。
#:   ⚠ 中文照用 ✓（这个 QLabel 是 **Qt** 画的 ⇒ 支持 Unicode ✓；`cv2.putText` 那一路才只认 ASCII ✓）。
_LEGEND_MOTION = (
    "绿圈：目标位置（我们以为目标在这）\n"
    "绿点（白圈套）：圆心标记；🔴红点=鼠标已脱离圆圈\n"
    "白点圈：模拟鼠标（光标）\n"
    "白箭头：目标这一拍【实际】相对群体位移（切橙线）\n"
    "黄箭头：目标【下一拍】预计相对群体位移（滑行＋框心指导之后）\n"
    "橙线：目标相对群体轨迹\n"
    "暗黄线：鼠标相对群体轨迹\n"
    "蓝框=正常 ｜ 洋红框=融合（真目标＋假目标）｜ 橙框=可疑 ｜ 灰框=残缺\n"
    "（⚠ 融合**刚结束**的「冷却」几拍**不再是洋红** ✗：那格框已经分开、就是真目标\n"
    "  自己那格 ⇒ 画成**正常色**，状态由面板写「冷却中」✓）\n"
    "洋红细框：融合框【夹取】圆心的允许区（圆必须待在里面）\n"
    "粉框：目标检出框丢失后的预测接力框\n"
    "框上小字 shape…：该框置信度 / 编号 / 匹配分\n"
    "「#N 推的」灰框：这一拍没配上、位置是推的\n"
    "🟡黄粗框：我刚点选的那一格\n"
    "框上的蓝/绿/黄小箭头：该框这一拍的运动方向\n"
    "（开「显示 KF 预测」时另有青色线/点/箭头）")
#: ⚠ **只这 7 条**（用户原话 ✓ 逐字 ✓）—— 画面里**只留**这 7 样东西（见 `draw` 里那道 `_vel` 闸 ✓）。
_LEGEND_VELOCITY = (
    "1. 蓝色框：yolo检出框（若注册过框左上有标签）\n"
    "2. 灰色框：有标签的假目标丢失检出框时的记录\n"
    "3. 绿圆：此时认为真目标所处的位置\n"
    "4. 红点：此时鼠标位置\n"
    "5. 蓝箭头：检出框的绝对速度\n"
    "6. 白箭头：真目标的相对群体速度\n"
    "7. 橙色线：圆心相对群体轨迹")
#: ⭐⭐⭐ **速度跟踪档里蓝箭头的显示长度**（**固定** ✓ 用户 2026-10-07 ✓ **最终口径**：原话
#:   "**蓝箭头不再以长短展示速度大小，只代表速度方向，固定长度**" ✓✓）。
#:   ⚠ 所以这一档：**长度恒 = 它**（显示域 px ✓）、**方向**取后端给的每拍位移 ✓；
#:     位移小到没方向（< 0.5px ✓）⇒ **不画** ✓（不编一个方向 ✗）；`_bdx/_bdy` 一个数不改 ✓。
#:   ⚠ 老两档**一个字不变** ✗（那边仍是"长度 = 位移" ✓ 见 `draw` 里那段 ✓）。
_VEL_ARR_LEN = 46.0


def wrap_px(lines, max_px, scale=0.75, thick=1):
    """把每行按**像素宽度** `max_px` **贪心断行** ⇒ 新的行表（**纯函数** ✓ 好测 ✓）。

    用户 2026-10-04 ✓ 原话："点选之后的信息框能做个**最小宽度+自动换行**吗？现在**整个屏幕都
    不够宽了**" ✓✓ —— ⚠⚠ `cv2.putText` **不认 `\\n`** ✗（画的时候本来就是**逐行画** ✓ 见
    `draw` ✓）⇒ 这里只要**多切几行** ✓，画法**一个字不动** ✓。
    ⚠ **按字符累加宽度、用 `cv2.getTextSize` 现量** ✓ —— 中文/全角在 `putText` 里会被替换成
      `?` ✗，而**量出来的宽度就是那个替换字形的宽** ✓ ⇒ 与最终画出来**一致** ✓（不会"量窄了、
      画出去" ✗）。
    ⚠ 断点：优先在**空格**处断（英文单词别劈开 ✓，回退不超过 12 个字符 ✓）；找不到就**硬断** ✓
      （中文长串本来就没有空格 ✓）。
    """
    _out = []
    for _ln in lines:
        _cur = ""
        for _ch in str(_ln):
            _t = _cur + _ch
            _w = cv2.getTextSize(_t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
            if _cur and _w > max_px:
                _cut = _cur.rfind(" ")
                if 0 < _cut and (len(_cur) - _cut) <= 12:      # 英文单词别劈开 ✓
                    _out.append(_cur[:_cut].rstrip())
                    _cur = (_cur[_cut + 1:] + _ch).lstrip()
                else:
                    _out.append(_cur.rstrip())
                    _cur = "" if _ch == " " else _ch
            else:
                _cur = _t
        if _cur.strip():
            _out.append(_cur.rstrip())
    return _out or [""]


def pick_label(pick, names=None):
    """点选框的**画布标签**（用户 2026-10-03 ✓ **两行** ✓ 原话："点选检出框时，与圆外接矩形的
    iou、圆外接矩形与其相交的比例**也显示下**；另外，以上参数**及旧的 iou 显示都放在第二行**" ✓）。

    · **第 1 行** = `类别 置信度 | (cx,cy) WxH=面积`（用户点名的那串数字 ✓）；
    · **第 2 行** = 几个**重叠量** ✓ —— 全部由 `_pick_box` 算好附在**第 7/8 位**（`pick[6:]` ✓）：
        `IoU最大砖 …`（旧有 ✓）｜ `圆矩IoU …` ｜ `圆矩∩框 …`（本次新增 ✓）；
      该项为空（未演算 / 砖表空 / 圆心或半径还没学到）⇒ **不显示那一段** ✓ 不猜 ✗。

    ⚠ 只出 **ASCII** ✗（乘号写 `x`、间隔写 `|`）：`cv2.putText` 不认全角/中文 ⇒ 会画成 `?` ✗
      —— 状态栏那行是 Qt 画的 ⇒ 那边仍用 `×` / `｜` ✓（见 `_status` ✓）。
    """
    if pick is None:
        return ""
    cx, cy = float(pick[1]), float(pick[2])
    w, h = float(pick[3]), float(pick[4])
    nm = ""
    if names:
        cl = int(pick[0])
        nm = (names[cl] if 0 <= cl < len(names) else str(cl))
    # ⚠ 置信度可能是 `None`（离线/自检链路的框只有 5 列或没带 conf ✓）⇒ 只在这时**别拼** ✓ 不崩 ✗
    if nm and len(pick) > 5 and pick[5] is not None:
        nm = "%s %.2f" % (nm, float(pick[5]))
    txt = "(%.1f,%.1f) %dx%d=%d" % (cx, cy, int(round(w)), int(round(h)),
                                   int(round(w * h)))
    _l1 = ("%s | %s" % (nm, txt)) if nm else txt          # 第 1 行：几何 / 面积 ✓
    # ⚠⚠ **第 2 行照旧 join** ✗ —— ⚠ 我一度在这里加了"只留纯 ASCII"的过滤 ✗✗ **结果自检当场
    #   抓住**（`test_pick_box` 断言经典模式那行是 `IoU最大砖 … | 圆矩IoU … | 圆矩∩框 …` ✓
    #   全被滤没了 ⇒ `IndexError` ✗）⇒ **回退** ✓。
    #   ⚠ 教训：`cv2.putText` 画不出中文（会变 `?` ✓）是**既有的老问题** ✗，而"第 2 行包含中文"
    #     是**既有契约** ✓ ⇒ **不能在改文案这轮顺手改它** ✗（会同时改掉接口 ✓ 得单独一轮 ✓）。
    #   ✅ 本轮只做**用户要的那件事**：把 `_pick_box` 给的**多行**原样铺开（一行一主题 ✓）。
    _segs = []
    for _x in pick[6:]:
        if not _x:
            continue
        for _ln in str(_x).split("\n"):
            if _ln:
                _segs.append(_ln)
    _l2 = " | ".join(_segs)
    return ("%s\n%s" % (_l1, _l2)) if _l2 else _l1


def pick_clip_text(pick, names=None, frame_no=None, total=None, state=None):
    """点选框的**剪贴板文本**（用户 2026-10-03 ✓ 原话："加一个**右键点击检出框选中并弹出菜单**，
    目前只有一项『**复制检出框信息**』，用来我**复制之后与你交流**" ✓）。

    与画布标签（`pick_label` ✓）**同一份数据** ✓，但按"**要粘出去给人看**"重排 ✗：
      · **全角 / 中文照用** ✓ —— 走 Qt 剪贴板，不受 `cv2.putText` 的 ASCII 限制 ✗；
      · 多一行**上下文**（帧号 / 状态 ✓）—— 交流时能说清"**是哪一拍**" ✓（不然只有一个框的
        数字，对不上是哪一帧 ✗）；
      · 三个**重叠量**另起一行 ✓（与画布第 2 行同源 ✓ `pick[6:]` ✓）。
    形如：
        帧 37/120 ｜ 状态 merged
        检出框 shape 0.39 ｜ 中心 (360.4, 323.0) ｜ 尺寸 179×208 ｜ 面积 37232
        IoU最大砖 0.62 @(120,80) ｜ 圆矩IoU 0.41 ｜ 圆矩∩框 0.88
    ⚠ **拿不到的量一律不编** ✗：没 `names` ⇒ 不写类别 ✓；没有重叠量 ⇒ 少一行 ✓；
      没给帧号 ⇒ 不出上下文那行 ✓（自检用例就靠这一条 ✓）。
    """
    if pick is None:
        return ""
    _cx, _cy = float(pick[1]), float(pick[2])
    _w, _h = float(pick[3]), float(pick[4])
    _nm = ""
    if names:
        _c = int(pick[0])
        _nm = (names[_c] if 0 <= _c < len(names) else str(_c))
    if _nm and len(pick) > 5 and pick[5] is not None:
        _nm = "%s %.2f" % (_nm, float(pick[5]))
    _out = []
    if frame_no is not None:
        _out.append("帧 %s%s ｜ 状态 %s"
                    % (frame_no, ("/%d" % total) if total else "", state or "-"))
    _out.append("检出框%s ｜ 中心 (%.1f, %.1f) ｜ 尺寸 %d×%d ｜ 面积 %d"
                % ((" " + _nm) if _nm else "", _cx, _cy,
                   int(round(_w)), int(round(_h)), int(round(_w * _h))))
    # ⚠ 那几段是 `pick_extra` 生成的 ✓ 内部用的是**半角 `|`** ✗（画布给 `cv2.putText` 用 ✓
    #   全角会画成 `?` ✗）⇒ 这里**换成全角 `｜`** ✓（剪贴板是给人看的 ✓ 混排会显脏 ✗）。
    # ⚠⚠ **逐行拆开、别 join 成一长行** ✗✗（用户 2026-10-04 ✓ 原话："**这也太长了，你能归纳
    #   信息分行显示吗？**" ✓）—— `_pick_box` 运动模式给的本来就是**多行** ✓
    #   ⇒ 这里**按行摆出去** ✓（`\n` 拆开 ✓ 保持"一行一个主题" ✓）。
    # ⚠⚠ **运动模式那一版有「多行 + 一个 ASCII 短码」两段** ⇒ 短码**别重复贴** ✗
    #   （它是**画布专用的浓缩版** ✓ 内容全在详细版里 ✓ 两个一起贴看着像重复 ✓）。
    #   ⚠⚠ **判据只能用「有没有换行」** ✗✗ —— 我第一版拿"含不含中文"当判据 ✓ **错** ✗：
    #     经典模式那两段（`IoU最大砖 …` / `圆矩IoU …`）**也是中文** ✓ ⇒ 判据会把它俩
    #     也砍成一段 ⇒ **自检当场抓住**（断言恰好 3 行 ✓ 实测只剩 2 行 ✗）✓
    #     ⇒ 而"运动模式给的是**多行**"才是**它独有的特征** ✓✓。
    #   ⚠ 另外**必须防元组长度**（自检里有 5 位的 `pick` ✗ ⇒ `pick[6]` 会 `IndexError` ✓）。
    _p6 = str(pick[6]) if len(pick) > 6 else ""
    _src = (pick[6:7] if "\n" in _p6 else pick[6:])
    # ⚠⚠ **经典模式那两段仍要 `join` 成一行** ✗✗（我一度把它们也拆成两行 ⇒ 行数 3→4 ⇒
    #   **自检当场抓住** ✓）；而**运动模式那一版自带 `\n`** ✓ ⇒ join 之后**自然还是多行** ✓✓
    #   （一个 `join` 同时满足两边 ✓ 不用分支 ✓）。
    _extra = " ｜ ".join([str(_x).replace("|", "｜") for _x in _src if _x])
    if _extra:
        _out.append(_extra)
    return "\n".join(_out)


def box_label_lines(tid, shape=None, m=None):
    """⭐⭐⭐⭐⭐ **检出框标签的「两行文字」**（用户 2026-10-04 ✓ 原话："**把目标标签里的例如
    「#14 m0.89」换一行显示**" ✓✓）—— 返回 `(第 1 行, 第 2 行)`：

      · **第 1 行** = `0.98`（**置信度** ✓ 2026-10-07 起**不带 `shape` 那个词**了 ✓ 见下 ✓）；
        拿不到 ⇒ `-` ✓（⚠ 那一行**必须占着** ✗ —— 排版按两行算 ✓）；
      · **第 2 行** = `#14  m0.89`（**是谁** ＋ **这格观测多可信** ✓；`m` 没给/为负 ⇒ 只写 `#14` ✓）。

    ⚠ 原来是一行 `shape 0.98 #14  m0.89` ✗ —— **又宽又挡**（横着盖掉半个框 ✓）。
    ⚠⚠ **抽成纯函数** ✗ 的理由（不是洁癖 ✓）：离屏自检只能从像素上看出"**几行**/**多宽**" ✗，
      看不出"**哪一行是哪一段**"（两串本来就差不多宽 ⇒ 判不出顺序 ✗）⇒ 顺序口径**只能**由这里钉 ✓
      （见 `selftest_lie_demo.test_box_label_two_lines` ✓）；而 `draw()` 用的**就是这个函数** ✓
      ⇒ 自检测的和画面上画的**同一个来源** ✓✓（"所见即所得"那条纪律 ✓）。
    """
    #   ⭐⭐⭐⭐⭐ **2026-10-07：`shape` 这个词拿掉了** ✗✗（用户同日两条 ✓：先"**替换检出框左上角的
    #     「shape xxx」**" ✓、再"**标签还是在框心挡视野**"＋示意图 ✓）——
    #     ⚠⚠ **病根**：**类别名恒为 `shape`** ✗（模型只有一个类 ✓ 实测 `DetsWorker.names`
    #       = `['shape']` ✓），而且这个参数**本来就是置信度** ✗（调用处传的是 `_it[5]` = conf ✓）
    #       ⇒ 写 `shape 0.98` = **把 conf 换个名字重说一遍** ✗ 纯冗余 ✓。
    #   ⇒ 第 1 行**只留那个数** ✓（`0.98` ✓ —— 与"没配上框"那颗小页签（`0.52` ✓ 见
    #     `box_corner_label` ✓）**同一个口径** ✓）；第 2 行照旧 `#14  m0.89` ✓（一字不动 ✓）。
    #   ⚠ 拿不到 ⇒ 写 `-` ✗ 不许省掉那一行 ✓（画的时候按"两行"排版 ✓ 少一行会错位 ✓）。
    _l1 = ("%.2f" % float(shape)) if shape is not None else "-"
    _l2 = "#%d" % int(tid)
    # ⭐ **匹配分一起写**（用户 2026-10-04 ✓ 原话："**你可以把匹配分写出来**" ✓）——
    #   `m` = match（这一格的观测**有多可信** ✓ 见 `lie_motion` 里那段定义 ✓）：
    #   接近 1 ⇒ 可信 ✓；**骤降** ⇒ **这一拍该跟着 KF 走** ✓（= 他 ④ 那条 ✓）；
    #   ⚠ 粘连时它会掉到很低 ✓（面积因子被压低 ✓）—— 正是"别信这一格"的提示 ✓。
    if m is not None and float(m) >= 0.0:
        _l2 += "  m%.2f" % float(m)
    return _l1, _l2


def track_label_anchor(_t, pos, scale_x, scale_y):
    """⭐⭐⭐⭐⭐ **"这一拍没框的那条账，标签钉在哪"**（用户 2026-10-06 ✓ 原话："**为什么 #8 推的标在
    这里？这帧附近所有的座位都已经被占了，#8 推的应该标在圆心**" ✓✓）。

    ⚠⚠⚠ **踩过的坑** ✗✗（**这就是把它抽出来的理由** ✓）：我第一版把"座位推的"锚点写成
      `motion["pos"]` ✗ ⇒ 而那个键**实测是 `None`** ✗（`10月1日` 帧 62~64 全空 ✓）⇒ 这段
      **从来没生效过** ✓，标签一直挂在**轨迹自己的推算位置** ✗ —— 而 `draw()` 手里本来就有
      **圆心**（画绿圈那个 `pos` ✓）⇒ 用它才叫"所见即所得" ✓。
    ⚠ 规则（**只对"座位那条"生效** ✗）：
      · **座位 + 这一拍是"推的"**（`sel` ✓ 且 `live == False` ✓）⇒ 钉在**圆心** ✓；
      · **其余**（别人的账 ✓ 或座位有框 ✓）⇒ 照旧钉在**它自己的推算位置 `p`** ✓（老行为不变 ✓）。
    """
    if bool(_t.get("sel")) and not _t.get("live") and pos is not None:
        return int(float(pos[0]) * scale_x), int(float(pos[1]) * scale_y)
    return int(float(_t["p"][0]) * scale_x), int(float(_t["p"][1]) * scale_y)


#: ⭐⭐⭐⭐⭐ **画面字号的"全局档位"**（用户 2026-10-07 ✓ 原话："**标签太挡视线了，把他们字体
#:   缩小一号**" ✓✓）—— `draw()` 里**每一处** cv2 字号都过 `_fs()` ✓ ⇒ 以后要再缩一号，
#:   只改这**一个数** ✓（⚠ 别再一处一处改 ✗ 一定会漏 ✓ —— 本轮 21 处就是一次机械替换包上的 ✓）。
#:   ⚠ 为什么取 **0.9** = "一号" ✗：本仓库既有字号就 0.45 / 0.6 / 0.63 / 0.68 / 0.75 / 1.0 这几档 ✓
#:     ⇒ 各乘 0.9 ⇒ 0.41 / 0.54 / 0.57 / 0.61 / 0.68 / 0.90 ✓（**正好各降一档** ✓）。
#:   ⚠ `0.75 → 0.68` 这档要注意 ✗：0.68 在别处已经有人用 ✓ ⇒ 看着"没变"是错觉 ✓（值确实降了 ✓）。
_FONT = 0.9


def _fs(base):
    """**画面上所有 cv2 字号的唯一出口** ✓（见 `_FONT` ✓）。

    ⚠ 只缩放**字号** ✗：`thick`（线宽）**不跟着缩** ✓ —— 缩了会糊成一团 ✗（字号小、线还粗 ⇒
      笔画粘死 ✓）。`getTextSize` 与 `putText` **必须都过这里** ✓（⚠ 老代码有一处量 0.68、
      画 0.45 ✗ ⇒ 底衬比字大一圈 ✓ 那本身就是"挡视线"的元凶之一 ✓）。
    """
    return float(base) * float(_FONT)


def box_corner_label(bid, conf, cls=None, names=None, use_id=False):
    """**检出框左上角那串字**（用户 2026-10-07 ✓ 原话："**替换检出框左上角的「shape xxx」**" ✓✓）。

      · `use_id=False`（**经典模式** ✓）⇒ **老文案一字不变** ✓：`类别名 置信度`（`shape 0.52` ✓
        —— 拿不到 `names` 就写类别号 ✓）；
      · `use_id=True`（**运动模式** ✓ 有 `box_v` ✓）⇒ **`#编号 置信度`** ✓（`#22 0.52` ✓）；
        ⚠ `bid is None`（那格框**没归属** ✓）⇒ **只写置信度** ✓（`0.52` ✓ —— **不许编一个号** ✗）。

    ⚠⚠ **为什么非改不可** ✗（**实测** ✓）：模型**只有一个类** ✓ 名字就叫 **`shape`** ✓
      （`DetsWorker.names` = `['shape']` ✓）⇒ 每格框都顶着**同一串零信息**的字 ✗，而且原来
      **底衬按 0.68 量、字按 0.45 画** ✗ ⇒ **衬比字大一圈** ✓（"挡视线"的元凶之一 ✓）。
    ⚠⚠ **抽成纯函数** ✗ 的理由（与 `box_label_lines` 同一个理由 ✓）：离屏自检只能从像素上看出
      "**有没有那串字**" ✗，看不出"**是哪一串**"⇒ 文案口径**只能**由这里钉 ✓
      （见 `selftest_lie_demo.test_box_corner_label_replaced` ✓）；而 `draw()` 用的**就是这个
      函数** ✓ ⇒ 自检测的与画面上画的**同一个来源** ✓✓（"所见即所得"那条纪律 ✓）。
    ⚠ 只出 **ASCII** ✗（`cv2.putText` 画不出中文 ⇒ 会变 `?` ✓ —— 既有约定 ✓）。
    """
    if not use_id:
        _nm = (names[cls] if (names and cls is not None
                              and 0 <= int(cls) < len(names)) else str(cls))
        return "%s %.2f" % (_nm, float(conf))
    return (("#%d %.2f" % (int(bid), float(conf))) if bid is not None
            else ("%.2f" % float(conf)))


def box_tab_pos(cx, cy, w, h, scale_x, scale_y, s1, s2, frame_wh=None, gap=3):
    """⭐⭐⭐⭐⭐ **"两行标签该摆在哪" = 框的左上角、框上沿之外（像个页签）** ✗✗（用户 2026-10-07 ✓
    原话："**标签还是在框心挡视野，看我的示意图**" ✓✓ —— 示意图里就是：一小块「标签」贴在检出框
    左上角**外面** ✓、框本体叫「检出框」✓）。

    ⚠⚠ **改前为什么挡视野** ✗：整块**以框心为中心**上下对称画 ✗（`_y1 = _ty - _hh//2 + …` ✓）
      ⇒ 标签**正好压住框心** ✓ —— 而框心恰恰是**我们在看的地方** ✗✗（绿圈圆心、融合框的推导
      位置、真目标本体都在那一带 ✓）⇒ 用户每帧都被它遮一下 ✓。
    ⇒ 现在：**锚点 = 框的左上角** ✓、**左对齐** ✓、**整块在框上沿之外** ✓ ⇒ 框内**一个字不落** ✓；
      ⚠ 锚点不是框（`w/h <= 0` ✓ = 这拍没框、标签挂的是**推导位置** ✓）⇒ 退回"**以那个点为左下**"
      往上摆 ✓（不猜框 ✗）。

    参数：`(cx, cy, w, h)` = **框心 ＋ 框尺寸**（全分辨率 ✓ 没有框就传 `w=h=0` ✓）；
      `s1 / s2` = 两行各自的 `cv2.getTextSize` 结果 ✓（`(宽, 高)` ✓ —— 口径由调用方量 ✓
      免得这里再猜一次字号 ✗）；`frame_wh` = 画面尺寸 ✓（给了就**横向夹住** ✗ 别画出画面 ✓）。
    返回 `(x, y1, y2)`：`x` = 两行**左对齐**的起点 ✓；`y1 / y2` = 两行**基线** ✓（`y2` 在下 ✓）。
    ⚠⚠ **抽成纯函数** ✗ 的理由（与 `box_label_lines` / `box_corner_label` 同一个理由 ✓）：离屏自检
      只能看出"**有黄字**"✗，看不出"**它有没有压在框心上**"✓（那要靠**几何**判 ✓）⇒ 口径只能由
      这里钉 ✓（见 `selftest_lie_demo.test_box_tab_pos_above_box_corner` ✓）；而 `draw()` 用的
      **就是这个函数** ✓ ⇒ 自检测的与画面上摆的**同一个来源** ✓✓（"所见即所得"那条纪律 ✓）。
    """
    _w = float(w or 0.0)
    _h = float(h or 0.0)
    _x2 = float(cx)
    _y_top = float(cy)
    if _w > 0.0 and _h > 0.0:                  # 有框 ⇒ **框左上角** ✓（左沿 ＋ 上沿 ✓）
        _x2 = float(cx) - _w / 2.0
        _y_top = float(cy) - _h / 2.0
    _x = int(round(_x2 * float(scale_x)))
    _y = int(round(_y_top * float(scale_y)))   # 框上沿 ✓（页签整块压在它**上面** ✓）
    _h1 = int(s1[1]) if s1 is not None else 0
    _h2 = int(s2[1]) if s2 is not None else 0
    _wm = max([int(s1[0]) if s1 is not None else 0,
               int(s2[0]) if s2 is not None else 0, 0])
    _y2 = _y - 3                               # 第 2 行基线（贴框上沿外侧 ✓）
    _y1 = _y2 - _h2 - int(gap)                 # 第 1 行基线 ✓（⚠ 顺序：1 在上 ✓）
    _drop = max(0, 2 - (_y1 - _h1))            # ⚠ 顶到画面上边 ⇒ **整体下移** ✓（别出画 ✗）
    _y1 += _drop
    _y2 += _drop
    if frame_wh is not None:                   # ⚠ 右/左也别出画 ✓
        _x = max(2, min(_x, int(frame_wh[0]) - _wm - 3))
    return _x, _y1, _y2


def draw(frame, res, scale_x, scale_y, hud=None, cursor=None, motion=None,
         names=None, mask=0.0, pick=None, kf=None, kf_trail=None):
    """🔴 红点（鼠标）/ 🟢 绿圈（目标）/ 框（类别名+置信度）/ 运动向量 ✓。

    `pick` = **鼠标点选的那一格检出框**（用户 2026-10-03 ✓ 原话："你能让我在**点选检出框**
    的时候显示 `(360.4,323.0) 179×208=37232` 这种信息吗" ✓）⇒ 画**黄框高亮 + 数字标签**
    ✓ —— 纯显示层：不动任何判定、不改帧数据 ✓（与红框/绿圈/登记同一把尺 = **加工域坐标** ✓）。
    传 `(cls, cx, cy, w, h[, conf])`（与检出框同布局 ✓）；`None` ⇒ 不画 ✓。
    """
    _out, pos, r, hit, d = res
    # ⭐⭐⭐⭐⭐ **模式闸**（用户 2026-10-07 ✓「速度跟踪」那一轮）—— ⚠ 口径 = **后端给的那个键**
    #   （`motion["mode"]` ✓ **不是**界面下拉 ✓ —— 两者在"点应用 ⇒ 重建 runner"之后一致 ✓）。
    #   · `"motion"` ⇒ **运动分离**：下面所有老分支**一个字不改** ✓（零污染 ✓）；
    #   · `_vel`（= `"velocity"`）⇒ **做减法**：整幅画面**只留那 7 样**（见 `_LEGEND_VELOCITY` ✓）
    #     ⇒ 每一处"多出来的东西"都由这一道闸挡掉 ✓（挡法一律 `and not _vel` ✓ 一句话 ✓）。
    #   ⚠⚠ **项目纪律**（用户 2026-10-07 ✓ 原话）："以后**不得私自新增**任何观测参数 / 图例条目 /
    #     面板文字 / 底栏提示。要新增**必须先问用户、得到同意**才做" ✓ ⇒ 所以速度跟踪档一律是
    #     **"清空"** ✗ 不是"换一套速度跟踪专用的新文字" ✓（要加先问 ✓）。
    _mo_now = str((motion or {}).get("mode") or "")
    _vel = (_mo_now == "velocity")
    # ⭐⭐⭐⭐⭐ **限定区域（ROI）画出来** ✗✗（用户 2026-10-07 ✓ 原话："**只在限定的区域计算（因为
    #   这是部分弹窗）**" ✓✓）—— ⚠⚠ **所见即所得** ✓：画面上这个框**就是判据用的那个框** ✓
    #   （框里的检出框 / 白块 / 轨迹才算 ✓ 框外一概看不见 ✓）。
    #   ⚠ **从 `motion` 里读** ✗（后端 `process` 把它放在返回值里了 ✓ 见 `lie_motion` ✓）
    #     ⇒ 所有 `draw()` 调用点**一处都不用改** ✓；经典模式没有这个键 ⇒ 不画 ✓（老画面不变 ✓）。
    #   ⚠ 标签只写 **ASCII** ✗（`cv2.putText` 画不出中文 ⇒ 会变 `?` ✓ —— 这是既有约定 ✓）。
    _roi = (motion or {}).get("roi")
    if _roi is not None:
        _rx0 = int(round(float(_roi[0]) * scale_x))
        _ry0 = int(round(float(_roi[1]) * scale_y))
        _rx1 = int(round(float(_roi[2]) * scale_x))
        _ry1 = int(round(float(_roi[3]) * scale_y))
        cv2.rectangle(_out, (_rx0, _ry0), (_rx1, _ry1), (0, 200, 255), 2)
        cv2.putText(_out, "ROI", (_rx0 + 6, max(18, _ry0 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, _fs(0.6), (0, 200, 255), 2, cv2.LINE_AA)
    # ⭐⭐ **绿圈半径 = 目标框半宽**（用户 2026-09-30 口径 ✓ 原话："将绿圈半径恒=目标框半宽" ✓）：
    #   红框（`_choose_target_box` ✓）与绿圈**同一个目标** ⇒ 半径取那格框的**半宽 × 显示缩放** ✓。
    #   ⚠ 没有目标框时（建档/丢帧）+ 没有框 ⇒ 退回原来的"面积等效半径" ✓（不留空 ✗）。
    _roles = list((motion or {}).get("box_roles") or [])
    # ⭐⭐⭐ **红框几何 = 后端给的 `tbox` 本身**（用户 2026-09-30："红框必须与后端完全一致" ✓✓）：
    #   既不"按索引找差不多的一格" ✗（会差一格 ✗ 实测第 19 帧就有 ✗），也不在前端另判一次 ✗
    #   （会带进拖动方向这种 UI 状态 ✗✗）。后端没给（建档 / 丢帧）⇒ **不画红框** ✓（如实呈现 ✓）。
    _tb = (motion or {}).get("tbox")
    _tgt_i = (motion or {}).get("red_i")
    if _tgt_i is not None and d and _tgt_i < len(d):
        _rad_disp = max(6.0, float(d[_tgt_i][3]) * scale_x / 2.0)     # 半宽（显示域 ✓）
    else:
        _rad_disp = max(6.0, float(r) * (scale_x + scale_y) / 2.0)    # 兜底：面积等效半径 ✓
    # ⭐⭐⭐ **优先用"学到的目标实际半径"**（用户 2026-09-30 ✓）：它在白块阶段跟着框半宽学 ✓，
    #   一进入融合/分离就**冻结** ⇒ 目标透明后绿圈半径**恒定** ✓ —— 它代表的是**目标本身的
    #   大小**，不是当前那格框的大小 ✓（融合框会呼吸 ✗ 拿它画圈会忽大忽小 ✗）。
    _rt = (motion or {}).get("tgt_radius")
    if _rt:
        _rad_disp = max(6.0, float(_rt) * scale_x)
    # 用户 2026-10-01：圆圈半径 = 真目标检出框(tbox)的半宽（覆盖上面所有来源）。
    # 用户 2026-10-01：半径 = 红框【内切圆】半径（min(宽,高)/2），丢框后**冻结**
    #（红框消失 ⇒ 保持最后一次的尺寸，不退回学到的半径/面积等效半径）。
    _tbr = (motion or {}).get("tbox_rad")
    if _tbr:
        _rad_disp = max(6.0, float(_tbr) * (scale_x + scale_y) / 2.0)
    # ---- ⭐ 先铺蒙版（挖洞），再画标记（标记不被压暗 ✓）----
    #   ⭐⭐ **2026-10-02 口径变更**（用户 ✓ 原话："你可以让非检出框都受灰蒙版影响吗？目前绿色
    #     圆圈不是检出框也不受灰蒙版影响" + "**所有的线条都保持鲜亮，只有镂空区域灰**" ✓）：
    #     · **镂空（挖洞）只留给"检出框"** ✓ —— **不再给目标圆挖洞** ✗（原来圆那块方形也保持
    #       原亮度 ⇒ 圆看着像"飘在灰蒙版上面" ✗）⇒ 现在目标圆所在区域**跟背景一起被压暗** ✓；
    #     · **线条一律照旧鲜亮** ✓（它们都是这一步**之后**画的 ✓ 不被压暗 ✓）：绿圈线 / 红框线 /
    #       登记青框 / 各框运动箭头 / 白线 / 黄白箭头 / 光标 / 左上角 HUD 全部保持 ✓。
    #   ♻ **要回退**成"圆也挖洞"：把下面 `target=...` 那两行加回来即可 ✓（`apply_mask` 的参数
    #     与逻辑都**原样保留** ✓ 没删 ✗）。
    # ⭐⭐⭐⭐⭐ **ROI 之外：模糊 ＋ 置灰**（用户 2026-10-07 ✓ 原话："**ROI外的区域置灰+模糊**" ✓）——
    #   ⚠ 放在**一切绘制之前** ✓（这样框 / 圆 / 箭头照旧鲜亮 ✓ 只有"框外那块底"被压掉 ✓）；
    #   ⚠ `roi` 为空 ⇒ `dim_outside_roi` **一个像素都不碰** ✓（= 老行为 ✓ 零污染 ✓）；
    #   ⚠ 与"灰蒙版"（`apply_mask` ✓ 那个挖洞是按**检出框**挖的 ✓）是**两件事** ✓ 各管各的 ✓。
    #   ⚠⚠⚠ **`scale` 必须传** ✗✗（用户 2026-10-07 ✓ 报："模糊的区域和框选的区域不一致" ✓）：
    #     `frame` 在这条链上是**原图域**（1920×1080 ✓ 与 `apply_mask` 同一个约定 ✓），而 `roi`
    #     是**加工域**（889×500 ✓）⇒ 不乘就是"把小矩形当大图上的矩形" ⇒ **整块弹窗被糊掉** ✗
    #     （实测 673,823 px ✗）＋ 框外漏糊 107,959 px ✗。
    frame = dim_outside_roi(frame, (motion or {}).get("roi"),
                            scale=(scale_x, scale_y))
    frame = apply_mask(frame, d or [], mask, scale=(scale_x, scale_y))
    if pos is not None:
        c = (int(pos[0] * scale_x), int(pos[1] * scale_y))
        # ⭐⭐⭐ **圆圈恒色**（用户 2026-10-01 定稿 ✓ 原话："圆圈不要有任何的变色与相关逻辑" ✓）
        cv2.circle(frame, c, int(_rad_disp), (0, 255, 0), 2)
        # 用户 2026-10-01 定稿：淡粉框 = "红框丢失后的预测接力"——
        #   · 红框**在**（后端 `tbox` 有 ✓）⇒ **不画**（检出范围由红框自己表达 ✓，画了反而
        #     看不出"什么时候丢了" ✗）；
        #   · 红框**丢**（`tbox=None` ✓）且有宽高快照 ⇒ 画：尺寸 = 最后一次红框宽高的
        #     **原始值**（不缩到圆内 ✗ 它代表的是检出范围本身 ✓）、中心跟着**报告位置**走
        #     （丢框期间位置本来就是预测外推 ✓）⇒ 一眼看出"检出丢了、这段是预测在接力"✓。
        _tw = (motion or {}).get("tbox_wh")
        # ⚠ 速度跟踪档：粉框（= "检出框丢失后的预测接力框" ✓）**不在那 7 条里** ⇒ 不画 ✓。
        if (not _vel) and _tb is None and _tw and float(_tw[0]) > 1.0 and float(_tw[1]) > 1.0:
            _pw = float(_tw[0]) / 2.0 * scale_x
            _ph = float(_tw[1]) / 2.0 * scale_y
            cv2.rectangle(frame, (int(c[0] - _pw), int(c[1] - _ph)),
                          (int(c[0] + _pw), int(c[1] + _ph)),
                          (203, 192, 255), 2, cv2.LINE_AA)
    # ⭐⭐⭐ **速度跟踪：鼠标 = 一个「红点」**（用户 2026-10-07 ✓ 图例第 4 条"红点：此时鼠标
    #   位置" ✓）—— ⚠ **只画这一个点** ✗：白线（鼠标→圆心）、白圈套光标、以及圆心那个
    #   "白圈 + 红/绿实心点"**都不在那 7 条里** ⇒ 该档一律不画 ✓（见下面那道 `and not _vel` ✓）。
    if cursor is not None and _vel:
        cv2.circle(frame, (int(cursor[0] * scale_x), int(cursor[1] * scale_y)),
                   4, (0, 0, 255), -1, cv2.LINE_AA)
    if cursor is not None and not _vel:
        cur = (int(cursor[0] * scale_x), int(cursor[1] * scale_y))
        if pos is not None:
            cv2.line(frame, cur, (int(pos[0] * scale_x), int(pos[1] * scale_y)),
                     (255, 255, 255), 1, cv2.LINE_AA)
            # ⭐⭐⭐ **圆心永远标注**（用户 2026-10-01 定稿 ✓ 原话："他的圆心永远都需要标出来，
            #   如果模拟鼠标脱离了圆圈，那么圆心就是**红色的套着白圈的实心小圆点**" ✓）：
            #   白圈套实心点 ✓；模拟鼠标【脱离圆圈】⇒ 圆心【红】✓，在圈内 ⇒【绿】✓。
            _inside = ((cur[0] - c[0]) ** 2 + (cur[1] - c[1]) ** 2
                       <= int(_rad_disp) ** 2)
            cv2.circle(frame, c, 5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, c, 4, (0, 0, 255) if not _inside else (0, 255, 0),
                       -1, cv2.LINE_AA)
        # ⭐⭐⭐ **有且仅有一个"套着白圈的实心小圆点" = 追踪器（模拟鼠标）** ✓（用户定稿 ✓）：
        #    恒定样式（白圈 + 白心 ✓），不随命中变色 ✗ —— 变色只属于圆心标记 ✓。
        cv2.circle(frame, cur, 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, cur, 5, (0, 255, 0), -1, cv2.LINE_AA)
    # ---- 检测框 + **类别名 + 置信度**（用户 2026-09-30 要求 ② ✓）----
    #   ⭐⭐⭐ **红框 = 后端那个框**（用户 2026-09-30："红框必须与后端完全一致" ✓✓）：
    #      `_tb`（= `motion["tbox"]` ✓）就是**后端认定的目标框几何**（全分辨率 ✓）⇒ 直接按它
    #      画 ✓ —— **不再**"按索引去检测框里找差不多的一格" ✗（会差一格 ✗ 实测第 19 帧就是 ✗），
    #      也不再让前端自己判 ✗（那会带进拖动方向这种 UI 状态 ✗✗）。
    #      后端没给（建档 / 丢帧）⇒ **不画红框** ✓（如实呈现"后端还没有答案" ✓）。
    #   ⚠ 若某格检测框恰好就是后端那格 ⇒ **跳过它**（否则会被画成细蓝框、压在红框上 ✗）。
    _tb_key = (None if not _tb else (round(float(_tb[0]), 1), round(float(_tb[1]), 1),
                                     round(float(_tb[2]), 1), round(float(_tb[3]), 1)))
    for _i, b in enumerate(d or []):
        cls = int(b[0])
        cx, cy, w, h = b[1], b[2], b[3], b[4]
        conf = float(b[5]) if len(b) > 5 else 0.0
        _key = (round(float(cx), 1), round(float(cy), 1),
                round(float(w), 1), round(float(h), 1))
        # ⚠ 速度跟踪档**没有那个唯一红框**（它不在 7 条里 ✓）⇒ 后端那格**照画蓝框** ✗ ——
        #   不然它会一个框都没有 ✓（"真目标自己那格"也是 **yolo 检出框** ✓ 图例第 1 条 ✓）。
        if (not _vel) and _tb_key is not None and _key == _tb_key:
            continue                               # 后端那格 ⇒ 交给红框画 ✓（别重复 ✗）
        p1 = (int((cx - w / 2) * scale_x), int((cy - h / 2) * scale_y))
        p2 = (int((cx + w / 2) * scale_x), int((cy + h / 2) * scale_y))
        _role = _roles[_i] if _i < len(_roles) else ""
        _col, _th = (255, 140, 0), 1
        if _role == "suspect":
            _col, _th = (0, 0, 255), 1             # 严查框（细红 ✓ 不当"唯一红框"抢戏 ✗）
        # ⭐⭐⭐⭐ **运动分离模式：按「框状态」上色**（用户 2026-10-04 ✓ 原话："**检出框是否应该
        #   有状态？现在只有无归属、归属某目标 —— 真假目标重叠的框呢？**" ✓✓）——
        #   ⚠ 他的洞察：**「归属」和「可信」是两个正交维度** ✗；而"归属"已经由**框上的编号**
        #     表达了（`shape 0.98 #22` ✓）⇒ 颜色这一路**专门表达"可信"** ✓ 两者不重复 ✓。
        #   状态码来自后端 `motion["box_v"][7]`（`0` 正常 ｜ `1` **融合** ｜ `2` **可疑**
        #   ｜ `3` **残缺** ｜ **`4` 冷却** ✓ 见 `lie_motion` 那段 ✓）—— **按位置反查**（与点选同一把尺 ✓）。
        #   ⚠⚠ **`4`（冷却）故意不特判** ✗✗（用户 2026-10-05 ✓ "**拆状态码**" ✓）：冷却期那格框
        #     **已经分开**、就是**真目标自己那格** ✓ ⇒ 画成**正常框那种颜色**才对 ✓
        #     （落进下面的默认色 ✓）；"冷却中"由**面板文案**说 ✓（见下面那段 ✓）。
        #   ⚠ **只在运动分离模式下改色** ✗ ⇒ **经典模式一个像素都不变** ✓（零污染 ✓）。
        # ⭐⭐⭐⭐⭐ **速度跟踪：融合框 = 洋红**（用户 2026-10-07 ✓ 原话："**融合用洋红**" ✓）——
        #   ⚠ 这一档**只有"融合"变颜色** ✗：上板 / 等待上板**都是蓝框** ✓（两者靠"**框左上有
        #     没有编号标签**"区分 ✓ 正是用户那句"**有标签就是上板，无标签就是等待上板**" ✓）
        #     ⇒ 不再按"可疑 / 残缺 / 冷却"上色 ✗（那几档是运动分离那套状态码 ✓ 见下面那个 `elif` ✓）。
        #   ⚠ 口径 = 后端 `motion["vel"]["state"]`（**逐位对齐 `box_v`** ✓ 见 `VelocityChassis` ✓）
        #     ⇒ 按位置反查（与点选、与下面"框标签"**同一把尺** ✓ 1.5px ✓）。
        if _vel:
            _bv9 = list((motion or {}).get("box_v") or [])
            _st9 = list(((motion or {}).get("vel") or {}).get("state") or [])
            for _k9, _it in enumerate(_bv9):
                if (len(_it) > 1 and abs(float(_it[0]) - float(cx)) < 1.5
                        and abs(float(_it[1]) - float(cy)) < 1.5):
                    if _k9 < len(_st9) and str(_st9[_k9]) == _VEL_MERGE:
                        _col, _th = (255, 0, 255), 2
                    break
        elif (motion or {}).get("mode") == "motion":
            for _it in (motion.get("box_v") or []):
                if (len(_it) > 7 and abs(float(_it[0]) - float(cx)) < 1.5
                        and abs(float(_it[1]) - float(cy)) < 1.5):
                    # 🟣 洋红 = **融合**（框心 = 两个目标的**中点** ✗ ⇒ 位置**已经不采信**了 ✓）
                    # 🟠 橙黄 = **可疑**（匹配分低 ⇒ 已经**降权 / 断掉** ✓）
                    # ⚫ 灰   = **残缺**（框只剩一小块 ⇒ 框心是**碎片重心** ✗）
                    _b7 = int(_it[7])
                    if _b7 == 1:
                        # ⭐⭐⭐⭐⭐ **融合还分两种**（用户 2026-10-04 ✓ 原话："**不跟真目标圆相交
                        #   的检出框肯定不是融合紫框**" ✓✓）—— 第 9 位 = `has_tgt` ✓（后端算的 ✓
                        #   见 `_Track.has_tgt` ✓）：
                        #     · **含真目标** ⇒ 洋红 ✓（这才是"真目标被融合了"✓ 唯一能反推真目标 ✓）；
                        #     · **两个别的目标撞在一起** ⇒ **暗紫** ✗（实测占那 42 个紫框的
                        #       **23.8%** ✓ 离真目标圆边缘中位 **42px** ✗ ⇒ 与真目标无关 ✓）。
                        _has = bool(_it[8]) if len(_it) > 8 else True
                        if _has:
                            _col, _th = (255, 0, 255), 2
                        else:
                            _col, _th = (150, 30, 110), 1
                    elif _b7 == 2:
                        _col, _th = (0, 165, 255), 2
                    elif _b7 == 3:
                        _col, _th = (120, 120, 120), 1
                    break
        cv2.rectangle(frame, p1, p2, _col, _th)
        #   ⭐⭐⭐⭐⭐ **2026-10-07：检出框左上角那串换掉了** ✗✗（用户原话："**标签太挡视线了，
        #     把他们字体缩小一号并替换检出框左上角的「shape xxx」**" ✓✓）——
        #     ⚠⚠ **病根两条** ✗：① 这里原来写的是 `"%s %.2f" % (nm, conf)` ✓，而 **`nm` 恒为
        #       `shape`** ✗（**实测** ✓ 模型只有一个类 ✓ `DetsWorker.names` = `['shape']` ✓）
        #       ⇒ 每格框都顶着同一串**零信息**的字 ✗；② 底衬按 **0.68** 量、字按 **0.45** 画 ✗
        #       ⇒ **衬比字大一圈** ✓（"挡视线"的元凶之一 ✓）。
        #   ⇒ 换成 **`#编号 置信度`**（如 `#22 0.52` ✓）—— 编号**按位置反查 `box_v`** ✓（与上面
        #     那段"按位置反查上色"、与点选**同一把尺** ✓；⚠ 这里的 `(cx, cy)` 就是 YOLO 那个
        #     框心 ✓ 见 `lie_motion` 里"坐标一律用框自己的中心"那段 ✓ ⇒ **天然对齐** ✓）；
        #     ⚠ **查不到 ⇒ 只写置信度** ✗（= 这格没归属 ✓ 不许编一个号出来 ✓）；
        #     ⚠ `box_v` **压根没有**（经典模式 ✓）⇒ **老文案一个字不变** ✓（零污染那条纪律 ✓）。
        _bv = (motion or {}).get("box_v") or []
        _bid = None
        if _bv:
            for _it in _bv:
                if (len(_it) > 4 and abs(float(_it[0]) - float(cx)) < 1.5
                        and abs(float(_it[1]) - float(cy)) < 1.5):
                    _bid = int(_it[4])
                    break
        #   ⭐⭐⭐⭐⭐ **2026-10-07：配上账的框，这颗小页签就不画了** ✗✗（用户示意图 ✓）——
        #     ⚠⚠ **两处都在"框左上角、上沿之外"** ✗（上面那颗两行的 `box_tab_pos` ✓ 与这里同名
        #       同角 ✓）⇒ 一起画就**叠成一坨** ✗✗；而且内容是重复的 ✓（小页签 `#22 0.96` ✗
        #       vs 两行页签第 1 行 `0.96` ＋ 第 2 行 `#22  m0.00` ✓ ⇒ **同两段信息** ✓）。
        #     ⇒ **配上账的交给两行页签** ✓（它带 `m` ✓ 信息更全 ✓）；这里只留"**没归属**"的框 ✓
        #       （= 那些**只有置信度**可说的 ✓，`0.52` ✓ 见 `box_corner_label` ✓）。
        #     ⚠ 框本体**照画** ✓（上面那句 `cv2.rectangle` 已经执行过了 ✓ 只是跳过标签 ✓）。
        # ⚠ 速度跟踪档：**没编号的框不挂这句话** ✗（= "等待上板" ✓ 用户口径"**无标签就是等待
        #   上板**" ✓ —— 所以"**没有标签**"本身就是它的画面表达 ✓ 不能反而给它挂一行置信度 ✓）。
        if _bid is not None or _vel:
            continue
        txt = box_corner_label(_bid, conf, cls, names, bool(_bv))
        #   ⚠⚠ **量与画必须同一个字号** ✗✗（老代码 0.68/0.45 不一致 ✓）⇒ 这里**都是 0.45** ✓
        #     （= 取**小的那个** ✓）：底衬当场缩到贴字 ✓（比原来小一大圈 ✓），而字形本身
        #     再乘 `_FONT`(0.9) ⇒ 正好是用户要的"**缩小一号**" ✓。
        (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.45), 1)
        ty = max(p1[1] - 3, th_ + 4)
        cv2.rectangle(frame, (p1[0], ty - th_ - 4), (p1[0] + tw + 4, ty + 2),
                      (40, 90, 160), -1)
        cv2.putText(frame, txt, (p1[0] + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    _fs(0.45), (255, 255, 255), 1, cv2.LINE_AA)
    # ---- ⭐⭐⭐ **登记表标记**（用户 2026-09-30 两条 ✓）----
    #   ① "登记的框需要**与标注框重叠**，表示'**这个框是登记过的**'" ✓ ⇒ **不给登记表另画一个
    #      框** ✗（我上一版那样画，位置一飘就变成"另一个东西" ✗ 看不出登记对没对 ✗）；
    #      改成**给"已登记"的那格检出框打标** ✓：**青色粗框 + `登记 #012`** ✓ ⇒ 一眼就能数
    #      "有几格被登记了、哪格没被登记" ✓。
    #   ② "我们先完全信任 yolo" ✓ ⇒ 判"这格登记过没"就用**重叠**（登记条目的中心落在这格里 ✓）。
    #   ⭐⭐ 2026-10-01 用户定稿 ✓："**只要是登记过的，都不能丢**"（"把他绘制在**红框的下层**" ✓）
    #     ⇒ **登记过的一律画** ✓（画在红框之前 = 下层 ✓ 红框永远压在上面 ✓），不再按"是不是
    #     真目标"跳过 ✗ —— 因为判据已换成**物理的"钉死在背景板上"** ✓（真目标永远钉不住 ⇒
    #     **自然没有青标** ✓ 用户第 1 条：与真目标重合的那块砖**照样能登记** ✓）。
    #   ⭐⭐ 2026-09-30 用户澄清（一字不改 ✓）："我们登记的是『**稳定尺寸、速度相同的假目标**』，
    #     出现红框融合时，虽然检出框变大了，但**这个登记的框依旧还是那么大**，**就算检出框消失了
    #     它也存在并跟着群体一起运动**，相当于『**这里精确标记了一个假目标**』" ✓
    #     ⇒ 所以画的是**条目自己的框**（尺寸 = 种下时的 ✓ 不跟检出框变 ✗），而不是"给检出框描边" ✗
    #     —— 融合期你就能看到"一个固定大小的小青框**贴在巨大的融合框里**" ✓、
    #     检出框整格消失时它也**照样在** ✓（这正是它存在的意义 ✓）。
    for _e in ((motion or {}).get("reg") or []):
        _ex = float(_e.get("x", 0.0)) * scale_x
        _ey = float(_e.get("y", 0.0)) * scale_y
        _ew = max(8.0, float(_e.get("w", 0.0)) * scale_x / 2.0)
        _eh = max(8.0, float(_e.get("h", 0.0)) * scale_y / 2.0)
        _q1 = (int(_ex - _ew), int(_ey - _eh))
        _q2 = (int(_ex + _ew), int(_ey + _eh))
        # ⭐⭐⭐ **两层颜色**（用户 2026-10-02 ✓ 定稿 ✓ 原话："它**首次标青色时为候选框**，以后
        #   **换个颜色标**，等他满足『**达到标准假目标群面积及尺寸比例**』的条件后**转正**，
        #   **再标青色框**" ✓）：
        #   · **🟠 橙色细框 + `候选(x,y)`** = 还没转正（边缘刚进相机、还在"生长跟进" ✓ 或面积/
        #     比例没达标 ✓）—— 跟踪它 ✓ 但它还**不是**"已登记的假目标" ✗；
        #   · **🟦 青色粗框 + `砖(x,y)`** = **已转正**（面积进了标准带、比例达标 ✓ = 钉死的假砖 ✓）。
        _bid = _e.get("bid") or (0, 0)
        if bool(_e.get("pending")):
            # ⭐⭐ **形状待定**（用户 2026-10-02 ✓ 原话："把形状待定的用边缘进视野的候选框
            #   一样的颜色" ✓）：与绿圈相交中、尺寸被真目标影响 ✗ ⇒ 画**橙色候选框**同款 ✓
            #   （哪怕已上板 ✗ —— 待定期间它的青框尺寸不可信 ✗）。
            cv2.rectangle(frame, _q1, _q2, (0, 165, 255), 1)
            _txt = "待定(%d,%d)" % (int(_bid[0]), int(_bid[1]))
            _lab_col = (0, 165, 255)
        elif bool(_e.get("ok")):
            cv2.rectangle(frame, _q1, _q2, (255, 255, 0), 2)
            _txt = "砖(%d,%d)" % (int(_bid[0]), int(_bid[1]))
            _lab_col = (255, 255, 0)
        else:
            cv2.rectangle(frame, _q1, _q2, (0, 165, 255), 1)
            _txt = "候选(%d,%d)" % (int(_bid[0]), int(_bid[1]))
            _lab_col = (0, 165, 255)
        (tw, th2), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.63), 1)
        _ty = max(_q1[1] - 3, th2 + 4)
        cv2.rectangle(frame, (_q1[0], _ty - th2 - 4), (_q1[0] + tw + 4, _ty + 2),
                      (90, 60, 10), -1)
        cv2.putText(frame, _txt, (_q1[0] + 2, _ty - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    _fs(0.42), _lab_col, 1, cv2.LINE_AA)
    # ---- 🔴 **唯一红框 = 后端那个框**（画在检测框之上 ✓ 用户："必须与后端完全一致" ✓）----
    if _tb is not None:
        _p1 = (int((float(_tb[0]) - float(_tb[2]) / 2.0) * scale_x),
               int((float(_tb[1]) - float(_tb[3]) / 2.0) * scale_y))
        _p2 = (int((float(_tb[0]) + float(_tb[2]) / 2.0) * scale_x),
               int((float(_tb[1]) + float(_tb[3]) / 2.0) * scale_y))
        cv2.rectangle(frame, _p1, _p2, (0, 0, 255), 3)
        # ⭐⭐ **融合红框 ⇒ 右上角标"融合"**（用户 2026-10-01 ✓ 原话："如果红框处于融合状态，
        #   就标在框右上角吧" ✓）：一眼看出这格红框 = 真假目标的并集 ✓（与左上角的青框"登记"
        #   分开 ✓ 不打架 ✓）。⚠ 只认后端给的 `merged`（`motion["merged"]` ✓ 同一口径 ✓）。
        if (motion or {}).get("merged"):
            _txt = "融合"
            (tw, th2), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), 1)
            _bx0 = _p2[0] - tw - 4
            _by0 = _p1[1]
            _by1 = _by0 + th2 + 4
            cv2.rectangle(frame, (_bx0, _by0), (_p2[0], _by1), (0, 0, 190), -1)
            cv2.putText(frame, _txt, (_bx0 + 2, _by0 + th2 + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), (255, 255, 255), 1, cv2.LINE_AA)
    # ---- ⭐⭐ 每个框的**运动向量**（用户 2026-09-30 定口径 ✓）：**从本框上一帧的位置
    #   指向本帧的位置**（= 逐框位移 ✓ 方向就是真实运动方向 ✓）。
    #   🔵 蓝 = 一般框 ｜ 🟢 绿 = 追踪器认定"运动最异常"的那个 ｜ 🟡 黄 = 实际在跟的那个。
    #   ⭐⭐⭐ **长度改了**（用户 2026-09-30 第 A2 条 ✓ 原话："**取消自定义箭头长度拖动条
    #     配置，速度向量箭头现在强制指向以当前速度预计下一帧到达的位置**" ✓）：
    #     不再有"基准长度/百分比/上限"那套 ✗ —— 箭头长度**就是这一拍的位移**（× 显示缩放 ✓）
    #     ⇒ 箭尖 = **按当前速度预计下一帧到达的位置** ✓ 所见即所测 ✓。
    if motion:
        moves = motion.get("box_moves") or []
        roles = motion.get("box_roles") or []
        HEAD_PX = 10.0                    # 箭头头固定 ~10px ✓（与尾长无关 ✓）

        def _arrow(x, y, dx_disp, dy_disp, col, th, hollow=False):
            """从 `(x, y)`（**显示坐标**）沿**位移** `(dx_disp, dy_disp)`（**显示像素** ✓）
            画箭头；长度**就是位移本身** ✓（用户 A2 ✓ 箭尖 = 预计下一帧的位置 ✓）。"""
            sp = (dx_disp * dx_disp + dy_disp * dy_disp) ** 0.5
            if sp < 1e-6:
                cv2.circle(frame, (int(x), int(y)), 3, col, 1, cv2.LINE_AA)
                return
            ex, ey = x + dx_disp, y + dy_disp
            cv2.arrowedLine(frame, (int(x), int(y)), (int(ex), int(ey)),
                            col, th, cv2.LINE_AA,
                            tipLength=min(0.5, HEAD_PX / max(sp, HEAD_PX)))
            cv2.circle(frame, (int(x), int(y)), 6 if hollow else 2, col,
                       1 if hollow else -1, cv2.LINE_AA)

        _pred = motion.get("box_pred") or []
        # ⚠⚠ **速度跟踪档不画这一路箭头** ✗（用户 2026-10-07 ✓）：这一路画的是"**逐框位移**"
        #   的箭头（还带 🟡"实际在跟的那个" / ⚪"轨迹预测补的"两种颜色 ✓）⇒ **不在那 7 条里** ✗；
        #   该档的"**蓝箭头 = 检出框的绝对速度**"由**下面 `box_v` 那一段**画 ✓（口径**只有一条** ✓
        #   别画两份 ✗ —— 两份的锚点/长度口径还不一样 ✓）。
        for i, b in enumerate(([] if _vel else (d or []))):
            mv = moves[i] if i < len(moves) else None
            role = roles[i] if i < len(roles) else ""
            col, th = (255, 110, 0), 2              # 🔵 蓝：一般框（实测位移 ✓）
            if role == "followed":
                col, th = (0, 215, 255), 3          # 🟡 黄：实际在跟的那个 ✓
            if i < len(_pred) and _pred[i]:
                # ⚪ 灰：这一格是**轨迹预测**补的（上一帧配不上 ✓ 用户第②问 ✓）
                col, th = (150, 150, 150), 1
            x, y = b[1] * scale_x, b[2] * scale_y
            if mv is None or (mv[0] ** 2 + mv[1] ** 2) ** 0.5 < _MOVE_MIN_PX:
                # ⭐ 位移小于噪声阈值（**量化噪声 ~1px** ✓）⇒ **只画点**（画箭头 = 假方向 ✗）
                cv2.circle(frame, (int(x), int(y)), 3, col, 1, cv2.LINE_AA)
                continue
            _arrow(x, y, mv[0] * scale_x, mv[1] * scale_y, col, th)
        # ⭐⭐ **"粘滞"轨迹的箭头**（用户第④问 ✓）：这拍**没有对应检测框**的轨迹（滑行中 ✓）
        #   也把预测速度画出来（灰色 ✓ 与"白色=追踪器自己"分开 ✓）。
        _dt = float(motion.get("dt") or 0.0)
        # ⚠ 速度跟踪档：这一路（"**粘滞**轨迹的灰色箭头" ✓）也**不在那 7 条里** ⇒ 不画 ✓。
        for t in ((motion.get("tracks") or []) if not _vel else []):
            if not t.get("sticky"):
                continue
            vx, vy = t.get("v") or (0.0, 0.0)
            if (not (vx or vy)) or _dt <= 0:
                continue
            _x, _y = t["p"][0] * scale_x, t["p"][1] * scale_y
            _arrow(_x, _y, vx * _dt * scale_x, vy * _dt * scale_y, (150, 150, 150), 1)
        # ⚠ **白线（圆心画面路径）已移除**（用户 2026-10-04 ✓ 原话："黄箭头、白线…也不需要，
        #   移除" ✓）⇒ 原 `path_pts` 那条白折线 + 端点空心点不再画 ✓。
        # ⭐ **白箭头 = 圆圈相对假目标群的速度**（用户 2026-10-01 定稿 ✓）——
        #   ⚠ **黄箭头（绝对速度）已移除**（用户 2026-10-04 ✓ 原话："没有看到黄箭头…也不需要，
        #   移除" ✓）⇒ 顶层只留白箭头 ✓。
        #   · 起点 = **绿圈圆心**（= 报告位置 `pos` ✓ 与绿圈同一个点 ✓）；
        #   · 长度 = 速度 × dt（= 按当前速度**预计下一帧到哪** ✓ 与画面其它速度箭头同口径 ✓）。
        _dt_a = float(motion.get("dt") or 0.0)
        if pos is not None and _dt_a > 0:
            _x, _y = pos[0] * scale_x, pos[1] * scale_y
            # ⭐⭐⭐⭐⭐ **运动模式：白箭头 = 圆「本该走」的群体相对位移**（px/拍 ✓ 用户 2026-10-06 ✓
            #   原话："**它在框内可能不动，但是我们的相对群体速度（白箭头）要保持，是一种"怼着
            #   墙走"的感觉**" ✓✓）—— ⚠⚠ **口径改过一次**（**实测逼的** ✓）：2026-10-04 那版画的是
            #   "**实际**走过"的位移 ✓（= 橙线最后一段 ✓），可 10-06 给融合期加了"**整圆出框就
            #   拉回**"那道夹取 ✓ ⇒ 贴墙那几拍圆几乎不动 ⇒ 箭头**瘪成一条** ✗（用户当场问：
            #   "**为什么到 59 帧圆心的相对群体速度变小这么多？**" ✓）⇒ 现在改画**约束之前**那个
            #   位置算出来的位移 ✓（`MotionTracker._pos_intent` ✓ 见 `process` / `motion_viz` ✓）。
            #   ⚠ 没贴墙的拍两者**本来就相等** ✓（约束没动位置 ✓）⇒ 平时画面不变 ✓；
            #     ⚠ 橙线照旧是"**实际**走过的路" ✗ ⇒ 贴墙那几拍"**箭头不切橙线**"是**有意**的 ✓
            #       （那截正是被墙挡住的 ✓）。
            #   ⚠⚠ **不乘 `dt`** ✗✗（运动模式单位是 **px/拍** ✓ —— 乘 `dt = 1/60` ⇒ 箭头只剩
            #     **0.42px** ⇒ **根本看不见** ✓ 实测帧 20 ✓，与黄箭头当年同一个坑 ✓）。
            _vl = motion.get("tgt_rel_last")
            if _vl is not None:
                _arrow(_x, _y, _vl[0] * scale_x, _vl[1] * scale_y,
                       (255, 255, 255), 2)                # ⚪ 白：圆本拍**实际**相对位移 ✓
            else:
                # ⚠ 经典模式没这个键 ✗ ⇒ 走老口径（那边的 `vel_rel` 是 px/s ✓ 要乘 `dt` ✓）
                _vr = motion.get("vel_rel") or (0.0, 0.0)
                _arrow(_x, _y, _vr[0] * _dt_a * scale_x, _vr[1] * _dt_a * scale_y,
                       (255, 255, 255), 2)                # ⚪ 白：相对假目标群 ✓
        # ⭐⭐⭐ **黄箭头 = 真目标「下一拍的群体相对位移」**（用户 2026-10-04 ✓ 原话："在真目标圆心
        #   增加**黄箭头**，指向最终我们计算出的（**滑行＋框心指导**）真目标下一拍的**群体相对
        #   位移**" ✓）—— ⚠ 这个键**只有运动分离模式给**（`MotionRunner` ✓ 见 `lie_motion` ✓）；
        #   经典模式没有 ⇒ 这里不画 ✓（那边另有白箭头 ✓）。
        #   ⚠ 锚点与白箭头相同（绿圈圆心 = 报告位置 `pos` ✓）、同口径（`× dt` ✓）。
        _tn = motion.get("tgt_rel_next")
        # ⚠ 速度跟踪档：黄箭头（= "目标**下一拍**预计相对群体位移" ✓）**不在那 7 条里** ⇒ 不画 ✓
        #   （那 7 条里跟"群体"有关的只有 **白箭头**（本拍）与 **橙线**（历史）✓）。
        if (not _vel) and pos is not None and _tn is not None:
            #   ⚠⚠ **别乘 `dt`** ✗✗（**实测踩到** ✓）：运动模式的 `vel` 单位是 **px/拍** ✗，
            #     而 `dt = 1/60` ⇒ 长度塌成 **≈0.1px** ⇒ **根本画不出来** ✗✗ ——
            #     用户报的"**没有看到黄箭头**"就是这个 ✓（现在按 px/拍 直接画 ✓）。
            _arrow(pos[0] * scale_x, pos[1] * scale_y,
                   _tn[0] * scale_x, _tn[1] * scale_y,
                   (0, 215, 255), 2)                    # 🟡 黄：目标的群体相对位移 ✓
        # 图例（左下角）：候选数 / 中位速度 / 目标偏差 / 跟随目标速度（都是数字 ✓）
        mvx, mvy = motion.get("median", (0.0, 0.0))
        fv = motion.get("followed_v") or (0.0, 0.0)
        lines = ["candidates %d  |  median v = (%+.0f, %+.0f) px/s  (|v|=%.0f)"
                 % (motion.get("n_cands", 0), mvx, mvy,
                    (mvx ** 2 + mvy ** 2) ** 0.5),
                 "motion-target dev = %.0f px/s  |  followed v = (%+.0f, %+.0f)  |v|=%.0f"
                 % (motion.get("dev", 0.0), fv[0], fv[1],
                    (fv[0] ** 2 + fv[1] ** 2) ** 0.5),
                 # ⭐ **标记说明**（用户 2026-09-30 A1/A2/B1 ✓ + 2026-10-01 速度双箭头 ✓）：
                 #   箭头长度 = **这一拍走了多少像素**（箭尖 = 按当前速度预计下一帧的位置 ✓）；
                 #   🟡**黄箭头 = 圆圈（绿圈圆心）的绝对速度** ✓、
                 #   ⚪**白箭头 = 它相对"假目标群"的速度** ✓（两支之差 = 群体在怎么走 ✓）；
                 #   灰箭头 = 轨迹预测补的那格 ✓。
                 "arrows = measured move (tip = next frame if speed holds)"
                 "   BLUE   = circle's next-frame ABSOLUTE displacement   WHITE = RELATIVE to fake-group"
                 # ⭐ **标记说明**（用户第②问 ✓ 原话："绿色圈圆心的十字标是当前跟踪器
                 #   位置。那绿色点是什么？红色箭头是什么？" ✓）—— 逐条写明免得再猜 ✓：
                 #   十字+圆 = 追踪器**报出的目标位置**（绿=命中 / 黄=未命中 ✓）；
                 #   白圈+红/绿实心点 = **模拟鼠标**（光标 ✓ 不是目标 ✗）；
                 #   绿箭头/绿点 = **判据认定最"不合群"的那个检测框**（该跟的 ✓）；
                 #   白空心箭头 = **粘滞**（该框这拍没检出 ⇒ 按预测速度画 ✓ 第④问）；
                 #   红粗框 = 真目标框 / 红细框 = "严查"框（怀疑真目标在内 ✓ 第⑤问）。
                 "cross+ring = tracker report (green=hit)   white ring + dot = MOUSE"
                 "   red bold box = the ONE box believed to hold the real target",
                 # ⭐ **怀疑链**（用户 2026-09-30 ①②③ ✓ 见 `lie_tracker` 常量区注释）：
                 #   目标是哪条轨迹、它有没有被怀疑（merge=可能重叠 / split=可能分离且
                 #   旁边新出框=严查 / hidden=可能分离且没检出 ⇒ 真目标就在面积消失处 ✓）、
                 #   以及"不合群"已经持续了多少秒（判据④：满 0.6s 才敢换 ✓）。
                 _sus_line(motion)]
        # ⚠⚠ **行高必须跟着字号** ✗✗（用户 2026-10-04 ✓ 字号调大后暴露的 ✓）——
        #   原来行高**硬编码 16px** ✗ 而字高已经 ~20px ⇒ **行与行叠在一起** ✓。
        # ⚠⚠⚠ **运动分离模式下这一段一律不画** ✗✗（用户 2026-10-04 ✓ 原话："**左上角这些框里
        #   是什么，太挡视野了**" ✓）—— 它讲的是**经典模式**那套标记（cross/ring/mouse/红框 ✓），
        #   而且"候选数/中位速度/目标偏差"那几个数**上面那块面板已经全有** ✗ ⇒ 属于**纯冗余** ✓
        #   ⇒ 在 motion 模式下**直接省掉**（画面底部立刻空出来 ✓）；经典模式**一个字不动** ✓。
        # ⚠⚠ **运动分离 / 速度跟踪两档一律不画这一段** ✗✗（用户 2026-10-04 ✓ "左上角这些框里
        #   是什么，太挡视野了"；2026-10-07 ✓ 速度跟踪"画面上正好只有那 7 条图例、别无他物" ✓）
        #   —— 它讲的是**经典模式那套标记**（cross / ring / mouse / 红框 ✓）✓ 那两档都不画 ✓。
        if _mo_now not in ("motion", "velocity"):
            _lh2 = int(cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, _fs(0.68), 1)[0][1]) + 8
            y0 = frame.shape[0] - _lh2 * len(lines) - 8
            for i, s in enumerate(lines):
                cv2.putText(frame, s, (10, y0 + _lh2 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            _fs(0.68), (0, 0, 0), 3)
                cv2.putText(frame, s, (10, y0 + _lh2 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            _fs(0.68), (235, 235, 235), 1)
    # ⭐⭐⭐ **点选的检出框**（用户 2026-10-03 ✓ 原话："你能让我在**点选检出框**的时候显示
    #   `(360.4,323.0) 179×208=37232` 这种信息吗" ✓）：**黄框高亮 + "几何=面积"标签** ✓。
    #   ⚠ 画在**最后**（压在所有标注之上 ✓ —— 点选框是"我现在最关心的那一格" ✓）；
    #   ⚠ 纯显示 ✓ 一个判定都不动 ✓；数字口径 = **加工域**（与红框/登记/状态栏同一把尺 ✓）。
    # ⭐⭐⭐ **KF 影子可视化**（用户 2026-10-03 ✓ 原话："**你能把 KF 也做可视化给我检验效果吗？**" ✓）
    #   —— 只在"位置估计 = **KF 影子**"时才有东西 ✓ **纯显示层** ✗ 一个判定都不动 ✓：
    #   · 🟦 **青色细折线** = KF 位置的历史（与那根**白线**（经典位置历史）**并排** ⇒
    #     "**抖不抖**"一眼看出 ✓ —— 这正是用户要检验的那件事 ✓）；
    #   · 🟦 **青色实心点 + 空心圈** = KF 这一刻估的位置（与**绿圈圆心**对照 ⇒
    #     两者差多少 = "平滑换来的滞后" ✓ 差太大 ⇒ R 偏大（太信模型）⇒ 该调小 ✓）；
    #   · 🟦 **青色小箭头** = KF 估的速度（长度 = 速度 × dt ✓ 与白/黄箭头**同一口径** ✓）。
    #   ⚠ 颜色：青 = BGR `(255,255,0)` ✓（与"点选框黄 (0,255,255)"、"白线白 (255,255,255)"
    #     都区分得开 ✓）；⚠ 坐标是**加工域** ⇒ 画时乘 `scale` ✓（与红框/绿圈同一把尺 ✓）。
    if kf_trail and len(kf_trail) >= 2:
        _kt = [(int(round(float(p[0]) * scale_x)), int(round(float(p[1]) * scale_y)))
               for p in kf_trail]
        cv2.polylines(frame, [np.array(_kt, np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)
    if kf is not None and kf.get("pos") is not None:
        _kx = int(round(float(kf["pos"][0]) * scale_x))
        _ky = int(round(float(kf["pos"][1]) * scale_y))
        cv2.circle(frame, (_kx, _ky), 3, (255, 255, 0), -1, cv2.LINE_AA)     # 实心点 ✓
        cv2.circle(frame, (_kx, _ky), 8, (255, 255, 0), 1, cv2.LINE_AA)      # 空心圈 ✓
        _kvx, _kvy = (kf.get("vel") or (0.0, 0.0))
        _kdt = float((motion or {}).get("dt") or 0.0)
        if (_kvx or _kvy) and _kdt > 0:
            _arrow(float(kf["pos"][0]) * scale_x, float(kf["pos"][1]) * scale_y,
                   float(_kvx) * _kdt * scale_x, float(_kvy) * _kdt * scale_y,
                   (255, 255, 0), 1)                                          # KF 速度 ✓
    # ⭐⭐⭐⭐⭐ **速度跟踪档的「选中」表达**（用户 2026-10-07 ✓ 原话："**选中的检出框需要高亮，
    #   你设计展示形式**" ✓）—— 我设计的 = **青色粗框 ＋ 四个角标**（像取景框 ✓）：
    #   · 为什么不用黄框 ✗：黄色已经是"框左上角那行编号"的颜色 ✓（`(0,255,255)` ✓）⇒ 撞色分不清 ✓；
    #   · 为什么还要角标 ✗：这一档一帧十几格框 ✓ 只描一圈边容易被旁边框的边混掉 ✓
    #     ⇒ 四个 L 角把"我选的那一格"圈出来 ✓ 一眼认得出 ✓；
    #   · ⚠ **一个字都不写** ✗（"面板文字"按 2026-10-07 那轮口径**保持清空** ✓ —— 要写什么字
    #     **先问用户** ✓ 项目纪律 ✓）。
    if pick is not None and _vel:
        _vx1 = int(round((float(pick[1]) - float(pick[3]) / 2.0) * scale_x))
        _vy1 = int(round((float(pick[2]) - float(pick[4]) / 2.0) * scale_y))
        _vx2 = int(round((float(pick[1]) + float(pick[3]) / 2.0) * scale_x))
        _vy2 = int(round((float(pick[2]) + float(pick[4]) / 2.0) * scale_y))
        _vc = (255, 255, 0)                       # 🩵 青 ✓（这一档没有别人用它 ✓ 与黄/橙/白都分得开 ✓）
        cv2.rectangle(frame, (_vx1, _vy1), (_vx2, _vy2), _vc, 3)
        _cl = max(12, min(28, int(min(_vx2 - _vx1, _vy2 - _vy1) * 0.35)))
        for (_sx2, _sy2, _dx2, _dy2) in ((_vx1, _vy1, 1, 1), (_vx2, _vy1, -1, 1),
                                         (_vx2, _vy2, -1, -1), (_vx1, _vy2, 1, -1)):
            cv2.line(frame, (_sx2, _sy2), (_sx2 + _dx2 * _cl, _sy2), _vc, 5, cv2.LINE_AA)
            cv2.line(frame, (_sx2, _sy2), (_sx2, _sy2 + _dy2 * _cl), _vc, 5, cv2.LINE_AA)
    if pick is not None and not _vel:
        _pcx, _pcy = float(pick[1]), float(pick[2])
        _pw, _ph = float(pick[3]), float(pick[4])
        _q1 = (int(round((_pcx - _pw / 2.0) * scale_x)), int(round((_pcy - _ph / 2.0) * scale_y)))
        _q2 = (int(round((_pcx + _pw / 2.0) * scale_x)), int(round((_pcy + _ph / 2.0) * scale_y)))
        cv2.rectangle(frame, _q1, _q2, (0, 255, 255), 3)          # 🟡 黄：点选高亮 ✓
        # 标签（**两行** ✓ 用户 2026-10-03 ✓ 见 `pick_label` ✓）：
        #   第 1 行 = `类别 置信度 | (cx,cy) WxH=面积`（用户要的 `(360.4,323.0) 179x208=37232` ✓）；
        #   第 2 行 = `IoU最大砖 … | 圆矩IoU … | 圆矩∩框 …`（重叠量 ✓ 用户点名放这行 ✓）。
        #   ⚠ 只用 ASCII ✗ —— `cv2.putText` 不认全角 `｜`/`×` ⇒ 会画成 `?` ✗。
        #   ⚠ `cv2.putText` **不认 `\n`** ✗ ⇒ 自己按行拆开逐行画 ✓（黑底高度 = 行数 × 行高 ✓）。
        _ptxt = pick_label(pick, names)
        # ⭐⭐⭐ **自动换行 + 最小宽度**（用户 2026-10-04 ✓ 原话："点选之后的信息框能做个**最小宽度+
        #   自动换行**吗？现在**整个屏幕都不够宽了**" ✓✓ —— 见 `_PICK_*` / `wrap_px` ✓）：
        #   · 换行宽 = `min(最大宽, 画面宽 − 2×留白)` ✓ ⇒ **绝不顶出屏幕** ✓；
        #   · 面板宽 = `max(最小宽, 最长那行的宽)` ✓ ⇒ 短信息也像块面板 ✓。
        _pwrap = max(120, min(_PICK_MAX_W, frame.shape[1] - 2 * _PICK_MARGIN))
        _pln = wrap_px([t for t in _ptxt.split("\n") if t], _pwrap, _fs(0.75))
        _psz = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), 1)[0] for t in _pln]
        _pth = max(s[1] for s in _psz)
        _pwmax = max([s[0] for s in _psz] + [_PICK_MIN_W])         # ⭐ 最小宽度 ✓
        _plh = _pth + 8                       # 行高（含行距 ✓）
        # ⚠ **竖着也得放得下** ✗：换了行 ⇒ 行数会涨 ✓ ⇒ 从 `_PICK_TOP` 到画面底能摆几行就摆几行 ✓
        #   多的**截断**（末行加 ASCII 的 `...` ✓ —— `cv2.putText` **不认 `…`** ✗ 会画成 `?` ✓）。
        _pmaxln = max(1, (frame.shape[0] - _PICK_TOP - 6) // _plh)
        if len(_pln) > _pmaxln:
            _pln = _pln[:_pmaxln]
            _pln[-1] = _pln[-1][:max(1, len(_pln[-1]) - 3)] + "..."
            _psz = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), 1)[0] for t in _pln]
            _pth = max(s[1] for s in _psz)
            _pwmax = max([s[0] for s in _psz] + [_PICK_MIN_W])
        _phbox = _plh * len(_pln)
        _pbot = _q2[1] + _phbox + 10          # 默认贴框**下方** ✓
        if _pbot > frame.shape[0] - 4:                            # 框贴底 ⇒ 标签挪到框内上方 ✓
            _pbot = max(_phbox + 4, _q1[1] - 8)
        _px0 = max(_PICK_MARGIN, min(_q1[0], frame.shape[1] - _pwmax - _PICK_MARGIN))
        _ptop = _pbot - _phbox
        # ⚠⚠ **别顶到 HUD** ✗（用户 2026-10-04 ✓ 字号调大后暴露的 ✓）—— HUD 画在 `y≈38` ✓
        #   而面板最高能顶到 `y≈0` ⇒ **第 1 行被 HUD 的黑描边压住** ✗ ⇒ 让它从 `_PICK_TOP` 起 ✓。
        if _ptop < _PICK_TOP:
            _ptop = _PICK_TOP
            _pbot = _ptop + _phbox
        cv2.rectangle(frame, (_px0, _ptop), (_px0 + _pwmax + 8, _pbot), (0, 0, 0), -1)
        for _pk_i, _pk_t in enumerate(_pln):
            cv2.putText(frame, _pk_t, (_px0 + 4, _ptop + _plh * _pk_i + _pth + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), (0, 255, 255), 1, cv2.LINE_AA)
    if hud:
        # ⚠ **字号调大后基线也得往下挪** ✗（用户 2026-10-04 ✓ "窗口字体能大点" ✓）——
        #   原来 `y=26` 是配 `0.7` 的 ✓；现在字高 ~30px ✗ ⇒ 会**顶到画面上边缘**。
        cv2.putText(frame, hud, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, _fs(1.0),
                    (0, 0, 0), 5)
        cv2.putText(frame, hud, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, _fs(1.0),
                    (255, 255, 255), 2)
    # ⭐⭐⭐ **运动分离（新）的「判定依据」叠加**（用户 2026-10-03 ✓ 原话："**我希望：能看到你判定的
    #   可视化依据，不然我无法汇报问题**" ✓ —— 这句就是这一段的全部理由 ✓）：
    #   · 每条候选画一个圈：**圈越大 = 累积分 `score` 越高** ✓ 旁边标出**分数**（一眼看谁在涨 ✓）；
    #   · **被选中的那条** ⇒ **红色双圈 + `★`** ✓（"算法认它是真目标" ✓）；
    #   · 每条再画一支**黄箭头** = 它**相对"群体中位位移"的偏离**（`mv − median` ✓）
    #     ⇒ **箭头越长 = 越"不合群"** ✓✓ —— 这就是**判据本身** ✓ 可以直接核对 ✓；
    #   · 右上角一块**黑底文字面板**：群体中位位移 / 候选数 / 选中者的三项（score·dev·hits）／
    #     **Top3 明细** ✓ ⇒ 用户截图就能把"依据"发过来 ✓✓。
    #   ⚠ 只在运动分离模式下画（`motion["mode"] == "motion"` ✓）⇒ **经典模式一个像素都不变** ✓。
    #   ⚠ `cv2.putText` **只认 ASCII** ✗ ⇒ 文案全用英文/数字 ✓（中文会画成 `?` ✗）。
    # ⚠⚠ **这一段两档共用**（`motion` ＋ `velocity` ✓ 用户 2026-10-07 ✓）—— 它就是那 7 条里的
    #   四样（**框标签** / **灰框（推的）** / **蓝箭头** / **橙线** ✓）＋ 一样要挡掉的
    #   （**暗黄鼠标轨迹** ✗；夹取细框在更下面单独挡 ✓）。
    #   ⚠ 经典档 ⇒ **一个字都不进**（老画面不变 ✓）。
    if _mo_now in ("motion", "velocity"):
        # ⭐ **本模式的图层开关**（由 `_show_precomputed` 塞进 `motion["show"]` ✓）
        #   `cands` = 候选圈+分数+箭头 ｜ `rel` = **两条"相对群体"轨迹线**（目标=橙 ✓ 鼠标=暗黄 ✓）
        #   ｜ `boxes` 在更上层就过滤掉了 ✓
        #   ⚠ 口径提醒（2026-10-04 ✓）：`rel` 这两条都是"**减掉相机**"的；圆心那条路径含相机 ✓
        #   ⚠⚠ **必须在最前面取** ✗✗（我第一版放在 `_trk` 那边 ⇒ 橙色轨迹段比它更靠前 ⇒
        #     `NameError: _sh_ok is not defined` ✗）
        _sh_ok = (motion or {}).get("show") or {}
        _trk = list(motion.get("tracks") or [])
        # ⭐⭐⭐ **「相对群体」的轨迹线**（用户 2026-10-03 ✓ 原话："我希望：像经典模式一样，
        #   也把**相对于群体的轨迹线**画出来，这样就能看出**历史结果**" ✓）——
        #   以**每条候选当前所在**为原点，把它的 `rel_hist` 铺出去 ✓（`rel_hist` 里的点已经是
        #   "相对它现在"的偏移 ✓ ⇒ 加上它当前的屏幕坐标即可 ✓ 不用管绝对坐标 ✓）。
        #   ⭐ **怎么读这条线** ✓：**绕在原地打转** ⇒ 它跟大家一起动（**假目标** ✓）；
        #     **朝一个方向延伸出去** ⇒ 它在**相对群体**移动（**真目标** ✓✓）。
        #   ⚠ **只画"够资格"的 + 目标** ✗ —— 不然满屏 30 条线 ⇒ 反而看不清 ✓。
        # ⚠⚠ **只画"目标那一条"** ✗✗（用户 2026-10-03 反馈："**这些细细的蓝色轨迹线是什么？
        #   我从未要求加过**" ✓）—— 我上一版**把所有"够资格"的候选**（实测 28 条 ✗）全画了，
        #   颜色又偏蓝 ✓ ⇒ 满屏斜线 ⇒ 看着像乱画 ✗。用户真正要的是："**像经典模式一样，也把
        #   相对于群体的轨迹线画出来，这样就能看出历史结果**" ✓ ⇒ **要的是目标那条** ✓。
        # ⚠ 只画**目标那一条** ✓ 且**听"相对轨迹"开关**（不是"候选圈" ✓ 两者独立 ✓）
        # ⭐⭐⭐ **橙色线 = 「鼠标在群体坐标系里的运动轨迹」**（用户 2026-10-04 ✓ 原话："我希望橙色线
        #   代表**鼠标实际在群体坐标系中的运动轨迹**" ✓✓）——
        #   ⚠⚠ **口径换了** ✗：以前这条橙线画的是"**目标候选**的相对轨迹"✓（那只是**过程量** ✗）；
        #     现在画的是**鼠标**（= 这个项目**真正要控制的东西** ✓）⇒ "**鼠标相对群体怎么走**"
        #     才是**结果指标** ✓✓（跟着真目标 ⇒ 会走出一条线 ✓；被假目标带着 ⇒ 绕在原地 ✓）。
        #   ⚠ 数据由后端给（`motion["cursor_rel"]` ✓ 见 `MotionRunner.step` ✓）：点已经是
        #     "**相对鼠标当前所在**的偏移" ✓ ⇒ 以**光标**为锚点铺出去即可 ✓（不用管绝对坐标 ✓）。
        #   ⚠⚠ **听"鼠标轨迹"这个开关** ✗✗（用户 2026-10-04 ✓ 他把"相对轨迹"拆成了两个 ✓
        #     见 `_show_layers` ✓）—— 原来它和"真目标轨迹"共用一个开关 ✗ ⇒ 想只看目标
        #     那条也做不到 ✓（而目标那条才是核心 ✓）。
        # ⚠ 速度跟踪档：暗黄线（= **鼠标**在群体坐标系里的轨迹 ✓）**不在那 7 条里** ⇒ 不画 ✓
        #   （那 7 条里跟"群体轨迹"有关的**只有橙色那条**（圆心）✓）。
        if _sh_ok.get("rel_mouse", True) and cursor is not None and not _vel:
            _ch = list(motion.get("cursor_rel") or [])
            _cx0 = float(cursor[0]) * scale_x
            _cy0 = float(cursor[1]) * scale_y
            # ⚠⚠⚠ **鼠标这条改成"暗黄"** ✗✗（用户 2026-10-04 ✓ 原话："**如果仅仅只是滑行，
            #   圆心不应该在橙色轨迹的切向吗？**" ✓）—— ⚠ 他**一直把橙色当成"目标的轨迹"**
            #   ✗（橙色画的其实是**鼠标** ✓ 见上面那段 ✓）⇒ 颜色把两件事搅在一起了 ✓✗
            #   ⇒ **橙色让给"目标"**（那才是他真正盯着看的、判据本体 ✓ 见下面那段 ✓）
            #     **鼠标这条降成暗黄**（次要 ✓ 它是"结果指标"✓ 但不用抢眼 ✓）。
            for _k2 in range(1, len(_ch)):
                cv2.line(frame,
                         (int(_cx0 + _ch[_k2 - 1][0] * scale_x),
                          int(_cy0 + _ch[_k2 - 1][1] * scale_y)),
                         (int(_cx0 + _ch[_k2][0] * scale_x),
                          int(_cy0 + _ch[_k2][1] * scale_y)),
                         # ⚠ **鼠标这条：橙 → 暗黄 + 细**（把"橙色"让给目标 ✓ 见上面那段 ✓）
                         (0, 170, 170), 1, cv2.LINE_AA)
        # ⚠ **方向置信度（鼠标周围那圈灰箭头 + `dir` 文字）已移除**（用户 2026-10-04 ✓ 原话：
        #   "灰辐射…也不需要，移除" ✓）。
        # ⭐⭐⭐⭐ **真目标那条「相对群体」轨迹**（= 判据本体 ✓ 用户真正盯着看的那条 ✓）——
        #   ⭐ **口径已改**（用户 2026-10-04 ✓ 原话："**橙色线就应该是报出位置相对群体走过的路**" ✓✓）：
        #     · **数据** = `motion["pos_rel"]`（= 每拍 `Δ圆心 − 相机` 的累计 ✓ 见 `MotionRunner.step` ✓）；
        #     · **锚点 = 绿圈（`pos`）** ✓ ⇒ **端点必然落在圆心上** ✓✓。
        #   ⚠⚠ **改掉了什么** ✗✗：原来画的是**候选轨迹的 `rel_hist`**、锚点是**候选框位置** ✗
        #     ⇒ 圆心一路滑出去时，线**留在框那儿**（用户原话："**圆心走那么远了，橙线怎么没跟过来**" ✓）。
        #   ⚠ 经典模式**没有** `pos_rel`（那是运动分离的 `MotionRunner` 给的 ✓）⇒ 这里**回退到老的
        #     逐轨迹画法** ✓（经典模式画面**不许变** ✗）。
        #   ⚠ 颜色**橙色 + 粗**（2026-10-04 从"细紫"改过来 ✓：橙色原先画的是**鼠标** ✗
        #     而他一直把橙色当目标 ✓ 见上面鼠标那段 ✓）；听"真目标轨迹"这个开关 ✓。
        _ph = list(motion.get("pos_rel") or [])
        if _ph and pos is not None and _sh_ok.get("rel_tgt", True):
            _bx = float(pos[0]) * scale_x
            _by = float(pos[1]) * scale_y
            for _k2 in range(1, len(_ph)):
                cv2.line(frame,
                         (int(_bx + _ph[_k2 - 1][0] * scale_x),
                          int(_by + _ph[_k2 - 1][1] * scale_y)),
                         (int(_bx + _ph[_k2][0] * scale_x),
                          int(_by + _ph[_k2][1] * scale_y)),
                         (0, 150, 255), 2, cv2.LINE_AA)
        else:
            # ⚠ 速度跟踪档：这条退路画的是"**候选轨迹**的相对历史" ✗ ⇒ 不在那 7 条里 ⇒ 空 ✓
            #   （该档要的橙线只有"圆心那条" ✓ 见上面 ✓）。
            for _lp in ((_trk if (_sh_ok.get("rel_tgt", True) and not _vel) else [])):
                if not _lp.get("sel"):
                    continue
                _h = _lp.get("rel_hist") or []
                if len(_h) < 2:
                    continue
                _bx = float(_lp["p"][0]) * scale_x
                _by = float(_lp["p"][1]) * scale_y
                for _k2 in range(1, len(_h)):
                    cv2.line(frame,
                             (int(_bx + _h[_k2 - 1][0] * scale_x),
                              int(_by + _h[_k2 - 1][1] * scale_y)),
                             (int(_bx + _h[_k2][0] * scale_x),
                              int(_by + _h[_k2][1] * scale_y)),
                             (0, 150, 255), 2, cv2.LINE_AA)
        # ⭐⭐⭐ **蓝箭头 = 每个检出框的「瞬时绝对速度」**（用户 2026-10-03 ✓ 原话："**每个假目标
        #   检出框**加**蓝色箭头**显示**瞬时绝对速度**，这个没看到" ✓）——
        #   `box_v` = 本拍**每个配对成功的检出框** `(x, y, dx, dy)` ✓ ⇒ **每个框一根** ✓✓
        #   （⚠ 与"够资格的候选"无关 ✗ —— 上一版挂在那层上 ⇒ 大部分框没有 ✗ 他就没看到 ✓）。
        #   长度**硬封顶**（`_ARR_MAX` ✓）：真目标偶尔一拍走 200+px ✗ 不封顶会横跨画面 ✓。
        _ARR_MAX = 110.0
        # ⭐⭐⭐ **在「检出框」上写轨迹号**（用户 2026-10-04 ✓ 原话："**没有看到标出 id**，
        #   希望能在**检出框上**写出来" ✓）—— ⚠ 他为什么没看到 ✗：上一版 `#N` 画在**候选圈
        #   那一层** ✗ ⇒ 一关「候选圈」图层（`cands`）就**整层消失** ✓；而"框"这一层是**常显**的
        #   ⇒ 号跟着框走 ⇒ **一定看得见** ✓✓。
        #   ⚠⚠ **每个框都写**（不只是配上的 ✗）：配上过的写 `#id` ✓、**没配上的写 `?`** ✓
        #     —— 后者本身就说明"**这一格还没有归属**"（新出现的 / 刚睡醒还没接上的 ✓）。
        # ⭐⭐⭐ **在检出框上写完整标签：`shape 0.90 #22`**（用户 2026-10-04 ✓ 原话："**没有看到
        #   编号**，你能**写明显点**吗？例如 `Shape 0.90 #22`" ✓✓）—— ⚠ 格式**照用户给的** ✓：
        #   类别名 ＋ 置信度 ＋ `#轨迹号` ✓（与经典模式的 `shape 0.98 #001` 同一套说法 ✓）。
        #   ⚠⚠ **黑底 + 亮黄字** ✓：背景是**褐色花岗岩**、还压了灰蒙版 ⇒ 光描边**看不清** ✗
        #     （"没看到"很可能就是这个原因之一 ✓）⇒ 给它一块**实心黑底** ✓ 一定跳出来 ✓。
        #   ⚠ 只在"运动分离"模式下写 ✗ —— 经典 `Runner` 的 `box_v` 是**另一个东西**（4 元组、
        #     没有 id ✗）⇒ 不加这道门会在经典模式里**乱写号** ✓（实测过 ✓）。
        # ⚠⚠ **"框标签"也归「检出框」开关管** ✗✗（用户 2026-10-04 ✓）—— 原来它**独立于此**
        #   ✗ ⇒ **关掉检出框之后标签还在** ✗ ⇒ 而**候选圈那边已经有一个 `#13` 牌** ✓
        #   ⇒ 画面上每个目标**挂两个标签**（`shape 0.98 #13 m0.82` ＋ `#13` ✓）⇒ 又乱又挡 ✓✗
        #   ⇒ 现在**同进同退** ✓：关掉"检出框" ⇒ **框和它的标签一起走** ✓
        #     ⇒ 画面只剩【候选圈 + 编号牌 + 轨迹线】✓✓（正好是用户要的"清屏"✓）。
        if _mo_now in ("motion", "velocity") and _sh_ok.get("boxes", True):
            # ⚠ 速度跟踪那三档状态**逐位对齐 `box_v`**（见 `VelocityChassis` ✓）⇒ 这里取一份
            #   给下面"等待上板不写编号"用 ✓；运动分离档给空列表 ✓（那条 `if` 不会命中 ✓）。
            _stv5 = (list(((motion or {}).get("vel") or {}).get("state") or [])
                     if _vel else [])
            for _k5, _it in enumerate(motion.get("box_v") or []):
                if len(_it) < 5:
                    continue
                # ⚠⚠ **速度跟踪档：「等待上板」的框一律不写编号** ✗✗（用户 2026-10-07 ✓ 口径
                #   "**有标签就是上板，无标签就是等待上板**" ✓）—— 融合框**照旧写"主人"那个号** ✓
                #   （它不在这条跳过里 ✓ 见下面 `_lid` 那段 ✓）；状态逐位对齐 `box_v` ✓ 见底盘 ✓。
                if (_vel and _k5 < len(_stv5) and str(_stv5[_k5]) == _VEL_WAIT):
                    continue
                # ⚠⚠⚠ **关掉「检出框编号」⇒ 整个标签都不画** ✗✗（**用户 2026-10-04 更正** ✓
                #   原话：「**检出框编号 不勾选的时候 不仅仅是不显示编号，框中间以及没框时候
                #   残留的 lable 都不要显示**」✓✓）—— ⚠ 我第一版**只砍了 `#N` 那一截** ✗
                #   （理由："`shape 0.98` 和匹配分不是编号"✓）⇒ 结果**关掉之后框上还挂着一行字**
                #   ✗ ⇒ 与他要的"清屏"完全不符 ✓（我理解错了需求 ✓ 不是代码写错 ✓）。
                #   ⇒ 现在的语义**干净**：这个勾 = "**画不画检出框/轨迹的编号标签**" ✓
                #     ⇒ 一关 ⇒ ① 框中间那行 `shape 0.98 #22 m0.82` 没了 ✓
                #              ② "睡着的"那块 `#22 推的` 没了 ✓（见下面 ✓）
                #              ③ 候选圈旁边那块 `#13` 也没了 ✓（见候选那段 ✓）
                #     而 **框本体 / 蓝箭头 / 灰框（位置）/ 分数** 都**不受影响** ✓（它们不是标签 ✓）。
                if not _sh_ok.get("box_id", True):
                    continue
                # ⭐⭐⭐⭐⭐ **标签拆成两行**（用户 2026-10-04 ✓ 原话："**把目标标签里的例如
                #   「#14 m0.89」换一行显示**" ✓✓）—— ⚠ 原来一行 `shape 0.98 #14  m0.89`
                #   **又宽又挡** ✗（框心那一条横着盖掉半个框 ✓）⇒ 拆成：
                #     · **第 1 行 = `shape 0.98`**（"像不像一个标准假目标" ✓）；
                #     · **第 2 行 = `#14  m0.89`**（"是谁" ＋ "这格观测多可信" ✓）。
                #   ⇒ 宽度**砍掉近一半** ✓（挡的面少 ✓ 数字一个不少 ✓）；
                #   ⚠ 文字口径全在 `box_label_lines` 里 ✓（**同源** ✓ 自检测的就是它 ✓）；
                #   ⚠ 整块**仍以框心为中心** ✓（上下对称 ✓ = 位置口径不变 ✓ 融合框那套
                #     "标签挂在推导位置上"（见下面 ✓）也照旧 ✓）。
                # ⭐⭐⭐⭐⭐ **融合框上的编号写"主人"（= 那格假目标 ✓）** ✗✗（用户 2026-10-06 ✓ 原话：
                #   "**先把融合框里的标签分配给假目标，例如 61 帧 #11 不应该写着灰框推的，因为它
                #   才是这个框的主人**" ✓✓）——
                #   ⚠ `_it[13]` = 后端算好的"**一直守着这一格的那条账**" ✓（见 `lie_motion`
                #     的 `_FUSE_OWNER_HITS` 那段 ✓）；取不到 ⇒**照旧按"配对到谁写谁"** ✓（老画面不变 ✓）。
                #   ⚠⚠ **只改标签** ✗：框本体 / 蓝箭头 / 判定**一个字不动** ✓。
                _lid = (int(_it[13]) if len(_it) > 13 and int(_it[13]) >= 0 else int(_it[4]))
                _b1, _b2 = box_label_lines(_lid,
                                           (_it[5] if len(_it) > 5 else None),
                                           (_it[6] if len(_it) > 6 else None))
                #   ⭐⭐⭐⭐⭐ **2026-10-07：锚点改「框的左上角」** ✗✗（用户示意图 ✓）—— 原来整块
                #     **以框心为中心**画 ✗ ⇒ 正好压住"我们在看的那一带"（绿圈圆心 / 推导位置 ✓）
                #     ⇒ 现在交给 `box_tab_pos`（**可钉** ✓）：左对齐 ＋ 整块在框**上沿之外** ✓。
                _acx, _acy = float(_it[0]), float(_it[1])          # 框心（加工域 ✓）
                _aw = (float(_it[10]) if len(_it) > 10 else 0.0)   # 框宽 ✓（全分辨率 ✓）
                _ah = (float(_it[11]) if len(_it) > 11 else 0.0)   # 框高 ✓
                _tx, _ty = int(_acx * scale_x), int(_acy * scale_y)
                # ⭐⭐⭐⭐⭐ **融合框（`st == 1`）的标签，改画在"算法推导出的那个位置"** ✗✗
                #   （用户 2026-10-04 ✓ 原话："帧 16 到帧 24，**#11 的标签还是明显在相对群体运动**" ✓；
                #   更早那句口径："我们要冻住的是**标签位置**，也就是**融合框内假目标的框心**，而不是
                #   「**融合框的框心**」" ✓✓）——
                #   ⚠ 病根：融合框的框心 = **两个目标的并集中点** ✗ ⇒ 它**天生**就相对群体在漂
                #     （实测 **3~30 px/拍** ✓）⇒ 编号挂在那儿 ⇒ 看着像"这条假目标在动" ✗✗。
                #   ⚠ 而**轨迹自己的位置**（`tracks` 里的 `p` ✓ = 推导值 ✓）现在是**群体静止**的 ✓
                #     （推进已改用"当拍群体中位" ✓ 见 `lie_motion` ✓）⇒ 标签挂过去就**钉住**了 ✓✓。
                #   ⚠ **框本体（洋红）照旧画在检出框上** ✓ —— 它表达"YOLO 看到的并集框" ✓ 不许挪 ✗。
                # ⭐⭐⭐⭐⭐ **R2：标签挂「我们真正在用的那个位置」**（用户 2026-10-05 ✓ 原话：
                #   "**R1+R2+R3**" ✓✓）—— ⚠⚠ 旧规则是"**有框且 `st != 1` ⇒ 挂框心**" ✗ ⇒ 只要配对
                #   给了**一格远框**（实测帧 13 的 `#1`：**复活轮**在 **107px** 外接了一格 ✓），
                #   标签就跟着它**跳 109px** ✗✗；可那一拍它走的是"粘住 / 冷却"链 ⇒
                #   **内部根本没用那格框** ✓ ⇒ 画面与口径自相矛盾 ✓（用户当场抓到 ✓）。
                #   ⇒ 现在：后端 **`trust == True`**（= 这拍 `obs` **真的取自那格框** ✓）⇒ 挂**框心** ✓；
                #     **否则**（粘住 / 滑行 / 断掉 ✓，或 `st == 1` 融合 ✓）⇒ 一律挂 **`p` = `obs`** ✓。
                #   ⚠ 老后端没这个字段时不崩 ✓（退回旧规则：`st == 1` 才挂 `p` ✓）。
                _tk2 = next((_t for _t in (motion.get("tracks") or [])
                             if int(_t.get("tid") or 0) == int(_lid)), None)
                _tr2 = None if _tk2 is None else _tk2.get("trust")
                if (_tk2 is not None and _tk2.get("p") is not None
                        and ((len(_it) > 7 and int(_it[7]) == 1)
                             or (_tr2 is not None and not bool(_tr2)))):
                    _tx = int(float(_tk2["p"][0]) * scale_x)
                    _ty = int(float(_tk2["p"][1]) * scale_y)
                    _acx, _acy = float(_tk2["p"][0]), float(_tk2["p"][1])
                    _aw = _ah = 0.0                  # ⚠ 挂的是**推导位置** ⇒ 没框 ⇒ 按"点"往上摆 ✓
                _s1 = cv2.getTextSize(_b1, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), 1)[0]
                _s2 = cv2.getTextSize(_b2, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), 1)[0]
                #   ⭐⭐⭐⭐⭐ **2026-10-07：位置 ＋ 底色两处一起改** ✗✗（用户 ✓："**标签还是在框心
                #     挡视野，看我的示意图**" ＋ "**另外把标签的黑色背景去掉**" ✓✓）——
                #     · **位置** = 框**左上角外侧**（左对齐 ✓ 整块在框上沿之上 ✓）⇒ 框内**一个字
                #       不落** ✓；口径全在 `box_tab_pos` 里 ✓（**可钉** ✓ 见那条钉子 ✓）；
                #     · **黑底去掉** ✗ = 删掉那句 `cv2.rectangle(..., (0, 0, 0), -1)` ✓ —— 原来
                #       那块黑牌子**比字还大一圈** ✗（`-3 / +4` 的边距 ✓）⇒ 一块一块黑膏药 ✓。
                #     ⚠⚠ 去掉底色后字是**黄字直接压在画面**上 ✓（`(0, 255, 255)` ✓）—— 若在某些
                #       亮背景上发飘 ✗ ⇒ 说的是"描边"（`thick` 先画一遍黑再画黄 ✓）而不是加回牌子 ✓
                #       （要就说一声 ✓ 一行的事 ✓）。
                _px2, _y1, _y2 = box_tab_pos(_acx, _acy, _aw, _ah, scale_x, scale_y,
                                             _s1, _s2, (frame.shape[1], frame.shape[0]))
                cv2.putText(frame, _b1, (_px2, _y1),
                            cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, _b2, (_px2, _y2),
                            cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), (0, 255, 255), 1, cv2.LINE_AA)
            # ⭐⭐⭐ **"睡着的"轨迹也留标签**（用户 2026-10-04 ✓ 原话："**即使检出框消失了，能让
            #   它的编号标签也一直显示吗**？" ✓✓）—— ⚠⚠ 上面那一轮**只覆盖"有检出框的"** ✗
            #   （`box_v` 里全是"这拍配上框"的 ✓）⇒ 而它们这一拍**没有框** ✗ ⇒ 位置是**推的** ✓
            #   ⇒ **号会跟着框一起消失** ✗✗（而用户要的恰恰是"框没了号还在" ✓）。
            #   ⇒ 所以**再遍历一遍 `tracks`** ✓：给"这拍没配上框"的那些，在**它推断出来的那个位置**
            #     上补一个标签 ✓ ＋ 一个**标准尺寸的灰框** ✓（"框没有了，但目标就在这儿" ✓
            #     ⚠ 尺寸用**标准尺寸** ✓ 正合用户那条"**所有目标尺寸一样**" ✓）。
            _bids = {int(_it[4]) for _it in (motion.get("box_v") or []) if len(_it) >= 5}
            #   ⭐⭐⭐⭐⭐ **融合框的"主人"这一拍是有座位的** ✗✗（用户 2026-10-06 ✓ 原话："**61 帧 #11
            #     不应该写着灰框推的，因为它才是这个框的主人**" ✓✓）—— ⇒ **不再给它画"#N 推的"
            #     ＋灰框** ✓（它的座位就是那格融合框 ✓、编号也已经写在那格框上了 ✓ 见上面那段 ✓）。
            _owners = {int(_it[13]) for _it in (motion.get("box_v") or [])
                       if len(_it) > 13 and int(_it[13]) >= 0}
            #   ⚠⚠ **这个基准是"老两档"用的** ✗（`MotionTracker.std_area` = 群体基准 ✓ 照旧 ✓）——
            #     速度跟踪档**没有群体标准面积** ✗（用户 2026-10-07 规则1 ✓："**群体不再有标准面积**，
            #     只有标准速度。**每个目标有自己的标准面积，存在自己的 id 里**" ✓✓）⇒ 那一档下面
            #     一律用**那个 id 自己**的 ✓（见循环里 `_t["std_area"]` ✓）；老两档没这个键 ⇒ 回落到
            #     这里 ✓（= 一个字不变 ✓ 零污染 ✓）。
            _sw_grp = ((float((motion or {}).get("std_area") or 0.0)) ** 0.5)
            # ⚠⚠ **速度跟踪档：只画「有标签的假目标丢失检出框时的记录」** ✗✗（用户 2026-10-07 ✓
            #   图例**第 2 条**的原话 ✓）—— "有标签" = **发过编号的座位**（= 上板 ✓）⇒ 由底盘的
            #   `board_ids` 给 ✓（⚠ 它**含"这一拍没框、但仍是座位"的那些** ✓ 正是这条记录要的 ✓）；
            #   等待上板那些**没有标签** ⇒ 不给它们画灰框 ✓（连"#N 推的"也不给 ✓ 因为压根没号 ✓）。
            #   ⚠ 运动分离档 ⇒ `_bids_v is None` ⇒ 那道闸**不生效** ✓（老画面一个字不变 ✓）。
            _bids_v = (set((motion.get("vel") or {}).get("board_ids") or []) if _vel else None)
            for _t in (motion.get("tracks") or []):
                if _bids_v is not None and int(_t.get("tid") or 0) not in _bids_v:
                    continue
                if int(_t.get("tid") or 0) in _bids or int(_t.get("tid") or 0) in _owners:
                    continue                     # 有框 / 是融合框的主人 ⇒ 上面已经写过 ✓ 不重复 ✗
                _tx, _ty = track_label_anchor(_t, pos, scale_x, scale_y)
                #   ⚠ 上面这个函数里写清了**锚点口径** ✓（座位"推的"⇒ 钉在**圆心** ✓；其余 ⇒ 钉它
                #     自己的推算位置 `p` ✓）—— ⚠⚠ **别改回 `motion["pos"]`** ✗（实测它是 `None` ✓
                #     我第一版就是这么错的 ⇒ 那段**从来没生效过** ✗ 见那个函数的 docstring ✓）。
                #   ⇒ 钉子：`selftest_lie_demo.test_track_label_anchor_circle` ✓。
                #   ⭐⭐ **灰框 = "它应该有多大"** ✓ —— 速度跟踪档取**这个 id 自己**的标准面积 ✓
                #     （规则1 ✓ 存在自己的 id 里 ✓）；老两档没这个键 ⇒ 用群体基准 ✓（照旧 ✓）。
                _a_id = float(_t.get("std_area") or 0.0)
                _sw = ((_a_id ** 0.5) if _a_id > 0.0 else _sw_grp)
                if _sw > 0.0:                    # 灰框 = "它应该有多大"（标准尺寸 ✓）
                    _hw = int(_sw * scale_x / 2.0)
                    _hh = int(_sw * scale_y / 2.0)
                    cv2.rectangle(frame, (_tx - _hw, _ty - _hh), (_tx + _hw, _ty + _hh),
                                  (150, 150, 150), 1, cv2.LINE_AA)
                # ⭐⭐⭐ **这块牌子"整块都是编号"** ✗ ⇒ 关掉「检出框编号」就**整个不画** ✓
                #   （用户 2026-10-04 ✓）—— ⚠ 但**上面那个灰框照留** ✗：它表达的是
                #   "**目标大概在这儿**"（位置信息 ✓）而不是"它是几号"（归属 ✗）。
                if not _sh_ok.get("box_id", True):
                    continue
                _bt2 = "#%d 推的" % int(_t.get("tid") or 0)
                _sz2 = cv2.getTextSize(_bt2, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75), 1)[0]
                _x2, _y2 = _tx - _sz2[0] // 2, _ty - 2
                #   ⭐⭐⭐⭐⭐ **黑底去掉** ✗✗（用户 2026-10-07 ✓ "**另外把标签的黑色背景去掉**" ✓✓）
                #     —— 原来这里也压着一块黑牌子 ✓（`(0, 0, 0), -1` ✓）⇒ 与两行页签那块一样，
                #     整块**比字大一圈** ✗ ⇒ 一并删掉 ✓（全画面"标签黑底"归零 ✓）。
                cv2.putText(frame, _bt2, (_x2, _y2),
                            cv2.FONT_HERSHEY_SIMPLEX, _fs(0.75),
                            (190, 190, 190), 1, cv2.LINE_AA)
        for _it in ((motion.get("box_v") or []) if _sh_ok.get("boxes", True) else []):
            # ⚠⚠ **蓝箭头（每个框的瞬时速度）也归「检出框」开关管** ✗（用户 2026-10-04 ✓）——
            #   它画的就是"**检出框的速度**" ✗ ⇒ 框都关了、箭头还留着 ⇒ 一屏蓝箭头 ✓✗
            #   ⇒ 与框、标签**同进同退** ✓（要"清屏"就三样一起走 ✓）。
            _bx0, _by0, _bdx, _bdy = (float(_it[0]), float(_it[1]),
                                      float(_it[2]), float(_it[3]))
            _bm = (_bdx ** 2 + _bdy ** 2) ** 0.5
            # ⭐⭐⭐⭐⭐ **速度跟踪档：蓝箭头「固定长度、只表方向」** ✗✗（用户 2026-10-07 ✓ **最终
            #   口径**："**蓝箭头不再以长短展示速度大小，只代表速度方向，固定长度**" ✓✓）——
            #   ⚠⚠ 所以这里**只用方向**（`_bdx/_bdy` 归一 ✓）、长度恒 = `_VEL_ARR_LEN` ✓；
            #     位移小到没方向（< 0.5px ✓）⇒ **不画** ✓（不编方向 ✗）；
            #   ⚠ 老两档**一个字不变** ✗（仍是"长度 = 位移"、2px 门 ✓ 零污染 ✓）。
            if _vel:
                if _bm < 0.5:
                    continue
                _sx0, _sy0 = int(_bx0 * scale_x), int(_by0 * scale_y)
                cv2.arrowedLine(frame, (_sx0, _sy0),
                                (int(_sx0 + _bdx / _bm * _VEL_ARR_LEN),
                                 int(_sy0 + _bdy / _bm * _VEL_ARR_LEN)),
                                (255, 120, 0), 2, tipLength=0.30, line_type=cv2.LINE_AA)
                continue
            if _bm < 2.0:
                continue
            _k = min(1.2, _ARR_MAX / max(1e-6, _bm))
            cv2.arrowedLine(frame, (int(_bx0 * scale_x), int(_by0 * scale_y)),
                            (int((_bx0 + _bdx * _k) * scale_x),
                             int((_by0 + _bdy * _k) * scale_y)),
                            (255, 120, 0), 2, tipLength=0.35, line_type=cv2.LINE_AA)
        # ⭐⭐⭐⭐⭐ **融合框«夹取圆»的允许区域**（用户 2026-10-04 ✓ 原话："**在有唯一融合框时，
        #   融合框应该夹取圆，它是融合期除了框心指导的第二种指导**" ✓✓）——
        #   ⚠⚠ **必须画出来** ✗✗（所见即所得 ✓）：不然"圆明明想往外走、却被按住"在画面上
        #     **看不出原因** ✓。画的**就是**后端这一拍真正生效的那个矩形（`out["clamp"]` ✓
        #     = 那个唯一融合框四边各内缩一个半径 ✓）⇒ **圆必须待在里面** ✓。
        #   ⚠ 归「检出框」开关管 ✓（它就是"检出框上的一条注解" ✓）；且**只在这一拍真夹了**才画 ✓
        #     （没夹 ⇒ `None` ⇒ 什么都不画 ✓ 不误导 ✓）。
        # ⚠ 速度跟踪档：这圈洋红细框（= 融合框**夹取**圆的允许区 ✓）**不在那 7 条里** ⇒ 不画 ✓。
        _cl = ((motion or {}).get("fuse_clamp")
               if (_sh_ok.get("boxes", True) and not _vel) else None)
        if _cl:
            cv2.rectangle(frame,
                          (int(float(_cl[0]) * scale_x), int(float(_cl[1]) * scale_y)),
                          (int(float(_cl[2]) * scale_x), int(float(_cl[3]) * scale_y)),
                          # ⚠ 洋红（与那个融合紫框同色系 ⇒ 一眼看出是"框的内缩区" ✓）＋**细**
                          #   （1px ✓ 它是"约束区域"、不是检出框 ⇒ 不许抢眼 ✗）
                          (255, 0, 255), 1, cv2.LINE_AA)
        # ⭐⭐⭐⭐⭐ **候选圈整块已移除**（用户 2026-10-04 ✓ 原话：「不需要候选圈 我们有且仅有一个圆，
        #   代表报告位置，就是我们认为此刻真目标所处的位置」✓✓）—— 这里原来画：候选圈 +「★」+ 分数
        #   ＋ 每候选的偏离黄箭头 ＋ **候选白箭头**（固定 56px 长 ✗ 看着像「与橙线分离」✓）⇒ **全删** ✓。
        #   ⚠⚠ **数据一个字没动**（`motion["tracks"]` 照旧 ✓）⇒ 实时信息栏 / 日志 / 判定**不受影响** ✓。
    return frame


# ══════════════════════════════════════════════════════════
# 窗口程序
# ══════════════════════════════════════════════════════════
def _code_stamp_now():
    """**代码版本戳** = 几个源文件里**最新的修改时间**（`MM-DD HH:MM` ✓ 写进窗口标题 ✓）。

    为什么要它（2026-10-01 实测教训 ✓）：排查"为什么登记框和检出框全错开"时，我这边量到
    0.0px、用户那边 17.8px，**同一份文件**却对不上 ⇒ 只有两种可能：跑的是旧代码、或跑的
    是另一份副本 ✗ —— 光看命中率/参数分辨不出来 ✗（那次命中率两边一样 ✓）。戳进标题里
    一眼可判 ✓。
    """
    try:
        import os as _os
        import time as _t
        _ps = [Path(__file__),
               Path(__file__).resolve().parent.parent / "perception" / "lie_tracker.py",
               Path(__file__).resolve().parent.parent / "perception" / "lie_registry.py"]
        _m = max(_os.path.getmtime(p) for p in _ps if p.exists())
        return _t.strftime("%m-%d %H:%M", _t.localtime(_m))
    except Exception:
        return "?"


# ⚠⚠ **必须在"进程启动那一刻"取一次并冻住** ✗✗ —— 每次 `setWindowTitle` 现算的话，
#   窗口**在改代码之前**就开着、之后才载入视频 ⇒ 标题里会显示**改后的时间** ✗ ⇒ 看着像
#   "跑的是新代码"，其实进程里还是旧模块 ✗（2026-10-01 排查"登记框错位"时就被它骗过一轮 ✓）。
#   冻在模块级 = 窗口里跑的到底是哪一版，一眼可判 ✓。
CODE_STAMP = _code_stamp_now()


def code_stamp():
    """窗口标题里用的版本戳（= **本进程启动时**这几个源文件的最新修改时间 ✓）。"""
    return CODE_STAMP


def run_window(args):
    from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
    from PyQt5.QtGui import (QColor, QCursor, QFont, QImage, QKeySequence, QPainter,
                             QPixmap)
    from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
                                 QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                                 QMessageBox,
                                 QMenu, QPlainTextEdit, QPushButton, QShortcut,
                                 QSlider, QToolBar, QVBoxLayout, QWidget)
    # ⚠ **滚轮不许改参数**（项目规范 ✓）⇒ 数字框一律 `NoWheel*`（与工作台同一份 ✓）
    from gui.widgets import NoWheelDoubleSpinBox, NoWheelSpinBox

    class _ImageView(QWidget):
        """⭐ **图像视图**（用户 2026-10-01 ✓ 原话："鼠标在视图上会持续显示坐标 / 滚轮可以
        缩放 / 双击适应窗口" ✓）。

        · **滚轮缩放**（`wheelEvent` ✓ 以鼠标为锚点 ✓）：上滚放大、下滚缩小，范围 0.05×~20× ✓；
        · **双击适应窗口**（`mouseDoubleClickEvent` ✓）：回到"整帧缩进窗口、居中" ✓（= 原来的
          默认行为 ✓）；
        · **鼠标坐标持续显示**（`setMouseTracking(True)` + `mouseMoveEvent` ✓ 不按键也报 ✓）：
          左上角一直显示鼠标下的**加工域坐标**（`x/sx, y/sy` ✓ 与后端红框/登记同一把尺 ✓），
          鼠标处画一个绿色十字 ✓。
        · ⚠ 缩放/双击只影响**这一窗的显示** ✗ 不动帧数据、不动任何判定 ✓（纯视图层 ✓）。
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMouseTracking(True)          # 不按键也持续报 mouseMove ✓（坐标 ✓）
            self.setMinimumSize(810, 540)
            self.setStyleSheet("background:#101010; color:#ccc;")
            self._pix = None                     # 当前帧（全分辨率 QPixmap ✓）
            self._text = ""                      # 无图时的提示文案（原 QLabel 的 setText ✓）
            self._sx = self._sy = 1.0            # 原图 → 加工域 的缩放（坐标显示用 ✓）
            self._fit = True                     # True = 适应窗口（自动缩放居中 ✓）
            self._zoom = 1.0                     # 手动缩放倍数（_fit=False 时用 ✓）
            self._off = QPointF(0.0, 0.0)        # 图像(0,0)在视图里的位置（_fit=False 时用 ✓）
            self._mouse = None                   # 鼠标在视图里的位置（QPointF ✓）
            self._mouse_img = None               # 鼠标在**图像坐标**里的位置 (x, y) 或 None ✓
            # ⭐ **中键拖动平移**（用户 2026-10-02 ✓）：
            self._pan_from = None                # 按下中键那一点（视图坐标 ✓）；None = 没在拖 ✓
            self._pan_off = None                 # 按下那一刻的图像偏移（`_off` 快照 ✓）
            # ⭐⭐ **左键"点选检出框"的回调**（用户 2026-10-03 ✓ 原话："你能让我在点选检出框的
            #   时候显示 (360.4,323.0) 179×208=37232 这种信息吗" ✓）：主窗设它 =
            #   `_on_pick_box` ✓，参数是**加工域坐标**（= 图像坐标 ÷ `_sx/_sy` ✓ 与后端同一把尺 ✓）。
            #   ⚠ 视图层只负责"把点报上去" ✗ 命中哪一格由主窗判 ✓（这里拿不到检出框 ✓）。
            self._on_pick = None
            # ⭐⭐⭐ **右键 = "选中 + 弹菜单"**（用户 2026-10-03 ✓ 原话："加一个**右键点击检出框
            #   选中并弹出菜单**，目前只有一项『**复制检出框信息**』，用来我**复制之后与你交流**"
            #   ✓）：主窗设它 = `_on_pick_menu` ✓，参数同样是**加工域坐标** ✓。
            #   ⚠ 与左键**同一份命中逻辑** ✗ —— 命中判在**主窗**（`pick_box_at` ✓），这一层只
            #     负责"把点报上去" ✓（它拿不到检出框 ✓）。
            self._on_context = None
            # ⭐⭐⭐⭐⭐ **ROI"拖一个框"那套**（用户 2026-10-07 ✓ 原话："**只在限定的区域计算（因为
            #   这是部分弹窗）**" ✓✓）：
            #   · `_roi_mode` = 主窗点了「框选区域」⇒ 左键**变成拖框**（点选框那套**让位** ✓）；
            #   · `_roi_from/_roi_to` = 拖框的起/终点（**加工域**坐标 ✓ 与判据同一把尺 ✓）；
            #   · `_on_roi` = 主窗设它 = `_on_roi_picked` ✓（松手时把**加工域矩形**报上去 ✓
            #     —— 这一层**只负责画与报** ✗，生效/落盘/重算全在主窗 ✓ 好测 ✓）。
            self._roi_mode = False
            self._roi_from = None
            self._roi_to = None
            self._on_roi = None

        def set_roi_mode(self, on):
            """进出"框选区域"模式（⚠ 光标也跟着变 ⇒ 一眼看得出现在在拖框 ✓）。"""
            self._roi_mode = bool(on)
            self._roi_from = self._roi_to = None
            self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
            self.update()

        def setText(self, s):
            self._text = str(s)
            self.update()

        def set_image(self, qimage, sx=1.0, sy=1.0):
            self._pix = QPixmap.fromImage(qimage)
            self._text = ""
            self._sx = float(sx) or 1.0
            self._sy = float(sy) or 1.0
            self.update()

        def _geo(self):
            """返回 `(zoom, ox, oy)`：缩放倍数 + 图像(0,0)在视图里的位置。"""
            if self._pix is None:
                return 1.0, 0.0, 0.0
            pw, ph = float(self._pix.width()), float(self._pix.height())
            if pw <= 0 or ph <= 0:
                return 1.0, 0.0, 0.0
            if self._fit:
                z = min(self.width() / pw, self.height() / ph)
                return z, (self.width() - pw * z) / 2.0, (self.height() - ph * z) / 2.0
            return self._zoom, self._off.x(), self._off.y()

        def _img_at(self, pos):
            """视图坐标 → 图像坐标（在图像外 ⇒ None ✓）。"""
            if self._pix is None:
                return None
            z, ox, oy = self._geo()
            ix = (pos.x() - ox) / z
            iy = (pos.y() - oy) / z
            if 0.0 <= ix < self._pix.width() and 0.0 <= iy < self._pix.height():
                return (ix, iy)
            return None

        def paintEvent(self, ev):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor("#101010"))
            if self._pix is None:
                if self._text:
                    p.setPen(QColor("#ccc"))
                    p.drawText(self.rect(), Qt.AlignCenter, self._text)
                p.end()
                return
            z, ox, oy = self._geo()
            pw, ph = self._pix.width() * z, self._pix.height() * z
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawPixmap(QRectF(ox, oy, pw, ph), self._pix, QRectF(self._pix.rect()))
            # ⭐ **坐标 overlay**（左上角 ✓ 持续显示 ✓）—— 加工域坐标（与后端同一把尺 ✓）
            if self._mouse_img is not None:
                _mx = int(round(self._mouse_img[0] / self._sx))
                _my = int(round(self._mouse_img[1] / self._sy))
                txt = "x=%d  y=%d" % (_mx, _my)
                p.setPen(QColor("#00ff88"))
                _fm = p.fontMetrics()
                _tw = _fm.horizontalAdvance(txt)
                p.fillRect(QRectF(6, 6, _tw + 14, 24), QColor(0, 0, 0, 170))
                p.drawText(QPointF(13, 23), txt)
                # 十字标在鼠标处
                mx, my = self._mouse.x(), self._mouse.y()
                p.drawLine(QPointF(mx - 7, my), QPointF(mx + 7, my))
                p.drawLine(QPointF(mx, my - 7), QPointF(mx, my + 7))
            # ⭐⭐⭐ **正在拖的 ROI 预览**（用户 2026-10-07 ✓ "框选区域" ✓）：白框 ✓
            #   ⚠ 只画**正在拖**的那一下 ✗ —— **已生效**的框是后端画在图像上的 ✓（见 `draw()` ✓），
            #     在这儿再画一遍会**两层线**（还容易对不齐 ✗）。
            #   ⚠ 换算：**加工域 → 视图**：`视图 = 偏移 + 加工域 × _sx × zoom` ✓（`_sx` 是"原图/加工"
            #     那一档 ✓ 与 `_img_at` 的除法**正好互逆** ✓）。
            if self._roi_from is not None and self._roi_to is not None:
                _p0 = (ox + self._roi_from[0] * self._sx * z,
                       oy + self._roi_from[1] * self._sy * z)
                _p1 = (ox + self._roi_to[0] * self._sx * z,
                       oy + self._roi_to[1] * self._sy * z)
                p.setPen(QColor("#ffffff"))
                p.drawRect(QRectF(QPointF(min(_p0[0], _p1[0]), min(_p0[1], _p1[1])),
                                  QPointF(max(_p0[0], _p1[0]), max(_p0[1], _p1[1]))))
            p.end()

        def mousePressEvent(self, ev):
            # ⭐ **按住中键拖动画面**（用户 2026-10-02 ✓）：以按下那一刻的几何为起点，
            #   移动量直接加到图像偏移上 ✓（在"适应窗口"模式下按下 ⇒ 先切到手动模式 ✓，
            #   以当前实际缩放/偏移为起点 ✓ 松手保持 ✓ 滚轮缩放/双击适应照常 ✓）。
            if ev.button() == Qt.MiddleButton and self._pix is not None:
                z, ox, oy = self._geo()
                self._fit = False
                self._zoom = z
                self._off = QPointF(ox, oy)
                self._pan_from = QPointF(ev.pos())
                self._pan_off = QPointF(ox, oy)
                self.setCursor(Qt.ClosedHandCursor)
                ev.accept()
                return
            # ⭐⭐⭐⭐⭐ **"框选区域"模式下的左键 = 拖框**（用户 2026-10-07 ✓ 见 `_roi_mode` 那段 ✓）
            #   —— ⚠⚠ **必须排在"点选框"前面** ✗（否则一按就被点选框那套吃掉 ✓）。
            if (self._roi_mode and ev.button() == Qt.LeftButton
                    and self._mouse_img is not None):
                self._roi_from = (self._mouse_img[0] / self._sx,
                                  self._mouse_img[1] / self._sy)
                self._roi_to = self._roi_from
                self.update()
                ev.accept()
                return
            # ⭐⭐⭐ **左键 = 点选检出框**（用户 2026-10-03 ✓）：把**加工域坐标**报给主窗 ✓
            #   （命中/高亮/状态栏全在主窗做 ✓ 这一层不碰数据 ✓）。
            if ev.button() == Qt.LeftButton and self._mouse_img is not None:
                self._mouse = QPointF(ev.pos())
                _f = getattr(self, "_on_pick", None)
                if callable(_f):
                    _f((self._mouse_img[0] / self._sx, self._mouse_img[1] / self._sy))
                    ev.accept()
                    return
            # ⭐⭐⭐ **右键 = 选中 + 弹菜单**（用户 2026-10-03 ✓ 见 `_on_context` 那段说明 ✓）：
            #   同样只把**加工域坐标**报给主窗 ✓（命中 / 选中 / 菜单 / 剪贴板全在主窗 ✓）。
            if ev.button() == Qt.RightButton and self._mouse_img is not None:
                self._mouse = QPointF(ev.pos())
                _f = getattr(self, "_on_context", None)
                if callable(_f):
                    _f((self._mouse_img[0] / self._sx, self._mouse_img[1] / self._sy))
                    ev.accept()
                    return
            super().mousePressEvent(ev)

        def mouseMoveEvent(self, ev):
            # ⭐ **正在拖 ROI**（用户 2026-10-07 ✓）：更新终点即可 ✓（预览在 `paintEvent` 里画 ✓）
            if self._roi_from is not None:
                self._mouse = QPointF(ev.pos())
                self._mouse_img = self._img_at(self._mouse)
                if self._mouse_img is not None:
                    self._roi_to = (self._mouse_img[0] / self._sx,
                                    self._mouse_img[1] / self._sy)
                self.update()
                ev.accept()
                return
            if self._pan_from is not None:
                self._off = QPointF(self._pan_off.x() + (ev.pos().x() - self._pan_from.x()),
                                    self._pan_off.y() + (ev.pos().y() - self._pan_from.y()))
                self._mouse = QPointF(ev.pos())
                self._mouse_img = self._img_at(self._mouse)
                self.update()
                ev.accept()
                return
            self._mouse = QPointF(ev.pos())
            self._mouse_img = self._img_at(self._mouse)
            self.update()
            super().mouseMoveEvent(ev)

        def mouseReleaseEvent(self, ev):
            # ⭐ **松手 ⇒ 把这一框报上去**（**加工域**坐标 ✓ 归一成 `x0<x1, y0<y1` ✓）——
            #   ⚠ 生效 / 落盘 / 重算全在主窗（`_on_roi_picked` ✓ 见它 ✓）。
            if ev.button() == Qt.LeftButton and self._roi_from is not None:
                _a = self._roi_from
                _b = self._roi_to or self._roi_from
                self._roi_from = self._roi_to = None
                _f = getattr(self, "_on_roi", None)
                if callable(_f):
                    _f((min(_a[0], _b[0]), min(_a[1], _b[1]),
                        max(_a[0], _b[0]), max(_a[1], _b[1])))
                self.update()
                ev.accept()
                return
            if ev.button() == Qt.MiddleButton and self._pan_from is not None:
                self._pan_from = None             # 松手 ⇒ 停在当前位置 ✓
                self.setCursor(Qt.ArrowCursor)
                ev.accept()
                return
            super().mouseReleaseEvent(ev)

        def leaveEvent(self, ev):
            self._mouse = None
            self._mouse_img = None
            self.update()
            super().leaveEvent(ev)

        def wheelEvent(self, ev):
            if self._pix is None:
                return
            _step = ev.angleDelta().y()
            if _step == 0:
                return
            factor = 1.25 if _step > 0 else (1.0 / 1.25)
            z, ox, oy = self._geo()
            nz = max(0.05, min(20.0, z * factor))
            # 以鼠标为锚点缩放：保持鼠标下那一点不动 ✓
            mx, my = ev.pos().x(), ev.pos().y()
            ix = (mx - ox) / z
            iy = (my - oy) / z
            self._fit = False
            self._zoom = nz
            self._off = QPointF(mx - ix * nz, my - iy * nz)
            self._mouse = QPointF(ev.pos())
            self._mouse_img = self._img_at(self._mouse)
            self.update()
            ev.accept()

        def mouseDoubleClickEvent(self, ev):
            self._fit = True                    # 双击 ⇒ 适应窗口（自动缩放居中 ✓）
            self.update()
            super().mouseDoubleClickEvent(ev)

    class DemoWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("测谎演示 · 红点=鼠标 / 绿圈=目标　[代码 %s]"
                                % code_stamp())
            self.resize(1180, 800)
            self.frames = []
            self.ts = []                       # 每帧**真实时间戳**（秒 ✓ GIF 不等间隔 ✓）
            self.dur = 0.0                     # 总时长（= ts[-1] ✓）
            self.is_gif = False                # 当前载入的是 **GIF** 还是视频（措辞用 ✓）
            self.fps = 30.0
            self.scale = (1.0, 1.0)
            self.runner = None
            self.worker = None
            self.dets = None
            self.results = None                # ⭐ 整段演算表（播放/拖动都查它 ✓）
            # ⭐⭐⭐ **右列（KF 位置驱动）的演算表**（用户 2026-10-03 ✓ 原话："当『KF预测』开启时，
            #   将**窗口分为两列视图**，左边跑经典右边跑 KF，这样对照才有意义" ✓）：勾了"显示
            #   KF 预测"才有 ✓（那一路 `Runner(kf_pos=True)` ✓ 真用 KF 位置跑 ✓）——
            #   两列**各有一份 `LieTracker`** ⇒ 状态完全隔离、互不污染 ✓。
            self.results_kf = None
            self.runner_kf = None
            self.names = None                  # 类别名（画框标签用 ✓ 第一次演算时**后台**取 ✓）
            self._names_th = None              # ⚠ 那个后台线程（见 `_fetch_names` ✓ 只起一次 ✓）
            self.i = 0
            self.t = 0.0
            self.rec = None
            self.playing = False
            # ⭐⭐⭐⭐⭐ **播放的"墙钟锚"**（用户 2026-10-07 ✓ 见 `play_delay_ms` ✓）——
            #   `(起点时刻, 起点帧号)` ⇒ "这一拍该等多久 = 目标时刻 − 现在" ✓
            #   （⚠ 老写法每拍**多等一个整间隔** ⇒ 实测只有 **0.65 倍速** ✗ = 用户说的"没原速" ✓）。
            self._play_t0 = None
            self._play_i0 = 0
            self.warm_n = 0
            self.warm_total = 0
            self._reuse_dets = False       # "应用"重算时 = True（复用检测框 ✓）

            tb = QToolBar("main")
            self.addToolBar(tb)
            self.btn_open = QPushButton("选择视频…")
            self.btn_open.clicked.connect(self.open_dialog)
            tb.addWidget(self.btn_open)
            # ⭐ **选择 GIF**（用户 2026-09-30 改 ✓ 原话："移除图片集的播放方式，改为
            #   『选取 gif』" ✓）：选一个 `.gif` 动图（素材放 `datasets/liedetectorgifs/` ✓）
            #   ⇒ 按 **GIF 自带的每帧延时**当视频放 ✓（这条路替掉了原来的"选图片集" ✗）。
            self.btn_gif = QPushButton("选择 GIF…")
            self.btn_gif.clicked.connect(self.pick_gif)
            tb.addWidget(self.btn_gif)
            # ⭐⭐⭐⭐⭐ **限定区域（ROI ✓ 用户 2026-10-07 ✓ 原话："**只在限定的区域计算（因为
            #   这是部分弹窗）**" ✓✓）**：
            #   · `框选区域`（**可勾**）⇒ 勾上后**在画面上拖一个框** ⇒ 只在这个框里算 ✓；
            #   · `清除区域` ⇒ 恢复整幅 ✓（= 改动前逐位一致 ✓）。
            self.btn_roi = QPushButton("框选区域")
            self.btn_roi.setCheckable(True)
            self.btn_roi.setToolTip(
                "点一下 ⇒ 开**带放大镜的框选窗**（放大镜 / 像素网格 / +/- 倍数 / Esc 取消 ✓\n"
                "与**数据工作台**那套是**同一份代码** ✓）：框住「要算的那块」⇒ 只在这个框里算 ✓\n"
                "（检测框 / 白块 / 轨迹 / 群体 ✓ 框外**一概看不见** ✓）\n"
                "⚠ 框外画面上会**置灰 ＋ 模糊** ✓（一眼看出它在生效 ✓）\n"
                "⚠ 这是给「整屏录像里只有一块弹窗」那种片子用的 ✓\n"
                "（例：10月7日.mp4 满屏蓝天白云 ⇒ 全幅找白会把云当白块 ✗）")
            self.btn_roi.toggled.connect(self._on_roi_mode)
            tb.addWidget(self.btn_roi)
            self.btn_roi_clear = QPushButton("清除区域")
            self.btn_roi_clear.setToolTip("清除限定区域 ⇒ 恢复整幅 ✓（回到改动前的行为 ✓）")
            self.btn_roi_clear.clicked.connect(self._on_roi_clear)
            tb.addWidget(self.btn_roi_clear)
            self.btn_play = QPushButton("播放")
            self.btn_play.clicked.connect(self.toggle_play)
            self.btn_play.setEnabled(False)
            tb.addWidget(self.btn_play)
            tb.addWidget(QLabel(" 速度 "))
            self.cmb_speed = QComboBox()
            for txt, v in (("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0),
                           ("2×", 2.0)):
                self.cmb_speed.addItem(txt, v)
            self.cmb_speed.setCurrentIndex(2)
            #   ⚠⚠ **换速度要"重锚"** ✗✗（用户 2026-10-07 ✓ 见 `play_delay_ms` ✓）：墙钟锚是
            #     "起点时刻 ＋ 起点帧号" ✓ ⇒ 换了速度**必须**把锚挪到当前帧 ✗，不然会按新速度
            #     从**起点**重算 ⇒ 一旦调慢 ⇒ 每拍都算出"早就该到" ⇒ 一路狂追 ✗。
            self.cmb_speed.currentIndexChanged.connect(self._on_speed_changed)
            tb.addWidget(self.cmb_speed)
            self.chk_dets = QCheckBox("用检测器")
            self.chk_dets.setChecked(not args.no_dets)

            self.chk_dets.setToolTip(
                "关掉 = 纯残差路线（M2a）；开着 = 加检测通道（M2b：建档种子 / 校验 / 吸附 ✓；"
                "⚠ 复活已于 2026-10-02 停用 ✓）好对比 ✗ 切换会重跑预热 ✓")
            self.chk_dets.stateChanged.connect(self.reload_current)
            # ⚠⚠⚠ **必须把这个 action 留下来** ✗✗（用户 2026-10-07 ✓）：速度跟踪档要它把这一项
            #   **从工具栏上真的拿掉** ✓ —— `setVisible(False)` **没用** ✗✗（**实测**：`QToolBar`
            #   的布局下一拍 `processEvents` 就把它**显示回来** ✓ 探针实测 ✓）；顺带记下
            #   "它后面那个 action" ⇒ 回来时**插回原位** ✓（工具栏顺序一个字不变 ✓）。
            self.act_dets = tb.addWidget(self.chk_dets)
            _acts = tb.actions()
            _i = _acts.index(self.act_dets)
            self.act_dets_after = _acts[_i + 1] if (_i + 1) < len(_acts) else None
            self.chk_loop = QCheckBox("循环")
            self.chk_loop.setChecked(True)
            tb.addWidget(self.chk_loop)
            # ⭐⭐⭐ **图层开关：自己取舍**（用户 2026-10-03 ✓ 原话："**这些细细的蓝色轨迹线是什么？
            #   我从未要求加过**" ✓）—— 画面上一旦东西多了就分不清谁是谁 ✗ ⇒ 与其我猜 ✗，
            #   不如把这几个图层**交给你自己开/关** ✓。
            #   ⚠ 默认值是按你的反馈定的 ✓：
            #     · **检出框** = 关 ✗（YOLO 的蓝框一大堆 ✓ 你没要求过 ✓ 需要时再开 ✓）；
            #     · **候选圈/分数** = 开 ✓（运动分离的**核心依据** ✓ 不开就没法核对判据 ✓）；
            #     · **相对群体轨迹线** = 开 ✓（就是你点名要的那条 ✓ 只画目标那一条 ✓）。
            self.chk_boxes = QCheckBox("检出框")
            # ⚠ **默认改成"开"** ✗（用户 2026-10-03："**每个假目标检出框**加蓝色箭头显示**瞬时绝对
            #   速度**，这个没看到" ✓ ⇒ 他要的正是"**看得见检出框 + 框上的蓝箭头**" ✓ ⇒
            #   上一版我默认关掉是猜错了 ✗）。
            self.chk_boxes.setChecked(True)
            self.chk_boxes.setToolTip(
                "画不画 **YOLO 检出框**（蓝色细矩形 ✓ 带 `shape 0.98` 标签 ✓）。\n\n"
                "· ⚠ 它**不影响任何判定** ✓ 纯显示 ✓；\n"
                "· 默认**关** —— 之前它满屏都是（十几~二十个框 ✗）⇒ 把它关掉，画面清爽得多 ✓；\n"
                "· 想核对「算法拿到的输入对不对」时再打开 ✓。")
            self.chk_boxes.stateChanged.connect(self._on_layer_toggle)
            tb.addWidget(self.chk_boxes)
            # ⭐⭐⭐ **「显示检出框编号」独立开关**（用户 2026-10-04 ✓ 原话："**加开关「显示检出框
            #   编号」**" ✓）—— ⚠ 他之前抱怨过"**左上角这些框太挡视野**" ✓；而这些**框上的
            #   标签**（`shape 0.98 #22  m0.82` ✓）比图例更贴目标、**更挡** ✗
            #   ⇒ 但"编号"本身是他**点名要的**（"没有看到标出 id，希望在检出框上写出来" ✓）
            #     ⇒ 不能整个删 ✗ ⇒ **给它一个独立开关** ✓。
            #   ⚠ 与「检出框」的关系：**框归框、字归字** ✗✗ ——
            #     `boxes` 管"框 + 蓝箭头" ✓、`box_id` 只管"**框上面那行字**" ✓
            #     ⇒ 于是能组成他想要的任意组合：① 全开（默认 ✓）；② **只留框、不要字**（截图用 ✓）；
            #        ③ 关「检出框」⇒ 框和字一起走 ✓（`box_id` 就无所谓了 ✓）。
            #   ⚠⚠⚠ **语义（2026-10-04 用户更正后 ✓）**：关掉它 = **画面上一块编号标签都不剩** ✗
            #     —— 原来的 `shape 0.98 #22  m0.82`（**框中间那行** ✓）、"睡着的"那块
            #     `#22 推的`、**候选圈旁边那块 `#13`**（"**没框时候残留的**" ✓）三处**一起走** ✓
            #     （他原话：「**不仅仅是不显示编号，框中间以及没框时候残留的 lable 都不要显示**」✓）。
            #   ⚠ ⚠ **不动的**（它们不是"标签" ✓）：**框本体** ✓、**蓝箭头** ✓、
            #     **灰框**（= "目标大概在这儿"的位置标记 ✓ 不是编号 ✗）、**分数牌** ✓、
            #     点选高亮那套（归属黄框 ✓ 那是**你主动点**才有的 ✓）。
            self.chk_boxid = QCheckBox("检出框编号")
            self.chk_boxid.setChecked(True)
            self.chk_boxid.setToolTip(
                "画不画**画面上那些编号标签**。\n\n"
                "· 关掉 ⇒ **三处一起消失**：① 框中间那行 `shape 0.98 #22 m0.82`；\n"
                "  ② 框没了之后那个 `#22 推的`；③ 候选圈旁边那块 `#13`（**没框时赖着的就是它**）；\n"
                "· ⚠ **不会动的**：框本体、蓝箭头、**灰框**（那是「目标大概在这儿」的位置标记，不是编号）、\n"
                "  分数牌 ⇒ 只是**把编号清干净** ✓ **截图 / 录屏时画面最干净** ✓；\n"
                "· ⚠ 单独关「检出框」也会连字一起去 ✗（字是挂在框上的 ✓）—— "
                "想「**留框不要字**」就关**这个**（`boxes` 保持开 ✓）；\n"
                "· ⚠ 编号是他要看的东西（`#22` = 这一格算到哪个目标头上 ✓）⇒ 平时建议**开着** ✓。")
            self.chk_boxid.stateChanged.connect(self._on_layer_toggle)
            tb.addWidget(self.chk_boxid)
            # ⚠⚠⚠ **拆成两个** ✗✗（用户 2026-10-04 ✓ 原话："**把 相对轨迹 勾选框 拆分成
            #   「真目标轨迹」「鼠标轨迹」**" ✓✓）—— ⚠ 原来一个"相对轨迹"管着**两条线** ✗
            #   ⇒ 想只看一条时**做不到** ✓（而他真正盯的是**真目标那条** ✓ 鼠标那条只是结果指标 ✓）。
            #   ⚠⚠ 两条线的**坐标系也不同**（都是"减掉相机" ✓ 但一个是目标、一个是鼠标 ✓）
            #     ⇒ 分开开关才不会互相干扰阅读 ✓。
            self.chk_rtgt = QCheckBox("真目标轨迹")
            self.chk_rtgt.setChecked(True)
            self.chk_rtgt.setToolTip(
                "画不画**真目标那条「相对群体」的轨迹线**（橙色 ✓ 粗线 ✓）。\n\n"
                "· 读法：**绕着原地打转** ⇒ 它跟大家一起动（假目标 ✗）；**朝一个方向延伸** ⇒\n"
                "  它真在相对群体移动（真目标 ✓✓）；\n"
                "· ⚠ 它画的是**减掉相机**之后的位置 ✓ ⇒ 不要把「圆心」往上套 ✗\n"
                "  （圆心是**画面坐标**、含相机 ✓ 两者本来就差一个相机位移 ✓）。")
            self.chk_rtgt.stateChanged.connect(self._on_layer_toggle)
            tb.addWidget(self.chk_rtgt)
            self.chk_rmouse = QCheckBox("鼠标轨迹")
            self.chk_rmouse.setChecked(True)
            self.chk_rmouse.setToolTip(
                "画不画**鼠标（模拟光标）那条「相对群体」的轨迹线**（暗黄 ✓ 细线 ✓）。\n\n"
                "· 它是**结果指标** ✓：鼠标跟对了真目标 ⇒ 会走出一条和真目标同向的线 ✓；\n"
                "  被假目标带着跑 ⇒ **绕在原地** ✗；\n"
                "· ⚠ 和「真目标轨迹」是**两条线** ✗（颜色、粗细都不同 ✓）⇒ 想只看目标那条，\n"
                "  把这个勾掉即可 ✓。")
            self.chk_rmouse.stateChanged.connect(self._on_layer_toggle)
            tb.addWidget(self.chk_rmouse)
            # ⭐⭐⭐ **信息面板开关**（用户 2026-10-04 ✓ 原话："**左上角这些框里是什么，
            #   太挡视野了**" ✓✓）—— 一个勾同时管**左上角那两块**：
            #   · `t=2.50s track hit=100%` 那行 **HUD**（时间 / 状态 / 命中率 ✓）；
            #   · 它下面那块**黑底图例面板**（颜色含义 + `cam/noise/cands` + `SEL` + Top3 ✓）。
            #   ⚠ 默认**开** ✓ —— 图例是"看懂画面"的钥匙（不解释颜色就不知道谁是谁 ✓）；
            #     但觉得挡视野时，勾掉即可 ⇒ **画面立刻只剩内容本身** ✓。
            #   ⭐ 2026-10-04 用户原话：「加个开关『显示图例』 通俗的把每种标记代表什么
            #   显示出来，格式为：{icon}:{解释}」⇒ 复用这个开关（原「信息面板」改名 ✓）、
            #   删掉旧的数据行（cam/noise/SEL/top3 ✓ 底部日志区已有 ✓）+ 旧图例行 ✓。
            self.chk_panel = QCheckBox("显示图例")
            self.chk_panel.setChecked(True)
            self.chk_panel.setToolTip(
                "画不画**右侧信息栏**（图例 + 实时信息 ✓）：\n"
                "· 图例 = motion 标记的通俗解释（绿圈/白点圈/白箭头/橙暗黄线/蓝洋红橙灰框）；\n"
                "· 实时信息 = 相机位移·噪声底·候选数 ＋ 目标 ＋ 分数前 3 候选。\n\n"
                "⚠ 纯显示，**不影响任何判定**；觉得挡视野就关掉。")

            self.chk_panel.stateChanged.connect(self._on_layer_toggle)
            tb.addWidget(self.chk_panel)
            # ---- ⭐ 两个拖动条（用户 2026-09-30 要求 ①③）：蒙版透明度 / 箭头拖尾 ----
            tb.addWidget(QLabel("  蒙版 "))
            self.sld_mask = QSlider(Qt.Horizontal)
            self.sld_mask.setRange(0, 80)
            self.sld_mask.setValue(int(args.mask * 100))
            self.sld_mask.setFixedWidth(110)
            self.sld_mask.setToolTip(
                "灰蒙版不透明度（框外压暗、**框里挖洞保持原亮度** ✓ —— ⚠ 挖洞**只给检出框** ✓\n"
                "目标圆不再挖洞（2026-10-02 ✓）；**所有线条一律鲜亮** ✓）")
            self.sld_mask.valueChanged.connect(self.on_view_change)
            tb.addWidget(self.sld_mask)
            self.lbl_mask = QLabel("%d%%" % self.sld_mask.value())
            tb.addWidget(self.lbl_mask)
            # ---- ⭐ **记住上次的配置**（用户 2026-10-01 ✓ 原话："让窗口能够记住我上次的
            #   配置" ✓）：这些开关/拖动条**离开窗口时**写回 `config/ui.yaml` 的 `lie_demo`
            #   段（按客户端唯一一份 ✓）；"应用"按钮那条在 `apply_path_ms` 里写 ✓。
            #   ⚠ 拖动条用 `sliderReleased`（不是 `valueChanged` ✗ —— 拖一下会写几十次文件 ✗）。
            self.cmb_speed.currentIndexChanged.connect(self._cfg_save)
            self.chk_dets.stateChanged.connect(self._cfg_save)
            self.chk_loop.stateChanged.connect(self._cfg_save)
            self.sld_mask.sliderReleased.connect(self._cfg_save)
            # ⚠ **"箭头长度"拖动条已删**（用户 2026-09-30 第 A2 条 ✓ 原话："取消自定义
            #   箭头长度拖动条配置，速度向量箭头现在**强制指向以当前速度预计下一帧到达的
            #   位置**" ✓）—— 箭头长度就是**这一拍的位移**（所见即所测 ✓），没什么可调的 ✓。

            # ---- ⭐ **配置行：轨迹预测窗口时间**（用户 2026-10-01 ✓ 原话："增加参数
            #   "轨迹预测窗口时间"（就是你刚说的这 500ms），在输入框右边加按钮"应用"，
            #   点击后重新运算并生效" ✓）----
            #   它 = `LieTracker.path_ms`：**切向预测回看多久的轨迹**（默认 500ms ✓
            #   10fps ⇒ 5 个采样点 ✓ 少于 3 个拟不出曲率 ✗、多于 ~7 个会把转弯前的旧
            #   方向带进来 ✗）⇒ 调大 = 更平滑更滞后、调小 = 更灵敏更抖 ✓。
            #   ⚠ 位置：**交互区（工具栏）下面、画面上面** ✓（交互区下加一行 ✓）。
            cfgrow = QHBoxLayout()
            self._row1 = cfgrow            # ⭐ 存起来：切运动分离时**逐项**收掉用不上的那几个
            #   （⚠ 第 1 行是**混着**的 —— "模式"/"鼠标跟随效率倍率"两边都要 ✓ ⇒ **整行藏不得** ✗）
            cfgrow.setContentsMargins(6, 2, 6, 2)
            # ⭐⭐⭐ **第二行**（用户 2026-10-02 ✓ 原话："与『融合判定阈值』一起的**共 3 个参数**
            #   一起放到**第二行**" ✓）：`融合框判定阈值` / `分离判定阈值` /
            #   `融合框继承面积限制` 三个"融合系"参数放这一行 ✓ —— 它们是一组（判定融合 →
            #   挑融合候选 → 结束融合）✓ 与第一行那些"轨迹/容差/上板"参数分开读 ✓。
            # ---- ⭐⭐⭐ **算法模式**（用户 2026-10-03 ✓ 原话："你可以在测谎演示里加个**模式下拉
            #   列表**，新的算法模式选中时，**不必要的参数配置都隐藏**" ✓）--------
            cfgrow.addWidget(QLabel("模式"))
            self.cmb_mode = QComboBox()
            self.cmb_mode.addItem("经典（白块 + 纹理）", "classic")
            self.cmb_mode.addItem("运动分离（新）", "motion")
            # ⭐⭐⭐⭐⭐ **第三档：速度跟踪**（用户 2026-10-07 ✓ 本轮"干净底盘" ✓）——
            #   ⚠⚠ **绝不许改动 `classic` 与 `motion` 两个模式的任何行为** ✗✗（用户原话"零污染"✓）：
            #     它俩的取值 / 分支 / 文案**一个字不改** ✓，新档只**多一条路** ✓。
            self.cmb_mode.addItem("速度跟踪", "velocity")
            self.cmb_mode.setToolTip(
                "**经典**：在纹理里认那个白色图形（原来那套）。\n"
                "**运动分离（新）**：按你 2026-10-03 给的规律 —— **用「颜色」当观测、用「运动」\n"
                "当预测**：真目标开局是白的 ⇒ 直接找**最白的那块**（⚠ **对「波浪式」几何扭曲免疫** ✓\n"
                "因为扭曲改的是位置、不是亮度 ✓）；它逐渐透明 / 与假目标视觉融合时 ⇒ 用\n"
                "「上一拍位置 ＋ 速度」**外推**接着走（外推不读图像 ⇒ 波浪也影响不到它 ✓）。\n"
                "⚠ **不再做「对齐 / 帧间差分」** ✗（那条路被几何扭曲毁掉 ⇒ 实测残差补不掉一半 ✗）。\n\n"
                "⚠ 选新模式会把「融合/内砖/边归属/允许倒退」那几行参数**整行隐藏** ✓\n"
                "  （新模式不用它们 ✓）；切回去它们就回来 ✓。\n"
                "\n\n"
                "**速度跟踪**（用户 2026-10-07 ✓ 本轮只做「**干净底盘**」✗ 还没做新判据 ✗）：\n"
                "底盘 = ① 检出框**状态枚举**（上板 / 等待上板 / 融合 ✓）；② **编号**沿用轨迹号\n"
                "（整框在视野内 且 不压在真目标框上 ⇒ 发号 = 上板 ✓）；③ 两把「标准尺」的记账\n"
                "（标准面积 / 标准速度 ✓）。\n"
                "⚠⚠ 选中它 ⇒ 画面**做减法**：**只留 7 样东西**（见右栏那 7 条图例 ✓）——\n"
                "    别的（粉框 / 黄箭头 / 暗黄线 / 洋红细夹取框 / 唯一红框 / KF / HUD / 点选面板 /\n"
                "    右侧实时信息 / 底栏中段 / 日志区）**一律不画** ✓（项目纪律：**不许私自新增**\n"
                "    任何观测参数 / 图例条目 / 面板文字 / 底栏提示 ✗ 要加先问用户 ✓）。\n"
                "⚠ 改完要**重新点「应用」**（或重开素材）才按新模式演算。")
            self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
            cfgrow.addWidget(self.cmb_mode)
            cfgrow2 = QHBoxLayout()
            cfgrow2.setContentsMargins(6, 2, 6, 2)
            self._row2 = cfgrow2                  # ⭐ 存起来（新模式要整行隐藏 ✓）
            self.lbl_pms = QLabel("轨迹预测窗口时间")
            self.lbl_pms.setToolTip(
                "切向轨迹预测**回看多久**的轨迹（毫秒 ✓ 默认 %d ms）。\n"
                "· 调大 ⇒ 曲线更平滑、更滞后（转弯时会被旧方向拖着 ✗）；\n"
                "· 调小 ⇒ 更贴当前方向、更易被单帧抖动带偏 ✗。\n"
                "改完点「应用」⇒ **重新演算整段**并生效（检测框复用 ⇒ 不用重跑 YOLO ✓）。"
                % int(_PATH_MS_DEFAULT))
            cfgrow.addWidget(self.lbl_pms)
            # ⚠ **滚轮不许改参数**（项目规范 ✓）⇒ 一律 `NoWheelSpinBox`。
            self.sp_path_ms = NoWheelSpinBox()
            self.sp_path_ms.setRange(50, 5000)
            self.sp_path_ms.setSingleStep(50)
            self.sp_path_ms.setSuffix(" ms")
            self.sp_path_ms.setValue(int(_PATH_MS_DEFAULT))
            self.sp_path_ms.setToolTip(self.lbl_pms.toolTip())
            cfgrow.addWidget(self.sp_path_ms)
            # ---- ⭐⭐⭐ **融合框判定 IoU**（用户 2026-10-03 ✓ 原话："把『融合框判定阈值』换成
            #   『**融合框判定 IoU**』：**小于这个 IoU 才行**，之前配的 1.2 改成 **0.5**" ✓）----
            #   判据 = `IoU(检出框, 该砖的登记框) < 它` ⇒ 判红框**融合**（= 目标 + 假目标的并集 ✓）
            #   ⇒ **不做信念夹取**、绿圈按白箭头（预测）走 ✓。
            #   ⚠ **越小越严** ✓（要求框与砖**越不重叠**才算并集 ✓ —— 因为"两者基本同一格"的
            #     IoU 会很大 ✓ 那已不是并集、而是"砖自己那一格"✗ 就不该标融合 ✓）。
            cfgrow2.addWidget(QLabel("融合框判定 IoU"))
            self.sp_merge = NoWheelDoubleSpinBox()
            self.sp_merge.setRange(0.0, 1.0)
            # ⭐⭐ **精度到小数点后 2 位**（沿用用户 2026-10-01 的要求 ✓："需要往后再精确 1 位
            #   小数" ✓）：实测判决常常"卡线" ⇒ 1 位小数调不到那个点上 ✗；箭头步长 0.05 ✓。
            self.sp_merge.setSingleStep(0.05)
            self.sp_merge.setDecimals(2)
            self.sp_merge.setValue(float(_MERGE_IOU_DEFAULT))
            self.sp_merge.setToolTip(
                "IoU（检出框 ∩ 那块砖的登记框 ÷ 两者并集）**小于它** ⇒ 判「红框融合」\n"
                "（框 = 目标 + 假目标的并集）。\n\n"
                "· 默认 0.50；**越小越严**（要求框与砖越不重叠才算并集）；\n"
                "· 0 = 关（只剩「整圈绿圈在框内 + 面积与砖相等」那条规则）；\n\n"
                "融合时**不做信念夹取**，绿圈继续按白箭头（预测）走 ✓；\n"
                "**没标融合的检出框一律不给红框**（那一拍走预测态 ✓）。\n"
                "精度 0.01；改完点「应用」⇒ 重新演算整段并生效。")
            # ---- ⭐⭐⭐ **"融合框判定 IoU · 退出"= 迟滞下半段**（用户 2026-10-03 ✓ 第 1 步 ✓）----
            #   进用上面那个（严 ✓）、**守**用这个（松 ✓）：已在融合保持期时，要 `IoU > 它`
            #   （两框明显同一格 ✗ 不像并集了 ✓）才**掉出**融合 ✓ ⇒ 边界不再"进进出出"横跳 ✓。
            cfgrow2.addWidget(QLabel("退出 IoU"))
            self.sp_merge_out = NoWheelDoubleSpinBox()
            self.sp_merge_out.setRange(0.0, 1.0)
            self.sp_merge_out.setSingleStep(0.05)
            self.sp_merge_out.setDecimals(2)
            self.sp_merge_out.setValue(float(_MERGE_IOU_OUT_DEFAULT))
            self.sp_merge_out.setToolTip(
                "**已经在融合里**（保持期）时用的 IoU 阈值：`IoU 大于它` ⇒ 两框已明显是同一格\n"
                "⇒ **掉出融合**（不再当并集）。\n\n"
                "· 与左边那个配合 = **迟滞**：**进严（0.50）、守松（0.75）** ⇒ 边界不横跳；\n"
                "· 想退回单阈值 ⇒ 把它设成与左边**相同**的值；\n"
                "· 调大 ⇒ 更不容易掉出（融合段更长、更稳，但可能赖着不走）。\n\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow2.addWidget(self.sp_merge_out)
            cfgrow2.addWidget(self.sp_merge)
            # ---- ⭐⭐⭐ **分离判定阈值**（用户 2026-10-02 ✓ 原话："融合期间融合框的最大面积 >=
            #   融合期间融合框的最小面积 * x，**这个 x 做成配置参数『分离判定阈值』加在『融合框
            #   判定阈值』右边**" ✓）：分离信号的**第三条** ✓ —— 融合期**最大框面积 ÷ 最小框
            #   面积 ≥ 它** 才允许判分离 ✓（= 框确实"从并集缩回过砖大小" ✓）。
            #   调大 ⇒ 更苛刻（要缩得更多才算分离 ⇒ 融合保持得更久 ✓）；调小到 1.0 ⇒ 该条
            #   恒成立（等价于没有这一条 ✓ 回到上一版口径 ✓）。
            cfgrow2.addWidget(QLabel("分离判定阈值"))
            self.sp_sep = NoWheelDoubleSpinBox()
            self.sp_sep.setRange(1.0, 5.0)
            self.sp_sep.setSingleStep(0.05)     # 与"融合框判定阈值"同一手感 ✓
            self.sp_sep.setDecimals(2)
            self.sp_sep.setValue(float(_SEP_RATIO_DEFAULT))
            self.sp_sep.setToolTip(
                "分离信号的第三条：**融合期间融合框的最大面积 ÷ 最小面积 ≥ 它**，才判「分离」\n"
                "（= 融合框确实从并集大小缩回过砖大小）。\n\n"
                "· 调大 ⇒ 要求缩得更多才算分离 ⇒ 融合保持更久（分离更保守）；\n"
                "· 调到 1.00 ⇒ 这一条恒成立（等于没有它，回到上一版口径）；\n"
                "· ⚠ 若整段融合期框面积几乎没变（框本来就是砖大小 ⇒ 当初多半量错了），\n"
                "  这条会挡住「缩成砖」那条假分离 ✓\n\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow2.addWidget(self.sp_sep)
            # ---- ⭐⭐⭐ **融合框继承距离限制**（用户 2026-10-02 ✓ 口径定稿 ✓ 原话："『S = 框∩
            #   绿圈外接矩形 ÷ 绿圈外接矩形 >"融合框继承面积限制"』改为『**绿圈圆心到框最近
            #   一条边的垂直距离(绿圆半径比例) > "融合框继承距离限制"**』" ✓ 随后两次收紧口径：
            #   "这个距离**正数判定才有意义**" ✓ + "**圆心在框外 才判定，圆心在框内直接允许融合**"
            #   ✓）：**融合框候选的第一条闸** ✓ —— **圆心在框内 ⇒ 直接允许** ✓；**圆心在框外**
            #   ⇒ 才判"**出框量 ≤ 本项 × 半径**" ✓（本项 = **允许圆心出框的倍数半径** ✓）。
            #   （取代老那条面积比 S ✗ —— 那把尺跟框的尺寸耦合 ✗；也**取代**同日早先那版
            #    "圆心离最近边还得 ≥ 本项 × 半径" ✗ —— 那版框内也受限 ⇒ 实测 9月30日 帧 38
            #    圆心明明在框里、只因离下边 0.563 半径就被挡 ✗ = 用户点名 ✓）。
            # ⚠⚠⭐ **本项 2026-10-03 起已屏蔽**（用户原话："**有了这个逻辑，就不需要『融合框
            #   继承距离限制』了，先屏蔽相关逻辑**" ✓ —— "这个逻辑" = "分离前不许换内砖"✓
            #   `LieTracker._same_inner_brick` ✓）：判据在追踪器里由开关 `_INHERIT_DIST_ON`
            #   （= `False` ✓）短路成**恒放行** ✓ ⇒ 界面上**置灰** + 文案标注 ✓（免得调了不生效 ✗）；
            #   值照旧落盘/回填 ✓（开关拨回 `True` 即恢复 ✓）。
            cfgrow2.addWidget(QLabel("融合框继承距离限制（已屏蔽）"))
            self.sp_inherit = NoWheelDoubleSpinBox()
            self.sp_inherit.setRange(0.0, 3.0)
            self.sp_inherit.setSingleStep(0.05)
            self.sp_inherit.setDecimals(2)
            self.sp_inherit.setValue(float(_INHERIT_DIST_DEFAULT))
            self.sp_inherit.setEnabled(False)       # ⭐ 置灰（已屏蔽 ✓ 见上）
            self.sp_inherit.setToolTip(
                "⚠ **已屏蔽（2026-10-03）**：本项**不生效** —— 融合候选改由「分离前不许换内砖」\n"
                "（内砖同一性）+「允许倒退距离」两道把关。要恢复本项，把追踪器里的\n"
                "`perception/lie_tracker.py::_INHERIT_DIST_ON` 拨回 `True`。\n\n"
                "（以下为原口径说明，仅供回退参考）\n"
                "融合框候选的第一条闸 = **允许圆心出框的倍数半径**。\n\n"
                "· 圆心**在框内** ⇒ **直接允许**（不看本项）；\n"
                "· 圆心**在框外** ⇒ 才判：「出框量 ≤ 本项 × 绿圈半径」才允许；\n"
                "· 0.00（默认）⇒ 一点也不许出框（圆心必须落在框内，贴着边也算在内）；\n"
                "· 调大（如 0.3）⇒ 允许圆心出框「0.3 × 半径」以内 ⇒ 融合候选更多、更松。\n"
                "· 它不跟框的大小挂钩，只问「圆心离边多远」，用**绿圈半径**归一 ⇒\n"
                "  半径自己变了（实测 56~61px），判别也不跟着变。\n\n"
                "第二条闸不在这里配：它固定是「沿白箭头推进**第一个接触到的框**」\n"
                "（把绿圈沿白箭头方向推，最先撞上哪一格）。\n\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow2.addWidget(self.sp_inherit)
            # ---- ⭐⭐⭐ **融合框挑选允许倒退距离(绿圆半径比例)**（用户 2026-10-02 ✓ 原话：
            #   "『融合框挑选允许倒退距离(px)』改为『**融合框挑选允许倒退距离(绿圆半径比例)**』"
            #   ✓）：第二条闸的**容差** ✓ —— 落位把圆心在白箭头方向上的位移分量 ≥
            #   `−它 × 绿圈半径` 就算"安全" ✓（见 `LieTracker._merge_fix_safe` ✓）。
            #   · **0（默认）= 一点也不能倒退**（= 上一版的布尔判据 ✓ 行为一字不变 ✓）；
            #   · 调大 ⇒ 允许"倒退一点点"的框也当候选 ⇒ 候选变多、融合段更连续 ✓（代价：圆
            #     偶尔被落位往反方向拽几像素 ✓）。
            #   ⚠ 换算参考：旧配置 20px ÷ 实测半径中位 60.8 ≈ **0.33** ✓。
            cfgrow2.addWidget(QLabel("融合框挑选允许倒退距离(步数比例)"))
            self.sp_back = NoWheelDoubleSpinBox()
            self.sp_back.setRange(0.0, 5.0)
            self.sp_back.setSingleStep(0.05)
            self.sp_back.setDecimals(2)
            self.sp_back.setValue(float(_ALLOW_BACK_RATIO_DEFAULT))
            self.sp_back.setToolTip(
                "融合框候选第二条闸的容差：落位会把圆心挪出去多远 —— 若落点在**白箭头后方**\n"
                "（180° 扇形内），则要求位移长度 ≤ **本项 × 「本拍通常该走多远」**。\n\n"
                "⚠⚠ **那把尺是 `|白箭头| × dt`，不是绿圈半径** ——\n"
                "  用户 2026-10-03 明确要求改成这种「B 型 / 步数比例」\n"
                "  （原来的「半径比例」写法会随**播放倍速**漂，B 型不会）。\n"
                "  ⇒ 本项 1.0 = 「允许倒退 一步的距离」，**远小于**一个半径（半径约 60px，一步约 20px）。\n\n"
                "· 0（默认）⇒ 一点也不能往反方向挪（最保守，候选最少）；\n"
                "· 调大 ⇒ 允许轻微倒退的框也算候选 ⇒ 融合段更连续、更少「假分手」，\n"
                "  但那一拍圆心可能被落位往反方向拽几像素。\n\n"
                "⚠ 换算示例（实测）：某拍落位倒退 **34.0px**、一步 **21.1px** ⇒ **要 ≥ 1.61 才过**。\n"
                "配到多少合适，用「应用」重算后看底部融合日志里「门内没有能挑的格」少了多少。")
            cfgrow2.addWidget(self.sp_back)
            # ---- ⭐⭐⭐ **砖最多拥有融合框边数量**（用户 2026-10-02 ✓ 原话："增加参数：『**砖
            #   最多拥有融合框边数量**』(1~4)，注意在**多条边选取最近的前 x 条**" ✓）：
            #   融合框的"边归属"（`_edge_ownership` ✓）里，归砖的边**只保留距离最近的这么条**
            #   ✓ —— 更远的那几条改判"真目标提供的边" ✓（⇒ 落位五档 `_merge_edge_own` 里的
            #   `n` 就被它限制住 ✓）。
            #   · **4（默认）= 不截断**（老行为一字不变 ✓）；
            #   · 设 1~3 ⇒ "一块砖最多只能占住这么多条边" ⇒ "四假边 ⇒ 不落位"那档不再发生 ✓。
            cfgrow2.addWidget(QLabel("砖最多拥有融合框边数量"))
            self.sp_emax = NoWheelDoubleSpinBox()
            self.sp_emax.setRange(1.0, 4.0)
            self.sp_emax.setSingleStep(1.0)
            self.sp_emax.setDecimals(0)
            self.sp_emax.setValue(float(_EDGE_MAX_DEFAULT))
            self.sp_emax.setToolTip(
                "融合框四边的「边归属」里，**允许一块砖最多占住几条边**（1~4）。\n\n"
                "· 每条边看「它离**最近的某块砖**的对应边有多远」（线段对线段）⇒ 在\n"
                "  「噪声容差(px)」以内就算这条边归砖；\n"
                "· 若归砖的边**多于**本项 ⇒ 按距离从小到大**只留最近的前 x 条**，\n"
                "  其余改判「真目标提供的边」；\n"
                "· 4（默认）= 不截断（保持原样）；设 1~3 会把「四条边全归砖 ⇒\n"
                "  不落位、照预测走「那一档变成」还有真边可用 ⇒ 照样落位「。\n\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow2.addWidget(self.sp_emax)
            # ---- ⭐⭐⭐ **真目标预测最大速度倍率**（用户 2026-10-03 ✓ 原话："将**分离期（淡粉色框
            #   时期）**的**最大相对假目标群速度（白箭头）** = a × **假目标群体的标准速度**，a 为
            #   新配置『**真目标预测最大速度倍率**』默认 1.0" ✓）：分离期（**红框丢失**、演示窗画
            #   "淡粉接力框"那段预测期 ✓）里白箭头（`vel_rel` = 相对假目标群的速度 ✓）的**模长上限**
            #   = 本项 × **当拍实测群体速度模长** ✓（只压不抬 ✓ 方向不变 ✓）。
            cfgrow2.addWidget(QLabel("真目标预测最大速度倍率"))
            self.sp_svmax = NoWheelDoubleSpinBox()
            self.sp_svmax.setRange(0.0, 5.0)
            self.sp_svmax.setSingleStep(0.1)
            self.sp_svmax.setDecimals(2)
            self.sp_svmax.setValue(float(_SEP_VEL_MAX_DEFAULT))
            self.sp_svmax.setToolTip(
                "**分离期**（红框丢失、画淡粉接力框那段预测期）里，白箭头（相对假目标群的\n"
                "速度）的**模长上限** = 本项 × **当拍实测的群体速度模长**。\n\n"
                "· 1.00（默认）⇒ 真目标的预测速度最多与假目标群的移动速度一样快；\n"
                "· 调小（如 0.5）⇒ 分离期预测更保守、圆走得更慢；\n"
                "· 0 ⇒ 分离期白箭头模长压到 0（圆不再按预测前进，极端值）。\n"
                "· **只压不抬**：没超上限就一个字不动；方向不变 ⇒ 「白箭头 = 圆真的会走多少」\n"
                "  仍成立；群体速度判不出来（没观测）时**不夹**。\n\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow2.addWidget(self.sp_svmax)
            # ⭐⭐ **第二行统一"居左、不平铺"**（用户 2026-10-02 ✓ 原话："第二行的配置布局统一
            #   一下**居左不要平铺**" ✓）：末尾加一根弹簧 ⇒ 控件按各自 sizeHint 靠左排、余量
            #   留在右边 ✓（不加弹簧时，若某个控件带横向扩展策略就会被拉宽 ⇒ 看着像"平铺" ✗）。
            # ---- ⭐⭐⭐ **分离期那一对（用户 2026-10-03 ✓ 参数分档第 2 行）** ----------------
            #   一句话口径（用户 2026-10-03 ✓ **二次修订**）："**圆外接矩形与检出框相交的比例 > ___
            #   时，检出框与内砖 IoU < ___ 判定为融合**" ✓ —— ⚠ 原来那一句是"检出框与圆外接矩形
            #   **IoU** > ___"✗（分母 = **并集** ⇒ 框一大就摊薄 ⇒ 天生偏小 ✗ 实测同局面仅 0.35~0.5
            #   ✗）⇒ 现在分母换成 **圆外接矩形面积** ✓（单向 ✓ 同局面直接 1.0 ✓）。
            #   ⚠ **量级提醒**（务必知悉 ✗）：可用区从 0.3~0.4 抬到 **0.70~0.95** ✓。
            cfgrow2.addWidget(QLabel("分离期：圆外接矩形 相交比例 >"))
            self.sp_ring_sep = NoWheelDoubleSpinBox()
            self.sp_ring_sep.setRange(0.0, 1.0)
            self.sp_ring_sep.setSingleStep(0.05)
            self.sp_ring_sep.setDecimals(2)
            self.sp_ring_sep.setValue(float(_RING_COV_SEP_DEFAULT))
            self.sp_ring_sep.setToolTip(
                "**分离期**（红框丢失、画淡粉接力框那段预测期）判「这格算出并集了吗」的前半条：\n\n"
                "**（圆外接矩形 ∩ 检出框）÷ 圆外接矩形面积 > 本值**\n"
                "⇒ 才算「绿圈那一块地方大体落在框里」。\n\n"
                "⚠ 这是**单向比例**（分母 = 圆的那个方块），**不是 IoU**：圆整个落在框里 = 1.00；\n"
                "圆只落进去三成 = 0.30。\n"
                "⚠ 量级 ⇒ **可用区约 0.70~0.95**（默认 0.80 = 「八成落在框里」）——\n"
                "别再拿 IoU 时代的 0.35 来配（那等于「三成半落在框里就算」⇒ 形同虚设）。\n\n"
                "· 调小 ⇒ 更容易认成融合；调大 ⇒ 更严。\n"
                "· 半径还没学到那几拍 ⇒ 这条**自动放行**（不拦）。\n\n"
                "⚠ 这条**取代**了原来的「圆心±噪声容差 与框相交」那条判据。\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow2.addWidget(self.sp_ring_sep)
            cfgrow2.addWidget(QLabel("↔内砖 IoU <"))
            self.sp_brick_sep = NoWheelDoubleSpinBox()
            self.sp_brick_sep.setRange(0.0, 1.0)
            self.sp_brick_sep.setSingleStep(0.05)
            self.sp_brick_sep.setDecimals(2)
            self.sp_brick_sep.setValue(float(_BRICK_IOU_SEP_DEFAULT))
            self.sp_brick_sep.setToolTip(
                "**分离期**判融合的后半条：\n\n"
                "**IoU(检出框, 它压着的那块砖) < 本值** ⇒ 两框**不是同一格** ⇒ 才是「并集」。\n\n"
                "· 默认 0.50（**进入严** ⇒ 不容易误判成融合）；\n"
                "· 与「融合期」那一项之差 = **迟滞**（进严守松）⇒ 边界不来回横跳。\n\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow2.addWidget(self.sp_brick_sep)
            cfgrow2.addStretch(1)
            # ---- ⭐⭐⭐ **融合期那一对（用户 2026-10-03 ✓ 参数分档第 3 行）** ----------------
            #   同一句话口径，但落点是"**保持**"：满足 ⇒ **继续融合** ✓；**否则就算分离** ✓
            #   （= 迟滞的下半段"守松" ✓ 治"融合段太碎" ✓）。
            cfgrow3 = QHBoxLayout()
            cfgrow3.setContentsMargins(6, 2, 6, 2)
            self._row3 = cfgrow3                  # ⭐ 同上（新模式整行隐藏 ✓）
            cfgrow3.addWidget(QLabel("融合期：圆外接矩形 相交比例 >"))
            self.sp_ring_fuse = NoWheelDoubleSpinBox()
            self.sp_ring_fuse.setRange(0.0, 1.0)
            self.sp_ring_fuse.setSingleStep(0.05)
            self.sp_ring_fuse.setDecimals(2)
            self.sp_ring_fuse.setValue(float(_RING_COV_FUSE_DEFAULT))
            self.sp_ring_fuse.setToolTip(
                "**融合期**（已在融合保持里）复核的**前半条**（与上面「分离期」那一项同一把尺）：\n\n"
                "**（圆外接矩形 ∩ 检出框）÷ 圆外接矩形面积 > 本值** ⇒ 这条算过。\n\n"
                "⚠ 同一把尺：**单向比例**不是 IoU ⇒ 量级 = 0.70~0.95（默认 0.80 ✓ 见上一项说明）。\n\n"
                "⚠ 这一行与下一项**任一不满足** ⇒ **判分离**（红框消失、走淡粉预测态）。\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow3.addWidget(self.sp_ring_fuse)
            cfgrow3.addWidget(QLabel("↔内砖 IoU <"))
            self.sp_brick_fuse = NoWheelDoubleSpinBox()
            self.sp_brick_fuse.setRange(0.0, 1.0)
            self.sp_brick_fuse.setSingleStep(0.05)
            self.sp_brick_fuse.setDecimals(2)
            self.sp_brick_fuse.setValue(float(_BRICK_IOU_FUSE_DEFAULT))
            self.sp_brick_fuse.setToolTip(
                "**融合期**复核的**后半条**：\n\n"
                "**IoU(检出框, 它压着的那块砖) < 本值** ⇒ 还算「并集」。\n\n"
                "· 默认 0.70（**守松** ⇒ 已经融合的不容易掉）；\n"
                "· 与「分离期」那一项（0.50）之差 = **迟滞**；\n"
                "· **想临时关掉「融合期退出」** ⇒ 把它设成 **1.00**（等价于「只要框还包着砖就继续」）。\n\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow3.addWidget(self.sp_brick_fuse)
            # ⚠ "允许倒退距离"两期共用（用户第 2/3 行都写了 ✓）⇒ 第 3 行这一份与第 2 行
            #   **双向同步** ✓（改任一处、另一处跟着变 ✓ 见 `_sync_ab` ✓）。
            cfgrow3.addWidget(QLabel("允许倒退距离(步数比例)"))
            self.sp_ab_fuse = NoWheelDoubleSpinBox()
            self.sp_ab_fuse.setRange(0.0, 5.0)
            self.sp_ab_fuse.setSingleStep(0.05)
            self.sp_ab_fuse.setDecimals(2)
            self.sp_ab_fuse.setValue(float(_ALLOW_BACK_RATIO_DEFAULT))
            self.sp_ab_fuse.setToolTip(
                "与上面那一行**同一个参数**（两期共用 ✓ 改任一处另一处同步 ✓）：\n\n"
                "落位点在白箭头**后方 180° 扇形**内时，允许挪多远 = 本值 × **本拍该走多远**\n"
                "（B 型 ⇒ **帧率 / 播放倍速无关** ✓）。\n\n"
                "· 0 ⇒ 一点不许倒退；调大 ⇒ 允许「退一点点」的框也算安全。\n\n"
                "⚠ 只看「融合第 2 拍起」（进门那一拍按流程图不判 ✓）。\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow3.addWidget(self.sp_ab_fuse)
            # ⚠ 这一句会把"砖最多拥有融合框边数量"**从第 2 行迁到第 3 行** ✓（Qt 会把它
            #   从旧布局摘走 ✓ = 用户要的"第 3 行放它" ✓ 不重复出现 ✓）。
            cfgrow3.addWidget(self.sp_emax)
            cfgrow3.addStretch(1)
            # ⭐ 同步：`sp_back`（第 2 行）↔ `sp_ab_fuse`（第 3 行）——**同一个参数** ✓
            self._sync_ab_guard = False
            self.sp_back.valueChanged.connect(
                lambda v: self._sync_ab(self.sp_back, self.sp_ab_fuse, v))
            self.sp_ab_fuse.valueChanged.connect(
                lambda v: self._sync_ab(self.sp_ab_fuse, self.sp_back, v))
            # ---- ⭐⭐ **噪声容差(px)**（用户 2026-10-02 ✓ 原话："加容差，且将这个容差作为
            #   配置参数『噪声容差(px)』加在『融合框判定阈值』的右边，依旧点应用重算后生效" ✓；
            #   ⭐ 同日口径更新 ✓ 原话："**圆心在框内改成『圆心±噪声容差范围与框相交』**" ✓）：
            #   规则① 判的就是"**圆心 ± 它** 的范围（以圆心为中心的方块）与检出框**相交**"✓
            #   —— 吸收检出框边缘的亚像素噪声（实测帧 16 圆心只出框 0.7px ✗ 严格判会漏 ✓）；
            #   0 = 范围退化成点 = 严格"圆心在框内" ✓（同一把尺 ✓ 见 `_pos_range_hits` ✓）。
            # ---- ⭐⭐⭐ **第 4 行：运动分离（新）专用参数**（用户 2026-10-03 ✓ 原话："运动分离
            #   **没有任何参数要配吗？**" ✓）----------------------------------------------------
            #   ⚠ **分两档给** ✗ 不一股脑全给 ✓：
            #     · **手感 / 权衡档**（跟得紧 ↔ 稳 ✓ 按素材调 ✓）⇒ **放这一行** ✓；
            #     · **物理契约档**（"什么算一个块"：**面积带** / **填充度** ✓ 见
            #       `perception/lie_motion.py` 顶上 ✓）⇒ **不放** ✗ —— 那几个是按"本素材那个白星的
            #       形状"定的 ✓ 调了会**认错东西** ✗（要换素材，改那几个常量 ✓ 比放界面上安全 ✓）。
            #   ⚠ 5 个**都是"点『应用』才生效"**（与其余参数同一个约定 ✓ 不自动重算 ✗ 免得卡界面 ✓）。
            cfgrowM = QHBoxLayout()
            cfgrowM.setContentsMargins(6, 2, 6, 2)
            self._rowM = cfgrowM            # ⭐ 存起来（**经典模式下整行隐藏** ✓ 见 `_on_mode_changed` ✓）
            cfgrowM.addWidget(QLabel("配对门限"))
            self.sp_pair = NoWheelDoubleSpinBox()
            self.sp_pair.setRange(20.0, 200.0)
            self.sp_pair.setSingleStep(5.0)
            self.sp_pair.setDecimals(0)
            self.sp_pair.setValue(float(_MOTION_PAIRED_DEFAULT))
            self.sp_pair.setToolTip(
                "两拍之间，同一个框最多挪这么远（px），超过就当成「新目标」另起一条。\n"
                "相机一帧能走几十 px ⇒ 调太小会把同一个目标断成两截。\n"
                "调大＝更不容易断（但离得近的两个目标容易并成一条）；调小＝更干净。")
            cfgrowM.addWidget(self.sp_pair)
            cfgrowM.addWidget(QLabel("分数衰减"))
            self.sp_sdecay = NoWheelDoubleSpinBox()
            self.sp_sdecay.setRange(0.50, 0.99)
            self.sp_sdecay.setSingleStep(0.01)
            self.sp_sdecay.setDecimals(2)
            self.sp_sdecay.setValue(float(_MOTION_SDECAY_DEFAULT))
            self.sp_sdecay.setToolTip(
                "分数每拍乘它：真目标「一直在动」⇒ 分数一直涨；假目标偶然跳一下 ⇒ 攒不住。\n"
                "调大＝记性长、更稳（但一旦跟错很难纠正）；调小＝只看最近几拍（反应快、易被带跑）。\n"
                "0.90 大约相当于「记住最近 10 拍」。")
            cfgrowM.addWidget(self.sp_sdecay)
            cfgrowM.addWidget(QLabel("换人余量"))
            self.sp_margin = NoWheelDoubleSpinBox()
            self.sp_margin.setRange(0.0, 40.0)
            self.sp_margin.setSingleStep(1.0)
            self.sp_margin.setDecimals(0)
            self.sp_margin.setValue(float(_MOTION_MARGIN_DEFAULT))
            self.sp_margin.setToolTip(
                "新目标要比现任高出这么多分才换人（防来回跳）。\n"
                "调大＝更稳（但真目标来了它还抱着旧的）；调小＝反应快（分数接近时会横跳）。")
            cfgrowM.addWidget(self.sp_margin)
            cfgrowM.addWidget(QLabel("建档拍数"))
            self.sp_minhits = NoWheelDoubleSpinBox()
            self.sp_minhits.setRange(1.0, 20.0)
            self.sp_minhits.setSingleStep(1.0)
            self.sp_minhits.setDecimals(0)
            self.sp_minhits.setValue(float(_MOTION_MINHITS_DEFAULT))
            self.sp_minhits.setToolTip(
                "一条轨迹被关联上这么多拍，才有资格当目标（防「刚建的空轨迹」抢位）。\n"
                "调大＝更保险（但开局晚几拍才出白线）；调小＝起跟快（第一拍挑错起点就歪）。")
            cfgrowM.addWidget(self.sp_minhits)
            cfgrowM.addWidget(QLabel("白度权重"))
            self.sp_whitew = NoWheelDoubleSpinBox()
            self.sp_whitew.setRange(0.0, 5.0)
            self.sp_whitew.setSingleStep(0.5)
            self.sp_whitew.setDecimals(1)
            self.sp_whitew.setValue(float(_MOTION_WHITEW_DEFAULT))
            self.sp_whitew.setToolTip(
                "目标落在「画面最白那块」上就加分（只在开局白星还在时有效，透明后自动归零）。\n"
                "0＝纯靠运动（开局晚几拍才定）；调大＝开局定得快（画面别处有白东西时会被带偏）。")
            cfgrowM.addWidget(self.sp_whitew)
            # ---- ⭐⭐ **第 5 行：跟踪手感 / 自检那一档**（用户 2026-10-03 ✓ 原话："把一些需要
            #   频繁调整的参数做成配置，然后告诉我应该怎么调、看什么" ✓）--------------------------------
            #   ⚠ 只放**真会调**的 ✓；名字短 ✓ tip 只讲"调大调小会怎样" ✓（原理写在
            #     `perception/lie_motion.py` 里 ✓ 这里不堆血泪史 ✗）。
            cfgrowQ = QHBoxLayout()
            cfgrowQ.setContentsMargins(6, 2, 6, 2)
            self._rowQ = cfgrowQ            # ⭐ 与第 4 行一样：**只在运动分离模式下显示** ✓
            cfgrowQ.addWidget(QLabel("位置平滑"))
            self.sp_smooth = NoWheelDoubleSpinBox()
            self.sp_smooth.setRange(0.0, 0.95)
            self.sp_smooth.setSingleStep(0.05)
            self.sp_smooth.setDecimals(2)
            self.sp_smooth.setValue(float(_MOTION_SMOOTH_DEFAULT))
            self.sp_smooth.setToolTip(
                "白线的跟手程度：新位置 = 本值×旧位置 + (1−本值)×观测位置。\n"
                "调大＝更稳但更滞后（实测 0.70 时滞后 87px）；调小＝跟手但抖。")
            cfgrowQ.addWidget(self.sp_smooth)
            cfgrowQ.addWidget(QLabel("合群门槛"))
            self.sp_qmin = NoWheelDoubleSpinBox()
            self.sp_qmin.setRange(0.5, 6.0)
            self.sp_qmin.setSingleStep(0.1)
            self.sp_qmin.setDecimals(1)
            self.sp_qmin.setValue(float(_MOTION_QMIN_DEFAULT))
            self.sp_qmin.setToolTip(
                "它的「相对速度」要大于噪声底这么多倍，才算「还在不合群」。\n"
                "调大＝更严（更容易判它跟丢）；调小＝更宽容（跟丢了也不说）。\n"
                "⚠ 看日志里的 `q`：`q` 小于本值就有一次算「没在合群」。")
            cfgrowQ.addWidget(self.sp_qmin)
            cfgrowQ.addWidget(QLabel("跟丢拍数"))
            self.sp_qneed = NoWheelDoubleSpinBox()
            self.sp_qneed.setRange(1.0, 10.0)
            self.sp_qneed.setSingleStep(1.0)
            self.sp_qneed.setDecimals(0)
            self.sp_qneed.setValue(float(_MOTION_QNEED_DEFAULT))
            self.sp_qneed.setToolTip(
                "最近 10 拍里有这么多拍「没在合群」⇒ 判它跟丢（降权、并允许换人）。\n"
                "调大＝更耐心（跟错了也拖着）；调小＝更快换人（可能误伤、来回跳）。")
            cfgrowQ.addWidget(self.sp_qneed)
            cfgrowQ.addWidget(QLabel("粘连阈值"))
            self.sp_stuck = NoWheelDoubleSpinBox()
            self.sp_stuck.setRange(1.05, 3.0)
            self.sp_stuck.setSingleStep(0.05)
            self.sp_stuck.setDecimals(2)
            self.sp_stuck.setValue(float(_MOTION_STUCK_DEFAULT))
            self.sp_stuck.setToolTip(
                "框面积超过自己基准这么多倍 ⇒ 认定「和别的目标粘住了」（两个目标被检成一个框）。\n"
                "粘住时框中心是两目标的中点 ⇒ 那一拍的位置和偏离都不采信（先滑过去）。\n"
                "调大＝更不敏感；调小＝更容易判粘住。⚠ 正常拍在 0.9~1.2，粘连拍在 1.7~1.9。")
            cfgrowQ.addWidget(self.sp_stuck)
            cfgrowQ.addWidget(QLabel("预测上限"))
            self.sp_predmax = NoWheelDoubleSpinBox()
            self.sp_predmax.setRange(0.0, 120.0)
            self.sp_predmax.setSingleStep(5.0)
            self.sp_predmax.setDecimals(0)
            self.sp_predmax.setValue(float(_MOTION_PREDMAX_DEFAULT))
            self.sp_predmax.setToolTip(
                "找框时最多按预测往前挪这么远（px），防止预测错了更糟。\n"
                "调大＝更敢预测（目标动得快时更跟得住）；调小＝更保守（贴着上一拍位置找）。\n"
                "设 0 ＝ 完全不用预测。")
            cfgrowQ.addWidget(self.sp_predmax)
            # ⭐⭐⭐⭐ **方向数 N**（用户 2026-10-04 ✓ 原话："**方向数 N 做成配置**" ✓）——
            #   就是"鼠标周围那一圈灰箭头"的个数 ✓（见 `_DIR_N` ✓）。
            cfgrowQ.addWidget(QLabel("方向数"))
            self.sp_dirn = NoWheelDoubleSpinBox()
            self.sp_dirn.setRange(4.0, 36.0)
            self.sp_dirn.setSingleStep(2.0)
            self.sp_dirn.setDecimals(0)
            self.sp_dirn.setValue(float(_MOTION_DIRN_DEFAULT))
            self.sp_dirn.setToolTip(
                "「方向置信度」把 360° 等分成几格 = 鼠标周围那圈灰箭头的个数。\n"
                "每格的值 = 有多少「框边界在群体坐标系里的运动」指向这个方向。\n"
                "调大＝分得更细（但每格票少、更抖）；调小＝更粗更稳。")
            cfgrowQ.addWidget(self.sp_dirn)
            # ⭐⭐⭐ **「框心力度」**（用户 2026-10-04 ✓ 原话："**我在哪里调控框心指导的力度？**" ✓
            #   ⇒ 就放这里 ✓ 见 `_FUSE_PULL` ✓）—— 融合期把速度**方向**往"框心合方向"拽多少比例 ✓
            #   （⚠ **只动转向、不动速度** ✓ 用户 2026-10-04 ✓）。
            cfgrowQ.addWidget(QLabel("框心力度"))
            self.sp_fpull = NoWheelDoubleSpinBox()
            # ⭐⭐⭐⭐⭐ **上限从 1.0 放到 3.0**（用户 2026-10-05 ✓ 原话："**框心力度无法调到 1
            #   以上**" ✓✓）—— ⚠ 原来卡在 `1.0` ✗，而 `1.0` 的语义是"**一步到位**"（把该方向的
            #   分量**换成**框心说的值 ✓ 见 `lie_motion._pull_to` ✓）⇒ 想更硬就**只能过冲** ✓
            #   ⇒ 那正是本值 `> 1` 的含义 ✓（⚠ 代码侧**没有任何夹取** ✗ —— 纯线性插值 ✓
            #   见 `_pull_to` ✓；所以这里放开就真生效 ✓）。
            #   ⚠⚠ **载入存档那一处也得跟着放** ✗✗（见下面 `_cfg_load` 里同一个数 ✓）——
            #     漏了它 ⇒ 存 `2.0` 进去、重开被夹回 `1.0` ✗（"改了不生效"的经典坑 ✓）。
            self.sp_fpull.setRange(0.0, 3.0)
            self.sp_fpull.setSingleStep(0.05)
            self.sp_fpull.setDecimals(2)
            self.sp_fpull.setValue(float(_MOTION_FPULL_DEFAULT))
            self.sp_fpull.setToolTip(
                "融合期：把**群体坐标系**里的速度**整体**拉向「框心说的目标速度」的比例。\n\n"
                "本值 k ⇒ v' = v + k × (框心那份 EMA − v)：\n"
                "· 框心说「目标相对群体更快」⇒ 沿它**加速**；\n"
                "· 说「更慢」⇒ **减速**；说「差不多」⇒ **不加**（所以不会无端起速）。\n\n"
                "· 0 ＝ 完全不加（只沿原速度滑）；\n"
                "· 1 ＝ 一步就把速度换成「框心说的值」；\n"
                "· 1~3 ＝ **过冲**：拉得**越过**「框心说的值」（框心那份被 EMA 平滑过 ⇒ 偏保守，\n"
                "  觉得力度不够就用这一段）；⚠ 太大有反效果 —— 框心说「这方向不动」时，\n"
                "  力度 2 会把现有速度**拉翻面** ✗（圆开始往反方向走 ✗）；\n"
                "· 默认 0.15。\n\n"
                "只改**速度方向/大小**，不动位置（位置归「夹取」✓）。\n"
                "改完点「应用」重算。")
            cfgrowQ.addWidget(self.sp_fpull)
            # ⭐⭐⭐⭐⭐ **「夹取增益」**（用户 2026-10-04 ✓ 原话："**位置被硬夹说明你要么快了要么
            #   慢了，夹取也要算进速度平滑。因为夹取是修正你的预测误差，夹取是较准确的测量值**"
            #   ✓✓）—— 见 `_CLAMP_GAIN` ✓：夹取挪了 `d` ⇒ **速度也跟着挪 `k×d`** ✓
            #   （⚠ 与「框心力度」**正交** ✗：那个改**速度方向** ✓ 这个改**速度本身** ✓）。
            cfgrowQ.addWidget(QLabel("夹取增益"))
            self.sp_cgain = NoWheelDoubleSpinBox()
            self.sp_cgain.setRange(0.0, 1.0)
            self.sp_cgain.setSingleStep(0.05)
            self.sp_cgain.setDecimals(2)
            self.sp_cgain.setValue(float(_MOTION_CGAIN_DEFAULT))
            self.sp_cgain.setToolTip(
                "融合期「夹取」把圆挪了 d（= 夹后 − 夹前）之后，**速度也跟着挪 k×d**：\n"
                "· 位置被硬夹 ⇒ 说明你**要么快了要么慢了**（d 就是那份预测误差）；\n"
                "· 不记进速度 ⇒ 下一拍照旧按老速度冲 ⇒ **夹取每拍都得再夹一次**（自己跟自己打架）；\n"
                "· 记进去 ⇒ 下一拍自然走到位 ⇒ 夹取自己就停了。\n\n"
                "· 0 ＝ 只挪位置、速度一个字不动（老行为）；\n"
                "· 1 ＝ 一步吃满（最信任这个测量，但夹错一次就整拍带歪）；\n"
                "· 默认 0.5 ＝ 按几何收敛，2~3 拍对齐（速度更平滑）。\n\n"
                "同时会写进 KF 的相对速度（否则下一拍被 KF 覆盖）。\n"
                "改完点「应用」重算。")
            cfgrowQ.addWidget(self.sp_cgain)
            # ⭐⭐⭐ **`?` = conf 的「统一说明处」**（用户 2026-10-04 ✓ 原话："我们需要有**统一的
            #   地方管理 conf 都与什么有关、每一项都是如何作用的**，建议**加个 `？` 写在里面**" ✓✓）
            #   ⇒ 点开就是**完整的构成表** ✓（通俗版 ✓ 与代码里的实现**一一对应** ✓）。
            self.btn_what = QPushButton("匹配分?")
            self.btn_what.setToolTip("点开看「匹配分」是怎么算出来的（每一项怎么作用）")
            self.btn_what.clicked.connect(self._explain_conf)
            cfgrowQ.addWidget(self.btn_what)
            cfgrowQ.addStretch(1)
            cfgrowM.addStretch(1)
            # ══════════════════════════════════════════════════════════════════════════════
            # ⭐⭐⭐ **速度跟踪（velocity）专用参数**（用户 2026-10-07 ✓ 原话："**清除通用参数外
            #   所有的配置参数，放上新增要用的配置参数**" ✓ ＋ 他点名问的那三个量 ✓）
            #   ⚠⚠ **本轮判据还没接** ✗（用户："**先不要做任何挑选真目标的逻辑**" ✓）⇒ 这三个值
            #     **先只落盘、还没生效** ✓（tooltip 里逐条写明 ✓ 免得看着像已经在起作用了 ✗）。
            #   ⚠ 值本身**一个都不许我自己拍** ✗ —— 下面是**提案默认值**（依据见 `lie_motion`
            #     顶上那三个常量的注释 ✓ 含实测数字 ✓），等用户定标 ✓ 界面上随时可改 ✓。
            #   ⚠ 量纲（用户 2026-10-07 问的 ✓）：半径 = **px**（加工域 ✓ 与检出框/绿圈同一把尺 ✓）；
            #     时长 = **秒**（⚠ 不用"拍"✗ —— 6fps / 30fps / 实时流都能比 ✓）；
            #     偏差 = **px/拍**（= 每拍"相对群体位移"的超出量 ✓ 与白箭头同量纲 ✓ 不用换算 ✓）。
            # ══════════════════════════════════════════════════════════════════════════════
            cfgrowV = QHBoxLayout()
            cfgrowV.setContentsMargins(6, 2, 6, 2)
            self._rowV = cfgrowV
            cfgrowV.addWidget(QLabel("捕获半径(px)"))
            self.sp_vcatch = NoWheelSpinBox()
            self.sp_vcatch.setRange(10, 2000)
            self.sp_vcatch.setSingleStep(10)
            self.sp_vcatch.setSuffix(" px")
            self.sp_vcatch.setValue(int(_VEL_CATCH_DEFAULT))
            self.sp_vcatch.setToolTip(
                "「**捕获多大范围内的不合群检出框**」\n"
                "· 量纲 = **画面像素（加工域）**，与检出框 / 绿圈同一把尺；\n"
                "· 参照点 = 这一拍**目标的预测位置**（圆心），只在这个半径里找「不合群」的检出框；\n"
                "· 物理含义 = 「几拍滑行 + 检测噪声」能积累出多大偏差（半径越大越宽容、也越容易被别人骗）；\n"
                "· 实测参照：本素材检出框宽中位 ≈ 150 px、每帧 17 个框 ⇒ 先给 120 px（≈ 0.8 个框宽）。\n\n"
                "⚠⚠ **本轮判据还没接** ⇒ 这个值先只落盘、**还没生效**（下一步接上）。")
            cfgrowV.addWidget(self.sp_vcatch)
            cfgrowV.addWidget(QLabel("不合群时长(s)"))
            self.sp_vhold = NoWheelDoubleSpinBox()
            self.sp_vhold.setRange(0.0, 30.0)
            self.sp_vhold.setSingleStep(0.1)
            self.sp_vhold.setDecimals(1)
            self.sp_vhold.setSuffix(" s")
            self.sp_vhold.setValue(float(_VEL_HOLD_DEFAULT))
            self.sp_vhold.setToolTip(
                "「**不合群多久算不合群**」\n"
                "· 量纲 = **秒**（⚠ 不用「拍」—— 素材有 6fps / 30fps / 实时流，只有秒跨素材可比）；\n"
                "· 物理含义 = 要连续这么多秒都在「明显偏离群体」，才认它是假目标座位（= 上板）；\n"
                "· 调大 ⇒ 更稳、但真目标一开始也更容易被当成砖；调小 ⇒ 更灵敏、也更容易被噪声骗；\n"
                "· 实测参照：6fps 素材 0.5 s ⇒ 3 拍；30fps 素材 0.5 s ⇒ 15 拍。\n\n"
                "⚠⚠ **本轮判据还没接** ⇒ 这个值先只落盘、**还没生效**（下一步接上）。")
            cfgrowV.addWidget(self.sp_vhold)
            cfgrowV.addWidget(QLabel("速度偏差门(px/拍)"))
            self.sp_vdev = NoWheelDoubleSpinBox()
            self.sp_vdev.setRange(0.0, 500.0)
            self.sp_vdev.setSingleStep(0.5)
            self.sp_vdev.setDecimals(1)
            self.sp_vdev.setSuffix(" px/拍")
            self.sp_vdev.setValue(float(_VEL_DEV_DEFAULT))
            self.sp_vdev.setToolTip(
                "「**速度偏差多少算不合群**」\n"
                "· 量纲 = **px/拍**（= 每拍「相对群体的位移」的超出量 ⇒ 与白箭头同一量纲，不用换算）；\n"
                "· 怎么量：这一拍的位移 − 群体中位位移（= 相机）⇒ 模长超过它 ⇒ 这拍算「不合群」；\n"
                "· 实测参照：噪声底（各格偏离的中位）约 1~3 px/拍；真目标「相对群体的自身速度」 10~22 px/拍\n"
                "  ⇒ 先给 8（落在两者之间、留余量）。\n\n"
                "⚠⚠ **本轮判据还没接** ⇒ 这个值先只落盘、**还没生效**（下一步接上）。")
            cfgrowV.addWidget(self.sp_vdev)
            cfgrowV.addStretch(1)
            cfgrow.addWidget(QLabel("噪声容差(px)"))
            self.sp_ntol = NoWheelDoubleSpinBox()
            self.sp_ntol.setRange(0.0, 100.0)
            self.sp_ntol.setSingleStep(0.5)
            self.sp_ntol.setDecimals(1)
            self.sp_ntol.setValue(float(_NOISE_TOL_DEFAULT))
            self.sp_ntol.setToolTip(
                "规则① 判的是「**圆心 ± 它** 的范围」与检出框**相交**（以圆心为中心的方块，\n"
                "碰到框就算相交 ⇒ 吸收检出框边缘的亚像素噪声 ✓）。\n"
                "只作用于**融合判定**（规则①）；0 = 范围退化成点 = 严格「圆心在框内」。\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow.addWidget(self.sp_ntol)
            # ---- ⭐⭐ **上板时长(秒)**（用户 2026-10-02 ✓ 原话："『多久判定为上板』应该以
            #   时长为单位做成配置，像『噪声容差』一样加到配置行" ✓）：登记条目**自出生起**
            #   跟着群体走满这么多秒、且面积/宽高比/不贴边判据都过 ⇒ 才转正上板 ✓；
            #   0 = 一帧就上板（老行为 ✓）。
            cfgrow.addWidget(QLabel("上板时长(秒)"))
            self.sp_board = NoWheelDoubleSpinBox()
            self.sp_board.setRange(0.0, 10.0)
            self.sp_board.setSingleStep(0.1)
            self.sp_board.setDecimals(1)
            self.sp_board.setValue(float(_BOARD_S_DEFAULT))
            self.sp_board.setToolTip(
                "登记条目自出生起要跟着群体走满这么多秒、且面积/宽高比/不贴边判据都满足，\n"
                "才转正为上板的青色砖。0 = 一帧就上板。\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow.addWidget(self.sp_board)
            # ---- ⭐⭐ **上板防抖重叠率**（用户 2026-10-02 ✓ 原话："目前有**旧砖被重复往上
            #   叠加砖**的问题。优化：如果**新砖与旧砖重叠率 >= 0.5**，判定为**同一登记砖**。
            #   **标砖的框选更接近假目标群体标准尺寸的那个**。这个 0.5 做成参数配置『**上板防抖
            #   重叠率**』" ✓）：紧门配不上、正要**新建条目**之前，先拿这个检出与**旧砖**比重叠率
            #   ✓（重叠率 = 交集 ÷ 两者中较小的面积 ✓ —— 小框叠在大砖上也接近 1.0 ✓）：
            #   ≥ 它 ⇒ 判**同一块砖**、**不新建** ✗，并把该砖的框改成**更接近群体标准尺寸**的
            #   那个 ✓。0 = 关（老行为一字不变 ✓）。判据在登记表里（`ShapeRegistry` ✓）。
            cfgrow.addWidget(QLabel("上板防抖重叠率"))
            self.sp_ov = NoWheelDoubleSpinBox()
            self.sp_ov.setRange(0.0, 1.0)
            self.sp_ov.setSingleStep(0.05)
            self.sp_ov.setDecimals(2)
            self.sp_ov.setValue(float(_BOARD_OVERLAP_DEFAULT))
            self.sp_ov.setToolTip(
                "新建登记砖之前，先拿这个检出与**已有旧砖**比一次重叠率\n"
                "（重叠率 = 交集 ÷ 两者中较小的面积 —— 小框叠在大砖上也接近 1.0）。\n"
                "≥ 它 ⇒ 判**同一块登记砖**、**不再新建**，并把该砖的框改成**更接近群体标准尺寸**\n"
                "（登记表定的标准面积 typ_area）的那个。0 = 关（老行为）。\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow.addWidget(self.sp_ov)
            # ---- ⭐⭐⭐ **鼠标跟随效率倍率**（用户 2026-10-03 ✓ 原话："能否开放一个系数配置，
            #   调**追踪器**跟上圆心的效率倍率？" ✓ + "**我不想影响位置计算**，只调追踪器
            #   （较大的白描边圈绿圆）" ✓）：画面上那个"**白圈 + 绿实心点**"的大点 = **控制器
            #   输出的模拟鼠标** ✓（不是 `LieTracker` 的位置 ✓）⇒ 本项只决定"**鼠标每拍朝圆心
            #   走多快**" ✗ 一点也不影响位置计算 ✓（融合落位 / 夹取 / 预测全都不看它 ✓）。
            #   · **1.00（默认）= 全量**（每拍尽量贴上去 ✓ 老行为一字不变 ✓）；
            #   · 调小 ⇒ 每拍只走剩余误差的一部分 ⇒ 光标**渐进**贴上圆心 ✓；
            #   · 调大 ⇒ 过冲 ⇒ 在"一拍延迟 + 速度前馈"的闭环里**提前压住滞后** ✓。
            cfgrow.addWidget(QLabel("鼠标跟随效率倍率"))
            self.sp_follow = NoWheelDoubleSpinBox()
            self.sp_follow.setRange(0.01, 5.0)      # ⚠ 0 = 光标几乎不动 ⇒ 无意义 ⇒ 下限 0.01 ✓
            self.sp_follow.setSingleStep(0.05)
            self.sp_follow.setDecimals(2)
            self.sp_follow.setValue(float(_FOLLOW_GAIN_DEFAULT))
            self.sp_follow.setToolTip(
                "**鼠标（追踪器那个白圈绿点）每拍朝圆心走多快** —— 每拍消掉「光标 → 圆心」\n"
                "误差的这么多倍（**只作用在控制器输出上**，不影响任何位置计算 ✓）。\n\n"
                "· 1.00（默认）⇒ 每拍尽量全量贴上去（仍受单步限幅 200 与整数取整限制）；\n"
                "· 调小（如 0.50）⇒ 每拍只走一半剩余 ⇒ 光标渐进贴上圆心；\n"
                "· 调大（如 1.50）⇒ 会过冲，但在「一拍延迟 + 速度前馈」的闭环里能提前到位。\n\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow.addWidget(self.sp_follow)
            # ---- ⭐⭐⭐ **缩成砖判定 · 最小 IoU**（用户 2026-10-03 ✓ 原话："我才学到 IoU 这个
            #   算法，我认为我们很多判定都可以换成这个算法（例如『噪声容差』改成『**噪声最小
            #   IoU**』）" ✓）：融合期判"这格框已经缩回**砖的大小/位置**了"原来只看"**四条边都在
            #   噪声容差(px) 内**"（绝对像素 ⇒ 大砖松、小砖紧 ✗）；本项补一把**尺度无关**的尺 =
            #   `IoU(框, 砖) ≥ 本值` ✓（两条**并列** ⇒ 满足任一即算"缩成砖" ✓）。
            #   ⚠ **只有这一处加 IoU** —— "覆盖率/是否包住砖"那些是**单向**"大框盖住小砖"（框多大
            #   不进分母 ✓）⇒ 换 IoU 反而会误判 ✗；登记表"上板防抖"要的是"小框叠大砖≈1"✓（IoU
            #   会偏小、拦不住 ✗）；"圆心±容差与框相交"是"离框边几像素"（点/边对区域 ✓）⇒ IoU
            #   没有对应物 ✓（详见 `perception/geom.py` 模块头 ✓）。
            #   · **0.70（默认）**≈ 上面那条"四条边差 20px 上下" ✓；越接近 1 越严 ✓；
            #   · **0 = 关**（只剩像素那条腿 ✓ 老行为一字不变 ✓）。
            cfgrow.addWidget(QLabel("缩成砖最小 IoU"))
            self.sp_ioub = NoWheelDoubleSpinBox()
            self.sp_ioub.setRange(0.0, 1.0)
            self.sp_ioub.setSingleStep(0.05)
            self.sp_ioub.setDecimals(2)
            self.sp_ioub.setValue(float(_IOU_BRICK_DEFAULT))
            self.sp_ioub.setToolTip(
                "融合期判「这格框已经缩回**砖的大小/位置**」的另一把尺：\n"
                "**IoU(框, 砖) ≥ 本值** 也算「缩成砖」（与「四条边都在噪声容差内」并列）。\n\n"
                "· 0.70（默认）≈ 「四条边差 20px 上下」同一量级；越接近 1 越严；\n"
                "· 0 = 关（只用「噪声容差(px)」那条腿，老行为一字不变）。\n\n"
                "⚠ 只有这一处判据用 IoU —— 「覆盖率 / 是否包住砖」是单向的「大框盖住小砖多少」，\n"
                "换 IoU 会误判（并集框本来就比砖大）；登记表「上板防抖重叠率」要的是「小框叠在\n"
                "大砖上 ≈ 1「，IoU 会偏小、拦不住。\n\n"
                "改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow.addWidget(self.sp_ioub)
            # ---- ⭐⭐⭐ **"位置估计"算法分支**（用户 2026-10-03 ✓ 原话："做 KF影子模式 然后找个
            #   地方加按钮或开关切换（**你决定在哪、叫什么**）—— 目的是可以**自由选择算法分支
            #   以防污染当前的进展**" ✓）：
            #   · **经典（现状）**（默认 ✓）= 现在这套链条（残差/白块观测 + 夹取 + 拟人平滑 ✓）
            #     —— **一字不变** ✓；
            #   · **KF 影子（只记录）** = 卡尔曼与经典**并行**跑 ✓ 把经典每拍的位置当观测喂给它 ✓
            #     **只记录** KF 的位置/速度/**新息** ✗ **绝不改位置** ✓（这就是"不污染" ✓）。
            #     数据在**状态栏尾**看：`KF 影子：差 p50/p90 ｜ 新息 p50/p90` ✓
            #     —— 新息量级 **直接告诉你观测噪声 σ 该取多大** ✓（`kf.KF_R_DEF` 现在是 6px ✓）。
            #   ⚠ 默认**经典** ⇒ 现有进展不动 ✓ 想对照就切到影子 ✓ 任何时候能切回来 ✓。
            cfgrow.addWidget(QLabel("显示 KF 预测"))
            self.chk_kf = QCheckBox()
            self.chk_kf.setChecked(False)
            self.chk_kf.setToolTip(
                "**位置估计用哪条算法分支**（切换后点「应用」重算）：\n\n"
                "· 经典（现状）⇒ 现在这套链条，行为一字不变（默认）；\n"
                "· KF 影子（只记录）⇒ 卡尔曼与经典**并行跑**，只记录不开车：\n"
                "   把经典每拍算出的位置当观测喂给卡尔曼，看它估到哪、以及「新息」多大。\n"
                "   结果在**状态栏尾**：差 p50/p90（KF 与经典差多少）、新息 p50/p90\n"
                "   （= 经典输出里的抖动/意外量级，直接告诉你观测噪声该设多大）。\n\n"
                "⚠ 影子模式**不影响**任何位置计算 ⇒ 可以放心开着跑、对照完再切回来。")
            cfgrow.addWidget(self.chk_kf)
            # ⭐ 勾选变化 ⇒ 立刻显示/隐藏**右列**（KF 位置驱动 ✓ 用户 2026-10-03 ✓）——
            #   ⚠ 数据要**点"应用"重算**才有（右列要重跑一整遍 ✓ 不自动重算免得卡界面 ✓）。
            self.chk_kf.stateChanged.connect(self.on_toggle_kf)
            self.btn_apply = QPushButton("应用")
            self.btn_apply.setToolTip("按当前「轨迹预测窗口时间」**重新演算**整段并生效 ✓")
            self.btn_apply.clicked.connect(self.apply_path_ms)
            self.btn_apply.setEnabled(False)          # 载入素材前点了没意义 ✓
            # ⭐⭐ **"应用"统一放整个配置区的右下角**（用户 2026-10-02 ✓ 原话："把应用按钮放到
            #   **整个配置区的右下角**（**以后也如此**）" ✓ ⇒ 已同步写进 `docs/UI规范.md` ✓）：
            #   所以这里**不**把它加进 `cfgrow` ✗（那一行只放参数 ✓ 组装见下面的配置区 ✓）。
            cfgrow.addStretch(1)

            self.lbl = _ImageView()
            self.lbl.setText("点左上角「选择视频…」放一段录像，或「选择 GIF…」"
                             "选一个动图（如 datasets/liedetectorgifs/test.gif）✓")
            # ⭐⭐⭐ **左键点选检出框**（用户 2026-10-03 ✓ 原话："你能让我在**点选检出框**的时候
            #   显示 `(360.4,323.0) 179×208=37232` 这种信息吗" ✓）：视图把**加工域坐标**报给
            #   这里 ⇒ 命中那格检出框 ⇒ 黄框高亮 + `几何=面积` 标签 + 状态栏一行 ✓（纯显示 ✓）。
            #   `_pick` = `{"frame": 第几帧, "box": 检出框元组}`（切帧后**不显示**但不清除 ✓
            #   —— 拖回那一帧还能看到 ✓）；点空白 ⇒ 取消（`box=None`）。
            self._pick = None
            self.lbl._on_pick = self._on_pick_box
            # ⭐⭐⭐ **右键**（用户 2026-10-03 ✓ 原话："加一个**右键点击检出框选中并弹出菜单**，
            #   目前只有一项『**复制检出框信息**』，用来我**复制之后与你交流**" ✓）⇒
            #   `_on_pick_menu` ✓（命中 ⇒ 选中 + 弹菜单 + 复制 ✓ 见它 ✓）。
            self.lbl._on_context = self._on_pick_menu
            # ⭐⭐⭐⭐⭐ **限定区域（ROI ✓ 用户 2026-10-07 ✓ 原话："**只在限定的区域计算（因为
            #   这是部分弹窗）**" ✓✓）** —— ⚠⚠ **为什么非有不可** ✗（**实测** `10月7日.mp4` ✓）：
            #   那条片是**整屏录像** ✓（蓝天白云 ✗ + 各种 UI ✗）⇒ 算法整幅在算 ⇒ `pick_white`
            #   把**一朵云**当成白块 ✗（实测 (462,64) = 天空 ✓）、座位从背景里选出来 ✗ ⇒ 全乱 ✓。
            #   ⇒ 演示窗里**拖一个框**（`框选区域` ✓）⇒ 只在这个框里算 ✓（检测框 / 白块 / 轨迹 /
            #     群体 ✓ 框外一概看不见 ✓）；**存进 `config/ui.yaml`** ✓（下次自动用 ✓）；
            #     画面上**一直画着** ✓（所见即所得 ✓ 见 `draw()` ✓）。
            #   ⚠ 默认 `None` = **整幅** ⇒ **行为一个字不变** ✓（你没框之前跟改动前逐位一致 ✓）。
            self.roi = None
            self.lbl._on_roi = self._on_roi_picked
            # ---- 播放进度条（可拖动定位 ✓ 用户 2026-09-30 要求 ①）----
            self.sld = QSlider(Qt.Horizontal)
            self.sld.setRange(0, 0)
            self.sld.setToolTip("拖动定位 ✓（会从该处**重新建档**跟踪：跟踪器要重新"
                                "认一次白色目标 ✓）")
            self.sld.sliderPressed.connect(self.on_seek_press)
            self.sld.sliderMoved.connect(self.on_seek)
            # ⭐⚠ `sliderMoved` **只在鼠标拖动时发** ✗ —— 用**方向键**改值走的是
            #   `valueChanged` ⇒ 不接它就会出现"值动了、画面没变" ✗（用户 2026-09-30 ✓）。
            #   播放中由 `tick` 同步滑块时已 `blockSignals` ✓ 不会自激 ✓。
            self.sld.valueChanged.connect(self.on_seek)
            self.sld.sliderReleased.connect(self.on_seek_done)
            self.lbl_time = QLabel("0.0 / 0.0 s")
            self.lbl_time.setStyleSheet("color:#ddd; padding:0 8px;")
            prow = QHBoxLayout()
            prow.setContentsMargins(6, 4, 6, 4)
            prow.addWidget(self.sld, 1)
            prow.addWidget(self.lbl_time)

            c = QWidget()
            lay = QVBoxLayout(c)
            lay.setContentsMargins(0, 0, 0, 0)
            # ⭐⭐⭐ **配置区组装**（用户 2026-10-02 ✓ 原话："把应用按钮放到**整个配置区的右下角**
            #   （**以后也如此**）" ✓）：两行参数**靠左排**（各自末尾都有弹簧 ✓ 不平铺 ✓），
            #   `应用`按钮摆在**整个配置区**的**右下角** ✓。规范已写进 `docs/UI规范.md` ✓。
            _cfgcol = QVBoxLayout()
            _cfgcol.setContentsMargins(0, 0, 0, 0)
            _cfgcol.setSpacing(2)
            # ⭐⭐⭐ **三行分档布局**（用户 2026-10-03 ✓ 原话："整理参数分档布局（设计一下，直观、
            #   美观、好理解点）"＋"按状态配置参数" ✓）：每行前面加一条**分组标题**（淡色小字 ✓
            #   不占高度 ✓）⇒ 一眼能看出"这行管哪一期" ✓。

            def _group(_title, _desc=""):
                _w = QWidget()
                _h = QHBoxLayout(_w)
                _h.setContentsMargins(6, 2, 6, 0)
                _h.setSpacing(6)
                _t = QLabel(_title)
                _t.setStyleSheet("color:#7fd; font-weight:bold; padding:0;")
                _h.addWidget(_t)
                if _desc:
                    _d = QLabel(_desc)
                    _d.setStyleSheet("color:#98a; padding:0;")
                    _h.addWidget(_d)
                _h.addStretch(1)
                return _w

            _cfgcol.addWidget(_group("通用", "轨迹预测 / 上板 / 跟随 / 同检出 IoU / 选框与夹取限速"))
            _cfgcol.addLayout(cfgrow)
            # ⭐ **两个分组标题也存起来**（用户 2026-10-04 ✓ 原话："在运动分离模式下，配置栏上面的
            #   分离期、融合期 两行移除" ✓ —— ⚠ 原来只收了**行内控件**、**标题没跟着收** ✗
            #   ⇒ motion 模式下留下一排空标题 ✓ 见 `_on_mode_changed` ①' ✓）。
            self._row2_head = _group("分离期", "红框丢失、走淡粉预测那一段："
                                                "这一对阈值管「算不算又融合了」（进严）")
            _cfgcol.addWidget(self._row2_head)
            _cfgcol.addLayout(cfgrow2)
            self._row3_head = _group("融合期", "已在融合保持里：同一对阈值管「还该不该继续」"
                                                "（守松）——不满足就判分离")
            _cfgcol.addWidget(self._row3_head)
            _cfgcol.addLayout(cfgrow3)
            # ⭐⭐⭐ **第 4 行：运动分离（新）专用**（用户 2026-10-03 ✓ 原话："运动分离 **没有任何
            #   参数要配吗？**" ✓）—— ⚠ 只在**新模式**下显示 ✓（经典模式整行隐藏 ✓ 见
            #   `_on_mode_changed` ✓）。
            _wM = _group("运动分离（新）",
                         "YOLO 给候选 + 「运动不合群」分辨 —— 第 4 行：配对 / 打分 / 换人；"
                         "第 5 行：平滑 / 自检 / 粘连"
                         "（⚠ 本模式**不用砖、不做融合/分离** ✓ 但**要看检出框** ✓）")
            _wM.setVisible(False)               # 默认是经典 ⇒ 先藏起来 ✓（切模式时同步 ✓）
            self._rowM_head = _wM               # ⭐ 标题也跟着收 ✓
            _cfgcol.addWidget(_wM)
            _cfgcol.addLayout(cfgrowM)
            _cfgcol.addLayout(cfgrowQ)          # ⭐ **第 5 行**（跟踪手感 / 自检 ✓ 见上面那段 ✓）
            # ⭐⭐⭐ **速度跟踪专用那一行**（用户 2026-10-07 ✓ 见上面那段 ✓）—— 只有它这一档显示 ✓
            _wV = _group("速度跟踪",
                         "捕获范围 / 不合群时长 / 速度偏差门"
                         "（⚠ 判据下一轮才接 ⇒ 这三个值先只落盘 ✓ 见每项 tooltip ✓）")
            _wV.setVisible(False)               # 默认经典 ⇒ 先藏 ✓（切模式时同步 ✓）
            self._rowV_head = _wV
            _cfgcol.addWidget(_wV)
            _cfgcol.addLayout(cfgrowV)
            _cfgbox = QHBoxLayout()
            _cfgbox.setContentsMargins(0, 0, 6, 2)
            _cfgbox.addLayout(_cfgcol)
            _cfgbox.addStretch(1)                                  # 参数列靠左、余量留右 ✓
            _cfgbox.addWidget(self.btn_apply, 0, Qt.AlignRight | Qt.AlignBottom)   # 右下角 ✓
            lay.addLayout(_cfgbox)
            # ⭐⭐⭐ **并排两列视图**（用户 2026-10-03 ✓ 原话："因为**预测结果不同，实际走向也会
            #   不同**，所以在**实时演算上叠加 KF 显示意义不大**。能不能：当『KF预测』开启时，
            #   将**窗口分为两列视图**，左边跑经典右边跑 KF，这样对照才有意义" ✓）：
            #   · **左列 = 经典**（= 现状 ✓ 一个字不变 ✓）；
            #   · **右列 = KF 位置驱动**（`Runner(kf_pos=True)` ✓ 真用 KF 位置跑整条链 ✓）；
            #   · **勾了"显示 KF 预测"才显示右列** ✓（不勾 ⇒ 右列整体 `hide()` ⇒ 与改动前
            #     **逐像素一致** ✓ 老行为一字不变 ✓）；
            #   · 两列**同帧号**对齐（各取各的演算表 ✓），缩放各自独立（滚轮/双击互不影响 ✓）。
            _cols = QHBoxLayout()
            _cols.setContentsMargins(0, 0, 0, 0)
            _cols.setSpacing(2)
            _col1 = QVBoxLayout()
            _col1.setContentsMargins(0, 0, 0, 0)
            _col1.setSpacing(1)
            self.lbl_k1 = QLabel("经典")
            self.lbl_k1.setStyleSheet("color:#9ad; padding:0 6px;")
            _col1.addWidget(self.lbl_k1)
            _col1.addWidget(self.lbl, 1)
            self.col2 = QWidget()
            _col2 = QVBoxLayout(self.col2)
            _col2.setContentsMargins(0, 0, 0, 0)
            _col2.setSpacing(1)
            self.lbl_k2 = QLabel("KF 位置估计")
            self.lbl_k2.setStyleSheet("color:#0ff; padding:0 6px;")
            self.lbl2 = _ImageView()
            self.lbl2.setText("（勾「显示 KF 预测」⇒ 这一列跑 KF ✓）")
            _col2.addWidget(self.lbl_k2)
            _col2.addWidget(self.lbl2, 1)
            _cols.addLayout(_col1, 1)
            _cols.addWidget(self.col2, 1)
            # ⭐ **右侧信息栏**（用户 2026-10-04 ✓ 原话：「信息面板和图例能不要放在视图区
            #   吗？可以在 gui 窗口开一列信息栏专门显示实时信息」）—— 把图例 + 实时数据
            #   从画面挪到这里 ✓；深色底、monospace、固定宽 280px（放得下 14 行图例 + 实时数 ✓）；
            #   ⚠ 用 QLabel（Qt 支持 Unicode）⇒ 图例可用**中文**（画面上 cv2 不认中文才用英文 ✓）；
            #   受 `chk_panel`（「显示图例」开关）控制 visible ✓ 见 `_on_layer_toggle` ✓。
            self.info_col = QFrame()
            self.info_col.setFixedWidth(280)
            self.info_col.setStyleSheet(
                "QFrame{background:#15171a;border:1px solid #303030;}"
                "QLabel{color:#cfd8dc;font-family:Consolas,'Cascadia Mono',monospace;"
                "font-size:13px;padding:1px 4px;}")
            _info_lay = QVBoxLayout(self.info_col)
            _info_lay.setContentsMargins(4, 4, 4, 4)
            _info_lay.setSpacing(2)
            _info_h1 = QLabel("图例")
            _info_h1.setStyleSheet("color:#9ad;font-weight:bold;padding:2px 4px;")
            _info_lay.addWidget(_info_h1)
            # ⭐⭐⭐⭐⭐ **图例补全**（用户 2026-10-04 ✓ 原话："**黄箭头代表什么？图例里没有。
            #   是不是还有其他东西图例里没写？**" ✓✓）—— 把画布上**实际画的东西**全收进来
            #   ✗（原来只有 9 条，实测漏了：黄箭头 / 洋红细内框 / 框上小字 / 「推的」灰框 /
            #     点选高亮 / 粉框 / 圆心红绿点 / 每框运动箭头 ✓）；⚠ 并修掉一条**已经不准的** ✗：
            #   "洋红框：融合（必含真目标）" —— 2026-10-04 改"所见即所得"之后，洋红框有**三种**：
            #   ① 真目标**自己那格**框；② 融合**含真目标**；③ 融合**跟真目标无关** ✗ ⇒ 原来那句
            #   会误导（帧 27 那种"两个别的目标相撞"的紫框**不含真目标** ✓ 面板已经如实分档 ✓）。
            #   ⚠ 归「显示图例」开关管（`chk_panel` ✓）；列宽 280px + 自动换行 ✓。
            #   ⭐⭐⭐⭐⭐ **文字按模式挑**（用户 2026-10-07 ✓ 速度跟踪那一轮）—— 运动分离那 16 行
            #     **整段搬去** `_LEGEND_MOTION` ✓（**一个字不改** ✓ 零污染 ✓）；速度跟踪走
            #     `_LEGEND_VELOCITY`（**只那 7 条** ✓ 见常量那段 ✓）。
            #     ⚠ 换模式时由 `_sync_legend()` 重设 ✓（见 `_on_mode_changed` ✓）。
            self.info_legend = QLabel(self._legend_text())
            self.info_legend.setWordWrap(True)
            self.info_legend.setAlignment(Qt.AlignTop)
            _info_lay.addWidget(self.info_legend)
            _info_lay.addStretch(1)
            _info_h2 = QLabel("实时信息")
            _info_h2.setStyleSheet("color:#9ad;font-weight:bold;padding:2px 4px;")
            _info_lay.addWidget(_info_h2)
            self.info_live = QLabel("（待演算）")
            self.info_live.setWordWrap(True)
            self.info_live.setAlignment(Qt.AlignTop)
            _info_lay.addWidget(self.info_live)
            _cols.addWidget(self.info_col)
            # 默认按当前模式显隐（chk_panel 勾上 + motion/velocity 才显示 ✓ 经典模式藏 ✓）
            self.info_col.setVisible(
                bool(self._mv()
                     and getattr(self, "chk_panel", None)
                     and self.chk_panel.isChecked()))
            lay.addLayout(_cols, 1)
            self.col2.setVisible(False)        # ⭐ 默认不显示（= KF 关 ⇒ 老样子 ✓）
            lay.addLayout(prow)
            # ---- ⭐⭐ **底部深色日志区**（用户 2026-10-02 ✓ 原话："给测谎演示窗口**下面**加上
            #   **深色背景的日志区**。当**融合框生成时**添加日志，**显示相关的砖**以及**解释
            #   判定为融合框的计算数据**" ✓）----
            #   · 内容 = 追踪器 `out["merge_log"]`（**只在"融合框生成"那一拍**非空 ✓ ——
            #     融合保持期不给 ✗ 见 `LieTracker._build_merge_log` ✓）⇒ 这里只管显示 ✓
            #     **不重算判据** ✗（口径不许分家 ✓）；
            #   · 固定高 176px（不抢画面太多 ✓）；`setMaximumBlockCount` 只留最近 400 条 ✓
            #     （稳态内存有界 ✓）；
            #   · 可**鼠标选中复制** ✓ —— 但它是"会吃 ←→/空格"的控件 ✗ ⇒ 加进 `_steal_on`
            #     （与数字框同一套 ✓ 见下 ✓）⇒ 焦点在日志上照样能切帧 ✓；
            #   · 回拖 / 换素材会**重建**（不重复刷 ✗ 见 `_log_sync` ✓）。
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setPlaceholderText(
                "融合框日志：判定为「融合框（真并集）」的那一拍（融合框生成）"
                "与「融合框消失」的那一拍（分离），都会在这里列出参照砖与判定数据")
            self.log.setMaximumBlockCount(400)
            # ⚠⚠ **字数与高度一起调** ✗（用户 2026-10-04 ✓ 原话："**窗口字体能大点嘛要瞎了**" ✓）——
            #   字号 11 → **15px** ✓、高度 176 → **260** ✓（不然行变高、可见行数骤减 ✗）。
            self.log.setFixedHeight(260)
            self.log.setStyleSheet(
                "QPlainTextEdit{background:#15171a;color:#cfd8dc;border:1px solid #303030;"
                "font-family:Consolas,'Cascadia Mono',monospace;font-size:15px;"
                "padding:4px 6px;selection-background-color:#37474f;}"
                "QScrollBar:vertical{background:#15171a;width:10px;margin:0;}"
                "QScrollBar::handle:vertical{background:#455a64;border-radius:5px;"
                "min-height:20px;}"
                "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")
            lay.addWidget(self.log)
            # ⚠⚠ **启动时若存的就是「速度跟踪」** ✗：那一刻 `_on_mode_changed` 早就跑过了（配置行
            #   在 2700 段、日志区在 3500 段 ✗）⇒ 日志区会**露着** ✗ ⇒ 这里**补一次显隐同步** ✓
            #   （只动这一处 ✓ 不重跑 `_on_mode_changed` ✗ 免得启动就弹一句状态栏提示 ✓）。
            if getattr(self, "log", None) is not None:
                self.log.setVisible(not self._vel())
            self.setCentralWidget(c)
            # ⭐ **状态栏也放大**（用户 2026-10-04 ✓ "窗口字体能大点" ✓）—— 它那句"点选 (x,y) W×H"
            #   是**要盯着看的** ✓ 默认字号偏小 ✗；`QStatusBar` 的样式**不随 `app.setFont` 走** ✗
            #   （Qt 给它自己一套默认样式 ✓）⇒ 单独设 ✓。
            self.statusBar().setStyleSheet("QStatusBar{font-size:15px;}")

            self.timer = QTimer(self)
            #   ⚠⚠⚠ **两个定时器槽都包一层"先落盘再吞掉"** ✗✗（用户 2026-10-07 ✓ 他报"**闪退**" ✓）：
            #     Qt 槽里抛出的**未捕获异常** ⇒ PyQt 直接 `qFatal`/`abort` ⇒ **窗口整个消失** ✓，
            #     而他跑的是 `pythonw`（无控制台 ✗）⇒ **traceback 一个字都看不到** ✗ ⇒ 只能靠猜
            #     （这次就是：`motion_viz` 一句日志拿 `None` 当浮点 ✓，崩在第 ~430 帧 ✓，
            #     我从"内存/线程/检测/定时器"挨个二分才挖到 ✓）⇒ 现在**先写 `logs/lie_demo_error.log`** ✓，
            #     并且**不让它把窗口带走** ✓（状态栏还会说明一句 ✓ 见 `_safe` ✓）。
            self.timer.timeout.connect(lambda: self._safe(self.tick))
            self.warm = QTimer(self)
            self.warm.timeout.connect(lambda: self._safe(self.warm_tick))

            # ---- ⭐⭐⭐ **保底键盘：←→ 切帧、空格 播放/暂停（不管焦点在哪 ✓）**（用户
            #   2026-10-01 ✓ 原话："优化窗口操作：不管聚焦在哪，左右方向键都是切帧、空格都是
            #   播放/暂停" ✓）----
            #   ⚠⚠ **为什么不能用 `keyPressEvent`**（原来那一版就是这么写的 ✗）：它只在
            #     **焦点在本窗口自己身上**时才收到 ✗ —— 而这一窗有一堆**会吃掉按键**的子控件：
            #     进度条滑块吃 ←→ ✓、蒙版滑块吃 ←→ ✓、速度下拉吃 ←→ ✓、勾选框与按钮吃空格 ✓、
            #     数字框编辑时吃 ←→ 和空格 ✓ ⇒ 用户点过它们之后方向键就"失灵"了 ✗（症状正是
            #     "有时候能切、有时候不能" ✗）。
            #   ⇒ 改用 `QShortcut` + `Qt.WindowShortcut`（默认上下文 ✓ 与主窗口"全局热键被
            #     占用 ⇒ 降级为窗口内快捷键"是同一套 ✓）：只要**本窗口是活动窗口**就触发 ✓，
            #     并且 Qt 的快捷键系统在把按键分发给焦点控件**之前**先匹配 ✓ ⇒ 焦点在哪都一样 ✓。
            #   ⚠ 明知故犯的代价（用户明确要求"不管聚焦在哪" ✓）：焦点在**进度条**上时 ←→
            #     不再挪滑块（改切帧 ✓ 切帧才是这窗口的主操作 ✓）；焦点在**数字框**里编辑时
            #     ←→ 也切帧 ✓（要靠键盘改数值：先 Tab 走 ✗ 或点它的小箭头 ✓）。
            #   ⚠ 按住不放会**自动重复** = 连切 ✓（与质检台同一体验 ✓ 用户口径一致 ✓）。
            for _key, _delta in ((Qt.Key_Left, -1), (Qt.Key_Right, 1)):
                _sc = QShortcut(QKeySequence(_key), self)
                _sc.setContext(Qt.WindowShortcut)
                _sc.activated.connect(lambda d=_delta: self.step_frame(d))
            _sc_sp = QShortcut(QKeySequence(Qt.Key_Space), self)
            _sc_sp.setContext(Qt.WindowShortcut)
            _sc_sp.activated.connect(self.toggle_play)

            #   ⚠⚠ **数字框是唯一的例外，要单独打一个洞**（2026-10-01 实测 ✓ 一个不落地验过
            #     七种焦点：进度条 ✓ 蒙版滑块 ✓ 速度下拉 ✓ 勾选框 ✓ 按钮 ✓ 都放行 ✓，
            #     **只有数字框不放行** ✗）：`QSpinBox` 内部有个 `QLineEdit`，它在
            #     `ShortcutOverride` 阶段就**抢先 accept** 掉 ←→（= 挪文本光标 ✓）和空格（= 打
            #     空格 ✓）⇒ Qt 认为"这个键归控件" ⇒ 我们的窗口快捷键**根本不触发** ✗。
            #   ⇒ 在数字框上装 `eventFilter`（= 本类的 `eventFilter` ✓ 见那边）：这两个键的
            #     `ShortcutOverride` **吞掉且不 accept** ⇒ 快捷键系统照常匹配 ✓。
            #     代价：数字框里不能再用 ←→ 挪光标（本来也用不上 —— 改值 = 全选重打 ✓）。
            #   ⚠ **底部日志区**（`QPlainTextEdit` ✓ 2026-10-02 加 ✓）也吃 ←→/空格 ⇒ 一并装上 ✓
            #     （它只读、不靠键盘操作 ⇒ 代价为零 ✓；**鼠标选中复制**照旧可用 ✓）。
            self._steal_on = set()
            for _w in (getattr(self, "sp_path_ms", None), getattr(self, "sp_merge", None),
                       getattr(self, "sp_ntol", None), getattr(self, "sp_board", None),
                       getattr(self, "sp_ov", None), getattr(self, "log", None)):
                if _w is not None:
                    _w.installEventFilter(self)
                    self._steal_on.add(_w)

            # ---- ⭐ **回填上次的配置**（用户 2026-10-01 ✓）----
            #   ⚠ 必须在 `load()` **之前**（"用检测器"决定走哪条演算路 ✓、
            #     "轨迹预测窗口时间"决定 Runner 的切向窗 ✓）。
            self._last_file = ""          # ⭐ "上次打开的文件"（`_cfg_apply` 回填 ✓）
            self._cfg_apply()

            # ⭐⭐ **记住上次打开的文件**（用户 2026-10-01 ✓ 原话："让窗口记住上次打开的文件" ✓）：
            #   显式 `--video`/`--gif` ⇒ 用它 ✗；否则回退到**上次打开的文件** ✓（再没有才退回
            #   目录里最新的 `DEF_VIDEO` ✗ —— 之前一直退回最新文件 ⇒ 用户明明看 `9月30日(1)` 却
            #   总被载入新的 `10月1日` ✗✗）。
            self._auto = str(args.video) if args.video else ""
            if not self._auto:
                self._auto = str(getattr(self, "_last_file", "") or "") or str(DEF_VIDEO)
            if self._auto:
                self.load(self._auto)

        # ---- 载入 / 预热 ----
        def open_dialog(self):
            start = str(VIDEO_DIR) if VIDEO_DIR.is_dir() else str(ROOT)
            f, _ = QFileDialog.getOpenFileName(self, "选一段测谎录像", start,
                                              VIDEO_EXT)
            if f:
                self.load(f)

        def pick_gif(self):
            """选 **GIF 动图**（`datasets/liedetectorgifs/*.gif` ✓ 用户 2026-09-30 改 ✓）。
            起始目录 = 上次选过的目录（连着看好几个 gif 时少点几下 ✓）。"""
            start = getattr(self, "_gif_root「, 」") or (
                str(GIF_DIR) if GIF_DIR.is_dir() else str(ROOT))
            f, _ = QFileDialog.getOpenFileName(self, "选一个 GIF 动图", start, GIF_EXT)
            if f:
                self._gif_root = str(Path(f).parent)
                self.load(f)

        def reload_current(self):
            if getattr(self, "_cur", ""):
                self.load(self._cur)

        def apply_path_ms(self):
            """⭐ **「应用」**：把「轨迹预测窗口时间"写进追踪器并**重新演算整段**（用户
            2026-10-01 ✓ 原话："点击后重新运算并生效" ✓）。

            ⚠ 为什么**不重跑检测**：检测框（`self.dets`）与这个参数**无关** ⇒ 复用它
            （`reuse=True` ✓）⇒ 只重跑"追踪 + 控制"那一段 ✓（相差一个数量级的耗时 ✓）。
            ⚠ 必须**整段重算** ✗ 不能只改参数接着播：切向拟合要用**新窗长回看历史** ⇒
            半路改窗长会让前后半段的轨迹不可比 ✓（播放/拖动查的还是那张表 ✓）。
            """
            if not self.frames:
                return
            self.stop()
            # ⭐⭐⭐ **运动分离（新）必须有 YOLO 检出框**（用户 2026-10-03 第 ⑤ 条："**还是需要
            #   依赖 yolo**，无论真假至少把目标定位出来" ✓）—— ⚠ 它的观测**就是检出框** ✗
            #   （`MotionTracker.process` 吃 `dets` ✓）⇒ 不勾"用检测器"就**完全没有观测** ✗
            #   ⇒ 一条候选轨迹都建不起来 ⇒ 白线根本不出 ✗（看着像"坏了" ✗）。
            #   ⇒ 这里**自动勾上**并告知 ✓（⚠ 用 `blockSignals` 包住 —— `chk_dets` 的
            #     `stateChanged` 会触发 `reload_current` ✗ 不挡会**递归重载** ✗✗）。
            #   ⚠ 口径 = **`_mv()`**（速度跟踪的底盘**同样只有检出框这一种观测** ✓ 用户 2026-10-07
            #     ✓ —— 它逐格读的就是 `box_v` ✓ 见 `VelocityChassis` ✓）⇒ 两档一起自动勾 ✓。
            if self._mv() and not self.chk_dets.isChecked():
                self.chk_dets.blockSignals(True)
                self.chk_dets.setChecked(True)
                self.chk_dets.blockSignals(False)
                self.statusBar().showMessage(
                    "⚠ 这一档的观测**就是 YOLO 检出框** ⇒ 已自动勾上「用检测器」✓", 10000)
            ms = int(self.sp_path_ms.value())
            mr = float(self.sp_merge.value())
            nt = float(self.sp_ntol.value())
            bs = float(self.sp_board.value())
            ov = float(self.sp_ov.value())
            # ⚠ 复用前先确认**检测框是齐的**（上次演算被打断 ⇒ 缺帧 ✗）：没齐就老实重检测 ✓
            #   （没勾"用检测器"⇒ 本来就不需要框 ⇒ 也走复用 ✓ 省一个检测子进程 ✓）。
            reuse = ((not self.chk_dets.isChecked())
                     or (bool(self.dets) and len(self.dets) >= len(self.frames)))
            self.lbl.setText("应用「轨迹预测窗口时间 = %d ms ／ 融合框判定 IoU = %.2f ／ "
                             "分离判定阈值 = %.2f ／ 融合框继承距离限制 = %.2f ／ "
                             "允许倒退距离 = %.2f ×半径 ／ 砖最多拥有边框数 = %d ／ "
                             "真目标预测最大速度倍率 = %.2f ／ "
                             "噪声容差 = %.1f px ／ "
                             "上板时长 = %.1f s ／ 上板防抖重叠率 = %.2f ／ "
                             "鼠标跟随效率倍率 = %.2f ／ 缩成砖最小 IoU = %.2f ／ "
                             "位置估计 = %s」"
                             "⇒ 重新演算 %d 帧…（%s）"
                             % (ms, mr, float(self.sp_sep.value()),
                                float(self.sp_inherit.value()), float(self.sp_back.value()),
                                int(self.sp_emax.value()),
                                float(self.sp_svmax.value()),
                                nt, bs, ov, float(self.sp_follow.value()),
                                float(self.sp_ioub.value()),
                                "开「 if self._show_kf() else 」关",
                                len(self.frames),
                                "检测框复用 ✓" if reuse else "重跑检测"))
            self.start_warmup(reuse=reuse)
            self._cfg_save()          # ⭐ 点"应用"= 这个值就是"我要的配置" ⇒ 落盘 ✓

        # ---- ⭐ 记住上次的配置（`config/ui.yaml` 的 `lie_demo` 段 ✓ 用户 2026-10-01）----
        def _cfg_apply(self):
            """回填上次的配置（**缺项/坏值一律保持默认** ✓ 绝不猜 ✗）。"""
            try:
                from gui import theme
                c = dict(theme.load_section("lie_demo") or {})
            except Exception:
                c = {}
            if not c:
                return
            # ① 轨迹预测窗口时间
            try:
                self.sp_path_ms.setValue(min(max(int(c.get("path_ms",
                                                            _PATH_MS_DEFAULT)), 50), 5000))
                # ⭐⭐⭐ **速度跟踪那三个参量回填**（用户 2026-10-07 ✓ 只这一档用 ✓）——
                #   ⚠ 每个都**各管各的 try** ✗（一个坏值不许把别的也带掉 ✓ 老约定 ✓）。
                self.sp_vcatch.setValue(min(max(int(c.get("vel_catch_px",
                                                          _VEL_CATCH_DEFAULT)), 10), 2000))
                self.sp_vhold.setValue(min(max(float(c.get("vel_hold_s",
                                                           _VEL_HOLD_DEFAULT)), 0.0), 30.0))
                self.sp_vdev.setValue(min(max(float(c.get("vel_dev_px",
                                                          _VEL_DEV_DEFAULT)), 0.0), 500.0))
            except Exception:
                pass
            # ①' 融合框判定 IoU
            try:
                # ⚠ 键名换成 `merge_iou`（用户 2026-10-03 ✓）；旧键 `merge_ratio`（面积比 ✓）**不再读** ✗
                #   —— 若读到 1.2 会被当成 IoU 1.2 ⇒ 恒不成立 ✗✗（夹进 [0,1] 后是 1.0 ⇒ 也几乎不成立 ✓）。
                self.sp_merge.setValue(min(max(float(c.get("merge_iou",
                                                             _MERGE_IOU_DEFAULT)), 0.0), 1.0))
                self.sp_merge_out.setValue(min(max(float(c.get("merge_iou_out",
                                                               _MERGE_IOU_OUT_DEFAULT)), 0.0), 1.0))
                # ⭐⭐⭐ **参数分档**（用户 2026-10-03 ✓ 第 2/3 行 ✓ 各自回填 ✓）
                self.sp_ring_sep.setValue(min(max(float(c.get("ring_cov_sep",
                                                              _RING_COV_SEP_DEFAULT)), 0.0), 1.0))
                self.sp_brick_sep.setValue(min(max(float(c.get("brick_iou_sep",
                                                               _BRICK_IOU_SEP_DEFAULT)), 0.0), 1.0))
                self.sp_ring_fuse.setValue(min(max(float(c.get("ring_cov_fuse",
                                                               _RING_COV_FUSE_DEFAULT)), 0.0), 1.0))
                self.sp_brick_fuse.setValue(min(max(float(c.get("brick_iou_fuse",
                                                                _BRICK_IOU_FUSE_DEFAULT)), 0.0), 1.0))
                # ⚠ "允许倒退"两处控件同步到同一个值（**同一个参数** ✓ 只存一份 ✓）
                _abv = min(max(float(c.get("allow_back_ratio", _ALLOW_BACK_RATIO_DEFAULT)),
                               0.0), 5.0)
                self.sp_back.setValue(_abv)
                self.sp_ab_fuse.setValue(_abv)
            except Exception:
                pass
            # ①'''' 融合框继承距离限制（用户 2026-10-02 ✓ 口径定稿：允许圆心出框的倍数半径 ✓）
            try:
                self.sp_inherit.setValue(min(max(float(c.get("inherit_dist",
                                                             _INHERIT_DIST_DEFAULT)),
                                                 0.0), 3.0))
            except Exception:
                pass
            # ①''''' 融合框挑选允许倒退距离(绿圆半径比例)（用户 2026-10-02 ✓ 第二条闸的容差 ✓）
            try:
                self.sp_back.setValue(min(max(float(c.get("allow_back_ratio",
                                                         _ALLOW_BACK_RATIO_DEFAULT)), 0.0), 5.0))
            except Exception:
                pass
            # ①'''''' 砖最多拥有融合框边数量（用户 2026-10-02 ✓ 1~4 ✓）
            try:
                self.sp_emax.setValue(min(max(float(c.get("edge_max", _EDGE_MAX_DEFAULT)),
                                              1.0), 4.0))
            except Exception:
                pass
            # ①''''''' 真目标预测最大速度倍率（用户 2026-10-03 ✓ 分离期白箭头模长上限 ✓）
            try:
                self.sp_svmax.setValue(min(max(float(c.get("sep_vel_max_ratio",
                                                           _SEP_VEL_MAX_DEFAULT)), 0.0), 5.0))
            except Exception:
                pass
            # ①''' 分离判定阈值（用户 2026-10-02 ✓ 融合期框面积 max/min 的比值门槛 ✓）
            try:
                self.sp_sep.setValue(min(max(float(c.get("sep_ratio",
                                                         _SEP_RATIO_DEFAULT)), 1.0), 5.0))
            except Exception:
                pass
            # ①'' 噪声容差(px)（用户 2026-10-02 ✓）
            try:
                self.sp_ntol.setValue(min(max(float(c.get("noise_tol",
                                                          _NOISE_TOL_DEFAULT)), 0.0), 100.0))
            except Exception:
                pass
            # ①''' 上板时长(秒)（用户 2026-10-02 ✓）
            try:
                self.sp_board.setValue(min(max(float(c.get("board_s",
                                                           _BOARD_S_DEFAULT)), 0.0), 10.0))
            except Exception:
                pass
            # ①'''' 上板防抖重叠率（用户 2026-10-02 ✓）
            try:
                self.sp_ov.setValue(min(max(float(c.get("board_overlap",
                                                       _BOARD_OVERLAP_DEFAULT)), 0.0), 1.0))
            except Exception:
                pass
            # ①''''' ''' 鼠标跟随效率倍率（用户 2026-10-03 ✓ 控制器侧 ✓ 不影响位置计算 ✓）
            try:
                self.sp_follow.setValue(min(max(float(c.get("follow_gain",
                                                            _FOLLOW_GAIN_DEFAULT)), 0.01), 5.0))
            except Exception:
                pass
            # ①'''''' 缩成砖判定 · 最小 IoU（用户 2026-10-03 ✓ IoU 尺 ✓ 0 = 关 ✓）
            try:
                self.sp_ioub.setValue(min(max(float(c.get("iou_brick",
                                                          _IOU_BRICK_DEFAULT)), 0.0), 1.0))
            except Exception:
                pass
            # ①''''''' ''' 显示 KF 预测（用户 2026-10-03 ✓ 开关 ✓ 缺项 = 关 ✓）
            try:
                self.chk_kf.setChecked(bool(c.get("show_kf", False)))
            except Exception:
                pass
            # ①'''''''''' **运动分离（新）那 5 个参数**（用户 2026-10-03 ✓ 见 `_cfg_save` 同名键 ✓）
            #   ⚠ 键名随算法换过两轮 ✓：旧键（`motion_q`/`motion_gate_k`/`motion_smooth`/
            #     `motion_vel_decay`/`motion_learn_pick` —— 那是"**颜色观测**"时代的 ✗）**不再读** ✗
            #     ⇒ 直接落新默认 ✓（免得把旧量纲的值当新参数用 ✗：例如旧的 `gate_k=2.5` 若被当成
            #     "配对门限" 用 ⇒ 2.5px ⇒ **一条都配不上** ✗✗）。
            try:
                self.sp_pair.setValue(min(max(float(c.get("motion_pair_gate",
                                                          _MOTION_PAIRED_DEFAULT)), 20.0), 200.0))
                self.sp_sdecay.setValue(min(max(float(c.get("motion_score_decay",
                                                            _MOTION_SDECAY_DEFAULT)), 0.5), 0.99))
                self.sp_margin.setValue(min(max(float(c.get("motion_switch_margin",
                                                            _MOTION_MARGIN_DEFAULT)), 0.0), 40.0))
                self.sp_minhits.setValue(min(max(float(c.get("motion_min_hits",
                                                             _MOTION_MINHITS_DEFAULT)), 1.0), 20.0))
                self.sp_whitew.setValue(min(max(float(c.get("motion_white_w",
                                                            _MOTION_WHITEW_DEFAULT)), 0.0), 5.0))
                # ---- ⭐ 第 5 行（跟踪手感 / 自检 / 粘连 ✓ 与 `_cfg_save` 同名键 ✓）----
                self.sp_smooth.setValue(min(max(float(c.get("motion_smooth",
                                                            _MOTION_SMOOTH_DEFAULT)), 0.0), 0.95))
                self.sp_qmin.setValue(min(max(float(c.get("motion_q_min",
                                                          _MOTION_QMIN_DEFAULT)), 0.5), 6.0))
                self.sp_qneed.setValue(min(max(float(c.get("motion_q_need",
                                                           _MOTION_QNEED_DEFAULT)), 1.0), 10.0))
                self.sp_stuck.setValue(min(max(float(c.get("motion_stuck_ratio",
                                                           _MOTION_STUCK_DEFAULT)), 1.05), 3.0))
                self.sp_predmax.setValue(min(max(float(c.get("motion_pred_max",
                                                             _MOTION_PREDMAX_DEFAULT)), 0.0), 120.0))
                self.sp_dirn.setValue(min(max(float(c.get("motion_dir_n",
                                                           _MOTION_DIRN_DEFAULT)), 4.0), 36.0))
                # ---- ⭐⭐⭐ **框心力度**（用户 2026-10-04 ✓ 见 `_FUSE_PULL` ✓）----
                #   ⚠⚠ 这里的上界**必须与建控件那处同一个数** ✗✗（用户 2026-10-05 ✓ "无法调到 1
                #     以上" ✓ 就是这两处都在夹 ✓）：只管控件不管这里 ⇒ 存进去 2.0、重开被夹回
                #     `1.0` ✗（看着像"改了不生效" ✓）。
                self.sp_fpull.setValue(min(max(float(c.get("motion_fuse_pull",
                                                           _MOTION_FPULL_DEFAULT)), 0.0), 3.0))
                # ---- ⭐⭐⭐⭐⭐ **夹取增益**（用户 2026-10-04 ✓ 见 `_CLAMP_GAIN` ✓）----
                self.sp_cgain.setValue(min(max(float(c.get("motion_clamp_gain",
                                                           _MOTION_CGAIN_DEFAULT)), 0.0), 1.0))
                # ⭐⭐⭐⭐⭐ **限定区域（ROI ✓ 用户 2026-10-07 ✓ 原话："**只在限定的区域计算（因为
                #   这是部分弹窗）**" ✓✓）** —— ⚠⚠ **单独 try** ✗：上面那一大块是"回填全部参数"的
                #   ✓，**别让一个坏 ROI 把其余参数连带跳过** ✗；非法值（数量不对 / 反了 / 退化）
                #   ⇒ **当没给** ✗（= 整幅 ✓ = 改动前的行为 ✓）。
                try:
                    _r = c.get("motion_roi")
                    if isinstance(_r, (list, tuple)) and len(_r) == 4:
                        _r4 = [float(_x) for _x in _r]
                        if _r4[2] > _r4[0] and _r4[3] > _r4[1]:
                            self.roi = tuple(_r4)
                except Exception:               # noqa: BLE001
                    pass
            except Exception:
                pass
            # ⭐ **算法模式**（用户 2026-10-03 ✓ 记住上次选的 ✓）—— ⚠ `setCurrentIndex` 会**自动
            #   触发** `currentIndexChanged` ⇒ `_on_mode_changed` 顺手把四条行的显隐摆正 ✓
            #   （那时控件全建好了 ✓ 安全 ✓）；键缺（老配置 ✓）⇒ 保持第 0 项 = 经典 ✓。
            try:
                _md = str(c.get("mode", "classic") or "classic")
                for _i in range(self.cmb_mode.count()):
                    if str(self.cmb_mode.itemData(_i)) == _md:
                        self.cmb_mode.setCurrentIndex(_i)
                        break
            except Exception:
                pass
            # ② 速度（按 `itemData` 找那一档 ✓ 别按下标 ✗ —— 档位可能变过 ✓）
            try:
                _sp = float(c["speed"])
                for _i in range(self.cmb_speed.count()):
                    if abs(float(self.cmb_speed.itemData(_i) or 1.0) - _sp) < 1e-6:
                        self.cmb_speed.setCurrentIndex(_i)
                        break
            except Exception:
                pass
            # ③ 蒙版
            try:
                self.sld_mask.setValue(min(max(int(c["mask"]), 0), 80))
            except Exception:
                pass
            # ④ 两个开关（⚠ `stateChanged` 会触发 `reload_current` ⇒ 那时还没素材 ⇒
            #    它早退 ✓ 安全 ✓）
            try:
                if "dets" in c:
                    self.chk_dets.setChecked(bool(c["dets"]))
                if "loop" in c:
                    self.chk_loop.setChecked(bool(c["loop"]))
            except Exception:
                pass
            # ⑤ ⭐ **上次打开的文件**（用户 2026-10-01 ✓ 原话："让窗口记住上次打开的文件" ✓）：
            #   存绝对路径 ✓（只当"候选"用 ✗ 载入前 `load()` 会再验"读得到" ✓ 读不到就不载 ✓）。
            try:
                self._last_file = str(c.get("file", "") or "")
            except Exception:
                self._last_file = ""

        def _cfg_save(self, *_a):
            """把当前配置写回 `config/ui.yaml` 的 `lie_demo` 段（开关/拖动条/「应用"都调 ✓）。

            ⚠ 写不进去（没权限/文件被占）**不许影响演示** ✓ —— 存偏好是附带的 ✓。
            """
            try:
                from gui import theme
                theme.save_section("lie_demo", {
                    "path_ms": int(self.sp_path_ms.value()),
                    # ⭐⭐⭐ **速度跟踪那三个参量**（用户 2026-10-07 ✓ 只在这一档显示 ✓）——
                    #   ⚠ 落**独立键名** ✗（别塞进 `motion_*` 那组 ✓ 那一组是老底盘的 ✓
                    #     混了以后换模式会互相覆盖 ✓）；⚠⚠ 判据本轮还没接 ⇒ 先只落盘 ✓。
                    "vel_catch_px": float(self.sp_vcatch.value()),
                    "vel_hold_s": float(self.sp_vhold.value()),
                    "vel_dev_px": float(self.sp_vdev.value()),
                    # ⭐ **融合框判定 IoU**（用户 2026-10-03 ✓ 键名同步 ✓）
                    "merge_iou": round(float(self.sp_merge.value()), 3),
                    # ⭐ **融合框判定 IoU · 退出**（迟滞下半段 ✓ 用户 2026-10-03 ✓）
                    "merge_iou_out": round(float(self.sp_merge_out.value()), 3),
                    # ⭐⭐⭐ **参数分档**（用户 2026-10-03 ✓ 第 2/3 行 ✓ 与"融合系"一起落盘 ✓）
                    "ring_cov_sep": round(float(self.sp_ring_sep.value()), 3),
                    "brick_iou_sep": round(float(self.sp_brick_sep.value()), 3),
                    "ring_cov_fuse": round(float(self.sp_ring_fuse.value()), 3),
                    "brick_iou_fuse": round(float(self.sp_brick_fuse.value()), 3),
                    # ⭐ **分离判定阈值**（用户 2026-10-02 ✓ 与"融合框判定阈值"一起落盘 ✓）
                    "sep_ratio": round(float(self.sp_sep.value()), 3),
                    # ⭐ **融合框继承距离限制**（用户 2026-10-02 ✓ "融合系"参数一起落盘 ✓）
                    "inherit_dist": round(float(self.sp_inherit.value()), 3),
                    # ⭐ **融合框挑选允许倒退距离(绿圆半径比例)**（用户 2026-10-02 ✓ 同上 ✓）
                    "allow_back_ratio": round(float(self.sp_back.value()), 3),
                    # ⭐ **砖最多拥有融合框边数量**（用户 2026-10-02 ✓ 1~4 ✓ "融合系"一起落盘 ✓）
                    "edge_max": int(self.sp_emax.value()),
                    # ⭐ **真目标预测最大速度倍率**（用户 2026-10-03 ✓ 分离期白箭头模长上限 ✓）
                    "sep_vel_max_ratio": round(float(self.sp_svmax.value()), 3),
                    "noise_tol": round(float(self.sp_ntol.value()), 2),
                    "board_s": round(float(self.sp_board.value()), 2),
                    "board_overlap": round(float(self.sp_ov.value()), 3),
                    # ⭐ **鼠标跟随效率倍率**（用户 2026-10-03 ✓ 控制器侧 ✓ 与"融合系"一起落盘 ✓）
                    "follow_gain": round(float(self.sp_follow.value()), 3),
                    # ⭐ **缩成砖判定·最小 IoU**（用户 2026-10-03 ✓ IoU 尺 ✓ 与"融合系"一起落盘 ✓）
                    "iou_brick": round(float(self.sp_ioub.value()), 3),
                    # ⭐ **显示 KF 预测**（用户 2026-10-03 ✓ 开关 ✓ 与参数一起落盘 ✓）
                    "show_kf": bool(self.chk_kf.isChecked()),
                    # ⭐⭐⭐ **算法模式**（用户 2026-10-03 ✓ 原话："加个模式下拉列表" ✓ ——
                    #   记住上次选的 ✓ 免得每次重选 ✓）
                    "mode": str(self.cmb_mode.currentData() or "classic"),
                    # ⭐⭐⭐ **运动分离（新）那 5 个参数**（用户 2026-10-03 ✓ 原话："运动分离
                    #   **没有任何参数要配吗？**" ✓）—— 与其余参数一起落盘 ✓。
                    #   ⚠ 键名随算法一起换过一轮 ✓（旧的是 `q/gate_k/smooth/vel_decay/learn_pick`
                    #   —— 那是"颜色观测"时代的参数 ✗ 现在**不再读** ✗ 直接落新默认 ✓）。
                    "motion_pair_gate": round(float(self.sp_pair.value()), 1),
                    "motion_score_decay": round(float(self.sp_sdecay.value()), 3),
                    "motion_switch_margin": round(float(self.sp_margin.value()), 2),
                    "motion_min_hits": int(round(float(self.sp_minhits.value()))),
                    "motion_white_w": round(float(self.sp_whitew.value()), 2),
                    # ---- ⭐⭐ **第 5 行**（跟踪手感 / 自检 / 粘连 ✓ 见 `_cfg_load` 同名键 ✓）----
                    "motion_smooth": round(float(self.sp_smooth.value()), 3),
                    "motion_q_min": round(float(self.sp_qmin.value()), 2),
                    "motion_q_need": int(round(float(self.sp_qneed.value()))),
                    "motion_stuck_ratio": round(float(self.sp_stuck.value()), 3),
                    "motion_pred_max": round(float(self.sp_predmax.value()), 1),
                    # ---- ⭐ **方向数 N**（用户 2026-10-04 ✓ 见 `_cfg_load` 同名键 ✓）----
                    "motion_dir_n": int(round(float(self.sp_dirn.value()))),
                    # ---- ⭐⭐⭐ **框心力度**（用户 2026-10-04 ✓ 见 `_cfg_load` 同名键 ✓）----
                    "motion_fuse_pull": round(float(self.sp_fpull.value()), 3),
                    # ---- ⭐⭐⭐⭐⭐ **夹取增益**（用户 2026-10-04 ✓ 见 `_cfg_load` 同名键 ✓）----
                    "motion_clamp_gain": round(float(self.sp_cgain.value()), 3),
                    # ⭐⭐⭐⭐⭐ **限定区域（ROI ✓ 用户 2026-10-07 ✓ 原话："**只在限定的区域计算
                    #   （因为这是部分弹窗）**" ✓✓）** —— ⚠ 键名与 `live_lie._MOTION_CFG_MAP`
                    #   那条读法**对上** ✓（`motion_roi` ✓ 4 个数 ✓）；**没框就存 `[]`** ✓
                    #   （= 整幅 ✓ ⇒ 下次打开行为与这次一致 ✓）。
                    "motion_roi": ([round(float(_v), 1) for _v in self.roi]
                                   if getattr(self, "roi", None) else []),
                    "speed": float(self.cmb_speed.currentData() or 1.0),
                    "mask": int(self.sld_mask.value()),
                    "dets": bool(self.chk_dets.isChecked()),
                    "loop": bool(self.chk_loop.isChecked()),
                    # ⭐ **上次打开的文件**（用户 2026-10-01 ✓ 原话："让窗口记住上次打开的文件" ✓）
                    "file": str(getattr(self, "_cur", "") or ""),
                })
            except Exception:
                pass

        def _path_ms(self):
            """当前界面上的「轨迹预测窗口时间」（毫秒 ✓ 没建控件时 = 默认值 ✓）。"""
            return int(getattr(self, "sp_path_ms", None).value()
                       if getattr(self, "sp_path_ms", None) is not None
                       else _PATH_MS_DEFAULT)

        def _merge_iou(self):
            """当前界面上的「**融合框判定 IoU**"（用户 2026-10-03 ✓ 判据 = `IoU(框, 砖) < 它`
            ✓ 没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_merge", None).value()
                         if getattr(self, "sp_merge", None) is not None
                         else _MERGE_IOU_DEFAULT)

        def _sync_ab(self, _src, _dst, _v):
            """「**允许倒退距离**"两处控件**双向同步**（用户 2026-10-03 ✓ 第 2/3 行都写了它 ✓
            但它是**同一个参数** ⇒ 改任一处另一处跟着变 ✓；用守卫位防互相触发死循环 ✗）。"""
            if getattr(self, "_sync_ab_guard", False):
                return
            self._sync_ab_guard = True
            try:
                if abs(float(_dst.value()) - float(_v)) > 1e-9:
                    _dst.setValue(float(_v))
            finally:
                self._sync_ab_guard = False

        # ---- ⭐⭐⭐ **参数分档的取值**（用户 2026-10-03 ✓ 第 2/3 行）----
        def _ring_cov_sep(self):
            """分离期「框↔圆外接矩形 IoU >」（没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_ring_sep", None).value()
                         if getattr(self, "sp_ring_sep", None) is not None
                         else _RING_COV_SEP_DEFAULT)

        def _brick_iou_sep(self):
            """分离期「框↔内砖 IoU <」（**进严** ✓ 没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_brick_sep", None).value()
                         if getattr(self, "sp_brick_sep", None) is not None
                         else _BRICK_IOU_SEP_DEFAULT)

        def _ring_cov_fuse(self):
            """融合期「框↔圆外接矩形 IoU >」（没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_ring_fuse", None).value()
                         if getattr(self, "sp_ring_fuse", None) is not None
                         else _RING_COV_FUSE_DEFAULT)

        def _brick_iou_fuse(self):
            """融合期「框↔内砖 IoU <」（**守松** ✓ 设 1.00 ⇒ 关掉「融合期退出」 ✓）。"""
            return float(getattr(self, "sp_brick_fuse", None).value()
                         if getattr(self, "sp_brick_fuse", None) is not None
                         else _BRICK_IOU_FUSE_DEFAULT)

        def _merge_iou_out(self):
            """当前界面上的「**融合框判定 IoU · 退出**"（用户 2026-10-03 ✓ 迟滞 ✓ 没建控件时
            = 默认值 ✓）。"""
            return float(getattr(self, "sp_merge_out", None).value()
                         if getattr(self, "sp_merge_out", None) is not None
                         else _MERGE_IOU_OUT_DEFAULT)

        def _sep_ratio(self):
            """当前界面上的「**分离判定阈值**"（用户 2026-10-02 ✓ 分离信号第三条的 x ✓
            没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_sep", None).value()
                         if getattr(self, "sp_sep", None) is not None
                         else _SEP_RATIO_DEFAULT)

        def _inherit_dist(self):
            """当前界面上的「**融合框继承距离限制**」（用户 2026-10-02 ✓ 口径已改为"圆心到框最近
            一条边的垂距 ÷ 绿圈半径" ✓ 融合框候选第一条闸 ✓ 没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_inherit", None).value()
                         if getattr(self, "sp_inherit", None) is not None
                         else _INHERIT_DIST_DEFAULT)

        def _allow_back_ratio(self):
            """当前界面上的「**融合框挑选允许倒退距离(绿圆半径比例)**"（用户 2026-10-02 ✓
            第二条闸的容差 ✓ 没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_back", None).value()
                         if getattr(self, "sp_back", None) is not None
                         else _ALLOW_BACK_RATIO_DEFAULT)

        def _edge_max(self):
            """当前界面上的「**砖最多拥有融合框边数量**"（用户 2026-10-02 ✓ 1~4 ✓ 融合框四边的
            「边归属」只留距离最近的前它条 ✓ 没建控件时 = 默认值 ✓）。"""
            return int(getattr(self, "sp_emax", None).value()
                       if getattr(self, "sp_emax", None) is not None
                       else _EDGE_MAX_DEFAULT)

        def _sep_vel_max(self):
            """当前界面上的「**真目标预测最大速度倍率**」（用户 2026-10-03 ✓ 分离期白箭头模长
            上限 = 本项 × 当拍群体速度模长 ✓ 没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_svmax", None).value()
                         if getattr(self, "sp_svmax", None) is not None
                         else _SEP_VEL_MAX_DEFAULT)

        def _noise_tol(self):
            """当前界面上的「噪声容差(px)」（规则①：「圆心 ± 它」的范围与框相交 ✓ 没建控件时 =
            默认值 ✓）。"""
            return float(getattr(self, "sp_ntol", None).value()
                         if getattr(self, "sp_ntol", None) is not None
                         else _NOISE_TOL_DEFAULT)

        def _board_s(self):
            """当前界面上的「上板时长(秒)」（登记条目转正的时长门 ✓ 没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_board", None).value()
                         if getattr(self, "sp_board", None) is not None
                         else _BOARD_S_DEFAULT)

        def _board_overlap(self):
            """当前界面上的「上板防抖重叠率」（用户 2026-10-02 ✓ 没建控件时 = 默认值 0.5 ✓）。"""
            return float(getattr(self, "sp_ov", None).value()
                         if getattr(self, "sp_ov", None) is not None
                         else _BOARD_OVERLAP_DEFAULT)

        def _follow_gain(self):
            """当前界面上的「**鼠标跟随效率倍率**"（用户 2026-10-03 ✓ 只作用在**控制器**的
            指令输出上 ✓ 不碰位置计算 ✓ 没建控件时 = 默认值 1.0 ✓）。"""
            return float(getattr(self, "sp_follow", None).value()
                         if getattr(self, "sp_follow", None) is not None
                         else _FOLLOW_GAIN_DEFAULT)

        def _explain_conf(self):
            """⭐ **「匹配分」的统一说明处**（用户 2026-10-04 ✓ 原话："我们需要有**统一的地方管理
            conf 都与什么有关、每一项都是如何作用的**，建议**加个 `？` 写在里面**" ✓✓）。

            ⚠⚠ **这里只是"给人看的说明"** ✗ —— **真正算它的地方只有一处** ✓：
            `perception/lie_motion.py` 的 `process` 打分段（搜 `匹配分` 就能跳过去 ✓）。
            改公式时**两边一起改** ✓。
            """
            QMessageBox.information(self, "匹配分（观测可信度）是怎么算出来的",
                "<b>一句话</b>：它是「<b>这一格框配到这条轨迹上，这次配对有多可信</b>」"
                "（0 = 完全不信，1 = 完全可信）。<br><br>"
                "<b>公式</b>：<code>匹配分 = 距离因子 × 面积因子</code>"
                "（都取 0~1 ⇒ 任何一项差，整体就被拉低）<br><br>"
                "<b>① 距离因子</b> = 1 − 距预测位置 ÷ 配对门限<br>"
                "&nbsp;&nbsp;&nbsp;· 离「我以为它该在的位置」越近 ⇒ 越可信（会被「配对门限」影响）<br>"
                # ⭐⭐⭐⭐⭐ **新增一条**（用户 2026-10-05 ✓ 原话："**他都跟圆相交了，它应该被认为
                #   "离预测真目标近"**" ✓✓ ｜ "**说明你的匹配分算的有问题**" ✓✓）——
                #   起因是实测：前 76 拍里**跟圆相交的框 53 个、其中 40 个（75%）**被判"位置可疑"
                #   ✗ —— 因为旧公式量的是"离**这条轨迹**的预测多远" ✗ 而不是"离**真目标**多近" ✓。
                "&nbsp;&nbsp;&nbsp;· ⚠ <b>若这格框罩着圆</b>（圆 = 我们相信真目标所在的地方）"
                "⇒ 这一项<b>直接记 1</b>（框里就是真目标 ⇒ 位置当然可信 ✓）<br>"
                "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（⚠ 这条**只管距离** ✗：融合大框照样会被"
                "「面积因子」压到 0 —— 那种框的框心是**两个目标的中间点** ✓）<br><br>"
                "<b>② 面积因子</b>（真正关键的那项）：<br>"
                "&nbsp;&nbsp;&nbsp;· 框面积 ÷ <b>标准面积</b> 落在 0.75~1.0 ⇒ 满分 1.0<br>"
                "&nbsp;&nbsp;&nbsp;· <b>超出</b>（框变大 = 两个目标粘成一个合体）⇒ 按超出比例"
                "<b>线性压到 0</b><br>"
                "&nbsp;&nbsp;&nbsp;· 为什么这么狠：粘连时 <b>框中心 = 两个目标的中间点</b>，"
                "这个观测量<b>根本不能信</b><br>"
                "&nbsp;&nbsp;&nbsp;· 「标准面积」= 最近 240 个框面积的中位（实测 ≈19000，"
                "白星时期也是这个数）<br><br>"
                "<b>③ 什么时候它压根不参与</b>：<br>"
                "&nbsp;&nbsp;&nbsp;· <b>粘连 / 冷却期</b>：整条观测链都不采信"
                "（位置改成沿预测滑、速度不更新）<br>"
                "&nbsp;&nbsp;&nbsp;· <b>没配上</b>：位置按「相机 + 它自己的相对速度」推<br><br>"
                "<b>怎么用</b>（你 ④⑤ 那两条）：<br>"
                "&nbsp;&nbsp;&nbsp;· <b>骤降</b> ⇒ 这一拍该<b>跟着 KF 走</b>（别信观测）<br>"
                "&nbsp;&nbsp;&nbsp;· <b>回升</b> ⇒ 可以修正轨迹<br><br>"
                "<b>画面上看</b>：每个检出框标签是 <code>shape 0.98 #20 m0.55</code>"
                "（末尾的 <code>m</code> 就是匹配分）；低于 0.3 基本不可信。")

        def _on_roi_mode(self, on):
            """「框选区域」⇒ 开**工作台那一套框选窗**（用户 2026-10-07 ✓ 原话：

            「**框选区域要有放大，参考数据工作台框选功能**」✓ ⇒ 直接用
            `gui.region_picker.LoupeSelector` ✓（**放大镜 ＋ 像素网格 ＋ `+/-` 倍数 ＋ `Esc` 取消**
            ✓ 见 `docs/UI规范.md` §8 ✓）—— 同一份代码、同一个手感 ✓ 不另写一套 ✗。
            ⚠ 框的是**当前那一帧原图**（显示域 ✓）⇒ 报回来的是**显示域矩形** ⇒ 这里**除以
              `self.scale`** 换成**加工域** ✓（与判据 / 画布同一把尺 ✓ 见 `_on_roi_picked` ✓）。
            ⚠ 画布上那套"直接在画布上拖"（`set_roi_mode` ✓）**保留** ✓：没素材 / 框选窗开不出来时
              才退回去 ✓（自检也钉着它 ✓ —— ⚠ "不留死代码"那条纪律在这里让位给"退路要留" ✓）。
            """
            if not on:
                self.statusBar().showMessage("已退出框选模式 ✓", 3000)
                return
            self.btn_roi.setChecked(False)          # 按一下就弹窗 ⇒ 立刻回弹 ✓（不是持续模式 ✓）
            if not self.frames:
                self.statusBar().showMessage("还没有素材 ⇒ 先打开一个文件 ✓", 5000)
                return
            from gui.region_picker import select_on_pixmap
            try:
                _i = self.i if 0 <= self.i < len(self.frames) else 0
                _big = decode_big(self.frames[_i][0])
                _rgb = cv2.cvtColor(_big, cv2.COLOR_BGR2RGB)
                _h, _w, _ = _rgb.shape
                _qi = QImage(_rgb.data, _w, _h, 3 * _w, QImage.Format_RGB888).copy()
                #   ⚠ 起始倍数给 **4**（工作台默认 8 ✓ 那是给 HP/MP 条那种细目标用的 ✓）——
                #     这里框的是"屏幕上一块弹窗" ✓ 8 倍反而看不清周边 ✓。
                _r = select_on_pixmap(QPixmap.fromImage(_qi), parent=self, zoom=4)
            except Exception as _e:                 # noqa: BLE001 —— 开不出来 ⇒ **退路** ✓
                log_exception("框选区域（放大镜）", _e)
                try:
                    self.lbl.set_roi_mode(True)
                    self.btn_roi.setChecked(True)
                    self.statusBar().showMessage(
                        "开不了放大镜框选 ⇒ 改在画面上直接拖框 ✓", 6000)
                except Exception:                   # noqa: BLE001
                    pass
                return
            if not _r:
                self.statusBar().showMessage("已取消框选 ✓", 3000)
                return
            #   显示域 ⇒ **加工域** ✓（`self.scale` 是"原图 ÷ 加工域" ✓ 与画布同一把尺 ✓）
            self._on_roi_picked((_r[0] / self.scale[0], _r[1] / self.scale[1],
                                 (_r[0] + _r[2]) / self.scale[0],
                                 (_r[1] + _r[3]) / self.scale[1]))

        def _on_roi_clear(self):
            """清除限定区域 ⇒ **恢复整幅** ✓（= 回到改动前的行为 ✓）。"""
            self.roi = None
            try:
                self.lbl.set_roi_mode(False)
                self.btn_roi.setChecked(False)
            except Exception:                       # noqa: BLE001
                pass
            self._cfg_save()
            if self.frames:
                self.start_warmup(reuse=True)       # ⚠ 复用检测框 ⇒ 只重跑追踪 ✓ 很快 ✓
            self.statusBar().showMessage("已清除限定区域 ⇒ 恢复整幅 ✓", 5000)

        def _on_roi_picked(self, _rect):
            """画布松手报了**加工域矩形** ⇒ 生效 ＋ 落盘 ＋ 重算（⚠ 复用检测框 ✓ 很快 ✓）。

            ⚠ 太小的框当没框 ✗（手一抖点一下就会给个 1px 的框 ⇒ 全看不见 ⇒ 像坏了 ✓）。
            """
            _r = tuple(float(_v) for _v in _rect)
            if (_r[2] - _r[0] < 20.0) or (_r[3] - _r[1] < 20.0):
                self.statusBar().showMessage("框太小（< 20px）⇒ 当作没框 ✗ 请重拖 ✓", 5000)
                return
            self.roi = _r
            try:
                self.btn_roi.setChecked(False)      # 拖完就退出框选模式 ✓（光标还原 ✓）
            except Exception:                       # noqa: BLE001
                pass
            self._cfg_save()
            if self.frames:
                self.start_warmup(reuse=True)
            self.statusBar().showMessage(
                "限定区域 = (%.0f,%.0f)-(%.0f,%.0f) ⇒ **只在框里算** ✓（框外一概看不见 ✓）"
                % (_r[0], _r[1], _r[2], _r[3]), 10000)

        def _motion_kw(self):
            """**运动分离（新）**的界面参数 ⇒ 一个 dict（直接喂 `MotionRunner` ✓）。

            用户 2026-10-03 ✓ 原话："运动分离 **没有任何参数要配吗？**" ✓ ⇒ 给它配第 4 行 ✓。
            ⚠ **只给"手感/权衡档"** ✗ —— "物理契约档"（面积带 `_AREA_LO/HI` / 填充度
              `_FILL_LO/HI` ✓ 见 `lie_motion` 顶上 ✓）**不给** ✓：那几个是"**什么算一个块**"的
              定义 ✓（按本素材那个白星的形状定的 ✓）⇒ 调了会认错东西 ✗（想用在别的素材上，
              改 `perception/lie_motion.py` 顶上那几个常量 ✓ 比放界面上安全 ✓）。
            ⚠⚠ **`_AREA_LO` 就是这条规矩的实例** ✓（用户 2026-10-05 ✓ 原话："**为什么找到了 明显
              属于假目标的检出框**" ✓✓）：它原来写 **300** ✗（画面 0.08% ⇒ 小亮斑也算白块 ⇒
              目标被劫持 ✓ 见 `lie_motion._AREA_LO` 那段实测 ✓）⇒ 已改成 **2000** ✓
              **只动常量、没上界面** ✓（本行上面那条规矩 ✓）。
            ⚠ 没建控件（离屏自检 / 老配置 ✓）⇒ 一律回默认值 ✓ 不崩 ✓。
            """
            def _v(_n, _d):
                _w = getattr(self, _n, None)
                return float(_w.value()) if _w is not None else float(_d)
            return {"pair_gate": _v("sp_pair", _MOTION_PAIRED_DEFAULT),
                    "score_decay": _v("sp_sdecay", _MOTION_SDECAY_DEFAULT),
                    "switch_margin": _v("sp_margin", _MOTION_MARGIN_DEFAULT),
                    "min_hits": max(1, int(round(_v("sp_minhits", _MOTION_MINHITS_DEFAULT)))),
                    "white_w": _v("sp_whitew", _MOTION_WHITEW_DEFAULT),
                    # ---- ⭐⭐ **第 5 行**（跟踪手感 / 自检 / 粘连 ✓ 用户 2026-10-03 ✓）----
                    "smooth": _v("sp_smooth", _MOTION_SMOOTH_DEFAULT),
                    "q_min": _v("sp_qmin", _MOTION_QMIN_DEFAULT),
                    "q_need": max(1, int(round(_v("sp_qneed", _MOTION_QNEED_DEFAULT)))),
                    "stuck_ratio": _v("sp_stuck", _MOTION_STUCK_DEFAULT),
                    "kf_pred_max": _v("sp_predmax", _MOTION_PREDMAX_DEFAULT),
                    # ---- ⭐ **方向数 N**（用户 2026-10-04 ✓ 见 `_DIR_N` ✓）----
                    "dir_n": max(4, int(round(_v("sp_dirn", _MOTION_DIRN_DEFAULT)))),
                    # ---- ⭐⭐⭐ **框心力度**（用户 2026-10-04 ✓ 见 `_FUSE_PULL` ✓）----
                    "fuse_pull": _v("sp_fpull", _MOTION_FPULL_DEFAULT),
                    # ⭐⭐⭐⭐⭐ **夹取增益**（用户 2026-10-04 ✓ 见 `_CLAMP_GAIN` ✓）——
                    #   ⚠ 与 `fuse_clamp`（夹取**开/关** ✓）**正交** ✗：这个是"夹取那份修正
                    #     记进速度多少" ✓。
                    "clamp_gain": _v("sp_cgain", _MOTION_CGAIN_DEFAULT),
                    # ⭐⭐⭐⭐⭐ **限定区域（ROI ✓ 用户 2026-10-07 ✓ 原话："**只在限定的区域计算
                    #   （因为这是部分弹窗）**" ✓✓）** —— ⚠⚠ **它不是一个数** ✗（是 4 元组 ✓）
                    #   ⇒ 不走上面那个 `_v(...)` ✓（那是"控件取数" ✓）；`None` = 整幅 ✓
                    #   （**与改动前逐位一致** ✓ —— 你没框之前行为一个字不变 ✓）。
                    "roi": getattr(self, "roi", None)}

        def _show_layers(self):
            """三个**图层开关**的当前状态（用户 2026-10-03 ✓ "这些细细的蓝色轨迹线是什么？
            **我从未要求加过**" ⇒ 与其我猜 ✗，交给你自己开关 ✓）。

            回 `{"boxes": 检出框, "rel_tgt": **真目标**那条相对轨迹,
            "rel_mouse": **鼠标**那条相对轨迹, "panel": 信息面板/HUD}` ✓；
            ⚠ 没建控件（离屏自检 ✓）⇒ 全 `True`（老行为 ✓ 自检不受影响 ✓）。
            ⚠⚠ **`rel` 在 2026-10-04 拆成 `rel_tgt` / `rel_mouse` 两个键** ✗（用户要求的 ✓
              ⇒ "想只看目标那条"以前做不到 ✓）；`draw()` 那边**两处分别读各自的键** ✓。
            """
            def _v(_n, _d):
                _w = getattr(self, _n, None)
                return bool(_w.isChecked()) if _w is not None else bool(_d)
            return {"boxes": _v("chk_boxes", False),
                    "rel_tgt": _v("chk_rtgt", True),
                    "rel_mouse": _v("chk_rmouse", True),
                    # ⭐⭐⭐ **「显示检出框编号」独立开关**（用户 2026-10-04 ✓）——
                    #   ⚠ 只管"**框上面那行字**"（`shape 0.98 #22  m0.82` ✓）✗ 不管框、不管箭头 ✓
                    #     ⇒ 与 `boxes` **正交**（可以"留框不要字" ✓ 见 `chk_boxid` 的 tooltip ✓）。
                    #   ⚠ 没建控件（离屏自检 ✓）⇒ `True`（老行为 ✓ 自检不受影响 ✓）。
                    "box_id": _v("chk_boxid", True),
                    # ⭐⭐⭐ **信息面板 / HUD 开关**（用户 2026-10-04 ✓ 原话："**左上角这些框里
                    #   是什么，太挡视野了**" ✓✓）—— 左上角那两块（`t=2.50s track hit=100%`
                    #   那行 HUD ✓ ＋ 它下面那块黑底图例面板 ✓）**挡得最狠** ✓
                    #   ⇒ 交给他一键收掉 ✓（⚠ 默认**开** ✓ 因为图例是"看懂画面"的钥匙 ✓
                    #     想清屏时关掉即可 ✓）。
                    "panel": _v("chk_panel", True)}

        def _on_layer_toggle(self, *_a):
            """图层开关变了 ⇒ **落盘 + 立刻重画当前帧** ✓。

            ⚠⚠ 用户 2026-10-03 报："**关掉候选圈什么变化也没有**" ✗ —— 因为原来只接
              `self._cfg_save`（**只落盘 ✗ 不刷新画面** ☠）⇒ 得**拖一下/播放**才生效
              ⇒ 看着就像"开关坏了" ✓。现在**勾/取消的瞬间就重画** ✓（走 `_show_precomputed`
              ✓ 只重画**当前这一帧** ⇒ 零等待、不重算 ✓）。
            """
            self._cfg_save()
            try:
                self._show_precomputed()
            except Exception:                 # noqa: BLE001 —— 画不出来也不许把开关搞崩 ✗
                pass
            # ⭐ info_col 显隐 = chk_panel 勾上 + motion/velocity 档（图例是那两档的 ✓ 经典藏 ✓）
            if getattr(self, "info_col", None) is not None:
                self.info_col.setVisible(
                    bool(self.chk_panel.isChecked()) and self._mv())

        def _update_info_live(self, motion):
            """把 motion 的实时数据 set 到侧栏 `info_live` QLabel（用户 2026-10-04 ✓ 侧栏 ✓）。
            ⚠ 只在 motion 模式有数据 ✓ 经典模式 info_live 保持「（待演算）」✓。
            ⚠ 速度跟踪档 ⇒ **整块清空**（见下面那段 ✓）。"""
            if getattr(self, "info_live", None) is None:
                return
            if not motion:
                self.info_live.setText("（待演算）")
                return
            # ⭐⭐⭐ **速度跟踪：实时信息整块清空**（用户 2026-10-07 ✓ "做减法：清空三处显示" ✓）——
            #   ⚠ 口径 = **只在速度跟踪档清** ✗（运动分离那 5 行**一个字不改** ✓ 零污染 ✓）。
            if self._vel():
                self.info_live.setText("")
                return
            _trk = list(motion.get("tracks") or [])
            _med = motion.get("median") or (0.0, 0.0)
            _lines = ["相机 (%+.0f,%+.0f) %.0fpx | 噪声底 %.1fpx | %d 候选（%s 真配上）"
                      % (_med[0], _med[1], float(motion.get("cam_len") or 0.0),
                         float(motion.get("dev") or 0.0),
                         len(_trk), motion.get("n_live", "?"))]
            if motion.get("sel_score") is not None:
                _sel = next((t for t in _trk if t.get("sel")), None)
                _lines.append("目标 #%s 分 %.1f 偏离 %.1f 命中 %s %s"
                              % (next((str(t.get("tid")) for t in _trk if t.get("sel")), "?"),
                                 float(motion["sel_score"]),
                                 float(motion.get("sel_dev") or 0.0),
                                 next((str(t.get("hits")) for t in _trk if t.get("sel")), "?"),
                                 "LIVE" if (_sel and _sel.get("live")) else "推的"))
            else:
                _lines.append("目标（还没选出）")
            _top = sorted(_trk, key=lambda t: -float(t.get("score") or 0.0))[:3]
            for _k, _t in enumerate(_top):
                _lines.append("#%d 分 %.1f 偏离 %.1f @(%d,%d)%s%s"
                              % (_k + 1, float(_t.get("score") or 0.0),
                                 float(_t.get("dev") or 0.0), _t["p"][0], _t["p"][1],
                                 " <目标" if _t.get("sel") else "",
                                 "" if _t.get("live") else " [推的]"))
            self.info_live.setText("\n".join(_lines))

        def _mode(self):
            """当前算法模式（没建控件 ⇒ `classic` ✓ 老行为 ✓）。"""
            _c = getattr(self, "cmb_mode", None)
            return str(_c.currentData() or "classic") if _c is not None else "classic"

        def _mv(self):
            """这一档**走不走「运动分离那套」底盘 / 侧栏**（`motion` ＋ `velocity` ✓ 用户 2026-10-07 ✓）。

            ⚠ 两档**共用同一条底盘**（`MotionRunner` + `MotionTracker` ✓）⇒ 侧栏 / 参数行 / 取帧
              都照同一套走 ✓；**差别只在一个地方** = `draw()` 里那道 `_vel` 闸 ✓（速度跟踪做减法 ✓）。
            """
            return self._mode() in ("motion", "velocity")

        def _vel(self):
            """是不是**速度跟踪**档（做减法那一档 ✓ 见 `_LEGEND_VELOCITY` ✓）。"""
            return self._mode() == "velocity"

        def _legend_text(self):
            """当前该显示哪份图例（速度跟踪 ⇒ **只那 7 条** ✓ 其余 ⇒ 运动分离那 16 行 ✓）。"""
            return _LEGEND_VELOCITY if self._vel() else _LEGEND_MOTION

        def _sync_legend(self):
            """把右栏图例摆成当前模式该有的那份（⚠ **切模式时必须调** ✗ 否则留着上一档的图例 ✓）。"""
            if getattr(self, "info_legend", None) is not None:
                self.info_legend.setText(self._legend_text())

        def _on_mode_changed(self, *_a):
            """切模式 ⇒ **整行显示 / 隐藏**那批「新模式用不上"的参数（用户 2026-10-03 ✓ 原话见下）。

            用户原话："你可以在测谎演示里加个**模式下拉列表**，**新的算法模式选中时，不必要的
            参数配置都隐藏**" ✓。
            ⚠ 隐藏的是**整行**（第 2/3 行 = 融合 / 内砖 / 边归属 / 允许倒退 / 继承距离那一整组 ✓）
              —— 逐控件藏会留下一排空标签 ✗ 更难看 ✓。
            ⚠ **不自动重算** ✗（与其它参数同一个约定 ✓ 改完点"应用"✓ 免得卡界面 ✓）。
            """
            # ⚠ 口径 = **`_mv()`**（`motion` ＋ `velocity` 两档共用同一条底盘 ✓ 见 `_mv` ✓）——
            #   用户 2026-10-07 ✓：速度跟踪也**不用**"融合 / 内砖 / 边归属 / 允许倒退 / 继承距离"
            #   那一组 ✗ ⇒ 与运动分离**同样整行收起** ✓；它真在用的那批（第 4/5 行的
            #   `MotionTracker` 参数 ✓）**照样放出** ✓（底盘就是同一个 tracker ✓）。
            #   ⚠ 经典档 ⇒ `_motion` 为假 ⇒ 老行为**一个字不变** ✓（零污染 ✓）。
            _motion = self._mv()
            #   ⚠⚠ **速度跟踪要再收一层** ✗（用户 2026-10-07 ✓："清除通用参数外所有的配置参数" ✓）
            #     ⇒ 老底盘那两行（第 4/5 行 ✓）它也不用 ✓ 见下面 `_mo_trk` ✓。
            _vel_now = self._vel()
            # ① **第 2/3 行**（融合 / 内砖 / 边归属 / 允许倒退 / 继承距离 ✓）⇒ **整行**收 ✓
            for _lay in (getattr(self, "_row2", None), getattr(self, "_row3", None)):
                if _lay is None:
                    continue
                for _i in range(_lay.count()):
                    _w = _lay.itemAt(_i).widget()
                    if _w is not None:
                        _w.setVisible(not _motion)
            # ①' ⭐ **两个分组标题**（"分离期" / "融合期"）也**一起收** ✓（用户 2026-10-04 ✓ 原话：
            #    "在运动分离模式下，配置栏上面的 分离期、融合期 两行移除" ✓）—— ⚠ 原来只收了**行内
            #    控件**、**标题没跟着收** ✗ ⇒ 留下一排空标题（看着像"两行还在" ✓ 见上面那段 ✓）。
            for _hd in (getattr(self, "_row2_head", None), getattr(self, "_row3_head", None)):
                if _hd is not None:
                    _hd.setVisible(not _motion)
            # ② **第 4 行 + 它的分组标题**（运动分离专用 ✓）⇒ 新模式才放出来 ✓
            #   ⚠⚠ **速度跟踪档不一样** ✗✗（用户 2026-10-07 ✓ 原话："**清除通用参数外所有的配置
            #     参数，放上新增要用的配置参数**" ✓）：它**不用**老底盘那批参数（配对 / 打分 /
            #     换人 / 平滑 / 自检 / 粘连 ✓ 新底盘 `VelocityTracker` 一条都不吃 ✗）
            #     ⇒ 第 4/5 行**一并收掉** ✓（只留通用三项 ＋ 它自己那三个新参量 ✓）。
            _mo_trk = bool(_motion and not _vel_now)
            _h = getattr(self, "_rowM_head", None)
            if _h is not None:
                _h.setVisible(_mo_trk)
            _layM = getattr(self, "_rowM", None)
            if _layM is not None:
                for _i in range(_layM.count()):
                    _w = _layM.itemAt(_i).widget()
                    if _w is not None:
                        _w.setVisible(_mo_trk)
            # ②' ⭐ **第 5 行**（平滑 / 自检 / 粘连 ✓ 同一个约定 ✓）
            _layQ = getattr(self, "_rowQ", None)
            if _layQ is not None:
                for _i in range(_layQ.count()):
                    _w = _layQ.itemAt(_i).widget()
                    if _w is not None:
                        _w.setVisible(_mo_trk)
            # ②'' ⭐⭐ **速度跟踪那一行**（用户 2026-10-07 ✓）—— 只有它这一档放出来 ✓
            _hV = getattr(self, "_rowV_head", None)
            if _hV is not None:
                _hV.setVisible(_vel_now)
            _layV = getattr(self, "_rowV", None)
            if _layV is not None:
                for _i in range(_layV.count()):
                    _w = _layV.itemAt(_i).widget()
                    if _w is not None:
                        _w.setVisible(_vel_now)
            # ③ ⚠ **第 1 行是混着的**（"模式" / "鼠标跟随效率倍率" 两边都要 ✓）⇒ **整行藏不得** ✗
            #    ⇒ 只把"新模式**用不上**"的那几项**逐项**收掉（连它**紧挨着的那个标签** ✓ ——
            #    布局是"标签, 控件, 标签, 控件…"⇒ 前一个 item 就是它的标签 ✓）。
            _hide_on_motion = ("sp_ntol", "sp_board", "sp_ov", "sp_ioub", "chk_kf")
            _lay1 = getattr(self, "_row1", None)
            if _lay1 is not None:
                for _i in range(_lay1.count()):
                    _w = _lay1.itemAt(_i).widget()
                    if _w is None:
                        continue
                    if not any(_w is getattr(self, _n, None) for _n in _hide_on_motion):
                        continue
                    _w.setVisible(not _motion)
                    _lb = _lay1.itemAt(_i - 1).widget() if _i > 0 else None
                    if isinstance(_lb, QLabel):
                        _lb.setVisible(not _motion)
            # ⭐ **图例文字跟着模式换**（速度跟踪 ⇒ 只那 7 条 ✓ 见 `_sync_legend` ✓）
            self._sync_legend()
            if self._vel():
                _tip = ("已切到「**速度跟踪**」⇒ **画面做减法**（只留那 7 条图例 ✓）、"
                        "融合系两行已收 ✓ 点「应用」重新演算 ✓")
            elif _motion:
                _tip = ("已切到「**运动分离（新）**」⇒ 融合系两行已收、**第 4 行专用参数已放出** ✓ "
                        "（⚠ 本模式**不看检出框、不用砖、不做融合/分离** ✓）点「应用」重新演算 ✓")
            else:
                _tip = "已切回「经典「⇒ 参数行都在、运动分离那行已收 ✓ 点「应用「重新演算 ✓"
            self.statusBar().showMessage(_tip, 12000)
            # ⭐ info_col 显隐跟着模式切（chk_panel 勾上 + motion/velocity 才显示 ✓）
            if getattr(self, "info_col", None) is not None:
                self.info_col.setVisible(
                    bool(_motion and getattr(self, "chk_panel", None)
                         and self.chk_panel.isChecked()))
            # ⭐⭐⭐ **「用检测器」在这一档不显示**（用户 2026-10-07 ✓ 原话：「**用检测器 这个没有
            #   必要 去掉**」✓）—— ⚠⚠ **只是不显示** ✗：它照旧**勾着**（`apply_path_ms` 会自动勾 ✓
            #   见那里 ✓）⇒ 检出框那一套运转**一个字不变** ✓（这一档的观测**就是**检出框 ✓）。
            self._apply_dets_visible()
            # ⚠⚠⚠ **日志区照旧留着** ✗✗（用户 2026-10-07 ✓ **更正**："**日志区需要保留，我说的是
            #   清除要打印的东西，不是把日志功能删了**" ✓✓）—— ⇒ 这里**不再 `setVisible(False)`** ✗
            #   （上一版我把它藏了 ⇒ 那是把功能删了 ✗）。
            #   ⚠ "**清除要打印的东西**"落在**后端**：这一档 `log_text` 是空的 ✓、也没有
            #     `merge_log / edge_log / pick_log / split_log` ✓（那条链这一档根本没有 ✓）
            #     ⇒ `_log_sync` 喂进去的是**空的** ✓ ⇒ 日志区**空着** ✓ 但功能在 ✓。
            #   ⚠ 切过来时**清一次残留** ✓（上一档刷进去的那些 ✓ 一次就够 ✓ 不是每帧 ✓）。
            if _vel_now and getattr(self, "log", None) is not None:
                self.log.clear()
                self._log_i = 0
                self._log_n = 0

        def on_toggle_kf(self):
            """「**显示 KF 预测**」勾选变化（用户 2026-10-03 ✓ 原话："当『KF预测』开启时，将窗口
            分为**两列视图**" ✓）：立刻显示/隐藏**右列**（`col2` ✓）并落盘配置 ✓。

            ⚠ **不自动重算** ✗ —— 右列要**重跑一整遍**才有数据（与左列同耗时 ✓）⇒ 自动重算会把
              界面卡住 ✓；⇒ 只切显示 + 提示用户点"**应用**"重算 ✓（与其余参数同一个约定 ✓）。
            """
            self._cfg_save()
            if getattr(self, "col2", None) is not None:
                self.col2.setVisible(bool(self._show_kf()))
            try:
                self._status(getattr(self, "i", 0))
            except Exception:
                pass

        def _show_kf(self):
            """「**显示 KF 预测**」开关（用户 2026-10-03 ✓ 原话：「1. 移除」位置估计"参数……
            新增"显示 KF 预测"的开关" ✓）—— ⚠ 它**只决定"跑不跑 KF + 画不画"** ✓（跑也只是
            **只记录** ✗ 不改位置 ✓）；没建控件时 = 关 ✓（零开销零污染 ✓）。"""
            _c = getattr(self, "chk_kf", None)
            # ⚠ **运动分离 / 速度跟踪两档一律为关**（那个勾选框在那两档下**已被收起** ✓ 见
            #   `_on_mode_changed` ③）—— 右列是"经典那一档"的东西 ✓ 新模式没有"KF 位置驱动"
            #   这回事 ✓ 免得白跑一遍 ✗。⚠ 口径 = `self._mv()`（两档同一条底盘 ✓ 见 `_mv` ✓）。
            if self._mv():
                return False
            return bool(_c is not None and _c.isChecked())

        def _iou_brick(self):
            """当前界面上的「**缩成砖判定 · 最小 IoU**」（用户 2026-10-03 ✓ 给「框缩回砖大小/位置」
            补的**尺度无关**尺 ✓ 与「四条边都在噪声容差内」并列 OR ✓；没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_ioub", None).value()
                         if getattr(self, "sp_ioub", None) is not None
                         else _IOU_BRICK_DEFAULT)

        def load(self, path):
            self.stop()
            self._cur = path
            if self.worker:
                self.worker.close()
                self.worker = None
            # ⭐⭐⭐⭐⭐ **读新素材之前，先把旧素材整份放掉** ✗✗（用户 2026-10-07 ✓ 原话："**测谎演示
            #   打开较长的视频后，再打开就会非常卡，无法正常使用了**" ✓✓）
            #   ⚠⚠ **病根**（**实测** ✓）：`load_video` 把**每一帧的两份都留在内存** ✗ ——
            #     实测那条 750×500 的片子：**原图 5.25MB ＋ 处理帧 1.12MB = 6.37MB/帧** ✗
            #     （⚠ "原图"其实是 **1.75M 像素**那一档 ✓ 不是 750×500 ✗）；折算到 1080p 档
            #     **7.55MB/帧** ⇒ **3 分钟内容（1800 帧）≈ 13.6GB** ✗✗。
            #   ⇒ 而 `self.frames` 原来是**等 `load_source` 返回之后**才被替换 ✗ ⇒ **读新的时候旧的
            #     还活着** ✓ ⇒ 峰值 **2×** ✗ ⇒ 一开长片就开始换页 ⇒ **非常卡、没法用** ✓✓
            #     （他说的正是"**再打开**"那一下 ✓ —— 第一次打开还是装得下的 ✓）。
            #   ⇒ 现在：**先把旧的丢掉再读**（frames / ts / dets / results / runner 全放 ✓ ＋ `gc` ✓）
            #     ⇒ 峰值回到 1× ✓。
            #   ⚠⚠ **顺序是承重的** ✗（所以下面那条钉子钉的是**先后** ✓）：把这几行挪到 `load_source`
            #     **后面** ⇒ 又变回 2× ✗（照旧会卡 ✓，而且一点报错都没有 ✗）。
            self.frames = []
            self.ts = []
            self.dets = None
            self.results = []
            self.results_kf = None
            self.runner = None
            self.runner_kf = None
            gc.collect()
            #   ⭐⭐⭐ **读入期间要让界面活着** ✗✗（用户 2026-10-07 ✓ "**优化直到能顺畅跑**" ✓）：
            #     ⚠ 长片读入实测 **13.1s**（那条片子 991 帧 ✓）✗ —— 全程不还手就是"点开就冻住、
            #       连'读入中'都刷不出来" ✓ ⇒ 每 ~20 帧回调一次：刷 label ＋ `processEvents()`
            #       让窗口喘气 ✓；⚠ 期间**禁用「选择视频…」** ✗（不然用户再点一次 ⇒ 重入 ⇒
            #       两趟读入打架 ✗）。
            self._loading = True
            try:
                self.btn_open.setEnabled(False)
            except Exception:                       # noqa: BLE001 —— 工具栏没建好也照样读 ✓
                pass

            def _prog(_kept, _seen):
                self.lbl.setText("读入素材… %d 帧（已读 %d）" % (_kept, _seen))
                _app = QApplication.instance()
                if _app is not None:
                    _app.processEvents()

            try:
                # ⭐ 统一入口：**`.gif` ⇒ GIF** ✓ / 其余 ⇒ 视频 ✓（两条路同形返回 ✓）
                self.frames, self.ts, self.proc, self.scale, self.fps = \
                    load_source(path, on_progress=_prog)
            finally:
                self._loading = False
                try:
                    self.btn_open.setEnabled(True)
                except Exception:                   # noqa: BLE001
                    pass
            self.dur = self.ts[-1] if self.ts else 0.0   # 总时长 = 最后一帧的时间戳 ✓
            self.is_gif = Path(str(path)).suffix.lower() == ".gif"   # 措辞用 ✓
            if not self.frames:
                self.lbl.setText("读不到%s：%s"
                                 % ("GIF" if self.is_gif else "视频", path))
                return
            self.setWindowTitle("测谎演示 · %s（%d 帧 @%.1ffps%s）　[代码 %s]"
                               % (Path(str(path)).name, len(self.frames), self.fps,
                                  " GIF" if self.is_gif else "", code_stamp()))
            self.dets = None
            self.i = 0
            self.t = 0.0
            self.runner = None
            self.sld.setRange(0, max(0, len(self.frames) - 1))
            self.sld.setValue(0)
            self.upd_time()
            self.btn_play.setEnabled(True)
            self.btn_apply.setEnabled(True)      # ⭐ 有素材了 ⇒"应用"才可用 ✓
            self.btn_play.setText("播放")
            #   ⚠⚠ **原图现在是 JPEG 字节** ✗（见 `_BIG_JPEG_Q` ✓）⇒ 要用就先解一次 ✓
            #     （`_big_of` 自带一格缓存 ✓ 同一帧只解一遍 ✓）。
            _b0 = decode_big(self.frames[0][0])
            if args.record:
                rp = Path(args.record)
                rp.parent.mkdir(parents=True, exist_ok=True)
                # 录像的 fps 用**实际平均帧率**（GIF 的时间戳不等间隔 ⇒ 写 `fps` 会偏 ✓）
                rec_fps = (max(0.5, (len(self.frames) - 1) / self.dur)
                           if self.dur > 1e-3 else self.fps)
                self.rec = cv2.VideoWriter(
                    str(rp), cv2.VideoWriter_fourcc(*"mp4v"), rec_fps,
                    (_b0.shape[1], _b0.shape[0]))
            self.show_frame(_b0)
            self.start_warmup()                # 一律先演算（无检测也是一种演算 ✓）
            # ⭐ **载入成功就记下这个文件**（用户 2026-10-01 ✓ 原话："让窗口记住上次打开的文件" ✓）：
            #   不等到 `closeEvent` ✗（关窗才落盘 ⇒ 崩溃/强杀就丢了 ✓）。
            self._cfg_save()

        def start_warmup(self, reuse=False):
            """`reuse=True` ⇒ **复用已算好的检测框**（只重跑追踪 ✓ 「应用」走这条 ✓）。

            ⚠ 只有"检测框与本次参数无关"时才可复用 ✓ —— "轨迹预测窗口时间"只影响
            切向拟合 ⇒ 可复用 ✓；而"用检测器"开关会**改变有没有框** ⇒ 必须重检测 ✗。
            """
            self._reuse_dets = bool(reuse)
            if not self._reuse_dets:
                self.worker = DetsWorker(args.weights or None, conf=args.conf)
                self.dets = []
            self.results = []
            # ⭐⭐⭐ **右列（KF 位置驱动）那一份也重来**（用户 2026-10-03 ✓ 双列对照 ✓）：
            #   勾了"显示 KF 预测"⇒ 右列显示 + 这一路**真用 KF 位置**跑（`kf_pos=True` ✓）；
            #   没勾 ⇒ 右列整体隐藏、这一路不跑（零开销 ✓ 与改动前逐像素一致 ✓）。
            self.results_kf = [] if self._show_kf() else None
            if getattr(self, "col2", None) is not None:
                self.col2.setVisible(bool(self._show_kf()))
            # ⭐ 日志区跟着新素材 / 新演算**重来**（用户 2026-10-02 ✓）—— `_log_i` 归零 ⇒
            #   `_log_sync` 会从头重建 ✓（换素材/换参数后不会留着上一段的融合日志 ✗）。
            self._log_i = 0
            self._log_n = 0
            if getattr(self, "log", None) is not None:
                self.log.clear()
            self.runner = Runner(dets=None, path_ms=self._path_ms(),
                                 # ⭐ **融合框判定 IoU**（用户 2026-10-03 ✓ 进严 ✓）
                                 merge_iou=self._merge_iou(),
                                 # ⭐ **融合框判定 IoU · 退出**（迟滞下半段 ✓ 守松 ✓）
                                 merge_iou_out=self._merge_iou_out(),
                                 # ⭐⭐⭐ **分离判定阈值**（用户 2026-10-02 ✓ 与"融合框判定
                                 #   阈值"同一个接线法 ✓）⇒ 点"应用"就按新值重算 ✓。
                                 sep_ratio=self._sep_ratio(),
                                 # ⭐ **融合框继承距离限制**（用户 2026-10-02 ✓ 融合框候选第一条 ✓）
                                 inherit_dist=self._inherit_dist(),
                                 # ⭐ **融合框挑选允许倒退距离(绿圆半径比例)**（用户 2026-10-02 ✓）
                                 allow_back_ratio=self._allow_back_ratio(),
                                 # ⭐ **砖最多拥有融合框边数量**（用户 2026-10-02 ✓ 1~4 ✓）
                                 edge_max=self._edge_max(),
                                 # ⭐ **真目标预测最大速度倍率**（用户 2026-10-03 ✓）
                                 sep_vel_max_ratio=self._sep_vel_max(),
                                 noise_tol=self._noise_tol(),
                                 board_s=self._board_s(),
                                 board_overlap=self._board_overlap(),
                                 # ⭐ **鼠标跟随效率倍率**（用户 2026-10-03 ✓ 只进控制器 ✓
                                 #   不影响位置计算 ✓）⇒ 点"应用"就按新值重算 ✓。
                                 follow_gain=self._follow_gain(),
                                 # ⭐ **缩成砖判定·最小 IoU**（用户 2026-10-03 ✓ 进追踪器 ✓）
                                 iou_brick=self._iou_brick(),
                                 # ⭐ **显示 KF 预测**（用户 2026-10-03 ✓ 勾上 ⇒ 卡尔曼并行跑、
                                 #   只记录 ✓ 位置仍走经典 ⇒ **不污染** ✓）
                                 show_kf=self._show_kf())
            # ⭐⭐⭐ **右列的 Runner = 同一套参数 + `kf_pos=True`**（用户 2026-10-03 ✓ 原话："左边
            #   跑经典右边跑 **KF**，这样对照才有意义" ✓）：**独立一份 `LieTracker`** ⇒ 状态与左列
            #   完全隔离 ✓；`kf_pos=True` ⇒ 它**真的用 KF 位置**跑整条链（不是叠加影子 ✓）。
            self.runner_kf = (Runner(dets=None, path_ms=self._path_ms(),
                                     # ⭐ **融合框判定 IoU**（用户 2026-10-03 ✓ 进严 ✓）
                                     merge_iou=self._merge_iou(),
                                     # ⭐ **融合框判定 IoU · 退出**（迟滞下半段 ✓ 守松 ✓）
                                     merge_iou_out=self._merge_iou_out(),
                                     # ⭐⭐⭐ **参数分档**（用户 2026-10-03 ✓ 第 2/3 行）
                                    ring_cov_sep=self._ring_cov_sep(),
                                    brick_iou_sep=self._brick_iou_sep(),
                                    ring_cov_fuse=self._ring_cov_fuse(),
                                    brick_iou_fuse=self._brick_iou_fuse(),
                                     sep_ratio=self._sep_ratio(),
                                     inherit_dist=self._inherit_dist(),
                                     allow_back_ratio=self._allow_back_ratio(),
                                     edge_max=self._edge_max(),
                                     sep_vel_max_ratio=self._sep_vel_max(),
                                     noise_tol=self._noise_tol(),
                                     board_s=self._board_s(),
                                     board_overlap=self._board_overlap(),
                                     follow_gain=self._follow_gain(),
                                     iou_brick=self._iou_brick(),
                                     kf_pos=True)
                             if self._show_kf() else None)
            # ⭐⭐⭐ **模式 = 运动分离（新）** ⇒ 换成 `MotionRunner`（用户 2026-10-03 ✓ 见模式下拉 ✓）
            #   ⚠ 它与 `Runner` **同签名**（`step` 回同样的 6 元组 ✓）⇒ `warm_tick` / 显示 / 拖动
            #     **一行都不用改** ✓（白线 = 报告位置历史、绿圈 = 目标、红点 = 光标 全是现成的 ✓）。
            #   ⚠ 新模式**不做右列（KF）** ⇒ 关掉 ✓（那是老算法那一档的东西 ✓）。
            # ⭐⭐⭐ **两档共用同一条底盘**（`motion` ＋ `velocity` ✓ 用户 2026-10-07 ✓ 见 `_mv` ✓）——
            #   ⚠ 只多一个 `mode=` 参数（它决定 `motion["mode"]` 那个键 ✓ 判据一个字不碰 ✓）。
            if self._mv():
                _pc = self.frames[0][1].shape
                # ⭐⭐⭐ **速度跟踪档换成 `VelocityRunner`**（用户 2026-10-07 ✓「重头开始」✓）——
                #   差别**只在底盘**：`VelocityTracker`（开局锁一次 ＋ 丢框**纯滑行** ✓
                #   **一条"挑谁是真目标"的逻辑都没有** ✗ 见那里 ✓）⇒ 其余全沿用 ✓。
                #   ⚠ 老底盘那批参数（第 4/5 行 ✓ `_motion_kw()`）在本档**已经被收掉** ✗
                #     ⇒ **一个都不传** ✓（新底盘也不吃 ✓ 见 `VelocityTracker.__init__` ✓）。
                _rcls = VelocityRunner if self._vel() else MotionRunner
                # ⚠⚠⚠ **ROI 必须转过去** ✗✗（**实测翻车** ✓）：我上一版写的是"速度跟踪档
                #   **一个参数都不传**" ✗ ⇒ 把 `roi` 也一起丢了 ⇒ 那一档实际是**整幅**在跑 ✗
                #   （用户 2026-10-07 报"**框选区域没有任何作用**" ＋ "**初始绿圆挑错了**" ✓ 同一条根因 ✓
                #   —— 实测量到：**无 ROI 首次锁在帧 72** ✗ ／ **带 ROI 在帧 58** ✓，
                #   而用户截图是"帧 73 时圆停在 (469,132) 那格普通砖上" ⇒ 完全对上 ✓）。
                #   ⇒ 这一档只传 `roi`（老底盘那批参数仍**一个都不传** ✓ 新底盘不吃 ✓）。
                _vel_kw = {}
                if self._vel():
                    _vel_kw["roi"] = getattr(self, "roi", None)
                self.runner = _rcls(
                    # ⚠⚠ **`gain` 千万别填 `self._follow_gain()`** ✗✗（那是"跟随效率倍率"，
                    #    是个**标量** ✓ 该给下面那个 `follow_gain` ✓）—— `gain` 要的是
                    #    **每像素的二元组**（`LieMouseController.__init__` 里 `tuple(gain)` ✓）
                    #    ⇒ 填标量会 `TypeError: 'float' object is not iterable` ✗ ⇒ **点"应用"
                    #    当场闪退** ✓（PyQt5 未捕获异常直接 `abort()` ✓ 用户实测就是这个 ✓）。
                    #    经典 `Runner` 那边**压根没传 `gain`** ✓（用 `load_gain()` 默认 ✓）⇒ 这里
                    #    照抄"不传"才是对的 ✓。
                    dets=None, gain=None,
                    assume=(_pc[1] / 2.0, _pc[0] / 2.0),
                    follow_gain=self._follow_gain(),
                    # ⭐⭐⭐ **第 4 行那 5 个参数**（用户 2026-10-03 ✓ 原话："运动分离 **没有任何
                    #   参数要配吗？**" ✓）—— `_motion_kw()` 回 `{q, gate_k, smooth, vel_decay,
                    #   learn_pick}` ✓ 与 `MotionRunner.__init__` 的形参**正好对上** ✓
                    #   （`q`/`gate_k` 是显式形参 ✓ 其余进 `**tracker_kw` ✓）。
                    # ⭐⭐ **模式标记透传**（用户 2026-10-07 ✓「速度跟踪」那一轮 ✓）—— 它**只决定
                    #   `motion["mode"]` 那个键** ✗（判据一个字不碰 ✓ 见 `MotionRunner` ✓）。
                    mode=self._mode(),
                    #   ⚠ 速度跟踪档**不传** `_motion_kw()`（那批参数已被收掉 ✓ 新底盘不吃 ✓）
                    #     —— 但 `**_vel_kw` 里那个 **`roi` 要传** ✓（见上面那段 ✓）。
                    **({} if self._vel() else self._motion_kw()),
                    **_vel_kw)
                self.runner_kf = None
                self.results_kf = None
                if getattr(self, "col2", None) is not None:
                    self.col2.setVisible(False)
            self.warm_n = 0
            self.warm_total = len(self.frames)
            self.warm.start(10)

        def _safe(self, _fn):
            """把"会抛的定时器槽"包一层：**先落盘、再吞掉** ✓（用户 2026-10-07 ✓ 见 `log_exception` ✓）。

            ⚠⚠ **不这么做会怎样** ✗：Python 异常一旦冒到 Qt 事件循环 ⇒ PyQt `abort` ⇒ **整窗消失** ✓
              （用户跑 `pythonw` ⇒ 连 traceback 都没有 ✓ = 他说的"**闪退**" ✓）。
            ⚠ 吞掉不等于隐瞒 ✓：traceback 写进 `logs/lie_demo_error.log` ✓ ＋ 状态栏说一句 ✓。
            """
            try:
                return _fn()
            except Exception as _e:               # noqa: BLE001 —— 这里就是要兜住一切 ✓
                log_exception("定时器槽 %s" % getattr(_fn, "__name__", "?"), _e)
                try:
                    self.statusBar().showMessage(
                        "⚠ 内部异常（已记入 logs/lie_demo_error.log）：%s" % _e, 8000)
                except Exception:                 # noqa: BLE001
                    pass
                return None

        def warm_tick(self):
            """⭐ **边检测边演算**（整段一次算完 ✓ 用户要求 ②）：检测 + 追踪 + 控制
            都在这里跑完 ⇒ 播放和拖动只是查表 ✓ 不重算 ✓。

            ⚠⚠⚠ **2026-10-07 改成"每拍 1 帧"** ✗✗（用户原话："**优化直到能顺畅跑
              D:\\Media\\record\\mxd\\10月7日.mp4**" ✓✓）——
              · 原来写的是 `for _ in range(4)` ✗（注释还写着"界面不卡" ✗ —— 恰恰相反 ✓）：
                每帧要 **检测 ~26.5ms（GPU ✓）＋ 追踪 ~9ms** ⇒ 一拍就是 **~140ms** ✗
                ⇒ 主线程 140ms 不还手 ⇒ 界面**一顿一顿** ✓（长片要演算 20s+ ⇒ 像卡死 ✓）；
              · 现在**一拍只做 1 帧** ✓（≈35ms ✓）⇒ 两帧之间事件循环能跑 ✓ 界面**能喘气** ✓
                （总时长不变 ✓ —— 演算量是定的 ✓）。
            """
            for _ in range(1):                 # 每拍 **1** 帧（⚠ 别再调大 ✗ 见上 ✓）
                if self.warm_n >= self.warm_total:
                    break
                i = self.warm_n
                d = None
                # ⭐ 复用模式（"应用"✓）：检测框**已经算好了** ⇒ 不再问检测器 ✓
                if not getattr(self, "_reuse_dets", False) and self.chk_dets.isChecked():
                    d = self.worker.detect(self.frames[i][1])
                    self.dets.append(d)
                self.runner.dets = self.dets if self.chk_dets.isChecked() else None
                # ⭐ 用**真实时间戳**（GIF 自带每帧延时 ✓ 可以不等间隔 ✓）
                o, pos, rad, hit, boxes, motion = self.runner.step(
                    self.frames[i][1], i, self.ts[i])
                self.results.append({"pos": pos, "r": rad, "hit": hit,
                                     # ⭐⭐⭐ **目标实际半径**（= `out["rad"]` = `LieTracker._rad` ✓
                                     #   与**后端 `_ring_cov`**、**画面上那个绿圈**用的是**同一把尺** ✓）。
                                     #   ⚠⚠ 别拿上面那个 `r` 去当"圆外接矩形"的半径 ✗✗ —— 那是
                                     #     `Runner` 按**面积等效**算的（`max(18, √area/1.7725)` ✗）：
                                     #     实测 `10月1日.mp4` 帧 12 两者 **18.0 vs 60.8**（差 3 倍 ✗）
                                     #     ⇒ 圆的方块缩成 36×36 ⇒ 塞进 169×206 的框里 ⇒ 点选标签
                                     #     算出 `圆矩IoU 0.04 ／ 圆矩∩框 1.00` ✗✗（用户 2026-10-03
                                     #     看出来的就是这个 ✓）。
                                     # ⭐⭐⭐⭐⭐ **面板里的"圆半径"也必须是那一个** ✗✗（用户
                                    #   2026-10-04 ✓ 原话："我们一定要所见即所得，不要表面搞
                                    #   一套背后搞一套" ✓✓）—— ⚠ 这个 `tgt_rad` 会被
                                    #   `pick_extra` 拿去算"圆矩 IoU / 圆矩∩框"（见下面 ✓）；
                                    #   运动模式画的那个圆**用的是后端 `tgt_rad`** ✓ ⇒ 这里
                                    #   **必须同源** ✓ 否则面板算的和画面画的是两个圆 ✗✗。
                                    #   （经典模式没有 `tgt_rad` 这个键 ⇒ 回落 `rad` ✓ 不变 ✓）
                                    "tgt_rad": (o.get("tgt_rad") or o.get("rad")),
                                     # ⭐⭐⭐⭐⭐ **夹取这一拍给速度的那一份修正**（用户 2026-10-04
                                     #   ✓ 原话："位置被硬夹说明你要么快了要么慢了，夹取也要算进
                                     #   速度平滑…夹取是较准确的测量值" ✓✓）—— 见 `lie_motion` ✓；
                                     #   `None` = 这一拍没夹（或增益 0 ✓）⇒ 面板不写那行 ✓ 不误导 ✓。
                                     "clamp_vel": o.get("clamp_vel"),
                                     "state": o["state"],
                                     "cursor": tuple(self.runner.cursor),
                                     "boxes": boxes or [], "motion": motion,
                                     # ⚠ 变量名是 `o`（追踪器输出 ✓）不是 `out` ✗
                                     #   （我上一版写成 `out` ⇒ 一开窗 NameError ⇒ 闪退 ✗）
                                     "white": o.get("white"),
                                     # ⭐⭐ **融合框生成**那一拍的事件（其余帧为 `None` ✓ 用户
                                     #   2026-10-02 ✓）⇒ 底部深色日志区显示它 ✓（`_log_sync` ✓）。
                                     "merge_log": o.get("merge_log"),
                                     # ⭐⭐ **融合框消失（分离）**那一拍的事件（用户 2026-10-02 ✓
                                     #   原话："把融合框消失（分离）的日志**也依据融合的格式**打印
                                     #   出来" ✓）⇒ 与融合日志同一个日志区、**同格式** ✓。
                                     "split_log": o.get("split_log"),
                                     # ⭐ **融合期"每拍"的边归属**（用户 2026-10-02 ✓ 原话：
                                     #   "融合时期每拍打印一下边的归属（砖或真目标）" ✓）⇒ 同一
                                     #   个日志区显示 ✓（与融合/分离日志并列 ✓ 三条互补 ✓）。
                                     "edge_log": o.get("edge_log"),
                                    # ⭐⭐⭐ **非融合红框选择**（用户 2026-10-03 ✓ 原话："在日志
                                    #   加一下**非融合红框选择**的判定信息吧" ✓）：红框**不是**
                                    #   融合框（= "目标自己那格" ✓）那一拍给 ✓ ⇒ 同日志区显示 ✓。
                                    "pick_log": o.get("pick_log"),
                                   # ⭐⭐ **本拍砖表快照**（用户 2026-10-03 ✓ 点选时算
                                   #   "IoU 最大的砖"要用它 ✓ 见 `_bricks_snap` ✓）
                                   "bricks": o.get("bricks"),
                                    # ⭐ KF 影子（用户 2026-10-03 ✓）：勾了"显示 KF 预测"才非 None
                                    #   （KF 位置/速度/新息/与经典的差/**群体系锚点** ✓ 只读不参与判定 ✓）
                                    "kf": o.get("kf")})
                # ⭐⭐⭐ **右列（KF 位置驱动）逐帧同步跑**（用户 2026-10-03 ✓ 双列对照 ✓）：
                #   同一批检测框（`self.dets` ✓ 检测只跑一次 ✓）、**同一时间戳** ✓ ⇒ 两列只在
                #   "位置估计"这一处不同 ✓ ⇒ 差异全部来自 KF ✓（这样对照才干净 ✓）。
                if self.runner_kf is not None and self.results_kf is not None:
                    self.runner_kf.dets = (self.dets if self.chk_dets.isChecked() else None)
                    o2, pos2, rad2, hit2, boxes2, motion2 = self.runner_kf.step(
                        self.frames[i][1], i, self.ts[i])
                    self.results_kf.append({"pos": pos2, "r": rad2, "hit": hit2,
                                            "state": o2["state"],
                                            "cursor": tuple(self.runner_kf.cursor),
                                            "boxes": boxes2 or [], "motion": motion2,
                                            "white": o2.get("white"),
                                            # ⚠ 右列是"KF 在跑"⇒ **不再叠加青色影子** ✗（用户
                                            #   原话"叠加 KF 显示意义不大" ✓）⇒ 日志/KF 诊断都不带 ✓
                                            "merge_log": None, "split_log": None,
                                            "edge_log": None, "pick_log": None,
                                            "kf": None})
                self.warm_n += 1
            acc = 100.0 * self.runner.hits / self.runner.n if self.runner.n else 0.0
            self.lbl.setText("演算中 %d / %d 帧…（检测 + 追踪 + 控制**一次算完** ⇒ "
                             "拖帧零等待 ✓）"
                             % (self.warm_n, self.warm_total))
            if self.names is None and self.chk_dets.isChecked():
                # ⚠⚠⚠ **2026-10-07：这一句挪到后台线程** ✗✗（用户原话："**优化直到能顺畅跑
                #   D:\\Media\\record\\mxd\\10月7日.mp4**" ✓✓）——
                #   它要**另起一个短命子进程**去加载模型问类别名 ✓ ⇒ **实测 3.0s** ✗（见 `_run_mxd` 那轮 ✓）
                #   ⇒ 主线程干等 3s = 界面**冻住 3 秒** ✓✓（比演算那 42ms 一拍严重两个数量级 ✗）。
                #   ⇒ 后台线程问 ✓（等 subprocess 时 GIL 是放开的 ✓）、问到再填 ✓；
                #   ⚠ `self.names` 始终是**只赋值一次** ✓（主线程绘制时读 ✓ ⇒ 不用锁 ✓）；
                #   ⚠ 没问到之前绘制**照旧**（退回显示编号 ✓ —— 老行为就是这个 ✓）。
                if getattr(self, "_names_th", None) is None:
                    self._names_th = threading.Thread(target=self._fetch_names, daemon=True)
                    self._names_th.start()
            if self.warm_n >= self.warm_total:
                self.warm.stop()
                print("演算完成：%d 帧 ｜ 命中率 %.1f%%（%d/%d）"
                      % (len(self.results), acc, self.runner.hits, self.runner.n))
                self.begin_play()

        def begin_play(self):
            if not self.results:
                return
            self.i = 0
            self.t = self.ts[0] if self.ts else 0.0
            self.btn_play.setText("暂停")
            self.playing = True
            self._play_anchor(0)         # ⭐ 重锚（墙钟起点 = 现在 ✓ 见 `play_delay_ms` ✓）
            self.retime()

        def dt_here(self, i=None):
            """第 `i` 帧 ⇒ 下一帧的**真实间隔**（秒 ✓ 末帧/异常兜底 1/fps ✓）。"""
            i = self.i if i is None else i
            if self.ts and i + 1 < len(self.ts):
                return max(1e-3, self.ts[i + 1] - self.ts[i])
            return 1.0 / max(1.0, self.fps)

        def _on_speed_changed(self, *_a):
            """速度下拉变了 ⇒ **重锚再排** ✓（⚠ 只 `retime()` 不重锚 ⇒ 调慢会一路狂追 ✗ 见上 ✓）。"""
            self._play_anchor()
            self.retime()

        def _play_anchor(self, i=None):
            """把"墙钟锚"对到当前帧 ✓（⚠ **每次开播 / 拖动落地 / 循环回头 都要重锚** ✗ ——
            不然锚还停在很久以前 ⇒ 每拍都算出"早就该到" ⇒ 一路狂追 ✗）。
            """
            self._play_i0 = int(self.i if i is None else i)
            self._play_t0 = time.monotonic()

        def retime(self, *_ignored):
            """按**当前帧到下一帧的真实间隔** × 速度重设定时器。

            ⭐ 为什么不是固定 `1/fps`（原写法 ✗）：GIF 的时间戳**不等间隔**
            （配对帧差 ~6ms、配对之间 ~0.6s ✓）⇒ 固定间隔播放会让"配对帧"各停 0.3s
            ✗ 完全不是素材的真实节奏 ✓；按 `ts` 走才对 ✓（视频那条路 `ts` 等间隔 ⇒
            行为与原来一致 ✓）。
            ⚠ 下拉的 `currentIndexChanged(int)` 会多传一个参数 ⇒ 签名要吃得下 ✗
            （不然槽里 TypeError 静默 ✗）。
            """
            v = float(self.cmb_speed.currentData() or 1.0)
            #   ⭐⭐⭐⭐⭐ **按墙钟排**（用户 2026-10-07 ✓ 见 `play_delay_ms` ✓）—— 老写法只按"这一帧
            #     的真实间隔"★**不扣**已经花掉的工作时间** ✗ ⇒ 每拍周期 = 工作量 ＋ 间隔 ✗
            #     （实测 64ms/拍 ⇒ **0.65 倍速** ✗ = 用户那句"没有以原速播放" ✓）。
            #   ⚠ `play_delay_ms` 回 `None`（裸窗口 / 没 ts ✓）⇒ 回落老算法 ✓ 不崩 ✓。
            _ms = play_delay_ms(getattr(self, "ts", None) or [],
                                getattr(self, "i", 0),
                                getattr(self, "_play_i0", 0),
                                getattr(self, "_play_t0", None) or time.monotonic(),
                                time.monotonic(), v)
            if _ms is None:
                _ms = 1000.0 * self.dt_here(self.i) / max(0.05, v)
            #   ⚠ 下界取 **0** 而不是 1 ✗：算出"已经迟到"⇒ **立刻**排下一拍（诚实追 ✓）；
            #     取 1 会每拍再多搭 1ms ✗（797 拍就是 0.8s ✗）。
            self.timer.start(max(0, int(round(_ms))))

        def toggle_play(self):
            if self.playing:
                self.stop()
            elif self.results:
                self.playing = True
                self.btn_play.setText("暂停")
                self._play_anchor()      # ⭐ 重锚（从当前帧接着走 ✓）
                self.retime()
            elif self.frames:
                self.begin_play()

        def stop(self):
            self.playing = False
            self.timer.stop()
            if self.btn_play.isEnabled():
                self.btn_play.setText("播放")

        def step_frame(self, d):
            """**←→ 切帧**（±1 帧 ✓ 任何焦点下都生效 ✓ 用户 2026-10-01 ✓）。

            实现上就是"**当成拖动**"：先停播 ✓，改 `self.i` ✓，再 `sld.setValue(i)` ⇒ 走
            `on_seek` 那条**已经在用**的路（画面 + 时间 + 状态栏一起同步 ✓）。

            ⚠ 必须**先停播**（与 `on_seek_press` 同一规矩 ✓）：播放中每 100ms 一次 `tick` 会把
              帧号推回去 ⇒ 按了像没按 ✗（用户 2026-09-30 报过"值动了画面没变"✗ 那条已修 ✓）。
            ⚠ 边界**夹住**（第 0 帧再按 ← / 末帧再按 → ⇒ 原地不动 ✓ 不绕圈 ✗ 绕圈会让人
              以为自己按错了 ✓）。
            """
            if not self.frames:
                return
            if self.playing:
                self.stop()
            n = len(self.frames)
            self.i = max(0, min(self.i + int(d), n - 1))
            self.sld.setValue(self.i)      # 走 `valueChanged` ⇒ `on_seek` 全流程 ✓

        # ---- 视图参数（蒙版/箭头 ✓ 只影响**渲染** ⇒ 改完当场重画当前帧 ✓ 不算 ✓）----
        def view(self):
            """**蒙版不透明度**（箭头长度不可调了 ✓ 用户 A2 ✓）"""
            return self.sld_mask.value() / 100.0

        def on_view_change(self, *_a):
            self.lbl_mask.setText("%d%%" % self.sld_mask.value())
            self._show_precomputed()               # 立刻按新参数重画 ✓

        # ---- 进度条（拖动定位 ✓）----
        def on_seek_press(self):
            self._was_playing = self.playing
            self.stop()

        # ---- ⭐⭐⭐ 点选检出框（用户 2026-10-03 ✓）----
        def _kf_trail(self, i):
            """**KF 轨迹（青线）= 群体系锚定**（用户 2026-10-03 ✓ 原话："**2. KF 青色线没有像白线
            一样利用群体速度『刻在背景板上』**" ✓）—— 与**白线完全同一套算法** ✓（见 `_path_viz` ✓）：

            · 每个历史点 = 它**自己那拍**的 `kf.pos_grp`（**已扣掉**它那拍的累计群体平移 ✓ 后端算好 ✓）
              ＋ **当拍**的累计群体平移 `kf.cum2` ✓ ⇒ 整条线**跟着假目标群一起动** ✓
              （不再跟着相机走 ✗ —— 这正是用户点名的那条 ✓）；
            · ⚠ `_path_cumT` 是**半域** ⇒ 后端已经 ×2 放进 `cum2` ✓ 这里不再换算 ✗（口径一处 ✓）；
            · ⚠ 纯显示层 ✗：`results` 里本来就有每帧的 `kf` ⇒ 这里只**连点** ✓ 不重算、不判定 ✓；
              勾选关掉 ⇒ 恒为空列表 ⇒ 画面什么都不多 ✓（零污染 ✓）。
            """
            _n = max(0, int(i))
            _now = None
            for _k in range(_n, -1, -1):          # 从当拍往回找**最近**的 `cum2`（那拍可能没 kf ✓）
                _kk = self.results[_k].get("kf") or {}
                if _kk.get("cum2") is not None:
                    _now = (float(_kk["cum2"][0]), float(_kk["cum2"][1]))
                    break
            if _now is None:
                return []
            _out = []
            for _k in range(0, _n + 1):
                _g = (self.results[_k].get("kf") or {}).get("pos_grp")
                if _g is not None:
                    _out.append((float(_g[0]) + _now[0], float(_g[1]) + _now[1]))
            return _out

        def _pick_box(self):
            """**当前帧**点选的那格检出框（没选 / 选的是别的帧 ⇒ `None` ✓ 纯显示用 ✓）。

            ⭐⭐⭐ 2026-10-03 ✓ 第一次：用户："**点选检出框的时候，希望他能显示 IoU 最大的砖的
            IoU**" ✓ ⇒ 这里算好"**与哪块砖 IoU 最大、值是多少**" ✓ 附在元组**第 7 位** ✓。
            ⭐⭐⭐ 2026-10-03 ✓ 第二次（原话："点选检出框时，与圆外接矩形的 iou、圆外接矩形与其
            相交的比例**也显示下**；另外，以上参数**及旧的 iou 显示都放在第二行**" ✓）：再补
            **圆那一侧的两个量** ✓ 附在**第 8 位** ✓（`pick_label` 把 `pick[6:]` 一起摆到
            **第二行** ✓）：
              · `圆矩IoU` = `IoU(检出框, 圆外接矩形)` ✓（= **旧口径**那个量 ✓ 与 `_ring_cov` 对照 ✓）；
              · `圆矩∩框` = `圆外接矩形与检出框相交的比例` ✓（分母 = **圆矩形面积** ✓ **单向** ✓）
                ⇒ 与后端 `_ring_cov` / `geom.cover_ratio` **同一把尺** ✓ 不自己写一份 ✗。
            ⚠ 圆心 / 半径取**这一帧演算出来的**（`results[i]["pos"] / ["r"]` ✓ 同一份数据 ✓）；
              拿不到（预测态没圆心 / 半径还没学到 ✓）⇒ **那一段留空** ⇒ 界面自动不显示 ✓ 不猜 ✗。
            ⚠ 基准 = **那一帧的砖表快照**（`results[i]["bricks"]` ✓ 见后端 `_bricks_snap` ✓）——
              砖表每拍都随群体平移 ✗ ⇒ 用"演算结束时的表"会差一个累计平移 ✗。
            ⚠ 三段量的**具体计算一律交给纯函数 `pick_extra`** ✓（口径一处 ✓ 好测 ✓ 自检直接钉
              它 ✗ 不在这里再写一遍 ✓）—— 本函数只负责"取哪一帧的数据" ✓。
            """
            # ⭐⭐⭐ **速度跟踪：文字照旧不给、但「选中」要看得见** ✗✗（用户 2026-10-07 ✓ **两条一起**：
            #   ①"点选检出框后出现的信息面板"要**清空** ✓；②"**选中的检出框需要高亮**，你设计展示
            #   形式" ✓）—— ⇒ 这里**照旧把那一格算出来回出去** ✓（高亮要用它 ✓），
            #   **文字那一截**由 `draw` 按 `_vel` 跳过 ✓（面板底板 ＋ 文字一起不画 ✓ 见那里 ✓）
            #   ⇒ 高亮有 ✓ 面板文字没有 ✓ 纪律不破 ✓。
            _p = getattr(self, "_pick", None)
            if not _p or _p.get("frame") != getattr(self, "i", -1):
                return None
            _b = _p.get("box")
            _br = []
            if self.results and 0 <= self.i < len(self.results):
                _br = self.results[self.i].get("bricks") or []
            # ⚠ 三段量**一律交给纯函数 `pick_extra`** 算 ✓（口径一处 ✓ 好测 ✓ 自检直接钉它 ✓）：
            #   第 7 位 = 砖那一段、第 8 位 = **圆那一侧那一段**（本次新增 ✓ 见 `pick_extra` ✓）。
            _row = (self.results[self.i]
                    if (self.results and 0 <= self.i < len(self.results)) else {})
            # ⭐⭐⭐ **运动分离模式：点选信息整段换掉**（用户 2026-10-04 ✓ 原话："在运动分离模式下，
            #   点选检出框的信息显示需要优化下，**旧逻辑的信息就不要了**；**你可以把匹配分写出来**" ✓✓）
            #   —— ⚠ 旧那三段（`IoU最大砖` / `圆矩IoU` / `圆矩∩框`）全是**砖 / 圆矩**那套说法 ✓
            #     而运动分离模式**压根没有砖、也不做融合** ✗ ⇒ 留着全是噪音 ⇒ **一个字不显示** ✓
            #     ⇒ 换成这套**这模式真有的量** ✓：
            #       · **匹配分**（用户点名要的 ✓ = "这一格的观测有多可信" ✓ 见 `lie_motion` ✓）；
            #       · `#id`（它配到哪条轨迹上 ✓）；
            #       · 该轨迹的**偏离 `dev`** / **分数** / 有观测还是推的 / 粘住 · 冷却 ✓。
            _mo = _row.get("motion") or {}
            if self._mode() == "motion":
                _bv = None
                for _it in (_mo.get("box_v") or []):
                    if (abs(float(_it[0]) - float(_b[1])) < 1.5
                            and abs(float(_it[1]) - float(_b[2])) < 1.5):
                        _bv = _it
                        break
                if _bv is None:
                    # ⭐⭐⭐⭐⭐ **先判"是不是 YOLO 重复检出"** ✗✗（用户 2026-10-04 ✓ 原话："帧 24
                    #   **#27 是凭空生成的**，它如果存在的话早就登记有编号了，实际这个检出框
                    #   **是 #1 的一部分（YOLO 重复检出）**" ✓✓）——
                    #   ⚠ 这种框后端**已经丢掉了**（不建档 ✓ 见 `_dedup` ✓）⇒ 面板必须**如实说明
                    #     原因** ✓，否则它会掉进下面那句"新出现 / 刚被挡住" ✗ ⇒ 看着像**跟踪漏了** ✓
                    #     （他报的就是这个 ✓）。
                    _dp = None
                    for _d in (_mo.get("dups") or []):
                        if (abs(float(_d[0]) - float(_b[1])) < 1.5
                                and abs(float(_d[1]) - float(_b[2])) < 1.5):
                            _dp = _d
                            break
                    if _dp is not None:
                        return tuple(_b) + (
                            "**YOLO 重复检出**：它 **%.0f%%** 落在 (%.0f,%.0f) 那格大框里"
                            " ⇒ 就是同一处的第二个框（所有目标等大 ⇒ 它不可能是独立目标）"
                            " ⇒ **已丢弃、不建档** ✓" % (100.0 * float(_dp[6]),
                                                       float(_dp[4]), float(_dp[5])),
                            "DUP DETECTION (dropped)")
                    # ⚠⚠ **文案必须"人话"** ✗✗（用户 2026-10-04 ✓ 他直接问："**什么是配对？
                    #   要归属谁？**" ✓ —— 我上一版写的是"未配对（这格还没有归属 ✓ 新出现 /
                    #   刚睡醒还没接上）"✗ 全是行话 ✓ 他看不懂 ✓ 这是我的问题 ✓）。
                    # ⭐⭐⭐⭐⭐ **"没配上"也要说清"它旁边那条账是谁"** ✗✗（用户 2026-10-06 ✓
                    #   原话："**这里明显应该是 #5 推的**" ✓✓）—— 原来一律回一句
                    #   "无编号：新出现，或刚被挡住又出来" ✗ ⇒ 实测（`9月30日(1)` 帧 24）
                    #   那格框**离 `#29` 的预测框只有 0px** ✗ 却只说"NO OWNER"⇒ 看着像**跟踪漏了** ✗
                    #   （其实旁边那条灰框就是它的 ✓）。⇒ 现在按 `orphans` 如实说三种话 ✓：
                    #     ① 旁边有账 ⇒ "**它应该是 #x 推的**" ✓（你要的那句 ✓）；
                    #     ② 有账但很远 ⇒ 说明距离 ✓；③ 真没有账 ⇒ 才说"附近确实没有账" ✓。
                    #   ⚠⚠ **界面文案不许带星号** ✗✗（用户 2026-10-04 ✓ 原话："这也太长了" ✓ ＋
                    #     他自己截的图里星号原样显示 ✓ —— 那是**我把写注释的习惯带进界面**的错 ✓）。
                    _ok = next((_o for _o in (_mo.get("orphans") or [])
                                if abs(float(_o[0]) - float(_b[1])) < 1.5
                                and abs(float(_o[1]) - float(_b[2])) < 1.5), None)
                    # ⭐⭐⭐⭐⭐ **"只有一部分在画面里"⇒ 说清"先不定号"** ✗✗（用户 2026-10-06 ✓
                    #   原话："**是否应该在其只漏部分"可疑"的时候先不急着标号，等漏全了比一下附近的
                    #   假目标**" ✓✓）—— ⚠ 实测：那格细条 (12,304) 上一版被我贴上 `#30` ✗ ⇒
                    #   用户一眼看出"**还是在 #5 占据的地方多了 #30**" ✓✗（框心是切出来的 ✗
                    #   凭它既判不了"是哪一条"、也不该给它新身份 ✓）。
                    if _ok is not None and bool(_ok[4] if len(_ok) > 4 else False):
                        return tuple(_b) + (
                            "无编号：它只有一部分在画面里（被切了）"
                            " ⇒ 先不定号，等它整个进画面再比",
                            "NO OWNER (CUT)")
                    if _ok is not None and _ok[2] is not None:
                        return tuple(_b) + (
                            "无编号：这格没配上框。旁边 #%d 的灰色预测框离它 %.0f px"
                            " ⇒ 它应该是 #%d 推的" % (int(_ok[2]), float(_ok[3]), int(_ok[2])),
                            "NO OWNER (#%d 推的)" % int(_ok[2]))
                    return tuple(_b) + ("无编号：新出现，或刚被挡住又出来"
                                        "（附近确实没有账的预测框）", "NO OWNER")
                _bid = int(_bv[4])
                _mk = float(_bv[6]) if len(_bv) > 6 else -1.0
                _tk = next((_t for _t in (_mo.get("tracks") or [])
                            if int(_t.get("tid") or 0) == _bid), None)
                # ⭐⭐⭐⭐ **分行、去掉 Markdown 标记**（用户 2026-10-04 ✓ 原话："**这也太长了，
                #   你能归纳信息分行显示吗？**" ✓✓）—— ⚠ 上一版有两个毛病，都是我的 ✗：
                #     ① **一行塞了七八项** ⇒ 长出画面 ✗；
                #     ② 我把**写注释的习惯**（`**加粗**` ✓/✗）**带进了界面文案** ✗✗
                #        ⇒ 界面上**原样显示星号** ✓ 看着很脏 ✓（用户截图里就是 ✓）。
                #   ⇒ 现在：**一行一个主题、最多三行、全用大白话** ✓：
                #     第 1 行 = **它是谁**（编号 + 这一格的匹配分）；
                #     第 2 行 = **它的数**（偏离 / 分数 / 有没有观测）；
                #     第 3 行 = **只在异常时出现**（融合 / 可疑 / 残缺 + 算法怎么办 ✓）。
                # ⭐⭐⭐⭐⭐ **"这一格到底是怎么来的"要说清** ✗✗（用户 2026-10-06 ✓ 原话："**为何刚
                #   出现就被标号了？**" ✓✓）：⚠ 实测那格显示的是 `编号 #20 ｜ 匹配分 -1.00` ✗ ——
                #   而 `-1` 是**内部哨兵值**（"没配上" ✓）⇒ 摊到界面上就是**天书** ✗。
                #   ⇒ 按 `_bv[12]` 分三种话（0=真配上 ✓ / 1=没配上、按旁边那条账贴的 ✓ /
                #     **2=这一拍刚为它建了新账** ✓）—— 用户问的那格正是 **2** ✓。
                _mk_kind = (int(_bv[12]) if len(_bv) > 12 else 0)
                if _mk_kind == 2:
                    _ls = ["编号 #%d ｜ 刚出现的框（这一拍为它新建的账）" % _bid]
                elif _mk_kind == 1:
                    _ls = ["编号 #%d ｜ 没配上框（按旁边那条账的预测贴的）" % _bid]
                else:
                    _ls = ["编号 #%d ｜ 匹配分 %.2f" % (_bid, _mk)]
                if _tk is not None:
                    _ls.append("偏离 %.1f ｜ 分数 %.0f ｜ %s"
                               % (float(_tk.get("dev") or 0.0),
                                  float(_tk.get("score") or 0.0),
                                  "有观测" if _tk.get("live") else "没观测（位置是推的）"))
                _stv = int(_bv[7]) if len(_bv) > 7 else 0
                if _stv:
                    # ⭐⭐⭐⭐⭐ **"含不含真目标"就按画面那个圆说 —— 三档，不许含糊** ✗✗
                    #   （用户 2026-10-04 ✓ 原话："**我们一定要所见即所得，不要表面搞一套背后
                    #   搞一套**" ✓✓）——
                    #   ⚠⚠ 判据 `_bv[8]`（`has_tgt` ✓）**已经和后端同一个圆**（`pos` ＋
                    #     `motion["tgt_radius"]` ✓ 见 `MotionTracker._circ_rad` 那段 ✓）⇒
                    #     **面板写的 = 画面上看到的** ✓✓；`_bv[9]` = **"就是真目标自己那格框"**
                    #     （第 10 位 ✓ 末尾统一填 ✓）。
                    #   ⚠ 为什么要**三档** ✗：原来只有两档 ⇒ 把"自己那格框"硬塞进"含真目标的
                    #     融合" ✗ ⇒ 帧 25 那个被**底边切碎**（159×99）、圆心只在框右边界**擦过
                    #     4.6px** 的框，也照说"裹着真目标和别人、框心是中点" ✗✗ —— 对它**根本
                    #     不成立**（自己的框心就是自己 ✓）。
                    _has = bool(_bv[8]) if len(_bv) > 8 else False
                    _own_b = bool(_bv[9]) if len(_bv) > 9 else False
                    _gap = None
                    if len(_bv) > 11 and _row.get("pos") is not None:
                        # ⚠ 半径也可能没有（还没学到 ✓）⇒ 那时只说"不相交"不打 px ✓
                        _gx = float(_row["pos"][0])
                        _gy = float(_row["pos"][1])
                        _gw = float(_bv[10])
                        _gh = float(_bv[11])
                        _gqx = min(max(_gx, _bv[0] - _gw / 2.0), _bv[0] + _gw / 2.0)
                        _gqy = min(max(_gy, _bv[1] - _gh / 2.0), _bv[1] + _gh / 2.0)
                        _gap = (((_gx - _gqx) ** 2 + (_gy - _gqy) ** 2) ** 0.5
                                - float(_row.get("tgt_rad") or 0.0))
                    if _stv == 1:
                        if _own_b:
                            # ⭐⭐⭐⭐⭐ **文案必须按"整个圆"如实说** ✗✗（用户 2026-10-06 ✓ 原话：
                            #   "**圆出框了，什么问题**" ✓✓）—— ⚠⚠ **我原来只写"绿圈落在框内 ✓"
                            #   ✗，可 `has_tgt` 的判据其实是"**圆盘与框相交**"**（≤ 半径 ✓）**✗
                            #   （见 `MotionTracker.process` 末尾"唯一圆 → 重算"那段 ✓）⇒ **说大了** ✓：
                            #   实测 `10月1日` **帧 19**：圆心 (319,255) **在框内** ✓（框
                            #   y 200.5~367.5、x 224~396 ✓），但**圆的顶边探出框上边 14.5px** ✗
                            #   （255 − 69 = 186 ✓）⇒ 面板照旧写"落在框内" ✗ ⇒ **与画面对不上** ✓
                            #   （违反"**所见即所得**"那条 ✓）。
                            #   ⚠ 为什么会探出 ✗（不是 bug ✓）：**融合框 = 两个目标的外接框、
                            #     框心是中点** ✓，而**圆 = 真目标自己**（半个）✓ ⇒ 圆偏在**上半块** ✓、
                            #     框又比圆的直径大不了多少（172×167 vs 直径 138 ✓）⇒ 圆心一挪到
                            #     自己那块的边上，圆边就**露到框外** ✓ —— 这正是用户 2026-10-06 的
                            #     新口径（"**整个圆**出框才往回夹"）**允许**的 ✓。
                            #   ⇒ 三档（**画面上是什么就说什么** ✓）：整圈在框内 ✓ ／ 圆心在框内、
                            #     圆边探出 N px ✓ ／ 圆心也探出 N px ✓（圆盘还搭在框上 ✓）。
                            _pk_c = _pk_e = None
                            if len(_bv) > 11 and _row.get("pos") is not None:
                                _prr = float(_row.get("tgt_rad") or 0.0)
                                _pk_c = max(abs(float(_row["pos"][0]) - _bv[0]) - _bv[10] / 2.0,
                                            abs(float(_row["pos"][1]) - _bv[1]) - _bv[11] / 2.0,
                                            0.0)
                                _pk_e = max(abs(float(_row["pos"][0]) - _bv[0])
                                            + _prr - _bv[10] / 2.0,
                                            abs(float(_row["pos"][1]) - _bv[1])
                                            + _prr - _bv[11] / 2.0,
                                            0.0)
                            #   ⭐⭐⭐⭐⭐ **融合框归根到底是假目标的座位** ✗✗（用户 2026-10-06 ✓ 原话：
                            #     "**如果检出框比作座位，真目标大部分时间都是没有座位的，它只能通过
                            #     融合框短暂的跟假目标『坐在一个座位上』，融合框最终肯定是优先还给
                            #     假目标**" ✓✓）—— ⚠ 所以面板**不许**再说"真目标自己那格框" ✗
                            #     （那是**框心在中点**的并集框 ✓，真目标只是**蹭坐** ✓）；有"主人"
                            #     （`_bv[13]` ✓ 见 `lie_motion` 的 `_FUSE_OWNER_HITS` 那段 ✓）时
                            #     **如实写主人是谁** ✓。
                            _ow5 = (int(_bv[13]) if len(_bv) > 13 and int(_bv[13]) >= 0 else None)
                            _pre5 = ("**融合框（两个目标挤一个座位）**：这格框是**假目标 #%d 的座位** ✓"
                                     "（它一直守着这一格 ✓）；真目标 `#%s` 是**蹭坐**的 ✓"
                                     "（⚠ 框心是**两目标中点**、不是它自己 ✗）"
                                     % (_ow5, _bid) if _ow5 is not None else
                                     "**真目标自己那格框**（不是融合别人）：框心就是**它自己** ✓")
                            if _has and _pk_e is not None and _pk_e > 0.5:
                                _ls.append(
                                    _pre5
                                    + ("，绿圈**探出框边 %.0f px** ✓（⚠ **圆心还在框里** ✓ ——"
                                       " 融合框心是**两目标中点**、圆是真目标自己 ✓ ⇒ 圆边露到"
                                       "框外是**正常**的 ✓；判据是「**整个圆出框才往回夹**」✓）"
                                       % _pk_c if _pk_c is not None and _pk_c <= 0.5 else
                                       "，绿圈**圆心就探出框边 %.0f px** ✓（圆盘还搭在框上 ✓ ⇒"
                                       " 按你 10-06 的口径**不夹** ✓）"
                                       % (0.0 if _pk_c is None else _pk_c)))
                            else:
                                _ls.append(
                                    _pre5
                                    + ("，绿圈**整圈落在框内** ✓"
                                       if _has else
                                       "；⚠ 但**绿圈离框 %.0f px** ✗ —— 这一段位置在**滑行**、没跟"
                                       "这个框 ✓（一眼可见 ✓）"
                                       % (0.0 if _gap is None else _gap)))
                        elif _has:
                            _ls.append("融合（**含真目标**）：框里裹着真目标和别人，框心是中点"
                                       " ⇒ 位置不采信（但它的变化能反推真目标 ✓）")
                        else:
                            _ls.append(
                                "融合（**跟真目标无关**）：框里是**两个别的目标**撞在一起"
                                " ✗ ⇒ 与真目标无关，别当成「真目标被融合了」"
                                + ("（绿圈离框 %.0f px ✗）" % _gap
                                   if _gap is not None else ""))
                    else:
                        # ⭐⭐⭐⭐⭐ **`2` 的文案改成"不冤枉距离"**（用户 2026-10-05 ✓ 原话：
                        #   "**他都跟圆相交了，它应该被认为"离预测真目标近"**" ✓✓ ｜ "**说明你的
                        #   匹配分算的有问题**" ✓✓）—— ⚠ 起因：原话术只写"**偏离预测较多**"✗，
                        #   于是"跟圆相交的框被判位置可疑"看着就自相矛盾 ✓；而 `m = 距离因子 ×
                        #   面积因子`，**一个数看不出来是哪一半低** ✗（实测本素材**干净框的面积
                        #   普遍是标准值的 1.15~1.35 倍** ✗ ⇒ 面积因子本来就 ≈0.5 ✓ ⇒ 很多"可疑"
                        #   其实是**面积的锅** ✗，不是距离 ✓）⇒ 文案**两半都写** ✓ 不含糊 ✓。
                        # ⭐⭐⭐⭐⭐ **`4` = 冷却 —— 但归属要按"主人"说** ✗✗（用户 2026-10-06 ✓ 原话：
                        #   "**真目标还是把属于 #11 的抢了**" ✓✓）—— 那格冷却框常常是**别人
                        #   （假目标）一直守着的座位** ✓（`_bv[13]` ✓ 见 `lie_motion` 的
                        #   `_FUSE_OWNER_HITS` 那段 ✓）⇒ 面板**不许**再写"真目标自己那格框" ✗
                        #   （实测 `10月1日` 帧 62：`#11` 就贴在那格上 **12px** ✓、下一拍框就
                        #   回到 `#11` ✓ = **座位本来就是它的** ✓）。
                        _ow4 = (int(_bv[13]) if len(_bv) > 13 and int(_bv[13]) >= 0 else None)
                        _t4s = ("**这格框是假目标 #%d 的座位**（它一直守着这一格 ✓）；真目标 `#%s`"
                                " 这一拍是**硬接**上来的 ✗（**冷却中**：融合已结束、框**已分开** ✓）"
                                % (_ow4, _bid) if _ow4 is not None else
                                "**真目标自己那格框**（**冷却中**：融合已结束、框**已分开** ✓）")
                        _ls.append({2: "可疑：这一格**不像一个干净的目标框**（离预测远，或框比一个大）⇒ 已降权",
                                    3: "残缺：框只剩一小块 ⇒ 位置不采信",
                                    # ⭐⭐⭐⭐⭐ **`4` = 冷却**（用户 2026-10-05 ✓ 原话："**拆状态码
                                    #   ＋ 冷却期位置采信**" ✓✓，起因："**这框这么小还能被判为融合
                                    #   框？**" ✓）—— 冷却期那格框**已经分开**（尺寸回到单目标 ✓）
                                    #   按铁律 2 它**不是**融合框 ✓ ⇒ 面板**不许**再写"融合"✗。
                                    4: _t4s + "：框心 ≈ 真目标中心 ⇒ **位置采信** ✓"
                                       "（⚠ 但框心**不参与**框心指导 ✓ —— 那条只在粘连期成立 ✓）"
                                    }.get(_stv, ""))
                    # ⭐⭐⭐ **"粘连中"还是"冷却中"要说清** ✗✗（用户 2026-10-04 ✓ 原话：
                    #   "把**冷却状态**需要写在**点选检出框的信息**里" ✓）—— ⚠ 两者**都**画成
                    #   融合紫框 ✓、`st` 码**都**是 1 ✗，但**含义完全不同** ✓：
                    #     · **粘连中**（`stuck>0`）⇒ 框心 = **两目标中点** ✓ ⇒ **参与框心指导** ✓；
                    #     · **冷却中**（`stuck==0` 但 `cool>0`）⇒ 框**已经分开**、框心 ≈ 真目标中心 ✗
                    #       ⇒ 位置**仍不采信** ✓、而且框心**不参与**框心指导 ✗（见 `lie_motion` ✓）。
                    if _stv in (1, 4) and _tk is not None:
                        _tk_sk = int(_tk.get("stuck") or 0)
                        _tk_ck = int(_tk.get("cool") or 0)
                        _ls.append(
                            "粘连中（第 %d 拍）：框心 = 两目标中点 ⇒ 参与框心指导" % _tk_sk
                            if _tk_sk > 0 else
                            "⚠ 冷却中（剩 %d 拍）：已分开、框心 ≈ 真目标中心 ⇒ **位置照旧采信** ✓、"
                            "但**不参与**框心指导 ✓" % _tk_ck)
                    # ⭐⭐⭐⭐⭐ **"夹取这一次给了速度多少"**（用户 2026-10-04 ✓ 原话："**位置被硬夹
                    #   说明你要么快了要么慢了，夹取也要算进速度平滑。因为夹取是修正你的预测
                    #   误差，夹取是较准确的测量值**" ✓✓）——
                    #   ⚠⚠ **必须写出来** ✗（"所见即所得" ✓）：夹取不只挪了位置 ✓，它**还改了速度**
                    #     ✓ ⇒ 而"黄箭头"（= `vel − 相机` ✓）下一拍就会跟着变 ✓ ⇒ 不写出来
                    #     就成了"画面自己变了、看不出原因" ✗✗。
                    #   ⚠ 只在**这一拍真夹了**才写 ✓（`None` ⇒ 什么都不写 ✓ 不误导 ✓）。
                    _cv = _row.get("clamp_vel")
                    if _stv in (1, 4) and _cv is not None:
                        _ls.append(
                            "夹取：位置被硬夹 ⇒ **速度也跟了 (%.1f, %.1f) px/拍** ✓"
                            "（= 把预测误差修在源头 ⇒ 下一拍自然走到位、不用每拍再夹 ✓）"
                            % (float(_cv[0]), float(_cv[1])))
                return tuple(_b) + ("\n".join([_x for _x in _ls if _x]), "")
            _btxt, _rtxt = pick_extra((float(_b[1]), float(_b[2]),
                                       float(_b[3]), float(_b[4])),
                                      _row.get("pos"), _row.get("tgt_rad"), _br)
            return tuple(_b) + (_btxt, _rtxt)

        def _on_pick_box(self, pt):
            """左键点了画面（`pt` = **加工域坐标** ✓）⇒ 命中那格检出框并显示其几何/面积。

            用户口径（2026-10-03 ✓ 原话："你能让我在点选检出框的时候显示
            `(360.4,323.0) 179×208=37232` 这种信息吗" ✓）：
              · 命中 = 点落在框内（**多个套着 ⇒ 取面积最小的那个** ✓ 里层优先 ✓）；
              · 没落在任何框里 ⇒ 退回**中心最近**且距离 ≤ 40px 的那格（手抖差一点也能选上 ✓）；
              · 点空白（连近的都没有）⇒ **取消**选中 ✓（再点同一格/别处照常 ✓）。
            ⚠ 纯显示层 ✗：不选中任何"判定"、不改红框/绿圈/参数 ✓（只是把数字摆出来 ✓）。
            """
            if not self.results or not self.frames:
                return
            _i = max(0, min(getattr(self, "i", 0), len(self.results) - 1))
            _best = pick_box_at(self.results[_i].get("boxes"), pt)   # ⭐ 命中逻辑在模块函数里 ✓ 好测 ✓
            self._pick = None if _best is None else {"frame": _i, "box": _best}
            self._show_precomputed()                    # 立刻重画（带高亮 ✓）
            self._status(_i)                            # 状态栏写上这一格的数字 ✓

        def _on_pick_menu(self, pt):
            """**右键点选检出框 ⇒ 选中 + 弹菜单**（用户 2026-10-03 ✓ 原话："加一个**右键点击检出框
            选中并弹出菜单**，目前只有一项『**复制检出框信息**』，用来我**复制之后与你交流**" ✓）。

            · 命中逻辑与左键**同一份**（`pick_box_at` ✓ 里层优先 / 差一点取最近 ✓ 不各写一份 ✗）；
            · ⚠ **没点在检出框上 ⇒ 不开菜单** ✓（不猜 ✗ —— 那唯一一项对"空选中"没有意义 ✓）；
            · 选中后**照左键那条路**刷新（高亮 + 状态栏 ✓），再弹菜单 ✓；
            · 菜单目前**只有一项** ✓（用户明说"目前只有一项" ✓ 结构留好 ✓ 以后加项就在这加 ✓）。
            ⚠ 纯显示层 ✗：不改任何判定 / 参数 ✓ —— 只是把这一格的数字**放进剪贴板** ✓。
            """
            if not self.results or not self.frames:
                return
            _i = max(0, min(getattr(self, "i", 0), len(self.results) - 1))
            _best = pick_box_at(self.results[_i].get("boxes"), pt)
            if _best is None:
                return                              # 没命中 ⇒ 不开菜单 ✓
            self._pick = {"frame": _i, "box": _best}
            self._show_precomputed()
            self._status(_i)
            _txt = self._pick_clip_text()
            _m = QMenu(self)
            _m.addAction("复制检出框信息").triggered.connect(
                lambda _=False: self._copy_pick_text(_txt))
            _m.exec_(QCursor.pos())                 # 在**鼠标处**弹（右键就该在指的地方 ✓）

        def _pick_clip_text(self):
            """当前点选框的**剪贴板文本**（组装在纯函数 `pick_clip_text` 里 ✓ 好测 ✓）。"""
            _pk = self._pick_box()
            if _pk is None:
                return ""
            _i = getattr(self, "i", 0)
            _st = None
            if self.results and 0 <= _i < len(self.results):
                _st = self.results[_i].get("state")
            return pick_clip_text(_pk, self.names, frame_no=(_i + 1),
                                  total=(len(self.frames) if self.frames else None),
                                  state=_st)

        def _copy_pick_text(self, txt):
            """写剪贴板 + 状态栏给个回执（用户 2026-10-03 ✓ 「用来我**复制之后与你交流**" ✓）。

            ⚠ 回执只有 6 秒、且播放中会被 `_status`（每帧刷 ✓）盖掉 ✗ —— 无所谓 ✓：用户复制完
              就切去粘贴了 ✓，这里只求"点了有反应" ✓（不为了回执去改状态栏那套刷新机制 ✗）。
            """
            if not txt:
                return
            QApplication.clipboard().setText(txt)
            self.statusBar().showMessage("已复制检出框信息（%d 字符）⇒ 直接粘贴即可 ✓" % len(txt),
                                         6000)

        def _show_precomputed(self):
            """按演算表画**当前帧**（拖动定位时用 ✓ 零等待 —— 用户要求 ② ✓）。"""
            if not self.results:
                return
            # ⚠⚠ **必须夹住**（2026-09-30 离屏测试踩到 ✓）：滑块的 range 一载入就设成
            #   全部帧数，而 `results` 是**边演算边长**的 ⇒ 演算没跑完时拖动 ⇒
            #   `self.results[self.i]` **IndexError 崩** ✗（pythonw 里只打 traceback、
            #   界面看着像"没反应" ✗）。夹到已算好的最后一帧 ⇒ 最多停一拍，不崩 ✓。
            self.i = min(self.i, len(self.results) - 1)
            r = self.results[self.i]
            big = self._big_of(self.i)          # ⚠ 原图那份是 JPEG ✓ ⇒ 解一次（带缓存 ✓）
            n = sum(1 for x in self.results[:self.i + 1] if x["pos"] is not None)
            h = sum(1 for x in self.results[:self.i + 1] if x["hit"])
            hud = "t=%5.2fs  %s  hit=%.0f%%" % (
                self.t, r["state"], 100.0 * h / n if n else 0.0)
            mk = self.view()
            # ⭐⭐ **右列（KF 位置驱动）**（用户 2026-10-03 ✓ 双列对照 ✓）：拖动/改视图参数时
            #   也一起刷新 ✓（同一帧号 ✓ 各取各的演算表 ✓）；KF 关 ⇒ `None` ⇒ 单列老样子 ✓。
            vis2 = None
            if self.results_kf:
                _i2 = min(self.i, len(self.results_kf) - 1)
                if _i2 >= 0:
                    r2 = self.results_kf[_i2]
                    vis2 = draw(big.copy(),
                                (None, r2["pos"], r2["r"], r2["hit"], r2["boxes"]),
                                self.scale[0], self.scale[1], hud + "  [KF]",
                                r2["cursor"], r2.get("motion"), self.names, mk)
            # ⭐⭐⭐ **图层过滤**（用户 2026-10-03 ✓ 他要按自己的需要开关图层 ✓）：在**显示入口**
            #   这里过滤，**不动 `draw()` 一个字** ✗ ⇒ 风险最小 ✓ 且经典模式完全不受影响 ✓。
            _sh = self._show_layers()
            _mo_d = dict(r.get("motion") or {})
            _mo_d["show"] = _sh
            if _sh.get("panel", True) is False:                # 关掉"候选圈"⇒ 候选 + 箭头 + 面板都不画 ✓
                _mo_d["tracks"] = []
            elif not _sh.get("rel_tgt", True):  # 只关「真目标轨迹」⇒ 保留候选圈 ✓
                _mo_d["tracks"] = [dict(_t, rel_hist=[]) for _t in (_mo_d.get("tracks") or [])]
            # ⚠⚠ **鼠标那条要单独清** ✗✗（用户 2026-10-04 ✓ "相对轨迹"拆成两个了 ✓）——
            #   它不在 `tracks` 里 ✗ 而是 `motion["cursor_rel"]` ✓ ⇒ 得清这个键 ✓
            #   （⚠ `draw()` 那边也读了开关 ⇒ **双保险** ✓ 这里清掉能让工具条一勾就生效 ✓）。
            if not _sh.get("rel_mouse", True):
                _mo_d["cursor_rel"] = []
            _boxes_d = (r["boxes"] if _sh["boxes"] else [])
            # ⚠⚠ **HUD 也归「信息面板」开关管** ✗（用户 2026-10-04 ✓ "左上角这些框…太挡视野"
            #   ✓）—— 它就是左上角那行 `t=2.50s track hit=100%` ✓ 和下面那块黑底面板是**两块** ✓
            #   ⇒ 一个开关**一起收** ✓（传 `None` ⇒ `draw` 里 `if hud:` 直接跳过 ✓）。
            _hud_d = (hud if _sh.get("panel", True) else None)
            # ⭐⭐⭐ **速度跟踪：HUD 一律不画**（用户 2026-10-07 ✓ "画面上正好只有那 7 条图例、
            #   别无他物" ✓）—— 左上角那行 `t=2.50s track hit=100%` **不在那 7 条里** ✗ ⇒ 传
            #   `None`（`draw` 里 `if hud:` 直接跳过 ✓ **不动 `draw` 一个字** ✓）。
            #   ⚠ 口径 = **只在速度跟踪档** ✗；运动分离档照旧听「信息面板」那个开关 ✓（零污染 ✓）。
            if self._vel():
                _hud_d = None
            self.show_frame(draw(big.copy(),
                                 (None, r["pos"], r["r"], r["hit"], _boxes_d),
                                 self.scale[0], self.scale[1], _hud_d, r["cursor"],
                                 _mo_d, self.names, mk,
                                 pick=(self._pick_box() if _sh["boxes"] else None),
                                 # ⭐ KF 影子可视化（用户 2026-10-03 ✓ 青线/青点/青箭头 ✓）
                                 kf=r.get("kf"), kf_trail=self._kf_trail(self.i)),
                            vis2)
            self._status(self.i)      # ⭐ 帧号一起刷新（拖动/改视图参数都走这儿 ✓）

        def on_seek(self, value):
            """拖动中：直接显示**已演算好**的那一帧（含红点/绿圈 ✓）—— 不重算 ✓。"""
            if not self.frames:
                return
            self.i = max(0, min(value, len(self.frames) - 1))
            self.t = self.ts[self.i] if self.ts else 0.0     # ⭐ 真实时间戳 ✓
            self._show_precomputed()
            self.upd_time()

        def _status(self, i=None):
            """状态栏那行（**帧号** + 状态 + 命中率）—— 播放与拖动**共用**它。

            ⚠⚠ 2026-09-30 用户报："拖动播放条的时候，下面显示的帧号没变" ✓ —— 真因：
            这行原来**只写在 `tick`（播放）里** ✗，而拖动走 `on_seek` ⇒ 帧号一直停在
            播放时那一帧 ✗（画面在动、数字不动，看着就像坏了 ✗）。
            """
            if not self.results or not self.frames:
                return
            i = self.i if i is None else i
            i = max(0, min(i, len(self.frames) - 1, len(self.results) - 1))
            r = self.results[i]
            n = sum(1 for x in self.results[:i + 1] if x["pos"] is not None)
            h = sum(1 for x in self.results[:i + 1] if x["hit"])
            acc = 100.0 * h / n if n else 0.0
            # ⭐⭐ **登记框与检出框的错位量**（中位 px ✓ 2026-10-01 加 ✓）：一眼就能判
            #   "登记表到底有没有贴合检出"（用户几次问"为什么错开" ⇒ 把它变成**数字** ✓
            #   截图也能直接对 ✓）。算法 = 每个检出 → 最近"已上板"条目的距离，取中位 ✓；
            #   0.0 = 全部严丝合缝 ✓。
            _mo = r.get("motion") or {}
            _reg = [e for e in (_mo.get("reg") or []) if e.get("ok")]
            #   ⚠⚠ 口径必须是"**已上板条目 → 最近检出**"✗✗ —— 反过来（检出 → 最近条目）会
            #   **跨砖匹配**：刚滚进画面、还没有条目的新砖，会被配到**邻砖**的条目上 ⇒ 算出
            #   十几~几十 px 的"假错位"（我自己就被它误导了一轮 ✓ 实测帧 20：反过来量是中位
            #   17.8px，正着量是 0.0px ✓）。中位对"少数顶着不动的条目"不敏感 ✓。
            _ds = []
            for _e in _reg:
                _bd = None
                for _b in (r.get("boxes") or []):
                    _dd = ((float(_e.get("x", 0.0)) - float(_b[1])) ** 2
                           + (float(_e.get("y", 0.0)) - float(_b[2])) ** 2) ** 0.5
                    if _bd is None or _dd < _bd:
                        _bd = _dd
                if _bd is not None:
                    _ds.append(_bd)
            _ds.sort()
            _mis = ("%.1f" % _ds[len(_ds) // 2]) if _ds else "-"
            # ⭐⭐⭐ **点选的检出框**（用户 2026-10-03 ✓）：把那格的数字**也写进状态栏** ✓
            #   —— 与画面上的黄框标签同一份（`(cx,cy) W×H=面积` ✓ 口径 = 加工域 ✓）。
            _pk = self._pick_box()
            _pks = ""
            if _pk is not None:
                _pks = " ｜ 点选 (%.1f,%.1f) %d×%d=%d" % (
                    float(_pk[1]), float(_pk[2]), int(round(float(_pk[3]))),
                    int(round(float(_pk[4]))),
                    int(round(float(_pk[3]) * float(_pk[4]))))
                # ⭐⭐ **第 2 行那几个量也写进来**（用户 2026-10-03 ✓ 原话："以上参数及旧的 iou
                #   显示**都放在第二行**" ✓）—— 状态栏只有一行 ⇒ 接在同一行**末尾** ✓
                #   （与画布标签**同一份数据** ✓ 一个算法两处显示 ✓ 不各算一份 ✗）。
                _pkx = " ｜ ".join([str(_x) for _x in _pk[6:] if _x])
                if _pkx:
                    _pks = "%s ｜ %s" % (_pks, _pkx)
            # ⭐⭐⭐ **KF 影子对照**（用户 2026-10-03 ✓）：只在"位置估计 = KF 影子"时显示 ✓
            #   数据来自 `LieTracker.kf_summary()`（**全片累计** ✓ 纯读 ✓ 不影响任何判定 ✗）：
            #   · `差 p50/p90` = KF 估的位置与经典位置的差（px ✓）；
            #   · `新息 p50/p90` = 观测 − 先验预测（px ✓）—— **它就是"经典输出里那部分抖动"**
            #     的量级 ✓ ⇒ 直接告诉我们观测噪声 σ 该设多大（现在默认 6px ✓ 见 `kf.KF_R_DEF` ✓）。
            _kfs = ""
            try:
                _ks = self.runner.tr.kf_summary() if self.runner is not None else None
                if _ks:
                    _kfs = (" ｜ KF 影子(全片 n=%d) 差 p50 %.1f/p90 %.1f px ｜ 新息 p50 %.1f/p90 %.1f px"
                            " ｜ 🟦青线=KF 轨迹 ｜ 青点/青圈=KF 位置 ｜ 青箭头=KF 速度"
                            % (int(_ks["n"]), float(_ks["d_p50"]), float(_ks["d_p90"]),
                               float(_ks["innov_p50"]), float(_ks["innov_p90"])))
            except Exception:                       # noqa: BLE001 —— 状态栏不许因为诊断崩 ✗
                _kfs = ""
            # ⚠⚠ **运动分离模式下，"登记条数 / 错位中位"这两项没有意义** ✗✗ —— 它们是**经典模式**
            #   的"假目标登记表 vs 检出框"两个量 ✓ 而本模式**根本没有登记表** ✗ ⇒ 恒显示
            #   `登记 0 条 ｜ 错位中位 -px` ✗（用户 2026-10-03 直接问："**信息栏位置状态有坐标显示，
            #   但是下面坐标是 —**" ✓ 问的就是这个 ✗）⇒ 换成**本模式真正有意义的量** ✓：
            #   **候选条数 + 目标的（分数 / 本拍相对偏离）** ✓。
            _mo_s = r.get("motion") or {}
            if _mo_s.get("mode") == "velocity":
                # ⭐⭐⭐ **速度跟踪：这行只留「帧号 / 状态 / 命中率」** ✗✗（用户 2026-10-07 ✓
                #   "做减法：清空三处显示" ✓ —— 底栏中段（`_mid`）正是那三处之一 ✓）。
                #   ⚠⚠ **不能只把 `_mid` 置空** ✗（我第一版就想那么干 ✓）：下面那个模板带着
                #     `｜ ` 前缀分隔符 ⇒ 会留下一个**悬空的分隔符** ✓ ⇒ 这里**单独拼一行** ✓。
                #   ⚠ `_pks`（点选）在速度跟踪档本来就恒空 ✓（`_pick_box` 恒回 `None` ✓ 见那里 ✓）、
                #     `_kfs`（KF）也恒空 ✓（`_show_kf` 该档恒关 ✓）。
                self.statusBar().showMessage(
                    "帧 %d/%d ｜ %s ｜ 命中率 %.1f%%（%d/%d）"
                    % (i + 1, len(self.frames), r["state"], acc, h, n))
                #   ⚠⚠ **日志区照旧刷** ✗✗（用户 2026-10-07 ✓ **更正**："**日志区需要保留，我说的是
                #     清除要打印的东西，不是把日志功能删了**" ✓）—— 喂进去的本来就是**空的** ✓
                #     （这一档 `log_text` 空 ✓ 也没那几条 `*_log` ✓ 见 `_on_mode_changed` ✓）
                #     ⇒ 日志区**空着但功能在** ✓；本档底栏中段照旧**单独拼一行** ✓（上面那句 ✓）。
                self._log_sync(i)
                return
            if _mo_s.get("mode") == "motion":
                _mid = "候选 %d 条 ｜ 目标 %s" % (
                    len(_mo_s.get("tracks") or []),
                    "（还没定）" if _mo_s.get("sel_score") is None else
                    "分数 %.0f ／ 本拍相对偏离 %.1f px"
                    % (float(_mo_s["sel_score"]), float(_mo_s.get("sel_dev") or 0.0)))
            else:
                _mid = "登记 %d 条 ｜ 错位中位 %s px" % (len(_reg), _mis)
            self.statusBar().showMessage(
                "帧 %d/%d ｜ %s ｜ 命中率 %.1f%%（%d/%d）｜ %s%s%s"
                % (i + 1, len(self.frames), r["state"], acc, h, n, _mid, _pks, _kfs))
            self._log_sync(i)                    # ⭐ 底部融合日志区跟着这一帧刷 ✓

        def _log_sync(self, i):
            """把底部**深色日志区**刷成「**截至第 i 帧**的融合框事件"（用户 2026-10-02 ✓）。

            · 正常播放（`i` 递增）⇒ **只追加新帧**的事件 ✓ O(1)/帧 ✓（不会随帧数变卡 ✗）；
            · 往回拖 / 换了素材（`results` 变短）⇒ **整段重建**（清空 + 从 0 扫到 i ✓）——
              拖动来回看**不重复刷** ✗、也一条不漏 ✓。

            ⚠ 只显示追踪器给的 `merge_log["text"]` ⇒ **判据不在这里重算** ✗（口径一处 ✓
              见 `LieTracker._build_merge_log` ✓）。
            """
            _n = len(self.results)
            if i < 0 or i >= _n:
                return
            if i < getattr(self, "_log_i", 0) or getattr(self, "_log_n", 0) > _n:
                self.log.clear()
                self._log_i = 0
            for k in range(self._log_i, i + 1):
                # ⭐⭐⭐ **运动分离模式的「通俗说明」**（用户 2026-10-03 ✓ 原话："我希望：你能把你
                #   大概的**计算**、必要的我可以跟你**沟通**的信息在**日志**里写出来
                #   （**最好通俗易懂**）" ✓）—— 每拍一行：相机走了多少／目标分数与位移／
                #   它是不是"最不合群"的那个／**有没有换目标及为什么** ✓
                #   ⇒ 用户**截一行日志就能跟我对话** ✓✓（比让他描述画面强得多 ✓）。
                _mo = self.results[k].get("motion") or {}
                if _mo.get("log_text"):
                    self.log.appendPlainText("帧 %d ｜ %s" % (k + 1, _mo["log_text"]))
                _m = self.results[k].get("merge_log")
                if _m:
                    self.log.appendPlainText("帧 %d ｜ %s" % (k + 1, _m.get("text", "")))
                # ⭐⭐ **融合框消失（分离）**（用户 2026-10-02 ✓）：与融合日志**同格式** ✓ 一起进
                #   这个日志区 ✓ —— 同一帧可能两条都有（"旧融合框作废 + 新融合框生成"同拍发生 ✓
                #   正是 `9月30日(1).mp4` 第 45 帧那种情形 ✓）并排看最清楚 ✓。
                # ⭐⭐ **融合期"每拍"的边归属**（用户 2026-10-02 ✓ 原话："日志里，融合时期每拍
                #   打印一下**边的归属（砖或真目标）**" ✓）：夹在"生成"与"消失"之间 ✓ ——
                #   于是日志区里一段融合看起来就是：生成 → 边归属×N → 消失 ✓ 一眼看到归属怎么变 ✓。
                _e = self.results[k].get("edge_log")
                if _e:
                    self.log.appendPlainText("帧 %d ｜ %s" % (k + 1, _e.get("text", "")))
                # ⭐⭐⭐ **非融合红框选择**（用户 2026-10-03 ✓ 原话："在日志加一下**非融合红框
                #   选择**的判定信息吧" ✓）—— 与融合 / 边归属 / 分离**四条互补** ✓：
                #   一条融合段在日志里看起来就是：生成 → 边归属×N → 分离；而**不融合的红框**
                #   （"目标自己那格"）由本条给出"凭什么挑中它" ✓（判据仍在 `_build_pick_log`
                #   那边组句子 ✓ 这里只显示 ✗ 口径一处 ✓）。
                _p = self.results[k].get("pick_log")
                if _p:
                    self.log.appendPlainText("帧 %d ｜ %s" % (k + 1, _p.get("text", "")))
                _s = self.results[k].get("split_log")
                if _s:
                    self.log.appendPlainText("帧 %d ｜ %s" % (k + 1, _s.get("text", "")))
            self._log_i = i + 1
            self._log_n = _n

        def on_seek_done(self):
            if getattr(self, "_was_playing", False) and self.results:
                self.playing = True
                self.btn_play.setText("暂停")
                self._play_anchor()      # ⭐ 拖完重锚（不然会一路"追"刚才那段时间 ✗ 见 `play_delay_ms` ✓）
                self.retime()

        def upd_time(self):
            self.lbl_time.setText("%.1f / %.1f s" % (self.t, self.dur))

        # ---- 播放 ----
        def tick(self):
            """播放 = **查演算表**（用户要求 ② ✓ 不重算 ✓ 拖帧也零等待 ✓）。"""
            if not self.frames or not self.results:
                return
            if self.i >= len(self.frames):
                if self.chk_loop.isChecked():
                    self.i = 0
                    self.t = self.ts[0] if self.ts else 0.0
                    self._play_anchor(0)     # ⭐ 循环回头也要重锚 ✓（否则第二圈会一路狂追 ✗）
                else:
                    self.stop()
                    return
            big = self._big_of(self.i)          # ⚠ 同上（JPEG ⇒ 解一次 ✓ 带缓存 ✓）
            r = self.results[self.i]
            n = sum(1 for x in self.results[:self.i + 1]
                    if x["pos"] is not None)
            h = sum(1 for x in self.results[:self.i + 1] if x["hit"])
            acc = 100.0 * h / n if n else 0.0
            hud = "t=%5.2fs  %s  hit=%.0f%%" % (self.t, r["state"], acc)
            mk = self.view()
            vis = draw(big.copy(), (None, r["pos"], r["r"], r["hit"], r["boxes"]),
                       self.scale[0], self.scale[1], hud, r["cursor"],
                       r.get("motion"), self.names, mk,
                       pick=self._pick_box(),               # ⭐ 点选的检出框（黄框+数字 ✓）
                       # ⭐ KF 影子可视化（用户 2026-10-03 ✓ 青线/青点/青箭头 ✓）
                       kf=r.get("kf"), kf_trail=self._kf_trail(self.i))
            # ⭐ 把 motion 实时数据 set 到侧栏 info_live（用户 2026-10-04 ✓ 侧栏 ✓）⇒ 每拍刷新。
            self._update_info_live(r.get("motion"))
            # ⭐⭐⭐ **右列（KF 位置驱动）**（用户 2026-10-03 ✓ 原话："左边跑经典右边跑 **KF**，
            #   这样对照才有意义" ✓）：**同一帧号**取它自己的演算表 ✓ —— 两列各画各的 ✓。
            #   ⚠ 右列**不叠加青色影子**（它就是 KF 本身 ✓ 用户原话"叠加 KF 显示意义不大" ✓）；
            #     也不画日志/点选高亮（那些是左列的事 ✓ 免得两列抢同一份 `_pick` ✓）。
            vis2 = None
            if self.results_kf:
                _i2 = min(self.i, len(self.results_kf) - 1)
                if _i2 >= 0:
                    r2 = self.results_kf[_i2]
                    vis2 = draw(big.copy(),
                                (None, r2["pos"], r2["r"], r2["hit"], r2["boxes"]),
                                self.scale[0], self.scale[1], hud + "  [KF]",
                                r2["cursor"], r2.get("motion"), self.names, mk)
            if self.rec is not None:
                self.rec.write(vis)
            self.show_frame(vis, vis2)
            self._status(self.i)                 # 状态栏（与拖动**共用**同一份 ✓）
            self.retime()                        # ⭐ 这一帧该停多久 = 它的**真实**帧间隔 ✓
            self.i += 1
            self.t = self.ts[self.i] if self.i < len(self.ts) else self.dur
            if not self.sld.isSliderDown():      # 拖动中不抢滑块 ✗
                self.sld.blockSignals(True)
                self.sld.setValue(min(self.i, len(self.frames) - 1))
                self.sld.blockSignals(False)
            self.upd_time()

        def _fetch_names(self):
            """后台问"类别名"（⚠ 主线程不许等它 ✗ —— 见 `warm_tick` 里那段注释 ✓）。

            ⚠ 为什么值得单开线程 ✗：`DetsWorker.names()` 要**起子进程 + 加载模型** ✓ **实测 3.0s** ✗；
              它只在**第一拍演算**时问一次 ✓ ⇒ 放主线程就是"一开窗先冻 3 秒" ✓（用户报的"卡"里
              很可能就有这一份 ✓）。⚠ 等 subprocess 时 Python 是**放开 GIL** 的 ✓ ⇒ 线程里等，
                界面照常刷新 ✓。
            ⚠ `self.names` **只写一次** ✓（主线程只读 ⇒ 不用锁 ✓）；没问到之前绘制退回显示编号 ✓。
            """
            try:
                _nm = DetsWorker.names(args.weights or None) or []
            except Exception:                   # noqa: BLE001 —— 问不到就退回编号 ✓
                _nm = []
            self.names = _nm

        def _big_of(self, _i):
            """第 `_i` 帧的**原图**（BGR ✓）—— ⚠⚠ **带一格缓存** ✗（用户 2026-10-07 ✓ 见
            `_BIG_JPEG_Q` ✓）。

            ⚠ 为什么必须缓存 ✗：原图现在是 **JPEG 字节** ✓ ⇒ 每次要都解一遍（1080p ≈ **6ms** ✗）；
              而同一帧会被问**好几次** ✓（播放那一拍：状态栏刷新 ＋ 画面重画 ＋ 录像 ✓）⇒ 不缓存就是
              白花 ✗。一格够用 ✓（访问是**顺序**的 ✓ 播放/拖动都只碰当前那一帧 ✓）。
            """
            if getattr(self, "_bigc_i", None) == _i and getattr(self, "_bigc_a", None) is not None:
                return self._bigc_a
            self._bigc_a = decode_big(self.frames[_i][0])
            self._bigc_i = _i
            return self._bigc_a

        def show_frame(self, bgr, bgr2=None):
            # ⭐⭐ **右列（KF 位置驱动）**（用户 2026-10-03 ✓ 双列对照 ✓）：`bgr2` 给了就一起显示 ✓
            #   （同一帧号、各画各的演算表 ✓）；⚠ 两列的缩放**各自独立**（各自滚轮/双击 ✓）。
            if bgr2 is not None and getattr(self, "lbl2", None) is not None:
                _rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
                _h2, _w2, _ = _rgb2.shape
                _qi2 = QImage(_rgb2.data, _w2, _h2, 3 * _w2, QImage.Format_RGB888).copy()
                self.lbl2.set_image(_qi2, self.scale[0], self.scale[1])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            # ⭐ 交给 `_ImageView` 画（自己管缩放/适应/坐标 ✓ 原 setPixmap 的"塞满窗口"由
            #   `_fit` 模式承担 ✓）—— 传原图→加工域的缩放，坐标显示才和红框/登记同一把尺 ✓。
            self.lbl.set_image(qi, self.scale[0], self.scale[1])

        def eventFilter(self, obj, ev):
            """**把 ←→ / 空格 从数字框手里抢回来**（用户 2026-10-01 ✓ 「不管聚焦在哪" ✓）。

            ⚠ 只有**数字框**需要这一手（`__init__` 里只装在 `sp_path_ms` / `sp_merge` 上 ✓）：
              其余子控件（滑块/下拉/勾选框/按钮 ✓）本来就不 accept 这个键 ⇒ 窗口快捷键自然
              优先 ✓（实测验过 ✓）。
            ⚠ 返回 `True` = **吃掉**这个事件 ✗：这样 `QSpinBox` 内部那个 `QLineEdit` 就**没机会
              accept** 它 ⇒ `QShortcutMap` 判"键不归控件" ⇒ 照常匹配我们的窗口快捷键 ✓
              （Qt 判定按键归属的依据正是 `ShortcutOverride` 有没有被 accept ✓）。
            ⚠ 只吞**这一个键**的 `ShortcutOverride`（不吞 `KeyPress` ✗）：真要没匹配上快捷键，
              按键照旧走到控件（不会把输入能力整个废掉 ✓）。
            """
            if (ev.type() == QEvent.ShortcutOverride and obj in getattr(self, "_steal_on", ())
                    and ev.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Space)):
                ev.ignore()          # 关键：**保持"没被 accept"**（accept 了就等于把键让回去 ✗）
                return True
            # ⭐⭐ **下拉也要挡滚轮**（项目规矩"滚轮不许改参数" ✓）：数字框已经有 `NoWheel*` ✓，
            #   而 `QComboBox` **默认会被滚轮切档** ✗ ⇒ 鼠标从上面划过就偷偷换了算法分支 ✗✗。
            #   ⚠ 只挡**下拉** ✗（底部日志区照旧能滚 ✓ —— 那是它唯一的用途 ✓）。
            if ev.type() == QEvent.Wheel and isinstance(obj, QComboBox):
                return True
            return super().eventFilter(obj, ev)

        def keyPressEvent(self, e):
            """**兜底**键盘（真正干活的是 `__init__` 里那几个 `QShortcut` ✓ 见那边的说明 ✓）。

            ⚠ 这里只能当兜底 ✗：`keyPressEvent` 收到按键的前提是"**焦点在本窗口自己身上**"
              —— 焦点一旦落到某个会吃按键的子控件（进度条/蒙版滑块/下拉/勾选框/按钮/数字框 ✓）
              就**永远走不到这儿** ✗（这正是用户报的"有时候能切、有时候不能"✗）。
              所以 ←→ / 空格 的正主是窗口级 `QShortcut`（**不管焦点在哪都触发** ✓）；
              这里保留同款分支**不是重复** ✓ 而是"万一快捷键被别的窗口/上下文抢走"时的保险 ✓
              （快捷键命中时按键会被吃掉 ⇒ 不会双触发 ✓）。
            O / G 仍是**字母键** ⇒ 不做成窗口级快捷键 ✓（免得以后哪个输入框想用字母时被抢 ✗）。
            """
            if e.key() == Qt.Key_Space:
                self.toggle_play()
            elif e.key() == Qt.Key_Left:
                self.step_frame(-1)
            elif e.key() == Qt.Key_Right:
                self.step_frame(1)
            elif e.key() == Qt.Key_O:
                self.open_dialog()
            elif e.key() == Qt.Key_G:              # ⭐ G = 选 GIF（O 已被视频占了 ✓）
                self.pick_gif()
            else:
                super().keyPressEvent(e)

        def _apply_dets_visible(self):
            """「**用检测器**」在速度跟踪档**从工具栏上真的拿掉**（用户 2026-10-07 ✓ 原话：
            「**用检测器 这个没有必要 去掉**」✓）。

            ⚠⚠ **为什么不用 `setVisible(False)`** ✗✗（**探针实测** ✓）：`chk_dets` 的父是
              `QToolBar` ✓，而工具栏**自己管**它的项的显隐 ✗ ⇒ `setVisible(False)` 当场是
              `hidden=True` ✓，可**下一拍 `processEvents` 就又变回 False** ✗（工具栏重新布局 ✓）
              ⇒ 只有 `removeAction` 才真的拿掉 ✓；回来时按**原来那个位置**插回去 ✓（顺序不变 ✓）。
            ⚠ 只是"不显示" ✗ —— 它照旧**勾着** ✓（`apply_path_ms` 里那道自动勾 ✓）⇒ 这一档的
              观测**就是**检出框 ✓ 检测照跑 ✓。
            """
            if getattr(self, "chk_dets", None) is None or getattr(self, "act_dets", None) is None:
                return
            _tb = self.chk_dets.parentWidget()
            _act = self.act_dets
            if _tb is None:
                return
            if self._vel():
                if _act in _tb.actions():
                    _tb.removeAction(_act)
            elif _act not in _tb.actions():
                _after = getattr(self, "act_dets_after", None)
                if _after is not None and _after in _tb.actions():
                    _tb.insertAction(_after, _act)
                else:
                    _tb.addAction(_act)

        def showEvent(self, e):
            """窗口**显示**时补一次模式相关的显隐（用户 2026-10-07 ✓）。

            ⚠⚠ **为什么非补不可** ✗✗（**实测踩到** ✓）：`_on_mode_changed` 在建窗期（`_cfg_apply`
              里 ✓）就把「用检测器」藏好了 ✓，可 **窗口 `show()` 之后它又被 Qt 显示回来** ✗
              （探针实测：那一刻 `isHidden()=False` ✓，再喊一次 `_on_mode_changed` ⇒ 立刻 True ✓）
              ⇒ 这里补一次 ✓（**幂等** ✓ 不重算 ✓ 不改任何判定 ✓）。
            ⚠ 这两件**与 `_on_mode_changed` 里是同一口径** ✓（`_vel()` ✓）；改动时**两处一起改** ✓。
            """
            super().showEvent(e)
            self._apply_dets_visible()          # ⚠ 这一件**必须**在这里补 ✓（`setVisible` 无效 ✗ 见它 ✓）
            #   ⚠ 日志区**不再被隐藏** ✓（用户 2026-10-07 更正 ✓ 见 `_on_mode_changed` ✓）。

        def closeEvent(self, e):
            # ⭐ 关窗时把"上次的配置"落盘（用户 2026-10-01："让窗口能够记住我上次的配置" ✓）
            self._cfg_save()
            self.timer.stop()
            self.warm.stop()
            if self.worker:
                self.worker.close()
            if self.rec is not None:
                self.rec.release()
                self.rec = None
            super().closeEvent(e)

    app = QApplication(sys.argv)
    # ⭐⭐⭐ **全局字号调大**（用户 2026-10-04 ✓ 原话："**窗口字体能大点嘛要瞎了**" ✓✓）——
    #   ⚠ **一处生效**：所有按钮 / 标签 / 下拉 / 状态栏 / 数字框 / 日志（未自带 `font-size`
    #     的都吃这个 ✓）；⚠ 自己 `setStyleSheet` 写过字号的控件**不受影响** ✗
    #     ⇒ 日志区**单独调**（见 `self.log` ✓）。
    #   ⚠ 取 **12pt**（Qt 默认一般 9pt ✓）⇒ 明显大一圈 ✓；再大布局会挤 ✗（工具条两行很满 ✓）。
    _f = QFont()
    _f.setPointSize(12)
    app.setFont(_f)
    win = DemoWindow()
    win.show()
    return app.exec_()


def run_headless(args):
    # ⭐ 统一入口：`.gif` ⇒ **GIF 动图** ✓（用户 2026-09-30 改 ✓）｜ 其余 ⇒ 视频 ✓
    #   ⚠ `--video` 留空 ⇒ 退回目录里最新的（headless 没有"上次打开"这回事 ✓）。
    frames, ts, _proc, (sx, sy), fps = load_source(args.video or str(DEF_VIDEO))
    is_gif = Path(str(args.video)).suffix.lower() == ".gif"
    if not frames:
        print("[X] 读不到%s：%s" % ("GIF" if is_gif else "视频", args.video))
        return 2
    if args.max_frames:
        frames, ts = frames[:args.max_frames], ts[:args.max_frames]
    dur = ts[-1] if ts else 0.0
    print("%s %s：%d 帧（%.1fs ｜ 平均 %.1f fps）"
          % ("GIF" if is_gif else "视频", Path(str(args.video)).name,
             len(frames), dur, len(frames) / max(1e-3, dur)))
    dets = None
    worker = None
    names = []
    if not args.no_dets:
        t0 = time.time()
        worker = DetsWorker(args.weights or None, conf=args.conf)
        names = DetsWorker.names(args.weights or None)
        dets = [worker.detect(small) for _f, small in frames]
        print("检测预跑完成：%d 帧 / %.1fs（常驻子进程 ✓ 类别名 %s ✓）"
              % (len(dets), time.time() - t0, names))
    runner = Runner(dets=dets)
    rec = None
    if args.record:
        rp = Path(args.record)
        rp.parent.mkdir(parents=True, exist_ok=True)
        _b0 = decode_big(frames[0][0])         # ⚠ 原图是 JPEG ✓（见 `_BIG_JPEG_Q` ✓）
        rec = cv2.VideoWriter(
            str(rp), cv2.VideoWriter_fourcc(*"mp4v"),
            max(0.5, (len(frames) - 1) / dur) if dur > 1e-3 else fps,
            (_b0.shape[1], _b0.shape[0]))
    t = 0.0
    t0 = time.time()
    for i, (blob, small) in enumerate(frames):
        t = ts[i]                     # ⭐ 真实时间戳（GIF 自带每帧延时 ✓）
        o, pos, rad, hit, boxes, motion = runner.step(small, i, t)
        acc = 100.0 * runner.hits / runner.n if runner.n else 0.0
        hud = "t=%5.2fs  %s  hit=%.0f%% (%d/%d)" % (
            t, o["state"], acc, runner.hits, runner.n)
        if rec is not None:
            rec.write(draw(decode_big(blob).copy(), (None, pos, rad, hit, boxes),
                           sx, sy, hud, runner.cursor, motion, names, args.mask))
    if rec is not None:
        rec.release()
    print("命中率 %.1f%%（%d/%d 帧）｜ 真实耗时 %.1fs"
          % (100.0 * runner.hits / max(1, runner.n), runner.hits, runner.n,
             time.time() - t0))
    if worker:
        worker.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="",
                    help="开窗时自动载入的**视频文件**，或 **GIF 动图**"
                         "（如 datasets/liedetectorgifs/test.gif ✓ 用户 2026-09-30 改；"
                         "GIF ⇒ 按它自带的每帧延时当视频放 ✓）。留空 = **上次打开的文件** ✓"
                         "（再没有才退回目录里最新的 ✓）")
    ap.add_argument("--gif", type=str, default="",
                    help="初始 **GIF 动图**（= `--video <x.gif>` 的快捷写法 ✓）")
    ap.add_argument("--record", type=str, default="")
    ap.add_argument("--headless", action="store_true", help="不开窗（自检/录制 ✓）")
    ap.add_argument("--no-dets", action="store_true", help="纯残差（对比 M2a ✓）")
    ap.add_argument("--weights", type=str, default="")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--mask", type=float, default=0.35,
                    help="灰蒙版不透明度 0~1（框外压暗、框里挖洞 ✓ 窗口里也能拖 ✓）")
    args = ap.parse_args()
    if args.gif:
        args.video = args.gif             # `--gif` 是 `--video <x.gif>` 的快捷写法 ✓
    if args.headless:
        return run_headless(args)
    return run_window(args)


if __name__ == "__main__":
    sys.exit(main())
