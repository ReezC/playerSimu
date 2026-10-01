# -*- coding: utf-8 -*-
"""测谎**效果演示窗**（2026-09-30 用户要求 ✓ 原话："做一个窗口程序，可以选择视频文件
播放，播放的时候实时显示效果"；前身是"红点代表鼠标"的定视频演示 ✓）。

窗口能做什么：
  · **选择视频文件**（工具栏「选择视频…」✓ 默认开在 `datasets/liedetectorVideo`）；
  · **播放 / 暂停**（空格也行 ✓）、**速度** 0.25~2×、**循环**；
  · **检测器开关**（关掉 = 纯残差路线 M2a，开着 = 加检测复活通道 M2b ✓ 好对比）；
  · 打开后**先预热检测**（进度显示在状态栏 ✓ 别让人对着黑屏等），然后自动开播 ✓；
  · 画面上：🔴 **大红点（带白圈）= 鼠标**（控制器输出的光标估计 ✓）、🟢 **绿圈 = 我们认为的
    真目标位置**（圆心 = 后端报告位置 ✓）、🟢 **实心绿点 = 追踪器真实位置**（未平滑 ✓）、
    🔴 **小红点 = 绿圈圆心**（只在绿点不在圆心时画 ⇒ 两种可能：**拟人平滑正在生效** ✓ 用户
    2026-09-30 认知 #3/#4 ✓，或**信念修正正在生效**（"并集吞下在册假目标 ⇒ 圆修正到对角" ✓
    用户 2026-10-01 ✓ 见 `LieTracker._merge_far_corner` ✓））、
    **粗红框 = 我们认为真目标被包含在里面**（后端 `tbox` ✓，绿圈**整圈**在它
    里面 ✓）、细蓝框 = 检测器看到的东西、白线 = 还差多少 ✓；命中 ⇒ 鼠标点转绿 ✓；
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
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from perception.lie_controller import LieMouseController  # noqa: E402
from perception.lie_tracker import LieTracker  # noqa: E402
# ⭐ 轨迹预测观察窗 / 融合框判定阈值的**默认值**（界面初值用它 ✓ 与追踪器同一处口径 ✓）
from perception.lie_tracker import _MERGE_RATIO as _MERGE_RATIO_DEFAULT  # noqa: E402
from perception.lie_tracker import _PATH_MS as _PATH_MS_DEFAULT  # noqa: E402

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

    def __init__(self, weights=None, conf=0.25, imgsz=960):
        cmd = [sys.executable, "-m", "tools.lie_dets_worker",
               "--conf", str(conf), "--imgsz", str(imgsz)]
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


def load_video(path):
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
        frames.append((f, small))
        prev_g = g
    cap.release()
    if not frames:
        return [], [], (pw, ph), (1.0, 1.0), fps
    fps_eff = fps * len(frames) / float(max(1, n_all))     # 折算成**内容帧率** ✓
    sx = frames[0][0].shape[1] / float(pw)
    sy = frames[0][0].shape[0] / float(ph)
    ts = [i / max(1.0, fps_eff) for i in range(len(frames))]   # 等间隔 ⇒ 就按帧率算 ✓
    return frames, ts, (pw, ph), (sx, sy), max(1.0, fps_eff)


def load_gif(path):
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
        frames.append((f, small))
        ts.append(t_acc)
        t_acc += dur
        prev_g = g
    if not frames:
        return [], [], (0, 0), (1.0, 1.0), 10.0
    span = ts[-1] if ts else 0.0
    fps = (len(ts) - 1) / span if (len(ts) > 1 and span > 1e-3) else 10.0
    sx = frames[0][0].shape[1] / float(pw)
    sy = frames[0][0].shape[0] / float(ph)
    return frames, ts, (pw, ph), (sx, sy), max(0.5, fps)


def load_source(path):
    """统一入口：**`.gif` ⇒ GIF 动图** ✓（用户 2026-09-30："改为『选取 gif』" ✓）；
    其余 ⇒ **视频文件** ✓。同形返回 `(frames, ts, (proc_w, proc_h), (sx, sy), fps)` ✓。

    ⚠ 原"选图片集（帧目录）"那条**已按用户要求移除** ✗ —— 目录不再是合法输入 ✓
      （传目录会被当视频读 ⇒ 读不到 ⇒ 界面提示"读不到视频" ✓ 不会静默 ✗）。
    """
    p = Path(str(path))
    if p.suffix.lower() == ".gif":
        return load_gif(p)
    return load_video(p)


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


_RED_MATCH_GATE = 25.0    # ⭐ 「追踪器认定的目标框」↔「本帧检测框」的配对门（px ✓ 它就是
                          #   从那批框里选的 ⇒ 基本重合 ✓ 25px 足够容下检测抖动 ✓）
_RAD_LO, _RAD_HI = 0.85, 1.25   # ⭐ **学习"目标实际半径"时只信这个面积带**的框（× 标准
                                #   面积 ✓）：≈1.0× = 单个目标 ✓；跑出去（融合/分离 ✗）
                                #   ⇒ **冻结**不学 ⇒ 透明阶段圆半径恒定 ✓（用户口径 ✓）
_DOT_GAP = 2.0            # ⭐⭐ **绿点（追踪器真实位置 = `raw_pos`）与绿圈圆心（报告位置 = `pos`）
                          #   差多少 px 才算"两者不一致"**（显示域 px ✓ 用户 2026-09-30
                          #   认知 #4 ✓：差超过它就**用红点标出圆心** ✓；重合 ⇒ 只画绿点 ✓）。
                          #   ⚠ 不一致有**两个来源** ✗：① **拟人平滑**（`_ease_pos` ✓ 老来源 ✓）；
                          #     ② **信念修正**（用户 2026-10-01 ✓："并集吞下在册假目标 ⇒ 圆修正到
                          #     对角" ✓ 只改**报告位置**那一层、内部状态不动 ⇒ 绿点必然落后于圆心 ✓
                          #     见 `LieTracker._merge_far_corner` ✓）。
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
    """图例那行"怀疑链"（用户 2026-09-30 ①②③④ ✓）：目标轨迹 id / 怀疑类型 / 不合群秒数。

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
                 merge_ratio=None):
        # ⭐ `path_ms` = **轨迹预测窗口时间**（毫秒 ✓ 演示窗那一行配 ✓）⇒ 透传给追踪器 ✓
        self.path_ms = path_ms
        # ⭐ `merge_ratio` = **融合框判定阈值**（面积比 ✓ 应用按钮左边配 ✓）⇒ 透传 ✓
        self.merge_ratio = merge_ratio
        self.tr = LieTracker(path_ms=path_ms, merge_ratio=merge_ratio)
        self.ctl = LieMouseController(gain=gain, assume=assume)
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
        _trad = out.get("rad")
        if _trad:
            self.tgt_rad = float(_trad)
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
        # ⭐⭐ **本拍红框是不是「融合框」**（用户 2026-10-01 新流程 ✓）也转发给演示窗/自检 ✓：
        #   融合 ⇒ 不做信念夹取、绿圈按白箭头（预测）走 ✓（"边归属相切"整套已移除 ✓）。
        motion["merged"] = bool(out.get("merged"))
        return out, pos, r, hit, raw, motion      # `raw` = 检测原框（带 cls/conf ✓ 画图用 ✓）

    def reset(self, assume=(375.0, 250.0)):
        self.__init__(dets=self.dets, gain=self.ctl.gain, assume=assume,
                      path_ms=self.path_ms, merge_ratio=self.merge_ratio)


def analyze(frames, dets, gain=None, assume=(375.0, 250.0), on_progress=None):
    """**整段一次算完**（用户 2026-09-30 要求 ② ✓ 原话："加载视频的时候就演算完，
    这样我拖帧就不用重算了"）⇒ 返回每帧结果列表，播放/拖动只是**查表** ✓。

    每项：`{"pos","r","hit","state","cursor","boxes"}`（都是**处理域**坐标 ✓ 画时再放大）。
    `on_progress(k, n)` 每帧回调（界面报进度 ✓）。
    """
    r = Runner(dets=dets, gain=gain, assume=assume)
    out = []
    for i, (_big, small) in enumerate(frames):
        o, pos, rad, hit, d, motion = r.step(small, i, i / 60.0)
        out.append({"pos": pos, "r": rad, "hit": hit, "state": o["state"],
                    "cursor": tuple(r.cursor), "boxes": d or [],
                    "motion": motion})
        if on_progress and (i % 20 == 0 or i == len(frames) - 1):
            on_progress(i + 1, len(frames))
    return out


def apply_mask(frame, boxes, alpha, target=None, scale=(1.0, 1.0)):
    """⭐ **灰蒙版 + 框挖洞**（用户 2026-09-30 要求 ① 原话："把 yolo 工作台的背景蒙版 +
    框挖洞那一套用到测谎演示里" ✓ 同款语义：**框外压暗、框里保持原亮度** ✓）。

    `alpha` = 蒙版不透明度（0 = 关 ✓）；洞 = 每个检测框 + 目标圆（要看的东西保持亮 ✓）。
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


def draw(frame, res, scale_x, scale_y, hud=None, cursor=None, motion=None,
         names=None, mask=0.0):
    """🔴 红点（鼠标）/ 🟢 绿圈（目标）/ 框（类别名+置信度）/ 运动向量 ✓。"""
    _out, pos, r, hit, d = res
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
    frame = apply_mask(frame, d or [], mask,
                       target=(pos[0] * scale_x, pos[1] * scale_y, _rad_disp)
                       if pos is not None else None,
                       scale=(scale_x, scale_y))
    if pos is not None:
        c = (int(pos[0] * scale_x), int(pos[1] * scale_y))
        # ⭐⭐⭐ **圆圈恒色**（用户 2026-10-01 定稿 ✓ 原话："圆圈不要有任何的变色与相关逻辑" ✓）
        cv2.circle(frame, c, int(_rad_disp), (0, 255, 0), 2)
        # 用户 2026-10-01 定稿：淡粉框 = 「红框丢失后的预测接力」——
        #   · 红框**在**（后端 `tbox` 有 ✓）⇒ **不画**（检出范围由红框自己表达 ✓，画了反而
        #     看不出"什么时候丢了" ✗）；
        #   · 红框**丢**（`tbox=None` ✓）且有宽高快照 ⇒ 画：尺寸 = 最后一次红框宽高的
        #     **原始值**（不缩到圆内 ✗ 它代表的是检出范围本身 ✓）、中心跟着**报告位置**走
        #     （丢框期间位置本来就是预测外推 ✓）⇒ 一眼看出「检出丢了、这段是预测在接力」✓。
        _tw = (motion or {}).get("tbox_wh")
        if _tb is None and _tw and float(_tw[0]) > 1.0 and float(_tw[1]) > 1.0:
            _pw = float(_tw[0]) / 2.0 * scale_x
            _ph = float(_tw[1]) / 2.0 * scale_y
            cv2.rectangle(frame, (int(c[0] - _pw), int(c[1] - _ph)),
                          (int(c[0] + _pw), int(c[1] + _ph)),
                          (203, 192, 255), 2, cv2.LINE_AA)
    if cursor is not None:
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
        if _tb_key is not None and _key == _tb_key:
            continue                               # 后端那格 ⇒ 交给红框画 ✓（别重复 ✗）
        p1 = (int((cx - w / 2) * scale_x), int((cy - h / 2) * scale_y))
        p2 = (int((cx + w / 2) * scale_x), int((cy + h / 2) * scale_y))
        _role = _roles[_i] if _i < len(_roles) else ""
        _col, _th = (255, 140, 0), 1
        if _role == "suspect":
            _col, _th = (0, 0, 255), 1             # 严查框（细红 ✓ 不当"唯一红框"抢戏 ✗）
        cv2.rectangle(frame, p1, p2, _col, _th)
        nm = names[cls] if (names and 0 <= cls < len(names)) else str(cls)
        txt = "%s %.2f" % (nm, conf)
        (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(p1[1] - 3, th_ + 4)
        cv2.rectangle(frame, (p1[0], ty - th_ - 4), (p1[0] + tw + 4, ty + 2),
                      (40, 90, 160), -1)
        cv2.putText(frame, txt, (p1[0] + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
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
        # ⭐⭐⭐ **"有了就标"**（用户 2026-10-01 定稿 ✓ 原话："**不需要灰框这么复杂的逻辑**：
        #   1.**有了就标**，跟踪他们" ✓）⇒ **一个样式**：**青色框 + `登记 #xxx`** ✓
        #   （不再分"青=假目标 / 灰=还没钉住"两层 ✗ —— 那套太绕 ✓；身份判定交给运动/后续逻辑 ✓）。
        cv2.rectangle(frame, _q1, _q2, (255, 255, 0), 2)
        _bid = _e.get("bid") or (0, 0)
        _txt = "砖(%d,%d)" % (int(_bid[0]), int(_bid[1]))
        (tw, th2), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        _ty = max(_q1[1] - 3, th2 + 4)
        cv2.rectangle(frame, (_q1[0], _ty - th2 - 4), (_q1[0] + tw + 4, _ty + 2),
                      (90, 60, 10), -1)
        cv2.putText(frame, _txt, (_q1[0] + 2, _ty - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 0), 1, cv2.LINE_AA)
        # ⚠ **未转正的候选不画** ✗（2026-09-30 实测：相机一走，候选能涨到上百条 ✗ ⇒ 全画出来
        #   会糊屏 ✓ 而且它们此刻**还不算"登记过的"** ✗ 画了反而误导 ✓）。只看实线的青框 ✓。
    # ---- 🔴 **唯一红框 = 后端那个框**（画在检测框之上 ✓ 用户："必须与后端完全一致" ✓）----
    if _tb is not None:
        _p1 = (int((float(_tb[0]) - float(_tb[2]) / 2.0) * scale_x),
               int((float(_tb[1]) - float(_tb[3]) / 2.0) * scale_y))
        _p2 = (int((float(_tb[0]) + float(_tb[2]) / 2.0) * scale_x),
               int((float(_tb[1]) + float(_tb[3]) / 2.0) * scale_y))
        cv2.rectangle(frame, _p1, _p2, (0, 0, 255), 3)
        # ⭐⭐ **融合红框 ⇒ 右上角标「融合」**（用户 2026-10-01 ✓ 原话："如果红框处于融合状态，
        #   就标在框右上角吧" ✓）：一眼看出这格红框 = 真假目标的并集 ✓（与左上角的青框「登记」
        #   分开 ✓ 不打架 ✓）。⚠ 只认后端给的 `merged`（`motion["merged"]` ✓ 同一口径 ✓）。
        if (motion or {}).get("merged"):
            _txt = "融合"
            (tw, th2), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            _bx0 = _p2[0] - tw - 4
            _by0 = _p1[1]
            _by1 = _by0 + th2 + 4
            cv2.rectangle(frame, (_bx0, _by0), (_p2[0], _by1), (0, 0, 190), -1)
            cv2.putText(frame, _txt, (_bx0 + 2, _by0 + th2 + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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
        for i, b in enumerate(d or []):
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
        for t in (motion.get("tracks") or []):
            if not t.get("sticky"):
                continue
            vx, vy = t.get("v") or (0.0, 0.0)
            if (not (vx or vy)) or _dt <= 0:
                continue
            _x, _y = t["p"][0] * scale_x, t["p"][1] * scale_y
            _arrow(_x, _y, vx * _dt * scale_x, vy * _dt * scale_y, (150, 150, 150), 1)
        # ⭐⭐ **纳入预测的轨迹（白线 ✓）**（用户 2026-10-01 ✓ 原话："用白色的线把纳入预测的轨迹
        #   画出来" ✓）：`_path_pred` 拟合的那条群体相对曲线（后端已换算到加工域 ✓）⇒ 连成白线 ✓
        #   一眼看出预测在跟哪条弧（转弯时能看到曲率 ✓）。画在白箭头**下面**（细线 1px ✗ 别抢
        #   白箭头 2px 的戏 ✓）。
        _pp = (motion or {}).get("path_pts") or []
        if len(_pp) >= 2:
            _pts = [(int(float(p[0]) * scale_x), int(float(p[1]) * scale_y)) for p in _pp]
            cv2.polylines(frame, [np.array(_pts, np.int32)], False, (255, 255, 255), 1,
                          cv2.LINE_AA)
            cv2.circle(frame, _pts[-1], 3, (255, 255, 255), 1, cv2.LINE_AA)   # 端点空心点 ✓
        # ⭐⭐⭐ **黄箭头 = 圆圈的绝对速度 ／ 白箭头 = 相对假目标群的速度**（用户 2026-10-01 定稿 ✓
        #   原话："用**黄色箭头**标圆圈的**绝对速度**，用**白色箭头**标圆圈**相对于假目标群
        #   的速度预测**" ✓）。
        #   · 起点都是**绿圈圆心**（= 报告位置 `pos` ✓ 与绿圈同一个点 ✓）；
        #   · 长度口径沿用 A2/2026-09-30 那条 ✓：**长度 = 速度 × dt**（= 按当前速度**预计
        #     下一帧到哪** ✓，与画面里其它速度箭头同一口径 ✓）；
        #   · **黄** = `vel_abs`（自身 + 群体 ✓ 就是对外报的绝对速度 ✓）；
        #   · **白** = `vel_rel`（**相对假目标群** ✓ 目标自己那一份运动 ✓）——
        #     用户要对比的就是这两支：**箭头差 = 群体在怎么走** ✓。
        #   ⚠ 原来的白箭头（`track_v` = **检测轨迹**速度 ✗）**撤掉了** ✗：白箭头现在归
        #     "相对速度" ✓（同一时刻两支白箭头只会打架 ✗）。
        _dt_a = float(motion.get("dt") or 0.0)
        if pos is not None and _dt_a > 0:
            _x, _y = pos[0] * scale_x, pos[1] * scale_y
            _va = motion.get("vel_abs") or (0.0, 0.0)
            _vr = motion.get("vel_rel") or (0.0, 0.0)
            # ⚠ 长度下限：速度≈0 时 `_arrow` 只画一个小点 ✓（不画假方向 ✗）
            _arrow(_x, _y, _va[0] * _dt_a * scale_x, _va[1] * _dt_a * scale_y,
                   (255, 0, 0), 2)                      # 蓝: 下帧绝对位移预测                      # 🟡 黄：绝对速度 ✓
            _arrow(_x, _y, _vr[0] * _dt_a * scale_x, _vr[1] * _dt_a * scale_y,
                   (255, 255, 255), 2)                    # ⚪ 白：相对假目标群 ✓
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
        y0 = frame.shape[0] - 16 * len(lines) - 8
        for i, s in enumerate(lines):
            cv2.putText(frame, s, (10, y0 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(frame, s, (10, y0 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (235, 235, 235), 1)
    if hud:
        cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4)
        cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
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
    from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPixmap
    from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
                                 QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                                 QPushButton, QShortcut, QSlider, QToolBar,
                                 QVBoxLayout, QWidget)
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
            p.end()

        def mouseMoveEvent(self, ev):
            self._mouse = QPointF(ev.pos())
            self._mouse_img = self._img_at(self._mouse)
            self.update()
            super().mouseMoveEvent(ev)

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
            self.names = None                  # 类别名（画框标签用 ✓ 第一次演算时取 ✓）
            self.i = 0
            self.t = 0.0
            self.rec = None
            self.playing = False
            self.warm_n = 0
            self.warm_total = 0
            self._reuse_dets = False       # 「应用」重算时 = True（复用检测框 ✓）

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
            self.cmb_speed.currentIndexChanged.connect(self.retime)
            tb.addWidget(self.cmb_speed)
            self.chk_dets = QCheckBox("用检测器")
            self.chk_dets.setChecked(not args.no_dets)

            self.chk_dets.setToolTip("关掉 = 纯残差路线（M2a）；开着 = 加检测复活"
                                     "通道（M2b）✓ 好对比 ✗ 切换会重跑预热 ✓")
            self.chk_dets.stateChanged.connect(self.reload_current)
            tb.addWidget(self.chk_dets)
            self.chk_loop = QCheckBox("循环")
            self.chk_loop.setChecked(True)
            tb.addWidget(self.chk_loop)
            # ---- ⭐ 两个拖动条（用户 2026-09-30 要求 ①③）：蒙版透明度 / 箭头拖尾 ----
            tb.addWidget(QLabel("  蒙版 "))
            self.sld_mask = QSlider(Qt.Horizontal)
            self.sld_mask.setRange(0, 80)
            self.sld_mask.setValue(int(args.mask * 100))
            self.sld_mask.setFixedWidth(110)
            self.sld_mask.setToolTip("灰蒙版不透明度（框外压暗、**框里挖洞保持原亮度** ✓）")
            self.sld_mask.valueChanged.connect(self.on_view_change)
            tb.addWidget(self.sld_mask)
            self.lbl_mask = QLabel("%d%%" % self.sld_mask.value())
            tb.addWidget(self.lbl_mask)
            # ---- ⭐ **记住上次的配置**（用户 2026-10-01 ✓ 原话："让窗口能够记住我上次的
            #   配置" ✓）：这些开关/拖动条**离开窗口时**写回 `config/ui.yaml` 的 `lie_demo`
            #   段（按客户端唯一一份 ✓）；「应用」按钮那条在 `apply_path_ms` 里写 ✓。
            #   ⚠ 拖动条用 `sliderReleased`（不是 `valueChanged` ✗ —— 拖一下会写几十次文件 ✗）。
            self.cmb_speed.currentIndexChanged.connect(self._cfg_save)
            self.chk_dets.stateChanged.connect(self._cfg_save)
            self.chk_loop.stateChanged.connect(self._cfg_save)
            self.sld_mask.sliderReleased.connect(self._cfg_save)
            # ⚠ **"箭头长度"拖动条已删**（用户 2026-09-30 第 A2 条 ✓ 原话："取消自定义
            #   箭头长度拖动条配置，速度向量箭头现在**强制指向以当前速度预计下一帧到达的
            #   位置**" ✓）—— 箭头长度就是**这一拍的位移**（所见即所测 ✓），没什么可调的 ✓。

            # ---- ⭐ **配置行：轨迹预测窗口时间**（用户 2026-10-01 ✓ 原话："增加参数
            #   「轨迹预测窗口时间」（就是你刚说的这 500ms），在输入框右边加按钮「应用」，
            #   点击后重新运算并生效" ✓）----
            #   它 = `LieTracker.path_ms`：**切向预测回看多久的轨迹**（默认 500ms ✓
            #   10fps ⇒ 5 个采样点 ✓ 少于 3 个拟不出曲率 ✗、多于 ~7 个会把转弯前的旧
            #   方向带进来 ✗）⇒ 调大 = 更平滑更滞后、调小 = 更灵敏更抖 ✓。
            #   ⚠ 位置：**交互区（工具栏）下面、画面上面** ✓（交互区下加一行 ✓）。
            cfgrow = QHBoxLayout()
            cfgrow.setContentsMargins(6, 2, 6, 2)
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
            # ---- ⭐⭐ **融合框判定阈值**（用户 2026-10-01 ✓ 原话："判定检出框面积 > 登记的
            #   假目标面积一定比例（**应用按钮左边加配置「融合框判定阈值」**）则判定红框融合" ✓）
            #   —— 面积比（检出框面积 ÷ 登记的假目标面积 ✓）：≥ 它 ⇒ 判红框**融合**
            #   （= 目标 + 假目标的并集 ✓）⇒ **不做信念夹取**、绿圈按白箭头（预测）走 ✓。
            #   调小 ⇒ 更容易判成融合（夹取少、更信预测 ✓）；调大 ⇒ 更少融合（框更常被当约束 ✓）。
            cfgrow.addWidget(QLabel("融合框判定阈值"))
            self.sp_merge = NoWheelDoubleSpinBox()
            self.sp_merge.setRange(1.0, 5.0)
            # ⭐⭐ **精度到小数点后 2 位**（用户 2026-10-01 ✓ 原话："融合框判定阈值需要往后再精确
            #   1 位小数" ✓）：实测判决常常"卡线"（`9月30日(1).mp4` 帧 16 的面积比是 **1.22** ⇒
            #   阈值 1.2 判融合 / 1.4 不判 ✓）⇒ 1 位小数根本调不到那个点上 ✗ ⇒ 改 2 位 ✓；
            #   箭头步长取 0.05（一键一格好按 ✓ 要 1.22 这种值直接手输即可 ✓）。
            self.sp_merge.setSingleStep(0.05)
            self.sp_merge.setDecimals(2)
            self.sp_merge.setValue(float(_MERGE_RATIO_DEFAULT))
            self.sp_merge.setToolTip(
                "检出框面积 ÷ 登记的假目标面积 ≥ 它 ⇒ 判「红框融合」（框 = 目标 + 假目标的并集）。\n"
                "融合时**不做信念夹取**，绿圈继续按白箭头（预测）走 ✓；\n"
                "**没标融合的检出框一律不给红框**（那一拍走预测态 ✓）。\n"
                "精度 0.01（可直接手输，例如 1.22 ✓）；改完点「应用」⇒ 重新演算整段并生效。")
            cfgrow.addWidget(self.sp_merge)
            self.btn_apply = QPushButton("应用")
            self.btn_apply.setToolTip("按当前「轨迹预测窗口时间」**重新演算**整段并生效 ✓")
            self.btn_apply.clicked.connect(self.apply_path_ms)
            self.btn_apply.setEnabled(False)          # 载入素材前点了没意义 ✓
            cfgrow.addWidget(self.btn_apply)
            cfgrow.addStretch(1)

            self.lbl = _ImageView()
            self.lbl.setText("点左上角「选择视频…」放一段录像，或「选择 GIF…」"
                             "选一个动图（如 datasets/liedetectorgifs/test.gif）✓")
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
            lay.addLayout(cfgrow)          # ⭐ 配置行在**画面上方**（交互区下面 ✓）
            lay.addWidget(self.lbl, 1)
            lay.addLayout(prow)
            self.setCentralWidget(c)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.warm = QTimer(self)
            self.warm.timeout.connect(self.warm_tick)

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
            self._steal_on = set()
            for _w in (getattr(self, "sp_path_ms", None), getattr(self, "sp_merge", None)):
                if _w is not None:
                    _w.installEventFilter(self)
                    self._steal_on.add(_w)

            # ---- ⭐ **回填上次的配置**（用户 2026-10-01 ✓）----
            #   ⚠ 必须在 `load()` **之前**（「用检测器」决定走哪条演算路 ✓、
            #     「轨迹预测窗口时间」决定 Runner 的切向窗 ✓）。
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
            start = getattr(self, "_gif_root", "") or (
                str(GIF_DIR) if GIF_DIR.is_dir() else str(ROOT))
            f, _ = QFileDialog.getOpenFileName(self, "选一个 GIF 动图", start, GIF_EXT)
            if f:
                self._gif_root = str(Path(f).parent)
                self.load(f)

        def reload_current(self):
            if getattr(self, "_cur", ""):
                self.load(self._cur)

        def apply_path_ms(self):
            """⭐ **「应用」**：把「轨迹预测窗口时间」写进追踪器并**重新演算整段**（用户
            2026-10-01 ✓ 原话："点击后重新运算并生效" ✓）。

            ⚠ 为什么**不重跑检测**：检测框（`self.dets`）与这个参数**无关** ⇒ 复用它
            （`reuse=True` ✓）⇒ 只重跑"追踪 + 控制"那一段 ✓（相差一个数量级的耗时 ✓）。
            ⚠ 必须**整段重算** ✗ 不能只改参数接着播：切向拟合要用**新窗长回看历史** ⇒
            半路改窗长会让前后半段的轨迹不可比 ✓（播放/拖动查的还是那张表 ✓）。
            """
            if not self.frames:
                return
            self.stop()
            ms = int(self.sp_path_ms.value())
            mr = float(self.sp_merge.value())
            # ⚠ 复用前先确认**检测框是齐的**（上次演算被打断 ⇒ 缺帧 ✗）：没齐就老实重检测 ✓
            #   （没勾「用检测器」⇒ 本来就不需要框 ⇒ 也走复用 ✓ 省一个检测子进程 ✓）。
            reuse = ((not self.chk_dets.isChecked())
                     or (bool(self.dets) and len(self.dets) >= len(self.frames)))
            self.lbl.setText("应用「轨迹预测窗口时间 = %d ms ／ 融合框判定阈值 = %.1f」"
                             "⇒ 重新演算 %d 帧…（%s）"
                             % (ms, mr, len(self.frames),
                                "检测框复用 ✓" if reuse else "重跑检测"))
            self.start_warmup(reuse=reuse)
            self._cfg_save()          # ⭐ 点「应用」= 这个值就是"我要的配置" ⇒ 落盘 ✓

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
            except Exception:
                pass
            # ①' 融合框判定阈值
            try:
                self.sp_merge.setValue(min(max(float(c.get("merge_ratio",
                                                           _MERGE_RATIO_DEFAULT)), 1.0), 5.0))
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
            """把当前配置写回 `config/ui.yaml` 的 `lie_demo` 段（开关/拖动条/「应用」都调 ✓）。

            ⚠ 写不进去（没权限/文件被占）**不许影响演示** ✓ —— 存偏好是附带的 ✓。
            """
            try:
                from gui import theme
                theme.save_section("lie_demo", {
                    "path_ms": int(self.sp_path_ms.value()),
                    "merge_ratio": round(float(self.sp_merge.value()), 3),
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

        def _merge_ratio(self):
            """当前界面上的「融合框判定阈值」（面积比 ✓ 没建控件时 = 默认值 ✓）。"""
            return float(getattr(self, "sp_merge", None).value()
                         if getattr(self, "sp_merge", None) is not None
                         else _MERGE_RATIO_DEFAULT)

        def load(self, path):
            self.stop()
            self._cur = path
            if self.worker:
                self.worker.close()
                self.worker = None
            # ⭐ 统一入口：**`.gif` ⇒ GIF** ✓ / 其余 ⇒ 视频 ✓（两条路同形返回 ✓）
            self.frames, self.ts, self.proc, self.scale, self.fps = \
                load_source(path)
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
            self.btn_apply.setEnabled(True)      # ⭐ 有素材了 ⇒「应用」才可用 ✓
            self.btn_play.setText("播放")
            if args.record:
                rp = Path(args.record)
                rp.parent.mkdir(parents=True, exist_ok=True)
                # 录像的 fps 用**实际平均帧率**（GIF 的时间戳不等间隔 ⇒ 写 `fps` 会偏 ✓）
                rec_fps = (max(0.5, (len(self.frames) - 1) / self.dur)
                           if self.dur > 1e-3 else self.fps)
                self.rec = cv2.VideoWriter(
                    str(rp), cv2.VideoWriter_fourcc(*"mp4v"), rec_fps,
                    (self.frames[0][0].shape[1], self.frames[0][0].shape[0]))
            self.show_frame(self.frames[0][0])
            self.start_warmup()                # 一律先演算（无检测也是一种演算 ✓）
            # ⭐ **载入成功就记下这个文件**（用户 2026-10-01 ✓ 原话："让窗口记住上次打开的文件" ✓）：
            #   不等到 `closeEvent` ✗（关窗才落盘 ⇒ 崩溃/强杀就丢了 ✓）。
            self._cfg_save()

        def start_warmup(self, reuse=False):
            """`reuse=True` ⇒ **复用已算好的检测框**（只重跑追踪 ✓ 「应用」走这条 ✓）。

            ⚠ 只有"检测框与本次参数无关"时才可复用 ✓ —— 「轨迹预测窗口时间」只影响
            切向拟合 ⇒ 可复用 ✓；而「用检测器」开关会**改变有没有框** ⇒ 必须重检测 ✗。
            """
            self._reuse_dets = bool(reuse)
            if not self._reuse_dets:
                self.worker = DetsWorker(args.weights or None, conf=args.conf)
                self.dets = []
            self.results = []
            self.runner = Runner(dets=None, path_ms=self._path_ms(),
                                 merge_ratio=self._merge_ratio())
            self.warm_n = 0
            self.warm_total = len(self.frames)
            self.warm.start(10)

        def warm_tick(self):
            """⭐ **边检测边演算**（整段一次算完 ✓ 用户要求 ②）：检测 + 追踪 + 控制
            都在这里跑完 ⇒ 播放和拖动只是查表 ✓ 不重算 ✓。"""
            for _ in range(4):                 # 每拍 4 帧（界面不卡 ✓）
                if self.warm_n >= self.warm_total:
                    break
                i = self.warm_n
                d = None
                # ⭐ 复用模式（「应用」✓）：检测框**已经算好了** ⇒ 不再问检测器 ✓
                if not getattr(self, "_reuse_dets", False) and self.chk_dets.isChecked():
                    d = self.worker.detect(self.frames[i][1])
                    self.dets.append(d)
                self.runner.dets = self.dets if self.chk_dets.isChecked() else None
                # ⭐ 用**真实时间戳**（GIF 自带每帧延时 ✓ 可以不等间隔 ✓）
                o, pos, rad, hit, boxes, motion = self.runner.step(
                    self.frames[i][1], i, self.ts[i])
                self.results.append({"pos": pos, "r": rad, "hit": hit,
                                     "state": o["state"],
                                     "cursor": tuple(self.runner.cursor),
                                     "boxes": boxes or [], "motion": motion,
                                     # ⚠ 变量名是 `o`（追踪器输出 ✓）不是 `out` ✗
                                     #   （我上一版写成 `out` ⇒ 一开窗 NameError ⇒ 闪退 ✗）
                                     "white": o.get("white")})
                self.warm_n += 1
            acc = 100.0 * self.runner.hits / self.runner.n if self.runner.n else 0.0
            self.lbl.setText("演算中 %d / %d 帧…（检测 + 追踪 + 控制**一次算完** ⇒ "
                             "拖帧零等待 ✓）"
                             % (self.warm_n, self.warm_total))
            if self.names is None and self.chk_dets.isChecked():
                # 类别名只在**第一次**问（~2-3s ✓ 标签已显示"演算中"作为提示 ✓）
                self.names = DetsWorker.names(args.weights or None) or []
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
            self.retime()

        def dt_here(self, i=None):
            """第 `i` 帧 ⇒ 下一帧的**真实间隔**（秒 ✓ 末帧/异常兜底 1/fps ✓）。"""
            i = self.i if i is None else i
            if self.ts and i + 1 < len(self.ts):
                return max(1e-3, self.ts[i + 1] - self.ts[i])
            return 1.0 / max(1.0, self.fps)

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
            ms = 1000.0 * self.dt_here(self.i) / max(0.05, v)
            self.timer.start(max(1, int(round(ms))))

        def toggle_play(self):
            if self.playing:
                self.stop()
            elif self.results:
                self.playing = True
                self.btn_play.setText("暂停")
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
            big = self.frames[self.i][0]
            n = sum(1 for x in self.results[:self.i + 1] if x["pos"] is not None)
            h = sum(1 for x in self.results[:self.i + 1] if x["hit"])
            hud = "t=%5.2fs  %s  hit=%.0f%%" % (
                self.t, r["state"], 100.0 * h / n if n else 0.0)
            mk = self.view()
            self.show_frame(draw(big.copy(),
                                 (None, r["pos"], r["r"], r["hit"], r["boxes"]),
                                 self.scale[0], self.scale[1], hud, r["cursor"],
                                 r.get("motion"), self.names, mk))
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
            #   截图也能直接对 ✓）。算法 = 每个检出 → 最近「已上板」条目的距离，取中位 ✓；
            #   0.0 = 全部严丝合缝 ✓。
            _mo = r.get("motion") or {}
            _reg = [e for e in (_mo.get("reg") or []) if e.get("ok")]
            #   ⚠⚠ 口径必须是「**已上板条目 → 最近检出**」✗✗ —— 反过来（检出 → 最近条目）会
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
            self.statusBar().showMessage(
                "帧 %d/%d ｜ %s ｜ 命中率 %.1f%%（%d/%d）｜ 登记 %d 条 ｜ 错位中位 %s px"
                % (i + 1, len(self.frames), r["state"], acc, h, n, len(_reg), _mis))

        def on_seek_done(self):
            if getattr(self, "_was_playing", False) and self.results:
                self.playing = True
                self.btn_play.setText("暂停")
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
                else:
                    self.stop()
                    return
            big, _small = self.frames[self.i]
            r = self.results[self.i]
            n = sum(1 for x in self.results[:self.i + 1]
                    if x["pos"] is not None)
            h = sum(1 for x in self.results[:self.i + 1] if x["hit"])
            acc = 100.0 * h / n if n else 0.0
            hud = "t=%5.2fs  %s  hit=%.0f%%" % (self.t, r["state"], acc)
            mk = self.view()
            vis = draw(big.copy(), (None, r["pos"], r["r"], r["hit"], r["boxes"]),
                       self.scale[0], self.scale[1], hud, r["cursor"],
                       r.get("motion"), self.names, mk)
            if self.rec is not None:
                self.rec.write(vis)
            self.show_frame(vis)
            self._status(self.i)                 # 状态栏（与拖动**共用**同一份 ✓）
            self.retime()                        # ⭐ 这一帧该停多久 = 它的**真实**帧间隔 ✓
            self.i += 1
            self.t = self.ts[self.i] if self.i < len(self.ts) else self.dur
            if not self.sld.isSliderDown():      # 拖动中不抢滑块 ✗
                self.sld.blockSignals(True)
                self.sld.setValue(min(self.i, len(self.frames) - 1))
                self.sld.blockSignals(False)
            self.upd_time()

        def show_frame(self, bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            # ⭐ 交给 `_ImageView` 画（自己管缩放/适应/坐标 ✓ 原 setPixmap 的"塞满窗口"由
            #   `_fit` 模式承担 ✓）—— 传原图→加工域的缩放，坐标显示才和红框/登记同一把尺 ✓。
            self.lbl.set_image(qi, self.scale[0], self.scale[1])

        def eventFilter(self, obj, ev):
            """**把 ←→ / 空格 从数字框手里抢回来**（用户 2026-10-01 ✓ "不管聚焦在哪" ✓）。

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

        def closeEvent(self, e):
            # ⭐ 关窗时把「上次的配置」落盘（用户 2026-10-01："让窗口能够记住我上次的配置" ✓）
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
        rec = cv2.VideoWriter(
            str(rp), cv2.VideoWriter_fourcc(*"mp4v"),
            max(0.5, (len(frames) - 1) / dur) if dur > 1e-3 else fps,
            (frames[0][0].shape[1], frames[0][0].shape[0]))
    t = 0.0
    t0 = time.time()
    for i, (big, small) in enumerate(frames):
        t = ts[i]                     # ⭐ 真实时间戳（GIF 自带每帧延时 ✓）
        o, pos, rad, hit, boxes, motion = runner.step(small, i, t)
        acc = 100.0 * runner.hits / runner.n if runner.n else 0.0
        hud = "t=%5.2fs  %s  hit=%.0f%% (%d/%d)" % (
            t, o["state"], acc, runner.hits, runner.n)
        if rec is not None:
            rec.write(draw(big.copy(), (None, pos, rad, hit, boxes), sx, sy, hud,
                           runner.cursor, motion, names, args.mask))
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
