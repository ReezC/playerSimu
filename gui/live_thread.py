"""实时收流 + 推理的后台线程。

**为什么必须独立线程**
    收流（等帧）和推理（等 GPU）都是阻塞操作。放主线程里窗口会立刻失去响应。

**为什么显示要限流，而且统计要拆成三个数**
    这是一条流水线，三个环节速度不同：

        收流 60fps  →  推理 8ms/帧（125fps 上限）  →  界面显示 30fps 就够看

    如果每帧都推给界面、还要等它画完，显示就成了瓶颈 ——
    推理被迫降到 30fps，测出来的帧率是假的，等于把最该看的数字掩盖了。

    所以：**推理照常跑，显示按自己的节奏抽帧**。界面跟不上就少显示几帧，
    但绝不能让界面拖慢推理。

    同时把三个速度分开上报（收流 fps / 推理 ms / 显示 fps）——
    只给一个 "fps" 的话，一出问题根本分不清是网络掉帧、模型太慢，
    还是界面拖累。
"""

import threading
import time

from PyQt5.QtCore import QThread, pyqtSignal

from core import behavior                  # 「查怪结果」打点（`mob_fh` ✓ 见 `_make_mob_sets_resolver`）
from tools.config import get, load_live

# 探针解码：**必须模块级**。原来它是在 `_run()` 里局部 import 的，那会让
# `probe_codec` 变成 `_run` 的**局部名** —— 函数里任何在那一行之前读它的地方都会
# UnboundLocalError（实测踩过：新建跨帧判据 `probe_codec.Verdict()` 放在计数区，
# 早于局部 import 一百多行，一开预览就"失败：UnboundLocalError"）。
# 仍然保留"不可用就降级"的语义：导入失败时置 None，`_run` 里按 None 判断关掉探针。
try:
    from tools import probe_codec
    from tools.probe_codec import decode_ms, resolve_delay_ms
except Exception:                     # pragma: no cover - 正常环境不会走到
    probe_codec = None
    decode_ms = resolve_delay_ms = None

from gui import theme
from perception.pos_state import PositionStateMachine

#: ⭐ 给**怪**挑 foothold 时，"面可以比它算出来的脚底**高**多少"（世界像素 ✓）——
#: 就是 `mapdata.foothold_below(..., above_tol=…)` 那个参数 ✓。用它的理由见
#: `_make_mob_sets_resolver` 的 `_resolve` 里那段（用户 2026-09-28 报
#: "**查#171 一个二楼的怪显示在底层**，现在已经卡住了" ✗）。
#: 取 200：够盖住"**大怪检测框没包住脚**"（实测差 84 ✓），又不至于把隔一整层的面拉进来
#: —— 而且比较**仍然是"|Δy| 最小"** ✓ ⇒ 怪真站在哪一层，那一层照样赢 ✓（不受影响 ✓）。
MOB_ABOVE_TOL = 200.0

#: 「朝『玩家 → 怪』方向找最近集合」那条降级路径里，**竖直距离小于它就不动**（世界像素 ✓）。
#: 理由：那说明怪和玩家**在同一高度** ⇒ 该由水平追击（`chase`）解决 ✓，上下跑是错的 ✗
#: （也是防抖：同一层的怪 y 差本来就只有几像素 ✓）。见 `_make_toward_set_resolver` ✓。
TOWARD_MIN_DY = 24.0

#: 「怪在哪层」那条**方向自洽校验**的最小可比距离（**画面像素** ✓）——两个框**中心**的 y 之差
#: 小于它就**不校验** ✓。理由：同一层的怪中心本来只差几像素，硬判符号只会误杀 ✓。
#: ⚠ **它只是下限**（2026-09-28 用户质疑后改 ✓）：实际阈值取
#: `max(MOB_DIR_MIN_PX, 怪框高 × 0.5)` —— 一个检测框的**固有不确定度**就是它自己半高的量级
#: （上下沿各可能差半个框 ✓）⇒ 让**框自己**决定"差多少才算明显跨层" ✓ 别拍脑袋定常数 ✗。
#: 见 `_make_mob_sets_resolver` 里那段（那里原来拿"框底"当参照 ⇒ 符号会乱 ⇒ 误杀 ✓）。
MOB_DIR_MIN_PX = 8.0

#: ⭐ **相机 y 的 EMA 平滑系数**（2026-09-28 加，治"怪的世界坐标在层之间乱跳" ✓ —— 用户报
#: "**#159 没找到，理由是什么？**"✓）。每条 `mob_fh` 记一次平滑 ✓（调用频率约 1~2 Hz ⇒
#: `0.25` 的平滑窗 ≈ 4 拍 ✓）。⚠ 不要调到 1.0（= 不平滑 ⇒ 抖动照旧 ✓）、也别太小（滞后太大 ✓）。
#: 详见 `_make_mob_sets_resolver` 的 `_resolve` 里那段说明（根因是**黄点与玩家框不同步** ✓）。
CAM_SMOOTH_A = 0.25
from perception.classes import (CLASS_MOB, CLASS_OTHER_PLAYER, CLASS_PLAYER,
                                ZH_NAMES as CLASS_NAMES)

# player_id → YOLO class id 映射（玩家也走 YOLO，每个玩家独占一个类）。
# 当前单玩家固定 class 0；未来多玩家时扩展成 {"角色A": 0, "角色B": 4, ...}
# （4 是「其他玩家」类，见 perception/classes.py —— 别再拿 2，那是 drop）。
PLAYER_CLASS_MAP = {"": CLASS_PLAYER}

#: 积压锁死的判据：端到端延迟中位超过 LAG_WARN_MS 且持续 LAG_WARN_SEC 秒不回落
#: ⇒ 认定"自己回不来了"，面板直说。回落到 LAG_RECOVER_MS 以下才撤销（迟滞，防闪）。
#:
#: **为什么判据不是 `recv_fps < 推流速率`**（那是更"直接"的说法，但会天天误报）：
#: `recv_fps` 是**读线程吞吐**这个间接量 —— 实测 2026-09-26：塌陷后 recv=101、A 机推
#: 117；可**健康段的 recv 也是 117~121**，而容器报的名义帧率是 **144**（A 实际发不满）。
#: 拿 recv 去和名义帧率比，健康时也判成"积压" ✗。真正定义这个问题的是"**延迟稳在高位、
#: 自己回不来**"，那正是探针能直接量到的东西（perf.log 的 e2e_probe_ms）。
LAG_WARN_MS = 500.0
LAG_WARN_SEC = 10.0
LAG_RECOVER_MS = 200.0


def _lag_text(med_ms, secs, peak_ms):
    return ("延迟积压锁死：%d ms 已持续 %.0f 秒（峰值 %d ms），不会自己回落"
            % (med_ms, secs, peak_ms or med_ms))


def lag_watchdog(med_ms, now, state):
    """积压锁死看门狗（**纯函数**）→ (告警短句, 本次是否刚刚判定)。

    `med_ms`：当前端到端延迟中位（拿不到给 None，此时**不报** —— 探针没开就别乱说）；
    `now`：`time.perf_counter()`；`state`：调用方持有的可变列表 `[起算时刻, 峰值, 已报]`。

    返回空串 = 现在没事。短句只写"是什么 + 多久"，**怎么办放在明细里**（状态行那一行
    已经塞满了，硬挤进去反而看不清数字 —— 和 load_warn/load_detail 同一套做法）。

    三种区间（迟滞是真需要的，不是保险）：
      · `≤ LAG_RECOVER_MS(200)`  → **真的回来了**：整段状态清掉，下次重新计时；
      · `≥ LAG_WARN_MS(500)`     → 计时；够 LAG_WARN_SEC 秒 ⇒ 判定"锁死"并一直挂着；
      · 中间（200~500）          → **已判定过就继续挂着**（不闪）；没判定过就把计时清掉
        —— 在 400ms 上耗满 10 秒不算"积压锁死"（那只是有点高），判据要守住边界。
    """
    if med_ms is None:
        return "", False
    if med_ms <= LAG_RECOVER_MS:
        state[0], state[1], state[2] = None, None, False
        return "", False
    if med_ms < LAG_WARN_MS and not state[2]:
        state[0], state[1], state[2] = None, None, False
        return "", False
    if state[0] is None:
        state[0] = now
    state[1] = max(state[1] or 0.0, med_ms)
    if not state[2]:
        if now - state[0] < LAG_WARN_SEC:
            return "", False
        state[2] = True                       # 判定："自己回不来"这句话从此挂着
        return _lag_text(med_ms, now - state[0], state[1]), True
    return _lag_text(med_ms, now - state[0], state[1]), False


#: 判"这一段卡在谁身上"的余量系数：本机一拍花的时间还不到**输入帧间隔**的这个比例，
#: 就说明本机有富余 —— 帧少是上游给的少（A 机 / 推流），不是我们算不动。
LIMIT_SLACK = 0.5


def limit_reason(recv_fps, proc_fps, infer_ms, gap_ms):
    """这一段的吞吐卡在谁身上 → `"input"` / `"self"` / `""`（**纯函数**，便于自检）。

    **为什么需要它**（2026-09-26 查性能时发现的坑）：状态行上「输入 fps」和「处理 fps」
    是两个计数器（读线程**收到**的 / 主回路**取走**的），可只要没怎么丢帧，它们就
    **必然几乎相等**（差的就是那几个丢帧）—— 于是"处理速度掉到 20"看着像 B 机算不动，
    实际常常是 A 机那边只推了 20。只看这两个数会把人引去查错机器。

    实测佐证（perf.log 最慢那段）：`gap_ms` 中位 29.6ms（≈34fps）而 `infer_ms` 才 24ms，
    本机 30fps 都还有余量 ⇒ 是输入先降下来的。所以判据用**输入间隔**和**本机耗时**比，
    而不是用那两个必然相等的 fps。

    三种结论：
      · `"self"`  收到了却没处理完（proc 明显小于 recv，丢帧在涨）⇒ 本机跟不上；
      · `"input"` 本机一拍只花 infer_ms，却隔 gap_ms 才等到一帧 ⇒ 上游给的少；
      · `""`      说不清（两个都慢、或压根还没数）—— **宁可不指方向，也别指错**。
    """
    if not recv_fps or not proc_fps:
        return ""
    if proc_fps < recv_fps * 0.9:
        return "self"
    if infer_ms and gap_ms and infer_ms < gap_ms * LIMIT_SLACK:
        return "input"
    return ""


#: 本机负载清点的间隔（秒）。
#: 为什么要有：训练/并行标注与实时预览同时跑的时候，`infer_ms` 会**与输入尺寸
#: 无关地**整体变慢（同一份配置实测 10.0 → 23.7 ms），而 GPU 时钟、功耗都正常。
#: 状态行必须能当场把这件事说出来，否则又变成「感觉今天特别卡」的玄学。详见
#: core/machineload.py。
LOAD_WATCH_SEC = 10.0

# 类别 → 框颜色（BGR）。**每个类别的颜色都能在设置里改**（theme.class_colors
# 统一从 config/ui.yaml 的 vis 段读）；类别清单本身在 perception/classes.py 定义。
def _box_colors():
    return theme.class_colors()


def _bar_fill_ratio(mask):
    """血条从左往右填充：血量 = 填充到的右边界 ÷ 条宽（0~1）。

    比「颜色像素占比」准——占比会被刻度线、数字、边框、渐变高光稀释
    （满蓝也只到 92% 就是这么来的）。列扫描只看填充到的位置，杂质无关。
    每列 30% 以上是填充色才算「填充列」，抗单行噪声。
    """
    import numpy as np
    if mask is None or mask.size == 0:
        return 0.0
    col_cnt = (mask > 0).sum(axis=0)
    filled = col_cnt >= mask.shape[0] * 0.3
    xs = np.nonzero(filled)[0]
    if len(xs) == 0:
        return 0.0
    return float(xs[-1] + 1) / mask.shape[1]


def _crop_ratio(frame, ratio):
    """按「画面比例」[nx, ny, nw, nh]（0~1）从画面帧截取区域；越界/无效返回 None。

    HP/MP 条现在存的是相对画面的比例；旧版的屏幕绝对坐标（如负的 x）不是比例，
    这里返回 None，需要重新在画面上框选一次。
    """
    if not isinstance(ratio, (list, tuple)) or len(ratio) != 4:
        return None
    h, w = frame.shape[:2]
    try:
        nx, ny, nw, nh = (float(v) for v in ratio)
    except Exception:
        return None
    if not all(0.0 <= v <= 1.0 for v in (nx, ny, nw, nh)):
        return None
    x = int(nx * w); y = int(ny * h)
    rw = max(1, int(nw * w)); rh = max(1, int(nh * h))
    if x < 0 or y < 0 or x + rw > w or y + rh > h:
        return None
    return frame[y:y + rh, x:x + rw]


def _blit_alpha(img, color, fn):
    """按**透明度**把 `fn(target, bgr)` 画到 `img` 上（用户 2026-09-27 要求）。

    OpenCV 的 `line/rectangle/putText` **没有 alpha 参数** ⇒ 半透明只能自己混：
      ① `a >= 255`（老配置、类别框色）⇒ **直接画**，一点额外开销都没有 ✓；
      ② 否则先按不透明画到**整幅副本**上，再 `addWeighted(副本, a, 原图, 1-a)` 混回来 ✓
         （一次全幅拷贝 ≈ 0.5ms，只在这条颜色真配了透明度时才走 ✓；而且它画的是
         **显示帧**、在收流/标注线程里，不在决策那条链上 ✓）。
    `color` 是 `theme.hex_to_bgra` 出来的 `(b, g, r, a)` ✓。
    """
    import cv2                     # 本文件一贯在函数内 import（见 `_draw_dashed_line`）✓
    b, g, r, a = color
    if a >= 255:
        fn(img, (b, g, r))
        return
    ov = img.copy()
    fn(ov, (b, g, r))
    cv2.addWeighted(ov, a / 255.0, img, 1.0 - a / 255.0, 0.0, img)


def _draw_dashed_line(img, y, color, dash_len=10, gap=6, thickness=1, x0=0, x1=None):
    """画一条水平虚线（y 固定）。x0~x1 指定范围，x1=None 表示到右边缘。"""
    h, w = img.shape[:2]
    if y < 0 or y >= h:
        return
    if x1 is None:
        x1 = w - 1
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    x = x0
    while x <= x1:
        import cv2
        cv2.line(img, (x, y), (min(x + dash_len, x1), y), color, thickness)
        x += dash_len + gap


def _draw_dashed_vline(img, x, color, dash_len=10, gap=6, thickness=1, y0=0, y1=None):
    """画一条纵向虚线（x 固定）。y0~y1 指定范围，y1=None 表示到底部。"""
    h, w = img.shape[:2]
    if x < 0 or x >= w:
        return
    if y1 is None:
        y1 = h - 1
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    y = y0
    while y <= y1:
        import cv2
        cv2.line(img, (x, y), (x, min(y + dash_len, y1)), color, thickness)
        y += dash_len + gap


def _vision_box_for(vis, player_box, s):
    """计算视野矩形 (left, top, right, bottom)（画面坐标 int，裁剪到画面内）。

    基准点 = 角色中心 或 画面中心（vision_center），叠加 x/y 偏移，
    再向上下左右扩展 vision_* 像素。某方向 <0 视为不限制（到画面边缘）。
    """
    h, w = vis.shape[:2]
    if s.vision_center or player_box is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = player_box[0], player_box[1]
    cx += s.vision_off_x
    cy += s.vision_off_y
    left = int(cx - s.vision_left) if s.vision_left >= 0 else 0
    top = int(cy - s.vision_top) if s.vision_top >= 0 else 0
    right = int(cx + s.vision_right) if s.vision_right >= 0 else w
    bottom = int(cy + s.vision_bottom) if s.vision_bottom >= 0 else h
    left = max(0, min(left, w))
    right = max(0, min(right, w))
    top = max(0, min(top, h))
    bottom = max(0, min(bottom, h))
    return left, top, right, bottom


def _limit_cpu_threads():
    """把 **torch / OpenCV 的线程池按死在 1**（用户 2026-09-29 链路提效 ✓）。

    **为什么**（这是"上游 98 ms"最可能的一截）：实时回路有三条线程在同几颗核上抢 ——
    推理主回路、收流的读线程（`_boost_reader` 那条 ✓）、以及 GUI。而 **torch / cv2 默认按
    核数开线程池**（128 核的机器就开上百条 ✗），GPU 推理时那些线程纯属空转抢 CPU ✗：
      · `model.predict` 走的是 CUDA，CPU 侧只做前/后处理 ⇒ 多线程**没有收益**，
        却会和读线程抢核 ⇒ 读线程被饿住 ⇒ 积压堆在 ffmpeg 队列 / 内核 UDP 缓冲里
        （见 `_boost_reader` 那段说明 ✓）⇒ **直接变成端到端延迟**（`e2e_probe_ms`）✗✗；
      · `cv2` 同理：`cvtColor(1080p)` / `resize` 默认多线程，和读线程抢同一批核 ✗。
    ⇒ 各按 1 跑，把核让给"每帧都有硬期限"的读线程 ✓（代价：cv2 那几个大操作各慢 ~1 ms
       —— 但那 1 ms 在**主回路**里，而省下来的是**上游排队**，那才是 e2e 的大头 ✓）。

    ⚠ 只对**实时这条链**生效（在 `run()` 里、开推理那一刻调 ✓）—— **训练 / 标注**那些
      工具不受影响（它们各自需要多线程 ✓；`tools/detect_*.py` 里本来就自己设了 cv2=1 ✓）。
    ⚠ 覆盖不了"外面已经设过"的情况：torch 的线程数若被环境变量（`OMP_NUM_THREADS` 等）
      改过，这里仍是最后一次说了算 ✓（本函数就在模型加载之前 ✓）。
    """
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                       # noqa: BLE001 —— 没有 torch / 老版本 ⇒ 算了 ✓
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:                       # noqa: BLE001
        pass


class _LatestSlot:
    """只保留最新一帧的槽位。

    **为什么需要它（这是实时链路里最要命的一环）**
        PyAVSource.read() 是阻塞的，它会按顺序把积压的包一帧帧交给你。
        可 B 机的处理速度（解码+推理）通常追不上 60fps 的推流速度，
        于是每秒欠下的帧全部堆在内核 UDP 缓冲里（默认几 MB）——

            A 机推 60fps  →  B 机只能处理 30fps  →  内核里越堆越多
                                                 ↓
                            画面一直在播「几秒前」的内容，像慢动作

        而且积压不会自己消失：处理多慢，延迟就一直累积下去。

        实时系统的铁律是**宁可丢帧、绝不排队**。所以让一个独立线程拼命读、
        只往这里放最新一帧，消费端取到的永远是最新画面，中间的直接丢。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._evt = threading.Event()
        self.dropped = 0
        self.put_count = 0

    def put(self, frame):
        """放一帧。已有未取走的帧就被覆盖 —— 那正是要丢掉的旧画面。"""
        with self._lock:
            self.put_count += 1
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
        self._evt.set()

    def take(self, timeout=0.5):
        """取最新一帧；超时返回 None（用于让调用方有机会检查停止标志）。"""
        if not self._evt.wait(timeout):
            return None
        with self._lock:
            f = self._frame
            self._frame = None
            self._evt.clear()
            return f

    def clear(self):
        with self._lock:
            self._frame = None
        self._evt.clear()


class LiveThread(QThread):
    #: **原生帧**（numpy BGR）：从流里解出来、**一个字都没画**的那份。
    #: 它专供**取帧做测量**的功能 —— 探针标定、HP/MP 条框选、小地图框选、
    #: 标定弹窗、叠图匹配分核对。它们量的都是像素，画过一笔就全毁 ✗
    #: （用户 2026-09-27 现场：框选小地图时把**视野灰色虚线**一起框了进去）。
    raw_frame_ready = pyqtSignal(object)
    #: **显示帧**（numpy BGR，已画好检测框 / 攻击线 / 视野虚线）：只给画面上看的那一份。
    frame_ready = pyqtSignal(object)     # numpy BGR 图（已画好框）
    stats_ready = pyqtSignal(dict)
    potions_ready = pyqtSignal(float, float)   # (hp, mp) 比例 0~1
    failed = pyqtSignal(str)
    stream_status = pyqtSignal(str)      # waiting / connected / no_stream

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self._p = dict(params)
        self._stop = threading.Event()
        self._infer = threading.Event()   # 推理开关：默认关，先只收画面
        self._last_potions = 0.0
        # 本机负载告警的 (文本, 明细)：由 _load_watch 线程整体重绑定、主回路只读。
        # 用元组整份替换而不是分别写两个键，读的一侧就不会拿到「一半新一半旧」。
        self._load = ("", "")

        # ---- 小地图定位（S3）：把玩家的**世界坐标 / 所在段**写进 WorldState ----
        # 决策层以后寻路要用它回答"我在哪块平台上"。三样东西随时会被改（切项目、
        # 换来源、重框小地图），所以都做成可写字段 + `set_mmap()`，不在启动时读死。
        self._mmap_mid = str(self._p.get("mmap_map_id") or "")
        self._mmap_src = self._p.get("mmap_src") or "stream"
        crop = self._p.get("mmap_crop")
        self._mmap_crop = [int(v) for v in crop] if crop and len(crop) == 4 else None
        self._locator = None            # perception.minimap.PlayerLocator（懒导入懒建）
        #: 「脚下属于哪些集合 / 贴在哪根绳上」用的 (terrain, zones) 缓存（见 _route_ctx）
        self._route_cache = {}
        #: 那份缓存最多活多久（秒）—— 编辑器里新圈了集合，最多这么久就生效（见 _route_ctx）
        self.ROUTE_CTX_TTL_S = 2.0
        self._mmap_cli = None           # 来源=收流 时那一路 TCP（懒起）
        # 黄点容差（界面上那排，键名见 perception.minimap.TRACK_KEYS）。
        # `_track_applied` 记住"已经喂给 locator 的那一份"，主回路比对后按需应用
        # —— **不在 set_mmap 里直接改 locator**：那是另一条线程正在用的对象。
        self._mmap_track = self._p.get("mmap_track") or None
        self._track_applied = None
        #: 「我现在站哪个集合」与**世界坐标**的最近一次定位结果 —— 路径解析器（择路）要用。
        #: 定位没输出时都清成空（别留旧值骗解析器 ✗，见 `_fill_route_ctx`）。
        self._player_here = []
        self._player_at = None
        #: ⭐ **位置状态机**（`perception/pos_state.py` ✓）—— 用户 2026-09-27 定：
        #: "所有的位置状态更新由**位置状态机**自治" ⇒ `_fill_route_ctx` 只喂读数与按键事实，
        #: 由它把 `here_sets` / `ladder_id` / `at_ladder_top` 算好、广播到 `Player` ✓。
        #: 它**有状态**（跨帧记"y 到过的最小值" ✓）⇒ 和本线程同寿命，别按帧重建 ✗。
        self._pos_state = PositionStateMachine()
        #: ⭐ 「**查过的怪框**」—— 用户 2026-09-27："只要是**查询到的怪框地点**，就标出来，
        #: **缓存失效再移除**" ✓。key = 怪号（字符串 ✓），value =
        #: `((x, y, w, h), 集合名列表, why, 首次查到时刻, 失效时刻)` ✓。
        #: ⚠ 写它的**唯一入口**是 `_mark_mob_query`（由 `mob_sets_of` 那个解析器调用 ✓ ——
        #: 它是所有"问怪在哪块平台"的**唯一漏斗** ✓）；读它的是画框那一段 ✓（顺手剪过期的 ✓）。
        self._mob_queries = {}

    def set_mmap(self, map_id=None, src=None, crop=None, track=None):
        """更新小地图定位要的东西（**运行中也改得动**：切项目/换来源/重框/改容差）。

        只传要改的那个。值都是不可变对象或新列表，读的一侧（主回路）拿到的是
        改前或改后的完整值，不会拿到半新半旧。
        """
        if map_id is not None:
            self._mmap_mid = str(map_id)
        if src is not None:
            self._mmap_src = src
        if crop is not None:
            self._mmap_crop = ([int(v) for v in crop] if len(crop) == 4 else None)
        if track is not None:
            self._mmap_track = dict(track)

    def _mmap_panel_from_frame(self, frame):
        """来源=从实时画面 → 从这一帧**复制**出小地图那一块；别的来源 → None。

        **为什么必须复制、还得在画框之前**：主回路是就地往 `vis` 上画玩家蓝框、
        平台线、攻击线的，而同一块 numpy 缓冲区后面的改动会串进"早先切的视图"里。
        黄点识别是按饱和色找的，一条画上去的框线正好能污染它。
        块很小（134×109 这种），一次拷贝可以忽略。
        """
        if self._mmap_src != "live" or not self._mmap_mid or frame is None:
            return None
        crop = self._mmap_crop
        if not crop:
            return None
        x, y, w, h = crop
        H, W = frame.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
            return None                 # 换了分辨率还没重框 → 由 note 说清楚
        import numpy as np
        return np.ascontiguousarray(frame[y:y + h, x:x + w])

    def _route_ctx(self, mid):
        """(terrain, zones) —— 按地图 id 缓存，**最多每 2 秒重读一次**。

        为什么要缓存：上绳执行器每帧都要问"脚下属于哪些集合、贴在哪根绳上"，而读地形
        JSON 是几十毫秒的活 —— 每帧读会把实时回路拖垮。
        为什么还要定期重读：在编辑器里新圈了集合 / 重导了地形，不重读就会拿**旧数据**
        判"到了没有"（那是"明明到了却说没到"里最难查的一种）。
        """
        now = time.monotonic()
        c = self._route_cache
        if (c.get("mid") == mid
                and now - float(c.get("ts") or 0.0) < self.ROUTE_CTX_TTL_S):
            return c.get("t"), c.get("z")
        t = z = None
        try:
            from core import mapdata
            from core import zones as zones_mod
            t = mapdata.load(mid, with_canvas=False)
            z = zones_mod.load(mid)
        except Exception:                       # noqa: BLE001
            t, z = None, None
        self._route_cache = {"mid": mid, "ts": now, "t": t, "z": z}
        return t, z

    def _make_route_resolver(self, stg):
        """给 agent 装的**路径解析器**：`dst_set → {"path","jobs","why","here"}`。

        「定点休息」要用它自己走到指定地点（agent 不持有地形/集合 ⇒ 解析这一层在这里）。

        ⚠ 为什么是**函数**、不是预存的表（`docs/开发计划.md` 里原先写的是
        `set_name → [job,…]`）：路径的**起点**是"我现在站哪个集合"，它每一拍都可能变
        ⇒ 只能**调用时**才解析。
        ⚠ 开销：`find_path` 是 BFS（毫秒级），`mapdata.load` 走 `_route_ctx` 的 **2 秒缓存**
        —— 一次休息最多解析两三次，**不在帧循环里** ✗（别把它挂进每拍）。
        """
        def resolve(dst_set):
            mid = str(getattr(self, "_mmap_mid", "") or "")
            if not mid:
                return {"jobs": [], "why": "还不知道当前是哪个项目 / 地图", "here": False}
            t, z = self._route_ctx(mid)
            if t is None or z is None:
                return {"jobs": [], "here": False,
                        "why": "读不到地形 / 集合数据（先在「路线识别」里生成地形图）"}
            src = next((n for n in (getattr(self, "_player_here", None) or [])
                        if n in z.sets), "")
            if not src:
                src = str(getattr(stg, "route_goto_set", "") or "")
            if not src:
                return {"jobs": [], "here": False,
                        "why": "你现在站的这块没圈进任何集合 ⇒ 解析不出路线"}
            from decision import route as route_mod
            try:
                return route_mod.plan_jobs(
                    t, z, src, dst_set,
                    tol_px=int(getattr(stg, "align_tol_px", 6) or 6),
                    hold_ms=int(getattr(stg, "align_hold_ms", 250) or 0),
                    # 「多久没进展算卡住」（爬不动 / 等落地 / **跳跃没落到** 三处共用 ✓）：
                    # **统一**取「移动操作尝试间隔(ms)」换算成秒 ✓（2026-09-27 用户要求：
                    # "移除卡住判定时长(s)，统一采用移动操作尝试间隔(ms)" ✓；下限 0.5 秒与
                    # 任务里的钳法一致 ✓。⚠ **走不吃它** —— 走照旧用 `route.STALL_S` 常量 +
                    # 「寻路超时时间」兜底 ✓，见 `route.job_for_edge`）
                    stall_s=route_mod.stall_s_from_retry_ms(
                        getattr(stg, "move_retry_ms", 3000)),
                    # 「起跳距离(px)」：「跳」那条边用的**临时参数** ✓
                    # ⚠ 面板（`route_panel._command_first_step`）也传了同一个值 ——
                    #   两边必须一致，否则「面板能下发、休息却走不过去」✗
                    jump_start_px=int(getattr(stg, "jump_start_px", 0) or 0),
                    # 玩家世界坐标：同一对集合有多条绳时**挑离我最近的那根** ✓
                    #（定位没输出时是 None ⇒ 退回文件顺序，不猜 ✗）
                    at=getattr(self, "_player_at", None))
            except ValueError as ex:            # 绳找不到 / 说不清上下 / 落点找不到
                return {"jobs": [], "why": str(ex), "here": False}
            except Exception as ex:             # noqa: BLE001
                return {"jobs": [], "why": "造任务时出错：%s" % ex, "here": False}
        return resolve

    def _make_mob_sets_resolver(self):
        """给 agent 装的**「这只怪在哪块平台上」解析器**（用户 2026-09-27 要求）。

        要求原话："chase状态需要判定一下怪物位于的foothold集合，若不与玩家处于同一个，
        需要先下达前往任务" ⇒ agent 要在追怪前知道"怪和我是不是同一块平台"。

        为什么放在这里、而不是塞进 agent：agent **不持有地形 / 集合**（和 `route_plan` 同一个
        理由 ✓），而本类有 `_route_ctx(mid)`（`mapdata.load` + `zones.load`，**2 秒缓存** ✓）
        和玩家世界坐标 ✓ ⇒ 这里换、那里用 ✓（**一处实现** ✓ 约定 10）。

        ⚠ 只对**当前锁定的那一只**调用（每拍最多一次 ✓）：`foothold_below` 要扫一遍 foothold
          （几百条），**不要**给每只怪都算 ✗（那是每帧几十次的量级）。

        回值：`{"world": (x, y), "sets": [集合名…], "why": ""}`；
            算不出（没定位 / 没地形 / 没圈进集合）⇒ `why` 说清、`sets` 空 ✓（**不猜** ✓）。
        """
        def _resolve(player, mob):
            from perception import minimap as mm

            mid = str(getattr(self, "_mmap_mid", "") or "")
            if not mid:
                return {"world": None, "sets": [], "why": "还不知道当前是哪个项目 / 地图"}
            t, z = self._route_ctx(mid)
            if t is None or z is None:
                return {"world": None, "sets": [],
                        "why": "读不到地形 / 集合数据（先在「路线识别」里生成地形图）"}
            # ⚠ 怪只有**画面坐标** ⇒ 用玩家做锚推世界坐标（口径同 `screen_to_world`：
            #   `画面 = 世界 − Camera` ✓）。脚底口径：怪框底 = `mob.y + mob.h / 2` ✓（和玩家的
            #   `bottom` 对齐 ✓）。
            _mh0 = float(getattr(mob, "h", 0.0) or 0.0)
            _my0 = float(getattr(mob, "y", 0.0) or 0.0)
            _mx0 = float(getattr(mob, "x", 0.0) or 0.0)
            _wy0 = getattr(player, "world_y", None)
            _wx0 = getattr(player, "world_x", None)
            _bt0 = getattr(player, "bottom", None)
            if _wy0 is None or _wx0 is None or _bt0 is None:
                return {"world": None, "sets": [],
                        "why": "玩家还没定位（拿不到世界坐标 ⇒ 推不出怪在哪）"}
            _cx = float(_wx0) - float(getattr(player, "x", 0.0) or 0.0)   # 相机 x（不平滑 ✓）
            # ⭐ 相机 y 用**脚底偏移**（用户 2026-09-28 ✓）：框底**不一定**正好压在脚底
            #   （鞋底阴影 / 披风 / 特效会让框多出一截 ✗）⇒ 用它补正 ✓。
            #   ⚠ 读的是 `world_state` 里的**镜像值**（`live_thread` 每帧灌 ✓）—— 这边**不能**
            #     反过来 import decision（会绕成循环依赖 ✗ 见 world_state 那段说明 ✓）。
            from perception import world_state as _ws_mod
            _cy_raw = float(_wy0) - (float(_bt0) + float(_ws_mod.FOOT_OFFSET_PX))
            # ⭐⭐ **优先用「可信相机」**（用户 2026-09-29 任务 2 ✓）：上面那两份是**每拍现算**
            #   的（玩家框抖 Δ ⇒ 每只怪的世界坐标同量平移 Δ ✗）；`PlayerTracker` 已经把
            #   "跟黄点对得上的那几拍"挑出来存在 `player.cam_x/cam_y` 上（口径与这里一致 ✓）
            #   ⇒ 有它就用它，**没有（None）就退回老口径**（行为一字不变 ✓，不许当成 0 ✗）。
            _pcx = getattr(player, "cam_x", None)
            _pcy = getattr(player, "cam_y", None)
            if _pcx is not None and _pcy is not None:
                _cx, _cy_raw = float(_pcx), float(_pcy)
            # ⚠⚠ **相机 y 必须做 EMA 平滑**（2026-09-28 治本 ✓ —— 用户报"**#159 没找到，理由是什么？**"✓）：
            #   实测 `behavior.log` 的 `mob_fh`：`ply_wy`（玩家世界 y）在 **−91 ~ +197** 之间摆
            #   **288 像素** ✗（图 105040303 的层距才 **540** ⇒ 摆了半层多 ✗）⇒ 换算出的怪世界
            #   坐标在**层之间乱跳** ⇒ 挑面阶梯**全落空** ⇒ `why=怪底下没找到 foothold`（92 次 ✓）。
            #   ⚠ **根因不在挑面** ✗：`screen_to_world` 用玩家做锚（`cam_y = world_y − bottom` ✓），
            #     而**黄点（`world_y`）与玩家框（`bottom`）不是同一时刻的** —— 黄点是**独立传感器**、
            #     有自己的刷新节奏 ✓ ⇒ 玩家一走动/跳跃，`cam_y` 就**抖** ✗
            #     ⇒ **所有"画面 → 世界"都跟着抖** ✗（怪 / 落点 / 够不够得着 都在抖 ✓）。
            #   ⇒ 平滑它 ✓。⚠ 代价：镜头**真的**快速滚动时有几十毫秒滞后 —— 而"挑**哪一层**"
            #     只要粗细够（层距几百 ✓）⇒ 可接受 ✓，比"层间乱跳"好得多 ✓。
            #   ⚠ 平滑值挂**本线程**上（`mob_cost_of` 也走这里 ✓ **同一份** ✓ 别各算一份 ✗）。
            # ⛔⛔ **2026-09-28：撤掉 EMA 平滑**（`CAM_SMOOTH_A` 已**停用** ✓ 别再启回来 ✗）。
            #   它当年的理由是"`ply_wy`（黄点世界 y）在 −91~+197 之间摆 **288 像素** ⇒ 相机抖"
            #   ✗ —— 那天**把"玩家真的换层"误读成"读数抖动"** ✗✗（2026-09-28 用真 log 核实：
            #   `ply_wy` 的取值就是**各层的面 y** ✓ 顶层 −176 / 三楼 −91 / 一楼 95 / 底层 196——
            #   玩家在楼层间跑，这个量**本来就该跳** ✓）。
            #   用真 log（`mob_fh` 最后 400 条）实测：平滑值偏离"原始相机"**中位 46.8、
            #   最大 362.5 像素** ✗；而挑面命中率 **38% → 38%（一点没提高）** ✗
            #   ⇒ 它**滤掉的是真事、留下的是误差** ⇒ 纯有害 ✓ 撤。
            #   ⇒ 直接用 `_cy_raw`（= `player.world_y − player.bottom` ✓ 语义清楚、可解释 ✓）。
            #   ⚠ 保留 `_cam_smooth` 这个**字段名**（打点 / 排查还在读它 ✓ —— 现在它恒等于原始值 ✓）。
            _sm = _cy_raw
            self._cam_smooth = _sm
            # 怪的世界坐标：**框底**（`band` / 兜底用）与**框中心**（"向下垂线"那级用 ✓）
            world = (_mx0 + _cx, (_my0 + _mh0 / 2.0) + _sm)     # 框底 ✓
            mid_w = (_mx0 + _cx, _my0 + _sm)                    # 框中心 ✓
            # ⭐ **`above_tol`**（2026-09-28 现场修）：`foothold_below` 默认只允许"面比给的 y
            #   高 40 像素以内"（那是按**玩家**定的：黄点画的是中心、脚底就在附近 ✓）。
            #   而怪这头给的是"**框底**换算的世界 y"，**大怪的检测框并不包住脚** ✗（实测：
            #   二楼那只石人，框底比它真正站的二楼面**高 84 像素** ⇒ 在老口径下那条面
            #   **直接被跳过** ⇒ 只剩"下方最近"的**一楼**⇒ 怪被判在下一层 ✗）。
            #   用户现场原话："查#171 一个二楼的怪显示在底层，现在已经卡住了"✓
            #   （`behavior.log`：`mob_goto_ok 已出发：一楼 → 底层` / `已出发：底层 → 一楼` 来回 ⟲）。
            #   ⇒ 给怪放宽到 `MOB_ABOVE_TOL`（比较仍然是"|Δy| 最小" ✓ ⇒ **怪真站在一楼、
            #     框底就是脚时，一楼 Δ≈0 照样赢** ✓ 不受影响 ✓）。
            # ⭐⭐ **`band` = "怪身体高度范围内的面"**（2026-09-28 定稿口径 ✓）：
            #   候选面必须落在 `[框底世界 y − 框高, 框底世界 y + 框高]` ✓。
            #   ⚠ 为什么不再用固定容差（`MOB_ABOVE_TOL=200` 那版 ✗）：**层距会到 540** ——
            #     实测图 105040303（顶层 375 / 三楼 915 / 二楼 1455 / 底层 1995）⇒ 相邻层
            #     差 **540** ✗ ⇒ 固定值连**半层**都盖不住 ⇒ 照样跨层 ✗（用户当天又报
            #     "**又把顶层怪判成底层怪了**" ✓）。
            #   ⚠ "框底 ≠ 脚底"是**双向**的（框没包住脚 ⇒ 面在框底**上方**；框包过头 ⇒ 面在
            #     **下方** ✓）⇒ 必须**对称**覆盖 ✓。
            #   ⚠ 身体高 `h` 是**检测框直接给的** ⇒ 不新造魔数 ✓、而且自适应：
            #     小怪 ≈40 ⇒ ±40（比老的 ±200 **更严** ✓）；大怪 ≈300 ⇒ ±300（盖得住 540 ✓）。
            #   ⚠ 区间里一条面都没有 ⇒ **退回**老口径（不瞎判 ✓ 见下面那行 ✓）。
            _mh = float(getattr(mob, "h", 0.0) or 0.0)
            _my = float(getattr(mob, "y", 0.0) or 0.0)
            _mx = float(getattr(mob, "x", 0.0) or 0.0)
            # ⚠ 用**平滑后**的相机（和上面同一个 `_sm` ✓ —— ⛔ 别再用 `screen_to_world`：
            #   它拿的是**未平滑**的 cam ⇒ 三者会打架 ✗）。
            _top = (_mx0 + _cx, (_my0 - _mh0 / 2.0) + _sm)
            _bot = (_mx0 + _cx, (_my0 + _mh0 / 2.0) + _sm)
            f = None
            #: ⭐ **卡在哪一级**（2026-09-28 加 ✓）：`why` 原来只有一句"怪底下没找到 foothold"
            #: ⇒ 排查时分不清"周围真的没有地形 / 框底偏了一层 / 方向自相矛盾" ✗
            #: （那天就在这一句上猜了很久 ✓）。各级挑面失败时改写它 ✓。
            _lvl = "①怪框覆盖的那一段里没有面"
            if _mh > 0 and _top is not None and _bot is not None:
                lo, hi = float(_top[1]), float(_bot[1])
                if lo > hi:
                    lo, hi = hi, lo            # 世界 y 越大越靠下 ⇒ 归一化（顶层在前 ✓）
                # ① **主口径：怪框覆盖的那一段**（`band=(框顶, 框底)` ✓）——
                #    怪**一定站在"穿过它身体"的那条面**上 ⇒ 不猜"脚底在哪一格" ✓。
                #    实测（图 105040303，层距 **540**）：框底偏差 ≤ 框高时**全对** ✓，
                #    而老的"框底 ± 容差"到**半层**就崩 ✗（300 就该是顶层、它给三楼 ✗）。
                f = t.foothold_below(world[0], world[1], band=(lo, hi))
                if f is None:
                    # ⭐⭐ ② **用户的"向下垂线"方案**（2026-09-28 ✓ 原话："用 <玩家脚底到怪框
                    #    **中心的向量**>在世界坐标对应到**怪框中心**的世界坐标后，**向下做垂线**，
                    #    接触到的第一个 foothold 集合就是该怪的" ✓）。
                    #    起点用**怪框中心**（不是框底 ✗）—— 中心和上一轮那条讨论同理：
                    #    误差更小、且在**框高度估错时上下沿反向、中心互相抵消** ✓；
                    #    而"框覆盖身体"时**中心一定在脚的上方** ⇒ **往下第一条面就是它踩的那条** ✓。
                    #    这正是 `foothold_below` 的**原始语义**（`fy >= y - tol`、**往下不限** ✓）
                    #    ⇒ `above_tol=0` = "从这点**往下**第一条" ✓（不用 `band` ✓）。
                    #    ⚠ **必须补上界** ✗（下一条就是本条的全部风险）：**往下不限** ⇒
                    #      框整体偏下时会**挑到下一层**、或挑到很远的"虚空面" ✗
                    #      ⇒ 要求它**不超过"框底 + 半个框高"** ✓（同一套"由框自己定"的自适应思路 ✓
                    #        只有"明显跑远"才拒 ✓）。
                    _mid = mid_w        # 框中心（**平滑后**的相机 ✓ 与上面同一份 ✓）
                    _f2 = t.foothold_below(float(_mid[0]), float(_mid[1]),
                                           above_tol=0.0)
                    if _f2 is not None:
                        try:
                            if float(_f2.y_at(float(_mid[0]))) <= hi + _mh * 0.5:
                                f = _f2
                        except Exception:       # noqa: BLE001 —— 取面高失败就当没这级 ✓
                            pass
                if f is None:
                    # ⭐ **就到这儿为止**（`sets` 空 ✓）—— 宁可**没找到**（不判）也**不要判错**
                    #   （乱下前往、来回跑 ✗）。
                    #   ⚠⚠ **原来的 ③「再往上放半个框高」已按用户要求删掉**（2026-09-28 ✓
                    #     原话："③ 还空 ⇒ 只往上放宽半个框高　这一步没有必要，去掉"）：
                    #     它只在"框底偏高、没包住脚"那一半有用，而**框没框全**这种事应该由
                    #     "**框面积闸**"挡在**查询之前**（推迟 / 不查 ✓）—— 靠**放宽挑面**去救
                    #     是错的方向 ✗（放宽必然把更远 / 更错的层拉进来 ✓，那条实测就在上面 ✓）。
                    _lvl = "②框中心向下垂线也没找到面"
            else:
                # 连框高 / 世界坐标都拿不到（老环境 / 怪框异常）⇒ 退回老口径 ✓。
                _lvl = "③怪框高 / 世界坐标拿不到 ⇒ 走了老口径"
                f = t.foothold_below(world[0], world[1], above_tol=MOB_ABOVE_TOL)
            # ⭐⭐ **方向自洽校验**（用户 2026-09-28 的思路："**至少顶层判定成底层这种离谱的
            #   向量都反了的应该能及时发现**"✓ —— 这是那条思路里**真正有效**的部分 ✓；
            #   另一半"y 投影 = 高度差"是**恒等式**，校不出"框底 ≠ 脚底"✗ 见 SKILL 97 ✓）。
            #   画面 y 与世界 y **同向**（越小越靠上 ✓）⇒ 硬约束：
            #     · 怪在**画面上比玩家高** ⇒ 它那一层必须**比玩家那层高（世界 y 更小）** ✓；
            #       低 —— 同理 ✓。
            #   ⚠ 现在的"挑最近面"**根本不管方向** ✗ ⇒ 才会出现"玩家在一楼、怪在顶层、
            #     却判成**底层**"这种**符号都反了**的结果 ✗（用户现场 ✓）。
            #   ⇒ 挑完（含各级兜底）之后**校验一次**：方向反了 ⇒ **丢**（`f = None`
            #     ⇒ `sets` 空 ⇒ 不判 ⇒ 照旧追 ✓ **宁缺勿错** ✓）。
            #   ⚠ 玩家那端用 `world_y`（黄点**脚底** ✓ = 玩家所在层的 y；已验证准到 1px ✓）。
            if f is not None:
                # ⚠⚠ **方向必须用"画面上的框中心"判，不能用"框底"**（2026-09-28 现场修 ✗）：
                #   原来拿 `world[1]`（**怪框底**换算的世界 y）当"怪在上下"—— 而"框底 ≠ 脚底"
                #   **本身就会偏几十上百像素** ⇒ 那个符号会**乱** ⇒ 把**本来判对**的也拒掉 ✗
                #   （用户当天报："**运行一段时间后所有的查询怪物全是没找到**"✓）。
                #   ⇒ 改成 `mob.y` vs `player.y`（**框中心** ✓，和 `agent._in_box` 判上下同一套 ✓）：
                #     两个都是**画面量**、同向（越小越靠上 ✓）⇒ 乘积判符号就够 ✓ 稳健 ✓。
                #   ⚠ `|中心之差| < MOB_DIR_MIN_PX` ⇒ **太近就不校验** ✓（同一层的怪中心本来
                #     只差几像素，硬判符号只会误杀 ✓）。
                try:
                    _fy = float(f.y_at(world[0]))
                    _pw = float(getattr(player, "world_y", 0.0) or 0.0)
                    _mcy = float(getattr(mob, "y", 0.0) or 0.0)
                    _pcy = float(getattr(player, "y", 0.0) or 0.0)
                    # ⚠⚠ **阈值由框自己定，不用拍脑袋的常数**（2026-09-28 用户质疑后改 ✓）：
                    #   原话："**怪框的 4 条边长都会有误差，凭什么认为框中心"不受框底偏差影响"？**"
                    #   —— 他说得对 ✓：`框中心 = (框顶 + 框底)/2` ⇒ **上下沿有误差中心就有误差** ✗
                    #   （"不受影响"那个说法**不准确** ✓）。
                    #   准确的差别是**敏感度差一个量级**：旧方案拿"**换算出的世界 y**"当参照
                    #   （和被校验的量**同一套坐标** ⇒ 误差**直接进符号** ⇒ 超过**半层**就翻 ✗）；
                    #   这条只回答"**谁在上面**"，而跨层的画面高度差是**几百像素** ⇒
                    #   上下沿各差几十像素**不改变谁在上面** ✓。
                    #   ⇒ 阈值取"**怪框高的一半**"= **这个框自身的固有不确定度**（上下沿各可能差半个框 ✓）
                    #     —— 只有"**明显是跨层**"才做校验 ✓ 判不准的一概**不拦** ✓（宁可少拦 ✓）。
                    _tol = max(MOB_DIR_MIN_PX, float(getattr(mob, "h", 0.0) or 0.0) * 0.5)
                    if (abs(_mcy - _pcy) > _tol
                            and (_mcy - _pcy) * (_fy - _pw) < 0.0):
                        f = None              # 画面说在上、层却在下（或反之）⇒ 离谱 ⇒ 拒判 ✓
                        _lvl = "④判出来了但方向自相矛盾（画面说在上面、层却在下面）"
                except Exception:             # noqa: BLE001 —— 校验坏掉就当没校（别把追怪弄停 ✗）
                    pass
            if f is None:
                # ⭐ **分档说清卡在哪一级**（2026-09-28 加 ✓）：原来只有一句"没找到 foothold"
                #   ⇒ 排查时**分不清**是"周围真的没有地形"、"框底偏了一层"、还是"方向自相矛盾"
                #   ✗（用户报"**又开始全程找不到怪的 foothold 集合了，这不应该**"那轮就卡在这句
                #   上猜了半天 ✓）。`_lvl` 由各级挑面失败时写 ✓。
                return {"world": world, "sets": [],
                        "why": "怪底下没找到 foothold（%s —— 它可能在半空 / 或那片没圈地形）"
                               % (_lvl or "说不清")}
            names = [str(n) for n in z.set_of(str(f.fid))]
            why = "" if names else "怪站的那条 foothold（#%s）没圈进任何集合" % f.fid
            return {"world": world, "sets": names, "why": why}

        def resolve(player, mob):
            """⭐ 这里是**唯一漏斗**：每问一次"这只怪在哪块平台"就**记一笔、画面上标出来** ✓。

            用户 2026-09-27："只要是**查询到的怪框地点**，就标出来，**缓存失效再移除**" ✓ ——
            记在漏斗里（`_mark_mob_query` ✓）的好处：`mob_cost_of` 内部也调本函数 ✓ ⇒
            agent 那边**一处都不用改** ✓（它在哪调、调几次都不影响"查过就有标记" ✓）。
            """
            info = _resolve(player, mob)
            # ⛔ 「**禁用杀怪寻路**」开着 ⇒ **不记账、也不打点**（用户 2026-09-28 ✓ 原话：
            #   "开启后**不再查询怪物框所属的 foothold 集合**（**也不显示**）"✓）。
            #   为什么**一处收口**就够了：记账（`_mark_mob_query` ✓）是画面上那行
            #   `查#怪号 集合名`的**唯一来源**（绘制段只读那份账 ✓）
            #   ⇒ **不记账 ⇒ 画面自然不出现** ✓（绘制那段一个字都不用动 ✓）。
            # ⚠⚠ **这句注释里别写出那个读口的名字** ✗：`selftest_live_panel.t_mob_box_labels`
            #   是用 `src.index("<读口名>()")` 去找绘制段、再在**固定窗口**里找 `_qlbl` 的
            #   ⇒ 注释里先出现一次，它就会定位到**注释**、窗口里当然找不到 ⇒ **用例假红** ✗
            #   （2026-09-28 就是这么踩的：注释比绘制段早 1400 行 ✓）。
            #   顺手把 `mob_fh` 打点也停了 ✓ —— 开关开着时"查怪在哪块平台"这件事**根本不该发生**
            #   （要排查就先把它关掉 ✓ 那时打得一样全 ✓）。
            # ⚠ 仍然 `return info`：**别处**还要用它 —— `_zone_only`（「禁止战斗」时的**区域筛** ✓）
            #   那是**另一个功能**，不在这个开关的范围内 ✗（见 `DecisionSettings` 那段说明 ✓）。
            # ⚠ **两层都要 `getattr`**（只护内层会炸 ✗）：自检 / 老环境里 `self.agent` 可能是个
            #   **替身**（没有 `.settings` ✓）⇒ 写成 `self.agent.settings.disable_...` 会当场
            #   `AttributeError` ⇒ 把"查过的怪框"那几条用例全带崩 ✗（2026-09-28 踩到 ✓）。
            _off = False
            try:
                _off = bool(self.agent.settings.disable_chase_pathfinding)
            except AttributeError:
                _off = False
            if _off:
                return info
            names = (info or {}).get("sets") or []
            why = (info or {}).get("why") or ""
            self._mark_mob_query(mob, names, why)
            # ⭐ **可观测性**（2026-09-28 加；用户："**运行一段时间后所有的查询怪物全是没找到**"✓
            #   —— 那时 log 里**只有** `mob_goto_*`，`why` 只存在面板里 ⇒ **查不下去** ✗）。
            #   ⚠ 只记**值变了**的那一次（同 `behavior.sample` 的语义，这里自己实现免依赖 ✓）——
            #     否则它和 `lock_scored` 一个量级（两万多条 ✗）会把 log 淹掉 ✗。
            try:
                _sig = (str(why), tuple(sorted(str(n) for n in names)),
                        round(float(getattr(mob, "x", 0.0) or 0.0) / 40.0),
                        round(float(getattr(mob, "y", 0.0) or 0.0) / 40.0))
                if _sig != getattr(self, "_last_mob_fh", None):
                    self._last_mob_fh = _sig
                    _w = (info or {}).get("world")
                    behavior.event(
                        "mob_fh",
                        why=(str(why)[:60] or "ok"),
                        sets=(",".join(str(n) for n in names)[:40] or "-"),
                        mob_cy=round(float(getattr(mob, "y", 0.0) or 0.0), 1),
                        ply_cy=round(float(getattr(player, "y", 0.0) or 0.0), 1),
                        w_x=(round(float(_w[0]), 1) if _w else None),
                        w_y=(round(float(_w[1]), 1) if _w else None),
                        ply_wy=round(float(getattr(player, "world_y", 0.0) or 0.0), 1),
                        # ⭐⭐ **2026-09-28 补四个量**（用户报"**又开始全程找不到怪的 foothold
                        #   集合了，这不应该**"✓ —— 排查时发现**缺的正是它们** ✗）：
                        #   · `mid` = 当前地图（不然**不知道拿哪张图去复现** ✗ —— 那天只能靠集合名猜 ✓）；
                        #   · `ply_wx` = 玩家**世界 x**（判"世界坐标是不是整体偏了" ✓）；
                        #   · `ply_cx` = 玩家**画面 x**（和 `w_x` 一减就**反推出相机 x** ✓）；
                        #   · `ply_bot` = 玩家**画面框底**（`cam` 的对照量 ✓ —— 现在 `cam = ply_wy − ply_bot` ✓）。
                        mid=str(getattr(self, "_mmap_mid", "") or ""),
                        ply_wx=round(float(getattr(player, "world_x", 0.0) or 0.0), 1),
                        ply_cx=round(float(getattr(player, "x", 0.0) or 0.0), 1),
                        ply_bot=round(float(getattr(player, "bottom", 0.0) or 0.0), 1),
                        cam=(round(float(getattr(self, "_cam_smooth", 0.0) or 0.0), 1)))
            except Exception:                 # noqa: BLE001 —— 打点坏了别影响查询 ✓
                pass
            return info

        return resolve

    # ---------------------------------------------------------------- 「查过的怪框」

    def _mob_query_ttl(self):
        """「查过的怪框」标多久 —— **就用缓存那一把尺**（用户："**缓存失效再移除**" ✓）。

        取 `agent._zone_cd_s()`（= 「**区域查询CD(s)**」的兜底值 ✓ —— 2026-09-28 起它是
        **每个战斗区域项**各自配的 ✓，见 `DecisionSettings.battle_zones` ✓；"查过的怪框"
        与"区域筛缓存"两处**必须同一把尺**，别新造 ✗）；拿不到就退回模块常量 ✓。
        """
        try:
            return max(0.5, float(self.agent._zone_cd_s()))
        except Exception:                      # noqa: BLE001 —— 老环境/替身对象 ⇒ 退回默认 ✓
            from decision.agent import ZONE_GOTO_RETRY_S

            return float(ZONE_GOTO_RETRY_S)

    def _mark_mob_query(self, mob, names, why="", now=None):
        """记下"**这只怪的框被查过**"（画面上要标出来 ✓），时效见 `_mob_query_ttl` ✓。

        存的是**画面坐标框**（怪每拍都在动 ⇒ 标记要跟着它走 ✓）与这次查到的集合名 ✓；
        查不出集合（`names` 空 ✓）**也照记** ✓ —— 用户要的是"**查询到**"就标 ✓。

        ⭐⭐ **"有 → 失败"时沿用上次有效结果 + 续期**（用户 2026-09-28 ✓ 原话："如果怪物
        查询从 有→失败，那么其应该使用使上次有效的数据缓存并刷新缓存时间而不是空"）：
          · 本次 `names` 空、而**上次存着非空集合** ⇒ **集合名沿用上次那份** ✓，失效时刻按
            `now + ttl` **续期** ✓ ⇒ 框不会因为一次判不出来就消失 ✗、也不会"查到过又变空" ✗；
          · **框坐标永远用本次的** ✓（怪在动，标记得跟着它走 ✗ 别用旧框）；
          · 本次的 `why`（失败原因）**照记** ✓ —— 它给 `mob_fh` 那条 log 看，而那边用的是
            `resolve` 里的**真值**（不受这里影响 ✓）⇒ 画面显示上次结果、log 里仍能看到
            "这次为什么没查到" ✓ **两边都不丢** ✓；
          · ⚠ 与用户**当天早些**那条要求（"失败就写失败、不要显示这么长"）的取舍：现在只要
            成功过一次，画面就**不再出现**「查#id 失败」✗ ⇒ "这次没查到"要去 log 的
            `mob_fh` 里看（它带 `why` ✓）。用户 2026-09-28 的新口径明确要"**不要空**"✓ 照办 ✓。
        """
        mid = getattr(mob, "id", None)
        if mid is None:
            return
        now = time.monotonic() if now is None else float(now)
        key = str(mid)
        prev = self._mob_queries.get(key)
        _names = [str(n) for n in (names or [])]
        if not _names and prev and prev[1]:
            _names = list(prev[1])      # ⭐ 沿用上次有效集合（用户 2026-09-28："不要空" ✓）
        self._mob_queries[key] = (
            (float(getattr(mob, "x", 0.0) or 0.0), float(getattr(mob, "y", 0.0) or 0.0),
             float(getattr(mob, "w", 0.0) or 0.0), float(getattr(mob, "h", 0.0) or 0.0)),
            _names, str(why or ""),
            float(prev[3]) if prev else now,          # 首次查到时刻（"多久前查的" ✓）
            now + self._mob_query_ttl())              # 失效时刻 ⇒ 到点就移除 ✓（沿用也照续期 ✓）

    def queried_mob_boxes(self, now=None):
        """还没失效的"查过的怪框" → `[(怪号, (框, 集合名, why, 首查时刻, 失效时刻)), …]`。

        ⚠ **顺手剪掉过期的**（用户："缓存失效再移除" ✓）—— 剪在这里（唯一的读口 ✓），
        画框那段直接用返回值即可，不必自己判时间 ✓。
        """
        now = time.monotonic() if now is None else float(now)
        for k in [k for k, v in self._mob_queries.items() if v[4] <= now]:
            del self._mob_queries[k]
        return list(self._mob_queries.items())

    def _make_set_cost_resolver(self):
        """给 agent 装的「**从我站的集合 → 目标集合**的寻路距离」解析器（用户 2026-09-28 ✓）。

        用途：人在一块**不能打**的平台上时，要挑一块**代价最低的可战斗区**
        （见 `decision/agent._pick_battle_zone` ✓）。口径与 `mob_cost_of` **完全同一套**
        （`route.path_cost` ✓ 一处实现 ✓），只是"目标"从**一只怪**换成**一个集合** ✓。

        回值：`float | None`；**算不出给 `None`** ⇒ 调用方退回"回第一条"的老行为 ✓（不猜 ✗）。
        ⚠ 与 `mob_cost_of` 同款签名（**收 `player`** ✓）—— 起点集合要从 `player.here_sets` 拿 ✓
        （agent 不持有地形/集合，这份换算只能在这儿做 ✓）。
        """
        def resolve(player, dst_sets):
            mid = getattr(self, "_mmap_mid", None)
            if not mid:
                return None
            try:
                t, z = self._route_ctx(mid)
            except Exception:                 # noqa: BLE001 —— 拿不到地形 ⇒ 不猜 ✓
                return None
            if t is None or z is None:
                return None
            from decision import route as route_mod

            return route_mod.path_cost(
                t, z, getattr(player, "here_sets", None), list(dst_sets or []),
                getattr(self, "_player_at", None))

        return resolve

    def _make_mob_cost_resolver(self, sets_of):
        """给 agent 装的**「走过去要多少寻路距离」解析器**（用户 2026-09-27 要求）。

        需求原话："优化：锁定目标优先级要按「**寻路距离**」最近，而不是旧的应该是**绝对距离**"
        —— 挑要打哪只怪时，要能回答"**走到它那儿有多远**" ✓。

        入参 `sets_of` = `mob_sets_of` 那个解析器（**共用同一次"画面→世界 + 落在哪块面"换算** ✓，
        一处实现 ✓）。回值：`{"cost": float|None, "sets": [...], "why": ""}`
        —— `cost` = 从**我站的集合**到**怪站的集合**的寻路距离（世界像素，`route.path_cost` 一处口径 ✓）；
        算不出来给 `None`（**不猜** ✓，调用方退回绝对距离 ✓）。

        ⚠ **只在"要挑目标"那一刻调用**（`agent._nearest`，锁定过期才走 ⇒ 500~1000ms 一次 ✓），
          而且是对**当时候选的怪**；一次 `path_cost` = 一次 BFS + 每段一次 `pick_edge`（偏重 ✗）
          ⇒ **绝不挂进每拍、也别对全屏幕的怪都算** ✗（同 `route_plan` 的纪律 ✓）。
        """
        def resolve(player, mob):
            mid = str(getattr(self, "_mmap_mid", "") or "")
            if not mid:
                return {"cost": None, "sets": [], "why": "还不知道当前是哪个项目 / 地图"}
            t, z = self._route_ctx(mid)
            if t is None or z is None:
                return {"cost": None, "sets": [],
                        "why": "读不到地形 / 集合数据（先在「路线识别」里生成地形图）"}
            info = sets_of(player, mob) or {}
            names = [str(s) for s in (info.get("sets") or []) if str(s)]
            if not names:
                return {"cost": None, "sets": [],
                        "why": info.get("why") or "怪站的 foothold 没圈进任何集合"}
            from decision import route as route_mod
            try:
                cost = route_mod.path_cost(
                    t, z, getattr(player, "here_sets", None), names,
                    getattr(self, "_player_at", None))
            except Exception as ex:              # noqa: BLE001
                return {"cost": None, "sets": names, "why": "算寻路距离出错：%s" % ex}
            why = "" if cost is not None else "寻路距离算不出来（缺几何 / 定位数据）"
            return {"cost": cost, "sets": names, "why": why}

        return resolve

    def _make_set_span_resolver(self):
        """给 agent 装的**「这个集合在世界上占的 x 范围」解析器**（用户 2026-09-27 要求 2）。

        用户原话："…且其框底边与角色玩家当前 foothold 集合的 x 范围有交集，那么就不需要下达
        寻路任务了可以直接锁" —— 要回答"我这块平台横着占多宽" ✓。

        回值：`(左, 右) | None`（世界像素；底层是 `route.set_span` ✓ —— 它复用 `_set_spans`，
        一处实现 ✓）。agent 不持有地形/集合 ⇒ 和 `route_plan` / `mob_sets_of` 一样由这里给 ✓，
        共用同一份 `_route_ctx` 2 秒缓存 ✓。
        """
        def resolve(name):
            mid = str(getattr(self, "_mmap_mid", "") or "")
            if not mid:
                return None
            t, z = self._route_ctx(mid)
            if t is None or z is None:
                return None
            from decision import route as route_mod
            try:
                return route_mod.set_span(t, z, str(name))
            except Exception:                    # noqa: BLE001
                return None

        return resolve

    def _make_toward_set_resolver(self):
        """给 agent 装的「**朝『玩家 → 怪』方向、最近的那个集合**」解析器。

        用户 2026-09-28 定的**降级路径**（原话："那就**在追击上做文章**：如果玩家的攻击范围框 x 长与
        怪框 x 长范围相交了却没有触发 attack，就下达一个前往『**玩家到怪物向量**』指向的最近一个
        foothold 集合的任务"✓；随后又补第 ③ 条："程序认为怪与玩家处于**相同** foothold 集合，
        但**攻击框 x 相交、y 不相交**（框不住怪的碰撞盒，典型矩形相交问题），**也要**逐层逼近"✓）。

        为什么要它：①「怪所在集合」那条路**要先判准怪在哪层** ✗（这几轮翻车的都是它）；
        **方向**却是可靠的（画面 1:1 已验证 ✓）⇒ 拿方向绕开层判定 ✓。
        回值：`集合名 | None`（判不出 ⇒ `None` ⇒ 调用方**不动** ✓ 宁缺勿错 ✓）。

        判据（都在这里，agent 不持有地形 ✓，共用 `_route_ctx` 缓存 ✓）：
          · **排除我脚下的集合** ✓（不然挑回自己 ⇒ 原地不动 ✗）；
          · 候选的 y 跨度必须**整段落在"怪那一侧"** ✓（`dy > 0` = 怪在**下面** ⇒ 要 `y0 >= 我`）；
          · x 要和「我 ↔ 怪」这段有交 ✓（否则会挑到地图另一头那层 ✗）；
          · 取**沿方向最近**的那个 ✓（`dy > 0` ⇒ 比 `y0`、否则比 `y1`）。
        ⚠ `|dy|` 太小（怪和我在同一高度 ✓）⇒ 返回 `None` ✓ —— 那是"纯水平"，不该上下跑 ✓。
        """
        def resolve(player, mob):
            mid = str(getattr(self, "_mmap_mid", "") or "")
            if not mid:
                return None
            t, z = self._route_ctx(mid)
            if t is None or z is None:
                return None
            from perception import minimap as mm
            from decision import route as route_mod

            sy = float(getattr(mob, "y", 0.0) or 0.0) + float(getattr(mob, "h", 0.0) or 0.0) / 2.0
            w = mm.screen_to_world(player, float(getattr(mob, "x", 0.0) or 0.0), sy)
            if w is None:
                return None
            px = float(getattr(player, "world_x", 0.0) or 0.0)
            py = float(getattr(player, "world_y", 0.0) or 0.0)
            dx = float(w[0]) - px
            dy = float(w[1]) - py
            if abs(dy) < TOWARD_MIN_DY:
                return None                     # 同一高度 ⇒ 不该上下跑 ✓
            here = set(str(s) for s in (getattr(player, "here_sets", None) or []))
            lo_x, hi_x = min(px, float(w[0])), max(px, float(w[0]))
            best, best_d = None, None
            for name in (z.sets or {}):
                nm = str(name)
                if nm in here:
                    continue                    # 排除自己脚下 ✓
                # ⛔⛔ **这里原来是 `set_span` + 四元组解包 ⇒ 每次都 IndexError** ✗✗
                #   （2026-09-28 现场修：`set_span` 只给 `(左, 右)` 两个值 ✗，而下面按
                #    `sp[2]`/`sp[3]` 取 y 范围 ⇒ 越界 ⇒ 这条"朝方向最近集合"的降级
                #    **从上线起一次都没成功过** ✓；异常又被 `agent._mob_goto_towards` 的
                #    `except` 吞掉 ⇒ **一条痕迹都没有** ✗ —— 现场现象 = "怪判不出集合时
                #    原地卡住、也不下任务"（用户："**没找到也没根据向量下达寻路任务**"✓）。
                #   ⇒ 用 `route.set_box`（集合的**包围盒** = 左/右 + 上y/下y ✓ 同一份
                #     `_set_spans` ✓）。
                box = route_mod.set_box(t, z, nm)
                if not box:
                    continue
                x0, x1, y0, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                if x1 < lo_x - 1.0 or x0 > hi_x + 1.0:
                    continue                    # x 完全不相干 ⇒ 不是"这条路上的层" ✗
                if dy > 0:
                    if y0 < py:                 # 得整个落在我下面 ✓
                        continue
                    d = y0 - py
                else:
                    if y1 > py:                 # 得整个落在我上面 ✓
                        continue
                    d = py - y1
                if best_d is None or d < best_d:
                    best, best_d = nm, d
            return best

        return resolve

    def _make_foothold_x_resolver(self):
        """给 agent 装的「**这条 foothold 的世界中心 x**」解析器（用户 2026-09-28 要求）。

        用途：「**idle 回归**」—— 区域项里的 `idle_foothold`（一条 foothold 的 **id** ✓）
        要说清"往哪边走、走到哪算到" ✓，而 agent **不持有地形** ✓ ⇒ 和 `mob_sets_of` /
        `route_plan` 一样由这里给（同一份 `_route_ctx` 2 秒缓存 ✓）。

        回值：`float`（世界 x 中心 ✓）/ `None`（没这个 id / 它是墙 / 读不到地形 ✓）。
        ⚠ **只给横向**：调用方（`agent._idle_walk_beat`）拿它算 `dx` 再按 ←/→ ✓ ——
          「**不跨层**」是那边的口径 ✓，这里不给 y ✗（要跨层是 `fight_dst` 那种"前往"的事 ✓）。
        """
        def resolve(fid):
            mid = str(getattr(self, "_mmap_mid", "") or "")
            want = str(fid or "").strip()
            if not mid or not want:
                return None
            t = self._route_ctx(mid)[0]
            if t is None:
                return None
            try:
                for f in (t.footholds or ()):
                    if str(getattr(f, "fid", "")) != want:
                        continue
                    if getattr(f, "is_wall", False):
                        return None              # 墙站不上去 ⇒ 当作没有 ✓
                    return (float(f.x1) + float(f.x2)) / 2.0
            except Exception:                    # noqa: BLE001
                return None
            return None

        return resolve

    def pos_state(self):
        """本线程的**位置状态机**（位置状态广播的**唯一写者** ✓）。

        给界面读广播事实用（例：`route_panel` 的「位置状态」那行要显示 `at_ladder_top` ✓）
        —— ⚠ **别拿它当第二个写者** ✗，界面只读 ✓（用户 2026-09-27 定的架构 ✓）。
        """
        return self._pos_state

    def _fill_route_ctx(self, player, loc, hold_vert=False):
        """把**位置状态**写进 `Player` —— ⭐ **全部交给位置状态机算**（见它的模块说明 ✓）。

        用户 2026-09-27 定："开始进行我说的**位置状态广播**架构（**所有的位置状态更新由
        位置状态机自治**）" ⇒ 这里**只做接线**：
          · 本拍的读数（`loc` ✓）与**执行器通知**（`hold_vert` 参数 ✓ —— 由调用方给的
            `agent.climbing_vertical()`：climb 执行器按住 ↑/↓ 后发起的通知，替代
            "本机按键状态"）喂进机器；
          · 机器把 `here_sets` / `ladder_id` / **`at_ladder_top`** 算好——这里照抄进 `Player` ✓
            （`Player` 就是**广播载体** ✓）；
          · 执行器**只读不算** ✗（`ClimbJob._arrived` 已经改成读 `at_ladder_top` ✓）。

        口径（**别在这里再算一遍** ✗）：
          · `here_sets` = 脚下 foothold 属于哪些命名集合（`core.zones.set_of` ✓）；
          · `ladder_id` = 贴着哪根绳（**许可入口 = climb 执行器按住 ↑/↓ 后发起的通知** ✓ ——
            用户 2026-09-27："除非按住了 ↑ 或 ↓，不能主动判定为在绳梯上，不然角色碰到绳子就
            卡住不走"；2026-09-28 诉求 1：这个"按住 ↑/↓"改由执行器通知（`climbing_vertical`），
            不再是本机按键状态 ✓）；
          · `at_ladder_top` = 上爬的**到达**判据（y ≤ 绳上端 + 「坐标对齐误差范围」且 y 在
            「移动操作尝试间隔」内没再变小 ✓，判据原文在 `perception/pos_state.py` ✓）。
        定位没输出时**一律清空**（空 = 判不出来，别拿旧值骗执行器 ✗）。
        """
        x, y = getattr(player, "world_x", None), getattr(player, "world_y", None)
        mid = str(getattr(self, "_mmap_mid", "") or "")
        # ⭐⭐⭐ **换图检测必须在最前面**（2026-09-28 现场修 ✗✗）。
        #   ⚠ 上一版把它放在"定位没输出"那条早退**之后** ⇒ **那条路根本走不到它** ✗：
        #     用户中途切过图（106010105 ⇄ 105090600 ✓ `behavior.log` 里 `task_begin dst=一楼/二楼/三楼`
        #     就是那张图的集合名 ✓），于是"沿用上一次位置状态"把**上一张图的集合名**留了下来
        #     （`_player_here = ['小平台']` ✗）⇒ 择路拿它当**起点** ⇒
        #     `core/zones.py` 报 `集合不存在：小平台` ⇒ 「回左上 / 右下→左上」**永远造不出路线**
        #     ⇒ `zone_goto_fail` 每 3 秒重试 ⇒ **卡死** ✗✗✗（用户报的"右下到左上的寻路报错了"✓）。
        #   ⇒ 三样一起清：机器内部那份 + `_player_here` + `_player_at` ✓（**同图内**才允许沿用 ✗）。
        if mid != getattr(self, "_pos_state_mid", None):
            self._pos_state_mid = mid
            self._pos_state.reset()
            self._player_here = []      # ⚠ 上一版的漏网之鱼：起点集合**也要跟着换图清** ✗
            self._player_at = None
        if x is None or y is None or not mid:
            # ⚠⚠ 2026-09-28 改（用户："位置状态给容错：**不许存在没站在平台上这种空类**，
            #   如果找不到就**按上一个位置状态**"✓）：这里原来**手动把七个字段清空** ✗
            #   ⇒ 定位掉一帧 / 判不出脚下集合 ⇒ `here_sets` 空 ⇒ 「战斗区域」判成"不在能打区"
            #   ⇒ 每拍下「回能打区」⇒ 那条路又造不出来 ⇒ **每 3 秒重试、卡死** ✗✗。
            #   ⇒ 现在**一律照抄机器的广播** ✓（`here_sets`/`here_span` 由机器**沿用上一次** ✓，
            #     "这一拍的事实"（绳/到顶/地面 y）仍然清 ✓ —— 机器内部就这么分的 ✓ 一处口径 ✓）。
            _snap = self._pos_state.update(None, loc=None, terrain=None, zones=None,
                                           hold_vert=False)
            player.here_sets = list(_snap.here_sets)
            player.here_span = _snap.here_span
            player.ladder_id = None
            player.at_ladder_top = None
            player.at_ladder_bottom = None
            player.on_rope_pos = None
            player.ground_y = None
            # ⭐ "起点集合"**也沿用**（同一份 ✓）：清掉的话「定点休息 / 回能打区」会报
            #   "你现在站的这块没圈进任何集合"⇒ 角色根本不过去 ✗（`t_route_resolver_...` 那条
            #   老用例钉的是"必须忘掉旧值"✗ —— 那条按用户 2026-09-28 的新口径**已同步翻转** ✓）。
            self._player_here = list(_snap.here_sets)
            self._player_at = None          # ⚠ 世界坐标**照旧清掉**（那是"这一拍的真实位置" ✓
            return                          #   拿旧坐标去挑最近的绳会挑错 ✗）
        t, z = self._route_ctx(mid)
        # ⭐ 诉求 1（用户 2026-09-28）：ladder_id 的许可从「**按住 ↑**」（本机按键状态）改成
        #   「**climb 执行器按住 ↑ 后发起通知**」（= 执行器上一拍发了 ↑/↓，`ClimbJob._last_vert`）。
        #   —— 执行器发键 → 远端角色真爬之间有端到端延迟/丢键，本机按键状态会跟执行器意图
        #   飘开 ✗；机器只吃**执行器通知**这个事实、自己不下发按键 ✓。
        # ⛔ **别在这儿自己去找 agent**（2026-09-28 踩的坑）：原来写成
        #   `getattr(self, "agent", None)`，而 `self.agent` **从来没人赋过值** ✗ ⇒ 恒
        #   `None` ⇒ `_hold_vert` 恒 `False` ⇒ `pos_state.ladder_id` **恒 `None`** ⇒
        #   执行器永远判不出"在绳上"（爬绳全废 / 位置状态判定失误 ✗ —— 用户 2026-09-28
        #   报的"**之前没有限制绳梯判定入口的时候是很准确及时的**"正是它 ✓，与几何尺
        #   `mapdata.LADDER_PAD` **无关** ✗）。⇒ 改成**调用方显式传进来** ✓：`run()` 里
        #   那个局部 `agent` 才是**真跑 tick** 的那个 ✓。
        _hold_vert = bool(hold_vert)
        # 设置**每拍现取**（用户随时会改「坐标对齐误差范围」/「移动操作尝试间隔」⇒ 下一拍生效 ✓）
        from decision.agent import settings as _stg
        # ⭐ **位置状态机自治**（用户 2026-09-27 ✓）：算好 → 直接广播到 `Player` ✓
        # ⚠ 世界坐标以 `Player` 上那份为准（就是定位写进去的 ✓）：把 `x/y` 并进 `loc` 再喂机器 ——
        #   机器只认"读数里带没带世界坐标"，这样它的入参**自洽**（不依赖调用方另传一份 ✓）。
        _snap = self._pos_state.update(
            None, loc=dict(loc or {}, world_x=x, world_y=y),
            terrain=t, zones=z, hold_vert=_hold_vert,
            align_tol_px=float(getattr(_stg, "align_tol_px", 10) or 10),
            move_retry_ms=float(getattr(_stg, "move_retry_ms", 3000) or 0),
            # "集合名 → (左, 右)"的查询：**机器在感知层，不 import 决策层** ✗ ⇒ 由这边
            # 把自己那个闭包喂进去（`route.set_span` = "集合里有哪些非墙 foothold"的
            # **唯一**口径 ✓，见 `_make_set_span_resolver` ✓）。
            # ⚠ `getattr` 兜底：用例里有时候拿个替身线程跑这一路（没有那个方法 ✓），
            #   宁可"这拍没有平台宽度"也不要当场 AttributeError 把整条链弄掉 ✗。
            span_of=self._make_set_span_resolver()
            if hasattr(self, "_make_set_span_resolver") else None)
        # ⭐ 快照**展开**进 `Player` 的七个扁平字段（它们才是唯一存储 ✓；要"整份"时用
        #   `PosSnapshot.of(player)` 在**一处**拼 ✓）—— 用户 2026-09-27："重新整理…**状态通信**、
        #   提高内聚降低耦合" ✓：字段名与快照字段**一一对应**，加字段只动 `PosSnapshot` ✓。
        player.here_sets = list(_snap.here_sets)
        player.ladder_id = _snap.ladder_id
        player.at_ladder_top = _snap.at_ladder_top
        player.at_ladder_bottom = _snap.at_ladder_bottom
        player.ground_y = _snap.ground_y
        player.on_rope_pos = _snap.on_rope_pos
        player.here_span = _snap.here_span
        player.climb_failed = _snap.climb_failed
        player.climb_stalled = _snap.climb_stalled
        # 「我现在站哪个集合」也给**路径解析器**留一份（`_make_route_resolver` 要用它当起点）。
        # ⚠ 2026-09-26 漏了这一步 ⇒ 解析器只能退回 `settings.route_goto_set`（那是路线面板
        #   在跟踪玩家时才写的东西 ✗），于是「定点休息」经常报"你现在站的这块没圈进任何集合"
        #   ⇒ **根本不过去** ✗（用户在右下点手动休息，角色原地不动就是这么来的）。
        self._player_here = list(player.here_sets)
        # 玩家世界坐标也留一份：**择路**要用它（"到目标集合有多条绳 ⇒ 挑最近的"，
        # 见 `decision.route.pick_edge` ✓）。解析器是闭包、拿不到每一帧的 WorldState，
        # 所以和 `_player_here` 一样存在线程上（都是"最近一次的定位结果" ✓）。
        self._player_at = (float(x), float(y))

    def _locate_mmap(self, panel):
        """这一拍定位一次玩家 → 结论 dict；没在用/没地图 → None。

        `panel` 是 `_mmap_panel_from_frame` 给的（来源=从实时画面）；来源=收流时
        这里自己去小地图推流那一路取最新帧（断线由 `MiniMapClient` 自己重连）。
        """
        mid = self._mmap_mid
        if not mid:
            return None
        from perception import minimap as mm
        if self._locator is None:
            self._locator = mm.PlayerLocator(
                mid, track=mm.tracker_kwargs(self._mmap_track or {}))
            self._track_applied = dict(self._mmap_track or {})
        elif self._locator.map_id != mid:
            self._locator.load(mid)
        # 界面上改了黄点容差 → **在这一拍之前对齐**（比对是 4 个数，每拍做一次
        # 无所谓）。注意应用它会重建 tracker（跨帧状态清零）—— 那正是"人改了参数"
        # 该有的效果，所以只在真的变了才做。
        if self._mmap_track is not None and self._mmap_track != self._track_applied:
            self._locator.use_track_config(self._mmap_track)
            self._track_applied = dict(self._mmap_track)

        src = self._mmap_src
        if src != "live":
            if self._mmap_cli is None:
                try:
                    self._mmap_cli = mm.MiniMapClient(
                        get("a_host"), port=get("minimap", "port", 5003)).start()
                except Exception as e:                      # noqa: BLE001
                    return {"ok": False, "note": "小地图推流起不来：%s: %s"
                            % (type(e).__name__, e)}
            panel, _t = self._mmap_cli.latest()
            if panel is None:
                return {"ok": False, "note": "还没收到小地图推流（%s）"
                        % (self._mmap_cli.err or "A 机那一路起了吗？")}
        elif panel is None:
            return {"ok": False, "note": ("没有可裁的小地图区域 —— %s"
                                          % ("先框选小地图" if not self._mmap_crop
                                             else "框选区超出当前画面，重框一次"))}
        # 「脚下的 foothold」也按**设置里的「坐标对齐误差范围」**兜一次（用户 2026-09-26）：
        # 站在平台边上时黄点会读到边界外几个像素，而老口径 x 一点容差都没有 ⇒ 报
        # "脚下没有平台 / 未分组" ⇒ 起点与到达判定全废 ✗
        # （实测 105090600 的 (650,283)：离平台左边界 7px，而设置的容差是 10px ✓）。
        from decision.agent import settings as decision_settings
        loc = self._locator.update(
            panel, src=src,
            fh_xtol=int(getattr(decision_settings, "align_tol_px", 0) or 0))
        # ⭐ 局部小地图（crop）的 view 跟踪成败**打点**（用户 2026-09-29 任务 1 ✓）：
        #   它不是 crop 时是 `None`（没这回事，不打点 ✓）。"跟不住"占比高 ⇒ 面板被挡 /
        #   方式选错 / 标定不对 ⇒ 那一拍的世界坐标是拿标定里那个位置算的（可能整体偏 ✗）。
        if loc.get("view_ok") is not None:
            from core import perf
            perf.count("crop_view_ok" if loc.get("view_ok") else "crop_view_miss")
        return loc

    # ---------------- 控制 ----------------

    def stop(self):
        self._stop.set()
        # 小地图定位那条 TCP（来源=收流 才起）：它自带一条接收线程，**必须显式收掉**
        # —— 留着会在工作台退出时跟 Qt 主线程抢着析构，表现是无征兆闪退（同
        # closeEvent 里那两条注释）。
        cli = self._mmap_cli
        if cli is not None:
            try:
                cli.stop()
            except Exception:
                pass

    def stopped(self):
        return self._stop.is_set()

    def set_infer(self, enabled):
        """开启/关闭推理。开启后主循环才跑 YOLO + 决策 + 画框。"""
        if enabled:
            self._infer.set()
        else:
            self._infer.clear()

    # ---------------- 主循环 ----------------

    def run(self):
        # 这条线程就是关键回路（收流 → 推理 → 决策），单独把**线程**优先级提一档：
        # Windows 会给「前台进程的线程」额外的调度优待，失焦时被压下去的往往正是
        # 这条要 10 ms 一拍、还不能迟到的回路。进程级的兜底见 core/winperf.py
        # （启动时已应用；这里再加一层，是因为线程优先级和进程优先级是两回事）。
        try:
            from core import winperf
            winperf.boost_thread()
        except Exception:
            pass          # 提不上去也要照常跑，绝不因此不启动
        try:
            self._run()
        except Exception as e:
            if not self._stop.is_set():
                self.failed.emit("%s: %s" % (type(e).__name__, e))

    def _run(self):
        import cv2

        url = self._p.get("url", "")
        # 怪物/玩家置信度分开配：推理时用两者的较小值（保证两类的低分框都
        # 进结果），后处理再按各自阈值过滤 —— 一次推理，不额外耗性能。
        conf_mob = float(self._p.get("conf_mob", 0.30))
        conf_player = float(self._p.get("conf_player", 0.50))
        conf = min(conf_mob, conf_player)
        imgsz = int(self._p.get("imgsz", 960))
        device = str(self._p.get("device", "0"))
        weights = self._p.get("weights") or ""
        draw = bool(self._p.get("draw", True))
        show_fps = max(1.0, float(self._p.get("show_fps", 30.0)))
        source = self._p.get("source", "stream")
        origin_x = 0      # 画面原点（屏幕坐标）：窗口模式下是框选区域左上角，
        origin_y = 0      # HP/MP 条框选的屏幕坐标要减去它转成画面内坐标

        from ultralytics import YOLO
        model = None       # 懒加载：首次「开始推理」才加载，收画面阶段不加载

        # 决策：找怪打。agent 跨帧持有按键状态，这里只建一次；
        # settings 是全局单例，和「决策参数」页签共享。
        # 注意：输入设备后端（本地/Pro Micro）由 player_panel 切换时设置，
        # 这里不再重复 use_network —— 否则会二次连接、旧连接泄漏、还可能
        # 因 relay 暂时不可达把已建好的连接回退成本地。
        from decision.agent import CombatAgent, settings as decision_settings
        from decision import agent as agent_mod
        from decision.input import tap as _tap
        from decision.reconnect import Reconnector
        from perception.tracker import MobTracker, PlayerTracker
        from perception.world_state import WorldState
        from perception import minimap as mm     # 小地图定位（S3）：apply_to_player 等

        agent = CombatAgent(decision_settings)
        # 登记成"当前 agent"：「路线识别」页的「命令前往」要给它挂**上绳/下跳任务**
        # （`start_climb`），而它原本只是本函数的局部变量 ⇒ 面板拿不到（见 decision/agent
        # 的 CURRENT）。只做引用赋值，不加锁 —— 和 settings 一个路子；退出时注销（见 finally）。
        agent_mod.CURRENT = agent
        # ⭐ **本线程也留一份引用**（`self.agent`）—— 有两处读它：`_mob_query_ttl`
        #   （「前往重下间隔」那把尺）与 `_fill_route_ctx` 里给位置状态机的 `hold_vert`。
        #   ⛔ **2026-09-28 修**：这两处一直写 `self.agent`，却**从来没人赋过值** ✗ ⇒
        #   `getattr(self, "agent", None)` 恒 `None` ⇒ `_hold_vert` 恒 `False` ⇒
        #   `pos_state.ladder_id` **恒 `None`** ⇒ 执行器永远判不出"在绳上"（爬绳全废、
        #   位置状态判定失误 —— 用户 2026-09-28 报的"**之前没有限制绳梯判定入口的时候是
        #   很准确及时的**"正是它 ✓，**不是** `mapdata.LADDER_PAD` 那把几何尺 ✗）；
        #   顺带 `_mob_query_ttl` 也恒走 except ⇒ 设置里「前往重下间隔(s)」实际不生效 ✗。
        #   一处赋值两处都好 ✓（退出时注销，见 finally ✓）。
        self.agent = agent
        # 给 agent 装**路径解析器**：「定点休息」要用它自己走到指定地点。
        # agent 不持有地形 / 集合 ⇒ 解析这一层由这里提供（和 `_fill_route_ctx` 同一处）。
        agent.route_plan = self._make_route_resolver(decision_settings)
        # 再装一个「**这只怪在哪块平台上**」的解析器（用户 2026-09-27 要求：chase 前先判怪的
        # foothold 集合，不与玩家同一块 ⇒ 先下「前往」任务 ✓）。agent 同样不持有地形/集合，
        # 所以和上面那条一样由这里给（同一份 `_route_ctx` 缓存 ✓）。
        agent.mob_sets_of = self._make_mob_sets_resolver()
        # 再装一个「**走过去要多少寻路距离**」的解析器（用户 2026-09-27："锁定目标优先级要按
        # 「寻路距离」最近，而不是旧的应该是绝对距离"）—— 它**复用上面那一个**做"画面→世界 +
        # 落在哪块面"（一次换算 ✓），只多一步 `route.path_cost`（择路同一套口径 ✓）。
        agent.mob_cost_of = self._make_mob_cost_resolver(agent.mob_sets_of)
        # ⭐ **再装一个「集合 → 集合」的代价**（用户 2026-09-28 ✓）：人在**不能打**的平台时，
        #   要挑一块**代价最低的可战斗区**（`agent._pick_battle_zone` ✓）。底层是**同一个**
        #   `route.path_cost`（口径一处 ✓）。没装 ⇒ 那边退回"回第一条"的老行为 ✓（不猜 ✗）。
        agent.set_cost_of = self._make_set_cost_resolver()
        # ⭐ 再装一个「**这条 foothold 的世界中心 x**」的解析器（用户 2026-09-28：「idle 回归」
        #    要"位于本集合时**水平走**向 `idle_foothold` 的中心" ✓）—— agent 同样不持有地形 ✓
        #    （同一份 `_route_ctx` 缓存 ✓）。没装 ⇒ agent 那边什么都不做（照旧站住 ✓）。
        agent.foothold_x = self._make_foothold_x_resolver()
        # ⭐ 再装一个「**朝『玩家 → 怪』方向、最近的那个集合**」的解析器（用户 2026-09-28 定稿 ✓）
        #    —— 给追击的**降级路径**用（怪那层判不出 / 走不到 / 判成"同一集合"但攻击框与怪框
        #    不相交时，改成"朝方向逐层逼近"✓，绕开"层判定"这个最不稳的环节 ✓）。
        agent.nearest_set_towards = self._make_toward_set_resolver()
        # ⚠ 这里原来把「某个集合在世界上占多宽（x 范围）」的解析器挂到 **agent** 上
        # （`agent.set_span_of`），让 `_reachable_without_path` **自己**算平台宽度 ——
        # 2026-09-27 用户要求**收编**（"同一件事两处算"是当天连踩两次的根因 ✗）⇒ **已删** ✗：
        # 这份查询现在喂给**位置状态机**（`_fill_route_ctx` 里的 `span_of=` ✓），
        # 决策层一律读广播的 `player.here_span` ✓（解析器本身仍是 `route.set_span` 那一份 ✓）。
        mob_tracker = MobTracker()   # 给怪稳定 id，供目标锁定 CD 跨帧匹配
        player_tracker = PlayerTracker()   # 跟住「我」：位置连续性，别把别人认成自己
        # 断线重连状态机（判界面 → 停自动 → 按 Enter/点鼠标走回游戏）
        reconnector = Reconnector(decision_settings)

        # 玩家也走 YOLO（不再是模板匹配）。每个玩家独占一个 YOLO 类，实时
        # 只取当前 player_id 对应的类；当前固定 class 0，多玩家时扩展
        # PLAYER_CLASS_MAP。
        _pid = self._p.get("player_id") or ""
        _player_class = int(self._p.get("player_class",
                                        PLAYER_CLASS_MAP.get(_pid, 0)) or 0)
        # 怪物类 id：默认按类别表（perception/classes.py）；模型加载后会改成以
        # **模型自己声明的类别名**为准，见下面 YOLO(weights) 那一段。
        _mob_class = CLASS_MOB

        # HP/MP 条识别：从实时帧 vis 里按「画面比例」裁出血条区域，再颜色过滤
        # + 列扫描算填充比例，不额外抓屏。限流 0.1s（识别本身 <1ms，限流只为抑制抖动）。
        _pot_last = [0.0]
        _pot_vals = [1.0, 1.0]      # (hp, mp) 比例缓存
        _last_player = [None]       # 上一帧玩家框 (cx, cy, bottom, conf)，漏检时兜底
        #: ⭐ 「玩家位置」组的**框面积滚动基线**（用户 2026-09-28 ✓）：最近若干拍的框面积 ✓。
        #:   上限写死 600（设置里的窗口 `player_box_area_base_n` **只在算均值时切片** ✓
        #:   ⇒ 改设置立刻生效，不用重建这颗缓冲 ✓）。
        _area_hist = []
        #: 最近一次**端到端延迟**（毫秒，探针解出来的）；写进 WorldState 给"对齐保持窗口"
        #: 用（见 decision/agent._climb_tick：窗口 = 设置里的保持时间 + 它）。
        _e2e_ref = [0.0]
        #: 视野矩形 (left, top, right, bottom) —— **每帧现算** ✓（见收流循环里那段说明：
        #: 2026-09-27 用户报"视野范围更新太慢"，根因就是它原来是**每 3 秒**才算一次 ✗）。
        _vision_box = [None]

        # 可视化配置：定期重读（颜色改了实时生效，不用重开实时预览）
        # ⚠ 这几个"框/线"的颜色都走 `hex_to_bgra`（**可能带透明度** ✓，2026-09-27 起
        #   用户能在颜色弹窗里用拖动条调）：`_blit_alpha` 按 alpha 决定"直画还是混色" ✓。
        _vis_cfg = theme.load_vis()
        _cls_colors = _box_colors()
        _lock_color = theme.hex_to_bgr(_vis_cfg["lock_color"])
        _attack_color = theme.hex_to_bgra(_vis_cfg["attack_color"])          # 攻击范围框
        _min_attack_color = theme.hex_to_bgra(_vis_cfg["min_attack_color"])  # 攻击盲区框
        _chase_jump_color = theme.hex_to_bgra(_vis_cfg["chase_jump_color"])  # 追击起跳框
        _vision_color = theme.hex_to_bgra(_vis_cfg["vision_color"])
        _vision_width = int(_vis_cfg["vision_width"])
        # 每项的**显示开关**（用户 2026-09-27："辅助线与标记组里每项参数前加开关"）——
        # 关掉 = 这一项不画（颜色还留在配置里 ✓）。⚠ 「跳跃攻击范围框」这里没有对应的
        # 局部变量：它的**逻辑还没做**（占位项，见 `decision/agent.py` 那条说明）✓。
        _on_lock = bool(_vis_cfg.get("lock_on", True))
        _on_attack = bool(_vis_cfg.get("attack_on", True))
        _on_blind = bool(_vis_cfg.get("min_attack_on", True))
        _on_chase = bool(_vis_cfg.get("chase_jump_on", True))
        _on_vision = bool(_vis_cfg.get("vision_on", True))
        _vis_refresh_last = 0.0

        # 读线程独立于推理：它拼命读，积压的旧帧在槽位里被直接覆盖丢掉。
        # 这样无论推理多慢，看到的都是最新画面，延迟不会累积。
        #
        # 两种来源共用同一个 slot + 推理主循环，只有 reader 的取帧方式不同：
        #   stream —— PyAVSource 收流（阻塞读，按序解）
        #   window —— wincap.grab_rect 循环抓本地窗口区域
        # 性能打点：关键路径的耗时留在仓库根 perf.log 里，事后直接看，不用现场加
        # print（开销见 core/perf.py 的说明：常开也只有纳秒级 + 每 30 秒一次落盘）。
        # 开关在「设置 → 性能日志」（存 config/live.yaml 的 perf_log），不在这里。
        from core import behavior, perf
        _b_on = bool(self._p.get("perf_log", True))
        perf.configure(_b_on)
        # 玩家行为打点（`core/behavior.py`）：初版只打"玩家的任务"事件，落 behavior.log。
        # 开关**跟性能日志共用**（`perf_log`）—— 用户没要求单开一格；将来要分开就加一格参数。
        behavior.configure(_b_on)

        slot = _LatestSlot()
        reader_done = threading.Event()

        # ---- 本机负载守望：**另起一条线程** ----
        # 它回答的是「为什么上面那些数字变差了」，所以要在推理的同时常备，又
        # **绝不能拖慢推理**：psutil 清点进程要 10~50 ms，放进主回路就是每 10 秒
        # 卡一帧 —— 那正好污染我们在盯的 infer_ms / pipe_ms，成了自己测自己。
        from core import machineload

        def _load_watch():
            while not self._stop.is_set():
                try:
                    d = machineload.probe()
                    self._load = (machineload.warn(d), machineload.detail(d))
                    # 顺手进性能日志：下次再遇到「推理为什么慢」，直接和 infer_ms
                    # 对齐着看就行，不必靠人工回忆当时机器上还跑着什么。
                    if d["avail_gb"] is not None:
                        perf.sample("avail_gb", d["avail_gb"])
                    if d["has_psutil"]:
                        perf.sample("mp_workers", float(d["workers"]))
                        perf.sample("mp_worker_mb", d["worker_mb"])
                except Exception as e:
                    # **不许静默 pass**：清点一直失败却什么都不显示，就等于这个
                    # 功能根本没做（而它又是「推理为什么慢」的唯一解释）。显示出来
                    # 才会有人去修；下一轮成功时它自己就恢复了。
                    self._load = ("负载清点失败", "本机负载清点抛异常，这个功能现在"
                                  "没有在工作（不影响收流与推理）：\n%s: %s"
                                  % (type(e).__name__, e))
                self._stop.wait(LOAD_WATCH_SEC)   # 停止时立刻返回，不用等满 10 秒

        threading.Thread(target=_load_watch, daemon=True).start()

        src = None
        size_ref = [None]      # (w, h)，第一帧后填，给统计条显示分辨率
        fps_ref = [0.0]
        # 帧预算（秒）= 输入帧间隔。**两个来源都必须赋值**，别只在窗口分支里定义 ——
        # 之前只在 window 分支算 `interval`，推流模式下主循环引用它就
        # UnboundLocalError（点「开始推理」必炸）。推流用流的 fps，窗口用抓帧 fps；
        # 拿不到 fps 就置 0，表示「不知道预算」，此时不统计超预算（宁可不报，别误报）。
        budget = 0.0

        def _boost_reader():
            """读线程也提一档优先级 —— **它是输入侧唯一的兜底**。

            为什么必须做（2026-09-26 现场："有时候会跑到 3000+ms 延迟"）：
            `_LatestSlot` 丢的是"**已经解出来**的旧帧"，而积压真正待的地方在它上游 ——
            ffmpeg 的解码队列和内核 UDP 缓冲。读线程一旦被饿住（推理/绘制占满 CPU、
            或抢 GIL），上游就堆起来，之后它再一帧帧慢慢啃回来 ⇒ 表现就是**延迟冲到
            几秒、再缓慢回落**，而且"有时候"才出现（机器忙的时候）。
            主回路早就提了优先级（见 run()），读线程原来没提 —— 它同样是 1/帧 的硬期限。
            """
            try:
                from core import winperf
                winperf.boost_thread()
            except Exception:
                pass          # 提不上去也要照常读，绝不因此不启动

        if source == "window":
            from core import wincap
            from link.frame import Frame

            rect = self._p.get("rect")
            if isinstance(rect, str):
                rect = tuple(int(v) for v in rect.split(","))
            rect = tuple(int(v) for v in rect)
            if len(rect) != 4 or rect[2] <= 0 or rect[3] <= 0:
                self.failed.emit("窗口区域无效：%s" % (rect,))
                return
            origin_x, origin_y = rect[0], rect[1]   # 画面内坐标 = 屏幕坐标 - 原点

            capture_fps = max(1.0, float(self._p.get("capture_fps", 15.0)))
            interval = 1.0 / capture_fps
            fps_ref[0] = capture_fps
            budget = interval
            frame_id = [0]

            def _reader():
                _boost_reader()
                try:
                    while not self._stop.is_set() and not reader_done.is_set():
                        t0 = time.perf_counter()
                        img = wincap.grab_rect(rect)
                        frame_id[0] += 1
                        if size_ref[0] is None:
                            size_ref[0] = (img.shape[1], img.shape[0])
                        slot.put(Frame(
                            image=img, frame_id=frame_id[0],
                            t_recv_wall=time.time(),
                            t_recv_mono=time.perf_counter()))
                        d = interval - (time.perf_counter() - t0)
                        if d > 0:
                            time.sleep(d)
                except Exception as e:
                    if not self._stop.is_set():
                        self.failed.emit("窗口抓帧失败：%s" % e)
                finally:
                    reader_done.set()
        else:
            from link import PyAVSource

            # 直接 open：udp 有 timeout（5 秒），A 没推流时 open 会超时抛 I/O error，
            # 不会无限黑屏。用 open 的错误类型区分「没推流」和「收到包但解码失败」。
            self.stream_status.emit("waiting")
            # BGR 直出，省一次颜色转换
            src = PyAVSource(url, decode_format="bgr24",
                             container_format=self._p.get("format"))
            try:
                src.open()
            except Exception as e:
                if self._stop.is_set():
                    return
                self.stream_status.emit("no_stream")
                msg = str(e).lower()
                if "timed out" in msg or "etimedout" in msg or "i/o error" in msg:
                    self.failed.emit(
                        "没收到流 —— A 机没在推流？检查 OBS 是否在推、"
                        "URL/IP/端口、B 机防火墙是否放行 UDP")
                else:
                    self.failed.emit("收到包但打不开流：%s" % e)
                return
            self.stream_status.emit("connected")
            size_ref[0] = src.size
            fps_ref[0] = src.fps or 0.0
            budget = 1.0 / fps_ref[0] if fps_ref[0] else 0.0

            def _reader():
                _boost_reader()
                try:
                    while not self._stop.is_set() and not reader_done.is_set():
                        try:
                            f = src.read()
                        except TimeoutError:
                            continue   # 暂时没帧（udp 超时），继续等，能及时响应停止
                        if f is None:
                            break
                        if size_ref[0] is None:
                            size_ref[0] = (f.image.shape[1], f.image.shape[0])
                        slot.put(f)
                except Exception:
                    pass
                finally:
                    reader_done.set()

        # 段头注记：同样的毫秒数，在不同预算/设备/分辨率下含义完全不同，
        # 日志自己把条件带上，事后评估才不用猜。**放在两个分支之后** —— 推流模式下
        # fps 要等 open() 才知道，写在分支里会漏掉（当初就漏了）。
        perf.note("budget_ms", round(budget * 1000.0, 1) if budget else "未知")
        perf.note("fps", round(fps_ref[0], 1) if fps_ref[0] else "未知")
        perf.note("source", source)
        perf.note("imgsz", self._p.get("imgsz", 640))
        perf.note("device", self._p.get("device", ""))

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        show_interval = 1.0 / show_fps
        t_start = time.perf_counter()
        t_last_show = 0.0
        t_last_stat = t_start

        n = 0
        n_show = 0
        n_boxes = 0
        infer_ms = []
        gaps = []
        last_mono = None
        # 上一次统计窗口的 [收帧数, 处理数, 丢弃数]，用来算**窗口内**的速率
        # （累计平均会把「刚刚开始恶化」抹平）
        _prev = [0, 0, 0]
        # 积压锁死看门狗的状态：[起算时刻, 峰值ms, 已报过]（见 lag_watchdog）
        _lag_state = [None, None, False]

        # 端到端延迟：解码 A 机屏幕上的时间码探针。
        #
        # **为什么必须用探针，而不是拿 pts 推算**
        #   pts 是 A 机编码器给的时间戳。拿它和"第一帧到达时刻"对齐，
        #   测出来的其实只是网络抖动 —— 编码器前面攒了多久，基线对齐时
        #   被整个抵消掉了。表现就是监控显示"延迟几毫秒"，体感却滞后几百毫秒。
        #
        #   探针是唯一可靠的基准：它是**画在画面里的绝对时刻**，
        #   跟着画面一起被采集编码传输，B 机解出来和本地时钟一比就是真实延迟。
        #   前提是 B 机做过时钟同步（tools.clock_sync --host <A机IP> --save）。
        # 本地窗口抓取没有跨机传输，探针测的延迟无意义，强制关闭
        probe_on = bool(self._p.get("probe", False)) and source == "stream"
        # 回退几何：link.yaml 里的绝对像素（没有人工标定时用）
        probe_px = (float(self._p.get("probe_x", 100)),
                    float(self._p.get("probe_y", 8)),
                    float(self._p.get("probe_cell", 16)),
                    float(self._p.get("probe_gap", 2)))
        bits = int(self._p.get("probe_bits", 40))

        # 探针几何：**项目标定优先**（decision_settings.probe_calib，按项目存 ——
        # 不同项目分辨率/客户端布局可能不同），再退到旧的全局标定文件，最后是
        # link.yaml 的像素值。每秒重读一次：框选完不用重开预览就能生效。
        _calib = [0.0, None]        # [上次重读时刻, (标定值, 来源)]
        _calib_src = [None]         # 上一次记进 perf 注记的来源

        def _probe_geom(shape):
            now = time.perf_counter()
            if now - _calib[0] >= 1.0:
                _calib[0] = now
                try:
                    _calib[1] = probe_codec.pick_calib(decision_settings.probe_calib)
                except Exception:
                    _calib[1] = (None, "配置值")
                src = _calib[1][1]
                if _calib_src[0] != src:
                    _calib_src[0] = src
                    perf.note("probe", src)     # 段头能看到这一段的几何是哪来的
            cal = _calib[1][0]
            if cal:
                return probe_codec.calib_to_px(cal, shape)
            return probe_px
        offset_ms = float(self._p.get("clock_offset_ms", 0.0) or 0.0)

        delays = []
        probe_miss = 0           # 解码失败（连码都解不出来）
        probe_invalid = 0        # 解出来了、但值是**不可能的**（≥ 一天）＝几何错了
        probe_ok = 0             # 解出了合法值（含下面被丢弃的）
        probe_reject = 0         # 值合法、但**算不出延迟** —— 双机时钟没对齐
        #: 跨帧单调性判据：判「这份几何可不可信」。**与 tools/probe_tune 共用一份
        #: 实现**（probe_codec.Verdict）—— 各写一份必然漂移，而漂移会造出"工具说
        #: OK、工作台说不对"这种最难查的分歧。
        #:
        #: 为什么非要它：`ts_plausible` 只问"解出的时刻接近现在吗"，而 40 位码的
        #: **低位是快变化的毫秒**，错位采样只把低位采错，凑出一个仍在 ±10 分钟
        #: 窗口里的值完全可能。实测（2026-09-25）：link.yaml 的几何比实际画的小
        #: 2.8px/块、42 块累积偏 118px，低位全错，解出的却"看着合法"，于是延迟被
        #: 报成一坨乱数（p95 4.7s，紧贴自己的 --max-delay 上限）。
        probe_vd = probe_codec.Verdict()
        probe_jumpy = 0          # 判据判成「乱跳」的帧数（>0 就是几何可疑）
        _geom_key = [None]       # 上一次用的几何；变了就重置判据（见下）
        #: [上次重读时刻, 当前偏移毫秒] —— `_clock_ms()` 用。**别删**：上面那次
        #: 加判据的编辑曾把它整行吃掉，现象是开预览就 `NameError: name '_clock'
        #: is not defined`（pyflakes 一跑就能看见）。
        _clock = [0.0, offset_ms]

        def _clock_ms():
            """当前该用的时钟偏移（毫秒）。

            启动时对过一次时，但两台机器的 NTP 各漂各的 —— 一旦漂出
            `resolve_delay_ms` 能接受的范围（±1 分钟内），现象就是
            「探针解码正常、却算不出延迟」，而界面上只写「解不出」。
            所以每 5 秒从文件重读一次：在别处重新对时后自动生效，不用重开预览。
            """
            now = time.perf_counter()
            if now - _clock[0] >= 5.0:
                _clock[0] = now
                try:
                    from tools.config import ROOT
                    _clock[1] = float((ROOT / "config" / "clock_offset.txt")
                                      .read_text(encoding="utf-8").strip())
                except Exception:
                    pass        # 读不到就沿用旧值（别把 0 写进去）
            return _clock[1]

        if probe_on:
            # 自动对时：offset 会随 Windows NTP 漂移，后台线程里重新对一次，
            # 失败就沿用 params 传进来的旧值。
            try:
                from tools.clock_sync import sync_offset
                off = sync_offset(save=True, quiet=True)
                if off is not None:
                    offset_ms = off
                    _clock[1] = off     # 立刻生效，不用等下一次重读
            except Exception:
                pass
            # 解码模块在**模块级**导入（见文件头说明）：这里只判"有没有"，
            # 别再局部 import —— 那会把 probe_codec 变成局部名，函数里更早的
            # 引用会 UnboundLocalError。
            if probe_codec is None or decode_ms is None:
                self.failed.emit("探针解码模块不可用（tools.probe_codec 导入失败）")
                probe_on = False

        # 时序拍的最小间隔：序列到点时主循环至少按这个粒度醒来一次（见 agent 的
        # TIMING_TICK）。序列计时走本机绝对时钟，这个粒度对 CD/delay 足够。
        from decision.agent import TIMING_TICK as _timing_tick
        ws = None            # 上一次的世界状态：时序拍不读它，只做兜底

        try:
            while not self._stop.is_set():
                # 等「下一帧」或「下一个序列到点时刻」，谁先到就干谁 ——
                # 序列里存的是绝对到点时刻，原来只有抓到一帧才检查「到点没」，
                # 于是输出CD/delay/跳间隔全被量化到 1/capture_fps。现在到点就醒。
                timeout = 0.5
                deadline = agent.next_deadline()
                if deadline is not None:
                    timeout = min(timeout,
                                  max(_timing_tick, deadline - time.perf_counter()))
                f = slot.take(timeout)
                _t_take = time.perf_counter()
                if f is None:
                    if reader_done.is_set():
                        break          # 流结束了
                    _t = time.perf_counter()
                    agent.tick_timing(ws)   # 时序拍：只发「到点该发」的，不做画面决策
                    perf.ms("timing_ms", _t)
                    perf.count("timing_calls")
                    continue           # 只是暂时没新帧，继续等
                n += 1

                if last_mono is not None:
                    _gap = (f.t_recv_mono - last_mono) * 1000.0
                    gaps.append(_gap)
                    perf.sample("gap_ms", _gap)
                    if len(gaps) > 60:      # 下面只用得到最近 60 个（见 stats_ready）
                        del gaps[0]
                last_mono = f.t_recv_mono
                # ⭐ **这一帧在槽里等了多久**（用户 2026-09-29 链路提效 ✓）：
                #   `_LatestSlot` 只丢"已经解码出来"的帧 ⇒ 回路一慢，帧就在这儿**变旧**
                #   ⇒ 这是"控制延迟"里 B 机自己造的那一段，和 `e2e_probe_ms`（上游）合起来
                #   才能把闭环拆开：`闭环 ≈ 上游 + 槽等待 + pipe + 发送 + A 机执行` ✓。
                #   ⚠ 用 `t_recv_mono`（读线程解码完成那一刻，同一个 `perf_counter` 时钟 ✓）
                #     —— 取负说明时钟被跳过 ⇒ 夹到 0，别把负数喂进分位数 ✗。
                perf.sample("slot_wait_ms",
                            max(0.0, (_t_take - float(f.t_recv_mono)) * 1000.0))
                _t_pipe = time.perf_counter()   # 「收到这一帧 → 决策完」的总耗时
                perf.frame_arrived(_t_pipe)     # 记下这帧被取走的时刻（算 out_key_ms）
                vis = f.image          # BGR（decode_format="bgr24" 直出）
                # **再留一份原生帧**（`raw_frame_ready` 发出去的那份）：下面会往 `vis`
                # 上**就地**画玩家蓝框/怪物绿框/攻击线/**视野虚线**，而取帧做测量的那几个
                # 功能（探针标定、HP/MP 条框选、小地图框选、标定弹窗、叠图核对）量的都是
                # **像素**，被画过就全毁 —— 用户 2026-09-27 现场正是这么撞上的
                #（"框选小地图时把视野灰色虚线一起框进去了"）。
                # ⚠ 为什么是"先拷再画"而不是"不画"：cv2 的绘制是**原地**改数组，
                #   `vis` 一旦画过，原始像素就找不回来了（下面 `_mmap_panel_from_frame`
                #   那条注释记的是同一个坑，只是它只护住了小地图那一小块）。
                # ⚠ 只在**真要画**的时候拷：本文件里所有 `cv2.*(vis, …)` 都在下面
                #   `if self._infer.is_set():` 那一块里 ⇒ 收画面阶段零开销 ✓
                #   （用例 `t_raw_frame_before_draw` 钉着这条，加绘制时顺手看它一眼）。
                # 代价：一次全帧 memcpy（1920×1080 ≈ 0.5 ms），在**收流/标注线程**里，
                # 不在决策那一条链上 ✓。
                raw = vis.copy() if self._infer.is_set() else vis
                # 小地图那块面板**趁现在裁**（复制一份）：下面会往 vis 上就地画
                # 玩家蓝框/攻击线，画过的像素会串进黄点识别里。见
                # _mmap_panel_from_frame 的说明。来源=收流时这里是 None（那一帧
                # 来自 A 机单独那一路，和主画面无关）。
                _mmap_panel = self._mmap_panel_from_frame(vis)

                # 定期重读可视化配置：标记颜色/线宽改了实时生效
                if time.perf_counter() - _vis_refresh_last >= 1.0:
                    _vis_refresh_last = time.perf_counter()
                    _vis_cfg = theme.load_vis()
                    _cls_colors = theme.class_colors()
                    # 画框/标签按**实际解析出来的**类 id 取（权重可能在别处训、
                    # 顺序和我们不同），取不到就回退到类别表里的默认项。
                    _p_color = _cls_colors.get(_player_class,
                                               _cls_colors[CLASS_PLAYER])
                    _m_color = _cls_colors.get(_mob_class,
                                               _cls_colors[CLASS_MOB])
                    _p_label = CLASS_NAMES.get(_player_class,
                                               CLASS_NAMES[CLASS_PLAYER])
                    _m_label = CLASS_NAMES.get(_mob_class,
                                               CLASS_NAMES[CLASS_MOB])
                    _lock_color = theme.hex_to_bgr(_vis_cfg["lock_color"])
                    _attack_color = theme.hex_to_bgra(_vis_cfg["attack_color"])
                    _min_attack_color = theme.hex_to_bgra(_vis_cfg["min_attack_color"])
                    _chase_jump_color = theme.hex_to_bgra(_vis_cfg["chase_jump_color"])
                    _vision_color = theme.hex_to_bgra(_vis_cfg["vision_color"])
                    _vision_width = int(_vis_cfg["vision_width"])
                    # ⭐⭐ **各框线宽**（用户 2026-09-28 ✓"给其他的粗细也加配置"）—— 就在
                    #   「界面 → 辅助线与标记」里**色块右边那一格**调 ✓；⚠ **默认值 = 原来
                    #   在下面写死的那个数**（锁定框 3、其余 2 ✓ 观感一字不变 ✓）。
                    _lock_width = int(_vis_cfg.get("lock_width", 3) or 3)
                    _attack_width = int(_vis_cfg.get("attack_width", 2) or 2)
                    _min_attack_width = int(_vis_cfg.get("min_attack_width", 2) or 2)
                    _jump_attack_width = int(_vis_cfg.get("jump_attack_width", 2) or 2)
                    _chase_jump_width = int(_vis_cfg.get("chase_jump_width", 2) or 2)
                    _on_lock = bool(_vis_cfg.get("lock_on", True))
                    _on_attack = bool(_vis_cfg.get("attack_on", True))
                    _on_blind = bool(_vis_cfg.get("min_attack_on", True))
                    _on_chase = bool(_vis_cfg.get("chase_jump_on", True))
                    _on_vision = bool(_vis_cfg.get("vision_on", True))
                    # conf 也实时生效：重读 config/live.yaml（UI 改动会即时写入）
                    try:
                        _live = load_live()
                        if "conf_mob" in _live:
                            conf_mob = float(_live["conf_mob"])
                            conf_player = float(_live["conf_player"])
                            conf = min(conf_mob, conf_player)
                    except Exception:
                        pass

                if probe_on:
                    # f.image 是 BGR，转灰度解码
                    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
                    # 几何每帧现算：人工标定是比例，要按当前画面尺寸换算
                    # （read_bits 支持浮点 cell，所以不必取整）
                    px, py, cell, gap = _probe_geom(gray.shape)
                    # 几何一变（刚标完 / 换了项目）就重置判据：换标定那一刻时间戳
                    # 必然跳一次，不该被算成"乱跳"—— 否则刚存完就报"几何可疑"，
                    # 人会被自己刚做的事吓到。
                    _key = (round(px, 2), round(py, 2), round(cell, 2),
                            round(gap, 2))
                    if _key != _geom_key[0]:
                        _geom_key[0] = _key
                        probe_vd.reset()
                    ts_a = decode_ms(gray, px, py, cell, gap, bits)
                    if ts_a is None:
                        probe_miss += 1
                        # 探针解不出来也要留痕：端到端延迟是「控制质量的真指标」，
                        # 但它一旦解码失败，日志里只剩少数样本，读出来会**偏小**，
                        # 看着像「延迟变好了」—— 那是假象（实测踩过：段内样本
                        # 1433→161，中位同步从 35ms 掉到 12ms）。把 miss/ok 记下来，
                        # 报告里一比就知道这个数能不能信。
                        perf.count("probe_miss")
                    elif not (0 <= ts_a < probe_codec.DAY_MS):
                        # 解出来了、但值不可能（≥ 一天）＝**几何错了**（采样点没落在
                        # 方块上），不是时钟问题。混进「时钟对不上」会把人引错方向
                        # （实测：解出 306001920ms ≈ 85 小时，界面却提示去对时）。
                        probe_invalid += 1
                        perf.count("probe_invalid")
                    else:
                        probe_ok += 1
                        perf.count("probe_ok")
                        # 跨帧单调性（只在这里记，结论交给界面）—— 判据没过时界面
                        # **不许显示延迟数**：报一个假延迟比不报糟得多。
                        # 步长上限用显示帧周期算（判据自己会乘一个宽松倍数）。
                        _t_recv_ms = (f.t_recv_wall + _clock_ms() / 1000.0) * 1000.0
                        # 原始差也喂进判据：单调但**整片平移**的解（错位采样的另一种
                        # 形态）只有它能识破，否则界面会把它误报成"时钟对不上，去对时"。
                        probe_vd.add(ts_a, 1000.0 / max(1.0, float(
                            self._p.get("show_fps", 30.0) or 30.0)),
                            delay_raw_ms=_t_recv_ms - (probe_codec.day_start_ms() + ts_a))
                        if not probe_vd.mono:
                            probe_jumpy += 1
                            perf.count("probe_jumpy")
                        if not probe_vd.value_ok:
                            perf.count("probe_value_off")
                        d = resolve_delay_ms(_t_recv_ms, ts_a)
                        # 超过 5 秒视为解码错误（而不是真的有 5 秒延迟）
                        if d is not None and d < 5000:
                            delays.append(d)
                            _e2e_ref[0] = d      # 给世界状态用（上绳对齐的保持窗口）
                            if len(delays) > 120:
                                del delays[0]
                            # 端到端延迟（A 机屏幕时间码 → 本机解码）才是最该盯的
                            # 性能指标，以前只在界面显示中位数、不进日志。
                            perf.sample("e2e_probe_ms", d)
                        else:
                            # 解出来了但算不出延迟：多半是双机时钟没对齐（也可能是
                            # 解错了一位数字）。**和「解码失败」是两回事**，界面上要
                            # 分开显示 —— 以前混成一句「未解出」，框选明明说解出了，
                            # 这里却显示未解出，看着像 bug。
                            probe_reject += 1
                            perf.count("probe_reject")

                # ---- 推理（由「开始推理」开关控制）----
                if self._infer.is_set():
                    if model is None:
                        # 首次开推理才加载模型（收画面阶段不加载，秒出纯画面）
                        _limit_cpu_threads()
                        model = YOLO(weights)
                        # 类别 id 以**模型自己声明的类别名**为准：类别表给的是我们
                        # 训练时的顺序，但权重可能是别处训的 —— 按名字对齐，怎么都
                        # 不会张冠李戴；认不出来就沿用类别表里的 id。
                        _by_name = {str(n).lower(): int(i) for i, n
                                    in (getattr(model, "names", None) or {}).items()}
                        if "player" in _by_name:
                            _player_class = _by_name["player"]
                        if "mob" in _by_name:
                            _mob_class = _by_name["mob"]

                    # 视野框：只用于**画那四条虚线**（决策层的过滤在 agent 里直接用
                    # `settings.vision_*`，不读这个框 ✓），不裁剪推理区域 —— 检测走全图 ✓。
                    #
                    # ⚠ **每帧现算**（2026-09-27 用户："视野范围的更新速度太慢了（至少虚线框
                    #   看起来是这样）"）：原来这里是 `if perf_counter() - _vision_last >= 3.0`
                    #   ⇒ 角色/镜头一直在动、而这个框**每 3 秒才动一次** ⇒ 看上去就是"卡住
                    #   的虚线"✗。`_vision_box_for` 只是几次整数加减（微秒级），比画它那四条
                    #   虚线本身还便宜 ⇒ 没有任何理由节流 ✓。
                    #   顺带：改成每帧算之后，设置里改「视野」那几个数**当场生效**（以前最多
                    #   滞后 3 秒 ✓）。
                    _vision_box[0] = _vision_box_for(vis, _last_player[0],
                                                     decision_settings)

                    # ---- 一次全图推理：玩家（class 0）+ 怪（class 1）----
                    t0 = time.perf_counter()
                    res = model.predict(vis, conf=conf, imgsz=imgsz,
                                        device=device, verbose=False)[0]
                    infer_ms.append((time.perf_counter() - t0) * 1000.0)
                    perf.ms("infer_ms", t0)
                    if len(infer_ms) > 60:
                        del infer_ms[0]

                    k = 0
                    mob_dets = []       # [(x1, y1, x2, y2, conf), ...] 给 MobTracker
                    player_cands = []   # [(x1, y1, x2, y2, conf), ...] 交给 PlayerTracker
                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.cpu().numpy()
                        cfs = boxes.conf.cpu().numpy()
                        try:
                            clss = boxes.cls.cpu().numpy().astype(int)
                        except Exception:
                            clss = [1] * len(cfs)
                        for (x1, y1, x2, y2), c, cls in zip(xyxy, cfs, clss):
                            if int(cls) == _mob_class:
                                if c < conf_mob:
                                    continue
                                k += 1
                                mob_dets.append((x1, y1, x2, y2, c))
                            elif int(cls) == _player_class:
                                if c < conf_player:
                                    continue
                                # 收集所有玩家候选，交给 PlayerTracker 挑出「我」
                                player_cands.append((x1, y1, x2, y2, c))
                            # 其余类别都不进决策，但照常画框（颜色见类别表）：
                            #   掉落 / NPC    一直是忽略的；
                            #   其他玩家      忽略（混进 mobs 会去打人，混进玩家候选
                            #                 会认错自己）—— 它的价值是「别的玩家
                            #                 不再被误当怪」，决策暂时不需要它。

                    # ---- 玩家定位：YOLO 候选 → PlayerTracker 跟住「我」 ----
                    # 用位置连续性（离预测位置最近）挑出操作者的框，避免「画面里多个
                    # 玩家时取置信度最高的、把别人当成自己」；漏检用速度外推补框，
                    # 连续跟丢才解锁重锁。max_jump 就是「玩家追踪阈值」参数。
                    # ---- 小地图定位**先算一遍**（在挑玩家框之前）----
                    # 用户 2026-09-27 提的："用小地图的玩家世界坐标辅助玩家框防抖" ⇒ 黄点是
                    # **独立于 YOLO 的第二个传感器**（一个认画面、一个读小地图 ✓）⇒ 挑框时
                    # 可以拿"**本帧我到底在哪**"当尺子 ✓（`PlayerTracker.update(world=…)` ✓）。
                    # ⚠ 必须"**本帧黄点 + 上一帧的框/相机**"才成立：同一帧的 world 与 box 相减
                    #   会退化成"离上一帧框多远"（= 老判据），等于没加 ✗ —— 所以定位得先跑 ✓。
                    # 算不出来时 `_world_now` 就是 None ⇒ 追踪器**自动退回老判据** ✓（不猜 ✗）。
                    # ---- ⭐⭐ **「玩家位置」组：每帧现读**（用户 2026-09-28 ✓）----
                    #   ① **脚底偏移**灌进 `world_state` 的镜像（那边不做反向 import，说明在
                    #      `perception/world_state.py` 的 `FOOT_OFFSET_PX` 上 ✓）；
                    _ploc_fo = float(getattr(decision_settings,
                                             "player_foot_offset_px", 0) or 0)
                    try:
                        from perception import world_state as _ws_mod
                        _ws_mod.FOOT_OFFSET_PX = _ploc_fo
                    except Exception:
                        pass
                    #   ②③④ **框面积闸**（用户原话："当前框面积 ≤ 近期滚动基线 × (1 − 容差%)
                    #      ⇒ 这一拍推迟 / 不做查询 ✓（**拦在查询之前**，而不是靠放宽挑面 ✗）"）。
                    #      ⚠ 只能拿**上一拍锁定的框**说事：本帧的框要等 `_locate_mmap` 的出参才
                    #        挑得出来（黄点是"辅助挑框"的尺子 ✓ 见上面那段）⇒ 这一拍先判、再查 ✓。
                    #      ⚠ 两个"关"的口子：**容差 0** / **基线还没攒够 3 拍** ⇒ 照常查询 ✓
                    #        （老行为一字不变 ✓）；**框面积最小占比 0** ⇒ 那条也关 ✓。
                    _gate_closed = False
                    _area_prev = 0.0
                    _pb_prev = _last_player[0]
                    if _pb_prev is not None and len(_pb_prev) >= 6:
                        _area_prev = float(_pb_prev[4]) * float(_pb_prev[5])
                    if _area_prev > 0.0:
                        # ⭐⭐ **基线窗口改成"过去 N 秒"**（用户 2026-09-28 ✓ 原话："『面积基线
                        #   窗口』，这个『拍』是什么？是多久？**需要可量化的描述**"✓）——
                        #   存 `(时刻, 面积)` 并按**时间窗**筛 ✓：**拍数依赖帧率** ✗
                        #   （30 拍在 30fps 是 1 秒、在 15fps 是 2 秒 ⇒ 说不清多久 ✗），秒才可量化 ✓。
                        _t_now = time.monotonic()
                        _area_hist.append((_t_now, _area_prev))
                        if len(_area_hist) > 2000:      # 只当内存上限（真筛选靠时间窗 ✓）
                            del _area_hist[:1000]
                        _pm = float(getattr(decision_settings,
                                            "player_box_min_area_pct", 0.0) or 0.0)
                        _pt = float(getattr(decision_settings,
                                            "player_box_area_tol_pct", 0.0) or 0.0)
                        _bs = max(0.5, float(getattr(
                            decision_settings, "player_box_area_base_s", 3.0) or 3.0))
                        _win = [_a for _t, _a in _area_hist if _t_now - _t <= _bs]
                        _base = (sum(_win) / float(len(_win))) if _win else 0.0
                        # ⭐⭐ **「框面积最小占比」改成"占基线"**（用户 2026-09-28 ✓ 原话：
                        #   "『框面积最小占比』应该是占**过去 n 秒内平均面积**的比例"✓）——
                        #   原来跟**画面面积**比 ✗（不同地图 / 分辨率下人物框本来就不一样大 ⇒
                        #   跟画面比**没有可比性** ✗）。
                        if _pm > 0.0 and _base > 0.0:
                            if _area_prev < _base * _pm / 100.0:
                                _gate_closed = True      # 比基线小太多 ⇒ 这一帧**检测无效** ✓
                                perf.count("player_box_too_small")
                        if not _gate_closed and _pt > 0.0 and len(_win) >= 3:
                            if _base > 0.0 and _area_prev <= _base * (1.0 - _pt / 100.0):
                                _gate_closed = True      # 比近期基线小 ⇒ 这一拍不查
                                perf.count("player_area_below_base")
                    #      ⚠ 拦住时 `_loc = None` ⇒ 下面 `apply_to_player` 不跑 ⇒ **世界坐标沿
                    #        用上一拍** = 这一拍"**不定位**"✓（而不是"定到错的地方"✗）。
                    _loc = None if _gate_closed else self._locate_mmap(_mmap_panel)
                    if _loc is not None:
                        perf.count("mmap_ok" if _loc.get("ok") else "mmap_miss")
                    _world_now = None
                    if _loc is not None:
                        _wx0, _wy0 = _loc.get("world_x"), _loc.get("world_y")
                        if _wx0 is not None and _wy0 is not None:
                            _world_now = (float(_wx0), float(_wy0))

                    player_tracker.max_jump = max(
                        1.0, float(decision_settings.player_track_jump))
                    player_box = player_tracker.update(player_cands, world=_world_now)
                    # ⭐ **可信相机**（用户 2026-09-29 任务 2 ✓）：追踪器拿黄点的**权威世界
                    #   坐标**核对过玩家框，只把"对得上"那几拍采纳成相机 ✓ ⇒ 下游所有
                    #   "画面 → 世界"都用它（`screen_to_world` / 怪的集合解析器 ✓），
                    #   玩家框抖 Δ 时不再把 Δ 原样灌进每只怪的世界坐标 ✓。
                    #   ⚠ `None` = 还没建立 ⇒ 下游退回老口径现算，行为一字不变 ✓（不许当 0 ✗）。
                    _cam_now = player_tracker.camera()
                    if player_tracker.cam_rejected:
                        # 这一拍"框与黄点对不上" ⇒ 相机冻着用旧的（**打点**，排查要用：
                        # 一直涨说明玩家框锁错了 / 标定偏了 / 黄点在骗人 ✓）
                        perf.count("cam_reject")
                    # ⭐ 相机与"两个传感器差多少"也写进**行为日志**（用户 2026-09-29 任务 2）：
                    #   `cam_resid`（世界像素）是"玩家框那一路稳不稳"唯一的数；"世界坐标
                    #   整体偏"那类排查要看的就是它（差几十像素 ⇒ 就是它把怪推了一层 ✓）。
                    if _cam_now is not None:
                        behavior.sample("cam_x", float(_cam_now[0]), min_gap=1.0)
                        behavior.sample("cam_y", float(_cam_now[1]), min_gap=1.0)
                    if player_tracker.cam_resid is not None:
                        behavior.sample("cam_resid", float(player_tracker.cam_resid),
                                        min_gap=1.0)
                    n_boxes = k

                    # ---- 断线判断 / 自动重连 ----
                    # 必须放在**画框之前**：此刻 vis 还是干净帧。下面会往 vis 上
                    # 画玩家蓝框、平台线……画过的帧会干扰 UI 模板匹配。
                    # 注意 player_found 要用追踪器刚返回的结果，不能用下面那个
                    # 「跟丢时拿上一帧兜底」的 player_box —— 那玩意永远不为 None。
                    _act = reconnector.update(vis, player_box is not None,
                                              time.perf_counter())
                    if _act and _act.get("act") == "tap":
                        _tap(_act["key"])

                    # ---- 决策：找怪打 ----
                    if player_box is not None:
                        _last_player[0] = player_box
                    else:
                        player_box = _last_player[0]   # 完全跟丢时用上一帧兜底

                    # 绘制玩家蓝框：用追踪/兜底后的 player_box，保证和攻击距离线、
                    # 扫平台倾向箭头画在同一位置（否则蓝框画原始检出、线条画追踪后位置会漂移）
                    if player_box is not None and draw:
                        _cx, _cy, _bot, _c, _bw, _bh = player_box
                        _x1i = int(_cx - _bw / 2); _y1i = int(_cy - _bh / 2)
                        _x2i = int(_cx + _bw / 2); _y2i = int(_cy + _bh / 2)
                        cv2.rectangle(vis, (_x1i, _y1i), (_x2i, _y2i),
                                      _p_color, 2)
                        _ty = _y1i - 5 if _y1i > 14 else _y1i + 16
                        cv2.putText(vis, "%s %.2f" % (_p_label, _c),
                                    (_x1i + 2, _ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    _p_color, 1, cv2.LINE_AA)

                    ws = WorldState(frame_id=f.frame_id, ts=f.t_recv_mono,
                                    width=vis.shape[1], height=vis.shape[0])
                    ws.e2e_ms = float(_e2e_ref[0])   # 上绳对齐的保持窗口要用（见下）
                    if player_box is not None:
                        ws.player.x, ws.player.y = player_box[0], player_box[1]
                        ws.player.bottom = player_box[2]
                        ws.player.w, ws.player.h = player_box[4], player_box[5]
                        ws.player.found = True
                    # ⭐ 可信相机跟着 WorldState 走（**一处口径**：谁要"画面 → 世界"都读它 ✓）——
                    #   它是"最近一次与黄点对得上的那一拍"的相机，与 `player_box` 是不是这一帧
                    #   新挑的无关（跟丢时也照样能用 ✓）。
                    if _cam_now is not None:
                        ws.player.cam_x, ws.player.cam_y = _cam_now

                    # 读 HP/MP 条：从当前画面帧 vis 里按「画面比例」截取（不再抓本机屏幕），
                    # 限流 0.1s。收流/窗口两种模式都从 vis 截，游戏机上不再有抓屏行为。
                    if time.perf_counter() - _pot_last[0] >= 0.1:
                        _pot_last[0] = time.perf_counter()
                        if decision_settings.hp_bar:
                            try:
                                r = _crop_ratio(vis, decision_settings.hp_bar)
                                if r is not None and r.size:
                                    low, high = decision_settings.hp_color or ((0, 0, 100), (90, 90, 255))
                                    m = cv2.inRange(r, tuple(low), tuple(high))
                                    ratio = _bar_fill_ratio(m)
                                    if ratio > 0.0:   # 0% = 丢失/抖动，保持上次值
                                        _pot_vals[0] = ratio
                            except Exception:
                                pass
                        if decision_settings.mp_bar:
                            try:
                                r = _crop_ratio(vis, decision_settings.mp_bar)
                                if r is not None and r.size:
                                    low, high = decision_settings.mp_color or ((100, 0, 0), (255, 90, 90))
                                    m = cv2.inRange(r, tuple(low), tuple(high))
                                    ratio = _bar_fill_ratio(m)
                                    if ratio > 0.0:
                                        _pot_vals[1] = ratio
                            except Exception:
                                pass
                    ws.player.hp = _pot_vals[0]
                    ws.player.mp = _pot_vals[1]
                    # 同步防抖参数（每帧读 settings，改了即时生效）
                    mob_tracker.debounce_conf = decision_settings.debounce_conf
                    mob_tracker.debounce_ms = decision_settings.debounce_ms
                    # 用 MobTracker 追踪，得到跨帧稳定的 id（目标锁定 CD 靠它匹配）
                    ws.mobs = mob_tracker.update(mob_dets)

                    # 小地图定位（S3）：写进 WorldState 的**玩家世界坐标 + 所在段**。
                    # 「命令前往」要它才知道"我在哪块平台上"（`_fh_seen` 的读数就是它）。
                    # 算不出来时写 None（**不是 0**，0 是地图西北角这个合法坐标，
                    # 见 perception/world_state.py）。
                    # （`_loc` 上面已经算过了 —— 它得先算，才能辅助挑玩家框 ✓）
                    if _loc is not None:
                        mm.apply_to_player(ws.player, _loc)
                    # 「脚下属于哪些集合 / 贴在哪根绳上」也写进去（上绳执行器要用）——
                    # 2026-09-26 补：以前**没有任何地方写**，于是"到了"判不出来、
                    # "从绳上掉下来"每 2 秒误判一次 ⇒ 任务不停失败重试。
                    # ⭐ **执行器通知显式传进去**（`agent.climbing_vertical()` = 上一拍
                    #   climb/drop 执行器真发了 ↑/↓ ✓）—— 别让 `_fill_route_ctx` 自己去
                    #   找 agent（2026-09-28 踩过：找的是个**根本不存在**的 `self.agent`
                    #   ✗ ⇒ `ladder_id` 恒 None ⇒ 爬绳全废）。
                    self._fill_route_ctx(ws.player, _loc, agent.climbing_vertical())
                    _t = time.perf_counter()
                    action = agent.tick(ws)
                    perf.ms("agent_ms", _t)
                    _pipe = (time.perf_counter() - _t_pipe) * 1000.0
                    perf.sample("pipe_ms", _pipe)   # 收到帧 → 决策完（含推理/追踪）
                    # 帧预算 = 输入帧间隔。处理时间超过预算就意味着这帧「做不完」，
                    # 下一帧已经在路上 —— 这是实时性风险的第一手信号，单看中位看不出来。
                    # 预算未知（fps 拿不到）时不统计：宁可不报，也别误报。
                    if budget > 0 and _pipe > budget * 1000.0:
                        perf.count("over_budget")
                    # 输出行为命中了防抖幽灵框 → 一次性消除这些轨迹，避免持续空放技能
                    for _mid in ((action or {}).get("kill_mobs") or []):
                        mob_tracker.kill(_mid)

                    # 把识别到的血/蓝比例推给 UI（限流 0.1s，跟识别节奏一致）
                    if time.perf_counter() - self._last_potions >= 0.1:
                        self._last_potions = time.perf_counter()
                        self.potions_ready.emit(ws.player.hp, ws.player.mp)
                else:
                    # 纯画面：不推理、不画框、不做决策。ws 给个空壳供统计用。
                    ws = WorldState(frame_id=f.frame_id, ts=f.t_recv_mono,
                                    width=vis.shape[1], height=vis.shape[0])
                    n_boxes = 0
                    action = None

                # 显示限流判断提前：只在「要显示的这一帧」画框，不显示的帧不白画
                now = time.perf_counter()
                should_show = now - t_last_show >= show_interval
                if should_show:
                    t_last_show = now
                    n_show += 1

                _t_draw = time.perf_counter()
                if should_show and draw and self._infer.is_set():
                    st = action.get("state", "?")
                    label = "决策:%s" % st
                    t_id = action.get("target")
                    if st == "chase" and t_id is not None:
                        # 巡逻分支（无锁定目标）也是 chase 态，此时没有怪号可显示
                        label += " -> 怪%d dist=%.0f" % (t_id,
                                                        action.get("dist", 0.0))
                    cv2.putText(vis, label, (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)

                    # **攻击范围 = 矩形**（用户 2026-09-27：以前只画一条水平线）：
                    #   攻击范围框 = 角色中心 → 最大攻击距离（含**上下攻击距离**）
                    #   攻击盲区框 = 角色中心 → 最小攻击距离（最小距离 = 0 ⇒ 没有盲区 ⇒ 不画）
                    # ⚠ 四条边怎么算**只有一处**：`decision.agent.attack_box_rect`
                    #   （和判据 `DecisionAgent._in_box` 同源 ✓）—— 这儿自己再算一套的话，
                    #   画出来的框和真能打到的范围迟早对不上 ✗。
                    # ⚠ 竖直方向 **0 = 不限** ⇒ `attack_box_rect` 会画到画面边（把"不限"
                    #   如实画出来 ✓）；两个都 0（老配置）就是整幅高度 ✓ = 老行为 ✓。
                    if player_box is not None:
                        cx, cy = int(player_box[0]), int(player_box[1])
                        facing = action.get("facing", 1)
                        dirn = 1 if facing > 0 else -1
                        max_ad = max(1, int(decision_settings.attack_dist))
                        min_ad = max(0, int(decision_settings.min_attack_dist))
                        # ⚠ **负数 = 该方向不限、0 = 就是 0**（用户 2026-09-27 纠正的口径）
                        #   ⇒ 这里**原样传**，**不许 clamp 成 0**（clamp 会把"不限"偷偷变成
                        #   "打不到" ✗）。哪些情况是空集 / 哪些是"不限"、边界怎么画，
                        #   全在 `agent.attack_box_rect` 一处判 ✓。
                        _up_raw = getattr(decision_settings, "attack_up_dist", -1)
                        _dn_raw = getattr(decision_settings, "attack_down_dist", -1)
                        up_ad = -1 if _up_raw is None else int(_up_raw)
                        down_ad = -1 if _dn_raw is None else int(_dn_raw)
                        # 起算点 = **角色中心**（ex = cx），和 agent 的 `_center_dist`
                        # 口径一致（中心 → 怪框最近的边）。画在朝向边缘就会差半个
                        # 框宽：怪框碰到框线时其实还没进攻击范围，看着像坏掉。
                        # OpenCV 5 的 line/arrowedLine 不接受 float 坐标，必须取整。
                        ex = cx
                        max_x = ex + dirn * max_ad
                        yellow = _attack_color[:3]        # 「攻击范围框」颜色（画箭头用）
                        orange = _min_attack_color[:3]
                        _fh, _fw = vis.shape[:2]

                        def _draw_box(ad, color, on, from_d=0.0, width=2):
                            """按四条边画一个框；**空集（面积 0）⇒ 不画** ✓（用户 2026-09-27）。

                            `width` = 线宽（用户 2026-09-28 ✓"给其他的粗细也加配置"）——
                            由调用方传「辅助线与标记」里那一格的值 ✓。

                            `from_d` = 水平方向的**起点距离**（默认 0 = 从角色中心起 ✓）——
                            「攻击范围框」传 `min_ad`（它画的是**可攻击区**，不含盲区 ✓）、
                            「追击起跳框」传 `max_ad + min` ✓。
                            """
                            if not on:
                                return
                            rect = agent_mod.attack_box_rect(
                                cx, cy, ad, up=up_ad, down=down_ad, facing=facing,
                                frame_w=_fw, frame_h=_fh, from_d=from_d)
                            if rect is None:
                                return          # 空集：判定跳过、这里也不画 ✓
                            x0, y0, x1, y1 = rect
                            p0 = (int(round(x0)), int(round(y0)))
                            p1 = (int(round(x1)), int(round(y1)))
                            _blit_alpha(vis, color,
                                        lambda t, c: cv2.rectangle(t, p0, p1, c, width))

                        # 「攻击范围框」= **可攻击区**（最小 → 最大，**不含盲区**那块 ✓，
                        # 用户 2026-09-27 的图上就是这么并排的）；盲区框 = 中心 → 最小 ✓
                        _draw_box(max_ad, _attack_color, _on_attack, from_d=min_ad,
                                  width=_attack_width)
                        _draw_box(min_ad, _min_attack_color, _on_blind,
                                  width=_min_attack_width)
                        cv2.circle(vis, (cx, cy), 4, yellow, -1)       # 角色中心（起算点）

                        # 「**追击起跳框**」（用户 2026-09-27：以前是条**绿线**、颜色还写死
                        # 在这儿 ✗）：区间 = [最大攻击距离 + min, 最大攻击距离 + max] ✓，
                        # 颜色/开关现在也在「设置 → 外观 → 辅助线与标记」里 ✓。
                        # ⚠ **高度先跟「攻击范围框」一样**：用户说"后续需要算上**跳跃攻击范围**
                        #   的高度（**当前先不算**）" ✓ ⇒ 那时这里是 `up_ad`/`down_ad` 再加上
                        #   跳跃范围的高度 ✓（现在别自己编一个 ✗）。
                        # ⚠ 它的**判定**仍是**水平距离**（`agent._in_chase_jump_range`）✓
                        #   —— 画成框只是为了让人看得见，判据没变 ✓（用例 `t_chase_jump_*` 钉着）。
                        if decision_settings.chase_jump_enabled and _on_chase:
                            jlo = int(decision_settings.chase_jump_min)
                            jhi = int(decision_settings.chase_jump_max)
                            if jlo > jhi:
                                jlo, jhi = jhi, jlo
                            _draw_box(max_ad + jhi, _chase_jump_color, True,
                                      from_d=max_ad + jlo,
                                      width=_chase_jump_width)

                        # 扫平台倾向朝向箭头：位于攻击距离上方，指向朝向方向，
                        # 尾巴延长至背后锁定距离（back_range）。back_range 也是
                        # 从角色中心量的（见 agent._candidates），起点同样取 ex。
                        if decision_settings.strategy == "sweep":
                            back_range = max(0, int(decision_settings.back_range))
                            arrow_len = max(6, int(max_ad * 0.4))   # 比较短
                            arrow_y = cy - 18                        # 攻击距离上方
                            cv2.arrowedLine(vis,
                                            (ex - dirn * back_range, arrow_y),
                                            (ex + dirn * arrow_len, arrow_y),
                                            yellow, 3, tipLength=0.35)

                    # ---- ⭐⭐ **「玩家位置」箭头**（用户 2026-09-28 ✓）----
                    #   用户原话："以玩家位置为原点画「向前箭头 + 向上箭头」，颜色、线段长度
                    #   可配 ✓（画在实时预览上 ✓ 正好用它调试脚底偏移）"。
                    #   ⇒ 以**脚底**（`框底 + 脚底偏移` ✓ 与相机 y 同一口径 ✓）为原点画两根：
                    #      · 一根沿**世界 x 正方向**（画面上朝右 ✓）；
                    #      · 一根沿**世界 y 正方向**（画面上朝上 ✓）。
                    #   ⚠ 两根**同一个颜色**（用户 2026-09-28 改口径："x 箭头和 y 箭头应该是
                    #     一个颜色"✓）＋**同一个粗细**（`player_arrow_width_px` ✓ 就在色块右边
                    #     那一格调 ✓）；`箭头长度(px)` = 0 ⇒ 两根都不画 ✓。
                    #   ⚠ 它**不参与任何决策**，纯粹给人看："箭头根部该正好落在脚底" ✓
                    #     —— 偏上/偏下就是 `脚底偏移(px)` 没调对 ✓（这正是要它干的事 ✓）。
                    #   ⚠ `箭头长度(px)` = 0 ⇒ **不画**（老行为 = 画面上没有任何新增 ✓）。
                    #   ⚠ 原点取 `_last_player[0]` 的**框底**（和第 ① 条闸用同一份上一拍框 ✓
                    #     —— 它在同一帧的玩家框算出来之前就已经有了 ✓）。
                    # ⭐⭐ **玩家坐标箭头** —— 配置在 **设置 → 界面 → 辅助线与标记（实时预览）** ✓
                    #   （用户 2026-09-28 要求搬过去 ✓：它属于"画面上可显示的东西"，规范 §4 ✓）。
                    #   颜色 / 粗细 / 长度都从 `vis` 段读（`theme.load_vis()` ✓ 那份是**定期重读**
                    #   ⇒ 改完**不用重启**就生效 ✓）；`player_arrow_on` 关掉 ⇒ 不画（颜色留着 ✓）。
                    _al = int(_vis_cfg.get("player_arrow_len", 60) or 0)
                    _aw = max(1, int(_vis_cfg.get("player_arrow_width", 2) or 2))
                    # ⭐ **箭头尖大小**（用户 2026-09-28 ✓"再加个箭头 size 配置"）——
                    #   存的是**占线段长的百分比** ⇒ 这里 `/100` 变成 `cv2.arrowedLine`
                    #   要的 `tipLength`（默认 25 ⇒ 0.25 = **原来写死的那个值** ✓ 观感不变 ✓）。
                    _atip = (max(5, min(100, int(
                        _vis_cfg.get("player_arrow_tip_pct", 25) or 25))) / 100.0)
                    if not bool(_vis_cfg.get("player_arrow_on", True)):
                        _al = 0                      # 开关关掉 ⇒ 不画 ✓
                    _pb_arrow = _last_player[0]
                    if _al > 0 and _pb_arrow is not None and len(_pb_arrow) >= 6:
                        try:
                            from perception.world_state import hex_to_bgr
                            # 两根箭头**同一个颜色**（用户 2026-09-28："x 箭头和 y 箭头应该是
                            # 一个颜色"✓）；粗细也是同一个值 ✓（可配 ✓）。
                            _acol = hex_to_bgr(str(
                                _vis_cfg.get("player_arrow_color", "")))
                            _ox = int(_pb_arrow[0])
                            _oy = int(float(_pb_arrow[2]) + _ploc_fo)
                            _h, _w = vis.shape[:2]
                            if 0 <= _ox < _w and 0 <= _oy < _h:
                                _ax = max(0, min(_ox + _al, _w - 1))
                                _ay = max(0, _oy - _al)
                                cv2.arrowedLine(vis, (_ox, _oy), (_ax, _oy),
                                                _acol, _aw, tipLength=_atip)
                                cv2.arrowedLine(vis, (_ox, _oy), (_ox, _ay),
                                                _acol, _aw, tipLength=_atip)
                        except Exception:
                            pass

                    # 视野矩形：黑色虚线画出上下左右四条边。
                    # 某方向 <0（不限制）则该边不画；`_on_vision` = 设置里那项的**开关** ✓。
                    if _vision_box[0] is not None and _on_vision:
                        vleft, vtop, vright, vbottom = _vision_box[0]
                        vleft = max(0, min(vleft, vis.shape[1] - 1))
                        vright = max(0, min(vright, vis.shape[1] - 1))
                        vtop = max(0, min(vtop, vis.shape[0] - 1))
                        vbottom = max(0, min(vbottom, vis.shape[0] - 1))
                        if decision_settings.vision_top >= 0:
                            _blit_alpha(vis, _vision_color, lambda t, c: _draw_dashed_line(
                                t, vtop, c, thickness=_vision_width, x0=vleft, x1=vright))
                        if decision_settings.vision_bottom >= 0:
                            _blit_alpha(vis, _vision_color, lambda t, c: _draw_dashed_line(
                                t, vbottom, c, thickness=_vision_width, x0=vleft, x1=vright))
                        if decision_settings.vision_left >= 0:
                            _blit_alpha(vis, _vision_color, lambda t, c: _draw_dashed_vline(
                                t, vleft, c, thickness=_vision_width, y0=vtop, y1=vbottom))
                        if decision_settings.vision_right >= 0:
                            _blit_alpha(vis, _vision_color, lambda t, c: _draw_dashed_vline(
                                t, vright, c, thickness=_vision_width, y0=vtop, y1=vbottom))

                    # 怪物绿框：用 ws.mobs（含防抖幽灵目标），漏检后延迟防抖时间才消失
                    for m in ws.mobs:
                        gx1 = int(m.x - m.w / 2)
                        gy1 = int(m.y - m.h / 2)
                        gx2 = int(m.x + m.w / 2)
                        gy2 = int(m.y + m.h / 2)
                        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), _m_color, 2)
                        gty = gy1 - 5 if gy1 > 14 else gy1 + 16
                        cv2.putText(vis, "%s %.2f" % (_m_label, m.conf),
                                    (gx1 + 2, gty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    _m_color, 1, cv2.LINE_AA)

                    # 锁定目标怪：框标红（加粗），方便观测（`_on_lock` = 设置里那项开关 ✓）
                    tid = action.get("target")
                    if tid is not None and _on_lock:
                        for m in ws.mobs:
                            if m.id == tid:
                                tx1 = int(m.x - m.w / 2)
                                ty1 = int(m.y - m.h / 2)
                                tx2 = int(m.x + m.w / 2)
                                ty2 = int(m.y + m.h / 2)
                                cv2.rectangle(vis, (tx1, ty1), (tx2, ty2),
                                              _lock_color, _lock_width)
                                # ⛔ 锁定框**只画框、不写地点**（用户 2026-09-27 明确：
                                #   "**以前的锁定框表地点就不要了**" ✗）。
                                #   ⚠ 以前这里读 `agent.current_target_sets()`（挑目标时顺手缓存
                                #   的那份集合 ✓）画在"框下方靠左" —— 用户否掉之后那份缓存
                                #   **也成了死数据** ⇒ 连同 `agent.current_target_sets()` 一起删了 ✗
                                #   （"死数据不留" ✓）；地点现在只在**查过的怪框**那一行上 ✓。
                                break

                # ⭐ **"查过的怪框"标出来**（用户 2026-09-27："只要是**查询到的怪框地点**，
                #   就标出来，**缓存失效再移除**" ✓）—— 数据来自解析器那个**唯一漏斗**记的账
                #   （`_mark_mob_query` ✓，见 `mob_sets_of` 的说明 ✓）⇒ "**查过**"和"画面上有框"
                #   是两件事：**只有真查过的才画** ✓（不是每只怪都画 ✗）。
                #   画法：细框 + `查#id 集合名`（查了却判不出集合 ⇒ 把 `why` 写出来 ✓，
                #   那正是最该看见的一档 ✗）；时间一到 `queried_mob_boxes()` 就把它剪掉 ✓。
                #   ⭐ **地点（集合名）写在框下方靠左**（用户 2026-09-27 要求 ✓）——
                #   以前画在**框上方** ✗；贴到画面下沿放不下时才翻回框上方 ✓。
                #   ⚠ 颜色复用 `_lock_color`（和锁定框同一个开关 ✓，满足 UI 规范
                #   "不许在绘制里写死颜色" ✗）；要独立颜色我再往 ui.yaml 加一行 ✓。
                if _on_lock:
                    for _qid, (_qbox, _qnames, _qwhy, _qsince, _qexp) in self.queried_mob_boxes():
                        _qx, _qy, _qw, _qh = _qbox
                        _qx1, _qy1 = int(_qx - _qw / 2), int(_qy - _qh / 2)
                        _qx2, _qy2 = int(_qx + _qw / 2), int(_qy + _qh / 2)
                        cv2.rectangle(vis, (_qx1, _qy1), (_qx2, _qy2), _lock_color, 1)
                        # ⛔ **判不出时只写「失败」**（用户 2026-09-28：*"这样太丑了，不要显示
                        #   这么长，**失败就写失败**，log 里会留痕迹"* ✗）。原来把**整句 `why`**
                        #   拼在框下方 ⇒ 那句又长又套娃（`_lvl` 分档说明 + "它可能在半空 / 或那片
                        #   没圈地形" ⇒ 70+ 字 ✗），而且**框一多就糊满半屏** ✗。
                        # ✅ 详情**本来就有地方去**：`mob_fh` 事件带 `why`（那里**已截断 60 字** ✓
                        #   够看到 `_lvl` 分到哪一级 ✓）⇒ 要排查去翻 log ✓，画面上只要"成没成" ✓。
                        # ⚠ **成功时照旧写集合名**（那是"地点"，用户 2026-09-27 明确要的 ✓ 且短 ✓）。
                        _qlbl = ("查#%s %s" % (_qid, "／".join(_qnames)) if _qnames
                                 else "查#%s 失败" % (_qid,))
                        # x 贴**框左边**（"靠左" ✓，留 2px 内缩）；y 在**框下沿往下 14px**
                        _qly = _qy2 + 14
                        if _qly > vis.shape[0] - 3:      # 贴画面下沿 ⇒ 翻到框上方 ✓
                            _qly = max(10, _qy1 - 6)
                        cv2.putText(vis, _qlbl, (_qx1 + 2, _qly),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _lock_color, 1,
                                    cv2.LINE_AA)

                # 画框耗时：没画框的帧这里接近 0（大部分帧是不画的），
                # 所以看 p95 / 最大才是真实开销。
                perf.ms("draw_ms", _t_draw)

                # **原生帧先发**（不受显示限流/合并影响）：框选/测量要的永远是**最新**
                # 那一帧 —— 显示限流只该影响"画面上多久刷一次"，不该影响"取帧拿到多新"
                #（面板那边只存个引用，几微秒，不画 ✓）。同一个线程发的两个队列信号按
                # 发射顺序到达 ⇒ 面板先收到原生帧、再收到显示帧 ✓。
                self.raw_frame_ready.emit(raw)

                # 显示限流：到点了才推一帧。emit 是队列信号，不阻塞推理，
                # 所以推理始终按自己的速度跑。
                if should_show:
                    self.frame_ready.emit(vis)

                if now - t_last_stat >= 0.5:
                    # 吞吐/丢弃/负载：recv 是链路能给的输入速度，proc 是我们真处理
                    # 掉的；两者差距 + drop 增量 = 「实时性已经在丢」的第一手证据。
                    win = now - t_last_stat
                    # 窗口内速率（累计平均会把"刚刚开始恶化"抹平）。算成局部变量：
                    # 看门狗的明细要用**当前**这两个数（累计值看着会"还挺好"）。
                    _recv_w = ((slot.put_count - _prev[0]) / win) if win > 0 else 0.0
                    _proc_w = ((n - _prev[1]) / win) if win > 0 else 0.0
                    if win > 0:
                        perf.sample("recv_fps", _recv_w)
                        perf.sample("proc_fps", _proc_w)
                    if slot.dropped > _prev[2]:
                        perf.count("drop", slot.dropped - _prev[2])
                    _prev[0], _prev[1], _prev[2] = slot.put_count, n, slot.dropped
                    perf.sample("boxes", float(n_boxes))   # 每帧的框数 = 负载
                    perf.flush()        # 到点（默认 30 秒）落一段到 perf.log
                    el = now - t_start
                    gaps_recent = gaps[-60:]
                    # 「卡在谁身上」要在报之前算出来：输入/处理两个 fps 在不丢帧时
                    # 必然相等（见 limit_reason），得靠**输入间隔**和**本机耗时**比。
                    _infer_med = (sorted(infer_ms)[len(infer_ms) // 2]
                                  if infer_ms else 0.0)
                    _gap_med = (sorted(gaps_recent)[len(gaps_recent) // 2]
                                if gaps_recent else 0.0)
                    _med_delay = (sorted(delays)[len(delays) // 2]
                                  if delays else None)
                    _lag_warn, _lag_fresh = lag_watchdog(_med_delay, now,
                                                         _lag_state)
                    if _lag_fresh:
                        # 判定那一下记进 perf.log：事后翻日志能看到"什么时候开始锁死的"，
                        # 而不是只知道"某一段很慢"
                        perf.count("lag_lock")
                    # 明细放 tooltip：含"怎么办"的三步（状态行那一行塞不下）。
                    # 用**窗口内**速率，不用累计值 —— 累计值在下滑时看着还挺好。
                    _lag_detail = ""
                    if _lag_warn:
                        _lag_detail = (
                            "端到端延迟中位持续 %.0f 秒不回落（峰值 %.0f ms）。\n"
                            "  现场：读线程 %.1f fps ｜ 主回路 %.1f fps ｜ 槽位丢帧 %d ｜ "
                            "推理 %.1f ms ｜ 容器名义帧率 %s。\n\n"
                            "  **为什么会这样**：UDP 没有反压 —— A 机按自己的速率发，"
                            "B 机哪一环处理不过来，数据就堆在中间（内核缓冲 + ffmpeg 队列）；"
                            "而 `_LatestSlot` 只丢得了「已经解出来」的旧帧，管不到那两层。\n"
                            "  只要「中间某一环的速度 < 上游给它的量」，积压就只增不减 ⇒ "
                            "延迟稳在高位、自己回不来。\n\n"
                            "  **怎么办**（按见效快慢）：\n"
                            "  1. 立刻缓解：**停止再开始预览** —— 丢掉当前积压，延迟马上回来"
                            "（根因没动）；\n"
                            "  2. 治本：把 A 机推流帧率降到 B 机处理得过来的档"
                            "（一般是 60fps 档：A 实际约 54fps）；\n"
                            "  3. 治本：把「推理尺寸」从 800 降到 640（推理约省 1/3，余量变大）。"
                            % (now - (_lag_state[0] or now), _lag_state[1] or 0.0,
                               _recv_w, _proc_w, slot.dropped,
                               (sorted(infer_ms)[len(infer_ms) // 2]
                                if infer_ms else 0.0),
                               ("%.0f" % fps_ref[0]) if fps_ref[0] else "未知"))
                    self.stats_ready.emit({
                        # 读线程真正收到的帧率 —— 这是链路能给的输入速度
                        "recv_fps": slot.put_count / el if el > 0 else 0.0,
                        # 我们实际处理了多少帧 —— 两个数差得越多，说明丢帧越多
                        "proc_fps": n / el if el > 0 else 0.0,
                        "dropped": slot.dropped,
                        "infer_ms": _infer_med,
                        # 输入帧间隔中位 + "这一段卡在谁身上"（面板据此贴一句提示）
                        "gap_med": _gap_med,
                        "limit": limit_reason(_recv_w, _proc_w, _infer_med, _gap_med),
                        "delay_ms": _med_delay,
                        # 积压锁死：短句上状态行（最高优先级）、明细进 tooltip
                        "lag_warn": _lag_warn,
                        "lag_detail": _lag_detail,
                        "probe_on": probe_on,
                        "probe_miss": probe_miss,
                        "probe_invalid": probe_invalid,
                        "probe_ok": probe_ok,
                        "probe_reject": probe_reject,
                        # 几何可信度：`probe_mono` False = 解出的时间戳在乱跳 →
                        # 界面上那个延迟数不可信（见 probe_codec.Verdict）。
                        "probe_mono": bool(probe_vd.mono),
                        "probe_jumpy": probe_jumpy,
                        "probe_step_ms": probe_vd.step_med,
                        "probe_samples": len(probe_vd.ts_hist),
                        # 值域判据：时间戳单调、也在一天之内，但**整片被挪了位置**
                        # （错位采样的另一种形态）→ 有它才能从"去对时"里分辨出来。
                        "probe_value_ok": bool(probe_vd.value_ok),
                        "probe_value_off_s": (
                            probe_vd.value_off_ms / 1000.0
                            if probe_vd.value_off_ms is not None else None),
                        "clock_offset_ms": _clock_ms(),
                        "show_fps": n_show / el if el > 0 else 0.0,
                        "boxes": n_boxes,
                        "frames": n,
                        "gap_p95": (sorted(gaps_recent)[int(len(gaps_recent) * 0.95)]
                                    if gaps_recent else 0.0),
                        "bad": getattr(src, "bad_packets", 0),
                        "size": size_ref[0] or (0, 0),
                        "fps_src": fps_ref[0],
                        # 断线重连状态文字（reconnect.py 写，空串 = 没在重连）
                        "reconnect": getattr(decision_settings, "reconnect_note", ""),
                        # 本机负载告警（_load_watch 线程写）：空串 = 现在没人跟我们
                        # 抢机器。非空时状态行会把它顶到最前面 —— 它是上面那些
                        # 数字「为什么变差」的解释，比数字本身更要紧。
                        "load_warn": self._load[0],
                        "load_detail": self._load[1],
                    })
                    t_last_stat = now

        finally:
            reader_done.set()
            agent_mod.CURRENT = None     # 注销：别再往一个停了的 agent 下命令
            self.agent = None            # 本线程那份也摘掉（同上一行：停了的 agent 不留引用 ✓）
            agent.route_plan = None      # 顺手摘掉路径解析器（它的闭包持有本线程）
            try:
                agent.shutdown()     # 释放所有按键，避免游戏里键一直按着
            except Exception:
                pass
            # 先等 reader 退出，再 close —— 反过来 close 会和阻塞中的 read 抢 ffmpeg
            # 上下文导致死锁（「正在停止」卡住 / 源不释放）。
            #
            # ⚠ **等待时间必须 ≥ UDP 的读超时**：`link/pyav_source.py` 里是 **5 秒**
            # （`timeout=5000000`）。原来这里写 3.0，注释还写着"最多 2 秒就能响应"
            # —— 那是更早的参数，早就不成立了。实测踩到过：reader 还卡在 `read()`
            # 里，主线程就把 `src.close()` 掉了 → **UDP 5000 一直不放开**：
            # 界面上预览已经是「停止」，可下一个收流工具（probe_tune / stream_sweep）
            # 一开就报 10048「端口已被占用」，人只能去猜是不是 A 机没推流。
            reader.join(timeout=6.0)
            if reader.is_alive():
                # 留痕：这种情况端口不会立刻还回来，用户需要知道要重开工作台
                print("[live] 警告：读取线程 6 秒内没退出，源可能没释放 —— "
                      "UDP 端口会继续被本进程占着（要跑别的收流工具请重开工作台）",
                      flush=True)
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass
