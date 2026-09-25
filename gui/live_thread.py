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
        self._mmap_cli = None           # 来源=收流 时那一路 TCP（懒起）
        # 黄点容差（界面上那排，键名见 perception.minimap.TRACK_KEYS）。
        # `_track_applied` 记住"已经喂给 locator 的那一份"，主回路比对后按需应用
        # —— **不在 set_mmap 里直接改 locator**：那是另一条线程正在用的对象。
        self._mmap_track = self._p.get("mmap_track") or None
        self._track_applied = None

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
        return self._locator.update(panel, src=src)

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
        from decision.input import tap as _tap
        from decision.reconnect import Reconnector
        from perception.tracker import MobTracker, PlayerTracker
        from perception.world_state import WorldState
        from perception import minimap as mm     # 小地图定位（S3）：apply_to_player 等
        from perception.platforms import (PlatformTracker, PlayerMotionTracker,
                                          relate_terrain)

        agent = CombatAgent(decision_settings)
        mob_tracker = MobTracker()   # 给怪稳定 id，供目标锁定 CD 跨帧匹配
        player_tracker = PlayerTracker()   # 跟住「我」：位置连续性，别把别人认成自己
        # 断线重连状态机（判界面 → 停自动 → 按 Enter/点鼠标走回游戏）
        reconnector = Reconnector(decision_settings)
        # 平台和玩家运动状态是 WorldState 的一部分，不交给 YOLO：平台顶边用
        # 轻量图像处理每帧更新，玩家速度则由连续框的位置估计。
        platform_tracker = PlatformTracker()
        player_motion = PlayerMotionTracker()

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
        _vision_box = [None]        # 视野矩形 (left, top, right, bottom)，3s 更新一次
        _vision_last = [0.0]        # 上次更新时间

        # 可视化配置：定期重读（颜色改了实时生效，不用重开实时预览）
        _vis_cfg = theme.load_vis()
        _cls_colors = _box_colors()
        _lock_color = theme.hex_to_bgr(_vis_cfg["lock_color"])
        _attack_color = theme.hex_to_bgr(_vis_cfg["attack_color"])
        _min_attack_color = theme.hex_to_bgr(_vis_cfg["min_attack_color"])
        _vision_color = theme.hex_to_bgr(_vis_cfg["vision_color"])
        _vision_width = int(_vis_cfg["vision_width"])
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
        from core import perf
        perf.configure(bool(self._p.get("perf_log", True)))

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
                _t_pipe = time.perf_counter()   # 「收到这一帧 → 决策完」的总耗时
                perf.frame_arrived(_t_pipe)     # 记下这帧被取走的时刻（算 out_key_ms）
                vis = f.image          # BGR（decode_format="bgr24" 直出）
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
                    _attack_color = theme.hex_to_bgr(_vis_cfg["attack_color"])
                    _min_attack_color = theme.hex_to_bgr(_vis_cfg["min_attack_color"])
                    _vision_color = theme.hex_to_bgr(_vis_cfg["vision_color"])
                    _vision_width = int(_vis_cfg["vision_width"])
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

                    # 视野框：只用于画虚线 + 决策层过滤（agent._filter_mobs），
                    # 不裁剪推理区域 —— 检测走全图，玩家和怪都从全图出。
                    if time.perf_counter() - _vision_last[0] >= 3.0:
                        _vision_last[0] = time.perf_counter()
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
                    player_tracker.max_jump = max(
                        1.0, float(decision_settings.player_track_jump))
                    player_box = player_tracker.update(player_cands)
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
                    if player_box is not None:
                        ws.player.x, ws.player.y = player_box[0], player_box[1]
                        ws.player.bottom = player_box[2]
                        ws.player.w, ws.player.h = player_box[4], player_box[5]
                        ws.player.found = True
                    # 路线识别（平台识别/跳跃预测）开关：关掉就跳过，省掉每帧图像处理
                    if decision_settings.route_enabled:
                        ws.platforms = platform_tracker.update(vis, f.t_recv_mono)
                    else:
                        ws.platforms = []
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
                    if decision_settings.route_enabled:
                        ws.jump_prediction = player_motion.update(
                            ws.player, ws.platforms, f.t_recv_mono)
                    else:
                        ws.jump_prediction = None
                    # 地形关系识别（感知层）：给每只怪打上所在平台 + 是否与玩家
                    # 当前平台连接。Agent 只消费 mob.reachable，不做几何运算。
                    # 开关关掉时**整块跳过**（等于"没有平台可对齐"）—— 那正是
                    # relate_terrain 在 platforms 为空时的分支结果：mob.platform_id
                    # 留着 None、reachable 保持默认 True。语义相同，少一遍每帧循环，
                    # 也让这个开关真正做到「关掉 = 路线识别的活一点不干」。
                    if decision_settings.route_enabled:
                        relate_terrain(ws.mobs, ws.player, ws.platforms)
                        # 小地图定位（S3）：写进 WorldState 的**玩家世界坐标 +
                        # 所在段**（决策层要它才知道"我在哪块平台上"）。和上面三处
                        # 一样挂在 route_enabled 下 —— 它属于「寻路那套」，关掉就
                        # 该一点活都不干。算不出来时写 None（**不是 0**，0 是地图
                        # 西北角这个合法坐标，见 perception/world_state.py）。
                        _loc = self._locate_mmap(_mmap_panel)
                        if _loc is not None:
                            mm.apply_to_player(ws.player, _loc)
                            perf.count("mmap_ok" if _loc.get("ok")
                                       else "mmap_miss")
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
                    # 平台顶边：青色实线；编号稳定于 PlatformTracker，便于排查地图
                    # 滚动/局部重检时的关联。当前平台额外画白色。
                    for p in ws.platforms:
                        color = (255, 255, 255) if p.id == ws.player.current_platform_id else (255, 255, 0)
                        cv2.line(vis, (int(p.x1), int(p.y)), (int(p.x2), int(p.y)), color, 2)
                        cv2.putText(vis, "P%d" % p.id, (int(p.x1), max(14, int(p.y) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    if ws.jump_prediction is not None:
                        jp = ws.jump_prediction
                        cv2.circle(vis, (int(jp.landing_x), int(jp.landing_y)), 5, (255, 0, 255), -1)
                        cv2.putText(vis, "land P%d %.0fms" % (jp.target_platform_id,
                                    jp.landing_time * 1000),
                                    (int(jp.landing_x) + 6, int(jp.landing_y) - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
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

                    # 攻击距离可视化：
                    #   角色中心 → 最小攻击距离：橙色线（太近的规避范围）
                    #   最小 → 最大攻击距离：黄色线（可攻击范围）
                    if player_box is not None:
                        cx, cy = int(player_box[0]), int(player_box[1])
                        facing = action.get("facing", 1)
                        dirn = 1 if facing > 0 else -1
                        max_ad = max(1, int(decision_settings.attack_dist))
                        min_ad = max(0, int(decision_settings.min_attack_dist))
                        # 起算点 = **角色中心**（ex = cx），和 agent 的 _center_dist
                        # 口径一致（中心 → 怪框最近的边）。画在朝向边缘就会差半个
                        # 框宽：怪框碰到黄线时其实还没进攻击范围，看着像坏掉。
                        # OpenCV 5 的 line/arrowedLine 不接受 float 坐标，必须取整。
                        ex = cx
                        max_x = ex + dirn * max_ad
                        yellow = _attack_color        # 最大攻击距离线颜色
                        orange = _min_attack_color    # 最小攻击距离/规避范围线颜色
                        if min_ad > 0:
                            min_x = ex + dirn * min_ad
                            cv2.line(vis, (ex, cy), (min_x, cy), orange, 2)
                            cv2.line(vis, (min_x, cy), (max_x, cy), yellow, 2)
                            # 最小攻击距离处加橙色小刻度
                            cv2.line(vis, (min_x, cy - 8), (min_x, cy + 8), orange, 2)
                        else:
                            cv2.line(vis, (ex, cy), (max_x, cy), yellow, 2)
                        cv2.circle(vis, (cx, cy), 4, yellow, -1)
                        cv2.line(vis, (max_x, cy - 8), (max_x, cy + 8), yellow, 2)

                        # 追击起跳区间：绿色线段（以最大攻击距离为基准偏移 min~max）
                        if decision_settings.chase_jump_enabled:
                            jlo = int(decision_settings.chase_jump_min)
                            jhi = int(decision_settings.chase_jump_max)
                            if jlo > jhi:
                                jlo, jhi = jhi, jlo
                            if jlo != jhi:
                                jx1 = max_x + dirn * jlo
                                jx2 = max_x + dirn * jhi
                                # 内侧部分会和黄色攻击线段重叠，下移几像素错开
                                jy = cy + 6 if jlo < 0 else cy
                                green = (0, 200, 0)
                                cv2.line(vis, (jx1, jy), (jx2, jy), green, 2)
                                cv2.line(vis, (jx2, jy - 8), (jx2, jy + 8), green, 2)

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

                    # 视野矩形：黑色虚线画出上下左右四条边。
                    # 某方向 <0（不限制）则该边不画。
                    if _vision_box[0] is not None:
                        vleft, vtop, vright, vbottom = _vision_box[0]
                        vleft = max(0, min(vleft, vis.shape[1] - 1))
                        vright = max(0, min(vright, vis.shape[1] - 1))
                        vtop = max(0, min(vtop, vis.shape[0] - 1))
                        vbottom = max(0, min(vbottom, vis.shape[0] - 1))
                        if decision_settings.vision_top >= 0:
                            _draw_dashed_line(vis, vtop, _vision_color,
                                              thickness=_vision_width, x0=vleft, x1=vright)
                        if decision_settings.vision_bottom >= 0:
                            _draw_dashed_line(vis, vbottom, _vision_color,
                                              thickness=_vision_width, x0=vleft, x1=vright)
                        if decision_settings.vision_left >= 0:
                            _draw_dashed_vline(vis, vleft, _vision_color,
                                               thickness=_vision_width, y0=vtop, y1=vbottom)
                        if decision_settings.vision_right >= 0:
                            _draw_dashed_vline(vis, vright, _vision_color,
                                               thickness=_vision_width, y0=vtop, y1=vbottom)

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

                    # 锁定目标怪：框标红（加粗），方便观测
                    tid = action.get("target")
                    if tid is not None:
                        for m in ws.mobs:
                            if m.id == tid:
                                tx1 = int(m.x - m.w / 2)
                                ty1 = int(m.y - m.h / 2)
                                tx2 = int(m.x + m.w / 2)
                                ty2 = int(m.y + m.h / 2)
                                cv2.rectangle(vis, (tx1, ty1), (tx2, ty2),
                                              _lock_color, 3)
                                break

                # 画框耗时：没画框的帧这里接近 0（大部分帧是不画的），
                # 所以看 p95 / 最大才是真实开销。
                perf.ms("draw_ms", _t_draw)

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
                        "infer_ms": sorted(infer_ms)[len(infer_ms) // 2] if infer_ms else 0.0,
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
                        "platforms": len(ws.platforms),
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
