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
from gui import theme

# 类别编号 → 显示名（和 data.yaml 的 class 对齐）
CLASS_NAMES = {0: "玩家", 1: "怪物", 2: "掉落", 3: "NPC"}

# player_id → YOLO class id 映射（玩家也走 YOLO，每个玩家独占一个类）。
# 当前单玩家固定 class 0；未来多玩家时扩展成 {"角色A": 0, "角色B": 2, ...}，
# 标注按此映射打类，实时只取当前 player_id 对应的类。默认 "" → 0。
PLAYER_CLASS_MAP = {"": 0}

# 类别 → 框颜色（BGR，和 data.yaml 的 class 对齐）。
# 玩家/怪物颜色从设置读（theme.load_vis），drop/npc 用固定默认。
def _box_colors():
    vis = theme.load_vis()
    return {
        0: theme.hex_to_bgr(vis["player_color"]),   # player
        1: theme.hex_to_bgr(vis["mob_color"]),      # mob
        2: (0, 255, 255),                           # drop 黄
        3: (0, 0, 255),                             # npc 红
    }


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

    # ---------------- 控制 ----------------

    def stop(self):
        self._stop.set()

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
        from perception.tracker import MobTracker
        from perception.world_state import WorldState
        from perception.platforms import (PlatformTracker, PlayerMotionTracker,
                                          relate_terrain)

        agent = CombatAgent(decision_settings)
        mob_tracker = MobTracker()   # 给怪稳定 id，供目标锁定 CD 跨帧匹配
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

        from core import wincap
        # HP/MP 条**独立抓取**：它们可能在实时帧框选范围之外（比如只框了
        # 游戏上半屏，HP/MP 条在屏幕更下方），从实时帧里截会越界读不到。
        # 按屏幕坐标直接 grab_rect，限流 0.2s 一次（grab_rect 有开销）。
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
        slot = _LatestSlot()
        reader_done = threading.Event()

        src = None
        size_ref = [None]      # (w, h)，第一帧后填，给统计条显示分辨率
        fps_ref = [0.0]

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
            frame_id = [0]

            def _reader():
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

            def _reader():
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
        px = int(self._p.get("probe_x", 100))
        py = int(self._p.get("probe_y", 8))
        cell = int(self._p.get("probe_cell", 16))
        gap = float(self._p.get("probe_gap", 2))
        bits = int(self._p.get("probe_bits", 40))
        offset_ms = float(self._p.get("clock_offset_ms", 0.0) or 0.0)

        delays = []
        probe_miss = 0
        if probe_on:
            # 自动对时：offset 会随 Windows NTP 漂移，后台线程里重新对一次，
            # 失败就沿用 params 传进来的旧值。
            try:
                from tools.clock_sync import sync_offset
                off = sync_offset(save=True, quiet=True)
                if off is not None:
                    offset_ms = off
            except Exception:
                pass
            try:
                from tools.probe_codec import decode_ms, resolve_delay_ms
            except Exception as e:
                self.failed.emit("探针解码模块不可用: %s" % e)
                probe_on = False
                decode_ms = resolve_delay_ms = None

        try:
            while not self._stop.is_set():
                f = slot.take(0.5)
                if f is None:
                    if reader_done.is_set():
                        break          # 流结束了
                    continue           # 只是暂时没新帧，继续等
                n += 1

                if last_mono is not None:
                    gaps.append((f.t_recv_mono - last_mono) * 1000.0)
                last_mono = f.t_recv_mono
                vis = f.image          # BGR（decode_format="bgr24" 直出）

                # 定期重读可视化配置：标记颜色/线宽改了实时生效
                if time.perf_counter() - _vis_refresh_last >= 1.0:
                    _vis_refresh_last = time.perf_counter()
                    _vis_cfg = theme.load_vis()
                    _cls_colors = {
                        0: theme.hex_to_bgr(_vis_cfg["player_color"]),
                        1: theme.hex_to_bgr(_vis_cfg["mob_color"]),
                        2: (0, 255, 255),
                        3: (0, 0, 255),
                    }
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
                    ts_a = decode_ms(gray, px, py, cell, gap, bits)
                    if ts_a is None:
                        probe_miss += 1
                    else:
                        d = resolve_delay_ms(
                            (f.t_recv_wall + offset_ms / 1000.0) * 1000.0, ts_a)
                        # 超过 5 秒视为解码错误（而不是真的有 5 秒延迟）
                        if d is not None and d < 5000:
                            delays.append(d)
                            if len(delays) > 120:
                                del delays[0]

                # ---- 推理（由「开始推理」开关控制）----
                if self._infer.is_set():
                    if model is None:
                        # 首次开推理才加载模型（收画面阶段不加载，秒出纯画面）
                        model = YOLO(weights)

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
                    if len(infer_ms) > 60:
                        del infer_ms[0]

                    k = 0
                    mob_dets = []       # [(x1, y1, x2, y2, conf), ...] 给 MobTracker
                    player_cand = None  # (x1, y1, x2, y2, conf)，YOLO 检出的玩家框
                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.cpu().numpy()
                        cfs = boxes.conf.cpu().numpy()
                        try:
                            clss = boxes.cls.cpu().numpy().astype(int)
                        except Exception:
                            clss = [1] * len(cfs)
                        for (x1, y1, x2, y2), c, cls in zip(xyxy, cfs, clss):
                            if int(cls) == 1:
                                if c < conf_mob:
                                    continue
                                k += 1
                                mob_dets.append((x1, y1, x2, y2, c))
                            elif int(cls) == _player_class:
                                if c < conf_player:
                                    continue
                                # 画面里通常只有一个玩家，取置信度最高的那个框
                                if player_cand is None or c > player_cand[4]:
                                    player_cand = (x1, y1, x2, y2, c)
                            # drop/npc（class 2/3）暂不参与决策

                    # ---- 玩家定位：YOLO 检出 ----
                    player_box = None
                    if player_cand is not None:
                        x1, y1, x2, y2, c = player_cand
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        bw = x2 - x1
                        bh = y2 - y1
                        player_box = (cx, cy, cy + bh / 2.0, c, bw, bh)
                    n_boxes = k

                    # ---- 决策：找怪打 ----
                    # 玩家识别防抖：检出框中心跳变超过防抖距离视为误检，沿用上一帧位置
                    pdd = max(0.0, float(decision_settings.player_debounce_dist))
                    if player_box is not None and _last_player[0] is not None and pdd > 0:
                        dx = player_box[0] - _last_player[0][0]
                        dy = player_box[1] - _last_player[0][1]
                        if (dx * dx + dy * dy) ** 0.5 > pdd:
                            player_box = _last_player[0]   # 跳变过大，用上一帧

                    # 玩家框（蓝框）短暂消失时，用上一帧位置兜底，避免 agent 丢失玩家
                    if player_box is None:
                        player_box = _last_player[0]
                    else:
                        _last_player[0] = player_box

                    # 绘制玩家蓝框：用防抖/兜底后的 player_box，保证和攻击距离线、
                    # 扫平台倾向箭头画在同一位置（否则蓝框画原始检出、线条画防抖后位置会漂移）
                    if player_box is not None and draw:
                        _cx, _cy, _bot, _c, _bw, _bh = player_box
                        _x1i = int(_cx - _bw / 2); _y1i = int(_cy - _bh / 2)
                        _x2i = int(_cx + _bw / 2); _y2i = int(_cy + _bh / 2)
                        cv2.rectangle(vis, (_x1i, _y1i), (_x2i, _y2i),
                                      _cls_colors[0], 2)
                        _ty = _y1i - 5 if _y1i > 14 else _y1i + 16
                        cv2.putText(vis, "%s %.2f" % (CLASS_NAMES[0], _c),
                                    (_x1i + 2, _ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    _cls_colors[0], 1, cv2.LINE_AA)

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
                    # 读 HP/MP 条：独立抓取（屏幕坐标直接 grab_rect），不依赖实时
                    # 帧范围，限流 0.5s（血蓝变化不需要太频繁，grab_rect 有开销）。
                    if time.perf_counter() - _pot_last[0] >= 0.5:
                        _pot_last[0] = time.perf_counter()
                        if decision_settings.hp_bar:
                            try:
                                r = wincap.grab_rect(tuple(int(v) for v in decision_settings.hp_bar))
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
                                r = wincap.grab_rect(tuple(int(v) for v in decision_settings.mp_bar))
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
                    relate_terrain(ws.mobs, ws.player, ws.platforms)
                    action = agent.tick(ws)
                    # 攻击了防抖幽灵框 → 立即消除该轨迹，避免持续空放技能
                    kill_mob = action.get("kill_mob") if action else None
                    if kill_mob is not None:
                        mob_tracker.kill(kill_mob)

                    # 把识别到的血/蓝比例推给 UI（限流 0.3s，避免刷屏）
                    if time.perf_counter() - self._last_potions >= 0.3:
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
                    #   角色 → 最小攻击距离：橙色线（太近的规避范围）
                    #   最小 → 最大攻击距离：黄色线（可攻击范围）
                    if player_box is not None:
                        px, py = int(player_box[0]), int(player_box[1])
                        facing = action.get("facing", 1)
                        dirn = 1 if facing > 0 else -1
                        max_ad = max(1, int(decision_settings.attack_dist))
                        min_ad = max(0, int(decision_settings.min_attack_dist))
                        max_x = px + dirn * max_ad
                        yellow = _attack_color        # 最大攻击距离线颜色
                        orange = _min_attack_color    # 最小攻击距离/规避范围线颜色
                        if min_ad > 0:
                            min_x = px + dirn * min_ad
                            cv2.line(vis, (px, py), (min_x, py), orange, 2)
                            cv2.line(vis, (min_x, py), (max_x, py), yellow, 2)
                            # 最小攻击距离处加橙色小刻度
                            cv2.line(vis, (min_x, py - 8), (min_x, py + 8), orange, 2)
                        else:
                            cv2.line(vis, (px, py), (max_x, py), yellow, 2)
                        cv2.circle(vis, (px, py), 4, yellow, -1)
                        cv2.line(vis, (max_x, py - 8), (max_x, py + 8), yellow, 2)

                        # 追击起跳距离：接在最大攻击距离前方，绿色线段 + 竖向刻度
                        cjd = max(0, int(decision_settings.chase_jump_dist))
                        if cjd > 0:
                            jump_x = max_x + dirn * cjd
                            green = (0, 200, 0)
                            cv2.line(vis, (max_x, py), (jump_x, py), green, 2)
                            cv2.line(vis, (jump_x, py - 8), (jump_x, py + 8), green, 2)

                        # 扫平台倾向朝向箭头：位于攻击距离上方，指向朝向方向，
                        # 尾巴延长至背后锁定距离（back_range）
                        if decision_settings.strategy == "sweep":
                            back_range = max(0, int(decision_settings.back_range))
                            arrow_len = max(6, int(max_ad * 0.4))   # 比较短
                            arrow_y = py - 18                        # 攻击距离上方
                            cv2.arrowedLine(vis,
                                            (px - dirn * back_range, arrow_y),
                                            (px + dirn * arrow_len, arrow_y),
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
                        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), _cls_colors[1], 2)
                        gty = gy1 - 5 if gy1 > 14 else gy1 + 16
                        cv2.putText(vis, "%s %.2f" % (CLASS_NAMES[1], m.conf),
                                    (gx1 + 2, gty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    _cls_colors[1], 1, cv2.LINE_AA)

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

                # 显示限流：到点了才推一帧。emit 是队列信号，不阻塞推理，
                # 所以推理始终按自己的速度跑。
                if should_show:
                    self.frame_ready.emit(vis)

                if now - t_last_stat >= 0.5:
                    el = now - t_start
                    gaps_recent = gaps[-60:]
                    self.stats_ready.emit({
                        # 读线程真正收到的帧率 —— 这是链路能给的输入速度
                        "recv_fps": slot.put_count / el if el > 0 else 0.0,
                        # 我们实际处理了多少帧 —— 两个数差得越多，说明丢帧越多
                        "proc_fps": n / el if el > 0 else 0.0,
                        "dropped": slot.dropped,
                        "infer_ms": sorted(infer_ms)[len(infer_ms) // 2] if infer_ms else 0.0,
                        "delay_ms": (sorted(delays)[len(delays) // 2]
                                     if delays else None),
                        "probe_on": probe_on,
                        "probe_miss": probe_miss,
                        "show_fps": n_show / el if el > 0 else 0.0,
                        "boxes": n_boxes,
                        "platforms": len(ws.platforms),
                        "frames": n,
                        "gap_p95": (sorted(gaps_recent)[int(len(gaps_recent) * 0.95)]
                                    if gaps_recent else 0.0),
                        "bad": getattr(src, "bad_packets", 0),
                        "size": size_ref[0] or (0, 0),
                        "fps_src": fps_ref[0],
                    })
                    t_last_stat = now

        finally:
            reader_done.set()
            try:
                agent.shutdown()     # 释放所有按键，避免游戏里键一直按着
            except Exception:
                pass
            # 先等 reader 退出（read 有 udp 超时，最多 2 秒就能响应），再 close ——
            # 反过来 close 会和阻塞中的 read 抢 ffmpeg 上下文导致死锁，表现为「正在停止」卡住。
            reader.join(timeout=3.0)
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass
