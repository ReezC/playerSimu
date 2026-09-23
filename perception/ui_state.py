"""界面判别：用模板锚点判断当前停在哪个界面。

**为什么用模板匹配，不训 YOLO、也不用 OCR**
    · 这些界面是**固定 UI** —— 元素的位置和外观不随游戏内容变化，模板匹配足够，
      而且零训练成本、可以直接拿录像裁出来用；
    · 文字（「连接」/「确定」/「与服务器连接发生错误」）走 OCR 反而更差：字体是
      位图带抗锯齿、还会随分辨率缩放，识别率和延迟都不如直接匹配像素；
    · 现有 YOLO 只有 player/mob/drop/npc 四类，跟 UI 无关，重训一遍不划算。

**判据是「锚点全中」**
    单个模板可能被动画、遮挡、鼠标指针干扰，所以一个界面登记多个锚点，
    要求**登记的全部锚点都命中**才认这个界面。都没命中 = 未知界面，返回 None。

**优先级**
    这几个界面是**层层叠着**的：断线提示框盖在登录界面上、频道面板盖在服务器
    列表上、排队弹窗又盖在频道面板上 —— 同一个瞬间会有多层可见。所以按
    「浮层优先」判定：先 login_err，再 queue / channel_panel / channel_list…

**选锚点的三条硬规矩**（都是踩过 / 量过才定的）
    1. **不含账号等用户数据**。第一版拿整个登录木牌当锚点，里面带着账号
       `15359879262` —— 换个号就匹配不上了。锚点必须是「什么账号都一样」的元素。
    2. **必须是该界面独有的**。CADPA 12+ 适龄提示框看着很标志性，但它从登录
       一直到选角都在屏幕上（实测 12.5s 时分数仍是 1.000），不是登录界面独有。
    3. **区分度要够**：真匹配接近 1.0，其余界面上的最高分要明显低。实测
       login_connect 1.000 vs 0.670（选频道/游戏内），login_agree 0.999 vs 0.444。

**分辨率与性能**
    模板按 1920x1080（基准分辨率）裁。判别时先把整帧缩到基准的 MATCH_SCALE 倍
    再匹配 —— 全分辨率下单个锚点要 100+ ms（实测 116.7ms），缩到 0.5 只要 ~20ms。
    这个方法本来就是「每秒判一两次」用的，不需要每帧跑。
    换分辨率不用重裁模板，但前提是游戏 UI 随分辨率等比缩放。

**模板图的来源**
    见 ANCHORS 表：每个锚点记了「从哪段素材的第几秒、裁哪个矩形」。
    模板图在 perception/ui_templates/（很小的 PNG，**进 git** —— 素材视频是
    gitignore 的，不把图存下来换台机器就重建不出来）。
    重新生成：python -m tools.make_ui_templates
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "ui_templates"

# 素材视频（模板的来源）。gitignore 里，所以模板图必须单独存。
SOURCE_VIDEO = ROOT / "datasets" / "records" / "断线重连素材.mp4"

# 模板的基准分辨率：ANCHORS 里的矩形坐标都是这个尺寸下的
BASE_W, BASE_H = 1920, 1080

# 判别时把画面缩到基准的这个比例再匹配（只影响耗时，不影响坐标口径）
MATCH_SCALE = 0.5

# 匹配阈值：TM_CCOEFF_NORMED 的分数。0.5 缩放下实测真匹配 >= 0.94、
# 其余界面上的最高分 <= 0.67，0.80 卡在中间，两边都留了余量。
THRESHOLD = 0.80

# 搜索范围：只允许锚点在登记位置附近偏移这么多像素（@1920x1080）再找。
# UI 是固定布局，给一点余量足够；把搜索限制在小窗口里比整帧扫快 ~20 倍
# （实测 7 个锚点整帧扫 100 ms/次，窗口扫约 7 ms/次），顺带也压低了误命中。
SEARCH_MARGIN = 60

# 界面 id
UI_LOGIN = "login"              # 登录界面（登录木牌可见，等按「连接」）
UI_LOGIN_ERR = "login_err"      # 断线提示框「与服务器连接发生错误」浮在上面
UI_CHANNEL_LIST = "channel_list"    # 选择频道：服务器列表（频道面板还没弹出来）
UI_CHANNEL_PANEL = "channel_panel"  # 选择频道：频道面板已弹出（盖在列表上）
UI_QUEUE = "queue"                  # 排队弹窗（盖在频道面板上）
UI_CHAR_SELECT = "char_select"      # 选择角色

# 界面中文名（状态提示 / 日志用）
UI_NAMES = {
    UI_LOGIN: "登录界面",
    UI_LOGIN_ERR: "断线提示框",
    UI_CHANNEL_LIST: "选择频道（服务器列表）",
    UI_CHANNEL_PANEL: "选择频道（频道面板）",
    UI_QUEUE: "排队弹窗",
    UI_CHAR_SELECT: "选择角色",
}

# 判定优先级：**浮层在前**。这几个界面是层层叠着的 ——
# 排队弹窗盖在频道面板上、频道面板盖在服务器列表上、断线提示框盖在登录界面上，
# 所以必须让「更靠上的那层」先判，否则下层的锚点先命中就报错了层。
PRIORITY = (UI_LOGIN_ERR, UI_QUEUE, UI_CHANNEL_PANEL, UI_CHANNEL_LIST,
            UI_CHAR_SELECT, UI_LOGIN)

# 锚点表：界面 id -> [(模板名, 裁剪矩形(x, y, w, h) @1920x1080, 源帧秒数, 说明)]
# 加新界面先在这里登记，再跑 tools/make_ui_templates.py 生成图。
#
# 矩形都是**实测**调出来的（不是目测）：对整段素材每 0.25s 采一帧，要求
# 「自己那段时间内的分数 >= 0.99，其它所有帧 < 0.80」。实测命中区间：
#     channel_title      11.75 ~ 18.50   （整个选频道界面，含面板/排队期间）
#     channel_panel_top  12.50 ~ 18.50   （频道面板弹出期间）
#     queue_text         16.00 ~ 18.25   （排队弹窗期间）
#     char_title         18.75 ~ 19.75   （选角界面）
#     login_connect       3.25 ~  5.25
#     login_err_text      0.00 ~  2.50
# 时间轴上的真实切换点（逐帧变化强度测出来的）：12.50s 面板弹出、16.00s 排队
# 弹窗（同时全屏压暗）、18.75s 切到选角、20.00s 淡出进加载。
ANCHORS = {
    UI_LOGIN: [
        ("login_connect", (1085, 505, 135, 70), 4.5, "登录木牌上的「连接」按钮"),
        ("login_agree", (965, 602, 425, 36), 4.5, "木牌下方「抵制不良游戏…」两行协议文字"),
    ],
    UI_LOGIN_ERR: [
        ("login_err_text", (748, 386, 245, 62), 0.2,
         "提示框里「红圈图标 + 与服务器连接发生错误」那一条"),
    ],
    # 「选择频道」和「选择角色」的标题在**同一位置、同一块深红标牌**上，只靠文字
    # 区分。这里必须**只裁文字、不含红底**：带着红底时背景占了相关性的大头，
    # 实测「选择频道」的模板打在选角界面上仍有 0.82（阈值 0.80）→ 误判成选频道。
    # 只裁文字后：自帧 1.00，对方界面上 0.64，余量足够。
    UI_CHANNEL_LIST: [
        ("channel_title", (38, 64, 120, 32), 12.0, "「选择频道」四个字（只裁文字，不含深红底）"),
    ],
    # 面板顶条：左边「蓝蜗牛」大标题 + 右边「到选择的世界去」。
    # 故意用**一条宽的**而不是两个小锚点：宽条被鼠标指针挡住一小块也还能匹配上，
    # 而「多个锚点全中」的规则下，任何一个小锚点被挡住整个界面就判不出来。
    # 实战：自帧 1.00，其它界面 ≤ 0.58。
    UI_CHANNEL_PANEL: [
        ("channel_panel_top", (700, 380, 620, 50), 15.0,
         "频道面板顶条（「蓝蜗牛」大标题 + 「到选择的世界去」按钮）"),
    ],
    # 弹窗里**避开人数**（「11 名」会变），只用两处固定文案：
    # 实测真帧 ≥ 0.995，其它界面 ≤ 0.67。
    UI_QUEUE: [
        ("queue_waiting", (866, 460, 158, 62), 17.0,
         "弹窗里两行固定文案「当前世界人数较多 / 正在排队进入游戏」"),
        ("queue_cancel", (926, 572, 72, 52), 17.0, "「取消」按钮（连边框一起裁）"),
    ],
    UI_CHAR_SELECT: [
        ("char_title", (38, 64, 120, 32), 19.5, "「选择角色」四个字（只裁文字，不含深红底）"),
    ],
}

_cache = {}


def anchor_rects():
    """展开成 [(界面id, 模板名, 矩形, 源帧秒数, 说明)]，供生成工具和自检用。"""
    return [(ui, name, rect, ts, desc)
            for ui, anchors in ANCHORS.items()
            for name, rect, ts, desc in anchors]


def template_path(name):
    return TEMPLATE_DIR / (name + ".png")


def load_template(name):
    """读模板图并缩到匹配比例（带缓存）。缺失或没有对比度返回 None。

    纯色模板不能用：TM_CCOEFF_NORMED 要除以模板标准差，全平会算出 NaN。
    """
    if name in _cache:
        return _cache[name]
    tpl = None
    p = template_path(name)
    if p.exists():
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)   # 文件名是 ASCII，不用绕中文路径
        if img is not None and float(img.std()) >= 3.0:
            if MATCH_SCALE != 1.0:
                img = cv2.resize(img, None, fx=MATCH_SCALE, fy=MATCH_SCALE,
                                 interpolation=cv2.INTER_AREA)
            if img.shape[0] >= 4 and img.shape[1] >= 4:
                tpl = img
    _cache[name] = tpl
    return tpl


def clear_cache():
    """丢掉模板缓存（重新生成模板图后调用）。"""
    _cache.clear()


def missing_templates():
    """返回还没生成 / 不可用的模板名；空列表表示判别器可用。"""
    return [name for _ui, name, _r, _t, _d in anchor_rects()
            if load_template(name) is None]


def available():
    """模板齐不齐 —— 不齐就别调 detect（会一直返回 None）。"""
    return not missing_templates()


def prepare(frame):
    """把整帧统一到「基准分辨率 x MATCH_SCALE」。"""
    h, w = frame.shape[:2]
    want_w = int(BASE_W * MATCH_SCALE)
    want_h = int(BASE_H * MATCH_SCALE)
    if (w, h) == (want_w, want_h):
        return frame
    interp = cv2.INTER_AREA if w > want_w else cv2.INTER_LINEAR
    return cv2.resize(frame, (want_w, want_h), interpolation=interp)


def _score(img, tpl, rect=None, margin=SEARCH_MARGIN):
    """模板匹配的最高分（0~1）。模板比图大时返回 -1（不匹配）。

    rect 给了就只在「登记位置 ± margin」的小窗口里找（快很多，见 SEARCH_MARGIN）；
    传 None 则整帧扫（排查用）。
    """
    ih, iw = img.shape[:2]
    th, tw = tpl.shape[:2]
    if th > ih or tw > iw:
        return -1.0
    roi = img
    if rect is not None:
        x, y, w, h = rect
        m = int(margin * MATCH_SCALE)
        x0 = max(0, int(x * MATCH_SCALE) - m)
        y0 = max(0, int(y * MATCH_SCALE) - m)
        x1 = min(iw, int((x + w) * MATCH_SCALE) + m)
        y1 = min(ih, int((y + h) * MATCH_SCALE) + m)
        if x1 - x0 < tw or y1 - y0 < th:
            return -1.0
        roi = img[y0:y1, x0:x1]
    res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
    return float(np.nanmax(res))


def scores(frame):
    """调试用：每个锚点的匹配分 + 各界面是否判定命中。

    返回 ({"锚点名": 分数}, {"界面id": bool})。
    """
    img = prepare(frame)
    per = {}
    for _ui, name, rect, _t, _d in anchor_rects():
        tpl = load_template(name)
        per[name] = -1.0 if tpl is None else _score(img, tpl, rect)
    hit = {}
    for ui, anchors in ANCHORS.items():
        names = [n for n, _r, _t, _d in anchors if load_template(n) is not None]
        hit[ui] = bool(names) and all(per[n] >= THRESHOLD for n in names)
    return per, hit


def detect(frame, threshold=THRESHOLD):
    """判定当前界面。返回界面 id（UI_* 之一）或 None（未知）。

    frame: BGR ndarray（抓屏或收流拿到的整帧画面）。

    注意「未知」不等于「在游戏里」：登录界面收起登录木牌后的「连接中」
    阶段（素材里 5.5~11.5s）也是只有背景、判不出来，它同样返回 None。
    所以调用方不能把 None 当成「已经回到游戏」——要靠别的信号（玩家框）判断。
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    img = prepare(frame)
    for ui in PRIORITY:
        anchors = ANCHORS.get(ui) or []
        ready = [(n, r) for n, r, _t, _d in anchors if load_template(n) is not None]
        if not ready:
            continue
        if all(_score(img, load_template(n), r) >= threshold for n, r in ready):
            return ui
    return None
