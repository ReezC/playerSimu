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
    断线提示框是盖在登录界面**上面**的浮层，两个界面会同时可见 ——
    所以按「浮层优先」判定：先 login_err，再 login。

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

# 界面 id
UI_LOGIN = "login"          # 登录界面（登录木牌可见，等按「连接」）
UI_LOGIN_ERR = "login_err"  # 断线提示框「与服务器连接发生错误」浮在上面

# 判定优先级：浮层在前 —— 提示框出现时登录界面的锚点也可能命中，先报提示框
PRIORITY = (UI_LOGIN_ERR, UI_LOGIN)

# 锚点表：界面 id -> [(模板名, 裁剪矩形(x, y, w, h) @1920x1080, 源帧秒数, 说明)]
# 加新界面（选频道 / 选角 …）先在这里登记，再跑 tools/make_ui_templates.py 生成图。
ANCHORS = {
    UI_LOGIN: [
        ("login_connect", (1085, 505, 135, 70), 4.5, "登录木牌上的「连接」按钮"),
        ("login_agree", (965, 602, 425, 36), 4.5, "木牌下方「抵制不良游戏…」两行协议文字"),
    ],
    UI_LOGIN_ERR: [
        ("login_err_text", (748, 386, 245, 62), 0.2,
         "提示框里「红圈图标 + 与服务器连接发生错误」那一条"),
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


def _score(img, tpl):
    """模板匹配的最高分（0~1）。模板比图大时返回 -1（不匹配）。"""
    ih, iw = img.shape[:2]
    th, tw = tpl.shape[:2]
    if th > ih or tw > iw:
        return -1.0
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    return float(np.nanmax(res))


def scores(frame):
    """调试用：每个锚点的匹配分 + 各界面是否判定命中。

    返回 ({"锚点名": 分数}, {"界面id": bool})。
    """
    img = prepare(frame)
    per = {}
    for _ui, name, _r, _t, _d in anchor_rects():
        tpl = load_template(name)
        per[name] = -1.0 if tpl is None else _score(img, tpl)
    hit = {}
    for ui, anchors in ANCHORS.items():
        names = [n for n, _r, _t, _d in anchors if load_template(n) is not None]
        hit[ui] = bool(names) and all(per[n] >= THRESHOLD for n in names)
    return per, hit


def detect(frame, threshold=THRESHOLD):
    """判定当前界面。返回界面 id（UI_LOGIN / UI_LOGIN_ERR）或 None（未知）。

    frame: BGR ndarray（抓屏或收流拿到的整帧画面）。

    注意「未知」不等于「在游戏里」：登录界面收起登录木牌后的「连接中」
    阶段（素材里 5.5~11.5s）也是只有背景、判不出来，它同样返回 None。
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    img = prepare(frame)
    for ui in PRIORITY:
        anchors = ANCHORS.get(ui) or []
        names = [n for n, _r, _t, _d in anchors if load_template(n) is not None]
        if not names:
            continue
        if all(_score(img, load_template(n)) >= threshold for n in names):
            return ui
    return None
