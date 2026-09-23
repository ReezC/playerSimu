"""检测类别表：**类别 id / 英文名 / 中文名 / 默认颜色的唯一定义处**。

**为什么要有它**
    这几样原来散在 6 个地方（gui/labelio.py、gui/live_thread.py ×2、gui/canvas.py、
    perception/predict.py、perception/prepare_dataset.py），加一个类别要改 6 处、
    漏一处就是「框颜色/名字对不上」或者「标注写成别的类」。现在都从这里 import。

**id 是固定的（`ORDER` 就是权威顺序）—— 把它当成座位号**
    程序里到处按号认类（实时识别、自动标注、迭代都写着「1 号 = 怪物」这种判断），
    所以：某个类别暂时没有数据，也要占住它的号，不能顺着重排。

    一旦变号，已经训好的权重和以前标好的数据就会被**张冠李戴**（1 号的框被当成
    别的类），而且不报错、照常画框 —— 极难发现。原来的做法就是「只声明有数据的
    类、连续重编号」，某次数据集里没有玩家帧，全体号就漂了一位。
    （详细说明与代价对比见 perception/prepare_dataset.py 里「座位号」那段。）

**颜色：每个类别都能在设置里改**
    最后一列是**默认色**（表格里到处用 BGR）。实际画框用的颜色从设置读：
    `gui/theme.py` 的 vis 段，键名 = 英文名 + "_color"（player_color / mob_color /
    drop_color / npc_color / other_player_color），实时预览、质检台、推理可视化
    用的是同一份，改完立刻生效、不用重开。

    本模块**不 import PyQt5、不读配置** —— 标注 worker（子进程）也要能 import 它，
    所以这里只放默认色；要真实颜色请用 `gui.theme.class_colors()`，
    拿不到配置时（比如纯命令行）才退回这里的默认色。

本模块**不 import PyQt5、不读配置** —— 标注 worker（子进程）也要能 import 它。
"""

CLASS_PLAYER = 0
CLASS_MOB = 1
CLASS_DROP = 2
CLASS_NPC = 3
CLASS_OTHER_PLAYER = 4      # 其他玩家：目前只画框、不进决策（见 gui/live_thread.py）

#: (id, 英文名, 中文名, 默认色 BGR)
#: 顺序即权威 id —— **不要在中间插队或删除**，那会让已经训好的权重整体错位。
#: 默认色只是兜底（读不到配置时用）；真实颜色在设置里改，见文件开头说明。
#: 玩家蓝 #4285f4 = BGR(244,133,66)，怪物绿 #34a853 = BGR(83,168,52)。
CLASSES = (
    (CLASS_PLAYER,       "player",       "玩家",     (244, 133, 66)),
    (CLASS_MOB,          "mob",          "怪物",     (83, 168, 52)),
    (CLASS_DROP,         "drop",         "掉落",     (0, 255, 255)),
    (CLASS_NPC,          "npc",          "NPC",      (0, 0, 255)),
    (CLASS_OTHER_PLAYER, "other_player", "其他玩家", (255, 0, 255)),
)

ORDER = [c[0] for c in CLASSES]
EN_NAMES = {c[0]: c[1] for c in CLASSES}
ZH_NAMES = {c[0]: c[2] for c in CLASSES}
EN_LIST = [c[1] for c in CLASSES]              # 按 id 索引的列表（数据集导出用）


def label(cls, fallback=None):
    """类别 id → 中文名；未知 id 回退成 fallback（默认就是 id 的字符串）。"""
    return ZH_NAMES.get(cls, fallback if fallback is not None else str(cls))


def bgr_to_rgb(bgr):
    """BGR → RGB（Qt 的 QColor 要 RGB）。"""
    b, g, r = (list(bgr) + [0, 0, 0])[:3]
    return (int(r), int(g), int(b))


def bgr_to_hex(bgr):
    r, g, b = bgr_to_rgb(bgr)
    return "#%02x%02x%02x" % (r, g, b)


def hex_to_rgb(hexstr):
    return (int(hexstr[1:3], 16), int(hexstr[3:5], 16), int(hexstr[5:7], 16))
