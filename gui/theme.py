"""界面主题：目前只管字号。

**为什么需要这一层**
    字号原本散落在十几处**内联样式**里（写死像素值的那种）。
    内联样式的优先级高于 QApplication 的全局字体，不清理掉的话
    setFont() 改完看起来毫无反应 —— 这正是"想调大字体"最难搞的地方。

    所以策略是：
      1. 内联样式**只保留颜色**，字号交给样式表
      2. 字号由这里按基准值统一推导（缩放时改用 QApplication.setFont，
         因为所有控件都继承它）

**为什么用基准值 + setFont 而不是直接写 QSS**
    setFont 会让所有未指定字号的控件立即跟随，无需重建界面，
    改完当场可见。
"""

from pathlib import Path

import yaml

from perception import classes   # 类别表（唯一定义处）：框颜色按它生成

#: 「**入口按钮**」样式：点了会**打开一个干活的地方**（不是"运行任务"，也不是"查看结果" ✓）。
#: 现在用它的是：**「寻路编辑器」**（gui/route_panel.py）、**「打开质检台」**（⑤ 卡片，
#: gui/steps/cards.py）。为什么单拎成一个角色：这套配色是按**角色**分的 ——
#: 蓝 `#1a73e8` = 运行/主按钮、绿 = 通过、橙 = 提醒、红 = 错误/停止、紫 = 编辑序列；
#: "打开一个地方"原来只有寻路编辑器一个按钮，2026-09-27 起有两个了 ⇒ 颜色值和含义
#: 都放**这一处**，别两份各写一遍 ✗（约定 10）。
#: ⭐⭐ **「确定 / 取消」的统一配色**（用户 2026-09-28 要求 ✓ 原话："遍历项目所有的确定、
#:   取消按钮，给他们设计按钮颜色并统一，然后把这个列入 UI 规范"）。
#:   · **确定 = 主按钮**：蓝底白字（`#1a73e8` ✓ 就是本文件第 27 行那条约定里的"运行/主按钮"✓）
#:     —— 一个对话框里**主操作只有这一种颜色** ✓（跟"绿=通过 / 橙=提醒 / 红=错误"不冲突 ✓）。
#:   · **取消 = 次按钮**：浅灰底 + 深灰字（`#f0f2f5` / `#5f6368` ✓ 与下面 `TAB_QSS` 的
#:     未选中页签同色 ✓）⇒ 一眼看出"它不是主操作" ✓。
#:   · ⚠ **只认这两个名字**（`确定` / `取消` ✓）—— 别的按钮（"应用"/"重置"/"删除"）**不套** ✗：
#:     那些各有语义色（红=破坏性 ✓），统一到这里会把语义抹掉 ✗。
OK_BTN_QSS = (
    "QPushButton { background:#1a73e8; color:#ffffff; font-weight:600;"
    " border:1px solid #1a73e8; border-radius:4px; padding:5px 16px; }"
    "QPushButton:hover { background:#1765cc; border-color:#1765cc; }"
    "QPushButton:pressed { background:#14539f; border-color:#14539f; }"
    "QPushButton:disabled { background:#c3c7cb; border-color:#c3c7cb; color:#ffffff; }")

CANCEL_BTN_QSS = (
    "QPushButton { background:#f0f2f5; color:#5f6368;"
    " border:1px solid #dadce0; border-radius:4px; padding:5px 16px; }"
    "QPushButton:hover { background:#e4e7eb; color:#202124; }"
    "QPushButton:pressed { background:#d8dce0; }"
    "QPushButton:disabled { color:#b0b4b8; border-color:#eceff1; }")


def unify_ok_cancel(ok=None, cancel=None):
    """把一对「**确定 / 取消**」按钮刷成项目统一配色（用户 2026-09-28 ✓）。

    用法（**一行**，放在按钮建好之后 ✓）：
        theme.unify_ok_cancel(bb.button(QDialogButtonBox.Ok),
                              bb.button(QDialogButtonBox.Cancel))
      或  theme.unify_ok_cancel(btn_ok, btn_cancel)

    ⚠ 传 `None` 就跳过那一半 ✓（有的界面只有"关闭"一个按钮 ✓）；
    ⚠ 只**加 QSS**、不动文字/尺寸/信号 ✓（各处的文案已经是"确定/取消"✓ 别在这儿改 ✗）；
    ⚠ **规范**：新建任何带"确定/取消"的对话框，**都必须调它一次** ✓
      （见 `docs/UI规范.md` 的按钮那一条 ✓）。
    """
    if ok is not None:
        try:
            ok.setStyleSheet(OK_BTN_QSS)
        except Exception:                             # noqa: BLE001 —— 样式失败不该拖垮界面 ✓
            pass
    if cancel is not None:
        try:
            cancel.setStyleSheet(CANCEL_BTN_QSS)
        except Exception:                             # noqa: BLE001
            pass


ENTRY_BTN_QSS = (
    "QPushButton { background:#e8eaf6; border:1px solid #9fa8da;"
    " color:#283593; font-weight:600; padding:4px 10px; }"
    "QPushButton:hover { background:#c5cae9; }")

FAMILY = "Microsoft YaHei UI"

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config" / "ui.yaml"

DEFAULT_SIZE = 11
MIN_SIZE = 9
MAX_SIZE = 20

#: 设置弹窗的页签栏样式。**两个界面共用这一份**（工作台 gui/settings_dialog.py、
#: 部署台 deploy/settings_dialog.py）。
#:
#: 为什么放这里：主窗口那份 QSS 是 `setStyleSheet()` 打在**主窗口**上的，对话框不
#: 继承 —— 所以每个对话框都得自己带一份。而各写一份必然漂移，最后表现为"两个界面
#: 的设置弹窗长得不一样"，那是最容易让人以为装错版本的那种差异。
TAB_QSS = """
QTabWidget::pane { border: 1px solid #e2e5ea; border-radius: 6px;
                   background: #ffffff; top: -1px; }
QTabBar::tab { background: #f0f2f5; color: #5f6368; padding: 6px 14px;
               border: 1px solid #e2e5ea; border-bottom: none;
               border-top-left-radius: 6px; border-top-right-radius: 6px;
               margin-right: 2px; }
QTabBar::tab:selected { background: #ffffff; color: #202124; font-weight: 600; }
QTabBar::tab:hover { color: #202124; }
"""


def clamp(v):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return DEFAULT_SIZE
    return max(MIN_SIZE, min(MAX_SIZE, v))


# ---- 框颜色：**每个类别一条**，键名 = 类别英文名 + "_color" ----
#
# 直接从类别表（perception/classes.py）生成，所以「设置里能改的颜色」永远等于
# 「存在的类别」—— 以后加一个类别，设置里自动多一行，不会漏掉某一类的框色。
CLASS_COLOR_KEYS = tuple(c[1] + "_color" for c in classes.CLASSES)

#: ⭐⭐ **每一类检出的「显示开关」**（用户 2026-10-06 ✓ 原话："在设置→界面页签→
#:   检测框颜色 每一项前面加勾选框，默认全部勾选，勾选后实时显示"）—— 键名 = 类别
#:   英文名 + "_on"（`player_on` / `mob_on` / `drop_on` / `npc_on` / `other_player_on`
#:   / `pet_on` ✓）。
#:
#: ⚠ 与颜色**同一个来源**（类别表 ✓）：以后加一个类别，颜色 + 开关**自动各多一条**
#:   —— 加类别时只改 `perception/classes.py` 一处就够 ✓（这正是用户当初要的"按类别表
#:   生成"那个口径 ✓ 别在这儿手写一份清单 ✗）。
#: ⚠ 语义**只管画**（实时预览里那一类的框画不画 ✓）—— **一律不影响决策** ✗
#:   （决策那条链的入口是 `live_thread.split_dets` ✓，它根本不认这几个键 ✓）：
#:   所以关掉"宠物"只是**看不见**它的框，不是"不理它"✓（后者本来就没发生过 ✓）。
CLASS_ON_KEYS = tuple("%s_on" % c[1] for c in classes.CLASSES)


def _class_color_defaults():
    """类别框颜色的默认值（RGB hex）—— 就是类别表（perception/classes.py）的最后一列。"""
    return {en: classes.bgr_to_hex(bgr)
            for _cid, en, _zh, bgr in classes.CLASSES if bgr}


# 可视化默认值（颜色用 RGB hex 或 #AARRGGBB；vision_width 是视野线宽，像素）
#
# ⚠ 2026-09-27 用户要求：
#   ① 颜色**支持透明度**（设置里用拖动条配）⇒ 取值可以是 `#RRGGBB`（不透明，老配置 ✓）
#      或 `#AARRGGBB`（带 alpha）✓，所以 `_valid_color` 两种长度都收 ✓；
#   ② 「辅助线与标记」组里**每项前面有开关**（`*_on`，True = 画 ✓）——
#      关掉只是不画，颜色留着（下次开回来还在 ✓）。
VIS_DEFAULTS = dict(
    # 类别框颜色（按类别表生成 ✓）+ 每类检出的**显示开关**：**默认全勾**（= 画 ✓ 用户
    # 2026-10-06 点名"默认全部勾选"✓）—— 于是"加这一组开关"本身**不改任何观感** ✓
    # （老配置里没有这些键 ⇒ 读出来也是 True ✓）。
    {**{"%s_color" % en: v for en, v in _class_color_defaults().items()},
     **{k: True for k in CLASS_ON_KEYS}},
    lock_color="#ff0000",         # 锁定目标框 红
    attack_color="#f9ab00",       # 「攻击范围框」黄（2026-09-27 前叫"最大攻击距离线"）
    min_attack_color="#ffa500",   # 「攻击盲区框」橙（前叫"最小攻击距离线"）
    # 「跳跃攻击范围框」（2026-09-27 新增的**占位项**）：跳跃物理还没做 ⇒ 现在**不画** ✓，
    # 只让用户先把颜色定下来（逻辑见 `decision/agent.py` 里那条"记一笔"）。
    jump_attack_color="#8ab4f8",
    # 「追击起跳框」（2026-09-27 用户要求：以前是条**绿线**、颜色还写死在 live_thread 里 ✗）
    # ⇒ 现在是一个**框**，颜色搬到这里（约定 26：色值只写 theme 一处 ✓）。
    # 默认值 = 原来那条线用的绿 `(0, 200, 0)` ⇒ 观感一点不变 ✓。
    chase_jump_color="#00c800",
    vision_color="#5f6368",       # 视野线 深灰
    vision_width=1,               # 视野线宽
    # ⭐⭐ **各框的线宽**（用户 2026-09-28 要求 ✓ 原话："**给其他的粗细也加配置**，
    #   并且用**网格对齐**，要求**参数名显示完整**"）。
    #   ⚠ 默认值 = **现在的写死值**（一个字都不许改观感 ✗）：攻击/盲区/追击框在原代码里
    #     是 `thickness=2` ✓、锁定框是 `3` ✓ ⇒ 所以默认分别给 2 / 2 / 2 / 2 / 3 ✓。
    #   ⚠ 抄 `vision_width` 那套：**只在这一组里配、只影响实时预览的画法** ✓（不参与决策 ✓）。
    lock_width=3,                 # 锁定框线宽（原写死 3）
    attack_width=2,               # 攻击范围框线宽（原写死 2）
    min_attack_width=2,           # 攻击盲区框线宽（原写死 2）
    jump_attack_width=2,          # 跳跃攻击范围框线宽（原写死 2）
    chase_jump_width=2,           # 追击起跳框线宽（原写死 2）
    # 「当前任务 / 定时任务」那几行的字色（贴在实时画面上，**不铺底色**，只有描边）——
    # 2026-09-26 用户要求"文字颜色在设置里配"。默认取亮黄：它要压在游戏画面上，
    # 暗色读不清（这里没有黑底可衬）。
    timer_color="#ffeb3b",
    # ⭐⭐ **玩家坐标箭头**（用户 2026-09-28 ✓ 原话："放在**设置 → 界面页签 → 辅助线与标记
    #   （实时预览）**"）：以**玩家脚底**为原点画两根 —— 一根朝画面右（世界 x 正方向）、
    #   一根朝上（世界 y 正方向）✓。
    #   ⚠ 它干的活是**照着调"脚底偏移"**（那个仍在「决策参数 → 玩家位置」✓）：箭头根部
    #     该正好落在脚底 ✓，偏了就是偏移没调对 ✓。
    #   ⚠ 颜色/粗细/长度**都归这一组**（规范 §4："画面上**每个可显示的东西**都该能在这一组里
    #     单独关掉" ✅ ⇒ 因此有 `player_arrow_on` 开关 ✓；长度 0 = 不画 ✓）。
    player_arrow_color="#00e5ff",
    player_arrow_width=2,         # 线宽(px)
    player_arrow_len=60,          # 线段长(px)，0 = 不画
    # ⭐ **箭头尖大小**（用户 2026-09-28 追加 ✓ 原话："再加个箭头 size 配置"）：
    #   就是两根箭头**尖端那个三角头**的大小 ✓ —— 单位是**占线段长的百分比**
    #   （25 = 尖长正好是线长的 1/4 ✓）。⚠ 这不是我另立的名目：它就是
    #   `cv2.arrowedLine` 的 `tipLength` 参数，而它**只吃比例、不吃像素** ✗
    #   ⇒ 所以这里直接按百分比存，画的时候 `/100` ✓。
    #   ⚠ 默认 25 = **原代码里写死的 `tipLength=0.25`** ✓（加参数**不许改观感** ✗）。
    player_arrow_tip_pct=25,
    # ---- 每项的**显示开关**（用户 2026-09-27："辅助线与标记组里每项参数前加开关"）----
    lock_on=True,                 # 锁定框
    attack_on=True,               # 攻击范围框
    min_attack_on=True,           # 攻击盲区框
    jump_attack_on=True,          # 跳跃攻击范围框（逻辑还没有 ⇒ 开关先摆着 ✓）
    chase_jump_on=True,           # 追击起跳框
    vision_on=True,               # 视野线
    timer_on=True,                # 「当前任务 / 定时任务」那几行字
    player_arrow_on=True,         # 玩家坐标箭头（⭐ 用户 2026-09-28 要求搬进这一组 ✓）
    # ⭐⭐ **x 箭头 = 玩家真实朝向**（用户 2026-10-10 ✓ 原话："**我想让坐标箭头代表玩家的
    #   真实朝向（需要尽量高效）**"✓）—— 勾上 ⇒ 那根 x 箭头不再画「世界 x 正方向」✗，
    #   改成按**真实朝向**朝左 / 朝右 ✓（判据 = `live_thread.move_face_dir` ✓：
    #   **世界坐标** x（`Player.world_x` ✓ **与相机无关** ✓）连几拍**同向**且净位移够 ✓
    #   —— 成本 **0** ✓ 不裁图、不标定、不用模板 ✓；
    #   形状 / 颜色 / 粗细 / 长度 / 尖端**全沿用**上面那几格 ✓ ⇒ 不加外观参数 ✓）。
    #   ⚠⚠ **别用画面 x** ✗（用户 2026-10-10 纠正 ✓："**应该用世界状态坐标的位移算，
    #     因为画面里相机镜头是会移动的**"✓）：镜头一动画面就假了 ✓。
    #   ⚠ **只有角色在走时才有读数**：停下 / 抖动 / 这一拍没定位 ⇒ 这根箭头**不画** ✓（不猜 ✓）。
    #   ⛔ 别再走"猜外观"那条路 ✗（"玩家框左右亮度差 + 走目标定" ✓ 用户 2026-10-10 实测
    #     "**并不灵敏**"✗ ⇒ 已撤 ✓ 三条硬伤见 `live_thread._face_beat` 的撤销留痕 ✓）。
    #   ⚠⚠ **不许**读 `ws.player.facing` ✗：那个字段名义上就是"视觉朝向" ✓，但它唯一的写者
    #     （`player_state.py` 的 `@L` 镜像模板链 ✓）**没接进实时回路** ⇒ 恒 `+1` ⇒
    #     **箭头永久朝右** ✗（用户 2026-10-10 现场报的正是它 ✓）。
    #   ⚠⚠ **它与「攻击范围框」不是同一份** ✗：那个框按 **Agent 的按键 / 意图**
    #     （`action["facing"]` ✓）走 —— **两者不一致的时候，正是这根箭头存在的理由** ✓
    #     （⛔ 别去"统一"它们 ✗）。
    #   ⚠ **没标定好 / 这一拍看不出来 ⇒ 这根箭头不画** ✓（`arrow_real_face_dir` 返回 `0` ✓
    #     宁缺勿错 ✓ 绝不假装朝右 ✓）。
    #   ⚠ **默认关** ✓（老观感一字不变 ✓）；勾着时 x 箭头就**不再**能验"世界正方向"了
    #     —— y 箭头照旧是「世界 y 正方向」✓，「脚底偏移」仍可对着它调 ✓。
    #   ⚠⚠ **故意不放进 `_VIS_ON_KEYS`** ✗：那个名单的语义是"缺省 / 写坏 ⇒ **True**（画 ✓）"
    #     ⇒ 硬塞进去会让**所有人**的箭头默认改成"真实朝向" ✗（= 偷改观感 ✗）。它必须是
    #     "缺省 ⇒ **False**" ✗ ⇒ 在 `load_vis` 里**单独一道读回** ✓。
    player_arrow_real_face=False,
)

#: 参与「颜色合法性校验」的键：类别框色 + 辅助框/线色（vision_width 是数值，不在内）
_VIS_COLOR_KEYS = CLASS_COLOR_KEYS + ("lock_color", "attack_color",
                                      "min_attack_color", "jump_attack_color",
                                      "chase_jump_color",
                                      "vision_color", "timer_color",
                                      "player_arrow_color")

#: 「显示开关」的键（布尔；缺省/写坏一律按 `True` = 画 ✓ —— 老配置里没有它们 ✓）
#: ⚠ 前六个是**类别框的开关**（`CLASS_ON_KEYS` ✓ 用户 2026-10-06 ✓）；后面那些是
#:   「辅助线与标记」的 ✓ —— 两类**同一套读回口径**（`load_vis` ✓ 只认真布尔 ✓）。
_VIS_ON_KEYS = CLASS_ON_KEYS + ("lock_on", "attack_on", "min_attack_on", "jump_attack_on",
                                "chase_jump_on", "vision_on", "timer_on", "player_arrow_on")


def _load():
    try:
        with open(CFG, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save(data):
    try:
        CFG.parent.mkdir(parents=True, exist_ok=True)
        with open(CFG, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)
    except Exception:
        pass


def load_section(name):
    """读 `config/ui.yaml` 里的一个**自定义段**（→ dict 副本；没有/坏值 ⇒ `{}` ✓）。

    用途 = 某个窗口自己的"**上次配置**"（用户 2026-10-01 要求："让窗口能够记住我上次的
    配置" ✓ 例：测谎演示窗的 `lie_demo` 段 ✓）——与 `windows:` / `views:` 同一个文件
    （**按客户端唯一一份** ✓ 不进项目 ✓ 换项目不该改界面偏好 ✓）。
    """
    try:
        d = _load()
    except Exception:
        return {}
    v = d.get(name) if isinstance(d, dict) else None
    return dict(v) if isinstance(v, dict) else {}


def save_section(name, data):
    """整体替换 `config/ui.yaml` 里的一个自定义段（**其余键一个不动** ✓）。"""
    try:
        d = _load()
    except Exception:
        d = {}
    if not isinstance(d, dict):
        d = {}
    d[name] = dict(data or {})
    _save(d)


# ══════════════════════════════════════════════════════════
# 质检台 · 蒙版透明度（2026-09-29 用户要求 ✓）
# ══════════════════════════════════════════════════════════
#: 看帧时在**画面上**罩一层灰蒙版（标注框画在蒙版上面、不受影响 ✓）——
#: 观察"框和目标对不对"时把画面压暗，框更跳 ✓。存 `config/ui.yaml`
#: 的 `review_mask_alpha`（0~1；**这台机器上的界面偏好**，跟项目无关 ✓）。
REVIEW_MASK_DEFAULT = 0.45


def review_mask_alpha():
    """→ 蒙版透明度（0~1，越大战检台画面越暗；0 = 不罩 ✓）。"""
    try:
        v = float((_load() or {}).get("review_mask_alpha", REVIEW_MASK_DEFAULT))
    except Exception:
        v = REVIEW_MASK_DEFAULT
    return max(0.0, min(1.0, v))


def set_review_mask_alpha(v):
    """写回蒙版透明度（截到 [0,1] ✓），返回生效值。"""
    d = _load() or {}
    d["review_mask_alpha"] = max(0.0, min(1.0, float(v)))
    _save(d)
    return d["review_mask_alpha"]


# ══════════════════════════════════════════════════════════
# yolo 工作台 · 蒙版透明度 / 界面字号（2026-09-30 用户要求 ✓）
# ══════════════════════════════════════════════════════════
#: 与质检台 `review_mask_alpha` 同一套画法（画布挖洞那套 ✓）、**各自的偏好键** ✓
#:（两个窗口想暗得不一样是正常的 ✓）。存 config/ui.yaml ✓。
YOLO_MASK_DEFAULT = 0.45


def yolo_mask_alpha():
    """→ yolo 工作台标注页的蒙版浓淡（0~1 ✓；0 = 不罩 ✓）。"""
    try:
        v = float((_load() or {}).get("yolo_mask_alpha", YOLO_MASK_DEFAULT))
    except Exception:
        v = YOLO_MASK_DEFAULT
    return max(0.0, min(1.0, v))


def set_yolo_mask_alpha(v):
    """写回蒙版浓淡（截到 [0,1] ✓），返回生效值。"""
    d = _load() or {}
    d["yolo_mask_alpha"] = max(0.0, min(1.0, float(v)))
    _save(d)
    return d["yolo_mask_alpha"]


def ui_font_pt(default=9):
    """→ yolo 工作台的界面字号（pt ✓；默认 9 ✓）。"""
    try:
        return int((_load() or {}).get("ui_font_pt", default))
    except Exception:
        return int(default)


def set_ui_font_pt(v):
    """写回界面字号（int ✓）。"""
    d = _load() or {}
    d["ui_font_pt"] = int(v)
    _save(d)
    return d["ui_font_pt"]


def yolo_session():
    """→ yolo 工作台**上次关闭前的数据状态**（dict ✓；没存过 = {} ✓）。
    路径/类别名/手选模型/页签/当前帧/筛选排序/训练参数 ✓ —— 同为本机偏好 ⇒
    存 config/ui.yaml ✓（§11 同款 ✓）。"""
    try:
        d = (_load() or {}).get("yolo_session")
        return dict(d) if isinstance(d, dict) else {}
    except Exception:
        return {}


def set_yolo_session(sess):
    """写回 yolo 工作台的数据状态（整包覆盖 ✓）。"""
    d = _load() or {}
    d["yolo_session"] = dict(sess or {})
    _save(d)
    return d["yolo_session"]


# ══════════════════════════════════════════════════════════
# 弹窗几何：**按客户端唯一一份**（存 config/ui.yaml 的 windows:）
# ══════════════════════════════════════════════════════════
# 为什么也放这个文件：`config/ui.yaml` 是**这台机器上的界面配置**（字号 / 颜色 / 窗口几何）
# 的唯一读写口 —— 窗口大小和位置跟项目无关、跟地图无关（换项目不该把窗口挪回去 ✗），
# 所以"按客户端唯一一份"这件事在**存储层**就是这一处 ✓（用法见 `bind_window_state`）。

#: 几何合法性的下限：比它小 / 缺键 ⇒ 当"没存过"（手改坏的 yaml 别把窗口缩成一个点，
#: 人还以为程序坏了 ✗）
WIN_MIN_W, WIN_MIN_H = 320, 240

#: 恢复几何时**下边至少留这么多**在屏内（像素）—— 标题栏在窗口顶部 ⇒ 留出它必定抓得到 ✓
#: （用户 2026-09-27 报"弹窗顶部的栏在屏幕外面、拖不动"⇒ 这一条就是治它的 ✓）
_VIS_MIN = 80


def load_window(key):
    """→ `(x, y, w, h)`；**没存过 / 值不合法 ⇒ None**（调用方就用它自己的默认尺寸 ✓）。"""
    d = (_load().get("windows") or {}).get(str(key))
    if not isinstance(d, dict):
        return None
    try:
        x, y, w, h = (int(d[k]) for k in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return None
    if w < WIN_MIN_W or h < WIN_MIN_H:
        return None
    return (x, y, w, h)


def save_window(key, rect):
    """记下某个弹窗的位置与大小（`rect` = `(x, y, w, h)`）。"""
    cfg = _load()
    wins = dict(cfg.get("windows") or {})
    wins[str(key)] = {"x": int(rect[0]), "y": int(rect[1]),
                      "w": int(rect[2]), "h": int(rect[3])}
    cfg["windows"] = wins
    _save(cfg)


def _field_layout_of(form, w):
    """`QFormLayout` 里**装着 `w` 的那个字段布局**（`addRow("标题", 行布局)` 那种 ✓）。
    找不到 ⇒ `None` ✓（直接 `addRow("标题", 控件)` 的情况本来就不用它 ✓）。
    """
    from PyQt5.QtWidgets import QFormLayout

    try:
        rows = form.rowCount()
    except Exception:                                   # noqa: BLE001
        return None
    for i in range(rows):
        it = form.itemAt(i, QFormLayout.FieldRole)
        lay = it.layout() if it is not None else None
        if lay is not None:
            try:
                if lay.indexOf(w) >= 0:
                    return lay
            except Exception:                           # noqa: BLE001
                continue
    return None


def _row_title_of(w):
    """配置控件 `w` **那一行的标题标签**（找不到 ⇒ `None` ✓）。

    两种行都要认（gui 里的配置行就这两种 ✓）：
      · **`QFormLayout` 的行** —— `labelForField(控件或行布局)` ✓（`addRow("标题", w)` 和
        `addRow("标题", 装有 w 的 QHBoxLayout)` 都给得出来 ✓）；
      · **普通 `QHBoxLayout` 的行**（手拼的"标题 + 控件 + 按钮" ✓）—— 取同一布局里
        **排在它前面**的那个 `QLabel` ✓。
    """
    from PyQt5.QtWidgets import QFormLayout, QLabel

    holder = w.parentWidget()
    node, outer = (holder.layout() if holder else None), None
    # ⚠⚠ **防环是必须的**（2026-10-04 ✓ 现场：整个工作台**卡死在启动里** ✗ —— 窗口永远出不来、
    #   进程活着、没有任何异常 ✓ 用户报的就是"数据工作台打不开了"✓）：
    #   控件在**普通 HBox 行**里时，`node.parentWidget().layout()` 返回的**还是同一个布局** ✗
    #   ⇒ 原来那个 `while node is not None` **原地打转、永不退出** ✗ ⇒ UI 线程死锁 ✓
    #   （当时"PlayerPanel 探针跑不完 / 套件 400s 超时"就是这个 ✗，我误判成"面板太重"绕过去了 ✗）。
    #   ⇒ 两道保险：**见过的不再走** + 层数上限 ✓（就算将来又出现奇怪的父子关系，也只会
    #     "找不到标题 ⇒ 提示留在原处" ✓，绝不会再把界面卡死 ✗）。
    seen = set()
    for _ in range(24):
        if node is None or id(node) in seen:
            break
        seen.add(id(node))
        if isinstance(node, QFormLayout):
            # ① 直接当字段加进去的（`addRow("标题", w)` ✓）
            lab = node.labelForField(w)
            # ② 外面还套了一层/几层布局（`addRow("标题", 行布局)` ✓）—— 先看链上的那层，
            #    再退一步：**扫这一表单每一行的字段槽**，谁装着 w 谁就是那一行 ✓
            #    （⚠ 没有这一步，嵌套行的标题永远找不到 ⇒ 提示留在原处 ✗ 见离屏实测 ✓）
            if lab is None and outer is not None:
                lab = node.labelForField(outer)
            if lab is None:
                lab = node.labelForField(_field_layout_of(node, w))
            return lab if isinstance(lab, QLabel) else None
        outer = node
        parent = node.parentWidget()
        node = parent.layout() if parent else None

    lay = holder.layout() if holder else None
    if lay is not None:
        i = lay.indexOf(w)
        for k in range(i - 1, -1, -1):
            it = lay.itemAt(k)
            cand = it.widget() if it is not None else None
            if isinstance(cand, QLabel):
                return cand
    return None


def move_tips_to_titles(root, include_buttons=False):
    """把"挂在**配置控件**上的提示"搬到**它那一行的标题**上 → 返回搬了几处 ✓。

    ⚠ 为什么要有这个"自动扫"（用户 2026-10-04 二次反馈 ✓ 原话："**实测没有实现提示挂字段标题**……
      我说的是 gui **所有的配置项**"）：上一版只搬了**手写过的**那一处（卡片 14 处 ✓）✗ ——
      而 gui 里这样的提示有 **近 300 处**（`player_panel` 62 / `route_panel` 10 / …）✗
      逐个手搬既不现实、也**必然漏**（漏一处就还是"指标题没反应、指控件乱弹" ✓ 正是用户看到的 ✓）
      ⇒ 改成**在窗口收尾处自动扫一遍** ✓（`finish_window` ✓ 与"可复制"同一处保证 ✓）。

    规则：
      · 只搬**配置控件**（下拉 / 输入框 / 数字框 / 滑块 / 勾选框 ✓；`include_buttons=True`
        时连按钮一起搬 ✓ —— 默认不搬：工具栏按钮**没有字段标题**，搬了会把提示弄丢 ✗）；
      · 目标 = `_row_title_of(w)` ✓；⚠ **标题已经有提示就不覆盖** ✓（手写的更具体 ✓）；
      · ⚠ **找不到标题 ⇒ 原样留着** ✗（宁可留在控件上，也别把提示弄丢 ✓）；
      · 搬完把标题设成**可复制** ✓（用户第 4 条 ✓）。
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (QAbstractSlider, QAbstractSpinBox, QCheckBox,
                                 QComboBox, QLineEdit, QPushButton,
                                 QRadioButton, QWidget)

    kinds = (QComboBox, QLineEdit, QAbstractSpinBox, QAbstractSlider,
             QCheckBox, QRadioButton)
    if include_buttons:
        kinds = kinds + (QPushButton,)

    n = 0
    for w in root.findChildren(QWidget):
        tip = w.toolTip()
        if not tip or not isinstance(w, kinds):
            continue
        lab = _row_title_of(w)
        if lab is None or lab.toolTip():
            continue
        lab.setToolTip(tip)
        # ⚠ 口径与 `gui.widgets.make_copyable` 一致 ✓（那边是给"当场就要"的场景用的 ✓）——
        #   这里是整窗扫，顺手设上 ✓（免得标题搬了提示却选不中 ✗）。
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        w.setToolTip("")                 # ⚠ 搬完必须清掉 ✗ 否则一指弹两个框 ✓
        n += 1
    return n


def finish_window(w, copyable=True, move_tips=True):
    """**窗口 / 面板收尾**：① 提示搬到字段标题上 ② 标题与说明可复制 ✓ → 返回 `(搬了几处, 标签数)` ✓。

    ⚠ 为什么要单独一个"收尾"函数（原来把这活塞在 `bind_window_state` 里 ✗）：
      `bind_window_state` **只有弹窗在调** ✗ —— 而用户反馈的正是**页签里的面板**
      （决策参数 ✓）⇒ 那条路根本走不到 ✗（"实测没有实现"就是这么来的 ✓）。
      ⇒ 现在有两条路都走它：**弹窗**（`bind_window_state` 里调 ✓）与
      **主窗口 / 每个页签面板**（`MainWindow` 建完 + 页签工厂各调一次 ✓）。
    """
    n_tips = move_tips_to_titles(w) if move_tips else 0
    n_lbl = enable_label_copy(w) if copyable else 0
    return n_tips, n_lbl


def enable_label_copy(root, flags=None):
    """把这个窗口里**所有 QLabel 的文字设成可选中复制**（用户 2026-10-04 ✓ 第 4 条：
    "所有字段标题、灰字说明文本希望能够复制文本"）→ 返回改了多少个 ✓。

    ⚠ **为什么是"按窗口扫一遍"而不是一个全局开关**：Qt 没有"给所有 QLabel 默认开选中"的开关 ✗
      —— `setTextInteractionFlags` 是**每个控件**自己的属性 ✓。逐个调用点去设 ⇒ 全文上百处 ✗
      且**新加的标签必然漏** ✗；在窗口收尾处扫一遍 ⇒ 建完就全有了 ✓。
    ⚠ 只加 `TextSelectableByMouse`（**故意不加** `TextSelectableByKeyboard` ✓）：键盘选中会把
      每个标签变成**可聚焦** ⇒ Tab 顺序里多出一堆"只能选字"的停靠点 ✗，`←/→` 那类快捷键
      也会先在标签里跑 ✗。
    ⚠ **谁会调**：`bind_window_state()`（每个窗口 / 面板的收尾都走它 ✓ 见那里那一行）——
      新窗口只要照常调它就有 ✓ 别各写一份 ✗。
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QLabel

    n = 0
    try:
        for lbl in root.findChildren(QLabel):
            lbl.setTextInteractionFlags(flags if flags is not None
                                       else Qt.TextSelectableByMouse)
            n += 1
    except Exception as e:                            # noqa: BLE001
        print("⚠ 标签可复制没设上（%s: %s）" % (type(e).__name__, e))
    return n


def bind_window_state(dlg, key):
    """让一个弹窗**记住自己拉成多大、摆在哪儿**（用户 2026-09-27 要求，已进规范 ✓）。

    用法：对话框 `__init__` 的**最后一行** `theme.bind_window_state(self, "<键>")` ✓
    （键就是"哪个弹窗"，全局唯一；它自己在 `__init__` 里 `resize(...)` 的默认值仍然管用
    —— **第一次**打开时用它 ✓）。

    为什么"按客户端唯一一份"：窗口大小/位置是**这台机器上的人**的习惯，跟项目/地图无关
    ⇒ 存在 `config/ui.yaml`（和字号、颜色同一个文件 ✓），不进 `project.yaml` ✗。

    ⚠ 三件必须做对（这就是判据）：
      · **恢复时把窗口夹回屏幕内**（2026-09-27 用户报："每个弹窗打开时，其顶部的栏在**屏幕
        外面**，我都无法拖动" ✗）：外接屏拔了 / 换了分辨率或 DPI 之后，存下来的坐标可能悬空
        ⇒ 恢复时挑**重叠最多**的那块屏（一块都不沾 ⇒ 用主屏 ✓），把矩形**夹进它的可用区** ✓
        ⇒ **标题栏一定露在屏幕里** ✓（"沾到 1 像素就原样恢复"✗ 与 "全在屏外就不恢复"✗
        两种老写法都救不了这种情况 —— 用例钉着现在是"夹进来"✓）；
      · **Show 恢复 / Hide 保存**（不是 Show 保存 ✗）：非模态弹窗的 Esc、点 × 只是
        hide ⇒ 存晚了就丢了；
      · 存的是 **`geometry()`（含边框的窗口矩形）**，不是 `size()` —— 只存大小会让它
        每次都跳到屏幕左上角 ✗。
      · ⭐⭐ **位置不记、尺寸记**（用户 2026-09-28 改口径 ✓ 原话："**居中、放弃位置记忆，
        我期望的是调整过的缩放数据记忆要有**"）：每次显示都**居中到父窗口**（跟随主窗口 ✓，
        主窗口挪到哪它就在哪居中 ✓），只有**尺寸**沿用存下来的那对数 ✓；存下来的 x/y 只当
        极端情况下的兜底（夹屏幕用 ✓），**不再直接套用** ✗。
    ⚠ 为什么用事件过滤器、不去改每个对话框的 `showEvent`/`hideEvent`：那些方法**各家都
      在用**（比如标定窗按 show 重新接数据流），两边各写一遍迟早互相踩 ✗；过滤器挂在
      弹窗上（`QObject` 的父子关系 ⇒ 弹窗没了它也没了 ✓），谁都不用改自己的事件处理 ✓。
    """
    from PyQt5.QtCore import QEvent, QObject, QRect
    from PyQt5.QtGui import QGuiApplication

    # ⭐ **窗口收尾**：提示搬到字段标题上 + 标签可复制（用户 2026-10-04 ✓ 第 3、4 条）。
    #   ⚠ 这里只能覆盖**弹窗**（只有弹窗会调 `bind_window_state` ✓）—— 页签面板走的是
    #   `MainWindow` / 页签工厂里的 `finish_window` ✓ 两处都指同一份实现 ✓。
    finish_window(dlg)

    if getattr(dlg, "_winstate", None) is not None:
        return                          # 已经绑过（重复调用无害 ✓）

    class _WinState(QObject):
        """内部：给弹窗挂 Show / Hide 钩子（只由 `bind_window_state` 建 ✓）。"""

        def __init__(self, owner, wkey):
            super().__init__(owner)     # 挂在弹窗上 ⇒ 弹窗没了它也没了 ✓
            self.owner, self.wkey = owner, wkey

        def _restore(self):
            r = load_window(self.wkey)
            # ⭐⭐ **位置不记、尺寸记**（用户 2026-09-28 口径 ✓）：
            #   · **尺寸**用存下来的那对数 ✓（= 用户自己拉出来的大小 —— "调整过的**缩放数据
            #     记忆要有**" ✓）；没存过 ⇒ 用它自己的**默认尺寸** ✓（不是"跟着 0 走" ✗）；
            #   · **位置**一律**现算 = 居中到父窗口** ✓（= 主窗口 ✓ 跟随主窗口 ✓）——
            #     存下来的 x/y **不再直接套用** ✗（这正是用户抱怨的"寻路编辑器/战斗区域
            #     不居中"的根因：它们以前恢复的是**老位置** ✗）。
            #   · 父窗口拿不到（无 parent / 还没显示）⇒ 退回**主屏居中** ✓（不抛异常 ✗）。
            if r is None:
                w, h = int(self.owner.width()), int(self.owner.height())
            else:
                w, h = int(r[2]), int(r[3])
            w, h = max(_VIS_MIN, w), max(_VIS_MIN, h)
            par = self.owner.parentWidget()
            if par is not None and par.isVisible():
                pg = par.frameGeometry()
                x = pg.x() + (pg.width() - w) // 2
                y = pg.y() + (pg.height() - h) // 2
            else:
                scr0 = QGuiApplication.primaryScreen()
                av0 = scr0.availableGeometry() if scr0 is not None else None
                x = (av0.x() + (av0.width() - w) // 2) if av0 is not None else 0
                y = (av0.y() + (av0.height() - h) // 2) if av0 is not None else 0
            # ⚠ **夹回屏幕**（2026-09-27 用户报过："弹窗顶部的栏在**屏幕外面**，拖不动"✗）：
            #   居中之后一般都在屏内 ✓，这几手是给极端情况兜的（弹窗比屏还大 / 多屏且主窗口
            #   在别的屏上 ⇒ 挑**重叠最多**的那块屏接住 ✓，一块都不沾 ⇒ 主屏 ✓）。
            #   ⚠ **只夹位置、不缩尺寸**（2026-09-27 实测踩过 ✓）：缩尺寸会把"用户拉出来的
            #     大小"改掉 ⇒ 与"缩放数据记忆要有"冲突 ✗。
            box = QRect(x, y, w, h)
            av, best_area = None, 0
            for s in QGuiApplication.screens():
                g = s.availableGeometry()
                inter = g.intersected(box)
                area = inter.width() * inter.height()
                if area > best_area:
                    av, best_area = g, area
            if av is None:                      # 悬空：一块屏都没沾上 ⇒ 用主屏接住 ✓
                scr = QGuiApplication.primaryScreen()
                if scr is None:
                    return
                av = scr.availableGeometry()
            # 窗口**比屏幕还宽**时 `av.right() - w` 是负数 ⇒ 那时**贴左**，
            # 保证左边（含标题栏那一段）露在屏里 ✓（标题栏横跨整宽 ⇒ 露一段就能拖 ✓）
            x = min(max(x, av.left()), max(av.left(), av.right() - w))
            # 下边至少留 `_VIS_MIN` 在屏内（标题栏在窗口顶部 ⇒ 这样它必定可见 ✓）
            y = min(max(y, av.top()), max(av.top(), av.bottom() - min(_VIS_MIN, h)))
            self.owner.setGeometry(QRect(x, y, w, h))

        def _save(self):
            g = self.owner.geometry()
            save_window(self.wkey, (g.x(), g.y(), g.width(), g.height()))

        def eventFilter(self, obj, ev):         # noqa: N802（Qt 的命名）
            if obj is self.owner:
                t = ev.type()
                if t == QEvent.Show:
                    self._restore()
                elif t in (QEvent.Hide, QEvent.Close):
                    self._save()
            return False

    f = _WinState(dlg, key)
    dlg._winstate = f                   # 保住引用（虽然 parent 也保着 —— 写明白更好查 ✓）
    dlg.installEventFilter(f)


#: 画布缩放倍数的合法范围（超出当"没存过"⇒ 别让手改坏的 yaml 把视图缩成一个点 ✗）
VIEW_SCALE_MIN, VIEW_SCALE_MAX = 0.02, 50.0


def load_view_zoom(key):
    """某个画布上次的**缩放倍数**（`float`）；没存过 / 不合法 ⇒ `None` ✓。"""
    d = (_load().get("views") or {}).get(str(key))
    if not isinstance(d, dict):
        return None
    try:
        s = float(d.get("scale"))
    except (TypeError, ValueError):
        return None
    if not (VIEW_SCALE_MIN <= s <= VIEW_SCALE_MAX):
        return None
    return s


def save_view_zoom(key, scale):
    """记下某个画布的缩放倍数（同 `config/ui.yaml` 的 `views:` 段 ✓）。"""
    cfg = _load()
    views = dict(cfg.get("views") or {})
    views[str(key)] = {"scale": round(float(scale), 4)}
    cfg["views"] = views
    _save(cfg)


def bind_view_zoom(view, key):
    """让一个画布（`gui/canvas.ZoomPanView` 那类）**记住上次的缩放** ✓。

    用户 2026-09-27 报（原话）："我每次打开弹窗都要重新调缩放，之前的存缩放数据功能没做？"
    —— **窗口尺寸/位置**那件事其实早做了（`bind_window_state` 上面那段 ✓），他每次重调的是
    **画布里的 zoom** ✗（画布每次新建 / 换图都会 `fit()` 适应窗口 ⇒ 不记就等于回默认 ✗）。

    用法：建好视图之后，和 `bind_window_state` 并排一行：
        `theme.bind_view_zoom(self.view, "zone_editor_view")` ✓
    行为：视图 **Show** 时把上次的倍数按回去、**Hide / Close** 时记下来 ✓；
    存哪：同 `config/ui.yaml`（`views:` 段 ✓，键名 = 哪个视图、全局唯一 ✓）。

    ⚠ 恢复用 `QTimer.singleShot(0, …)` **延后一拍**：画布自己会在 show / 换图时 `fit()`
      （"适应窗口"是正常行为 ✓）—— 同步恢复会被它盖掉 ✗，延后一拍才稳 ✓。
    ⚠ 没存过 ⇒ **什么都不做**（让它照旧 `fit()` ✓，不猜倍数 ✗）；坏值一律当"没存过" ✓。
    """
    from PyQt5.QtCore import QEvent, QObject, QTimer

    class _ViewZoom(QObject):
        """内部：给画布挂 Show / Hide 钩子（只由 `bind_view_zoom` 建 ✓）。"""

        def __init__(self, owner, vkey):
            super().__init__(owner)         # 挂在画布上 ⇒ 画布没了它也没了 ✓
            self.owner, self.vkey = owner, vkey

        def _restore(self):
            s = load_view_zoom(self.vkey)
            if s is None:
                return
            self.owner.resetTransform()
            self.owner.scale(s, s)

        def _save(self):
            save_view_zoom(self.vkey, float(self.owner.transform().m11()))

        def eventFilter(self, obj, ev):         # noqa: N802（Qt 的命名）
            if obj is self.owner:
                t = ev.type()
                if t == QEvent.Show:
                    QTimer.singleShot(0, self._restore)     # 让画布自己的 fit 先跑完 ✓
                elif t in (QEvent.Hide, QEvent.Close):
                    self._save()
            return False

    f = _ViewZoom(view, key)
    view._viewzoom = f                  # 保住引用（parent 也保着 —— 写明白更好查 ✓）
    view.installEventFilter(f)


# ---- foothold 集合编辑器的线条宽度（gui/zone_editor.py）----
#
# 单位是**屏幕像素**，不是场景单位：编辑器的缩放跨度很大（整图 fit 时约 0.35 倍、
# 放大看细节能到 8 倍），按场景单位给宽度的话"同一个值"在两种视图下差 20 多倍 ——
# 要么整图时看不见线、要么放大时线糊成一片。所以这里定的是"屏幕上多粗"，
# 编辑器每次重绘都按当前缩放折算（见 gui/zone_editor._ZoneView.set_line_width）。
FOOTHOLD_W_DEFAULT = 1
FOOTHOLD_W_MIN = 1
FOOTHOLD_W_MAX = 8


def load_foothold_width():
    """读编辑器的线条宽度（屏幕像素，1~8）；文件坏了退回默认。"""
    try:
        v = int(_load().get("foothold_line_w", FOOTHOLD_W_DEFAULT))
    except (TypeError, ValueError):
        v = FOOTHOLD_W_DEFAULT
    return max(FOOTHOLD_W_MIN, min(FOOTHOLD_W_MAX, v))


def save_foothold_width(v):
    """保存编辑器的线条宽度，返回实际生效的值（夹到范围内）。"""
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = FOOTHOLD_W_DEFAULT
    v = max(FOOTHOLD_W_MIN, min(FOOTHOLD_W_MAX, v))
    data = _load()
    data["foothold_line_w"] = v
    _save(data)
    return v


def load_size():
    """读配置里的字号；文件不存在或损坏时退回默认值。"""
    return clamp(_load().get("font_size", DEFAULT_SIZE))


def save_size(v):
    """保存字号，返回实际生效的值（可能被夹到允许范围内）。"""
    v = clamp(v)
    data = _load()
    data["font_size"] = v
    _save(data)
    return v


def _valid_color(v):
    """合法的颜色串：`#RRGGBB`（不透明）或 `#AARRGGBB`（**带透明度**，2026-09-27 起 ✓）。

    ⚠ 为什么必须收 9 位那种：用户要求颜色弹窗能配透明度 ⇒ 存的就是 `#AARRGGBB`
      （Qt 的 `QColor.name(QColor.HexArgb)`）。这里只认 7 位的话，配好的透明度
      **下一次读回来就被丢掉**（会被当成非法值、静默退回默认色 ✗）。
    """
    if not (isinstance(v, str) and v.startswith("#") and len(v) in (7, 9)):
        return False
    return all(c in "0123456789abcdefABCDEF" for c in v[1:])


def load_vis():
    """读可视化配置，返回 dict（颜色 + 线宽 + 每项的显示开关，含默认值兜底）。"""
    vis = _load().get("vis") or {}
    out = dict(VIS_DEFAULTS)
    for k in _VIS_COLOR_KEYS:
        if _valid_color(vis.get(k)):
            out[k] = vis[k]
    # 显示开关：只认真正的布尔；缺省/写坏 ⇒ True（**画** = 老行为 ✓）
    for k in _VIS_ON_KEYS:
        out[k] = bool(vis[k]) if isinstance(vis.get(k), bool) else True
    try:
        out["vision_width"] = max(1, min(10, int(vis.get("vision_width", 1))))
    except (TypeError, ValueError):
        out["vision_width"] = 1
    # ⭐⭐ **玩家坐标箭头的两个数值**（用户 2026-09-28 报："**玩家坐标线段粗细无法调整至
    #   1px**" ✗ —— 根因正是**这里漏了它们**：上面那两道白名单只管**颜色 / 开关**，线宽
    #   只处理了 `vision_width` ⇒ 箭头的粗细 / 长度**永远读不回来**、一直停在默认的 2 / 60
    #   ✗ ⇒ 表现就是"改了没用、也调不到 1"✓）。
    #   ⚠ 教训：**往 `vis:` 加数值键时，必须同时在这儿加一道"读回 + 夹范围"** ——
    #     只加 `VIS_DEFAULTS` 是不够的（那只是"没存过时用谁" ✗ 用户存的值根本进不来 ✗）；
    #     颜色靠 `_VIS_COLOR_KEYS`、开关靠 `_VIS_ON_KEYS`、**数值就得靠这一段** ✓。
    #   ⚠ 坏值（缺键 / 不是数 / 超范围）⇒ 回到默认（规范："坏值当没存过" ✓）。
    # ⭐⭐ **各框线宽**（用户 2026-09-28 ✓"给其他的粗细也加配置"）—— 同一道口径：
    #   **读回 + 夹 `1~20`**，坏值回默认（规范："坏值当没存过" ✓）。
    #   ⚠ 教训（上一轮刚踩过）⇒ **往 `vis:` 加数值键，必须同时在这儿加一道** ✗：
    #     只写 `VIS_DEFAULTS` 是**不够**的（那只是"没存过时用谁"✗ 用户存的值根本进不来 ✗
    #     ⇒ 表现就是"改了没用 / 调不到 1px"✓）。
    for _wk, _wd in (("lock_width", 3), ("attack_width", 2),
                     ("min_attack_width", 2), ("jump_attack_width", 2),
                     ("chase_jump_width", 2)):
        try:
            out[_wk] = max(1, min(20, int(vis.get(_wk, _wd))))
        except (TypeError, ValueError):
            out[_wk] = _wd
    try:
        out["player_arrow_width"] = max(
            1, min(20, int(vis.get("player_arrow_width", 2))))
    except (TypeError, ValueError):
        out["player_arrow_width"] = 2
    try:
        out["player_arrow_len"] = max(
            0, min(2000, int(vis.get("player_arrow_len", 60))))
    except (TypeError, ValueError):
        out["player_arrow_len"] = 60
    # ⭐ **箭头尖大小**（用户 2026-09-28 ✓"再加个箭头 size 配置"）—— 同一道口径：
    #   **读回 + 夹 `5~100`（%）**，坏值回默认（规范："坏值当没存过" ✓）。
    #   ⚠ 又踩一遍那个坑的代价 = "改了没用"✗ ⇒ 加键**必须同时**改这里（只写
    #     `VIS_DEFAULTS` 只是"没存过时用谁"✗）。
    try:
        out["player_arrow_tip_pct"] = max(
            5, min(100, int(vis.get("player_arrow_tip_pct", 25))))
    except (TypeError, ValueError):
        out["player_arrow_tip_pct"] = 25
    # ⭐⭐ **x 箭头 = 玩家真实朝向**（用户 2026-10-10 ✓ 见 `VIS_DEFAULTS` 那段说明 ✓）——
    #   **单独一道** ✗：它**不**进 `_VIS_ON_KEYS`（那个名单是"缺省 ⇒ True"✗ 会把所有人的
    #   箭头默认改成"真实朝向" ✗）；这里**缺省 / 坏值一律 `False`** ✓（= 老观感 ✓
    #   同"坏值当没存过"✓）。
    out["player_arrow_real_face"] = (vis.get("player_arrow_real_face")
                                     if isinstance(vis.get("player_arrow_real_face"), bool)
                                     else False)
    return out


def save_vis(cfg):
    """保存可视化配置。cfg 为 {key: value} 的子集。"""
    data = _load()
    vis = dict((data.get("vis") or {}))
    vis.update(cfg)
    data["vis"] = vis
    _save(data)


def class_colors():
    """{类别 id: (b, g, r)} —— 画框用。**每个类别**的颜色都能在设置里改。

    load_vis() 保证类别框色的键都存在（缺的用默认值补），所以这里直接取。
    """
    vis = load_vis()
    return {cid: hex_to_bgr(vis["%s_color" % en])
            for cid, en, _zh, _bgr in classes.CLASSES}


def class_color(cls):
    """单个类别的框色 (b, g, r)。给命令行工具这类只画一种框的调用方。"""
    return class_colors().get(cls, (255, 255, 255))


def hex_to_bgra(hexstr):
    """'#rrggbb' / '#aarrggbb' → (b, g, r, **a**)，供 cv2 绘制用（用户 2026-09-27 要求）。

    ⚠ alpha 只有 **9 位**那种写法才带；6 位的（老配置、类别框色）一律 `255` = 不透明 ✓。
    ⚠ cv2 的 `line/rectangle` **没有 alpha** ⇒ 半透明的画法见
      `gui/live_thread.py::_blit_alpha`（画到副本上再按 alpha 混回来 ✓）。
    """
    h = hexstr.lstrip("#")
    a = int(h[0:2], 16) if len(h) == 8 else 255
    rgb = h[-6:]                       # 后 6 位永远是 `RRGGBB`（8 位那种前两位是 alpha ✓）
    # ⚠ 返回顺序是 **(b, g, r, a)** —— cv2 要 BGR ✓（写反了整幅画面的颜色都会变 ✗）
    return (int(rgb[4:6], 16), int(rgb[2:4], 16), int(rgb[0:2], 16), a)


def hex_to_bgr(hexstr):
    """'#rrggbb'（或带 alpha 的 `#aarrggbb`）→ (b, g, r)，供 cv2 绘制用。

    只取前三位：(b, g, r) ✓ —— 要透明度请用 `hex_to_bgra`（多了第 4 个值 ✓）。
    """
    b, g, r, _a = hex_to_bgra(hexstr)
    return (b, g, r)


def repolish(w):
    """让 `w` 及其子控件按**当前**全局字号重新解析一遍样式。

    **为什么非要有这一步**（实测踩到的坑，2026-09-25）：光调 `app.setFont()`，
    **打过样式表的控件完全不跟**。
      · 没打过样式表：`setFont` 立刻跟随（实测 11 → 18 ✓）
      · 打过样式表（部署台/工作台的主窗口都打了一整份 QSS）：**一动不动** ✗
        实测：父窗口一份 QSS、**里面一个字号都没写**，子控件在 setFont 之后 11 → 11。
    根因是 `QStyleSheetStyle` 在 polish 时会把自己解析出的字体 `setFont` 到控件上 ——
    等于给控件标了"显式字体"，此后全局字体再变它就不理了。

    修法是**把同一个样式表字符串再设一遍**：`setStyleSheet` 会递归 repolish，
    字体就按新的全局字号重新解析了（实测 11 → 18 ✓）。
    它不像"遍历控件逐个 setFont"那样把**有意设过**的字体一起冲掉
    （日志正文那份等宽字体就是有意的，见 deploy/app.py 的 LogPane.apply_font）。
    """
    from PyQt5.QtWidgets import QWidget

    try:
        if w.styleSheet():
            w.setStyleSheet(w.styleSheet())
        for c in w.findChildren(QWidget):
            if c.styleSheet():
                c.setStyleSheet(c.styleSheet())
    except Exception:                    # noqa: BLE001
        pass                             # 刷新失败不该把设置弹窗弄崩


def apply(app, size=None):
    """把字号应用到整个程序 —— **包括已经存在的控件**。

    app  : QApplication 实例
    size : 字号；None 表示读配置

    两步缺一不可：`setFont` 管"新建的控件"和"没打过样式表的控件"，
    `repolish` 管"已经存在的、打过样式表的控件"（见 repolish 的说明）。
    """
    from PyQt5.QtGui import QFont

    size = clamp(size if size is not None else load_size())
    app.setFont(QFont(FAMILY, size))
    # 样式表也可能挂在 QApplication 上，那 topLevelWidgets 里就没有带 QSS 的了
    try:
        if app.styleSheet():
            app.setStyleSheet(app.styleSheet())
    except Exception:                    # noqa: BLE001
        pass
    for w in app.topLevelWidgets():
        repolish(w)
    return size
