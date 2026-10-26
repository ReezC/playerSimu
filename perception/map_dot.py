"""小地图黄点识别（**单帧**认点；从 `perception/minimap.py` 切出，2026-10-18）。

**这一块只干一件事**：在面板上认「哪个点是玩家」并把结论换算成世界坐标
（`find_player_dot`）—— 跨帧的平滑/定位在 `perception/map_locator.py`。
标定几何在 `perception/map_calib.py`，收流在 `perception/map_client.py`，
叠图核对在 `perception/map_overlay.py`，定位实现留在 `perception/minimap.py`。

⚠ 依赖方向是**单向**的：本模块 import `minimap` 拿 `CropViewTracker` /
`try_autofit` / `_gray`，import `map_overlay` 拿 `diff_panel_vs_canvas`
（它们都不属于"认黄点"），而那两个模块**不** import 本模块 —— 别反过来加，否则立刻成环。
"""

import time

import cv2
import numpy as np

from core import mapdata, zones
from perception import map_calib
from perception import map_const
from perception.map_overlay import diff_panel_vs_canvas
from perception.minimap import CropViewTracker, _gray, try_autofit  # noqa: F401


def dot_feet(r):
    """黄点结论（`find_player_dot` 的返回）→ 面板像素 **(x, 下沿 y)**。

    ⚠ **这是"黄点在面板里算哪个位置"的唯一口径：采样（双点标定）和读数
      （`PlayerLocator.update`）必须都走这里**。用户 2026-09-27 现场验过：
      只统一一边就凭空差半个点 —— 这张图 1 个面板像素 = 7.9 世界像素，6 像素高的点
      差 12 世界像素（实测：站在 L1 上端报 (48,-222)，而真值是 (56,-205)）。

    为什么是**下沿**：游戏把小地图上的玩家标记画成一个小黄点、**下沿对着脚下**，
    所以下沿才是"这个人在地图上的位置"；重心比它高半个点，直接拿重心换算会整体抬高，
    而且和「坐标系偏移」补的那点差撞在一起、谁也说不清哪边错了 ✗。

    ⚠⚠ **不许取整**（2026-09-27 改；用户报的"同一个位置读数差 9"）：
      `find_player_dot` 给的 `x/y` 本来就是**亚像素**浮点（`cv2` 的质心），取整等于把
      读数量化成"整面板像素"一格一格跳 —— 而这张图上
      **1 个面板像素 = 8.55 世界单位**（`px_per_world / scale` = 16.082 / 1.880076）
      ⇒ **形状半个像素的变化**（抗锯齿、压缩噪声、**人在绳上/地上时那个点的画法略有
      差别** —— 眼睛看着就是"没动"）能让读数跳 **9 个世界单位** ✗✗。
      **现场**：站在 L1 上（x=56）顺着绳下来后显示 x=47（正好差一格），而他盯着收流
      小地图确认黄点**没有水平位移** —— 差的正是这一格。
      分辨率的上限消不掉（±0.5 面板像素 ≈ ±4.3 世界单位，那是**测量**本身的分辨率）；
      能消的是这层**白送的量化** ✓。
    ⚠ 脚底锚点本身不变：还是"重心 + (h-1)/2"（h=6 的点占 88..93 行 ⇒ 91+2.5 = 93.5，
      正好压在最后一行上 ✓）。原来这里 `int()` 向下取整的理由是"重心已经被取整过、
      +半个点会落到 x.5 上 ⇒ `round` 会多走一格"，可现在重心**不再被取整**了，
      那条理由也就不成立了 ✓。
    ⚠ 采样那一侧（双点标定）填的是**整数框**，`int(round())` 在那儿照旧 —— 那不是口径
      分叉：口径管的是"取哪个锚点"（下沿），取整只是**界面**的显示精度 ✓。
    """
    h = int(r.get("h") or 1)
    return (float(r.get("x") or 0.0), float(r.get("y") or 0.0) + (h - 1) / 2.0)


# ══════════════════════════════════════════════════════════════
# 玩家标记（「黄点」）识别 —— S3 的门槛
# ══════════════════════════════════════════════════════════════
#
# **为什么这里宁可说「认不出」也不硬猜**：这个位置会被 `segment_of` 拿去判
# 「我在哪块平台上」，猜错比认不出糟得多 —— 认不出只是这一拍没有定位，猜错会
# 让寻路朝反方向走。所以下面的结论有三种，不是两种：
#     认得出 / 认不出 / **这张图的颜色层根本不可用**（要换来源，或改用底图相减）
#
# 标记的画法见 `docs/寻路设计.md` §2.2：
#   · WZ 素材 `Map/MapHelper.img/minimap/user`（外观随客户端，实测是亮黄点）
#   · 无素材兜底 **5×5 青色方块** `Color(0,255,255)`
#   · NPC / portal 点 `Color(132,216,243)` —— **要排除**，它不是玩家
#
# 实测（2026-09-25，寺院通道2 的采集帧）：玩家点**亮黄、6×6、22~25 像素、
# 单连通块**，中心色 **BGR≈(65,243,245)** —— B 分量 65 是压缩/抗锯齿渗出来的，
# 所以阈值**不能按纯黄 (0,255,255) 卡死**。同一天把同一套阈值放到「东部岩山V」
# 的图上，直接命中 2000~7000 像素 / 87~110 块（那张图底图本身就到处黄褐色）
# → 「颜色层够不够用」必须**当场判出来**，这正是下面 `flood` 那一路的由来。

#: 两个「颜色家族」：黄（WZ 素材 `user`，实测玩家点就是它）与青（无素材兜底方块）。
#: **分开判、黄优先**：这两族在同一张图上经常只有一族干净 —— 实测寺院通道2 的
#: **地图本身就是饱和青色**，青族被淹了，而黄族只有玩家那一个点。
#: 判据直接在 RGB 上卡「两个通道很亮、第三个很暗」，不用 HSV：
#:   黄：(R,G ≥ 200) 且 (B ≤ 150) 且 |R−G| ≤ 60
#:   青：(G,B ≥ 200) 且 (R ≤ 120) 且 |G−B| ≤ 60
#: 这样一刀同时挡掉三类捣乱的：NPC/portal 的蓝 RGB(132,216,243)（R=132 > 120）、
#: 底图的黄褐纹理 RGB(218,168,103)（G=168 < 200）、白字白框 (B/R 都很高)。
DOT_YELLOW = ("R>=200", "G>=200", "B<=150", "|R-G|<=60")
DOT_CYAN = ("G>=200", "B>=200", "R<=120", "|G-B|<=60")
#: 家族顺序 = 优先级（玩家标记实测是黄的；青是无素材时的兜底画法）。
DOT_FAMILIES = ("yellow", "cyan")
#: 亮度下限：两族判据里"亮通道 ≥ 200"已经隐含了，这个值只给文档/调参用。
DOT_V_MIN = 170
#: 候选点的边长下限 / 上限。上限按**面板宽度**给 —— 换分辨率/换窗口，面板会大一倍。
DOT_SIDE_MIN = 3
DOT_SIDE_MAX_DIV = 16          # 上限 = max(6, 面板宽 // 这个数)
#: 方形度下限（面积 / 外接矩形）：点近似方块；细长条（描边、地形线）不是点。
DOT_FILL_MIN = 0.45
DOT_ASPECT_MAX = 2.6
#: 面板边缘这些像素不算 —— 面板有边框和地图名，那圈亮色会冒充标记。
DOT_INSET = 3
#: 颜色层「被淹」的两条线（见 _is_flood）：像点的块个数上限，以及
#: 成片像素的占比 / 绝对下限（取大的那个）。**不要用掩码总像素判**：压缩噪声
#: 会撒出成百上千个 1~2 像素的碎点，谁也不像标记，却能顶到几千。
#:
#: 绝对值给得**小**（60）是实测定的：面板大小差 5.6 倍（独立推流 753×612、
#: 从实时画面 134×109），而标记在两种面板里都只占 **0.09%** —— 也就是说
#: 「多大算成片」必须跟着面板走，绝对值只兜住"面板特别小"那种情况。
#: 踩过的坑：绝对值原来写 400，而独立推流那颗 28×24 的点就有 411 像素 ——
#: 差一点就把正常的一帧判成「颜色层不可用」。
DOT_MAX_CANDS = 12
DOT_FLOOD_RATIO = 0.01
DOT_FLOOD_PX = 60
#: 跨帧：相邻两拍位移上限（超过就当噪声，把上一拍位置丢掉重捕），
#: 以及「连续几拍都找到」才算确认（单帧亮斑不算，屏幕上到处是亮斑）。
DOT_MAX_JUMP_PX = 40.0
DOT_CONFIRM_FRAMES = 2

#: 漏检之后**沿用上一帧位置**的时间上限（毫秒）—— 口径同 `perception/track/tracker.py`
#: 的 `debounce_ms`（那边是"漏检期间保留幽灵框"）。为什么不干脆一直沿用：那位置
#: 会越来越像真的（人跑远了它还在原地），而且寻路会拿它当"我在哪块平台上"。
#: ⚠ 2026-10-18 从 **500 → 1500**（用户："**不管怎么样黄点必须是最优先要认出来的**"✓）：
#:   现场日志实测（`behavior.log` 最近 120 拍）：真认出 **43** / 沿用上一帧 **69** / 认不出
#:   **8** —— 也就是**一多半时间**在"沿用"。而 0.5 秒的窗口在 30 拍/秒下只有 15 拍，
#:   小地图上一闪（怪/传送点/鼠标悬停把点盖住几拍）就掉到"认不出 ⇒ EMA 保底" ✗。
#:   1.5 秒 = 45 拍，足够跨过那种短闪 ✓，又短到"人真跑了"不会被一个旧位置骗太久 ✓。
DOT_HOLD_MS = 1500.0
#: 有上一帧位置时，只在它周围这么大一块里搜（面板像素，半边长）。
#: **这么做有两个好处**：① 快 —— 收流那条来源的面板是 753×612，全图掩码 +
#: 连通域每拍要几毫秒，缩到 37×37 基本免费；② 稳 —— 底图上的杂色根本进不来，
#: 「被淹」那个否决条件也就用不上了（见 find_player_dot 的 `near`）。
DOT_ROI_PAD = 18
#: 漏检期间按（衰减的）速度外推：每漏一拍乘一次这个系数，总位移有上限
#: （同 MobTracker._drift：不能冻住，也不能飘到地图另一头）。
DOT_GHOST_DECAY = 0.75
DOT_GHOST_MAX_SHIFT = 8.0        # 幽灵外推的总位移上限（面板像素）

#: ⭐⭐⭐ 「**哪一个才是玩家那个点**」—— 2026-10-10 照用户指的路补的三件
#: （`E:\MyPrograms\Maple_xfeat\src\vision\tracker.py` 的认法 ✓ 用户原话三条：
#:   "**队友橙色点头上的小标记也是黄点，大黄点锁定错了**"✗ /
#:   "你的锁定标记会**乱飘**到别的点上去，根本没坐标偏差有做任何平滑"✗ /
#:   "你**瞄准的不是黄点正中心**"✗）。
#:
#: 病根一句话：**以前挑点是"谁离上一拍最近就锁谁"** ✗（见 `find_player_dot` 里那段
#: `min(cands, key=距离)` ✓）—— 于是队友头上那个**小黄标记**只要离得近，就能把
#: **大黄点**（真玩家标记）顶掉 ✓；两块分数接近时还会**来回翻** ⇒ 读数乱飘 ✓。
#: ⇒ 补三件：
#:   ① **尺寸要像玩家点**（`DOT_PLAYER_SIDE_REL` ✓）：玩家标记是**大**的（实测
#:      26×25 像素 @ 面板宽 753 ≈ **3.2%** ✓ 见 `DOT_CORE_*` 那段量测 ✓）⇒
#:      队友那种小标记、碎块、糊成一片的黄，尺寸分都低 ✓；
#:   ② **滞后**（`DOT_LOCK_MARGIN` ✓）：新候选要在分数上**明显赢**过现在锁着的那个
#:      才换 ✗ —— 从此不再"谁近就锁谁" ✓（那正是乱飘的原因 ✓），但**传送**那种
#:      真跳变照样能靠"分数差得够大"换过去 ✓（同 `DOT_JUMP_GIVEUP_N` 那条纪律 ✓）；
#:   ③ **报出去的位置做一阶低通**（`DOT_SMOOTH_ALPHA` ✓）—— 只抹抖动、不引入明显滞后 ✓。
#: ⚠ 为什么不用 Maple_xfeat 那套 **HSV 窄黄**：他们那张图上"自己"是**唯一黄点**
#:   （队友是**红**点 ✓）⇒ 颜色一分就完事 ✓；我们这张图上**队友头上也是黄** ✗
#:   ⇒ 颜色分不开，**尺寸 + 连续性**才是判据 ✓（而且本仓的颜色判据是**量测**出来的
#:   `BGR(108,255,255)` ✓ 见 `DOT_CORE_*` ✓，动它风险大于收益 ✗ —— 这条记在这儿，
#:   免得下一个人又去改颜色 ✗）。
DOT_PLAYER_SIDE_REL = 0.032       # 玩家点短边 ≈ 面板宽 × 3.2%（实测 ✓ 判据见上）
# ⭐⭐⭐ **2026-10-11 实测推翻了一条假设** ✗（用户现场："**队友橙色点上方有个很小的黄点标记，
#   经常认错**"✗；量测脚本 `tools/mmap_dot_like_probe.py` + `data/dot_probe/*.png` ✓）：
#     面板宽 1164 那块真点 **26×24 = 2.15%** ⇒ like **只有 0.25** ✗（期望边长算成 37px ✗）；
#     而同一帧别的黄块（大概率是 portals ✓）28×21 / 27×21 也是 0.20~0.28 ⇒ **同档** ✗✗；
#     另外两块：536 宽 ⇒ 20×16（3.36% ✓ like 0.61）；78 宽 ⇒ 6×6（7.7% ✓ like 0.80）。
#   ⇒ 结论：**"短边 = 面板宽 × 3.2%"这条跨来源不成立** ✗（点的大小不随面板线性缩放 ✓）
#      ⇒ 拿它当"期望尺寸"时，真点会被判成"偏小" ⇒ 跟杂块同档 ⇒ 胜负全交给距离
#      ⇒ **又回到"谁近谁赢"** ✓（老病灶 ✓）。
#   ⇒ 正确做法：**期望尺寸自标定**（`exp_side` ✓ 见 `find_player_dot` ✓）—— 拿"**最近锁定住的
#      那个点**"的短边中位数当尺子 ✓（锁定的那个默认就是真点 ✓）。上面这个比例**只当首帧兜底** ✓。
#   ⚠ 别再去调那个 0.032 了 ✗（它治不了跨来源 ✓，只会让别的来源变坏 ✓）。
#: 自标定尺子的样本数下限（少于它 ⇒ 退回比例兜底 ✓ 不猜 ✗）。
DOT_EXP_MIN_SAMPLES = 5
#: 自标定窗口（最近多少个**锁定成功**的点 ✓；中位数抗单帧异常 ✓）。
DOT_EXP_HISTORY = 15
#: ⭐ 滞后分**只保护"像个玩家点"的候选**（`DOT_LIKE_MIN` ✓ 2026-10-11 缺口①）：
#: 原来只要"离上一拍近"就 +`DOT_LOCK_MARGIN` ✗ ⇒ 锁在一个**形状完全不像**的东西上（队友那个
#: 小标记 ✓）以后，它**永远拿这份保护** ⇒ 真点出现也难挤下去 ✓。
#: 取值照量测：真点最低 **0.20**（1164 那块 ✓）⇒ 取 **0.15**（留一点余量 ✓ 又能挡住"几个像素的
#: 杂点 / 细长条"那一档 ✓ —— 它们在 `dot_player_like` 里本来就只有 0 或 ~0.01 ✓）。
DOT_LIKE_MIN = 0.15
#: ⭐ **"可达半径"**（`DOT_REACH_REL` ✓ 2026-10-11 缺口③）：候选离上一拍超过 `它 × 期望边长`
#: ⇒ **一律垫底**（不再是线性扣分 ✗）✓。
#: 为什么必须改：旧口径线性罚最高 **0.7** ✗ 而实测 `like` 之间的差距常在 **0.05~0.2** ✓
#: ⇒ 实际上是"**谁近谁赢**" ✓（那正是乱飘 / 锁错的来源 ✓）。
#: 改后：**半径内按 `like` 排**（距离只占很小权重 `DOT_DIST_W` ✓）⇒ "像不像"才是主判据 ✓。
DOT_REACH_REL = 3.0
#: 半径内的距离权重（很小 ✓ 只用来打破"两个都像"的平局 ✓）。
DOT_DIST_W = 0.10
#: ⚠ **硬尺寸闸**（相对面板宽 ✓ 2026-10-11 ✓）：短边小于"面板宽 × 它"的块**不算候选** ✓。
#: 取值照量测（14 个真样本复核过 ✓）：最"小"的真点是 78 宽面板上的 **6×6 = 7.7%** ✓、
#: 960 宽那块 **16×16 = 1.67%** ✓ ⇒ 门槛取 **0.8%**（真点的 1/2 以下 ✓）**再夹到 ≥3px** ✓
#: ⇒ 1164 那块的门槛是 9.3px ✓ ⇒ 队友那个几像素的小标记**直接出局** ✓ 而 16/20/24/26 全过 ✓。
#: ⚠ 零回归兜底：**筛空 ⇒ 原样返回**（绝不把本来找得到的点弄丢 ✗ 见 `_dot_candidates` ✓）。
DOT_PLAYER_SIDE_HARD_REL = 0.008

#: ⭐⭐⭐ **绝对下限**（短边像素 ✓ 2026-10-11 第二轮 ✓ —— 用户原话："**还在锁小黄点
#: 拿大小判定一下不行吗？**"✓）。
#: ⚠⚠ 为什么非有它不可（**现场量出来的** ✓ 不是猜 ✗）：那道相对闸有一条"**筛空就原样返回**"
#:   的零回归兜底 ✓，而它**把"连像都不像"的碎块也一起放行了** ✗✗ —— 拿用户那块真面板
#:   （`data/dot_probe/now_panel.png` ✓ 1164 宽 ✓）实测：
#:     · 黄族**被淹**（命中 25774 px ⇒ 颜色层判死 ✓ 见 `_is_flood` ✓）；
#:     · 于是退到**青族** ✓，而青族那一片里**只有一个**候选：**(628,350) 的 3×5**（= 0.34% ✗）
#:       —— 均值 4px < 门槛 9.3px ⇒ 该被挡 ✓ 可它**就是唯一那个** ⇒ 兜底原样放行 ✗
#:       ⇒ `find_player_dot` 报 `ok=True` 锁住它 ✓（真点在 **(1135,480)** ✓ 离 523px ✗）✓
#:       ⇒ 这就是"**还在锁小黄点**" ✓（换任何尺子都一样 ✓ 实测 ✓ —— 因为尺子压根没参与 ✓）。
#:   ⇒ 修法：把"**绝对**不可能"和"**相对**可疑"拆开 ✓：
#:     · **绝对**（短边 < 它）⇒ **没有兜底** ✗ —— 6 个真来源量下来，真点**核心块**最小是
#:       536 宽面板上的 **8×12**（短边 8 ✓）、黄色层最小 **16×16** ✓ ⇒ 取 **4px**
#:       （= 真点最小短边的**一半** ✓ 与相对闸"取真点比例的一半"**同一口径** ✓）✓
#:       ⇒ 这时**如实报"认不出"** ✓ 比报一个假坐标强得多 ✓
#:       （模块自己的信条 ✓：假坐标会让 Agent 一路怼墙 ✓ 见 `force_reacquire` 那段现场 ✓）。
#:     · **相对**（短边 < 面板宽 × `DOT_PLAYER_SIDE_HARD_REL`）⇒ **保留兜底** ✓
#:       （小面板 / 换分辨率的容错 ✓ 那条兜底正是为 78 宽那块 **6×6** 留的 ✓）。
#:   ⚠⚠ **为什么是 4 而不是 6**（我第一版写 6，当场被 5 条既有用例按住 ✓ 如实记）：
#:     "无素材时 5×5 的青色方块也算玩家"是**故意**钉着的（真·青色标记、测试替身都靠它 ✓）
#:     ⇒ 绝对下限只该拦"**连替身都不如**"的那一档 ✓ —— 用户现场那块是 **3×5**（短边 3 ✓）
#:     ⇒ 4px 照样拦得住 ✓✓（两边都不伤 ✓）。
#: ⚠ 记着：这条**只看尺寸**（"拿大小判定" ✓），它治不了"**同样大小**的地图图标"✗ ——
#:   那一档要靠核心色（`DOT_CORE_*` ✓ 已经在用 ✓ 见 `_local_result_trustworthy` ✓）
#:   与"底图相减"（那条治的就是"地图上的假黄色图标不许赢过真点" ✓ 见它用例 ✓）。
DOT_SIDE_HARD_ABS = 4.0
#: ⛔⛔ **上面那个 4px 是死值 ⇒ 会卡死小面板** ✗（2026-10-13 现场复盘 ✓ 用户报：
#:   "**石人寺院III 项目，提示「世界坐标：实时在跑，但这一拍还没定上位」就会发呆**"✗
#:   "**严重问题**"✓）。证据链（都可复现 ✓）：
#:     · 那张图 `106010106` 的标定里记着 **live 源的面板 = 129×71** ✓（`mmap_crop` 也是
#:       [5,71,129,71] ✓），画布 **127×61** ✓ —— **小面板** ✓；
#:     · 这游戏的小地图标记是**固定像素大小**的 ✓（本文件上面量过并写着：
#:       "**点的大小不随面板线性缩放**"✗）⇒ 129 宽那块面板上真点只有 **3~5px** ✓；
#:     · 死值 4px（**短边、且没有兜底** ✗）= 我 2026-10-11 定的 ⇒ **正好把真点挡死** ✗
#:       ⇒ `find_player_dot` 报认不出 ⇒ 没有世界坐标 ⇒ `LAST_POS.world_x=None` ⇒
#:       界面那行显示"这一拍还没定上位" ⇒ Agent「未定位玩家」⇒ **发呆** ✓✓（现象逐条吻合 ✓）。
#:   ⇒ 现在改成**跟着面板尺寸走**（一行算式 ✓ 三个数都有量测支撑 ✓）：
#:       `下限 = max(2px, min(6px, 0.4% × 面板宽))`
#:     · **1164 宽** ⇒ 4.66px ✓（2026-10-11 那块 3×5 的碎块**照样挡得住** ✓ 那天的修法不倒退 ✓）；
#:     · **129 宽** ⇒ 2px ✓（真点 3px **过得去** ✓✓ 本轮的 bug 修掉 ✓）；
#:     · **4656 宽**（推流原始像素）⇒ 6px（夹住 ✓ 真点 84px 离得远 ✓ 不误伤 ✓）。
#:   ⚠ 2px 那一头是"**连 1~2px 的噪声都不认**"的意思 ✓（真点最小 3px ✓ 有量测 ✓）。
DOT_SIDE_HARD_ABS_MIN = 2.0
DOT_SIDE_HARD_ABS_REL = 0.004
DOT_SIDE_HARD_ABS_MAX = 6.0
#: 换锁要赢这么多分（0~1 尺度 ✓）：太小 = 来回跳 ✗，太大 = 传送时换不过去 ✗。
DOT_LOCK_MARGIN = 0.12
#: 报出去的位置做一阶低通：**1 = 不过滤** ✓；取 0.6 = 抹抖但不拖泥带水 ✓。
DOT_SMOOTH_ALPHA = 0.6

#: ⭐⭐⭐ **「定位保底」**（2026-10-13 ✓ 用户原话："**做保底：定位后，不准找不到，
#:   丢了就用上次EMA平滑的**"✓；同一次现场："**石人寺院III 项目，提示「世界坐标：实时在跑，
#:   但这一拍还没定上位」就会发呆**"✗ "**严重问题**"✓）。
#: 口径（一句）：**只要这张图上定过一次位** ⇒ 之后认不出黄点时**不再报"找不到"** ✓，
#:   输出"**最后一次定位的 EMA 平滑值**" ✓，并如实标上 `held=True`（沿用的 ✓）、
#:   `stale=True`（是保底值 ✓）、`stale_s`（多少秒前那次真定位 ✓）—— 标志齐全 ⇒
#:   "要新鲜观测"的消费方自己看标志 ✓，不要的照用 ✓（**绝不假装它是这一拍量出来的** ✗）。
#: 为什么必须做（现场链条）：小面板（129×71 ✓）上黄点本来就难认 ✓ ⇒ 认不出 ⇒ `world_x=None`
#:   ⇒ 界面那行写"这一拍还没定上位" ⇒ Agent `_player_located` 判"未定位玩家" ⇒ **发呆** ✓
#:   （整条自动等于停了 ✗ —— 用户要的就是"别停" ✓）。
#: ⚠ **前提是"定位后"** ✓：从没定过位（刚开 / 没标定 / 刚换图 ✓）⇒ 照旧如实说找不到 ✓（不猜 ✗）。
#: ⚠ 换图（`PlayerLocator.load`）**清掉** ✓：上一张图的位置在另一张图上毫无意义 ✗。
PLAYER_FB_EMA_ALPHA = 0.4

#: 上面这四个可以**从 config/live.yaml 覆盖**（界面在「路线识别 → 小地图定位」里调）。
#: 键名 → 默认值：改键名要一起改 `track_params()` 和界面那排输入框。
TRACK_KEYS = (("mmap_hold_ms", DOT_HOLD_MS),
              ("mmap_roi_pad", DOT_ROI_PAD),
              ("mmap_max_jump", DOT_MAX_JUMP_PX),
              ("mmap_ghost_shift", DOT_GHOST_MAX_SHIFT))


def track_params(live_cfg=None):
    """从 config/live.yaml 的 dict 里取出四个跟踪参数（缺的用默认值）。

    **只有这一处知道键名与默认值**：界面、实时线程、命令行工具都调它，
    免得三处各写一份、改了一处另外两处还是老值。
    """
    cfg = live_cfg or {}
    out = {}
    for key, dflt in TRACK_KEYS:
        raw = cfg.get(key)
        try:
            out[key] = float(dflt) if raw is None else float(raw)
        except (TypeError, ValueError):
            out[key] = float(dflt)          # 配置写坏了 → 退回默认，不炸
    return out


def tracker_kwargs(track):
    """`track_params()` 的 dict → `PlayerDotTracker(**…)` 的关键字。"""
    return {"hold_ms": track.get("mmap_hold_ms"),
            "roi_pad": track.get("mmap_roi_pad"),
            "max_jump": track.get("mmap_max_jump"),
            "ghost_max_shift": track.get("mmap_ghost_shift")}


#: ⭐⭐ **玩家点的"核心色"：量测值 + 一道宽松邻域**（用户 2026-10-06 ✓ 现场用仓里现成的
#: `tools/mmap_dot_probe.py` 连推流量到的：**BGR = (108, 255, 255)** ✓ 那块点中位 26×25、362 像素 ✓）。
#: **用途只有一个** ✓：**底图相减层**（`basemap` ✓）里再拒掉"不像玩家点"的候选 ——
#:   那一层只看"比底图多出来的东西" ⇒ 底图杂色多时会**锁假点** ✗（现场：那块地图底图大片黄褐 ⇒
#:   颜色层被判「被淹」⇒ 退到这一层 ⇒ 锁住一个不动的假点 ⇒ Agent 照它往右顶墙、人在底层最右角落
#:   卡死 ✓）。
#: ⚠ 邻域是**量出来的**（第一版我按"宽松"配 180 —— 实测**一点都拒不掉** ✗：真底图上黄族命中
#:   100% 都落在 180 那条线内 ⇒ 闸等于没装 ✗）。**真实面板**（`data/dot_probe/stream_panel.png` ✓）
#:   两边的数一摆就清楚了：
#:     · 玩家点（探针直接量到的）      **BGR = (108, 255, 255)** ✓
#:     · 淹掉的那一大片（黄族命中 3.78% ✓ 均值）**BGR ≈ (116, 211, 244)** ✓（块 1056×284 ✗）
#:     · 黄族像素的通道分布：B 中位 117 · **G 中位 211** · **R 中位 244** ✓
#:   ⇒ 分界线就是"**G、R 要接近满格**" ✓：`G,R ≥ 240` 一口气把 3312 px（0.8% ✓）挑出来、
#:     把 41 万 px 的洪水滤掉 ✓✓（**B ≤ 160** 同时挡掉偏白的黄 ✓）。
#: ⚠ 调用处**只在"能保住至少一个候选"时才采用** ✓（一个都没保住 ⇒ 退回原来的候选 ✓）
#:   ⇒ **零回归**：这条闸最坏只是"没用上"，绝不会把本来能找到的点弄丢 ✓。
DOT_CORE_G_MIN = 240
DOT_CORE_R_MIN = 240
DOT_CORE_B_MAX = 160
#: ⭐ **核心色那一路的最小边长**（用户 2026-10-06 ✓）：**量出来的** ✓ —— 探针在真实推流上
#: 量到的玩家标记是 **26×25 / 362 像素**（`DOT_CORE_*` 说明里那段）、现场另一帧 **24×25 / 323**、
#: `_prefer_on_foothold` 的现场记录里是 **28×24 / 411** ✓ ⇒ 真标记的短边 ≥ 21 ✓。
#: 取 **10**（真标记的一半不到 ✓）当门：只为挡"**几个像素的杂点**" ✗（那种东西在核心色那条
#: 紧阈值下偶尔会冒出来 ✓），**不是**在限制真标记 ✓（21~25 轻松过 ✓）。
DOT_CORE_MIN_SIDE = 10


def dot_core_mask(img):
    """「**像玩家点核心色**」的像素掩码（uint8 0/1 ✓）—— 就是上面那个**量测邻域** ✓。

    ⚠ 别拿它当主判据 ✗：颜色层那一路照旧用 `dot_color` 的族判据 ✓（不动它 ✓）；这一条只服务
      **底图相减层**的"拒假点" ✓（见 `_blob_looks_core` / `find_player_dot` ✓）。
    """
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    return ((g >= DOT_CORE_G_MIN) & (r >= DOT_CORE_R_MIN)
            & (b <= DOT_CORE_B_MAX)).astype(np.uint8)


def _blob_looks_core(panel, m, c):
    """这个候选块**像玩家点核心色**吗 —— 拿它**在掩码里的那些像素**的均值判 ✓（同量测口径 ✓）。"""
    try:
        x0 = max(0, int(round(c["x"] - c["w"] / 2.0)) - 1)
        y0 = max(0, int(round(c["y"] - c["h"] / 2.0)) - 1)
        x1 = x0 + int(c["w"]) + 3
        y1 = y0 + int(c["h"]) + 3
        sub_m = np.asarray(m[y0:y1, x0:x1])
        sub = np.asarray(panel[y0:y1, x0:x1])
        if sub_m.size == 0 or not sub_m.any():
            return False
        ys, xs = np.nonzero(sub_m)
        mean = sub[ys, xs].mean(axis=0)
    except Exception:                       # noqa: BLE001 —— "像不像"而已，坏形状当"不像" ✓
        return False
    return bool(mean[1] >= DOT_CORE_G_MIN and mean[2] >= DOT_CORE_R_MIN
                and mean[0] <= DOT_CORE_B_MAX)


def dot_color(panel, family=None):
    """族判据本身（RGB 上直接卡），**不含面板边缘的 inset** —— 也用来量颜色。

    `family` 取 "yellow" / "cyan" 只取一族；None = 两族并集。
    """
    b = panel[:, :, 0].astype(np.int16)
    g = panel[:, :, 1].astype(np.int16)
    r = panel[:, :, 2].astype(np.int16)
    yellow = (r >= 200) & (g >= 200) & (b <= 150) & (np.abs(r - g) <= 60)
    cyan = (g >= 200) & (b >= 200) & (r <= 120) & (np.abs(g - b) <= 60)
    if family == "yellow":
        return yellow
    if family == "cyan":
        return cyan
    return yellow | cyan


def dot_mask(panel, family=None):
    """「玩家标记色」掩码（uint8 0/1），已挖掉面板边缘。

    `family` 取 "yellow" / "cyan" 只取一族；None = 两族并集（**只用于看证据图**，
    判断该用哪族请走 find_player_dot —— 两族分开判才不会互相拖累）。
    """
    m = dot_color(panel, family).astype(np.uint8)
    ins = DOT_INSET
    if m.shape[0] > 2 * ins and m.shape[1] > 2 * ins:
        m[:ins, :] = 0
        m[-ins:, :] = 0
        m[:, :ins] = 0
        m[:, -ins:] = 0
    return m


#: 小面板上「像点」的边长上限**保底**（2026-10-18 现场 ✓）：
#:
#: 天空项目走**远程收流**，面板只有 **97×122**（部署台裁得小）⇒ 真玩家点就是 **6×7**，
#: 而 `max(6, 97//16)` 正好 = **6** ⇒ 高 7 的那个点被这道闸**当成"太大"扔掉** ✗
#: ⇒ 整帧 `候选 0 个` ⇒ 报"认不出玩家标记"（用户现场 ✓）。
#: 保底抬到 8：小面板只放宽 2px（大面板照旧按面板宽走 ✓ —— 1164 宽那块是 72 ✓ 一点没动）。
DOT_SIDE_MAX_FLOOR = 8


def dot_side_max(panel_w):
    """「像点」的边长上限：按**面板宽度**给（换分辨率/换窗口，面板会大一倍）。"""
    return max(int(DOT_SIDE_MAX_FLOOR), int(panel_w) // DOT_SIDE_MAX_DIV)


def dot_fill_min(w, h):
    """「够不够实」的下限：**小点放宽**（2026-10-18 现场 ✓）。

    6~9 px 的点，抗锯齿 + 和小地图底色混色会吃掉约 4 成面积 ⇒ 实测真点
    `area 17 / (6×7) = 0.40`，被死值 `DOT_FILL_MIN = 0.45` 挡死 ✗（天空项目现场）。
    大点没有这个问题（边角占比小）⇒ **只对小点放宽**，免得把碎块/描边放进来 ✗。
    """
    try:
        m = max(int(w), int(h))
    except (TypeError, ValueError):
        return float(DOT_FILL_MIN)
    return float(DOT_FILL_MIN) * 0.72 if m <= 9 else float(DOT_FILL_MIN)


def dot_player_like(c, panel_w, exp_side=None):
    """这个候选块**像不像「玩家那个点」** ⇒ 0~1 分（越大越像 ✓）；坏形状 ⇒ 0 ✓。

    ⭐⭐ `exp_side` = **自标定的"期望边长"**（2026-10-11 ✓ 现场量出来的 ✓ 见
      `DOT_PLAYER_SIDE_REL` 上面那张表 ✓）：拿"最近锁定住的那些点"的短边中位数当尺子 ✓
      ⇒ **跨来源也成立** ✓（比例兜底只在首帧用 ✗）。
      ⚠ 为什么这条非加不可：1164 宽那片面板上，真点 26×24 ⇒ 比例口径把期望算成 37px ✗
      ⇒ like 只剩 **0.25** ⇒ 跟同帧别的黄块（portals ✓ 0.20~0.28）**同档** ✗
      ⇒ 挑点退化成"谁近谁赢" ✓（用户报的"经常认错"就是它 ✓）。
      自标定之后：真点 ≈ 1.0 ✓、队友那个几像素的小标记 ≈ 0 ✓ ⇒ 分得干干净净 ✓。

    ⭐ 2026-10-10 加（用户三条抱怨见 `DOT_PLAYER_SIDE_REL` 上面那段 ✓）。三个因子相乘，
    任一项离谱就把总分拉低 ✓：

      · **尺寸**：期望短边 `exp = DOT_PLAYER_SIDE_REL × 面板宽` ✓（= 玩家标记的实测比例 ✓）。
        用**对数距离**算分 ⇒ "大两倍"和"小一半"等价地扣分 ✓ —— 关键是**偏心小那一侧**：
        队友头上的小标记远小于 `exp` ⇒ 分数直接掉下去 ✓；糊成一大片黄也掉 ✓；
      · **密度**（面积 / 外接框）：越实越像 ✓；低于候选门槛 `DOT_FILL_MIN` ⇒ 直接 0 ✓
        （碎块 / 细线不算点 ✓）;
      · **方圆**（短边 / 长边）：≈1 才像方块点 ✓（描边、地形线是细长的 ✓）。

    ⚠ 口径出处：`E:\\MyPrograms\\Maple_xfeat\\src\\vision\\tracker.py` 那边是
      「尺寸白名单 + 面积 5~60 + 密度 ≥ 0.28，按**最像那个 6×6 黄点**排」✓ ——
      它那里的"6×6"是**它那张图的尺度** ✗，我们这张图的面板尺寸差 5.6 倍（753 宽 ↔ 134 宽 ✓
      见 `DOT_FLOOD_PX` 那段 ✓）⇒ 这里换算成**相对面板宽的比例** ✓（换分辨率不用重调 ✓）。
    """
    try:
        w = float(c.get("w") or 0.0)
        h = float(c.get("h") or 0.0)
        a = float(c.get("area") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if w <= 0.0 or h <= 0.0:
        return 0.0
    # ⭐ 期望边长：**自标定优先**（`exp_side` ✓ 见函数头那段 ✓），没有才退回比例兜底 ✓
    try:
        _es = float(exp_side) if exp_side is not None else 0.0
    except (TypeError, ValueError):
        _es = 0.0
    if _es >= 2.0:
        exp = _es
    else:
        exp = max(4.0, float(DOT_PLAYER_SIDE_REL) * max(1.0, float(panel_w)))
    side = 0.5 * (w + h)
    d = abs(float(np.log(max(1e-6, side / exp))))
    # ⚠⚠ **两侧不对称**（2026-10-10 ✓ 现场定的）：真玩家点比预期**大**是常事
    #   （抗锯齿、描边、和底图糊在一起 ✓）⇒ 宽容一点（σ=0.60 ✓）；
    #   而比预期**明显小**的，几乎都是**队友头上那种小标记** / 碎块 ✗ ⇒ 罚得重（σ=0.32 ✓）。
    #   ⚠ 这条不对称正是"大黄点 vs 小标记"能分开的关键 ✓ —— 对称 σ 时两者只差 0.1 分，
    #     `DOT_LOCK_MARGIN` 的滞后分足以让**锁错的那个小点一直赖着** ✗（实测过 ✓）。
    _sd = 0.32 if side < exp else 0.60
    size = float(np.exp(-(d * d) / (2.0 * _sd * _sd)))
    if max(w, h) > DOT_ASPECT_MAX * min(w, h):
        return 0.0                      # 细长条（描边 / 地形线 / 好几块糊一起）**不是点** ✓
    fill = min(1.0, max(0.0, a / (w * h)))
    if fill < dot_fill_min(w, h):            # 小点放宽（与 _dot_candidates 同一把尺 ✓）
        return 0.0
    sq = min(w, h) / max(w, h)
    return float(size * min(1.0, fill / 0.85) * sq)


def _dot_candidates(m, panel_w, exp_side=None):
    """掩码 → (像「点」的连通块列表, 成片像素数)。

    每项：{x, y（重心）, w, h, area, fill}。**不做「选一个」** —— 多候选时该选谁
    要跨帧才看得出来（见 PlayerDotTracker），单帧选一个等于瞎猜。

    第二个返回值 `dense` 是「边长达到最小点尺寸的块」的像素总和 —— **判断"颜色层
    有没有被淹"要用它，不能用掩码总像素**：压缩噪声会撒出成百上千个 1~2 像素的
    碎点，它们谁也不像标记，却能把总像素顶到几千。
    """
    side_max = dot_side_max(panel_w)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    dense = 0
    for i in range(1, n):
        x, y, w, h, a = (int(t) for t in stats[i])
        if w < DOT_SIDE_MIN or h < DOT_SIDE_MIN:
            continue                       # 碎点：不算候选，也不算"成片"
        dense += a
        if w > side_max or h > side_max:
            continue
        if a < DOT_SIDE_MIN * DOT_SIDE_MIN - 2:      # 3×3 至少 7 个像素
            continue
        fill = a / float(max(1, w * h))
        if fill < dot_fill_min(w, h):        # 小点放宽（见 dot_fill_min ✓）
            continue
        if max(w, h) > DOT_ASPECT_MAX * min(w, h):
            continue
        out.append({"x": float(cent[i][0]), "y": float(cent[i][1]),
                    "w": w, "h": h, "area": a, "fill": round(fill, 2)})
    # ⭐⭐ **排序口径 2026-10-10 改**（原来是"越大越方" ✓）：先按「**像不像玩家那个点**」
    #   （`dot_player_like` ✓ 尺寸比例 + 密度 + 方圆 ✓），再按大小/实心度兜底 ✓
    #   ⇒ "兜底挑第一个"时挑到的是**最像玩家标记的那个** ✓，而不是"最大的那一坨" ✗
    #   （用户："队友橙色点头上的小标记也是黄点，**大黄点锁定错了**"✗）。
    #   ⚠ **算分不写进候选字典** ✗（`all` 是要给界面/诊断看的 ✓ 加字段会牵动一堆比较 ✓）：
    #     排序时现算（候选通常个位数 ✓ 便宜 ✓）。
    _pw = float(panel_w)
    out.sort(key=lambda c: (-dot_player_like(c, _pw, exp_side), -c["area"], -c["fill"]))
    # ⭐⭐ **硬尺寸闸**（2026-10-11 ✓ 见 `DOT_PLAYER_SIDE_HARD_REL` 那段量测 ✓）：
    #   短边小于"面板宽 × 0.8%（并夹到 ≥3px）"的块 ⇒ **不是候选** ✗ —— 队友头上那个
    #   几像素的小黄标记**就死在这一条** ✓（它以前只是"分低" ✓ 还一直排在候选表里等着被锁 ✓）。
    #   ⚠⚠ **零回归**（老规矩 ✓）：这一道要是把候选**筛空**了 ⇒ **整段作废、按老的来** ✓
    #     —— 绝不允许它把"本来认得出"的那一拍弄丢 ✗（宁可多留几个候选 ✓ 不许一个都不剩 ✓）。
    # ⭐⭐⭐ 2026-10-11 第二轮改：**两道分开**（用户："**还在锁小黄点 拿大小判定一下不行吗？**"✓）
    #   ① **绝对下限** `DOT_SIDE_HARD_ABS`（短边 6px ✓）⇒ **没有兜底** ✗ ——
    #      几像素的碎块**绝不可能是玩家点** ✓（6 个真来源量下来真点最小短边 8px ✓ 见那里 ✓）
    #      ⇒ 这时**如实报"认不出"** ✓ 比锁一个假点强得多 ✓；
    #   ② **相对下限**（面板宽 × `DOT_PLAYER_SIDE_HARD_REL` ✓）⇒ **保留"筛空就退回"的兜底** ✓
    #      —— 老兜底的本意是"别因为换分辨率把点弄丢"✓，那说的是**这一层** ✓ 不是"什么都放行" ✗。
    #   ⚠ 口径同时**改对**：一律用**短边**（`min(w,h)` ✓ —— 与常量的说明一致 ✓）：
    #     老写法用的是 `0.5*(w+h)`（**长短边平均** ✗）⇒ 一条 63×27 的长条能靠平均分混过去 ✗。
    #   ⚠⚠ 现场铁证（`data/dot_probe/now_panel.png` ✓ 1164 宽 ✓）：青族里只有一个 **3×5**
    #     的碎块（真点在 (1135,480) ✓）⇒ 老写法：均值 4 < 9.3 ⇒ 被挡 ⇒ **但它是唯一那个** ⇒
    #     兜底原样放行 ✗ ⇒ `ok=True` 锁住它（离真点 523px ✗）⇒ 就是用户看到的"**还在锁**" ✓。
    try:
        # ⚠ **跟着面板尺寸走**（2026-10-13 ✓ 见上面 `DOT_SIDE_HARD_ABS_MIN` 那段现场 ✓）：
        #   死值会卡死小面板（129 宽那块真点只有 3px ✓ 被 4px 挡死 ⇒ 一直"没定上位"⇒ 发呆 ✗）。
        _abs = max(float(DOT_SIDE_HARD_ABS_MIN),
                   min(float(DOT_SIDE_HARD_ABS_MAX),
                       float(DOT_SIDE_HARD_ABS_REL) * _pw))
    except (TypeError, ValueError):
        _abs = float(DOT_SIDE_HARD_ABS_MIN)
    try:
        _rel = max(_abs, float(DOT_PLAYER_SIDE_HARD_REL) * _pw)
    except (TypeError, ValueError):
        _rel = _abs
    _ok = [c for c in out if min(float(c["w"]), float(c["h"])) >= _abs]
    _big = [c for c in _ok if min(float(c["w"]), float(c["h"])) >= _rel]
    return (_big or _ok), dense


def _dot_candidates_edge(panel, fam=None, ref_w=None, exp_side=None):
    """黄点**贴着小地图边缘**时的那一档（2026-10-23 ✓ 用户："**怎么没认出来**"）。

    **现场**（用户截图 + `data/dot_probe/app_panel.png` ✓ 面板 **109×114**）：玩家走到地图
    最右边，游戏就把黄点**画在面板最外那 2~3 px 里** —— 而 `dot_mask` 会把边缘
    `DOT_INSET(3)` 圈**整圈挖掉**（那是为了不让面板描边/边框被当成点 ✓）
    ⇒ 掩码 **0 像素** ⇒ 颜色层没候选 ⇒ 退到差分层 ⇒ 出口"颜色不像" ⇒ **认不出** ✗
    ⇒ 界面上的圈退回旧位置（用户看到的：**圈不在黄点上** ✓）。

    **实测**（同一个黄点，只挪 x，面板 109 宽）：中心 103 ⇒ 掩码 36 px ✓；105 ⇒ 24 ✓；
      106 ⇒ 18 ✓；**107 / 108 ⇒ 0** ✗ —— 边界正好落在"最外 3 px"那一条上 ✓。

    ⇒ 这一档：**不挖边**再找一遍，只收**贴着最外 2 px** 的块，并且
      · 边长照 `dot_side_max`（面板描边是**整条长线** ⇒ 过不去 ✓）；
      · 短边 ≥ `DOT_SIDE_MIN`、面积 ≥ 8、实心度 ≥ 0.35
        （贴边必然被裁掉一角 ⇒ 口径比常规松一档 ✓ 不然刚被裁就又不认了 ✗）。
    ⚠ **只在常规两档（底图相减 / 颜色）都没结果时才用**（见 `find_player_dot` ✓）；
      认到的点会带 `layer="edge"` + 一句"贴边（被裁了一点，位置差 ≤ 2 px）" ✓。
    返回 `(候选列表, 掩码像素数)` ✓（空 = 这一档也没用 ✓）。
    """
    if panel is None or getattr(panel, "ndim", 0) != 3:
        return [], 0
    try:
        H, W = int(panel.shape[0]), int(panel.shape[1])
        m = dot_color(panel, fam).astype(np.uint8)
    except Exception:                          # noqa: BLE001 —— 认不出就算了，绝不抛 ✓
        return [], 0
    total = int(m.sum())
    if total <= 0:
        return [], 0
    side_max = dot_side_max(ref_w or W)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(t) for t in stats[i])
        if max(w, h) > side_max:
            continue                           # 整条描边 / 一大片 ⇒ 不是点 ✓
        if min(w, h) < DOT_SIDE_MIN:
            continue
        if a < 8:
            continue
        fill = a / float(max(1, w * h))
        if fill < 0.35:
            continue
        if not (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1):
            continue                           # 只收**贴边**的（里面那些常规档已经找过了 ✓）
        out.append({"x": float(cent[i][0]), "y": float(cent[i][1]),
                    "w": w, "h": h, "area": a, "fill": round(fill, 2)})
    out.sort(key=lambda c: (-c["area"], -c["fill"]))
    return out, total


def roi_around(center, shape, pad):
    """以 center 为中心、半边长 pad 的矩形（裁到画面内）→ (x0,y0,x1,y1) 或 None。

    画面太小（裁出来没意义）返回 None，让调用方退回全画面搜。
    """
    if center is None:
        return None
    h, w = shape[:2]
    cx, cy = float(center[0]), float(center[1])
    x0, x1 = int(cx - pad), int(cx + pad) + 1
    y0, y1 = int(cy - pad), int(cy + pad) + 1
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 2 * DOT_SIDE_MIN or y1 - y0 < 2 * DOT_SIDE_MIN:
        return None
    return (x0, y0, x1, y1)


def _is_flood(n_cands, dense, area_search):
    """这个搜索区里的颜色层还能不能信 → True = 不可用。

    两条独立的信号，命中任一条就算淹：
      · 像点的块**个数**太多（一片碎块里总有一个"最像的"，但选它是瞎猜）；
      · 成片像素占比太高（底图自己就一大片标记色，比如东部岩山V）。
    """
    if n_cands > DOT_MAX_CANDS:
        return True
    return dense > max(DOT_FLOOD_PX, DOT_FLOOD_RATIO * max(1, area_search))


def _calib_roi(panel_shape, calib, terrain):
    """「底图覆盖到的那块面板区域」(x0,y0,x1,y1) —— 把面板标题/边框排除在外。

    **为什么要它**（实测踩到）：面板上方有一条「地图名 + 图标」的标题带，颜色是
    暖色 —— 颜色层把它整块命中，一块 214×272 的面板光那条就吃掉 4000+ 像素，
    于是每帧都被判成「颜色层不可用」。底图只覆盖地图区，所以**拿底图在面板上的
    范围当搜索区**，标题自然落在外面。

    只在 fit 方式下算得出来：crop 方式下面板显示的整块就是底图的一部分（标题在
    面板自己的边框里，靠 DOT_INSET 那一圈挡掉）。
    """
    if not map_calib.has_geometry(calib) or calib.get("mode") != map_const.MODE_FIT:
        return None
    cv_ = getattr(terrain, "canvas", None)
    if cv_ is None:
        return None
    sx, sy = map_calib.scales_of(calib)                  # 两轴各自（老文件 ⇒ 两个一样 ✓）
    ox, oy = calib.get("offset") or (0, 0)
    ch, cw = cv_.shape[:2]
    # ⚠ 参数是 numpy 的 `.shape`（**高在前**）：按 (宽, 高) 解会把宽高弄反，
    # 搜索结果区被裁成正方形的一角 —— 实测表现是「底图相减明明减掉了，却找不到点」。
    h, w = panel_shape[:2]
    x0, y0 = max(0, int(ox)), max(0, int(oy))
    x1, y1 = min(w, int(ox + cw * sx)), min(h, int(oy + ch * sy))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def _apply_roi(m, roi):
    """把掩码裁到 roi 里（其余置 0）。roi 为 None 就原样返回。"""
    if not roi:
        return m
    out = np.zeros_like(m)
    x0, y0, x1, y1 = [int(v) for v in roi]
    np.copyto(out[y0:y1, x0:x1], m[y0:y1, x0:x1])
    return out


def _basemap_extra_mask(panel, calib, terrain, m, family=None):
    """「底图上这块也是标记色」的掩码 → 从面板掩码里减掉它，只留多出来的。

    **为什么要这一层**：有的图底图本身就到处黄褐色（实测东部岩山V 那张图，
    纯颜色层命中 7000+ 像素）。玩家标记是**画在底图之上**的东西，所以
    「面板命中 − 底图在同一个位置也命中」才是标记（NPC 点同样会被留下，靠
    尺寸/方形度和跨帧跟踪再筛）。底图与面板有几何误差，减之前把底图掩码**胀**开
    几个像素，宁可信底图（少留几个候选）也别把底图的纹理当标记。
    """
    sx, sy = map_calib.scales_of(calib)
    ox, oy = calib.get("offset") or (0, 0)
    vx, vy = (calib.get("view") or (0, 0)) if calib.get("mode") == map_const.MODE_CROP else (0, 0)
    h, w = panel.shape[:2]
    # canvas = (panel - offset)/scale + view  ⇒  panel = (canvas - view)*scale + offset
    mat = np.float32([[sx, 0, ox - vx * sx], [0, sy, oy - vy * sy]])
    warped = cv2.warpAffine(terrain.canvas, mat, (w, h),
                            flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
    base = dot_mask(warped, family)
    # 膨胀核跟着缩放走（2026-09-27 改两轴后取两轴里**大的那个**：宁可多胀一点，
    # 也不要比原来胀得少 —— 这条本来就是"宁可信底图"）
    k = max(3, int(round(max(sx, sy) * 2))) | 1        # 奇数核
    base = cv2.dilate(base, np.ones((k, k), np.uint8))
    return (m & (1 - base)).astype(np.uint8)


def dot_candidates_diff(panel, calib, terrain, ref_w=None, roi=None, exp_side=None):
    """⭐ **差分层**的候选（用户 2026-10-06「下一步」✓）→ `(cands, dense, info)`。

    为什么要有它（只补**色族救不回来**的那种情况 ✓）：色族那两层（`color` / `basemap`）
    都**只看"玩家标记色"**（黄/青 ✓）⇒ 标记画成别的颜色、或底色偏色时**认不出** ✗。
    这一层**不看颜色** ✓ —— 面板 **减** 底图那一块（`diff_panel_vs_canvas` ✓，**一处实现** ✓）
    ⇒ 差出来的就是"**底图上没有的东西**"（玩家点 / 实时元素 ✓）。

    ⚠⚠ **两道闸**（2026-10-12 起口径见下 ✓）：
      · **对比度**：`peak < DOT_DIFF_MIN_CONTRAST` ⇒ 这图上没有"**比周围明显**"的块
        ⇒ **不给候选** ✓（否则归一化会把噪声放大成"亮点"✗）；
      · **不同源**：`floor ≥ DIFF_FLOOR_WARN` ⇒ **只留一句提示、不拦** ✗（用户 2026-10-06 ✓
        "黑底 + 黄/红/橙点的地图很适合拿来识别坐标"✓；2026-10-12 起差分还改成**逐块局部
        背景** ✓ ⇒ 均匀/成片的偏色在归一化那步就没了 ✓ 真正把关的是**局部残差**✓）。

    ⚠ 它**只是候选生成器** ✗：真假由下游定 —— 尺寸/方形度/跨帧跟踪（`PlayerDotTracker` ✓）、
      以及"离上一帧最近" ✓。**别在这儿选一个当结论** ✗（同 `_dot_candidates` 的取舍 ✓）。
    """
    if terrain is None or getattr(terrain, "canvas", None) is None:
        return [], 0, {"ok": False, "why": "没有底图"}
    vis, di = diff_panel_vs_canvas(panel, terrain.canvas, calib)
    if vis is None:
        return [], 0, dict(di, ok=False,
                           why="差分算不出来（%s）" % (di.get("why") or "?"))
    floor = float(di.get("floor") or 0.0)
    # ⚠ **`peak` 取的是"局部对比度"**（`di["peak"]` ✓ = 比周围多出来多少 ✓）——
    #   **不是** `raw_peak`（整幅最大 ✗ 见下面闸②那段说明 ✓）。
    peak = float(di.get("peak") or 0.0)
    # ⭐⭐ **"不同源"不再一票否决**（用户 2026-10-06 ✓ 原话："**黑底 + 黄/红/橙点 的地图很适合
    #   拿来识别坐标**"✓）—— 原来这一条一响就**整层拒掉** ✗（理由："整幅都在差 ⇒ 差分不是
    #   在'看内容'"✓）；可我们要的是"**多出来的那个亮团**" ✓ —— **整幅底色偏**（底图素材与
    #   游戏画面色调差 ✓ 实测这张图 `floor` 必然 ≥ 30 ✓）**不影响亮团分离** ✓（界面那张
    #   黑底图之所以好用，正因为它是**归一化**过的 ✓）。
    #   ⇒ 降级成**一句提示**（进 `info` / 日志 ✓，人看得到"这张图整幅偏"✓，但**不拦** ✗）；
    #     真正把关的是下面那条**对比度**闸 ✓ 与"被淹"（`_is_flood` ✓）——
    #     极端情形（整幅噪声）会表现成 **候选一大片** ⇒ 被淹那条自然会挡 ✓（不靠这一条 ✓）。
    #   ⭐ 2026-10-12 起**"整幅偏"连提示都不用那么重了** ✓：差分已经改成**逐块局部背景**
    #     （见 `DIFF_LOCAL_PX` ✓）⇒ 均匀/成片的亮度差在归一化那一步就没了 ✓ —— 所以那句
    #     提示改成拿 **`res_med`（局部归一之后的残差中位）** 来说话 ✓（真不同源时才拦得上话 ✓）。
    why_flat = ""
    _res_med = float(di.get("res_med") or 0.0)
    if _res_med >= map_const.DIFF_FLOOR_WARN:
        why_flat = ("连局部归一后残差还有 %g ≥ %g（底图与游戏小地图**真的**不同源）—— "
                    "但「亮团比周围明显」仍采信" % (_res_med, map_const.DIFF_FLOOR_WARN))
    elif floor >= map_const.DIFF_FLOOR_WARN:
        why_flat = ("整幅偏 %g 灰阶（底图与游戏小地图底色/透明不同）—— **已按局部背景扣掉** ✓"
                    "（残差只剩 %g）" % (floor, _res_med))
    # ⭐⭐ **闸②（对比度）**：门槛是 `DOT_DIFF_MIN_CONTRAST`(8) ✓。
    #   为什么原来那 30 会挡住好东西 ✗：界面那张黑底图**归一化**过（"峰值 ⇒ 255"✓ 见
    #   `diff_panel_vs_canvas` 的 `vis` 说明 ✓）⇒ 看着**很跳** ✓，可**绝对**对比度
    #   （`raw_peak − floor`）往往**不到 30** ✗（实测图 `110040000`：闸①拆掉后仍被挡住 ✓）。
    #   ⚠⚠ **2026-10-12 起 `peak` 已经是「局部对比度」**（差分减的是逐块局部背景 ✓ 见
    #     `DIFF_LOCAL_PX` ✓）⇒ 直接拿 `peak` 比 ✓ —— 老写法 `peak − floor`（局部峰值 −
    #     **整幅**中位 ✗）在换口径之后是**两个量纲混着算** ✗（整幅偏 60 的那张图上，一个
    #     只比周围亮 10 的假亮块会算出 10 − 60 < 0 ✗ 反而"过不了"；而真点也说不清 ✓）。
    #   于是：**门槛语义变成"比周围亮了多少"** ✓（正是这个量该管的事 ✓）——
    #   纯噪声（差分全 0 或 ±两三灰阶 ✓）⇒ `peak` 也就两三 ⇒ 仍然 `ok=False` ✓ 不动纪律 ✓。
    #   真正把关的另外三条照旧 ✓：`DOT_DIFF_THR`(128，归一后 ✓)、`_dot_candidates`（尺寸/方形度 ✓）、
    #   `_is_flood`（成片一大片 ⇒ 被淹 ✓）。
    #   ⚠ 用例 `t_dot_diff_layer` ③（**带噪声、没有亮点 ⇒ 不许给候选** ✓）就是这条的守门员 ✓。
    if peak < map_const.DOT_DIFF_MIN_CONTRAST:
        return [], 0, dict(di, ok=False,
                           why="没有比周围明显突出的块（最突出的才比周围高 %g 灰阶 < %g%s）"
                               % (peak, map_const.DOT_DIFF_MIN_CONTRAST,
                                  "；且" + why_flat if why_flat else ""))
    # ⭐⭐ **别再来一次 `vis.max(axis=2)`**（用户 2026-10-06 ✓ 帧率现场：`_amax` 占 **9.4%** ✗）：
    #   `vis` 是 `cvtColor(gv, GRAY2BGR)` 出来的 ✓ ⇒ **三个通道完全相等** ⇒ `max` 等于
    #   **任何一个通道** ✓（同值 ✓）；而 `info["gray"]` 就是那份本色 ✓ ⇒ 直接拿它算门限 ✓
    #   （省掉一次整幅三通道归约 + 一次 BGR 拷贝 ✓，**掩码逐像素同值** ✓）。
    #   ⚠ `color=True`（界面那一路把点按原色画 ✓）时三通道**不等** ⇒ **退回 `max`** ✓（不赌 ✓）。
    _g = di.get("gray")
    if _g is not None and not di.get("color"):
        m = (_g >= map_const.DOT_DIFF_THR).astype(np.uint8)
    else:
        m = (vis.max(axis=2) >= map_const.DOT_DIFF_THR).astype(np.uint8)
    m = _apply_roi(m, roi)
    # ⭐⭐ **底图透明的地方，不是"多出来的东西"**（用户 2026-10-06 ✓ 现场实测 ✓）：
    #   这张图的底图 PNG **65.0% 像素是透明的**（小地图面板之外的圆角区 ✓，读进来是**纯黑** ✗）
    #   ⇒ 游戏在那儿画什么都会"比底图多" ✗ ⇒ 冒出一堆假候选（现场那个"像核心色"的假图标
    #   就落在半空 (914.5,313) ⇒ 世界 (546,**−327**) ✗，正是这么来的 ✓）。
    #   判据用**底图自己的 alpha**（`canvas_alpha` ✓ 现成字段 ✓ 不新造数 ✓）：透明 ⇒ 这格
    #   **不算"多出来"** ✓（那儿本来就不是地图，谈不上"多了个东西"✓）。
    #   ⚠ 没有 alpha（PNG 不带透明 ✓）、或标定算不出来 ⇒ **不筛** ✓（零回归 ✓）。
    _alpha = getattr(terrain, "canvas_alpha", None)
    if _alpha is not None:
        try:
            _sx, _sy = map_calib.scales_of(calib)
            _ox, _oy = (list(calib.get("offset") or (0, 0)) + [0, 0])[:2]
            _vx, _vy = ((calib.get("view") or (0, 0))
                        if calib.get("mode") == map_const.MODE_CROP else (0, 0))
            _mat = np.float32([[_sx, 0, _ox - _vx * _sx],
                               [0, _sy, _oy - _vy * _sy]])
            _aw = cv2.warpAffine(_alpha, _mat,
                                 (panel.shape[1], panel.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
            m = (m & (_aw > 0).astype(np.uint8)).astype(np.uint8)
        except Exception:                   # noqa: BLE001 —— 算不出来就当没有 alpha ✓
            pass
    cands, dense = _dot_candidates(m, ref_w or panel.shape[1], exp_side)
    return cands, dense, dict(di, ok=True, flat=why_flat)


def _drop_on_canvas_marks(cands, fam, calib, terrain, panel):
    """**候选落在"底图上本来就有的黄/青"上 ⇒ 它是地图元素 ⇒ 剔掉** ✓（用户 2026-10-06 ✓）。

    **为什么非要这一刀**（实测，图 `110040000`、`tools/_probe_cands.py` ✓）：
      · **颜色层**这一帧给出 **15 个候选 / 25568 像素** ✗ —— 逐个查"换到底图同一位置是什么"，
        **15 个全坐在底图自己的黄色地形/图标上** ✓（`(102,204,238)`、`(119,221,255)`… ✓）
        ⇒ 这图的底图**本身就是一片黄褐** ⇒ 颜色层在这张图上"认谁都是玩家" ✗；
      · 上一场"锁在假点上"就是这么来的 ✓（跨帧跟踪挑离上一帧最近的那个 ⇒ 一锁锁死 ✓）。
    **判据**（一处实现 ✓ 不新造数 ✓）：`panel_to_canvas` 把候选换到底图像素 ✓，
      再看底图在该点的 `dot_mask(..., fam)`（**同一套色族判据** ✓）—— 有 ⇒ 地图元素 ⇒ 剔 ✓。
    ⚠⚠ **零回归**（同本项目一贯口径 ✓）：**一个都没剩下 ⇒ 原样返回** ✗ ——
      "宁可这一拍认不出（由防抖沿用上一帧 ✓、再触发重捕 ✓），也绝不把本来找得到的点弄丢" ✓。
    ⚠ 换不到底图上（点在底图外 / 标定不全 ✓）⇒ **不剔** ✓（没证据就别判死 ✗）。
    """
    if not cands or terrain is None or getattr(terrain, "canvas", None) is None \
            or not map_calib.has_geometry(calib):
        return cands
    try:
        cmask = dot_mask(terrain.canvas, fam)
    except Exception:                       # noqa: BLE001 —— 底图算不出来就不剔 ✓
        return cands
    ch, cw = cmask.shape[:2]
    kept = []
    for c in cands:
        px = py = None
        if isinstance(c, dict):
            px, py = c.get("x"), c.get("y")
        else:
            try:
                px, py = c[0], c[1]
            except Exception:               # noqa: BLE001
                px = py = None
        if px is None or py is None:
            kept.append(c)                  # 认不出这个候选的形状 ⇒ 不剔 ✓
            continue
        try:
            gx, gy = map_calib.panel_to_canvas(float(px), float(py), calib)
            gxi, gyi = int(round(gx)), int(round(gy))
            y0, y1 = max(0, gyi - 2), min(ch, gyi + 3)
            x0, x1 = max(0, gxi - 2), min(cw, gxi + 3)
            if y1 <= y0 or x1 <= x0:
                kept.append(c)              # 落在底图外 ⇒ 不剔 ✓
                continue
            if bool(cmask[y0:y1, x0:x1].any()):
                continue                    # ⭐ 底图那儿本来就有这个色 ⇒ 地图元素 ⇒ 剔 ✓
        except Exception:                   # noqa: BLE001
            pass
        kept.append(c)
    return kept if kept else cands          # ⚠ 全被剔掉 ⇒ 原样返回（零回归 ✓）


def _prefer_on_foothold(cands, calib, terrain, panel):
    """⭐ **只留下"站在平台上"的候选**（用户 2026-10-06 ✓ 现场："我现在到了 1176,178，
    Agent 为什么不认这个坐标？"✗ —— 它挑的是最左边那个 ⇒ 报 −1911 ✗）。

    **为什么这条最硬**：人**只能站在平台上** ✓ —— 候选换出来的世界坐标，用
      `terrain.foothold_below` 查一下**脚下面** ✓ ⇒ **没有平台的一定不是玩家** ✓。
    现场数字（图 `110040000`、面板 `1164×585` ✓）：
      · 真点 `(1135.2,479.5)` ⇒ 世界 `(1176.2,178.2)` ✓ —— y=178 **落在「底层」平台
        `y 143..182` 里** ✓ ⇒ **留** ✓；
      · 假点 `(45.8,473.9)` ⇒ 世界 `x≈−1911`、**y≈128** ✗ —— **平台外（半空）** ✗ ⇒ **剔** ✓。
    ⚠ 单看 x 分不开（两个 x 都在段 9 的 `−2002..1202` 里 ✓）—— **是 y 这一维分开的** ✓，
      所以判据必须是"**脚下有没有面**"（几何 ✓），不能只看 x ✗。
    ⚠⚠ **零回归**（老规矩 ✓）：**一个都没站在平台上 ⇒ 原样返回** ✗ ——
      "宁可这一拍按老的挑法（不猜 ✓），也绝不把本来找得到的点弄丢" ✓。
    ⚠ 换不出坐标（无标定 / 无地形 / 点在底图外 ✓）⇒ **不剔** ✓（没证据就别判死 ✗）。
    """
    if not cands or terrain is None or not map_calib.has_geometry(calib):
        return cands
    kept = []
    for c in cands:
        if isinstance(c, dict):
            px, py = c.get("x"), c.get("y")
        else:
            try:
                px, py = c[0], c[1]
            except Exception:                       # noqa: BLE001
                px = py = None
        if px is None or py is None:
            kept.append(c)                          # 形状认不出 ⇒ 不剔 ✓
            continue
        try:
            wx, wy = map_calib.panel_to_world(float(px), float(py), calib, terrain)
            f = terrain.foothold_below(wx, wy)
            if f is not None:
                # ⭐⭐ **还要"真的站在面上"**（用户 2026-10-06 ✓ 现场：App 挑到 (198,195) ✗
                #   —— 而「底层」那条面在 y 143..182 ✓ ⇒ y=195 是**贴着面下方的半空** ✗，
                #   光判"脚下面存在"会把这种**收下** ✗（我第一版就是这么松的 ✓）。
                #   判据：|玩家 y − 那个面在 x 处的 y| ≤ **12**（`PORTAL_DY`/`LADDER_PAD` 同款
                #   "同一层"容差 ✓ 不新造数 ✓）；面没有 `y_at`（老对象/桩 ✓）⇒ 不收紧 ✓。
                _yat = getattr(f, "y_at", None)
                _gy = None
                if callable(_yat):
                    try:
                        _gy = float(_yat(wx))
                    except Exception:               # noqa: BLE001
                        _gy = None
                if _gy is None or abs(float(wy) - _gy) <= 12.0:
                    kept.append(c)                  # ⭐ 就站在面上 ⇒ 像个玩家 ✓
        except Exception:                           # noqa: BLE001
            kept.append(c)
    return kept if kept else cands


def find_player_dot(panel, calib=None, terrain=None, roi=None, near=None,
                    ref_w=None, exp_side=None):
    """面板里找玩家标记 → 结论 dict（**认不出也是结论，不要硬凑**）。

    搜索顺序 = **黄族 → 青族**（⚠ **每族先「底图相减层」、再「颜色层」** —— 用户 2026-10-06：
    "**优先使用底图相减找黄点**"✓ 理由与现场见下面 loop 里那段 ✓）⇒ 都落空才试
    ⭐ **差分层**（`dot_candidates_diff` ✓ 用户 2026-10-06「下一步」✓）；第一个「有候选
    且不算被淹」的组合胜出。分开判很要紧：实测寺院通道2 的**地图本身就是饱和青色**，
    青族被淹，而黄族只有玩家那一个点 —— 混在一起判会把这条结论一起丢掉。
    ⚠ **差分层为什么排最后**：前两层是"**认标记色**"（准 ✓ 漏检时全落空 ✗），差分层是
      "**认底图上没有的东西**"（不看颜色 ⇒ 补得了漏检 ✓ 但整幅不同源时会乱响 ✗）
      ⇒ 只在前面全落空时兜一次 ✓（"宁可认不出，也别给假的"✓）。

    返回：
        ok         找到像标记的点
        x, y       面板像素坐标（重心；ok 时才有意义）
        w, h, area 外接矩形与像素数
        bgr        重心附近的实际颜色（中位）
        mask_px    胜出族的命中像素数
        dense_px   胜出族的「成片」像素数（判断被淹用它，不是 mask_px）
        candidates 胜出族的候选个数
        family     "yellow" / "cyan" / "diff"（`"diff"` = **不看颜色**那层 ⇒ 它没有族 ✓）
        layer      "color" / "basemap" / "diff"（结论来自哪一层）
        flood      True = 颜色层没帮上忙（靠底图层给的，或干脆给不出）
        diag       每族的诊断句（拒绝时拼进 reason，便于人判断该怎么救）
        roi        实际用的搜索区（有 fit 标定时会自动排除面板标题带）
        reason     一句话，能直接贴进日志/状态行

    `roi` 不传时：有 fit 标定就自动用「底图覆盖区」（排除面板标题/边框）。
    `near`（面板坐标）不传时是**首次捕获**语义；传了表示"我上一帧在这儿" ——
    那时多候选按**离 near 最近**挑，而且"被淹"不再是否决条件（见 loop 里那段）。
    """
    used_roi = roi or _calib_roi(panel.shape[:2], calib, terrain)
    if used_roi:
        rx0, ry0, rx1, ry1 = [int(v) for v in used_roi]
        area_search = max(1, (rx1 - rx0) * (ry1 - ry0))
    else:
        area_search = max(1, int(panel.shape[0] * panel.shape[1]))

    diag = []
    any_flood = False
    hit = None
    _diff_first_tried = False
    # ---- ⓪'' ⭐⭐⭐ **先拿「玩家点核心色」当探针**（用户 2026-10-06 ✓ 现场实测 ✓）----
    #   为什么它要排在**所有**层前面（现场铁证，图 `110040000`、人卡在「底层」右下角 ✓）：
    #     **玩家标记是画在"一片同族的沙"上面的** ✗ ⇒ "按族取掩码 + 找连通块"这一套会把
    #     标记和沙**连成一块** ✓（实测掩码 25719 像素、最大块 63×27 / 975 像素 ✗）⇒ 标记被
    #     吞掉 ✓ ⇒ 颜色层给出的 15 个候选**全是沙**（色中位 (102..136, 203..221, 217..255) ✗
    #     —— 与"底图同点"一模一样 ✓ 即地图元素 ✓）；而全场唯一"像点核心色"的那个偏偏是
    #     **半空里的另一个黄图标**（(914.5,313) ⇒ 世界 (546,**−327**) ✗ 人不可能在那）✓。
    #   `dot_core_mask` 用的是**探针量到的真色**（`DOT_CORE_*`：G,R ≥ 240、B ≤ 160 ✓）——
    #     沙的 G 中位只有 211 ✗ 过不去、标记核心过得去 ✓ ⇒ **标记从沙里被"分"出来** ✓
    #     （连通块不再和沙相连 ⇒ 它自己成一个候选 ✓）。
    #   ⚠ **零回归**（老规矩 ✓）：这一层**没给出候选**、或候选**全不站在平台上** ⇒
    #     照旧往下走 ✓（颜色层 / 相减层一个字不变 ✓）。
    #   ⚠ 与"底图相减层"的分工：那一层要**几何+底图** ✓（有图才跑 ✓），这一层**纯颜色** ✓
    #     ⇒ 没有底图的来源照样能用 ✓。
    if terrain is not None and map_calib.has_geometry(calib):
        _m_core = _apply_roi(dot_core_mask(panel), used_roi)
        if int(_m_core.sum()) >= DOT_SIDE_MIN * DOT_SIDE_MIN:
            _c_core, _d_core = _dot_candidates(_m_core, panel.shape[1])
            # ⚠ 先按**实测的标记尺寸**过一道（`DOT_CORE_MIN_SIDE` ✓ 说明在那里 ✓）：
            #   核心色阈值紧 ⇒ 冒出来的大多是**几个像素的杂点** ✗，而真标记短边 ≥ 21 ✓。
            #   这条同时保证"**改动不改老行为**" ✓：桩/合成的小方块（用例里的假点 ✓）
            #   过不了 ⇒ 这一层不抢戏 ⇒ 下面原路照走 ✓（实测：不加它，99/101 ✗）。
            _c_core = [c for c in _c_core
                       if min(int(c["w"]), int(c["h"])) >= DOT_CORE_MIN_SIDE]
            # ⭐ 只认"**真的站在平台上**"的那些（人只能站在平台上 ✓ —— 同
            #   `_prefer_on_foothold` 的口径 ✓）：半空那个假图标就出局了 ✓。
            _keep_core = _prefer_on_foothold(_c_core, calib, terrain, panel) \
                if _c_core else _c_core
            if _c_core and _keep_core is not _c_core:
                hit = {"family": "core", "layer": "core",
                       "mask_px": int(_m_core.sum()), "dense_px": _d_core,
                       "candidates": len(_keep_core), "all": _keep_core,
                       "flooded": False}
                diag.append("核心色层（**优先** ✓ 玩家点真色 G,R≥%d、B≤%d）：像点的块 %d 个、"
                            "其中站在平台上 %d 个 ✓"
                            % (DOT_CORE_G_MIN, DOT_CORE_B_MAX,
                               len(_c_core), len(_keep_core)))
            else:
                diag.append("核心色层（优先）：像点的块 %d 个，**没有一个站在平台上**"
                            "⇒ 这层不采信、照旧往下走 ✓" % (len(_c_core),))
    # ---- ⓪ ⭐⭐⭐ **差分层优先**（用户 2026-10-06 ✓ 原话："**黑底 + 黄/红/橙点 的地图很适合
    #   拿来识别坐标**"✓）----
    #   为什么提到最前：这一层**不看颜色** ✓ —— 面板直接减**同一张底图** ⇒ 差出来的就是
    #   "**底图上没有的东西**" ✓；而"黑底 + 亮团"正是**最好认的输入** ✓（背景被压掉、
    #   亮团一眼分离 ✓）。
    #   ⚠ 反过来说，**认颜色**那两族在"**底图本身一片黄褐**"的图上会挑错 ✗ —— 2026-10-06
    #     现场实锤（图 `110040000`、`tools/_probe_cands.py` ✓）：颜色层一口气给出
    #     **15 个候选 / 25568 像素**、**逐个查"底图同位置"全是黄的** ✗ ⇒ 锁假点 ⇒
    #     人卡在「底层」右下角、`pos` 报个平台下面的坐标（"脚下没有平台"✓）✓。
    #   ⚠ 两道老闸照旧、缺一不可 ✓：`DIFF_FLOOR_WARN`（整幅不同源 ⇒ 这层**自己就不给候选** ✓）、
    #     "被淹"（`_is_flood` ✓；`near` 有值时不再否决 ✓ 同老口径 ✓）。
    #   ⚠ **零回归**：这一层给不出候选 ⇒ 下面那两族照旧全跑一遍 ✓（顺序换了，路都在 ✓）；
    #     而且失败时**不做第二次**（`_diff_first_tried` ✓ 下面 ③ 那段据此跳过 ✓）。
    if map_calib.has_geometry(calib) and terrain is not None \
            and getattr(terrain, "canvas", None) is not None:
        _diff_first_tried = True
        _dc, _dd, _di = dot_candidates_diff(panel, calib, terrain,
                                            ref_w=ref_w, roi=used_roi,
                                            exp_side=exp_side)
        # ⭐⭐⭐ **把"最亮那一点落在谁身上"那个候选排到最前**（用户 2026-10-06 ✓ 现场实锤 ✓）——
        #   用户原话："**为什么还要担心被沙吞掉？从黑色背景里找黄点应该都没有沙了**"✓
        #   —— 完全对 ✓：差分图（面板 − 底图）**已经是黑底 + 亮团** ✓，沙被减掉了 ✓。
        #   现场（图 `110040000`、卡在底层右下角 ✓、`tools/_probe_diff_now.py` ✓）：
        #     差分层给 **4 个**候选 ✓，其中**真点就是** `(1133.4,485.8) ⇒ 世界 (1171,166)` ✓✓
        #     （用户从小地图下方信息栏读到的是 (1176,178) ✓ 同一个点 ✓）；
        #     而差分的 `info["at"]`（**"多差得最厉害"那一点** ✓ 现成字段 ✓）= **(1134,471)** ✓
        #     —— 就在真点这块上 ✓（|Δx|=0.6 ✓、|Δy|=14.8 ≤ 它的高 15 ✓）。
        #   ⛔ 坏在"挑哪一个" ✗：`_dot_candidates` 按**面积**排序 ⇒ 跟踪器弃锁重捕后
        #     `cands[0]` = `(637.2,379)`（面积 320，最大 ✗）⇒ 又锁到假点上 ✓（现场循环就是这么来的 ✓）。
        #   ⇒ 判据：`at` 落在这个候选**自己的尺寸范围**内（`|Δ| ≤ w / ≤ h` ✓）⇒ 它就是"**最亮的
        #     那一团**" ✓ ⇒ **排到最前** ✓（没有候选满足 ⇒ 一动不动、保持原序 ✓ 零回归 ✓）。
        #   ⚠ 为什么这条踏实：它复用**差分那一层已经算好的**`at` ✓（不是新造数 ✓），而且只改
        #     **顺序**（多候选仍在 ✓ 给跨帧跟踪 ✓）—— 认颜色/形状那些老闸一个字没动 ✓。
        try:
            _at = _di.get("at")
            if _at and _dc:
                def _on_at(c, _a=_at):
                    return (abs(float(c["x"]) - float(_a[0])) <= float(c["w"])
                            and abs(float(c["y"]) - float(_a[1])) <= float(c["h"]))
                _front = [c for c in _dc if _on_at(c)]
                if _front:
                    _dc = _front + [c for c in _dc if c not in _front]
                    _di = dict(_di, on_at=True)
        except Exception:                   # noqa: BLE001 —— 排不了就保持原序 ✓
            pass
        _df = _is_flood(len(_dc), _dd, area_search)
        if _dc and (not _df or near is not None):
            hit = {"family": "diff", "layer": "diff", "mask_px": 0,
                   "dense_px": _dd, "candidates": len(_dc),
                   "all": _dc, "flooded": bool(_df)}
            diag.append("差分层（**优先** ✓ 不看颜色）：像点的块 %d 个、成片 %d 像素%s"
                        % (len(_dc), _dd,
                           "；" + str(_di.get("flat")) if _di.get("flat") else ""))
        else:
            # ⚠ "整幅偏"这句话**无论成功失败都要出现**（用户 2026-10-06 口径 ✓）——
            #   它是"这张图底色不一样"的**透明度** ✓：只影响人怎么读这条结论 ✓，**不拦** ✓。
            diag.append("差分层（优先）：%s%s%s"
                        % (_di.get("why") or "没给出候选",
                           "（被淹）" if _df else "",
                           "；" + str(_di.get("flat")) if _di.get("flat") else ""))
    for fam in DOT_FAMILIES:                   # 黄族优先（实测玩家标记就是黄的）
        m0 = _apply_roi(dot_mask(panel, fam), used_roi)
        mask_px = int(m0.sum())
        # ⭐⭐⭐ **底图相减优先**（用户 2026-10-06 ✓ 原话："**优先使用底图相减找黄点**"✓）——
        #   原来是"颜色层先、相减层后" ✗ ⇒ 颜色层只要凑出**任意一个**不算被淹的候选就赢 ✓
        #   ⇒ 而小地图上**别的黄色图标**（NPC / 传送点 / 任务点 ✓）跟玩家点同属"黄族" ✗
        #   ⇒ 它们一起进候选 ✓，再由"跨帧跟踪"按"离上一帧最近"挑 ✗ ⇒ **第一拍挑错就锁死** ✓。
        #   现场实锤（2026-10-06，图 `110040000`、实时取证 `tools/_probe_chains.py` ✓）：
        #     面板 1164×585 ✓、标定自洽（反推 1161×584 ✓）、**每帧都有 2 个候选** ✓、
        #     跟踪锁死其中一个 ⇒ **5 帧一动不动** ✓ ⇒ 世界 (−212.7,−834.7) 落在**段 4** ✗
        #     （真值是**段 9「底层」右端 ≈1202** ✓）⇒ Agent 拿它择路 ⇒ 一路怼墙 ✓。
        #   **为什么相减优先能治**：底图上**本来就有的**东西会被减掉 ✓ ⇒ 假图标（地图元素 ✓）
        #   从候选里消失 ✓，而玩家点是"**底图上没有的东西**" ✓ ⇒ 留得下 ✓。
        #   ⚠ 代价（照旧保留退路 ✓）：相减层在"整幅不同源"时会乱响 ✗ ⇒
        #     ① `mode` 没几何 / 没底图 ⇒ 这层直接跳过 ✓（下面 `continue` ✓）；
        #     ② 减掉的像素不到一半 ⇒ 这层没意义 ⇒ 跳过 ✓（原判据 ✓ 一行没动 ✓）；
        #     ③ 黄族相减层仍走 `_blob_looks_core`（得像玩家点核心色 ✓）那道闸 ✓；
        #     ⇒ 相减层给不出候选时，**颜色层照旧兜底** ✓（顺序换了，两条路都还在 ✓）。
        for layer in ("basemap", "color"):
            if layer == "basemap":
                # 颜色层救不回来时才用「面板 − 底图」：这一层要标定 + 底图
                if not map_calib.has_geometry(calib) or terrain is None \
                        or getattr(terrain, "canvas", None) is None:
                    continue
                m = _basemap_extra_mask(panel, calib, terrain, m0, fam)
                if int(m.sum()) >= mask_px * 0.5:   # 没减掉多少 → 这层没意义
                    continue
            else:
                m = m0
            # `ref_w`：**给裁剪图用**。像点的边长上限是按面板宽度定的，而局部
            # 搜索传进来的是一小块（37×37）—— 按那块的宽度算，上限只剩 6px，
            # 28×24 的玩家点会被尺寸筛选直接挡掉（实测就是这么"局部搜索没生效"的）。
            cands, dense = _dot_candidates(m, ref_w or panel.shape[1], exp_side)
            # ⭐⭐ **再剔一道"落在底图自己的标记色上"的假候选**（用户 2026-10-06 ✓ 见
            #   `_drop_on_canvas_marks` 里那段实测：颜色层 15 个候选**全坐在底图黄色上** ✗）。
            #   ⚠⚠ **这一道对"跟踪态"同样生效** ✓ —— 跟踪态走的是小范围局部搜 ✓，上面那条
            #     "底图相减层"常常被它的三道退路跳过 ✓ ⇒ 那时**只有颜色层在说话** ✗ ⇒
            #     假点照样锁着 ✓（2026-10-06 现场："**重启了还是 198.3**"✗ 缺的就是这半 ✓）。
            _n_before = len(cands)
            cands = _drop_on_canvas_marks(cands, fam, calib, terrain, panel)
            # ⭐ **剔了几个要留痕**（用户 2026-10-06 ✓ 现场："一到底层就疯狂报认不出黄点"✗）——
            #   底层那一带**底图本身就是黄褐** ⇒ 这道闸有"把真点也剔了"的风险 ✗ ⇒
            #   日志里必须看得到"剔了几个 / 剩几个" ✓（否则只能猜 ✓ 我最怕的就是这种
            #   自己加的判据悄悄把好点弄丢 ✗）。
            if len(cands) != _n_before:
                diag.append("%s族/%s层：%d 个候选里「落在底图标记色上」剔掉 %d 个"
                            "（剩 %d 个%s）"
                            % ("黄" if fam == "yellow" else "青", layer,
                               _n_before, _n_before - len(cands), len(cands),
                               "；全剔光 ⇒ 退回原候选" if len(cands) == _n_before else ""))
            # ⭐⭐ **底图相减层里再拒一道"得像玩家点"**（用户 2026-10-06 ✓ 现场原话："**那就是位置
            #   状态把黄点认错了，刚才人在底层最右边角落卡死**"✓ —— 真色是探针量到的
            #   **BGR(108,255,255)** ✓ 见 `DOT_CORE_*` 的说明 ✓）：
            #   这一层只看"比底图多出来的东西" ⇒ 底图杂色（黄褐 ✓）多时会**锁假点** ✗。
            #   ⚠ **只在黄族、且只在"能保住至少一个候选"时才采用** ✓（一个都没保住 ⇒ 用原来的 ✓）
            #     ⇒ **零回归**：这条闸最坏就是没生效 ✓（绝不把本来找得到的点弄丢 ✗）。
            if layer == "basemap" and fam == "yellow" and cands:
                _n_all = len(cands)
                _core = [c for c in cands if _blob_looks_core(panel, m, c)]
                if _core:
                    cands = _core
                    diag.append("黄族/basemap层：%d 个候选里只留 %d 个「像玩家点核心色」✓"
                                % (_n_all, len(_core)))
                else:
                    diag.append("黄族/basemap层：%d 个候选**没有一个像玩家点核心色**"
                                "（量测邻域 ✓）⇒ 这条闸不生效、照用原候选 ✓" % (_n_all,))
            flooded = _is_flood(len(cands), dense, area_search)
            # `near`（有上一帧位置）时**被淹不再是否决条件**：那会儿搜的就是
            # 上一帧周围一小块，"离上一帧最近"比"整张图大不大"可靠得多。
            if cands and (not flooded or near is not None):
                hit = {"family": fam, "layer": layer, "mask_px": mask_px,
                       "dense_px": dense, "candidates": len(cands),
                       "all": cands, "flooded": bool(flooded)}
                break
            any_flood = any_flood or flooded
            diag.append("%s族/%s层：像点的块 %d 个、成片 %d 像素%s"
                        % ("黄" if fam == "yellow" else "青", layer, len(cands),
                           dense, "（被淹）" if flooded else ""))
        # ⭐⭐⭐⭐ **贴边那一档**（2026-10-23 ✓ 用户："**怎么没认出来**"）：
        #   常规两档都没结果时，用**不挖边**的颜色掩码再找一遍，只收贴着最外 2 px 的块 ✓
        #   （病根见 `_dot_candidates_edge` 的说明：`DOT_INSET=3` 把贴边的黄点整圈挖掉了 ✓
        #     实测同一个点：中心 x=106 ⇒ 认得出；107/108 ⇒ 掩码 0 ⇒ 认不出 ✗）。
        #   ⚠ 放在这儿（**两档之后、差分层兜底之前**）：差分层不看颜色、假点风险更大 ✓；
        #     而且这一档本身**就是按黄族颜色卡的** ✓ 比差分层可信 ✓。
        if hit is None:
            _ec, _etot = _dot_candidates_edge(panel, fam, ref_w or panel.shape[1], exp_side)
            if _ec:
                hit = {"family": fam, "layer": "edge", "mask_px": _etot,
                       "dense_px": _etot, "candidates": len(_ec),
                       "all": _ec, "flooded": False}
                diag.append("%s族/贴边层：常规模板挖掉了边缘 3px ⇒ 这一档在**贴边**找到 %d 个"
                            "（黄点走到地图最边上时就这样 ✓ 位置差 ≤ 2px ✓）"
                            % ("黄" if fam == "yellow" else "青", len(_ec)))
        if hit is not None:
            break

    # ---- ③ 差分层**兜底那一份**（⚠ 2026-10-06 起它已经**提到最前面**了 ✓ 见上面 ⓪ ✓）----
    #   ⇒ 这里只在"**上面那次没试过**"时才跑 ✓（`_diff_first_tried` ✓）—— 否则同一帧会
    #   算两遍差分、`diag` 里还会出现两条一样的话 ✗（那正是"两处各写一份"的观感 ✗）。
    #   ⚠ 保留这一段（而不是删掉 ✗）的理由：**调用方可能不进上面那条路**（没有几何/没有
    #   底图时 ✓ 上面 ⓪ 整段跳过 ⇒ `_diff_first_tried` 仍 False ✓）⇒ 这儿好歹再试一次 ✓
    #   （等价于老行为 ✓ 零回归 ✓）。
    if hit is None and not _diff_first_tried and map_calib.has_geometry(calib) \
            and terrain is not None \
            and getattr(terrain, "canvas", None) is not None:
        dcands, ddense, dinfo = dot_candidates_diff(panel, calib, terrain,
                                                    ref_w=ref_w, roi=used_roi,
                                                    exp_side=exp_side)
        dflood = _is_flood(len(dcands), ddense, area_search)
        if dcands and (not dflood or near is not None):
            hit = {"family": "diff", "layer": "diff", "mask_px": 0,
                   "dense_px": ddense, "candidates": len(dcands),
                   "all": dcands, "flooded": bool(dflood)}
            diag.append("差分层：像点的块 %d 个、成片 %d 像素（**不看颜色** ✓ 兜底）"
                        % (len(dcands), ddense))
        else:
            diag.append("差分层：%s%s"
                        % (dinfo.get("why") or "没给出候选",
                           "（被淹）" if dflood else ""))

    base = {"mask_px": 0, "dense_px": 0, "flood": False, "candidates": 0,
            "layer": "", "family": "", "all": [], "roi": used_roi,
            "diag": diag, "w": 0, "h": 0, "area": 0,
            "x": 0.0, "y": 0.0, "bgr": None}

    if hit is None:
        # 认不出就**认不出**：这时候给一个"最像的块"，下游会当成"我在哪块平台上"，
        # 比认不出糟得多（实测东部岩山V 那张图：底图本身就到处黄褐色）。
        why = "；".join(diag) if diag else "两族都没有像点的块"
        pre = "颜色层不可用" if any_flood else "认不出玩家标记"
        return dict(base, ok=False, flood=any_flood, reason=(
            "%s（%s）—— 换来源（独立推流），或给它一份底图标定走「底图相减」"
            % (pre, why)))

    cands = hit["all"]
    # ⭐⭐ **最终挑之前，先只留"站在平台上"的**（用户 2026-10-06 ✓ 见 `_prefer_on_foothold` ✓）——
    #   ⚠ 放在**这里**（胜出层已经定、还没挑那一个 ✓）⇒ **所有层**（黄/青/差分层 ✓）一起受益 ✓，
    #     不用在各层里各写一遍 ✗（一处实现 ✓）。
    #   为什么非要有它：现场"4 个候选里挑到最左边那个"✗ ⇒ 报 **−1911**（半空 ✓）⇒ 一路怼墙 ✓；
    #     而真点是 `(1135,479) ⇒ 世界 (1176,178)`，**脚下正好是「底层」那条平台** ✓。
    # ⛔⛔ **这里曾经有过一刀："一个候选都不在平台上 ⇒ 这一拍不报坐标"** ✗ —— 已**撤回** ✓。
    #   为什么撤（如实记 ✓）：判据本身是对的（人只能站在平台上 ✓），可它**假设"地形里
    #   一定查得到平台"** ✗ —— 用例里的桩地形（只有几个 foothold 的简对象 ✓）与"玩家在
    #   半空跳"这种**正常帧**都满足不了它 ⇒ 一开就让套件 **95/101** ✗（我实测的 ✓），
    #   还会把半空中的正常读数判成"认不出" ✗ ⇒ 属于"我新加的判据悄悄把好点弄丢" ✓
    #   （本项目最怕那一类 ✗）。**治理"拿假坐标去择路"要另找地方** ✓：正确的位置是
    #   **消费端**（位置状态 / 择路那一层 ✓ 那里才知道"脚下有没有平台"意味着什么 ✓），
    #   不是挑点这一层 ✗（挑点的职责止于"这像素像不像标记" ✓）。
    cands = _prefer_on_foothold(cands, calib, terrain, panel)
    # ⭐⭐⭐ **认颜色**（用户 2026-10-06 ✓ 原话："**角色传送点是会突变的，你得认颜色**"✓）——
    #   为什么不能用"位移/连续性"当主判据 ✗：**传送会突变** ✓（位置整个跳走 ✓），
    #   靠"离上一帧最近"会把传送当成假点丢掉 ✗。
    #   本项目的做法一直是**认标记色** ✓：玩家点的核心色是**探针量出来的 BGR(108,255,255)**
    #   ✓（见 `DOT_CORE_*` 与 `_blob_looks_core` ✓），而底图上那些"黄"是**暗一档的黄褐** ✗
    #   ⇒ 拿真色卡一遍就能把它们排掉 ✓（这正是"底图一片黄褐"那张图上唯一能分辨的东西 ✓）。
    #   ⚠ 复用现成那道闸（`_blob_looks_core` ✓ 一处实现 ✓ 不新造颜色常数 ✓）；
    #   ⚠⚠ **零回归**：一个都没保住 ⇒ **原样返回** ✗（宁可照老样子，也不把点弄丢 ✓）；
    #   ⚠ 抛异常（面板形状怪 / 掩码算不出 ✓）⇒ 不筛 ✓。
    try:
        _myellow = dot_mask(panel, "yellow")
        _core_ok = [c for c in cands if _blob_looks_core(panel, _myellow, c)]
        if _core_ok:
            cands = _core_ok
    except Exception:                       # noqa: BLE001 —— 筛不动就照旧 ✓
        pass
    # ⭐⭐⭐ **挑哪一个：分数 = "像玩家点" − 距离惩罚，再加"锁定滞后"**（2026-10-10 ✓
    #   用户三条："**队友橙色点头上的小标记也是黄点，大黄点锁定错了**"✗ /
    #   "你的锁定标记会**乱飘**到别的点上去"✗ / "**瞄准的不是黄点正中心**"✗
    #   ⇒ 见 `DOT_PLAYER_SIDE_REL` 上面那段完整说明 ✓）。
    #   ⚠⚠ 老写法是 `min(cands, key=离上一拍的距离)` ✗ —— **谁近就锁谁**：
    #     队友头上那个**小黄标记**只要离得近，就能把**大黄点**（真玩家标记）顶掉 ✓；
    #     两块分数接近时还会**来回翻** ⇒ 读数乱飘 ✓（这就是"乱飘"的来源 ✓）。
    _pw = float(ref_w or panel.shape[1])
    # ⭐⭐ 期望边长：**自标定优先**（`exp_side` ✓ 见 `DOT_PLAYER_SIDE_REL` 那张量测表 ✓）
    try:
        _es = float(exp_side) if exp_side is not None else 0.0
    except (TypeError, ValueError):
        _es = 0.0
    _exp = _es if _es >= 2.0 else max(4.0, float(DOT_PLAYER_SIDE_REL) * _pw)
    if near is not None:
        def _score(t):
            """像不像玩家点（越大越好 ✓）− **很小的**距离权重；还贴着上一拍那个点 ⇒ 再加滞后分 ✓。

            ⚠⚠ 2026-10-11 两处改动（见 `DOT_REACH_REL` / `DOT_LIKE_MIN` 的说明 ✓）：
              · **距离从"线性重罚"改成"闸 + 很小的权重"** ✗：旧口径罚分最高 0.7，而实测
                `like` 之间的差距常在 0.05~0.2 ⇒ 实际上是"谁近谁赢" ✗（乱飘 / 锁错的来源 ✓）；
              · **滞后分要"像个玩家点"才给** ✗：原来只要离上一拍近就 +0.12 ⇒ 锁在一个
                形状完全不像的东西上以后，它**永远拿这份保护** ✓。
            """
            _like = dot_player_like(t, _pw, exp_side)
            _dist = ((t["x"] - near[0]) ** 2 + (t["y"] - near[1]) ** 2) ** 0.5
            _reach = max(6.0, float(DOT_REACH_REL) * _exp)
            if _dist > _reach:
                _s = _like - 0.7            # 超出"可达半径" ⇒ **一律垫底** ✓（人走不了那么远 ✓）
            else:
                _s = _like - float(DOT_DIST_W) * (_dist / _reach)
            if _dist <= max(_exp, 3.0) and _like >= DOT_LIKE_MIN:
                _s += DOT_LOCK_MARGIN        # ⭐ 还是"现在锁着的那个" **且它像个点** ⇒ 滞后 ✓
            return _s
        c = max(cands, key=_score)
    else:
        # 没有上一帧（首帧 / 重捕后）⇒ 取"**最像玩家点**"的那个 ✓
        #   （`_dot_candidates` 已按它排过序 ✓；差分层那份在这里现算一遍 ⇒ 口径一处 ✓）
        c = max(cands, key=lambda t: (dot_player_like(t, _pw, exp_side),
                                      t.get("area") or 0))
    # ⭐⭐⭐⭐ **出口再验一次颜色**（用户 2026-10-17 ✓ 原话：「**实测黄点识别十分不准会把红点、
    #   地图背景完全跟黄点不像的认成黄点**」✗）
    #   ⛔ 病根：**差分层 / 底图层不看颜色** ✓（它们认的是"底图上没有的东西" / "和底图对不上" ✓）
    #      ⇒ 红点、绿点、队友标记、甚至一块背景，只要"和底图不一样"就都能当候选 ✗；
    #      而上面那道 `_blob_looks_core` 是**整批筛**（`if _core_ok: cands = _core_ok` ✓）——
    #      **一个都没过时它就不筛** ✗ ⇒ 红点原样过关 ✓（正是用户看到的那件事 ✓）。
    #   ⇒ 现在：**只要结论来自"不看颜色的层"**（`diff` / `basemap` ✓），最后选中的这个点
    #     **必须自己过得了"玩家点核心色"** ✓，过不了就**如实说认不出** ✓
    #     （本文件的老规矩：宁可认不出，也别给假的 ✓ —— 假坐标会被 `segment_of` 拿去择路 ✓）。
    #   ⚠ 零回归：颜色层（`yellow` 族 ✓）**不拦** ✗ —— 它就是按黄色卡出来的 ✓；
    #     核心色层（`core` ✓）本来就过了核心色 ✓；真点（BGR≈(108,255,255) ✓）照旧过 ✓；
    #     合成用例里那些**黄色**测试点也照旧过 ✓（不是黄色的小方块会**当场变红** ✓ 见用例 ✓）。
    if hit.get("layer") in ("diff", "basemap") or hit.get("family") in ("diff", "core"):
        try:
            _ok_y = _blob_looks_core(panel, dot_mask(panel, "yellow"), c)
        except Exception:                      # noqa: BLE001 —— 算不出就**不拦** ✓（零回归 ✓）
            _ok_y = True
        if not _ok_y:
            return dict(base, ok=False, flood=True, reason=(
                "候选 (%.0f,%.0f) 块 %dx%d **颜色不像玩家标记**（%s 不看颜色 ⇒ 可能是红点/"
                "背景 ✓）⇒ 宁可不认 ✓"
                % (c["x"], c["y"], int(c["w"]), int(c["h"]),
                   "差分层" if hit.get("layer") == "diff" else
                   ("底图层" if hit.get("layer") == "basemap" else "核心色层"))))
    cx, cy = int(round(c["x"])), int(round(c["y"]))
    patch = panel[max(0, cy - 3):cy + 4, max(0, cx - 3):cx + 4]
    # 颜色取**这个族判据命中的那些像素**的中位 —— 直接取补丁中位会被周围压暗，
    # 报出来是个"看着不像标记"的颜色（实测：标记核心 R,G≈245，补丁中位只有 167）。
    pm = dot_color(patch, hit["family"]) if patch.size else None
    sel = patch[pm] if pm is not None else None
    if sel is not None and len(sel):
        med = np.median(sel, axis=0).astype(int)
    elif patch.size:
        med = np.median(patch.reshape(-1, 3), axis=0).astype(int)
    else:
        med = np.zeros(3, int)
    # ⚠ 差分层**没有"族"**（它不看颜色 ✓）：别写成"diff 族 / diff 层" ✗（日志里看着像坏了 ✓）。
    _who = ("差分层" if hit["layer"] == "diff"
            else "%s 族 / %s 层" % (hit["family"],
                                    "贴边" if hit["layer"] == "edge" else hit["layer"]))
    reason = ("找到玩家标记（%s）" % _who if len(cands) == 1 else
              "找到 %d 个候选（%s；跨帧跟踪会挑）" % (len(cands), _who))
    if hit["layer"] == "edge":
        # ⭐ **如实说"这个点贴边、被裁了一点"**（2026-10-23 ✓）：位置误差 ≤ 2 px
        #   ⇒ 换到世界坐标是 ≤ 17 px（这张图 1 面板像素 ≈ 8.5 世界单位 ✓）
        #   —— 现场/日志要看得见，别让人以为"和平时一样准" ✓。
        reason += "·**贴边（被面板边缘裁掉一角，位置差 ≤ 2 px）**"
    out = dict(base, ok=True, x=c["x"], y=c["y"], w=c["w"], h=c["h"],
               area=c["area"], bgr=tuple(int(v) for v in med), reason=reason)
    out.update(hit)
    # flood 的含义：**颜色层没帮上忙**（结论要么是靠底图层 / 差分层得到的，要么根本给不出）
    out["flood"] = (hit["layer"] in ("basemap", "diff"))
    return out


#: ⭐⭐ **连续几次"跳变被丢"就承认"旧锁已失效"**（用户 2026-10-06 ✓ 原话："**角色传送点是
#: 会突变的，你得认颜色**"✓ —— 传送本来就该跳 ✓，跳变保护**不能变成牢笼** ✗）。
#: 现场（`behavior.log` ✓）：`mmap_pick n=0 px=955/910/824 … 候选位置跳变 117/79/109 px
#: （阈值 40）—— 当噪声丢弃`，而 `pos` **一分钟都停在 198** ✗ ⇒ 旧锁是个**假点** ⇒ 真读数
#: **必然**每拍都"跳变" ✗ ⇒ 保护把**所有正确的读数**都丢了 ✓ ⇒ 一次也出不来 ✓。
#: ⇒ 连丢这么多次 ⇒ 不再"照抄上一拍" ✓，**直接接受新候选** ✓（宁可跳一下，也不再锁死 ✓）。
#: 3 次 ≈ 3 拍（≈100ms @30fps）—— 噪声通常不会连着 3 拍都冒出同一个远处的候选 ✓。
DOT_JUMP_GIVEUP_N = 3
#: 「被丢」计数的**时间窗**（秒 ✓）：窗内累计、超过它重新起算 ✓ ——
#: ⚠ 不能"接受就减一" ✗（第二版那么写 ⇒ 现场 `第 **1** 次` 一直重复 ✗：丢掉之后下一拍
#:   整幅搜又接受了**同一个假点** ⇒ 丢/接受交替 ⇒ 计数永远回到 0 ✓）。8 秒 ≈ 十来个决策拍 ✓。
DOT_JUMP_WINDOW_S = 8.0


def _shift_calib(calib, dx, dy):
    """把一份标定**平移到"裁出来的那一小块"**上（面板原点挪了 `(dx, dy)` ✓）。

    ⭐ 为什么必须有它（用户 2026-10-06 ✓ 帧率现场 ✓）：定位原来走"把面板**裁成一小块**再调
      `find_player_dot`"—— **0.24 ms** ✓，比整幅 **62 ms** 便宜 **250 倍** ✓
      （`tools/_bench_locate.py` ✓）；可那一趟**不传标定/地形** ✗ ⇒ 要几何的层全被跳过 ✗
      ⇒ 假锁甩不掉 ✓（正确性现场见 SKILL 285 ✓）。
      ⇒ 正解**不是**退回整幅（那要 62 ms ✗），而是**让裁块也带上正确的几何** ✓ ——
        标定里只有 `offset`（面板原点）与裁剪有关 ✓ ⇒ **平移它就够** ✓：
        `canvas = (panel − offset) / scale + view`，子图是 `panel' = panel − (dx, dy)`
        ⇒ `offset' = offset − (dx, dy)` ✓，`scale` / `view` 一个字不改 ✓。
    """
    if not calib:
        return calib
    try:
        ox, oy = (list(calib.get("offset") or (0, 0)) + [0, 0])[:2]
        return dict(calib, offset=[float(ox) - float(dx), float(oy) - float(dy)])
    except Exception:                       # noqa: BLE001 —— 平移不了就原样（宁可慢，别错 ✓）
        return calib


def _local_result_trustworthy(sub, rr, calib=None, terrain=None):
    """这一小块里找到的东西**可信吗** —— 不可信就别认它、回全画面 ✓。

    **为什么要这道闸**（用户 2026-10-06 ✓ 同一次现场）：裁块便宜 250 倍 ✓，可它只在**锁
      附近**看 ✓ —— 锁是假的时候，窗口里**总能**挑到一个像样的假块 ✗（现场就是沙上的色斑 ✓）
      ⇒ 认了它 ⇒ 永远回不到全画面 ✗（这正是老版锁死的方式 ✓）。
    判据（一处实现 ✓ 不新造数 ✓），**两道**：
      · **认色**：这一小块里有**玩家点真色**的像素（`dot_core_mask` ✓ 探针量到的
        `(108,255,255)` ✓ ⇒ G,R ≥ 240、B ≤ 160 ✓）⇒ 那找到的这块**得像核心色**才作数 ✓
        （沙斑 `(102,204,238)` 当场出局 ✓）；
      · **认几何**（色判不了时兜底 ✓ —— 别的图的青族标记就没核心色 ✓）：**人只能站在
        平台上** ✓ ⇒ 一个候选都不在平台上 ⇒ 不可信 ⇒ 回全画面 ✓
        （用 `_prefer_on_foothold` ✓ 一处实现 ✓；桩/假地形不判 ✓）。
    ⚠ 代价（如实 ✓）：锁是假的时候，这一拍会**多跑一次全画面**（62 ms ✓）—— 那正是"甩掉
      假锁"必需的一枪 ✓，而甩掉之后每拍都走便宜的裁块路 ✓。
    """
    try:
        if not int(dot_core_mask(sub).sum()):
            # **判不了色**（这一块里根本没有玩家点真色 ✓ —— 别的图的青族标记就是这样 ✓）
            # ⇒ 退一步用**几何**兜（⭐ 这条只有现在才做得到：裁块从 2026-10-06 起
            #   **带着平移后的标定与地形** ✓，所以"站在平台上"在裁块里也问得出来 ✓）：
            #   人**只能站在平台上** ✓ ⇒ 一个候选都不在平台上 ⇒ **这一拍不可信** ⇒ 回全画面 ✓。
            #   ⚠ 桩/假地形（没有 `canvas` ✓ = 真实地形那份才有 ✓）⇒ **不判** ✓（用例不受影响 ✓）。
            if (calib is None or terrain is None
                    or getattr(terrain, "canvas", None) is None):
                return True
            cands = rr.get("all") or [rr]
            return _prefer_on_foothold(cands, calib, terrain, sub) is not cands
        fam = rr.get("family")
        m = dot_mask(sub, fam if fam in ("yellow", "cyan") else None)
        return bool(_blob_looks_core(sub, m, rr))
    except Exception:                       # noqa: BLE001 —— 判不了就认它（别丢点 ✓）
        return True
