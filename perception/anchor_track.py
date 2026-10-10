# -*- coding: utf-8 -*-
"""锚定追踪（**anchor** ✓）—— **"一堆静止的同形物体里，只有一片在动"** 那一类场景的底盘。

═══════════════════════════════════════════════════════════════════════════════
## 它是什么 ✗（**与老两档的差别，先讲清楚**）

场景（用户 2026-10-08 ✓ 他口述的）：

  · 一片浅水河底，沉着一堆**形状完全相同**的透明玻璃片；
  · **只有一片**在"边顺时针旋转、边运动"；
  · 相机**很慢、而且只有平移**（不旋转）；
  · 水面**偶尔**起波浪，把画面临时搅一下。

⇒ 老两档（经典 / 运动分离 / 速度跟踪）都是在"**找出哪个是真的**"上打转 ✗（`score` / 认领 /
  融合 / 上板……）；这一档不吃那一套 ✓，它吃**三条几乎免费的结构性事实** ✓✓：

  ① **相机只要"平移"** ⇒ 全局位移＝一个二维矢量 ✓ ⇒ 用"**绝大多数静止物体的中位位移**"当相机
     （⚠ 与它们的"群体中位位移"同一个物理量 ✓ 纯平移下**无偏** ✓），再叠一层**相位相关主峰**
     当**可信度** ✓；
  ② **同形 + 静止那堆可以当"锚"** ✗✗ ⇒ 它们**长期位置**就是群体坐标系里的**基准** ✓
     ⇒ 逐帧累加相机位移那点漂移（每拍 ±0.2px 攒起来很吓人 ✓）**每拍拉回** ✓（= "锚定" ✓
     本档名字的来历 ✓）；
  ③ **只有一片在动** ⇒ 在群体坐标系里，**谁的位置在变、谁就是它** ✓（并且它**转** ⇒ 角度也在变 ✓
     双证据 ✓ —— 其余片位置和角度**都该纹丝不动** ✓）。

## 波浪怎么处理 ✗（用户 2026-10-08 ✓："**偶尔水面会有波浪干扰**" ✓）

三层，缺一不可（见 `anc_wave_score` / `_camera` / `process`）：

  · **认出来**：三个便宜指标 —— 相位相关主峰变弱 `resp` ✗、静止片位移的离散度 `mad` 变大 ✗、
    "与相机一致的那些片"占比 `frac` 掉下来 ✗（任一条爆 ⇒ 波浪分上去 ✓）；
  · **滑过去**：波浪段内**不再更新相机**（沿用上一拍的值 ✓）、**冻结静止档案与角度** ✓
    （拿被搅过的帧去立档 = 把波浪写进基准 ⇒ 事后全是冤枉 ✗）；
  · **回来**：波浪段有**最短持续**（`_ANC_WAVE_HOLD` ✓）⇒ 不会一帧抖就进出 ✓；过去之后
    靠"锚定校正"把累加量一次拉回基准 ✓。

## 与老底盘的关系 ✗

  · **只借壳**：`AnchorRunner` 只是 `MotionRunner` 换一个底盘（`_make_tracker` ✓）⇒ 演示窗那套
    （`warm_tick` / `draw` / 拖动 / 播放 ✓）**一行都不用改** ✓（与 `VelocityRunner` **同一个套路** ✓）；
  · **判据一条都不继承** ✗：没有 `score / persist / 认领 / 融合 / 上板 / 标准面积` 那一套 ✓；
    位置口径仍是"加工域"（与老两档**同一把尺** ✓ 演示窗按 `scale_x/scale_y` 画 ✓）。
═══════════════════════════════════════════════════════════════════════════════
"""

import math

import cv2
import numpy as np

#: 模式键（演示窗 `cmb_mode` 的 `currentData()` ✓ 与 `motion["mode"]` ✓ **同一处口径** ✓）。
ANCHOR_MODE = "anchor"
#: 模式名（界面显示 ✓）。
ANCHOR_NAME = "锚定追踪"

#: ⚠⚠ **两处"借老底盘的口径"** ✗✗（都是**同一件事就必须同一把尺** ✓ 各写一份一定会漂 ✓）：
#:   · `pick_white` = **白块认定**（用户 2026-10-05 ✓ "任何时候出现白色图形都需要认真目标" ✓✓
#:     ＋ 它那两道去抖门 `_WHITE_STREAK` / `_WHITE_STEP` ✓）；
#:   · `_POS_STEP_MAX` = **报出位置的每拍上限**（用户 2026-10-04 ✓）✓。
from perception.lie_motion import (pick_white as _pick_white,     # noqa: E402
                                   _blobs as _white_blobs,
                                   _WHITE_STREAK as _WHITE_STREAK,
                                   _WHITE_STEP as _WHITE_STEP,
                                   _POS_STEP_MAX as _POS_STEP_MAX)

# ---------------- 常量（都用注释写清来路 ✓ 别在代码里散落魔数 ✗）----------------

#: **配对门**（px ✓）：这一拍的框与上一拍那条账的预测位置差多少以内算"同一片" ✓。
_ANC_GATE = 0.35          # × 它自己的边长（片本身大概多大 ⇒ 门随尺寸走 ✓）
_ANC_GATE_MIN = 18.0      # 绝对下限（px ✓ 小片也不会配得太死 ✓）

#: **相机位移的平滑**（0~1 ✓ 越大越信本拍 ✓）—— 慢速平移 ⇒ 取小一点（滤波比测量准 ✓）。
_ANC_CAM_SMOOTH = 0.35
#: **相机位移每拍限幅**（px ✓）：慢速平移下，一步超过它就是被波浪/误配骗了 ✓ 直接削 ✓。
_ANC_CAM_STEP_MAX = 22.0
#: **相机"热身完了"的判据**：有这么多条账"跟拍 ≥3 拍"（`mv` 可信 ✓ 见 `_camera` ✓）⇒ 才算量到相机 ✓。
#:   ⚠⚠ 在这之前**选人整段停掉** ✗✗（用户 2026-10-08 ✓ "**一开始就认错了**" ✓）：相机还没量到 ⇒
#:     群体坐标里**每个框都像在动**（其实是相机在动 ✓）⇒ 谁先攒够分谁中 ⇒ **开局必错** ✓。
_ANC_CAM_LIVE_N = 3

#: **"静止"的判定**（群体坐标系里 ✓）：偏离自己的长期记录这么多 px 以内 ⇒ 还算它没动 ✓。
_ANC_STATIC_DEV = 5.0
#: 连续这么多拍都"没动" ⇒ 才给它立**静止档案**（用来当锚 ✓ 也用来筛"谁在动" ✓）。
_ANC_STATIC_N = 10
#: **锚定校正**每拍最多拉回多少 px ✓（防一次性把相机量掰断 ✓）。
_ANC_ANCHOR_MAX = 6.0

#: **波浪分**的三个判据门（见 `anc_wave_score` ✓）：
#:   · `resp_ok` = 相位相关主峰的"正常下限"（⚠ `cv2.phaseCorrelate` 回的那个响应值 ✓ 实测正常帧
#:     ~0.3~0.6 ✓ 被波浪搅时掉到 0.1 上下 ✓ —— 默认值就是照实测取的 ✓）；
#:   · `mad_ok` = 静止片位移（扣掉相机后）的离散度正常上限（px ✓）；
#:   · `frac_ok` = "与相机一致"的片占比正常下限 ✓。
_ANC_RESP_OK = 0.22
_ANC_MAD_OK = 3.0
_ANC_FRAC_OK = 0.70
#: **波浪分门**（≥ 它 ⇒ 这一拍算波浪段 ✓ 见 `process` ✓）。
_ANC_WAVE_T = 0.5
#: 波浪段**最短持续**（拍 ✓）：进来之后至少这么些拍才算"过去" ✓（防一帧抖就进出 ✓）。
_ANC_WAVE_HOLD = 5

#: ⭐⭐⭐⭐⭐ **"谁在动"那个量 = 一段时间里的"净位移"** ✗✗（用户 2026-10-08 ✓ 原话：
#:   "**圆满屏乱跳，看不出什么时间连续性**" ✓✓）——
#:   ⚠⚠⚠ **实测病根**（第一版就是这么写的 ✗）：老写法 `score += |本拍位移|` ⇒ **抖动片与真移动片
#:     攒得一样快** ✗✗（检出框抖一下也是几十像素 ✓）⇒ 15 个框里一堆在抖 ⇒ **谁都够 26 分** ⇒
#:     身份满屏乱换 ⇒ 圆跟着乱跳 ✓ = 用户看到的那一幕 ✓。
#:   ⇒ 现在量 `|g(t) − g(t − _ANC_NET_WIN)|`（**净位移** ✓）：抖来抖去 ⇒ 净走 ≈ 0 ✓✓；
#:     一直在走 ⇒ **线性增长** ✓（"时间连续性"就体现在这个"线性"上 ✓ 也正是用户要看的东西 ✓）。
_ANC_NET_WIN = 1.2
#: 净位移要攒到这么多 px 才有资格当目标 ✓。
_ANC_SCORE_MIN = 26.0
#: ⭐⭐⭐⭐⭐ **第一次锁定必须"明显领先"** ✗✗（用户 2026-10-08 ✓ 原话："**一开始就认错了**" ✓✓）——
#:   ⚠⚠⚠ **病根**（**实测** ✓）：`anc_pick` 那道"超过现任 ×`_ANC_SWITCH_K`"的迟滞**只在有现任时**
#:     起作用 ✗ ⇒ **第一次锁**退化成"**谁先攒够 26px 就是谁**" ✗✗ —— 而开局那几十拍里，一堆
#:     同款片（砖 / 星）各自在晃 ✓ 谁先够谁中 ⇒ **一开场就认错** ✓（实测真素材第一次锁在帧 71 ✓
#:     锁的是"先够分的那一格" ✗，不是那片真在动的 ✓）。
#:   ⇒ 第一次锁要求 **第一名 ≥ 第二名 × 本值**（没有第二名 ⇒ 直接算过 ✓）：只有"**明显在动的那一片**"
#:     才锁得上 ✓；一堆在晃 ⇒ **拉不开 ⇒ 不锁** ✓（**宁可不锁，也别认错** ✓ ← 这正是用户这句的字面 ✓）。
_ANC_FIRST_K = 1.5
#: **换人迟滞**：新候选要超过现任这么多倍才换 ✓（防来回跳 ✓）。
_ANC_SWITCH_K = 1.7
#: ⭐⭐⭐ **换人还要"连续领先"这么多拍才真换** ✗✗（**实测** ✓ 用户 2026-10-08 的素材上：
#:   只领先"一轮"就换 ⇒ 200 拍里换了 4 次、其中帧 161/164 **两拍来回跳** ✗✗）—— 单拍尖峰
#:   （检出框抖一下 / 认领那一下位置跳 ✓）不该改身份 ✓；⚠ 与"认领要连续"同一个套路 ✓。
_ANC_SWITCH_N = 3
#: ⭐⭐⭐⭐⭐ **白块认定** ✗✗（用户 2026-10-08 ✓ 原话："**圆一开始就乱锁，没有认白色图形**" ✓✓）——
#:   ⚠⚠⚠ 老两条底盘**都有**这条（`lie_motion.pick_white` ✓ 用户 2026-10-05 ✓ "**任何时候出现白色
#:   图形都需要认真目标**" ✓✓），**我新建这一档时漏了** ✗ ⇒ 开局只能靠"累积运动"慢慢攒 ⇒
#:   用户看到的就是"**一开始就乱锁**" ✓。⇒ 现在补上 ✓，且**口径一律复用** `pick_white` ✗
#:   （⚠ 别在别的模块里再写一份"白度阈值" ✗ —— 真目标**会逐渐透明** ⇒ 绝对白度一直掉 ✓
#:   只有它那套"相对 + 增益"的判法扛得住 ✓ 见 `lie_motion` 那段 ✓）。
#:   ⚠ 白块**认到的那一片**：身份**按住**这么多拍 ✓（刚认到就被人抢走 = 白认 ✓）。
_ANC_WHITE_HOLD = 8
#: 白块认定的**朝里吸附半径**（× 边长 ✓）：白块落在这片框内/边上 ⇒ 算"就是它" ✓。
_ANC_WHITE_SNAP = 0.65
#: ⭐⭐⭐⭐ **"持有者"多久不白才算作废**（拍 ✓）—— 用户 2026-10-08 ✓ 原话："**只认那个一直在的**"
#:   ✓✓：一旦认下某片白块，就**一直用它** ✓ 不许被画面里别的白片抢走 ✗（实测上一版在 #2 ↔ #23
#:   之间来回认 ✗）；⚠ 但它**连着这么多拍都不白**了（真白片会逐渐透明 ✓ / 飘出视野 ✓）⇒ 才放下 ✓
#:   ⇒ 那时才允许去认下一个 ✓（这样"一直在的那个"才有意义 ✓）。
_ANC_WHITE_GRACE = 4
#: ⭐⭐⭐ **"持有者那一问"的搜索半径**（× 边长 ✓ 见 `_white` ① ✓）—— ⚠⚠ **实测逼出来的** ✗✗：
#:   `pick_white` 的判据是"**门内取前 5% 最亮的**"＋"**这块得比周围亮 `min_gain`**" ✓ ⇒
#:   搜索窗**不能太小** ✗：我第一版用 0.65×边长（窗口 ≈ 1.3 个白块大 ✓）⇒ 白块占了窗口的
#:   **59%** ✗ ⇒ 窗口自己的中位**就是白块**（250 ✓）⇒ "比周围亮 12" 当场不成立 ✗ ⇒ **永远回
#:   `None`** ✗ ⇒ 持有者每 4 拍被放下一次 ⇒ 又去认下一个 ✓ = "来回认"的最后一层病根 ✓。
#:   ⇒ 取 **1.3×边长**：窗口 ≈ 2.6×2.6 个白块 ⇒ 白块占 ~15% ✓（>5% 才进得了分位 ✓、
#:   又不至于把窗口自己的中位顶上去 ✓）—— 实测这一问稳定回"还在" ✓。
_ANC_WHITE_SEEK = 1.3
#: ⭐⭐⭐⭐⭐ **白块认定还要"看它在不在动"** ✗✗（用户 2026-10-08 ✓ 原话："**一开始就认错了**" ✓✓）——
#:   ⚠⚠ **实测病根**：光认"白 + 够大 + 连着两拍" ✗ ⇒ 画面里那些**白色 UI / 高光**（又白又大又稳 ✓）
#:   一开场就被认走 ✗✗（真素材实测帧 127 认的 #2 就是那种 ✗）；而这一档的场景前提是
#:   "**只有那一片在边转边走**" ✓ ⇒ 白块**必须同时也在动** ✓ 才配当那个目标 ✓。
#:   ⚠ 用**短窗净位移**（`_ANC_WHITE_WIN` 秒 ✓ 比"攒到 26px"那扇窗短得多 ✓）⇒ 慢速移动也能凑得出来 ✓、
#:     静止的白东西（UI ✓）永远是 0 ✗。
_ANC_WHITE_WIN = 0.5
#: 白块那道"也在动"的门：`max(绝对下限, 比例 × 边长)`（px ✓ 窗内净位移 ✓）。
_ANC_WHITE_MOVE_MIN = 6.0
_ANC_WHITE_MOVE_K = 0.20
#: ⭐⭐⭐ **白块"也在动"那道门的实现：逐拍相对速度下限**（px/拍 ✓ 见 `_white_moving` ✓）——
#:   ⚠⚠ **实测两次**（一头一尾）：① 用**净位移**那套 ⇒ 在这段素材上目标动得慢 ✗ ⇒ **永远过不了门**
#:     ⇒ 白块认不上 ✗；② 用 `_ANC_WHITE_MOVE_MIN/4 = 1.5px/拍` ⇒ 还是太紧 ✗（用户场景是"**缓速移动**"
#:     ✓ 他自己那句话 ✓）⇒ 现在取 **0.6px/拍**：只要**不是一帧里完全不动**就算"在动" ✓ ——
#:     实测静止的 UI 白斑是 **0.00px/拍** ✓、真在挪的片 ≥ 0.3 ✓ ⇒ 这两类分得开 ✓✓。
_ANC_WHITE_V = 0.6
#: ⭐⭐⭐⭐⭐ **"白块得像一块形状"** ✗✗（用户 2026-10-08 ✓ 原话："**一开始你就错了，正中心的白块你都
#:   认不准，看起来你是在没有任何依据的瞎猜**" ✓✓）—— ⚠⚠⚠ **实测**（帧 61/121，加工域 ✓）：
#:     · 正中心那颗**白星** = `(445.0, 240.6) 85×74 均值 251` ✓ **就在候选表里、就在 ROI 正中** ✓；
#:     · 可 `pick_white` 咬的是 `(462.5, 64.0) 454×9 均值 252` ✗ = **一行标题文字**（又宽又扁 ✗，
#:       而且**在 ROI 之外** ✗）；帧 1~21 咬的也是 `(461.8,130) 246×9` ✗（另一行文字 ✓）。
#:   ⇒ 两处**实测出来的**差别，加两道门就够（不必猜别的）：
#:     ① **短边/长边 ≥ `_ANC_WHITE_ASP`**（文字条实测 9/246 = **0.04** ✗；白星 74/85 = **0.87** ✓）；
#:     ② **面积落在 `_ANC_WHITE_A_LO..A_HI`**（白星实测 **3779** ✓；文字条 2208~4086 也在这个带里 ✗
#:        ⇒ 所以真正把它判掉的是 ① 那条 ✓ —— 面积这两道只是顺手兜底 ✓）。
_ANC_WHITE_ASP = 0.35
_ANC_WHITE_A_LO = 600.0
_ANC_WHITE_A_HI = 40000.0
#: ⭐⭐⭐⭐⭐ **"按外观自己跟"那一套** ✗✗（用户 2026-10-08 ✓ 原话："**你看起来只是在认检出框，
#:   真目标大部分时候都是没有检出框的**" ✓✓）—— 这一句把观测源从"检出框"改成"**外观**" ✗：
#:   · **模板** = 目标那块灰度小图（尺寸 = 它那格框 ✓ 零均值 ✓ 与 `_ANC_ANG_*` 同一套 NCC ✓）；
#:   · 每拍在**群体坐标**里预测（相机已减掉 ✓）⇒ 再在预测点附近做**旋转 ＋ 平移**搜索 ✓
#:     ⇒ **一个检出框都不需要** ✓✓ 就能给出位置与角度 ✓；
#:   · ⚠ 检出框**只当辅助** ✗：有 ⇒ 顺手校对一下（见 `_patch` 里的 re-anchor ✓）；没有 ⇒ 照跟 ✓。
_ANC_PATCH_NCC = 0.45
#: 模板慢更新（EMA ✓）：目标会**逐渐透明 / 转向** ⇒ 模板要慢慢跟上 ✓（⚠ 快了会被误配带跑 ✗）。
_ANC_PATCH_A = 0.06
#: 搜索半径（× 目标边长 ✓）：慢速场景 0.35 够 ✓（大了又慢、又容易被旁边的同款片吸走 ✗）。
_ANC_PATCH_RAD = 0.35
#: 模板重取（re-anchor）时"框与预测差多少以内算同一块"（× 边长 ✓）。
_ANC_PATCH_SNAP = 0.5
#: **一拍照样最多挪多少**（× 目标边长 ✓）：同款片彼此相像 ⇒ 模板会"一步跳到隔壁那块" ✗
#:   （实测不管它 ⇒ 26 拍偏 290px ✗✗）⇒ 慢速场景这道理应很紧 ✓（一条上限就把它摁住 ✓）。
_ANC_PATCH_STEP = 0.15

#: **滑行**：丢了之后位置外推的每拍最大位移（px ✓ 慢速场景 ✓ 削掉野值 ✓）。
_ANC_COAST_V_MAX = 26.0
#: **认领门**（px ✓）：滑行期出现"无主框"且落在预测位置这么近 ⇒ 认回来 ✓（滑得越久预测越飘 ⇒
#:   比配对门宽 ✓）。
_ANC_CLAIM_GATE = 95.0
#: 丢多少拍之后开始允许认领 ✓（1 拍就认容易认到旁边的静止片 ✗）。
#: ⭐⭐⭐⭐⭐ **用户 2026-10-08 ✓ 原话："帧 278之后就在满屏幕乱窜 无法正常跟踪目标"** ✓✓ ——
#:   ⚠⚠⚠ **实测的真凶之一就是这里那个 2** ✗✗：真目标（"逐渐透明"那颗星 ✓）会**短暂看不见**
#:   （帧 **274~277 连续 4 拍**局部白度也是空的 ✓ 见 `_local_white` 那段数 ✓），而 `2` 一拍就允许
#:   **认领别人的框** ✗ ⇒ 位置被拽到 `(446,260) → (420,275) → (395,291) → …`（每拍 30px 直线乱窜 ✓）
#:   ⇒ 等目标再亮起来时，位置已经跑到别处 ⇒ 局部白度也找不回来了 ✗✗ = **那段"满屏乱窜"** ✓✓。
#:   ⇒ 短缺口**一律滑行** ✓（滑行是按**上一拍速度**外推 ✓ 实测那 4 拍只差几个像素：帧 280 目标在
#:   `(480.9, 251.9)`、滑行预测 ≈ `(482, 244)` ✓✓）—— 只有真丢够久（**半秒** ⇒ 30fps 下 15 拍 ✓）
#:   才允许去认领别的框 ✓。
_ANC_CLAIM_MISS = 15
#: ⭐⭐⭐⭐⭐ **"看不见就报丢"** ✗✗（用户 2026-10-08 ✓ 定案："**动**" ✓）—— 连着这么多拍**一个观测都没有**
#:   （外观：局部白度接力／模板匹配 ✓ 几何：压在我们预测位置上的框 ✓ 都没有 ✓）⇒ 状态标 `lost` ✓、
#:   **位置报 `None`** ✓（画面上不画 ✗）—— "**没依据就别声称目标在这儿**" ✓。
#:   ⚠ 与上面那把尺**同一个值**（半秒 ✓ 30fps 下 15 拍 ✓ 一处口径 ✓）。
_ANC_LOST_N = _ANC_CLAIM_MISS
#: ⭐ **报丢之后"找回"的半径**（群体坐标 px ✓ 用户 2026-10-09 ✓）—— 只在这半径内、**且净位移够门**
#:   （`_ANC_SCORE_MIN` ✓ = "真在走" ✓ 波纹免疫 ✓）的那一片才认回来 ✓。
_ANC_REACQ_R = 120.0
#: 找回也要**连续这么多拍**都是同一片 ✓（单拍尖峰不算 ✓ 与本档"要连续"的纪律一致 ✓）。
_ANC_REACQ_N = 3
#: ⭐⭐⭐⭐⭐ **"融合"那三个口径** ✗✗（用户 2026-10-09 ✓ "**直到能跟上真值为止**" ✓✓ 实测定的 ✓）：
#:   · `_ANC_FUSE_R` = **认框半径**（px ✓）：外观位置 25px 内那一块才吃 ✓ —— **实测**：目标框常贴
#:     **1~17px** ✓，而门一开大（30~45px）就会吃到"墙上并行漂的诱饵" ✗（误差 45~140px ✓）；
#:   · `_ANC_FUSE_SZ_LO/HI` = **尺寸带**（px ✓）：实测那些压在目标上的框是 **60~170px** ✓
#:     （比它小的多半是碎块 ✓、比它大的多半是融合块 ✓ 都不是目标 ✓）。
_ANC_FUSE_R = 25.0
_ANC_FUSE_SZ_LO = 55.0
_ANC_FUSE_SZ_HI = 190.0

#: **角度**：窗口 = `半径系数 × 片边长`（px ✓）；搜索 ±`_ANC_ANG_RANGE` 度、步长 `_ANC_ANG_STEP`
#: 度（慢速旋转 ⇒ 相邻帧变化很小 ⇒ 小范围搜既快又准 ✓）。
_ANC_ANG_WIN = 0.85
_ANC_ANG_RANGE = 6.0
_ANC_ANG_STEPD = 0.5
#: 角度更新的平滑（0~1 ✓）：慢转 ⇒ 取小一点抗噪 ✓。
_ANC_ANG_SMOOTH = 0.45
#: 相关度低于它 ⇒ 这块图**没有结构**（纯色 / 全糊 ✓）⇒ 这一拍**不给角度** ✓（不猜 ✗）。
_ANC_ANG_NCC = 0.30

#: `cv2.phaseCorrelate` 用的**下采样倍数**（1 = 原尺寸 ✓）—— 只为省时间 ✓ 慢平移下 2 足够 ✓。
_ANC_PSR_DS = 2


# ══════════════════════════════════════════════════════════════════════════
# 纯函数（**好钉** ✓ 判据全在这几个里 ✓ —— 类里只负责串流程 ✓）
# ══════════════════════════════════════════════════════════════════════════

def anc_median2(vs):
    """一串二维点 ⇒ **逐分量中位** ✓（空 ⇒ `(0.0, 0.0)` ✓）。**纯函数** ✓。"""
    _v = [(float(_p[0]), float(_p[1])) for _p in (vs or []) if _p is not None]
    if not _v:
        return (0.0, 0.0)
    return (float(np.median([_p[0] for _p in _v])),
            float(np.median([_p[1] for _p in _v])))


def anc_wave_score(resp, mad, frac, resp_ok=_ANC_RESP_OK, mad_ok=_ANC_MAD_OK,
                   frac_ok=_ANC_FRAC_OK):
    """**波浪分** ⇒ `0.0 ~ 1.0`（越大越像被搅过 ✓）。**纯函数** ✓。

    三个独立信号，**取最大**（任一条爆就算波浪 ✓ —— 三个都爆才认，会漏掉"只在某一路明显"的
    那种 ✓ 实测波浪里 `resp` 掉得最狠、`mad` 跟着涨 ✓）：

      · `resp` = 相位相关主峰响应 ✓（波浪把"全局唯一平移"这件事打散 ⇒ 主峰变弱 ✗）；
      · `mad` = 静止片（扣掉相机后）位移的离散度 ✓（波浪让本该一致的那堆各走各的 ✗）；
      · `frac` = "与相机一致（±3px ✓）"的片占比 ✓（同上，更直白 ✓）。

    ⚠ `resp ≤ 0`（没上一帧 / 相位相关失败 ✓）⇒ 那一路**不参与** ✗（别把"没测到"当成"波浪" ✗）。
    """
    _a = 0.0
    if float(resp) > 0.0:
        _a = max(0.0, 1.0 - float(resp) / max(1e-6, float(resp_ok)))
    _b = max(0.0, (float(mad) - float(mad_ok)) / max(1e-6, 2.0 * float(mad_ok)))
    _c = max(0.0, (float(frac_ok) - float(frac)) / max(1e-6, float(frac_ok)))
    return float(min(1.0, max(_a, _b, _c)))


def anc_coast(pos, vel, k=1.0):
    """滑行：`位置 ＋ 速度 × k`（**每拍位移限幅** ✓ 削野值 ✓）。**纯函数** ✓。"""
    _k = float(k)
    _vx = max(-_ANC_COAST_V_MAX, min(_ANC_COAST_V_MAX, float(vel[0])))
    _vy = max(-_ANC_COAST_V_MAX, min(_ANC_COAST_V_MAX, float(vel[1])))
    return (float(pos[0]) + _vx * _k, float(pos[1]) + _vy * _k)


def anc_pick(scores, cur=None, k=_ANC_SWITCH_K, _kf=None):
    """**谁是那个"在动的"** ⇒ 账的 id（`None` = 一个都不够格 ✓）。**纯函数** ✓。

    · 分数最高的那条 ⇒ 候选 ✓；⚠ 要 ≥ `_ANC_SCORE_MIN` 才算"真的在动" ✓；
    · ⭐ **没有现任（第一次锁）⇒ 还要"明显领先第二名"** ✓（`_ANC_FIRST_K` ✓ 见那段说明 ✓）；
    · **迟滞**：现任还在（分数 × `k` 仍不低于候选 ✓）⇒ **不换** ✗（防两片分数此起彼伏时来回跳 ✓）。
    """
    _ok = [(float(_s), _i) for _i, _s in (scores or {}).items()]
    if not _ok:
        return None
    _ok.sort(reverse=True)
    _top, _tid = _ok[0]
    if _top < float(_ANC_SCORE_MIN):
        return None
    #   ⚠⚠⚠ **第一次锁定必须"明显领先"** ✗✗（用户 2026-10-08 ✓ "**一开始就认错了**" ✓✓）——
    #     没有现任时，**不许"谁先够分谁中"** ✗（实测那正是开局认错的原因 ✓）；要拉开
    #     `_ANC_FIRST_K` 倍才行 ✓ ⇒ 一堆同款片各晃各的 ⇒ **拉不开 ⇒ 不锁** ✓（宁可不锁 ✓）。
    #     ⚠ 只有"**第一名 vs 第二名**" ✗ 不看第三名起（那堆噪声不参与 ✓）。
    if cur is None:
        if len(_ok) > 1 and _top < float(_ANC_FIRST_K if _kf is None else _kf) * _ok[1][0]:
            return None
        return _tid
    if cur in (scores or {}):
        if float(scores[cur]) * float(k) >= _top:
            return cur
    return _tid


def anc_white_cands(gray, roi=None, pos=None, radius=None, asp=_ANC_WHITE_ASP,
                    area_lo=_ANC_WHITE_A_LO, area_hi=_ANC_WHITE_A_HI):
    """**白块候选**（⇒ `[(cx, cy, area, mean)]` ✓ 按均值降序 ✓）—— **阈值口径一律复用
    `pick_white`** ✗（连它的掩膜都拿过来用 ✓ ⇒ 不另写一套白度判据 ✓ 见那两段说明 ✓）。**纯函数** ✓。

    ⭐⭐⭐⭐⭐ 用户 2026-10-08 ✓ 原话："**一开始你就错了，正中心的白块你都认不准，看起来你是在没有
      任何依据的瞎猜**" ✓✓ —— ⚠⚠ **实测**（`10月7日.mp4` 帧 61/121 ✓）：`pick_white` 咬的是
      `(462.5, 64.0) 454×9`（**一行标题文字** ✗ 且在 ROI 外 ✗）、而**正中心那颗白星**
      `(445.0, 240.6) 85×74 均值 251` ✓ **本来就在候选表里** ✓ —— 只是被文字条按"均值最亮"排前面 ✗。
    ⇒ 本函数干两件事（都有实测依据 ✓ 见 `_ANC_WHITE_ASP` 那段 ✓）：
      ① 把 `pick_white` 的掩膜切开成一块块（`_blobs` ✓ 同一套连通域口径 ✓）⇒ 逐块量均值 ✓；
      ② **判掉"又宽又扁的细长条"**（短边/长边 < `asp` ✓ 实测文字条 9/246 = 0.04 ✗、白星 74/85 = 0.87 ✓）
         ＋ 面积带 ✓ ＋ **中心落在 `roi` 里**（给了才筛 ✓）。
    ⚠ `pos` / `radius` 给了 ⇒ 只看那个方圆之内 ✓（"持有者还在不在"那一问用 ✓ 见 `_is_white` ✓）。
    ⚠ 拿不到掩膜 / 一块都不合格 ⇒ 回**空表** ✓（**不猜** ✗ —— 上层就当"这一拍没有白块" ✓）。
    """
    if gray is None:
        return []
    try:
        #   ⚠⚠⚠ **`pos` / `radius` 必须透传给 `pick_white`** ✗✗（**实测踩到** ✓）：它的阈值是
        #     "**门内**取前 `100-q`% 最亮的" ✓ ⇒ 不过去 ⇒ 阈值被**画面里更亮的那一块**顶高 ✗
        #     ⇒ 手里这块（暗一点）**整块掉出掩膜** ✗ ⇒ "持有者还在不在"那一问**永远回否** ✗
        #     ⇒ 认下的白块每 4 拍被放下、转头认另一个 ✓ = 用户看到的「**来回认**」✓✓
        #     （实测：两块 80×80、亮度 245 / 250 交替 ⇒ 候选表每拍只剩亮的那一块 ✗）。
        if pos is not None and radius is not None and float(radius) > 1.0:
            _r = _pick_white(gray, pos=(float(pos[0]), float(pos[1])),
                             gate=float(radius), want_mask=True)
        else:
            _r = _pick_white(gray, want_mask=True)
        _mask = _r[1]                      # ⚠ 只要掩膜（见下面那句 ✓）
    except Exception:                      # noqa: BLE001
        return []
    #   ⚠ 只要掩膜 ✓ 不看 `_res` ✗：`_res` 是 `pick_white` 自己按"均值最亮"挑的那一块（可能就是
    #     要被判掉的那条文字 ✗），而**候选表是从掩膜重新切的** ✓ ⇒ 它 `None` 不代表没有候选 ✓。
    if _mask is None:
        return []
    try:
        _bl, _ = _white_blobs(_mask, float(area_lo), float(area_hi))
    except Exception:                      # noqa: BLE001
        return []
    _g = np.asarray(gray)
    _out = []
    for _b in _bl:
        _cx, _cy, _a = float(_b[0]), float(_b[1]), float(_b[2])
        _x0, _y0, _w0, _h0 = int(_b[5]), int(_b[6]), int(_b[7]), int(_b[8])
        if _w0 <= 0 or _h0 <= 0:
            continue
        if min(_w0, _h0) < float(asp) * float(max(_w0, _h0)):
            continue                       # ① 细长条（文字 / UI 线）⇒ 不是"一块形状" ✓ 出局 ✓
        if roi is not None and not (float(roi[0]) <= _cx <= float(roi[2])
                                   and float(roi[1]) <= _cy <= float(roi[3])):
            continue                       # ② 不在框选区域里 ⇒ 出局 ✓（实测标题条就在 ROI 外 ✓）
        if pos is not None and radius is not None and float(radius) > 0.0:
            if math.hypot(_cx - float(pos[0]), _cy - float(pos[1])) > float(radius):
                continue                   # ③ "就在这一带找"（持有者那一问 ✓）
        _sub = _mask[_y0:_y0 + _h0, _x0:_x0 + _w0] > 0
        _m = (float(_g[_y0:_y0 + _h0, _x0:_x0 + _w0][_sub].mean())
              if bool(_sub.any()) else 0.0)
        _out.append((_cx, _cy, _a, _m))
    _out.sort(key=lambda _r: (-float(_r[3]), -float(_r[2])))
    return _out


#: ⭐⭐⭐⭐⭐ **"局部白度接力"那一带的半径** ✗✗（用户 2026-10-08 ✓ 原话："**看起来你的算法完全无法
#:   跟踪目标 ； 帧 278之后就在满屏幕乱窜 无法正常跟踪目标**" ✓✓）—— 取值 `1.6 × 目标边长`
#:   （**实测**：88px 的目标、半径 140px ✓ 见 `_local_white` 里那段数 ✓）。
_ANC_WHITE_LOCAL_R = 1.6


def anc_patch_match(gray, tpl, cx, cy, rad, rng=_ANC_ANG_RANGE, step=_ANC_ANG_STEPD):
    """**在 `(cx, cy)` 附近、`±rng` 度里，找模板最像的位置** ⇒ `(score, dx, dy, dtheta)` ✓（**纯函数** ✓）。

    ⭐⭐⭐⭐⭐ 用户 2026-10-08 ✓ 原话："**你看起来只是在认检出框，真目标大部分时候都是没有检出框的**"
      ✓✓ —— ⚠⚠⚠ 这一句把这一档的**观测源**从"检出框"彻底改成"**外观**" ✗：
        检出框**有没有都行** ✓（它有 ⇒ 顺手拿来校对一下 ✓；没有 ⇒ **照样跟** ✓✓）；
        真目标**大部分时间没有框**（透明 / 变淡 / 出框 ✓）⇒ 只在检出框里挑 **从根上就不成立** ✗✗。

    口径 = 与 `anc_patch_dtheta` **同一套**（旋转搜索 ＋ 圆形掩膜 ＋ 零均值 NCC ✓）——
      · **平移**用 `cv2.matchTemplate(..., TM_CCOEFF_NORMED)` ✓（它在窗口里一次算完 ✓ 比逐点 NCC 快
        两个量级 ✓）；
      · **角度**在 `±rng` 里扫（默认 ±6°、0.5° 一步 ✓ —— 用户场景是"**缓速**"旋转 ✓ 每帧差得很少 ✓）。
    ⚠ 找不到像的（纯色 / 全糊 / 全被挡 ✓）⇒ 回 `(0.0, 0, 0, 0)` ✓（**不猜** ✗ —— 上层拿这个当"这一拍
      没有观测" ✓ 走滑行 ✓）。
    """
    _g = np.asarray(gray, dtype=np.float32)
    _t = np.asarray(tpl, dtype=np.float32)
    if _g.ndim != 2 or _t.ndim != 2 or min(_t.shape) < 8:
        return (0.0, 0.0, 0.0, 0.0)
    _th, _tw = _t.shape
    _h, _w = _g.shape
    #   窗口 = 以 (cx, cy) 为中心、半径 rad ＋ 半个模板 ⇒ 再裁到画面内 ✓
    _x0 = int(math.floor(float(cx) - float(rad) - _tw / 2.0))
    _y0 = int(math.floor(float(cy) - float(rad) - _th / 2.0))
    _x1 = int(math.ceil(float(cx) + float(rad) + _tw / 2.0))
    _y1 = int(math.ceil(float(cy) + float(rad) + _th / 2.0))
    _x0, _y0 = max(0, _x0), max(0, _y0)
    _x1, _y1 = min(_w, _x1), min(_h, _y1)
    if _x1 - _x0 < _tw + 2 or _y1 - _y0 < _th + 2:
        return (0.0, 0.0, 0.0, 0.0)
    _win = _g[_y0:_y1, _x0:_x1]
    _best = (0.0, 0.0, 0.0, 0.0)
    _yy, _xx = np.ogrid[:_th, :_tw]
    _mask = (((_xx - (_tw - 1) / 2.0) ** 2 + (_yy - (_th - 1) / 2.0) ** 2)
             <= (0.48 * min(_th, _tw)) ** 2)
    for _d in np.arange(-float(rng), float(rng) + 1e-9, float(step)):
        _M = cv2.getRotationMatrix2D(((_tw - 1) / 2.0, (_th - 1) / 2.0), float(_d), 1.0)
        _tr = cv2.warpAffine(_t, _M, (_tw, _th), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        _tv = _tr[_mask]
        _tv = _tv - float(_tv.mean())
        _tn = float(np.sqrt(float((_tv ** 2).sum())))
        if _tn < 1e-6:
            continue
        if _tv.size < 40:
            continue
        try:
            _r = cv2.matchTemplate(_win, (_tr - float(_tr.mean())), cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        if _r.size == 0 or not np.isfinite(_r).any():
            continue
        _r = np.nan_to_num(_r, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _mv, _ml = float(_r.max()), cv2.minMaxLoc(_r)[3]
        if _mv > _best[0]:
            _best = (_mv,
                     float(_x0 + _ml[0] + (_tw - 1) / 2.0) - float(cx),
                     float(_y0 + _ml[1] + (_th - 1) / 2.0) - float(cy),
                     float(_d))
    return _best


def anc_patch_dtheta(prev_g, cur_g, rng=_ANC_ANG_RANGE, step=_ANC_ANG_STEPD):
    """两块**同心**的灰度小图 ⇒ **相对旋转角**（度 ✓ 正 = OpenCV 的正方向 ✓）。**纯函数** ✓。

    做法 = **旋转搜索 + 圆形掩膜 NCC** ✓（**特意不用** log-polar 相位相关 ✗：那个对边界、
    尺度、窗口大小都敏感 ⇒ 实测在小窗口上不稳 ✗；而"慢速旋转"意味着**每帧只差几度** ✓
    ⇒ 在 ±`rng` 度里扫一遍既便宜又可控 ✓）。

    ⚠ **先要"有结构"** ✗：透明片上如果这一段是纯色 / 全糊 ⇒ 任何角度都差不多 ⇒ 返回 `None` ✓
      （**不猜一个角度出来** ✗）；判据 = 最佳 NCC 要 ≥ `_ANC_ANG_NCC` ✓。
    """
    _a = np.asarray(prev_g, dtype=np.float32)
    _b = np.asarray(cur_g, dtype=np.float32)
    if _a.shape != _b.shape or _a.ndim != 2 or min(_a.shape) < 8:
        return None
    _h, _w = _b.shape
    _yy, _xx = np.ogrid[:_h, :_w]
    _m = (((_xx - (_w - 1) / 2.0) ** 2 + (_yy - (_h - 1) / 2.0) ** 2)
          <= (0.46 * min(_h, _w)) ** 2)
    _bv = _b[_m]
    if _bv.size < 40 or float(_bv.std()) < 1e-3:
        return None
    _bv = _bv - float(_bv.mean())
    _bn = float(np.sqrt(float((_bv ** 2).sum())))
    if _bn < 1e-6:
        return None
    _best, _bd, _bs = None, -2.0, -2.0
    for _d in np.arange(-float(rng), float(rng) + 1e-9, float(step)):
        _M = cv2.getRotationMatrix2D(((_w - 1) / 2.0, (_h - 1) / 2.0), float(_d), 1.0)
        _r = cv2.warpAffine(_a, _M, (_w, _h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        _rv = _r[_m]
        _rn = float(np.sqrt(float(((_rv - _rv.mean()) ** 2).sum())))
        if _rn < 1e-6:
            continue
        _c = float(((_rv - _rv.mean()) * _bv).sum()) / (_rn * _bn)
        if _c > _bd:
            _bd, _bs, _best = _c, _c, float(_d)
    if _best is None or _bs < float(_ANC_ANG_NCC):
        return None
    return _best


def anc_angle_line(c, theta_deg, half):
    """角度那条线（画面上用 ✓）：过圆心、沿 `theta` 方向的**两个端点** ✓。**纯函数** ✓。

    ⚠ 演示窗按 `scale_x/scale_y` 缩放之后才画 ✓（本函数只给**加工域**的两个点 ✓）。
    ⚠ 坐标轴 = 图像坐标系（**y 朝下** ✓）⇒ 内部按 `(cos, sin)` 直接用 ✓（与
      `cv2.getRotationMatrix2D` 的正方向**一致** ✓ —— 这样"角度数字"和"画出来的线"
      **必然同一个口径** ✓ 实测过 ✓）。
    """
    _t = math.radians(float(theta_deg))
    _dx, _dy = math.cos(_t) * float(half), math.sin(_t) * float(half)
    return ((float(c[0]) - _dx, float(c[1]) - _dy),
            (float(c[0]) + _dx, float(c[1]) + _dy))


# ══════════════════════════════════════════════════════════════════════════
# 底盘
# ══════════════════════════════════════════════════════════════════════════

class AnchorTracker:
    """**锚定追踪**的底盘（接口对齐 `VelocityTracker` ✓ ⇒ `AnchorRunner` 只换 `_make_tracker` ✓）。

    `process(...)` 一拍走完：配对 → 相机位移（中位 ＋ 相位相关主峰）→ 波浪分 → 群体坐标
    （＋锚定校正）→ 选"在动的那片" → 观测/滑行/认领 → 角度 → 出口。
    `motion_viz(...)` 只报"画面上要看的东西" ✓（**判据全在 `process` 里** ✓ 别把判据搬进 viz ✗）。
    """

    def __init__(self, mode=ANCHOR_MODE, roi=None, cam_smooth=_ANC_CAM_SMOOTH,
                 cam_step_max=_ANC_CAM_STEP_MAX, wave_t=_ANC_WAVE_T,
                 static_dev=_ANC_STATIC_DEV, score_min=_ANC_SCORE_MIN, **kw):
        #   ⚠ 收下多余关键字（`MotionRunner` 会把 `mode` 塞进来 ✓）**不报错** ✓：
        #     与老底盘同一个写法 ✓（多给一个键就崩 ⇒ 演示窗点"应用"当场闪退 ✓ 踩过 ✓）。
        del kw
        self.mode = str(mode or ANCHOR_MODE)
        self.roi = (None if not roi else
                    (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])))
        self.cam_smooth = float(cam_smooth)
        self.cam_step_max = float(cam_step_max)
        self.wave_t = float(wave_t)
        self.static_dev = float(static_dev)
        self.score_min = float(score_min)
        #: 画面尺寸（演示窗每拍刷 ✓ 与老底盘同一个用法 ✓）—— 只用来判"这格框在不在画面里" ✓。
        self.frame_wh = None
        #: ⭐ **老壳要的那两个"相机量"** ✗（`MotionRunner.step` 里读它们算"黄箭头 / 橙线"✓ 见
        #:   `lie_motion` 那段 ✓）—— 本档**同一个物理量**：`cam_mv` = 这一拍相机走了多少 ✓
        #:   ⇒ 照实给出去 ✓（⚠ 老两档用"群体中位位移"✓ 那是**同一件事** ✓ 慢速纯平移下无偏 ✓）。
        self._med_ema = (0.0, 0.0)
        self._median_mv = (0.0, 0.0)
        self._n = 0
        self._t = 0.0
        self._ts_prev = None

        # ---- 账本（每条 = 一片玻璃 ✓）----
        self._objs = {}                    # id → 账（见 `_new_obj` ✓）
        self._next_id = 1

        # ---- 相机（**本档的核心** ✓）----
        self.cam_cum = (0.0, 0.0)          # 累计位移（**群体坐标系的原点** ✓）
        self.cam_mv = (0.0, 0.0)           # 这一拍估出来的位移（平滑后 ✓）
        self._gray_prev = None             # 上一拍灰度（相位相关用 ✓）

        # ---- 波浪 ----
        self.wave = 0.0                    # 这一拍的分（0~1 ✓）
        self.wave_now = False              # 这一拍算不算波浪段（含最短持续 ✓）
        self._wave_hold = 0
        self.wave_n = 0                    # 累计"波浪段"拍数（验收看它 ✓）

        # ---- 目标 ----
        self.tid = None
        self._cand = None                  # 换人候选（要**连续领先**才真换 ✓ 见 `_target` ✓）
        self._cand_n = 0
        #: ⭐ **白块认定那套状态**（用户 2026-10-08 ✓ 见 `_white` ✓）：`_white_pt` = 上一拍白块在哪
        #:   （用来判"相邻两拍挪了没有" ✓）、`_white_n` = 连着几拍 ✓、`_white_hold` = 认到之后
        #:   **按住**多少拍不许换人 ✓。
        self._white_pt = None
        self._white_n = 0
        self._white_hold = 0
        #: ⭐⭐ **相机量到了没有**（`_ANC_CAM_LIVE_N` ✓ 见 `_camera` ✓）—— 没量到之前：**不选人、不认白块**
        #:   ✗（那几拍群体坐标是脏的 ✓ 见 `_camera` 里那段 ✓）。
        self._cam_live = False
        #: ⭐⭐⭐⭐⭐ **"按外观跟"那套的状态**（用户 2026-10-08 ✓ "真目标大部分时候都是没有检出框的"
        #:   ✓✓ 见 `_patch` ✓）：`_tpl` = 目标模板（float32 灰度 ✓）、`_tpl_wh` = 模板尺寸、
        #:   `_tpl_score` = 这一拍的匹配分（诊断/画面上用 ✓）。⚠ 它们是**这一档真正的观测源** ✗
        #:   —— 检出框只是辅助 ✓（有就用一下、没有照跟 ✓）。
        self._tpl = None
        self._tpl_wh = None
        self._tpl_score = 0.0
        #: ⭐⭐⭐⭐⭐ **这一拍的"局部白度接力"观测** ✗✗（用户 2026-10-08 ✓ "**帧 278之后就在满屏幕
        #:   乱窜 无法正常跟踪目标**" ✓✓）—— 它是**画面坐标**（`(x, y)` ✓）或 `None` ✓；
        #:   `_target` 靠它压住"改用它人 / 认领别人的框"（见 `_local_white` ＋ `_target` 那两段 ✓）。
        self._obs_local = None
        #: ⭐ **"这一拍有没有真观测"** ✗（用户 2026-10-08 ✓ "**350+帧之后就完全跟丢了，这时候真目标
        #:   检出框都还在**" ✓✓）—— 由局部白度接力（`_local_white` ✓）或模板匹配（`_patch` ✓）点亮 ✓；
        #:   只有它亮着才允许 `_bind_box` 去"认那块压在目标上的框" ✓（纯滑行的位置不许去认框 ✗）。
        self._obs_now = False
        #: ⭐ **"上一拍有没有真观测"** ✗（同上 ✓）—— `_target` 跑在 `_patch` **之前** ✗ ⇒ 它只能看
        #:   上一拍的观测结论 ✓（`_patch` 是这一档最主要的观测源 ⇒ 少看这一眼 ⇒ 挑框器会在"明明
        #:   跟得好好的"那些拍上乱换身份 ✓ 实测帧 353/369/421 ✓）。
        self._obs_prev = False
        #: ⭐ **"连着多少拍没有外观观测了"** ✗（同上 ✓）—— ⚠ 门要量**它**，不是"目标那条框缺了多少拍"
        #:   ✗（**实测**：那份素材上目标框**大半时间不在** ✓ ⇒ 量框缺口 ⇒ 门老是到点 ⇒ 挑框器乱换人 ✓）。
        self._obs_gap = 0
        #: ⭐ **这一拍"借了几何"的那块框** ✗（用户 2026-10-08 ✓ 定案："**动**" ✓）—— **画面坐标**
        #:   `(cx, cy, w, h)` ✓ 或 `None` ✓；`target_box` 报的就是它 ✓（框只是"目标在哪"的测量 ✓
        #:   不是身份 ✗ 见 `_box_on_pred` ✓）。
        self._obs_box = None
        #: ⚠ **上一拍借过那一格的号** ✓ —— **只当"优先续用"的提示** ✗（墙上有两块叠着的框 ⇒ 每拍
        #:   取最近那块会让位置抖 ✓ 实测帧 157 ✓）；**不是身份** ✗（身份永远是 `self.tid` ✓ 不再换 ✓）。
        self._obs_bid = None
        #: ⭐ **"丢后找回"那一片的号 ＋ 连着几拍了** ✓（见 `_reacq` ✓ 只在 `lost` 里用 ✓）。
        self._reacq_bid = None
        self._reacq_n = 0
        #: ⭐⭐ **"持有者"**（= "那个一直在的白块" ✓ 用户 2026-10-08 ✓ 见 `_white` ① ✓）：
        #:   `_white_tid` = 认下来的那一片 ✓／`_white_seen` = 它最近一次"还白着"的拍号 ✓
        #:   （超过 `_ANC_WHITE_GRACE` 拍不白 ⇒ 作废 ✓）；`_white_cand*` = 全局候选的连拍计数 ✓。
        self._white_tid = None
        #: ⭐ **"首选号"**（= 认过的**第一个**白块 ✓ 永不改 ✓ 见 `_white` ③）—— "只认那个一直在的"
        #:   就落在这一个字段上 ✓（它只要还白 ⇒ 永远优先 ✓ 不许被画面里别的白片抢 ✗）。
        self._white_pref = None
        self._white_seen = -10 ** 9
        self._white_cand = None
        self._white_cand_n = 0
        #: **上一拍报出去的位置**（画面坐标 ✓）—— 只给"每拍限速"用 ✓（见 `process` ✓）。
        self._pos_prev_report = None
        self.pos = None                    # 报出位置（**画面坐标** = 群体位置 ＋ `cam_cum` ✓）
        self.pos_g = None                  # 群体坐标系里的位置（**判据都在这儿** ✓）
        self.vel = None
        self.miss = 0
        self.theta = 0.0                   # 累计角度（度 ✓）
        self.state = "init"
        self.log_text = None               # 这一拍的事件那一行（演示窗底部日志区 ✓）

        # ---- 诊断 ----
        self._patch_prev = None            # 上一拍的"目标窗口"（灰度 ✓ 算角度用 ✓）
        self.box_v = []                    # 画面账本（**逐位对齐老布局** ✓ 演示窗按同一索引读 ✓）
        self.rad = 0.0
        self.dups = []

    # ---------------- 账 ----------------

    @staticmethod
    def _new_obj(_cx, _cy, _w, _h, _cf, _i):
        return {"id": int(_i), "c": (float(_cx), float(_cy)), "wh": (float(_w), float(_h)),
                "conf": float(_cf), "miss": 0, "hits": 1, "mv": (0.0, 0.0),
                "g": (float(_cx), float(_cy)), "g_prev": None, "g_rec": None,
                "static_n": 0, "static": False, "score": 0.0,
                #   ⭐ **净位移那扇窗**（`(时刻, 群体坐标)` ✓ 见 `_target` ② ✓）—— 新账从空开始 ✓。
                "ghist": []}

    def _boxes(self, dets):
        """检出框 ⇒ `[(cx, cy, w, h, conf)]`（**加工域** ✓ ROI 之外的一律丢 ✓）。"""
        _out = []
        for _d in (dets or []):
            if _d is None or len(_d) < 5:
                continue
            _cx, _cy = float(_d[1]), float(_d[2])
            _w, _h = float(_d[3]), float(_d[4])
            _cf = float(_d[5]) if len(_d) > 5 else 0.0
            if self.roi is not None:
                _r = self.roi
                if not (_r[0] <= _cx <= _r[2] and _r[1] <= _cy <= _r[3]):
                    continue
            _out.append((_cx, _cy, _w, _h, _cf))
        return _out

    def _match(self, bs):
        """配对（**贪心最近 ✓ + 位置门** ✓）⇒ 更新账、回 `[接到哪条账 or None]` ✓。

        ⚠ 门**随尺寸走**（`_ANC_GATE × 边长` ✓）—— 小片用小门、大片用大门 ✓（一刀切的门
          在小片上会把邻居配进来 ✗ 实测那样最脏 ✓）。
        ⚠ 一条账**一拍只接一格框** ✓（贪心先生成候选、按距离升序发 ✓）。
        """
        _cands = []
        for _k, (_cx, _cy, _w, _h, _cf) in enumerate(bs):
            for _i, _o in self._objs.items():
                _gate = max(_ANC_GATE_MIN, _ANC_GATE * max(float(_o["wh"][0]),
                                                           float(_o["wh"][1])))
                _dd = math.hypot(_cx - float(_o["c"][0]), _cy - float(_o["c"][1]))
                if _dd <= _gate:
                    _cands.append((_dd, _i, _k))
        _cands.sort()
        _used_o, _used_b = set(), set()
        _got = [None] * len(bs)
        for _dd, _i, _k in _cands:
            if _i in _used_o or _k in _used_b:
                continue
            _used_o.add(_i)
            _used_b.add(_k)
            _got[_k] = _i
        #   ① 接上的：更新位置 / 位移（**画面坐标** ✓ 相机那一份等会儿减 ✓）
        for _k, (_cx, _cy, _w, _h, _cf) in enumerate(bs):
            _i = _got[_k]
            if _i is None:
                continue
            _o = self._objs[_i]
            _o["mv"] = (float(_cx) - float(_o["c"][0]), float(_cy) - float(_o["c"][1]))
            _o["c"] = (float(_cx), float(_cy))
            _o["wh"] = (float(_w), float(_h))
            _o["conf"] = float(_cf)
            _o["miss"] = 0
            _o["hits"] = int(_o["hits"]) + 1
        #   ② 没接上的：老账 `miss+1`（位移记 0 ✓ 不进中位 ✓）
        for _i, _o in self._objs.items():
            if _i in _used_o:
                continue
            _o["miss"] = int(_o["miss"]) + 1
            _o["mv"] = (0.0, 0.0)
        #   ③ 新框（没配上任何账 ✓）⇒ 生一条新账 ✓（⚠ **静止片也照样立账** ✗：它们就是"锚" ✓）
        for _k, (_cx, _cy, _w, _h, _cf) in enumerate(bs):
            if _got[_k] is not None:
                continue
            _i = int(self._next_id)
            self._next_id += 1
            self._objs[_i] = self._new_obj(_cx, _cy, _w, _h, _cf, _i)
        return _got

    # ---------------- 相机 ----------------

    def _psr(self, gray):
        """`cv2.phaseCorrelate` ⇒ `(位移 ✓ 加工域, 主峰响应 ✓)`（拿不到 ⇒ `((0,0), 0.0)` ✓）。"""
        if gray is None or self._gray_prev is None:
            return (0.0, 0.0), 0.0
        try:
            if _ANC_PSR_DS > 1:
                _a = cv2.resize(self._gray_prev, None, fx=1.0 / _ANC_PSR_DS,
                                fy=1.0 / _ANC_PSR_DS, interpolation=cv2.INTER_AREA)
                _b = cv2.resize(gray, None, fx=1.0 / _ANC_PSR_DS, fy=1.0 / _ANC_PSR_DS,
                                interpolation=cv2.INTER_AREA)
            else:
                _a, _b = self._gray_prev, gray
            if _a.shape != _b.shape or min(_a.shape) < 16:
                return (0.0, 0.0), 0.0
            _w = cv2.createHanningWindow((_b.shape[1], _b.shape[0]), cv2.CV_32F)
            (_dx, _dy), _resp = cv2.phaseCorrelate(np.float32(_a), np.float32(_b), _w)
            return ((float(_dx) * _ANC_PSR_DS, float(_dy) * _ANC_PSR_DS),
                    float(_resp))
        except cv2.error:
            return (0.0, 0.0), 0.0

    def _camera(self, gray):
        """这一拍的相机位移 ＋ 波浪分 ＋ **锚定校正**（改 `self.cam_cum` ✓）。

        ⚠ 顺序**不能换** ✗：先算"静止片的中位位移"（它们的位移**就是**相机 ✓）⇒ 再拿
          "静止片在自己档案上的偏差"把累加量**拉回去** ✓（= 锚定 ✓ 治漂移 ✓）。
        ⚠ 波浪段内**一个字都不动** ✓（沿用上一拍 ✓ 见模块头"波浪怎么处理" ✓）。
        """
        _moves = [_o["mv"] for _o in self._objs.values()
                  if int(_o["miss"]) == 0 and int(_o["hits"]) >= 3]
        #   ⭐ **相机"热身完了没有"** ✗✗（用户 2026-10-08 ✓ "**一开始就认错了**" 的另一半 ✓）：
        #     开局那两三拍"跟拍 ≥3 拍的账"还没攒够 ✗ ⇒ 相机 **= 0** ⇒ 群体坐标里**每个框都像在动** ✗
        #     （其实是相机在动 ✓）⇒ 谁先攒过 26px 谁就被选中 ⇒ **开局必错** ✓。
        #     ⇒ 在相机量到手之前：**选人那一段整个停掉** ✓（`_cam_live` ✓ 见 `_target` ② ✓）、
        #       白块那道"看在不在动"也一并停掉 ✓（它读的就是群体坐标 ✓ 同样是脏的 ✓）。
        if len(_moves) >= _ANC_CAM_LIVE_N:
            self._cam_live = True
        _cam_med = anc_median2(_moves)
        _psr_mv, _resp = self._psr(gray)
        #   波浪分（三个信号 ✓ 见 `anc_wave_score` ✓）：
        _mads = [math.hypot(float(_m[0]) - _cam_med[0], float(_m[1]) - _cam_med[1])
                 for _m in _moves]
        _mad = float(np.median(_mads)) if _mads else 0.0
        _frac = ((sum(1 for _d in _mads if _d <= 3.0) / float(len(_mads)))
                 if _mads else 1.0)
        self.wave = anc_wave_score(_resp, _mad, _frac)
        if self.wave >= self.wave_t:
            self._wave_hold = _ANC_WAVE_HOLD
        elif self._wave_hold > 0:
            self._wave_hold -= 1
        self.wave_now = bool(self._wave_hold > 0)
        if self.wave_now:
            self.wave_n += 1
            return                                    # ⚠ 波浪段：相机、档案一律冻住 ✓
        #   ① 测量：静止片中位 ⇄ 相位相关（**两个都可用 ⇒ 取中位** ✓ —— 片位移是"点测量"、
        #     相位相关是"全图测量" ✓ 慢平移下前者更稳 ✓ 后者当兜底 ✓）。
        _step = _cam_med
        if (abs(_step[0]) < 0.05 and abs(_step[1]) < 0.05
                and (abs(_psr_mv[0]) > 0.5 or abs(_psr_mv[1]) > 0.5)):
            _step = _psr_mv
        #   ② 限幅（慢平移 ✓ 一步超过它 = 被骗 ✓）
        _m = math.hypot(_step[0], _step[1])
        if _m > self.cam_step_max:
            _k = self.cam_step_max / _m
            _step = (_step[0] * _k, _step[1] * _k)
        #   ③ 平滑（慢速 ⇒ 滤波比本拍测量准 ✓）
        _a = float(self.cam_smooth)
        self.cam_mv = (float(self.cam_mv[0]) + _a * (_step[0] - float(self.cam_mv[0])),
                       float(self.cam_mv[1]) + _a * (_step[1] - float(self.cam_mv[1])))
        #   ⭐⭐ **逐格"相对群体的速度"**（= 逐拍位移 − 相机 ✓ 的 EMA ✓ px/拍 ✓）—— 给白块那一道
        #     "**它也在动吗**"的门用 ✓（`_white_moving` ✓）；⚠ 它**只给那一处用** ✗，判据别的一律
        #     不碰 ✓（见那段"实测：小样本场景里群体坐标那套不稳 ✓ 这个稳 ✓"）。
        for _o in self._objs.values():
            _rel = (float(_o["mv"][0]) - float(self.cam_mv[0]),
                    float(_o["mv"][1]) - float(self.cam_mv[1]))
            _pv = _o.get("rel_v")
            _o["rel_v"] = (_rel if _pv is None
                           else (float(_pv[0]) + 0.45 * (_rel[0] - float(_pv[0])),
                                 float(_pv[1]) + 0.45 * (_rel[1] - float(_pv[1]))))
        _new = (float(self.cam_cum[0]) + float(self.cam_mv[0]),
                float(self.cam_cum[1]) + float(self.cam_mv[1]))
        #   ④ **锚定校正**：静止档案在哪、它们就该在哪 ✓ 偏差 = 累积漂移 ✓ 拉回来（限幅 ✓）
        _res = []
        for _o in self._objs.values():
            if not _o["static"] or _o["g_rec"] is None:
                continue
            _res.append((float(_o["c"][0]) - _new[0] - float(_o["g_rec"][0]),
                         float(_o["c"][1]) - _new[1] - float(_o["g_rec"][1])))
        if _res:
            _corr = anc_median2(_res)
            _cm = math.hypot(_corr[0], _corr[1])
            if _cm > _ANC_ANCHOR_MAX:
                _k2 = _ANC_ANCHOR_MAX / _cm
                _corr = (_corr[0] * _k2, _corr[1] * _k2)
            _new = (_new[0] + _corr[0], _new[1] + _corr[1])
        self.cam_cum = (_new[0], _new[1])

    # ---------------- 白块认定（用户 2026-10-08 ✓ 补上 ✓）----------------

    def _white(self, gray):
        """**白块认定** ⇒ 它落在哪条账里就认哪条（回那个 id ✓ 没认到 ⇒ `None` ✓）。

        ⭐⭐⭐⭐⭐ 用户 2026-10-08 ✓ 原话："**圆一开始就乱锁，没有认白色图形**" ✓✓ ——
          ⚠⚠⚠ **老两条底盘都有这条**（`lie_motion.pick_white` ✓ 用户 2026-10-05 ✓ "**任何时候出现
          白色图形都需要认真目标**" ✓✓），**我新建这一档时漏了** ✗ ⇒ 开局只能靠"净位移"慢慢攒分
          ⇒ 用户看到的就是"一开始就乱锁" ✓。
        ⚠⚠ **口径一律复用** `pick_white` ✗（**别在这儿再写一份白度阈值** ✗）：真目标**会逐渐透明**
          ⇒ 绝对白度一直掉 ✓ 只有它那套"门内相对亮度 ＋ 必须真有增益"的判法扛得住 ✓；
          再加它那两道**去抖门**（连续 `_WHITE_STREAK` 拍 ✓、相邻两拍位移 ≤ `_WHITE_STEP` ✓
          —— 老底盘实测：真白块**每拍只挪 2~3px** ✓，假白块**全是单拍闪现** ✗）。
        ⚠ 拿不到灰度图 / 没找到 / 连续拍数不够 ⇒ **什么都不做** ✓（不猜 ✗）。
        """
        if gray is None or not self._objs:
            return None
        #   ① **持有者还在不在**（= "只认那个一直在的" ✓）：就在**它自己那格框的范围内**问一句
        #      "你还白着吗" ✓（`pick_white(pos, gate)` ✓ 老底盘就是这么用的 ✓）。
        if self._white_tid is not None:
            _oh = self._objs.get(int(self._white_tid))
            if _oh is None:
                self._white_tid = None         # 那条账没了 ⇒ 持有者随之作废 ✓（不猜 ✗）
            else:
                _lh = max(_ANC_WHITE_SEEK * max(float(_oh["wh"][0]), float(_oh["wh"][1])),
                          _ANC_GATE_MIN)
                if self._is_white(gray, float(_oh["c"][0]), float(_oh["c"][1]), _lh):
                    self._white_seen = int(self._n)        # 还白着 ⇒ **续命** ✓
                    self._white_cand, self._white_cand_n = None, 0
                    return int(self._white_tid)            # ⚠ **不再去看别人** ✗（"只认那个一直在的" ✓）
                if (int(self._n) - int(self._white_seen)) > _ANC_WHITE_GRACE:
                    self._white_tid = None      # 连着这么多拍不白 ⇒ **放下** ✓（真白片会逐渐透明 ✓）
        #   ② 没有持有者 ⇒ 找**全局最白**那块（＋"相邻两拍不许乱跳"那道去抖门 ✓）
        #   ⚠⚠⚠ **候选一律走 `anc_white_cands`** ✗✗（用户 2026-10-08 ✓ "**正中心的白块你都认不准**" ✓✓）——
        #     它 = `pick_white` 那套阈值口径 ✓ ＋ 两道实测门（判掉**文字细长条** ✗、只留 **ROI 内** ✓）；
        #     直接调 `pick_white` 会咬到 `(462.5,64) 454×9` 那行标题文字 ✗（实测 ✓ 见那两段说明 ✓）。
        _cands = anc_white_cands(gray, self.roi)
        _pt = (None if not _cands else (float(_cands[0][0]), float(_cands[0][1])))
        if _pt is None:
            self._white_n = 0
            return None
        if not self._cam_live:
            return None                        # 相机还没量到 ⇒ 群体坐标是脏的 ⇒ 那道"在动"门不可信 ✓
        if (self._white_pt is not None
                and math.hypot(_pt[0] - float(self._white_pt[0]),
                               _pt[1] - float(self._white_pt[1])) <= _WHITE_STEP):
            self._white_n += 1
        else:
            self._white_n = 1                  # ⚠ 乱跳 ⇒ **重新数**（闪现过不来 ✓ 见 `_WHITE_STEP` ✓）
        self._white_pt = _pt
        if self._white_n < _WHITE_STREAK:
            return None
        _best, _bd = None, None
        for _i, _o in self._objs.items():
            _lim = max(_ANC_WHITE_SNAP * max(float(_o["wh"][0]), float(_o["wh"][1])),
                       _ANC_GATE_MIN)
            _d = math.hypot(_pt[0] - float(_o["c"][0]), _pt[1] - float(_o["c"][1]))
            if _d <= _lim and (_bd is None or _d < _bd):
                _best, _bd = int(_i), _d
        if _best is None:
            return None                        # 白块落在**哪条账都不属于**的位置 ⇒ 不认 ✓（不猜 ✗）
        #   ③ **同一个 id 连着 `_WHITE_STREAK` 拍** ⇒ 才当持有者 ✓（⚠ 两个都一直在的白片：
        #      先到先得 ✓ 之后**再也不抢** ✓ 见 ① ✓）。
        if self._white_cand == _best:
            self._white_cand_n += 1
        else:
            self._white_cand, self._white_cand_n = int(_best), 1
        if self._white_cand_n < _WHITE_STREAK:
            return None
        #   ⚠⚠⚠ **不动的白东西不许认** ✗✗（用户 2026-10-08 ✓ 原话："**一开始就认错了**" ✓✓）——
        #     实测：白 UI / 高光又白又大又稳 ⇒ 一开场就被认走 ✗；而这一档的前提是"**那一片在动**" ✓
        #     ⇒ 白块**必须同时也在动** ✓（看 `ghist` 短窗净位移 ✓ 见 `_white_moving` ✓）。
        #     ⚠ 只对"**新认**"要求 ✗：**持有者 / 首选号**已经在手上 ⇒ 它停下来了也照样用它 ✓
        #     （"一直在的那个"不许因为它停一下就被抢 ✓）。
        _ob = self._objs.get(int(_best))
        if _ob is None or not self._white_moving(_ob):
            return None
        #   ⚠⚠⚠ **认过的那一片"只要还白着"就永远优先** ✗✗（用户 2026-10-08 ✓ 原话："**只认那个
        #     一直在的**" ✓✓）—— ⚠ **实测**：光靠"持有者活着就不换"还不够 ✗：持有者偶有一两段
        #     查不白（被盖 / 分位门抖 ✓）⇒ 放下 ⇒ 转眼认了别人 ⇒ 过几拍又认回来 ⇒ **在 #2 ↔ #23
        #     之间来回**（200 拍里"白块认定"还写了 6 次 ✓ 目标号交替 12 次 ✗）。⇒ 备一个**首选号**
        #     （`_white_pref` ✓ 认过的第一个 ✓ 永不改 ✓）：**它此刻要是白的，就直接用它** ✓，
        #     不许认新的 ✗（"一直在的那个"的字面 ✓）。
        if self._white_pref is not None and int(self._white_pref) != int(_best):
            _op = self._objs.get(int(self._white_pref))
            if _op is not None:
                _lp = max(_ANC_WHITE_SEEK * max(float(_op["wh"][0]), float(_op["wh"][1])),
                          _ANC_GATE_MIN)
                if self._is_white(gray, float(_op["c"][0]), float(_op["c"][1]), _lp):
                    self._white_tid = int(self._white_pref)
                    self._white_seen = int(self._n)
                    self._white_hold = _ANC_WHITE_HOLD
                    return int(self._white_pref)
        self._white_tid = int(_best)           # 认到 ⇒ 它就是**持有者** ✓
        if self._white_pref is None:
            self._white_pref = int(_best)      # ⭐ **首选号**（只记第一次 ✓ 见上面那段 ✓）
        self._white_seen = int(self._n)
        self._white_hold = _ANC_WHITE_HOLD     # ＋ **按住**这些拍 ✓（见 `_target` ✓）
        return int(_best)

    def _white_moving(self, _o):
        """这一格**最近几拍真的相对群体在动**吗（白块认定那道额外的门 ✓ 见常量 ✓）。

        ⚠⚠ **口径**：`rel_v` = "**逐拍位移 − 相机**"的 EMA（px/拍 ✓ 见 `_camera` ✓）——
          **相机带着一起漂不算"它在动"** ✓（正是要的 ✓）、**不用群体坐标历史** ✗（**实测**：
          那种"4 个框里 1 个在动"的小样本场景里，中位/锚定会被带动 ⇒ 群体坐标算出来的净位移
          只有 ~1px/拍 ✗ ⇒ 门永远过不去 ✓；而 `rel_v` 这种**逐拍**量在这个场景里是 **6px/拍** ✓ 稳 ✓）。
        ⚠ 还没攒到速度（刚出生 / 相机没热身 ✓）⇒ 回 `False` ✓（**不猜** ✗ —— 宁可晚几拍认 ✓）。
        """
        _rv = _o.get("rel_v")
        if _rv is None:
            return False
        return math.hypot(float(_rv[0]), float(_rv[1])) >= _ANC_WHITE_V

    def _local_white(self, gray):
        """⭐⭐⭐⭐⭐ **局部白度接力** ✗✗（用户 2026-10-08 ✓ 原话："**看起来你的算法完全无法跟踪目标 ；
        帧 278之后就在满屏幕乱窜 无法正常跟踪目标**" ✓✓ —— 这一条就是治它的 ✓）。

        ⚠⚠⚠ **真因**（**实测** ✓ 见下）：目标按游戏规则"**逐渐透明**" ⇒ 白块判据**全图**那一版看不见
          它了 ✗（全图取前 5% 最亮的 ✗ —— 对话框一消失，背后那面**石墙**又亮又花 ⇒ 阈值被它顶走 ✗✗，
          实测帧 280~330 全图搜索**全程 `None`** ✗）；可**在它自己那一带**它仍然是**最亮、最像一块形状**
          的那块 ✓ ⇒ 只是在**局部**量才行 ✓✓。丢观测之后 ⇒ `_target` 里"滑行 2 拍就认领别人的框"
          （`_ANC_CLAIM_MISS = 2` ✓）＋ 挑框器换人 ⇒ **满屏乱窜** ✓✓。

        **实测那条接力**（真素材 `10月7日.mp4`，半径 140px、每拍只在上一位置附近找 ✓）：
          ```
            帧 280 (480.9,251.9) → 285 (483.0,250.5) → 290 (489.2,247.6) → 300 (497.4,245.6)
              → 306 (503.5,234.9) → 312 (510.0,232.3) → 324 (524.6,225.2) → 330 (529.5,224.1)
            亮度均值 238 → 228 → 223 → 220 → 213 → 201（**逐渐透明** ✓ 正是那游戏说的 ✓）
          ```
          平滑、单调 ✓ 与"人眼在画面上看到的那条轨迹"一致 ✓ ⇒ 拿它当**观测** ✓。

        ⚠ 三步纪律（与 `_patch` 完全同一套 ✓ 不另立判据 ✗）：
          ① `pos` / `radius` 一起喂给 `anc_white_cands` ⇒ 阈值**在那个方圆之内**算 ✓（这才看得见 ✓）；
          ② **步长门**照旧（`_ANC_PATCH_STEP × 边长` ✓）⇒ 一步蹦太远的观测**不采信** ✗
             （实测踩到过：帧 318 摸到一块 601px 的渣 ✗ 离上一拍 20px ⇒ 被这道门挡住 ✓）；
          ③ 认下就**同步把模板也重取一次** ✓（外观跟着"变灰"走 ✓ ⇒ 后面模板匹配也能接着用 ✓）。
        ⚠ 只在"**已经有目标**"时用 ✓（没身份时这一步没有意义 ✓ 认身份还是 `_white` 那条 ✓）；
          波浪段不采信位置 ✓（与全档纪律一致 ✓）。
        """
        self._obs_local = None
        if gray is None or self.tid is None or self.pos_g is None:
            return
        if self.wave_now:
            return
        _o = self._objs.get(int(self.tid))
        _sz = (max(float(self._tpl_wh[0]), float(self._tpl_wh[1]))
               if self._tpl_wh is not None else
               (max(float(_o["wh"][0]), float(_o["wh"][1])) if _o is not None else 60.0))
        _lim = max(24.0, float(_ANC_WHITE_LOCAL_R) * float(_sz))
        _sx = float(self.pos_g[0]) + float(self.cam_cum[0])
        _sy = float(self.pos_g[1]) + float(self.cam_cum[1])
        try:
            _c = anc_white_cands(gray, self.roi, pos=(_sx, _sy), radius=_lim)
        except Exception:                      # noqa: BLE001
            return
        if not _c:
            return
        _cx, _cy = float(_c[0][0]), float(_c[0][1])
        _step_max = max(6.0, _ANC_PATCH_STEP * float(_sz))
        if math.hypot(_cx - _sx, _cy - _sy) > _step_max:
            return                             # ② 一步蹦太远 ⇒ 不是它 ✓（不猜 ✗）
        _ng = (_cx - float(self.cam_cum[0]), _cy - float(self.cam_cum[1]))
        _dv = (_ng[0] - float(self.pos_g[0]), _ng[1] - float(self.pos_g[1]))
        _a = 0.35
        self.vel = ((0.0, 0.0) if self.vel is None else
                    (float(self.vel[0]) * (1 - _a) + _dv[0] * _a,
                     float(self.vel[1]) * (1 - _a) + _dv[1] * _a))
        self.pos_g = _ng
        self.miss = 0
        self.state = "track"
        if self._tpl_wh is not None:
            _nt = self._grab(gray, _ng, self._tpl_wh)
            if _nt is not None:
                self._tpl = _nt                 # ③ 外观跟着"变灰"走 ✓
        self._obs_local = (_cx, _cy)
        self._obs_now = True                   # ⭐ 这一拍有真观测（见 `_bind_box` ✓）

    def _patch(self, gray):
        """**按外观跟目标**（一帧一次 ✓ 用户 2026-10-08 ✓ "**真目标大部分时候都是没有检出框的**" ✓✓）。

        三条（这一档真正的观测源 ✓ 见模块头 ＋ `_ANC_PATCH_*` 那几个常量 ✓）：
          ① **没模板** ⇒ 有目标（`self.tid` 已定：白块认的 / 选人选的 ✓）就用**它那格框**当模板 ✓
             （⚠ 只在"框在视野内"时取 ✗ —— 半出画的框剪出来的模板是假的 ✓）；
          ② **有模板** ⇒ 在**群体坐标的预测点**附近（先加回相机 ⇒ 画面坐标 ✓）做旋转＋平移搜索 ✓
             ⇒ 分数 ≥ `_ANC_PATCH_NCC` ⇒ 这一拍**有观测**：位置 / 速度 / 角度一起更新 ✓、
             模板慢 EMA 跟上 ✓；否则 ⇒ **这一拍没有观测** ✓（目标被挡 / 全透明 / 出了画面 ✓）
             ⇒ 位置由**滑行**顶着（`_target` 已经做过了 ✓）✓；
          ③ **检出框只当辅助** ✗：万一正好有一格框落在预测附近 ✓ ⇒ 把位置**轻轻拉到框心**（re-anchor ✓）
             ＋ 重取模板一次 ✓（治长期漂移 ✓）；⚠ 没有框 ⇒ **什么都不做** ✓ 照跟 ✓。
        ⚠ 拿不到灰度图（自检里只喂 dets ✓）⇒ **只做 ③**（纯几何那部分 ✓ 不炸 ✓）。
        """
        if self.tid is None or self.tid not in self._objs or self.pos_g is None:
            return
        _o = self._objs.get(self.tid)
        _cap = (None if _o is None else
                (float(_o["wh"][0]), float(_o["wh"][1])))
        _score = 0.0
        if gray is not None and self._tpl is not None:
            _rad = max(8.0, _ANC_PATCH_RAD * max(float(self._tpl_wh[0]),
                                                 float(self._tpl_wh[1])))
            _sx = float(self.pos_g[0]) + float(self.cam_cum[0])
            _sy = float(self.pos_g[1]) + float(self.cam_cum[1])
            _score, _dx, _dy, _dth = anc_patch_match(gray, self._tpl, _sx, _sy, _rad)
            #   ⚠⚠⚠ **一步走多大才算"同一块"** ✗✗（**实测踩到** ✓）：光靠"分够高"不够 ✗ ——
            #     同款片（砖 / 星）彼此相像 ⇒ 模板**会一步跳到隔壁那块**上去 ✗ ⇒ 位置每拍偏几像素、
            #     速度又把它吸收 ⇒ **越走越远**（实测 26 拍偏了 **290px** ✗✗）。⇒ 用户场景是"**缓速**"
            #     ✓ ⇒ 加一道**步长门**：一步超过 `_ANC_PATCH_STEP × 边长` 的观测**一律不采信** ✓
            #     （当成"这一拍没有观测" ⇒ 滑行 ✓）。
            _step_max = max(6.0, _ANC_PATCH_STEP * max(float(self._tpl_wh[0]),
                                                       float(self._tpl_wh[1])))
            if _score >= _ANC_PATCH_NCC and math.hypot(_dx, _dy) <= _step_max:
                _ng = (float(self.pos_g[0]) + _dx, float(self.pos_g[1]) + _dy)
                _dv = (_ng[0] - float(self.pos_g[0]), _ng[1] - float(self.pos_g[1]))
                _a = 0.35
                self.vel = ((0.0, 0.0) if self.vel is None else
                            (float(self.vel[0]) * (1 - _a) + _dv[0] * _a,
                             float(self.vel[1]) * (1 - _a) + _dv[1] * _a))
                self.pos_g = _ng
                self.theta = (float(self.theta) + float(_dth)) % 360.0
                self.miss = 0
                self.state = "track"
                #   ⚠⚠ **`or` 对 numpy 数组不能用** ✗✗（**实测踩到** ✓：`ValueError: truth value of
                #     an array ... is ambiguous` ✓）⇒ 老老实实判 `None` ✓。
                _nt = self._grab(gray, _ng, self._tpl_wh)
                if _nt is not None:
                    self._tpl = _nt
            self._tpl_score = float(_score)
            if _score >= _ANC_PATCH_NCC and math.hypot(_dx, _dy) <= _step_max:
                self._obs_now = True           # ⭐ 这一拍有真观测（见 `_bind_box` ✓）
        #   ③ 检出框当辅助（**没有框也照走** ✓）
        _snap = None
        if _cap is not None and _o is not None and int(_o["miss"]) == 0:
            _d = math.hypot(float(_o["g"][0]) - float(self.pos_g[0]),
                            float(_o["g"][1]) - float(self.pos_g[1]))
            if _d <= max(10.0, _ANC_PATCH_SNAP * max(_cap)):
                _snap = (float(_o["g"][0]), float(_o["g"][1]))
        if _snap is not None:
            self.pos_g = (_snap[0] if self._tpl is None else
                          (float(self.pos_g[0]) * 0.7 + _snap[0] * 0.3),
                          _snap[1] if self._tpl is None else
                          (float(self.pos_g[1]) * 0.7 + _snap[1] * 0.3))
            self.miss = 0
            self.state = "track"
            if gray is not None and _cap is not None and self._tpl is None:
                self._tpl_wh = _cap
                self._tpl = self._grab(gray, self.pos_g, _cap)

    def _fuse_box(self):
        """⭐⭐⭐⭐⭐ **融合（用"这一拍"的路标 ✓）** ✗✗（用户 2026-10-09 ✓ "**直到能跟上真值为止**" ✓✓）。

        ⚠⚠ **实测**：`_target` 那一处借框是**上一拍**的路标（`_target` 跑在 `_patch` 之前 ✗）
          ⇒ 实测 300/320 拍还是 **8.4 / 9.5px** ✗；而**用这一拍的外观位置当路标**、
          只吃"25px 内那一块尺寸像目标的框" ⇒ **3.2 / 2.5px** ✓✓（同一段素材 ✓）。
        ⇒ 所以本函数放在 `_patch` **之后** ✓：外观这一拍刚报过位置（`_obs_now` ✓）⇒ 拿它当路标 ✓，
          把**贴着的框**当精修 ✓（**门只有 25px** ✗：实测目标框常贴 1~17px ✓，而门一开大（30~45px）
          就会跟到墙上并行漂的诱饵 ✗ 误差 45~140px ✓）；
        ⚠ **没有外观观测的这一拍，一个字都不动** ✗（那正是"看不清"的时候，绝不能去认框 ✗）。
        """
        if (not self._obs_now or self.pos_g is None or self.tid is None
                or self.wave_now):
            return
        _sx = float(self.pos_g[0]) + float(self.cam_cum[0])
        _sy = float(self.pos_g[1]) + float(self.cam_cum[1])
        _best = None
        for _o in self._objs.values():
            if int(_o["miss"]) != 0 or _o["static"]:
                continue
            _c = (float(_o["c"][0]), float(_o["c"][1]))
            if self.roi is not None and not (
                    float(self.roi[0]) <= _c[0] <= float(self.roi[2])
                    and float(self.roi[1]) <= _c[1] <= float(self.roi[3])):
                continue                      # 只在框选区域里 ✓（用户给的 ROI ✓）
            _sz = 0.5 * (float(_o["wh"][0]) + float(_o["wh"][1]))
            if not (_ANC_FUSE_SZ_LO <= _sz <= _ANC_FUSE_SZ_HI):
                continue
            _d = math.hypot(_c[0] - _sx, _c[1] - _sy)
            if _d <= _ANC_FUSE_R and (_best is None or _d < _best[0]):
                _best = (_d, _c, _sz, _o)
        if _best is None:
            return
        _ng = (_best[1][0] - float(self.cam_cum[0]), _best[1][1] - float(self.cam_cum[1]))
        _dv = (_ng[0] - float(self.pos_g[0]), _ng[1] - float(self.pos_g[1]))
        _step_max = max(6.0, _ANC_PATCH_STEP * max(float(_best[2]) * 2.0, 12.0))
        if math.hypot(_dv[0], _dv[1]) > _step_max:
            return                            # 一步太大 ⇒ 不吃它 ✗（防"跳一下" ✓）
        self.pos_g = _ng
        self._obs_box = (_best[1][0], _best[1][1],
                         float(_best[3]["wh"][0]), float(_best[3]["wh"][1]))
        self.miss = 0
        self.state = "track"

    def _reacq(self):
        """**报丢之后的"找回"** ✓（⇒ 认回来就改 `tid`/`pos_g`/`vel`/`miss` ✓）—— **只在 `lost` 里调** ✓。

        ⭐⭐⭐⭐⭐ 用户 2026-10-09 ✓ 原话："**假目标虽然是静止不动的，但是会有类似水面波纹的噪声使其
          几何也发生变化**" ✓✓ —— 这句就是判据 ✓：
            · **`_ANC_REACQ_R` 半径内**（群体坐标 ✓ 冻结着我们丢之前的位置 ✓）；
            · **净位移 ≥ `_ANC_SCORE_MIN`** ✓（= "真在走" ✓ 波纹免疫 ✓：实测抖 ±9px 的片子净位移
              只有十几 px ✗ 而真在走的 41px ✓）；
            · **连续 `_ANC_REACQ_N` 拍都是同一片** ✓（单拍尖峰不算 ✓）。
        ⇒ 静止而被波纹扭着的假目标**永远过不了** ✓✓；真目标（在动的那片）会回来 ✓。
        ⚠ **换身份只允许发生在 `lost` 里** ✗（跟踪中一律不换 ✓ 见 `_target` ② ✓）—— 与用户的
          C′ 定案不冲突 ✓：那是"丢了之后重新找回"，不是"跟着跟着跳走" ✓。
        """
        if self.pos_g is None:
            self._reacq_bid, self._reacq_n = None, 0
            return
        _best = None
        for _i, _o in self._objs.items():
            if int(_o["miss"]) != 0 or _o["static"]:
                continue
            if float(_o.get("score") or 0.0) < float(_ANC_SCORE_MIN):
                continue                      # 不是"真在走"的 ⇒ 出局 ✓（波纹扭着的静止片在这儿挡掉 ✓）
            _d = math.hypot(float(_o["g"][0]) - float(self.pos_g[0]),
                            float(_o["g"][1]) - float(self.pos_g[1]))
            if _d <= float(_ANC_REACQ_R) and (_best is None or _d < _best[0]):
                _best = (_d, int(_i), _o)
        if _best is None:
            self._reacq_bid, self._reacq_n = None, 0
            return
        if self._reacq_bid == int(_best[1]):
            self._reacq_n += 1
        else:
            self._reacq_bid, self._reacq_n = int(_best[1]), 1
        if self._reacq_n < int(_ANC_REACQ_N):
            return
        self.tid = int(_best[1])
        self.pos_g = (float(_best[2]["g"][0]), float(_best[2]["g"][1]))
        self.vel = (0.0, 0.0)
        self.miss = 0
        self.state = "track"
        self.log_text = ("锚定：丢后找回 #%d（它在真走：净位移 %.0f ≥ 门 %.0f，离 %.0fpx）"
                         % (int(self.tid), float(_best[2].get("score") or 0.0),
                            float(_ANC_SCORE_MIN), float(_best[0])))

    def _box_on_pred(self):
        """**"压在我们预测位置上那块框"的几何**（⇒ 那一格的 dict ✓ 或 `None` ✓）—— **只借几何** ✓。

        ⭐⭐⭐⭐⭐ 用户 2026-10-08 ✓ 定案（"**动**" ✓；配套两句话："**这时候真目标检出框都还在**" ✓✓
          ＋ "**框大部分都是错的**" ✓✓）—— ⚠⚠ 这两句合起来说的正是本函数：**错的只是那些号** ✗，
          **压在目标上那块框的几何一直是对的** ✓：
            · 实测：有位置的 455 拍里「**有检出框正好压在我位置上**」**442 拍（97%）** ✓✓；
            · 实测那些框的尺寸**连续**（`98×114 / 94×98 / 99×102 / 103×100 / 109×102 …` ✓）、
              中心离目标约 **28px** ✓（框半边长 ~50 ⇒ **确实压在目标上** ✓）；
            · 而它们的**号**每几十拍换一个 ✗（`10 → 9/11 → 12 → 10 → 11 → 1 → 2` ✓）。
        ⇒ 于是：**号一概不认** ✗（不再 `self.tid = 那一格的号` ✗、不再并框 ✗）；只把它的
          **中心/尺寸**当作"这一拍目标在哪"的测量 ✓（位置**轻拉 30%** ✓ 见 `_target` ② ✓）。
        ⚠ 稳定性：墙上常有两块框**叠在一块儿**（实测帧 157：一块离 2px、一块离 27px ✓）⇒ 要是
          每拍都取"最近的那块" ⇒ 位置会在两块之间**抖** ✗ ⇒ 用 `_obs_bid`（上一拍用过那一格的号 ✓
          **只当"优先续用"的提示** ✓ 不是身份 ✗）⇒ 它还在、还压着 ⇒ 继续用它 ✓，断了才另择最近 ✓。
        ⚠ 静止片不借 ✗（"压在目标上的框"必然是在动的那一块 ✓）；波浪段由调用方跳过 ✓。
        """
        if self.pos_g is None:
            return None
        _sx = float(self.pos_g[0]) + float(self.cam_cum[0])
        _sy = float(self.pos_g[1]) + float(self.cam_cum[1])

        def _on(_o):
            _lim = max(12.0, 0.5 * max(float(_o["wh"][0]), float(_o["wh"][1])))
            return math.hypot(float(_o["c"][0]) - _sx, float(_o["c"][1]) - _sy) <= _lim

        if self._obs_bid is not None:
            _p = self._objs.get(int(self._obs_bid))
            if (_p is not None and int(_p["miss"]) == 0 and not _p["static"]
                    and float(_p.get("score") or 0.0) >= float(_ANC_SCORE_MIN) and _on(_p)):
                return _p
        _best = None
        for _i, _o in self._objs.items():
            if int(_o["miss"]) != 0 or _o["static"] or not _on(_o):
                continue
            #   ⚠⚠⚠ **借之前先问一句"它是不是真的在走"** ✗✗（用户 2026-10-09 ✓ 原话："**假目标虽然是
            #     静止不动的，但是会有类似水面波纹的噪声使其几何也发生变化**" ✓✓）—— 把那句话翻成判据
            #     就是：**不能看"几何变没变"** ✗（波纹会让静止的假目标几何也变 ✓ 实测墙上那些方形装饰
            #     就是被波纹扭着 ✓）；要看**群体坐标里一段时间（`_ANC_NET_WIN` = 1.2s ✓）的净位移** ✓
            #     —— 波纹是**往复**的 ⇒ 会自己抵消 ✓✓（这条尺本档本来就有 ✓ 也实测过：抖 ±9px 的片子
            #     净位移只有十几 px ✗ 而真在走的 41px ✓ 门 = `_ANC_SCORE_MIN`(26) ✓ 见
            #     `test_anchor_jitter_immunity` ✓）。⇒ **净位移不到门的框一律不许借** ✗
            #     （宁可不借 ⇒ 走滑行/报丢 ✓ 也不跟着一块"被波纹扭着转的静止假框"跑 ✓）。
            if float(_o.get("score") or 0.0) < float(_ANC_SCORE_MIN):
                continue
            _d = math.hypot(float(_o["c"][0]) - _sx, float(_o["c"][1]) - _sy)
            if _best is None or _d < _best[0]:
                _best = (_d, int(_i), _o)
        if _best is None:
            self._obs_bid = None
            return None
        self._obs_bid = int(_best[1])
        return _best[2]

    def _grab(self, gray, _pg, _wh):
        """取模板（画面坐标 = 群体坐标 ＋ 相机 ✓）—— 取不到（贴边 / 太糊）⇒ `None` ✓（不猜 ✗）。"""
        _g = np.asarray(gray, dtype=np.float32)
        _h, _w = _g.shape
        _sx = float(_pg[0]) + float(self.cam_cum[0])
        _sy = float(_pg[1]) + float(self.cam_cum[1])
        _tw, _th = max(8, int(round(float(_wh[0])))), max(8, int(round(float(_wh[1]))))
        _x0 = int(round(_sx - _tw / 2.0))
        _y0 = int(round(_sy - _th / 2.0))
        if (_x0 < 0 or _y0 < 0 or _x0 + _tw > _w or _y0 + _th > _h
                or _tw < 8 or _th < 8):
            return None
        _t = _g[_y0:_y0 + _th, _x0:_x0 + _tw]
        if _t.size < 64 or float(_t.std()) < 1e-3:
            return None                        # ⚠ 一片糊 / 纯色 ⇒ 拿它当模板只会乱配 ✗（不猜 ✓）
        return np.ascontiguousarray(_t)

    def _is_white(self, gray, _cx, _cy, _lim):
        """这一格框的范围内**还有白块吗**（`pick_white(pos, gate)` ✓ 回 `True/False` ✓）。

        ⚠ 只回答"**还在不在**" ✗ 不回点（调用方要的是"持有者还能不能续命" ✓ 见 `_white` ① ✓）；
        ⚠ `pick_white` 出问题 ⇒ 回 `False` ✓（白块只是**辅助证据** ⇒ 不许炸主路 ✓）。
        """
        try:
            #   ⚠ 一律走 `anc_white_cands` ✗（**同一把尺** ✓：判掉细长条 ＋ 只认 ROI 内 ✓）——
            #     直接调 `pick_white` 时，这一带要是压着一行文字 ⇒ 它就先中 ✗（实测踩到 ✓）。
            return bool(anc_white_cands(gray, self.roi, pos=(float(_cx), float(_cy)),
                                        radius=float(_lim)))
        except Exception:                      # noqa: BLE001
            return False

    # ---------------- 目标 ----------------

    def _target(self):
        """选"在动的那片" ＋ 观测 / 滑行 / 认领（改 `self.pos_g` / `vel` / `miss` / `theta` ✓）。"""
        #   ① 群体坐标（= 画面 − 相机累计 ✓）＋ 累积分 ＋ 静止档案（**波浪段一律不更新** ✓）
        for _i, _o in self._objs.items():
            _o["g"] = (float(_o["c"][0]) - self.cam_cum[0],
                       float(_o["c"][1]) - self.cam_cum[1])
            if _o["g_rec"] is None:
                _o["g_rec"] = _o["g"]
                _o["static_n"] = 0
                _o["static"] = False
                continue
            _dd = math.hypot(_o["g"][0] - float(_o["g_rec"][0]),
                             _o["g"][1] - float(_o["g_rec"][1]))
            if self.wave_now:
                pass                                  # ⚠ 波浪段：档案与分数都冻住 ✓
            elif _dd <= self.static_dev:
                _o["static_n"] = int(_o["static_n"]) + 1
                if int(_o["static_n"]) >= _ANC_STATIC_N:
                    _o["static"] = True
                #   档 = 慢 EMA（慢漂移是允许的 ✓ 例如水面把整幅慢慢移位 ✓）
                _o["g_rec"] = (float(_o["g_rec"][0]) + 0.02 * (_o["g"][0] - float(_o["g_rec"][0])),
                               float(_o["g_rec"][1]) + 0.02 * (_o["g"][1] - float(_o["g_rec"][1])))
            else:
                _o["static"] = False
                _o["static_n"] = 0
        #   ② **净位移**（= 这段窗口里它净走了多少 ✓ 见 `_ANC_NET_WIN` 那段 ✓）——
        #   ⚠⚠⚠ **老写法是"逐拍位移累加"** ✗✗（`score×0.92 + |本拍位移|` ✓）：**抖动片与真移动片
        #     攒得一样快** ✗（检出框抖一下也是几十像素 ✓）⇒ 一堆框同时够分 ⇒ 身份满屏换 ⇒
        #     圆跟着乱跳 ✓ = 用户 2026-10-08 看到的"**圆满屏乱跳，看不出什么时间连续性**" ✓✓。
        #   ⇒ 现在量"**净位移**"：抖来抖去净走 ≈ 0 ✓✓；一直在走 ⇒ **线性增长**（这才叫时间连续 ✓）。
        if not self.wave_now and self._cam_live:
            for _o in self._objs.values():
                _h = _o["ghist"]
                if _o["g"] is not None and int(_o["miss"]) == 0:
                    _h.append((float(self._t), _o["g"]))
                while len(_h) > 1 and (float(self._t) - float(_h[0][0])) > _ANC_NET_WIN:
                    _h.pop(0)
                if _h and _o["g"] is not None:
                    _o["score"] = math.hypot(float(_o["g"][0]) - float(_h[0][1][0]),
                                             float(_o["g"][1]) - float(_h[0][1][1]))
            for _o in self._objs.values():
                _o["g_prev"] = _o["g"]
        #   ⚠⚠⚠ **"在动的"必须这一拍看得见** ✗✗（**实测踩到** ✓）：不筛 `miss` ⇒ 挑框器会把目标换到
        #     一个**框早就没了**的旧身份上 ✗（实测：帧 50 刚靠"认领"接上一个新身份 ✓，帧 53 又被
        #     "改用 #7" 抢回去 —— 而那个 #7 **已经漏了 27 拍** ✗ ⇒ 状态立刻掉回 `coast` ✗✗）。
        #   ⚠ 这也正是"靠净位移打分"的题中之义 ✓：**看不见的片子，这一拍谈不到"在动"** ✓。
        _scores = {_i: float(_o["score"]) for _i, _o in self._objs.items()
                   if not _o["static"] and int(_o["miss"]) == 0}
        #   ⚠⚠⚠ **"上一拍有没有观测"也算数** ✗✗（**实测踩到** ✓）：模板匹配（`_patch` ✓）是这一档最
        #     主要的观测源 ✓，可它跑在 `_target` **之后** ✗ ⇒ `_target` 里只看得见"局部白度接力"那
        #     一路（实测 700 拍里只有 253 拍有 ✓ 36% ✗）⇒ 其余那些拍（明明模板匹配得好好的 ✓）
        #     挑框器就"自由"了 ⇒ 实测帧 353/369/421 冒出 `改用 #8`／`重新认领 #24` ✗ ⇒ 身份乱跳、
        #     位置被拽（单拍撞到限速上限 **30px** ✗）。⇒ 认"上一拍有没有观测" ✓（一拍延迟 ✓ 代价
        #     只是真丢了会晚一拍松手 ✓ 无妨 ✓）。
        #   ⚠⚠⚠ **"短缺口"要量"观测的缺口"，不是"框的缺口"** ✗✗（**实测踩到** ✓）：这份素材上目标
        #     那条框**大半时间根本不在**（实测"我那条框还在"只有 **83%** ✓ ⇒ 剩下 17% 全是"框缺口" ✗）
        #     ⇒ 用框缺口当门 ⇒ 动不动就"丢够 15 拍 ⇒ 放行" ✗ ⇒ 挑框器又开始换人（实测帧 367/381/424
        #     `改用 #35/#24/#30` ＋ 帧 421 `重新认领 #15` ✗ ⇒ 位置单拍撞到限速上限 30px ✗）。
        #   ⇒ 门改量**外观观测**的缺口（`_obs_gap` ✓ 局部白度接力 或 模板匹配给过观测 ⇒ 归零 ✓）：
        #     只要还在"半秒"（`_ANC_CLAIM_MISS` ✓ 同一把尺 ✓）内看见过目标 ⇒ 一律**滑行＋稳住身份** ✓。
        #   ⭐⭐⭐⭐⭐ **C′：身份一旦立下，就再也不换** ✗✗（用户 2026-10-08 ✓ 定案 ✓："**动**" ✓）——
        #
        #   ⚠⚠⚠ **为什么把三条路一起删掉** ✗✗（全部是用户亲眼看到的"满屏乱窜" ✓ 全都有实测 ✓）：
        #     · **改用别人的号**（挑框器"谁在动" ✓）：实测帧 278/353/367/381/424 `改用 #21/#8/#35/#24/#30`
        #       ✗ —— 位置当场被拽到石墙的错块上，单拍撞到限速上限 **30px** ✓；
        #     · **重新认领某块框**（`_ANC_CLAIM_MISS` 到点就去抱一块 ✓）：实测帧 279/421/470 ✗；
        #     · **并框**（把压在目标上的框并到我们名下 ✓）：连这个也不要了 ✓ —— 因为**框现在只当
        #       "目标这一拍在哪"的测量** ✓（见 ③ ✓ 与 `_box_on_pred` ✓），**再也不当身份** ✗。
        #   ⇒ 用户那句"**框大部分都是错的**"里的"错"，说的正是**这些号** ✗（实测：号每几十拍换一个 ✓
        #     `10 → 9/11 → 12 → 10 → 11 → 1 → 2` ✓）；而**压在目标上那块框的几何一直是对的**
        #     （实测：97% 的拍有框压在目标上 ✓ 尺寸连续 `98×114 / 94×98 / 99×102 / 103×100 …` ✓）。
        #
        #   ⇒ 现在的规则只有一条：**只有在"还没有身份"时才挑一次** ✓（白块优先 ✓ 见 `_white` ✓，
        #     它没动静时用"开局那片在动的" ✓）—— 之后无论谁多像、多"在动" ✗ **都不换** ✓。
        if self.tid is None:
            _seed = anc_pick(_scores, cur=None)
            if _seed is not None and int(_seed) in self._objs:
                self.tid = int(_seed)
                _o0 = self._objs[int(_seed)]
                self.pos_g = _o0["g"]
                self.vel = (0.0, 0.0)
                self.miss = 0
                self.state = "track"
                self.log_text = "锚定：认定 #%d 当目标（开局那片在动的）" % int(_seed)
        #   ③ 观测 / 滑行 / 认领
        if self.tid is None or self.tid not in self._objs:
            self.state = "init"
            self.pos_g = None
            self.pos = None
            self.vel = None
            return
        _o = self._objs[self.tid]
        _local = (self._obs_local is not None)      # 这一拍的"外观观测"（局部白度接力 ✓）
        _bx = (None if (_local or self.wave_now) else self._box_on_pred())
        if _local:
            #   ① **外观观测**（局部白度接力 ✓ 位置/速度/模板那一步都更新过了 ✓）—— **最高优先级** ✓
            self.miss = 0
            self.state = "track"
        elif _bx is not None:
            #   ② ⭐ **"压在我们预测位置上那块框"的几何**当测量 ✗✗（用户 2026-10-08 ✓ 定案："**动**" ✓）
            #      —— ⚠ **只借它的几何**（中心/尺寸 ✓）**不看它的号** ✗（号每几十拍就换一个 ✓ 实测 ✓）。
            #      ⚠ 位置**轻拉**（有模板时只吃 30% ✓ 老规矩 ✓）：实测这些框的中心离目标约 **28px**
            #        （框半边长 ~50 ✓ 所以还压在目标上 ✓）⇒ 全吃会把位置拖偏 ✗ ⇒ 拉一点就够 ✓
            #        （治长期漂移 ✓），主要位置还是外观那条路给的 ✓。
            if self.pos_g is not None:
                _sg = (float(_bx["c"][0]) - float(self.cam_cum[0]),
                       float(_bx["c"][1]) - float(self.cam_cum[1]))
                _dv = (_sg[0] - float(self.pos_g[0]), _sg[1] - float(self.pos_g[1]))
                #   ⚠⚠⚠ **这一步也必须限速** ✗✗（**实测踩到** ✓）：无模板时我原来写"全吃框心"（`_a = 1.0`
                #     ✓）⇒ 那条框的中心离我们可能有几十像素 ⇒ **实测单拍跳 48.2px** ✗（原来 11.4px ✓）。
                #     ⇒ 用本档现成那把"**慢速**"尺（`_ANC_PATCH_STEP × 边长` ✓ 与外观那两路**同一个门** ✓）
                #     给这一拉限步 ✓：一步最多走那么多 ✓（88px 的框 ⇒ 13px ✓）；剩下的下一拍继续拉 ✓。
                _dvn = math.hypot(_dv[0], _dv[1])
                _step_max = max(6.0, _ANC_PATCH_STEP * max(float(_bx["wh"][0]),
                                                           float(_bx["wh"][1])))
                #   ⭐⭐⭐⭐⭐ **融合：外观当"路标"、框当"精修"** ✗✗（用户 2026-10-09 ✓ 给我全权
                #     "**直到能跟上真值为止**" ✓✓）—— ⚠⚠ **实测**：外观那条路在"看得见"时差
                #     **8.4 / 9.5px** ✓，而**贴着目标的那块框**差 **3.2 / 2.5px** ✓✓（300/320 拍 ✓）。
                #     ⇒ 判据：**外观刚给过观测**（`_obs_gap ≤ 2` ✓ = 路标可信 ✓）**且框就在路标
                #     25px 内**（实测目标框常贴 1~17px ✓）⇒ **直接吃框心**（`_a = 1` ✓ 不掺水 ✗）；
                #     ⚠ 否则**退回老口径 30% 轻拉** ✓ —— 那正是"外观看不见时跟着诱饵框跑"的场景 ✗
                #     （实测：门开 30~45px 就会跟到墙上并行漂的诱饵 ✗ 误差 45~140px ✓）。
                _a = 1.0 if (int(self._obs_gap) <= 2 and _dvn <= 25.0) else 0.30
                if _dvn * _a > _step_max:
                    _a = _step_max / max(1e-6, _dvn)
                self.pos_g = (float(self.pos_g[0]) + _dv[0] * _a,
                              float(self.pos_g[1]) + _dv[1] * _a)
                self.vel = ((0.0, 0.0) if self.vel is None else
                            (float(self.vel[0]) * 0.65 + _dv[0] * 0.35,
                             float(self.vel[1]) * 0.65 + _dv[1] * 0.35))
            self._obs_box = (float(_bx["c"][0]), float(_bx["c"][1]),
                             float(_bx["wh"][0]), float(_bx["wh"][1]))
            self.miss = 0
            self.state = "track"
        elif self.wave_now:
            #   波浪段里"有框"也不拿它当测量 ✗（被搅过的位置进滤波 ⇒ 事后全是抖 ✓）
            #   但也**不判丢** ✓（只是这一拍不可信 ✓）—— 沿用上一拍位置 ✓
            self.state = "track" if self.pos_g is not None else "init"
        else:
            #   ③ **纯滑行**（慢速外推 ✓）—— ⚠⚠ **滑够 `_ANC_LOST_N` 拍就报"丢了"** ✗✗（用户 2026-10-08 ✓
            #     定案 ✓）：**位置报 `None`**（画面上不画 ✗）—— "没依据就别声称目标在这儿" ✓；
            #     ⚠ 内部**保留** `pos_g`/`vel` ✓（等它再露头 ⇒ ②① 任一成立就续上 ✓ 不猜新的位置 ✗）。
            self.miss = int(self.miss) + 1
            if int(self.miss) < int(_ANC_LOST_N):
                #   半秒内：按速度滑行 ✓（短缺口靠它接住 ✓ 实测帧 274~277 那种 ✓）
                if self.pos_g is not None and self.vel is not None:
                    self.pos_g = anc_coast(self.pos_g, self.vel)
                self.state = "coast"
            else:
                #   ⭐ 超过半秒 ⇒ **报丢**：⚠⚠ **位置就冻在这儿、不再往前推** ✗✗（**实测踩到** ✓）：
                #     一直推进去 ⇒ 预测越跑越远 ⇒ **再也回不来**（实测帧 380 之后连续 306 拍报丢 ✗）。
                #     ⇒ 冻住 + 在附近找"**真在走**的那一片"（见 `_reacq` ✓ 用你给的波纹判据 ✓）。
                self.state = "lost"
                self._reacq()
        if self.pos_g is None or self.state == "lost":
            self.pos = None
            return
        self.pos = (float(self.pos_g[0]) + self.cam_cum[0],
                    float(self.pos_g[1]) + self.cam_cum[1])

    def _angle(self, gray):
        """目标窗口的**相对旋转**累加到 `self.theta` ✓（拿不到 ⇒ 保持 ✓ **不猜** ✗）。"""
        if gray is None or self.tid is None or self.tid not in self._objs:
            self._patch_prev = None
            return
        _o = self._objs[self.tid]
        if int(_o["miss"]) > 0 or self.wave_now:
            self._patch_prev = None            # ⚠ 波浪段不量角度 ✓（窗口被搅过 ⇒ 量出来是假的 ✓）
            return
        _w = int(max(16, min(140, _ANC_ANG_WIN * max(float(_o["wh"][0]),
                                                     float(_o["wh"][1])))))
        _cx, _cy = int(round(float(_o["c"][0]))), int(round(float(_o["c"][1])))
        _h2, _w2 = _w // 2, _w // 2
        _h, _fw = gray.shape[0], gray.shape[1]
        _y0, _x0 = max(0, _cy - _h2), max(0, _cx - _w2)
        _y1, _x1 = min(_h, _cy + _h2), min(_fw, _cx + _w2)
        if _y1 - _y0 < 12 or _x1 - _x0 < 12 or (_y1 - _y0) != (_x1 - _x0):
            self._patch_prev = None
            return
        _p = np.ascontiguousarray(gray[_y0:_y1, _x0:_x1])
        if self._patch_prev is not None and self._patch_prev.shape == _p.shape:
            _d = anc_patch_dtheta(self._patch_prev, _p)
            if _d is not None:
                self.theta = float(self.theta) + _ANC_ANG_SMOOTH * float(_d)
        self._patch_prev = _p

    # ---------------- 主流程 ----------------

    def process(self, img, ts=None, dets=None, idx=None):
        """一拍：见类说明 ✓。`idx` = 帧号（日志用 ✓）；`ts` = 秒（算步长用 ✓ 可缺）。"""
        self._n += 1
        self.log_text = None
        if ts is not None:
            self._dt = (1.0 if self._ts_prev is None
                        else max(1e-3, float(ts) - float(self._ts_prev)))
            self._ts_prev = float(ts)
            #   ⚠⚠⚠ **时钟必须真的走** ✗✗（**实测踩到** ✓）：`self._t` 原来是**只初始化、从没推进**
            #     过 ✗ ⇒ "净位移那扇窗"（`_ANC_NET_WIN` ✓）**永远不弹出旧样本** ✗ ⇒ 分数实际是
            #     "**从它出生到现在**的净位移" ✗✗（既"忘了时间"、又**只涨不落** ✓）⇒ 换过人之后
            #     旧账还挂着高分 ✓（这次"乱跳"的帮凶之一 ✓）。⇒ 现在按 `ts` 走**真实秒数** ✓。
            self._t = float(ts)
        else:
            self._dt = 1.0
            self._t = float(self._t) + 1.0     # 没给时刻 ⇒ 按"一拍一秒"的**名义钟** ✓（仍单调 ✓）
        gray = None
        if img is not None and getattr(img, "shape", None) is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            gray = np.ascontiguousarray(gray)
            self.frame_wh = (int(gray.shape[1]), int(gray.shape[0]))
        _bs = self._boxes(dets)
        self._match(_bs)
        self._obs_now = False                  # ⚠ "这一拍有没有真观测"（诊断 ✓ 见 `_local_white` ✓）
        self._obs_box = None                   # ⚠ 每拍重取：这一拍"借了几何"的那块框（见 `_target` ② ✓）
        _prev_wave = bool(self.wave_now)
        self._camera(gray)
        #   ⭐⭐⭐⭐⭐ **白块认定**（用户 2026-10-08 ✓ 见 `_white` ✓）—— ⚠ **必须在 `_target` 之前**
        #     ✗：白块是**直接把身份定下来**的（`self.tid` ✓）⇒ `_target` 那一道就会认这个身份 ✓。
        _wt = self._white(gray)
        _wmsg = ""
        if _wt is not None and int(_wt) != int(self.tid if self.tid is not None else -1):
            self.tid = int(_wt)
            _ow = self._objs.get(int(_wt))
            if _ow is not None:
                self.pos_g = _ow["g"]
                self.vel = (0.0, 0.0)
                self.miss = 0
            #   ⚠⚠ **这句要留到最后再说** ✗✗（**实测踩到** ✓）：`_target` 里那几支会**清空 / 覆盖**
            #     `log_text`（每拍开头就置 `None` ✓）⇒ 写在这儿 ⇒ 界面上**一个字都看不到** ✗
            #     可这行恰恰是用户要的"认证"证据 ✓ ⇒ 先存着、`_target` 之后再拼上去 ✓。
            _wmsg = "锚定：白块认定 #%d（出现白色图形 ⇒ 就是它）" % int(_wt)
        #   ⭐⭐⭐⭐⭐ **局部白度接力** ✗✗（用户 2026-10-08 ✓ "**帧 278之后就在满屏幕乱窜 无法正常跟踪
        #     目标**" ✓✓）—— ⚠ **必须在 `_target` 之前** ✗：它是**观测** ✓ ⇒ `_target` 那边要靠这个
        #     标志压住"改用它人 / 认领别人的框"那两条（实测：278 之后挑到的框**全是石墙上的错块** ✗
        #     —— 就是它们把圆拖得满屏乱窜 ✓ 见 `_local_white` 那段数 ✓）。
        self._local_white(gray)
        self._target()
        if _wmsg:
            self.log_text = ((self.log_text + "；" + _wmsg) if self.log_text else _wmsg)
        self._white_hold = max(0, int(self._white_hold) - 1)
        #   ⚠⚠ **报出位置每拍限速** ✗✗（**一处口径** = 老底盘那个 `_POS_STEP_MAX` ✓ 30px ✓）——
        #     用户 2026-10-08 ✓ "**圆满屏乱跳**" 的**另一半病根**：换一次人 ⇒ 圆一步跨几百 px ✗
        #     ⇒ 跟老底盘一样削平 ✓（老底盘当年治的正是"认领那一下的猛冲"✓ 见 `lie_motion` ✓）。
        if self.pos is not None and self._pos_prev_report is not None:
            _rdx = float(self.pos[0]) - float(self._pos_prev_report[0])
            _rdy = float(self.pos[1]) - float(self._pos_prev_report[1])
            _rlen = math.hypot(_rdx, _rdy)
            if _rlen > _POS_STEP_MAX:
                _rk = _POS_STEP_MAX / _rlen
                self.pos = (float(self._pos_prev_report[0]) + _rdx * _rk,
                            float(self._pos_prev_report[1]) + _rdy * _rk)
        #   ⚠⚠⚠ **限速器的参照不要在"报丢"那几拍清空** ✗✗（**实测踩到** ✓）：清空 ⇒ 恢复报位置的那一
        #     拍就没有参照 ⇒ **实测单拍跳 48.2px** ✗（跨过 9 个"丢"的拍 ✓）。⇒ 只在**报了位置**时更新 ✓
        #     ⇒ 恢复那一拍照样受 `_POS_STEP_MAX`（**一处口径** ✓ 30px ✓）约束 ✓。
        if self.pos is not None:
            self._pos_prev_report = (float(self.pos[0]), float(self.pos[1]))
        #   ⭐⭐⭐⭐⭐ **按外观跟**（用户 2026-10-08 ✓ "**真目标大部分时候都是没有检出框的**" ✓✓）——
        #     它**独立于检出框**给出位置/速度/角度 ✓ ⇒ 拿不到框的那几十拍照样跟得住 ✓。
        self._patch(gray)
        #   ⭐⭐⭐⭐⭐ **融合（用"这一拍"的路标 ✓）** ✗✗（用户 2026-10-09 ✓ "**直到能跟上真值为止**" ✓✓）——
        #     ⚠ 必须放在 `_patch` **之后** ✗：`_target` 里那次借框只看得见**上一拍**的外观位置 ✓
        #     （实测 300/320 拍还差 8.4 / 9.5px ✗）；这里用**这一拍**的 ⇒ 实测 **3.2 / 2.5px** ✓✓。
        self._fuse_box()
        #   ⚠⚠⚠ **融合之后必须把"报出去的位置"重算一次** ✗✗（**实测踩到** ✓）：`self.pos` 是在
        #     `_target` 里（走到 `_fuse_box` **之前** ✗）就算好的 ⇒ 不重算 ⇒ 页面上那个圆还是老位置
        #     ✗（实测：加了融合，误差一个点都没变 ✓✓ —— 全靠这条才看出来 ✗）。
        if self.pos_g is not None and self.state != "lost":
            self.pos = (float(self.pos_g[0]) + self.cam_cum[0],
                        float(self.pos_g[1]) + self.cam_cum[1])
        self._obs_prev = bool(self._obs_now)   # ⚠ 留给**下一拍**的诊断/参考用（见上面那段 ✓）
        self._obs_gap = (0 if self._obs_now else int(self._obs_gap) + 1)
        #   ⚠ 角度**二选一处口径** ✗：有模板 ⇒ 角度由 `_patch` 的旋转搜索给（`+Δθ` 累加 ✓）；
        #     没有模板 ⇒ 才用 `_angle` 那条（两片灰度小图比一次 ✓ 见那段 ✓）—— 两条一起上会**双重计数** ✗。
        if self._tpl is None:
            self._angle(gray)
        if self.wave_now and not _prev_wave:
            _line = ("波浪段开始（波浪分 %.2f ⇒ 相机与静止档案冻结）" % float(self.wave))
            self.log_text = ((self.log_text + "；" + _line) if self.log_text else _line)
        elif _prev_wave and not self.wave_now:
            _line = "波浪段结束（相机恢复，锚定校正把累加量拉回）"
            self.log_text = ((self.log_text + "；" + _line) if self.log_text else _line)
        if gray is not None:
            self._gray_prev = gray
        #   ⭐ 老壳（`MotionRunner.step` ✓）要的"相机量" = 本档的 `cam_mv` ✓（见 `__init__` ✓）。
        self._median_mv = (float(self.cam_mv[0]), float(self.cam_mv[1]))
        self._med_ema = (float(self.cam_mv[0]), float(self.cam_mv[1]))
        #   ⚠ **半径口径**：暂用"目标那格框的面积开方的一半" ✓（本档**不做**"标准面积"那本账 ✗
        #     —— 它属于老两档的"上板/发号"那套 ✓ 见模块头 ✓）；太小 ⇒ 给 18px 下限 ✓
        #     （演示窗用它画绿圈 ✓ 太小就看不见了 ✓）。
        self.rad = 0.0
        if self.tid in self._objs:
            _o = self._objs[self.tid]
            self.rad = float(max(18.0, 0.5 * math.sqrt(
                max(1.0, float(_o["wh"][0]) * float(_o["wh"][1])))))
        # ---- 画面账本 `box_v`（**逐位对齐老布局** ✓ 演示窗按同一索引读 ✓ 见 `VelocityChassis` ✓）----
        _bv = []
        for _i in sorted(self._objs):
            _o = self._objs[_i]
            if int(_o["miss"]) > 0:
                continue
            _has = bool(self.pos is not None
                        and abs(float(_o["c"][0]) - self.pos[0]) <= float(_o["wh"][0]) / 2.0
                        and abs(float(_o["c"][1]) - self.pos[1]) <= float(_o["wh"][1]) / 2.0)
            #   状态码（这一档借老布局的第 8 位 ✓）：`0` 静止（蓝 ✓）｜`1` **在动的那个目标**
            #   （演示窗在这一档不按它上色 ✗ 但**点选面板/诊断要看得见** ✓ 见 `motion_viz` 的
            #   `target_tid` ✓ —— 真正的强调画在 `draw` 里 ✓）。
            _bv.append((float(_o["c"][0]), float(_o["c"][1]),
                        float(_o["mv"][0]), float(_o["mv"][1]), int(_i), float(_o["conf"]),
                        -1.0, (1 if int(_i) == self.tid else 0), _has,
                        bool(int(_i) == self.tid),
                        float(_o["wh"][0]), float(_o["wh"][1]), 0, -1))
        self.box_v = _bv
        return {"state": self.state, "pos": self.pos, "vel": self.vel, "rad": self.rad}

    def rel_arrow(self, inc):
        """白箭头口径（`MotionRunner` 会调它 ✓ 见 `lie_motion` 那段 ✓）—— 本档**没有夹取 /
        没有融合** ✗ ⇒ **原样回** ✓（与 `VelocityTracker.rel_arrow` 同一个写法 ✓）。"""
        return inc

    def set_cfg(self, **kw):
        """**当场改判据参数**（演示窗那三格数字框推它 ✓ 与 `set_vtx_k` 同一个套路 ✓）——
        **下一拍就生效** ✓（判据在 `process` 里现读 ✓ 不用重跑 ✓）。

        ⚠ 只认**本来就有**的属性（`hasattr` 白名单 ✓）：这样多递一个键**不炸** ✓（界面加了新旋钮
          忘接线时也只是"没生效" ✓ 而不是 `AttributeError` ⇒ PyQt `abort()` ⇒ **闪退** ✗ 踩过 ✓）。
        """
        for _k, _v in (kw or {}).items():
            if _v is None or not hasattr(self, _k):
                continue
            try:
                setattr(self, _k, float(_v))
            except (TypeError, ValueError):
                continue
        return None

    # ---------------- 出口（给演示窗）----------------

    def motion_viz(self, followed_pos=None):
        """给演示窗的那一份 ✓（键**对齐老出口** ✓ 少一个就可能某处硬取 ⇒ 闪退 ✓ 踩过 ✓）。"""
        del followed_pos
        _trk = []
        for _i in sorted(self._objs):
            _o = self._objs[_i]
            _trk.append({"tid": int(_i), "p": (float(_o["c"][0]), float(_o["c"][1])),
                         "in_view": True, "in_roi": True, "trust": bool(int(_o["miss"]) == 0),
                         "sticky": True, "rel_hist": [], "v": (float(_o["mv"][0]),
                                                               float(_o["mv"][1])),
                         "score": float(_o["score"]), "dev": float(_o["score"]),
                         "hits": int(_o["hits"]), "live": bool(int(_o["miss"]) == 0),
                         "sel": bool(int(_i) == self.tid), "std_area": None,
                         #   ⭐ 本档自己的字段（给"锚定"这套看的 ✓ 见 `draw` ✓）：
                         "g": (float(_o["g"][0]), float(_o["g"][1])),
                         "static": bool(_o["static"]),
                         "mv": (float(_o["mv"][0]), float(_o["mv"][1])),
                         "miss": int(_o["miss"])})
        return {
            "mode": self.mode,
            # ---- 本档真正要画的东西 ✓ ----
            "tracks": _trk,
            "box_v": list(self.box_v),
            "cam_cum": (float(self.cam_cum[0]), float(self.cam_cum[1])),
            "cam_mv": (float(self.cam_mv[0]), float(self.cam_mv[1])),
            "wave": float(self.wave),
            "wave_now": bool(self.wave_now),
            "wave_n": int(self.wave_n),
            "theta": float(self.theta),
            #   ⭐⭐⭐⭐⭐ **目标那格框 —— 由"外观跟踪"给出来** ✗✗（用户 2026-10-08 ✓ 原话：
            #     "**真目标大部分时候都是没有检出框的**" ✓✓）—— ⚠⚠ `box_v` 里**没有它**是常态 ✗
            #     ⇒ 演示窗**必须**用这一个来画目标框（`(cx, cy, w, h)` ✓ **画面坐标** ✓ 相机已加回 ✓），
            #     别再去 `box_v` 里按 id 找 ✗（找不到就画不出来 ✓ 正是用户抱怨的那一幕 ✓）。
            #     ⚠ 还没定目标 / 还没模板 ⇒ `None` ✓（演示窗那时不画 ✓ 不猜 ✗）。
            #   ⚠ 尺寸优先用**这一拍借了几何的那块框**（`_obs_box` ✓ 用户 2026-10-08 ✓ "**这时候真目标
            #     检出框都还在**" ✓✓）—— 有它 ⇒ 画出来就是**他眼里那块框** ✓；没有 ⇒ 退回模板尺寸 ✓。
            #   ⚠ `_obs_box` 是 4 元组、`_tpl_wh` 是 2 元组 ✗ ⇒ 不能拿同一个下标去取（实测 `IndexError` ✓）。
            "target_box": (None if self.pos is None else
                           (float(self.pos[0]), float(self.pos[1]))
                           + ((float(self._obs_box[2]), float(self._obs_box[3]))
                              if self._obs_box else
                              ((float(self._tpl_wh[0]), float(self._tpl_wh[1]))
                               if self._tpl_wh else (0.0, 0.0)))),
            #   ⭐ 这一拍的**模板匹配分**（0~1 ✓）：它是"有没有观测"的唯一判据 ✓（≥ `_ANC_PATCH_NCC` ✓）
            #     ⇒ 面板上写出来，用户一眼能看出"现在是在跟（分高）还是在滑（分低）" ✓。
            "tpl_score": float(self._tpl_score),
            "target_tid": (None if self.tid is None else int(self.tid)),
            "pos_g": (None if self.pos_g is None else (float(self.pos_g[0]),
                                                       float(self.pos_g[1]))),
            # ---- 其余键**对齐老出口**（形状给全 ✓ 老两档那边一个字不变 ✓）----
            "vel": self.vel, "pos": self.pos, "rad": self.rad, "tgt_radius": self.rad,
            "tbox": None, "tbox_rad": None, "tbox_wh": None, "miss": int(self.miss),
            "state": self.state, "raw_pos": self.pos, "median": (0.0, 0.0),
            "cam_len": 0.0, "dev": 0.0, "n_live": len(self.box_v), "n_cands": len(self.box_v),
            "roi": self.roi, "log_text": (self.log_text or ""), "dups": list(self.dups),
            "white": None, "followed_pos": None, "path_pts": [], "clamp": None,
            "clamp_vel": None, "red_i": None, "vel_abs": None, "vel_rel": None, "reg": [],
            "merged": False, "typ_area": None, "sel_score": None, "sel_dev": 0.0,
            "tid": self.tid, "followed": self.tid, "q": None, "q_low": 0, "q_lost": False,
            "dir_conf": [], "dir_top": None, "dir_strength": 0.0, "dir_n": 0,
            "box_moves": [], "box_roles": [], "box_pred": [], "tgt_rel_next": None,
            "tgt_rel_last": None, "pos_rel": [], "cursor_rel": [], "rad_frozen": True,
        }


# ══════════════════════════════════════════════════════════════════════════
# 壳（**只换底盘** ✓ 与 `VelocityRunner` 同一个套路 ✓）
# ══════════════════════════════════════════════════════════════════════════

from perception.lie_motion import MotionRunner                    # noqa: E402


class AnchorRunner(MotionRunner):
    """锚定追踪档的壳 —— 与 `MotionRunner` **同签名同返回** ✓，只把底盘换成 `AnchorTracker` ✓
    ⇒ 演示窗那整套（`warm_tick` / `_show_precomputed` / `draw` / 拖动 / 播放）**一行都不用改** ✓。"""

    def _make_tracker(self, kw):
        #   ⚠ 只透传 `mode` 与 `roi` ✓（本底盘不吃老那批 `q/gate_k/smooth/...` ✓
        #     但**收下不报错** ✓ 见 `AnchorTracker.__init__` ✓）。
        return AnchorTracker(mode=kw.get("mode") or ANCHOR_MODE, roi=kw.get("roi"))

    def _proc_kw(self, i):
        """递帧号（日志用 ✓ 与速度跟踪档同一个约定：**1 起** ✓ 跟画面上的帧号对齐 ✓）。"""
        return {"idx": int(i) + 1}
