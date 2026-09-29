# -*- coding: utf-8 -*-
"""**位置状态机** —— 「位置状态」的**唯一写者**（用户 2026-09-27 定：

"开始进行我说的**位置状态广播**架构（**所有的位置状态更新由位置状态机自治**）"）。

## 位置状态是什么（口径只此一处）

玩家在**地形语义**上"站在哪"的**粗粒度快照**，加上由它派生出来的**本拍事实**：

| 字段 | 含义（**中文口径**，别只报字段名 ✗） | 谁读 |
|---|---|---|
| `here_sets` | 脚下那个 foothold 属于的**全部命名集合**（站没圈的地方 = 空集 ✓） | 决策层（路线终点、区域筛选、任务起点 ✓）与界面 ✓ |
| `here_span` | 脚下这些集合横着占的 **x 范围** `(左, 右)`（非墙 foothold 的并集 ✓）—— "我这块平台有多宽" ✓ | `agent._reachable_without_path`（"够得着就不用下寻路任务"那条豁免 ✓） |
| `ladder_id` | 现在**贴在哪根绳**上（`"L1"` 这种稳定编号；不在绳上 = `None`）—— ⚠ **带许可**：许可的入口是「**climb 执行器按住 ↑/↓ 后发起通知**」（2026-09-28 诉求 1：不再是"按住 ↑"这个本机按键状态 ✓）| 决策层（上绳/下爬执行器、"掉下来了" ✓）与界面 ✓ |
| `on_rope_pos` | ⭐ **只看位置**编出来的绳号（**不管按没按 ↑/↓** ✓）—— 治的是那条**循环依赖**：执行器自己会松键 ⇒ 带许可的 `ladder_id` 当场变空 ⇒ 执行器"自己把自己判成没上绳" ✗ | `ClimbJob`（"位置还在绳段里"那条记忆 ✓）与 `agent`（"攀爬进行中不打架" ✓） |
| `at_ladder_top` | **到没到某根绳的上端**（绳号）—— 上爬的"到达"判据 ✓ | `ClimbJob` ✓（**它不再自己比坐标** ✗） |
| `at_ladder_bottom` | **到没到某根绳的下端**（绳号）—— 下爬的"到达"判据 ✓ | `ClimbJob` ✓（同上 ✗） |
| `ground_y` | **脚下那块面的 y**（在该 foothold 上、玩家 x 处取的 y ✓）—— 下爬"落地面"的判据 ✓ | `ClimbJob` ✓ |
| `note` | 一句话原因/状态（给界面/日志） | ✓ |

**两个容差是不同的东西，报数必须带名字**（2026-09-27 用户要求进纪律 ✓）：

* `align_tol_px` = 设置里的「**坐标对齐误差范围**」（默认 **10** 世界像素）—— 执行器判
  "**x 对齐好了没有**"用的容差 ✓（`ClimbJob.tol_px` 就是它 ✓，"对齐了就按跳"那条 ✓）；
* `mapdata.LADDER_DX` = **24** —— **感知**判"**算不算在绳上**"的 x 半宽 ✓（比上面那把尺宽 ✓）。
  ⚠ 口径是 `LADDER_DX >= align_tol_px`（宽的不许比窄的更紧 ✓，否则同一次读数会出现
  "执行器说对齐好了、位置状态说没在绳上" ✗，实测踩过 ✓）。

## 为什么要有它（要解决的问题）

1. 以前"位置状态"被写在两个模块里（`perception.minimap.apply_to_player` 写世界坐标、
   `gui.live_thread._fill_route_ctx` 写 `here_sets`/`ladder_id`），而**执行器还会自己再算一遍**
   （`ClimbJob._at_rope` / `_arrived` 比坐标、`agent._reachable_without_path` 比区间 ✗）——
   同一件事两处算 ⇒ 只要两边的尺不一样就会互相打架（2026-09-27 就连着踩了两次：
   `ALIGN_TOL_PX` 与 `LADDER_DX` 边界相反、"在不在绳上"四处各算一遍 ✗）。
2. 现在：**读数和按键许可进来，状态算好、广播出去；执行器只读不算** ✓。

## 硬边界（别越过）

* 机器**只吃**"**climb 执行器按住 ↑/↓ 后发起的通知**"这个**事实**（调用方去问
  `agent.climbing_vertical()` ✓ —— 2026-09-28 诉求 1 起改用它，替代原来"本机按没按 ↑/↓"
  （`holding_vertical` 看 `KeyState.pressed`）✗：本机按键状态跟"执行器发键 → 远端真爬"之间
  有延迟/丢键，会跟执行器意图飘开 ✓），**自己不下发按键、也不做按键记账** ✗ —— 那是执行器
  的事（`ClimbJob._out` ✓）。这条界线就是"世界事实"归状态机、"控制/按键记账"归执行器 ✓
  （用户 2026-09-27 认可的分法）。
* 时间一律用 **`time.monotonic()`**（与定位同源 ✓）；**别**混用 `perf_counter` ✗。
* 容差/时长**只用现成的两个设置**：`align_tol_px`（「坐标对齐误差范围」✓，默认 10）与
  `move_retry_ms`（「移动操作尝试间隔」✓，默认 3000）—— **不新造常数** ✓。
* 几何**只在机器里算一次** ✓：执行器/决策层要"我脚下这块平台多宽""我在不在绳段里"，
  一律读这里的广播 ✗（**别再自己比一遍** ✗ —— 那是 2026-09-27 连踩两次的根因 ✓）。
"""
from __future__ import annotations

import dataclasses
import time

#: 「到顶 / 到底」判据里"不再沿绳移动"要观察多久 —— **就是「移动操作尝试间隔」**，不新造 ✓。
#: （见 `update` 的说明：用户原话"y 在『移动操作尝试间隔」时间内不再变变小，则判定到顶" ✓）


@dataclasses.dataclass(frozen=True)
class PosSnapshot:
    """**一拍的「位置状态」广播快照** —— `PositionStateMachine.update()` 的返回值 ✓。

    为什么要**一个对象**，而不是以前那串 7 元组（用户 2026-09-27："重新整理寻路执行器与
    **状态通信**，优化代码结构，提高内聚降低耦合" ✓）：

      · 以前：机器返回 7 元组 ⇒ 调用方**按位置解包**（`live_thread` 写 `Player` ✓），
        执行器侧再由 `agent` 一个个 `getattr` 拆成 kwargs ✗ ⇒ 加一个字段要动三处、
        顺序错了还不报错 ✗（这一轮就踩过"字段顺序对不上" ✗）；
      · 现在：机器算完给**它** ✓；`Player.pos` 存**它** ✓；执行器 `**pos.as_kwargs()` 拿 ✓
        ⇒ **加字段只动这一个类** ✓。

    字段的中文口径见模块 docstring 那张表 ✓（**报数必须带中文名** ✗ —— 项目纪律 ✓）。
    ⚠ 它是**不可变**的（`frozen=True` ✓）：谁都不许就地改广播 ✗（要改只能让机器重新算 ✓）。
    """

    here_sets: tuple = ()
    ladder_id: str | None = None
    at_ladder_top: str | None = None
    at_ladder_bottom: str | None = None
    ground_y: float | None = None
    on_rope_pos: str | None = None
    here_span: tuple | None = None
    climb_failed: bool = False
    climb_stalled: bool = False

    # ---------------- 便捷判据（**只读快照，不含任何几何** ✓）----------------

    def on_rope(self, lid):
        """广播说"人正贴着**这根绳**"吗（`lid` = 本任务那根绳 ✓；带按键许可 ✓）。"""
        return bool(self.ladder_id and lid and str(self.ladder_id) == str(lid))

    def at_rope_end(self, lid, top=True):
        """广播说"已到**这根绳**的**上端/下端**"吗（`top=False` 看下端 ✓）。"""
        v = self.at_ladder_top if top else self.at_ladder_bottom
        return bool(v and lid and str(v) == str(lid))

    def in_rope_span(self, lid):
        """广播说"**只看位置**也在**这根绳**的绳段里"吗（与按键许可无关 ✓）。"""
        return bool(self.on_rope_pos and lid and str(self.on_rope_pos) == str(lid))

    @classmethod
    def of(cls, obj):
        """从**任何一个带那七个字段的东西**上拼一份快照 ✓（`Player` ✓，也用例里的替身 ✓）。

        ⚠ 口径：**扁平字段是唯一存储**（`Player.here_sets` / `ladder_id` / … ✓，机器每拍写它们 ✓，
          用例也直接摆它们 ✓）；要"整份"时**只在这里**拼一次 ✓ —— 这就把以前散在
          `agent` / 执行器里的那堆 `getattr` 拆字段收成了一处 ✓（用户要求"提高内聚降低耦合" ✓）。
        """
        sets = getattr(obj, "here_sets", None) or ()
        return cls(here_sets=tuple(str(s) for s in sets),
                   ladder_id=getattr(obj, "ladder_id", None),
                   at_ladder_top=getattr(obj, "at_ladder_top", None),
                   at_ladder_bottom=getattr(obj, "at_ladder_bottom", None),
                   ground_y=getattr(obj, "ground_y", None),
                   on_rope_pos=getattr(obj, "on_rope_pos", None),
                   here_span=getattr(obj, "here_span", None),
                   climb_failed=bool(getattr(obj, "climb_failed", False)),
                   climb_stalled=bool(getattr(obj, "climb_stalled", False)))

    def as_kwargs(self):
        """→ 执行器 `update(...)` 的那几个关键字参数 —— **唯一一处映射** ✓。

        ⚠ 执行器家族的 ABI 就靠它统一（四个执行器**都**收这几个名字 ✓，用不上的不看 ✓）
          ⇒ `agent` 再也**不用按类名挑参数** ✗（那是这一轮之前的耦合面 ✓）。
        """
        return dict(ladder_id=self.ladder_id, here_sets=list(self.here_sets),
                    at_top=self.at_ladder_top, at_bottom=self.at_ladder_bottom,
                    ground_y=self.ground_y, on_rope_pos=self.on_rope_pos,
                    climb_failed=self.climb_failed, climb_stalled=self.climb_stalled)

    def fmt(self):
        """一句话（日志/界面 ✓）：三态要看得出来（`判不了` / `否` / 绳号 ✓）。"""
        return ("集合=%s　绳梯=%s　到顶=%s　到底=%s　脚下y=%s　只看位置=%s　平台x=%s"
                % (("、".join(self.here_sets) or "—"), self.ladder_id or "不在绳上",
                   self._tri(self.at_ladder_top), self._tri(self.at_ladder_bottom),
                   ("%.0f" % self.ground_y) if self.ground_y is not None else "—",
                   self.on_rope_pos or "—",
                   ("%.0f~%.0f" % self.here_span) if self.here_span else "—"))

    @staticmethod
    def _tri(v):
        return "判不了" if v is None else (v or "否")


class PositionStateMachine:
    """见模块 docstring ✓。**有状态**（跨帧记忆"y 到过的极值"），每拍 `update` 一次 ✓。"""

    def __init__(self):
        #: 每根绳「y 到过的**最小值** + 它**最后一次变小**的时刻」—— "N 秒内没再变小"要用 ✓。
        #: ⚠ 只记**当前正贴着的那根**（换绳就重来 ✓）：留着上一根的极值会让"刚贴上去"当场判到顶 ✗。
        #: ⭐ **「沿用上一次位置状态」的记忆**（用户 2026-09-28：「不许存在没站在平台上这种
        #: 空类，如果找不到就按上一个位置状态」✓）—— 只在**真的判出来过**时才更新 ✓，
        #: 判不出来时由 `_broadcast` 把它们填回去 ✓。⚠ 换图要 `reset()` 清掉 ✗（见那里 ✓）。
        self._last_here = []
        self._last_span = None
        #: 这一拍广播的 `here_sets` 是不是**沿用**来的（给界面/排查用 ✓；同图内抖一下很正常 ✓）。
        self.here_sticky = False
        # ⚠ 上爬**不再**记「y 到过的最小值 + 它最后一次变小的时刻」了（用户 2026-09-28 ✓
        #   原话："移除这一项『y 在「移动操作尝试间隔」内不再变小』，用『广播"到顶"之后还会
        #   再按住 ↑ 250ms』仅此一项兜底"）⇒ 判到顶只看"y ≤ 绳顶 + 容差"✓。
        #: 每根绳「y 到过的**最大值** + 它**最后一次变大**的时刻」—— 下爬到**下端**要用 ✓
        #: ⚠ **只保留下爬这一份**（上爬那份已按用户要求删掉 ✗ 别再加回来）。
        self._best_bot = {}
        #: 广播字段（读 `update` 的返回值；也留着方便用例直接看 ✓）
        self.here_sets = []
        self.here_span = None
        self.ladder_id = None
        self.on_rope_pos = None
        self.at_ladder_top = None
        self.at_ladder_bottom = None
        self.ground_y = None
        self.climb_failed = False
        self.climb_stalled = False
        #: 「补发↑」判据要用的：每根绳「y 上一次**变化 ≥ 容差**的时刻」（见 `update` ③c ✓）
        self._moved = {}
        self.note = ""
        #: `foothold_id -> 脚下那块面的 y` 的小缓存（key 里带 `id(terrain)`：
        #: 换地图 ⇒ terrain 换了 ⇒ 缓存自动失效 ✓，不会拿上一张图的数骗人 ✗）
        self._fy = {}
        #: 这一拍玩家的世界 x（取"脚下那一点的面 y"要用 ✓）
        self._last_x = 0.0

    # ---------------------------------------------------------------- 每拍

    def update(self, now, *, loc, terrain, zones, hold_vert,
               align_tol_px=10.0, move_retry_ms=3000.0, span_of=None):
        """喂这一拍的读数 → 算出并**广播**位置状态。

        返回 `(here_sets, ladder_id, at_ladder_top, at_ladder_bottom, ground_y,
        on_rope_pos, here_span)`（顺序见下 ✓；老调用方只取前三个也照样能跑 ✓）。

        `loc` 是 `PlayerLocator.update` 那份 dict（要有 `world_x` / `world_y` / `foothold_id` ✓）。
        `hold_vert` = 这一拍**按没按 ↑/↓**（事实 ✓，由调用方提供 ✓）。
        `span_of` = "集合名 → `(左, 右)`"的**回调查询**（调用方给 ✓ —— 机器在感知层，
          **不 import 决策层** ✗；`gui.live_thread` 那边本来就有这个闭包 ✓）。
        判据：
          ⭐ **上端（到顶）**：「世界坐标 `y ≤ 绳梯上端的 y + 「坐标对齐误差范围」`」⇒ 判**到顶** ✓
          —— 2026-09-28 用户**删掉了原来并列的第二条**「`y` 在「移动操作尝试间隔」时间内
          **不再变小**」✗（原话："移除这一项『y 在「移动操作尝试间隔」内不再变小』，用『广播
          "到顶"之后还会再按住 ↑ 250ms』**仅此一项**兜底"）。那条要**干等**「移动操作尝试
          间隔」（默认 **3000ms**）才认到顶 ⇒ 现场就是"人早停在绳顶了、状态还写着没到"✗。
          ⚠ 误判的兜底**不在感知层**：执行器判到到达后**不立刻松手**，会再按住 ↑
            `base_hold_ms`（=「坐标对齐误差时间」，默认 **250ms**）才收工 ✓
           （`decision/route.py` 的 `ClimbJob._arrived_hold` ✓）—— 那一步本来就是"迈上平台"
            要按着 ↑ ✓ ⇒ **一箭双雕**：既是游戏机制，又接住了"y 抖一下误判到顶" ✓。
          **下端（到底）**：仍然**两条**：「`y ≥ 绳梯下端的 y - 「坐标对齐误差范围」`」**且**
          「`y` 在「移动操作尝试间隔」时间内**不再变大**」⇒ 判**到底** ✓
          （⚠ 下爬这次**没动** —— 用户只点了上爬那一项 ✓ 要动说一声）。
        ⚠ 判不了（这一拍没有世界坐标 / 没有地形）⇒ **一律给空**（`here_sets=[]`、
          `ladder_id=None`、`at_ladder_top=None`、… ✓）—— 绝不留旧值骗执行器 ✗。
        """
        now = time.monotonic() if now is None else float(now)
        x = loc.get("world_x") if loc else None
        y = loc.get("world_y") if loc else None
        if x is None or y is None or terrain is None:
            self._clear()
            self.note = "没有世界坐标读数（定位没输出）⇒ 位置状态清空"
            return self._broadcast()

        self._last_x = float(x)                 # "脚下那一点的面 y" 要用 ✓
        tol = max(0.0, float(align_tol_px or 0.0))   # 提前：climb_failed / climb_stalled 也要用 ✓
        # ① 语义：脚下 foothold 属于哪些集合（**一处口径**：`core.zones.set_of` ✓）
        fid = str((loc or {}).get("foothold_id") or "")
        self.here_sets = list(zones.set_of(fid)) if (zones is not None and fid) else []
        # ①' 脚下这些集合横着占多宽（"我这块平台有多宽" ✓）—— 交给调用方给的解析器 ✓
        self.here_span = self._span_of_sets(span_of)
        # ①'' 脚下那块面的 y（下爬"落地面"的判据 ✓；判不出来 None ✓）
        self.ground_y = self._ground_y(terrain, fid)

        # ② 绳：
        #   · `ladder_id`（**带许可** ✓）：许可的入口是「climb 执行器按住 ↑/↓ 后发起的通知」
        #     （2026-09-28 诉求 1：`hold_vert` 现在由 `agent.climbing_vertical()` 给，= 执行器
        #     上一拍发了 ↑/↓，替代原来"本机按没按 ↑/↓" ✗）—— 防"走路碰到绳子就被判在绳上 ⇒
        #     卡住不走" ✓（那条防线还在：走路时执行器不发 ↑/↓ ⇒ 通知就是假 ⇒ 不算在绳上 ✓）。
        #   · `on_rope_pos`（**只看位置** ✓）：不管按没按键 —— 执行器自己会松键 ⇒ 拿带许可的
        #     那份会把"我自己松的键"当成"没上绳"（**循环依赖** ✗，用户 2026-09-27 报的"执行器
        #     还说没上"就是它 ✓）。
        lad_pos = self._ladder_at(terrain, x, y)
        self.on_rope_pos = self._lid_of(terrain, lad_pos)
        self.ladder_id = self._lid_of(terrain, lad_pos if hold_vert else None)
        # ②' 「攀爬失败」(2a) **挪到 ③c 之后才算**（见下面那段 ✓）：它还要看"y 僵没僵"，
        #   而那个窗口（`_moved`）到 ③c 才建好 ✗（2026-09-28 修 ✓）。先按"不算"起步 ——
        #   中途走"不在绳上 ⇒ 早退"那支时也不会留旧值 ✓。
        self.climb_failed = False

        top = bot = None
        if lad_pos is not None:                 # ⚠ 极值用**真贴上哪根**算，与按键许可无关 ✓
            top = min(float(lad_pos.y1), float(lad_pos.y2))
            bot = max(float(lad_pos.y1), float(lad_pos.y2))

        # ③ 「到顶 / 到底」三态（`ClimbJob._arrived` 就靠这个区别 ✓）：
        #   `None` = **判不出来**（这一拍没有读数 / 不在任何绳段里 ⇒ 调用方可以走自己的兜底 ✓）；
        #   `""`   = **判过了，没到**（⇒ 执行器**不许再自己比坐标** ✗）；
        #   `"L2"` = 已到那根绳的那一端 ✓。
        self.at_ladder_top = ""
        self.at_ladder_bottom = ""
        if not self.ladder_id:
            self._best_bot.clear()              # 不在绳上（或没按 ↑/↓）⇒ 极值重来（下一根绳从零算 ✓）
            self._moved.clear()
            self.climb_failed = False
            self.climb_stalled = False
            self.note = ""
            return self._broadcast()
        lid = self.ladder_id
        if list(self._best_bot) != [lid]:       # 换了根绳（或刚贴上来）⇒ 只留当前这根 ✓
            self._best_bot = {}
        if list(self._moved) != [lid]:
            self._moved = {}
        retain = max(0.0, float(move_retry_ms or 0.0)) / 1000.0
        # ③a 上端：⭐ **只看"够不够高"**（用户 2026-09-28 ✓ 原话："移除这一项『y 在「移动操作
        #   尝试间隔」内不再变小』，用『广播"到顶"之后还会再按住 ↑ 250ms』**仅此一项**兜底"）
        #   ⇒ 从这里**删掉**了原来的 `(now - b[1]) >= retain` —— 那一项要**干等**
        #   「移动操作尝试间隔」（默认 3000ms）才认到顶 ⇒ 现场就是"人早停在绳顶了、状态还写着
        #   没到"（用户 2026-09-28 问的"攀爬到顶后还发呆一段时间"✗ 主因就是它 ✓）。
        #   ⚠ 误判（y 抖一下就够高）的兜底**不在这一层**：执行器判到到达后**不立刻松手**，
        #     会再按住 ↑ `base_hold_ms`（「坐标对齐误差时间」，默认 250ms）才收工 ✓
        #     （`ClimbJob._arrived_hold` ✓）—— 那本来就是"迈上平台"要按着 ↑ ✓ **一箭双雕** ✓。
        #   ⚠ 所以这里**故意不再记 y 极值**（上面那份 `_best` 随之删掉 ✓ 死数据不留 ✗）。
        if top is not None and float(y) <= top + tol:
            self.at_ladder_top = lid
        # ③b 下端：y 越**大**越靠下 ⇒ 记"y 到过的最大值 + 最后一次变大的时刻" ✓（对称 ✓）
        bb = self._best_bot.get(lid)
        if bb is None or float(y) > bb[0]:
            self._best_bot[lid] = (float(y), now)
            bb = self._best_bot[lid]
        if bot is not None and float(y) >= bot - tol and (now - bb[1]) >= retain:
            self.at_ladder_bottom = lid
        # ③c 「补发↑」(2c)：y 在「移动操作尝试间隔」内**变化没超过容差** ⇒ 卡住 ⇒
        #   请执行器**补发按住 ↑**（`climb_stalled` ✓）。到顶了就不再算卡住 ✓。
        m = self._moved.get(lid)
        if m is None or abs(float(y) - m[0]) >= tol:
            self._moved[lid] = (float(y), now)
            m = self._moved[lid]
        _stuck = bool((now - m[1]) >= retain)   # y 在这个窗口里**一次都没动**（"僵着"✓）
        self.climb_stalled = bool(not self.at_ladder_top and _stuck)
        # ②' 「攀爬失败」(2a)：贴着绳（`lad_pos` = **宽尺** `LADDER_DX`=24 ✓）但 x **偏出
        #   「坐标对齐误差范围」**（`tol` ✓），**而且僵着不动**（上面 `_stuck` ✓）
        #   ⇒ 这才是真的"抓空 / 贴着绳磨" ⇒ 失败重来 ✓。
        #   ⛔ 2026-09-28 修：原来**只看前两条**（`lad_pos` 非空 + x 偏）⇒ 把**斜跳全杀了** ✗：
        #   斜跳是"离绳 20~80px 起跳、朝绳飞 + 按住 ↑"，飞过去时 x 从 70 收到 0，**必然穿过
        #   `tol`(10) < |dx| ≤ `LADDER_DX`(24) 那一段**，而那时人**已经吸上绳、y 正在上升**
        #   （实测 `106010105`：dx=21、y 从 -174 升到 -293、`ladder_id` 已判 "L2" ✓）⇒ 只看
        #   x 就是"**每跳必杀**"（用户 2026-09-28 报的"**斜向跳上绳全部失败**"✗）—— 它跟
        #   "算不算在绳上"那把**宽**尺口径打架，正是用户 2026-09-27 踩过的同款坑（两把尺
        #   边界相反 ✗）的翻版 ✓。⚠ 原地跳不中招：它跳之前已经对齐到 `|dx| ≤ tol` ✓。
        #   ⇒ 加"**僵着**"这一条：飞过（y 在动）⇒ 不算 ✓；贴歪了磨绳（y 不动、超过
        #   「移动操作尝试间隔」）⇒ 才算 ✓ —— 用户要的"抓空 ⇒ 失败重来"一个字没丢 ✓。
        self.climb_failed = bool(lad_pos is not None
                                 and abs(float(x) - float(lad_pos.x)) > tol
                                 and _stuck)
        if self.at_ladder_top:
            self.note = ("位置状态：已到 %s 上端（y=%.0f ≤ %.0f+%.0f，"
                         "且 y 在「移动操作尝试间隔」%.0f ms 内没再变小）"
                         % (lid, float(y), top, tol, retain * 1000.0))
        elif self.at_ladder_bottom:
            self.note = ("位置状态：已到 %s 下端（y=%.0f ≥ %.0f-%.0f，"
                         "且 y 在「移动操作尝试间隔」%.0f ms 内没再变大）"
                         % (lid, float(y), bot, tol, retain * 1000.0))
        else:
            self.note = ""                      # 判过了、没到 ⇒ 两个字段保持 `""` ✓
        return self._broadcast()

    # ---------------------------------------------------------------- 内部

    def reset(self):
        """⭐ **换图 / 换项目时清掉"沿用上一次"那份记忆**（2026-09-28 ✓）。

        ⚠ 为什么必须有它：下面 `_broadcast` 会把"上一次判出来的 `here_sets`"**沿用**下去
          （见那段说明 ✓）—— 同图内正确 ✓，但**换图之后**再沿用就是拿**上一张图的集合名**
          当"我现在站这儿" ✗✗（比空着还糟：会去"回"一个本图不存在的集合 ✓）。
        ⇒ 调用方（`gui.live_thread._fill_route_ctx` ✓）在**发现图号变了**时叫一声 ✓。
        """
        self._last_here = []
        self._last_span = None

    def _broadcast(self):
        """`update` 的返回：**一整个快照对象** ✓（字段口径见 `PosSnapshot` ✓）。

        ⚠ 以前是一串 7 元组（按位置解包 ✗）—— 2026-09-27 起改成这个对象 ✓：
          加字段只动 `PosSnapshot` 一处 ✓（用户要求"提高内聚、降低耦合" ✓）。

        ⭐⭐ **位置状态容错：不许存在"没站在平台上"这种空类**（用户 2026-09-28 明确要求 ✓
        原话："位置状态给容错：**不许存在没站在平台上这种空类**，如果找不到就**按上一个
        位置状态**"✓）：
          · 起因（现场事故 ✓）：判不出脚下集合 ⇒ `here_sets` 空 ⇒ 「战斗区域」规则认为
            "不在能打区" ⇒ 每拍下「回能打区」⇒ 而**那条路造不出来**（`task_plan_fail`）⇒
            **每 3 秒重试、永远失败** ⇒ 人在原地**彻底卡死** ✗✗（`behavior.log` 里
            `zone_goto_fail` 连续刷 ✓）。
          · 所以这里兜一层：**判不出来 ⇒ 沿用上一次判出来的那份** ✓（同图内，上一拍的位置
            就是最好的估计 ✓）；`here_span` 一起沿用 ✓（它由 `here_sets` 派生 ✓ 口径一致 ✓）。
        ⚠⚠ **换图必须清**（见 `reset()` ✓）—— 那里的判据由调用方给（它才知道图号 ✓）。
        ⚠ 只兜 `here_sets` / `here_span` **这两个** ✗：`ladder_id` / `at_ladder_top` /
          `ground_y` 那些是"**这一拍的事实**"，拿旧值骗执行器会出真错（同 `_clear` 的纪律 ✓）。
        """
        if self.here_sets:
            self._last_here = list(self.here_sets)      # 记下"最近一次真的判出来了" ✓
            self._last_span = self.here_span
            self.here_sticky = False
        elif self._last_here:
            self.here_sets = list(self._last_here)      # ⭐ 找不到 ⇒ 按上一个位置状态 ✓
            if self.here_span is None:
                self.here_span = self._last_span
            self.here_sticky = True                     # 给界面/日志：这份是**沿用**来的 ✓
        return PosSnapshot(here_sets=tuple(self.here_sets),
                           ladder_id=self.ladder_id,
                           at_ladder_top=self.at_ladder_top,
                           at_ladder_bottom=self.at_ladder_bottom,
                           ground_y=self.ground_y,
                           on_rope_pos=self.on_rope_pos,
                           here_span=self.here_span,
                           climb_failed=self.climb_failed,
                           climb_stalled=self.climb_stalled)

    def _clear(self):
        self.here_sets, self.here_span = [], None
        self.ladder_id, self.on_rope_pos = None, None
        self.at_ladder_top, self.at_ladder_bottom = None, None
        self.ground_y = None
        self.climb_failed = False
        self.climb_stalled = False
        self._best_bot.clear()
        self._moved.clear()

    @staticmethod
    def _ladder_at(terrain, x, y):
        """这一拍世界坐标落在哪根绳的**绳段**里（`terrain.ladder_at` = **感知那把尺**：
        x 半宽 `mapdata.LADDER_DX`(**24**) + y 上下各 `LADDER_PAD`(**12**) ✓）。

        ⚠ **不看按键许可**（那是调用方在该不该认 `ladder_id` 时的事 ✓）：
        执行器要判"位置还在不在绳段里"，必须有一份**与按键无关**的事实 ✓。
        """
        try:
            return terrain.ladder_at(float(x), float(y))
        except Exception:                       # noqa: BLE001 —— 算不出来就当不在绳段里 ✓
            return None

    @staticmethod
    def _lid_of(terrain, lad):
        """绳对象 → 稳定绳号（`core.zones.ladder_ids` ✓）；给不了号就 `None` ✓。"""
        if lad is None:
            return None
        try:
            from core import zones as zones_mod

            return zones_mod.ladder_ids(terrain).get(id(lad)) or None
        except Exception:                       # noqa: BLE001 —— 编不出号就当不在绳上 ✓
            return None

    def _span_of_sets(self, span_of):
        """脚下这些集合的 **x 范围**（`(左, 右)`）：每个集合问 `span_of`、取包络 ✓。

        ⚠ 口径来自调用方给的解析器（`gui.live_thread` 那条 = `route.set_span` ✓，
          "集合里有哪些**非墙** foothold"只此一处 ✓）—— 这里**不自己再算一遍** ✗。
        """
        if not span_of or not self.here_sets:
            return None
        los, his = [], []
        for name in self.here_sets:
            try:
                sp = span_of(str(name))
            except Exception:                   # noqa: BLE001
                continue
            if sp:
                los.append(float(sp[0]))
                his.append(float(sp[1]))
        if not los:
            return None
        return (min(los), max(his))

    def _ground_y(self, terrain, fid):
        """**脚下那块面的 y**：在这块 foothold 上、按**玩家 x** 取的面 y ✓。

        ⚠ 与 `route._set_spans` 的"面 y"**不是同一件事**（那里取的是 foothold 中点的 y ✓，
          给"这块平台多宽/多高"用 ✓）；这里是"**我脚下这一点**"的面 y ⇒ 下爬判"落到
          目标平台的面"要靠它 ✓（斜坡上两者会差，取玩家 x 才对得上脚底 ✓）。
        判不出来（没读数 / 这块 id 不在地形里）⇒ `None` ✓（**别拿 0 骗人** ✗）。
        """
        if not fid or terrain is None:
            return None
        key = (id(terrain), str(fid))
        hit = self._fy.get(key)
        if hit is not None:
            return hit
        for f in getattr(terrain, "footholds", ()) or ():
            if str(f.fid) == str(fid):
                try:
                    val = float(f.y_at(float(self._last_x)))
                except Exception:               # noqa: BLE001 —— 取不到就退回面的中点 y ✓
                    val = float(f.y_at((float(f.left) + float(f.right)) / 2.0))
                self._fy[key] = val
                return val
        return None

    # ---------------------------------------------------------------- 给界面/日志

    def status(self):
        """一句话状态（日志/调试用）：集合 / 平台宽度 / 绳梯 / 到顶 / 到底 / 脚下 y ✓。"""
        return ("位置状态：集合=%s　平台x=%s　绳梯=%s　到顶=%s　到底=%s"
                "　脚下y=%s%s"
                % (("、".join(self.here_sets) or "（没圈进任何集合）"),
                   ("%.0f~%.0f" % self.here_span) if self.here_span else "—",
                   self.ladder_id or "不在绳上",
                   self.at_ladder_top or "否", self.at_ladder_bottom or "否",
                   ("%.0f" % self.ground_y) if self.ground_y is not None else "—",
                   ("　" + self.note) if self.note else ""))
