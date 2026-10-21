# -*- coding: utf-8 -*-
"""「**多久没进展就算卡住**」这一个口径的**唯一实现**（用户 2026-09-27 定的）。

用户原话：「移除卡住判定时长(s)，**统一采用移动操作尝试间隔(ms)**」✓
—— 原来那件 `DecisionSettings.stall_s` 已删 ✓；上爬「爬不动」/ 下爬「等落地」/ 跳跃
「没落到」这些判据现在**都从这一件参数换算** ✓。

## 为什么它住在 `core/`（而不是 `decision/jobs/base.py`）

2026-10-21 现场（用户：「天空项目 3→2 在 L1 上不按 ↑ 了」✗）查出**同一个设置被两处各算一遍**：

* 执行器那侧：`decision.jobs.base.stall_s_from_retry_ms(ms)` —— 带 **0.5 秒下限** ✓；
* 感知那侧（`perception/world/pos_state.py`）：`retain = ms / 1000` —— **没有下限** ✗。

天空项目 `move_retry_ms` = 350 ⇒ 执行器按 0.5 秒算、**广播却按 0.35 秒算** ⇒ 广播先报
"爬不动（`climb_stalled`）" ⇒ `ClimbJob` 走失败逻辑 ⇒ **松 ↑** ⇒ 人往下滑 ⇒ 重来 ⇒
用户看到的就是"**在 L1 上不按 ↑ 了**"✓。
⇒ 换算收成一处（`core` 是两层都能 import 的最底层 ✓），`decision` 那边**保留同名再导出**
（外部三个调用点不用改 ✓），`perception` 直接用它 ✓ —— **不许再各写一份** ✗。
"""

from __future__ import annotations

#: 「移动操作尝试间隔」缺失/非法时的兜底（毫秒）—— 与设置里的默认值一致 ✓
DEFAULT_MOVE_RETRY_MS = 3000

#: 「卡住」判定的**下限**（秒）：再小就等于"刚按一下就判卡住" ✗（用户 2026-09-27 同意 ✓）
STALL_MIN_S = 0.5


def stall_s_from_retry_ms(ms):
    """「移动操作尝试间隔(ms)」→ 那些"多久没进展就算卡住"的判据要的**秒**（下限 0.5 ✓）。

    三个调用点（`agent.start_climb` 灌 job ✓、实时线程的路线解析器 ✓、面板预览 ✓）
    ＋ 感知那侧的位置状态机（`pos_state` ✓）**都走它** ⇒ 免得几处各写一份钳法、
    以后又分叉 ✗。
    """
    try:
        n = int(ms)
    except (TypeError, ValueError):
        n = DEFAULT_MOVE_RETRY_MS
    return max(STALL_MIN_S, n / 1000.0)
