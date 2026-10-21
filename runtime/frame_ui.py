"""一帧的第 1 段：**界面状态检测拍**（阶段 1「编排收口」第二刀 ✓ 2026-10-20 ✓）。

搬自 `gui/live_thread.py::_run`（**行为零改动** ✓）：原来那 6 行是

    if time.monotonic() - self._screen_last >= 1.0:
        self._screen_last = time.monotonic()
        self._screen_beat(raw)

⇒ 现在连限流一起搬进来 ✓，`_screen_beat` 本体**一个字没动** ✓（每拍职责 / 状态切换打点 / 报警 /
存帧都还在那个方法里 ✓ —— 那几条的判据见 `selftest_arch::t_screen_beat_*` 与
`selftest_screen_state` ✓）。

⚠ **为什么这里喂的是 `ctx.raw`（原生帧）而不是 `ctx.vis`**：
  `vis` 上已经画了框/线 ⇒ 整帧模板匹配会被我们自己画的像素污染 ✗
  （用户 2026-09-27 那条"框选把虚线框进去"是同一个坑 ✓）。
⚠ **限流 1 秒是刻意的**：弹窗报警不需要帧级延迟 ✓，而整帧匹配 ~0.1 秒 ⇒ 每帧都跑会吃掉推理预算 ✗
  （`_screen_last` 就是这把闸 ✓）。
"""

import time

#: 界面状态检测的限流间隔（秒）—— 见文件开头那句"刻意"✓
SCREEN_EVERY_S = 1.0


def step(th, raw):
    """到点就判一次界面状态 ⇒ `True` = 这一拍真判了（`False` = 被限流跳过 ✓）。

    :param th: `LiveThread`（本段读写的只有 `th._screen_last` 这一个**跨帧**状态 ✓ ——
               它不进 `FrameCtx`：盒子只装"这一帧的东西" ✓ 见 `FrameCtx` 的说明 ✓）
    :param raw: **原生帧**（画框之前那份 ✓ 见文件开头 ⚠）
    """
    now = time.monotonic()
    if now - float(th._screen_last) < SCREEN_EVERY_S:
        return False
    # ⚠ 先记时刻再判：`_screen_beat` 里如果抛异常，也不该让这把闸"卡在上一拍"一直重跑 ✗
    th._screen_last = now
    th._screen_beat(raw)
    return True
