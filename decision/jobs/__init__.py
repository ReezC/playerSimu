"""角色动作执行器：爬 / 下跳 / 走 / 传送门 / 跳（2026-10-18 从 `decision/route.py` 拆出）。

每个执行器只回答三件事：**往哪挪、按不按、到没到**（`update()` 的回值）——
不碰按键、不读画面，所以能用假坐标把整条流程跑完（`tools/selftest_decision.py`）。
"""

from decision.jobs import params  # noqa: F401
from decision.jobs.align import align_x_decide
from decision.jobs.base import _PausableJob, stall_s_from_retry_ms
from decision.jobs.climb import ClimbJob
from decision.jobs.drop import DropJob
from decision.jobs.jump import JumpJob
from decision.jobs.params import (ALIGN_TOL_PX, ARM_HOLD_S, DEFAULT_TIMEOUT_S,
                                  INFLIGHT_PHASES, JOB_KINDS, JUMP_START_PX,
                                  MID_ARM_S, MODE_FIXED, NEAR_PX,
                                  OFF_LADDER_GRACE_S, PAUSE_MIN_S, STALL_GAIN_PX,
                                  STALL_S, TAP_ON_S, TAP_PERIOD_S, WALK_TOL_PX)
from decision.jobs.portal import PortalJob
from decision.jobs.walk import WALK_Y_BAND_PX, WalkJob
