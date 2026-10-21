"""寻路规划：代价 → 选边 → 造任务（2026-10-18 从 `decision/route.py` 拆出）。

**一个坐标系：世界坐标**（面板像素只用于显示）。改寻路口径只能改这里，
`jobs/` 不许反向依赖本包。
"""

from decision.pathing.cost import FALL_DISCOUNT, _vert_cost, hop_cost
from decision.pathing.plan import (drop_job_for_edge, dst_center_x, dst_surface_y,
                                   job_for_edge, job_label, jump_job_for_edge,
                                   path_cost, pick_edge, plan_jobs,
                                   portal_job_for_edge, walk_job_for_edge)
from decision.pathing.spans import _set_spans, _x_span_gap, set_box, set_span
