from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """一帧画面及其时间戳。

    时间基准说明（跨机延迟分析的全部依据）：
      - t_recv_wall : B 机接收时刻，time.time()（墙钟，可与 A 机对齐）
      - t_recv_mono : B 机接收时刻，time.perf_counter()（单调，用于本地间隔统计）
      - t_content   : A 机内容生成时刻（由屏幕时间码探针解码得到，A 机墙钟；未启用探针时为 None）
    """

    image: np.ndarray
    frame_id: int
    t_recv_wall: float
    t_recv_mono: float
    t_content: float | None = None
    pts: int | None = None
    meta: dict = field(default_factory=dict)
