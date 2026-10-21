"""测谎：**被判者小游戏**的识别与跟踪（独立成包 ✓ 2026-10-17）。

`lie_controller.py`  鼠标/面板驱动的薄封装
`lie_group.py`       同组运动模型
`lie_motion.py`      真假目标的运动判据（最大，7239 行）
`lie_registry.py`    形状/面板登记表
`lie_tracker.py`     跟踪主体（5599 行）

⚠ 允许依赖 `core.geom` / `core.kf`（同层 1 ✓ 尺子/滤波器是通用件 ✓）；
  **不许** import `decision` / `gui` / `tools` ✗（判据见 `tools/arch_scan.py::PKGS`）。
"""
