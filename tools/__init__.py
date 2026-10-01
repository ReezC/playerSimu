"""自检与工具集。

注意分工：
  - clock_server.py / probe_gen.py  → 在 A 机（仿真机）上运行
  - probe_recv.py / clock_sync.py / record.py → 在 B 机（工作机）上运行
  - probe_codec.py                  → 两机共用
  - arch_scan.py                    → 架构风险扫描（只读，谁都可以跑）

⚠ 本目录只放「命令行工具 + 自检 + 两机共用的协议编解码」—— **库代码一律进
core/**（2026-09-29 起强制：`config.py` 已搬去 `core/config.py`，gui / perception /
core 都不再低头 import tools ✓ 分层规则见 docs/交接.md §1）。
"""
