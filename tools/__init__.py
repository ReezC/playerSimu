"""自检与工具集。

注意分工：
  - clock_server.py / probe_gen.py  → 在 A 机（仿真机）上运行
  - probe_recv.py / clock_sync.py / record.py → 在 B 机（工作机）上运行
  - probe_codec.py / config.py      → 两机共用
"""
