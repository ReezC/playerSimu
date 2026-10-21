"""执行层：**只负责把"要发什么"变成真的按键/鼠标**（不含任何策略）。

`input.py`      键位、KeyState、按键/点鼠标、远端 relay、链路健康
`mouse_aim.py`  屏幕比例 → 像素 → 指令单位的换算 + 撞角归零 + 走位 + 左键

⚠ 这一层**不许** import `decision` / `gui` / `perception` ✗
   （判据在 `tools/arch_scan.py` 的 `PKGS`：execution = 1 层 ✓）。
"""
