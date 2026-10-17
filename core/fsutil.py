"""落盘小工具：**原子写**（同机多实例的基本保障 ✓ 用户 2026-10-17「多实例」）。

⛔ 为什么不能直接 `path.write_text(...)` ✗：
  · 它是"打开→截断→写" ✗ ⇒ 正在读这份文件的人（**另一个工作台实例** ✓、
    或者崩溃后重启的自己 ✓）有机会读到**半截文件** ✓；
  · 两条写同时发生 ⇒ 互相截断 ⇒ 文件直接坏掉（`yaml` 解析报错 / `json` 解析报错 ✓）。
  · 现场症状很误导：\"配置莫名其妙变空了\"、\"标定丢了\"、\"集合全没了\" ✓。

⭐ 做法：写到**同目录的临时文件**（同一分区 ✓ 才能原子替换 ✓）→ `os.replace` ✓
  —— 替换是原子操作 ✓ 读的人要么看到旧整份、要么看到新整份 ✓ 永远不会看到半截 ✓。
"""

from __future__ import annotations

import os
from pathlib import Path


def write_text_atomic(path, text, encoding="utf-8"):
    """把 `text` 原子地写进 `path`（父目录不存在就建 ✓）。返回 `Path` ✓。"""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:                      # noqa: BLE001 —— 建不出来让下面 open 去报 ✓
        pass
    tmp = p.with_name("%s.tmp%d" % (p.name, os.getpid()))
    try:
        with open(tmp, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
            try:
                f.flush()
                os.fsync(f.fileno())       # 掉电也不留半截 ✓（这几份文件都不大 ✓）
            except Exception:              # noqa: BLE001 —— 某些网络盘 fsync 会失败 ✓
                pass
        os.replace(tmp, p)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:                  # noqa: BLE001
            pass
    return p
