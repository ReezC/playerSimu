"""Windows 上的 **DLL 预加载**：必须在**导入 PyQt5 之前**调用（2026-10-09 ✓ 抽出来的 ✓）。

**为什么**
    PyQt5/Qt5/bin/ 里自带一整套 MSVC 运行时（`msvcp140.dll` / `vcruntime140.dll` … ✓）。
    它一旦先加载，那套**旧版本**就被钉在进程里 ✗ ⇒ 之后 torch 的 `c10.dll` 要链接系统里的
    新版运行时 ⇒ **`WinError 1114`（DLL 初始化例程失败）** ✗。表现极具迷惑性：
    **同一个训练脚本，命令行能跑，在 GUI 里必失败** ✓（2026-09-29 现场 ✓）。

⚠⚠ **更狠的一层**（2026-10-09 ✓ 实测，也就是把它从 `gui/app.py` 抽出来单独成模块的原因）：
    在**已经加载了 PyQt5 / cv2** 的进程里再 `import torch`，不只是抛 1114 ——
    它抛完**当场 access violation** ✗（`0xC0000005`，**连 Python traceback 都没有** ✓）
    ⇒ 自检套件"跑着跑着突然死掉、连失败清单都打不出来"就是这么来的 ✓
    （`selftest_decision` 卡在 `live_thread._limit_cpu_threads()` 里那句 `import torch` ✓）。
    ⇒ 结论：**先把系统那份拉起来**（`ctypes.CDLL` 不带路径 ⇒ 走系统搜索顺序 ✓），
      PyQt5 之后就会复用它、不再加载自己那份 ✓。代价约 **1 ms**（比"提前 import torch"
      那 3~5 秒便宜得多 ✓，而且不必每次启动都付 ✓）。

**谁要调**（越早越好 ✓ 就在**第一个** `import PyQt5` 之前 ✓）：
    · `gui/app.py` —— 工作台入口 ✓；
    · 会建 Qt 控件、又可能碰到 torch 的自检套件（`tools/selftest_decision` ✓）。
"""

import os

#: 要抢在 Qt 前面加载的那几个（顺序按依赖：`vcruntime*` 是 `msvcp*` 的底座 ✓）
MSVC_DLLS = ("vcruntime140.dll", "vcruntime140_1.dll",
             "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll")


def preload_msvc_runtime():
    """抢先把**系统那份** MSVC 运行时装进进程 → 返回成功加载的个数 ✓。

    ⚠ 一个都装不上也照样返回 0 ✓ —— 调用方**不许**因为这个崩 ✗（它只是"尽量提前"✓）。
    ⚠ 非 Windows 直接返回 0 ✓（本函数只为那一个平台存在 ✓）。
    """
    if os.name != "nt":
        return 0
    import ctypes
    n = 0
    for dll in MSVC_DLLS:
        try:
            ctypes.CDLL(dll)        # 不带路径 ⇒ 系统搜索顺序 ⇒ 拿系统那份 ✓
            n += 1
        except OSError:
            pass
    return n
