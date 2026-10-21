"""一帧的第 2 段：**给自动测谎投一帧**（阶段 1「编排收口」第二刀 ✓ 2026-10-20 ✓）。

搬自 `gui/live_thread.py::_run`（**行为零改动** ✓）：原来那 11 行只干三件事 ——
  ① 开关判定（`th._vt_want or th._vt_on` ✓）；
  ② 原点优先用**跟着窗口走**的那份（`origin_ref` ✓ 没有才退回常量 ✓）；
  ③ 把**原生帧**交给 `th._vt_beat`（真正的工作全在那个方法里 ✓ 本段只负责"按顺序喂"✓）。

⚠ **必须是 `raw`（画框之前那份）** ✗：SDK 要求"输入图不能包含人为绘制的检测框"
  （`docs/INTEGRATION.md:28` ✓）；`vis` 上那时候确实还没画（本段排在画框之前 ✓），
  但**别改成 `vis`** —— 顺序一挪就会被污染，而这种事不会报错 ✓。
⚠ 真正的推理在 `_vt_loop` 那条**独立线程**里（实测单帧最慢 2232ms ✗ 放实时回路里就是画面卡两秒 ✓）。
⚠ **不按拍节流**：喂帧线程只留最新一帧（自己丢帧 ✓）⇒ 这里每拍不过一次引用赋值 ✓。
⚠ 后半句 `or th._vt_on` 是为了"**关掉开关**"那一下也能走到 `_vt_beat`
  ⇒ `_vt_ensure(False)` 才会把线程收掉 ✓（少了它就只能等下一拍才关 ✓）。
"""


def step(th, raw, origin_ref=None, origin=(0, 0)):
    """这一拍要不要投帧、投到哪个原点 ⇒ `True` = 真投了（`False` = 开关没开 ✓）。

    :param th: `LiveThread`（只读 `_vt_want` / `_vt_on`，只调 `_vt_beat` ✓）
    :param raw: **原生帧**（画框之前那份 ✓ 见文件开头 ⚠）
    :param origin_ref: "跟着窗口走"的原点（窗口模式那条路给 ✓）—— 优先于 `origin` ✓
    :param origin: 兜底原点（收流那条路用 ✓）
    """
    if not (th._vt_want or th._vt_on):
        return False
    # ⚠ 窗口模式那条路的原点会**跟着窗口**走 ⇒ 优先读它 ✓（否则窗口一挪，喂进去的
    #   画面坐标就不对了 ⇒ 瞄准跟着歪 ✓ 用户 2026-10-10 现场 ✓）
    ox, oy = ((origin_ref[0], origin_ref[1]) if origin_ref
              else (origin[0], origin[1]))
    th._vt_beat(raw, ox, oy)
    return True
