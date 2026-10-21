"""一帧的第 0 段：**到手之后先把"要量像素的那份"留下来**（阶段 1「编排收口」第二刀 ✓）。

搬自 `gui/live_thread.py::_run`（2026-10-20 ✓ **行为零改动** ✓ —— 原来那 25 行连注释一起搬过来，
只把 `self.` 换成 `th.`、把 `raw` / `_mmap_panel` 两个局部名换成"写进盒子 / 当返回值" ✓）。

这一段干两件事，顺序不能换 ✗：

  ① **留一份原生帧**（`ctx.raw`）：下面会往 `ctx.vis` 上**就地**画玩家蓝框/怪物绿框/攻击线/
     视野虚线（`cv2` 的绘制是原地改数组 ✓），而取帧做测量的那几个功能（探针标定 / HP·MP 条框选 /
     小地图框选 / 标定弹窗 / 叠图核对）量的都是**像素**，被画过就全毁 ✓
     （用户 2026-09-27 现场："框选小地图时把视野灰色虚线一起框进去了" ✓）。
  ② **趁还没画，把小地图那块面板裁一份**（下一段定位要用 ✓ 见 `_mmap_panel_from_frame` ✓）。

⚠ **只在"真要画"时才拷**：本文件之外所有 `cv2.*(vis, …)` 都在推理那道门里 ⇒ 推理关着时
  `ctx.raw is ctx.vis`（同一个数组、零拷贝 ✓ 收画面阶段零开销 ✓）。
  判据 `selftest_live_panel::t_raw_frame_before_draw` 钉着这条（搬完改成**按文件**查 ✓）。
"""


def step(th, ctx):
    """跑这一段 ⇒ 返回**这一拍的小地图面板**（`None` = 没框选 / 裁不到 ✓）。

    :param th: `LiveThread`（要 `th._infer` / `th._mmap_panel_from_frame` / `th._mmap_panel_cur`）
               —— 阶段是"调度"，不是"算法" ⇒ 它只管按顺序调，具体怎么裁还是线程那边的事 ✓。
    :param ctx: 一帧的盒子（`runtime.pipeline.FrameCtx` ✓）—— 本段只写 `ctx.raw` 一个字段 ✓
                （《什么进盒子》见 `FrameCtx` 的说明 ✓）。
    """
    # ⚠ 这一行的**条件必须原样**：它假设"所有绘制都在推理门里" ✓（见文件开头 ① ✓）
    ctx.raw = ctx.vis.copy() if th._infer.is_set() else ctx.vis
    # ⚠ 存下来给定位那一路（2026-10-09 修 ✗）：以前算出来就扔了 ⇒ 来源=「从实时画面」时
    #   定位那边**永远拿到 None** ⇒ 每拍都回"没有可裁的小地图区域"✗
    #   （用户现场："本地来源的自动定位不好使、认不出黄点" ✓）。
    _panel = th._mmap_panel_from_frame(ctx.vis)
    th._mmap_panel_cur = _panel
    return _panel
