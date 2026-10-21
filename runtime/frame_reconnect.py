"""一帧的第 5 段：**断线判断 / 自动重连**（阶段 1「编排收口」第二刀 ✓ 2026-10-20 ✓）。

搬自 `gui/live_thread.py::_run`（**行为零改动** ✓ —— 函数体原样搬过来，只把 `self.` 换成 `th.`、
`reconnector` 换成本地的 `rc` 别名 ✓ 注释一个字没动 ✓）：

  · 状态机 `Reconnector` 自己判"现在该按 Enter / 该点服务器 / 该滚频道" ✓（判据在里面 ✓
    见 `decision/reconnect.py`）—— 本段只负责**把动作发出去** ✓；
  · 五个出口：`tap`（按键）/ `home`（**只撞角归零、这一拍不点** ✓ 用户 2026-10-17）/
    `click`（比例 → 撞角 → 走位 → 左键）/ `click2`（双击第二下）/ `scroll`（先滚到顶再往下 ✓）；
  · 每条结果都要 `rc.feedback(ok, why)` **回填**（成败只有发完才知道 ✓ 否则界面只剩"准备点…"）。

⚠⚠ **必须排在画框之前**（见函数体第 2 行）：此刻 `vis` 还是**干净帧**，画过蓝框/平台线之后
  会干扰 UI 模板匹配 ✓（顺序由 `selftest_live_panel::t_frame_stage_order` 钉着 ✓）。
⚠ 这条线程里**不许 sleep** ✗（实时回路硬规矩 ✓）：要"等生效"就靠**下一拍** ✓
  （`home` 与 `scroll` 都是"一格一拍"就为这个 ✓）。
⚠ 状态机 `rc` 是**跨帧**对象（用户 2026-10-20 选的 B ✓）⇒ 挂在 `th._reconnector` 上 ✓
  （本段只借用、不创建 ✓）。

判据 `selftest_live_panel::t_frame_reconnect_stage`：五个出口各发一次正确的输入 · 每条都回填 ·
坐标走 `_aim_geo` 折回整帧 · `home` 那一拍**不点** ✓。
"""

import time

#: ⚠ 输入只走 `execution.mouse_aim`（换算/发指令**只有一处** ✓ 用户 2026-10-09 ✓）——
#:   ⛔ 别在本段直接摸输入层 ✗（那条线程里没有 `dinput` 这个名字 ⇒ 当场 NameError ✓ 2026-10-08 栽过）。
from execution import mouse_aim


def step(th, vis, player_box, origin_ref, origin_x, origin_y, _tap):
    """判这一拍要不要插手断线/选角 ⇒ 该发什么就发什么（**返回这一拍的动作名或 `""`**）。

    :param th: `LiveThread`（用 `th._reconnector` / `th._aim_geo` / `th._screen_state` ✓）
    :param vis: 这一帧（**干净帧** ✓ 画框之前 ✓ 比例都相对它算 ✓）
    :param player_box: 追踪器**刚返回**的结果（`None` = 这帧没找到玩家 ✓）——
                       ⚠ 别传下面那个"跟丢时拿上一帧兜底"的版本 ✗（它永远不为 `None` ✓
                       状态机就永远以为"人还在" ⇒ 断线重连会失灵 ✓）
    :param origin_ref: "跟着窗口走"的原点（窗口模式 ✓ 优先）
    :param origin_x: 兜底原点 x（收流那条路 ✓）
    :param origin_y: 兜底原点 y
    :param _tap: `execution.input.tap`（由调用方给 ✓ 与 `_run` 里那份**同一个函数** ✓）
    :return: 这一拍的动作名（`"tap"` / `"home"` / `"click"` / `"click2"` / `"scroll"` / `""` ✓ 打点用）
    """
    rc = th._reconnector
    # ---- 断线判断 / 自动重连 ----
    # 必须放在**画框之前**：此刻 vis 还是干净帧。下面会往 vis 上
    # 画玩家蓝框、平台线……画过的帧会干扰 UI 模板匹配。
    # 注意 player_found 要用追踪器刚返回的结果，不能用下面那个
    # 「跟丢时拿上一帧兜底」的 player_box —— 那玩意永远不为 None。
    #   ⭐ `ui_hint` = 本帧的**画面状态广播**（`_screen_state` ✓ 同样走
    #     `ui_state.detect` ✓ 还带那套限流+留住 ✓）—— **只用来立"选角 ⇒ 还欠一次
    #     Enter"那份待办** ✓（2026-10-12 用户"现在就上"✓）：状态机自己那一帧万一
    #     没认出来 / 认成别的 ✗ 也不至于把「按 Enter 进游戏」那一步整个吞掉 ✓。
    #     ⚠ 状态机那边的判定口径**没变** ✗（它照旧自己 detect ✓ 见 `update` 的说明 ✓）。
    _act = rc.update(vis, player_box is not None,
                                                        time.perf_counter(),
                                                        th._screen_state)
    if _act and _act.get("act") == "tap":
            _tap(_act["key"])
    elif _act and _act.get("act") == "home":
            # ⭐⭐⭐ **只撞角归零、这一拍不点**（用户 2026-10-17 ✓ 原话：
            #   「**地铁二号线 无法选中频道**」✗）。
            #   ⛔ 收流那条路读不到光标 ⇒ 只能"撞角 + 绝对走位"✓，而老写法
            #     **同一拍**把两条发出去 ✗ —— 撞角 `-4000,-4000` 固件要分块发完
            #     （~34 块/轴 ✓）⇒ 走位紧跟着排上去 ⇒ **起点不是角** ⇒ 点歪 ✓
            #     （`remote_kbd/A_relay_trace.log` 实证：两条 `MOVE` 同一毫秒 ✓）。
            #   ⇒ 同 `scroll` 的"**一格一拍**"✓：这一拍只归零 ✓，下一拍状态机再出
            #     `click`（带 `homed` ✓）⇒ 那时角早就到位了 ✓。
            #   ⚠ 这里**不 sleep** ✗（实时回路硬规矩 ✓ 等生效靠"下一拍" ✓）。
            _ok, _why = mouse_aim.corner_only(vis.shape)
            rc.feedback(_ok, _why)

    elif _act and _act.get("act") == "click":
            # ⭐ 「点服务器 / 点频道」（2026-10-06 ✓）：比例 → 撞角归零 →
            #   走位 → 左键，**换算与发指令只有 `execution.mouse_aim` 一处** ✓。
            #   ⚠ 比例是相对**这一帧**算的（`vis.shape` = 我们看到的画面 ✓）——
            #     状态机给的比例与画面同源，别拿别的尺寸换算 ✗。
            #   ⚠ 结果要**回填**给状态机（成败只有发完才知道 ✓）：否则界面上
            #     只剩"准备点…"，点没点上没人知道 ✗。
            # ⚠ 帧原点一起递（窗口模式那条路会**跟着窗口**走 ✓ 见 `origin_ref` ✓；
            #   收流那条路用 `origin_x/y` ✓ = 老行为 ✓）—— 窗口拖到副屏（x<0 ✓）
            #   全靠这一步才点得中 ✓（用户 2026-10-10 ✓）。
            _org = ((origin_ref[0], origin_ref[1]) if origin_ref
                            else (origin_x, origin_y))
            # ⭐⭐ **断线重连的点击也要折回整帧**（用户 2026-10-17 ✓ 原话提醒：
            #   「注意**断线重连**、自动测谎的鼠标适配」✓）—— 状态机给的比例是
            #   相对**我们看到的这块**（框了游戏区域之后 ✓）⇒ 不折就整块偏 ✓
            #   （\"点服务器/点频道\"那种格点尤其致命 ✗ 差一格就点到别的频道 ✓）。
            # ⭐⭐⭐ **A 机游戏窗口会移动**（用户 2026-10-17 ✓「你保证你的方案
            #   能适配吗？」✓）⇒ `_aim_geo` 会先问窗口在哪：
            #     屏幕坐标 = 窗口原点 + 比例 × 窗口尺寸 ✓
            #   ⇒ 窗口挪/缩放/重启、换分辨率**全自动跟上** ✓
            #     （问不到 ⇒ 退回 `origin_ref`/`origin_x` 那条老路 ✓ 不会更糟 ✓）
            _org, _mroi, _mfull, _gain = th._aim_geo(vis, _org)
            _ok, _why = mouse_aim.click_ratio(vis.shape, _act["x"], _act["y"],
                                                                                gain=_gain,
                                                                                origin=_org,
                                                                                roi=_mroi, full_shape=_mfull,
                                                                                # ⭐ 上一拍已经**只归零**过（`act="home"`
                                                                                # ✓ 2026-10-17）⇒ 这一拍**别再撞角** ✗
                                                                                # （撞角与走位分拍 ⇒ 起点才是真的角 ✓）
                                                                                homed=bool(_act.get("homed")))
            rc.feedback(_ok, _why)

    elif _act and _act.get("act") == "click2":
            # ⭐⭐ 「双击」的第二下（2026-10-07 ✓ 真机验出来：**频道面板单击只是
            #   "选中"** ✗，必须双击才进 ✓）。⚠ 这一下**不重新走位** —— 光标还在
            #   原地（第一下之后我们没动过它 ✓）⇒ 只补一条左键 ✓。
            #   ⚠ 间隔由**状态机**控（`reconnect.DOUBLE_CLICK_GAP` ✓ 120ms ✓）：
            #   实时回路这条线程**不许 sleep** ✗（睡一下就是丢帧 / 积压 ✓）。
            _ok, _why = mouse_aim.repeat_click()
            rc.feedback(_ok, _why)

    elif _act and _act.get("act") == "scroll":
            # ⭐⭐ 「先滚到顶、再往下滚 N 格」（第 21~60 个频道 ✓ 见
            #   `reconnect.CHANNEL_TOTAL` ✓）。两条滚动**连着发** ✓（正数向上 ✓）；
            #   ⚠ 滚完**这一拍不点** ✗ —— 由状态机**下一拍**再出点击动作 ✓
            #   （实时回路这条线程**不许 sleep** ✗ 等滚动生效要靠"下一拍" ✓）。
            _up = int(_act.get("up") or 0)
            _down = int(_act.get("down") or 0)
            # ⚠ 走 `mouse_aim.scroll` ✓ 别在这儿直接摸输入层 ✗
            #   （这条线程里**没有** `dinput` 这个名字 ✓ 摸它会当场 NameError ✓
            #    2026-10-08 真栽过、被 live_panel 的 pyflakes 静态检查逮住 ✓）。
            # ⚠⚠ **坐标要一起递进去**（2026-10-09 ✓）：滚轮只作用在光标底下的
            #   控件上 ✗ ⇒ 不带坐标就是"在画面上方滚" ⇒ 列表一格没滚 ⇒
            #   接着点第一屏第 2 格 ⇒ 进错频道（用户："选 22、去了 2" ✓）。
            # ⚠ `_org` **不跨分支借用** ✗（那两条 elif 各自独立 ⇒ 谁先跑没保证 ✓）：
            #   这里自己算一份 ✓（就是同一句 `origin_ref` 优先的写法 ✓）。
            _org_s = ((origin_ref[0], origin_ref[1]) if origin_ref
                                else (origin_x, origin_y))
            _ok, _why = mouse_aim.scroll(_up, _down, vis.shape,
                                                                      _act.get("x"), _act.get("y"),
                                                                      origin=_org_s)
            rc.feedback(_ok, _why)

    return str(_act.get("act") or "") if _act else ""
