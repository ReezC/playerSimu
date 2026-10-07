# -*- coding: utf-8 -*-
"""鼠标**绝对定位**（断线重连「点服务器 / 点频道」用的那一层）。

分工（与 `decision/reconnect.py` 的模块头同一条纪律 —— **一处实现**）：

    `decision/reconnect.py`   什么时候点、点**画面上的哪个比例位置**（策略）
    `decision/mouse_aim.py`   ① 比例 → 像素 → 指令单位 的**换算**；② 撞角归零 + 走位 + 左键（发输入）
    `decision/input.py`       真正把 `MOVE` / `CLICK` 发到 Pro Micro

**为什么必须有这一层**（`docs/断线重连设计.md` §6.1）：鼠标通道**只有相对位移**
（`input.mouse_move(dx, dy)` ✓ —— 固件是相对 HID 鼠标 ✗ 不是绝对定位设备），
没有"移动到某坐标"这种指令。而选服务器 / 选频道**不能用键盘**（用户实测 ✓），
所以只能：**先撞到已知的角（绝对原点）→ 再按标定好的比例走位 → 点击**。

    1) `MOVE -4000 -4000` ⇒ 光标被 OS 夹到屏幕左上角 **(0, 0)**（已知绝对原点 ✓）
    2) 目标画面像素 P ⇒ 指令单位 = P / gain，发 `MOVE dx dy`
    3) `CLICK LEFT`

**三条前提 / 纪律**（缺一个坐标就是错的，所以宁可不动手 ✗）：

  1. **游戏画面左上角 = 屏幕左上角**（全屏 / 无边框窗口，且在主显示器上）。
     撞角归零落在**屏幕** (0,0)，而目标比例是相对**画面**算的 ⇒ 两者不同源就没法用。
     不是这个情况 ⇒ 这一层**不报错、也不猜**（用户要自己保证，见 tooltip ✓）。
  2. **gain（= 画面像素 / 指令单位）必须标定过**，且必须在**同一路画面**上量的
     （标定走的哪份画面，用的时候就得是那份 —— 流画面被缩放过的话，
     物理像素与流像素差一个比例，混用就差一个比例 ✗）。
     没标定 ⇒ **返回失败 + 人话**，绝不退回 1.0 硬点 ✗（宁可不做，也不乱点 ✓）。
  3. **指针加速必须关**（Windows "提高指针精确度" ✗）：开了之后 count→像素是**非线性**
     的，任何单一 gain 都不成立（标定工具会顺带量线性度 ✓ 见 §6.1）。

**gain 存在哪 / 怎么量**：`config/mouse_gain.json`（`{"gain_x": px/单位, "gain_y": ...}` ✓）
—— 这是**本机（游戏机侧）**的属性（指针速度 + 有没有加速 + 流缩放），与"哪个项目"无关 ⇒
放机器级配置 ✓ 不进 `project.yaml` ✗。写它的工具是 `tools/mouse_aim_calib.py`
（`tools/mouse_gain_calib.py` 的同款"帧差量光标位移"手法，但它能直接从**流**取帧 ✓
—— 本仓库的实时画面来自推流，`live.yaml` 里没有窗口 `rect` ⇒ 老工具那条路走不通 ✗）。

⚠ 与 `perception/lie_controller.load_gain()` **不是同一个读者**：那个是测谎闭环用的，
坏文件时**退回 1.0**（闭环能自己纠 ✓）；这一层是**开环绝对定位**，退回 1.0 就是
点错地方 ✗ ⇒ 这里**缺文件一律返回 None**（判据不同，故意不合并 ✓）。
两个读者读的是**同一个文件、同一套单位**（"px/单位" ✓）。
"""

from pathlib import Path

from decision import input as dinput

ROOT = Path(__file__).resolve().parent.parent
GAIN_PATH = ROOT / "config" / "mouse_gain.json"

#: 撞角那一下发多大（单位 = 指令 count）。**必须远大于任何屏幕的边长**：
#: OS 会把光标夹在屏幕里，多出来的部分自然被吃掉（发 4000 与发 9999 结果一样 ✓）。
#: ⚠ 固件对 `MOVE` 是**分块发送**的（`mouseMoveChunked`，每块 ≤120 ✓）——
#: 直接把 4000 塞给 `Mouse.move()` 会被截成 signed char（4000 → -96，**方向都反了** ✗），
#: 那个坑已经在固件里修掉了（见 `remote_kbd/pro_micro/pro_micro.ino` ✓）。
CORNER_STEP = 4000


def load_gain():
    """读标定文件 → `(gain_x, gain_y)`；**没标定过 / 坏值 ⇒ None**（绝不退回 1.0 ✗）。

    两个分量都必须是**正数**：非正说明方向反了或量歪了 ⇒ 当没标定过 ✓
    （`tools/mouse_aim_calib.py` 也是这么判的 ✓）。
    """
    import json
    try:
        if not GAIN_PATH.exists():
            return None
        d = json.loads(GAIN_PATH.read_text(encoding="utf-8"))
        gx = float(d.get("gain_x"))
        gy = float(d.get("gain_y"))
    except Exception:                       # noqa: BLE001 —— 坏文件当"没标定过" ✓
        return None
    if gx <= 0 or gy <= 0:
        return None
    return gx, gy


def target_px(frame_w, frame_h, xr, yr):
    """画面比例 (0~1) → 画面像素。判不出来 ⇒ `(None, why)`。

    比例越界**不夹取**、直接判失败：目标写错了就该**停下来报**，
    而不是"顺手"点到边上（那可能正好是「回到首页」/「删除角色」✗ 见设计文档 §9）。
    """
    try:
        w = float(frame_w)
        h = float(frame_h)
        x = float(xr)
        y = float(yr)
    except (TypeError, ValueError):
        return None, "目标比例不是数字"
    if w <= 0 or h <= 0:
        return None, "拿不到画面尺寸"
    if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
        return None, "目标比例超出画面（要看 0~1：x=%s y=%s）" % (xr, yr)
    return (x * w, y * h), ""


def counts_for(px, py, gain_x, gain_y):
    """画面像素 → **从屏幕左上角要走多少指令单位**（两个轴各用自己的增益 ✓）。

    返回 `((dx, dy), why)`：算不出来时第一个是 `None`、`why` 是人话。
    `gain` = 画面像素 / 指令单位。
    """
    try:
        gx = float(gain_x)
        gy = float(gain_y)
    except (TypeError, ValueError):
        return None, "鼠标增益不是数字"
    if gx <= 0 or gy <= 0:
        return None, "鼠标还没标定（增益非正：%s / %s）" % (gain_x, gain_y)
    return (int(round(float(px) / gx)), int(round(float(py) / gy))), ""


def aim_available():
    """这一层现在**能不能真的点**：鼠标通道在 + 标定过了。返回 `(ok, why)`。

    为什么要单独一个"能不能"：断线重连的**状态机**要在"发出点击动作"之前
    就知道该不该动（没标定就一点都不该动 ✓ 而且要说清缺什么 ✓），
    而不是发一条注定失败的点击再回退 ✗。
    """
    if not dinput.mouse_available():
        return False, ("鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
                       "⇒ 先把「被控机部署台」的 ProMicro 后端连上")
    if load_gain() is None:
        # ⚠ 报"在哪跑"要说准：这个工具跑在**工作机（跑工作台那台）**上 ——
        #   鼠标指令经 relay 打到游戏机 ✓（见 `tools/mouse_aim_calib.py` 的模块头 ✓）。
        #   写成"在游戏机上跑"会把人引到 A 机上去找仓库 ✗（2026-10-07 用户正是这么问的 ✓）。
        return False, ("鼠标没标定（缺 config/mouse_gain.json）"
                       "⇒ 先在**跑工作台这台机器**上跑 tools/mouse_aim_calib.py"
                       "（指令经 relay 打到游戏机 ✓，且要先停掉实时预览 ✓）")
    return True, ""


def corner_zero():
    """把光标撞到屏幕左上角 (0, 0)（见模块头第 1 步）。"""
    dinput.mouse_move(-CORNER_STEP, -CORNER_STEP)


def repeat_click():
    """**再点一下**（不重新走位 ✓ —— 给「双击」的第二下用 ✓）。返回 `(ok, why)`。

    ⚠⚠ 为什么第二下要单独一个出口、为什么**不**在 `click_ratio` 里点两下：
      `click_ratio` 跑在**实时回路线程**里 —— 那里**不许 sleep** ✗（睡一下就是丢帧 / 积压，
      见它的说明 ✓）。而"双击"要求两下有间隔 ✓ ⇒ 只能由**状态机**在**下一拍**补
      （`decision/reconnect.py` 的 `DOUBLE_CLICK_UIS` / `DOUBLE_CLICK_GAP` ✓）。
      所以这一层只负责"**发一条左键**" ✓ —— 光标还在原地（我们没动过它 ✓）⇒ 不用再撞角走位 ✓。

    ⚠ 2026-10-07 真机验出来的用法：**频道面板**单击只是"选中" ✗ ⇒ 必须双击才进频道 ✓
      （服务器行单击就行 ✓ 别搞混 ✗ 见设计文档 §11 ✓）。
    """
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    dinput.mouse_click("left")
    return True, "补第二下左键（双击 ✓ 光标没动 ⇒ 不用重新走位 ✓）"


def click_ratio(frame_shape, xr, yr, gain=None):
    """**撞角归零 → 走位 → 左键点击**（这一层的唯一出口）。

    `frame_shape`：`frame.shape`（只要前两维 ✓）。
    `gain`：不给就现读 `config/mouse_gain.json`（`(gx, gy)` 或 None ✓）。

    返回 `(ok, why)`：`why` 一定是人话（成功也说清"走多少、点哪儿"✓ ——
    事后复盘只靠 `behavior.log`，看不到画面 ✓）。

    ⚠ **按顺序发三条指令**（`MOVE → MOVE → CLICK`）：固件按行顺序执行、
    OS 按到达顺序处理 HID 报文 ⇒ 按钮按下时系统里的光标**已经在目标上**了 ✓
    （`WM_LBUTTONDOWN` 带的是当时的坐标 ✓）。**所以这里不需要 sleep** ✗
    —— 这条函数跑在**实时回路线程**里，睡一下就是丢帧 / 积压（实测过代价 ✓）。
    """
    if frame_shape is None or len(frame_shape) < 2:
        return False, "拿不到画面尺寸"
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if gain is None:
        gain = load_gain()
    if not gain:
        return False, "鼠标没标定（缺 config/mouse_gain.json）"
    gx, gy = gain
    px, why = target_px(w, h, xr, yr)
    if px is None:
        return False, why
    d, why = counts_for(px[0], px[1], gx, gy)
    if d is None:
        return False, why
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    dx, dy = d
    corner_zero()                       # ① 到已知绝对原点
    dinput.mouse_move(dx, dy)           # ② 走到目标（x 用 gx、y 用 gy ✓）
    dinput.mouse_click("left")          # ③ 点
    return True, ("撞角归零 → 走 (%d, %d) 指令单位 → 左键点击"
                  "（画面 (%.0f, %.0f) 像素，比例 %.3f, %.3f）"
                  % (dx, dy, px[0], px[1], xr, yr))
