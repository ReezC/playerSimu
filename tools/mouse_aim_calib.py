# -*- coding: utf-8 -*-
"""鼠标**绝对定位标定**：量出「1 个鼠标指令单位 = 画面里走多少像素」⇒ 写
`config/mouse_gain.json`（断线重连「点服务器 / 点频道」要用它，见 `decision/mouse_aim.py`）。

**为什么必须标定**（`docs/断线重连设计.md` §6.1）：鼠标通道**只有相对位移**
（固件是相对 HID 鼠标），绝对定位靠"撞到屏幕左上角（已知原点）→ 按比例走位"。
那个"走位"必须知道 **counts → 像素** 的换算比例 —— 它就是这里量的 gain。

**怎么量**（不依赖游戏里画不画光标模板 ✓）：
    连拍两帧、中间发一条**已知单位数**的 `MOVE`；光标是画面里**唯一在动**的东西
    ⇒ 前后帧差里恰好两个小连通域（旧位 / 新位），它们的质心位移就是光标实际走了多少
    像素 ⇒ `gain = 位移 / 单位数`。
    （量位移的那段代码是 `tools/mouse_gain_calib.py::_cursor_delta` —— **同一份实现** ✓
      这里只是把帧来源换成**流**、并顺带量线性度。）

**为什么要换帧来源**：老工具从 `config/live.yaml` 的窗口 `rect` 抓本机窗口 ✗ ——
本仓库的实时画面来自 **A 机推流**（`live.yaml` 里没有 `rect`），那条路走不通。
所以这里直接从 `config/link.yaml` 的 `stream.url` 开流 ✓（和实时页看到的是同一路画面 ✓
—— 这一点很关键：gain 必须在**同一路画面**上量，流被缩放过的话物理像素与流像素差一个
比例，混用就差一个比例 ✗）。

用法（在**工作机（B 机）**上跑，鼠标指令会经 relay 打到**游戏机（A 机）**）：

    python -X utf8 -m tools.mouse_aim_calib                 # 量 + 写 config/mouse_gain.json
    python -X utf8 -m tools.mouse_aim_calib --dry-run       # 只打印，不写文件
    python -X utf8 -m tools.mouse_aim_calib --rounds 5      # 多量几轮（取中位）
    python -X utf8 -m tools.mouse_aim_calib --dump data/_calib_dump
                                                            # ⭐ 量不到/不过时：存现场帧
                                                            # （算法真正看到的两帧 + 差分 ✓）
                                                            # **看图，别猜** ✓

⚠ 跑之前：
  1. **游戏停在一个画面基本不动的界面**（登录界面 / 选频道界面都行）——
     画面自己一直在动的话，帧差里就不止光标在动了，量出来是噪声 ✗；
  2. **鼠标指针要在画面里可见**（被游戏隐藏了就没法量 ✗）；
  3. 游戏机 Windows 的指针速度保持 **6/11**、**关闭「提高指针精确度」**（见 §6.1）——
     开着加速 count→像素是**非线性**的，单一 gain 不成立。本工具会**道道设卡**：
     两个不同单位数各量一遍查线性度 ✓，**再拿一个大位移（`STEP_VERIFY`）验一遍"小位移的
     gain 能不能预测它"** ✓ —— ⚠ 2026-10-07 真机栽过：小位移那两档即使开着加速也**接近
     线性**（自证不了 ✗），写出来的 gain 让 `MOVE 1905 668` **把光标甩到屏幕角** ✗
     ⇒ 任何一道不过都**拒绝写文件**（要强行写：`--force` ✓）。

⚠ 需要输入通道在线：没连就按 `config/link.yaml` 的 `kbd` 段自己连一次
（同 `tools/selftest_link.py` 的做法 ✓）。连不上就**明确报错**，不写任何东西 ✓。
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get                       # noqa: E402
from decision import input as dinput              # noqa: E402
from tools.mouse_gain_calib import _cursor_delta  # noqa: E402 —— 量位移只有这一份实现

OUT = ROOT / "config" / "mouse_gain.json"

#: ⚠⚠ **两个小档都必须 ≤ 127**（2026-10-07 发现 ✓）：固件里 `Mouse.move()` 的入参是
#: `signed char`，**老固件**（没有 `mouseMoveChunked`）会把 >127 的值截断 ✗
#: （4000 → −96、200 → −56 ✓ 文档 §6.1 专门记过 ✓）⇒ 那时"两个小档"就已经不可信 ✗，
#: 工具却会报"线性度不过 ⇒ 疑似开了指针加速" ⇒ **又把人引错** ✗。
#: ⇒ 小档留在 ≤127 的**可信区间**里（60 / 120 ✓）；**大位移**
#: （`STEP_VERIFY`）专门用来验"大数能不能走对"—— 它不过有两种可能，见那边的提示 ✓。
STEP_SMALL = 60
STEP_BIG = 120
#: 两个单位数量出来的 gain 相差超过这个比例 ⇒ 疑似开了指针加速（非线性 ✗）。
LINEAR_TOL = 0.25
#: ⭐⭐ **校验档（大位移）** —— 两档小位移算出来的 gain **必须也能预测这一档**。
#:   ⚠⚠ 为什么非要有它（2026-10-07 ✓ 断线重连开箱**真机上栽了**）：小位移那段曲线
#:   即使开着指针加速也**接近线性** ⇒ 60/200 两档量出来 17.0 / 47.6 成比例 ✓、
#:   线性度校验**通过** ✗ ⇒ 工具高高兴兴写了个 gain；而真机上按它算出来的
#:   `MOVE 1905 668`（= 选频道服务器那一下 ✓）**让光标一路冲到屏幕角** ✗
#:   （行为日志：`撞角归零 → 走 (1905, 668) → 左键点击` 点了三次、界面没变 ⇒ 放弃 ✓）。
#:   ⇒ 那是"**小位移的尺子量不了大位移**"✓（指针加速的非线性在大位移上才暴露 ✓）。
#:   ⇒ 这一档**预测不准就拒绝写文件** ✓（宁可不做，也别写一个会把光标甩飞、乱点一通的 gain ✗）。
STEP_VERIFY = 800
#: ⭐ 「撞角归零之后，先把光标**挪到探针带下面**再量」—— 每步 `PARK_DY` 个单位、走 `PARK_STEPS` 步
#:   （**每步都 ≤127** ⇒ 老固件也不会截断 ✓）。
#:   ⚠⚠ 为什么（2026-10-07 ✓ 实测踩到、绕了好几圈）：撞角归零把光标停在画面 **(0,0)** ✓，
#:   而**探针带就在画面顶部**（y≈19..35 ✗）—— 于是两难：
#:     把带子**遮掉** ⇒ 光标一往右走就钻进去 ✗（帧差里只剩"旧位"一个块 ⇒ `_cursor_delta`
#:     只认"恰好两个块" ⇒ 直接返回 None ⇒ 工具报"量不到位移" ✗，还会把人往"指针速度/
#:     指针不可见"上引 ✗）；
#:     不遮 ⇒ 带子每帧都在变 ⇒ 量到的是它 ✗（另一轮现场 ✓）。
#:   ⇒ 归零后先**往下**走（4×120 ≈ 让光标落到流画面 y≈120 一带 ✓）⇒ 之后的横向测试全在
#:   **带子下面的干净行**里 ✓ 遮罩和光标**不再打架** ✓。
PARK_DY = 120
PARK_STEPS = 4
#: ⭐⭐ 「停靠点再往右挪多少」—— 撞角归零后光标**紧贴画面左边缘**，箭头会被裁掉半个 ✗
#:   ⇒ 帧差里"旧位"只剩几个像素（现场：`a=6` ✗）⇒ 与"新位"（`a=156`）面积差 20 倍以上 ⇒
#:   成对判据把它**误拒** ⇒ 有些轮次"量不到" ✗。挪 `PARK_DX` 个单位（一条指令 ✓ ≤127 ✓）
#:   就整支都在画面里 ✓，而且仍落在"左边缘"那条先验里（`left_edge = 60` ✓）。
PARK_DX = 40


#: ⭐⭐⭐ **慢发对照档**（2026-10-07 ✓ 现场量出来的）：同样走 `SLOW_TOTAL` 个单位，但拆成
#: `SLOW_TOTAL / SLOW_CHUNK` 条小指令、每条之间隔 `SLOW_GAP` 秒 ⇒ **每个 HID 报文只带
#: `SLOW_CHUNK` counts** ✓。
#: ⚠⚠ 为什么非要有它（前面的线性度 / 大位移校验都**抓不到**这一种 ✗）：Windows 的指针加速
#:   （「提高指针精确度」）是**按速度**生效的 ✗ ⇒ 「小位移接近线性」✓ 甚至 800 单位也能蒙对 ✓
#:   ⇒ 三档全过、工具高高兴兴写了个 gain ✗，而真点击时那条大指令因为**发得快**被放大 ≈2 倍 ✗
#:   ⇒ 光标甩飞、点在别处 ✓（第四跑现场：`MOVE 1905 668` 点了三次界面没变 ✓）。
#: 判据：**快慢两条的 gain 必须相当**（`SLOW_TOL` 内）✓ —— 不相当 ⇒ 加速还在 ⇒ **拒绝写文件** ✓。
#:   （2026-10-07 实测：慢发 329.8px / 800 单位 = 0.412 vs 快发 0.802 ⇒ 差 95% ✗ ⇒ 抓到了 ✓）
SLOW_TOTAL = 800
SLOW_CHUNK = 10
SLOW_GAP = 0.02
SLOW_TOL = 0.15


def measure_slow(src, total=SLOW_TOTAL, chunk=SLOW_CHUNK, gap=SLOW_GAP, dump=None, anchor=None):
    """**慢慢发** `total` 个单位（拆成 `chunk` 一条 ✓ 每条 ≤127 ✓）⇒ 量总位移（px ✓）。

    ⚠ 与 `measure` 的区别只有一个：**发得快不快** ✓（本函数每个报文只 `chunk` counts ✓
    ⇒ 指针加速（按速度生效 ✗）不会介入 ✓）⇒ 拿它和"一条大指令"的 gain 对照，就是
    **加速开没开**的直接判据 ✓（见 `SLOW_TOTAL` 那段 ✓）。
    """
    band = None
    before = _grab(src)
    if before is None:
        return None
    band = probe_band_rect(before.shape[1], before.shape[0]) or ()
    n = max(1, int(round(float(total) / float(chunk))))
    sign = 1 if total > 0 else -1
    for i in range(n):
        # ⚠ **相邻两条不要一样**（`chunk±1` 交替 ✓ 总和不变 ✓）：万一中继 / 固件 / HID 层
        #   对**重复的相同报文**做了去重或合并 ✗，80 条一模一样的 `MOVE 10` 就会少走 ✗
        #   ⇒ 慢发档量出来的 gain 偏小 ⇒ 又被误判成"加速还在" ✗（2026-10-07 ✓ 排查掉这条 ✓）。
        c = int(chunk) + (1 if i % 2 else -1)
        dinput.mouse_move(sign * c, 0)
        time.sleep(gap)
    # ⚠ 等**足够久**再抓"之后"那一帧（2026-10-07 ✓ 踩过）：这一档故意发得很碎（`n` 条 ✓），
    #   若串口 / HID 那条链有排队 ⇒ 收尾时可能**还在走** ✗ ⇒ 抓到的是"走了一半"的画面
    #   ⇒ 量出来的 gain 偏小 ⇒ 工具会误判成"加速还在" ✗（那正是误报的来源之一 ✓）。
    #   ⇒ 这里给足 1.5s（标定不是实时路径 ✓ 慢一点无所谓 ✓）。
    time.sleep(1.5)
    after = _grab(src)
    if band:
        _blank(before, band)
        _blank(after, band)
    d = _cursor_delta(before, after, direction=sign, anchor=anchor)
    _dump_round(dump, "slow%d" % int(total), 0, before, after, d)
    # 回程（同样慢慢发 ✓ 不然回程自己就触发了加速 ⇒ 光标漂走 ✗）
    for _ in range(n):
        dinput.mouse_move(-sign * int(chunk), 0)
        time.sleep(gap)
    time.sleep(0.3)
    return d


def _connect_input():
    """确保输入通道在线（鼠标指令只有 ProMicro 后端能发 ✓）。返回 `(ok, why)`。"""
    if dinput.mouse_available():
        return True, "输入通道已就绪"
    host = get("kbd", "host")
    port = int(get("kbd", "port", 9000) or 9000)
    cert = get("kbd", "cert", "remote_kbd/certs/cert.pem")
    try:
        dinput.use_network(host, port, cert)
    except Exception as e:                      # noqa: BLE001
        return False, ("连不上被控机 %s:%d（%s: %s）—— 查「键盘中继」在不在跑 / "
                       "端口证书 / 能不能 ping 到" % (host, port, type(e).__name__, e))
    if not dinput.mouse_available():
        return False, "连上了但鼠标后端不可用（只有 ProMicro 后端有硬件鼠标）"
    return True, "已连上 %s:%d" % (host, port)


def _open_stream():
    """按 `config/link.yaml` 的 `stream` 开流（**只读** ✓）。"""
    from link.pyav_source import PyAVSource
    url = str(get("stream", "url", "udp://0.0.0.0:5000"))
    fmt = get("stream", "format") or None
    src = PyAVSource(url, decode_format="bgr24", container_format=fmt)
    src.open()
    return src, url


def _grab(src, timeout=5.0, refresh=0.25):
    """要一帧 —— ⭐⭐ **要"最新"的那一张**，不是管道里排在前面那张 ✗。

    ⚠⚠ 为什么（2026-10-07 ✓ 标定第二大病根）：推流是**单向管道**（UDP → 解码 → 我们读 ✓），
    读到的帧**天生滞后**（端到端实测约 130ms ✓ 见部署台那张卡 ✓）。原来拿到一帧就返回 ✗
    ⇒ "指令之前"和"指令之后"抓到的**可能是同一张旧画面** ✗ ⇒ 差图为空 ⇒ 工具报
    "量不到位移" ✗（现场帧里 `u120_r00` 的 before/after **逐像素相同** ✓ 就是这么来的 ✓）。
    ⇒ 拿到第一帧后再**继续读 `refresh` 秒**、只留**最后**一张（= 现在这一刻的画面 ✓，
      把滞后与积压一起冲掉 ✓）。0.25s ≈ 覆盖那个 130ms 延迟 ✓。

    ⚠ 代价：每次抓帧多花 `refresh` 秒（一次标定几十次 ⇒ 多几十秒 ✓ 完全值 ✓）。
    """
    t0 = time.perf_counter()
    f = None
    while time.perf_counter() - t0 < timeout:
        try:
            g = src.read()
        except Exception:                       # noqa: BLE001 —— 没帧就再等 ✓
            g = None
        if g is not None:
            f = g.image
            break
    if f is None:
        return None
    t1 = time.perf_counter()
    while time.perf_counter() - t1 < float(refresh):
        try:
            g = src.read()
        except Exception:                       # noqa: BLE001
            g = None
        if g is not None:
            f = g.image
    return f


def park_below_band():
    """撞角归零 ⇒ 再把光标挪到**探针带下面的干净行**（见 `PARK_DY` 那段说明 ✓）。

    ⚠ 分 `PARK_STEPS` 步走（每步 ≤127 ⇒ 不怕老固件的 signed char 截断 ✓）。
    """
    dinput.mouse_move(-4000, -4000)
    time.sleep(0.25)
    for _ in range(PARK_STEPS):
        dinput.mouse_move(0, PARK_DY)
        time.sleep(0.08)
    # ⚠ 再往右挪一点（`PARK_DX` ✓）：撞角归零后光标**紧贴画面左边缘** ⇒ 箭头被裁掉半个 ✗
    #   ⇒ 帧差里"旧位"那一团只剩几个像素 ⇒ 与"新位"的面积差能到 20 倍以上 ✗ ⇒ 成对判据
    #   （面积相当 ✓）把它误拒 ⇒ 有些轮次"量不到" ✗（2026-10-07 现场：旧位 `a=6` / 新位
    #   `a=156` ✓）。挪开几十个单位就整支都在画面里 ✓，而且仍在"左边缘"那条先验内 ✓
    #   （`_cursor_delta` 的 `left_edge = 60` ✓）。
    dinput.mouse_move(PARK_DX, 0)
    time.sleep(0.08)


def agree_px(vals, tol=0.25, need=2):
    """从一堆位移里挑「**至少 `need` 个互相一致**（±`tol`）」的那一簇 ⇒ 取其中位；没有 ⇒ None。

    ⚠⚠ 为什么要这条（2026-10-07 ✓ 绕了一大圈才对）：
      画面里**只要还有别的东西在动**（选频道界面的负载条 / 各种 UI 动画 ✗），
      `_cursor_delta` 那句"恰好两个块 = 光标新旧位"就会**随机**抓到它们 ⇒ 每轮数字乱跳 ✗
      （现场：480 单位量 48.0px、120 单位量 48.1px ⇒ 同一个"固定伪影" ✗）。
      而 ⭐ **光标对同一条指令的位移是确定的** ✓（每轮都从同一处出发、量完还有回程 ✓）
      ⇒ **"两轮以上一致"的那个数才是光标** ✓，背景噪声不会重复 ✓✓。
      返回的第二个值（一致轮数）也会写进输出 ⇒ `n=2/5` 这种一眼能看出可信度 ✓。
    """
    if not vals:
        return None, 0
    vs = sorted(float(v) for v in vals)
    best_center, best_n = None, 0
    for v in vs:
        grp = [x for x in vs if abs(x - v) <= max(2.0, tol * max(v, 1.0))]
        if len(grp) > best_n:
            best_n, best_center = len(grp), grp[len(grp) // 2]
    if best_n < int(need):
        return None, best_n
    return best_center, best_n


def _dump_round(dump, tag, i, before, after, d):
    """把**这轮算法真正看到的**存下来（`--dump DIR` ✓）。存三种：

    · `<tag>_rNN_before.png` / `<tag>_rNN_after.png` —— 发指令前 / 后的两帧，
      **已经按算法遮过探针带** ✓（就是它拿去比的那两张 ✓）；
    · `<tag>_rNN_diff.png` —— 放大 4 倍的差分伪彩 ✓；
    · `<tag>_rNN.txt` —— `_cursor_delta` 那轮的返回原值 ✓。

    ⚠⚠ 为什么非要有这个口子（2026-10-07 ✓ 绕了一整晚的教训）：量不到 / 线性度不过时，
    工具只扔一句结论 ✗ ⇒ 人只能瞎猜（"指针速度不对""推流不画指针""固件坏了" ✗），
    而**真相看一眼帧就完了** ✓ —— 这一轮最后就是靠"人肉看 A 机屏幕"才定位的 ✓。
    ⇒ 以后凡是"量不到"，先 `--dump` 看现场，别再猜 ✓（失败信息里也会提示 ✓）。
    """
    if not dump:
        return
    try:
        import cv2
        t = Path(dump)
        t.mkdir(parents=True, exist_ok=True)
        stem = "%s_r%02d" % (tag, i)
        cv2.imwrite(str(t / (stem + "_before.png")), before)
        cv2.imwrite(str(t / (stem + "_after.png")), after)
        gray = cv2.cvtColor(cv2.absdiff(before, after), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(t / (stem + "_diff.png")),
                    cv2.applyColorMap(cv2.convertScaleAbs(gray, alpha=4.0),
                                      cv2.COLORMAP_JET))
        (t / (stem + ".txt")).write_text(
            "_cursor_delta 返回 %r（两个块 = 光标旧位/新位；只有一个块 ⇒ 返回 None ✓）\n" % (d,),
            encoding="utf-8")
    except Exception:                       # noqa: BLE001 —— 存现场失败不该影响测量 ✓
        pass


def measure_auto(src, units, rounds=6, max_scale=8, dump=None, anchor=None):
    """先按 `units` 量；**量不到就成倍放大**再试 ⇒ `(units_used, result)`。

    ⚠ 为什么要这条（2026-10-07 ✓）：指针速度低的时候（滑块偏左 ✓），60 个单位只走
    一两个像素 ⇒ 两个块**糊在一起** ⇒ `_cursor_delta` 返回 None ⇒ 工具直接报"量不到
    位移" ✗ —— 而那**不是**"指针不在画面里" ✗（真机上就是这么被误导过 ✓）。
    ⇒ 量不到 ⇒ `×2` 再试（最多 `max_scale` 倍 ✓），并把**实际用的单位数**带回去 ✓
      （算 gain 要用它算 ✓ 见调用处）。
    """
    u = int(units)
    for _ in range(max(1, int(max_scale)).bit_length()):
        r = measure(src, u, rounds, dump=dump, anchor=anchor)
        if r is not None:
            return u, r
        u *= 2
        time.sleep(0.1)
    return u, None


def probe_band_rect(w, h):
    """「屏幕时间码探针」那一条在**流画面**里的矩形 ⇒ `(x, y, w, h)` | `None`。

    **为什么标定要管它**（2026-10-07 实测 ✓）：探针是 A 机在画面**顶部**画的一条时间码方块带
    （`link.yaml` 的 `probe` 段：`x: 99, y: 19, cell: 16, gap: 2.25, bits: 40` @1366 流）——
    ⚠ 它**每帧都在变** ⇒ 帧差里一堆块，`_cursor_delta` 那句"恰好两个块 = 光标新旧位"的判据
    当场被淹 ✗（现场：60 单位量成 **-27.5px**、200 单位反而 **64.5px**，两次还一模一样 ✓）。
    ⇒ 量位移前把这条带**涂掉**（置 0 ✓）：它跟"取景/裁切"无关 ✓，纯粹是叠在画面上的已知符号 ✓。

    ⚠ **光标不会被误伤**：撞角归零落在画面 (0,0)、标定只沿 x 走 ⇒ 光标一直在带子的**上方**
    （带 y≈19..35、光标 y≈0..16 ✓），而带子从 x=99 才开始 ✓。
    ⚠ 读不到配置 / 探针没开 ⇒ `None` = **什么都不遮** ✓（绝不猜 ✗）。
    """
    try:
        from core.config import get
        if not bool(get("probe", "enabled", True)):
            return None
        x = float(get("probe", "x", 99))
        y = float(get("probe", "y", 19))
        cell = float(get("probe", "cell", 16))
        gap = float(get("probe", "gap", 2.25))
        bits = int(get("probe", "bits", 40))
    except Exception:                                  # noqa: BLE001
        return None
    if w <= 0 or h <= 0 or x < 0 or y < 0 or cell <= 0 or bits <= 0:
        return None
    # ⚠ x 方向多留（2026-10-07 ✓ 实测：A 机画出来的带子**右端比按配置算出来的宽十几像素**
    #   ⇒ 残留一块每帧都在变的方块在遮罩外面 ✗ 标定就量到它 ✗）；y 方向**收紧**（只留 2px ✓）
    #   —— 因为"撞角归零后光标停在 y≈0..16"，y 留太宽会把**光标自己**遮掉 ✗（实测过 ✓）。
    pad_x, pad_y = 40, 2
    x1 = max(0, int(x) - pad_x)
    y1 = max(0, int(y) - pad_y)
    x2 = min(int(w), int(x + bits * (cell + gap)) + pad_x)
    y2 = min(int(h), int(y + cell) + pad_y)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _blank(img, rect):
    """把 `rect` 涂成 0（量位移前盖掉已知叠层 ✓）。"""
    x, y, w, h = rect
    img[y:y + h, x:x + w] = 0


def measure(src, units, rounds=3, settle=0.15, dump=None, anchor=None):
    """发 `units` 个单位的移动 ⇒ 量画面里的位移（多轮取中位 ✓）。

    返回 `(dx, dy, n, anchor)`：
      · `n` = **几轮互相一致**（可信度 ✓）；· `anchor` = 这轮认出来的**旧位**（下一轮/下一档
      可以拿它当先验 ✓ —— 每轮都会走回来，那个落点基本不动 ✓）。

    ⚠ 量之前先把**探针带**涂掉（见 `probe_band_rect` ✓）—— 不涂的话帧差里全是探针的块 ✗。
    ⚠ 量位移时**告诉 `_cursor_delta` 我们发的是哪个方向**（`direction` ✓）：它就能用物理先验
      挑对那一对（同水平线 + 旧位在左边缘/锚点附近 ✓），并给出**带正确符号**的位移 ✓
      —— 2026-10-07 大位移那档就是靠这个才从噪声里认出来的 ✓（见 `_cursor_delta` 的说明 ✓）。
    """
    xs, ys = [], []
    band = None
    for i in range(rounds):
        before = _grab(src)
        if before is None:
            continue
        if band is None:
            band = probe_band_rect(before.shape[1], before.shape[0]) or ()
        time.sleep(settle)
        dinput.mouse_move(units, 0)              # 只动 x：y 方向单独不量（同一套指针速度）
        time.sleep(settle)
        after = _grab(src)
        if after is None:
            continue
        if band:
            _blank(before, band)
            _blank(after, band)
        d = _cursor_delta(before, after, direction=(1 if units > 0 else -1), anchor=anchor)
        _dump_round(dump, "u%d" % int(units), i, before, after, d)   # 现场帧（--dump ✓）
        # ⚠ **回程**：把光标挪回原处（`-units`，绝对值也 ≤127 ⇒ 走"单条指令"那条安全路 ✓）——
        #   撞角归零靠的是 `MOVE -4000` ✗，而**老固件会被截断**（见 `STEP_SMALL` 那段说明 ✓）
        #   ⇒ 不能指望它 ✓；没有回程的话多轮会一路把光标漂到屏幕边上 ✗（量到边缘就全废 ✓）。
        dinput.mouse_move(-units, 0)
        if d:
            # ⚠ 返回值现在是 `(dx, dy, 旧位)`：**符号可信**了 ✓（我们告诉了它方向 ✓
            #   见 `_cursor_delta` 的 `direction` ✓）⇒ 取**大小**照样稳（老固件 / 抗锯齿
            #   会让面积随机 ✓ 但方向已由指令定 ✓）；顺带**查 y**：横向指令的 y 抖动量
            #   应当远小于 x ✓，大得离谱 ⇒ 那两团根本不是光标（画面在动 / 探针没遮住 ✓）
            #   ⇒ **这一轮丢掉** ✓。
            dx_, dy_ = abs(float(d[0])), abs(float(d[1]))
            if dx_ < 3.0 or dy_ > max(6.0, 0.5 * dx_):
                time.sleep(0.2)
                continue
            xs.append(dx_)
            ys.append(float(d[1]))
            # ⭐ 认出了旧位 ⇒ 下一轮拿它当先验（比"左边缘"那条更准 ✓ 也能容忍轻微漂移 ✓）
            if len(d) > 2 and d[2] is not None:
                anchor = d[2]
        time.sleep(0.2)
    px, n_agree = agree_px(xs)
    if px is None:
        return None
    ys.sort()
    # ⚠ 第三个返回值是「**几轮互相一致**」（不是"跑了几轮" ✓）⇒ 一眼看出可信度 ✓
    return px, ys[len(ys) // 2], n_agree, anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3, help="每个单位数量几轮（取中位）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    ap.add_argument("--force", action="store_true", help="线性度不过也照写（不推荐）")
    ap.add_argument("--dump", default="", metavar="DIR",
                    help="把每轮**算法真正看到的**两帧 + 差分存进 DIR "
                         "（量不到 / 不过时先看它 ✓ —— 别再靠猜 ✗ 2026-10-07 的教训）")
    a = ap.parse_args()

    ok, why = _connect_input()
    print("[1/4] %s" % why)
    if not ok:
        return 2
    try:
        src, url = _open_stream()
    except Exception as e:                      # noqa: BLE001
        print("[X] 开不了流（%s: %s）" % (type(e).__name__, e))
        print("    检查 A 机在推流、config/link.yaml 的 stream.url / format 对得上。")
        return 2
    print("[2/4] 已开流：%s" % url)
    # ⚠ 量位移前会遮掉「屏幕时间码探针」那条带（它每帧都在变 ✓ 见 `probe_band_rect` ✓）
    #   ——**遮了就明说** ✓ 别默默换判据 ✗（2026-10-07：不遮时 60/200 两个单位数量出来的
    #   gain 自相矛盾，工具却报"线性度不过 ⇒ 疑似开了指针加速" ⇒ 把人引错方向 ✗）。
    print("      （量位移前遮掉「屏幕时间码探针」带：读 `link.yaml` 的 probe 段 ✓）")

    try:
        # 撞角归零 + **挪到探针带下面**（见 `PARK_DY` 那段说明 ✓）—— 不这么做的话，
        # 光标一往右走就钻进遮罩/带子里 ⇒ 帧差只剩一个块 ⇒ 工具会误报"量不到位移" ✗。
        # ⭐ `anchor` = 上一档认出来的**旧位**（= 撞角归零后那个落点 ✓ 每档都回同一处 ✓）
        #   ⇒ 拿它当先验比"左边缘"那条更准（能容忍轻微漂移 ✓）。第一档没有 ⇒ None ✓
        #   （那时 `_cursor_delta` 就用"旧位在左边缘"那条 ✓）。
        anchor = None
        slow = None                         # ⚠ 先定义：下面那个 try 万一早退，后面还要用它 ✓
        park_below_band()
        u_s, small = measure_auto(src, STEP_SMALL, a.rounds, dump=a.dump, anchor=anchor)
        if small and len(small) > 3 and small[3] is not None:
            anchor = small[3]
        park_below_band()
        u_b, big = measure_auto(src, STEP_BIG, a.rounds, dump=a.dump, anchor=anchor)
        if big and len(big) > 3 and big[3] is not None:
            anchor = big[3]
        # ⭐⭐ **校验档：大位移**（见 `STEP_VERIFY` 那段说明 ✓）—— 两档小位移的 gain 必须
        #   也能预测它 ✓；预测不准有两种可能（指针加速 / 老固件截断大数 ✓ 见那边的提示 ✓）。
        park_below_band()
        u_v, verify = measure_auto(src, STEP_VERIFY, a.rounds, dump=a.dump, anchor=anchor)
        # ⭐⭐⭐ **慢发对照**（见 `SLOW_TOTAL` 那段 ✓）—— ⚠ 必须在这里量 ✗：
        #   它要**读流**，出了下面那个 finally 源头就关了 ⇒ "量不到"（2026-10-07 踩过 ✓）。
        park_below_band()
        try:
            slow = measure_slow(src, SLOW_TOTAL, dump=a.dump, anchor=anchor)
        except Exception:                       # noqa: BLE001
            slow = None
        if slow and len(slow) > 2 and slow[2] is not None:
            anchor = slow[2]
    finally:
        try:
            src.close()
        except Exception:                       # noqa: BLE001
            pass

    if small is None or big is None:
        print("[X] 量不到光标位移（画面里找不到两个差块）")
        print("    已经自动放大过步长（最多 %d 倍 ⇒ 最大 %d 单位）还是量不到 ✓，按顺序查："
              % (8, max(u_s, u_b)))
        print("      ① ⭐ **指针速度滑块是不是被调得很低**（要在**中间档 6/11**）——")
        print("         速度极低时几十个单位只走一两个像素 ⇒ 两个块糊在一起 ⇒ 量不到 ✗")
        print("         （2026-10-07 真机就这么被误导过 ✓ 见 SKILL 297 ✓）")
        print("      ② 游戏停在静止画面上吗（画面自己在动就量不了）")
        print("      ③ 鼠标指针在画面里**可见**吗 —— 注意这里有**两种**不同病因，别混 ✗：")
        print("         · 游戏/系统把指针藏了（独占全屏、游戏自绘指针 ✗）")
        print("         · ⭐ **推流那条链本身没画指针**（`ddagrab` 的 `draw_mouse` ✓ "
              "2026-10-07 查过全仓没有一处 ✓）")
        print("         判别（30 秒、不改任何配置）：`data/_wiggle.py` 让指针来回扫，"
              "**人盯 A 机屏幕** + **看 B 机实时画面**：")
        print("           屏幕上动了、画面里没动 ⇒ **推流没画指针** ⇒ 给采集那条加 "
              "`draw_mouse=1` ✓（**别换采集方式**，那条推流是给游戏用的 ✓）")
        print("           两边都没动 ⇒ 鼠标通道问题（查固件 / 中继 ✓）")
        print("      ④ settle 够不够（%s 秒）" % 0.15)
        print("      ⑤ ⭐⭐ **先 `--dump DIR` 看现场**（算法真正看到的那两帧 + 差分伪彩 ✓）"
              "—— 别再靠猜 ✗ 这是 2026-10-07 绕了一晚换来的教训 ✓")
        return 2

    g_small = small[0] / float(u_s)
    g_big = big[0] / float(u_b)
    print("[3/4] %d 单位 ⇒ 移动 %.1f px（%d 轮一致）⇒ gain %.4f px/单位%s"
          % (u_s, small[0], small[2], g_small,
             "" if u_s == STEP_SMALL else "  ← 小步量不到，自动放大过 ✓"))
    print("      %d 单位 ⇒ 移动 %.1f px（%d 轮一致）⇒ gain %.4f px/单位"
          % (u_b, big[0], big[2], g_big))
    print("      （⚠ 这是**位移的大小**：帧差的符号随机不可信 ✓ 见 `_cursor_delta` ✓）")

    if g_small <= 0 or g_big <= 0:
        print("[X] 增益非正 ⇒ 方向对不上（正数指令光标往左走？）—— 不写文件 ✗")
        return 2
    rel = abs(g_small - g_big) / max(g_small, g_big)
    if rel > LINEAR_TOL:
        print("[!] **线性度不过**：两个单位数的 gain 差了 %.0f%%（> %.0f%%）——"
              % (rel * 100.0, LINEAR_TOL * 100.0))
        print("    最可能是指针加速开着（「提高指针精确度」）⇒ 关掉它再跑一次。")
        print("    加速开着时 count→像素是非线性的，用单一 gain 做绝对定位**会点偏** ✗。")
        if not a.force:
            print("    ⇒ **拒绝写文件**（宁可不做，也不乱点 ✓）。要强行写：--force")
            return 2
        print("    （--force ⇒ 照写；后果自负）")

    gain_x = (g_small + g_big) / 2.0
    print("[4/4] 采用 gain_x = %.4f px/单位（两个单位数的均值）" % gain_x)
    print("      ⚠ 纵向同一套指针速度 ⇒ gain_y 取同一个值（本工具只沿 x 量）")

    # ⭐⭐ **大位移校验**（2026-10-07 加的 ✓ 见 `STEP_VERIFY` 那段说明 ✓）：
    #   上面两档都是**小位移** —— 指针加速在小位移上就接近线性 ⇒ 它们**自证不了** ✗
    #   （真机就是这么栽的：小位移一致 ⇒ 写了 gain ⇒ 真点击时 `MOVE 1905 668`
    #    把光标甩到屏幕角、点了三次界面没变 ✗）。⇒ 必须拿一个**大位移**来验 ✓。
    rel_v = None
    if verify is None:
        print("[X] **大位移（%d 单位）量不到** ⇒ 没法校验这个 gain 在大位移上成不成立 ✗"
              % STEP_VERIFY)
        print("    多半是：画面在动 / 指针不在画面里 / 光标中途冲出了画面 ✓")
        print("    ⇒ **拒绝写文件**（这正是「不该写的那种 gain」✓ 要强行写：--force）")
        if not a.force:
            return 2
    else:
        moved_v = float(verify[0])
        pred_v = u_v * gain_x
        rel_v = abs(moved_v - pred_v) / max(1.0, pred_v)
        print("      校验档 %d 单位 ⇒ 实际移动 %.1f px（%d 轮一致）—— 按 gain 预测 %.1f px"
              " ⇒ 差 %.0f%%" % (u_v, moved_v, verify[2], pred_v, rel_v * 100.0))
        if rel_v > LINEAR_TOL:
            print("[X] **大位移对不上**：小位移算出来的 gain（%.4f px/单位）拿到 %d 单位上"
                  "差了 %.0f%% ✗" % (gain_x, u_v, rel_v * 100.0))
            print("    **两种可能，都要查**（小位移档已经守在 ≤127 的可信区间里 ✓ 所以问题"
                  "只出在「大数」上 ✓）：")
            print("    ① **指针加速（「提高指针精确度」）还开着** —— 小位移接近线性、"
                  "拉长就放大 ✓")
            print("    ② ⭐ **ProMicro 固件是老版本**（没有 `mouseMoveChunked`）—— 固件里")
            print("       `Mouse.move()` 的入参是 **signed char**，>127 的值会被**截断** ✗")
            print("       （4000 → −96、200 → −56 ✓ 设计文档 §6.1 记过 ✓）⇒ **撞角归零"
                  "（`MOVE -4000`）和大位移全废** ✗")
            print("       → 重烧固件：`remote_kbd/pro_micro/pro_micro.ino` ✓（烧完重连中继 ✓）")
            print("    ⇒ 拿这个 gain 做绝对定位**会把光标甩飞、点在不该点的地方** ✗")
            print("      （2026-10-07 真机现场：`MOVE 1905 668` 点了三次、界面没变 ⇒ 放弃 ✓）")
            print("    ⇒ **拒绝写文件**（宁可不做，也不乱点 ✓）。要强行写：--force")
            if not a.force:
                return 2
    # ⭐⭐⭐ **慢发对照：抓「提高指针精确度」**（见 `SLOW_TOTAL` 那段说明 ✓）
    #   上面三档（60 / 120 / 800）**都是一条指令发出去的**（快 ✗）⇒ 加速若开着，它们会被
    #   **一起**放大 ⇒ 三档互相"一致" ✓ 却**全都不是真 gain** ✗（第四跑就这么写了个假 gain ✓，
    #   真点击时 `MOVE 1905 668` 把光标甩到屏幕角、点了三次界面没变 ✗）。
    #   ⇒ 同样 800 单位**慢慢发**（每个 HID 报文只 10 counts ✓）再量一次：
    #     **两条 gain 必须相当** ✓ 不相当 = 加速还在 ⇒ 拒绝写文件 ✓。
    if slow is None:
        print("[!] 慢发对照没量到（%d 单位拆成 %d 一条 ✓）⇒ 少了一道判据 ✗"
              % (SLOW_TOTAL, SLOW_CHUNK))
        print("    多半是画面在动 / 光标出了可见区 ✓ 现场帧已存（`--dump` ✓）⇒ **先看图** ✓")
    else:
        g_slow = abs(float(slow[0])) / float(SLOW_TOTAL)
        rel_s = abs(g_slow - gain_x) / max(g_slow, gain_x)
        print("      慢发对照 %d 单位（拆 %d 一条 / 间隔 %.2fs ✓）⇒ 移动 %.1f px"
              " ⇒ gain %.4f px/单位 —— 与上面那条 %.4f 差 %.0f%%"
              % (SLOW_TOTAL, SLOW_CHUNK, SLOW_GAP, abs(float(slow[0])),
                 g_slow, gain_x, rel_s * 100.0))
        if rel_s > SLOW_TOL:
            print("[X] ⭐⭐ **快慢两条 gain 不一致**（差 %.0f%%）⇒ 「**提高指针精确度**」还在生效 ✗"
                  % (rel_s * 100.0))
            print("    它的加速按**速度**算 ⇒ 快指令被放大（实测 ≈2 倍 ✓）⇒ 小位移怎么量都像")
            print("    线性的（所以前三档能互相通过 ✗）⇒ 放到真点击 `MOVE ...` 上就**甩飞** ✗")
            print("    ⇒ A 机：鼠标设置 → 指针选项 → **取消勾选「提高指针精确度」** ✓")
            print("      （指针速度滑块停在正中 6/11 ✓）→ 重跑本工具 ✓")
            print("      **验收判据：快慢两条 gain 相等** ✓（这条闸自己会告诉你 ✓）")
            print("    ⇒ **拒绝写文件**（宁可不做，也不乱点 ✓）。要强行写：--force")
            if not a.force:
                return 2
    if a.dry_run:
        print("（--dry-run ⇒ 不写 %s）" % OUT)
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"gain_x": gain_x, "gain_y": gain_x,
         "probe": {"units": [u_s, u_b, u_v],
                   "measured_px": [small[0], big[0],
                                   (None if verify is None else verify[0])],
                   "n": [small[2], big[2], (None if verify is None else verify[2])],
                   "linear_rel": round(rel, 4),
                   "verify_rel": (None if rel_v is None else round(rel_v, 4)),
                   "source": "stream"}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写 %s ✓（断线重连「点服务器 / 点频道」下一拍就会用它 ✓）" % OUT)
    print("⚠ 换了「指针速度 / 加速状态 / 推流分辨率」都要**重新标定** ✓。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
