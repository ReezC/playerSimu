# -*- coding: utf-8 -*-
"""断线重连**实机开箱前的自检**（2026-10-07 ✓ 配 `docs/断线重连设计.md` §11 那份清单用）。

**它解决的问题**：开箱那一次很贵（要真机 + 账号 + 真断一次线），而整条链路里
有**四件事**会让这次白跑，而且**在界面上都看不出来** ✗：

  1. **输入设备不是 ProMicro** ⇒ `mouse_aim` 说"鼠标通道不可用" ⇒ "点服务器 / 点频道"
     那两步**一直停着不动手**（状态栏只说"暂时不动手"，人会以为程序坏了）；
  1b. ⭐ **中继卡死**（2026-10-07 现场踩到 ✓ 见 SKILL 296）：⚠ `mouse_available()` 那时
      **照样为真** ✗ ⇒ 上面那一条**查不出来** ✗ ⇒ 本工具另做一次**轻量探测**
      （TCP 3s + TLS 握手 5s，**不发任何指令** ✓ 见 `probe_relay` ✓）；
  2. **鼠标没标定**（`config/mouse_gain.json`）⇒ 同上，一个字节都不发 ✓（**故意的** ✓）；
  3. **界面锚点在这一路画面上认不出**（素材是 1920×1080 客户区 ✓，真机分辨率 / 窗口模式 /
     UI 版本一变就失准）⇒ 状态机压根不会开始（它只认已知界面 ✓）；
  4. **重连开关 / 四个点击比例**没配（比例越界 ⇒ 判成"暂时不动手" ✓ 同样看不出来）。

⛔ **本工具只读**：不按键、不动鼠标、不写任何文件 ✓（唯一的外部动作是**开流抓一帧**
   判界面 ✓，`--no-stream` 可以连这一步也省掉 ✓）。
⚠ 它**不替代** `tools/mouse_aim_calib.py`（那个要真发鼠标指令 ⇒ 会动到游戏机 ✓）；
   这里只说"标定过没有" ✓。

用法：
    python -X utf8 -m tools.reconnect_preflight              # 全查（含抓一帧判界面）
    python -X utf8 -m tools.reconnect_preflight --no-stream  # 不开流（不想连 / 流被占用）
    python -X utf8 -m tools.reconnect_preflight --scores     # 顺带打每个锚点的分数

退出码：0 = 四项都齐（可以去开箱 ✓）；2 = 还有拦路虎（下面会写清"去哪修" ✓）。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: 重连要用的界面（`decision/reconnect.py::STEPS` 那六个 ✓）—— 少一个都走不完流程。
NEED_UIS = ("login_err", "login", "channel_list", "channel_panel", "queue", "char_select")
#: 四个点击比例（名 → 人话 ✓）
RATIOS = (("reconnect_server_x", "服务器 X"), ("reconnect_server_y", "服务器 Y"),
          ("reconnect_channel_x", "频道 X"), ("reconnect_channel_y", "频道 Y"))
#: 六个状态机参数（名 → 人话 → 打印格式 ✓）—— 只是**报现状**，不判好坏 ✓
PARAMS = (("reconnect_enabled", "重连总开关（要不要自动走回游戏 ✓）", "%s"),
          ("reconnect_probe_after_lost_sec", "玩家框丢多久开始探界面（秒）", "%.2f"),
          ("reconnect_step_timeout_ms", "每步等界面变化的超时（毫秒）", "%s"),
          ("reconnect_queue_timeout_ms", "排队弹窗专用超时（毫秒）", "%s"),
          ("reconnect_max_retry", "同一步最多重试（次）", "%s"),
          ("reconnect_resume_auto", "回到游戏后请求恢复自动（要过体检 ✓）", "%s"))


def load_project_settings():
    """把"当前生效的那份决策参数"读进来（**只读、不落盘** ✓）。

    口径抄主窗口那条（`main_window._load_project_settings` 的"没打开项目"那一支 ✓）：
    没打开项目 ⇒ 用**最近打开的那个项目**那份；都没有 ⇒ 全默认值 ✓。
    ⚠ `set_save_hook(None)`：本工具**不许**把任何东西写进项目文件 ✗（那是界面的事 ✓）。
    """
    from decision.agent import set_save_hook, settings
    set_save_hook(None)
    src, where = None, "（没有最近打开的项目 ⇒ 全是默认值）"
    try:
        from gui.project import last_opened
        src = last_opened()
        if src is not None:
            where = str(getattr(src, "root", "?") or "?")
    except Exception as e:                                 # noqa: BLE001
        where = "（读最近项目失败：%s）" % e
    settings.from_dict((src.get("decision") if src is not None else None) or {})
    return settings, where


def probe_relay(host, port, tcp_timeout=3.0, tls_timeout=5.0):
    """**中继应答吗**：TCP(3s) + TLS 握手(5s) 轻量探测 ⇒ `(ok, 人话)`。

    ⛔ **不发任何指令** ✓（要连指令一起验，用 `tools/selftest_link` ✓ —— 它会占 relay
      那**唯一**一个客户端位几秒 ✓ ⇒ 开箱前先跑它、再开自动 ✓）。
    ⚠ 为什么非要有这一段（2026-10-07 现场 ✓）：`mouse_available()` **只说明后端是 ProMicro**
      ✗ —— 中继卡死时它照样为真 ✗，于是自检会说"通道就绪"、而标定/重连一上手就在第 1 步
      失败 ✗（现场：先"TCP 通但握手超时"，两分钟后"连 SYN 都不应" ✓ 见 SKILL **296** ✓）。
    """
    import socket
    import ssl
    import time
    if not host or not port:
        return None, "拿不到 `link.yaml` 的 kbd 段 ⇒ 跳过这一项 ✓"
    t0 = time.time()
    try:
        s = socket.create_connection((str(host), int(port)), timeout=tcp_timeout)
    except Exception as e:                                 # noqa: BLE001
        return False, ("中继 %s:%d **连不上**（%.1fs，%s）⇒ 在 A 机部署台把「键盘中继」"
                       "**停止 → 启动** ✓（`tools/selftest_link` 会报同一条 ✓）"
                       % (host, port, time.time() - t0, type(e).__name__))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    s.settimeout(tls_timeout)
    try:
        ss = ctx.wrap_socket(s, server_hostname=str(host))
        ver = ss.version()
        ss.close()
    except Exception as e:                                 # noqa: BLE001
        try:
            s.close()
        except Exception:                                  # noqa: BLE001
            pass
        return False, ("中继 %s:%d 端口通、但 **TLS 握手没应答**（%.1fs，%s）⇒ **中继卡死了**："
                       "A 机部署台「键盘中继」**停止 → 启动** ✓（机理见 SKILL 296 ✓）"
                       % (host, port, time.time() - t0, type(e).__name__))
    return True, ("中继 %s:%d 应答正常（TLS %s，%.2fs）"
                  % (host, port, ver, time.time() - t0))


def kbd_addr():
    """`link.yaml` 的 kbd 段 ⇒ `(host, port)`（读不到 ⇒ `("", 0)` ✓）。"""
    try:
        from core.config import get
        return get("kbd", "host"), int(get("kbd", "port", 9000) or 9000)
    except Exception:                                      # noqa: BLE001
        return "", 0


def check_input(s):
    """① 输入通道：**面板设的**输入设备（`input_device` ✓）+ 中继应答（**按需**探 ✓）。

    ⚠ 两个坑都是 2026-10-07 踩出来的 ✓：
      · ⛔ **别用 `mouse_aim.aim_available()` 判**：它读的是**本进程**的后端 ✗ —— 本工具
        **不连** relay（故意的：不该抢那唯一一个客户端位 ✓）⇒ 那句永远是"本地 SendInput
        模式" ✗，**哪怕面板已经切到 ProMicro 也会误报** ✓。改成读 `settings.input_device`
        （面板那个格子的值 ✓ 随项目存 ✓）。
      · ⛔ **面板设成 ProMicro 时别去探 relay**：relay 是"`accept` → `bridge`"的**单连接**
        循环（见 SKILL **296** ✓）⇒ 工作台正占着那位时，第二个客户端连上去只会**握手超时** ✗
        ⇒ 探了反而**误报"中继卡死"** ✗ ⇒ 那种情况改成"说明它被占着" ✓；要单独验通道用
        `tools/selftest_link`（⚠ 先停自动：它要抢那唯一一位几秒 ✓）。
    """
    lines, ok = [], True
    dev = str(getattr(s, "input_device", "") or "").strip().lower()
    host, port = kbd_addr()
    if dev in ("remote", "serial"):
        lines.append("✓ 面板输入设备 = **%s**（ProMicro %s）✓"
                     % (dev, "远程" if dev == "remote" else "本地直连"))
        lines.append("（工作台正占着 relay 那**唯一**一个客户端位 ⇒ 这里**不探**它 ✓；"
                     "要单独验通道：`python -X utf8 -m tools.selftest_link` ⚠ 先停自动 ✓）")
    elif dev == "local":
        ok = False
        lines.append("✗ 面板输入设备 = **local**（本机 SendInput，没有硬件鼠标）"
                     "⇒ 切到「ProMicro(远程)」✓")
        r_ok, r_why = probe_relay(host, port)
        lines.append(("✓ " if r_ok else "✗ ") + r_why if r_ok is not None else r_why)
    else:
        ok = False
        lines.append("✗ 读不到面板的输入设备（`input_device` = %r）⇒ 面板那格确认一下 ✓"
                     % (dev,))
    return ok, "\n      ".join(lines)


def check_gain():
    """② 标定（`config/mouse_gain.json` ✓）。返回 `(ok, 人话)`。"""
    try:
        from decision import mouse_aim
        g = mouse_aim.load_gain()
    except Exception as e:                                 # noqa: BLE001
        return False, "读不到标定：%s" % e
    if not g:
        return False, ("**没标定** ⇒ 点服务器 / 点频道那两步会**停着不动手** ✓"
                       "（故意的 ✓）\n"
                       "      → 在**跑工作台这台机器**上跑（不是游戏机 ✗ 指令经 relay 打过去 ✓），"
                       "且**先停掉实时预览**（抢同一个 UDP 端口 ✗）：\n"
                       "        python -X utf8 -m tools.mouse_aim_calib")
    return True, "已标定：%.4f / %.4f 画面像素每指令单位" % (float(g[0]), float(g[1]))


def check_templates():
    """③ 界面模板齐不齐。返回 `(ok, 人话)`。"""
    try:
        from perception import ui_state
    except Exception as e:                                 # noqa: BLE001
        return False, "import 不了 perception.ui_state：%s" % e
    miss = ui_state.missing_templates()
    if miss:
        return False, ("缺 %d 张模板图 ⇒ 判别器用不了 ✗：%s\n"
                       "      → python -X utf8 -m tools.make_ui_templates"
                       % (len(miss), "、".join(str(x) for x in miss[:6])))
    return True, "六类界面（%s）的锚点模板都在 ✓" % "、".join(
        ui_state.UI_NAMES.get(k, k) for k in NEED_UIS)


def check_live_frame(show_scores=False):
    """④ **真机这一路画面**上锚点认不认得出（唯一一条只有连流才能验的 ✓）。

    ⛔ 只读：**开流抓一帧** + 跑 `ui_state.detect` ✓ 不发任何键 / 不动鼠标 ✓。
    ⚠ 判出来是 `combat`（`None`）**不一定是错** —— 现在停在游戏里本来就判不出 ✓
      （见设计文档 §6.3："判不出"与"在游戏里"要靠玩家框区分 ✓）。所以这里报的是
      **"最高分的锚点是哪个、多少分"**，那才是"锚点准不准"的证据 ✓。
    """
    from perception import ui_state
    try:
        from tools.mouse_aim_calib import _grab, _open_stream
    except Exception as e:                                 # noqa: BLE001
        return None, "拿不到开流那两份实现（%s）—— 跳过这项 ✓" % e
    try:
        src, url = _open_stream()
    except Exception as e:                                 # noqa: BLE001
        _m = str(e)
        if "10048" in _m or "in use" in _m.lower():
            # 实测最常见的一条（GUI 的实时回路已经把这个 UDP 端口绑走了 ✓）——
            # ⚠ 别报成"没推流"：那会把人引到 A 机上去查 ✗
            return None, ("**这条流已经被占用了**（%s）⇒ 大概率是 GUI 正在收流 ✓\n"
                          "      → 用 `--no-stream` 跳过这一项，或先停掉实时预览再跑 ✓"
                          % _m)
        if "Errno 5" in _m or "I/O error" in _m:
            # 端口**空着、但没有数据**（2026-10-07 实测 ✓）—— 与 10048 是**两回事** ✗
            return None, ("**这条流开着但收不到数据**（%s）⇒ A 机没在推流 ✓\n"
                          "      → A 机部署台 →「**屏幕推流（ffmpeg）**」启动 ✓"
                          "（部署顺序：对时 → 探针 → 键盘 → 推流 → 小地图 ✓）" % _m)
        return None, ("开不了流（%s: %s）\n"
                      "      检查 A 机在推流 / `config/link.yaml` 的 stream.url ✓"
                      % (type(e).__name__, _m))
    try:
        fr = _grab(src)
    finally:
        try:
            src.close()
        except Exception:                                  # noqa: BLE001
            pass
    if fr is None:
        return None, "连上了 %s 但抓不到帧（推流没数据？）" % url
    try:
        h, w = int(fr.shape[0]), int(fr.shape[1])
    except Exception:                                      # noqa: BLE001
        h = w = 0
    ui = ui_state.detect(fr)
    per, _hit = ui_state.scores(fr)
    best = ""
    if per:
        name, sc = max(per.items(), key=lambda kv: kv[1])
        best = "最高分锚点：%s = %.3f（阈值 %.2f）" % (name, sc, ui_state.THRESHOLD)
        if show_scores:
            best += "\n      " + "  ".join("%s=%.2f" % (k, v)
                                           for k, v in sorted(per.items()))
    txt = "画面 %d×%d（流：%s）；现在判到：%s%s" % (
        w, h, url, ui_state.UI_NAMES.get(ui, "（判不出 —— 在游戏里 / 连击中都会这样 ✓）"),
        ("\n      " + best) if best else "")
    # ⚠ 只把"锚点准不准"当结论：判得出已知界面 ⇒ 准 ✓；判不出**不作判**（会在游戏里 ✓）
    return bool(ui) or None, txt


def check_settings(s):
    """⑤ 重连相关的设置与录屏开关（**只报现状** ✓）。返回 `(ok, 人话列表)`。"""
    lines, ok = [], True
    for name, label, fmt in PARAMS:
        try:
            lines.append("  · %s = %s" % (label, fmt % getattr(s, name)))
        except Exception:                                  # noqa: BLE001
            lines.append("  · %s = （读不到 ✗）" % label)
    for name, label in RATIOS:
        try:
            v = float(getattr(s, name))
        except Exception:                                  # noqa: BLE001
            lines.append("  · %s = （不是数字 ✗ ⇒ 那两步会判成「暂时不动手」）" % label)
            ok = False
            continue
        good = 0.0 <= v <= 1.0
        ok = ok and good
        lines.append("  · %s = %.4f%s" % (label, v, "" if good else "  ⚠ 越界（0~1 之外）"
                                          "⇒ 那两步会判成「暂时不动手」✗"))
    try:
        from core.config import load_live
        rec = bool(load_live().get("rec_disc", True))
        lines.append("  · %s = %s%s" % ("保留断线录屏（复盘最有用的东西 ✓）", rec,
                                        "" if rec else "  ⚠ 关着 ⇒ 事后没有现场可看 ✗"))
    except Exception as e:                                 # noqa: BLE001
        lines.append("  读不到录屏开关：%s" % e)
    return ok, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-stream", action="store_true", help="不开流（跳过「真机画面」那项）")
    ap.add_argument("--scores", action="store_true", help="顺带打每个锚点的分数")
    a = ap.parse_args()

    print("断线重连 · 实机开箱前自检（**只读**：不按键、不动鼠标、不写文件 ✓）\n")
    s, where = load_project_settings()
    print("[0/5] 参数来源：%s" % where)

    ok1, why1 = check_input(s)
    print("[1/5] 输入通道（面板输入设备 + 中继应答）：%s\n      %s"
          % ("✓ 就绪" if ok1 else "✗ 不可用", why1))

    ok2, why2 = check_gain()
    print("[2/5] 鼠标标定：%s\n      %s" % ("✓ 有" if ok2 else "✗ 缺", why2))

    ok3, why3 = check_templates()
    print("[3/5] 界面模板：%s\n      %s" % ("✓ 齐" if ok3 else "✗ 缺", why3))

    if a.no_stream:
        ok4, why4 = None, "（--no-stream：跳过 ✓）"
    else:
        ok4, why4 = check_live_frame(a.scores)
    print("[4/5] 真机画面 · 锚点认不认得出：%s\n      %s"
          % ("✓ 认得出" if ok4 else ("— 没判（正常）" if ok4 is None else "✗ 有问题"), why4))

    ok5, lines5 = check_settings(s)
    print("[5/5] 重连设置 / 录屏：%s" % ("✓ 比例都合法" if ok5 else "⚠ 有越界的比例（见下）"))
    for x in lines5:
        print(x)

    blockers = []
    if not ok1:
        blockers.append("输入通道不行 —— 看 [1/5] 那两行：面板「输入设备」切 ProMicro(远程)，"
                        "或在 **A 机**部署台把「键盘中继」停止 → 启动 ✓")
    if not ok2:
        blockers.append("鼠标没标定（跑 tools.mouse_aim_calib ✓）")
    if not ok3:
        blockers.append("界面模板缺（跑 tools.make_ui_templates ✓）")
    if not ok5:
        blockers.append("有点击比例越界（设置 → 保护与恢复 → 断线自动重连 ✓）")
    print()
    if blockers:
        print("结论：**还不能开箱** ✗ —— 先修这几件：")
        for b in blockers:
            print("  · " + b)
        print("（⚠ 「点服务器 / 点频道」那两步**故意**不动手 ⇒ 缺东西时它只会停在那儿 ✓"
              "  不是坏了 ✓）")
        return 2
    print("结论：**可以开箱** ✓\n")
    print("下一步（详见 docs/断线重连设计.md §11）：")
    print("  1. 确认游戏机：全屏 / 无边框 + 主显示器 + 指针速度 6/11 + 关「提高指针精确度」；")
    print("  2. 面板上：输入设备 = ProMicro(远程) ✓、开「自动」+ 开「检测到断线后自动走回游戏」；")
    print("  3. 断线（优先**拔网线**：客户端崩溃会让画面变桌面 ⇒ 判不出界面 ✗）；")
    print("  4. 盯着实时面板状态栏：先出现【界面：断线提示框】+「已停止自动，开始重连」，")
    print("     然后按步骤走；玩家框回来 ⇒ 「已回到游戏 —— …」；")
    print("  5. 复盘：tail 这个日志的三条事件（reconnect / reconnect_click / reconnect_resume）：")
    print("       Select-String -Path behavior.log -Pattern \"reconnect|screen_state\""
          " | Select-Object -Last 60")
    print("     录屏（断线那一整段 ✓）：data/recordings/disc_*.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
