"""【A 机运行】把游戏里的小地图**单独截屏**、推给 B 机（寻路定位用）。

**平时不用敲命令**：A 机上的一切都在「被控机部署台」的「小地图推流」卡片里
（区域框选、端口、帧率、启停、日志）。部署台用**同一个解释器**把本模块按那些
参数拉起来，卡片底部还能看到它实际执行的那条命令。下面这些是等价的手工做法，
部署台本身起不来时才用：

    python -m tools.minimap_push --pick            # 先在屏幕上框一次小地图区域
    python -m tools.minimap_push --x 100 --y 40 --w 200 --h 150
    python -m tools.minimap_push --zoom 4 --fps 30

**为什么要单独推一路**（而不是从主画面里抠）：
  · 主画面是 H.264 压过的，小地图上的黄点只有 2~5 像素，压完就糊了；
  · 主画面里小地图面板可能被 UI 挡、被玩家挪走、被缩放；
  · 这一路是**原始像素**（JPEG 质量 100），B 机拿到的黄点位置几乎无损。

**协议**（TCP，长度帧）：`[4 字节大端长度][JPEG 数据]` 循环。
为什么不是 UDP：一帧 JPEG 几十 KB，UDP 会分片、丢一片就整帧废；而"只留最新帧"
的语义本来就在**接收端**做（B 机 `perception/minimap.py`），用 TCP 反而简单可靠。

区域配置存在 `config/minimap_region.json`（`--pick` 会写进去），下次直接跑即可。

**按地图 id 存放 / 运行中换图**（2026-10-01）：
一条推流可以**不停进程就换抓取区域** —— B 机连一条新的连接、先报 `HELLO mmap-ctl`，
再发一行 `MAP <地图id>` ⇒ A 机从 `config/minimap_regions/<地图id>.json` 里取出那张图
自己的区域与 zoom，**下一帧**推的就是新区域的画面。不用新端口、不用重启、不用多起服务
（协议见 `tools/mmap_regions.py`，A/B 两边共用那一份）。
启动时也可以直接指定图：`--map-id <id>`（没给 ⇒ 用上面那份单值老配置，行为同以前 ✓）。

**"当前是哪张图"由 B 机推**（2026-10-02 用户定 ✓，原话："只靠 B 机推（A 机不能自己选）"）：
  · 启动解析优先级的**唯一一处**是 `mmap_regions.startup_region`：
    `--map-id` → **本机记着的当前图**（`config/minimap_current.json` ✓ 上次 B 机推来的）
    → 老单值兜底 → **一块都没有**；
  · ⭐ **一块都没有也照常起来等**（不再直接退出 ✗）：A 机没框过也能先把服务开着，
    B 机一句 `MAP <id>` 之后——库里**有**那张图 ⇒ 立刻开始推；**没有** ⇒ 回 `ERR no-such-map`，
    人再去部署台卡片里框一次（那时 id 已经知道了 ✓）；
  · 每次 `MAP` 处理成功都**顺手把 id 记在本机**（`mmap_regions.set_current` ✓）——
    部署台的「当前地图」那一行读它，框选也就是存到这张图名下 ✓。
"""

import argparse
import json
import select
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import mmap_regions                              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGION_FILE = ROOT / "config" / "minimap_region.json"

# ⭐⭐ **fps 60 / quality 80**（用户 2026-10-03 ✓ 原话："A机 fps 已经 60 了，quality 允许降；
#    游戏（小地图）应该有 200+ [刷新率]"）：
#   · 小地图是**权威世界坐标**（`deploy/services.py` 那条口径 ✓）⇒ 帧率越高坐标越新 ✓；
#     以前默认 30 是"怕 B 机吃力"（那时定位是**每个推理帧**才做一次 ✗ 见
#     `gui/live_thread.py` 的高频定位线程 ✓ 现在已经不是瓶颈了 ✓）；
#   · `quality` 100 → 80：面板只有一两百像素宽 ✓ 100 是浪费 —— 降到 80 **肉眼看不出来**，
#     而 JPEG 体积约减半 ⇒ A 机编码时间（它自述"编码是大头"✓）和带宽都省一半 ✓。
#   ⚠ 老 `config/deploy.json` 里存着 30/100 的机器**照旧用自己那份**（这里只是默认值 ✓）——
#     真要提速，得把那台 A 机的 fps/quality 也改过来（部署台卡片上改 ✓ 或删掉那两个键 ✓）。
DEFAULTS = {"bind": "0.0.0.0", "port": 5003, "zoom": 3, "fps": 60,
            "quality": 80, "region": None}


def load_cfg():
    cfg = dict(DEFAULTS)
    if REGION_FILE.exists():
        try:
            cfg.update(json.loads(REGION_FILE.read_text(encoding="utf-8")) or {})
        except Exception:
            pass
    return cfg


def save_cfg(cfg):
    REGION_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGION_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def ask_region(owner=None, wait=0.0):
    """抓全屏图，弹框选窗让人拖出小地图区域 → (x, y, w, h)；取消返回 None。

    **交互走全仓库唯一那份实现**（`gui.region_picker`，带放大镜 / 像素网格 /
    `+/-` 倍数，见 docs/UI规范.md §8）：小地图面板的边框就差一两个像素，没有
    放大镜只能靠感觉 —— 而框大了会把血条/聊天一起推给 B 机，那边的现象是
    「底图对不上」，看着像寻路坏了。

    **为什么能在部署台里被调用**（以前不行）：老实现结尾是 `app.exec_()` —— 在
    一个**已经有事件循环**的 Qt 程序（部署台）里再跑一次会报
    「The event loop is already running」，而且是卡住不返回，按钮看着像死了。
    region_picker 用的是 `QDialog.exec_()`（嵌套的模态循环，Qt 本来就支持），
    所以命令行和部署台两条路共用同一个实现。

    owner 传了就先把它的窗口藏起来再抓屏（region_picker 负责），不然抓到的
    截图里盖着部署台自己；框完/取消都会 show() 回来。

    wait > 0 时先等这么多秒再抓屏 —— 命令行用：给用户切回游戏窗口的时间。
    """
    if wait > 0:
        print("  %.0f 秒后抓屏 —— 现在切到游戏窗口，让小地图露出来（别被挡）"
              % wait, flush=True)
        time.sleep(wait)

    from gui.region_picker import select_screen_region
    return select_screen_region(parent=owner, hide_owner=True)


def _already_listening(port, host="127.0.0.1", timeout=0.4):
    """这个端口上**已经有东西在监听**吗（连一下就知道）。

    为什么要有这一步（2026-09-26 排查"B 机连上却一帧都收不到"）：
    `SO_REUSEADDR` 让**第二个推流**也能 bind 成功 ⇒ 于是可能同时有两个推流进程，
    而 B 机的连接只会落到**其中一个**上 —— 落到旧的那个（比如上次没关干净的、
    或者卡在抓屏上的）时，现象就是"连得上、永远没帧"，且新起的这个日志里什么都看不到。
    """
    import socket as _s
    try:
        with _s.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:                       # noqa: BLE001
        return False


# ---------------- 控制连接（B 机 → 本进程：换图 / 问状态） ----------------

def _close(sock):
    try:
        sock.close()
    except Exception:                                        # noqa: BLE001
        pass


def _send_line(sock, raw):
    """往控制连接上回**一行**（结尾补 `\\n`）。写不出去就当那一路已经没了。"""
    try:
        sock.sendall(raw if raw.endswith(b"\n") else raw + b"\n")
        return True
    except Exception:                                        # noqa: BLE001
        return False


def _sniff_hello(c, timeout):
    """新连接**第一行**是什么（等不到就回 `b""` ✓）—— 判断它是不是"控制连接" ✓。

    ⚠⚠ 用户 2026-10-03 ✓ 现场铁证：A 的控制台里**一行「控制连接 … 已连上」都没有** ✗、
      全是「B 机 … 已连上」✓；B 那边报「`mmap_map=失败 A 机的回执看不懂：''`」✓
      ⇒ B 明明发了 `HELLO mmap-ctl`，A 却把它**判成了收帧连接** ⇒ ① A 一句话不回 ✓
      ② id 永远记不上 ⇒ 部署台永远「还没有」✓。

    病根：**原来只 `recv(64)` 一次** ✗ —— TCP **不保证**一次 `recv` 就给整行，拿到
      `b"HELLO "` 这种半截时，`hello[:20] == b"HELLO mmap-ctl"` **必然不成立** ✗
      （20 字节都没凑齐 ✓）⇒ 于是一个字都没错、但**永远认不出来** ✓。

    改：**循环收，直到见到换行（或整段超时）** ✓ —— 总等待时长就是调用方给的
      `timeout`（= 原来那个握手窗口 ✓）：**收帧客户端本来一声不吭，最坏还是等这么久** ✓
      不会拖慢它的首帧 ✓；而控制客户端说完第一行就立刻被认出来 ✓。

    ⚠⚠ 回值**不只是第一行**：可能还带着**同一口气里跟来的字节**（多半就是紧跟的那条
      `MAP <id>` ✓ —— B 是"连上就把 `HELLO` 和 `MAP` 一起发出来"的 ✓，两下 `sendall`
      只隔几微秒，而我们**还没轮到 `accept()`**，所以两个包早都躺在内核缓冲里了 ✓）。
      ⇒ 调用方**必须把余下的字节交给 `_ctl_feed`**（`_accept_ctl` 就是干这个的 ✓），
        **别丢掉** ✗ —— 丢掉就是 2026-10-04 那个"B 发的 `MAP` 石沉大海"✓。
    """
    end = time.time() + max(0.05, float(timeout))
    # ⚠⚠ **超时得由这里自己设**（2026-10-04 ✓ 现场：整套自检**永远跑不完** ✗）：
    #   上面那个"整段超时"（`while … time.time() < end`）**只在 socket 本身带超时时才成立** ——
    #   调用方给的是一个**阻塞 socket**（`settimeout(None)`，比如 `socket.socketpair()` 的默认值）
    #   时，`recv` 会**一直不返回** ⇒ 循环条件根本没机会判 ⇒ 函数挂死 ✗。
    #   `main()` 那条路恰好先 `settimeout(CTL_HANDSHAKE_TIMEOUT)` 才调它 ⇒ 一直没暴露；
    #   而 `selftest_mmap_regions.t_sniff_hello_survives_split`（用 socketpair ✓）第 ② 段
    #   "一声不吭的收帧客户端"就卡在这儿 ⇒ **整套自检一条都跑不完**（用户："每次跑都要等
    #   几个小时"✓ 根因就是它）。
    _prev_to = None
    try:
        _prev_to = c.gettimeout()
        c.settimeout(max(0.05, float(timeout)))
    except Exception:                                # noqa: BLE001 —— 假 socket（用例替身）没这两个方法也算过 ✓
        _prev_to = None
    try:
        buf = b""
        while b"\n" not in buf and len(buf) < 64 and time.time() < end:
            try:
                chunk = c.recv(64 - len(buf))
            except Exception:                        # noqa: BLE001 —— 超时/对端关了都算"没有" ✓
                break
            if not chunk:
                break
            buf += chunk
    finally:
        try:
            c.settimeout(_prev_to)                   # 原样还回去（调用方自己会按需再设 ✓）
        except Exception:                            # noqa: BLE001
            pass
    return buf


def _answer_ctl(sock, raw, state, log=print):
    """处理**一条**控制命令 ⇒ 改 `state`（区域/zoom）并回一行。

    `state = {"map_id": str, "box": [x,y,w,h], "zoom": int}` —— 它**故意是可变的**：
    帧循环每拍都读它 ⇒ 改完**下一帧**抓的就是新区域（"不重启也能换图"的全部机关 ✓）。

    ⚠ `MAP` 只认**库里有**的那份（`mmap_regions.load`），查不到 ⇒ 回 `ERR` 且
      **原来的区域一个字节都不动**：半切换的状态（框变了图没变 / 反过来）比"没切成"
      难查得多 ✗。
    """
    parts = raw.decode("utf-8", "replace").split()
    cmd = (parts[0] if parts else "").upper()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == mmap_regions.CMD_PING:
        return _send_line(sock, b"PONG")
    if cmd == mmap_regions.CMD_LIST:
        return _send_line(sock, mmap_regions.list_reply(
            mmap_regions.list_ids()).encode("utf-8"))
    if cmd == mmap_regions.CMD_STATE:
        # ⚠ 还没有区域（启动时一块都没框 ⇒ 等 B 机 `MAP`）⇒ 回**四个 0**：
        #   协议里 `STATE - x y w h zoom` 的 `-` 就是"没指定图"那一档 ✓
        #   （`state_reply`/`ok_reply` 只会摊 int ⇒ 别把 None 丢进去 ✗）。
        return _send_line(sock, mmap_regions.state_reply(
            state.get("map_id") or "-", state.get("box") or (0, 0, 0, 0),
            state.get("zoom") or 1).encode("utf-8"))
    if cmd == mmap_regions.CMD_MAP:
        got = mmap_regions.load(arg)
        if not got:
            # ⭐⭐ **未命中也要把 id 记下来**（用户 2026-10-03 ✓ 现场："改成收流了 A 那行还是
            #   「还没有」"✗）。本模块开头那段设计**本来就是这么说的** ✓：
            #     "没有 ⇒ 回 `ERR no-such-map`，人再去部署台卡片里框一次（**那时 id 已经
            #      知道了** ✓）"
            #   ⚠ 可代码只写了回 `ERR` ✗ ⇒ 部署台那行永远读不到 id ⇒ 人**没法**按这张图的
            #     名字去框（框选是存到"当前这张图"名下的 ✓）⇒ **死结** ✓✓。
            #   ⚠ 记的是**"当前地图"这个标签**（`set_current` ✓），**不是**要把抓取区域换掉 ✗ ——
            #     `state` 一个字节都不动（还是"原来的区域"✓），所以上面那条"半切换"的纪律
            #     照样守着 ✓：唯一变化是"人现在知道该往哪张图名下框" ✓。
            try:
                mmap_regions.set_current(arg, by="B机(未命中)")
            except Exception:                       # noqa: BLE001 —— 记标签失败也不许打断回执 ✗
                pass
            # 光说"找不到"人会重框错地方 ⇒ 顺手告诉他**已经配了哪些**
            _ids = mmap_regions.list_ids()
            why = ("这张图还没框过（%s）；已配的有：%s"
                   % (arg or "没给 id", "、".join(_ids) if _ids else "（一张都没有）"))
            return _send_line(sock, mmap_regions.err_reply(
                "no-such-map", why).encode("utf-8"))
        state["map_id"] = got["map_id"]
        state["box"] = [got["x"], got["y"], got["w"], got["h"]]
        state["zoom"] = got["zoom"]
        # ⭐ **顺手记在 A 机本地**（用户 2026-10-02："当前是哪张图由 B 机推"✓）：部署台的
        #   「当前地图」那一行读它、框选也是存到这张图名下 ⇒ "B 机说在跑哪张图"和"A 机
        #   往哪张名下框"从此是**同一件事** ✓（不记的话，框选时人根本不知道该填哪个 id ✗）。
        try:
            mmap_regions.set_current(got["map_id"], by="B机")
        except Exception as e:                              # noqa: BLE001
            log("[%s] ⚠ 记不下「当前图」（%s: %s）—— 图切了，但部署台那边可能还显示旧的"
                % (time.strftime("%H:%M:%S"), type(e).__name__, e))
        _send_line(sock, mmap_regions.ok_reply(
            got["map_id"], state["box"], got["zoom"]).encode("utf-8"))
        log("[%s] 按 B 机要求换图 → %s　区域 (%d,%d) %dx%d　zoom=%d（下一帧生效）"
            % (time.strftime("%H:%M:%S"), got["map_id"], got["x"], got["y"],
               got["w"], got["h"], got["zoom"]))
        return True
    return _send_line(sock, mmap_regions.err_reply(
        "bad-command",
        "不认识的命令 %r（可用：MAP / STATE / LIST / PING）" % (cmd,)).encode("utf-8"))


def _ctl_feed(rec, buf, state, log=print):
    """把一段字节喂给一条控制连接：**完整行**交给 `_answer_ctl`，**没成行的尾巴留着** ✓。

    `rec = [sock, 对端名, 尾巴]` —— `rec[2]` 就是"上一口气里只到了半条命令"的那半个，
    必须留着等下一段（TCP 只保证字节序、**不保证按行到** ✓）。

    ⭐⭐ 为什么这个函数非有不可（2026-10-04 ✓ 现场：B 的 `MAP` 石沉大海）：
      B 是**连上就一口气把 `HELLO` 和 `MAP <id>` 都发出来**的
      （`perception/minimap.py::MiniMapClient.switch_map` ✓），而那两下 `sendall` 之间只隔
      几微秒 —— A 那边多半**还没轮到 `accept()`**，两个包就都已经躺进内核接收缓冲 ⇒ 握手
      那次 `recv` **一次就把整行 hello + 整行 MAP 拿回来了** ✓。
      原来认完 hello 就把这坨字节**丢掉** ⇒ `MAP` 永远没人处理 ⇒ B 等 3 秒收到空气 ⇒
      `mmap_map=失败 A 机的回执看不懂：''` ✓ —— 而且**每次都这样**（不是偶发 ✓），
      所以现场看着像"A 机压根不支持换图" ✗。
    """
    c, name = rec[0], rec[1]
    pend = (rec[2] if len(rec) > 2 else b"") + buf.replace(b"\r\n", b"\n")
    lines = pend.split(b"\n")
    rec[2:] = [lines.pop()]        # 尾巴 = 最后一段**没有换行**的（可能是半行 ✓）
    for line in lines:
        line = line.strip()
        if line:
            _answer_ctl(c, line, state, log=log)


def _accept_ctl(controls, c, name, hello, state, log=print):
    """新连接第一口气 `hello` 认出来是**控制连接** ⇒ 登记 + **就地把它喂掉** ⇒ True；否则 False。

    `hello` 就是 `_sniff_hello` 的原样回值：**第一行 + 同一口气里跟来的字节** ✓ ——
    跟来的那部分（多半是 `MAP <id>` ✓）**一个字都不许丢** ✗（见 `_ctl_feed` 的说明 ✓）。

    ⚠ 抽成独立函数是为了**能单测**：这段原来嵌在 `main()` 的 accept 循环里（要 Qt + 抓屏 +
      真 socket 才跑得到 ✗）⇒ "余下字节被丢掉"这个 bug 才藏了这么久 ✓
      （`selftest_mmap_regions` 的 `t_hello_and_map_in_one_chunk` 与端到端那条都在钉它 ✓）。
    """
    _hl = mmap_regions.CTL_HELLO.rstrip(b"\n")
    if hello[:len(_hl)] != _hl:
        return False                   # 不是控制连接 ⇒ 交给收帧那条路（老客户端一声不吭 ✓）
    rest = hello[len(_hl):]
    while rest[:1] in (b"\r", b"\n"):
        rest = rest[1:]                # 剥掉握手行自己的换行
    c.settimeout(10)
    rec = [c, name, rest]              # ⭐ 尾巴**必须留着**：它就是紧跟的那条 `MAP` ✓
    controls.append(rec)
    log("[%s] 控制连接 %s 已连上（发 `MAP <地图id>` 换图，不重启）"
        % (time.strftime("%H:%M:%S"), name))
    _ctl_feed(rec, b"", state, log=log)      # ⭐ 同一口气里已经到的命令**当场**处理 ✓
    return True


def _poll_controls(controls, state, log=print):
    """把每条控制连接上**已经到了**的命令处理掉 ⇒ 返回还活着的那几条。

    ⚠ 用 `select(…, 0)` 问一句"有没有数据"，**不给 socket 设超时去 recv**：帧循环
      30 拍/秒，任何阻塞 recv 都会直接拖低帧率 —— 而这里多数时候一根毛的数据都没有。

    ⚠ 必须在 `if not clients: continue` **之前**调用：B 机常常是"先问一句再连"（切图
      那一刻也可能压根没收流），少了这一步，命令会一直没人理 ✗ 且看起来像"A 机挂了"。

    ⚠ 字节一律走 `_ctl_feed`（**接到的可能是半条命令** ⇒ 尾巴要留 ✓；也可能是**两条**
      命令连着来（`PING` + `LIST` ✓）⇒ 逐行处理 ✓）—— 别再在这里自己 `split` ✗，
      那会把"半行"当成一条命令发出去 ✓。
    """
    if not controls:
        return controls
    try:
        rd, _, _ = select.select([rec[0] for rec in controls], [], [], 0.0)
    except (OSError, ValueError):                # 有一路已经废掉了 ⇒ 退化成"都没数据"
        rd = []
    keep = []
    for rec in controls:
        c, name = rec[0], rec[1]
        if c in rd:
            try:
                buf = c.recv(8192)
            except Exception as e:                           # noqa: BLE001
                log("[%s] 控制连接 %s 读不了（%s: %s），移除"
                    % (time.strftime("%H:%M:%S"), name, type(e).__name__, e))
                _close(c)
                continue
            if not buf:                                      # 对端关了
                _close(c)
                continue
            _ctl_feed(rec, buf, state, log=log)               # ⭐ 半行/多行都由它收口 ✓
        keep.append(rec)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description="A 机：小地图截屏推流")
    ap.add_argument("--bind", default=None, help="监听地址，默认 0.0.0.0")
    ap.add_argument("--port", type=int, default=None, help="监听端口，默认 5003")
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--y", type=int, default=None)
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--zoom", type=int, default=None,
                    help="抓完放大几倍再发（小地图像素太小，放大只是为了让黄点"
                         "在传输/解码里少掉一点细节；不会增加信息量）")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--quality", type=int, default=None, help="JPEG 质量，默认 100")
    ap.add_argument("--pick", action="store_true", help="先框选一次区域并保存")
    ap.add_argument("--no-gui", action="store_true",
                    help="不开窗口。**目前无实际作用**（保留兼容）：不加 --pick 时"
                         "本来就不开窗口，抓屏直接抓桌面")
    ap.add_argument("--force", action="store_true",
                    help="端口上已经有东西在监听时也强行启动（默认拦住 —— 两个推流"
                         "同端口会让 B 机连到旧的那个，表现为「连上却收不到帧」）")
    ap.add_argument("--map-id", dest="map_id", default=None,
                    help="启动时就推**这张图**的区域 —— 从 "
                         "config/minimap_regions/<id>.json 取（连同那张图自己的 zoom ✓）。"
                         "不给 ⇒ 用 config/minimap_region.json 那份单值老配置，"
                         "行为和以前一模一样 ✓。运行途中 B 机还能发 `MAP <id>` 换掉，"
                         "**不用重启本进程**。")
    args = ap.parse_args()

    from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
    from PyQt5.QtGui import QGuiApplication
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    cfg = load_cfg()
    for k in ("bind", "port", "zoom", "fps", "quality"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    if args.pick:
        # 命令行框选：等 5 秒（要切回游戏窗口）—— 部署台那条路不用等，
        # 它会把窗口自己藏起来（见 ask_region）。
        r = ask_region(wait=5.0)
        if not r:
            print("没框选（取消，或框得太小）")
            return 1
        cfg["region"] = list(r)
        save_cfg(cfg)
        print("已保存区域: x=%d y=%d w=%d h=%d → %s" % (r[0], r[1], r[2], r[3],
                                                      REGION_FILE))
        # ⭐ 同时给了 `--map-id` ⇒ **这次框的这块就归这张图**（存进 per-map 库 ✓）。
        #   带 box 和 zoom 一起存是有讲究的：标定是按某个 zoom 标出来的，两者必须是
        #   同一份，换图时才知道该用它自己那个 zoom（见 `mmap_regions.save`）。
        _pmid = str(args.map_id or "").strip()
        if _pmid:
            _p = mmap_regions.save(_pmid, r[0], r[1], r[2], r[3],
                                   zoom=int(cfg["zoom"]),
                                   note="命令行 --pick 存进来的")
            print("并已存给「%s」这张图 → %s（zoom=%d）\n"
                  "  以后推它就跑：python -m tools.minimap_push --map-id %s"
                  % (_pmid, _p, int(cfg["zoom"]), _pmid))

    if args.x is not None:
        # 部署台走这条：区域在它的卡片里框、由命令行传进来。
        # 顺手写一份到 minimap_region.json，命令行单跑时不用再框一次。
        cfg["region"] = [args.x, args.y, args.w, args.h]
        save_cfg(cfg)

    # ⭐ **区域按地图 id 取**（2026-10-01；2026-10-02 起**没框过也允许先起来**）：
    #   优先级只有一处（`mmap_regions.startup_region` ✓ 部署台自检共用那一份）：
    #     `--map-id` → 本机记着的"当前图"（B 机推来的 ✓）→ 老单值兜底 → **一块都没有**。
    #   ⚠ 点名了 `--map-id` 却没有那张图 ⇒ **报错退出**（2）：这时人是在明确要那张图，
    #     拿别的图顶上会推一片无关画面、B 机那边表现为"底图对不上"✗（难查）。
    #   ⚠ **一块都没有也照常起监听**（不再 return 2 ✗）：用户 2026-10-02 定"当前是哪张图
    #     **只由 B 机推**" ⇒ A 机必须能先开服务、等 B 机那句 `MAP <id>`（协议里 `STATE`
    #     的"还没指定图"那一档就是给这个状态留的 ✓）。开着不动比"起不来"省事得多 ✓。
    _mid = str(args.map_id or "").strip()
    _start = mmap_regions.startup_region(_mid)
    if _mid and not _start:
        _ids = mmap_regions.list_ids()
        print("「%s」这张图还没有框选数据。\n"
              "  文件：%s\n"
              "  库里现在有：%s\n\n"
              "  先在这台机上框一次：部署台「小地图推流」卡片（B 机开始实时时会告诉\n"
              "  本机现在跑的是哪张图，卡片的「当前地图」会跟着变 ✓）。"
              % (_mid, mmap_regions.path_of(_mid),
                 "、".join(_ids) if _ids else "（一张都没有）"))
        return 2
    if _start:
        mid0, box0, zoom0, _src = _start
    else:
        mid0, box0, zoom0, _src = "", None, max(1, int(cfg["zoom"])), "无"

    quality = int(cfg["quality"])
    fps = max(1, int(cfg["fps"]))
    screen = QGuiApplication.primaryScreen()

    # **端口上已经有东西在听** ⇒ 大声拦住（见 _already_listening 的说明）。
    # 不拦的话：两个推流同时在跑，B 机可能连到旧的那个 ⇒ "连上却一帧都收不到"，
    # 而且新起的这个日志里一片安静，根本看不出问题在哪。
    if _already_listening(cfg["port"]) and not args.force:
        print("⚠ 端口 %d 上**已经有东西在监听** —— 多半是上一个「小地图推流」还开着。\n"
              "  两个推流同端口时，B 机只会连到其中一个：连到旧的那个就表现为\n"
              "  「连得上、但一帧都收不到」。\n"
              "  先把它关掉（部署台那张卡片点停止 / 任务管理器里的 python -m "
              "tools.minimap_push），或者确认那是别的服务后加 --force 强制启动。"
              % int(cfg["port"]))
        return 3

    # **A 机监听、B 机连进来**（和 A 机已有的 relay / clock_server 同一方向）：
    # 这样 B 机只用 A 机的 IP（link.yaml 的 a_host 本来就有），A 机不用知道 B 的地址；
    # 防火墙也只需在 A 机放行这一个端口。
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((cfg.get("bind") or "0.0.0.0", int(cfg["port"])))
    srv.listen(8)
    # **非阻塞 accept**：同一帧要广播给**每一路**客户端（见下面 clients 的说明）
    srv.setblocking(False)
    # ⭐ **当前在推哪一块**是**可变的**状态，不再是常量：B 机连一条控制连接发一句
    #   `MAP <地图id>` 就能改它 ⇒ 帧循环每拍读它，下一帧生效（不用重启进程 ✓）。
    # ⭐ `box` 允许是 `None`（**还没框过** ✓）：那时只接受控制命令、一帧都不抓
    #   （B 机发 `MAP <id>` 且库里**有**这张图 ⇒ 当场补上区域，见 `_answer_ctl` ✓）。
    state = {"map_id": mid0, "box": (list(box0) if box0 else None),
             "zoom": int(zoom0)}
    if box0:
        print("监听 %s:%d   图=%s（%s）   区域 (%d,%d) %dx%d   zoom=%d   %d fps   JPEG q=%d"
              % (cfg.get("bind") or "0.0.0.0", int(cfg["port"]), mid0 or "（老单值）",
                 _src, box0[0], box0[1], box0[2], box0[3], zoom0, fps, quality))
    else:
        print("监听 %s:%d   ⚠ **还没有小地图区域** —— 先开着，等 B 机告诉本机"
              "「现在跑的是哪张图」（`MAP <地图id>`）\n"
              "  　收到之后：库里**有**这张图 ⇒ 立刻开始推；没有 ⇒ 回一句 ERR，"
              "在部署台「小地图推流」卡片里点「框选…」框一次即可（id 已经知道了 ✓）"
              % (cfg.get("bind") or "0.0.0.0", int(cfg["port"])))
    print("等 B 机连进来…（**换图时发 `MAP <地图id>` 即可，不必重启本进程** ✓）")

    # **支持多个 B 机客户端**（2026-09-26 修）：原来一次只服务一个连接 —— B 机上
    # 只要多一条客户端（工作台面板的定位 + 「标定…」弹窗 + 忘了关的命令行探针都是
    # 各连一条），**后连的那条就只能排队、永远收不到帧**：现象是"TCP 连得上、
    # 一帧都不来"（B 机报「等帧超时」），而 A 机日志里一片正常 —— 极难查。
    # 现在每帧**广播**给所有连着的客户端，谁都不会被饿死。
    clients = []             # [[sock, 对端名, 本次已发帧数, 连上的时刻, 上一帧耗时ms], ...]
    _skip = {}               # 对端名 → 累计**跳帧**数（跟不上就跳 ✓ 见帧循环那段 ✓）
    controls = []            # [[sock, 对端名], ...] —— 只"说话"（换图/问状态）的那几路
    n = 0
    t0 = time.time()
    #: 「还没收到图」那条提醒上次说了什么时候（**只提醒、不刷屏** ✓ 见帧循环那一段 ✓）
    _no_box_note = 0.0
    while True:
        try:
            # ① 收新连接（非阻塞：有多少收多少，别让谁排队等）
            while True:
                try:
                    c, peer = srv.accept()
                except (BlockingIOError, socket.timeout):
                    break
                except OSError:
                    break
                name = "%s:%d" % (peer[0], peer[1])
                # ⭐ **先认一下是不是控制连接**：新连上来的第一行如果是 `HELLO mmap-ctl`
                #   ⇒ 它只想说话（换图 / 问状态），**不收帧** ⇒ 千万别塞进 clients，
                #   否则每帧往它那儿推 JPEG，它一句都看不懂。
                #   ⚠ 等握手的时间必须很短（0.25s）：正常的收流客户端连上后**一声不吭**
                #   等着收帧 ⇒ 它只会被这个等待拖慢"首帧"，拖久了就是"B 机等帧超时"。
                c.settimeout(mmap_regions.CTL_HANDSHAKE_TIMEOUT)
                # ⚠⚠ 见 `_sniff_hello`：**一次 `recv` 认不出来**（"半截 hello"那条病根 ✓）；
                #   而它回值里**还带着同一口气跟来的字节**（B 是"连上就把 `HELLO` 和 `MAP`
                #   一起发"的 ⇒ 多半紧跟一条 `MAP` ✓）⇒ **一个字都不许丢** ✗
                #   （丢掉的后果就是现场那个"B 的 `MAP` 石沉大海、回执看不懂：''" ✓）
                #   ⇒ 认连接 + 登记 + **就地把余下字节喂掉**，全部收在 `_accept_ctl` 一处 ✓。
                hello = _sniff_hello(c, mmap_regions.CTL_HANDSHAKE_TIMEOUT)
                if _accept_ctl(controls, c, name, hello, state):
                    continue
                c.settimeout(10)
                # ⚠ 第 5 项 = **上一帧发这一路用了多少毫秒**（跳帧背压的判据 ✓ 见帧循环那段 ✓）
                clients.append([c, name, 0, time.perf_counter(), 0.0])
                print("[%s] B 机 %s 已连上（现在 %d 路）"
                      % (time.strftime("%H:%M:%S"), name, len(clients)))

            # ② **先把控制命令处理掉，再看有没有人收帧** —— B 机常常是"先叫换图、再连
            #   收流"，这一步放到 `if not clients` 后面的话，命令会一直没人理 ✗
            #   （现象看着像 A 机挂了，其实只是还没人收帧而已）。
            controls = _poll_controls(controls, state)
            if not clients:
                time.sleep(0.5)          # 没人连着：不必白抓屏
                continue
            # ⭐ **还没有区域**（启动时一块都没框 ⇒ 在等 B 机 `MAP` ✓）：不抓屏 —— 抓了也
            #   不知道抓哪儿。每 10 秒提醒一句（不刷屏 ✓）：B 机那边收到的会是**一帧都没有**，
            #   日志里这句就是"为什么没有"的答案 ✓（否则看着像推流坏了 ✗）。
            if not state["box"]:
                _now = time.time()
                if _now - _no_box_note >= 10.0:
                    _no_box_note = _now
                    print("[%s] ⚠ 还没收到图（%d 路连着但一帧都推不了）：等 B 机发 "
                          "`MAP <地图id>`，或者在部署台「小地图推流」卡片里框一次"
                          % (time.strftime("%H:%M:%S"), len(clients)))
                time.sleep(0.5)
                continue
            t_frame = time.perf_counter()
            # ⭐ 每拍**重新读** state（上一拍可能刚被 `MAP` 换掉 ⇒ 这一帧就是新区域 ✓）
            _bx, _bz = state["box"], int(state["zoom"] or 1)
            shot = screen.grabWindow(0, _bx[0], _bx[1], _bx[2], _bx[3])
            if _bz > 1:
                shot = shot.scaled(shot.width() * _bz, shot.height() * _bz)
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.WriteOnly)
            shot.save(buf, "JPEG", quality)
            buf.close()
            data = bytes(ba)
            pkt = len(data).to_bytes(4, "big") + data
            n += 1
            keep = []
            #: 一帧预算的一半（毫秒）—— 某一路**上一帧**超过它就算"跟不上"⇒ 这一帧跳过它 ✓
            _budget_ms = 500.0 / max(1, fps)
            for rec in clients:
                c, name, sent, c_t0, _last_ms = rec
                # ⭐⭐ **跳帧背压**（用户 2026-10-03 ✓ fps 提到 60 之后必须有它）：
                #   老做法是**阻塞** `sendall` （"一帧编一次、发多路"✓）⇒ 只要有一路
                #   网络/对端慢，`sendall` 就把**整个帧循环**拖住 ⇒ 所有客户端一起变慢 ✗。
                #   ⇒ 现在：这一路**上一帧**发送耗时超过半帧预算 ⇒ 判定它跟不上 ⇒
                #     **这一帧干脆不发它**（跳帧 ✓）。跳帧在这里**无损**：B 机本来就
                #     "只留最新帧"（`MiniMapClient` ✓）⇒ 跳过反而让它**马上追到最新** ✓✓。
                if _last_ms > _budget_ms:
                    rec[4] = _last_ms * 0.5              # 慢慢衰减 ⇒ 后面还有机会 ✓
                    _skip[name] = _skip.get(name, 0) + 1
                    keep.append(rec)
                    continue
                t_send = time.perf_counter()
                try:
                    c.sendall(pkt)       # 一帧编一次、发多路（编码是大头）
                except Exception as e:                  # noqa: BLE001
                    print("[%s] %s 断了（%s: %s），移除这一路（剩 %d 路）"
                          % (time.strftime("%H:%M:%S"), name, type(e).__name__, e,
                             len(clients) - 1))
                    try:
                        c.close()
                    except Exception:                   # noqa: BLE001
                        pass
                    continue
                rec[4] = (time.perf_counter() - t_send) * 1000.0
                rec[2] = sent + 1
                if rec[2] == 1:
                    # **第一帧**单独报：B 机"等帧"最需要知道的就是"推出去了没有"
                    print("[%s] 第一帧已发出 → %s（%.0f ms，%.1f KB）"
                          % (time.strftime("%H:%M:%S"), name,
                             (time.perf_counter() - t_frame) * 1000.0,
                             len(data) / 1024.0))
                elif rec[2] % max(1, fps * 10) == 0:    # 之后每 10 秒报一次
                    print("[%s] → %s 已发 %d 帧（本次）　累计 %d 帧　约 %.1f fps  "
                          "单帧 %.1f KB　跳帧 %d"
                          % (time.strftime("%H:%M:%S"), name, rec[2], n,
                             n / max(1e-6, time.time() - t0), len(data) / 1024.0,
                             _skip.get(name, 0)))
                elif rec[2] == 2 and time.perf_counter() - c_t0 > 3.0:
                    # **发出去了没有**：连上 3 秒才发出第二帧，八成卡在抓屏上
                    #（A 机在远程桌面/锁屏/最小化时 `grabWindow` 会卡住）
                    print("[%s] ⚠ %s 连上 %.0f 秒才发出第二帧 —— 抓屏卡住了？"
                          "（A 机在远程桌面/锁屏/最小化时截屏会卡）"
                          % (time.strftime("%H:%M:%S"), name,
                             time.perf_counter() - c_t0))
                keep.append(rec)
            clients = keep
            dt = 1.0 / fps - (time.perf_counter() - t_frame)
            if dt > 0:
                time.sleep(dt)
        except KeyboardInterrupt:
            print("退出（共发 %d 帧）" % n)
            return 0
        except Exception as e:
            print("[%s] 抓帧/编码出错（%s: %s），2 秒后继续"
                  % (time.strftime("%H:%M:%S"), type(e).__name__, e))
            time.sleep(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
