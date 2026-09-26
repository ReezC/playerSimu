"""【A 机运行】把游戏里的小地图**单独截屏**、推给 B 机（寻路定位用）。

**平时不用敲命令**：A 机上的一切都在「被控机部署台」的「小地图推流」卡片里
（区域框选、端口、帧率、启停、日志）。部署台用**同一个解释器**把本模块按那些
参数拉起来，卡片底部还能看到它实际执行的那条命令。下面这些是等价的手工做法，
部署台本身起不来时才用：

    python -m tools.minimap_push --pick            # 先在屏幕上框一次小地图区域
    python -m tools.minimap_push --x 100 --y 40 --w 200 --h 150
    python -m tools.minimap_push --zoom 4 --fps 10

**为什么要单独推一路**（而不是从主画面里抠）：
  · 主画面是 H.264 压过的，小地图上的黄点只有 2~5 像素，压完就糊了；
  · 主画面里小地图面板可能被 UI 挡、被玩家挪走、被缩放；
  · 这一路是**原始像素**（JPEG 质量 100），B 机拿到的黄点位置几乎无损。

**协议**（TCP，长度帧）：`[4 字节大端长度][JPEG 数据]` 循环。
为什么不是 UDP：一帧 JPEG 几十 KB，UDP 会分片、丢一片就整帧废；而"只留最新帧"
的语义本来就在**接收端**做（B 机 `perception/minimap.py`），用 TCP 反而简单可靠。

区域配置存在 `config/minimap_region.json`（`--pick` 会写进去），下次直接跑即可。
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
REGION_FILE = ROOT / "config" / "minimap_region.json"

DEFAULTS = {"bind": "0.0.0.0", "port": 5003, "zoom": 3, "fps": 10,
            "quality": 100, "region": None}


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
    `+/-` 倍数，见 docs/UI规范.md §9）：小地图面板的边框就差一两个像素，没有
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

    if args.x is not None:
        # 部署台走这条：区域在它的卡片里框、由命令行传进来。
        # 顺手写一份到 minimap_region.json，命令行单跑时不用再框一次。
        cfg["region"] = [args.x, args.y, args.w, args.h]
        save_cfg(cfg)

    region = cfg.get("region")
    if not region:
        print("还没有小地图区域。A 机上：部署台「小地图推流」卡片里点「框选…」；"
              "命令行：python -m tools.minimap_push --pick")
        return 2

    x, y, w, h = (int(v) for v in region)
    zoom = max(1, int(cfg["zoom"]))
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
    print("监听 %s:%d   区域 (%d,%d) %dx%d   zoom=%d   %d fps   JPEG q=%d"
          % (cfg.get("bind") or "0.0.0.0", int(cfg["port"]),
             x, y, w, h, zoom, fps, quality))
    print("等 B 机连进来…（B 机跑：python -m perception.minimap --map <地图id>）")

    # **支持多个 B 机客户端**（2026-09-26 修）：原来一次只服务一个连接 —— B 机上
    # 只要多一条客户端（工作台面板的定位 + 「标定…」弹窗 + 忘了关的命令行探针都是
    # 各连一条），**后连的那条就只能排队、永远收不到帧**：现象是"TCP 连得上、
    # 一帧都不来"（B 机报「等帧超时」），而 A 机日志里一片正常 —— 极难查。
    # 现在每帧**广播**给所有连着的客户端，谁都不会被饿死。
    clients = []             # [[sock, 对端名, 本次已发帧数, 连上的时刻], ...]
    n = 0
    t0 = time.time()
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
                c.settimeout(10)
                name = "%s:%d" % (peer[0], peer[1])
                clients.append([c, name, 0, time.perf_counter()])
                print("[%s] B 机 %s 已连上（现在 %d 路）"
                      % (time.strftime("%H:%M:%S"), name, len(clients)))
            if not clients:
                time.sleep(0.5)          # 没人连着：不必白抓屏
                continue
            t_frame = time.perf_counter()
            shot = screen.grabWindow(0, x, y, w, h)
            if zoom > 1:
                shot = shot.scaled(shot.width() * zoom, shot.height() * zoom)
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.WriteOnly)
            shot.save(buf, "JPEG", quality)
            buf.close()
            data = bytes(ba)
            pkt = len(data).to_bytes(4, "big") + data
            n += 1
            keep = []
            for rec in clients:
                c, name, sent, c_t0 = rec
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
                rec[2] = sent + 1
                if rec[2] == 1:
                    # **第一帧**单独报：B 机"等帧"最需要知道的就是"推出去了没有"
                    print("[%s] 第一帧已发出 → %s（%.0f ms，%.1f KB）"
                          % (time.strftime("%H:%M:%S"), name,
                             (time.perf_counter() - t_frame) * 1000.0,
                             len(data) / 1024.0))
                elif rec[2] % max(1, fps * 10) == 0:    # 之后每 10 秒报一次
                    print("[%s] → %s 已发 %d 帧（本次）　累计 %d 帧　约 %.1f fps  "
                          "单帧 %.1f KB"
                          % (time.strftime("%H:%M:%S"), name, rec[2], n,
                             n / max(1e-6, time.time() - t0), len(data) / 1024.0))
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
