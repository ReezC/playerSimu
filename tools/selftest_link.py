"""指令通道自检：从控制机这侧确认「指令能不能真的到 Pro Micro」。

用法（在 B 机 / 控制机的仓库根）：
    python -m tools.selftest_link             # 读 config/link.yaml 的 kbd.host/port
    python -m tools.selftest_link --host 192.168.1.8 --port 9000

**它发什么**：只发一条 `MOVE 0 0`（鼠标位移 0，等于没动）+ 一条 `RELEASEALL`
（松开所有键，无副作用）。**不发任何会动角色 / 会点鼠标的指令**，所以游戏里
不会有任何可见变化 —— 你随时可以放心跑。

**它判什么**：固件对每条指令都会回 DONE。所以判据是「发出去之后，短时间内
收到回执」—— 这比「TCP 发成功」可靠得多：relay 卡在串口写、或者 B 机对着一条
已经死掉的连接发，两种情况发送都不报错，但永远等不到回执。

链路健康度用的是 `decision.input.link_health()`（界面那行状态字用的同一套）。
跑之前最好**先停自动**：它会占用 relay 那唯一的客户端位几秒钟。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision import input as dinput      # noqa: E402
from tools.config import get              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--cert", default=None)
    ap.add_argument("--timeout", type=float, default=3.0,
                    help="等固件回执的秒数")
    args = ap.parse_args()

    host = args.host or get("kbd", "host")
    port = args.port or int(get("kbd", "port", 9000))
    cert = args.cert or get("kbd", "cert", "remote_kbd/certs/cert.pem")
    print("目标：%s:%d   证书：%s" % (host, port, cert))

    # ---- 1. 建连接 ----
    t0 = time.perf_counter()
    try:
        dinput.use_network(host, port, cert)
    except Exception as e:
        print("[X] 连不上：%s: %s" % (type(e).__name__, e))
        print("    按顺序查：")
        print("      1. A 机上「键盘中继」在跑吗（部署台那张卡片）")
        print("      2. 端口 / 证书对不对（部署台「环境自检」会对着 link.yaml 比）")
        print("      3. B 机能 ping 到 A 机吗")
        return 2
    print("[1/3] 已连上（%.2f 秒）" % (time.perf_counter() - t0))

    # ---- 2. 发一条无副作用指令 ----
    dinput.release_all_remote()          # 先松键，避免上次留下的卡键
    time.sleep(0.2)
    dinput.mouse_move(0, 0)              # 位移 0：等于没动，但固件会回 DONE
    h0 = dinput.link_health()
    print("[2/3] 已发出探测指令（后端=%s）" % h0.get("backend"))

    # ---- 3. 等回执 ----
    deadline = time.time() + args.timeout
    h = None
    while time.time() < deadline:
        time.sleep(0.1)
        h = dinput.link_health()
        if not h.get("silent"):          # silent 归零 = 最后一条有回执
            break
    silent = float((h or {}).get("silent") or 0.0)
    print("[3/3] %s" % ("收到固件回执" if not silent
                        else "等了 %.1f 秒没有任何回执" % silent))

    ok = bool(h and h.get("ok")) and not silent
    print()
    if ok:
        print("=== 通 ===")
        print("指令能到 Pro Micro。接下来手测：在自动关着的情况下按几个映射键、")
        print("划一下触控板 —— 游戏里应当有反应。")
        print("A 机上可以对照 %s 里的 tcp->serial 行，确认每条指令都到了串口。"
              % "remote_kbd/relay_trace.log")
    else:
        print("=== 不通 ===")
        print("原因（界面那行状态字显示的是同一个）：%s" % (h.get("err") or "链路无回执"))
        print("按顺序查：")
        print("  1. A 机 relay 的控制台：有「意外退出」吗？（崩了 → 看 remote_kbd/relay_crash.log）")
        print("  2. A 机 relay_trace.log：最后几行在做什么（有没有 serial write/read error）")
        print("  3. A 机 relay_trace.log 里有没有 tcp->serial 行 —— 没有说明指令根本没到 relay")
        print("  4. 板子还在吗：A 机部署台「环境自检」看串口列不列得出来")
    dinput.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
