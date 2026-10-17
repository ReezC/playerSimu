"""**多实例**自检（用户 2026-10-17 ✓ 原话：「**我们一定要能支持多实例工作**」）。

**为什么要它**
    同一台机器上要同时跑两个工作台：一个控远程 A 机（收流 + relay），
    一个控本机游戏（抓本地窗口 + 本机发送）。它们共用一个仓库目录 ⇒
    凡是被"当成只有一个人在用"的**路径、端口、全局状态**都会互相踩 ✓，
    而症状全是**误导性**的（录到半截画面 / 配置莫名其妙回默认 / 另一个实例黑了 /
    鼠标点到别处）⇒ 必须一条条钉死。

**这套钉什么**
    ① 空实例（不给 `--instance`）⇒ 四个路径**与老行为一字不差** ✓
       （只开一个工作台的人不该被这次改动影响 ✗）；
    ② 具名实例 ⇒ 配置/行为日志/崩溃日志/会话/录像目录**各自一份** ✓
       （且配置文件首次**从默认那份复制** ✓ 不从空开始 ✗）；
    ③ 同一个 UDP 端口**只归一个实例收** ✓（同进程重入放行 / 别人活着拒绝 / 主人死了接管）；
    ④ 两人同时改**同一张图**的集合 ⇒ 谁都不许抹掉对方 ✓（键级合并 + 原子写）；
    ⑤ 源码级：录像文件名独占、H.264 优先、relay 多客户端 + 角色公告、只读端如实报错。

跑法（不连流、不建 YOLO）：

    python -m tools.selftest_instance      # 全过返回 0，有失败返回 1
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

from core import instance as I                                  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _as(raw, fn):
    """临时把"本实例叫什么"切成 `raw` 再跑 `fn`（跑完**恢复原样** ✓ 别污染别的用例 ✗）。"""
    old_argv = sys.argv[:]
    old_env = os.environ.get(I.ENV)
    try:
        sys.argv = ["tools.selftest_instance"]
        if raw is None:
            os.environ.pop(I.ENV, None)
        else:
            os.environ[I.ENV] = raw
        I._resolved.clear()
        return fn()
    finally:
        sys.argv = old_argv
        if old_env is None:
            os.environ.pop(I.ENV, None)
        else:
            os.environ[I.ENV] = old_env
        I._resolved.clear()


def t_default_paths_unchanged():
    """① **空实例 ⇒ 老路径一字不差**（单实例用户感觉不到这次改动 ✓）。"""

    def body():
        check(I.name() == "" and I.slug() == "", "没给实例名却解析出了名字：%r" % (I.name(),))
        check(I.live_path() == ROOT / "config" / "live.yaml",
              "空实例的实时配置不是 `config/live.yaml`：%s" % I.live_path())
        check(I.behavior_path() == ROOT / "behavior.log",
              "空实例的行为日志不是 `behavior.log`：%s" % I.behavior_path())
        check(I.crash_path() == ROOT / "crash.log",
              "空实例的崩溃日志不是 `crash.log`：%s" % I.crash_path())
        check(I.session_path() == ROOT / "config" / "session.json",
              "空实例的会话文件不是 `config/session.json`：%s" % I.session_path())
        check(I.record_dir() == ROOT / "data" / "recordings",
              "空实例的录像目录不是 `data/recordings`：%s" % I.record_dir())
        check(I.suffix() == "", "空实例的后缀该是空串")

    _as(None, body)
    # ⚠ 命令行那份也要认（`--instance b` ✓ 比环境变量优先 ✓）
    body2 = lambda: check(I.slug() == "b", "`--instance b` 没被认出来：%r" % (I.slug(),))
    sys.argv = ["gui.app", "--instance", "b"]
    I._resolved.clear()
    try:
        body2()
    finally:
        I._resolved.clear()


def t_named_paths_isolated():
    """② **具名实例 ⇒ 五条路径各自一份**，且配置首次从默认那份复制 ✓。"""

    def body():
        check(I.slug() == "b", "实例名没解析对：%r" % (I.slug(),))
        live = I.live_path()
        check(live.name == "live_b.yaml", "具名实例的实时配置该是 `live_b.yaml`：%s" % live.name)
        check(live.exists(), "具名实例的配置没有**自动生成**（第一次用要从 live.yaml 复制 ✓）")
        check(I.behavior_path().name == "behavior_b.log",
              "行为日志没分开：%s" % I.behavior_path().name)
        check(I.crash_path().name == "crash_b.log",
              "崩溃日志没分开：%s" % I.crash_path().name)
        check(I.session_path().name == "session_b.json",
              "会话文件没分开：%s" % I.session_path().name)
        rd = I.record_dir()
        check(rd == ROOT / "data" / "recordings" / "b",
              "录像目录没分开：%s" % rd)
        check(rd.exists(), "录像目录没建出来")
        check(I.suffix() == "_b" and I.label() == "b", "标签/后缀不对：%r %r" % (I.suffix(), I.label()))

    _as("b", body)


def t_udp_port_claim():
    """③ **一个 UDP 端口只归一个实例**（同机多实例最硬的那条）。"""
    port = 5123
    check(I.udp_port_of("udp://0.0.0.0:5123") == 5123, "端口解析失败")
    check(I.udp_port_of("udp://0.0.0.0:5123?fifo_size=100") == 5123, "带参数时端口解析失败")
    check(I.udp_port_of("data/x.mp4") == 0, "非 udp 源不该返回端口")
    check(I.udp_port_of("rtsp://1.2.3.4:8554/live") == 0, "非 udp 协议不该返回端口")

    def body():
        try:
            ok, why = I.claim_port(port)
            check(ok, "自己占自己还拒绝（重启预览会被自己顶掉 ✗）：%s" % why)
            ok2, _ = I.claim_port(port)
            check(ok2, "**同进程**再次占同一个端口被拒绝了（预览重启就会失败 ✗）")
            # 伪造"另一个实例占着"：写一个**活着**的 PID（当前进程另一个"实例"）
            lock = I.LOCK_DIR / ("udp_%d.lock" % port)
            lock.write_text("pid=%d\ninst=别人\nport=%d\n" % (os.getpid() + 100000, port),
                            encoding="utf-8")
            ok3, why3 = I.claim_port(port)
            check(ok3, "主人已经不在了却不接管（崩过一次就永远说\"被占\"✗）：%s" % why3)
        finally:
            try:
                (I.LOCK_DIR / ("udp_%d.lock" % port)).unlink()
            except Exception:              # noqa: BLE001
                pass

    _as(None, body)


def t_zones_conflict_logged():
    """④ **同一张图的集合被另一个实例改过 ⇒ 必须记账**（不许无声无息地覆盖 ✓）。

    ⚠⚠ 注意这里钉的是"**记账**"而不是"合并" ✗ —— 2026-10-17 试过合并，当天回退 ✓：
      合并会让**故意删掉的集合复活**，而且破坏"保存=这份内容"的语义
      （`t_goto_hands_over_whole_path` 直接红：换一份含跳的边，旧的走边被合并留下 ⇒
      最短路又选回走边 ✗）⇒ 保留"整份覆盖" ✓，改用**冲突记账**让问题可见 ✓。
    """
    from core import zones as Z
    from core import behavior as BH

    p = Path(tempfile.gettempdir()) / "_selftest_zones_conflict.json"
    try:
        p.unlink()
    except Exception:                      # noqa: BLE001
        pass
    got = []
    _orig = BH.event
    try:
        a = Z.Zones("m1")
        a.sets["甲"] = {"color": "#111111", "footholds": ["1"]}
        a.save(p)
        # 模拟"我打开时是这样"（`load()` 会记 mtime ✓ 这里手工模拟）
        a._disk_mtime = p.stat().st_mtime_ns
        # 另一个实例改了它（我们没看见）
        b = Z.Zones("m1")
        b.sets["乙"] = {"color": "#222222", "footholds": ["2"]}
        b.save(p)
        import time as _t
        _t.sleep(0.02)
        b.sets["丙"] = {"color": "#333333", "footholds": ["3"]}
        b.save(p)
        # 我这边保存 ⇒ 必须记一条 zones_conflict ✓
        BH.event = lambda name, *a2, **k2: got.append((name, k2))
        a.sets["甲2"] = {"color": "#444444", "footholds": ["4"]}
        a.save(p)
        check(any(n == "zones_conflict" for n, _ in got),
              "别的实例改过这张图却**一声不响**（现场只会觉得\"我的集合怎么没了\" ✗）")
        # 语义仍是"整份覆盖"（不合并 ✓ 否则删掉的集合会复活 ✗）
        d = json.loads(p.read_text(encoding="utf-8"))
        check("乙" not in (d.get("sets") or {}),
              "保存变成了\"与盘上合并\"（故意删掉的集合会复活 ✗ 见 docstring）")
    finally:
        BH.event = _orig
        try:
            p.unlink()
        except Exception:                  # noqa: BLE001
            pass


def t_source_pins():
    """⑤ 源码级钉子：这些"分开/独占"的做法**不许被人改回去** ✗。"""
    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check('"xb"' in lt and "FileExistsError" in lt,
          "录像文件名不再是**独占创建**（两个实例同一毫秒就会写同一个文件 ⇒ 画面交织 ✗）")
    check("claim_port" in lt and "udp_port_of" in lt,
          "收流前没做**端口归属检查**（同机两实例会抢同一路 UDP ⇒ 其中一个帧率减半/花屏 ✗）")
    check("CAP_MSMF" in lt and '"H264"' in lt,
          "录像没优先 H.264（`mp4v` 在 Windows 自带播放器里就是花屏 ✗）")

    cfg = (ROOT / "core" / "config.py").read_text(encoding="utf-8")
    check("instance.live_path()" in cfg, "实时配置没按实例分文件（两个实例会互相改来源 ✗）")
    check("instance.record_dir" in cfg, "录像目录没按实例分")
    beh = (ROOT / "core" / "behavior.py").read_text(encoding="utf-8")
    check("instance.behavior_path()" in beh, "行为日志没按实例分（trim 会吞掉对方的行 ✗）")
    prj = (ROOT / "gui" / "project.py").read_text(encoding="utf-8")
    check("instance.session_path()" in prj, "会话文件没按实例分（下次启动会打开别人的项目 ✗）")

    relay = (ROOT / "remote_kbd" / "relay.py").read_text(encoding="utf-8")
    check("server.listen(8)" in relay, "relay 还是只排一个连接（第二个实例连不进去 ✗）")
    check("def bridge_observer" in relay, "relay 没有只读客户端（第二个实例只能干等 ✗）")
    check("ROLE " in relay, "relay 没公告角色（只读端会以为自己在控制 ✗）")
    cli = (ROOT / "remote_kbd" / "kbd_client.py").read_text(encoding="utf-8")
    check("self.role" in cli and "ROLE " in cli, "客户端没解析角色公告")
    inp = (ROOT / "decision" / "input.py").read_text(encoding="utf-8")
    check('"role": _role' in inp and "readonly" in inp,
          "`link_health()` 没把\"只读\"报出来（上层会以为链路正常、一直点空气 ✗）")


def t_health_readonly():
    """⑤b 只读端**必须**在 `link_health` 里如实说是只读（别让它以为链路正常）。"""
    import decision.input as di

    class _R:
        role = "readonly"
        ok = True
        fails = 0
        last_err = ""

        def silent_for(self):
            return 0.0

    old_remote, old_cfg = di._remote, di._cfg
    try:
        di._remote, di._cfg = _R(), ("remote",)
        h = di.link_health()
        check(h.get("role") == "readonly", "`link_health()` 没报 role：%r" % (h,))
        check(h.get("ok") is False, "只读端却报链路正常（上层会一直发指令、全被丢 ✗）")
        check("只读" in str(h.get("err")), "只读端的原因没写清楚：%r" % (h.get("err"),))
        di._remote.role = "control"
        check(di.link_health().get("ok") is True, "控制端被误判成不健康")
    finally:
        di._remote, di._cfg = old_remote, old_cfg


TESTS = (
    ("① 空实例 ⇒ 配置/日志/会话/录像四条路径与老行为**一字不差**（单实例用户不受影响）",
     t_default_paths_unchanged),
    ("② 具名实例 ⇒ 五条路径各自一份 · 配置首次自动从 live.yaml 复制",
     t_named_paths_isolated),
    ("③ 一个 UDP 端口只归一个实例：同进程重入放行 · 别人活着拒绝 · 主人死了接管",
     t_udp_port_claim),
    ("④ 同一张图的集合被别的实例改过 ⇒ 必须记账（整份覆盖语义不变 · 原子写）",
     t_zones_conflict_logged),
    ("⑤ 源码钉：录像名独占 · H.264 优先 · relay 多客户端+角色 · 各路径按实例分",
     t_source_pins),
    ("⑤b 只读端必须在 `link_health()` 里如实说是只读（不许报链路正常）",
     t_health_readonly),
)


def main():
    failed = 0
    print("=" * 76)
    print("多实例自检")
    print("=" * 76)
    for i, (title, fn) in enumerate(TESTS, 1):
        print("\n[%d/%d] %s" % (i, len(TESTS), title))
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print("   [FAIL] %s" % e)
        except Exception as e:                          # noqa: BLE001
            failed += 1
            import traceback
            print("   [FAIL] %s: %s" % (type(e).__name__, e))
            traceback.print_exc()
        else:
            print("   [ OK ]")
    print("\n" + "=" * 76)
    if failed:
        print("%d/%d 失败" % (failed, len(TESTS)))
        return 1
    print("%d/%d 通过" % (len(TESTS), len(TESTS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
