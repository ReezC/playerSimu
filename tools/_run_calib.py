# -*- coding: utf-8 -*-
"""临时：先自查"能不能安全标定"，再跑标定（用完删）。

自查两件（都会让标定白跑，甚至把中继搞卡 ✗）：
  ① 实时预览占着 udp:5000 ⇒ 标定开不了流；
  ② 工作台在跑 + 面板设的是 ProMicro ⇒ 它占着 relay 那**唯一**一个客户端位 ⇒
     标定的连接会握手超时（还可能把中继卡死 ✓ 见 SKILL 296）。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))


def sh(*cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=30)
    return (p.stdout or "") + (p.stderr or "")


def main():
    # ⚠ 别用 `wmic`（这台机器上没有它 ✗ 会静默返回空 ⇒ 误判"工作台没在跑" ✗）
    out = sh("powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
             "ForEach-Object { $_.CommandLine }")
    rows = [l.strip() for l in out.splitlines() if l.strip()]
    app = [r for r in rows if "gui.app" in r]
    udp = [l.strip() for l in sh("netstat", "-an", "-p", "udp").splitlines()
           if ":5000" in l]

    import decision.agent as ag
    ag.DecisionSettings.save = lambda self: None
    from decision.agent import set_save_hook, settings
    set_save_hook(None)
    dev = "?"
    try:
        from gui.project import last_opened
        p = last_opened()
        settings.from_dict((p.get("decision") if p is not None else None) or {})
        dev = settings.input_device
    except Exception as e:                                     # noqa: BLE001
        print("（读项目失败：%s）" % e)

    print("工作台进程数 = %d ／ 面板输入设备 = %s ／ udp:5000 = %s"
          % (len(app), dev, udp or "空闲 ✓"))
    if udp:
        print("⛔ 实时预览还占着流 ⇒ 先停掉它再跑 ✗")
        return 2
    if app and str(dev).lower() in ("remote", "serial"):
        print("⛔ 工作台在跑且面板是 ProMicro ⇒ 它占着 relay 那唯一一个位 ⇒ 先关掉工作台"
              "（或把「输入设备」切到本地(仅测试)）✓ 免得把中继搞卡 ✗")
        return 2
    print("⇒ 可以跑 ✓ 开始标定（会真的动游戏机光标、不点任何键 ✓）…\n")
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "tools.mouse_aim_calib",
                        "--rounds", "8"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    print((p.stdout or "") + (p.stderr or ""))
    print("退出码 = %d" % p.returncode)
    cfg = ROOT / "config" / "mouse_gain.json"
    if cfg.exists():
        print("\n===== config/mouse_gain.json =====")
        print(cfg.read_text(encoding="utf-8"))
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
