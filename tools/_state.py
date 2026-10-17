# -*- coding: utf-8 -*-
"""临时：看现在能不能安全标定（用完删）——① 工作台还在不在 ② relay 会不会被占 ③ 流空不空。"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))


def sh(*cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=20)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:                                     # noqa: BLE001
        return "（失败：%s）" % e


out = sh("wmic", "process", "where", "name like '%python%'",
         "get", "ProcessId,CommandLine")
rows = [l.strip() for l in out.splitlines() if l.strip() and "ProcessId" not in l]
app = [r for r in rows if "gui.app" in r]
print("① 工作台（gui.app）进程数 = %d" % len(app))
for r in app:
    print("   " + r[:100])
print("   其他 python 进程数 = %d" % (len(rows) - len(app)))

udp = sh("netstat", "-an", "-p", "udp")
print("② udp:5000 = %s" % ([l.strip() for l in udp.splitlines() if ":5000" in l]
                            or "空闲 ✓（实时预览停了 ✓）"))

import decision.agent as ag                                     # noqa: E402
ag.DecisionSettings.save = lambda self: None
from decision.agent import set_save_hook, settings              # noqa: E402
set_save_hook(None)
try:
    from gui.project import last_opened
    p = last_opened()
    settings.from_dict((p.get("decision") if p is not None else None) or {})
    print("③ 项目 = %s ／ 面板设的输入设备 = %s ／ 重连开关 = %s"
          % (getattr(p, "root", "?"), settings.input_device, settings.reconnect_enabled))
except Exception as e:                                          # noqa: BLE001
    print("③ 读项目失败：%s" % e)
print("\n结论提示：①=0（工作台关了）⇒ relay 空着 ✓ 可以标定；"
      "①>0 但 ③ 是 local ⇒ 工作台不会占 relay ✓ 也能标 ✓")
