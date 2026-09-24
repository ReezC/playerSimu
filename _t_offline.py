"""临时：验证「挂机中途掉线」的翻译 + 意外退出带运行时长。跑完即删。"""
import os
import sys
import time
import types
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, ".")

from deploy import services

RAW = "Error submitting a packet to the muxer: Error number -10051 occurred"
h = services.explain(RAW)
print("== 1. 翻译（-10051）==")
for ln in h.splitlines():
    print("   " + ln)
ok = [("两种情形都写了", "一启动就报" in h and "挂了一阵才报" in h),
      ("指到了省电排查", "powercfg" in h and "A_SETUP" in h)]

print()
print("== 2. 部署台：意外退出带上运行时长 ==")
import PyQt5.QtWidgets as W

app = W.QApplication([])
cfg_file = Path("config/deploy.json")
before = cfg_file.read_bytes() if cfg_file.exists() else None

from deploy.app import DeployWindow, _fmt_dur

ok.append(("时长格式化",
           _fmt_dur(45) == "45 秒" and _fmt_dur(125) == "2 分 5 秒"
           and _fmt_dur(7500) == "2 时 5 分"))

w = DeployWindow()
w.procs["push"] = types.SimpleNamespace(started_at=time.time() - 125,
                                        running=lambda: False)
w._stopping.discard("push")
w._on_exit("push", -10051)
txt = w.log_pane.txt.toPlainText()
line = [l for l in txt.splitlines() if "意外退出" in l][-1]
print("   %s" % line[-70:])
ok.append(("意外退出写了「已运行 2 分 5 秒」", "已运行 2 分 5 秒" in line))
ok.append(("退出时翻译也已在日志里", "挂了一阵才报" in txt))

w.procs["push"] = types.SimpleNamespace(started_at=0.0,
                                        running=lambda: False)   # 没有起始时间
w._on_exit("push", -10051)
line2 = [l for l in w.log_pane.txt.toPlainText().splitlines()
         if "意外退出" in l][-1]
ok.append(("没有起始时间就不硬写时长", "已运行" not in line2))

del w
app.quit()
if before is not None and cfg_file.read_bytes() != before:
    cfg_file.write_bytes(before)
    print("   （config/deploy.json 已还原）")

print()
for n, good in ok:
    print("   %-30s [%s]" % (n, "OK" if good else "NG"))
print()
print("共 %d 项，NG %d 项" % (len(ok), sum(1 for _n, g in ok if not g)))
