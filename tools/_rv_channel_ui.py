"""反向验证：把「频道」放在**主窗口挂机保护页**这条链逐处拆开 ⇒ 用例当场红（跑完自动还原 ✓）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PP = ROOT / "gui" / "player_panel.py"
SD = ROOT / "gui" / "settings_dialog.py"

RUN = """
import sys; sys.path.insert(0, '.')
import decision.agent as ag
ag.DecisionSettings.save = lambda self: None
import tools.selftest_decision as S
for n, f in S.CHECKS:
    if f.__name__ == 't_reconnect_channel_num':
        try:
            f(); print('GREEN')
        except AssertionError as e:
            print('RED  ' + str(e)[:150])
        break
"""


def drop_pyc(path):
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for f in cache.glob(path.stem + ".*.pyc"):
            try:
                f.unlink()
            except OSError:
                pass


def rv(title, path, old, new):
    src = path.read_text(encoding="utf-8")
    if old not in src:
        print("%-42s SKIP（锚点没找到 ✓）" % title)
        return
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    drop_pyc(path)
    try:
        out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))]
        print("%-42s %s" % (title, line[0] if line else (out.stderr or "")[-100:]))
    finally:
        path.write_text(src, encoding="utf-8")
        drop_pyc(path)


print("=== 反向验证：t_reconnect_channel_num（位置/接线那几条钉子）===")
rv("① 组名改掉（换个窗口/换个组）", PP,
   'QGroupBox("断线重连")', 'QGroupBox("别的组")')
rv("② 范围写错（1~200）", PP,
   "self.sp_rc_channel.setRange(1, 20)", "self.sp_rc_channel.setRange(1, 200)")
rv("③ 改了不写回设置", PP,
   "settings.reconnect_channel = int(self.sp_rc_channel.value())",
   "pass")
rv("④ _sync_from_settings 不同步", PP,
   "        if hasattr(self, \"sp_rc_channel\"):\n"
   "            self.sp_rc_channel.blockSignals(True)\n"
   "            self.sp_rc_channel.setValue(int(getattr(settings, \"reconnect_channel\", 1) or 1))",
   "        if hasattr(self, \"sp_rc_channel\"):\n"
   "            pass")
rv("⑤ 设置弹窗里又留一份（放错窗口）", SD,
   "# ⚠ 「频道」（想进第几格）**不在这儿** ✗",
   "self.sp_rc_channel = None   # ⚠ 「频道」（想进第几格）**不在这儿** ✗")
print("=== 复核：还原后跑一遍（应 GREEN ✓）===")
out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print([l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))] or out.stderr[-120:])
