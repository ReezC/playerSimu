"""反向验证：「第 21~60 个频道 ⇒ 先滚再点」逐处拆开 ⇒ 用例当场红（跑完自动还原 ✓）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RC = ROOT / "decision" / "reconnect.py"
MA = ROOT / "execution" / "mouse_aim.py"

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
            print('RED  ' + str(e)[:140])
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
        print("%-44s SKIP（锚点没找到 ✓）" % title)
        return
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    drop_pyc(path)
    try:
        out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))]
        print("%-44s %s" % (title, line[0] if line else (out.stderr or "")[-100:]))
    finally:
        path.write_text(src, encoding="utf-8")
        drop_pyc(path)


print("=== 反向验证：1~60（先滚再点）===")
rv("① 需要滚动时不滚（直接点第一屏）", RC,
   "            if row >= CHANNEL_GRID_ROWS:",
   "            if False:")
rv("② 到底不折算夹取（末尾几个点错行）", RC,
   "                on_screen = row - min(row, CHANNEL_MAX_SCROLL_ROWS)",
   "                on_screen = row")
rv("③ 只认一屏（总频道数改回 20）", RC,
   "CHANNEL_TOTAL = 60 ", "CHANNEL_TOTAL = 20 ")
rv("④ 滚动动作没人执行（live_thread 不认 act=scroll）", RC,
   "                return {\"act\": \"scroll\", \"up\": scroll[0], \"down\": scroll[1], \"desc\": desc}",
   "                return {\"act\": \"tap\", \"key\": \"enter\", \"desc\": desc}")
print("=== 复核：还原后跑一遍（应 GREEN ✓）===")
out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print([l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))] or out.stderr[-120:])
