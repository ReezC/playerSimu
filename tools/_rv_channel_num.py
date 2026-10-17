"""反向验证：「频道」参数那条链逐处拆开 ⇒ `t_reconnect_channel_num` 当场红（跑完自动还原 ✓）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RC = ROOT / "decision" / "reconnect.py"

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
            print('RED  ' + str(e)[:160])
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


def rv(title, old, new):
    src = RC.read_text(encoding="utf-8")
    if old not in src:
        print("%-44s SKIP（锚点没找到 ✓）" % title)
        return
    RC.write_text(src.replace(old, new, 1), encoding="utf-8")
    drop_pyc(RC)
    try:
        out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))]
        print("%-44s %s" % (title, line[0] if line else (out.stderr or "")[-100:]))
    finally:
        RC.write_text(src, encoding="utf-8")
        drop_pyc(RC)


print("=== 反向验证：t_reconnect_channel_num ===")
rv("① 格号不折算（永远点第 1 格）",
   "            x += ((n - 1) % CHANNEL_GRID_COLS) * CHANNEL_GRID_DX\n"
   "            y += ((n - 1) // CHANNEL_GRID_COLS) * CHANNEL_GRID_DY",
   "            x += 0 * CHANNEL_GRID_DX\n"
   "            y += 0 * CHANNEL_GRID_DY")
rv("② 换行口径写错（第 5 格不换行）",
   "            y += ((n - 1) // CHANNEL_GRID_COLS) * CHANNEL_GRID_DY",
   "            y += ((n - 1) % CHANNEL_GRID_COLS) * CHANNEL_GRID_DY")
rv("③ 越界不拒绝（夹到边上点）",
   "            if not (1 <= n <= CHANNEL_GRID_MAX):",
   "            if False:")
rv("④ 服务器那步也被带偏",
   "        if ui == ui_state.UI_CHANNEL_PANEL:\n            try:\n                # ⚠ **别写 `or 1`**",
   "        if ui in (ui_state.UI_CHANNEL_PANEL, ui_state.UI_CHANNEL_LIST):\n            try:\n                # ⚠ **别写 `or 1`**")
print("=== 复核：还原后跑一遍（应 GREEN ✓）===")
out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print([l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))] or out.stderr[-120:])
