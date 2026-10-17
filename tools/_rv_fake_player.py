"""反向验证：「假玩家框 ⇒ 提前收工」那条修法逐处拆开 ⇒ 用例当场红（跑完自动还原 ✓）。"""
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
    if f.__name__ == 't_reconnect_fake_player_box':
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


print("=== 反向验证：t_reconnect_fake_player_box ===")
rv("① 那道闸删掉（回到「提前收工」）",
   "            if self.ui not in FAKE_PLAYER_UIS:", "            if True:")
rv("② 口径扩成「所有断线系」（副作用更重）",
   "FAKE_PLAYER_UIS = (ui_state.UI_CHAR_SELECT,)",
   "FAKE_PLAYER_UIS = tuple(RECONNECT_UIS)")
rv("③ 连「判不出界面」也堵上（永远走不完）",
   "            if self.ui not in FAKE_PLAYER_UIS:", "            if False:")
rv("④ 那句状态文字删掉（复盘看不出为什么没收工）",
   '                self._say("画面里有像玩家框的东西，但还停在%s —— 继续走重连 ✓"',
   '                _unused = "x" + (')
print("=== 复核：还原后跑一遍（应 GREEN ✓）===")
out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print([l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))] or out.stderr[-120:])
