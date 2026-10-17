"""反向验证：「被打断回来连目的地一起重算」逐处拆开 ⇒ 用例当场红（跑完自动还原 ✓）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AG = ROOT / "decision" / "agent.py"
RT = ROOT / "decision" / "route.py"

RUN = """
import sys; sys.path.insert(0, '.')
import decision.agent as ag
ag.DecisionSettings.save = lambda self: None
import tools.selftest_decision as S
for n, f in S.CHECKS:
    if getattr(f, '__name__', '') == 't_replan_on_resume_after_attack':
        try:
            f(); print('GREEN')
        except AssertionError as e:
            print('RED  ' + str(e)[:130])
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
        print("%-46s SKIP（锚点没找到 ✓）" % title)
        return
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    drop_pyc(path)
    try:
        out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))]
        print("%-46s %s" % (title, line[0] if line else (out.stderr or "")[-90:]))
    finally:
        path.write_text(src, encoding="utf-8")
        drop_pyc(path)


print("=== 反向验证：t_replan_on_resume_after_attack ===")
rv("① 强制那段砍掉（回到「接着上次跑」）", AG,
   "        if (_paused_at is not None\n"
   "                and (float(now) - float(_paused_at))\n"
   "                >= float(getattr(job, \"PAUSE_MIN_S\", 0.25) or 0.25)):",
   "        if False:")
rv("② 阈值大得永远够不着", AG,
   '                >= float(getattr(job, "PAUSE_MIN_S", 0.25) or 0.25)):',
   '                >= 9999.0):')
rv("③ 短打断也算（阈值降到 0）", AG,
   '                >= float(getattr(job, "PAUSE_MIN_S", 0.25) or 0.25)):',
   '                >= 0.0):')
rv("④ 类属性与模块常量脱钩（两处分叉）", RT,
   "    PAUSE_MIN_S = PAUSE_MIN_S", "    PAUSE_MIN_S = 0.5")
print("=== 复核：还原后跑一遍（应 GREEN ✓）===")
out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print([l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))] or out.stderr[-120:])
