# -*- coding: utf-8 -*-
"""临时：分片跑 tools/selftest_decision（每片一个进程，绕开"先 Qt 后 torch"那条原生崩）。

用完可删（落在 data/，本来 gitignore）。
加 `--verbose`：每条用例往 stderr 打一行（原生崩时能看出崩在哪条 ✓）。
"""
import contextlib
import io
import sys

sys.path.insert(0, ".")
import decision.agent as ag                                   # noqa: E402

ag.DecisionSettings.save = lambda self: None
import tools.selftest_decision as S                            # noqa: E402

_argv = [a for a in sys.argv[1:] if a != "--verbose"]
verbose = "--verbose" in sys.argv
a, b = int(_argv[0]), int(_argv[1])
bad = []
n = 0
for name, fn in S.CHECKS[a:b]:
    n += 1
    if verbose:
        sys.stderr.write("[%d] %s\n" % (a + n - 1, name[:60]))
        sys.stderr.flush()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except Exception as e:                                     # noqa: BLE001
        bad.append((name, "%s: %s" % (type(e).__name__, str(e)[:90])))
open("data/_slice_%d_%d.txt" % (a, b), "w", encoding="utf-8").write(
    "n=%d bad=%d\n" % (n, len(bad))
    + "\n".join("  [NG] %s | %s" % (x[:50], y) for x, y in bad))
