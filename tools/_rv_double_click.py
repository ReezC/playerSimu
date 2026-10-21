"""反向验证：把"双击"这条链逐处拆开，确认 `t_reconnect_double_click` 当场红（跑完自动还原 ✓）。

本仓库的规矩（见开发日志 244 ✓）：改完源码要删对应 pyc，防"拿旧字节码跑"的假结果 ✓。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RC = ROOT / "decision" / "reconnect.py"
MA = ROOT / "execution" / "mouse_aim.py"
LT = ROOT / "gui" / "live_thread.py"

RUN = """
import sys; sys.path.insert(0, '.')
import decision.agent as ag
ag.DecisionSettings.save = lambda self: None
import tools.selftest_decision as S
for n, f in S.CHECKS:
    if f.__name__ == 't_reconnect_double_click':
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
        print("%-46s SKIP（锚点没找到 ✓ 源码变了？）" % title)
        return
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    drop_pyc(path)
    try:
        out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN],
                             cwd=str(ROOT), capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        line = [l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))]
        print("%-46s %s" % (title, line[0] if line else "（没输出）" + (out.stderr or "")[-90:]))
    finally:
        path.write_text(src, encoding="utf-8")
        drop_pyc(path)


print("=== 反向验证：t_reconnect_double_click ===")
rv("① 点击时不记时刻（没人补第二下）", RC,
   "            if self.ui in DOUBLE_CLICK_UIS:\n                self._click_at = now",
   "            if False:\n                self._click_at = now")
rv("② 到点不补第二下（砍掉 click2 分支）", RC,
   "        if (ui in DOUBLE_CLICK_UIS and self._click_at",
   "        if (False and self._click_at")
rv("③ 每拍都补（不置 _doubled）", RC,
   "            self._doubled = True\n            self._declined = \"\"\n            self._say(\"%s：补第二下",
   "            self._doubled = False\n            self._declined = \"\"\n            self._say(\"%s：补第二下")
rv("④ 补第二下也吃重试（一次双击=两次）", RC,
   "            self._doubled = True\n            self._declined = \"\"\n            self._say(\"%s：补第二下",
   "            self._doubled = True\n            self._tries += 1\n            self._declined = \"\"\n            self._say(\"%s：补第二下")
rv("⑤ 服务器行也双击（名单里加上列表）", RC,
   "DOUBLE_CLICK_UIS = (ui_state.UI_CHANNEL_PANEL,)",
   "DOUBLE_CLICK_UIS = (ui_state.UI_CHANNEL_PANEL, ui_state.UI_CHANNEL_LIST)")
rv("⑥ mouse_aim 里睡一下再点（实时线程纪律）", MA,
   "def repeat_click():",
   "def repeat_click():\n    import time as _t; _t.sleep(0.1)")
rv("⑦ live_thread 不处理 click2（没接上）", LT,
   'elif _act and _act.get("act") == "click2":',
   'elif False:')
print("=== 复核：还原后直接跑一遍（应当 GREEN ✓）===")
out = subprocess.run([sys.executable, "-X", "utf8", "-c", RUN], cwd=str(ROOT),
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print([l for l in (out.stdout or "").splitlines() if l.startswith(("RED", "GREEN"))] or out.stderr[-120:])
