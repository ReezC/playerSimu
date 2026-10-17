# -*- coding: utf-8 -*-
"""临时反向验证：「重连回到游戏 ⇒ 体检才恢复自动」（2026-10-07）。

逐处把新链拆开，看目标用例是不是**当场红**（每例跑完立刻还原 ✓ 用完删 ✓）。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PP = ROOT / "gui" / "player_panel.py"
RC = ROOT / "decision" / "reconnect.py"

_CODE = """
import sys
sys.path.insert(0, r"{root}")
import decision.agent as ag
ag.DecisionSettings.save = lambda self: None
{imp} as f
try:
    f(); print("PASS")
except AssertionError as e:
    print("AssertionError:", e)
"""


def run(imp):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", _CODE.format(root=ROOT, imp=imp)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT), env=env)
    out = (p.stdout or "") + (p.stderr or "")
    if "PASS" in out:
        return "PASS  ← 反向验证失败 ✗（拆了还绿）"
    for ln in [x.strip() for x in reversed(out.splitlines()) if x.strip()]:
        if ln.startswith("AssertionError:"):
            return "RED   " + ln.split(":", 1)[1].strip()[:100]
    tail = [x for x in out.strip().splitlines() if x.strip()]
    return "RED   " + (tail[-1][:100] if tail else "（没输出 / 原生崩 ✗）")


def _drop_pyc(path):
    """删掉对应 pyc —— 本仓库踩过"改完源码却拿**旧字节码**跑 ⇒ 结果全是假的"✗（见开发日志 244 ✓）。"""
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for f in cache.glob(path.stem + ".*.pyc"):
            try:
                f.unlink()
            except OSError:
                pass


def rv(title, path, old, new, imp):
    src = path.read_text(encoding="utf-8")
    if old not in src:
        print("%-52s SKIP  锚点没找到（源码变了？）" % title)
        return
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    _drop_pyc(path)
    try:
        print("%-52s %s" % (title, run(imp)))
    finally:
        path.write_text(src, encoding="utf-8")
        _drop_pyc(path)


DEC = "from tools.selftest_decision import t_reconnect_stop_auto"
LIV = "from tools.selftest_live_panel import t_reconnect_resume_gate"

rv("① 状态机改回「直接开自动」（不置旗子）", RC,
   "self.s.reconnect_resume_pending = True", "self.s.enabled = True", DEC)
rv("② 消费点不接进 _poll_auto_state", PP,
   "        self._poll_reconnect_resume()", "        pass  # RV", LIV)
rv("③ 体检不过也放行（判据恒真）", PP,
   "        if probs:\n            return False, \"；\".join",
   "        if False:\n            return False, \"；\".join", LIV)
rv("④ 不尊重「人已经自己开着」", PP,
   "        if bool(settings.enabled):\n            return                      # 人自己已经开了",
   "        if bool(settings.enabled):\n            pass                        # 人自己已经开了", LIV)
rv("⑤ 前台自己手写一份体检参数（不共用）", PP,
   "        if self._auto_confirming:\n            return True                         # 已经在问别的了，别叠对话框 ✗\n"
   "        probs = self._auto_precheck_problems()",
   "        if self._auto_confirming:\n            return True                         # 已经在问别的了，别叠对话框 ✗\n"
   "        probs = self._precheck_problems(self._map_id_for_check())", LIV)
rv("⑥ 体检不过不写状态栏", PP,
   "            settings.reconnect_note = (\n", "            settings.reconnect_note_off = (\n",
   LIV)
