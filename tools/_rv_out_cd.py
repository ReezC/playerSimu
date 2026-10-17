# -*- coding: utf-8 -*-
"""反向验证：「输出行为CD ⇒ 虚线框 + 转向推迟」这条链**逐处拆开**，看用例是否当场红。

用法：`python data/_rv_out_cd.py`（跑完自动还原 ✓ 只改 agent / live_thread 两份 ✓）
为什么要有它：新用例要是"拆了也不红"，那它钉的就是空气（本仓库的规矩 ✓）。
"""
import io
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
AG = ROOT / "decision" / "agent.py"
LT = ROOT / "gui" / "live_thread.py"

_PRE = ("import sys; sys.path.insert(0, '.'); "
        "import decision.agent as ag; ag.DecisionSettings.save = lambda self: None; ")
RUN_DEC = [sys.executable, "-X", "utf8", "-c",
           _PRE + "import tools.selftest_decision as S; "
                  "S.t_out_cd_turn_defer(); print('PASS')"]
RUN_LP = [sys.executable, "-X", "utf8", "-c",
          _PRE + "import tools.selftest_live_panel as L; "
                 "L.t_out_cd_dashed_attack_box(); print('PASS')"]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    out = (p.stdout or "") + (p.stderr or "")
    if "PASS" in out:
        return "PASS  ← 反向验证失败 ✗"
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    for ln in reversed(lines):
        if ln.startswith("AssertionError:"):
            return "RED   " + ln.split(":", 1)[1].strip()[:110]
        if ln.startswith(("TypeError:", "KeyError:", "AttributeError:")):
            return "RED   " + ln[:110]
    return "RED   " + (lines[-1][:110] if lines else "（没输出）")



def _drop_pyc(path):
    """还原之后**删掉对应的 `.pyc`** —— 同一秒内重写同一个 `.py`，`.pyc` 头里的 mtime
    **只到秒** ⇒ Python 会认为缓存还新鲜、拿**旧字节码**跑 ✗（本轮踩过两次 ✓）。"""
    p = pathlib.Path(path)
    d = p.parent / "__pycache__"
    if not d.is_dir():
        return
    for f in d.glob(p.stem + "*.pyc"):
        try:
            f.unlink()
        except OSError:
            pass

def rv(name, path, old, new, cmd=None):
    p = pathlib.Path(path)
    src = io.open(p, encoding="utf-8").read()
    assert old in src, "锚点没找到：" + name
    io.open(p, "w", encoding="utf-8").write(src.replace(old, new, 1))
    try:
        print("%s | %s" % (run(cmd or RUN_DEC), name))
    finally:
        io.open(p, "w", encoding="utf-8").write(src)
        _drop_pyc(p)


# ① 站桩开窗不往 CD 后推（恢复"开窗就用 now"）
rv("① 站桩补朝向的开窗不推到 CD 之后", AG,
   "_start = max(float(now), self._out_cd_end()) if hold > 0 else float(now)",
   "_start = float(now)")
# ② 补键那一段没有下界（窗口推了也照样提前按）
rv("② 补键缺少「窗口开始」那个下界", AG,
   "                and now >= float(self._turn_due_at or 0.0)):",
   "                and True):")
# ③ 换向那一下（_hold_turn）没有 CD 门
rv("③ `_hold_turn` 里那道 CD 门被拆掉", AG,
   "        if not self.out_cd_over(now):\n"
   "            # 还在输出行为CD 里 ⇒ **把换向锚推到 CD 到点**（这一拍不按 ✓ 见上面那段 ✓）\n"
   "            self._turn_at = self._out_cd_end()\n"
   "            return keys\n", "")
# ④ 转身够得着背后那只：CD 门被拆掉
rv("④ `_turn_to_reachable_target` 里那道 CD 门被拆掉", AG,
   "        _now = time.monotonic() if now is None else now\n"
   "        if not self.out_cd_over(_now):",
   "        _now = time.monotonic() if now is None else now\n"
   "        if False:")
# ⑤ 读数口径坏了（恒 0 ⇒ 决策侧那条断言红）
rv("⑤ `out_cd_left_s()` 恒 0（读数口径坏了）", AG,
   "        return float(self._out_cd_end()) - _now",
   "        return 0.0 - _now" if False else "        return 0.0")
# ⑥ 虚线画成实线（`_draw_dashed_segment` 一次画满整条）
rv("⑥ 虚线被画成实线", LT,
   "        s1 = min(s0 + dash, length)",
   "        s1 = length", cmd=RUN_LP)
# ⑦ `dashed=` 也喂给盲区框（不止攻击范围框）
rv("⑦ `dashed=` 多喂了一处（盲区框也吃）", LT,
   "                _draw_box(min_ad, _min_attack_color, _on_blind,\n"
   "                          width=_min_attack_width)",
   "                _draw_box(min_ad, _min_attack_color, _on_blind,\n"
   "                          width=_min_attack_width, dashed=_out_cd_dash)", cmd=RUN_LP)
# ⑧ **"扩所有转向"**：正常路径（走位/追击/巡逻/规避都走它）那道门拆掉
rv("⑧ 正常路径发键前那道门被拆掉（走位/追击照样按左右）", AG,
   "        keys = self._gate_turn_keys(keys, now)\n"
   "        self.keys.set(keys)",
   "        self.keys.set(keys)")
# ⑨ 门里把上下也一起摘了（绳上松手会掉下来 ✗）
rv("⑨ 那道门连**上下**一起摘（绳上危险）", AG,
   '        _lr = {km.get("left"), km.get("right")}\n'
   "        _lr.discard(None)\n"
   "        return {k for k in keys if k not in _lr}",
   '        _lr = {km.get("left"), km.get("right"), km.get("up"), km.get("down")}\n'
   "        _lr.discard(None)\n"
   "        return {k for k in keys if k not in _lr}")
# ⑩ 回身输出那条序列的起跑门拆掉
rv("⑩ 回身输出序列在 CD 里就开跑（起跑门拆掉）", AG,
   "            if self._back_ctx is None and not turn_wait and self.out_cd_over(now):",
   "            if self._back_ctx is None and not turn_wait:")
# ⑪ idle 那个发键点没过门（源码钉该红）
rv("⑪ idle 那个发键点没过门", AG,
   "                _ikeys = self._gate_turn_keys(_ikeys, now)   # ⭐ 同款（见 `_gate_turn_keys`）\n",
   "")
# ⑫ 转向闸里"刚换向那一窗"那一项拆掉（CD 一到点输出就抢拍 ⇒ 转向被切成半下）
rv("⑫ 转向闸没算进「刚换向」那一窗", AG,
   "        if hold > 0.0 and self._turn_at and (_now - float(self._turn_at)) < hold:\n"
   "            # 正按着\"刚换向\"那一下（含\"被 CD 推到将来\"的锚 ✓）⇒ 输出等它按完 + 延迟 ✓\n"
   "            blk = max(blk, float(self._turn_at) + hold + delay)\n", "")
# ⑬ 只留一处"接回被推迟的转身"（另一处漏了 = 那一边还是会丢意图 ✗）
rv("⑬ 「接回被推迟的转身」只剩一处调用", AG,
   "                # ⭐ 同「多点巡逻」：先接回那次被推迟的转身（一处口径 ✓ 见 `_turn_resume_target`）\n"
   "                target = self._turn_resume_target(target, mobs, ws)\n", "")
print("（跑完已全部还原 ✓）")
