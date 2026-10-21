# -*- coding: utf-8 -*-
"""反向验证：「自动拾取 + 键盘映射『拾取』」这条链逐处拆开，看用例是否当场红。

用法：`python data/_rv_auto_pickup.py`（跑完自动还原 ✓ 只改 agent / live_thread / input / panel ✓）
"""
import io
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
AG = ROOT / "decision" / "agent.py"
LT = ROOT / "gui" / "live_thread.py"
INP = ROOT / "execution" / "input.py"
PP = ROOT / "gui" / "player_panel.py"

_PRE = ("import sys; sys.path.insert(0, '.'); "
        "import decision.agent as ag; ag.DecisionSettings.save = lambda self: None; ")
RUN_DEC = [sys.executable, "-X", "utf8", "-c",
           _PRE + "import tools.selftest_decision as S; "
                  "S.t_auto_pickup(); print('PASS')"]
RUN_LP = [sys.executable, "-X", "utf8", "-c",
          _PRE + "import tools.selftest_live_panel as L; "
                 "L.t_drop_touch(); print('PASS')"]


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


# ① 默认改成「开」
rv("① 默认值改成开（老项目会乱点键）", AG,
   "        self.auto_pickup = False", "        self.auto_pickup = True")
# ② 老配置兜底改成开
rv("② `from_dict` 兜底改成开", AG,
   '        self.auto_pickup = (data.get("auto_pickup") is True)',
   '        self.auto_pickup = (data.get("auto_pickup") is not False)')
# ③ 拆掉「接触」那道门（没碰着也点）
rv("③ 拆掉「接触」那道门", AG,
   '        if not bool(getattr(getattr(ws, "player", None), "touching_drop", False)):\n'
   "            return False                     # 没碰着掉落物 ⇒ 不点 ✓\n", "")
# ④ 拆掉「点按间隔」那道门（每拍狂点）
rv("④ 拆掉点按间隔那道门（每拍狂点）", AG,
   "        if now < self._next_pickup:\n            return False                     # 上一次点按的间隔还没过 ✓\n",
   "")
# ⑤ 判据恒 False（永远判不出接触）
rv("⑤ `drop_touch` 恒 False", LT,
   "    if not player_box or not drop_boxes:\n        return False",
   "    if not player_box or not drop_boxes:\n        return False\n"
   "    return False", cmd=RUN_LP)
# ⑥ 判据只看水平重叠（竖直不管）
rv("⑥ `drop_touch` 只看水平重叠", LT,
   "        if min(x2, px2) < max(x1, px1) or min(y2, py2) < max(y1, py1):",
   "        if min(x2, px2) < max(x1, px1):", cmd=RUN_LP)
# ⑦ 感知层不写那个布尔（决策侧永远 False）
rv("⑦ 感知层不写 `touching_drop`", LT,
   "                    ws.player.touching_drop = drop_touch(player_box or _last_player[0],\n"
   "                                                         _drop_boxes)",
   "                    pass", cmd=RUN_LP)
# ⑧ 界面那一行的默认值和 `DEFAULT_KEYMAP` 不一致
rv("⑧ 键盘映射行的默认值与 `DEFAULT_KEYMAP` 分叉", PP,
   '    ("pickup", "拾取", "z"),', '    ("pickup", "拾取", "x"),')
print("（跑完已全部还原 ✓）")
