"""反向验证：把「检测框颜色 每一类前面那个勾」这条链**逐处拆开**，看用例是否**当场红**。

用法：`python data/_rv_class_vis.py`（跑完自动还原 ✓ 只改这三份文件、只跑一条用例 ✓）
为什么要有它：新用例要是"拆了也不红"，那它钉的就是空气（本仓库的规矩：新钉子都要反向验一遍 ✓）。
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LT = ROOT / "gui" / "live_thread.py"
SD = ROOT / "gui" / "settings_dialog.py"
TH = ROOT / "gui" / "theme.py"

RUN = [sys.executable, "-X", "utf8", "-c",
       "import sys; sys.path.insert(0, '.');"
       "import tools.selftest_live_panel as L;"
       "L.t_class_box_switches(); print('PASS')"]


def run():
    p = subprocess.run(RUN, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    out = (p.stdout or "") + (p.stderr or "")
    if "PASS" in out:
        return "PASS  ← 反向验证失败 ✗"
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    for ln in reversed(lines):                    # 从后往前找那条断言消息 ✓
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


def rv(name, path, old, new):
    p = pathlib.Path(path)
    src = p.read_text(encoding="utf-8")
    assert old in src, "锚点没找到：" + name
    p.write_text(src.replace(old, new, 1), encoding="utf-8")
    try:
        print("%s | %s" % (run(), name))
    finally:
        p.write_text(src, encoding="utf-8")
        _drop_pyc(p)                          # 同秒重写 ⇒ 旧 `.pyc` 会被当新鲜 ✗ 见 SKILL 295


rv("① 开关的兜底口径改成「不画」（默认不再全勾）", TH,
   "out[k] = bool(vis[k]) if isinstance(vis.get(k), bool) else True",
   "out[k] = bool(vis[k]) if isinstance(vis.get(k), bool) else False")
rv("② 界面只给第一个类别做那一行（其余类别没勾选框）", SD,
   "        for cid, en, zh, _bgr in classes.CLASSES:",
   "        for cid, en, zh, _bgr in classes.CLASSES[:1]:")
rv("③ 界面勾了不即时写盘（取消那根连接）", SD,
   "                lambda _st, k=_on_key: theme.save_vis(\n"
   "                    {k: bool(self._vis_on[k].isChecked())}))",
   "                lambda _st: None)")
rv("④ 纯函数不看开关（永远画）", LT,
   "        if isinstance(on, bool) and not on:\n            continue",
   "        if False:\n            continue")
rv("⑤ 玩家框不吃它那个勾", LT,
   '                            and _vis_cfg.get("player_on", True)):', "):")
rv("⑥ 只去掉「画异步」那个调用点的框", LT,
   "(_attack_color, _attack_width, _chase_jump_color, _chase_jump_width, _cls_colors, _cls_dets,",
   "(_attack_color, _attack_width, _chase_jump_color, _chase_jump_width, _cls_colors, [],")
print("（跑完已全部还原 ✓）")
