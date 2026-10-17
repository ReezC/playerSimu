"""只读探针：`PlayerLocator` 加了「框到可见部分」之后，**默认关**时与 HEAD 那版**逐帧比**。

为什么要有它：改的是**运行时也要 import 的那个类**（`perception/player_state.py` 拿它定位 ✓）
⇒ "默认关 ⇒ 一字不差"这句话必须**量过**才能说 ✓（本仓库的老规矩：改了没变化 ⇒ 先采样 ✓）。
比法：`git show HEAD:perception/player_locator.py` 落成临时模块 ⇒ 两边在**同一批真帧**上
各 `locate()` 一遍 ⇒ 六元组必须**完全一致** ✓。
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from core.imgio import imread  # noqa: E402
from perception.player_locator import PlayerLocator  # noqa: E402

# 找一个真有玩家模板的项目（挑第一个能建起来的 ✓）
spr = ROOT / "datasets" / "sprites" / "player"
pid = next((d.name for d in sorted(spr.iterdir()) if d.is_dir() and list(d.glob("*.png"))), "")
print("玩家模板目录：", spr / pid if pid else "（一个都没有 ✗）")
if not pid:
    sys.exit(0)

frames_dir = None
for p in sorted((ROOT / "projects").iterdir()):
    fr = p / "frames"
    if fr.is_dir() and list(fr.glob("*.png")):
        frames_dir = fr
        break
if frames_dir is None:
    print("没有哪份项目带 frames ⇒ 拿不到真帧，跳过")
    sys.exit(0)
frames = sorted(frames_dir.glob("*.png"))[:6]
print("真帧：%s（%d 张）" % (frames_dir, len(frames)))

src = subprocess.run(["git", "show", "HEAD:perception/player_locator.py"],
                     cwd=str(ROOT), capture_output=True, text=True)
if src.returncode != 0 or not src.stdout.strip():
    print("拿不到 HEAD 那版（%s）⇒ 跳过" % src.stderr.strip()[:80])
    sys.exit(0)
tmp = Path(tempfile.mkdtemp(prefix="plequiv_"))
f = tmp / "pl_head.py"
f.write_text(src.stdout, encoding="utf-8")
spec = importlib.util.spec_from_file_location("pl_head", f)
mod = importlib.util.module_from_spec(spec)
sys.modules["pl_head"] = mod
spec.loader.exec_module(mod)

a = PlayerLocator(pid)                                  # 新（默认 visible=False ✓）
b = mod.PlayerLocator(pid)                              # HEAD 那版
print("模板数：新 %d / HEAD %d" % (a.template_count, b.template_count))
bad = 0
for fp in frames:
    img = imread(fp)
    ra, rb = a.locate(img), b.locate(img)
    same = (ra is None and rb is None) or (
        ra is not None and rb is not None
        and all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(ra[:4], rb[:4]))
        and ra[5] == rb[5])
    if not same:
        bad += 1
        print("  ✗ %s\n     新   = %r\n     HEAD = %r" % (fp.stem, ra, rb))
    else:
        print("  ✓ %s -> %s" % (fp.stem, "None" if ra is None else
                               "(%d, %d, %d, %d, %.4f, %s)"
                               % (ra[0], ra[1], ra[2], ra[3], ra[4], ra[5])))
print("结论：%d/%d 帧完全一致%s" % (len(frames) - bad, len(frames),
                                " ⇒ **默认关 = 老行为一字不差** ✓" if not bad else " ✗ 有差异！"))
