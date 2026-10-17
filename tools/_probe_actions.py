"""临时诊断：对比新旧模板选择策略的动作覆盖情况。"""
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.detect_mobs import _pick_frames

P = pathlib.Path("datasets/sprites/mob")
MOBS = ["0130100", "0130101", "0210100", "1210100"]


def old_pick(files, per_mob):
    pref = [f for f in files if f.name.startswith(("stand_", "move_"))]
    rest = [f for f in files if f not in pref]
    return (pref + rest)[:per_mob]


for per in (6, 10):
    print("=== per_mob = %d ===" % per)
    for mid in MOBS:
        d = P / mid
        if not d.is_dir():
            continue

        fs = sorted(d.glob("*.png"))
        old = old_pick(fs, per)
        new = _pick_frames(fs, per)

        all_acts = {f.name.rsplit("_", 1)[0] for f in fs}
        old_acts = {f.name.rsplit("_", 1)[0] for f in old}
        new_acts = {f.name.rsplit("_", 1)[0] for f in new}

        print("%-10s 全部动作 %s" % (mid, sorted(all_acts)))
        print("%-10s   旧: 漏 %s" % ("", sorted(all_acts - old_acts) or "无"))
        print("%-10s   新: 漏 %s   (模板 %d 个)" % ("", sorted(all_acts - new_acts) or "无", len(new)))
    print()
