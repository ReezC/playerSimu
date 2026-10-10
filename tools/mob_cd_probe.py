"""灰框取证（**只读**）：把「经常重复打灰框」这件事量成能回贴的结论。

**为什么要有它**（用户 2026-10-10 ✓ 原话："**经常出现重复打灰框的现象**"✗）：
灰框 = 「目标被攻击CD」（`gui/live_thread.MOB_CD_GRAY` ✓ = "我打过这只怪，它在冷却" ✓）。
"重复"有三种**完全不同**的形状，光看画面分不出来 ✗ —— 这个工具把它们分开：

  ① **续盖**（同一只怪被反复盖戳 ⇒ CD 被往后推 ⇒ 灰框"赖着不走"）
     ⇒ 判据：`ext=1` 连着一串 ✓；
  ② **正常轮换**（一只一只换、不重样）
     ⇒ 判据：`ext=0` 且**怪号**不重复 ✓；
  ③ **同一只怪被反复打**（每 CD 时长一次 ⇒ 灰框周期性地在同一只身上出现）
     ⇒ 判据：同一怪号反复出现，**且间隔 ≈ 配置的 CD**（实测现场：`mob=820` 每 **2.00** 秒一次 ✓）；
  ④ **换号**（同一只怪"丢了又认回来"换了新 id ⇒ CD 按号记账**罩不住它** ✗ 那才是真 bug）
     ⇒ 判据：按**站位**（玩家世界坐标 ±`--bucket`）分桶，同一块地方出现**很多**不同怪号。

数据来源：`behavior.log` 的
  · `mob_atk_cd  cd_ms=… ext=0/1 mob=… n=…`（**盖戳留痕** ✓ 2026-10-10 补 ✓）
  · `pos  … x=… y=…`（玩家的世界坐标 ✓ 用来做"站位分桶" ✓）

跑法（仓库根）：
    python -m tools.mob_cd_probe                # 读默认 behavior.log
    python -m tools.mob_cd_probe --log 别的.log  # 换文件
    python -m tools.mob_cd_probe --bucket 100   # 站位桶大小（世界像素）

⚠ **只读** ✓：不按键、不写配置、不碰 `datasets/` 与 `projects/` ✓。
"""

import argparse
import collections
import datetime as dt
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EV = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\s+(\S+)\s+(.*)$")
KV = re.compile(r"(\w+)=([^\s]+)")


def _ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def collect(log_path):
    """读日志 → `(盖戳列表[(t, 怪号, ext, cd_ms)], 站位列表[(t, x, y)])`。"""
    stamps, poses = [], []
    try:
        text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print("  读不到日志：%s（%s）" % (log_path, e))
        return stamps, poses
    for ln in text.splitlines():
        m = EV.match(ln)
        if not m:
            continue
        t, kind, rest = m.groups()
        if kind == "mob_atk_cd":
            d = dict(KV.findall(rest))
            if d.get("mob"):
                try:
                    stamps.append((t, d["mob"], int(d.get("ext") or 0),
                                   int(d.get("cd_ms") or 0)))
                except ValueError:
                    pass
        elif kind == "pos":
            d = dict(KV.findall(rest))
            try:
                poses.append((t, float(d["x"]), float(d["y"])))
            except (KeyError, ValueError):
                pass
    return stamps, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "behavior.log"))
    ap.add_argument("--bucket", type=float, default=50.0,
                    help="站位分桶大小（世界像素；同一块地方冒出很多怪号 ⇒ 换号 ✗）")
    ap.add_argument("--tail", type=int, default=12)
    a = ap.parse_args()

    stamps, poses = collect(a.log)
    print("灰框取证（只读）  %s" % dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("日志：%s" % a.log)
    print("  mob_atk_cd %d 条 ｜ pos %d 条" % (len(stamps), len(poses)))
    if not stamps:
        print("  （还没有盖戳留痕 —— 要么没打过怪，要么工作台跑的是 2026-10-10 之前的代码 ✓）")
        return 0

    print("  时间范围: %s → %s" % (stamps[0][0][11:], stamps[-1][0][11:]))
    ex = collections.Counter(s[2] for s in stamps)
    print("  ext: 新盖(0)=%d  续盖(1)=%d   ← 续盖连串 = ①；全 0 = 没有「赖着不走」✓"
          % (ex[0], ex[1]))

    per = collections.defaultdict(list)
    for i, (t, mob, ext, cd) in enumerate(stamps):
        per[mob].append((t, ext, cd, i))
    rep = [(m, v) for m, v in per.items() if len(v) >= 2]
    print("  同一只怪被盖过 ≥2 次的：%d 只（共 %d 只）" % (len(rep), len(per)))
    for mob, v in sorted(rep, key=lambda kv: -len(kv[1]))[:6]:
        gaps = []
        for j in range(1, len(v)):
            gaps.append("%.2fs" % (_ts(v[j][0]) - _ts(v[j - 1][0])).total_seconds())
        cds = sorted({x[2] for x in v})
        print("     mob=%-8s 盖 %d 次  间隔: %s   配置 CD: %s ms"
              % (mob, len(v), " ".join(gaps[:5]), "/".join(str(c) for c in cds)))
        if cds and all(g.endswith("s") for g in gaps):
            near = [abs(float(g[:-1]) - cds[0] / 1000.0) < 0.35 for g in gaps]
            if near and all(near):
                print("        ⇒ ③ **每 CD 时长就再打一次**（灰框周期性出现在同一只身上 ✓"
                      " —— 这是「打过一次歇一会儿」的定义行为 ✓ 想少看到就把 CD 调大 ✓）")

    if poses:
        pi, joined = 0, []
        for t, mob, _ext, _cd in stamps:
            while pi + 1 < len(poses) and poses[pi + 1][0] <= t:
                pi += 1
            joined.append((mob, poses[pi][1], poses[pi][2]))
        buckets = collections.defaultdict(set)
        for mob, x, y in joined:
            buckets[(round(x / a.bucket), round(y / a.bucket))].add(mob)
        top = sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:6]
        print("  按站位分桶（±%g 世界像素）⇒ 同一块地方出现过的不同怪号数：" % a.bucket)
        for (bx, by), ids in top:
            print("    站位≈(%6d, %6d)  %d 个怪号" % (bx * a.bucket, by * a.bucket, len(ids)))
        many = [k for k, v in buckets.items() if len(v) >= 4]
        print("  ⇒ 有 %d 块地方冒出过 ≥4 个怪号 %s"
              % (len(many), "（**疑似换号** ✗ 见工具说明 ④）" if many else "（没看到换号 ✓）"))

    print("  末尾 %d 条：" % a.tail)
    prev = None
    for t, mob, ext, cd in stamps[-a.tail:]:
        gap = ""
        if prev:
            gap = "  (+%.2fs)" % (_ts(t) - _ts(prev)).total_seconds()
        print("    %s  mob=%-7s ext=%d cd_ms=%-5d%s" % (t[11:], mob, ext, cd, gap))
        prev = t
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
