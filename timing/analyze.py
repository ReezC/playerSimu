"""按键间隔分析：读采集器输出，计算间隔分布统计量，做两两分布对比。

依赖: numpy, scipy（ultralytics 项目一般已带）

用法:
    python analyze.py human.json inject_fix.json inject_rnd.json

关键指标解释:
    cv（变异系数 = std/mean）: 真人敲键通常在 0.2~0.5；固定间隔注入接近 0。
    KS 检验: D 越小 / p 越大，说明两个样本越像同一个分布。
"""
import json
import sys

import numpy as np


def load_down_times(path):
    with open(path) as f:
        events = json.load(f)
    return [e[0] for e in events if e[1] == "down"]


def intervals_ms(path):
    ts = load_down_times(path)
    return [(ts[i + 1] - ts[i]) / 1e6 for i in range(len(ts) - 1)]


def stats(name, iv):
    a = np.asarray(iv, dtype=float)
    if len(a) == 0:
        print(f"[{name}] 无间隔数据")
        return None
    cv = a.std() / a.mean() if a.mean() > 0 else float("inf")
    print(f"[{name}] n={len(a)}  mean={a.mean():.2f}ms  std={a.std():.2f}ms  "
          f"cv={cv:.3f}  min={a.min():.2f}  max={a.max():.2f}")
    return a


def main():
    paths = sys.argv[1:]
    if not paths:
        print("用法: python analyze.py <events1.json> [events2.json] ...")
        return

    arrs = {}
    for p in paths:
        arrs[p] = stats(p, intervals_ms(p))

    valid = {k: v for k, v in arrs.items() if v is not None and len(v) >= 5}
    if len(valid) >= 2:
        from scipy import stats as st
        names = list(valid)
        print("\n--- KS 检验（D 越小越像同一分布；p 越大越不能拒绝同分布）---")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = valid[names[i]], valid[names[j]]
                d, p = st.ks_2samp(a, b)
                print(f"{names[i]}  vs  {names[j]}: D={d:.3f}  p={p:.3g}")


if __name__ == "__main__":
    main()
