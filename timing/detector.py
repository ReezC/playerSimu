"""按键注入检测器：基于时序统计特征，判定按键间隔序列的来源。

用法:
    python detector.py human_tap.json inject_fix.json   # 基线 + 待测样本
    python detector.py human_tap.json inject_rnd.json

第一个参数是真人基线样本（如 human_tap.json），第二个是待检测样本。

判定规则:
    R1: cv < 0.05            -> 固定间隔注入（间隔几乎恒定）
    R2: KS 检验 vs 基线 D>0.4 -> 分布形状异常（如均匀随机注入）
"""
import json
import sys

import numpy as np


def load_intervals(path):
    with open(path) as f:
        events = json.load(f)
    ts = [e[0] for e in events if e[1] == "down"]
    return [(ts[i + 1] - ts[i]) / 1e6 for i in range(len(ts) - 1)]


def detect(intervals, baseline):
    a = np.asarray(intervals, dtype=float)
    b = np.asarray(baseline, dtype=float)

    if len(a) < 5:
        print("样本不足（<5 个间隔），无法判定")
        return None

    mean, std = a.mean(), a.std()
    cv = std / mean if mean > 0 else float("inf")
    print(f"n={len(a)}  mean={mean:.2f}ms  std={std:.2f}ms  cv={cv:.3f}")

    # R1: 固定间隔
    if cv < 0.05:
        print(f"[判定] 固定间隔注入  (cv={cv:.3f} < 0.05，间隔几乎恒定)")
        return "fixed"

    # R2: KS 检验 vs 真人基线
    from scipy import stats as st
    D, p = st.ks_2samp(a, b)
    print(f"KS vs 基线: D={D:.3f}  p={p:.3g}")

    if D > 0.4:
        print(f"[判定] 分布异常注入  (D={D:.3f} > 0.4，分布形状偏离真人基线)")
        return "random"
    print("[判定] 真人  (分布与基线一致)")
    return "human"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python detector.py <baseline.json> <sample.json>")
        sys.exit(1)
    bl = load_intervals(sys.argv[1])
    sp = load_intervals(sys.argv[2])
    detect(sp, bl)
