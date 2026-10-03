"""⭐ 测谎对照台（**常驻工具** ✓ 用户 2026-10-03 第 0 步 ✓）。

为什么要有它 ✗✗：本轮调参反复出现"改一处、另一处塌"（选框速度门一开 ⇒ 融合 25→1 帧 ✗；
融合 IoU 用 0.5 ⇒ 融合只剩 1~19 帧 ✗ ……）⇒ **只靠"看着像好了"没法判断得失** ✗
⇒ 每次改动都用同一套数字量一遍 ✓。

用法（在项目根目录）：

    python -m tools.ab_compare                     # 跑默认素材，打印"当前"一组
    python -m tools.ab_compare --save base         # 存成快照 `tools/ab_runs/base.json`
    python -m tools.ab_compare --set merge_iou=0.75 --diff base
                                                   # 改参数跑一遍，**并与 base 逐项对比**

`--set` 可覆盖（`Runner` 透传的那批参数 + 登记表参数）：
    path_ms / merge_iou / merge_iou_out / sep_ratio / inherit_dist / allow_back_ratio /
    edge_max / sep_vel_max_ratio / noise_tol / board_s / board_overlap / iou_brick /
    select_step_ratio / clamp_step_ratio / follow_gain

指标（每段素材一行；⚠ 都是**素材可直接量到的**，不含真值）：
    · `融合`  = `_tbox_merged` 为真的帧数（会话活多久 ✓）
    · `分离`  = `_merge_split` 为真的次数（判出去几次 ✓）
    · `跟丢`  = 报告位置缺失的帧数
    · `折角`  = 白线（`path_pts` 末点）逐帧位移的最大值 + Top3（"抖不抖" ✓）
⚠ **真值类指标**（位置误差 p50/p90）要合成台那条链，不在这里 ✗（本轮先把"可量"的立起来 ✓）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path(__file__).resolve().parents[1] / "datasets" / "liedetectorVideo"
RUNS = Path(__file__).resolve().parent / "ab_runs"
#: 默认素材（四段 ✓ —— 与用户平时看的那几段一致 ✓）
DEFAULT_SRC = ("10月1日.mp4", "10月2日.mp4", "9月30日(1).mp4", "10月3日.mp4")
#: `--set` 允许的名字（= 一路透传到 `Runner` 的那批 ✓）
TUNABLE = ("path_ms", "merge_iou", "merge_iou_out", "sep_ratio", "inherit_dist",
           "allow_back_ratio", "edge_max", "sep_vel_max_ratio", "noise_tol", "board_s",
           "board_overlap", "iou_brick", "select_step_ratio", "clamp_step_ratio",
           "follow_gain",
           # ⭐ **参数分档（用户 2026-10-03 ✓ 第 2/3 行）**：每档一对 ✓
           "ring_cov_sep", "brick_iou_sep", "ring_cov_fuse", "brick_iou_fuse")


def _parse_sets(pairs):
    out = {}
    for _p in pairs or []:
        if "=" not in _p:
            raise SystemExit("--set 要写成 k=v（收到 %r）" % (_p,))
        _k, _v = _p.split("=", 1)
        _k = _k.strip()
        if _k not in TUNABLE:
            raise SystemExit("--set 不认这个参数：%r（可用：%s）" % (_k, ", ".join(TUNABLE)))
        out[_k] = float(_v) if ("." in _v or "e" in _v.lower()) else int(_v)
    return out


def _measure(name, cfg, override):
    """跑一段素材 ⇒ 指标字典（**全部来自 run 的返回/内部状态** ✓ 不猜 ✗）。"""
    from tools.lie_demo import DetsWorker, Runner, load_source

    frames, ts, _p, _s, _fps = load_source(str(BASE / name))
    _w = DetsWorker(None, conf=0.25)
    dets = [_w.detect(s) for _f, s in frames]
    _w.close()
    kw = {"dets": dets, "path_ms": cfg.get("path_ms"), "sep_ratio": cfg.get("sep_ratio"),
          "inherit_dist": cfg.get("inherit_dist"),
          "allow_back_ratio": cfg.get("allow_back_ratio"), "edge_max": cfg.get("edge_max"),
          "sep_vel_max_ratio": cfg.get("sep_vel_max_ratio"),
          "noise_tol": cfg.get("noise_tol"), "board_s": cfg.get("board_s"),
          "board_overlap": cfg.get("board_overlap"), "iou_brick": cfg.get("iou_brick"),
          "merge_iou": cfg.get("merge_iou"), "merge_iou_out": cfg.get("merge_iou_out"),
          "ring_cov_sep": cfg.get("ring_cov_sep"), "brick_iou_sep": cfg.get("brick_iou_sep"),
          "ring_cov_fuse": cfg.get("ring_cov_fuse"),
          "brick_iou_fuse": cfg.get("brick_iou_fuse")}
    kw.update(override)
    rr = Runner(**kw)
    tr = rr.tr
    _nmg = _nsp = _nla = 0
    _prev, _jumps = None, []
    for i, fr in enumerate(frames):
        tr._cf = i + 1
        _o, _pos, _rad, _h, _b, _mo = rr.step(fr[1], i, ts[i])
        _nmg += 1 if bool(getattr(tr, "_tbox_merged", False)) else 0
        _nsp += 1 if bool(getattr(tr, "_merge_split", False)) else 0
        _nla += 1 if _pos is None else 0
        _pp = (_mo or {}).get("path_pts") or []
        _tail = None if not _pp else (float(_pp[-1][0]), float(_pp[-1][1]))
        if _prev is not None and _tail is not None:
            _jumps.append(round(((_tail[0] - _prev[0]) ** 2
                                 + (_tail[1] - _prev[1]) ** 2) ** 0.5, 1))
        _prev = _tail
    _top = sorted(_jumps, reverse=True)[:3]
    return {"frames": len(frames), "merged": _nmg, "split": _nsp, "lost": _nla,
            "jump_max": (_top[0] if _top else 0.0), "jump_top3": _top}


def _line(name, m):
    return ("%-14s ｜ 融合 %3d 帧 ｜ 分离 %2d 次 ｜ 跟丢 %2d 帧 ｜ 最大折角 %6.1f ｜ Top3 %s"
            % (name, m["merged"], m["split"], m["lost"], m["jump_max"], m["jump_top3"]))


def _diff(name, old, new):
    def _d(k):
        return new[k] - old[k]
    _jm = ("%+.1f" % _d("jump_max")) if old.get("jump_max") is not None else "?"
    print("%-14s ｜ 融合 %+4d ｜ 分离 %+3d ｜ 跟丢 %+3d ｜ 最大折角 %s（%.1f → %.1f）"
          % (name, _d("merged"), _d("split"), _d("lost"), _jm,
             old.get("jump_max", 0.0), new.get("jump_max", 0.0)))


def main():
    ap = argparse.ArgumentParser(description="测谎对照台（第 0 步 ✓）")
    ap.add_argument("--src", nargs="*", default=list(DEFAULT_SRC), help="素材文件名")
    ap.add_argument("--set", nargs="*", default=[], help="参数覆盖 k=v（可多个）")
    ap.add_argument("--save", default=None, help="把这次结果存成快照名")
    ap.add_argument("--diff", default=None, help="与某个快照逐项对比")
    args = ap.parse_args()

    from gui import theme
    cfg = dict(theme.load_section("lie_demo") or {})
    override = _parse_sets(args.set)
    print("参数覆盖：%s" % (override or "（无 ✓ 用界面里存的那份配置）"))
    out = {}
    _t0 = time.time()
    for _name in args.src:
        if not (BASE / _name).exists():
            print("⚠ 找不到素材：%s" % (_name,))
            continue
        _m = _measure(_name, cfg, override)
        out[_name] = _m
        print(_line(_name, _m))
    print("（%.1f s）" % (time.time() - _t0))
    if args.diff:
        _f = RUNS / ("%s.json" % (args.diff,))
        if not _f.exists():
            print("⚠ 快照不存在：%s（先 --save %s）" % (_f, args.diff))
        else:
            _old = json.loads(_f.read_text(encoding="utf-8"))
            _old = _old.get("result", _old)      # ⚠ 快照外层包了 `{"override", "result"}` ✓
            print("---- 与快照「%s」的差（新 − 旧）----" % (args.diff,))
            for _name in out:
                if _name in _old:
                    _diff(_name, _old[_name], out[_name])
    if args.save:
        RUNS.mkdir(parents=True, exist_ok=True)
        _f = RUNS / ("%s.json" % (args.save,))
        _f.write_text(json.dumps({"override": override, "result": out}, ensure_ascii=False,
                                 indent=1), encoding="utf-8")
        print("已存快照：%s" % (_f,))


if __name__ == "__main__":
    main()
