"""标定画面里精灵的实际缩放比例。

**为什么必须做**
    模板匹配和合成数据集都要求「模板尺寸 = 画面里目标的尺寸」。
    这个比例取决于游戏分辨率、窗口大小、推流缩放，
    没有文档可查，只能实测。

**原理**
    拿每只怪的 stand 帧当模板，在画面上按一组尺度逐个匹配，
    取「区分度（峰值 − 高分位数）」最高的那一组 ——
    它同时给出：哪只怪 + 缩放比例 + 位置。

**注意**
    标定结果是「内部中间量」，界面上不该给用户看。
    用户只需要知道画面多大、标定过没有、要不要重标。

CLI:
    python -m tools.calibrate_scale --frames data/frames ^
        --sprites datasets/sprites/mob --mobs 0130100,0130101

GUI:
    调 run_calibrate(params, ctx)，见 core/context.py
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core.context import ConsoleContext, TaskContext
from core.imgio import imread


def find_stand_frames(mob_root, limit, only=None):
    """每只怪只取 stand 的第一帧作为模板。"""
    out = []
    root = Path(mob_root)

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if only and d.name != only:
            continue

        stands = sorted(d.glob("stand_*.png"))
        if not stands:
            continue

        out.append((d.name, stands[0]))
        if limit and len(out) >= limit:
            break

    return out


def _parse_region(region):
    if isinstance(region, str) and region.strip():
        try:
            return tuple(int(v) for v in region.split(","))
        except Exception:
            raise ValueError('区域格式应为 "x,y,w,h"，收到: %s' % region)
    return None


def _trim(tpl):
    """把模板裁到 alpha 包围盒。大片透明边框会把响应图整体抬高，制造假匹配。"""
    if tpl is None or tpl.ndim != 3 or tpl.shape[2] != 4:
        return None

    b, al = tpl[:, :, :3], tpl[:, :, 3]
    ys, xs = np.nonzero(al > 128)
    if len(xs) < 200:
        return None

    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    if b.shape[0] < 20 or b.shape[1] < 20:
        return None

    return b, al


def run_calibrate(params, ctx=None):
    """多尺度模板匹配，找出画面里精灵的实际缩放。

    params:
        frames      画面目录（会均匀采样几帧）
        sprites     精灵库目录
        mobs        候选怪种（留空则用库里前 max_mobs 只）
        min_scale   扫描下限
        max_scale   扫描上限
        step        扫描步长
        sample      采样几帧（默认 5）
        max_mobs    没有 mobs 时最多用几只怪（默认 60）
        region      "x,y,w,h" 或 None
    returns:
        dict，含 scale / distinct / confidence / summary
    """
    ctx = ctx or TaskContext()

    frames_dir = Path(params["frames"])
    all_frames = sorted(frames_dir.glob("*.png"))
    if not all_frames:
        raise FileNotFoundError("画面目录里没有 png：%s\n请先完成 ② 采集" % frames_dir)

    sprites = params.get("sprites") or ""
    if not Path(sprites).is_dir():
        raise FileNotFoundError("精灵库不存在：%s" % sprites)

    min_s = float(params.get("min_scale", 0.4))
    max_s = float(params.get("max_scale", 2.4))
    step = float(params.get("step", 0.1))
    sample = max(1, int(params.get("sample", 5)))
    max_mobs = int(params.get("max_mobs", 60))
    region = _parse_region(params.get("region"))

    if step <= 0 or max_s <= min_s:
        raise ValueError("扫描范围不对：下限 %.2f 上限 %.2f 步长 %.3f"
                         % (min_s, max_s, step))

    # 均匀采样几帧 —— 单帧可能正好没有目标，多帧能显著提高成功率
    n = min(sample, len(all_frames))
    if n == 1:
        frames = [all_frames[0]]
    else:
        idxs = [int(round(i * (len(all_frames) - 1) / (n - 1))) for i in range(n)]
        frames = [all_frames[i] for i in sorted(set(idxs))]

    # 候选模板
    mobs = [str(m).strip() for m in (params.get("mobs") or []) if str(m).strip()]
    cand = find_stand_frames(sprites, max_mobs)
    if mobs:
        want = set(mobs)
        cand = [c for c in cand if c[0] in want]
        if not cand:
            raise RuntimeError(
                "精灵库里找不到这些怪：%s\n"
                "（检查 config/wz.yaml 的 sprite_dir，以及怪种 ID 是否补足 7 位）"
                % ", ".join(mobs[:8]))

    tpls = []
    for mob_id, path in cand:
        t = _trim(imread(path, cv2.IMREAD_UNCHANGED))
        if t is not None:
            tpls.append((mob_id, t[0], t[1]))

    if not tpls:
        raise RuntimeError("没有可用模板 —— 精灵库里这些怪都没有 stand 帧")

    scales = []
    s = min_s
    while s <= max_s + 1e-9:
        scales.append(round(s, 4))
        s += step

    ctx.log("采样画面 %d 帧" % len(frames))
    ctx.log("候选怪种 %d 个（用 stand 首帧作模板）" % len(tpls))
    ctx.log("扫描尺度 %.2f ~ %.2f，步长 %.3f（%d 档）" % (min_s, max_s, step, len(scales)))
    if region:
        ctx.log("搜索区域 %s" % (region,))
    ctx.log("共 %d 次匹配" % (len(frames) * len(tpls) * len(scales)))
    ctx.log("")

    results = []
    total = len(frames) * len(tpls) * len(scales)
    done = 0
    t0 = time.perf_counter()

    for fi, fp in enumerate(frames):
        img = imread(fp)
        if img is None:
            ctx.log("读不到 %s" % fp.name, "warn")
            done += len(tpls) * len(scales)
            continue

        full = img
        ox = oy = 0
        if region:
            ox, oy, rw, rh = region
            img = full[oy:oy + rh, ox:ox + rw]

        for mob_id, b, al in tpls:
            for sc in scales:
                if ctx.canceled():
                    ctx.log("已取消", "warn")
                    return {"summary": "已取消"}

                h = max(8, int(round(b.shape[0] * sc)))
                w = max(8, int(round(b.shape[1] * sc)))
                if h >= img.shape[0] or w >= img.shape[1]:
                    done += 1
                    continue

                interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR
                t = cv2.resize(b, (w, h), interpolation=interp)
                m = (cv2.resize(al, (w, h), interpolation=interp) > 128)
                m = m.astype(np.uint8) * 255

                try:
                    r = cv2.matchTemplate(img, t, cv2.TM_CCORR_NORMED, mask=m)
                except Exception:
                    done += 1
                    continue

                r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                peak = float(r.max())
                # 区分度 = 峰值 − 高分位数。到处都是高分说明这个模板没有辨识力
                distinct = peak - float(np.percentile(r, 99.5))
                _, _, _, ml = cv2.minMaxLoc(r)

                results.append((distinct, peak, mob_id, sc,
                                (ml[0] + ox, ml[1] + oy), w, h, fp.name))
                done += 1

            if done % 40 == 0:
                ctx.progress(done, total, "已匹配 %d" % done)

        ctx.progress(done, total, "第 %d/%d 帧" % (fi + 1, len(frames)))
        ctx.log("  [%d/%d] %s  用时 %.0fs"
                % (fi + 1, len(frames), fp.name, time.perf_counter() - t0))

    if not results:
        raise RuntimeError("没有产生任何有效匹配 —— 检查画面对不对")

    results.sort(key=lambda r: -r[0])
    best = results[0]
    distinct, peak, mob_id, sc, loc, w, h, fname = best

    # 置信度判据：区分度够高 + 峰值附近的尺度集中。
    #
    # 注意只统计**同一只怪**的强候选 —— 不同怪的匹配质量差很多
    # （绿色怪在草地上就极易混淆，响应虽高但都是噪声），
    # 混在一起算尺度离散度会被噪声带偏，把成功判成失败。
    strong = [r for r in results if r[2] == mob_id and r[0] >= distinct * 0.4]
    spread = float(np.std([r[3] for r in strong])) if len(strong) > 1 else 0.0

    if distinct >= 0.05 and spread < 0.2:
        conf = "high"
    elif distinct >= 0.03:
        conf = "low"
    else:
        conf = "fail"

    top = results[:min(20, len(results))]

    ctx.log("")
    ctx.log("── 标定结果 ──", "ok" if conf == "high" else "warn")
    ctx.log("  最佳尺度 %.3f（区分度 %.4f，峰值 %.4f）" % (sc, distinct, peak))
    ctx.log("  来自 %s 的 %s，尺寸 %d×%d，位置 %s" % (fname, mob_id, w, h, loc))
    ctx.log("  同怪强候选 %d 个，尺度离散度 %.3f（阈值 0.2）" % (len(strong), spread))

    if conf == "high":
        ctx.log("  判定：可信", "ok")
    elif conf == "low":
        ctx.log("  判定：勉强可用 —— 建议增加画面里的目标数量后重标", "warn")
    else:
        ctx.log("  判定：不可信 —— 画面里可能没有这些怪，或尺度差异过大", "error")
        ctx.log("  提示：先用 ④ 自动标注跑几帧看看，确认画面里到底有什么", "warn")

    summary = "尺度 %.3f（%s）" % (
        sc, {"high": "可信", "low": "存疑", "fail": "失败"}[conf])

    ctx.progress(total, total, summary)
    return {
        "scale": float(sc),
        "distinct": float(distinct),
        "peak": float(peak),
        "mob": mob_id,
        "size": (int(w), int(h)),
        "frame": fname,
        "confidence": conf,
        "spread": spread,
        "strong": len(strong),
        "candidates": [
            {"distinct": float(r[0]), "peak": float(r[1]), "mob": r[2],
             "scale": float(r[3]), "pos": r[4], "size": (r[5], r[6]), "frame": r[7]}
            for r in top
        ],
        "summary": summary,
    }


def run_calibrate_player(params, ctx=None):
    """标定玩家 scale：玩家站姿模板多尺度匹配。

    和 run_calibrate 同一套原理，只是模板换成玩家站姿帧。
    params: frames, player_id, player_root, min_scale, max_scale, step, sample
    returns: {scale, confidence, summary, ...}
    """
    ctx = ctx or TaskContext()

    frames_dir = Path(params["frames"])
    all_frames = sorted(frames_dir.glob("*.png"))
    if not all_frames:
        raise FileNotFoundError("画面目录里没有 png：%s\n请先完成 ② 采集" % frames_dir)

    player_id = (params.get("player_id") or "").strip()
    if not player_id:
        raise ValueError("没有指定玩家 —— 请先在 ① 识别目标选项 里选角色")

    player_root = Path(params.get("player_root", "datasets/sprites/player"))
    pd = player_root / player_id
    if not pd.is_dir():
        raise FileNotFoundError("找不到玩家模板目录: %s" % pd)

    min_s = float(params.get("min_scale", 0.4))
    max_s = float(params.get("max_scale", 2.4))
    step = float(params.get("step", 0.1))
    sample = max(1, int(params.get("sample", 5)))

    tpls = []
    for f in sorted(pd.glob("*stand*.png")):
        t = _trim(imread(f, cv2.IMREAD_UNCHANGED))
        if t is not None:
            b, al = t
            tpls.append((f.stem, b, al))
            # 镜像一份 —— 模板可能都是朝左的，玩家朝右时也匹配得到
            tpls.append((f.stem + "@L", cv2.flip(b, 1), cv2.flip(al, 1)))
    if not tpls:
        raise RuntimeError("玩家模板目录里没有 stand 帧: %s" % pd)

    n = min(sample, len(all_frames))
    if n == 1:
        frames = [all_frames[0]]
    else:
        idxs = [int(round(i * (len(all_frames) - 1) / (n - 1))) for i in range(n)]
        frames = [all_frames[i] for i in sorted(set(idxs))]

    scales = []
    s = min_s
    while s <= max_s + 1e-9:
        scales.append(round(s, 4))
        s += step

    ctx.log("玩家 %s   模板 %d 个   扫 scale %.2f~%.2f（%d 档）"
            % (player_id, len(tpls), min_s, max_s, len(scales)))
    ctx.log("采样 %d 帧" % len(frames))
    ctx.log("")

    results = []
    total = len(frames) * len(tpls) * len(scales)
    done = 0
    t0 = time.perf_counter()

    for fi, fp in enumerate(frames):
        img = imread(fp)          # BGR，直接匹配（不要 cvtColor）
        if img is None:
            continue
        for name, b, al in tpls:
            for sc in scales:
                if ctx.canceled():
                    ctx.log("已取消", "warn")
                    return {"summary": "已取消"}
                h = max(8, int(round(b.shape[0] * sc)))
                w = max(8, int(round(b.shape[1] * sc)))
                if h >= img.shape[0] or w >= img.shape[1]:
                    done += 1
                    continue
                interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR
                t = cv2.resize(b, (w, h), interpolation=interp)
                m = (cv2.resize(al, (w, h), interpolation=interp) > 128)
                m = m.astype(np.uint8) * 255
                try:
                    r = cv2.matchTemplate(img, t, cv2.TM_CCORR_NORMED, mask=m)
                except Exception:
                    done += 1
                    continue
                r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                peak = float(r.max())
                distinct = peak - float(np.percentile(r, 99.5))
                _, _, _, ml = cv2.minMaxLoc(r)
                results.append((distinct, peak, name, sc,
                                (ml[0], ml[1]), w, h, fp.name))
                done += 1
            if done % 40 == 0:
                ctx.progress(done, total, "已匹配 %d" % done)
        ctx.progress(done, total, "第 %d/%d 帧" % (fi + 1, len(frames)))
        ctx.log("  [%d/%d] %s  用时 %.0fs"
                % (fi + 1, len(frames), fp.name, time.perf_counter() - t0))

    if not results:
        raise RuntimeError("没有产生任何有效匹配 —— 检查画面里有没有玩家")

    results.sort(key=lambda r: -r[0])
    distinct, peak, name, sc, loc, w, h, fname = results[0]

    strong = [r for r in results if r[0] >= distinct * 0.4]
    spread = float(np.std([r[3] for r in strong])) if len(strong) > 1 else 0.0

    if distinct >= 0.05 and spread < 0.2:
        conf = "high"
    elif distinct >= 0.03:
        conf = "low"
    else:
        conf = "fail"

    ctx.log("")
    ctx.log("── 玩家标定结果 ──", "ok" if conf == "high" else "warn")
    ctx.log("  最佳尺度 %.3f（区分度 %.4f，峰值 %.4f）" % (sc, distinct, peak))
    ctx.log("  来自 %s 的 %s，尺寸 %d×%d，位置 %s" % (fname, name, w, h, loc))
    if conf == "high":
        ctx.log("  判定：可信", "ok")
    elif conf == "low":
        ctx.log("  判定：勉强可用", "warn")
    else:
        ctx.log("  判定：不可信 —— 画面里可能没有玩家，或姿态不符", "error")

    summary = "玩家尺度 %.3f（%s）" % (
        sc, {"high": "可信", "low": "存疑", "fail": "失败"}[conf])
    ctx.progress(total, total, summary)
    return {
        "scale": float(sc),
        "confidence": conf,
        "distinct": float(distinct),
        "peak": float(peak),
        "player": player_id,
        "summary": summary,
    }


def run_calibrate_combined(params, ctx=None):
    """标定怪物 + 玩家两个 scale，一次任务跑完。"""
    mob_res = run_calibrate(params["mob"], ctx)

    ctx.log("")
    ctx.log("—— 玩家尺度标定 ——", "info")
    player_res = None
    player_err = None
    try:
        player_res = run_calibrate_player(params["player"], ctx)
    except Exception as e:
        player_err = "%s: %s" % (type(e).__name__, e)
        ctx.log("玩家标定失败（怪物标定结果已保留）：%s" % player_err, "warn")

    summary = mob_res.get("summary", "")
    if player_res is not None:
        summary += "  ·  " + player_res.get("summary", "")
    elif player_err:
        summary += "  ·  玩家标定失败"

    return {
        "mob": mob_res,
        "player": player_res,
        "player_error": player_err,
        "summary": summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="画面目录")
    ap.add_argument("--sprites", required=True)
    ap.add_argument("--mobs", default="", help="逗号分隔的怪种 id")
    ap.add_argument("--min-scale", type=float, default=0.4)
    ap.add_argument("--max-scale", type=float, default=2.4)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--max-mobs", type=int, default=60)
    ap.add_argument("--region", default=None, help="x,y,w,h")
    a = ap.parse_args()

    params = {
        "frames": a.frames,
        "sprites": a.sprites,
        "mobs": [m.strip() for m in a.mobs.split(",") if m.strip()],
        "min_scale": a.min_scale,
        "max_scale": a.max_scale,
        "step": a.step,
        "sample": a.sample,
        "max_mobs": a.max_mobs,
        "region": a.region,
    }

    try:
        run_calibrate(params, ConsoleContext())
    except Exception as e:
        print("[calib] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
