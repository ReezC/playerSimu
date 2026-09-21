"""玩家自动标注（class 0）：模板匹配定位玩家，追加到已有标注。

和怪物标注（detect_mobs.py）的区别：
    - 怪物：多怪种、多模板、多尺度、NMS —— 画面里有多只怪
    - 玩家：单目标、单套站姿模板 —— 画面里只有一个玩家

所以玩家标注简单得多：每帧用 PlayerLocator 定位一次，取最高分命中。
输出**追加**到 labels_auto 的怪物标注后面（LabelCard 先跑怪物、再跑玩家）。

**必须全图匹配**：角色可能出现在画面角落，不能套「中央区域」假设。
全图 stand 模板约 1s/帧，靠多进程并行加速（离线标注，慢点无所谓）。

CLI:
    python -m tools.detect_player --player 珠缨 --frames data/frames ^
        --out projects/x/labels_auto --scale 1.0
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2

from core.context import ConsoleContext, TaskContext
from core.imgio import imread, imwrite

CLASS_PLAYER = 0
_W = {}


def init_worker(cfg):
    from perception.player_locator import PlayerLocator
    cv2.setNumThreads(1)
    _W["cfg"] = cfg
    _W["loc"] = PlayerLocator(
        cfg["player_id"], root=cfg["player_root"], scale=cfg["scale"],
        threshold=cfg["thresh"], filter_prefix=cfg.get("filter_prefix"),
        frame_scales=cfg.get("frame_scales", {}))


def work(fp_str):
    cfg = _W["cfg"]
    fp = Path(fp_str)

    out = Path(cfg["out"]) / (fp.stem + ".txt")

    full = imread(fp)
    if full is None:
        return None

    h, w = full.shape[:2]
    # imread 已经返回 BGR（cv2.imdecode），不要再 cvtColor ——
    # 之前多转了一次 RGB2BGR，把橙发转成蓝发，模板在颜色错乱的图上匹配不上。
    r = _W["loc"].locate(full)         # 全图匹配（玩家可能在角落）

    # 玩家框（class 0）一律重写，不做「断点续标」：重新采集后旧 txt 里
    # 残留的 class 0 框位置是错的，若按「已有 class 0 就跳过」会整批跳过，
    # 造成「命中 0 帧」的假象。怪物框（class 1）原样保留。
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    keep = [ln for ln in existing.splitlines()
            if ln.split() and ln.split()[0] != str(CLASS_PLAYER)]

    score = None
    if r is not None:
        cx, cy, bw, bh, score, _name = r
        keep.append("%d %.6f %.6f %.6f %.6f" % (
            CLASS_PLAYER, cx / w, cy / h, bw / w, bh / h))

    out.write_text(("\n".join(keep) + "\n") if keep else "\n", encoding="utf-8")

    if cfg["vis"]:
        if r is not None:
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            cv2.rectangle(full, (x1, y1), (x1 + int(bw), y1 + int(bh)),
                          (255, 128, 0), 2)   # 蓝色（BGR），区别于怪物的绿
        # 没命中也重画一帧（无框原图），覆盖旧的可视化图
        imwrite(Path(cfg["vis_dir"]) / (fp.stem + ".jpg"), full, quality=82)

    return score


def run_detect_player(params, ctx=None):
    """玩家自动标注，追加到已有标注。

    params:
        frames       画面目录
        out          标注输出目录（labels_auto，已有怪物标注）
        player_id    玩家 id（datasets/sprites/player/ 下的目录名）
        player_root  玩家模板根目录（默认 datasets/sprites/player）
        scale        玩家匹配 scale
        thresh       匹配阈值
        filter_prefix 只加载该动作前缀的模板（默认 stand）
        vis / vis_dir
        workers      进程数，0=自动
    """
    ctx = ctx or TaskContext()

    frames_dir = Path(params["frames"])
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError("画面目录里没有 png：%s" % frames_dir)

    pid = (params.get("player_id") or "").strip()
    if not pid:
        raise ValueError("没有指定玩家 —— 请先在 ① 识别目标选项 里选角色")

    out = Path(params["out"])
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "player_id": pid,
        "player_root": params.get("player_root", "datasets/sprites/player"),
        "scale": float(params.get("scale", 1.0)),
        "thresh": float(params.get("thresh", 0.78)),
        "filter_prefix": params.get("filter_prefix"),   # None = 加载所有动作帧
        "frame_scales": params.get("frame_scales") or {},  # {player_id:帧stem -> scale}
        "out": str(out),
        "vis": bool(params.get("vis", True)),
        "vis_dir": params.get("vis_dir") or str(out.parent / "vis"),
    }

    # 重新采集后帧数变少时，清掉帧号超界的旧标注/可视化（同 detect_mobs）
    from tools.detect_mobs import clean_stale_outputs
    removed = clean_stale_outputs(str(out), cfg["vis_dir"], len(frames))
    if removed:
        ctx.log("清理上次残留 %d 个文件（帧号超出本次 %d 帧）" % (removed, len(frames)))

    # 主进程先验一次模板能不能加载，避免开了几十个进程才发现模板空
    ctx.progress(0, 0, "加载玩家模板…")
    from perception.player_locator import PlayerLocator
    loc = PlayerLocator(pid, root=cfg["player_root"], scale=cfg["scale"],
                        threshold=cfg["thresh"],
                        filter_prefix=cfg["filter_prefix"],
                        frame_scales=cfg["frame_scales"])
    if loc.template_count == 0:
        raise RuntimeError("玩家模板为空：%s/%s" % (cfg["player_root"], pid))

    ctx.log("玩家 %s   模板 %d 个   scale %.3f   阈值 %.2f"
            % (pid, loc.template_count, cfg["scale"], cfg["thresh"]))
    ctx.log("全图匹配（玩家可能在角落），%d 帧" % len(frames))

    workers = int(params.get("workers", 0)) or max(1, (os.cpu_count() or 4) - 1)
    ctx.log("并行进程 %d" % workers)
    ctx.log("")

    t0 = time.perf_counter()
    hit = 0
    n = len(frames)

    def consume(iterator):
        nonlocal hit
        for i, r in enumerate(iterator, 1):
            if ctx.canceled():
                return False
            if r is not None:
                hit += 1
            if i % 20 == 0 or i == n:
                ctx.progress(i, n, "已定位玩家 %d 帧" % hit)
                ctx.log("  [%d/%d]  命中 %d  用时 %.0fs"
                        % (i, n, hit, time.perf_counter() - t0))
        return True

    try:
        if workers <= 1:
            init_worker(cfg)
            ok = consume(work(str(f)) for f in frames)
        else:
            pool = mp.Pool(workers, initializer=init_worker, initargs=(cfg,))
            try:
                ok = consume(pool.imap(work, [str(f) for f in frames], chunksize=2))
            except Exception:
                pool.terminate()   # 出错也立即终止子进程
                raise
            if ok:
                pool.close()
            else:
                pool.terminate()   # 取消：立即终止，不等正在跑的帧
            pool.join()
    except Exception as e:
        raise RuntimeError("玩家标注失败: %s: %s" % (type(e).__name__, e))

    dt = time.perf_counter() - t0

    if not ok:
        ctx.log("已取消", "warn")
        return {"summary": "已取消", "frames": n, "hits": hit}

    ctx.log("")
    ctx.log("── 玩家标注完成 ──", "ok")
    ctx.log("  画面 %d 张 / 命中玩家 %d 帧（%.0f%%）"
            % (n, hit, 100.0 * hit / max(1, n)))
    ctx.log("  用时 %.0fs" % dt)
    if hit == 0:
        ctx.log("一帧都没命中玩家 —— 检查 scale/阈值，或画面里没有玩家", "warn")
    elif hit < n * 0.5:
        ctx.log("命中率偏低（%d/%d）—— 玩家在角落/被遮挡时会漏，"
                "质检台可用「玩家框丢失」筛选补" % (hit, n), "warn")
    ctx.log("")

    return {
        "frames": n,
        "hits": hit,
        "seconds": dt,
        "summary": "%d 帧 / 命中 %d（%.0f%%）" % (n, hit, 100.0 * hit / max(1, n)),
    }


class _OffsetCtx:
    """把子任务的进度映射到「帧 × 类」的统一进度条。

    自动标注 = 怪物 + 玩家两段，共 N 帧 × 2 类 = 2N 个单元。
    怪物段占 [0, N]，玩家段占 [N, 2N]，每完成一帧的一类 = 完成本帧的一半。
    """

    def __init__(self, ctx, offset, unit_total):
        self._ctx = ctx
        self._offset = offset
        self._unit_total = unit_total

    def log(self, msg, level="info"):
        self._ctx.log(msg, level)

    def canceled(self):
        return self._ctx.canceled()

    def progress(self, cur, total=0, text=""):
        if total > 0:
            self._ctx.progress(self._offset + int(cur), self._unit_total, text)
        else:
            self._ctx.progress(0, 0, text)


def run_detect_combined(params, ctx=None):
    """自动标注：怪物 + 玩家，一次任务跑完两类。

    params:
        mob    怪物标注参数（透传给 run_detect）
        player 玩家标注参数（透传给 run_detect_player）

    顺序：先怪物（写 labels_auto），再玩家（追加玩家框）——
    玩家标注读已有 txt 追加，所以必须先跑怪物。
    """
    from tools.detect_mobs import run_detect

    ctx = ctx or TaskContext()

    frames_dir = Path(params["mob"]["frames"])
    n_frames = len(sorted(frames_dir.glob("*.png")))
    unit_total = 2 * max(1, n_frames)

    ctx.log("—— 第 1 步：怪物标注 ——", "info")
    mob_res = run_detect(params["mob"], _OffsetCtx(ctx, 0, unit_total))

    ctx.log("")
    ctx.log("—— 第 2 步：玩家标注 ——", "info")
    player_res = None
    player_err = None
    try:
        player_res = run_detect_player(params["player"],
                                       _OffsetCtx(ctx, n_frames, unit_total))
    except Exception as e:
        # 玩家标注失败不要整体抛错 —— 怪物标注已经完成并落盘，
        # 抛错会把整个任务标成失败，让人误以为怪物也白跑了。
        player_err = "%s: %s" % (type(e).__name__, e)
        ctx.log("玩家标注失败（怪物标注结果已保留，可直接重跑补齐）：%s"
                % player_err, "warn")

    summary = mob_res.get("summary", "")
    if player_res is not None:
        summary += "  ·  玩家 %s" % player_res.get("summary", "")
    elif player_err:
        summary += "  ·  玩家标注失败"

    return {
        "summary": summary,
        "mob": mob_res,
        "player": player_res,
        "player_error": player_err,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--player-root", default="datasets/sprites/player")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--thresh", type=float, default=0.78)
    ap.add_argument("--filter-prefix", default="stand")
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--vis-dir", default=None)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    try:
        run_detect_player({
            "frames": a.frames,
            "out": a.out,
            "player_id": a.player,
            "player_root": a.player_root,
            "scale": a.scale,
            "thresh": a.thresh,
            "filter_prefix": a.filter_prefix,
            "vis": a.vis,
            "vis_dir": a.vis_dir,
            "workers": a.workers,
        }, ConsoleContext())
    except Exception as e:
        print("[detect_player] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
