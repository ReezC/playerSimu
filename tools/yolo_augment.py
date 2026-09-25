"""YOLO 辅助自动标注：用其它项目已训练的模型预测，把预测框合并进已有标注。

**为什么需要**
    模板匹配的召回被「姿态必须和模板一致」卡死，漏掉不少目标。
    其它地图训好的 YOLO 模型学的是外观特征，能补上这些漏检。
    这里**不重写**标注，只把模型预测的框追加进去：模板匹配的高精度框
    原样保留，模型只补「和已有框不重叠」的那部分。

**怪物 vs 玩家**
    · 怪物（class 1）：语义统一（就是「怪」），任何项目的模型都能辅助。
    · 玩家（class 0）：不同角色是不同的玩家，**只有模型所属项目的玩家角色
      和当前项目一致时**才辅助玩家，否则会把别的角色误标成当前玩家。

CLI:
    python -m tools.yolo_augment --weights projects/x/models/detect_v1.pt \\
        --frames projects/y/frames --out projects/y/labels_auto \\
        --player-id 珠缨
"""

import argparse
import json
import sys
import time
from pathlib import Path

from core.context import ConsoleContext, TaskContext

CLASS_MOB = 1       # 类别表见 perception/classes.py（id 固定，别在本地另立一份）
CLASS_PLAYER = 0
OVERLAP_THR = 0.3     # 和已有框的 IoU 超过它就认为重复，丢弃


def _project_player_id(proj_dir):
    """读项目 project.yaml 里的 player_id，读不到返回空串。"""
    yaml_path = Path(proj_dir) / "project.yaml"
    if not yaml_path.exists():
        return ""
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return (data.get("player_id") or "").strip()
    except Exception:
        return ""


def list_weights(projects_root=None):
    """列出所有项目里训练好的权重，按修改时间倒序。

    返回 [(label, path, player_id)]，player_id 来自该权重所属项目，
    供「辅助玩家时按角色过滤」用。
    """
    root = Path(projects_root) if projects_root \
        else Path(__file__).resolve().parent.parent / "projects"
    out = []
    if not root.is_dir():
        return out

    def _mtime(p):
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        pid = _project_player_id(d)
        models = sorted((d / "models").glob("*.pt"), key=_mtime, reverse=True)
        if models:
            # 有归档就只列归档 —— models/{name}.pt 就是 runs 里 best.pt 的拷贝，
            # 再列 runs 会把同一份模型重复算一遍。
            for p in models:
                out.append(("%s / %s" % (d.name, p.name), str(p), pid))
            continue
        # models/ 空时才退回 runs/**/weights/best.pt（和 GUI 查找逻辑一致）
        for p in sorted((d / "runs").glob("**/weights/best.pt"),
                        key=_mtime, reverse=True):
            out.append(("%s / runs/%s" % (d.name, p.parent.parent.name),
                        str(p), pid))

    out.sort(key=lambda x: _mtime(Path(x[1])), reverse=True)
    return out


def _read_boxes(path):
    """读一个 YOLO 标注文件，返回 [(cls, cx, cy, w, h)]；读不到返回 []。"""
    if not path.exists():
        return []
    out = []
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) >= 5:
                try:
                    out.append((int(float(p[0])), float(p[1]), float(p[2]),
                                float(p[3]), float(p[4])))
                except ValueError:
                    continue
    except Exception:
        pass
    return out


def _overlap(cx, cy, w, h, boxes, thr=OVERLAP_THR):
    """判断 (cx,cy,w,h) 是否和 boxes 里任一个框 IoU 超过 thr。"""
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    area = w * h
    if area <= 0:
        return False
    for _c, bcx, bcy, bw, bh in boxes:
        bx1, by1, bx2, by2 = bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2
        ix = max(0.0, min(x2, bx2) - max(x1, bx1))
        iy = max(0.0, min(y2, by2) - max(y1, by1))
        inter = ix * iy
        if inter <= 0:
            continue
        union = area + bw * bh - inter
        if union > 0 and inter / union > thr:
            return True
    return False


def _roi_from_box(x1, y1, x2, y2, shape, margin=0.5, min_side=64.0):
    """玩家框 → 模板匹配的搜索区域（自然外扩一圈再裁到图内）。

    margin 是相对框**长边**的外扩比例。**必须留足余量**：`player_locator` 会跳过
    「比搜索区还大」的模板（`perception/player_locator.py:122`），区域贴着模板大小
    时稍有偏差就整帧白跑 —— 那等于白丢召回。

    shape: (h, w)。返回 (x, y, w, h)；框无效/太小返回 None。
    """
    h, w = int(shape[0]), int(shape[1])
    bw, bh = float(x2) - float(x1), float(y2) - float(y1)
    if bw <= 0 or bh <= 0 or w <= 0 or h <= 0:
        return None
    m = max(min_side / 2.0, margin * max(bw, bh))
    rx1 = int(max(0.0, float(x1) - m))
    ry1 = int(max(0.0, float(y1) - m))
    rx2 = int(min(float(w), float(x2) + m))
    ry2 = int(min(float(h), float(y2) + m))
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return None
    return (rx1, ry1, rx2 - rx1, ry2 - ry1)


def player_rois(frames, weights, player_id="", weights_player_id="",
                conf=0.25, imgsz=960, device=None, margin=0.5, ctx=None):
    """跑一遍 YOLO，得到**每帧玩家框所在的搜索区域** → `{帧stem: (x, y, w, h)}`。

    **为什么**：玩家标注的成本全在模板匹配上（102 个模板 × 整幅图，实测约
    11 秒/帧），而同一批帧 YOLO 只要约 0.1 秒/帧（批推理 + GPU + imgsz 缩小）。
    先用 YOLO 圈出玩家可能在的地方，模板匹配只在圈里找。

    **没检出的帧不进返回表**，调用方对缺项按「全图匹配」处理 —— 这是刻意设计：
    YOLO 漏检（玩家在角落、被怪挡住、姿态罕见）时那一帧仍走原来的全图路径，
    **召回率与不优化时完全一致**。优化只动速度，不动结果。
    角色不一致的模型直接返回空表（圈出来的区域没有意义），同样退回全图。
    """
    if player_id and weights_player_id and player_id != weights_player_id:
        if ctx:
            ctx.log("模型角色「%s」≠ 当前角色「%s」，跳过 YOLO 粗定位"
                    % (weights_player_id, player_id), "warn")
        return {}

    try:
        from ultralytics import YOLO
    except ImportError:
        if ctx:
            ctx.log("没有装 ultralytics，跳过 YOLO 粗定位", "warn")
        return {}

    if ctx:
        ctx.progress(0, 0, "加载模型（粗定位）…")
    model = YOLO(str(weights))

    out = {}
    stream = model.predict(source=[str(f) for f in frames], conf=conf,
                           iou=0.5, imgsz=imgsz, device=device,
                           verbose=False, stream=True)
    for r in stream:
        if ctx is not None and ctx.canceled():
            if ctx:
                ctx.log("已取消，粗定位中断（剩下的帧按全图匹配）", "warn")
            break
        path = Path(getattr(r, "path", "") or "")
        if not path.stem:
            continue
        boxes = getattr(r, "boxes", None)
        shape = getattr(r, "orig_shape", None)
        if boxes is None or not len(boxes) or not shape:
            continue
        cls = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        best = None
        for c, cf, box in zip(cls, confs, xyxy):
            if int(c) != CLASS_PLAYER:
                continue
            if best is None or float(cf) > best[0]:
                best = (float(cf), float(box[0]), float(box[1]),
                        float(box[2]), float(box[3]))
        if best is None:
            continue
        roi = _roi_from_box(best[1], best[2], best[3], best[4], shape, margin)
        if roi:
            out[path.stem] = roi
    return out


def run_yolo_augment(params, ctx=None):
    """用已有模型预测，把预测框合并进已有标注。

    params:
        weights           权重文件
        frames            画面目录
        out               标注目录（labels_auto，已有模板匹配标注）
        mode              "mob" 只辅助怪物 / "player" 只辅助玩家（默认 mob）
        player_id         当前项目的玩家角色（mode=player 时用）
        weights_player_id 权重所属项目的玩家角色（mode=player 时要求 == player_id）
        conf              置信度阈值（默认 0.40，比模板匹配宽松以补召回）
        iou               NMS 的 IoU 阈值
        imgsz             推理尺寸
        device            设备
        limit             只跑前 N 张，0 = 全部

    返回 dict，卡片读它渲染摘要。
    """
    ctx = ctx or TaskContext()

    weights = Path(params.get("weights") or "")
    if not weights.exists():
        raise FileNotFoundError("权重不存在：%s" % weights)

    frames_dir = Path(params.get("frames") or "")
    if not frames_dir.is_dir():
        raise FileNotFoundError("画面目录不存在：%s" % frames_dir)

    frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError("画面目录里没有 png/jpg：%s" % frames_dir)

    limit = int(params.get("limit", 0) or 0)
    if limit:
        frames = frames[:limit]

    # 只辅助选中的帧（和模板匹配那一趟用同一份清单）—— 不然用户只勾了几帧，
    # 辅助却把没勾的帧也改了标注，等于白选。
    only = {str(s) for s in (params.get("only") or [])}
    if only:
        frames = [f for f in frames if f.stem in only]
        if not frames:
            raise ValueError("选中的帧在画面目录里一个都找不到（选了 %d 个）"
                             % len(only))

    out = Path(params["out"])
    out.mkdir(parents=True, exist_ok=True)

    conf = float(params.get("conf", 0.40))
    iou = float(params.get("iou", 0.45))
    imgsz = int(params.get("imgsz", 960))
    device = str(params.get("device", "0"))

    mode = params.get("mode", "mob")   # "mob" 只辅助怪物 / "player" 只辅助玩家
    player_id = (params.get("player_id") or "").strip()
    weights_player_id = (params.get("weights_player_id") or "").strip()

    ctx.log("权重       %s" % weights)
    ctx.log("画面       %s（%d 张）" % (frames_dir, len(frames)))
    if mode == "player":
        if not (player_id and weights_player_id == player_id):
            raise ValueError(
                "模型角色「%s」≠ 当前角色「%s」，不能用它辅助玩家 —— 换一个角色一致的模型"
                % (weights_player_id or "（无）", player_id or "（无）"))
        allowed = {CLASS_PLAYER}
        ctx.log("置信度 %.2f   只合并 class 0（玩家，角色已匹配）" % conf)
    else:
        allowed = {CLASS_MOB}
        ctx.log("置信度 %.2f   只合并 class 1（怪物）" % conf)
    ctx.log("输出       %s" % out)
    ctx.log("")

    # 轮到这个阶段时任务可能已经被取消了 —— 别为一个已取消的任务去加载模型：
    # YOLO(...) 首次加载要几秒到几十秒（还要初始化 CUDA），这一下拦不住。
    if ctx.canceled():
        ctx.log("已取消，跳过 YOLO 辅助", "warn")
        return {"frames": 0, "boxes_added": 0, "frames_merged": 0, "mode": mode,
                "seconds": 0.0, "weights": str(weights), "summary": "已取消"}

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("没有装 ultralytics。\n请执行：pip install ultralytics")

    ctx.progress(0, 0, "加载模型…")
    model = YOLO(str(weights))
    ctx.progress(0, len(frames), "开始推理")

    t0 = time.perf_counter()
    n_merged = 0      # 有多少帧补了框
    n_boxes = 0       # 共补了多少框
    n = len(frames)

    stream = model.predict(
        source=[str(f) for f in frames],
        conf=conf, iou=iou, imgsz=imgsz, device=device,
        verbose=False, stream=True,
    )

    for i, r in enumerate(stream, 1):
        if ctx.canceled():
            ctx.log("已取消", "warn")
            break

        path = Path(getattr(r, "path", "") or "")
        stem = path.stem
        if not stem:
            continue

        target = out / (stem + ".txt")
        existing = _read_boxes(target)

        preds = []
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes):
            cls = boxes.cls.cpu().numpy()
            xywhn = boxes.xywhn.cpu().numpy()
            for c, (cx, cy, w, h) in zip(cls, xywhn):
                if int(c) in allowed:
                    preds.append((int(c), float(cx), float(cy),
                                  float(w), float(h)))

        added = 0
        for pc, pcx, pcy, pw, ph in preds:
            if _overlap(pcx, pcy, pw, ph, existing):
                continue
            existing.append((pc, pcx, pcy, pw, ph))
            added += 1
            n_boxes += 1

        if added:
            lines = ["%d %.6f %.6f %.6f %.6f" % t for t in existing]
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_merged += 1

        if i % 20 == 0 or i == n:
            ctx.progress(i, n, "已补 %d 框" % n_boxes)
            ctx.log("  [%d/%d]  补 %d  用时 %.0fs"
                    % (i, n, n_boxes, time.perf_counter() - t0))

    dt = time.perf_counter() - t0

    ctx.log("")
    ctx.log("── YOLO 辅助标注完成 ──", "ok")
    ctx.log("  画面 %d 张 / 补充 %d 框 / 涉及 %d 帧" % (n, n_boxes, n_merged))
    ctx.log("  用时 %.1fs" % dt)
    if n_boxes == 0:
        ctx.log("  一个框都没补 —— 模型对这些画面不适应，或画面里本就没有目标",
                "warn")
    ctx.log("")

    return {
        "frames": n,
        "boxes_added": n_boxes,
        "frames_merged": n_merged,
        "mode": mode,
        "seconds": dt,
        "weights": str(weights),
        "summary": "补充 %d 框 / %d 帧（%s）"
                   % (n_boxes, n_merged,
                      "玩家" if mode == "player" else "怪物"),
    }


def run_detect_mob_augmented(params, ctx=None):
    """怪物标注：模板匹配 + 可选 YOLO 辅助。

    params:
        detect   怪物模板匹配参数（透传给 run_detect）
        augment  可选，YOLO 辅助参数（透传给 run_yolo_augment，mode=mob）
    """
    from tools.detect_mobs import run_detect
    ctx = ctx or TaskContext()
    res = run_detect(params["detect"], ctx)
    aug = params.get("augment")
    if aug:
        # **取消最容易被当成失效的地方**：模板匹配那步已经被取消了，这里却接着跑
        # YOLO —— 用户点完取消还要再等模型加载 + 第一帧推理，看起来就是没反应。
        if ctx.canceled():
            ctx.log("已取消，跳过 YOLO 辅助", "warn")
            return res
        ctx.log("")
        ctx.log("—— YOLO 辅助怪物 ——", "info")
        res["augment"] = run_yolo_augment(aug, ctx)
    return res


def run_detect_player_augmented(params, ctx=None):
    """玩家标注：模板匹配 + 可选 YOLO 辅助。

    params:
        detect   玩家模板匹配参数（透传给 run_detect_player）
        augment  可选，YOLO 辅助参数（透传给 run_yolo_augment，mode=player）
    """
    from tools.detect_player import run_detect_player
    ctx = ctx or TaskContext()
    res = run_detect_player(params["detect"], ctx)
    aug = params.get("augment")
    if aug:
        if ctx.canceled():       # 同 run_detect_mob_augmented：别让取消白等一轮 YOLO
            ctx.log("已取消，跳过 YOLO 辅助", "warn")
            return res
        ctx.log("")
        ctx.log("—— YOLO 辅助玩家 ——", "info")
        res["augment"] = run_yolo_augment(aug, ctx)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="mob", choices=["mob", "player"])
    ap.add_argument("--player-id", default="")
    ap.add_argument("--weights-player-id", default="")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    try:
        run_yolo_augment({
            "weights": a.weights,
            "frames": a.frames,
            "out": a.out,
            "mode": a.mode,
            "player_id": a.player_id,
            "weights_player_id": a.weights_player_id,
            "conf": a.conf,
            "imgsz": a.imgsz,
            "device": a.device,
            "limit": a.limit,
        }, ConsoleContext())
    except Exception as e:
        print("[yolo_augment] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
