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
#: ⭐ 掉落物（用户 2026-10-04 ✓）：**照 `perception/classes.py` 导入**（别在本地另立一份 ✗ ——
#:   上面那两个是历史遗留的本地副本 ✓ 新的一律从类别表拿 ✓）。
from perception.classes import CLASS_DROP, CLASS_PET               # noqa: E402
#: 每个 mode 只合并**这一类**，`summary` 里用哪个中文名 —— **一处口径** ✓
#: （加一类只改这张表 ✓ 别在下面各处再写 `if mode == ...` ✗）。
MODE_CLASS = {"mob": CLASS_MOB, "player": CLASS_PLAYER, "drop": CLASS_DROP,
              "pet": CLASS_PET}
MODE_ZH = {"mob": "怪物", "player": "玩家", "drop": "掉落物", "pet": "宠物"}
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


#: ⭐⭐ **这一支的版本号**（用户 2026-10-10 ✓ 现场逼出来的）：`run_yolo_augment` 是**在任务里
#:   才 `import` 的** ⇒ 工作台进程一旦**早就启动过**、跑过一次，`sys.modules` 里就**一直是旧的那份**
#:   ✗ —— 于是"我明明改了，跑出来还是老行为"✓（2026-10-10 现场：跑了三趟，记录里都没有新加的
#:   `first_inputs` 字段 ⇒ 说明那颗进程里是旧模块 ✓ 而人以为"已经是最新的"✗）。
#:   ⇒ 每次跑都在日志第一行与记录里写下它 ✓：**看到 `版本` 与代码里一致 = 真的是新代码** ✓。
AUGMENT_BUILD = "2026-10-10b"


def _stem_of(path, frame_map):
    """`r.path` → **画面目录里那一帧的 stem**；对不上 ⇒ `None`（调用方跳过 ✓）。

    ⭐⭐ **为什么不能直接用 `path.stem`**（2026-10-09 现场 ✓）：`run_yolo_augment` 是**通用**工具
    ✓ —— 谁都能用 `--frames <别的目录>` 跑它（CLI / 别的卡片 ✓，本函数的参数就叫 `frames` ✓）。
    那种情况下"喂进来的图叫什么、标注就叫什么" ✗ ⇒ 写出一堆**对不上任何画面**的标注 ✗：
    实测（鳄鱼潭1）`--frames` 指到了另一个目录（那批图叫 `image0.jpg…image178.jpg` ✓）
    ⇒ `labels_auto/` 里多出 **163 个 `image0.txt…image178.txt`** ✗ 而画面叫
    `frame_00000.png…` ✓。
    后果很安静也很坏 ✓：**质检台按帧名找标注** ⇒ 那些帧看到的还是**旧**标注 ✗
    （用户报的"YOLO 标注了 frame_179 往后、质检台打开看不到新的框"✓ 就是它 ✓）；
    数据集那边同样读不到 ✓。
    ⇒ 一律**映射回画面目录里的帧名** ✓；对不上就**跳过**（并汇总报出来 ✓ 不静默 ✗）。
    """
    if not path:
        return None
    try:
        key = Path(path).resolve()
    except Exception:                        # noqa: BLE001
        return None
    stem = frame_map.get(key)
    if stem:
        return stem
    # 兜底：同一个**文件名**（软链 / 相对路径写法不同 ⇒ 解析出来不一样 ✓）
    name = Path(path).name
    for p, s in frame_map.items():
        if p.name == name:
            return s
    return None


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
        mode              "mob" 只辅助怪物 / "player" 只辅助玩家 / "drop" 只补掉落物
                          （默认 mob；类别号见 `MODE_CLASS` ✓ 一处口径 ✓）
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

    # ⭐⭐ **输出名一律走"画面目录里的帧名"**（2026-10-09 ✓ 见 `_stem_of` 的说明）：
    #   对不上的图**跳过**，跑完汇总报出来 —— 绝不把标注写到"没人认的名字"上 ✗。
    frame_map = {f.resolve(): f.stem for f in frames}
    skipped = []
    #: 有多少条是**靠"顺序"**对回帧名的（= ultralytics 把回报名写成 `imageN.jpg` 的那些 ✓
    #: 见下面循环里那段说明 ✓）。正常情况它 == 总条数 ✓，说明"顺序映射"这条路在干活 ✓。
    order_fixed = 0
    #: ⭐ 头几条"**模型回报的名字**"（与"喂入"那行对照 ✓ 见下面 ✓）：2026-10-09 那次
    #:   "目录记对了、喂进去的却是别的名字"的局面，就是靠这两行才分得清 ✗✓。
    seen_paths = []

    conf = float(params.get("conf", 0.40))
    iou = float(params.get("iou", 0.45))
    imgsz = int(params.get("imgsz", 960))
    device = str(params.get("device", "0"))

    mode = params.get("mode", "mob")   # "mob" 只辅助怪物 / "player" 只辅助玩家
    player_id = (params.get("player_id") or "").strip()
    weights_player_id = (params.get("weights_player_id") or "").strip()

    # ⭐⭐ 第一行就报版本 ✓：让人**当场**就能确认"跑的是不是最新代码" ✗✓（见 `AUGMENT_BUILD` ✓）
    ctx.log("版本       %s（yolo_augment）" % AUGMENT_BUILD)
    ctx.log("权重       %s" % weights)
    ctx.log("画面       %s（%d 张）" % (frames_dir, len(frames)))
    if mode == "player":
        if not (player_id and weights_player_id == player_id):
            raise ValueError(
                "模型角色「%s」≠ 当前角色「%s」，不能用它辅助玩家 —— 换一个角色一致的模型"
                % (weights_player_id or "（无）", player_id or "（无）"))
        allowed = {CLASS_PLAYER}
        ctx.log("置信度 %.2f   只合并 class 0（玩家，角色已匹配）" % conf)
    elif mode == "drop":
        # ⭐ 掉落物（用户 2026-10-04 ✓）：与怪物同一套"只补不覆盖"的合并逻辑 ✓，
        #   只是类别号不同（class 2 ✓）—— 模型吐 class 2 的框才要 ✓。
        allowed = {CLASS_DROP}
        ctx.log("置信度 %.2f   只合并 class %d（掉落物）" % (conf, CLASS_DROP))
    elif mode == "pet":
        # ⭐ 宠物（用户 2026-10-04 ✓ 方案 C ✓）：同上，类别号 **class 5（宠物）** ✓
        #   —— 它是"干扰类"（标它只为让模型学会区分宠物 vs 怪 ✓ 见 `perception/classes.py` ✓）。
        allowed = {CLASS_PET}
        ctx.log("置信度 %.2f   只合并 class %d（宠物）" % (conf, CLASS_PET))
    else:
        allowed = {CLASS_MOB}
        ctx.log("置信度 %.2f   只合并 class 1（怪物）" % conf)
    ctx.log("输出       %s" % out)
    # ⭐⭐ **"喂给模型的到底是哪些图"要留证据**（2026-10-09 ✓）：这次现场出现了一种**对不上**的
    #   局面 —— `last_augment.json` 里 `frames_dir` 记的是**正确的**项目画面目录（359 张
    #   `frame_*.png` ✓），可这趟实际喂进去的是 **180 张 `imageN.jpg`** ✗。而"枚举源头"和
    #   "预测源"都是同一份 `frames` 列表 ✓ ⇒ 只报"目录 + 条数"已经分不清了 ✗ ⇒ 直接把**实际的
    #   前几个名字**打出来 ✓（推理前一次、推理中再记 ultralytics 回给我们的 `r.path` 前几个 ✓）。
    ctx.log("喂入       %d 张；前 3 个：%s"
            % (len(frames), "、".join(f.name for f in frames[:3]) or "（空）"))
    ctx.log("")

    # 轮到这个阶段时任务可能已经被取消了 —— 别为一个已取消的任务去加载模型：
    # YOLO(...) 首次加载要几秒到几十秒（还要初始化 CUDA），这一下拦不住。
    if ctx.canceled():
        ctx.log("已取消，跳过 YOLO 辅助", "warn")
        # ⭐⭐ **取消也要留记录**（2026-10-09 ✓）：用户现场"重新跑了一趟、却什么都没变" ✗
        #   —— "被取消 / 没跑 / 补了 0 框"三种在磁盘上长得一模一样 ✓ ⇒ 这一条专门区分它们 ✓
        #   （模型加载要几秒~几十秒 ✓ 期间点取消/被新任务顶掉，是最常见的"跑了但没写"✓）。
        _write_last_run(out, {
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build": AUGMENT_BUILD,
            "frames_dir": str(frames_dir), "out": str(out), "weights": str(weights),
            "mode": mode, "frames": 0, "boxes_added": 0, "frames_merged": 0,
            "skipped": 0, "skipped_names": [], "seconds": 0.0, "canceled": True})
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
        # ⭐ 头 3 条把"ultralytics 回报的路径"也打出来 ✓（与上面"喂入"那行对照 ⇒ 一眼看出
        #   是"喂错了"还是"回报的名字变了"✓ —— 2026-10-10 就是靠这行才分得清 ✓）
        if i <= 3:
            seen_paths.append(path.name or "（空 path）")
            ctx.log("  回报 %d/%d：%s" % (i, n, path.name or "（空 path）"))
        # ⭐⭐ **`r.path` 靠不住**（2026-10-10 实测 ✓）：给 `predict` 传**路径列表**时，ultralytics
        #   会把回报里的 `path` 写成 `image0.jpg / image1.jpg …`（**它内部的序号名** ✗），跟喂进去的
        #   `frame_00179.png` 对不上 ✓ —— 最小复现（真权重、喂 3 张真帧）：
        #     `喂入 ['frame_00179.png','frame_00180.png','frame_00181.png']`
        #     `回报 path='image0.jpg' / 'image1.jpg' / 'image2.jpg'`
        #     （`orig_shape` 是**真的** (1080,1920) ✓、框也检出来了 ✓ ⇒ 图本身没喂错 ✓）
        #   ⇒ 所以**绝不能拿 `r.path` 当"图喂错了"的判据** ✗（上一版就是这么误判的 ✓）。
        #   规则：先按路径 / 文件名对 ✓；对不上 ⇒ **按顺序**（第 i 条 ↔ 第 i 张 ✓ ——
        #   `stream` 是**按输入顺序**吐的 ✓ 上面那个复现也印证了：1→image0、2→image1 ✓）。
        stem = _stem_of(path, frame_map)
        if stem is None and i <= len(frames):
            stem = frames[i - 1].stem
            order_fixed += 1
        if stem is None:
            skipped.append(path.name or "（空 path）")
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
    # ⛔⛔ **全军覆没 = 目录指错了**（2026-10-09 ✓ 用户现场就是这条 ✓）：一张都对不上时，
    #   别把它混进普通的"0 框"里（那句"模型不适应"会把人带偏 ✗）—— 这次实测用户的日志就是
    #   `画面 180 张 / 补充 0 框 / 跳过 180 张` ✗，人第一反应是"模型没检出"✗，其实是**画面目录
    #   指到了临时导出的那一份**（名字对不上 ⇒ 一张都没处理 ✓）。
    if order_fixed:
        # ⚠ 这条**不是**错误 ✓：ultralytics 对"路径列表"输入会把回报的 `path` 写成 `imageN.jpg`
        #   （内部序号名 ✓ 见循环里那段实测 ✓）⇒ 我们按**顺序**对回帧名 ✓ 照样干活 ✓。
        ctx.log("  ⓘ ultralytics 把回报里的 `path` 换成了 `imageN.jpg`（它的内部序号名 ✓）"
                "⇒ **已按顺序**对应回帧名：%d 条 ✓" % order_fixed, "info")
    # ⛔ 只有"**连顺序都兜不住**"（`order_fixed == 0`）时才是真的"这批图不属于这个目录" ✗
    if skipped and order_fixed == 0:
        ctx.log("⛔ **这一趟一张都对不上**：喂给模型的 %d 张图，路径与 `--frames` 那个画面目录里的"
                "**一个都不匹配** ✗ ⇒ 「画面目录」要指向**项目的画面目录**"
                "（`projects/<项目>/frames` ✓），别指向临时导出的那一份 ✗。"
                "（`last_augment.json` 里已记下 `frames_dir` 与这些名字样本 ✓）"
                % len(skipped), "warn")
    if skipped:
        _u = sorted(set(skipped))
        ctx.log("  ⚠ **跳过 %d 张对不上帧的图** ✗ —— 写出去也没人认 ✓，所以宁可跳过：%s%s"
                % (len(skipped), "、".join(_u[:5]), "…" if len(_u) > 5 else ""), "warn")
    ctx.log("── YOLO 辅助标注完成 ──", "ok")
    ctx.log("  画面 %d 张 / 补充 %d 框 / 涉及 %d 帧%s"
            % (n, n_boxes, n_merged,
               "／跳过 %d 张" % len(skipped) if skipped else ""))
    ctx.log("  用时 %.1fs" % dt)
    if n_boxes == 0:
        ctx.log("  一个框都没补 —— 模型对这些画面不适应，或画面里本就没有目标"
                "（⚠ 已有框若**和预测框重叠**则按「只补不覆盖」不补 ✓）", "warn")
    # ⭐⭐ 落盘一份"这趟干了什么"（见 `_write_last_run` ✓）：日志只在界面里，过一会儿查不了 ✗
    if _write_last_run(out, {
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build": AUGMENT_BUILD,
            "frames_dir": str(frames_dir),
            "out": str(out),
            "weights": str(weights),
            "mode": mode,
            "frames": n,
            "boxes_added": n_boxes,
            "frames_merged": n_merged,
            "skipped": len(skipped),
            "skipped_names": sorted(set(skipped))[:20],
            # ⭐ "喂入的"与"模型回报的"各留几个名字 ⇒ **"目录对不上"还是"名字被换了"一眼分得清** ✓
            "first_inputs": [f.name for f in frames[:5]],
            "first_returned": seen_paths,
            #: 靠"顺序"对回帧名的条数（`r.path` 被 ultralytics 换成 `imageN.jpg` 的那些 ✓）
            "order_mapped": order_fixed,
            "seconds": round(dt, 1),
            "canceled": False}):
        ctx.log("  记录 %s" % (out / "last_augment.json"))
    ctx.log("")

    return {
        "frames": n,
        "boxes_added": n_boxes,
        "frames_merged": n_merged,
        "skipped": len(skipped),
        "mode": mode,
        "seconds": dt,
        "weights": str(weights),
        "summary": "补充 %d 框 / %d 帧（%s）%s"
                   % (n_boxes, n_merged, MODE_ZH.get(mode, "怪物"),
                      "，跳过 %d 张对不上的" % len(skipped) if skipped else ""),
    }


def _write_last_run(out, rec):
    """把"这一趟干了什么"落到 `out/last_augment.json`（**写不进去也不许把这一趟判失败** ✗）。

    ⭐⭐ **为什么要它**（2026-10-09 现场 ✓）：任务的日志只在**界面**里（`gui/worker.py` 的
    `sig_log` ⇒ 只有当时看得见 ✓）⇒ 过一会儿就说不清"上次那趟到底写了没有" ✗。今天就卡在
    这儿：用户"重新跑了一趟"，而 `labels_auto/` 里**一个 mtime 都没变** ✓ —— 但"没变"到底是
    "**没跑** / **被跳过**（`--frames` 指错 ✓）/ **补了 0 框**（预测框都被已有框盖住 ✓
    「只补不覆盖」✓）"这三种里哪一种，**从磁盘上看不出来** ✗。
    ⇒ 每次跑完留一份：含 `frames_dir` ✓（**指错目录一眼就看见** ✓）、`skipped` 与它的名字样本 ✓、
    补了多少 ✓。人（和排查的人）不用猜 ✓。
    """
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / "last_augment.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:                    # noqa: BLE001 —— 记不上不该把这一趟判失败 ✗
        return False


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
