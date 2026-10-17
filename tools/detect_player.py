"""玩家自动标注（class 0）：模板匹配定位玩家，追加到已有标注。

和怪物标注（detect_mobs.py）的区别：
    - 怪物：多怪种、多模板、多尺度、NMS —— 画面里有多只怪
    - 玩家：单目标、单套站姿模板 —— 画面里只有一个玩家

所以玩家标注简单得多：每帧用 PlayerLocator 定位一次，取最高分命中。
输出**追加**到 labels_auto 的怪物标注后面（LabelCard 先跑怪物、再跑玩家）。

**为什么默认全图匹配**：角色可能出现在画面角落，不能套「中央区域」假设。
全图 stand 模板约 11 秒/帧（102 个模板 × 整幅图），靠多进程并行加速。

**YOLO 粗定位（提速用）**：params 里给了 `yolo`（权重等）时，先跑一遍模型圈出
玩家可能出现的区域，模板匹配只在区域内做；**YOLO 没给出框的帧仍走全图** ——
漏检（玩家在角落/被遮挡）时召回与优化前完全一致，只是常见帧快一个量级。
关掉：`yolo_coarse=False`（或项目里不选模型）。

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

from core.context import ConsoleContext, TaskContext, cancelable
from core.imgio import imread, imwrite
# ⭐⭐ 「框到可见部分」的两个默认值（与怪物/宠物/掉落**同一份** ✓ 见 `perception/visible_box.py` ✓）
from perception import region as region_mod         # ⭐ 框选区域的一处口径 ✓（见 `region.py` ✓）
from perception.visible_box import VISIBLE_KEEP, VISIBLE_TOL


def keep_old_player_lines(existing, region, w, h, cls=None):
    """玩家链写盘前该保留的旧行 ⇒「别的类」+「**区域外**的旧玩家框」✓（2026-10-17 事故修 ✓）。

    ⚠ 与 `detect_mobs.merge_class_lines` **同一套判据** ✓（`perception.region.center_inside`
      ✓ **中心点**在不在区域里 ✓ 归一化坐标乘回画面像素 ✓），但**语义不同** ✗：
        · 那边：**区域内**的本类旧框参与"覆盖 / 0 检出保命"✓；
        · 这边：**区域内**的旧玩家框**一律不保留** ✓ —— 玩家那条**有意整帧重写 class 0** ✓
          （重新采集后旧框位置是错的 ✗ 见 `work` 里那段注释的由来 ✓），"保命"反而会留下
          **位置错的框** ✗ ⇒ 不能照搬那边的保命分支 ✓。
    ⚠ **没给区域** ⇒ 返回"只剩别的类" ✓（= 老行为：整帧重写 class 0 ✓ **零回归** ✓）。
    ⚠ 读不懂的行 ⇒ **留着** ✓（不猜不删 ✓ 与 `merge_class_lines` 同口径 ✓）。
    """
    # ⚠ 默认值**不能写成 `cls=CLASS_PLAYER`** ✗（2026-10-17 实测踩到 ✓）：默认参数是在
    #   **函数定义时**求值的 ✗，而本助手插在文件**靠前**（常量在后面才定义 ✓）⇒
    #   导入这个模块就 `NameError: name 'CLASS_PLAYER' is not defined` ✗✗
    #   ⇒ 一律改成"**调用时**再取"✓（`cls=None` 哨兵 ✓）。
    cls = CLASS_PLAYER if cls is None else cls
    keep = []
    for ln in (existing or "").splitlines():
        p = ln.split()
        if not p:
            continue
        if p[0] != str(cls):
            keep.append(ln)                 # 别的类（怪物 / 掉落 / 宠物）：原样保留 ✓
            continue
        if not region:
            continue                        # 无区域 ⇒ 整帧重写（老行为 ✓）
        try:
            cx, cy = float(p[1]) * w, float(p[2]) * h
        except (IndexError, ValueError, TypeError):
            keep.append(ln)                 # 读不懂 ⇒ 留着（不猜不删 ✓）
            continue
        if region_mod.center_inside(region, cx, cy):
            continue                        # 区域内 ⇒ 交给这一趟的新结果 ✓
        keep.append(ln)                     # ⭐ 区域外 ⇒ **保留** ✓（事故就在这儿 ✗）
    return keep
# 可视化框颜色跟「设置 → 可视化」走（gui.theme 不 import PyQt5，子进程可用）
from gui.theme import class_color

CLASS_PLAYER = 0    # 类别表见 perception/classes.py（id 固定，别在本地另立一份）
_W = {}


def init_worker(cfg):
    from perception.player_locator import PlayerLocator
    cv2.setNumThreads(1)
    _W["cfg"] = cfg
    _W["loc"] = PlayerLocator(
        cfg["player_id"], root=cfg["player_root"], scale=cfg["scale"],
        threshold=cfg["thresh"], filter_prefix=cfg.get("filter_prefix"),
        frame_scales=cfg.get("frame_scales", {}),
        # ⭐ 2026-10-05：玩家那 5 个参数（见 `run_detect_player` 的 cfg 段说明 ✓）
        #   ⚠⚠ 子进程这条**必须一起传** ✗ —— 只改主进程那个预检等于**没接上** ✓
        #     （真正的定位都发生在子进程 / `workers<=1` 时的这条初始化里 ✓）。
        per_mob=cfg.get("per_mob", 0), min_distinct=cfg.get("min_distinct", 0.0),
        max_peaks=cfg.get("max_peaks", 1), downscale=cfg.get("ds", 1.0),
        # ⭐ 任务 3：用户手工挑的模板帧（缺 = 老逻辑 ✓；⚠ 子进程这条必须一起传 ✗）
        frames_sel=cfg.get("frames_sel") or None,
        # ⭐⭐ **框到可见部分**（用户 2026-10-06 ✓ "所有的匹配都需要这个参数" ✓）——
        #   ⚠⚠ 子进程这条**必须一起传** ✗（真正的定位发生在这里 / 下面的预检 ✓）；
        #   缺键 = 关 = 老行为 ✓（老项目 / 老调用方一字不变 ✓）。
        visible=bool(cfg.get("visible", False)),
        visible_tol=cfg.get("visible_tol"),
        visible_keep=cfg.get("visible_keep"))


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
    # 搜索范围：粗定位给了区域就只在区域内找（快得多）；没给（漏检帧 / 没开
    # 粗定位）就是 None → 全图，与优化前**完全一致**（locate 会把偏移加回，
    # 所以 cx/cy 仍是画面坐标，写出的标注格式不变）。
    r = _W["loc"].locate(full, search_rect=(cfg.get("roi") or {}).get(fp.stem),
                         downscale=cfg.get("ds", 1.0))

    # 玩家框（class 0）一律重写，不做「断点续标」：重新采集后旧 txt 里
    # 残留的 class 0 框位置是错的，若按「已有 class 0 就跳过」会整批跳过，
    # 造成「命中 0 帧」的假象。怪物框（class 1）原样保留。
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    # ⚠⚠⚠ **2026-10-17 修（与 `detect_mobs` 同一种病 ✗ 用户原话：「**并不是只标注我框选的
    #   区域，而是把整帧其他的标注框全删掉了**」✓）**：
    #   老写法 `keep` 把**所有** class 0 行都剔掉 ✗ ⇒ **区域模式**（这一趟只搜了框选那块 ✓）
    #   下会把**区域外的玩家框一并丢掉** ✗✗ = 真丢数据 ✓。
    #   ⇒ 现在按区域**分两桶** ✓（判据与 `detect_mobs.merge_class_lines` **同一套** ✓
    #     `perception.region.center_inside` ✓ 中心点 ✓ 归一化乘回像素 ✓）：
    #       · **区域内**的旧玩家框 ⇒ 照旧**不保留** ✓（这一趟的新结果会替掉它 ✓
    #         = "只重标这一块"该有的语义 ✓）；
    #       · **区域外**的旧玩家框 ⇒ **保留** ✓（与怪物框同等对待 ✓ 本次修的就是它 ✓）。
    #   ⚠ **没框选**（`region` 为空 ✓）⇒ **一字不改** ✓ —— 仍是"整帧重写 class 0"✓
    #     （那是**有意**的语义 ✓ 见上面那段注释的由来：重新采集后旧 class 0 位置是错的 ✓）。
    #   ⚠ 读不懂的行 ⇒ **留着** ✓（不猜不删 ✓ 与 `merge_class_lines` 同口径 ✓）。
    _reg = cfg.get("region")
    keep = keep_old_player_lines(existing, _reg, w, h)

    score = None
    if r is not None:
        cx, cy, bw, bh, score, _name = r
        # ⭐⭐ **只标框选区域内的**（用户 2026-10-11 ✓ 见 `perception/region.py` ✓）——
        #   判据是"**中心点在不在区域里**"✓（跨边界的框只要主要在区域内就算 ✓ 见那边说明 ✓）。
        #   ⚠ 没框选（`region` 为空 ✓）⇒ 一律 True ✓ ⇒ **老行为一字不变** ✓。
        if not region_mod.center_inside(cfg.get("region"), cx, cy):
            # ⇒ 下面既不加框、也不画可视化 ✓；⭐ `score` 也清掉 ✗ —— 它是"命中帧数"的账 ✓
            #   被区域丢掉的框**不算命中** ✓（否则摘要会报"命中 30 帧"而框一个没有 ✓）
            r = None
            score = None
        else:
            keep.append("%d %.6f %.6f %.6f %.6f" % (
                CLASS_PLAYER, cx / w, cy / h, bw / w, bh / h))

    out.write_text(("\n".join(keep) + "\n") if keep else "\n", encoding="utf-8")

    if cfg["vis"]:
        if r is not None:
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            cv2.rectangle(full, (x1, y1), (x1 + int(bw), y1 + int(bh)),
                          cfg.get("box_color") or class_color(CLASS_PLAYER), 2)
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

    # 同 detect_mobs：进来先看有没有被取消，别为已取消的任务白加载一遍模板
    if ctx.canceled():
        ctx.log("已取消（还没开始）", "warn")
        return {"summary": "已取消", "frames": 0, "hits": 0}

    frames_dir = Path(params["frames"])
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError("画面目录里没有 png：%s" % frames_dir)

    n_all = len(frames)
    # 只处理选中的帧（「选帧弹窗」给的；None / 空 = 全部）。
    # n_all 必须在过滤前取 —— 同 detect_mobs：清理残留按总帧数判断。
    only = {str(s) for s in (params.get("only") or [])}
    if only:
        frames = [f for f in frames if f.stem in only]
        if not frames:
            raise ValueError("选中的帧在画面目录里一个都找不到（选了 %d 个）"
                             % len(only))
        ctx.log("只处理选中的 %d 帧（画面共 %d 张）" % (len(frames), n_all))

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
        # ⭐ 2026-10-05（用户："参数都放，缺功能的就补功能" ✓）—— 玩家这一段也有了自己的
        #   「区分度 / 最大模板帧 / 每模板峰数 / 降采样」✓（④ 那一段要摆这 5 个 ✓）。
        #   ⚠⚠ **默认值全部 = 老行为**（老项目 / 老调用方一字不变 ✓）：
        #     `per_mob=0` ⇒ 模板帧不截断 ✓；`min_distinct=0.0` ⇒ 不设"最佳/次佳差距"闸 ✓；
        #     `max_peaks=1` ⇒ 每模板只取一个峰 ✓；`downscale=1.0` ⇒ 不缩 ✓。
        #   语义见 `perception/player_locator.py` 里那两段说明（尤其"区分度"对玩家是
        #   **可信度闸**而不是去重闸 ✓）—— 改默认值前先读那两段 ✗。
        "per_mob": int(params.get("per_mob", 0) or 0),
        "min_distinct": float(params.get("min_distinct", 0.0) or 0.0),
        "max_peaks": int(params.get("max_peaks", 1)),   # ⚠ 0/-1 都要原样传 ✓（`or 1` 会把 0 吃掉 ✗）
        "ds": float(params.get("downscale", 1.0) or 1.0),
        # ⭐⭐ 玩家这条手工挑的**模板帧**（任务 3 ✓）：整份字典按 `<target>:<id>` 存 ✓
        #   ⇒ 这里摘出 `player:<角色id>` 那一份（缺 = 老逻辑"全用" ✓）。
        "frames_sel": [str(s) for s in ((params.get("frames_sel") or {})
                                        .get("player:%s" % pid) or [])],
        "filter_prefix": params.get("filter_prefix"),   # None = 加载所有动作帧
        "frame_scales": params.get("frame_scales") or {},  # {player_id:帧stem -> scale}
        "out": str(out),
        "vis": bool(params.get("vis", True)),
        "vis_dir": params.get("vis_dir") or str(out.parent / "vis"),
        # 可视化框颜色：用设置里配的玩家框颜色（主进程取一次，别让每个子进程读配置）
        "box_color": class_color(CLASS_PLAYER),
        # 粗定位结果 {帧stem: (x,y,w,h)}：**主进程算好**再随 initargs 下发
        # （子进程里绝不能碰模型）；空表 = 全部全图匹配
        "roi": {},
        # ⭐⭐ **框到可见部分**（用户 2026-10-06 ✓ 原话："**所有的匹配都需要「框到可见部分」
        #   参数**"）—— 玩家这条链 2026-10-06 才补上（之前只有 阈值/尺度 + 那 5 个 ✓）。
        #   ⚠ **默认关** ✗（改的是**标注语义**、不是修 bug ✓ ⇒ 老项目 / 老调用方一字不变 ✓）；
        #     要开就按项目显式开（设置里那个勾 ✓）；开了之后同一帧的框会变小 ⇒ 要重跑标注 ✓。
        #   ⚠ 实现是 `perception/visible_box.py`（与怪物/宠物/掉落**共用那一份** ✓）。
        # ⭐⭐ **只在框选区域内标注**（用户 2026-10-11 ✓ 原话："点击各个标注按钮，只会在
        #   当前帧框选的**区域内**进行自动标注"✓）—— 口径见 `perception/region.py` ✓。
        #   ⚠ 不传 / 空 ⇒ `None` ⇒ **整幅画面都算** ✓（= 老行为一字不变 ✓）。
        "region": params.get("region"),
        "visible": bool(params.get("visible", False)),
        "visible_tol": float(params.get("visible_tol", VISIBLE_TOL)),
        "visible_keep": float(params.get("visible_keep", VISIBLE_KEEP)),
    }

    # 重新采集后帧数变少时，清掉**帧号已不存在**的旧标注/可视化（同 detect_mobs）
    #
    # ⚠⚠⚠ **2026-10-16 修（事故级 ✓ 与 `detect_mobs` 同一处病 ✓）**：
    #   · 原来**无条件**执行 ✗（"只标这一帧"也照清 ✗）+ 老判据是"帧号 > 帧数"✗
    #     ⇒ **帧号稀疏**的项目每跑一次就删掉一批别人的标注 ✗✗
    #     （用户原话："**区域自动标注…实测影响了其他帧**"✓ 详见
    #      `tools/detect_mobs.py::clean_stale_outputs` 那段说明 ✓）；
    #   · 现在：**`only` 非空 ⇒ 根本不清** ✗（只标几帧就不许动别的帧 ✓）；
    #     真要清 ⇒ 判据换成"**帧号在不在现有帧集合里**"✓。
    from tools.detect_mobs import clean_stale_outputs
    if only:
        ctx.log("只跑选中的 %d 帧 ⇒ **跳过全局清理**（不动其他帧的文件 ✓）" % len(frames))
    else:
        removed = clean_stale_outputs(str(out), cfg["vis_dir"], n_all,
                                      keep_stems=[f.stem for f in frames])
        if removed:
            ctx.log("清理上次残留 %d 个文件（帧号已不在现有帧集合里）" % removed)

    # 主进程先验一次模板能不能加载，避免开了几十个进程才发现模板空
    ctx.progress(0, 0, "加载玩家模板…")
    from perception.player_locator import PlayerLocator
    # ⭐⭐ **边加载边看取消**（用户 2026-10-05 ✓ 原话："能不等他返回吗？"）—— 逐张 `imread`
    #   原来是个整块动作 ✗ ⇒ 点了取消得干等它读完 ✓；传 `ctx.canceled` 就能几百毫秒内收工 ✓
    #   （与怪物那条**同一份口径** ✓ 见 `perception/player_locator.PlayerLocator.__init__` ✓）。
    loc = PlayerLocator(pid, root=cfg["player_root"], scale=cfg["scale"],
                        threshold=cfg["thresh"],
                        filter_prefix=cfg["filter_prefix"],
                        frame_scales=cfg["frame_scales"],
                        per_mob=cfg["per_mob"], min_distinct=cfg["min_distinct"],
                        max_peaks=cfg["max_peaks"], downscale=cfg["ds"],
                        frames_sel=cfg.get("frames_sel") or None,
                        should_stop=ctx.canceled,
                        # ⭐⭐ **框到可见部分**（用户 2026-10-06 ✓）—— 预检这个 locator 也一起给 ✓
                        #   （`workers<=1` 时定位就是用它跑的 ✓ 见上面那段"必须一起传" ✓）
                        visible=bool(cfg.get("visible", False)),
                        visible_tol=cfg.get("visible_tol"),
                        visible_keep=cfg.get("visible_keep"))
    if ctx.canceled():
        ctx.log("已取消（模板加载到一半）", "warn")
        return {"summary": "已取消", "frames": 0, "hits": 0}
    if loc.template_count == 0:
        raise RuntimeError("玩家模板为空：%s/%s" % (cfg["player_root"], pid))

    ctx.log("玩家 %s   模板 %d 个   scale %.3f   阈值 %.2f"
            % (pid, loc.template_count, cfg["scale"], cfg["thresh"]))

    # ---- YOLO 粗定位（可选）：把模板匹配限制在玩家可能出现的区域 ----
    # 实测模板匹配约 11 秒/帧（102 模板 × 整幅图），YOLO 同一批帧约 0.1 秒/帧
    # → 先圈区域再匹配，常见帧快一个量级。**没圈到的帧仍全图匹配**，所以
    # 漏检（玩家在角落/被遮挡）时的召回与优化前完全一致。
    y = params.get("yolo") or {}
    if y.get("weights") and params.get("yolo_coarse", True):
        try:
            from tools.yolo_augment import player_rois
            cfg["roi"] = player_rois(
                [str(f) for f in frames], y["weights"],
                player_id=cfg["player_id"],
                weights_player_id=y.get("weights_player_id", ""),
                conf=float(y.get("conf", 0.25)),
                imgsz=int(y.get("imgsz", 960)),
                device=y.get("device"),
                ctx=ctx)
        except Exception as e:
            # 粗定位只是提速手段 —— 失败就退回全图，不能让标注整体失败
            ctx.log("YOLO 粗定位失败（%s: %s），这批退回全图匹配"
                    % (type(e).__name__, e), "warn")
            cfg["roi"] = {}

    if cfg["roi"]:
        ctx.log("YOLO 粗定位：%d/%d 帧圈出区域（其余 %d 帧仍全图匹配，漏检兜底）"
                % (len(cfg["roi"]), len(frames), len(frames) - len(cfg["roi"])))
    else:
        ctx.log("全图匹配（玩家可能在角落），%d 帧" % len(frames))

    workers = int(params.get("workers", 0)) or max(1, (os.cpu_count() or 4) - 1)
    ctx.log("并行进程 %d" % workers)
    ctx.log("")

    t0 = time.perf_counter()
    hit = 0
    n = len(frames)

    def consume(iterator):
        """收结果。取消由外层 cancelable() 掐断（说明见 core/context.py）。"""
        nonlocal hit
        for i, r in enumerate(iterator, 1):
            if r is not None:
                hit += 1
            if i % 20 == 0 or i == n:
                ctx.progress(i, n, "已定位玩家 %d 帧" % hit)
                ctx.log("  [%d/%d]  命中 %d  用时 %.0fs"
                        % (i, n, hit, time.perf_counter() - t0))
        # 收完了，还是被取消掐断了 —— 后者不能去记台账（见 detect_mobs 同名函数）
        return not ctx.canceled()

    try:
        if workers <= 1:
            init_worker(cfg)
            ok = consume(cancelable((work(str(f)) for f in frames), ctx))
        else:
            pool = mp.Pool(workers, initializer=init_worker, initargs=(cfg,))
            try:
                # cancelable：预热期间也能响应取消（池子还没出结果时 for 会死等）
                ok = consume(cancelable(
                    pool.imap(work, [str(f) for f in frames], chunksize=2), ctx))
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

    # 台账：记下这一轮「玩家标注跑过哪些帧」（选帧弹窗靠它显示已处理/未处理）
    try:
        from gui import labelio
        labelio.mark_processed(out, "player", [f.stem for f in frames])
    except Exception:
        pass

    ctx.log("")
    ctx.log("── 玩家标注完成 ──", "ok")
    ctx.log("  画面 %d 张 / 命中玩家 %d 帧（%.0f%%）"
            % (n, hit, 100.0 * hit / max(1, n)))
    ctx.log("  用时 %.0fs" % dt)
    if hit == 0:
        ctx.log("一帧都没命中玩家 —— 检查 scale/阈值，或画面里没有玩家", "warn")
    elif hit < n * 0.5:
        ctx.log("命中率偏低（%d/%d）—— 玩家在角落/被遮挡时会漏，"
                "质检台可用「玩家框丢失或重复」筛选补" % (hit, n), "warn")
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

    # 第 1 步被取消了就别接着跑第 2 步 —— 否则点取消之后还要再等一整趟玩家标注
    if ctx.canceled():
        ctx.log("已取消，跳过玩家标注", "warn")
        return {"summary": "已取消", "mob": mob_res, "player": None,
                "player_error": None}

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
