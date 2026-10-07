"""掉落物（道具）自动标注 —— 「自动标注」卡片里那两个按钮的**执行端**（用户 2026-10-04 ✓）。

两种模式（与卡片上「模板匹配标注」/「YOLO标注」一一对应 ✓）：

  · `template` —— **模板匹配**：复用 `tools/detect_mobs.run_detect`（怪物那条**同一条算法** ✓），
    模板来自**掉落物图库**（`datasets/sprites/drop/<id>/*.png` ✓），写出去的类别号是
    **class 2（drop）** ✓ —— 见 `perception/classes.py` 的类别表 ✓。
  · `yolo`     —— **模型补框**：复用 `tools/yolo_augment.run_yolo_augment(mode="drop")` ✓
    （和「YOLO 标注（怪物）」同一套"只补不覆盖"逻辑 ✓ 只是类别号不同 ✓）。

⚠ **图库**：`WzProbe dump-items` 导出道具图标 + 清单（`datasets/sprites/drop/<id>/*.png`
  + `datasets/drops.json` ✓）。⚠ 这里原来写着"**图库现在必然缺**（要等 WzProbe 加 `dump-item`）"
  —— **已经过时** ✗（2026-10-06 实测：清单 **4549 项** ✓ 每项一张 `icon_0.png` ✓）。
  ⇒ 仍然保留**先体检**这一步（`wzexport.drop_library_state()` ✓ 缺什么就说清 ✓），
    再干活 ✓ —— ⛔ **不做"空跑当成功"** ✗（那会让人以为标注跑过了、其实一个字没写 ✓）。

⭐⭐ **掉落物这条链和怪物那条的**四处**不同**（都是小图标逼出来的，改之前先看这里 ✓）：

  · `scale`      —— 用**项目里掉落物自己的**尺度（`drop_scale`，没标过回落总尺度 ✓
                    见 `gui/steps/cards.py::_sprite_scale(p, "drop")` ✓）；不传就兜 `DEFAULT_DROP_SCALE` ✓；
  · `min_side` / `min_alpha` —— **放松**成 5px / 60（图标本来就小 ✓ 见 `DROP_MIN_SIDE` 的说明 ✓）；
  · `min_energy_ratio` —— ⭐ **收紧**成 `DROP_MIN_ENERGY_RATIO`（0.55）：小图标会在"纯色 / 黑边"
                    上留下高分假峰 ⇒ 用能量闸砍掉 ✓（默认 0.35 是怪物那条的 ✓）；
  · `mirror`     —— ⭐ **不镜像**（`False`）：道具图标没有左右朝向 ⇒ 省一半时间、
                    少一批"只有镜像才像"的假匹配 ✓。
"""

from pathlib import Path

from core.context import TaskContext
from core import wzexport

#: 缩放的**兜底值**（只在调用方**没给** `scale` 时用 ✓）。
#: ⚠ 2026-10-06 更正：这里原来写着"掉落物**没有被标定过** ⇒ 先按 1.0 走"——**已经过时** ✗：
#:   ③ 标定尺度那张卡现在**支持掉落物**（`drop_scale` ✓ 用户 2026-10-05 ✓），
#:   ④ 的 `make_task` 也**总会**把 `drop_scale`（没标过就回落总尺度 ✓）传下来 ✓
#:   ⇒ 正常链路**走不到这个 1.0** ✓；它只给"命令行 / 别的调用方什么都不传"兜底 ✓。
#:   ⚠ 真要按 1.0 跑（比如直接调 `run_detect`）时注意：模板会小 ~40% ⇒ **一个都匹配不上** ✗，
#:     那跟阈值无关，别往阈值方向找 ✓（同 `cards.py::_sprite_scale` 那段说明 ✓）。
DEFAULT_DROP_SCALE = 1.0

#: ⭐⭐ 「窗口能量比」的下限（**掉落物专用**，用户 2026-10-06 ✓）—— 见 `detect_mobs.work` 里
#: `_masked_energy` 那一段 ✓。
#:
#: 它管的是"**这个窗口里到底有没有一个精灵那么多能量**"（振幅比，不是相似度 ✓）：
#:   窗口能量 < 这个比例² × 模板能量 ⇒ 判为**退化解**、这个峰丢掉继续找 ✓。
#: 默认 0.35 是给**怪物**定的（怪大、画面里总有纹理 ✓），而**小图标**在"纯色 / 黑边 /
#:   一块平静的地面"上也会出现高分峰 ✗（几个杂散亮像素就够）⇒ 掉落物提到 **0.55** 更合适 ✓
#:   （把那种"画面那头是一片纯色"的峰砍掉 ✓）。
#: ⚠ 它不是万能的：**亮而花**的背景（草地纹理 / 招牌）照样能过 ✗ —— 那种靠
#:   「匹配阈值」和「区分度」两道闸 ✓（见卡片那两格的说明 ✓）。
#: ⚠ 相机/别的调用方想覆盖 ⇒ 传 `min_energy_ratio=` ✓（`run_detect` 本来就认这个参数 ✓）。
DROP_MIN_ENERGY_RATIO = 0.55

#: ⭐ **"多小的图算废图"那两道闸要为掉落物放松**（用户 2026-10-04 ✓）。
#: 怪物那条默认 20px / 200 不透明像素（小图当模板会制造假匹配 ⇒ 宁缺勿错 ✓）；
#: 而**掉落物图标本来就小**：实测金币 4 帧 = 23×24 / 25×24 / 23×24 / **5×24**（不透明 116~468）
#: ⇒ 照老阈值，**那张 5×24 的侧视帧会被丢掉** ✗ ⇒ 金币转到侧面那一段永远匹配不上 ✓。
#: ⚠ 放松的代价是假匹配变多 ⇒ 靠 `thresh` / `min_distinct` 两道阈值兜（就是下面那套参数 ✓）。
DROP_MIN_SIDE = 5
DROP_MIN_ALPHA = 60


def _drops_of(params):
    """归一化卡片给的掉落物列表 → `[{"id", "name"}]`（老配置可能只存了 id 字符串 ✓）。"""
    out = []
    for d in (params.get("drops") or []):
        did = str((d or {}).get("id") if isinstance(d, dict) else d or "").strip()
        if did:
            out.append({"id": did, "name": str((d or {}).get("name") or ""
                                               if isinstance(d, dict) else "")})
    return out


def run_label_drops(params, ctx=None):
    """标注掉落物（`template` / `yolo` 两种模式 ✓）。

    params:
        mode         "template"（模板匹配，默认）/ "yolo"（模型补框）
        drops        `[{"id", "name"}]` —— 卡片上那张列表 ✓
        frames       画面目录
        out          标注输出目录（labels_auto ✓）
        vis_dir      可视化图目录（template 模式 ✓）
        only         只处理这些帧（[] / 缺 = 全部 ✓）
        thresh / min_distinct / per_mob / max_peaks / downscale
                     template 模式的模板匹配参数（与怪物那张卡**同一套** ✓）
        weights / conf / imgsz / device / player_id
                     yolo 模式用（与「YOLO 标注（怪物）」同一套 ✓）
    返回 dict（卡片拿它渲染摘要 ✓；`yolo` 模式直接返回 `run_yolo_augment` 的结果 ✓）。
    """
    ctx = ctx or TaskContext()
    mode = str(params.get("mode") or "template").strip().lower()
    drops = _drops_of(params)
    if not drops:
        raise ValueError("还没有选掉落物 —— 先在卡片里加至少一项 ✓")

    # ---- ① 前置体检：图库在不在（**缺什么就说清** ✓ 不假成功 ✗）----
    ok, why = wzexport.drop_library_state()
    if not ok:
        raise RuntimeError("掉落物图库还不能用：\n\n%s" % why)

    ids = [d["id"] for d in drops]
    have = [i for i in ids if wzexport.drop_icon_path(i)]
    missing = [i for i in ids if i not in have]
    if missing:
        ctx.log("图库里没有这几项的图标（会跳过）：%s" % "、".join(missing[:8])
                + ("…" if len(missing) > 8 else ""), "warn")
    if not have:
        raise RuntimeError(
            "选中的掉落物在图库里**一张图都没有** ⇒ 没有模板可用 ✓\n\n%s" % why)

    out = Path(params.get("out") or "")
    if not str(out):
        raise ValueError("没给标注输出目录（out）")
    frames = str(params.get("frames") or "")
    only = [str(s) for s in (params.get("only") or [])]

    # ---- ② 干活：两条都复用**既有实现**（一处算法 ✓ 别在这里另写一套匹配/推理 ✗）----
    if mode == "yolo":
        w = str(params.get("weights") or "")
        if not w:
            raise ValueError("还没选 YOLO 权重 —— 先在卡片的「YOLO 权重」里选一个模型 ✓")
        from tools.yolo_augment import run_yolo_augment
        ctx.log("掉落物：YOLO 补框（class %d）· %d 项" % (2, len(have)))
        return run_yolo_augment({
            "weights": w,
            "frames": frames,
            "out": str(out),
            "mode": "drop",                  # ⇒ 只合并 class 2 ✓ 见 yolo_augment.MODE_CLASS ✓
            "player_id": str(params.get("player_id") or ""),
            "weights_player_id": str(params.get("weights_player_id") or ""),
            "conf": float(params.get("conf", 0.40)),
            "imgsz": int(params.get("imgsz", 960) or 960),
            "device": str(params.get("device", "0")),
            "only": only,
        }, ctx)

    from tools.detect_mobs import run_detect
    from perception.classes import CLASS_DROP
    # ⭐ 「框到可见部分」的两个默认值从**实现那边**拿（一处口径 ✓ 见 `perception/visible_box.py` ✓）
    from perception.visible_box import VISIBLE_KEEP, VISIBLE_TOL
    ctx.log("掉落物：模板匹配（class %d）· %d 项 / %d 项有图"
            % (CLASS_DROP, len(drops), len(have)))
    return run_detect({
        # ⚠ 这里的 `mobs` 就是"模板目录名"（`<sprites>/<id>/*.png` ✓）—— 掉落物当模板名用 ✓
        "mobs": have,
        "sprites": str(wzexport.drop_sprite_dir()),
        "frames": frames,
        "out": str(out),
        "vis_dir": str(params.get("vis_dir") or ""),
        "scale": float(params.get("scale", DEFAULT_DROP_SCALE)),
        # ⭐ 小图闸放松（见 `DROP_MIN_SIDE` 的说明 ✓；不传 ⇒ 走掉落物默认 ✓）
        "min_side": int(params.get("min_side", DROP_MIN_SIDE) or DROP_MIN_SIDE),
        "min_alpha": int(params.get("min_alpha", DROP_MIN_ALPHA) or DROP_MIN_ALPHA),
        "downscale": int(params.get("downscale", 1) or 1),
        "thresh": float(params.get("thresh", 0.90)),
        # ⭐⭐ **掉落物专用**：窗口能量比下限（用户 2026-10-06 ✓ 见常量说明 ✓）
        #   ⚠ 不传 ⇒ `run_detect` 会用 **0.35（怪物那个默认）** ✗ ⇒ 小图标会被"纯色区"
        #     里的高分峰骗到 ✓ ⇒ 这里**一定要**给 ✓。
        "min_energy_ratio": float(params.get("min_energy_ratio",
                                             DROP_MIN_ENERGY_RATIO)),
        # ⭐⭐ **掉落物不镜像**（用户 2026-10-06 ✓ 见 `load_templates` 的说明 ✓）：
        #   道具图标没有左右朝向 ⇒ 镜像那份只是多花一倍时间 + 多一批"只有镜像才像"的假匹配 ✗。
        #   ⚠ 怪物 / 宠物那条**照旧要镜像**（它们真有朝向 ✓）⇒ 只有这里传 False ✓。
        "mirror": bool(params.get("mirror", False)),
        # ⭐ 粗筛开关（同宠物那条 ✓ 见 `detect_mobs.coarse_gate` ✓）；⚠ 默认关 ✗（实测没省时间 ✓）
        "coarse": bool(params.get("coarse", False)),
        "coarse_min": float(params.get("coarse_min", 0.25)),
        # ⭐⭐ 框到**可见部分**（用户 2026-10-05 ✓ 见 `perception/visible_box.py` ✓）—— 原样转发 ✓
        #   ⚠ 默认**关** ✓（改的是标注语义 ⇒ 要开就按项目显式开 ✓）
        #   ⚠ 两个默认值**从实现那边 import**（一处口径 ✓ 别在这儿再写一遍 26.0/0.2 ✗）
        "visible": bool(params.get("visible", False)),
        "visible_tol": float(params.get("visible_tol", VISIBLE_TOL)),
        "visible_keep": float(params.get("visible_keep", VISIBLE_KEEP)),
        "min_distinct": float(params.get("min_distinct", 0.06)),
        "per_mob": int(params.get("per_mob", 20) or 20),
        # ⭐⭐ 用户手工挑的**模板帧**（任务 3 ✓ 键 `<target>:<id>` ✓ 原样转发 ⇒ `run_detect`
        #   按 `target` 摘出自己那一份 ✓ 缺键 = 老逻辑 ✓）
        "frames_sel": params.get("frames_sel") or {},
        "max_peaks": int(params.get("max_peaks", 4)),   # ⚠ 0/-1 原样传 ✓
        "vis": bool(params.get("vis", True)),
        "cls": CLASS_DROP,                    # ⭐ 写 class 2 ✓（默认 1 = 怪物 ✓ 老行为不变 ✓）
        # ⭐ 台账记在 **"drop"** 名下 ✓（用户 2026-10-04 ✓）—— 不传的话 `run_detect` 会按默认
        #   `"mob"` 记，掉落物跑过的帧就会被当成**怪物标过了** ✗（选帧弹窗的"只选未处理"全乱 ✓）。
        "target": "drop",
        "only": only,
    }, ctx)
