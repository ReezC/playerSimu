"""宠物自动标注 —— 「自动标注」卡片里宠物那块两个按钮的**执行端**（用户 2026-10-04 ✓ 方案 C ✓）。

两种模式（与卡片上「模板匹配标注」/「YOLO标注」一一对应 ✓）：

  · `template` —— **模板匹配**：复用 `tools/detect_mobs.run_detect`（与怪物/掉落**同一条算法** ✓），
    模板来自**组合外观**（`petlib.compose_pet(pet, equips)` 现算 ✓ ⇒
    `datasets/sprites/pet_combo/<宠物>__<装备+装备>/stand0_*.png` ✓），写出去的类别号是
    **class 5（宠物）** ✓ —— 见 `perception/classes.py`（宠物是**干扰类**：标它只为让模型
    学会区分"宠物 vs 怪"✓）。
  · `yolo`     —— **模型补框**：复用 `tools/yolo_augment.run_yolo_augment(mode="pet")` ✓
    （同一套"只补不覆盖"逻辑 ✓ 只是类别号不同 ✓）。

⚠ 与掉落物那条**同一套纪律**（见 `tools/label_drops.py` ✓）：
  · 先**体检**（`petlib.pet_library_state()` ✓ 缺什么说清 ✓），再干活 ✓；
  · ⛔ **不做"空跑当成功"** ✗（那会让人以为标注跑过了、其实一个字没写 ✓）；
  · 组合**做不出来**（宠物没那动作 / 图读不了）⇒ **如实报**并跳过那一条 ✓，别静默 ✗。
"""

from pathlib import Path

from core import petlib
from core.context import TaskContext

#: 宠物模板比金币大得多（实测褐色小猫 stand0 = 39×38 ✓），但个别帧仍可能很小
#: ⇒ 比怪物那条（20px / 200 不透明像素）松一档 ✓，比掉落物那条（5 / 60）紧一档 ✓。
PET_MIN_SIDE = 8
PET_MIN_ALPHA = 80


def _pets_of(params):
    """归一化卡片给的宠物列表 → `[{"pet", "name", "equips"}]` ✓（容错老/手写的几种写法 ✓）。

    认这三种（都当同一件事 ✓）：
      · `{"pet": "5000000", "equips": ["01802000"], "name": "褐色小猫"}` ✓（卡片存的就是它 ✓）
      · `{"id": "5000000", "equips": [...]}` ✓（把 `id` 当宠物 id ✓）
      · `"5000000"` ✓（只有 id ✓ ⇒ 只标本体、不戴装备 ✓）
    """
    out = []
    for d in (params.get("pets") or []):
        if isinstance(d, dict):
            pid = str(d.get("pet") or d.get("id") or "").strip()
            eq = [str(x).strip() for x in (d.get("equips") or []) if str(x).strip()]
            nm = str(d.get("name") or "").strip()
        else:
            pid, eq, nm = str(d or "").strip(), [], ""
        if pid:
            out.append({"pet": pid, "name": nm, "equips": sorted(set(eq))})
    return out


def run_label_pets(params, ctx=None):
    """标注宠物（`template` / `yolo` 两种模式 ✓）。

    params:
        mode         "template"（模板匹配，默认）/ "yolo"（模型补框）
        pets         `[{"pet", "name", "equips"}]` —— 卡片上那张列表 ✓
        frames       画面目录
        out          标注输出目录（labels_auto ✓）
        vis_dir      可视化图目录（template 模式 ✓）
        only         只处理这些帧（[] / 缺 = 全部 ✓）
        thresh / min_distinct / per_mob / max_peaks / downscale
                     template 模式的模板匹配参数（与怪物**同一套** ✓；`thresh` 由卡片上
                     「匹配阈值（宠物）」给 ✓）
        weights / conf / imgsz / device / player_id
                     yolo 模式用（与「YOLO 标注（怪物）」同一套 ✓）
    返回 dict（卡片拿它渲染摘要 ✓）。
    """
    ctx = ctx or TaskContext()
    mode = str(params.get("mode") or "template").strip().lower()
    pets = _pets_of(params)
    if not pets:
        raise ValueError("还没有选宠物 —— 先在卡片里加至少一项 ✓")

    # ---- ① 前置体检：图库在不在（**缺什么就说清** ✓ 不假成功 ✗）----
    ok, why = petlib.pet_library_state()
    if not ok:
        raise RuntimeError("宠物图库还不能用：\n\n%s" % why)

    # ---- ② **把组合外观算出来**（方案 C 的核心 ✓：模板 = 本体 + 装备按原点叠 ✓）----
    #   ⚠ 这一步只对 `template` 模式做（YOLO 那条不需要模板 ✓ 省得白算 ✓）。
    combos = []                     # [(组合目录名, 那一条 pets 项)]
    if mode != "yolo":
        for p in pets:
            d = petlib.compose_pet(p["pet"], p["equips"])
            if d is None:
                ctx.log("宠物 %s 做不出组合外观（没有 stand0 帧 / 图读不了）⇒ 跳过这一条"
                        % p["pet"], "warn")
                continue
            combos.append((d.name, p))
        if not combos:
            raise RuntimeError(
                "选中的宠物**一个组合外观都做不出来** ⇒ 没有模板可用 ✓\n\n%s" % why)

    out = Path(params.get("out") or "")
    if not str(out):
        raise ValueError("没给标注输出目录（out）")
    frames = str(params.get("frames") or "")
    only = [str(s) for s in (params.get("only") or [])]

    # ---- ③ 干活：两条都复用**既有实现**（一处算法 ✓ 别在这里另写一套匹配/推理 ✗）----
    if mode == "yolo":
        w = str(params.get("weights") or "")
        if not w:
            raise ValueError("还没选 YOLO 权重 —— 先在卡片的「YOLO 权重」里选一个模型 ✓")
        from perception.classes import CLASS_PET
        from tools.yolo_augment import run_yolo_augment
        ctx.log("宠物：YOLO 补框（class %d）· %d 条" % (CLASS_PET, len(pets)))
        return run_yolo_augment({
            "weights": w,
            "frames": frames,
            "out": str(out),
            "mode": "pet",                   # ⇒ 只合并 class 5 ✓ 见 yolo_augment.MODE_CLASS ✓
            "player_id": str(params.get("player_id") or ""),
            "weights_player_id": str(params.get("weights_player_id") or ""),
            "conf": float(params.get("conf", 0.40)),
            "imgsz": int(params.get("imgsz", 960) or 960),
            "device": str(params.get("device", "0")),
            "only": only,
        }, ctx)

    from perception.classes import CLASS_PET
    from tools.detect_mobs import run_detect
    # ⭐ 「框到可见部分」的两个默认值从**实现那边**拿（一处口径 ✓ 见 `perception/visible_box.py` ✓）
    from perception.visible_box import VISIBLE_KEEP, VISIBLE_TOL
    ctx.log("宠物：模板匹配（class %d）· %d 条组合外观" % (CLASS_PET, len(combos)))
    return run_detect({
        # ⚠ 这里的 `mobs` 就是"模板目录名"（`<sprites>/<名>/*.png` ✓）—— 组合目录名当模板名用 ✓
        #   （`petlib.combo_key` 是**可复现**的 ✓ 见它的说明 ✓）
        "mobs": [name for name, _p in combos],
        "sprites": str(petlib.combo_root()),
        "frames": frames,
        "out": str(out),
        "vis_dir": str(params.get("vis_dir") or ""),
        "scale": float(params.get("scale", 1.0)),
        # ⭐ 小图闸放松一档（见 `PET_MIN_SIDE` 的说明 ✓；不传 ⇒ 走宠物默认 ✓）
        "min_side": int(params.get("min_side", PET_MIN_SIDE) or PET_MIN_SIDE),
        "min_alpha": int(params.get("min_alpha", PET_MIN_ALPHA) or PET_MIN_ALPHA),
        "downscale": int(params.get("downscale", 1) or 1),
        "thresh": float(params.get("thresh", 0.90)),
        # ⭐ 粗筛开关（用户 2026-10-04 ✓ "外观不符合的就不要运算了" ✓ 见 `detect_mobs.coarse_gate` ✓）
        #   ⚠ **默认关** ✗：实测检出完全一致、但**没省时间**（同一只宠物的模板彼此太像 ✓）
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
        "cls": CLASS_PET,                 # ⭐ 写 class 5（宠物 ✓；默认 1 = 怪物 ✓ 老行为不变 ✓）
        # ⭐ 台账记在 **"pet"** 名下 ✓（用户 2026-10-04 ✓ 同掉落物那条：不传就成了冒名记进
        #   "mob" ✗ ⇒ 选帧弹窗里"只选未处理"会把宠物跑过的帧当成怪物标过了 ✓）。
        "target": "pet",
        "only": only,
    }, ctx)
