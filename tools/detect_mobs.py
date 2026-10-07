"""模板匹配自动标注器（多进程）。

用 WZ 导出的精灵作模板，在真实画面上做归一化互相关匹配，
高分命中即为伪标注，再 NMS 去重。

**性能**（GPU 能不能加速、该先做什么）见 `docs/自动标注性能探究.md`：
实测 GPU 只比现在的 19 进程快约 2×（不划算），而**改灰度匹配快 2.6×**
（检出同一批框、位置差 ≤1px，代价是丢掉颜色信息 —— 要做成开关）。

CLI:
    python -m tools.detect_mobs --mobs 0130100,0130101 --frames data/plain2 ^
        --sprites datasets/sprites/mob --out datasets/labels_plain2 --downscale 2

GUI:
    调 run_detect(params, ctx)，见 core/context.py
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core.context import ConsoleContext, TaskContext, cancelable
from core.imgio import imread, imwrite
# 可视化框颜色：跟「设置 → 可视化」里配的走（gui/theme 只依赖 yaml + 类别表，
# 不 import PyQt5，所以子进程里 import 它也不会拖进 Qt）
from gui.theme import class_color

CLASS_MOB = 1   # 类别表见 perception/classes.py（id 固定，别在本地另立一份）
_W = {}


# ══════════════════════════════════════════════════════════════
# 模板
# ══════════════════════════════════════════════════════════════
# 死亡动画不做模板。
#
# 怪物临死时的样子和活着时差异极大（倒下、碎裂、淡出、变半透明），
# 拿它当模板会带来两个坏处：
#   1. 把"正在消失的怪"也框出来 —— 而 bot 不该攻击一只马上就没的目标，
#      这等于主动教模型学错
#   2. 死亡帧透明度高、边缘糊，匹配位置也不稳
# 排除之后这些帧自然不会被检出，正是我们想要的结果。
SKIP_ACTIONS = ("die",)


#: 粗筛（"外观像不像"的廉价预判）把画面/模板缩到 1/`COARSE_FACTOR` ✓（代价约 1/64 ✓）。
COARSE_FACTOR = 8
#: 模板缩完**最短边**小于它就不做粗筛、直接精算 ✓（金币那种小图标 ✓ 缩完就没形状了 ✗）。
COARSE_MIN_SIDE = 10
#: 粗筛**放行线**：粗分低于它 ⇒ 连精算都不做 ✓。
#: ⚠⚠ 必须**宽容** ✗：粗筛只负责掐"**完全不像**"的 ✓ —— 松了只是慢一点 ✓，紧了会**漏检** ✗
#:   （用例钉着"加粗筛前后检出框一字不差" ✓ 实测 0.25 既不漏也省 ✓）。
COARSE_MIN_SCORE = 0.25


def coarse_gate(img_small, tpl_gray, h, w, factor=COARSE_FACTOR,
                min_score=COARSE_MIN_SCORE):
    """这一帧里**像不像**这个模板 ⇒ `True` = 值得精算 ✓ / `False` = 完全不像、跳过 ✓。

    ⭐ 为什么要它（用户 2026-10-04 ✓ 原话："多姿态进组合外观，注意在匹配时，**外观不符合的
      就不要运算了**"）：多姿态把宠物模板从 3 帧涨到 13 帧 ✓（实测耗时 **5.8s → 24.0s** ✗），
      而**每帧里绝大多数模板都不像**（宠物只在少数帧出现、背景还各不相同 ✓）
      ⇒ 先花约 1/64 的代价粗算一遍 ✓，明显不像的**不进 `matchTemplate`**（这条路上最贵的一步 ✓）。
    ⚠ 判据用 `TM_CCOEFF_NORMED`（**不掩码** ✓ 便宜 ✓）：它比的是"形状/明暗格局像不像" ✓，
      不是绝对分数 ⇒ **不能**拿它替代精算的阈值 ✗（那两套分数不可比 ✓），只当**门**用 ✓。
    ⚠ 缩完最短边 < `COARSE_MIN_SIDE` ⇒ **放行**（不做粗筛 ✓）：那么小的模板缩下去没有形状可言 ✓，
      掐了纯粹是漏检风险 ✗。
    ⚠ 任何异常（尺寸不合法等）⇒ **放行** ✓ —— 粗筛只是省时间 ✓，**绝不许**因为它漏检 ✗。
    """
    if img_small is None:
        return True
    th, tw = max(1, int(h) // int(factor)), max(1, int(w) // int(factor))
    if th < COARSE_MIN_SIDE or tw < COARSE_MIN_SIDE:
        return True
    if th >= img_small.shape[0] or tw >= img_small.shape[1]:
        return True
    try:
        ts = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
        # ⚠ **平坦模板**（标准差≈0 ✓ 例如纯色块）在 `TM_CCOEFF_NORMED` 下恒为 0/nan ✗
        #   ⇒ 会被**误拦**（用例 `t_coarse_gate` 抓到的 ✓）⇒ 这种一律放行 ✓。
        if float(ts.std()) < 1.0:
            return True
        cs = cv2.matchTemplate(img_small, ts, cv2.TM_CCOEFF_NORMED)
        best = float(np.nanmax(cs))
    except Exception:                                   # noqa: BLE001 —— 见上：放行 ✓
        return True
    return best >= float(min_score)


def _pick_frames(files, max_per_mob=0):
    """挑出用作模板的帧：**排除死亡动画**，其余**全部**用。

    **为什么默认全用**
        怪物的动作帧本来就不多（本项目实测：绿水灵 12 帧、木妖 6 帧、
        猪猪 8 帧），全用上代价很小，却能完整覆盖每个动画相位。

        曾经用 per_mob=6 做取样，结果把绿水灵 stand 的第三个相位
        （最窄的 stand_2）挤掉了 —— 画面里处于该相位的怪**永远匹配不上**，
        于是自动标注从来不标它，训练集里一个样本都没有，模型也就不认识。
        这种"看不见的洞"比零星误检更难发现。

    max_per_mob 是**上限，不是目标数**：
        · 非死亡帧数 ≤ 上限  → 全部使用，不截断（常规情况）
        · 非死亡帧数 > 上限  → 才按动作轮询取样，保证各动作都有代表帧
        · 0 或负数           → 完全不限制

    **为什么仍然保留上限**
        少数怪（尤其是 Boss）可能有上百个动作帧，全用会让匹配耗时失控，
        所以留一道保险，而不是放任自流。
    """
    groups = {}
    for f in files:
        act = f.name.rsplit("_", 1)[0]
        if act.startswith(SKIP_ACTIONS):
            continue
        groups.setdefault(act, []).append(f)

    # 保持原始文件顺序，避免下游对顺序有依赖时行为突变
    flat = [f for f in files if f.name.rsplit("_", 1)[0] in groups]

    if not flat:
        return []
    if max_per_mob <= 0 or len(flat) <= max_per_mob:
        return flat          # ← 不超上限就全用，这是默认路径

    # 战斗里最常见的姿态优先排队；未列入的动作按名字排在后面
    order = ("stand", "fly", "move", "hit1", "hit2",
             "attack1", "attack2", "skill1", "jump")
    acts = [a for a in order if a in groups]
    acts += [a for a in sorted(groups) if a not in acts]

    picked = []
    round_ = 0
    while len(picked) < max_per_mob:
        added = False
        for a in acts:
            if round_ < len(groups[a]):
                picked.append(groups[a][round_])
                added = True
                if len(picked) >= max_per_mob:
                    break
        if not added:
            break
        round_ += 1

    return picked


def load_templates(root, mob_ids, max_per_mob=0, min_side=20, min_alpha=200,
                   frames_sel=None, should_stop=None, mirror=True):
    """载入模板并裁到 alpha 包围盒。

    max_per_mob 是**上限**（默认 20）：每只怪取"除死亡动画外的所有帧"，
    帧数超过这个上限时才按动作轮询取样。详见 _pick_frames 的说明。

    ⭐ **`min_side` / `min_alpha` 是"多小的图算废图"那两道闸**（用户 2026-10-04 ✓ 加参数）：
      · 默认 **20px / 200 不透明像素** = **怪物那条线的老行为，一字不变** ✓
        （小图当模板会制造假匹配 ⇒ 怪物那边宁缺勿错 ✓）；
      · 但**掉落物图标本来就小**（实测金币 4 帧：23×24 / 25×24 / 23×24 / **5×24** ✗，
        不透明 116~468 ✓）—— 照老阈值，**那张 5×24 的侧视帧会被直接丢掉** ✗，
        于是"金币转到侧面"这个相位永远匹配不上 ✓ ⇒ 掉落物那条传更松的值 ✓
        （见 `tools/label_drops.py` ✓）。⚠ 放松会让假匹配变多 ⇒ 靠 `thresh`/`min_distinct`
        两道阈值兜（掉落物那条就是那么配的 ✓）。

    ⭐⭐ **`should_stop`：让这一步"边读边看有没有被取消"**（用户 2026-10-05 ✓ 原话："能不等他返回吗？"）：
      这一趟要逐个 `imread` 几百张精灵 ✓（外加裁剪/镜像 ✓）—— 原来它是一个**整块调用** ✗，
      点了取消只能等它整块读完才轮到调用方的 `ctx.canceled()` ✓ ⇒ 界面上就是
      "已请求取消：正在收尾…"卡着不动 ✓（正是用户看到的那条 ✓）。
      传一个"该停了吗"的可调用（调用方给 `ctx.canceled` ✓）⇒ 每读一张看一眼 ✓ ⇒ 几百毫秒内收工 ✓。
      ⚠⚠ **被取消时返回的是"已经读到的"那部分** ✗ —— 调用方**必须紧接着查 `ctx.canceled()`**
        并自己收工（别把它当成完整模板 ✗）；默认 `None` = 老行为，一字不变 ✓。

    ⭐ **`mirror`（用户 2026-10-06 ✓）**：每张图要不要**再镜像一份**当模板 ✓。
      · 默认 **True = 老行为**（怪会朝左右两个方向，镜像召回明显更好 ✓、宠物同理 ✓）；
      · `False` 给**没有左右朝向的那一类**用（**掉落物图标** ✓）：省一半匹配时间，
        也少一批"只有镜像才像"的假匹配 ✓（见 `tools/label_drops.py` ✓）。
    """
    out = []
    root = Path(root)

    for mid in mob_ids:
        d = root / mid
        if not d.is_dir():
            continue

        _all = sorted(d.glob("*.png"))
        # ⭐⭐ **用户手工挑的模板帧**（任务 3 ✓ 用户 2026-10-05 ✓）：有名单 ⇒ **先过滤** ✓。
        #   ⚠ 名单是**候选集**，不是替换 ✗ —— 后面 `_pick_frames` 那套（排除死亡动画 ✓
        #     不超过「最大模板帧」就全用 ✓ 超了按动作轮询 ✓）**照旧在它之上跑** ✓
        #     （两根轴正交 ✓ 见 `gui/frame_sel_dialog.py` 开头那段说明 ✓）。
        #   ⚠ 名单里一个都没命中 ⇒ 这一条**没有模板** ✓ —— **不许**回落到全用 ✗：
        #     用户把帧全取消勾，意思就是"这条这一趟别用" ✓ 如实照办 ✓。
        if frames_sel and mid in frames_sel:
            _want = {str(s) for s in frames_sel[mid]}
            _all = [f for f in _all if f.stem in _want]
        files = _pick_frames(_all, max_per_mob)

        for f in files:
            # ⭐ 每读一张看一眼"该停了吗"（见 `should_stop` 的说明 ✓）：被取消 ⇒ 立刻返回
            #   **已经读到的**那部分 ✓（调用方紧接着会再查一次 `ctx.canceled()` ✓）
            if should_stop is not None and should_stop():
                return out
            t = imread(f, cv2.IMREAD_UNCHANGED)
            if t is None or t.ndim != 3 or t.shape[2] != 4:
                continue

            b, al = t[:, :, :3], t[:, :, 3]

            # 裁掉透明边框 —— 大片透明区会把整幅响应图抬高，制造假匹配
            ys, xs = np.nonzero(al > 128)
            if len(xs) < int(min_alpha):
                continue

            b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            al = al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

            if b.shape[0] < int(min_side) or b.shape[1] < int(min_side):
                continue

            frame = f.stem   # 帧 stem（如 stand_0），供 per-帧 scale 查表
            out.append((mid, frame, b, al))
            # 怪会朝左右两个方向，镜像多一份模板，召回明显更好 ✓ **默认开**（老行为 ✓）
            # ⭐⭐ `mirror=False`（用户 2026-10-06 ✓ 掉落物那条用它）：
            #   **道具图标没有左右朝向** ⇒ 镜像那一份纯属
            #     ① 多花一倍匹配时间 ✗；② 多出一批"**只有镜像才像**"的假匹配 ✗
            #     （实测：某图标在背景上"镜像之后才匹配得上"的那种峰，全是误标 ✓）。
            #   ⚠ 默认仍 `True` ✗ —— 怪物 / 宠物那条**要吃镜像**（它们真有朝向 ✓），
            #     只有明确说"这类没有朝向"的调用方才传 False ✓（见 `tools/label_drops.py` ✓）。
            if mirror:
                out.append((mid, frame, cv2.flip(b, 1), cv2.flip(al, 1)))

    return out


def clean_stale_outputs(out_dir, vis_dir, keep_n):
    """清掉帧号 >= keep_n 的残留标注/可视化，返回删除个数。

    重新采集后帧数往往变少（如 500→109），labels_auto/vis 里会留下旧
    采集的 txt/jpg。它们不会被本次运行覆盖（帧号对不上），却会被质检台
    当成有效帧显示 —— 旧尺度的零框残留会造成「运行中翻帧全是空框」的
    假象，也会混进数据集构建。
    """
    n = 0
    for d in (out_dir, vis_dir):
        if not d:
            continue
        d = Path(d)
        if not d.is_dir():
            continue
        for f in d.iterdir():
            m = re.fullmatch(r"frame_(\d+)\.(?:txt|jpg)", f.name)
            # ⚠⚠ 判据是 **`> keep_n`**，不是 `>= keep_n` ✗ —— 帧号是**从 1 起**的
            #   （`frame_00001` … `frame_00186` ✓ 见 `core.wincap` 的命名 ✓），而 `keep_n` 是
            #   **帧数** ⇒ 写 `>=` 会把**最后一帧**当成"超界的残留"删掉 ✗（实测：`keep_n=2`
            #   把 `frame_00002.txt` 删了 ✓）。后果是每跑一趟就丢最后一帧里**别的类的框**
            #   （它会被 `work` 重建成"只有本类" ✓）—— 本轮就是被用例抓出来的 ✓
            #   （`t_second_run_protects_and_reports` 的 class-0 框凭空消失 ✓）。
            if m and int(m.group(1)) > keep_n:
                try:
                    os.remove(str(f))
                    n += 1
                except OSError:
                    pass
    return n


# ══════════════════════════════════════════════════════════════
# NMS
# ══════════════════════════════════════════════════════════════
def _masked_energy(gray, mask_bool, x, y):
    """窗口在**掩码范围内**的灰度平方和（能量）。越界返回 0。

    为什么要算这个：`cv2.matchTemplate(TM_CCORR_NORMED, mask=...)` 的分数是

        分数 = 掩码内互相关 / sqrt(模板能量 × **窗口能量**)

    窗口能量趋 0（纯黑、近黑、纯色 UI 面板）时分母趋 0 —— OpenCV 这时不返回 0，
    而是给出 FLT_MAX 或 NaN。于是"黑区里几个亮像素"就能拿到 1.0 这种满分，
    而每个模板只取前几个峰 → **峰全被黑区吃掉，真正的怪一个都轮不上**（实测：
    148 帧 2543 框，绝大部分在黑边里，画面里的怪基本没标上）。

    判据用「窗口能量 / 模板能量」而不是绝对亮度或绝对标准差：
      · 绝对亮度 → 暗处的怪会被误杀；
      · 绝对标准差（老代码的 min_texture=4）→ 近黑区几个亮像素就有 5~7，拦不住；
      · 能量比 → 比的是「这块到底像不像一个精灵」，与画面明暗无关。
    """
    h, w = mask_bool.shape
    if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
        return 0.0
    patch = gray[y:y + h, x:x + w]
    return float((patch[mask_bool] ** 2).sum())


# ══════════════════════════════════════════════════════════════
# ⭐⭐ 框到**可见部分**（用户 2026-10-05 ✓）—— **实现已搬去 `perception/visible_box.py`** ✓
# ══════════════════════════════════════════════════════════════
# ⭐ 2026-10-06 搬迁（用户："**所有的匹配都需要「框到可见部分」参数**"✓）：这条判据要被
#   **两条内核**共用 —— 本文件（怪物 / 宠物 / 掉落 ✓）与 `perception/player_locator`
#   （玩家 ✓）。而 `perception` **不能**反向 import `tools/detect_mobs` ✗（本模块头上有
#   `from gui.theme import class_color` ⇒ 会把 gui 拖进运行时依赖 ✓）⇒ 搬到中性层 ✓。
#   ⚠ **"为什么这么做"那段说明（用户原话 + 判据 + 天花板）随实现一起搬过去了** ✓（在
#   `perception/visible_box.py` 的模块头 ✓）—— 本文件只留**用它的地方**（`work` ✓）与
#   那两个默认值（`VISIBLE_TOL` / `VISIBLE_KEEP` 得从那边 import ✓，见下面 ✓）。
# ══════════════════════════════════════════════════════════════
# ⭐ 实现搬去 `perception/visible_box.py`（见上面那段说明 ✓）—— 两条内核共用**那一份** ✓
from perception.visible_box import VISIBLE_KEEP, VISIBLE_TOL, visible_box


def nms(dets):
    """dets: [(score, x, y, w, h, tag)]，按分数贪心抑制重叠框。"""
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []

    for d in dets:
        _, x, y, w, h, _ = d
        for k in keep:
            _, kx, ky, kw, kh, _ = k
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            u = w * h + kw * kh - ix * iy
            if u > 0 and ix * iy / u > 0.35:
                break
        else:
            keep.append(d)

    return keep


def _iou_xywh(a, b):
    """两个 (x, y, w, h) 框的 IoU。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _read_manual_mobs(cfg, stem, W, H):
    """读人工修正目录里该帧的人工怪物框（class 1），返回 [(x, y, w, h)] 像素坐标。

    重标怪物时用来判断「新框和人工框是否太近」—— 太近就跳过，不覆盖人工修正。
    """
    manual_dir = cfg.get("manual_dir")
    if not manual_dir:
        return []
    path = Path(manual_dir) / (stem + ".txt")
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    # ⭐ 认哪一类的框跟着**本次目标类**走（`cfg["cls"]` ✓ 默认 class 1 怪物 = 老行为 ✓）——
    #   否则标掉落物（class 2）时会拿**怪物**的人工框去比"太近"⇒ 白跳过一片 ✓
    cls_s = str(int(cfg.get("cls", CLASS_MOB)))
    out = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 5 or parts[0] != cls_s:
            continue
        try:
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        w = bw * W
        h = bh * H
        out.append((cx * W - w / 2.0, cy * H - h / 2.0, w, h))
    return out


# ══════════════════════════════════════════════════════════════
# 进程池 worker（不能碰 ctx —— 它跑在子进程里）
# ══════════════════════════════════════════════════════════════
def init_worker(cfg):
    cv2.setNumThreads(1)
    _W["cfg"] = cfg
    _W["tpl"] = load_templates(Path(cfg["sprites"]), cfg["mobs"], cfg["per_mob"],
                               cfg.get("min_side", 20), cfg.get("min_alpha", 200),
                               # ⭐⭐ 任务 3：用户手工挑的模板帧（缺键 = 老逻辑 ✓）
                               # ⚠ 子进程这条**必须一起传** ✗ —— 真正的匹配都在子进程里 ✓
                               frames_sel=cfg.get("frames_sel") or {},
                               # ⭐ 要不要镜像（缺键 = True = 老行为 ✓ 掉落物那条传 False ✓）
                               mirror=bool(cfg.get("mirror", True)))


def work(fp_str):
    cfg = _W["cfg"]
    fp = Path(fp_str)

    out_path = Path(cfg["out"]) / (fp.stem + ".txt")

    full = imread(fp)
    if full is None:
        return None

    H, W = full.shape[:2]
    img = full
    ox = oy = 0

    if cfg["region"]:
        ox, oy, rw, rh = cfg["region"]
        img = full[oy:oy + rh, ox:ox + rw]

    ds = cfg["ds"]
    if ds > 1:
        img = cv2.resize(img, None, fx=1.0 / ds, fy=1.0 / ds,
                         interpolation=cv2.INTER_AREA)

    dets = []
    mob_scales = cfg.get("mob_scales", {})
    default_scale = cfg.get("scale", 1.12)

    # 浮点**灰度**底图（每帧一次）：下面按掩码算「窗口能量」要用。
    # 必须用灰度：直接对 BGR 求能量会把三个通道加起来，窗口之间没法比。
    imgf = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 窗口能量至少要达到模板能量的这个比例（振幅比）才算「这块真有个精灵」。
    # 0.35 是宽容的：游戏里怪可能略暗/带半透明特效，振幅不会掉一半以上；
    # 而黑区里几个杂散亮像素的能量比通常只有几个百分点。
    min_ratio = float(cfg.get("min_energy_ratio", 0.35))

    # ⭐⭐ **框到可见部分**（用户 2026-10-05 ✓ **原话与判据见 `perception/visible_box.py` 头部** ✓）
    #   —— **默认关** ✓（老项目 / 老调用方一字不变 ✓）；开了 ⇒ 被挡掉的像素不进框 ✓。
    want_visible = bool(cfg.get("visible", False))
    vis_tol = float(cfg.get("visible_tol", VISIBLE_TOL))
    vis_keep = float(cfg.get("visible_keep", VISIBLE_KEEP))

    # ⭐ 粗筛底图（**每帧一次** ✓）：缩到 1/COARSE_FACTOR ✓ 见 `coarse_gate` 的说明 ✓。
    #   `cfg["coarse"]=False` ⇒ 不建、也不筛（对照用 ✓ 用例靠它证明"加粗筛前后一字不差" ✓）。
    img_small = None
    if cfg.get("coarse", True):
        try:
            img_small = cv2.resize(imgf, None, fx=1.0 / COARSE_FACTOR, fy=1.0 / COARSE_FACTOR,
                                   interpolation=cv2.INTER_AREA)
        except Exception:                               # noqa: BLE001 —— 建不出来就不筛 ✓
            img_small = None

    for mid, frame, b, al in _W["tpl"]:
        # 尺度按「怪+帧 → 怪 → 全局默认」三级查找，支持逐帧微调
        sc = mob_scales.get("%s:%s" % (mid, frame),
                            mob_scales.get(mid, default_scale))
        h = int(round(b.shape[0] * sc / ds))
        w = int(round(b.shape[1] * sc / ds))
        if h < 8 or w < 8 or h >= img.shape[0] or w >= img.shape[1]:
            continue

        t = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
        m = (cv2.resize(al, (w, h), interpolation=cv2.INTER_AREA) > 128)
        m = m.astype(np.uint8) * 255
        mbool = m > 0

        # 模板自身能量（掩码内）—— 判据的基准，见 _masked_energy。
        tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32)
        e_t = float((tg[mbool] ** 2).sum())
        if e_t <= 0:
            continue                    # 全黑模板：匹配不出任何东西，别浪费一轮

        # ⭐ 可见框要用模板的 Sobel ✓ —— **每个模板算一次**（别在"每个框"里重算 ✗：
        #   一个模板可能出好几个峰 ✓）。不开这个功能 ⇒ 一次都不算 ✓（零开销 ✓）。
        tgx = tgy = None
        if want_visible:
            tgx = cv2.Sobel(tg, cv2.CV_32F, 1, 0, ksize=3)
            tgy = cv2.Sobel(tg, cv2.CV_32F, 0, 1, ksize=3)

        # ⭐⭐ **先粗筛、再精算**（用户 2026-10-04 ✓ 原话："匹配时，外观不符合的就不要运算了" ✓）：
        #   下面那句 `matchTemplate` 是这条路上**最贵**的一步（整幅 FFT ✓）
        #   ⇒ 这一帧里"完全不像"的模板（多数情况 ✓）连进都不进 ✓ 见 `coarse_gate` ✓。
        if not coarse_gate(img_small, tg, h, w, min_score=cfg.get("coarse_min", COARSE_MIN_SCORE)):
            continue

        try:
            r = cv2.matchTemplate(img, t, cv2.TM_CCORR_NORMED, mask=m)
        except Exception:
            continue

        # **先掐掉退化解**。分母（窗口能量）趋 0 时 OpenCV 不返回 0，而是给
        # FLT_MAX 或 NaN；np.nan_to_num 只把 NaN 变成 0，FLT_MAX 会原样留下 ——
        # 它必然大于阈值，于是每个模板的「前几个峰」全被黑区吃掉，真正的怪
        # 一个都轮不上（实测：满屏黑框 + 画面里的怪基本没标上）。
        # 归一化互相关**按定义不超过 1**，超了就说明是退化解，直接判 0。
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        r[r > 1.0] = 0.0

        if r.max() < cfg["thr"]:
            continue

        # 区分度判据：峰值必须显著高于该响应图的背景水平。
        # 只看绝对分数会在画面各处产生海量假匹配。
        p = float(np.percentile(r, 99.5))

        # 配额只按**收下的**峰算：退化解（能量不足）被否掉时不该消耗名额 ——
        # 否则黑区里几个杂散亮点就把这只怪的 3 个名额用完了。
        # ⭐⭐ 用户 2026-10-05 ✓：`-1` = **无限**（有几个标几个 ✓ 给个安全上限防跑飞 ✓）、
        #   `0` = **不标注**（这一类这一帧一个框都不写 ✓）。`n>0` 照旧 = 最多 n 个 ✓。
        _cap = int(cfg["peaks"])
        got = 0
        if _cap == 0:
            continue                     # 0 = 不标注 ⇒ 这一个模板直接跳过 ✓
        _limit = (_cap * 4 + 4) if _cap > 0 else 4096
        for _ in range(_limit):
            if _cap > 0 and got >= _cap:
                break
            _, mx, _, ml = cv2.minMaxLoc(r)
            if mx < cfg["thr"] or mx - p < cfg["dist"]:
                break

            x, y = int(ml[0]), int(ml[1])
            if min_ratio > 0:
                e_i = _masked_energy(imgf, mbool, x, y)
                if e_i < (min_ratio ** 2) * e_t:
                    # 窗口里没有"一个精灵那么多"的能量（黑边 / 纯色 UI 面板）
                    # → 这个峰是退化解：抹掉它继续找，别当成检出。
                    r[y, x] = 0.0
                    continue

            # ⭐⭐ 写出去的框 = **可见部分**（开了才收 ✓ 见 `perception/visible_box.py` ✓）。
            #   ⚠ 位置（峰值 x,y）与"抹掉邻域"用的 w/h **一律不动** ✗ —— 那些是**模板尺寸**的事 ✓
            #     （收小了去抹 ⇒ 同一只怪会被反复计入 ✓）。
            vx, vy, vw, vh = x, y, w, h
            if want_visible:
                # ⚠ `imgf` 是**这一帧的浮点灰度**、`tg/tgx/tgy/mbool` 是**当前模板**的
                #   （都已和匹配同一尺度 ✓）—— 由 `visible_box` 自己判"证据够不够" ✓
                #   （判不了就原样回全尺寸 ✓ 见那边的说明 ✓）。
                vx, vy, vw, vh = visible_box(imgf, tg, tgx, tgy, mbool, x, y,
                                             vis_tol, vis_keep)

            dets.append((float(mx), vx * ds + ox, vy * ds + oy,
                         vw * ds, vh * ds, mid))
            got += 1

            # 抹掉该峰邻域，避免同一只怪被反复计入
            r[max(0, y - h // 2):y + h // 2 + 1,
              max(0, x - w // 2):x + w // 2 + 1] = 0.0

    kept = nms(dets)

    # 重标怪只补非人工的框：新框和人工怪物框 IoU 过高就跳过，不覆盖人工修正
    manual_boxes = _read_manual_mobs(cfg, fp.stem, W, H)
    if manual_boxes:
        kept = [d for d in kept
                if not any(_iou_xywh(d[1:5], m) > 0.5 for m in manual_boxes)]

    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ⭐ **写哪一类**（`cfg["cls"]` ✓ 默认 class 1 = 怪物 = 老行为一字不变 ✓）：
    #   掉落物（class 2）走同一条模板匹配，只是类别号不同 ✓（用户 2026-10-04 ✓）。
    _cls = int(cfg.get("cls", CLASS_MOB))
    new_lines = ["%d %.6f %.6f %.6f %.6f" % (
        _cls, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
        for _, x, y, w, h, _ in kept]
    # 覆盖**本类**的旧框、保留别的类（例如标怪物时不动玩家框 ✓）—— 改尺度重标时不留旧框 ✓
    existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    keep = [ln for ln in existing.splitlines()
            if ln.split() and ln.split()[0] != str(_cls)]
    old_mine = [ln for ln in existing.splitlines()
                if ln.split() and ln.split()[0] == str(_cls)]
    # ⭐⭐ **这一帧本类一个都没检出、而它原来有本类的框 ⇒ 保留旧框**（用户 2026-10-05 ✓）。
    #   起因是个真坑（用户现场："模板匹配怪物一波之后往后每次提示 检出 0 框"✓）：原来这里
    #   **无条件覆盖**本类旧框 ✗ ⇒ 那一波标出来的框被**悄悄清空** ✓（实测那个项目：113 帧有怪框、
    #   **59 帧只剩玩家/宠物框**、14 帧全空 ✓ 形状吻合 ✓）。
    #   ⚠ **只在"这一帧本类 0 检出"时才保命** ✗：检出非空时**照旧覆盖** ✓ —— 那才是"重标一遍"
    #     的正常语义（改了尺度/阈值就该整帧换新的 ✓ 新旧混一起更坏 ✓）。
    protected = 0
    if not new_lines and old_mine:
        lines = keep + old_mine
        protected = 1
    else:
        lines = keep + new_lines
    out_path.write_text(("\n".join(lines) + "\n") if lines else "\n",
                        encoding="utf-8")

    if cfg["vis"]:
        color = cfg.get("box_color") or class_color(_cls)
        for _, x, y, w, h, _ in kept:
            cv2.rectangle(full, (x, y), (x + w, y + h), color, 2)
        imwrite(Path(cfg["vis_dir"]) / (fp.stem + ".jpg"), full, quality=82)

    # ⚠ 返回三元组（`protected` = 这一帧保住了旧框 ✓ 见上面那段 ✓）—— 收结果那边跟着改 ✓。
    return len(dets), len(kept), protected


# ══════════════════════════════════════════════════════════════
# 统计（质检台用它判断标注质量）
# ══════════════════════════════════════════════════════════════
def collect_stats(labels_dir, frame_dir=None):
    """扫描标注文件算诊断统计。

    界面靠这两组数字判断标注能不能用：
      · 每帧框数 —— 全 0 说明模板根本不匹配；个别帧暴涨说明误检
      · 框尺寸   —— 明显偏大/偏小说明 scale 标定错了，所有标注都得重来
    """
    counts = []
    sides = []          # 框的等效边长（像素）
    W = H = 0
    # ⭐⭐ **按类别**记一份（用户 2026-10-05 ✓）：这一份是"**目录里有什么**"（含别的类、
    #   含上一波的框 ✓）—— 与"本次这一类新增了几框"是**两回事** ✗，报告里必须分开念 ✓
    #   （起因：跑第二趟时 0 检出，日志却报"每帧框数 中位 4"⇒ 看着自相矛盾 ✓）。
    per_cls = {}

    if frame_dir:
        first = next(iter(sorted(Path(frame_dir).glob("*.png"))), None)
        if first is not None:
            img = imread(first)
            if img is not None:
                H, W = img.shape[:2]

    for f in sorted(Path(labels_dir).glob("*.txt")):
        try:
            lines = [x for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        except Exception:
            continue

        counts.append(len(lines))
        for ln in lines:
            p = ln.split()
            if p:                                   # ⭐ 按类别记一笔（报告要按类别念 ✓）
                per_cls[p[0]] = per_cls.get(p[0], 0) + 1
            if len(p) >= 5:
                try:
                    bw, bh = float(p[3]), float(p[4])
                except ValueError:
                    continue
                if W and H:
                    sides.append((bw * W * bh * H) ** 0.5)
                else:
                    sides.append((bw * bh) ** 0.5)

    def m(arr, q):
        if not arr:
            return 0
        return float(np.percentile(arr, q))

    return {
        "frames": len(counts),
        "boxes": int(sum(counts)),
        "zero_frames": int(sum(1 for c in counts if c == 0)),
        "max_boxes": int(max(counts)) if counts else 0,
        "median_boxes": m(counts, 50),
        "size_median": m(sides, 50),
        "size_p10": m(sides, 10),
        "size_p90": m(sides, 90),
        "img_size": (W, H),
        # ⭐ 各类别各有几框（报告按它念 ✓ 见 `per_cls` 上面那段 ✓）
        "per_class": per_cls,
    }


# ══════════════════════════════════════════════════════════════
# 主入口（GUI / CLI 共用）
# ══════════════════════════════════════════════════════════════
def run_detect(params, ctx=None):
    """自动标注。

    params:
        mobs         怪种 id 列表（用这些做模板）
        sprites      精灵库根目录
        frames       画面目录
        out          标注输出目录
        vis_dir      可视化图目录（vis=True 时用）
        scale        精灵缩放比例
        downscale    降采样倍数
        thresh       匹配阈值
        min_distinct 区分度
        min_energy_ratio  窗口能量 / 模板能量 的下限（振幅比，默认 0.35）
        coarse        ⭐ 粗筛开关（**默认 False = 关** ✗）：开了先把画面/模板缩到 1/8 粗算一遍 ✓，
                      明显不像的模板**不进精算** ✓；⚠ 实测（同一批帧）：检出**完全一致** ✓，
                      但**没省时间** ✗ ⇒ 默认关 ✓（见 `coarse_gate` 的说明 ✓）
        coarse_min    粗筛放行线（默认 0.25 ✓ 见 `COARSE_MIN_SCORE` ✓ 只配 `coarse=True` 用 ✓）
        cls          写进标注的**类别号**（默认 1 = 怪物 ✓；掉落物 = 2 ⇒ 同一条算法换类别 ✓）
        target       台账记在谁名下（`"mob"` 默认 ✓ / `"drop"` / `"pet"` …）—— 选帧弹窗按它
                     显示"已处理/未处理"✓；**不许写死**（见函数体里那段说明 ✓）
        min_side     模板最短边小于它就丢掉（默认 20 ✓ = 老行为；掉落物传更松的 ✓）
        min_alpha    模板不透明像素少于它就丢掉（默认 200 ✓ = 老行为 ✓）
        mirror       ⭐ 每张模板要不要**再镜像一份**（默认 True = 老行为 ✓）。
                     **没有左右朝向的那一类**（掉落物图标 ✓）传 False：省一半匹配时间，
                     也少一批"只有镜像才像"的假匹配 ✓（见 `load_templates` 的说明 ✓）。
        per_mob      每种怪用几帧做模板，0=全部
        max_peaks    每个模板最多取几个峰
        region       "x,y,w,h" 或 None
        vis          是否输出可视化图
        workers      进程数，0=自动
    """
    ctx = ctx or TaskContext()

    # 进来先看一眼有没有被取消：这一趟开头要清理残留 + 加载模板（读几百张精灵），
    # 为一个已经取消的任务白做一遍，看上去就是「点了取消没反应」。
    if ctx.canceled():
        ctx.log("已取消（还没开始）", "warn")
        return {"summary": "已取消", "frames": 0, "detections": 0}

    frames_dir = Path(params["frames"])
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError("画面目录里没有 png：%s\n请先完成 ② 采集" % frames_dir)

    limit = int(params.get("limit", 0) or 0)
    if limit:
        frames = frames[:limit]

    n_all = len(frames)
    # 只处理选中的帧（「选帧弹窗」给的；None / 空 = 全部）。
    # **n_all 必须在过滤前取**：下面的「清理帧号超界的残留」是按总帧数判断的，
    # 用过滤后的数量会把没选中的那些帧的标注当成残留删掉 —— 那是真丢数据。
    only = {str(s) for s in (params.get("only") or [])}
    if only:
        frames = [f for f in frames if f.stem in only]
        if not frames:
            raise ValueError("选中的帧在画面目录里一个都找不到（选了 %d 个）"
                             % len(only))
        ctx.log("只处理选中的 %d 帧（画面共 %d 张）" % (len(frames), n_all))

    mobs = [str(m).strip() for m in (params.get("mobs") or []) if str(m).strip()]
    if not mobs:
        raise ValueError("没有指定怪种 —— 请先在 ① 地图 里选一张有怪的地图")

    sprites = params.get("sprites") or ""
    if not Path(sprites).is_dir():
        raise FileNotFoundError("精灵库不存在：%s" % sprites)

    # ⭐ 写进标注文件的**类别号**（`cls` ✓ 默认 class 1 怪物 ⇒ 老调用方一个字都不用改 ✓；
    #   掉落物 = class 2，见 `perception/classes.py` ✓ 用户 2026-10-04 ✓）。
    cls = int(params.get("cls", CLASS_MOB) or CLASS_MOB)
    # ⭐⭐ **这一趟算哪一类标注**（= 台账那个 key ✓；用户 2026-10-04 ✓）：`"mob"`（默认 ✓ 老行为
    #   一字不变 ✓）/ `"drop"` / `"pet"` … —— 决定**记进哪本台账**（`labelio.mark_processed` ✓）。
    #   ⚠⚠ 原来这句是**写死的** `mark_processed(out, "mob", …)` ✗ ⇒ 掉落物/宠物跑完会**冒名**记进
    #     「怪物」那本台账 ✗：① 选帧弹窗里"只选未处理"把掉落物跑过的帧当成**怪物标过了** ✗；
    #     ② 真正该显示"未处理"的掉落物帧反倒显示成已处理 ✗（用户这轮接上选帧弹窗后必踩 ✓）。
    #   ⚠ 与 `cls` 是**两件事**：`cls` 是"框写成几号类别"✓，`target` 是"台账记在谁名下"✓
    #     —— 默认都对应怪物，所以老调用方什么都不用改 ✓。
    target = str(params.get("target") or "mob").strip() or "mob"
    # ⭐ 「多小的图算废图」那两道闸（默认 20px / 200 不透明像素 = 怪物那条的**老行为** ✓；
    #   掉落物图标小，那条会传更松的值 ✓ 见 `load_templates` 的说明与 `tools/label_drops.py` ✓）
    min_side = int(params.get("min_side", 20) or 20)
    min_alpha = int(params.get("min_alpha", 200) or 200)
    # ⭐ 要不要镜像（默认 True = 老行为 ✓ 见 `load_templates` 的说明 ✓）
    mirror = bool(params.get("mirror", True))

    out = Path(params["out"])
    out.mkdir(parents=True, exist_ok=True)

    ds = max(1, int(params.get("downscale", 1)))
    # 键名沿用 per_mob 以兼容已有项目配置，语义是「上限」：
    # 非死亡帧数不超过它就不截断，全量使用。默认 20 足够覆盖绝大多数怪。
    per_mob = int(params.get("per_mob", 20))
    # ⭐⭐ 用户手工挑的**模板帧**（任务 3 ✓）：整份字典是按 `<target>:<id>` 存的 ✓
    #   ⇒ 这里按这一趟的 `target`（mob / drop / pet ✓）**摘出自己那一份** ✓，
    #   再按**模板 id**（= 目录名 ✓）喂给 `load_templates` ✓。缺键 = 老逻辑 ✓。
    _fs_all = params.get("frames_sel") or {}
    frames_sel = {str(k).split(":", 1)[1]: list(v)
                  for k, v in _fs_all.items()
                  if str(k).startswith(target + ":") and ":" in str(k)}
    thr = float(params.get("thresh", 0.90))
    dist = float(params.get("min_distinct", 0.06))
    peaks = int(params.get("max_peaks", 4))
    # 窗口能量相对模板能量的下限（**振幅比**）：低于它就判为退化解 —— 黑边、
    # 纯色 UI 面板里的几个杂散亮像素，分数看着是 1.0，其实毫无意义。
    # 这是"往黑色区域乱标"的真正闸门：老代码用的是局部标准差的绝对阈值
    # （min_texture=4.0），而近黑区域里几个亮像素的 std 就有 5~7，根本拦不住。
    # 判据细节见 _masked_energy。
    min_energy_ratio = float(params.get("min_energy_ratio", 0.35))
    # ⭐ 粗筛开关 —— **默认关** ✗（见 `coarse_gate` 的实测：检出保证一致 ✓ 但**没省时间** ✗，
    #   因为同一只宠物的十几个姿态模板彼此太像 ✓ 粗筛掐不掉 ✓）。
    #   `coarse=True` 时才开 ✓（模板彼此**不像**的场合才值得试 ✓，比如上百个掉落物图标 ✓）。
    coarse = bool(params.get("coarse", False))

    region = params.get("region")
    if isinstance(region, str) and region.strip():
        try:
            region = tuple(int(v) for v in region.split(","))
        except Exception:
            raise ValueError('搜索区域格式应为 "x,y,w,h"，收到: %s' % region)
    else:
        region = None

    vis = bool(params.get("vis", True))
    vis_dir = params.get("vis_dir") or str(out.parent / "vis_map")

    # 上次采集可能帧数更多，先清掉帧号超界的旧标注/可视化，防止旧尺度的
    # 零框残留混进本次结果（质检台显示和数据集构建都会被污染）
    removed = clean_stale_outputs(str(out), vis_dir, n_all)
    if removed:
        ctx.log("清理上次残留 %d 个文件（帧号超出已有的 %d 帧）" % (removed, n_all))

    mob_scales = params.get("mob_scales") or {}
    if not isinstance(mob_scales, dict):
        mob_scales = {}
    cfg = {
        "sprites": sprites,
        "mobs": mobs,
        "ds": ds,
        "scale": float(params.get("scale", 1.12)),
        "mob_scales": mob_scales,   # {mob_id: scale, "mob_id:action": scale}
        "thr": thr,
        "dist": dist,
        "peaks": peaks,
        "min_energy_ratio": min_energy_ratio,
        "coarse": coarse,                                   # ⭐ 粗筛开关 ✓ 见 `coarse_gate` ✓
        "coarse_min": float(params.get("coarse_min", COARSE_MIN_SCORE)),
        # ⭐⭐ **框到可见部分**（用户 2026-10-05 ✓ 见 `perception/visible_box.py` ✓）—— **默认关** ✗
        #   ⇒ 老项目 / 老调用方一字不变 ✓（这是改"标注语义"⇒ 要开就按项目显式开 ✓）。
        "visible": bool(params.get("visible", False)),
        "visible_tol": float(params.get("visible_tol", VISIBLE_TOL)),
        "visible_keep": float(params.get("visible_keep", VISIBLE_KEEP)),
        "per_mob": per_mob,
        "frames_sel": frames_sel,                           # ⭐ 任务 3 ✓ 见上面那段 ✓
        "out": str(out),
        "manual_dir": str(out.parent / "labels"),   # 人工修正目录，重标时保留其怪物框
        "vis": vis,
        "vis_dir": vis_dir,
        "region": region,
        # ⭐ 本次写哪一类（`cls` ✓ 默认 1 怪物；掉落物 = 2 ⇒ 同一条模板匹配只是换类别号 ✓）
        "cls": cls,
        # ⭐ 小图闸（子进程里 `load_templates` 要用 ✓ 默认 = 老行为 ✓）
        "min_side": min_side,
        "min_alpha": min_alpha,
        # ⭐ 镜像开关（子进程里 `load_templates` 要用 ✓ 默认 True = 老行为 ✓）
        "mirror": mirror,
        # 可视化框颜色：用设置里配的**本类**框颜色（跟着「设置 → 可视化」走，
        # 和实时预览/质检台同一份）。在**主进程**取一次放进 cfg，
        # 免得每个子进程都去读一遍配置文件。
        "box_color": class_color(cls),
    }

    ctx.progress(0, 0, "加载怪物模板…")
    # ⭐⭐ **边加载边看取消**（用户 2026-10-05 ✓ 原话："能不等他返回吗？"）—— 这一步要逐个
    #   `imread` 几百张精灵 ✓，原来它是**整块**跑完才轮到下一个检查点 ✗ ⇒ 点了取消要干等它
    #   读完 ✓（界面上就是那句"已请求取消：正在收尾…"卡着 ✓）。
    #   ⚠ 传进去的是 `ctx.canceled`（可调用 ✓ 见 `load_templates` 的说明 ✓）；它被取消时返回的是
    #     **半份** ✗ ⇒ 下面**立刻**再查一次、整趟收工 ✓（绝不拿半份往下走 ✗）。
    _tpl = load_templates(Path(sprites), mobs, per_mob, min_side, min_alpha,
                          frames_sel=frames_sel, should_stop=ctx.canceled,
                          mirror=mirror)
    if ctx.canceled():
        ctx.log("已取消（模板加载到一半）", "warn")
        return {"summary": "已取消", "frames": 0, "detections": 0}
    n_tpl = len(_tpl)
    if n_tpl == 0:
        raise RuntimeError(
            "模板为空 —— 精灵库里找不到这些怪：\n  %s\n"
            "（检查 config/wz.yaml 的 sprite_dir，以及怪种 ID 是否补足 7 位）"
            % ", ".join(mobs[:8]))

    # ⚠ 日志要**说实话**（别写死"含镜像" ✗）：掉落物那条传 `mirror=False` ✓
    ctx.log("模板 %d 个（%d 种怪%s）"
            % (n_tpl, len(mobs), "，含镜像" if mirror else "，**不镜像**"))
    ctx.log("写入类别 class %d" % cls)
    ctx.log("画面 %d 张   默认尺度 %.3f   降采样 %d" % (len(frames), cfg["scale"], ds))
    ctx.log("阈值 %.2f   区分度 %.3f   每帧最多几个框 %d   最小能量比 %.2f"
            % (thr, dist, peaks, min_energy_ratio))
    if region:
        ctx.log("搜索区域 %s" % (region,))
    ctx.log("每帧 %d 次匹配" % n_tpl)

    workers = int(params.get("workers", 0)) or max(1, (os.cpu_count() or 4) - 1)
    ctx.log("并行进程 %d" % workers)
    ctx.log("")

    t0 = time.perf_counter()
    total = 0
    frames_with = 0
    protected = 0        # ⭐ 本类 0 检出 ⇒ 原样保留旧框的帧数（用户 2026-10-05 ✓ 见 `work` ✓）
    n = len(frames)

    def consume(iterator):
        """收结果。取消由外层 cancelable() 负责掐断（见 core/context.py）。

        这里不再自己等结果 —— 直接 `for` 会在 worker 预热期间死等（十几秒），
        那段时间读不到取消。
        """
        nonlocal total, frames_with, protected
        for i, r in enumerate(iterator, 1):
            if r:
                _, kept, prot = r
                total += kept
                protected += prot
                if kept:
                    frames_with += 1
            if i % 20 == 0 or i == n:
                ctx.progress(i, n, "已检出 %d 框" % total)
                ctx.log("  [%d/%d]  检出 %d  用时 %.0fs"
                        % (i, n, total, time.perf_counter() - t0))
        # 迭代结束可能是收完了，也可能是被 cancelable 掐断 —— 要如实区分，
        # 否则会走去 collect_stats / 记台账：未处理的帧会被标成「已处理」。
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
        raise RuntimeError("标注失败: %s: %s" % (type(e).__name__, e))

    dt = time.perf_counter() - t0

    if not ok:
        ctx.log("已取消", "warn")
        return {"summary": "已取消", "frames": n, "detections": total}

    stats = collect_stats(out, frames_dir)

    # 落盘一份，界面的卡片摘要直接读它，不用每次重扫几百个 txt
    try:
        with open(out / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 台账：记下这一轮「**这一类的**标注跑过哪些帧」——选帧弹窗靠它显示已处理/未处理。
    # 取消（ok=False）时不记：那一轮的标注是半截的，宁可让它显示成未处理。
    # ⚠ target 跟着调用方走（见上面那段说明 ✓），**不许再写死** `"mob"` ✗。
    try:
        from gui import labelio
        labelio.mark_processed(out, target, [f.stem for f in frames])
    except Exception:
        pass

    # ⭐⭐ **报告按类别分开念**（用户 2026-10-05 ✓）：`total` 是"**本次这一类**新收下的框" ✓，
    #   而"每帧框数 / 空帧 / 框尺寸"扫的是**整个目录**（含别的类、含上一波的框 ✓
    #   `collect_stats` 不分类 ✓）—— 以前两组并排念 ✗ ⇒ 跑第二趟 0 检出时会出现
    #   "检出 0 框" 与 "每帧框数 中位 4" 并存的怪象 ✓（实测那个项目：793 框 = 玩家 133 +
    #   怪 288 + 宠物 372 ✓）。现在把**口径写在字面上** ✓，谁也别猜 ✓。
    try:
        from perception.classes import ZH_NAMES as _ZH
    except Exception:                       # noqa: BLE001 —— 名字取不到也不能耽误写日志 ✗
        _ZH = {}

    def _cn(c):
        try:
            return _ZH.get(int(c), "?")
        except Exception:                   # noqa: BLE001
            return "?"

    _cls = int(cfg.get("cls", CLASS_MOB))
    _per = stats.get("per_class") or {}
    _dir_total = int(sum(_per.values()))
    _dir_txt = " · ".join("%s %d" % (_cn(k), v) for k, v in sorted(_per.items()))

    ctx.log("")
    ctx.log("── 标注完成 ──", "ok")
    ctx.log("  画面 %d 张 / **本次检出** %d 框（class %d %s）/ 本次有检出 %d 帧（%.0f%%）"
            % (n, total, _cls, _cn(_cls), frames_with,
               100.0 * frames_with / max(1, n)))
    if _dir_total:
        ctx.log("  该目录里的框（**含别的类**、**含上一波** ✓）: 合计 %d —— %s"
                % (_dir_total, _dir_txt))
    ctx.log("  每帧框数（照**该目录全部类**算 ✓）: 中位 %.0f，最多 %d，空帧 %d"
            % (stats["median_boxes"], stats["max_boxes"], stats["zero_frames"]))
    if stats["size_median"]:
        ctx.log("  框尺寸（同上口径 ✓）: 中位 %.0f px（%.0f ~ %.0f）"
                % (stats["size_median"], stats["size_p10"], stats["size_p90"]))
    ctx.log("  用时 %.1fs（%.0f 帧/秒）" % (dt, n / max(dt, 0.001)))
    if protected:
        ctx.log("  ⚠ 有 %d 帧**本次一个都没检出、但原来有本类的框** ⇒ 已**原样保留**（没清空 ✓）"
                "—— 这些帧要真清空，就自己把对应的 txt 删掉再跑 ✓" % protected, "warn")
    ctx.log("")

    # 诊断提示 —— 让用户不必自己看数字找问题
    if frames_with == 0:
        if protected:
            ctx.log("本次没有任何帧检出目标（其中 %d 帧的旧框已原样保留 ✓）。"
                    "先看 vis/ 里的可视化图：图上**一个框都没有** ⇒ 模板/尺度/区域不对；"
                    "图上有框 ⇒ 那次跑的不是这批帧" % protected, "warn")
        else:
            ctx.log("没有任何帧检出目标。可能是：模板与画面姿态差异过大、"
                    "scale 不对、或搜索区域把目标排除了", "warn")
    elif stats["zero_frames"] > n * 0.5:
        ctx.log("过半帧没有检出（%d/%d，按目录算）—— 检查 scale 是否标定正确"
                % (stats["zero_frames"], n), "warn")

    summary = "%d 帧 / 本次 %d 框 / 有检出 %.0f%%" % (n, total,
                                                     100.0 * frames_with / max(1, n))
    if protected:
        summary += "（保留 %d 帧旧框）" % protected

    return {
        "frames": n,
        "detections": total,
        "frames_with_det": frames_with,
        # ⭐ 本类 0 检出 ⇒ 原样保留旧框的帧数（用户 2026-10-05 ✓ 见 `work` ✓）——
        #   卡片摘要 / 用例都读它 ✓（"检出 0" 与 "旧框还在" 这两件事必须能分开说 ✓）。
        "protected": protected,
        "stats": stats,
        "seconds": dt,
        "out_dir": str(out),
        "vis_dir": vis_dir if vis else "",
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mobs", required=True)
    ap.add_argument("--sprites", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="datasets/labels_map")
    ap.add_argument("--vis-dir", default=None)
    ap.add_argument("--scale", type=float, default=1.12)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--thresh", type=float, default=0.90)
    ap.add_argument("--min-distinct", type=float, default=0.06)
    ap.add_argument("--per-mob", type=int, default=20,
                    help="每只怪的模板帧**上限**（默认 20）。"
                         "非死亡帧数不超过它时全部使用，不截断")
    ap.add_argument("--max-peaks", type=int, default=4)
    ap.add_argument("--min-energy-ratio", type=float, default=0.35,
                    help="窗口能量 / 模板能量的下限（振幅比）：低于它的峰判为退化解"
                         "（黑边、纯色面板里的杂散亮点），不当检出。设 0 关闭")
    ap.add_argument("--region", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--vis", action="store_true")
    a = ap.parse_args()

    params = {
        "mobs": [m.strip() for m in a.mobs.split(",") if m.strip()],
        "sprites": a.sprites,
        "frames": a.frames,
        "out": a.out,
        "vis_dir": a.vis_dir,
        "scale": a.scale,
        "downscale": a.downscale,
        "thresh": a.thresh,
        "min_distinct": a.min_distinct,
        "min_energy_ratio": a.min_energy_ratio,
        "per_mob": a.per_mob,
        "max_peaks": a.max_peaks,
        "region": a.region,
        "vis": a.vis,
        "workers": a.workers,
    }

    try:
        run_detect(params, ConsoleContext())
    except Exception as e:
        print("[detect] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
