"""把**多个项目**的标注素材汇成一份训练集：外来项目只取怪/掉落/NPC，把「玩家」类丢掉。

**为什么不直接把这些项目的 labels/ 一起塞给 `prepare_dataset`**
    每个项目的帧名都从 `frame_00000.png` 开始 —— **全仓同名** ✗。
    扁平合并时，B 项目的 `frame_00007.txt` 会配到 A 项目的 `frame_00007.png` 上
    ⇒ **框画在别人的画面上**，训练照样能跑、照样出 mAP ⇒ 只能靠肉眼发现 ✗。
    （`prepare_dataset._merge_label_lines` 是「按类合并**同一个 stem**」✓，它解决的是
      "同一帧的多层标注" ✓，管不了"不同帧同名" ✗ —— 这是两件事，别混 ✓。）

**这里的做法**
    · 主项目（`--into`）的帧**保持原名** ⇒ `dataset/split.json` 里老帧的 train/val 归属
      继续有效 ✓（换成前缀会让全部帧变"新帧" ⇒ val 被重洗 ⇒ 两次训练的 mAP 不可比 ✗）；
    · 外来项目（`--from`）的帧统一加前缀 `p01__` / `p02__` … ⇒ 永不同名 ✓，
      来源记在 `<stage>/sources.json` 里（p01 是哪个项目、多少帧、丢了哪些类 ✓）；
    · 每张图用**硬链接**进暂存目录（同盘瞬时、不占额外空间 ✓；跨盘自动退回复制 ✓）；
    · 标注**重写**（不是复制文件）：按「人工 labels → labels_iter → labels_auto」的优先级
      合并 ✓，再把外来项目的 `--drop` 类（默认 **0 玩家**）整行删掉 ✓；
      丢完变成**空**的帧直接跳过（空图配空标注 = 教模型"这里什么都没有" ✗
      —— 那帧本来有玩家，只是我们不想要这个类 ✓）。
    · **只要某几类**（`--keep 1` = 只取怪物标注 ✓ 用户 2026-10-24："其他项目没有标注
      宠物、掉落物，我只想取他们的怪物标注"）：白名单优先 —— 类别表里**不在**白名单的
      类一律折进 `drop` 一起丢 ✓。⚠ 它只过滤**标注行**：那些画面里如果有**没标过的**
      宠物/掉落，模型会把它们当背景 ⇒ 想让新模型认宠物/掉落，就别拿这些帧当负样本 ✓。

用法（先只汇到暂存目录，看一眼没问题再构建数据集）::

    python -m perception.merge_projects --into 天空 ^
        --from 森林迷宫_珠缨 --from 鳄鱼潭1

    python -m perception.prepare_dataset --frames data/_merge_天空/frames ^
        --labels data/_merge_天空/labels --out projects/天空/dataset --min-boxes 1

或者一条命令直接构建（`--build`，产物进 `<into>/dataset` ✓）::

    python -m perception.merge_projects --into 天空 --from 森林迷宫_珠缨 --build

⚠ 构建会**改名**旧的 `dataset/`（见 `prepare_dataset` 那段说明 ✓），确认新数据集没问题
  之后再自己删 `dataset_old/` ✓。
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from core.context import ConsoleContext, TaskContext
from perception.detect import classes
from perception.prepare_dataset import _merge_label_lines

#: 项目的标注层，**顺序即优先级**（跟 `gui/steps/cards.py` 的 ⑥ 数据集卡片一致 ✓）
LABEL_LAYERS = ("labels", "labels_iter", "labels_auto")

ROOT = Path(__file__).resolve().parent.parent


def parse_projects(text):
    """界面/命令行那串「其他项目」→ 列表。

    认这几种分隔（随便混用都行 ✓）：`；` `;` `，` `,` `、` 换行、以及**任何空白**
    （含中文输入法那个全角空格 `　` —— 实测就是它把名字粘成了一串 ✓）。
    去空项、去重（保序 ✓）—— 重复写同一个项目会让它被收两遍（同名帧被前缀成
    `p01__` / `p02__` ⇒ 同一张图进两次训练 ✗），所以这里必须去重 ✓。
    """
    out, seen = [], set()
    # ⚠ 界面上是**多选下拉**（`MultiCheckCombo` ✓ 2026-10-24）⇒ 传下来是**列表** ✓；
    #   命令行/老项目里存的是**字符串** ⇒ 两种都得吃 ✓。
    if isinstance(text, (list, tuple, set)):
        raw = [str(x) for x in text]
    else:
        raw = re.split(r"[,;；，、\s]+", str(text or ""))
    for item in raw:
        name = item.strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def parse_drop(text):
    """界面/命令行那串「要丢掉的类」→ id 列表（如 `"0,4"` → `[0, 4]`）。

    ⚠ **写错就报错、不猜** ✗：这个串决定"哪些标注不进训练" ✓ —— 静默忽略一个打错的
      数字（`o` / 空）就等于"什么都没丢" ⇒ 训出来的模型还带着别人的玩家框，
      而且**看不出哪里错了** ✓ ⇒ 干脆 ValueError（界面会把它原样弹出来 ✓）。
    空串 = **不丢任何类** ✓（用户显式清空就是这个意思 ✓）。
    """
    out = []
    # 界面多选下拉传下来是**列表**（可能是 int ✓ 也可能是 "1,2" 这种字符串 ✗ 不该有，
    # 但老项目/手改 yaml 里可能有 ⇒ 一起吃掉 ✓）。
    if isinstance(text, (list, tuple, set)):
        raw = []
        for x in text:
            raw.extend(str(x).replace("；", ",").replace("，", ",").split(","))
    else:
        raw = str(text or "").replace("；", ",").replace("，", ",").split(",")
    for item in raw:
        t = item.strip()
        if not t:
            continue
        if not t.lstrip("-").isdigit():
            raise ValueError("「要丢掉的类」里有个看不懂的项：%r（只收类 id，"
                             "多个用逗号隔开，例如 0,4 —— 空着 = 一个都不丢）" % t)
        v = int(t)
        if v not in out:
            out.append(v)
    return out


def _project_dir(name_or_path):
    """项目名 / 路径 → 目录（名字按 `projects/<名字>` 找 ✓）。"""
    p = Path(name_or_path)
    if p.is_dir():
        return p.resolve()
    cand = ROOT / "projects" / str(name_or_path)
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError("找不到项目：%s（既不是目录，也不在 projects/ 下）"
                            % name_or_path)


def _label_dirs(pdir):
    """该项目**实际存在**的标注层（顺序 = 优先级 ✓）。"""
    return [pdir / n for n in LABEL_LAYERS if (pdir / n).is_dir()]


def _link_or_copy(src, dst):
    """优先硬链接（同盘瞬时 ✓），不行再复制 ✓。"""
    try:
        os.link(str(src), str(dst))
        return "link"
    except OSError:
        shutil.copy2(str(src), str(dst))
        return "copy"


def stage(params, ctx=None):
    """把各项目的「帧 + 标注」汇进暂存目录。返回统计 dict。

    params:
        into     主项目（名字或路径）—— 帧**不加前缀**
        from     外来项目列表 —— 帧加前缀，并按 `keep` / `drop` 过滤类别
        keep     外来项目**只要**这些类（白名单；非空时**其余类一律丢掉** ✓）
        drop     外来项目要丢的类 id（默认 `[0]` = 玩家 ✓；`keep` 非空时再叠一层）
        stage    暂存目录（默认 `data/_merge_<主项目名>`）
    """
    ctx = ctx or TaskContext()
    main = _project_dir(params["into"])
    others = [_project_dir(x) for x in (params.get("from") or [])]
    # ⚠ **没传** `drop` 才默认丢玩家；**传了空列表**就是"一个都不丢" ✓
    #   （界面选「丢弃勾选的类」但一个都不勾 ⇒ 必须真的不丢 ✗ 见 ⑥ 卡 `make_task` ✓）
    _raw_drop = params.get("drop")
    drop = [int(c) for c in ([0] if _raw_drop is None else _raw_drop)]
    keep = [int(c) for c in (params.get("keep") or [])]
    # ⭐⭐⭐ **白名单优先**（2026-10-24 ✓ 用户："**其他项目没有标注宠物、掉落物，
    #   我只想取他们的怪物标注**"）：`keep` 非空 ⇒ 把"类别表里**不在** keep 的"
    #   全折进 `drop` 一起丢掉 ✓。
    #   为什么折成 drop 而不是新写一条过滤 ✗：`_merge_label_lines` 的按类合并是**唯一**
    #   出口（多层标注按类回退 ✓）⇒ 只在这一层加白名单，合并语义一个字不用动 ✓。
    #   ⚠ 用**类别表**（`classes.ORDER`）算"其余的"，不是"这一帧出现过哪些"——
    #     后者会把"这帧正好没标的类"漏掉（以后加了新类也不会自动被挡 ✗）。
    if keep:
        drop = sorted(set(drop) | (set(classes.ORDER) - set(keep)))
    stage_dir = Path(params.get("stage") or (ROOT / "data" /
                                             ("_merge_" + main.name)))

    # 暂存目录是我们自己的产物 ⇒ 每次重建（不清的话旧帧会混进来 ✓）
    for sub in ("frames", "labels"):
        shutil.rmtree(stage_dir / sub, ignore_errors=True)
        (stage_dir / sub).mkdir(parents=True, exist_ok=True)

    sources = {}
    ctx.log("主项目 %s（帧不加前缀，标注原样保留）" % main.name)
    if keep:
        ctx.log("外来项目只取这些类：%s —— 其余（%s）的框一律丢掉"
                % (keep, sorted(set(classes.ORDER) - set(keep))), "warn")
        ctx.log("  ⚠ 只在**标注行**上过滤：那些画面里如果有**没标过的**宠物/掉落，"
                "模型会把它们当背景 ⇒ 想让新模型认宠物/掉落，要么把宠物/掉落补标上，"
                "要么别用这些帧 ✓", "warn")

    # 主项目排第一：`p00` 只是记账用，文件名**不加**前缀 ✓
    todo = [(main, "p00", [], "")]
    for i, d in enumerate(others, start=1):
        todo.append((d, "p%02d" % i, drop, "p%02d__" % i))

    total_img = total_lbl = 0
    for pdir, tag, drop_here, prefix in todo:
        ldirs = _label_dirs(pdir)
        frames_dir = pdir / "frames"
        if not frames_dir.is_dir():
            ctx.log("  %s：没有 frames/，跳过" % pdir.name, "warn")
            continue
        if not ldirs:
            ctx.log("  %s：没有标注目录（%s），跳过"
                    % (pdir.name, " / ".join(LABEL_LAYERS)), "warn")
            continue

        # ⚠ 只在**外来项目**上这么说（`prefix` 非空 ✓）：主项目的 keep 永远是空的
        #   （`todo` 里给它的是 `[]` ✓）—— 拿全局 `keep` 去拼日志会把主项目也写成
        #   "只取类 1"，看着像把主项目的玩家/掉落也丢了 ✗（实测就是这个误会 ✓）。
        _keep_here = keep if prefix else []
        ctx.log("  %s（%s）标注层：%s%s"
                % (pdir.name, tag, " → ".join(d.name for d in ldirs),
                   ("　**只取类 %s**（其余一律丢 ✓）" % (_keep_here,)) if _keep_here
                   else (("　丢弃类：%s" % (drop_here,)) if drop_here else "")))

        n_ok = n_no_label = n_empty = 0
        for f in sorted(frames_dir.glob("*.png")):
            if ctx.canceled():
                ctx.log("已取消", "warn")
                break
            lines = _merge_label_lines(f.stem, ldirs, drop_here)
            if lines is None:
                n_no_label += 1          # 这帧压根没标过 ✓
                continue
            if not lines:
                n_empty += 1             # 标过、但（丢完之后）没东西 ⇒ 别拿去做负样本 ✗
                continue
            dst_stem = prefix + f.stem
            mode = _link_or_copy(f, stage_dir / "frames" / (dst_stem + ".png"))
            (stage_dir / "labels" / (dst_stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            n_ok += 1
            ctx.progress(n_ok, 0, "%s %s（%s）" % (pdir.name, f.stem, mode))

        total_img += n_ok
        total_lbl += n_ok
        sources[tag] = {"project": pdir.name, "path": pdir.as_posix(),
                        "prefix": prefix, "keep": keep, "drop": drop_here,
                        "frames": n_ok, "no_label": n_no_label,
                        "dropped_empty": n_empty}
        ctx.log("    → 收 %d 张（没标注 %d、丢完变空 %d）"
                % (n_ok, n_no_label, n_empty))

    (stage_dir / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ctx.log("")
    ctx.log("暂存 %d 张 → %s" % (total_img, stage_dir.as_posix()))
    ctx.log("来源记账：%s" % (stage_dir / "sources.json").as_posix())
    return {"frames": total_img, "labels": total_lbl,
            "frames_dir": str(stage_dir / "frames"),
            "labels_dir": str(stage_dir / "labels"),
            "sources": sources, "stage": str(stage_dir),
            "summary": "暂存 %d 张" % total_img}


def run_merge(params, ctx=None):
    """暂存 +（`build` 为真时）直接构建到 `<主项目>/dataset`。"""
    ctx = ctx or TaskContext()
    st = stage(params, ctx)
    if not params.get("build"):
        ctx.log("")
        ctx.log("先看一眼暂存目录；确认没问题后构建：", "ok")
        ctx.log("  python -m perception.prepare_dataset --frames %s --labels %s "
                "--out %s/dataset --min-boxes %s"
                % (st["frames_dir"], st["labels_dir"],
                   _project_dir(params["into"]).as_posix(),
                   int(params.get("min_boxes", 1))))
        return st

    from perception.prepare_dataset import run_dataset

    out = _project_dir(params["into"]) / "dataset"
    ctx.log("")
    ctx.log("构建数据集 → %s" % out.as_posix())
    got = run_dataset({
        "frames": st["frames_dir"],
        "labels": [st["labels_dir"]],
        "out": str(out),
        "val_ratio": float(params.get("val_ratio", 0.2)),
        "min_boxes": int(params.get("min_boxes", 1)),
        "quality": int(params.get("quality", 92)),
        "seed": int(params.get("seed", 42)),
        # ⭐ 划分方式跟着下游走（默认"按采集段留尾" ✓ 2026-10-24 见 `_assign_split`）
        "split_mode": params.get("split_mode", "segment"),
    }, ctx)
    got["stage"] = st["stage"]
    got["sources"] = st["sources"]
    got["summary"] = "%s（合并 %d 个来源）" % (got.get("summary", ""),
                                              len(st["sources"]))
    return got


def main():
    ap = argparse.ArgumentParser(
        description="把其他项目的标注素材（丢掉玩家类）汇进来一起训练")
    ap.add_argument("--into", required=True, help="主项目（名字或目录）")
    ap.add_argument("--from", dest="from_", action="append", default=[],
                    metavar="项目", help="外来项目，可重复")
    ap.add_argument("--drop", action="append", type=int, default=None,
                    metavar="类id",
                    help="外来项目要丢的类，可重复（默认 0 = 玩家）")
    ap.add_argument("--keep", action="append", type=int, default=None,
                    metavar="类id",
                    help="外来项目**只要**这些类（白名单，可重复）—— "
                         "例如 `--keep 1` 只取怪物标注，其余类一律丢 ✓")
    ap.add_argument("--stage", default="", help="暂存目录（默认 data/_merge_<主项目>）")
    ap.add_argument("--build", action="store_true", help="顺带构建到 <主项目>/dataset")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--min-boxes", type=int, default=1)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()

    params = {"into": a.into, "from": a.from_, "stage": a.stage,
              "build": a.build, "val_ratio": a.val_ratio,
              "min_boxes": a.min_boxes, "quality": a.quality}
    if a.drop is not None:
        params["drop"] = a.drop
    if a.keep is not None:
        params["keep"] = a.keep
    try:
        got = run_merge(params, ConsoleContext())
    except Exception as e:                      # noqa: BLE001 —— CLI 要一句人话 ✓
        print("[merge] 失败: %s" % e)
        return 1
    print("[merge] %s" % got.get("summary", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
