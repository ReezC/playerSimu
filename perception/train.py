"""训练 YOLO 怪物检测器。

CLI:
    python -m perception.train --data datasets/yolo/data.yaml --epochs 120

GUI:
    调 run_train(params, ctx)，见 core/context.py

**为什么要用回调而不是解析 stdout**
    ultralytics 打印的进度是给人看的，格式随版本变。挂在
    on_fit_epoch_end 回调上拿 trainer.metrics 是稳定接口，
    而且能在界面里画出真正的 mAP 曲线。
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

from core.context import ConsoleContext, TaskContext


def _fmt(v):
    try:
        return "%.3f" % float(v)
    except (TypeError, ValueError):
        return "  -  "


def _pick_metric(metrics, *names):
    """从 trainer.metrics 里取指标。

    ultralytics 的 trainer.metrics 是**扁平字典**（形如 'metrics/mAP50(B)'），
    不是 model.val() 那种嵌套结构 —— 按嵌套写会全部取到 None，
    表现就是界面上所有指标都是 "-"。

    键名在各版本间也不完全一致，所以先精确匹配、再退化为子串匹配。
    调用方要把「更具体的名字放前面」：先 mAP50-95 再 mAP50，
    否则 'mAP50' 会误中 'metrics/mAP50-95(B)'。
    """
    if not metrics:
        return None

    for n in names:
        if n in metrics:
            return metrics[n]

    for key, val in metrics.items():
        low = key.lower()
        for n in names:
            if n.lower() in low:
                return val
    return None


class _TrainCanceled(Exception):
    """训练被用户取消时在 epoch 回调里抛出，用于中断 model.train()。"""


def find_last_ckpt(project_dir):
    """在 `project_dir` 下找**最近**一个带 `weights/last.pt` 的 run，返回那个 `last.pt`。

    ⭐ 用户 2026-09-29 ✓ 原话："**自动找最近一个 run 的 last.pt**、崩了能一键续、并且在续
    之前先报一句当前显存够不够"。

    ⚠ **为什么按 `last.pt` 的 mtime 找，而不是按目录名排**：Ultralytics 的目录名是
    `detect_v1` / `detect_v2` / … 或者 `name` / `name2` / `name3`（撞名时加数字后缀 ✗）——
    **字典序不等于时间序** ✗（`detect_v10` 会排在 `detect_v9` 前面 ✓）。按文件的修改时间找
    才是"最近跑的那个" ✓。返回 `None` = 没得续 ✓。
    """
    root = Path(project_dir)
    if not root.exists():
        return None
    best = None
    for ck in root.glob("*/weights/last.pt"):
        try:
            m = ck.stat().st_mtime
        except OSError:
            continue
        if best is None or m > best[0]:
            best = (m, ck)
    return best[1] if best else None


def _run_args(ck):
    """从 run 目录里的 `args.yaml` 读这次训练**真正用的**参数（imgsz / batch / epochs / name）。

    ⚠ **为什么不去 load 检查点**：`last.pt` 里确实存着 `train_args`，但它是**权重文件**
    （几十 MB，还含优化器状态）⇒ 为了读几个整数就 `torch.load` 一次太亏 ✗。
    Ultralytics 每个 run 都会把参数**另存一份纯文本** `args.yaml` ✓ ⇒ 读它 ✓。
    """
    p = Path(ck).parent.parent / "args.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


#: 显存余量低于这个数（GB）就提一句"偏紧" —— 留给碎片 + 训练中途的缓存增长 ✓
VRAM_TIGHT_GB = 1.0


def _vram_verdict(free_gb, need_gb):
    """判这次显存**要不要得下**，返回 `"ok"` / `"tight"` / `"short"`。

    ⭐ 抽成**纯函数**就是为了能测：真实显存没法在用例里造（`free` 是真读数 ✗），
    而 2026-09-29 的错**恰好就出在这个判据上** ✓ ⇒ 定死了要能单测 ✓。

    · `"short"` —— **要不下**（`need > free`）⇒ 必须提醒"先腾空间"✓
      ⚠ **也只有这一档才该提"刚崩过就重启工作台"** —— 那条只在**空间没还回来**时才是原因 ✓；
      空间够的时候顺口说它，会把明明能跑的人劝退 ✗（2026-09-29 用户贴回日志就是这个 ✓）。
    · `"tight"` —— 要得下、但余量不到 `VRAM_TIGHT_GB` ⇒ 提一句"偏紧"就够了 ✓。
    · `"ok"` —— 正常 ✓ ⇒ **什么都别说**（说了就是噪音 ✓）。
    """
    try:
        free_gb = float(free_gb)
        need_gb = float(need_gb)
    except (TypeError, ValueError):
        return "ok"
    if need_gb > free_gb:
        return "short"
    if need_gb > free_gb - VRAM_TIGHT_GB:
        return "tight"
    return "ok"


def _vram_note(imgsz, batch, ctx):
    """续训前先报一句**当前显存够不够**（用户 2026-09-29 ✓ 要的就是这句）。

    ⚠ **为什么要报**：训练崩在 `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`
    最常见的成因就是**显存不够**（2026-09-28 就是这么崩的：12GB 的 5070 被 Edge / QQ /
    微信 / Steam / Excel 这些桌面应用占掉 9.6GB，再叠 `imgsz=1280 + batch=8`（约 9GB）✗）。
    续训前先看一眼，比崩了再回头翻日志强 ✓。
    ⚠ 这是**粗略经验值**、不是精确预算 —— 目的是"提醒你关点东西"，不是"拦住你" ✗（所以
    只 `warn`、不抛异常 ✓）。系数按实测标定：1280²×8 ≈ 9GB ⇒ k ≈ 7e-7 GB/px²。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free_b, total_b = torch.cuda.mem_get_info()
        free, total = free_b / 2 ** 30, total_b / 2 ** 30
    except Exception:
        return
    try:
        need = (float(imgsz) ** 2) * float(batch) * 7e-7
    except (TypeError, ValueError):
        return
    msg = ("显存    空闲 %.1f / 共 %.1f GB，本次约需 %.1f GB"
           % (free, total, need))
    # ⚠⚠ **判据分三档**（2026-09-29 修正 ✓ —— 用户把日志贴回来说"明明够还劝退我"）：
    #   原来写的 `need > free * 0.8` **太保守** ✗：空闲 10.8GB、需 9.2GB 时也报"可能不够"，
    #   可实际**跑得动** ✓ ⇒ 这种误报比不报更糟（人会不再信这个提醒 ✗）。
    #   真实问题不是"有没有八成富余"，而是"**要不要得下**" ✓ ⇒ 见 `_vram_verdict`。
    _v = _vram_verdict(free, need)
    if _v == "short":
        ctx.log(msg, "warn")
        ctx.log("⚠ 显存不够：训练中途报 cuDNN / OOM 就是这个原因。"
                "先关掉 Edge / QQ / 微信 / Steam / Excel 等占显存的程序，再续训 ✓", "warn")
        # ⭐⭐ **2026-09-29 实测踩到的坑**（用户问"我刚才没有开实时啊"才查出来 ✓）：
        #   训练崩了之后，它 fork 出来的 DataLoader 子进程（`multiprocessing.spawn`
        #   ⇒ `--multiprocessing-fork`）**不会自己退** ✗ ⇒ 会把几 GB 显存**一直攥着** ✓。
        #   实测：`CUDNN_STATUS_EXECUTION_FAILED` 崩掉 10 分钟后，`nvidia-smi` 里仍是
        #   **9.6GB / 12GB**，14 个 `pythonw` 子进程（父进程 = `pythonw -m gui.app`）
        #   全在原地 ✓ —— 这时直接续训 ⇒ **一续又崩** ✗（不是"上次的参数不对" ✓
        #   是空间根本没还回来 ✓）。
        #   ⚠ **这句话只在"真的不够"时说**（别无条件挂上 ✗ —— 空间够的时候它是误导 ✓）。
        ctx.log("⚠ **刚崩过一次的话，还要先重启工作台** —— 训练崩掉时它的 DataLoader "
                "子进程（`multiprocessing.spawn`）不会自己退出，会把几 GB 显存一直攥着 ✗"
                "（2026-09-29 实测：崩掉 10 分钟后仍占 9.6GB / 12GB）。"
                "**不释放就续训 ⇒ 一续又崩** ✓", "warn")
    elif _v == "tight":
        # 要得下、只是余量小 ⇒ 提一句就够了：**别喊"先重启工作台"**（那会把人劝退 ✗）
        ctx.log(msg + "  ⚠ 余量偏紧", "warn")
        ctx.log("⚠ 余量不到 %.1fGB：训练中途可能因碎片 / 缓存增长而 OOM ⇒ "
                "顺手关掉几个占显存的程序更稳 ✓（不需要重启工作台）" % VRAM_TIGHT_GB, "warn")
    else:
        ctx.log(msg)          # 够就是够 —— 安静 ✓（多说一个字都是噪音 ✗）


def count_dataset_images(data_yaml):
    """`data.yaml` 那份数据集里**实际有多少张** → `(train, val, 合计)` / `None`（数不出来 ✓）。

    ⚠ 口径是"**磁盘上真有几张**"，**不是**"`split.json` 里记了几张" ✗：训练读的就是磁盘
      （ultralytics 自己扫 `images/train` / `images/val` ✓），两者不一致时**以磁盘为准** ✓
      —— 这也正是"补了帧却忘了重跑 ⑥"能从这儿露出来的原因 ✓。
    ⚠ 数不出来（yaml 缺键 / 目录不在）**一律不抛**：这是给人看的一行字，
      绝不许因为它把整次训练弄失败 ✗。
    """
    try:
        cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
        root = Path(str(cfg.get("path") or Path(data_yaml).parent))
        n = []
        for key in ("train", "val"):
            rel = cfg.get(key)
            d = None
            if rel:
                p = Path(str(rel))
                d = p if p.is_absolute() else (root / p)
            n.append(sum(1 for f in d.glob("*") if f.is_file()) if d and d.is_dir()
                     else 0)
        return (n[0], n[1], n[0] + n[1])
    except Exception:                       # noqa: BLE001
        return None


def run_train(params, ctx=None):
    """训练。

    params:
        data         data.yaml 路径
        model        基础权重（本地没有则自动下载）
        epochs / imgsz / batch / device / patience
        project_dir  训练输出根目录
        name         本次运行的子目录名
        copy_to      训练结束后把 best.pt 复制到这里（可选）
        exist_ok     同名目录是否覆盖

    返回 dict，卡片读它渲染摘要。
    """
    ctx = ctx or TaskContext()

    data = Path(params["data"])
    if not data.exists():
        raise FileNotFoundError(
            "data.yaml 不存在：%s\n请先完成 ⑥ 数据集" % data)

    model_name = params.get("model") or "yolo26n.pt"
    epochs = int(params.get("epochs", 120))
    imgsz = int(params.get("imgsz", 960))
    batch = int(params.get("batch", 8))
    device = str(params.get("device", "0"))
    patience = int(params.get("patience", 40))
    project_dir = params.get("project_dir") or "runs/detect"
    name = params.get("name") or "mob_v1"
    copy_to = params.get("copy_to")
    #: ⭐ **接着上次跑**（用户 2026-09-29 ✓）—— `True` 时忽略下面那些参数、改去续训 ✓
    resume = bool(params.get("resume"))

    # 权重不在本地时 ultralytics 会自己去下载。网络不通就是长时间静默卡住，
    # 界面上看起来像死机 —— 先提示一句，省得盯着猜。
    if not Path(model_name).exists() and not any(
            c in model_name for c in ("/", "\\")):
        ctx.log("本地没有 %s，ultralytics 会尝试自动下载（需要联网）"
                % model_name, "warn")

    ctx.log("基础权重 %s" % model_name)
    # 上一版权重就在手边时提一句：加数据重训有两种都合理的做法 —— 从基础权重整个
    # 重学一遍，或在上一版上继续微调。别糊里糊涂就用了默认的那个名字。
    try:
        _prev = sorted(Path(copy_to).glob("*.pt")) if copy_to else []
    except Exception:
        _prev = []
    if _prev and Path(model_name).name not in {q.name for q in _prev}:
        ctx.log("上一版权重 %s —— 想接着微调就把「基础权重」填它"
                % _prev[-1].as_posix())
    ctx.log("数据     %s" % data.as_posix())
    # ⭐⭐ **本次到底用多少张**（用户 2026-09-29 ✓ 原话："我后补的帧数据训练，它如果早退了
    #   那是不是白补了？"）—— 训练只吃 `data.yaml` 指向的这份**物理拷贝**
    #   （`dataset/images/{train,val}` ✓），`frames/` 里后来补的帧**不会**自动进来；
    #   而全仓原来**没有任何地方**报"训练用了多少帧" ✗ ⇒ 补了帧却忘了重跑 ⑥ 时，
    #   从日志到报告都看不出新帧没被用上 ✗✗（同一份数还要写进 `run.json` ✓
    #   见下面 —— 让人**事后**也能查"这一版到底用了多少张" ✓）。
    _n = count_dataset_images(data)
    if _n is not None:
        ctx.log("本次用到 %d 张（train %d / val %d）" % (_n[2], _n[0], _n[1]))
    ctx.log("轮数 %d   尺寸 %d   批 %d   设备 %s"
            % (epochs, imgsz, batch, device))
    ctx.log("输出     %s" % Path(project_dir).as_posix())
    ctx.log("")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "没有装 ultralytics。\n请执行：pip install ultralytics")

    # ⭐⭐ **续训分支**（用户 2026-09-29 ✓ 原话："自动找最近一个 run 的 last.pt、崩了能
    #    一键续、并且在续之前先报一句当前显存够不够"）。
    #   ⚠ `resume=True` 时 ultralytics **只认 `resume`**，其它参数（data / epochs / imgsz /
    #     batch）一律被忽略、从检查点里读回原来那套 ✗ ⇒ 所以界面上填的那些对续训
    #     **不起作用** ✓ —— 必须在日志里**说清楚**，否则人会以为"把 batch 改小就能续"✗
    #     （**续训救不了显存**：想省显存得在上一次没崩的时候就把 batch 设小 ✓，或者干脆
    #      用普通模式换个 batch 从头跑 ✓）。
    if resume:
        ck = find_last_ckpt(project_dir)
        if ck is None:
            raise FileNotFoundError(
                "没有可续训的检查点：%s 下找不到 */weights/last.pt\n"
                "（还没跑过训练，或者那个 run 目录被删了）"
                % Path(project_dir).as_posix())
        _ra = _run_args(ck)
        # 从 `args.yaml` 读回真实参数 ⇒ 后面的日志、`run.json`、界面摘要说的都是**实话**
        # （否则会显示"界面上填的 120 轮"，而实际续的是原来那个 200 轮 ✗）。
        imgsz = int(_ra.get("imgsz") or imgsz)
        batch = int(_ra.get("batch") or batch)
        epochs = int(_ra.get("epochs") or epochs)
        name = ck.parent.parent.name or name
        project_dir = ck.parent.parent.parent
        ctx.log("")
        ctx.log("── 接着上次跑（resume）──", "warn")
        ctx.log("检查点   %s" % ck.as_posix())
        ctx.log("         最后写入 %s" % time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(ck.stat().st_mtime)))
        ctx.log("参数     沿用检查点里那套（共 %d 轮 · 尺寸 %d · 批 %d）"
                "—— 界面上填的**不生效**" % (epochs, imgsz, batch))
        _vram_note(imgsz, batch, ctx)
        ctx.log("")
        ctx.progress(0, 0, "加载检查点…")
        model = YOLO(str(ck))
    else:
        ctx.progress(0, 0, "加载模型（首次可能下载权重）…")
        model = YOLO(model_name)
    ctx.progress(0, epochs, "开始训练")

    t0 = time.perf_counter()
    best_holder = {}
    #: ⭐ **实际跑到第几轮**（`on_epoch_end` 每轮刷 ✓）—— ⚠ 早停时它 **< `epochs`** ✗，
    #: 而 `run.json` / 摘要 / 界面以前一律写 `epochs`（= **计划**轮数）⇒ 把"跑到 94 轮就早停"
    #: 说成"200 轮" ✗（2026-09-29 用户看到后问"怎么回事" ✓ 正是这个）。
    best_holder["last_epoch"] = 0

    def on_epoch_end(trainer):
        """每个 epoch 结束把指标推到界面。

        挂回调而不是解析 stdout：ultralytics 的打印格式随版本变，
        回调里拿到的 trainer.metrics 是稳定接口。
        """
        e = int(getattr(trainer, "epoch", 0)) + 1
        best_holder["last_epoch"] = e       # ⭐ 记实际轮数（早停时 < epochs ✓ 见上面说明）
        total = int(getattr(trainer, "epochs", 0) or epochs)

        m = getattr(trainer, "metrics", None) or {}
        p = _pick_metric(m, "precision(B)", "precision")
        r = _pick_metric(m, "recall(B)", "recall")
        m50 = _pick_metric(m, "mAP50(B)", "mAP50")
        m95 = _pick_metric(m, "mAP50-95(B)", "mAP50-95")

        ctx.progress(e, total, "epoch %d" % e)
        ctx.log("epoch %3d/%-3d   P %s  R %s   mAP50 %s   mAP50-95 %s"
                % (e, total, _fmt(p), _fmt(r), _fmt(m50), _fmt(m95)))

        # 每个 epoch 结束检查取消 —— model.train() 是阻塞的，
        # 不在这里中断就得等它跑完全部 epoch。
        if ctx.canceled():
            raise _TrainCanceled()

    def on_train_end(trainer):
        sd = getattr(trainer, "save_dir", None)
        if sd:
            best_holder["save_dir"] = str(sd)

    # 回调必须用 add_callback 注册。
    # model.train(callbacks={...}) 在 ultralytics 里不是合法参数，会直接
    # 抛 SyntaxError: 'callbacks' is not a valid YOLO argument。
    model.add_callback("on_fit_epoch_end", on_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    try:
        if resume:
            # ⚠ **只传 `resume=True`**：多传别的 ultralytics 会警告"resume 时被忽略" ✗
            #   （它自己从检查点里读回数据 / 轮数 / 尺寸 / 批 ✓）
            model.train(resume=True)
        else:
            model.train(
                data=str(data),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                patience=patience,
                project=str(project_dir),
                name=name,
                exist_ok=True,
                plots=True,
                val=True,
                verbose=False,
            )
    except _TrainCanceled:
        ctx.log("已取消训练", "warn")
        return {"summary": "已取消"}
    except Exception:
        # ⭐ 崩了**当场**告诉用户怎么接着跑（用户 2026-09-29 ✓ 要的就是"崩了能一键续"）。
        #   ⚠ **不许在这里吞掉异常** ✗ —— 上层（worker / 卡片）要靠它把这一步判成失败 ✓，
        #     否则界面会显示"完成"而其实没训完 ✗。只补一句提示，然后把异常**原样抛上去** ✓。
        ctx.log("", "warn")
        ctx.log("⚠ 训练中断 —— 已经跑出来的进度**没有丢**：把「接着上次跑」勾上再跑一次，"
                "就会从最后一个 epoch 接着训 ✓", "warn")
        # ⚠ **必须现在就提醒"重启工作台"**（2026-09-29 实测 ✓）：崩掉的这套 DataLoader
        #   子进程（`multiprocessing.spawn`）**不会自己退**，会把几 GB 显存一直攥着 ✗；
        #   等用户去续训时才发现"怎么又崩了"就太晚了 ✓（这时说，他顺手重启一下就好 ✓）。
        ctx.log("⚠ 崩掉的这批数据加载子进程**不会自己退**、会一直占着显存（实测崩后 10 分钟"
                "仍占 9.6GB / 12GB ✗）⇒ **要接着训就先重启工作台**，否则一续又崩 ✓", "warn")
        raise

    dt = time.perf_counter() - t0
    # ⭐ **实际跑完的轮数**（早停 ⇒ 小于 `epochs` ✓；拿不到就退回计划值，别写 0 ✗）
    done_epochs = int(best_holder.get("last_epoch") or 0) or epochs
    early_stopped = done_epochs < epochs

    save_dir = Path(best_holder.get("save_dir")
                    or (Path(project_dir) / name))
    best = save_dir / "weights" / "best.pt"

    ctx.log("")
    if not best.exists():
        ctx.log("训练结束但没找到 best.pt（%s）" % best, "warn")
    else:
        ctx.log("best.pt  %.1f MB" % (best.stat().st_size / 1024 / 1024))

        if copy_to:
            dst = Path(copy_to)
            dst.mkdir(parents=True, exist_ok=True)
            target = dst / (name + ".pt")
            try:
                shutil.copy2(str(best), str(target))
                ctx.log("已复制到 %s" % target.as_posix())
            except Exception as e:
                ctx.log("复制失败：%s" % e, "warn")

    # 最终指标从 trainer 读一次，作为摘要
    final = {}
    try:
        m = model.trainer.metrics or {}
        final = {
            "precision": _pick_metric(m, "precision(B)", "precision"),
            "recall": _pick_metric(m, "recall(B)", "recall"),
            "map50": _pick_metric(m, "mAP50(B)", "mAP50"),
            "map": _pick_metric(m, "mAP50-95(B)", "mAP50-95"),
        }
    except Exception:
        pass

    # 这一轮的结果落一份 run.json。⑦ 的卡片要显示「历次版本」就靠它 ——
    # runs/ 里其实也有 ultralytics 写的 results.csv，但那得反解它的列名
    # （各版本会变），而且没法顺带说明这版是用什么基础权重、多少轮训出来的。
    run_info = {
        "name": name,
        "base": model_name,
        # ⚠⚠ **`epochs` = 实际跑完的轮数**（不是计划值 ✗）—— 2026-09-29 修正：早停在 94 轮
        #   时这里写着 200 ⇒ 「训练报告」的历次版本表、以及 `metrics.advise` 那句
        #   "指标：… （200 轮）"全都在说假话 ✗（人会以为真跑满了 ✓）。
        #   计划轮数另存 `epochs_planned`（信息不丢 ✓）。
        "epochs": done_epochs,
        "epochs_planned": epochs,
        "early_stopped": early_stopped,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "seconds": dt,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": final,
        # ⭐ **这一版到底用了多少张**（用户 2026-09-29 ✓）：事后翻 run.json / 训练报告就能
        #   核对"补的帧进没进这一版" ✓ —— 口径同上面那行日志（`count_dataset_images` ✓），
        #   数不出来时**不写这个键**（别写个假的 0 ✗）。
        **({"frames": {"train": _n[0], "val": _n[1], "total": _n[2]}}
           if _n is not None else {}),
    }
    try:
        (save_dir / "run.json").write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    except Exception as e:
        # 只影响卡片显示，不影响权重本身 —— 别让它把整次训练判成失败
        ctx.log("写 run.json 失败（训练结果不受影响）：%s" % e, "warn")

    ctx.log("")
    ctx.log("── 训练完成 ──", "ok")
    if final.get("map50") is not None:
        ctx.log("  P %s  R %s   mAP50 %s   mAP50-95 %s"
                % (_fmt(final.get("precision")), _fmt(final.get("recall")),
                   _fmt(final.get("map50")), _fmt(final.get("map"))))
    ctx.log("  用时 %.1f 分钟" % (dt / 60.0))
    # ⚠ **必须明说"早停"**（2026-09-29 用户看到日志只到 epoch 95 就"完成"了，直接来问
    #   "怎么回事" ✓）—— Ultralytics 自己的早停提示被 `verbose=False` 关掉了 ✗，
    #   不补这句的话，界面上就是"从 95 跳到完成" ✓ 看着像 bug ✓。
    if early_stopped:
        ctx.log("  ⚠ 第 %d 轮**早停**了（连续 %d 轮没再提升）⇒ 实际跑了 %d / %d 轮，"
                "`best.pt` 是**最好那一轮**的权重 ✓（后面那 %d 轮本来就没涨 ✓）"
                % (done_epochs, patience, done_epochs, epochs, epochs - done_epochs),
                "warn")
    ctx.log("  输出 %s" % save_dir.as_posix())
    # 通俗评估 + 建议（同一份口径也贴在训练卡片上，见 gui/steps/cards.TrainCard.summarize）
    try:
        from perception.metrics import advise
        ctx.log("")
        ctx.log(advise(run_info))
    except Exception as e:                      # noqa: BLE001
        ctx.log("生成建议失败（训练结果不受影响）：%s" % e, "warn")
    ctx.log("")

    return {
        "best": str(best) if best.exists() else "",
        "save_dir": str(save_dir),
        "seconds": dt,
        "metrics": final,
        # ⚠ 摘要也说**实际**轮数 + 标出早停（2026-09-29 修正：以前一律写计划值 ⇒ 明明
        #   早停在 94 轮，界面上写着"200 轮" ✗）
        "summary": "mAP50 %s · %d 轮%s" % (
            _fmt(final.get("map50")), done_epochs,
            "（早停）" if early_stopped else ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/yolo/data.yaml")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--project", dest="project_dir", default="runs/detect")
    ap.add_argument("--name", default="mob_v1")
    ap.add_argument("--copy-to", dest="copy_to", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="接着最近一个 run 的 last.pt 继续训（参数沿用检查点里那套）")
    a = ap.parse_args()

    try:
        run_train({
            "data": a.data,
            "model": a.model,
            "epochs": a.epochs,
            "imgsz": a.imgsz,
            "batch": a.batch,
            "device": a.device,
            "patience": a.patience,
            "project_dir": a.project_dir,
            "name": a.name,
            "copy_to": a.copy_to,
            "resume": a.resume,
        }, ConsoleContext())
    except Exception as e:
        print("[train] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
