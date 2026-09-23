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
    ctx.log("轮数 %d   尺寸 %d   批 %d   设备 %s"
            % (epochs, imgsz, batch, device))
    ctx.log("输出     %s" % Path(project_dir).as_posix())
    ctx.log("")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "没有装 ultralytics。\n请执行：pip install ultralytics")

    ctx.progress(0, 0, "加载模型（首次可能下载权重）…")
    model = YOLO(model_name)
    ctx.progress(0, epochs, "开始训练")

    t0 = time.perf_counter()
    best_holder = {}

    def on_epoch_end(trainer):
        """每个 epoch 结束把指标推到界面。

        挂回调而不是解析 stdout：ultralytics 的打印格式随版本变，
        回调里拿到的 trainer.metrics 是稳定接口。
        """
        e = int(getattr(trainer, "epoch", 0)) + 1
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

    dt = time.perf_counter() - t0

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
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "seconds": dt,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": final,
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
    ctx.log("  输出 %s" % save_dir.as_posix())
    ctx.log("")

    return {
        "best": str(best) if best.exists() else "",
        "save_dir": str(save_dir),
        "seconds": dt,
        "metrics": final,
        "summary": "mAP50 %s · %d 轮" % (_fmt(final.get("map50")), epochs),
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
        }, ConsoleContext())
    except Exception as e:
        print("[train] 失败: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
