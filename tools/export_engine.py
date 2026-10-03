"""把 YOLO 权重导出成 TensorRT 引擎（`.engine`）—— 推理最快的形态。

为什么要有它
    ultralytics 的 `predict()` 内部会把模型 `.float()` ⇒ 即使加载后 `.half()`，
    FP16 也会被覆盖回 FP32（实测：RTX 5070 上 `half=fp16` 但 `infer_ms` 没降）。
    导出成 TensorRT 引擎后，FP16 是**编译期固化**的，不走 predict wrapper ⇒
    真生效。5070 + TensorRT + FP16 + imgsz 640 ⇒ 推理预期 ~3-5 ms（原 22 ms）。

用法
    # 导出（生成同名 .engine 文件）
    python -m tools.export_engine --weights projects/xxx/models/mob_v1.pt --imgsz 640

    # 之后在推理参数里把权重路径改成那个 .engine 文件即可
    # （界面上的权重选择 / config/live.yaml 的 weights）

⚠ 导出固定 imgsz：换尺寸要重新导出。
⚠ GPU 架构绑定：换显卡要重新导出。
⚠ 需要装 TensorRT：pip install tensorrt（或 NVIDIA 官网下载）。
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def export(weights, imgsz=640, half=True, device=0, batch=1):
    """导出一条 TensorRT 引擎。

    weights   .pt 权重路径
    imgsz     推理尺寸（导出后固定，换尺寸要重新导出）
    half      FP16（True）/ FP32（False）
    device    GPU id
    batch     批大小（实时推理用 1）
    """
    from ultralytics import YOLO

    p = Path(weights)
    if not p.exists():
        raise FileNotFoundError("权重文件不存在：%s" % p)

    # ⚠⚠ **中文路径会让 TensorRT 的底层 C 库静默失败**（"failed to load ONNX file"）——
    #   ultralytics / onnx / TensorRT 的 C 扩展读不了中文路径（项目名就是中文地图名 ⇒
    #   这条路必然踩 ✗）。⇒ 检测到非 ASCII ⇒ 复制到无中文的临时目录导出、导出完搬回来 ✓。
    import shutil
    import tempfile
    _orig = p
    _tmpdir = None
    if any(ord(c) > 127 for c in str(p)):
        _tmpdir = Path(tempfile.mkdtemp(prefix="engine_export_"))
        p = _tmpdir / p.name
        shutil.copy2(_orig, p)
        print("⚠ 权重路径含中文，复制到临时目录导出：%s" % p)

    print("加载模型：%s" % p)
    model = YOLO(str(p))

    print("导出 TensorRT 引擎：imgsz=%d  half=%s  device=%s  batch=%d"
          % (imgsz, half, device, batch))
    print("（首次导出可能要几分钟——TensorRT 在自动调优 kernel）...")

    path = model.export(
        format="engine",
        imgsz=int(imgsz),
        half=bool(half),
        device=str(device),
        batch=int(batch),
    )

    # 导出完：把 .engine 搬回原目录（让 bind 能找到它 ✓）
    if _tmpdir is not None:
        engine_src = Path(path)
        engine_dst = _orig.with_suffix(".engine")
        shutil.copy2(engine_src, engine_dst)
        # 清理临时目录（.onnx 等中间文件不搬，只要 .engine ✓）
        shutil.rmtree(_tmpdir, ignore_errors=True)
        path = str(engine_dst)
        print("（已搬回原目录：%s）" % path)

    print("\n✓ 导出完成：%s" % path)
    print("\n下一步：在推理参数里把权重改成这个 .engine 文件即可。")
    print("  界面：推理参数那行的权重选择")
    print("  或 config/live.yaml 的 weights 键")
    return str(path)


def main():
    ap = argparse.ArgumentParser(
        description="把 YOLO 权重导出成 TensorRT 引擎（推理最快的形态）")
    ap.add_argument("--weights", "-w", required=True,
                    help="权重文件路径（.pt）")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="推理尺寸（导出后固定，换尺寸要重新导出）默认 640")
    ap.add_argument("--half", action="store_true", default=True,
                    help="FP16 半精度（默认开；取消用 --no-half）")
    ap.add_argument("--no-half", dest="half", action="store_false",
                    help="导出 FP32（不用 FP16）")
    ap.add_argument("--device", default="0",
                    help="GPU id（默认 0）")
    ap.add_argument("--batch", type=int, default=1,
                    help="批大小（实时推理用 1）默认 1")
    args = ap.parse_args()

    export(args.weights, imgsz=args.imgsz, half=args.half,
           device=args.device, batch=args.batch)


if __name__ == "__main__":
    raise SystemExit(main())
