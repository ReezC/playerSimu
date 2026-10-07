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


def pt_for(weights):
    """从「**当前在用的权重**」推出**该拿哪份 `.pt` 去导出**（GUI 一键导出用 ✓ 2026-10-03 ✓）。

    用户原话："想真用别的尺寸：重新导出那个尺寸的引擎 —— 就不能填写后自动导出吗？
    不合适的话弹提示指引然后搞个按钮一键导出也行" ✓。
    （**"填完自动导出"不做** ✗：导出要几分钟的重编译，而 `imgsz` 是逐位敲进去的
      ⇒ `1`/`12`/`128` 每一击都触发一次全量导出 ✓ 而且会把正在跑的实时卡死 ✓。
      ⇒ 走"按钮一键导出" ✓。）

    · 给 `.pt`      ⇒ 就是它 ✓；
    · 给 `.engine`  ⇒ **同目录的 `.pt`** ✓ —— 先找"**名字一模一样**"那份 ✓
      （`detect_v5.engine` ↔ `detect_v5.pt` ✓）；**找不到就再试一次"去掉结尾尺寸"** ✓✗
      （`detect_v5_960.engine` ↔ `detect_v5.pt` ✓ —— ⚠⚠ **2026-10-05 补的** ✗✗：本模块自己把引擎
       改名成 `<pt名>_<尺寸>.engine` ✗（见 `export_for` ✓），可这里原来只认"同名"✗
       ⇒ **凡是用带尺寸的引擎，就必然弹出"找不到 detect_v5_960.pt"** ✓✓（用户 2026-10-05 撞上 ✓
       —— 而旁边明明躺着 `detect_v5.pt` ✗）。⇒ 补这一步"**去掉结尾 `_数字` 再试**" ✓）；
    · 找不到 ⇒ 返回 `(None, 原因)` ✓ —— **不许猜** ✗（别自己去 `runs/**/best.pt` 里翻一个 ✗：
      那多半不是这份引擎的来源，按它导出来的东西会**名字对、内容错** ✓）。
    返回 `(Path | None, 说明文本)` ✓。
    """
    raw = str(weights or "").strip()
    if not raw:
        return None, "还没选权重（先在「权重」那一栏选一个 .pt 或 .engine）"
    p = Path(raw)
    if p.suffix.lower() == ".pt":
        return (p, "") if p.exists() else (None, "权重文件不存在：%s" % p)
    if p.suffix.lower() == ".engine":
        cand = p.with_suffix(".pt")
        if cand.exists():
            return cand, ""
        # ⭐⭐⭐⭐⭐ **带尺寸的引擎**（`<pt名>_<尺寸>.engine` ✓ = 本模块自己导出的默认名 ✓）：
        #   去掉结尾的 `_数字` 再找一次 ✓（`detect_v5_960.engine` ⇒ `detect_v5.pt` ✓）。
        import re
        _base = re.sub(r"_\d+$", "", p.stem)
        if _base and _base != p.stem:
            cand2 = p.with_name(_base + ".pt")
            if cand2.exists():
                return cand2, ""
            return None, ("找不到 %s —— 引擎是从一份 .pt 导出来的，同目录那份不在"
                          "（找过 %s 和 %s）。\n请把该 .pt 放回 %s，或者把「权重」改选成 .pt 再导。"
                          % (cand2.name, cand.name, cand2.name, p.parent))
        return None, ("找不到 %s —— 引擎是从一份 .pt 导出来的，同目录同名的那个不在。\n"
                      "请把该 .pt 放回 %s，或者把「权重」改选成 .pt 再导。"
                      % (cand.name, p.parent))
    return None, "认不出这个权重类型：%s（要 .pt 或 .engine）" % p.name


def export_for(weights, imgsz, half=True, device=0, batch=1):
    """按「当前在用的权重」导出**带尺寸**的引擎 ✓（GUI 那个按钮用 ✓）。

    ⚠ 为什么输出名要**带上尺寸**（`<stem>_<imgsz>.engine`）：ultralytics 默认写成
      `<stem>.engine` ⇒ **换个尺寸导出就把原来那个覆盖了** ✗（640 / 960 没法共存 ✓）。
      带上尺寸之后：几个尺寸**互不覆盖** ✓；而实时挑权重是**按 mtime 取最新的 `.engine`**
      ✓（`gui/live_panel.py` 那段 ✓）⇒ 刚导出的那个**自动被选中** ✓。
    ⚠ 导出过程 ultralytics 会先写 `<stem>.engine`（覆盖旧的 ✓）⇒ 若实时正开着、那个文件
      被占用，Windows 会直接报 `PermissionError` ✓ ⇒ **调用方该先提示"停实时"** ✓
      （GUI 的确认框里写了 ✓）。
    返回最终引擎路径（`str`）✓。
    """
    import os

    pt, note = pt_for(weights)
    if pt is None:
        raise FileNotFoundError(note)

    out = Path(export(str(pt), imgsz=int(imgsz), half=bool(half),
                      device=device, batch=int(batch)))
    dst = out.with_name("%s_%d.engine" % (Path(pt).stem, int(imgsz)))
    if out != dst and out.exists():
        try:
            os.replace(str(out), str(dst))       # 同名尺寸重复导出 ⇒ 覆盖它自己 ✓（对 ✓）
        except Exception as e:                   # noqa: BLE001
            raise RuntimeError("导出的引擎改名失败（%s → %s）：%s"
                               % (out.name, dst.name, e))
        print("（已改名为带尺寸的名字：%s）" % dst.name)
    print("✓ 完成：%s" % dst)
    return str(dst)


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
