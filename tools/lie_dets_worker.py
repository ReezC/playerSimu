# -*- coding: utf-8 -*-
"""测谎演示/实机用的**常驻检测子进程**：stdin 收 JPEG 帧、stdout 吐检测 JSON。

为什么必须是独立进程（2026-09-30 ✓ 既有结论）：**"先 Qt 后 torch"的进程里
c10.dll 初始化必炸**（WinError 1114 ✗）—— 工作台/演示窗这类 Qt 进程里**不能**
进程内 import ultralytics。常驻（不是每帧起一个 ✗ 那每次要 ~2s 启动）：
    · 帧协议：`4 字节大端长度 + JPEG 字节` 循环；`长度 == 0` 表示收工 ✓；
    · 输出协议：每帧一行 JSON `[[cls, cx, cy, w, h, conf], ...]`（**输入帧像素坐标** ✓
      置信度 2026-09-30 加 ✓ 演示窗要写"类别名 + 置信度" ✓）。

用法（一般由 `tools/lie_demo.py` 拉起，不直接跑）：
    python -m tools.lie_dets_worker --weights <best.pt> [--conf 0.25] [--imgsz 960]
    python -m tools.lie_dets_worker --names        # 打印类别名 JSON 后退出（一次性 ✓）
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def resolve_device(spec):
    """把 `--device` 的取值落成 **ultralytics 认的那个字符串** ✓（用户 2026-10-07「甲」✓）。

    ⭐⭐⭐⭐⭐ **2026-10-07「甲」那轮加的**（用户选了"甲 ＋ 乙" ✓ 起因："**我们这套算法性能够实时
      使用吗？**" ✓）：原话里没有设备这回事 ✗ —— 是**实测**量出来的（**RTX 5070** ✓ torch2.11+cu128
      ✓ 权重才 **5.4MB** ✗）：`model.device` = **cpu** ✗ ⇒ 每帧 **28.6ms** ✗，而且 960→640→512 几乎
      不变（28.6→26.4→24.8 ✓）**正好证明瓶颈不在算力、在"跑错了设备"** ✓。
    `auto` ⇒ 有 CUDA 就 **"0"** ✓、否则 **"cpu"** ✓（没有卡也不许崩 ✗）；别的取值**原样透传** ✓
      （`"cpu"` / `"0"` / `"0,1"` … ✓ —— A/B 与自检就靠它 ✓）。
    ⚠ 抽成函数的**唯一理由 = 可钉** ✗（否则那条"默认上 GPU"只能靠"起个 worker 读 stderr"去碰 ✓
      —— 那要加载模型 ~10s ✗ 太贵 ✓）：见 `selftest_lie_motion.test_dets_worker_prefers_gpu` ✓。
    """
    _s = str(spec or "auto")
    if _s != "auto":
        return _s
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:                        # noqa: BLE001 —— 没有 torch 就当 CPU ✓
        return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str,
                    default=str(ROOT / "datasets" / "runs" / "detect" / "v2"
                                / "weights" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="auto",
                    help="推理设备：auto（有 CUDA 就 0 ✓ 否则 cpu ✓）/ 0 / cpu …")
    ap.add_argument("--names", action="store_true",
                    help="只打印模型的类别名（JSON 数组 ✓）后退出 ✓")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    if args.names:
        nm = getattr(model, "names", None) or {}
        if isinstance(nm, dict):
            out = [str(nm[k]) for k in sorted(nm, key=lambda x: int(x))]
        else:
            out = [str(x) for x in nm]
        print(json.dumps(out, ensure_ascii=False))
        return 0
    # ⭐⭐⭐⭐⭐ **2026-10-07「甲」那轮：推理上 GPU** ✗✗（用户选了"甲 ＋ 乙" ✓ 起因："**我们这套
    #   算法性能够实时使用吗？**" ✓）—— ⚠⚠ **实测**（**RTX 5070** ✓ torch2.11+cu128 ✓ 权重才
    #   **5.4MB** ✗）：原来 `model.device` = **cpu** ✗ ⇒ 每帧 **28.6ms** ✗，而且 960→640→512
    #   几乎不变（28.6→26.4→24.8 ✓）**正好证明瓶颈不在算力** ✓ —— 是**跑在 CPU 上** ✓。
    #   ⇒ 默认 **auto = 有 CUDA 就上 0** ✓（没有就老实回 cpu ✓，不许因为没卡而崩 ✗）。
    #   ⚠⚠ **绝不往 stdout 打东西** ✗✗：那条是"每帧一行 JSON"的协议 ✓（见文件头 ✓）⇒ 日志一律
    #     stderr ✓（`lie_demo.DetsWorker` 那边把 stderr 丢掉了 ✓ ⇒ 不干扰，也不刷屏 ✓）。
    args.device = resolve_device(args.device)
    print("dets worker: device=%s ｜ imgsz=%d ｜ conf=%.2f"
          % (args.device, args.imgsz, args.conf), file=sys.stderr)
    #   ⚠ **预热一次**（CUDA 首次调用含上下文/显存分配 ⇒ 否则第一帧要多等 ~1s ✗ —— 用户看到的就是
    #     "**点开卡一下**" ✗）：拿一块纯色图跑一遍 ✓（结果丢掉 ✓）。
    try:
        model.predict(np.zeros((int(args.imgsz), int(args.imgsz), 3), np.uint8),
                      conf=args.conf, imgsz=args.imgsz, device=args.device,
                      verbose=False)
    except Exception as _e:                  # noqa: BLE001 —— 预热失败也照跑（下面真调用会报 ✓）
        print("dets worker: 预热失败：%s" % _e, file=sys.stderr)
    inp = sys.stdin.buffer
    out = sys.stdout
    while True:
        head = inp.read(4)
        if len(head) < 4:
            break
        n = struct.unpack(">I", head)[0]
        if n == 0:
            break
        buf = b""
        while len(buf) < n:                      # 读满一帧（管道会分片 ✓）
            chunk = inp.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            out.write("[]\n")
            out.flush()
            continue
        res = model.predict(img, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
        boxes = [[int(b.cls[0])] + [float(v) for v in b.xywh[0]]
                 + [float(b.conf[0])] for b in (res.boxes or [])]
        out.write(json.dumps(boxes) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
