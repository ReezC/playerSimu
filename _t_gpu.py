"""临时：量自动标注的 CPU 真实耗时，并用 torch 在 GPU 上复算一遍（含逐像素比对）。

只读真实项目帧；标注输出写进临时目录（跑完删）。
"""
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                            # noqa: E402
import torch.nn.functional as F                          # noqa: E402

import tools.detect_mobs as dm                          # noqa: E402

PROJ = Path("projects/猴林迷宫I")
FRAMES = sorted((PROJ / "frames").glob("*.png"))
MOBS = ["3230100", "4230101"]
TMP = Path("_t_gpu_out")
N_FRAMES = 12
DS = 2
SCALE = 1.12

print("=== 数据 ===")
print("  帧 %d 张（用前 %d）  %s" % (len(FRAMES), N_FRAMES, FRAMES[0].name if FRAMES else "-"))
img0 = dm.imread(FRAMES[0]) if FRAMES else None
print("  画面 %s" % (None if img0 is None else "%dx%d" % (img0.shape[1], img0.shape[0])))

tpl = dm.load_templates("datasets/sprites/mob", MOBS, 20)
print("  模板 %d 个（%s，含镜像）" % (len(tpl), "+".join(MOBS)))

cfg = {
    "sprites": "datasets/sprites/mob", "mobs": MOBS, "ds": DS, "scale": SCALE,
    "mob_scales": {}, "thr": 0.90, "dist": 0.06, "peaks": 4,
    "min_energy_ratio": 0.35, "per_mob": 20,
    "out": str(TMP / "labels"), "manual_dir": str(TMP / "labels"),
    "vis": False, "vis_dir": str(TMP / "vis"), "region": None,
    "box_color": (0, 255, 0),
}

print("\n=== 1) 单进程 CPU（work()，也就是每个子进程干的活） ===")
dm.init_worker(cfg)
frames = [str(f) for f in FRAMES[:N_FRAMES]]
t0 = time.perf_counter()
for fp in frames:
    dm.work(fp)
dt_single = time.perf_counter() - t0
print("  %d 帧 %.2f s → %.0f ms/帧 → %.2f 帧/秒/进程"
      % (len(frames), dt_single, dt_single / len(frames) * 1000,
         len(frames) / dt_single))

print("\n=== 2) 解码 PNG 的硬成本（GPU 也躲不掉） ===")
t0 = time.perf_counter()
for fp in frames:
    dm.imread(fp)
dt_io = time.perf_counter() - t0
print("  %.0f ms/帧" % (dt_io / len(frames) * 1000))

if not torch.cuda.is_available():
    print("\n没有 CUDA，跳过 GPU 部分")
    raise SystemExit(0)

dev = torch.device("cuda")
print("\n=== 3) GPU 原型：用 torch 卷积复算同一个归一化互相关 ===")
print("  设备 %s" % torch.cuda.get_device_name(0))

img = dm.imread(frames[0])
H, W = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
I = torch.from_numpy(gray)[None, None].to(dev)
I2 = I * I
print("  画面灰度上传完成 %dx%d" % (W, H))


def gpu_response(t_bgr, t_alpha):
    """torch 复算 cv2.matchTemplate(TM_CCORR_NORMED, mask=)：返回 numpy 响应图。"""
    h = int(round(t_bgr.shape[0] * SCALE / DS))
    w = int(round(t_bgr.shape[1] * SCALE / DS))
    t = cv2.resize(t_bgr, (w, h), interpolation=cv2.INTER_AREA)
    m = (cv2.resize(t_alpha, (w, h), interpolation=cv2.INTER_AREA) > 128)
    tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32)
    K = torch.from_numpy((tg * m.astype(np.float32)))[None, None].to(dev)
    M = torch.from_numpy(m.astype(np.float32))[None, None].to(dev)
    e_t = float((tg[m] ** 2).sum())
    num = F.conv2d(I, K)                     # Σ T·I·M
    den = F.conv2d(I2, M)                    # Σ I²·M
    r = num / torch.sqrt(torch.clamp(den, min=1e-6) * e_t)
    return r[0, 0].cpu().numpy(), (w, h)


print("\n  逐模板比对 CPU 与 GPU 的响应图（灰度，同输入同掩码）：")
gpu_ms = []
worst = 0.0
for i, (mid, frame, b, al) in enumerate(tpl[:6]):
    h = int(round(b.shape[0] * SCALE / DS))
    w = int(round(b.shape[1] * SCALE / DS))
    t = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    m = (cv2.resize(al, (w, h), interpolation=cv2.INTER_AREA) > 128).astype(np.uint8) * 255
    tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cpu_r = cv2.matchTemplate(gray, tg, cv2.TM_CCORR_NORMED, mask=m)
    cpu_r = np.nan_to_num(cpu_r, nan=0.0, posinf=0.0, neginf=0.0)
    cpu_r[cpu_r > 1.0] = 0.0

    for _ in range(3):                       # 预热
        gpu_r, _wh = gpu_response(b, al)
    t0 = time.perf_counter()
    for _ in range(10):
        gpu_r, _wh = gpu_response(b, al)
    gpu_ms.append((time.perf_counter() - t0) / 10 * 1000)

    d = float(np.abs(np.nan_to_num(gpu_r, nan=0.0) - cpu_r[:gpu_r.shape[0], :gpu_r.shape[1]]).max())
    worst = max(worst, d)
    cm = float(cpu_r.max())
    gm = float(gpu_r.max())
    print("    %s %-10s cpu 峰 %.4f  gpu 峰 %.4f  最大逐像素差 %.2e"
          % (mid, frame, cm, gm, d))

print("  → 单模板 GPU %.2f ms（含数据在卡上，不含上传）；CPU 单模板 %.2f ms"
      % (sum(gpu_ms) / len(gpu_ms), dt_single / len(frames) * 1000 / len(tpl)))
print("  → 最大逐像素差 %.2e（1e-4 量级以内就说明是同一个公式）" % worst)

# 整帧（全部模板）跑一遍，看 GPU 摊到每帧多少
t0 = time.perf_counter()
for (mid, frame, b, al) in tpl:
    h = int(round(b.shape[0] * SCALE / DS))
    w = int(round(b.shape[1] * SCALE / DS))
    if h < 8 or w < 8 or h >= I.shape[2] or w >= I.shape[3]:
        continue
    gpu_response(b, al)
dt_gpu_frame = time.perf_counter() - t0
print("\n  整帧（%d 个模板全跑一遍，不含上传）%.0f ms → 单进程 CPU 是 %.0f ms"
      % (len(tpl), dt_gpu_frame * 1000, dt_single / len(frames) * 1000))

shutil.rmtree(TMP, ignore_errors=True)
