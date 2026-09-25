"""推理延迟探针：判「推理慢」到底是谁的锅 —— 先看本机负载，再看前向本身。

**为什么要有它**（2026-09-25 那轮调查的结论）
    perf.log 里同一份 imgsz=800 配置，`infer_ms` 从 8.75 ms（13:45）涨到
    稳定 27~34 ms（16:04 之后）；而 GPU 利用率只有 18%、时钟 2790~2955 of
    3090 MHz、功耗 58~68 of 250 W、`clocks_throttle_reasons=0x0` ——
    **不是降频、不是输入尺寸、不是配置**。
    剩下的嫌疑只有两个，而且指纹一样（**前向耗时与输入尺寸无关**：
    64×64 与 800×800 都量到 ~30 ms，像素差 160 倍）：

      甲、**每次 op 的固定开销**（单 op 27~31 µs、提交 19.8 µs/op，
          而真计算很便宜：800×800 卷积 191 µs）。两个进程抢 GPU 时
          每个 op 的**完成**延迟会变长，所以看起来就是纯固定开销。
      乙、**整机内存压力**。每次 op 都有主机侧的内存分配/映射，物理内存
          见底之后这部分被整体放大 —— 同样是「与输入尺寸无关地变慢」。

    这两个都不是猜的：`--load-only` 那一节就是来读甲/乙的直接证据。

判据（B 机，**先把实时预览和别的标注/训练任务都停掉**再跑）
    看**绝对值**：全图 imgsz=800 的中位要落在 7~13 ms（基线 8.75~11）。
    脏环境下它会涨到 24~34 ms —— 那是环境，不是模型。
    **不要**拿 64×64 去要求「~1 ms」：实测干净机上它也是 10.4 ms，因为
    `model.predict()` 每次调用有 ~9~10 ms 与输入尺寸无关的固定开销。
    64×64 的正确用法是**对照**：
      · 它与全图都落在基线内        → 正常（二者本就该相等）
      · 两者一起涨到 24~34 ms       → 环境（先看第 0 节）
    只有 64×64 明显比全图还慢，才是另一类问题。

    机器不干净时本探针会在结论里直接标「这次的数不可信」—— 别拿它下判断。

只读：不改任何配置、不写 perf.log、不碰网络与推流。
imgsz 和 half 都不动 —— 前者在当前状态下实测无收益，后者反而更慢且已弃用。

跑法：
    python -m tools.infer_probe --load-only       # 只看负载，不加载模型（几乎不占资源）
    python -m tools.infer_probe                   # 完整跑：负载 + 前向 + 裸 op
    python -m tools.infer_probe --n 50            # 多采几次
"""

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ 判据
#
# **判据是绝对值，不是「64×64 应该很快」。** 2026-09-25 在干净的 B 机上实测
# （预览与训练都停、可用内存 15 GB）：64×64 前向 10.39 ms、全图 imgsz=800
# 10.51 ms —— 两者几乎相等，且全图正好回到记忆里的 8.75~11 ms 基线。
# 原因是 `model.predict()` 每次调用都有 ~9~10 ms **与输入尺寸无关**的固定
# 开销（letterbox / NMS / Results 构造），64×64 也躲不掉。所以「64×64 应该
# 掉到 ~1 ms」这个判据本身是错的；真正的异常信号是**两者一起涨到 24~34 ms**。
FULL_BAND_MS = (7.0, 13.0)      # imgsz=800 全图：基线 8.75~11，干净机 10.5
TINY_BAND_MS = (7.0, 15.0)      # 64×64 前向：干净机实测 10.4（不是 1 ms）
TINY_REF_MS = 30.0              # 脏环境下量到的值，只在报告里做对照
FULL_REF_MS = 24.7
CLEAN_AVAIL_GB = 4.0    # 可用物理内存低于它，这次的数就不该信


# ---------------------------------------------------------------- 本机负载

def _gpu_stats():
    """GPU 利用率与显存。返回 (util%, used MiB, total MiB) 或 None。

    **为什么它不在 core/machineload.py 里**：`nvidia-smi` 一次要 100~200 ms，
    而那边每 10 秒要采一次、结果还要进实时面板的状态行 —— 放进去就成了自己
    污染自己要量的东西。所以只有这种一次性旁路工具才查 GPU。

    也**故意不列「谁在占卡」**：Windows(WDDM) 下 `nvidia-smi` 的 compute-apps
    会把所有挂着 D3D 上下文的进程都列出来（explorer、开始菜单、搜索、输入法…），
    而且 used_memory 全是 N/A —— 它区分不出谁在真算 GPU，列出来只会让人以为
    "关掉浏览器就好了"。真要判「有没有人在算」，看这个利用率。
    """
    import shutil
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    parts = (r.stdout or "").strip().splitlines()
    if not parts:
        return None
    try:
        u, used, total = [x.strip() for x in parts[0].split(",")]
        return float(u), float(used), float(total)
    except Exception:
        return None


def report_load():
    """打印本机负载，返回它是否「干净」。

    内存与并行子进程的清点**不在这里重写**，走 core/machineload.py ——
    实时预览面板的状态行用的是**同一份**判据。两边各写一份的话，迟早出现
    「工具说干净、面板在告警」这种自相矛盾，而那种矛盾比没有检查更糟。
    """
    from core import machineload
    d = machineload.probe()
    clean = True

    av, tot = d["avail_gb"], d["total_gb"]
    if av is None:
        print("   物理内存   （读不到，跳过）")
    else:
        flag = ""
        if av < CLEAN_AVAIL_GB:
            flag = "   ← **见底了**：每次 op 的主机侧分配会被整体放大"
            clean = False
        print("   物理内存   可用 %.1f / 总 %.1f GB%s" % (av, tot, flag))

    if not d["has_psutil"]:
        print("   并行子进程 （没装 psutil，清不出来；pip install psutil 就能看）")
    elif d["workers"]:
        print("   并行子进程 %d 个（合计 %s；pid %s）"
              % (d["workers"], machineload.fmt_mb(d["worker_mb"]),
                 ", ".join(str(p) for p in d["pids"][:6])))
        print("   ← 训练取数 / 并行标注都会开出它们，每个约 1 GB 且一直在吃 CPU：")
        print("      **等它跑完或取消掉，再复测推理** —— 现在量到的数不干净")
        clean = False
    else:
        print("   并行子进程 0 个")
    if d["app_mb"] is not None:
        print("   工作台      常驻 %.0f MB" % d["app_mb"])

    g = _gpu_stats()
    if g is None:
        print("   GPU         （没有 nvidia-smi，跳过）")
    else:
        u, used, total = g
        print("   GPU         利用率 %.0f%%   显存 %.0f / %.0f MiB"
              % (u, used, total))
        if u >= 70:
            print("   ← 利用率这么高：确实有东西在算卡，复测前先把它停掉")
            clean = False
        else:
            print("   （利用率不高 → 卡没被算满：慢就不是「GPU 算不过来」，"
                  "而是每次 op 的固定开销或内存压力）")

    w = machineload.warn(d)
    if w:
        print("   实时面板会显示：【%s】" % w)     # 同一份判据，两边必须一致
    return clean


# ---------------------------------------------------------------- 取模型/帧

def _pick_weights(project):
    """按**修改时间**取最新的模型（和 gui/live_panel.py 的选法一致）。

    不能按文件名排序：`detect_v1 < mob_v1`，`[-1]` 会选中旧的单类模型。
    """
    if project is None:
        return None
    got = sorted(project.dir_of("models").glob("*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if not got:
        got = sorted(project.dir_of("runs").glob("**/weights/best.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    return got[0] if got else None


def _sample_frame(project):
    """优先用真实帧（② 抽帧产物，和实时流同尺寸同内容分布）。

    合成零图也能量前向，但后处理（NMS、画框）在真实图上才有意义。
    """
    if project is not None:
        got = sorted(project.dir_of("frames").glob("*.png"))
        if got:
            try:
                import cv2
                img = cv2.imread(str(got[0]))
                if img is not None:
                    return img, got[0].name
            except Exception:
                pass
    import numpy as np
    return np.zeros((768, 1366, 3), dtype=np.uint8), "合成 1366x768 零图"


# ------------------------------------------------------------------ 计时

def bench(fn, n, warmup):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000.0)
    xs.sort()
    return {"中位": statistics.median(xs),
            "最小": xs[0],
            "p95": xs[min(len(xs) - 1, int(len(xs) * 0.95))]}


def line(name, r, ref=None):
    tail = "" if ref is None else "   （慢的时候约 %.1f ms）" % ref
    print("   %-24s 中位 %8.2f ms   最小 %8.2f   p95 %8.2f%s"
          % (name, r["中位"], r["最小"], r["p95"], tail))


# --------------------------------------------------------- 裸 torch op 对比

def raw_ops():
    """把「每次 op 的固定开销」和「真计算」分开量。

    固定开销主导时，64×64 与 800×800 的**单 op** 耗时几乎一样，
    因为时间花在 CUDA 下发/取结果（还有主机侧分配）上，不在乘加里。
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        print("   （没有 torch，跳过：%s）" % e)
        return None
    if not torch.cuda.is_available():
        print("   （没有可用 CUDA，跳过）")
        return None

    dev = "cuda"
    out = {}
    for size in (64, 800):
        x = torch.randn(1, 3, size, size, device=dev)
        w = torch.randn(32, 3, 3, 3, device=dev)
        for _ in range(20):
            F.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            F.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        out[size] = (time.perf_counter() - t0) / 50 * 1e6      # µs/op

    # 1 元素的加法：真计算可以忽略，量到的就是「一次 op 的下发+完成」
    a = torch.zeros(1, device=dev)
    b = torch.zeros(1, device=dev)
    for _ in range(20):
        torch.add(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        torch.add(a, b)
    torch.cuda.synchronize()
    out["提交"] = (time.perf_counter() - t0) / 200 * 1e6

    print("   conv2d 3→32  64×64          %8.1f µs/op" % out[64])
    print("   conv2d 3→32 800×800         %8.1f µs/op" % out[800])
    print("   1 元素加法（纯下发+完成）   %8.1f µs/op" % out["提交"])
    print("   → 64 与 800 的单 op 之比 %.2fx（真计算应约 150x；"
          "接近 1 就是固定开销主导）" % (out[800] / max(out[64], 1e-9)))
    return out


def speed_breakdown(model, img, sz, conf, device):
    """ultralytics 自己报的 preprocess / inference / postprocess 毫秒。

    这一行直接回答「那 ~10 ms 花在哪」：如果 `inference` 只有 2~3 ms、其余全在
    pre/post 上，那么换成 TensorRT 引擎（省的是 inference）也压不下去多少 ——
    得先动前后处理这条重路。反过来，如果 inference 占大头，TensorRT 才值得做。
    """
    try:
        r = model.predict(img, conf=conf, imgsz=sz, device=device,
                          verbose=False)[0]
        sp = getattr(r, "speed", None) or {}
        return {k: float(v) for k, v in sp.items()}
    except Exception as e:                      # 老版本没有 speed 也不该炸
        return {"错误": e}


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser(description="推理延迟探针（只读，不写配置）")
    ap.add_argument("--weights", default="",
                    help="默认取当前项目 models/ 里最新的 .pt")
    ap.add_argument("--imgsz", type=int, default=0, help="默认读 config/live.yaml")
    ap.add_argument("--device", default="", help="默认读 config/live.yaml")
    ap.add_argument("--n", type=int, default=30, help="每组采样次数")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--load-only", action="store_true",
                    help="只看本机负载，不加载模型（几乎不占资源）")
    ap.add_argument("--skip-raw", action="store_true", help="跳过裸 torch op 对比")
    a = ap.parse_args()

    from tools.config import load_live
    live = load_live()
    imgsz = a.imgsz or int(live.get("imgsz", 800))
    device = a.device or str(live.get("device", "0"))
    conf = float(live.get("conf_mob", 0.45))

    print("推理延迟探针")
    print("\n== 0. 本机负载（慢的真正常在这里）==")
    clean = report_load()

    if a.load_only:
        if clean:
            print("\n   本机是干净的 —— 可以跑完整探针（去掉 --load-only）")
        else:
            print("\n   本机不干净 —— 先把上面的占用者停掉再跑完整探针，")
            print("   否则量到的 64×64 前向会一样慢，会把结论带偏到「模型有毛病」上。")
        return 0

    from gui.project import last_opened
    project = last_opened()
    w = Path(a.weights) if a.weights else _pick_weights(project)
    if not w or not Path(w).exists():
        print("没有可用的权重 —— 先完成 ⑦ 训练，或用 --weights 指定")
        return 1

    frame, fname = _sample_frame(project)
    print("\n   权重   %s" % w)
    print("   项目   %s" % (project.root.name if project is not None
                            else "（没有最近项目）"))
    print("   帧     %s  %dx%d" % (fname, frame.shape[1], frame.shape[0]))
    print("   参数   imgsz=%d  device=%s  conf=%.2f  采样 %d 次（预热 %d）"
          % (imgsz, device, conf, a.n, a.warmup))

    print("\n== 1. ultralytics 前向（真实调用路径：model.predict）==")
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("   没有 ultralytics：%s" % e)
        return 1

    t0 = time.perf_counter()
    model = YOLO(str(w))
    print("   模型加载 %.1f s" % (time.perf_counter() - t0))

    import numpy as np
    tiny = np.zeros((64, 64, 3), dtype=np.uint8)

    def run(img, sz):
        return lambda: model.predict(img, conf=conf, imgsz=sz,
                                     device=device, verbose=False)[0]

    r_tiny64 = bench(run(tiny, 64), a.n, a.warmup)          # 真·64×64 前向
    line("64×64 @imgsz=64", r_tiny64, TINY_REF_MS)
    r_full = bench(run(frame, imgsz), a.n, a.warmup)        # 实际跑的那条路
    line("全图 @imgsz=%d" % imgsz, r_full, FULL_REF_MS)
    if a.imgsz == 0:
        # 64×64 的图喂给 imgsz=800 会被 letterbox 放大，计算量回到全图 ——
        # 这一组与上一组的差，正好把「前后处理包装」这层单独暴露出来。
        line("64×64 @imgsz=%d(放大)" % imgsz, bench(run(tiny, imgsz), a.n, a.warmup))

    print("\n== 1b. 这些毫秒花在哪（ultralytics 自报，单位 ms）==")
    for tag, im, sz in (("64×64 @imgsz=64", tiny, 64),
                        ("全图 @imgsz=%d" % imgsz, frame, imgsz)):
        sp = speed_breakdown(model, im, sz, conf, device)
        print("   %-22s %s" % (tag, "  ".join("%s %.2f" % (k, v)
                                             for k, v in sp.items())))

    print("\n== 2. 裸 torch op（固定开销 vs 真计算）==")
    raw = None
    if a.skip_raw:
        print("   跳过了（--skip-raw）")
    else:
        raw = raw_ops()

    print("\n== 3. 结论 ==")
    lo_f, hi_f = FULL_BAND_MS
    lo_t, hi_t = TINY_BAND_MS
    ok_full = lo_f <= r_full["中位"] <= hi_f
    ok_tiny = lo_t <= r_tiny64["中位"] <= hi_t
    print("   全图 imgsz=%-4d %.2f ms（基线 %.0f~%.0f；记忆值 8.75~11）  %s"
          % (imgsz, r_full["中位"], lo_f, hi_f, "达标" if ok_full else "**不达标**"))
    print("   64×64 前向    %.2f ms（基线 %.0f~%.0f；干净机实测 10.4）  %s"
          % (r_tiny64["中位"], lo_t, hi_t, "达标" if ok_tiny else "**不达标**"))
    if not clean:
        print("   ⚠ 本机不干净（见第 0 节）：**这次的数不可信**，别照它下结论。")
    if ok_full and ok_tiny:
        print("   → 推理正常，与记忆里的基线一致。")
        print("     日志里那些 24~34 ms 是环境造成的（同时跑训练/标注 + 内存见底），")
        print("     与模型和输入尺寸都无关 —— 别为此去动 imgsz 或模型。")
    elif not ok_full and not ok_tiny:
        print("   → 全图和 64×64 **一起**慢：这是环境，不是模型。")
        print("     按第 0 节把占用者停掉再测；别拿这个数去下「该上 TensorRT」的结论。")
    elif not ok_full:
        print("   → 只有全图慢：增量来自真算全图的那部分（imgsz/批大小），"
              "与固定开销无关。")
    else:
        print("   → 只有 64×64 慢：与尺寸无关的那层开销被放大了，先看第 0 节的负载。")
    if ok_full and not ok_tiny and r_tiny64["中位"] > hi_t:
        print("     （64×64 高出基线而全图正常，通常是采样抖动或别的东西在抢；"
              "先重跑一次 --n 50 确认。）")
    if raw is not None:
        print("   裸 op：64×64 %.1f µs 对 800×800 %.1f µs，比值 %.2fx"
              % (raw[64], raw[800], raw[800] / max(raw[64], 1e-9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
