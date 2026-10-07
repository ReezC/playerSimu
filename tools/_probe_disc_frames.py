# -*- coding: utf-8 -*-
"""只读取证：从 `datasets/records/断线重连素材.mp4` 抽关键帧 + 叠坐标网格。

目的：把「点服务器 / 点频道」要用到的**目标坐标**从"估算值"变成"量出来的值"
（docs/断线重连设计.md 附录 A 那几条自己写着"估算值，需复核"）。

**只读**：不改任何配置、不按键、不写项目文件；产物落在 `data/_disc_probe/`
（临时取证目录，`data/` 本来就 gitignore）。

用法：
    python -X utf8 -m tools._probe_disc_frames            # 默认几个时间点
    python -X utf8 -m tools._probe_disc_frames --t 12.0 15.0
    python -X utf8 -m tools._probe_disc_frames --crop 700,380,620,430 --t 15.0
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO = ROOT / "datasets" / "records" / "断线重连素材.mp4"
OUT = ROOT / "data" / "_disc_probe"


def _grid(img, step=100, big=500, thick=1, origin=(0, 0)):
    """叠一层坐标网格（每 step 一条细线、每 big 一条粗线 + 标数字）。

    `origin` = 这张图左上角在原帧里的绝对坐标 ⇒ 裁图也能**直接读出原帧坐标** ✓
    （不写它的话，裁图里的数字是裁图自己的相对坐标 = 量出来没法用 ✗）。
    """
    out = img.copy()
    h, w = out.shape[:2]
    ox, oy = int(origin[0]), int(origin[1])
    for x in range(0, w, step):
        c = (0, 0, 255) if (x + ox) % big == 0 else (0, 200, 200)
        cv2.line(out, (x, 0), (x, h - 1), c, 2 if (x + ox) % big == 0 else thick)
    for y in range(0, h, step):
        c = (0, 0, 255) if (y + oy) % big == 0 else (0, 200, 200)
        cv2.line(out, (0, y), (w - 1, y), c, 2 if (y + oy) % big == 0 else thick)
    for x in range(0, w, big):
        for y in range(0, h, big):
            cv2.putText(out, "%d,%d" % (x + ox, y + oy), (x + 4, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, "%d,%d" % (x + ox, y + oy), (x + 4, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def grab(times, crop=None, zoom=1):
    """按时间点抓帧（顺序读、不用 set/retrieve ✓ 那样不推进帧号会重复拿到同一帧 ✓）。"""
    if not VIDEO.exists():
        print("[X] 素材不存在：%s" % VIDEO)
        return 1
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print("[X] 打不开素材：%s" % VIDEO)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("素材 %dx%d %.3f fps 共 %d 帧（%.2fs）" % (w, h, fps, n, n / fps))
    OUT.mkdir(parents=True, exist_ok=True)
    want = sorted(set(float(t) for t in times))
    i = 0
    done = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / fps
        for wt in want:
            if wt in done:
                continue
            if abs(t - wt) < 1.0 / fps / 2:
                done.append(wt)
                vis = fr
                tag = ""
                if crop:
                    x, y, cw, ch = crop
                    vis = fr[y:y + ch, x:x + cw].copy()
                    tag = "_crop%d_%d" % (x, y)
                    # 裁图里标出"裁自哪里"的绝对坐标（叠加时用 20px 网格 ✓）
                    vis = _grid(vis, step=20, big=100, origin=(x, y))
                    if zoom > 1:
                        vis = cv2.resize(vis, None, fx=zoom, fy=zoom,
                                         interpolation=cv2.INTER_NEAREST)
                        tag += "_x%d" % zoom
                else:
                    vis = _grid(vis, step=100, big=500)
                p = OUT / ("f%05.2f%s.png" % (wt, tag))
                cv2.imwrite(str(p), vis)
                print("  已存 %s（%s）" % (p, "crop" if crop else "整帧"))
        i += 1
    cap.release()
    miss = [t for t in want if t not in done]
    if miss:
        print("[!] 没抓到这些时间点（超出片长？）：%s" % miss)
    print("产物目录：%s" % OUT)
    return 0


def _read_at(t):
    """读某一秒的帧（顺序读 ✓ 同 `grab` 的理由）。"""
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    i = 0
    out = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if abs(i / fps - t) < 1.0 / fps / 2:
            out = fr
            break
        i += 1
    cap.release()
    return out


def analyze_bars(t, region):
    """**数值取证**：在画面某块区域里找"彩色状态条"，聚类出行/列 ⇒ 打印中心坐标。

    为什么不用肉眼读：面板里的格子边界和彩色条只差十几像素，肉眼在缩放图上读
    偏差就有 ±30px（已经吃过一次亏 ✓）。彩色条**饱和度极高**、面板底色是米黄
    （低饱和）⇒ 用 HSV 饱和度掩模能干净地分出条来，连通域的外接矩形就是格子。
    """
    fr = _read_at(t)
    if fr is None:
        print("[X] 读不到 t=%.2fs 的帧" % t)
        return 1
    x, y, w, h = region
    sub = fr[y:y + h, x:x + w]
    # ⚠ 一开始用「HSV 饱和度高」当判据 —— **没用** ✗：面板底色是米黄（实测 S 中位
    #   就有 88、整块都过阈）⇒ 掩模连成一片，只找到一个整幅的大框 ✓（踩过）。
    #   真正的判据是**通道间的关系**：状态条是纯绿（`3ffc83`：G 远大于 R/B），
    #   而底色米黄（`99cae9`：R 最大）、木框是灰蓝（`a2b9cc`：R 最大）——
    #   所以「**G 明显压过 R 和 B**」才只剩条 ✓。
    b, g, r = sub[:, :, 0].astype(np.int16), sub[:, :, 1].astype(np.int16), \
        sub[:, :, 2].astype(np.int16)
    mask = ((g > 120) & (g > r + 40) & (g > b + 40)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = res[0] if isinstance(res, tuple) else res
    boxes = []
    for c in cs:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 20 or not (3 <= bh <= 24):
            continue
        boxes.append((bx + x, by + y, bw, bh))
    if not boxes:
        print("[!] 没找到彩色条（区域选对了吗？%s）" % (region,))
        return 1
    boxes.sort(key=lambda b: (b[1] + b[3] / 2, b[0]))
    # 按 y 中心分行（同一行的条 y 中心相差 < 半行高）
    rows, cur = [], []
    for b in boxes:
        cy = b[1] + b[3] / 2.0
        if cur and abs(cy - (cur[-1][1] + cur[-1][3] / 2.0)) > 14:
            rows.append(cur)
            cur = []
        cur.append(b)
    if cur:
        rows.append(cur)
    print("t=%.2fs 区域 %s：共 %d 条彩色条、归成 %d 行" % (t, region, len(boxes), len(rows)))
    for ri, row in enumerate(rows):
        row.sort(key=lambda b: b[0])
        ys = [b[1] + b[3] / 2.0 for b in row]
        print("  第%d行  y中心中位 %.1f ⇒ 比例 %.4f（共 %d 条）"
              % (ri + 1, sorted(ys)[len(ys) // 2],
                 sorted(ys)[len(ys) // 2] / 1080.0, len(row)))
        for b in row:
            cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
            print("      x %4d..%4d（宽%3d） y %4d..%4d  中心 (%6.1f,%6.1f)"
                  "  比例 (%.4f, %.4f)"
                  % (b[0], b[0] + b[2], b[2], b[1], b[1] + b[3], cx, cy,
                     cx / 1920.0, cy / 1080.0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, nargs="*",
                    default=[12.0, 12.2, 15.0, 15.4, 16.6, 19.5],
                    help="要抓的时间点（秒）")
    ap.add_argument("--bars", action="store_true",
                    help="数值取证模式：在 --crop 那块里找彩色状态条并打印行/列中心")
    ap.add_argument("--region", default="660,480,640,300",
                    help="--bars 的搜索区域：x,y,w,h")
    ap.add_argument("--crop", default="",
                    help="只裁一块：x,y,w,h（给了就叠 20px 细网格）")
    ap.add_argument("--zoom", type=int, default=1, help="裁图放大倍数（1 = 原样）")
    a = ap.parse_args()
    crop = None
    if a.crop:
        crop = tuple(int(v) for v in a.crop.split(","))
        if len(crop) != 4:
            print("[X] --crop 要四个数：x,y,w,h")
            return 2
    if a.bars:
        reg = tuple(int(v) for v in a.region.split(","))
        if len(reg) != 4:
            print("[X] --region 要四个数：x,y,w,h")
            return 2
        rc = 0
        for t in a.t:
            rc |= analyze_bars(t, reg)
        return rc
    return grab(a.t, crop, a.zoom)


if __name__ == "__main__":
    sys.exit(main())
