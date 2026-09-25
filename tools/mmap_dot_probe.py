"""黄点取证（**只读**）：把「玩家标记认不认得出」这件事量出来，给出能回贴的结论。

**为什么要有它**：玩家标记只有几个像素，而两条来源的画质差一个数量级 ——

    stream   A 机独立小地图推流（原始像素、JPEG q100，`MiniMapClient`）
    live     从实时画面里裁一块（**H.264 压过的**，= 工作台「小地图来源 → 从实时
             画面框选」，`gui/live_panel.LiveFrameRegionClient`）

「live 那条到底认不认得出」正是 S3 的门槛判据（`perception/minimap.py` 文件头与
`docs/寻路设计.md` 都标着"待实测"）。这里不猜：同一块面板、同一套识别器，两条来源
各跑一遍，把「命中多少像素 / 几个候选 / 几帧确认」摆出来。

**它不改任何东西**：不按键、不写配置、不碰 datasets/ 与 projects/。导出只落在
`data/dot_probe/`（`--out` 可改）。

跑法：
    # 跟游戏一起跑（先在 A 机把两条推流都起起来）
    python -m tools.mmap_dot_probe --src both --frames 30

    # 只量一条
    python -m tools.mmap_dot_probe --src live
    python -m tools.mmap_dot_probe --src stream

    # 离线复盘（随便一张存下来的画面）
    python -m tools.mmap_dot_probe --image frame_00001.png
    python -m tools.mmap_dot_probe --image frame_00001.png --box 6,72,134,109

`--map <id>` 可选：给了就同时用「底图相减」那一层（有的图底图本身就到处黄褐色，
纯颜色层会淹掉，见 `perception/minimap.find_player_dot`）。
"""

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402

from core import mapdata                                    # noqa: E402
from core.imgio import imread, imwrite                      # noqa: E402
from perception import minimap as mm                        # noqa: E402
from tools import config                                    # noqa: E402

OUT_DEFAULT = ROOT / "data" / "dot_probe"


# ---------------------------------------------------------------- 取帧

def grab_stream(n, seconds):
    """A 机独立小地图推流（原始像素）→ (面板列表, 备注)。"""
    host = config.get("a_host")                 # link.yaml 顶层
    port = config.get("minimap", "port", 5003)
    if not host:
        return [], "link.yaml 里没读到 a_host"
    cli = mm.MiniMapClient(host, port=port, timeout=5.0).start()
    out = []
    deadline = time.time() + seconds
    try:
        while len(out) < n and time.time() < deadline:
            f, _t = cli.latest(clear=True)
            if f is None:
                time.sleep(0.02)
                continue
            out.append(f)
    finally:
        cli.stop()
    note = "收到 %d 帧（%.1f 拍/秒）" % (len(out), cli.fps)
    if not out:
        note += "；err=%s —— A 机的「小地图推流」起了吗？" % (cli.err or "无")
    return out, note


def grab_live(crop, n, seconds):
    """实时画面（H.264 流）裁一块 → (面板列表, 备注)。

    **必须跑在带超时的线程里**：UDP 收不到包时 `av.open/decode` 会一直阻塞，
    主线程就卡住了（这个工具的用途之一恰恰是"没起来"的时候告诉你为什么）。
    """
    try:
        import av
    except ImportError:
        return [], "没装 PyAV（pip install av）"
    url = config.get("stream", "url")
    x, y, w, h = crop
    res = {"out": [], "note": ""}
    t0 = time.time()

    def work():
        try:
            c = av.open(url, mode="r", timeout=5.0)
        except Exception as e:                              # noqa: BLE001
            msg = str(e)
            if "10048" in msg or "address already in use" in msg.lower():
                # 实测最常撞的就是这条：工作台的实时预览正占着这个 UDP 口。
                # 说清楚怎么办，别只丢一个 errno。
                res["note"] = (
                    "打不开实时流 %s：本机这个 UDP 口已被占用（10048）—— "
                    "**工作台的实时预览正开着**。要量这条来源：\n"
                    "      先到「实时」页点「停止」，跑完这个工具再开回来"
                    "（约 20 秒）；或先只量 --src stream" % url)
            else:
                res["note"] = "打不开实时流 %s：%s" % (url, e)
            return
        try:
            st = c.streams.video[0]
            for frame in c.decode(st):
                if len(res["out"]) >= n or time.time() - t0 > seconds:
                    break
                img = frame.to_ndarray(format="bgr24")
                H, W = img.shape[:2]
                if x < 0 or y < 0 or x + w > W or y + h > H:
                    res["note"] = ("框选区域 %s 超出当前画面 %dx%d —— 重新框一次"
                                   % (list(crop), W, H))
                    return
                res["out"].append(np.ascontiguousarray(img[y:y + h, x:x + w]))
        except Exception as e:                              # noqa: BLE001
            if not res["note"]:
                res["note"] = "解码中断：%s: %s" % (type(e).__name__, e)
        finally:
            try:
                c.close()
            except Exception:                               # noqa: BLE001
                pass

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(seconds + 5.0)
    if not res["out"] and not res["note"]:
        res["note"] = "%.0f 秒内没收到帧（%s）" % (time.time() - t0, url)
    return res["out"], res["note"]


def grab_image(path, box):
    """离线：一张图（整块当面板，给了 --box 就先裁）。"""
    img = imread(path)
    if img is None:
        return [], "读不到：%s（中文路径要用 core.imgio，这里已经用了）" % path
    if box:
        x, y, w, h = box
        H, W = img.shape[:2]
        if x + w > W or y + h > H:
            return [], "box %s 超出图片 %dx%d" % (list(box), W, H)
        img = img[y:y + h, x:x + w]
    return [np.ascontiguousarray(img)], "离线单帧 %dx%d" % (img.shape[1], img.shape[0])


# ---------------------------------------------------------------- 量 & 报

def measure(panels, terrain, calib, tag, out_dir, track=True, roi=None):
    """逐帧跑识别器（多帧时带跨帧跟踪）→ 统计 dict。"""
    if track:
        tr = mm.PlayerDotTracker()
        rows = [tr.update(p, calib=calib, terrain=terrain, roi=roi)
                for p in panels]
    else:
        # 离线单帧：**不跨帧**（一拍谈不上"连续确认"），直接给单帧结论
        rows = [dict(mm.find_player_dot(p, calib=calib, terrain=terrain, roi=roi),
                     confirmed=False) for p in panels]

    def med(key, pred=None):
        vals = [r[key] for r in rows if (pred is None or pred(r))]
        return float(np.median(vals)) if vals else None

    ok_rows = [r for r in rows if r["ok"]]
    conf_rows = [r for r in rows if r.get("confirmed")]
    stat = {
        "tag": tag, "n": len(rows), "track": track,
        "panel": (panels[0].shape[1], panels[0].shape[0]) if panels else (0, 0),
        "mask_med": med("mask_px"),
        "dense_med": med("dense_px"),
        "mask_max": max([r["mask_px"] for r in rows], default=0),
        "cand_med": med("candidates"),
        "cand_max": max([r["candidates"] for r in rows], default=0),
        "roi": rows[0]["roi"] if rows else None,
        "flood_frames": sum(1 for r in rows if r["flood"]),
        "ok_frames": len(ok_rows), "conf_frames": len(conf_rows),
        "w_med": med("w", lambda r: r["ok"]), "h_med": med("h", lambda r: r["ok"]),
        "area_med": med("area", lambda r: r["ok"]),
        "xy": ((float(np.median([r["x"] for r in ok_rows])),
                float(np.median([r["y"] for r in ok_rows]))) if ok_rows else None),
        "span": ((float(max(r["x"] for r in ok_rows) - min(r["x"] for r in ok_rows)),
                  float(max(r["y"] for r in ok_rows) - min(r["y"] for r in ok_rows)))
                 if len(ok_rows) > 1 else (0.0, 0.0)),
        "bgr": (tuple(int(v) for v in np.median(
            [r["bgr"] for r in ok_rows], axis=0)) if ok_rows else None),
        "reason": (rows[-1]["reason"] if rows else "没有帧"),
        "layer": rows[-1]["layer"] if rows else "",
    }

    # 证据图：面板 ×4（圈出找到的点）+ 掩码 ×4（颜色层到底看到了什么）
    if panels:
        p = panels[-1]
        z = 4
        big = p.copy()
        for r in rows:
            if r["ok"]:
                cv2_circle(big, r["x"], r["y"], max(4, int(r["w"])), (255, 255, 255))
        vis = cv2_resize(big, z)
        put_text(vis, "%s  %dx%d  frames=%d conf=%d"
                 % (tag, stat["panel"][0], stat["panel"][1], stat["n"],
                    stat["conf_frames"]))
        f1 = out_dir / ("%s_panel.png" % tag)
        imwrite(f1, vis)
        m = mm.dot_mask(p)
        mvis = cv2_resize(np.dstack([m * 255] * 3), z)
        f2 = out_dir / ("%s_mask.png" % tag)
        imwrite(f2, mvis)
        stat["panel_png"], stat["mask_png"] = str(f1), str(f2)
    return stat


def cv2_resize(img, z):
    import cv2
    return cv2.resize(img, (img.shape[1] * z, img.shape[0] * z),
                      interpolation=cv2.INTER_NEAREST)


def cv2_circle(img, x, y, r, color):
    import cv2
    cv2.circle(img, (int(round(x)), int(round(y))), r, color, 2)


def put_text(img, text):
    import cv2
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 5)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)


def verdict(s):
    """一句话判据（**三态**：认得出 / 认不出 / 颜色层不可用）。"""
    if not s["n"]:
        return "没量到帧"
    flood_frac = s["flood_frames"] / float(s["n"])
    conf_frac = s["conf_frames"] / float(s["n"])
    if not s["track"]:                    # 离线单帧：没有"连续确认"这回事
        if s["ok_frames"]:
            return "单帧找到像标记的块（离线模式不跨帧确认，要判稳定性得跑实时）"
        if flood_frac >= 0.5:
            return ("单帧颜色层不可用（命中 %s 像素）—— 换来源，或先做底图相减"
                    % fmt(s["mask_med"]))
        return "单帧没找到像标记的块"
    if conf_frac >= 0.8:
        return "认得出（%.0f%% 的帧连续两拍都确认到点）" % (100 * conf_frac)
    if conf_frac == 0 and s["ok_frames"] == 0:
        if flood_frac >= 0.5:
            return ("颜色层不可用（%.0f%% 的帧命中像素过多：中位 %s）—— "
                    "换来源，或先做底图相减" % (100 * flood_frac, s["mask_med"]))
        return "认不出（一帧都没找到像标记的块）"
    if conf_frac == 0:
        return ("只找到零散候选、没有连续确认（%d/%d 帧有候选）—— 缩小视野/提高阈值再看"
                % (s["ok_frames"], s["n"]))
    return "时好时坏（%.0f%% 的帧确认到点）—— 别直接接到定位上" % (100 * conf_frac)


def show(s):
    print("── %s" % s["tag"])
    print("   面板      %dx%d，量了 %d 帧；搜索区 %s"
          % (s["panel"][0], s["panel"][1], s["n"],
             "整块" if not s["roi"] else list(s["roi"])))
    print("   命中像素  中位 %s（成片 %s），最大 %s（flood 帧 %d/%d）"
          % (fmt(s["mask_med"]), fmt(s["dense_med"]), s["mask_max"],
             s["flood_frames"], s["n"]))
    print("   候选数    中位 %s / 最大 %s"
          % (fmt(s["cand_med"]), s["cand_max"]))
    print("   找到/确认  %d / %d 帧" % (s["ok_frames"], s["conf_frames"]))
    if s["xy"]:
        print("   位置      面板 (%.1f, %.1f)；跨度 %.1f × %.1f 像素"
              % (s["xy"][0], s["xy"][1], s["span"][0], s["span"][1]))
        # 跨度用来分辨「跟着人走的标记」和「不动的地图纹理」：人在走，跨度就该
        # 是几十像素；跨度为 0 而人明明在动 → 抓到的是死纹理。
    if s["w_med"]:
        print("   点的尺寸  中位 %sx%s，%s 像素；颜色 BGR=%s"
              % (fmt(s["w_med"]), fmt(s["h_med"]), fmt(s["area_med"]), s["bgr"]))
    print("   层/原因   %s | %s" % (s["layer"], s["reason"]))
    print("   判据      %s" % verdict(s))
    if s.get("panel_png"):
        print("   证据图    %s\n             %s"
              % (s["panel_png"], s["mask_png"]))


def fmt(v):
    return "-" if v is None else ("%.1f" % v if isinstance(v, float) else str(v))


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser(description="黄点取证（只读）")
    ap.add_argument("--src", default="both",
                    choices=["both", "stream", "live"],
                    help="量哪条来源（默认两条都比）")
    ap.add_argument("--frames", type=int, default=30, help="每条来源量多少帧")
    ap.add_argument("--seconds", type=float, default=20.0, help="每条来源最多等多久")
    ap.add_argument("--image", help="离线模式：直接量一张图")
    ap.add_argument("--box", help="离线模式的裁剪区 x,y,w,h")
    ap.add_argument("--map", dest="map_id", help="用哪张图的地形/标定（可做底图相减层）")
    ap.add_argument("--roi", help="搜索区 x0,y0,x1,y1（面板像素）；不给就按标定"
                                  "自动取「底图覆盖区」，排除面板标题带")
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="证据图落盘目录")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    roi = None
    if a.roi:
        roi = [int(v) for v in a.roi.split(",")]
        if len(roi) != 4:
            print("--roi 要形如 x0,y0,x1,y1")
            return 2

    terrain = None
    if a.map_id:
        terrain = mapdata.load(a.map_id, with_canvas=True)
        print("图 %s：地形 %s" % (a.map_id, "有" if terrain else "没有"))

    tasks = []
    offline = bool(a.image)
    if a.image:
        box = None
        if a.box:
            box = [int(v) for v in a.box.split(",")]
            if len(box) != 4:
                print("--box 要形如 x,y,w,h")
                return 2
        tasks.append(("image", lambda: grab_image(a.image, box)))
    else:
        live = config.load_live()
        crop = live.get("mmap_crop") or []
        if a.src in ("both", "stream"):
            tasks.append(("stream", lambda: grab_stream(a.frames, a.seconds)))
        if a.src in ("both", "live"):
            if len(crop) != 4:
                print("live 来源需要 config/live.yaml 里的 mmap_crop（先在工作台"
                      "「路线识别」里框一次小地图）")
            else:
                tasks.append(("live", lambda: grab_live(crop, a.frames, a.seconds)))

    print("黄点取证（只读）  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    stats = []
    for tag, fn in tasks:
        panels, note = fn()
        print("── %s 取帧：%s" % (tag, note))
        if not panels:
            continue
        # 标定**按来源分开存**（两条来源的面板尺寸差 5.6 倍）：量哪条来源就用哪份
        cal_src = tag if tag in (mm.SRC_STREAM, mm.SRC_LIVE) else None
        calib = mapdata.load_calib(a.map_id, cal_src) if a.map_id else None
        if a.map_id:
            print("   标定（%s）：%s"
                  % (mm.SRC_LABEL.get(cal_src, cal_src or "不限"),
                     "有" if mm.has_geometry(calib) else "没有"))
        s = measure(panels, terrain, calib, tag, out_dir,
                    track=not offline, roi=roi)
        show(s)
        stats.append(s)

    if len(stats) == 2:
        a_, b_ = stats
        print("══ 两条来源对照")
        print("   stream(原始像素) 确认 %.0f%%  命中像素中位 %s"
              % (100.0 * a_["conf_frames"] / max(1, a_["n"]), fmt(a_["mask_med"])))
        print("   live(H.264)      确认 %.0f%%  命中像素中位 %s"
              % (100.0 * b_["conf_frames"] / max(1, b_["n"]), fmt(b_["mask_med"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
