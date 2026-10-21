"""小地图诊断命令行（2026-10-18 从 `perception.minimap` 搬到 `tools/`）。

    python -m tools.mmap_diag --map <地图id>            # 量一次：面板怎么显示底图
    python -m tools.mmap_diag --map <id> --watch        # 持续看（边跑边报）

⚠ 这里是**工具**不是运行库：生产链路（实时线程 / 界面）不许 import 本模块 ✓
（`gui` 只把这条命令**当提示文字**显示给用户，见 `gui/route_panel.py`）。
"""

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import mapdata                     # noqa: E402
from core.imgio import imwrite               # noqa: E402
from perception import map_calib, map_const  # noqa: E402
from perception.map_client import MiniMapClient  # noqa: E402
from perception.map_overlay import (crop_compare, draw_on_overlay,  # noqa: E402
                                    overlay_zoom, view_rects)
from perception.minimap import locate, locate_crop, locate_fit  # noqa: E402


# ══════════════════════════════════════════════════════════════
# 诊断（B 机跑一次，把面板显示方式量出来）
# ══════════════════════════════════════════════════════════════

def _desc(calib):
    sx, sy = map_calib.scales_of(calib)
    s = ("scale=%.3f" % sx if abs(sy - sx) < 1e-9
         else "scale x=%.3f y=%.3f" % (sx, sy))     # 两轴不同时必须说出来（不然像量过一样）
    if calib["mode"] == map_const.MODE_FIT:
        return "fit  %s offset=%s" % (s, calib["offset"])
    return "crop %s view=%s" % (s, calib["view"])


def _mode_zh(mode):
    """内部值 → 界面上的说法（和「路线识别」下拉里一致）。"""
    return {map_const.MODE_FIT: "全局小地图", map_const.MODE_CROP: "局部小地图"}.get(mode, str(mode))


def diagnose(map_id, host, port=5003, wait=8.0, out=None, mode="saved", save=True):
    t = mapdata.load(map_id, with_canvas=True)
    if t is None or t.canvas is None:
        print("没有 %s 的地形/底图，先导出："
              "WzProbe.exe dump-terrain <WZ目录> datasets/map --only %s"
              % (map_id, map_id))
        return 2

    # **方式由用户自己在工作台里选**（每张图一份，见 datasets/map/<id>.mapcalib.json）：
    # 默认（mode="saved"）就读他选的那个，**不按匹配分自动挑**（项目规范）。
    # 放在连流之前：没选就直接告诉他去选，不必先等一帧。
    cal = mapdata.load_calib(map_id) or {}
    use = cal.get("mode") if mode == "saved" else mode
    if not use:
        print("这张图还没选显示方式 —— 到「路线识别 → 小地图定位」的「显示方式」里选")
        print("（全局小地图 / 局部小地图），选完这里就按选的那个跑。")
        return 2

    cli = MiniMapClient(host, port).start()
    print("连 %s:%s …（等最多 %.0f 秒）" % (host, port, wait))
    frame = None
    t0 = time.time()
    while time.time() - t0 < wait:
        f, _ = cli.latest()
        if f is not None:
            frame = f
            break
        time.sleep(0.2)
    if frame is None:
        print("没收到帧。%s" % (cli.err or "检查 A 机是否在跑 minimap_push、"
                                          "端口是否放行"))
        cli.stop()
        return 1

    fh, fw = frame.shape[:2]
    print("收到一帧 %dx%d（%.1f KB）" % (fw, fh, frame.nbytes / 1024.0))

    # 方式已经在连流之前定好了（use）；匹配分只用来判断"这一次定位可不可信"。
    loc = locate(frame, t.canvas, mode=use)
    if loc is None:
        print("按你选的「%s」没对上 —— 可能：" % _mode_zh(use))
        print("  · A 机框的区域不只是小地图（框多了/框错了）")
        print("  · 推的不是这张图（现在是 %s）" % map_id)
        print("  · 面板被游戏 UI 挡住了一部分")
        # 两种方式的分数只当**诊断线索**（不是让分数决定方式；改方式去工作台改）
        print("  参考：", end="")
        for tag, r in (("全局小地图", locate_fit(frame, t.canvas)),
                       ("局部小地图", locate_crop(frame, t.canvas))):
            print(" %s=%s" % (tag, "没对上" if r is None
                            else "%.3f" % r["score"]), end="")
        print("（方式不在这里改 —— 到「路线识别」里换）")
        cli.stop()
        return 1

    sx, sy = map_calib.scales_of(loc)          # 模板匹配只有**一个**尺度 ⇒ 这里两轴必然相同 ✓
    s = sx
    ox, oy = loc["offset"]
    cw, ch = t.canvas.shape[1], t.canvas.shape[0]
    print("采用: %s  匹配分=%.3f" % (_desc(loc), loc["score"]))
    if loc["mode"] == map_const.MODE_FIT:
        print("  底图 %dx%d → 面板里 %dx%d，位置 (%d,%d)"
              % (cw, ch, int(cw * sx), int(ch * sy), ox, oy))
        print("  换算：底图像素 = (面板像素 - 偏移) / %.3f" % s)
    else:
        print("  面板 1:1 显示底图的一块，当前那块从底图的 %s 开始" % (loc["view"],))
        print("  换算：底图像素 = 面板像素 - 偏移 + %s" % (loc["view"],))
    print("  世界坐标 = 底图像素 * %.2f - (%s, %s)"
          % (t.px_per_world, t.mini.get("centerX"), t.mini.get("centerY")))

    # 角点对照：面板里底图的四角 → 世界坐标（应当等于世界范围四角）
    b = t.bounds
    for name, (px, py) in (("左上", (ox, oy)), ("右下", (ox + cw * s, oy + ch * s))):
        wx, wy = map_calib.panel_to_world(px, py, loc, t)
        print("  面板%s角 → 世界 (%.0f, %.0f)" % (name, wx, wy))
    print("  世界范围应为 %s" % (tuple(round(v) for v in b),))

    if save:
        # 只把**量出来的几何**存进去，**方式保持你选的那个**（不覆盖）。
        # ⚠ 走 `with_scales`：这个工具量不出两轴差别（模板匹配只有一个尺度）⇒ 存完
        #   两轴视为相同、顺手清掉旧的 `scale_y`（留着就是一份**改过的**几何里混着
        #   上一轮的 y 缩放 ✗）。
        cal = map_calib.with_scales(cal, sx, sy)
        cal.update({k: loc[k] for k in ("offset", "view", "score")})
        cal["mode"] = loc["mode"]
        mapdata.save_calib(map_id, cal)
        print("已保存标定: %s（显示方式 = %s，你在 GUI 里选的那个）"
              % (mapdata.calib_path(map_id), loc["mode"]))

    p = Path(out) if out else (mapdata.map_dir() / "_minimap_live.png")
    if loc["mode"] == map_const.MODE_CROP:
        # **crop（局部小地图）下"现在显示的是底图哪一块"就是 view** ——
        # 原来这里也画绿框，但画的是"整张底图铺在面板上"，对 crop 是错的
        # （底图比面板大，框会整个跑到画面外）。改成两张真正看得懂的对照图。
        rect_c, rect_o = view_rects(loc, (frame.shape[1], frame.shape[0]), (cw, ch))
        print("  底图上那一块: %s  （面板 %dx%d）"
              % (tuple(rect_c[:2]), frame.shape[1], frame.shape[0]))
        print("  在叠加图上大致是: %s   （底图放大 %d 倍；用过 --dx/--dy 要自己加）"
              % (tuple(int(v) for v in rect_o), overlay_zoom(cw)))
        imwrite(p, crop_compare(frame, t.canvas, rect_c))
        print("已写出 %s（左 = 实时面板，右 = 底图上匹配到的那一块）" % p)
        op = draw_on_overlay(map_id, rect_o)
        if op:
            print("已写出 %s（叠加图上框出「当前是这一块」）" % op)
        else:
            print("（还没有 %s_overlay.png —— 在工作台「路线识别」点一下"
                  "「生成地形图」就有了）" % map_id)
    else:
        vis = frame.copy()
        cv2.rectangle(vis, (ox, oy), (int(ox + cw * s), int(oy + ch * s)),
                      (0, 255, 0), 2)
        cv2.putText(vis, "scale=%.3f off=(%d,%d) score=%.2f"
                    % (s, ox, oy, loc["score"]),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        imwrite(p, vis)
        print("已写出 %s（绿框 = 匹配到的底图，拿去和游戏里的小地图对照）" % p)
    cli.stop()
    return 0


def watch(map_id, host, port=5003, mode=None, seconds=0.0, fps=5.0,
          show=True, out=None):
    """连续看「面板 ↔ 底图」的对应关系（crop / 局部小地图的主力验证手段）。

    每拍：定位一次 → 打印 view / 匹配分 / 与上一拍的**位移** → 更新两张对照图：

        datasets/map/_minimap_live.png        左 = 实时面板，右 = 底图那一块
        datasets/map/<id>_overlay_live.png    叠加图上框出「当前是这一块」

    **为什么非得连续看**：crop 下"现在显示的是底图的哪一块"完全由 matchTemplate
    匹配出来（`view`），单帧结果对错是看不出来的。走两步再看才有判据：
      · 往右走 → view 应该**朝同一个方向**平移（地形图跟着人滑动）；
      · 匹配分不该掉（掉了说明面板被 UI 挡住 / 换图了 / 选错方式）；
      · 位移量级要和走的距离相符（`view` 的单位就是底图像素）。
    按 q（或 Ctrl+C）退出。
    """
    from core.config import get

    t = mapdata.load(map_id, with_canvas=True)
    if t is None or t.canvas is None:
        print("没有 %s 的地形/底图，先导出：" % map_id)
        print("  WzProbe.exe dump-terrain <WZ目录> datasets/map --only %s" % map_id)
        return 2

    # 方式沿用**你在工作台里选的那个**（不按分数自动挑，见项目规范）；
    # 没选就让人先去选 —— 这个工具不该替他决定。
    if mode in (None, "", "saved"):
        mode = (mapdata.load_calib(map_id) or {}).get("mode")
    if not mode:
        print("这张图还没选显示方式 —— 到「路线识别 → 小地图定位」里选"
              "（全局小地图 / 局部小地图）")
        return 2
    if mode != map_const.MODE_CROP:
        print("这张图选的是「%s」—— --watch 是给**局部小地图（crop）**看 view 用的；"
              % _mode_zh(mode))
        print("全局小地图整张都显示，没有「现在是哪一块」这回事。")
        return 2

    host = host or get("a_host")
    cli = MiniMapClient(host, port).start()
    print("连 %s:%s …  方式=%s  %.0f 拍/秒  （q 退出）" % (host, port, mode, fps))
    print("  判据：往右走 → view 也朝右平移；匹配分不掉；位移量级和走的距离相符")

    cw, ch = t.canvas.shape[1], t.canvas.shape[0]
    prev = None                      # 上一拍的底图块左上角
    last_frame = None
    t0 = time.time()
    n = 0
    try:
        while True:
            if seconds and time.time() - t0 >= seconds:
                break
            frame, _ts = cli.latest()
            if frame is None or frame is last_frame:
                time.sleep(0.05)
                continue
            last_frame = frame

            loc = locate(frame, t.canvas, mode=mode)
            if loc is None:
                print("  没对上（面板被挡住？换图了？方式选错？）")
                time.sleep(1.0 / fps)
                continue
            if loc["mode"] != map_const.MODE_CROP:
                print("  ⚠ 这次定位更像「%s」—— 方式要改去「路线识别」里改，"
                      "这里不自动换" % _mode_zh(loc["mode"]))
                break

            rect_c, rect_o = view_rects(loc, (frame.shape[1], frame.shape[0]),
                                        (cw, ch))
            mv = ""
            if prev is not None:
                mv = "  位移 (%+d, %+d)" % (rect_c[0] - prev[0], rect_c[1] - prev[1])
            prev = rect_c[:2]
            n += 1
            print("  [%3d] view=%-16s 匹配分=%.3f  底图块起于 %-14s%s"
                  % (n, tuple(loc["view"]), loc["score"], rect_c[:2], mv))

            imwrite(out or (mapdata.map_dir() / "_minimap_live.png"),
                    crop_compare(frame, t.canvas, rect_c))
            draw_on_overlay(map_id, rect_o)

            if show:
                try:
                    vis = crop_compare(frame, t.canvas, rect_c)
                    cv2.imshow("minimap live (left=panel right=base map)",
                               cv2.resize(vis, None, fx=0.75, fy=0.75))
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                except Exception as e:
                    show = False      # 没有 GUI 后端（headless）→ 只写文件
                    print("  （打不开预览窗口：%s；每拍的对照图仍在写 "
                          "_minimap_live.png）" % e)
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    finally:
        cli.stop()
        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    print("退出（共 %d 拍）" % n)
    return 0


def main() -> int:
    import argparse
    from core.config import get
    ap = argparse.ArgumentParser(description="小地图定位诊断（B 机）")
    ap.add_argument("--map", dest="map_id", default="105090600")
    ap.add_argument("--host", default=None, help="A 机 IP，默认读 link.yaml 的 a_host")
    ap.add_argument("--port", type=int, default=None,
                    help="A 机小地图推流端口，默认读 link.yaml 的 minimap.port")
    ap.add_argument("--wait", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=("saved", map_const.MODE_FIT, map_const.MODE_CROP),
                    default="saved",
                    help="saved（默认）= 用你在工作台「路线识别 → 小地图定位」里"
                         "选的那个；fit / crop = 本次临时按这个跑（方式由你定，"
                         "程序不按匹配分自动挑）")
    ap.add_argument("--no-save", dest="save", action="store_false",
                    help="不写标定文件（只想看看时用）")
    ap.add_argument("--watch", type=float, default=0.0, metavar="秒",
                    help="连续确认模式：跑这么多秒（0 = 一直跑到按 q / Ctrl+C）。"
                         "crop（局部小地图）用它边走边看 view 对不对")
    ap.add_argument("--fps", type=float, default=5.0, help="--watch 的刷新率")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="--watch 时不开预览窗口（只写对照图）")
    a = ap.parse_args()
    # 端口和 host 一样，默认从**双机共用**的 link.yaml 读 —— A 机部署台那张
    # 「小地图推流」卡片也是照它对（自检里会比对，不一致当场说）。
    port = a.port or int(get("minimap", "port", 5003))
    if a.watch:
        # watch 自己会去读工作台里选的那个（mode=None → 读标定文件）
        return watch(a.map_id, a.host or get("a_host"), port,
                     None if a.mode == "saved" else a.mode,
                     a.watch, a.fps, a.show, a.out)
    return diagnose(a.map_id, a.host or get("a_host"), port, a.wait, a.out,
                    a.mode, a.save)


if __name__ == "__main__":
    raise SystemExit(main())
