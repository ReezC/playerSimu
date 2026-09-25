"""【B 机运行】手动微调探针采样几何（把「绿框」推到码带上）。

用法：
    python -m tools.probe_tune                    # 收实时流
    python -m tools.probe_tune --file rec.mkv     # 离线回放（推荐：画面稳、能反复调）
    python -m tools.probe_tune --selftest         # 自检（不需要流）

**为什么要有这个工具**：`probe_auto --solve` 是"全自动解"，而实测踩到的坑是
几何**只偏一点点**（cell/gap 差 0.25~1px）时，自动解照样能给出一个"看着合法"
的几何 —— 解码出来的是**错位凑的假时间戳**，延迟读数于是变成一坨乱数（实测过：
p95 4.7s、max 4.9s，紧贴 --max-delay 上限；而真延迟不可能堆在自己过滤器的边上）。
人眼对着画面推框，配合下面那道「单调性」判据，比自动解可信。

按键：
    w / a / s / d    上/左/下/右 各 1px（**Shift + 这些 = 10px**）
    e / r            cell  -/+ 0.25（方块边长；框高就是它）
    , / .            gap   -/+ 0.25（方块间距）
    t                就近自动重解一次（probe_auto.locate 当起点，之后手工微调）
    Enter            保存（写全局标定 + 当前项目的 probe_calib；并打印 link.yaml
                     该抄的两行 —— link.yaml 注释多，不代写，免得冲掉）
    p                把当前几何再打一遍到终端
    q / Esc          退出

**判据（屏幕上实时显示，也是这个工具存在的理由）**：
    ① **单调性** —— 解出的时间戳必须只增不减、步长≈帧周期。错位采样会解出
       "乱跳"的时间戳，这是最容易看出来的特征；
    ② 与本地时刻相符 —— 解出的时刻不能离本机"现在"太远（跨日按环形算）；
       **离线回放（--file）自动关掉这一条**：拿昨天的录像跑，解出的时刻当然
       离现在几小时，留着它会把好几何也否掉（单调性照样有效）；
    ③ **只有 ①② 都过才显示延迟**，而且**判据没过就不给保存**（要硬存加
       `--force`）。报一个假延迟、或者把一份错几何静默存进项目，都比不报糟得多
       —— 今天就是被它带偏了半天。

**为什么只信单调性**："解出一个接近此刻的时间戳"**不能证明几何对**：40 位里
低位是快变化的毫秒，错位采样只会把低位采错，凑出一个仍在 ±10 分钟窗口里的值
完全可能（何况框选那条路一次要试几百个候选）。而单调性是**时间轴上的性质**，
错位必然破坏它。

保存（Enter）：写 `config/probe_calib.json` + 项目的 `decision.probe_calib`
（`--project` 可显式指定写哪个项目，默认"最近打开的那个"），并打印 link.yaml
该抄的两行。**link.yaml 不代写**：那份注释很多，用 yaml 重写会把注释全冲掉。

几何来源与工作台**同一套优先级**（项目标定 → 全局文件 → link.yaml，见
`probe_codec.pick_calib`）。老工具（`probe_recv`/`live_detect`）只读 link.yaml，
于是工具和工作台在两套坐标里比大小 —— 今天就是这么岔开的。
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from link import FileSource, PyAVSource
from tools import probe_codec
from tools.config import get
# 判据与工作台**共用一份实现**（`probe_codec.Verdict`）：两边各写一份必然漂移，
# 而漂移的那一份会让"工具说 OK、工作台说不对"这种最难查的分歧出现。
from tools.probe_codec import VERDICT_HIST as HIST      # noqa: F401  （兼容旧名）
from tools.probe_codec import VERDICT_STEP_MUL as STEP_MAX_MUL  # noqa: F401
from tools.probe_codec import Verdict                   # noqa: F401

#: 时钟偏移的重读间隔（秒）—— 两台机器的 NTP 各漂各的，实测 ~45ms/小时
OFFSET_REFRESH = 5.0

COLORS = {"ok": (80, 220, 80), "bad": (60, 60, 240), "warn": (0, 200, 255)}


# ══════════════════════════════════════════
# 采样与判据（纯函数，自检直接打它们）
# ══════════════════════════════════════════

def sample_cells(gray, x, y, cell, gap, bits):
    """按几何采样 2+bits 个方块 → (bits 列表, 每块灰度均值)。

    **取整规则必须与 `probe_codec.read_bits` 完全一致** —— 差半个像素的取整会
    在 42 个方块上累积，越往后越偏，那正是这个工具要修的毛病。
    """
    n = 2 + bits
    h, w = gray.shape[:2]
    half = max(1, int(float(cell) // 4))
    cy = int(round(float(y) + float(cell) / 2.0))
    out, means = [], []
    for i in range(n):
        cx = int(round(float(x) + i * (float(cell) + float(gap)) + float(cell) / 2.0))
        if cy >= h or cx >= w:
            out.append(0)
            means.append(-1.0)
            continue
        y0, y1 = max(0, cy - half), min(h, cy + half)
        xa, xb = max(0, cx - half), min(w, cx + half)
        patch = gray[y0:y1, xa:xb]
        if patch.size == 0:
            out.append(0)
            means.append(-1.0)
            continue
        m = float(patch.mean())
        means.append(m)
        out.append(1 if m > 128.0 else 0)
    return out, means


def decode_with(gray, geo, bits):
    """用给定几何解一次 → (ts 或 None, bits 序列, 每块灰度均值)。"""
    seq, means = sample_cells(gray, geo["x"], geo["y"], geo["cell"], geo["gap"], bits)
    return probe_codec.bits_to_ms(seq, bits), seq, means


def nudge_key(geo, key, shift=False):
    """按键 → 新几何（None = 这个键不改几何）。**纯函数**，方便自检钉住步进。"""
    g = dict(geo)
    step = 10.0 if shift else 1.0
    if key == "a":
        g["x"] -= step
    elif key == "d":
        g["x"] += step
    elif key == "w":
        g["y"] -= step
    elif key == "s":
        g["y"] += step
    elif key == "e":
        g["cell"] = max(3.0, g["cell"] - 0.25)
    elif key == "r":
        g["cell"] += 0.25
    elif key == ",":
        g["gap"] = max(0.0, g["gap"] - 0.25)
    elif key == ".":
        g["gap"] += 0.25
    else:
        return None
    return g


def apply_key(geo, ch):
    """界面按键字符 → 新几何（None = 这个键不改几何）。

    **为什么单独一层**：Shift 走的是**大写字符**（cv2 的 waitKey 给 'D'），
    直接拿 'D' 去比 "wasd" 会全部落到 None —— 10px 步进就成死的了。
    冒烟测试逮到过这个（自检当时只测了 `nudge_key`，绕过了这层分派）。
    """
    if not ch:
        return None
    return nudge_key(geo, ch.lower(), shift=ch.isupper())


def frame_period_ms(src):
    """流的一帧有多少毫秒（判步长用；取不到就给 60fps 的值）。"""
    try:
        fps = float(src.fps or 0)
    except Exception:
        fps = 0.0
    return 1000.0 / fps if fps > 1 else 16.7


# ══════════════════════════════════════════
# 几何的来源与保存
# ══════════════════════════════════════════

def open_project(path=None):
    """→ Project 实例（`--project` 指定，或"最近打开的那个"），拿不到就 None。"""
    try:
        from gui.project import Project, last_opened
        if path:
            return Project.open(path)
        return last_opened()
    except Exception as e:
        print("[probe_tune] 打不开项目（%s: %s）" % (type(e).__name__, e))
        return None


def project_calib(path=None):
    """该项目里存的那份探针标定 → dict 或 None。

    工作台优先用**项目标定**（`decision.probe_calib`）。工具要跟它对齐，就得自己把
    这份捞出来 —— `pick_calib()` 不带参数时只看全局文件，会**和工作台取到不同的
    一份**（今天岔开的根源）。`path` 为空时用"最近打开的那个项目"。
    """
    p = open_project(path)
    if p is None:
        return None
    try:
        d = (p.get("decision") or {}).get("probe_calib")
        if isinstance(d, dict) and d:
            return dict(d)
    except Exception:
        pass
    return None


def should_save(verdict, force=False):
    """判据没过时**不许保存**（要硬存得显式 --force）→ (能不能存, 原因)。

    这是今天事故的直接教训：错的几何被存进项目后，工作台和工具**都**按它采样，
    延迟读数从此全是假的，而界面上一切正常。所以存之前必须过判据。
    """
    ok, why = verdict.summary()
    if ok:
        return True, why
    if force:
        return True, "判据没过但 --force：照存（几何可能不对）"
    return False, why


def initial_geo(frame_shape, bits=40, project_path=None):
    """起始几何 —— 与工作台**同一套优先级**（项目标定 → 全局文件 → link.yaml）。"""
    cal, src = probe_codec.pick_calib(project_calib(project_path))
    if cal:
        x, y, cell, gap = probe_codec.calib_to_px(cal, frame_shape)
        return {"x": x, "y": y, "cell": cell, "gap": gap}, src
    return ({"x": float(get("probe", "x", 100)), "y": float(get("probe", "y", 8)),
             "cell": float(get("probe", "cell", 16)),
             "gap": float(get("probe", "gap", 2))}, "配置值")


def save_geo(geo, frame_shape, bits=40, project=True, project_path=None):
    """存下来 → (写成功的去处列表, link.yaml 该抄的行)。

    两处都写，**不让人猜哪份生效**：
      · `config/probe_calib.json` —— 全局标定（`pick_calib` 的兜底，工具读它）；
      · 当前项目的 `decision.probe_calib` —— 工作台优先用这份。
    link.yaml 不代写：那份注释很多，用 yaml 重写会全冲掉（见 probe_codec 的说明），
    所以打印出来让人自己抄。
    """
    done = []
    cal = probe_codec.calib_from_geo(geo, frame_shape, bits)
    try:
        probe_codec.save_calib(geo, frame_shape, bits)
        done.append("config/probe_calib.json")
    except Exception as e:
        print("[probe_tune] 写全局标定失败：%s: %s" % (type(e).__name__, e))

    if project:
        if project_path:
            # 显式指定了项目就写它 —— **不猜**。写错项目的后果是"工作台按一份错的
            # 几何采样"，而界面上看不出来（今天就是这样）。
            try:
                from gui.project import Project
                p = Project.open(project_path)
                if p is None:
                    raise RuntimeError("打不开 %s" % project_path)
                d = dict(p.get("decision") or {})
                d["probe_calib"] = cal
                p.set("decision", d)
                p.save()
                done.append(str(p.root))
            except Exception as e:
                print("[probe_tune] 写指定项目失败（全局那份已写）：%s: %s"
                      % (type(e).__name__, e))
        else:
            p = open_project()
            if p is not None:
                try:
                    d = dict(p.get("decision") or {})
                    d["probe_calib"] = cal
                    p.set("decision", d)
                    p.save()
                    done.append(str(p.root))
                except Exception as e:
                    print("[probe_tune] 写「最近打开的项目」失败（全局那份已写）：%s: %s"
                          % (type(e).__name__, e))
            else:
                print("[probe_tune] 没有「最近打开的项目」，跳过项目那份（"
                      "要指定就加 --project 项目目录）")
    return done, cal


# ══════════════════════════════════════════
# 画面
# ══════════════════════════════════════════

def draw(vis, geo, bits, seq, vd, lat_ms, note):
    """把几何、每块的采样点、判据画上去（这就是"绿框"）。"""
    n = 2 + bits
    cell, gap = float(geo["cell"]), float(geo["gap"])
    x2 = int(round(geo["x"] + n * (cell + gap)))
    y2 = int(round(geo["y"] + cell))
    good = vd.mono and (vd.plausible or not vd.need_now)
    color = COLORS["ok"] if good else COLORS["bad"]
    cv2.rectangle(vis, (int(round(geo["x"])) - 2, int(round(geo["y"])) - 2),
                  (x2 + 2, y2 + 2), color, 1)

    # 每块采样点的位置 + 判成 0/1：**错位时一眼能看出来**（白块的采样点落到黑块上，
    # 颜色就反了）。这是"手动推框"能成立的反馈。
    for i in range(n):
        cx = int(round(geo["x"] + i * (cell + gap) + cell / 2.0))
        cy = int(round(geo["y"] + cell / 2.0))
        cv2.drawMarker(vis, (cx, cy), (0, 255, 255) if seq[i] else (200, 90, 90),
                       cv2.MARKER_CROSS, 5, 1)
        if i < 2:                       # 头两个是固定标记（白、黑）
            cv2.circle(vis, (cx, cy), int(cell / 2) + 1, (255, 0, 255), 1)

    ok, why = vd.summary()
    cv2.putText(vis, "%s | x=%.1f y=%.1f cell=%.2f gap=%.2f"
                % ("几何 OK" if ok else "几何不对", geo["x"], geo["y"], cell, gap),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, why[:80], (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if ok and lat_ms is not None:
        cv2.putText(vis, "端到端延迟 %.1f ms" % lat_ms, (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["ok"], 2)
    elif not ok:
        cv2.putText(vis, "延迟暂不可报（几何不可信）", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["bad"], 2)
    cv2.putText(vis, "wasd 1px(Shift 10) e/r cell ,/. gap t 自动 Enter 保存 q 退出",
                (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COLORS["warn"], 1)
    if note:
        cv2.putText(vis, note, (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLORS["warn"], 1)
    return vis


# ══════════════════════════════════════════
# 运行
# ══════════════════════════════════════════

def selftest():
    """转给 tools/selftest_probe_tune（统一一份断言，别写两套）。"""
    from tools.selftest_probe_tune import main as st
    return st()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=None)
    ap.add_argument("--file", default=None, help="离线回放（推荐：画面稳，能反复调）")
    ap.add_argument("--format", default=None, help="强制输入格式（裸流必须给）")
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="跑满这么多秒就自动退出（0=手动 q 退出）")
    ap.add_argument("--no-save", action="store_true", help="只调、只打印，不写盘")
    ap.add_argument("--project", default=None,
                    help="指定项目目录（保存时写它；默认用「最近打开的项目」）")
    ap.add_argument("--force", action="store_true",
                    help="判据没过也保存（不建议：错几何会让工作台的延迟读数全变假）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    bits = args.bits or int(get("probe", "bits", 40))
    src = (FileSource(args.file, realtime=False) if args.file
           else PyAVSource(args.url or get("stream", "url"),
                           container_format=args.format))
    src.open()
    w, h = src.size or (0, 0)
    frame = src.read()
    if frame is None:
        print("[probe_tune] 没读到帧：流没起？还是裸流没给 --format？")
        return 1
    shape = np.asarray(frame.image).shape
    geo, src_name = initial_geo(shape, bits, args.project)
    period = frame_period_ms(src)

    # 时钟偏移：启动时自动对一次（失败就用旧文件），之后每 5 秒重读一次文件 ——
    # 两台机器的 NTP 各漂各的（实测 ~45ms/小时），看完数不重读会把漂移当成延迟变化。
    try:
        from tools.probe_recv import load_offset_ms
        offset_s = float(load_offset_ms(None)) / 1000.0
        print("[probe_tune] 时钟偏移 %.1f ms（本工具启动时会自动对时）"
              % (offset_s * 1000.0))
    except Exception as e:
        offset_s = 0.0
        print("[probe_tune] 没拿到时钟偏移（%s）：延迟数仅供参考，先跑 tools.clock_sync"
              % type(e).__name__)
    _off_t = [time.perf_counter()]

    def offset():
        now = time.perf_counter()
        if now - _off_t[0] >= OFFSET_REFRESH:
            _off_t[0] = now
            try:
                from tools.config import ROOT
                p = ROOT / "config" / "clock_offset.txt"
                _off_v = float(p.read_text(encoding="utf-8").strip()) / 1000.0
                return _off_v
            except Exception:
                pass
        return offset_s

    print("[probe_tune] %sx%s fps=%s  几何来源=%s -> x=%.1f y=%.1f cell=%.2f gap=%.2f"
          % (w, h, src.fps, src_name, geo["x"], geo["y"], geo["cell"], geo["gap"]))
    print("[probe_tune] 按键：wasd 挪 1px（Shift=10px）  e/r 改 cell  ,/. 改 gap  "
          "t 自动重解  Enter 保存  q 退出")

    # 离线回放：判据只留单调性（回放的画面里"时刻"当然不等于现在），延迟也不适用。
    offline = bool(args.file)
    t_start = time.perf_counter()
    vd = Verdict(need_now=not offline)
    lat = None
    note = ("离线回放：只看单调性，延迟不适用" if offline
            else "几何来源：%s" % src_name)
    if offline:
        print("[probe_tune] 离线回放模式：判据只看**单调性**（时间戳是否乱跳）")

    while True:
        gray = cv2.cvtColor(np.asarray(frame.image), cv2.COLOR_RGB2GRAY)
        ts, seq, _means = decode_with(gray, geo, bits)
        if ts is None:
            vd = Verdict(need_now=not offline)
            lat = None
            note = "标记块没解出（头两位必须是 白,黑）—— 先挪 x/y 把起点对准"
        else:
            vd.add(ts, period)
            ok, _why = vd.summary()
            if ok and not offline:
                d = probe_codec.resolve_delay_ms(
                    (frame.t_recv_wall + offset()) * 1000.0, ts)
                lat = d if (d is not None and d < 5000) else None
            else:
                lat = None

        vis = cv2.cvtColor(np.asarray(frame.image), cv2.COLOR_RGB2BGR)
        draw(vis, geo, bits, seq, vd, lat, note)
        try:
            cv2.imshow("probe_tune", vis)
        except cv2.error as e:
            print("[probe_tune] 打不开显示窗口（%s）—— 这个工具要在有桌面的会话里跑"
                  % str(e).splitlines()[0])
            src.close()
            return 1
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        ch = chr(key) if 0 < key < 128 else ""
        ng = apply_key(geo, ch)
        if ng is not None:
            geo = ng
            note = ""
        elif key == ord("p"):
            print("[probe_tune] x=%.2f y=%.2f cell=%.2f gap=%.2f"
                  % (geo["x"], geo["y"], geo["cell"], geo["gap"]))
        elif key == ord("t"):
            try:
                from tools.probe_auto import locate
                r = locate(gray, int(round(geo["cell"])), float(geo["gap"]), bits)
                if r:
                    _c, ax, ay = r
                    geo["x"], geo["y"] = float(ax), float(ay)
                    vd = Verdict(need_now=not offline)
                    note = "自动重解：x=%.0f y=%.0f（接着手工微调）" % (ax, ay)
                else:
                    note = "自动重解失败：这一帧没搜到探针"
            except Exception as e:
                note = "自动重解不可用：%s" % type(e).__name__
        elif key in (10, 13):                       # Enter
            ok_save, why_save = should_save(vd, args.force)
            if not ok_save:
                # **判据没过就不存**：错的几何一旦存进项目，工作台从此按它采样，
                # 延迟读数全是假的而界面上一切正常（今天正是这么被坑的）。
                note = "判据没过，没保存：%s" % why_save[:30]
                print("\n[probe_tune] **不保存**：%s" % why_save)
                print("[probe_tune] 继续调（wasd/e/r/,/.），确认要硬存加 --force")
                nxt = src.read()
                if nxt is not None:
                    frame = nxt
                continue
            if args.no_save:
                print("\n[probe_tune] --no-save：不写盘，只给值")
                done, cal = [], probe_codec.calib_from_geo(geo, shape, bits)
            else:
                done, cal = save_geo(geo, shape, bits, project_path=args.project)
            note = "已保存：%s" % ("、".join(done) or "（没写）")
            print("\n[probe_tune] 已保存 -> %s" % ("、".join(done) or "（没写）"))
            print("[probe_tune] 标定比例（工作台那份）："
                  "x_ratio=%.6f y_ratio=%.6f cell_ratio=%.6f gap_ratio=%.6f"
                  % (cal["x_ratio"], cal["y_ratio"], cal["cell_ratio"], cal["gap_ratio"]))
            print("[probe_tune] link.yaml 的 probe 段抄这几行（cell/gap 允许小数）：")
            print("  x: %g\n  y: %g\n  cell: %g\n  gap: %g"
                  % (geo["x"], geo["y"], geo["cell"], geo["gap"]))

        nxt = src.read()
        if nxt is not None:
            frame = nxt
            if offline:
                time.sleep(1.0 / 30.0)      # 回放放慢到 ~30fps：看得清、来得及按键
        elif offline:
            time.sleep(0.05)        # 离线到头了：停住最后一帧，别空转烧 CPU

        if args.seconds and (time.perf_counter() - t_start) >= args.seconds:
            print("[probe_tune] 到了 --seconds %.1f，退出" % args.seconds)
            break

    src.close()
    cv2.destroyAllWindows()
    print("[probe_tune] 退出。最后几何：x=%.2f y=%.2f cell=%.2f gap=%.2f"
          % (geo["x"], geo["y"], geo["cell"], geo["gap"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
