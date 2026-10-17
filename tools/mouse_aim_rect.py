# -*- coding: utf-8 -*-
"""量「**抓取区域**」（帧 ↔ 屏幕的对应 ✓）⇒ 写进 `config/mouse_gain.json` 的 `frame_rect` ✓。

**为什么非要它**（2026-10-17 ✓ 用户现场两句话：「**选频道界面鼠标操作完全歪了**」✗ ＋
  「**刚跑了标定 还是没用**」✗）：
  · `decision/mouse_aim.to_screen()` 老版本**只加原点、不折算比例** ✗ —— 而帧是**推流**来的、
    被缩过 ✗（本机实测：抓取区域 **1920×1080** ✓ 而帧 **1366×768** ✓ ⇒ 比例 **k = 1.4054** ✗）
    ⇒ 目标点整体偏 k 倍 ✓；再叠上桌面 **5120×1440** / 虚拟原点 **(−2560, 0)** 的原点折返 ✓
    ⇒ 现场算出 **4317** 单位（正确约 **700** ✓）⇒ 差 6 倍 = "完全歪了" ✓。
  · ⚠ `tools/mouse_aim_calib.py` 的自动路**拒绝猜**这个折算比**是对的** ✓（它当时报的就是
    "这一帧跟那台机器的屏幕不是 1:1（5120x1440）✗、relay 也没报抓取区域 ✗ ⇒ 算不出来 ✓
    （**不猜** ✓）"✓）⇒ 那一步只能**实测** ✓ —— 就是本工具 ✓。

**怎么量（全是实测对应点 ✓ 一处都不猜 ✓）**：
  · **屏幕**坐标 = 本机 `GetCursorPos` ✓（绝对 ✓ —— ⚠ 只有"鼠标打本机"时才有意义 ✓）；
  · **帧**坐标 = 帧差里那个「旧位」块 ✓（`mouse_gain_calib._cursor_delta` ✓ **同一份实现** ✓）；
  · ⚠⚠ **必须一次走位链**（撞角 → 小挪 NUDGE → 逐点走 → 每点再小挪 ✓）：
    `_cursor_delta` 的先验是"**旧位在左边缘**"✗（只有撞角后第一下成立 ✓）
    ⇒ 之后每个点都要把**上一点的位置**当 `anchor` 传进去 ✓（它靠这条把光标那对从噪声里挑出来 ✓）；
    ⚠ **纵轴也用小挪（横向）去取帧坐标** ✗ —— 不能用纵向位移 ✗（那条先验写着"横向指令 ⇒
      两端必在同一行"✓，纵向位移直接不满足 ✓）。
  · 两点联立 ⇒ `k = Δ屏幕 / Δ帧`（**比例** ✓）、`原点 = 屏幕 − 帧 × k` ✓（横纵各一组 ✓）；
  · ⭐ **第三点复核**：用算出来的映射**预测**它的帧位置 ⇒ 与实测比，差超过 `VERIFY_TOL_PX`
    ⇒ **一个字都不写** ✗（宁可退老行为，也不按一条没验过的记录去点 ✓）。

**用法**（在**跑工作台那台机器**上跑 ✓ 与标定工具同一台 ✓）：
```
python -X utf8 -m tools.mouse_aim_rect                # 量 + 写
python -X utf8 -m tools.mouse_aim_rect --dry-run      # 只打印，不写文件
```

⚠ 前提（**不满足就明确报错、绝不瞎写** ✗）：
  ① 鼠标必须是**本机硬件**（ProMicro 插在本机 ⇒ 读得到系统光标 ✓）——
     若是"指令经 relay 打到另一台"✗，那台的屏幕坐标这边问不到 ⇒ 本工具**拒绝** ✓；
  ② **画面里看得见光标** ✓（帧差要靠它 ✓ —— 本地窗口那种取帧路**不含**系统光标 ✗
     见 `mouse_aim_calib._cursor_pos` 的说明 ✓ ⇒ 那种情况会直说量不到 ✓）；
  ③ 游戏停在**一屏不动的画面** ✓（画面自己在动 ⇒ 帧差里不止光标 ✓）；
  ④ 先跑过一次标定 ✓（`frame_rect` 是**补在那一帧尺寸的记录上** ✓ 见 `save_gain` ✓）。
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision import input as dinput                        # noqa: E402
from tools.mouse_aim_calib import _cursor_pos, _grab, _open_stream  # noqa: E402
from tools.mouse_gain_calib import _cursor_delta            # noqa: E402

#: 取点用的**指令数**（≈ 屏幕像素 ✓ 实测 1 单位 = 1 屏幕像素 ✓）。
#: ⚠ 取这么大是故意的：① 要**离开左边缘**不少 ✓（两点才拉得开、比例才算得准 ✓）；
#: ② 几百像素的基线能把几十像素的取点误差摊薄 ✓。
PTS = (600, 1000, 1400)

#: 每个点取帧坐标那一下**往右小挪**多少 ✓（凑出"旧位/新位"两块 ✓；太小两块会糊 ✓）。
NUDGE = 120

#: 每步之后等多久（撞角是大位移 ✓ 慢 relay / 慢流下要留够 ✓）。
SETTLE = 0.35

#: 复核容忍（帧像素 ✓）：预测与实测差超过它 ⇒ **不写** ✗。
VERIFY_TOL_PX = 12.0


def _local_mouse():
    """鼠标是不是**打本机**（只有这样才读得到"光标在屏幕哪" ✓）⇒ `(ok, why)` ✓。"""
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    name = type(getattr(dinput, "_remote", None)).__name__
    if name != "SerialKbd":
        return False, ("鼠标后端是 **%s** ⇒ 指令打到**另一台**机器上 ✗，那台的屏幕坐标这边"
                       "问不到 ⇒ 量不了（**不猜** ✓）。要么让鼠标走本机 ProMicro，"
                       "要么手工填那台机器的 `frame_rect`（见本文件头 ✓）" % (name or "未知"))
    return True, ""


def _walk(kbd, src, axis):
    """撞角 → 沿一个轴走 `PTS` 那几点 ⇒ `[(屏幕坐标, 帧坐标), …]`（量不到 ⇒ `None`）✓。

    ⚠ `anchor` 链（见文件头 ✓）：每点都把"上一点"当锚 ⇒ `_cursor_delta` 才挑得对 ✓。
    """
    def move(units):
        kbd.send("MOVE %d 0" % int(units) if axis == "x" else "MOVE 0 %d" % int(units))

    def probe():
        """小挪一下 ⇒ 拿"挪之前那个位置"的帧坐标 ✓（+ 挪之后的位置，留给下一轮当锚 ✓）。

        ⚠ **一律横向挪** ✓（纵向轴也一样 ✓）—— `_cursor_delta` 的方向先验写着
          "横向指令 ⇒ 两端必在同一行" ✓，纵向位移不满足它 ✗（见文件头 ✓）。
        """
        before = _grab(src)
        if before is None:
            return None
        kbd.send("MOVE %d 0" % int(NUDGE))         # 横向小挪 ✓
        time.sleep(SETTLE)
        after = _grab(src)
        if after is None:
            return None
        d = _cursor_delta(before, after, direction=1, anchor=_walk.anchor)
        if not d or d[2] is None:
            return None
        _walk.anchor = (d[2][0] + d[0], d[2][1] + d[1])     # 挪完之后在哪 ✓（下一轮的锚 ✓）
        return (float(d[2][0]), float(d[2][1]))

    _walk.anchor = None                          # 撞角后第一下 ⇒ 用"左边缘"那条先验 ✓
    kbd.send("MOVE -4000 -4000")
    time.sleep(SETTLE)
    move(NUDGE)                                  # 先离开左边缘一点 ⇒ 有"旧位"可认 ✓
    time.sleep(SETTLE)
    if probe() is None:
        return None
    out = []
    cur = NUDGE
    for u in PTS:
        move(u - cur)                            # 走到第 u 个点 ✓
        time.sleep(SETTLE)
        s = _cursor_pos()                        # **绝对屏幕坐标** ✓
        f = probe()                              # 该点的**帧坐标** ✓
        if s is None or f is None:
            return None
        out.append((float(s[0]), float(s[1]), f[0], f[1]))
        cur = u + NUDGE                          # 小挪之后光标停在 u+NUDGE ✓
    return out


def _solve(pts):
    """`[(屏幕, 帧), …]` ⇒ `(k, 原点, 人话)`；算不出 / 复核不过 ⇒ `(None, None, 为什么)` ✓。

    ⚠ 两点联立（取前两点 ✓），**第三点只用来复核** ✓ —— 这才是"量完敢写"的依据 ✓
      （只联立不复核 ⇒ 有可能把一次噪声当成标定 ✓ 那就会一直点飞 ✓）。
    """
    if not pts or len(pts) < 3:
        return None, None, "取到的点不够（要 3 个 ✓）"
    s1, t1 = pts[0]
    s2, t2 = pts[1]
    s3, t3 = pts[2]
    ds = float(s2) - float(s1)
    dt = float(t2) - float(t1)
    if abs(ds) < 20.0 or abs(dt) < 20.0:
        return None, None, "两点没拉开（屏幕差 %.1f / 帧差 %.1f ✗）" % (ds, dt)
    k = ds / dt
    if not math.isfinite(k) or k <= 0:
        return None, None, "比例不是正数（%r ✗）" % (k,)
    org = float(s2) - float(t2) * k
    pred = (float(s3) - org) / k
    err = abs(pred - float(t3))
    if err > VERIFY_TOL_PX:
        return None, None, ("复核不过（第三点：预测帧 %.1f / 实测 %.1f ⇒ 差 %.1f px > %.0f ✗）"
                            "⇒ **不写** ✓" % (pred, t3, err, VERIFY_TOL_PX))
    return k, org, ("k=%.4f 原点=%.1f；第三点复核：预测 %.1f / 实测 %.1f（差 %.1f px ✓）"
                    % (k, org, pred, t3, err))


def main(argv=None):
    ap = argparse.ArgumentParser(description="量抓取区域（帧↔屏幕）⇒ 写 frame_rect ✓")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件 ✓")
    args = ap.parse_args(argv)

    from core.config import get
    from remote_kbd.serial_kbd import find_pro_micro_port
    port = str(get("kbd", "serial_local", "") or "").strip() or (find_pro_micro_port() or "")
    if not port:
        print("[X] 找不到本机 ProMicro 串口（`config/link.yaml` 的 `kbd.serial_local` ✓）")
        return 2

    # ⚠⚠ **必须走本地直连**（`input.use_serial` ✓）—— 只有这条路 `GetCursorPos` 才是真值 ✓：
    #   网络 relay 那条（`mouse_aim_calib._connect_input()` ✓）会把指令打到**另一台** ✗，
    #   那台的屏幕坐标这边**问不到** ⇒ 量不了（**不猜** ✓ 见 `_local_mouse` ✓）。
    #   ⚠ 所以**别**用 `_connect_input()` ✗（它会优先连 relay ✓ = 量了个寂寞 ✓）。
    # ⚠⚠ **输入闸**（`input.net_allowed` ✓）：带 `QT_QPA_PLATFORM=offscreen` 又没设
    #   `PSIMU_ALLOW_NET=1` 时 ⇒ `use_serial` **一个串口都不开** ✗（只转"什么都不发"✓）
    #   ⇒ 那样跑会报成"鼠标通道不可用"✗ = **假失败** ✓ ⇒ 先把它说清 ✓（别让人以为串口坏了 ✓）。
    if not dinput.net_allowed():
        print("[X] 输入闸关着（`QT_QPA_PLATFORM=offscreen` 且没设 `PSIMU_ALLOW_NET=1` ✗）"
              "⇒ 本工具**不会**去开串口 ✓")
        print("    ⇒ 在**正常桌面**上跑（别带 offscreen ✓），或先设 `PSIMU_ALLOW_NET=1` ✓")
        return 2
    try:
        dinput.use_serial(port)
    except Exception as e:                      # noqa: BLE001 —— 那句报错本身就是给人看的 ✓
        print("[X] %s" % e)
        print("    ⚠ 若是\"被占用\"⇒ 先**停掉实时预览 / 关掉工作台**再跑 ✓（它握着同一路流 ✓）")
        return 2
    ok, why = _local_mouse()
    if not ok:
        print("[X] %s" % why)
        return 2
    print("[1/3] 输入通道：本地直连 %s（SerialKbd ✓）" % port)

    kbd, src = dinput._remote, None             # ⚠ 复用 `use_serial` 开出来的那一个 ✓
    try:
        try:
            src, url = _open_stream()
        except Exception as e:                  # noqa: BLE001 —— 它是给人看的（别抛栈 ✗）
            print("[X] 开不了流：%s" % e)
            print("    ⚠ 若是「**地址已被占用**」（`Errno 10048` ✓）⇒ **有别人正听着同一个端口**")
            print("      （多半就是**工作台** ✓ 它握着同一路流 ✓）")
            print("    ⇒ **先停掉实时预览 / 关掉工作台**再跑 ✓（约 30 秒就够 ✓）；")
            print("      ⚠ 量位移**非要这路流不可** ✗ —— 本地窗口那种取帧路**不含系统光标** ✓")
            return 2
        print("[2/3] 流：%s" % url)
        time.sleep(0.6)                          # 让流稳一下 ✓
        f0 = _grab(src)
        if f0 is None:
            print("[X] 抓不到帧 ✗（流没通 / 没在推 ✓）")
            return 2
        fh, fw = int(f0.shape[0]), int(f0.shape[1])
        print("      帧 %dx%d" % (fw, fh))
        print("[3/3] 撞角归零 → 取点（⚠ 游戏请停在**一屏不动**的画面 ✓；"
              "画面里要看得见光标 ✓）")
        pts_x = _walk(kbd, src, "x")
        if not pts_x:
            print("[X] 横轴取不到点（帧差里找不到光标 ✗ —— 画面看不见光标？画面在动？"
                  "本地窗口那种取帧路**不含**系统光标 ✓）")
            return 2
        pts_y = _walk(kbd, src, "y")
        if not pts_y:
            print("[X] 纵轴取不到点 ✗")
            return 2
    finally:
        try:
            dinput.use_local()                   # 关掉串口 / socket（不泄漏 ✓ 见它说明 ✓）
        except Exception:                        # noqa: BLE001
            pass
        if src is not None:
            try:
                src.close()
            except Exception:                    # noqa: BLE001
                pass

    # 横：屏幕 x ↔ 帧 x ✓；纵：屏幕 y ↔ 帧 y ✓
    kx, ox, wx = _solve([(p[0], p[2]) for p in pts_x])
    if kx is None:
        print("[X] 横轴算不出：%s" % wx)
        return 2
    print("      横轴 %s ✓" % wx)
    ky, oy, wy = _solve([(p[1], p[3]) for p in pts_y])
    if ky is None:
        print("[X] 纵轴算不出：%s" % wy)
        return 2
    print("      纵轴 %s ✓" % wy)

    if not (0 < kx <= 4.0 and 0 < ky <= 4.0):
        print("[X] 比例离谱（%.4f / %.4f ✗）⇒ **不写** ✓" % (kx, ky))
        return 2
    cw, ch = int(round(fw * kx)), int(round(fh * ky))
    rect = [int(round(ox)), int(round(oy)), cw, ch]
    print("")
    print("抓取区域（屏幕像素）= %r ⇒ 帧 %dx%d 的比例 (%.4f, %.4f) ✓"
          % (rect, fw, fh, kx, ky))
    if args.dry_run:
        print("（--dry-run ⇒ 不写 config/mouse_gain.json ✓）")
        return 0

    import decision.mouse_aim as _ma
    g = _ma.load_gain((fh, fw))
    if not g:
        print("[X] 这个帧尺寸（%dx%d）**还没标定过** ⇒ 先跑 `tools/mouse_aim_calib.py` ✓"
              "（`frame_rect` 是补在那份记录上的 ✓）" % (fw, fh))
        return 2
    p = _ma.save_gain(g[0], g[1], frame_shape=(fh, fw),
                      extra={"frame_rect": rect,
                             "frame_rect_probe": {
                                 "pts": list(PTS), "nudge": NUDGE,
                                 "k": [round(float(kx), 6), round(float(ky), 6)],
                                 "origin": [int(round(ox)), int(round(oy))],
                                 "ver": 1}})
    print("已写 %s（`frame_rect` 补在 %dx%d 那份上 ✓；别的尺寸那几份原样保留 ✓）"
          % (p, fw, fh))
    print("⚠ **重启工作台**才生效 ✓；之后 `to_screen()` 会按它把帧像素折成屏幕像素 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
