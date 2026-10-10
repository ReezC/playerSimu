"""`perception/vt_sdk.py` 的自检（自动测谎接入用的 SDK 薄封装）。

钉九件事（每条都断言"可测量的结果"，不写"我觉得" ✗）：
  ① 模板几何：`qr.png`（确认按钮）在 `cg.png`（成功面板）里 score≥0.99 ⇒ 比例**量得出来**
     且落在右下（实测 (0.9201, 0.9232) ✓）；
  ② 加载：`load()` 后 `ready=True`、`load_ms>0`、`error` 空；模型真的在跑（GPU ✓）；
  ③ 真实素材：喂原始分辨率帧 ⇒ 会 **LOCKED**，ROI 在帧内、瞄点在 ROI 内，帧时中位 <150ms；
  ④ 成功判定：合成"成功面板帧"⇒ `SUCCESS_PENDING` →（下一拍）**SUCCESS** ✓
     （与 SDK 自己那份实现同一阈值 `.86` ✓）；
  ⑤ 确认按钮定位：合成帧上量出的点击点与真值误差 **≤8px**（实测 0.6px ✓）；
  ⑥ **不许误报**：没有面板的真实帧上 ⇒ 返回 None（阈值挡住 ✓ 2026-10-09 踩过 0.804 的假点 ✗）；
  ⑦ 截图尺寸一变 ⇒ 自动重开一轮，不抛异常、还能接着喂（SDK 自己会 `capture_geometry_changed` ✓）；
  ⑧ SDK 目录不对 ⇒ 人话错误、`ready=False`、**不崩**（调用方据此停用 ✓）；
  ⑨ 时间戳重复 ⇒ 内部顶一下，**不抛**（SDK 原生会 `ValueError` ✗）。

跑法：`.venv\\Scripts\\python.exe -X utf8 tools/selftest_vt_sdk.py`
（会真加载 YOLO 权重：实测 ~11s；整条跑完约 30s ✓）
"""
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
# ⚠ SDK 的包（`liescc`/`vision_sdk`）要先能 import 才能引用它的常量（`locate.HOLE_*` ✓）
# —— 生产代码里这步在 `VtLieSdk._import_sdk()` 里做（延迟 ✓），自检要提前用同一份路径 ✓
sys.path.insert(0, str(_ROOT / "visual_tracking_sdk_20260920"))

import cv2                                                            # noqa: E402
import numpy as np                                                    # noqa: E402

from liescc import locate                                             # noqa: E402

from perception.vt_sdk import DEFAULT_SDK_DIR, VtLieSdk               # noqa: E402

_VIDEO = r"D:\Media\record\mxd\10月7日.mp4"
#: 真实素材里"弹窗已经出现"的帧（探针 1 在原始分辨率上第 138 帧才 LOCKED ⇒ 取 300 稳 ✓）
_FRAME_IDX = 300


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _gray(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _match(gray, templ_gray, scales):
    """多尺度匹配 ⇒ (score, x, y, w, h)。"""
    best = (-1.0, 0, 0, 0, 0)
    for sc in scales:
        tw, th = int(templ_gray.shape[1] * sc), int(templ_gray.shape[0] * sc)
        if tw < 16 or th < 12 or tw > gray.shape[1] or th > gray.shape[0]:
            continue
        templ = cv2.resize(templ_gray, (tw, th), interpolation=cv2.INTER_AREA)
        _mn, mx, _ml, loc = cv2.minMaxLoc(
            cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED))
        if mx > best[0]:
            best = (mx, loc[0], loc[1], tw, th)
    return best


def _video_frames(n, start=0):
    """读原始分辨率的前 n 帧（从 start 起）。"""
    cap = cv2.VideoCapture(_VIDEO)
    out = []
    for _ in range(start):
        if not cap.read()[0]:
            break
    for _ in range(n):
        ok, frm = cap.read()
        if not ok:
            break
        out.append(frm)
    cap.release()
    return out


def _synthetic_success(frame, roi, scale=1.2):
    """把 `cg.png`（成功面板）按 `scale` 贴到 `roi` 处 ⇒ (合成帧, 真实按钮中心)。

    ⚠ `scale` 必须落在 SDK 的 `SCALES`（0.40~1.30）里 —— 超了就是"贴上去也匹配不到" ✗
      （2026-10-09 实测踩过 ✓）。
    """
    from vision_sdk.success import SCALES
    check(min(SCALES) <= scale <= max(SCALES),
          "用例自己的贴图尺度 %.2f 不在 SDK 尺度表里 ⇒ 这条断言本身没意义 ✗" % scale)
    cg = _gray(DEFAULT_SDK_DIR / "liescc" / "cg.png")
    qr = _gray(DEFAULT_SDK_DIR / "liescc" / "qr.png")
    _mn, _mx, _ml, loc = cv2.minMaxLoc(cv2.matchTemplate(cg, qr, cv2.TM_CCOEFF_NORMED))
    btn = ((loc[0] + qr.shape[1] / 2.0) / cg.shape[1],
           (loc[1] + qr.shape[0] / 2.0) / cg.shape[0])
    x, y, w, h = roi
    ow, oh = w / locate.HOLE_RW, h / locate.HOLE_RH
    ox, oy = x - ow * locate.HOLE_RX, y - oh * locate.HOLE_RY
    cg_r = cv2.resize(cg, (int(cg.shape[1] * scale), int(cg.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)
    px, py = int(ox + 40), int(oy + 60)
    out = frame.copy()
    out[py:py + cg_r.shape[0], px:px + cg_r.shape[1]] = cv2.cvtColor(cg_r, cv2.COLOR_GRAY2BGR)
    # ⚠ 真值要按**贴图后的像素尺寸**推（`btn` 是**比例** ✓）—— 拿比例去乘 `scale` 是错的 ✗
    #   （2026-10-09 自检第一次跑就报 599px，根因是**用例自己算错真值**；模块给的点只差
    #     0.6px ✓ —— 量一遍就分得清"产品错"还是"用例错" ✓）。
    return out, (px + btn[0] * cg_r.shape[1], py + btn[1] * cg_r.shape[0])


def t_templates():
    """① 模板几何：比例量得出来、且在右下。"""
    sdk = VtLieSdk()
    check(sdk._templates(), "模板加载/量比例失败（cg.png / qr.png 缺失或不像？）")
    rx, ry = sdk._btn_ratio
    check(0.80 <= rx <= 0.99 and 0.80 <= ry <= 0.99,
          "确认按钮在面板里的比例不合理（该在右下）：(%r, %r)" % (rx, ry))


def t_bad_dir():
    """⑧ SDK 目录不对 ⇒ 人话错误、不崩。"""
    bad = VtLieSdk("Z:\\definitely\\no\\such\\sdk")
    check(bad.load() is False, "SDK 目录不对却报告加载成功 ✗")
    check("SDK 目录不存在" in bad.error, "错误原因不是人话：%r" % (bad.error,))
    check(bad.feed(np.zeros((64, 64, 3), np.uint8), 1.0) is None,
          "没就绪时 feed 该返回 None（调用方靠它跳过 ✓）")


def t_frame_rect():
    """框坐标换算：**追踪图坐标 → 喂进去那张图的坐标**（与 `aim_frame` 同一口径 ✓）。

    为什么要单钉它：`live_thread._vt_overlay` 靠它画**目标框** ✓ —— 算错就是"框画在别处"，
    而那种错**看着像"跟踪不准"** ✗（2026-10-09 用户报的"看不到框"补的就是这条 ✓）。
    用**手算得出的数**对（不跑模型、不要 GPU ✓）：
      roi=(100,50,640,360)、追踪图 320×180（正好一半）⇒ 追踪图里 (10,20)-(30,40)
      ⇒ 画面里 (100+10*2, 50+20*2)-(100+30*2, 50+40*2) = (120,90)-(160,130) ✓
    """
    import types

    from perception.vt_sdk import frame_rect_of

    _obs = types.SimpleNamespace(
        roi=(100, 50, 640, 360),
        tracking_frame=np.zeros((180, 320, 3), np.uint8),
        result=types.SimpleNamespace(target_bbox=types.SimpleNamespace(
            x0=10.0, y0=20.0, x1=30.0, y1=40.0)))
    r = frame_rect_of(_obs)
    check(r is not None, "有 roi / 有框 / 有追踪图，却算不出来 ✗")
    for got, want, what in zip(r, (120.0, 90.0, 160.0, 130.0),
                               ("x0", "y0", "x1", "y1")):
        check(abs(got - want) < 1e-6, "%s 算错了：%.1f ≠ %.1f" % (what, got, want))
    check(frame_rect_of(types.SimpleNamespace(roi=None, tracking_frame=None,
                                              result=None)) is None,
          "没 roi / 没框的时候该返回 None（不猜 ✓）")


def t_real_video_and_confirm():
    """③④⑤⑥⑦⑨ 真实素材 + 合成成功面板：一条把"喂帧→锁定→成功→点确定"全走完。"""
    if not pathlib.Path(_VIDEO).exists():
        print("      （跳过：素材 %s 不在本机 ✓）" % _VIDEO)
        return
    sdk = VtLieSdk()
    check(sdk.load(blocking=True), "SDK 加载失败：%s" % (sdk.error or "（无原因）"))
    check(sdk.ready and sdk.load_ms > 0, "ready/load_ms 不对：%r / %r" % (sdk.ready, sdk.load_ms))
    print("      （加载耗时 %.1fs ✓）" % (sdk.load_ms / 1000.0))

    frames = _video_frames(200)
    check(len(frames) >= 100, "素材读不到帧：%d" % len(frames))

    # ---- ③ 真实帧：锁定 + 瞄点 + 帧时 ----
    sdk.reset()
    ms, locked, roi, aim, t = [], None, None, None, 0.0
    for i, frm in enumerate(frames):
        t += 1.0 / 30.0
        t0 = time.perf_counter()
        obs = sdk.feed(frm, t, screen_origin=(0.0, 0.0))
        ms.append((time.perf_counter() - t0) * 1000.0)
        if obs.phase == "LOCKED" and locked is None:
            locked, roi, aim = i, obs.roi, obs.aim_frame
    check(locked is not None,
          "真实素材喂了 %d 帧都没 LOCKED（phase=%s）" % (len(frames), sdk.last_phase))
    x, y, w, h = roi
    check(w >= 32 and h >= 32 and x >= 0 and y >= 0
          and x + w <= frames[0].shape[1] and y + h <= frames[0].shape[0],
          "LOCKED 的 ROI 跑出帧外：%r（帧 %s）" % (roi, frames[0].shape))
    check(aim is not None and x <= aim[0] <= x + w and y <= aim[1] <= y + h,
          "首个瞄点不在 ROI 里：aim=%r roi=%r" % (aim, roi))
    ms.sort()
    _med = ms[len(ms) // 2]
    check(_med < 150.0, "帧时中位 %.1fms 太慢（>150ms ⇒ 喂不动实时）" % _med)
    print("      （真实帧：第 %d 帧 LOCKED，ROI=%r，瞄点=(%.1f,%.1f)，中位 %.1fms ✓）"
          % (locked, roi, aim[0], aim[1], _med))

    # ---- ⑥ 真实帧上不许误报到按钮 ----
    check(sdk.find_confirm_point(frames[-1], roi) is None,
          "成功面板根本不在，却报出了确认按钮（阈值没挡住）✗")

    # ---- ④⑤ 合成成功面板 ⇒ SUCCESS_PENDING → SUCCESS + 点确定 ----
    synth, true_btn = _synthetic_success(frames[-1], roi)
    sdk.reset(roi=roi)
    ph = []
    for tt in (0.0, 16.0, 16.6):          # 跨过 SDK 的"ROI 锁定 15s 后才扫成功"那道闸 ✓
        obs = sdk.feed(synth, tt, screen_origin=(0.0, 0.0))
        ph.append(obs.phase)
    check("SUCCESS_PENDING" in ph, "合成成功面板没触发 SUCCESS_PENDING：%s" % ph)
    check(ph[-1] == "SUCCESS", "两次连续命中该给 SUCCESS，却给了 %s" % ph[-1])

    pt = sdk.find_confirm_point(synth, roi)
    check(pt is not None, "面板在、阈值内，却找不到确认按钮 ✗")
    err = ((pt[0] - true_btn[0]) ** 2 + (pt[1] - true_btn[1]) ** 2) ** 0.5
    check(err <= 8.0, "确认按钮定位误差 %.1fpx（>8px ⇒ 点不准 ✗）" % err)
    print("      （合成帧：%s ⇒ 按钮 (%.1f,%.1f) 误差 %.1fpx ✓）" % (ph, pt[0], pt[1], err))

    # ---- ⑦ 尺寸一变 ⇒ 自动重开，还能接着喂 ----
    sdk.reset()
    a = sdk.feed(frames[0], 1.0, screen_origin=(0.0, 0.0))
    small = cv2.resize(frames[0], (889, 500), interpolation=cv2.INTER_AREA)
    b = sdk.feed(small, 1.1, screen_origin=(0.0, 0.0))       # 尺寸变了 ⇒ 内部 reset ✓
    check(a is not None and b is not None, "尺寸变化后喂不动了：%r / %r" % (a, b))
    check(b.timestamp_s > a.timestamp_s, "重开之后时间戳没往前走：%r" % (b.timestamp_s,))

    # ---- ⑨ 时间戳重复 ⇒ 不抛 ----
    c = sdk.feed(small, 1.1, screen_origin=(0.0, 0.0))       # 同一个 ts ✓
    check(c is not None, "重复时间戳直接抛了（该内部顶一下 ✓）")
    sdk.close()


def t_live_integration_yields_under_lag():
    """⭐⭐ **实时链路里的测谎接线**：三条契约（都是 2026-10-09/10 现场逼出来的 ✓）。

    为什么钉在这儿（而不是只测 SDK 自己那份 ✓）：这三条都在 `gui/live_thread.py` 里，
    **只在真跑起来时才看得出来** ✗ ——
      ① 喂帧走**独立线程** ✓（SDK 的 `process` 探针实测中位 ~40ms、**最慢 2232ms** ✗
         ⇒ 放主回路就是"画面卡两秒"✗）；
      ② 只留**最新一帧**（不排队 ✓ 同 `_LatestSlot` ✓）⇒ 画面不越拖越旧 ✓；
      ③ ⭐⭐ **本机积压时让路**（用户 2026-10-10 ✓ 现场：`src_lag=on` ＋ 预览已被自动压到
         最低档 12fps 仍锁死 3.7~4.7 秒 ✗）：测谎自带**第二套 YOLO** ⇒ 抢 GPU/CPU ⇒
         必须在本机持续跟不上源时**停止喂帧**（省下那份推理 ✓）、恢复后自动接着喂 ✓
         —— **可选功能不许反过来拖垮实时** ✗。
    """
    src = (_ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("threading.Thread(target=self._vt_loop" in src,
          "测谎喂帧没走**独立线程** ✗（单帧最慢 2.2s ⇒ 主回路会卡住 ✓）")
    check("item, self._vt_frame = self._vt_frame, None" in src,
          "喂帧不是「**只留最新一帧**」✗（排队 = 画面越拖越旧 ✓）")
    check("self._vt_yield" in src and "VT_FPS_UNDER_LAG" in src
          and 'perf.note("vt_yield"' in src,
          "「**本机积压时测谎降速让路**」没接上 ✗（可选功能会一直抢实时链路 ✓）")
    # ⭐⭐ 用户口径（2026-10-10 ✓ 原话："**「自动测谎」不能关**，要不然这功能做了干嘛？"）：
    #   积压时**降速（2fps）**、**不是停掉** ✗；另外**常态也限速**（`vt_max_fps` 默认 10 ✓
    #   —— 判谎根本用不到 40fps ✓ 这是"不抢实时"的第一道闸 ✓）。
    check("vt_max_fps" in src and 'self._vt_cfg.get("max_fps"' in src,
          "测谎喂帧没限速（会跟着 40fps 一直抢 GPU/CPU ✗）")
    check("min(_want, VT_FPS_UNDER_LAG)" in src,
          "积压时**没降速**到 `VT_FPS_UNDER_LAG`（要么还在全速抢 ✗、要么被写成了停掉 ✗）"
          "—— 用户要的是「降速但仍能跑」✓")


def main():
    tests = [("① 模板几何：确认按钮比例量得出来", t_templates),
             ("⭐⭐ 实时链路契约：独立线程 / 只留最新一帧 / **积压时让路**（2026-10-10）",
              t_live_integration_yields_under_lag),
             ("⑩ 框坐标换算：追踪图 → 喂进去那张图（手算对数 ✓）", t_frame_rect),
             ("⑧ SDK 目录不对 ⇒ 人话错误不崩", t_bad_dir),
             ("③④⑤⑥⑦⑨ 真实素材 + 合成成功面板全链", t_real_video_and_confirm)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print("  [OK] %s" % name)
        except Exception as e:                                        # noqa: BLE001
            bad += 1
            print("  [NG] %s\n         %s: %s" % (name, type(e).__name__, e))
    print()
    if bad:
        print("vt_sdk 自检：%d/%d 通过，%d 条失败" % (len(tests) - bad, len(tests), bad))
        return 1
    print("vt_sdk 自检全部通过（%d 条）" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
