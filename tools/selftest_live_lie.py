# -*- coding: utf-8 -*-
"""`tools/live_lie.py`（**实时测谎测试 · 独立工具窗** ✓）的自检 —— 离屏 ✓ 不弹窗 ✗ 无需显示器 ✓。

这个工具现在接的是**实时源**（用户 2026-10-03 ✓ 原话："**不用选择素材**，我的需求是**实时计算**，
类似数据工作台的**框选本地屏幕区域**或者**收流**" ✓）⇒ 要害有两条，都得**量出来** ✓：
  ① **框选那块真的把外面的东西隔离掉了** ✓（合成夹具：ROI 内一个浅白块 + ROI 外一个纯白块 ✓
     见 `test_roi_isolation` ✓）；
  ② **加工尺度**：屏幕 / 流抓来是 1080p+，而 `LieTracker` 是纯 CPU ⇒ 必须先缩 ✗ 否则"实时"就没了
     （`fit_proc` ✓ 见 `test_fit_proc` ✓）。
其余（夹矩形 / 裁 ROI / 引擎状态机 / 录像源节流）都是**纯逻辑** ✓ 顺手钉掉 ✓。
⚠ 屏幕源 / 收流源本身**不在这里真跑**（要 win32 / 网络 ✗）—— 它们的"取帧"属于环境，不是逻辑 ✓；
  逻辑那半（裁 / 缩 / 算）全被下面钉住了 ✓。

跑法：`python -m tools.selftest_live_lie`（要连窗口一起验 ⇒ 加 `--window` ✓）
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.live_lie import (MIN_SIDE, PROC_W, LiveLieEngine,    # noqa: E402
                            ScreenSource, StreamSource, VideoSource,
                            clamp_rect, crop_roi, fit_proc, load_lie_cfg)

_FAILED = 0


def check(ok, msg):
    global _FAILED
    print(("  [OK] " if ok else "  [NG] ") + msg)
    if not ok:
        _FAILED += 1


def test_clamp_rect():
    """矩形夹进画面 / 屏幕 ✓（⚠ **屏幕坐标允许为负** ✗ —— 多显示器左边那块就是这样 ✓）。"""
    check(clamp_rect((10, 20, 100, 80), 400, 300) == (10, 20, 100, 80),
          "画面里正常框 ⇒ 原样 ✓")
    check(clamp_rect((-200, -100, 300, 200), 400, 300) == (-200, -100, 300, 200),
          "**屏幕坐标可以是负的** ⇒ 原样保留 ✓（虚拟屏幕左边/上边的显示器 ✓ 别把它夹成 0 ✗）")
    check(clamp_rect((300, 200, 500, 500), 400, 300) == (300, 200, 100, 100),
          "超出右下 ⇒ 宽高截到剩多少算多少 ✓（还剩 100×100 ≥ %d ⇒ 采用 ✓）" % MIN_SIDE)
    check(clamp_rect((10, 10, MIN_SIDE - 1, 100), 400, 300) is None
          and clamp_rect((10, 10, 100, MIN_SIDE - 1), 400, 300) is None,
          "**太小**（任一边 < %d px）⇒ `None` ✓（那点地方放不下目标 ⇒ 硬跑只会白算 ✗）" % MIN_SIDE)
    check(clamp_rect(None, 400, 300) is None and clamp_rect((1, 2, 3, 4), 0, 300) is None
          and clamp_rect(("a", "b", "c", "d"), 400, 300) is None,
          "没给 / 尺寸为 0 / 一堆非数字 ⇒ 一律 `None` ✓ 不崩 ✗")


def test_crop_roi():
    """裁 ROI（**裁是调用方的事** ✗ 源不管 ROI ✓）。"""
    _f = np.zeros((60, 120, 3), np.uint8)
    _f[10:30, 20:50] = 200
    _r = crop_roi(_f, (20, 10, 30, 20))
    check(_r is not None and _r.shape[:2] == (20, 30) and int(_r.mean()) == 200,
          "裁出来的就是那一块（形状 %s ✓）" % (None if _r is None else _r.shape[:2],))
    check(crop_roi(None, (0, 0, 60, 60)) is None and crop_roi(_f, None) is None
          and crop_roi(_f, (200, 200, 50, 50)) is None,
          "帧为空 / 没 ROI / ROI 全在画外 ⇒ `None` ✓（空数组也拦掉 ✓）")
    check(crop_roi(_f, (-5, 0, 60, 60)) is None,
          "**负起点**（那是屏幕坐标的写法 ✗）⇒ `None` ✓ —— 画面里的 ROI 不该有负数 ✓")


def test_fit_proc():
    """⭐⭐ **加工尺度**：屏幕 / 流是 1080p+，`LieTracker` 是纯 CPU ⇒ 必须先缩 ✗（用户要的"实时"靠它 ✓）。"""
    _big = np.zeros((1080, 1920, 3), np.uint8)
    _p = fit_proc(_big)
    check(_p is not None and _p.shape[1] == PROC_W
          and abs(_p.shape[0] - round(1080 * PROC_W / 1920.0)) <= 1,
          "1920×1080 ⇒ 缩到宽 %d（高按比例 %d ✓ 长宽比保住 ✓）" % (PROC_W, _p.shape[0]))
    _small = np.zeros((120, 200, 3), np.uint8)
    check(fit_proc(_small) is _small,
          "本来就比 %d 小 ⇒ **原样别动** ✓（放大既慢又没信息 ✗）" % PROC_W)
    _tiny = np.zeros((40, 60, 3), np.uint8)
    check(fit_proc(_tiny) is _tiny,
          "太小（宽 < 2×%d）⇒ 也不放大 ✓（否则把插值噪声喂给追踪器 ✗）" % MIN_SIDE)
    check(fit_proc(None) is None, "没图 ⇒ `None` ✓")


def _synth(n=60, w=140, h=70, sp=1.2):
    """合成夹具：**左边一个浅白块**（ROI 内 ✓）+ **右边一个纯白块**（ROI 外 ✓），两块都在动。

    ⚠ 刻意把"哪个更白"拉开 ✓ —— 只看 ROI 就追左边那个 ✓、看全图会去追右边那个 ✗
      ⇒ 这才量得出"框选到底隔没隔开" ✓。
    """
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        _lx = int(18 + i * sp)
        cv2.rectangle(f, (_lx - 6, 28), (_lx + 6, 40), (232, 232, 232), -1)     # ROI 内
        _rx = int(100 + i * sp)
        cv2.rectangle(f, (_rx - 7, 28), (_rx + 7, 40), (255, 255, 255), -1)     # ROI 外
        out.append(f)
    return out


def test_engine_state():
    """引擎：惰性建 ✓ 跑得起来 ✓ `reset` 之后**回到「重新建档」** ✓。"""
    _e = LiveLieEngine(cfg={})
    _fr = _synth()
    _st = {}
    _last = None
    for _i, _f in enumerate(_fr):
        _r = _e.step(_f, _i / 30.0)
        if _r is None:
            continue
        _st[_r[0]["state"]] = _st.get(_r[0]["state"], 0) + 1
        _last = _r
    check(_e.size is not None and _e.size[0] == _fr[0].shape[1],
          "第一帧**惰性建**（处理尺度 = 这一帧的尺寸 %s ✓ —— 之前不知道高宽 ✓）" % (_e.size,))
    check(_st.get("track", 0) > 20 and _last is not None and _last[1] is not None,
          "喂真有目标的序列 ⇒ 跑得起来（状态分布 %s ✓ —— ⚠ 这张合成图 140×70 比 ROI 那份**大** ✓"
          " ⇒ 建档会多花十几帧 ✓ 正常 ✓）" % (_st,))
    check(abs(float(_last[1][1]) - 34.0) < 3.0,
          "纵坐标跟着目标走（y ≈ 34 ✓ 实测 %.1f）" % float(_last[1][1]))
    _e.reset()
    _r0 = _e.step(_fr[0], 0.0)
    check(_e.runner is not None and _r0 is not None and _r0[0]["state"] == "init",
          "`reset` ⇒ **重新建档**（首帧 = init ✓ 不是接着上一段跑 ✗）+ 惰性重建 ✓")
    check(_e.n == 1, "拍数跟着 `reset` 归零（现在 %d ✓）" % _e.n)


def test_detector_wiring():
    """⭐⭐⭐ **检测器有没有真的接进链**（用户 2026-10-03 ✓ 原话："实时测谎：**没有任何检出框**" ✓）。

    这是上一轮那个取舍的代价 ✗：`dets=None` ⇒ 融合 / 分离 / 砖表 / 点选**全不工作** ✗
    ⇒ 现在接上了 ✓，但**"接上了"这件事必须量出来** ✗（不然哪天回归了没人知道 ✗）。
    用一个**桩检测器**（不依赖模型 ✓ 离屏就能测 ✓）：
      · 接上 ⇒ `Runner` 拿到的 `boxes` **非空** ✓（检出框真的进了链 ✓）；
      · 不接 ⇒ `boxes` **空** ✓（= 用户截图里那种"一个框都没有" ✓ 正是这条在钉的 ✓）。
    ⚠ 桩的返回布局必须是 `(cls, cx, cy, w, h, conf)` ✗ —— `Runner` 里是 `b[1:5]` 取的 ✓
      （写错了就会静默变成"框全是 0" ✗ 踩过同类坑 ✓）。
    """

    class _Stub(object):
        """假检测器：恒给一个盖住合成目标的框 ✓。"""

        def __init__(self, box):
            self.box = box
            self.calls = 0

        def detect(self, _img):
            self.calls += 1
            return [self.box]

        def close(self):
            pass

    _fr = _synth(12)
    _h, _w = _fr[0].shape[:2]
    _stub = _Stub((0, 34.0, 34.0, 26.0, 16.0, 0.9))       # (cls, cx, cy, w, h, conf) ✓
    _e = LiveLieEngine(cfg={}, detector=_stub)
    _boxes = None
    for _i, _f in enumerate(_fr):
        _r = _e.step(_f, _i / 30.0)
        if _r is not None:
            _boxes = _r[4]
    check(_stub.calls >= len(_fr) - 1,
          "接上检测器 ⇒ **每拍都调它**（调了 %d 次 ✓）" % _stub.calls)
    check(bool(_boxes),
          "接上 ⇒ `boxes` **非空**（%s ✓ —— 检出框真的进了判定链 ✓）" % (_boxes,))
    _e2 = LiveLieEngine(cfg={})
    _b2 = None
    for _i, _f in enumerate(_fr):
        _r2 = _e2.step(_f, _i / 30.0)
        if _r2 is not None:
            _b2 = _r2[4]
    check(not _b2,
          "**不接** ⇒ `boxes` 空（%s ✓）—— 就是用户截图里那种「一个框都没有」✓ "
          "（关掉检测器 = 只剩白块跟踪 ✓ 这条把两种状态都钉住 ✓）" % (_b2,))
    del _h, _w


def test_roi_isolation():
    """⭐⭐⭐ **要害**：框出来的那块**真的把外面隔开了** ✓（用户："框选窗口的模式" / "框选本地屏幕区域" ✓）。

    两条一比（同 `_synth` 夹具 ✓）：
      · **ROI = 左半** ⇒ 追踪器只看得见里头那个 ⇒ 末位置 x 落在 **ROI 内** ✓；
      · **不裁**（= 整幅 ✓ 相当于"没框"✗）⇒ 它去追**更白**的那个 ⇒ 跑出左半 ✓。
    """
    def _run(crop):
        _e = LiveLieEngine(cfg={})
        _last = None
        for _i, _f in enumerate(_synth()):
            _roi = crop_roi(_f, crop) if crop is not None else _f
            _r = _e.step(_roi, _i / 30.0)
            if _r is not None and _r[1] is not None:
                _last = _r
        return _last

    _in = _run((0, 0, 64, 70))
    check(_in is not None and float(_in[1][0]) < 64.0,
          "ROI = 左半 ⇒ 末位置 x = %.1f **落在 ROI 内**（< 64 ✓ 追的是里头那个浅白的 ✓）"
          % (float(_in[1][0]) if _in is not None else -1.0))
    _all = _run(None)
    check(_all is not None and float(_all[1][0]) > 64.0,
          "**对照**：不裁（整幅）⇒ 末位置 x = %.1f **跑出左半**（> 64 ✓ 去追右边那个纯白的 ✓）"
          " —— 两条一比 ⇒ **框选真的隔离了区域外的东西** ✓ 不是「我觉得」✓"
          % (float(_all[1][0]) if _all is not None else -1.0))


def test_video_source():
    """录像源（**只给自检 / 命令行调试** ✓ 界面上不暴露 ✓）—— 按**素材自己的时间戳**节流 ✓。"""
    _src = Path(__file__).resolve().parent.parent / "datasets" / "liedetectorVideo" / "10月2日.mp4"
    if not _src.exists():
        check(True, "跳过录像源（没找到素材 %s ✓）" % _src.name)
        return
    _v = VideoSource(str(_src))
    _f1 = _v.grab()
    check(_f1 is not None and _f1.ndim == 3,
          "第一帧拿得到（%s ✓）" % (None if _f1 is None else _f1.shape,))
    _f2 = _v.grab()
    _ts1 = float(_v.ts[1]) if len(_v.ts) > 1 else 0.0
    check(_f2 is None or _ts1 <= 0.001,
          "紧接着再抓 ⇒ **没到点就不给** ✓（这一拍该等到 %.3f s ✓ 这就是「像实时」那把尺 ✓）" % _ts1)
    check(VideoSource(str(_src)).done() is False, "还没放完 ⇒ `done()` = False ✓")
    # ⚠⚠ **不能靠"连着 grab 到 None 为止"来放完** ✗✗（踩过 ✓）：有**时间戳节流** ⇒ 第一次之后
    #   就一直是 `None`（还没到点 ✓）⇒ 那个循环**立刻退出** ✗ 等于没放 ✓。
    #   ⇒ 把"起始时刻"往过去挪一大截（= 假装已经过了很久 ✓）⇒ 每次都能取到帧 ✓ 几毫秒放完 ✓。
    _v2 = VideoSource(str(_src))
    _v2._t0 = time.time() - 999.0
    for _ in range(len(_v2.frames) + 4):
        _v2.grab()
    check(_v2.done(), "放完 ⇒ `done()` = True ✓（调用方据此收尾 ✓ 不空转 ✗ 实测 i=%d/%d ✓）"
          % (_v2.i, len(_v2.frames)))
    check(ScreenSource((0, 0, 100, 100)).rect == (0, 0, 100, 100)
          and StreamSource("").url,
          "两个实时源的构造是纯的（不吃系统库 ✓ 只有真 `grab` 才 import ✓）")


def test_cfg():
    """参数**跟着 `lie_demo` 那份界面配置走**（拿不到就空 ⇒ 用追踪器默认 ✓ 不硬编 ✗）。"""
    _c = load_lie_cfg()
    check(isinstance(_c, dict) and all(isinstance(k, str) for k in _c),
          "`load_lie_cfg()` 回 dict（当前拿到 %d 项 ✓ 拿不到就是空 ✓ 照样能跑 ✓）" % len(_c))


def test_window_smoke():
    """**窗口那一层**也要真跑一次（纯逻辑绿了 ≠ 建窗 / 线程 / 渲染不报错 ✗）。

    ⚠ 用 `--video`（录像当实时 ✓）跑 —— 屏幕 / 收流那条在这台机器上不一定有环境 ✗，但它们
      **共用同一套 UI / 线程 / 绘制** ✓ ⇒ 这条冒烟覆盖的正是那部分 ✓。
    ⚠ 必须**自动收尾**（`QTimer` ✓）；且 `QTimer.singleShot` 要在 `QApplication` **之后**排 ✗
      （没有 app 时排的定时器是**哑的** ⇒ `exec_()` 永不返回 ⇒ 命令行挂死 ✗ 踩过 ✓）。
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _src = (Path(__file__).resolve().parent.parent
            / "datasets" / "liedetectorVideo" / "10月2日.mp4")
    if not _src.exists():
        check(True, "跳过窗口冒烟（没找到素材 %s ✓）" % _src.name)
        return
    try:
        import argparse

        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        import tools.live_lie as LL
    except Exception as _e:                     # noqa: BLE001 —— 环境不齐 ⇒ 跳过 ✓
        check(True, "跳过窗口冒烟（环境不齐：%s ✓）" % _e)
        return
    # ⚠ `no_dets=True`：冒烟**不建真检测器**（那要模型 + 子进程 ✗ 慢且和"窗口"无关 ✓）
    #   —— 检测器那条链由 `test_detector_wiring` 用**桩**钉 ✓ 两件事分得开 ✓。
    _args = argparse.Namespace(video=str(_src), rect="100,60,300,240", loop=False,
                               autostart=True, stream="", format=None,
                               no_dets=True, conf=0.25, weights="")
    _out = {}

    def _find():
        return [w for w in QApplication.topLevelWidgets()
                if str(w.windowTitle()).startswith("实时测谎测试")]

    def _report():
        _ws = _find()
        if _ws:
            _w = _ws[0]
            _out["crop"] = _w.crop
            _out["n"] = (None if _w.worker is None else None)
            _out["engine_n"] = _w.engine.n
            _out["state"] = None
            _out["px"] = (_w.view.pixmap() is not None and not _w.view.pixmap().isNull())
        QApplication.instance().quit()

    _app = QApplication.instance() or QApplication(sys.argv)
    QTimer.singleShot(6000, _app.quit)          # 兜底退出（万一 `_report` 没跑到 ✓）
    QTimer.singleShot(2600, _report)
    try:
        LL.run_window(_args)
    except Exception as _e:                     # noqa: BLE001
        _out["err"] = repr(_e)
    check("err" not in _out, "窗口跑起来**没报错**（%s ✓）" % (_out.get("err") or "无异常"))
    check(_out.get("crop") is not None,
          "`--rect` 生效 ⇒ 画面 ROI = %s ✓" % (_out.get("crop"),))
    check(_out.get("engine_n", 0) > 0,
          "取帧线程真的在跑（引擎走了 %s 拍 ✓）" % (_out.get("engine_n"),))
    check(_out.get("px") is True,
          "画面**真的画出来了**（不是「建了窗但一片空白」✗）")


def main():
    import argparse
    _ap = argparse.ArgumentParser(description="`tools/live_lie.py` 自检")
    _ap.add_argument("--window", action="store_true",
                     help="**额外**跑一次离屏窗口冒烟（约 3 秒；会建一个看不见的窗 ✓ "
                          "默认**不跑** ✗ —— 它要进 Qt 事件循环，别拖累日常跑自检 ✓）")
    _a = _ap.parse_args()
    print("实时测谎测试（独立工具）自检：")
    test_clamp_rect()
    test_crop_roi()
    test_fit_proc()
    test_engine_state()
    test_detector_wiring()
    test_roi_isolation()
    test_video_source()
    test_cfg()
    if _a.window:
        test_window_smoke()
    else:
        print("  [--] 跳过**窗口冒烟**（那一步要进 Qt 事件循环 ⇒ 默认不跑 ✓ "
              "要验就 `python -m tools.selftest_live_lie --window` ✓）")
    if _FAILED:
        print("自检：%d 条失败" % _FAILED)
        return 1
    print("实时测谎测试自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
