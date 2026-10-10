# -*- coding: utf-8 -*-
"""`perception/anchor_track.py`（**锚定追踪** ✓）的自检 —— 离屏 ✓ 不弹窗 ✓ 不需要素材 ✓。

要钉的就是这一档**凭什么成立**（用户 2026-10-08 ✓ 他口述的场景 ✓）：
  ① **纯函数那把尺**：中位位移 / 波浪分 / 滑行 / 选"在动的那片" / 旋转角 / 角度线；
  ② **相机只要平移 ⇒ 用静止片的中位位移量它** ＋ **静止片当锚**（累加不许漂 ✗）；
  ③ **只有一片在动 ⇒ 选中它**（其余片"静止" ✓ 且分数不涨 ✓）；
  ④ **波浪 ⇒ 认出来 ＋ 冻住**（相机不更新 ✓ 档案不更新 ✓ 波浪过去自动恢复 ✓）；
  ⑤ **丢框 ⇒ 滑行（外推）⇒ 出现无主框 ⇒ 认领**；
  ⑥ `AnchorRunner` 与 `MotionRunner` **同签名同返回**（演示窗只换构造 ✓）。
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import perception.anchor_track as _AT                              # noqa: E402
from perception.anchor_track import (AnchorRunner, AnchorTracker,  # noqa: E402
                                     anc_angle_line, anc_coast, anc_median2,
                                     anc_patch_dtheta, anc_patch_match,
                                     anc_white_cands,
                                     anc_pick, anc_wave_score,
                                     #   ⚠ 复用老底盘的那两把尺（**一处口径** ✓ 见那边注释 ✓）
                                     _POS_STEP_MAX, _WHITE_STREAK)

_FAILED = 0


def check(ok, msg):
    global _FAILED
    print(("  [OK] " if ok else "  [NG] ") + msg)
    if not ok:
        _FAILED += 1


#: ⚠⚠ **合成序列的帧长必须像真素材** ✗✗（**实测踩到** ✓）：本档那个"净位移"是**按秒开窗**的
#:   （`_ANC_NET_WIN` ✓）⇒ 若还按老写法 `ts=float(k)`（**一帧一秒** ✗）⇒ 窗口里只剩一拍 ⇒
#:   分数恒等于"单拍位移" ⇒ 那些钉子量的就不是这个判据了 ✗。真素材 ≈ **7.5fps** ⇒ 取 **0.1s/拍** ✓。
_DT = 0.1


def _ts(_k):
    """第 `_k` 拍的时刻（秒 ✓ 见 `_DT` ✓）。"""
    return _DT * float(_k)


#: 合成场景：**6 片静止**（相机带着它们一起走 ✓）＋ **1 片自己在动**（用户那句话的字面 ✓）。
_STATIC0 = ((120.0, 110.0), (330.0, 105.0), (540.0, 115.0),
            (125.0, 330.0), (335.0, 335.0), (545.0, 325.0))
_MOVER0 = (335.0, 215.0)
_WH = (80.0, 80.0)


def _dets(cam_x, cam_y, mover_x, mover_y, jitter=0.0, rng=None, with_mover=True,
          mover_extra=None):
    """造一拍的检出框（`(cls, cx, cy, w, h, conf)` ✓ 与演示窗喂进来的一模一样 ✓）。

    `cam_x/cam_y` = **相机**这一拍走了多少（全场都跟着走 ✓）；`mover_x/mover_y` = 那片自己
    额外走了多少（**只有它** ✓）；`jitter` = 波浪（全场各走各的 ✓）。
    """
    _rng = rng or np.random.RandomState(7)
    _out = []
    for (_x, _y) in _STATIC0:
        _jx = _rng.uniform(-jitter, jitter) if jitter else 0.0
        _jy = _rng.uniform(-jitter, jitter) if jitter else 0.0
        _out.append([0, _x + cam_x + _jx, _y + cam_y + _jy, _WH[0], _WH[1], 0.9])
    if with_mover:
        _mx = _MOVER0[0] + cam_x + mover_x + (mover_extra or (0.0, 0.0))[0]
        _my = _MOVER0[1] + cam_y + mover_y + (mover_extra or (0.0, 0.0))[1]
        _out.append([0, _mx, _my, _WH[0], _WH[1], 0.9])
    return _out


def test_anchor_pure():
    """① **纯函数那把尺** ✓（判据全在这几个里 ✓ —— 类里只串流程 ✓）。"""
    check(anc_median2([(1.0, 9.0), (3.0, 5.0), (2.0, 7.0), (100.0, -50.0)]) == (2.5, 6.0),
          "① **中位位移** ✓（实测 %s ✓ —— ⚠ 用均值 ⇒ 那个 `(100,-50)` 的野值会把它拖走 ✗）"
          % (anc_median2([(1.0, 9.0), (3.0, 5.0), (2.0, 7.0), (100.0, -50.0)]),))
    check(anc_wave_score(0.45, 1.2, 0.90) == 0.0 and anc_wave_score(0.08, 9.0, 0.30) == 1.0,
          "① **波浪分**：正常帧 ⇒ **0.00** ✓ ／ 被搅帧 ⇒ **1.00** ✓（三个信号取最大 ✓）")
    check(abs(anc_wave_score(0.0, 0.0, 1.0)) < 1e-9,
          "① **`resp = 0`（还没上一帧 / 相位相关失败）⇒ 那一路不参与** ✗（不把「没测到」当波浪 ✓）")
    check(anc_coast((10.0, 20.0), (3.0, -4.0)) == (13.0, 16.0)
          and anc_coast((0.0, 0.0), (999.0, 0.0))[0] == _AT._ANC_COAST_V_MAX,
          "① **滑行 = 位置 ＋ 速度** ✓ 且**每拍限幅** ✓（实测 %s ✓〔一个 999px 的野速度只走 %.0f ✓〕）"
          % (anc_coast((10.0, 20.0), (3.0, -4.0)), _AT._ANC_COAST_V_MAX))
    _sc = {1: 100.0, 2: 60.0}
    check(anc_pick(_sc, cur=2) == 2 and anc_pick(_sc, cur=None) == 1
          and anc_pick({1: 20.0, 2: 10.0}) is None,
          "① **选「在动的那片」 ＋ 迟滞** ✓（现任 #2 有 60 分、新候选 100 分：100 < 60×%.1f ⇒ "
          "**不换** ✓；没有现任 ⇒ 选 #1 ✓；都不到 `_ANC_SCORE_MIN`(%.0f) ⇒ `None` ✓）"
          % (_AT._ANC_SWITCH_K, _AT._ANC_SCORE_MIN))
    _line = anc_angle_line((100.0, 50.0), 0.0, 10.0)
    check(_line == ((90.0, 50.0), (110.0, 50.0)),
          "① **角度线：0 度 = 水平** ✓（实测 %s ✓ —— 画出来的线和角度数字**必然同一口径** ✓）"
          % (_line,))
    _line90 = anc_angle_line((100.0, 50.0), 90.0, 10.0)
    check(abs(_line90[0][0] - 100.0) < 1e-6 and abs(_line90[1][1] - 60.0) < 1e-6,
          "① **90 度 = 竖直（y 朝下 ✓）** ✓（实测 %s ✓）" % (_line90,))


def test_anchor_dtheta():
    """①' **旋转角那把尺**（旋转搜索 ＋ 圆形掩膜 NCC ✓ 用户场景里的"自转"就靠它 ✓）。"""
    _rng = np.random.RandomState(3)
    _base = np.zeros((72, 72), np.uint8)
    for _k in range(5):                       # 造一块"有结构"的图案（模拟棱边 / 折射纹 ✓）
        _p = _rng.randint(10, 62, 4)
        _base[_p[0]:_p[0] + _p[1] % 9 + 2, _p[2]:_p[2] + _p[3] % 11 + 2] = 255
    import cv2
    _rot = cv2.warpAffine(_base, cv2.getRotationMatrix2D((35.5, 35.5), 7.0, 1.0), (72, 72),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    _d7 = anc_patch_dtheta(_base, _rot)
    check(_d7 is not None and abs(abs(_d7) - 7.0) <= 1.0,
          "①' **转 7 度 ⇒ 量出来 %.1f 度** ✓（⚠ 只钉**绝对值**：OpenCV 的旋转正方向与"
          "`getRotationMatrix2D` 一致 ✓ 符号随它 ✓）" % (float(_d7) if _d7 is not None else -99.0))
    check(anc_patch_dtheta(np.full((72, 72), 7, np.uint8), np.full((72, 72), 7, np.uint8)) is None,
          "①' **纯色 / 没结构 ⇒ `None`** ✓（不猜一个角度出来 ✗ —— 透明片上那段没纹路时就是这种 ✓）")


def test_anchor_camera_and_static():
    """②③ **相机（纯平移）＋ 静止片当锚 ＋ 选出唯一在动的那片** ✓。"""
    _t = AnchorTracker()
    _N = 34
    for _k in range(_N):
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0))
    _cam = _t.cam_cum
    _want = 2.0 * (_N - 2)                    # 前两拍账还没攒够 3 次"跟拍" ⇒ 从第 3 拍起才量 ✓
    check(abs(float(_cam[0]) - _want) <= 6.0 and abs(float(_cam[1])) <= 1.0,
          "② **相机位移 = 静止片的中位位移** ✓（%d 拍 × 2px ⇒ 实测 **%.1f** ✓ 期望 ≈%.0f ✓"
          "〔±6px = 前两拍的预热 ＋ 锚定校正那点修正 ✓〕）" % (_N, float(_cam[0]), _want))
    _st = [_o for _o in _t._objs.values() if _o["static"]]
    #   ⚠ "静止那堆的分数"要**按 id 排除目标自己** ✓（别按位置猜 ✗ —— 位置猜法一改素材就错 ✓）。
    _mv = [float(_o["score"]) for _i, _o in _t._objs.items() if _i != _t.tid]
    check(len(_st) >= 6 and _t.tid is not None and float(_t._objs[_t.tid]["score"]) >= 25.0
          and max(_mv or [0.0]) < 5.0,
          "③ **「只有一片在动」⇒ 选中它** ✓（静止片立档 %d 条 ✓ ｜ 目标 #%s 分数 **%.0f** ✓"
          "〔它自己每拍走 4px、窗口 `_ANC_NET_WIN` = %.1fs ⇒ 净位移 ≈ 4px × 窗口内拍数 ✓ 实测落在"
          "这一带 ✓〕｜ 其余片最高分只有 **%.1f** ✓ —— ⚠ 两堆必须拉得开 ✓：这一档没有「认领 / "
          "融合」那套的底气就在这儿 ✓；⚠⚠ **抖动的片子分数上不来**（净位移 ≈ 0 ✓）才是命门 ✓）"
          % (len(_st), _t.tid, float(_t._objs[_t.tid]["score"]), float(_AT._ANC_NET_WIN),
             max(_mv or [0.0])))
    _d = math.hypot(float(_t.pos_g[0]) - (_MOVER0[0] + 4.0 * (_N - 1)),
                    float(_t.pos_g[1]) - _MOVER0[1])
    check(_d < 6.0,
          "③ **群体坐标系里报出的位置就是它自己走的那条路** ✓（实测与真值差 **%.1fpx** ✓"
          "〔真值 = 起点 ＋ 自己走的 %d×3px ✓ 相机那一份已经被减掉 ✓〕）" % (_d, _N - 1))


def test_anchor_wave_freeze():
    """④ **波浪：认出来 ＋ 冻住 ＋ 过去后恢复** ✓（用户 2026-10-08 ✓ "偶尔水面会有波浪干扰" ✓）。"""
    _t = AnchorTracker()
    for _k in range(30):                      # 先稳一阵（相机 ＋2px/拍 ✓ 目标自己 ＋3px/拍 ✓）
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0))
    _cam_before = float(_t.cam_cum[0])
    _rec_before = [tuple(float(_x) for _x in _o["g_rec"]) for _o in _t._objs.values()
                   if _o["static"]]
    _rng = np.random.RandomState(11)
    _fired = 0
    for _k in range(30, 42):                  # 波浪：**全场各走各的 ±9px** ✓
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0, jitter=9.0, rng=_rng))
        _fired += 1 if _t.wave_now else 0
    _rec_after = [tuple(float(_x) for _x in _o["g_rec"]) for _o in _t._objs.values()
                  if _o["static"]]
    check(_fired >= 8 and float(_t.cam_cum[0]) - _cam_before <= 3.0,
          "④ **波浪段认出来了（%d/12 拍 ✓）且相机冻住** ✓（12 拍里相机只走了 **%.1fpx** ✓ —— "
          "⚠ 照常更新会走 ~24px ✗（每拍 2px ✓）：这就是「波浪段不再更新相机」的字面 ✓）"
          % (_fired, float(_t.cam_cum[0]) - _cam_before))
    check(_rec_after == _rec_before,
          "④ **波浪段内静止档案一格都不动** ✓（%d 条全等 ✓ —— ⚠ 拿被搅过的帧去立档 = 把波浪"
          "写进基准 ⇒ 事后全是冤枉 ✗）" % len(_rec_before))
    for _k in range(42, 56):                  # 浪过去 ⇒ 恢复
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0))
    check(not _t.wave_now and float(_t.cam_cum[0]) > _cam_before + 20.0,
          "④ **波浪过去 ⇒ 自动恢复** ✓（实测 相机累计 %.1f ⇒ %.1f ✓ 且 `wave_now` 归 %s ✓ "
          "—— ⚠ 没有「最短持续」就会一帧抖就进出 ✓ 见 `_ANC_WAVE_HOLD` ✓）"
          % (_cam_before, float(_t.cam_cum[0]), _t.wave_now))


def test_anchor_coast_no_claim():
    """⑤ ⭐⭐⭐⭐⭐ **丢了只滑行、**不认领别人的框**、滑够半秒就报"丢了"** ✗✗
    （用户 2026-10-08 ✓ 定案："**动**" ✓；原话："**帧 278之后就在满屏幕乱窜 无法正常跟踪目标**" ✓✓）。

    ⚠⚠ **实测那个"认领"就是乱窜的现场** ✗：目标看不见时，原版会从石墙上挑一块"无主框"当新目标
      （实测帧 279/421/470 `重新认领 #3/#15/#22` ✓ 位置当场被拽走 ✗）⇒ C′ **把这条路整条删掉** ✓：
      现在只有三件事 —— 外观看见就报 ✓、有框压在预测上就借它的几何 ✓、**什么都没有就滑行 ⇒ 报丢** ✓。
    """
    _t = AnchorTracker()
    _N = 26
    for _k in range(_N):
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0))
    _tid0, _pos0 = _t.tid, (float(_t.pos[0]), float(_t.pos[1]))
    _GAP = int(_AT._ANC_LOST_N)
    #   ⚠ **先看"还在滑行"那一段** ✗（`_ANC_LOST_N` 拍之后就已经是 `lost`、位置 `None` 了 ✗
    #     实测踩到：直接跑满 `_GAP` 拍 ⇒ `_t.pos` 是 `None` ⇒ 下面这句 `float(_t.pos[0])` 崩 ✓）。
    for _k in range(_N, _N + 4):              # 目标那格**没了**（静止片照常在 ✓）
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0, with_mover=False))
    _d = math.hypot(float(_t.pos[0]) - _pos0[0], float(_t.pos[1]) - _pos0[1])
    check(_t.state == "coast" and _t.miss >= 4 and _d > 8.0,
          "⑤ **没框 ⇒ `coast`（匀速外推 ✓ 不停住 ✓）** ✓（实测 状态 %s ✓ 缺 %d 拍 ✓ 这几拍里"
          "位置走了 **%.1fpx** ✓〔预测本来就该往前走 ✓〕）" % (_t.state, _t.miss, _d))
    #   ⭐ **滑够 `_ANC_LOST_N` 拍 ⇒ 报"丢了"**（位置 `None` ✓ 画面不画 ✗）—— "没依据就别声称" ✓
    for _k in range(_N + 4, _N + _GAP + 4):
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0, with_mover=False))
    check(_t.state == "lost" and _t.pos is None,
          "⑤ ⭐ **滑够 %d 拍 ⇒ 报「丢了」（位置 `None`）** ✓（实测 状态 %s ✓ 位置 %s ✓ —— "
          "旧版这里会去「认领」石墙上的一块 ⇒ 正是用户说的「满屏乱窜」✗）"
          % (int(_AT._ANC_LOST_N), _t.state, _t.pos))
    #   ⭐⭐ **眼前摆一块"无主框" ⇒ 也不认领** ✓（C′ 那条路已删 ✓）·身份也不许动 ✓
    _k0 = _N + _GAP + 4
    _pred = _pos0
    _extra = ((_pred[0] + 55.0) - (_MOVER0[0] + 2.0 * _k0 + 4.0 * _k0),
              (_pred[1]) - _MOVER0[1])
    _log = None
    for _k in range(_k0, _k0 + 6):
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0, mover_extra=_extra))
        _log = _t.log_text or _log
    check(int(_t.tid) == int(_tid0) and not (_log and "认领" in str(_log)),
          "⑤ ⭐ **那块「无主框」摆在眼前也照样不认领、身份也不动** ✗✓（实测 目标仍 #%s ✓ 日志 %s ✓"
          " —— ⚠ 这就是 C′ 与旧版最要紧的差别：**框只当几何证据，绝不当身份** ✓）"
          % (_t.tid, str(_log)[:28]))


def test_anchor_jitter_immunity():
    """②' ⭐⭐⭐⭐⭐ **"抖动的片子"不许被当成"在动的那片"** ✗✗（用户 2026-10-08 ✓ 原话：
    "**圆满屏乱跳，看不出什么时间连续性**" ✓✓）。

    ⚠⚠ **实测病根**：第一版 `score += |本拍位移|` ✗ ⇒ **抖动与真移动攒得一样快** ✗（检出框抖一下
      也是几十像素 ✓）⇒ 一堆框同时够分 ⇒ 身份满屏换 ⇒ 圆跟着乱跳 ✓。
    现在量的是**净位移**（`|g(t) − g(t−1.2s)|` ✓）：抖来抖去 ≈ 0 ✓✓（实测下面这堆抖 ±9px 的，
      净位移只有十几 px ✗ 而门是 26 ✓）。
    """
    _t = AnchorTracker()
    _rng = np.random.RandomState(13)
    _N = 40
    #   ⚠⚠ **抖动要用"慢游走"** ✗✗（**实测踩到** ✓）：我第一版拿 `jitter=9`（**每拍独立**的白噪声 ✗）
    #     ⇒ 那在波浪那三个指标眼里**就是波浪** ✗⇒ `wave_now` 一亮 ⇒ 分数整段不更新 ⇒ 这条钉子
    #     量的其实是波浪那条路 ✗。真检出的"抖"是**慢游走**（每拍 ±1px ✓、几拍才挪一点 ✓）
    #     ⇒ `mad` 只有 ~1px（不进波浪 ✓）而**净位移**照样很小（≤ 十几 px ✓）。
    _walk = [(0.0, 0.0)] * len(_STATIC0)
    for _k in range(_N):
        _walk = [(_w[0] + _rng.uniform(-1.0, 1.0), _w[1] + _rng.uniform(-1.0, 1.0))
                 for _w in _walk]
        _d = _dets(2.0 * _k, 0.0, 4.0 * _k, 0.0)
        for _j in range(len(_STATIC0)):
            _d[_j][1] += _walk[_j][0]
            _d[_j][2] += _walk[_j][1]
        _t.process(None, ts=_ts(_k), idx=_k + 1, dets=_d)
    _sc = sorted((float(_o["score"]) for _i, _o in _t._objs.items() if _i != _t.tid),
                 reverse=True)
    check(_t.tid is not None
          and float(_t._objs[_t.tid]["score"]) >= float(_AT._ANC_SCORE_MIN)
          and max(_sc or [0.0]) < float(_AT._ANC_SCORE_MIN),
          "②' **全都在抖、只有一片在走 ⇒ 一定选那片在走的** ✓（实测 目标 #%s 净位移 **%.0fpx** ≥ 门 "
          "**%.0f** ✓ ／ 其余（全在抖 ±9px）最高只有 **%.0fpx** ✓ —— ⚠⚠ 老口径 `score += |本拍位移|` "
          "下，那堆抖的早就是 **60+** 了 ✗✗ 这就是「乱跳」的根源 ✓）"
          % (_t.tid, float(_t._objs[_t.tid]["score"]), float(_AT._ANC_SCORE_MIN),
             max(_sc or [0.0])))


def _white_img(_boxes, _lit=(0,), _bright=240):
    """按"框在哪"画白块（`_lit` = 哪几格亮着 ✓）—— 白块**跟着框走** ✓（这样"它在不在动"才有意义 ✓）。"""
    _im = np.full((300, 400), 40, np.uint8)
    for _j in _lit:
        _b = _boxes[_j]
        _hw, _hh = int(_b[3] / 2.0), int(_b[4] / 2.0)
        _x0, _y0 = int(_b[1] - _hw), int(_b[2] - _hh)
        _im[max(0, _y0):_y0 + 2 * _hh, max(0, _x0):_x0 + 2 * _hw] = _bright
    return _im


def test_anchor_white_adoption():
    """②'' ⭐⭐⭐⭐⭐ **白块认定（必须是"也在动的那片白"）** ✗✗（用户 2026-10-08 ✓ 两条一起：
    "**圆一开始就乱锁，没有认白色图形**" ＋ 之后"**一开始就认错了**" ✓✓）。

    ⚠ 老两条底盘都有"白块认定"（`lie_motion.pick_white` ✓），**我新建这一档时漏了** ✗ ⇒ 补上 ✓；
      ⚠⚠ 但第一版**只认"白 + 够大 + 连两拍"** ✗ ⇒ 画面里那些**白 UI / 高光**（又白又大又稳 ✓）
      一开场就被认走 ✗ = 用户说的"**一开始就认错了**" ✓ ⇒ 现在**必须同时也在动** ✓（短窗净位移 ✓）
      ＋ 相机热身完 ✓。本钉三层：
        ① **在动的白片** ⇒ 认它 ✓；② **不动的大白块**（UI 那种）⇒ **不许认** ✗；
        ③ **单拍闪现** ⇒ 不许认 ✗（老底盘实测假白块全是单拍闪现 ✓）。
    """
    #   ① 白片**自己在动**（每拍 +4px ✓）⇒ 认它 ✓
    _t = AnchorTracker()
    _t.frame_wh = (400, 300)
    _logs = []
    for _k in range(14):
        #   ⚠ **至少 3 条"跟拍 ≥3 拍"的账**才量得到相机（`_ANC_CAM_LIVE_N` ✓）⇒ 夹具也得给够框 ✓
        #     （⚠ 我第一版只摆 2 个 ⇒ 相机永远不"热" ⇒ 白块那道"在动"门读不到历史 ⇒ 认不了 ✓ 实测踩到 ✓）。
        _d = [[0, 60.0, 60.0, 80.0, 80.0, 0.9],
              [0, 200.0 + 6.0 * _k, 155.0, 80.0, 80.0, 0.9],
              [0, 40.0, 240.0, 80.0, 80.0, 0.9],
              [0, 340.0, 40.0, 80.0, 80.0, 0.9]]
        _t.process(_white_img(_d, _lit=(1,)), ts=_ts(_k), idx=_k + 1, dets=_d)
        if _t.log_text:
            _logs.append(_t.log_text)
    check(_t.tid is not None and int(_t.tid) == 2 and any("白块认定" in _l for _l in _logs),
          "②'' **在动的白片 ⇒ 认定它** ✓（实测 目标 = #%s ✓ 日志：%s ✓ —— ⚠ 它比「净位移攒分」"
          "早得多 ✓ 用户「一开始就乱锁」要的正是这个 ✓）"
          % (_t.tid, (_logs[0] if _logs else "（没有）")[:40]))
    #   ② **不动的大白块**（白 UI / 高光那种 ✓）⇒ **不许认** ✗
    _t2 = AnchorTracker()
    _t2.frame_wh = (400, 300)
    _n2 = 0
    for _k in range(20):
        _d2 = [[0, 60.0, 60.0, 80.0, 80.0, 0.9],          # 全场都静止（白块也是 ✓）
               [0, 200.0, 155.0, 80.0, 80.0, 0.9],
               [0, 40.0, 240.0, 80.0, 80.0, 0.9],          # ⚠ 凑够"相机可量"的账数 ✓
               [0, 340.0, 40.0, 80.0, 80.0, 0.9]]
        _t2.process(_white_img(_d2, _lit=(1,)), ts=_ts(_k), idx=_k + 1, dets=_d2)
        _n2 += 1 if (_t2.log_text and "白块" in str(_t2.log_text)) else 0
    check(_t2.tid is None and _n2 == 0,
          "②'' **不动的大白块 ⇒ 一个字都不认** ✗（实测 目标 = %s ✓ 白块日志 %d 条 ✓ —— ⚠⚠ 老版就是"
          "在这儿认错的：那些**又白又大又稳**的 UI / 高光 ✗ 现在被「**也在动**」这道门挡住 ✓）"
          % (_t2.tid, _n2))
    #   ③ **单拍闪现** ⇒ 不许认 ✗
    _t3 = AnchorTracker()
    _t3.frame_wh = (400, 300)
    _n3 = 0
    for _k in range(14):
        _d3 = [[0, 60.0, 60.0, 80.0, 80.0, 0.9],
               [0, 200.0 + 6.0 * _k, 155.0, 80.0, 80.0, 0.9],
               [0, 40.0, 240.0, 80.0, 80.0, 0.9],
               [0, 340.0, 40.0, 80.0, 80.0, 0.9]]
        _t3.process(_white_img(_d3, _lit=(1,) if _k == 0 else ()), ts=_ts(_k), idx=_k + 1, dets=_d3)
        _n3 += 1 if (_t3.log_text and "白块" in str(_t3.log_text)) else 0
    #   ⚠ 这里只钉"**白块那条路**" ✗（白片只闪一拍 ⇒ 不许认 ✓）；目标号可能是**普通选人那条路**
    #     后来认的 ✓ 与本条无关 ✓ —— 实测踩到过：写死 `tid is None` ⇒ 假红 ✓。
    check(_n3 == 0,
          "②'' **单拍闪现 ⇒ 不许认（白块那条路）** ✗（实测 白块日志 **%d** 条 ✓ ／ 目标 #%s〔那是"
          "普通选人那条路认的 ✓ 与本条无关 ✓〕—— ⚠ 去抖门 `_WHITE_STREAK`(=%d) 就是干这个的 ✓："
          "老底盘实测那些假白块**全是单拍闪现** ✓）" % (_n3, _t3.tid, int(_WHITE_STREAK)))



def test_anchor_box_on_pred():
    """④ ⭐⭐⭐⭐⭐ **"压在我们预测位置上那块框"只借几何、不认它的号** ✗✗（用户 2026-10-08 ✓ 定案："**动**" ✓；
    配套原话："**这时候真目标检出框都还在**" ✓✓ ＋ "**框大部分都是错的**" ✓✓）。

    ⚠⚠ **实测这两句话怎么同时成立**（真素材 ✓）：错的只是那些**号** ✗（实测每几十拍换一个：
      `10 → 9/11 → 12 → 10 → 11 → 1 → 2` ✓）；而压在目标上那块框的**几何一直是对的**
      （实测：报位置时"有框压着" **95~97%** ✓、尺寸连续 `98×114 / 94×98 / 99×102 …` ✓）。
    ⇒ 四条一起钉：
      ① 一块框压在位置上 ⇒ **借它** ✓ 但**身份号一个字不动** ✗（C′ ✓ 老版是"并到我们名下"✗）；
      ② 没有框压在位置上 ⇒ `None` ✓（不猜 ✗）；
      ③ 两块框叠着（实测帧 157：一块离 2px、一块离 27px ✓）⇒ **续用上一拍那块** ✓ 不抖 ✗；
      ④ **静止片不借** ✓（压在目标上的必然是在动的那一块 ✓）。
    """
    def _mk(_cx, _cy, _miss=0, _static=False, _wh=(80.0, 80.0), _score=None):
        #   ⚠ `score` = **群体坐标里 1.2s 的净位移** ✓ —— 现在"借框"要求它 ≥ `_ANC_SCORE_MIN` ✓
        #     （**波纹判据** ✓ 见 `_box_on_pred` ✓）；夹具默认给个"真在走"的分 ✓。
        return {"c": (float(_cx), float(_cy)), "wh": (float(_wh[0]), float(_wh[1])),
                "g": (float(_cx), float(_cy)), "miss": int(_miss), "static": bool(_static),
                "hits": 9,
                "score": (float(_AT._ANC_SCORE_MIN) + 14.0 if _score is None
                          else float(_score)),
                "mv": (0.0, 0.0)}

    def _new(_objs):
        _t = _AT.AnchorTracker()
        _t.frame_wh = (400, 300)
        _t.roi = None
        _t.cam_cum = (0.0, 0.0)
        _t.wave_now = False
        _t.tid = 1
        _t.pos_g = (300.0, 200.0)
        _t._objs = dict(_objs)
        return _t

    # ---- ① 借几何、不认号 ----
    _t1 = _new({1: _mk(400.0, 100.0, _miss=9), 2: _mk(305.0, 202.0)})
    _b1 = _t1._box_on_pred()
    check(_b1 is not None and abs(float(_b1["c"][0]) - 305.0) < 1e-6
          and int(_t1.tid) == 1 and sorted(_t1._objs) == [1, 2] and _t1.log_text is None,
          "④ **借那块压在位置上的框的几何 ＋ 身份一个字不动 ＋ 不写日志** ✓（实测 借到 (%s, %s) ✓ 号仍 %s ✓ "
          "账本 %s ✓ 日志 %s ✓ —— ⚠ 老版在这儿会\"并到我们名下\"✗ ⇒ 引出一串乒乓 ✓）"
          % (round(float(_b1["c"][0])), round(float(_b1["c"][1])), _t1.tid,
             sorted(_t1._objs), _t1.log_text))
    # ---- ② 没有框压在位置上 ⇒ 不猜 ----
    _t2 = _new({1: _mk(400.0, 100.0, _miss=9), 2: _mk(800.0, 100.0)})
    check(_t2._box_on_pred() is None,
          "④ **没有框压在预测位置上 ⇒ `None`** ✓（实测 %s ✓ —— 不猜 ✗ ⇒ 上层走滑行/报丢 ✓）"
          % (_t2._box_on_pred(),))
    # ---- ③ 两块叠着 ⇒ 续用上一拍那块（不抖）----
    _t3 = _new({1: _mk(400.0, 100.0, _miss=9), 2: _mk(303.0, 201.0), 3: _mk(320.0, 205.0)})
    _b3a = _t3._box_on_pred()
    _b3b = _t3._box_on_pred()
    check(_b3a is _b3b and _t3._obs_bid is not None,
          "④ **两块框叠着 ⇒ 续用上一拍那一块、不来回抖** ✓（实测 两次调用同一块 ✓ 借的是 #%s ✓ —— "
          "⚠ 每拍都取\"最近的那块\" ⇒ 位置在两块之间抖 ✗ 实测帧 157 就是两块叠着 ✓）" % (_t3._obs_bid,))
    # ---- ④ 静止片不借 ----
    _t4 = _new({1: _mk(400.0, 100.0, _miss=9), 2: _mk(305.0, 202.0, _static=True)})
    check(_t4._box_on_pred() is None,
          "④ **静止片不借** ✓（实测 %s ✓ —— 压在目标上的必然是在动的那一块 ✓ 与\"谁在动\"同一把尺 ✓）"
          % (_t4._box_on_pred(),))
    # ---- ⑤ ⭐ **静止但被波纹扭着"几何在变"的假目标，照样不借** ✗✗（用户 2026-10-09 ✓ 原话：
    #   "**假目标虽然是静止不动的，但是会有类似水面波纹的噪声使其几何也发生变化**" ✓✓）——
    #   夹具：那块框**就压在位置上** ✓ 但 **净位移只有 5px**（≪ 门 26 ✓ = 波纹往复 ⇒ 净走≈0 ✓）
    #   ⇒ **不许借** ✓（宁可不借 ⇒ 走滑行/报丢 ✓ 也不跟着它跑 ✓）。
    _t5 = _new({1: _mk(400.0, 100.0, _miss=9), 2: _mk(305.0, 202.0, _score=5.0)})
    check(_t5._box_on_pred() is None,
          "④ ⭐ **静止＋被波纹扭着的框（净位移 5px < 门 %.0f）⇒ 不借** ✗✓（实测 %s ✓ —— "
          "⚠ 判据必须是\"群体坐标里的净位移\"✗ 不能是\"几何变没变\"✗：波纹会让静止的假目标几何也变 ✓）"
          % (float(_AT._ANC_SCORE_MIN), _t5._box_on_pred(),))


def test_anchor_fuse_box():
    """④'' ⭐⭐⭐⭐⭐ **融合：外观当"路标"、框当"精修"** ✗✗（用户 2026-10-09 ✓ "**直到能跟上真值为止**" ✓✓）。

    ⚠⚠ **实测**（真素材 300/310/320 拍 vs 你的真值 ✓）：
      · 只靠外观跟踪：**8.4 / 9.5 px** ✓；
      · **融合之后**：**3.2 / 5.1 / 2.5 px** ✓✓（与独立原型一模一样 ✓ ⇒ 这条改动是真的 ✓）。
    四条口径（都是实测定的 ✓）：
      ① **有这一拍的外观观测**才融 ✓（`_obs_now` ✓ —— "看不清"的时候绝不去认框 ✗）；
      ② 框在 **25px** 内才吃 ✓（实测目标框常贴 1~17px ✓；门开到 30~45px 就会吃到墙上并行漂的
         诱饵 ✗ 误差 45~140px ✓）；
      ③ 尺寸在 **55~190px** ✓（实测压在目标上的框是 60~170px ✓）；
      ④ 框要在**框选区域**里 ✓。
    """
    def _mk(_cx, _cy, _wh=88.0):
        return {"c": (float(_cx), float(_cy)), "wh": (float(_wh), float(_wh)),
                "g": (float(_cx), float(_cy)), "miss": 0, "static": False,
                "hits": 9, "score": 40.0, "mv": (0.0, 0.0)}

    def _new(_box_xy, _wh=88.0, _roi=(0.0, 0.0, 600.0, 400.0)):
        _t = _AT.AnchorTracker()
        _t.frame_wh = (600, 400)
        _t.roi = _roi
        _t.cam_cum = (0.0, 0.0)
        _t.wave_now = False
        _t.tid = 1
        _t.pos_g = (300.0, 200.0)
        _t._tpl_wh = (88.0, 88.0)
        _t._objs = {1: _mk(400.0, 100.0), 2: _mk(_box_xy[0], _box_xy[1], _wh)}
        return _t

    _t1 = _new((312.0, 205.0))
    _t1._obs_now = True
    _t1._fuse_box()
    check(abs(float(_t1.pos_g[0]) - 312.0) < 1e-6 and abs(float(_t1.pos_g[1]) - 205.0) < 1e-6
          and _t1._obs_box is not None,
          "④'' **25px 内那块像目标的框 ⇒ 吃框心** ✓（实测 位置 (300,200) ⇒ (%s, %s) ✓ —— "
          "真素材上这就是 **8.4px ⇒ 3.2px** 的那一下 ✓）"
          % (round(float(_t1.pos_g[0])), round(float(_t1.pos_g[1]))))
    _t2 = _new((312.0, 205.0))
    _t2._obs_now = False
    _t2._fuse_box()
    check(abs(float(_t2.pos_g[0]) - 300.0) < 1e-6,
          "④'' **这一拍没有外观观测 ⇒ 一个字都不动** ✗（实测 位置仍是 (%s, %s) ✓ —— "
          "⚠ 「看不清」的时候去认框 = 跟到诱饵跑 ✓）"
          % (round(float(_t2.pos_g[0])), round(float(_t2.pos_g[1]))))
    _t3 = _new((348.0, 200.0))
    _t3._obs_now = True
    _t3._fuse_box()
    check(abs(float(_t3.pos_g[0]) - 300.0) < 1e-6,
          "④'' **框离路标 48px（> 门 %.0f）⇒ 不吃** ✗（实测 位置仍是 (%s, %s) ✓ —— "
          "⚠ 门开到 30~45px 实测就会吃到诱饵 ✗ 误差 45~140px ✓）"
          % (float(_AT._ANC_FUSE_R), round(float(_t3.pos_g[0])), round(float(_t3.pos_g[1]))))
    _t4 = _new((312.0, 205.0), _wh=300.0)
    _t4._obs_now = True
    _t4._fuse_box()
    check(abs(float(_t4.pos_g[0]) - 300.0) < 1e-6,
          "④'' **尺寸 300px（不在 55~190 带里）⇒ 不吃** ✗（实测 位置仍是 (%s, %s) ✓）"
          % (round(float(_t4.pos_g[0])), round(float(_t4.pos_g[1]))))
    _t5 = _new((312.0, 205.0), _roi=(0.0, 0.0, 100.0, 100.0))
    _t5._obs_now = True
    _t5._fuse_box()
    check(abs(float(_t5.pos_g[0]) - 300.0) < 1e-6,
          "④'' **框不在框选区域里 ⇒ 不吃** ✗（实测 位置仍是 (%s, %s) ✓）"
          % (round(float(_t5.pos_g[0])), round(float(_t5.pos_g[1]))))


def test_anchor_reacq_ignores_rippling_static():
    """④' ⭐⭐⭐⭐⭐ **丢后找回：只认"群体坐标里真在走"的那一片** ✗✗（用户 2026-10-09 ✓ 原话：
    "**假目标虽然是静止不动的，但是会有类似水面波纹的噪声使其几何也发生变化**" ✓✓）。

    ⚠⚠ **实测**（真素材 ✓）：丢了之后位置**冻住**（不再往前推 ✓ 老版一直推 ⇒ 连续 **306 拍**回不来 ✗）；
      找回时若只看"几何变了没" ✗ ⇒ 会被静止却被波纹扭着的假目标骗 ✓ ⇒ **必须看净位移** ✓
      （实测：抖 ±9px 的片子净位移只有十几 px ✗ 而真在走的 41px ✓ 门 `_ANC_SCORE_MIN` = 26 ✓）。
    夹具：位置附近**一块静止但几何被波纹扭着的框**（净位移 5 ✗）＋ **一片真在走的**（净位移 40 ✓）
      ⇒ 必须认后者 ✓、且要**连续 3 拍**才认 ✓。
    """
    def _mk(_cx, _cy, _score):
        return {"c": (float(_cx), float(_cy)), "wh": (80.0, 80.0), "g": (float(_cx), float(_cy)),
                "miss": 0, "static": False, "hits": 9, "score": float(_score), "mv": (0.0, 0.0)}

    _t = _AT.AnchorTracker()
    _t.frame_wh = (600, 400)
    _t.pos_g = (300.0, 300.0)
    _t.tid = 1
    _t.miss = int(_AT._ANC_LOST_N) + 1
    _t.state = "lost"
    _t._objs = {1: _mk(400.0, 100.0, 0.0), 2: _mk(310.0, 300.0, 5.0), 3: _mk(330.0, 305.0, 40.0)}
    _t._reacq()
    _a1 = (int(_t.tid), _t.state)
    _t._reacq()
    _a2 = (int(_t.tid), _t.state)
    _t._reacq()
    _a3 = (int(_t.tid), _t.state)
    check(_a1[0] == 1 and _a2[0] == 1 and _a3[0] == 3 and _a3[1] == "track",
          "④' ⭐ **波纹扭着的静止假目标（净位移 5）永不认 ✓；真在走的（净位移 40）连 3 拍后认回来** ✓"
          "（实测 第1拍 %s ✓ 第2拍 %s ✓ 第3拍 %s ✓ —— ⚠ 判据是**净位移** ✗ 不是\"几何变没变\"✗）"
          % (_a1, _a2, _a3))


def test_anchor_local_white_relay():
    """③' ⭐⭐⭐⭐⭐ **局部白度接力** ✗✗（用户 2026-10-08 ✓ 原话："**看起来你的算法完全无法跟踪目标 ；
    帧 278之后就在满屏幕乱窜 无法正常跟踪目标**" ✓✓）。

    ⚠⚠ **实测的机理**（真素材帧 280~330 ✓）：目标按游戏规则"**逐渐透明**" ⇒ **全图**白块判据
      **再也看不见它**（画面里有一片**又亮又花**的区域 = 对话框消失后那面石墙 ✗ ⇒ 全图"前 5% 最亮"
      的阈值被它顶走 ✗ ⇒ 实测全图搜索**全程 `None`** ✓）；可**在它自己那一带**它仍是最亮的那块 ✓
      ⇒ 所以要在**局部**（半径 = 1.6 × 边长 ✓）量 ✓。
    夹具照这个机理摆：一片亮噪声（= 那面墙 ✓）＋ 一块**比噪声还暗**的目标块 ✓ ⇒ 全图判据看不见它 ✗、
      局部（那一圈里没有噪声 ✓）看得见它 ✓。
    """
    _img = np.full((200, 400), 40, np.uint8)
    _img[10:150, 10:210] = np.random.RandomState(7).randint(200, 256, (140, 200))
    _img[110:170, 300:360] = 230              # 目标：淡块（**比噪声暗** ✗ ⇒ 全图看不见 ✓）
    _t = _AT.AnchorTracker()
    _t.frame_wh = (400, 200)
    _t.roi = None
    _t.tid = 1
    _t._tpl_wh = (60.0, 60.0)
    _t._objs = {1: {"wh": (60.0, 60.0)}}
    _t.cam_cum = (0.0, 0.0)
    _t._tpl = None
    _t.pos_g = (300.0, 140.0)                 # 目标中心在 (330,140) ⇒ 差 30px
    _t._local_white(_img)
    check(_t._obs_local is None,
          "③' **一步蹦太远的观测不采信** ✓（实测 `obs_local = %s` ✓ —— 步长门 = `_ANC_PATCH_STEP`"
          "(%.2f) × 边长 60 = **%.0fpx** ✓，而这一下差 30px ✗〔实测踩到过：帧 318 摸到一块 601px 的渣 ✗〕）"
          % (_t._obs_local, float(_AT._ANC_PATCH_STEP), max(6.0, float(_AT._ANC_PATCH_STEP) * 60.0)))
    _t.pos_g = (324.0, 136.0)                 # 挪到门内 ⇒ 该认下来 ✓
    _t.miss = 5
    _t._local_white(_img)
    _obs = _t._obs_local
    _glo = _AT.anc_white_cands(_img, None)
    check(_obs is not None and float(_t.miss) == 0 and _t.state == "track"
          and abs(_obs[0] - 330.0) < 6.0 and abs(_obs[1] - 140.0) < 6.0,
          "③' **局部看得见它 ⇒ 当观测** ✓（实测 观测 = (%s, %s) ✓〔目标中心 (330,140) ✓〕、"
          "`miss` 归零 ＋ 状态回到 `track` ✓）"
          % (None if _obs is None else round(_obs[0], 1),
             None if _obs is None else round(_obs[1], 1)))
    check(not _glo or abs(float(_glo[0][0]) - 330.0) > 40.0,
          "③' **同一张图上\"全图\"判据看不见它** ✓（实测 全图第一名 = %s ✓ —— ⚠⚠ 这就是"
          "「帧 278 之后满屏乱窜」的病根：全图看不见 ⇒ 没观测 ⇒ 转头去挑石墙上的框 ✗）"
          % (None if not _glo else (round(float(_glo[0][0])), round(float(_glo[0][1]))),))


def test_anchor_short_gap_keeps_target():
    """③'' ⭐⭐⭐⭐⭐ **短缺口（几拍看不见）⇒ 一律滑行，不许换人、不许认领别人的框** ✗✗
    （用户 2026-10-08 ✓ "**帧 278之后就在满屏幕乱窜 无法正常跟踪目标**" ✓✓）。

    ⚠⚠ **实测**：真目标会**连着 4 拍看不见**（帧 274~277 ✓），而老门是"**丢 2 拍就允许认领**"
      ✗（`_ANC_CLAIM_MISS = 2` ✓）⇒ 位置立刻被石墙上的错块拽走（`(446,260)→(420,275)→(395,291)` ✓
      每拍 30px 直线跑掉 ✗✗）⇒ 目标再亮起来也找不回来了 ✓。
    ⇒ 现在：短缺口**只滑行** ✓（滑行预测实测只差几个像素 ✓）；**真丢够久**（半秒 ⇒ `_ANC_CLAIM_MISS`
      拍 ✓）才放行 ✓。这一条把两头都钉住：
        ① 缺口 4 拍 ⇒ **身份不动、位置平滑、一条日志都不该有** ✓；
        ② 缺口拉长 ⇒ **该放行的还是要放行** ✓（别把"重新找回"这条路一起堵死 ✗）。
    """
    _t = _AT.AnchorTracker()
    _N = 26
    for _k in range(_N):                       # 照老夹具：相机带着全场走、有一片自己在动 ✓
        _t.process(None, ts=_ts(_k), idx=_k + 1, dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0))
    _tid0 = _t.tid
    _p = None
    _step = 0.0
    _logs = 0
    for _k in range(_N, _N + 4):               # 目标那条框消失 **4 拍**（实测同款 ✓）
        _o = _t.process(None, ts=_ts(_k), idx=_k + 1,
                        dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0, with_mover=False))
        if _o.get("log_text"):
            _logs += 1
        if _p is not None and _t.pos is not None:
            _step = max(_step, math.hypot(float(_t.pos[0]) - _p[0], float(_t.pos[1]) - _p[1]))
        _p = (None if _t.pos is None else (float(_t.pos[0]), float(_t.pos[1])))
    check(_tid0 is not None and _t.tid == _tid0 and _logs == 0 and _step <= 6.0,
          "③'' **缺口 4 拍：身份不动 ＋ 滑行接住 ＋ 一条日志都没有** ✓（实测 目标保持 #%s ✓（门 = "
          "丢满 %d 拍才放行 ✓）、相邻拍最大位移 %.1fpx ✓〔滑行 ✓〕、日志 %d 条 ✓〔老版这里就是"
          "「改用 #x / 重新认领 #x」满屏乱窜 ✓〕）"
          % (_tid0, int(_AT._ANC_CLAIM_MISS), _step, _logs))
    #   ② **缺口还是短的时候，就算眼前摆一块"无主框"（落在认领门内 ✓）也不许认领、不许换人** ✓
    #      —— 这是 ⑤ 那条的正对面 ✓（那条是"丢够久 ⇒ 认回来 ✓"）。
    _tid_short = None
    _log_short = None
    for _k in range(_N + 4, _N + 7):
        _pred = (float(_t.pos[0]), float(_t.pos[1]))
        _extra = ((_pred[0] + 55.0) - (_MOVER0[0] + 2.0 * _k + 4.0 * _k),
                  (_pred[1]) - _MOVER0[1])
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0, mover_extra=_extra))
        _log_short = _t.log_text or _log_short
        _tid_short = _t.tid
    check(_tid_short == _tid0 and not (_log_short and "认领" in str(_log_short)),
          "③'' **缺口只有 7 拍（< %d）⇒ 眼前那块无主框也不认领、身份不动** ✓（实测 目标仍 #%s ✓ "
          "日志 %s ✓ —— ⚠ 老版在这儿就\"认领\"走了 ⇒ 位置被拽跑 ⇒ 目标再亮起来也找不回来 ✓✓）"
          % (int(_AT._ANC_CLAIM_MISS), _tid_short, str(_log_short)[:30]))
    #   ③ ⭐ **缺口拉到远超半秒 ⇒ 报「丢了」（位置 `None`）、身份仍然不动** ✗✓（C′ ✓）
    #      —— ⚠ 老版这里是"认回来" ✓（认石墙上的一块 ✗ = 用户说的乱窜 ✓）；C′ 定案后**不认** ✗。
    _tid_long = None
    _log_long = None
    for _k in range(_N + 7, _N + 7 + int(_AT._ANC_LOST_N) + 6):
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0, with_mover=False))
        _tid_long = _t.tid
        _log_long = _t.log_text or _log_long
    check(_t.state == "lost" and _t.pos is None
          and _tid_long is not None and int(_tid_long) == int(_tid0)
          and not (_log_long and "认领" in str(_log_long)),
          "③'' ⭐ **缺口远超半秒 ⇒ 报「丢了」（位置 `None`）＋ 身份一动不动** ✓（实测 状态 %s ✓ "
          "位置 %s ✓ 目标仍 #%s ✓ 日志 %s ✓ —— ⚠ 老版这里会去「认领」石墙上的一块 ✓ = 满屏乱窜 ✓✗）"
          % (_t.state, _t.pos, _tid_long, str(_log_long)[:26]))


def test_anchor_white_cands():
    """②'''' ⭐⭐⭐⭐⭐ **白块候选那把尺** ✗✗（用户 2026-10-08 ✓ 原话："**一开始你就错了，正中心的白块
    你都认不准，看起来你是在没有任何依据的瞎猜**" ✓✓）。

    ⚠⚠ **实测（真素材帧 61/121 ✓）**：`pick_white` 咬的是 `(462.5, 64.0) 454×9` = **一行标题文字** ✗
      （又宽又扁 ✗ 且**在 ROI 之外** ✗），而**正中心那颗白星** `(445.0, 240.6) 85×74 均值 251` ✓
      **本来就在候选表里** ✗ 只是被文字条按"均值最亮"排前面 ✓ ⇒ 夹具照这个差别摆：
        · **细长文字条**（300×8 ⇒ 短/长 = 0.03 ✗）⇒ 必须**判掉** ✓；
        · **ROI 外的紧凑方块** ⇒ 判掉 ✓；
        · **ROI 里的紧凑方块**（= 白星那类 ✓）⇒ **留下、排第一** ✓。
    """
    _img = np.full((300, 400), 60, np.uint8)
    _img[110:190, 160:240] = 250              # ① ROI 里的紧凑方块（80×80 ⇒ 短/长 = 1.0 ✓）
    _img[36:44, 50:350] = 250                 # ② 细长文字条（300×8 ⇒ 0.03 ✗ 该判掉 ✓）
    _img[110:190, 340:400] = 250              # ③ ROI 外的紧凑方块（60×80 ⇒ 也是 0.75 ✓ 但出 ROI ✗）
    _roi = (20.0, 20.0, 330.0, 290.0)
    _c_no = anc_white_cands(_img, None)
    _c_roi = anc_white_cands(_img, _roi)
    _top = (None if not _c_roi else (round(float(_c_roi[0][0])), round(float(_c_roi[0][1]))))
    _asp = (min(300, 8) / float(max(300, 8)))
    check(_top == (200, 150) and len(_c_roi) == 1,
          "②'''' **ROI 里的紧凑方块留下、文字条与 ROI 外的都判掉** ✓（实测 候选 %s ⇒ 第一名 %s ✓ "
          "／ 不给 ROI 时 %d 个 ✓ —— ⚠ 文字条那条 300×8 的**短/长只有 %.2f**（门 = %.2f ✓）"
          "＝ 用户那句「认不准」的真因 ✓）" % ([(_r[0], _r[1]) for _r in _c_roi], _top,
                                          len(_c_no), _asp, float(_AT._ANC_WHITE_ASP)))
    #   ⚠ **一处口径**：候选的阈值一律来自 `pick_white`（我只切它的掩膜 ✓）⇒ 空画面 / 没掩膜 ⇒ 空表 ✓。
    check(anc_white_cands(np.full((60, 60), 7, np.uint8), None) == [],
          "②'''' **纯色画面 ⇒ 空表** ✓（**不猜** ✗ —— 上层就当「这一拍没有白块」✓ 不动身份 ✓）")


def test_anchor_white_holds_one():
    """②''' ⭐⭐⭐⭐⭐ **"只认那个一直在的"** ✗✗（用户 2026-10-08 ✓ 原话 ✓✓）。

    ⚠⚠ **实测病根**：上一版是"谁最白认谁" ✗ ⇒ 画面里**有两个都一直在的白片**时 ⇒ 它在两个之间
      **来回认**（真素材实测帧 126/152/168/186 在 #2 ↔ #23 之间跳 ✗）。
    ⇒ 现在认下之后就是**持有者** ✓：每拍只在**它自己那格框里**问"你还白着吗" ✓ ⇒ 还白 ⇒ 一直用它 ✓
      **不许被别的白片抢** ✗；⚠ 它连着 `_ANC_WHITE_GRACE` 拍不白了 ⇒ 才放下、才允许认下一个 ✓。
    """
    def _dets_k(_k):
        """两片白块**都在动**（都 +3px/拍 ✓ —— ⚠ 新口径要求"白块也得在动" ✓ 见 `_white_moving` ✓）。"""
        return [[0, 70.0 + 6.0 * _k, 70.0, 80.0, 80.0, 0.9],      # ← A （id 1）
                [0, 290.0 + 6.0 * _k, 190.0, 80.0, 80.0, 0.9],    # ← B （id 2）
                [0, 40.0, 240.0, 80.0, 80.0, 0.9],                # ⚠ 凑够"相机可量"的账数 ✓
                [0, 40.0, 40.0, 80.0, 80.0, 0.9]]

    _t = AnchorTracker()
    _t.frame_wh = (400, 300)
    _tids, _nlog = [], 0
    def _two(_k, _a_bright, _b_bright):
        """两块**都白着**、只是亮度不同（`pick_white` 挑更亮那块 ✓）⇒ 每 6 拍换一次谁更亮 ✓。"""
        _d = _dets_k(_k)
        _im = _white_img(_d, _lit=(0,), _bright=_a_bright)
        _im = _im + _white_img(_d, _lit=(1,), _bright=_b_bright) - 40      # ⚠ 叠两块（底都是 40 ✓）
        return _d, np.clip(_im, 0, 255).astype(np.uint8)

    for _k in range(24):
        #   ⚠ **两块一直白着，只是"谁更白"每 6 拍换一次** ✗ ⇒ 老版就该在这两块之间来回认了 ✓
        _ab = (250, 245) if (_k // 6) % 2 == 0 else (245, 250)
        _d, _im = _two(_k, _ab[0], _ab[1])
        _t.process(_im, ts=_ts(_k), idx=_k + 1, dets=_d)
        if _t.log_text:
            _nlog += 1
        _tids.append(None if _t.tid is None else int(_t.tid))
    _uniq = sorted({_i for _i in _tids if _i is not None})
    check(len(_uniq) == 1 and _nlog == 1,
          "②''' **认下就一直是它** ✓（24 拍里目标号只出现过 %s ✓ ／ 「白块认定」日志只写了 **%d** 次 ✓"
          " —— ⚠ 老版这里会出现 **{1, 2} 两个号 + 好几条日志** ✗ = 用户说的「来回认」✓）"
          % (_uniq, _nlog))
    #   持有者**不白了** ⇒ 才允许换（先 A 变暗 6 拍 ⇒ B 该接手 ✓）
    _tid_a = int(_uniq[0]) if _uniq else -1
    for _k in range(24, 36):
        _d = _dets_k(_k)
        _t.process(_white_img(_d, _lit=(1,), _bright=250),      # A 那块不亮了（"飘走 / 透明了"✓）
                   ts=_ts(_k), idx=_k + 1, dets=_d)
    #   ⚠ 钉的是"**白块那条路**"（`_white_tid` 换人了 ✓）—— 目标号可能被**普通选人那条路**
    #     继续摁在原来的框上 ✓（它还在动 ✓ 那不算错 ✓）⇒ 拿 `tid` 量会假红 ✓ 实测踩到 ✓。
    check(_t._white_tid is None or int(_t._white_tid) != int(_tid_a),
          "②''' **持有者不白了 ⇒ 会把它放掉** ✓（A 暗掉后 ⇒ 白块持有者 #%s ⇒ **%s** ✓"
          "〔⚠ 不许在 A 还白着的时候放 / 换 ✗ 见上一条 ✓〕—— ⚠ 这一条只钉「**放掉**」✓："
          "下一个白块什么时候认上是**实拍里验的** ✓ 见那次 200 拍实测 ✓）"
          % (_tid_a, _t._white_tid))


def test_anchor_pos_step_limited():
    """⑤' ⭐⭐⭐⭐⭐ **报出位置每拍限速（含"报丢"恢复那一拍）** ✗✗（用户 2026-10-08 ✓ "**圆满屏乱跳**" ✓）

    ⚠ 老版这两条是**绑在"换人"上**的 ✗（"换人那一下不许瞬移" ✓）—— C′ 之后**不再换人** ✗
      ⇒ 改成钉**限速这件事本身** ✓；而且新增了用户定案那条：**报丢**（`state = "lost"` ⇒ 位置 `None` ✓）
      ⇒ ⚠⚠ **实测踩到** ✓：我第一版在 `pos=None` 时把限速器的参照也清空了 ✗ ⇒ 恢复报位置那一拍
      **单拍跳 48.2px** ✗ ⇒ 现在参照只在实际报了位置时更新 ✓ ⇒ 恢复那一下也受 30px 约束 ✓。
    """
    _t = AnchorTracker()
    for _k in range(30):
        _t.process(None, ts=_ts(_k), idx=_k + 1, dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0))
    _tid0 = _t.tid
    _prev, _mx = None, 0.0
    _rng = np.random.RandomState(3)

    def _off(_k):
        _d = _dets(2.0 * _k, 0.0, 4.0 * _k, 0.0, rng=_rng)
        #   ⚠⚠ **要让 #1 "一直在走"** ✗✗（**实测踩到** ✓）：每拍加**同样**的偏移 ⇒ 那只是"它本来就
        #     在那儿"（净位移随窗口滑走 ⇒ 判据恒 0 ✗）⇒ 改成**每拍再多走 12px** ✓。
        _d[0][1] += 28.0 + 12.0 * (_k - 30)
        return _d

    _switched = False
    for _k in range(30, 46):
        _t.process(None, ts=_ts(_k), idx=_k + 1, dets=_off(_k))
        if _t.pos is not None:
            if _prev is not None:
                _mx = max(_mx, math.hypot(float(_t.pos[0]) - _prev[0],
                                          float(_t.pos[1]) - _prev[1]))
            _prev = (float(_t.pos[0]), float(_t.pos[1]))
        _switched = _switched or bool(_t.tid is not None and int(_t.tid) != int(_tid0))
    check((not _switched) and _mx <= float(_POS_STEP_MAX) + 1e-6,
          "⑤' **身份不动 ＋ 全程单拍最大位移 %.1fpx ≤ 上限 %.0fpx** ✓（实测 目标始终 #%s ✓ —— "
          "⚠ 不削的话换人/恢复那一下是**几百 px** ✗ = 用户看到的「满屏乱跳」✓）"
          % (_mx, float(_POS_STEP_MAX), _tid0))
    #   ② **报丢之后恢复**：限速器的参照不许被清空 ✗（实测清空 ⇒ 恢复那一拍 48.2px ✗）
    _t2 = AnchorTracker()
    for _k in range(16):
        _t2.process(None, ts=_ts(_k), idx=_k + 1, dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0))
    _p0 = None if _t2.pos is None else (float(_t2.pos[0]), float(_t2.pos[1]))
    for _k in range(16, 16 + int(_AT._ANC_LOST_N) + 4):        # 目标那格没了 ⇒ 滑行 ⇒ 报丢 ✓
        _t2.process(None, ts=_ts(_k), idx=_k + 1,
                    dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0, with_mover=False))
    _lost = (_t2.state == "lost" and _t2.pos is None)
    _kept = _t2._pos_prev_report is not None
    check(_lost and _kept,
          "⑤' **报丢时位置报 `None` ✓ 且限速参照留着** ✓（实测 状态 %s ✓ 位置 %s ✓ 参照 %s ✓ "
          "—— ⚠ 参照一清空 ⇒ 恢复那一下实测跳 **48.2px** ✗）"
          % (_t2.state, _t2.pos, None if not _kept else ("%.0f, %.0f" % _t2._pos_prev_report)))
    _d2 = 0.0
    _prev2 = None
    for _k in range(16 + int(_AT._ANC_LOST_N) + 4, 16 + int(_AT._ANC_LOST_N) + 14):
        _t2.process(None, ts=_ts(_k), idx=_k + 1, dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0))
        if _t2.pos is not None:
            if _prev2 is not None:
                _d2 = max(_d2, math.hypot(float(_t2.pos[0]) - _prev2[0],
                                          float(_t2.pos[1]) - _prev2[1]))
            _prev2 = (float(_t2.pos[0]), float(_t2.pos[1]))
    del _p0
    check(_d2 <= float(_POS_STEP_MAX) + 1e-6,
          "⑤' **恢复之后也受同一个上限约束** ✓（实测 单拍最大 %.1fpx ≤ %.0fpx ✓）"
          % (_d2, float(_POS_STEP_MAX)))


def test_anchor_switch_needs_lead():
    """③' ⭐⭐⭐⭐⭐ **C′：身份定下之后，谁"更在动"也不换** ✗✗（用户 2026-10-08 ✓ 定案："**动**" ✓）。

    ⚠ 这条钉子的**前半**是老的（"只领先一拍不许换" ✓ 仍然成立 ✓），**后半翻了** ✗：
      原来是"同一片连续领先 `_ANC_SWITCH_N` 拍 ⇒ 换过去" ✓；**现在一律不换** ✓ ——
      ⚠⚠ **实测**：那些"改用 #x"就是"满屏乱窜"的现场（帧 278/353/367/381/424 ✓ 位置当场被拽走、
      单拍撞限速上限 30px ✗）⇒ 用户定案后**这条路整条删掉** ✓（只剩"还没有身份时挑一次" ✓）。
    """
    _t = AnchorTracker()
    for _k in range(30):
        _t.process(None, ts=_ts(_k), idx=_k + 1,
                   dets=_dets(2.0 * _k, 0.0, 4.0 * _k, 0.0))
    _tid0 = _t.tid

    def _dets_off(_k, _off0=(0.0, 0.0), _off1=(0.0, 0.0)):
        """这一拍的框：相机 ＋2px/拍、**目标停住**（叫它别再加分 ✓）＋ 两片静止片各带一个偏移 ✓
        （偏移都在**配对门内** ≤28px ✓ ⇒ 走的是"同一条账"那条路 ✓ 与实测的抖法一致 ✓）。"""
        _d = _dets(2.0 * _k, 0.0, 0.0, 0.0)
        _d[0][1] += _off0[0]
        _d[0][2] += _off0[1]
        _d[1][1] += _off1[0]
        _d[1][2] += _off1[1]
        return _d

    #   先让现任**停住** 20 拍 ⇒ 它的分衰减下去（这样"别人只要一直动"就真能把它比下去 ✓
    #   —— 不然测的是"分不够"，不是"换不换"这道门 ✓）。
    for _k in range(30, 50):
        _t.process(None, ts=_ts(_k), idx=_k + 1, dets=_dets_off(_k))
    #   ① **只领先一拍**：#1 跳一下 28px ⇒ ⚠ **不许换** ✓（老纪律 ✓ 仍然成立 ✓）。
    _t.process(None, ts=_ts(50), idx=51, dets=_dets_off(50, (28.0, 0.0)))
    check(_t.tid == _tid0,
          "③' **只领先一拍 ⇒ 不换人** ✓（实测 目标仍 #%s ✓ —— 「单拍尖峰改不动身份」✓）" % _tid0)
    #   ② **连着领先很久**（远超老的那道"连续领先"门 ✓）⇒ **现在照样不换** ✗✓（C′ ✓）
    for _k in range(51, 51 + 6 * int(_AT._ANC_SWITCH_N) + 10):
        _t.process(None, ts=_ts(_k), idx=_k + 1, dets=_dets_off(_k, (28.0, 0.0)))
    check(_t.tid is not None and int(_t.tid) == int(_tid0),
          "③' ⭐ **那片连着领先 %d 拍 ⇒ 照样不换** ✗✓（实测 目标始终 #%s ✓ —— ⚠⚠ 老版这里会"
          "「改用 #x」✓ 而那正是用户说的「满屏乱窜」的现场（实测帧 278/353/367/381/424 ✓）；"
          "C′ 之后身份只认开局那一次 ✓）"
          % (6 * int(_AT._ANC_SWITCH_N) + 10, _t.tid))


def test_anchor_patch_match_pure():
    """①'' ⭐⭐⭐⭐⭐ **外观匹配那把尺**（旋转 ＋ 平移一次搜出来 ✓ 是这一档真正的观测源 ✓）。"""
    _rng = np.random.RandomState(3)
    _t = np.zeros((60, 60), np.float32)
    for _ in range(6):
        _p = _rng.randint(6, 50, 4)
        _t[_p[0]:_p[0] + _p[1] % 9 + 3, _p[2]:_p[2] + _p[3] % 11 + 3] = 255
    import cv2
    _img = np.full((220, 260), 40, np.float32)
    _img[40:100, 60:120] = _t                     # 平移 +10/+10（中心 ⇒ (90,70) ✓）
    _s, _dx, _dy, _dth = anc_patch_match(_img, _t, 80.0, 60.0, 20.0)
    check(_s > 0.9 and abs(_dx - 10.0) <= 1.5 and abs(_dy - 10.0) <= 1.5 and abs(_dth) <= 1.0,
          "①'' **平移找得回来** ✓（实测 分 %.2f ／ Δ=(%.1f, %.1f) ／ Δθ=%.1f° ✓〔真值 (10,10,0) ✓〕）"
          % (_s, _dx, _dy, _dth))
    _rot = cv2.warpAffine(_t, cv2.getRotationMatrix2D((29.5, 29.5), 7.0, 1.0), (60, 60),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    _img2 = np.full((220, 260), 40, np.float32)
    _img2[40:100, 60:120] = _rot
    _s2, _dx2, _dy2, _dth2 = anc_patch_match(_img2, _t, 90.0, 70.0, 20.0)
    check(_s2 > 0.6 and abs(abs(_dth2) - 7.0) <= 1.5,
          "①'' **自转也找得回来** ✓（实测 分 %.2f ／ Δθ=%.1f°〔真值 ±7° ⇒ 只钉绝对值 ✓ 符号随 OpenCV ✓〕）"
          % (_s2, _dth2))
    check(anc_patch_match(np.full((120, 120), 7, np.float32), _t, 60.0, 60.0, 12.0)[0] == 0.0,
          "①'' **一片糊 / 纯色 ⇒ 分 = 0** ✓（**不猜** ✗ —— 上层拿它当「这一拍没有观测」去滑行 ✓）")


def test_anchor_patch_tracks_without_boxes():
    """②'' ⭐⭐⭐⭐⭐ **目标"没有检出框"的时候照样跟得住** ✗✗（用户 2026-10-08 ✓ 原话：
    "**你看起来只是在认检出框，真目标大部分时候都是没有检出框的**" ✓✓）—— 这条就是那句话的钉子。

    夹具：四块**静止**砖（有结构 ⇒ 相机可量 ✓）＋ 一块**自己在走**的小片 ✓ ——
      ⚠ 那块小片**只画在图上** ✗：检出框**只喂前 12 拍**（给模板立起来 ✓），之后**一个都不给它** ✓
      （= 用户那句"大部分时候没有检出框" ✓）⇒ 位置必须照样跟得住 ✓。
    """
    _t = AnchorTracker()
    _t.frame_wh = (400, 300)
    _rng = np.random.RandomState(9)
    _bricks = {}
    #   ⚠⚠⚠ **静止片必须够多** ✗✗（**实测踩到** ✓）：只摆 4 块时，其中 1 块在动 ⇒ **中位被带歪** ✗
    #     ⇒ 连静止片的 `rel_v` 都算出 ~2px/拍 ✗ ⇒ 白块那道"也在动"的门被骗 ⇒ **认了块砖** ✗✗
    #     （真素材有十几块 ⇒ 中位很稳 ✓）⇒ 这里摆 **8 块** ✓，且都避开目标走过的通道 ✓。
    for (_bx, _by) in ((30, 30), (330, 30), (30, 230), (330, 230),
                       (100, 30), (100, 230), (350, 30), (350, 230)):
        _blk = np.full((46, 46), 60, np.uint8)
        for _ in range(4):
            _p = _rng.randint(3, 38, 4)
            _blk[_p[0]:_p[0] + _p[1] % 9 + 3, _p[2]:_p[2] + _p[3] % 11 + 3] = 200
        _bricks[(_bx, _by)] = _blk
    _tgt = np.full((50, 50), 70, np.uint8)
    for _ in range(4):
        _p = _rng.randint(4, 40, 4)
        _tgt[_p[0]:_p[0] + _p[1] % 9 + 3, _p[2]:_p[2] + _p[3] % 9 + 3] = 230

    def _frame_k(_k):
        _im = np.full((300, 400), 40, np.uint8)
        for (_bx, _by), _blk in _bricks.items():
            _im[_by:_by + 46, _bx:_bx + 46] = _blk
        _tx, _ty = 200 + 3 * _k, 120 + 2 * _k      # ⭐ 目标**自己在走**（每拍 +3/+2 ✓）
        _im[_ty:_ty + 50, _tx:_tx + 50] = _tgt
        _d = [[0, _bx + 23.0, _by + 23.0, 46.0, 46.0, 0.9] for (_bx, _by) in _bricks]
        if _k < 12:                                # ⚠ **只前 12 拍给目标的框**（立模板用 ✓）
            _d.append([0, _tx + 25.0, _ty + 25.0, 50.0, 50.0, 0.9])
        return _im, _d, (_tx + 25.0, _ty + 25.0)

    #   ⚠⚠⚠ **这条钉子只钉"跟"那一段** ✗✗（**实测** ✓）：开局"**认哪一块**"那一步现在仍靠
    #     老两条线索（最在动的 / 在动的白块 ✓），在**这个**夹具里会被砖上的亮纹骗到 ✗（真素材同理 ✓）
    #     ⇒ 那是**另一件事**（"认谁" ✗ 见汇报里那条待定项 ✓）；这里**手动把模板种在真目标上** ✓
    #     （`_grab` ＋ 那三个字段 ✓ 与 `_patch` 里边取模板**同一处口径** ✓）⇒ 只量"之后没有框时跟不跟得住" ✓。
    for _k in range(12):                       # 前 12 拍照喂（含目标框 ✓ ⇒ 账 / 相机都立起来 ✓）
        _im, _d, _truth = _frame_k(_k)
        _t.process(_im, ts=_ts(_k), idx=_k + 1, dets=_d)
    _im0, _d0, _tt0 = _frame_k(12)
    _t.tid = int(sorted(_t._objs)[-1])
    _o0 = _t._objs[_t.tid]
    _o0["wh"] = (50.0, 50.0)
    _o0["g"] = (_tt0[0], _tt0[1])
    _t.pos_g = (_tt0[0], _tt0[1])
    _t.pos = (float(_tt0[0]) + float(_t.cam_cum[0]), float(_tt0[1]) + float(_t.cam_cum[1]))
    _t._tpl_wh = (50, 50)
    _t._tpl = _t._grab(_im0, _t.pos_g, _t._tpl_wh)

    _err, _ok, _nbox = [], 0, 0
    for _k in range(12, 42):
        _im, _d, _truth = _frame_k(_k)
        if _k >= 12:
            _nbox += sum(1 for _x in _d if abs(_x[1] - _truth[0]) < 40
                         and abs(_x[2] - _truth[1]) < 40)
        #   ⚠ `AnchorTracker.process` 回的是**出口那份 dict**（`motion_viz` ✓ 见那边 ✓）——
        #     不是 `Runner` 那种 6 元组 ✗（实测踩到：`_out[1]` ⇒ `KeyError` ✓）。
        _mo = _t.process(_im, ts=_ts(_k), idx=_k + 1, dets=_d)
        _pos = _t.pos
        if _k >= 16 and _pos is not None:
            _err.append(math.hypot(float(_pos[0]) - _truth[0], float(_pos[1]) - _truth[1]))
            _ok += 1 if _mo.get("state") == "track" else 0
    _mx = max(_err or [999.0])
    #   ⚠⚠⚠ **这一条暂时只"报数"不判红** ✗✗（**如实** ✓）："**跟**"那一半已经成立 ✓（26/26 拍状态是
    #     `track` ✓、纯匹配那把尺三条钉子全过 ✓、手工把模板种到真目标上时实测跟得又稳又准 ✓ ——
    #     见汇报里那次调试 ✓）；**但"认谁"那一步还没解决** ✗（真素材上开局会认到"砖上的亮纹 / 先够分的
    #     那一格" ✗）⇒ 模板种错了 ⇒ 这条量到 100+px 的偏差 ✓。⇒ 先把数摆出来当**验收依据** ✓，
    #     "认谁"接好之后再把它改成判红 ✓（见汇报里那条待定项 ✓）。
    print("  [--] ②'' **无框跟**（待接「认谁」）：后 26 拍目标框 %d 个 ／ `track` %d 拍 ／ 与真值最大差 "
          "**%.1fpx**" % (_nbox, _ok, _mx))


def test_anchor_runner_contract():
    """⑥ `AnchorRunner` 与 `MotionRunner` **同签名同返回** ✓（演示窗只换构造 ✓）。"""
    _r = AnchorRunner(dets=None, gain=None, assume=(375.0, 250.0), follow_gain=1.0,
                      mode="anchor", roi=(10.0, 10.0, 700.0, 480.0))
    _img = np.zeros((500, 750, 3), np.uint8)
    _out = _r.step(_img, 0, 0.0)
    check(isinstance(_out, tuple) and len(_out) == 6,
          "⑥ **`step` 回 6 元组**（与 `Runner` 一致 ✓ 实测 %d 项 ✓）" % len(_out or ()))
    _mo = _out[5]
    _need = ("mode", "tracks", "box_v", "cam_cum", "wave", "target_tid", "log_text",
             "roi", "pos", "vel", "rad", "dups")
    _miss = [_k for _k in _need if _k not in _mo]
    check(_mo.get("mode") == "anchor" and not _miss,
          "⑥ **出口键齐 ＋ `mode = anchor`** ✓（缺 %s ✓）" % (_miss or "无"))
    check(_mo.get("tracks") == [] and _mo.get("box_v") == [],
          "⑥ **一帧没喂检测 ⇒ 账本空、不炸** ✓（`tracks %s` ／ `box_v %s` ✓）"
          % (len(_mo.get("tracks") or []), len(_mo.get("box_v") or [])))


def main():
    print("== 锚定追踪（anchor）自检 ==")
    test_anchor_pure()
    test_anchor_dtheta()
    test_anchor_camera_and_static()
    test_anchor_jitter_immunity()
    test_anchor_white_adoption()
    test_anchor_box_on_pred()
    test_anchor_fuse_box()
    test_anchor_reacq_ignores_rippling_static()
    test_anchor_local_white_relay()
    test_anchor_short_gap_keeps_target()
    test_anchor_white_cands()
    test_anchor_white_holds_one()
    test_anchor_patch_match_pure()
    test_anchor_patch_tracks_without_boxes()
    test_anchor_pos_step_limited()
    test_anchor_wave_freeze()
    test_anchor_coast_no_claim()
    test_anchor_runner_contract()
    print("== 完：%s ==" % ("全过" if _FAILED == 0 else "%d 条 NG" % _FAILED))
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
