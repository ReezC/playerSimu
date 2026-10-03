"""**卡尔曼滤波（匀速模型）** —— 影子模式用（用户 2026-10-03 ✓ 原话："**做 KF影子模式** 然后找个
地方加按钮或开关切换 …… 目的是可以**自由选择算法分支以防污染当前的进展**" ✓）。

模型的来历（用户 2026-10-03 ✓ 原话："**红框落位是测量值，轨迹预测是估算值**" ✓）：

| 卡尔曼的部件 | 本项目的对应物 |
|---|---|
| 状态 `x = [px, py, vx, vy]` | 圆心位置 + 速度（**加工域** ✓ 与红框/绿圈同一把尺 ✓）|
| **预测步**（估算值 ✓）| 轨迹预测（白箭头 / 曲线外推 ✓）|
| **更新步**（测量值 ✓）| 红框落位点 / 观测位置 ✓ |
| 过程噪声 `Q` | 目标**机动能力**（q = 加速度噪声谱密度 ✓）|
| 观测噪声 `R` | 观测源的**可信度**（落位点是几何推断 ⇒ R 大 ✓ 见 `_KF_R_DEF_*` ✓）|

⚠⚠ **本模块只被"影子模式"调用**（`LieTracker.show_kf` 勾上「**显示 KF 预测**」时 ✓）：它**只记录**
KF 的位置/速度与**新息**（= 观测 − 先验 ✓ KF 里最值钱的诊断量 ✓），**一个字节都不改**经典链条算出来
的位置 ✓✓（用户原话："以防污染当前的进展" ✓）。等对照数据够了，再谈"接管"（那是后面的事 ✗）。

为什么**不引第三方**（不用 `filterpy` 之类 ✓）：一是没必要（4 维匀速模型手写 20 行 ✓），二是
**口径要看得见**（这一项目所有判据都写在源码里 ✓ 便于逐行核对 ✓ 与 `geom.py` 同一规矩 ✓）。

数值用 `numpy`（项目本来就依赖它 ✓）；矩阵都是 4×4 / 2×2，规模小 ⇒ 直接 `linalg.inv` ✓ 够快 ✓
（影子模式下每拍只算一次 ✓）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["KF4"]

#: 过程噪声谱密度默认值（px²/s³ ✓）：把"目标机动"当成**白噪声加速度** ✓
#: `q = a²`（a = 加速度噪声的标准差 px/s² ✓）⇒ 默认 `a = 20 px/s²`（十几 px/帧² 量级 ✓
#: 现场素材目标每帧走 10~40px、偶尔转向 ⇒ 这个量级够"跟得上转弯"、又不至于滤成一根直线 ✓）。
KF_Q_DEF = 400.0
#: 观测噪声默认值（px² ✓）：`r = σ²`（σ = 观测位置的标准差 px ✓）⇒ 默认 `σ = 6 px`
#: （残差路线实测位置误差中位 ~1px、但夹取/落位那几拍会跳几十 px ✓ ⇒ 取个中间偏保守的值 ✓；
#: 影子模式的**主要产出之一**就是用新息数据反推"σ 该取多少" ✓ 见 `LieTracker` 的汇总 ✓）。
KF_R_DEF = 36.0
#: 速度初始不确定度（px/s ✓）：第一次观测时速度完全未知 ⇒ 方差给 `v0²`。
KF_V0_DEF = 120.0


class KF4:
    """**4 维匀速模型卡尔曼**：状态 `[px, py, vx, vy]` / 观测 `[px, py]`。

    用法（影子模式 ✓）：
        kf = KF4()
        for pos, ts in 逐拍:
            kf.predict(dt)                 # 估算值（上一拍 → 这一拍的外推 ✓）
            innov, nis = kf.update(pos)    # 测量值（这一拍的观测 ✓）
            kf.pos, kf.vel                 # 与经典链条的输出**对照** ✓（不改它 ✗）

    ⚠ `dt` 一律由调用方给（**真实时间戳之差** ✓ ⇒ 帧率 / 播放倍速无关 ✓ 与 B 型归一化同一纪律 ✓）；
      `dt ≤ 0`（第一拍 / 乱序 ✓）⇒ 调用方**跳过 predict** ✓（本类也容错 ✓）。
    """

    def __init__(self, q: float = KF_Q_DEF, r: float = KF_R_DEF,
                 v0: float = KF_V0_DEF) -> None:
        self.q = max(1e-9, float(q))
        self.r = max(1e-9, float(r))
        self.v0 = max(1e-6, float(v0))
        self.x: Optional[np.ndarray] = None      # 状态（列向量 ✓）；None = 还没吃到观测
        self.P: Optional[np.ndarray] = None      # 协方差
        self.n_seen = 0                          # 吃过几次观测（诊断 ✓）
        self.nis = 0.0                           # 上一次的归一化新息平方（诊断 ✓）

    # ---- 内部：Q（连续白噪声加速度 ✓ 标准式）----
    def _Q(self, dt: float) -> np.ndarray:
        _d2, _d3, _d4 = dt * dt, dt ** 3, dt ** 4
        _q = self.q
        return _q * np.array([
            [_d4 / 4.0, 0.0, _d3 / 2.0, 0.0],
            [0.0, _d4 / 4.0, 0.0, _d3 / 2.0],
            [_d3 / 2.0, 0.0, _d2, 0.0],
            [0.0, _d3 / 2.0, 0.0, _d2],
        ], dtype=float)

    def _F(self, dt: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    # ---- 初始化：第一次观测（速度未知 ⇒ 0、方差给 v0✓）----
    def _init(self, z: np.ndarray) -> None:
        self.x = np.array([float(z[0]), float(z[1]), 0.0, 0.0], dtype=float)
        self.P = np.diag([self.r, self.r, self.v0 ** 2, self.v0 ** 2]).astype(float)

    # ---- 预测步（估算值 ✓）----
    def predict(self, dt: float) -> None:
        """状态外推一步（`dt` 秒 ✓）。`dt ≤ 0` 或还没初始化 ⇒ **什么都不做** ✓（不猜 ✗）。"""
        if self.x is None or dt is None or float(dt) <= 0.0:
            return
        _dt = float(dt)
        _F = self._F(_dt)
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + self._Q(_dt)

    # ---- 更新步（测量值 ✓）----
    def update(self, x: float, y: float) -> Tuple[np.ndarray, float]:
        """用一次位置观测 `(x, y)`（**加工域** ✓）更新 ⇒ 返回 `(新息, NIS)`。

        · **新息** = `观测 − 先验预测`（2 维 ✓）—— 这是影子模式最值钱的产物 ✓：
          它的量级**直接告诉我们观测噪声该取多大** ✓（KF 的 R 标定就是看它 ✓）；
        · **NIS** = 新息ᵀ·S⁻¹·新息（S = 新息协方差 ✓）—— 归一化后**与单位无关** ✓（诊断用 ✓：
          均值该 ≈ 2（2 维观测 ✓）；远大于 2 ⇒ 观测里有离群跳变（比如夹取那一拍 ✓）✓）。
        ⚠ 第一次观测 ⇒ **初始化**（不返回有意义的新息 ✓ 回 0 向量 ✓）。
        """
        _z = np.array([float(x), float(y)], dtype=float)
        if self.x is None:
            self._init(_z)
            self.n_seen = 1
            return np.zeros(2), 0.0
        _H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        _R = np.eye(2) * self.r
        _y = _z - _H @ self.x                       # 新息（观测 − 先验 ✓）
        _S = _H @ self.P @ _H.T + _R                # 新息协方差
        try:
            _Sinv = np.linalg.inv(_S)
        except np.linalg.LinAlgError:               # noqa: BLE001 —— 数值异常 ⇒ 这一拍不更新 ✓
            return _y, 0.0
        _K = self.P @ _H.T @ _Sinv                  # 卡尔曼增益
        self.x = self.x + _K @ _y
        self.P = (np.eye(4) - _K @ _H) @ self.P
        self.n_seen += 1
        _nis = float(_y @ _Sinv @ _y) if _y.size else 0.0
        self.nis = _nis
        return _y, _nis

    # ---- 读取（对照用 ✓）----
    @property
    def pos(self) -> Optional[Tuple[float, float]]:
        """KF 估计的位置（加工域 ✓）；还没吃到观测 ⇒ `None` ✓。"""
        if self.x is None:
            return None
        return (float(self.x[0]), float(self.x[1]))

    @property
    def vel(self) -> Optional[Tuple[float, float]]:
        """KF 估计的速度（px/s ✓）；还没吃到观测 ⇒ `None` ✓。"""
        if self.x is None:
            return None
        return (float(self.x[2]), float(self.x[3]))

    def step(self, x: float, y: float, dt: Optional[float] = None) -> Dict:
        """便捷：`predict(dt)` + `update(x, y)` ⇒ 返回诊断字典（影子模式直接塞进输出 ✓）。

        `dt = None / ≤ 0` ⇒ **不预测、只更新** ✓（第一拍 / 时间戳异常 ✓ 不猜 ✗）。
        """
        self.predict(0.0 if dt is None else float(dt))
        _innov, _nis = self.update(x, y)
        _p, _v = self.pos, self.vel
        return {"pos": _p, "vel": _v, "nis": float(_nis),
                "innov": (float(_innov[0]), float(_innov[1])),
                "n": int(self.n_seen)}

    def reset(self) -> None:
        """新一局 / 换素材 ⇒ 清空（与 `LieTracker.reset()` 一起调 ✓）。"""
        self.x = None
        self.P = None
        self.n_seen = 0
        self.nis = 0.0
