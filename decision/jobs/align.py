"""「先站到目标 x 上」这一段（爬 / 传送门都要用，从 `decision/route.py` 切出）。

⚠ **一处口径**：对齐成功 = 偏差 ≤ 容差 **并且保持够久**，两个数都来自设置页。
"""

from decision.jobs.params import TAP_ON_S


#: 「**对齐 x 有进步**」的判据（世界像素）：`|dx|` 比纪录**小了这么多**才算真的靠近了 ✓
#: （读数的抖动不算进步 ✗ —— 不然"补按方向"永远不触发 ✓）。
ALIGN_PROGRESS_PX = 1.0


def align_x_decide(now, dx, *, press_dir, press_at, gap_ms, near_px, tol_px,
                   retry_ms=0, progress_at=None, best_dx=None):
    """「**把 x 对齐到某个数**」这一拍该怎么办 —— **爬绳与走传送点共用的一处实现** ✓。

    ⭐ 来源与用户口径（2026-10-06 ✓ 原话："我们应该**整套照搬对齐式爬绳梯的逻辑**，
      **还要补按方向**"✓）：规则**逐条**都是 2026-09-27/28 在爬绳上踩出来的 ✓
      （原实现内联在 `ClimbJob._align_step` ✓ 现在两边都调这里 ✓ 约定 10：一处实现 ✓）。

    规则（`kind` 三选一）：
      · `"wait"` —— 这一拍**不按键**（`move` 无意义 ✓）：
        ⓐ 近处（`|dx| ≤ near_px`）点按**按够了 `TAP_ON_S`(30ms)** ⇒ 松开
           （"别一路蹭过去 = 冲过头"✓）；
        ⓑ ⭐ 上一次按下距今**还没到「对齐绳梯移动延迟」`gap_ms`** ⇒ 站住等 ——
           ⚠⚠ **过冲之后往回按（换了方向）也走这一条** ⇒ **不会在目标两边来回抖** ✗✓
           （用户 2026-10-06 现场："在从5到4的 h010 传送门**左右来回晃**"✗ = 老 `PortalJob`
            每拍都按方向键 ✓ 这条正是治它的 ✓）；
      · `"press"` —— 这一拍按 `move`（±1 的横向方向键 ✓）；`reassert=True` ⇒ 调用方
        **再发一次 PRESS**（键集没变也发 ✓ —— `KeyState` 只在键集变化时发键，
        "对面把键丢了"本机不知道 ✗ 同 `ClimbJob` 那套补按 ✓）；
      · `"in_tol"` —— `|dx| ≤ tol_px` ⇒ **对齐好了**（调用方站住 / 稳住 `hold_ms` / 动作 ✓）。

    ⚠ **"远（> `near_px`）一口气按住 / 近了点按"** 这个分档也在里面：远距离点按会走得太慢、
      直接撞超时 ✗（同 `ClimbJob` 的口径 ✓）。
    ⚠ **只有"新的一次按下"才记 `press_at`**（2026-09-27 修的坑 ✓）：每拍都记 ⇒
      `now - press_at` 只有一帧(<60ms) ⇒ "按够 60ms 就松"**永不成立** ⇒ 变成一路按住 ✗（⚠ 那时 `TAP_ON_S` 是 **60ms**；**2026-10-06 起改成 30ms** ✓ 这条说的是当年的现象 ✓）。
    ⭐ **补按方向**（`retry_ms > 0` 才开 ✓ 用户 2026-10-06："还要补按方向"✓）：已经按着某个
      方向、可 `|dx|` 在「移动操作尝试间隔」(`retry_ms` ✓) 内**没进步**（改善 < `ALIGN_PROGRESS_PX`）
      ⇒ 回 `reassert=True` ⇒ 再发一次 PRESS ✓。
      ⚠ `ClimbJob` 的对齐相**用 `retry_ms=0` 关掉它** ✓（那边行为一个字不改 ✓ 它 CLIMB 相
        本来就有自己的补按 ↑ ✓ 别叠两层 ✗）。

    → `(kind, move, reassert, why, state)`；`state` 是**该写回**的记账
      `{press_dir, press_at, progress_at, best_dx, reasserts}` ✓（调用方自己存哪儿都行 ✓）。
    """
    now = float(now)
    dx = float(dx)
    gap_ms = max(0, int(gap_ms or 0))
    retry_ms = max(0, int(retry_ms))
    near_px = max(1.0, float(near_px))
    tol_px = max(1.0, float(tol_px))
    st = {"press_dir": int(press_dir or 0),
          "press_at": None if press_at is None else float(press_at),
          "progress_at": None if progress_at is None else float(progress_at),
          "best_dx": best_dx,
          "reasserts": 0}
    # ---- 进步记账（只服务"补按方向" ✓）：真的靠近了才刷新 ✓ ----
    _d = abs(dx)
    if st["best_dx"] is None or _d < float(st["best_dx"]) - ALIGN_PROGRESS_PX:
        st["best_dx"] = _d
        st["progress_at"] = now
    elif st["progress_at"] is None:
        st["progress_at"] = now
    # ---- ① 对齐好了 ⇒ 松开方向（"站住"✓ 剩下的交给调用方 ✓）----
    if _d <= tol_px:
        st["press_dir"] = 0
        return ("in_tol", 0, False,
                "对齐好了：差 %.0f px（容许 %d）—— 站住 ✓" % (dx, int(tol_px)), st)
    sign = 1 if dx > 0 else -1
    _held = (st["press_dir"] == sign)
    _since = (None if st["press_at"] is None else now - st["press_at"])
    # ---- ② 近处点按：按够 TAP_ON_S 就松开（别冲过头 ✓）----
    if (_held and _d <= near_px and _since is not None and _since >= TAP_ON_S):
        st["press_dir"] = 0
        return ("wait", 0, False, "微调：差 %.0f px（松开这一拍，别冲过头）" % dx, st)
    # ---- ③ 两次按下之间要隔 gap_ms（**换方向回来按也算** ⇒ 不会两边来回抖 ✓）----
    if (not _held and _since is not None and gap_ms > 0
            and _since * 1000.0 < gap_ms):
        st["press_dir"] = 0
        return ("wait", 0, False,
                "对齐中：差 %.0f px（等移动延迟 %d ms）" % (dx, gap_ms), st)
    # ---- ④ 补按方向（可选）：按着却一直没进步 ⇒ 再发一次 PRESS ✓ ----
    if (retry_ms > 0 and _held and st["progress_at"] is not None
            and (now - float(st["progress_at"])) * 1000.0 >= retry_ms):
        st["press_at"] = now
        st["progress_at"] = now
        st["reasserts"] += 1
        return ("press", sign, True,
                "对齐：差 %.0f px 一直没动 ⇒ **补按一次 %s**（第 %d 次 ✓）"
                % (dx, "→" if sign > 0 else "←", st["reasserts"]), st)
    # ---- ⑤ 按下去（远距离 ⇒ 一口气按住 ✓ / 近距离 ⇒ 一下一下点 ✓）----
    if not _held:
        st["press_at"] = now          # ⚠ 只有"新的一次按下"才记 ✓
    st["press_dir"] = sign
    if _d <= near_px:
        return ("press", sign, False,
                "微调：差 %.0f px（点按 %s）" % (dx, "→" if sign > 0 else "←"), st)
    return ("press", sign, False, "对齐中：差 %.0f px（容许 %d）" % (dx, int(tol_px)), st)
