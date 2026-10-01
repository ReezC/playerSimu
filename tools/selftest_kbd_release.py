"""自检：**"把键松干净"那条路**（用户 2026-10-01 ✓）。

现场有两件事，必须分开看，别混成一个问题：

  · **"停了自动，它还在自己按输出键、左右走"** —— 那是**已经发出去**的命令还排在后
    面（固件逐条执行，`TAP`/`FIX`/`RND` 内部还会原地等）⇒ 收尾靠**停自动时补一条
    `RELEASEALL`**（就是本文件守着的那条路 ✓）。
  · 曾经试过从**源头限流**（按键背压：同一时刻只让少量指令在飞）来掐掉那个队列 ——
    **连着三次把按键通道搞到不可用**（特别卡 / 输入全无效 / 方向键发不出去 ✗）
    ⇒ **整个撤掉了**（2026-10-01，见 `decision/agent.py` 里那段说明）。
    ⚠ 要重做的话，**先量 `perf.log` 里的 `kbd_rtt_ms`**（B 发指令 → A 固件回执的真实
      往返和抖动），拿数据说话 —— 前三次全是"照代码推断"拍出来的，都错了 ✗。

跑法：
    python -m tools.selftest_kbd_release
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认是 **GBK**：用例名里带 ⇒ / ✗ 这类字符时 `print` 会抛
# `UnicodeEncodeError` —— 那和被测的东西无关，是终端的编码问题 ⇒ 切成 UTF-8 ✓。
try:
    sys.stdout.reconfigure(encoding="utf-8")            # type: ignore[attr-defined]
except Exception:                                       # noqa: BLE001
    pass


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ══════════════════════ 用例 ══════════════════════

def t_stop_automation_releases_keys():
    """**停自动 ⇒ 真的发一条 RELEASEALL**：固件那边这才第一次被告知"松手" ✓。

    为什么钉它：老路子是"不再发新命令"就完事 —— 可**已经进管道的照走不误** ⇒ 现场就是
    "关了自动还在自己按"。这条收尾一旦被谁删掉，那个现象会**原样回来而且没有任何报错**
    （所以必须钉住 ✓）。
    """
    src = (ROOT / "decision" / "agent.py").read_text(encoding="utf-8")
    check("if not s.enabled and self._was_enabled:" in src,
          "找不到「关闭自动」的下降沿分支（被删了？）⇒ 会重现「停了还在自己按」✗")
    check("self._release_keys_now()" in src,
          "下降沿没有调用抽出后的松键入口 ⇒ 本地/远程两种后端会漏一种 ✗")
    # ⚠ 停自动**不许**把键重按回去（`_reassert_ctx`）——那等于停了又按起来 ✗
    _i = src.find("if not s.enabled and self._was_enabled:")
    _seg = src[_i:_i + 1600]
    check("_release_keys_now" in _seg, "下降沿分支里看不到松键调用 ✗")
    check("_reassert_ctx" not in _seg,
          "停自动的分支里出现了 `_reassert_ctx` ⇒ 松完又按回去 ✗（那等于没停）")


def t_release_keys_covers_both_backends():
    """松键那唯一一处写法必须**同时认两种后端**（2026-09-26 现场踩过 ✗）。

      · 远程/串口：固件一条 `RELEASEALL` 全清 ⇒ 本机**只**清记录（再补发 up 是错的）；
      · 本地(仅测试)：没有固件，`release_all_remote()` 是空操作 ⇒ 必须逐键发 up，
        只清记录的话那个键在本机就**永远按着** ✗（实测只看得见 down、没有 up）。
    """
    src = (ROOT / "decision" / "agent.py").read_text(encoding="utf-8")
    _i = src.find("def _release_keys_now(self):")
    check(_i >= 0, "找不到 `_release_keys_now`（松键的唯一写法被拆散了？）✗")
    _seg = src[_i:_i + 2200]
    check('== "local"' in _seg, "没按后端分岔 —— 两种后端会被当成一种 ✗")
    check("keys.release_all()" in _seg, "本地后端那条路没了 ⇒ 键在本机永远按着 ✗")
    check("release_all_remote()" in _seg and "keys.clear()" in _seg,
          "远程后端那条路不全（固件清完之后本机记录也得同步清 ✓）✗")


def t_backpressure_is_gone():
    """⭐ **限流（背压）已经被整个撤掉** —— 这条是"别再悄悄加回来"的闸门 ✗。

    它连着三次把按键通道搞到不可用（特别卡 / 输入全无效 / 方向键发不出去）。撤掉是
    用户 2026-10-01 定的 ✓。谁要再加，得先拿 `perf.log` 的 `kbd_rtt_ms` 说话 ✗。
    """
    import inspect

    from remote_kbd import kbd_client as kc

    check(not hasattr(kc.KbdClient, "_backpressure"),
          "`_backpressure` 又回来了？要重做请先量 `kbd_rtt_ms` 再谈 ✗")
    check(not hasattr(kc.KbdClient, "MAX_INFLIGHT"),
          "`MAX_INFLIGHT` 又回来了 ✗")
    _src = inspect.getsource(kc.KbdClient.send)
    check("_backpressure" not in _src and "MAX_INFLIGHT" not in _src,
          "`send()` 里又在做限流了 ✗")
    # 参数也别留着（留着就会有人以为它有用 ✗）
    from decision.agent import DecisionSettings
    check(not hasattr(DecisionSettings(), "kbd_max_inflight"),
          "设置对象上还留着 `kbd_max_inflight` ⇒ 界面上那个旋钮的残骸 ✗")


TESTS = (
    ("停自动要真的松干净（下降沿 + 不许把键按回去）", t_stop_automation_releases_keys),
    ("松键那条唯一写法同时认两种后端", t_release_keys_covers_both_backends),
    ("限流（背压）已整个撤掉、别悄悄加回来", t_backpressure_is_gone),
)


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:                             # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
