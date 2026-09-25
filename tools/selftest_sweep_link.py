"""握手通道自检：**把"对上"这条性质钉死**。

为什么要单列一个文件：这套东西的价值全在"两边量的必须是同一段"。而它出错的时候
**不会报错** —— 两边都收到了真流、都量出了合理的数字，只是不是同一套参数，
最后得到一张"每条都有数、没有一条对得上"的表（2026-09-25 实测白跑一轮）。
所以这里逐条钉住：晚起能对上、忙一下能跳到最新、过期的不认、收不到就明说退回老流程。

**不碰网络配置**：端口全用临时空闲端口，只在环回上跑，不动 config/ 里任何文件。
"""

import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import sweep_link                                       # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _free_port():
    """挑一个空闲的 UDP 端口（握手不需要固定号；只有 config/link.yaml 里那个才固定）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _pair():
    """一对（A 侧公告器, B 侧监听器），都在环回上。"""
    port = _free_port()
    return sweep_link.Announcer("127.0.0.1", port, bind_port=_free_port()), \
        sweep_link.Listener(port, host="127.0.0.1")


def _drain_acks(ann, want, timeout=1.5):
    """等回执到齐（UDP 环回很快，但别用 sleep 猜时长）。"""
    t0 = time.monotonic()
    got = []
    while time.monotonic() - t0 < timeout and len(got) < want:
        got.extend(ann.poll_acks())
        time.sleep(0.02)
    return got


# ---------------------------------------------------------------- 协议本体

def t_roundtrip_and_ack():
    """公告里的字段要原样到 B 机；B 机的回执要能回到 A 机那张 socket 上。"""
    ann, lis = _pair()
    try:
        preset = {"name": "60fps-6M-g30", "fps": 60, "bitrate": "6M", "gop": 30}
        check(ann.seg(2, 18, preset, 20.0, 4.0), "公告发不出去")
        seg = lis.wait_seg(2.0)
        check(seg is not None, "B 机没收到公告")
        check(seg["i"] == 2 and seg["n"] == 18, "段的序号/总数不对：%s" % (seg,))
        check(seg["preset"] == preset,
              "参数没带过去（B 机就不必持有同一份清单了）：%s" % (seg.get("preset"),))
        check(seg["seconds"] == 20.0 and seg["settle"] == 4.0,
              "测量窗口没带过去：%s" % (seg,))
        check(lis.ack(seg), "回执发不出去")
        acks = _drain_acks(ann, 1)
        check(len(acks) == 1 and acks[0]["seg"] == 2,
              "A 机没收到回执（部署台那行「B 机已跟上」就是它）：%s" % (acks,))
    finally:
        ann.close()
        lis.close()


def t_late_start_still_aligns():
    """**B 机晚起也必须对上** —— 这是整个模块存在的理由。

    模拟今天那次事故：A 机在 B 机开始听之前已经公告过 3 段（那 3 段的测量窗口早就过去），
    现在正在跑第 4 段。B 机（晚起）拿到的必须是**第 4 段**，不是它缓冲里的第 1 段。
    """
    ann, lis = _pair()
    try:
        now = time.time()
        for i in range(3):                      # 三段"早就量完了"的公告
            ann.seg(i, 18, {"name": "old%d" % i}, 20.0, 4.0)
        time.sleep(0.15)
        ann.seg(3, 18, {"name": "now"}, 20.0, 4.0)          # 现在这段
        seg = lis.wait_seg(1.0)
        check(seg is not None, "B 机没收到公告")
        check(seg["i"] == 3 and seg["preset"]["name"] == "now",
              "晚起的 B 机对到了过期的段：%s（应当对齐到第 4 段）" % (seg,))
        check(not sweep_link.seg_is_stale(seg), "刚公告的段被判成过期了")
        check(sweep_link.seg_age(seg, now + 60) > 55, "算出来的年龄不对")
    finally:
        ann.close()
        lis.close()


def t_slow_consumer_skips_forward():
    """B 机忙一下（没及时调 wait_seg）时，要跳到**最新**那段，而不是慢慢补课。

    补课的后果是"一直落后一段"：每段都量到了，但量的是上一套参数。
    """
    ann, lis = _pair()
    try:
        for i in range(4):
            ann.seg(i, 18, {"name": "p%d" % i}, 20.0, 4.0)
        time.sleep(0.2)                          # 积压 4 条，B 机现在才来收
        seg = lis.wait_seg(1.0)
        check(seg is not None and seg["i"] == 3,
              "积压时没跳到最新那段：%s（应当是最新的第 4 段）" % (seg,))
    finally:
        ann.close()
        lis.close()


def t_new_sweep_beats_old_leftovers():
    """A 机连着重开一轮时，上一轮残留的高段号不能骗过对齐。

    残留那条的段号更大（在"谁更新"里显得更靠后），但它属于**上一轮** ——
    不排掉的话 B 机会一直追着上一轮的尾巴跑，而 A 机早就在新的一轮里了。
    """
    ann, lis = _pair()
    try:
        ann.seg(3, 18, {"name": "上一轮第4段"}, 20.0, 4.0)      # 上一轮残留
        time.sleep(0.1)
        ann.start(18)                                          # 新一轮开工
        ann.seg(0, 18, {"name": "新一轮第1段"}, 20.0, 4.0)
        time.sleep(0.1)
        seg = lis.wait_seg(1.0)
        check(seg is not None, "B 机没收到公告")
        check(seg["i"] == 0 and seg["preset"]["name"] == "新一轮第1段",
              "被上一轮的残留带跑了：%s（应当对齐到新一轮的第 1 段）" % (seg,))
    finally:
        ann.close()
        lis.close()


def t_no_handshake_is_visible():
    """收不到公告时要**能判断出来**（外面据此退回按清单顺序那套老流程）。"""
    ann = sweep_link.Announcer("127.0.0.1", _free_port(), bind_port=_free_port())
    lis = sweep_link.Listener(_free_port(), host="127.0.0.1")
    try:
        t0 = time.monotonic()
        check(lis.wait_seg(0.3) is None, "没有公告却返回了一段")
        check(time.monotonic() - t0 < 1.0, "没有公告时等太久了（每段都要等，很浪费时间）")
        check(lis.saw_any is False, "没收到公告却报告收到了（会不退回落流程）")
        check(lis.done_seen is False, "没收到 done 却报告收到了")
    finally:
        ann.close()
        lis.close()


def t_done_stops_waiting():
    """A 机报 done 之后不许再等下去（否则每轮结尾都要干等一个超时）。"""
    ann, lis = _pair()
    try:
        ann.start(18)
        ann.done(18)
        time.sleep(0.15)
        t0 = time.monotonic()
        check(lis.wait_seg(5.0) is None, "done 之后还返回了段")
        check(time.monotonic() - t0 < 1.0, "收到 done 之后还在等")
        check(lis.done_seen, "没记住 done（下一条 wait_seg 还会继续等）")
    finally:
        ann.close()
        lis.close()


def t_garbage_is_ignored():
    """端口上什么都可能有：乱码 / 别的版本 / 别的 kind —— 一律当没听懂，不许炸。"""
    for bad in (b"", b"\x00\x01\x02", b"not json", b"[1,2,3]",
                b'{"v":99,"kind":"seg","i":0}',
                b'{"v":1,"kind":"\xe5\x88\xab\xe7\x9a\x84"}'):
        check(sweep_link.decode(bad) is None,
              "垃圾包被当成协议报文了：%r" % (bad,))
    ann, lis = _pair()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for bad in (b"\x00\x01", b"not json", b'{"v":99,"kind":"seg"}'):
            s.sendto(bad, ("127.0.0.1", lis.port))
        s.close()
        time.sleep(0.1)
        ann.seg(5, 18, {"name": "ok"}, 20.0, 4.0)
        seg = lis.wait_seg(1.0)
        check(seg is not None and seg["i"] == 5,
              "垃圾包把真公告挡住了：%s" % (seg,))
    finally:
        ann.close()
        lis.close()


# ---------------------------------------------------------------- 报告 / 结论的整段文本

def t_blob_reaches_both_sides():
    """报告与结论的整段文本要能原样过去（报告走 A→B，结论走 B→A）。

    这就是"两个开关"的传输层：B 机不用去 A 机拷 `perf_push_A.json`，
    A 机也不用去 B 机抄那张表 —— 都走同一条握手通道。
    """
    ann, lis = _pair()
    try:
        # 用**不可压缩**的内容，才验得到真分片：JSON/重复串压完之后常常一片就发完了
        # （第一版就是这么写的：断言"≥2 片"直接挂 —— 压缩比太大）。
        report = os.urandom(4000).hex()
        check(ann.send_blob("rep", report) >= 2, "长报告没分片（应当 ≥2 片）")
        got = lis.wait_blob("rep", 3.0)
        check(got == report,
              "A→B 的报告没原样到（%d vs %d 字节）" % (len(got or ""), len(report)))

        # B 回结论的地址是**从公告里学来的**（不用配 A 机 IP）
        res = "表\n结论：用 144fps-20M-g30"
        check(lis.send_blob("res", res) >= 1, "回结论发不出去")
        back = ann.wait_blob("res", 3.0)
        check(back == res, "B→A 的结论没原样到：%r" % (back,))
    finally:
        ann.close()
        lis.close()


def t_blob_loss_does_not_raise():
    """丢片只能"没收到"，绝不许抛 —— 现场是 UDP，丢一片很常见。"""
    ann, lis = _pair()
    try:
        text = os.urandom(4000).hex()              # 不可压缩，才分得出多片
        parts = sweep_link.blob_msgs(text)
        check(len(parts) >= 3, "（前置）这段该分成 ≥3 片，实际 %d" % len(parts))
        for m in parts[:-1]:                       # 故意漏掉最后一片
            ann.send("rep", **{k: v for k, v in m.items() if k != "kind"})
        check(lis.wait_blob("rep", 0.6) is None, "缺片却拼出了一段文本")

        # 乱序也要能拼回来（UDP 不保证顺序）
        asm = sweep_link.BlobAssembler(5.0)
        got = None
        for m in reversed(parts):
            got = asm.feed(m) or got
        check(got == text, "乱序的片没拼回来")
    finally:
        ann.close()
        lis.close()


def t_auto_flow_end_to_end():
    """**"两个开关"的完整回路**：A 公告 → B 逐段量 → A 发报告 → B 出表 → 结论回 A。

    整条在本地跑（不占真实配置端口、不需要真流）：这是"以后只要点两个开关"那句话的
    可执行证据，也是唯一能一次性钉住"分段 → 报告 → 合并 → 回传"四段接线的用例。
    """
    import json
    import tempfile
    import threading
    import unittest.mock as mock

    from tools import push_presets as pp
    from tools.stream_sweep import finish_and_report, iter_segments

    port = _free_port()
    ann = sweep_link.Announcer("127.0.0.1", port, bind_port=_free_port())
    lis = sweep_link.Listener(port, host="127.0.0.1")
    presets = [{"name": "A机第%d条" % (i + 1)} for i in range(3)]
    res_seen = []

    def a_side():
        """模拟部署台那一侧：公告三段 → 发报告 → 等结论。

        段间隔**必须大于 `wait_seg` 的排空窗口（0.25s）**：那段安静期用来"只认最新"，
        间距太近会把连着发的几段合并且只留最后一段（真机是 24s 一段，不会碰上）。
        第一版这里写了 0.15s，于是 B 只量到第 3 段 —— 用例自己先绊了一跤。
        """
        ann.start(3)
        for i, p in enumerate(presets):
            ann.seg(i, 3, p, 20.0, 4.0)
            time.sleep(0.45)
        rows = [pp.row_a(presets[0]["name"], 0, presets[0], 1.0, 2.0, []),
                pp.row_a(presets[1]["name"], 1, presets[1], 3.0, 4.0, [])]
        ann.done(3)
        time.sleep(0.1)
        ann.send_blob("rep", json.dumps({"rows": rows}, ensure_ascii=False))
        res_seen.append(ann.wait_blob("res", 6.0))

    tmpdir = Path(tempfile.mkdtemp(prefix="sweepauto_"))
    th = None
    try:
        th = threading.Thread(target=a_side, daemon=True)
        th.start()
        segs = [(i, p["name"]) for i, p, _s, _st, _t in
                iter_segments(lis, [], 20.0, 4.0, on_line=lambda _t: None)]
        check([s[0] for s in segs] == [0, 1, 2],
              "B 机没按 A 机的公告走：%s" % (segs,))
        # B 机那份记账（真跑时由逐段测量写出来）
        b_rows = [pp.row_b(p["name"], i, p, 1.0, 2.0,
                           {"recv_fps": 60.0, "lat_p50": 120.0 + i,
                            "lat_p95": 140.0 + i, "probe_mono": True})
                  for i, p in enumerate(presets)]
        b_path = tmpdir / "perf_push_B.json"
        b_path.write_text(json.dumps({"rows": b_rows}, ensure_ascii=False),
                          encoding="utf-8")
        with mock.patch.object(pp, "A_REPORT", str(tmpdir / "perf_push_A.json")):
            text = finish_and_report(lis, b_path, say=lambda _t: None)
            th.join(timeout=6)
            check(text and len(text) > 40, "结论文本太短：%r" % (text,))
            check(res_seen and res_seen[0] == text,
                  "A 机没收到（或收到的不是同一份）结论：%r" % (res_seen,))
            check("把最优那条写回部署台" in text or "没有可比的组合" in text,
                  "结论里没有「怎么用」那一段：%s" % text[:200])
    finally:
        if th is not None:
            th.join(timeout=8)        # 先让它自己收尾，别把 socket 从它脚下抽走
        ann.close()
        lis.close()
        import shutil
        shutil.rmtree(str(tmpdir), ignore_errors=True)


# ---------------------------------------------------------------- B 机侧：段的来源

class _FakeLink:
    """假握手通道：按脚本吐出公告，记下回执（不碰网络，能确定性地驱动用例）。"""

    def __init__(self, segs):
        self._segs = list(segs)
        self.acks = []
        self.saw_any = bool(segs)

    def wait_seg(self, _timeout):
        return self._segs.pop(0) if self._segs else None

    def ack(self, seg, **_extra):
        self.acks.append(seg.get("i"))
        return True


def t_iter_segments_follows_a_not_local_list():
    """有公告时**按 A 机的公告走**，连自己手里那份清单都不用（顺序/内容都可以不一样）。"""
    from tools.stream_sweep import iter_segments

    local = [{"name": "本地第1条"}, {"name": "本地第2条"}]
    link = _FakeLink([
        {"i": 7, "n": 9, "t": time.time(), "seconds": 20.0, "settle": 4.0,
         "preset": {"name": "A机第8条"}},
        {"i": 8, "n": 9, "t": time.time(), "seconds": 25.0, "settle": 5.0,
         "preset": {"name": "A机第9条"}},
    ])
    got = list(iter_segments(link, local, 20.0, 4.0, on_line=lambda _t: None))
    check([g[0] for g in got] == [7, 8],
          "没按 A 机的段号走（晚起就会这样错位）：%s" % ([g[0] for g in got],))
    check([g[1]["name"] for g in got] == ["A机第8条", "A机第9条"],
          "用了本机清单而不是公告里的参数：%s" % ([g[1]["name"] for g in got],))
    check(got[1][2] == 25.0 and got[1][3] == 5.0,
          "没按公告里的测量窗口（A 机说了算）：%s" % (got[1][2:4],))
    check(link.acks == [7, 8], "没对收到的每段回执：%s" % (link.acks,))


def t_iter_segments_falls_back_when_silent():
    """一条公告都没有 → 退回按清单顺序（老流程还能用），并且**说出来**。"""
    from tools.stream_sweep import iter_segments

    local = [{"name": "本地第1条"}, {"name": "本地第2条"}]
    said = []
    got = list(iter_segments(_FakeLink([]), local, 20.0, 4.0, on_line=said.append))
    check([g[0] for g in got] == [0, 1], "退回后没按清单顺序走：%s" % (got,))
    check(any("退回" in s for s in said),
          "退回老流程时没告诉人（现场会以为握手生效了）：%s" % (said,))

    got2 = list(iter_segments(None, local, 20.0, 4.0, on_line=lambda _t: None))
    check([g[0] for g in got2] == [0, 1], "没有通道时也没退回：%s" % (got2,))


# ---------------------------------------------------------------- A 机侧：接线

def t_push_sweep_announces():
    """A 机的自检线程要能按配置把公告发出去（端口/IP 都来自配置）。"""
    from deploy import push_sweep

    port = _free_port()
    ann = push_sweep.make_announcer({"push": {"host": "127.0.0.1"},
                                     "sweep": {"port": port}})
    check(ann is not None, "按配置建不出公告通道")
    lis = sweep_link.Listener(port, host="127.0.0.1")
    try:
        check(ann.seg(0, 18, {"name": "x", "fps": 60}, 20.0, 4.0), "公告发不出去")
        seg = lis.wait_seg(2.0)
        check(seg is not None and seg["i"] == 0, "A 机侧没发出去：%s" % (seg,))
        check(seg["preset"]["fps"] == 60, "参数没带上：%s" % (seg.get("preset"),))
    finally:
        ann.close()
        lis.close()

    # 配置里没有 B 机地址 → 不硬造一个，返回 None（自检照常跑，只是没握手）
    check(push_sweep.make_announcer({"push": {}}) is None,
          "没有 B 机地址却建出了通道")


TESTS = (
    ("公告/回执能往返（含参数）", t_roundtrip_and_ack),
    ("B 机晚起也对得上（过期公告不认）", t_late_start_still_aligns),
    ("B 机忙一下会跳到最新那段", t_slow_consumer_skips_forward),
    ("A 机重开一轮时不受上一轮残留影响", t_new_sweep_beats_old_leftovers),
    ("收不到公告能判断出来（好退回老流程）", t_no_handshake_is_visible),
    ("A 机说 done 之后不再等", t_done_stops_waiting),
    ("垃圾包一律当没听懂", t_garbage_is_ignored),
    ("报告/结论的整段文本能双向送达", t_blob_reaches_both_sides),
    ("丢片/乱序只当没收到，不许抛", t_blob_loss_does_not_raise),
    ("两个开关的完整回路（公告→量→报告→出表→回传）", t_auto_flow_end_to_end),
    ("B 机按 A 机公告走，不按本机清单", t_iter_segments_follows_a_not_local_list),
    ("收不到公告就退回按清单顺序（并说出来）", t_iter_segments_falls_back_when_silent),
    ("A 机的公告按配置发得出去", t_push_sweep_announces),
)


def main():
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print("[ OK ] %s" % name)
        except Exception as e:                     # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.exit(main())
