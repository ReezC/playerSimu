"""推流自检的握手通道：A 机**公告**每段开始，B 机据此对齐并回**回执**。

**它解决什么**
    原先两边靠"先点 A 机开始、紧接着（几秒内）跑 B 机命令 + 各自记墙钟、最后对账"。
    这套做法把正确性押在**人掐时间**上：2026-09-25 实测 B 机晚了 3.7 分钟才起，
    A 机已经在量第 10 段，B 机却从第 1 段开始量 —— **两边都在量真流，量的却不是
    同一套参数**，最后得到一张"每条都有数、但没有一条对得上"的表。

**怎么握手**（一条 UDP，A → B；谁都不用等谁）
    1. A 机每段开始时发一条 `seg`：第几段 / 共几段 / 段名 / **该段的完整参数** /
       测量窗口（seconds + settle）；
    2. B 机收到就**按它**量这一段，不再认自己那份清单的顺序 —— 于是 B 机早跑、晚跑、
       中途重启都不影响，连"两边清单必须完全一致"这个隐含前提也一起消掉了
       （参数是 A 机公告过来的）；
    3. B 机回一条 `ack`，A 机的进度日志里就能显示"B 机已跟上"。

**为什么不做成"双向阻塞式"握手**：那会引入"B 机没起来 → A 机自检卡住"这种新故障，
而 A 机侧那几个数（speed / 丢帧 / CPU）本来就自己成立。这里要的是**对上**，不是绑死。

**两条不能省的细节**（都是"看起来能跑、其实会错"的那类）
    · **只认最新的那段**：B 机一次忙几百毫秒，socket 里可能压着两三条公告。
      按顺序从最旧那个开始量，就会一直落后一段。所以 `wait_seg` 把积压的先收干净、
      返回**最新**的那条。
    · **过期的要丢掉**：B 机刚绑上端口时，缓冲里可能躺着半分钟前那条（早就量完了）。
      只凭 `t` 判断：比"公告里的测量窗口 + 余量"还旧的，一律不认。
      两台机器已对时（差 ~1ms，见 tools/clock_sync.py），所以墙钟可以直接比。

**收不到公告怎么办**：B 机退回原来那套（按清单顺序 + 墙钟对账），并且**明说**自己
退回去了 —— 老流程仍然能用，只是继续需要人掐时间。
"""

import json
import socket
import time

#: 握手端口。**两台机器各自绑同一个号**（A 绑它用来收 B 的回执，B 绑它用来听公告）。
#: 5000 推流、5001 对时、5003 小地图，所以用 5002；配置项见 config/link.yaml 的 sweep.port。
DEFAULT_PORT = 5002

#: 协议版本。对不上就整条丢掉（宁可退回老流程，也别按半懂的消息去对齐）。
VERSION = 1

#: 报文大小上限。公告里带的是段名 + 几个数字（~250 字节），远远不到 UDP 的安全线。
MAX_PKT = 1200

#: 过期判定的余量（秒）：公告里的 (seconds + settle) 再加这么多，超过就认为那段早结束了。
#: 留余量是因为 ffmpeg 起停比标称的窗口长一点（重启本身要一两秒）。
STALE_SLACK = 5.0

_KINDS = ("start", "seg", "done", "ack")


def encode(msg):
    """dict → bytes。"""
    m = dict(msg)
    m["v"] = VERSION
    return json.dumps(m, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode(data):
    """bytes → dict；**不是本协议的一律返回 None，绝不抛**。

    这条端口上什么都可能有（另一个程序、上一个版本的报文、扫描包），
    解析失败必须只是"没听懂"，不能让收公告的那一侧炸掉。
    """
    try:
        m = json.loads(data.decode("utf-8", "replace"))
    except Exception:
        return None
    if not isinstance(m, dict) or m.get("v") != VERSION:
        return None
    if m.get("kind") not in _KINDS:
        return None
    return m


def seg_key(seg):
    """排队里"谁更新"的排序键：**先比段号**，再比时间戳。

    为什么不能只看时间戳：Windows 上 `time.time()` 的分辨率约 1ms，挨着发的几条会拿到
    **完全相同**的 `t`（实测 4 条里 3 条同值）。于是"严格大于才算更新"就会一直留着
    最先收到的那条 —— 现场表现正是"永远落后一段"。段号才是一轮自检里的顺序号
    （单调递增，不会重复）；完全相同时由调用方按**后到者更新**处理。
    """
    return (int(seg.get("i") or 0), float(seg.get("t") or 0.0))


def seg_age(seg, now=None):
    """这条公告距今多少秒（A/B 已对时，直接比墙钟）。缺 `t` 时返回 None。"""
    t = seg.get("t")
    if not isinstance(t, (int, float)):
        return None
    return (now if now is not None else time.time()) - float(t)


def seg_is_stale(seg, now=None):
    """这条公告是不是"早就量完了"的那种（见 STALE_SLACK）。"""
    age = seg_age(seg, now)
    if age is None:
        return False                     # 没有时间戳：判不了，当它有效
    win = float(seg.get("seconds") or 0.0) + float(seg.get("settle") or 0.0)
    return age > win + STALE_SLACK


class Announcer:
    """A 侧：往 B 机发公告，顺手（非阻塞地）收 B 机的回执。

    绑到 `bind_port`（默认就是公告端口）**是为了让回执能回来**：B 机是回给
    "这条公告的源地址:源端口"的，而源端口就是这个 socket 绑的那个。
    """

    def __init__(self, host, port=DEFAULT_PORT, bind_port=None):
        self.host = str(host)
        self.port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", int(DEFAULT_PORT if bind_port is None
                                      else bind_port)))
        self.sock.setblocking(False)

    def send(self, kind, **fields):
        """发一条。**失败只返回 False 不上抛** —— 网络出问题不该把自检弄崩。"""
        fields["kind"] = kind
        fields.setdefault("t", time.time())
        fields.setdefault("who", socket.gethostname())
        try:
            self.sock.sendto(encode(fields), (self.host, self.port))
            return True
        except Exception:                    # noqa: BLE001
            return False

    def start(self, n):
        return self.send("start", n=int(n))

    def seg(self, i, n, preset, seconds, settle):
        """第 i 段（从 0 数）开始。**参数一起带过去**，B 机就不必和 A 机持有同一份清单。"""
        return self.send("seg", i=int(i), n=int(n), preset=dict(preset),
                         seconds=float(seconds), settle=float(settle))

    def done(self, n):
        return self.send("done", n=int(n))

    def poll_acks(self):
        """收干净已到的回执 → [ack dict]（非阻塞，随时调）。"""
        out = []
        while True:
            try:
                data, _addr = self.sock.recvfrom(MAX_PKT)
            except (BlockingIOError, OSError):
                break
            m = decode(data)
            if m and m.get("kind") == "ack":
                out.append(m)
        return out

    def close(self):
        try:
            self.sock.close()
        except Exception:                    # noqa: BLE001
            pass


class Listener:
    """B 侧：听 A 机的公告，并对收到的每一段回一条回执。

    **故意不设 SO_REUSEADDR**：Windows 上它会允许两个 stream_sweep 同时绑同一个端口，
    公告就被随机分给其中一个（现场表现为"有时候收得到、有时候收不到"，极难查）。
    绑不上说明已经有一个在跑 —— 让人看见，别静默分票。
    """

    def __init__(self, port=DEFAULT_PORT, host="0.0.0.0"):
        self.port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, self.port))
        self.sock.settimeout(0.25)
        self.saw_any = False        # 收过公告没有（决定要不要退回老流程）
        self.done_seen = False      # 收到 done（A 机跑完了，别再等下去）
        self._seg_addr = None       # 最近一条公告的来源（回执发给它）
        self._sweep_t = 0.0         # 这一轮自检的开工时刻（用来丢掉上一轮的残留公告）

    def wait_seg(self, timeout):
        """等一条 `seg` 公告（最多 timeout 秒）→ dict / None。

        规则有三条，都是必须的：
          · **只认最新的**：先把积压的收干净再返回"最新"那条（按 **段号** 排，见 seg_key；
            B 机忙一下就落后一段的那种错，靠这条挡住）；
          · **过期的丢掉**：按公告里的测量窗口判断早就结束的，不认（见 seg_is_stale）；
          · **上一轮的残留也丢掉**：A 机连着重开一轮时，上一轮的高段号会显得"更新"，
            所以记下这一轮 `start` 的时刻，比它还早的段一律不认。
        中间的 `start`/`done`/`ack`/垃圾包按各自的方式处理，不会当成段返回。
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        best = None
        while True:
            if self.done_seen:
                return None
            left = deadline - time.monotonic()
            self.sock.settimeout(0.0 if left <= 0 else min(0.25, left))
            try:
                data, addr = self.sock.recvfrom(MAX_PKT)
            except (socket.timeout, BlockingIOError):
                data = None
            except OSError:
                return None
            if data is not None:
                m = decode(data)
                if m:
                    if m["kind"] == "start":
                        # 这一轮自检的开工时刻。**同时把之前攒的丢掉**：那些段属于
                        # 上一轮（它们比新一轮先到，所以已经进了 best，光靠时刻比较
                        # 追不回来 —— 上一轮残留的段号更大，看起来"更新"）。
                        self._sweep_t = float(m.get("t") or time.time())
                        best = None
                    elif m["kind"] == "done":
                        self.done_seen = True
                        if best is not None:
                            self._seg_addr = addr
                            return best
                        return None
                    elif m["kind"] == "seg" and not seg_is_stale(m) \
                            and not self._from_old_sweep(m):
                        # `>=` 而不是 `>`：Windows 上相邻两条的 t 可能**完全相等**
                        # （time.time() 分辨率 ~1ms），用严格比较会一直留着旧那条。
                        if best is None or seg_key(m) >= seg_key(best):
                            best = m
                            self._seg_addr = addr
                if best is not None and left <= 0:
                    self.saw_any = True
                    return best
                continue
            if best is not None:
                self.saw_any = True
                return best
            if left <= 0:
                return None

    def _from_old_sweep(self, seg):
        """这条公告是不是**上一轮自检的残留**（A 机连着重开一轮时会出现）。

        残留的高段号在 seg_key 里显得"更新"，不排掉就会一直追着上一轮的尾巴跑。
        """
        return bool(self._sweep_t) and float(seg.get("t") or 0.0) < self._sweep_t

    def ack(self, seg, **extra):
        """回一条回执给**这条公告的源地址**（A 机的日志里会显示"B 机已跟上"）。"""
        if self._seg_addr is None:
            return False
        body = dict(extra)
        body["kind"] = "ack"
        body["v"] = VERSION
        body["t"] = time.time()
        body["who"] = socket.gethostname()
        body["seg"] = seg.get("i")
        try:
            self.sock.sendto(json.dumps(body, ensure_ascii=False,
                                        separators=(",", ":")).encode("utf-8"),
                             self._seg_addr)
            return True
        except Exception:                    # noqa: BLE001
            return False

    def close(self):
        try:
            self.sock.close()
        except Exception:                    # noqa: BLE001
            pass
