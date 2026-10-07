"""玩家行为打点：把「角色做了什么、为什么停」按**事件**记成一行的日志。

**和 `core/perf.py` 的分工**（别混）
    `perf.log` 回答的是**性能**问题：每帧花多久、链路抖不抖、资源涨不涨 —— 它自己的
    说明里第一句就写了"这是给性能评估用的，**不是行为统计**"。
    而"这一趟任务为什么停""断在第几步"这类问题，混在那些统计块里既难读，也会把性能
    日志的性质带偏 ✗ ⇒ 单开这一份（用户 2026-09-27 要求）。

**范围（用户 2026-09-28 定，这就是分工线）**
    **所有「行为 / 状态」类打点都在这里**：任务下达、战斗、寻路、爬绳、**角色状态**、
    防掉线休息、定时行为……；而 `perf.log` **只管 GUI / 推流收流 / 性能** ✓。
    两类写法（都在本文件里，别再去 `perf`）：
      · `event(name, …)` —— **离散事件**（起跑 / 收工 / 失败 / 上绳 / 掉下来 / 打了怪…）；
      · `sample(name, value)` —— **每拍都在变的状态**（y 坐标、离绳距离、在不在绳上…）：
        **节流**记，不刷爆日志 ✓。⚠ **两类量要分开选**（选错 = 不节流 ✗）：
        离散状态（在不在绳上…）用默认参数（**值变了立刻记** ✓）；**连续量**（距离、坐标，
        **每拍都在变**）**必须传 `min_gap`** ⇒ 按「**定频**」记（两条之间至少隔 `min_gap`
        秒，不管值变不变 ✓）—— 实测 `chase_dist` / `climb_y` / `climb_dx` 就因为"每拍都
        在变 + 没给 `min_gap`"，把日志刷到 **760 KB** ✗（用户 2026-09-28 要求改 ✓）。

    事件名（`event()` 的第一个参数）举例：

    | 事件 | 什么时候 | 关键字段 |
    |---|---|---|
    | `task_start` | 一条多步路线**起跑**（「命令前往」/「定点休息」都走它）| `why` / `steps` |
    | `task_quick` | **瞬时任务**（起跑到收工 < 1 秒）⇒ 起跑 + 收工**合成这一条** | `why` / `secs` / `note` |
    | `task_step` | 某一步**走完**（接着走下一步）| `step` / `secs` / `note` |
    | `task_done` | **整条**走完（单步路线也走这里）| `steps` / `secs` / `note` |
    | `task_fail` | 某一步**失败** ⇒ 整条停掉 | `step` / `secs` / `left` / `reason` |
    | `task_cancel` | 被**叫停**（关自动 / 手动 / 休息结束…）| `why` / `step` / `secs` |
    | `task_plan_fail` | **压根没起跑**（解析不出路线）| `dst` / `why` |

    2026-09-28 起加了**任务内部变化**（用户："寻路的各类变化也做进行为打点"）——
    接线点只有 `agent._mark_route_change` 一处（`_climb_tick` 里 `job.update` 之后调 ✓）：

    | 事件 | 什么时候 | 关键字段 |
    |---|---|---|
    | `route_phase` | 任何任务的 `phase` 变了（align→climb、climb→done…）| `ladder` / `prev` / `to` / `note` |
    | `route_diag` | 爬绳的**斜跳**起跳 / 结束 | `ladder` / `what`（起跳 / 结束·吸上绳 / 结束·没吸上）/ `note` |
    | `route_retry` | 一次尝试**失败重来** | `ladder` / `attempt` / `note` |

    2026-09-28 起**范围扩到全部「行为 / 状态」**（用户："任务下达、战斗、寻路、角色状态等等
    打点应该在 behavior.log；perf.log 只管 GUI / 推流收流 / 性能"）—— 原来"借"在 `perf.log`
    里的那批（`route_*` / `climb_*` / `goto_*` / `replan*` / `afk_*` / `key_*` / `player_*` /
    `zone_*` / `mob_goto_*` / `jump` / `out_evade` / 战斗状态 `st_*` … 的**计数与曲线**）
    **全迁过来了**：离散的走 `event`、每拍状态走 `sample`（节流 ✓）。
    ⚠ 加新打点**一律用这两个**，别再去 `perf` ✗（`perf` 只留 GUI / 推流 / 性能指标 ✓）。

**格式**：一行一个事件，**人直接读得懂**（不用工具、不用解析）：
    2026-09-27 10:12:03.412  task_start   why=命令前往：左上平台  steps=3
    2026-09-27 10:12:19.052  task_step    step=1/3  secs=15.6  note=到了「右下休息平台」
    2026-09-27 10:12:44.900  task_fail    step=2/3  secs=25.8  left=1  reason=爬到 y=100 …
    字段是 `key=value`（key 一律用英文，好 grep；value 保留人话 ✓），值与值之间两个空格。

**开销**：`event` 只在**离散时刻**调（起跑 / 收工 / 上绳 / 掉下来…）⇒ 可忽略 ✓；
    `sample` 是每拍调的，靠**内部节流**（值变了或每 `min_sec` 秒才真记一条）把落盘量压住 ✓；
    关掉时两者都只剩一次 bool 判断（同 `perf` ✓）。
**落盘**：**一行一次**（故意不攒缓冲）—— 事件本来就稀疏（一趟任务几条），而"卡住前最后
    那几条"恰恰是最想看的；攒在内存里它们可能永远写不出去 ✗（2026-09-27 那次"卡了之后
    日志里什么都查不到"就是这类教训 ✓）。
    ⚠ **一个例外**（用户 2026-09-28 要求，代价已确认 ✓）：**任务起跑**那一行走
    `task_open` ⇒ **暂存最多 `QUICK_S`（1 秒）** 再落盘 —— 为的是把"**瞬时任务**"
    （起跑到收工 < 1 秒，占 `task_done` 的 ~45% ✗）**合并成一条** `task_quick` ✓：
      · 长任务（绝大多数）：1 秒后由 `event`/`sample` **顺手补写**（`_flush_pending` ✓），
        时间戳仍是**起跑那一刻** ✓；
      · 所以进程若在这 1 秒内挂掉，**最近那一条 `task_start` 会丢** ✗ —— 这条让步换来的
        是砍掉一半多的日志量 ✓。
    文件有上限（`MAX_BYTES`，超了砍掉前半截），不会无限长 ✓。
**线程**：和 `perf` 一样**不加锁** —— 调用点都在实时线程那一条（GUI 线程只读不写）。
    将来若有人从别的线程打点，先把锁补上 ✗（别默认它是线程安全的）。

日志：仓库根 `behavior.log`（和 perf.log / crash.log / stdout.log 一起，`*.log` 已 gitignore）。
"""

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "behavior.log"

# ⚠ **别调小**（教训同 `perf.MAX_BYTES`）：2026-09-28 起「行为 / 状态」打点**全在这里**
# （含每秒几条的 `sample`）⇒ 窗口太短的话，"出事那一段"会被裁掉、永远查不到 ✗。
MAX_BYTES = 8 * 1024 * 1024     # 日志超过它就把前半截掉（保留最近的一段历史）

#: 总开关：关着的时候每个事件都是「读一个 bool 就返回」。
#: 由实时线程按 `config/live.yaml` 的 `perf_log` 一起置位（和性能日志同一个开关 ——
#: 初版不单开一格参数，用户没要求；将来要分开就是加一格 + 两行调用 ✓）。
ENABLED = False

_NAME_W = 14            # 事件名对齐宽度（读起来整齐 ✓）


def set_enabled(on):
    """只翻开关（改设置时用，和 `perf.set_enabled` 同义）。"""
    global ENABLED
    ENABLED = bool(on)
    return ENABLED


def configure(on, log=None):
    """设置开关与日志路径（实时线程启动时调一次，和 `perf.configure` 同一处调 ✓）。"""
    global ENABLED, LOG, _pending
    ENABLED = bool(on)
    if log is not None:
        LOG = Path(log)
    _last_sample.clear()        # 换了日志/重开一段 ⇒ 采样基线也重来 ✓
    _pending = None             # 暂存的那条"任务起跑"也丢掉（换了日志 ⇒ 别写进新文件 ✗）
    return ENABLED


def event(name, message="", ts=None, **fields):
    """记一个事件。`name` 见模块说明那张表；`message` 是一句人话；`fields` 是 `key=value`。

    ⚠ **值里不要放换行**（一行一个事件是这份日志的全部可读性来源 ✗）：换行/连续空白会被
    压成单个空格（`_flatten` ✓），免得多出一行把日志读乱。
    `ts`：**指定时间戳**（默认现在 ✓）—— 只给 `task_open` 那条"补写"用（见它 ✓）；
    ⚠ 它是**保留参数**，字段（`**fields`）里不许再叫 `ts` ✗。
    """
    if not ENABLED:
        return
    _flush_pending()        # 顺手把该补的"任务起跑"补上（长任务最多晚 `QUICK_S` 秒落盘 ✓）
    now = time.time() if ts is None else float(ts)
    parts = []
    if message:
        parts.append(_flatten(message))
    for k in sorted(fields):
        v = fields[k]
        if v is None or v == "":
            continue                      # 缺的字段**不写**（别编一个数/空值 ✗）
        parts.append("%s=%s" % (k, _flatten(v)))
    _write("%s  %-*s %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                            + (".%03d" % int((now % 1) * 1000)),
                            _NAME_W, str(name), "  ".join(parts)))


#: 节流采样用：`name -> (上次记录时刻, 上次的值)`
_last_sample = {}


def sample(name, value, min_sec=1.0, min_gap=0.0):
    """**节流**记一个状态值（"每拍都在变"的量：y 坐标、离绳距离、在不在绳上…）。

    用户 2026-09-28 定：**行为/状态类的打点一律进这里**（`perf.log` 只管 GUI/推流/性能 ✓）。

    **两类节流，按量的性质选**（⚠ 选错了等于不节流 ✗）：
      · **离散状态**（在不在绳上、到哪一相、y 有没有读数…）⇒ 用默认参数：
        **值变了立刻记** ✓（跳变最要紧）；值没变则每 `min_sec` 秒一条心跳 ✓
        （心跳专门用来看"它一直停在这个值上" ✓）。
      · **连续量**（距离、坐标 —— **每拍都在变**）⇒ **必须传 `min_gap`（> 0）**：
        **两条之间至少隔它**，不管值变不变 ✓ —— 否则"值变了就记"= **每拍都记** ✗
        （30 拍/秒就是 30 行/秒）。实测 `chase_dist` / `climb_y` / `climb_dx` 就是这样把
        日志刷到 760 KB（用户 2026-09-28 要求改「**定频**」✓）；1.0 秒一个点，看趋势够用 ✓。
    `min_sec=0` / `min_gap=0` = 那一维不节流（仅调试用 ✓）。
    """
    if not ENABLED:
        return
    now = time.time()
    key = str(name)
    prev = _last_sample.get(key)
    if prev is not None:
        _dt = now - prev[0]
        if min_gap > 0 and _dt < float(min_gap):
            return                      # 定频：还没到下一个采样点 ⇒ 这一拍不记 ✓
        if prev[1] == value and _dt < float(min_sec):
            return                      # 值没变 ⇒ 按心跳间隔记 ✓
    _last_sample[key] = (now, value)
    event(key, v=value)


#: **短任务阈值（秒）**：起跑后在它以内就收工的（"下一条、完成一条"）⇒ 起跑 + 收工
#: **合成一条** `task_quick` ✓（用户 2026-09-28 要求 —— 实测这类占 `task_done` 的 ~45%、
#: 连上 `task_start` 一共吃掉一半多的日志量 ✗）。
#: ⚠ 它同时是"暂存最多挂多久"（见 `_flush_pending` ✓）—— 两处**必须同一个数** ✓。
QUICK_S = 1.0

#: 「**还没落盘的那一条任务起跑**」：`(ts, name, message, fields)` —— 见 `task_open` ✓
_pending = None


def _flush_pending():
    """暂存的"任务起跑"挂够 `QUICK_S` 了 ⇒ **按原时间戳补写** ✓。

    ⚠ 由 `event()` 每次顺手调 —— 实时线程每拍都在打点（`sample` ✓）⇒ 跑得久的任务最多
    **晚 `QUICK_S` 秒落盘** ✓（这是"合并瞬时任务"要付的代价，来龙去脉见 `task_open` ✓）。
    """
    global _pending
    if _pending is None:
        return
    ts, name, message, fields = _pending
    if (time.time() - ts) < QUICK_S:
        return
    _pending = None         # ⚠ **先清再写**：`event()` 里还会调回这里 ⇒ 不能无限递归 ✗
    event(name, message, ts=ts, **fields)


def task_open(name, message="", **fields):
    """一条任务**起跑**：**先暂存**（这一拍不落盘 ✗），等 `task_settle` 决定怎么写 ✓。

    用户 2026-09-28 要求："`secs < 1s` 的『下一条、完成一条』**合成一行**（`task_quick`）"
    ⇒ 起跑那一行**必须晚一点写**才合得起来 ✓。代价说清楚：
      · 暂存**最多挂 `QUICK_S`（1 秒）** ⇒ 跑得久的任务（绝大多数）会被 `event` 顺手补写
        （`_flush_pending` ✓）⇒ 只是"晚落盘 1 秒"，**格式与时间戳都不变** ✓；
      · ⚠ 代价：若进程在这 1 秒内挂掉，**最近那一条 `task_start` 会丢** ✗ —— 本模块原来
        那条"一行一次落盘、卡住前最后几条最想看"的纪律，**只对这一条让步** ✓
        （换来把一半多的日志量砍掉 ✓，用户已确认 ✓）。
    """
    global _pending
    if not ENABLED:
        return
    _flush_pending()                # 上一条万一还挂着（保险 ✓）
    _pending = (time.time(), str(name), message, dict(fields))


def task_settle(name, message="", secs=None, merge=None, **fields):
    """一条任务**收工**（`task_done` / `task_fail` / `task_cancel`）⇒ 决定怎么写 ✓。

    · 暂存**还在**（= 起跑后不到 `QUICK_S` 秒就收工 ⇒ **瞬时任务**）**且该合并** ⇒
      **合成一条** `task_quick`（带上起跑那句 `why` ✓，用户 2026-09-28 要求 ✓）；
    · 否则（长任务 / 暂存已被补写掉 / `merge=False`）⇒ 先把起跑那条**按原时间戳补上**、
      再写这一条 ✓（`event()` 开头会先 flush ⇒ 顺序天然正确 ✓）。
    `merge`：`None`（默认）= 按 `secs < QUICK_S` **自动判** ✓；**`False` = 强制不合并** ——
      `task_fail` / `task_cancel` 必须传它 ✗：那不是"走完了"，伪装成 `task_quick` 会误导 ✓。
    ⚠ `task_step`（还有下一步）**不结算**任务 ⇒ 直接 `event` 写它 ✓（见 `agent` 那边的调用）。
    """
    global _pending
    if not ENABLED:
        return
    pend, _pending = _pending, None
    if pend is not None:
        ts, _name, _msg, _f = pend
        _auto = (secs is not None and float(secs) < QUICK_S)
        if bool(_auto if merge is None else merge):
            event("task_quick", message, ts=ts, why=_f.get("why", ""),
                  secs=secs, **fields)      # ⭐ 瞬时任务：合成一行 ✓
            return
        event(_name, _msg, ts=ts, **_f)     # 长任务 / 失败 / 取消：起跑那条按原时间戳补上 ✓
    event(name, message, ts=None, secs=secs, **fields)


def _flatten(v):
    """把值弄成"能放进一行"的字符串（换行/连续空白压成单个空格 ✓）。"""
    return " ".join(str(v).split())


def _write(line):
    """把一行事件追加进文件（**一行一次落盘**，见模块说明里的取舍 ✓）。

    打点**永远不许**把主流程搞挂（和 `perf` 同一条纪律 ✓）：写不进去就当这一条没记。
    """
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        _trim()
    except Exception:                            # noqa: BLE001
        pass


#: ⭐ 上一次截断之后**记下的文件大小**（迟滞用；见 `_trim` ✓）。
_trim_seen = 0


def _trim():
    """文件太大就**只留尾巴**（用户 2026-10-06 ✓ 现场铁证 ✓）。

    ⛔⛔ **老写法每次打点都可能烧掉几十到几百毫秒** ✗（实测 ✓，`py-spy` 栈里逮到它 ✓）：
      它 `read_text()` **整个文件**、再 `write_text()` 整个后半截 ✗ ⇒ 现场
      `behavior.log` **14.67 MB**（`MAX_BYTES` 才 **8 MB** ✗）⇒ **每一条事件**都读 14 MB
      （读一遍 **41.5 ms** ✗）⇒ 实测 `event()` 平均 **23.3 ms**、最坏 **~460 ms** ✗
      ⇒ 每帧 3 条打点就是 **70 ms** ✗ ⇒ **帧率被它一个人按在 ~14 fps** ✓
      （用户现场："预览只有 10~13fps、整个窗口都发木"✓）。
    ⛔ 更糟的是它**静默失败** ✗（`except: pass` ✓ —— 文件被别处持有时 `write_text` 就会失败 ✓）
      ⇒ 文件**永远停在超限状态** ✓ ⇒ **每一条事件都重来一遍** ✓（现场那个死循环 ✓）。
    ⇒ 现在三件（都只影响"超限那一下" ✓ 平时的开销还是一行 append ✓）：
      ① **只读尾巴**（`seek` 到"要留多少"再读 ✓）⇒ 读的量与**保留量**成正比 ✓，与文件多大无关 ✓；
      ② **原子替换**（写临时文件 + `os.replace` ✓）⇒ 不会静默失败、也不会被并发 append 搅乱 ✓；
      ③ **迟滞**（`_trim_seen` ✓）：只在"上次截过之后又长了一截"时才再截 ✓
        ⇒ 不会一边截一边被自己的写入顶回去 ✓。
    """
    global _trim_seen
    try:
        size = LOG.stat().st_size
        if size <= MAX_BYTES:
            _trim_seen = size
            return
        if size < _trim_seen + MAX_BYTES // 8:
            return                      # 迟滞：还没明显长出来 ⇒ 这一条不截 ✓
        keep = MAX_BYTES // 2           # 留一半（同老口径 ✓）
        with open(LOG, "rb") as f:
            f.seek(max(0, size - keep))
            f.readline()                # 丢掉可能被腰斩的半行 ✓
            tail = f.read()
        tmp = LOG.with_name(LOG.name + ".trim")
        with open(tmp, "wb") as f:
            f.write(tail)
        tmp.replace(LOG)                # 原子换 ✓（`pathlib` 自带，省一个 import ✓）
        _trim_seen = len(tail)
    except Exception:                            # noqa: BLE001 —— 打点不许把主流程搞挂 ✓
        try:
            LOG.with_name(LOG.name + ".trim").unlink()
        except Exception:                        # noqa: BLE001
            pass
