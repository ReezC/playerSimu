# AI 结构：感知 / 决策 / 执行 + 编排（边界与「牙齿」）

**一句话**：这套 AI 名义上是三层（采集 → 识别 → 决策 → 执行），**问题不在分层本身** ✓ ——
而在**边界没有牙齿** ✗：同一事实两条链、每拍职责被「早退」吃掉、隐式契约只活在注释里 ✓。

**证据（2026-10-10 那轮连报 6 个 bug ✓）**：其中 **5 个**是上面这三条 ✗ ——
· 地形图自己用差分算玩家坐标（和信息栏两个数 ✗）；
· `ws.screen`（原生识别）被拿去当 Agent 的停手判据（于是测谎小游戏那段一「匹配不上」⇒
  Agent 立刻恢复按方向键 ✗）；
· 界面各自 `new PlayerLocator`（2026-10-06 那次 ✗）；
· `if st == self._screen_state: return` 把「按时间到点就该做」的判据挡在后面 ✗（同一处**三次** ✗）。

⇒ 本文记三件事：**① 现状（代码实际怎么分 ✓）② 四条不变量 + 它们的机械判据 ✓
③ 路线（按性价比排 ✓）**。

---

## 1. 现状：名义分层 vs 代码实际

层号（`tools/arch_scan.py::PKGS` ✓ 数越小越底层）：

```
core(0) < link / remote_kbd / perception / execution / liedetect(1) < decision(2) < gui / deploy(3) < tools(4)
```

| 层 | 名义职责 | 代码里**实际**装了什么 ✗ |
|---|---|---|
| `perception/` | 感知 | ✅ **2026-10-17 已分四组**（日志 472）：`detect/`(一帧→框：classes/predict/visible_box) · `track/`(框→id：tracker/bytetrack/assign) · `world/`(画面→坐标：world_state/pos_state/player_state/player_locator/anchor_track/region) · `screen/`(界面状态机：screen_state/ui_state)；顶层：`minimap.py`(**1679**，只管"量"：定位 / 来源 / 标定调用；日志 480-481 三刀后) / `map_dot.py`(1265，**单帧**认黄点 ✓) / `map_locator.py`(727，**跨帧**定位：PlayerDotTracker/PlayerLocator ✓) / `map_overlay.py`(380，叠图核对：画/差异/打分 ✓) / `map_calib.py`(604，换算与标定 ✓) / `map_const.py`(175 ✓) / `map_client.py`(300，收流客户端 ✓) / `vt_sdk.py` ＋ **`train/prepare_dataset/relabel/metrics`**（**离线训练工具** ✗ 阶段 4 归位）。`geom`/`kf` 已下沉 `core/`（纯数学件，`liedetect` 也吃 ✓） |
| `decision/` | 决策 | 策略 ＋ **`jobs/`（怎么走：爬/下跳/走/传送门/跳 五个执行器 ✓ 2026-10-18 从 `route.py` 拆出）** ＋ **`pathing/`（走哪条：世界坐标代价 → 选边 → 造任务 ✓）** ＋ `route.py` **已删** ✓。⏳ 仍待搬：`hotkey.py` / `manual_input.py` / `reconnect.py`（那是**执行 / 键盘鼠标** ✗） |
| `execution/` | 按键鼠标 | ✅ **2026-10-17 已建包并拆完 5 块**：`keys.py`(键位·零依赖 90) / `win32.py`(ctypes 胶水 26) / `pointer.py`(指针 125) / `input.py`(发键 245) / `remote.py`(链路连/断/重连 407) / `aim_convert.py`(换算·标定 929) / `mouse_aim.py`(瞄准动作 482) |
| `liedetect/` | 测谎 | ✅ **2026-10-17 已建包** ✓：`lie_controller` / `lie_group` / `lie_motion`(7239) / `lie_registry`(1277) / `lie_tracker`(5599) 从 `perception/` 搬来 ✓ 允许依赖 `perception.geom` / `perception.kf`（同层 ✓ 通用件 ✓） |
| 编排 | ？ | 全在 **`gui/live_thread.py`**（6558 行 ✗，2026-10-20 实测）—— 一帧的阶段顺序、限速闸、接管状态、发布 `ws`、录屏、perf 全在里面 ✓；**第 0~7 段已搬 `runtime/`** ✓（见下面"第 5 波"） |

实测（`python -m tools.arch_scan` ✓ **2026-10-17 重测** ✓）：
**分层违反 0 条** ✓（13 条里最后 3 条 `gui/deploy → tools` 也已清零 ✓）；**模块级循环依赖 0 个** ✓（好 ✓ ——
说明「谁 import 谁」大体是干净的 ✓）；**≥1500 行文件 25 个** ✗（含 `tools/` 自检脚本 ✓）
（`tools/selftest_decision.py` 23829 ✓、`decision/agent.py` 11326 ✓ 最大运行期文件 ✓、
`liedetect/lie_motion.py` 7239 ✓、`gui/live_thread.py` 5484 ✓…）。

**⚠ 两条新棘轮（2026-10-17 加，日志 467）——「模块级 0」并不等于边界干净**：

| 判据 | 现状 | 含义 |
|---|---|---|
| E-③ 函数内**向上**依赖 | **29 处**：`gui→tools` 17 · `deploy→tools` 10 · `perception→tools` 1 · `perception→gui` 1 | 写一行 import 就跨两层；这是模块级判据看不见的地方 ✗ |
| E-④ 单函数 ≥600 行 | **12 个**：`lie_demo.run_window` 3904 · `lie_motion.process` 2511 · `live_thread._run` **1860**（2026-10-20 实测，原 1868）· `player_panel._build` 1638 · `agent.tick` 1210 … | 拆块期间最容易复发的，就是"拆完又写一个超长函数" ✗ |

两条都按**棘轮**办（只许更少 ✓ 不许新增 ✗），且都做过**反向验证**（故意造一处向上依赖 / 一个 640 行函数 ⇒ 当场红 ✓）。

**重构进度（6 阶段，用户 2026-10-13 拍板 ✓）**：

| # | 阶段 | 状态 |
|---|---|---|
| 1 | 编排收口 → `runtime/pipeline.py` | ✅ 已落地（日志 371，`FrameCtx` / `HeavyGate` / `vt_*` 纯函数都在这里 ✓） |
| 2 | 执行层独立 → `execution/` | ✅ 已落地（2026-10-17，日志 465：`input.py` + `mouse_aim.py` 搬出 `decision/` ✓） |
| 3 | 测谎独立 → `liedetect/` | ✅ 已落地（2026-10-17，日志 466：5 个 `lie_*` 全部搬出 `perception/` ✓） |
| 4 | 离线训练工具归位 → `training/` | 🔶 部分（`data/*.py` 已搬 `tools/` ✓） |
| 5 | 目录卫生（根 `agent.py` / 日志归位） | ✅ 已落地（日志 372：根 `agent.py` 已删 ✓、`logs/` 已建 ✓） |
| 6 | 文档同步 | 🔄 本节即同步记录 ✓ |

**拆块波次（细账）**：第 0/1/2 波 ✓（棘轮 → `execution/` → `perception/` 四组）；
第 3 波 = `minimap.py` 瘦身（4264 → **1679** ✓ 已完成；⚠ **2026-10-20 实测回潮到 2233** ——
新加的 flow / relock / 近峰选择又都写回这一个文件了 ✗ 下次要么再切一刀、要么明确"显示区跟踪"是它的单一职责 ✓）：
`map_const` ✓ → `map_client` ✓ →
`map_calib` ✓ → **`map_dot` + `map_locator` ✓（日志 480）** → **`map_overlay` + `tools/mmap_diag`
✓（日志 481，命令行入口改 `python -m tools.mmap_diag --map <id>`）**。
**第 4 波 = `decision/route.py`（4995 行）拆分 ✓（日志 482）**：`decision/jobs/`（怎么走：五个执行器，
单向只认 `jobs.params/base/align`）+ `decision/pathing/`（走哪条：`spans → cost → plan`，
世界坐标为唯一计算坐标系），`route.py` 已删。

**第 5 波 = `_run`（1860 行）按阶段搬进 `runtime/`（2026-10-20 起 ✓ 日志 519）**：
用户在"还差什么"清单里点了"开始"⇒ 先搬最薄的两段 ✓
`runtime/frame_pic.py`（第 0 段：先留原生帧 + 趁没画裁小地图面板）+ `runtime/frame_ui.py`
（第 1 段：界面状态检测拍，限流 1 秒 ✓ 喂**原生帧** ✓）。顺序由
`selftest_live_panel::t_frame_stage_order` 钉住（留像素 → 取回 → 判界面 → 喂测谎 → 决策 → 画框 → 发原生帧 ✓
⚠ **画框在决策后面**，第一版判据写反了当场红 ✓）。
第 2 段 ✓ `runtime/frame_vt.py`（投一帧给测谎：两个开关 / 原点跟窗口 / 原样投原生帧 ✓
日志 520）；第 3 段 ✓ `runtime/frame_vis.py`（可视化配置热读 ✓ 日志 521）；
第 4 段 ✓ `runtime/frame_probe.py`（时间戳探针解码：四类账 + 延迟窗口 ✓ 日志 522）；
第 5 段 ✓ `runtime/frame_reconnect.py`（断线判断 / 自动重连的五个出口 ✓ 日志 524）；
第 6 段 ✓ `runtime/frame_infer.py`（一次推理 + 谁进决策：剪裁 / 偏移加回 / 四路分流 ✓ 日志 525）；
第 7 段 ✓ `runtime/frame_locate.py`（玩家定位：脚底偏移 / 框面积闸 / 黄点那份世界坐标 / 可信相机 ✓
日志 526）⇒ `_run` **1461 行**、`live_thread.py` 6558 行。
⚠ **第 7 段沿用第 6 段的教训**：它也是"真跑"用例（`t_frame_locate_stage`，83/83 ✓）——
  一搬就抓出 `n_boxes = k` 那个**跨段误用**（`k` 是第 6 段的产物，定位段里根本没有它 ✗）。
⚠⚠ **第 6 段的教训（比第 5 段更贵）**：一次搬 130 行 + 两个几何辅助，**源码钉一条都不会红**，
  但**真跑起来连炸四次**：裸 `self`（`getattr(self, "_vision_player_miss")` 里的，`.replace("self.", …)`
  换不到 ✓）、被改名的 `settings`、丢掉默认值的 `_cls_dets`、签名与函数体参数名不一致（`lie_mode`
  vs `_lie_mode`）—— 全都是 **NameError / UnboundLocal**，只在**第一帧推理**时炸 ✗。
  ⇒ **搬这种"带大量局部量"的段，必须配一条"真跑"用例**（用桩模型 + 桩数组，
  见 `t_frame_infer_stage` ✓），源码钉只能证明"字还在"，证不了"还能跑" ✗。
⚠⚠ **第 5 段的教训（最贵的一条）**：那 90 行原来坐在 `if self._infer.is_set():` 的缩进体里
（`=== 这里被遮住了 ===`）⇒ 搬之前必须**按缩进层**确认这一段的"同级兄弟"到哪结束，
别按注释标题切 ✗（第 3 段就是这么切出 `IndentationError` 的 ✓）。这一段还顺手暴露了
**`player_box`（追踪器刚返回）vs `player_box`（跟丢兜底）**的区别：传给状态机的必须是前者，
传兜底版 ⇒ 它永远不为 `None` ⇒ 状态机以为人还在 ⇒ 自动重连失灵 ✓。
⚠ **重新量过**（第一版按行数估的"测谎 280 行"是**注释撑起来的** ✗）：那一段里真正是语句的
只有投帧 7 行；剩下的大块是 `vis 配置热读` 与 `探针解码`（约 78 行）。
**第 3 段的值挂在线程上而不是进 `FrameCtx`**（用户 2026-10-20 选的 **B** ✓）：颜色/线宽/开关
是**跨帧活着**的（1 秒重读一次、中间每拍用上一份 ✓）⇒ 按 `FrameCtx` 自己的判据
"下一帧还会用吗？"⇒ 会 ⇒ 不进盒子 ✗。落地形态：`th._vis_ui`（`VisUI`）+ `th._vis_refresh_last`，
`_run` 里只留一列**本拍别名**（绘制端照旧读短名字 ⇒ 行为零改动 ✓）。
⚠ 顺带发现原来"可视化配置"和"推理参数（conf/iou/max_det）"**共用一把 1 秒闸** ⇒ 拆开后
各管各的（都还是 1 秒内生效 ✓），原来的 `_vis_refresh_last` 局部变量由 `th._vis_refresh_last` 接手、
新增 `_inf_refresh_last` 给推理参数那一半 ✓。
⚠ 搬之前先看：`selftest_live_panel` / `selftest_screen_state` 里有几条是**源码钉**
（数 `self._xxx` 出现次数、比全文下标）⇒ 必须改成**按文件**判 + 接线断言，
不然假红会逼着人把钉子删掉 ✗（这一轮已改两条：`t_raw_frame_before_draw` / `t_wiring_source_pins` ✓）。

⚠ **一个待定夺的发现** ✗：**根目录 `agent.py`（9293 行）** 是 `decision/agent.py` 的**旧副本** ✓
—— 全仓库**没有任何地方 import 根 `agent`** ✓（2026-10-10 实测 ✓）⇒ 它现在只是**import 遮蔽隐患**
（万一哪天在根目录跑东西、`import agent` 会拿到那份旧的 ✗）。要不要删，等用户一句话 ✓
（现在**只记录、不动它** ✗ —— 顺手删别人的大文件不是小事 ✓）。

---

## 2. 四条不变量（每条都有机械判据 ✓）

判据全在 **`tools/selftest_arch.py`**（8 条 ✓ **纯静态** ⇒ 不 import Qt/torch，毫秒级 ✓）。
⚠ 它**故意**只用 AST ✗ 不用文本查找 —— 本仓库注释密度极高，而注释里**照着抄出**被钉的代码是
常态（那正是「这块为什么这么写」的教材 ✓）⇒ 文本判据会被注释骗 ✗（**本次实测踩到** ✓：
D 那条第一版把注释里的代码当正文 ⇒ 误报 ✗）。

### R1 一个事实一个写者

| 事实 | 允许的实现写点 | 为什么必须唯一 |
|---|---|---|
| `LAST_POS` | `gui/live_thread.py` ×2（L2141 / L2205 ✓；声明在 `decision/agent.py` L1830 ✓） | 玩家坐标的**唯一来源**（状态行 + 地形图的圈 ✓）。多一个写者 ⇒「两处坐标对不上」✗（2026-10-06 实锤过 ✓） |
| `ws.screen` | `gui/live_thread.py` ×2（推理路 L5085 / 只收流路 L5237 ✓；默认值声明在 `perception/world_state.py` L141 ✓） | **Agent 的停手判据** ✓ 两处都必须取自 `_takeover_st` ✓（取值由 `selftest_screen_state` 钉 ✓） |
| `_screen_state` | `gui/live_thread.py` ×2（初值 L1238 + `_screen_beat` L3554 ✓） | **原生识别结果**（录屏 / 样本 / 状态行 / 报警读它 ✓）—— 不许拿去当 Agent 判据 ✗ |
| `_takeover_st` | `gui/live_thread.py` ×2（初值 L1247 + L3467 ✓） | **该报给 Agent 的那份**（留住接管 ✓）—— 多一个写者 =「谁说了算」又说不清 ✗ |
| `_lie_warn_at` | `gui/live_thread.py` ×4（全在 `_screen_beat` ✓） | 「**这一场测谎**」的账本 ✓ 录屏的停录判据与 Agent 的留住判据**共用**它 ✓ |
| `_lie_seen_game` | `gui/live_thread.py` ×4（全在同一个函数 ✓） | 同一本账的「这场进过小游戏吗」✓ ⚠ 2026-10-10 晚**口径变过**：现在它**只作打点证据** ✓（不再决定接管时长 ✓） |
| `_lie_sess_t0` | `gui/live_thread.py` ×2 | 同一场测谎的起点（`LIE_SESSION_MAX_S` 判超时 ✓）与录屏同一口径 ✓ |
| `_pipe_fps` / `_heavy_at` | `gui/live_thread.py` 3 / 2 | **重活闸**的档位与上次放行时刻 ✓（「现在到底限没限速」必须有唯一权威 ✓） |

**后门**：`setattr(self, "_screen_state", …)` 这种静态判据看不见 ✗ ⇒ 单列一条 B 钉住
（实测全运行期 **0 处** ✓）。

### R2 一帧一个顺序

`gui/live_thread.py` 里**唯一**的 `agent.tick(ws)` ✓；而**喂给 Agent 的 `ws` 必须在那之前装好** ✓
（`ws.screen` 的第一个写点排在 `agent.tick` **之前** ✓）。
否则 Agent 那拍读到的是**上一拍**的界面状态 ✗ —— 症状是「弹窗都出来了它还在打」✗ /
「弹窗关了它还停着」✗，而且**只在切换那一拍对不上** ⇒ 极难查 ✓。

### R3 每拍职责 vs 边沿职责

`_screen_beat` 里 `if st == self._screen_state: return` 是「状态没变就不往下走」的短路 ✓ ——
**按时间到点的判据**（留住接管「这一场还在不在」✓、录屏「待停到点没到」✓）**每秒都必须被评估** ✗
⇒ 一律排在早退**之前** ✓，且两者之间**不许夹 `return`** ✓，写 `_takeover_st` **只有一处** ✓。

⚠ 这条是最贵的：实测 `lie_success` 之后状态**再也不变** ⇒ 待停时刻**永远没人查** ✗
⇒ 录屏一路录到 300 秒保险才被切 ✗（那段录像 6819 帧 / 整 300 秒 ✓ 与推论完全一致 ✓）。

### R4 分层只有「冻结基线」，没有「例外」

· 层违反基线 **3 条**（`deploy.push_sweep → tools` ✓、`gui.live_thread → tools` ✓、
  `gui.live_thread → tools.probe_codec` ✓）—— **只许更少** ✓；
· **循环依赖必须保持 0** ✓（一有环，「层」就没有意义了 ✗）；
· 巨型运行期文件基线 **16 个** ✓（含上面那个根目录 stray ✓）—— **不许再多一个** ✗
  （每个新巨物都该先问：它是不是又混了两件事 ✗）。

⚠ 要往基线里加，必须**在同一个提交里**改 `tools/selftest_arch.py` 的名单 **并**写清理由 ✓
—— 那是「有意的门槛」✓ 不是「顺手放行」✗。

---

## 3. 本次落地

| 文件 | 做了什么 |
|---|---|
| `tools/arch_scan.py` | 从「只会打印的报告器」改成**可复用引擎** ✓（`scan()` 返回结构化结果 ✓ 人看报告 / 钉子拦截**同一套判据** ✓ 不再两套 ✗）；顺手修**两个真缺陷** ✗：① venv 过滤用的是**精确相等**（`.venv` / 各子项目 `.venv` 全漏进来 ⇒ 1951 行输出里绝大多数是 torch 自己的文件 ✗）；② 路径返回**反斜杠**（钉子的基线名单写的是 `/` ⇒ 一比较全不相等 ✗ 实测把 26 个文件全报成「新巨型」✗） |
| `tools/selftest_arch.py` | **新增 8 条钉子** ✓（A-① / A-② / A-③ / B / C / D / E-① / E-② ✓）；`--list` 可列各事实的写点 ✓（复核 / 写文档用 ✓） |
| **反向验证** | 三处**故意做坏**（坐标写点数量改错 ✗ / 留住判据与早退之间塞一句死代码 `return` ✗ / 从基线里拿掉一条既有违例 ✗）⇒ **三条一起当场红** ✓（含精确到行号的报错 ✓），随后全撤 ✓ |

跑法：

```bash
python -X utf8 -m tools.selftest_arch          # 8 条，全过 0 / 有失败 1
python -X utf8 -m tools.selftest_arch --list   # 只看写点（复核用）
python -X utf8 -m tools.arch_scan              # 人看的完整报告（含懒加载 / 私有符号外借 ✓）
```

---

## 4. 路线（按性价比排 ✓）

| 顺序 | 做什么 | 状态 / 风险 |
|---|---|---|
| **1** | **只加约束、不动结构**：R1 / R3 做成判据 ✓（本轮 ✓） | **已完成** ✓ 零行为改动 ✓ |
| 2 | 把 `_screen_beat` 拆成 `on_transition( st )` ＋ `every_tick( now )` 两个口 ✓（从**语法上**消灭 R3 那类坑 ✓）；`lie_*` 从 `perception/` 挪到「行为」那侧 ✓（它确实在下决策 ✗） | 小改动 ✓ 边界立刻清楚 ✓ |
| 3 | 从 `live_thread` 抽出 `runtime/pipeline.py`（一帧的阶段顺序 + 限速闸 + 接管状态 ✓），`live_thread` 只留 Qt 胶水 ✓；`ws` 收成**不可变快照**（写者唯一由类型保证 ✓） | 中改动 ✓；之后「滞后一拍 / 闸 / 接管」才有地方统一管 ✓ |
| 4 | `agent.py` 按职责拆（目标选择 / 输出序列 / 区域策略 / idle 策略 / 路由胶水 ✓） | **先别做** ✗：现有用例大多是**源码钉子** ✓，直接拆会全红 ⇒ 先补**行为测试** ✓ |
| 5 | 观测：结构化事件 + run id + 轮转（`behavior.log` 现在**每轮重写** ✗）+ 每条重要决策记「配置来源 + 分支」 | 便宜、长期收益大 ✓ |

---

## 5. 相关

· `tools/selftest_live_panel.py`：**真跑**的那几条（重活闸跳帧要补喂时间拍 ✓ / 解出来没人要就不做
  颜色转换 ✓ / 测谎期间一直接管 ✓）；
· `tools/selftest_screen_state.py`：两路 `ws` 必须带**接管版**界面状态 ✓；
· `docs/开发日志.md`：每条结论的现场来龙去脉 ✓；
· `.codebuddy/skills/playersimu-workbench/SKILL.md`：编号纪律（踩过就加一条 ✓）。
