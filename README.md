# playerSimu

双机视觉自动化 + 反作弊研究平台。

- **A 机（仿真机）**：自研 2D 横版 ARPG（素材来自 WZ），OBS 推流，Pro Micro 注入 HID 输入
- **B 机（工作机，本项目主战场）**：收流解码 → YOLO 检测追踪 → World State → HFSM 决策 → 脚本段编码 → 下发

## 架构

```
A机: 仿真环境 ──OBS/UDP/SRT──> B机: FrameSource
                                      ↓
                              YOLO + Tracker + Platform Vision
                                      ↓
                       World State（玩家速度/腾空/平台顶边/落点）
                                      ↓
                              HFSM Decision (Intent)
                                      ↓
                              ActionScript (二进制 + CRC)
                                      ↓
                              Actuator ──> PrintActuator (当前)
                                           TcpBridgeActuator
                                           HardwareActuator
```

## 铁律

**A 机仿真环境只向外输出画面，绝不向 B 机泄漏任何内部状态。**
真值仅用于离线评估（落盘在 A 机，B 机不可读）。

## 快速开始

```powershell
pip install -r requirements.txt
# 先确认 A 机已在推流，然后：
python -m tools.probe_recv --show
```

A 机（被控机）那边：

```powershell
pip install -r deploy/requirements.txt     # 只有 PyQt5 + pyserial + PyYAML
python -m deploy.app                       # 对时 / 探针 / 键盘中继 / 推流，一键起
```

## 目录

- `link/` — 收流：统一 `FrameSource`（PyAV-UDP/SRT、文件回放）
- `perception/` — YOLO 训练、推理、追踪、World State
- `decision/` — HFSM + Utility AI
- `tools/` — 自检、对时、录制、延迟探针
- `gui/` — B 机：数据集工作台（GUI）
- `deploy/` — A 机：被控机部署台（GUI；配置在 `config/deploy.json`）
- `remote_kbd/` — A 机键盘中继 + Pro Micro 固件
- `config/` — 配置（`link.yaml` 双机共用）

## 工程约定：不许拍脑袋补数（2026-09-26 用户定）

**没有数据/依据支持时，不要自己发明逻辑；就算要"先估一个数"，也必须先经用户同意。**

为什么写进规范：`decision/route.py` 里曾经有一个 `ARRIVE_PAD = 14.0` —— 语义是
"爬到离绳顶还差 14 像素就算到达"。**用户从没提过这个需求**，那个 14 也没量过任何东西；
后果是角色**停在没踩上平台**的地方（实测停在 -204，而平台面是 -208，差 4px 不上去）。
这类"为了防某个毛病、随手加的一个容差"最阴 —— 看着像让程序更稳，实际是**悄悄改了行为**。

落地要求：

1. 判据只用**能指出数据来源**的量（地形/集合/标定里量出来的），别用"感觉差不多"的常数；
2. 确实需要容差/阈值时：**先说清它防的是什么、数据来源是什么、不生效会怎样**，
   等用户点头再写；
3. 已有的估数要能被审：放常量区、写清来历；**来历讲不清的，按 bug 处理**。

## 视觉预测与决策边界

**平台视觉识别已整块移除**（2026-09-26，用户确认不再需要）。原来实时线程会用 HSV
筛选 + 水平形态学从每帧提取平台碰撞顶边（`PlatformTracker` / `PlayerMotionTracker` /
`relate_terrain`），并由此衍生落点预测（`jump_prediction`）、"怪是否与玩家同平台"的
`mob.reachable` 过滤，以及实时画面上那些青色 `P###` 平台线 —— 现在**这一套全都不在了**
（`perception/platforms.py` 已删除）。

连带的**行为变化**：CombatAgent 不再按"平台可达"过滤怪 —— 视野里别的平台上、其实过不
去的怪也会被盯上（原来那道过滤没有依据了）。

仍然有效的纪律：**没标定过的视觉预测不许变成按键动作**。跨平台起跳要先拿实机视频标定
跳跃初速度和空中修正，再谈自动下发。
