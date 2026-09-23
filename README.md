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

## 平台视觉与决策边界

实时线程会用 HSV 颜色筛选与水平形态学从每帧提取平台碰撞顶边，输出
`Platform(id, x1, x2, y)`；`PlatformTracker` 负责保持跨帧 ID。玩家检测框的连续
位置会补齐 `vx`、`vy`、`grounded`、`jumping`、`falling`、`current_platform_id`，下落时
还会给出 `jump_prediction`（当前速度下的预测落点）。这些字段都在 `WorldState` 中。

当前 CombatAgent 会在已识别出当前平台时，仅选择同一平台上的怪物；平台短暂失检时
自动退化到原有的视野过滤逻辑。跨平台起跳不会自动下发，必须先用实机视频标定跳跃
初速度和空中修正，避免把尚未校准的视觉预测变成按键动作。
