# playerSimu

双机视觉自动化 + 反作弊研究平台。

- **A 机（仿真机）**：自研 2D 横版 ARPG（素材来自 WZ），OBS 推流，Pro Micro 注入 HID 输入
- **B 机（工作机，本项目主战场）**：收流解码 → YOLO 检测追踪 → World State → HFSM 决策 → 脚本段编码 → 下发

## 架构

```
A机: 仿真环境 ──OBS/UDP/SRT──> B机: FrameSource
                                      ↓
                              YOLO + Tracker
                                      ↓
                              World State
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

## 目录

- `link/` — 收流：统一 `FrameSource`（PyAV-UDP/SRT、文件回放）
- `perception/` — YOLO 训练、推理、追踪、World State
- `decision/` — HFSM + Utility AI
- `control/` — Actuator 抽象 + 指令协议
- `telemetry/` — 延迟标定与指标采集
- `tools/` — 自检、对时、录制、延迟探针
- `eval/` — 离线评估与反作弊研究
- `config/` — 配置
