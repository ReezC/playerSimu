# A 机（仿真机 192.168.1.8）配置手册

> 本文件所列操作全部在 **A 机** 上执行。B 机侧代码在仓库根目录。

---

## 0. 前置：填 B 机 IP

在 **B 机** 执行 `ipconfig`，拿到 IPv4 地址，填到 `config/link.yaml` 的 `b_host`。

---

## 1. 放行进站 UDP（管理员 PowerShell）

```powershell
netsh advfirewall firewall add rule name="playerSimu clock 5001" dir=in action=allow protocol=UDP localport=5001
```

---

## 2. 启动时钟对时服务（必须先跑，B 机延迟统计依赖它）

```powershell
cd <仓库路径>
python -m tools.clock_server --port 5001
```

保持窗口常开。

---

## 3. 启动屏幕时间码探针

```powershell
python -m tools.probe_gen
```

会在屏幕顶部出现一条去边框置顶的黑白方块条。
坐标默认 `(100, 8)`，需与 `config/link.yaml` 的 `probe.x / probe.y` 一致。
**如果 A 机屏幕顶部有其他 UI 遮挡，改坐标并同步改配置。**

---

## 4. OBS 推流配置

### 4.1 采集源
- 来源 → `+` → **显示器采集**（DXGI，不注入游戏进程，符合"零痕迹"原则）
- **不要**用"游戏采集"（会 hook 进程）
- 仿真环境请设为**无边框全屏**并固定 `1280x720`（独占全屏可能导致采集黑屏）

### 4.2 输出（设置 → 输出 → 输出模式：高级 → 录制页）
- 类型：**自定义输出 (FFmpeg)**
- FFmpeg 输出类型：**输出到 URL**
- 文件路径或 URL：`udp://<B机IP>:5000?pkt_size=1316`
- 容器格式：`mpegts`
- 视频编码器：`h264_nvenc`（无 N 卡则 `libx264`）
- 视频编码器设置（若可填）：
  ```
  preset=p1 tune=ull rc=cbr bitrate=12000M ... 
  ```
  简化写法：`-b:v 12M -maxrate 12M -bufsize 2M -g 60 -bf 0 -pix_fmt yuv420p`

### 4.3 视频（设置 → 视频）
- 输出（缩放）分辨率：`1280x720`
- 帧率：`60`（与仿真环境一致，避免重复/丢帧）

### 4.4 高级
- 渲染器：Direct3D 11
- 色彩格式：NV12 / 色彩空间 709 / 色彩范围 **部分**（与编码器一致，避免画面发灰）

---

## 5. FFmpeg 备选（不装 OBS 时的最小验证）

```powershell
# 有 N 卡
ffmpeg -f gdigrab -framerate 60 -video_size 1280x720 -i desktop ^
  -c:v h264_nvenc -preset p1 -tune ull -delay 0 -zerolatency 1 ^
  -b:v 12M -maxrate 12M -bufsize 2M -g 60 -bf 0 -pix_fmt yuv420p ^
  -f mpegts udp://<B机IP>:5000?pkt_size=1316

# 无 N 卡
ffmpeg -f gdigrab -framerate 60 -video_size 1280x720 -i desktop ^
  -c:v libx264 -preset ultrafast -tune zerolatency -b:v 10M -g 60 -bf 0 -pix_fmt yuv420p ^
  -f mpegts udp://<B机IP>:5000?pkt_size=1316
```

---

## 6. 自检 Checklist

- [ ] B 机 `python -m tools.clock_sync --host 192.168.1.8 --save` 能拿到 offset（jitter < 1ms 为佳）
- [ ] B 机 `python -m tools.probe_recv --show` 能看到画面、fps 接近 60
- [ ] 画面里能识别到绿色探针框，且显示 `lat xx.x ms`
- [ ] `python -m tools.record --seconds 30` 能录出可回放的文件
- [ ] B 机 `python -m tools.probe_recv --file data/recordings/xxx.mkv` 回放正常

---

## 7. 后续（P3 起需要）

- [ ] 仿真环境：确定性种子、固定步长 60Hz、真值记录仪（parquet 落盘，**不向 B 机开放**）
- [ ] 仿真环境：数据集导出模式（批量出 `png + YOLO txt`）
- [ ] WZ 资源解析：怪物/角色/地图/技能 sprite → 引擎素材
- [ ] `HID Bridge`：监听 TCP，把 B 机指令转发给 Pro Micro（硬件到货后）
