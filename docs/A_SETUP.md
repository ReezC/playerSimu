# A 机（仿真机 192.168.1.8）配置手册

> 本文件所列操作全部在 **A 机** 上执行。B 机侧代码在仓库根目录。

---

## 0. 前置：填 B 机 IP

在 **B 机** 执行 `ipconfig`，拿到 IPv4 地址，填到 `config/link.yaml` 的 `b_host`。

---

## 1.5 部署台（GUI，推荐）

上面这些命令（**对时 / 探针 / 键盘中继 / 推流**）都做成了按钮，装在 A 机上直接跑。

装依赖（在**仓库根**做一次）：

**省事的做法**：把整个仓库拷到 A 机，在仓库根**双击「被控机部署台（装环境）.bat」** ——
它会找 Python、建 `venv`、装这三个包、验证导入，最后把还差的东西一次列出来。
（等价的命令行在下面，脚本起不来或想看每一步时用。）

```powershell
# 1) 建一个 venv 并装依赖
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\pip install -r deploy\requirements.txt   # 只有 PyQt5 + pyserial + PyYAML

# 2) 启动（双击仓库根的「被控机部署台.bat」也行）
.venv\Scripts\python -m deploy.app
```

**前置**：A 机要有 Python 3.10+（`winget install -e --id Python.Python.3.10`，
或官网安装包勾上「Add python.exe to PATH」）。装环境的脚本会检查这一条。

**依赖装进哪个解释器**：装进「启动部署台的那个」就够了 —— 四个服务都由部署台用
**同一个**解释器拉起（`deploy/services.py` 的 `python_exe()` = `sys.executable`）。
用 venv 还顺带解决另一件事：`被控机部署台.bat` 会**优先**用 `.venv\Scripts\pythonw.exe`，
不会撞上系统里那个没装 PyQt5 的 python。

**装这些就够了**：根目录的 `requirements.txt`（av / opencv / torch 那套）是 B 机的；
`remote_kbd/requirements.txt` 里的 `cryptography` 只在**重新生成 TLS 证书**时才要
（证书已随仓库放在 `remote_kbd/certs/`）。

pip 之外还要的：

| 要什么 | 谁需要 | 怎么来 |
|---|---|---|
| **ffmpeg** | 推流 | `winget install --id Gyan.FFmpeg -e`；没 N 卡就把推流的编码器改成 `libx264` |
| `tkinter` | 探针窗口（标准库） | 精简安装包可能没带；缺了会在探针卡片的日志里看到 `No module named 'tkinter'` |
| **OBS** | 只在走 §4 的 OBS 推流时 | 用部署台的推流卡片则不需要 |
| **Pro Micro 固件** | 键盘中继 | Arduino IDE 烧 `remote_kbd/pro_micro/pro_micro.ino` —— **烧之前先在部署台停掉「键盘中继」**！否则板子重新枚举时，中继手里的串口句柄会失效，可能直接原生崩掉（`0xC0000005`），表现是「界面还显示连接成功，但手动输入和鼠标都传不过去」 |
| 放行入站 UDP 5001 | 对时服务 | 见 §1 |

装完点界面里的**环境自检**，它一次查完：ffmpeg / 有没有 NVENC / 串口能不能枚举 /
B 机通不通 / TLS 证书在不在 / 参数和 `config/link.yaml` 对不对得上。

**起不来时怎么查**：「被控机部署台.bat」会先检查上面这三个包，缺了会直接说装哪条
命令；界面自己起不来（例如 Qt 出问题）会**弹一个错误框**并把完整堆栈写进仓库根的
`deploy_crash.log`。想看原始报错就用带控制台的方式跑：`python -m deploy.app`
（或双击「被控机部署台（调试）.bat」）。

界面里有什么：

- **四张服务卡片**（顺序就是部署顺序）：参数、启动/停止、状态、已运行时长
- **环境自检**：ffmpeg 在不在 / 有没有 NVENC / 串口列不列得出来 / B 机通不通 /
  证书在不在 / **参数和 `config/link.yaml` 对不对得上**
  （最后一条最省钱：不一致时的现象是「流发出去了一切正常，B 机就是没画面」）
- **运行日志**：四个服务的输出混在一起，可以按服务筛、可以存文件
- **每张卡片底部写着「将执行」的完整命令行**，可复制 —— 界面上做的任何事都等价于
  手敲那条命令，想手工排查时直接贴到 PowerShell 里跑

参数改一下**立刻存进 `config/deploy.json`**，下次打开原样复现（包括窗口大小）。
推流分辨率 / 帧率 / 端口 / 码率 / 采集方式 / 编码器、串口号、对时端口都是可选项，
默认值从 `config/link.yaml` 推出来 —— 两边不一致时自检会明说，并给一个
「按 link.yaml 填」按钮一键对齐。

下面第 2、3、4、5 节是**等价的手工做法**，留着备用（比如部署台本身起不来时）。

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

---

## 附：挂机久了推流自己断（本机到 B 机「网络不可达」）

**现象**：ffmpeg 报 `Error number -10051`（Windows 的 `WSAENETUNREACH` = 网络不可达），
而且通常是**挂了一段时间**才报。

**怎么读这个错**：它是**本机**的路由错误，不是 B 机拒绝你。所以分两种情况：

- **一启动就报** → 网段/路由没配通：查 A 机自己的 IP 是不是和 B 机同网段
  （`ipconfig /all`；`169.254.x.x` = 没拿到 IP）、B 机现在的 IP 还是配置里那个吗；
  确实不同网段就改 `config/link.yaml` 的 `b_host`，再点部署台的「按 link.yaml 填」。
- **挂了一阵才报** → 配置本来是对的，是**运行中途路由丢了**。挂机时没人碰键鼠，
  最常见的就是下面这三条。

先确认：

```powershell
powercfg /a        # 看有没有启用「待机 / 现代待机（S0）」
```

事件查看器 → Windows 日志 → 系统：找 **Kernel-Power 42**（进入睡眠）/ NDIS 相关事件（网卡断开）。

再关掉这些（**管理员** PowerShell，`-ac` = 接通电源的档位）：

```powershell
powercfg /change standby-timeout-ac 0      # 永不睡眠（这条最关键）
powercfg /change hibernate-timeout-ac 0    # 永不休眠
powercfg /change disk-timeout-ac 0         # 硬盘不停转
powercfg /change monitor-timeout-ac 0      # 显示器不关 —— 关屏会让采集画面变黑

# 关「USB 选择性暂停」（USB 网卡 / 采集设备）
powercfg /setacvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0
powercfg /setactive SCHEME_CURRENT
```

网卡省电（设备管理器 → 网络适配器 → 属性 → 电源管理 → 取消「允许计算机关闭此设备
以节约电源」），或者用 PowerShell：

```powershell
Get-NetAdapter                                        # 先看网卡叫什么
Disable-NetAdapterPowerManagement -Name "以太网"       # 名字换成你自己的
```

笔记本还要管电池档位和合盖：`powercfg /change standby-timeout-dc 0`；
控制面板 → 电源选项 → 选择关闭盖子的功能 → 都选「不采取任何操作」。

**另外**：Windows 更新自动重启、驱动重装也会让流中断 —— 报错前如果有重启记录，
那是另一回事（事件查看器里 `Kernel-Boot` / `WindowsUpdateClient`）。
