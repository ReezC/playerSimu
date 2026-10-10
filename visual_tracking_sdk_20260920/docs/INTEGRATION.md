# 接入接口及行为契约

## 1. 对方Agent的任务

保持接收方项目结构、版本号、输入系统和用户配置。先在独立目录运行测试GUI，再接实时截图和鼠标回调。不复制原主窗口、不改变无关业务、不按字符串版本号决定兼容性。

保留 `liescd/`、`liescc/`、`vision_sdk/` 的包名与相对目录。若接收方有同名包，应隔离进程或统一重命名所有内部导入；不要混用旧包和新包。模型路径可由 `weights=`显式指定。不要把本包的算法版本写成接收方应用版本。

本次主力包含V2.5.1分支候选空间调整，**不只是七个参数**。将整个追踪核心及必需依赖接入，才能获得当前实现；只套七参数不等价。

## 2. 数据流和尺度

```text
调用方截图（BGR、uint8、源时间、屏幕原点）
        ↓
Session：双锚点定位，最近20帧内3次稳定确认后固定ROI
        ↓
内容区最长边最多640（不放大小图），保留实际取整后宽高
        ↓
YOLO V4 + 高亮 → BYTE低分检测 → V2.5.1身份追踪
        ↓
输出平滑 → 按实际缩放比还原到截图坐标 → 加屏幕原点
        ↓
Observation → 调用方渲染/记录；MouseOutput → 调用方鼠标后端
```

`roi=(x,y,width,height)`相对于**输入截图**。不是屏幕绝对坐标，也不是x1/y1。
输入图不能包含人为绘制的检测框或轨迹；先process原图，再绘制。

令ROI原始尺寸为W,H，实际追踪图为w,h，追踪输出为u,v：

```text
aim_frame = (roi.x + u*W/w, roi.y + v*H/h)
aim_screen = (screen_origin.x + aim_frame.x,
              screen_origin.y + aim_frame.y)
```

例：ROI=(100,50,1280,720)，追踪图=640×360，点=(100,50)，截图屏幕原点=(-200,30)，最终屏幕点=(100,180)。支持负屏幕坐标；DPI逻辑坐标必须由调用方转成物理像素。

## 3. Session

```python
Session(weights=..., confidence=0.15, detector=None,
        success_detection=True, region='CN', max_duration_s=120.)
```

- `prepare()`：阻塞等待模型加载和预热；应在任务开场前完成。失败抛异常，不静默假装准备成功。
- `reset(roi=None)`：新会话清空身份、平滑、轨迹、时间、定位共识和成功计数。提供ROI可跳过定位；必须是边界内整数、宽高至少32。调用方同时重置MouseOutput。
- `process(frame_bgr,timestamp_s,screen_origin=(0,0))`：同步处理一帧；单一所有者串行调用，不支持同时由多个线程更新同一会话。实时输入必须为最新帧，不无限排队。
- `finish(reason='stopped')`：进入ENDED，后续不再输出瞄点，直到reset。
- `close()`：幂等关闭。关闭后不能reset/process；独占模型由会话释放，共享传入的detector由调用方释放。
- `with Session() as ...`：确保异常路径也关闭模型执行器。正常finish/reset可保留已预热模型，避免每轮重新加载。

时间戳必须有限且严格递增，同一帧不得重复计入。跨视频先reset。截图尺寸或屏幕原点改变会结束会话，调用方重新获取ROI，避免把旧坐标发送到新窗口。处理异常会终止本轮并向调用方抛出；调用方应停止鼠标输出并报告错误。

默认120秒包含定位阶段，超时返回ENDED。截图断流时没有process调用，也不会自行计时触发返回；调用方必须有自己的采集看门狗、停止逻辑和窗口有效性检查。

### Observation字段

| 字段 | 含义 |
|---|---|
| `timestamp_s` | 原始源帧时间 |
| `phase` | LOCATING、WAITING、LOCKED、COAST、LOST、SUCCESS_PENDING、SUCCESS、ENDED |
| `roi` | 截图内固定内容区 |
| `aim_frame` | 原始截图坐标；无可用点为None |
| `aim_screen` | 原始屏幕物理坐标；鼠标应使用此字段 |
| `result` | 原生TrackResult；其中框、aim_point都是缩小后的追踪坐标 |
| `tracking_frame` | 裁剪并缩放后的BGR图；定位/成功/结束时可以为None |
| `detections` | 用于可视化的主要检测框；BYTE低分框仍用于内部追踪 |
| `trail` | 追踪图坐标的输出轨迹；None表示断线；最多2000个元素 |
| `reason` | 结束或状态说明 |
| `session_id` | 每次reset递增，用来防止混发旧任务结果 |

`COAST`表示预测延续，不等于真实检测；`LOST`不代表任务完成。没有输出不能当作正确追踪，不能把上一点无限继续发送。

## 4. 渲染

```python
from liescd.gui import render_frame
if result.result is not None:
    image = render_frame(result.tracking_frame, result.detections,
                         result.result, result.trail, True)
```

该渲染器用于追踪图尺寸。若对方要画在全屏覆盖层，按第2节转换所有点和框；不要只给点加ROI偏移却漏掉比例还原。渲染器所在GUI模块需要PySide6；无GUI服务可以只使用Session返回值自行绘图。

## 5. 鼠标回调

```python
output = MouseOutput(move_mouse, max_age_s=0.2, release_grace_s=0.15)
output.manual(is_left_down, now=time.monotonic())
sent = output.emit(result, now=time.monotonic())
```

回调只接收屏幕整数x,y，必须短时返回，不应内部排队。SDK不点击、不按键、不加载驱动。调用方可接Windows、VM或现有远程输入后端。

拒绝：非LOCKED/COAST状态、无点、非有限坐标、未来/超过0.2秒的点、同一时间重复点、跨session点、手动按住及释放宽限、成功或结束状态。拒绝的点不会以后补发。超过0.2秒包含采集后的全部耗时，不只是YOLO推理时间。

必须在推理前后以及真正发出输入前读取人工接管状态；MouseOutput不装全局钩子。实时示例做了发出前的左键/ESC复检，但不能代替对方输入系统的独占仲裁。若人工按下松开都发生在两次采样间，纯轮询可能漏掉；对方有可靠事件钩子应直接调用manual。

一旦看到SUCCESS_PENDING/SUCCESS/ENDED，MouseOutput锁住本轮，避免成功界面还在跟随。确认新任务才reset；本适配器没有生产端8ms插值定时器，也不假装复刻原程序的输入调度。鼠标回调异常向上抛出，调用方负责停止和告警。

## 6. 成功与结束

共享当前CN/TW成功模板、0.86阈值及多尺度匹配。ROI锁定后15秒启用，最多2Hz扫描，在固定ROI对应外框加12%边距中匹配。首次命中返回SUCCESS_PENDING并停止输出，两次连续扫描命中返回SUCCESS并冻结本轮。成功面板确认按钮模板qr也随包提供，**本接口不自动点击按钮**。

这是简化的独立同步实现：成功匹配和追踪在同一个process内串行执行，不包含原程序的独立低频工作线程、点击后的消失确认或恢复挂机/休息。单次模板扫描可能增加帧耗时，MouseOutput仍执行新鲜度限制；不能把这里的实时调度宣称成生产链路逐帧等价。

如果自己的程序有成熟成功判断，可 `success_detection=False`，收到成功后先停止输出，再 `session.finish('host_confirmed_success')`，由自己处理按钮和后续流程。内容区录像可能根本没有成功面板，不应依赖模板自动结束；视频结束调用close。实时 `--content` 模式关闭成功检测，由ESC或120秒上限结束。

星形提示模板仅作为资源提供。调用方决定何时开始任务，本示例从用户启动命令时进入定位，不包含全天候自动触发、账号状态或挂机恢复逻辑。

## 7. 对齐与替换检查

| 默认参数 | 数值 |
|---|---:|
| motion_tolerance | 2.034259190993516 |
| opening_protection_seconds | 6.81722669232694 |
| collision_margin | 0.10004642158694585 |
| missing_margin | 0.21605278925691423 |
| supervisor_belief | 0.6298260137416737 |
| evidence_weight | 0.39805252395074353 |
| smoothing_strength | 5 |

这些默认值已经在核心里生效，不需要对方再次逐项抄写。GUI显示2.034不会丢失默认内部精度。V2.5.1还扩大了分支空间门限，核心源码一并包含。若改参数，先对相同素材做回归，不承诺其他项目上自动达到相同成绩。

保持V4模型、FP32/rect/batch1、最长边640、匹配度0.15、BYTE辅助检测、track_buffer=60及默认七参数。GPU/CPU和驱动变化可能带来检测数值差异。若需要同口径比较，使用相同观测逐帧比较状态、身份、框、原始/平滑点及断轨标记；有损录像重新推理不是原输入逐位重放。

窗口进入、人工接管、窗口移动、截图中断、失焦、成功确认、手动停止都应在对方环境实测。提供的离线/合成验证不是实际Windows游戏输入或VM接管验收。
