# UI 规范

> **动界面就必须遵守。** 这不是风格偏好，每条都是踩过坑写下来的 —— 我会写明
> 「违反了会怎样」，方便判断某条能不能有例外。
>
> 能自动查的条目已经写进脚本：
>
> ```
> python -m tools.check_ui          # 有错返回 1，可以挂到提交前
> python -m tools.check_ui --all    # 连提示一起看
> ```

---

## 1. 滚轮不许改参数（硬规则，没有例外）

**现象**：想滚页面，指针顺路划过参数框 —— 数字被改了，而且**没有任何提示**。
更隐蔽的是「聚焦残留」：点过一次数字框后焦点还留在那儿，之后再滚页面，
滚轮仍被当成「在调这个参数」。

**所以：所有参数控件都不响应滚轮 —— 但页面必须照滚。**

三条一起做：

| # | 做法 | 说明 |
|---|------|------|
| ① | 参数控件一律用 `gui.widgets` 里的 `NoWheelSpinBox` / `NoWheelDoubleSpinBox` / `NoWheelComboBox` / `NoWheelSlider` | 它们的 `wheelEvent` 会**代为滚动外层滚动区** → 参数不变、页面照滚 |
| ② | 应用启动时装一次全局守卫 `install_wheel_guard(app)`（已在 `gui/app.py`） | 兜底：谁忘了用 NoWheel*，滚轮也不会改值，同样代为滚页面 |
| ③ | `from gui.widgets import ...` **必须写在模块顶部** | 写在方法/类体里是局部名字，构造函数里用到就 `UnboundLocalError`（本仓库炸过一次） |

**不允许**：裸用 `QSpinBox` / `QDoubleSpinBox` / `QComboBox` / `QSlider`；
也不要「在 `wheelEvent` 里把值改回去」这种写法。

守卫还顺带管了 **标签栏**（`QTabBar` 滚轮会切页，同样是误触）。

### 实现要点：不要写 `ev.ignore()` 就完事

常见说法是「`wheelEvent` 里 `ignore()`，事件就冒泡给父级滚动区」。**这里不能靠它**：
那条链路依赖平台层怎么派发滚轮，细节不由我们控制 —— 实测无头环境下合成事件 +
`ignore()` 时外层 QScrollArea 纹丝不动（而视觉症状就是「指针压在参数框上页面不滚」）。

统一用 `gui.widgets.forward_wheel(ev, widget)`：它找**最近的外层
`QAbstractScrollArea`**，按「系统滚轮行数 × `singleStep`」直接驱动滚动条
（触摸板的 `pixelDelta` 按像素算，Shift + 滚轮 = 横向），内层滚到头就继续往外找，
嵌套滚动区也能用。转发成功要 `ev.accept()`，否则外层可能再滚一遍。

**QScrollBar 必须放行**：它也是 `QAbstractSlider` 的子类，被守卫吃掉的话，
指针压在最右边那条滚动条上时页面就不滚了。

**数字框的 `wheelEvent` 根本不会被调用**：`QAbstractSpinBox` 在 `event()` 里
就把滚轮忽略掉了（不是交给 `wheelEvent`），所以「只在 `wheelEvent` 里转发」
对 `QSpinBox` / `QDoubleSpinBox` 无效。真正兜住它的是 `_WheelGuard` ——
守卫在事件到达控件**之前**跑，对 `NoWheel*` 也代为转发一次（转成功就 accept，
免得再冒上去滚两遍）。所以守卫不能只盯着裸控件。

**控件在滚动区外面**（比如置顶常驻的组）：往上永远找不到滚动区，就在它的
容器上挂一个 `_wheel_target` 指向那个滚动区，`forward_wheel` 会认这个指路
（见 `gui/player_panel.py` 的「操控」组）。

> 检查项①③由 `tools/check_ui.py` 强制；②是兜底。

---

## 2. 新增一个参数的清单（最容易漏，一次漏一个坑）

| # | 做什么 | 漏了会怎样 |
|---|--------|-----------|
| 1 | `DecisionSettings.__init__` 加字段，**注释写清单位 + 0 的含义** | 下次没人知道 0 是关闭还是真的 0 |
| 2 | `to_dict` 加键 | 改完存不下去，重启就丢 |
| 3 | `from_dict` 读键（**要有默认值**） | 重启就丢 / 旧配置直接崩 |
| 4 | GUI 加控件 + `setToolTip` + `valueChanged` → 写回 `settings` 再 `save()` | 配了不生效 |
| 5 | 载入/切换时**用 `blockSignals` 回填控件值** | 回填触发 `valueChanged` → 把默认值写进配置，或者把配置覆盖成默认值 |
| 6 | 决定**存全局还是按项目**（HP/MP 条是按项目存的，见 `project.yaml` 的 `bars`） | 换个项目参数就串了 |
| 7 | 键名要改的话，**兼容读一次旧键** | 别人的旧模板/旧配置直接失效 |

> 「运行时状态、不持久化」的字段请显式写在 `__init__` 里那句
> `# 以下是运行时状态，不持久化（to_dict 不导出）：` 之后，或在字段注释里写上
> **不持久化** 三个字 —— `tools/check_ui.py` 靠这个标记区分，否则会误报。

---

## 3. 信号与回填

- **改了立刻 `settings.save()`**。本项目没有「应用」按钮，参数是即时生效的
  （字号除外，它走 `theme.apply` 立即生效但不用 save）。
- **回填一律 `blockSignals(True/False)` 包住**，不要用 `if value != xxx` 绕。
- **后台线程写、界面只读的量用轮询同步**（例：`_poll_auto_state` 定时读
  `settings.enabled` 刷新按钮状态）—— agent 在实时线程里也会改 `enabled`。
- 跨线程一律用 `pyqtSignal`，**不要在后台线程里直接 setText/setValue**。

---

## 4. 布局与分组

- **通用参数放通用组，策略专属参数放策略组并随策略显隐**
  （见 `_refresh_strategy_ui`：`sp_turn_cd` / `sp_back_range` 只在「扫平台」时可见）。
  不要把一个通用参数塞进某个策略的专属区 —— 换策略它就不见了，用户以为丢了。
- **代码顺序 = 视觉顺序**。要挪位置就整块代码搬，不要只在最后补一句
  `root.addWidget(...)`（那样下次改这组代码会找错地方）。
- 长面板放进 `QScrollArea`。
- 多列用 `QGridLayout` / `addStretch`，别靠固定宽度对齐 —— 缩放窗口就散了。

---

## 5. 线程

**任何会阻塞的事情都不许放主线程**：连网络、开串口、加载模型、等待设备回应。
做法：后台线程干活 → `pyqtSignal` 回主线程改 UI。

反例（本仓库真实踩过）：`_apply_input_device` 里直接连远程键盘，界面卡住几秒；
后来的做法是起线程 + `device_connected` 信号。
同理 `_reset_link`（重置指令通道）重连前先 `QApplication.processEvents()`
把「正在重连…」画出来，否则用户以为按钮没反应。

---

## 6. 文案与提示

- 标签用中文 + **单位**：`输出行为CD(ms)`、`喂宠间隔(min)`、`最大攻击距离`（像素）
- **参数控件的 tooltip 至少说清三件事**：
  1. 这是什么、单位是什么；
  2. **0 / 负值代表什么**（关闭？不限制？）；
  3. 调大调小会怎样（代价是什么）。
- 报错要写**怎么修**，不要只写「失败」：
  「找不到 Pro Micro 串口（确认已插入，或检查 config/link.yaml 的 serial_local）」
- 危险操作（删除、覆盖）要二次确认。

---

## 7. 自动检查项必须可信

只在 `tools/check_ui.py` 里留**确凿**的检查。一个「每次跑都报一堆、但每一条都
合理」的检查等于没有检查 —— 大家会习惯性忽略它的输出。

本仓库就砍掉过一条「不许硬编码颜色」的检查：内联样式表写颜色是本项目的既定
风格，扫出上百条，没有可执行性。**宁可少查，不可乱报。**

---

## 8. 提交前

```
python -m tools.check_ui        # 必须 exit=0
```

涉及界面的改动，另外建议跑一遍 `tools/gui_smoke.py`（能构造出各个面板就说明
没写崩），以及 `python -m tools.verify_ui_state`（如果改了实时预览相关的界面）。
