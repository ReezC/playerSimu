"""A 机运行所需的东西：**一份清单** —— 装环境、启动器、环境自检、自检用例都照它。

**为什么单独列一处**：`python -m remote_kbd.gen_cert` 缺 `cryptography` 那次，
根因就是"入口只 import 了标准库 + 一个没列进依赖清单的包"。这类缺口只在**用的时候**
才暴露，而且现象常和缺的东西八竿子打不着（生成证书失败 / 键盘连不上 / 探针窗口起不来）。
所以装完之后要**当场验**，而不是等到现场。

**清单口径**：A 机上**会被启动的入口**（部署台五张卡 + 自检 + 安装时用到的工具）。
少列一个的代价 = 现场缺包；多列一个的代价 = 无 —— 所以宁可列全。

**bash 侧怎么用**：`被控机部署台.bat` 直接 `python -m deploy.deps`（退出码 2 = 有缺），
这样"装完了"和"真能用"是同一件事，而不是"我记得装过那三个包"。
"""

#: (模块名, 为什么需要它)。列的是 import 名（不是 pip 名）。
MODULES = (
    ("deploy.app", "部署台界面"),
    ("deploy.push_sweep", "推流自检（A 机侧）"),
    ("deploy.selfcheck", "部署台右上角那块环境自检"),
    ("remote_kbd.relay", "键盘中继（把按键转发给 Pro Micro 硬件键盘）"),
    ("remote_kbd.gen_cert", "键盘卡片上的「生成证书…」按钮"),
    ("tools.probe_gen", "屏幕时间码探针"),
    ("tools.clock_server", "时钟对时服务"),
    ("tools.minimap_push", "小地图推流（寻路定位时把框好的那一块推给 B 机）"),
    ("PyQt5", "界面（requirements.txt 里的 PyQt5）"),
    ("serial", "串口（requirements.txt 里的 pyserial）"),
    ("yaml", "配置读写（requirements.txt 里的 PyYAML）"),
    ("cryptography", "生成证书（requirements.txt 里也有）"),
    ("psutil", "推流自检的 CPU%（requirements.txt 里也有）"),
    ("tkinter", "探针窗口 —— **标准库**，但装 Python 时没勾 tcl/tk 就没有，"
                "pip 也装不了（得重跑安装器）"),
)


def check():
    """缺哪些模块 → [(模块, 为什么)]；都齐就是 []。**不抛异常。**"""
    import importlib

    missing = []
    for name, why in MODULES:
        try:
            importlib.import_module(name)
        except Exception:                        # noqa: BLE001
            missing.append((name, why))
    return missing


def main():
    miss = check()
    if not miss:
        print("A 机运行所需的 %d 个模块都在。" % len(MODULES))
        return 0
    print("缺 %d 个模块（装环境脚本装齐就不会有）：" % len(miss))
    for name, why in miss:
        print("    %-22s %s" % (name, why))
    print()
    print("先跑一次装环境脚本。tkinter 例外：它 pip 装不了，")
    print("要重跑 Python 安装器并勾上 tcl/tk。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
