"""被控机（A 机）部署台。

    python -m deploy.app        （或双击仓库根的「被控机部署台.bat」）

把 docs/A_SETUP.md 里那几条手工命令做成按钮：时钟对时 / 时间码探针 /
键盘中继 / 屏幕推流。参数改一下立刻存进 config/deploy.json，下次打开原样复现。

**为什么单独一个包，不塞进 gui/**
    gui/ 是 B 机（工作机）的数据集工作台，跑在另一台机器上。这个部署台只装在
    A 机，装的东西都不一样（不需要 torch / ultralytics，只需要 PyQt5 + pyserial）。
    混在一起会让 A 机被迫装一堆用不到的依赖。

界面控件复用 gui/widgets.py 的 NoWheel* 与 gui/theme.py 的字号 —— 两个界面
长得一致、滚轮行为一致，但依赖关系只有「deploy → gui」，不会反过来。
"""
