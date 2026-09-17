"""数据集工作台 GUI（PyQt5）。

设计要点：
    界面只负责「输入参数 + 显示结果」，所有业务逻辑放在 tools/ 下，
    通过 TaskContext 交互。这样同一份逻辑 CLI 和 GUI 都能跑。
"""
