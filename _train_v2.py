# -*- coding: utf-8 -*-
"""CLI 训练 v2（2026-09-30 用户委托）：微调 yolo26n12 → datasets/yolo（792 帧标准布局）。
参数与工作台上一轮一致（epochs=120 / imgsz=960 / batch=8）。

⭐ 必须有 `if __name__ == "__main__"` 保护（Windows ✗）：dataloader workers=8 的
   子进程会 re-import 本模块 —— 没保护 = 每个 worker 又递归启动一轮训练 ✗✗
   （ultralytics 的 freeze_support 警告就是说的这个，实测撞过 ✓）。
"""
from ultralytics import YOLO


def main():
    m = YOLO(r"E:\MyPrograms\playerSimu\datasets\train\runs\yolo26n12\weights\best.pt")
    m.train(
        data=r"E:\MyPrograms\playerSimu\datasets\yolo\data.yaml",
        epochs=120,
        imgsz=960,
        batch=8,
        project=r"E:\MyPrograms\playerSimu\datasets\runs\detect",
        name="v2",
    )
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
