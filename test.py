from ultralytics import YOLO

model = YOLO("yolo26n.pt")


if __name__ == "__main__":
    # print(model.task) # 检验模型存在
    
    # 使用摄像头进行实时预测：source=0
    model.predict(
        source=0,
        save=False,
        show=True,
        # line_width=2 # 预测框线
        # visualize=True # 特征图
    )
    
    # print(model.names) # 打印模型的类别名称
    print(sum(p.numel() for p in model.parameters()))
    print('done')