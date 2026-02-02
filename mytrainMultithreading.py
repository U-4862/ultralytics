from ultralytics import YOLO

ms = [
    'yolo11n','yolo11m','yolo11l',
    'yolo11s',
]

if __name__ == "__main__":
    for m in ms:
        model = YOLO(m + ".pt")
        model.train(                #我们要开始训练
            data=r"duantou.yaml",     #打算训练***这个训练集
            epochs=3,              #训练**轮
            imgsz=640,              #全称image size，即图片尺寸
            batch=10,                #批
            cache="ram",            #缓存"ram"orFalse
            workers=2,              #进程
            #val=False,              #取消验证
            #device="cpu",           #用CPU进行训练
            project="results",
            name=m,
    )
