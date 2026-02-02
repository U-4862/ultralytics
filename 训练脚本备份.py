from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")
    model.train(                #我们要开始训练
        data=r"coco8.yaml",     #打算训练coco8这个训练集
        epochs=30,              #训练**轮
        imgsz=640,              #全称image size，即图片尺寸
        batch=2,                #批
        cache=False,            #缓存
        workers=0,              #进程
    )