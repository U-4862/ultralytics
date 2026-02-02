from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8n-pose.pt")
    model.train(
        data='coco8-pose.yaml',
        epochs=100,
        imgsz=640,
        )
