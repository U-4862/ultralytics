import cv2

from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")  # 模型路径
results = model.predict(
    source=0,
    stream=True,
)

for result in results:
    plotted = result.plot()
    cv2.imshow("YOLO inference", plotted)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
