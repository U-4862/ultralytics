from ultralytics import YOLO

model = YOLO(r"D:\Users\Desk\deeplearning\ultralytics-8.3.163\results\yolov5n6u_duantou_320\weights\best.pt")#训练好的train里的best文件
model.predict(
    source=0,#预测目标路径r""，改为0是预测摄像头
    save=False,#储存
    show=True,#实时显示
    #save_txt=True,#把预测结果保存为.txt
)