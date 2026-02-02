import cv2
from ultralytics import YOLO

# 1. 加载你的模型 (请确保路径正确)
model = YOLO("runs/pose/train25/weights/best.pt")  # 替换为你自己的 .pt 模型路径

# 2. 初始化摄像头 (0 通常是默认摄像头)
cap = cv2.VideoCapture(0)

print("开始检测，按 'q' 键退出...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. 执行推理
        # 注意: 不要在这里使用 show=True，我们要自己控制显示以获取数据
        results = model(frame)

        # 4. 【关键步骤】提取检测参数
        # 遍历每一帧的检测结果
        for result in results:
            # 获取包含 xyxy (左上角x,左上角y,右下角x,右下角y), 置信度, 类别 的数组
            boxes = result.boxes.xyxy.cpu().numpy()      # 坐标
            confidences = result.boxes.conf.cpu().numpy() # 置信度
            classes = result.boxes.cls.cpu().numpy()      # 类别ID

            # 5. 实时“传出”参数 (这里以打印为例，你可以改为发送网络请求或保存文件)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i]) # 转为整数像素坐标
                conf = confidences[i]
                cls_id = int(classes[i])
                cls_name = model.names[cls_id] # 将ID转为名称 (如 'person', 'car')

                # ✅ 这里就是你实时获取到的数据！
                print(f"检测到: {cls_name} (ID:{cls_id}) | "
                      f"置信度: {conf:.2f} | "
                      f"位置: ({x1}, {y1}) 到 ({x2}, {y2})")

        # 6. 可视化 (可选，用于确认检测效果)
        annotated_frame = results[0].plot() # 自动画框和标签
        cv2.imshow("YOLO 实时检测 - 参数已输出至终端", annotated_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n检测已手动中断。")

finally:
    # 7. 释放资源
    cap.release()
    cv2.destroyAllWindows()