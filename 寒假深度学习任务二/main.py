from ultralytics import YOLO
import cv2

# 加载YOLO模型
model = YOLO("runs/detect/train4/weights/last.pt")
# model = YOLO("yolov8n.pt")

# 打开视频文件
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建输出视频的编码器和写入器
output_path = "output_video_1.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 遍历视频的每一帧进行目标检测和绘制边界框
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False)

    # 绘制边界框
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        confidences = result.boxes.conf.numpy()
        classes = result.boxes.cls.numpy()

        for box, confidence, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = box.astype(int)
            label = f"{model.names[int(cls)]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()