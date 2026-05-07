from ultralytics import YOLO
import os

# yolo model creation
model = YOLO(os.path.join(os.path.dirname(__file__), "yolo-weights/yolov8l.pt"))
model.train(data="coco128.yaml", imgsz=320, batch=4, epochs=20, workers=0)