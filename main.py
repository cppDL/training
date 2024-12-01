from ultralytics import YOLO

# тренировка
model = YOLO("yolov8n-seg.pt")
result = model.train(data = "config.yaml", epochs=300)

# # тестирование
# model = YOLO("runs/segment/train2/weights/best.pt")
# result = model.predict(source = 'data/train/images/8.png', conf = 0.25, save = True, hide_labels = True, boxes=False)