from ultralytics import YOLO

# # тренировка
# model = YOLO("yolov8n-seg.pt")
# result = model.train(data = "config.yaml", epochs=300)

# тестирование
model = YOLO("runs/segment/train5/weights/best.pt")
result = model.predict(source = 'image_20240816_215029_043858.png', conf = 0.1, save = True, hide_labels = True, boxes=False)