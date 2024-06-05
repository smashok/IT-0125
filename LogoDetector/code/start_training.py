from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")


# Use the model
model.train(data="config.yaml", epochs=30)
