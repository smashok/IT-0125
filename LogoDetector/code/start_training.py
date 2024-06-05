from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch


# Use the model
model.train(data="config.yaml", epochs=30)  # train the model
