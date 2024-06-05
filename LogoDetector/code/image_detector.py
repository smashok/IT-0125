import os
from ultralytics import YOLO
import cv2
import numpy as np

def apply_blur(image, x1, y1, x2, y2, blur_factor=55):

    roi = image[int(y1):int(y2), int(x1):int(x2)]

    blurred_roi = cv2.GaussianBlur(roi, (blur_factor, blur_factor), 0)

    image[int(y1):int(y2), int(x1):int(x2)] = blurred_roi
    return image

def perform_object_detection(image_path, model_path='C:/Users/Anton/Desktop/LogoDetectionYOLOv8/code/runs/detect/train2/weights/last.pt', threshold=0.1, return_result_without_boxes=False):

    model = YOLO(model_path)


    frame = cv2.imread(image_path)
    H, W, _ = frame.shape
    original_frame = frame.copy()


    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            frame = apply_blur(frame, x1, y1, x2, y2)
            original_frame = apply_blur(original_frame, x1, y1, x2, y2)

    if return_result_without_boxes:
        return frame, original_frame
    else:
        return frame
