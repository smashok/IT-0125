import os
from ultralytics import YOLO
import cv2
import numpy as np

def apply_blur(image, x1, y1, x2, y2, blur_factor=55):
    # Extract the region of interest
    roi = image[int(y1):int(y2), int(x1):int(x2)]
    # Apply Gaussian blur to the region of interest
    blurred_roi = cv2.GaussianBlur(roi, (blur_factor, blur_factor), 0)
    # Replace the original region with the blurred one
    image[int(y1):int(y2), int(x1):int(x2)] = blurred_roi
    return image

def perform_object_detection(image_path, model_path='C:/Users/Anton/Desktop/LogoDetectionYOLOv8/code/runs/detect/train2/weights/last.pt', threshold=0.1, return_result_without_boxes=False):
    # Load a model
    model = YOLO(model_path)  # load a custom model

    # Read the input image
    frame = cv2.imread(image_path)
    H, W, _ = frame.shape
    original_frame = frame.copy()  # Создаем копию исходного изображения

    # Perform object detection
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            # Put text with the class name on the image
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            # Apply blur to the detected object
            frame = apply_blur(frame, x1, y1, x2, y2)
            original_frame = apply_blur(original_frame, x1, y1, x2, y2)  # Применяем размытие к копии без рамок

    if return_result_without_boxes:
        return frame, original_frame  # Возвращаем оба изображения
    else:
        return frame
