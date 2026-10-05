"""
YOLOv8 Real-Time Person and Object Detection Engine
===================================================
Provides fast, lightweight person and classroom object detection
(cell phone, book, laptop) with caching and bounding box conversion.
"""

import os
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    _YOLO_AVAILABLE = False

_yolo_model = None

def get_yolo_model():
    """Lazily load YOLOv8n model."""
    global _yolo_model
    if not _YOLO_AVAILABLE:
        return None
    if _yolo_model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "face_data", "models", "yolov8n.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "yolov8n.pt")
        if os.path.exists(model_path):
            try:
                _yolo_model = YOLO(model_path)
            except Exception as e:
                print(f"[warning] Failed to load YOLO model: {e}")
                _yolo_model = None
    return _yolo_model

def detect_objects_yolo(image_bgr, conf_threshold=0.35):
    """
    Detect persons and classroom objects using YOLOv8.
    Returns dict:
      {
        "persons": [ {"box": (x, y, w, h), "conf": float}, ... ],
        "phones": [ {"box": (x, y, w, h), "conf": float}, ... ],
        "books": [ {"box": (x, y, w, h), "conf": float}, ... ],
        "laptops": [ {"box": (x, y, w, h), "conf": float}, ... ],
        "all_objects": [ {"label": str, "box": (x, y, w, h), "conf": float}, ... ]
      }
    """
    model = get_yolo_model()
    if model is None:
        return {"persons": [], "phones": [], "books": [], "laptops": [], "all_objects": []}

    try:
        results = model(image_bgr, verbose=False, conf=conf_threshold)
        if not results:
            return {"persons": [], "phones": [], "books": [], "laptops": [], "all_objects": []}

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return {"persons": [], "phones": [], "books": [], "laptops": [], "all_objects": []}

        persons = []
        phones = []
        books = []
        laptops = []
        all_objs = []

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().astype(float)
        xyxy = boxes.xyxy.cpu().numpy().astype(int)

        names = model.names

        for i in range(len(cls_ids)):
            cid = cls_ids[i]
            label = names.get(cid, "object")
            score = float(confs[i])
            x1, y1, x2, y2 = xyxy[i]
            box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            obj_info = {"label": label, "box": box, "conf": round(score, 3)}
            all_objs.append(obj_info)

            if label == "person":
                persons.append(obj_info)
            elif label == "cell phone":
                phones.append(obj_info)
            elif label == "book":
                books.append(obj_info)
            elif label == "laptop":
                laptops.append(obj_info)

        return {
            "persons": persons,
            "phones": phones,
            "books": books,
            "laptops": laptops,
            "all_objects": all_objs
        }
    except Exception as e:
        print(f"[warning] YOLO inference error: {e}")
        return {"persons": [], "phones": [], "books": [], "laptops": [], "all_objects": []}
