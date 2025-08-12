from typing import Dict, List, Tuple
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        results = self.model(frame, imgsz=640, conf=self.conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                'class_id': int(box.cls),
                'class_name': self.class_names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': [x1, y1, x2, y2],
                'area': (x2 - x1) * (y2 - y1)
            })
        return annotated_frame, detections