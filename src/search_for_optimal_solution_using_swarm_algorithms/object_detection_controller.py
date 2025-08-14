import cv2
import time
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Optional, Any
from camera_real import CameraReal
from yolo_detector import YOLODetector


class ObjectDetectionController:
    def __init__(self,
                 pioneer: Any,
                 camera_source: str|int = 0,
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.5,
                 min_detections: int = 5,
                 buffer_size: int = 10):
        """
        Инициализация контроллера детекции объектов (без потоков)
        """
        self.pioneer = pioneer
        self.min_detections = min_detections
        self.buffer_size = buffer_size
        
        # Инициализация компонентов
        self.camera = CameraReal(source=camera_source)
        self.detector = YOLODetector(model_path=model_path, conf_threshold=conf_threshold)
        self.class_names = self.detector.class_names
        
        # Буферы и статистика
        self._detection_buffer = deque(maxlen=buffer_size)
        self._confirmed_objects = []
        self._last_annotated_frame = None
        self._stats = {
            'processed_frames': 0,
            'fps': 0.0,
            'last_frame_time': time.time()
        }
        self._last_detection_time = 0
        self._frame_count = 0
        self._fps_last_time = time.time()

    def process_frame(self, detection_interval: float = 0.1):
        """
        Обрабатывает один кадр из видеопотока
        :param detection_interval: минимальный интервал между обработкой кадров (секунды)
        """
        current_time = time.time()
        
        # Пропускаем кадр если интервал слишком мал
        if current_time - self._last_detection_time < detection_interval:
            return
        
        self._last_detection_time = current_time
        
        try:
            # Получаем кадр
            frame = self.camera.get_cv_frame()
            if frame is None:
                return
                
            # Детекция объектов
            annotated_frame, detections = self.detector.detect(frame)
            self._last_annotated_frame = annotated_frame
            
            # Обновление буфера детекций
            self._update_detection_buffer(detections)
            
            # Анализ буфера для подтверждения объектов
            self._analyze_detections()
            
            # Обновление статистики FPS
            self._frame_count += 1
            if current_time - self._fps_last_time >= 1.0:
                self._stats['fps'] = self._frame_count / (current_time - self._fps_last_time)
                self._stats['processed_frames'] += self._frame_count
                self._frame_count = 0
                self._fps_last_time = current_time
                
        except Exception as e:
            print(f"Ошибка обработки кадра: {str(e)}")

    def get_confirmed_objects(self) -> List[str]:
        """Возвращает список подтвержденных объектов"""
        return self._confirmed_objects.copy()

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Возвращает последний обработанный кадр"""
        return self._last_annotated_frame

    def get_stats(self) -> Dict:
        """Возвращает статистику детекции"""
        return self._stats.copy()

    def show_detection(self):
        """Отображает окно с результатами детекции"""
        frame = self._last_annotated_frame
        if frame is not None:
            # Добавляем информацию о подтвержденных объектах
            stats = self.get_stats()
            cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self._confirmed_objects:
                cv2.putText(frame, "Confirmed:", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                for i, obj in enumerate(self._confirmed_objects):
                    cv2.putText(frame, f"- {obj}", (20, 90 + i*30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            cv2.imshow('Object Detection', frame)
            cv2.waitKey(1)

    def _update_detection_buffer(self, detections: List[Dict]):
        """Обновляет буфер последних детекций"""
        current_classes = {d['class_name'] for d in detections}
        self._detection_buffer.append(current_classes)

    def _analyze_detections(self):
        """Анализирует буфер детекций для подтверждения объектов"""
        if not self._detection_buffer:
            return
        
        # Считаем количество детекций для каждого класса
        class_counts = defaultdict(int)
        for frame_detections in self._detection_buffer:
            for class_name in frame_detections:
                class_counts[class_name] += 1
        
        # Определяем подтвержденные классы
        confirmed = [
            class_name for class_name, count in class_counts.items()
            if count >= self.min_detections
        ]
        
        # Обновляем список подтвержденных объектов
        if confirmed != self._confirmed_objects:
            self._confirmed_objects = confirmed
            if confirmed:
                print(f"\nПодтвержденные объекты (детектированы {self.min_detections}+ раз):")
                for obj in confirmed:
                    print(f"- {obj}")
