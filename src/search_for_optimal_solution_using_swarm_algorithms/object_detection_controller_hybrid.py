import time
import threading
from collections import deque
from typing import Deque, Optional, Tuple, List, Dict, Union
import cv2
import numpy as np
from ultralytics import YOLO
import math


class YOLODetector:
    """YOLO детектор для обнаружения объектов с поддержкой нескольких моделей"""
    
    def __init__(self, 
                 model_path: Union[str, List[str]] = 'yolov8n.pt', 
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 classes: Optional[List[int]] = None,
                 device: Optional[str] = None):
        """
        Args:
            model_path: путь к файлу модели YOLO или список путей
            conf_threshold: порог уверенности для детекции (0-1)
            iou_threshold: порог IoU для NMS (0-1)
            classes: список ID классов для фильтрации (None - все классы)
            device: устройство для вычислений (None - автоопределение)
        """
        self.models = []
        self.class_names = {}
        
        if isinstance(model_path, str):
            model_path = [model_path]
            
        for path in model_path:
            model = YOLO(path)
            if device is not None:
                model.to(device)
            self.models.append(model)
            self.class_names.update(model.names)
            
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        print(f"Загружены YOLO модели: {model_path}")
        print(f"Объединенные классы: {self.class_names}")

    def detect(self, 
               frame: np.ndarray, 
               imgsz: int = 640,
               augment: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """
        Детектирует объекты на кадре с использованием всех моделей.
        
        Args:
            frame: входной кадр BGR
            imgsz: размер изображения для обработки
            augment: применять аугментацию
            
        Returns:
            annotated_frame: кадр с аннотациями (от последней модели)
            detections: объединенный список детекций с метаданными
        """
        if frame is None:
            return frame, []
            
        all_detections = []
        annotated_frame = frame.copy()
        
        try:
            for model in self.models:
                # Выполняем детекцию
                results = model(
                    frame, 
                    imgsz=imgsz,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=self.classes,
                    augment=augment,
                    verbose=False
                )
                
                # Аннотированный кадр (будет от последней модели)
                annotated_frame = results[0].plot()
                
                # Извлекаем детекции
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf)
                        cls_id = int(box.cls)
                        
                        all_detections.append({
                            'class_id': cls_id,
                            'class_name': self.class_names.get(cls_id, f'class_{cls_id}'),
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                            'area': (x2 - x1) * (y2 - y1)
                        })
            
            return annotated_frame, all_detections
            
        except Exception as e:
            print(f"Ошибка YOLO детекции: {e}")
            return frame, []



class NeuralObjectDetectionController:
    """Полнофункциональный контроллер детекции объектов с поддержкой YOLO и цветовой детекции"""

    # Зоны для животных (координаты в метрах)
    ZONES = {
        "cows": [
            (-2.88, -3.32),  # Загон 1 для коров
            (3.55, -2.85),    # Загон 2 для коров
            (-2.83, 1.61)     # Загон 3 для коров
        ],
        "wolf": [
            (2.75, 3.02)      # Загон для волка
        ]
    }

    def __init__(self,
                 model_path: Union[str, List[str]] = 'yolov8n.pt',
                 conf_threshold: float = 0.5,
                 min_detections: int = 5,
                 buffer_size: int = 10,
                 use_yolo: bool = True,
                 device: Optional[str] = None,
                 average_window: int = 5,
                 report_distance_epsilon: float = 0.1,
                 pixel_distance_epsilon: int = 10,
                 forget_after_misses: int = 10):
        """
        Args:
            model_path: путь к модели YOLO или список путей
            conf_threshold: порог уверенности для детекции
            min_detections: минимальное количество детекций для подтверждения
            buffer_size: размер буфера подтверждения
            use_yolo: использовать YOLO (False - цветовая детекция)
            device: устройство для вычислений (cuda/cpu)
            average_window: размер окна для усреднения координат
            report_distance_epsilon: минимальное изменение координат для события (в метрах)
            pixel_distance_epsilon: минимальное изменение в пикселях для события
            forget_after_misses: сколько кадров без детекции до сброса состояния
        """
        self.use_yolo = use_yolo
        self.conf_threshold = conf_threshold
        self.min_detections = min_detections
        self.buffer_size = buffer_size
        self.average_window = average_window
        self.report_distance_epsilon = report_distance_epsilon
        self.pixel_distance_epsilon = pixel_distance_epsilon
        self.forget_after_misses = forget_after_misses

        # Инициализация детектора
        if self.use_yolo:
            self.detector = YOLODetector(
                model_path=model_path,
                conf_threshold=conf_threshold,
                classes=None,
                device=device
            )
            print(f"Инициализирован YOLO детектор с моделью: {model_path}")
        else:
            self.detector = None
            print("Инициализирован цветовой детектор")

        # Состояние системы
        self._stop_event = threading.Event()
        self._processing_thread = None
        self._last_frame = None
        self._last_detection = {}
        self._detection_buffer = deque(maxlen=buffer_size)
        self._confirmed_event = threading.Event()
        self._external_stop_event = None
        self._pioneer = None
        self._detection_reported = False
        self._detection_paused = False
        self._missed_frames = 0

        # Трекинг координат
        self._last_global_coords = None
        self._last_pixel_coords = None
        self._coord_buffer = deque(maxlen=average_window)
        self._averaged_coords = None
        self._last_assignment = ("unknown", -1)
        self._last_assignment_coords = None

    def start(self):
        """Запускает поток обработки кадров"""
        if self._processing_thread is None:
            self._stop_event.clear()
            self._processing_thread = threading.Thread(
                target=self._process_frames,
                daemon=True
            )
            self._processing_thread.start()
            print("Контроллер детекции запущен")

    def stop(self):
        """Останавливает поток обработки кадров"""
        self._stop_event.set()
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=1)
            self._processing_thread = None
        print("Контроллер детекции остановлен")

    def pause_detection(self):
        """Приостанавливает детекцию"""
        self._detection_paused = True
        self._detection_buffer.clear()
        self._confirmed_event.clear()
        self._coord_buffer.clear()
        print("Детекция приостановлена")

    def resume_detection(self):
        """Возобновляет детекцию"""
        self._detection_paused = False
        print("Детекция возобновлена")

    def is_detection_paused(self) -> bool:
        """Проверяет, приостановлена ли детекция"""
        return self._detection_paused

    def is_confirmed(self) -> bool:
        """Проверяет, подтверждена ли детекция"""
        return self._confirmed_event.is_set()

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Возвращает последний обработанный кадр"""
        return self._last_frame

    def show_frame(self, window_name: str = 'Detection'):
        """Отображает последний кадр"""
        if self._last_frame is not None:
            cv2.imshow(window_name, self._last_frame)
            cv2.waitKey(1)

    def set_external_stop_event(self, stop_event: threading.Event):
        """Устанавливает внешнее событие остановки"""
        self._external_stop_event = stop_event

    def set_pioneer(self, pioneer):
        """Устанавливает объект дрона для получения позиции"""
        self._pioneer = pioneer

    def process_frame(self,
                     frame: np.ndarray,
                     drone_x: float = 0.0,
                     drone_y: float = 0.0) -> Dict:
        """
        Основной метод обработки кадра.
        
        Returns:
            Словарь с результатами:
            {
                "annotated": аннотированный кадр,
                "detections": список детекций,
                "global_coords": текущие координаты,
                "averaged_global_coords": усредненные координаты,
                "zone_type": тип текущей зоны,
                "zone_index": индекс зоны,
                "assigned_zone_type": назначенный тип зоны,
                "assigned_zone_index": назначенный индекс зоны,
                "coords_event": флаг изменения координат,
                "assignment_event": флаг изменения назначения,
                "detection_event": флаг новой детекции
            }
        """
        result = {
            "annotated": frame,
            "detections": [],
            "global_coords": None,
            "averaged_global_coords": None,
            "zone_type": "unknown",
            "zone_index": -1,
            "assigned_zone_type": "unknown",
            "assigned_zone_index": -1,
            "coords_event": False,
            "assignment_event": False,
            "detection_event": False
        }

        if frame is None or self._detection_paused:
            return result

        # Детекция объектов
        detections = []
        annotated_frame = frame.copy()

        if self.use_yolo and self.detector:
            annotated_frame, detections = self.detector.detect(frame)
        else:
            # Реализация цветовой детекции может быть добавлена здесь
            pass

        result["annotated"] = annotated_frame
        result["detections"] = detections
        self._last_frame = annotated_frame

        # Обработка детекций
        if detections:
            self._missed_frames = 0
            main_detection = max(detections, key=lambda x: x['confidence'])
            center = main_detection['center']

            # Вычисление глобальных координат
            global_coords = self.get_global_coordinates(center, drone_x, drone_y)
            result["global_coords"] = global_coords

            # Усреднение координат
            self._coord_buffer.append(global_coords)
            avg_x = sum(c[0] for c in self._coord_buffer) / len(self._coord_buffer)
            avg_y = sum(c[1] for c in self._coord_buffer) / len(self._coord_buffer)
            result["averaged_global_coords"] = (avg_x, avg_y)

            # Проверка изменения координат
            coord_event = False
            if self._last_global_coords is not None:
                dx = avg_x - self._last_global_coords[0]
                dy = avg_y - self._last_global_coords[1]
                dist = math.sqrt(dx**2 + dy**2)
                coord_event = dist > self.report_distance_epsilon

                # Проверка изменения в пикселях
                if self._last_pixel_coords:
                    px_dist = math.sqrt(
                        (center[0] - self._last_pixel_coords[0])**2 +
                        (center[1] - self._last_pixel_coords[1])**2
                    )
                    coord_event = coord_event or (px_dist > self.pixel_distance_epsilon)

            result["coords_event"] = coord_event
            self._last_global_coords = (avg_x, avg_y)
            self._last_pixel_coords = center

            # Определение текущей зоны
            zone_type, zone_idx = self.assign_to_zone((avg_x, avg_y))
            result["zone_type"] = zone_type
            result["zone_index"] = zone_idx

            # Назначение в ближайший загон
            assigned_type, assigned_idx = self._assign_to_nearest_zone((avg_x, avg_y))
            result["assigned_zone_type"] = assigned_type
            result["assigned_zone_index"] = assigned_idx

            # Проверка изменения назначения
            assignment_event = False
            if (assigned_type, assigned_idx) != self._last_assignment:
                assignment_event = True
                self._last_assignment = (assigned_type, assigned_idx)
                self._last_assignment_coords = (avg_x, avg_y)

            result["assignment_event"] = assignment_event

            # Обновление буфера детекций
            self._detection_buffer.append(True)
            true_count = sum(1 for v in self._detection_buffer if v)
            if true_count >= self.min_detections:
                if not self._confirmed_event.is_set():
                    result["detection_event"] = True
                self._confirmed_event.set()
            else:
                self._confirmed_event.clear()
        else:
            # Нет детекций
            self._missed_frames += 1
            self._detection_buffer.append(False)
            if self._missed_frames >= self.forget_after_misses:
                self._coord_buffer.clear()
                self._last_global_coords = None
                self._last_pixel_coords = None
                self._confirmed_event.clear()

        return result

    def get_global_coordinates(self,
                              center: Tuple[int, int],
                              drone_x: float,
                              drone_y: float) -> Tuple[float, float]:
        """
        Преобразует координаты центра объекта на изображении в глобальные координаты.
        
        Args:
            center: (x, y) координаты центра объекта в пикселях
            drone_x: текущая X координата дрона (метры)
            drone_y: текущая Y координата дрона (метры)
            
        Returns:
            Глобальные координаты (x, y) в метрах
        """
        img_width, img_height = 640, 480  # Предполагаемый размер кадра
        img_mid_x, img_mid_y = img_width / 2, img_height / 2
        
        # Коэффициенты преобразования пикселей в метры
        # (подбираются экспериментально для вашей высоты полета)
        scale_x = 2.4 / img_width  # метра на пиксель по X
        scale_y = 1.4 / img_height  # метра на пиксель по Y
        
        # Вычисляем смещение от центра кадра
        dx = (center[0] - img_mid_x) * scale_x
        dy = (img_mid_y - center[1]) * scale_y  # Ось Y инвертирована
        
        # Глобальные координаты объекта
        global_x = drone_x + dx
        global_y = drone_y + dy
        
        return global_x, global_y

    def assign_to_zone(self, coords: Tuple[float, float]) -> Tuple[str, int]:
        """
        Определяет, в какую зону попадают координаты.
        
        Args:
            coords: глобальные координаты (x, y)
            
        Returns:
            (тип зоны, индекс зоны) или ("unknown", -1)
        """
        x, y = coords
        zone_radius = 1.0  # Радиус зоны в метрах

        for zone_type, zone_list in self.ZONES.items():
            for i, (zone_x, zone_y) in enumerate(zone_list):
                if math.sqrt((x - zone_x)**2 + (y - zone_y)**2) <= zone_radius:
                    return zone_type, i

        return "unknown", -1

    def _assign_to_nearest_zone(self, coords: Tuple[float, float]) -> Tuple[str, int]:
        """
        Находит ближайшую зону для указанных координат.
        
        Args:
            coords: глобальные координаты (x, y)
            
        Returns:
            (тип зоны, индекс зоны) или ("unknown", -1)
        """
        x, y = coords
        min_dist = float('inf')
        best_zone = ("unknown", -1)

        for zone_type, zone_list in self.ZONES.items():
            for i, (zone_x, zone_y) in enumerate(zone_list):
                dist = math.sqrt((x - zone_x)**2 + (y - zone_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_zone = (zone_type, i)

        return best_zone

    def _process_frames(self):
        """Основной цикл обработки кадров (для работы в отдельном потоке)"""
        while not self._stop_event.is_set():
            if self._external_stop_event and self._external_stop_event.is_set():
                break

            # Здесь может быть реализация обработки кадров из видеопотока
            time.sleep(0.01)