import cv2
import time
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, List
import os

# Импорты для YOLO детекции
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO не доступен - будет использоваться только цветовая детекция")


def detect_red_pixels_bgr(frame: np.ndarray, target_bgr: Tuple[int, int, int] = (0, 0, 200), 
                         tolerance: int = 30) -> Tuple[bool, Tuple[int, int] | None, Tuple[int, int, int, int] | None]:
    """Детекция красных пикселей в BGR формате. Возвращает (found, center, bbox)"""
    if frame is None:
        return False, None, None

    b, g, r = target_bgr
    lower_bgr = np.array([max(0, b - tolerance), max(0, g - tolerance), max(0, r - tolerance)])
    upper_bgr = np.array([min(255, b + tolerance), min(255, g + tolerance), min(255, r + tolerance)])

    mask = cv2.inRange(frame, lower_bgr, upper_bgr)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 500:
        return False, None, None

    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2
    return True, (cx, cy), (x, y, w, h)


def detect_red_cow(frame: np.ndarray) -> Tuple[bool, Tuple[int, int] | None, Tuple[int, int, int, int] | None]:
    """Детекция красного объекта (коровы) через HSV маску. Возвращает (found, center, bbox)"""
    if frame is None:
        return False, None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 1000 or area > 5000:
        return False, None, None

    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2
    return True, (cx, cy), (x, y, w, h)


def detect_red_combined(frame: np.ndarray) -> Tuple[bool, Tuple[int, int] | None, Tuple[int, int, int, int] | None, str]:
    """Комбинированная детекция красных объектов: HSV + BGR маски."""
    if frame is None:
        return False, None, None, "none"

    found_hsv, center_hsv, bbox_hsv = detect_red_cow(frame)
    if found_hsv:
        return found_hsv, center_hsv, bbox_hsv, "hsv_cow"

    found_bgr, center_bgr, bbox_bgr = detect_red_pixels_bgr(frame, target_bgr=(0, 0, 200), tolerance=30)
    if found_bgr:
        return found_bgr, center_bgr, bbox_bgr, "bgr_red_pixels"

    return False, None, None, "none"


class YOLODetector:
    """YOLO детектор для обнаружения объектов"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO не доступен. Установите ultralytics: pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        results = self.model(frame, imgsz=640, conf=self.conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detections.append({
                'class_id': int(box.cls),
                'class_name': self.class_names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': [x1, y1, x2, y2],
                'center': [center_x, center_y],
                'area': (x2 - x1) * (y2 - y1)
            })
        return annotated_frame, detections


class ObjectDetectionController:
    """Синхронный контроллер детекции и распределения (без потоков).

    Подтверждение детекции основано на минимальном количестве попаданий в скользящем окне.
    Глобальные координаты усредняются по последним N попаданиям для стабилизации.
    Поддерживает как цветовую детекцию, так и YOLO детекцию.
    """

    ZONES = {
        "cows": [
             (3.55, -2.85),
            (-2.88, -3.32),
            (-2.83, 1.61)
        ],
        "wolf": [
            (2.75, 3.02)
        ]
    }

    def __init__(self, 
                 min_detections: int = 5, 
                 buffer_size: int = 10,
                 min_confirmations: Optional[int] = None,
                 average_window: int = 5,
                 report_distance_epsilon: float = 0.05,
                 pixel_distance_epsilon: int = 6,
                 forget_after_misses: int = 8,
                 use_yolo: bool = False,
                 yolo_model_path: str = 'yolov8n.pt',
                 yolo_conf_threshold: float = 0.5):
        """Инициализация контроллера.

        Args:
            min_detections: размер окна для накапливания попаданий (True/False).
            buffer_size: длина буфера присутствия для подтверждения.
            min_confirmations: минимальное количество True в окне для подтверждения; если None, берется min_detections.
            average_window: длина окна усреднения глобальных координат.
            report_distance_epsilon: минимальный сдвиг усредненных координат (в метрах) для повторного вывода.
            pixel_distance_epsilon: порог сдвига центра (в пикселях) для определения стабильности объекта между кадрами.
            forget_after_misses: после скольких подряд кадров без объекта сбрасывать трекинг и назначение.
            use_yolo: использовать ли YOLO детекцию вместо цветовой.
            yolo_model_path: путь к YOLO модели.
            yolo_conf_threshold: порог уверенности для YOLO детекции.
        """
        # Минимум попаданий для подтверждения (если не указан, используем min_detections)
        self.min_detections = min_detections
        self.min_confirmations = min_confirmations if min_confirmations is not None else min_detections

        # Буфер присутствия (True/False) для подтверждения
        self._presence_buffer = deque(maxlen=buffer_size)

        # Буфер координат для усреднения
        self._coords_buffer: deque[Tuple[float, float]] = deque(maxlen=average_window)

        # Последний аннотированный кадр
        self._last_frame: Optional[np.ndarray] = None

        # Порог изменения координат для повторного вывода
        self._report_eps = report_distance_epsilon

        # Последние «доложенные» значения для подавления повторов
        self._last_reported_zone: Tuple[str, int] | None = None
        self._last_reported_coords: Tuple[float, float] | None = None

        # Трекинг объекта между кадрами по центру (в пикселях)
        self._last_center_px: Tuple[int, int] | None = None
        self._pixel_eps: int = pixel_distance_epsilon

        # Трекинг потери объекта
        self._not_found_counter: int = 0
        self._forget_after_misses: int = forget_after_misses

        # Занятость загонов (персистентно на время работы контроллера)
        self._occupied_zones: Dict[str, list[bool]] = {
            "cows": [False, False, False],
            "wolf": [False]
        }

        # Последняя назначенная зона (для данного текущего животного)
        self._last_assigned_zone: Tuple[str, int] | None = None
        
        # YOLO детектор (если используется)
        self.use_yolo = use_yolo
        self.yolo_detector = None
        if use_yolo:
            if YOLO_AVAILABLE:
                try:
                    self.yolo_detector = YOLODetector(yolo_model_path, yolo_conf_threshold)
                    print(f"YOLO детектор инициализирован с моделью: {yolo_model_path}")
                except Exception as e:
                    print(f"Ошибка инициализации YOLO: {e}")
                    self.use_yolo = False
            else:
                print("YOLO недоступен - переключаемся на цветовую детекцию")
                self.use_yolo = False

    def get_last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def get_global_coordinates(self, center: Tuple[int, int], drone_x: float, drone_y: float) -> Tuple[float, float]:
        """Вычисляет глобальные координаты на основе позиции дрона."""
        img_width, img_height = 640, 640
        img_mid_x, img_mid_y = img_width / 2, img_height / 2
        scale_x = 1 / img_width
        scale_y = 1 / img_height
        global_x = drone_x + (center[0] - img_mid_x) * scale_x
        global_y = drone_y - (center[1] - img_mid_y) * scale_y
        return global_x, global_y

    def assign_to_zone(self, global_coords: Tuple[float, float]) -> Tuple[str, int]:
        x, y = global_coords
        for i, (zx, zy) in enumerate(self.ZONES["cows"]):
            if abs(x - zx) < 1.0 and abs(y - zy) < 1.0:
                return "cows", i
        for i, (zx, zy) in enumerate(self.ZONES["wolf"]):
            if abs(x - zx) < 1.0 and abs(y - zy) < 1.0:
                return "wolf", i
        return "unknown", -1

    def _detect_with_yolo(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int] | None, Tuple[int, int, int, int] | None, str]:
        """Детекция объектов с помощью YOLO"""
        if not self.use_yolo or self.yolo_detector is None:
            return False, None, None, "none"
        
        try:
            annotated_frame, detections = self.yolo_detector.detect(frame)
            
            # Ищем корову или волка среди детекций
            for detection in detections:
                class_name = detection['class_name'].lower()
                if 'cow' in class_name or 'cattle' in class_name or 'animal' in class_name:
                    center = detection['center']
                    bbox = detection['bbox']
                    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    return True, center, (x, y, w, h), "yolo_cow"
                elif 'wolf' in class_name or 'dog' in class_name:
                    center = detection['center']
                    bbox = detection['bbox']
                    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    return True, center, (x, y, w, h), "yolo_wolf"
            
            return False, None, None, "none"
            
        except Exception as e:
            print(f"Ошибка YOLO детекции: {e}")
            return False, None, None, "none"

    def _detect_combined(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int] | None, Tuple[int, int, int, int] | None, str]:
        """Комбинированная детекция: YOLO + цветовые маски"""
        if frame is None:
            return False, None, None, "none"

        # Сначала пробуем YOLO
        if self.use_yolo:
            found_yolo, center_yolo, bbox_yolo, det_type_yolo = self._detect_with_yolo(frame)
            if found_yolo:
                return found_yolo, center_yolo, bbox_yolo, det_type_yolo

        # Если YOLO не сработал или недоступен - используем цветовые маски
        found_hsv, center_hsv, bbox_hsv = detect_red_cow(frame)
        if found_hsv:
            return found_hsv, center_hsv, bbox_hsv, "hsv_cow"

        found_bgr, center_bgr, bbox_bgr = detect_red_pixels_bgr(frame, target_bgr=(0, 0, 200), tolerance=30)
        if found_bgr:
            return found_bgr, center_bgr, bbox_bgr, "bgr_red_pixels"

        return False, None, None, "none"

    def process_frame(self, frame: np.ndarray, drone_x: float, drone_y: float) -> Dict:
        """Обработка одного кадра: детекция, усреднение координат, распределение, аннотация, подавление повторов."""
        found, center, bbox, det_type = self._detect_combined(frame)
        annotated = frame.copy() if frame is not None else None

        # Обновляем буфер присутствия
        self._presence_buffer.append(bool(found))
        confirmed = sum(1 for v in self._presence_buffer if v) >= self.min_confirmations

        # Координаты (сырые и усредненные)
        global_coords: Optional[Tuple[float, float]] = None
        averaged_coords: Optional[Tuple[float, float]] = None

        if found and center is not None:
            if bbox is not None:
                x, y, w, h = bbox
                # Разные цвета для разных типов детекции
                if det_type.startswith("yolo"):
                    color = (0, 255, 255)  # Желтый для YOLO
                elif det_type == "hsv_cow":
                    color = (0, 255, 0)    # Зеленый для HSV
                else:
                    color = (0, 0, 255)    # Красный для BGR
                
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, f"{det_type}: {w*h}", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.circle(annotated, center, 8, (255, 255, 0), 2)
            cv2.putText(annotated, det_type, (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Сырые глобальные координаты
            global_coords = self.get_global_coordinates(center, drone_x, drone_y)

            # Кладем в буфер и считаем среднее
            self._coords_buffer.append(global_coords)
            avg_x = sum(c[0] for c in self._coords_buffer) / len(self._coords_buffer)
            avg_y = sum(c[1] for c in self._coords_buffer) / len(self._coords_buffer)
            averaged_coords = (avg_x, avg_y)

        # На основании усредненных координат определяем зону (геометрическое попадание)
        zone_type, zone_idx = "unknown", -1
        if averaged_coords is not None:
            zone_type, zone_idx = self.assign_to_zone(averaged_coords)

        # Рисуем вспомогательные подписи
        if averaged_coords is not None:
            cv2.putText(annotated, f"Avg: ({averaged_coords[0]:.2f}, {averaged_coords[1]:.2f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if zone_type != "unknown":
                cv2.putText(annotated, f"Zone: {zone_type} #{zone_idx+1}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Добавляем информацию о типе детекции
        cv2.putText(annotated, f"Detection: {'YOLO' if self.use_yolo else 'Color'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Подавление повторов для сообщений о координатах/геометрической зоне
        coords_event = False
        if confirmed and averaged_coords is not None:
            need_report = False
            if self._last_reported_zone is None or self._last_reported_coords is None:
                need_report = True
            else:
                last_zone_type, last_zone_idx = self._last_reported_zone
                last_x, last_y = self._last_reported_coords
                dx = averaged_coords[0] - last_x
                dy = averaged_coords[1] - last_y
                moved_far = (dx*dx + dy*dy) ** 0.5 > self._report_eps
                zone_changed = (zone_type != last_zone_type) or (zone_idx != last_zone_idx)
                need_report = moved_far or zone_changed

            if need_report:
                self._last_reported_zone = (zone_type, zone_idx)
                self._last_reported_coords = averaged_coords
                coords_event = True

        # Назначение в ближайший свободный загон по событию подтверждения (без дублей)
        assigned_zone_type, assigned_zone_idx = "unknown", -1
        assignment_event = False
        if confirmed and averaged_coords is not None:
            # Определяем тип животного на основе детекции
            animal_type = None
            if det_type.startswith("yolo_cow") or det_type == "hsv_cow":
                animal_type = "cows"
            elif det_type.startswith("yolo_wolf"):
                animal_type = "wolf"

            if animal_type:
                # Если зона ещё не назначалась для текущего текуще отслеживаемого объекта — пытаемся назначить
                if self._last_assigned_zone is None:
                    assigned_zone_type, assigned_zone_idx = self._assign_nearest_free_zone(averaged_coords, animal_type)
                    if assigned_zone_type != "unknown" and assigned_zone_idx >= 0:
                        self._last_assigned_zone = (assigned_zone_type, assigned_zone_idx)
                        assignment_event = True
                else:
                    assigned_zone_type, assigned_zone_idx = self._last_assigned_zone

        # Обновляем последнюю точку центра (пиксели)
        if center is not None:
            self._last_center_px = center

        # Если длительное отсутствие объекта — очищаем буфер координат и сбрасываем отчеты
        if not found:
            # Счетчик пропусков
            self._not_found_counter += 1
            if self._not_found_counter >= self._forget_after_misses:
                # Забываем трек: координаты/центр/назначение (занятость зон НЕ сбрасываем)
                self._coords_buffer.clear()
                self._last_center_px = None
                self._last_assigned_zone = None
                self._last_reported_zone = None
                self._last_reported_coords = None
                self._not_found_counter = 0
        else:
            # Объект виден — сбрасываем счетчик пропусков
            self._not_found_counter = 0

        self._last_frame = annotated
        return {
            "found": bool(found),
            "confirmed": confirmed,
            "center": center,
            "bbox": bbox,
            "type": det_type,
            "global_coords": global_coords,
            "averaged_global_coords": averaged_coords,
            "zone_type": zone_type,
            "zone_index": zone_idx,
            "assigned_zone_type": assigned_zone_type,
            "assigned_zone_index": assigned_zone_idx,
            "annotated": annotated,
            "coords_event": coords_event,
            "assignment_event": assignment_event,
        }

    def _assign_nearest_free_zone(self, coords: Tuple[float, float], animal_type: str) -> Tuple[str, int]:
        """Назначает ближайший свободный загон для переданного типа животного.
        Возвращает (тип, индекс) либо ("unknown", -1), если свободных нет/тип неизвестен."""
        if animal_type not in self.ZONES:
            return "unknown", -1

        free_indices = [i for i, occupied in enumerate(self._occupied_zones.get(animal_type, [])) if not occupied]
        if not free_indices:
            return "unknown", -1

        # Ищем ближайший свободный
        best_idx = -1
        best_dist = float("inf")
        for i in free_indices:
            zx, zy = self.ZONES[animal_type][i]
            dx = coords[0] - zx
            dy = coords[1] - zy
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            self._occupied_zones[animal_type][best_idx] = True
            return animal_type, best_idx
        return "unknown", -1

    def reset_zones(self):
        """Сброс занятости всех загонов и последнего назначения."""
        self._occupied_zones = {
            "cows": [False, False, False],
            "wolf": [False]
        }
        self._last_assigned_zone = None
