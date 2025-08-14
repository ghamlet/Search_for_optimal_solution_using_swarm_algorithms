import time
import threading
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from camera_sim import CameraSim


def detect_red_pixels_bgr(frame: np.ndarray, target_bgr: Tuple[int, int, int] = (0, 0, 200), 
                         tolerance: int = 30) -> Tuple[bool, Tuple[int, int] | None, Tuple[int, int, int, int] | None]:
    """Детекция красных пикселей в BGR формате.
    Возвращает (found, center, bbox)"""
    if frame is None:
        return False, None, None

    # Создаем маску для красных пикселей в BGR
    b, g, r = target_bgr
    lower_bgr = np.array([max(0, b - tolerance), max(0, g - tolerance), max(0, r - tolerance)])
    upper_bgr = np.array([min(255, b + tolerance), min(255, g + tolerance), min(255, r + tolerance)])
    
    # Создаем маску
    mask = cv2.inRange(frame, lower_bgr, upper_bgr)
    
    # Морфологические операции для улучшения маски
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None

    # Выбираем самый большой контур
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    
    # Фильтруем по размеру области
    if area < 500:  # Минимальная площадь для красных пикселей
        return False, None, None

    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2
    return True, (cx, cy), (x, y, w, h)


def detect_red_cow(frame: np.ndarray) -> Tuple[bool, Tuple[int, int] | None, Tuple[int, int, int, int] | None]:
    """Детекция красного объекта (коровы) через HSV маску.
    Возвращает (found, center, bbox)"""
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
    """Комбинированная детекция красных объектов: HSV + BGR маски.
    Возвращает (found, center, bbox, detection_type)"""
    if frame is None:
        return False, None, None, "none"

    # Пробуем детекцию через HSV (для коров)
    found_hsv, center_hsv, bbox_hsv = detect_red_cow(frame)
    if found_hsv:
        return found_hsv, center_hsv, bbox_hsv, "hsv_cow"

    # Пробуем детекцию через BGR (для красных пикселей)
    found_bgr, center_bgr, bbox_bgr = detect_red_pixels_bgr(frame, target_bgr=(0, 0, 200), tolerance=30)
    if found_bgr:
        return found_bgr, center_bgr, bbox_bgr, "bgr_red_pixels"

    return False, None, None, "none"


class ObjectDetectionControllerSim:
    """Контроллер сим-детекции с подтверждением по нескольким кадрам."""

    # Зоны для животных (координаты в метрах)
    ZONES = {
        "cows": [
            (-2.88, -3.32),  # Загон 1 для коров
            (3.55, -2.85),   # Загон 2 для коров
            (-2.83, 1.61)    # Загон 3 для коров
        ],
        "wolf": [
            (2.75, 3.02)     # Загон для волка
        ]
    }

    def __init__(self, camera: CameraSim, min_detections: int = 5, buffer_size: int = 10):
        self.camera = camera
        self.min_detections = min_detections
        self.buffer_size = buffer_size

        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_detection: Tuple[bool, Tuple[int, int] | None, Tuple[int, int, int, int] | None] = (False, None, None)
        self._detection_buffer: Deque[bool] = deque(maxlen=buffer_size)
        self._confirmed_event = threading.Event()
        self._external_stop_event: Optional[threading.Event] = None
        self._pioneer = None
        self._detection_reported = False
        self._detection_paused = False  # Флаг для приостановки детекции

    def start(self):
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self._processing_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=1)

    def is_confirmed(self) -> bool:
        return self._confirmed_event.is_set()

    def get_last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def show_frame(self, window_name: str = 'SIM Detection'):
        if self._last_frame is not None:
            cv2.imshow(window_name, self._last_frame)
            cv2.waitKey(1)

    def set_external_stop_event(self, stop_event: threading.Event):
        self._external_stop_event = stop_event

    def set_pioneer(self, pioneer):
        """Устанавливает объект дрона для получения позиции."""
        self._pioneer = pioneer

    def pause_detection(self):
        """Приостанавливает детекцию."""
        self._detection_paused = True
        # Очищаем буфер детекции и сбрасываем событие подтверждения
        self._detection_buffer.clear()
        self._confirmed_event.clear()
        print("Детекция приостановлена, буфер очищен")

    def resume_detection(self):
        """Возобновляет детекцию."""
        self._detection_paused = False
        print("Детекция возобновлена")

    def is_detection_paused(self):
        """Проверяет, приостановлена ли детекция."""
        return self._detection_paused

    def get_global_coordinates(self, center: Tuple[int, int]) -> Tuple[float, float]:
        """Вычисляет глобальные координаты объекта на основе позиции дрона."""
        if self._pioneer is None:
            return (0.0, 0.0)
        
        drone_x, drone_y = self._pioneer.position[0], self._pioneer.position[1]
        
        # Параметры камеры (как в основном контроллере)
        img_width, img_height = 640, 480
        img_mid_x, img_mid_y = img_width / 2, img_height / 2
        
        # Коэффициенты перевода пикселей в метры
        scale_x = 2.4 / img_width
        scale_y = 1.4 / img_height
        
        # Расчет глобальных координат
        global_x = drone_x + (center[0] - img_mid_x) * scale_x
        global_y = drone_y - (center[1] - img_mid_y) * scale_y
        
      
        return global_x, global_y

    def assign_to_zone(self, global_coords: Tuple[float, float]) -> Tuple[str, int]:
        """Определяет, в какой загон попадает объект."""
        x, y = global_coords
        
        # Проверяем загоны для коров
        for i, (zone_x, zone_y) in enumerate(self.ZONES["cows"]):
            if abs(x - zone_x) < 1.0 and abs(y - zone_y) < 1.0:  # радиус 1 метр
                return "cows", i
        
        # Проверяем загон для волка
        for i, (zone_x, zone_y) in enumerate(self.ZONES["wolf"]):
            if abs(x - zone_x) < 1.0 and abs(y - zone_y) < 1.0:
                return "wolf", i
        
        return "unknown", -1

    def _process_frames(self):
        while not self._stop_event.is_set():
            frame = self.camera.get_cv_frame()
            if frame is None:
                # если камера вернула None, просто подождём и не перерисовываем
                time.sleep(0.01)
                continue

            # Проверяем, не приостановлена ли детекция
            if self._detection_paused:
                # Если детекция приостановлена, просто показываем кадр без обработки
                self._last_frame = frame.copy()
                time.sleep(0.01)
                continue

            found, center, bbox, detection_type = detect_red_combined(frame)

            # аннотируем кадр
            annotated = frame.copy()
            if found and center is not None:
                if bbox is not None:
                    x, y, w, h = bbox
                    
                    # Выбираем цвет в зависимости от типа детекции
                    if detection_type == "hsv_cow":
                        color = (0, 255, 0)  # Зеленый для коров
                        label = "RED COW"
                    elif detection_type == "bgr_red_pixels":
                        color = (0, 0, 255)  # Красный для красных пикселей
                        label = "RED PIXELS"
                    else:
                        color = (255, 255, 0)  # Голубой для неизвестного типа
                        label = "UNKNOWN"
                    
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    area_px = w * h
                    cv2.putText(annotated, f"Area: {area_px}", (x, max(0, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Вычисляем глобальные координаты и определяем загон
                    global_coords = self.get_global_coordinates(center)
                    zone_type, zone_idx = self.assign_to_zone(global_coords)
                    
                    # Выводим информацию на кадр
                    coord_text = f"Global: ({global_coords[0]:.2f}, {global_coords[1]:.2f})"
                    cv2.putText(annotated, coord_text, (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if zone_type != "unknown":
                        zone_text = f"Zone: {zone_type} #{zone_idx + 1}"
                        cv2.putText(annotated, zone_text, (x, y + h + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Выводим в консоль при подтверждении (только один раз)
                    if self._confirmed_event.is_set() and not hasattr(self, '_detection_reported'):
                        if detection_type == "hsv_cow":
                            print(f"Красная корова обнаружена!")
                        elif detection_type == "bgr_red_pixels":
                            print(f"Красные пиксели обнаружены!")
                        else:
                            print(f"Красный объект обнаружен (тип: {detection_type})!")
                        
                        print(f"Глобальные координаты: ({global_coords[0]:.2f}, {global_coords[1]:.2f})")
                        if zone_type != "unknown":
                            print(f"Назначена в загон: {zone_type} #{zone_idx + 1}")
                        else:
                            print("Не попадает ни в один загон")
                        self._detection_reported = True
                
                # Выбираем цвет для круга и текста
                if detection_type == "hsv_cow":
                    color = (0, 255, 0)  # Зеленый для коров
                    label = "RED COW"
                elif detection_type == "bgr_red_pixels":
                    color = (0, 0, 255)  # Красный для красных пикселей
                    label = "RED PIXELS"
                else:
                    color = (255, 255, 0)  # Голубой для неизвестного типа
                    label = "UNKNOWN"
                
                cv2.circle(annotated, center, 8, color, 2)
                cv2.putText(annotated, label, (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self._last_frame = annotated
            self._last_detection = (found, center, bbox)
            
            # Отладочная информация для проверки данных детекции
            if found and center is not None:
                # print(f"DETECTOR DEBUG: found={found}, center={center}, bbox={bbox}")
                global_coords = self.get_global_coordinates(center)
                # print(f"DETECTOR DEBUG: global_coords={global_coords}")

            # обновляем буфер подтверждений
            self._detection_buffer.append(bool(found))
            true_count = sum(1 for v in self._detection_buffer if v)
            if true_count >= self.min_detections:
                self._confirmed_event.set()

            time.sleep(0.01)


