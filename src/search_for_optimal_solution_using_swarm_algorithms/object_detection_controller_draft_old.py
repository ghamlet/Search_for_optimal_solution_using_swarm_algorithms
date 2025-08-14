import cv2
import time
import threading
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Optional, Tuple, Any, Union
from queue import Queue
from camera_real import CameraReal
from yolo_detector import YOLODetector


class ObjectDetectionController:
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
    
    # Классы животных
    COW_CLASSES = ['white', 'black', 'brown']
    WOLF_CLASS = ['wolf']

    def __init__(self,
                 pioneer: Any,
                 camera_source: Union[str, int] = 0,
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.5,
                 min_detections: int = 5,
                 buffer_size: int = 10):
        """
        Инициализация контроллера детекции объектов
        
        Args:
            pioneer: Объект для управления дроном
            camera_source: Источник видео (путь/индекс камеры)
            model_path: Путь к модели YOLO
            conf_threshold: Порог уверенности детекции (0-1)
            min_detections: Минимальное количество детекций для подтверждения
            buffer_size: Размер буфера для анализа детекций
        """
        self.pioneer = pioneer
        self.min_detections = min_detections
        self.buffer_size = buffer_size
        
        # Инициализация компонентов
        self.camera = CameraReal(source=camera_source)
        self.detector = YOLODetector(model_path=model_path, conf_threshold=conf_threshold)
        
        # Потоковые атрибуты
        self._stop_event = threading.Event()
        self._frame_queue = Queue(maxsize=1)
        self._results_queue = Queue(maxsize=10)
        
        # Состояние системы
        self._detection_buffer = deque(maxlen=buffer_size)
        self._confirmed_objects = []
        self._last_frame = None
        self._occupied_zones = {
            'cows': [False, False, False],
            'wolf': [False]
        }
        self._stats = {
            'processed_frames': 0,
            'current_fps': 0.0,
            'detection_time': 0.0
        }
        self._frame_counter = 0
        self._last_fps_update = time.time()

    def start(self):
        """Запуск всех потоков обработки"""
        self._stop_event.clear()
        
        # Поток захвата кадров
        self._capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True
        )
        
        # Поток обработки
        self._processing_thread = threading.Thread(
            target=self._process_frames,
            daemon=True
        )
        
        # Поток вывода координат
        self._position_thread = threading.Thread(
            target=self._print_position,
            daemon=True
        )
        
        # Поток отчета по детекциям и зонам
        self._report_thread = threading.Thread(
            target=self._report_detections_and_zones,
            daemon=True
        )
        
        self._capture_thread.start()
        self._processing_thread.start()
        self._position_thread.start()
        self._report_thread.start()
        print("Все потоки запущены")

    def stop(self):
        """Остановка всех потоков и освобождение ресурсов"""
        self._stop_event.set()
        
        if hasattr(self, '_capture_thread'):
            self._capture_thread.join(timeout=1)
        if hasattr(self, '_processing_thread'):
            self._processing_thread.join(timeout=1)
        if hasattr(self, '_position_thread'):
            self._position_thread.join(timeout=1)
        if hasattr(self, '_report_thread'):
            self._report_thread.join(timeout=1)
            
        self.camera.release()
        cv2.destroyAllWindows()
        print("Система остановлена")

    def get_current_position(self) -> Tuple[float, float]:
        """Получение текущих координат дрона (x, y)"""
        return self.pioneer.position[0], self.pioneer.position[1]

    def get_last_detections(self) -> List[Dict]:
        """Получение последних результатов детекции"""
        results = []
        while not self._results_queue.empty():
            results.append(self._results_queue.get())
        return results

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Получение последнего обработанного кадра"""
        return self._last_frame

    def get_confirmed_objects(self) -> List[str]:
        """Получение списка подтвержденных объектов"""
        return self._confirmed_objects.copy()

    def calculate_object_coordinates(self, drone_x: float, drone_y: float, 
                                  object_x: float, object_y: float) -> Tuple[float, float]:
        """
        Расчет глобальных координат объекта
        
        Args:
            drone_x: X-координата дрона
            drone_y: Y-координата дрона
            object_x: X-координата объекта на изображении
            object_y: Y-координата объекта на изображении
            
        Returns:
            Tuple: Глобальные координаты объекта (x, y)
        """
        img_width = 640  # Ширина изображения модели YOLO
        img_height = 640 # Высота изображения модели YOLO
        img_mid_x = img_width / 2
        img_mid_y = img_height / 2
        
        # Коэффициенты перевода пикселей в метры
        scale_x = 0.9 / img_width  # 0.9 метров по X на весь кадр
        scale_y = 0.9 / img_height # 0.9 метров по Y на весь кадр
        
        # Расчет глобальных координат
        global_x = drone_x + (object_x - img_mid_x) * scale_x
        global_y = drone_y - (object_y - img_mid_y) * scale_y
        
        return global_x, global_y

    def get_last_detections_with_coords(self) -> List[Dict]:
        """
        Получение детекций с глобальными координатами
        
        Returns:
            List: Детекции с добавленными глобальными координатами
        """
        drone_x, drone_y = self.get_current_position()
        detections = self.get_last_detections()
        result = []
        
        for frame_data in detections:
            frame_detections = []
            for det in frame_data['detections']:
                # Расчет глобальных координат для каждого объекта
                obj_x, obj_y = det['center']
                global_x, global_y = self.calculate_object_coordinates(
                    drone_x, drone_y, obj_x, obj_y
                )
                
                # Создание копии детекции с добавленными координатами
                det_with_coords = det.copy()
                det_with_coords['global_coords'] = (global_x, global_y)
                frame_detections.append(det_with_coords)
            
            # Сохранение результатов для кадра
            result.append({
                'timestamp': frame_data['timestamp'],
                'detections': frame_detections,
                'frame': frame_data['frame']
            })
        
        return result

    def assign_to_zones(self, detections: List[Dict]) -> Dict:
        """
        Распределение объектов по загонам
        
        Args:
            detections: Список детекций с глобальными координатами
            
        Returns:
            Dict: {
                'assigned': [(объект, зона), ...],
                'unassigned': [объекты]
            }
        """
        result = {
            'assigned': [],
            'unassigned': []
        }
        
        # Фильтрация животных
        wolves = []
        cows = []
        for frame in detections:
            for det in frame['detections']:
                if det['class_name'] in self.WOLF_CLASS:
                    wolves.append(det)
                elif det['class_name'] in self.COW_CLASSES:
                    cows.append(det)
        
        # Обработка волков
        for wolf in wolves:
            if not self._occupied_zones['wolf'][0]:
                zone = self.ZONES['wolf'][0]
                result['assigned'].append((wolf, zone))
                self._occupied_zones['wolf'][0] = True
            else:
                result['unassigned'].append(wolf)
        
        # Обработка коров
        for cow in cows:
            zone, idx = self._find_nearest_free_zone(cow, 'cows')
            if zone is not None:
                result['assigned'].append((cow, zone))
                self._occupied_zones['cows'][idx] = True
            else:
                result['unassigned'].append(cow)
        
        return result

    def reset_zones(self):
        """Сброс статуса занятости всех загонов"""
        self._occupied_zones = {
            'cows': [False, False, False],
            'wolf': [False]
        }

    def show_frame(self, window_name: str = 'Detection'):
        """Отображение последнего кадра с аннотациями"""
        if self._last_frame is not None:
            # Добавление статистики
            self._add_stats_to_frame()
            
            # Отображение
            cv2.imshow(window_name, self._last_frame)
            cv2.waitKey(1)


    def _capture_frames(self):
        """Поток захвата кадров с камеры"""
        while not self._stop_event.is_set():
            frame = self.camera.get_cv_frame()
            if frame is not None and self._frame_queue.empty():
                self._frame_queue.put(frame)
            time.sleep(0.01)


    def _process_frames(self):
        """Поток обработки кадров"""
        while not self._stop_event.is_set():
            if self._frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame = self._frame_queue.get()
            start_time = time.time()
            
            # Детекция объектов
            annotated_frame, detections = self.detector.detect(frame)
            self._last_frame = annotated_frame
            
            # Обновление буфера детекций
            self._update_detection_buffer(detections)
            
            # Сохранение результатов
            if detections:
                self._results_queue.put({
                    'timestamp': time.time(),
                    'detections': detections,
                    'frame': annotated_frame
                })
            
            # Обновление статистики
            self._update_stats(start_time)

    def _print_position(self):
        """Поток вывода текущих координат дрона"""
        while not self._stop_event.is_set():
            pos = self.pioneer.position
            print(f"Position: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
            time.sleep(0.2)

    def _report_detections_and_zones(self):
        """Периодический вывод глобальных координат объектов и распределения по загонам"""
        while not self._stop_event.is_set():
            try:
                detections_with_coords = self.get_last_detections_with_coords()
                if detections_with_coords:
                    # Печать глобальных координат
                    printed_any = False
                    for frame in detections_with_coords:
                        for det in frame['detections']:
                            gx, gy = det.get('global_coords', (None, None))
                            if gx is not None and gy is not None:
                                if not printed_any:
                                    print("Глобальные координаты обнаруженных объектов:")
                                    printed_any = True
                                print(f"- {det['class_name']}: ({gx:.2f}, {gy:.2f})")

                    # Распределение по зонам (сброс на текущую итерацию)
                    self.reset_zones()
                    assignment = self.assign_to_zones(detections_with_coords)
                    cows_zone_counts = [0] * len(self.ZONES['cows'])
                    wolf_zone_counts = [0] * len(self.ZONES['wolf'])

                    for animal, zone in assignment['assigned']:
                        if animal['class_name'] in self.COW_CLASSES:
                            if zone in self.ZONES['cows']:
                                idx = self.ZONES['cows'].index(zone)
                                cows_zone_counts[idx] += 1
                        elif animal['class_name'] in self.WOLF_CLASS:
                            if zone in self.ZONES['wolf']:
                                idx = self.ZONES['wolf'].index(zone)
                                wolf_zone_counts[idx] += 1

                    if printed_any:
                        print("Распределение по загонам:")
                        print(f"- Коровы: {cows_zone_counts}")
                        print(f"- Волк: {wolf_zone_counts}")
                        if assignment['unassigned']:
                            not_assigned = [a['class_name'] for a in assignment['unassigned']]
                            print(f"- Без назначения: {not_assigned}")
            except Exception as e:
                print(f"Ошибка отчета детекций: {str(e)}")

            time.sleep(0.5)

    def _update_detection_buffer(self, detections: List[Dict]):
        """Обновление буфера детекций"""
        current_classes = {d['class_name'] for d in detections}
        self._detection_buffer.append(current_classes)
        
        # Анализ подтвержденных объектов
        class_counts = defaultdict(int)
        for frame_detections in self._detection_buffer:
            for class_name in frame_detections:
                class_counts[class_name] += 1
                
        self._confirmed_objects = [
            cls for cls, cnt in class_counts.items() 
            if cnt >= self.min_detections
        ]

    def _update_stats(self, process_start: float):
        """Обновление статистики производительности"""
        self._frame_counter += 1
        self._stats['detection_time'] = time.time() - process_start
        self._stats['processed_frames'] += 1
        
        # Обновление FPS
        current_time = time.time()
        if current_time - self._last_fps_update >= 1.0:
            self._stats['current_fps'] = self._frame_counter / (current_time - self._last_fps_update)
            self._frame_counter = 0
            self._last_fps_update = current_time

    def _add_stats_to_frame(self):
        """Добавление статистики на кадр"""
        stats_text = [
            f"FPS: {self._stats['current_fps']:.1f}",
            f"Frames: {self._stats['processed_frames']}",
            f"Latency: {self._stats['detection_time']*1000:.1f}ms"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(
                self._last_frame, text, (10, 30 + i*30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    def _find_nearest_free_zone(self, animal: Dict, animal_type: str) -> Tuple[Optional[Tuple], int]:
        """
        Поиск ближайшего свободного загона
        
        Args:
            animal: Данные животного
            animal_type: Тип животного ('cows' или 'wolf')
            
        Returns:
            Tuple: (координаты зоны, индекс) или (None, -1)
        """
        if animal_type not in self.ZONES:
            return None, -1
            
        current_pos = animal['global_coords']
        min_dist = float('inf')
        best_zone = None
        best_idx = -1
        
        for i, zone_pos in enumerate(self.ZONES[animal_type]):
            if not self._occupied_zones[animal_type][i]:
                dist = self._calculate_distance(current_pos, zone_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_zone = zone_pos
                    best_idx = i
                    
        return best_zone, best_idx

    def _calculate_distance(self, pos1: Tuple[float], pos2: Tuple[float]) -> float:
        """Расчет расстояния между точками"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5