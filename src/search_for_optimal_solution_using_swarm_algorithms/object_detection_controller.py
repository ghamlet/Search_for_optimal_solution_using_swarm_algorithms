import os
import cv2
import numpy as np
import threading
import time
from collections import defaultdict, deque
from typing import List, Dict, Optional, Any
from camera import Camera
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
        Инициализация контроллера детекции объектов
        
        :param pioneer: экземпляр дрона Pion
        :param camera_source: источник видеопотока
        :param model_path: путь к модели YOLO
        :param conf_threshold: порог уверенности для детекции
        :param min_detections: минимальное количество детекций для подтверждения объекта
        :param buffer_size: размер буфера для анализа детекций
        """
        self.pioneer = pioneer
        self.min_detections = min_detections
        self.buffer_size = buffer_size
        
        # Инициализация компонентов
        self.camera = Camera(source=camera_source)
        self.detector = YOLODetector(model_path=model_path, conf_threshold=conf_threshold)
        self.class_names = self.detector.class_names
        
        # Потоковые атрибуты
        self._stop_event = threading.Event()
        self._detection_thread = None
        self._detection_buffer = deque(maxlen=buffer_size)
        self._confirmed_objects = []
        self._last_annotated_frame = None
        self._stats = {
            'processed_frames': 0,
            'fps': 0.0,
            'last_update': time.time()
        }

    def start(self):
        """Запускает поток детекции"""
        if self._detection_thread is None or not self._detection_thread.is_alive():
            self._stop_event.clear()
            self._detection_thread = threading.Thread(
                target=self._run_detection_loop,
                daemon=True
            )
            self._detection_thread.start()
            print("Детекция запущена в фоновом режиме")

    def stop(self):
        """Останавливает поток детекции"""
        self._stop_event.set()
        if self._detection_thread is not None:
            self._detection_thread.join(timeout=1)
        self.camera.release()
        cv2.destroyAllWindows()
        print("Детекция остановлена")

    def get_confirmed_objects(self) -> List[str]:
        """Возвращает список подтвержденных объектов"""
        return self._confirmed_objects.copy()

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Возвращает последний обработанный кадр"""
        return self._last_annotated_frame

    def get_stats(self) -> Dict:
        """Возвращает статистику детекции"""
        return self._stats.copy()

    def _run_detection_loop(self):
        """Основной цикл детекции (работает в отдельном потоке)"""
        last_time = time.time()
        frame_count = 0
        
        while not self._stop_event.is_set():
            try:
                # Получаем кадр
                frame = self.camera.get_cv_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Детекция объектов
                annotated_frame, detections = self.detector.detect(frame)
                self._last_annotated_frame = annotated_frame
                
                # Вывод информации о текущих детекциях
                self._print_detection_info(detections)
                
                # Обновление буфера детекций
                self._update_detection_buffer(detections)
                
                # Анализ буфера для подтверждения объектов
                self._analyze_detections()
                
                # Обновление статистики
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    self._stats['fps'] = frame_count / (current_time - last_time)
                    self._stats['processed_frames'] += frame_count
                    self._stats['last_update'] = current_time
                    frame_count = 0
                    last_time = current_time
                
                # Небольшая пауза для снижения нагрузки
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Ошибка в потоке детекции: {str(e)}")
                time.sleep(1)

    def _print_detection_info(self, detections: List[Dict]):
        """Выводит информацию о детектированных объектах"""
        if detections:
            print("\nТекущие детекции:")
            for det in detections:
                bbox = det['bbox']  # [x1, y1, x2, y2]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                print(f"- Класс: {det['class_name']} | "
                      f"Точность: {det['confidence']:.2f} | "
                      f"Площадь bbox: {area:.1f} пикс. | "
                      f"Координаты: ({bbox[0]:.1f}, {bbox[1]:.1f}) - ({bbox[2]:.1f}, {bbox[3]:.1f})")
        # else:
        #     print("На текущем кадре объекты не обнаружены")

    def _update_detection_buffer(self, detections: List[Dict]):
        """Обновляет буфер последних детекций"""
        current_classes = {d['class_name'] for d in detections}
        self._detection_buffer.append(current_classes)
        
        # Ограничиваем размер буфера
        if len(self._detection_buffer) > self.buffer_size:
            self._detection_buffer.popleft()

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
            else:
                print("\nНет подтвержденных объектов")

    def show_detection(self):
        """Отображает окно с результатами детекции (должен вызываться из основного потока)"""
        while not self._stop_event.is_set():
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
                
                cv2.imshow('Continuous Object Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




# Пример использования
if __name__ == "__main__":
    from pion import Pion  # Импорт должен быть реальным
    
    # Инициализация дрона (замените на реальные параметры)
    pioneer = Pion(ip="127.0.0.1", mavlink_port=8000)
    
    try:
        # Создание контроллера детекции
        detector = ObjectDetectionController(
            pioneer=pioneer,
            camera_source="/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/videos/output_pascal_line2.mp4",  # Веб-камера
            model_path="/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/weights/best_first.pt",
            conf_threshold=0.6,
            min_detections=3,
            buffer_size=15
        )
        
        # Запуск детекции в фоновом режиме
        detector.start()
        
        # Отображение результатов в основном потоке
        detector.show_detection()
        

        
    except KeyboardInterrupt:
        print("\nОстановка по запросу пользователя")
    finally:
        detector.stop()