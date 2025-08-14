import time
import threading
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pion import Pion
from flight_utils import load_flight_coordinates
from object_detection_controller_draft_old import ObjectDetectionController
from camera_real import CameraReal


class MissionController:
    # Константы для управления полетом
    MAIN_FLIGHT_HEIGHT = 1.5  # Основная высота полета (м)
    SCARE_HEIGHT = 0.7        # Высота для запугивания животных (м)
    APPROACH_DISTANCE = 1.5   # Дистанция сближения с животным (м)
    MAX_SCARE_ATTEMPTS = 4    # Максимальное количество попыток сближения
    DETECTION_COOLDOWN = 5   # Время блокировки повторной детекции (сек)
    COW_PROCESS_RADIUS = 3.0  # Радиус для определения уже обработанной коровы (м)

    def __init__(self, drone: Pion):
        self.drone = drone
        self.mission_points: List[Tuple[float, float, float]] = []
        self.current_point_index = 0
        self.last_detection_time = 0
        self._detector = None
        self._detection_event = threading.Event()
        
        # Трекинг животных
        self.processed_animals: Dict[str, Tuple[Tuple[float, float], str, float]] = {}
        self.current_animal_id = 0

    def load_mission(self, points: List[Tuple[float, float, float]]):
        """Загрузка точек маршрута"""
        self.mission_points = points

    def set_detector(self, detector: ObjectDetectionController):
        """Установка детектора для получения координат животных"""
        self._detector = detector

    def execute_mission(self, stop_on_detection: bool = True):
        """Основной метод выполнения миссии"""
        if not self.mission_points:
            print("Ошибка: маршрут не загружен!")
            return

        try:
            print(f"Начало миссии ({len(self.mission_points)} точек)")
            
            while self.current_point_index < len(self.mission_points) and not self._detection_event.is_set():
                # Проверка обнаружения животного
                if stop_on_detection and self._check_animal_detection():
                    print("Обнаружено животное - начинаем логику сближения")
                    self._approach_animal_logic()
                    print("Продолжаем миссию с текущей точки")
                    continue

                # Перелет к следующей точке
                x, y, z = self.mission_points[self.current_point_index]
                print(f"Перелет {self.current_point_index + 1}/{len(self.mission_points)} -> X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}")
                
                self.drone.goto(x, y, z, yaw=0, wait=True, accuracy=0.2)
                self.current_point_index += 1

                # Неблокирующая пауза с проверкой событий
                self._non_blocking_sleep(0.1)

            if self.current_point_index >= len(self.mission_points):
                print("Миссия успешно завершена")
                self._print_statistics()

        except Exception as e:
            print(f"Ошибка выполнения миссии: {e}")
            self._emergency_procedures()

    def _check_animal_detection(self) -> bool:
        """Проверка обнаружения животного с учетом времени блокировки"""
        current_time = time.time()
        if not self._detection_event.is_set():
            return False
        if current_time - self.last_detection_time < self.DETECTION_COOLDOWN:
            self._detection_event.clear()
            return False
        
        # Получаем координаты животного
        animal_coords = self._get_animal_coordinates()
        if not animal_coords:
            self._detection_event.clear()
            return False
            
        # Проверяем, не было ли это животное уже обработано
        if self._is_animal_processed(animal_coords):
            print("Животное уже было обработано ранее, игнорируем")
            self._detection_event.clear()
            return False
            
        self.last_detection_time = current_time
        return True

    def _get_animal_coordinates(self) -> Optional[Tuple[float, float]]:
        """Получение глобальных координат животного"""
        if self._detector is None:
            return None
            
        detections = self._detector.get_last_detections_with_coords()
        if not detections:
            return None
            
        # Ищем первое подходящее животное
        for frame in detections:
            for det in frame['detections']:
                if det['class_name'] in self._detector.COW_CLASSES + self._detector.WOLF_CLASS:
                    return det['global_coords']
        return None

    def _is_animal_processed(self, coords: Tuple[float, float]) -> bool:
        """Проверяет, было ли животное в этой области уже обработано"""
        for animal_data in self.processed_animals.values():
            (stored_coords, status, timestamp), _ = animal_data
            distance = self._calculate_distance(coords, stored_coords)
            
            if distance <= self.COW_PROCESS_RADIUS and time.time() - timestamp < 600:
                return True
        return False

    def _approach_animal_logic(self):
        """Логика сближения с животным и направления его в загон"""
        try:
            # Получаем текущую позицию дрона и координаты животного
            drone_pos = self.drone.position
            animal_coords = self._get_animal_coordinates()
            
            if None in (drone_pos, animal_coords):
                print("Не удалось получить позиции для сближения")
                return

            drone_x, drone_y, drone_z = drone_pos
            animal_x, animal_y = animal_coords
            
            # Определяем тип животного и соответствующий загон
            animal_type = self._determine_animal_type(animal_coords)
            target_zone = self._select_target_zone(animal_type, drone_pos)
            
            if target_zone is None:
                print("Не удалось определить целевой загон")
                return
                
            zone_x, zone_y, zone_name = target_zone
            
            # Вычисляем вектор от животного к загону
            dx, dy = zone_x - animal_x, zone_y - animal_y
            distance = (dx**2 + dy**2)**0.5
            dx_norm, dy_norm = dx/distance, dy/distance if distance > 0 else (0, 0)
            
            # Рассчитываем yaw для направления к животному
            yaw = np.degrees(np.arctan2(animal_y - drone_y, animal_x - drone_x))
            
            # Позиция для сближения - за животным по направлению к загону
            approach_x = animal_x - dx_norm * self.APPROACH_DISTANCE
            approach_y = animal_y - dy_norm * self.APPROACH_DISTANCE
            
            # Выполняем последовательность сближения
            for attempt in range(1, self.MAX_SCARE_ATTEMPTS + 1):
                print(f"\nПопытка сближения #{attempt}/{self.MAX_SCARE_ATTEMPTS}")
                
                # 1. Перелет к позиции сближения
                print(f"1. Перелет к позиции сближения ({approach_x:.2f}, {approach_y:.2f})")
                self.drone.goto(approach_x, approach_y, self.MAIN_FLIGHT_HEIGHT, yaw=yaw, wait=True, accuracy=0.2)
                
                if self._detection_event.is_set():
                    return
                
                # 2. Перелет к животному на малой высоте
                print(f"2. Перелет к животному ({animal_x:.2f}, {animal_y:.2f})")
                self.drone.goto(animal_x, animal_y, self.SCARE_HEIGHT, yaw=yaw, wait=True, accuracy=0.2)
                
                if self._detection_event.is_set():
                    return
                
                # 3. Подъем для запугивания
                print("3. Подъем для запугивания")
                self.drone.goto(animal_x, animal_y, self.MAIN_FLIGHT_HEIGHT, yaw=yaw, wait=True, accuracy=0.2)
                
                # Проверяем результат
                if self._is_animal_in_zone(animal_coords, target_zone):
                    print(f"Животное направлено в загон {zone_name}!")
                    break
                
                time.sleep(1)  # Пауза между попытками
            
            # Фиксируем результат обработки животного
            self._record_animal_processing(animal_coords, attempt)
            
        except Exception as e:
            print(f"Ошибка в логике сближения: {e}")
        finally:
            self._detection_event.clear()

    def _determine_animal_type(self, coords: Tuple[float, float]) -> str:
        """Определяет тип животного по координатам (используя детектор)"""
        if self._detector is None:
            return "cow"
            
        detections = self._detector.get_last_detections_with_coords()
        if not detections:
            return "cow"
            
        for frame in detections:
            for det in frame['detections']:
                if det['global_coords'] == coords:
                    return "wolf" if det['class_name'] in self._detector.WOLF_CLASS else "cow"
        return "cow"

    def _select_target_zone(self, animal_type: str, drone_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, str]]:
        """Выбирает целевой загон для животного"""
        if self._detector is None:
            zones = {
                "cow": [(-2.88, -3.32), (3.55, -2.85), (-2.83, 1.61)],
                "wolf": [(2.75, 3.02)]
            }.get(animal_type, [])
        else:
            zones = self._detector.ZONES.get("cows" if animal_type == "cow" else "wolf", [])
        
        if not zones:
            return None
            
        # Находим ближайший загон к текущей позиции дрона
        drone_x, drone_y, _ = drone_pos
        nearest_zone = min(zones, key=lambda z: self._calculate_distance((drone_x, drone_y), z))
        zone_name = f"{animal_type} zone #{zones.index(nearest_zone) + 1}"
        
        return (*nearest_zone, zone_name)

    def _is_animal_in_zone(self, animal_coords: Tuple[float, float], zone: Tuple[float, float, str]) -> bool:
        """Проверяет, находится ли животное в зоне"""
        animal_x, animal_y = animal_coords
        zone_x, zone_y, _ = zone
        zone_radius = 1.5  # Радиус зоны в метрах
        
        distance = self._calculate_distance(animal_coords, (zone_x, zone_y))
        return distance <= zone_radius

    def _record_animal_processing(self, coords: Tuple[float, float], attempts: int):
        """Записывает информацию об обработанном животном"""
        animal_id = f"animal_{self.current_animal_id}"
        self.current_animal_id += 1
        
        status = "herded" if attempts < self.MAX_SCARE_ATTEMPTS else "skipped"
        self.processed_animals[animal_id] = (coords, status, time.time())
        
        print(f"Животное {animal_id} обработано: статус={status}, попыток={attempts}")

    def _print_statistics(self):
        """Выводит статистику по обработанным животным"""
        print("\n=== Статистика миссии ===")
        print(f"Всего обработано животных: {len(self.processed_animals)}")
        
        if self.processed_animals:
            print("\nДетали по животным:")
            for animal_id, (coords, status, _) in self.processed_animals.items():
                print(f"  {animal_id}: ({coords[0]:.2f}, {coords[1]:.2f}) - {status}")
        
        print("========================")

    def _non_blocking_sleep(self, seconds: float):
        """Неблокирующая пауза с проверкой событий"""
        start = time.time()
        while time.time() - start < seconds and not self._detection_event.is_set():
            time.sleep(0.01)

    def _emergency_procedures(self):
        """Процедуры аварийного завершения"""
        print("Аварийные процедуры...")
        try:
            self.drone.stop_moving()
            self.drone.land()
        except Exception as e:
            print(f"Ошибка при аварийной посадке: {e}")

    @staticmethod
    def _calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Вычисляет расстояние между двумя точками"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


def display_and_process_detections(detector: ObjectDetectionController, 
                                 stop_event: threading.Event,
                                 detection_event: threading.Event):
    """Цикл отображения и обработки детекций"""
    while not stop_event.is_set():
        # Отображаем кадр с детекциями
        detector.show_frame()
        
        # Проверяем детекции и устанавливаем событие при обнаружении
        detections = detector.get_last_detections_with_coords()
        if detections:
            for frame in detections:
                for det in frame['detections']:
                    if det['class_name'] in detector.COW_CLASSES + detector.WOLF_CLASS:
                        detection_event.set()
                        break
                if detection_event.is_set():
                    break
        
        # Выход по клавише q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
        
        time.sleep(0.1)


def main():
    # Инициализация дрона
    pioneer = Pion(ip="10.1.100.160", mavlink_port=5656, logger=True, dt=0.0, mass=0.5)
    
    # Конфигурация детекции
    CAMERA_SOURCE = "rtsp://10.1.100.160:8554/pioneer_stream"
    MODEL_PATH = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/weights/best_last_sana.pt"
    
    # Создание и запуск детектора
    detector = ObjectDetectionController(
        pioneer=pioneer,
        camera_source=CAMERA_SOURCE,
        model_path=MODEL_PATH,
        conf_threshold=0.6,
        min_detections=5,
        buffer_size=20
    )
    detector.start()
    
    # События управления потоками
    stop_event = threading.Event()
    detection_event = threading.Event()
    
    # Загрузка маршрута
    mission = MissionController(pioneer)
    flight_path = load_flight_coordinates("flight_path.json")
    if not flight_path:
        print("Не удалось загрузить маршрут, используем тестовые точки")
        flight_path = [(0, 0, 1), (5, 0, 1), (5, 5, 1), (0, 5, 1)]
    mission.load_mission(flight_path)
    mission.set_detector(detector)
    
    # Поток отображения и обработки детекций
    display_thread = threading.Thread(
        target=display_and_process_detections,
        args=(detector, stop_event, detection_event),
        daemon=True
    )
    display_thread.start()
    
    try:
        # Армирование и взлет
        print("Армирование дрона...")
        pioneer.arm()
        print("Взлет...")
        pioneer.takeoff()
        time.sleep(2)
        
        # Запуск миссии
        mission.execute_mission(stop_on_detection=True)
        
    except KeyboardInterrupt:
        print("\nМиссия прервана пользователем")
        stop_event.set()
    except Exception as e:
        print(f"\nОшибка: {str(e)}")
        stop_event.set()
    finally:
        # Остановка всех процессов
        stop_event.set()
        
        # Ожидание завершения потоков
        display_thread.join(timeout=1)
        
        # Посадка
        try:
            pioneer.land()
            time.sleep(3)
        except Exception as e:
            print(f"Ошибка при посадке: {e}")
        
        # Освобождение ресурсов
        detector.stop()
        cv2.destroyAllWindows()
        print("Программа завершена")


if __name__ == "__main__":
    main()