# main.py
import time
import threading
from typing import List, Tuple
from pion import Pion
from flight_utils import load_flight_coordinates
from object_detection_controller_draft import ObjectDetectionController
import cv2

class MissionController:
    def __init__(self, drone: Pion):
        self.drone = drone
        self.mission_points = []
        self.stop_event = threading.Event()
        self.current_point_index = 0

    def load_mission(self, points: List[Tuple[float, float, float]]):
        self.mission_points = points


    def execute_mission_non_blocking(self, stop_event: threading.Event, 
                               detection_event: threading.Event, 
                               stop_on_detection: bool,
                               pioneer: Pion = None):
        """
        Выполняет миссию в неблокирующем режиме с возможностью посадки дрона
        
        Args:
            stop_event: событие для остановки миссии
            detection_event: событие обнаружения объекта
            stop_on_detection: флаг остановки при обнаружении
            pioneer: объект дрона Pioneer (опционально)
        """
        if not self.mission_points:
            print("Ошибка: маршрут не загружен!")
            return

        # Сохраняем ссылку на событие детекции для сброса
        self._detection_event = detection_event

        try:
            print(f"Начало миссии ({len(self.mission_points)} точек)")
            
            while self.current_point_index < len(self.mission_points) and not stop_event.is_set():
                if stop_on_detection and detection_event.is_set():
                    print("Обнаружена корова - начинаем логику сближения")
                    self._approach_cow_logic(stop_event)
                    # После сближения продолжаем миссию с текущей точки
                    print(f"Продолжаем миссию с точки {self.current_point_index + 1}")
                    continue

                x, y, z = self.mission_points[self.current_point_index]
                print(f"Перелет {self.current_point_index + 1}/{len(self.mission_points)} -> X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}")
                
                self.drone.goto(x, y, z, yaw=0, wait=True, accuracy=0.2)
                self.current_point_index += 1

                # короткая неблокирующая пауза с проверкой событий
                start_time = time.time()
                while time.time() - start_time < 0.1:
                    if stop_event.is_set() or (stop_on_detection and detection_event.is_set()):
                        break
                    time.sleep(0.01)

            # Логика завершения миссии
            if not detection_event.is_set() and self.current_point_index >= len(self.mission_points):
                print("Миссия успешно завершена")
                if pioneer is not None:
                    try:
                        print("Выполняем посадку...")
                        pioneer.land()
                        time.sleep(3)  # Даем время на посадку
                    except Exception as e:
                        print(f"Ошибка при посадке: {e}")

        except Exception as e:
            print(f"Ошибка выполнения миссии: {e}")
            if pioneer is not None:
                try:
                    print("Аварийная посадка...")
                    pioneer.land()
                except Exception as e:
                    print(f"Ошибка при аварийной посадке: {e}")



    def _approach_cow_logic(self, stop_event: threading.Event):
        """Логика сближения с коровой для направления её в загон."""
        try:
            # Получаем текущую позицию дрона
            drone_x, drone_y, drone_z = self.drone.position[0], self.drone.position[1], self.drone.position[2]
            print(f"Позиция дрона: X:{drone_x:.2f}, Y:{drone_y:.2f}, Z:{drone_z:.2f}")
            
            # Определяем ближайший загон для коров
            cow_zones = [
                (-2.88, -3.32),  # Загон 1 для коров
                (3.55, -2.85),   # Загон 2 для коров
                (-2.83, 1.61)    # Загон 3 для коров
            ]
            
            # Находим ближайший загон
            nearest_zone = None
            min_distance = float('inf')
            for i, (zone_x, zone_y) in enumerate(cow_zones):
                distance = ((drone_x - zone_x) ** 2 + (drone_y - zone_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_zone = (zone_x, zone_y, i + 1)
            
            if nearest_zone is None:
                print("Ошибка: не удалось определить загон")
                return
            
            zone_x, zone_y, zone_num = nearest_zone
            print(f"Выбран загон #{zone_num} в координатах ({zone_x:.2f}, {zone_y:.2f})")
            
            # Получаем координаты коровы из детектора
            if hasattr(self, '_detector') and self._detector is not None:
                detections = self._detector.get_latest_detections_with_coords()
                if detections and detections['detections']:
                    cow_det = next((d for d in detections['detections'] if d['class_name'] in self._detector.COW_CLASSES), None)
                    if cow_det:
                        cow_x, cow_y = cow_det['global_coords']
                        print(f"Используем координаты коровы из детектора: ({cow_x:.2f}, {cow_y:.2f})")
                    else:
                        cow_x, cow_y = drone_x, drone_y
                        print("Корова не обнаружена, используем позицию дрона")
                else:
                    cow_x, cow_y = drone_x, drone_y
                    print("Нет данных детекции, используем позицию дрона")
            else:
                cow_x, cow_y = drone_x, drone_y
                print("Детектор недоступен, используем позицию дрона")
            
            # Вычисляем вектор от коровы к загону
            dx = zone_x - cow_x
            dy = zone_y - cow_y
            
            # Нормализуем вектор
            distance_to_zone = (dx ** 2 + dy ** 2) ** 0.5
            if distance_to_zone > 0:
                dx_norm = dx / distance_to_zone
                dy_norm = dy / distance_to_zone
            else:
                dx_norm, dy_norm = 0, 0
            
            # Позиция для сближения - на одной прямой с коровой и загоном, но сзади коровы
            approach_distance = 1.5  # расстояние сзади коровы
            approach_x = cow_x - dx_norm * approach_distance
            approach_y = cow_y - dy_norm * approach_distance
            
            print(f"Корова: ({cow_x:.2f}, {cow_y:.2f})")
            print(f"Загон: ({zone_x:.2f}, {zone_y:.2f})")
            print(f"Позиция для сближения: X:{approach_x:.2f}, Y:{approach_y:.2f}")
            
            # 1. Подлетаем к позиции сближения на высоте 1.5м
            print("1. Перелет к позиции сближения на высоте 1.5м...")
            self.drone.goto(approach_x, approach_y, 1.5, yaw=0, wait=True, accuracy=0.2)
            
            if stop_event.is_set():
                return
            
            # 2. Сразу летим к корове на высоте 0.4м
            print("2. Перелет к корове на высоте 0.4м...")
            self.drone.goto(cow_x, cow_y, 0.7, yaw=0, wait=True, accuracy=0.2)
            
            if stop_event.is_set():
                return
            
            # 3. Поднимаемся вверх для запугивания
            print("3. Подъем вверх для запугивания коровы...")
            self.drone.goto(cow_x, cow_y, 1.5, yaw=0, wait=True, accuracy=0.2)
            
            print("Логика сближения завершена - корова должна направиться в загон")
            
            # Сбрасываем событие детекции, чтобы продолжить миссию
            if hasattr(self, '_detection_event'):
                self._detection_event.clear()
                print("Детекция сброшена - продолжаем миссию")
            
        except Exception as e:
            print(f"Ошибка в логике сближения: {e}")

    def stop_mission(self):
        self.stop_event.set()
        self.drone.stop_moving()


def display_and_process_detections(detector: ObjectDetectionController, pioneer: Pion, stop_event):
    """Цикл отображения и обработки детекций"""
    while not stop_event.is_set():
        # Отображаем кадр с детекциями
        detector.show_frame()
        
        # Выход по клавише q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
        
        time.sleep(0.1)

def cow_detection_loop(detector: ObjectDetectionController, detection_event: threading.Event):
    """Поток для детекции коров и установки события детекции."""
    while not detection_event.is_set():
        if detector.is_confirmed():
            detection_event.set()
            break
        time.sleep(0.1)

def main():
    # Инициализация дрона
    pioneer = Pion(ip="127.0.0.1", mavlink_port=8004, logger=True, dt=0.0, mass=0.5)
    
    # Конфигурация детекции
    # CAMERA_SOURCE = "rtsp://10.1.100.160:8554/pioneer_stream"  # или путь к видеофайлу
    CAMERA_SOURCE = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/videos/output_pascal_line2.mp4"  # или путь к видеофайлу

    MODEL_PATH = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/weights/best_last_sana.pt"  # путь к модели YOLO
    

    # Единое стоп-событие для всех потоков
    stop_event = threading.Event()
    detection_event = threading.Event()

    # Инициализация контроллера детекции
    detector = ObjectDetectionController(
        pioneer=pioneer,
        camera_source=CAMERA_SOURCE,
        model_path=MODEL_PATH,
        conf_threshold=0.6, 
        min_detections=5,
        buffer_size=10
    )
    detector.start()

    # Поток отображения и обработки детекций
    display_thread = threading.Thread(
        target=display_and_process_detections,
        args=(detector, pioneer, stop_event),
        daemon=True
    )
    display_thread.start()

    # Поток для детекции коров
    detection_thread = threading.Thread(
        target=cow_detection_loop,
        args=(detector, detection_event),
        daemon=True
    )
    detection_thread.start()

    try:
       # Инициализация миссии
        mission = MissionController(drone=pioneer)
        mission._detector = detector  # Передаем детектор в миссию
        flight_path = load_flight_coordinates("flight_path.json")
        mission.load_mission(flight_path)

        


        print("Армирование дрона...")
        pioneer.arm()
        print("Взлет...")
        pioneer.takeoff()
        time.sleep(7)


        # Запускаем миссию в отдельном потоке
        mission_thread = threading.Thread(
            target=mission.execute_mission_non_blocking,
            kwargs={
                'stop_event': stop_event,
                'detection_event': detection_event,
                'stop_on_detection': True,
                'pioneer': pioneer  # Передаем объект пионера
            },
            daemon=True
        )
        mission_thread.start()

       

        # Основной цикл ожидания завершения миссии или команды остановки
        while mission_thread.is_alive() and not stop_event.is_set():
            time.sleep(0.1)

        # Ждем завершения потока миссии
        if 'mission_thread' in locals():
            mission_thread.join(timeout=30)

        print("Посадка...")
        pioneer.land()
        time.sleep(3)

    except KeyboardInterrupt:
        print("\nМиссия прервана пользователем")
        stop_event.set()
    except Exception as e:
        print(f"\nОшибка: {str(e)}")
        stop_event.set()
    finally:
        # Остановка всех потоков
        stop_event.set()
        detector.stop()

        if 'display_thread' in locals():
            display_thread.join(timeout=1)
        if 'mission_thread' in locals():
            mission_thread.join(timeout=1)
        if 'detection_thread' in locals():
            detection_thread.join(timeout=1)

        print("Завершение работы...")
        try:
            pioneer.disarm()
        except Exception:
            pass
        try:
            pioneer.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()