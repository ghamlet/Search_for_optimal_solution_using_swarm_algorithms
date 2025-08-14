# main.py (updated version)
import time
import threading
import json
from typing import List, Tuple
from pion import Pion
from object_detection_controller_draft import ObjectDetectionController
from camera_real import CameraReal
import cv2

class MissionController:
    def __init__(self, drone: Pion):
        self.drone = drone
        self.mission_points = []
        self.stop_event = threading.Event()
        self.current_point_index = 0
        self._detector = None
        self._detection_event = threading.Event()

    def load_mission(self, points: List[Tuple[float, float, float]]):
        self.mission_points = points

    def set_detector(self, detector: ObjectDetectionController):
        self._detector = detector
        self._detector.COW_CLASSES = ['cow', 'cattle', 'animal']  # Определяем классы коров

    def execute_mission(self, stop_on_detection: bool = True):
        """Основной метод выполнения миссии"""
        if not self.mission_points:
            print("Ошибка: маршрут не загружен!")
            return

        try:
            print(f"Начало миссии ({len(self.mission_points)} точек)")
            
            while self.current_point_index < len(self.mission_points) and not self.stop_event.is_set():
                # Проверка обнаружения коровы
                if stop_on_detection and self._detection_event.is_set():
                    print("Обнаружена корова - начинаем логику сближения")
                    self._approach_cow_logic()
                    self._detection_event.clear()
                    print("Продолжаем миссию с текущей точки")
                    continue

                x, y, z = self.mission_points[self.current_point_index]
                print(f"Перелет {self.current_point_index + 1}/{len(self.mission_points)} -> X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}")
                
                self.drone.goto(x, y, z, yaw=0, wait=True, accuracy=0.2)
                self.current_point_index += 1

                # Короткая пауза с проверкой событий
                time.sleep(0.1)

            # Завершение миссии
            if self.current_point_index >= len(self.mission_points):
                print("Миссия успешно завершена")
                self._land_drone()

        except Exception as e:
            print(f"Ошибка выполнения миссии: {e}")
            self._emergency_land()

    def _approach_cow_logic(self):
        """Логика сближения с коровой"""
        try:
            # Получаем текущую позицию дрона
            drone_pos = self.drone.position
            if drone_pos is None:
                print("Не удалось получить позицию дрона")
                return

            drone_x, drone_y, drone_z = drone_pos[0], drone_pos[1], drone_pos[2]
            print(f"Позиция дрона: X:{drone_x:.2f}, Y:{drone_y:.2f}, Z:{drone_z:.2f}")
            
            # Получаем координаты коровы из детектора
            if self._detector is not None:
                detections = self._detector.get_latest_detections_with_coords()
                if detections and detections.get('detections'):
                    cow_det = next((d for d in detections['detections'] 
                                  if d['class_name'] in getattr(self._detector, 'COW_CLASSES', [])), None)
                    if cow_det:
                        cow_x, cow_y = cow_det['global_coords']
                        print(f"Координаты коровы: X:{cow_x:.2f}, Y:{cow_y:.2f}")
                    else:
                        cow_x, cow_y = drone_x, drone_y
                        print("Не удалось получить координаты коровы, используем позицию дрона")
                else:
                    cow_x, cow_y = drone_x, drone_y
                    print("Нет данных детекции, используем позицию дрона")
            else:
                cow_x, cow_y = drone_x, drone_y
                print("Детектор недоступен, используем позицию дрона")
            
            # Определяем ближайший загон
            cow_zones = [
                (-2.88, -3.32),  # Загон 1
                (3.55, -2.85),   # Загон 2
                (-2.83, 1.61)    # Загон 3
            ]
            
            nearest_zone = min(
                cow_zones,
                key=lambda zone: ((drone_x - zone[0])**2 + (drone_y - zone[1])**2)**0.5
            )
            
            print(f"Ближайший загон: {nearest_zone}")
            
            # Вычисляем вектор от коровы к загону
            dx = nearest_zone[0] - cow_x
            dy = nearest_zone[1] - cow_y
            
            # Нормализуем вектор
            distance = (dx**2 + dy**2)**0.5
            if distance > 0:
                dx_norm = dx / distance
                dy_norm = dy / distance
            else:
                dx_norm, dy_norm = 0, 0
            
            # Позиция для сближения - за коровой по направлению к загону
            approach_distance = 1.5
            approach_x = cow_x - dx_norm * approach_distance
            approach_y = cow_y - dy_norm * approach_distance
            
            # 1. Перелет к позиции сближения
            print("1. Перелет к позиции сближения...")
            self.drone.goto(approach_x, approach_y, 1.5, yaw=0, wait=True, accuracy=0.2)
            
            if self.stop_event.is_set():
                return
            
            # 2. Перелет к корове на малой высоте
            print("2. Перелет к корове...")
            self.drone.goto(cow_x, cow_y, 0.7, yaw=0, wait=True, accuracy=0.2)
            
            if self.stop_event.is_set():
                return
            
            # 3. Подъем для запугивания коровы
            print("3. Подъем для запугивания...")
            self.drone.goto(cow_x, cow_y, 1.5, yaw=0, wait=True, accuracy=0.2)
            
            print("Логика сближения завершена")

        except Exception as e:
            print(f"Ошибка в логике сближения: {e}")

    def _land_drone(self):
        """Посадка дрона"""
        try:
            print("Выполняем посадку...")
            self.drone.land()
            time.sleep(3)
        except Exception as e:
            print(f"Ошибка при посадке: {e}")

    def _emergency_land(self):
        """Аварийная посадка"""
        try:
            print("Аварийная посадка...")
            self.drone.land()
        except Exception as e:
            print(f"Ошибка при аварийной посадке: {e}")

    def stop_mission(self):
        """Остановка миссии"""
        self.stop_event.set()
        self.drone.stop_moving()

def detection_thread(detector: ObjectDetectionController, 
                    camera: CameraReal,
                    detection_event: threading.Event,
                    stop_event: threading.Event,
                    pioneer: Pion):
    """Поток для обработки детекции"""
    while not stop_event.is_set():
        frame = camera.get_cv_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Получаем позицию дрона
        try:
            drone_pos = pioneer.position
            if drone_pos is None:
                drone_pos = (0, 0, 0)  # Заглушка, если позиция недоступна
        except:
            drone_pos = (0, 0, 0)
        
        # Обрабатываем кадр
        result = detector.process_frame(frame, drone_pos[0], drone_pos[1])
        
        # Если обнаружена корова - устанавливаем событие
        if result.get('confirmed', False) and any(cow_word in result.get('type', '').lower() 
                                               for cow_word in ['cow', 'cattle', 'animal']):
            detection_event.set()
        
        # Отображаем результат
        annotated_frame = result.get('annotated', frame)
        cv2.imshow('Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

def load_flight_coordinates(filename: str) -> List[Tuple[float, float, float]]:
    """Загрузка координат полета из JSON файла"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return [(point['x'], point['y'], point['z']) for point in data['points']]
    except Exception as e:
        print(f"Ошибка загрузки координат: {e}")
        return []

def main():
    # Инициализация дрона
    pioneer = Pion(ip="127.0.0.1", mavlink_port=8004, logger=True, dt=0.0, mass=0.5)
    
    # Инициализация камеры
    camera_source = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/videos/output_pascal_line2.mp4"
    camera = CameraReal(camera_source)
    
    # Инициализация детектора
    model_path = "/полный/путь/к/вашей/модели.pt"

    detector = ObjectDetectionController(
        camera_source=CAMERA_SOURCE,
        model_path=model_path,  # Передаем путь к модели
        yolo_conf_threshold=0.6,  # Порог уверенности для YOLO
        min_detections=5,
        buffer_size=10,
        use_yolo=True  # Убедитесь, что этот параметр установлен в True
    )
    
    # События для управления потоками
    stop_event = threading.Event()
    detection_event = threading.Event()
    
    # Загрузка маршрута
    mission = MissionController(pioneer)
    mission.set_detector(detector)
    flight_path = load_flight_coordinates("/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/flight_path.json")
    if not flight_path:
        print("Не удалось загрузить маршрут, используем тестовые точки")
        flight_path = [(0, 0, 1), (5, 0, 1), (5, 5, 1), (0, 5, 1)]
    mission.load_mission(flight_path)
    
    # Поток детекции
    det_thread = threading.Thread(
        target=detection_thread,
        args=(detector, camera, detection_event, stop_event, pioneer),
        daemon=True
    )
    det_thread.start()
    
    try:
        # Армирование и взлет
        print("Армирование дрона...")
        pioneer.arm()
        print("Взлет...")
        pioneer.takeoff()
        time.sleep(5)
        
        # Запуск миссии
        mission.execute_mission(stop_on_detection=True)
        
    except KeyboardInterrupt:
        print("\nМиссия прервана пользователем")
        stop_event.set()
    except Exception as e:
        print(f"\nОшибка: {e}")
        stop_event.set()
    finally:
        # Остановка всех процессов
        stop_event.set()
        mission.stop_mission()
        
        # Посадка
        try:
            pioneer.land()
            time.sleep(3)
        except Exception as e:
            print(f"Ошибка при посадке: {e}")
        
        # Освобождение ресурсов
        camera.release()
        cv2.destroyAllWindows()
        
        # Ожидание завершения потока детекции
        det_thread.join(timeout=1)
        
        print("Программа завершена")

if __name__ == "__main__":
    main()