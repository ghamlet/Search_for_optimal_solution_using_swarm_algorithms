import time
import threading
from typing import List, Tuple
from pion import Pion
from flight_utils import load_flight_coordinates
from object_detection_controller_draft import ObjectDetectionController
import numpy as np
import cv2

class MissionController:
    def __init__(self, drone: Pion):
        self.drone = drone
        self.mission_points = []
        self.stop_event = threading.Event()

    def load_mission(self, points: List[Tuple[float, float, float]]):
        self.mission_points = points

    def execute_mission(self):
        if not self.mission_points:
            print("Ошибка: маршрут не загружен!")
            return

        try:
            print(f"Начало миссии ({len(self.mission_points)} точек)")

            for i, (x, y, z) in enumerate(self.mission_points, 1):
                if self.stop_event.is_set():
                    print("Миссия прервана")
                    break

                print(f"Перелет {i}/{len(self.mission_points)} -> X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}")
                self.drone.goto(x, y, z, yaw=0, wait=True, accuracy=0.2)
                time.sleep(0.1)

            print("Миссия успешно завершена")

        except Exception as e:
            print(f"Ошибка выполнения миссии: {str(e)}")
            raise

    def stop_mission(self):
        self.stop_event.set()
        self.drone.stop_moving()






def display_and_process_detections(detector: ObjectDetectionController, pioneer: Pion, stop_event):
    """Цикл отображения и обработки детекций"""
    while not stop_event.is_set():
        # Отображаем кадр с детекциями
        detector.show_frame()
        
        # Получаем и обрабатываем детекции
        detections = detector.get_last_detections_with_coords()
        if detections:
            assignment = detector.assign_to_zones(detections)
            for animal, zone in assignment['assigned']:
                print(f"Назначено: {animal['class_name']} в зону {zone}")
        
        # Выход по клавише q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
        
        time.sleep(0.1)



def execute_mission_non_blocking(mission: MissionController, stop_event):
    """Неблокирующее выполнение миссии"""
    if not mission.mission_points:
        print("Ошибка: маршрут не загружен!")
        return

    try:
        print(f"Начало миссии ({len(mission.mission_points)} точек)")
        
        for i, (x, y, z) in enumerate(mission.mission_points, 1):
            if stop_event.is_set():
                print("Миссия прервана")
                break

            print(f"Перелет {i}/{len(mission.mission_points)} -> X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}")
            mission.drone.goto(x, y, z, yaw=0, wait=True, accuracy=0.2)
            
            # Короткая задержка вместо time.sleep(0.1)
            start_time = time.time()
            while time.time() - start_time < 0.1 and not stop_event.is_set():
                time.sleep(0.01)

        print("Миссия успешно завершена")

    except Exception as e:
        print(f"Ошибка выполнения миссии: {str(e)}")
        raise





def main():
    # Инициализация дрона
    pioneer = Pion(ip="127.0.0.1", mavlink_port=8000, logger=True, dt=0.0, mass=0.5)
    # pioneer = Pion(ip="10.1.100.160", mavlink_port=5656, logger=True, dt=0.0, mass=0.5)

    
    # Конфигурация детекции
    CAMERA_SOURCE = "rtsp://10.1.100.160:8554/pioneer_stream"
    # CAMERA_SOURCE = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/videos/output_pascal_line2.mp4"  # сократил путь для читаемости
    MODEL_PATH = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/weights/best_last_sana.pt"
    
    # Единое стоп-событие для всех потоков
    stop_event = threading.Event()

    

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

    try:
        # Инициализация миссии
        mission = MissionController(drone=pioneer)
        flight_path = load_flight_coordinates("flight_path.json")
        mission.load_mission(flight_path)

        print("Армирование дрона...")
        pioneer.arm()
        print("Взлет...")
        pioneer.takeoff()
        time.sleep(2)

        # Запуск миссии в отдельном потоке
        mission_thread = threading.Thread(
            target=execute_mission_non_blocking,
            args=(mission, stop_event),
            daemon=True
        )
        mission_thread.start()

        # Основной цикл ожидания завершения миссии или команды остановки
        while mission_thread.is_alive() and not stop_event.is_set():
            time.sleep(0.1)

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