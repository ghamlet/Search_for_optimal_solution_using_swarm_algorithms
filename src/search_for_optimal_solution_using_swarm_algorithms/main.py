import time
import threading
from typing import List, Tuple
from pion import Pion
from flight_utils import load_flight_coordinates
from object_detection_controller import ObjectDetectionController
import numpy as np

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

def print_position(pioneer, stop_event):
    """Поток для вывода текущих координат"""
    while not stop_event.is_set():
        print(f"Текущие координаты: xyz {np.round(pioneer.position[0:3], 3)}")
        time.sleep(1)

def main():
    # Инициализация дрона
    pioneer = Pion(ip="127.0.0.1", mavlink_port=8000, logger=True)
    
    # Конфигурация детекции
    CAMERA_SOURCE = "/path/to/video.mp4"  # или "rtsp://..."
    MODEL_PATH = "/path/to/model.pt"
    
    # Создаем и запускаем потоки
    position_stop_event = threading.Event()
    detection_stop_event = threading.Event()

    # Поток вывода координат
    position_thread = threading.Thread(
        target=print_position,
        args=(pioneer, position_stop_event),
        daemon=True
    )
    position_thread.start()

    # Контроллер детекции
    detector = ObjectDetectionController(
        pioneer=pioneer,
        camera_source=CAMERA_SOURCE,
        model_path=MODEL_PATH,
        conf_threshold=0.6,
        min_detections=3,
        buffer_size=15
    )

    # Поток детекции
    detection_thread = threading.Thread(
        target=detector.run_detection,
        daemon=True
    )
    detection_thread.start()

    # Поток отображения
    display_thread = threading.Thread(
        target=detector.show_detection,
        daemon=True
    )
    display_thread.start()

    try:
        # Инициализация и выполнение миссии
        mission = MissionController(drone=pioneer)
        flight_path = load_flight_coordinates("flight_path.json")
        mission.load_mission(flight_path)

        print("Армирование дрона...")
        pioneer.arm()
        print("Взлет...")
        pioneer.takeoff()
        time.sleep(2)

        print("Начало миссии")
        mission.execute_mission()

        print("Посадка...")
        pioneer.land()
        time.sleep(3)

    except KeyboardInterrupt:
        print("\nМиссия прервана пользователем")
        mission.stop_mission()
    except Exception as e:
        print(f"\nОшибка: {str(e)}")
        mission.stop_mission()
    finally:
        # Остановка всех потоков
        position_stop_event.set()
        detection_stop_event.set()
        detector.stop()
        
        position_thread.join(timeout=1)
        detection_thread.join(timeout=1)
        display_thread.join(timeout=1)
        
        print("Завершение работы...")
        pioneer.disarm()
        pioneer.stop()

if __name__ == "__main__":
    main()