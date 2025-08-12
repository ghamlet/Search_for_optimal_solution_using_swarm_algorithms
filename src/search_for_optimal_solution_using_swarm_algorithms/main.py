import time
import threading
from typing import List, Tuple
from pion import Pion
from flight_utils import load_flight_coordinates
from object_detection_controller import ObjectDetectionController
import numpy as np

class MissionController:
    def __init__(self, drone: Pion):
        """
        Инициализация контроллера миссии
        
        :param drone: экземпляр дрона Pion
        """
        self.drone = drone
        self.mission_points = []
        self.stop_event = threading.Event()
    
    def load_mission(self, points: List[Tuple[float, float, float]]):
        """
        Загрузка маршрута из списка точек
        
        :param points: список точек в формате [(x1, y1, z1), (x2, y2, z2), ...]
        """
        self.mission_points = points
    
    def execute_mission(self):
        """Выполнение полета по загруженному маршруту"""
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
                
                # Перелет к точке с ожиданием достижения
                self.drone.goto(x, y, z, yaw=0, wait=True, accuracy=0.2)
                
                # Короткая пауза для стабилизации
                time.sleep(0.3)
            
            print("Миссия успешно завершена")
            
        except Exception as e:
            print(f"Ошибка выполнения миссии: {str(e)}")
            raise
    
    def stop_mission(self):
        """Экстренная остановка миссии"""
        self.stop_event.set()
        self.drone.stop_moving()


def print_position(pioneer, stop_event):
    """Функция для вывода координат в отдельном потоке"""
    while not stop_event.is_set():
        print(f"Текущие координаты: xyz {np.round(pioneer.position[0:3], 3)}")
        time.sleep(1)  # Интервал обновления (1 секунда)




def main():
    # Инициализация дрона (подставьте свои параметры)
    pioneer = Pion(ip="127.0.0.1", mavlink_port=8000, logger=True)
    # pioneer = Pion(ip="10.1.100.160", mavlink_port=5656, logger=True)


    position_stop_event = threading.Event()
    
    # Запускаем поток вывода координат
    position_thread = threading.Thread(
        target=print_position,
        args=(pioneer, position_stop_event),
        daemon=True
    )
    position_thread.start()

    
    try:
        # Создание контроллера миссии
        mission = MissionController(drone=pioneer)
        
        # Пример маршрута (можно загрузить из JSON)
        flight_path = load_flight_coordinates("flight_path.json")
        mission.load_mission(flight_path)
        
        # Запуск миссии
        print("Армирование дрона...")
        pioneer.arm()
        print("Взлет...")
        pioneer.takeoff()
        time.sleep(2)
        
        print("Начало миссии")


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
                # Запуск детекции в фоновом режиме
        detector.start()

        # Запуск отображения в отдельном потоке
        display_thread = threading.Thread(
            target=detector.show_detection,
            daemon=True
        )
        display_thread.start()



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
         # Остановка потока вывода координат
        position_stop_event.set()
        position_thread.join()
        print("Завершение работы...")
        pioneer.disarm()
        pioneer.stop()


if __name__ == "__main__":
    main()