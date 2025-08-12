import time
import threading
from typing import List, Tuple
from pion import Pion
from flight_utils import load_flight_coordinates
from object_detection_controller import ObjectDetectionController
import numpy as np
from flight_mission_runner import FlightMissionRunner



    
    


# def print_position(pioneer, stop_event):
#     """Функция для вывода координат в отдельном потоке"""
#     while not stop_event.is_set():
#         print(f"Текущие координаты: xyz {np.round(pioneer.position[0:3], 3)}")
#         time.sleep(1)  # Интервал обновления (1 секунда)




def main():
    # Инициализация дрона (подставьте свои параметры)
    pioneer = Pion(ip="127.0.0.1", mavlink_port=8000, logger=True)
    # pioneer = Pion(ip="10.1.100.160", mavlink_port=5656, logger=True)

    CAMERA_SOURCE = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/videos/output_pascal_line2.mp4"
    MODEL_PATH = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/weights/best_first.pt"
    
    # CAMERA_SOURCE = "rtsp://10.1.100.160/8585/pioneer_stream"


    # position_stop_event = threading.Event()
    
    # # Запускаем поток вывода координат
    # position_thread = threading.Thread(
    #     target=print_position,
    #     args=(pioneer, position_stop_event),
    #     daemon=True
    # )
    # position_thread.start()

    
    try:
        MAP_POINTS = load_flight_coordinates("flight_path.json")
        mission = FlightMissionRunner(MAP_POINTS)

        
        
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
            camera_source=CAMERA_SOURCE,  # Веб-камера
            model_path=MODEL_PATH,
            conf_threshold=0.6,
            min_detections=3,
            buffer_size=15
        )
        
        # Запуск детекции в фоновом режиме
                # Запуск детекции в фоновом режиме
        # detector.start()

        # # Запуск отображения в отдельном потоке
        # display_thread = threading.Thread(
        #     target=detector.show_detection,
        #     daemon=True
        # )
        # display_thread.start()




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
        # position_stop_event.set()
        # position_thread.join()
        print("Завершение работы...")
        pioneer.disarm()
        pioneer.stop()


if __name__ == "__main__":
    main()