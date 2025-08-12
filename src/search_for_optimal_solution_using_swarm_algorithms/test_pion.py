from pion import Pion
import time
from flight_mission_runner import FlightMissionRunner
from flight_utils import load_flight_coordinates
import numpy as np
import cv2

from object_detection_controller import ObjectDetectionController

pioneer = Pion(ip="127.0.0.1", mavlink_port=8000, logger=True, dt=0.0, mass=0.3)
# pioneer = Pion(ip="10.1.100.160", mavlink_port=5656, logger=True, dt=0.0, mass=0.3)


print("Армирование дрона...")
pioneer.arm()
print("Взлет...")
pioneer.takeoff()
time.sleep(2)


# CAMERA_SOURCE = "rtsp://10.1.100.160:8554/pioneer_stream"

CAMERA_SOURCE = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/videos/output_pascal_line2.mp4"
MODEL_PATH = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/weights/best_first.pt"

print("Начало миссии")
MAP_POINTS = load_flight_coordinates("flight_path.json")
mission = FlightMissionRunner(MAP_POINTS)

# Инициализация детектора
od_controller = ObjectDetectionController(
    pioneer=pioneer,
    camera_source=CAMERA_SOURCE,
    model_path=MODEL_PATH,
    conf_threshold=0.5
)


flight_height = 1.5



POINT_REACHED_FIRST = True 
first_point = mission.get_next_point()
x, y, z = first_point
pioneer.goto(x=x, y=y, z=flight_height, yaw=0, accuracy=0.3)


try:
    while not mission.is_complete():
        point_state = pioneer.point_reached  
        print(f"Состояние точки: {point_state}")


        print(f"Текущие координаты: xyz {np.round(pioneer.position[0:3], 3)}")


        
        # Если точка достигнута И это первое обнаружение достижения
        if point_state and POINT_REACHED_FIRST:  


                  
            next_point = mission.get_next_point()
            if next_point:
                x, y, z = next_point
                pioneer.goto(x=x, y=y, z=flight_height, yaw=0, accuracy=0.05)
                print("GO TO")
                POINT_REACHED_FIRST = False

            
        

        elif not point_state:
            POINT_REACHED_FIRST = True



        # Обработка видеопотока (с ограничением частоты)
        # od_controller.process_frame(detection_interval=0.1)  # ~10 FPS
        
        # # Отображение результатов
        # od_controller.show_detection()
        
        # Прерывание по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.05)  


    pioneer.land()


except KeyboardInterrupt:
    print("\nПолучен сигнал прерывания (Ctrl+C)")
    pioneer.land()
    pioneer.stop()


except Exception as e:
    print(f"Ошибка во время выполнения миссии: {str(e)}")
    raise


finally:
    cv2.destroyAllWindows()

    pioneer.land()
    pioneer.disarm()
    pioneer.stop()
