from pion import Pion
import time
from flight_mission_runner import FlightMissionRunner
from flight_utils import load_flight_coordinates

pioneer = Pion(ip="127.0.0.1", mavlink_port=8000, logger=True, dt=0.0, mass=0.3)

print("Армирование дрона...")
pioneer.arm()
print("Взлет...")
pioneer.takeoff()
time.sleep(2)

print("Начало миссии")
MAP_POINTS = load_flight_coordinates("flight_path.json")
mission = FlightMissionRunner(MAP_POINTS)

flight_height = 1.5



POINT_REACHED_FIRST = True 
first_point = mission.get_next_point()
x, y, z = first_point
pioneer.goto(x=x, y=y, z=flight_height, yaw=0, accuracy=0.3)


try:
    while not mission.is_complete():
        point_state = pioneer.point_reached  
        print(f"Состояние точки: {point_state}")

        
        # Если точка достигнута И это первое обнаружение достижения
        if point_state and POINT_REACHED_FIRST:  


                  
            next_point = mission.get_next_point()
            if next_point:
                x, y, z = next_point
                pioneer.goto(x=x, y=y, z=flight_height, yaw=0, accuracy=0.3)
                POINT_REACHED_FIRST = False

            
        

        elif not point_state:
            POINT_REACHED_FIRST = True
        
        time.sleep(0.2)  


    pioneer.land()


except KeyboardInterrupt:
    print("\nПолучен сигнал прерывания (Ctrl+C)")
    pioneer.land()


except Exception as e:
    print(f"Ошибка во время выполнения миссии: {str(e)}")
    raise


finally:
    pioneer.land()
    pioneer.disarm()
