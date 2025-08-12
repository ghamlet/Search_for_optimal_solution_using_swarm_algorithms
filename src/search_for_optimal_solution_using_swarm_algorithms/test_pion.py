from pion import Pion
import time


pioneer = Pion(ip="127.0.0.1", mavlink_port=8000, logger=True, dt=0.0, mass=0.3)

print("Армирование дрона...")
pioneer.arm()
print("Взлет...")
pioneer.takeoff()
time.sleep(2)

print("Начало миссии")
MAP = [
(0,0,2),
(2,2,2),
(4,4,2)


]

i = 0  
POINT_REACHED_FIRST = True 

pioneer.goto(MAP[i][0], MAP[i][1], MAP[i][2], yaw=0, accuracy=0.2)
i += 1  

while True:
    point_state = pioneer.point_reached  
    print(point_state)
    
    # Если точка достигнута И это первое обнаружение достижения
    if point_state and POINT_REACHED_FIRST:
        print(f"Точка {i} достигнута, летим к точке {i+1}")
        
        if i < len(MAP):
            pioneer.goto(MAP[i][0], MAP[i][1], MAP[i][2], yaw=0, accuracy=0.2)
            print("goto")
            i += 1
            POINT_REACHED_FIRST = False
        
        else:
            print("Маршрут завершен!")
            pioneer.land()
            break
    

    elif not point_state:
        POINT_REACHED_FIRST = True
    
    time.sleep(0.2)  
