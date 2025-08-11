import sys
import time

import numpy as np

from pion import Pion

GLOBAL = "10.1.100.160"

# Демонстрационный полет симуляционного дрона
drone = Pion(
    ip="127.0.0.1", mavlink_port=5656, mass=0.3, dt=0.0, logger=True
)



drone.arm()
drone.takeoff()
time.sleep(5)


drone.goto(0, 0, 2, 0, wait=True, accuracy=0.1)


drone.goto(2, 2, 2, 0, wait=True, accuracy=0.1)




drone.land()
print("stop ----------------<")
drone.stop()

