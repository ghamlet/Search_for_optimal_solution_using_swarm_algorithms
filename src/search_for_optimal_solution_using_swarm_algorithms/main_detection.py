import os
import time
import threading
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pion import Pion
from flight_utils import load_flight_coordinates
from queue import Queue





class MissionController:
    def __init__(self, drone: Pion, model_path: str, camera_source: str):
        self.drone = drone
        self.mission_points = []
        self.analyzer = ObjectDetectionAnalyzer(
            model_path=model_path,
            camera_source=camera_source,
            min_confidence=0.6,
            min_detections=3,
            verbose=True
        )
    
    def load_mission(self, points: List[Tuple[float, float, float]]):
        self.mission_points = points
    
    def execute_mission(self):
        """Выполнение миссии с параллельной обработкой видео"""
        video_thread = threading.Thread(target=self.analyzer.process_video_stream)
        video_thread.daemon = True
        video_thread.start()
        
        try:
            for i, (x, y, z) in enumerate(self.mission_points, 1):
                if self.analyzer.stop_event.is_set():
                    break
                    
                print(f"Перелет к точке {i}/{len(self.mission_points)}: ({x:.2f}, {y:.2f}, {z:.2f})")
                self.drone.goto(x, y, z, yaw=0, wait=True, accuracy=0.2)
                self._process_detections()
                time.sleep(0.1)
                    
        finally:
            self.analyzer.release()
            video_thread.join(timeout=1)
    
    def _process_detections(self):
        """Обработка обнаруженных объектов"""
        while not self.analyzer.detection_queue.empty():
            det = self.analyzer.detection_queue.get()
            print(f"Обнаружен объект: {det['class_name']} (уверенность: {det['confidence']:.2f})")
            # Здесь можно добавить логику реакции на обнаруженные объекты


def main():
    pioneer = Pion(ip="127.0.0.1", mavlink_port=8000, logger=True, dt=0.01, mass=0.3)
    
    try:
        mission = MissionController(
            drone=pioneer,
            model_path="/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/weights/best_first.pt",
            camera_source="/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/videos/output_pascal_line2.mp4"
        )
        MAP = load_flight_coordinates("flight_path.json")
        mission.load_mission(MAP)
        
        print("Армирование дрона...")
        pioneer.arm()
        print("Взлет...")
        pioneer.takeoff()
        time.sleep(2)
        
        print("Начинаем выполнение миссии")
        mission.execute_mission()
        
        print("Миссия завершена, посадка...")
        pioneer.land()
        time.sleep(5)


        
    except KeyboardInterrupt:
        print("\nМиссия прервана пользователем")
    except Exception as e:
        print(f"\nОшибка во время выполнения миссии: {str(e)}")
    finally:
        print("Завершение работы...")
        pioneer.stop_moving()
        pioneer.disarm()
        pioneer.stop()

if __name__ == "__main__":
    main()