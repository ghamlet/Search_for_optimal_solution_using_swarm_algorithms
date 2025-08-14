import time
import threading
import cv2
from pion import Pion
from object_detection_controller_draft import ObjectDetectionController


def detection_loop(detector: ObjectDetectionController, stop_event: threading.Event):
    """Показывает кадры и завершает работу по нажатию 'q'."""
    while not stop_event.is_set():
        detector.show_frame()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
        time.sleep(0.05)


def main():
    # Без полёта: инициализируем Pion только для чтения позиции и инфраструктуры
    pioneer = Pion(ip="10.1.100.237", mavlink_port=5656, logger=True, dt=0.0, mass=0.5)
    # pioneer = Pion(ip="127.0.0.1", mavlink_port=8000, logger=True, dt=0.0, mass=0.5)


    # Видеопоток и веса модели (как в main.py)
    CAMERA_SOURCE = "rtsp://10.1.100.237:8554/pioneer_stream"
    # CAMERA_SOURCE = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/videos/output_pascal_line2.mp4"
    MODEL_PATH = "/home/arrma/PROGRAMMS/Search_for_optimal_solution_using_swarm_algorithms/src/search_for_optimal_solution_using_swarm_algorithms/weights/best_last_sana.pt"

    stop_event = threading.Event()

    # Контроллер детекции сам поднимет необходимые потоки и будет печатать
    detector = ObjectDetectionController(
        pioneer=pioneer,
        camera_source=CAMERA_SOURCE,
        model_path=MODEL_PATH,
        conf_threshold=0.6,
        min_detections=5,
        buffer_size=10,
    )
    detector.start()

    # Отдельный поток показа, чтобы основной мог ловить KeyboardInterrupt
    display_thread = threading.Thread(
        target=detection_loop,
        args=(detector, stop_event),
        daemon=True,
    )
    display_thread.start()

    print("Детекция запущена. Нажмите 'q' в окне видео или Ctrl+C для завершения.")

    try:
        while not stop_event.is_set() and display_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C")
        stop_event.set()
    finally:
        stop_event.set()
        detector.stop()
        if 'display_thread' in locals():
            display_thread.join(timeout=1)
        try:
            pioneer.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


