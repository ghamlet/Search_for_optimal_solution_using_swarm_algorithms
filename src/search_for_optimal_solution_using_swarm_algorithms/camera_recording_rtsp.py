# /// script
# dependencies = [
#   "opencv-python",
#   "numpy",
# ]
# ///
import argparse
import os
import signal
import socket
import sys
import time
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
from numpy.typing import NDArray


class BaseCamera(ABC):
    @abstractmethod
    def get_cv_frame(self) -> Optional[NDArray]:
        """
        Должен возвращать кадр в формате cv2 или None, если кадр не получен.
        """
        pass


class VideoRecorder:
    def __init__(self, output_dir: str = "recordings", fps: int = 30, frame_size: tuple = (640, 640)):
        self.frame_size = frame_size  # Фиксированный размер
        self.output_dir = output_dir
        self.fps = fps
        self.writer = None
        self.current_file = None
        self.frame_size = None
        
        os.makedirs(output_dir, exist_ok=True)

    def start_recording(self, frame_size: tuple):
        """Начинает запись нового видеофайла"""
        if self.writer is not None:
            self.writer.release()
            
        self.frame_size = frame_size
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
        
        # Используем кодек H.264
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.current_file, 
            fourcc, 
            self.fps, 
            frame_size
        )
        print(f"Начата запись видео: {self.current_file}")

    def write_frame(self, frame: NDArray):
        """Записывает кадр в видео с ресайзом при необходимости"""
        if self.writer is None:
            self.start_recording((640, 640))
        
        # Приводим кадр к нужному размеру
        if frame.shape[:2] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)

    def stop_recording(self):
        """Останавливает запись и закрывает файл"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Запись завершена: {self.current_file}")


# Реализация для RTSP камеры
class RTSPCamera(BaseCamera):
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def get_cv_frame(self) -> Optional[NDArray]:
        if not self.cap.isOpened():
            self.cap.open(self.rtsp_url)
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()





def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY")) or sys.platform.startswith("win")


def run_loop(cam: BaseCamera, show_window: bool = True, record_video: bool = True) -> None:
    win = "pion_camera"
    last_dump_ts = 0.0
    dump_interval = 5.0  # сек, для headless
    
    # Инициализация видеозаписи
    recorder = VideoRecorder() if record_video else None
    first_frame = True

    # Корректное завершение по Ctrl+C
    stop = {"flag": False}

    def _sigint_handler(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    try:
        if show_window:
            try:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            except Exception:
                show_window = False

        while not stop["flag"]:
            frame = cam.get_cv_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Запись видео
            if recorder is not None:
                recorder.write_frame(frame)

            if show_window:
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                now = time.time()
                if now - last_dump_ts > dump_interval:
                    cv2.imwrite("/tmp/last_frame.jpg", frame)
                    last_dump_ts = now
    finally:
        if isinstance(cam, RTSPCamera):
            cam.release()
        if recorder is not None:
            recorder.stop_recording()
        try:
            if show_window:
                cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run camera reader")
    parser.add_argument("--ip", type=str, help="IP", default="10.1.100.160")
    parser.add_argument("--port", type=int, help="порт", default="8554")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Без окна (сбрасывать кадры в /tmp/last_frame.jpg)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Не записывать видео",
    )
    args = parser.parse_args()

    show = (not args.headless) and _has_display()
    record = True

    cam = RTSPCamera(rtsp_url=f"rtsp://{args.ip}:{args.port}/pioneer_stream")
    run_loop(cam, show_window=show, record_video=record)


if __name__ == "__main__":
    main()