from typing import Dict, List, Optional, Tuple
import cv2
from numpy.typing import NDArray


class CameraReal:
    def __init__(self, source: str):
        self.source = source
        self.is_video_file = not source.startswith(('rtsp://', 'http://', 'https://'))
        self.cap = self._init_capture()

    def _init_capture(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть источник: {self.source}")
        return cap

    def get_cv_frame(self) -> Optional[NDArray]:
        ret, frame = self.cap.read()
        if not ret and self.is_video_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

