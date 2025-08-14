import threading

import cv2
import numpy as np
import socket


class CameraSim:

    def __init__(self, timeout=0.5, ip='192.168.4.1', port=8888, video_buffer_size=65000, log_connection=True):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.VIDEO_BUFFER_SIZE = video_buffer_size
        self.tcp = None
        self.udp = None
        self.raw_video_frame = 0
        self._video_frame_buffer = bytes()
        self.raw_video_frame = bytes()
        self.connected = False
        self._last_success_frame: bytes | None = None
        self.log_connection = log_connection

    def new_tcp(self):
        """Returns new TCP socket"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(self.timeout)
        return sock

    def new_udp(self):
        """Returns new UDP socket"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(self.timeout)
        return sock

    def connect(self):
        """Connect to TCP and UDP sockets. Creates new ones if necessary."""
        self.disconnect()
        self.tcp = self.new_tcp()
        self.udp = self.new_udp()
        try:
            self.tcp.connect((self.ip, self.port))
            self.udp.bind(self.tcp.getsockname())
        except (TimeoutError, socket.timeout, OSError):
            return False
        return True

    def disconnect(self):
        """Disconnect."""
        self.connected = False
        if self.tcp is not None:
            self.tcp.close()
            self.tcp = None
        if self.udp is not None:
            self.udp.close()
            self.udp = None

    def get_frame(self):
        """Get bytes of frame.
        If UDP socket timeout then Exception raised.
        Returns :
            boolean: True if success, False if bad data of no data in UDP socket
            bytes(): Bytes of frame.
                     If first arg is True then current frame.
                     If first arg is False then last success frame.
                     If no last success frame then None."""
        try:
            if not self.connected:
                if self.connect():
                    self.connected = True
                    if self.log_connection:
                        print('Camera CONNECTED')
                else:
                    return None
            self._video_frame_buffer, addr = self.udp.recvfrom(self.VIDEO_BUFFER_SIZE)
            beginning = self._video_frame_buffer.find(b'\xff\xd8')
            if beginning == -1:
                return self._last_success_frame
            self._video_frame_buffer = self._video_frame_buffer[beginning:]
            end = self._video_frame_buffer.find(b'\xff\xd9')
            if end == -1:
                return self._last_success_frame
            self.raw_video_frame = self._video_frame_buffer[:end + 2]
            self._last_success_frame = self.raw_video_frame
            return self.raw_video_frame
        except (TimeoutError, socket.timeout):
            if self.connected:
                self.connected = False
                if self.log_connection:
                    print('Camera DISCONNECTED')
            return self._last_success_frame
        except Exception:
            # Любая другая ошибка не должна заваливать поток
            return self._last_success_frame

    def get_cv_frame(self):
        """
        get cv_frame
        :return: cv_frame or None
        """

        frame = self.get_frame()
        if frame is not None:
            frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame


