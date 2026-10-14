"""The webcam: a background reader holding the latest frame, and its calibrated pose."""

import json
import os
import threading
import time

import cv2
import numpy as np

from control_desk_arm.arm import CALIBRATION_DIR

CAMERA_INDEX = int(os.environ.get("ARM_CAMERA", "0"))


class Camera:
    def __init__(self, index: int = CAMERA_INDEX, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            with self._lock:
                self._frame = frame

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


class CameraPose:
    """Projects robot-frame points (meters) into the image, from calibration/camera.json."""

    def __init__(self):
        cal = json.loads((CALIBRATION_DIR / "camera.json").read_text())
        f = cal["focal_px"]
        cx, cy = cal["principal_point"]
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        self.rvec = np.array(cal["rvec"], float)
        self.tvec = np.array(cal["tvec"], float)
        self.R = cv2.Rodrigues(self.rvec)[0]  # robot frame -> camera frame

    def project(self, points) -> np.ndarray:
        pts = np.asarray(points, float).reshape(-1, 3)
        return cv2.projectPoints(pts, self.rvec, self.tvec, self.K, None)[0].reshape(-1, 2)
