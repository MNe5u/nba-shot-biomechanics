"""Video read/write helpers built on OpenCV."""

from __future__ import annotations

from typing import Iterator, Tuple

import cv2
import numpy as np


def read_frames(path: str) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_index, frame_bgr) for every frame in the video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


def video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


class VideoWriter:
    def __init__(self, path: str, fps: float, size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
