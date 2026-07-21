"""Thin wrapper around Ultralytics YOLO-pose (PyTorch).

Keeps the rest of the codebase model-agnostic: everything downstream just sees
a (T, 17, 2) array of keypoints, so you can swap in MMPose or a custom PyTorch
model later without touching the biomechanics layer.
"""

from __future__ import annotations

import numpy as np


class PoseEstimator:
    def __init__(self, model_name: str = "yolo11n-pose.pt"):
        # Imported lazily so the geometry/tests don't require torch installed.
        from ultralytics import YOLO

        self.model = YOLO(model_name)

    def keypoints(self, frame: np.ndarray) -> np.ndarray | None:
        """Return (17, 2) keypoints for the most prominent person, or None.

        v1 strategy: pick the highest-confidence detection. For a controlled,
        single-player clip that's the shooter. For multi-player footage you'd
        switch to model.track() and follow one ID — noted in the roadmap.
        """
        results = self.model(frame, verbose=False)
        if not results or results[0].keypoints is None:
            return None
        kpts = results[0].keypoints
        if kpts.xy.shape[0] == 0:
            return None
        # Choose the detection with the largest bounding box (closest/biggest).
        boxes = results[0].boxes.xywh.cpu().numpy()  # (n, 4): x, y, w, h
        areas = boxes[:, 2] * boxes[:, 3]
        best = int(np.argmax(areas))
        return kpts.xy[best].cpu().numpy()
