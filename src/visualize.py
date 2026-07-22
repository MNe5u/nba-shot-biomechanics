"""Drawing helpers: skeleton overlay on frames and a metrics plot."""

from __future__ import annotations

import cv2
import numpy as np

# COCO-17 skeleton edges (pairs of keypoint indices to connect with a line)
SKELETON = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 6), (11, 12),    # shoulders, hips
    (5, 11), (6, 12),    # torso sides
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


def draw_skeleton(frame: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Overlay keypoints and skeleton edges on a copy of the frame."""
    out = frame.copy()
    for a, b in SKELETON:
        pa, pb = kp[a], kp[b]
        if np.all(pa > 0) and np.all(pb > 0):
            cv2.line(out, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 255, 0), 2)
    for x, y in kp:
        if x > 0 and y > 0:
            cv2.circle(out, (int(x), int(y)), 4, (0, 128, 255), -1)
    return out


def annotate_metrics(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    for i, text in enumerate(lines):
        cv2.putText(out, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def plot_elbow_series(series: list[float], release_frame: int, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series, label="elbow angle (deg)")
    ax.axvline(release_frame, color="r", linestyle="--", label="release")
    ax.set_xlabel("frame")
    ax.set_ylabel("angle (deg)")
    ax.set_title("Shooting elbow extension over the shot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
