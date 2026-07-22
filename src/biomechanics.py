"""
Biomechanics analysis from 2D pose keypoints.

This is the analytically interesting layer of the project: it turns a sequence
of pose keypoints (one set per video frame) into shot mechanics — joint angles,
release/launch angle, and jump height.

Keypoints follow the COCO-17 format that YOLO-pose produces. Each keypoint is an
(x, y) pixel coordinate, with the image origin at the TOP-LEFT, so a SMALLER y
means HIGHER up in the frame. We flip sign where needed so that "up" reads
intuitively in the outputs.
"""

from __future__ import annotations

import numpy as np

# COCO-17 keypoint indices (the order YOLO-pose returns)
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle (degrees) at point ``b`` formed by segments b->a and b->c.

    Identical idea to the angle-between-vectors you use in kinematics:
        cos(theta) = (ba . bc) / (|ba| |bc|)
    """
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def segment_angle_to_horizontal(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """Angle (degrees) of the segment p_from->p_to above the horizontal.

    Image y grows downward, so we negate dy to make "pointing up" positive.
    Used as a launch-angle proxy from the forearm orientation at release.
    """
    dx = p_to[0] - p_from[0]
    dy = -(p_to[1] - p_from[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def shooting_arm(keypoints_seq: np.ndarray) -> str:
    """Guess the shooting arm as the wrist that reaches highest during the clip.

    keypoints_seq: array of shape (T, 17, 2). Returns "right" or "left".
    Highest = smallest y. NaNs (missing detections) are ignored.
    """
    r = np.nanmin(keypoints_seq[:, R_WRIST, 1])
    l = np.nanmin(keypoints_seq[:, L_WRIST, 1])
    return "right" if r <= l else "left"


def torso_length(kp: np.ndarray) -> float:
    """Shoulder-to-hip distance (pixels), used to normalise jump height so the
    metric is roughly scale-free regardless of how far the camera is."""
    shoulder = np.nanmean([kp[L_SHOULDER], kp[R_SHOULDER]], axis=0)
    hip = np.nanmean([kp[L_HIP], kp[R_HIP]], axis=0)
    return float(np.linalg.norm(shoulder - hip))


def detect_release_frame(keypoints_seq: np.ndarray, arm: str) -> int:
    """Heuristic release frame: the highest wrist point, restricted to frames
    where the wrist is above the shoulder.
    """
    wrist_idx = R_WRIST if arm == "right" else L_WRIST
    shoulder_idx = R_SHOULDER if arm == "right" else L_SHOULDER

    wrist_y = keypoints_seq[:, wrist_idx, 1]
    shoulder_y = keypoints_seq[:, shoulder_idx, 1]

    above_shoulder = wrist_y < shoulder_y
    candidates = np.where(above_shoulder)[0]

    if len(candidates) == 0:
        if np.all(np.isnan(wrist_y)):
            return 0
        return int(np.nanargmin(wrist_y))

    valid_ys = wrist_y[candidates]
    best_local = int(np.nanargmin(valid_ys))
    return int(candidates[best_local])

def analyze(keypoints_seq: np.ndarray, arm: str | None = None) -> dict:
    """Compute shot metrics from a sequence of per-frame keypoints.

    Parameters
    ----------
    keypoints_seq : np.ndarray, shape (T, 17, 2)
        One set of 17 (x, y) keypoints per frame. Use NaN for missing joints.
    arm : "right" | "left" | None
        Shooting arm. If None, inferred automatically.

    Returns
    -------
    dict with per-frame elbow angle, the detected release frame, release-frame
    elbow + launch angle, and a normalised jump height.
    """
    kps = np.asarray(keypoints_seq, dtype=float)
    if kps.ndim != 3 or kps.shape[1:] != (17, 2):
        raise ValueError(f"expected (T, 17, 2), got {kps.shape}")

    arm = arm or shooting_arm(kps)
    sh, el, wr = (
        (R_SHOULDER, R_ELBOW, R_WRIST)
        if arm == "right"
        else (L_SHOULDER, L_ELBOW, L_WRIST)
    )

    elbow_angles = np.array(
        [joint_angle(kps[t, sh], kps[t, el], kps[t, wr]) for t in range(len(kps))]
    )

    release = detect_release_frame(kps, arm)
    release_elbow = float(elbow_angles[release])
    launch_angle = segment_angle_to_horizontal(kps[release, el], kps[release, wr])

    # Jump height: track hip centre. Baseline = median y over the first 25% of
    # the clip (assumed pre-jump stance). Height = how far above baseline the
    # hips rise, normalised by torso length so it's comparable across clips.
    hip_y = np.nanmean(kps[:, [L_HIP, R_HIP], 1], axis=1)
    n_base = max(1, len(hip_y) // 4)
    baseline = np.nanmedian(hip_y[:n_base])
    peak_rise = baseline - np.nanmin(hip_y)  # pixels, up is positive
    torso = np.nanmedian([torso_length(kps[t]) for t in range(len(kps))])
    jump_norm = float(peak_rise / torso) if torso else float("nan")

    return {
        "shooting_arm": arm,
        "release_frame": release,
        "release_elbow_angle_deg": round(release_elbow, 1),
        "launch_angle_deg": round(launch_angle, 1),
        "jump_height_torso_units": round(jump_norm, 2),
        "elbow_angle_series": elbow_angles.tolist(),
    }
