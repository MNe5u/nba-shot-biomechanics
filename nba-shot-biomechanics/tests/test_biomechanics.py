"""Synthetic-data tests for the biomechanics maths.

These need no model and no video — they validate the geometry directly, which
is the part most worth getting right. Run with: python -m pytest tests/ -v
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.biomechanics import joint_angle, segment_angle_to_horizontal, analyze


def test_right_angle():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert abs(joint_angle(a, b, c) - 90.0) < 1e-6


def test_straight_arm_is_180():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([2.0, 0.0])
    assert abs(joint_angle(a, b, c) - 180.0) < 1e-6


def test_launch_angle_45_degrees_up():
    # forearm pointing up-and-to-the-right; remember image y grows downward
    p_from = np.array([0.0, 0.0])
    p_to = np.array([1.0, -1.0])
    assert abs(segment_angle_to_horizontal(p_from, p_to) - 45.0) < 1e-6


def test_analyze_end_to_end_synthetic():
    # Build a fake 10-frame shot: arm extends and wrist rises over time.
    T = 10
    kps = np.zeros((T, 17, 2))
    for t in range(T):
        rise = t * 5.0
        kps[t, 6] = [100, 100]            # R shoulder (fixed)
        kps[t, 8] = [120, 90 - rise]      # R elbow rises
        kps[t, 10] = [140, 80 - 2 * rise] # R wrist rises faster
        kps[t, 11] = [90, 200 - rise]     # L hip rises (jump)
        kps[t, 12] = [110, 200 - rise]    # R hip rises (jump)
    out = analyze(kps, arm="right")
    assert out["shooting_arm"] == "right"
    assert out["release_frame"] == T - 1          # wrist highest at the end
    assert out["jump_height_torso_units"] > 0     # hips rose
    assert len(out["elbow_angle_series"]) == T


if __name__ == "__main__":
    test_right_angle()
    test_straight_arm_is_180()
    test_launch_angle_45_degrees_up()
    test_analyze_end_to_end_synthetic()
    print("All synthetic biomechanics tests passed.")
