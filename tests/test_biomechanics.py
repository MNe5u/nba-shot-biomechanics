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


def test_release_ignores_relaxed_arm_after_shot():
    """Regression test for the real bug this fix targets: a shot followed by
    the arm relaxing back down to the athlete's side must NOT be mistaken
    for a second, higher-scoring release -- even though a straight relaxed
    arm and a straight raised arm can have a near-identical elbow angle.
    """
    T = 30
    kps = np.zeros((T, 17, 2))
    shoulder_y = 100.0
    for t in range(T):
        kps[t, 6] = [100, shoulder_y]  # R shoulder fixed
        if t < 10:
            # Rising toward release: wrist goes from below to above shoulder.
            wrist_y = shoulder_y - (t * 8.0)
        elif t < 15:
            # True release + a brief follow-through hold near the peak.
            wrist_y = shoulder_y - 80.0
        else:
            # Arm relaxes back down to the athlete's side, well BELOW the
            # shoulder -- this is the part that used to fool the heuristic
            # if noise made it briefly read as "straight".
            wrist_y = shoulder_y + 60.0 + (t - 15) * 2.0
        kps[t, 8] = [110, (shoulder_y + wrist_y) / 2]  # elbow, roughly between
        kps[t, 10] = [120, wrist_y]

    out = analyze(kps, arm="right")
    # Release must land in the true 10-14 window (above shoulder), never in
    # the relaxed-arm tail even if some frame there has a near-straight angle.
    assert 9 <= out["release_frame"] <= 14
    assert out["release_frame"] < 15


if __name__ == "__main__":
    test_right_angle()
    test_straight_arm_is_180()
    test_launch_angle_45_degrees_up()
    test_analyze_end_to_end_synthetic()
    test_release_ignores_relaxed_arm_after_shot()
    print("All synthetic biomechanics tests passed.")

