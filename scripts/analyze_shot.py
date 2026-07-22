"""End-to-end: video in -> annotated video + metrics out.

Usage:
    python scripts/analyze_shot.py --video data/my_shot.mp4 --out runs/shot1
    python scripts/analyze_shot.py --video data/my_shot.mp4 --out runs/shot1 --arm right
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import video, visualize
from src.biomechanics import analyze
from src.pose import PoseEstimator


def main() -> None:
    ap = argparse.ArgumentParser(description="Basketball shot biomechanics from video")
    ap.add_argument("--video", required=True, help="path to input clip")
    ap.add_argument("--out", default="runs/shot", help="output directory")
    ap.add_argument("--arm", choices=["right", "left"], default=None)
    ap.add_argument("--model", default="yolo11n-pose.pt")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = video.meta = video.video_meta(args.video)

    estimator = PoseEstimator(args.model)
    writer = video.VideoWriter(
        os.path.join(args.out, "annotated.mp4"),
        meta["fps"], (meta["width"], meta["height"]),
    )

    keypoints_seq = []
    frames = []
    print("Running pose estimation...")
    for idx, frame in video.read_frames(args.video):
        kp = estimator.keypoints(frame)
        if kp is None:
            kp = np.full((17, 2), np.nan)
        keypoints_seq.append(kp)
        frames.append(frame)

    keypoints_seq = np.stack(keypoints_seq)
    metrics = analyze(keypoints_seq, arm=args.arm)

    # Write the annotated video with skeleton + a live metrics readout.
    overlay_lines = [
        f"arm: {metrics['shooting_arm']}",
        f"release elbow: {metrics['release_elbow_angle_deg']} deg",
        f"launch angle: {metrics['launch_angle_deg']} deg",
        f"jump (torso units): {metrics['jump_height_torso_units']}",
    ]
    for i, frame in enumerate(frames):
        kp = keypoints_seq[i]
        drawn = visualize.draw_skeleton(frame, np.nan_to_num(kp))
        drawn = visualize.annotate_metrics(drawn, overlay_lines)
        writer.write(drawn)
    writer.close()

    visualize.plot_elbow_series(
        metrics["elbow_angle_series"], metrics["release_frame"],
        os.path.join(args.out, "elbow_angle.png"),
    )

    # Drop the long per-frame series from the saved summary for readability.
    summary = {k: v for k, v in metrics.items() if k != "elbow_angle_series"}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Results in", args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
