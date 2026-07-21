# 🏀 NBA Shot Biomechanics from Video

Extract a player's pose from a jump-shot clip and measure the mechanics of the
shot — release elbow angle, launch angle, and jump height — from nothing but
ordinary video. A computer-vision pipeline that turns footage into quantified,
shot-to-shot biomechanics.

> Built as a portfolio project exploring pose estimation + applied kinematics.
> The perception-and-analysis loop here is the same one that underpins
> image-based robot control.

---

## What it does

```
clip.mp4  ──▶  person detection  ──▶  2D pose (17 keypoints)  ──▶  biomechanics
                                                                      │
                          annotated.mp4 + metrics.json + plots  ◀─────┘
```

1. **Pose estimation** — YOLO-pose (PyTorch) gives 17 COCO keypoints per frame.
2. **Biomechanics** — keypoint sequences become joint angles via vector geometry,
   plus a heuristic release-frame detector and a scale-normalised jump-height
   estimate.
3. **Output** — an annotated video with the skeleton overlaid, a JSON metrics
   summary, and an elbow-extension-over-time plot.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# verify the geometry (no model/video needed)
python -m pytest tests/ -v

# analyse a clip
python scripts/analyze_shot.py --video data/my_shot.mp4 --out runs/shot1
```

Best first input: a few seconds of **one player** shooting, filmed side-on with a
**fixed camera**. That removes player-ID and camera-motion confounds so the
mechanics come through cleanly.

## Example output

```json
{
  "shooting_arm": "right",
  "release_frame": 47,
  "release_elbow_angle_deg": 168.4,
  "launch_angle_deg": 52.1,
  "jump_height_torso_units": 0.83
}
```

## How the metrics are defined

- **Elbow angle** — interior angle at the elbow between the upper arm and forearm
  vectors. A near-straight arm (~180°) at release indicates full extension.
- **Launch angle** — orientation of the forearm above horizontal at release, used
  as a proxy for the ball's initial trajectory.
- **Jump height** — rise of the hip centre above its pre-jump baseline,
  normalised by torso length so it's comparable regardless of camera distance.

## Design choices

- The biomechanics layer only ever sees a `(T, 17, 2)` keypoint array, so the
  pose backbone is swappable — YOLO-pose today, MMPose or a custom PyTorch model
  later, with zero downstream changes.
- The geometry is unit-tested against synthetic poses, independent of any model.

## Known limitations (and the roadmap they imply)

This is an honest v1. The interesting work is in fixing these:

- **Release detection is a heuristic** (highest wrist position). *Next:* fine-tune
  a small object detector to find the ball and detect the true release — the main
  PyTorch-training milestone.
- **2D only** — angles are projections, so camera angle matters. *Next:* multi-view
  or a monocular 3D-pose model to recover true joint angles.
- **Single player, controlled clip.** *Next:* multi-object tracking to follow one
  player through broadcast footage.
- **No metric scale** for jump height. *Next:* calibrate against a known reference
  (hoop height, court markings).

## Project layout

```
src/biomechanics.py   # joint angles, release detection, jump height (+ tests)
src/pose.py           # YOLO-pose wrapper (model-agnostic interface)
src/video.py          # frame read/write
src/visualize.py      # skeleton overlay + plots
scripts/analyze_shot.py  # end-to-end CLI
tests/                # synthetic-data geometry tests
```

## License

MIT
