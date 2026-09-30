# paper-extrinsics

Camera pose per video frame for a handheld top-down video, from an A4 sheet lying still on the table.
Vocabulary is in [CONTEXT.md](CONTEXT.md); why the focal length is estimated from the paper is in
[docs/adr/0001-focal-from-reference-paper.md](docs/adr/0001-focal-from-reference-paper.md).

```bash
# one clip, checking its frame count against the hand pose estimator's CSV
uv run extract-extrinsics clip_overlay.mp4 --joints clip_raw_joints.csv --out out/

# a folder: the focal length is estimated once from all clips and saved to out/intrinsics.json;
# later runs with the same --out (or --intrinsics) reuse it. <name>_raw_joints.csv next to
# <name>_overlay.mp4 is picked up for the check.
uv run extract-extrinsics "clips/*_overlay.mp4" --glob --out out/
```

Per clip it writes `<name>_camera_poses.csv` (T_world_cam per frame, raw and smoothed, with the pose source)
and `<name>_camera_debug.mp4` (paper outline and paper-frame axes drawn on the video).

To look at a processed clip in 3D (table, reference paper, moving camera with its frustum, hand skeleton,
and the video frame alongside):

```bash
uv run view-scene clip_overlay.mp4 --out out/            # interactive window
uv run view-scene clip_overlay.mp4 --out out/ --save scene.gif
```

Run the tests with `env -u PYTHONPATH uv run pytest` (unset PYTHONPATH so a sourced ROS install does not load its pytest plugins).
