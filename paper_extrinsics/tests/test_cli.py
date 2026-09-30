import cv2
import numpy as np
import pytest

from paper_extrinsics.cli import main
from paper_extrinsics.io import load_intrinsics, read_poses_csv

from synthetic import make_clip

WRIST_OFFSET = np.array([0.05, 0.04, 0.6])


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A synthetic clip written as <name>_overlay.mp4, with a joints CSV next to it."""
    folder = tmp_path_factory.mktemp("clips")
    clip = make_clip(n_frames=40)
    video = folder / "scene_overlay.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), clip.fps, (832, 480))
    for frame in clip.frames:
        writer.write(frame)
    writer.release()

    first = clip.poses[0]
    on_table = np.column_stack([np.linspace(0.0, 0.1, 21), np.linspace(-0.25, -0.2, 21), np.full(21, 0.01)])
    in_camera = (on_table - first.translation) @ first.rotation  # R^T (p - t)
    lines = ["# synthetic: raw predicted hand joints",
             f"# origin: right wrist at frame 0 (raw camera position {WRIST_OFFSET[0]:.4f}, {WRIST_OFFSET[1]:.4f}, "
             f"{WRIST_OFFSET[2]:.4f} m subtracted)",
             "frame,time_s,hand,joint,name,x_m,y_m,z_m"]
    for frame in range(len(clip.frames)):
        for joint, p in enumerate(in_camera - WRIST_OFFSET):
            lines.append(f"{frame},{frame / 24:.6f},right,{joint},j{joint},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}")
    (folder / "scene_raw_joints.csv").write_text("\n".join(lines) + "\n")
    return folder, clip


def test_glob_run_writes_poses_intrinsics_and_debug_video(recorded, tmp_path, capsys):
    folder, clip = recorded
    code = main([str(folder / "*_overlay.mp4"), "--glob", "--out", str(tmp_path), "--pixel-aspect", "1"])
    assert code == 0

    rows = read_poses_csv(tmp_path / "scene_overlay_camera_poses.csv")
    assert [int(r["frame"]) for r in rows] == list(range(len(clip.frames)))
    assert all(r["source"] in {"paper", "tracked"} for r in rows)
    assert float(rows[0]["tz_m"]) == pytest.approx(clip.poses[0].translation[2], abs=0.02)
    assert (tmp_path / "scene_overlay_camera_debug.mp4").stat().st_size > 0

    saved = load_intrinsics(tmp_path / "intrinsics.json")
    assert saved.intrinsics.fx == pytest.approx(clip.intrinsics.fx, rel=0.05)
    assert "WARNING" not in capsys.readouterr().out  # the joints CSV's frame count matches the video


def test_second_run_reuses_the_saved_focal_length(recorded, tmp_path):
    folder, _ = recorded
    video = str(folder / "scene_overlay.mp4")
    main([video, "--out", str(tmp_path), "--focal-px", "612", "--no-debug-video", "--pixel-aspect", "1"])
    main([video, "--out", str(tmp_path), "--no-debug-video"])
    assert load_intrinsics(tmp_path / "intrinsics.json").intrinsics.fx == 612
    header = (tmp_path / "scene_overlay_camera_poses.csv").read_text()
    assert "fx=612.00" in header
