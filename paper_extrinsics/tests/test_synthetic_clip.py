import numpy as np
import pytest

from paper_extrinsics.focal import estimate_focal
from paper_extrinsics.geometry import Intrinsics, Pose
from paper_extrinsics.pipeline import solve_poses, track_paper_corners

from synthetic import CORNER_OCCLUSION_FRAMES, EDGE_OCCLUSION_FRAMES, make_clip

MAX_ROTATION_ERROR_DEG = 1.0
MAX_TRANSLATION_ERROR_M = 0.005


def rotation_error_deg(a: Pose, b: Pose) -> float:
    cos = (np.trace(a.rotation.T @ b.rotation) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


@pytest.fixture(scope="module")
def clip():
    return make_clip()


@pytest.fixture(scope="module")
def corners(clip):
    return track_paper_corners(clip.frames)


@pytest.fixture(scope="module")
def focal(corners, clip):
    measured = [c.corners for c in corners if c.source == "paper"]
    return estimate_focal(measured, (clip.intrinsics.width, clip.intrinsics.height))


@pytest.fixture(scope="module")
def poses(corners, focal, clip):
    intrinsics = Intrinsics.centred(focal.focal_px, clip.intrinsics.width, clip.intrinsics.height)
    return solve_poses(corners, intrinsics)


def test_focal_length_recovered_within_3_percent(focal, clip):
    assert focal.focal_px == pytest.approx(clip.intrinsics.fx, rel=0.03)
    assert focal.sigma_px > 0


def test_every_frame_has_a_pose(poses):
    assert all(p.pose is not None for p in poses)


def test_sources_follow_the_occlusions(poses):
    for i, p in enumerate(poses):
        expected = "tracked" if i in EDGE_OCCLUSION_FRAMES else "paper"
        assert p.source == expected, f"frame {i}"


@pytest.mark.parametrize("frames", [range(0, 40), CORNER_OCCLUSION_FRAMES, EDGE_OCCLUSION_FRAMES, range(75, 90)],
                         ids=["unoccluded", "corner_hidden", "edge_hidden", "after"])
def test_poses_match_ground_truth(poses, clip, frames):
    for i in frames:
        estimated, truth = poses[i].pose, clip.poses[i]
        assert rotation_error_deg(estimated, truth) < MAX_ROTATION_ERROR_DEG, f"frame {i}"
        assert np.linalg.norm(estimated.translation - truth.translation) < MAX_TRANSLATION_ERROR_M, f"frame {i}"


def test_paper_frame_x_points_to_top_of_first_frame(poses):
    first = poses[0].pose
    camera_y_in_world = first.rotation[:, 1]  # image down
    assert camera_y_in_world @ np.array([1.0, 0, 0]) < -0.9
