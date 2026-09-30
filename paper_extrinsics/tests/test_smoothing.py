import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from paper_extrinsics.geometry import Pose
from paper_extrinsics.pipeline import FAILED, PAPER, FramePose
from paper_extrinsics.smoothing import smooth_poses


def straight_line(n: int, noise_m: float = 0.0, seed: int = 0) -> list[FramePose]:
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n):
        rotation = Rotation.from_euler("xyz", [180, 0, 0.5 * i], degrees=True).as_matrix()
        translation = np.array([0.01 * i, 0.0, 0.6]) + rng.normal(0, noise_m, 3)
        poses.append(FramePose(Pose(rotation, translation), 0.3, PAPER))
    return poses


def with_gap(poses: list[FramePose], start: int, length: int) -> list[FramePose]:
    return [FramePose(None, float("nan"), FAILED) if start <= i < start + length else p for i, p in enumerate(poses)]


def test_short_gap_is_filled_on_the_line():
    smoothed = smooth_poses(with_gap(straight_line(40), 15, 12), max_gap=12)
    for i in range(15, 27):
        assert smoothed[i] is not None
        assert smoothed[i].translation == pytest.approx([0.01 * i, 0.0, 0.6], abs=1e-6)


def test_long_gap_stays_empty():
    smoothed = smooth_poses(with_gap(straight_line(40), 15, 13), max_gap=12)
    assert all(smoothed[i] is None for i in range(15, 28))
    assert smoothed[14] is not None and smoothed[28] is not None


def test_noise_is_reduced_without_bending_a_straight_motion():
    truth = straight_line(60)
    noisy = straight_line(60, noise_m=0.002, seed=1)
    smoothed = smooth_poses(noisy, sigma_frames=2.0)
    raw_error = np.std([n.pose.translation - t.pose.translation for n, t in zip(noisy, truth)])
    smooth_error = np.std([s.translation - t.pose.translation for s, t in zip(smoothed, truth)])
    assert smooth_error < 0.6 * raw_error
