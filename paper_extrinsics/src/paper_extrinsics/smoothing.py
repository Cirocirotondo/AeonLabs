"""Offline smoothing of a clip's camera poses, looking both backward and forward in time."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .geometry import Pose
from .pipeline import PAPER, TRACKED, FramePose

SOURCE_WEIGHT = {PAPER: 1.0, TRACKED: 0.5}
FILLED_WEIGHT = 0.2  # frames inside a short gap, interpolated before smoothing


def _local_linear(values: np.ndarray, weights: np.ndarray, available: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-weighted local linear fit of each column at every available frame."""
    n = len(values)
    radius = int(np.ceil(3 * sigma))
    out = np.full_like(values, np.nan)
    for i in np.flatnonzero(available):
        j = np.arange(max(0, i - radius), min(n, i + radius + 1))
        j = j[available[j]]
        d = (j - i).astype(float)
        w = weights[j] * np.exp(-0.5 * (d / sigma) ** 2)
        design = np.column_stack([np.ones_like(d), d])
        normal = design.T @ (w[:, None] * design)
        if len(j) >= 2 and np.linalg.cond(normal) < 1e8:
            out[i] = np.linalg.solve(normal, design.T @ (w[:, None] * values[j]))[0]
        else:
            out[i] = (w[:, None] * values[j]).sum(axis=0) / w.sum()
    return out


def smooth_poses(poses: Sequence[FramePose], sigma_frames: float = 2.0, max_gap: int = 12) -> list[Pose | None]:
    """Smoothed pose per frame; gaps of at most `max_gap` frames between two poses are filled, longer ones stay None."""
    n = len(poses)
    valid = np.array([p.pose is not None for p in poses])
    translations = np.full((n, 3), np.nan)
    quaternions = np.full((n, 4), np.nan)
    weights = np.zeros(n)
    for i, p in enumerate(poses):
        if p.pose is not None:
            translations[i] = p.pose.translation
            quaternions[i] = p.pose.quaternion_xyzw()
            weights[i] = SOURCE_WEIGHT.get(p.source, 0.5)

    available = valid.copy()
    known = np.flatnonzero(valid)
    for a, b in zip(known[:-1], known[1:]):
        if 1 < b - a <= max_gap + 1:
            gap = np.arange(a + 1, b)
            s = (gap - a) / (b - a)
            translations[gap] = (1 - s)[:, None] * translations[a] + s[:, None] * translations[b]
            quaternions[gap] = Slerp([0, 1], Rotation.from_quat([quaternions[a], quaternions[b]]))(s).as_quat()
            weights[gap] = FILLED_WEIGHT
            available[gap] = True

    # Put consecutive quaternions in the same hemisphere so that averaging them is meaningful.
    previous = None
    for i in np.flatnonzero(available):
        if previous is not None and quaternions[i] @ quaternions[previous] < 0:
            quaternions[i] *= -1
        previous = i

    t_smooth = _local_linear(translations, weights, available, sigma_frames)
    q_smooth = _local_linear(quaternions, weights, available, sigma_frames)
    return [
        Pose.from_quaternion(q_smooth[i] / np.linalg.norm(q_smooth[i]), t_smooth[i]) if available[i] else None
        for i in range(n)
    ]
