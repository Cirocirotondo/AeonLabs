"""Per-clip stages: locate the paper's corners in every frame, then turn corners into camera poses.

Corner tracking works in pixels only, so it runs before the intrinsics are known; the focal
length is then estimated from the frames where the paper was measured (see focal.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .edges import find_initial_corners, fit_edges
from .geometry import Intrinsics, Pose, corners_from_edges, edges_from_corners, paper_corners_world
from .pose import solve_pose
from .segmentation import ColourThresholds, FrameMasks, classify
from .tracking import plane_homography

PAPER, TRACKED, FAILED = "paper", "tracked", "failed"

# A fully measured outline further than this from the tracked prediction is not trusted.
MAX_JUMP_PX = 20.0
WIDE_BAND_PX = 30.0


@dataclass
class FrameCorners:
    corners: np.ndarray | None  # (4, 2) image corners C0..C3
    source: str
    measured_edges: int


@dataclass
class FramePose:
    pose: Pose | None
    reprojection_px: float
    source: str


def _next_corners(reference: FrameMasks, reference_corners: np.ndarray, current: FrameMasks) -> FrameCorners:
    homography = plane_homography(reference, current)
    if homography is not None:
        predicted = cv2.perspectiveTransform(reference_corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
        fits = fit_edges(current, predicted)
    else:
        predicted = reference_corners
        fits = fit_edges(current, predicted)
        if not all(f.measured for f in fits):
            fits = fit_edges(current, predicted, band_px=WIDE_BAND_PX)

    measured = sum(f.measured for f in fits)
    if measured == 4:
        corners = corners_from_edges([f.line for f in fits])
        if homography is None or np.max(np.linalg.norm(corners - predicted, axis=1)) < MAX_JUMP_PX:
            return FrameCorners(corners, PAPER, 4)
        return FrameCorners(predicted, TRACKED, 0)
    if homography is None:
        return FrameCorners(None, FAILED, measured)
    # Keep the edges that were measured and take the rest from the tracked prediction.
    guessed = edges_from_corners(predicted)
    lines = [fit.line if fit.measured else guess for fit, guess in zip(fits, guessed)]
    return FrameCorners(corners_from_edges(lines), TRACKED, measured)


def track_paper_corners(frames: Sequence[np.ndarray], thresholds: ColourThresholds = ColourThresholds()) -> list[FrameCorners]:
    """Image corners of the reference paper in every frame, with the pose source of each.

    Starts from the first frame that shows the whole sheet, which also fixes the paper frame's
    orientation, then walks forward and backward from it.
    """
    masks = [classify(f, thresholds) for f in frames]
    results: list[FrameCorners] = [FrameCorners(None, FAILED, 0) for _ in frames]
    start, initial = next(((i, c) for i, m in enumerate(masks) if (c := find_initial_corners(m)) is not None),
                          (None, None))
    if start is None:
        return results
    results[start] = FrameCorners(initial, PAPER, 4)

    for order in (range(start + 1, len(frames)), range(start - 1, -1, -1)):
        reference = start
        for i in order:
            results[i] = _next_corners(masks[reference], results[reference].corners, masks[i])
            if results[i].corners is not None:
                reference = i
    return results


def solve_poses(corners: Sequence[FrameCorners], intrinsics: Intrinsics,
                paper: np.ndarray | None = None) -> list[FramePose]:
    paper = paper_corners_world() if paper is None else paper
    poses: list[FramePose] = []
    previous: Pose | None = None
    for frame in corners:
        solved = None if frame.corners is None else solve_pose(frame.corners, intrinsics, previous, paper)
        if solved is None:
            poses.append(FramePose(None, float("nan"), FAILED))
            continue
        pose, rms = solved
        poses.append(FramePose(pose, rms, frame.source))
        previous = pose
    return poses
